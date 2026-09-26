"""Exact tree DP with time-envelope / cost Pareto states.

State semantics (node ``v``)
----------------------------
A state is one *concrete* assignment of every edge in the subtree of ``v``.
Leaf arrival times are stored relative to the time of ``v``:

* ``[m, M]``  -- envelope of the subtree's leaf arrival times;
* ``r = M-m`` -- internal skew (invariant under the common shift a parent
  edge adds, so ``r > S`` is pruned immediately);
* ``[wlo, whi]`` -- set of *common prefixes* ``p`` (delay from root to ``v``)
  for which every leaf window of the subtree can still hold.  Leaf windows
  seed the interval at the leaf; crossing an edge of delay ``d`` shifts it
  to ``[wlo-d, whi-d]``; merging siblings intersects intervals.  ``None``
   denotes -inf / +inf;
* [plo, phi] -- same construction for the pulse-width constraint when a
  root clock is given: a leaf seeds the allowed interval of
  d = t_fall - t_rise as [min_high - high_width,
  period - min_low - high_width] and crossing an edge shifts it by
  -(fall_delay - delay) of the chosen option.  Without a clock both
  bounds stay None and the model degenerates to the single-edge one;
* ``cost`` -- summed option cost over subtree edges;
* ``vec``  -- option indices in the subtree's local DFS-preorder edge slots
  (its first entry is the edge to the first child, then that child's
  subtree, then the next child edge, ...).

Dominance (exact, not heuristic)
--------------------------------
State ``k`` dominates ``c`` iff for *every* continuation above ``v`` (any
common prefix and any sibling subtrees), feasibility of ``c`` implies
feasibility of ``k``:

    cost_k <= cost_c,
    [m_k, M_k] subseteq [m_c, M_c],
    W_c subseteq W_k.

Envelope containment makes the global range no larger when combined with
any sibling envelope (extrema are monotone); a wider prefix window set
never loses a feasible root prefix.  States are never unioned, so every
retained state stays an exact assignment and every reported number is an
exact integer.

Budget semantics
----------------
``budget`` caps the Pareto frontier size at every node.  Exceeding it
truncates (cheapest / tightest-envelope states kept) and the answer is
reported :py:member:`Status.BUDGET_EXHAUSTED`: a feasible incumbent may be
returned, but optimality is explicitly *not proven*; if no feasible state
survives anywhere after a cutoff, feasibility is not proven either.
:py:member:`Status.INFEASIBLE` is returned only when no cutoff happened and
the root has no feasible state -- proof of impossibility and budget
exhaustion are always kept apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from .model import (
    Clock,
    ClockTree,
    InputTypeError,
    InputValueError,
    _check_int,
)

DEFAULT_BUDGET = 4_000
MIN_BUDGET = 1
_INF_WIDTH = 1 << 62  # any real interval width in inputs is far smaller


class Status(Enum):
    OPTIMAL = "optimal"                    # exhaustive, optimum proven
    INFEASIBLE = "infeasible"              # exhaustively proven impossible
    BUDGET_EXHAUSTED = "budget_exhausted"  # frontier cap hit; not proven


@dataclass(frozen=True)
class PruningInfo:
    """Exact search counts and the recorded pruning bound.

    dominated_states: assignments removed by the exact Pareto rule (or by
                      de-duplicating an identical geometric signature).
    window_states:    sibling joins whose shifted-window intersection was
                      empty (some leaf window could not be met jointly).
    skew_states:      joins whose leaf envelope already spanned more than S.
    pulse_states:     joins whose shifted pulse-width intersection was
                      empty (some leaf's min high/low width could not be
                      met jointly); 0 when no clock was given.
    budget_states:    assignments dropped solely by the frontier cap.
    cutoff_min_cost:  smallest subtree cost among budget-dropped states -- a
                      cutoff *record*, explicitly NOT an optimality bound;
                      None when nothing was dropped.
    nodes_visited:    tree vertices processed by the DP.
    """

    dominated_states: int
    window_states: int
    skew_states: int
    pulse_states: int
    budget_states: int
    cutoff_min_cost: int | None
    nodes_visited: int

    @property
    def truncated(self) -> bool:
        return self.budget_states > 0


@dataclass(frozen=True)
class LeafTiming:
    """Both edge times and pulse widths at one leaf (exact integers).

    Unified absolute-time representation rooted at cycle 0: the root
    rising edge leaves at t=0 and the root falling edge at
    clock.high_width, so ``rise`` / ``fall`` are root-relative arrival
    times of the two edges.  ``fall`` may exceed ``period`` (the falling
    edge then lands in the next cycle); the solver guarantees
    ``0 <= fall - rise <= period`` for every reported leaf, i.e. the
    falling edge never crosses the next rising edge.  ``high`` is the
    high-pulse width ``fall - rise`` and ``low`` is ``period - high``.
    """

    rise: int
    fall: int
    high: int
    low: int


@dataclass(frozen=True)
class SolveResult:
    status: Status
    tree: ClockTree
    skew_limit: int
    total_cost: int | None            # None iff no feasible incumbent
    max_delay: int | None             # latest leaf arrival (root-relative)
    min_delay: int | None             # earliest leaf arrival
    skew: int | None                  # max_delay - min_delay
    choices: dict[str, int]           # edge id (child label) -> option index
    leaf_arrivals: dict[str, int]     # leaf -> summed root-to-leaf delay
    leaf_path_edges: dict[str, tuple[str, ...]]
    leaf_timing: dict[str, LeafTiming] | None  # None iff no clock given
    pruning: PruningInfo
    budget: int | None

    @property
    def proven_optimal(self) -> bool:
        return self.status is Status.OPTIMAL


@dataclass(slots=True)
class _State:
    m: int
    M: int
    wlo: int | None
    whi: int | None
    plo: int | None
    phi: int | None
    cost: int
    vec: tuple[int, ...]


@dataclass(slots=True)
class _Stats:
    dominated: int = 0
    window: int = 0
    skew: int = 0
    pulse: int = 0
    budget_drops: int = 0
    cutoff: int | None = None
    nodes: int = 0

    def record_cutoff(self, cost: int) -> None:
        self.budget_drops += 1
        if self.cutoff is None or cost < self.cutoff:
            self.cutoff = cost


def _shift(wlo, whi, d):
    return (None if wlo is None else wlo - d,
            None if whi is None else whi - d)


def _intersect(a, b):
    """Intersection of closed [lo, hi] intervals with None = infinity."""
    if a[0] is None:
        lo = b[0]
    elif b[0] is None:
        lo = a[0]
    else:
        lo = a[0] if a[0] > b[0] else b[0]
    if a[1] is None:
        hi = b[1]
    elif b[1] is None:
        hi = a[1]
    else:
        hi = a[1] if a[1] < b[1] else b[1]
    if lo is not None and hi is not None and lo > hi:
        return None
    return (lo, hi)


def _window_span(s: _State) -> int:
    # Smaller is better for the cutoff ordering; an infinite side means the
    # window set is wide, so it sorts after tight finite windows.
    if s.wlo is not None and s.whi is not None:
        return s.whi - s.wlo
    return _INF_WIDTH


def _pulse_span(s: _State) -> int:
    if s.plo is not None and s.phi is not None:
        return s.phi - s.plo
    return _INF_WIDTH


def _dominates(k: _State, c: _State) -> bool:
    """Exact dominance under objective (cost, root max-delay M, vec).

    Geometry: envelope containment + no-narrower prefix-window set makes k
    feasible whenever c is, with root max-delay no larger for ANY sibling
    continuation.  A strict cost win settles it.  On equal cost the root
    max-delay only ties or strictly improves, so the local vec decides on a
    tie: subtree slots form one contiguous block of the root preorder
    vector, hence local vec order equals root vec order for every identical
    continuation.  A subtree-strict M is NOT enough by itself, because a
    sibling leaf may supply the global maximum and make root M tie.
    """
    if k.cost > c.cost:
        return False
    if not (k.m >= c.m and k.M <= c.M):
        return False
    wlo_ok = (k.wlo is None) or (c.wlo is not None and k.wlo <= c.wlo)
    whi_ok = (k.whi is None) or (c.whi is not None and k.whi >= c.whi)
    if not (wlo_ok and whi_ok):
        return False
    plo_ok = (k.plo is None) or (c.plo is not None and k.plo <= c.plo)
    phi_ok = (k.phi is None) or (c.phi is not None and k.phi >= c.phi)
    if not (plo_ok and phi_ok):
        return False
    if k.cost < c.cost:
        return True
    return k.vec <= c.vec


def _pareto(states: list[_State], cap: int, st: _Stats) -> list[_State]:
    """Collapse equal signatures, apply exact dominance, optionally cap."""
    best: dict[tuple, _State] = {}
    for s in states:
        key = (s.m, s.M, s.wlo, s.whi, s.plo, s.phi)
        old = best.get(key)
        if old is None or (s.cost, s.vec) < (old.cost, old.vec):
            if old is not None:
                st.dominated += 1
            best[key] = s
        else:
            st.dominated += 1

    pts = list(best.values())
    pts.sort(key=lambda s: (
        s.cost, s.m, s.M,
        s.wlo if s.wlo is not None else -_INF_WIDTH,
        s.whi if s.whi is not None else _INF_WIDTH,
        s.plo if s.plo is not None else -_INF_WIDTH,
        s.phi if s.phi is not None else _INF_WIDTH,
        s.vec,
    ))
    kept: list[_State] = []
    for cand in pts:
        # cand can only be dominated by an equal-or-cheaper state; if it
        # survives, it may in turn dominate equal-cost states already kept.
        if any(_dominates(k, cand) for k in kept):
            st.dominated += 1
            continue
        kept = [k for k in kept if not _dominates(cand, k)]
        kept.append(cand)

    if cap >= 0 and len(kept) > cap:
        # The cutoff is a resource limit, not a proof: mirror the objective
        # order (cost, then max delay M, then option order), keeping tight
        # envelopes / flexible windows as secondary heuristics, and flag the
        # whole result unproven.
        kept.sort(key=lambda s: (s.cost, s.M, s.M - s.m,
                                 _window_span(s), _pulse_span(s), s.vec))
        for d in kept[cap:]:
            st.record_cutoff(d.cost)
        kept = kept[:cap]
    return kept


def validate_tree(tree: ClockTree) -> dict:
    """Re-run structural validation eagerly; return a small summary.

    Constructing :class:`ClockTree` already validates, so this exists as an
    explicit named entry point and as the place ``solve`` trusts.  Raises
    the package's input errors on any problem; never returns ``False``.
    """
    if not isinstance(tree, ClockTree):
        raise InputTypeError(
            f"tree must be a ClockTree, got {type(tree).__name__}"
        )
    return {
        "root": tree.root,
        "n_edges": tree.n_edges,
        "leaves": tuple(sorted(tree.leaf_set)),
    }


def solve(tree: ClockTree, skew_limit: int, *,
          clock: Clock | None = None,
          budget: int | None = DEFAULT_BUDGET) -> SolveResult:
    """Minimise total buffer cost under windows and the global skew bound.

    When a clock is given, every leaf must additionally keep a high
    pulse of at least window.min_high and a low pulse of at least
    window.min_low: with root-relative rise/fall sums t_r / t_f the
    leaf high width is clock.high_width + t_f - t_r and must lie in
    [min_high, clock.period - min_low] (the upper bound also forbids
    the falling edge crossing the next rising edge).  Pulse width is a
    hard feasibility constraint, never a soft objective; the objective
    order stays (total cost, max rise delay, option vector).

    See module docstring for state/dominance semantics and
    :class:`SolveResult` / :class:`Status` for the outcome contract.
    """
    validate_tree(tree)
    s_limit = _check_int(skew_limit, "skew_limit")
    if s_limit < 0:
        raise InputValueError(f"skew_limit must be >= 0, got {s_limit}")
    if clock is not None and not isinstance(clock, Clock):
        raise InputTypeError(
            f"clock must be a Clock or None, got {type(clock).__name__}"
        )
    if budget is None:
        cap = -1
    else:
        cap = _check_int(budget, "budget")
        if cap < MIN_BUDGET:
            raise InputValueError(
                f"budget must be >= {MIN_BUDGET} or None, got {cap}"
            )

    stats = _Stats()
    children = tree.children
    windows = {w.leaf: w for w in tree.windows}

    def node_dp(v: str) -> list[_State]:
        stats.nodes += 1
        kids = children.get(v)
        if not kids:
            w = windows[v]
            if clock is None:
                plo = phi = None
            else:
                # Pulse constraint as a prefix interval on d = t_f - t_r:
                # min_high <= high_width + d <= period - min_low.
                plo = w.min_high - clock.high_width
                phi = clock.period - clock.high_width - w.min_low
            return [_State(0, 0, w.lo, w.hi, plo, phi, 0, ())]

        frontier: list[_State] = []
        first = True
        for ch, edge in sorted(kids, key=lambda x: x[0]):
            child_states = node_dp(ch)
            produced: list[_State] = []
            if first:
                # No earlier siblings: envelope and prefix set come straight
                # from this child, shifted across the parent edge.
                for cs in child_states:
                    for opt_idx, opt in enumerate(edge.options):
                        d = edge.base_delay + opt.delay
                        wlo, whi = _shift(cs.wlo, cs.whi, d)
                        pd = opt.fall_delay - opt.delay
                        plo, phi = _shift(cs.plo, cs.phi, pd)
                        produced.append(_State(
                            cs.m + d, cs.M + d, wlo, whi, plo, phi,
                            cs.cost + opt.cost, (opt_idx,) + cs.vec,
                        ))
                first = False
            else:
                for fs in frontier:
                    for opt_idx, opt in enumerate(edge.options):
                        d = edge.base_delay + opt.delay
                        pd = opt.fall_delay - opt.delay
                        for cs in child_states:
                            wlo, whi = _shift(cs.wlo, cs.whi, d)
                            wint = _intersect(
                                (fs.wlo, fs.whi), (wlo, whi)
                            )
                            if wint is None:
                                stats.window += 1
                                continue
                            plo, phi = _shift(cs.plo, cs.phi, pd)
                            pint = _intersect(
                                (fs.plo, fs.phi), (plo, phi)
                            )
                            if pint is None:
                                stats.pulse += 1
                                continue
                            cm = cs.m + d
                            cM = cs.M + d
                            em = fs.m if fs.m < cm else cm
                            eM = fs.M if fs.M > cM else cM
                            if eM - em > s_limit:
                                stats.skew += 1
                                continue
                            produced.append(_State(
                                em, eM, wint[0], wint[1],
                                pint[0], pint[1],
                                fs.cost + opt.cost + cs.cost,
                                fs.vec + (opt_idx,) + cs.vec,
                            ))
            frontier = _pareto(produced, cap, stats)
        return frontier

    # Degenerate tree: a lone root, no edges or leaves.  Nothing to choose.
    if tree.n_edges == 0:
        root_states = [_State(0, 0, None, None, None, None, 0, ())]
    else:
        root_states = node_dp(tree.root)

    feasible = [
        s for s in root_states
        if (s.wlo is None or s.wlo <= 0) and (s.whi is None or 0 <= s.whi)
        and (s.plo is None or s.plo <= 0) and (s.phi is None or 0 <= s.phi)
    ]
    info = PruningInfo(
        stats.dominated, stats.window, stats.skew, stats.pulse,
        stats.budget_drops, stats.cutoff, stats.nodes,
    )

    def no_incumbent(reason: Status) -> SolveResult:
        return SolveResult(
            reason, tree, s_limit, None, None, None, None,
            {}, {}, {}, None, info, budget,
        )

    if not feasible:
        if info.truncated:
            # A cutoff may have discarded the only feasible assignments.
            return no_incumbent(Status.BUDGET_EXHAUSTED)
        return no_incumbent(Status.INFEASIBLE)

    # Contract objective: lexicographic (total cost, latest leaf delay M,
    # option-index vector in canonical edge order).
    best = min(feasible, key=lambda s: (s.cost, s.M, s.vec))
    status = Status.BUDGET_EXHAUSTED if info.truncated else Status.OPTIMAL

    choices, arrivals, paths, timing = _reconstruct(tree, best.vec, clock)
    return SolveResult(
        status, tree, s_limit,
        best.cost, best.M, best.m, best.M - best.m,
        choices, arrivals, paths, timing, info, budget,
    )


def _reconstruct(tree: ClockTree, vec: tuple[int, ...],
                 clock: Clock | None):
    """Turn the root preorder option vector into per-edge choices,
    exact root-to-leaf rise-delay sums, and (with a clock) per-leaf
    two-edge timing in the unified absolute representation."""
    choices: dict[str, int] = {}
    pos = 0

    def walk(v: str) -> None:
        nonlocal pos
        for ch, _e in sorted(tree.children.get(v, ()), key=lambda x: x[0]):
            choices[ch] = vec[pos]
            pos += 1
            walk(ch)

    walk(tree.root)

    arrivals: dict[str, int] = {}
    paths: dict[str, tuple[str, ...]] = {}
    timing: dict[str, LeafTiming] | None = {} if clock is not None else None
    for leaf in sorted(tree.leaf_set):
        chain: list[str] = []
        v = leaf
        total = 0
        fall_total = 0
        while v != tree.root:
            e = tree.parent_edge[v]
            oi = choices[v]
            total += e.base_delay + e.options[oi].delay
            fall_total += e.base_delay + e.options[oi].fall_delay
            chain.append(v)  # edge identified by its child endpoint
            v = e.parent
        chain.reverse()
        arrivals[leaf] = total
        paths[leaf] = tuple(chain)
        if timing is not None:
            fall = clock.high_width + fall_total
            high = fall - total
            timing[leaf] = LeafTiming(
                rise=total, fall=fall, high=high,
                low=clock.period - high,
            )
    return choices, arrivals, paths, timing

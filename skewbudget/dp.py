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
* ``cost`` -- summed option cost over subtree edges;
* ``vec``  -- option indices in the subtree's local DFS-preorder edge slots
  (its first entry is the edge to the first child, then that child's
  subtree, then the next child edge, ...).

Pulse-width semantics (only when a RootClock is supplied)
---------------------------------------------------------
Each option delays the rising and falling edges separately
(``rise_delay`` / ``fall_delay``).  Arrival windows, the skew bound and
the objective all act on the *rising* edge, exactly as before; the
falling edge only enters through per-leaf pulse widths.  For a leaf with
path shift sum ``D = sum(fall_delay - rise_delay)`` the high width is
``clock.high_width + D`` and the low width is ``period - high_width``, so
the leaf constraint is the closed interval
``min_high <= high_width + D <= period - min_low`` -- which also forbids
the fall edge from crossing the next rising edge.  Because every edge
above a node shifts all its leaves' ``D`` by the same amount, the
constraint is carried exactly like the arrival windows: a per-state
feasible prefix interval ``[dlo, dhi]`` seeded at the leaves, shifted by
``-(fall_delay - rise_delay)`` across each edge and intersected at
sibling merges.  All times are reported in one uniform representation:
absolute from the root rising edge at t=0, with phases given modulo the
period, so period crossings need no special case.

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
    ClockTree,
    InputTypeError,
    InputValueError,
    RootClock,
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
    budget_states:    assignments dropped solely by the frontier cap.
    pulse_states:     joins/leaf seeds dropped because the high/low pulse
                      width interval went empty (fall edge vanished or
                      crossed the next rising edge).
    cutoff_min_cost:  smallest subtree cost among budget-dropped states -- a
                      cutoff *record*, explicitly NOT an optimality bound;
                      None when nothing was dropped.
    nodes_visited:    tree vertices processed by the DP.
    """

    dominated_states: int
    window_states: int
    skew_states: int
    budget_states: int
    cutoff_min_cost: int | None
    nodes_visited: int
    pulse_states: int = 0

    @property
    def truncated(self) -> bool:
        return self.budget_states > 0


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
    pruning: PruningInfo
    budget: int | None
    clock: RootClock | None = None
    leaf_edges: dict[str, "LeafTiming"] | None = None

    @property
    def proven_optimal(self) -> bool:
        return self.status is Status.OPTIMAL


@dataclass(frozen=True)
class LeafTiming:
    """Both edge times and pulse widths at one leaf (exact integers).

    Uniform representation: every time is absolute, measured from the
    root rising edge at t=0 of the reference cycle; ``*_phase`` values
    are the same times reduced modulo the clock period, so a fall edge
    that crossed a period boundary is still reported consistently.

    rise:        rising-edge arrival at the leaf (== leaf_arrivals value).
    fall:        falling-edge arrival of the same cycle at the leaf.
    high_width:  fall - rise (must be >= the leaf's min_high).
    low_width:   period - high_width (must be >= the leaf's min_low).
    rise_phase / fall_phase: rise % period / fall % period.
    """

    rise: int
    fall: int
    high_width: int
    low_width: int
    rise_phase: int
    fall_phase: int


@dataclass(slots=True)
class _State:
    m: int
    M: int
    wlo: int | None
    whi: int | None
    cost: int
    vec: tuple[int, ...]
    # Feasible interval for the prefix pulse-shift sum P (sum of
    # option.fall_delay - option.rise_delay over edges root->v) such that
    # every leaf pulse constraint in the subtree can still hold.  None
    # denotes -inf / +inf (no clock supplied -> unconstrained).
    dlo: int | None = None
    dhi: int | None = None


@dataclass(slots=True)
class _Stats:
    dominated: int = 0
    window: int = 0
    skew: int = 0
    budget_drops: int = 0
    cutoff: int | None = None
    nodes: int = 0
    pulse: int = 0

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


def _delta_span(s: _State) -> int:
    if s.dlo is not None and s.dhi is not None:
        return s.dhi - s.dlo
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
    dlo_ok = (k.dlo is None) or (c.dlo is not None and k.dlo <= c.dlo)
    dhi_ok = (k.dhi is None) or (c.dhi is not None and k.dhi >= c.dhi)
    if not (dlo_ok and dhi_ok):
        return False
    if k.cost < c.cost:
        return True
    return k.vec <= c.vec


def _pareto(states: list[_State], cap: int, st: _Stats) -> list[_State]:
    """Collapse equal signatures, apply exact dominance, optionally cap."""
    best: dict[tuple, _State] = {}
    for s in states:
        key = (s.m, s.M, s.wlo, s.whi, s.dlo, s.dhi)
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
        s.dlo if s.dlo is not None else -_INF_WIDTH,
        s.dhi if s.dhi is not None else _INF_WIDTH,
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
                                 _window_span(s), _delta_span(s), s.vec))
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
          clock: RootClock | None = None,
          budget: int | None = DEFAULT_BUDGET) -> SolveResult:
    """Minimise total buffer cost under windows and the global skew bound.

    When ``clock`` is given, every leaf must additionally satisfy its
    min_high / min_low pulse widths under that root clock, and a fall
    edge that would land past the next rising edge is rejected.
    See module docstring for state/dominance semantics and
    :class:`SolveResult` / :class:`Status` for the outcome contract.
    """
    validate_tree(tree)
    s_limit = _check_int(skew_limit, "skew_limit")
    if s_limit < 0:
        raise InputValueError(f"skew_limit must be >= 0, got {s_limit}")
    if clock is not None and not isinstance(clock, RootClock):
        raise InputTypeError(
            f"clock must be a RootClock or None, got {type(clock).__name__}"
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
                return [_State(0, 0, w.lo, w.hi, 0, ())]
            # Leaf pulse constraint on the prefix shift sum P:
            #   min_high <= high_width + P <= period - min_low
            dlo = w.min_high - clock.high_width
            dhi = clock.period - w.min_low - clock.high_width
            if dlo > dhi:
                stats.pulse += 1
                return []
            return [_State(0, 0, w.lo, w.hi, 0, (), dlo, dhi)]

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
                        d = edge.base_delay + opt.rise_delay
                        delta = opt.fall_delay - opt.rise_delay
                        wlo, whi = _shift(cs.wlo, cs.whi, d)
                        dlo, dhi = _shift(cs.dlo, cs.dhi, delta)
                        produced.append(_State(
                            cs.m + d, cs.M + d, wlo, whi,
                            cs.cost + opt.cost, (opt_idx,) + cs.vec,
                            dlo, dhi,
                        ))
                first = False
            else:
                for fs in frontier:
                    for opt_idx, opt in enumerate(edge.options):
                        d = edge.base_delay + opt.rise_delay
                        delta = opt.fall_delay - opt.rise_delay
                        for cs in child_states:
                            wlo, whi = _shift(cs.wlo, cs.whi, d)
                            wint = _intersect(
                                (fs.wlo, fs.whi), (wlo, whi)
                            )
                            if wint is None:
                                stats.window += 1
                                continue
                            dlo, dhi = _shift(cs.dlo, cs.dhi, delta)
                            dint = _intersect((fs.dlo, fs.dhi), (dlo, dhi))
                            if dint is None:
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
                                fs.cost + opt.cost + cs.cost,
                                fs.vec + (opt_idx,) + cs.vec,
                                dint[0], dint[1],
                            ))
            frontier = _pareto(produced, cap, stats)
        return frontier

    # Degenerate tree: a lone root, no edges or leaves.  Nothing to choose.
    if tree.n_edges == 0:
        root_states = [_State(0, 0, None, None, 0, ())]
    else:
        root_states = node_dp(tree.root)

    feasible = []
    for s in root_states:
        if not ((s.wlo is None or s.wlo <= 0)
                and (s.whi is None or 0 <= s.whi)):
            stats.window += 1
            continue
        if not ((s.dlo is None or s.dlo <= 0)
                and (s.dhi is None or 0 <= s.dhi)):
            stats.pulse += 1
            continue
        feasible.append(s)
    info = PruningInfo(
        stats.dominated, stats.window, stats.skew,
        stats.budget_drops, stats.cutoff, stats.nodes,
        pulse_states=stats.pulse,
    )

    def no_incumbent(reason: Status) -> SolveResult:
        return SolveResult(
            reason, tree, s_limit, None, None, None, None,
            {}, {}, {}, info, budget, clock, {},
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

    choices, arrivals, paths, leaf_edges = _reconstruct(tree, best.vec, clock)
    return SolveResult(
        status, tree, s_limit,
        best.cost, best.M, best.m, best.M - best.m,
        choices, arrivals, paths, info, budget, clock, leaf_edges,
    )


def _reconstruct(tree: ClockTree, vec: tuple[int, ...],
                 clock: RootClock | None):
    """Turn the root preorder option vector into per-edge choices and
    exact root-to-leaf delay sums (plus both-edge leaf timing when a
    root clock is in force)."""
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
    leaf_edges: dict[str, LeafTiming] = {}
    for leaf in sorted(tree.leaf_set):
        chain: list[str] = []
        v = leaf
        total = 0
        fall_total = 0
        while v != tree.root:
            e = tree.parent_edge[v]
            oi = choices[v]
            total += e.base_delay + e.options[oi].rise_delay
            fall_total += e.base_delay + e.options[oi].fall_delay
            chain.append(v)  # edge identified by its child endpoint
            v = e.parent
        chain.reverse()
        arrivals[leaf] = total
        paths[leaf] = tuple(chain)
        if clock is not None:
            fall = clock.high_width + fall_total
            high = fall - total
            leaf_edges[leaf] = LeafTiming(
                rise=total,
                fall=fall,
                high_width=high,
                low_width=clock.period - high,
                rise_phase=total % clock.period,
                fall_phase=fall % clock.period,
            )
    return choices, arrivals, paths, leaf_edges

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
from .model import ClockTree, InputTypeError, InputValueError, _check_int

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
    # Reference-constrained solving (solve_with_reference) extras; empty /
    # None for plain solve().  Never part of the objective.
    modifications: tuple = ()          # tuple[Modification, ...]
    max_changes: int | None = None
    n_changes: int | None = None

    @property
    def proven_optimal(self) -> bool:
        return self.status is Status.OPTIMAL


@dataclass(frozen=True)
class Modification:
    """One edge whose chosen option differs from the reference plan.

    edge:             child-endpoint label identifying the edge.
    reference_index:  option index the reference plan assigned.
    chosen_index:     option index actually chosen.
    Frozen edges never appear here (they are not counted as changes).
    """

    edge: str
    reference_index: int
    chosen_index: int


@dataclass(slots=True)
class _State:
    m: int
    M: int
    wlo: int | None
    whi: int | None
    cost: int
    vec: tuple[int, ...]
    ch: int = 0   # changes vs the reference plan so far (constraint only)


@dataclass(slots=True)
class _Stats:
    dominated: int = 0
    window: int = 0
    skew: int = 0
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
    Change counts only constrain (<= K), never optimise: a state with no
    more changes is feasible wherever a more-changed one is, so k must
    also satisfy k.ch <= c.ch.
    """
    if k.cost > c.cost:
        return False
    if k.ch > c.ch:
        return False
    if not (k.m >= c.m and k.M <= c.M):
        return False
    wlo_ok = (k.wlo is None) or (c.wlo is not None and k.wlo <= c.wlo)
    whi_ok = (k.whi is None) or (c.whi is not None and k.whi >= c.whi)
    if not (wlo_ok and whi_ok):
        return False
    if k.cost < c.cost:
        return True
    return k.vec <= c.vec


def _pareto(states: list[_State], cap: int, st: _Stats) -> list[_State]:
    """Collapse equal signatures, apply exact dominance, optionally cap."""
    best: dict[tuple, _State] = {}
    for s in states:
        key = (s.m, s.M, s.wlo, s.whi, s.ch)
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
                                 _window_span(s), s.ch, s.vec))
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
          budget: int | None = DEFAULT_BUDGET) -> SolveResult:
    """Minimise total buffer cost under windows and the global skew bound.

    See module docstring for state/dominance semantics and
    :class:`SolveResult` / :class:`Status` for the outcome contract.
    """
    validate_tree(tree)
    s_limit = _check_int(skew_limit, "skew_limit")
    if s_limit < 0:
        raise InputValueError(f"skew_limit must be >= 0, got {s_limit}")
    cap = _budget_cap(budget)
    stats = _Stats()
    root_states = _run_dp(tree, s_limit, cap, stats, None, None, 0)
    return _finish(tree, s_limit, root_states, stats, budget, None, None, None)


def _budget_cap(budget: int | None) -> int:
    if budget is None:
        return -1
    cap = _check_int(budget, "budget")
    if cap < MIN_BUDGET:
        raise InputValueError(
            f"budget must be >= {MIN_BUDGET} or None, got {cap}"
        )
    return cap


def _check_option_table(table, what: str, tree: ClockTree,
                        require_full: bool) -> dict[str, int]:
    """Validate a {child label: option index} mapping against the tree."""
    from collections.abc import Mapping
    if not isinstance(table, Mapping):
        raise InputTypeError(
            f"{what} must be a mapping {{edge child label: option index}}, "
            f"got {type(table).__name__}"
        )
    out: dict[str, int] = {}
    for label, idx in table.items():
        if not isinstance(label, str) or label not in tree.edge_index:
            raise InputValueError(
                f"{what}: {label!r} is not an edge (child label) of the tree"
            )
        i = _check_int(idx, f"{what}[{label!r}]")
        n_opt = len(tree.parent_edge[label].options)
        if not 0 <= i < n_opt:
            raise InputValueError(
                f"{what}[{label!r}] = {i} out of range "
                f"[0, {n_opt}) for that edge's options"
            )
        out[label] = i
    if require_full:
        missing = sorted(set(tree.edge_index) - set(out))
        if missing:
            raise InputValueError(
                f"{what} must assign an option index to every edge; "
                f"missing: {missing}"
            )
    return out


def solve_with_reference(tree: ClockTree, skew_limit: int,
                         reference, max_changes: int, *,
                         frozen=None,
                         budget: int | None = DEFAULT_BUDGET) -> SolveResult:
    """Solve under a freeze table and a change budget vs a reference plan.

    reference:   {edge child label: option index} covering EVERY edge; the
                 plan itself may be infeasible, but every index must be a
                 legal option index of its edge.
    max_changes: at most this many *non-frozen* edges may choose an option
                 different from the reference (K).  K=0 merely validates
                 the reference plan (frozen edges still override it).
    frozen:      optional {edge child label: option index}; those edges
                 MUST use exactly that option.  A frozen edge is never
                 pruned into another option and never counts as a change.

    The objective is unchanged -- lexicographic (total cost, max leaf
    delay, option vector); the change count is a constraint only and is
    never mixed into the cost.  Status semantics match :func:`solve`:
    budget exhaustion is reported separately from proven infeasibility.
    """
    validate_tree(tree)
    s_limit = _check_int(skew_limit, "skew_limit")
    if s_limit < 0:
        raise InputValueError(f"skew_limit must be >= 0, got {s_limit}")
    kmax = _check_int(max_changes, "max_changes")
    if kmax < 0:
        raise InputValueError(f"max_changes must be >= 0, got {kmax}")
    ref = _check_option_table(reference, "reference", tree, require_full=True)
    frz = (_check_option_table(frozen, "frozen", tree, require_full=False)
           if frozen is not None else {})
    cap = _budget_cap(budget)
    stats = _Stats()
    root_states = _run_dp(tree, s_limit, cap, stats, frz, ref, kmax)
    return _finish(tree, s_limit, root_states, stats, budget, ref, frz, kmax)


def _run_dp(tree: ClockTree, s_limit: int, cap: int, stats: _Stats,
            frozen: dict[str, int] | None,
            ref: dict[str, int] | None, kmax: int) -> list[_State]:
    """Shared exact DP.  frozen/ref restrict per-edge option iteration and
    seed the change counter; both None gives the plain solve behaviour."""
    children = tree.children
    windows = {w.leaf: w for w in tree.windows}

    def edge_options(edge) -> tuple[tuple[int, int], ...]:
        # (option index, change increment) pairs allowed on this edge.
        if frozen is not None and edge.child in frozen:
            return ((frozen[edge.child], 0),)
        if ref is None:
            return tuple((i, 0) for i in range(len(edge.options)))
        ridx = ref[edge.child]
        return tuple((i, 0 if i == ridx else 1)
                     for i in range(len(edge.options)))

    def node_dp(v: str) -> list[_State]:
        stats.nodes += 1
        kids = children.get(v)
        if not kids:
            w = windows[v]
            return [_State(0, 0, w.lo, w.hi, 0, ())]

        frontier: list[_State] = []
        first = True
        for ch, edge in sorted(kids, key=lambda x: x[0]):
            child_states = node_dp(ch)
            produced: list[_State] = []
            allowed = edge_options(edge)
            if first:
                # No earlier siblings: envelope and prefix set come straight
                # from this child, shifted across the parent edge.
                for cs in child_states:
                    for opt_idx, dch in allowed:
                        nch = cs.ch + dch
                        if nch > kmax:
                            continue
                        opt = edge.options[opt_idx]
                        d = edge.base_delay + opt.delay
                        wlo, whi = _shift(cs.wlo, cs.whi, d)
                        produced.append(_State(
                            cs.m + d, cs.M + d, wlo, whi,
                            cs.cost + opt.cost, (opt_idx,) + cs.vec, nch,
                        ))
                first = False
            else:
                for fs in frontier:
                    for opt_idx, dch in allowed:
                        opt = edge.options[opt_idx]
                        d = edge.base_delay + opt.delay
                        for cs in child_states:
                            nch = fs.ch + dch + cs.ch
                            if nch > kmax:
                                continue
                            wlo, whi = _shift(cs.wlo, cs.whi, d)
                            wint = _intersect(
                                (fs.wlo, fs.whi), (wlo, whi)
                            )
                            if wint is None:
                                stats.window += 1
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
                                fs.vec + (opt_idx,) + cs.vec, nch,
                            ))
            frontier = _pareto(produced, cap, stats)
        return frontier

    # Degenerate tree: a lone root, no edges or leaves.  Nothing to choose.
    if tree.n_edges == 0:
        root_states = [_State(0, 0, None, None, 0, ())]
    else:
        root_states = node_dp(tree.root)
    return root_states


def _finish(tree: ClockTree, s_limit: int, root_states: list[_State],
            stats: _Stats, budget: int | None,
            ref: dict[str, int] | None,
            frz: dict[str, int] | None,
            kmax: int | None) -> SolveResult:
    feasible = [
        s for s in root_states
        if (s.wlo is None or s.wlo <= 0) and (s.whi is None or 0 <= s.whi)
    ]
    info = PruningInfo(
        stats.dominated, stats.window, stats.skew,
        stats.budget_drops, stats.cutoff, stats.nodes,
    )

    def no_incumbent(reason: Status) -> SolveResult:
        return SolveResult(
            reason, tree, s_limit, None, None, None, None,
            {}, {}, {}, info, budget, (), kmax, None,
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

    choices, arrivals, paths = _reconstruct(tree, best.vec)
    modifications: tuple = ()
    n_changes = None
    if ref is not None:
        mods = []
        for child, _idx in sorted(tree.edge_index.items(),
                                  key=lambda kv: kv[1]):
            if frz and child in frz:
                continue
            if choices[child] != ref[child]:
                mods.append(Modification(child, ref[child], choices[child]))
        modifications = tuple(mods)
        n_changes = len(mods)
    return SolveResult(
        status, tree, s_limit,
        best.cost, best.M, best.m, best.M - best.m,
        choices, arrivals, paths, info, budget,
        modifications, kmax, n_changes,
    )


def _reconstruct(tree: ClockTree, vec: tuple[int, ...]):
    """Turn the root preorder option vector into per-edge choices and
    exact root-to-leaf delay sums."""
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
    for leaf in sorted(tree.leaf_set):
        chain: list[str] = []
        v = leaf
        total = 0
        while v != tree.root:
            e = tree.parent_edge[v]
            oi = choices[v]
            total += e.base_delay + e.options[oi].delay
            chain.append(v)  # edge identified by its child endpoint
            v = e.parent
        chain.reverse()
        arrivals[leaf] = total
        paths[leaf] = tuple(chain)
    return choices, arrivals, paths

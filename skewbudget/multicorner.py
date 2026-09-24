"""Exhaustive multi-corner solver with one shared physical option per edge.

Unlike the single-corner tree DP (:mod:`skewbudget.dp`), the physical
option selected on an edge has a *different* realised delay in every
operating corner.  The same option vector must therefore satisfy every
corner simultaneously, so per-corner Pareto frontiers cannot be solved
independently.

The problem statement bounds the search explicitly: at most
``MAX_CHOICE_POINTS`` (=8) edges carry more than one option and each such
edge has at most ``MAX_MC_OPTIONS`` (=3) options, hence at most
``3**8 = 6561`` distinct assignments.  Every assignment is enumerated
exactly once (single-option edges are forced), so the reported minimum
physical cost is *proven* optimal; there is no truncation state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product

from .model import InputTypeError, MultiCornerTree


class MultiCornerStatus(Enum):
    OPTIMAL = "optimal"        # exhaustive enumeration, optimum proven
    INFEASIBLE = "infeasible"  # exhaustive enumeration, no assignment works


@dataclass(frozen=True)
class CornerReport:
    """Per-corner verification values of one physical assignment."""

    name: str
    skew_limit: int
    feasible: bool
    leaf_arrivals: dict[str, int]
    min_delay: int | None
    max_delay: int | None
    skew: int | None
    # leaf -> True iff arrival lies inside that leaf's closed window.
    window_ok: dict[str, bool]

    @property
    def skew_ok(self) -> bool:
        return self.skew is not None and self.skew <= self.skew_limit


@dataclass(frozen=True)
class MultiCornerResult:
    status: MultiCornerStatus
    tree: MultiCornerTree
    total_cost: int | None                  # None iff infeasible
    choices: dict[str, int]                 # edge id (child label) -> opt idx
    corners: tuple[CornerReport, ...]       # in input corner order
    assignments_enumerated: int
    feasible_assignments: int
    # Populated only when solve_multicorner(..., return_all=True); every
    # option vector attaining the minimum physical cost (winner first).
    all_optimal_choices: tuple[dict[str, int], ...]

    @property
    def proven_optimal(self) -> bool:
        return self.status is MultiCornerStatus.OPTIMAL

    def corner(self, name: str) -> CornerReport:
        for rpt in self.corners:
            if rpt.name == name:
                return rpt
        raise KeyError(name)


def validate_multicorner_tree(tree: MultiCornerTree) -> dict:
    """Eager validation entry point; construction already validates.

    Raises the package's input errors on any problem; never returns
    ``False``.
    """
    if not isinstance(tree, MultiCornerTree):
        raise InputTypeError(
            f"tree must be a MultiCornerTree, got {type(tree).__name__}"
        )
    return {
        "root": tree.root,
        "n_edges": tree.n_edges,
        "n_corners": tree.n_corners,
        "leaves": tuple(sorted(tree.leaf_set)),
        "choice_points": tree.choice_points,
        "corners": tuple(c.name for c in tree.corners),
    }


def _evaluate(tree: MultiCornerTree, picks: dict[str, int]):
    """Score one full assignment (every edge has a chosen option index).

    Returns ``(cost, reports, feasible)`` with a DFS per corner that
    accumulates root-relative arrival at every vertex once.
    """
    edges = tree.edges
    corners = tree.corners
    children = tree.children
    # Pre-fetch per-edge, per-corner total latency for the picked option so
    # the hot loops touch no attribute chains.
    edge_delay: dict[str, list[int]] = {}
    cost = 0
    for e in edges:
        oi = picks[e.child]
        cost += e.options[oi].cost
        edge_delay[e.child] = [
            e.base_delays[k] + e.options[oi].delays[k]
            for k in range(len(corners))
        ]

    reports: list[CornerReport] = []
    all_ok = True
    for k, corner in enumerate(corners):
        arrivals: dict[str, int] = {}
        # Iterative DFS from the root; a tree has exactly one path to each
        # vertex, so the accumulated value is the root-to-vertex sum.
        stack: list[tuple[str, int]] = [(tree.root, 0)]
        while stack:
            v, at = stack.pop()
            for ch, _e in sorted(children.get(v, ()), key=lambda x: x[0]):
                at_ch = at + edge_delay[ch][k]
                if ch in tree.leaf_set:
                    arrivals[ch] = at_ch
                stack.append((ch, at_ch))
        if arrivals:
            lo = min(arrivals.values())
            hi = max(arrivals.values())
            skew = hi - lo
        else:  # degenerate lone-root tree: nothing arrives anywhere
            lo = hi = skew = 0
        window_ok = {
            w.leaf: w.lo <= arrivals[w.leaf] <= w.hi
            for w in corner.windows
        }
        feasible = skew <= corner.skew_limit and all(window_ok.values())
        all_ok = all_ok and feasible
        reports.append(CornerReport(
            corner.name, corner.skew_limit, feasible,
            dict(arrivals), lo, hi, skew, window_ok,
        ))
    return cost, tuple(reports), all_ok


def solve_multicorner(
    tree: MultiCornerTree,
    *,
    return_all: bool = False,
) -> MultiCornerResult:
    """Enumerate every physical assignment and return the cheapest one that
    satisfies *all* corners simultaneously.

    Tie-break on equal physical cost: the lexicographically smallest option
    index vector in the canonical DFS-preorder edge order (forced
    single-option edges contribute their fixed 0), matching the
    single-corner convention.  With at most 8 choice points and 3 options
    each, at most 6561 assignments are examined -- the result is always
    either proven optimal or, only when the complete enumeration found
    nothing, proven infeasible.
    """
    validate_multicorner_tree(tree)
    if not isinstance(return_all, bool):
        raise InputTypeError(
            f"return_all must be a bool, got {type(return_all).__name__}"
        )

    choice_order = tree.choice_edge_order
    ranges = [range(len(tree.parent_edge[ch].options)) for ch in choice_order]
    n_enum = 1
    for r in ranges:
        n_enum *= len(r)

    # Full canonical vector (all edges); forced edges always pick index 0.
    canonical_order = [
        ch for ch, _idx in sorted(tree.edge_index.items(), key=lambda kv: kv[1])
    ]
    base_picks = {ch: 0 for ch in canonical_order}

    best_key = None          # (cost, canonical vector)
    best_cost = None
    best_reports = None
    best_picks = {}
    n_feasible = 0
    all_optima: list[dict[str, int]] = []

    for combo in product(*ranges):
        picks = dict(base_picks)
        picks.update(zip(choice_order, combo))
        cost, reports, feasible = _evaluate(tree, picks)
        if not feasible:
            continue
        n_feasible += 1
        vec = tuple(picks[ch] for ch in canonical_order)
        key = (cost, vec)
        if best_key is None or key < best_key:
            best_key = key
            best_cost = cost
            best_reports = reports
            best_picks = picks
            if return_all:
                all_optima = [dict(picks)]
        elif return_all and cost == best_key[0]:
            all_optima.append(dict(picks))

    if best_key is None:
        return MultiCornerResult(
            MultiCornerStatus.INFEASIBLE, tree, None, {}, (),
            n_enum, 0, (),
        )
    return MultiCornerResult(
        MultiCornerStatus.OPTIMAL, tree, best_cost, best_picks,
        best_reports, n_enum, n_feasible,
        tuple(all_optima) if return_all else (),
    )

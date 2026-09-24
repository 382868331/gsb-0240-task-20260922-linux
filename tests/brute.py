"""Independent brute-force reference.

Enumerates EVERY edge-option combination with itertools.product and scores
each assignment directly from the input model.  It never imports or calls
the DP core, so equality with it is a genuine cross-check, not a
round-trip.  Intended for tiny trees only (tests keep <= ~6 edges).
"""

from __future__ import annotations

from itertools import product

from skewbudget.model import ClockTree


def enumerate_solutions(tree: ClockTree, skew_limit: int):
    """Return all feasible assignments as dicts child_label -> option idx."""
    edges_by_child = {ch: e for e in tree.edges for ch in (e.child,)}
    children_sorted = sorted(edges_by_child)
    choices_ranges = [range(len(edges_by_child[ch].options))
                      for ch in children_sorted]
    out = []
    for combo in product(*choices_ranges):
        choice = dict(zip(children_sorted, combo))
        arrivals = {}
        for leaf in tree.leaf_set:
            total = 0
            v = leaf
            while v != tree.root:
                e = edges_by_child[v]
                oi = choice[v]
                total += e.base_delay + e.options[oi].delay
                v = e.parent
            arrivals[leaf] = total
        cost = sum(edges_by_child[ch].options[oi].cost
                   for ch, oi in choice.items())
        if not arrivals:
            lo = hi = 0
        else:
            lo = min(arrivals.values())
            hi = max(arrivals.values())
        if hi - lo > skew_limit:
            continue
        ok = True
        for w in tree.windows:
            if not (w.lo <= arrivals[w.leaf] <= w.hi):
                ok = False
                break
        if ok:
            out.append((choice, arrivals, cost, hi, lo))
    return out


def optimal_reference(tree: ClockTree, skew_limit: int):
    """Canonical optimum: min (cost, max delay, option vector)."""
    feasible = enumerate_solutions(tree, skew_limit)
    if not feasible:
        return None
    # Canonical vector follows the tree's public DFS-preorder edge slots.
    order = [ch for ch, _idx in
             sorted(tree.edge_index.items(), key=lambda kv: kv[1])]

    def key(item):
        choice, _arrivals, cost, hi, _lo = item
        return (cost, hi, tuple(choice[ch] for ch in order))

    return min(feasible, key=key)


# ---------------------------------------------------------------------------
# Independent multi-corner enumerator.  Like the single-corner one, it only
# reads the input model and itertools.product; it never imports the solver.
# ---------------------------------------------------------------------------


def enumerate_multicorner(tree):
    """Enumerate EVERY physical assignment of a MultiCornerTree.

    Returns ``(entries, n_total)`` where each entry is
    ``(picks, cost, per_corner)`` for a *feasible* assignment and
    ``per_corner`` is a tuple aligned with ``tree.corners`` of
    ``(arrivals, lo, hi, skew, window_ok)``.  ``n_total`` counts every
    assignment, feasible or not.
    """
    edges_by_child = {e.child: e for e in tree.edges}
    # Canonical DFS-preorder edge order.
    order = [ch for ch, _idx in
             sorted(tree.edge_index.items(), key=lambda kv: kv[1])]
    ranges = [range(len(edges_by_child[ch].options)) for ch in order]
    entries = []
    n_total = 0
    for combo in product(*ranges):
        n_total += 1
        picks = dict(zip(order, combo))
        cost = sum(edges_by_child[ch].options[oi].cost
                   for ch, oi in picks.items())
        per_corner = []
        feasible = True
        for corner in tree.corners:
            arrivals = {}
            for leaf in tree.leaf_set:
                total = 0
                v = leaf
                k = tree.corners.index(corner)
                while v != tree.root:
                    e = edges_by_child[v]
                    oi = picks[v]
                    total += e.base_delays[k] + e.options[oi].delays[k]
                    v = e.parent
                arrivals[leaf] = total
            if arrivals:
                lo, hi = min(arrivals.values()), max(arrivals.values())
                skew = hi - lo
            else:
                lo = hi = skew = 0
            window_ok = {w.leaf: w.lo <= arrivals[w.leaf] <= w.hi
                         for w in corner.windows}
            if skew > corner.skew_limit or not all(window_ok.values()):
                feasible = False
            per_corner.append((arrivals, lo, hi, skew, window_ok))
        if feasible:
            entries.append((picks, cost, tuple(per_corner)))
    return entries, n_total


def optimal_multicorner_reference(tree):
    """Canonical multi-corner optimum: min (cost, option vector).

    Returns ``(picks, cost, per_corner)`` or ``None`` when nothing works
    for all corners simultaneously.  Tie-break is the option-index vector
    in the tree's canonical DFS-preorder edge order only (no per-corner
    delay key).
    """
    entries, _n = enumerate_multicorner(tree)
    if not entries:
        return None
    order = [ch for ch, _idx in
             sorted(tree.edge_index.items(), key=lambda kv: kv[1])]

    def key(item):
        picks, cost, _per_corner = item
        return (cost, tuple(picks[ch] for ch in order))

    return min(entries, key=key)

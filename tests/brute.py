"""Independent brute-force reference.

Enumerates EVERY edge-option combination with itertools.product and scores
each assignment directly from the input model.  It never imports or calls
the DP core, so equality with it is a genuine cross-check, not a
round-trip.  Intended for tiny trees only (tests keep <= ~6 edges).
"""

from __future__ import annotations

from itertools import product

from skewbudget.model import ClockTree


def enumerate_solutions(tree: ClockTree, skew_limit: int,
                        reference=None, max_changes=None, frozen=None):
    """Return all feasible assignments as dicts child_label -> option idx.

    With ``reference``/``max_changes``/``frozen`` only assignments that
    honour the freeze table and differ from the reference on at most
    ``max_changes`` non-frozen edges are kept.
    """
    edges_by_child = {ch: e for e in tree.edges for ch in (e.child,)}
    children_sorted = sorted(edges_by_child)
    frozen = frozen or {}

    def allowed(ch):
        if ch in frozen:
            return (frozen[ch],)
        return range(len(edges_by_child[ch].options))

    choices_ranges = [allowed(ch) for ch in children_sorted]
    out = []
    for combo in product(*choices_ranges):
        choice = dict(zip(children_sorted, combo))
        if reference is not None and max_changes is not None:
            n_diff = sum(
                1 for ch in children_sorted
                if ch not in frozen and choice[ch] != reference[ch]
            )
            if n_diff > max_changes:
                continue
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


def optimal_reference(tree: ClockTree, skew_limit: int,
                      reference=None, max_changes=None, frozen=None):
    """Canonical optimum: min (cost, max delay, option vector)."""
    feasible = enumerate_solutions(tree, skew_limit,
                                   reference, max_changes, frozen)
    if not feasible:
        return None
    # Canonical vector follows the tree's public DFS-preorder edge slots.
    order = [ch for ch, _idx in
             sorted(tree.edge_index.items(), key=lambda kv: kv[1])]

    def key(item):
        choice, _arrivals, cost, hi, _lo = item
        return (cost, hi, tuple(choice[ch] for ch in order))

    return min(feasible, key=key)

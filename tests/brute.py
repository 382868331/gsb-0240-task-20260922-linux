"""Independent brute-force reference.

Enumerates EVERY edge-option combination with itertools.product and scores
each assignment directly from the input model.  It never imports or calls
the DP core, so equality with it is a genuine cross-check, not a
round-trip.  Intended for tiny trees only (tests keep <= ~6 edges).
"""

from __future__ import annotations

from itertools import product

from skewbudget.model import ClockTree


def enumerate_solutions(tree: ClockTree, skew_limit: int, clock=None):
    """Return all feasible assignments as dicts child_label -> option idx.

    When ``clock`` (a RootClock) is given, each leaf must additionally
    meet its min_high / min_low pulse widths, and a fall edge past the
    next rising edge is rejected -- scored directly from the model,
    never via the DP core.
    """
    edges_by_child = {ch: e for e in tree.edges for ch in (e.child,)}
    children_sorted = sorted(edges_by_child)
    choices_ranges = [range(len(edges_by_child[ch].options))
                      for ch in children_sorted]
    out = []
    for combo in product(*choices_ranges):
        choice = dict(zip(children_sorted, combo))
        arrivals = {}
        falls = {}
        for leaf in tree.leaf_set:
            total = 0
            fall_total = 0
            v = leaf
            while v != tree.root:
                e = edges_by_child[v]
                oi = choice[v]
                total += e.base_delay + e.options[oi].rise_delay
                fall_total += e.base_delay + e.options[oi].fall_delay
                v = e.parent
            arrivals[leaf] = total
            falls[leaf] = fall_total
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
        if ok and clock is not None:
            for w in tree.windows:
                high = clock.high_width + falls[w.leaf] - arrivals[w.leaf]
                low = clock.period - high
                # high < min_high: pulse vanished; low < min_low (in
                # particular low < 0): fall edge crossed the next rise.
                if high < w.min_high or low < w.min_low:
                    ok = False
                    break
        if ok:
            out.append((choice, arrivals, cost, hi, lo))
    return out


def optimal_reference(tree: ClockTree, skew_limit: int, clock=None):
    """Canonical optimum: min (cost, max delay, option vector)."""
    feasible = enumerate_solutions(tree, skew_limit, clock)
    if not feasible:
        return None
    # Canonical vector follows the tree's public DFS-preorder edge slots.
    order = [ch for ch, _idx in
             sorted(tree.edge_index.items(), key=lambda kv: kv[1])]

    def key(item):
        choice, _arrivals, cost, hi, _lo = item
        return (cost, hi, tuple(choice[ch] for ch in order))

    return min(feasible, key=key)

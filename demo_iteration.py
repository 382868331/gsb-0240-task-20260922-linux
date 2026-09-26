#!/usr/bin/env python3
"""Iteration demo: two-edge buffer delays and per-leaf pulse widths.

Run:  python demo_iteration.py

Everything is computed live from the fixed inputs below (no sleeps, no
pre-recorded output).  It prints:
  1. a NORMAL result under a root clock: per-leaf rising/falling edge
     times, high/low pulse widths, and the skew/cost trade-off;
  2. BOUNDARY results that actually trigger: a rising-feasible option
     whose high pulse vanishes, and a fall edge crossing the next rise;
  3. the degenerate case: rise_delay == fall_delay everywhere reproduces
     the original single-edge answer exactly.
"""

from __future__ import annotations

import time

from skewbudget import (
    ClockTree,
    Edge,
    LeafWindow,
    Option,
    RootClock,
    Status,
    solve,
)

CLK = RootClock(period=10, high_width=4)


def hr(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def show(label, res, tree) -> None:
    print(f"[{label}]")
    print(f"  status       : {res.status.value}")
    print(f"  total cost   : {res.total_cost}")
    if res.total_cost is not None:
        print(f"  option picks : {res.choices}")
        print(f"  rise arrivals: {res.leaf_arrivals}"
              f"  (skew {res.skew} <= S={res.skew_limit})")
        for leaf in sorted(res.leaf_edges):
            lt = res.leaf_edges[leaf]
            w = tree.window_of(leaf)
            print(f"  leaf {leaf!r}: rise={lt.rise} (phase {lt.rise_phase})"
                  f" fall={lt.fall} (phase {lt.fall_phase})"
                  f" high={lt.high_width} (>={w.min_high})"
                  f" low={lt.low_width} (>={w.min_low})")
    p = res.pruning
    print(f"  Pareto prune : dominated={p.dominated_states}"
          f" window={p.window_states} skew={p.skew_states}"
          f" pulse={p.pulse_states}")


def build_tree() -> ClockTree:
    #        r  (clock: period 10, high 4)
    #        | base 5 ; trunk (0,0,r0,f0) | (2,4,r2,f3)
    #        u
    #       / \
    #   a b1    b b4   branch (0,0,r0,f0) | (3,2,r3,f2)
    eq = [Option(0, 0, rise_delay=0, fall_delay=0),
          Option(3, 2, rise_delay=3, fall_delay=2)]
    return ClockTree(
        "r",
        [
            Edge("r", "u", 5, [Option(0, 0, rise_delay=0, fall_delay=0),
                               Option(2, 4, rise_delay=2, fall_delay=3)]),
            Edge("u", "a", 1, eq),
            Edge("u", "b", 4, eq),
        ],
        [LeafWindow("a", 0, 12, min_high=2, min_low=2),
         LeafWindow("b", 0, 12, min_high=2, min_low=2)],
    )


def main() -> None:
    t0 = time.perf_counter()

    hr("1. NORMAL RESULT -- two-edge delays, per-leaf pulse widths")
    tree = build_tree()
    print(f"root clock: period={CLK.period}, high_width={CLK.high_width};"
          " every leaf needs high>=2 and low>=2")
    res = solve(tree, skew_limit=0, clock=CLK)
    show("S=0 with clock", res, tree)
    assert res.status is Status.OPTIMAL
    for leaf, lt in res.leaf_edges.items():
        w = tree.window_of(leaf)
        assert lt.high_width >= w.min_high and lt.low_width >= w.min_low
        assert lt.fall <= lt.rise + CLK.period  # fall before next rise

    hr("2. BOUNDARY -- rising edge feasible but the high pulse vanishes")
    bad = ClockTree(
        "r",
        [Edge("r", "a", 5, [Option(4, 0, rise_delay=4, fall_delay=1)])],
        [LeafWindow("a", 8, 10, min_high=3)],
    )
    plain = solve(bad, skew_limit=0)
    print(f"  rising-edge-only solve : {plain.status.value},"
          f" arrival {plain.leaf_arrivals} (window [8,10] satisfied)")
    clocked = solve(bad, skew_limit=0, clock=CLK)
    print(f"  same tree with clock   : {clocked.status.value}"
          f" (high pulse would be {CLK.high_width + 1 - 4} < min_high 3,"
          f" pulse-pruned states={clocked.pruning.pulse_states})")
    assert plain.status is Status.OPTIMAL
    assert clocked.status is Status.INFEASIBLE

    hr("3. BOUNDARY -- fall edge crossing the next rising edge")
    cross = ClockTree(
        "r",
        [Edge("r", "a", 0, [Option(0, 0, rise_delay=0, fall_delay=8)])],
        [LeafWindow("a", 0, 10)],
    )
    rc = solve(cross, skew_limit=0, clock=CLK)
    print(f"  fall_delay-rise_delay=8 > period-high_width=6:"
          f" {rc.status.value} (fall would land past the next rise)")
    assert rc.status is Status.INFEASIBLE
    edge_ok = ClockTree(
        "r",
        [Edge("r", "a", 0, [Option(0, 0, rise_delay=0, fall_delay=6)])],
        [LeafWindow("a", 0, 10)],
    )
    rok = solve(edge_ok, skew_limit=0, clock=CLK)
    lt = rok.leaf_edges["a"]
    print(f"  fall_delay-rise_delay=6 (fall == next rise, allowed):"
          f" {rok.status.value}, high={lt.high_width}, low={lt.low_width}")
    assert rok.status is Status.OPTIMAL

    hr("4. DEGENERATE -- equal rise/fall delays reproduce the old problem")
    old_tree = ClockTree(
        "r",
        [Edge("r", "u", 5, [Option(0, 0), Option(2, 4)]),
         Edge("u", "a", 1, [Option(0, 0), Option(3, 2)]),
         Edge("u", "b", 4, [Option(0, 0), Option(3, 2)])],
        [LeafWindow("a", 0, 12), LeafWindow("b", 0, 12)],
    )
    for S in (3, 0):
        a = solve(old_tree, skew_limit=S)
        b = solve(old_tree, skew_limit=S, clock=CLK)
        same = (a.total_cost, a.max_delay, a.choices) == \
            (b.total_cost, b.max_delay, b.choices)
        print(f"  S={S}: without clock cost={a.total_cost}"
              f" picks={a.choices} | with clock cost={b.total_cost}"
              f" picks={b.choices} -> identical: {same}")
        assert same

    print()
    print(f"demo_iteration finished in {time.perf_counter() - t0:.3f} s"
          " (all numbers computed now from the fixed inputs above).")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Iteration demo: two-edge delays, root clock and per-leaf pulse widths.

Run:  python demo_iteration.py

Everything is computed live from the fixed inputs below (no sleeps, no
pre-recorded output, well under 90 s).  It prints:
  1. a NORMAL result with a root clock: per-leaf rise/fall edge times and
     high/low pulse widths in the unified absolute-time representation;
  2. a BOUNDARY result that actually triggers: rise timing is feasible
     but the high pulse vanishes, so the cheap option is rejected and a
     costlier one is forced -- plus a case where the falling edge would
     cross the next rising edge and the instance is proven infeasible;
  3. the degeneration check: equal rise/fall delays reproduce the old
     single-edge answer exactly.
"""

from __future__ import annotations

import time

from skewbudget import (
    Clock,
    ClockTree,
    Edge,
    LeafWindow,
    Option,
    Status,
    solve,
)


def hr(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def show(label, res, clock) -> None:
    print(f"[{label}]")
    print(f"  status       : {res.status.value}")
    print(f"  total cost   : {res.total_cost}")
    if res.total_cost is not None:
        print(f"  option picks : {res.choices}")
        print(f"  rise arrivals: {res.leaf_arrivals}  (skew {res.skew})")
        for leaf in sorted(res.leaf_timing):
            t = res.leaf_timing[leaf]
            note = "  (fall lands in next cycle)" if t.fall > clock.period else ""
            print(f"  leaf {leaf!r:6}: rise={t.rise:3d} fall={t.fall:3d}"
                  f"  high={t.high} low={t.low}{note}")
    p = res.pruning
    print(f"  pruned       : window={p.window_states} pulse={p.pulse_states}"
          f" skew={p.skew_states} dominated={p.dominated_states}")


def main() -> None:
    t0 = time.perf_counter()
    clock = Clock(period=20, high_width=8)

    hr("1. NORMAL -- clock P=20, high=8; both edges buffered per option")
    #        r
    #        | base 5, trunk (rise 0/fall 0, cost 0) | (rise 2/fall 1, cost 4)
    #        u
    #       / \
    #   a b1   b b4   branch (0/0,c0) | (rise 3/fall 2, cost 2)
    tree = ClockTree(
        "r",
        [
            Edge("r", "u", 5, [Option(0, 0, fall_delay=0),
                               Option(2, 4, fall_delay=1)]),
            Edge("u", "a", 1, [Option(0, 0, fall_delay=0),
                               Option(3, 2, fall_delay=2)]),
            Edge("u", "b", 4, [Option(0, 0, fall_delay=0),
                               Option(3, 2, fall_delay=2)]),
        ],
        [
            LeafWindow("a", 0, 14, min_high=4, min_low=4),
            LeafWindow("b", 0, 14, min_high=4, min_low=4),
        ],
    )
    print("tree: r-(5)->u ; u-(1)->a, u-(4)->b ; per-option (rise/fall)")
    print("      delays differ; leaves need high>=4 and low>=4 of P=20")
    res = solve(tree, skew_limit=0, clock=clock)
    show("S=0, min_high=min_low=4", res, clock)
    assert res.status is Status.OPTIMAL

    hr("2. BOUNDARY -- rise timing fine, but the high pulse vanishes")
    edge_case = ClockTree(
        "r",
        [Edge("r", "a", 0, [Option(6, 0, fall_delay=0),   # high = 8-6 = 2 < 4
                            Option(6, 3, fall_delay=5)])  # high = 8-1 = 7 ok
        ],
        [LeafWindow("a", 6, 6, min_high=4)],
    )
    cheap = solve(edge_case, 0)
    print(f"  without a clock the cheap option wins: cost {cheap.total_cost}")
    fixed = solve(edge_case, 0, clock=clock)
    show("min_high=4 forces the costlier option", fixed, clock)
    assert fixed.total_cost == 3 and fixed.choices == {"a": 1}

    print()
    print("  falling edge pushed past the next rising edge:")
    crossing = ClockTree(
        "r",
        [Edge("r", "a", 0, [Option(0, 0, fall_delay=13)])],  # high = 8+13 > 20
        [LeafWindow("a", 0, 9)],
    )
    bad = solve(crossing, 9, clock=clock)
    show("high would exceed the period -> proven infeasible", bad, clock)
    assert bad.status is Status.INFEASIBLE and bad.total_cost is None

    hr("3. DEGENERATE -- fall_delay == delay everywhere reproduces the old"
       " single-edge answer")
    plain = solve(tree, 0)
    same = solve(
        ClockTree(
            "r",
            [
                Edge("r", "u", 5, [Option(0, 0), Option(2, 4)]),
                Edge("u", "a", 1, [Option(0, 0), Option(3, 2)]),
                Edge("u", "b", 4, [Option(0, 0), Option(3, 2)]),
            ],
            [LeafWindow("a", 0, 14), LeafWindow("b", 0, 14)],
        ),
        0,
        clock=clock,
    )
    print(f"  old model (no clock)   : cost {plain.total_cost},"
          f" picks {plain.choices}")
    print(f"  equal-delay + clock    : cost {same.total_cost},"
          f" picks {same.choices}")
    assert (same.total_cost, same.choices) == (plain.total_cost, plain.choices)
    print("  identical optimum and option vector, as required.")

    print()
    print(f"demo_iteration finished in {time.perf_counter() - t0:.3f} s"
          " (all numbers computed now from the fixed inputs above).")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fixed-input demo for the skewbudget clock-tree allocation library.

Run:  python demo.py

Everything below is computed live (no sleeps, no pre-recorded output).  It
prints:
  1. a normal optimum at a loose skew bound and the same tree at a tight
     skew bound -- the skew/cost trade-off, with per-leaf path sums and the
     exact Pareto pruning counts;
  2. a rejection boundary that is actually triggered (NaN and bool inputs);
  3. the distinction between a *proven* infeasible window set and a
     budget cutoff whose result is explicitly not proven optimal;
  4. the multi-corner problem: one shared physical option per edge with
     per-corner delays, its proven optimum and per-corner verification
     values, a cross-corner conflict proven infeasible, and an actual
     over-limit rejection.
"""

from __future__ import annotations

import time

from skewbudget import (
    ClockTree,
    Corner,
    Edge,
    InvalidNumericError,
    InputTypeError,
    InputValueError,
    LeafWindow,
    MultiCornerTree,
    Option,
    PhysicalEdge,
    PhysicalOption,
    Status,
    solve,
    solve_multicorner,
)


def hr(title: str) -> None:
    print()
    print("=" * 68)
    print(title)
    print("=" * 68)


def show(label, res) -> None:
    print(f"[{label}]")
    print(f"  status       : {res.status.value}")
    print(f"  total cost   : {res.total_cost}")
    if res.total_cost is not None:
        print(f"  leaf arrivals: {res.leaf_arrivals}")
        print(f"  min/max delay: {res.min_delay} / {res.max_delay}"
              f"  (skew {res.skew} <= S={res.skew_limit})")
        print(f"  option picks : {res.choices}  (edge id = child vertex,"
              " option index in input order)")
    p = res.pruning
    print(f"  Pareto prune : dominated={p.dominated_states}"
          f" window-infeasible={p.window_states}"
          f" skew-infeasible={p.skew_states}")
    if p.budget_states:
        print(f"  BUDGET CUTOFF: {p.budget_states} states dropped;"
              f" cheapest dropped subtree cost = {p.cutoff_min_cost}"
              " (cutoff record only, NOT an optimality bound)")


def build_tree() -> ClockTree:
    #        r
    #        | base 5, shared trunk: (delay 0,cost 0) | (delay 2,cost 4)
    #        u
    #       / \
    #  a b1    b b4   branch options: (0,0) | (3,2)
    return ClockTree(
        "r",
        [
            Edge("r", "u", 5, [Option(0, 0), Option(2, 4)]),
            Edge("u", "a", 1, [Option(0, 0), Option(3, 2)]),
            Edge("u", "b", 4, [Option(0, 0), Option(3, 2)]),
        ],
        [LeafWindow("a", 0, 12), LeafWindow("b", 0, 12)],
    )


def main() -> None:
    t0 = time.perf_counter()

    hr("1. NORMAL RESULTS -- skew budget vs buffer cost (live exact DP)")
    tree = build_tree()
    print("tree: r-(5)->u ; u-(1)->a, u-(4)->b ; options per branch edge")
    print("      (delay 0,cost 0) or (delay 3,cost 2); trunk (0,0)/(2,4)")

    loose = solve(tree, skew_limit=3)
    show("loose bound S=3", loose)
    tight = solve(tree, skew_limit=0)
    show("tight bound S=0", tight)
    print()
    print("  trade-off: S=3 needs no buffer (cost 0, arrivals 6/9);")
    print("             S=0 forces one +3 branch buffer (cost 2, 9/9).")
    print("  leaf paths: a -> ['u','a'] sums base+option along that chain;"
          " trunk cost is paid once though both leaves share it.")

    hr("2. REJECTION BOUNDARY -- invalid numeric input is refused")
    for bad in (
        ("NaN delay", lambda: Option(float("nan"), 0)),
        ("bool where int is required", lambda: Edge("r", "a", True,
                                                    [Option(0, 0)])),
    ):
        name, make = bad
        try:
            make()
        except (InvalidNumericError, InputTypeError) as exc:
            print(f"  triggered for {name!r}: {type(exc).__name__}: {exc}")
        else:  # pragma: no cover - demo contract: this must never happen
            raise AssertionError(f"invalid input {name} was silently accepted")

    hr("3. PROVED INFEASIBLE  vs  BUDGET EXHAUSTED (not proven)")
    bad_windows = ClockTree(
        "r",
        [Edge("r", "a", 1, [Option(0, 0)]),
         Edge("r", "b", 1, [Option(0, 0)])],
        [LeafWindow("a", 0, 1), LeafWindow("b", 5, 6)],
    )
    inf = solve(bad_windows, skew_limit=10)
    show("disjoint windows, full search", inf)
    assert inf.status is Status.INFEASIBLE and inf.total_cost is None

    # A frontier cap of 1 on a 3-option/2-leaf tree forces a real cutoff;
    # the returned incumbent is feasible but optimality is not certified.
    cut_tree = ClockTree(
        "r",
        [Edge("r", "a", 0, [Option(0, 0), Option(1, 2), Option(2, 4)]),
         Edge("r", "b", 0, [Option(0, 0), Option(1, 2), Option(2, 4)])],
        [LeafWindow("a", 0, 9), LeafWindow("b", 0, 9)],
    )
    cut = solve(cut_tree, skew_limit=5, budget=1)
    show("frontier cap budget=1", cut)
    assert cut.status is Status.BUDGET_EXHAUSTED
    full = solve(cut_tree, skew_limit=5)
    print(f"  same instance with the default exhaustive frontier:"
          f" {full.status.value}, cost {full.total_cost}")

    hr("4. MULTI-CORNER -- one shared physical option, per-corner delays")
    # Two leaves, two corners.  Base latency differs by corner:
    #   fast: a=0, b=5   (b arrives 5 late)
    #   slow: a=5, b=0   (a arrives 5 late)
    # Every edge has the SAME physical catalogue (cost, fast-delay,
    # slow-delay): idx0 (0,0,0) | idx1 (6,5,0) | idx2 (6,0,5).
    # The option physically installed is shared across both corners, so
    # corner-specific retiming must come from one option's delay vector.
    mc_opts = [
        PhysicalOption(0, (0, 0)),
        PhysicalOption(6, (5, 0)),
        PhysicalOption(6, (0, 5)),
    ]
    mc_tree = MultiCornerTree(
        "r",
        [PhysicalEdge("r", "a", (0, 5), mc_opts),
         PhysicalEdge("r", "b", (5, 0), mc_opts)],
        [Corner("fast", 0, [LeafWindow("a", 0, 9), LeafWindow("b", 0, 9)]),
         Corner("slow", 0, [LeafWindow("a", 0, 9), LeafWindow("b", 0, 9)])],
    )
    mc = solve_multicorner(mc_tree)
    print(f"  status={mc.status.value}  cost={mc.total_cost}  "
          f"enumerated={mc.assignments_enumerated} "
          f"(feasible {mc.feasible_assignments})")
    for rpt in mc.corners:
        print(f"  corner {rpt.name:>4}: arrivals={rpt.leaf_arrivals} "
              f"skew={rpt.skew} (limit {rpt.skew_limit}) "
              f"window_ok={rpt.window_ok}")
    print(f"  physical picks: {mc.choices}  (a gets (5,0), b gets (0,5);")
    print("   installing corner-specific buffers independently is not an"
          " option -- cost 6+6 = 12)")
    assert mc.status.value == "optimal" and mc.total_cost == 12
    assert mc.choices == {"a": 1, "b": 2}

    # Boundary: windows that no single physical vector can meet in both
    # corners -- proven infeasible by exhaustive enumeration.
    conflict = MultiCornerTree(
        "r",
        [PhysicalEdge("r", "a", (0, 0),
                      [PhysicalOption(0, (0, 0)),
                       PhysicalOption(1, (2, 2))])],
        [Corner("fast", 0, [LeafWindow("a", 2, 2)]),
         Corner("slow", 0, [LeafWindow("a", 0, 0)])],
    )
    bad = solve_multicorner(conflict)
    print(f"  conflicting windows across corners: status="
          f"{bad.status.value} (enumerated "
          f"{bad.assignments_enumerated}, feasible "
          f"{bad.feasible_assignments})")
    assert bad.status.value == "infeasible" and bad.total_cost is None

    # Actual rejection: the enumeration contract caps choice points at 8;
    # a 9th multi-option edge is refused at construction (no half state).
    try:
        MultiCornerTree(
            "r",
            [PhysicalEdge("r", f"x{i}", (0,),
                          [PhysicalOption(0, (0,)), PhysicalOption(1, (1,))])
             for i in range(9)],
            [Corner("c", 9, [LeafWindow(f"x{i}", 0, 99) for i in range(9)])],
        )
    except InputValueError as exc:
        print(f"  triggered over-limit construction: "
              f"{type(exc).__name__}: {exc}")
    else:  # pragma: no cover - demo contract: this must never happen
        raise AssertionError("9 choice points were silently accepted")

    print()
    print(f"demo finished in {time.perf_counter() - t0:.3f} s"
          " (all numbers computed now from the fixed inputs above).")


if __name__ == "__main__":
    main()

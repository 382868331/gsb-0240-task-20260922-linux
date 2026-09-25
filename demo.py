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
     budget cutoff whose result is explicitly not proven optimal.
"""

from __future__ import annotations

import time

from skewbudget import (
    ClockTree,
    Edge,
    InvalidNumericError,
    InputTypeError,
    LeafWindow,
    Option,
    Status,
    solve,
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

    hr("4. FROZEN EDGES + CHANGE BUDGET K vs a reference plan")
    tree = build_tree()
    # Pin the shared trunk to its +2 buffer; one pin moves both leaves.
    frozen_res = solve(tree, skew_limit=0, frozen={"u": 1})
    show("trunk edge u frozen to option 1, S=0", frozen_res)
    assert frozen_res.choices["u"] == 1
    print("  frozen pin survived all pruning; both leaves share the"
          " shifted trunk arrival.")

    # Reference plan = all zero buffers (itself infeasible at S=0);
    # allow at most K=1 edge to deviate.
    reference = {"u": 0, "a": 0, "b": 0}
    ref_res = solve(tree, skew_limit=0, reference=reference, max_changes=1)
    show("reference all-zero plan, K=1 edit allowed", ref_res)
    print(f"  edits vs reference: {ref_res.changes} (K=1)")
    print(f"  modifications     : {ref_res.modifications}"
          "  (edge -> (reference idx, chosen idx))")
    print(f"  per-leaf paths    : {ref_res.leaf_path_edges}")
    print(f"  per-leaf arrivals : {ref_res.leaf_arrivals}")
    k0 = solve(tree, skew_limit=0, reference=reference, max_changes=0)
    print(f"  K=0 just validates the reference plan -> {k0.status.value}"
          " (the all-zero plan has skew 3 > 0, so it is infeasible)")
    assert k0.status is Status.INFEASIBLE

    print()
    print(f"demo finished in {time.perf_counter() - t0:.3f} s"
          " (all numbers computed now from the fixed inputs above).")


if __name__ == "__main__":
    main()

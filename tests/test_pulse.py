"""Pulse-width (two-edge) behaviour: independent brute-force cross-checks.

The reference in tests/brute.py enumerates every option combination and
scores rise/fall sums directly from the input model; it never imports the
DP core, so equality is a genuine cross-check.
"""

from __future__ import annotations

import random
import unittest

from skewbudget import (
    Clock,
    ClockTree,
    Edge,
    InputTypeError,
    InputValueError,
    LeafTiming,
    LeafWindow,
    Option,
    Status,
    solve,
)
from brute import optimal_reference


def assert_matches_reference(tc, tree, S, clock):
    res = solve(tree, S, clock=clock)
    ref = optimal_reference(tree, S, clock)
    if ref is None:
        tc.assertEqual(res.status, Status.INFEASIBLE)
        tc.assertIsNone(res.total_cost)
        return res
    choice, _arr, cost, hi, _lo = ref
    tc.assertEqual(res.status, Status.OPTIMAL)
    tc.assertEqual(res.total_cost, cost)
    tc.assertEqual(res.max_delay, hi)
    tc.assertEqual(res.choices, choice)
    return res


class PulseConstraintTests(unittest.TestCase):
    def test_rise_feasible_but_high_pulse_vanishes(self):
        # Counterexample: option 0 meets the rise window and skew, but its
        # fall delay 0 vs rise delay 4 shrinks the high pulse to 1 < 2.
        opts = [Option(4, 0, fall_delay=0), Option(4, 3, fall_delay=4)]
        tree = ClockTree(
            "r",
            [Edge("r", "a", 0, opts)],
            [LeafWindow("a", 4, 4, min_high=2)],
        )
        clock = Clock(period=10, high_width=5)
        no_clock = solve(tree, 0)
        self.assertEqual(no_clock.status, Status.OPTIMAL)
        self.assertEqual(no_clock.total_cost, 0)  # old model: opt 0 fine
        res = solve(tree, 0, clock=clock)
        self.assertEqual(res.status, Status.OPTIMAL)
        self.assertEqual(res.total_cost, 3)       # forced onto costlier opt
        self.assertEqual(res.choices, {"a": 1})
        self.assertEqual(res.leaf_timing["a"],
                         LeafTiming(rise=4, fall=9, high=5, low=5))
        assert_matches_reference(self, tree, 0, clock)

    def test_fall_crossing_next_rise_is_rejected(self):
        # high = 5 + 6 - 0 = 11 > period 10: the falling edge lands past
        # the next rising edge; infeasible even with min_low = 0.
        tree = ClockTree(
            "r",
            [Edge("r", "a", 0, [Option(0, 0, fall_delay=6)])],
            [LeafWindow("a", 0, 9)],
        )
        res = solve(tree, 9, clock=Clock(10, 5))
        self.assertEqual(res.status, Status.INFEASIBLE)
        self.assertGreater(res.pruning.pulse_states +
                           (1 if res.total_cost is None else 0), 0)
        # ... but the same tree without a clock is trivially feasible.
        self.assertEqual(solve(tree, 9).status, Status.OPTIMAL)

    def test_low_pulse_bound_is_checked(self):
        # high = 5 + 3 = 8, period 10 -> low = 2; min_low = 3 rejects it,
        # min_low = 2 accepts it exactly (closed bound).
        opts = [Option(0, 0, fall_delay=3)]
        for min_low, expect in ((3, Status.INFEASIBLE), (2, Status.OPTIMAL)):
            tree = ClockTree(
                "r",
                [Edge("r", "a", 0, opts)],
                [LeafWindow("a", 0, 9, min_low=min_low)],
            )
            res = solve(tree, 9, clock=Clock(10, 5))
            self.assertEqual(res.status, expect)
            if expect is Status.OPTIMAL:
                self.assertEqual(res.leaf_timing["a"].low, 2)

    def test_min_high_bound_closed_endpoint(self):
        # high = 5 + 1 - 3 = 3; min_high = 3 accepts exactly, 4 rejects.
        opts = [Option(3, 0, fall_delay=1)]
        for min_high, expect in ((3, Status.OPTIMAL), (4, Status.INFEASIBLE)):
            tree = ClockTree(
                "r",
                [Edge("r", "a", 0, opts)],
                [LeafWindow("a", 3, 3, min_high=min_high)],
            )
            res = solve(tree, 9, clock=Clock(10, 5))
            self.assertEqual(res.status, expect)

    def test_fall_may_cross_period_boundary_unified_output(self):
        # fall = 6 + 8 = 14 > period 10 (next cycle) but high = 6 <= 10:
        # legal, reported in the unified absolute representation.
        tree = ClockTree(
            "r",
            [Edge("r", "a", 8, [Option(0, 0)])],
            [LeafWindow("a", 8, 8, min_high=1, min_low=1)],
        )
        res = solve(tree, 0, clock=Clock(10, 6))
        self.assertEqual(res.status, Status.OPTIMAL)
        t = res.leaf_timing["a"]
        self.assertEqual((t.rise, t.fall, t.high, t.low), (8, 14, 6, 4))
        self.assertGreater(t.fall, 10)

    def test_skew_still_measured_on_rise_edges_only(self):
        # Identical rise arrivals (skew 0) but very different fall sums:
        # the skew bound must not look at falling edges.
        tree = ClockTree(
            "r",
            [Edge("r", "a", 0, [Option(2, 0, fall_delay=0)]),
             Edge("r", "b", 0, [Option(2, 0, fall_delay=4)])],
            [LeafWindow("a", 2, 2), LeafWindow("b", 2, 2)],
        )
        res = solve(tree, 0, clock=Clock(20, 6))
        self.assertEqual(res.status, Status.OPTIMAL)
        self.assertEqual(res.skew, 0)
        self.assertEqual(res.leaf_timing["a"].high, 4)
        self.assertEqual(res.leaf_timing["b"].high, 8)

    def test_pulse_is_hard_constraint_not_soft_objective(self):
        # The cheapest rise-feasible option violates min_high; the solver
        # must pay more rather than trade pulse width off against cost.
        opts = [Option(0, 0, fall_delay=0), Option(0, 5, fall_delay=2)]
        tree = ClockTree(
            "r",
            [Edge("r", "a", 0, opts)],
            [LeafWindow("a", 0, 0, min_high=4)],
        )
        res = solve(tree, 0, clock=Clock(10, 3))
        self.assertEqual(res.total_cost, 5)
        self.assertEqual(res.choices, {"a": 1})


class DegenerationTests(unittest.TestCase):
    def test_equal_edge_delays_reduce_to_old_problem(self):
        # fall_delay defaults to delay everywhere: with a clock the only
        # extra constraint is 0 <= high_width <= period, always true, so
        # the answer must equal the no-clock solve bit for bit.
        tree = ClockTree(
            "r",
            [Edge("r", "u", 5, [Option(0, 0), Option(2, 4)]),
             Edge("u", "a", 1, [Option(0, 0), Option(3, 2)]),
             Edge("u", "b", 4, [Option(0, 0), Option(3, 2)])],
            [LeafWindow("a", 0, 12), LeafWindow("b", 0, 12)],
        )
        plain = solve(tree, 0)
        clocked = solve(tree, 0, clock=Clock(20, 7))
        self.assertEqual(clocked.status, Status.OPTIMAL)
        self.assertEqual(
            (clocked.total_cost, clocked.max_delay, clocked.choices),
            (plain.total_cost, plain.max_delay, plain.choices),
        )
        for leaf, t in clocked.leaf_timing.items():
            self.assertEqual(t.high, 7)
            self.assertEqual(t.low, 13)
        assert_matches_reference(self, tree, 0, Clock(20, 7))


class RandomPulseComparisonTests(unittest.TestCase):
    def test_many_random_small_trees_with_clock(self):
        rng = random.Random(20260926)
        for trial in range(60):
            tree, S, clock = self._make(rng)
            with self.subTest(trial=trial):
                res = assert_matches_reference(self, tree, S, clock)
                if res.total_cost is not None:
                    # every reported leaf timing satisfies the contract
                    for w in tree.windows:
                        t = res.leaf_timing[w.leaf]
                        self.assertTrue(
                            w.min_high <= t.high
                            <= clock.period - w.min_low)
                        self.assertEqual(t.high + t.low, clock.period)
                        self.assertEqual(t.rise, res.leaf_arrivals[w.leaf])

    @staticmethod
    def _make(rng):
        n_edges = rng.randint(1, 5)
        labels = ["r"]
        edges = []
        next_id = 0
        while len(edges) < n_edges:
            parent = rng.choice(labels)
            child = f"n{next_id}"
            next_id += 1
            base = rng.randint(0, 4)
            opts = []
            for _ in range(rng.randint(1, 3)):
                opts.append(Option(rng.randint(0, 5), rng.randint(0, 6),
                                   fall_delay=rng.randint(0, 5)))
            edges.append(Edge(parent, child, base, opts))
            labels.append(child)
        children = {}
        for e in edges:
            children.setdefault(e.parent, []).append(e.child)
        period = rng.randint(8, 16)
        clock = Clock(period, rng.randint(0, period))
        windows = []
        for lf in (x for x in labels if x not in children):
            lo = rng.randint(0, 10)
            hi = lo + rng.randint(0, 6)
            mh = rng.randint(0, 3)
            ml = rng.randint(0, 3)
            windows.append(LeafWindow(lf, lo, hi, min_high=mh, min_low=ml))
        return ClockTree("r", edges, windows), rng.randint(0, 6), clock


class PulseInputValidationTests(unittest.TestCase):
    def test_clock_validation(self):
        with self.assertRaises(InputValueError):
            Clock(0, 0)
        with self.assertRaises(InputValueError):
            Clock(10, 11)
        with self.assertRaises(InputValueError):
            Clock(10, -1)
        with self.assertRaises(InputTypeError):
            Clock(True, 0)
        with self.assertRaises(InputTypeError):
            Clock(10.0, 5)

    def test_option_fall_delay_validation(self):
        with self.assertRaises(InputValueError):
            Option(0, 0, fall_delay=-1)
        with self.assertRaises(InputTypeError):
            Option(0, 0, fall_delay=True)
        with self.assertRaises(InputTypeError):
            Option(0, 0, fall_delay=1.5)
        self.assertEqual(Option(2, 1).fall_delay, 2)  # default = delay

    def test_window_pulse_bound_validation(self):
        with self.assertRaises(InputValueError):
            LeafWindow("a", 0, 1, min_high=-1)
        with self.assertRaises(InputTypeError):
            LeafWindow("a", 0, 1, min_low=False)

    def test_solve_clock_type(self):
        tree = ClockTree(
            "r", [Edge("r", "a", 0, [Option(0, 0)])],
            [LeafWindow("a", 0, 1)],
        )
        with self.assertRaises(InputTypeError):
            solve(tree, 0, clock="not-a-clock")

    def test_same_rise_delay_distinct_fall_delay_not_deduped(self):
        e = Edge("r", "a", 0, [Option(2, 1, fall_delay=0),
                               Option(2, 1, fall_delay=3)])
        self.assertEqual(len(e.options), 2)


if __name__ == "__main__":
    unittest.main()


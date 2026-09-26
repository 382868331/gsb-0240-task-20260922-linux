"""Pulse-width (two-edge) behaviour, cross-checked against the
independent exhaustive enumerator in tests/brute.py (which never
imports the solver)."""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from brute import enumerate_solutions, optimal_reference  # noqa: E402

from skewbudget import (  # noqa: E402
    ClockTree,
    Edge,
    InputTypeError,
    InputValueError,
    LeafWindow,
    Option,
    RootClock,
    Status,
    solve,
)

CLK = RootClock(period=10, high_width=4)


def assert_matches_reference(testcase, tree, S, clock, budget=None):
    ref = optimal_reference(tree, S, clock)
    res = solve(tree, S, clock=clock, budget=budget)
    if ref is None:
        testcase.assertIn(
            res.status, (Status.INFEASIBLE, Status.BUDGET_EXHAUSTED))
        testcase.assertIsNone(res.total_cost)
        return res
    testcase.assertEqual(res.status, Status.OPTIMAL)
    choice, arrivals, cost, hi, lo = ref
    testcase.assertEqual(res.total_cost, cost)
    testcase.assertEqual(res.max_delay, hi)
    testcase.assertEqual(res.min_delay, lo)
    testcase.assertEqual(res.leaf_arrivals, arrivals)
    testcase.assertEqual(res.choices, choice)
    # Per-leaf two-edge timing must be reproducible from the raw model.
    if clock is not None:
        testcase.assertIsNotNone(res.leaf_edges)
        for leaf in tree.leaf_set:
            rise = fall_sum = 0
            v = leaf
            while v != tree.root:
                e = tree.parent_edge[v]
                o = e.options[res.choices[v]]
                rise += e.base_delay + o.rise_delay
                fall_sum += e.base_delay + o.fall_delay
                v = e.parent
            lt = res.leaf_edges[leaf]
            testcase.assertEqual(lt.rise, rise)
            testcase.assertEqual(lt.fall, clock.high_width + fall_sum)
            testcase.assertEqual(lt.high_width, lt.fall - lt.rise)
            testcase.assertEqual(lt.low_width, clock.period - lt.high_width)
            testcase.assertEqual(lt.rise_phase, rise % clock.period)
            testcase.assertEqual(lt.fall_phase, lt.fall % clock.period)
            w = tree.window_of(leaf)
            testcase.assertGreaterEqual(lt.high_width, w.min_high)
            testcase.assertGreaterEqual(lt.low_width, w.min_low)
    return res


class DegenerateTests(unittest.TestCase):
    def test_equal_edge_delays_reduce_to_old_problem(self):
        # rise_delay == fall_delay == delay everywhere: the clock must not
        # change the answer (all pulse widths stay at the source values).
        edges = [
            Edge("r", "u", 5, [Option(0, 0), Option(2, 4)]),
            Edge("u", "a", 1, [Option(0, 0), Option(3, 2)]),
            Edge("u", "b", 4, [Option(0, 0), Option(3, 2)]),
        ]
        wins = [LeafWindow("a", 0, 12), LeafWindow("b", 0, 12)]
        tree = ClockTree("r", edges, wins)
        for S in (0, 1, 3):
            plain = solve(tree, S)
            clocked = solve(tree, S, clock=CLK)
            self.assertEqual(clocked.status, plain.status)
            self.assertEqual(
                (clocked.total_cost, clocked.max_delay, clocked.choices),
                (plain.total_cost, plain.max_delay, plain.choices))
            assert_matches_reference(self, tree, S, CLK)
        # source high width propagates untouched through equal delays
        res = solve(tree, 3, clock=CLK)
        for lt in res.leaf_edges.values():
            self.assertEqual(lt.high_width, CLK.high_width)
            self.assertEqual(lt.low_width, CLK.period - CLK.high_width)

    def test_no_clock_means_no_pulse_fields(self):
        tree = ClockTree(
            "r", [Edge("r", "a", 3, [Option(0, 0)])],
            [LeafWindow("a", 0, 9)])
        res = solve(tree, 0)
        self.assertIsNone(res.clock)
        self.assertEqual(res.leaf_edges, {})


class PulseCounterexampleTests(unittest.TestCase):
    def test_rise_feasible_but_high_pulse_vanishes(self):
        # Rising edge lands inside the window (arrival 9 in [8, 10]) and
        # skew is fine, but fall_delay < rise_delay shrinks the high pulse
        # from 4 to 1 < min_high=3: rising-edge-only checking would
        # wrongly accept this option.
        tree = ClockTree(
            "r",
            [Edge("r", "a", 5,
                  [Option(4, 0, rise_delay=4, fall_delay=1)])],
            [LeafWindow("a", 8, 10, min_high=3)])
        plain = solve(tree, 0)
        self.assertEqual(plain.status, Status.OPTIMAL)  # old check passes
        res = solve(tree, 0, clock=CLK)
        self.assertEqual(res.status, Status.INFEASIBLE)
        self.assertIsNone(res.total_cost)
        self.assertGreaterEqual(res.pruning.pulse_states, 1)
        self.assertIsNone(optimal_reference(tree, 0, CLK))

    def test_fall_crossing_next_rise_is_rejected(self):
        # fall_delay - rise_delay = 8 > period - high_width = 6: the fall
        # edge would land past the next rising edge even with min_low=0.
        tree = ClockTree(
            "r",
            [Edge("r", "a", 0,
                  [Option(0, 0, rise_delay=0, fall_delay=8)])],
            [LeafWindow("a", 0, 10)])
        res = solve(tree, 0, clock=CLK)
        self.assertEqual(res.status, Status.INFEASIBLE)
        self.assertIsNone(optimal_reference(tree, 0, CLK))
        # Exactly at the boundary (fall == next rise) is allowed.
        tree_ok = ClockTree(
            "r",
            [Edge("r", "a", 0,
                  [Option(0, 0, rise_delay=0, fall_delay=6)])],
            [LeafWindow("a", 0, 10)])
        res_ok = solve(tree_ok, 0, clock=CLK)
        self.assertEqual(res_ok.status, Status.OPTIMAL)
        lt = res_ok.leaf_edges["a"]
        self.assertEqual((lt.high_width, lt.low_width), (10, 0))
        self.assertEqual(lt.fall, lt.rise + CLK.period)

    def test_min_low_forces_slower_fall_option(self):
        # Cheap option keeps the high pulse but its low pulse 10-8=2 < 4;
        # the expensive option with a slower fall edge is the only fit.
        tree = ClockTree(
            "r",
            [Edge("r", "a", 0,
                  [Option(0, 0, rise_delay=0, fall_delay=4),
                   Option(0, 5, rise_delay=0, fall_delay=2)])],
            [LeafWindow("a", 0, 10, min_low=4)])
        res = solve(tree, 0, clock=CLK)
        self.assertEqual(res.status, Status.OPTIMAL)
        self.assertEqual(res.choices, {"a": 1})
        self.assertEqual(res.total_cost, 5)
        lt = res.leaf_edges["a"]
        self.assertEqual((lt.high_width, lt.low_width), (6, 4))
        assert_matches_reference(self, tree, 0, CLK)

    def test_pulse_conflict_between_siblings(self):
        # Shared trunk shifts both leaves' pulses together; only one
        # trunk/branch combination keeps both leaves inside [min_high,
        # period-min_low].  Checked against full enumeration.
        tree = ClockTree(
            "r",
            [Edge("r", "u", 2,
                  [Option(0, 0, rise_delay=0, fall_delay=0),
                   Option(1, 3, rise_delay=1, fall_delay=3)]),
             Edge("u", "a", 1,
                  [Option(0, 0, rise_delay=0, fall_delay=0),
                   Option(2, 2, rise_delay=2, fall_delay=1)]),
             Edge("u", "b", 3,
                  [Option(0, 0, rise_delay=0, fall_delay=0),
                   Option(1, 1, rise_delay=1, fall_delay=2)])],
            [LeafWindow("a", 0, 12, min_high=2, min_low=2),
             LeafWindow("b", 0, 12, min_high=2, min_low=2)])
        res = assert_matches_reference(self, tree, 6, CLK)
        self.assertEqual(res.status, Status.OPTIMAL)
        for leaf, lt in res.leaf_edges.items():
            w = tree.window_of(leaf)
            self.assertTrue(w.min_high <= lt.high_width)
            self.assertTrue(w.min_low <= lt.low_width)

    def test_period_crossing_uniform_phases(self):
        # Arrival beyond one period: phases are reported modulo the
        # period while absolute times stay exact.
        tree = ClockTree(
            "r",
            [Edge("r", "a", 12,
                  [Option(0, 0, rise_delay=1, fall_delay=2)])],
            [LeafWindow("a", 0, 20)])
        res = solve(tree, 0, clock=CLK)
        self.assertEqual(res.status, Status.OPTIMAL)
        lt = res.leaf_edges["a"]
        self.assertEqual(lt.rise, 13)
        self.assertEqual(lt.fall, 4 + 14)
        self.assertEqual(lt.rise_phase, 13 % 10)
        self.assertEqual(lt.fall_phase, 18 % 10)
        self.assertEqual(lt.high_width, 5)


class ObjectiveOrderTests(unittest.TestCase):
    def test_pulse_width_is_a_hard_constraint_not_a_penalty(self):
        # Option 0 is cheapest and gives the *smallest* high width; if
        # pulse width were a soft objective the solver might pick wider
        # pulses.  With min_high=0 both are feasible and the choice must
        # follow (cost, max rise delay, option vector) only.
        tree = ClockTree(
            "r",
            [Edge("r", "a", 0,
                  [Option(0, 0, rise_delay=0, fall_delay=0),
                   Option(0, 0, rise_delay=0, fall_delay=3)])],
            [LeafWindow("a", 0, 10)])
        res = solve(tree, 0, clock=CLK)
        self.assertEqual(res.choices, {"a": 0})
        self.assertEqual(res.leaf_edges["a"].high_width, CLK.high_width)

    def test_tie_break_still_cost_maxdelay_vector(self):
        tree = ClockTree(
            "r",
            [Edge("r", "a", 0,
                  [Option(0, 0), Option(2, 3, rise_delay=2, fall_delay=2),
                   Option(4, 3, rise_delay=4, fall_delay=4)]),
             Edge("r", "b", 0,
                  [Option(0, 0), Option(2, 3, rise_delay=2, fall_delay=2),
                   Option(4, 3, rise_delay=4, fall_delay=4)])],
            [LeafWindow("a", 2, 4), LeafWindow("b", 2, 4)])
        res = solve(tree, 0, clock=CLK)
        self.assertEqual((res.total_cost, res.max_delay), (6, 2))
        assert_matches_reference(self, tree, 0, CLK)


class RandomPulseComparisonTests(unittest.TestCase):
    def test_random_small_trees_with_clock(self):
        rng = random.Random(20260926)
        for trial in range(60):
            tree, S, clock = self._make(rng)
            with self.subTest(trial=trial):
                assert_matches_reference(self, tree, S, clock)

    @staticmethod
    def _make(rng):
        period = rng.randint(6, 12)
        clock = RootClock(period, rng.randint(1, period - 1))
        labels = ["r"]
        edges = []
        next_id = 0
        for _ in range(rng.randint(1, 5)):
            parent = rng.choice(labels)
            child = f"n{next_id}"
            next_id += 1
            base = rng.randint(0, 4)
            opts = []
            for _ in range(rng.randint(1, 3)):
                rd = rng.randint(0, 5)
                fd = rng.randint(0, 5)
                opts.append(Option(max(rd, fd), rng.randint(0, 6),
                                   rise_delay=rd, fall_delay=fd))
            edges.append(Edge(parent, child, base, opts))
            labels.append(child)
        children = {}
        for e in edges:
            children.setdefault(e.parent, []).append(e.child)
        windows = []
        for lf in (x for x in labels if x not in children):
            lo = rng.randint(0, 8)
            windows.append(LeafWindow(
                lf, lo, lo + rng.randint(0, 6),
                min_high=rng.randint(0, 3), min_low=rng.randint(0, 3)))
        return ClockTree("r", edges, windows), rng.randint(0, 6), clock


class PulseInputValidationTests(unittest.TestCase):
    def test_root_clock_validation(self):
        with self.assertRaises(InputValueError):
            RootClock(0, 0)
        with self.assertRaises(InputValueError):
            RootClock(10, 11)
        with self.assertRaises(InputTypeError):
            RootClock(True, 1)
        with self.assertRaises(InputTypeError):
            RootClock(10.0, 4)

    def test_option_edge_delay_validation(self):
        with self.assertRaises(InputValueError):
            Option(0, 0, rise_delay=-1)
        with self.assertRaises(InputValueError):
            Option(0, 0, fall_delay=-2)
        with self.assertRaises(InputTypeError):
            Option(0, 0, rise_delay=1.5)
        with self.assertRaises(InputTypeError):
            Option(0, 0, fall_delay=True)

    def test_leaf_window_pulse_validation(self):
        with self.assertRaises(InputValueError):
            LeafWindow("a", 0, 1, min_high=-1)
        with self.assertRaises(InputTypeError):
            LeafWindow("a", 0, 1, min_low=2.0)

    def test_solve_clock_type(self):
        tree = ClockTree(
            "r", [Edge("r", "a", 0, [Option(0, 0)])],
            [LeafWindow("a", 0, 1)])
        with self.assertRaises(InputTypeError):
            solve(tree, 0, clock="not-a-clock")

    def test_same_delay_different_edges_are_distinct_options(self):
        # Same nominal delay but different fall delays: both must survive
        # preprocessing (dedup keys on the full timing triple).
        e = Edge("r", "a", 0,
                 [Option(2, 1, rise_delay=2, fall_delay=2),
                  Option(2, 1, rise_delay=2, fall_delay=5)])
        self.assertEqual(len(e.options), 2)
        # Identical timing triple: cheapest kept, first wins on ties.
        e2 = Edge("r", "a", 0,
                  [Option(2, 9), Option(2, 3), Option(2, 7)])
        self.assertEqual([(o.delay, o.cost) for o in e2.options], [(2, 3)])

    def test_impossible_pulse_minimums_are_infeasible_not_an_error(self):
        # min_high + min_low > period: no waveform can satisfy it.
        tree = ClockTree(
            "r", [Edge("r", "a", 0, [Option(0, 0)])],
            [LeafWindow("a", 0, 10, min_high=8, min_low=8)])
        res = solve(tree, 0, clock=CLK)
        self.assertEqual(res.status, Status.INFEASIBLE)
        self.assertIsNone(optimal_reference(tree, 0, CLK))


if __name__ == "__main__":
    unittest.main()

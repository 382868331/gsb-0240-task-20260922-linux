"""DP correctness: every small case is cross-checked against the
independent exhaustive enumerator in tests/brute.py (which never imports
the solver)."""

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
    Status,
    solve,
    validate_tree,
)


def assert_matches_reference(testcase, tree, S, budget=None):
    ref = optimal_reference(tree, S)
    res = solve(tree, S, budget=budget)
    if ref is None:
        testcase.assertIn(
            res.status, (Status.INFEASIBLE, Status.BUDGET_EXHAUSTED))
        testcase.assertIsNone(res.total_cost)
        return
    testcase.assertEqual(res.status, Status.OPTIMAL)
    _choice, arrivals, cost, hi, lo = ref
    testcase.assertEqual(res.total_cost, cost)
    testcase.assertEqual(res.max_delay, hi)
    testcase.assertEqual(res.min_delay, lo)
    testcase.assertEqual(res.leaf_arrivals, arrivals)
    testcase.assertEqual(res.skew, hi - lo)
    # Arrival sums must reproduce along the reported path edge lists.
    for leaf, chain in res.leaf_path_edges.items():
        total = 0
        for child_endpoint in chain:
            e = tree.parent_edge[child_endpoint]
            total += e.base_delay + e.options[res.choices[child_endpoint]].delay
        testcase.assertEqual(total, arrivals[leaf])
    return res


class BruteForceComparisonTests(unittest.TestCase):
    def test_single_edge(self):
        t = ClockTree(
            "r", [Edge("r", "a", 3, [Option(0, 0), Option(2, 1)])],
            [LeafWindow("a", 0, 6)])
        assert_matches_reference(self, t, 0)

    def test_two_leaves_forces_buffer_for_skew(self):
        t = ClockTree(
            "r",
            [Edge("r", "a", 1, [Option(0, 0), Option(2, 5)]),
             Edge("r", "b", 3, [Option(0, 0), Option(2, 5)])],
            [LeafWindow("a", 0, 9), LeafWindow("b", 0, 9)])
        res = assert_matches_reference(self, t, 0)
        # concrete check: only buffering 'a' by 2 balances 3 vs 3
        self.assertEqual(res.choices, {"a": 1, "b": 0})
        self.assertEqual(res.leaf_arrivals, {"a": 3, "b": 3})
        self.assertEqual(res.total_cost, 5)

    def test_shared_trunk_cost_counted_once(self):
        # r -> x (shared) -> a,b. Trunk buffer cost must be counted once
        # even though two leaf paths traverse it.
        t = ClockTree(
            "r",
            [Edge("r", "x", 10, [Option(0, 0), Option(1, 7)]),
             Edge("x", "a", 1, [Option(0, 0), Option(1, 1)]),
             Edge("x", "b", 1, [Option(0, 0), Option(1, 1)])],
            [LeafWindow("a", 0, 20), LeafWindow("b", 0, 20)])
        res = assert_matches_reference(self, t, 1)
        self.assertEqual(res.total_cost, 0)

    def test_infeasible_windows(self):
        t = ClockTree(
            "r",
            [Edge("r", "a", 1, [Option(0, 0)]),
             Edge("r", "b", 1, [Option(0, 0)])],
            [LeafWindow("a", 0, 1), LeafWindow("b", 5, 6)])
        res = solve(t, 10)
        self.assertEqual(res.status, Status.INFEASIBLE)
        self.assertFalse(res.pruning.truncated)
        self.assertIsNone(res.total_cost)
        self.assertEqual(optimal_reference(t, 10), None)

    def test_skew_infeasible_even_with_wide_windows(self):
        t = ClockTree(
            "r",
            [Edge("r", "a", 0, [Option(0, 0)]),
             Edge("r", "b", 5, [Option(0, 0)])],
            [LeafWindow("a", 0, 100), LeafWindow("b", 0, 100)])
        res = solve(t, 2)
        self.assertEqual(res.status, Status.INFEASIBLE)
        self.assertGreaterEqual(res.pruning.skew_states, 1)

    def test_zero_buffer_only_options(self):
        t = ClockTree(
            "r",
            [Edge("r", "a", 2, [Option(0, 0)]),
             Edge("r", "b", 2, [Option(0, 0)])],
            [LeafWindow("a", 2, 2), LeafWindow("b", 2, 2)])
        res = assert_matches_reference(self, t, 0)
        self.assertEqual(res.choices, {"a": 0, "b": 0})

    def test_nonconvex_options(self):
        # delays 0,3,4 with costs 0,1,100: delay-3 dominates delay-4 on
        # (delay,cost) here, but other edges make the global picture
        # non-convex; the DP keeps every non-dominated combination.
        t = ClockTree(
            "r",
            [Edge("r", "a", 0, [Option(0, 0), Option(3, 1), Option(4, 100)]),
             Edge("r", "b", 0, [Option(0, 0), Option(3, 1), Option(4, 100)])],
            [LeafWindow("a", 0, 4), LeafWindow("b", 0, 4)])
        assert_matches_reference(self, t, 1)
        assert_matches_reference(self, t, 0)

    def test_nonconvex_dominated_delay_still_needed_for_window(self):
        # delay 4 is NOT dominated here because cheap delays cannot hit the
        # upper/lower window jointly in a sibling shape.
        t = ClockTree(
            "r",
            [Edge("r", "a", 0, [Option(0, 0), Option(3, 10), Option(4, 1)]),
             Edge("r", "b", 0, [Option(0, 0), Option(3, 10), Option(4, 1)])],
            [LeafWindow("a", 4, 4), LeafWindow("b", 4, 4)])
        res = solve(t, 0)
        self.assertEqual(res.status, Status.OPTIMAL)
        self.assertEqual(res.total_cost, 2)
        self.assertEqual(res.choices, {"a": 2, "b": 2})

    def test_tie_break_cost_then_maxdelay_then_vector(self):
        # Windows exclude the zero-cost choice.  At minimum cost 6 the
        # zero-skew choices give either arrivals (2,2) or (4,4), both
        # cost 6: the smaller max delay (2) must win.
        t = ClockTree(
            "r",
            [Edge("r", "a", 0, [Option(0, 0), Option(2, 3), Option(4, 3)]),
             Edge("r", "b", 0, [Option(0, 0), Option(2, 3), Option(4, 3)])],
            [LeafWindow("a", 2, 4), LeafWindow("b", 2, 4)])
        res = solve(t, 0)
        self.assertEqual(res.status, Status.OPTIMAL)
        self.assertEqual((res.total_cost, res.max_delay), (6, 2))
        assert_matches_reference(self, t, 0)

    def test_tie_break_on_option_vector(self):
        # Same geometry/cost for option 0 vs option 1: smaller indices win.
        t = ClockTree(
            "r",
            [Edge("r", "a", 0, [Option(2, 1), Option(2, 1)]),
             Edge("r", "b", 0, [Option(2, 1), Option(2, 1)])],
            [LeafWindow("a", 0, 5), LeafWindow("b", 0, 5)])
        res = solve(t, 0)
        self.assertEqual(res.choices, {"a": 0, "b": 0})
        assert_matches_reference(self, t, 0)

    def test_closed_window_endpoints_are_feasible(self):
        t = ClockTree(
            "r", [Edge("r", "a", 5, [Option(0, 0)])],
            [LeafWindow("a", 5, 5)])
        res = solve(t, 0)
        self.assertEqual(res.status, Status.OPTIMAL)
        self.assertEqual(res.leaf_arrivals["a"], 5)

    def test_pruning_counts_are_nonnegative_and_consistent(self):
        t = ClockTree(
            "r",
            [Edge("r", "a", 0, [Option(0, 0), Option(5, 1)]),
             Edge("r", "b", 0, [Option(0, 0), Option(5, 1)])],
            [LeafWindow("a", 0, 5), LeafWindow("b", 0, 5)])
        res = solve(t, 0)
        p = res.pruning
        for n in (p.dominated_states, p.window_states, p.skew_states,
                  p.budget_states, p.nodes_visited):
            self.assertGreaterEqual(n, 0)
        self.assertEqual(p.nodes_visited, 3)


class RandomTreeComparisonTests(unittest.TestCase):
    """Deterministically generated small trees: DP answer must equal the
    brute-force optimum on every instance."""

    def test_many_random_small_trees(self):
        rng = random.Random(20260923)
        for trial in range(60):
            tree, S = self._make(rng, trial)
            with self.subTest(trial=trial):
                assert_matches_reference(self, tree, S)

    @staticmethod
    def _make(rng, trial):
        n_internal_edges = rng.randint(1, 5)
        # grow random rooted tree
        labels = ["r"]
        edges = []
        next_id = 0
        while len(edges) < n_internal_edges:
            parent = rng.choice(labels)
            child = f"n{next_id}"
            next_id += 1
            base = rng.randint(0, 4)
            n_opt = rng.randint(1, 3)
            # random non-convex option set
            delays = sorted(rng.sample(range(0, 7), n_opt))
            opts = [Option(d, rng.randint(0, 6)) for d in delays]
            edges.append(Edge(parent, child, base, opts))
            labels.append(child)
        children = {}
        for e in edges:
            children.setdefault(e.parent, []).append(e.child)
        leaves = [x for x in labels if x not in children]
        windows = []
        for lf in leaves:
            lo = rng.randint(0, 8)
            hi = lo + rng.randint(0, 6)
            windows.append(LeafWindow(lf, lo, hi))
        tree = ClockTree("r", edges, windows)
        S = rng.randint(0, 6)
        return tree, S


class BudgetTests(unittest.TestCase):
    def test_budget_validation(self):
        t = ClockTree(
            "r", [Edge("r", "a", 0, [Option(0, 0)])],
            [LeafWindow("a", 0, 1)])
        with self.assertRaises(InputValueError):
            solve(t, 0, budget=0)
        with self.assertRaises(InputTypeError):
            solve(t, 0, budget=True)
        with self.assertRaises(InputValueError):
            solve(t, -1)
        with self.assertRaises(InputTypeError):
            solve("not-a-tree", 0)

    def test_tiny_budget_flags_unproven_but_keeps_incumbent(self):
        # 3 options per edge, 2 leaves: cap 1 forces truncation.
        t = ClockTree(
            "r",
            [Edge("r", "a", 0, [Option(0, 0), Option(1, 2), Option(2, 4)]),
             Edge("r", "b", 0, [Option(0, 0), Option(1, 2), Option(2, 4)])],
            [LeafWindow("a", 0, 9), LeafWindow("b", 0, 9)])
        res = solve(t, 5, budget=1)
        self.assertEqual(res.status, Status.BUDGET_EXHAUSTED)
        self.assertGreater(res.pruning.budget_states, 0)
        self.assertIsNotNone(res.pruning.cutoff_min_cost)
        self.assertFalse(res.proven_optimal)
        # an incumbent, and it is genuinely feasible
        self.assertIsNotNone(res.total_cost)
        self.assertLessEqual(res.skew, 5)
        for w in t.windows:
            self.assertTrue(w.lo <= res.leaf_arrivals[w.leaf] <= w.hi)

    def test_unlimited_budget_equals_default_on_small_case(self):
        t = ClockTree(
            "r",
            [Edge("r", "a", 0, [Option(0, 0), Option(3, 1), Option(4, 100)]),
             Edge("r", "b", 0, [Option(0, 0), Option(3, 1), Option(4, 100)])],
            [LeafWindow("a", 0, 4), LeafWindow("b", 0, 4)])
        r1 = solve(t, 1)
        r2 = solve(t, 1, budget=None)
        self.assertEqual(r1.status, Status.OPTIMAL)
        self.assertEqual((r1.total_cost, r1.max_delay, r1.choices),
                         (r2.total_cost, r2.max_delay, r2.choices))

    def test_budget_exhaustion_distinct_from_infeasible(self):
        t = ClockTree(
            "r",
            [Edge("r", "a", 0, [Option(0, 0)]),
             Edge("r", "b", 0, [Option(0, 0)])],
            [LeafWindow("a", 9, 9), LeafWindow("b", 9, 9)])
        # No possible delay reaches 9; no truncation occurs either.
        res = solve(t, 0, budget=1)
        self.assertEqual(res.status, Status.INFEASIBLE)
        self.assertEqual(res.pruning.budget_states, 0)


class DegenerateAndAPITests(unittest.TestCase):
    def test_validate_tree_summary(self):
        t = ClockTree("r", [], [])
        summary = validate_tree(t)
        self.assertEqual(summary, {"root": "r", "n_edges": 0, "leaves": ()})

    def test_lone_root(self):
        t = ClockTree("r", [], [])
        res = solve(t, 0)
        self.assertEqual(res.status, Status.OPTIMAL)
        self.assertEqual(res.total_cost, 0)
        self.assertEqual(res.leaf_arrivals, {})

    def test_choices_use_public_option_indices(self):
        opts = [Option(0, 0), Option(2, 5)]
        t = ClockTree(
            "r",
            [Edge("r", "a", 1, opts), Edge("r", "b", 3, opts)],
            [LeafWindow("a", 0, 9), LeafWindow("b", 0, 9)])
        res = solve(t, 0)
        for child, idx in res.choices.items():
            edge = t.parent_edge[child]
            # reconstruction must match reported arrivals
            self.assertEqual(
                edge.options[idx].delay,
                res.leaf_arrivals[child] - edge.base_delay)


if __name__ == "__main__":
    unittest.main()

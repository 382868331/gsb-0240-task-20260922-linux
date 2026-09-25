"""Freeze-table + change-budget (solve_with_reference) tests.

Every small case is cross-checked against the independent exhaustive
enumerator in tests/brute.py (which never imports the solver core).
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from brute import optimal_reference  # noqa: E402

from skewbudget import (  # noqa: E402
    ClockTree,
    Edge,
    InputTypeError,
    InputValueError,
    LeafWindow,
    Modification,
    Option,
    Status,
    solve,
    solve_with_reference,
)


def trunk_tree():
    # r -(5)-> u -(1)-> a ; u -(4)-> b
    # trunk options (0,0)/(2,4); branch options (0,0)/(3,2)
    return ClockTree(
        "r",
        [
            Edge("r", "u", 5, [Option(0, 0), Option(2, 4)]),
            Edge("u", "a", 1, [Option(0, 0), Option(3, 2)]),
            Edge("u", "b", 4, [Option(0, 0), Option(3, 2)]),
        ],
        [LeafWindow("a", 8, 8), LeafWindow("b", 11, 11)],
    )


ZERO_REF = {"u": 0, "a": 0, "b": 0}


def assert_matches_reference(tc, tree, S, ref, K, frozen=None, budget=None):
    want = optimal_reference(tree, S, ref, K, frozen)
    res = solve_with_reference(tree, S, ref, K, frozen=frozen, budget=budget)
    if want is None:
        tc.assertIn(res.status, (Status.INFEASIBLE, Status.BUDGET_EXHAUSTED))
        tc.assertIsNone(res.total_cost)
        return res
    tc.assertEqual(res.status, Status.OPTIMAL)
    choice, arrivals, cost, hi, lo = want
    tc.assertEqual(res.choices, choice)
    tc.assertEqual(res.total_cost, cost)
    tc.assertEqual(res.max_delay, hi)
    tc.assertEqual(res.min_delay, lo)
    tc.assertEqual(res.leaf_arrivals, arrivals)
    # change accounting: exactly the non-frozen edges that differ
    frozen = frozen or {}
    diffs = {ch for ch in choice if ch not in frozen and choice[ch] != ref[ch]}
    tc.assertEqual(res.n_changes, len(diffs))
    tc.assertLessEqual(res.n_changes, K)
    tc.assertEqual({m.edge for m in res.modifications}, diffs)
    for m in res.modifications:
        tc.assertIsInstance(m, Modification)
        tc.assertEqual(m.reference_index, ref[m.edge])
        tc.assertEqual(m.chosen_index, choice[m.edge])
    # per-leaf path sums reproduce the reported arrivals
    for leaf, chain in res.leaf_path_edges.items():
        total = 0
        for child_endpoint in chain:
            e = tree.parent_edge[child_endpoint]
            total += e.base_delay + e.options[res.choices[child_endpoint]].delay
        tc.assertEqual(total, arrivals[leaf])
    # frozen edges keep their welded option
    for ch, idx in frozen.items():
        tc.assertEqual(res.choices[ch], idx)
    return res


class FreezeAndChangeBudgetTests(unittest.TestCase):
    def test_shared_trunk_buffer_fixes_two_leaves_with_one_change(self):
        # Reference (all zero) gives arrivals 6/9, outside windows 8/11.
        # One trunk buffer shifts BOTH leaves at once: a single change.
        res = assert_matches_reference(self, trunk_tree(), 3, ZERO_REF, 1)
        self.assertEqual(res.choices, {"u": 1, "a": 0, "b": 0})
        self.assertEqual(res.leaf_arrivals, {"a": 8, "b": 11})
        self.assertEqual(res.n_changes, 1)
        self.assertEqual(res.total_cost, 4)
        self.assertEqual([m.edge for m in res.modifications], ["u"])
        self.assertEqual(res.leaf_path_edges["a"], ("u", "a"))

    def test_k0_only_validates_reference(self):
        # K=0 with the infeasible all-zero reference: proven infeasible.
        res = solve_with_reference(trunk_tree(), 3, ZERO_REF, 0)
        self.assertEqual(res.status, Status.INFEASIBLE)
        self.assertFalse(res.pruning.truncated)
        self.assertIsNone(res.total_cost)
        self.assertEqual(res.modifications, ())

    def test_exactly_k_changes(self):
        # Windows demand both branch buffers (a=9, b=12): exactly 2 changes.
        t = ClockTree(
            "r",
            [Edge("r", "u", 5, [Option(0, 0), Option(2, 4)]),
             Edge("u", "a", 1, [Option(0, 0), Option(3, 2)]),
             Edge("u", "b", 4, [Option(0, 0), Option(3, 2)])],
            [LeafWindow("a", 9, 9), LeafWindow("b", 12, 12)])
        res = assert_matches_reference(self, t, 3, ZERO_REF, 2)
        self.assertEqual(res.n_changes, 2)
        self.assertEqual(res.choices, {"u": 0, "a": 1, "b": 1})
        # K=1 cannot reach the windows (trunk +2 gives 8/11): infeasible.
        res1 = solve_with_reference(t, 3, ZERO_REF, 1)
        self.assertEqual(res1.status, Status.INFEASIBLE)

    def test_frozen_edge_causes_infeasible(self):
        # Weld the trunk to the zero option; branch changes alone cannot
        # hit windows 8/11, so the freeze makes the instance infeasible.
        res = solve_with_reference(trunk_tree(), 3, ZERO_REF, 2,
                                   frozen={"u": 0})
        self.assertEqual(res.status, Status.INFEASIBLE)
        self.assertFalse(res.pruning.truncated)

    def test_frozen_edge_not_pruned_and_not_counted(self):
        # Freeze trunk to the buffered option (index 1), which differs from
        # the reference; with K=0 the frozen override is NOT a change.
        res = solve_with_reference(trunk_tree(), 3, ZERO_REF, 0,
                                   frozen={"u": 1})
        self.assertEqual(res.status, Status.OPTIMAL)
        self.assertEqual(res.choices["u"], 1)
        self.assertEqual(res.n_changes, 0)
        self.assertEqual(res.modifications, ())

    def test_equivalent_options_different_index(self):
        # Options 0 and 1 on edge 'a' are equivalent FOR THIS INSTANCE:
        # same cost, and the global max delay is 5 either way (supplied by
        # leaf 'c').  The reference uses index 1; K=0 must keep exactly
        # that index even though index 0 scores identically.
        t = ClockTree(
            "r",
            [Edge("r", "a", 0, [Option(2, 1), Option(4, 1)]),
             Edge("r", "c", 5, [Option(0, 0)])],
            [LeafWindow("a", 0, 9), LeafWindow("c", 5, 5)])
        ref = {"a": 1, "c": 0}
        res = solve_with_reference(t, 5, ref, 0)
        self.assertEqual(res.status, Status.OPTIMAL)
        self.assertEqual(res.choices, {"a": 1, "c": 0})
        self.assertEqual(res.n_changes, 0)
        # Unconstrained solve picks index 0 via the option-order tie-break.
        self.assertEqual(solve(t, 5).choices["a"], 0)
        # K=1 lets the objective switch to the equivalent smaller index.
        res1 = assert_matches_reference(self, t, 5, ref, 1)
        self.assertEqual(res1.choices, {"a": 0, "c": 0})
        self.assertEqual(res1.n_changes, 1)

    def test_change_count_not_in_objective(self):
        # Two feasible optima on (cost, M, vec): the one needing FEWER
        # changes must not win on that account; objective order decides.
        t = ClockTree(
            "r",
            [Edge("r", "a", 0, [Option(0, 0), Option(2, 5)]),
             Edge("r", "b", 0, [Option(0, 0), Option(2, 5)])],
            [LeafWindow("a", 0, 9), LeafWindow("b", 0, 9)])
        ref = {"a": 1, "b": 1}  # reference itself is feasible, cost 10
        res = solve_with_reference(t, 5, ref, 2)
        # zero-option plan (cost 0) wins on cost despite 2 changes
        self.assertEqual(res.choices, {"a": 0, "b": 0})
        self.assertEqual(res.n_changes, 2)
        self.assertEqual(res.total_cost, 0)


class ReferenceValidationTests(unittest.TestCase):
    def setUp(self):
        self.tree = trunk_tree()

    def test_reference_must_cover_every_edge(self):
        with self.assertRaises(InputValueError):
            solve_with_reference(self.tree, 3, {"u": 0, "a": 0}, 1)

    def test_reference_index_range(self):
        with self.assertRaises(InputValueError):
            solve_with_reference(self.tree, 3, {"u": 2, "a": 0, "b": 0}, 1)
        with self.assertRaises(InputValueError):
            solve_with_reference(self.tree, 3, {"u": -1, "a": 0, "b": 0}, 1)

    def test_reference_types(self):
        with self.assertRaises(InputTypeError):
            solve_with_reference(self.tree, 3, [0, 0, 0], 1)
        with self.assertRaises(InputTypeError):
            solve_with_reference(self.tree, 3,
                                 {"u": True, "a": 0, "b": 0}, 1)
        with self.assertRaises(InputValueError):
            solve_with_reference(self.tree, 3,
                                 {"u": 0, "a": 0, "b": 0, "zz": 0}, 1)

    def test_frozen_validation(self):
        with self.assertRaises(InputValueError):
            solve_with_reference(self.tree, 3, ZERO_REF, 1,
                                 frozen={"u": 5})
        with self.assertRaises(InputValueError):
            solve_with_reference(self.tree, 3, ZERO_REF, 1,
                                 frozen={"nope": 0})
        with self.assertRaises(InputTypeError):
            solve_with_reference(self.tree, 3, ZERO_REF, 1,
                                 frozen={"u": 1.0})

    def test_max_changes_validation(self):
        with self.assertRaises(InputValueError):
            solve_with_reference(self.tree, 3, ZERO_REF, -1)
        with self.assertRaises(InputTypeError):
            solve_with_reference(self.tree, 3, ZERO_REF, True)

    def test_budget_exhaustion_distinct_from_infeasible(self):
        # Proven infeasible under constraints, no truncation involved.
        res = solve_with_reference(trunk_tree(), 3, ZERO_REF, 0, budget=1)
        self.assertEqual(res.status, Status.INFEASIBLE)
        self.assertEqual(res.pruning.budget_states, 0)
        # A cap of 1 on a multi-option instance forces a real cutoff.
        t = ClockTree(
            "r",
            [Edge("r", "a", 0, [Option(0, 0), Option(1, 2), Option(2, 4)]),
             Edge("r", "b", 0, [Option(0, 0), Option(1, 2), Option(2, 4)])],
            [LeafWindow("a", 0, 9), LeafWindow("b", 0, 9)])
        res2 = solve_with_reference(t, 5, {"a": 0, "b": 0}, 2, budget=1)
        self.assertEqual(res2.status, Status.BUDGET_EXHAUSTED)
        self.assertGreater(res2.pruning.budget_states, 0)


class RandomConstrainedComparisonTests(unittest.TestCase):
    def test_random_small_trees_with_reference_and_freeze(self):
        rng = random.Random(20260925)
        for trial in range(60):
            tree, S, ref, K, frozen = self._make(rng)
            with self.subTest(trial=trial):
                assert_matches_reference(self, tree, S, ref, K, frozen)

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
            n_opt = rng.randint(1, 3)
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
            windows.append(LeafWindow(lf, lo, lo + rng.randint(0, 6)))
        tree = ClockTree("r", edges, windows)
        S = rng.randint(0, 6)
        ref = {e.child: rng.randrange(len(e.options)) for e in edges}
        K = rng.randint(0, n_edges)
        frozen = {}
        for e in edges:
            if rng.random() < 0.3:
                frozen[e.child] = rng.randrange(len(e.options))
        return tree, S, ref, K, frozen


if __name__ == "__main__":
    unittest.main()

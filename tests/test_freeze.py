"""Freeze / change-budget extension tests.

The reference enumerator below is independent: it enumerates every option
combination of the UNFROZEN edges only (frozen edges are pinned by
construction), scores each assignment directly from the input model, and
never imports the DP core.
"""

import os
import random
import sys
import unittest
from itertools import product

sys.path.insert(0, os.path.dirname(__file__))

from skewbudget import (  # noqa: E402
    ClockTree,
    Edge,
    InputTypeError,
    InputValueError,
    LeafWindow,
    Option,
    Status,
    solve,
)


def enumerate_constrained(tree, skew_limit, frozen=None, reference=None,
                          max_changes=None):
    """All feasible assignments honouring freeze pins and the K edit cap."""
    frozen = frozen or {}
    edges_by_child = {ch: e for e in tree.edges for ch in (e.child,)}
    labels = sorted(edges_by_child)
    ranges = []
    for ch in labels:
        if ch in frozen:
            ranges.append((frozen[ch],))
        else:
            ranges.append(range(len(edges_by_child[ch].options)))
    out = []
    for combo in product(*ranges):
        choice = dict(zip(labels, combo))
        if reference is not None:
            edits = sum(1 for ch in labels if choice[ch] != reference[ch])
            if edits > max_changes:
                continue
        arrivals = {}
        for leaf in tree.leaf_set:
            total = 0
            v = leaf
            while v != tree.root:
                e = edges_by_child[v]
                total += e.base_delay + e.options[choice[v]].delay
                v = e.parent
            arrivals[leaf] = total
        lo = min(arrivals.values())
        hi = max(arrivals.values())
        if hi - lo > skew_limit:
            continue
        if all(w.lo <= arrivals[w.leaf] <= w.hi for w in tree.windows):
            cost = sum(edges_by_child[ch].options[oi].cost
                       for ch, oi in choice.items())
            out.append((choice, arrivals, cost, hi, lo))
    return out


def optimal_constrained(tree, skew_limit, frozen=None, reference=None,
                        max_changes=None):
    feasible = enumerate_constrained(tree, skew_limit, frozen,
                                     reference, max_changes)
    if not feasible:
        return None
    order = [ch for ch, _ in
             sorted(tree.edge_index.items(), key=lambda kv: kv[1])]

    def key(item):
        choice, _arr, cost, hi, _lo = item
        return (cost, hi, tuple(choice[ch] for ch in order))

    return min(feasible, key=key)


def assert_matches(testcase, tree, S, frozen=None, reference=None,
                   max_changes=None, budget=None):
    ref = optimal_constrained(tree, S, frozen, reference, max_changes)
    res = solve(tree, S, budget=budget, frozen=frozen,
                reference=reference, max_changes=max_changes)
    if ref is None:
        testcase.assertIn(res.status,
                          (Status.INFEASIBLE, Status.BUDGET_EXHAUSTED))
        testcase.assertIsNone(res.total_cost)
        return res
    testcase.assertEqual(res.status, Status.OPTIMAL)
    choice, arrivals, cost, hi, lo = ref
    testcase.assertEqual(res.total_cost, cost)
    testcase.assertEqual(res.max_delay, hi)
    testcase.assertEqual(res.min_delay, lo)
    testcase.assertEqual(res.leaf_arrivals, arrivals)
    testcase.assertEqual(res.choices, choice)
    if frozen:
        for ch, idx in frozen.items():
            testcase.assertEqual(res.choices[ch], idx)
    if reference is not None:
        edits = {ch: (reference[ch], choice[ch])
                 for ch in choice if choice[ch] != reference[ch]}
        testcase.assertEqual(res.changes, len(edits))
        testcase.assertEqual(res.modifications, edits)
        testcase.assertLessEqual(res.changes, max_changes)
    return res


def trunk_tree():
    #        r
    #        | base 5, trunk options (0,0) | (2,4)
    #        u
    #       / \
    #   a b1   b b4   branch options (0,0) | (3,2)
    return ClockTree(
        "r",
        [Edge("r", "u", 5, [Option(0, 0), Option(2, 4)]),
         Edge("u", "a", 1, [Option(0, 0), Option(3, 2)]),
         Edge("u", "b", 4, [Option(0, 0), Option(3, 2)])],
        [LeafWindow("a", 0, 12), LeafWindow("b", 0, 12)],
    )


class FreezeTests(unittest.TestCase):
    def test_shared_trunk_buffer_fixes_two_leaves(self):
        # Freezing the shared trunk to its +2 buffer shifts both leaves to
        # 8/11; one +3 branch buffer on a then aligns both at 11 -- the
        # single trunk pin does the heavy lifting for two leaves at once.
        tree = trunk_tree()
        res = assert_matches(self, tree, 0, frozen={"u": 1})
        self.assertEqual(res.choices, {"u": 1, "a": 1, "b": 0})
        self.assertEqual(res.leaf_arrivals, {"a": 11, "b": 11})
        self.assertEqual(res.total_cost, 6)
        self.assertEqual(res.frozen, {"u": 1})
        # Per-leaf path sums reproduce along the reported edge chains.
        for leaf, chain in res.leaf_path_edges.items():
            total = sum(
                tree.parent_edge[c].base_delay
                + tree.parent_edge[c].options[res.choices[c]].delay
                for c in chain)
            self.assertEqual(total, res.leaf_arrivals[leaf])

    def test_frozen_edge_cannot_be_pruned_away(self):
        # The unconstrained optimum uses trunk option 0; freezing trunk to
        # option 1 must survive every Pareto/budget prune.
        tree = trunk_tree()
        plain = solve(tree, 3)
        self.assertEqual(plain.choices["u"], 0)
        res = solve(tree, 3, frozen={"u": 1}, budget=1)
        self.assertEqual(res.choices["u"], 1)

    def test_freeze_makes_instance_infeasible(self):
        # Trunk pinned to +2 pushes leaf b to 11 > hi=10: proven infeasible,
        # not a budget artifact.
        tree = ClockTree(
            "r",
            [Edge("r", "u", 5, [Option(0, 0), Option(2, 4)]),
             Edge("u", "a", 1, [Option(0, 0)]),
             Edge("u", "b", 4, [Option(0, 0)])],
            [LeafWindow("a", 0, 10), LeafWindow("b", 0, 10)],
        )
        res = solve(tree, 10, frozen={"u": 1})
        self.assertIs(res.status, Status.INFEASIBLE)
        self.assertIsNone(res.total_cost)
        self.assertFalse(res.pruning.truncated)
        # Unfrozen, the same tree is feasible.
        self.assertIs(solve(tree, 10).status, Status.OPTIMAL)


class ChangeBudgetTests(unittest.TestCase):
    def test_exactly_k_changes(self):
        # Reference = all zero buffers.  S=0 needs exactly one branch
        # buffer, so K=1 is tight: K=0 infeasible, K=1 optimal with
        # exactly one modification recorded.
        tree = trunk_tree()
        reference = {"u": 0, "a": 0, "b": 0}
        res0 = solve(tree, 0, reference=reference, max_changes=0)
        self.assertIs(res0.status, Status.INFEASIBLE)
        res = assert_matches(self, tree, 0, reference=reference,
                             max_changes=1)
        self.assertEqual(res.changes, 1)
        self.assertEqual(len(res.modifications), 1)
        (edge, (ref_idx, new_idx)), = res.modifications.items()
        self.assertIn(edge, ("a", "b"))
        self.assertEqual((ref_idx, new_idx), (0, 1))
        self.assertEqual(res.max_changes, 1)
        self.assertEqual(res.reference, reference)

    def test_k0_validates_reference_plan(self):
        tree = trunk_tree()
        # A feasible reference: trunk buffer + branch a buffer, S=0, 11/11.
        good = {"u": 1, "a": 1, "b": 0}
        res = solve(tree, 0, reference=good, max_changes=0)
        self.assertIs(res.status, Status.OPTIMAL)
        self.assertEqual(res.choices, good)
        self.assertEqual(res.changes, 0)
        self.assertEqual(res.modifications, {})
        # An infeasible reference (skew 3 > S=0) is reported infeasible.
        bad = {"u": 0, "a": 0, "b": 0}
        res_bad = solve(tree, 0, reference=bad, max_changes=0)
        self.assertIs(res_bad.status, Status.INFEASIBLE)

    def test_equivalent_options_different_indices(self):
        # Options 0 and 1 on edge a are equivalent under the objective:
        # same cost, and leaf b always supplies the global max delay, so
        # both give (cost 0, M 10).  Free choice picks the lower index;
        # K=0 must return the reference's higher index verbatim.
        tree = ClockTree(
            "r",
            [Edge("r", "a", 5, [Option(0, 0), Option(1, 0)]),
             Edge("r", "b", 10, [Option(0, 0)])],
            [LeafWindow("a", 0, 10), LeafWindow("b", 0, 10)],
        )
        plain = solve(tree, 10)
        self.assertEqual(plain.choices["a"], 0)
        ref = {"a": 1, "b": 0}
        res = solve(tree, 10, reference=ref, max_changes=0)
        self.assertIs(res.status, Status.OPTIMAL)
        self.assertEqual(res.choices, ref)
        self.assertEqual(res.changes, 0)
        # With K=1 the solver is free to move to the lexicographically
        # smaller equivalent index; the edit count stays a constraint,
        # never part of the objective.
        res1 = solve(tree, 10, reference=ref, max_changes=1)
        self.assertEqual(res1.choices["a"], 0)
        self.assertEqual(res1.changes, 1)
        self.assertEqual(res1.modifications, {"a": (1, 0)})
        self.assertEqual(res1.total_cost, plain.total_cost)
        self.assertEqual(res1.max_delay, plain.max_delay)

    def test_change_count_not_in_objective(self):
        # A cheaper plan exists 2 edits away; with K=1 the solver must
        # return the best 1-edit plan, not mix edit count into cost.
        tree = ClockTree(
            "r",
            [Edge("r", "a", 0, [Option(0, 5), Option(1, 0)]),
             Edge("r", "b", 0, [Option(0, 5), Option(1, 0)])],
            [LeafWindow("a", 0, 9), LeafWindow("b", 0, 9)],
        )
        reference = {"a": 0, "b": 0}
        res = solve(tree, 9, reference=reference, max_changes=1)
        self.assertEqual(res.total_cost, 5)   # one edge upgraded
        self.assertEqual(res.changes, 1)
        res2 = solve(tree, 9, reference=reference, max_changes=2)
        self.assertEqual(res2.total_cost, 0)
        self.assertEqual(res2.changes, 2)


class ChangeBudgetValidationTests(unittest.TestCase):
    def setUp(self):
        self.tree = trunk_tree()

    def test_frozen_unknown_edge(self):
        with self.assertRaises(InputValueError):
            solve(self.tree, 3, frozen={"nope": 0})

    def test_frozen_index_out_of_range(self):
        with self.assertRaises(InputValueError):
            solve(self.tree, 3, frozen={"u": 2})
        with self.assertRaises(InputValueError):
            solve(self.tree, 3, frozen={"u": -1})

    def test_frozen_bool_rejected(self):
        with self.assertRaises(InputTypeError):
            solve(self.tree, 3, frozen={"u": True})

    def test_reference_must_cover_every_edge(self):
        with self.assertRaises(InputValueError):
            solve(self.tree, 3, reference={"u": 0}, max_changes=1)
        with self.assertRaises(InputValueError):
            solve(self.tree, 3,
                  reference={"u": 0, "a": 0, "b": 0, "zz": 0},
                  max_changes=1)

    def test_reference_index_out_of_range(self):
        with self.assertRaises(InputValueError):
            solve(self.tree, 3,
                  reference={"u": 9, "a": 0, "b": 0}, max_changes=1)

    def test_reference_requires_k_and_vice_versa(self):
        with self.assertRaises(InputValueError):
            solve(self.tree, 3, reference={"u": 0, "a": 0, "b": 0})
        with self.assertRaises(InputValueError):
            solve(self.tree, 3, max_changes=1)
        with self.assertRaises(InputValueError):
            solve(self.tree, 3, reference={"u": 0, "a": 0, "b": 0},
                  max_changes=-1)

    def test_budget_exhaustion_kept_apart_from_infeasible(self):
        # A frontier cap of 1 forces real cutoffs under a freeze pin; the
        # status must be BUDGET_EXHAUSTED, never INFEASIBLE.
        tree = ClockTree(
            "r",
            [Edge("r", "a", 0, [Option(0, 0), Option(1, 2), Option(2, 4)]),
             Edge("r", "b", 0, [Option(0, 0), Option(1, 2), Option(2, 4)])],
            [LeafWindow("a", 0, 9), LeafWindow("b", 0, 9)],
        )
        res = solve(tree, 5, budget=1, frozen={"a": 2},
                    reference={"a": 0, "b": 0}, max_changes=2)
        self.assertIs(res.status, Status.BUDGET_EXHAUSTED)
        if res.total_cost is not None:
            self.assertEqual(res.choices["a"], 2)


class RandomizedCrossCheck(unittest.TestCase):
    def test_random_small_trees_against_enumeration(self):
        rng = random.Random(20260925)
        for trial in range(40):
            n_leaves = rng.randint(2, 3)
            edges = [Edge("r", "u", rng.randint(0, 4),
                          [Option(rng.randint(0, 3), rng.randint(0, 4))
                           for _ in range(rng.randint(1, 3))])]
            windows = []
            for i in range(n_leaves):
                leaf = f"L{i}"
                edges.append(Edge(
                    "u", leaf, rng.randint(0, 4),
                    [Option(rng.randint(0, 3), rng.randint(0, 4))
                     for _ in range(rng.randint(1, 3))]))
                lo = rng.randint(0, 6)
                windows.append(LeafWindow(leaf, lo, lo + rng.randint(0, 8)))
            tree = ClockTree("r", edges, windows)
            labels = sorted(tree.parent_edge)
            frozen = {ch: rng.randrange(len(tree.parent_edge[ch].options))
                      for ch in labels if rng.random() < 0.4}
            reference = {ch: rng.randrange(len(tree.parent_edge[ch].options))
                         for ch in labels}
            k = rng.randint(0, len(labels))
            s = rng.randint(0, 8)
            assert_matches(self, tree, s, frozen=frozen,
                           reference=reference, max_changes=k)


if __name__ == "__main__":
    unittest.main()

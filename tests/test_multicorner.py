"""Multi-corner solver tests.

Correctness is cross-checked two ways:

* hand-computed examples whose expected numbers are worked out in the test
  itself (never produced by the solver);
* the independent exhaustive enumerator in tests/brute.py
  (``optimal_multicorner_reference`` / ``enumerate_multicorner``), which
  only reads the input model and ``itertools.product``.
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from brute import enumerate_multicorner, optimal_multicorner_reference  # noqa

from skewbudget import (  # noqa: E402
    Corner,
    CornerReport,
    InputTypeError,
    InputValueError,
    LeafWindow,
    MAX_CHOICE_POINTS,
    MAX_CORNERS,
    MAX_MC_OPTIONS,
    MultiCornerStatus,
    MultiCornerTree,
    PhysicalEdge,
    PhysicalOption,
    TreeValidationError,
    solve_multicorner,
    validate_multicorner_tree,
)


def assert_matches_reference(testcase, tree):
    ref = optimal_multicorner_reference(tree)
    res = solve_multicorner(tree)
    entries, n_total = enumerate_multicorner(tree)
    testcase.assertEqual(res.assignments_enumerated, n_total)
    if ref is None:
        testcase.assertEqual(res.status, MultiCornerStatus.INFEASIBLE)
        testcase.assertIsNone(res.total_cost)
        testcase.assertEqual(res.feasible_assignments, 0)
        return res
    testcase.assertEqual(res.status, MultiCornerStatus.OPTIMAL)
    ref_picks, ref_cost, ref_corner = ref
    testcase.assertEqual(res.total_cost, ref_cost)
    testcase.assertEqual(res.choices, ref_picks)
    testcase.assertEqual(res.feasible_assignments, len(entries))
    testcase.assertEqual(len(res.corners), len(tree.corners))
    for rpt, (arrivals, lo, hi, skew, window_ok) in zip(
            res.corners, ref_corner):
        testcase.assertEqual(rpt.leaf_arrivals, arrivals)
        testcase.assertEqual((rpt.min_delay, rpt.max_delay), (lo, hi))
        testcase.assertEqual(rpt.skew, skew)
        testcase.assertEqual(rpt.window_ok, window_ok)
        testcase.assertTrue(rpt.feasible)
        testcase.assertTrue(rpt.skew_ok)
    return res


class HandComputedTests(unittest.TestCase):
    def _tree(self, include_shared):
        if include_shared:
            opts = [
                PhysicalOption(0, (0, 0)),
                PhysicalOption(3, (2, 0)),
                PhysicalOption(7, (2, 4)),
            ]
        else:
            opts = [
                PhysicalOption(0, (0, 0)),
                PhysicalOption(3, (2, 0)),
                PhysicalOption(3, (0, 4)),
            ]
        return MultiCornerTree(
            "r",
            [PhysicalEdge("r", "a", (0, 0), opts),
             PhysicalEdge("r", "b", (0, 0), opts)],
            [Corner("fast", 0, [LeafWindow("a", 2, 2), LeafWindow("b", 2, 2)]),
             Corner("slow", 0, [LeafWindow("a", 4, 4), LeafWindow("b", 4, 4)])],
        )

    def test_shared_physical_option_forces_expensive_choice(self):
        # Hand analysis: fast needs corner-0 delay 2 on BOTH edges -> each
        # edge must use option 1 or 2 (idx 2 has delays (2,4)); slow needs
        # corner-1 delay 4 on BOTH edges -> only option 2 qualifies.
        # Hence option 2 is forced on both edges: physical cost 7+7 = 14.
        # Buying per-corner buffers (option 1 on both for fast, a different
        # one for slow) is impossible because the physical pick is shared.
        t = self._tree(include_shared=True)
        res = solve_multicorner(t)
        self.assertEqual(res.status, MultiCornerStatus.OPTIMAL)
        self.assertEqual(res.choices, {"a": 2, "b": 2})
        self.assertEqual(res.total_cost, 14)
        self.assertEqual(res.assignments_enumerated, 9)
        self.assertEqual(res.feasible_assignments, 1)
        fast, slow = res.corners
        self.assertEqual(fast.name, "fast")
        self.assertEqual(fast.leaf_arrivals, {"a": 2, "b": 2})
        self.assertEqual((fast.min_delay, fast.max_delay, fast.skew), (2, 2, 0))
        self.assertEqual(slow.leaf_arrivals, {"a": 4, "b": 4})
        self.assertEqual((slow.min_delay, slow.max_delay, slow.skew),
                         (4, 4, 0))
        self.assertTrue(all(fast.window_ok.values()))
        self.assertTrue(all(slow.window_ok.values()))
        # corner lookup by name
        self.assertIs(res.corner("fast"), fast)
        with self.assertRaises(KeyError):
            res.corner("nope")

    def test_corner_specific_buffers_cannot_be_split(self):
        # Without the (2,4) option: fast wants option 1 on each edge, slow
        # wants an option with corner-1 delay 4 (idx 2 has corner-0 delay 0)
        # -- no single vector meets both corners: proven infeasible.
        t = self._tree(include_shared=False)
        res = solve_multicorner(t)
        self.assertEqual(res.status, MultiCornerStatus.INFEASIBLE)
        self.assertIsNone(res.total_cost)
        self.assertEqual(res.choices, {})
        self.assertEqual(res.corners, ())
        self.assertEqual(res.assignments_enumerated, 9)
        self.assertEqual(res.feasible_assignments, 0)
        self.assertFalse(res.proven_optimal)

    def test_corner_relative_base_delays_count(self):
        # Corner-specific BASE delays and per-corner option delays both
        # apply.  fast bases a=1,b=3 (gap 2); slow bases a=2,b=0 (gap 2).
        # Option 1 on a adds (2,0): fast arrivals 3/3 (option closes the
        # gap, skew 0); slow arrivals 2/0 (option adds nothing, the base
        # gap shows, skew 2 == closed bound S=2).
        opts_a = [PhysicalOption(0, (0, 0)), PhysicalOption(4, (2, 0))]
        opts_b = [PhysicalOption(0, (0, 0))]
        t = MultiCornerTree(
            "r",
            [PhysicalEdge("r", "a", (1, 2), opts_a),
             PhysicalEdge("r", "b", (3, 0), opts_b)],
            [Corner("fast", 0, [LeafWindow("a", 0, 9), LeafWindow("b", 0, 9)]),
             Corner("slow", 2, [LeafWindow("a", 0, 9), LeafWindow("b", 0, 9)])],
        )
        res = solve_multicorner(t)
        self.assertEqual(res.status, MultiCornerStatus.OPTIMAL)
        self.assertEqual(res.total_cost, 4)
        self.assertEqual(res.choices, {"a": 1, "b": 0})
        fast, slow = res.corners
        self.assertEqual(fast.leaf_arrivals, {"a": 3, "b": 3})
        self.assertEqual(slow.leaf_arrivals, {"a": 2, "b": 0})
        self.assertEqual(slow.skew, 2)  # closed bound: == S is feasible

    def test_skew_violated_in_one_corner_only(self):
        # fast base gap a/b = 5, S_fast = 0; slow bases equal, S=4.
        # a: (0,0) cost 0 or (5,5) cost 4 -> fast forces option 1, which
        # makes slow arrivals a=5,b=0: slow skew 5 > 4 -> no vector works.
        t = MultiCornerTree(
            "r",
            [PhysicalEdge("r", "a", (0, 0),
                          [PhysicalOption(0, (0, 0)),
                           PhysicalOption(4, (5, 5))]),
             PhysicalEdge("r", "b", (5, 0), [PhysicalOption(0, (0, 0))])],
            [Corner("fast", 0, [LeafWindow("a", 0, 99), LeafWindow("b", 0, 99)]),
             Corner("slow", 4, [LeafWindow("a", 0, 99), LeafWindow("b", 0, 99)])],
        )
        res = solve_multicorner(t)
        self.assertEqual(res.status, MultiCornerStatus.INFEASIBLE)
        # widening the slow bound to 5 makes the same vector feasible
        t2 = MultiCornerTree(
            "r", t.edges,
            [t.corners[0],
             Corner("slow", 5, [LeafWindow("a", 0, 99),
                                LeafWindow("b", 0, 99)])],
        )
        res2 = solve_multicorner(t2)
        self.assertEqual(res2.status, MultiCornerStatus.OPTIMAL)
        self.assertEqual(res2.total_cost, 4)

    def test_closed_window_endpoints(self):
        # single leaf, single corner: delay 1 is cheaper but window [0,0]
        # excludes it; arrival exactly hi (0) binds.
        t = MultiCornerTree(
            "r",
            [PhysicalEdge("r", "a", (0,),
                          [PhysicalOption(5, (0,)),
                           PhysicalOption(1, (1,))])],
            [Corner("only", 0, [LeafWindow("a", 0, 0)])],
        )
        res = solve_multicorner(t)
        self.assertEqual(res.status, MultiCornerStatus.OPTIMAL)
        self.assertEqual(res.total_cost, 5)
        self.assertEqual(res.choices, {"a": 0})
        rpt = res.corners[0]
        self.assertEqual(rpt.leaf_arrivals, {"a": 0})
        self.assertEqual(rpt.window_ok, {"a": True})

    def test_cost_tie_breaks_on_option_vector(self):
        # Two equally cheap feasible options (0,0) and (2,2), both cost 1;
        # the canonical vector tie-break must pick the smaller index.
        t = MultiCornerTree(
            "r",
            [PhysicalEdge("r", "a", (0, 0),
                          [PhysicalOption(1, (0, 0)),
                           PhysicalOption(1, (2, 2))])],
            [Corner("c1", 3, [LeafWindow("a", 0, 9)]),
             Corner("c2", 3, [LeafWindow("a", 0, 9)])],
        )
        res = solve_multicorner(t)
        self.assertEqual(res.total_cost, 1)
        self.assertEqual(res.choices, {"a": 0})
        all_res = solve_multicorner(t, return_all=True)
        self.assertEqual(len(all_res.all_optimal_choices), 2)
        self.assertEqual(all_res.all_optimal_choices[0], {"a": 0})
        self.assertEqual(all_res.all_optimal_choices[1], {"a": 1})

    def test_return_all_default_empty(self):
        t = MultiCornerTree(
            "r",
            [PhysicalEdge("r", "a", (0,), [PhysicalOption(0, (0,))])],
            [Corner("c", 0, [LeafWindow("a", 0, 1)])],
        )
        self.assertEqual(solve_multicorner(t).all_optimal_choices, ())

    def test_lone_root(self):
        t = MultiCornerTree(
            "r", [],
            [Corner("c1", 0, []), Corner("c2", 1, [])],
        )
        res = solve_multicorner(t)
        self.assertEqual(res.status, MultiCornerStatus.OPTIMAL)
        self.assertEqual(res.total_cost, 0)
        self.assertEqual(res.choices, {})
        self.assertEqual(res.assignments_enumerated, 1)
        self.assertEqual(res.feasible_assignments, 1)
        for rpt in res.corners:
            self.assertTrue(rpt.feasible)
            self.assertEqual(rpt.leaf_arrivals, {})
            self.assertEqual(rpt.skew, 0)

    def test_forced_single_option_edges_are_not_choice_points(self):
        forced = PhysicalEdge("r", "a", (0,), [PhysicalOption(2, (1,))])
        choice = PhysicalEdge(
            "r", "b", (0,),
            [PhysicalOption(0, (0,)), PhysicalOption(1, (2,))])
        t = MultiCornerTree(
            "r", [forced, choice],
            [Corner("c", 5, [LeafWindow("a", 0, 9), LeafWindow("b", 0, 9)])],
        )
        self.assertEqual(t.choice_points, 1)
        self.assertEqual(t.choice_edge_order, ("b",))
        res = solve_multicorner(t)
        # forced option still contributes its cost and per-corner delay;
        # cheapest feasible b is option 0
        self.assertEqual(res.choices, {"a": 0, "b": 0})
        self.assertEqual(res.total_cost, 2)
        self.assertEqual(res.assignments_enumerated, 2)
        self.assertEqual(res.corners[0].leaf_arrivals,
                         {"a": 1, "b": 0})


class RandomComparisonTests(unittest.TestCase):
    def test_random_small_trees_against_enumeration(self):
        rng = random.Random(20260924)
        for trial in range(50):
            tree = self._make(rng, trial)
            with self.subTest(trial=trial):
                assert_matches_reference(self, tree)

    @staticmethod
    def _make(rng, trial):
        n_corners = rng.randint(1, 3)
        n_edges = rng.randint(1, 5)
        labels = ["r"]
        edges = []
        next_id = 0
        while len(edges) < n_edges:
            parent = rng.choice(labels)
            child = f"n{next_id}"
            next_id += 1
            base = [rng.randint(0, 3) for _ in range(n_corners)]
            n_opts = rng.randint(1, MAX_MC_OPTIONS)
            seen = set()
            opts = []
            while len(opts) < n_opts:
                vec = tuple(rng.randint(0, 4) for _ in range(n_corners))
                if vec in seen:
                    continue
                seen.add(vec)
                opts.append(PhysicalOption(rng.randint(0, 5), list(vec)))
            edges.append(PhysicalEdge(parent, child, base, opts))
            labels.append(child)
        children = {}
        for e in edges:
            children.setdefault(e.parent, []).append(e.child)
        leaves = [x for x in labels if x not in children]
        corners = []
        for k in range(n_corners):
            windows = []
            for lf in leaves:
                lo = rng.randint(0, 7)
                windows.append(LeafWindow(lf, lo, lo + rng.randint(0, 6)))
            corners.append(Corner(f"corner{k}", rng.randint(0, 5), windows))
        return MultiCornerTree("r", edges, corners)


class ValidationTests(unittest.TestCase):
    def test_physical_option_checks(self):
        with self.assertRaises(InputValueError):
            PhysicalOption(-1, (0,))
        with self.assertRaises(InputValueError):
            PhysicalOption(0, (-1,))
        with self.assertRaises(InputTypeError):
            PhysicalOption(True, (0,))
        with self.assertRaises(InputTypeError):
            PhysicalOption(0, (1.0,))
        with self.assertRaises(InputTypeError):
            PhysicalOption(0, 0)
        with self.assertRaises(InputValueError):
            PhysicalOption(0, ())
        with self.assertRaises(InputValueError):
            PhysicalOption(0, tuple(range(MAX_CORNERS + 1)))

    def test_physical_edge_checks(self):
        opt = PhysicalOption(0, (0, 0))
        with self.assertRaises(InputValueError):
            PhysicalEdge("r", "a", (0, 0), [])
        with self.assertRaises(InputValueError):
            PhysicalEdge(
                "r", "a", (0, 0),
                [PhysicalOption(0, (0, 0)) for _ in range(MAX_MC_OPTIONS + 1)])
        with self.assertRaises(InputValueError):
            PhysicalEdge("r", "a", (-1, 0), [opt])
        with self.assertRaises(InputValueError):
            PhysicalEdge("r", "a", (0,), [opt])  # corner count mismatch
        with self.assertRaises(InputTypeError):
            PhysicalEdge("r", "a", (0, 0), [1])
        with self.assertRaises(InputTypeError):
            PhysicalEdge("r", "a", 0, [opt])

    def test_corner_checks(self):
        with self.assertRaises(InputValueError):
            Corner("c", -1, [])
        with self.assertRaises(InputTypeError):
            Corner("c", False, [])
        with self.assertRaises(TreeValidationError):
            Corner("", 0, [])
        with self.assertRaises(InputValueError):
            Corner("c", 0,
                   [LeafWindow("a", 0, 1), LeafWindow("a", 0, 2)])
        with self.assertRaises(InputTypeError):
            Corner("c", 0, [LeafWindow("a", 0, 1.0)])

    def _ok_edge(self, parent="r", child="a", n_opts=2, n_corners=2):
        return PhysicalEdge(
            parent, child, tuple([0] * n_corners),
            [PhysicalOption(i, tuple([i] * n_corners))
             for i in range(n_opts)])

    def test_tree_corner_count_mismatch(self):
        with self.assertRaises(TreeValidationError):
            MultiCornerTree(
                "r", [self._ok_edge()],
                [Corner("only", 0, [LeafWindow("a", 0, 9)])])

    def test_no_corners(self):
        with self.assertRaises(InputValueError):
            MultiCornerTree("r", [self._ok_edge()], [])

    def test_too_many_corners(self):
        # Corner-count cap is enforced at tree level; an edgeless lone-root
        # tree with 9 corners trips it without any PhysicalOption (which is
        # itself capped at 8 per-corner entries).
        corners = [Corner(f"c{i}", 0, [])
                   for i in range(MAX_CORNERS + 1)]
        with self.assertRaises(InputValueError):
            MultiCornerTree("r", [], corners)

    def test_duplicate_corner_name(self):
        with self.assertRaises(TreeValidationError):
            MultiCornerTree(
                "r", [self._ok_edge()],
                [Corner("x", 0, [LeafWindow("a", 0, 9)]),
                 Corner("x", 0, [LeafWindow("a", 0, 9)])])

    def test_window_placement_per_corner(self):
        edge = self._ok_edge()
        corners_ok = [Corner("c", 0, [LeafWindow("a", 0, 9)])]
        with self.assertRaises(TreeValidationError):
            # missing window in the second corner
            MultiCornerTree("r", [edge],
                            [Corner("c1", 0, [LeafWindow("a", 0, 9)]),
                             Corner("c2", 0, [])])
        with self.assertRaises(TreeValidationError):
            # window on an unknown vertex in one corner
            MultiCornerTree("r", [edge],
                            [Corner("c1", 0, [LeafWindow("a", 0, 9)]),
                             Corner("c2", 0, [LeafWindow("zzz", 0, 9)])])

    def test_window_on_internal_vertex_rejected(self):
        edges = [self._ok_edge("r", "a"), self._ok_edge("a", "a1")]
        with self.assertRaises(TreeValidationError):
            MultiCornerTree(
                "r", edges,
                [Corner("c", 0,
                        [LeafWindow("a", 0, 9), LeafWindow("a1", 0, 9)])])

    def test_topology_errors_are_reused(self):
        # two parents, dangling edge, self-loop -- same semantics as the
        # single-corner model
        with self.assertRaises(TreeValidationError):
            MultiCornerTree(
                "r",
                [self._ok_edge("r", "a"),
                 self._ok_edge("a", "b"),
                 self._ok_edge("r", "b")],
                [Corner("c", 0,
                        [LeafWindow("a", 0, 9), LeafWindow("b", 0, 9)])])
        with self.assertRaises(TreeValidationError):
            MultiCornerTree(
                "r", [self._ok_edge("x", "a")],
                [Corner("c", 0, [LeafWindow("a", 0, 9)])])
        with self.assertRaises(TreeValidationError):
            MultiCornerTree(
                "r", [PhysicalEdge("a", "a", (0, 0),
                                   [PhysicalOption(0, (0, 0))])],
                [Corner("c", 0, [LeafWindow("a", 0, 9)])])

    def test_too_many_choice_points(self):
        edges = [self._ok_edge("r", f"a{i}", n_corners=1)
                 for i in range(MAX_CHOICE_POINTS + 1)]
        windows = [LeafWindow(f"a{i}", 0, 99) for i in range(len(edges))]
        with self.assertRaises(InputValueError) as cm:
            MultiCornerTree("r", edges, [Corner("c", 99, windows)])
        self.assertIn("choice points", str(cm.exception))

    def test_eight_choice_points_allowed_with_forced_extra(self):
        edges = [self._ok_edge("r", f"a{i}", n_corners=1)
                 for i in range(MAX_CHOICE_POINTS)]
        edges.append(PhysicalEdge(
            "r", "forced", (0,), [PhysicalOption(0, (0,))]))
        windows = [LeafWindow(f"a{i}", 0, 99) for i in
                   range(MAX_CHOICE_POINTS)] + [LeafWindow("forced", 0, 99)]
        t = MultiCornerTree("r", edges, [Corner("c", 99, windows)])
        self.assertEqual(t.choice_points, MAX_CHOICE_POINTS)
        res = solve_multicorner(t)
        self.assertEqual(res.status, MultiCornerStatus.OPTIMAL)
        self.assertEqual(res.assignments_enumerated,
                         2 ** MAX_CHOICE_POINTS)

    def test_identical_delay_vectors_dedup_cheapest(self):
        e = PhysicalEdge(
            "r", "a", (0, 0),
            [PhysicalOption(9, (2, 2)),
             PhysicalOption(3, (2, 2)),
             PhysicalOption(1, (0, 0))])
        self.assertEqual([(o.cost, o.delays) for o in e.options],
                         [(3, (2, 2)), (1, (0, 0))])

    def test_bad_containers_and_types(self):
        with self.assertRaises(InputTypeError):
            MultiCornerTree("r", self._ok_edge(),
                            [Corner("c", 0, [LeafWindow("a", 0, 9)])])
        with self.assertRaises(TreeValidationError):
            MultiCornerTree(1, [], [])

    def test_solver_argument_validation(self):
        t = MultiCornerTree(
            "r",
            [PhysicalEdge("r", "a", (0,), [PhysicalOption(0, (0,))])],
            [Corner("c", 0, [LeafWindow("a", 0, 1)])],
        )
        with self.assertRaises(InputTypeError):
            solve_multicorner("not-a-tree")
        with self.assertRaises(InputTypeError):
            solve_multicorner(t, return_all=1)

    def test_validate_summary(self):
        t = MultiCornerTree(
            "r",
            [PhysicalEdge("r", "a", (0, 0),
                          [PhysicalOption(0, (0, 0)),
                           PhysicalOption(1, (1, 1))])],
            [Corner("hot", 0, [LeafWindow("a", 0, 9)]),
             Corner("cold", 0, [LeafWindow("a", 0, 9)])],
        )
        self.assertEqual(validate_multicorner_tree(t), {
            "root": "r", "n_edges": 1, "n_corners": 2,
            "leaves": ("a",), "choice_points": 1,
            "corners": ("hot", "cold"),
        })
        with self.assertRaises(InputTypeError):
            validate_multicorner_tree(object())

    def test_failed_construction_leaves_no_half_state(self):
        good = MultiCornerTree(
            "r",
            [PhysicalEdge("r", "a", (0,), [PhysicalOption(0, (0,))])],
            [Corner("c", 0, [LeafWindow("a", 0, 1)])],
        )
        with self.assertRaises(TreeValidationError):
            MultiCornerTree(
                "r",
                [PhysicalEdge("r", "a", (0, 0),
                              [PhysicalOption(0, (0, 0))])],
                [Corner("c", 0, [LeafWindow("a", 0, 1)]),
                 Corner("d", 0, [])])
        # the previously built tree is untouched and still solves
        res = solve_multicorner(good)
        self.assertEqual(res.status, MultiCornerStatus.OPTIMAL)

    def test_corner_report_is_value_type(self):
        rpt = CornerReport("c", 1, True, {"a": 0}, 0, 0, 0, {"a": True})
        self.assertTrue(rpt.skew_ok)
        rpt2 = CornerReport("c", 0, False, {"a": 5}, 5, 5, 5, {"a": False})
        self.assertFalse(rpt2.skew_ok)


if __name__ == "__main__":
    unittest.main()

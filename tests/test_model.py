import unittest

from skewbudget import (
    ClockTree,
    Edge,
    InputTypeError,
    InputValueError,
    InvalidNumericError,
    LeafWindow,
    MAX_EDGES,
    Option,
    TreeValidationError,
)


def two_leaf(**kw):
    return ClockTree(
        "r",
        [
            Edge("r", "a", 1, [Option(0, 0)]),
            Edge("r", "b", 1, [Option(0, 0)]),
        ],
        [LeafWindow("a", 0, 5), LeafWindow("b", 0, 5)],
    )


class OptionTests(unittest.TestCase):
    def test_ok_zero_buffer(self):
        o = Option(0, 0)
        self.assertEqual((o.delay, o.cost), (0, 0))

    def test_bool_rejected(self):
        with self.assertRaises(InputTypeError):
            Option(True, 0)
        with self.assertRaises(InputTypeError):
            Option(0, False)

    def test_float_rejected_even_integral(self):
        with self.assertRaises(InputTypeError):
            Option(1.0, 0)

    def test_nan_infinity(self):
        with self.assertRaises(InvalidNumericError):
            Option(float("nan"), 0)
        with self.assertRaises(InvalidNumericError):
            Option(float("inf"), 0)
        with self.assertRaises(InvalidNumericError):
            Option(0, float("-inf"))

    def test_negative(self):
        with self.assertRaises(InputValueError):
            Option(-1, 0)
        with self.assertRaises(InputValueError):
            Option(0, -2)

    def test_wrong_type(self):
        with self.assertRaises(InputTypeError):
            Option("1", 0)
        with self.assertRaises(InputTypeError):
            Option(0, None)

    def test_frozen(self):
        with self.assertRaises(AttributeError):
            Option(0, 0).delay = 3

    def test_duplicate_delay_keeps_cheapest_and_order(self):
        e = Edge("r", "a", 0,
                 [Option(2, 9), Option(2, 3), Option(5, 1), Option(2, 7)])
        self.assertEqual([(o.delay, o.cost) for o in e.options],
                         [(2, 3), (5, 1)])


class EdgeTests(unittest.TestCase):
    def test_no_options(self):
        with self.assertRaises(InputValueError):
            Edge("r", "a", 0, [])

    def test_too_many_options(self):
        opts = [Option(i, i) for i in range(65)]
        with self.assertRaises(InputValueError):
            Edge("r", "a", 0, opts)

    def test_bad_base_delay(self):
        with self.assertRaises(InputValueError):
            Edge("r", "a", -1, [Option(0, 0)])
        with self.assertRaises(InputTypeError):
            Edge("r", "a", True, [Option(0, 0)])

    def test_options_container_type(self):
        with self.assertRaises(InputTypeError):
            Edge("r", "a", 0, (Option(0, 0) for _ in range(1)))
        with self.assertRaises(InputTypeError):
            Edge("r", "a", 0, [1])


class WindowTests(unittest.TestCase):
    def test_lo_gt_hi(self):
        with self.assertRaises(InputValueError):
            LeafWindow("a", 5, 4)

    def test_closed_endpoints_allowed(self):
        w = LeafWindow("a", 3, 3)
        self.assertEqual((w.lo, w.hi), (3, 3))

    def test_bool_and_float(self):
        with self.assertRaises(InputTypeError):
            LeafWindow("a", True, 3)
        with self.assertRaises(InputTypeError):
            LeafWindow("a", 0, 3.0)


class TreeStructureTests(unittest.TestCase):
    def _edges(self, n):
        return [Edge("r", f"v{i}", 1, [Option(0, 0)]) for i in range(n)]

    def test_too_many_edges(self):
        with self.assertRaises(InputValueError):
            ClockTree("r", self._edges(MAX_EDGES + 1), ())

    def test_missing_leaf_window(self):
        with self.assertRaises(TreeValidationError) as cm:
            ClockTree(
                "r",
                [Edge("r", "a", 0, [Option(0, 0)]),
                 Edge("r", "b", 0, [Option(0, 0)])],
                [LeafWindow("a", 0, 1)],
            )
        self.assertIn("without arrival window", str(cm.exception))
        self.assertEqual(cm.exception.path, "windows")

    def test_window_on_internal_vertex(self):
        with self.assertRaises(TreeValidationError):
            ClockTree(
                "r",
                [Edge("r", "a", 0, [Option(0, 0)]),
                 Edge("a", "a1", 0, [Option(0, 0)])],
                [LeafWindow("a", 0, 1), LeafWindow("a1", 0, 1)],
            )

    def test_window_unknown_label(self):
        with self.assertRaises(TreeValidationError):
            ClockTree(
                "r", [Edge("r", "a", 0, [Option(0, 0)])],
                [LeafWindow("a", 0, 1), LeafWindow("zzz", 0, 1)],
            )

    def test_duplicate_window(self):
        with self.assertRaises(TreeValidationError):
            ClockTree(
                "r", [Edge("r", "a", 0, [Option(0, 0)]),
                      Edge("r", "b", 0, [Option(0, 0)])],
                [LeafWindow("a", 0, 1), LeafWindow("a", 0, 2),
                 LeafWindow("b", 0, 1)],
            )

    def test_two_parents(self):
        with self.assertRaises(TreeValidationError) as cm:
            ClockTree(
                "r",
                [Edge("r", "a", 0, [Option(0, 0)]),
                 Edge("a", "b", 0, [Option(0, 0)]),
                 Edge("r", "b", 0, [Option(0, 0)])],
                [LeafWindow("a", 0, 1), LeafWindow("b", 0, 1)],
            )
        self.assertIn("two parents", str(cm.exception))

    def test_dangling_parent(self):
        with self.assertRaises(TreeValidationError) as cm:
            ClockTree(
                "r",
                [Edge("x", "a", 0, [Option(0, 0)])],
                [LeafWindow("a", 0, 1)],
            )
        self.assertIn("dangling", str(cm.exception))

    def test_self_loop(self):
        with self.assertRaises(TreeValidationError):
            ClockTree(
                "r", [Edge("a", "a", 0, [Option(0, 0)])],
                [LeafWindow("a", 0, 1)],
            )

    def test_disconnected_second_root(self):
        # Two edges, 3 non-root labels but one component rooted elsewhere:
        # the dangling check fires first; here build a 2-edge shape whose
        # label count does not match a tree.
        with self.assertRaises(TreeValidationError):
            ClockTree(
                "r",
                [Edge("x", "a", 0, [Option(0, 0)]),
                 Edge("r", "b", 0, [Option(0, 0)])],
                [LeafWindow("a", 0, 1), LeafWindow("b", 0, 1)],
            )

    def test_bad_root_label(self):
        with self.assertRaises(TreeValidationError):
            ClockTree("", [], [])
        with self.assertRaises(TreeValidationError):
            ClockTree(1, [], [])

    def test_bad_containers(self):
        with self.assertRaises(InputTypeError):
            ClockTree("r", Edge("r", "a", 0, [Option(0, 0)]), [])
        with self.assertRaises(InputTypeError):
            ClockTree("r", [], {LeafWindow("a", 0, 1)})

    def test_leaf_window_nan(self):
        with self.assertRaises(InvalidNumericError):
            LeafWindow("a", float("nan"), 1)

    def test_failed_construction_leaves_nothing_behind(self):
        # A failed construction raises; an independently built tree is
        # untouched (no shared mutable global state).
        t = two_leaf()
        with self.assertRaises(TreeValidationError):
            ClockTree(
                "r", [Edge("r", "a", 0, [Option(0, 0)]),
                      Edge("r", "b", 0, [Option(0, 0)])],
                [LeafWindow("a", 0, 1), LeafWindow("a", 0, 1),
                 LeafWindow("b", 0, 1)],
            )
        self.assertEqual(t.n_edges, 2)
        self.assertEqual(t.leaf_set, frozenset({"a", "b"}))


if __name__ == "__main__":
    unittest.main()

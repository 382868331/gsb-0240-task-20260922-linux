"""输入校验与错误语义测试。"""

import math
import unittest

from clocktree.model import (
    BufferOption,
    ClockTree,
    Edge,
    LeafWindow,
    InvalidSpecification,
    compile_layout,
)
from clocktree.solver import solve


def _edge(eid, p, c, base=0, options=()):
    return Edge(eid, p, c, base, tuple(options))


class ValidationTests(unittest.TestCase):
    def test_reject_nan_infinity_via_float(self):
        # 数值字段必须是 int：NaN/Infinity 作为 float 被直接拒绝。
        for bad in (math.nan, math.inf, -math.inf, 1.0, 0.5):
            with self.assertRaises(InvalidSpecification):
                BufferOption(bad, 1)
            with self.assertRaises(InvalidSpecification):
                BufferOption(1, bad)
            with self.assertRaises(InvalidSpecification):
                _edge("e", "r", "a", base=bad)

    def test_reject_bool_as_int(self):
        with self.assertRaises(InvalidSpecification):
            BufferOption(True, 1)
        with self.assertRaises(InvalidSpecification):
            _edge("e", "r", "a", base=False)
        with self.assertRaises(InvalidSpecification):
            ClockTree(
                (_edge("e", "r", "a"),),
                (LeafWindow("a", 0, 1),),
                True,
            )

    def test_reject_negative_and_empty_interval(self):
        with self.assertRaises(InvalidSpecification):
            BufferOption(-1, 0)
        with self.assertRaises(InvalidSpecification):
            LeafWindow("a", 5, 4)
        with self.assertRaises(InvalidSpecification):
            ClockTree((), (), 0)  # 空树
        with self.assertRaises(InvalidSpecification):
            ClockTree((_edge("e", "r", "a"),), (), -1)

    def test_reject_bad_identifiers_and_types(self):
        with self.assertRaises(InvalidSpecification):
            _edge("", "r", "a")
        with self.assertRaises(InvalidSpecification):
            _edge("e", "r", 1)  # type: ignore[arg-type]
        with self.assertRaises(InvalidSpecification):
            _edge("e", "r", "r")
        with self.assertRaises(InvalidSpecification):
            _edge("e", "r", "a", options=(("x", 1),))  # type: ignore[arg-type]

    def test_reject_graph_errors(self):
        # 重复边 id
        with self.assertRaises(InvalidSpecification):
            compile_layout(
                ClockTree(
                    (_edge("e", "r", "a"), _edge("e", "r", "b")),
                    (LeafWindow("a", 0, 9), LeafWindow("b", 0, 9)),
                    9,
                )
            )
        # 森林（两个根）
        with self.assertRaises(InvalidSpecification):
            compile_layout(
                ClockTree(
                    (_edge("e1", "r", "a"), _edge("e2", "q", "b")),
                    (LeafWindow("a", 0, 9), LeafWindow("b", 0, 9)),
                    9,
                )
            )
        # 环 a->b->a（两个无父节点 → 也无唯一根；且不连通）
        with self.assertRaises(InvalidSpecification):
            compile_layout(
                ClockTree(
                    (
                        _edge("e1", "a", "b"),
                        _edge("e2", "b", "c"),
                        _edge("e3", "c", "a"),
                    ),
                    (LeafWindow("b", 0, 9),),
                    9,
                )
            )
        # 同一节点两条入边
        with self.assertRaises(InvalidSpecification):
            compile_layout(
                ClockTree(
                    (_edge("e1", "r", "a"), _edge("e2", "r", "a")),
                    (LeafWindow("a", 0, 9),),
                    9,
                )
            )

    def test_window_coverage(self):
        edges = (_edge("e1", "r", "a"), _edge("e2", "r", "b"))
        # 缺叶子窗口
        with self.assertRaises(InvalidSpecification):
            compile_layout(ClockTree(edges, (LeafWindow("a", 0, 9),), 9))
        # 窗口挂内部节点
        with self.assertRaises(InvalidSpecification):
            compile_layout(
                ClockTree(
                    (_edge("e1", "r", "m"), _edge("e2", "m", "a")),
                    (
                        LeafWindow("m", 0, 9),
                        LeafWindow("a", 0, 9),
                    ),
                    9,
                )
            )
        # 窗口指向不存在节点 / 重复窗口
        with self.assertRaises(InvalidSpecification):
            compile_layout(
                ClockTree(edges, (LeafWindow("x", 0, 9), LeafWindow("b", 0, 9)), 9)
            )
        with self.assertRaises(InvalidSpecification):
            compile_layout(
                ClockTree(
                    edges,
                    (
                        LeafWindow("a", 0, 9),
                        LeafWindow("a", 1, 9),
                        LeafWindow("b", 0, 9),
                    ),
                    9,
                )
            )

    def test_too_many_edges(self):
        edges = tuple(_edge(f"e{i}", f"n{i}", f"n{i+1}") for i in range(26))
        with self.assertRaises(InvalidSpecification):
            ClockTree(edges, (LeafWindow("n26", 0, 99),), 99)
        # 恰好 25 条允许
        edges25 = tuple(_edge(f"e{i}", f"n{i}", f"n{i+1}") for i in range(25))
        layout = compile_layout(
            ClockTree(edges25, (LeafWindow("n25", 0, 99),), 99)
        )
        self.assertEqual(len(layout.edges), 25)

    def test_budget_params_validated(self):
        tree = ClockTree(
            (_edge("e", "r", "a"),), (LeafWindow("a", 0, 9),), 9
        )
        for kwargs in (
            {"node_vector_cap": 0},
            {"work_cap": 0},
            {"node_vector_cap": -1},
            {"node_vector_cap": True},
            {"work_cap": 1.0},
        ):
            with self.assertRaises(InvalidSpecification):
                solve(tree, **kwargs)

    def test_input_is_copied_and_frozen(self):
        opts = [BufferOption(1, 2)]
        e = _edge("e", "r", "a", options=opts)
        opts.append(BufferOption(9, 9))  # 调用方后续修改不影响已构造对象
        self.assertEqual(len(e.options), 1)
        with self.assertRaises(Exception):
            e.base_delay = 5  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()

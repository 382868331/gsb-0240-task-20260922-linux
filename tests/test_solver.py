"""求解器功能测试：与独立枚举参考交叉核对，并覆盖题面要求的场景。"""

import random
import unittest

from clocktree.model import (
    BufferOption,
    ClockTree,
    Edge,
    LeafWindow,
)
from clocktree.solver import (
    BUDGET_EXCEEDED,
    INFEASIBLE,
    OPTIMAL,
    solve,
)

from tests.brute_reference import brute_force


def _edge(eid, p, c, base=0, options=()):
    return Edge(eid, p, c, base, tuple(options))


def _assert_matches_brute(testcase: unittest.TestCase, tree: ClockTree) -> None:
    """求解器结果必须与独立枚举参考逐字段一致（或同时判无解）。"""
    sol = solve(tree)
    ref = brute_force(tree)
    if ref is None:
        testcase.assertEqual(
            sol.status,
            INFEASIBLE,
            msg=f"枚举判无解但求解器状态={sol.status}",
        )
        testcase.assertIsNone(sol.choices)
        return

    testcase.assertEqual(
        sol.status,
        OPTIMAL,
        msg=f"小实例不应触发预算截断，状态={sol.status}, 截断节点={sol.prune.truncated_nodes}",
    )
    testcase.assertEqual(sol.total_cost, ref["cost"])
    testcase.assertEqual(sol.max_arrival, ref["max_arrival"])
    testcase.assertEqual(sol.skew, ref["skew"])
    testcase.assertEqual(sol.arrivals, ref["arrivals"])
    solver_picks = tuple(
        next(c.option_index for c in sol.choices if c.edge_id == e.edge_id)
        for e in tree.edges
    )
    testcase.assertEqual(solver_picks, ref["picks"], msg="选项序决胜不一致")

    # 结果自洽性：逐叶路径累加必须等于 arrivals，窗口与 S 必须满足。
    by_id = {e.edge_id: e for e in tree.edges}
    parent_edge = {e.child: e for e in tree.edges}
    root = next(
        n for n in {x for e in tree.edges for x in (e.parent, e.child)}
        if n not in parent_edge
    )
    for leaf, win in ((w.leaf, w) for w in tree.windows):
        path_total = 0
        node = leaf
        while node != root:
            e = parent_edge[node]
            sel = next(c for c in sol.choices if c.edge_id == e.edge_id)
            path_total += sel.edge_delay
            node = e.parent
        testcase.assertEqual(path_total, sol.arrivals[leaf])
        testcase.assertTrue(win.lo <= sol.arrivals[leaf] <= win.hi)
    testcase.assertLessEqual(sol.skew, tree.skew_bound)
    testcase.assertEqual(
        sol.total_cost, sum(c.buffer_cost for c in sol.choices)
    )


def _random_tree(rng: random.Random) -> ClockTree:
    """生成 1..6 边的随机小树（含内部节点/共享干线）。"""
    n_nodes = rng.randint(2, 7)
    names = [f"n{i}" for i in range(n_nodes)]
    edges = []
    for i in range(1, n_nodes):
        parent = names[rng.randint(0, i - 1)]
        opts = []
        for _ in range(rng.randint(0, 3)):
            opts.append(
                BufferOption(rng.randint(0, 3), rng.randint(0, 4))
            )
        edges.append(
            _edge(
                f"e{i}",
                parent,
                names[i],
                base=rng.randint(0, 2),
                options=opts,
            )
        )
    children: dict[str, list[str]] = {}
    for e in edges:
        children.setdefault(e.parent, []).append(e.child)
    leaves = [n for n in names if not children.get(n)]

    # 用枚举自身的可达延迟范围生成窗口（故意随机制造部分不可行实例）。
    tables = [
        [(0, 0)] + [(o.delay, o.cost) for o in e.options] for e in edges
    ]
    incoming = {e.child: e for e in edges}

    def leaf_arrival_extremes(leaf: str):
        # 路径上边索引
        idxs = []
        node = leaf
        while node in incoming:
            e = incoming[node]
            idxs.append(edges.index(e))
            node = e.parent
        lo = sum(edges[i].base_delay + min(d for d, _ in tables[i]) for i in idxs)
        hi = sum(edges[i].base_delay + max(d for d, _ in tables[i]) for i in idxs)
        return lo, hi

    windows = []
    for lf in leaves:
        lo_possible, hi_possible = leaf_arrival_extremes(lf)
        l = rng.randint(lo_possible - 2, hi_possible)
        h = rng.randint(l, hi_possible + 2)
        windows.append(LeafWindow(lf, l, h))
    return ClockTree(tuple(edges), tuple(windows), rng.randint(0, 5))


class SolverTests(unittest.TestCase):
    def test_random_vs_enumeration(self):
        rng = random.Random(20260923)
        for trial in range(400):
            with self.subTest(trial=trial):
                _assert_matches_brute(self, _random_tree(rng))

    def test_shared_trunk_hand_computed(self):
        # 根 -> m（共享干线，base=1，唯一显式缓冲 +1/cost10）
        #   m -> a（base=0，显式缓冲 +2/cost1）
        #   m -> b（base=0，仅隐式零缓冲）
        # 窗口 a∈[2,4]，b∈[1,3]，S=1。
        # 不加干线时只有 a+2 才能到窗口(a=3)，b=1 → skew=2 违反 S=1；
        # 必须在共享干线上加缓冲：a=2,b=2，skew=0，cost=10。
        tree = ClockTree(
            (
                _edge("trunk", "r", "m", base=1,
                      options=[BufferOption(1, 10)]),
                _edge("arm_a", "m", "a", options=[BufferOption(2, 1)]),
                _edge("arm_b", "m", "b"),
            ),
            (LeafWindow("a", 2, 4), LeafWindow("b", 1, 3)),
            1,
        )
        sol = solve(tree)
        self.assertEqual(sol.status, OPTIMAL)
        self.assertEqual(sol.total_cost, 10)
        self.assertEqual(sol.arrivals, {"a": 2, "b": 2})
        self.assertEqual(sol.max_arrival, 2)
        self.assertEqual(sol.skew, 0)
        picks = {c.edge_id: c.option_index for c in sol.choices}
        self.assertEqual(picks, {"trunk": 1, "arm_a": 0, "arm_b": 0})

    def test_infeasible_windows_proven(self):
        # 零选项、固定到达 1 与 2，窗口要求同时等于 5：无解必须证明。
        tree = ClockTree(
            (
                _edge("e1", "r", "a", base=1),
                _edge("e2", "r", "b", base=2),
            ),
            (LeafWindow("a", 5, 5), LeafWindow("b", 5, 5)),
            2,
        )
        sol = solve(tree)
        self.assertEqual(sol.status, INFEASIBLE)
        self.assertFalse(sol.prune.frontier_truncated)
        self.assertIsNone(sol.choices)

    def test_infeasible_skew_proven(self):
        # 到达 1 与 4 固定，窗口都宽松，但 S=2 < skew=3：无解。
        tree = ClockTree(
            (
                _edge("e1", "r", "a", base=1),
                _edge("e2", "r", "b", base=4),
            ),
            (LeafWindow("a", 0, 9), LeafWindow("b", 0, 9)),
            2,
        )
        self.assertEqual(solve(tree).status, INFEASIBLE)

    def test_zero_buffer_only_edges(self):
        tree = ClockTree(
            (
                _edge("e1", "r", "a", base=3),
                _edge("e2", "a", "b", base=2),
            ),
            (LeafWindow("b", 5, 5),),
            0,
        )
        sol = solve(tree)
        self.assertEqual(sol.status, OPTIMAL)
        self.assertEqual(sol.total_cost, 0)
        self.assertEqual(sol.arrivals, {"b": 5})
        self.assertTrue(all(c.option_index == 0 for c in sol.choices))

    def test_nonconvex_options(self):
        # 非凸/反常权衡：更慢的缓冲更便宜（delay2/cost0 优于 delay1/cost5
        # 的成本序），窗口恰好要求到达 2 → 必须选慢而便宜的显式项。
        tree = ClockTree(
            (
                _edge(
                    "e",
                    "r", "a",
                    options=[BufferOption(1, 5), BufferOption(2, 0)],
                ),
            ),
            (LeafWindow("a", 2, 2),),
            0,
        )
        sol = solve(tree)
        self.assertEqual(sol.status, OPTIMAL)
        self.assertEqual(sol.total_cost, 0)
        self.assertEqual(sol.choices[0].option_index, 2)
        self.assertEqual(sol.arrivals, {"a": 2})

    def test_tie_break_max_arrival_then_option_order(self):
        # 两组合同成本：取最大到达更小者；再相同则取选项序字典序。
        tree = ClockTree(
            (
                _edge("e1", "r", "a",
                      options=[BufferOption(2, 1), BufferOption(1, 1)]),
                _edge("e2", "r", "b",
                      options=[BufferOption(1, 1), BufferOption(2, 1)]),
            ),
            (LeafWindow("a", 1, 3), LeafWindow("b", 1, 3)),
            5,
        )
        sol = solve(tree)
        ref = brute_force(tree)
        self.assertIsNotNone(ref)
        # 成本最小都是 2；选择使最大到达最小的组合。
        self.assertEqual(sol.total_cost, ref["cost"])
        self.assertEqual(sol.max_arrival, ref["max_arrival"])
        picks = {c.edge_id: c.option_index for c in sol.choices}
        # e1 选 +1（index2），e2 选 +1（index1）→ 到达均为 1。
        self.assertEqual(picks, {"e1": 2, "e2": 1})

    def test_budget_exceeded_is_distinct_from_infeasible(self):
        # 选项足够多的链，node_vector_cap=1 强制截断：
        # 状态必须是 budget_exceeded 而非 infeasible。
        edges = [
            _edge(
                f"e{i}", f"n{i}", f"n{i+1}",
                options=[BufferOption(d, d) for d in range(1, 4)],
            )
            for i in range(4)
        ]
        tree = ClockTree(
            tuple(edges),
            (LeafWindow("n4", 0, 30),),
            30,
        )
        sol = solve(tree, node_vector_cap=1, work_cap=10**9)
        self.assertEqual(sol.status, BUDGET_EXCEEDED)
        self.assertTrue(sol.prune.frontier_truncated)
        self.assertTrue(sol.prune.truncated_nodes)
        # 若返回了可行解，它本身必须合法（未证明最优 ≠ 随便给）。
        if sol.choices is not None:
            self.assertIn(sol.arrivals["n4"], range(0, 31))
            self.assertEqual(
                sol.total_cost, sum(c.buffer_cost for c in sol.choices)
            )

        # 工作预算耗尽同样不得误判无解。
        sol2 = solve(tree, node_vector_cap=10**9, work_cap=2)
        self.assertEqual(sol2.status, BUDGET_EXCEEDED)
        self.assertGreaterEqual(sol2.prune.work_used, 2)

    def test_budget_dropped_lower_bound_is_lower_bound(self):
        # dropped_cost_lower_bound 的语义：任何**经过被丢弃状态**的完整
        # 可行解，其成本 >= lb。因此当参考最优成本严格小于求解器保留的
        # 最优成本（或求解器未保留可行解）时，真最优必经过被丢弃状态，
        # 必须满足 lb <= ref_cost；若求解器保留的已是真最优，lb 与其无关。
        rng = random.Random(7)
        checked = 0
        for _ in range(100):
            tree = _random_tree(rng)
            sol = solve(tree, node_vector_cap=2, work_cap=10**9)
            if sol.status != BUDGET_EXCEEDED:
                continue
            ref = brute_force(tree)
            lb = sol.prune.dropped_cost_lower_bound
            if ref is None or lb is None:
                continue
            if sol.total_cost is None or ref["cost"] < sol.total_cost:
                self.assertLessEqual(lb, ref["cost"])
                checked += 1
        self.assertGreater(checked, 0)

    def test_prune_stats_and_translation_bounds(self):
        tree = ClockTree(
            (
                _edge("e1", "r", "m", base=1,
                      options=[BufferOption(2, 3)]),
                _edge("e2", "m", "a", options=[BufferOption(1, 1)]),
                _edge("e3", "m", "b", options=[BufferOption(3, 2)]),
            ),
            (LeafWindow("a", 0, 50), LeafWindow("b", 0, 50)),
            50,
        )
        sol = solve(tree)
        self.assertEqual(sol.status, OPTIMAL)
        p = sol.prune
        self.assertGreaterEqual(p.dominance_pruned, 0)
        self.assertGreaterEqual(p.bound_pruned, 0)
        self.assertGreaterEqual(p.vectors_kept, 1)
        self.assertFalse(p.frontier_truncated)
        # m 的平移界：路径 r->m 的延迟取 [1, 3]。
        self.assertEqual(p.translation_bounds["m"], (1, 3))
        # a 的平移界：[1+0, 3+1]。
        self.assertEqual(p.translation_bounds["a"], (1, 4))

    def test_full_25_edge_boundary_runs(self):
        # 接口上限 25 边的规模必须可解（单叶链，选项少）。
        edges = tuple(
            _edge(f"e{i}", f"n{i}", f"n{i+1}", base=1,
                  options=[BufferOption(1, 1)])
            for i in range(25)
        )
        tree = ClockTree(edges, (LeafWindow("n25", 25, 50),), 0)
        sol = solve(tree)
        self.assertEqual(sol.status, OPTIMAL)
        # 全零缓冲即到达 25，成本 0。
        self.assertEqual(sol.arrivals["n25"], 25)
        self.assertEqual(sol.total_cost, 0)


if __name__ == "__main__":
    unittest.main()

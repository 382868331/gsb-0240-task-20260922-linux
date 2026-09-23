"""固定输入演示：偏差与成本的取舍，以及一个实际触发的拒绝边界。

运行：python demo.py（仅标准库，确定性数据，无网络/无 sleep）。
"""

from __future__ import annotations

from clocktree import (
    BufferOption,
    ClockTree,
    Edge,
    InvalidSpecification,
    LeafWindow,
    OPTIMAL,
    solve,
)


def build_tree(skew_bound: int) -> ClockTree:
    """固定的小型时钟树（数据写死，可复现）。

        r --trunk(base1)--> m --arm_a(base2)--> a(leaf)
                            \\--arm_b(base5)--> b(leaf)

    * trunk：共享干线，缓冲只能整体平移两叶（不能消除相互偏差）；
    * arm_a：可加 +1/cost1 或 +3/cost2；
    * arm_b：无缓冲（仅隐式零缓冲）。
    窗口 a,b ∈ [3, 12]。改变 S 展示偏差/成本取舍。
    """
    return ClockTree(
        edges=(
            Edge("trunk", "r", "m", base_delay=1,
                 options=(BufferOption(2, 8), BufferOption(4, 14))),
            Edge("arm_a", "m", "a", base_delay=2,
                 options=(BufferOption(1, 1), BufferOption(3, 2))),
            Edge("arm_b", "m", "b", base_delay=5, options=()),
        ),
        windows=(LeafWindow("a", 3, 12), LeafWindow("b", 3, 12)),
        skew_bound=skew_bound,
    )


def show(title: str, tree: ClockTree) -> None:
    print("=" * 68)
    print(title)
    sol = solve(tree)
    assert sol.status == OPTIMAL, f"固定用例应证明最优，实际 {sol.status}"
    print(f"状态              : {sol.status}（已证明全局最优）")
    print(f"最小总成本        : {sol.total_cost}")
    print(f"叶最大到达/最大偏差: max={sol.max_arrival}, skew={sol.skew} "
          f"(约束 S={tree.skew_bound})")

    by_id = {c.edge_id: c for c in sol.choices}
    layout_edges = {e.edge_id: e for e in tree.edges}
    parent_of = {e.child: e.edge_id for e in tree.edges}
    root = next(
        n for n in {x for e in tree.edges for x in (e.parent, e.child)}
        if n not in parent_of
    )
    print("每边选项（0=隐式零缓冲）:")
    for c in sol.choices:
        print(f"  {c.edge_id:6s} 选项{c.option_index}  "
              f"延迟={c.edge_delay}(base {c.base_delay}+buf {c.buffer_delay}) "
              f"成本={c.buffer_cost}")
    print("每叶路径累加:")
    for leaf, total in sorted(sol.arrivals.items()):
        path, node, acc = [], leaf, 0
        while node != root:
            eid = parent_of[node]
            sel = by_id[eid]
            path.append(f"{eid}:{sel.edge_delay}")
            acc += sel.edge_delay
            node = layout_edges[eid].parent
        path.reverse()
        print(f"  {leaf}: {' + '.join(path)} = {acc}（窗口"
              f"[{next(w for w in tree.windows if w.leaf == leaf).lo},"
              f"{next(w for w in tree.windows if w.leaf == leaf).hi}]）"
              f"  累加核对={'OK' if acc == total else 'FAIL'}")
    p = sol.prune
    print(f"剪枝界            : 界剪枝 {p.bound_pruned} 条，"
          f"占优剪枝 {p.dominance_pruned} 条，"
          f"保留 {p.vectors_kept} 个状态，截断={p.frontier_truncated}")


def show_rejection() -> None:
    print("=" * 68)
    print("拒绝边界（实际抛出的可定位异常，不静默纠正）:")
    # 整数参数传入 bool —— 必须被拒绝并指出字段。
    try:
        solve(
            ClockTree(
                edges=(Edge("e", "r", "a", base_delay=1),),
                windows=(LeafWindow("a", 1, 1),),
                skew_bound=True,  # type: ignore[arg-type]
            )
        )
    except InvalidSpecification as exc:
        print(f"  捕获 InvalidSpecification: {exc}")
    else:
        raise AssertionError("bool 型 skew_bound 未被拒绝")

    # 空闭区间同样被拒。
    try:
        LeafWindow("a", lo=9, hi=3)
    except InvalidSpecification as exc:
        print(f"  捕获 InvalidSpecification: {exc}")
    else:
        raise AssertionError("空窗口未被拒绝")


def main() -> None:
    show("场景 1：S=3（偏差宽松）—— 零成本即可满足", build_tree(3))
    show("场景 2：S=0（要求零偏差）—— 必须付费在支线补延迟", build_tree(0))
    show_rejection()
    print("=" * 68)
    print("演示完成：提高偏差约束会把成本从 0 提高到 2，"
          "且共享干线缓冲只能整体平移、不能消除偏差。")


if __name__ == "__main__":
    main()

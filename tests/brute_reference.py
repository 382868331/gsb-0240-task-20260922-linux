"""独立暴力参考：枚举每条边的全部选项组合。

本模块**不导入也不调用** clocktree.solver，只直接读取 ClockTree 的
原始字段（edges/windows/skew_bound），用自成一体的 DFS 与枚举求期望
最优值，供测试交叉核对。仅用于小树（测试内）。
"""

from __future__ import annotations

from itertools import product
from typing import Optional

from clocktree.model import ClockTree


def brute_force(tree: ClockTree) -> Optional[dict]:
    """枚举所有选项组合，返回最优解；无解返回 None。

    目标序与被测契约一致：(总成本, 最大叶到达, 按边输入序的选项向量)。
    """
    edges = list(tree.edges)
    # 自成一体的父子结构与根推导。
    children: dict[str, list[str]] = {}
    incoming: dict[str, str] = {}
    nodes: set[str] = set()
    for e in edges:
        children.setdefault(e.parent, []).append(e.child)
        children.setdefault(e.child, [])
        incoming[e.child] = e.edge_id
        nodes.add(e.parent)
        nodes.add(e.child)
    root = next(n for n in nodes if n not in incoming)

    # 每边选项表：索引 0 为隐式零缓冲。
    choice_tables = [
        [(0, 0)] + [(o.delay, o.cost) for o in e.options] for e in edges
    ]

    win = {w.leaf: (w.lo, w.hi) for w in tree.windows}
    leaves = [n for n, ch in children.items() if not ch]

    best = None
    ranges = [range(len(t)) for t in choice_tables]
    for picks in product(*ranges):
        # 边索引 -> 实际延迟/成本
        delay_by_edge = {
            e.edge_id: e.base_delay + choice_tables[i][picks[i]][0]
            for i, e in enumerate(edges)
        }
        cost_by_edge = {
            e.edge_id: choice_tables[i][picks[i]][1]
            for i, e in enumerate(edges)
        }
        # DFS 累加到各叶。
        arrivals: dict[str, int] = {}
        total_cost = 0
        stack: list[tuple[str, int]] = [(root, 0)]
        while stack:
            node, acc = stack.pop()
            ch = children[node]
            if not ch:
                arrivals[node] = acc
                continue
            for c in ch:
                eid = incoming[c]
                total_cost += cost_by_edge[eid]
                stack.append((c, acc + delay_by_edge[eid]))
        ok = True
        for lf in leaves:
            lo, hi = win[lf]
            if not (lo <= arrivals[lf] <= hi):
                ok = False
                break
        if ok:
            skew = max(arrivals.values()) - min(arrivals.values())
            if skew > tree.skew_bound:
                ok = False
        if not ok:
            continue
        key = (total_cost, max(arrivals.values()), picks)
        if best is None or key < best[0]:
            best = (
                key,
                {
                    "cost": total_cost,
                    "arrivals": dict(arrivals),
                    "picks": picks,
                    "skew": max(arrivals.values()) - min(arrivals.values()),
                    "max_arrival": max(arrivals.values()),
                },
            )
    return None if best is None else best[1]

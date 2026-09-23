"""树 DP 求解器：时间包络 + 成本 Pareto 状态。

算法概述
========
每个节点 ``u`` 维护一个按成本升序的 Pareto 前沿 ``states``：

    cost(int) -> {相对到达向量: 该子树内边的选项序列}

* 相对到达向量按 ``u`` 子树叶的 DFS 序对齐，记录各叶相对 ``u`` 的到达；
  选项序列为 ``(edge_index, option_index)`` 的有序元组，仅覆盖子树内边。
* 同一向量只保留达到它的**最小成本**配置：更便宜的配置对未来“整体平移 +
  各叶窗口 + 全局极差”约束永远不劣。同成本同向量保留选项序字典序最小者。
* 合并子节点：子边选项整体平移子节点前沿，与当前前沿做笛卡尔卷积，
  随后执行两条安全剪枝：

    1. **界剪枝（bound）**：祖先边只对子树施加共同平移 ``t``。由根→u
       路径上各边选项算出整数松弛区间 ``[t_min(u), t_max(u)]``（各边独立
       取最小/最大，是可达平移集合的外包区间）。向量 ``x`` 存活当且仅当
       ``range(x) <= S`` 且
       ``max(t_min, max_l(lo_l - x_l)) <= min(t_max, min_l(hi_l - x_l))``。
       在根处 ``t_min=t_max=0``，判定即真实可行性。
    2. **占优剪枝（dominance）**：成本升序扫描，更便宜状态已实现的向量
       在更贵层直接丢弃。

* 前沿完整构建且无截断时，根处最小成本经占优论证为**已证明全局最优**；
  同级目标依次为最小最大叶到达、选项序字典序。
* 前沿向量数或卷积工作量超过预算时**确定性截断**（保留低成本前缀）并沿
  树向上传播截断标记：根处有可行解则状态为 :data:`BUDGET_EXCEEDED`
  （未证明最优），无可行解也仍为 :data:`BUDGET_EXCEEDED`（不能证明无解），
  与严格证明的 :data:`INFEASIBLE` 明确分开。

全程整数运算，无浮点。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .model import ClockTree, Layout, compile_layout
from .model import InvalidSpecification

__all__ = [
    "OPTIMAL",
    "INFEASIBLE",
    "BUDGET_EXCEEDED",
    "SelectedOption",
    "PruneStats",
    "Solution",
    "solve",
]

OPTIMAL = "optimal"
INFEASIBLE = "infeasible"
BUDGET_EXCEEDED = "budget_exceeded"

# 防御性默认预算（≤25 边的接口边界下通常不会触及；非题面约束）。
DEFAULT_NODE_VECTOR_CAP = 100_000
DEFAULT_WORK_CAP = 5_000_000

# 前沿：cost -> {相对向量: 选项序列}
Frontier = dict[int, dict[tuple[int, ...], tuple[tuple[int, int], ...]]]


@dataclass(frozen=True)
class SelectedOption:
    """一条边上选中的选项。``option_index=0`` 表示隐式零缓冲。"""

    edge_index: int
    edge_id: str
    option_index: int
    base_delay: int
    buffer_delay: int
    buffer_cost: int

    @property
    def edge_delay(self) -> int:
        return self.base_delay + self.buffer_delay


@dataclass(frozen=True)
class PruneStats:
    """剪枝与预算统计（“被剪枝界”的可核对明细）。"""

    vectors_kept: int
    dominance_pruned: int
    bound_pruned: int
    work_used: int
    largest_frontier: int
    frontier_truncated: bool
    truncated_nodes: tuple[str, ...]
    node_vector_cap: int
    work_cap: int
    #: 截断时被丢弃前缀的最小成本下界（无截断为 None）。
    dropped_cost_lower_bound: Optional[int]
    #: 每节点祖先平移外包区间 [t_min, t_max]（界剪枝使用的整数界）。
    translation_bounds: dict[str, tuple[int, int]] = field(default_factory=dict)


@dataclass(frozen=True)
class Solution:
    """求解结果。

    ``status``：
      * :data:`OPTIMAL`：已证明最优；
      * :data:`INFEASIBLE`：已证明无解（窗口或 S 不可满足）；
      * :data:`BUDGET_EXCEEDED`：搜索预算耗尽，未证明最优；``choices`` 等
        字段给出已找到的最优可行解，找不到时为 ``None``。
    """

    status: str
    choices: Optional[tuple[SelectedOption, ...]]
    arrivals: Optional[dict[str, int]]
    total_cost: Optional[int]
    max_arrival: Optional[int]
    skew: Optional[int]
    prune: PruneStats


class _Budget:
    def __init__(self, node_cap: int, work_cap: int) -> None:
        self.node_cap = node_cap
        self.work_cap = work_cap
        self.work_used = 0
        self.truncated_nodes: set[str] = set()
        self.dropped_lb: Optional[int] = None

    def note_truncation(self, node: str, lb: Optional[int]) -> None:
        self.truncated_nodes.add(node)
        if lb is not None and (
            self.dropped_lb is None or lb < self.dropped_lb
        ):
            self.dropped_lb = lb


def _sorted_frontier(states: Frontier) -> Frontier:
    """返回按成本升序重建的新前沿，并删除空成本层。"""
    return {c: states[c] for c in sorted(states) if states[c]}


def _merge_same_cost(target: dict, source: dict) -> None:
    """同成本层合并：同向量保留选项序字典序最小者。"""
    for vec, seq in source.items():
        old = target.get(vec)
        if old is None or seq < old:
            target[vec] = seq


def _bound_filter(
    states: Frontier,
    leaves: tuple[str, ...],
    windows: dict[str, "object"],
    t_lo: int,
    t_hi: int,
    skew_bound: int,
) -> int:
    """删除不可能被任何祖先平移挽救的向量，返回删除条数。

    向量 ``x`` 存活要求：range(x) <= S，且存在整数 t∈[t_lo,t_hi] 使
    每个叶 l 满足 lo_l <= x_l + t <= hi_l。
    根节点调用时 t_lo=t_hi=0，即真实可行性判定。
    """
    los = tuple(windows[lf].lo for lf in leaves)
    his = tuple(windows[lf].hi for lf in leaves)
    n = len(leaves)
    removed = 0
    for cost, m in states.items():
        alive: dict[tuple, tuple] = {}
        for vec, seq in m.items():
            mn = min(vec)
            if max(vec) - mn > skew_bound:
                removed += 1
                continue
            lo_t = t_lo
            hi_t = t_hi
            for i in range(n):
                need_lo = los[i] - vec[i]
                need_hi = his[i] - vec[i]
                if need_lo > lo_t:
                    lo_t = need_lo
                if need_hi < hi_t:
                    hi_t = need_hi
                if lo_t > hi_t:
                    break
            if lo_t <= hi_t:
                alive[vec] = seq
            else:
                removed += 1
        states[cost] = alive
    return removed


def _dominance_sweep(states: Frontier) -> int:
    """成本升序丢弃已被更便宜状态实现的向量，返回删除条数。"""
    seen: set[tuple] = set()
    removed = 0
    for cost in sorted(states):
        m = states[cost]
        kept = {}
        for vec, seq in m.items():
            if vec in seen:
                removed += 1
            else:
                kept[vec] = seq
        seen.update(kept)
        states[cost] = kept
    for cost in [c for c, m in states.items() if not m]:
        del states[cost]
    return removed


def _cap_to_budget(
    out: Frontier,
    generated: int,
    budget: _Budget,
    node: str,
    lb_unexplored: Optional[int],
) -> None:
    """按成本（同成本按向量、选项序）确定性截断到 node_vector_cap。"""
    flat: list[tuple[int, tuple, tuple]] = []
    for cost in sorted(out):
        for vec, seq in sorted(out[cost].items()):
            flat.append((cost, vec, seq))
    lb: Optional[int] = None
    if len(flat) > budget.node_cap:
        lb = flat[budget.node_cap][0]
    if generated > 0 and lb_unexplored is not None:
        lb = lb_unexplored if lb is None else min(lb, lb_unexplored)
    if len(flat) > budget.node_cap or lb_unexplored is not None:
        budget.note_truncation(node, lb)
    kept = flat[: budget.node_cap]
    out.clear()
    for cost, vec, seq in kept:
        out.setdefault(cost, {})[vec] = seq


def _convolve(
    acc: Frontier,
    branch: Frontier,
    budget: _Budget,
    node: str,
) -> Frontier:
    """两个前沿的成本卷积（向量拼接、成本相加、选项序归并）。

    超预算时停止生成并截断，记账被丢弃界。对未生成的成本层，采用
    “行主序游标”计算其真实最小成本：当前行剩余最小 cb、后续行的
    最小 cb，再与该行 ca 相加——不能用 ca + b_costs[0] 近似，
    否则在两个前沿成本层错位时会高估丢弃界。
    """
    a_costs = sorted(acc)
    b_costs = sorted(branch)
    out: Frontier = {}
    stop = False
    lb_unexplored: Optional[int] = None
    generated = 0
    cap = budget.node_cap

    def _remaining_lb(ia: int, ib_next: int) -> Optional[int]:
        candidates = []
        if ib_next < len(b_costs):
            candidates.append(a_costs[ia] + b_costs[ib_next])
        for ja in range(ia + 1, len(a_costs)):
            candidates.append(a_costs[ja] + b_costs[0])
        return min(candidates) if candidates else None

    for ia, ca in enumerate(a_costs):
        ma = acc[ca]
        for ib, cb in enumerate(b_costs):
            mb = branch[cb]
            budget.work_used += len(ma) * len(mb)
            if budget.work_used > budget.work_cap and not stop:
                stop = True
                # 当前成本层整体尚未生成（下方 stop 跳过），故含 ib。
                lb_unexplored = _remaining_lb(ia, ib)
            if stop:
                continue
            cost = ca + cb
            target = out.setdefault(cost, {})
            for va, sa in ma.items():
                for vb, sb in mb.items():
                    vec = va + vb
                    seq = tuple(sorted(sa + sb))
                    old = target.get(vec)
                    if old is None or seq < old:
                        target[vec] = seq
                    generated += 1
            if generated >= cap:
                # 向量数超预算：确定性停止并记账下一未处理成本层。
                stop = True
                lb_unexplored = _remaining_lb(ia, ib + 1)
                break
        if stop:
            break

    if stop:
        _cap_to_budget(out, generated, budget, node, lb_unexplored)
    return _sorted_frontier(out)


def solve(
    tree: ClockTree,
    *,
    node_vector_cap: int = DEFAULT_NODE_VECTOR_CAP,
    work_cap: int = DEFAULT_WORK_CAP,
) -> Solution:
    """求解时钟树偏差预算分配问题。

    参数
    ----
    tree:
        :class:`~clocktree.model.ClockTree`，先经严格校验与编译。
    node_vector_cap:
        每节点前沿最大保留向量数（正整数，拒绝 bool）。
    work_cap:
        全局卷积工作量上限（正整数，单位：向量对运算，拒绝 bool）。

    异常
    ----
    InvalidSpecification:
        树规格或预算参数非法。
    """
    if isinstance(node_vector_cap, bool) or not isinstance(node_vector_cap, int):
        raise InvalidSpecification("node_vector_cap 必须是 int 且非 bool")
    if isinstance(work_cap, bool) or not isinstance(work_cap, int):
        raise InvalidSpecification("work_cap 必须是 int 且非 bool")
    if node_vector_cap <= 0 or work_cap <= 0:
        raise InvalidSpecification("预算参数必须为正整数")

    layout = compile_layout(tree)
    return _solve_layout(layout, tree.skew_bound, node_vector_cap, work_cap)


def _solve_layout(
    layout: Layout,
    skew_bound: int,
    node_cap: int,
    work_cap: int,
) -> Solution:
    budget = _Budget(node_cap, work_cap)
    windows = layout.windows

    # 各节点全局到达时间的祖先平移外包区间（含其入边选项）。
    t_min: dict[str, int] = {layout.root: 0}
    t_max: dict[str, int] = {layout.root: 0}
    for n in layout.nodes:  # DFS 前序：父先于子
        if n == layout.root:
            continue
        pe = layout.parent_edge[n]
        dmin = min(d for d, _ in pe.choices)
        dmax = max(d for d, _ in pe.choices)
        t_min[n] = t_min[pe.parent] + pe.base_delay + dmin
        t_max[n] = t_max[pe.parent] + pe.base_delay + dmax

    # 子树叶子表（DFS 序）。
    leaves_of: dict[str, tuple[str, ...]] = {}
    for u in layout.postorder:
        kids = layout.children[u]
        if not kids:
            leaves_of[u] = (u,)
        else:
            buf: list[str] = []
            for v in kids:
                buf.extend(leaves_of[v])
            leaves_of[u] = tuple(buf)

    stats = {"dom": 0, "bound": 0, "largest": 0}
    node_states: dict[str, Frontier] = {}

    for u in layout.postorder:
        kids = layout.children[u]
        if not kids:
            states: Frontier = {0: {(0,): ()}}
        else:
            states = {0: {(): ()}}
            prefix_leaves: list[str] = []
            for v in kids:
                pe = layout.parent_edge[v]
                # 子边选项平移子节点前沿 -> 相对 u 的分支前沿。
                branch: Frontier = {}
                child = node_states[v]
                for j, (bdelay, bcost) in enumerate(pe.choices):
                    add_t = pe.base_delay + bdelay
                    for c, m in child.items():
                        cost = c + bcost
                        layer: dict[tuple, tuple] = {}
                        for vec, seq in m.items():
                            new_vec = tuple(x + add_t for x in vec)
                            new_seq = tuple(sorted(seq + ((pe.index, j),)))
                            old = layer.get(new_vec)
                            if old is None or new_seq < old:
                                layer[new_vec] = new_seq
                        tgt = branch.setdefault(cost, {})
                        _merge_same_cost(tgt, layer)
                branch = _sorted_frontier(branch)
                if v in budget.truncated_nodes:
                    budget.note_truncation(u, None)
                prefix_leaves.extend(leaves_of[v])
                stats["bound"] += _bound_filter(
                    branch,
                    leaves_of[v],
                    windows,
                    t_min[u],
                    t_max[u],
                    skew_bound,
                )
                stats["dom"] += _dominance_sweep(branch)
                states = _convolve(states, branch, budget, u)
                stats["bound"] += _bound_filter(
                    states,
                    tuple(prefix_leaves),
                    windows,
                    t_min[u],
                    t_max[u],
                    skew_bound,
                )
                stats["dom"] += _dominance_sweep(states)
        # 对完整子树前沿按节点自身的祖先平移界过滤（根处即真实可行性）。
        stats["bound"] += _bound_filter(
            states, leaves_of[u], windows, t_min[u], t_max[u], skew_bound
        )
        stats["dom"] += _dominance_sweep(states)
        size = sum(len(m) for m in states.values())
        stats["largest"] = max(stats["largest"], size)
        node_states[u] = states

    root_states = node_states[layout.root]
    feasible: list[tuple[int, int, tuple, tuple]] = []
    for cost in sorted(root_states):
        for vec, seq in root_states[cost].items():
            if max(vec) - min(vec) > skew_bound:
                continue
            if all(
                windows[lf].lo <= vec[i] <= windows[lf].hi
                for i, lf in enumerate(layout.leaves)
            ):
                feasible.append((cost, max(vec), seq, vec))

    total_kept = sum(
        sum(len(m) for m in st.values()) for st in node_states.values()
    )
    truncated = bool(budget.truncated_nodes)
    prune = PruneStats(
        vectors_kept=total_kept,
        dominance_pruned=stats["dom"],
        bound_pruned=stats["bound"],
        work_used=budget.work_used,
        largest_frontier=stats["largest"],
        frontier_truncated=truncated,
        truncated_nodes=tuple(sorted(budget.truncated_nodes)),
        node_vector_cap=node_cap,
        work_cap=work_cap,
        dropped_cost_lower_bound=budget.dropped_lb,
        translation_bounds={n: (t_min[n], t_max[n]) for n in layout.nodes},
    )

    if not feasible:
        # 截断意味着可能丢过可行状态，不能声称无解。
        status = BUDGET_EXCEEDED if truncated else INFEASIBLE
        return Solution(status, None, None, None, None, None, prune)

    feasible.sort(key=lambda z: (z[0], z[1], z[2]))
    cost, mx, seq, _vec = feasible[0]
    choices, arrivals = _reconstruct(layout, seq)
    skew = max(arrivals.values()) - min(arrivals.values())
    return Solution(
        status=BUDGET_EXCEEDED if truncated else OPTIMAL,
        choices=choices,
        arrivals=arrivals,
        total_cost=cost,
        max_arrival=mx,
        skew=skew,
        prune=prune,
    )


def _reconstruct(
    layout: Layout, seq: tuple[tuple[int, int], ...]
) -> tuple[tuple[SelectedOption, ...], dict[str, int]]:
    choice_by_edge = dict(seq)
    choices: list[SelectedOption] = []
    for pe in layout.edges:
        j = choice_by_edge.get(pe.index, 0)
        bdelay, bcost = pe.choices[j]
        choices.append(
            SelectedOption(
                edge_index=pe.index,
                edge_id=pe.edge_id,
                option_index=j,
                base_delay=pe.base_delay,
                buffer_delay=bdelay,
                buffer_cost=bcost,
            )
        )
    arrivals: dict[str, int] = {}
    for lf in layout.leaves:
        total = 0
        for pe in layout.leaf_path_edges(lf):
            j = choice_by_edge.get(pe.index, 0)
            bdelay, _ = pe.choices[j]
            total += pe.base_delay + bdelay
        arrivals[lf] = total
    return tuple(choices), arrivals

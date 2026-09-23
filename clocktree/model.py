"""公开数据结构、输入校验与树布局编译。

模型
----
一棵有根树：

* 每条 :class:`Edge` 有 ``edge_id``、父节点 ``parent``、子节点 ``child``、
  非负整数 ``base_delay`` 与至少一个离散选项 ``options``。
  选项 0 是隐式零缓冲（delay=0, cost=0），选项 1..k-1 为显式缓冲
  :class:`BufferOption`；每边恰好选择一个选项。
* 叶子通过 :class:`LeafWindow` 给出到达时间闭区间 ``[lo, hi]``。
* 内部边只用于承载延迟/成本，不做窗口约束。

边 u→v 的实际延迟 = ``base_delay + options[choice].delay``，
叶子 v 的到达时间 = 根→v 路径上所有边的实际延迟之和（根时间为 0）。

所有数值均为整数（``bool`` 不是合法整数），拒绝 NaN/Infinity 一类
非有限值由“必须是 int”这一前提天然覆盖。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

__all__ = [
    "BufferOption",
    "ClockTreeError",
    "InvalidSpecification",
    "Edge",
    "LeafWindow",
    "ClockTree",
    "CompiledEdge",
    "Layout",
    "compile_layout",
]


class ClockTreeError(Exception):
    """本库所有错误的基类。"""


class InvalidSpecification(ClockTreeError, ValueError):
    """输入规格非法（可定位到具体字段）。"""


# ---------------------------------------------------------------------------
# 输入校验小工具
# ---------------------------------------------------------------------------

def _int(value, what: str) -> int:
    """要求严格的 ``int`` 且拒绝 ``bool``。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidSpecification(
            f"{what} 必须是 int（拒绝 bool 与浮点），实际为 {type(value).__name__}: {value!r}"
        )
    return value


def _identifier(value, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise InvalidSpecification(f"{what} 必须是非空 str，实际为 {value!r}")
    return value


# ---------------------------------------------------------------------------
# 输入数据类
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BufferOption:
    """显式缓冲选项：附加延迟 ``delay``（非负整数）与成本 ``cost``（非负整数）。"""

    delay: int
    cost: int

    def __post_init__(self) -> None:
        # dataclass(frozen=True) 下用 object.__setattr__ 不需要；这里只做校验。
        d = _int(self.delay, "BufferOption.delay")
        c = _int(self.cost, "BufferOption.cost")
        if d < 0:
            raise InvalidSpecification(f"BufferOption.delay 必须 >= 0，实际为 {d}")
        if c < 0:
            raise InvalidSpecification(f"BufferOption.cost 必须 >= 0，实际为 {c}")


@dataclass(frozen=True)
class Edge:
    """树边。

    ``options`` 为显式缓冲列表，可为空（表示只有零缓冲可选）。
    求解时选项 0 始终是隐式零缓冲 (delay=0, cost=0)，其后依次是
    ``options`` 中的显式项；“选项序”即这个顺序。
    """

    edge_id: str
    parent: str
    child: str
    base_delay: int
    options: Sequence[BufferOption] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _identifier(self.edge_id, "Edge.edge_id")
        _identifier(self.parent, f"Edge({self.edge_id!r}).parent")
        _identifier(self.child, f"Edge({self.edge_id!r}).child")
        if self.parent == self.child:
            raise InvalidSpecification(
                f"Edge {self.edge_id!r} 的 parent 与 child 不能相同: {self.child!r}"
            )
        _int(self.base_delay, f"Edge({self.edge_id!r}).base_delay")
        if self.base_delay < 0:
            raise InvalidSpecification(
                f"Edge {self.edge_id!r}.base_delay 必须 >= 0，实际为 {self.base_delay}"
            )
        if not isinstance(self.options, (tuple, list)):
            raise InvalidSpecification(
                f"Edge {self.edge_id!r}.options 必须是 list/tuple，"
                f"实际为 {type(self.options).__name__}"
            )
        # 拷贝成不可变 tuple，逐项触发 BufferOption 校验。
        opts = tuple(self.options)
        for i, opt in enumerate(opts):
            if not isinstance(opt, BufferOption):
                raise InvalidSpecification(
                    f"Edge {self.edge_id!r}.options[{i}] 必须是 BufferOption，"
                    f"实际为 {type(opt).__name__}"
                )
        object.__setattr__(self, "options", opts)

    @property
    def option_count(self) -> int:
        """含隐式零缓冲在内的选项数。"""
        return len(self.options) + 1


@dataclass(frozen=True)
class LeafWindow:
    """叶子到达时间闭区间 ``[lo, hi]``（整数，含端点）。"""

    leaf: str
    lo: int
    hi: int

    def __post_init__(self) -> None:
        _identifier(self.leaf, "LeafWindow.leaf")
        lo = _int(self.lo, f"LeafWindow({self.leaf!r}).lo")
        hi = _int(self.hi, f"LeafWindow({self.leaf!r}).hi")
        if lo > hi:
            raise InvalidSpecification(
                f"LeafWindow {self.leaf!r} 的 lo({lo}) > hi({hi})，闭区间为空"
            )


@dataclass(frozen=True)
class ClockTree:
    """一个时钟树问题实例。

    ``edges`` 至多 25 条；``windows`` 必须恰好覆盖所有叶子
    （没有子边的节点；根在无边情形下也算叶子）。
    ``skew_bound`` 为全树叶最大偏差上界 ``S``（>= 0）。
    """

    edges: Sequence[Edge]
    windows: Sequence[LeafWindow]
    skew_bound: int

    #: 接口边界：根树边数上限。
    MAX_EDGES = 25

    def __post_init__(self) -> None:
        if not isinstance(self.edges, (tuple, list)):
            raise InvalidSpecification(
                f"edges 必须是 list/tuple，实际为 {type(self.edges).__name__}"
            )
        edges = tuple(self.edges)
        if not edges:
            raise InvalidSpecification("空树不支持：必须能够确定根与叶子")
        if not isinstance(self.windows, (tuple, list)):
            raise InvalidSpecification(
                f"windows 必须是 list/tuple，实际为 {type(self.windows).__name__}"
            )
        windows = tuple(self.windows)
        s = _int(self.skew_bound, "skew_bound")
        if s < 0:
            raise InvalidSpecification(f"skew_bound 必须 >= 0，实际为 {s}")
        if len(edges) > self.MAX_EDGES:
            raise InvalidSpecification(
                f"边数 {len(edges)} 超过接口上限 MAX_EDGES={self.MAX_EDGES}"
            )
        for e in edges:
            if not isinstance(e, Edge):
                raise InvalidSpecification(
                    f"edges 中存在非 Edge 元素: {type(e).__name__}"
                )
        for w in windows:
            if not isinstance(w, LeafWindow):
                raise InvalidSpecification(
                    f"windows 中存在非 LeafWindow 元素: {type(w).__name__}"
                )
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "windows", windows)


# ---------------------------------------------------------------------------
# 编译后的树布局（求解器内部使用，也作为公开结果的一部分）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompiledEdge:
    """编译后边：稳定索引 + 含零缓冲在内的 (delay, cost) 选项表。"""

    index: int
    edge_id: str
    parent: str
    child: str
    base_delay: int
    choices: tuple[tuple[int, int], ...]

    @property
    def option_count(self) -> int:
        return len(self.choices)


@dataclass(frozen=True)
class Layout:
    """校验并拓扑化之后的不可变树布局。"""

    root: str
    nodes: tuple[str, ...]
    edges: tuple[CompiledEdge, ...]
    children: Mapping[str, tuple[str, ...]]
    parent_edge: Mapping[str, CompiledEdge]
    leaves: tuple[str, ...]
    windows: Mapping[str, LeafWindow]
    # 后序（子先于父）供树 DP 使用；叶子次序为 DFS 前序。
    postorder: tuple[str, ...]

    def leaf_path_edges(self, leaf: str) -> tuple[CompiledEdge, ...]:
        """从根到叶子的路径边（根→叶顺序）。"""
        out: list[CompiledEdge] = []
        node = leaf
        while node != self.root:
            e = self.parent_edge[node]
            out.append(e)
            node = e.parent
        return tuple(reversed(out))


def compile_layout(tree: ClockTree) -> Layout:
    """校验 :class:`ClockTree` 并编译成 :class:`Layout`。

    检查内容：边 id 唯一、节点命名一致、恰好一棵以单一根生长的有根树、
    无环、窗口恰好覆盖叶子、边数不超上限。
    """
    if not isinstance(tree, ClockTree):
        raise InvalidSpecification(
            f"compile_layout 需要 ClockTree，实际为 {type(tree).__name__}"
        )
    edges = tree.edges
    if not edges:
        raise InvalidSpecification("空树不支持：必须能够确定根与叶子")

    seen_ids: set[str] = set()
    child_of: dict[str, str] = {}
    raw_children: dict[str, list[str]] = {}
    nodes: set[str] = set()
    for e in edges:
        if e.edge_id in seen_ids:
            raise InvalidSpecification(f"边 id 重复: {e.edge_id!r}")
        seen_ids.add(e.edge_id)
        if e.child in child_of:
            raise InvalidSpecification(
                f"节点 {e.child!r} 有两条入边（{child_of[e.child]!r} 与 {e.edge_id!r}）"
            )
        child_of[e.child] = e.edge_id
        raw_children.setdefault(e.parent, []).append(e.child)
        raw_children.setdefault(e.child, [])
        nodes.add(e.parent)
        nodes.add(e.child)

    roots = [n for n in nodes if n not in child_of]
    if len(roots) != 1:
        raise InvalidSpecification(
            f"必须恰好有一个根节点，实际找到 {len(roots)} 个: {sorted(roots)}"
        )
    root = roots[0]

    # 从根做确定性 DFS 前序遍历（子节点按名字排序），确认连通且无环。
    # child_of 已限制入度 <= 1；不可达分量意味着森林或环。
    # DFS 前序同时保证：父先于子（供路径量递推）、叶子次序与
    # 子树叶子的 DFS 拼接次序一致（供 DP 向量对齐）。
    order: list[str] = []
    reached: set[str] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node in reached:
            raise InvalidSpecification(f"检测到环或重复可达节点: {node!r}")
        reached.add(node)
        order.append(node)
        for ch in sorted(raw_children.get(node, ()), reverse=True):
            stack.append(ch)
    if reached != nodes:
        missing = sorted(nodes - reached)
        raise InvalidSpecification(f"树不连通，以下节点不可由根到达: {missing}")

    children = {n: tuple(sorted(raw_children.get(n, ()))) for n in order}
    leaves = tuple(n for n in order if not children[n])

    win_map: dict[str, LeafWindow] = {}
    for w in tree.windows:
        if w.leaf not in nodes:
            raise InvalidSpecification(
                f"窗口引用了不存在的节点: {w.leaf!r}"
            )
        if w.leaf in win_map:
            raise InvalidSpecification(f"叶子 {w.leaf!r} 有多个窗口")
        if w.leaf not in leaves:
            raise InvalidSpecification(
                f"窗口只能挂在叶子上，{w.leaf!r} 有子边，是内部节点"
            )
        win_map[w.leaf] = w
    missing_win = [n for n in leaves if n not in win_map]
    if missing_win:
        raise InvalidSpecification(f"以下叶子缺少到达时间窗口: {missing_win}")
    extra = sorted(set(win_map) - set(leaves))
    if extra:  # 理论上前面已挡，保留为防御性检查。
        raise InvalidSpecification(f"窗口挂在非叶子节点上: {extra}")

    compiled = []
    for i, e in enumerate(edges):
        choices = [(0, 0)]
        for opt in e.options:
            choices.append((opt.delay, opt.cost))
        compiled.append(
            CompiledEdge(
                index=i,
                edge_id=e.edge_id,
                parent=e.parent,
                child=e.child,
                base_delay=e.base_delay,
                choices=tuple(choices),
            )
        )
    edge_by_child = {ce.child: ce for ce in compiled}

    # 后序（子先于父）供树 DP 使用；叶子按 DFS 前序编号。
    postorder = tuple(reversed(order))

    return Layout(
        root=root,
        nodes=tuple(order),
        edges=tuple(compiled),
        children=children,
        parent_edge=edge_by_child,
        leaves=leaves,
        windows=win_map,
        postorder=postorder,
    )

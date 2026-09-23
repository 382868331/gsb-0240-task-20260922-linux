"""硬件时钟树偏差预算分配（clock-tree skew budgeting）。

公开接口见 :mod:`clocktree.model` 与 :func:`clocktree.solver.solve`。
全部时间/成本均为整数，求解过程只使用整数运算。
"""

from .model import (
    BufferOption,
    ClockTree,
    Edge,
    LeafWindow,
    InvalidSpecification,
    ClockTreeError,
    CompiledEdge,
    Layout,
    compile_layout,
)
from .solver import (
    BUDGET_EXCEEDED,
    INFEASIBLE,
    OPTIMAL,
    PruneStats,
    SelectedOption,
    Solution,
    solve,
)

__all__ = [
    "BufferOption",
    "ClockTree",
    "Edge",
    "LeafWindow",
    "InvalidSpecification",
    "ClockTreeError",
    "CompiledEdge",
    "Layout",
    "compile_layout",
    "BUDGET_EXCEEDED",
    "INFEASIBLE",
    "OPTIMAL",
    "PruneStats",
    "SelectedOption",
    "Solution",
    "solve",
]

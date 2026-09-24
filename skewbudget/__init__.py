"""Hardware clock-tree skew-budget allocation (integer time, exact DP).

Public API:
    Option, Edge, LeafWindow, ClockTree       -- single-corner inputs
    PhysicalOption, PhysicalEdge, Corner,
    MultiCornerTree                           -- multi-corner inputs
    Status, SolveResult, PruningInfo          -- single-corner outputs
    MultiCornerStatus, MultiCornerResult,
    CornerReport                              -- multi-corner outputs
    solve, validate_tree                      -- single-corner algorithm
    solve_multicorner,
    validate_multicorner_tree                 -- multi-corner algorithm
    SkewBudgetError and subclasses            -- error semantics
"""

from .model import (
    MAX_CHOICE_POINTS,
    MAX_CORNERS,
    MAX_EDGES,
    MAX_MC_OPTIONS,
    MAX_OPTIONS_PER_EDGE,
    ClockTree,
    Corner,
    Edge,
    InputTypeError,
    InputValueError,
    InvalidNumericError,
    LeafWindow,
    MultiCornerTree,
    Option,
    PhysicalEdge,
    PhysicalOption,
    SkewBudgetError,
    TreeValidationError,
)
from .dp import PruningInfo, SolveResult, Status, solve, validate_tree
from .multicorner import (
    CornerReport,
    MultiCornerResult,
    MultiCornerStatus,
    solve_multicorner,
    validate_multicorner_tree,
)

__all__ = [
    "MAX_EDGES",
    "MAX_OPTIONS_PER_EDGE",
    "MAX_CHOICE_POINTS",
    "MAX_MC_OPTIONS",
    "MAX_CORNERS",
    "ClockTree",
    "Edge",
    "LeafWindow",
    "Option",
    "Corner",
    "PhysicalEdge",
    "PhysicalOption",
    "MultiCornerTree",
    "Status",
    "SolveResult",
    "PruningInfo",
    "MultiCornerStatus",
    "MultiCornerResult",
    "CornerReport",
    "solve",
    "validate_tree",
    "solve_multicorner",
    "validate_multicorner_tree",
    "SkewBudgetError",
    "InputTypeError",
    "InputValueError",
    "InvalidNumericError",
    "TreeValidationError",
]

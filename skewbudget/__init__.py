"""Hardware clock-tree skew-budget allocation (integer time, exact DP).

Public API:
    Option, Edge, LeafWindow, ClockTree -- input data structures
    Status, SolveResult, PruningInfo    -- output data structures
    solve, validate_tree                -- algorithm / eager validation
    SkewBudgetError and subclasses      -- error semantics
"""

from .model import (
    MAX_EDGES,
    ClockTree,
    Edge,
    InputTypeError,
    InputValueError,
    InvalidNumericError,
    LeafWindow,
    Option,
    SkewBudgetError,
    TreeValidationError,
)
from .dp import PruningInfo, SolveResult, Status, solve, validate_tree

__all__ = [
    "MAX_EDGES",
    "ClockTree",
    "Edge",
    "LeafWindow",
    "Option",
    "Status",
    "SolveResult",
    "PruningInfo",
    "solve",
    "validate_tree",
    "SkewBudgetError",
    "InputTypeError",
    "InputValueError",
    "InvalidNumericError",
    "TreeValidationError",
]

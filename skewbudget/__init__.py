"""Hardware clock-tree skew-budget allocation (integer time, exact DP).

Public API:
    Option, Edge, LeafWindow, ClockTree -- input data structures
    Status, SolveResult, PruningInfo    -- output data structures
    solve, validate_tree                -- algorithm / eager validation
    SkewBudgetError and subclasses      -- error semantics
"""

from .model import (
    MAX_EDGES,
    Clock,
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
from .dp import LeafTiming

__all__ = [
    "MAX_EDGES",
    "Clock",
    "ClockTree",
    "Edge",
    "LeafWindow",
    "Option",
    "LeafTiming",
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

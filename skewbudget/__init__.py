"""Hardware clock-tree skew-budget allocation (integer time, exact DP).

Public API:
    Option, Edge, LeafWindow, ClockTree -- input data structures
    Status, SolveResult, PruningInfo, Modification -- output structures
    solve, solve_with_reference, validate_tree     -- algorithms / validation
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
from .dp import (
    Modification,
    PruningInfo,
    SolveResult,
    Status,
    solve,
    solve_with_reference,
    validate_tree,
)

__all__ = [
    "MAX_EDGES",
    "ClockTree",
    "Edge",
    "LeafWindow",
    "Option",
    "Status",
    "SolveResult",
    "PruningInfo",
    "Modification",
    "solve",
    "solve_with_reference",
    "validate_tree",
    "SkewBudgetError",
    "InputTypeError",
    "InputValueError",
    "InvalidNumericError",
    "TreeValidationError",
]

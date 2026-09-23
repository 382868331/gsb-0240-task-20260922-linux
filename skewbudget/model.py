"""Public input/output data structures and error semantics.

All times and costs are *integers*.  Booleans are rejected wherever an
integer is demanded (``bool`` is a subclass of ``int`` in Python, but it is
never a meaningful clock time).  Floats are rejected outright rather than
truncated, so no rounding tolerance is ever needed by callers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MAX_EDGES = 25
MAX_OPTIONS_PER_EDGE = 64
MAX_LABEL_LEN = 64


class SkewBudgetError(Exception):
    """Base class for every error raised by this package."""


class InvalidNumericError(SkewBudgetError, TypeError):
    """A numeric argument is NaN / Infinity / wrong kind of number."""


class InputTypeError(SkewBudgetError, TypeError):
    """An argument has the wrong (non-numeric) type."""


class InputValueError(SkewBudgetError, ValueError):
    """A well-typed argument violates a documented contract."""


class TreeValidationError(SkewBudgetError, ValueError):
    """The tree description is structurally invalid; ``path`` locates it."""

    def __init__(self, message: str, path: str = "<tree>") -> None:
        self.path = path
        super().__init__(f"{path}: {message}")


def _check_int(value, what: str) -> int:
    # bool must be rejected explicitly: isinstance(True, int) is True.
    if isinstance(value, bool):
        raise InputTypeError(f"{what} must be an int, got bool")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != value:  # NaN
            raise InvalidNumericError(f"{what} must be a finite integer, got NaN")
        if value in (float("inf"), float("-inf")):
            raise InvalidNumericError(f"{what} must be a finite integer, got Infinity")
        raise InputTypeError(
            f"{what} must be an int (integer time domain), got float {value!r}"
        )
    raise InputTypeError(f"{what} must be an int, got {type(value).__name__}")


def _check_str(value, what: str, path: str) -> str:
    if not isinstance(value, str):
        raise TreeValidationError(
            f"{what} must be a str, got {type(value).__name__}", path
        )
    if not value:
        raise TreeValidationError(f"{what} must be non-empty", path)
    if len(value) > MAX_LABEL_LEN:
        raise TreeValidationError(
            f"{what} longer than {MAX_LABEL_LEN} characters", path
        )
    return value


@dataclass(frozen=True)
class Option:
    """One discrete buffer choice on an edge.

    delay: added path latency, non-negative integer.
    cost:  buffer cost, non-negative integer.
    The zero-buffer choice is ``Option(0, 0)``.  Duplicate delay values are
    allowed (only the cheapest survives preprocessing); non-convex /
    non-dominatable shapes are preserved.
    """

    delay: int
    cost: int

    def __post_init__(self) -> None:
        d = _check_int(self.delay, "Option.delay")
        c = _check_int(self.cost, "Option.cost")
        if d < 0:
            raise InputValueError(f"Option.delay must be >= 0, got {d}")
        if c < 0:
            raise InputValueError(f"Option.cost must be >= 0, got {c}")
        object.__setattr__(self, "delay", d)
        object.__setattr__(self, "cost", c)


@dataclass(frozen=True)
class Edge:
    """An edge ``parent -> child`` with a base delay and buffer options.

    At least one option is required; choosing none is not a legal action.
    """

    parent: str
    child: str
    base_delay: int
    options: tuple[Option, ...]

    def __init__(
        self,
        parent: str,
        child: str,
        base_delay: int,
        options,
    ) -> None:
        p = _check_str(parent, "Edge.parent", "edge")
        c = _check_str(child, "Edge.child", "edge")
        b = _check_int(base_delay, "Edge.base_delay")
        if b < 0:
            raise InputValueError(f"Edge.base_delay must be >= 0, got {b}")
        if not isinstance(options, (list, tuple)):
            raise InputTypeError(
                "Edge.options must be a list or tuple of Option, got "
                f"{type(options).__name__}"
            )
        if len(options) == 0:
            raise InputValueError("Edge.options must contain at least one Option")
        if len(options) > MAX_OPTIONS_PER_EDGE:
            raise InputValueError(
                f"Edge.options has {len(options)} entries, "
                f"limit is {MAX_OPTIONS_PER_EDGE}"
            )
        opts: list[Option] = []
        seen_delay: dict[int, int] = {}
        for i, opt in enumerate(options):
            if not isinstance(opt, Option):
                raise InputTypeError(
                    f"Edge.options[{i}] must be Option, got {type(opt).__name__}"
                )
            # Duplicate delay: keep cheapest; equal delay+cost: first wins.
            prev = seen_delay.get(opt.delay)
            if prev is None or opt.cost < prev:
                seen_delay[opt.delay] = opt.cost
        # Deterministic public option order: input order, dedup keeping
        # cheapest for a delay, never re-sorted.
        kept: set[int] = set()
        for opt in options:
            if opt.delay in kept:
                continue
            if seen_delay[opt.delay] == opt.cost:
                opts.append(opt)
                kept.add(opt.delay)
        object.__setattr__(self, "parent", p)
        object.__setattr__(self, "child", c)
        object.__setattr__(self, "base_delay", b)
        object.__setattr__(self, "options", tuple(opts))


@dataclass(frozen=True)
class LeafWindow:
    """Closed arrival-time window ``[lo, hi]`` for a leaf (inclusive)."""

    leaf: str
    lo: int
    hi: int

    def __post_init__(self) -> None:
        _check_str(self.leaf, "LeafWindow.leaf", "leaf-window")
        lo = _check_int(self.lo, "LeafWindow.lo")
        hi = _check_int(self.hi, "LeafWindow.hi")
        if lo > hi:
            raise InputValueError(
                f"LeafWindow({self.leaf!r}): lo ({lo}) > hi ({hi})"
            )
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)


@dataclass(frozen=True)
class ClockTree:
    """A rooted tree description.

    root:     label of the root vertex (time 0 reference).
    edges:    every edge of the tree; each child has exactly one parent.
    windows:  one closed arrival window per leaf (non-root degree-0 vertex);
              leaves without a window are rejected.
    """

    root: str
    edges: tuple[Edge, ...]
    windows: tuple[LeafWindow, ...]
    # Populated by validation; internal adjacency.
    children: dict[str, list[tuple[str, Edge]]] = field(default_factory=dict, repr=False)
    parent_edge: dict[str, Edge] = field(default_factory=dict, repr=False)
    leaf_set: frozenset[str] = frozenset()
    # Public canonical edge index: depth-first preorder from root.
    edge_index: dict[str, int] = field(default_factory=dict, repr=False)

    def __init__(self, root, edges, windows) -> None:
        r = _check_str(root, "root", "root")
        if not isinstance(edges, (list, tuple)):
            raise InputTypeError(
                f"edges must be a list/tuple, got {type(edges).__name__}"
            )
        if not isinstance(windows, (list, tuple)):
            raise InputTypeError(
                f"windows must be a list/tuple, got {type(windows).__name__}"
            )
        if len(edges) > MAX_EDGES:
            raise InputValueError(
                f"tree has {len(edges)} edges, limit is {MAX_EDGES}"
            )
        es: list[Edge] = []
        for i, e in enumerate(edges):
            path = f"edges[{i}]"
            if not isinstance(e, Edge):
                raise TreeValidationError(
                    f"must be Edge, got {type(e).__name__}", path
                )
            es.append(e)
        ws: list[LeafWindow] = []
        win_labels: set[str] = set()
        for i, w in enumerate(windows):
            path = f"windows[{i}]"
            if not isinstance(w, LeafWindow):
                raise TreeValidationError(
                    f"must be LeafWindow, got {type(w).__name__}", path
                )
            _check_str(w.leaf, "LeafWindow.leaf", path)
            if w.leaf in win_labels:
                raise TreeValidationError(f"duplicate window for {w.leaf!r}", path)
            win_labels.add(w.leaf)
            ws.append(w)

        object.__setattr__(self, "root", r)
        object.__setattr__(self, "edges", tuple(es))
        object.__setattr__(self, "windows", tuple(ws))
        # Structural checks are deferred so a broken constructor input can
        # never leave a half-built object: build adjacency in locals first.
        children: dict[str, list[tuple[str, Edge]]] = {}
        parent_edge: dict[str, Edge] = {}
        labels: set[str] = {r}
        for i, e in enumerate(es):
            if e.child == r:
                raise TreeValidationError(
                    f"edge {e.parent!r}->{e.child!r} points at the root",
                    f"edges[{i}]",
                )
            if e.child in parent_edge:
                raise TreeValidationError(
                    f"vertex {e.child!r} has two parents "
                    f"({parent_edge[e.child].parent!r}, {e.parent!r})",
                    f"edges[{i}]",
                )
            if e.parent == e.child:
                raise TreeValidationError("self-loop is not allowed", f"edges[{i}]")
            parent_edge[e.child] = e
            children.setdefault(e.parent, []).append((e.child, e))
            labels.add(e.child)
        for i, e in enumerate(es):
            if e.parent not in labels:
                raise TreeValidationError(
                    f"parent {e.parent!r} is not the root and is not any "
                    "edge's child (dangling edge)",
                    f"edges[{i}]",
                )
        # Every non-root label must be reachable from root; parent map +
        # connected labels guarantees a tree iff edge count = vertices - 1.
        if len(es) != len(labels) - 1:
            raise TreeValidationError(
                "edges do not form a single rooted tree "
                f"({len(es)} edges, {len(labels)} labelled vertices)",
                "edges",
            )
        # DFS preorder (children in sorted-label order) establishes the
        # canonical, deterministic edge order; subtree slots stay contiguous.
        order: list[str] = []
        edge_index: dict[str, int] = {}

        def walk(v: str, seen_local: set[str]) -> None:
            for ch, _e in sorted(children.get(v, ()), key=lambda x: x[0]):
                if ch in seen_local:  # defensive: cycles already excluded
                    raise TreeValidationError("cycle detected", "edges")
                seen_local.add(ch)
                edge_index[ch] = len(order)
                order.append(ch)
                walk(ch, seen_local)

        walk(r, {r})
        if len(order) != len(es):
            raise TreeValidationError("not every vertex is reachable from root", "edges")

        leaf_set = frozenset(
            v for v in labels if v != r and not children.get(v)
        )
        missing = sorted(leaf_set - win_labels)
        extra = sorted(win_labels - leaf_set)
        if missing:
            raise TreeValidationError(
                f"leaf(ves) without arrival window: {missing}", "windows"
            )
        if extra:
            raise TreeValidationError(
                f"window(s) on non-leaf or unknown vertex: {extra}", "windows"
            )
        object.__setattr__(self, "children", children)
        object.__setattr__(self, "parent_edge", parent_edge)
        object.__setattr__(self, "leaf_set", leaf_set)
        object.__setattr__(self, "edge_index", edge_index)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def window_of(self, leaf: str) -> LeafWindow:
        for w in self.windows:
            if w.leaf == leaf:
                return w
        raise KeyError(leaf)

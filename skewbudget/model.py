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
# Multi-corner problem: the problem statement bounds the search to small
# trees so exhaustive enumeration is exact -- at most 8 "choice points"
# (edges with >= 2 physical options), at most 3 options at each, and a
# small fixed corner count.
MAX_CHOICE_POINTS = 8
MAX_MC_OPTIONS = 3
MAX_CORNERS = 8


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


def _validate_topology(root: str, edges, win_labels=None):
    """Validate that ``edges`` form one rooted tree at ``root``.

    Returns ``(children, parent_edge, leaf_set, edge_index, labels)``.
    ``win_labels`` (a set, or ``None`` to skip window placement checks)
    must contain exactly the leaves.  Error messages / ``path`` tags match
    the documented single-corner contract so both tree kinds raise
    identically located errors.
    """
    children: dict[str, list[tuple[str, Edge]]] = {}
    parent_edge: dict[str, Edge] = {}
    labels: set[str] = {root}
    for i, e in enumerate(edges):
        if e.child == root:
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
    for i, e in enumerate(edges):
        if e.parent not in labels:
            raise TreeValidationError(
                f"parent {e.parent!r} is not the root and is not any "
                "edge's child (dangling edge)",
                f"edges[{i}]",
            )
    # Every non-root label must be reachable from root; parent map +
    # connected labels guarantees a tree iff edge count = vertices - 1.
    if len(edges) != len(labels) - 1:
        raise TreeValidationError(
            "edges do not form a single rooted tree "
            f"({len(edges)} edges, {len(labels)} labelled vertices)",
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

    walk(root, {root})
    if len(order) != len(edges):
        raise TreeValidationError("not every vertex is reachable from root", "edges")

    leaf_set = frozenset(v for v in labels if v != root and not children.get(v))
    if win_labels is not None:
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
    return children, parent_edge, leaf_set, edge_index, labels


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
        # Structural checks run only after every input object has been
        # accepted, so a broken constructor input can never leave a
        # half-built object.
        children, parent_edge, leaf_set, edge_index, _labels = (
            _validate_topology(r, es, win_labels=win_labels)
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


# ---------------------------------------------------------------------------
# Multi-corner model: ONE physical buffer option per edge, with a *different*
# realised delay in each operating corner.  The chosen option vector must
# satisfy every corner's leaf windows and skew limit simultaneously.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhysicalOption:
    """One physical buffer choice with one realised delay per corner.

    cost:   physical cost, paid once regardless of how many corners use it.
    delays: per-corner added latency, aligned position-for-position with the
            owning :class:`MultiCornerTree` corner list (non-negative ints).
    """

    cost: int
    delays: tuple[int, ...]

    def __init__(self, cost, delays) -> None:
        c = _check_int(cost, "PhysicalOption.cost")
        if c < 0:
            raise InputValueError(f"PhysicalOption.cost must be >= 0, got {c}")
        if not isinstance(delays, (list, tuple)):
            raise InputTypeError(
                "PhysicalOption.delays must be a list/tuple of int, got "
                f"{type(delays).__name__}"
            )
        if len(delays) == 0:
            raise InputValueError(
                "PhysicalOption.delays must contain one delay per corner"
            )
        if len(delays) > MAX_CORNERS:
            raise InputValueError(
                f"PhysicalOption.delays has {len(delays)} entries, "
                f"limit is {MAX_CORNERS} corners"
            )
        ds: list[int] = []
        for i, d in enumerate(delays):
            di = _check_int(d, f"PhysicalOption.delays[{i}]")
            if di < 0:
                raise InputValueError(
                    f"PhysicalOption.delays[{i}] must be >= 0, got {di}"
                )
            ds.append(di)
        object.__setattr__(self, "cost", c)
        object.__setattr__(self, "delays", tuple(ds))


@dataclass(frozen=True)
class PhysicalEdge:
    """An edge ``parent -> child`` with per-corner base delays.

    base_delays: one base latency per corner (aligned with the corner list).
    options:     1..MAX_MC_OPTIONS physical options; every option must carry
                 one delay per corner.  Options with an identical per-corner
                 delay vector are de-duplicated keeping the cheapest (equal
                 cost keeps the first), exactly like single-corner edges.
    """

    parent: str
    child: str
    base_delays: tuple[int, ...]
    options: tuple[PhysicalOption, ...]

    def __init__(self, parent, child, base_delays, options) -> None:
        p = _check_str(parent, "PhysicalEdge.parent", "edge")
        c = _check_str(child, "PhysicalEdge.child", "edge")
        if not isinstance(base_delays, (list, tuple)):
            raise InputTypeError(
                "PhysicalEdge.base_delays must be a list/tuple of int, got "
                f"{type(base_delays).__name__}"
            )
        if len(base_delays) == 0:
            raise InputValueError(
                "PhysicalEdge.base_delays must contain one delay per corner"
            )
        if len(base_delays) > MAX_CORNERS:
            raise InputValueError(
                f"PhysicalEdge.base_delays has {len(base_delays)} entries, "
                f"limit is {MAX_CORNERS} corners"
            )
        bd: list[int] = []
        for i, b in enumerate(base_delays):
            bi = _check_int(b, f"PhysicalEdge.base_delays[{i}]")
            if bi < 0:
                raise InputValueError(
                    f"PhysicalEdge.base_delays[{i}] must be >= 0, got {bi}"
                )
            bd.append(bi)
        if not isinstance(options, (list, tuple)):
            raise InputTypeError(
                "PhysicalEdge.options must be a list or tuple of "
                f"PhysicalOption, got {type(options).__name__}"
            )
        if len(options) == 0:
            raise InputValueError(
                "PhysicalEdge.options must contain at least one PhysicalOption"
            )
        if len(options) > MAX_MC_OPTIONS:
            raise InputValueError(
                f"PhysicalEdge.options has {len(options)} entries, "
                f"limit is {MAX_MC_OPTIONS} (enumeration bound)"
            )
        n = len(bd)
        opts: list[PhysicalOption] = []
        for i, opt in enumerate(options):
            if not isinstance(opt, PhysicalOption):
                raise InputTypeError(
                    "PhysicalEdge.options[" + str(i) + "] must be "
                    f"PhysicalOption, got {type(opt).__name__}"
                )
            if len(opt.delays) != n:
                raise InputValueError(
                    f"PhysicalEdge.options[{i}] has {len(opt.delays)} corner "
                    f"delays but base_delays has {n}"
                )
            opts.append(opt)
        # De-duplicate identical per-corner delay vectors, cheapest first.
        cheapest: dict[tuple[int, ...], int] = {}
        for opt in opts:
            prev = cheapest.get(opt.delays)
            if prev is None or opt.cost < prev:
                cheapest[opt.delays] = opt.cost
        kept: list[PhysicalOption] = []
        seen: set[tuple[int, ...]] = set()
        for opt in opts:
            if opt.delays in seen:
                continue
            if cheapest[opt.delays] == opt.cost:
                kept.append(opt)
                seen.add(opt.delays)
        object.__setattr__(self, "parent", p)
        object.__setattr__(self, "child", c)
        object.__setattr__(self, "base_delays", tuple(bd))
        object.__setattr__(self, "options", tuple(kept))


@dataclass(frozen=True)
class Corner:
    """One operating corner: a skew limit and one closed window per leaf."""

    name: str
    skew_limit: int
    windows: tuple[LeafWindow, ...]

    def __init__(self, name, skew_limit, windows) -> None:
        n = _check_str(name, "Corner.name", "corner")
        s = _check_int(skew_limit, "Corner.skew_limit")
        if s < 0:
            raise InputValueError(
                f"Corner.skew_limit must be >= 0, got {s}"
            )
        if not isinstance(windows, (list, tuple)):
            raise InputTypeError(
                f"Corner.windows must be a list/tuple, got "
                f"{type(windows).__name__}"
            )
        ws: list[LeafWindow] = []
        labels: set[str] = set()
        for i, w in enumerate(windows):
            if not isinstance(w, LeafWindow):
                raise InputTypeError(
                    f"Corner.windows[{i}] must be LeafWindow, "
                    f"got {type(w).__name__}"
                )
            if w.leaf in labels:
                raise InputValueError(
                    f"corner {n!r}: duplicate window for leaf {w.leaf!r}"
                )
            labels.add(w.leaf)
            ws.append(w)
        object.__setattr__(self, "name", n)
        object.__setattr__(self, "skew_limit", s)
        object.__setattr__(self, "windows", tuple(ws))


@dataclass(frozen=True)
class MultiCornerTree:
    """A rooted tree plus a list of operating corners.

    Exactly one physical option is selected per edge; its per-corner delays
    apply corner by corner.  Enumeration is exhaustive, so the number of
    *choice points* (edges with >= 2 options) is capped at
    :data:`MAX_CHOICE_POINTS`; edges with a single forced option do not
    count.  Every corner must define exactly one window per leaf.
    """

    root: str
    edges: tuple[PhysicalEdge, ...]
    corners: tuple[Corner, ...]
    children: dict[str, list[tuple[str, PhysicalEdge]]] = field(
        default_factory=dict, repr=False)
    parent_edge: dict[str, PhysicalEdge] = field(default_factory=dict, repr=False)
    leaf_set: frozenset[str] = frozenset()
    edge_index: dict[str, int] = field(default_factory=dict, repr=False)
    # Edges with >= 2 options, in canonical DFS-preorder edge order.
    choice_edge_order: tuple[str, ...] = ()

    def __init__(self, root, edges, corners) -> None:
        r = _check_str(root, "root", "root")
        if not isinstance(edges, (list, tuple)):
            raise InputTypeError(
                f"edges must be a list/tuple, got {type(edges).__name__}"
            )
        if not isinstance(corners, (list, tuple)):
            raise InputTypeError(
                f"corners must be a list/tuple, got {type(corners).__name__}"
            )
        if len(edges) > MAX_EDGES:
            raise InputValueError(
                f"tree has {len(edges)} edges, limit is {MAX_EDGES}"
            )
        if len(corners) == 0:
            raise InputValueError("corners must contain at least one Corner")
        if len(corners) > MAX_CORNERS:
            raise InputValueError(
                f"{len(corners)} corners given, limit is {MAX_CORNERS}"
            )
        es: list[PhysicalEdge] = []
        for i, e in enumerate(edges):
            if not isinstance(e, PhysicalEdge):
                raise TreeValidationError(
                    f"must be PhysicalEdge, got {type(e).__name__}",
                    f"edges[{i}]",
                )
            es.append(e)
        cs: list[Corner] = []
        corner_names: set[str] = set()
        for i, c in enumerate(corners):
            path = f"corners[{i}]"
            if not isinstance(c, Corner):
                raise TreeValidationError(
                    f"must be Corner, got {type(c).__name__}", path
                )
            if c.name in corner_names:
                raise TreeValidationError(
                    f"duplicate corner name {c.name!r}", path
                )
            corner_names.add(c.name)
            cs.append(c)

        n_corners = len(cs)
        for i, e in enumerate(es):
            if len(e.base_delays) != n_corners:
                raise TreeValidationError(
                    f"base_delays has {len(e.base_delays)} entries but tree "
                    f"has {n_corners} corners",
                    f"edges[{i}]",
                )
            for j, opt in enumerate(e.options):
                if len(opt.delays) != n_corners:
                    raise TreeValidationError(
                        f"option {j} has {len(opt.delays)} corner delays but "
                        f"tree has {n_corners} corners",
                        f"edges[{i}]",
                    )

        object.__setattr__(self, "root", r)
        object.__setattr__(self, "edges", tuple(es))
        object.__setattr__(self, "corners", tuple(cs))
        # Topology first (no windows are embedded in edges here)...
        children, parent_edge, leaf_set, edge_index, _labels = (
            _validate_topology(r, es, win_labels=None)
        )
        # ...then each corner must place exactly one closed window per leaf.
        for k, c in enumerate(cs):
            labels_here = {w.leaf for w in c.windows}
            missing = sorted(leaf_set - labels_here)
            extra = sorted(labels_here - leaf_set)
            path = f"corners[{k}].windows"
            if missing:
                raise TreeValidationError(
                    f"corner {c.name!r}: leaf(ves) without arrival window: "
                    f"{missing}", path
                )
            if extra:
                raise TreeValidationError(
                    f"corner {c.name!r}: window(s) on non-leaf or unknown "
                    f"vertex: {extra}", path
                )
        choice_points = sum(1 for e in es if len(e.options) > 1)
        if choice_points > MAX_CHOICE_POINTS:
            raise InputValueError(
                f"{choice_points} choice points (edges with >1 option), "
                f"enumeration limit is {MAX_CHOICE_POINTS}"
            )
        choice_order = tuple(
            ch for ch, _idx in sorted(edge_index.items(), key=lambda kv: kv[1])
            if len(parent_edge[ch].options) > 1
        )
        object.__setattr__(self, "children", children)
        object.__setattr__(self, "parent_edge", parent_edge)
        object.__setattr__(self, "leaf_set", leaf_set)
        object.__setattr__(self, "edge_index", edge_index)
        object.__setattr__(self, "choice_edge_order", choice_order)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def n_corners(self) -> int:
        return len(self.corners)

    @property
    def choice_points(self) -> int:
        return len(self.choice_edge_order)

    def window_map(self, corner_index: int) -> dict[str, LeafWindow]:
        return {w.leaf: w for w in self.corners[corner_index].windows}

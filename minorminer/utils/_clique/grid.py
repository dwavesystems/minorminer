# Copyright 2026 D-Wave
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


"""
Grid: a (possibly faulty) Zephyr topology surveyed into fast-access structures.

Vocabulary
----------
Throughout this module (and the rest of the clique embedder) coordinates are Zephyr Cartesian
coordinates — plain ``(x, y, k)`` tuples following the convention in
:meth:`dwave.graphs.zephyr_coordinates.zephyr_to_cartesian`

Naming: a "quo"/"quotient" prefix means ``k``-agnostic -- the object with its ``k`` index folded
out, standing for every ``k`` at once. Terms carrying it (quo external path, quo_span below) are
``k``-agnostic by definition, so their entries do not repeat it.

node / qubit      ``(x, y, k)``          a Zephyr node in Cartesian coordinates, ``k in range(t)``
block             ``(x, y)``             quotient node (ignores ``k``)

Orientation is fixed by parity: a vertical node has ``x`` even and ``y`` odd; a horizontal node has
``x`` odd and ``y`` even.

At a fixed even ``x`` the vertical nodes have odd ``y``, so ``y % 4`` is 1 or 3: the two interleaved
vertical kinds v1 and v3, offset by 2 in ``y`` and each stepping by 4. Likewise at a fixed even
``y`` the two horizontal kinds h1 and h3 have ``x % 4`` equal to 1 or 3. In each case, the remainder
of the odd coordinate of a block by 4 is called the shift. This orientation-plus-shift label is a
node's kind.

kind              "v1"|"v3"|"h1"|"h3"    orientation + which of the two shifts
                  vertical  : ``x`` even, ``y`` odd, ``shift = y % 4`` (so ``shift in {1, 3}``)
                  horizontal: ``x`` odd,  ``y`` even, ``shift = x % 4`` (so ``shift in {1, 3}``)

An external path through a node is a path that contains it and uses only external couplers. It is
therefore either vertical -- through nodes sharing ``x = fixed_coord`` and ``k``, with ``y``
differing by multiples of 4 -- or horizontal -- through nodes sharing ``y = fixed_coord`` and ``k``,
with ``x`` differing by multiples of 4. In an ideal Zephyr all nodes sharing a
``(kind, fixed_coord, k)`` lie on one external path; in a faulty one, yield loss (missing nodes or
external couplers) can break that path into pieces. Each maximal such piece is a RUN.

quo external path ``(kind, fixed_coord)``  an external path with ``k`` folded out;
                                         two per column/row (the two shifts)
run               ``(start, end)``         a MAXIMAL external path at a given
                                         ``(kind, fixed_coord, k)``; grid.runs holds these
quo_span          ``(fixed, shift, a, b)`` a span ``[a, b]`` on a quo external path that a
                                         chain requires -- not necessarily maximal;
                                         realized when some run covers it (_covers /
                                         el_reachable)
elbow             elbow(v, h): the pair ``(vp, hp)`` with vp on v's external path and
                  hp on h's external path (ideal Zephyr), joined by an internal edge --
                  where the el-path hops from v's path to h's. ``vp = (v_x, vp_y)``,
                  ``hp = (hp_x, h_y)``; returned as ``(vp_y, hp_x)``.
el_template       ``(v_quo_span, h_quo_span)`` quotient recipe for an L-shaped chain
chain             an instantiated path (el_template + chosen ``v_k``, ``h_k``)
el_reachable      method ``el_reachable(v, h) -> (v_quo_span, h_quo_span, v_k, h_k)`` or None:
                  whether the L-shaped v->h path exists in the faulty grid

Construction
------------
Grid.from_graph(graph)      builds + runs the full survey, caches everything.

After from_graph, these attributes are available:
  .present_qubits[kind]       ``{(x, y, k): r}``               present qubits -> linear index
  .missing_qubits[kind]       frozenset of ``(x, y, k)``       absent qubits
  .edges                      set of ``(lo, hi)``              present couplers, r-index pairs
  .missing_internal_couplers  ``{(v, hp): (v_base, h_base)}``  missing internal couplers
  .runs               ``runs[kind][coord][k] -> {(start, end), ...}``
  .pos                position_quo result: ``{side: {(x, y): {orig_k: pos}}}``

Geometry methods (ideal el-template geometry, computed on demand; a query is a few arithmetic ops):
  .el_template_length(v_quo, h_quo) -> chain_length, or None if not an el_template
  .el_reachable(v, h)             -> ``(v_quo_span, h_quo_span, v_k, h_k)`` or None
el_reachable is memoized per grid (see its docstring).

Performance notes
-----------------
- from_graph runs the full survey once. The L-shaped reachability between blocks is resolved per
  pair on demand by el_reachable and cached, not surveyed.
- The ideal el-template geometry (elbows, el_template, lengths) depends only on ``m`` (independent
  of which nodes/edges are faulty) and is computed on demand through the methods above. Every result
  is ``O(1)`` arithmetic from its key; el_reachable's own memo keeps repeated queries cheap.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from itertools import product
from operator import itemgetter
from typing import TYPE_CHECKING

from minorminer.utils._clique import el_geometry
from minorminer.utils._clique.el_geometry import QuoSpan

if TYPE_CHECKING:
    import networkx as nx


# --- type aliases -----------------------------------------------------------
Node = tuple[int, int, int]  # cartesian qubit (x, y, k)
ZCoord = tuple[int, int, int, int, int]  # Zephyr coordinate (u, w, k, j, z)
Block = tuple[int, int]  # quotient node (x, y), k ignored
Run = tuple[int, int]  # an intact stretch (start, end) on a line
# runs[kind][coord][k] -> {(start, end), ...}
Runs = dict[str, dict[int, dict[int, set[Run]]]]
# pos[side][(x, y)][orig_k] -> position 2-tuple
Pos = dict[str, dict[Block, dict[int, tuple[int, int]]]]

_KINDS = ("v1", "v3", "h1", "h3")
_MISSING = object()  # el_reachable cache: .get() default only, never stored -> distinguishes
# "not computed" from a cached result (which may be None = unreachable)


def zephyr_to_cartesian(zcoord: ZCoord) -> Node:
    """Convert a Zephyr coordinate ``(u, w, k, j, z)`` to cartesian ``(x, y, k)``.

    ``u = 0`` is a vertical qubit, ``u = 1`` horizontal. Assumes a valid Zephyr coordinate.
    """
    u, w, k, j, z = zcoord
    if u == 0:
        x = 2 * w
        y = 4 * z + 2 * j + 1
    else:
        x = 4 * z + 2 * j + 1
        y = 2 * w
    return (x, y, k)


def cartesian_to_zephyr(ccoord: Node) -> ZCoord:
    """Convert a cartesian coordinate ``(x, y, k)`` to Zephyr ``(u, w, k, j, z)``.

    ``x`` even -> vertical (``u = 0``); ``x`` odd -> horizontal (``u = 1``). Assumes a valid
    cartesian coordinate. Inverse of zephyr_to_cartesian.
    """
    x, y, k = ccoord
    if x % 2 == 0:
        u = 0
        w = x // 2
        j = ((y - 1) % 4) // 2
        z = y // 4
    else:
        u = 1
        w = y // 2
        j = ((x - 1) % 4) // 2
        z = x // 4
    return (u, w, k, j, z)


class Grid:
    """A (possibly faulty) Zephyr chip surveyed once at construction into fast lookup structures.

    "Faulty" means some qubits or couplers may be absent; "surveyed" means from_graph walks the
    whole graph a single time and precomputes the tables (present/missing qubits, couplers, per-line
    runs, ...) that the rest of the algorithm queries, rather than re-walking the graph on every
    lookup.

    Build with Grid.from_graph(graph). See the module docstring for the full list of attributes and
    the vocabulary.
    """

    # attribute types (bare annotations; the storage is __slots__ below)
    m: int
    t: int
    abs_min: int
    abs_max: int
    labels: str  # output mode: "int"/"coordinates"/"cartesian"
    present_qubits: dict[str, dict[Node, int]]  # present_qubits[kind][node] -> linear index r
    missing_qubits: dict[str, frozenset[Node] | set[Node]]
    edges: set[tuple[int, int]] | None  # present couplers as (lo, hi) r-pairs
    missing_internal_couplers: dict | None  # see _missing_internal_couplers
    runs: Runs | None
    _el_cache: dict  # el_reachable memo
    pos: Pos | None
    __slots__ = (
        "m",
        "t",
        "abs_min",
        "abs_max",
        "labels",
        "present_qubits",
        "missing_qubits",
        "edges",
        "missing_internal_couplers",
        "runs",
        "_el_cache",
        "pos",
    )

    # ------------------------------------------------------------------ build
    def __init__(self, m: int, t: int) -> None:
        """Create an empty Grid for a tile-size-``t`` Zephyr of grid size ``m``.

        Coordinates range over ``[0, 4m]``. This only allocates the empty containers; from_graph
        calls _classify and _run_survey to populate them. Prefer the from_graph factory.
        """
        self.m = m
        self.t = t
        self.abs_min = 0
        self.abs_max = 4 * m
        self.labels = "int"  # output mode: "int", "coordinates", or "cartesian"
        self.present_qubits = {k: {} for k in _KINDS}
        self.missing_qubits = {k: set() for k in _KINDS}
        self.edges = None
        # survey outputs (filled by _run_survey)
        self.missing_internal_couplers = None
        self.runs = None
        self._el_cache = {}
        self.pos = None

    @classmethod
    def from_graph(cls, graph: nx.Graph) -> Grid:
        """Build a Grid from a networkx Zephyr graph and run the full survey.

        The graph must carry Zephyr topology metadata in graph.graph: family == "zephyr", rows (=
        ``m``) and tile (= ``t``). Raises ValueError if the topology is not zephyr.

        The label mode is inferred from the node type, not from graph metadata. Integer nodes are
        linear indices (as a D-Wave sampler's nodelist), matched directly against the linear-index
        lattice walk. 5-tuple nodes are Zephyr coordinates ``(u, w, k, j, z)``; each is converted to
        cartesian ``(x, y, k)``, then to its linear index r via the walk. 3-tuple nodes are already
        cartesian ``(x, y, k)``.

        Either way the internal representation is identical (cartesian nodes, linear-index r,
        ``(r, r)`` edges); only find_clique's OUTPUT format follows the inferred mode (linear
        indices for "int", Zephyr coordinates for "coordinates", cartesian for "cartesian").
        """
        info = graph.graph
        family = info.get("family")
        if family != "zephyr":
            raise ValueError(f"Expected a graph with zephyr topology, got family={family!r}")
        m = info.get("rows")
        if m is None:
            m = info.get("columns")
        t = info.get("tile")
        if m is None or t is None:
            raise ValueError("zephyr graph missing 'rows'/'columns'/'tile' metadata")

        # Decide the label mode from the ACTUAL node type/shape, which is
        # unambiguous, rather than trusting graph.graph["labels"] (dwave_networkx
        # spells it "coordinate"; other producers may differ or omit it). Integer
        # nodes -> linear-index mode; a 5-tuple (Zephyr u,w,k,j,z) -> zephyr
        # "coordinates" mode; a 3-tuple (cartesian x,y,k) -> "cartesian" mode.
        sample = next(iter(graph.nodes()))
        if isinstance(sample, tuple):
            if len(sample) == 5:
                labels = "coordinates"  # Zephyr 5-tuple
            elif len(sample) == 3:
                labels = "cartesian"  # cartesian (x, y, k)
            else:
                raise ValueError(
                    f"cannot infer zephyr label mode from tuple node {sample!r}; "
                    f"expected a 5-tuple (Zephyr) or 3-tuple (cartesian)"
                )
        elif isinstance(sample, (int,)) and not isinstance(sample, bool):
            labels = "int"
        else:
            raise ValueError(
                f"cannot infer zephyr label mode from node {sample!r}; expected "
                f"an int (linear), a 5-tuple (Zephyr), or a 3-tuple (cartesian)"
            )

        grid = cls(m, t)
        grid.labels = labels

        if labels == "int":
            present_r = set(graph.nodes())
            grid._classify(present_r)
            grid.edges = {(a, b) if a < b else (b, a) for a, b in graph.edges()}
        elif labels == "cartesian":  # nodes are cartesian (x, y, k) already
            grid._classify_from_present_nodes(set(graph.nodes()))
            # cartesian edges -> linear r pairs (internal edge set stays linear)
            edges = set()
            for ccoord_a, ccoord_b in graph.edges():
                lcoord_a = grid.cartesian_to_linear(ccoord_a)
                lcoord_b = grid.cartesian_to_linear(ccoord_b)
                if lcoord_a is None or lcoord_b is None:
                    continue
                edges.add((lcoord_a, lcoord_b) if lcoord_a < lcoord_b else (lcoord_b, lcoord_a))
            grid.edges = edges
        else:  # coordinates: nodes are Zephyr 5-tuples
            grid._classify_from_present_nodes({zephyr_to_cartesian(z) for z in graph.nodes()})
            # convert coordinate edges -> cartesian -> linear r pairs
            edges = set()
            for zcoord_a, zcoord_b in graph.edges():
                lcoord_a = grid.cartesian_to_linear(zephyr_to_cartesian(zcoord_a))
                lcoord_b = grid.cartesian_to_linear(zephyr_to_cartesian(zcoord_b))
                if lcoord_a is None or lcoord_b is None:
                    continue
                edges.add((lcoord_a, lcoord_b) if lcoord_a < lcoord_b else (lcoord_b, lcoord_a))
            grid.edges = edges

        grid._run_survey()
        return grid

    def _classify(self, present_r: set[int]) -> None:
        """Sort every ideal-lattice node into its kind bucket, in linear-index order.

        Walks the full ideal lattice, filing each node under present_qubits[kind] or
        missing_qubits[kind] and recording the r -> cartesian map. A node is present iff its linear
        index r is in present_r.

        The walk order is what DEFINES r, and it must match the sampler's own indexing exactly --
        verticals first (``x = 0, 2, ...``; for each ``k``, the ``y % 4 == 1`` nodes then the
        ``y % 4 == 3`` nodes), then horizontals (``y = 0, 2, ...``; for each ``k``, ``x % 4 == 1``
        then ``x % 4 == 3``). Do not reorder without re-verifying r.
        """
        self._walk(lambda r, node: r in present_r)

    def _classify_from_present_nodes(self, present_nodes: set[Node]) -> None:
        """Classify by cartesian node membership instead of by linear index.

        Same lattice walk / r numbering as _classify, but a node is present iff its cartesian
        ``(x, y, k)`` is in present_nodes (used for coordinate-labeled graphs, where presence is
        known by node, not by linear index).
        """
        self._walk(lambda r, node: node in present_nodes)

    def _walk(self, is_present: Callable[[int, Node], bool]) -> None:
        """Assign linear index r to every ideal node and file it as present or missing.

        Shared by both classify entry points: walks the canonical order and applies the
        is_present(r, node) predicate. Populates present_qubits[kind] and finalizes
        missing_qubits[kind] as frozensets (membership tests are the only downstream use).
        """
        m, t = self.m, self.t
        num_w = 2 * m + 1
        present, missing = self.present_qubits, self.missing_qubits
        r = 0
        for w in range(num_w):
            x = 2 * w
            for k in range(t):
                for shift, kind in ((1, "v1"), (3, "v3")):
                    kind_present, kind_missing = present[kind], missing[kind]
                    for z in range(m):
                        node = (x, 4 * z + shift, k)
                        if is_present(r, node):
                            kind_present[node] = r
                        else:
                            kind_missing.add(node)
                        r += 1
        for w in range(num_w):
            y = 2 * w
            for k in range(t):
                for shift, kind in ((1, "h1"), (3, "h3")):
                    kind_present, kind_missing = present[kind], missing[kind]
                    for z in range(m):
                        node = (4 * z + shift, y, k)
                        if is_present(r, node):
                            kind_present[node] = r
                        else:
                            kind_missing.add(node)
                        r += 1

        # freeze the missing buckets as soon as they're built: membership tests are
        # the only downstream use, and frozenset makes the immutability explicit
        self.missing_qubits = {k: frozenset(s) for k, s in self.missing_qubits.items()}

    def _run_survey(self) -> None:
        """Run every survey pass in dependency order and cache the results.

        Order matters: missing_internal_couplers feeds el_reachable's reachability checks;
        survey_ext's runs feed both el_reachable and position_quo. The ideal el-template geometry is
        fault-independent, so it is not surveyed here; the el_template_length / el_reachable methods
        compute it on demand.
        """
        self.missing_internal_couplers = self._missing_internal_couplers()
        self.runs = self._survey_ext()
        self.pos = self._position_quo()

    # -------------------------------------------------------------- accessors
    def kind_of(self, ccoord: Node) -> str:
        """Return the kind ("v1"/"v3"/"h1"/"h3") of a cartesian coord from its parity.

        ``x`` even -> vertical (``v1 if y % 4 == 1 else v3``); ``x`` odd -> horizontal
        (``h1 if x % 4 == 1 else h3``). Pure coordinate arithmetic; no lookup.
        """
        x, y, _ = ccoord
        if (x & 1) == 0:
            return "v1" if (y & 3) == 1 else "v3"
        return "h1" if (x & 3) == 1 else "h3"

    def cartesian_to_linear(self, ccoord: Node) -> int | None:
        """Linear index (lcoord) of a present cartesian coord, or None if absent/missing."""
        return self.present_qubits[self.kind_of(ccoord)].get(ccoord)

    def is_present(self, ccoord: Node) -> bool:
        """True if the cartesian coord exists in the (faulty) grid."""
        return ccoord in self.present_qubits[self.kind_of(ccoord)]

    def has_edge(self, lcoord1: int, lcoord2: int) -> bool:
        """True if a coupler exists between linear indices lcoord1 and lcoord2.

        Order-independent.
        """
        e = (lcoord1, lcoord2) if lcoord1 < lcoord2 else (lcoord2, lcoord1)
        return e in self.edges

    # ------------------------------------------------- el-template geometry
    # Ideal, fault-independent geometry, computed on demand.

    def el_template_length(self, v_quo: QuoSpan, h_quo: QuoSpan) -> int | None:
        """chain_length of the el_template ``(v_quo, h_quo)``, or None if it is not a real one.

        (The None case doubles as the el_template test used by expand_el_templates.)
        """
        return el_geometry.el_template_length(self.m, v_quo, h_quo)

    # ----------------------------------------------------------- survey passes
    def _missing_internal_couplers(
        self,
    ) -> dict[tuple[Node, Node], tuple[tuple[int, int, int], tuple[int, int, int]]]:
        """Find internal couplers that SHOULD exist (ideal geometry) but are absent.

        Internal couplers join a vertical node ``v = (x, y, k)`` to a horizontal node
        ``hp = (x+-1, y+-1, kp)`` for every ``kp in range(t)``. For each present vertical node and
        each of its four diagonal horizontal neighbours (in bounds, present), we check whether the
        coupler exists in `edges`; if not, record it. Returns ``{(v, hp): (v_base, hp_base)}`` where
        the values are the quotient (``k``-folded) descriptors ``(x, v_shift, k)`` and
        ``(hp_shift, hp_y, kp)``.
        """
        abs_min, abs_max, t = self.abs_min, self.abs_max, self.t
        edges, present, missing = self.edges, self.present_qubits, self.missing_qubits
        H_KIND = {1: "h1", 3: "h3"}
        krange = range(t)
        OFFS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
        missing_couplers = {}
        for v_shift, v_kind in ((1, "v1"), (3, "v3")):
            for v, v_r in present[v_kind].items():
                x, y, k = v
                for dx, dy in OFFS:
                    hp_x = x + dx
                    if hp_x < abs_min or hp_x > abs_max:
                        continue
                    hp_y = y + dy
                    if hp_y < abs_min or hp_y > abs_max:
                        continue
                    hp_shift = hp_x & 3
                    hp_nodes = present[H_KIND[hp_shift]]
                    hp_missing = missing[H_KIND[hp_shift]]
                    v_base = (x, v_shift, k)
                    # internal couplers are all-to-all in k: v's k couples to
                    # every hp kp, so there is no k == kp guard here.
                    for kp in krange:
                        hp = (hp_x, hp_y, kp)
                        if hp in hp_missing:
                            continue
                        hp_r = hp_nodes[hp]
                        coupler = (v_r, hp_r) if v_r < hp_r else (hp_r, v_r)
                        if coupler not in edges:
                            missing_couplers[(v, hp)] = (v_base, (hp_shift, hp_y, kp))
        return missing_couplers

    def _survey_ext(self) -> Runs:
        """Walk every external line and split it into RUNS (maximal intact stretches).

        An external line is ``(kind, fixed_coord)``; nodes on it step by 4 in the varying
        coordinate. Starting from each present node, we walk forward while the next node exists and
        the connecting external coupler is present; a break (missing node or missing coupler) ends
        the current run and starts scanning for the next one.

        Returns runs, where ``runs[kind][coord][k]`` -> set of ``(start, end)`` run intervals. This
        nested shape mirrors how nodes are keyed (``kind -> coord -> k``) and is what el_reachable
        and position_quo consume.
        """
        abs_max, t, m = self.abs_max, self.t, self.m
        edges, present, missing = self.edges, self.present_qubits, self.missing_qubits
        runs = {k: {} for k in _KINDS}
        for direction, shift in product(("v", "h"), (1, 3)):
            kind = direction + str(shift)
            kind_nodes, kind_missing = present[kind], missing[kind]
            is_vertical = direction == "v"
            for fixed_coord in range(0, 4 * m + 1, 2):
                coord_map = runs[kind].setdefault(fixed_coord, {})
                for k in range(t):
                    runs_for_k = set()
                    coord = shift
                    while coord <= abs_max:
                        if is_vertical:
                            node = (fixed_coord, coord, k)
                        else:
                            node = (coord, fixed_coord, k)
                        if node not in kind_missing:
                            prev = coord
                            inc = 4
                            end = prev
                            next_coord = None
                            while prev + inc <= abs_max:
                                if is_vertical:
                                    previous_node = (fixed_coord, prev, k)
                                    next_node = (fixed_coord, prev + inc, k)
                                else:
                                    previous_node = (prev, fixed_coord, k)
                                    next_node = (prev + inc, fixed_coord, k)
                                if next_node in kind_missing:
                                    # next node is a hole: end the run and resume
                                    # PAST it (a missing node can't start a run).
                                    end, next_coord = prev, prev + 2 * inc
                                    break
                                if (kind_nodes[previous_node], kind_nodes[next_node]) in edges:
                                    prev += inc
                                else:
                                    # coupler is broken but next_node exists: end here
                                    # and resume AT it, which starts the next run.
                                    end, next_coord = prev, prev + inc
                                    break
                            else:
                                end, next_coord = prev, None
                            runs_for_k.add((coord, end))
                            if next_coord is None:
                                break
                            coord = next_coord
                        else:
                            coord += 4
                    coord_map[k] = runs_for_k
        return runs

    def el_reachable(self, v: Node, h: Node) -> tuple[QuoSpan, QuoSpan, int, int] | None:
        """Does the L-shaped ("el") shortest path from v to h exist in the faulty grid?

        The el-path travels along v's external line to the elbow with h's external line, hops one
        internal coupler (between vp and hp), then travels along h's line to h. It exists iff:
          * the block pair ``(v_block, h_block)`` has an el_template (i.e. the two lines cross in
            the ideal grid),
          * none of the four involved nodes v, vp, h, hp are missing,
          * the connecting internal coupler is not in missing_internal_couplers,
          * both required quotient runs are actually covered by real runs for the chosen ``v_k`` /
            ``h_k`` in the faulty grid.

        Lazy + memoized. The block pair determines the el_template uniquely; the el_template is
        computed on demand, so a query is a handful of arithmetic ops. Results (including ``None``,
        stored as a sentinel) are cached, so repeated queries across overlapping windows in a sweep
        are effectively free. Returns the reachability descriptor.
        Returns ``(v_quo_span, h_quo_span, v_k, h_k)`` if reachable, else None.
        """
        cache = self._el_cache
        hit = cache.get((v, h), _MISSING)
        if hit is not _MISSING:  # cached: a descriptor tuple, or None (unreachable)
            return hit

        v_x, v_y, v_k = v
        h_x, h_y, h_k = h
        template = el_geometry.el_template(self.m, v_x, v_y, h_x, h_y)
        result = None
        if template is not None:
            # el_geometry.el_template returns (vp_y, hp_x, v_quo_span, h_quo_span);
            # everything else this method needs is carried inside the two spans
            # (v_quo_span = (v_x, v_shift, v_a, v_b); h_quo_span = (h_shift, h_y, h_a, h_b)).
            vp_y, hp_x, v_quo_span, h_quo_span = template
            _v_x, v_y_shift, v_a, v_b = v_quo_span
            h_x_shift, _h_y, h_a, h_b = h_quo_span
            v_kind = "v1" if v_y_shift == 1 else "v3"
            h_kind = "h1" if h_x_shift == 1 else "h3"
            vp = (v_x, vp_y, v_k)
            hp = (hp_x, h_y, h_k)
            missing = self.missing_qubits
            if (
                v not in missing[v_kind]
                and vp not in missing[v_kind]
                and h not in missing[h_kind]
                and hp not in missing[h_kind]
                and (vp, hp) not in self.missing_internal_couplers
            ):
                v_runs = self.runs[v_kind].get(v_x, {}).get(v_k, ())
                h_runs = self.runs[h_kind].get(h_y, {}).get(h_k, ())
                if _covers(v_runs, v_a, v_b) and _covers(h_runs, h_a, h_b):
                    result = (v_quo_span, h_quo_span, v_k, h_k)

        cache[(v, h)] = result  # store None directly (unreachable is a real answer)
        return result

    def clear_el_cache(self) -> None:
        """Empty the el_reachable memo cache (self._el_cache).

        el_reachable memoizes each ``(v, h)`` result. Within one window the four corners share those
        lookups, but a worker processes many windows before it is recycled, so left alone the memo
        would grow without bound. Calling this once per window keeps it scoped to a single window's
        work. Cheap -- it just drops the dict; results are recomputed on demand as needed.
        """
        self._el_cache = {}

    def _position_quo(self) -> Pos:
        """Rank each external line's qubits by reach from each side and emit position maps.

        The ``t`` parallel qubits on a line are ranked by how far they reach from each of the four
        sides. For every coordinate along a line, each present ``k`` has a distance to the near end
        of its run: for verticals, from the bottom ("b") and top ("t"); for horizontals, from the
        left ("l") and right ("r"). The ``k``'s are ranked by that distance and reassigned physical
        positions via the Zephyr index formula, so downstream "position order" == "reach order".
        This is what lets the sliding-window embedding turn a 2D non-crossing constraint into a 1D
        longest-increasing-subsequence problem.

        Returns ``{side: {(x, y): {orig_k: pos}}}`` for ``side in {"r", "l", "b", "t"}``.
        Coordinates covered by no run get an empty ``{}``. Consumes self.runs.
        """
        abs_max, t = self.abs_max, self.t
        runs = self.runs

        v_bottom, v_top, h_left, h_right = {}, {}, {}, {}

        # vertical: x even; two lines per x (v1, v3)
        for x in range(0, abs_max + 1, 2):
            bottom_reach_by_y, top_reach_by_y = {}, {}
            for kind in ("v1", "v3"):
                runs_by_k = runs[kind].get(x, {})
                for k, runs_for_k in runs_by_k.items():
                    for run_start, run_end in runs_for_k:
                        y = run_start
                        while y <= run_end:
                            bottom_reach = bottom_reach_by_y.get(y)
                            if bottom_reach is None:
                                bottom_reach = bottom_reach_by_y[y] = {}
                                top_reach_by_y[y] = {}
                            # reach distance to each end of this run:
                            # bottom = y - run_start, top (up) = run_end - y
                            bottom_reach[k] = y - run_start
                            top_reach_by_y[y][k] = run_end - y
                            y += 4
            for y, bottom_reach in bottom_reach_by_y.items():
                v_bottom[(x, y)] = _ranked_pos_fast(x, y, bottom_reach, t, True)
                v_top[(x, y)] = _ranked_pos_fast(x, y, top_reach_by_y[y], t, True)

        # horizontal: y even; two lines per y (h1, h3)
        for y in range(0, abs_max + 1, 2):
            left_reach_by_x, right_reach_by_x = {}, {}
            for kind in ("h1", "h3"):
                runs_by_k = runs[kind].get(y, {})
                for k, runs_for_k in runs_by_k.items():
                    for run_start, run_end in runs_for_k:
                        x = run_start
                        while x <= run_end:
                            left_reach = left_reach_by_x.get(x)
                            if left_reach is None:
                                left_reach = left_reach_by_x[x] = {}
                                right_reach_by_x[x] = {}
                            left_reach[k] = run_end - x
                            right_reach_by_x[x][k] = x - run_start
                            x += 4
            for x, left_reach in left_reach_by_x.items():
                h_left[(x, y)] = _ranked_pos_fast(x, y, left_reach, t, False)
                h_right[(x, y)] = _ranked_pos_fast(x, y, right_reach_by_x[x], t, False)

        # backfill empty dicts for uncovered in-range coords
        for x in range(0, abs_max + 1, 2):
            for y in range(1, abs_max + 1, 2):
                key = (x, y)
                if key not in v_bottom:
                    v_bottom[key] = {}
                    v_top[key] = {}
        for x in range(1, abs_max + 1, 2):
            for y in range(0, abs_max + 1, 2):
                key = (x, y)
                if key not in h_left:
                    h_left[key] = {}
                    h_right[key] = {}

        return {"r": h_right, "l": h_left, "b": v_bottom, "t": v_top}


def _covers(runs_for_k: Iterable[Run], a: int, b: int) -> bool:
    """True if some run ``(start, end)`` in the set fully spans ``[a, b]``.

    That is, ``start <= a and end >= b``. Used to test whether the faulty grid provides an intact
    stretch long enough for an el_template's required quo_span.
    """
    for s, e in runs_for_k:
        if s <= a and e >= b:
            return True
    return False


def _ranked_pos_fast(
    x: int, y: int, reach_by_k: dict[int, int], t: int, is_vertical: bool
) -> dict[int, tuple[int, int]]:
    """Rank the ``k``'s in reach_by_k by reach distance and assign each a position.

    Ranking is by ascending reach distance; each ``k`` then gets the closed-form Zephyr index for
    its rank, with the constant part of the formula
    hoisted out of the loop: within one ``(x, y)`` only the ``2*rank`` term varies, so the base is
    computed once and the varying value steps by 2 per rank. Returns ``{orig_k: (a, b)}``;
    is_vertical selects which tuple slot carries the varying value (verticals -> slot 0,
    horizontals -> slot 1).
    """
    if is_vertical:
        j = ((y - 1) & 3) // 2  # Zephyr j of this line
        fixed_val = 2 * (t + 1) * (j + 2 * (y // 4))
        varying_base = 1 + (t + 1) * x + j
        pos_by_k = {}
        rank = 0
        for k, _ in sorted(reach_by_k.items(), key=itemgetter(1)):
            pos_by_k[k] = (varying_base + 2 * rank, fixed_val)
            rank += 1
        return pos_by_k
    else:
        j = ((x - 1) & 3) // 2  # Zephyr j of this line
        fixed_val = 2 * (t + 1) * (j + 2 * (x // 4))
        varying_base = 1 + (t + 1) * y + j
        pos_by_k = {}
        rank = 0
        for k, _ in sorted(reach_by_k.items(), key=itemgetter(1)):
            pos_by_k[k] = (fixed_val, varying_base + 2 * rank)
            rank += 1
        return pos_by_k

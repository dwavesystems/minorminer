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
  .present[kind]      ``{(x, y, k): r}``               present qubits -> linear index
  .missing[kind]      frozenset of ``(x, y, k)``       absent qubits
  .edges              set of ``(lo, hi)``              present couplers, r-index pairs
  .missing_int        ``{(v, hp): (v_base, h_base)}``  missing internal couplers
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
    present: dict[str, dict[Node, int]]  # present[kind][node] -> linear index r
    missing: dict[str, frozenset[Node] | set[Node]]
    _edges: set[tuple[int, int]] | None  # present couplers as (lo, hi) r-pairs
    missing_int: dict | None  # see _missing_int
    runs: Runs | None
    _el_cache: dict  # el_reachable memo
    pos: Pos | None
    __slots__ = (
        "m",
        "t",
        "abs_min",
        "abs_max",
        "labels",
        "present",
        "missing",
        "_edges",
        "missing_int",
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
        self.labels = "int"  # output mode: "int" or "coordinates"
        self.present = {k: {} for k in _KINDS}
        self.missing = {k: set() for k in _KINDS}
        self._edges = None
        # survey outputs (filled by _run_survey)
        self.missing_int = None
        self.runs = None
        self._el_cache = {}
        self.pos = None

    @classmethod
    def from_graph(cls, graph: nx.Graph) -> Grid:
        """Build a Grid from a networkx Zephyr graph and run the full survey.

        The graph must carry Zephyr topology metadata in graph.graph: family == "zephyr", rows (=
        ``m``), tile (= ``t``), and labels in {"int", "coordinates"}. Raises ValueError if the
        topology is not zephyr.

        labels == "int": nodes are linear indices (as a D-Wave sampler's nodelist); matched directly
        against the linear-index lattice walk. labels == "coordinates": nodes are Zephyr coordinates
        ``(u, w, k, j, z)``; each is converted to cartesian ``(x, y, k)``, then to its linear index
        r via the walk.

        Either way the internal representation is identical (cartesian nodes, linear-index r,
        ``(r, r)`` edges); only find_clique's OUTPUT format follows `labels` (linear indices for
        "int", Zephyr coordinates for "coordinates").
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
        # The labels string, if present, is used only as a sanity cross-check.
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

        g = cls(m, t)
        g.labels = labels

        if labels == "int":
            present_r = set(graph.nodes())
            g._classify(present_r)
            g._edges = {(a, b) if a < b else (b, a) for a, b in graph.edges()}
        elif labels == "cartesian":  # nodes are cartesian (x, y, k) already
            g._classify_from_present_nodes(set(graph.nodes()))
            # cartesian edges -> linear r pairs (internal edge set stays linear)
            lin = g.cartesian_to_linear
            edges = set()
            for ca, cb in graph.edges():
                ra, rb = lin(ca), lin(cb)
                if ra is None or rb is None:
                    continue
                edges.add((ra, rb) if ra < rb else (rb, ra))
            g._edges = edges
        else:  # coordinates: nodes are Zephyr 5-tuples
            g._classify_from_present_nodes({zephyr_to_cartesian(z) for z in graph.nodes()})
            # convert coordinate edges -> cartesian -> linear r pairs
            lin = g.cartesian_to_linear
            edges = set()
            for za, zb in graph.edges():
                ra = lin(zephyr_to_cartesian(za))
                rb = lin(zephyr_to_cartesian(zb))
                if ra is None or rb is None:
                    continue
                edges.add((ra, rb) if ra < rb else (rb, ra))
            g._edges = edges

        g._freeze()
        g._run_survey()
        return g

    def _classify(self, present_r: set[int]) -> None:
        """Walk the full ideal lattice in linear-index order, sorting each node into its kind bucket
        (present[kind] or missing[kind]) and recording the r -> cartesian map. A node is present iff
        its linear index r is in present_r.

        The walk order is what DEFINES r, and it must match the sampler's own indexing exactly --
        verticals first (``x = 0, 2, ...``; for each ``k``, the ``y % 4 == 1`` nodes then the
        ``y % 4 == 3`` nodes), then horizontals (``y = 0, 2, ...``; for each ``k``, ``x % 4 == 1``
        then ``x % 4 == 3``). Do not reorder without re-verifying r.
        """
        self._walk(lambda r, node: r in present_r)

    def _classify_from_present_nodes(self, present_nodes: set[Node]) -> None:
        """Same lattice walk / r numbering as _classify, but a node is present iff its cartesian
        ``(x, y, k)`` is in present_nodes (used for coordinate- labeled graphs, where presence is
        known by node, not by linear index)."""
        self._walk(lambda r, node: node in present_nodes)

    def _walk(self, is_present: Callable[[int, Node], bool]) -> None:
        """Shared lattice walk: assign linear index r to every ideal node in the canonical order,
        and file it as present/missing per the is_present(r, node) predicate. Populates
        present[kind] and missing[kind]."""
        m, t = self.m, self.t
        M = 2 * m + 1
        present, missing = self.present, self.missing
        r = 0
        for w in range(M):
            x = 2 * w
            for k in range(t):
                for shift, kind in ((1, "v1"), (3, "v3")):
                    pres, miss = present[kind], missing[kind]
                    for z in range(m):
                        node = (x, 4 * z + shift, k)
                        if is_present(r, node):
                            pres[node] = r
                        else:
                            miss.add(node)
                        r += 1
        for w in range(M):
            y = 2 * w
            for k in range(t):
                for shift, kind in ((1, "h1"), (3, "h3")):
                    pres, miss = present[kind], missing[kind]
                    for z in range(m):
                        node = (4 * z + shift, y, k)
                        if is_present(r, node):
                            pres[node] = r
                        else:
                            miss.add(node)
                        r += 1

    def _freeze(self) -> None:
        """Convert the mutable missing sets to frozensets after classification. Signals that the
        missing buckets are final; membership tests are the hot use, and frozenset makes the
        immutability explicit."""
        self.missing = {k: frozenset(s) for k, s in self.missing.items()}

    def _run_survey(self) -> None:
        """Run every survey pass in dependency order and cache the results.

        Order matters: missing_int feeds el_reachable's reachability checks; survey_ext's runs feed
        both el_reachable and position_quo. The ideal el-template geometry is fault-independent, so
        it is not surveyed here; the el_template_length / el_reachable methods compute it on demand.
        """
        self.missing_int = self._missing_int()
        self.runs = self._survey_ext()
        self.pos = self._position_quo()

    # -------------------------------------------------------------- accessors
    def kind_of(self, ccoord: Node) -> str:
        """Return the kind ("v1"/"v3"/"h1"/"h3") of a cartesian coord from its parity. ``x`` even
        -> vertical (``v1 if y % 4 == 1 else v3``); ``x`` odd -> horizontal
        (``h1 if x % 4 == 1 else h3``). Pure coordinate arithmetic; no lookup."""
        x, y, _ = ccoord
        if (x & 1) == 0:
            return "v1" if (y & 3) == 1 else "v3"
        return "h1" if (x & 3) == 1 else "h3"

    def cartesian_to_linear(self, ccoord: Node) -> int | None:
        """Linear index (lcoord) of a present cartesian coord, or None if absent/missing."""
        return self.present[self.kind_of(ccoord)].get(ccoord)

    def is_present(self, ccoord: Node) -> bool:
        """True if the cartesian coord exists in the (faulty) grid."""
        return ccoord in self.present[self.kind_of(ccoord)]

    @property
    def edges(self) -> set[tuple[int, int]]:
        """The set of present couplers as ``(lo, hi)`` linear-index pairs."""
        return self._edges

    def has_edge(self, lcoord1: int, lcoord2: int) -> bool:
        """True if a coupler exists between linear indices lcoord1 and lcoord2
        (order-independent)."""
        e = (lcoord1, lcoord2) if lcoord1 < lcoord2 else (lcoord2, lcoord1)
        return e in self._edges

    # ------------------------------------------------- el-template geometry
    # Ideal, fault-independent geometry, computed on demand.

    def el_template_length(self, v_quo: QuoSpan, h_quo: QuoSpan) -> int | None:
        """chain_length of the el_template ``(v_quo, h_quo)``, or None if the pair is not a real
        el_template. (The None case doubles as the el_template test used by expand_el_templates.)"""
        return el_geometry.el_template_length(self.m, v_quo, h_quo)

    # ----------------------------------------------------------- survey passes
    def _missing_int(
        self,
    ) -> dict[tuple[Node, Node], tuple[tuple[int, int, int], tuple[int, int, int]]]:
        """Find internal couplers that SHOULD exist (ideal geometry) but are absent in the faulty
        grid.

        Internal couplers join a vertical node ``v = (x, y, k)`` to a horizontal node
        ``hp = (x+-1, y+-1, kp)`` for every ``kp in range(t)``. For each present vertical node and
        each of its four diagonal horizontal neighbours (in bounds, present), we check whether the
        coupler exists in `edges`; if not, record it. Returns ``{(v, hp): (v_base, hp_base)}`` where
        the values are the quotient (``k``-folded) descriptors ``(x, v_shift, k)`` and
        ``(hp_shift, yj, kp)``.
        """
        abs_min, abs_max, t = self.abs_min, self.abs_max, self.t
        edges, present, missing = self._edges, self.present, self.missing
        H_KIND = {1: "h1", 3: "h3"}
        krange = range(t)
        OFFS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
        out = {}
        for v_shift, v_kind in ((1, "v1"), (3, "v3")):
            for v, v_r in present[v_kind].items():
                x, y, k = v
                for dx, dy in OFFS:
                    xi = x + dx
                    if xi < abs_min or xi > abs_max:
                        continue
                    yj = y + dy
                    if yj < abs_min or yj > abs_max:
                        continue
                    hp_shift = xi & 3
                    hp_nodes = present[H_KIND[hp_shift]]
                    hp_missing = missing[H_KIND[hp_shift]]
                    base = (x, v_shift, k)
                    # internal couplers are all-to-all in k: v's k couples to
                    # every hp kp, so there is no k == kp guard here.
                    for kp in krange:
                        hp = (xi, yj, kp)
                        if hp in hp_missing:
                            continue
                        hp_r = hp_nodes[hp]
                        er = (v_r, hp_r) if v_r < hp_r else (hp_r, v_r)
                        if er not in edges:
                            out[(v, hp)] = (base, (hp_shift, yj, kp))
        return out

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
        edges, present, missing = self._edges, self.present, self.missing
        runs = {k: {} for k in _KINDS}
        for dir_str, s0 in product(("v", "h"), (1, 3)):
            kind = dir_str + str(s0)
            _nodes, _missing = present[kind], missing[kind]
            is_v = dir_str == "v"
            for i in range(0, 4 * m + 1, 2):
                coord_map = runs[kind].setdefault(i, {})
                for k in range(t):
                    res = set()
                    s = s0
                    while s <= abs_max:
                        node = (i, s, k) if is_v else (s, i, k)
                        if node not in _missing:
                            prev = s
                            inc = 4
                            end = prev
                            next_s = None
                            while prev + inc <= abs_max:
                                if is_v:
                                    pn = (i, prev, k)
                                    nn = (i, prev + inc, k)
                                else:
                                    pn = (prev, i, k)
                                    nn = (prev + inc, i, k)
                                if nn in _missing:
                                    # next node is a hole: end the run and resume
                                    # PAST it (a missing node can't start a run).
                                    end, next_s = prev, prev + 2 * inc
                                    break
                                if (_nodes[pn], _nodes[nn]) in edges:
                                    prev += inc
                                else:
                                    # coupler is broken but nn exists: end here and
                                    # resume AT nn, which starts the next run.
                                    end, next_s = prev, prev + inc
                                    break
                            else:
                                end, next_s = prev, None
                            res.add((s, end))
                            if next_s is None:
                                break
                            s = next_s
                        else:
                            s += 4
                    coord_map[k] = res
        return runs

    def el_reachable(self, v: Node, h: Node) -> tuple[QuoSpan, QuoSpan, int, int] | None:
        """Does the L-shaped ("el") shortest path from v to h exist in the faulty grid?

        The el-path travels along v's external line to the elbow with h's external line, hops one
        internal coupler (between vp and hp), then travels along h's line to h. It exists iff:
          * the block pair ``(v_block, h_block)`` has an el_template (i.e. the two lines cross in
            the ideal grid),
          * none of the four involved nodes v, vp, h, hp are missing,
          * the connecting internal coupler is not in missing_int,
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
        tmpl = el_geometry.el_template(self.m, v_x, v_y, h_x, h_y)
        result = None
        if tmpl is not None:
            # el_geometry.el_template returns (vp_y, hp_x, v_quo_span, h_quo_span);
            # everything else this method needs is carried inside the two spans
            # (v_quo_span = (v_x, v_shift, v_a, v_b); h_quo_span = (h_shift, h_y, h_a, h_b)).
            vp_y, hp_x, v_quo_span, h_quo_span = tmpl
            _v_x, v_y_shift, v_a, v_b = v_quo_span
            h_x_shift, _h_y, h_a, h_b = h_quo_span
            v_kind = "v1" if v_y_shift == 1 else "v3"
            h_kind = "h1" if h_x_shift == 1 else "h3"
            vp = (v_x, vp_y, v_k)
            hp = (hp_x, h_y, h_k)
            missing = self.missing
            if (
                v not in missing[v_kind]
                and vp not in missing[v_kind]
                and h not in missing[h_kind]
                and hp not in missing[h_kind]
                and (vp, hp) not in self.missing_int
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
        """Rank the ``t`` parallel qubits on each external line by how far they reach from each of
        the four sides, and emit position maps.

        For every coordinate along a line, each present ``k`` has a distance to the near end of its
        run: for verticals, distance from the bottom ("b") and top ("t"); for horizontals, from the
        left ("l") and right ("r"). The ``k``'s are ranked by that distance and reassigned physical
        positions via the Zephyr index formula, so downstream "position order" == "reach order".
        This is what lets the sliding-window embedding turn a 2D non-crossing constraint into a 1D
        longest-increasing-subsequence problem.

        Returns ``{side: {(x, y): {orig_k: pos}}}`` for ``side in {"r", "l", "b", "t"}``.
        Coordinates covered by no run get an empty ``{}``. Consumes self.runs.
        """
        abs_max, t = self.abs_max, self.t
        runs = self.runs

        verb, vert, horl, horr = {}, {}, {}, {}

        # vertical: x even; two lines per x (v1, v3)
        for x in range(0, abs_max + 1, 2):
            b_by_y, u_by_y = {}, {}
            for kind in ("v1", "v3"):
                per_k = runs[kind].get(x, {})
                for k, segset in per_k.items():
                    for start, end in segset:
                        yy = start
                        while yy <= end:
                            bd = b_by_y.get(yy)
                            if bd is None:
                                bd = b_by_y[yy] = {}
                                u_by_y[yy] = {}
                            # reach distance to each end of this run:
                            # bottom = yy - start, top (up) = end - yy
                            bd[k] = yy - start
                            u_by_y[yy][k] = end - yy
                            yy += 4
            for yy, bd in b_by_y.items():
                verb[(x, yy)] = _ranked_pos_fast(x, yy, bd, t, True)
                vert[(x, yy)] = _ranked_pos_fast(x, yy, u_by_y[yy], t, True)

        # horizontal: y even; two lines per y (h1, h3)
        for y in range(0, abs_max + 1, 2):
            l_by_x, r_by_x = {}, {}
            for kind in ("h1", "h3"):
                per_k = runs[kind].get(y, {})
                for k, segset in per_k.items():
                    for start, end in segset:
                        xx = start
                        while xx <= end:
                            ld = l_by_x.get(xx)
                            if ld is None:
                                ld = l_by_x[xx] = {}
                                r_by_x[xx] = {}
                            ld[k] = end - xx
                            r_by_x[xx][k] = xx - start
                            xx += 4
            for xx, ld in l_by_x.items():
                horl[(xx, y)] = _ranked_pos_fast(xx, y, ld, t, False)
                horr[(xx, y)] = _ranked_pos_fast(xx, y, r_by_x[xx], t, False)

        # backfill empty dicts for uncovered in-range coords
        for x in range(0, abs_max + 1, 2):
            for yy in range(1, abs_max + 1, 2):
                key = (x, yy)
                if key not in verb:
                    verb[key] = {}
                    vert[key] = {}
        for xx in range(1, abs_max + 1, 2):
            for y in range(0, abs_max + 1, 2):
                key = (xx, y)
                if key not in horl:
                    horl[key] = {}
                    horr[key] = {}

        return {"r": horr, "l": horl, "b": verb, "t": vert}


def _covers(runs_for_k: Iterable[Run], a: int, b: int) -> bool:
    """True if some run ``(start, end)`` in the set fully spans ``[a, b]``
    (``start <= a and end >= b``). Used to test whether the faulty grid provides an intact
    stretch long enough for an el_template's required quo_span."""
    for s, e in runs_for_k:
        if s <= a and e >= b:
            return True
    return False


def _ranked_pos_fast(
    x: int, y: int, dist_by_k: dict[int, int], t: int, is_vertical: bool
) -> dict[int, tuple[int, int]]:
    """Rank the ``k``'s in dist_by_k by ascending distance and assign each a Zephyr position for its
    rank.

    Each ``k`` gets the closed-form Zephyr index for its rank, with the constant part of the formula
    hoisted out of the loop: within one ``(x, y)`` only the ``2*rank`` term varies, so the base is
    computed once and val steps by 2 per rank. Returns ``{orig_k: (a, b)}``; is_vertical selects
    which tuple slot carries the varying value (verticals -> slot 0, horizontals -> slot 1).
    """
    if is_vertical:
        j = ((y - 1) & 3) // 2
        start = 2 * (t + 1) * (j + 2 * (y // 4))
        val0 = 1 + (t + 1) * x + j
        out = {}
        rank = 0
        for k, _ in sorted(dist_by_k.items(), key=itemgetter(1)):
            out[k] = (val0 + 2 * rank, start)
            rank += 1
        return out
    else:
        j = ((x - 1) & 3) // 2
        start = 2 * (t + 1) * (j + 2 * (x // 4))
        val0 = 1 + (t + 1) * y + j
        out = {}
        rank = 0
        for k, _ in sorted(dist_by_k.items(), key=itemgetter(1)):
            out[k] = (start, val0 + 2 * rank)
            rank += 1
        return out

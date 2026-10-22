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
coordinates — plain ``(x, y, k)`` tuples following the convention of
:class:`dwave.graphs.ZephyrCartesianCoord`.

Naming: a "quo"/"quotient" prefix means ``k``-agnostic -- the object with its ``k`` index folded
out, standing for every ``k`` at once. Terms carrying it (quo external path, quo_span below) are
``k``-agnostic by definition, so their entries do not repeat it.

node / qubit
    ``(x, y, k)``: a Zephyr node in Cartesian coordinates, ``k in range(t)``.
block
    ``(x, y)``: quotient node (ignores ``k``).

Orientation is fixed by parity: a vertical node has ``x`` even and ``y`` odd; a horizontal node has
``x`` odd and ``y`` even.

At a fixed even ``x`` the vertical nodes have odd ``y``, so ``y % 4`` is 1 or 3: the two interleaved
vertical kinds v1 and v3, offset by 2 in ``y`` and each stepping by 4. Likewise at a fixed even
``y`` the two horizontal kinds h1 and h3 have ``x % 4`` equal to 1 or 3. In each case, the remainder
of the odd coordinate of a block by 4 is called the shift. This orientation-plus-shift label is a
node's kind.

kind
    "v1"|"v3"|"h1"|"h3": orientation + which of the two shifts.

    - vertical  : ``x`` even, ``y`` odd, ``shift = y % 4`` (so ``shift in {1, 3}``)
    - horizontal: ``x`` odd,  ``y`` even, ``shift = x % 4`` (so ``shift in {1, 3}``)

An external path through a node is a path that contains it and uses only external couplers. It is
therefore either vertical -- through nodes sharing ``x = fixed_coord`` and ``k``, with ``y``
differing by multiples of 4 -- or horizontal -- through nodes sharing ``y = fixed_coord`` and ``k``,
with ``x`` differing by multiples of 4. In an ideal Zephyr all nodes sharing a
``(kind, fixed_coord, k)`` lie on one external path; in a faulty one, yield loss (missing nodes or
external couplers) can break that path into pieces. Each maximal such piece is a RUN.

quo external path
    ``(kind, fixed_coord)``: An external path with ``k`` folded out; two per column/row (the two
    shifts).

run
    ``(start, end)``: A MAXIMAL external path at a given ``(kind, fixed_coord, k)``; grid.runs holds
    these.

quo_span
    A :data:`.el_geometry.QuoSpan` -- a span ``[a, b]`` on a quo external path that a chain requires
    (not necessarily maximal); realized when some run covers it. Vertical spans are
    ``(v_x, shift, a, b)``, horizontal spans ``(shift, h_y, a, b)``.

el_reachable
    Method ``el_reachable(v, h) -> (v_quo_span, h_quo_span) | None``: whether the L-shaped v->h path
    (along v's line, one internal hop at the elbow, along h's line) survives in *this* faulty grid
    -- the fault-dependent counterpart to el_geometry's ideal geometry.

side order
    ``side_order[side][block][k]``: an integer ordering nodes along a window side
    (``side in {"t", "b", "l", "r"}``), first by block and then, within a block, by how far the
    node's run reaches from that side into the window (see :meth:`Grid._survey_side_order`).

Construction
------------
``Grid(G)`` builds the object and runs the full survey, caching everything.
"""

from __future__ import annotations

import numbers
from collections.abc import Callable, Iterable
from operator import itemgetter
from typing import TYPE_CHECKING

from dwave.graphs import zephyr_coordinates

from minorminer.utils._clique import el_geometry
from minorminer.utils._clique.el_geometry import QuoSpan

if TYPE_CHECKING:
    import networkx as nx


# --- type aliases -----------------------------------------------------------
Node = tuple[int, int, int]  # cartesian qubit (x, y, k)
Block = tuple[int, int]  # quotient node (x, y), k ignored
Run = tuple[int, int]  # an intact stretch (start, end) on a line
# runs[kind][coord][k] -> {(start, end), ...}
Runs = dict[str, dict[int, dict[int, set[Run]]]]
# side_order[side][(x, y)][orig_k] -> order
SideOrder = dict[str, dict[Block, dict[int, int]]]

_NODE_KINDS = ("v1", "v3", "h1", "h3")  # kinds of nodes
_V_KIND_BY_SHIFT = {1: "v1", 3: "v3"}  # vertical kind from y % 4
_H_KIND_BY_SHIFT = {1: "h1", 3: "h3"}  # horizontal kind from x % 4

# (dx, dy) from a node to its four diagonal neighbours of the other orientation (internal couplers)
_INTERNAL_COUPLER_OFFSETS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
# (dx, dy) from a node to its same-k neighbours: odd couplers (+-2), external couplers (+-4)
_V_SAME_K_OFFSETS = ((0, 2), (0, -2), (0, 4), (0, -4))
_H_SAME_K_OFFSETS = ((2, 0), (-2, 0), (4, 0), (-4, 0))

# Default for el_reachable's cache lookup: cache.get(key, _EL_REACHABLE_NOT_CACHED). Its own unique
# object, never stored in the cache, so getting it back means the (v, h) pair hasn't been computed
# yet. None can't play this role because None is a real cached result (indicating the pair is
# unreachable).
_EL_REACHABLE_NOT_CACHED = object()


class Grid:
    """A (possibly faulty) Zephyr chip surveyed once at construction into fast lookup structures.

    "Faulty" means some qubits or couplers may be absent; construction walks the whole graph a
    single time and precomputes the tables (present/missing qubits, couplers, per-line runs, ...)
    that the rest of the algorithm queries, rather than re-walking the graph on every lookup.

    Args:
        G: A networkx Zephyr graph.
    """

    def __init__(self, G: nx.Graph) -> None:
        # family/rows/tile checks + label-mode inference
        self.m, self.t, self.labels = _parse_zephyr_metadata(G)
        self.abs_min: int = 0  # smallest valid coordinate
        self.abs_max: int = 4 * self.m  # largest valid coordinate
        self.present_qubits: dict[str, dict[Node, int]] = {k: {} for k in _NODE_KINDS}
        self.missing_qubits: dict[str, frozenset[Node] | set[Node]] = {
            k: set() for k in _NODE_KINDS
        }
        self._el_cache: dict[tuple[Node, Node], tuple[QuoSpan, QuoSpan] | None] = {}

        if self.labels == "int":
            present_linear = set(G.nodes())
            self._classify_present_linear(present_linear)
            edges = {(a, b) if a < b else (b, a) for a, b in G.edges()}
        else:  # tuple nodes: cartesian (x, y, k) directly, or Zephyr 5-tuples to convert
            if self.labels == "coordinates":
                to_cartesian = zephyr_coordinates.zephyr_to_cartesian
            else:  # cartesian: nodes are already (x, y, k)
                to_cartesian = lambda node: node
            self._classify_present_cartesian({to_cartesian(n) for n in G.nodes()})
            # node edges -> cartesian -> linear-coordinate pairs (the edge set is always stored
            # by linear coordinate, regardless of input label mode)
            edges = set()
            for node_a, node_b in G.edges():
                lcoord_a = self.cartesian_to_linear(to_cartesian(node_a))
                lcoord_b = self.cartesian_to_linear(to_cartesian(node_b))
                if lcoord_a is None or lcoord_b is None:
                    continue
                edges.add((lcoord_a, lcoord_b) if lcoord_a < lcoord_b else (lcoord_b, lcoord_a))
        self.edges: set[tuple[int, int]] = edges

        # Order matters: missing_internal_couplers feeds el_reachable's reachability checks;
        # _survey_ext's runs feed both el_reachable and _survey_side_order. The ideal el-template
        # geometry is fault-independent, so it is not surveyed here; the el_template_length /
        # el_reachable methods compute it on demand.
        self.missing_internal_couplers = self._missing_internal_couplers()
        self.runs = self._survey_ext()
        self.side_order = self._survey_side_order()

    def _classify_present_linear(self, present_nodes_linear: set[int]) -> None:
        """Sort every ideal-Zephyr node into its kind bucket, in linear-coordinate order.

        Walks the full ideal Zephyr, filing each node under present_qubits[kind] or
        missing_qubits[kind] and recording each present node's linear coordinate. A node is present
        iff its linear coordinate is in present_nodes_linear.

        Args:
            present_nodes_linear: The present qubits represented with their linear coordinates.
        """
        self._walk(lambda lcoord, ccoord: lcoord in present_nodes_linear)

    def _classify_present_cartesian(self, present_nodes_cartesian: set[Node]) -> None:
        """Classify by cartesian-node membership instead of by linear coordinate.

        Same Zephyr walk / linear-coordinate numbering as :meth:`._classify_present_linear`, but a
        node is present iff its cartesian ``(x, y, k)`` is in present_nodes_cartesian (used for
        coordinate-labeled graphs, where presence is known by node, not by linear coordinate).

        Args:
            present_nodes_cartesian: The present qubits represented with their cartesian
                coordinates.
        """
        self._walk(lambda lcoord, ccoord: ccoord in present_nodes_cartesian)

    def _walk(self, is_present_fn: Callable[[int, Node], bool]) -> None:
        """Assign a linear coordinate to every ideal-Zephyr node and file it as present or missing.

        The walk order is what DEFINES each node's linear coordinate, and it must match the graph's
        own indexing exactly -- verticals first, then horizontals.

        Shared by both classify entry points: Walks the canonical order and applies the
        is_present_fn(lcoord, node) function. Populates present_qubits[kind] and finalizes
        missing_qubits[kind] as frozensets (membership tests are the only downstream use).

        Args:
            is_present_fn: A function ``(lcoord, node) -> bool`` deciding whether the node at linear
                coordinate ``lcoord`` with cartesian coordinate ``node`` is present.
        """
        m, t = self.m, self.t
        num_w = 2 * m + 1
        present, missing = self.present_qubits, self.missing_qubits
        lcoord = 0
        for w in range(num_w):
            x = 2 * w
            for k in range(t):
                for shift, kind in _V_KIND_BY_SHIFT.items():
                    kind_present, kind_missing = present[kind], missing[kind]
                    for z in range(m):
                        node = (x, 4 * z + shift, k)
                        if is_present_fn(lcoord, node):
                            kind_present[node] = lcoord
                        else:
                            kind_missing.add(node)
                        lcoord += 1
        for w in range(num_w):
            y = 2 * w
            for k in range(t):
                for shift, kind in _H_KIND_BY_SHIFT.items():
                    kind_present, kind_missing = present[kind], missing[kind]
                    for z in range(m):
                        node = (4 * z + shift, y, k)
                        if is_present_fn(lcoord, node):
                            kind_present[node] = lcoord
                        else:
                            kind_missing.add(node)
                        lcoord += 1
        # freeze the missing buckets as soon as they're built: membership tests are
        # the only downstream use, and frozenset makes the immutability explicit
        self.missing_qubits = {k: frozenset(s) for k, s in self.missing_qubits.items()}

    def is_vertical(self, ccoord: Node) -> bool:
        """Decides whether a node is vertical.

        Args:
            ccoord: A node represented with its cartesian coordinate.

        Returns:
            ``True`` if the node is vertical, else ``False``.
        """
        return (ccoord[0] & 1) == 0

    def kind_of(self, ccoord: Node) -> str:
        """Return the kind ("v1"/"v3"/"h1"/"h3") of a cartesian coord.

        ``x`` even -> vertical (``v1 if y % 4 == 1 else v3``); ``x`` odd -> horizontal
        (``h1 if x % 4 == 1 else h3``). Pure coordinate arithmetic; no lookup.

        Args:
            ccoord: A node represented with its cartesian coordinate.

        Returns:
            The node's kind, one of ``"v1"``, ``"v3"``, ``"h1"``, ``"h3"``.
        """
        x, y, _ = ccoord
        if self.is_vertical(ccoord):
            return "v1" if (y & 3) == 1 else "v3"
        return "h1" if (x & 3) == 1 else "h3"

    def cartesian_to_linear(self, ccoord: Node) -> int | None:
        """Linear coordinate of a present node, or None if absent/missing.

        Args:
            ccoord: A node represented with its cartesian coordinate.

        Returns:
            The linear coordinate of ``ccoord``, or ``None`` if the ``ccoord`` is absent.
        """
        return self.present_qubits[self.kind_of(ccoord)].get(ccoord)

    def is_present(self, ccoord: Node) -> bool:
        """True if the cartesian coord exists in the (faulty) grid.

        Args:
            ccoord: A node with its cartesian coordinate.

        Returns:
            ``True`` if ``ccoord`` exists in the faulty grid, else ``False``.
        """
        return ccoord in self.present_qubits[self.kind_of(ccoord)]

    def has_edge(self, lcoord1: int, lcoord2: int) -> bool:
        """Decides if a coupler exists between two nodes, given by linear coordinates.

        Args:
            lcoord1: One endpoint's linear coordinate.
            lcoord2: The other endpoint's linear coordinate.

        Returns:
            ``True`` if there is a coupler between the two nodes, else ``False``.
        """
        e = (lcoord1, lcoord2) if lcoord1 < lcoord2 else (lcoord2, lcoord1)
        return e in self.edges

    def neighbors(self, ccoord: Node) -> list[Node]:
        """All neighbors of a node in the grid.

        A neighbor is a present node joined to ``ccoord`` by a working coupler of
        any type (internal, odd, or external). An absent ``ccoord`` has none.

        Args:
            ccoord: A node represented with its cartesian coordinate.

        Returns:
            The neighbors of ``ccoord``: internal ones first, then odd and external ones.
        """
        node_lcoord = self.cartesian_to_linear(ccoord)
        if node_lcoord is None:
            return []
        x, y, k = ccoord
        to_linear, has_edge = self.cartesian_to_linear, self.has_edge
        same_k_offsets = _V_SAME_K_OFFSETS if self.is_vertical(ccoord) else _H_SAME_K_OFFSETS
        out = []
        for dx, dy in _INTERNAL_COUPLER_OFFSETS:
            for kp in range(self.t):
                nbr = (x + dx, y + dy, kp)
                nbr_lcoord = to_linear(nbr)
                if nbr_lcoord is not None and has_edge(node_lcoord, nbr_lcoord):
                    out.append(nbr)
        for dx, dy in same_k_offsets:
            nbr = (x + dx, y + dy, k)
            nbr_lcoord = to_linear(nbr)
            if nbr_lcoord is not None and has_edge(node_lcoord, nbr_lcoord):
                out.append(nbr)
        return out

    # Ideal, fault-independent geometry, computed on demand.
    def el_template_length(self, v_quo: QuoSpan, h_quo: QuoSpan) -> int | None:
        """Block count of the el_template ``(v_quo, h_quo)``, or None if it is not a real one.

        Args:
            v_quo: The vertical arm of the el-template.
            h_quo: The horizontal arm of the el-template.

        Returns:
            The block count, or ``None`` if the pair is not a real el-template.
        """
        return el_geometry.el_template_length(self.m, v_quo, h_quo)

    def _missing_internal_couplers(self) -> frozenset[tuple[Node, Node]]:
        """Find internal couplers that SHOULD exist (ideal geometry) but are absent.

        Internal couplers join a vertical node ``v = (x, y, k)`` to a horizontal node
        ``hp = (x+-1, y+-1, kp)`` for every ``kp in range(t)``. For each present vertical node and
        each of its four diagonal horizontal neighbours (in bounds, present), we check whether the
        coupler exists in ``edges``; if not, record it.

        Returns:
            The missing internal couplers as ``(v, hp)`` pairs, vertical node first.
        """
        abs_min, abs_max, t = self.abs_min, self.abs_max, self.t
        edges, present, missing = self.edges, self.present_qubits, self.missing_qubits
        krange = range(t)
        missing_couplers = set()
        for v_kind in _V_KIND_BY_SHIFT.values():
            for v, v_lcoord in present[v_kind].items():
                x, y, k = v
                for dx, dy in _INTERNAL_COUPLER_OFFSETS:
                    hp_x = x + dx
                    if hp_x < abs_min or hp_x > abs_max:
                        continue
                    hp_y = y + dy
                    if hp_y < abs_min or hp_y > abs_max:
                        continue
                    hp_kind = _H_KIND_BY_SHIFT[hp_x & 3]
                    hp_nodes = present[hp_kind]
                    hp_missing = missing[hp_kind]
                    # internal couplers are all-to-all in k: v's k couples to
                    # every hp kp, so there is no k == kp guard here.
                    for kp in krange:
                        hp = (hp_x, hp_y, kp)
                        if hp in hp_missing:
                            continue
                        hp_lcoord = hp_nodes[hp]
                        coupler = (
                            (v_lcoord, hp_lcoord) if v_lcoord < hp_lcoord else (hp_lcoord, v_lcoord)
                        )
                        if coupler not in edges:
                            missing_couplers.add((v, hp))
        return frozenset(missing_couplers)

    def _survey_ext(self) -> Runs:
        """Walk every external line and split it into RUNS (maximal intact stretches).

        An external line is ``(kind, fixed_coord)``; nodes on it step by 4 in the varying
        coordinate. Starting from each present node, we walk forward while the next node exists and
        the connecting external coupler is present; a break (missing node or missing coupler) ends
        the current run and starts scanning for the next one.

        Returns:
            ``runs[kind][coord][k]`` -> set of ``(start, end)`` run intervals. This nested shape
            mirrors how nodes are keyed (``kind -> coord -> k``) and is what :meth:`el_reachable`
            and :meth:`_survey_side_order` consume.
        """
        abs_min, abs_max, t = self.abs_min, self.abs_max, self.t
        edges, present, missing = self.edges, self.present_qubits, self.missing_qubits
        runs = {k: {} for k in _NODE_KINDS}
        for vertical, kind_by_shift in ((True, _V_KIND_BY_SHIFT), (False, _H_KIND_BY_SHIFT)):
            for shift, kind in kind_by_shift.items():
                kind_nodes, kind_missing = present[kind], missing[kind]
                for fixed_coord in range(abs_min, abs_max + 1, 2):
                    coord_map = runs[kind].setdefault(fixed_coord, {})
                    for k in range(t):
                        runs_for_k = set()
                        coord = shift
                        while coord <= abs_max:
                            if vertical:
                                node = (fixed_coord, coord, k)
                            else:
                                node = (coord, fixed_coord, k)
                            if node not in kind_missing:
                                prev = coord
                                inc = 4
                                end = prev
                                next_coord = None
                                while prev + inc <= abs_max:
                                    if vertical:
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

    def el_reachable(self, v: Node, h: Node) -> tuple[QuoSpan, QuoSpan] | None:
        """Does the L-shaped ("el") shortest path from v to h exist in the faulty grid?

        The el-path travels along v's external line to the elbow with h's external line, hops one
        internal coupler (between vp and hp), then travels along h's line to h. It exists iff:

        * the block pair ``(v_block, h_block)`` has an el_template (i.e. the two lines cross in
          the ideal grid),
        * none of the four involved nodes v, vp, h, hp are missing,
        * the connecting internal coupler is not in missing_internal_couplers,
        * both required quotient runs are actually covered by real runs for ``v``'s and ``h``'s
          ``k`` in the faulty grid.

        Lazy + memoized. The block pair determines the el_template uniquely; the el_template is
        computed on demand, so a query is a handful of arithmetic ops. Results (including ``None``,
        for unreachable) are cached, so repeated queries across overlapping windows in a sweep are
        effectively free.

        Args:
            v: The vertical endpoint.
            h: The horizontal endpoint.

        Returns:
            ``(v_quo_span, h_quo_span)`` if ``v`` and ``h`` are reachable, else ``None``. The path's
            vertical arm is ``v_quo_span`` at ``v``'s ``k``, its horizontal arm ``h_quo_span`` at
            ``h``'s ``k``.
        """
        cache = self._el_cache
        hit = cache.get((v, h), _EL_REACHABLE_NOT_CACHED)
        if hit is not _EL_REACHABLE_NOT_CACHED:  # cached: a descriptor tuple, or None (unreachable)
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
            v_kind = _V_KIND_BY_SHIFT[v_y_shift]
            h_kind = _H_KIND_BY_SHIFT[h_x_shift]
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
                    result = (v_quo_span, h_quo_span)

        cache[(v, h)] = result  # store None directly (unreachable is a real answer)
        return result

    def clear_el_cache(self) -> None:
        """Empty the el_reachable memo cache (self._el_cache).

        el_reachable memoizes each ``(v, h)`` result. Within one window the four corners share those
        lookups, but a worker processes many windows before it is recycled, so left alone the memo
        would grow without bound. Calling this once per window keeps it scoped to a single window's
        work. Cheap -- it just drops the dict; results are recomputed on demand as needed.
        """
        self._el_cache.clear()

    def _survey_side_order(self) -> SideOrder:
        """Order the nodes along each window side, by block and then by reach.

        Every present node lies on exactly one run (see :meth:`_survey_ext`). Its reach from a side
        is the distance, in coordinate units, from the node to the end of its run in the direction
        pointing away from that side; the run is not clipped to any window. ``y`` grows downward,
        so a vertical node's reach from the top ("t") is ``run_end - y`` and from the bottom ("b")
        ``y - run_start``; a horizontal node's reach from the left ("l") is ``run_end - x`` and
        from the right ("r") ``x - run_start``.

        Returns:
            ``{side: {(x, y): {k: order}}}``. Sides "t" and "b" hold vertical blocks, "l" and "r"
            horizontal blocks. Blocks with no present node are absent.
        """
        abs_max, t = self.abs_max, self.t
        runs = self.runs

        from_top, from_bottom, from_left, from_right = {}, {}, {}, {}

        # vertical: x even; two lines per x (v1, v3)
        for x in range(0, abs_max + 1, 2):
            reach_from_top_by_y, reach_from_bottom_by_y = {}, {}
            for kind in _V_KIND_BY_SHIFT.values():
                runs_by_k = runs[kind].get(x, {})
                for k, runs_for_k in runs_by_k.items():
                    for run_start, run_end in runs_for_k:
                        y = run_start
                        while y <= run_end:
                            reach_from_top = reach_from_top_by_y.get(y)
                            if reach_from_top is None:
                                reach_from_top = reach_from_top_by_y[y] = {}
                                reach_from_bottom_by_y[y] = {}
                            reach_from_top[k] = run_end - y
                            reach_from_bottom_by_y[y][k] = y - run_start
                            y += 4
            for y, reach_from_top in reach_from_top_by_y.items():
                from_top[(x, y)] = _block_side_order(x, y, reach_from_top, t)
                from_bottom[(x, y)] = _block_side_order(x, y, reach_from_bottom_by_y[y], t)

        # horizontal: y even; two lines per y (h1, h3)
        for y in range(0, abs_max + 1, 2):
            reach_from_left_by_x, reach_from_right_by_x = {}, {}
            for kind in _H_KIND_BY_SHIFT.values():
                runs_by_k = runs[kind].get(y, {})
                for k, runs_for_k in runs_by_k.items():
                    for run_start, run_end in runs_for_k:
                        x = run_start
                        while x <= run_end:
                            reach_from_left = reach_from_left_by_x.get(x)
                            if reach_from_left is None:
                                reach_from_left = reach_from_left_by_x[x] = {}
                                reach_from_right_by_x[x] = {}
                            reach_from_left[k] = run_end - x
                            reach_from_right_by_x[x][k] = x - run_start
                            x += 4
            for x, reach_from_left in reach_from_left_by_x.items():
                from_left[(x, y)] = _block_side_order(y, x, reach_from_left, t)
                from_right[(x, y)] = _block_side_order(y, x, reach_from_right_by_x[x], t)

        return {"t": from_top, "b": from_bottom, "l": from_left, "r": from_right}


def _covers(runs_for_k: Iterable[Run], a: int, b: int) -> bool:
    """True if some run ``(start, end)`` in the set fully spans ``[a, b]``.

    That is, ``start <= a and end >= b``. Used to test whether the faulty grid provides an intact
    stretch long enough for an el_template's required quo_span.

    Args:
        runs_for_k: The runs ``(start, end)`` on one ``(kind, fixed_coord, k)`` line.
        a: Lower endpoint of the span that must be covered.
        b: Upper endpoint of the span that must be covered.

    Returns:
        ``True`` if some run satisfies ``start <= a and end >= b``, else ``False``.
    """
    for s, e in runs_for_k:
        if s <= a and e >= b:
            return True
    return False


def _block_side_order(
    along_side: int, across_side: int, reach_by_k: dict[int, int], t: int
) -> dict[int, int]:
    """Side orders of one block's nodes: rank ``r`` by ascending reach gets
    ``1 + (t + 1) * along_side + j + 2 r``.

    Args:
        along_side: The block's coordinate along the window side: ``x`` for a vertical block,
            ``y`` for a horizontal one.
        across_side: The block's other coordinate (``y`` for vertical, ``x`` for horizontal);
            its residue mod 4 gives the Zephyr ``j`` index.
        reach_by_k: Maps each present ``k`` of the block to its reach from the side.
        t: The Zephyr tile size (number of parallel qubits per line).

    Returns:
        ``{orig_k: order}``.
    """
    j = ((across_side - 1) & 3) // 2  # Zephyr j of this line
    base = 1 + (t + 1) * along_side + j
    return {
        k: base + 2 * rank
        for rank, (k, _) in enumerate(sorted(reach_by_k.items(), key=itemgetter(1)))
    }


def _parse_zephyr_metadata(G: nx.Graph) -> tuple[int, int, str]:
    """Validate a Zephyr graph's metadata and infer its node-label mode.

    The label mode is inferred from the node type, not from graph metadata: integer nodes are linear
    indices ("int"), 5-tuples are Zephyr coordinates ``(u, w, k, j, z)`` ("coordinates"), and
    3-tuples are cartesian ``(x, y, k)`` ("cartesian").

    Args:
        G: A networkx Zephyr graph.

    Returns:
        ``(m, t, labels)``: The Zephyr grid size, the tile size, and the label mode.

    Raises:
        ValueError: If the graph is not a Zephyr graph, lacks size metadata, or has nodes of an
            unrecognized type.
    """
    info = G.graph
    family = info.get("family")
    if family != "zephyr":
        raise ValueError(f"Expected a graph with zephyr topology, got family={family!r}")
    m = info.get("rows")
    if m is None:
        m = info.get("columns")
    t = info.get("tile")
    if m is None or t is None:
        raise ValueError("zephyr graph missing 'rows'/'columns'/'tile' metadata")

    # handle empty graphs
    if not G:
        raise ValueError("cannot build a Grid from an empty graph")

    # Decide the label mode from the ACTUAL node type/shape, which is
    # unambiguous, rather than trusting graph.graph["labels"] (dwave.graphs
    # spells it "coordinate"; other producers may differ or omit it).
    sample = next(iter(G.nodes()))
    if isinstance(sample, tuple):
        if len(sample) == 5:
            return m, t, "coordinates"
        if len(sample) == 3:
            return m, t, "cartesian"
        raise ValueError(
            f"cannot infer zephyr label mode from tuple node {sample!r}; "
            f"expected a 5-tuple (Zephyr) or 3-tuple (cartesian)"
        )
    if isinstance(sample, numbers.Integral) and not isinstance(sample, bool):
        return m, t, "int"
    raise ValueError(
        f"cannot infer zephyr label mode from node {sample!r}; expected "
        f"an int (linear), a 5-tuple (Zephyr), or a 3-tuple (cartesian)"
    )

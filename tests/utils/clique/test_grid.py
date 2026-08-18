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


import random
import unittest

import networkx as nx
from dwave.graphs import zephyr_graph
from parameterized import parameterized

from minorminer.utils._clique import el_geometry
from minorminer.utils._clique.grid import (
    Grid,
    _covers,
    _ranked_pos_fast,
    cartesian_to_zephyr,
    zephyr_to_cartesian,
)


# --------------------------------------------------------------------------
# helpers: build ideal / faulty Zephyr graphs via dwave.graphs, in any label mode.
# Faults are expressed in cartesian (x, y, k) coordinates -- grid's internal space --
# and mapped into whatever label mode the graph is built in.
# --------------------------------------------------------------------------
def zephyr(m, t, drop_nodes=(), drop_edges=(), labels="cartesian"):
    drop_nodes = set(drop_nodes)
    dropE = {(a, b) if a < b else (b, a) for (a, b) in drop_edges}

    full = zephyr_graph(m, t, coordinates=True, data=True)   # authoritative topology
    z2i = dict(full.nodes(data="linear_index"))                  # Zephyr 5-tuple -> linear int

    pres_z = [z for z in full.nodes() if zephyr_to_cartesian(z) not in drop_nodes]
    presset = set(pres_z)
    pres_e = []
    for za, zb in full.edges():
        if za not in presset or zb not in presset:
            continue
        ca, cb = zephyr_to_cartesian(za), zephyr_to_cartesian(zb)
        ce = (ca, cb) if ca < cb else (cb, ca)
        if ce in dropE:
            continue
        pres_e.append((za, zb))

    if labels == "coordinates":
        return zephyr_graph(m, t, node_list=pres_z, edge_list=pres_e, coordinates=True)
    if labels == "int":
        nl = [z2i[z] for z in pres_z]
        el = [(z2i[a], z2i[b]) for (a, b) in pres_e]
        return zephyr_graph(m, t, node_list=nl, edge_list=el, coordinates=False)
    # cartesian: dwave.graphs has no cartesian labelling, so relabel a coordinate
    # graph's nodes to (x, y, k) -- grid infers "cartesian" from the 3-tuple.
    gz = zephyr_graph(m, t, node_list=pres_z, edge_list=pres_e, coordinates=True)
    return nx.relabel_nodes(gz, {z: zephyr_to_cartesian(z) for z in gz.nodes()})


def verticals(g):
    return list(g.present["v1"]) + list(g.present["v3"])


def horizontals(g):
    return list(g.present["h1"]) + list(g.present["h3"])


def find_reachable(g, min_v_arm=0):
    for v in verticals(g):
        for h in horizontals(g):
            tmpl = el_geometry.el_template(g.m, v[0], v[1], h[0], h[1])
            if tmpl is None:
                continue
            vp_y, hp_x, v_quo, h_quo = tmpl
            if v_quo[3] - v_quo[2] < min_v_arm:
                continue
            if g.el_reachable(v, h) is None:
                continue
            return {
                "v": v, "h": h,
                "vp": (v[0], vp_y, v[2]), "hp": (hp_x, h[1], h[2]),
                "v_quo": v_quo, "h_quo": h_quo,
            }
    return None


def find_unreachable(g):
    for v in verticals(g):
        for h in horizontals(g):
            if g.el_reachable(v, h) is None:
                return (v, h)
    return None


_CONVERT_CASES = [
    ("vert_origin", (0, 0, 0, 0, 0), (0, 1, 0)),
    ("vert_inner", (0, 3, 1, 1, 2), (6, 11, 1)),
    ("horz_origin", (1, 0, 0, 0, 0), (1, 0, 0)),
    ("horz_inner", (1, 3, 1, 1, 2), (11, 6, 1)),
]


class TestGrid(unittest.TestCase):
    def setUp(self):
        self.m, self.t = 3, 2
        self.g = Grid.from_graph(zephyr(self.m, self.t))

    # ---------------- module-level coordinate converters ----------------
    @parameterized.expand(_CONVERT_CASES)
    def test_zephyr_to_cartesian(self, _name, zc, cc):
        self.assertEqual(zephyr_to_cartesian(zc), cc)

    @parameterized.expand(_CONVERT_CASES)
    def test_cartesian_to_zephyr(self, _name, zc, cc):
        self.assertEqual(cartesian_to_zephyr(cc), zc)

    def test_coordinate_roundtrip(self):
        full = zephyr_graph(self.m, self.t, coordinates=True)
        for z in full.nodes():
            self.assertEqual(cartesian_to_zephyr(zephyr_to_cartesian(z)), z)

    # ---------------- module-level helpers ----------------
    @parameterized.expand([
        ("spans", [(1, 9)], 3, 7, True),
        ("exact", [(1, 9)], 1, 9, True),
        ("gap", [(1, 5), (9, 9)], 3, 7, False),
        ("empty", [], 1, 1, False),
    ])
    def test_covers(self, _name, runs, a, b, expected):
        self.assertEqual(_covers(runs, a, b), expected)

    @parameterized.expand([("vertical", True), ("horizontal", False)])
    def test_ranked_pos_fast(self, _name, vertical):
        if vertical:
            out = _ranked_pos_fast(2, 5, {0: 5, 1: 1, 2: 3}, 4, True)
            const_slot, var_slot = 1, 0
        else:
            out = _ranked_pos_fast(5, 2, {0: 5, 1: 1, 2: 3}, 4, False)
            const_slot, var_slot = 0, 1
        self.assertEqual(set(out), {0, 1, 2})
        self.assertEqual(len({out[k][const_slot] for k in out}), 1)   # constant slot
        # ranked by ascending distance: k=1 (1) < k=2 (3) < k=0 (5)
        self.assertLess(out[1][var_slot], out[2][var_slot])
        self.assertLess(out[2][var_slot], out[0][var_slot])

    # ---------------- from_graph: label modes ----------------
    def test_from_graph_cartesian_perfect(self):
        g = self.g
        self.assertEqual((g.m, g.t, g.labels), (self.m, self.t, "cartesian"))
        self.assertTrue(all(len(s) == 0 for s in g.missing.values()))
        self.assertEqual(len(g.missing_int), 0)

    @parameterized.expand([("int",), ("coordinates",)])
    def test_label_mode_matches_cartesian(self, labels):
        drop_n = [(2, 5, 0)]
        drop_e = [((2, 1, 0), (2, 9, 0))]
        gc = Grid.from_graph(zephyr(self.m, self.t, drop_n, drop_e, "cartesian"))
        go = Grid.from_graph(zephyr(self.m, self.t, drop_n, drop_e, labels))
        self.assertEqual(go.labels, labels)
        for attr in ("present", "missing", "_edges", "runs", "missing_int", "pos"):
            self.assertEqual(getattr(gc, attr), getattr(go, attr))

    def test_columns_metadata_fallback(self):
        G = zephyr(self.m, self.t)
        G.graph.pop("rows", None)                    # force the columns fallback
        G.graph["columns"] = self.m
        g = Grid.from_graph(G)
        self.assertEqual(g.m, self.m)

    # ---------------- from_graph: error / edge branches ----------------
    @parameterized.expand([
        ("bad_family", {"family": "chimera", "rows": 2, "tile": 2}, [(0, 1, 0)]),
        ("no_rows_or_columns", {"family": "zephyr", "tile": 2}, [(0, 1, 0)]),
        ("no_tile", {"family": "zephyr", "rows": 2}, [(0, 1, 0)]),
        ("bad_tuple_length", {"family": "zephyr", "rows": 2, "tile": 2}, [(0, 0, 0, 0)]),
        ("non_int_non_tuple", {"family": "zephyr", "rows": 2, "tile": 2}, ["not-a-node"]),
        ("bool_node", {"family": "zephyr", "rows": 2, "tile": 2}, [True]),
    ])
    def test_from_graph_invalid_raises(self, _name, meta, nodes):
        G = nx.Graph()
        G.graph.update(meta)
        G.add_nodes_from(nodes)
        with self.assertRaises(ValueError):
            Grid.from_graph(G)

    def test_cartesian_edge_to_unclassified_node_skipped(self):
        # a non-lattice node (even/even parity) never gets a linear index, so an
        # edge touching it hits the `ra is None -> continue` guard in ingestion.
        G = zephyr(self.m, self.t, labels="cartesian")
        G.add_node((0, 0, 0))                        # invalid parity -> unclassified
        G.add_edge((0, 1, 0), (0, 0, 0))
        g = Grid.from_graph(G)
        self.assertEqual(g._edges, self.g._edges)    # bogus edge dropped
        self.assertIsNone(g.cartesian_to_linear((0, 0, 0)))

    def test_coordinates_edge_to_out_of_range_node_skipped(self):
        # a Zephyr coord with z == m maps outside [0, 4m] and is never classified,
        # so its edge is skipped during coordinate ingestion.
        G = zephyr(self.m, self.t, labels="coordinates")
        far = (0, 0, 0, 0, self.m)                   # -> cartesian (0, 4m+1, 0)
        G.add_node(far)
        G.add_edge((0, 0, 0, 0, 0), far)
        g = Grid.from_graph(G)
        self.assertEqual(g.labels, "coordinates")
        self.assertIsNone(g.cartesian_to_linear(zephyr_to_cartesian(far)))

    # ---------------- accessors ----------------
    @parameterized.expand([
        ("v1", (2, 1, 0), "v1"),
        ("v3", (2, 3, 0), "v3"),
        ("h1", (1, 2, 0), "h1"),
        ("h3", (3, 2, 0), "h3"),
    ])
    def test_kind_of(self, _name, coord, kind):
        self.assertEqual(self.g.kind_of(coord), kind)

    def test_cartesian_to_linear_present_and_absent(self):
        self.assertIsInstance(self.g.cartesian_to_linear((2, 5, 0)), int)
        self.assertIsNone(self.g.cartesian_to_linear((2, 5, 99)))

    def test_is_present(self):
        self.assertTrue(self.g.is_present((2, 5, 0)))
        self.assertFalse(self.g.is_present((2, 5, 99)))

    def test_edges_property(self):
        self.assertIs(self.g.edges, self.g._edges)
        self.assertIsInstance(self.g.edges, set)

    def test_has_edge_order_independent(self):
        r1 = self.g.cartesian_to_linear((2, 5, 0))
        r2 = self.g.cartesian_to_linear((2, 9, 0))          # external neighbour
        self.assertTrue(self.g.has_edge(r1, r2))
        self.assertTrue(self.g.has_edge(r2, r1))
        r3 = self.g.cartesian_to_linear((10, 5, 0))         # far, uncoupled
        self.assertFalse(self.g.has_edge(r1, r3))

    # ---------------- el-template geometry ----------------
    def test_el_template_length_valid_and_none(self):
        info = find_reachable(self.g, min_v_arm=4)
        self.assertIsNotNone(info)
        length = self.g.el_template_length(info["v_quo"], info["h_quo"])
        v_a, v_b = info["v_quo"][2], info["v_quo"][3]
        h_a, h_b = info["h_quo"][2], info["h_quo"][3]
        self.assertEqual(length, (v_b - v_a + h_b - h_a + 8) // 4)
        # a shift that cannot match its own endpoints -> not a real template
        self.assertIsNone(self.g.el_template_length((0, 3, 1, 1), (1, 0, 1, 1)))

    # ---------------- _missing_int ----------------
    def test_missing_int_records_missing_internal_coupler(self):
        v, hp = (2, 5, 0), (3, 6, 0)                        # diagonal internal pair
        g = Grid.from_graph(zephyr(self.m, self.t, drop_edges=[(v, hp)]))
        self.assertIn((v, hp), g.missing_int)
        self.assertEqual(g.missing_int[(v, hp)], ((2, 1, 0), (3, 6, 0)))

    def test_missing_int_skips_absent_horizontal_neighbor(self):
        # removing a diagonal horizontal node must be skipped (not recorded, no
        # KeyError); the grid still builds and missing_int stays empty.
        g = Grid.from_graph(zephyr(self.m, self.t, drop_nodes=[(3, 6, 0)]))
        self.assertEqual(len(g.missing_int), 0)

    # ---------------- _survey_ext ----------------
    @parameterized.expand([
        ("perfect", [], [], {(1, 9)}),
        ("missing_node", [(2, 5, 0)], [], {(1, 1), (9, 9)}),
        ("leading_missing_node", [(2, 1, 0)], [], {(5, 9)}),
        ("broken_coupler", [], [((2, 1, 0), (2, 5, 0))], {(1, 1), (5, 9)}),
    ])
    def test_survey_ext_runs(self, _name, drop_nodes, drop_edges, expected):
        g = Grid.from_graph(zephyr(self.m, self.t, drop_nodes, drop_edges))
        self.assertEqual(g.runs["v1"].get(2, {}).get(0), expected)

    # ---------------- el_reachable ----------------
    def test_el_reachable_reachable_returns_descriptor(self):
        info = find_reachable(self.g, min_v_arm=4)
        self.assertIsNotNone(info)
        v, h = info["v"], info["h"]
        self.assertEqual(self.g.el_reachable(v, h),
                         (info["v_quo"], info["h_quo"], v[2], h[2]))

    def test_el_reachable_no_template_returns_none(self):
        pair = find_unreachable(self.g)
        self.assertIsNotNone(pair)
        self.assertIsNone(self.g.el_reachable(*pair))

    @parameterized.expand([("missing_node",), ("missing_internal",), ("run_not_covered",)])
    def test_el_reachable_returns_none(self, reason):
        info = find_reachable(self.g, min_v_arm=4)
        self.assertIsNotNone(info)
        v, h, vp, hp = info["v"], info["h"], info["vp"], info["hp"]
        if reason == "missing_node":
            g = Grid.from_graph(zephyr(self.m, self.t, drop_nodes=[vp]))
        elif reason == "missing_internal":
            g = Grid.from_graph(zephyr(self.m, self.t, drop_edges=[(vp, hp)]))
            self.assertIn((vp, hp), g.missing_int)
        else:  # run_not_covered: break an external coupler inside the vertical arm
            g = Grid.from_graph(zephyr(self.m, self.t,
                                       drop_edges=[(v, (v[0], v[1] + 4, v[2]))]))
            self.assertTrue(g.is_present(v) and g.is_present(vp))
        self.assertIsNone(g.el_reachable(v, h))

    def test_el_reachable_caches_result_and_none(self):
        info = find_reachable(self.g, min_v_arm=4)
        v, h = info["v"], info["h"]
        first = self.g.el_reachable(v, h)
        self.assertIn((v, h), self.g._el_cache)
        self.assertEqual(self.g.el_reachable(v, h), first)     # cache-hit path
        pair = find_unreachable(self.g)
        self.g.el_reachable(*pair)
        self.assertIn(pair, self.g._el_cache)
        self.assertIsNone(self.g._el_cache[pair])

    def test_clear_el_cache(self):
        info = find_reachable(self.g, min_v_arm=4)
        self.g.el_reachable(info["v"], info["h"])
        self.assertGreater(len(self.g._el_cache), 0)
        self.g.clear_el_cache()
        self.assertEqual(self.g._el_cache, {})

    # ---------------- _position_quo ----------------
    def test_position_quo_structure(self):
        pos = self.g.pos
        self.assertEqual(set(pos), {"r", "l", "b", "t"})
        self.assertIn((2, 5), pos["b"])
        self.assertIn((2, 5), pos["t"])
        for _k, val in pos["b"][(2, 5)].items():
            self.assertEqual(len(val), 2)
        self.assertIn((1, 2), pos["l"])
        self.assertIn((1, 2), pos["r"])

    def test_position_quo_backfills_empty_lines(self):
        drop = ([(4, y, k) for y in range(0, 4 * self.m + 1) for k in range(self.t)]
                + [(x, 4, k) for x in range(0, 4 * self.m + 1) for k in range(self.t)])
        g = Grid.from_graph(zephyr(self.m, self.t, drop_nodes=drop))
        self.assertEqual(g.pos["b"][(4, 1)], {})     # emptied vertical column
        self.assertEqual(g.pos["t"][(4, 1)], {})
        self.assertEqual(g.pos["l"][(1, 4)], {})     # emptied horizontal row
        self.assertEqual(g.pos["r"][(1, 4)], {})

    # ---------------- realistic faulty grid (typical yields) ----------------
    def test_random_faulty_grid_invariants(self):
        m, t = 4, 4
        rng = random.Random(20260818)
        full = zephyr_graph(m, t, coordinates=True)
        cart_nodes = [zephyr_to_cartesian(z) for z in full.nodes()]
        cart_edges = [(zephyr_to_cartesian(a), zephyr_to_cartesian(b)) for a, b in full.edges()]
        drop_nodes = {n for n in cart_nodes if rng.random() < 0.05}       # ~95% node yield
        remaining = [e for e in cart_edges
                     if e[0] not in drop_nodes and e[1] not in drop_nodes]
        drop_edges = {e for e in remaining if rng.random() < 0.01}        # ~99% coupler yield
        g = Grid.from_graph(zephyr(m, t, drop_nodes, drop_edges, "coordinates"))

        ideal_by_kind = {}
        for n in cart_nodes:
            ideal_by_kind.setdefault(g.kind_of(n), set()).add(n)
        for kind in ("v1", "v3", "h1", "h3"):
            self.assertEqual(set(g.present[kind]) | set(g.missing[kind]), ideal_by_kind[kind])
            self.assertEqual(set(g.present[kind]) & set(g.missing[kind]), set())

        # runs: ordered, endpoints present, consecutive nodes actually coupled
        for kind, per_coord in g.runs.items():
            is_v = kind[0] == "v"
            for coord, per_k in per_coord.items():
                for k, runset in per_k.items():
                    for s, e in runset:
                        self.assertLessEqual(s, e)
                        prev = None
                        for c in range(s, e + 1, 4):
                            node = (coord, c, k) if is_v else (c, coord, k)
                            self.assertTrue(g.is_present(node))
                            if prev is not None:
                                self.assertTrue(g.has_edge(g.cartesian_to_linear(prev),
                                                           g.cartesian_to_linear(node)))
                            prev = node

        # missing_int only references present nodes and genuinely-absent couplers
        for (v, hp), _desc in g.missing_int.items():
            self.assertTrue(g.is_present(v))
            self.assertTrue(g.is_present(hp))
            self.assertFalse(g.has_edge(g.cartesian_to_linear(v), g.cartesian_to_linear(hp)))

        # el_reachable returns a well-formed descriptor or None (sampled)
        Vs, Hs = verticals(g), horizontals(g)
        checked = 0
        for v in Vs[:15]:
            for h in Hs[:15]:
                res = g.el_reachable(v, h)
                if res is not None:
                    v_quo, h_quo, vk, hk = res
                    self.assertEqual((len(v_quo), len(h_quo)), (4, 4))
                    self.assertEqual((vk, hk), (v[2], h[2]))
                    checked += 1
        self.assertGreater(checked, 0)

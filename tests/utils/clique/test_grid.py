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
from dwave.graphs import zephyr_coordinates, zephyr_graph
from parameterized import parameterized

from minorminer.utils._clique import el_geometry
from minorminer.utils._clique.grid import Grid, _block_side_order, _covers
from .faulty_zephyr import faulty_zephyr


def verticals(g):
    return list(g.present_qubits["v1"]) + list(g.present_qubits["v3"])


def horizontals(g):
    return list(g.present_qubits["h1"]) + list(g.present_qubits["h3"])


def find_reachable(g, min_v_arm=0):
    for v in verticals(g):
        for h in horizontals(g):
            template = el_geometry.el_template(g.m, v[0], v[1], h[0], h[1])
            if template is None:
                continue
            vp_y, hp_x, v_quo, h_quo = template
            if v_quo[3] - v_quo[2] < min_v_arm:
                continue
            if g.el_reachable(v, h) is None:
                continue
            return {
                "v": v,
                "h": h,
                "vp": (v[0], vp_y, v[2]),
                "hp": (hp_x, h[1], h[2]),
                "v_quo": v_quo,
                "h_quo": h_quo,
            }
    return None


def find_unreachable(g):
    for v in verticals(g):
        for h in horizontals(g):
            if g.el_reachable(v, h) is None:
                return (v, h)
    return None


class TestGrid(unittest.TestCase):
    def setUp(self):
        self.m, self.t = 3, 2
        self.g = Grid(faulty_zephyr(self.m, self.t))

    @parameterized.expand(
        [
            ("spans", [(1, 9)], 3, 7, True),
            ("exact", [(1, 9)], 1, 9, True),
            ("gap", [(1, 5), (9, 9)], 3, 7, False),
            ("empty", [], 1, 1, False),
        ]
    )
    def test_covers(self, _name, runs, a, b, expected):
        self.assertEqual(_covers(runs, a, b), expected)

    @parameterized.expand(
        [
            # across_side 5 -> j = 0, base = 1 + 5 * 2 + 0 = 11
            ("j0", 2, 5, 11),
            # across_side 7 -> j = 1, base = 1 + 5 * 2 + 1 = 12
            ("j1", 2, 7, 12),
        ]
    )
    def test_block_side_order(self, _name, along_side, across_side, base):
        out = _block_side_order(along_side, across_side, {0: 5, 1: 1, 2: 3}, 4)
        # ranked by ascending reach: k=1 (1) < k=2 (3) < k=0 (5), stepping by 2
        self.assertEqual(out, {1: base, 2: base + 2, 0: base + 4})

    def test_side_order_ranks_follow_each_blocks_reach(self):
        m, t = 6, 2
        g = Grid(faulty_zephyr(m, t, drop_edges=[((2, 5, 0), (2, 9, 0)), ((2, 1, 1), (2, 5, 1))]))
        self.assertEqual(g.runs["v1"][2][0], {(1, 5), (9, 21)})
        self.assertEqual(g.runs["v1"][2][1], {(1, 1), (5, 21)})

        base = 1 + (t + 1) * 2 + 0  # x = 2; y % 4 == 1, so j = 0
        top = g.side_order["t"]
        self.assertEqual(top[(2, 1)], {1: base, 0: base + 2})
        self.assertEqual(top[(2, 5)], {0: base, 1: base + 2})
        # the same numbers, held by different k's
        self.assertEqual(set(top[(2, 1)].values()), set(top[(2, 5)].values()))
        self.assertNotEqual(top[(2, 1)], top[(2, 5)])

    def test_cartesian_perfect(self):
        g = self.g
        self.assertEqual((g.m, g.t, g.labels), (self.m, self.t, "cartesian"))
        self.assertTrue(all(len(s) == 0 for s in g.missing_qubits.values()))
        self.assertEqual(len(g.missing_internal_couplers), 0)

    @parameterized.expand([("int",), ("coordinates",), ("cartesian",)])
    def test_perfect_graph_smoke(self, labels):
        m, t = 2, 4
        G = faulty_zephyr(m, t, labels=labels)
        g = Grid(G)
        self.assertEqual(g.labels, labels)
        # every coupler made it into the linear edge set
        self.assertEqual(len(g.edges), G.number_of_edges())
        # perfect grid: each (kind, fixed_coord, k) line is one unbroken run end to end
        for kind, runs_by_coord in g.runs.items():
            shift = int(kind[1])
            for runs_by_k in runs_by_coord.values():
                for k in range(t):
                    self.assertEqual(runs_by_k[k], {(shift, shift + 4 * (m - 1))})
        # every block on every side has a side order for all t qubits
        for side in ("t", "b", "l", "r"):
            self.assertTrue(g.side_order[side])
            for orders in g.side_order[side].values():
                self.assertEqual(len(orders), t)

    @parameterized.expand([("int",), ("coordinates",)])
    def test_label_mode_matches_cartesian(self, labels):
        drop_n = [(2, 5, 0)]
        drop_e = [((2, 1, 0), (2, 9, 0))]
        gc = Grid(
            faulty_zephyr(self.m, self.t, drop_nodes=drop_n, drop_edges=drop_e, labels="cartesian")
        )
        go = Grid(
            faulty_zephyr(self.m, self.t, drop_nodes=drop_n, drop_edges=drop_e, labels=labels)
        )
        self.assertEqual(go.labels, labels)
        for attr in (
            "present_qubits",
            "missing_qubits",
            "edges",
            "runs",
            "missing_internal_couplers",
            "side_order",
        ):
            self.assertEqual(getattr(gc, attr), getattr(go, attr))

    def test_columns_metadata_fallback(self):
        G = faulty_zephyr(self.m, self.t)
        G.graph.pop("rows", None)  # force the columns fallback
        G.graph["columns"] = self.m
        g = Grid(G)
        self.assertEqual(g.m, self.m)

    @parameterized.expand(
        [
            ("bad_family", {"family": "chimera", "rows": 2, "tile": 2}, [(0, 1, 0)]),
            ("no_rows_or_columns", {"family": "zephyr", "tile": 2}, [(0, 1, 0)]),
            ("no_tile", {"family": "zephyr", "rows": 2}, [(0, 1, 0)]),
            ("bad_tuple_length", {"family": "zephyr", "rows": 2, "tile": 2}, [(0, 0, 0, 0)]),
            ("non_int_non_tuple", {"family": "zephyr", "rows": 2, "tile": 2}, ["not-a-node"]),
            ("bool_node", {"family": "zephyr", "rows": 2, "tile": 2}, [True]),
            ("empty_graph", {"family": "zephyr", "rows": 2, "tile": 2}, []),
        ]
    )
    def test_invalid_raises(self, _name, meta, nodes):
        G = nx.Graph()
        G.graph.update(meta)
        G.add_nodes_from(nodes)
        with self.assertRaises(ValueError):
            Grid(G)

    def test_cartesian_edge_to_unclassified_node_skipped(self):
        G = faulty_zephyr(self.m, self.t, labels="cartesian")
        G.add_node((0, 0, 0))  # invalid parity -> unclassified
        G.add_edge((0, 1, 0), (0, 0, 0))
        g = Grid(G)
        self.assertEqual(g.edges, self.g.edges)  # bogus edge dropped
        self.assertIsNone(g.cartesian_to_linear((0, 0, 0)))

    def test_coordinates_edge_to_out_of_range_node_skipped(self):
        # a Zephyr coord with z == m maps outside [0, 4m] and is never classified,
        # so its edge is skipped during coordinate ingestion.
        G = faulty_zephyr(self.m, self.t, labels="coordinates")
        far = (0, 0, 0, 0, self.m)  # -> cartesian (0, 4m+1, 0)
        G.add_node(far)
        G.add_edge((0, 0, 0, 0, 0), far)
        g = Grid(G)
        self.assertEqual(g.labels, "coordinates")
        self.assertIsNone(g.cartesian_to_linear(zephyr_coordinates.zephyr_to_cartesian(far)))

    @parameterized.expand(
        [
            ("v1", (2, 1, 0), "v1"),
            ("v3", (2, 3, 0), "v3"),
            ("h1", (1, 2, 0), "h1"),
            ("h3", (3, 2, 0), "h3"),
        ]
    )
    def test_kind_of(self, _name, coord, kind):
        self.assertEqual(self.g.kind_of(coord), kind)

    def test_cartesian_to_linear_present_and_absent(self):
        self.assertIsInstance(self.g.cartesian_to_linear((2, 5, 0)), int)
        self.assertIsNone(self.g.cartesian_to_linear((2, 5, 99)))

    def test_is_present(self):
        self.assertTrue(self.g.is_present((2, 5, 0)))
        self.assertFalse(self.g.is_present((2, 5, 99)))

    def test_has_edge_order_independent(self):
        r1 = self.g.cartesian_to_linear((2, 5, 0))
        r2 = self.g.cartesian_to_linear((2, 9, 0))  # external neighbour
        self.assertTrue(self.g.has_edge(r1, r2))
        self.assertTrue(self.g.has_edge(r2, r1))
        r3 = self.g.cartesian_to_linear((10, 5, 0))  # far, uncoupled
        self.assertFalse(self.g.has_edge(r1, r3))

    def test_neighbors_match_graph(self):
        # cartesian labels, so G's adjacency is directly comparable; the other label modes
        # build the same grid (test_label_mode_matches_cartesian)
        dropped = (2, 5, 0)
        drop_e = [((2, 1, 1), (2, 5, 1)), ((2, 1, 1), (3, 2, 0)), ((1, 2, 0), (3, 2, 0))]
        G = faulty_zephyr(self.m, self.t, drop_nodes=[dropped], drop_edges=drop_e)
        g = Grid(G)
        for c in G.nodes():
            nbrs = g.neighbors(c)
            self.assertEqual(len(nbrs), len(set(nbrs)))
            self.assertEqual(set(nbrs), set(G[c]))
        self.assertEqual(g.neighbors(dropped), [])

    @parameterized.expand(
        [
            # dy = -4 would leave the grid (y = -1); the coupler to (9, 4, 1) is missing
            (
                "interior",
                (8, 3, 0),
                [
                    (7, 2, 0),
                    (7, 2, 1),
                    (7, 4, 0),
                    (7, 4, 1),
                    (9, 2, 0),
                    (9, 2, 1),
                    (9, 4, 0),
                    (8, 1, 0),
                    (8, 5, 0),
                    (8, 7, 0),
                ],
            ),
            # x = 12 is the right edge: no dx = +1, and dy = +4 would give y = 13
            (
                "boundary",
                (12, 9, 1),
                [
                    (11, 8, 0),
                    (11, 8, 1),
                    (11, 10, 0),
                    (11, 10, 1),
                    (12, 7, 1),
                    (12, 11, 1),
                    (12, 5, 1),
                ],
            ),
            # 11 neighbors in the ideal grid; the missing coupler to (8, 3, 0) leaves 10;
            # dx = +4 would give x = 13, off the grid
            (
                "missing_coupler",
                (9, 4, 1),
                [
                    (8, 3, 1),
                    (8, 5, 0),
                    (8, 5, 1),
                    (10, 3, 0),
                    (10, 3, 1),
                    (10, 5, 0),
                    (10, 5, 1),
                    (7, 4, 1),
                    (11, 4, 1),
                    (5, 4, 1),
                ],
            ),
            ("absent_vertical", (6, 5, 1), []),
            ("absent_horizontal", (5, 2, 0), []),
        ]
    )
    def test_neighbors_hand_checked(self, _name, node, expected):
        g = Grid(
            faulty_zephyr(
                self.m,
                self.t,
                drop_nodes=[(5, 2, 0), (6, 5, 1)],
                drop_edges=[((8, 3, 0), (9, 4, 1))],
            )
        )
        self.assertCountEqual(g.neighbors(node), expected)

    def test_el_template_length_valid_and_none(self):
        info = find_reachable(self.g, min_v_arm=4)
        self.assertIsNotNone(info)
        length = self.g.el_template_length(info["v_quo"], info["h_quo"])
        v_a, v_b = info["v_quo"][2], info["v_quo"][3]
        h_a, h_b = info["h_quo"][2], info["h_quo"][3]
        self.assertEqual(length, (v_b - v_a + h_b - h_a + 8) // 4)
        # a shift that cannot match its own endpoints -> not a real template
        self.assertIsNone(self.g.el_template_length((0, 3, 1, 1), (1, 0, 1, 1)))

    def test_missing_internal_couplers_records_missing_internal_coupler(self):
        v, hp = (2, 5, 0), (3, 6, 0)  # diagonal internal pair
        g = Grid(faulty_zephyr(self.m, self.t, drop_edges=[(v, hp)]))
        self.assertEqual(g.missing_internal_couplers, frozenset({(v, hp)}))

    def test_missing_internal_couplers_skips_absent_horizontal_neighbor(self):
        # removing a diagonal horizontal node must be skipped (not recorded, no
        # KeyError); the grid still builds and missing_internal_couplers stays empty.
        g = Grid(faulty_zephyr(self.m, self.t, drop_nodes=[(3, 6, 0)]))
        self.assertEqual(len(g.missing_internal_couplers), 0)

    @parameterized.expand(
        [
            ("perfect", [], [], {(1, 9)}),
            ("missing_node", [(2, 5, 0)], [], {(1, 1), (9, 9)}),
            ("leading_missing_node", [(2, 1, 0)], [], {(5, 9)}),
            ("broken_coupler", [], [((2, 1, 0), (2, 5, 0))], {(1, 1), (5, 9)}),
        ]
    )
    def test_survey_ext_runs(self, _name, drop_nodes, drop_edges, expected):
        g = Grid(faulty_zephyr(self.m, self.t, drop_nodes=drop_nodes, drop_edges=drop_edges))
        self.assertEqual(g.runs["v1"].get(2, {}).get(0), expected)

    def test_el_reachable_reachable_returns_descriptor(self):
        info = find_reachable(self.g, min_v_arm=4)
        self.assertIsNotNone(info)
        v, h = info["v"], info["h"]
        self.assertEqual(self.g.el_reachable(v, h), (info["v_quo"], info["h_quo"]))

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
            g = Grid(faulty_zephyr(self.m, self.t, drop_nodes=[vp]))
        elif reason == "missing_internal":
            g = Grid(faulty_zephyr(self.m, self.t, drop_edges=[(vp, hp)]))
            self.assertIn((vp, hp), g.missing_internal_couplers)
        else:  # run_not_covered: break an external coupler inside the vertical arm
            g = Grid(faulty_zephyr(self.m, self.t, drop_edges=[(v, (v[0], v[1] + 4, v[2]))]))
            self.assertTrue(g.is_present(v) and g.is_present(vp))
        self.assertIsNone(g.el_reachable(v, h))

    def test_el_reachable_caches_result_and_none(self):
        info = find_reachable(self.g, min_v_arm=4)
        v, h = info["v"], info["h"]
        first = self.g.el_reachable(v, h)
        self.assertIn((v, h), self.g._el_cache)
        self.assertEqual(self.g.el_reachable(v, h), first)  # cache-hit path
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

    def test_survey_side_order_structure(self):
        side_order = self.g.side_order
        self.assertEqual(set(side_order), {"t", "b", "l", "r"})
        self.assertIn((2, 5), side_order["t"])
        self.assertIn((2, 5), side_order["b"])
        for order in side_order["b"][(2, 5)].values():
            self.assertIsInstance(order, int)
        self.assertIn((1, 2), side_order["l"])
        self.assertIn((1, 2), side_order["r"])

    def test_survey_side_order_omits_uncovered_blocks(self):
        drop = [(4, y, k) for y in range(0, 4 * self.m + 1) for k in range(self.t)] + [
            (x, 4, k) for x in range(0, 4 * self.m + 1) for k in range(self.t)
        ]
        g = Grid(faulty_zephyr(self.m, self.t, drop_nodes=drop))
        self.assertNotIn((4, 1), g.side_order["t"])  # emptied vertical column
        self.assertNotIn((4, 1), g.side_order["b"])
        self.assertNotIn((1, 4), g.side_order["l"])  # emptied horizontal row
        self.assertNotIn((1, 4), g.side_order["r"])

    def test_random_faulty_grid_invariants(self):
        m, t = 4, 4
        rng = random.Random(20260818)
        full = zephyr_graph(m, t, coordinates=True)
        cart_nodes = [zephyr_coordinates.zephyr_to_cartesian(z) for z in full.nodes()]
        cart_edges = [
            (zephyr_coordinates.zephyr_to_cartesian(a), zephyr_coordinates.zephyr_to_cartesian(b))
            for a, b in full.edges()
        ]
        drop_nodes = {n for n in cart_nodes if rng.random() < 0.05}  # ~95% node yield
        remaining = [e for e in cart_edges if e[0] not in drop_nodes and e[1] not in drop_nodes]
        drop_edges = {e for e in remaining if rng.random() < 0.01}  # ~99% coupler yield
        g = Grid(
            faulty_zephyr(
                m, t, drop_nodes=drop_nodes, drop_edges=drop_edges, labels="coordinates"
                )
            )

        ideal_by_kind = {}
        for n in cart_nodes:
            ideal_by_kind.setdefault(g.kind_of(n), set()).add(n)
        for kind in ("v1", "v3", "h1", "h3"):
            self.assertEqual(
                set(g.present_qubits[kind]) | set(g.missing_qubits[kind]), ideal_by_kind[kind]
            )
            self.assertEqual(set(g.present_qubits[kind]) & set(g.missing_qubits[kind]), set())

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
                                self.assertTrue(
                                    g.has_edge(
                                        g.cartesian_to_linear(prev), g.cartesian_to_linear(node)
                                    )
                                )
                            prev = node

        # missing_internal_couplers only references present nodes and genuinely-absent couplers
        for v, hp in g.missing_internal_couplers:
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
                    v_quo, h_quo = res
                    self.assertEqual((len(v_quo), len(h_quo)), (4, 4))
                    checked += 1
        self.assertGreater(checked, 0)

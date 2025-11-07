# Copyright 2025 D-Wave Systems Inc.
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
#
# ================================================================================================


import os
import pickle
from collections import Counter
from functools import cache
from itertools import product
from unittest import TestCase

import dwave_networkx as dnx

from minorminer.cube_embedding import TileKind, ZCoupling
from minorminer.cube_embedding.perfect_cube_embedding import find_cube_embedding
from minorminer._lattice_utils import ZLatticeSurvey



PARAM_SETS = [
    ("case1", (4, 2, (1, 1, 1), True)),
    ("case2", (4, 2, (1, 1, 1), False)),
    ("case3", (2, 6, (1, 1, 1), False)),
    ("case4", (4, 2, (4, 2, 3), True)),
    ("case5", (4, 2, (4, 2, 3), False)),
    ("case6", (2, 6, (2, 2, 12), True)),
    ("case7", (3, 4, (2, 2, 14), False)),
    ("case8", (3, 4, (3, 3, 10), True)),
    ("case9", (3, 4, (8, 3, 2), False)),
]

# for each (m, t) in [(4, 2), (2, 6), (3, 4)], gives a tuple where
# - 0-th element is a randomly generated zephyr graph by removal of 3% of nodes and 2% of remaining edges.
# - 1-st element is the ZLatticeSurvey of graph
HERE = os.path.dirname(__file__)
RAND_GRAPH_PATH = os.path.join(HERE, "random_zephyr_graphs.pkl")
with open(RAND_GRAPH_PATH, "rb") as f:
    rand_graphs = pickle.load(f)

FOUND_CUBOIDS = [
    ("bay20-case1", ((8, 8, 8), True, TileKind.LADDER, ZCoupling.EITHER)),
    ("bay20-case2", ((7, 7, 7), True, TileKind.LADDER, ZCoupling.ZERO_ONE_ONE_ZERO)),
    ("bay20-case3", ((9, 8, 8), True, None, ZCoupling.EITHER)),
    ("bay20-case4", ((8, 8, 8), False, None, ZCoupling.ZERO_ONE_ONE_ZERO)),
    ("bay24-case1", ((5, 5, 6), True, TileKind.LADDER, ZCoupling.ZERO_ONE)),
    ("bay24-case2", ((5, 5, 5), False, TileKind.LADDER, ZCoupling.ZERO_ONE_ONE_ZERO)),
    ("bay24-case3", ((5, 5, 6), True, TileKind.LADDER, ZCoupling.ONE_ZERO)),
]


BAY20_GRAPH_PATH = os.path.join(HERE, "bay20_hardcoded.pkl")
with open(BAY20_GRAPH_PATH, "rb") as f:
    bay20 = pickle.load(f)

BAY24_GRAPH_PATH = os.path.join(HERE, "bay24_hardcoded.pkl")
with open(BAY24_GRAPH_PATH, "rb") as f:
    bay24 = pickle.load(f)


@cache
def cached_cuboid_embedding(m, t, dim, periodic):
    G, sur = rand_graphs[(m, t)]
    return find_cube_embedding(dim=dim, G=G, z_periodic=periodic, lattice_survey=sur)


@cache
def cached_cuboid_emb_bays(
    name: str,
    dim,
    periodic,
    tile_kind: TileKind | None = None,
    coupling: ZCoupling | None = None,
):
    if name.startswith("bay20"):
        G, sur = bay20, ZLatticeSurvey(bay20)
    elif name.startswith("bay24"):
        G, sur = bay24, ZLatticeSurvey(bay24)
    return find_cube_embedding(
        dim=dim,
        G=G,
        z_periodic=periodic,
        lattice_survey=sur,
        tile_kind=tile_kind,
        z_coupling=coupling,
    )


@cache
def cached_lsurvey_perfect(m, t):
    G = dnx.zephyr_graph(m=m, t=t)
    return G, ZLatticeSurvey(G=G)


class TestCuboidEmbedding(TestCase):
    def test_finds_max_on_complete_zephyr(self):
        for m, t, coupling in [
            (4, 2, ZCoupling.EITHER),
            (3, 4, ZCoupling.ZERO_ONE),
            (6, 4, ZCoupling.ZERO_ONE_ONE_ZERO),
            (2, 3, ZCoupling.ONE_ZERO),
        ]:
            G, lattice_survey = cached_lsurvey_perfect(m, t)
            dim = (m, m, 4 * t)
            for periodic in [True, False]:
                emb = find_cube_embedding(
                    dim=dim,
                    G=G,
                    z_periodic=periodic,
                    lattice_survey=lattice_survey,
                    z_coupling=coupling,
                )
                msg = f"On perfect Zephyr graph with{(m, t) = }, no embedding of cuboid with dimension {dim} was found"
                self.assertEqual(len(emb), m * m * 4 * t, msg=msg)

            L = min(m, 4 * t)
            for periodic in [True, False]:
                for tile_kind in TileKind:
                    emb = find_cube_embedding(
                        G=G,
                        z_periodic=periodic,
                        lattice_survey=lattice_survey,
                        tile_kind=tile_kind,
                    )
                    msg = f"On perfect Zephyr graph with{(m, t) = }, no embedding with tile-kind {tile_kind} of cube with dimension {(L, L, L)} and {periodic = } was found"
                    self.assertEqual(len(emb), L * L * L, msg=msg)


    def test_finds_big_on_chips(self):
        for name, (dim, periodic, tile_kind, coupling) in FOUND_CUBOIDS:
            emb = cached_cuboid_emb_bays(
                name, dim, periodic, tile_kind=tile_kind, coupling=coupling
            )
            self.assertTrue(
                emb,
                msg=f"No embedding of cuboid with dimensions {dim} with {periodic = } with tile kind {tile_kind = }, {coupling = } on hard-coded {name} graph was found",
            )

    def test_finds_embedding_on_small_dims(self):
        for name, (m, t, dim, periodic) in PARAM_SETS:
            with self.subTest(case=name):
                emb = cached_cuboid_embedding(m, t, dim, periodic)
                self.assertTrue(
                    emb,
                    f"No embedding of cuboid with dimensions {dim} with {periodic = } on an instance of zephyr graph with {(m, t) = } was found",
                )

    def test_embedding_covers_all_nodes(self):
        for name, (m, t, dim, periodic) in PARAM_SETS:
            with self.subTest(case=name):
                emb = cached_cuboid_embedding(m, t, dim, periodic)
                for x, y, z in product(range(dim[0]), range(dim[1]), range(dim[2])):
                    self.assertTrue(
                        (x, y, z) in emb,
                        f"In the embedding of cuboid with dimensions {dim} with {periodic = } on an instance of zephyr graph with {(m, t) = }, node {(x, y, z)} hasn't been mapped",
                    )

        for name, (dim, periodic, tile_kind, coupling) in FOUND_CUBOIDS:
            with self.subTest(case=name):
                emb = cached_cuboid_emb_bays(
                    name, dim, periodic, tile_kind=tile_kind, coupling=coupling
                )
                for x, y, z in product(range(dim[0]), range(dim[1]), range(dim[2])):
                    self.assertTrue(
                        (x, y, z) in emb,
                        msg=f"In embedding of cuboid with dimensions {dim} with {periodic = } with tile kind {tile_kind = }, {coupling = } on hard-coded {name}, , node {(x, y, z)} hasn't been mapped",
                    )


    def test_chains_are_disjoint(self):
        for name, (m, t, dim, periodic) in PARAM_SETS:
            with self.subTest(case=name):
                emb = cached_cuboid_embedding(m, t, dim, periodic)
                counter_node = Counter([node for chain in emb.values() for node in chain])
                for node, occ_num in counter_node.items():
                    self.assertEqual(
                        occ_num,
                        1,
                        msg=f"In embedding of cuboid with dimensions {dim} with {periodic = } on an instance of zephyr graph with {(m, t) = }, node {node} has occured {occ_num} times",
                    )

        for name, (dim, periodic, tile_kind, coupling) in FOUND_CUBOIDS:
            with self.subTest(case=name):
                emb = cached_cuboid_emb_bays(
                    name, dim, periodic, tile_kind=tile_kind, coupling=coupling
                )
                counter_node = Counter([node for chain in emb.values() for node in chain])
                for node, occ_num in counter_node.items():
                    self.assertEqual(
                        occ_num,
                        1,
                        msg=f"In embedding of cuboid with dimensions {dim} with {periodic = } with tile kind {tile_kind = }, {coupling = } on hard-coded {name}, node {node} has occured {occ_num} times",
                    )

    def test_use_only_graph_nodes(self):
        for name, (m, t, dim, periodic) in PARAM_SETS:
            with self.subTest(case=name):
                emb = cached_cuboid_embedding(m, t, dim, periodic)
                G_nodes = rand_graphs[(m, t)][0].nodes()
                nodes_used = [node for chain in emb.values() for node in chain]
                for node in nodes_used:
                    self.assertTrue(
                        node in G_nodes,
                        msg=f"In embedding of cuboid with dimensions {dim} with {periodic = } node {node} has been used which is not in an instance of zephyr graph with {(m, t) = }",
                    )

        bay20_nodes = bay20.nodes()
        bay24_nodes = bay24.nodes()
        for name, (dim, periodic, tile_kind, coupling) in FOUND_CUBOIDS:
            with self.subTest(case=name):
                emb = cached_cuboid_emb_bays(
                    name, dim, periodic, tile_kind=tile_kind, coupling=coupling
                )
                nodes_used = [node for chain in emb.values() for node in chain]
                if name.startswith("bay20"):
                    bay_nodes = bay20_nodes
                elif name.startswith("bay24"):
                    bay_nodes = bay24_nodes
                for node in nodes_used:
                    self.assertTrue(
                        node in bay_nodes,
                        msg=f"In embedding of cuboid with dimensions {dim} with {periodic = } with tile kind {tile_kind = }, {coupling = } on hard-coded {name}, node {node} has been used which is not in hard-coded {name}",
                    )

    def test_use_only_graph_edges(self):
        for name, (m, t, dim, periodic) in PARAM_SETS:
            with self.subTest(case=name):
                emb = cached_cuboid_embedding(m, t, dim, periodic)
                G_edges = rand_graphs[(m, t)][0].edges()
                for chain in emb.values():
                    self.assertTrue(
                        chain in G_edges,
                        msg=f"In embedding of cuboid with dimensions {dim} with {periodic = } edge {chain} has been used which is not in an instance of zephyr graph with {(m, t) = }",
                    )

        bay20_edges = bay20.edges()
        bay24_edges = bay24.edges()
        for name, (dim, periodic, tile_kind, coupling) in FOUND_CUBOIDS:
            with self.subTest(case=name):
                emb = cached_cuboid_emb_bays(
                    name, dim, periodic, tile_kind=tile_kind, coupling=coupling
                )
                if name.startswith("bay20"):
                    bay_edges = bay20_edges
                elif name.startswith("bay24"):
                    bay_edges = bay24_edges
                for chain in emb.values():
                    self.assertTrue(
                        chain in bay_edges,
                        msg=f"In embedding of cuboid with dimensions {dim} with {periodic = } with tile kind {tile_kind = }, {coupling = } on hard-coded {name}, edge {chain} has been used which is not in hard-coded {name}",
                    )

    def test_chains_are_connected(self):
        def is_chain(c, G_edges):
            x, y = c
            if (x, y) in G_edges or (y, x) in G_edges:
                return True
            return False

        for name, (m, t, dim, periodic) in PARAM_SETS:
            with self.subTest(case=name):
                emb = cached_cuboid_embedding(m, t, dim, periodic)
                G_edges = rand_graphs[(m, t)][0].edges()
                for xyz_chain in emb.values():
                    self.assertTrue(
                        is_chain(xyz_chain, G_edges),
                        f"In the embedding of cuboid with dimensions {dim} on an instance of zephyr graph with {(m, t) = }, chain {xyz_chain} is not connected",
                    )
        bay20_edges = bay20.edges()
        bay24_edges = bay24.edges()

        for name, (dim, periodic, tile_kind, coupling) in FOUND_CUBOIDS:
            with self.subTest(case=name):
                emb = cached_cuboid_emb_bays(
                    name, dim, periodic, tile_kind=tile_kind, coupling=coupling
                )
                if name.startswith("bay20"):
                    bay_edges = bay20_edges
                elif name.startswith("bay24"):
                    bay_edges = bay24_edges
                for chain in emb.values():
                    self.assertTrue(
                        is_chain(chain, bay_edges),
                        msg=f"In embedding of cuboid with dimensions {dim} with {periodic = } with tile kind {tile_kind = }, {coupling = } on hard-coded {name}, chain {chain} is not connected",
                    )

    def test_embedding_covers_all_edges(self):
        def cube_edges(dim, z_periodic):
            cube_nodes = list(product(range(dim[0]), range(dim[1]), range(dim[2])))
            generic_cube_edges = (
                [((x, y, z), (x + 1, y, z)) for (x, y, z) in cube_nodes if x + 1 in range(dim[0])]
                + [((x, y, z), (x, y + 1, z)) for (x, y, z) in cube_nodes if y + 1 in range(dim[1])]
                + [((x, y, z), (x, y, z + 1)) for (x, y, z) in cube_nodes if z + 1 in range(dim[2])]
            )
            if z_periodic:
                periodic_cube_edges = [
                    ((x, y, dim[2] - 1), (x, y, 0)) for x in range(dim[0]) for y in range(dim[1])
                ]
            else:
                periodic_cube_edges = []
            return generic_cube_edges + periodic_cube_edges

        def chains_have_connection(c1, c2, G_edges):
            for x in c1:
                for y in c2:
                    if (x, y) in G_edges or (y, x) in G_edges:
                        return True
            return False

        for name, (m, t, dim, periodic) in PARAM_SETS:
            with self.subTest(case=name):
                emb = cached_cuboid_embedding(m, t, dim, periodic)
                G_edges = rand_graphs[(m, t)][0].edges()
                needed_cube_edges = cube_edges(dim=dim, z_periodic=periodic)
                for u, v in needed_cube_edges:
                    chain_u, chain_v = emb.get(u), emb.get(v)
                    self.assertTrue(
                        chains_have_connection(chain_u, chain_v, G_edges),
                        f"In the embedding of cuboid with dimensions {dim} with {periodic = } on an instance of zephyr graph with {(m, t) = }, there is no edge between {chain_u, chain_v}",
                    )
        bay20_edges = bay20.edges()
        bay24_edges = bay24.edges()

        for name, (dim, periodic, tile_kind, coupling) in FOUND_CUBOIDS:
            with self.subTest(case=name):
                emb = cached_cuboid_emb_bays(
                    name, dim, periodic, tile_kind=tile_kind, coupling=coupling
                )
                if name.startswith("bay20"):
                    bay_edges = bay20_edges
                elif name.startswith("bay24"):
                    bay_edges = bay24_edges
                needed_cube_edges = cube_edges(dim=dim, z_periodic=periodic)
                for u, v in needed_cube_edges:
                    chain_u, chain_v = emb.get(u), emb.get(v)
                    self.assertTrue(
                        chains_have_connection(chain_u, chain_v, bay_edges),
                        msg=f"In embedding of cuboid with dimensions {dim} with {periodic = } with tile kind {tile_kind = }, {coupling = } on hard-coded {name}, there is no edge between {chain_u, chain_v}",
                    )

    def test_coupling(self):
        def chains_are_coupled(c1, c2, G_edges, coupling):
            convert = dnx.zephyr_coordinates(m=12, t=4).linear_to_zephyr
            for x in c1:
                x_conv = convert(x)
                if x_conv[0] == 0:
                    c1_v = x
                else:
                    c1_h = x
            for x in c2:
                x_conv = convert(x)
                if x_conv[0] == 0:
                    c2_v = x
                else:
                    c2_h = x
            if coupling is ZCoupling.ZERO_ONE_ONE_ZERO:
                return (c1_v, c2_h) in G_edges and (c2_v, c1_h) in G_edges
            if coupling is ZCoupling.ZERO_ONE:
                return (c1_v, c2_h) in G_edges
            if coupling is ZCoupling.ONE_ZERO:
                return (c1_h, c2_v) in G_edges

        bay20_edges = bay20.edges()
        bay24_edges = bay24.edges()

        for name, (dim, periodic, tile_kind, coupling) in FOUND_CUBOIDS:
            if coupling is ZCoupling.EITHER or coupling is None:
                continue
            with self.subTest(case=name):
                emb = cached_cuboid_emb_bays(
                    name, dim, periodic, tile_kind=tile_kind, coupling=coupling
                )
                if name.startswith("bay20"):
                    bay_edges = bay20_edges
                elif name.startswith("bay24"):
                    bay_edges = bay24_edges
                for x, y, z in product(range(dim[0]), range(dim[1]), range(dim[2])):
                    if z + 1 in range(dim[2]):
                        u = (x, y, z)
                        v = (x, y, z + 1)
                    elif periodic:
                        u = (x, y, z)
                        v = (x, y, 0)
                    else:
                        continue
                    chain_u, chain_v = emb.get(u), emb.get(v)
                    self.assertTrue(
                        chains_are_coupled(chain_u, chain_v, bay_edges, coupling=coupling),
                        msg=f"In embedding of cuboid with dimensions {dim} with {periodic = } with tile kind {tile_kind = }, {coupling = } on hard-coded {name}, the coupling between {chain_u, chain_v} is not as it should be",
                    )


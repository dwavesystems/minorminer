# Copyright 2024 D-Wave
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest
import numpy as np

import networkx as nx

import dwave_networkx as dnx

from minorminer.utils.feasibility import (
    embedding_feasibility_filter,
    lattice_size,
    lattice_size_lower_bound,
)


def random_tree(n, seed=np.random):
    # networkx used to have a quick function to make random trees, which
    # was deprecated and is now gone in modern versions.  This way of making
    # trees is highly nonuniform but that's really not an issue for us

    if n <= 2:
        return nx.complete_graph(n)

    return nx.from_prufer_sequence(seed.choice(n, n - 2))


def construct_chain_major(G, construct_chain):
    _random = np.random
    _shuffle = _random.shuffle
    half_edge_label = {}

    H = nx.Graph()
    for v, nbrs in G._adj.items():
        if nbrs:
            nbrs = list(nbrs)
            _shuffle(nbrs)
            construct_chain(H, half_edge_label, v, nbrs)
        else:
            H.add_node((0, v))

    H.add_edges_from((half_edge_label[u, v], half_edge_label[v, u]) for u, v in G.edges)
    return H


def random_linear_chain_major(G):
    def construct_chain(H, half_edge_label, v, nbrs):
        nx.add_path(H, ((i, v) for i in range(len(nbrs))))
        half_edge_label.update(((v, u), (i, v)) for i, u in enumerate(nbrs))

    return construct_chain_major(G, construct_chain)


def random_tree_chain_major(G):
    _random = np.random
    _choice = _random.choice
    _randint = _random.randint

    def construct_chain(H, half_edge_label, v, nbrs):
        deg = len(nbrs)
        chain_length = 1 if deg == 1 else _randint(1, deg)
        tree = random_tree(chain_length, seed=_random)
        H.add_edges_from(((i, v), (j, v)) for i, j in tree.edges)

        # now let's populate the tree with half-edges -- put at least one on
        # each leaf, and randomly distribute the rest
        index = [i for i, d in tree.degree if d == 1]
        index.extend(_choice(chain_length, deg - len(index)))

        half_edge_label.update(((v, u), (i, v)) for i, u in zip(index, nbrs))

    return construct_chain_major(G, construct_chain)


def random_split_major(G):
    _random = np.random
    _randint = _random.randint
    if len(G) <= 2:
        return random_tree(_randint(len(G), 4), _random)

    # first, subdivide about half of nodes
    def construct_chain(H, half_edge_label, v, nbrs):
        if _randint(0, 1):
            H.add_edge((0, v), (1, v))
            half_edge_label.update(((v, u), (_randint(0, 1), v)) for u in nbrs)
        else:
            half_edge_label.update(((v, u), (0, v)) for u in nbrs)

    H = construct_chain_major(G, construct_chain)

    # next, subdivide about half of edges
    remove_edges = [e for e in H.edges if _randint(0, 1)]
    H.remove_edges_from(remove_edges)
    H.add_edges_from((x, e) for e in remove_edges for x in e)

    return H


class TestEmbeddings(unittest.TestCase):
    def test_embedding_feasibility_filter(self):
        m = 7  # Odd m
        T = dnx.chimera_graph(m)
        S = dnx.chimera_graph(m - 1)
        for one_to_one in [True, False]:
            self.assertTrue(
                embedding_feasibility_filter(S, T, one_to_one=one_to_one),
                "embedding expected to be feasible",
            )
        S.add_edges_from(
            (i, i + 1) for i in range(S.number_of_nodes(), T.number_of_nodes())
        )
        for one_to_one in [True, False]:
            self.assertFalse(embedding_feasibility_filter(S, T, one_to_one=one_to_one))
        # Too many edges:
        S = dnx.zephyr_graph(m // 2)
        for one_to_one in [True, False]:
            self.assertFalse(embedding_feasibility_filter(S, T, one_to_one=one_to_one))
        # Subtle failure case (by ordered degrees filter):
        m = 4
        T = dnx.chimera_graph(m)
        S = dnx.chimera_torus(m - 1)
        self.assertTrue(
            S.number_of_edges() < T.number_of_edges()
            and S.number_of_nodes() < T.number_of_nodes()
        )
        self.assertFalse(
            embedding_feasibility_filter(S, T, one_to_one=True),
            "Should fail because not enough connectivity 6 nodes",
        )
        self.assertTrue(
            embedding_feasibility_filter(S, T, one_to_one=False),
            "T {5:64, 6: 64}; S {6: 72}; making 8 degree-8 "
            "chains, each from 2 degree-5 nodes, allows embedding",
        )
        # Check tetrahedron cannot be embedded on a graph with a triangle + 0,1,2 connectivity nodes.

        self.assertTrue(
            embedding_feasibility_filter(nx.empty_graph(), T),
            "it's always feasible to embed the empty graph",
        )
        self.assertFalse(
            embedding_feasibility_filter(S, nx.empty_graph()),
            "it's always infeasible to embed into an empty graph",
        )

    def test_lattice_size_subgraph_upper_bound(self):
        L = np.random.randint(2) + 2
        T = dnx.zephyr_graph(L - 1)
        self.assertEqual(L - 1, lattice_size(T=T))
        T = dnx.pegasus_graph(L)
        self.assertEqual(L, lattice_size(T=T))
        T = dnx.chimera_graph(L, L - 1, 1)
        self.assertEqual(L, lattice_size(T=T))

    def test_lattice_size_lower_bound(self):
        L = np.random.randint(2) + 2
        T = dnx.zephyr_graph(L - 1)
        self.assertEqual(L - 1, lattice_size_lower_bound(S=T, T=T, one_to_one=True))
        self.assertEqual(
            L - 1, lattice_size_lower_bound(S=T, topology="zephyr", one_to_one=True)
        )
        # Test raise error when T and topology are inconsistent
        with self.assertRaises(ValueError):
            lattice_size_lower_bound(S=T, T=T, topology="chimera")

        T = dnx.pegasus_graph(L)
        self.assertEqual(L, lattice_size_lower_bound(S=T, T=T, one_to_one=True))
        self.assertEqual(
            L, lattice_size_lower_bound(S=T, topology="pegasus", one_to_one=True)
        )
        T = dnx.chimera_graph(L, L - 1, 1)
        self.assertEqual(L, lattice_size_lower_bound(S=T, T=T, one_to_one=True))
        self.assertEqual(
            L, lattice_size_lower_bound(S=T, topology="chimera", t=1, one_to_one=True)
        )

        # Test raise error when T and topology is both none
        with self.assertRaises(ValueError):
            lattice_size_lower_bound(S=T, T=None, topology=None)
        # Test raise error when graph is not dwave networkx graph
        S = nx.complete_graph(5)
        with self.assertRaises(ValueError):
            lattice_size_lower_bound(S=S, T=S, t=1)

        m = 6
        S = dnx.chimera_graph(m)  # Embeds onto Zephyr[m//2]
        self.assertEqual(
            m // 2, lattice_size_lower_bound(S=S, topology="zephyr", one_to_one=True)
        )
        T = dnx.zephyr_graph(m)
        self.assertEqual(m // 2, lattice_size_lower_bound(S=S, T=T, one_to_one=True))

    def test_lattice_size_lower_bound_one_to_one_false(self):
        self.skipTest(
            "TODO: make lattice_size_lower_bound work in the one_to_one=False case"
        )

        S = nx.complete_graph(12)
        self.assertEqual(
            lattice_size_lower_bound(S=S, topology="chimera", one_to_one=False), 3
        )

    def test_embedding_feasibility_filter_atlas_and_ER(self):
        for i, G in enumerate(nx.atlas.graph_atlas_g()):
            H = random_linear_chain_major(G)
            self.assertTrue(
                embedding_feasibility_filter(G, H),
                f"Atlas graph {i} with random linear chains",
            )
            H = random_tree_chain_major(G)
            self.assertTrue(
                embedding_feasibility_filter(G, H),
                f"Atlas graph {i} with random tree chains",
            )
            H = random_split_major(G)
            self.assertTrue(
                embedding_feasibility_filter(G, H),
                f"Atlas graph {i} with random split major",
            )

        G = nx.erdos_renyi_graph(1000, 0.005)
        H = random_linear_chain_major(G)
        self.assertTrue(
            embedding_feasibility_filter(G, H),
            f"Erdos-Renyi graph with random linear chains",
        )
        H = random_tree_chain_major(G)
        self.assertTrue(
            embedding_feasibility_filter(G, H),
            f"Erdos-Renyi graph with random tree chains",
        )
        H = random_split_major(G)
        self.assertTrue(
            embedding_feasibility_filter(G, H),
            f"Erdos-Renyi graph with random split major",
        )


if __name__ == "__main__":
    unittest.main()

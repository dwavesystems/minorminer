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
from minorminer import find_embedding

from minorminer.utils.parallel_embeddings import (
    lattice_size,
    shuffle_graph,
    embeddings_to_array,
    find_multiple_embeddings,
    find_sublattice_embeddings,
)


class TestLatticeSize(unittest.TestCase):
    # This is a very basic helper function
    def test_lattice_size(self):
        L = np.random.randint(2) + 2
        T = dnx.zephyr_graph(L - 1)
        self.assertEqual(L - 1, lattice_size(T=T))
        T = dnx.pegasus_graph(L)
        self.assertEqual(L, lattice_size(T=T))
        T = dnx.chimera_graph(L, L - 1, 1)
        self.assertEqual(L, lattice_size(T=T))


class TestEmbeddings(unittest.TestCase):

    def test_shuffle_graph(self):
        prng = np.random.default_rng()
        T = dnx.zephyr_graph(1)
        Ts = shuffle_graph(T, prng)
        self.assertEqual(list(T.nodes()), list(T.nodes()))
        self.assertEqual(list(T.edges()), list(T.edges()))
        self.assertNotEqual(list(T.nodes()), list(Ts.nodes()))
        self.assertNotEqual(list(T.edges()), list(Ts.edges()))

        seed = 42
        prng1 = np.random.default_rng(seed)
        G1 = shuffle_graph(T, seed=prng1)
        prng2 = np.random.default_rng(seed)
        G2 = shuffle_graph(T, seed=prng2)
        self.assertEqual(
            list(G1.nodes), list(G2.nodes), "seed does not allow reproducibility"
        )
        self.assertEqual(
            list(G1.edges), list(G2.edges), "seed does not allow reproducibility"
        )
        prng3 = np.random.default_rng(seed + 1)
        G2 = shuffle_graph(T, seed=prng3)
        self.assertNotEqual(
            list(G1.nodes), list(G2.nodes), "different seeds give same Graph"
        )
        self.assertNotEqual(
            list(G1.edges), list(G2.edges), "different seeds give same Graph"
        )

    def test_embeddings_to_array(self):
        """Test the embeddings_to_array function with various inputs"""

        # embedding without node_order
        embs = [{0: 0, 1: 1}, {0: 2, 1: 3}]
        expected = np.array([[0, 1], [2, 3]])
        result = embeddings_to_array(embs, as_ndarray=True)
        np.testing.assert_array_equal(
            result, expected, "Failed to convert embeddings without node_order."
        )
        result = embeddings_to_array(embs, as_ndarray=False)
        self.assertEqual(type(result), list)
        np.testing.assert_array_equal(
            np.array(result),
            expected,
            "Failed to convert embeddings without node_order.",
        )

        # embedding with node order
        node_order = [1, 0]
        expected = np.array([[1, 0], [3, 2]])
        result = embeddings_to_array(embs, node_order=node_order, as_ndarray=True)
        np.testing.assert_array_equal(
            result, expected, "Failed to convert embeddings with node_order."
        )

        # empty embedding wthout node order, raises error in numpy case as
        # 2d array expected but the shape is unknown in the variables dimension.
        embeddings_to_array([], node_order=None, as_ndarray=False)
        with self.assertRaises(ValueError):
            embeddings_to_array([], node_order=None, as_ndarray=True)

        # empty embedding with node order
        node_order = [0, 1]
        expected = np.empty((0, 2), dtype=int)  # Shape (0, number of nodes)
        result = embeddings_to_array([], node_order=node_order, as_ndarray=True)
        np.testing.assert_array_equal(
            result, expected, "Failed to handle empty embeddings with node_order."
        )

        # inconsistent node_order with embeddings
        node_order = [2, 0]
        with self.assertRaises(KeyError):
            embeddings_to_array(embs, node_order=node_order)

    def test_find_multiple_embeddings_basic(self):
        square = {
            ((0, 0), (0, 1)),
            ((0, 1), (1, 1)),
            ((1, 1), (1, 0)),
            ((1, 0), (0, 0)),
        }
        L = 2
        squares = {
            tuple((v1 + i, v2 + j) for v1, v2 in e)
            for i in range(L)
            for j in range(L)
            for e in square
        }
        S = nx.from_edgelist(square)
        T = nx.from_edgelist(squares)  # Room for 1!
        embs = find_multiple_embeddings(S, T)
        self.assertEqual(len(embs), 1, "mismatched number of embeddings")
        L = 3  # Room for 4 embeddings, might find less (as few as 1)
        squares = {
            tuple((v1 + i, v2 + j) for v1, v2 in e)
            for i in range(L)
            for j in range(L)
            for e in square
        }
        T = nx.from_edgelist(squares)  # Room for 4!
        for max_num_emb in [None, 6]:  # None means unbounded
            embs = find_multiple_embeddings(S, T, max_num_emb=max_num_emb)
            self.assertLess(len(embs), 5, "Impossibly many")
        self.assertTrue(
            all(set(emb.keys()) == set(S.nodes()) for emb in embs),
            "bad keys in embedding(s)",
        )
        self.assertTrue(
            all(set(emb.values()).issubset(set(T.nodes())) for emb in embs),
            "bad values in embedding(s)",
        )
        value_list = [v for emb in embs for v in emb.values()]
        self.assertEqual(len(set(value_list)), len(value_list), "embeddings overlap")

    def test_find_multiple_embeddings_advanced(self):
        # timeout (via embedder_kwargs)
        m = 3  # Feasible, but takes significantly more than a second
        S = dnx.chimera_graph(2 * m)
        T = dnx.zephyr_graph(m)
        embedder_kwargs = {"timeout": 1}
        embs = find_multiple_embeddings(S, T, embedder_kwargs=embedder_kwargs)
        self.assertEqual(len(embs), 0)

        # max_num_embs
        square = {
            ((0, 0), (0, 1)),
            ((0, 1), (1, 1)),
            ((1, 1), (1, 0)),
            ((1, 0), (0, 0)),
        }
        L = 3
        squares = {
            tuple((v1 + i, v2 + j) for v1, v2 in e)
            for i in range(L)
            for j in range(L)
            for e in square
        }
        S = nx.from_edgelist(square)
        T = nx.from_edgelist(squares)

        embss = [find_multiple_embeddings(S, T, max_num_emb=mne) for mne in range(1, 4)]
        for mne in range(1, 4):
            i = mne - 1
            self.assertLessEqual(len(embss[i]), mne)
            if i > 0:
                self.assertLessEqual(len(embss[i - 1]), len(embss[i]))
                # Should proceed deterministically, sequentially:
                self.assertTrue(
                    embs[i - 1][idx] == emb[idx] for idx, emb in enumerate(embss[i])
                )

        seed = 42
        embs_run1 = find_multiple_embeddings(
            S, T, max_num_emb=4, seed=seed, shuffle_all_graphs=True
        )
        prng2 = np.random.default_rng(seed)
        embs_run2 = find_multiple_embeddings(
            S, T, max_num_emb=4, seed=prng2, shuffle_all_graphs=True
        )
        self.assertEqual(embs_run1, embs_run2, "seed does not allow reproducibility")

        seed = seed + 1
        embs_run3 = find_multiple_embeddings(
            S, T, max_num_emb=4, seed=seed, shuffle_all_graphs=True
        )
        self.assertNotEqual(
            embs_run1,
            embs_run3,
            "different seeds give same embedding, this should not occur "
            "with high probability",
        )

        # embedder, embedder_kwargs, one_to_iterable
        triangle = {(0, 1), (1, 2), (0, 2)}
        S = nx.from_edgelist(triangle)  # Cannot embed 1:1 on T, which is bipartite.
        embs = find_multiple_embeddings(
            S, T, embedder=find_embedding, one_to_iterable=True
        )
        self.assertEqual(len(embs), 1)
        emb = embs[0]
        node_list = [n for c in emb.values() for n in c]
        node_set = set(node_list)
        self.assertEqual(len(node_list), 4)
        self.assertEqual(len(node_set), 4)
        self.assertTrue(node_set.issubset(set(T.nodes())), "bad values in embedding(s)")

        embedder_kwargs = {"initial_chains": emb}  # A valid embedding
        embs = find_multiple_embeddings(
            S,
            T,
            embedder=find_embedding,
            embedder_kwargs=embedder_kwargs,
            one_to_iterable=True,
        )
        # NB - ordering within chains can change (seems to!)
        self.assertTrue(all(set(emb[i]) == set(embs[0][i]) for i in range(3)))

    def test_find_sublattice_embeddings_basic(self):
        # defaults and basic arguments
        for topology in ["chimera", "pegasus", "zephyr"]:
            if topology == "chimera":
                min_sublattice_size = 1
                S = dnx.chimera_graph(min_sublattice_size)
                T = dnx.chimera_graph(min_sublattice_size + 1)
                num_emb = 4
            elif topology == "pegasus":
                min_sublattice_size = 2
                S = dnx.pegasus_graph(min_sublattice_size)
                T = dnx.pegasus_graph(min_sublattice_size + 1)
                num_emb = 2
            elif topology == "zephyr":
                min_sublattice_size = 1
                S = dnx.zephyr_graph(min_sublattice_size)
                T = dnx.zephyr_graph(min_sublattice_size + 1)
                num_emb = 2

            embs = find_sublattice_embeddings(S, T, sublattice_size=min_sublattice_size)
            self.assertEqual(len(embs), 1, "mismatched number of embeddings")

            embs = find_sublattice_embeddings(
                S, T, sublattice_size=min_sublattice_size, max_num_emb=None
            )
            self.assertEqual(len(embs), num_emb, "mismatched number of embeddings")
            self.assertTrue(
                all(set(emb.keys()) == set(S.nodes()) for emb in embs),
                "bad keys in embedding(s)",
            )
            self.assertTrue(
                all(set(emb.values()).issubset(set(T.nodes())) for emb in embs),
                "bad values in embedding(s)",
            )
            value_list = [v for emb in embs for v in emb.values()]
            self.assertEqual(
                len(set(value_list)), len(value_list), "embeddings overlap"
            )

    def test_find_sublattice_embeddings_tile(self):
        # Check function responds correctly to tile specification
        min_sublattice_size = 1
        S = nx.from_edgelist({(i, i + 1) for i in range(5)})  # 6 nodes
        T = dnx.chimera_graph(min_sublattice_size + 1)
        tile = dnx.chimera_graph(min_sublattice_size, node_list=list(range(1, 8)))
        embs = find_sublattice_embeddings(
            S,
            T,
            sublattice_size=min_sublattice_size,
            max_num_emb=None,
            tile=tile,
        )
        self.assertEqual(len(embs), 4)
        nodes_used = {v for emb in embs for v in emb.values()}
        self.assertEqual(len(nodes_used), S.number_of_nodes() * len(embs))
        self.assertTrue(
            all(n % 8 > 0 for n in nodes_used), "Every 8th node excluded by tile"
        )
        tile5 = dnx.chimera_graph(min_sublattice_size, node_list=list(range(3, 8)))
        embs = find_sublattice_embeddings(
            S,
            T,
            sublattice_size=min_sublattice_size,
            max_num_emb=None,
            tile=tile5,
        )
        self.assertEqual(len(embs), 0, "Tile is too small")

        S = tile
        embs = find_sublattice_embeddings(
            S,
            T,
            sublattice_size=min_sublattice_size,
            max_num_emb=None,
            tile=tile,
            embedder=lambda x: "without S=tile trigger error",
        )
        self.assertEqual(len(embs), 4)
        nodes_used = {v for emb in embs for v in emb.values()}
        self.assertEqual(len(nodes_used), S.number_of_nodes() * len(embs))

        invalid_T = nx.complete_graph(5)  # Complete graph is not a valid topology
        with self.assertRaises(ValueError):
            find_sublattice_embeddings(
                S, invalid_T, sublattice_size=min_sublattice_size, tile=tile
            )

        small_T = dnx.chimera_graph(m=2, n=2)
        small_S = dnx.chimera_graph(m=2, n=1)
        sublattice_size = 1  # Too small
        with self.assertWarns(Warning):
            find_sublattice_embeddings(
                small_S, small_T, sublattice_size=sublattice_size, use_filter=True
            )
        tile = dnx.chimera_graph(m=1)  # Too small
        with self.assertWarns(Warning):
            find_sublattice_embeddings(small_S, small_T, tile=tile, use_filter=True)


if __name__ == "__main__":
    unittest.main()

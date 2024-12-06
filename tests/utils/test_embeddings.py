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
import os
import numpy as np

import networkx as nx
from itertools import product

import dwave_networkx as dnx

from minorminer.utils.embeddings import (
    visualize_embeddings,
    shuffle_graph,
    embeddings_to_array,
)

_display = os.environ.get("DISPLAY", "") != ""


class TestEmbeddings(unittest.TestCase):

    @unittest.skipUnless(_display, " No display found")
    def test_visualize_embeddings(self):
        embeddings = [{}]
        T = dnx.chimera_graph(2)
        visualize_embeddings(T, embeddings)
        blocks_of = [1, 8]
        one_to_iterable = [True, False]
        for b, o in product(blocks_of, one_to_iterable):
            if o:
                embeddings = [
                    {0: tuple(n + idx * b for n in range(b))}
                    for idx in range(T.number_of_nodes() // b)
                ]  # Blocks of 8
            else:
                embeddings = [
                    {n: n + idx * b for n in range(b)}
                    for idx in range(T.number_of_nodes() // b)
                ]  # Blocks of 8

            visualize_embeddings(T, embeddings, one_to_iterable=o)
            prng = np.random.default_rng()
            visualize_embeddings(T, embeddings, seed=prng, one_to_iterable=o)

        S = nx.Graph()
        S.add_nodes_from({i for i in T.nodes})
        emb = {n: n for n in T.nodes}
        visualize_embeddings(
            T, embeddings=[emb], S=S
        )  # Should plot every nodes but no edges
        S.add_edges_from(list(T.edges)[:2])
        visualize_embeddings(
            T, embeddings=[emb], S=S
        )  # Should plot every node, and two edges
        visualize_embeddings(
            T, embeddings=[emb], S=None
        )  # Should plot every nodes and edges

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


if __name__ == "__main__":
    unittest.main()

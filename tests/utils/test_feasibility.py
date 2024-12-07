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
    lattice_size_upper_bound,
    lattice_size_lower_bound,
)


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

    def test_lattice_size_subgraph_upper_bound(self):
        L = np.random.randint(2) + 2
        T = dnx.zephyr_graph(L - 1)
        self.assertEqual(L - 1, lattice_size_upper_bound(T=T))
        T = dnx.pegasus_graph(L)
        self.assertEqual(L, lattice_size_upper_bound(T=T))
        T = dnx.chimera_graph(L, L - 1, 1)
        self.assertEqual(L, lattice_size_upper_bound(T=T))

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


if __name__ == "__main__":
    unittest.main()

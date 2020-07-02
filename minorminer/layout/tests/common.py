# Copyright 2020 D-Wave Systems Inc.
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

import unittest

import dwave_networkx as dnx
import networkx as nx
import numpy as np

import minorminer.layout as mml


class TestLayoutPlacement(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestLayoutPlacement, self).__init__(*args, **kwargs)

        # Graphs for testing
        self.S = nx.random_regular_graph(3, 50)
        self.S_small = nx.random_regular_graph(3, 10)
        self.G = nx.Graph()
        self.H = nx.complete_graph(1)
        self.C = dnx.chimera_graph(4)
        self.C_coord = dnx.chimera_graph(4, coordinates=True)
        self.C_blank = dnx.chimera_graph(4, data=False)
        self.P = dnx.pegasus_graph(4)
        self.P_coord = dnx.pegasus_graph(4, coordinates=True)
        self.P_nice = dnx.pegasus_graph(4, nice_coordinates=True)
        self.P_blank = dnx.pegasus_graph(4, data=False)

        # Compute some layouts
        self.S_layout = mml.Layout(self.S)
        self.S_small_layout = mml.Layout(self.S_small)
        self.G_layout = mml.Layout(self.G)
        self.C_layout = mml.Layout(self.C)
        self.C_coord_layout = mml.Layout(self.C_coord)
        self.C_blank_layout = mml.Layout(self.C_blank)
        self.C_layout_3 = mml.Layout(self.C, dim=3)
        self.P_layout = mml.Layout(self.P)
        self.P_coord_layout = mml.Layout(self.P_coord)
        self.P_nice_layout = mml.Layout(self.P_nice)
        self.P_blank_layout = mml.Layout(self.P_blank)

    def assertArrayEqual(self, a, b):
        """
        Tests that two arrays are equal via numpy.
        """
        np.testing.assert_almost_equal(a, b)

    def assertLayoutEqual(self, G, layout_1, layout_2):
        """
        Tests that two layouts are equal by iterating through them.
        """
        for v in G:
            self.assertArrayEqual(layout_1[v], layout_2[v])

    def assertIsLayout(self, S, layout):
        """
        Tests that layout is a mapping from S to R^d
        """
        for u in S:
            self.assertEqual(len(layout[u]), layout.dim)

    def assertIsPlacement(self, S, T, placement):
        """
        Tests that placement is a mapping from S to 2^T
        """
        for u in S:
            self.assertTrue(set(placement[u]) <= set(T))

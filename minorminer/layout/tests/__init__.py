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
        self.G = nx.Graph()
        self.H = nx.complete_graph(1)
        self.C = dnx.chimera_graph(4)

        # Compute some layouts
        self.S_layout = mml.Layout(self.S)
        self.G_layout = mml.Layout(self.G)
        self.C_layout = mml.Layout(self.C)
        self.C_layout_3 = mml.Layout(self.C, d=3)

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
            self.assertEqual(len(layout[u]), layout.d)

    def assertIsPlacement(self, S, T, placement):
        """
        Tests that placement is a mapping from S to 2^T
        """
        for u in S:
            self.assertTrue(set(placement[u]) <= set(T))

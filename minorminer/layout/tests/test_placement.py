import random
import unittest

import dwave_networkx as dnx
import minorminer.layout as mml
import networkx as nx


class TestPlacement(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestPlacement, self).__init__(*args, **kwargs)

        # Graphs for testing
        self.S = nx.random_regular_graph(3, 50)
        self.G = nx.Graph()
        self.C = dnx.chimera_graph(4)

        # Layouts for testing
        self.S_layout = mml.Layout(self.S, nx.spectral_layout)
        self.G_layout = mml.Layout(self.G)
        self.C_layout = mml.Layout(self.C, dnx.chimera_layout)
        self.C_layout_3 = mml.Layout(self.C, dnx.chimera_layout, dim=3)

    def test_precomputed_placement(self):
        """
        Tests passing in a placement as a dictionary
        """
        placement = rando_placer(self.S_layout, self.C_layout)
        mml.Placement(self.S_layout, self.C_layout, placement)

    def test_placement_functions(self):
        """
        Functions can be passed in to Placement objects.
        """

        mml.Placement(self.S_layout, self.C_layout, rando_placer)

    def test_placement_class(self):
        """
        Test the placement mutable mapping behavior.
        """
        P = mml.Placement(self.G_layout, self.G_layout)

        # Test __setitem__
        P['a'] = 1

        # Test __iter__ and __getitem__
        for k, v in P.items():
            self.assertEqual(k, 'a')
            self.assertEqual(v, 1)

        # Test __len__
        self.assertEqual(len(P), 1)

        # Test __del__
        del P['a']

        # Test __repr__
        self.assertEqual(repr(P), "{}")

    def test_failures(self):
        """
        Test some failure conditions
        """
        # Dimension mismatch failure
        self.assertRaises(ValueError, mml.Placement,
                          self.S_layout, self.C_layout_3)

        # Layout input failure
        self.assertRaises(TypeError, mml.Placement,
                          nx.spectral_layout(self.S), self.C_layout)
        self.assertRaises(TypeError, mml.Placement, (), self.C_layout)


def rando_placer(S_layout, T_layout):
    T_vertices = list(T_layout)
    return {v: [random.choice(T_vertices)] for v in S_layout}


if __name__ == '__main__':
    unittest.main()

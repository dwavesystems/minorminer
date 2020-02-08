import random
import unittest
from itertools import product

import dwave_networkx as dnx
import networkx as nx

import minorminer.layout.layout as mml
from minorminer.layout.utils import layout_utils

# Set a seed to standardize the randomness.
seed = 2035


class TestLayout(unittest.TestCase):

    def test_kamada_kawai(self):
        """
        Tests that the Kamada-Kawai layout is correct for K_4,4 (dimensions 1, 2, 3, 10).
        """
        G = nx.complete_bipartite_graph(4, 4)
        dims = [1, 2, 3, 10]

        layouts = [
            mml.kamada_kawai(G, d=d, seed=seed).layout for d in dims
        ]

        for layout in layouts:
            for p in layout.values():
                for coordinate in p:
                    self.assertGreaterEqual(coordinate, -1)
                    self.assertLessEqual(coordinate,  1)

    def test_chimera(self):
        """
        Tests that the layout is correct for Chimera(4) (dimensions 2 and 3).
        """
        G = dnx.chimera_graph(4)
        dims = [2, 3, 10]

        layouts = [
            mml.dnx_layout(G, d=d).layout for d in dims
        ]

        for layout in layouts:
            for p in layout.values():
                for coordinate in p:
                    self.assertGreaterEqual(coordinate, -1)
                    self.assertLessEqual(coordinate,  1)

    def test_pca(self):
        """
        Tests that the PCA layout is correct for K_4,4 (dimensions 1, 2, 3, 8).
        """
        G = nx.complete_bipartite_graph(4, 4)
        dims = [1, 2, 3, 8]

        layouts = [
            mml.pca(G, d=d, seed=seed).layout for d in dims
        ]

        for layout in layouts:
            for p in layout.values():
                for coordinate in p:
                    self.assertGreaterEqual(coordinate, -1)
                    self.assertLessEqual(coordinate,  1)

    def test_center_scale(self):
        """
        Tests that all layouts correctly recenter and scale. Chooses a random center in [-10, 10]^2 and a random scale
        in [0, 10].
        """
        G = dnx.chimera_graph(4)
        center = (random.random() * random.randint(-10, 10),
                  random.random() * random.randint(-10, 10))
        scale = random.random() * random.randint(1, 10)

        # Test all layouts
        layouts = []
        layouts.append(mml.dnx_layout(G, center=center, scale=scale).layout)
        layouts.append(mml.kamada_kawai(G, center=center, scale=scale).layout)
        layouts.append(mml.pca(G, center=center, scale=scale).layout)

        for layout in layouts:
            for p in layout.values():
                for i, x in enumerate(p):
                    # My own implementations of self.assertAlmostGreater[Less]Equal
                    self.assertTrue(round(x, 7) >=
                                    round(center[i] - scale, 7))
                    self.assertTrue(round(x, 7) <=
                                    round(center[i] + scale, 7))

    def test_integer_lattice_layout(self):
        """
        Tests that a layout correctly bins to an integer lattice. Randomly chooses integers between [4, 100] as
        dimensions for a 2d layout; 4 is the lower bound since the size of dnx.*_graphs() dictate the integer lattice
        and G is dnx.chimera_graph(4) is a test case.
        """
        G = dnx.chimera_graph(4)
        H = dnx.chimera_graph(4, data=False)
        J = nx.random_regular_graph(3, 60)

        # Test all layouts
        empty_layout = mml.Layout(G)
        chimera_layout = mml.dnx_layout(H)
        kamada_kawai_layout = mml.kamada_kawai(J)

        layouts = [empty_layout, chimera_layout, kamada_kawai_layout]

        for _ in range(5):
            rand_x = random.randint(4, 100)
            rand_y = random.randint(4, 100)
            squares = [layout.integer_lattice_layout(
                rand_x) for layout in layouts]
            rectangles = [layout.integer_lattice_layout(
                (rand_x, rand_y)) for layout in layouts]

            for square in squares:
                for p in square.values():
                    self.assertIn(
                        p, list(product(range(rand_x), range(rand_x))))
            for rectangle in rectangles:
                for p in rectangle.values():
                    self.assertIn(
                        p, list(product(range(rand_x), range(rand_y))))


if __name__ == '__main__':
    unittest.main()

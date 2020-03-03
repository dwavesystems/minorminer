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

    def test_p_norm(self):
        """
        Tests that the Kamada-Kawai layout is correct for K_4,4 (dimensions 1, 2, 3, 10).
        """
        G = nx.complete_bipartite_graph(4, 4)
        dims = [1, 2, 3, 10]

        for d in dims:
            _ = mml.p_norm(G, d=d)

    def test_chimera(self):
        """
        Tests that the layout is correct for Chimera(4) (dimensions 2 and 3).
        """
        G = dnx.chimera_graph(4)
        dims = [2, 3, 10]

        for d in dims:
            _ = mml.dnx_layout(G, d=d)

    def test_pca(self):
        """
        Tests that the PCA layout is correct for K_4,4 (dimensions 1, 2, 3, 8).
        """
        G = nx.complete_bipartite_graph(4, 4)
        dims = [1, 2, 3, 8]

        for d in dims:
            _ = mml.pca(G, d=d, seed=seed)

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
        layouts.append(mml.p_norm(G, center=center, scale=scale).layout)
        layouts.append(mml.pca(G, center=center, scale=scale).layout)

        for layout in layouts:
            for p in layout.values():
                for i, x in enumerate(p):
                    # My own implementations of self.assertAlmostGreater[Less]Equal
                    self.assertTrue(round(x, 7) >=
                                    round(center[i] - scale, 7))
                    self.assertTrue(round(x, 7) <=
                                    round(center[i] + scale, 7))


if __name__ == '__main__':
    unittest.main()

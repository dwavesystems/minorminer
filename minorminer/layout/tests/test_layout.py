import random
import unittest

import dwave_networkx as dnx
import networkx as nx

import minorminer.layout.layout as mml
from .data import precomputed_chimera, precomputed_kamada_kawai

# Set a seed to standardize the randomness.
n = 9999
random.seed(n)
seed = random.randint(1, n)


class TestLayout(unittest.TestCase):

    def test_kamada_kawai(self):
        """
        Tests that the Kamada-Kawai layout is correct for K_4,4 (dimensions 1, 2, 3, 10); i.e. that it matches the 
        precomputed layout.
        """
        G = nx.complete_bipartite_graph(4, 4)

        layouts = {d: mml.kamada_kawai(
            G, d=d, seed=seed) for d in precomputed_kamada_kawai}

        for d, l in precomputed_kamada_kawai.items():
            for v, p in l.items():
                for i, x in enumerate(p):
                    coordinate = layouts[d][v][i]
                    self.assertAlmostEqual(coordinate, x, 3)
                    self.assertGreaterEqual(coordinate, -1)
                    self.assertLessEqual(coordinate, 1)

    def test_chimera(self):
        """
        Tests that the layout is correct for Chimera(4) (dimensions 2 and 3); i.e. that it matches the precomputed 
        layout.
        """
        G = dnx.chimera_graph(4)
        layouts = {d: mml.chimera(
            G, d=d) for d in precomputed_chimera}

        for d, l in precomputed_chimera.items():
            for v, p in l.items():
                for i, x in enumerate(p):
                    coordinate = layouts[d][v][i]
                    self.assertAlmostEqual(coordinate, x, 3)
                    self.assertGreaterEqual(coordinate, -1)
                    self.assertLessEqual(coordinate, 1)

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
        layouts.append(mml.chimera(G, center=center, scale=scale))
        layouts.append(mml.kamada_kawai(G, center=center, scale=scale))

        for layout in layouts:
            for v, p in layout.items():
                for i, x in enumerate(p):
                    coordinate = layout[v][i]
                    # My own implementations of self.assertAlmostGreater[Less]Equal
                    self.assertTrue(round(coordinate, 7) >=
                                    round(center[i] - scale, 7))
                    self.assertTrue(round(coordinate, 7) <=
                                    round(center[i] + scale, 7))


if __name__ == '__main__':
    unittest.main()

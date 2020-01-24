import random
import unittest

import dwave_networkx as dnx
import networkx as nx

import minorminer.layout.layout as mml
from data import precomputed_chimera, precomputed_kamada_kawai

# Set a seed to standardize the randomness.
n = 9999
random.seed(n)
seed = random.randint(1, n)


class TestLayout(unittest.TestCase):

    def test_kamada_kawai(self):
        """
        Tests that the Kamada-Kawai layout is correct for K_4,4 (dimensions 1, 2, 3, 10); i.e. that it matches the pre-
        computed values in [-1, 1]^d.
        """
        G = nx.complete_bipartite_graph(4, 4)

        layouts = {d: mml.kamada_kawai(
            G, d=d, seed=seed) for d in precomputed_kamada_kawai}

        for d, l in precomputed_kamada_kawai.items():
            for v, p in l.items():
                for i, x in enumerate(p):
                    coordinate = layouts[d][v][i]
                    self.assertAlmostEqual(coordinate, x)
                    self.assertGreaterEqual(coordinate, -1)
                    self.assertLessEqual(coordinate, 1)

    def test_chimera(self):
        """
        Tests that the layout is correct for Chimera(4) (dimensions 2 and 3); i.e. that it matches the pre-computed 
        values in [-1, 1]^d.
        """
        G = dnx.chimera_graph(4)
        layouts = {d: mml.chimera(
            G, d=d) for d in precomputed_chimera}

        for d, l in precomputed_chimera.items():
            for v, p in l.items():
                for i, x in enumerate(p):
                    coordinate = layouts[d][v][i]
                    self.assertAlmostEqual(coordinate, x)
                    self.assertGreaterEqual(coordinate, -1)
                    self.assertLessEqual(coordinate, 1)


if __name__ == '__main__':
    unittest.main()

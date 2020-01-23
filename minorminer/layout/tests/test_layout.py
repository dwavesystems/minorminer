import random
import unittest

import networkx as nx
from numpy import array

from minorminer.layout import layout

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

        my_kamada_kawai = {d: layout.kamada_kawai(
            G, d=d, seed=seed) for d in pre_computed_kamada_kawai}

        for d, l in pre_computed_kamada_kawai.items():
            for v, p in l.items():
                for i, x in enumerate(p):
                    coordinate = my_kamada_kawai[d][v][i]
                    self.assertAlmostEqual(coordinate, x)
                    self.assertGreaterEqual(coordinate, -1)
                    self.assertLessEqual(coordinate, 1)

    def test_kamada_kawai_1(self):
        """
        Tests that the Kamada-Kawai layout is correct for K_4,4 in 1d.
        """
        G = nx.complete_bipartite_graph(4, 4)

        my_kamada_kawai = layout.kamada_kawai(G, d=1)
        kamada_kawai = {
            0: array([-1.]),
            1: array([-0.74999997]),
            2: array([-0.49999993]),
            3: array([-0.2499999]),
            4: array([0.2499999]),
            5: array([0.49999993]),
            6: array([0.74999997]),
            7: array([1.])
        }

        for v, p in kamada_kawai.items():
            for i, x in enumerate(p):
                coordinate = my_kamada_kawai[v][i]
                self.assertAlmostEqual(coordinate, x)
                self.assertGreaterEqual(coordinate, -1)
                self.assertLessEqual(coordinate, 1)

    def test_kamada_kawai_2(self):
        """
        Tests that the Kamada-Kawai layout is correct for K_4,4 in 2d.
        """
        G = nx.complete_bipartite_graph(4, 4)

        my_kamada_kawai = layout.kamada_kawai(G)
        kamada_kawai = {
            0: array([1., -0.06320495]),
            1: array([0.53618338, 0.48755357]),
            2: array([-0.03438645,  0.72389135]),
            3: array([-0.75179941,  0.66241415]),
            4: array([-0.99999975,  0.06320475]),
            5: array([-0.53618328, -0.48755351]),
            6: array([0.03438642, -0.72389123]),
            7: array([0.75179908, -0.66241412])
        }

        for v, p in kamada_kawai.items():
            for i, x in enumerate(p):
                coordinate = my_kamada_kawai[v][i]
                self.assertAlmostEqual(coordinate, x)
                self.assertGreaterEqual(coordinate, -1)
                self.assertLessEqual(coordinate, 1)

    def test_kamada_kawai_3(self):
        """
        Tests that the Kamada-Kawai layout is correct for K_4,4 in 3d.
        """
        G = nx.complete_bipartite_graph(4, 4)

        my_kamada_kawai = layout.kamada_kawai(G, d=3, seed=seed)
        kamada_kawai = {
            0: array([-0.98709905, -0.1467455, -0.14392173]),
            1: array([0.64128769, -0.68976331, -0.36001532]),
            2: array([0.98708649, 0.14677012, 0.14393496]),
            3: array([-0.64126773,  0.68975957,  0.3600116]),
            4: array([-0.12808806,  0.01386342, -0.99999759]),
            5: array([-0.21772691, -0.85033526,  0.49607745]),
            6: array([0.2177274,  0.85035448, -0.49608937]),
            7: array([0.12808017, -0.01390352,  1.])
        }

        for v, p in kamada_kawai.items():
            for i, x in enumerate(p):
                coordinate = my_kamada_kawai[v][i]
                self.assertAlmostEqual(coordinate, x)
                self.assertGreaterEqual(coordinate, -1)
                self.assertLessEqual(coordinate, 1)

    def test_kamada_kawai_10(self):
        """
        Tests that the Kamada-Kawai layout is correct for K_4,4 in 10d.
        """
        G = nx.complete_bipartite_graph(4, 4)

        my_kamada_kawai = layout.kamada_kawai(G, d=10, seed=seed)
        kamada_kawai = {
            0: array([-1.,  0.14029899,  0.28687387,  0.18651374,  0.39668944,
                      -0.43120035,  0.15087843,  0.15002928,  0.70819066, -0.70950176]),
            1: array([0.74702881,  0.77938909,  0.09432316, -0.50056582, -0.24865196,
                      -0.25723954,  0.60890604, -0.69002551,  0.13703349,  0.33409856]),
            2: array([-0.22853146, -0.45075593, -0.7181411,  0.10456035,  0.60312017,
                      0.89164998, -0.11406198, -0.36983147, -0.62121529,  0.23176585]),
            3: array([0.48150304, -0.46892587,  0.33694644,  0.20949211, -0.75115823,
                      -0.20321651, -0.64572577,  0.90983074, -0.2240106,  0.1436377]),
            4: array([0.5653143, -0.13324933, -0.14883313,  0.96652688,  0.13539118,
                      0.35343568,  0.85628352,  0.38361186,  0.45911682, -0.04684403]),
            5: array([-0.25021981, -0.14126589, -0.79179585, -0.66333981, -0.9165724,
                      0.12704637, -0.19481089, -0.20852614,  0.21568323, -0.62970882]),
            6: array([-0.26440385,  0.78508537,  0.46821357,  0.64575101,  0.02804133,
                      -0.06709947, -0.84028861, -0.50792323, -0.43359484,  0.22454071]),
            7: array([-0.05069104, -0.51057644,  0.47241304, -0.94893846,  0.75314046,
                      -0.41337617,  0.17881927,  0.33283447, -0.24120348,  0.45201179])
        }

        for v, p in kamada_kawai.items():
            for i, x in enumerate(p):
                coordinate = my_kamada_kawai[v][i]
                self.assertAlmostEqual(coordinate, x)
                self.assertGreaterEqual(coordinate, -1)
                self.assertLessEqual(coordinate, 1)


pre_computed_kamada_kawai = {
    1: {
        0: array([-1.]),
        1: array([-0.74999997]),
        2: array([-0.49999993]),
        3: array([-0.2499999]),
        4: array([0.2499999]),
        5: array([0.49999993]),
        6: array([0.74999997]),
        7: array([1.])
    },
    2: {
        0: array([1., -0.06320495]),
        1: array([0.53618338, 0.48755357]),
        2: array([-0.03438645,  0.72389135]),
        3: array([-0.75179941,  0.66241415]),
        4: array([-0.99999975,  0.06320475]),
        5: array([-0.53618328, -0.48755351]),
        6: array([0.03438642, -0.72389123]),
        7: array([0.75179908, -0.66241412])
    },
    3: {
        0: array([-0.98709905, -0.1467455, -0.14392173]),
        1: array([0.64128769, -0.68976331, -0.36001532]),
        2: array([0.98708649, 0.14677012, 0.14393496]),
        3: array([-0.64126773,  0.68975957,  0.3600116]),
        4: array([-0.12808806,  0.01386342, -0.99999759]),
        5: array([-0.21772691, -0.85033526,  0.49607745]),
        6: array([0.2177274,  0.85035448, -0.49608937]),
        7: array([0.12808017, -0.01390352,  1.])
    },
    10: {
        0: array([-1.,  0.14029899,  0.28687387,  0.18651374,  0.39668944,
                  -0.43120035,  0.15087843,  0.15002928,  0.70819066, -0.70950176]),
        1: array([0.74702881,  0.77938909,  0.09432316, -0.50056582, -0.24865196,
                  -0.25723954,  0.60890604, -0.69002551,  0.13703349,  0.33409856]),
        2: array([-0.22853146, -0.45075593, -0.7181411,  0.10456035,  0.60312017,
                  0.89164998, -0.11406198, -0.36983147, -0.62121529,  0.23176585]),
        3: array([0.48150304, -0.46892587,  0.33694644,  0.20949211, -0.75115823,
                  -0.20321651, -0.64572577,  0.90983074, -0.2240106,  0.1436377]),
        4: array([0.5653143, -0.13324933, -0.14883313,  0.96652688,  0.13539118,
                  0.35343568,  0.85628352,  0.38361186,  0.45911682, -0.04684403]),
        5: array([-0.25021981, -0.14126589, -0.79179585, -0.66333981, -0.9165724,
                  0.12704637, -0.19481089, -0.20852614,  0.21568323, -0.62970882]),
        6: array([-0.26440385,  0.78508537,  0.46821357,  0.64575101,  0.02804133,
                  -0.06709947, -0.84028861, -0.50792323, -0.43359484,  0.22454071]),
        7: array([-0.05069104, -0.51057644,  0.47241304, -0.94893846,  0.75314046,
                  -0.41337617,  0.17881927,  0.33283447, -0.24120348,  0.45201179])
    }
}

if __name__ == '__main__':
    unittest.main()

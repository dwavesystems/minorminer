import random
import unittest
from itertools import product

import dwave_networkx as dnx
import minorminer.layout as mml
import networkx as nx
import numpy as np
from minorminer.layout.utils import layout_utils

# Set a seed to standardize the randomness.
seed = 2035


class TestLayout(unittest.TestCase):

    def assertArrayEqual(self, a, b):
        np.testing.assert_array_equal(a, b)

    def assertInBounds(self, a, center, scale):
        np.testing.assert_array_less(np.abs(a - center), scale+.01)

    def test_p_norm(self):
        """
        Tests that the p-norm (Kamada-Kawai-esque) layout is correct.
        """
        G = nx.random_regular_graph(3, 10)

        # Randomly choose dimensions, p-norm, center, and scale
        p_1 = 1
        p_rand = random.randint(2, 100)
        p_inf = float("inf")
        dim_spec = random.randint(1, len(G)-1)
        dim_rand = random.randint(len(G), 100)
        center_spec = tuple(random.random() * random.randint(-10, 10)
                            for _ in range(dim_spec))
        center_rand = tuple(random.random() * random.randint(-10, 10)
                            for _ in range(dim_rand))
        scale = random.random() * random.randint(1, 10)

        # Some starting layout stuff
        G_distances = dict(nx.all_pairs_shortest_path_length(G))
        spec_start = nx.spectral_layout(G, dim=dim_spec)
        rand_start = nx.random_layout(G, dim=dim_rand, seed=seed)

        # Compute layout with the spectral starting layout and p_rand
        layout_1 = mml.p_norm(G, p=p_rand, d=dim_spec,
                              center=center_spec, scale=scale)
        layout_2 = mml.p_norm(
            G, p=p_rand, starting_layout=spec_start, G_distances=G_distances, d=dim_spec, center=center_spec, scale=scale)

        self.assertInBounds(layout_1.layout_array, center_spec, scale)
        self.assertArrayEqual(layout_1.layout_array,
                              layout_2.layout_array)

        # Compute layout with the random starting layout and p_1
        layout_1 = mml.p_norm(G, p=p_1, d=dim_rand,
                              center=center_rand, scale=scale, seed=seed)
        layout_2 = mml.p_norm(
            G, p=p_1, starting_layout=rand_start, G_distances=G_distances, d=dim_rand, center=center_rand, scale=scale)

        self.assertInBounds(layout_1.layout_array, center_rand, scale)
        self.assertArrayEqual(layout_1.layout_array,
                              layout_2.layout_array)

        # Compute the layout with p_inf
        layout = mml.p_norm(G, p=p_inf)
        self.assertInBounds(layout.layout_array, layout.center, layout.scale)

    def test_chimera(self):
        """
        Tests that the layout is correct for Chimera.
        """
        n = 4
        G = dnx.chimera_graph(n)

        # Randomly choose a dimension, center, and scale
        dim = random.randint(2, len(G))
        center = tuple(random.random() * random.randint(-10, 10)
                       for _ in range(dim))

        scale = random.random() * random.randint(1, 10)

        # Compute the layout specifying a scale
        layout = mml.dnx_layout(G, d=dim, center=center, scale=scale)
        self.assertInBounds(layout.layout_array, center, scale)

        # Compute the layout using the default scale
        layout = mml.dnx_layout(G)
        self.assertInBounds(layout.layout_array, (0, 0), n/2)

        # Compute the layout using the dnx default scale
        layout = mml.dnx_layout(G, rescale=False)
        self.assertInBounds(layout.layout_array, (1/2, -1/2), 1/2)

    def test_pegasus(self):
        """
        Tests that the layout is correct for Pegasus.
        """
        n = 4
        G = dnx.pegasus_graph(n)

        # Randomly choose a dimension, center, and scale
        dim = random.randint(2, len(G))
        center = tuple(random.random() * random.randint(-10, 10)
                       for _ in range(dim))
        scale = random.random() * random.randint(1, 10)

        # Compute the layout specifying a scale
        layout = mml.dnx_layout(G, d=dim, center=center, scale=scale)
        self.assertInBounds(layout.layout_array, center, scale)

        # Compute the layout using the default scale
        layout = mml.dnx_layout(G)
        self.assertInBounds(layout.layout_array, (0, 0), n/2)

        # Compute the layout using the dnx default scale
        layout = mml.dnx_layout(G, rescale=False)
        self.assertInBounds(layout.layout_array, (1/2, -1/2), 1/2)

    def test_pca(self):
        """
        Tests that the PCA layout is correct.
        """
        G = nx.random_regular_graph(3, 10)

        # Randomly choose final dimensions, intermediate dimensions, and scale
        dim = random.randint(1, len(G))
        dim_fail = random.randint(len(G)+1, 100)
        m = random.randint(dim, len(G))
        m_fail = random.randint(len(G)+1, 100)
        pca_center = tuple(random.random() * random.randint(-10, 10)
                       for _ in range(dim))
        non_pca_center = tuple(random.random() * random.randint(-10, 10)
                       for _ in range(m))
        scale = random.random() * random.randint(1, 10)

        # Compute the layout with PCA
        layout = mml.pca(
            G, d=dim, m=m, center=pca_center, scale=scale, seed=seed)
        self.assertInBounds(layout.layout_array, pca_center, scale)

        # Compute the layout without PCA
        layout = mml.pca(
            G, d=dim, m=m, pca=False, center=non_pca_center, scale=scale, seed=seed)
        self.assertInBounds(layout.layout_array, non_pca_center, scale)

        # Testing failure conditions
        self.assertRaises(AssertionError, mml.pca, G, dim_fail)
        self.assertRaises(AssertionError, mml.pca, G, dim, m_fail)


if __name__ == '__main__':
    unittest.main()

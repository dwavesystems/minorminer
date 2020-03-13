import random
import unittest
from itertools import product

import dwave_networkx as dnx
import minorminer.layout as mml
import networkx as nx
import numpy as np
from minorminer.layout.layout import center_layout, invert_layout, scale_layout
from minorminer.layout.utils import layout_utils


class TestLayout(unittest.TestCase):
    # FIXME: This should go in a base class, but if I do that, the testing suite won't recognize the tests
    def __init__(self, *args, **kwargs):
        super(TestLayout, self).__init__(*args, **kwargs)

        self.n = 4

        self.S = nx.random_regular_graph(3, 10)
        self.C = dnx.chimera_graph(self.n)
        self.P = dnx.pegasus_graph(self.n)

        self.seed = 2035

    def assertArrayEqual(self, a, b):
        np.testing.assert_array_equal(a, b)

    def assertInBounds(self, a, center, scale):
        np.testing.assert_array_less(np.abs(a - center), scale+.01)

    def test_p_norm(self):
        """
        Tests that the p-norm (Kamada-Kawai-esque) layout is correct.
        """
        # Randomly choose dimensions, p-norm, center, and scale
        p_1 = 1
        p_rand = random.randint(2, 50)
        p_inf = float("inf")
        dim_spec = random.randint(1, len(self.S)-1)
        dim_rand = random.randint(len(self.S), 50)
        center_spec = tuple(random.random() * random.randint(-10, 10)
                            for _ in range(dim_spec))
        center_rand = tuple(random.random() * random.randint(-10, 10)
                            for _ in range(dim_rand))
        scale = random.random() * random.randint(1, 10)

        # Some starting layout stuff
        G_distances = dict(nx.all_pairs_shortest_path_length(self.S))
        spec_start = nx.spectral_layout(self.S, dim=dim_spec)
        rand_start = nx.random_layout(self.S, dim=dim_rand, seed=self.seed)

        # Compute layout with the spectral starting layout and p_rand
        layout_1 = mml.p_norm(self.S, p=p_rand, d=dim_spec,
                              center=center_spec, scale=scale)
        layout_2 = mml.p_norm(
            self.S, p=p_rand, starting_layout=spec_start, G_distances=G_distances, d=dim_spec, center=center_spec, scale=scale)

        self.assertInBounds(layout_1.layout_array, center_spec, scale)
        self.assertArrayEqual(layout_1.layout_array,
                              layout_2.layout_array)

        # Compute layout with the random starting layout and p_1
        layout_1 = mml.p_norm(self.S, p=p_1, d=dim_rand,
                              center=center_rand, scale=scale, seed=self.seed)
        layout_2 = mml.p_norm(
            self.S, p=p_1, starting_layout=rand_start, G_distances=G_distances, d=dim_rand, center=center_rand, scale=scale)

        self.assertInBounds(layout_1.layout_array, center_rand, scale)
        self.assertArrayEqual(layout_1.layout_array,
                              layout_2.layout_array)

        # Compute the layout with p_inf
        layout = mml.p_norm(self.S, p=p_inf)
        self.assertInBounds(layout.layout_array, layout.center, layout.scale)

    def test_dnx_layout(self):
        """
        Tests that the dnx_layout is correct for Chimera, Pegasus, and throws an error for other graphs.
        """
        # Randomly choose a dimension, center, and scale
        dim = random.randint(2, len(self.C))
        center = tuple(random.random() * random.randint(-10, 10)
                       for _ in range(dim))

        scale = random.random() * random.randint(1, 10)

        # Compute the layout specifying a scale
        layout_C = mml.dnx_layout(self.C, d=dim, center=center, scale=scale)
        self.assertInBounds(layout_C.layout_array, center, scale)

        layout_P = mml.dnx_layout(self.P, d=dim, center=center, scale=scale)
        self.assertInBounds(layout_P.layout_array, center, scale)

        # Compute the layout using the default scale
        layout_C = mml.dnx_layout(self.C)
        self.assertInBounds(layout_C.layout_array, (0, 0), self.n/2)

        layout_P = mml.dnx_layout(self.P)
        self.assertInBounds(layout_P.layout_array, (0, 0), self.n/2)

        # Compute the layout using the dnx default scale
        layout_C = mml.dnx_layout(self.C, rescale=False)
        self.assertInBounds(layout_C.layout_array, (1/2, -1/2), 1/2)

        layout_P = mml.dnx_layout(self.P, rescale=False)
        self.assertInBounds(layout_P.layout_array, (1/2, -1/2), 1/2)

        # Testing failure conditions
        self.assertRaises(ValueError, mml.dnx_layout, self.S)

    def test_pca(self):
        """
        Tests that the PCA layout is correct.
        """
        # Randomly choose final dimensions, intermediate dimensions, and scale
        dim = random.randint(1, len(self.S))
        dim_fail = random.randint(len(self.S)+1, 100)
        m = random.randint(dim, len(self.S))
        m_fail = random.randint(len(self.S)+1, 100)
        pca_center = tuple(random.random() * random.randint(-10, 10)
                           for _ in range(dim))
        non_pca_center = tuple(random.random() * random.randint(-10, 10)
                               for _ in range(m))
        scale = random.random() * random.randint(1, 10)

        # Compute the layout with PCA
        layout = mml.pca(
            self.S, d=dim, m=m, center=pca_center, scale=scale, seed=self.seed)
        self.assertInBounds(layout.layout_array, pca_center, scale)

        # Compute the layout without PCA
        layout = mml.pca(
            self.S, d=dim, m=m, pca=False, center=non_pca_center, scale=scale, seed=self.seed)
        self.assertInBounds(layout.layout_array, non_pca_center, scale)

        # Testing failure conditions
        self.assertRaises(AssertionError, mml.pca, self.S, dim_fail)
        self.assertRaises(AssertionError, mml.pca, self.S, dim, m_fail)

    def test_precomputed_layout(self):
        """
        Pass in a precomputed layout to the Layout class.
        """
        # Pick an arbitrary layout to precompute
        layout = nx.spectral_layout(self.S)
        layout_array = np.array([layout[v] for v in self.S])

        # Initialize each layout object
        layout_1 = mml.Layout(self.S, layout)
        layout_2 = mml.Layout(self.S, layout_array)

        self.assertArrayEqual(layout_1.layout_array,
                              layout_2.layout_array)

    def test_transformations(self):
        """
        Test transformation functions on layouts--mostly here so that this class provides full coverage for all of layouts.
        """
        # Randomly choose new scale and center
        new_scale = random.random() * random.randint(1, 10)
        new_center = tuple(random.random() * random.randint(-10, 10)
                           for _ in range(2))

        # Pick an arbitrary layout to precompute
        layout = mml.p_norm(self.S)

        # Scale the layout without passing in the prior center or scale
        scaled_layout = scale_layout(layout.layout_array, new_scale)
        self.assertInBounds(scaled_layout, layout.center, new_scale)

        # Center the layout without passing in the prior center
        centered_layout = center_layout(layout.layout_array, new_center)
        self.assertInBounds(centered_layout, new_center, layout.scale)

        # Invert layout without passing in the prior center
        inverted_layout = invert_layout(layout.layout_array)
        self.assertInBounds(inverted_layout, layout.center, layout.scale)

        # Invert the object (for full coverage)
        layout.invert_layout()
        self.assertInBounds(layout.layout_array, layout.center, layout.scale)

    def test_silly_graphs(self):
        """
        Make sure things don't break for trivial graphs.
        """
        G = nx.complete_graph(0)
        H = nx.complete_graph(1)

        self.assertRaises(ValueError, mml.p_norm, G)
        mml.p_norm(H)
        self.assertRaises(AssertionError, mml.pca, G)
        self.assertRaises(AssertionError, mml.pca, H)

    def test_layout_as_dictionary(self):
        """
        Test the layout dictionary behavior.
        """
        L = mml.Layout(self.S)
        L['a'] = 1
        del L['a']
        self.assertEqual(repr(L), "{}")


if __name__ == '__main__':
    unittest.main()

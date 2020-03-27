import random
import unittest
from itertools import product

import dwave_networkx as dnx
import networkx as nx
import numpy as np

import minorminer.layout as mml
from minorminer.layout.layout import (_center_layout, _dimension_layout,
                                      _scale_layout)


class TestLayout(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestLayout, self).__init__(*args, **kwargs)

        self.S = nx.random_regular_graph(3, 10)
        self.G = nx.Graph()
        self.H = nx.complete_graph(1)

    def assertArrayEqual(self, a, b):
        np.testing.assert_almost_equal(a, b)

    def assertLayoutEqual(self, G, layout_1, layout_2):
        for v in G:
            self.assertArrayEqual(layout_1[v], layout_2[v])

    def test_precomputed_layout(self):
        """
        Pass in a precomputed layout to the Layout class.
        """
        # Pick an arbitrary layout to precompute
        layout = nx.random_layout(self.S)

        # Initialize the layout object
        layout_obj = mml.Layout(self.S, layout)

        self.assertLayoutEqual(self.S, layout, layout_obj)

    def test_dimension(self):
        """
        Change the dimension of a layout.
        """
        d = random.randint(3, 10)

        # Pass in d as an argument
        layout_pre = mml.Layout(self.S, d=d)
        self.assertEqual(layout_pre.d, d)

        # Change the layout to have the d
        layout_post = mml.Layout(self.S)
        layout_post.d = d
        self.assertEqual(layout_post.d, d)

        # Change the dimension without changing the object,
        layout = mml.Layout(self.S, d=2)
        new_layout_array = _dimension_layout(layout.layout_array, d)
        self.assertEqual(layout.d, 2)

        # The layouts should match each other
        self.assertLayoutEqual(self.S, layout_pre, layout_post)
        self.assertArrayEqual(new_layout_array, layout_pre.layout_array)

        # Test dimension too small
        self.assertRaises(ValueError, mml.Layout, self.S, d=1)

    def test_center(self):
        """
        Recenter a layout.
        """
        center = (random.randint(-10, 10), random.randint(-10, 10))

        # Pass in center as an argument
        layout_pre = mml.Layout(self.S, center=center)
        self.assertArrayEqual(layout_pre.center, center)

        # Change the layout to have the center
        layout_post = mml.Layout(self.S)
        layout_post.center = center
        self.assertArrayEqual(layout_post.center, center)

        # Change the center without changing the object,
        layout = mml.Layout(self.S, center=(0, 0))
        new_layout_array = _center_layout(layout.layout_array, center)
        self.assertArrayEqual(layout.center, (0, 0))

        # The layouts should match each other
        self.assertLayoutEqual(self.S, layout_pre, layout_post)
        self.assertArrayEqual(new_layout_array, layout_pre.layout_array)

    def test_scale(self):
        """
        Rescale a layout.
        """
        scale = random.random()*random.randint(1, 10)

        # Pass in scale as an argument
        layout_pre = mml.Layout(self.S, scale=scale)
        self.assertEqual(layout_pre.scale, scale)

        # Change the layout to have the scale
        layout_post = mml.Layout(self.S)
        layout_post.scale = scale
        self.assertEqual(layout_post.scale, scale)

        # Change the scale without changing the object,
        layout = mml.Layout(self.S, scale=1)
        new_layout_array = _scale_layout(layout.layout_array, scale)
        self.assertArrayEqual(layout.scale, 1)

        # The layouts should match each other
        self.assertLayoutEqual(self.S, layout_pre, layout_post)
        self.assertArrayEqual(new_layout_array, layout_pre.layout_array)

    def test_layout_functions(self):
        """
        Functions can be passed in to Layout objects.
        """
        mml.Layout(self.S, nx.circular_layout)
        mml.Layout(self.S, nx.random_layout)

    def test_edge_input(self):
        """
        Layouts can be computed with edges instead of graph objects.
        """
        mml.Layout(self.S.edges)

    def test_silly_graphs(self):
        """
        Make sure things don't break for trivial graphs.
        """
        mml.Layout(self.G)
        mml.Layout(self.H)

    def test_layout_class(self):
        """
        Test the layout mutable mapping behavior.
        """
        L = mml.Layout(nx.Graph())

        # Test __setitem__
        L['a'] = 1

        # Test __iter__ and __getitem__
        for k, v in L.items():
            self.assertEqual(k, 'a')
            self.assertEqual(v, 1)

        # Test __len__
        self.assertEqual(len(L), 1)

        # Test __del__
        del L['a']

        # Test __repr__
        self.assertEqual(repr(L), "{}")


if __name__ == '__main__':
    unittest.main()

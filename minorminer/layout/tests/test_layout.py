import random
import unittest
from itertools import product

import dwave_networkx as dnx
import networkx as nx
import numpy as np

import minorminer.layout as mml
from minorminer.layout.layout import (_center_layout, _dimension_layout,
                                      _scale_layout)
from .common import TestLayoutPlacement


class TestLayout(TestLayoutPlacement):
    def test_pnorm(self):
        """
        Test the p_norm layout strategy.
        """
        # Some specs to test with
        low_dim = random.randint(3, 9)
        high_dim = len(self.S_small)
        center = (1, 1)
        scale = random.random()*random.randint(1, 10)

        # Default behavior
        mml.p_norm(self.S_small)

        # Using a starting_layout
        mml.p_norm(self.S_small, starting_layout=nx.random_layout(self.S_small))

        # Passing in G_distances
        mml.p_norm(
            self.S_small, G_distances=nx.all_pairs_shortest_path_length(self.S_small))

        # Passing in dim
        mml.p_norm(self.S_small, dim=low_dim)
        mml.p_norm(self.S_small, dim=high_dim)

        # Passing in center
        mml.p_norm(self.S_small, center=center)

        # Passing in scale
        mml.p_norm(self.S_small, scale=scale)

        # Different p-norms
        mml.p_norm(self.S_small, p=1)
        mml.p_norm(self.S_small, p=3)
        mml.p_norm(self.S_small, p=float("inf"))

        # # Playing with a starting_layout and a dimension
        # starting_layout = nx.spectral_layout(self.S, dim=dim)
        # layout_in = mml.Layout(
        #     self.S, mml.p_norm, starting_layout=starting_layout)
        # layout_out = mml.Layout(
        #     self.S, mml.p_norm, starting_layout=nx.spectral_layout, dim=dim)
        # self.assertLayoutEqual(self.S, layout_in, layout_out)
        # self.assertIsLayout(self.S, layout_in)

    def test_precomputed_layout(self):
        """
        Pass in a precomputed layout to the Layout class.
        """
        # Pick an arbitrary layout to precompute
        layout = nx.random_layout(self.S)

        # Initialize the layout object
        layout_obj = mml.Layout(self.S, layout)

        self.assertLayoutEqual(self.S, layout, layout_obj)
        self.assertIsLayout(self.S, layout_obj)

    def test_dimension(self):
        """
        Change the dimension of a layout.
        """
        dim = random.randint(3, 10)

        # Pass in dim as an argument
        layout_pre = mml.Layout(self.S, dim=dim)
        self.assertEqual(layout_pre.dim, dim)
        self.assertIsLayout(self.S, layout_pre)

        # Change the layout to have the dim
        layout_post = mml.Layout(self.S)
        layout_post.dim = dim
        self.assertEqual(layout_post.dim, dim)
        self.assertIsLayout(self.S, layout_post)

        # Change the dimension without changing the object,
        layout = mml.Layout(self.S, dim=2)
        new_layout_array = _dimension_layout(layout.layout_array, dim)
        self.assertEqual(layout.dim, 2)

        # The layout_arrays changed after the fact should match each other
        self.assertArrayEqual(new_layout_array, layout_post.layout_array)

        # Test dimension too small
        layout = mml.Layout(self.S, dim=2)
        self.assertRaises(ValueError, _dimension_layout,
                          layout.layout_array, 1)

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
        self.assertIsLayout(self.S, layout_pre)
        self.assertArrayEqual(new_layout_array, layout_pre.layout_array)

    def test_scale(self):
        """
        Rescale a layout.
        """
        scale = random.random()*random.randint(1, 10)

        # Pass in scale as an argument
        layout_pre = mml.Layout(self.S, scale=scale)
        self.assertAlmostEqual(layout_pre.scale, scale)

        # Change the layout to have the scale
        layout_post = mml.Layout(self.S)
        layout_post.scale = scale
        self.assertAlmostEqual(layout_post.scale, scale)

        # Change the scale without changing the object,
        layout = mml.Layout(self.S, scale=1)
        new_layout_array = _scale_layout(layout.layout_array, scale)
        self.assertAlmostEqual(layout.scale, 1)

        # The layouts should match each other
        self.assertLayoutEqual(self.S, layout_pre, layout_post)
        self.assertIsLayout(self.S, layout_pre)
        self.assertArrayEqual(new_layout_array, layout_pre.layout_array)

    def test_layout_functions(self):
        """
        Functions can be passed in to Layout objects.
        """
        # Circular
        layout = mml.Layout(self.S, nx.circular_layout)
        self.assertIsLayout(self.S, layout)

        # Random
        layout = mml.Layout(self.S, nx.random_layout)
        self.assertIsLayout(self.S, layout)

    def test_edge_input(self):
        """
        Layouts can be computed with edges instead of graph objects.
        """
        layout = mml.Layout(self.S.edges)
        self.assertIsLayout(self.S, layout)

    def test_silly_graphs(self):
        """
        Make sure things don't break for trivial graphs.
        """
        # Empty graph
        layout = mml.Layout(self.G)
        self.assertIsLayout(self.G, layout)

        # Single vertex graph
        layout = mml.Layout(self.H)
        self.assertIsLayout(self.H, layout)

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

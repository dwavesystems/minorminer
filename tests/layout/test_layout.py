# Copyright 2020 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import random
import unittest
from itertools import product

import dwave_networkx as dnx
import networkx as nx
import numpy as np

import minorminer.layout as mml
from minorminer.layout.layout import (_center_layout, _dimension_layout,
                                      _scale_layout, _graph_distance_matrix, dnx_layout)
from .common import TestLayoutPlacement


class TestLayout(TestLayoutPlacement):
    def test_pnorm(self):
        """
        Test the p_norm layout algorithm.
        """
        # Some specs to test with
        low_dim = random.randint(3, 9)
        high_dim = len(self.S_small)
        center = low_dim*(1, )
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

        # Test through the Layout object
        layout = mml.Layout(self.S_small, mml.p_norm, dim=low_dim,
                            center=center, scale=scale)

        self.assertArrayEqual(layout.center, center)
        self.assertAlmostEqual(layout.scale, scale)

        # Ensure that the rectangle packer works
        # TODO fix p_norm's issues with disconnected graphs
        layout = mml.Layout(self.S_components, mml.p_norm, dim=2,
                            center=[12, 20], scale=scale)

        self.assertArrayEqual(layout.center, [12, 20])
        self.assertAlmostEqual(layout.scale, scale)

    def test_dnx(self):
        """
        Test the dnx layout.
        """
        # Some specs to test with
        dim = random.randint(3, 9)
        center = dim*(1, )
        scale = random.random()*random.randint(1, 10)

        # Default behavior
        # Chimera
        mml.dnx_layout(self.C)
        # Pegasus
        mml.dnx_layout(self.P)
        # Zephyr
        mml.dnx_layout(self.Z)

        # Passing in dim
        mml.dnx_layout(self.C, dim=dim)

        # Passing in center
        mml.dnx_layout(self.C, center=center)

        # Passing in scale
        mml.dnx_layout(self.C, scale=scale)

        # Test through the Layout object
        layout = mml.Layout(self.C, mml.dnx_layout, dim=dim,
                            center=center, scale=scale)
        self.assertArrayEqual(layout.center, center)
        self.assertAlmostEqual(layout.scale, scale)

        # Test non-dnx_graph
        self.assertRaises(ValueError, mml.dnx_layout, self.S)

        # Test dim and center mismatch
        self.assertRaises(ValueError, mml.dnx_layout,
                          self.C, dim=3, center=(0, 0))

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
        layout_pre = mml.Layout(self.S, scale=scale, layout = self.S_layout)
        self.assertAlmostEqual(layout_pre.scale, scale)

        # Change the layout to have the scale
        layout_post = mml.Layout(self.S, layout = self.S_layout)
        layout_post.scale = scale
        self.assertAlmostEqual(layout_post.scale, scale)

        # Change the scale without changing the object,
        layout = mml.Layout(self.S, scale=1, layout = self.S_layout)
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

        # Random -- nx.random_layout doesn't accept a "scale" parameter, so we
        # need to disable component packing
        layout = mml.Layout(self.S, nx.random_layout, pack_components = False)
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

    def test_graph_distance_matrix(self):
        G = nx.Graph()
        G.add_nodes_from([2, 1, 3, 4])
        G.add_edges_from([(2, 1), (1, 3), (4, 2)])
        dist_mat = _graph_distance_matrix(G)
        self.assertTrue(np.array_equal(dist_mat, dist_mat.T), "The graph distance matrix is not symmetric")
        
    def test_dnx_layout(self):
        G = dnx.zephyr_graph(2)
        scale=10
        G_dnx_layout = dnx_layout(G, scale=scale)
        G_dnx_layout_arr = np.array([G_dnx_layout[v] for v in G.nodes()])
        self.assertTrue(np.all((G_dnx_layout_arr >= -scale) & (G_dnx_layout_arr <= scale)),
                        msg=f"Values are not within [{-scale}, {scale}]^2")
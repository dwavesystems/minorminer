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

import dwave_networkx as dnx
import minorminer.layout as mml
import networkx as nx
from minorminer.layout.placement import (_lookup_intersection_coordinates,
                                         _parse_layout,
                                         _minimize_overlap)

from .common import TestLayoutPlacement


class TestPlacement(TestLayoutPlacement):
    def test_intersection(self):
        """
        Tests the intersection placement strategy.
        """
        # Default behavior
        S_layout = mml.Layout(self.S)
        placement = mml.intersection(S_layout, self.C_layout)
        self.assertIsPlacement(self.S, self.C, placement)

        # Test different scale ratios
        placement_1 = mml.intersection(
            S_layout, self.C_layout, scale_ratio=.5)
        placement_2 = mml.intersection(
            S_layout, self.C_layout, scale_ratio=1)
        placement_3 = mml.intersection(
            S_layout, self.C_layout, scale_ratio=1.5)
        self.assertIsPlacement(self.S, self.C, placement_1)
        self.assertIsPlacement(self.S, self.C, placement_2)
        self.assertIsPlacement(self.S, self.C, placement_3)

        # Test a coordinate variations of chimera
        placement = mml.intersection(S_layout, self.C_coord_layout)
        self.assertIsPlacement(self.S, self.C_coord, placement)

        placement = mml.intersection(S_layout, self.C_blank_layout)
        self.assertIsPlacement(self.S, self.C_blank, placement)

        # Test coordinate variations of pegasus
        placement = mml.intersection(S_layout, self.P_coord_layout)
        self.assertIsPlacement(self.S, self.P_coord, placement)

        placement = mml.intersection(S_layout, self.P_layout)
        self.assertIsPlacement(self.S, self.P, placement)

        placement = mml.intersection(S_layout, self.P_nice_layout)
        self.assertIsPlacement(self.S, self.P_nice, placement)

        placement = mml.intersection(S_layout, self.P_nice_layout)
        self.assertIsPlacement(self.S, self.P_nice, placement)

        placement = mml.intersection(S_layout, self.P_blank_layout)
        self.assertIsPlacement(self.S, self.P_blank, placement)

        # Test coordinate variations of zephyr
        placement = mml.intersection(S_layout, self.Z_coord_layout)
        self.assertIsPlacement(self.S, self.Z_coord, placement)

        placement = mml.intersection(S_layout, self.Z_layout)
        self.assertIsPlacement(self.S, self.Z, placement)

        placement = mml.intersection(S_layout, self.Z_blank_layout)
        self.assertIsPlacement(self.S, self.Z_blank, placement)

    def test_closest(self):
        """
        Tests the closest placement strategy.
        """
        # Default behavior
        placement = mml.closest(self.S_layout, self.C_layout)
        self.assertIsPlacement(self.S, self.C, placement)

        # Different subset sizes
        placement = mml.closest(
            self.S_layout, self.C_layout, subset_size=(1, 2))
        self.assertIsPlacement(self.S, self.C, placement)

        placement = mml.closest(
            self.S_layout, self.C_layout, subset_size=(2, 2))
        self.assertIsPlacement(self.S, self.C, placement)

        # Different number of neighbors to query
        placement = mml.closest(self.S_layout, self.C_layout, num_neighbors=5)
        self.assertIsPlacement(self.S, self.C, placement)

        # All parameters
        placement = mml.closest(
            self.S_layout, self.C_layout, subset_size=(2, 3), num_neighbors=10)
        self.assertIsPlacement(self.S, self.C, placement)

    def test_precomputed_placement(self):
        """
        Tests passing in a placement as a dictionary
        """
        rando = rando_placer(self.S_layout, self.C_layout)
        placement = mml.Placement(self.S_layout, self.C_layout, rando)
        self.assertIsPlacement(self.S, self.C, placement)

    def test_placement_functions(self):
        """
        Functions can be passed in to Placement objects.
        """
        placement = mml.Placement(self.S_layout, self.C_layout, rando_placer)
        self.assertIsPlacement(self.S, self.C, placement)

    def test_scale_ratio(self):
        """
        Make sure filling works correctly.
        """
        # Make C_layout bigger than S_layout
        S_layout = mml.Layout(self.S, scale=1)
        C_layout = mml.Layout(self.C, scale=10)

        # Have S_layout scale to various ratios
        placement = mml.Placement(S_layout, C_layout, scale_ratio=1)
        self.assertAlmostEqual(placement.S_layout.scale, 10)

        placement = mml.Placement(S_layout, C_layout, scale_ratio=.5)
        self.assertAlmostEqual(placement.S_layout.scale, 5)

    def test_placement_class(self):
        """
        Test the placement mutable mapping behavior.
        """
        P = mml.Placement(self.G_layout, self.G_layout)

        # Test __setitem__
        P['a'] = 1

        # Test __iter__ and __getitem__
        for k, v in P.items():
            self.assertEqual(k, 'a')
            self.assertEqual(v, 1)

        # Test __len__
        self.assertEqual(len(P), 1)

        # Test __del__
        del P['a']

        # Test __repr__
        self.assertEqual(repr(P), "{}")

    def test_failures(self):
        """
        Test some failure conditions
        """
        # Dimension mismatch failure
        self.assertRaises(ValueError, mml.Placement,
                          self.S_layout, self.C_layout_3)

        # Layout input failure
        self.assertRaises(TypeError, mml.Placement,
                          "not a layout", self.C_layout)
        self.assertRaises(TypeError, _parse_layout, "not a layout")       


def rando_placer(S_layout, T_layout):
    T_vertices = list(T_layout)
    return {v: [random.choice(T_vertices)] for v in S_layout}

def test_minimize_overlap():
    dim_a = 10
    dim_b = 55
    grid_edges = list(nx.grid_graph([dim_a, dim_b]).edges())
    diagonal_edges = [((i, j), (i+1, j+1)) for i in range(dim_b-1) for j in range(dim_a-1)] + [((i, j), (i-1, j+1)) for i in range(1, dim_b) for j in range(dim_a-1)]
    king = nx.Graph(grid_edges+diagonal_edges)
    T = dnx.zephyr_graph(15)
    mml.find_embedding(king, T, scale=1)
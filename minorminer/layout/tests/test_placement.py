import random
import unittest

import dwave_networkx as dnx
import networkx as nx

import minorminer.layout as mml
from minorminer.layout.placement import _parse_layout
from . import TestLayoutPlacement


class TestPlacement(TestLayoutPlacement):
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

    def test_fill_T(self):
        """
        Make sure filling works correctly.
        """
        # Make C_layout bigger than S_layout
        S_layout = mml.Layout(self.S, scale=1)
        C_layout = mml.Layout(self.C, scale=2)

        # Have S_layout fill_T
        placement = mml.Placement(S_layout, C_layout, fill_T=True)

        # Check that the scale changed
        self.assertAlmostEqual(placement.S_layout.scale, 2)

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


if __name__ == '__main__':
    unittest.main()

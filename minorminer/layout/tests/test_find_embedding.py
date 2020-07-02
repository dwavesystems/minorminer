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
import networkx as nx
import numpy as np

import minorminer.layout as mml

from .common import TestLayoutPlacement


class TestFindEmb(TestLayoutPlacement):
    def test_default(self):
        """
        Minimal find_embedding call
        """
        # Test a dnx_graph
        mml.find_embedding(self.S, self.C)

        # Test a non-dnx_graph
        mml.find_embedding(self.S_small, self.S)

    def test_timeout(self):
        """
        Test the timeout parameter
        """
        # Subtract time from layout and placement and give it to mm.find_embedding
        mml.find_embedding(self.S, self.C, timeout=10)

        # Test layout and placement taking longer than timeout
        self.assertRaises(TimeoutError, mml.find_embedding,
                          self.S, self.C, timeout=0)

    def test_mm_hints(self):
        """
        Different types of mm.find_embedding hinting.
        """
        mml.find_embedding(self.S, self.C, mm_hint_type="initial_chains")
        mml.find_embedding(self.S, self.C, mm_hint_type="suspend_chains")

        self.assertRaises(ValueError, mml.find_embedding,
                          self.S, self.C, mm_hint_type="dance_party")

    def test_layout_returning(self):
        """
        Layouts can be returned.
        """
        _, (S_layout, C_layout) = mml.find_embedding(
            self.S, self.C, return_layouts=True)
        self.assertIsLayout(self.S, S_layout)
        self.assertIsLayout(self.C, C_layout)

    def test_layout_kwargs(self):
        """
        Pass in layout kwargs.
        """
        # Pick some values to pass in
        dim = 3
        center = (0, 0, 0)
        scale = 2

        _, (S_layout, C_layout) = mml.find_embedding(self.S, self.C,
                                                     dim=dim, center=center, scale=scale, return_layouts=True)
        # Test that S_layout matches
        self.assertEqual(S_layout.dim, dim)
        self.assertArrayEqual(S_layout.center, center)
        self.assertAlmostEqual(S_layout.scale, scale)
        # Test that C_layout matches
        self.assertEqual(C_layout.dim, dim)
        self.assertArrayEqual(C_layout.center, center)
        self.assertAlmostEqual(C_layout.scale, scale)

    def test_placement_kwargs(self):
        """
        Pass in placement kwargs.
        """
        # Pick some values to pass in
        scale_ratio = .8

        mml.find_embedding(self.S, self.C, scale_ratio=scale_ratio)

    def test_placement_closest(self):
        """
        Test the closest placement strategy
        """
        # Pick some values to pass in
        subset_size = (1, 2)
        num_neighbors = 5

        mml.find_embedding(self.S, self.C, placement=mml.closest,
                           subset_size=subset_size, num_neighbors=num_neighbors)

    def test_placement_interesction(self):
        """
        Test the intersection placement strategy
        """
        mml.find_embedding(self.S, self.C, placement=mml.intersection)

    def test_layout_parameter(self):
        """
        It can be a function or a 2-tuple of various things.
        """
        # A function
        mml.find_embedding(self.S, self.C, layout=nx.circular_layout)

        # 2-tuples
        # Two functions
        mml.find_embedding(self.S, self.C, layout=(
            nx.circular_layout, dnx.chimera_layout))
        # Two dictionaries
        mml.find_embedding(self.S, self.C, layout=(
            nx.circular_layout(self.S), dnx.chimera_layout(self.C)))
        # Two layouts
        mml.find_embedding(self.S, self.C, layout=(
            self.S_layout, self.C_layout))
        # A function and a layout
        mml.find_embedding(self.S, self.C, layout=(
            self.S_layout, dnx.chimera_layout(self.C)))

        # Failures
        # Too many things in layout
        self.assertRaises(ValueError, mml.find_embedding,
                          self.S, self.C, layout=(1, 2, 3))

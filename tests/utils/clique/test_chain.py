# Copyright 2026 D-Wave
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


import unittest

from minorminer.utils._clique.chain import Segment, chain_nodes


class TestSegment(unittest.TestCase):
    def test_vertical_nodes(self):
        segment = Segment(vertical=True, fixed_coord=4, k=2, start=1, end=9)
        self.assertEqual(segment.nodes(), [(4, 1, 2), (4, 5, 2), (4, 9, 2)])

    def test_horizontal_nodes(self):
        segment = Segment(vertical=False, fixed_coord=6, k=0, start=3, end=7)
        self.assertEqual(segment.nodes(), [(3, 6, 0), (7, 6, 0)])

    def test_single_node(self):
        segment = Segment(vertical=True, fixed_coord=0, k=1, start=5, end=5)
        self.assertEqual(segment.nodes(), [(0, 5, 1)])


class TestChainNodes(unittest.TestCase):
    def test_concatenates_segments_in_order(self):
        chain = (
            Segment(vertical=True, fixed_coord=2, k=0, start=1, end=5),
            Segment(vertical=False, fixed_coord=6, k=1, start=3, end=3),
        )
        self.assertEqual(chain_nodes(chain), [(2, 1, 0), (2, 5, 0), (3, 6, 1)])

    def test_empty_chain(self):
        self.assertEqual(chain_nodes(()), [])

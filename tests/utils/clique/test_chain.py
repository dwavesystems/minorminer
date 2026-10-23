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
        self.assertEqual(Segment(True, 4, 2, 1, 9).nodes(), [(4, 1, 2), (4, 5, 2), (4, 9, 2)])

    def test_horizontal_nodes(self):
        self.assertEqual(Segment(False, 6, 0, 3, 7).nodes(), [(3, 6, 0), (7, 6, 0)])

    def test_single_node(self):
        self.assertEqual(Segment(True, 0, 1, 5, 5).nodes(), [(0, 5, 1)])

    def test_chain_nodes_concatenates_segments_in_order(self):
        chain = (Segment(True, 2, 0, 1, 5), Segment(False, 6, 1, 3, 3))
        self.assertEqual(chain_nodes(chain), [(2, 1, 0), (2, 5, 0), (3, 6, 1)])

    def test_chain_nodes_of_empty_chain(self):
        self.assertEqual(chain_nodes(()), [])

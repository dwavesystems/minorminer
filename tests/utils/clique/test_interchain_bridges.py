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

from parameterized import parameterized

from minorminer.utils._clique.grid import Grid
from minorminer.utils._clique.interchain_bridges import chains_are_adjacent, find_interchain_bridges
from .faulty_zephyr import faulty_zephyr

# faulty Z(3, 2): two missing nodes and one missing internal coupler
_M, _T = 3, 2
_MISSING_NODES = [(5, 2, 0), (6, 5, 1)]
_MISSING_EDGES = [((8, 3, 0), (9, 4, 1))]

# bridges from chain A to _CHAIN_B
_CHAIN_B = [(10, 7, 1), (9, 6, 0)]
_DIRECT_CHAIN_A = [(3, 6, 0), (5, 6, 0)]  # (5, 6, 0) -- (9, 6, 0) is an external coupler
_CHAIN_A = [(3, 6, 0)]
_LONG_CHAIN_A = [(1, 6, 0), (3, 6, 0)]  # (1, 6, 0) is adjacent to (5, 6, 0) but no bridge node

# _CHAIN_A, nothing used
_ONE_NODE_BRIDGES = [((5, 6, 0),), ((7, 6, 0),)]
_TWO_NODE_BRIDGES = [
    ((1, 6, 0), (5, 6, 0)),
    ((4, 5, 0), (5, 6, 0)),
    ((4, 5, 1), (5, 6, 0)),
    ((4, 7, 0), (5, 6, 0)),
    ((4, 7, 1), (5, 6, 0)),
    ((5, 6, 0), (7, 6, 0)),
    ((7, 6, 0), (5, 6, 0)),
    ((7, 6, 0), (8, 5, 0)),
    ((7, 6, 0), (8, 5, 1)),
    ((7, 6, 0), (8, 7, 0)),
    ((7, 6, 0), (8, 7, 1)),
    ((7, 6, 0), (11, 6, 0)),
]
# _CHAIN_A, (5, 6, 0) used
_USED_TWO_NODE_BRIDGES = [
    ((7, 6, 0), (8, 5, 0)),
    ((7, 6, 0), (8, 5, 1)),
    ((7, 6, 0), (8, 7, 0)),
    ((7, 6, 0), (8, 7, 1)),
    ((7, 6, 0), (11, 6, 0)),
]
# _LONG_CHAIN_A, nothing used
_LONG_CHAIN_TWO_NODE_BRIDGES = [
    ((4, 5, 0), (5, 6, 0)),
    ((4, 5, 1), (5, 6, 0)),
    ((4, 7, 0), (5, 6, 0)),
    ((4, 7, 1), (5, 6, 0)),
    ((5, 6, 0), (7, 6, 0)),
    ((7, 6, 0), (5, 6, 0)),
    ((7, 6, 0), (8, 5, 0)),
    ((7, 6, 0), (8, 5, 1)),
    ((7, 6, 0), (8, 7, 0)),
    ((7, 6, 0), (8, 7, 1)),
    ((7, 6, 0), (11, 6, 0)),
]
_USED = {(5, 6, 0)}
_ALL_BRIDGES_USED = {(5, 6, 0), (7, 6, 0)}  # every route from A to B goes through these

# (name, chain_a, used_nodes, max_interchain_bridge_nodes, expected)
_BRIDGE_CASES = [
    ("directly_connected_cap_0", _DIRECT_CHAIN_A, set(), 0, [()]),
    ("directly_connected_cap_1", _DIRECT_CHAIN_A, set(), 1, [()]),
    ("directly_connected_cap_2", _DIRECT_CHAIN_A, set(), 2, [()]),
    ("nothing_used_cap_0", _CHAIN_A, set(), 0, []),
    ("nothing_used_cap_1", _CHAIN_A, set(), 1, _ONE_NODE_BRIDGES),
    ("nothing_used_cap_2", _CHAIN_A, set(), 2, _ONE_NODE_BRIDGES + _TWO_NODE_BRIDGES),
    ("nothing_used_cap_3", _CHAIN_A, set(), 3, _ONE_NODE_BRIDGES + _TWO_NODE_BRIDGES),
    ("used_node_cap_0", _CHAIN_A, _USED, 0, []),
    ("used_node_cap_1", _CHAIN_A, _USED, 1, [((7, 6, 0),)]),
    ("used_node_cap_2", _CHAIN_A, _USED, 2, [((7, 6, 0),)] + _USED_TWO_NODE_BRIDGES),
    ("chain_nodes_cap_0", _LONG_CHAIN_A, set(), 0, []),
    ("chain_nodes_cap_1", _LONG_CHAIN_A, set(), 1, _ONE_NODE_BRIDGES),
    (
        "chain_nodes_cap_2",
        _LONG_CHAIN_A,
        set(),
        2,
        _ONE_NODE_BRIDGES + _LONG_CHAIN_TWO_NODE_BRIDGES,
    ),
    ("all_bridges_used_cap_0", _CHAIN_A, _ALL_BRIDGES_USED, 0, []),
    ("all_bridges_used_cap_1", _CHAIN_A, _ALL_BRIDGES_USED, 1, []),
    ("all_bridges_used_cap_2", _CHAIN_A, _ALL_BRIDGES_USED, 2, []),
]


class TestFindInterchainBridges(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grid = Grid(
            faulty_zephyr(_M, _T, drop_nodes=_MISSING_NODES, drop_edges=_MISSING_EDGES)
        )

    @parameterized.expand(_BRIDGE_CASES)
    def test_bridges(self, _name, chain_a, used_nodes, cap, expected):
        result = find_interchain_bridges(self.grid, chain_a, _CHAIN_B, used_nodes, cap)
        self.assertCountEqual(result, expected)
        lengths = [len(bridge) for bridge in result]
        self.assertEqual(lengths, sorted(lengths))  # fewest bridge nodes first

    def test_default_cap(self):
        result = find_interchain_bridges(self.grid, _CHAIN_A, _CHAIN_B, set())
        self.assertCountEqual(result, _ONE_NODE_BRIDGES + _TWO_NODE_BRIDGES)


class TestChainsAreAdjacent(unittest.TestCase):
    v = (0, 1, 0)
    h = (1, 0, 1)  # internal coupler (dx, dy) = (1, -1)

    def test_adjacent_on_perfect_grid(self):
        grid = Grid(faulty_zephyr(1, 2))
        self.assertTrue(chains_are_adjacent(grid, [self.v], [self.h]))
        self.assertTrue(chains_are_adjacent(grid, [self.h], [self.v]))

    def test_missing_coupler(self):
        grid = Grid(faulty_zephyr(1, 2, drop_edges=[(self.v, self.h)]))
        self.assertFalse(chains_are_adjacent(grid, [self.v], [self.h]))
        # another node of the first chain is still joined to h by a coupler
        self.assertTrue(chains_are_adjacent(grid, [self.v, (2, 1, 0)], [self.h]))

    def test_absent_nodes_are_ignored(self):
        grid = Grid(faulty_zephyr(1, 2, drop_nodes=[self.h]))
        self.assertFalse(chains_are_adjacent(grid, [self.v], [self.h]))
        self.assertFalse(chains_are_adjacent(grid, [], [self.v]))
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


import functools
import itertools
import unittest

from parameterized import parameterized

from minorminer.utils._clique.chain import chain_nodes
from minorminer.utils._clique.grid import Grid
from minorminer.utils._clique.skeleton import skeleton_chains
from minorminer.utils._clique.windows import CORNERS, candidate_windows, window_bound

from .faulty_zephyr import faulty_zephyr


@functools.cache
def _perfect_grid(m, t):
    """Return a fault-free Zephyr grid, built once per ``(m, t)`` and shared across test cases."""
    return Grid(faulty_zephyr(m, t))


def _ideally_adjacent(nodes_a, nodes_b) -> bool:
    """Test whether the IDEAL grid has an internal coupler between the two node sets.

    Internal couplers join every vertical (x, y, k) to every horizontal (x +- 1, y +- 1, k').
    Unlike a coupler lookup, this ignores faults, so it tests the skeleton's geometry.
    """
    return any(abs(p[0] - q[0]) == 1 and abs(p[1] - q[1]) == 1 for p in nodes_a for q in nodes_b)


class TestSkeletonChains(unittest.TestCase):
    def assert_well_formed(self, grid, window, chains):
        lcoord = grid.cartesian_to_linear
        x0, x1, y0, y1 = window
        self.assertLessEqual(len(chains), window_bound(window, grid.t))
        seen = set()
        for v_segment, h_segment in chains:
            self.assertTrue(v_segment.vertical)
            self.assertFalse(h_segment.vertical)
            nodes = chain_nodes((v_segment, h_segment))
            for x, y, k in nodes:
                self.assertTrue(grid.is_present((x, y, k)))
                self.assertTrue(x0 - 1 <= x <= x1 + 1 and y0 - 1 <= y <= y1 + 1)
            self.assertFalse(seen & set(nodes))
            seen |= set(nodes)
            for segment in (v_segment, h_segment):
                path = segment.nodes()
                for a, b in zip(path, path[1:]):
                    self.assertTrue(grid.has_edge(lcoord(a), lcoord(b)))
            self.assertTrue(
                any(
                    grid.has_edge(lcoord(v), lcoord(h))
                    for v in v_segment.nodes()
                    for h in h_segment.nodes()
                )
            )
        for a, b in itertools.combinations(map(chain_nodes, chains), 2):
            self.assertTrue(_ideally_adjacent(a, b))

    @parameterized.expand(
        (m, t, side, corner)
        for m, t in ((2, 4), (3, 2), (4, 2), (3, 4))
        for side in range(2, 4 * m - 4 + 1, 2)
        for corner in CORNERS
    )
    def test_perfect_grid_attains_bound(self, m, t, side, corner):
        # away from the chip boundary, a window of even width w holds 2 t n chains, n = w / 2 + 1
        grid = _perfect_grid(m, t)
        window = (2, 2 + side, 2, 2 + side)
        chains = skeleton_chains(grid, window, corner)
        self.assert_well_formed(grid, window, chains)
        self.assertEqual(len(chains), window_bound(window, t))

    def test_faulty_grids(self):
        # every window and corner, uniform and clustered faults
        for seed in range(4):
            G = faulty_zephyr(
                3, 2, node_fault_rate=0.07, edge_fault_rate=0.01, seed=seed, clustered=seed % 2 == 1
            )
            grid = Grid(G)
            for window in candidate_windows(3, 2, 1, 1, seed=0):
                for corner in CORNERS:
                    with self.subTest(seed=seed, window=window, corner=corner):
                        self.assert_well_formed(grid, window, skeleton_chains(grid, window, corner))

    def test_faulty_grids_t4(self):
        for seed in range(2):
            G = faulty_zephyr(
                3, 4, node_fault_rate=0.07, edge_fault_rate=0.01, seed=seed, clustered=seed % 2 == 1
            )
            grid = Grid(G)
            for window in ((0, 12, 0, 12), (2, 10, 2, 10), (4, 8, 0, 4)):
                for corner in CORNERS:
                    with self.subTest(seed=seed, window=window, corner=corner):
                        chains = skeleton_chains(grid, window, corner)
                        self.assertTrue(chains)
                        self.assert_well_formed(grid, window, chains)

    def test_deterministic(self):
        """Check that repeated calls return the same chains.

        This guards the contract if the LIS step gains randomized tie-breaking.
        """
        grid = Grid(
            faulty_zephyr(3, 4, node_fault_rate=0.05, edge_fault_rate=0.01, seed=4, clustered=True)
        )
        window, corner = (2, 10, 0, 8), ("b", "r")
        self.assertEqual(
            skeleton_chains(grid, window, corner), skeleton_chains(grid, window, corner)
        )

    def test_no_reachable_pairs(self):
        G = faulty_zephyr(1, 2)
        G.remove_edges_from(list(G.edges()))
        self.assertEqual(skeleton_chains(Grid(G), (0, 4, 0, 4), ("t", "l")), [])

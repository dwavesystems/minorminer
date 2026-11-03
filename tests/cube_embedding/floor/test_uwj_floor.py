# Copyright 2025 D-Wave Systems Inc.
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
#
# ================================================================================================


from unittest import TestCase

from dwave_networkx import zephyr_graph
from minorminer.utils.zephyr.node_edge import NodeKind, ZNode
from minorminer.utils.zephyr.plane_shift import ZPlaneShift

from minorminer.cube_embedding._floor.uwj_floor import TileCoordIdx, UWJFloor
from minorminer.cube_embedding._tile.ladder import all_ladder_idx_chains, ladder_tiles
from minorminer.cube_embedding._tile.square import square_tiles
from minorminer._lattice_utils import UWJ, UWKJZ, QuoFloor, ZLatticeSurvey


class TestUWJFloor(TestCase):
    def setUp(self):
        shape = m, t = 6, 4
        nodes_to_remove = [(3, 2, 0), (10, 1, 1), (10, 21, 2), (14, 19, 3)]
        nodes_to_remove = [tuple(ZNode(xyk, shape=shape).zcoord) for xyk in nodes_to_remove]
        odd_edges_to_remove = [((4, 7, 1), (4, 9, 1)), ((7, 2, 3), (9, 2, 3))]
        odd_edges_to_remove = [
            (tuple(ZNode(x, shape=shape).zcoord), tuple(ZNode(y, shape=shape).zcoord))
            for (x, y) in odd_edges_to_remove
        ]
        ext_edges_to_remove = [
            ((10, 9, 1), (10, 13, 1)),
            ((14, 3, 3), (14, 7, 3)),
            ((3, 16, 2), (7, 16, 2)),
        ]
        ext_edges_to_remove = [
            (tuple(ZNode(x, shape=shape).zcoord), tuple(ZNode(y, shape=shape).zcoord))
            for (x, y) in ext_edges_to_remove
        ]
        int_edges_within_tile = [((4, 3, 3), (3, 4, 1)), ((6, 7, 0), (7, 8, 0))]
        int_edges_within_tile = [
            (tuple(ZNode(x, shape=shape).zcoord), tuple(ZNode(y, shape=shape).zcoord))
            for (x, y) in int_edges_within_tile
        ]
        int_edges_btwn_tile = [((3, 6, 0), (4, 5, 3)), ((6, 5, 1), (7, 4, 1))]
        int_edges_btwn_tile = [
            (tuple(ZNode(x, shape=shape).zcoord), tuple(ZNode(y, shape=shape).zcoord))
            for (x, y) in int_edges_btwn_tile
        ]

        G = zephyr_graph(m=m, t=t, coordinates=True)
        G.remove_nodes_from(nodes_to_remove)
        G.remove_edges_from(
            odd_edges_to_remove + ext_edges_to_remove + int_edges_within_tile + int_edges_btwn_tile
        )
        self.lsurvey = ZLatticeSurvey(G)
        self.tile = ladder_tiles["main"]
        self.qfloor = QuoFloor(tile=self.tile, dim=(6, 6))
        self.G = G
        self.missed_int_within_tile = [
            (UWKJZ(u=0, w=2, k=3, j=1, z=0), UWKJZ(u=1, w=2, k=1, j=1, z=0)),
            (UWKJZ(u=0, w=3, k=0, j=1, z=1), UWKJZ(u=1, w=4, k=0, j=1, z=1)),
        ]

    def test_from_qfloor_graph_runs_survey(self):
        for sub_kind in ["main", "anti"]:
            for dim in [(1, 1), (4, 2), (3, 6)]:
                qfloor = QuoFloor(tile=ladder_tiles[sub_kind], dim=dim)
                UWJFloor.from_qfloor_graph(qfloor=qfloor, lattice_survey=self.lsurvey)
                qfloor = QuoFloor(tile=square_tiles[sub_kind], dim=dim)
                UWJFloor.from_qfloor_graph(qfloor=qfloor, lattice_survey=self.lsurvey)

    def test_from_qfloor_graph_runs_graph(self):
        for sub_kind in ["main", "anti"]:
            for dim in [(1, 1), (4, 3), (3, 6)]:
                qfloor = QuoFloor(tile=ladder_tiles[sub_kind], dim=dim)
                UWJFloor.from_qfloor_graph(qfloor=qfloor, G=self.G)
                qfloor = QuoFloor(tile=square_tiles[sub_kind], dim=dim)
                UWJFloor.from_qfloor_graph(qfloor=qfloor, G=self.G)

    def test_floor(self):
        floor = UWJFloor.from_qfloor_graph(qfloor=self.qfloor, lattice_survey=self.lsurvey).floor
        quo_ext_path_num_per_k = {
            UWJ(u=1, w=1, j=1): 3,
            UWJ(u=0, w=5, j=0): 2,
            UWJ(u=0, w=7, j=1): 3,
            UWJ(u=1, w=8, j=1): 3,
        }
        for uwj, uwj_dict in floor.items():
            if uwj not in quo_ext_path_num_per_k:
                self.assertEqual(len(uwj_dict["perfect_ks"]), 4)
            else:
                self.assertEqual(len(uwj_dict["perfect_ks"]), quo_ext_path_num_per_k[uwj])

    def test_indices(self):
        uwj_floor_corner_indices = UWJFloor.from_qfloor_graph(
            qfloor=self.qfloor, lattice_survey=self.lsurvey
        ).indices
        qfloor_33 = QuoFloor(tile=self.tile + ZPlaneShift(3, 3), dim=(4, 4))
        uwj_floor_indices_33 = UWJFloor.from_qfloor_graph(
            qfloor=qfloor_33, lattice_survey=self.lsurvey
        ).indices
        for dir, dict_dir in uwj_floor_corner_indices.items():
            opposite_dir = NodeKind.HORIZONTAL if dir == NodeKind.VERTICAL else NodeKind.VERTICAL
            self.assertEqual(
                {idx: node + ZPlaneShift(3, 3) for idx, node in dict_dir.items()},
                uwj_floor_indices_33[opposite_dir]
                )

        qfloor_66 = QuoFloor(tile=self.tile + ZPlaneShift(6, 6), dim=(1, 1))
        uwj_floor_indices_66 = UWJFloor.from_qfloor_graph(
            qfloor=qfloor_66, lattice_survey=self.lsurvey
        ).indices
        for dir, dict_dir in uwj_floor_corner_indices.items():
            self.assertEqual(
                {idx: node + ZPlaneShift(6, 6) for idx, node in dict_dir.items()},
                uwj_floor_indices_66[dir],
            )

    def test_get_col_row_nodes(self):
        uwj_floor = UWJFloor.from_qfloor_graph(qfloor=self.qfloor, lattice_survey=self.lsurvey)
        nodes_corner_shifted_12 = [UWKJZ(*(a + ZPlaneShift(4, 8)).zcoord) for a in self.tile]
        nodes_12 = uwj_floor.get_col_row_nodes(Lx=2, Ly=3)[(1, 2)]
        for idx_list in nodes_12.values():
            for uwkjz in idx_list:
                self.assertTrue(
                    UWKJZ(u=uwkjz.u, w=uwkjz.w, j=uwkjz.j, z=uwkjz.z) in nodes_corner_shifted_12
                )

    def test_get_tile_coord_idx_none(self):
        uwj_floor = UWJFloor.from_qfloor_graph(qfloor=self.qfloor, lattice_survey=self.lsurvey)
        for uwkjz in [UWKJZ(0, 5, 2, 0, 5), UWKJZ(0, 5, 1, 0, 2)]:
            self.assertIsNone(uwj_floor.get_tile_coord_idx(uwkjz=uwkjz))

    def test_get_tile_coord_idx(self):
        uwj_floor = UWJFloor.from_qfloor_graph(qfloor=self.qfloor, lattice_survey=self.lsurvey)
        uwkjz1 = UWKJZ(u=0, w=11, k=0, j=1, z=3)
        uwkjz_loc_idx1 = uwj_floor.get_tile_coord_idx(uwkjz=uwkjz1)
        expected1 = TileCoordIdx(tile_coord=(5, 3), idx=4)
        self.assertEqual(uwkjz_loc_idx1, expected1)

        uwkjz2 = UWKJZ(u=1, w=10, k=0, j=1, z=0)
        uwkjz_loc_idx2 = uwj_floor.get_tile_coord_idx(uwkjz=uwkjz2)
        expected2 = TileCoordIdx(tile_coord=(0, 4), idx=6)
        self.assertEqual(uwkjz_loc_idx2, expected2)

    def test_get_uwj(self):
        uwj_floor = UWJFloor.from_qfloor_graph(qfloor=self.qfloor, lattice_survey=self.lsurvey)
        idx1, tile_coord1 = 6, (0, 3)
        expected_uwj1 = UWJ(u=1, w=8, j=1)
        self.assertEqual(uwj_floor.get_uwj(idx=idx1, tile_coord=tile_coord1), expected_uwj1)

        idx2, tile_coord2 = 7, (2, 5)
        expected_uwj2 = UWJ(u=0, w=6, j=1)
        self.assertEqual(uwj_floor.get_uwj(idx=idx2, tile_coord=tile_coord2), expected_uwj2)

    def test_missing_internal_edges(self):
        uwj_floor = UWJFloor.from_qfloor_graph(qfloor=self.qfloor, lattice_survey=self.lsurvey)
        idx_chains = all_ladder_idx_chains["main"]
        missed_within_tiles = uwj_floor.missing_internal_edges(lattice_survey=self.lsurvey)
        missed_within_tiles = {
            xy: xy_tile_coord_edge
            for xy, xy_tile_coord_edge in missed_within_tiles.items()
            if xy_tile_coord_edge.edge in idx_chains
        }

        for x, y in missed_within_tiles:
            self.assertTrue(
                (x, y) in self.missed_int_within_tile or (y, x) in self.missed_int_within_tile
            )
        self.assertEqual(len(missed_within_tiles), len(self.missed_int_within_tile))

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



from collections import Counter
from itertools import islice, product
from unittest import TestCase

from dwave_networkx import zephyr_graph
from minorminer.utils.zephyr.node_edge import ZNode

from minorminer.cube_embedding._floor.lz_provider import provider_floor_zpaths
from minorminer.cube_embedding._floor.trim import trim
from minorminer.cube_embedding._tile import TileKind, ZCoupling
from minorminer.cube_embedding._tile.ladder import all_ladder_idx_chains, ladder_tiles
from minorminer.cube_embedding._tile.square import all_square_idx_chains, square_tiles
from minorminer._lattice_utils import UWKJ, ZLatticeSurvey



class TestTrim(TestCase):
    def setUp(self) -> None:
        shape = m, t = 6, 4
        int_edges_within_tile = [
            ((4, 3, 3), (3, 4, 1)),
            ((6, 7, 0), (7, 8, 0)),
            ((4, 7, 3), (3, 8, 1)),
            ((11, 4, 1), (12, 3, 0)),
        ]
        int_edges_within_lad_main_tile = [
            (tuple(ZNode(x, shape=shape).zcoord), tuple(ZNode(y, shape=shape).zcoord))
            for (x, y) in int_edges_within_tile
        ]

        G = zephyr_graph(m=m, t=t, coordinates=True)
        G.remove_edges_from(int_edges_within_lad_main_tile)
        self.lsurvey = ZLatticeSurvey(G)
        self.nums = [0, 3, 5]

        self.lad_idx_to_zedge = all_ladder_idx_chains
        self.sq_idx_to_zedge = all_square_idx_chains
        self.sq_indices = {
            sub_kind: {i: node.node_kind for i, node in enumerate(tile.zns)}
            for sub_kind, tile in square_tiles.items()
        }
        self.lad_indices = {
            sub_kind: {i: node.node_kind for i, node in enumerate(tile.zns)}
            for sub_kind, tile in ladder_tiles.items()
        }

    def test_runs(self):
        params = [
            (TileKind.LADDER, "main", True, True, ZCoupling.ZERO_ONE_ONE_ZERO),
            (TileKind.LADDER, "anti", False, True, ZCoupling.ZERO_ONE),
            (TileKind.SQUARE, "main", True, True, ZCoupling.EITHER),
            (TileKind.LADDER, "anti", False, False, ZCoupling.ZERO_ONE),
        ]
        dimensions = [(1, 1, 1), (6, 6, 16)]
        for Lx, Ly, Lz in dimensions:
            for tile_kind, sub_kind, prescribed, periodic, coupling in params:
                provider_pf = provider_floor_zpaths(
                    tile_kind=tile_kind,
                    sub_kind=sub_kind,
                    Lx=Lx,
                    Ly=Ly,
                    Lz=Lz,
                    lattice_survey=self.lsurvey,
                    prescribed=prescribed,
                    periodic=periodic,
                    z_coupling=coupling,
                )
                for i, (uwj_floor, path_info) in enumerate(
                    islice(provider_pf, (max(self.nums) + 1))
                ):
                    if i not in self.nums:
                        continue
                    working_uwj_floor = uwj_floor.copy()
                    trim(
                        uwj_floor=working_uwj_floor,
                        path_info=path_info,
                        lattice_survey=self.lsurvey,
                    )

    def test_trim_removes_worst_uwkj(self):
        dimensions = [(6, 6, 12), (5, 5, 15)]
        prescribed = False
        periodic = True
        sub_kind = "main"
        worst_uwkj0 = UWKJ(u=0, w=2, k=3, j=1)
        worst_uwj0, worst_k0 = worst_uwkj0.uwj, worst_uwkj0.k
        worst_uwkj1 = UWKJ(u=1, w=2, k=1, j=1)
        worst_uwj1, worst_k1 = worst_uwkj1.uwj, worst_uwkj1.k
        for tile_kind, (Lx, Ly, Lz) in product([TileKind.LADDER, TileKind.SQUARE], dimensions):
            provider_pf = provider_floor_zpaths(
                tile_kind=tile_kind,
                sub_kind=sub_kind,
                Lx=Lx,
                Ly=Ly,
                Lz=Lz,
                lattice_survey=self.lsurvey,
                prescribed=prescribed,
                periodic=periodic,
            )
            for i, (uwj_floor, path_info) in enumerate(islice(provider_pf, max(self.nums) + 1)):
                if i not in self.nums:
                    continue

                working_uwj_floor = uwj_floor.copy()

                trim(
                    uwj_floor=working_uwj_floor,
                    path_info=path_info,
                    lattice_survey=self.lsurvey,
                )
                working_floor = working_uwj_floor.floor
                self.assertFalse(worst_k0 in working_floor[worst_uwj0])
                self.assertFalse(worst_k1 in working_floor[worst_uwj1])

    def test_trim_leaves_no_extra(self):
        dimensions = [(6, 6, 12), (5, 5, 15)]
        prescribed = False
        periodic = True
        sub_kind = "main"
        for tile_kind, (Lx, Ly, Lz) in product([TileKind.LADDER, TileKind.SQUARE], dimensions):
            provider_pf = provider_floor_zpaths(
                tile_kind=tile_kind,
                sub_kind=sub_kind,
                Lx=Lx,
                Ly=Ly,
                Lz=Lz,
                lattice_survey=self.lsurvey,
                prescribed=prescribed,
                periodic=periodic,
            )
            for i, (uwj_floor, path_info) in enumerate(islice(provider_pf, max(self.nums) + 1)):
                if i not in self.nums:
                    continue

                counter_node = Counter(x for pair in path_info.path_vh for x in pair)

                working_uwj_floor = uwj_floor.copy()

                trim(
                    uwj_floor=working_uwj_floor,
                    path_info=path_info,
                    lattice_survey=self.lsurvey,
                )
                working_floor = working_uwj_floor.floor
                for uwj_dict in working_floor.values():
                    idx = uwj_dict["idx"]
                    self.assertEqual(len(uwj_dict["perfect_ks"]), counter_node[idx])

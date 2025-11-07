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
from minorminer.utils.zephyr.node_edge import Edge, ZNode

from minorminer.cube_embedding._floor.cure import cure, get_all_impactful_missing
from minorminer.cube_embedding._floor.lz_provider import provider_floor_zpaths
from minorminer.cube_embedding._floor.trim import trim
from minorminer.cube_embedding._floor.uwj_floor import QuoFloor, UWJFloor
from minorminer.cube_embedding._tile import TileKind, ZCoupling, ladder_tiles, parse_seq
from minorminer.cube_embedding._tile.ladder import all_ladder_idx_chains
from minorminer._lattice_utils import UWKJZ, ZLatticeSurvey


class TestGetAllImpactful(TestCase):
    def setUp(self) -> None:
        shape = m, t = 6, 4
        int_edges = [
            ((4, 3, 3), (3, 4, 1)),
            ((6, 1, 2), (7, 2, 0)),
            ((10, 1, 0), (11, 2, 0)),
            ((11, 4, 1), (12, 3, 0)),
            ((1, 6, 1), (2, 7, 2)),
            ((6, 7, 0), (7, 8, 0)),
            ((4, 7, 3), (3, 8, 1)),
            ((9, 6, 1), (10, 7, 1)),
            ((4, 9, 0), (5, 10, 1)),
            ((5, 8, 2), (6, 9, 3)),
            ((10, 11, 1), (11, 10, 1)),
        ]
        int_edges = [
            (tuple(ZNode(x, shape=shape).zcoord), tuple(ZNode(y, shape=shape).zcoord))
            for (x, y) in int_edges
        ]

        G = zephyr_graph(m=m, t=t, coordinates=True)
        G.remove_edges_from(int_edges)
        self.lsurvey = ZLatticeSurvey(G)

        self.lad_idx_chains = all_ladder_idx_chains
        self.lad_indices = {
            sub_kind: {i: node.node_kind for i, node in enumerate(tile.zns)}
            for sub_kind, tile in ladder_tiles.items()
        }

    def test_impactful_lad_main(self):
        sub_kind = "main"
        tile = ladder_tiles[sub_kind]
        for dim in [(6, 6), (3, 3)]:
            qfloor = QuoFloor(tile=tile, dim=dim)
            uwj_floor = UWJFloor.from_qfloor_graph(qfloor=qfloor, lattice_survey=self.lsurvey)
            shape = (6, 4)
            expected_imp_edges = [
                ((4, 3, 3), (3, 4, 1)),  # Edge(6, 7)
                ((11, 4, 1), (12, 3, 0)),  # Edge(6, 7)
                ((4, 7, 3), (3, 8, 1)),  # Edge(6, 7)
                ((4, 9, 0), (5, 10, 1)),  # Edge(0, 2)
                ((5, 8, 2), (6, 9, 3)),  # Edge(1, 3)
                ((10, 11, 1), (11, 10, 1)),  # Edge(4, 5)
            ]
            expected_imp_edges = [
                (UWKJZ(*ZNode(x, shape=shape).zcoord), UWKJZ(*ZNode(y, shape=shape).zcoord))
                for (x, y) in expected_imp_edges
            ]
            path = [Edge(0, 1), Edge(2, 3), Edge(4, 5), Edge(6, 7)]
            path_info = next(
                parse_seq(
                    seq=path,
                    idx_chains=self.lad_idx_chains[sub_kind],
                    periodic=False,
                    indices=self.lad_indices[sub_kind],
                    z_coupling=ZCoupling.EITHER,
                )
            )
            impactful_edges = get_all_impactful_missing(
                uwj_floor=uwj_floor, lattice_survey=self.lsurvey, zpath_info=path_info
            )
            self.assertEqual(len(impactful_edges), len(expected_imp_edges))
            for xy in expected_imp_edges:
                self.assertTrue(xy in impactful_edges or xy[::-1] in impactful_edges)

    def test_impactful_lad_anti(self):
        sub_kind = "anti"
        tile = ladder_tiles[sub_kind]
        for dim in [(6, 6), (3, 3)]:
            qfloor = QuoFloor(tile=tile, dim=dim)
            uwj_floor = UWJFloor.from_qfloor_graph(qfloor=qfloor, lattice_survey=self.lsurvey)
            shape = (6, 4)
            expected_imp_edges = [
                ((6, 1, 2), (7, 2, 0)),
                ((10, 1, 0), (11, 2, 0)),
                ((1, 6, 1), (2, 7, 2)),
                ((9, 6, 1), (10, 7, 1)),
            ]
            expected_imp_edges = [
                (UWKJZ(*ZNode(x, shape=shape).zcoord), UWKJZ(*ZNode(y, shape=shape).zcoord))
                for (x, y) in expected_imp_edges
            ]
            path = [Edge(0, 2), Edge(1, 4), Edge(3, 6), Edge(5, 7)]
            path_info = next(
                parse_seq(
                    seq=path,
                    idx_chains=self.lad_idx_chains[sub_kind],
                    periodic=False,
                    indices=self.lad_indices[sub_kind],
                    z_coupling=ZCoupling.EITHER,
                )
            )
            impactful_edges = get_all_impactful_missing(
                uwj_floor=uwj_floor, lattice_survey=self.lsurvey, zpath_info=path_info
            )
            self.assertEqual(len(impactful_edges), len(expected_imp_edges))
            for xy in expected_imp_edges:
                self.assertTrue(xy in impactful_edges or xy[::-1] in impactful_edges)


class TestCure(TestCase):
    def setUp(self) -> None:
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
        int_edges_to_remove = [
            ((4, 3, 3), (3, 4, 1)),  # Edge(6, 7)
            ((6, 1, 2), (7, 2, 0)),  # Edge(3, 5)
            ((10, 1, 0), (11, 2, 0)),  # Edge(3, 5)
            ((11, 4, 1), (12, 3, 0)),  # Edge(6, 7)
            ((1, 6, 1), (2, 7, 2)),  # Edge(2, 4)
            ((6, 7, 0), (7, 8, 0)),  # Edge(4, 6)
            ((4, 7, 3), (3, 8, 1)),  # Edge(6, 7)
            ((9, 6, 1), (10, 7, 1)),  # Edge(2, 4)
            ((4, 9, 0), (5, 10, 1)),  # Edge(0, 2)
            ((5, 8, 2), (6, 9, 3)),  # Edge(1, 3)
            ((10, 11, 1), (11, 10, 1)),  # Edge(4, 5)
            ((3, 6, 0), (4, 5, 3)),
            ((6, 5, 1), (7, 4, 1)),
        ]
        int_edges_to_remove = [
            (tuple(ZNode(x, shape=shape).zcoord), tuple(ZNode(y, shape=shape).zcoord))
            for (x, y) in int_edges_to_remove
        ]

        G = zephyr_graph(m=m, t=t, coordinates=True)
        G.remove_nodes_from(nodes_to_remove)
        G.remove_edges_from(odd_edges_to_remove + ext_edges_to_remove + int_edges_to_remove)

        G = zephyr_graph(m=m, t=t, coordinates=True)
        G.remove_edges_from(int_edges_to_remove)
        self.lsurvey = ZLatticeSurvey(G)
        self.lsurvey_edges = self.lsurvey.edges
        self.nums = [1, 2, 4]

    def test_chains(self):
        dimensions = [(1, 1, 1), (3, 3, 5), (3, 6, 14)]
        params = [
            (True, True, TileKind.LADDER, "anti", ZCoupling.ZERO_ONE),
            (True, True, TileKind.LADDER, "anti", ZCoupling.ZERO_ONE_ONE_ZERO),
            (True, True, TileKind.LADDER, "main", ZCoupling.EITHER),
            (True, False, TileKind.LADDER, "anti", ZCoupling.ZERO_ONE),
            (True, True, TileKind.SQUARE, "anti", ZCoupling.EITHER),
            (False, False, TileKind.SQUARE, "main", ZCoupling.ZERO_ONE),
            (False, False, TileKind.SQUARE, "main", ZCoupling.EITHER),
        ]
        for periodic, prescribed, tile_kind, sub_kind, coupling in params:
            for Lx, Ly, Lz in dimensions:
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
                for uwj_floor, path_info in provider_pf:
                    working_uwj_floor = uwj_floor.copy()
                    trim(
                        uwj_floor=working_uwj_floor,
                        path_info=path_info,
                        lattice_survey=self.lsurvey,
                    )
                    try:
                        cure(
                            uwj_floor=working_uwj_floor,
                            zpath_info=path_info,
                            lattice_survey=self.lsurvey,
                        )
                        path = path_info.path_vh
                        col_row_nodes = working_uwj_floor.get_col_row_nodes(Lx=Lx, Ly=Ly)
                        for xy_nodes in col_row_nodes.values():
                            for a, b in path:
                                edge = Edge(xy_nodes[a].pop(0), xy_nodes[b].pop(0))
                                self.assertTrue(
                                    edge in self.lsurvey_edges,
                                    msg=f"on {Lx, Ly, Lz = }, {periodic, prescribed, tile_kind, sub_kind},  {edge = } in {self.lsurvey.missing_edges}",
                                )
                        break
                    except ValueError:
                        pass

    def test_inter_chains(self):
        def chains_have_connection(c1, c2):
            for x in c1:
                for y in c2:
                    if Edge(x, y) in self.lsurvey_edges:
                        return True
            return False

        dimensions = [(3, 3, 5), (3, 6, 14)]
        params = [
            (True, True, TileKind.LADDER, "anti", ZCoupling.ZERO_ONE),
            (True, True, TileKind.LADDER, "anti", ZCoupling.ZERO_ONE_ONE_ZERO),
            (True, True, TileKind.LADDER, "main", ZCoupling.EITHER),
            (True, False, TileKind.LADDER, "anti", ZCoupling.ZERO_ONE),
            (True, True, TileKind.SQUARE, "anti", ZCoupling.EITHER),
            (False, False, TileKind.SQUARE, "main", ZCoupling.ZERO_ONE),
            (False, False, TileKind.SQUARE, "main", ZCoupling.EITHER),
        ]
        for periodic, prescribed, tile_kind, sub_kind, coupling in params:
            for Lx, Ly, Lz in dimensions:
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
                for uwj_floor, path_info in provider_pf:
                    working_uwj_floor = uwj_floor.copy()
                    trim(
                        uwj_floor=working_uwj_floor,
                        path_info=path_info,
                        lattice_survey=self.lsurvey,
                    )
                    try:
                        cure(
                            uwj_floor=working_uwj_floor,
                            zpath_info=path_info,
                            lattice_survey=self.lsurvey,
                        )
                        path = path_info.path_vh
                        cube_chains = {}
                        col_row_nodes = working_uwj_floor.get_col_row_nodes(Lx=Lx, Ly=Ly)
                        for (x, y), xy_nodes in col_row_nodes.items():
                            for z, (a, b) in enumerate(path):
                                edge = Edge(xy_nodes[a].pop(0), xy_nodes[b].pop(0))
                                cube_chains[(x, y, z)] = edge
                        for (x, y, z), xyz_chain in cube_chains.items():
                            if z < Lz - 1:
                                xyzp1_chain = cube_chains[(x, y, z + 1)]
                                self.assertTrue(chains_have_connection(xyz_chain, xyzp1_chain))
                            if z == Lz - 1 and periodic:
                                xyzp1_chain = cube_chains[(x, y, 0)]
                                self.assertTrue(chains_have_connection(xyz_chain, xyzp1_chain))
                        break
                    except ValueError:
                        pass

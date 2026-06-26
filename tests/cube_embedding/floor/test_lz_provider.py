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

from minorminer.cube_embedding._floor.lz_provider import (
    _potentially_provides_nonprescribed,
    _potentially_provides_prescribed,
    provider_floor_zpaths,
)
from minorminer.cube_embedding._tile.kind import TileKind
from minorminer.cube_embedding._tile.ladder import ladder_idx_chain
from minorminer.cube_embedding._tile.square import square_idx_chain
from minorminer._lattice_utils import ZLatticeSurvey


class TestPotentiallyProvidesPrescribed(TestCase):
    def setUp(self) -> None:
        self.sq_idx_chains = list(square_idx_chain(sub_kind="main", prescribed=True))
        self.lad_idx_chains_main = list(ladder_idx_chain(sub_kind="main", prescribed=True))
        self.lad_idx_chains_anti = list(ladder_idx_chain(sub_kind="anti", prescribed=True))
        self.idx_supply1 = {0: 1, 1: 4, 3: 4, 4: 3, 6: 2, 5: 2, 7: 3}
        self.idx_supply2 = {0: 0, 1: 1, 2: 3, 3: 2, 4: 4, 5: 4, 6: 2}

    def test_sq(self):
        for idx_supply in [self.idx_supply1, self.idx_supply2]:
            max_supply = sum(
                [
                    min(idx_supply.get(idx_chain[0], 0), idx_supply.get(idx_chain[1], 0))
                    for idx_chain in self.sq_idx_chains
                ]
            )
            for supply in [0, 1, max_supply // 2, max_supply, max_supply + 1, max_supply + 3]:
                if supply <= max_supply:
                    self.assertTrue(
                        _potentially_provides_prescribed(
                            idx_supply=idx_supply,
                            needed_supply=supply,
                            idx_chains=self.sq_idx_chains,
                        )
                    )
                else:
                    self.assertFalse(
                        _potentially_provides_prescribed(
                            idx_supply=idx_supply,
                            needed_supply=supply,
                            idx_chains=self.sq_idx_chains,
                        )
                    )

    def test_lad_main(self):
        for idx_supply in [self.idx_supply1, self.idx_supply2]:
            max_supply = sum(
                [
                    min(idx_supply.get(idx_chain[0], 0), idx_supply.get(idx_chain[1], 0))
                    for idx_chain in self.lad_idx_chains_main
                ]
            )
            for supply in [0, 1, max_supply // 2, max_supply, max_supply + 1, max_supply + 3]:
                if supply <= max_supply:
                    self.assertTrue(
                        _potentially_provides_prescribed(
                            idx_supply=idx_supply,
                            needed_supply=supply,
                            idx_chains=self.lad_idx_chains_main,
                        )
                    )
                else:
                    self.assertFalse(
                        _potentially_provides_prescribed(
                            idx_supply=idx_supply,
                            needed_supply=supply,
                            idx_chains=self.lad_idx_chains_main,
                        )
                    )

    def test_lad_anti(self):
        for idx_supply in [self.idx_supply1, self.idx_supply2]:
            max_supply = sum(
                [
                    min(idx_supply.get(idx_chain[0], 0), idx_supply.get(idx_chain[1], 0))
                    for idx_chain in self.lad_idx_chains_anti
                ]
            )
            for supply in [0, 1, max_supply // 2, max_supply, max_supply + 1, max_supply + 3]:
                if supply <= max_supply:
                    self.assertTrue(
                        _potentially_provides_prescribed(
                            idx_supply=idx_supply,
                            needed_supply=supply,
                            idx_chains=self.lad_idx_chains_anti,
                        )
                    )
                else:
                    self.assertFalse(
                        _potentially_provides_prescribed(
                            idx_supply=idx_supply,
                            needed_supply=supply,
                            idx_chains=self.lad_idx_chains_anti,
                        )
                    )


class TestPotentiallyProvidesNonprescribed(TestCase):
    def test_provides_nonpres(self) -> None:
        idx_supply1 = {0: 1, 1: 4, 3: 4, 4: 3, 6: 2, 7: 3}
        idx_supply2 = {0: 0, 1: 1, 2: 3, 3: 2, 4: 4, 5: 4, 6: 2}
        indices1 = {NodeKind.VERTICAL: [0, 1, 2, 3], NodeKind.HORIZONTAL: [4, 5, 6, 7]}
        indices2 = {NodeKind.VERTICAL: list(range(8))}
        indices3 = {NodeKind.VERTICAL: [0, 7], NodeKind.HORIZONTAL: list(range(1, 8))}
        indices4 = {NodeKind.VERTICAL: [7], NodeKind.HORIZONTAL: list(range(1, 8))}
        for idx_supply in [idx_supply1, idx_supply2]:
            for indices in [indices1, indices2, indices3, indices4]:
                supply_v, supply_h = 0, 0
                for idx, idx_supp in idx_supply.items():
                    if idx in indices[NodeKind.VERTICAL]:
                        supply_v += idx_supp
                    elif idx in indices[NodeKind.HORIZONTAL]:
                        supply_h += idx_supp
                max_supply = min(supply_v, supply_h)
                for supply in [0, 1, max_supply // 2, max_supply, max_supply + 1, max_supply + 3]:
                    if supply <= max_supply:
                        self.assertTrue(
                            _potentially_provides_nonprescribed(
                                idx_supply=idx_supply, needed_supply=supply, indices=indices
                            )
                        )
                    else:
                        self.assertFalse(
                            _potentially_provides_nonprescribed(
                                idx_supply=idx_supply, needed_supply=supply, indices=indices
                            )
                        )


class TestProviderFloorPaths(TestCase):
    def setUp(self):
        shape = m, t = 6, 4
        nodes_to_remove = [(3, 2, 0), (10, 1, 1), (10, 21, 2), (14, 19, 3)]
        nodes_to_remove = [tuple(ZNode(xyk, shape=shape).zcoord) for xyk in nodes_to_remove]
        ext_edges_to_remove = [
            ((10, 9, 1), (10, 13, 1)),
            ((14, 3, 3), (14, 7, 3)),
            ((3, 16, 2), (7, 16, 2)),
        ]
        ext_edges_to_remove = [
            (tuple(ZNode(x, shape=shape).zcoord), tuple(ZNode(y, shape=shape).zcoord))
            for (x, y) in ext_edges_to_remove
        ]

        G = zephyr_graph(m=m, t=t, coordinates=True)
        G.remove_nodes_from(nodes_to_remove)
        G.remove_edges_from(ext_edges_to_remove)
        self.lsurvey = ZLatticeSurvey(G)
        self.border_dim = [(1, 1, 1), (1, 1, 16), (3, 3, 16), (4, 4, 14)]

    def test_lad_per_pres(self):
        tile_kind = TileKind.LADDER
        periodic, prescribed = True, True
        extra_dim = [(6, 6, 12)]
        for sub_kind in ["main", "anti"]:
            for dim in self.border_dim + extra_dim:
                possibilities = provider_floor_zpaths(
                    tile_kind=tile_kind,
                    sub_kind=sub_kind,
                    Lx=dim[0],
                    Ly=dim[1],
                    Lz=dim[2],
                    lattice_survey=self.lsurvey,
                    prescribed=prescribed,
                    periodic=periodic,
                )
                try:
                    first = next(possibilities)
                    self.assertIsNotNone(first)
                    break
                except StopIteration:
                    self.fail(f"yielded nothing on {dim = }")

    def test_lad_per_nonpres(self):
        tile_kind = TileKind.LADDER
        periodic, prescribed = True, False
        extra_dim = [(6, 6, 13)]
        for sub_kind in ["main", "anti"]:
            for dim in self.border_dim + extra_dim:
                possibilities = provider_floor_zpaths(
                    tile_kind=tile_kind,
                    sub_kind=sub_kind,
                    Lx=dim[0],
                    Ly=dim[1],
                    Lz=dim[2],
                    lattice_survey=self.lsurvey,
                    prescribed=prescribed,
                    periodic=periodic,
                )
                try:
                    first = next(possibilities)
                    self.assertIsNotNone(first)
                    break
                except StopIteration:
                    self.fail(f"yielded nothing on {dim = }")

    def test_lad_nonper_pres(self):
        tile_kind = TileKind.LADDER
        periodic, prescribed = False, True
        extra_dim = [(6, 6, 12)]
        for sub_kind in ["main", "anti"]:
            for dim in self.border_dim + extra_dim:
                possibilities = provider_floor_zpaths(
                    tile_kind=tile_kind,
                    sub_kind=sub_kind,
                    Lx=dim[0],
                    Ly=dim[1],
                    Lz=dim[2],
                    lattice_survey=self.lsurvey,
                    prescribed=prescribed,
                    periodic=periodic,
                )
                try:
                    first = next(possibilities)
                    self.assertIsNotNone(first)
                    break
                except StopIteration:
                    self.fail(f"yielded nothing on {dim = }")

    def test_lad_nonper_nonpres(self):
        tile_kind = TileKind.LADDER
        periodic, prescribed = False, False
        extra_dim = [(6, 6, 13)]
        for sub_kind in ["main", "anti"]:
            for dim in self.border_dim + extra_dim:
                possibilities = provider_floor_zpaths(
                    tile_kind=tile_kind,
                    sub_kind=sub_kind,
                    Lx=dim[0],
                    Ly=dim[1],
                    Lz=dim[2],
                    lattice_survey=self.lsurvey,
                    prescribed=prescribed,
                    periodic=periodic,
                )
                try:
                    first = next(possibilities)
                    self.assertIsNotNone(first)
                    break
                except StopIteration:
                    self.fail(f"yielded nothing on {dim = }")

    def test_sq_main(self):
        tile_kind = TileKind.SQUARE
        sub_kind = "main"
        extra_dim = [(6, 6, 13)]
        for periodic in [True, False]:
            for prescribed in [True, False]:
                for dim in self.border_dim + extra_dim:
                    possibilities = provider_floor_zpaths(
                        tile_kind=tile_kind,
                        sub_kind=sub_kind,
                        Lx=dim[0],
                        Ly=dim[1],
                        Lz=dim[2],
                        lattice_survey=self.lsurvey,
                        prescribed=prescribed,
                        periodic=periodic,
                    )
                    try:
                        first = next(possibilities)
                        self.assertIsNotNone(first)
                        break
                    except StopIteration:
                        self.fail(f"yielded nothing on {dim = } for {prescribed = }, {periodic = }")

    def test_sq_anti(self):
        tile_kind = TileKind.SQUARE
        sub_kind = "anti"
        extra_dim = [(6, 6, 11)]
        for periodic in [True, False]:
            for prescribed in [True, False]:
                for dim in self.border_dim + extra_dim:
                    possibilities = provider_floor_zpaths(
                        tile_kind=tile_kind,
                        sub_kind=sub_kind,
                        Lx=dim[0],
                        Ly=dim[1],
                        Lz=dim[2],
                        lattice_survey=self.lsurvey,
                        prescribed=prescribed,
                        periodic=periodic,
                    )
                    try:
                        first = next(possibilities)
                        self.assertIsNotNone(first)
                        break
                    except StopIteration:
                        self.fail(f"yielded nothing on {dim = } for {prescribed = }, {periodic = }")

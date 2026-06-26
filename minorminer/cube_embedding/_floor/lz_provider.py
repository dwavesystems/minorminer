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


"""
Contains tools to find the floors within a (partially-yielded) Zephyr graph
which, ignoring the missing internal edges within the tiles, provide a cube embedding of
desired dimensions.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Generator, Literal

from minorminer.utils.zephyr.node_edge import Edge, NodeKind, ZNode
from minorminer.utils.zephyr.plane_shift import ZPlaneShift

from minorminer.cube_embedding._floor.uwj_floor import QuoFloor, UWJFloor
from minorminer.cube_embedding._tile import (PathInfo, TileKind, ZCoupling, ladder_idx_chain,
                                          ladder_tiles, ladder_z_paths, square_idx_chain,
                                          square_tiles, square_z_paths)
from minorminer._lattice_utils import QuoTile, ZLatticeSurvey

__all__ = ["provider_floor_zpaths"]


def _get_idx_supply(uwj_floor: UWJFloor) -> dict[int, int]:
    """Finds the supply of node indices in the floor.

    Args:
        uwj_floor (UWJFloor): floor to find the supply of node indices for.

    Returns:
        dict[int, int]: A dictionary of the form {idx1: supply_idx1, idx2: supply_idx2, ...}
        which maps the index of a node in the floor to its supply.
    """
    floor = uwj_floor.floor
    all_idx_supply = defaultdict(list)
    for uwj_dict in floor.values():
        idx = uwj_dict["idx"]
        all_idx_supply[idx].append(len(uwj_dict["perfect_ks"]))
    return {idx: min(idx_num_perf) for idx, idx_num_perf in all_idx_supply.items()}


def _potentially_provides_prescribed(
    idx_supply: dict[int, int], needed_supply: int, idx_chains: list[Edge]
) -> bool:
    """Checks if the supply of node indices and the prescribed index chains potentially
        allow constructing the required number of chains.

    Note: Assumes elements of ``idx_chains`` are pairwise disjoint
        (i.e. no Edge(0, 1), Edge(0, 2), ...)

    Args:
        idx_supply (dict[int, int]): The available supply of node indices.
        needed_supply (int): The total number of prescribed chains to be constructed.
        idx_chains (list[Edge]): The prescribed index chains.

    Returns:
        bool: ``True`` if the given number of chains can be constructed; ``False`` otherwise.
    """
    supply = 0
    for idx1, idx2 in idx_chains:
        supply += min(idx_supply.get(idx1, 0), idx_supply.get(idx2, 0))
    return supply >= needed_supply


def _potentially_provides_nonprescribed(
    idx_supply: dict[int, int], needed_supply: int, indices: dict[NodeKind, list[int]]
) -> bool:
    """Checks if the supply of node indices, given the node index kinds, potentially
        allow constructing the required number of chains.

    Args:
        idx_supply (dict[int, int]): The available supply of node indices.
        needed_supply (int): The total number of chains to be constructed.
        indices (dict[NodeKind, list[int]]): A dicctionary of the form
            {NodeKind.VERTICAL: [vidx1, vidx2, ...], NodeKind.HORIZONTAL: [hidx1, hidx2, ...]}
            which maps each node kind to the list of node indices of that kind.
    Returns:
        bool: ``True`` if the given number of chains can be constructed; ``False`` otherwise.
    """
    v_supply = sum([idx_supply.get(v, 0) for v in indices.get(NodeKind.VERTICAL, list())])
    h_supply = sum([idx_supply.get(h, 0) for h in indices.get(NodeKind.HORIZONTAL, list())])
    return min(v_supply, h_supply) >= needed_supply


def provider_floor_zpaths(
    tile_kind: TileKind,
    sub_kind: Literal["main", "anti"],
    Lx: int,
    Ly: int,
    Lz: int,
    lattice_survey: ZLatticeSurvey,
    prescribed: bool,
    periodic: bool,
    z_coupling: ZCoupling | None = None,
    **kwargs,
) -> Generator[tuple[UWJFloor, PathInfo], None, None]:
    """Generates the floors with dimension ``(Lx, Ly)``,
        together with information about z-paths of
        length ``Lz`` on the tiles of the floor,
        respecting the given tile kind and z-coupling
        constraints and prescription and periodicity.

    Args:
        tile_kind (TileKind): The kind of tile forming the floor.
        sub_kind (Literal[&quot;main&quot;, &quot;anti&quot;]): The sub-kind of tile forming the floor.
        Lx (int): x-dimension of the floor to be generated.
        Ly (int): y-dimension of the floor to be generated.
        Lz (int): The length of path to be generated.
        lattice_survey (LatticeSurvey): LatticeSurvey of the Zephyr graph to construct the floor on.
        prescribed (bool):  If ``True``, restricts path construction to prescribed chains.
        periodic (bool): If ``True``, restricts path construction to periodic paths (i.e. "closed trails" in graph theory terminology).

    Yields:
        Generator[tuple[UWJFloor, PathInfo], None, None]: A pair of the form (floor, path_info)
        where floor has dimension ``(Lx, Ly)`` and z-path has length ``Lz``
        which can be constructed on the floor, while respecting the
        prescription, periodicty, z-coupling and tile kind constraints.
    """

    def path_finder(*args, **kwargs):
        if tile_kind is TileKind.LADDER:
            return ladder_z_paths(*args, **kwargs)
        else:
            return square_z_paths(*args, **kwargs)

    m = lattice_survey.m
    if Lx > m or Ly > m:
        return

    if tile_kind is TileKind.LADDER:
        corner_tile00 = ladder_tiles[sub_kind]
    elif tile_kind is TileKind.SQUARE:
        corner_tile00 = square_tiles[sub_kind]

    # Pass on the grid size of lattice_survey to the tile nodes
    corner_tile00_zns = [ZNode(coord=x.ccoord, shape=(m, None)) for x in corner_tile00]

    corner_tile00 = QuoTile(zns=corner_tile00_zns)

    if z_coupling is None:
        z_coupling = ZCoupling.EITHER

    for x_shift, y_shift in [(0, 0), (1, 1)]:
        try:
            dr_corner_zns = [
                a + ZPlaneShift(4 * (Lx - 1) + x_shift, 4 * (Ly - 1) + y_shift)
                for a in corner_tile00
            ]
        except ValueError:
            continue
        max_x_original_box = max([a.ccoord.x for a in dr_corner_zns])
        max_y_original_box = max([a.ccoord.y for a in dr_corner_zns])
        max_right_shift = (4 * m - max_x_original_box) // 2
        max_down_shift = (4 * m - max_y_original_box) // 2
        for right_step in range(max_right_shift + 1):
            for down_step in range(max_down_shift + 1):
                corner_tile = corner_tile00 + ZPlaneShift(2 * right_step, 2 * down_step)
                qfloor = QuoFloor(tile=corner_tile, dim=(Lx, Ly))
                uwj_floor_ij = UWJFloor.from_qfloor_graph(
                    qfloor=qfloor, lattice_survey=lattice_survey
                )

                idx_supply = _get_idx_supply(uwj_floor=uwj_floor_ij)

                if prescribed:
                    if tile_kind is TileKind.LADDER:
                        idx_chains = list(ladder_idx_chain(sub_kind=sub_kind, prescribed=True))
                    else:
                        idx_chains = list(square_idx_chain(sub_kind=sub_kind, prescribed=True))

                    provides = _potentially_provides_prescribed(
                        idx_supply=idx_supply,
                        needed_supply=Lz,
                        idx_chains=idx_chains,
                    )

                else:
                    indices_ij = {
                        dim: list(dim_dict.keys()) for dim, dim_dict in uwj_floor_ij.indices.items()
                    }
                    provides = _potentially_provides_nonprescribed(
                        idx_supply=idx_supply,
                        needed_supply=Lz,
                        indices=indices_ij,
                    )
                if not provides:
                    continue

                indices_ij: dict[int, NodeKind] = {
                    i: kind for kind, dict_kind in uwj_floor_ij.indices.items() for i in dict_kind
                }

                for path_info in path_finder(
                    initial_supply=idx_supply,
                    sub_kind=sub_kind,
                    num=Lz,
                    periodic=periodic,
                    prescribed=prescribed,
                    z_coupling=z_coupling,
                    indices=indices_ij,
                ):
                    yield (uwj_floor_ij, path_info)

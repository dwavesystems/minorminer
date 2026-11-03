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
A helper class for working with floors, equipped with tools needed for cube embedding purposes.
"""

from __future__ import annotations

from collections import defaultdict, namedtuple
from itertools import product
from typing import Iterable, TypedDict

import networkx as nx
from dwave.system import DWaveSampler
from minorminer.utils.zephyr.node_edge import Edge, NodeKind, ZNode
from minorminer.utils.zephyr.plane_shift import ZPlaneShift
from minorminer._lattice_utils import ZSE, UWJ, UWKJZ, QuoFloor, ZLatticeSurvey


__all__ = ["UWJFloor"]


class UWJStats(TypedDict):
    """A helper object to represent statistics related to a quotient external path (``UWJ``)
    in a floor object.
    """

    dim_num: int  # The column (row) number of a floor the
    # vertical (horizontal) quotient external path belongs to.
    idx: int  # The index of nodes on the quotient external path in the quotient tiles forming the quotient floor.
    zse: ZSE  # The ZSE corresponding to the z-strech needed to be covered for the quotient external path
    perfect_ks: list[int]  # The list of k's corresponding to the external paths
    # UWKJ(u=uwj.u, w=uwj.w, k=k, j=uwj.j) that cover all the strech needed.


TileCoordIdx = namedtuple("TileCoordIdx", ["tile_coord", "idx"])
TileCoordEdge = namedtuple("TileCoordEdge", ["tile_coord", "edge"])


class UWJFloor:
    """Initializes a ``UWJFloor`` object with indices and floor.
        An object to capture the information related to the quotient
        external paths of a floor constructed on a quotient floor ``QuoFloor``
        on a partially-yielded Zephyr graph.

    Args:
        indices (dict[NodeKind, dict[int, ZNode]]):
            - A dictionary of the form {NodeKind.VERTICAL: {idx: node}, NodeKind.HORIZONTAL: {idx: node}}.
            - Value of a key maps the indices of that type to the corresponding nodes in the top left corner tile of the floor.
        floor (dict[UWJ, UWJStats]):
            - Keys are quotient external paths of the floor.
            - Values are ``UWJStats`` capturing the statistics of the quotient external path in the floor.
    """

    def __init__(
        self,
        indices: dict[NodeKind, dict[int, ZNode]],
        floor: dict[UWJ, UWJStats],
    ):
        self._indices: dict[NodeKind, dict[int, ZNode]] = indices
        self._floor: dict[UWJ, UWJStats] = floor

    @staticmethod
    def from_qfloor_graph(
        qfloor: QuoFloor,
        G: nx.Graph | DWaveSampler | None = None,
        lattice_survey: ZLatticeSurvey | None = None,
    ) -> UWJFloor:
        """Constructs the floor on a quotient floor on a Zephyr graph.

        Args:
            qfloor (QuoFloor): The quotient floor on which the floor gets constructed.
            G (nx.Graph | DWaveSampler | None, optional):
                A graph or DWaveSampler with Zephyr topology to construct the floor on.
                Defaults to None.
            lattice_survey (LatticeSurvey | None, optional):
                LatticeSurvey of the Zephyr graph to construct the floor on.
                Defaults to None.

        Returns:
            UWJFloor: The floor constructed on the quotient floor on the
            provided Zephyr graph/sampler.
        """

        def is_covered(needed: ZSE, z_stretches: Iterable[ZSE]) -> bool:
            """
            Checks whether the z_stretch ``needed`` is fully covered by some z-stretch in ``z_stretches``.

            Args:
                needed: A ZSE representing the z-stretch to check.
                z_stretches: An iterable of ZSE representing available z-stretches.

            Returns:
                True if there exists a z-strech that fully contains ``needed``, False otherwise.
            """
            s, e = needed.z_start, needed.z_end
            return any(a <= s and b >= e for a, b in z_stretches)

        corner_zns = qfloor.tile.zns
        Lx, Ly = qfloor.dim

        if lattice_survey is None:
            if G is None:
                raise ValueError(f"Provide at least one of G or lattice_survey.")
            else:
                lattice_survey = ZLatticeSurvey(G)

        t = lattice_survey.t
        ext_paths_survey = lattice_survey.calculate_external_paths_stretch()

        uwj_floor = {}
        indices = {NodeKind.VERTICAL: dict(), NodeKind.HORIZONTAL: dict()}
        for idx, node in enumerate(corner_zns):
            indices[node.node_kind][idx] = node
            uwjz = UWKJZ(*node.zcoord)
            uwj = uwjz.uwj
            z_start = uwjz.z
            if uwjz.u == 0:  # vertical external path
                num_copies, z_stretch = Lx, Ly
                atom_shift = ZPlaneShift(4, 0)
            else:  # horizontal external path
                num_copies, z_stretch = Ly, Lx
                atom_shift = ZPlaneShift(0, 4)
            window = ZSE(z_start=z_start, z_end=z_start + z_stretch - 1)
            for dim_num in range(num_copies):
                uwjz = (node + dim_num * atom_shift).zcoord
                uwj = UWJ(u=uwjz.u, w=uwjz.w, j=uwjz.j)
                perfect_ks = [
                    k
                    for k in range(t)
                    if is_covered(needed=window, z_stretches=ext_paths_survey[uwj][k])
                ]
                uwj_floor[uwj] = UWJStats(
                    dim_num=dim_num, idx=idx, zse=window, perfect_ks=perfect_ks
                )
        return UWJFloor(indices=indices, floor=uwj_floor)

    @property
    def indices(self) -> dict[NodeKind, dict[int, ZNode]]:
        """
        Returns:
            dict[NodeKind, dict[int, ZNode]]:
                - A dictionary of the form
                  {NodeKind.VERTICAL: {idx: node}, NodeKind.HORIZONTAL: {idx: node}}.
                - Value of a key maps the indices of that type to the
                  corresponding nodes in the top left corner tile of the floor.
        """
        return self._indices

    @property
    def floor(self) -> dict[UWJ, UWJStats]:
        """Returns ``floor`` of self.
        - Keys are the quotient external paths.
        - Values are ``UWJStats`` capturing the statistics of the quotient external path.
        """
        return self._floor

    @floor.setter
    def floor(self, new_floor: dict[UWJ, UWJStats]):
        """Sets ``floor`` of self."""
        self._floor = new_floor

    def get_col_row_nodes(self, Lx: int, Ly: int) -> dict[tuple[int, int], dict[int, list[UWKJZ]]]:
        """Finds the dictionary where
        - keys are (col, row)
        - values are dictionaries where:
            -- key is the index of node in tile at coordinate (col, row)
            -- value is the list of uwkjz of nodes corresponding to the
                index of node in tile, with k's respecting the order of
                "perfect_ks" of the corresponding uwj.
        """
        col_row_uwkjz = defaultdict(dict)
        uwjz_corner_tile = {}
        for dir_dict in self.indices.values():
            for idx, node in dir_dict.items():
                uwjz_corner_tile[idx] = node
        for idx, node in uwjz_corner_tile.items():
            for col, row in product(range(Lx), range(Ly)):
                node_shifted = node + ZPlaneShift(4 * col, 4 * row)
                uwjz_col_row = UWKJZ(*node_shifted.zcoord)
                perfect_ks = self.floor[uwjz_col_row.uwj]["perfect_ks"]
                col_row_uwkjz[(col, row)][idx] = [
                    UWKJZ(
                        u=uwjz_col_row.u, w=uwjz_col_row.w, k=k, j=uwjz_col_row.j, z=uwjz_col_row.z
                    )
                    for k in perfect_ks
                ]
        return dict(col_row_uwkjz)

    def get_tile_coord_idx(self, uwkjz: UWKJZ) -> TileCoordIdx | None:
        """Returns the tile coordinates that a node belongs to and its node index.
            Returns ``None`` if the node isn't in self.

        Args:
            uwkjz (UWKJZ): the node to be located and indexed.

        Returns:
            dict[str, int] | None:
            If the node belongs to the floor, it gives a ``TileCoordIdx`` object.
                - The value of "tile_coord" is the tile's coordinate the node belongs to.
                - The value of "idx" is the node index of the node within the quotient floor underlying the floor.
            Returns ``None`` if the node isn't in the floor.
        """
        uwj = uwkjz.uwj
        if uwj not in self.floor:
            return None
        uwj_dict = self.floor[uwj]
        if not uwkjz.k in uwj_dict["perfect_ks"]:
            return None
        zs, ze = uwj_dict["zse"]
        z = uwkjz.z
        if z < zs or z > ze:
            return None
        dim_num = uwj_dict["dim_num"]
        if uwkjz.u == 0:
            col, row = dim_num, z - zs
        else:
            col, row = z - zs, dim_num
        return TileCoordIdx(tile_coord=(col, row), idx=uwj_dict["idx"])

    def get_uwj(self, idx: int, tile_coord: tuple[int, int]) -> UWJ:
        """Returns the quotient external path of a node index passing through
            the tile at coordinate.

        Args:
            idx (int): node index
            tile_coord (tuple[int, int]): coordinates of the tile.

        Returns:
            UWJ: The quotient external path of the node index passing through
            the tile at the given coordinate.
        """
        col, row = tile_coord
        for dict_dir in self.indices.values():
            if idx in dict_dir:
                node_corner = dict_dir[idx]
                break
        node_zcoord = (node_corner + ZPlaneShift(4 * col, 4 * row)).zcoord
        return UWJ(u=node_zcoord.u, w=node_zcoord.w, j=node_zcoord.j)

    def missing_internal_edges(
        self, lattice_survey: ZLatticeSurvey
    ) -> dict[tuple[UWKJZ, UWKJZ], TileCoordEdge]:
        """Finds all missing internal edges of a Zephyr graph that lie inside the tiles of the floor
            and finds the coordinates of the tile they lie in and the index edge within the tile they represent.

        Args:
            lattice_survey (LatticeSurvey): The survey of the zephyr graph.

        Returns:
            dict[tuple[UWKJZ, UWKJZ], TileCoordEdge]:
            - Keys are missing internal edges that lie completely inside a tile of the floor.
            - Values are ``TileCoordEdge``
        """
        floor_miss_internal = {}
        for a, b in lattice_survey.extra_missing_edges:
            if a.u == b.u:
                continue

            a_floor = self.floor.get(a.uwj)
            b_floor = self.floor.get(b.uwj)
            if a_floor is None or b_floor is None:
                continue
            if (not a.k in a_floor["perfect_ks"]) or (not b.k in b_floor["perfect_ks"]):
                continue

            a_col_row_idx = self.get_tile_coord_idx(a)
            b_col_row_idx = self.get_tile_coord_idx(b)
            if (not a_col_row_idx) or (not b_col_row_idx):  # a or b not in the floor
                continue

            a_coord = a_col_row_idx.tile_coord
            # Ensure a and b belong to the same tile
            if a_coord != b_col_row_idx.tile_coord:
                continue

            a_b_idx = (a_col_row_idx.idx, b_col_row_idx.idx)
            edge = Edge(*a_b_idx)
            if a_b_idx == (edge[0], edge[1]):
                a_b_normalized = (a, b)
            elif a_b_idx == (edge[1], edge[0]):
                a_b_normalized = (b, a)
            floor_miss_internal[a_b_normalized] = TileCoordEdge(edge=edge, tile_coord=a_coord)

        return floor_miss_internal

    def copy(self):
        """Returns a "deep" copy of ``self``."""

        def copy_stats(stats_dict) -> dict:
            copied_dict = {}
            for var, val in stats_dict.items():
                if isinstance(val, int):
                    copied_dict[var] = val
                elif isinstance(val, list):
                    copied_dict[var] = [k for k in val]
                elif isinstance(val, (ZSE, UWJ)):
                    copied_dict[var] = val
            return copied_dict

        copied_uwj_floor = {uwj: copy_stats(uwj_dict) for uwj, uwj_dict in self._floor.items()}
        return UWJFloor(indices=self.indices, floor=copied_uwj_floor)

    def __repr__(self) -> str:
        indices_str = f"indices={self._indices!r}"
        uwj_floor_str = f", floor={self._floor!r}"
        return f"{type(self).__name__}(" + indices_str + uwj_floor_str + ")"

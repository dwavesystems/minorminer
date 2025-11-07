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
Contains the tools for "curing" a floor given the z-path on its tiles; i.e. changing the
order of external paths of the floor within each quotient external paths so that the z-paths
can be constructed.
"""

from collections import defaultdict
from typing import NamedTuple

from minorminer.utils.zephyr.node_edge import Edge

from minorminer.cube_embedding._floor.uwj_floor import TileCoordEdge, UWJFloor
from minorminer.cube_embedding._tile import PathInfo
from minorminer._lattice_utils import UWJ, UWKJ, UWKJZ, ZLatticeSurvey


class EdgeOccInd(NamedTuple):
    edge: Edge
    occurrence_ind: list[tuple[int, int]]


"""A helper object to collect the impactful occurrence indices for each edge;
    i.e. the pair of occurrence indices which is necessary for successful forming of a z-path.
"""


def get_all_impactful_missing(
    uwj_floor: UWJFloor, lattice_survey: ZLatticeSurvey, zpath_info: PathInfo
) -> dict[tuple[UWKJZ, UWKJZ], EdgeOccInd]:
    """Finds all the missing internal edges lying completely within a tile of the floor
        which correspond to an "impactful" connection; i.e. a connection which needs
        to be cured. It also finds the ``EdgeOccInd`` of each edge containing all the
        "impactful" occurence indices of the edge.

    Args:
        uwj_floor (UWJFloor): The floor to get the impactful missing internal edges for.
        lattice_survey (LatticeSurvey): LatticeSurvey of the Zephyr graph the floor lies on.
        zpath_info (PathInfo): The statistics of the z-path on the floor's tiles.

    Returns:
        dict[tuple[UWKJZ, UWKJZ], EdgeOccInd]: A dictionary where
        - Keys are impactful missing internal edges lying completely within
            a tile of the floor.
        - Values are ``EdgeOccInd`` where
            - its ``edge`` value is the index edge the edge represents in the tile
            - its ``occurrence_ind`` value contains all pairs of occurrence indices
                which make the absence of ``edge`` impactful.
    """
    missing_internal_couplers = uwj_floor.missing_internal_edges(lattice_survey=lattice_survey)
    missing_tilecoord_edges: list[TileCoordEdge] = list(missing_internal_couplers.values())
    all_impactful_missing: dict[tuple[UWKJZ, UWKJZ], EdgeOccInd] = {}

    for ab, (ab_tile_coord, ab_edge) in missing_internal_couplers.items():
        for cr_edge_pos in zpath_info.crucial_edge_pos:
            if ab_edge == cr_edge_pos.edge:
                if not ab in all_impactful_missing:
                    all_impactful_missing[ab] = EdgeOccInd(edge=ab_edge, occurrence_ind=[])
                all_impactful_missing[ab].occurrence_ind.append(cr_edge_pos.pos)

        for edge_pos0, edge_pos1 in zpath_info.alt_edge_pos.items():
            if ab_edge == edge_pos0.edge:
                if (
                    TileCoordEdge(tile_coord=ab_tile_coord, edge=edge_pos1.edge)
                    in missing_tilecoord_edges
                ):
                    if not ab in all_impactful_missing:
                        all_impactful_missing[ab] = EdgeOccInd(edge=ab_edge, occurrence_ind=[])
                    all_impactful_missing[ab].occurrence_ind.append(edge_pos0.pos)
    return all_impactful_missing


def cure(
    uwj_floor: UWJFloor,
    zpath_info: PathInfo,
    lattice_survey: ZLatticeSurvey,
):
    """
        Given the floor and the information of the z-path
        desired to be constructed on the floor, it uses a backtracking
        algorithm to explore all possibilities to "cure" the floor;
        i.e. it attempts to order the external paths of the floor
        within each quotient external paths so that the z-paths
        can be constructed in all tiles.

    Args:
        uwj_floor (UWJFloor): The floor to cure.
        zpath_info (PathInfo): The ``PathInfo`` of the z-path desired to construct in
            the floor's tiles.
        lattice_survey (ZLatticeSurvey):
            ZLatticeSurvey of the Zephyr graph the floor lies on.
    """

    def find_good_occurrence_index() -> dict[UWKJ, int] | None:
        """For all external paths in the floor which miss an internal edge
            within a tile, it finds a "good" occurrence index; i.e.
            an occurrence index within the stack of its quotient external path
            which avoids using the missing internal edge in the embedding.

        Returns:
            dict[UWKJ, int] | None: Mapping an external path to its "good"
            occurrence index.
        """
        def are_consistent(u_ext: UWKJ, v_ext: UWKJ, u_oi: int, v_oi: int) -> bool:
            """Checks whether the proposed occurrence indices of two
                external paths is consistent.

            Args:
                u_ext (UWKJ): First external path.
                v_ext (UWKJ): Second external path
                u_oi (int): Occurrence index of first external path.
                v_oi (int): Occurrence index of second external path.

            Returns:
                bool: Whether the proposed occurrence indices of two
                external paths is consistent.
            """

            if u_ext == v_ext:
                return u_oi == v_oi
            if quo[u_ext] == quo[v_ext]:
                return u_oi != v_oi
            if not u_ext in impacted_graph or not v_ext in impacted_graph[u_ext]:
                return True
            return not (u_oi, v_oi) in impacted_graph[u_ext][v_ext]

        def backtrack(assigned_oi):
            """Uses backtracking to find occurrence index."""
            if len(assigned_oi) == len(impacted_graph):
                return assigned_oi

            # choose unassigned external path with fewest number of possibilities
            ext = min(
                (x for x in impacted_graph if x not in assigned_oi),
                key=lambda x: len(possible_oi[x]),
            )

            for v_oi in possible_oi[ext]:
                new_assigned_oi = assigned_oi.copy()
                new_assigned_oi[ext] = v_oi

                # forward checking
                old_assigned = {u: possible_oi[u].copy() for u in impacted_graph}
                consistent_flag = True
                for u in impacted_graph:
                    if u not in assigned_oi:
                        possible_oi[u] = {
                            cu for cu in possible_oi[u] if are_consistent(ext, u, v_oi, cu)
                        }
                        if not possible_oi[u]:
                            consistent_flag = False
                            break

                if consistent_flag:
                    result = backtrack(new_assigned_oi)
                    if result:
                        return result

                possible_oi.update(old_assigned)

            return None

        # Initially, each external path could appear anywhere in the list of perfect ks
        possible_oi = {
            ext: set(range(len(uwj_floor.floor[quo[ext]]["perfect_ks"]))) for ext in impacted_graph
        }
        return backtrack({})

    all_impactful_missing = get_all_impactful_missing(
        uwj_floor=uwj_floor, lattice_survey=lattice_survey, zpath_info=zpath_info
    )
    if not all_impactful_missing: # No internal edges is missing within a tile, nothing to cure
        return

    # Make the graph of external paths which miss an internal connection within a tile.
    impacted_graph = defaultdict(dict)
    for (u, v), list_uv in all_impactful_missing.items():
        u_ext_path = u.uwkj
        v_ext_path = v.uwkj
        impacted_graph[u_ext_path][v_ext_path] = list_uv.occurrence_ind
        impacted_graph[v_ext_path][u_ext_path] = [pos[::-1] for pos in list_uv.occurrence_ind]

    # Map external paths to their quotient
    quo: dict[UWKJ, UWJ] = {ext_path: ext_path.uwj for ext_path in impacted_graph}

    # Find an occurrence index for the impacted external paths
    prescribed_occ_ind: dict[UWKJZ, int] | None = find_good_occurrence_index()
    if not prescribed_occ_ind: # Exhaustive search, so if fails it cannot be cured
        raise ValueError(f"Cannot be cured")

    prescribed_oi = defaultdict(dict)
    for uwkjz, uwkjz_oi in prescribed_occ_ind.items():
        prescribed_oi[quo[uwkjz]][uwkjz.k] = uwkjz_oi

    # Finally, order the perfect_ks of each impacted quotient external path
    # according to the prescription.
    for uwj, uwj_prescription in prescribed_oi.items():
        uwj_stack = uwj_floor.floor[uwj]["perfect_ks"]
        temp = [None] * len(uwj_stack)
        for k, prescribed_idx in uwj_prescription.items():
            temp[prescribed_idx] = k
        remainder = [k for k in uwj_stack if not k in temp]
        for i, k_t in enumerate(temp):
            if k_t is None:
                k = remainder.pop()
                temp[i] = k
        uwj_floor.floor[uwj]["perfect_ks"] = temp

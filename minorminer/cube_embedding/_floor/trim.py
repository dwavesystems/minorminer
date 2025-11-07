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
Contains the function which "trims" a floor given the z-path on its tiles;
i.e. removes excessive external paths from the floor.
"""

from collections import Counter, defaultdict

from minorminer.utils.zephyr.node_edge import Edge

from minorminer.cube_embedding._floor.uwj_floor import UWJFloor
from minorminer.cube_embedding._tile import PathInfo
from minorminer._lattice_utils import UWKJ, UWKJZ, ZLatticeSurvey


def trim(
    uwj_floor: UWJFloor,
    path_info: PathInfo,
    lattice_survey: ZLatticeSurvey,
):
    """Iteratively removes excessive external paths from the floor until there is just enough
        external paths to form a given z-path in every tile.
        In removing paths, it gives priority to hard-to-cure external paths.

    Args:
        uwj_floor (UWJFloor): The floor to trim
        path_info (PathInfo): The information of the z-path
            to be constructed on the floor's tiles.
        lattice_survey (ZLatticeSurvey):
            ZLatticeSurvey of the Zephyr graph the floor lies on.
    """

    def pick_uwkj_to_remove(all_cand_uwkj: list[UWKJ]) -> UWKJ:
        """Picks an external path from ``all_cand_uwkj`` to remove from ``uwj_floor_sur``.
            Gives priority to the hard-to-cure ones.

        Args:
            all_cand_uwkj (list[UWKJ]): The list of all external paths than can be removed.

        Returns:
            UWKJ: The external path chosen to be removed. Priority is given to hard-to-cure
                external paths.
        """
        cr_missing_deg_uwkj = {uwkj: len(uwkj_dict) for uwkj, uwkj_dict in cr_missing.items()}
        cr_missing_deg_parent = defaultdict(int)
        for uwkj, deg_uwkj in cr_missing_deg_uwkj.items():
            uwj = uwkj.uwj
            cr_missing_deg_parent[uwj] += deg_uwkj

        imp_missing_deg_uwkj = {uwkj: len(uwkj_dict) for uwkj, uwkj_dict in imp_missing.items()}
        imp_missing_deg_parent = defaultdict(int)
        for uwkj, deg_uwkj in imp_missing_deg_uwkj.items():
            uwj = uwkj.uwj
            imp_missing_deg_parent[uwj] += deg_uwkj

        del_priority = {}
        for uwkj in all_cand_uwkj:
            parent = uwkj.uwj
            first = cr_missing_deg_uwkj.get(uwkj, 0)
            second = imp_missing_deg_uwkj.get(uwkj, 0)
            third = cr_missing_deg_parent.get(parent, 0)
            fourth = imp_missing_deg_parent.get(parent, 0)
            fifth = floor_sur[parent]["surplus"]
            del_priority[uwkj] = (first, second, third, fourth, fifth)
        return max(del_priority, key=lambda uwkj: del_priority[uwkj])

    def remove_uwkj(
        uwkj: UWKJ,
        all_cand_uwkj: list[UWKJ],
        cr_missing: dict[UWKJ, list[UWKJ]],
        imp_missing: dict[UWKJ, list[UWKJ]],
    ):
        """Updates ``uwj_floor_sur``, ``all_cand_uwkj``, ``cr_missing``, and ``imp_missing``
        by removing an external path from them.

        Args:
            uwkj (UWKJ): The external path to be removed.
            all_cand_uwkj (list[UWKJ]): The list of all external paths than can be removed.
            cr_missing (dict[UWKJ, list[UWKJ]]): A dictionary that maps each external path
                to the list of external paths it misses a "cucial" connection with.
            imp_missing (dict[UWKJ, list[UWKJ]]):  A dictionary that maps each external path
                to the list of external paths it misses an "important" connection with.
        """

        def remove_from_adj_dict(
            uwkj: UWKJ, adj_dict: dict[UWKJ, list[UWKJ]]
        ) -> dict[UWKJ, list[UWKJ]]:
            """
            Updates a dictionary of (non-)adjacencies of external paths by removing
                an external path from it.

            Note: Assumes the dictionary of (non-)adjacencies of external paths is symmetric.

            Args:
                uwkj (UWKJ): The external path to remove from the dictionary
                    of (non-)adjacencies of external paths.
                adj_dict (dict[UWKJ, list[UWKJ]]): A symmetric dictionary
                    of (non-)adjacencies of the external paths of the form
                    {uwkj0: [uwkj1, uwkj2, ...], uwkj1: [uwkj0, ...]}
                    which maps each external path to the list of all external paths
                    it has (no) adjacencies with.

            Returns:
                dict[UWKJ, list[UWKJ]]: The updated dictionary of (non-)adjacencies of
                    external paths after removing ``uwkj``.
            """
            uwkj_missing_nbrs = adj_dict[uwkj]
            for missing_nbr in uwkj_missing_nbrs:
                adj_dict[missing_nbr].remove(uwkj)
                if not adj_dict[missing_nbr]:
                    del adj_dict[missing_nbr]
            del adj_dict[uwkj]
            return adj_dict

        # Update uwj_floor_sur[uwj]
        uwj, k = uwkj.uwj, uwkj.k
        uwj_dict = floor_sur[uwj]
        uwj_dict["perfect_ks"].remove(k)
        uwj_dict["surplus"] -= 1

        # Update all_cand_uwkj
        all_cand_uwkj.remove(uwkj)
        if uwj_dict["surplus"] == 0:
            for k in uwj_dict["perfect_ks"]:
                all_cand_uwkj.remove(UWKJ(u=uwj.u, w=uwj.w, k=k, j=uwj.j))

        # Update cr_missing, imp_missing
        if uwkj in cr_missing:
            cr_missing = remove_from_adj_dict(uwkj=uwkj, adj_dict=cr_missing)
        if uwkj in imp_missing:
            imp_missing = remove_from_adj_dict(uwkj=uwkj, adj_dict=imp_missing)

    def get_uwkj_wise(missing_bonds: list[tuple[UWKJZ, UWKJZ]]) -> dict[UWKJ, list[UWKJ]]:
        """Given a list of missing edges, it finds the missing connections between
            the external paths.

        Args:
            missing_bonds (list[tuple[UWKJZ, UWKJZ]]): A list of missing edges.

        Returns:
            dict[UWKJ, list[UWKJ]]: Maps each extrnal path to the list of external paths
            which miss a connection with.
        """
        missing_uwkj = defaultdict(list)
        for uwkjz1, uwkjz2 in missing_bonds:
            uwkj1 = uwkjz1.uwkj
            uwkj2 = uwkjz2.uwkj
            missing_uwkj[uwkj1].append(uwkj2)
            missing_uwkj[uwkj2].append(uwkj1)
        return dict(missing_uwkj)

    counter_node = Counter(x for pair in path_info.path_vh for x in pair)
    crucial_idx_edges: list[Edge] = [a.edge for a in path_info.crucial_edge_pos]
    important_idx_edges: list[Edge] = [a.edge for a in path_info.alt_edge_pos]

    all_missing_edges = uwj_floor.missing_internal_edges(lattice_survey=lattice_survey)
    # "crucial" missing couplers of uwj_floor, i.e. connections necessary for forming chains
    # or the unique connections between the chains
    crucial_missing = [
        xy
        for xy, xy_tile_coord_edge in all_missing_edges.items()
        if xy_tile_coord_edge.edge in crucial_idx_edges
    ]

    # "important" missing couplers of uwj_floor, i.e. connections necessary between the chains with two connections between them
    important_missing = [
        xy
        for xy, xy_tile_coord_edge in all_missing_edges.items()
        if xy_tile_coord_edge.edge in important_idx_edges
    ]

    # a dictionary that maps an external path to the list of all its crossing uwkj's that
    # miss a crucial bond with
    cr_missing = get_uwkj_wise(crucial_missing)

    # a dictionary that maps an external path to the list of all its crossing uwkj's that
    # miss an important bond with
    imp_missing = get_uwkj_wise(important_missing)

    floor_sur = uwj_floor.floor
    for uwj_stats in floor_sur.values():
        uwj_stats["surplus"] = len(uwj_stats["perfect_ks"]) - counter_node[uwj_stats["idx"]]

    all_cand_uwkj = []  # Collect all that could be removed
    for uwj, uwj_dict in floor_sur.items():
        if uwj_dict["surplus"] > 0:
            for k in uwj_dict["perfect_ks"]:
                all_cand_uwkj.append(UWKJ(u=uwj.u, w=uwj.w, k=k, j=uwj.j))

    num_surplus = sum(uwj_dict["surplus"] for uwj_dict in floor_sur.values())
    for _ in range(num_surplus):
        uwkj = pick_uwkj_to_remove(all_cand_uwkj=all_cand_uwkj)
        remove_uwkj(
            uwkj=uwkj, all_cand_uwkj=all_cand_uwkj, cr_missing=cr_missing, imp_missing=imp_missing
        )

    for uwj_dict in floor_sur.values():
        del uwj_dict["surplus"]

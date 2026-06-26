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
Contains helper function and objects to construct paths
within a tile on a perfect-yielded Zephyr, and enable calculating
"crucial edge-positions" and "edge-positions" which have an "altrnative edge-position"
in the path.
"""

from typing import Generator, NamedTuple

from minorminer.utils.zephyr.node_edge import Edge, NodeKind

from minorminer.cube_embedding._tile.kind import ZCoupling

__all__ = ["PathInfo", "VHIdxEdge", "parse_seq"]

class EdgePos(NamedTuple):
    edge: Edge
    pos: tuple[int, int]
    """
    A helper to represent some information of a path of chains of node indices (within a :class:`QuoTile`).
    edge (Edge): Edge of node indices represnting the index-chain
    pos (tuple[int, int]): pair of positions, the i-th element representing the number of occurence of i-th element of the index-chain when visiting the index-chain
    
    Eg: For [(0, 1), (0, 2), (2, 3), (3, 4)] the EdgePos for (0, 2) has chain = (0, 2) and pos = (1, 0)
    0 in (0, 2) is the 1-st occurence of 0 in the path (after its appearance in (0, 1)).
    2 in (0, 2) is the 0-th occurence of 2 in the path
    """


class VHIdxEdge(Edge):
    """
    Initializes an index edge (Edge between two node indices within a :class:`QuoTile`).
    The edge must correspond to an internal edge; 
    i.e. the two indices must have different kinds.
    """
    def __init__(
        self,
        idx1: int,
        idx2: int,
        idx_kind: dict[int, NodeKind],
    ) -> None:
        self._edge = self._set_edge(idx1, idx2, idx_kind)

    def _set_edge(self, idx1: int, idx2: int, idx_kind: dict[int, NodeKind]) -> tuple[int, int]:
        """Returns the tuple corresponding to an edge of two indices,
            where the 0-th index is vertical and the 1-st index is horizontal.

        Args:
            idx1 (int): Index of one endpoint of the edge
            idx2 (int): Index of another endpoint of the edge
            idx_kind (dict[int, NodeKind]): A dictionary mapping each node index to its kind.

        Raises:
            ValueError: if ``idx1`` or ``idx2`` not in ``idx_kind``
            ValueError: if ``idx1`` and ``idx2`` have the same kind.

        Returns:
            tuple[int, int]:  The tuple corresponding to idx1, idx2,
            where the 0-th index is vertical and the 1-st index is horizontal.
        """
        idx1_kind = idx_kind.get(idx1)
        idx2_kind = idx_kind.get(idx2)
        if any(idx is None for idx in [idx1_kind, idx2_kind]):
            raise ValueError(
                f"Expected idx1, idx2 to be in idx_kind, got {(idx1, idx2) = } and {idx_kind = }"
            )
        if idx1_kind is NodeKind.VERTICAL and idx2_kind is NodeKind.HORIZONTAL:
            return (idx1, idx2)
        elif idx2_kind is NodeKind.VERTICAL and idx1_kind is NodeKind.HORIZONTAL:
            return (idx2, idx1)
        raise ValueError(
            f"Expected idx1, idx2 to have different kind in NodeKind, got {(idx1, idx2) = } and {idx_kind = }"
        )


class PathInfo(NamedTuple):
    crucial_edge_pos: list[EdgePos]
    alt_edge_pos: dict[EdgePos, EdgePos]
    path_vh: list[VHIdxEdge]
    """
    A helper object for capturing the information of a path relevant to embedding.
    crucial_edge_pos (list[EdgePos]): To capture all "crucial edge-positions" of a path.
    alt_edge_pos (dict[EdgePos, EdgePos]): To capture the dictionary of the "alternative edge-positions" of a path.
    path_vh (list[VHIdxEdge]): To present the path in a way that for each chain in the path, the
        0-th element corresponds to a vertical node and thus the 1-st element corresponds
        to a horizontal node.

    Note: A "crucial" edge-position ``EdgePos`` is defined as a edge-position
        with edge=(idx0, idx1) and pos=(pos0, pos1) such that the internal coupler
        between idx0 at its occurrence pos0 and idx1 at its occurrence pos1 is
        necessary for making a chain or for making an absolutely
        necessary connection between two chains. In other words, if such a edge-position
        is missing, it cannot be made up for.
        We say a edge-position with edge=(idx0, idx1) and pos=(pos0, pos1) has an
        "alternative" edge-position alt_edge=(a_idx0, a_idx1) and pos=(a_pos0, a_pos1)
        if a potentially missing internal coupler between idx0 at its pos0 occurrence and
        idx1 at its pos1 occurrence can be made up for with the internal coupler between
        a_idx0 at its a_pos1 occurrence and a_idx1 at its a_pos1 occurrence.
    """


def parse_seq(
    seq: list[Edge],
    idx_chains: list[Edge],
    periodic: bool,
    indices: dict[int, NodeKind],
    z_coupling: ZCoupling,
    *kwargs,
) -> Generator[PathInfo, None, None]:
    """Given a sequence of index edges, it generates zero, one or two
        :class:`PathInfo`s , each of which corresponding to
        a valid path respecting periodicity and z-coupling.

    Args:
        seq (list[Edge]): Sequence of index edges within a :class:`QuoTile`.
        idx_chains (list[Edge]): The list of all index edges corresponding to an internal edge of a :class:`QuoTile`.
        periodic (bool):  Whether there necessarily must be an edge from the pair of indices
        forming the last chain in the path to the pair of indices forming the first chain in the path.
        indices (dict[int, NodeKind]): A dictionary mapping each node index to its kind.
        z_coupling (ZCoupling): The kind of coupling between chains
            corresponding to consecutive nodes in a z-paths.

    Yields:
        Generator[PathInfo, None, None]: :class:`PathInfo`s objects, each of which correspond to
        a valid path respecting periodicity and z-coupling.
        
    Note: Not implemented for ``ZCoupling.ONE_ZERO``, as this case is handled
        within the :func:`find_cube_embedding` by swapping with ``ZCoupling.ZERO_ONE``
    """

    def normalized_iep(bond: tuple[int, int], pos: tuple[int, int]) -> EdgePos:
        edge = Edge(*bond)
        if (edge[0], edge[1]) == bond:
            return EdgePos(edge=edge, pos=pos)
        elif (edge[1], edge[0]) == bond:
            return EdgePos(edge=edge, pos=pos[::-1])

    def get_crucial_alt() -> tuple[list[EdgePos, dict[EdgePos, EdgePos]]]:
        if z_coupling is ZCoupling.ZERO_ONE_ONE_ZERO:
            return (crucial_edge_pos + con_01 + con_01_bkwd, {})

        if z_coupling is ZCoupling.EITHER:
            alt_edge_pos: dict[EdgePos, EdgePos] = {}
            for i, ep_01 in enumerate(con_01):
                ep_10 = con_01_bkwd[i]
                if ep_01 is None:
                    crucial_edge_pos.append(ep_10)
                elif ep_10 is None:
                    crucial_edge_pos.append(ep_01)
                else:
                    alt_edge_pos[ep_10] = ep_01
                    alt_edge_pos[ep_01] = ep_10
            return (crucial_edge_pos, alt_edge_pos)

        if z_coupling is ZCoupling.ZERO_ONE:
            return (crucial_edge_pos + con_01, {})

    if z_coupling is ZCoupling.ONE_ZERO:
        raise NotImplementedError(f"parse_seq not implemented for ZCoupling.ONE_ZERO")
    # Potentially may need to travel the path in two directions
    for fwd in (True, False):
        if fwd:
            chance_01_fwd, chance_01_bkwd = True, True
            con_01_bkwd: list[EdgePos] = []
        else:
            # Spare the reverse path travel in unnecessary cases 
            if not (chance_01_bkwd and z_coupling is ZCoupling.ZERO_ONE):
                break
            seq = seq[::-1]

        path_vh = [VHIdxEdge(*e, indices) for e in seq]
        len_path = len(path_vh)
        unique_indices = {x for pair in path_vh for x in pair}

        con_01: list[EdgePos] = []

        # crucial edge-positions
        crucial_edge_pos: list[EdgePos] = []
        index_pos = {x: -1 for x in unique_indices}
        for i, (v, h) in enumerate(path_vh):
            index_pos[v] += 1
            index_pos[h] += 1
            e_vh = seq[i]

            # bonds needed to form chain:
            pos = (index_pos[e_vh[0]], index_pos[e_vh[1]])
            crucial_edge_pos.append(EdgePos(edge=e_vh, pos=pos))

            if i == len_path - 1 and not periodic:
                continue
            # bonds for inter-chain bonds:
            if i < len_path - 1:
                (v_n, h_n) = path_vh[i + 1]
                (pos_v_n, pos_h_n) = (index_pos[v_n] + 1, index_pos[h_n] + 1)
            else:  # periodic
                if i == 1:  # a path of length 2--connections are already considered when i = 0
                    continue
                (v_n, h_n) = path_vh[0]
                (pos_v_n, pos_h_n) = (0, 0)

            first_bond_cand = (v, h_n)
            first_pos = (index_pos[v], pos_h_n)
            first_iep = normalized_iep(bond=first_bond_cand, pos=first_pos)

            if fwd:
                sec_bond_cand = (v_n, h)
                sec_pos = (pos_v_n, index_pos[h])
                sec_iep = normalized_iep(bond=sec_bond_cand, pos=sec_pos)
                first_bond_exists = first_iep.edge in idx_chains
                sec_bond_exists = sec_iep.edge in idx_chains
                if not (first_bond_exists or sec_bond_exists):
                    return
                elif z_coupling is ZCoupling.ZERO_ONE_ONE_ZERO:
                    if not (first_bond_exists and sec_bond_exists):
                        return

                if not first_bond_exists:
                    con_01.append(None)
                    chance_01_fwd = False
                else:
                    con_01.append(first_iep)
                if not sec_bond_exists:
                    con_01_bkwd.append(None)
                    chance_01_bkwd = False
                else:
                    con_01_bkwd.append(sec_iep)
                if not (chance_01_fwd or chance_01_bkwd):
                    if not z_coupling is ZCoupling.EITHER:
                        return
            else:
                con_01.append(first_iep)
        if z_coupling is ZCoupling.ZERO_ONE and fwd and not chance_01_fwd:
            break
        crucial, alt = get_crucial_alt()
        yield PathInfo(crucial_edge_pos=crucial, alt_edge_pos=alt, path_vh=path_vh)

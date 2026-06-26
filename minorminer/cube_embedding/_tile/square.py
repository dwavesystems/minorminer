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
A collection of tools for ``TileKind.SQUARE``
"""

from itertools import permutations
from typing import Generator, Literal

from minorminer.utils.zephyr.node_edge import Edge, NodeKind, ZNode

from minorminer.cube_embedding._tile.chain_supply import generate_chain_supply
from minorminer.cube_embedding._tile.kind import ZCoupling
from minorminer.cube_embedding._tile.z_path import PathInfo, parse_seq
from minorminer._lattice_utils import QuoTile

__all__ = [
    "square_tiles",
    "square_z_paths",
    "square_idx_chain",
]


square_tiles = {
    "main": QuoTile(
        [
            ZNode((0, 1)),
            ZNode((0, 3)),
            ZNode((1, 0)),
            ZNode((1, 2)),
            ZNode((2, 1)),
            ZNode((2, 3)),
            ZNode((3, 0)),
            ZNode((3, 2)),
        ]
    ),
    "anti": QuoTile(
        [
            ZNode((1, 0)),
            ZNode((1, 2)),
            ZNode((2, 1)),
            ZNode((2, 3)),
            ZNode((3, 0)),
            ZNode((3, 2)),
            ZNode((4, 1)),
            ZNode((4, 3)),
        ]
    ),
}


all_square_idx_chains: dict[str, list[Edge]] = {
    "main": [
        Edge(4, 7),
        Edge(2, 4),
        Edge(4, 6),
        Edge(0, 2),
        Edge(3, 5),
        Edge(5, 7),
        Edge(1, 3),
        Edge(0, 3),
        Edge(3, 4),
    ],
    "anti": [
        Edge(2, 5),
        Edge(5, 7),
        Edge(0, 2),
        Edge(2, 4),
        Edge(1, 3),
        Edge(3, 5),
        Edge(5, 6),
        Edge(4, 6),
        Edge(1, 2),
    ],
}

"""A dictionary whose keys are the two subkinds of ``TileKind.SQUARE``--"main", "anti".
    The value is a dictionary whose
    - keys are the Edges corresponding to a pair of node indices within the tile
        which can form a cube chain (i.e. there is an internal coupler between them).
    - values are the ZEdges corresponding to each Edge key.
"""


def square_idx_chain(
    sub_kind: Literal["main", "anti"],
    prescribed: bool = False,
) -> list[Edge]:
    """Returns a dictionary which maps the internal edges of a square tile to the edges
        corresponding to the indices of their nodes within the tile.

    Args:
        sub_kind (Literal[&quot;main&quot;, &quot;anti&quot;]): Sub-kinds of a square tile--"main" or "anti".
        prescribed (bool, optional): whether to limit the ZEdges to only "prescribed" edges. Defaults to False.

    Returns:
        dict[ZEdge, Edge]:  A dictionary whose
        - keys correspond to an internal edge within the tile.
        - values correspond to the node indices (within the tile) of each key.
    """
    pres_chains = [Edge(4, 6), Edge(1, 3), Edge(0, 2), Edge(5, 7)]
    if prescribed:
        return pres_chains
    if sub_kind == "main":
        nonpres_chains = [Edge(0, 3), Edge(2, 4), Edge(3, 4), Edge(3, 5), Edge(4, 7)]
    else:
        nonpres_chains = [Edge(1, 2), Edge(2, 4), Edge(2, 5), Edge(3, 5), Edge(5, 6)]
    return pres_chains + nonpres_chains


def get_num_supply_square(
    initial_supply: dict[int, int],
    num: int,
    prescribed: bool,
    sub_kind: Literal["main", "anti"],
    **kwargs,
) -> list[dict[Edge, int]]:
    """
    Generates all possible chain allocation configurations for a given number of chains to be constructed,
    while respecting initial supply constraints.

    Each chain allocation is a dictionary:
        {Edge(idx1, idx2): count, Edge(idx1, idx3): count, ...}
    indicating how many chains of each type can be constructed.

    Args:
        initial_supply: Maps each index (of a node in a square tile)
            to its available supply count in partially-yielded Zephyr.
        num: Total number of chains to construct.
        prescribed: If True, restricts chain construction to square prescribed chains.
        sub_kind (Literal[&quot;main&quot;, &quot;anti&quot;]): Subkind of ``TileKind.SQUARE``--"main", "anti".
    Returns:
        list[dict[Edge, int]]: A list of dictionaries, each representing a supply allocation for the square chains which has ``num`` chains.
    """
    if sub_kind == "main":
        phase_chains = {
            1: [Edge(0, 2), Edge(1, 3), Edge(4, 6), Edge(5, 7)],
            2: [Edge(3, 4)],
            3: [Edge(2, 4), Edge(3, 5)],
            4: [Edge(0, 3), Edge(4, 7)],
        }
    else:
        phase_chains = {
            1: [Edge(0, 2), Edge(1, 3), Edge(4, 6), Edge(5, 7)],
            2: [Edge(2, 5)],
            3: [Edge(2, 4), Edge(3, 5)],
            4: [Edge(1, 2), Edge(5, 6)],
        }

    # If ``prescribed=True``, only phase-1 chains are allowed.
    num_phases = 1 if prescribed else None
    return generate_chain_supply(
        initial_supply=initial_supply,
        phase_chains=phase_chains,
        num=num,
        num_phases=num_phases,
    )


def prescribed_seq_square(
    sub_kind: Literal["main", "anti"],
    periodic: bool,
    **kwargs,
) -> list[list[Edge]]:
    """Returns a list of "prescribed" sequences for a square tile respecting periodicity.

    Args:
        sub_kind (Literal[&quot;main&quot;, &quot;anti&quot;]): Subkind of ``TileKind.SQUARE``--"main" or "anti".
        periodic (bool): Whether there necessarily must be an edge from the Edge of indices
            forming the last chain in the path to the Edge of indices forming the first chain in the path.

    Returns:
        list[list[Edge]]: list of "prescribed" sequences for a square tile.
    """

    if sub_kind == "main":
        seq0 = [Edge(1, 3), Edge(5, 7), Edge(4, 6), Edge(0, 2)]
    else:
        seq0 = [Edge(0, 2), Edge(4, 6), Edge(5, 7), Edge(1, 3)]

    if periodic:
        return [seq0]
    else:
        return [seq0[i:] + seq0[:i] for i in range(len(seq0))]


def relaxed_seq_square(
    sub_kind: Literal["main", "anti"],
    **kwargs,
) -> list[list[Edge]]:
    """Returns a list of "relaxed" sequences (i.e. non-prescribed).
    Note: Due to computational considerations, the list is non-exhaustive.

    Args:
        sub_kind (Literal[&quot;main&quot;, &quot;anti&quot;]): ubkind of ``TileKind.SQUARE``--"main", "anti".

    Returns:
        list[list[Edge]]: list of "relaxed" sequences for a square tile.
    """

    if sub_kind == "main":
        chunk = {
            0: [Edge(0, 3), Edge(0, 2), Edge(2, 4)],
            1: [Edge(4, 7), Edge(5, 7), Edge(3, 5)],
            2: [Edge(4, 6)],
            3: [Edge(1, 3)],
            4: [Edge(3, 4)],
        }
    else:
        chunk = {
            0: [Edge(3, 5), Edge(1, 3), Edge(1, 2)],
            1: [Edge(2, 4), Edge(4, 6), Edge(5, 6)],
            2: [Edge(2, 5)],
            3: [Edge(0, 2)],
            4: [Edge(5, 7)],
        }
    all_seqs = []

    for perm_14 in permutations(range(1, 5)):
        perm = (0,) + perm_14
        seq = []
        for i in perm:
            seq += chunk[perm[i]]
        all_seqs.append(seq)

    return all_seqs


def square_z_paths(
    initial_supply: dict[int, int],
    sub_kind: Literal["main", "anti"],
    num: int,
    periodic: bool,
    prescribed: bool,
    z_coupling: ZCoupling | None = None,
    indices: dict[int, NodeKind] | None = None,
) -> Generator[PathInfo, None, None]:
    """Generates paths of length ``num`` that can be constructed with the given
    ``initial_supply`` with square schema and sub-kind ``sub_kind`` that satisfy
    ``prescribed``, ``periodic`` and ``z-coupling``.
    Note: Due to computational considerations, the generator yields paths non-exhaustively.

    Args:
        initial_supply (dict[int, int]): Maps each index (of a node in a square tile)
            to its available supply count in partially-yielded Zephyr.
        sub_kind (Literal[&quot;main&quot;, &quot;anti&quot;]): The square sub-kind
        num (int): Length of path.
        periodic (bool): Whether there necessarily must be an edge from the Edge of indices
            forming the last chain in the paths to the Edge of indices
            forming the first chain in the paths.
        prescribed (bool): If True, restricts chains of the path to square prescribed chains.
        z_coupling (ZCoupling | None, optional): The kind of coupling between chains
            corresponding to consecutive nodes in a z-paths. Defaults to None.
        indices (dict[int, NodeKind] | None, optional): A dcitionary mapping a node index to its kind. Defaults to None.

    Yields:
        Generator[PathInfo, None, None]: Path constructed using
        the available chain supply respecting periodicity and z-coupling.
    """
    if z_coupling is None:
        z_coupling = ZCoupling.EITHER

    if indices is None:
        tile = square_tiles[sub_kind]
        indices = {i: node.node_kind for i, node in enumerate(tile.zns)}

    idx_chains = all_square_idx_chains[sub_kind]
    output_seqs = []
    output_paths = []
    for chain_supply in get_num_supply_square(
        initial_supply=initial_supply,
        sub_kind=sub_kind,
        num=num,
        prescribed=prescribed,
    ):

        if prescribed:
            path_get_func = prescribed_seq_square
        else:
            path_get_func = relaxed_seq_square

        for seq in path_get_func(
            sub_kind=sub_kind,
            periodic=periodic,
        ):
            candidate_seq = tuple([e for e in seq for _ in range(chain_supply.get(e, 0))])
            if candidate_seq in output_seqs or len(candidate_seq) != num:
                continue
            output_seqs.append(candidate_seq)
            for path_ in parse_seq(
                seq=candidate_seq,
                idx_chains=idx_chains,
                periodic=periodic,
                indices=indices,
                z_coupling=z_coupling,
            ):
                if path_ in output_paths:
                    continue
                output_paths.append(path_)
                yield path_

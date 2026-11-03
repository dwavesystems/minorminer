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
A collection of tools for ``TileKind.LADDER``
"""

from itertools import product
from typing import Generator, Iterable, Literal

from minorminer.utils.zephyr.node_edge import Edge, NodeKind, ZEdge, ZNode
from minorminer.utils.zephyr.plane_shift import ZPlaneShift

from minorminer.cube_embedding._tile.chain_supply import generate_chain_supply, prune_zeros
from minorminer.cube_embedding._tile.kind import ZCoupling
from minorminer.cube_embedding._tile.z_path import PathInfo, parse_seq
from minorminer._lattice_utils import QuoTile

__all__ = [
    "ladder_tiles",
    "ladder_z_paths",
    "ladder_idx_chain",
]

ladder_atoms = {"main": (ZNode((0, 1)), ZNode((1, 0))), "anti": (ZNode((3, 0)), ZNode((4, 1)))}
ladder_generic_chains: dict[Literal["main", "anti"], set[ZEdge]] = dict()
for sub_kind, atoms_sub_kind in ladder_atoms.items():
    if sub_kind == "main":
        shift_sub_kind = ZPlaneShift(1, 1)
    else:
        shift_sub_kind = ZPlaneShift(-1, 1)
    ladder_generic_chains[sub_kind] = {
        ZEdge(atoms_sub_kind[0] + k * shift_sub_kind, atoms_sub_kind[1] + k * shift_sub_kind)
        for k in range(4)
    }

ladder_tiles = {
    "main": QuoTile(
        [
            ZNode((0, 1)),
            ZNode((1, 0)),
            ZNode((1, 2)),
            ZNode((2, 1)),
            ZNode((2, 3)),
            ZNode((3, 2)),
            ZNode((3, 4)),
            ZNode((4, 3)),
        ]
    ),
    "anti": QuoTile(
        [
            ZNode((0, 3)),
            ZNode((1, 2)),
            ZNode((1, 4)),
            ZNode((2, 1)),
            ZNode((2, 3)),
            ZNode((3, 0)),
            ZNode((3, 2)),
            ZNode((4, 1)),
        ]
    ),
}
"""A dictionary whose keys are the two subkinds of ``TileKind.LADDER``--"main", "anti".
    The value is the ``QuoTile`` corresponding to each subkind."""


all_ladder_idx_chains = {
    "main": [
        Edge(3, 5),
        Edge(5, 7),
        Edge(1, 3),
        Edge(0, 1),
        Edge(2, 4),
        Edge(4, 6),
        Edge(4, 5),
        Edge(0, 2),
        Edge(6, 7),
        Edge(2, 3),
    ],
    "anti": {
        Edge(3, 6),
        Edge(3, 5),
        Edge(2, 4),
        Edge(1, 4),
        Edge(0, 2),
        Edge(4, 6),
        Edge(6, 7),
        Edge(5, 7),
        Edge(0, 1),
        Edge(1, 3),
    },
}
"""A dictionary whose keys are the two subkinds of ``TileKind.LADDER``--"main", "anti".
    The value is a dictionary whose
    - keys are the Edges corresponding to a pair of node indices within the tile
        which can form a cube chain (i.e. there is an internal coupler between them).
    - values are the ZEdges corresponding to each Edge key.
"""


def normalized(chains: Iterable[Edge], sub_kind: Literal["main", "anti"]) -> Iterable[Edge]:
    """Maps the chain of indices of 'main' sub-kind to the corresponding chain of indices of ``subkind`` sub-kind"""
    if sub_kind == "main":
        return chains

    main_to_anti_idx = {6: 0, 7: 2, 4: 1, 5: 4, 2: 3, 3: 6, 0: 5, 1: 7}
    return [Edge(main_to_anti_idx[e[0]], main_to_anti_idx[e[1]]) for e in chains]


def ladder_idx_chain(
    sub_kind: Literal["main", "anti"],
    prescribed: bool = False,
) -> list[Edge]:
    """Returns a dictionary which maps the internal edges of a ladder tile to the edges
        corresponding to the indices of their nodes within the tile.

    Args:
        sub_kind (Literal[&quot;main&quot;, &quot;anti&quot;]): sub-kind of laddertile--"main" or "anti".
        prescribed (bool, optional): whether to limit the ZEdges to only "prescribed" edges. Defaults to False.

    Returns:
        dict[ZEdge, Edge]: A dictionary whose
        - keys correspond to an internal edge within the tile.
        - values correspond to the node indices (within the tile) of each key.
    """
    pres_chains = normalized([Edge(0, 1), Edge(2, 3), Edge(4, 5), Edge(6, 7)], sub_kind=sub_kind)
    if prescribed:
        return pres_chains
    nonpres_chains = normalized(
        [Edge(3, 5), Edge(5, 7), Edge(1, 3), Edge(2, 4), Edge(4, 6)], sub_kind=sub_kind
    )
    return pres_chains + nonpres_chains


def get_num_supply_ladder(
    initial_supply: dict[int, int],
    sub_kind: Literal["main", "anti"],
    num: int,
    prescribed: bool,
) -> list[dict[Edge, int]]:
    """
    Generates all possible chain allocation configurations for a given number of chains to be constructed,
    while respecting initial supply constraints.

    Each chain allocation is a dictionary:
        {Edge(idx1, idx2): count, Edge(idx1, idx3): count, ...}
    indicating how many chains of each type can be constructed.

    Args:
        initial_supply: Maps each index (of a node in a ladder tile)
            to its available supply count in partially-yielded Zephyr.
        sub_kind: Literal["main", "anti"]: The ladder sub-kind
        num: Total number of chains to construct.
        prescribed: If True, restricts chain construction to ladder prescribed chains.

    Returns:
        list[dict[Edge, int]]: A list of dictionaries, each representing a supply allocation
        for the ladder chains with sub-kind ``sub_kind`` which has ``num`` chains.
    """

    phase_chains = {
        1: normalized([Edge(0, 1), Edge(2, 3), Edge(4, 5), Edge(6, 7)], sub_kind=sub_kind),
        2: normalized([Edge(1, 3), Edge(2, 4), Edge(5, 7)], sub_kind=sub_kind),
        3: normalized([Edge(0, 2), Edge(3, 5), Edge(4, 6)], sub_kind=sub_kind),
    }

    # If ``prescribed=True``, only phase-1 chains are allowed.
    num_phases = 1 if prescribed else None
    return generate_chain_supply(
        initial_supply=initial_supply,
        num=num,
        phase_chains=phase_chains,
        num_phases=num_phases,
    )


def prescribed_path_ladder(
    chain_supply: dict[Edge, int],
    sub_kind: Literal["main", "anti"],
    periodic: bool,
    z_coupling: ZCoupling,
    indices: dict[int, NodeKind],
    num: int,
    **kwargs,
) -> Generator[PathInfo, None, None]:
    """Yields a "prescribed" path of given length using the
    available chain supply respecting periodicity and z_coupling.

    Args:
        chain_supply (dict[Edge, int]): A dictionary where
            - a key corresponds to an Edge of indices corresponding to a prescribed chain,
            - a value corresponds to the number the chain can be constructed.
        sub_kind (Literal[&quot;main&quot;, &quot;anti&quot;]): Subkind of ``TileKind.LADDER``--"main" or "anti".
        periodic (bool): Whether there necessarily must be an edge from the Edge of indices
            forming the last chain in the path to the Edge of indices forming the first chain in the path.
        z_coupling (ZCoupling): The kind of coupling between chains corresponding to consecutive nodes in a z-paths.
        indices (dict[int, NodeKind]): A dcitionary mapping a node index to its kind.
        num (int): length of path

    Yields:
        Generator[PathInfo, None, None]: Path constructed using the available chain supply respecting periodicity and z-coupling.
    """

    chains = normalized([Edge(0, 1), Edge(2, 3), Edge(4, 5), Edge(6, 7)], sub_kind=sub_kind)
    chain_supply = prune_zeros(chain_supply)
    candidate_seq = []
    used_chains_indices = []
    for i, c in enumerate(chains):
        if c in chain_supply:
            candidate_seq += [c] * chain_supply[c]
            used_chains_indices.append(i)

    num_indices = len(used_chains_indices)
    middles = used_chains_indices[1:-1] if num_indices > 2 else None

    if not periodic or middles is None:
        c0, c1, c2, c3 = chains
        if c0 and c1 in candidate_seq:
            while c3 in candidate_seq:
                candidate_seq.remove(c3)
        elif c2 and c3 in candidate_seq:
            while c0 in candidate_seq:
                candidate_seq.remove(c0)
        for path in parse_seq(
            seq=candidate_seq,
            idx_chains=all_ladder_idx_chains[sub_kind],
            periodic=periodic,
            indices=indices,
            z_coupling=z_coupling,
        ):
            yield path
        return

    if len(middles) == 1:
        if used_chains_indices[-1] - used_chains_indices[0] == 3:  # is disconnected
            return

    path_forth = []
    path_back = []
    for i in used_chains_indices:
        c = chains[i]
        c_supply = chain_supply[c]
        if i in middles:
            if c_supply < 2:  # no chance can make a periodic path out of it
                return

            path_forth += [c] * (c_supply - 1)
            path_back += [c]
        else:
            path_forth += [c] * c_supply
    candidate_seq = path_forth + (path_back[::-1])
    if len(candidate_seq) != num:
        return
    for path in parse_seq(
        seq=candidate_seq,
        idx_chains=chains,
        periodic=periodic,
        indices=indices,
        z_coupling=z_coupling,
    ):
        yield path


def relaxed_paths_ladder(
    chain_supply: dict[Edge, int],
    sub_kind: Literal["main", "anti"],
    periodic: bool,
    z_coupling: ZCoupling,
    indices: dict[int, NodeKind],
    **kwargs,
) -> Generator[PathInfo, None, None]:
    """Generates "relaxed" paths (i.e. non-prescribed) using the available chain supply respecting periodicity and z-coupling.
    Note: Due to computational considerations, the generator yields paths non-exhaustively.

    Args:
        chain_supply (dict[Edge, int]): A dictionary where
            - a key corresponds to an Edge of indices corresponding to a prescribed chain,
            - a value corresponds to the number the chain can be constructed.
        sub_kind (Literal[&quot;main&quot;, &quot;anti&quot;]): Subkind of ``TileKind.LADDER``--"main" or "anti".
        periodic (bool): Whether there necessarily must be an edge from the Edge of indices
            forming the last chain to the Edge of indices forming the first chain.
        z_coupling (ZCoupling): The kind of coupling between chains corresponding to consecutive nodes in a z-paths.
        indices (dict[int, NodeKind]): A dcitionary mapping a node index to its kind.

    Yields:
        Generator[PathInfo, None, None]: Path constructed using the available chain supply respecting periodicity and z-coupling.
    """
    chain_supply = prune_zeros(chain_supply)
    upstairs_seqs = [
        normalized(
            [Edge(2, 4), Edge(2, 3), Edge(0, 2), Edge(0, 1), Edge(1, 3), Edge(3, 5)],
            sub_kind=sub_kind,
        ),
        normalized(
            [Edge(2, 4), Edge(0, 2), Edge(0, 1), Edge(1, 3), Edge(2, 3), Edge(3, 5)],
            sub_kind=sub_kind,
        ),
    ]
    upstairs_seqs += [path[::-1] for path in upstairs_seqs]
    downstairs_seqs = [
        normalized([Edge(4, 5), Edge(5, 7), Edge(6, 7), Edge(4, 6)], sub_kind=sub_kind),
        normalized([Edge(5, 7), Edge(6, 7), Edge(4, 5), Edge(4, 6)], sub_kind=sub_kind),
        normalized([Edge(4, 5), Edge(4, 6), Edge(6, 7), Edge(5, 7)], sub_kind=sub_kind),
        normalized([Edge(4, 6), Edge(6, 7), Edge(4, 5), Edge(5, 7)], sub_kind=sub_kind),
    ]
    downstairs_seqs += [path[::-1] for path in downstairs_seqs]

    idx_chains = all_ladder_idx_chains[sub_kind]
    output_seqs = []
    for seq1, seq2 in product(upstairs_seqs, downstairs_seqs):
        seq = seq1 + seq2
        relaxed_seq = tuple([edge for edge in seq for _ in range(chain_supply.get(edge, 0))])
        if relaxed_seq in output_seqs:
            continue
        for path in parse_seq(
            seq=relaxed_seq,
            idx_chains=idx_chains,
            periodic=periodic,
            indices=indices,
            z_coupling=z_coupling,
        ):
            yield path


def ladder_z_paths(
    initial_supply: dict[int, int],
    sub_kind: Literal["main", "anti"],
    num: int,
    periodic: bool,
    prescribed: bool,
    z_coupling: ZCoupling | None = None,
    indices: dict[int, NodeKind] | None = None,
) -> Generator[PathInfo, None, None]:
    """Generates paths of length ``num`` that can be constructed with the given
    ``initial_supply`` with ladder schema and sub-kind ``sub_kind`` that satisfy
    ``prescribed``, ``periodic`` and ``z-coupling``.
    Note: Due to computational considerations, the generator yields paths non-exhaustively.

    Args:
        initial_supply (dict[int, int]): Maps each index (of a node in a ladder tile)
            to its available supply count in partially-yielded Zephyr.
        sub_kind (Literal[&quot;main&quot;, &quot;anti&quot;]): The ladder sub-kind
        num (int): Length of path.
        periodic (bool): Whether there necessarily must be an edge from the Edge of indices
            forming the last chain in the paths to the Edge of indices
            forming the first chain in the paths.
        prescribed (bool): If True, restricts chains of the path to ladder prescribed chains.
        z_coupling (ZCoupling | None, optional): The kind of coupling between chains corresponding to consecutive nodes in a z-paths. Defaults to None.
        indices (dict[int, NodeKind] | None, optional): A dcitionary mapping a node index to its kind. Defaults to None.

    Yields:
        Generator[PathInfo, None, None]: Path constructed using
        the available chain supply respecting periodicity and z-coupling.
    """
    if z_coupling is None:
        z_coupling = ZCoupling.EITHER

    if indices is None:
        tile = ladder_tiles[sub_kind]
        indices = {i: node.node_kind for i, node in enumerate(tile.zns)}

    output_paths = []
    for chain_supply in get_num_supply_ladder(
        initial_supply=initial_supply,
        sub_kind=sub_kind,
        num=num,
        prescribed=prescribed,
    ):
        if prescribed:
            path_getters = [prescribed_path_ladder]
        else:
            path_getters = [prescribed_path_ladder, relaxed_paths_ladder]

        for get_path_func in path_getters:
            for path in get_path_func(
                chain_supply=chain_supply,
                sub_kind=sub_kind,
                periodic=periodic,
                z_coupling=z_coupling,
                indices=indices,
                num=num,
            ):
                if len(path.path_vh) != num:
                    continue
                if not path in output_paths:
                    output_paths.append(path)
                    yield path

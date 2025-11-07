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
Contains the cube embedding function.
"""

from itertools import permutations

import dwave_networkx as dnx
import networkx as nx
from dwave.system import DWaveSampler

from minorminer.cube_embedding._floor import embed_on_floor_zpath, provider_floor_zpaths
from minorminer.cube_embedding._tile import TileKind, ZCoupling
from minorminer._lattice_utils import ZLatticeSurvey

__all__ = ["find_cube_embedding"]


def find_cube_embedding(
    G: nx.Graph | DWaveSampler,
    dim: tuple[int, int, int] | None=None,
    z_periodic: bool = False,
    lattice_survey: ZLatticeSurvey | None = None,
    **kwargs,
) -> dict[
    tuple[int, int, int],
    tuple[int, int] | tuple[tuple[int, int, int, int, int], tuple[int, int, int, int, int]],
]:
    """Finds a uniform chain length 2 embedding of cube, with optional periodicity in the z-direction,
        on G which must be a Zephyr graph or DWaveSampler with Zephyr topology.

    Args:
        G (nx.Graph | DWaveSampler): Target graph or sampler to get the embedding on.
            - Must be a dwave_networkx zephyr_graph or ``DWaveSampler`` with Zephyr topology
        dim (tuple[int, int, int], optional): Dimension of cube to be found.
            - If ``None``, it finds the largest embeddable cube with the same x-, y- and z-dimensions. 
            - Defaults to ``None``.
        z_periodic (bool, optional): True if the cube is periodic on z-dimension, and False otherwise.
            - Defaults to ``False``.
        lattice_survey (ZLatticeSurvey | None, optional): ZLatticeSurvey of ``G``.
            - Defaults to None.
            - Note: Passing it saves computation time. 
        
        **kwargs: Optional arguments to specify the type of tiles or paths inside the tiles used for finding embedding.
            - "z_coupling" can take either names of :class:`ZCoupling`.
                - If provided, limits the type of coupling between consecutive z-chains
                    to the specified z-coupling.
                - By default, it returns an embedding with ``ZCoupling.EITHER``.
            - "tile_kind" can take either names of :class:`TileKind`.
                - If provided, limits the tile used for embedding finding to the specified tile kind.
                - By default, it searches over all implemented tile kinds to find an embedding.
            - "prescribed" can be ``True`` or ``False``.
                - If ``True``, it limits the chain construction to the "prescribed" chains for each tile kind.
                - Defaults to ``False``.

    Returns:
        dict[tuple[int, int, int], tuple[int, int] | tuple[tuple[int, int, int, int, int], tuple[int, int, int, int, int]]]:
            A mapping from cube nodes to chains on the G graph.
            - Keys are (x, y, z) coordinates of the cube's nodes.
            - Values are pairs representing the length-2 chains of the cube's node.
                - Each element of the pair is a node of ``G`` which is expressed as
                    an integer if ``G``'s nodes are labelled as "int";
                    it's expressed as a tuple of five integers if the
                    ``G``'s nodes are labelled as "coordinates"
                    (i.e. have Zephyr coordinates).

    Example (1):
    >>> from dwave_networkx import zephyr_graph
    >>> from minorminer.cube_embedding import find_cube_embedding
    >>> Z3 = zephyr_graph(m=3, t=1)
    >>> emb = find_cube_embedding(Z3)
    >>> print(max(emb))
    27
    

    Example (2):
    >>> from dwave_networkx import zephyr_graph
    >>> from minorminer.cube_embedding import find_cube_embedding
    >>> m, t = 6, 4
    >>> Z6 = zephyr_graph(m=m, t=t)
    >>> emb = find_cube_embedding(Z6, dim=(m, m, 4*t), z_periodic=True)
    >>> print(len(emb))
    576
    """

    if dim is None:
        return _find_largest_cube_embedding(G, **kwargs)
    
    try:
        lattice_survey = lattice_survey or ZLatticeSurvey(G=G)
    except ValueError:
        raise NotImplementedError(
            f"Implemented only for a graph or sampler with Zephyr topology, got {G}"
            )

    m, t, label = lattice_survey.m, lattice_survey.t, lattice_survey.label

    # To make the node label consistent with ``G``'s
    # (i.e. "int" or "coordinates")
    if label == "coordinates":
        convert = lambda x: x
    else:
        convert = dnx.zephyr_coordinates(m=m, t=t).zephyr_to_linear

    prescribed = kwargs.get("prescribed", False)

    z_coupling = kwargs.get("z_coupling", ZCoupling.EITHER)
    if z_coupling is ZCoupling.ONE_ZERO: # inner functions are implemented for ZERO_ONE
        order_z = lambda xyz: (xyz[0], xyz[1], (dim[2] - 1 - xyz[2]))
        z_coupling = ZCoupling.ZERO_ONE
    else:
        order_z = lambda xyz: xyz

    tile_kind = kwargs.get("tile_kind")
    if tile_kind:
        if tile_kind is TileKind.LADDER or tile_kind is TileKind.SQUARE:
            # all corner tile kinds to be considered
            all_corner_tiles = (tile_kind,)
        else:
            raise NotImplementedError(
                f"This is currently implemented only for TileKind.LADDER and TileKind.SQUARE tiles"
            )
    else:
        all_corner_tiles = (TileKind.LADDER, TileKind.SQUARE)

    # Additionally attempt finding embedding of cube with dimension (Ly, Lx, Lz),
    # and if not z_periodic (Lz, Ly, Lx), ...
    if z_periodic:
        all_perms = [perm + (2,) for perm in permutations(range(2))]
    else:
        all_perms = list(permutations(range(3)))

    original_dim = dim
    visited_dims = []
    for perm in all_perms:
        perm_dim = tuple(original_dim[i] for i in perm)
        if perm_dim in visited_dims:
            continue
        visited_dims.append(perm_dim)

        x_dim, y_dim, z_dim = perm_dim
        for tile_kind in all_corner_tiles:
            for sub_kind in ["main", "anti"]:
                # for all pairs (floor, path), attempt finding an embedding
                for uwj_floor, path_info in provider_floor_zpaths(
                    tile_kind=tile_kind,
                    sub_kind=sub_kind,
                    Lx=x_dim,
                    Ly=y_dim,
                    Lz=z_dim,
                    lattice_survey=lattice_survey,
                    prescribed=prescribed,
                    periodic=z_periodic,
                    z_coupling=z_coupling,
                ):
                    emb = embed_on_floor_zpath(
                        uwj_floor=uwj_floor,
                        path_info=path_info,
                        lattice_survey=lattice_survey,
                        Lx=x_dim,
                        Ly=y_dim,
                    )
                    if emb:
                        inv_perm = {var: i for i, var in enumerate(perm)}
                        return {
                            order_z(
                                tuple(
                                xyz[inv_perm[i]] for i in range(3)
                            )): (  # to restore the order of dimensions
                                convert(u),
                                convert(v),
                            )  # make the output have the same label as G
                            for xyz, (u, v) in emb.items()
                        }
    return dict()


def _find_largest_cube_embedding(
    G: nx.Graph | DWaveSampler,
    z_periodic: bool = False,
    lattice_survey: ZLatticeSurvey | None = None,
    **kwargs,
) -> dict[
    tuple[int, int, int],
    tuple[int, int] | tuple[tuple[int, int, int, int, int], tuple[int, int, int, int, int]],
]:
    """
        Finds the largest cube with the same dimensions on x-, y-, and z- directions
        that can be embedded with uniform chain length 2 and gives its embedding,
        with optional periodicity in the z-direction, on G which must be
        a Zephyr graph or DWaveSampler with Zephyr topology.

    Args:
        G (nx.Graph | DWaveSampler): Target graph or sampler to get the embedding on.
            - Must be a dwave_networkx zephyr_graph or ``DWaveSampler`` with Zephyr topology
        z_periodic (bool, optional): True if the cube is periodic on z-dimension.
            - Defaults to ``False``.
        lattice_survey (LatticeSurvey | None, optional): LatticeSurvey of ``G``. Defaults to None.

    Returns:
        dict[tuple[int, int, int], tuple[int, int] | tuple[tuple[int, int, int, int, int], tuple[int, int, int, int, int]]]:
            - The largest cube for which an embedding on ``G`` has been found.
            - A mapping from cube nodes to chains on the G graph.
                - Keys are (x, y, z) coordinates of the largest embeddable cube's nodes.
                - Values are pairs representing the length-2 chains of the cube's node.
                    - Each element of the pair is a node of ``G`` which is expressed as
                        an integer if ``G``'s nodes are labelled as "int";
                        it's expressed as a tuple of five integers if the
                        ``G``'s nodes are labelled as "coordinates"
                        (i.e. have Zephyr coordinates).

    """

    lattice_survey = lattice_survey or ZLatticeSurvey(G=G)
    # largest embedding so far
    largest_emb = {}

    # the largest dimension of cube we have an embedding for so far
    largest_dim = 0

    # the smallest dimension of cube we can't get an embedding for so far
    impossible_dim = min(lattice_survey.m, 4 * lattice_survey.t) + 1

    while largest_dim < impossible_dim:
        # check whether we get a cube embedded with the mid-point dimension
        mid = (largest_dim + impossible_dim) // 2
        if mid <= largest_dim:  # an embedding this large is already found
            break
        emb = find_cube_embedding(G=G,
            dim=(mid, mid, mid),
            z_periodic=z_periodic,
            lattice_survey=lattice_survey,
            **kwargs,
        )
        if emb:  # update largest_dim, largest_emb
            largest_dim = mid
            largest_emb = emb
        else:  # update impossible_dim
            impossible_dim = mid

    return largest_emb

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


from __future__ import annotations

from typing import Callable, Iterable, Iterator, NamedTuple

from minorminer.utils.zephyr.coordinate_systems import *
from minorminer.utils.zephyr.node_edge import EdgeKind, ZEdge, ZNode
from minorminer.utils.zephyr.plane_shift import ZPlaneShift
from minorminer.utils.zephyr.qfloor import QuoTile

__all__ = ["QuoTile", "QuoFloor"]


class QuoTile:
    """Initializes a 'QuoTile' instance from a collection of ``ZNode`` objects.

    Args:
        zns (Iterable[ZNode]): The ``ZNode``s the tile contains.

    Example:
    .. code-block:: python
        >>> from minorminer.utils.zephyr.node_edge import ZNode
        >>> from minorminer.utils.zephyr.qfloor import QuoTile
        >>> ccoords = [(k, 1+k) for k in range(4)] + [(1+k, k) for k in range(4)]
        >>> zns = [ZNode(coord=ccoord) for ccoord in ccoords]
        >>> tile = QuoTile(zns)
        >>> print(f"{tile.edges() = }\n{tile.seed = }\n{tile.shifts = }")
    """

    def __init__(
        self,
        zns: Iterable[ZNode],
    ) -> None:
        if len(zns) == 0:
            raise ValueError(f"Expected zns to be non-empty, got {zns}")
        for zn in zns:
            if not zn.is_quo():
                raise ValueError(f"Expected elements of {zns} to be quotient, got {zn}")
        zns_shape = {zn.shape for zn in zns}
        if len(zns_shape) != 1:
            raise ValueError(
                f"Expected all elements of zns to have the same shape, got {zns_shape}"
            )
        temp_zns = sorted(list(set(zns)))
        seed_convert = temp_zns[0].convert_to_z
        zns_result = []
        for zn in temp_zns:
            zn.convert_to_z = seed_convert
            zns_result.append(zn)
        self._zns: list[ZNode] = zns_result

    @property
    def zns(self) -> list[ZNode]:
        """Returns the sorted list of :class:`ZNode`s the tile contains."""
        return self._zns

    def edges(
        self,
        nbr_kind: EdgeKind | Iterable[EdgeKind] | None = None,
        where: Callable[[CartesianCoord | ZephyrCoord], bool] = lambda coord: True,
    ) -> list[ZEdge]:
        """Returns the list of edges of the graph induced on the tile, when restricted by ``nbr_kind`` and ``where``.

        Args:
            nbr_kind (EdgeKind | Iterable[EdgeKind] | None, optional):
                Edge kind filter. Restricts returned edges to those having the given edge kind(s).
                If None, no filtering is applied. Defaults to None.
            where (Callable[[CartesianCoord | ZephyrCoord], bool]):
                A coordinate filter. Applies to ``ccoord`` if ``self.convert_to_z`` is ``False``,
                or to ``zcoord`` if :py:attr:`self.convert_to_z` is ``True``. Defaults to always.

        Returns:
            list[ZEdge]: List of edges of the graph induced on the tile, when restricted by ``nbr_kind`` and ``where``.
        """
        zns = self._zns
        tile_coords = [zn.zcoord for zn in zns] if self.convert_to_z else [zn.ccoord for zn in zns]
        where_tile = lambda coord: where(coord) and coord in tile_coords
        _edges = {
            edge for zn in zns for edge in zn.incident_edges(nbr_kind=nbr_kind, where=where_tile)
        }
        return list(_edges)

    def __len__(self) -> int:
        return len(self._zns)

    def __iter__(self) -> Iterator[ZNode]:
        for zn in self._zns:
            yield zn

    def __getitem__(self, key) -> ZNode:
        return self._zns[key]

    def __hash__(self) -> int:
        return hash(self._zns)

    def __eq__(self, other: QuoTile) -> bool:
        return self._zns == other._zns

    def __add__(self, shift: ZPlaneShift) -> QuoTile:
        return QuoTile(zns=[zn + shift for zn in self._zns])

    def __repr__(self) -> str:
        return f"{type(self).__name__}{self._zns!r}"

    def __str__(self) -> str:
        return f"{type(self).__name__}{self._zns}"



class QuoFloor(NamedTuple):
    tile: QuoTile
    dim: tuple[int, int]
    """A helper object with a tile representing the top left corner tile and 2-d dimension
    """
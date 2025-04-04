# Copyright 2025 D-Wave
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

from collections import defaultdict, namedtuple
from itertools import product
from typing import Callable, Iterable, Iterator

from minorminer.utils.zephyr.coordinate_systems import *
from minorminer.utils.zephyr.node_edge import ZEdge, ZNode, ZShape
from minorminer.utils.zephyr.plane_shift import PlaneShift

Dim = namedtuple("Dim", ["Lx", "Ly"])
UWJ = namedtuple("UWJ", ["u", "w", "j"])
ZSE = namedtuple("ZSE", ["z_start", "z_end"])


class QuoTile:
    """Initializes a 'QuoTile' object with zns.

    Args:
        zns (Iterable[ZNode]): The ZNodes the tile contains

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
        self.zns = zns

    @property
    def zns(self) -> list[ZNode]:
        """Returns the sorted list of ZNodes the tile contains"""
        return self._zns

    @zns.setter
    def zns(self, new_zns: Iterable[ZNode]) -> None:
        """Sets the zns"""
        if not isinstance(new_zns, Iterable):
            raise TypeError(f"Expected {new_zns} to be Iterable[ZNode], got {type(new_zns)}")
        for zn in new_zns:
            if not isinstance(zn, ZNode):
                raise TypeError(f"Expected elements of {new_zns} to be ZNode, got {type(zn)}")
            if not zn.is_quo():
                raise ValueError(f"Expected elements of {new_zns} to be quotient, got {zn}")
        new_zns_shape = {zn.shape for zn in new_zns}
        if len(new_zns_shape) != 1:
            raise ValueError(
                f"Expected all elements of zns to have the same shape, got {new_zns_shape}"
            )
        temp_zns = sorted(list(set(new_zns)))
        if len(temp_zns) == 0:
            return temp_zns
        seed_convert = temp_zns[0].convert_to_z
        result = []
        for zn in temp_zns:
            zn.convert_to_z = seed_convert
            result.append(zn)
        self._zns = result

    @property
    def seed(self) -> ZNode:
        """Returns the smallest ZNode in Zns"""
        return self._zns[0]

    @property
    def shifts(self) -> list[PlaneShift]:
        """Returns the list of shift of each ZNode in zns from seed"""
        seed = self.seed
        return [zn - seed for zn in self._zns]

    @property
    def shape(self) -> ZShape:
        """Returns the shape of the Zephyr graph the tile belongs to."""
        return self.seed.shape

    @property
    def convert_to_z(self) -> bool:
        """Returns the convert_to_z attribute of the ZNodes of the tile"""
        return self.seed.convert_to_z

    @property
    def ver_zns(self) -> list[ZNode]:
        """Returns the list of vertical ZNodes of the tile"""
        return [zn for zn in self._zns if zn.is_vertical()]

    @property
    def hor_zns(self) -> list[ZNode]:
        """Returns the list of horizontal ZNodes of the tile"""
        return [zn for zn in self._zns if zn.is_horizontal()]

    def edges(
        self,
        where: Callable[[CartesianCoord | ZephyrCoord], bool] = lambda coord: True,
        nbr_kind: str | Iterable[str] | None = None,
    ) -> list[ZEdge]:
        """Returns the list of edges of the graph induced on the tile,
        when resticted to nbr_kind and where."""
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

    def __add__(self, shift: PlaneShift) -> QuoTile:
        return QuoTile(zns=[zn + shift for zn in self._zns])

    def __repr__(self) -> str:
        return f"{type(self).__name__}{self._zns!r}"

    def __str__(self) -> str:
        return f"{type(self).__name__}{self._zns}"


class QuoFloor:
    """Initializes QuoFloor object with a corner tile and dimension and
    optional tile connector.

    Args:
        corner_qtile (QuoTile): The tile at the top left corner of desired
        subgrid of Quotient Zephyr graph.
        dim (Dim | tuple[int]): The dimension of floor, i.e.
        the number of columns of floor, the number of rows of floor.
        tile_connector (dict[tuple[int], PlaneShift], optional):
        Determines how to get the tiles (x+1, y) and (x, y+1) from tile (x, y).
        Defaults to tile_connector0.
    Example:
    .. code-block:: python
    >>> from minorminer.utils.zephyr.node_edge import ZNode
    >>> from minorminer.utils.zephyr.qfloor import QuoFloor
    >>> coords = [(k, k+1) for k in range(4)] + [(k+1, k) for k in range(4)]
    >>> zns = [ZNode(coord=c) for c in coords]
    >>> floor = QuoFloor(corner_qtile=zns, dim=(3, 5))
    >>> print(f"{floor.qtile_xy(2, 3) = }")
    floor.qtile_xy(2, 3) = QuoTile[ZNode(CartesianCoord(x=8, y=13, k=None)), ZNode(CartesianCoord(x=9, y=12, k=None)), ZNode(CartesianCoord(x=9, y=14, k=None)), ZNode(CartesianCoord(x=10, y=13, k=None)), ZNode(CartesianCoord(x=10, y=15, k=None)), ZNode(CartesianCoord(x=11, y=14, k=None)), ZNode(CartesianCoord(x=11, y=16, k=None)), ZNode(CartesianCoord(x=12, y=15, k=None))]

    """

    implemented_connectors: tuple[dict[tuple[int], PlaneShift]] = (
        {(1, 0): PlaneShift(4, 0), (0, 1): PlaneShift(0, 4)},
    )
    tile_connector0 = implemented_connectors[0]

    def __init__(
        self,
        corner_qtile: QuoTile | Iterable[ZNode],
        dim: Dim | tuple[int],
        tile_connector: dict[tuple[int], PlaneShift] = tile_connector0,
    ) -> None:
        self.tile_connector = tile_connector
        self.dim = dim
        self.corner_qtile = corner_qtile

    @property
    def tile_connector(self) -> dict[tuple[int], PlaneShift] | None:
        """Returns the tile connector"""
        return self.tile_connector

    @tile_connector.setter
    def tile_connector(self, new_connector: dict[tuple[int], PlaneShift]) -> None:
        """Sets the tile connector"""
        if not isinstance(new_connector, dict):
            raise TypeError(f"Expected tile_connector to be dict, got {type(new_connector)}")
        if not new_connector in self.implemented_connectors:
            raise NotImplementedError(
                f"{new_connector} not implemented. "
                f"Availabale options are {self.implemented_connectors}"
            )
        if any(dir_con not in new_connector.keys() for dir_con in ((1, 0), (0, 1))):
            raise ValueError(
                f"Expected tile_connector to have (1, 0), (0, 1) as keys, got {new_connector}"
            )
        self._tile_connector = new_connector

    def check_dim_corner_qtile_compatibility(
        self,
        dim: Dim,
        corner_qtile: QuoTile,
    ) -> None:
        """Checks whether dimension and corner tile are compatible, i.e.
        whether given the tile connector, the floor can be
        constructed with the provided corner tile and dimensions.
        """
        if any(par is None for par in (dim, corner_qtile, self._tile_connector)):
            return
        ver_step = self._tile_connector[(0, 1)]
        hor_step = self._tile_connector[(1, 0)]
        try:
            [h + hor_step * (dim.Lx - 1) for h in corner_qtile.hor_zns]
            [v + ver_step * (dim.Ly - 1) for v in corner_qtile.ver_zns]
        except (ValueError, TypeError):
            raise ValueError(f"{dim, corner_qtile} are not compatible")

    @property
    def dim(self) -> Dim:
        """Returns dimension of the floor"""
        return self._dim

    @dim.setter
    def dim(self, new_dim: Dim | tuple[int]) -> None:
        """Sets dimension of the floor"""
        if isinstance(new_dim, tuple):
            new_dim = Dim(*new_dim)
        if not isinstance(new_dim, Dim):
            raise TypeError(f"Expected dim to be Dim or tuple[int], got {type(new_dim)}")
        if not all(isinstance(x, int) for x in new_dim):
            raise TypeError(f"Expected Dim elements to be int, got {new_dim}")
        if any(x <= 0 for x in new_dim):
            raise ValueError(f"Expected elements of Dim to be positive integers, got {Dim}")
        if hasattr(self, "_corner_qtile"):
            self.check_dim_corner_qtile_compatibility(dim=new_dim, corner_qtile=self._corner_qtile)
        self._dim = new_dim

    @property
    def corner_qtile(self) -> QuoTile:
        """Returns the corner tile of floor"""
        return self._corner_qtile

    @corner_qtile.setter
    def corner_qtile(self, new_qtile: QuoTile) -> None:
        """Sets corner tile of the floor"""
        if isinstance(new_qtile, Iterable):
            new_qtile = QuoTile(zns=new_qtile)
        if not isinstance(new_qtile, QuoTile):
            raise TypeError(f"Expected corner_qtile to be QuoTile, got {type(new_qtile)}")
        ccoords = [zn.ccoord for zn in new_qtile.zns]
        x_values = defaultdict(list)
        y_values = defaultdict(list)
        for x, y, *_ in ccoords:
            x_values[x].append(y)
            y_values[y].append(x)
        x_diff = {x: max(x_list) - min(x_list) for x, x_list in x_values.items()}
        y_diff = {y: max(y_list) - min(y_list) for y, y_list in y_values.items()}
        x_jump = self._tile_connector[(1, 0)].x
        y_jump = self._tile_connector[(0, 1)].y

        for y, xdiff in y_diff.items():
            if abs(xdiff) >= y_jump:
                raise ValueError(f"This tile may overlap other tiles on {y = }")
        for x, ydiff in x_diff.items():
            if abs(ydiff) >= x_jump:
                raise ValueError(f"This tile may overlap other tiles on {x = }")
        if hasattr(self, "_dim"):
            self.check_dim_corner_qtile_compatibility(dim=self._dim, corner_qtile=new_qtile)
        self._corner_qtile = new_qtile

    def qtile_xy(self, x: int, y: int) -> QuoTile:
        """Returns the tile at the position x, y of floor, i.e. column x, row y."""
        if (x, y) not in product(range(self._dim.Lx), range(self._dim.Ly)):
            raise ValueError(
                f"Expected x to be in {range(self._dim.Lx)} and y to be in {range(self._dim.Ly)}. "
                f"Got {x, y}."
            )
        xy_shift = x * self._tile_connector[(1, 0)] + y * self._tile_connector[(0, 1)]
        return self._corner_qtile + xy_shift

    @property
    def qtiles(self) -> dict[tuple[int], QuoTile]:
        """Returns the dictionary where the keys are positions of floor,
        and the values are the tiles corresponding to the position.
        """
        if any(par is None for par in (self._dim, self._corner_qtile, self._tile_connector)):
            raise AttributeError(
                f"Cannot access 'qtiles' because either 'dim', 'corner_qtile', or 'tile_connector' is None."
            )
        return {
            (x, y): self.qtile_xy(x=x, y=y)
            for (x, y) in product(range(self._dim.Lx), range(self._dim.Ly))
        }

    @property
    def zns(self) -> dict[tuple[int], list[ZNode]]:
        """Returns the dictionary where the keys are positions of floor,
        and the values are the ZNodes the tile corresponding to the position
        contains."""
        return {xy: xy_tile.zns for xy, xy_tile in self.qtiles.items()}

    @property
    def ver_zns(self) -> dict[tuple[int], list[ZNode]]:
        """Returns the dictionary where the keys are positions of floor,
        and the values are the vertical ZNodes the tile corresponding to
        the position contains."""
        return {xy: xy_tile.ver_zns for xy, xy_tile in self.qtiles.items()}

    @property
    def hor_zns(self) -> dict[tuple[int], list[ZNode]]:
        """Returns the dictionary where the keys are positions of floor,
        and the values are the horizontal ZNodes the tile corresponding to
        the position contains."""
        return {xy: xy_tile.hor_zns for xy, xy_tile in self.qtiles.items()}

    @property
    def quo_ext_paths(self) -> dict[str, dict[int, tuple[UWJ, ZSE]]]:
        """
        Returns {"col": {col_num: (UWJ, SE)}, "row": {hor_num: (UWJ, SE)}}
        of the quo-external-paths
        necessary to be covered from z_start to z_end.
        """
        if any(par is None for par in (self._dim, self._corner_qtile, self._tile_connector)):
            raise AttributeError(
                f"Cannot access 'quo_ext_paths' because either 'dim', 'corner_qtile', or 'tile_connector' is None."
            )
        result = {"col": defaultdict(list), "row": defaultdict(list)}
        hor_con = self._tile_connector[(1, 0)]
        ver_con = self._tile_connector[(0, 1)]
        if hor_con == PlaneShift(4, 0):
            for row_num in range(self._dim.Ly):
                for hzn in self.qtile_xy(0, row_num).hor_zns:
                    hzn_z = hzn.zcoord
                    result["row"][row_num].append(
                        (
                            UWJ(u=hzn_z.u, w=hzn_z.w, j=hzn_z.j),
                            ZSE(z_start=hzn_z.z, z_end=hzn_z.z + self._dim.Lx - 1),
                        )
                    )
        elif hor_con == PlaneShift(2, 0):
            result["row"] = dict()
        else:
            raise NotImplementedError
        if ver_con == PlaneShift(0, 4):
            for col_num in range(self._dim.Lx):
                for vzn in self.qtile_xy(col_num, 0).ver_zns:
                    vzn_z = vzn.zcoord
                    result["col"][col_num].append(
                        (
                            UWJ(u=vzn_z.u, w=vzn_z.w, j=vzn_z.j),
                            ZSE(z_start=vzn_z.z, z_end=vzn_z.z + self._dim.Ly - 1),
                        )
                    )
        elif ver_con == PlaneShift(0, 2):
            result["col"] = dict()
        else:
            raise NotImplementedError
        return {direction: dict(dir_dict) for direction, dir_dict in result.items()}

    def __repr__(self) -> str:
        if self._tile_connector == self.tile_connector0:
            tile_connector_str = ")"
        else:
            tile_connector_str = ", {self._tile_connector!r})"
        return f"{type(self).__name__}({self._corner_qtile!r}, {self._dim!r}" + tile_connector_str

    def __str__(self) -> str:
        if self._tile_connector == self.tile_connector0:
            tile_connector_str = ")"
        else:
            tile_connector_str = ", {self._tile_connector})"
        return f"{type(self).__name__}({self._corner_qtile}, {self._dim})" + tile_connector_str

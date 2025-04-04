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

from collections import namedtuple
from itertools import product
from typing import Callable, Generator, Iterable

from minorminer.utils.zephyr.coordinate_systems import (
    CartesianCoord,
    ZephyrCoord,
    cartesian_to_zephyr,
    zephyr_to_cartesian,
)
from minorminer.utils.zephyr.plane_shift import PlaneShift

ZShape = namedtuple("ZShape", ["m", "t"], defaults=(None, None))


class Edge:
    """Initializes an Edge with nodes x, y.

    Args:
        x : One endpoint of edge.
        y : Another endpoint of edge.
    """

    def __init__(
        self,
        x,
        y,
    ) -> None:
        self._edge = self._set_edge(x, y)

    def _set_edge(self, x, y):
        if x < y:
            return (x, y)
        else:
            return (y, x)

    def __hash__(self):
        return hash(self._edge)

    def __getitem__(self, index: int) -> int:
        return self._edge[index]

    def __eq__(self, other: Edge):
        return self._edge == other._edge

    def __str__(self) -> str:
        return f"{type(self).__name__}{self._edge}"

    def __repr__(self) -> str:
        return f"{type(self).__name__}{self._edge}"


class ZEdge(Edge):
    """Initializes a ZEdge with 'ZNode' nodes x, y

    Args:
        x (ZNode): One endpoint of edge.
        y (ZNode): Another endpoint of edge.
        check_edge_valid (bool, optional): Flag to whether check the validity of values and types of x, y.
        Defaults to True.

    Raises:
        TypeError: If either of x or y is not 'ZNode'.
        ValueError: If x, y do not have the same shape.
        ValueError: If x, y are not neighbours in a perfect yield (quotient)
        Zephyr graph.

    Example 1:
    >>> from zephyr_utils.node_edge import ZNode, ZEdge
    >>> e = ZEdge(ZNode((3, 2)), ZNode((7, 2)))
    >>> print(e)
    ZEdge(ZNode(CartesianCoord(x=3, y=2, k=None)), ZNode(CartesianCoord(x=7, y=2, k=None)))
    Example 2:
    >>> from zephyr_utils.node_edge import ZNode, ZEdge
    >>> ZEdge(ZNode((2, 3)), ZNode((6, 3))) # raises error, since the two are not neighbors
    """

    def __init__(
        self,
        x: ZNode,
        y: ZNode,
        check_edge_valid: bool = True,
    ) -> None:
        if check_edge_valid:
            if not isinstance(x, ZNode) or not isinstance(y, ZNode):
                raise TypeError(f"Expected x, y to be ZNode, got {type(x), type(y)}")
            if x.shape != y.shape:
                raise ValueError(f"Expected x, y to have the same shape, got {x.shape, y.shape}")
            kind_found = False
            for kind in ("internal", "external", "odd"):
                if x.is_neighbor(y, nbr_kind=kind):
                    kind_found = True
                    break
            if not kind_found:
                raise ValueError(f"Expected x, y to be neighbours, got {x, y}")

        self._edge = self._set_edge(x, y)


class ZNode:
    """Initializes 'ZNode' with coord and optional shape.

    Args:
        coord (CartesianCoord | ZephyrCoord | tuple[int]): coordinate in (quotient) Zephyr or (quotient) Cartesian
        shape (ZShape | tuple[int | None] | None, optional): shape of Zephyr graph containing ZNode.
        m: grid size, t: tile size
        Defaults to None.
        convert_to_z (bool | None, optional): Whether to express the coordinates in ZephyrCoordinates.
        Defaults to None.

    Note: If the given coord has non-None k value (in either Cartesian or Zephyr coordinates),
        shape = None raises ValueError. In this case the tile size of Zephyr, t,
        must be provided.

    Example:
    >>> from zephyr_utils.node_edge import ZNode, ZShape
    >>> zn1 = ZNode((5, 2), ZShape(m=5))
    >>> zn1.neighbors()
    [ZNode(CartesianCoord(x=4, y=1, k=None), shape=ZShape(m=5, t=None)),
        ZNode(CartesianCoord(x=4, y=3, k=None), shape=ZShape(m=5, t=None)),
        ZNode(CartesianCoord(x=6, y=1, k=None), shape=ZShape(m=5, t=None)),
        ZNode(CartesianCoord(x=6, y=3, k=None), shape=ZShape(m=5, t=None)),
        ZNode(CartesianCoord(x=1, y=2, k=None), shape=ZShape(m=5, t=None)),
        ZNode(CartesianCoord(x=9, y=2, k=None), shape=ZShape(m=5, t=None)),
        ZNode(CartesianCoord(x=3, y=2, k=None), shape=ZShape(m=5, t=None)),
        ZNode(CartesianCoord(x=7, y=2, k=None), shape=ZShape(m=5, t=None))]
    >>> from zephyr_utils.node_edge import ZNode, ZShape
    >>> zn1 = ZNode((5, 2), ZShape(m=5))
    >>> zn1.neighbors(nbr_kind="odd")
    [ZNode(CartesianCoord(x=3, y=2, k=None), shape=ZShape(m=5, t=None)),
        ZNode(CartesianCoord(x=7, y=2, k=None), shape=ZShape(m=5, t=None))]
    """

    def __init__(
        self,
        coord: CartesianCoord | ZephyrCoord | tuple[int],
        shape: ZShape | tuple[int | None] | None = None,
        convert_to_z: bool | None = None,
    ) -> None:
        self.shape = shape

        if convert_to_z is None:
            self.convert_to_z = len(coord) in (4, 5)
        else:
            self.convert_to_z = convert_to_z

        # convert coord to CartesianCoord or ZephyrCoord
        if not isinstance(coord, (CartesianCoord, ZephyrCoord)):
            coord = self.get_coord(coord)

        # convert coord to CartesianCoord
        if isinstance(coord, ZephyrCoord):
            coord = zephyr_to_cartesian(coord)

        self.ccoord = coord

    @property
    def shape(self) -> ZShape | None:
        """Returns the shape of the Zephyr graph ZNode belongs to."""
        return self._shape

    @shape.setter
    def shape(self, new_shape) -> None:
        """Sets a new value for shape"""
        if new_shape is None:
            new_shape = ZShape()
        elif isinstance(new_shape, tuple):
            new_shape = ZShape(*new_shape)
        if not isinstance(new_shape, ZShape):
            raise TypeError(
                f"Expected shape to be tuple[int | None] or ZShape or None, got {type(new_shape)}"
            )
        if hasattr(self, "_ccoord"):
            if (self._ccoord.k is None) != (new_shape.t is None):
                raise ValueError(
                    f"ccoord, shape must be both quotient or non-quotient, got {self._ccoord, new_shape}"
                )
        for var, val in {"m": new_shape.m, "t": new_shape.t}.items():
            if (val is not None) and (not isinstance(val, int)):
                raise TypeError(f"Expected {var} to be None or 'int', got {type(val)}")
        self._shape = new_shape

    @property
    def ccoord(self) -> CartesianCoord:
        """Returns the CartesianCoord of self, ccoord"""
        return self._ccoord

    @ccoord.setter
    def ccoord(self, new_ccoord: CartesianCoord | tuple[int]):
        """Sets a new value for ccoord"""
        if isinstance(new_ccoord, tuple):
            new_ccoord = CartesianCoord(*new_ccoord)
        if not isinstance(new_ccoord, CartesianCoord):
            raise TypeError(
                f"Expected ccoord to be CartesianCoord or tuple[int], got {type(new_ccoord)}"
            )
        for c in (new_ccoord.x, new_ccoord.y):
            if not isinstance(c, int):
                raise TypeError(f"Expected ccoord.x and ccoord.y to be 'int', got {type(c)}")
            if c < 0:
                raise ValueError(f"Expected ccoord.x and ccoord.y to be non-negative, got {c}")
        if new_ccoord.x % 2 == new_ccoord.y % 2:
            raise ValueError(
                f"Expected ccoord.x and ccoord.y to differ in parity, got {new_ccoord.x, new_ccoord.y}"
            )
        # check k value of CartesianCoord is consistent with t
        if hasattr(self, "_shape"):
            if (self._shape.t is None) != (new_ccoord.k is None):
                raise ValueError(
                    f"shape and ccoord must be both quotient or non-quotient, got {self._shape}, {new_ccoord}"
                )
            if (self._shape.t is not None) and (new_ccoord.k not in range(self._shape.t)):
                raise ValueError(f"Expected k to be in {range(self._shape.t)}, got {new_ccoord.k}")

            # check x, y value of CartesianCoord is consistent with m
            if self._shape.m is not None:
                if all(val in range(4 * self._shape.m + 1) for val in (new_ccoord.x, new_ccoord.y)):
                    self._ccoord = new_ccoord
                else:
                    raise ValueError(
                        f"Expected ccoord.x and ccoord.y to be in {range(4*self._shape.m+1)}, got {new_ccoord.x, new_ccoord.y}"
                    )
        self._ccoord = new_ccoord

    @staticmethod
    def get_coord(coord: tuple[int]) -> CartesianCoord | ZephyrCoord:
        """Takes a tuple[int] and returns the corresponding CartesianCoord
        or ZephyrCoord"""
        if (not isinstance(coord, tuple)) or (not all(isinstance(c, int) for c in coord)):
            raise TypeError(f"Expected {coord} to be a tuple[int], got {coord}")
        if any(c < 0 for c in coord):
            raise ValueError(f"Expected elements of coord to be non-negative, got {coord}")
        len_coord = len(coord)
        if len_coord in (2, 3):
            x, y, *k = coord
            if x % 2 == y % 2:
                raise ValueError(f"Expected x, y to differ in parity, got {x, y}")
            return CartesianCoord(x=x, y=y) if len(k) == 0 else CartesianCoord(x=x, y=y, k=k[0])
        if len_coord in (4, 5):
            u, w, *k, j, z = coord
            for var, val in {"u": u, "j": j}.items():
                if not val in [0, 1]:
                    raise ValueError(f"Expected {var} to be in [0, 1], got {val}")
            return (
                ZephyrCoord(u=u, w=w, j=j, z=z)
                if len(k) == 0
                else ZephyrCoord(u=u, w=w, k=k[0], j=j, z=z)
            )
        raise ValueError(f"coord can have length 2, 3, 4 or 5, got {len_coord}")

    @property
    def zcoord(self) -> ZephyrCoord:
        """Returns ZephyrCoordinate corresponding to ccoord"""
        return cartesian_to_zephyr(self._ccoord)

    def is_quo(self) -> bool:
        """Decides if the ZNode object is quotient"""
        return (self._ccoord.k is None) and (self._shape.t is None)

    def to_quo(self) -> ZNode:
        """Returns the quotient ZNode corresponding to self"""
        qshape = ZShape(m=self._shape.m)
        qccoord = CartesianCoord(x=self._ccoord, y=self._ccoord)
        return ZNode(coord=qccoord, shape=qshape, convert_to_z=self.convert_to_z)

    @property
    def direction(self) -> int:
        """Returns direction, 0 or 1"""
        return self._ccoord.x % 2

    def is_vertical(self) -> bool:
        """Decides whether self is a vertical qubit"""
        return self.direction == 0

    def is_horizontal(self) -> bool:
        """Decides whether self is a horizontal qubit"""
        return self.direction == 1

    def node_kind(self) -> str:
        """Returns the node kind, vertical or horizontal"""
        return "vertical" if self.is_vertical() else "horizontal"

    def neighbor_kind(
        self,
        other: ZNode,
    ) -> str | None:
        """Returns the kind of coupler between self and other,
        'internal', 'external', 'odd', or None."""
        if not isinstance(other, ZNode):
            other = ZNode(other)
        if self._shape != other._shape:
            return
        coord1 = self._ccoord
        coord2 = other._ccoord
        x1, y1 = coord1.x, coord1.y
        x2, y2 = coord2.x, coord2.y
        if abs(x1 - x2) == abs(y1 - y2) == 1:
            return "internal"
        if x1 % 2 != x2 % 2:
            return
        if coord1.k != coord2.k:  # odd, external neighbors only on the same k
            return
        if self.is_vertical():  # self vertical
            if x1 != x2:  # odd, external neighbors only on the same vertical lines
                return
            diff_y = abs(y1 - y2)
            return "odd" if diff_y == 2 else "external" if diff_y == 4 else None
        else:
            if y1 != y2:  # odd, external neighbors only on the same horizontal lines
                return
            diff_x = abs(x1 - x2)
            return "odd" if diff_x == 2 else "external" if diff_x == 4 else None

    def internal_neighbors_generator(
        self,
        where: Callable[[CartesianCoord | ZephyrCoord], bool] = lambda coord: True,
    ) -> Generator[ZNode]:
        """Generator of internal neighbors"""
        x, y, _ = self._ccoord
        convert = self.convert_to_z
        k_vals = [None] if self._shape.t is None else range(self._shape.t)
        for i, j, k in product((-1, 1), (-1, 1), k_vals):
            ccoord = CartesianCoord(x=x + i, y=y + j, k=k)
            coord = ccoord if not convert else cartesian_to_zephyr(ccoord)
            if not where(coord):
                continue
            try:
                yield ZNode(
                    coord=ccoord,
                    shape=self._shape,
                    convert_to_z=convert,
                )
            except GeneratorExit:
                raise
            except Exception:
                pass

    def external_neighbors_generator(
        self,
        where: Callable[[CartesianCoord | ZephyrCoord], bool] = lambda coord: True,
    ) -> Generator[ZNode]:
        """Generator of external neighbors"""
        x, y, k = self._ccoord
        convert = self.convert_to_z
        changing_index = 1 if x % 2 == 0 else 0
        for s in [-4, 4]:
            new_x = x + s if changing_index == 0 else x
            new_y = y + s if changing_index == 1 else y
            ccoord = CartesianCoord(x=new_x, y=new_y, k=k)
            coord = ccoord if not convert else cartesian_to_zephyr(ccoord)
            if not where(coord):
                continue
            try:
                yield ZNode(
                    coord=ccoord,
                    shape=self._shape,
                    convert_to_z=convert,
                )
            except GeneratorExit:
                raise
            except Exception:
                pass

    def odd_neighbors_generator(
        self,
        where: Callable[[CartesianCoord | ZephyrCoord], bool] = lambda coord: True,
    ) -> Generator[ZNode]:
        """Generator of odd neighbors"""
        x, y, k = self._ccoord
        convert = self.convert_to_z
        changing_index = 1 if x % 2 == 0 else 0
        for s in [-2, 2]:
            new_x = x + s if changing_index == 0 else x
            new_y = y + s if changing_index == 1 else y
            ccoord = CartesianCoord(x=new_x, y=new_y, k=k)
            coord = ccoord if not convert else cartesian_to_zephyr(ccoord)
            if not where(coord):
                continue
            try:
                yield ZNode(
                    coord=ccoord,
                    shape=self._shape,
                    convert_to_z=convert,
                )
            except GeneratorExit:
                raise
            except Exception:
                pass

    def neighbors_generator(
        self,
        nbr_kind: str | Iterable[str] | None = None,
        where: Callable[[CartesianCoord | ZephyrCoord], bool] = lambda coord: True,
    ) -> Generator[ZNode]:
        if nbr_kind is None:
            kinds = {"internal", "external", "odd"}
        else:
            if isinstance(nbr_kind, str):
                kinds = {nbr_kind}
            elif isinstance(nbr_kind, Iterable):
                kinds = set(nbr_kind)
            else:
                raise TypeError(f"Expected 'str' or Iterable[str] or None for nbr_kind")
            kinds = kinds.intersection({"internal", "external", "odd"})
        if "internal" in kinds:
            for cc in self.internal_neighbors_generator(where=where):
                yield cc
        if "external" in kinds:
            for cc in self.external_neighbors_generator(where=where):
                yield cc
        if "odd" in kinds:
            for cc in self.odd_neighbors_generator(where=where):
                yield cc

    def neighbors(
        self,
        nbr_kind: str | Iterable[str] | None = None,
        where: Callable[[CartesianCoord | ZephyrCoord], bool] = lambda coord: True,
    ) -> set[ZNode]:
        """Returns neighbors when restricted to nbr_kind and where"""
        return set(self.neighbors_generator(nbr_kind=nbr_kind, where=where))

    def internal_neighbors(
        self,
        where: Callable[[CartesianCoord | ZephyrCoord], bool] = lambda coord: True,
    ) -> set[ZNode]:
        """Returns internal neighbors when restricted to where"""
        return set(self.neighbors_generator(nbr_kind="internal", where=where))

    def external_neighbors(
        self,
        where: Callable[[CartesianCoord | ZephyrCoord], bool] = lambda coord: True,
    ) -> set[ZNode]:
        """Returns external neighbors when restricted to where"""
        return set(self.neighbors_generator(nbr_kind="external", where=where))

    def odd_neighbors(
        self,
        where: Callable[[CartesianCoord | ZephyrCoord], bool] = lambda coord: True,
    ) -> set[ZNode]:
        """Returns odd neighbors when restricted to where"""
        return set(self.neighbors_generator(nbr_kind="odd", where=where))

    def is_internal_neighbor(
        self,
        other: ZNode,
        where: Callable[[CartesianCoord | ZephyrCoord], bool] = lambda coord: True,
    ) -> bool:
        """Tells if another ZNode is an internal neighbor when restricted to where"""
        if not isinstance(other, ZNode):
            raise TypeError(f"Expected {other} to be ZNode, got {type(other)}")
        for nbr in self.internal_neighbors_generator(where=where):
            if other == nbr:
                return True
        return False

    def is_external_neighbor(
        self,
        other: ZNode,
        where: Callable[[CartesianCoord | ZephyrCoord], bool] = lambda coord: True,
    ) -> bool:
        """Tells if another ZNode is an external neighbor when restricted to where"""
        if not isinstance(other, ZNode):
            raise TypeError(f"Expected {other} to be ZNode, got {type(other)}")
        for nbr in self.external_neighbors_generator(where=where):
            if other == nbr:
                return True
        return False

    def is_odd_neighbor(
        self,
        other: ZNode,
        where: Callable[[CartesianCoord | ZephyrCoord], bool] = lambda coord: True,
    ) -> bool:
        """Tells if another ZNode is an odd neighbor when restricted to where"""
        if not isinstance(other, ZNode):
            raise TypeError(f"Expected {other} to be ZNode, got {type(other)}")
        for nbr in self.odd_neighbors_generator(where=where):
            if other == nbr:
                return True
        return False

    def is_neighbor(
        self,
        other: ZNode,
        nbr_kind: str | Iterable[str] | None = None,
        where: Callable[[CartesianCoord | ZephyrCoord], bool] = lambda coord: True,
    ) -> bool:
        """Tells if another ZNode is a neighbor when restricted to nbr_kind and where"""
        if not isinstance(other, ZNode):
            raise TypeError(f"Expected {other} to be ZNode, got {type(other)}")
        for nbr in self.neighbors_generator(nbr_kind=nbr_kind, where=where):
            if other == nbr:
                return True
        return False

    def incident_edges(
        self,
        nbr_kind: str | Iterable[str] | None = None,
        where: Callable[[CartesianCoord | ZephyrCoord], bool] = lambda coord: True,
    ) -> list[ZEdge]:
        """Returns incident edges when restricted to nbr_kind and where"""
        return [ZEdge(self, v) for v in self.neighbors(nbr_kind=nbr_kind, where=where)]

    def degree(
        self,
        nbr_kind: str | Iterable[str] | None = None,
        where: Callable[[CartesianCoord | ZephyrCoord], bool] = lambda coord: True,
    ) -> int:
        """Returns degree when restricted to nbr_kind and where"""
        return len(self.neighbors(nbr_kind=nbr_kind, where=where))

    def __eq__(self, other: ZNode) -> bool:
        if not isinstance(other, ZNode):
            raise TypeError(f"Expected {other} to be {type(self).__name__}, got {type(other)}")
        if self._shape != other._shape:
            raise ValueError(
                f"Expected {self, other} to have the same shape, got {self._shape, other._shape}"
            )
        return self._ccoord == other._ccoord

    def __ne__(self, other: ZNode) -> bool:
        if not isinstance(other, ZNode):
            raise TypeError(f"Expected {other} to be {type(self).__name__}, got {type(other)}")
        if self._shape != other._shape:
            raise ValueError(
                f"Expected {self, other} to have the same shape, got {self._shape, other._shape}"
            )
        return self._ccoord != other._ccoord

    def __gt__(self, other: ZNode) -> bool:
        if not isinstance(other, ZNode):
            raise TypeError(f"Expected {other} to be {type(self).__name__}, got {type(other)}")
        if self._shape != other._shape:
            raise ValueError(
                f"Expected {self, other} to have the same shape, got {self._shape, other._shape}"
            )
        return self._ccoord > other._ccoord

    def __ge__(self, other: ZNode) -> bool:
        if not isinstance(other, ZNode):
            raise TypeError(f"Expected {other} to be {type(self).__name__}, got {type(other)}")
        if self._shape != other._shape:
            raise ValueError(
                f"Expected {self, other} to have the same shape, got {self._shape, other._shape}"
            )
        return self._ccoord >= other._ccoord

    def __lt__(self, other: ZNode) -> bool:
        if not isinstance(other, ZNode):
            raise TypeError(f"Expected {other} to be {type(self).__name__}, got {type(other)}")
        if self._shape != other._shape:
            raise ValueError(
                f"Expected {self, other} to have the same shape, got {self._shape, other._shape}"
            )
        return self._ccoord < other._ccoord

    def __le__(self, other: ZNode) -> bool:
        if not isinstance(other, ZNode):
            raise TypeError(f"Expected {other} to be {type(self).__name__}, got {type(other)}")
        if self._shape != other._shape:
            raise ValueError(
                f"Expected {self, other} to have the same shape, got {self._shape, other._shape}"
            )
        return self._ccoord <= other._ccoord

    def __add__(
        self,
        shift: PlaneShift | tuple[int],
    ) -> ZNode:
        if not isinstance(shift, PlaneShift):
            shift = PlaneShift(*shift)
        x, y, k = self._ccoord
        new_x = x + shift[0]
        new_y = y + shift[1]

        return ZNode(coord=CartesianCoord(x=new_x, y=new_y, k=k), shape=self._shape)

    def __sub__(
        self,
        other: ZNode,
    ) -> PlaneShift:
        if not isinstance(other, ZNode):
            raise TypeError(f"Expected {other} to be {type(self).__name__}, got {type(other)}")
        if self._shape != other._shape:
            raise ValueError(
                f"Expected {self, other} to have the same shape, got {self._shape, other._shape}"
            )
        x_shift: int = self._ccoord.x - other._ccoord.x
        y_shift: int = self._ccoord.y - other._ccoord.y
        try:
            return PlaneShift(x_shift=x_shift, y_shift=y_shift)
        except:
            raise ValueError(f"{other} cannot be subtracted from {self}")

    def __hash__(self) -> int:
        return (self._ccoord, self._shape).__hash__()

    def __repr__(self) -> str:
        if self.convert_to_z:
            coord = self.zcoord
            if coord.k is None:
                coord_str = f"{coord.u, coord.w, coord.j, coord.z}"
            else:
                coord_str = f"{coord.u, coord.w, coord.k, coord.j, coord.z}"
        else:
            coord = self._ccoord
            if coord.k is None:
                coord_str = f"{coord.x, coord.y}"
            else:
                coord_str = f"{coord.x, coord.y, coord.k}"
        if self._shape == ZShape():
            shape_str = ""
        else:
            shape_str = f", shape={self._shape.m, self._shape.t!r}"
        return f"{type(self).__name__}(" + coord_str + shape_str + ")"

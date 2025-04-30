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
from enum import Enum
from itertools import product
from typing import Callable, Generator, Iterable

from minorminer.utils.zephyr.coordinate_systems import (CartesianCoord, ZephyrCoord,
                                                        cartesian_to_zephyr, zephyr_to_cartesian)
from minorminer.utils.zephyr.plane_shift import PlaneShift

ZShape = namedtuple("ZShape", ["m", "t"], defaults=(None, None))


class EdgeKind(Enum):
    """Kinds of an edge (coupler) between two nodes in a Zephyr graph."""
    INTERNAL = 1
    EXTERNAL = 2
    ODD = 3


class NodeKind(Enum):
    """Kinds of a node (qubit) in a Zephyr graph."""
    VERTICAL = 0    # The `u` coordinate of a Zephyr coordinate of a vertical node (qubit) is zero.
    HORIZONTAL = 1  # The `u` coordinate of a Zephyr coordinate of a horizontal node (qubit) is one.


class Edge:
    """Initializes an Edge with nodes x, y.

    Args:
        x: One endpoint of edge.
        y: Another endpoint of edge.
    """

    def __init__(self, x: int , y: int) -> None:
        self._edge = self._set_edge(x, y)

    def _set_edge(self, x: int, y: int) -> tuple[int, int]:
        """Returns ordered tuple corresponding to the set {x, y}."""
        if x < y:
            return (x, y)
        else:
            return (y, x)

    def __hash__(self):
        return hash(self._edge)

    def __getitem__(self, index: int) -> int:
        return self._edge[index]

    def __eq__(self, other: Edge) -> bool:
        return self._edge == other._edge

    def __str__(self) -> str:
        return f"{type(self).__name__}{self._edge}"

    def __repr__(self) -> str:
        return f"{type(self).__name__}{self._edge}"


class ZEdge(Edge):
    """Initializes a ZEdge with 'ZNode' nodes x, y.

    Args:
        x (ZNode): Endpoint of edge. Must have same shape as ``y``.
        y (ZNode): Endpoint of edge. Must have same shape as ``x``.
        check_edge_valid (bool, optional): Flag to whether check the validity of values and types of ``x``, ``y``.
            Defaults to True.

    Raises:
        TypeError: If either of x or y is not 'ZNode'.
        ValueError: If x, y do not have the same shape.
        ValueError: If x, y are not neighbors in a perfect yield (quotient)
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

            for kind in EdgeKind:
                if x.is_neighbor(y, nbr_kind=kind):
                    self._edge_kind = kind
                    break
            else:
                raise ValueError(f"Expected x, y to be neighbors, got {x, y}")

        self._edge = self._set_edge(x, y)

    @property
    def edge_kind(self) -> EdgeKind:
        return self._edge_kind


class ZNode:
    """Initializes 'ZNode' with coord and optional shape.

    Args:
        coord (CartesianCoord | ZephyrCoord | tuple[int]): Coordinate in (quotient) Zephyr or (quotient) Cartesian.
        shape (ZShape | tuple[int | None] | None, optional): Shape of the Zephyr graph containing this ZNode.
            If a ZShape is passed, it should be a namedtuple with fields `m` (grid size of the Zephyr graph) 
            and `t` (tile size of the Zephyr graph). Defaults to None.
        convert_to_z (bool | None, optional): Whether to express the coordinates in ZephyrCoordinates.
            Defaults to None.

    Note: 
        If the `k` field of the given `coord` (whether a `CartesianCoord` or `ZephyrCoord`) is not `None`,
        then `shape` must be provided and its `t` field (the tile size of the Zephyr graph) must not be `None`.
        Otherwise, a `ValueError` will be raised.

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
    >>> zn1.neighbors(nbr_kind=EdgeKind.ODD)
    [ZNode(CartesianCoord(x=3, y=2, k=None), shape=ZShape(m=5, t=None)),
        ZNode(CartesianCoord(x=7, y=2, k=None), shape=ZShape(m=5, t=None))]
    """

    def __init__(
        self,
        coord: CartesianCoord | ZephyrCoord | tuple[int],
        shape: ZShape | tuple[int | None] | None = None,
        convert_to_z: bool | None = None,
    ) -> None:
        if shape:
            self.shape = ZShape(*shape)
        else:
            self.shape = ZShape()

        if convert_to_z is None:
            self.convert_to_z = len(coord) in (4, 5)
        else:
            self.convert_to_z = convert_to_z

        # convert coord to CartesianCoord or ZephyrCoord
        if not isinstance(coord, (CartesianCoord, ZephyrCoord)):
            coord = self.tuple_to_coord(coord)

        # convert coord to CartesianCoord
        if isinstance(coord, ZephyrCoord):
            coord = zephyr_to_cartesian(coord)

        self.ccoord = coord

    @property
    def shape(self) -> ZShape | None:
        """Returns the shape of the Zephyr graph the node belongs to."""
        return self._shape

    @shape.setter
    def shape(self, new_shape: ZShape | tuple[int | None]) -> None:
        """Sets a new value for shape"""
        if isinstance(new_shape, tuple):
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

        for var, val in new_shape._asdict().items():
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
        for c in new_ccoord:
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
    def tuple_to_coord(coord: tuple[int]) -> CartesianCoord | ZephyrCoord:
        """Takes a tuple[int] and returns the corresponding ``CartesianCoord`` or ``ZephyrCoord``"""
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
            for var, val in [("u", u), ("j", j)]:
                if not val in [0, 1]:
                    raise ValueError(f"Expected {var} to be in [0, 1], got {val}")

            if len(k) == 0:
                return ZephyrCoord(u=u, w=w, j=j, z=z)

            return ZephyrCoord(u=u, w=w, k=k[0], j=j, z=z)
            
        raise ValueError(f"coord can have length 2, 3, 4 or 5, got {len_coord}")

    @property
    def zcoord(self) -> ZephyrCoord:
        """Returns ZephyrCoordinate corresponding to ccoord"""
        return cartesian_to_zephyr(self._ccoord)

    @property
    def node_kind(self) -> NodeKind:
        """Returns the node kind of self"""
        if self._ccoord.x % 2 == 0:
            return NodeKind.VERTICAL
        return NodeKind.HORIZONTAL

    @property
    def direction(self) -> int:
        """Returns the direction of node, i.e. its `u` coordinate in Zephyr coordinates."""
        return self.node_kind.value

    def is_quo(self) -> bool:
        """Decides if the ZNode object is quotient"""
        return (self._ccoord.k is None) and (self._shape.t is None)

    def to_quo(self) -> ZNode:
        """Returns the quotient ZNode corresponding to self"""
        qshape = ZShape(m=self._shape.m)
        qccoord = CartesianCoord(x=self._ccoord, y=self._ccoord)
        return ZNode(coord=qccoord, shape=qshape, convert_to_z=self.convert_to_z)

    def is_vertical(self) -> bool:
        """Returns True if the node represents a vertical qubit (i.e., its `u` coordinate in Zephyr coordinates is 0)."""
        return self.node_kind is NodeKind.VERTICAL

    def is_horizontal(self) -> bool:
        """Returns True if the node represents a horizontal qubit (i.e., its `u` coordinate in Zephyr coordinates is 1)."""
        return self.node_kind is NodeKind.HORIZONTAL

    def neighbor_kind(
        self,
        other: ZNode,
    ) -> EdgeKind | None:
        """Returns the kind of edge between two ZNodes.

        :class:`EdgeKind` if there is an edge in perfect Zephyr between them or ``None``.

        Args:
            other (ZNode): The neigboring :class:`ZNode`.

        Returns:
            EdgeKind | None: The edge kind between self and other in perfect Zephyr or None if there is no edge between in perfect Zephyr.
        """
        if not isinstance(other, ZNode):
            other = ZNode(other)

        if self._shape != other._shape:
            return None

        coord1 = self._ccoord
        coord2 = other._ccoord
        x1, y1 = coord1.x, coord1.y
        x2, y2 = coord2.x, coord2.y
        if abs(x1 - x2) == abs(y1 - y2) == 1:
            return EdgeKind.INTERNAL

        if x1 % 2 != x2 % 2:
            return None

        if coord1.k != coord2.k:  # odd, external neighbors only on the same k
            return None

        if self.node_kind is NodeKind.VERTICAL:  # self vertical
            if x1 != x2:  # odd, external neighbors only on the same vertical lines
                return None
    
            diff_y = abs(y1 - y2)
            return EdgeKind.ODD if diff_y == 2 else EdgeKind.EXTERNAL if diff_y == 4 else None

        # else, self is horizontal
        if y1 != y2:  # odd, external neighbors only on the same horizontal lines
            return None

        diff_x = abs(x1 - x2)
        return EdgeKind.ODD if diff_x == 2 else EdgeKind.EXTERNAL if diff_x == 4 else None

    def internal_neighbors_generator(
        self,
        where: Callable[[CartesianCoord | ZephyrCoord], bool] = lambda coord: True,
    ) -> Generator[ZNode]:
        """Generator of internal neighbors of self when restricted by `where`.

        Args:
            where (Callable[[CartesianCoord | ZephyrCoord], bool], optional):
                A coordinate filter. Applies to `ccoord` if `self.convert_to_z` is False,
                or to `zcoord` if `self.convert_to_z` is True. Defaults to `lambda coord: True`.

        Yields:
            ZNode: Internal neighbors of self when restricted by `where`.
        """
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
            except (TypeError, ValueError):
                pass

    def external_neighbors_generator(
        self,
        where: Callable[[CartesianCoord | ZephyrCoord], bool] = lambda coord: True,
    ) -> Generator[ZNode]:
        """Generator of external neighbors of self when restricted by `where`.

        Args:
            where (Callable[[CartesianCoord | ZephyrCoord], bool], optional):
                A coordinate filter. Applies to `ccoord` if `self.convert_to_z` is False,
                or to `zcoord` if `self.convert_to_z` is True. Defaults to `lambda coord: True`.

        Yields:
            ZNode: External neighbors of self when restricted by `where`.
        """
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
            except (TypeError, ValueError):
                pass

    def odd_neighbors_generator(
        self,
        where: Callable[[CartesianCoord | ZephyrCoord], bool] = lambda coord: True,
    ) -> Generator[ZNode]:
        """Generator of odd neighbors of self when restricted by `where`.

        Args:
            where (Callable[[CartesianCoord | ZephyrCoord], bool], optional):
                A coordinate filter. Applies to `ccoord` if `self.convert_to_z` is False,
                or to `zcoord` if `self.convert_to_z` is True. Defaults to `lambda coord: True`.

        Yields:
            ZNode: Odd neighbors of self when restricted by `where`.
        """
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
            except (TypeError, ValueError):
                pass

    def neighbors_generator(
        self,
        nbr_kind: EdgeKind | Iterable[EdgeKind] | None = None,
        where: Callable[[CartesianCoord | ZephyrCoord], bool] = lambda coord: True,
    ) -> Generator[ZNode]:
        """Generator of neighbors of self when restricted by `nbr_kind` and `where`.

        Args:
            nbr_kind (EdgeKind | Iterable[EdgeKind] | None, optional):
                Edge kind filter. Restricts yielded neighbors to those connected by the given edge kind(s).
                If None, no filtering is applied. Defaults to None.
            where (Callable[[CartesianCoord | ZephyrCoord], bool], optional):
                A coordinate filter. Applies to `ccoord` if `self.convert_to_z` is False,
                or to `zcoord` if `self.convert_to_z` is True. Defaults to `lambda coord: True`.

        Yields:
            ZNode: Neighbors of self when restricted by `nbr_kind` and `where`.
        """
        if nbr_kind is None:
            kinds = {kind for kind in EdgeKind}
        elif isinstance(nbr_kind, EdgeKind):
            kinds = {nbr_kind}
        else:
            kinds = set(nbr_kind)
        if EdgeKind.INTERNAL in kinds:
            for cc in self.internal_neighbors_generator(where=where):
                yield cc
        if EdgeKind.EXTERNAL in kinds:
            for cc in self.external_neighbors_generator(where=where):
                yield cc
        if EdgeKind.ODD in kinds:
            for cc in self.odd_neighbors_generator(where=where):
                yield cc

    def neighbors(
        self,
        nbr_kind: EdgeKind | Iterable[EdgeKind] | None = None,
        where: Callable[[CartesianCoord | ZephyrCoord], bool] = lambda coord: True,
    ) -> set[ZNode]:
        """Returns set of neighbors of self when restricted by `nbr_kind` and `where`.

        Args:
            nbr_kind (EdgeKind | Iterable[EdgeKind] | None, optional):
                Edge kind filter. Restricts returned neighbors to those connected by the given edge kind(s).
                If None, no filtering is applied. Defaults to None.
            where (Callable[[CartesianCoord | ZephyrCoord], bool], optional):
                A coordinate filter. Applies to `ccoord` if `self.convert_to_z` is False,
                or to `zcoord` if `self.convert_to_z` is True. Defaults to `lambda coord: True`.
        Returns:
            set[ZNode]: Set of neighbors of self when restricted by `nbr_kind` and `where`.
        """
        return set(self.neighbors_generator(nbr_kind=nbr_kind, where=where))

    def internal_neighbors(
        self,
        where: Callable[[CartesianCoord | ZephyrCoord], bool] = lambda coord: True,
    ) -> set[ZNode]:
        """Returns the set of internal neighbors of self when restricted by `where`.

        Args:
            where (Callable[[CartesianCoord | ZephyrCoord], bool], optional):
                A coordinate filter. Applies to `ccoord` if `self.convert_to_z` is False,
                or to `zcoord` if `self.convert_to_z` is True. Defaults to `lambda coord: True`.
        Returns:
            set[ZNode]: Set of internal neighbors of self when restricted by `where`.
        """
        return set(self.neighbors_generator(nbr_kind=EdgeKind.INTERNAL, where=where))

    def external_neighbors(
        self,
        where: Callable[[CartesianCoord | ZephyrCoord], bool] = lambda coord: True,
    ) -> set[ZNode]:
        """Returns the set of external neighbors of self when restricted by `where`.

        Args:
            where (Callable[[CartesianCoord | ZephyrCoord], bool], optional):
                A coordinate filter. Applies to `ccoord` if `self.convert_to_z` is False,
                or to `zcoord` if `self.convert_to_z` is True. Defaults to `lambda coord: True`.
        Returns:
            set[ZNode]: Set of external neighbors of self when restricted by `where`.
        """
        return set(self.neighbors_generator(nbr_kind=EdgeKind.EXTERNAL, where=where))

    def odd_neighbors(
        self,
        where: Callable[[CartesianCoord | ZephyrCoord], bool] = lambda coord: True,
    ) -> set[ZNode]:
        """Returns the set of odd neighbors of self when restricted by `where`.

        Args:
            where (Callable[[CartesianCoord | ZephyrCoord], bool], optional):
                A coordinate filter. Applies to `ccoord` if `self.convert_to_z` is False,
                or to `zcoord` if `self.convert_to_z` is True. Defaults to `lambda coord: True`.
        Returns:
            set[ZNode]: Set of odd neighbors of self when restricted by `where`.
        """
        return set(self.neighbors_generator(nbr_kind=EdgeKind.ODD, where=where))

    def is_internal_neighbor(
        self,
        other: ZNode,
        where: Callable[[CartesianCoord | ZephyrCoord], bool] = lambda coord: True,
    ) -> bool:
        """Returns whether other is an internal neighbor of self when restricted by `where`.

        Args:
            other (ZNode): Another instance to compare with self.
            where (Callable[[CartesianCoord | ZephyrCoord], bool], optional):
                A coordinate filter. Applies to `ccoord` if `self.convert_to_z` is False,
                or to `zcoord` if `self.convert_to_z` is True. Defaults to `lambda coord: True`.

        Returns:
            bool: Whether other is an internal neighbor of self when restricted by `where`.
        """
        for nbr in self.internal_neighbors_generator(where=where):
            if other == nbr:
                return True
        return False

    def is_external_neighbor(
        self,
        other: ZNode,
        where: Callable[[CartesianCoord | ZephyrCoord], bool] = lambda coord: True,
    ) -> bool:
        """Returns whether other is an external neighbor of self when restricted by `where`.

        Args:
            other (ZNode): Another instance to compare with self.
            where (Callable[[CartesianCoord | ZephyrCoord], bool], optional):
                A coordinate filter. Applies to `ccoord` if `self.convert_to_z` is False,
                or to `zcoord` if `self.convert_to_z` is True. Defaults to `lambda coord: True`.

        Returns:
            bool: Whether other is an external neighbor of self when restricted by `where`.
        """
        for nbr in self.external_neighbors_generator(where=where):
            if other == nbr:
                return True
        return False

    def is_odd_neighbor(
        self,
        other: ZNode,
        where: Callable[[CartesianCoord | ZephyrCoord], bool] = lambda coord: True,
    ) -> bool:
        """Returns whether other is an odd neighbor of self when restricted by `where`.

        Args:
            other (ZNode): Another instance to compare with self.
            where (Callable[[CartesianCoord | ZephyrCoord], bool], optional):
                A coordinate filter. Applies to `ccoord` if `self.convert_to_z` is False,
                or to `zcoord` if `self.convert_to_z` is True. Defaults to `lambda coord: True`.

        Returns:
            bool: Whether other is an odd neighbor of self when restricted by `where`.
        """
        for nbr in self.odd_neighbors_generator(where=where):
            if other == nbr:
                return True
        return False

    def is_neighbor(
        self,
        other: ZNode,
        nbr_kind: EdgeKind | Iterable[EdgeKind] | None = None,
        where: Callable[[CartesianCoord | ZephyrCoord], bool] = lambda coord: True,
    ) -> bool:
        """Returns whether other is a neighbor of self when restricted by `nbr_kind` and `where`.

        Args:
            other (ZNode): Another instance to compare with self.
            nbr_kind (EdgeKind | Iterable[EdgeKind] | None, optional):
                Edge kind filter. Restricts neighbors to those connected by the given edge kind(s).
                If None, no filtering is applied. Defaults to None.
            where (Callable[[CartesianCoord | ZephyrCoord], bool], optional):
                A coordinate filter. Applies to `ccoord` if `self.convert_to_z` is False,
                or to `zcoord` if `self.convert_to_z` is True. Defaults to `lambda coord: True`.

        Returns:
            bool: Whether other is a neighbor of self when restricted by `nbr_kind` and `where`.
        """
        for nbr in self.neighbors_generator(nbr_kind=nbr_kind, where=where):
            if other == nbr:
                return True
        return False

    def incident_edges(
        self,
        nbr_kind: EdgeKind | Iterable[EdgeKind] | None = None,
        where: Callable[[CartesianCoord | ZephyrCoord], bool] = lambda coord: True,
    ) -> list[ZEdge]:
        """Returns incident edges with self when restricted by `nbr_kind` and `where`.

        Args:
            nbr_kind (EdgeKind | Iterable[EdgeKind] | None, optional):
                Edge kind filter. Restricts returned edges to those having the given edge kind(s).
                If None, no filtering is applied. Defaults to None.
            where (Callable[[CartesianCoord | ZephyrCoord], bool], optional):
                A coordinate filter. Applies to `ccoord` if `self.convert_to_z` is False,
                or to `zcoord` if `self.convert_to_z` is True. Defaults to `lambda coord: True`.

        Returns:
            list[ZEdge]: list edges incident with self when restricted by `nbr_kind` and `where`.
        """
        return [ZEdge(self, v) for v in self.neighbors(nbr_kind=nbr_kind, where=where)]

    def degree(
        self,
        nbr_kind: EdgeKind | Iterable[EdgeKind] | None = None,
        where: Callable[[CartesianCoord | ZephyrCoord], bool] = lambda coord: True,
    ) -> int:
        """Returns degree of self when restricted by `nbr_kind` and `where`.

        Args:
            nbr_kind (EdgeKind | Iterable[EdgeKind] | None, optional):
                Edge kind filter. Restricts counting the neighbors to those connected by the given edge kind(s).
                If None, no filtering is applied. Defaults to None.
            where (Callable[[CartesianCoord | ZephyrCoord], bool], optional):
                A coordinate filter. Applies to `ccoord` if `self.convert_to_z` is False,
                or to `zcoord` if `self.convert_to_z` is True. Defaults to `lambda coord: True`.

        Returns:
            int: degree of self when restricted by `nbr_kind` and `where`.
        """
        return len(self.neighbors(nbr_kind=nbr_kind, where=where))

    def __eq__(self, other: ZNode) -> bool:
        if not isinstance(other, ZNode):
            raise TypeError(f"Expected {other} to be {type(self).__name__}, got {type(other)}")
        if self._shape != other._shape:
            raise ValueError(
                f"Expected {self, other} to have the same shape, got {self._shape, other._shape}"
            )
        return self._ccoord == other._ccoord


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

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

from typing import Iterator


class PlaneShift:
    """Initializes PlaneShift with an x, y.

    Args:
        x (int): The displacement in the x-direction of a Cartesian coordinate.
        y (int): The displacement in the y-direction of a Cartesian coordinate.


    Example:
    >>> from minorminer.utils.zephyr.plane_shift import PlaneShift
    >>> ps1 = PlaneShift(1, 3)
    >>> ps2 = PlaneShift(2, -4)
    >>> print(f"{ps1 + ps2  = }, {2*ps1 = }")
    """
    def __init__(self, x: int, y: int) -> None:
        self._xy = (x, y)

    @property
    def x(self) -> int:
        """Returns the shift in x direction"""
        return self._xy[0]

    @property
    def y(self) -> int:
        """Returns the shift in y direction"""
        return self._xy[1]

    def __mul__(self, scale: int) -> PlaneShift:
        """Multiplies the self from left by the number value ``scale``.

        Args:
            scale (int): The scale for left-multiplying self with.

        Returns:
            PlaneShift: The result of left-multiplying self by ``scale``.
        """
        return type(self)(scale * self.x, scale * self.y)


    def __rmul__(self, scale: int) -> PlaneShift:
        """Multiplies the ``self`` from right by the number value ``scale``.

        Args:
            scale (int): The scale for right-multiplying ``self`` with.

        Returns:
            PlaneShift: The result of right-multiplying ``self`` by ``scale``.
        """
        return self * scale

    def __add__(self, other: PlaneShift) -> PlaneShift:
        """
        Adds another PlaneShift object to self.

        Args:
            other (PlaneShift): The object to add self by.

        Raises:
            TypeError: If other is not PlaneShift

        Returns:
            PlaneShift: The displacement in CartesianCoord by self followed by other.
        """
        if type(self) != type(other):
            raise TypeError(f"Expected other to be {type(self).__name__}, got {type(other).__name__}")        
        return type(self)(self.x + other.x, self.y + other.y)

    def __iter__(self) -> Iterator[int]:
        return self._xy.__iter__()

    def __len__(self) -> int:
        return len(self._xy)

    def __hash__(self) -> int:
        return hash(self._xy)

    def __getitem__(self, key) -> int:
        return self._xy[key]

    def __eq__(self, other: PlaneShift) -> bool:
        return type(self) == type(other) and self._xy == other._xy


class ZPlaneShift(PlaneShift):
    """Initializes ZPlaneShift with an x, y.

    Args:
        x (int): The displacement in the x-direction of a Cartesian coordinate.
        y (int): The displacement in the y-direction of a Cartesian coordinate.

    Raises:
        TypeError: If x or y is not 'int'.
        ValueError: If x and y have different parity.

    Example:
    >>> from minorminer.utils.zephyr.plane_shift import ZPlaneShift
    >>> ps1 = ZPlaneShift(1, 3)
    >>> ps2 = ZPlaneShift(2, -4)
    >>> print(f"{ps1 + ps2  = }, {2*ps1 = }")
    """

    def __init__(
        self,
        x: int,
        y: int,
    ) -> None:
        for shift in [x, y]:
            if not isinstance(shift, int):
                raise TypeError(f"Expected {shift} to be 'int', got {type(shift)}")
        if x % 2 != y % 2:
            raise ValueError(
                f"Expected x, y to have the same parity, got {x, y}"
            )
        self._xy = (x, y)

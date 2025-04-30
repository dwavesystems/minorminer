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
from typing import Iterator

Shift = namedtuple("Shift", ["x", "y"])


class PlaneShift:
    """Initializes PlaneShift with an x_shift, y_shift.

    Args:
        x_shift (int): The displacement in the x-direction of a CartesianCoord.
        y_shift (int): The displacement in the y-direction of a CartesianCoord.

    Raises:
        TypeError: If x_shift or y_shift is not 'int'.
        ValueError: If x_shift and y_shift have different parity.

    Example:
    >>> from minorminer.utils.zephyr.plane_shift import PlaneShift
    >>> ps1 = PlaneShift(1, 3)
    >>> ps2 = PlaneShift(2, -4)
    >>> print(f"{ps1 + ps2  = }, {2*ps1 = }")
    """

    def __init__(
        self,
        x_shift: int,
        y_shift: int,
    ) -> None:
        for shift in [x_shift, y_shift]:
            if not isinstance(shift, int):
                raise TypeError(f"Expected {shift} to be 'int', got {type(shift)}")
        if x_shift % 2 != y_shift % 2:
            raise ValueError(
                f"Expected x_shift, y_shift to have the same parity, got {x_shift, y_shift}"
            )
        self._shift = Shift(x_shift, y_shift)

    @property
    def x(self) -> int:
        """Returns the shift in x direction"""
        return self._shift.x

    @property
    def y(self) -> int:
        """Returns the shift in y direction"""
        return self._shift.y

    def __mul__(self, scale: int | float) -> PlaneShift:
        """Multiplies the self from left by the number value ``scale``.

        Args:
            scale (int | float): The scale for left-multiplying self with.

        Raises:
            TypeError: If scale is not 'int' or 'float'.
            ValueError: If the resulting PlaneShift has non-whole values.

        Returns:
            PlaneShift: The result of left-multiplying self by scale.
        """
        if not isinstance(scale, (int, float)):
            raise TypeError(f"Expected scale to be int or float, got {type(scale)}")

        new_shift_x = scale * self._shift.x
        new_shift_y = scale * self._shift.y
        if int(new_shift_x) != new_shift_x or int(new_shift_y) != new_shift_y:
            raise ValueError(f"{scale} cannot be multiplied by {self}")

        return PlaneShift(int(new_shift_x), int(new_shift_y))

    def __rmul__(self, scale: int | float) -> PlaneShift:
        """Multiplies the self from right by the number value ``scale``.

        Args:
            scale (int | float): The scale for right-multiplying self with.

        Raises:
            TypeError: If scale is not 'int' or 'float'.
            ValueError: If the resulting PlaneShift has non-whole values.

        Returns:
            PlaneShift: The result of right-multiplying self by scale.
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
        if not isinstance(other, PlaneShift):
            raise TypeError(f"Expected other to be PlaneShift, got {type(other)}")
        return PlaneShift(self._shift.x + other._shift.x, self._shift.y + other._shift.y)

    def __iter__(self) -> Iterator[int]:
        return self._shift.__iter__()

    def __len__(self) -> int:
        return len(self._shift)

    def __hash__(self) -> int:
        return self._shift.__hash__()

    def __getitem__(self, key) -> int:
        return self._shift.__getitem__(key)

    def __eq__(self, other: PlaneShift) -> bool:
        if not isinstance(other, PlaneShift):
            raise TypeError(f"Expected other to be PlaneShift, got {type(other)}")
        return self._shift == other._shift

    def __ne__(self, other: PlaneShift) -> bool:
        if not isinstance(other, PlaneShift):
            raise TypeError(f"Expected other to be PlaneShift, got {type(other)}")
        return not self._shift == other._shift

    def __lt__(self, other: PlaneShift) -> bool:
        if not isinstance(other, PlaneShift):
            raise TypeError(f"Expected other to be PlaneShift, got {type(other)}")
        return self._shift < other._shift

    def __le__(self, other: PlaneShift) -> bool:
        if not isinstance(other, PlaneShift):
            raise TypeError(f"Expected other to be PlaneShift, got {type(other)}")
        return (self == other) or (self < other)

    def __gt__(self, other: PlaneShift) -> bool:
        if not isinstance(other, PlaneShift):
            raise TypeError(f"Expected other to be PlaneShift, got {type(other)}")
        return self._shift > other._shift

    def __ge__(self, other: PlaneShift) -> bool:
        if not isinstance(other, PlaneShift):
            raise TypeError(f"Expected other to be PlaneShift, got {type(other)}")
        return (self == other) or (self > other)

    def __repr__(self) -> str:
        return f"{type(self).__name__}{self._shift.x, self._shift.y}"

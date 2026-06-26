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

"""
Specialized coordinate types for embedding-related algorithms on Zephyr topology.

These classes represent structured combinations of fields (u, w, k, j, z),
with each subclass omitting fields that are irrelevant to a specific context.
"""

from __future__ import annotations

from typing import Iterator

from minorminer.utils.zephyr.coordinate_systems import ZephyrCoord

__all__ = ["UWKJZ", "UWKJ", "UWJ"]


class UWKJZ(ZephyrCoord):
    """A subclass of ZephyrCoord that provides convenient representations and helpers for embedding-related algorithms.
    It captures the full coordinate of a Zephyr node.

    Example:
    >>> from burnaby.lattice_embedding.auxiliary_coordinates import UWKJZ
    >>> node = UWKJZ(u=0, w=2, k=3, j=1, z=0)
    >>> external_path = node.uwkj
    >>> quotient_external_path = node.uwj
    >>> print(f"The node {node} lies on the external path {external_path}, and belongs to the quotient external path {quotient_external_path}.")
    The node UWKJZ(u=0, w=2, k=3, j=1, z=0) lies on the external path UWKJ(u=0, w=2, k=3, j=1), and belongs to the quotient external path UWJ(u=0, w=2, j=1).
    """

    _fields = ("u", "w", "k", "j", "z")

    @property
    def uwkj(self) -> UWKJ:
        """Returns the ``UWKJ`` object containing the (u, w, k, j) fields of the coordinate."""
        return UWKJ(u=self.u, w=self.w, k=self.k, j=self.j)

    @property
    def uwj(self) -> UWJ:
        """Returns the ``UWJ`` object containing the (u, w, j) fields of the coordinate."""
        return UWJ(u=self.u, w=self.w, j=self.j)

    def to_tuple(self) -> tuple[int]:
        """Rerurns the tuple corresponding to the coordinate."""
        return (self.u, self.w, self.k, self.j, self.z)

    def __iter__(self) -> Iterator[int]:
        return (
            getattr(self, f)
            for f in self._fields
            # if getattr(self, f) is not None
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(u={self.u}, w={self.w}, k={self.k}, j={self.j}, z={self.z})"


class UWKJ(UWKJZ):
    """A specialization of ``UWKJZ`` that includes fields u, w, k, and j.

    It omits z, and so captures `external paths` in Zephyr topology.

    Provides convenient representations and helpers for embedding-related algorithms.
    Example:
    >>> from burnaby.lattice_embedding.auxiliary_coordinates import UWKJ
    >>> external_path = UWKJ(u=0, w=6, k=2, j=1)
    >>> quotient_external_path = external_path.uwj
    >>> print(f"The external path {external_path} belongs to the quotient external path {quotient_external_path}.")
    The external path UWKJ(u=0, w=6, k=2, j=1) belongs to the quotient external path UWJ(u=0, w=6, j=1).
    """

    _fields = ("u", "w", "k", "j")

    def to_tuple(self) -> tuple[int]:
        """Rerurns the tuple corresponding to the coordinate."""
        return (self.u, self.w, self.k, self.j)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(u={self.u}, w={self.w}, k={self.k}, j={self.j})"


class UWJ(UWKJ):
    """A specialization of ``UWKJ`` that includes fields u, w, and j.

    It omits both z and k from the full coordinate and is meant to capture a `quotient of external paths` in Zephyr topology,
    in the sense that many ``UWKJ``s can map to the same ``UWJ``.

    Provides convenient representations and helpers for embedding-related algorithms.
    """

    _fields = ("u", "w", "j")

    def to_tuple(self) -> tuple[int]:
        """Rerurns the tuple corresponding to the coordinate."""
        return (self.u, self.w, self.j)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(u={self.u}, w={self.w}, j={self.j})"

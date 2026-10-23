# Copyright 2026 D-Wave
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


"""Chains for the Zephyr clique embedder.

Vocabulary
----------
segment
    A contiguous stretch of one external path: the nodes ``(x, y, k)`` sharing the path's fixed
    coordinate and ``k``, with the varying coordinate stepping by 4 from ``start`` to ``end``
    (both inclusive). A segment only names nodes; whether consecutive ones are actually joined by
    couplers in a faulty grid is the caller's responsibility (segments built from :attr:`.Grid.runs`
    are).

chain
    A tuple of segments. An el-chain is two segments, its vertical arm followed by its horizontal
    arm, joined by an internal coupler at the elbow.

Chains are node sets plus their structure; :func:`chain_nodes` flattens one to its nodes. Nothing
here depends on which qubits or couplers a grid has; questions about how chains connect in a faulty
grid live in :mod:`.interchain_bridges`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from minorminer.utils._clique.grid import Node


class Segment(NamedTuple):
    """A contiguous stretch of one external path.

    Attributes:
        vertical: ``True`` for a vertical path (fixed ``x``), ``False`` for a horizontal one
            (fixed ``y``).
        fixed_coord: The path's fixed coordinate: ``x`` if vertical, ``y`` if horizontal.
        k: The path's ``k`` index.
        start: First varying coordinate (``y`` if vertical, ``x`` if horizontal).
        end: Last varying coordinate, ``end >= start`` and ``end - start`` a multiple of 4.
    """

    vertical: bool
    fixed_coord: int
    k: int
    start: int
    end: int

    def nodes(self) -> list[Node]:
        """The segment's nodes in order of increasing varying coordinate.

        Returns:
            The nodes ``(x, y, k)`` of the segment.
        """
        fixed, k = self.fixed_coord, self.k
        if self.vertical:
            return [(fixed, y, k) for y in range(self.start, self.end + 1, 4)]
        return [(x, fixed, k) for x in range(self.start, self.end + 1, 4)]


Chain = tuple[Segment, ...]  # an ordered collection of segments


def chain_nodes(chain: Chain) -> list[Node]:
    """All nodes of a chain, segment by segment.

    Args:
        chain: The chain.

    Returns:
        The nodes of every segment of ``chain``, in segment order.
    """
    return [node for segment in chain for node in segment.nodes()]

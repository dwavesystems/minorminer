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

"""Connections between two chains in a (possibly faulty) Zephyr grid.

Vocabulary
----------
adjacent
    Two chains are adjacent if some working coupler joins a node of one to a node of the other.
    A clique embedding needs every pair of its chains adjacent.

interchain bridge
    The internal nodes of a path that starts at a node of chain A, ends at a node of chain B,
    and uses only existing couplers and nodes that are not used by any chain. Bridges are
    listed in path order, from the chain A end to the chain B end. The empty bridge ``()`` means
    the chains are already adjacent.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minorminer.utils._clique.grid import Grid, Node

# deepest bridge implemented: a -- m1 -- m2 -- b
MAX_IMPLEMENTED_BRIDGE_NODES = 2


def chains_are_adjacent(grid: Grid, chain_a: Iterable[Node], chain_b: Iterable[Node]) -> bool:
    """Decide whether some working coupler joins two chains.

    Args:
        grid: The (possibly faulty) Zephyr grid.
        chain_a: The nodes of one chain.
        chain_b: The nodes of the other chain.

    Returns:
        ``True`` if a coupler joins some node of ``chain_a`` to some node of ``chain_b``, else
        ``False``.
    """
    ccoord_to_lcoord = grid.cartesian_to_linear
    edges = grid.edges
    lcoords_a = [lcoord for lcoord in map(ccoord_to_lcoord, chain_a) if lcoord is not None]
    for node_b in chain_b:
        lcoord_b = ccoord_to_lcoord(node_b)
        if lcoord_b is None:
            continue
        for lcoord_a in lcoords_a:
            edge = (lcoord_a, lcoord_b) if lcoord_a < lcoord_b else (lcoord_b, lcoord_a)
            if edge in edges:
                return True
    return False


def find_interchain_bridges(
    grid: Grid,
    chain_a: Collection[Node],
    chain_b: Collection[Node],
    used_nodes: set[Node],
    max_interchain_bridge_nodes: int = 2,
) -> list[tuple[Node, ...]]:
    """Find short interchain bridges between two chains.

    An interchain bridge between ``chain_a`` and ``chain_b`` is the sequence of internal nodes of
    a path in a possibly faulty grid that starts at a node of ``chain_a``, ends at a node of
    ``chain_b``, and whose internal nodes are all free, i.e. not in ``used_nodes``, ``chain_a``,
    or ``chain_b``. The endpoints stay in their chains; only the internal nodes form the bridge,
    listed in path order from the ``chain_a`` end to the ``chain_b`` end. The empty bridge ``()``
    comes from a path with no internal nodes, i.e. a single edge joining the two chains.

    This function returns every distinct bridge with at most ``max_interchain_bridge_nodes``
    nodes. Paths that differ only in their endpoints give the same bridge, which is listed once.
    If the chains are already joined by an edge, only the empty bridge is returned.

    Non-minimal bridges are included: a two-node bridge ``(m1, m2)`` is listed even when ``m1``
    alone already touches ``chain_b``. Such a bridge can still be the better repair, because
    its extra node may make one of the chains adjacent to a third chain as well.

    Note:
        Bridges with more than ``MAX_IMPLEMENTED_BRIDGE_NODES`` nodes are not implemented, so
        larger values of ``max_interchain_bridge_nodes`` are treated as
        ``MAX_IMPLEMENTED_BRIDGE_NODES``.

    Args:
        grid: The grid to find a bridge between the two chains in.
        chain_a: One of the chains.
        chain_b: The other chain.
        used_nodes: Nodes already used by other chains; a bridge cannot use them.
        max_interchain_bridge_nodes: Maximum number of middle nodes in a bridge,
            not counting the two endpoints. Values above 2 are treated as 2.
            With 0, only the direct-connection check is done: the result is
            ``[()]`` if the chains are directly connected and ``[]`` otherwise.

    Returns:
        A list of tuples of intermediate nodes, ordered by number of middle
        nodes (fewest first). Each tuple is one bridge:

        - ``()``: The two chains are already directly connected (no bridge
          needed). If present, it is the only element.
        - ``(m1,)``: One middle node.
        - ``(m1, m2)``: Two middle nodes, ``m1`` adjacent to ``chain_a`` and ``m2`` adjacent
          to ``chain_b``.
    """
    A = set(chain_a)
    B = set(chain_b)
    neighbors = grid.neighbors

    def is_free(n: Node) -> bool:
        return n not in used_nodes and n not in A and n not in B

    # One pass over A's neighbors does two jobs:
    # 1) it detects a direct A--B edge (the result is then just the empty bridge),
    # 2) it collects A-side candidates.
    a_neigh: set[Node] = set()
    for a in A:
        for nbr in neighbors(a):
            if nbr in B:
                return [()]
            if is_free(nbr):
                a_neigh.add(nbr)

    if max_interchain_bridge_nodes < 1:
        return []

    # B-side candidates; only membership is tested.
    b_neigh: set[Node] = {nbr for b in B for nbr in neighbors(b) if is_free(nbr)}

    all_bridges: list[tuple[Node, ...]] = [(m,) for m in a_neigh if m in b_neigh]

    if max_interchain_bridge_nodes < 2:
        return all_bridges

    # Look for a -- m1 -- m2 -- b. m1 adjacent to A, m2 adjacent to B, m1--m2 edge.
    # Enumerate from A-side middles, hop once more, check adjacency to B.
    for bridge_node_1 in a_neigh:
        for bridge_node_2 in neighbors(bridge_node_1):
            if bridge_node_2 in b_neigh:
                all_bridges.append((bridge_node_1, bridge_node_2))
    return all_bridges

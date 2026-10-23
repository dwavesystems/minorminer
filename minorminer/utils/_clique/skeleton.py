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

"""Phase 1 chains of the clique embedder: the el-chain skeleton of one window corner.

An el-chain runs from a vertical endpoint ``v`` next to the corner's horizontal window edge,
along ``v``'s external path, across one internal coupler at the elbow, and along a horizontal
external path to an endpoint ``h`` next to the corner's vertical window edge. It exists in a
faulty grid iff :meth:`.Grid.el_reachable` says so. The skeleton is a largest set of node-disjoint
el-chains whose endpoint pairing is monotone (defined below), which makes every two of them meet
at an internal coupler of the ideal grid: a set a clique embedding can be built on.

Reduction to a longest strictly increasing subsequence
------------------------------------------------------
An el-chain is uniquely determined by its endpoint pair, so node-disjointness reduces to distinct
endpoints. Pairing the two sides monotonically -- the vertical endpoint nearer the corner taking
the horizontal endpoint farther from it -- makes every two chains meet.

Each endpoint's key is its :attr:`.Grid.side_order` for the corner's side. The key folds the
block coordinate along the window side together with the node's reach rank within its block, and
is unique among the blocks next to one side, which are the only ones compared here.

Listing reachable pairs by vertical key (oriented by ``dir_x``), and within one vertical by
horizontal key against the subsequence direction, a strictly increasing subsequence of the
oriented horizontal keys ``dir_y * h_key`` uses each vertical at most once, each horizontal at
most once, and pairs them monotonically. A longest one is therefore a largest monotone skeleton.
"""

from __future__ import annotations

from collections import defaultdict
from operator import itemgetter
from typing import TYPE_CHECKING

from minorminer.utils._clique.chain import Chain, Segment
from minorminer.utils._clique.longest_strictly_increasing_subsequence import \
    longest_strictly_increasing_subsequence
from minorminer.utils._clique.windows import corner_orientation

if TYPE_CHECKING:
    from minorminer.utils._clique.grid import Block, Grid
    from minorminer.utils._clique.windows import Corner, Window


def skeleton_chains(grid: Grid, window: Window, corner: Corner) -> list[Chain]:
    """Find a largest monotone set of el-chains for one corner of a window.

    Args:
        grid: The (possibly faulty) grid.
        window: The window to find the set of el-chains over.
        corner: The corner the chains bend around.

    Returns:
        The skeleton chains, each ``(vertical_segment, horizontal_segment)``. The chains are
        pairwise node-disjoint and every node is present; whether every pair of them is
        adjacent in a faulty grid is checked later, by :func:`.broken_pairs`.
    """
    bt, lr = corner
    dir_x, dir_y = corner_orientation(corner)
    v_blocks, h_blocks = _endpoint_blocks(grid, window, corner)
    v_order_by_block = grid.side_order[bt]
    h_order_by_block = grid.side_order[lr]
    el_reachable = grid.el_reachable

    candidates_by_v_key: dict[int, list[tuple[int, Chain]]] = defaultdict(list)
    for v_block in v_blocks:
        v_order_by_k = v_order_by_block.get(v_block)
        if not v_order_by_k:
            continue
        for h_block in h_blocks:
            h_order_by_k = h_order_by_block.get(h_block)
            if not h_order_by_k:
                continue
            for v_k, v_key in v_order_by_k.items():
                v = (*v_block, v_k)
                for h_k, h_key in h_order_by_k.items():
                    reach = el_reachable(v, (*h_block, h_k))
                    if reach is None:
                        continue
                    v_quo, h_quo = reach
                    chain = (
                        Segment(True, v_quo[0], v_k, v_quo[2], v_quo[3]),
                        Segment(False, h_quo[1], h_k, h_quo[2], h_quo[3]),
                    )
                    candidates_by_v_key[v_key].append((h_key, chain))

    # verticals in dir_x order; within one vertical, horizontals AGAINST the subsequence
    # direction, so a strictly increasing subsequence takes at most one per vertical
    chains = []
    values = []
    for v_key in sorted(candidates_by_v_key, reverse=(dir_x == -1)):
        for h_key, chain in sorted(
            candidates_by_v_key[v_key], key=itemgetter(0), reverse=(dir_y == 1)
        ):
            chains.append(chain)
            values.append(dir_y * h_key)

    _, indices = longest_strictly_increasing_subsequence(values)
    return [chains[i] for i in indices]


def _endpoint_blocks(grid: Grid, window: Window, corner: Corner) -> tuple[list[Block], list[Block]]:
    """Blocks that can hold a skeleton chain's endpoints.

    Vertical endpoints sit in the two rows of vertical blocks adjacent to the corner's
    horizontal window edge; horizontal endpoints in the two columns of horizontal blocks
    adjacent to its vertical window edge. Rows or columns outside the grid are dropped.

    Args:
        grid: The grid.
        window: The window to find the blocks around its edges.
        corner: The corner the skeleton chains bend around.

    Returns:
        The pair ``(v_blocks, h_blocks)`` where ``v_blocks`` is the list of vertical
        endpoint blocks and ``h_blocks`` is the list of horizontal endpoint blocks.
    """
    x0, x1, y0, y1 = window
    bt, lr = corner
    lo, hi = grid.abs_min, grid.abs_max
    edge_y = y0 if bt == "t" else y1
    edge_x = x0 if lr == "l" else x1
    v_blocks = [
        (x, y) for x in range(x0, x1 + 1, 2) for y in (edge_y - 1, edge_y + 1) if lo <= y <= hi
    ]
    h_blocks = [
        (x, y) for x in (edge_x - 1, edge_x + 1) for y in range(y0, y1 + 1, 2) if lo <= x <= hi
    ]
    return v_blocks, h_blocks

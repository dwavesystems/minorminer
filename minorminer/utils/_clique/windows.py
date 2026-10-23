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


"""Windows and corners: the units the clique search sweeps over.

Vocabulary
----------
window
    ``(x0, x1, y0, y1)``: an axis-aligned box of Zephyr Cartesian coordinates, all four even,
    with ``0 <= x0 < x1 <= 4m`` and ``0 <= y0 < y1 <= 4m``. A side ``w`` units long spans
    ``n = w / 2 + 1`` even coordinates.

corner
    ``(bt, lr)`` with ``bt`` in ``{"t", "b"}`` (top/bottom) and ``lr`` in ``{"l", "r"}``
    (left/right). Chains for a corner run from the window's top or bottom edge (vertical arm)
    to its left or right edge (horizontal arm). ``y`` grows downward, so "top" is ``y0``.
"""

from __future__ import annotations

import random
from math import ceil

Window = tuple[int, int, int, int]  # (x0, x1, y0, y1)
Corner = tuple[str, str]  # (bt, lr)

CORNERS: tuple[Corner, Corner, Corner, Corner] = (("t", "l"), ("t", "r"), ("b", "l"), ("b", "r"))


def corner_orientation(corner: Corner) -> tuple[int, int]:
    """Axis signs that make "increasing" point toward a corner.

    Multiplying ``x`` by ``dir_x`` and ``y`` by ``dir_y`` maps every corner onto the same
    orientation, so one longest-increasing-subsequence formulation serves all four corners.

    Args:
        corner: The corner ``(bt, lr)``.

    Returns:
        ``(dir_x, dir_y)``: ``dir_x`` is ``1`` for right corners and ``-1`` for left ones;
        ``dir_y`` is ``1`` for top corners and ``-1`` for bottom ones.
    """
    bt, lr = corner
    return (1 if lr == "r" else -1, 1 if bt == "t" else -1)


def candidate_windows(
    m: int, t: int, min_size: int, aspect_band: int, seed: int | None = None
) -> list[Window]:
    """All windows that could hold a clique of at least ``min_size`` chains, shuffled.

    A side spanning ``n`` even coordinates holds at most ``2 t n`` chains (see
    :func:`window_bound`), so every side must span at least ``ceil(min_size / (2 t))`` of them.
    Side lengths run in steps of 2 from that minimum up to the chip's ``4m``. For each square side
    ``w``, rectangles ``w x w'`` and ``w' x w`` with ``w < w' <= w + 2 aspect_band`` are included
    too. Every start position (even coordinates) of every size is listed.

    Args:
        m: Zephyr grid size.
        t: Zephyr tile size.
        min_size: Smallest clique size of interest; windows too small to reach it are not
            generated.
        aspect_band: How many more even coordinates a window's long side may span than its
            short side. ``0`` generates square windows only.
        seed: Seed for the shuffle, so the order is reproducible. ``None`` leaves the windows
            unshuffled, in order of size and then start position.
    Returns:
        The windows in a seeded random order; empty if no window can reach ``min_size``.
    """
    min_side = max(2, 2 * (ceil(min_size / (2 * t)) - 1))
    if min_side > 4 * m:
        return []

    sides = []
    for short in range(min_side, 4 * m + 1, 2):
        sides.append((short, short))
        for long in range(short + 2, min(short + 2 * aspect_band, 4 * m) + 1, 2):
            sides.append((short, long))
            sides.append((long, short))

    windows = [
        (x0, x0 + side_x, y0, y0 + side_y)
        for side_x, side_y in sides
        for x0 in range(0, 4 * m - side_x + 1, 2)
        for y0 in range(0, 4 * m - side_y + 1, 2)
    ]
    if seed is not None:
        random.Random(seed).shuffle(windows)
    return windows


def window_bound(window: Window, t: int) -> int:
    """Upper bound on the size of a clique found in a window.

    ``2 t min(n_x, n_y)`` where ``n_x`` and ``n_y`` count the even coordinates along each side.
    For skeleton chains this is a proof: their vertical endpoints are distinct nodes in the two
    rows next to one horizontal window edge (``2 t n_x`` candidates) and their horizontal
    endpoints distinct nodes in the two columns next to one vertical edge (``2 t n_y``), and
    repair only drops chains. On a perfect grid, a square window away from the chip boundary
    reaches the bound exactly; windows touching the boundary can fall short of it.

    Args:
        window: The window ``(x0, x1, y0, y1)``.
        t: Zephyr tile size.

    Returns:
        The bound.
    """
    x0, x1, y0, y1 = window
    n_x = (x1 - x0) // 2 + 1
    n_y = (y1 - y0) // 2 + 1
    return 2 * t * min(n_x, n_y)

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


"""
el_geometry.py -- ideal el-template geometry for Zephyr.

Pure geometry of the L-shaped ("el") templates used to build clique embeddings on a Zephyr graph.
Every value computed here depends only on the Zephyr grid size m and the ideal Zephyr topology;
nothing depends on the tile size of Zephyr, t, or which nodes or edges a particular Zephyr graph
actually has.

Coordinates
-----------
Coordinates are Zephyr Cartesian (x, y, k), following the convention of
:class:`dwave.graphs.ZephyrCartesianCoord`, with x, y integer lattice coordinates with different
parity in [0, 4m] and k in range(t). Orientation is fixed by the parity of x: even x is a vertical
qubit (y is then odd), odd x is a horizontal qubit (y is then even).

    node / qubit   (x, y, k)   a Zephyr node in Cartesian coordinates, k in range(t)
    block          (x, y)      a quotient Zephyr node (k folded out)

The functions here are entirely QUOTIENT (k-agnostic): they take and return block coordinates (x, y)
only, with k folded out, so nothing here depends on the tile size t. A "quo" in a name -- the
QuoSpan type, the v_quo/h_quo values -- marks exactly that: the object stands for every k at once.

El-templates
------------
A pair of a vertical block and a horizontal block (v_block, h_block) generically induce a unique
L-shaped path between the blocks. That path is the shortest path you would take from v_block to
h_block: travel along v_block's quotient external path toward h_block, hop one internal coupler
across to the nearest-to-h_block block on h_block's quotient external path, then continue along that
path to h_block. The induced el-template is this L-shaped path -- two arms meeting at the elbow,
one from each block. Pairs whose elbow falls outside [0, 4m] induce no el-template.

A QuoSpan is a 4-tuple giving an interval [a, b] on a single quotient external path, together with
the fixed coordinate of the path and its shift -- the varying coordinate's residue mod 4 (1 or 3) --
selecting which of the two interleaved quotient external paths with that fixed coordinate is meant.
The vertical and horizontal spans do NOT share a tuple layout -- the fixed coordinate and the shift
occupy opposite slots:

    vertical   QuoSpan corresponding to a vertical block (v_x, v_y):
        (v_x,       v_y % 4,  v_a, v_b)   ->  y-span [v_a, v_b] on column v_x
    horizontal QuoSpan corresponding to a horizontal block (h_x, h_y):
        (h_x % 4,  h_y,       h_a, h_b)   ->  x-span [h_a, h_b] on row    h_y

so a vertical span is (fixed x, shift, ...) and a horizontal span is (shift, fixed y, ...); callers
unpack the two positionally and must respect the difference. A template also has a length -- the
number of blocks along both arms, equivalently the qubit count of an instantiated chain.
"""

__all__ = ["elbow", "el_template", "el_template_length"]

QuoSpan = tuple[int, int, int, int]


def elbow(m: int, v_x: int, v_y: int, h_x: int, h_y: int) -> tuple[int, int] | None:
    """The elbow of the L-path from a vertical block to horizontal block.

    The elbow is the two blocks ``(v_x, v_y)`` and ``(h_x, h_y)`` joined by the single internal
    coupler that bridges the vertical arm to the horizontal arm. Those two blocks are
    ``vp = (v_x, vp_y)`` on the vertical path and ``hp = (hp_x, h_y)`` on the horizontal path. Only
    their varying coordinates vary, so we return ``(vp_y, hp_x)``.

    Args:
        m: Zephyr grid size.
        v_x: x-coordinate of the vertical block
        v_y: y-coordinate of the vertical block
        h_x: x-coordinate of the horizontal block
        h_y: y-coordinate of the horizontal block

    Returns:
        ``(vp_y, hp_x)``, or ``None`` if there is no such elbow.
    """
    abs_max = 4 * m
    x_diff = abs(v_x - h_x)
    y_diff = abs(h_y - v_y)
    x_sign = 1 if v_x > h_x else -1
    y_sign = 1 if h_y > v_y else -1
    hp_x = v_x - x_sign if (x_diff & 3) == 1 else v_x + x_sign
    if hp_x < 0 or hp_x > abs_max:
        return None
    vp_y = h_y + y_sign if (y_diff & 3) == 3 else h_y - y_sign
    if vp_y < 0 or vp_y > abs_max:
        return None
    return (vp_y, hp_x)


def el_template(
    m: int, v_x: int, v_y: int, h_x: int, h_y: int
) -> tuple[int, int, QuoSpan, QuoSpan] | None:
    """The el-template that a vertical block and a horizontal block induce.

    Args:
        m: grid size of Zephyr
        v_x: x-coordinate of the vertical block
        v_y: y-coordinate of the vertical block
        h_x: x-coordinate of the horizontal block
        h_y: y-coordinate of the horizontal block

    Returns:
        ``None`` if the two quotient external paths have no edge between any of their blocks.
        Otherwise the elbow coordinates and the two arm spans:

        * ``vp_y`` -- y-coordinate of the block on the vertical quotient path with an edge to a
          block on the horizontal quotient path
        * ``hp_x`` -- x-coordinate of the block on the horizontal quotient path with an edge to a
          block on the vertical quotient path
        * ``v_quo`` -- vertical quo-span of the L-path from ``(v_x, v_y)`` to ``(h_x, h_y)``
        * ``h_quo`` -- horizontal quo-span of the L-path from ``(v_x, v_y)`` to ``(h_x, h_y)``
    """
    elb = elbow(m, v_x, v_y, h_x, h_y)
    if elb is None:
        return None

    vp_y, hp_x = elb
    v_a, v_b = (v_y, vp_y) if v_y < vp_y else (vp_y, v_y)
    h_a, h_b = (h_x, hp_x) if h_x < hp_x else (hp_x, h_x)
    v_quo = (v_x, v_y & 3, v_a, v_b)
    h_quo = (h_x & 3, h_y, h_a, h_b)
    return (vp_y, hp_x, v_quo, h_quo)


def el_template_length(m: int, v_quo: QuoSpan, h_quo: QuoSpan) -> int | None:
    """Block count of the el-template.

    Args:
        m: grid size of Zephyr
        v_quo: the vertical arm of el-template
        h_quo: the horizontal arm of the el-template

    Returns:
        Block count of the el-template ``(v_quo, h_quo)``, or ``None`` if the pair is
        not a real el-template.
    """
    v_x, v_y_shift, v_a, v_b = v_quo
    h_x_shift, h_y, h_a, h_b = h_quo
    for v_y in (v_a, v_b):
        if v_y & 3 != v_y_shift:
            continue
        vp_y = v_b if v_y == v_a else v_a
        for h_x in (h_a, h_b):
            if h_x & 3 != h_x_shift:
                continue
            hp_x = h_b if h_x == h_a else h_a
            if elbow(m, v_x, v_y, h_x, h_y) == (vp_y, hp_x):
                return (v_b - v_a + h_b - h_a + 8) // 4  # the number of blocks
    return None

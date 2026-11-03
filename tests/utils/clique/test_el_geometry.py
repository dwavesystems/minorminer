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


import unittest

from parameterized import parameterized

from minorminer.utils._clique.el_geometry import el_template, el_template_length, elbow

_M_CASES = [(m,) for m in (1, 2)]  # small grids


def vertical_blocks(m):
    """All vertical blocks (x even, y odd) with coords in [0, 4m]."""
    abs_max = 4 * m
    return [(x, y) for x in range(0, abs_max + 1, 2) for y in range(1, abs_max + 1, 2)]


def horizontal_blocks(m):
    """All horizontal blocks (x odd, y even) with coords in [0, 4m]."""
    abs_max = 4 * m
    return [(x, y) for x in range(1, abs_max + 1, 2) for y in range(0, abs_max + 1, 2)]


def block_pairs(m):
    """All (vertical block, horizontal block) pairs on a grid of size m."""
    for v_x, v_y in vertical_blocks(m):
        for h_x, h_y in horizontal_blocks(m):
            yield v_x, v_y, h_x, h_y


def diagonal_adjacent_family():
    """Diagonally-adjacent (``|v_x - h_x| = 1``, ``|v_y - h_y| = 1``) anchor cases: all four
    diagonal directions around a fixed interior horizontal block, at two grid
    sizes. The expected template comes from Zephyr topology (both arms collapse:
    ``vp_y = v_y``, ``hp_x = h_x``).

    Returns ``(m, v_x, v_y, h_x, h_y, expected_template)`` where
    ``expected_template = (v_y, h_x, (v_x, v_y % 4, v_y, v_y), (h_x % 4, h_y, h_x, h_x))``.
    """
    out = []
    for m, h_x, h_y in ((1, 1, 2), (12, 15, 10)):
        for v_x in (h_x - 1, h_x + 1):
            for v_y in (h_y - 1, h_y + 1):
                v_quo = (v_x, v_y % 4, v_y, v_y)
                h_quo = (h_x % 4, h_y, h_x, h_x)
                out.append((m, v_x, v_y, h_x, h_y, (v_y, h_x, v_quo, h_quo)))
    return out


# diagonally-adjacent family: (m, v_x, v_y, h_x, h_y, expected_template,),
# expected values from Zephyr topology.
KNOWN_ADJACENT = diagonal_adjacent_family()


class TestElbow(unittest.TestCase):

    # (m, v_x, v_y, h_x, h_y, expected): expected is the (vp_y, hp_x) elbow, or None
    # when the pair induces no template (crossing off-lattice).
    KNOWN_ELBOWS = [
        (3, 0, 3, 5, 8, (7, 1)),
        (6, 8, 3, 21, 12, (11, 9)),
        (6, 8, 3, 15, 12, (11, 7)),
        (1, 2, 3, 3, 2, (3, 3)),  # degenerate: single-block arms
        (7, 6, 17, 9, 24, (25, 5)),
        (4, 12, 11, 9, 12, (11, 13)),
        (1, 0, 1, 3, 0, None),  # hp_x = -1 off-lattice
        (6, 6, 17, 9, 24, None),  # vp_y = 25 > 4 * 6 off-lattice
        (3, 12, 11, 9, 12, None),  # hp_x = 13 > 4 * 3 off-lattice (+x side)
    ]

    @parameterized.expand(_M_CASES)
    def test_elbow_placement_invariants(self, m):
        """``hp_x`` sits one column off ``v_x`` (``hp_x = v_x +- 1``, hence odd), and
        ``vp_y`` sits one row off ``h_y`` (``vp_y = h_y +- 1``, hence odd)."""
        for v_x, v_y, h_x, h_y in block_pairs(m):
            res = elbow(m, v_x, v_y, h_x, h_y)
            if res is None:
                continue
            vp_y, hp_x = res
            self.assertIn(hp_x - v_x, (-1, 1))
            self.assertIn(vp_y - h_y, (-1, 1))
            self.assertEqual(hp_x % 2, 1)
            self.assertEqual(vp_y % 2, 1)

    @parameterized.expand(KNOWN_ELBOWS)
    def test_known_elbows(self, m, v_x, v_y, h_x, h_y, expected):
        self.assertEqual(elbow(m, v_x, v_y, h_x, h_y), expected)


class TestElTemplate(unittest.TestCase):

    # (m, v_x, v_y, h_x, h_y, expected): full el_template output
    # (vp_y, hp_x, v_quo, h_quo), or None for a non-template pair.
    KNOWN_TEMPLATES = [
        (3, 0, 3, 5, 8, (7, 1, (0, 3, 3, 7), (1, 8, 1, 5))),
        (6, 8, 3, 21, 12, (11, 9, (8, 3, 3, 11), (1, 12, 9, 21))),
        (6, 8, 3, 15, 12, (11, 7, (8, 3, 3, 11), (3, 12, 7, 15))),
        (1, 2, 3, 3, 2, (3, 3, (2, 3, 3, 3), (3, 2, 3, 3))),
        (12, 2, 3, 3, 2, (3, 3, (2, 3, 3, 3), (3, 2, 3, 3))),
        (7, 6, 17, 9, 24, (25, 5, (6, 1, 17, 25), (1, 24, 5, 9))),
        (4, 12, 11, 9, 12, (11, 13, (12, 3, 11, 11), (1, 12, 9, 13))),
        (1, 0, 1, 3, 0, None),
        (5, 0, 1, 3, 0, None),
        (6, 6, 17, 9, 24, None),
        (3, 12, 11, 9, 12, None),
    ]

    @parameterized.expand(_M_CASES)
    def test_el_template_agrees_with_elbow(self, m):
        """el_template is None exactly when elbow is None, and when present its
        first two elements are exactly the elbow coordinates."""
        for v_x, v_y, h_x, h_y in block_pairs(m):
            e = elbow(m, v_x, v_y, h_x, h_y)
            t = el_template(m, v_x, v_y, h_x, h_y)
            self.assertEqual(e is None, t is None)
            if t is None:
                continue
            vp_y, hp_x, _v_quo, _h_quo = t
            self.assertEqual((vp_y, hp_x), e)

    @parameterized.expand(_M_CASES)
    def test_at_least_one_template_exists(self, m):
        """Sanity: the sweep is not vacuous -- some pair induces a template."""
        found = any(
            el_template(m, v_x, v_y, h_x, h_y) is not None for v_x, v_y, h_x, h_y in block_pairs(m)
        )
        self.assertTrue(found, f"no template found on m={m}")

    @parameterized.expand(KNOWN_TEMPLATES)
    def test_known_templates(self, m, v_x, v_y, h_x, h_y, expected):
        self.assertEqual(el_template(m, v_x, v_y, h_x, h_y), expected)

    @parameterized.expand(KNOWN_ADJACENT)
    def test_known_adjacent(self, m, v_x, v_y, h_x, h_y, expected_template):
        """Diagonally-adjacent pair: full el_template output matches the
        closed form (both arms collapsed to a single block)."""
        self.assertEqual(el_template(m, v_x, v_y, h_x, h_y), expected_template)


class TestElTemplateLength(unittest.TestCase):

    # (m, v_quo, h_quo, expected_length)
    KNOWN_ACCEPT_SPANS = [
        (3, (0, 3, 3, 7), (1, 8, 1, 5), 4),
        (6, (8, 3, 3, 11), (1, 12, 9, 21), 7),
        (6, (8, 3, 3, 11), (3, 12, 7, 15), 6),
        (1, (2, 3, 3, 3), (3, 2, 3, 3), 2),  # degenerate single-block arms
        (12, (2, 3, 3, 3), (3, 2, 3, 3), 2),
        (7, (6, 1, 17, 25), (1, 24, 5, 9), 5),
        (4, (12, 3, 11, 11), (1, 12, 9, 13), 3),
    ]

    # (m, v_quo, h_quo): span pairs el_template_length must reject (-> None), one
    # per rejection path.
    KNOWN_REJECT_SPANS = [
        # off-lattice crossing (elbow returns None, nothing matches)
        (1, (0, 1, 0, 1), (3, 0, -1, 3)),
        (5, (0, 1, 0, 1), (3, 0, -1, 3)),
        # same spans that are a real template at m=7, but off-lattice at m=6
        (6, (6, 1, 17, 25), (1, 24, 5, 9)),
        # same spans that are a real template at m=4, but off-lattice at m=3
        (3, (12, 3, 11, 11), (1, 12, 9, 13)),
        # residues + bounds all pass; only the elbow re-check rejects
        (7, (6, 1, 17, 25), (1, 24, 1, 9)),
        (4, (12, 3, 11, 11), (1, 12, 5, 9)),
        # residue mismatch: shift slot (1) disagrees with the endpoints (== 3)
        (1, (2, 1, 3, 3), (3, 2, 3, 3)),
    ]

    @parameterized.expand(_M_CASES)
    def test_roundtrip_constructor_is_a_real_template(self, m):
        """Every (v_quo, h_quo) produced by el_template must be accepted by
        el_template_length (positive int), never rejected as a non-template."""
        for v_x, v_y, h_x, h_y in block_pairs(m):
            t = el_template(m, v_x, v_y, h_x, h_y)
            if t is None:
                continue
            _vp_y, _hp_x, v_quo, h_quo = t
            length = el_template_length(m, v_quo, h_quo)
            self.assertIsNotNone(length)
            self.assertIsInstance(length, int)
            self.assertGreaterEqual(length, 2)  # >=1 block per arm

    @parameterized.expand(KNOWN_ACCEPT_SPANS)
    def test_known_accept_spans(self, m, v_quo, h_quo, expected):
        self.assertEqual(el_template_length(m, v_quo, h_quo), expected)

    @parameterized.expand(KNOWN_REJECT_SPANS)
    def test_known_reject_spans(self, m, v_quo, h_quo):
        self.assertIsNone(el_template_length(m, v_quo, h_quo))

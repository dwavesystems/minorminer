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

from minorminer.utils._clique.windows import (
    CORNERS,
    candidate_windows,
    corner_orientation,
    window_bound,
)


class TestCandidateWindows(unittest.TestCase):
    def test_windows_are_well_formed(self):
        m, t = 3, 4
        for x0, x1, y0, y1 in candidate_windows(m, t, min_size=1, aspect_band=2, seed=0):
            with self.subTest(window=(x0, x1, y0, y1)):
                for coord in (x0, x1, y0, y1):
                    self.assertEqual(coord % 2, 0)
                    self.assertTrue(0 <= coord <= 4 * m)
                self.assertGreater(x1, x0)
                self.assertGreater(y1, y0)
                self.assertLessEqual(abs((x1 - x0) - (y1 - y0)), 2 * 2)

    def test_every_even_side_length(self):
        # m=2: sides 2, 4, 6, 8 have 4, 3, 2, 1 start positions per axis
        windows = candidate_windows(2, 4, min_size=1, aspect_band=0, seed=0)
        self.assertEqual(len(windows), 4 * 4 + 3 * 3 + 2 * 2 + 1)
        self.assertEqual(len(set(windows)), len(windows))
        self.assertEqual({x1 - x0 for x0, x1, _, _ in windows}, {2, 4, 6, 8})

    def test_aspect_band_adds_rectangles(self):
        square = set(candidate_windows(3, 4, min_size=1, aspect_band=0, seed=0))
        banded = set(candidate_windows(3, 4, min_size=1, aspect_band=1, seed=0))
        self.assertLess(square, banded)
        self.assertIn((0, 4, 0, 6), banded)  # one more even coordinate on the long side
        self.assertNotIn((0, 4, 0, 8), banded)
        # a band wider than the chip is clipped to the chip
        self.assertEqual(
            set(candidate_windows(2, 4, 1, aspect_band=10, seed=0)),
            set(candidate_windows(2, 4, 1, aspect_band=4, seed=0)),
        )

    def test_smallest_windows_just_reach_min_size(self):
        # the rule keeps exactly the sides whose bound 2 t n can reach min_size
        m, t = 4, 4
        for min_size in range(1, 2 * t * (2 * m + 1) + 1):
            with self.subTest(min_size=min_size):
                windows = candidate_windows(m, t, min_size, 0, seed=0)
                smallest = min(window_bound(window, t) for window in windows)
                self.assertGreaterEqual(smallest, min_size)
                side = min(x1 - x0 for x0, x1, _, _ in windows)
                if side > 2:  # one size smaller could not have reached min_size
                    self.assertLess(window_bound((0, side - 2, 0, side - 2), t), min_size)

    def test_min_size_beyond_the_chip(self):
        m, t = 3, 4
        full = 2 * t * (2 * m + 1)  # bound of the whole chip
        self.assertEqual(candidate_windows(m, t, full, 0, seed=0), [(0, 12, 0, 12)])
        self.assertEqual(candidate_windows(m, t, full + 1, 0, seed=0), [])

    def test_seed_controls_order_only(self):
        a = candidate_windows(3, 4, 1, 1, seed=1)
        self.assertEqual(a, candidate_windows(3, 4, 1, 1, seed=1))
        b = candidate_windows(3, 4, 1, 1, seed=2)
        self.assertNotEqual(a, b)
        self.assertEqual(sorted(a), sorted(b))


class TestWindowGeometry(unittest.TestCase):
    def test_window_bound(self):
        self.assertEqual(window_bound((0, 4, 0, 4), 4), 2 * 4 * 3)
        self.assertEqual(window_bound((2, 46, 2, 46), 4), 184)
        # the shorter side decides
        self.assertEqual(window_bound((0, 8, 0, 4), 2), 2 * 2 * 3)

    def test_corner_orientation(self):
        self.assertEqual(corner_orientation(("t", "l")), (-1, 1))
        self.assertEqual(corner_orientation(("t", "r")), (1, 1))
        self.assertEqual(corner_orientation(("b", "l")), (-1, -1))
        self.assertEqual(corner_orientation(("b", "r")), (1, -1))
        self.assertEqual(len(CORNERS), 4)

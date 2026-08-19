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

"""Tests for longest_strictly_increasing_subsequence."""

import unittest
from parameterized import parameterized

from minorminer.utils._clique import longest_strictly_increasing_subsequence as lsis


class TestLongestStrictlyIncreasingSubsequence(unittest.TestCase):
    """Behavior and edge cases of the returned (subsequence, indices) pair."""

    # (sequence, expected_subsequence, expected_indices). Every case here has a
    # UNIQUE longest strictly-increasing subsequence, so the expected output is
    # forced by the problem -- not by this implementation's tie resolution. That
    # keeps these as true contract tests: any correct implementation must return
    # exactly this, and a change to tie-breaking cannot make them fail.
    @parameterized.expand([
        # --- edge cases ---
        ([], [], []),
        ([42], [42], [0]),
        # --- fully increasing (whole sequence is the answer) ---
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [0, 1, 2, 3, 4]),
        ([2, 4, 6, 8], [2, 4, 6, 8], [0, 1, 2, 3]),
        ([-5, -3, -1, 0, 2], [-5, -3, -1, 0, 2], [0, 1, 2, 3, 4]),
        ([-2.5, 0.0, 1.5, 3.0], [-2.5, 0.0, 1.5, 3.0], [0, 1, 2, 3]),
        ([(0, 0), (0, 1), (1, 0), (1, 1)],
         [(0, 0), (0, 1), (1, 0), (1, 1)], [0, 1, 2, 3]),
        ([(0, 0), (1, 1), (2, 2)], [(0, 0), (1, 1), (2, 2)], [0, 1, 2]),
        # --- fully decreasing ---
        ([3, 2, 1], [1], [2]),
        # --- a leading high element that must be skipped ---
        ([9, 1, 2, 3], [1, 2, 3], [1, 2, 3]),
        ([10, 1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]),
        ([50, 1, 2, 3, 4, 5, 6],
         [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]),
        ([3.0, -1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [1, 2, 3]),
        ([15, 1, 3, 5, 7, 9, 11, 13],
         [1, 3, 5, 7, 9, 11, 13], [1, 2, 3, 4, 5, 6, 7]),
        ([(2, 0), (0, 1), (0, 2), (0, 3)],
         [(0, 1), (0, 2), (0, 3)], [1, 2, 3]),
        # --- a mid spike that must be skipped ---
        ([1, 100, 2, 3, 4, 5], [1, 2, 3, 4, 5], [0, 2, 3, 4, 5]),
        ([1, 2, 10, 3, 4], [1, 2, 3, 4], [0, 1, 3, 4]),
        # --- a dip in the middle, rise continues ---
        ([1, 2, 3, 0, 4], [1, 2, 3, 4], [0, 1, 2, 4]),
        # --- a trailing low element that must be skipped ---
        ([4, 5, 6, 7, 1], [4, 5, 6, 7], [0, 1, 2, 3]),
        ([1, 3, 5, 7, 9, 2], [1, 3, 5, 7, 9], [0, 1, 2, 3, 4]),
        ([1, 2, 3, 4, 5, 6, 7, 0],
         [1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6]),
        # --- descending prefix then a unique rise ---
        ([5, 3, 1, 2, 4, 6], [1, 2, 4, 6], [2, 3, 4, 5]),
        # --- a short early run vs a longer, unique later run ---
        ([5, 6, 1, 2, 3, 4], [1, 2, 3, 4], [2, 3, 4, 5]),
        ([7, 8, 9, 1, 2, 3, 4, 5],
         [1, 2, 3, 4, 5], [3, 4, 5, 6, 7]),
        # --- duplicates ---
        ([3, 3], [3], [1]),
        ([1, 2, 2, 4], [1, 2, 4], [0, 2, 3]),
    ])
    def test_output(self, seq, expected_sub, expected_idx):
        sub, idx = lsis(seq)
        self.assertEqual(sub, expected_sub)
        self.assertEqual(idx, expected_idx)

    def test_does_not_mutate_input(self):
        seq = [3, 1, 4, 1, 5]
        before = list(seq)
        lsis(seq)
        self.assertEqual(seq, before)

    def test_deterministic_same_input_same_output(self):
        # two calls on the same input must agree (statelessness); a single
        # golden assertion cannot catch a differing-but-valid second answer
        seq = [5, 1, 5, 2, 5, 3, 5, 4]
        self.assertEqual(lsis(seq), lsis(seq))

    def test_incomparable_elements_raise(self):
        # mixing incomparable types must surface as TypeError (documented)
        with self.assertRaises(TypeError):
            lsis([1, "a", 2])

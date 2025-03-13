# Copyright 2025 D-Wave Systems Inc.
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


import unittest
from itertools import combinations

from minorminer.utils.zephyr.plane_shift import PlaneShift


class TestPlaneShift(unittest.TestCase):
    def setUp(self) -> None:
        shifts = [
            (0, 2),
            (-3, -1),
            (-2, 0),
            (1, 1),
            (1, -3),
            (-4, 6),
            (10, 4),
            (0, 0),
        ]
        self.shifts = shifts

    def test_valid_input_runs(self) -> None:
        for shift in self.shifts:
            PlaneShift(*shift)

    def test_invalid_input_gives_error(self) -> None:
        invalid_input_types = [5, "NE", (0, 2, None), (2, 0.5), (-4, 6.0)]
        with self.assertRaises(TypeError):
            for invalid_type_ in invalid_input_types:
                PlaneShift(*invalid_type_)

        invalid_input_vals = [(4, 1), (0, 1)]
        with self.assertRaises(ValueError):
            for invalid_val_ in invalid_input_vals:
                PlaneShift(*invalid_val_)

    def test_multiply(self) -> None:
        for shift in self.shifts:
            for scale in [0, 1, 2, 5, 10, -3]:
                self.assertEqual(
                    PlaneShift(shift[0] * scale, shift[1] * scale), PlaneShift(*shift) * scale
                )
                self.assertEqual(
                    PlaneShift(shift[0] * scale, shift[1] * scale), scale * PlaneShift(*shift)
                )

    def test_add(self) -> None:
        for s0, s1 in combinations(self.shifts, 2):
            self.assertEqual(
                PlaneShift(*s0) + PlaneShift(*s1), PlaneShift(s0[0] + s1[0], s0[1] + s1[1])
            )

    def test_mul(self) -> None:
        self.assertEqual(1.5 * PlaneShift(0, -4), PlaneShift(0, -6))
        with self.assertRaises(ValueError):
            0.5 * PlaneShift(0, -6)
            PlaneShift(0, -4) * 0.75
        with self.assertRaises(TypeError):
            "hello" * PlaneShift(0, -4)

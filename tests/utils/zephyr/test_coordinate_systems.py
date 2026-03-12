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


import unittest

from parameterized import parameterized

from minorminer.utils.zephyr.coordinate_systems import (
    CartesianCoord,
    ZephyrCoord,
    cartesian_to_zephyr,
    zephyr_to_cartesian,
)


class TestCoordinateSystems(unittest.TestCase):

    def test_cartesian_to_zephyr_runs(self):
        xyks = [(0, 1), (1, 0, None), (12, -3)]
        ccoords = [CartesianCoord(*xyk) for xyk in xyks]
        for ccoord in ccoords:
            cartesian_to_zephyr(ccoord=ccoord)

    @parameterized.expand(
        [
            ((0, 0, None, 0, 0), (0, 1, None)),
            ((1, 0, None, 0, 0), (1, 0, None)),
            ((1, 6, 3, 0, 1), (5, 12, 3)),
            ]
        )
    def test_cartesian_to_zephyr(self, zcoord, ccoord):
        self.assertEqual(
            ZephyrCoord(*zcoord), cartesian_to_zephyr(CartesianCoord(*ccoord))
        )
        self.assertEqual(CartesianCoord(*ccoord), zephyr_to_cartesian(ZephyrCoord(*zcoord)))

    def test_zephyr_to_cartesian_runs(self):
        uwkjzs = [(0, 2, 4, 1, 5), (1, 3, 3, 0, 0), (1, 2, None, 1, 5)]
        zcoords = [ZephyrCoord(*uwkjz) for uwkjz in uwkjzs]
        for zcoord in zcoords:
            zephyr_to_cartesian(zcoord=zcoord)

    def test_coordinate_systems_match(self):
        valid_uwkjzs = [(0, 2, 4, 1, 5), (1, 3, 3, 0, 0), (1, 2, None, 1, 5)]
        zcoords = [ZephyrCoord(*uwkjz) for uwkjz in valid_uwkjzs]
        for zcoord in zcoords:
            ccoord = zephyr_to_cartesian(zcoord=zcoord)
            self.assertEqual(zcoord, cartesian_to_zephyr(ccoord=ccoord))

        valid_xyks = [(0, 1), (1, 0, None), (12, -3)]
        ccoords = [CartesianCoord(*xyk) for xyk in valid_xyks]
        for ccoord in ccoords:
            zcoord = cartesian_to_zephyr(ccoord=ccoord)
            self.assertEqual(ccoord, zephyr_to_cartesian(zcoord=zcoord))

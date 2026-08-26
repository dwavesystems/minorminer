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


from itertools import product
from unittest import TestCase

from minorminer.utils.zephyr.node_edge import ZNode
from minorminer.utils.zephyr.plane_shift import ZPlaneShift

from minorminer.cube_embedding._tile.square import get_num_supply_square, square_tiles, square_z_paths
from minorminer._lattice_utils import QuoTile


class TestSquareTiles(TestCase):
    def test_main(self):
        atoms_main = [ZNode((2, 3)), ZNode((3, 2))]
        zns_main = [
            zn + ZPlaneShift(-2 * i, -2 * j) for zn in atoms_main for i in (0, 1) for j in (0, 1)
        ]
        self.assertEqual(QuoTile(zns=zns_main), square_tiles["main"])

    def test_anti(self):
        atoms_anti = ZNode((3, 2)), ZNode((4, 3))
        zns_anti = [
            zn + ZPlaneShift(-2 * i, -2 * j) for zn in atoms_anti for i in (0, 1) for j in (0, 1)
        ]
        self.assertEqual(QuoTile(zns=zns_anti), square_tiles["anti"])


class TestNumSupplySquare(TestCase):
    def setUp(self) -> None:
        self.small_num = 3
        self.small_nums = [1, self.small_num]
        self.big_num = self.small_num + 1
        self.main_initial_supply_pres = {0: self.small_num, 2: self.small_num}
        self.main_initial_supply_nonpres = {3: self.small_num + 1, 4: self.small_num}
        self.anti_initial_supply_pres = {1: self.small_num + 1, 3: self.small_num}
        self.anti_initial_supply_nonpres = {3: self.small_num, 5: self.small_num}

    def test_main_prescribed(self):
        sub_kind = "main"
        prescribed = True
        for num in self.small_nums:
            self.assertTrue(
                get_num_supply_square(
                    initial_supply=self.main_initial_supply_pres,
                    sub_kind=sub_kind,
                    num=num,
                    prescribed=prescribed,
                )
            )
            self.assertFalse(
                get_num_supply_square(
                    initial_supply=self.main_initial_supply_nonpres,
                    sub_kind=sub_kind,
                    num=num,
                    prescribed=prescribed,
                )
            )
        self.assertFalse(
            get_num_supply_square(
                initial_supply=self.main_initial_supply_pres,
                sub_kind=sub_kind,
                num=self.big_num,
                prescribed=prescribed,
            )
        )

    def test_main_nonprescribed(self):
        sub_kind = "main"
        prescribed = False
        for num in self.small_nums:
            self.assertTrue(
                get_num_supply_square(
                    initial_supply=self.main_initial_supply_pres,
                    sub_kind=sub_kind,
                    num=num,
                    prescribed=prescribed,
                )
            )
            self.assertTrue(
                get_num_supply_square(
                    initial_supply=self.main_initial_supply_nonpres,
                    sub_kind=sub_kind,
                    num=num,
                    prescribed=prescribed,
                )
            )
        self.assertFalse(
            get_num_supply_square(
                initial_supply=self.main_initial_supply_pres,
                sub_kind=sub_kind,
                num=self.big_num,
                prescribed=prescribed,
            )
        )

    def test_anti_prescribed(self):
        sub_kind = "anti"
        prescribed = True
        for num in self.small_nums:
            self.assertTrue(
                get_num_supply_square(
                    initial_supply=self.anti_initial_supply_pres,
                    sub_kind=sub_kind,
                    num=num,
                    prescribed=prescribed,
                )
            )
            self.assertFalse(
                get_num_supply_square(
                    initial_supply=self.anti_initial_supply_nonpres,
                    sub_kind=sub_kind,
                    num=num,
                    prescribed=prescribed,
                )
            )
        self.assertFalse(
            get_num_supply_square(
                initial_supply=self.anti_initial_supply_pres,
                sub_kind=sub_kind,
                num=self.big_num,
                prescribed=prescribed,
            )
        )

    def test_anti_nonprescribed(self):
        sub_kind = "anti"
        prescribed = False
        for num in self.small_nums:
            self.assertTrue(
                get_num_supply_square(
                    initial_supply=self.anti_initial_supply_pres,
                    sub_kind=sub_kind,
                    num=num,
                    prescribed=prescribed,
                )
            )
            self.assertTrue(
                get_num_supply_square(
                    initial_supply=self.anti_initial_supply_nonpres,
                    sub_kind=sub_kind,
                    num=num,
                    prescribed=prescribed,
                )
            )
        self.assertFalse(
            get_num_supply_square(
                initial_supply=self.anti_initial_supply_pres,
                sub_kind=sub_kind,
                num=self.big_num,
                prescribed=prescribed,
            )
        )


class TestSquarePaths(TestCase):
    def setUp(self) -> None:
        self.small_num = 2
        self.big_num_base = self.small_num + 1

    def test_base_cases(self):
        initial_supplies = [
            {0: self.small_num, 2: self.small_num + 1},
            {7: self.small_num, 5: self.small_num},
        ]
        for sub_kind in ["main", "anti"]:
            for initial_supply in initial_supplies:
                for periodic in [True, False]:
                    for prescribed in [True, False]:
                        self.assertTrue(
                            list(
                                square_z_paths(
                                    initial_supply=initial_supply,
                                    num=self.small_num,
                                    prescribed=prescribed,
                                    sub_kind=sub_kind,
                                    periodic=periodic,
                                )
                            )
                        )
                        self.assertFalse(
                            list(
                                square_z_paths(
                                    initial_supply=initial_supply,
                                    num=self.big_num_base,
                                    prescribed=prescribed,
                                    sub_kind=sub_kind,
                                    periodic=periodic,
                                )
                            )
                        )

    def test_pres_num(self):
        pres_initial_supplies = {
            "main": [
                {0: 1, 1: 1, 2: 1, 3: 1, 5: 1, 7: 1},
            ],
            "anti": [{1: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1}],
        }
        prescribed = True
        periodic = True
        for sub_kind, L_subkind in pres_initial_supplies.items():
            for initial_supply in L_subkind:
                self.assertFalse(
                    list(
                        square_z_paths(
                            initial_supply=initial_supply,
                            num=3,
                            prescribed=prescribed,
                            sub_kind=sub_kind,
                            periodic=periodic,
                        )
                    )
                )
                self.assertTrue(
                    list(
                        square_z_paths(
                            initial_supply=initial_supply,
                            num=2,
                            prescribed=prescribed,
                            sub_kind=sub_kind,
                            periodic=periodic,
                        )
                    )
                )

    def test_per_num(self):
        periodic = True
        num_ch = 3
        per_initial_supplies_pres = {
            "main": [
                {0: num_ch, 1: 1, 2: num_ch, 3: 1, 5: 1, 7: 1},
                {0: 1, 1: 1, 2: 1, 3: 1, 5: num_ch, 7: num_ch},
            ],
            "anti": [
                {1: num_ch, 3: num_ch, 4: 1, 5: 1, 6: 1, 7: 1},
                {1: 1, 3: 1, 4: num_ch, 5: 1, 6: num_ch, 7: 1},
            ],
        }
        for sub_kind, L_subkind in per_initial_supplies_pres.items():
            for initial_supply in L_subkind:
                for prescribed in [True, False]:
                    self.assertFalse(
                        list(
                            square_z_paths(
                                initial_supply=initial_supply,
                                num=num_ch + 2,
                                prescribed=prescribed,
                                sub_kind=sub_kind,
                                periodic=periodic,
                            )
                        )
                    )
                    self.assertTrue(
                        list(
                            square_z_paths(
                                initial_supply=initial_supply,
                                num=num_ch + 1,
                                prescribed=prescribed,
                                sub_kind=sub_kind,
                                periodic=periodic,
                            )
                        )
                    )

    def test_pres_nonper(self):
        num_ch = 3
        nonper_initial_supplies_nonpres = {
            "main": [
                {0: num_ch, 2: num_ch, 3: 1, 4: 1, 5: 1, 7: 1},
                {0: 1, 2: 1, 3: 1, 4: 1, 5: num_ch, 7: num_ch},
            ],
            "anti": [
                {1: num_ch, 2: 1, 3: num_ch, 4: 1, 5: 1, 6: 1},
                {1: 1, 2: 1, 3: 1, 4: num_ch, 5: 1, 6: num_ch},
            ],
        }
        periodic = False
        for sub_kind, L_subkind in nonper_initial_supplies_nonpres.items():
            for initial_supply in L_subkind:
                self.assertFalse(
                    list(
                        square_z_paths(
                            initial_supply=initial_supply,
                            num=num_ch + 2,
                            prescribed=True,
                            sub_kind=sub_kind,
                            periodic=periodic,
                        )
                    )
                )

                self.assertTrue(
                    list(
                        square_z_paths(
                            initial_supply=initial_supply,
                            num=num_ch + 2,
                            prescribed=False,
                            sub_kind=sub_kind,
                            periodic=periodic,
                        )
                    )
                )

    def test_nonpres_per(self):
        per_initial_supplies_nonpres = {
            "main": [
                {3: 1, 4: 2, 5: 1, 7: 2},
                {1: 1, 3: 1, 4: 1, 5: 1, 7: 2},
            ],
            "anti": [
                {1: 1, 2: 2, 4: 2, 6: 1},
                {0: 1, 2: 2, 5: 2, 7: 1},
            ],
        }
        for sub_kind, L_subkind in per_initial_supplies_nonpres.items():
            for initial_supply in L_subkind:
                for periodic in [True, False]:
                    self.assertFalse(
                        list(
                            square_z_paths(
                                initial_supply=initial_supply,
                                num=3,
                                prescribed=True,
                                sub_kind=sub_kind,
                                periodic=periodic,
                            )
                        )
                    )

                    self.assertTrue(
                        list(
                            square_z_paths(
                                initial_supply=initial_supply,
                                num=3,
                                prescribed=False,
                                sub_kind=sub_kind,
                                periodic=periodic,
                            )
                        )
                    )

    def test_pres(self):
        per_initial_supplies_pres = {
            "main": [
                {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1},
                {0: 2, 1: 2, 2: 2, 3: 2},
                {0: 2, 1: 1, 2: 2, 3: 1, 4: 1, 6: 1},
            ],
            "anti": [
                {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1},
                {0: 2, 1: 2, 2: 2, 3: 2},
                {0: 1, 1: 2, 2: 1, 3: 2, 5: 1, 7: 1},
            ],
        }
        for sub_kind, L_subkind in per_initial_supplies_pres.items():
            for initial_supply in L_subkind:
                for periodic in [True, False]:
                    for prescibed in [True, False]:
                        self.assertTrue(
                            list(
                                square_z_paths(
                                    initial_supply=initial_supply,
                                    num=4,
                                    prescribed=prescibed,
                                    sub_kind=sub_kind,
                                    periodic=periodic,
                                )
                            )
                        )

    def test_disc(self):
        for num_ch1, num_ch2, periodic, prescribed in product(
            [1, 2], [1, 3], [True, False], [True, False]
        ):
            initial_supply_main = {0: num_ch1, 2: num_ch1, 5: num_ch2, 7: num_ch2}
            self.assertFalse(
                list(
                    square_z_paths(
                        initial_supply=initial_supply_main,
                        num=num_ch1 + num_ch2,
                        prescribed=prescribed,
                        sub_kind="main",
                        periodic=periodic,
                    )
                )
            )
            initial_supply_anti = {1: num_ch1, 3: num_ch1, 4: num_ch2, 6: num_ch2}
            self.assertFalse(
                list(
                    square_z_paths(
                        initial_supply=initial_supply_anti,
                        num=num_ch1 + num_ch2,
                        prescribed=prescribed,
                        sub_kind="anti",
                        periodic=periodic,
                    )
                )
            )

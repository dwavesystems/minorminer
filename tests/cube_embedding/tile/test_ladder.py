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



from unittest import TestCase

from minorminer.utils.zephyr.node_edge import ZNode
from minorminer.utils.zephyr.plane_shift import ZPlaneShift

from minorminer.cube_embedding._tile.ladder import get_num_supply_ladder, ladder_tiles, ladder_z_paths
from minorminer._lattice_utils import QuoTile


class TestLadderTiles(TestCase):
    def test_main(self):
        atom0, atom1 = ZNode((3, 4)), ZNode((4, 3))
        zns = [atom + k * ZPlaneShift(-1, -1) for atom in [atom0, atom1] for k in range(4)]
        self.assertEqual(QuoTile(zns=zns), ladder_tiles["main"])

    def test_anti(self):
        atom0, atom1 = ZNode((0, 3)), ZNode((1, 4))
        zns = [atom + k * ZPlaneShift(1, -1) for atom in [atom0, atom1] for k in range(4)]
        self.assertEqual(QuoTile(zns=zns), ladder_tiles["anti"])


class TestNumSupplyLadder(TestCase):
    def setUp(self) -> None:
        self.small_num = 3
        self.small_nums = [1, self.small_num]
        self.big_num = self.small_num + 1
        self.main_initial_supply_pres = {0: self.small_num, 1: self.small_num}
        self.main_initial_supply_nonpres = {2: self.small_num + 1, 4: self.small_num}
        self.anti_initial_supply_pres = {3: self.small_num + 1, 6: self.small_num}
        self.anti_initial_supply_nonpres = {1: self.small_num, 3: self.small_num}

    def test_main_prescribed(self):
        sub_kind = "main"
        prescribed = True
        for num in self.small_nums:
            self.assertTrue(
                get_num_supply_ladder(
                    initial_supply=self.main_initial_supply_pres,
                    sub_kind=sub_kind,
                    num=num,
                    prescribed=prescribed,
                )
            )
            self.assertFalse(
                get_num_supply_ladder(
                    initial_supply=self.main_initial_supply_nonpres,
                    sub_kind=sub_kind,
                    num=num,
                    prescribed=prescribed,
                )
            )
        self.assertFalse(
            get_num_supply_ladder(
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
                get_num_supply_ladder(
                    initial_supply=self.main_initial_supply_pres,
                    sub_kind=sub_kind,
                    num=num,
                    prescribed=prescribed,
                )
            )
            self.assertTrue(
                get_num_supply_ladder(
                    initial_supply=self.main_initial_supply_nonpres,
                    sub_kind=sub_kind,
                    num=num,
                    prescribed=prescribed,
                )
            )
        self.assertFalse(
            get_num_supply_ladder(
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
                get_num_supply_ladder(
                    initial_supply=self.anti_initial_supply_pres,
                    sub_kind=sub_kind,
                    num=num,
                    prescribed=prescribed,
                )
            )
            self.assertFalse(
                get_num_supply_ladder(
                    initial_supply=self.anti_initial_supply_nonpres,
                    sub_kind=sub_kind,
                    num=num,
                    prescribed=prescribed,
                )
            )
        self.assertFalse(
            get_num_supply_ladder(
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
                get_num_supply_ladder(
                    initial_supply=self.anti_initial_supply_pres,
                    sub_kind=sub_kind,
                    num=num,
                    prescribed=prescribed,
                )
            )
            self.assertTrue(
                get_num_supply_ladder(
                    initial_supply=self.anti_initial_supply_nonpres,
                    sub_kind=sub_kind,
                    num=num,
                    prescribed=prescribed,
                )
            )
        self.assertFalse(
            get_num_supply_ladder(
                initial_supply=self.anti_initial_supply_pres,
                sub_kind=sub_kind,
                num=self.big_num,
                prescribed=prescribed,
            )
        )


class TestLadderPaths(TestCase):
    def setUp(self) -> None:
        self.small_num = 2
        self.big_num_base = self.small_num + 1

    def test_base_cases(self):
        initial_supplies = {
            "main": [
                {0: self.small_num, 1: self.small_num + 1},
                {4: self.small_num, 5: self.small_num},
            ],
            "anti": [
                {1: self.small_num, 4: self.small_num},
                {5: self.small_num + 1, 7: self.small_num},
            ],
        }
        for sub_kind, L_subkind in initial_supplies.items():
            for initial_supply in L_subkind:
                for periodic in [True, False]:
                    for prescribed in [True, False]:
                        self.assertTrue(
                            list(
                                ladder_z_paths(
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
                                ladder_z_paths(
                                    initial_supply=initial_supply,
                                    num=self.big_num_base,
                                    prescribed=prescribed,
                                    sub_kind=sub_kind,
                                    periodic=periodic,
                                )
                            )
                        )

    def test_connected_pres(self):
        pres_initial_supplies = {
            "main": [{0: 1, 1: 1, 2: 1, 3: 1, 6: 1, 7: 1}, {0: 2, 1: 2, 4: 1, 5: 1}],
            "anti": [{0: 1, 1: 1, 2: 1, 4: 1, 5: 1, 7: 1}, {0: 1, 2: 1, 5: 2, 7: 2}],
        }
        for sub_kind, L_subkind in pres_initial_supplies.items():
            for initial_supply in L_subkind:
                for periodic in [True, False]:
                    for prescribed in [True, False]:
                        self.assertFalse(
                            list(
                                ladder_z_paths(
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
                                ladder_z_paths(
                                    initial_supply=initial_supply,
                                    num=2,
                                    prescribed=prescribed,
                                    sub_kind=sub_kind,
                                    periodic=periodic,
                                )
                            ),
                            msg=f"No prescribed chains of length 2 with {sub_kind = },  {initial_supply = }, {periodic = }, {prescribed = }",
                        )

        nonpres_initial_supplies = {
            "main": [{0: 1, 1: 1, 4: 1, 5: 1, 6: 1, 7: 1}, {0: 2, 2: 2, 5: 1, 7: 1}],
            "anti": [{0: 1, 2: 1, 3: 1, 5: 1, 6: 1, 7: 1}, {2: 2, 4: 2, 5: 2, 7: 2}],
        }
        for sub_kind, L_subkind in nonpres_initial_supplies.items():
            for initial_supply in L_subkind:
                for periodic in [True, False]:
                    for prescribed in [False]:
                        self.assertFalse(
                            list(
                                ladder_z_paths(
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
                                ladder_z_paths(
                                    initial_supply=initial_supply,
                                    num=2,
                                    prescribed=prescribed,
                                    sub_kind=sub_kind,
                                    periodic=periodic,
                                )
                            )
                        )

    def test_base_cases_nonprescribed(self):
        initial_supplies = {
            "main": [
                {1: self.small_num, 3: self.small_num + 1},  # only non-prescribed
                {2: self.small_num, 4: self.small_num},  # only non-prescribed
                {4: self.small_num, 5: self.big_num_base},  # prescribed and non-prescribed
            ],
            "anti": [
                {1: self.small_num, 0: self.small_num},  # only non-prescribed
                {6: self.small_num + 1, 7: self.small_num},  # only non-prescribed
                {0: self.small_num, 2: self.big_num_base},  # prescribed and non-prescribed
            ],
        }
        prescribed = False
        for sub_kind, L_subkind in initial_supplies.items():
            for initial_supply in L_subkind:
                for periodic in [True, False]:

                    self.assertTrue(
                        list(
                            ladder_z_paths(
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
                            ladder_z_paths(
                                initial_supply=initial_supply,
                                num=self.big_num_base,
                                prescribed=prescribed,
                                sub_kind=sub_kind,
                                periodic=periodic,
                            )
                        )
                    )

    def test_base_cases_prescribed(self):
        pres_initial_supplies = {
            "main": [
                {4: self.small_num, 5: self.small_num + 1},
                {0: self.small_num, 1: self.small_num},
            ],
            "anti": [
                {0: self.small_num, 2: self.small_num},
                {1: self.small_num + 1, 4: self.small_num},
            ],
        }
        nonpres_initial_supplies = {
            "main": [
                {1: self.small_num, 3: self.small_num + 1},  # non-prescribed
                {2: self.small_num, 4: self.small_num},  # non-prescribed
            ],
            "anti": [
                {1: self.small_num, 0: self.small_num},  # non-prescribed
                {6: self.small_num, 7: self.small_num + 1},  # non-prescribed
            ],
        }
        prescribed = True
        for sub_kind, L_subkind in pres_initial_supplies.items():
            for initial_supply in L_subkind:
                for periodic in [True, False]:
                    self.assertTrue(
                        list(
                            ladder_z_paths(
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
                            ladder_z_paths(
                                initial_supply=initial_supply,
                                num=self.big_num_base,
                                prescribed=prescribed,
                                sub_kind=sub_kind,
                                periodic=periodic,
                            )
                        )
                    )
        for sub_kind, L_subkind in nonpres_initial_supplies.items():
            for initial_supply in L_subkind:
                for periodic in [True, False]:
                    self.assertFalse(
                        list(
                            ladder_z_paths(
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
                            ladder_z_paths(
                                initial_supply=initial_supply,
                                num=self.big_num_base,
                                prescribed=prescribed,
                                sub_kind=sub_kind,
                                periodic=periodic,
                            )
                        )
                    )

    def test_base_cases_nonperiodic(self):
        per_initial_supplies_nonpres = {
            "main": [
                {1: 1, 2: 1, 3: 1, 4: 1},  # periodic and non-periodic
                {0: 1, 1: 1, 2: 1, 4: 1},  # periodic and non-periodic
                {2: self.small_num, 4: self.big_num_base},  # periodic and non-periodic
            ],
            "anti": [
                {4: 1, 5: 1, 6: 1, 7: 1},  # periodic and non-periodic
                {0: 1, 1: 1, 3: 1, 5: 1},  # periodic and non-periodic
                {0: self.small_num, 1: self.big_num_base},  # periodic and non-periodic
            ],
        }
        periodic = False
        for sub_kind, L_subkind in per_initial_supplies_nonpres.items():
            for initial_supply in L_subkind:
                self.assertTrue(
                    list(
                        ladder_z_paths(
                            initial_supply=initial_supply,
                            num=self.small_num,
                            prescribed=False,
                            sub_kind=sub_kind,
                            periodic=periodic,
                        )
                    )
                )

                self.assertFalse(
                    list(
                        ladder_z_paths(
                            initial_supply=initial_supply,
                            num=self.big_num_base,
                            prescribed=False,
                            sub_kind=sub_kind,
                            periodic=periodic,
                        )
                    )
                )
                self.assertFalse(
                    list(
                        ladder_z_paths(
                            initial_supply=initial_supply,
                            num=self.small_num,
                            prescribed=True,
                            sub_kind=sub_kind,
                            periodic=periodic,
                        )
                    )
                )

        nonper_initial_supplies_nonpres = {
            "main": [
                {1: 1, 2: 1, 3: 1, 4: 1, 6: 1, 7: 1},  # only non-periodic
                {0: 1, 1: 1, 2: 1, 4: 1, 5: 1, 7: 1},  # only non-periodic
            ],
            "anti": [
                {0: 1, 2: 1, 4: 1, 5: 1, 6: 1, 7: 1},  # only non-periodic
            ],
        }
        for sub_kind, L_subkind in nonper_initial_supplies_nonpres.items():
            for initial_supply in L_subkind:
                self.assertTrue(
                    list(
                        ladder_z_paths(
                            initial_supply=initial_supply,
                            num=3,
                            prescribed=False,
                            sub_kind=sub_kind,
                            periodic=periodic,
                        )
                    )
                )
                self.assertFalse(
                    list(
                        ladder_z_paths(
                            initial_supply=initial_supply,
                            num=3,
                            prescribed=True,
                            sub_kind=sub_kind,
                            periodic=periodic,
                        )
                    )
                )

    def test_base_cases_periodic(self):
        nonper_initial_supplies_nonpres = {
            "main": [
                {1: 1, 2: 1, 3: 1, 4: 1, 6: 1, 7: 1},  # only non-periodic
                {0: 1, 1: 1, 2: 1, 4: 1, 5: 1, 7: 1},  # only non-periodic
            ],
            "anti": [
                {0: 1, 2: 1, 4: 1, 5: 1, 6: 1, 7: 1},  # only non-periodic
                {0: 1, 1: 1, 2: 1, 4: 1, 6: 1, 7: 1},  # only non-periodic
            ],
        }

        periodic = True
        for sub_kind, L_subkind in nonper_initial_supplies_nonpres.items():
            for initial_supply in L_subkind:
                self.assertFalse(
                    list(
                        ladder_z_paths(
                            initial_supply=initial_supply,
                            num=3,
                            prescribed=False,
                            sub_kind=sub_kind,
                            periodic=periodic,
                        )
                    )
                )

        per_initial_supplies_nonpres = {
            "main": [
                {2: self.big_num_base, 4: self.big_num_base},  # periodic and non-periodic
            ],
            "anti": [
                {0: self.big_num_base, 1: self.big_num_base},  # periodic and non-periodic
                {0: 1, 1: 1, 3: 1, 4: 1, 5: 1, 6: 1},  # periodic and non-periodic
            ],
        }
        for sub_kind, L_subkind in per_initial_supplies_nonpres.items():
            for initial_supply in L_subkind:
                self.assertTrue(
                    list(
                        ladder_z_paths(
                            initial_supply=initial_supply,
                            num=self.big_num_base,
                            prescribed=False,
                            sub_kind=sub_kind,
                            periodic=periodic,
                        )
                    )
                )
                self.assertFalse(
                    list(
                        ladder_z_paths(
                            initial_supply=initial_supply,
                            num=self.big_num_base,
                            prescribed=True,
                            sub_kind=sub_kind,
                            periodic=periodic,
                        )
                    )
                )

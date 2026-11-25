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

from minorminer.utils.zephyr.node_edge import Edge

from minorminer.cube_embedding._tile.kind import ZCoupling
from minorminer.cube_embedding._tile.ladder import all_ladder_idx_chains, ladder_tiles
from minorminer.cube_embedding._tile.square import all_square_idx_chains, square_tiles
from minorminer.cube_embedding._tile.z_path import EdgePos, parse_seq


class TestParsePath(TestCase):
    @staticmethod
    def bond_inverse(bond_pos: EdgePos):
        return EdgePos(edge=bond_pos.edge[::-1], pos=bond_pos.pos[::-1])

    @staticmethod
    def normalized_ibp(bond: tuple[int, int], pos: tuple[int, int]) -> EdgePos:
        edge = Edge(*bond)
        if (edge[0], edge[1]) == bond:
            return EdgePos(edge=edge, pos=pos)
        elif (edge[1], edge[0]) == bond:
            return EdgePos(edge=edge, pos=pos[::-1])

    @staticmethod
    def lad_normalized(bond: tuple[int], sub_kind: str):
        if sub_kind == "main":
            return bond
        lad_main_to_anti_idx = {6: 0, 7: 2, 4: 1, 5: 4, 2: 3, 3: 6, 0: 5, 1: 7}
        return (lad_main_to_anti_idx[bond[0]], lad_main_to_anti_idx[bond[1]])

    def setUp(self) -> None:
        self.lad_idx_to_zedge = all_ladder_idx_chains
        self.sq_idx_to_zedge = all_square_idx_chains
        self.sq_indices = {
            sub_kind: {i: node.node_kind for i, node in enumerate(tile.zns)}
            for sub_kind, tile in square_tiles.items()
        }
        self.lad_indices = {
            sub_kind: {i: node.node_kind for i, node in enumerate(tile.zns)}
            for sub_kind, tile in ladder_tiles.items()
        }

    def test_lad_main_disconnected(self):
        lad_main_disc = [[Edge(0, 1), Edge(0, 1), Edge(4, 6)], [Edge(0, 1), Edge(4, 5), Edge(5, 7)]]
        sub_kind = "main"
        for disc_path in lad_main_disc:
            for closed in [True, False]:
                with self.assertRaises(StopIteration):
                    next(
                        parse_seq(
                            seq=disc_path,
                            idx_chains=self.lad_idx_to_zedge[sub_kind],
                            periodic=closed,
                            indices=self.lad_indices[sub_kind],
                            z_coupling=ZCoupling.EITHER,
                        )
                    )

    def test_lad_anti_disconnected(self):
        lad_anti_disc = [[Edge(6, 7), Edge(5, 7), Edge(0, 1)], [Edge(0, 2), Edge(2, 4), Edge(3, 5)]]
        sub_kind = "anti"
        for disc_path in lad_anti_disc:
            for closed in [True, False]:
                with self.assertRaises(StopIteration):
                    next(
                        parse_seq(
                            seq=disc_path,
                            idx_chains=self.lad_idx_to_zedge[sub_kind],
                            periodic=closed,
                            indices=self.lad_indices[sub_kind],
                            z_coupling=ZCoupling.EITHER,
                        )
                    )

    def test_sq_main_disconnected(self):
        disc_path = [Edge(0, 2), Edge(0, 2), Edge(5, 7)]
        sub_kind = "main"
        for closed in [True, False]:
            with self.assertRaises(StopIteration):
                next(
                    parse_seq(
                        seq=disc_path,
                        idx_chains=self.sq_idx_to_zedge[sub_kind],
                        periodic=closed,
                        indices=self.sq_indices[sub_kind],
                        z_coupling=ZCoupling.EITHER,
                    )
                )

    def test_sq_anti_disconnected(self):
        disc_path = [Edge(1, 3), Edge(4, 6), Edge(4, 6)]
        sub_kind = "anti"
        for closed in [True, False]:
            with self.assertRaises(StopIteration):
                next(
                    parse_seq(
                        seq=disc_path,
                        idx_chains=self.sq_idx_to_zedge[sub_kind],
                        periodic=closed,
                        indices=self.sq_indices[sub_kind],
                        z_coupling=ZCoupling.EITHER,
                    )
                )

    def test_sq_main_connected(self):
        sq_main_con = [
            [Edge(3, 4)],
            [Edge(2, 4), Edge(4, 6), Edge(1, 3)],
            [Edge(0, 3), Edge(3, 5), Edge(4, 6), Edge(4, 6)],
        ]
        sub_kind = "main"
        for con_path in sq_main_con:
            for closed in [True, False]:
                next(
                    parse_seq(
                        seq=con_path,
                        idx_chains=self.sq_idx_to_zedge[sub_kind],
                        periodic=closed,
                        indices=self.sq_indices[sub_kind],
                        z_coupling=ZCoupling.EITHER,
                    )
                )

    def test_sq_anti_connected(self):
        sq_anti_con = [
            [Edge(1, 3)],
            [Edge(0, 2), Edge(2, 4), Edge(3, 5)],
            [Edge(5, 7), Edge(5, 6), Edge(1, 2)],
        ]
        sub_kind = "anti"
        for con_path in sq_anti_con:
            for closed in [True, False]:
                next(
                    parse_seq(
                        seq=con_path,
                        idx_chains=self.sq_idx_to_zedge[sub_kind],
                        periodic=closed,
                        indices=self.sq_indices[sub_kind],
                        z_coupling=ZCoupling.EITHER,
                    )
                )

    def test_lad_main_closed(self):
        lad_main_con_closed = [
            [Edge(4, 5)],
            [Edge(0, 2), Edge(3, 5)],
            [Edge(0, 1), Edge(2, 3), Edge(4, 5), Edge(6, 7), Edge(3, 5)],
        ]
        sub_kind = "main"
        for con_path in lad_main_con_closed:
            for closed in [True, False]:
                next(
                    parse_seq(
                        seq=con_path,
                        idx_chains=self.lad_idx_to_zedge[sub_kind],
                        periodic=closed,
                        indices=self.lad_indices[sub_kind],
                        z_coupling=ZCoupling.EITHER,
                    )
                )

    def test_lad_main_nonclosed(self):
        lad_main_con_nonclosed = [
            [Edge(0, 1), Edge(2, 4), Edge(5, 7)],
            [Edge(0, 2), Edge(3, 5)],
            [Edge(0, 1), Edge(2, 3), Edge(4, 5), Edge(6, 7)],
        ]
        sub_kind = "main"
        for con_path in lad_main_con_nonclosed:
            next(
                parse_seq(
                    seq=con_path,
                    idx_chains=self.lad_idx_to_zedge[sub_kind],
                    periodic=False,
                    indices=self.lad_indices[sub_kind],
                    z_coupling=ZCoupling.EITHER,
                )
            )
            if len(con_path) > 2:
                with self.assertRaises(StopIteration):
                    next(
                        parse_seq(
                            seq=con_path,
                            idx_chains=self.lad_idx_to_zedge[sub_kind],
                            periodic=True,
                            indices=self.lad_indices[sub_kind],
                            z_coupling=ZCoupling.EITHER,
                        )
                    )

    def test_lad_anti_closed(self):
        lad_anti_con_closed = [
            [Edge(3, 5)],
            [Edge(0, 2), Edge(1, 4)],
            [Edge(0, 1), Edge(3, 5), Edge(4, 6), Edge(2, 4)],
        ]
        sub_kind = "anti"
        for con_path in lad_anti_con_closed:
            for closed in [True, False]:
                next(
                    parse_seq(
                        seq=con_path,
                        idx_chains=self.lad_idx_to_zedge[sub_kind],
                        periodic=closed,
                        indices=self.lad_indices[sub_kind],
                        z_coupling=ZCoupling.EITHER,
                    )
                )

    def test_lad_anti_nonclosed(self):
        lad_anti_con_nonclosed = [
            [Edge(0, 2), Edge(4, 6), Edge(3, 5)],
            [Edge(0, 2), Edge(1, 4), Edge(6, 7)],
        ]
        sub_kind = "anti"
        for con_path in lad_anti_con_nonclosed:
            next(
                parse_seq(
                    seq=con_path,
                    idx_chains=self.lad_idx_to_zedge[sub_kind],
                    periodic=False,
                    indices=self.lad_indices[sub_kind],
                    z_coupling=ZCoupling.EITHER,
                )
            )
            if len(con_path) > 2:
                with self.assertRaises(StopIteration):
                    next(
                        parse_seq(
                            seq=con_path,
                            idx_chains=self.lad_idx_to_zedge[sub_kind],
                            periodic=True,
                            indices=self.lad_indices[sub_kind],
                            z_coupling=ZCoupling.EITHER,
                        )
                    )

    def test_sq_main_closed(self):
        con_path = [Edge(4, 6), Edge(3, 4), Edge(1, 3)]
        sub_kind = "main"
        for closed in [True, False]:
            next(
                parse_seq(
                    seq=con_path,
                    idx_chains=self.sq_idx_to_zedge[sub_kind],
                    periodic=closed,
                    indices=self.sq_indices[sub_kind],
                    z_coupling=ZCoupling.EITHER,
                )
            )

    def test_sq_main_nonclosed(self):
        nonclosed = [Edge(0, 2), Edge(3, 4), Edge(5, 7)]
        sub_kind = "main"
        next(
            parse_seq(
                seq=nonclosed,
                idx_chains=self.sq_idx_to_zedge[sub_kind],
                periodic=False,
                indices=self.sq_indices[sub_kind],
                z_coupling=ZCoupling.EITHER,
            )
        )
        with self.assertRaises(StopIteration):
            next(
                parse_seq(
                    seq=nonclosed,
                    idx_chains=self.sq_idx_to_zedge[sub_kind],
                    periodic=True,
                    indices=self.sq_indices[sub_kind],
                    z_coupling=ZCoupling.EITHER,
                )
            )

    def test_sq_anti_closed(self):
        sq_anti_closed = [
            [Edge(4, 6), Edge(3, 5), Edge(1, 2)],
            [Edge(0, 2), Edge(2, 5), Edge(5, 7)],
            [Edge(0, 2), Edge(1, 2), Edge(4, 6)],
        ]
        sub_kind = "anti"
        for closed_seq in sq_anti_closed:
            for closed in [True, False]:
                next(
                    parse_seq(
                        seq=closed_seq,
                        idx_chains=self.sq_idx_to_zedge[sub_kind],
                        periodic=closed,
                        indices=self.sq_indices[sub_kind],
                        z_coupling=ZCoupling.EITHER,
                    )
                )

    def test_sq_anti_nonclosed(self):
        sq_anti_con_nonclosed = [Edge(4, 6), Edge(5, 7), Edge(1, 3)]
        sub_kind = "anti"
        next(
            parse_seq(
                seq=sq_anti_con_nonclosed,
                idx_chains=self.sq_idx_to_zedge[sub_kind],
                periodic=False,
                indices=self.sq_indices[sub_kind],
                z_coupling=ZCoupling.EITHER,
            )
        )
        with self.assertRaises(StopIteration):
            next(
                parse_seq(
                    seq=sq_anti_con_nonclosed,
                    idx_chains=self.sq_idx_to_zedge[sub_kind],
                    periodic=True,
                    indices=self.sq_indices[sub_kind],
                    z_coupling=ZCoupling.EITHER,
                )
            )

    def test_ladder_base_periodic(self):
        for sub_kind in ["main", "anti"]:
            path = [self.lad_normalized((2 * i, 2 * i + 1), sub_kind=sub_kind) for i in range(4)]
            with self.assertRaises(StopIteration):
                next(
                    parse_seq(
                        seq=path,
                        idx_chains=self.lad_idx_to_zedge[sub_kind],
                        periodic=True,
                        indices=self.lad_indices[sub_kind],
                        z_coupling=ZCoupling.EITHER,
                    )
                )

    def test_ladder_base_nonperiodic(self):
        for sub_kind in ["main", "anti"]:
            path = [
                Edge(*self.lad_normalized((2 * i, 2 * i + 1), sub_kind=sub_kind)) for i in range(4)
            ]
            expected_crucial = [
                self.normalized_ibp(
                    bond=self.lad_normalized((2 * i, 2 * i + 1), sub_kind=sub_kind), pos=(0, 0)
                )
                for i in range(4)
            ]
            periodic = False
            path_parsed = next(
                parse_seq(
                    seq=path,
                    idx_chains=self.lad_idx_to_zedge[sub_kind],
                    periodic=periodic,
                    indices=self.lad_indices[sub_kind],
                    z_coupling=ZCoupling.EITHER,
                )
            )
            crucial_bonds_pos = path_parsed.crucial_edge_pos
            self.assertEqual(len(crucial_bonds_pos), 4)
            alt_chain_bonds_pos = path_parsed.alt_edge_pos
            self.assertEqual(len(alt_chain_bonds_pos), 6)
            for bond_pos in expected_crucial:
                self.assertTrue(
                    bond_pos in crucial_bonds_pos
                    or self.bond_inverse(bond_pos) in crucial_bonds_pos,
                    msg=f"Neither {bond_pos} nor {self.bond_inverse(bond_pos)} in {crucial_bonds_pos}",
                )
            for i in range(3):
                self.assertTrue(
                    self.normalized_ibp(
                        bond=self.lad_normalized((2 * i, 2 * i + 2), sub_kind=sub_kind), pos=(0, 0)
                    ),
                    self.normalized_ibp(
                        bond=self.lad_normalized((2 * i + 1, 2 * i + 3), sub_kind=sub_kind),
                        pos=(0, 0),
                    ),
                ) in alt_chain_bonds_pos.items()
                self.assertTrue(
                    self.normalized_ibp(
                        bond=self.lad_normalized((2 * i + 1, 2 * i + 3), sub_kind=sub_kind),
                        pos=(0, 0),
                    ),
                    self.normalized_ibp(
                        bond=self.lad_normalized((2 * i, 2 * i + 2), sub_kind=sub_kind), pos=(0, 0)
                    ),
                ) in alt_chain_bonds_pos.items()

    def test_ladder_main(self):
        path = [Edge(0, 1), Edge(1, 3), Edge(2, 4)]
        sub_kind = "main"
        expected_crucial = [
            self.normalized_ibp(bond=(0, 1), pos=(0, 0)),
            self.normalized_ibp(bond=(3, 1), pos=(0, 1)),
            self.normalized_ibp(bond=(4, 2), pos=(0, 0)),
            self.normalized_ibp(bond=(3, 2), pos=(0, 0)),
        ]
        for periodic in [True, False]:
            path_parsed = next(
                parse_seq(
                    seq=path,
                    idx_chains=self.lad_idx_to_zedge[sub_kind],
                    periodic=periodic,
                    indices=self.lad_indices[sub_kind],
                    z_coupling=ZCoupling.EITHER,
                )
            )
            crucial_bonds_pos = path_parsed.crucial_edge_pos
            alt_chain_bonds_pos = path_parsed.alt_edge_pos
            for bond_pos in expected_crucial:
                self.assertTrue(bond_pos in crucial_bonds_pos)
            if periodic:
                self.assertTrue(self.normalized_ibp(bond=(0, 2), pos=(0, 0)) in crucial_bonds_pos)
            else:
                self.assertFalse(self.normalized_ibp(bond=(0, 2), pos=(0, 0)) in crucial_bonds_pos)
            self.assertTrue(
                self.normalized_ibp(bond=(0, 1), pos=(0, 1)),
                self.normalized_ibp(bond=(3, 1), pos=(0, 0)) in alt_chain_bonds_pos.items(),
            )

    def test_ladder_anti(self):
        path = [Edge(0, 2), Edge(4, 2), Edge(4, 6), Edge(3, 5), Edge(3, 1)]
        sub_kind = "anti"
        expected_crucial = [
            self.normalized_ibp(bond=(0, 2), pos=(0, 0)),
            self.normalized_ibp(bond=(4, 2), pos=(0, 1)),
            self.normalized_ibp(bond=(4, 6), pos=(1, 0)),
            self.normalized_ibp(bond=(3, 6), pos=(0, 0)),
            self.normalized_ibp(bond=(3, 5), pos=(0, 0)),
            self.normalized_ibp(bond=(3, 1), pos=(1, 0)),
        ]
        expected_alt = [
            (
                self.normalized_ibp(bond=(4, 2), pos=(1, 1)),
                self.normalized_ibp(bond=(4, 6), pos=(0, 0)),
            ),
            (
                self.normalized_ibp(bond=(0, 2), pos=(0, 1)),
                self.normalized_ibp(bond=(4, 2), pos=(0, 0)),
            ),
            (
                self.normalized_ibp(bond=(3, 1), pos=(0, 0)),
                self.normalized_ibp(bond=(3, 5), pos=(1, 0)),
            ),
        ]
        for periodic in [True, False]:
            path_parsed = next(
                parse_seq(
                    seq=path,
                    idx_chains=self.lad_idx_to_zedge[sub_kind],
                    periodic=periodic,
                    indices=self.lad_indices[sub_kind],
                    z_coupling=ZCoupling.EITHER,
                )
            )
            crucial_bonds_pos = path_parsed.crucial_edge_pos
            alt_chain_bonds_pos = path_parsed.alt_edge_pos
            self.assertEqual(len(alt_chain_bonds_pos), 6)
            for bond_pos in expected_crucial:
                self.assertTrue(bond_pos in crucial_bonds_pos)
            if periodic:
                self.assertTrue(self.normalized_ibp(bond=(0, 1), pos=(0, 0)) in crucial_bonds_pos)
                self.assertEqual(len(crucial_bonds_pos), 7)
            else:
                self.assertFalse(self.normalized_ibp(bond=(0, 1), pos=(0, 0)) in crucial_bonds_pos)
                self.assertEqual(len(crucial_bonds_pos), 6)
            for x, y in expected_alt:
                self.assertTrue((x, y) in alt_chain_bonds_pos.items())
                self.assertTrue((y, x) in alt_chain_bonds_pos.items())

    def test_square_main_base(self):
        path = [Edge(0, 2), Edge(4, 3), Edge(5, 7)]
        sub_kind = "main"
        expected_crucial = [
            self.normalized_ibp(bond=(0, 2), pos=(0, 0)),
            self.normalized_ibp(bond=(4, 3), pos=(0, 0)),
            self.normalized_ibp(bond=(5, 7), pos=(0, 0)),
        ]
        expected_alt = [
            (
                self.normalized_ibp(bond=(0, 3), pos=(0, 0)),
                self.normalized_ibp(bond=(4, 2), pos=(0, 0)),
            ),
            (
                self.normalized_ibp(bond=(5, 3), pos=(0, 0)),
                self.normalized_ibp(bond=(4, 7), pos=(0, 0)),
            ),
        ]
        periodic = False
        path_parsed = next(
            parse_seq(
                seq=path,
                idx_chains=self.sq_idx_to_zedge[sub_kind],
                periodic=periodic,
                indices=self.sq_indices[sub_kind],
                z_coupling=ZCoupling.EITHER,
            )
        )
        crucial_bonds_pos = path_parsed.crucial_edge_pos
        alt_chain_bonds_pos = path_parsed.alt_edge_pos
        self.assertEqual(len(crucial_bonds_pos), 3)
        self.assertEqual(len(alt_chain_bonds_pos), 4)
        for bond_pos in expected_crucial:
            self.assertTrue(bond_pos in crucial_bonds_pos)
        for x, y in expected_alt:
            self.assertTrue((x, y) in alt_chain_bonds_pos.items())
            self.assertTrue((y, x) in alt_chain_bonds_pos.items())

    def test_square_main_raises_error(self):
        path = [Edge(0, 2), Edge(4, 6), Edge(5, 7)]
        sub_kind = "main"
        periodic = True
        with self.assertRaises(StopIteration):
            next(
                parse_seq(
                    seq=path,
                    idx_chains=self.sq_idx_to_zedge[sub_kind],
                    periodic=periodic,
                    indices=self.sq_indices[sub_kind],
                    z_coupling=ZCoupling.EITHER,
                )
            )

    def test_square_anti_base(self):
        path = [Edge(0, 2), Edge(4, 6), Edge(5, 7)]
        sub_kind = "anti"
        periodic = True
        path_parsed = next(
            parse_seq(
                seq=path,
                idx_chains=self.sq_idx_to_zedge[sub_kind],
                periodic=periodic,
                indices=self.sq_indices[sub_kind],
                z_coupling=ZCoupling.EITHER,
            )
        )
        crucial_bonds_pos = path_parsed.crucial_edge_pos
        alt_chain_bonds_pos = path_parsed.alt_edge_pos
        expected_crucial = [
            self.normalized_ibp(bond=(2, 0), pos=(0, 0)),
            self.normalized_ibp(bond=(2, 4), pos=(0, 0)),
            self.normalized_ibp(bond=(6, 4), pos=(0, 0)),
            self.normalized_ibp(bond=(7, 5), pos=(0, 0)),
            self.normalized_ibp(bond=(6, 5), pos=(0, 0)),
            self.normalized_ibp(bond=(2, 5), pos=(0, 0)),
        ]
        self.assertEqual(len(crucial_bonds_pos), 6)
        for bond_pos in expected_crucial:
            self.assertTrue(bond_pos in crucial_bonds_pos)
        self.assertEqual(len(alt_chain_bonds_pos), 0)

    def test_square_anti_raises_error(self):
        path = [(4, 6), (0, 2), (1, 3)]
        sub_kind = "anti"
        periodic = True
        with self.assertRaises(StopIteration):
            next(
                parse_seq(
                    seq=path,
                    idx_chains=self.sq_idx_to_zedge[sub_kind],
                    periodic=periodic,
                    indices=self.sq_indices[sub_kind],
                    z_coupling=ZCoupling.EITHER,
                )
            )

    def test_square_main(self):
        path = [Edge(0, 2), Edge(4, 6), Edge(4, 3)]
        sub_kind = "main"
        expected_crucial = [
            self.normalized_ibp(bond=(0, 2), pos=(0, 0)),
            self.normalized_ibp(bond=(4, 6), pos=(0, 0)),
            self.normalized_ibp(bond=(4, 3), pos=(1, 0)),
            self.normalized_ibp(bond=(4, 2), pos=(0, 0)),
        ]

        for periodic in [True, False]:
            path_parsed = next(
                parse_seq(
                    seq=path,
                    idx_chains=self.sq_idx_to_zedge[sub_kind],
                    periodic=periodic,
                    indices=self.sq_indices[sub_kind],
                    z_coupling=ZCoupling.EITHER,
                )
            )
            crucial_bonds_pos = path_parsed.crucial_edge_pos
            self.assertEqual(len(crucial_bonds_pos), 4)
            alt_chain_bonds_pos = path_parsed.alt_edge_pos

            for bond_pos in expected_crucial:
                self.assertTrue(bond_pos in crucial_bonds_pos)
            if periodic:
                expected_alt = [
                    (
                        self.normalized_ibp(bond=(4, 3), pos=(0, 0)),
                        self.normalized_ibp(bond=(4, 6), pos=(1, 0)),
                    ),
                    (
                        self.normalized_ibp(bond=(0, 3), pos=(0, 0)),
                        self.normalized_ibp(bond=(4, 2), pos=(1, 0)),
                    ),
                ]
                self.assertEqual(len(alt_chain_bonds_pos), 4)
            else:
                expected_alt = [
                    (
                        self.normalized_ibp(bond=(4, 3), pos=(0, 0)),
                        self.normalized_ibp(bond=(4, 6), pos=(1, 0)),
                    )
                ]
                self.assertEqual(len(alt_chain_bonds_pos), 2)
            for x, y in expected_alt:
                self.assertTrue((x, y) in alt_chain_bonds_pos.items())
                self.assertTrue((y, x) in alt_chain_bonds_pos.items())

    def test_square_anti(self):
        sub_kind = "anti"
        path = [Edge(4, 6), Edge(7, 5), Edge(3, 5), Edge(1, 2)]
        expected_crucial = [
            self.normalized_ibp(bond=(6, 4), pos=(0, 0)),
            self.normalized_ibp(bond=(6, 5), pos=(0, 0)),
            self.normalized_ibp(bond=(7, 5), pos=(0, 0)),
            self.normalized_ibp(bond=(3, 5), pos=(0, 1)),
            self.normalized_ibp(bond=(2, 1), pos=(0, 0)),
        ]
        expected_alt = [
            (
                self.normalized_ibp(bond=(3, 5), pos=(0, 0)),
                self.normalized_ibp(bond=(7, 5), pos=(0, 1)),
            ),
            (
                self.normalized_ibp(bond=(2, 5), pos=(0, 1)),
                self.normalized_ibp(bond=(3, 1), pos=(0, 0)),
            ),
        ]
        for periodic in [True, False]:
            path_parsed = next(
                parse_seq(
                    seq=path,
                    idx_chains=self.sq_idx_to_zedge[sub_kind],
                    periodic=periodic,
                    indices=self.sq_indices[sub_kind],
                    z_coupling=ZCoupling.EITHER,
                )
            )
            crucial_bonds_pos = path_parsed.crucial_edge_pos
            for bond_pos in expected_crucial:
                self.assertTrue(bond_pos in expected_crucial)
            if periodic:
                self.assertEqual(len(crucial_bonds_pos), len(expected_crucial) + 1)
                self.assertTrue(self.normalized_ibp(bond=(2, 4), pos=(0, 0)) in crucial_bonds_pos)
            else:
                self.assertEqual(len(crucial_bonds_pos), len(expected_crucial))
            alt_chain_bonds_pos = path_parsed.alt_edge_pos
            self.assertEqual(len(alt_chain_bonds_pos), 4)
            for x, y in expected_alt:
                self.assertTrue((x, y) in alt_chain_bonds_pos.items())
                self.assertTrue((y, x) in alt_chain_bonds_pos.items())

    def test_both_coupling_lad_main_closed(self):
        sub_kind = "main"
        coupler_both_per = [
            [Edge(0, 1), Edge(1, 3), Edge(2, 3)],
            [Edge(2, 3), Edge(0, 1), Edge(2, 3), Edge(4, 5), Edge(6, 7), Edge(4, 5)],
        ]
        for path in coupler_both_per:
            for closed in [True, False]:
                for coupling in ZCoupling:
                    if coupling is ZCoupling.ONE_ZERO:
                        continue
                    next(
                        parse_seq(
                            seq=path,
                            idx_chains=self.lad_idx_to_zedge[sub_kind],
                            periodic=closed,
                            indices=self.lad_indices[sub_kind],
                            z_coupling=coupling,
                        )
                    )

    def test_both_coupling_lad_main_nonclosed(self):
        sub_kind = "main"
        coupler_both_nonper = [
            [Edge(0, 1), Edge(1, 3), Edge(3, 5), Edge(5, 7)],
            [Edge(0, 1), Edge(2, 3), Edge(4, 5), Edge(6, 7), Edge(4, 5)],
        ]
        for path in coupler_both_nonper:
            for coupling in ZCoupling:
                if coupling is ZCoupling.ONE_ZERO:
                    continue
                next(
                    parse_seq(
                        seq=path,
                        idx_chains=self.lad_idx_to_zedge[sub_kind],
                        periodic=False,
                        indices=self.lad_indices[sub_kind],
                        z_coupling=coupling,
                    )
                )
                with self.assertRaises(StopIteration):
                    next(
                        parse_seq(
                            seq=path,
                            idx_chains=self.lad_idx_to_zedge[sub_kind],
                            periodic=True,
                            indices=self.lad_indices[sub_kind],
                            z_coupling=coupling,
                        )
                    )

    def test_01_coupling_lad_main_nonclosed(self):
        sub_kind = "anti"
        path_nonclosed = [Edge(5, 7), Edge(4, 6)]
        closed = False
        for coupling in (ZCoupling.EITHER, ZCoupling.ZERO_ONE):
            if coupling is ZCoupling.ONE_ZERO:
                continue
            next(
                parse_seq(
                    seq=path_nonclosed,
                    idx_chains=self.lad_idx_to_zedge[sub_kind],
                    periodic=closed,
                    indices=self.lad_indices[sub_kind],
                    z_coupling=coupling,
                )
            )
        with self.assertRaises(StopIteration):
            next(
                parse_seq(
                    seq=path_nonclosed,
                    idx_chains=self.lad_idx_to_zedge[sub_kind],
                    periodic=closed,
                    indices=self.lad_indices[sub_kind],
                    z_coupling=ZCoupling.ZERO_ONE_ONE_ZERO,
                )
            )

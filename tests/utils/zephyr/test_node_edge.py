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
from itertools import product

from minorminer.utils.zephyr.coordinate_systems import CartesianCoord, ZephyrCoord
from minorminer.utils.zephyr.node_edge import Edge, EdgeKind, NodeKind, ZEdge, ZNode, ZShape
from minorminer.utils.zephyr.plane_shift import ZPlaneShift


class TestEdge(unittest.TestCase):
    def test_valid_input_runs(self) -> None:
        Edge(2, 4)
        Edge(5, -1)

    def test_invalid_input_raises_error(self) -> None:
        with self.assertRaises(TypeError):
            Edge(2, "string")
            Edge(2, ZNode(ZephyrCoord(0, 10, 3, 1, 3), ZShape(t=4)))

    def test_equal(self) -> None:
        self.assertEqual(Edge(2, 4), Edge(4, 2))


class TestZEdge(unittest.TestCase):
    def test_valid_input_runs(self) -> None:
        valid_edges = [
            (
                ZNode(coord=ZephyrCoord(0, 10, 3, 1, 3), shape=ZShape(6, 4)),
                ZNode(coord=ZephyrCoord(0, 10, 3, 1, 2), shape=ZShape(6, 4)),
            ),
            (
                ZNode(coord=CartesianCoord(4, 3, 2), shape=(6, 3)),
                ZNode(coord=CartesianCoord(4, 1, 2), shape=(6, 3)),
            ),
            (ZNode(coord=(1, 6)), ZNode(coord=(5, 6))),
        ]
        for x, y in valid_edges:
            ZEdge(x, y)

    def test_invalid_input_raises_error(self):
        with self.assertRaises((TypeError, ValueError)):
            ZEdge(2, 4)
            ZEdge(
                ZNode(coord=CartesianCoord(4, 3, 2), shape=(6, 3)),
                ZNode(coord=CartesianCoord(4, 3, 2), shape=(6, 3)),
            )


class TestZNode(unittest.TestCase):
    def setUp(self):
        self.u_vals = [0, 1]
        self.invalid_u_vals = [2, -1, None, 3.5]
        self.invalid_w_vals = [-1]
        self.j_vals = [0, 1]
        self.invalid_j_vals = [2, -1, None, 3.5]
        self.m_vals = [6, 1, 20]
        self.invalid_m_vals = [0, 2.5, -3]
        self.t_vals = [6, 1, 20]
        self.invalid_t_vals = [0, 2.5, -2]
        self.invalid_z_vals = [-1, None]
        xym_vals = [
            ((0, 3), 1),
            ((5, 2), 6),
            ((16, 1), 4),
            ((1, 12), 3),
            ((3, 0), 4),
            ((5, 4), 6),
            ((0, 3), 6),
            ((6, 3), 5),
        ]
        self.xyms = xym_vals + [(xy, None) for xy, _ in xym_vals]
        self.left_up_xyms = [((0, 3), 6), ((0, 3), None), ((11, 0), None), ((0, 5), 8)]
        self.right_down_xyms = [((1, 12), 3), ((16, 1), 4)]
        self.midgrid_xyms = [((5, 2), 6), ((5, 4), 6), ((6, 7), 5)]

    def test_zephyr_node_runs(self) -> None:
        for xy, m in self.xyms:
            ZNode(xy, ZShape(m=m))
            ZNode(xy, ZShape(m=m), convert_to_z=True)

        for u in self.u_vals:
            for j in self.j_vals:
                for m in self.m_vals:
                    w_vals = range(2 * m + 1)
                    z_vals = range(m)
                    for w in w_vals:
                        for z in z_vals:
                            for t in self.t_vals:
                                k_vals = range(t)
                                for k in k_vals:
                                    ZNode(coord=(u, w, k, j, z), shape=(m, t))

    def test_zephyr_node_invalid_args_raises_error(self) -> None:
        invalid_xyms = [
            ((-1, 2), 6),
            ((6, -1), 4),
            ((1, 3), -5),
            ((1, 3), 6),
            ((2, 4), 6),
            ((17, 0), 4),
            ((0, 17), 4),
            ((0, 0, 0), 1),
            ((-1, 2), None),
        ]
        for xy, m in invalid_xyms:
            with self.assertRaises((ValueError, TypeError)):
                ZNode(coord=xy, shape=ZShape(m=m))
        # All good except u_vals
        for u in self.invalid_u_vals:
            for j in self.j_vals:
                for m in self.m_vals:
                    w_vals = range(2 * m + 1)
                    z_vals = range(m)
                    for w in w_vals:
                        for z in z_vals:
                            for t in self.t_vals:
                                k_vals = range(t)
                                for k in k_vals:
                                    with self.assertRaises((ValueError, TypeError)):
                                        ZNode(coord=(u, w, k, j, z), shape=(m, t))

        # All good except w_vals
        for u in self.u_vals:
            for j in self.j_vals:
                for m in self.m_vals:
                    z_vals = range(m)
                    for w in self.invalid_w_vals:
                        for z in z_vals:
                            for t in self.t_vals:
                                k_vals = range(t)
                                for k in k_vals:
                                    with self.assertRaises((ValueError, TypeError)):
                                        ZNode(coord=(u, w, k, j, z), shape=(m, t))

        # All good except k_vals
        for u in self.u_vals:
            for j in self.j_vals:
                for m in self.m_vals:
                    w_vals = range(2 * m + 1)
                    z_vals = range(m)
                    for w in w_vals:
                        for z in z_vals:
                            for t in self.t_vals:
                                invalid_k_vals = [-1, t, 2.5, None]
                                for k in invalid_k_vals:
                                    with self.assertRaises((ValueError, TypeError)):
                                        ZNode(coord=(u, w, k, j, z), shape=(m, t))

        # All good except j_vals
        for u in self.u_vals:
            for j in self.invalid_j_vals:
                for m in self.m_vals:
                    w_vals = range(2 * m + 1)
                    z_vals = range(m)
                    for w in w_vals:
                        for z in z_vals:
                            for t in self.t_vals:
                                k_vals = range(t)
                                for k in k_vals:
                                    with self.assertRaises((ValueError, TypeError)):
                                        ZNode(coord=(u, w, k, j, z), shape=(m, t))
        # All good except z_vals
        for u in self.u_vals:
            for j in self.j_vals:
                for m in self.m_vals:
                    w_vals = range(2 * m + 1)
                    invalid_z_vals = [None, -1, m, 1.5]
                    for w in w_vals:
                        for z in invalid_z_vals:
                            for t in self.t_vals:
                                k_vals = range(t)
                                for k in k_vals:
                                    with self.assertRaises((ValueError, TypeError)):
                                        ZNode(coord=(u, w, k, j, z), shape=(m, t))

        # All good except m_vals
        for u in self.u_vals:
            for j in self.j_vals:
                for m in self.invalid_m_vals:
                    w_vals = [0]
                    z_vals = [0]
                    for w in w_vals:
                        for z in z_vals:
                            for t in self.t_vals:
                                k_vals = range(t)
                                for k in k_vals:
                                    with self.assertRaises((ValueError, TypeError)):
                                        ZNode(coord=(u, w, k, j, z), shape=(m, t))

        # All good except t_vals
        for u in self.u_vals:
            for j in self.j_vals:
                for m in self.invalid_m_vals:
                    w_vals = [0]
                    z_vals = [0]
                    for w in w_vals:
                        for z in z_vals:
                            for t in self.invalid_t_vals:
                                k_vals = [0, 1, 3]
                                for k in k_vals:
                                    with self.assertRaises((ValueError, TypeError)):
                                        ZNode(coord=(u, w, k, j, z), shape=(m, t))

    def test_add_sub_runs(self) -> None:
        left_up_pcs = [ZNode(xy, ZShape(m=m)) for xy, m in self.left_up_xyms]
        lu_qps = ZPlaneShift(-1, -1)
        right_down_pcs = [ZNode(xy, ZShape(m=m)) for xy, m in self.right_down_xyms]
        rd_qps = ZPlaneShift(1, 1)
        midgrid_pcs = [ZNode(xy, ZShape(m=m)) for xy, m in self.midgrid_xyms]
        for pc in midgrid_pcs:
            for s1, s2 in product((-2, 2), (-2, 2)):
                pc + ZPlaneShift(s1, s2)
        for pc in left_up_pcs:
            with self.assertRaises(ValueError):
                pc + lu_qps
        for pc in right_down_pcs:
            with self.assertRaises(ValueError):
                pc + rd_qps

    def test_add_sub(self) -> None:
        midgrid_pcs = [ZNode(xy, ZShape(m=m)) for xy, m in self.midgrid_xyms]
        right_down_pcs = [ZNode(xy, ZShape(m=m)) for xy, m in self.right_down_xyms]
        left_up_pcs = [ZNode(xy, ZShape(m=m)) for xy, m in self.left_up_xyms]

        for pc in midgrid_pcs + right_down_pcs + left_up_pcs:
            self.assertEqual(pc + ZPlaneShift(0, 0), pc)

    def test_neighbors_generator_runs(self) -> None:
        for _ in ZNode((1, 12, 4), ZShape(t=6)).neighbors_generator():
            _

    def test_zcoord(self) -> None:
        ZNode((11, 12, 4), ZShape(t=6)).zcoord == ZephyrCoord(1, 0, 4, 0, 2)
        ZNode((1, 0)).zcoord == ZephyrCoord(1, 0, None, 0, 0)
        ZNode((0, 1)).zcoord == ZephyrCoord(0, 0, None, 0, 0)

    def test_direction_node_kind(self) -> None:
        for u in self.u_vals:
            for j in self.j_vals:
                for m in self.m_vals:
                    w_vals = range(2 * m + 1)
                    z_vals = range(m)
                    for w in w_vals:
                        for z in z_vals:
                            for t in self.t_vals:
                                k_vals = range(t)
                                for k in k_vals:
                                    zn = ZNode(coord=(u, w, k, j, z), shape=(m, t))
                                    self.assertEqual(zn.direction, u)
                                    if u == 0:
                                        self.assertTrue(zn.is_vertical())
                                        self.assertEqual(zn.node_kind, NodeKind.VERTICAL)
                                    else:
                                        self.assertTrue(zn.is_horizontal())
                                        self.assertEqual(zn.node_kind, NodeKind.HORIZONTAL)

    def test_neighbor_kind(self) -> None:
        zn = ZNode((0, 1))
        self.assertTrue(zn.neighbor_kind(ZNode((1, 0))) is EdgeKind.INTERNAL)
        self.assertTrue(zn.neighbor_kind(ZNode((1, 2))) is EdgeKind.INTERNAL)
        self.assertTrue(zn.neighbor_kind(ZNode((0, 3))) is EdgeKind.ODD)
        self.assertTrue(zn.neighbor_kind(ZNode((0, 5))) is EdgeKind.EXTERNAL)
        self.assertTrue(zn.neighbor_kind(ZNode((0, 7))) is None)
        self.assertTrue(zn.neighbor_kind(ZNode((1, 6))) is None)

    def test_internal_generator(self) -> None:
        zn1 = ZNode((0, 1))
        set_internal1 = {x for x in zn1.internal_neighbors_generator()}
        expected1 = {ZNode((1, 0)), ZNode((1, 2))}
        self.assertEqual(set_internal1, expected1)

        zn2 = ZNode((0, 1, 0), ZShape(t=4))
        set_internal2 = {x for x in zn2.internal_neighbors_generator()}
        expected2 = {ZNode((1, 0, k), ZShape(t=4)) for k in range(4)} | {
            ZNode((1, 2, k), ZShape(t=4)) for k in range(4)
        }
        self.assertEqual(set_internal2, expected2)

    def test_external_generator(self) -> None:
        zn = ZNode((0, 1))
        set_external = {x for x in zn.external_neighbors_generator()}
        expected = {ZNode((0, 5))}
        self.assertEqual(set_external, expected)

        zn2 = ZNode((0, 1, 2), ZShape(t=4))
        set_external2 = {x for x in zn2.external_neighbors_generator()}
        expected2 = {ZNode((0, 5, 2), ZShape(t=4))}
        self.assertEqual(set_external2, expected2)

    def test_odd_generator(self) -> None:
        zn = ZNode((0, 1))
        set_odd = {x for x in zn.odd_neighbors_generator()}
        expected = {ZNode((0, 3))}
        self.assertEqual(set_odd, expected)

        zn2 = ZNode((0, 1, 2), ZShape(t=4))
        set_odd2 = {x for x in zn2.odd_neighbors_generator()}
        expected2 = {ZNode((0, 3, 2), ZShape(t=4))}
        self.assertEqual(set_odd2, expected2)

    def test_degree(self) -> None:
        for (x, y), m in self.midgrid_xyms:
            qzn1 = ZNode(coord=(x, y), shape=ZShape(m=m))
            self.assertEqual(len(qzn1.neighbors()), 8)
            self.assertEqual(qzn1.degree(), 8)
            self.assertEqual(qzn1.degree(nbr_kind=EdgeKind.INTERNAL), 4)
            self.assertEqual(qzn1.degree(nbr_kind=EdgeKind.EXTERNAL), 2)
            self.assertEqual(qzn1.degree(nbr_kind=EdgeKind.ODD), 2)
            for t in [1, 4, 6]:
                zn1 = ZNode(coord=(x, y, 0), shape=ZShape(m=m, t=t))
                self.assertEqual(zn1.degree(), 4 * t + 4)
                self.assertEqual(zn1.degree(nbr_kind=EdgeKind.INTERNAL), 4 * t)
                self.assertEqual(zn1.degree(nbr_kind=EdgeKind.EXTERNAL), 2)
                self.assertEqual(zn1.degree(nbr_kind=EdgeKind.ODD), 2)

        qzn2 = ZNode(coord=(0, 1))
        self.assertEqual(qzn2.degree(), 4)
        self.assertEqual(qzn2.degree(nbr_kind=EdgeKind.INTERNAL), 2)
        self.assertEqual(qzn2.degree(nbr_kind=EdgeKind.EXTERNAL), 1)
        self.assertEqual(qzn2.degree(nbr_kind=EdgeKind.ODD), 1)
        for t in [1, 4, 6]:
            zn2 = ZNode(coord=(0, 1, 0), shape=ZShape(t=t))
            self.assertEqual(zn2.degree(), 2 * t + 2)
            self.assertEqual(zn2.degree(nbr_kind=EdgeKind.INTERNAL), 2 * t)
            self.assertEqual(zn2.degree(nbr_kind=EdgeKind.EXTERNAL), 1)
            self.assertEqual(zn2.degree(nbr_kind=EdgeKind.ODD), 1)

        qzn3 = ZNode((24, 5), ZShape(m=6))
        self.assertEqual(qzn3.degree(), 6)
        self.assertEqual(qzn3.degree(nbr_kind=EdgeKind.INTERNAL), 2)
        self.assertEqual(qzn3.degree(nbr_kind=EdgeKind.EXTERNAL), 2)
        self.assertEqual(qzn3.degree(nbr_kind=EdgeKind.ODD), 2)
        for t in [1, 5, 6]:
            zn3 = ZNode((24, 5, 0), ZShape(m=6, t=t))
            self.assertEqual(zn3.degree(), 2 * t + 4)
            self.assertEqual(zn3.degree(nbr_kind=EdgeKind.INTERNAL), 2 * t)
            self.assertEqual(zn3.degree(nbr_kind=EdgeKind.EXTERNAL), 2)
            self.assertEqual(zn3.degree(nbr_kind=EdgeKind.ODD), 2)

        qzn4 = ZNode((24, 5))
        self.assertEqual(qzn4.degree(), 8)
        self.assertEqual(qzn4.degree(nbr_kind=EdgeKind.INTERNAL), 4)
        self.assertEqual(qzn4.degree(nbr_kind=EdgeKind.EXTERNAL), 2)
        self.assertEqual(qzn4.degree(nbr_kind=EdgeKind.ODD), 2)
        for t in [1, 5, 6]:
            zn4 = ZNode((24, 5, 0), ZShape(t=t))
            self.assertEqual(zn4.degree(), 4 * t + 4)
            self.assertEqual(zn4.degree(nbr_kind=EdgeKind.INTERNAL), 4 * t)
            self.assertEqual(zn4.degree(nbr_kind=EdgeKind.EXTERNAL), 2)
            self.assertEqual(zn4.degree(nbr_kind=EdgeKind.ODD), 2)

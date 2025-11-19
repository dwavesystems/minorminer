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
    @parameterized.expand(
        [
            (ZephyrCoord(0, 10, 3, 1, 3), ZephyrCoord(0, 10, 3, 1, 2), ZShape(6, 4)),
            (CartesianCoord(4, 3, 2), CartesianCoord(4, 1, 2), (6, 3)),
            ((1, 6), (5, 6), None)
            ]
        )
    def test_valid_input_runs(self, x, y, shape) -> None:
        zn_x = ZNode(coord=x, shape=shape)
        zn_y = ZNode(coord=y, shape=shape)
        
        ZEdge(zn_x, zn_y)

    @parameterized.expand(
        [
            ((2, 4), TypeError),
            ((ZNode(coord=CartesianCoord(4, 3, 2), shape=(6, 3)), ZNode(coord=CartesianCoord(4, 3, 2), shape=(6, 3))), ValueError),
            ]
        )
    def test_invalid_input_raises_error(self, invalid_edge, expected_err):
        with self.assertRaises(expected_err):
            ZEdge(*invalid_edge)


U_VALS = [0, 1]
J_VALS = [0, 1]
M_VALS = [1, 6, 12]
T_VALS = [1, 2, 4, 6]


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

    @parameterized.expand(
        [
            ((u, w, k, j, z), (m, t))
            for u in U_VALS
            for j in J_VALS
            for m in M_VALS
            for t in T_VALS
            for w in range(2 * m + 1)
            for k in range(t)
            for z in range(m)
        ][::100]
    )
    def test_znode_zcoord_runs(self, uwkjz, mt):
        ZNode(coord=uwkjz, shape=mt)

    @parameterized.expand(
        [
            ((0, 3), 1),
            ((5, 2), 6),
            ((16, 1), 4),
            ((1, 12), 3),
            ((3, 0), 4),
            ((5, 4), 6),
            ((0, 3), 6),
            ((6, 3), 5),
            ((3, 0), None),
            ((5, 4), None),
            ((0, 3), None),
        ]
    )
    def test_znode_ccoord_runs(self, xy, m):
        ZNode(xy, ZShape(m=m))
        ZNode(xy, ZShape(m=m), convert_to_z=True)

    @parameterized.expand(
        [
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
    )
    def test_bad_args_raises_error_ccoord(self, xy, m):
        with self.assertRaises((ValueError, TypeError)):
            ZNode(coord=xy, shape=ZShape(m=m))

    @parameterized.expand(
        [
            ((2, 0, 0, 0, 0), (6, 4)),  # All good except u_val
            ((None, 3, 0, 0, 4), (6, 1)),  # All good except u_val
            ((3.5, 8, 2, 0, 0), (8, 4)),  # All good except u_val
            ((0, -1, 1, 1, 3), (6, 4)),  # All good except w_val
            ((0, 10, 2.5, 0, 5), (6, 4)),  # All good except k_val
            ((1, 23, -1, 1, 5), (12, 2)),  # All good except k_val
            ((1, 24, 1, 3.5, 9), (12, 4)),  # All good except j_val
            ((1, 24, 0, 1, None), (12, 2)),  # All good except z_val
            ((1, 20, 3, 1, 12), (12, 6)),  # All good except z_val
            ((0, 0, 0, 0, 0), (0, 0)),  # All good except m_val
            ((0, 0, 0, 0, 0), (1, -1)),  # All good except t_val
        ]
    )
    def test_bad_args_raises_error_ccoord(self, uwkjz, mt):
        with self.assertRaises((ValueError, TypeError)):
            ZNode(coord=uwkjz, shape=mt)

    @parameterized.expand(
        [
            ((0, 3), 6, ZPlaneShift(-1, -1)),
            ((0, 3), None, ZPlaneShift(-1, -1)),
            ((11, 0), None, ZPlaneShift(-1, -1)),
            ((0, 5), 8, ZPlaneShift(-1, -1)),
            ((1, 12), 3, ZPlaneShift(1, 1)),
            ((16, 1), 4, ZPlaneShift(1, 1)),
        ]
    )
    def test_add_sub_raises_error_invalid(self, xy, m, zps) -> None:
        zn = ZNode(xy, ZShape(m=m))
        with self.assertRaises(ValueError):
            zn + zps

    @parameterized.expand(
        [
            (ZNode((5, 2), ZShape(6)), ZPlaneShift(-2, -2), ZNode((3, 0), ZShape(6))),
            (ZNode((5, 2), ZShape(6)), ZPlaneShift(2, -2), ZNode((7, 0), ZShape(6))),
            (ZNode((5, 2), ZShape(6)), ZPlaneShift(2, 2), ZNode((7, 4), ZShape(6))),
            (ZNode((5, 2), ZShape(6)), ZPlaneShift(-2, 2), ZNode((3, 4), ZShape(6))),
            (ZNode((5, 4), ZShape(4)), ZPlaneShift(-4, -4), ZNode((1, 0), ZShape(4))),
            (ZNode((6, 7), ZShape(5)), ZPlaneShift(1, 11), ZNode((7, 18), ZShape(5))),
            (ZNode((0, 3), ZShape(6)), ZPlaneShift(0, 0), ZNode((0, 3), ZShape(6))),
            (ZNode((1, 12), ZShape(3)), ZPlaneShift(1, -1), ZNode((2, 11), ZShape(3))),
        ]
    )
    def test_add_sub(self, zn, zps, expected) -> None:
        self.assertEqual(zn + zps, expected)

    def test_neighbors_boundary(self) -> None:
        x, y, k, t = 1, 12, 4, 6
        expected_nbrs = {
            ZNode((x + i, y + j, kp), ZShape(t=t))
            for i in (-1, 1)
            for j in (-1, 1)
            for kp in range(t)
        }
        expected_nbrs |= {ZNode((x + 2, y, k), ZShape(t=t)), ZNode((x + 4, y, k), ZShape(t=t))}
        self.assertEqual(set(ZNode((x, y, k), ZShape(t=t)).neighbors()), expected_nbrs)

    def test_neighbors_mid(self):
        x, y, k, t = 10, 5, 3, 6
        expected_nbrs = {
            ZNode((x + i, y + j, kp), ZShape(t=t))
            for i in (-1, 1)
            for j in (-1, 1)
            for kp in range(t)
        }
        expected_nbrs |= {ZNode((x, y + 4, k), ZShape(t=t)), ZNode((x, y - 4, k), ZShape(t=t))}
        expected_nbrs |= {ZNode((x, y + 2, k), ZShape(t=t)), ZNode((x, y - 2, k), ZShape(t=t))}
        self.assertEqual(set(ZNode((x, y, k), ZShape(t=t)).neighbors()), expected_nbrs)

    def test_zcoord(self) -> None:
        ZNode((11, 12, 4), ZShape(t=6)).zcoord == ZephyrCoord(1, 0, 4, 0, 2)
        ZNode((1, 0)).zcoord == ZephyrCoord(1, 0, None, 0, 0)
        ZNode((0, 1)).zcoord == ZephyrCoord(0, 0, None, 0, 0)

    @parameterized.expand(
        [
            ((u, w, k, j, z), (m, t))
            for u in U_VALS
            for j in J_VALS
            for m in M_VALS
            for t in T_VALS
            for w in range(2 * m + 1)
            for k in range(t)
            for z in range(m)
        ][::300]
    )
    def test_direction(self, uwkjz, mt) -> None:
        zn = ZNode(coord=uwkjz, shape=mt)
        self.assertEqual(zn.direction, uwkjz[0])

    @parameterized.expand(
        [
            ((u, w, k, j, z), (m, t))
            for u in U_VALS
            for j in J_VALS
            for m in M_VALS
            for t in T_VALS
            for w in range(2 * m + 1)
            for k in range(t)
            for z in range(m)
        ][::200]
    )
    def test_node_kind(self, uwkjz, mt) -> None:
        zn = ZNode(coord=uwkjz, shape=mt)
        if uwkjz[0] == 0:
            self.assertTrue(zn.is_vertical())
            self.assertEqual(zn.node_kind, NodeKind.VERTICAL)
        else:
            self.assertTrue(zn.is_horizontal())
            self.assertEqual(zn.node_kind, NodeKind.HORIZONTAL)

    @parameterized.expand(
        [
            (ZNode((1, 0)), EdgeKind.INTERNAL),
            (ZNode((1, 2)), EdgeKind.INTERNAL),
            (ZNode((0, 3)), EdgeKind.ODD),
            (ZNode((0, 5)), EdgeKind.EXTERNAL),
            (ZNode((0, 7)), None),
            (ZNode((1, 6)), None),
        ]
    )
    def test_neighbor_kind(self, zn1, nbr_kind) -> None:
        zn0 = ZNode((0, 1))
        self.assertIs(zn0.neighbor_kind(zn1), nbr_kind)

    @parameterized.expand(
        [
            (ZNode((0, 1)), {ZNode((1, 0)), ZNode((1, 2))}),
            (
                ZNode((0, 1, 0), ZShape(t=4)),
                {ZNode((1, 0, k), ZShape(t=4)) for k in range(4)}
                | {ZNode((1, 2, k), ZShape(t=4)) for k in range(4)},
            ),
        ]
    )
    def test_internal_generator(self, zn, expected) -> None:
        set_internal = {x for x in zn.internal_neighbors_generator()}
        self.assertEqual(set_internal, expected)

    @parameterized.expand(
        [
            (ZNode((0, 1)), {ZNode((0, 5))}),
            (ZNode((0, 1, 2), ZShape(t=4)), {ZNode((0, 5, 2), ZShape(t=4))}),
            (
                ZNode((11, 6, 3), ZShape(t=4)),
                {ZNode((7, 6, 3), ZShape(t=4)), ZNode((15, 6, 3), ZShape(t=4))},
            ),
        ]
    )
    def test_external_generator(self, zn, expected) -> None:
        set_external = {x for x in zn.external_neighbors_generator()}
        self.assertEqual(set_external, expected)

    @parameterized.expand(
        [
            (ZNode((0, 1)), {ZNode((0, 3))}),
            (ZNode((0, 1, 2), ZShape(t=4)), {ZNode((0, 3, 2), ZShape(t=4))}),
            (ZNode((15, 8)), {ZNode((13, 8)), ZNode((17, 8))}),
        ]
    )
    def test_odd_generator(self, zn, expected) -> None:
        set_odd = {x for x in zn.odd_neighbors_generator()}
        self.assertEqual(set_odd, expected)

    @parameterized.expand(
        [
            ((5, 2), 6, None, 4, 4),
            ((5, 2), 6, EdgeKind.INTERNAL, 4, 0),
            ((5, 2), 10, EdgeKind.EXTERNAL, 0, 2),
            ((5, 2), 3, EdgeKind.ODD, 0, 2),
            ((6, 7), 12, None, 4, 4),
            ((6, 7), 8, EdgeKind.INTERNAL, 4, 0),
            ((6, 7), 12, EdgeKind.EXTERNAL, 0, 2),
            ((6, 7), 12, EdgeKind.ODD, 0, 2),
            ((0, 1), 4, EdgeKind.INTERNAL, 2, 0),
            ((0, 1), 2, EdgeKind.ODD, 0, 1),
            ((0, 1), 4, EdgeKind.EXTERNAL, 0, 1),
            ((0, 1), 4, None, 2, 2),
            ((24, 5), None, EdgeKind.INTERNAL, 4, 0),
            ((24, 5), None, EdgeKind.ODD, 0, 2),
            ((24, 5), None, EdgeKind.EXTERNAL, 0, 2),
            ((24, 5), None, None, 4, 4),
            ((24, 5), 6, EdgeKind.INTERNAL, 2, 0),
            ((24, 5), 8, EdgeKind.ODD, 0, 2),
            ((24, 5), 6, EdgeKind.EXTERNAL, 0, 2),
            ((24, 5), 8, None, 4, 4),
        ]
    )
    def test_degree(self, xy, m, nbr_kind, a, b) -> None:
            for t in [None, 1, 4, 6]:
                if t is None:
                    coord, t_p = xy, 1
                else:
                    coord, t_p = xy + (0, ), t
                zn = ZNode(coord=coord, shape=ZShape(m=m, t=t))

                with self.subTest(case=t):
                    self.assertEqual(zn.degree(nbr_kind=nbr_kind), a * t_p + b)


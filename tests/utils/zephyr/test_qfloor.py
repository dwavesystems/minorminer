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



from itertools import product
import unittest
from minorminer.utils.zephyr.plane_shift import PlaneShift
from minorminer.utils.zephyr.node_edge import ZNode, ZShape, ZEdge
from minorminer.utils.zephyr.qfloor import QuoTile, QuoFloor




class TestQuoTile(unittest.TestCase):
    def test_quo_tile_runs(self) -> None:
        zn = ZNode((0, 1), ZShape(m=6))
        shifts0 = [(1, -1), (0, 2), (1, 1), (1, 3)]
        zns = [zn + PlaneShift(*x) for x in shifts0]
        QuoTile(zns=zns)

    def test_zns_define(self) -> None:
        seed0 = ZNode((0, 1), ZShape(m=6))
        shifts0 = [PlaneShift(1, 1), PlaneShift(0, 2)]
        zns0 = [seed0+shift for shift in shifts0]
        seed1 = seed0+PlaneShift(0, 2)
        shifts1 = [PlaneShift(1, -1), PlaneShift(0, 0)]
        zns1 = [seed1+shift for shift in shifts1]
        zns2 = [ZNode((1, 2), ZShape(m=6)), ZNode((0, 0, 1, 0), ZShape(m=6))]
        qtile0 = QuoTile(zns=zns0)
        qtile1 = QuoTile(zns=zns1)
        qtile2 = QuoTile(zns=zns2)
        self.assertEqual(qtile0, qtile1)
        self.assertEqual(qtile0, qtile2)
        self.assertEqual(qtile2, qtile1)
        shifts = [PlaneShift(1, 1), PlaneShift(0, 2)]
        shifts_repeat = 2*shifts
        for zn in [ZNode((0, 1), ZShape(m=6)), ZNode((5, 12))]:
            qtile0 = QuoTile(zns=[zn+shift for shift in shifts])
            qtile1 = QuoTile(zns=[zn+shift for shift in shifts_repeat])
            self.assertEqual(qtile0, qtile1)

    def test_edges(self) -> None:
        shifts = [PlaneShift(1, 1), PlaneShift(0, 2)]
        for zn in [ZNode((0, 1), ZShape(m=6)), ZNode((5, 12))]:
            edges_ = QuoTile([zn+shift for shift in shifts]).edges()
            for xy in edges_:
                self.assertTrue(isinstance(xy, ZEdge))
                self.assertTrue(isinstance(xy[0], ZNode))
                self.assertTrue(isinstance(xy[1], ZNode))
                self.assertTrue(xy[0].is_neighbor(xy[1]))




class TestQuoFloor(unittest.TestCase):
    valid_connector = QuoFloor.tile_connector0

    def test_init_runs_as_expected(self) -> None:
        corner_qtile = [ZNode((0, 1)), ZNode((1, 0)), ZNode((1, 2)), ZNode((2, 3))]
        bad_corner_qtile = [ZNode((0, 1)), ((1, 0))]
        not_imp_connector = {(1, 0): PlaneShift(4, 0), (0, 1): PlaneShift(0, 2)}
        QuoFloor(corner_qtile=corner_qtile, dim = (100, 100))
        for m in [6, 8]:
            corner_qtile2 = [ZNode((0, 1), ZShape(m=m)), ZNode((1, 0), ZShape(m=m))]
            for dim in [(1, 1), (m, m), (m, 1), (1, m), (m//2, m//2)]:
                qfl = QuoFloor(corner_qtile=corner_qtile2, dim=dim)
                self.assertEqual(qfl.dim, dim)
                self.assertEqual(set(qfl.corner_qtile.zns), set(corner_qtile2))
                with self.assertRaises(TypeError):
                    qfl.corner_qtile = bad_corner_qtile
                with self.assertRaises(ValueError):
                    qfl.dim = (1, m+1)
                with self.assertRaises(ValueError):
                    qfl.dim = (1, 0)
                with self.assertRaises(ValueError):
                    qfl.dim = (0, 1)
                with self.assertRaises(NotImplementedError):
                    qfl.tile_connector = not_imp_connector
            with self.assertRaises(ValueError):
                QuoFloor(corner_qtile=corner_qtile2, dim = (m+1, m+1))
            corner_qtile3 = [ZNode((0, 1)), ZNode((1, 0), ZShape(m=m))]
            with self.assertRaises(ValueError):
                QuoFloor(corner_qtile=corner_qtile3, dim = (1, 1))
        with self.assertRaises(TypeError):
            QuoFloor(corner_qtile=bad_corner_qtile, dim = (1, 1))
        bad_coords = [(k, k+1) for k in range(4)] + [(k+1, k) for k in range(4)] + [(6, 3)]
        bad_corner_qtile2 = [ZNode(coord=coord) for coord in bad_coords]
        with self.assertRaises(ValueError):
            QuoFloor(corner_qtile=bad_corner_qtile2, dim=(1, 1))

    def test_qtile_xy(self) -> None:
        coords = [(k, k+1) for k in range(4)] + [(k+1, k) for k in range(4)]
        corner_qtile = QuoTile([ZNode(coord=coord) for coord in coords])
        qfl = QuoFloor(corner_qtile=corner_qtile, dim=(10, 7))
        # Check the ZNodes in each tile are as expected
        for x, y in product([1, 6, 8], [1, 2, 3, 6]):
            xy_tile = qfl.qtile_xy(x, y)
            coords_xy = sorted([(a+4*x, b+4*y) for (a, b) in coords])
            xy_tile_expected_zns = [ZNode(coord = c) for c in coords_xy]
            self.assertEqual(xy_tile.zns, xy_tile_expected_zns)

    def test_qtiles(self) -> None:
        coords = [(k, k+1) for k in range(4)] + [(k+1, k) for k in range(4)]
        corner_qtile = QuoTile([ZNode(coord=coord) for coord in coords])
        for a, b in product([1, 6, 8], [1, 3, 6]):
            qfl = QuoFloor(corner_qtile=corner_qtile, dim=(a, b))
            tiles = qfl.qtiles
            # Check it has m*n values in it
            self.assertEqual(len(tiles), a*b)
            # Check each tile contains the same number of znodes, none is a repeat
            for xy, xy_tile in tiles.items():
                self.assertEqual(len(set(xy_tile)), len(set(coords)))
                self.assertEqual(len(set(xy_tile)), len(xy_tile))
                self.assertTrue(xy in product(range(a), range(b)))
        for m in [4, 6, 7]:
            corner_qtile = QuoTile([ZNode(coord=coord, shape=ZShape(m=m)) for coord in coords])
            for a, b in product([1, 6, 8], [1, 3, 6]):
                if a <= m and b <= m:
                    qfl = QuoFloor(corner_qtile=corner_qtile, dim=(a, b))
                    tiles = qfl.qtiles
                    # Check it has m*n values in it
                    self.assertEqual(len(tiles), a*b)
                    # Check each tile contains the same number of znodes, none is a repeat
                    for xy, xy_tile in tiles.items():
                        self.assertEqual(len(set(xy_tile)), len(set(coords)))
                        self.assertEqual(len(set(xy_tile)), len(xy_tile))
                        self.assertTrue(xy in product(range(a), range(b)))
                else:
                    with self.assertRaises(ValueError):
                        QuoFloor(corner_qtile=corner_qtile, dim=(a, b))

    def test_zns(self) -> None:
        coords = [(k, k+1) for k in range(4)] + [(k+1, k) for k in range(3)]
        corner_qtile = QuoTile([ZNode(coord=coord) for coord in coords])
        for m, n in product([1, 6, 8], [1, 2, 3, 6]):
            qfl = QuoFloor(corner_qtile=corner_qtile, dim=(m, n))
            tiles_zns = qfl.zns
            # Check it has m*n values in it
            self.assertEqual(len(tiles_zns), m*n)
            # Check each tile contains the same number of znodes, none is a repeat
            for xy, xy_zns in tiles_zns.items():
                self.assertEqual(len(set(xy_zns)), len(set(coords)))
                self.assertEqual(len(set(xy_zns)), len(xy_zns))
                self.assertTrue(xy in product(range(m), range(n)))

    def test_ver_zns(self) -> None:
        coords = [(0, 1), (1, 0)]
        nodes = [ZNode(coord=c) for c in coords]
        nodes += [n + p for n in nodes for p in [PlaneShift(2, 0), PlaneShift(0, 2), PlaneShift(2, 2)]]
        for dim in [(1, 1), (3, 5), (6, 6)]:
            qfl = QuoFloor(corner_qtile=nodes, dim=dim)
            qfl_ver = qfl.ver_zns
            for xy , xy_ver in qfl_ver.items():
                self.assertTrue(xy in product(range(dim[0]), range(dim[1])))
                for v in xy_ver:
                    self.assertTrue(v.is_vertical())
            self.assertEqual(len(qfl_ver), dim[0]*dim[1])

    def test_hor_zns(self) -> None:
        coords = [(0, 1), (1, 2)]
        nodes = [ZNode(coord=c) for c in coords]
        nodes += [n + p for n in nodes for p in [PlaneShift(2, 0), PlaneShift(0, 2), PlaneShift(2, 2)]]
        for dim in [(4, 7), (1, 1)]:
            qfl = QuoFloor(corner_qtile=nodes, dim=dim)
            qfl_hor = qfl.hor_zns
            for xy , xy_hor in qfl_hor.items():
                self.assertTrue(xy in product(range(dim[0]), range(dim[1])))
                for v in xy_hor:
                    self.assertTrue(v.is_horizontal())
            self.assertEqual(len(qfl_hor), dim[0]*dim[1])
        for m in [4, 5]:
            nodes2 = [ZNode(coord=c, shape=ZShape(m=m)) for c in coords]
            nodes2 += [n + p for n in nodes2 for p in [PlaneShift(2, 0), PlaneShift(0, 2), PlaneShift(2, 2)]]
            for dim in [(4, 7), (1, 1), (4, 4), (2, 3)]:
                if dim[0] <= m and dim[1] <= m: 
                    qfl = QuoFloor(corner_qtile=nodes2, dim=dim)
                    qfl_hor = qfl.hor_zns
                    for xy , xy_hor in qfl_hor.items():
                        self.assertTrue(xy in product(range(dim[0]), range(dim[1])))
                        for v in xy_hor:
                            self.assertTrue(v.is_horizontal())
                    self.assertEqual(len(qfl_hor), dim[0]*dim[1])
                else:
                    with self.assertRaises(ValueError):
                        QuoFloor(corner_qtile=nodes2, dim=dim)


    def test_quo_ext_paths(self) -> None:
        coords1 = [(2, 1), (1, 2)]
        nodes1 = [ZNode(coord=c) for c in coords1]
        nodes1 += [n + p for n in nodes1 for p in [PlaneShift(2, 0), PlaneShift(0, 2), PlaneShift(2, 2)]]
        coords2 = [(k, k+1) for k in range(4)] + [(k+1, k) for k in range(3)]
        nodes2 = [ZNode(coord=coord) for coord in coords2]
        for dim in [(1, 1), (5, 5), (3, 6)]:
            floors = [QuoFloor(corner_qtile=nodes1, dim=dim), QuoFloor(corner_qtile=nodes2, dim=dim)]
            for qfl in floors:
                ext_paths = qfl.quo_ext_paths
                self.assertTrue("col" in ext_paths)
                self.assertTrue("row" in ext_paths)
                col_ext_paths = ext_paths["col"]
                row_ext_paths = ext_paths["row"]
                self.assertEqual(len(col_ext_paths), dim[0])
                self.assertEqual(len(row_ext_paths), dim[1])
                for list_c in col_ext_paths.values():
                    # Check no repeat
                    uwjs_c = [x[0] for x in list_c]
                    self.assertEqual(len(uwjs_c), len(set(uwjs_c)))
                    for _, zse in list_c:
                        self.assertEqual(zse.z_end-zse.z_start+1, dim[1])
                for list_r in row_ext_paths.values():
                    # Check no repeat
                    uwjs_r = [x[0] for x in list_r]
                    self.assertEqual(len(uwjs_r), len(set(uwjs_r)))
                    for _, zse in list_r:
                        self.assertEqual(zse.z_end-zse.z_start+1, dim[0])
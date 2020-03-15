import random
import unittest

import dwave_networkx as dnx
import minorminer.layout as mml
import networkx as nx


class TestPlacement(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestPlacement, self).__init__(*args, **kwargs)

        # Graphs for testing
        self.S = nx.random_regular_graph(3, 50)
        self.G = nx.random_regular_graph(3, 150)
        self.C = dnx.chimera_graph(4)
        self.C_coord = dnx.chimera_graph(4, coordinates=True)
        self.P = dnx.pegasus_graph(4)

        # Layouts for testing
        self.S_layout = mml.p_norm(self.S)
        self.S_layout_3 = mml.p_norm(self.S, d=3)
        self.G_layout = mml.p_norm(self.G)
        self.C_layout = mml.dnx_layout(self.C)
        self.C_coord_layout = mml.dnx_layout(self.C_coord)
        self.P_layout = mml.dnx_layout(self.P)

    def test_closest(self):
        """
        Tests that closest placement is working correctly
        """
        # Test different input versions of T
        placement_1 = mml.closest(self.S_layout, self.C_layout)
        placement_2 = mml.closest(self.S_layout, self.C)
        placement_3 = mml.closest(self.S_layout, self.C_layout.layout)
        self.assertDictEqual(placement_1, placement_2)
        self.assertDictEqual(placement_2, placement_3)

        # Test the parameters: Randomly generate subset sizes and number of neighbors to query
        max_subset_size_1 = (1, random.randint(2, 5))
        lb = random.randint(2, 5)
        max_subset_size_2 = (lb, random.randint(lb, 5))
        num_neighbors = random.randint(2, 20)

        mml.closest(
            self.S_layout, self.C_layout, max_subset_size=max_subset_size_1)
        mml.closest(
            self.S_layout, self.C_layout, max_subset_size=max_subset_size_1, num_neighbors=num_neighbors)
        mml.closest(
            self.S_layout, self.C_layout, max_subset_size=max_subset_size_2)
        mml.closest(
            self.S_layout, self.C_layout, max_subset_size=max_subset_size_2, num_neighbors=num_neighbors)

        # Test non dnx graphs
        mml.closest(self.S_layout, self.S)

    def test_injective(self):
        """
        Tests that injective placement is working correctly.
        """
        # Test different input versions of T
        placement_1 = mml.injective(self.S_layout, self.C_layout)
        placement_2 = mml.injective(self.S_layout, self.C)
        placement_3 = mml.injective(self.S_layout, self.C_layout.layout)
        self.assertDictEqual(placement_1, placement_2)
        self.assertDictEqual(placement_2, placement_3)

        # Test non dnx graphs
        mml.injective(self.S_layout, self.S)

    def test_intersection(self):
        """
        Tests that intersection placement is working correctly.
        """
        # Test different input versions of T
        placement_1 = mml.intersection(self.S_layout, self.C_layout)
        placement_2 = mml.intersection(self.S_layout, self.C)
        self.assertDictEqual(placement_1, placement_2)

        # Test a coordinate version of chimera
        mml.intersection(self.S_layout, self.C_coord_layout)

        # Test bad inputs
        # Dictionary is not allowed for T
        self.assertRaises(TypeError, mml.intersection,
                          self.S_layout, self.C_layout.layout)
        # Pegasus is not allowed
        self.assertRaises(NotImplementedError, mml.intersection,
                          self.S_layout, self.P)
        # Layouts must be 2d
        self.assertRaises(NotImplementedError, mml.intersection,
                          self.S_layout_3, self.C)

        # Test the parameters
        mml.intersection(self.S_layout, self.C_layout, fill_processor=False)

    def test_binning(self):
        """
        Tests that intersection placement is working correctly.
        """
        # Test different input versions of T
        placement_1 = mml.binning(
            self.S_layout, self.C_layout, strategy="cycle")
        placement_2 = mml.binning(self.S_layout, self.C, strategy="cycle")
        self.assertDictEqual(placement_1, placement_2)

        # Test a coordinate version of chimera
        mml.binning(self.S_layout, self.C_coord_layout)

        # # Test pegasus
        # mml.binning(self.S_layout, self.P_layout)

        # Test bad inputs
        # Dictionary is not allowed for T
        self.assertRaises(TypeError, mml.binning,
                          self.S_layout, self.C_layout.layout)
        # G is too big for C
        self.assertRaises(RuntimeError, mml.binning,
                          self.G_layout, self.C_layout)
        # T must be a dnx graph
        self.assertRaises(NotImplementedError, mml.binning,
                          self.S_layout, self.G_layout)
        # Layouts must be 2d
        self.assertRaises(NotImplementedError, mml.binning,
                          self.S_layout_3, self.C)

        # Test the parameters
        mml.binning(self.S_layout, self.C_layout)
        mml.binning(self.S_layout, self.C_layout, strategy="all")

        # Unit_tile_capacity too small
        self.assertRaises(RuntimeError, mml.binning,
                          self.S_layout, self.C, 3)
        # Topple failure
        # FIXME: This gives 100% coverage, but makes the test take longer.
        # self.assertRaises(RuntimeError, mml.binning,
        #                   self.S_layout, self.C, 1)


if __name__ == '__main__':
    unittest.main()

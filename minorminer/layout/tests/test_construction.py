import random
import unittest

import dwave_networkx as dnx
import minorminer.layout as mml
import networkx as nx


class TestConstruction(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestConstruction, self).__init__(*args, **kwargs)

        # Graphs for testing
        self.S = nx.random_regular_graph(3, 50)
        self.C = dnx.chimera_graph(4)
        self.C_coord = dnx.chimera_graph(4, coordinates=True)

        # Layouts for testing
        self.S_layout = mml.p_norm(self.S)
        self.C_layout = mml.dnx_layout(self.C)
        self.C_coord_layout = mml.dnx_layout(self.C_coord)

        # Placements for testing
        self.closest = mml.closest(self.S_layout, self.C_layout)
        self.coord_closest = mml.closest(self.S_layout, self.C_coord_layout)
        self.injective = mml.injective(self.S_layout, self.C_layout)
        self.intersection = mml.intersection(self.S_layout, self.C_layout)
        self.binning = mml.binning(self.S_layout, self.C_layout)
        C_nodes = list(self.C)
        self.random = {v: random.choice(C_nodes) for v in self.S}

    def test_pass_along(self):
        """
        Tests that pass_along construction is working correctly.
        """
        # Test different placements
        mml.pass_along(self.closest)
        mml.pass_along(self.injective)
        mml.pass_along(self.intersection)
        mml.pass_along(self.binning)
        mml.pass_along(self.random)

    def test_crosses(self):
        """
        Tests that crosses construction is working correctly.
        """
        # Test different placements
        mml.crosses(self.closest, self.S_layout, self.C_layout)
        mml.crosses(self.coord_closest, self.S_layout, self.C_coord_layout)
        mml.crosses(self.injective, self.S_layout, self.C_layout)
        mml.crosses(self.intersection, self.S_layout, self.C_layout)
        mml.crosses(self.binning, self.S_layout, self.C_layout)
        mml.crosses(self.random, self.S_layout, self.C_layout)

    def test_neighborhood(self):
        """
        Tests that crosses construction is working correctly.
        """
        # Test different placements
        mml.neighborhood(self.closest, self.S_layout, self.C_layout)
        mml.neighborhood(self.injective, self.S_layout, self.C_layout)
        mml.neighborhood(self.intersection, self.S_layout, self.C_layout)
        mml.neighborhood(self.binning, self.S_layout, self.C_layout)
        mml.neighborhood(self.random, self.S_layout, self.C_layout)

        # Test the parameters
        mml.neighborhood(self.closest, self.S_layout,
                         self.C_layout, second=True)
        mml.neighborhood(self.random, self.S_layout,
                         self.C_layout, second=True)


if __name__ == '__main__':
    unittest.main()

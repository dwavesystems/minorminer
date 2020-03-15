import random
import unittest

import dwave_networkx as dnx
import minorminer.layout as mml
import networkx as nx


class TestConnection(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestConnection, self).__init__(*args, **kwargs)

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

    def test_crosses(self):
        """
        Tests that crosses construction is working correctly.
        """
        # Test different placements
        mml.crosses(self.S_layout, self.C_layout, self.closest)
        mml.crosses(self.S_layout, self.C_coord_layout, self.coord_closest)
        mml.crosses(self.S_layout, self.C_layout, self.injective)
        mml.crosses(self.S_layout, self.C_layout, self.intersection)
        mml.crosses(self.S_layout, self.C_layout, self.binning)
        mml.crosses(self.S_layout, self.C_layout, self.random)

    def test_shortest_paths(self):
        """
        Tests that crosses construction is working correctly.
        """
        # Test different placements
        mml.shortest_paths(self.S_layout, self.C_layout, self.closest)
        mml.shortest_paths(
            self.S_layout, self.C_coord_layout, self.coord_closest)
        mml.shortest_paths(self.S_layout, self.C_layout, self.injective)
        mml.shortest_paths(self.S_layout, self.C_layout, self.intersection)
        mml.shortest_paths(self.S_layout, self.C_layout, self.binning)
        mml.shortest_paths(self.S_layout, self.C_layout, self.random)


if __name__ == '__main__':
    unittest.main()

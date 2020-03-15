import random
import unittest

import dwave_networkx as dnx
import minorminer.layout as mml
import networkx as nx


class TestExpansion(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestExpansion, self).__init__(*args, **kwargs)

        # Graphs for testing
        self.S = nx.random_regular_graph(3, 50)
        self.C = dnx.chimera_graph(4)

        # Layouts for testing
        self.S_layout = mml.p_norm(self.S)
        self.C_layout = mml.dnx_layout(self.C)

        # Placements for testing
        self.closest = mml.closest(self.S_layout, self.C_layout)
        self.intersection = mml.intersection(self.S_layout, self.C_layout)

        # Connections for testing
        self.crosses = mml.crosses(
            self.S_layout, self.C_layout, self.intersection)
        self.shortest_paths = mml.shortest_paths(
            self.S_layout, self.C_layout, self.intersection)

    def test_neighborhood(self):
        """
        Tests that crosses construction is working correctly.
        """
        # Test different placements
        mml.neighborhood(self.S_layout, self.C_layout, self.closest)
        mml.neighborhood(self.S_layout, self.C_layout, self.intersection)

        # Test different connections
        mml.neighborhood(self.S_layout, self.C_layout, self.crosses)
        mml.neighborhood(self.S_layout, self.C_layout, self.shortest_paths)

        # Test the parameters
        mml.neighborhood(self.S_layout, self.C_layout,
                         self.closest, second=True)
        mml.neighborhood(self.S_layout, self.C_layout,
                         self.crosses, second=True)


if __name__ == '__main__':
    unittest.main()

import random
import unittest

import dwave_networkx as dnx
import minorminer.layout as mml
import networkx as nx


class TestHinting(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestHinting, self).__init__(*args, **kwargs)

        # Graphs for testing
        self.S = nx.random_regular_graph(3, 50)
        self.C = dnx.chimera_graph(4)

        # Layouts for testing
        S_layout = mml.p_norm(self.S)
        C_layout = mml.dnx_layout(self.C)

        # The placement for testing
        intersection = mml.intersection(S_layout, C_layout)

        # Constructions for testing
        self.pass_along = mml.pass_along(intersection)
        self.crosses = mml.crosses(intersection, S_layout, C_layout)

    def test_initial(self):
        """
        Tests that initial hinting is working correctly.
        """
        # Test different placements
        mml.initial(self.S, self.C, self.pass_along)
        mml.initial(self.S, self.C, self.crosses)

        # Test the parameters
        mml.initial(self.S, self.C, self.pass_along, extend=True)
        mml.initial(self.S, self.C, self.pass_along, percent=2/3)
        mml.initial(self.S, self.C, self.crosses, percent=2/3)
        mml.initial(self.S, self.C, self.pass_along, extend=True, percent=2/3)

    def test_suspend(self):
        """
        Tests that initial hinting is working correctly.
        """
        # Test different placements
        mml.suspend(self.S, self.C, self.pass_along)
        mml.suspend(self.S, self.C, self.crosses)

        # Test the parameters
        mml.suspend(self.S, self.C, self.pass_along, extend=True)
        mml.suspend(self.S, self.C, self.pass_along, percent=2/3)
        mml.suspend(self.S, self.C, self.crosses, percent=2/3)
        mml.suspend(self.S, self.C, self.pass_along, extend=True, percent=2/3)


if __name__ == '__main__':
    unittest.main()

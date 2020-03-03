import unittest

import dwave_networkx as dnx
import minorminer.layout as mml
import networkx as nx


class TestPlacement(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestPlacement, self).__init__(*args, **kwargs)

        self.S = nx.random_regular_graph(3, 50)
        self.T = dnx.chimera_graph(4)
        self.S_layout = mml.p_norm(self.S)
        self.T_layout = mml.dnx_layout(self.T)

    def test_closest(self):
        """
        Tests that closest placement is correct for embedding a random cubic graph, S, in Chimera(4); i.e. that points 
        in the layout of S are greedily matched to their closest counter parts in the layout of Chimera(4).
        """
        _ = mml.closest(self.S_layout, self.T_layout)

    def test_injective(self):
        """
        Tests that injective placement is correct for embedding random cubic graph, S, in Chimera(4); i.e. that points 
        in the layout of S are optimally injectively mapped to the closest points in the layout of Chimera(4).
        """
        _ = mml.injective(self.S_layout, self.T_layout)

    def test_intersection(self):
        """
        Tests that injective placement is correct for embedding random cubic graph, S, in Chimera(4); i.e. that points 
        in the layout of S are optimally injectively mapped to the closest points in the layout of Chimera(4).
        """
        _ = mml.intersection(self.S_layout, self.T_layout)


if __name__ == '__main__':
    unittest.main()

import unittest

import dwave_networkx as dnx
import minorminer.layout as mml
import networkx as nx


class TestLayout(unittest.TestCase):

    def test_closest(self):
        """
        Tests that closest placement is correct for embedding a random cubic graph, S, in Chimera(4); i.e. that points 
        in the layout of S are greedily matched to their closest counter parts in the layout of Chimera(4).
        """
        S = nx.random_regular_graph(3, 20)
        S_layout = mml.kamada_kawai(S)
        T = dnx.chimera_graph(4)
        T_layout = mml.dnx_layout(T)

        mml.closest(S_layout, T_layout)

    def test_injective(self):
        """
        Tests that injective placement is correct for embedding random cubic graph, S, in Chimera(4); i.e. that points 
        in the layout of S are optimally injectively mapped to the closest points in the layout of Chimera(4).
        """
        S = nx.random_regular_graph(3, 20)
        S_layout = mml.kamada_kawai(S)
        T = dnx.chimera_graph(4)
        T_layout = mml.dnx_layout(T)

        mml.injective(S_layout, T_layout)

    def test_binning(self):
        """
        Tests that injective placement is correct for embedding random cubic graph, S, in Chimera(4); i.e. that points 
        in the layout of S are optimally injectively mapped to the closest points in the layout of Chimera(4).
        """
        S = nx.random_regular_graph(3, 20)
        S_layout = mml.kamada_kawai(S)
        T = dnx.chimera_graph(4)
        T_layout = mml.dnx_layout(T)

        mml.binning(S_layout, T_layout)


if __name__ == '__main__':
    unittest.main()

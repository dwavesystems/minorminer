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
        P = dnx.pegasus_graph(4)
        P_layout = mml.dnx_layout(P)
        P_coord = dnx.pegasus_graph(4, coordinates=True)
        P_coord_layout = mml.dnx_layout(P_coord)
        P_nice = dnx.pegasus_graph(4, nice_coordinates=True)
        P_nice_layout = mml.dnx_layout(P_nice)

        mml.binning(S_layout, T_layout)
        mml.binning(S_layout, P_layout)
        mml.binning(S_layout, P_coord_layout)
        mml.binning(S_layout, P_nice_layout)


if __name__ == '__main__':
    unittest.main()

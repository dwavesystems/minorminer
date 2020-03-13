import random
import unittest

import dwave_networkx as dnx
import minorminer.layout as mml
import networkx as nx


class TestFull(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestFull, self).__init__(*args, **kwargs)

        # Graphs for testing
        self.S = nx.random_regular_graph(3, 50)
        self.G = nx.random_regular_graph(3, 150)
        self.C = dnx.chimera_graph(4)

        # Compute some layouts
        self.S_layout = mml.p_norm(self.S)
        self.C_layout = mml.dnx_layout(self.C)

        # Layout kwargs
        self.d = 3
        self.seed = 2035
        self.center = (0, 0, 0)
        self.scale = 2
        self.rescale = False

        # Placement kwargs
        self.max_subset_size = (2, 2)
        self.strategy = "all"
        self.num_neighbors = 3
        self.full_fit = False
        self.unit_tile_capacity = 3

        # Construction kwargs
        self.second = True

        # Hinting kwargs
        self.percent = 2/3
        self.extend = True

        # A minorminer.find_embedding kwarg
        self.timeout = 2

    def test_find_embedding(self):
        """
        Tests that find_embedding is working correctly.
        """
        # Test weird layout input configurations
        mml.find_embedding(self.S, self.C, layout=(
            self.S_layout.layout, self.C_layout.layout))
        mml.find_embedding(self.S, self.C, layout=(
            self.S_layout, self.C_layout))
        mml.find_embedding(self.S, self.C, layout=(mml.p_norm, mml.dnx_layout))

        # Test passing in a placement dictionary
        C_nodes = list(self.C)
        mml.find_embedding(self.S, self.C, placement={
                           v: random.choice(C_nodes) for v in self.S})

        # Test a non dnx_graph
        mml.find_embedding(self.S, self.G, placement=mml.closest,
                           construction=mml.pass_along)

        # Test a non graph
        mml.find_embedding(self.S.edges, self.C)

        # Test all the options at once
        mml.find_embedding(
            self.S,
            self.C,
            placement=mml.closest,
            construction=mml.pass_along,
            d=self.d,
            seed=self.seed,
            center=self.center,
            scale=self.scale,
            rescale=self.rescale,
            max_subset_size=self.max_subset_size,
            strategy=self.strategy,
            num_neighbors=self.num_neighbors,
            full_fit=self.full_fit,
            unit_tile_capacity=self.unit_tile_capacity,
            second=self.second,
            percent=self.percent,
            extend=self.extend,
            timeout=self.timeout,
            return_layouts=True
        )


if __name__ == '__main__':
    unittest.main()

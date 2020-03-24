import unittest

import minorminer.layout.placement as mml
from .data import precomputed_chimera, precomputed_random_cubic_layout, precomputed_closest, precomputed_injective


class TestLayout(unittest.TestCase):

    def test_closest(self):
        """
        Tests that closest placement is correct for embedding a random cubic graph, S, in Chimera(4); i.e. that points 
        in the layout of S are greedily matched to their closest counter parts in the layout of Chimera(4).
        """
        S_layout = precomputed_random_cubic_layout
        T_layout = precomputed_chimera[2]

        placement = mml.closest(S_layout, T_layout)
        self.assertEqual(placement, precomputed_closest)

    def test_injective(self):
        """
        Tests that injective placement is correct for embedding random cubic graph, S, in Chimera(4); i.e. that points 
        in the layout of S are optimally injectively mapped to the closest points in the layout of Chimera(4).
        """
        S_layout = precomputed_random_cubic_layout
        T_layout = precomputed_chimera[2]

        placement = mml.injective(S_layout, T_layout)
        self.assertEqual(placement, precomputed_injective)


if __name__ == '__main__':
    unittest.main()

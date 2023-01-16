from minorminer import subgraph
from minorminer.utils import verify_embedding
import unittest, random, itertools, dwave_networkx as dnx, networkx as nx, os


class TestSubgraph(unittest.TestCase):
    def test_smoketest(self):
        b = nx.complete_bipartite_graph(4, 4)
        c = nx.cubical_graph()
        emb = subgraph.find_subgraph(c, b)
        emb = {k: [v] for k, v in emb.items()}
        verify_embedding(emb, c, b)


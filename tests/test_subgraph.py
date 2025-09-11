from minorminer import subgraph
from minorminer.utils import verify_embedding
from dwave_networkx import chimera_graph
import unittest, random, itertools, dwave_networkx as dnx, networkx as nx, os


class TestSubgraph(unittest.TestCase):
    def test_smoketest(self):
        b = nx.complete_bipartite_graph(4, 4)
        c = nx.cubical_graph()
        emb = subgraph.find_subgraph(c, b)
        emb = {k: [v] for k, v in emb.items()}
        verify_embedding(emb, c, b)

    def test_as_embedding(self):
        b = nx.complete_bipartite_graph(4, 4)
        c = nx.cubical_graph()
        emb = subgraph.find_subgraph(c, b, as_embedding=True)
        verify_embedding(emb, c, b)

    def test_node_colors(self):
        path_5 = nx.path_graph(5)
        path_10 = nx.path_graph(10)
        node_labels = {0: "start", 4:"end"}, {5: "start", 1: "end"}
        emb = subgraph.find_subgraph(path_5, path_10, node_labels=node_labels)
        for v in path_5:
           self.assertEqual(node_labels[0].get(v), node_labels[1].get(emb[v]))

        node_labels = {0: "start", 4:"end"}, {5: "start", 0: "end"} #impossible
        emb = subgraph.find_subgraph(path_5, path_10, node_labels=node_labels)
        self.assertEqual(emb, {})

    def test_edge_colors(self):
        path_5 = nx.path_graph(5)
        path_10 = nx.path_graph(10)

        #this is a directed edge thing
        edge_labels = {(0,1): "start"}, {(1, 0): "start"} #impossible
        emb = subgraph.find_subgraph(path_5, path_10, edge_labels=edge_labels)
        self.assertEqual(emb, {})

        #try that again but undirected
        edge_labels = {(0,1): "start", (1, 0): "start"}, {(0,1): "start", (1, 0): "start"}
        emb = subgraph.find_subgraph(path_5, path_10, edge_labels=edge_labels)
        for uv in path_5.edges:
            u, v = uv
            pq = emb[u], emb[v]
            self.assertEqual(edge_labels[0].get(uv), edge_labels[1].get(pq))
            self.assertEqual(edge_labels[0].get(uv[::-1]), edge_labels[1].get(pq[::-1]))

        #now with a solvable directed problem
        edge_labels = {(0,1): "start"}, {(7, 6): "start"}
        emb = subgraph.find_subgraph(path_5, path_10, edge_labels=edge_labels)
        for uv in path_5.edges:
            u, v = uv
            pq = emb[u], emb[v]
            self.assertEqual(edge_labels[0].get(uv), edge_labels[1].get(pq))
            self.assertEqual(edge_labels[0].get(uv[::-1]), edge_labels[1].get(pq[::-1]))
            
    def test_timeout(self):
        source = chimera_graph(8)
        target = chimera_graph(15, coordinates=True)
        #pop out a vertex from the central tile to make a minimally-impossible
        #problem (no fully-yielded 8x8s) that GSS know how to reason about
        target.remove_node((7,7,0,0)) 
        emb = subgraph.find_subgraph(source, target, timeout=2)
        self.assertEqual(emb, {})


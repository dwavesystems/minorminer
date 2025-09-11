from minorminer import subgraph
from minorminer.utils import verify_embedding
import unittest, random, itertools, dwave_networkx as dnx, networkx as nx, os

def verify_homomorphism(emb, source, target, locallyinjective = False):
    for s in source:
        if s not in emb:
            raise RuntimeError("homomorphism node fail")

        nbrs = {emb[v] for v in source[s]}
        if not nbrs.issubset(target[emb[s]]):
            raise RuntimeError("homomorphism edge fail")

        if locallyinjective and len(source[s]) + 1 > len(nbrs | {emb[s]}):
            raise RuntimeError("homomorphism locality fail")

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
        source = dnx.chimera_graph(8)
        target = dnx.chimera_graph(15, coordinates=True)
        #pop out a vertex from the central tile to make a minimally-impossible
        #problem (no fully-yielded 8x8s) that GSS know how to reason about
        target.remove_node((7,7,0,0)) 
        emb = subgraph.find_subgraph(source, target, timeout=2)
        self.assertEqual(emb, {})

    def test_noninjective(self):
        # find a 2-coloring of Chimera!
        source = dnx.chimera_graph(4)
        target = nx.path_graph(2)
        emb = subgraph.find_subgraph(source, target, injectivity='noninjective')
        verify_homomorphism(emb, source, target)

        # find a 4-coloring of zephyr!
        source = dnx.zephyr_graph(4, t=1)
        target = nx.complete_graph(4)
        emb = subgraph.find_subgraph(source, target, injectivity='noninjective')
        verify_homomorphism(emb, source, target)

        # but not a 3-coloring!
        target = nx.complete_graph(3)
        emb = subgraph.find_subgraph(source, target, injectivity='noninjective')
        self.assertEqual(emb, {})

    def test_locally_injective(self):
        # find a triple-cover of a 3-cycle by a 9-cycle
        source = nx.cycle_graph(9)
        target = nx.cycle_graph(3)
        emb = subgraph.find_subgraph(source, target, injectivity='locally injective')
        verify_homomorphism(emb, source, target, locallyinjective=True)

        # can't find a locally injective homomorphism for a 10-cycle into a 3-cycle
        source = nx.cycle_graph(10)
        emb = subgraph.find_subgraph(source, target, injectivity='locally injective')
        self.assertEqual(emb, {})

    def test_isolated_nodes(self):
        source = nx.cycle_graph(9)
        target = nx.cycle_graph(9)
        source.add_node('a')

        emb = subgraph.find_subgraph(source, target)
        self.assertEqual(emb, {})

        emb = subgraph.find_subgraph(source, target, injectivity='locally injective')
        verify_homomorphism(emb, source, target, locallyinjective=True)

        target.add_node('a')
        emb = subgraph.find_subgraph(source, target)
        verify_embedding(emb, source, target)

        target.add_edge('a', 0)
        emb = subgraph.find_subgraph(source, target)
        verify_embedding(emb, source, target)


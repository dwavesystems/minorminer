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

def as_embedding(emb):
    return {k: (v,) for k, v in emb.items()}

class TestSubgraph(unittest.TestCase):
    def verify_node_coloring(self, emb, source, node_labels):
        for v in source:
           self.assertEqual(node_labels[0].get(v), node_labels[1].get(emb[v]))

    def verify_edge_coloring(self, emb, source, edge_labels):
        for uv in source.edges:
            u, v = uv
            pq = emb[u], emb[v]
            self.assertEqual(edge_labels[0].get(uv), edge_labels[1].get(pq))
            self.assertEqual(edge_labels[0].get(uv[::-1]), edge_labels[1].get(pq[::-1]))

    def test_smoketest(self):
        b = nx.complete_bipartite_graph(4, 4)
        c = nx.cubical_graph()
        emb = subgraph.find_subgraph(c, b)
        verify_embedding(as_embedding(emb), c, b)

    def test_as_embedding(self):
        b = nx.complete_bipartite_graph(4, 4)
        c = nx.cubical_graph()
        emb = subgraph.find_subgraph(c, b, as_embedding=True)
        verify_embedding(emb, c, b)

    def test_node_colors(self):
        path_5 = nx.path_graph(5)
        path_10 = nx.path_graph('qrstuvwxyz')
        node_labels = {0: "start", 4:"end"}, {'v': "start", 'r': "end"}
        emb = subgraph.find_subgraph(path_5, path_10, node_labels=node_labels)
        self.verify_node_coloring(emb, path_5, node_labels)
        verify_embedding(as_embedding(emb), path_5, path_10)

        #do it again with generators instead of graphs
        emb = subgraph.find_subgraph(iter(path_5.edges), iter(path_10.edges), node_labels=node_labels)
        self.verify_node_coloring(emb, path_5, node_labels)
        verify_embedding(as_embedding(emb), path_5, path_10)

        #and again with edge lists
        emb = subgraph.find_subgraph(list(path_5.edges), list(path_10.edges), node_labels=node_labels)
        self.verify_node_coloring(emb, path_5, node_labels)
        verify_embedding(as_embedding(emb), path_5, path_10)

        node_labels = {0: "start", 4:"end"}, {'v': "start", 'q': "end"} #impossible
        emb = subgraph.find_subgraph(path_5, path_10, node_labels=node_labels)
        self.assertEqual(emb, {})

        

    def test_edge_colors(self):
        path_5 = nx.path_graph('vwxyz')
        path_10 = nx.path_graph(10)

        #this is a directed edge thing
        edge_labels = {('v','w'): "start"}, {(1, 0): "start"} #impossible
        emb = subgraph.find_subgraph(path_5, path_10, edge_labels=edge_labels)
        self.assertEqual(emb, {})

        #try that again but undirected
        edge_labels = {('v','w'): "start", ('w', 'v'): "start"}, {(0,1): "start", (1, 0): "start"}
        emb = subgraph.find_subgraph(path_5, path_10, edge_labels=edge_labels)
        self.verify_edge_coloring(emb, path_5, edge_labels)
        verify_embedding(as_embedding(emb), path_5, path_10)

        #now with a solvable directed problem
        edge_labels = {('v','w'): "start"}, {(7, 6): "start"}
        emb = subgraph.find_subgraph(path_5, path_10, edge_labels=edge_labels)
        self.verify_edge_coloring(emb, path_5, edge_labels)
        verify_embedding(as_embedding(emb), path_5, path_10)
            
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
        source = nx.cycle_graph('rstuvwxyz')
        target = nx.cycle_graph(3)
        emb = subgraph.find_subgraph(source, target, injectivity='locally injective')
        verify_homomorphism(emb, source, target, locallyinjective=True)

        # can't find a locally injective homomorphism for a 10-cycle into a 3-cycle
        source = nx.cycle_graph(10)
        emb = subgraph.find_subgraph(source, target, injectivity='locally injective')
        self.assertEqual(emb, {})

    def test_isolated_nodes(self):
        source = nx.cycle_graph('rstuvwxyz')
        target = nx.cycle_graph(9)
        source.add_node('a')

        emb = subgraph.find_subgraph(source, target)
        self.assertEqual(emb, {})

        emb = subgraph.find_subgraph(source, target, injectivity='locally injective')
        verify_homomorphism(emb, source, target, locallyinjective=True)

        target.add_node('a')
        emb = subgraph.find_subgraph(source, target, as_embedding=True)
        verify_embedding(emb, source, target)

        target.add_edge('a', 'x')
        emb = subgraph.find_subgraph(source, target, as_embedding=True)
        verify_embedding(emb, source, target)

    def test_seed(self):
        g = dnx.chimera_graph(3, 3, 1, coordinates=True)
        emb = subgraph.find_subgraph(g, g, seed=54321, as_embedding=True)
        verify_embedding(emb, g, g)

        emb = subgraph.find_subgraph(g, g, seed=random, as_embedding=True)
        verify_embedding(emb, g, g)

        emb = subgraph.find_subgraph(g, g, seed=random.Random(12345), as_embedding=True)
        verify_embedding(emb, g, g)

        #again, with iterable of edges
        emb = subgraph.find_subgraph(iter(g.edges), g, seed=random.Random(12345), as_embedding=True)
        verify_embedding(emb, g, g)

        #again, with list of edges
        emb = subgraph.find_subgraph(list(g.edges), g, seed=random.Random(12345), as_embedding=True)
        verify_embedding(emb, g, g)

    def test_seed_and_colors(self):
        g = nx.petersen_graph()
        label_nodes = random.sample(list(g), 2)
        e0, e1 = random.sample(list(g.edges), 2)
        node_labels = {x: "hello" for x in label_nodes}
        edge_labels = {x: "hello" for x in (e0, e1, e1[::-1])}
        node_labels = node_labels, node_labels
        edge_labels = edge_labels, edge_labels

        emb = subgraph.find_subgraph(g, g, seed=random, node_labels=node_labels)
        verify_embedding(as_embedding(emb), g, g)
        self.verify_node_coloring(emb, g, node_labels)
        
        emb = subgraph.find_subgraph(g, g, seed=random, edge_labels=edge_labels)
        verify_embedding(as_embedding(emb), g, g)
        self.verify_edge_coloring(emb, g, node_labels)

        emb = subgraph.find_subgraph(g, g, seed=random, node_labels=node_labels, edge_labels=edge_labels)
        verify_embedding(as_embedding(emb), g, g)
        self.verify_node_coloring(emb, g, node_labels)
        self.verify_edge_coloring(emb, g, node_labels)

        #again with list / iterable of edges
        emb = subgraph.find_subgraph(list(g.edges), iter(g.edges), seed=random, node_labels=node_labels)
        verify_embedding(as_embedding(emb), g, g)
        self.verify_node_coloring(emb, g, node_labels)
        
        emb = subgraph.find_subgraph(list(g.edges), iter(g.edges), seed=random, edge_labels=edge_labels)
        verify_embedding(as_embedding(emb), g, g)
        self.verify_edge_coloring(emb, g, node_labels)

        emb = subgraph.find_subgraph(list(g.edges), iter(g.edges), seed=random, node_labels=node_labels, edge_labels=edge_labels)
        verify_embedding(as_embedding(emb), g, g)
        self.verify_node_coloring(emb, g, node_labels)
        self.verify_edge_coloring(emb, g, node_labels)


from minorminer import busclique
import minorminer as mm, dwave.embedding as dwe
import unittest, random, itertools, dwave_networkx as dnx, networkx as nx, os

def subgraph_node_yield(g, q):
    """
    returns a subgraph of g produced by removing nodes with probability 1-q
    """
    h = g.copy()
    h.remove_nodes_from([v for v in h if random.random() > q])
    return h

def subgraph_edge_yield(g, p):
    """
    returns a subgraph of g produced by removing edges with probability 1-p.
    """
    h = g.copy()
    h.remove_edges_from([e for e in h.edges() if random.random() > p])
    return h

def subgraph_edge_yield_few_bad(g, p, b):
    """
    returns a subgraph of g produced by removing b internal edges and 
    odd/external edges with probability 1-p.

    g must be a chimera graph or a pegasus graph with coordinate labels
    """
    if g.graph['family'] == 'pegasus':
        def icpl(p, q):
            return p[0] != q[0]
    elif g.graph['family'] == 'chimera':
        def icpl(p, q):
            return p[2] != q[2]
    else:
        raise ValueError("g is neither a pegasus nor a chimera")

    h = g.copy()
    iedges, oedges = [], []
    for e in h.edges():
        if icpl(*e):
            iedges.append(e)
        elif random.random() > p:
            oedges.append(e)

    if len(iedges) >= b:
        iedges = random.sample(iedges, b)
    else:
        iedges = []

    h.remove_edges_from(itertools.chain(iedges, oedges))
    return h

def relabel(g, node_label, edge_label, *args, **kwargs):
    nodes = node_label(g)
    edges = edge_label(g.edges())

    if g.graph['family'] == 'pegasus':
        f = dnx.pegasus_graph
    elif g.graph['family'] == 'chimera':
        f = dnx.chimera_graph
    else:
        raise ValueError("g is neither a pegasus nor a chimera")

    return f(*args, edge_list = edges, node_list = nodes, **kwargs)

def max_chainlength(emb):
    if emb:
        return max(len(c) for c in emb.values())

class TestBusclique(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestBusclique, self).__init__(*args, **kwargs)

        self.c16 = dnx.chimera_graph(16)
        self.p16 = dnx.pegasus_graph(16)
        self.c16c = dnx.chimera_graph(16, coordinates=True, data = False)
        self.c428 = dnx.chimera_graph(4, n = 2, t = 8)
        self.c248 = dnx.chimera_graph(4, n = 2, t = 8)
        self.c42A = dnx.chimera_graph(4, n = 2, t = 9)
        c4c_0 = subgraph_node_yield(dnx.chimera_graph(4, coordinates = True), .95)
        p4c_0 = subgraph_node_yield(dnx.pegasus_graph(4, coordinates = True), .95)
        c4c = [c4c_0,
               subgraph_edge_yield(c4c_0, .95),
               subgraph_edge_yield_few_bad(c4c_0, .95, 6)]
        p4c = [p4c_0,
               subgraph_edge_yield(p4c_0, .95),
               subgraph_edge_yield_few_bad(p4c_0, .95, 6)]

        p4coords = dnx.pegasus_coordinates(4)
        c4coords = dnx.chimera_coordinates(4, 4, 4)

        c4 = [relabel(c,
                      c4coords.iter_chimera_to_linear,
                      c4coords.iter_chimera_to_linear_pairs,
                      4) for c in c4c]

        p4 = [relabel(p,
                      p4coords.iter_pegasus_to_linear,
                      p4coords.iter_pegasus_to_linear_pairs,
                      4) for p in p4c]

        p4n = [relabel(p,
                       p4coords.iter_pegasus_to_nice,
                       p4coords.iter_pegasus_to_nice_pairs,
                       4, nice_coordinates = True) for p in p4c]

        self.c4, self.c4_nd, self.c4_d = list(zip(c4, c4c))
        self.p4, self.p4_nd, self.c4_d = list(zip(p4, p4c, p4n))

    def test_p16(self):
        test = self.embed_battery(self.p16,
                    lambda nodes: dnx.pegasus_graph(16, node_list = nodes))
        size, cl = next(test)
        self.assertEqual(size, 180)
        self.assertEqual(cl, 17)
        s = nx.complete_graph(size)
        for i, (g, emb) in enumerate(test):
            self.assertEqual(len(emb), size)
            self.assertEqual(max_chainlength(emb), cl)
            dwe.verify_embedding(emb, s, g)

    def test_c16(self):
        test = self.embed_battery(self.c16,
                    lambda nodes: dnx.chimera_graph(16, node_list = nodes))
        size, cl = next(test)
        self.assertEqual(size, 64)
        self.assertEqual(cl, 17)
        s = nx.complete_graph(size)
        for i, (g, emb) in enumerate(test):
            self.assertEqual(len(emb), size)
            self.assertEqual(max_chainlength(emb), cl)
            dwe.verify_embedding(emb, s, g)

    def embed_battery(self, g, reconstruct):
        bgcg = busclique.busgraph_cache(g)
        emb0 = bgcg.largest_clique()
        size = len(emb0)
        cl = max_chainlength(emb0)
        N = range(size)
        yield size, cl
        yield g, emb0
        yield g, bgcg.find_clique_embedding(size)
        yield g, busclique.find_clique_embedding(size, g)
        yield g, busclique.find_clique_embedding(size, g, use_cache = False)
        yield g, bgcg.largest_clique_by_chainlength(cl)

        nodes = set(itertools.chain.from_iterable(emb0.values()))
        h = reconstruct(nodes)
        bgch = busclique.busgraph_cache(h)
        yield h, busclique.find_clique_embedding(N, h)
        yield h, busclique.find_clique_embedding(N, h, use_cache = False)
        yield h, bgch.largest_clique()
        yield h, bgch.find_clique_embedding(N)
        yield h, bgch.largest_clique_by_chainlength(cl)

        rootdir = busclique.busgraph_cache.cache_rootdir()
        self.assertTrue(os.path.exists(rootdir))
        busclique.busgraph_cache.clear_all_caches()
        self.assertFalse(os.path.exists(rootdir))


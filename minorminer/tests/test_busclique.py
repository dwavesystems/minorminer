from minorminer import busclique
from minorminer.utils import verify_embedding, chimera, pegasus
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
        self.c248 = dnx.chimera_graph(2, n = 4, t = 8)
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
        self.p4, self.p4_nd, self.p4_d = list(zip(p4, p4c, p4n))

    def test_p16(self):
        def reconstruct(nodes):
            return dnx.pegasus_graph(16, node_list = nodes)
        self.run_battery('p16', self.p16, reconstruct, 180, 17, 172, 15)

    def test_c16(self):
        def reconstruct(nodes):
            return dnx.chimera_graph(16, node_list = nodes)
        self.run_battery('c16', self.c16, reconstruct, 64, 17, 64, 16)

    def run_battery(self, name, g, reconstruct,
                          cliquesize, cliquelength,
                          bicliquesize, bicliquelength,
                          test_python = False, test_nocache = True):
        labels = g.graph['labels']

        test = self.clique_battery(g, reconstruct,
                                   test_python = test_python,
                                   test_nocache = test_nocache)
        size, cl = next(test)
        self.assertEqual(size, cliquesize)
        self.assertEqual(cl, cliquelength)
        s = nx.complete_graph(size)
        for i, (h, emb, kind, check_cl) in enumerate(test):
            print(name, labels, kind)
            if check_cl:
                self.assertEqual(max_chainlength(emb), cl)
            verify_embedding(emb, s, h)

        test = self.biclique_battery(g, reconstruct)
        size, cl = next(test)
        self.assertEqual(size, bicliquesize)
        if bicliquelength is not None:
            self.assertEqual(cl, bicliquelength)
        s = nx.complete_bipartite_graph(size, size)
        for i, (h, emb, kind) in enumerate(test):
            print(name, labels, kind)
            if bicliquelength is not None:
                self.assertEqual(max_chainlength(emb), bicliquelength)
            verify_embedding(emb, s, h)


    def clique_battery(self, g, reconstruct,
                       test_python = False, test_nocache = True):
        bgcg = busclique.busgraph_cache(g)
        emb0 = bgcg.largest_clique()
        size = len(emb0)
        cl = max_chainlength(emb0)
        N = range(size)
        yield size, cl
        yield g, emb0, 'g:bcg.lc', True
        yield g, bgcg.find_clique_embedding(size), 'g:bcg.fce', True
        yield g, busclique.find_clique_embedding(size, g), 'g:bc.fce', True
        if test_nocache:
            yield (g,
                   busclique.find_clique_embedding(size, g, use_cache = False),
                   'g:bc.fce,nc', True)
        yield g, bgcg.largest_clique_by_chainlength(cl), 'g:bc.lcbc', True
        if test_python:
            if g.graph['family'] == 'chimera':
                if g.graph['labels'] == 'int':
                    # this fails on coordinate-labeled graphs... TODO?
                    args = size, g.graph['rows']
                    kwargs = dict(target_edges = g.edges)
                    yield (g,
                           chimera.find_clique_embedding(*args, **kwargs),
                           'g:legacy.fce', True)
            if g.graph['family'] == 'pegasus':
                kwargs = dict(target_graph = g)
                yield (g, pegasus.find_clique_embedding(size, **kwargs),
                       'g:legacy.fce', False)

        nodes = set(itertools.chain.from_iterable(emb0.values()))
        h = reconstruct(nodes)
        bgch = busclique.busgraph_cache(h)
        yield h, busclique.find_clique_embedding(N, h), 'h:bc.fce', True
        if test_nocache:
            yield (h, busclique.find_clique_embedding(N, h, use_cache = False), 
                   'h:bc.fce,nc', True)
        yield h, bgch.largest_clique(), 'h:bgc.lc', True
        yield h, bgch.find_clique_embedding(N), 'h:bgc.fce', True
        yield h, bgch.largest_clique_by_chainlength(cl), 'h:bgc.lcbc', True
        if test_python:
            if g.graph['family'] == 'chimera':
                if g.graph['labels'] == 'int':
                    # this fails on coordinate-labeled graphs... TODO?
                    args = size, h.graph['rows']
                    kwargs = dict(target_edges = h.edges)
                    yield (h,
                           chimera.find_clique_embedding(*args, **kwargs),
                           'h:legacy.fce', True)
            if g.graph['family'] == 'pegasus':
                kwargs = dict(target_graph = h)
                yield (h, pegasus.find_clique_embedding(size, **kwargs),
                       'h:legacy.fce', False)
            

    def biclique_battery(self, g, reconstruct):
        bgcg = busclique.busgraph_cache(g)
        emb0 = bgcg.largest_balanced_biclique()
        size = len(emb0)//2
        cl = max_chainlength(emb0)
        N = range(size)
        yield size, cl
        yield g, emb0, 'bgc.lbb'
        yield (g, bgcg.find_biclique_embedding(N, range(size, 2*size)), 
                  'bgc.fbe,list')
        yield g, bgcg.find_biclique_embedding(size, size), 'bgc.fbe,ints'

    @classmethod
    def tearDownClass(cls):
        rootdir = busclique.busgraph_cache.cache_rootdir()
        if not os.path.exists(rootdir):
            raise RuntimeError("cache rootdir not found before cleanup")
        busclique.busgraph_cache.clear_all_caches()
        if os.path.exists(rootdir):
            raise RuntimeError("cache rootdir exists after cleanup")

    def test_chimera_weird_sizes(self):
        self.assertRaises(NotImplementedError,
                          busclique.busgraph_cache,
                          self.c42A)

        self.assertRaises(NotImplementedError,
                          busclique.find_clique_embedding,
                          999, self.c42A)

        self.assertRaises(NotImplementedError,
                          busclique.find_clique_embedding,
                          999, self.c42A, use_cache = False)

        def reconstructor(m, n, t):
            return lambda nodes: dnx.chimera_graph(m, n = n, t = t,
                                                   node_list = nodes)
        for g, params in (self.c428, (4, 2, 8)), (self.c248, (2, 4, 8)):
            reconstruct = reconstructor(*params)
            self.run_battery('c%d%d%d'%params, g, reconstruct, 16, 3, 16, None)

    def test_labelings(self):
        def reconstructor(g):
            return lambda nodes: g.subgraph(nodes).copy()

        names = 'c4_nd', 'c4', 'c4_d', 'p4_nd', 'p4', 'p4_d'
        nocache = False, True, True, False, True, True
        topos = self.c4_nd, self.c4, self.c4_d, self.p4_nd, self.p4, self.p4_d

        for (name, test_nocache, G) in zip(names, nocache, topos):
            g0 = G[0]
            bgc = busclique.busgraph_cache(g0)
            K = bgc.largest_clique()
            B = bgc.largest_balanced_biclique()
            for g in G:
                self.run_battery(name, g, reconstructor(g), 
                                 len(K), max_chainlength(K),
                                 len(B)//2, None,
                                 test_python = test_nocache,
                                 test_nocache = test_nocache)

    def test_k4_bug(self):
        edges = [30, 2940], [30, 2955], [45, 2940], [45, 2955], [2940, 2955]
        p = dnx.pegasus_graph(16, edge_list = edges)
        k4 = busclique.find_clique_embedding(4, p)

        self.assertEquals(k4, {})

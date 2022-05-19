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
    if g.graph['family'] == 'pegasus' or g.graph['family'] == 'zephyr':
        def icpl(p, q):
            return p[0] != q[0]
    elif g.graph['family'] == 'chimera':
        def icpl(p, q):
            return p[2] != q[2]
    else:
        raise ValueError("g is not a pegasus, zephyr or chimera")

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
    elif g.graph['family'] == 'zephyr':
        f = dnx.zephyr_graph
    else:
        raise ValueError("g is not a pegasus, zephyr or chimera")

    return f(*args, edge_list = edges, node_list = nodes, **kwargs)

def max_chainlength(emb):
    if emb:
        return max(len(c) for c in emb.values())

class TestBusclique(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestBusclique, self).__init__(*args, **kwargs)

        self.c16 = dnx.chimera_graph(16)
        self.p16 = dnx.pegasus_graph(16)
        self.z6 = dnx.zephyr_graph(6)
        self.c16c = dnx.chimera_graph(16, coordinates=True, data = False)
        self.c428 = dnx.chimera_graph(4, n = 2, t = 8)
        self.c248 = dnx.chimera_graph(2, n = 4, t = 8)
        self.c42A = dnx.chimera_graph(4, n = 2, t = 9)
        c4c_0 = subgraph_node_yield(dnx.chimera_graph(4, coordinates = True), .95)
        p4c_0 = subgraph_node_yield(dnx.pegasus_graph(4, coordinates = True), .95)
        z4c_0 = subgraph_node_yield(dnx.zephyr_graph(4, coordinates = True), .95)
        c4c = [c4c_0,
               subgraph_edge_yield(c4c_0, .95),
               subgraph_edge_yield_few_bad(c4c_0, .95, 6)]
        p4c = [p4c_0,
               subgraph_edge_yield(p4c_0, .95),
               subgraph_edge_yield_few_bad(p4c_0, .95, 6)]
        z4c = [z4c_0,
               subgraph_edge_yield(z4c_0, .95),
               subgraph_edge_yield_few_bad(z4c_0, .95, 6)]

        p4coords = dnx.pegasus_coordinates(4)
        c4coords = dnx.chimera_coordinates(4, 4, 4)
        z4coords = dnx.zephyr_coordinates(4)

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

        z4 = [relabel(z,
                      z4coords.iter_zephyr_to_linear,
                      z4coords.iter_zephyr_to_linear_pairs,
                      4) for z in z4c]


        self.c4, self.c4_nd, self.c4_d = list(zip(c4, c4c))
        self.p4, self.p4_nd, self.p4_d = list(zip(p4, p4c, p4n))
        self.z4, self.z4_nd, self.z4_d = list(zip(z4, z4c))

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

        clique_init_ok = False
        test = self.clique_battery(g, reconstruct,
                                   test_python = test_python,
                                   test_nocache = test_nocache)
        with self.subTest(msg=f"clique_battery init"):
            emb0, size, cl = next(test)
            s = nx.complete_graph(size)
            verify_embedding(emb0, s, g)
            self.assertEqual(size, cliquesize)
            self.assertEqual(cl, cliquelength)
            clique_init_ok = True
        if clique_init_ok:
            for i, (h, emb_f, kind, check_cl) in enumerate(test):
                with self.subTest(msg=f"clique_battery({name}, {labels}, {kind})"):
                    emb = emb_f()
                    verify_embedding(emb, s, h)
                    if check_cl:
                        self.assertEqual(max_chainlength(emb), cl)

        biclique_init_ok = False
        test = self.biclique_battery(g, reconstruct)
        with self.subTest(msg=f"biclique_battery init"):
            emb0, size, cl = next(test)
            s = nx.complete_bipartite_graph(size, size)
            verify_embedding(emb0, s, g)
            if bicliquelength is not None:
                self.assertEqual(cl, bicliquelength)
            biclique_init_ok = True
        if biclique_init_ok:
            for i, (h, emb_f, kind) in enumerate(test):
                with self.subTest(msg=f"biclique_battery({name}, {labels}, {kind})"):
                    emb = emb_f()
                    verify_embedding(emb, s, h)
                    if bicliquelength is not None:
                        self.assertEqual(max_chainlength(emb), bicliquelength)

    def clique_battery(self, g, reconstruct,
                       test_python = False, test_nocache = True):
        bgcg = busclique.busgraph_cache(g)
        emb0 = bgcg.largest_clique()
        size = len(emb0)
        cl = max_chainlength(emb0)
        N = range(size)
        yield emb0, size, cl
        yield g, lambda:emb0, 'g:bcg.lc', True
        yield g, lambda:bgcg.find_clique_embedding(size), 'g:bcg.fce', True
        yield g, lambda:busclique.find_clique_embedding(size, g), 'g:bc.fce', True
        if test_nocache:
            yield (g,
                   lambda:busclique.find_clique_embedding(size, g, use_cache = False),
                   'g:bc.fce,nc', True)
        yield g, lambda:bgcg.largest_clique_by_chainlength(cl), 'g:bc.lcbc', True
        if test_python:
            if g.graph['family'] == 'chimera':
                if g.graph['labels'] == 'int':
                    # this fails on coordinate-labeled graphs... TODO?
                    args = size, g.graph['rows']
                    kwargs = dict(target_edges = g.edges)
                    yield (g,
                           lambda:chimera.find_clique_embedding(*args, **kwargs),
                           'g:legacy.fce', True)
            if g.graph['family'] == 'pegasus':
                kwargs = dict(target_graph = g)
                yield (g, lambda:pegasus.find_clique_embedding(size, **kwargs),
                       'g:legacy.fce', False)

        nodes = set(itertools.chain.from_iterable(emb0.values()))
        h = reconstruct(nodes)
        bgch = busclique.busgraph_cache(h)
        yield h, lambda:busclique.find_clique_embedding(N, h), 'h:bc.fce', True
        if test_nocache:
            yield (h, lambda:busclique.find_clique_embedding(N, h, use_cache = False),
                   'h:bc.fce,nc', True)
        yield h, lambda:bgch.largest_clique(), 'h:bgc.lc', True
        yield h, lambda:bgch.find_clique_embedding(N), 'h:bgc.fce', True
        yield h, lambda:bgch.largest_clique_by_chainlength(cl), 'h:bgc.lcbc', True
        if test_python:
            if g.graph['family'] == 'chimera':
                if g.graph['labels'] == 'int':
                    # this fails on coordinate-labeled graphs... TODO?
                    args = size, h.graph['rows']
                    kwargs = dict(target_edges = h.edges)
                    yield (h,
                           lambda:chimera.find_clique_embedding(*args, **kwargs),
                           'h:legacy.fce', True)
            if g.graph['family'] == 'pegasus':
                kwargs = dict(target_graph = h)
                yield (h, lambda:pegasus.find_clique_embedding(size, **kwargs),
                       'h:legacy.fce', False)

    def biclique_battery(self, g, reconstruct):
        bgcg = busclique.busgraph_cache(g)
        emb0 = bgcg.largest_balanced_biclique()
        size = len(emb0)//2
        cl = max_chainlength(emb0)
        N = range(size)
        yield emb0, size, cl
        yield g, lambda:emb0, 'bgc.lbb'
        yield (g, lambda:bgcg.find_biclique_embedding(N, range(size, 2*size)),
                  'bgc.fbe,list')
        yield g, lambda:bgcg.find_biclique_embedding(size, size), 'bgc.fbe,ints'

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

        names = 'c4_nd', 'c4', 'c4_d', 'p4_nd', 'p4', 'p4_d', 'z4_nd', 'z4', 'z4_d'
        nocache = False, True, True, False, True, True, False, True, True
        python = False, True, True, False, True, True, False, False, False
        topos = self.c4_nd, self.c4, self.c4_d, self.p4_nd, self.p4, self.p4_d, self.z4_nd, self.z4, self.z4_d

        for (name, test_nocache, test_python, G) in zip(names, nocache, python, topos):
            g0 = G[0]
            bgc = busclique.busgraph_cache(g0)
            K = bgc.largest_clique()
            skip_K = True
            with self.subTest(msg = f"test_labelings setup for {name} cliques"):
                verify_embedding(K, nx.complete_graph(len(K)), g0)
                skip_K = False

            B = bgc.largest_balanced_biclique()
            skip_B = True
            with self.subTest(msg = f"test_labelings setup for {name} bicliques"):
                b2 = len(B)//2
                verify_embedding(B, nx.complete_bipartite_graph(b2, b2), g0)
                skip_B = False
            if skip_K or skip_B:
                continue
            for g in G:
                self.run_battery(name, g, reconstructor(g), 
                                 len(K), max_chainlength(K),
                                 len(B)//2, None,
                                 test_python = test_python,
                                 test_nocache = test_nocache)

    def test_k4_bug(self):
        edges = [30, 2940], [30, 2955], [45, 2940], [45, 2955], [2940, 2955]
        p = dnx.pegasus_graph(16, edge_list = edges)
        k4 = busclique.find_clique_embedding(4, p)

        self.assertEqual(k4, {})
        
    def test_seed_determinism(self):
        for g in (self.c4_nd[0], self.z4_nd[0], self.p4_nd[0]):
            seed = random.randint(0, 2**32-1)
            with self.subTest(msg=f"seed determinism test: {g.graph['family']}"):
                bgc = busclique.busgraph_cache(g, seed = seed)
                Klarge = bgc.largest_clique()
                nK = len(Klarge)
                Blarge = bgc.largest_balanced_biclique()
                nB = len(Blarge)//2
                Kfamily = [bgc.find_clique_embedding(i) for i in range(nK+1)]
                Bfamily = [bgc.find_biclique_embedding(i,i) for i in range(nB+1)]
                # now, clear the cache and go again
                busclique.busgraph_cache.clear_all_caches()
                bgc = busclique.busgraph_cache(g, seed = seed)
                self.assertEqual(Klarge, bgc.largest_clique())
                self.assertEqual(Blarge, bgc.largest_balanced_biclique())
                self.assertEqual(Kfamily, [bgc.find_clique_embedding(i) for i in range(nK+1)])
                self.assertEqual(Bfamily, [bgc.find_biclique_embedding(i,i) for i in range(nB+1)])

    def test_perfect_z6_clique(self):
        k88 = nx.complete_graph(88)
        bgc = busclique.busgraph_cache(self.z6)
        e88_cache = bgc.largest_clique()
        verify_embedding(e88_cache, k88, self.z6)

        e88_cache2 = busclique.find_clique_embedding(88, self.z6, use_cache=True)
        verify_embedding(e88_cache2, k88, self.z6)

        e88 = busclique.find_clique_embedding(88, self.z6, use_cache=False)
        verify_embedding(e88, k88, self.z6)
        
    def test_perfect_z6_biclique(self):
        k88_88 = nx.complete_bipartite_graph(88, 88)
        bgc = busclique.busgraph_cache(self.z6)
        e88_88_cache = bgc.largest_balanced_biclique()
        verify_embedding(e88_88_cache, k88_88, self.z6)

    def test_z3_determinism(self):
        #first, check an explicit sample found with find_nondeterminism
        z3 = dnx.zephyr_graph(3)
        z3.remove_nodes_from([11, 43, 61, 77, 102, 107, 143, 145, 162, 174, 178, 185, 240, 333])
        z3.remove_edges_from([
            (28, 29), (28, 264), (37, 40), (55, 259), (73, 76), (96, 99), (97, 98), (98, 302),
            (108, 111), (110, 298), (117, 118), (117, 218), (128, 131), (133, 135), (140, 142),
            (158, 161), (160, 281), (169, 171), (187, 190), (207, 208), (212, 215), (235, 236),
            (236, 239), (242, 245), (265, 266), (316, 317), (319, 321),
        ])
        emb0 = busclique.busgraph_cache(z3).largest_clique()
        emb1 = busclique.find_clique_embedding(len(emb0), z3, use_cache=False)
        self.assertEqual(sorted(emb0.values()), sorted(emb1.values()))

        #then, do a quick scan for another example -- this will present as a flaky test if
        #we merely got lucky hammering down the tet above
        self.assertEqual(None, find_nondeterminism('zephyr', 3, 10))

    def test_p4_determinism(self):
        #first, check an explicit sample found with find_nondeterminism
        p4 = dnx.pegasus_graph(4)
        p4.remove_nodes_from([
            38, 56, 69, 81, 92, 97, 107, 112, 113, 207, 209, 212, 240, 258, 264, 277, 281
        ])
        p4.remove_edges_from([
            (12, 13), (14, 17), (27, 28), (28, 219), (29, 234), (32, 255), (37, 189), (44, 225),
            (52, 53), (54, 55), (61, 62), (66, 67), (74, 77), (85, 86), (114, 115), (123, 176),
            (152, 155), (154, 155), (156, 159), (193, 196), (198, 199), (204, 205), (241, 242),
            (249, 250), (266, 269), (271, 272),
        ])
        emb0 = busclique.busgraph_cache(p4).largest_clique()
        emb1 = busclique.find_clique_embedding(len(emb0), p4, use_cache=False)
        self.assertEqual(sorted(emb0.values()), sorted(emb1.values()))

        #then, do a quick scan for another example -- this will present as a flaky test if
        #we merely got lucky hammering down the tet above
        self.assertEqual(None, find_nondeterminism('pegasus', 4, 10))

    def test_c4_determinism(self):
        #first, check an explicit sample found with find_nondeterminism
        c4 = dnx.chimera_graph(4)
        c4.remove_nodes_from([1, 47, 60, 62, 59, 107])
        c4.remove_edges_from([
            (0, 32), (24, 56), (26, 58), (32, 64), (37, 35), (39, 34), (44, 42), (53, 51),
            (64, 69), (87, 95), (96, 103),
        ])
        emb0 = busclique.busgraph_cache(c4).largest_clique()
        emb1 = busclique.find_clique_embedding(len(emb0), c4, use_cache=False)
        self.assertEqual(sorted(emb0.values()), sorted(emb1.values()))

        #then, do a quick scan for another example -- this will present as a flaky test if
        #we merely got lucky hammering down the tet above
        self.assertEqual(None, find_nondeterminism('chimera', 4, 10))

def find_nondeterminism(family, size=4, tries=10):
    if family == 'pegasus':
        generator = dnx.pegasus_graph
        coordinates = dnx.pegasus_coordinates
    elif family == 'zephyr':
        generator = dnx.zephyr_graph
        coordinates = dnx.zephyr_coordinates
    elif family == 'chimera':
        generator = dnx.chimera_graph
        coordinates = dnx.chimera_coordinates

    for i in range(tries):
        H = generator(size, coordinates=True)
        G = subgraph_edge_yield_few_bad(subgraph_node_yield(H, .95), .95, 6)
        emb0 = busclique.busgraph_cache(G, seed = random.randint(0, 2**32-1)).largest_clique()
        emb1 = busclique.find_clique_embedding(len(emb0), G, use_cache=False, seed=random.randint(0, 2**32-1))
        if sorted(emb0.values()) != sorted(emb1.values()):
            coords = coordinates(size)
            bad_nodes = [v for v in H if v not in G]
            node_f = getattr(coords, f"iter_{family}_to_linear");
            edge_f = getattr(coords, f"iter_{family}_to_linear_pairs");
            return (
                list(node_f(bad_nodes)),
                list(edge_f(e for e in H.edges if not G.has_edge(*e) and all(v in G for v in e)))
            )



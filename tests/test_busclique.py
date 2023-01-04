from minorminer import busclique
from minorminer.utils import verify_embedding
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

        busclique.busgraph_cache.clear_all_caches()

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

    def run_battery(self, name, g, reconstruct, cliquesize, cliquelength,
                          bicliquesize, bicliquelength, test_nocache = True):
        labels = g.graph['labels']

        clique_init_ok = False
        test = self.clique_battery(g, reconstruct, test_nocache = test_nocache)
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

    def clique_battery(self, g, reconstruct, test_nocache = True):
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
        topos = self.c4_nd, self.c4, self.c4_d, self.p4_nd, self.p4, self.p4_d, self.z4_nd, self.z4, self.z4_d

        for (name, test_nocache, G) in zip(names, nocache, topos):
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

    def test_cache_export_import(self):
        # make sure we're nondeterministic so accidentally re-computing the 
        # cache will be detected
        p4 = self.p4_nd[0]
        bgc = busclique.busgraph_cache(p4, seed=100)
        cliques = [bgc.find_clique_embedding(n) for n in range(len(bgc.largest_clique()))]
        for sparse in (True, False):
            for compress in (True, False):
                with self.subTest(msg=f"sparse: {sparse}, compress: {compress}"):
                    tmp = busclique.busgraph_cache(p4, seed=101)
                    tmp.update_clique_cache([], write_to_disk=True)
                    self.assertEqual(tmp.largest_clique(), {})

                    cache = bgc.get_clique_cache(sparse=sparse, compress=compress)
                    tmp.update_clique_cache(cache, write_to_disk=False)
                    #test loading from memory
                    for n, c in enumerate(cliques):
                        self.assertEqual(set(c.values()), set(tmp.find_clique_embedding(n).values()))

                    tmp.update_clique_cache(cache, write_to_disk=True)

                    tmp = busclique.busgraph_cache(p4, seed=101)
                    #test loading from disk
                    for n, c in enumerate(cliques):
                        self.assertEqual(set(c.values()), set(tmp.find_clique_embedding(n).values()))        

    def test_cache_merge(self):
        p4 = self.p4_nd[0] 
        bgc0 = busclique.busgraph_cache(p4, seed=200)
        bgc1 = busclique.busgraph_cache(p4, seed=201)
        bgc = busclique.busgraph_cache(p4, seed=202)
        bgc.update_clique_cache(bgc0.get_clique_cache(), write_to_disk=False)
        bgc.merge_clique_cache(bgc1, write_to_disk=False)
        for n in range(1, len(bgc.largest_clique())):
            emb = bgc.find_clique_embedding(n)
            emb0 = bgc0.find_clique_embedding(n)
            emb1 = bgc1.find_clique_embedding(n)
            maxlen0 = max(map(len, emb0.values())) if emb0 else 999
            maxlen1 = max(map(len, emb1.values())) if emb1 else 999
            maxlen = max(map(len, emb.values()))
            self.assertEqual(maxlen, min(maxlen0, maxlen1))

        maxlen = max(map(len, bgc.largest_clique().values()))
        for n in range(maxlen):
            emb = bgc.largest_clique_by_chainlength(n)
            emb0 = bgc0.largest_clique_by_chainlength(n)
            emb1 = bgc1.largest_clique_by_chainlength(n)
            self.assertEqual(len(emb), max(len(emb0), len(emb1)))

    def test_cache_insert(self):
        p4 = self.p4_nd[0]
        bgc = busclique.busgraph_cache(p4, seed=300)
        emb = bgc.largest_clique()
        chain = max(emb.values(), key=len)
        
        bgc.insert_clique_embedding([chain], direct=False) # should not overwrite
        emb0 = bgc.find_clique_embedding(1) #should be chainlength 1
        self.assertEqual(len(emb0[0]), 1)
        
        bgc.insert_clique_embedding([chain], direct=True) # should overwrite
        emb1 = bgc.find_clique_embedding(1) #should be the newly-inserted embedding
        self.assertEqual(len(emb1[0]), len(chain))


    def test_mine_clique_embeddings_smoketest(self):
        busclique.mine_clique_embeddings(
            self.p4_nd[0],
            num_seeds=1,
            heuristic_bound=8,
            heuristic_args=dict(tries=1, max_no_improvement=1, chainlength_patience=0, verbose=2),
            heuristic_runs=1,
        )

    def test_regularize_embedding_bug_215(self):
        emb = {0: [206], 1: [127, 130, 134, 242, 133], 2: [129], 3: [126], 4: [209], 5: [179, 135, 136]}
        missing_nodes = (17, 32, 132, 155, 212, 215, 230, 263, 280)
        missing_edges = [
            (8, 11), (9, 162), (13, 231), (15, 198), (22, 216), (25, 222),
            (26, 264), (27, 195), (31, 222), (31, 225), (34, 219), (34, 198),
            (35, 249), (36, 174), (37, 204), (40, 201), (40, 210), (43, 186),
            (45, 163), (46, 216), (48, 189), (49, 243), (54, 181), (54, 174),
            (56, 256), (57, 177), (58, 232), (59, 253), (59, 262), (61, 223),
            (61, 204), (64, 226), (65, 262), (66, 184), (67, 220), (67, 214),
            (70, 217), (70, 220), (71, 253), (72, 163), (77, 229), (80, 253),
            (81, 181), (82, 83), (82, 214), (84, 175), (86, 250), (88, 220),
            (90, 200), (92, 275), (93, 178), (95, 260), (95, 272), (97, 200),
            (97, 205), (98, 236), (99, 169), (100, 224), (103, 203), (103, 209),
            (105, 185), (105, 197), (108, 182), (109, 218), (112, 200),
            (113, 236), (114, 117), (117, 185), (120, 191), (130, 245),
            (131, 278), (131, 281), (134, 251), (164, 167), (172, 173),
            (180, 183), (187, 188), (194, 197), (198, 199), (216, 217),
            (217, 218), (222, 225), (237, 238), (238, 239), (270, 273)
        ]
        p4 = dnx.pegasus_graph(4)
        p4.remove_nodes_from(missing_nodes)
        p4.remove_edges_from(missing_edges)
        busclique._regularize_embedding(p4, emb)                  

    def test_topology_identifier(self):
        perfect_id = '38cad89632b234831d58675091f1bc581c96de65d4b2a0c06c0d94a7f97e21a7'
        p16 = dnx.pegasus_graph(16, coordinates=True)
        bgc = busclique.busgraph_cache(p16)
        self.assertEqual(
            bgc.topology_identifier(),
            perfect_id,
            f'Topology identifier does not match expectation.  If busclique.__cache_version changed, this test needs to be updated.'
        )

        # see minorminer issue #227 on github -- the busclique algorithm does
        # not depend on odd edges and does not include them in its serialization
        odd_edges = []
        relevant_edges = []
        for p, q in p16.edges:
            if p[0] == q[0] and p[-1] == q[-1]:
                odd_edges.append((p, q))
            else:
                relevant_edges.append((p, q))

        # it's actually possible that we will include those edges in the future
        # though -- there are some optimal clique embeddings which utilize odd
        # edges -- let's put in an explicit test to remind ourselves to update
        # this test under that eventuality
        e = random.choice(odd_edges)
        p16.remove_edge(*e)
        bgc_o = busclique.busgraph_cache(p16)
        self.assertEqual(
            bgc_o.topology_identifier(),
            perfect_id,
            f'topology identifier changed after deleting odd edge {e}'
        )

        e = random.choice(relevant_edges)
        p16.remove_edge(*e)
        bgc_e = busclique.busgraph_cache(p16)
        self.assertNotEqual(
            bgc_e.topology_identifier(),
            perfect_id,
            f'topology identifier did not change after removing non-odd edge {e}'
        )

        p16 = dnx.pegasus_graph(16)
        n = random.choice(list(p16.nodes))
        p16.remove_node(n)
        bgc_n = busclique.busgraph_cache(p16)
        self.assertNotEqual(
            bgc_n.topology_identifier(),
            perfect_id,
            f'topology identifier did not change after removing node {n}'
        )


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



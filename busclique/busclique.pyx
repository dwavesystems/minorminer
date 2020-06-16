# distutils: language = c++
# cython: language_level=3
include "busclique_h.pxi"

import homebase, os, pathlib, fasteners, threading
from pickle import dump, load
import networkx as nx
cdef int __cache_version__ = 1
cdef int __lru_size__ = 100

def chimera_clique(g, size, use_cache = True):
    cdef size_t tile = g.graph['tile']
    if tile > 8:
        raise NotImplementedError(("this clique embedder only supports chimera "
                                   "graphs with a tile size of 8 or less"))
    if use_cache:
        G = busgraph_cache(g)
        return G.get_clique(size)

    cdef chimera_spec *chim = new chimera_spec(g.graph['rows'], g.graph['columns'], tile)
    cdef nodes_t nodes = g.nodes()
    cdef edges_t edges = g.edges()
    cdef embedding_t emb
    find_clique(chim[0], nodes, edges, size, emb)
    try:
        return emb
    finally:
        del chim

def pegasus_clique(g, size, use_cache = True):
    if use_cache:
        G = busgraph_cache(g)
        return G.get_clique(size)

    cdef pegasus_spec *peg = new pegasus_spec(g.graph['rows'],
                                   [o//2 for o in g.graph['vertical_offsets'][::2]],
                                   [o//2 for o in g.graph['horizontal_offsets'][::2]])
    cdef nodes_t nodes = g.nodes()
    cdef edges_t edges = g.edges()
    cdef embedding_t emb
    find_clique(peg[0], nodes, edges, size, emb)
    try:
        return emb
    finally:
        del peg

cdef dict __global_locks__ = {'clique': threading.Lock(), 'biclique': threading.Lock()}
class busgraph_cache:
    def __init__(self, g):
        self._family = g.graph['family']
        if(self._family == 'chimera'):
            self._graph = chimera_busgraph(g)
        elif(self._family == 'pegasus'):
            self._graph = pegasus_busgraph(g)
        else:
            raise ValueError(("input graph must either be a "
                              "dwave_networkx.pegasus_graph or a "
                              "dwave_networkx.chimera_graph"))
        self._cliques = None
        self._bicliques = None

    def _ensure_clique_cache(self):
        if self._cliques is None:
            self._cliques = self._fetch_cache('clique', self._graph.cliques)


    def _ensure_biclique_cache(self):
        if self._bicliques is None:
            self._bicliques = self._fetch_cache('biclique', self._graph.bicliques)

    def _fetch_cache(self, dirname, compute):
        rootdir = homebase.user_data_dir('busclique', 'dwave', __cache_version__)
        basedir = os.path.join(rootdir, dirname)
        pathlib.Path(basedir).mkdir(parents=True, exist_ok=True)
        lockfile = os.path.join(basedir, ".lock")
        lrufile = os.path.join(basedir, ".lru")
        identifier = self._graph.identifier
        shortcode = str(hash(identifier))
        #this solves thread-safety?
        with __global_locks__[dirname]:
            #this solves inter-process safety?
            with fasteners.InterProcessLock(lockfile):
                cachefile = os.path.join(basedir, str(shortcode))
                if os.path.exists(cachefile):
                    with open(cachefile, 'rb') as filecache:
                        currcache = load(filecache)
                else:
                    currcache = {}
                cache = currcache.get(identifier)
                if cache is None:
                    cache = compute()
                    currcache[identifier] = cache
                    with open(cachefile, 'wb') as filecache:
                        dump(currcache, filecache)

                #now, update the LRU cache -- if this was being done in-memory,
                #there are more efficient algorithms... but since we're doing 
                #linear-time read&write, might as well do it the lazy way.
                try:
                    with open(lrufile, 'rb') as lru:
                        LRU = load(lru)
                except FileNotFoundError:
                    LRU = []
                newLRU = [shortcode]
                for x in LRU:
                    if x != shortcode:
                        newLRU.append(x)
                while len(newLRU) > __lru_size__:
                    oldkey = newLRU.pop()
                    os.remove(os.path.join(basedir, oldkey))
                with open(lrufile, 'wb') as lru:
                    dump(newLRU, lru)
        return cache

    def largest_clique(self):
        self._ensure_clique_cache()
        embs = self._cliques['raw']
        keys = self._cliques['size']
        return dict(enumerate(embs[keys[max(keys)]]))

    def largest_clique_by_chainlength(self, chainlength = None):
        self._ensure_clique_cache()
        embs = self._cliques['raw']

        if 0 <= chainlength < len(embs):
            return dict(enumerate(embs[chainlength]))
        else:
            raise ValueError(("no clique of that chainlength found; "
                              "maximum chainlength is {}").format(len(embs)-1))

    def find_clique_embedding(self, nn):
        try:
            nodes = list(range(nn))
            num = nn
        except TypeError:
            nodes = tuple(nn)
            num = len(nodes)
        self._ensure_clique_cache()
        keys = self._cliques['size']
        key = keys.get(num)
        if key is None:
            raise ValueError(("no clique of that size found; "
                              "maximum size is {}").format(max(keys.values())))
        emb = self._cliques['raw'][key]
        return dict(zip(nodes, emb))

    def largest_balanced_biclique(self):
        self._ensure_biclique_cache()
        biggest = self._bicliques['max_side']
        embs = self._bicliques['raw']
        s0, s1 = key = max(embs, key=lambda x: min(x))
        raw_emb = embs[key]
        raw_emb0 = sorted(raw_emb[:s0], key=len)
        raw_emb1 = sorted(raw_emb[s0:s0+s1], key=len)
        s_min = min(s0, s1)
        emb0 = raw_emb0[:s_min]
        emb1 = raw_emb1[:s_min]
        return dict(enumerate(emb0 + emb1))

    def find_biclique_embedding(self, nn, mm):
        self._ensure_biclique_cache()
        biggest = self._bicliques['max_side']
        by_size = self._bicliques['size']
        raw = self._bicliques['raw']
        try:
            N = list(range(nn))
            M = list(range(nn, nn + mm))
            n = nn
            m = mm
        except TypeError:
            N = tuple(nn)
            M = tuple(mm)
            n = len(N)
            m = len(M)
        if(m < n):
            n, N, m, M = m, M, n, N
        if m == 0:
            emb = self._graph.independent_set(n)
            if emb is None:
                raise ValueError("no biclique of that size found")
            return dict(zip(N, emb))
        nmax = biggest[None]
        for i in range(n, nmax + 1):
            mmax = biggest.get(i, 0)
            if mmax < m:
                continue
            row = by_size.get(i, {0:None})
            for j in range(m, mmax + 1):
                key = row.get(j)
                if key is None:
                    continue
                emb = raw[key]
                s0, s1 = key
                emb0 = emb[:s0]
                emb1 = emb[s0:]
                if s0 <= n and s1 <= m:
                    return dict(zip(N + M, emb0[:n] + emb1))
                else:
                    return dict(zip(N + M, emb1[:n] + emb0))
        raise ValueError("no biclique of that size found")

cdef dict _make_clique_cache(vector[embedding_t] &embs):
    cdef embedding_t emb
    cdef size_t maxsize = 0
    cdef size_t maxlength = 0
    cdef size_t i, j
    cdef dict by_size = {}
    cdef list raw = []
    for length, emb in enumerate(embs):
        raw.append(tuple(map(tuple, emb)))
        maxsize = max(emb.size(), maxsize)
        for i in range(maxsize, -1, -1):
            if by_size.setdefault(i, length) != length:
                break
    return {'raw': raw, 'size': by_size}

cdef _keep_biclique_key(tuple key0, tuple key1, dict chainlength):
    if key0 == key1:
        return True
    long0, short0 = chainlength[key0]
    long1, short1 = chainlength[key1]
    if long0 < long1 or (long0 == long1 and short0 <= short1):
        return True
    else:
        return False

cdef dict _make_biclique_cache(vector[pair[pair[size_t, size_t], embedding_t]] &embs):
    cdef embedding_t emb
    cdef dict raw = {(s0, s1): emb for (s0, s1), emb in embs if emb.size()}
    cdef dict chainlength = {key: (max(map(len, emb)), min(map(len, emb)))
                             for key, emb in embs if emb.size()}

    cdef dict by_size = {}
    cdef dict by_length = {}
    for key, val in raw.items():
        s0, s1 = key
        emb0 = tuple(val[:s0])
        emb1 = tuple(val[s0:])
        if(s0 < s1):
            s1, s0 = s0, s1
        realkey = by_size.setdefault(s0, {}).setdefault(s1, key)
        if not _keep_biclique_key(realkey, key, chainlength):
            by_size[s0][s1] = key

    max_side = {None: max(by_size)}
    for key, val in by_size.items():
        max_side[key] = max(val)
    return {'raw': raw, 'size': by_size, 'max_side': max_side}

cdef class pegasus_busgraph:
    cdef topo_cache[pegasus_spec] *topo
    cdef nodes_t nodes
    cdef embedding_t emb_1
    cdef readonly object identifier
    def __cinit__(self, g):
        rows = g.graph['rows']
        voff = [o//2 for o in g.graph['vertical_offsets'][::2]]
        hoff = [o//2 for o in g.graph['horizontal_offsets'][::2]]
        cdef pegasus_spec *peg = new pegasus_spec(rows, voff, hoff)
        self.nodes = g.nodes()
        cdef edges_t edges = g.edges()
        self.topo = new topo_cache[pegasus_spec](peg[0], self.nodes, edges)
        short_clique(peg[0], self.nodes, edges, self.emb_1)

        #TODO replace this garbage with data from topo
        self.identifier = (rows, tuple(voff), tuple(hoff), tuple(sorted(self.nodes)),
                           tuple(sorted(map(tuple, map(sorted, edges)))))
        del peg

    def __dealloc__(self):
        del self.topo

    def bicliques(self):
        cdef vector[pair[pair[size_t, size_t], embedding_t]] embs
        best_bicliques[pegasus_spec](self.topo[0], embs)
        return _make_biclique_cache(embs)

    def cliques(self):
        cdef vector[embedding_t] embs
        best_cliques[pegasus_spec](self.topo[0], embs, self.emb_1)
        return _make_clique_cache(embs)

    def independent_set(self, size):
        cdef int i
        if size < self.nodes.size():
            emb = [self.nodes[i] for i in range(size)]
            return emb

cdef class chimera_busgraph:
    cdef topo_cache[chimera_spec] *topo
    cdef embedding_t emb_1
    cdef nodes_t nodes
    cdef readonly object identifier
    def __cinit__(self, g):
        rows = g.graph['rows']
        cols = g.graph['columns']
        tile = g.graph['tile']
        if tile > 8:
            raise NotImplementedError(("this clique embedder only supports chimera "
                                       "graphs with a tile size of 8 or less"))

        cdef chimera_spec *chim = new chimera_spec(rows, cols, tile)
        self.nodes = g.nodes()
        cdef edges_t edges = g.edges()
        self.topo = new topo_cache[chimera_spec](chim[0], self.nodes, edges)
        short_clique(chim[0], self.nodes, edges, self.emb_1)

        #TODO replace this garbage with data from topo
        self.identifier = (rows, cols, tile, tuple(sorted(self.nodes)),
                           tuple(sorted(tuple(sorted(e)) for e in edges)))

        del chim


    def __dealloc__(self):
        del self.topo

    def bicliques(self):
        cdef vector[pair[pair[size_t, size_t], embedding_t]] embs
        best_bicliques(self.topo[0], embs)
        return _make_biclique_cache(embs)

    def cliques(self):
        cdef vector[embedding_t] embs
        best_cliques(self.topo[0], embs, self.emb_1)
        return _make_clique_cache(embs)

    def independent_set(self, size):
        cdef int i
        if size < self.nodes.size():
            emb = [self.nodes[i] for i in range(size)]
            return emb

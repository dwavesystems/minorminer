# distutils: language = c++
# cython: language_level=3

from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libc.stdint cimport uint8_t
from libcpp.unordered_map cimport unordered_map
from pickle import dump, load
import homebase, os, pathlib
ctypedef vector[size_t] nodes_t
ctypedef vector[vector[size_t]] embedding_t
ctypedef vector[pair[size_t,size_t]] edges_t
cdef int __cache_version__ = 1

cdef extern from "../include/util.hpp" namespace "busclique":
    cdef cppclass pegasus_spec:
        pegasus_spec(size_t, vector[uint8_t], vector[uint8_t])

    cdef cppclass chimera_spec:
        chimera_spec(vector[size_t], uint8_t)
        chimera_spec(size_t, size_t, uint8_t)

cdef extern from "../include/cell_cache.hpp" namespace "busclique":
    cdef cppclass cell_cache[T]:
        cell_cache(T, nodes_t, edges_t)

cdef extern from "../include/bundle_cache.hpp" namespace "busclique":
    cdef cppclass bundle_cache[T]:
        bundle_cache(cell_cache[T] &)

cdef extern from "../include/clique_cache.hpp" namespace "busclique":
    cdef cppclass clique_cache[T]:
        clique_cache(cell_cache[T] &, bundle_cache[T] &, size_t)
        clique_cache(cell_cache[T] &, bundle_cache[T] &, size_t, size_t)

    cdef cppclass clique_iterator[T]:
        clique_iterator(cell_cache[T] &, clique_cache[T] &)
        int next(embedding_t &)

cdef extern from "../include/topo_cache.hpp" namespace "busclique":
    cdef cppclass topo_cache[T]:
        topo_cache(T, nodes_t &, edges_t &)

cdef extern from "../include/find_clique.hpp" namespace "busclique":
    int find_clique[T](T, nodes_t, edges_t, size_t, embedding_t &)
    int find_clique_nice[T](T, nodes_t, edges_t, size_t, embedding_t &)
    void best_cliques[T](topo_cache[T], vector[embedding_t] &, embedding_t &)
    int short_clique[T](T, nodes_t, edges_t, embedding_t &)


cdef extern from "../include/find_biclique.hpp" namespace "busclique":
    void best_bicliques[T](topo_cache[T], vector[pair[pair[size_t, size_t], embedding_t]] &)


def chimera_clique(g, size, use_cache = True):
    if use_cache:
        G = busclique_cache(g)
        return G.get_clique(size)

    cdef chimera_spec *chim = new chimera_spec(g.graph['rows'],
                                               g.graph['columns'],
                                               g.graph['tile'])
    cdef nodes_t nodes = g.nodes()
    cdef edges_t edges = g.edges()
    cdef embedding_t emb
    find_clique(chim[0], nodes, edges, size, emb)
    try:
        return emb
    finally:
        del chim

def pegasus_clique(g, size, use_cache = True):
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

class busclique_cache:
    def __init__(self, g):
        self._family = g.graph['family']
        if(self._family == 'chimera'):
            self._graph = chimera_graph(g)
        elif(self._family == 'pegasus'):
            self._graph = pegasus_graph(g)
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
        #TODO clean up this data dir a la LRU cache
        basedir = homebase.user_data_dir('busclique', 'dwave', __cache_version__,
                                         dirname)
        pathlib.Path(basedir).mkdir(parents=True, exist_ok=True)
        identifier = self._graph.identifier
        shortcode = str(hash(identifier))
        filename = os.path.join(basedir, str(shortcode))
        if os.path.exists(filename):
            with open(filename, 'rb') as filecache:
                currcache = load(filecache)
        else:
            currcache = {}
        cache = currcache.get(identifier)
        if cache is None:
            cache = compute()
            currcache[identifier] = cache
            with open(filename, 'wb') as filecache:
                dump(currcache, filecache)
        return cache

    def largest_clique_by_chainlength(self, chainlength):
        self._ensure_clique_cache()
        emb = self._cliques['length'].get(chainlength)
        if emb is None:
            raise ValueError("no clique of that chainlength found")
        return emb

    def get_clique(self, size):
        self._ensure_clique_cache()
        emb = self._cliques['size'].get(size)
        if emb is None:
            raise ValueError("no clique of that size found")
        return emb

cdef _clean_clique_cache(vector[embedding_t] embs):
    cdef embedding_t emb
    cdef size_t maxsize = 0
    cdef size_t i, j
    cdef dict by_length = {}
    cdef dict by_size = {}
    for length, emb in enumerate(embs):
        if maxsize >= emb.size():
            continue
        maxsize = max(emb.size(), maxsize)
        by_length[length] = emb
        by_size[emb.size()] = emb
    for i in range(maxsize+1):
        for j in range(i, maxsize+1):
            if j in by_size:
                by_size[i] = dict(enumerate(by_size[j][:i]))
                break
    for l, e in by_length.items():
        by_length[l] = by_size[len(e)]
    return {'length': by_length, 'size': by_size}

cdef class pegasus_graph:
    cdef topo_cache[pegasus_spec] *topo
    cdef embedding_t emb_1
    cdef readonly object identifier
    def __cinit__(self, g):
        rows = g.graph['rows']
        voff = [o//2 for o in g.graph['vertical_offsets'][::2]]
        hoff = [o//2 for o in g.graph['horizontal_offsets'][::2]]
        cdef pegasus_spec *peg = new pegasus_spec(rows, voff, hoff)
        cdef nodes_t nodes = g.nodes()
        cdef edges_t edges = g.edges()
        self.topo = new topo_cache[pegasus_spec](peg[0], nodes, edges)
        short_clique(peg[0], nodes, edges, self.emb_1)

        #TODO replace this garbage with data from topo
        self.identifier = (rows, tuple(voff), tuple(hoff), tuple(sorted(nodes)),
                           tuple(sorted(map(tuple, map(sorted, edges)))))
        del peg

    def __dealloc__(self):
        del self.topo

    def bicliques(self):
        cdef vector[pair[pair[size_t, size_t], embedding_t]] embs
        best_bicliques[pegasus_spec](self.topo[0], embs)
        return {(s0, s1) : emb for (s0, s1), emb in embs}

    def cliques(self):
        cdef vector[embedding_t] embs
        best_cliques[pegasus_spec](self.topo[0], embs, self.emb_1)
        return _clean_clique_cache(embs)

cdef class chimera_graph:
    cdef topo_cache[chimera_spec] *topo
    cdef embedding_t emb_1
    cdef readonly object identifier
    def __cinit__(self, g):
        rows = g.graph['rows']
        cols = g.graph['columns']
        tile = g.graph['tile']
        if tile > 8:
            raise NotImplementedError(("this clique embedder only supports chimera "
                                       "graphs with a tile size of 8 or less"))

        cdef chimera_spec *chim = new chimera_spec(rows, cols, tile)
        cdef nodes_t nodes = g.nodes()
        cdef edges_t edges = g.edges()
        self.topo = new topo_cache[chimera_spec](chim[0], nodes, edges)
        short_clique(chim[0], nodes, edges, self.emb_1)

        #TODO replace this garbage with data from topo
        self.identifier = (rows, cols, tile, tuple(sorted(nodes)),
                           tuple(sorted(tuple(sorted(e)) for e in edges)))

        del chim


    def __dealloc__(self):
        del self.topo

    def bicliques(self):
        cdef vector[pair[pair[size_t, size_t], embedding_t]] embs
        best_bicliques(self.topo[0], embs)
        return {(s0, s1) : emb for (s0, s1), emb in embs}

    def cliques(self):
        cdef vector[embedding_t] embs
        best_cliques(self.topo[0], embs, self.emb_1)
        return _clean_clique_cache(embs)





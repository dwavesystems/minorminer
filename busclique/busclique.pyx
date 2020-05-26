# distutils: language = c++
# cython: language_level=3

from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libc.stdint cimport uint8_t
ctypedef vector[size_t] nodes_t
ctypedef vector[vector[size_t]] embedding_t
ctypedef vector[pair[size_t,size_t]] edges_t

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

cdef extern from "../include/find_clique.hpp" namespace "busclique":
    int find_clique[T](T, nodes_t, edges_t, size_t, embedding_t &)
    int find_clique_nice[T](T, nodes_t, edges_t, size_t, embedding_t &)

def chimera_clique(g, size):
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

def pegasus_clique(g, size):
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

def pegasus_clique_residency(g, width, maxlen=0):
    cdef pegasus_spec *peg = new pegasus_spec(g.graph['rows'],
                                   [o//2 for o in g.graph['vertical_offsets'][::2]],
                                   [o//2 for o in g.graph['horizontal_offsets'][::2]])
    cdef nodes_t nodes = g.nodes()
    cdef edges_t edges = g.edges()
    cdef embedding_t emb
    cdef cell_cache[pegasus_spec] *cells = new cell_cache[pegasus_spec](peg[0], nodes, edges)
    cdef bundle_cache[pegasus_spec] *bundles = new bundle_cache[pegasus_spec](cells[0])
    cdef clique_cache[pegasus_spec] *cliques = new clique_cache[pegasus_spec](cells[0], bundles[0], width)
    cdef clique_iterator[pegasus_spec] *clique_iter = new clique_iterator[pegasus_spec](cells[0], cliques[0])

    cdef vector[size_t] count = [0]*(max(g)+1)
    cdef size_t i = 0
    while clique_iter.next(emb):
        for chain in emb:
            for q in chain:
                count[q] += 1
        if (i%1048576) == 0:
            print(i)
        i += 1

    try:
        return count
    finally:
        del clique_iter
        del cliques
        del bundles
        del cells
        del peg

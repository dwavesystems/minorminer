# distutils: language = c++
# cython: language_level=3

from libcpp.vector cimport vector
from libcpp.pair cimport pair
ctypedef vector[vector[size_t]] embedding_t
ctypedef vector[pair[size_t,size_t]] edges_t

cdef extern from "../include/clique_embed.hpp" namespace "chimera_clique":
    cdef cppclass clique_generator:
        clique_generator(size_t, size_t, size_t, edges_t, size_t)
        int next(embedding_t &)

    void experiment(size_t, size_t, size_t, vector[pair[size_t, size_t]], size_t, vector[vector[size_t]]&);

def run_experiment(g, width):
    cdef size_t dim0 = g.graph['rows']
    cdef size_t dim1 = g.graph['columns']
    cdef size_t shore = g.graph['tile']
    cdef vector[pair[size_t, size_t]] edges = g.edges();
    cdef vector[vector[size_t]] emb = vector[vector[size_t]](0,vector[size_t](0,0));
    experiment(dim0, dim1, shore, edges, width, emb)

#    for chain in emb:
#        for q in chain:
#            print(q, ((q%8)&4)//4, end=", ")
#        print()
    return [[q for q in c] for c in emb]

def all_max_cliques(g, width):
    cdef size_t dim0 = g.graph['rows']
    cdef size_t dim1 = g.graph['columns']
    cdef size_t shore = g.graph['tile']
    cdef edges_t edges = g.edges();
    cdef embedding_t emb;
    cdef clique_generator *cg = new clique_generator(dim0, dim1, shore, edges, width)
    try:
        while cg.next(emb):
            yield emb
    finally:
        del cg

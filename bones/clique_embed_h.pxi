from libcpp.vector cimport vector
from libcpp.pair cimport pair

cdef extern from "clique_embed.hpp" namespace "find_embedding":
    vector[vector[size_t]] experiment(size_t, size_t, vector[pair[size_t, size_t]]);



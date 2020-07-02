# Copyright 2020 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libc.stdint cimport uint8_t
ctypedef vector[size_t] nodes_t
ctypedef vector[vector[size_t]] embedding_t
ctypedef vector[pair[size_t,size_t]] edges_t

cdef extern from "../include/busclique/util.hpp" namespace "busclique":
    cdef cppclass pegasus_spec:
        pegasus_spec(size_t, vector[uint8_t], vector[uint8_t])

    cdef cppclass chimera_spec:
        chimera_spec(vector[size_t], uint8_t)
        chimera_spec(size_t, size_t, uint8_t)

cdef extern from "../include/busclique/cell_cache.hpp" namespace "busclique":
    cdef cppclass cell_cache[T]:
        cell_cache(T, nodes_t, edges_t)

cdef extern from "../include/busclique/bundle_cache.hpp" namespace "busclique":
    cdef cppclass bundle_cache[T]:
        bundle_cache(cell_cache[T] &)

cdef extern from "../include/busclique/clique_cache.hpp" namespace "busclique":
    cdef cppclass clique_cache[T]:
        clique_cache(cell_cache[T] &, bundle_cache[T] &, size_t)
        clique_cache(cell_cache[T] &, bundle_cache[T] &, size_t, size_t)

    cdef cppclass clique_iterator[T]:
        clique_iterator(cell_cache[T] &, clique_cache[T] &)
        int next(embedding_t &)

cdef extern from "../include/busclique/topo_cache.hpp" namespace "busclique":
    cdef cppclass topo_cache[T]:
        topo_cache(T, nodes_t &, edges_t &)

cdef extern from "../include/busclique/find_clique.hpp" namespace "busclique":
#    int find_clique[T](T, nodes_t, edges_t, size_t, embedding_t &)
    int find_clique[T](topo_cache[T] &, size_t, embedding_t &)
#    int find_clique_nice[T](T, nodes_t, edges_t, size_t, embedding_t &)
    void best_cliques[T](topo_cache[T], vector[embedding_t] &, embedding_t &)
    int short_clique[T](T, nodes_t, edges_t, embedding_t &)


cdef extern from "../include/busclique/find_biclique.hpp" namespace "busclique":
    void best_bicliques[T](topo_cache[T], vector[pair[pair[size_t, size_t], embedding_t]] &)



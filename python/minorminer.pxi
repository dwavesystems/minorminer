from libcpp.vector cimport vector
from libcpp.set cimport set
from libcpp.map cimport map
from libcpp.pair cimport pair

ctypedef pair[int,int] intpair
ctypedef pair[intpair, int] intpairint
ctypedef map[intpair, int] edgemap
ctypedef map[int, vector[int]] chainmap

cdef class labeldict(dict):
    cdef list _label
    def __init__(self,*args,**kwargs):
        super(labeldict,self).__init__(args,**kwargs)
        self._label = []
    def __missing__(self,l):
        self[l] = k = len(self._label)
        self._label.append(l)
        return k
    def label(self,k):
        return self._label[k]

cdef extern from "<memory>" namespace "std":
    cdef cppclass shared_ptr[T]:
        shared_ptr() nogil
        void reset(T*)

cdef extern from "<random>" namespace "std":
    cdef cppclass default_random_engine:
        pass

cdef extern from "../include/graph.hpp" namespace "graph":
    cppclass input_graph:
        input_graph()
        void push_back(int,int)
        int num_nodes()
        void clear()


cdef extern from "../include/pairing_queue.hpp" namespace "pairing_queue":
    cppclass pairing_queue
    cppclass pairing_queue_fast_reset

cdef extern from "../include/pathfinder.hpp" namespace "find_embedding":
    cppclass pathfinder

cdef extern from "../include/embedding_problem.hpp" namespace "find_embedding":
    pass

cdef extern from "../include/embedding.hpp" namespace "find_embedding":
    pass

cdef extern from "../include/util.hpp" namespace "find_embedding":
    cppclass LocalInteraction:
        pass

    ctypedef shared_ptr[LocalInteraction] LocalInteractionPtr


    cppclass optional_parameters:
        optional_parameters()
        void seed(unsigned int)

        bint fast_embedding
        LocalInteractionPtr localInteractionPtr
        int max_no_improvement
        default_random_engine rng
        double timeout
        int tries
        int verbose
        int inner_rounds
        int max_fill
        int chainlength_patience
        bint return_overlap
        bint skip_initialization
        chainmap fixed_chains
        chainmap initial_chains
        chainmap restrict_chains
        int threads


cdef extern from "../include/find_embedding.hpp" namespace "find_embedding":
    int findEmbedding(input_graph, input_graph, optional_parameters, vector[vector[int]]&) except +


cdef extern from "minorminer.pyx.hpp" namespace "":
    cppclass LocalInteractionPython(LocalInteraction):
        LocalInteractionPython()

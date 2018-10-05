from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.pair cimport pair
from libc.stdint cimport uint8_t, uint64_t

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

    cdef cppclass unique_ptr[T]:
        unique_ptr() nogil

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
    cppclass pathfinder_public_interface

cdef extern from "../include/embedding_problem.hpp" namespace "find_embedding":
    cpdef enum VARORDER:
        VARORDER_SHUFFLE = 0
        VARORDER_DFS = 1
        VARORDER_BFS = 2
        VARORDER_PFS = 3
        VARORDER_RPFS = 4
        VARORDER_KEEP = 5

    cppclass parameter_processor:
        int num_vars
        optional_parameters params

cdef extern from "../include/embedding.hpp" namespace "find_embedding":
    pass

cdef extern from "../include/chain.hpp" namespace "find_embedding":
    cppclass pathfinder_wrapper:
        pathfinder_wrapper()
        parameter_processor pp
        unique_ptr[pathfinder_public_interface] pf
        pathfinder_wrapper(input_graph &, input_graph &, optional_parameters &)
        int heuristicEmbedding()
        int num_vars()
        void get_chain(int, vector[int] &)
        void set_initial_chains(chainmap &)
        void quickPass(const vector[int] &, int, int, bool, bool, double)
        void quickPass(VARORDER, int, int, bool, bool, double)

    cppclass chain:
        chain(vector[int] &w, int l)
        inline int size() const 
        inline int count(const int q) const
        inline int get_link(const int x) const 
        inline void set_link(const int x, const int q)
        inline int drop_link(const int x)
        inline void set_root(const int q)
        inline void clear()
        inline void add_leaf(const int q, const int parent)
        inline int trim_branch(int q)
        inline int trim_leaf(int q)
        inline int parent(const int q) const
        inline int refcount(const int q) const
        void link_path(chain &other, int q, const vector [int] &parents)
        inline void diagnostic(char *last_op)


cdef class cppembedding:
    cdef vector[chain] chains
    cdef vector[int] qubit_weights
    def __cinit__(self, int num_vars, int num_qubits):
        pass
    

cdef extern from "../include/util.hpp" namespace "find_embedding":
    cppclass LocalInteraction:
        pass

    ctypedef shared_ptr[LocalInteraction] LocalInteractionPtr


    cppclass optional_parameters:
        optional_parameters()
        void seed(uint64_t)

        LocalInteractionPtr localInteractionPtr
        int max_no_improvement
        default_random_engine rng
        double timeout
        double max_beta
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

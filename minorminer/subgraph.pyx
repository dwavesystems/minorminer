# Copyright 2022 D-Wave Systems Inc.
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
#
# This file excerpts from the Glasgow Subgraph Solver, license follows.
#
# Copyright (c) 2013-2021 Ciaran McCreesh
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# distutils: language = c++
# cython: language_level = 3

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.pair cimport pair
from libc.stdint cimport uint8_t, uint64_t
from libcpp.string cimport string

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


cdef extern from "<chrono>" namespace "std::chrono" nogil:
    cdef cppclass time_point[T]:
        pass
    cdef cppclass steady_clock:
        pass
    cdef cppclass seconds:
        seconds()
    cdef cppclass milliseconds:
        seconds()
    cdef seconds make_seconds "std::chrono::seconds"(int)
    cdef milliseconds make_milliseconds "std::chrono::milliseconds"(int)
    cdef time_point[steady_clock] steady_clock_now "std::chrono::steady_clock::now"()

cdef extern from "glasgow-subgraph-solver/src/timeout.hh" nogil:
    cppclass Timeout:
        Timeout(seconds)

cdef extern from "glasgow-subgraph-solver/src/formats/input_graph.hh" nogil:
    cdef cppclass InputGraph:
        InputGraph(int, bool, bool)
        void add_edge(int, int)
        void resize(int)

cdef extern from "glasgow-subgraph-solver/src/restarts.hh" nogil:
    cdef cppclass RestartsSchedule:
        pass
        
    cdef unsigned long long default_luby_multiplier "LubyRestartsSchedule::default_multiplier"
    cdef milliseconds default_timed_duration "TimedRestartsSchedule::default_duration"
    cdef unsigned long long default_timed_backtracks "TimedRestartsSchedule::default_minimum_backtracks"
    cdef double default_geometric_constant "GeometricRestartsSchedule::default_initial_value"
    cdef double default_geometric_multiplier "GeometricRestartsSchedule::default_multiplier"

cdef extern from "<memory>" namespace "std" nogil:
    cdef cppclass shared_ptr[T]:
        pass
    cdef cppclass unique_ptr[T]:
        pass
    cdef shared_ptr[Timeout] make_shared_timeout "std::make_shared<Timeout>"(seconds)
    cdef shared_ptr[InputGraph] make_shared[InputGraph](int, bool, bool)
    cdef InputGraph deref "*"(shared_ptr[InputGraph])
    cdef unique_ptr[RestartsSchedule] make_no_restarts_schedule "std::make_unique<NoRestartsSchedule>"()
    cdef unique_ptr[RestartsSchedule] make_luby_restarts_schedule "std::make_unique<LubyRestartsSchedule>"(unsigned long long)
    cdef unique_ptr[RestartsSchedule] make_geometric_restarts_schedule "std::make_unique<GeometricRestartsSchedule>"(double, double)
    cdef unique_ptr[RestartsSchedule] make_timed_restarts_schedule "std::make_unique<TimedRestartsSchedule>"(milliseconds, unsigned long long)


cdef extern from "glasgow-subgraph-solver/src/vertex_to_vertex_mapping.hh" nogil:
    cdef cppclass VertexToVertexMapping:
        pass

cdef extern from "glasgow-subgraph-solver/src/value_ordering.hh" nogil:
    #this enum contains None
    cdef enum ValueOrdering 'ValueOrdering':
        _VO_None 'ValueOrdering::None'
        _VO_Biased 'ValueOrdering::Biased'
        _VO_Degree 'ValueOrdering::Degree'
        _VO_AntiDegree 'ValueOrdering::AntiDegree'
        _VO_Random 'ValueOrdering::Random'

cdef extern from "glasgow-subgraph-solver/src/homomorphism.hh" nogil:
    cppclass HomomorphismParams:
        shared_ptr[Timeout] timeout
        time_point[steady_clock] start_time
        unique_ptr[RestartsSchedule] restarts_schedule
        unsigned n_threads
        bool triggered_restarts
        bool delay_thread_creation
        ValueOrdering value_ordering_heuristic
        bool clique_detection
        bool distance3
        bool k4
        bool no_supplementals
        int number_of_exact_path_graphs
        bool clique_size_constraints
        bool clique_size_constraints_on_supplementals
        bool no_nds

    cppclass HomomorphismResult:
        map[int, int] mapping

cdef extern from "glasgow-subgraph-solver/src/sip_decomposer.hh" nogil:
    cdef HomomorphismResult solve_sip_by_decomposition(InputGraph, InputGraph, HomomorphismParams) except+

def find_subgraph(source, target, **kwargs):
    """
    Use the Glasgow Subgraph Solver to find a subgraph isomorphism from source
    to target.
    
    For more information, see the Glasgow subgraph solver at
    
        https://github.com/ciaranm/glasgow-subgraph-solver
        
    The full solver has more functionality than is exposed in this wrapper.
    
    If you use this function for research, please cite
    
        https://dblp.org/rec/conf/gg/McCreeshP020.html
        
    or if you use it in a non-research setting, please contact Ciaran McCreesh
    (ciaran.mccreesh@glasgow.ac.uk) if possible.  Their software is an output of
    taxpayer funded research, and it is very helpful for them if they can
    demonstrate real-world impact when they write grant applications.

    Args:
        S (iterable/NetworkX Graph): 
            The source graph as an iterable of label pairs representing the 
            edges, or a NetworkX Graph.

        T (iterable/NetworkX Graph):
            The target graph as an iterable of label pairs representing the 
            edges, or a NetworkX Graph.

        **params (optional): See below.

    Returns: 
        A dict that maps labels in S to labels in T. If no isomorphism
        is found, and empty dictionary is returned.

    Optional Parameters:
        timeout  (int, optional, default=0)
            Abort after this many seconds
        parallel (bool, optional, default=False):
            Use auto-configured parallel search (highly nondeterministic runtimes)

    Advanced Parallelism Options
        threads (int, optional, default=1):
            Use threaded search, with this many threads (0 to auto-detect)
        triggered_restarts (bool, optional, default=False):
            Have one thread trigger restarts (more nondeterminism, better performance)
        delay_thread_creation (bool, optional, default=False):
            Do not create threads until after the first restart

    Advanced Search Configuration Options:
        restarts_policy (string, optional, default='luby'):
            Specify restart policy ('luby', 'geometric', 'timed' or 'none')
        luby_constant (int, optional, default=666):
            Specify the starting constant / multiplier for Luby restarts
        geometric_constant (double, optional, 5400):
            Specify starting constant for geometric restarts
        geometric_multiplier (double, optional, default=1.0):
            Specify multiplier for geometric restarts
        restart_interval (int, optional, default=100):
            Specify the restart interval in milliseconds for timed restarts
        restart_minimum (int, optional, default=100):
            Specify a minimum number of backtracks before a timed restart can trigger    
        value_ordering (string, optional, default='biased'):
            Specify value-ordering heuristic ('biased', 'degree', 'antidegree', 'random', 'none')

    Advanced Input Processing Options:
        clique_detection (bool, optional, default=True):
            Enable clique / independent set detection
        no_supplementals (bool, optional, default=False):
            Do not use supplemental graphs
        no_nds (bool, optional, default=false):
            Do not use neighbourhood degree sequences

    Hidden Options:
        distance3 (bool, optional, default=False):
            Use distance 3 filtering (experimental)
        k4 (bool, optional, default=False):
            Use 4-clique filtering (experimental)
        n_exact_path_graphs (int, optional, default=4):
            Specify number of exact path graphs
        cliques (bool, optional, default=False):
            Use clique size constraints
        cliques_on_supplementals (bool, optional, default=False):
            Use clique size constraints on supplemental graphs too

    """
    cdef shared_ptr[InputGraph] source_g = make_shared[InputGraph](0, False, False)
    cdef shared_ptr[InputGraph] target_g = make_shared[InputGraph](0, False, False)
    cdef labeldict source_labels = _read_graph(deref(source_g), source)
    cdef labeldict target_labels = _read_graph(deref(target_g), target)
    cdef HomomorphismParams params

    cdef bool parallel = kwargs.pop('parallel', False)

    params.triggered_restarts = kwargs.pop('triggered_restarts', parallel)

    restarts_policy = kwargs.pop('restarts_policy', None)
    if restarts_policy == 'luby':
        multiplier = kwargs.pop('luby_constant', default_luby_multiplier)
        params.restarts_schedule = make_luby_restarts_schedule(multiplier)
    elif restarts_policy == 'geometric':
        constant = kwargs.pop('geometric_constant', default_geometric_constant)
        multiplier = kwargs.pop('geometric_multiplier', default_geometric_multiplier)
        params.restarts_schedule = make_geometric_restarts_schedule(constant, multiplier)
    elif restarts_policy == 'timed':
        interval = kwargs.pop('restart_interval', None)
        backtracks = kwargs.pop('restart_minimum', default_timed_backtracks)
        params.restarts_schedule = make_timed_restarts_schedule(
            default_timed_duration if interval is None else make_milliseconds(interval),
            backtracks
        )
    elif restarts_policy == 'none':
        params.restarts_schedule = make_no_restarts_schedule()
    elif restarts_policy is None:
        if parallel:
            params.restarts_schedule = make_timed_restarts_schedule(default_timed_duration, default_timed_backtracks)
        else:
            params.restarts_schedule = make_luby_restarts_schedule(default_luby_multiplier)                
    else:
        raise ValueError(f"restarts_policy {restarts_policy} not recognized")

    params.n_threads = kwargs.pop('threads', 1)
    if parallel:
        params.n_threads = 0

    params.delay_thread_creation = kwargs.pop('delay_thread_creation', parallel)

    value_ordering = kwargs.pop('value_ordering', 'biased')
    if value_ordering == 'none':
        params.value_ordering_heuristic = _VO_None
    elif value_ordering == 'biased':
        params.value_ordering_heuristic = _VO_Biased
    elif value_ordering == 'degree':
        params.value_ordering_heuristic = _VO_Degree
    elif value_ordering == 'antidegree':
        params.value_ordering_heuristic = _VO_AntiDegree
    elif value_ordering == 'random':
        params.value_ordering_heuristic = _VO_Random
    else:
        raise RuntimeError("unknown value ordering heuristic")

    params.clique_detection = kwargs.pop('clique_detection', params.clique_detection)
    params.distance3 = kwargs.pop('distance3', params.distance3)
    params.k4 = kwargs.pop('k4', params.k4)
    params.number_of_exact_path_graphs = kwargs.pop('n_exact_path_graphs', params.number_of_exact_path_graphs)
    params.no_supplementals = kwargs.pop('no_supplementals', params.no_supplementals)
    params.no_nds = kwargs.pop('no_nds', params.no_nds)
    params.clique_size_constraints = kwargs.pop('cliques', params.clique_size_constraints)
    params.clique_size_constraints_on_supplementals = kwargs.pop(
        'cliques_on_supplementals',
        params.clique_size_constraints_on_supplementals
    )

    params.timeout = make_shared_timeout(make_seconds(kwargs.pop('timeout', 0)))
    params.start_time = steady_clock_now()

    if kwargs:
        raise ValueError("unknown/unused parameters: {list(kwargs.keys())}")

    cdef HomomorphismResult result = solve_sip_by_decomposition(deref(source_g), deref(target_g), params)

    return dict((source_labels.label(s), target_labels.label(t)) for s, t in result.mapping)

cdef _read_graph(InputGraph &g, E):
    cdef labeldict L = labeldict()
    if hasattr(E, 'edges'):
        G = E
        E = E.edges()
        for a in G.nodes():
            L[a]
        
    for a, b in E:
        g.add_edge(L[a],L[b])

    g.resize(len(L))
    return L

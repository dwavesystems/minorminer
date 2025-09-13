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
import random as _random

cdef class _labeldict(dict):
    cdef list _label
    def __init__(self,*args,**kwargs):
        super(_labeldict,self).__init__(args,**kwargs)
        self._label = []
    def __missing__(self,l):
        self[l] = k = len(self._label)
        self._label.append(l)
        return k
    def shuffle(self, random):
        random.shuffle(self._label)
        for i, l in enumerate(self._label):
            self[l] = i
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

cdef extern from "glasgow-subgraph-solver/gss/timeout.hh" namespace "gss" nogil:
    cppclass Timeout:
        Timeout(seconds)

cdef extern from "glasgow-subgraph-solver/gss/formats/input_graph.hh" nogil:
    cdef cppclass InputGraph:
        InputGraph(int, bool, bool)
        void add_edge(int, int)
        void add_directed_edge(int, int, string)
        void resize(int)
        void set_vertex_label(int, string)

cdef extern from "glasgow-subgraph-solver/gss/restarts.hh" namespace "gss" nogil:
    cdef cppclass RestartsSchedule:
        pass
        
    cdef unsigned long long default_luby_multiplier "gss::LubyRestartsSchedule::default_multiplier"
    cdef milliseconds default_timed_duration "gss::TimedRestartsSchedule::default_duration"
    cdef unsigned long long default_timed_backtracks "gss::TimedRestartsSchedule::default_minimum_backtracks"
    cdef double default_geometric_constant "gss::GeometricRestartsSchedule::default_initial_value"
    cdef double default_geometric_multiplier "gss::GeometricRestartsSchedule::default_multiplier"

cdef extern from "<memory>" namespace "std" nogil:
    cdef cppclass shared_ptr[T]:
        pass
    cdef cppclass unique_ptr[T]:
        pass
    cdef shared_ptr[Timeout] make_shared_timeout "std::make_shared<gss::Timeout>"(seconds)
    cdef shared_ptr[InputGraph] make_shared[InputGraph](int, bool, bool)
    cdef InputGraph deref "*"(shared_ptr[InputGraph])
    cdef unique_ptr[RestartsSchedule] make_no_restarts_schedule "std::make_unique<gss::NoRestartsSchedule>"()
    cdef unique_ptr[RestartsSchedule] make_luby_restarts_schedule "std::make_unique<gss::LubyRestartsSchedule>"(unsigned long long)
    cdef unique_ptr[RestartsSchedule] make_geometric_restarts_schedule "std::make_unique<gss::GeometricRestartsSchedule>"(double, double)
    cdef unique_ptr[RestartsSchedule] make_timed_restarts_schedule "std::make_unique<gss::TimedRestartsSchedule>"(milliseconds, unsigned long long)


cdef extern from "glasgow-subgraph-solver/gss/vertex_to_vertex_mapping.hh" namespace "gss" nogil:
    cdef cppclass VertexToVertexMapping:
        pass

cdef extern from "glasgow-subgraph-solver/gss/value_ordering.hh" namespace "gss" nogil:
    #this enum contains None
    cdef enum ValueOrdering 'gss::ValueOrdering':
        _VO_None 'gss::ValueOrdering::None'
        _VO_Biased 'gss::ValueOrdering::Biased'
        _VO_Degree 'gss::ValueOrdering::Degree'
        _VO_AntiDegree 'gss::ValueOrdering::AntiDegree'
        _VO_Random 'gss::ValueOrdering::Random'

cdef extern from "glasgow-subgraph-solver/gss/homomorphism.hh" namespace "gss" nogil:
    cdef enum Injectivity 'gss:Injectivity':
        _I_Injective 'gss::Injectivity::Injective'
        _I_LocallyInjective 'gss::Injectivity::LocallyInjective'
        _I_NonInjective 'gss::Injectivity::NonInjective'

    cppclass HomomorphismParams:
        shared_ptr[Timeout] timeout
        time_point[steady_clock] start_time
        unique_ptr[RestartsSchedule] restarts_schedule
        unsigned n_threads
        bool triggered_restarts
        bool delay_thread_creation
        ValueOrdering value_ordering_heuristic
        Injectivity injectivity
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

cdef extern from "glasgow-subgraph-solver/gss/sip_decomposer.hh" namespace "gss" nogil:
    cdef HomomorphismResult solve_sip_by_decomposition(InputGraph, InputGraph, HomomorphismParams) except+

_default_kwarg = object()
def _check_kwarg(kwargs, name, default):
    value = kwargs.pop(name)
    return default if value is _default_kwarg else value

def find_subgraph(
    source,
    target,
    timeout=0,
    parallel=False,
    node_labels=None,
    edge_labels=None,
    as_embedding=False,
    injectivity='injective',
    seed=None,
    threads=1,
    triggered_restarts=_default_kwarg,
    delay_thread_creation=_default_kwarg,
    restarts_policy=_default_kwarg,
    luby_constant=_default_kwarg,
    geometric_constant=_default_kwarg,
    geometric_multiplier=_default_kwarg,
    restart_interval=_default_kwarg,
    restart_minimum=_default_kwarg,
    value_ordering='biased',
    clique_detection=True,
    no_supplementals=False,
    no_nds=False,
    distance3=False,
    k4=False,
    n_exact_path_graphs=4,
    cliques=False,
    cliques_on_supplementals=False,
    ):
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
            The source graph as an iterable of node pairs representing the 
            edges, or a NetworkX Graph.

        T (iterable/NetworkX Graph):
            The target graph as an iterable of node pairs representing the 
            edges, or a NetworkX Graph.

        **params (optional): See below.

    Returns: 
        A dict that maps nodes in S to nodes in T. If no isomorphism is found,
        an empty dictionary is returned.

    Optional Parameters:
        timeout  (int, optional, default=0)
            Abort after this many seconds
        parallel (bool, optional, default=False):
            Use auto-configured parallel search (highly nondeterministic runtimes)
        node_labels (tuple, optional, default=None):
            If not ``None``, a pair of dicts (``S_labels``, ``T_labels``) whose keys are
            nodes and values are strings.  Unlabeled nodes are labeled with
            the empty string "".
        edge_labels (tuple, optional, default=None):
            If not ``None``, a pair of dicts (``S_labels``, ``T_labels``) whose keys are
            (source, dest) pairs of nodes corresponding to directed edges, and
            values are strings.  Unlabeled directed edges are labeled with the
            empty string "".  If the label on an edge (u, v) is intended to be
            undirected, you must provide the same label for both directions
            (u, v) and (v, u).
        as_embedding (bool, optional, default=False):
            If ``True``, the values of the returned dictionary will be singleton
            tuples similar to the return type of ``find_embedding``.
        injectivity (string, optional, default='injective'):
            Must be one of ('injective', 'locally injective', 'noninjective').
            By default, this function searches for subgraphs by finding injective
            homomorphisms.  That is, nodes of the target graph can only occur
            once in the output of the mapping.  By providing the default value
            'injective' to the `injectivity` parameter, that is true.  A mapping
            can be said to be 'locally injective' if the mapping is injective
            on the neighborhood of every node.
        seed (int/object, optional, default=None):
            If ``seed`` is an int, it will be used as a seed to a random number
            generator to randomize the algorithm.  This randomization is
            accomplished by shuffling the nodes and edges of the source and
            target graphs.  If ``seed`` is an object with an attribute named
            ``shuffle``, then that function will be called, with the expectation
            that it is equivalent to ``random.shuffle``.

            Note that the placement of nodes without incident edges is not
            subject to explicit randomization.

    Advanced Parallelism Options
        threads (int, optional, default=1):
            Use threaded search, with this many threads (0 to auto-detect)
            (this value is overridden to zero if ``paralell` is ``True``)
        triggered_restarts (bool, optional, default=False):
            Have one thread trigger restarts (more nondeterminism, better performance)
        delay_thread_creation (bool, optional, default=False):
            Do not create threads until after the first restart
            (default is changed to ``True`` if ``parallel`` is ``True``

    Advanced Search Configuration Options:
        restarts_policy (string, optional, default='luby'):
            Specify restart policy ('luby', 'geometric', 'timed' or 'none')
            (default policy is 'timed' with default parameters if ``parallel`` is ``True``)
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
        no_nds (bool, optional, default=False):
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
    if node_labels is None:
        node_labels = (None, None)
    if edge_labels is None:
        edge_labels = (None, None)

    cdef shared_ptr[InputGraph] source_g = make_shared[InputGraph](0, node_labels[0], edge_labels[0])
    cdef shared_ptr[InputGraph] target_g = make_shared[InputGraph](0, node_labels[1], edge_labels[1])

    if seed is None:
        random = None
    elif not hasattr(seed, 'shuffle'):
        random = _random.Random(seed)
    else:
        random = seed

    cdef _labeldict source_labels
    cdef _labeldict target_labels
    source_labels, source_isolated = _read_graph(deref(source_g), source, node_labels[0], edge_labels[0], random)
    target_labels, target_isolated  = _read_graph(deref(target_g), target, node_labels[1], edge_labels[1], random)
    cdef HomomorphismParams params

    if triggered_restarts is _default_kwarg:
        triggered_restarts = parallel

    check_kwargs = {
        'luby_constant': luby_constant,
        'geometric_constant': geometric_constant,
        'geometric_multiplier': geometric_multiplier,
        'restart_interval': restart_interval,
        'restart_minimum': restart_minimum,
    }
    if restarts_policy == 'luby':
        multiplier = _check_kwarg(check_kwargs, 'luby_constant', default_luby_multiplier)
        params.restarts_schedule = make_luby_restarts_schedule(multiplier)
    elif restarts_policy == 'geometric':
        constant = _check_kwarg(check_kwargs, 'geometric_constant', default_geometric_constant)
        multiplier = _check_kwarg(check_kwargs, 'geometric_multiplier', default_geometric_multiplier)
        params.restarts_schedule = make_geometric_restarts_schedule(constant, multiplier)
    elif restarts_policy == 'timed':
        interval = _check_kwarg(check_kwargs, 'restart_interval', None)
        backtracks = _check_kwarg(check_kwargs, 'restart_minimum', default_timed_backtracks)
        params.restarts_schedule = make_timed_restarts_schedule(
            default_timed_duration if interval is None else make_milliseconds(interval),
            backtracks
        )
    elif restarts_policy == 'none':
        params.restarts_schedule = make_no_restarts_schedule()
    elif restarts_policy is _default_kwarg:
        if parallel:
            params.restarts_schedule = make_timed_restarts_schedule(default_timed_duration, default_timed_backtracks)
        else:
            params.restarts_schedule = make_luby_restarts_schedule(default_luby_multiplier)                
    else:
        raise ValueError(f"restarts_policy {restarts_policy} not recognized")

    params.n_threads = 0 if parallel else threads

    if delay_thread_creation is _default_kwarg:
        delay_thread_creation = parallel
    params.delay_thread_creation = delay_thread_creation

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

    if injectivity == 'injective':
        params.injectivity = _I_Injective
    elif injectivity == 'locally injective':
        params.injectivity = _I_LocallyInjective
    elif injectivity == 'noninjective':
        params.injectivity = _I_NonInjective
    else:
        raise RuntimeError("unrecognized injectivity option")

    if clique_detection is not _default_kwarg:
        params.clique_detection = clique_detection
    if distance3 is not _default_kwarg:
        params.distance3 = distance3
    if k4 is not _default_kwarg:
        params.k4 = k4
    if n_exact_path_graphs is not _default_kwarg:
        params.number_of_exact_path_graphs = n_exact_path_graphs
    if n_exact_path_graphs is not _default_kwarg:
        params.no_supplementals = no_supplementals
    if no_nds is not _default_kwarg:
        params.no_nds = no_nds
    if cliques is not _default_kwarg:
        params.clique_size_constraints = cliques
    if cliques_on_supplementals is not _default_kwarg:
        params.clique_size_constraints_on_supplementals = cliques_on_supplementals

    params.timeout = make_shared_timeout(make_seconds(timeout))
    params.start_time = steady_clock_now()

    check_kwargs = {k: v for k, v in check_kwargs.items() if v is not _default_kwarg}
    if check_kwargs:
        raise ValueError(f"unused parameters: {list(check_kwargs.keys())}")

    cdef HomomorphismResult result;
    if len(source_labels) + len(source_isolated) <= len(target_labels) + len(target_isolated) or injectivity != 'injective':
        result = solve_sip_by_decomposition(deref(source_g), deref(target_g), params)
        emb = dict((source_labels.label(s), target_labels.label(t)) for s, t in result.mapping)
    else:
        emb = {}

    if source_isolated and (len(emb) == len(source_labels)):
        if injectivity == 'injective':
            target_isolated.extend(set(target_labels)-set(emb.values()))
            if len(source_isolated) <= len(target_isolated):
                for s, t in zip(source_isolated, target_isolated):
                    emb[s] = t
        elif target_isolated or target_labels:
            t = next(iter(target_isolated or target_labels))
            for s in source_isolated:
                emb[s] = t

    if as_embedding:
        emb = {k: (v,) for k, v in emb.items()}

    return emb

cdef _read_graph(InputGraph &g, E, node_labels, edge_labels, random):
    cdef _labeldict L = _labeldict()
    cdef str label
    isolated_nodes = []
    if hasattr(E, 'edges'):
        G = E
        E = list(E.edges())
        for a, d in G.degree():
            if d:
                L[a]
            else:
                isolated_nodes.append(a)
    else:
        G = None

    if random is not None or node_labels is not None:
        if G is None:
            # E might be a generator... this is a silly-looking line but it
            # walks over the edge-list, puts every node into L, and leaves E
            # functionally unchanged
            E = [(L[a], L[b]) and (a, b) for a, b in E]

    if random is not None:
        L.shuffle(random)
        random.shuffle(E)

    if node_labels is not None:
        # performance note: we really wanna do this in order because as of
        # writing, the Glasgow implementation of set_vertex_label uses vector
        # erase/insert which can result in accidentally-quadratic runtime if
        # we don't write at the end
        for i, a in enumerate(L._label):
            label = node_labels.get(a)
            if label is not None:
                g.resize(i+1)
                g.set_vertex_label(i, bytes(label, "utf8"))

    if edge_labels is None:
        for a, b in E:
            g.add_edge(L[a],L[b])
    else:
        for a, b in E:
            label = edge_labels.get((a, b), "")
            g.add_directed_edge(L[a], L[b], bytes(label, "utf8"))
            label = edge_labels.get((b, a), "")
            g.add_directed_edge(L[b], L[a], bytes(label, "utf8"))

    g.resize(len(L))
    return L, isolated_nodes

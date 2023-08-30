# Copyright 2017 - 2020 D-Wave Systems Inc.
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

# distutils: language = c++
# cython: language_level=2
# Note: chosing to use language_level=2 because of how `enum VARORDER` is exposed
"""The primary function :py:func:`find_embedding` is a modernized implementation 
of the Cai, Macready and Roy [1] algorithm with several new features to give users 
finer control and address a wider class of problems.

Definitions
===========

Let :math:`S` and :math:`T` be graphs, which we call source and target. If a set 
of target nodes is either size 1 or it's a connected subgraph of :math:`T`, we 
call it a `chain`.  A mapping :math:`f` from source nodes to chains is an 
embedding of :math:`S` into :math:`T` when:

* For every pair of nodes :math:`s_1 \neq s_2` of :math:`S`, the chains :math:`f(s_1)` 
  and :math:`f(s_2)` are disjoint, and

* For every source edge :math:`(s_1, s_2)`, there is at least one target edge 
  :math:`(t_1, t_2)` for which :math:`t_1 \in f(s_1)` and :math:`t_2 \in f(s_2)`.

In cases where two chains are not disjoint, we say that they `overlap`. If a 
mapping has overlapping chains, and some of its source edges are represented by 
qubits shared by their associated chains but the others are all proper, we call 
that mapping an `overlapped embedding`.

High-level Algorithm Description
================================

This is a very rough description of the heuristic more properly described in 
[1], and most accurately described by the source code.

While it is difficult to find proper embeddings, it is much easier to find 
embeddings where chains are allowed to overlap.  The key operation of this 
algorithm is a placement heuristic.  We initialize by setting 
:math:`f(s_0) = {t_0}` for chosen source and target nodes, and then proceed to 
place nodes heedless of accumulating overlaps. We persist: tear out a chain, 
clean up its neighboring chains, and replace it.  The placement heuristic 
attempts to avoid qubits involved in overlaps, and once it finds an embedding, 
continues in the same fashion with the aim of minimizing the sizes of its chains.

Placement Heuristic
-------------------

Let :math:`s` be a source node with neighbors :math:`n_1, \cdots, n_d`.  We 
first measure the distance from each neighbor's chain, :math:`f(n_i)` to all 
target nodes. First we select a target node :math:`t_0` that minimizes the sum 
of distances to those chains.  Next we follow a minimum-length path from 
:math:`t_0` to each neighbor's chain, and the union of those paths is the new 
chain for :math:`s`.  Distances are computed in :math:`T` as a node-weighted 
graph, where the weight of a node is an exponential function of the number of 
chains which use it.

Hinting and Constraining
========================

This implementation adds several useful features:

* ``initial_chains``: Initial chains are used during the initialization procedure, 
  and can be used to provide hints in the form of an overlapped, partial, or 
  otherwise faulty embedding.
* ``fixed_chains``: Fixed chains are held constant during the execution of the 
  algorithm.
* ``restrict_chains``: Chains can be restricted to being contained within a 
  user-defined subset of :math:`T` -- this constraint is somewhat soft, and the 
  algorithm can be expected to violate it.

[1] https://arxiv.org/abs/1406.2741
"""

include "_minorminer_h.pxi"
import os as _os, logging as _logging

def find_embedding(S, T, **params):
    """Heuristically attempt to find a minor-embedding of source graph S 
    into a target graph T.

    Args:
        S (iterable/NetworkX Graph): 
            The source graph as an iterable of label pairs representing the 
            edges, or a NetworkX Graph.

        T (iterable/NetworkX Graph):
            The target graph as an iterable of label pairs representing the 
            edges, or a NetworkX Graph.

        **params (optional): See below.
 
    Returns: 
        When the optional parameter ``return_overlap`` is False (the default), 
        the function returns a dict that maps labels in S to lists of labels in 
        T. If the heuristic fails to find an embedding, an empty dictionary is 
        returned.

        When ``return_overlap`` is True, the function returns a tuple consisting 
        of a dict that maps labels in S to lists of labels in T and a bool 
        indicating whether or not a valid embedding was found.

        When interrupted by Ctrl-C, the function returns the best embedding found 
        so far.

        Note that failure to return an embedding does not prove that no embedding 
        exists.

    Optional Parameters:
        max_no_improvement (int, optional, default=10):
            Maximum number of failed iterations to improve the current solution, 
            where each iteration attempts to find an embedding for each variable 
            of S such that it is adjacent to all its neighbours.

        random_seed (int, optional, default=None):
            Seed for the random number generator. If None, seed is set by 
            ``os.urandom()``.

        timeout (int, optional, default=1000):
            Algorithm gives up after timeout seconds.

        max_beta (double, optional, max_beta=None):
            Qubits are assigned weight according to a formula (beta^n)
            where n is the number of chains containing that qubit. This value
            should never be less than or equal to 1. If None, ``max_beta`` is 
            effectively infinite.

        tries (int, optional, default=10):
            Number of restart attempts before the algorithm stops. On
            D-WAVE 2000Q, a typical restart takes between 1 and 60 seconds.

        inner_rounds (int, optional, default=None):
            The algorithm takes at most this many iterations between
            restart attempts; restart attempts are typically terminated due to
            ``max_no_improvement``. If None, ``inner_rounds`` is effectively 
            infinite.

        chainlength_patience (int, optional, default=10):
            Maximum number of failed iterations to improve chain lengths in the 
            current solution, where each iteration attempts to find an embedding 
            for each variable of S such that it is adjacent to all its neighbours. 

        max_fill (int, optional, default=None):
            Restricts the number of chains that can simultaneously incorporate 
            the same qubit during the search. Values above 63 are treated as 63.
            If None, ``max_fill`` is effectively infinite.

        threads (int, optional, default=1):
            Maximum number of threads to use. Note that the parallelization is 
            only advantageous where the expected degree of variables is 
            significantly greater than the number of threads. Value must be 
            greater than 1.

        return_overlap (bool, optional, default=False):
            This function returns an embedding, regardless of whether or not
            qubits are used by multiple variables. ``return_overlap`` determines
            the function's return value. If True, a 2-tuple is returned, in which 
            the first element is the embedding and the second element is
            a bool representing the embedding validity. If False, only an
            embedding is returned.

        skip_initialization (bool, optional, default=False):
            Skip the initialization pass. Note that this only works if the chains 
            passed in through ``initial_chains`` and ``fixed_chains`` are 
            semi-valid. A semi-valid embedding is a collection of chains such 
            that every adjacent pair of variables (u,v) has a coupler (p,q) in 
            the hardware graph where p is in chain(u) and q is in chain(v). This 
            can be used on a valid embedding to immediately skip to the chain 
            length improvement phase. Another good source of semi-valid embeddings 
            is the output of this function with the ``return_overlap`` parameter 
            enabled. 

        verbose (int, optional, default=0):
            Level of output verbosity.
            
            When set to 0:
                Output is quiet until the final result. 
            
            When set to 1: 
                Output looks like this:

                .. code-block:: bash

                    initialized
                    max qubit fill 3; num maxfull qubits=3
                    embedding trial 1
                    max qubit fill 2; num maxfull qubits=21
                    embedding trial 2
                    embedding trial 3
                    embedding trial 4
                    embedding trial 5
                    embedding found.
                    max chain length 4; num max chains=1
                    reducing chain lengths
                    max chain length 3; num max chains=5

            When set to 2:
                Output the information for lower levels and also report 
                progress on minor statistics (when searching for an embedding, 
                this is when the number of maxfull qubits decreases; when 
                improving, this is when the number of max chains decreases).

            When set to 3:
                Report before each pass. Look here when tweaking ``tries``, 
                ``inner_rounds``, and ``chainlength_patience``.

            When set to 4:
                Report additional debugging information. By default, this package 
                is built without this functionality. In the C++ headers, this is 
                controlled by the ``CPPDEBUG`` flag.

            Detailed explanation of the output information:
                max qubit fill:
                    Largest number of variables represented in a qubit.
                num maxfull:
                    Number of qubits that have max overfill.
                max chain length:
                    Largest number of qubits representing a single variable.
                num max chains:
                    Number of variables that have max chain size.

        interactive (bool, optional, default=False):
            If `logging` is None or False, the verbose output will be printed
            to stdout/stderr as appropriate, and keyboard interrupts will stop 
            the embedding process and the current state will be returned to the 
            user. Otherwise, output will be directed to the logger 
            ``logging.getLogger(minorminer.__name__)`` and keyboard interrupts 
            will be propagated back to the user. Errors will use ``logger.error()``, 
            verbosity levels 1 through 3 will use ``logger.info()`` and level 4 
            will use ``logger.debug()``.

        initial_chains (dict, optional):
            Initial chains inserted into an embedding before ``fixed_chains`` are 
            placed, which occurs before the initialization pass. These can be 
            used to restart the algorithm in a similar state to a previous 
            embedding; for example, to improve chain length of a valid embedding 
            or to reduce overlap in a semi-valid embedding (see 
            ``skip_initialization``) previously returned by the algorithm. Missing
            or empty entries are ignored. Each value in the dictionary is a list
            of qubit labels.

        fixed_chains (dict, optional):
            Fixed chains inserted into an embedding before the initialization 
            pass. As the algorithm proceeds, these chains are not allowed to 
            change, and the qubits used by these chains are not used by other 
            chains. Missing or empty entries are ignored. Each value in the
            dictionary is a list of qubit labels.

        restrict_chains (dict, optional):
            Throughout the algorithm, it is guaranteed that chain[i] is a subset 
            of ``restrict_chains[i]`` for each i, except those with missing or 
            empty entries. Each value in the dictionary is a list of qubit labels.

        suspend_chains (dict, optional):
            This is a metafeature that is only implemented in the Python
            interface. ``suspend_chains[i]`` is an iterable of iterables; for 
            example, ``suspend_chains[i] = [blob_1, blob_2]``, with each `blob_j` 
            an iterable of target node labels.
            
            This enforces the following:

            .. code-block:: text

                for each suspended variable i,
                    for each blob_j in the suspension of i,
                        at least one qubit from blob_j will be contained in the chain for i

            We accomplish this through the following problem transformation
            for each iterable `blob_j` in ``suspend_chains[i]``,
                * Add an auxiliary node `Zij` to both source and target graphs
                * Set `fixed_chains[Zij]` = `[Zij]`
                * Add the edge `(i,Zij)` to the source graph
                * Add the edges `(q,Zij)` to the target graph for each `q` in `blob_j`
    """
    cdef _input_parser _in
    try:
        _in = _input_parser(S, T, params)
    except EmptySourceGraphError:
        return {}

    cdef vector[int] chain
    cdef vector[vector[int]] chains
    cdef int success = findEmbedding(_in.Sg, _in.Tg, _in.opts, chains)

    cdef int nc = chains.size()

    rchain = {}
    if chains.size():
        for v in range(nc-_in.pincount):
            chain = chains[v]
            rchain[_in.SL.label(v)] = [_in.TL.label(z) for z in chain]

    if _in.opts.return_overlap:
        return rchain, success
    else:
        return rchain

class EmptySourceGraphError(RuntimeError):
    pass

# Even though this is marked as noexcept, exceptions raised by the logger will
# still eventually be caught.
cdef void wrap_logger(void *logger, int loglevel, const string &msg) noexcept:
    if loglevel == 0:
        (<object>logger).error(msg.rstrip())
    elif 1 <= loglevel < 4:
        (<object>logger).info(msg.rstrip())
    else:
        (<object>logger).debug(msg.rstrip())

cdef class _input_parser:
    cdef input_graph Sg, Tg
    cdef labeldict SL, TL
    cdef optional_parameters opts
    cdef int pincount
    def __init__(self, S, T, params):
        cdef uint64_t *seed
        cdef object z

        names = {"max_no_improvement", "random_seed", "timeout", "tries", "verbose",
                 "fixed_chains", "initial_chains", "max_fill", "chainlength_patience",
                 "return_overlap", "skip_initialization", "inner_rounds", "threads",
                 "restrict_chains", "suspend_chains", "max_beta", "interactive"}

        for name in params:
            if name not in names:
                raise ValueError("%s is not a valid parameter for find_embedding"%name)

        z = params.get("interactive")
        if z is None or not z:
            self.opts.interactive = 0
            z = _logging.getLogger(__name__)
            self.opts.localInteractionPtr.reset(
                new LocalInteractionLogger(wrap_logger, <void *>z))
        else:
            self.opts.interactive = 1
            self.opts.localInteractionPtr.reset(new LocalInteractionPython())

        z = params.get("max_no_improvement")
        if z is not None:
            self.opts.max_no_improvement = int(z)

        z = params.get("skip_initialization")
        if z is not None:
            self.opts.skip_initialization = int(z)

        z = params.get("chainlength_patience")
        if z is not None:
            self.opts.chainlength_patience = int(z)

        z = params.get("random_seed")
        if z is not None:
            self.opts.seed( long(z) )
        else:
            seed_obj = _os.urandom(sizeof(uint64_t))
            seed = <uint64_t *>(<void *>(<uint8_t *>(seed_obj)))
            self.opts.seed(seed[0])

        z = params.get("tries")
        if z is not None:
            self.opts.tries = int(z)

        z = params.get("verbose")
        if z is not None:
            self.opts.verbose = int(z)

        z = params.get("inner_rounds")
        if z is not None:
            self.opts.inner_rounds = int(z)

        z = params.get("timeout")
        if z is not None:
            self.opts.timeout = float(z)

        z = params.get("max_beta")
        if z is not None:
            self.opts.max_beta = float(z)

        z = params.get("return_overlap")
        if z is not None:
            self.opts.return_overlap = int(z)

        z = params.get("max_fill")
        if z is not None:
            self.opts.max_fill = int(z)

        z = params.get("threads")
        if z is not None:
            self.opts.threads = int(z)

        self.SL = _read_graph(self.Sg, S)
        if not self.SL:
            raise EmptySourceGraphError

        self.TL = _read_graph(self.Tg, T)
        if not self.TL:
            raise ValueError("Cannot embed a non-empty source graph into an empty target graph.")

        _get_chainmap(params.get("fixed_chains", ()), self.opts.fixed_chains, self.SL, self.TL, "fixed_chains")
        _get_chainmap(params.get("initial_chains", ()), self.opts.initial_chains, self.SL, self.TL, "initial_chains")
        _get_chainmap(params.get("restrict_chains", ()), self.opts.restrict_chains, self.SL, self.TL, "restrict_chains")

        self.pincount = 0
        cdef int nonempty
        cdef int pinlabel
        cdef vector[int] chain
        suspend_chains = params.get("suspend_chains", ())
        if suspend_chains:
            for v, blobs in suspend_chains.items():
                for i,blob in enumerate(blobs):
                    nonempty = 0
                    pin = "__MINORMINER_INTERNAL_PIN_FOR_SUSPENSION", v, i
                    if pin in self.SL:
                        raise ValueError("node label %s is a special value used by the suspend_chains feature; please relabel your source graph"%(pin,))
                    if pin in self.TL:
                        raise ValueError("node label %s is a special value used by the suspend_chains feature; please relabel your target graph"%(pin,))

                    for q in blob:
                        if q in self.TL:
                            self.Tg.push_back(self.TL[pin], self.TL[q])
                        else:
                            raise RuntimeError("suspend_chains use target node labels that weren't referred to by any edges")
                        nonempty = 1
                    if nonempty:
                        chain.clear()
                        self.pincount += 1
                        pinlabel = <int> self.SL[pin]
                        chain.push_back(<int> self.TL[pin])
                        self.opts.fixed_chains.insert(pair[int,vector[int]](pinlabel,chain))
                        if v in self.SL:
                            self.Sg.push_back(self.SL[v], self.SL[pin])
                        else:
                            raise RuntimeError("suspend_chains use source node labels that weren't referred to by any edges")

cdef class miner:
    """
    A class for higher-level algorithms based on the heuristic embedding algorithm components.

    Args::

        S: an iterable of label pairs representing the edges in the source graph

        T: an iterable of label pairs representing the edges in the target graph

        **params (optional): see documentation of minorminer.find_embedding

    """
    cdef _input_parser _in
    cdef bool quickpassed
    cdef pathfinder_wrapper *pf
    def __cinit__(self, S, T, **params):
        try:
            self._in = _input_parser(S, T, params)
        except EmptySourceGraphError:
            raise ValueError, "The source graph has zero edges; cowardly refusing to construct a miner object for a trivial problem."
        self.quickpassed = False
        self.pf = new pathfinder_wrapper(self._in.Sg, self._in.Tg, self._in.opts)

    def __dealloc__(self):
        del self.pf

    def find_embedding(self):
        """
        Finds a single embedding, and returns it.  If the state of this object has not been changed,
        this is equivalent to calling `minorminer.find_embedding(S, T, **params)` where S, T, and
        params were all set during the construction of this object.

        Returns::

            When return_overlap = False (the default), returns a dict that maps labels in S to lists of labels in T

            When return_overlap = True, returns a tuple consisting of a dict that maps labels in S to lists of labels in T and a bool indicating whether or not a valid embedding was foun
        """
        cdef int i, success = self.pf.heuristicEmbedding()
        cdef vector[int] chain

        rchain = {}
        if self._in.opts.return_overlap or success:
            for v in range(self.pf.num_vars()-self._in.pincount):
                chain.clear()
                self.pf.get_chain(v, chain)
                rchain[self._in.SL.label(v)] = [self._in.TL.label(z) for z in chain]

        if self._in.opts.return_overlap:
            return rchain, success
        else:
            return rchain

    def quickpass(self, varorder=None, VARORDER strategy = VARORDER_RPFS, int chainlength_bound=0, int overlap_bound = 0, bool local_search=False, bool clear_first=True, double round_beta = 1e64):
        """
        Attempts to find an embedding through a very greedy strategy:

            if clear_first:
                set embedding to miner's initial_chains parameter
            else:
                recall embedding from miner's internal state

            for each node (in the variable order, if provided):
                attempt to find a new embedding of that node
                if the attempt produces a chain, record it (unless, optionaly, the chain is above the chainlength bound)

        Args::

            varorder: list (default None), a list of source graph node labels

            strategy: VARORDER (default VARORDER_RPFS), a variable ordering strategy from the enum type minorminer.VARORDER

            chainlength_bound: int (default 0), if nonzero, this is the maximum allowable chainlength

            overlap_bound: int (default 0), this is the maximum overlap count at any given qubit (zero means resulting chains will
                be properly embedded)

            local_search: bool (default False), if True, use a localized chain search and try harder to find short chains.  this
                is much faster in many cases

            clear_first: bool (default True), if True, re-initialize the embedding in the miner's internal state.
                note, on the very first call to quickpass, this parameter is ignored and set to True

        """
        cdef vector[int] neworder
        cdef vector[int] chain

        if not self.quickpassed:
            clear_first = True
            self.quickpassed = True

        if varorder is not None:
            for v in varorder:
                if v not in self._in.SL:
                   raise ValueError, "entries of the variable ordering must be source graph labels"
                else:
                    neworder.push_back(self._in.SL[v])
            self.pf.quickPass(neworder, chainlength_bound, overlap_bound, local_search, clear_first, round_beta)
        else:
            self.pf.quickPass(strategy, chainlength_bound, overlap_bound, local_search, clear_first, round_beta)

        rchain = {}
        for v in range(self.pf.num_vars()-self._in.pincount):
            chain.clear()
            self.pf.get_chain(v, chain)
            if chain.size():
                rchain[self._in.SL.label(v)] = [self._in.TL.label(z) for z in chain]
        return rchain

    def find_embeddings(self, int n, bool force = False):
        """
        Finds n embeddings, and returns them.

        Args::

            n: int, the number of embeddings to find

            force: bool, whether or not to retry failed embeddings until n successes are counted

        Returns::

            a list of embeddings (which may have overlaps if return_overlap is True)

        """

        embs = []

        while n > 0:
            emb = self.find_embedding()
            if isinstance(emb, tuple):
                emb = emb[0]
                succ = 1
            else:
                succ = len(emb)
            if succ:
                embs.append(emb)
            if succ or not force:
                n -= 1
        return embs

    def set_initial_chains(self, emb):
        """
        Update the initial chains.

        Args::
            emb: dict, the new set of chains to initialize embedding algorithms with

        """

        cdef chainmap c = chainmap()
        _get_chainmap(emb, c, self._in.SL, self._in.TL, "initial_chains")
        self.pf.set_initial_chains(c)

    def improve_embeddings(self, list embs):
        """
        For each embedding in the input,
            * update the initial_chains parameter in this object, and
            * compute a new embedding with that initialization

        Args::

            embs: list of dicts

        Returns::

            a list of embeddings with the same length as embs.  note, embeddings may have overlaps if return_overlap is True

        """
        cdef int n = len(embs)
        cdef list _embs = []
        for i in range(len(embs)):
            self.set_initial_chains(embs[i])
            emb = self.find_embedding()
            if isinstance(emb, tuple):
                emb = emb[0]
            _embs.append(emb)
        return _embs

    cdef dict count_overlaps(self, list chains):
        cdef dict o = {}
        for chain in chains:
            for q in chain:
                o[q] = 1+o.get(q,-1)
        return o

    cdef list histogram_key(self, list sizes):
        cdef dict h = {}
        cdef int s, x
        cdef tuple t
        for s in sizes:
            if s:
                h[s] = 1+h.get(s, 0)

        return [t[i] for t in sorted(h.items(), reverse = True) for i in (0,1)]

    def quality_key(self, emb, embedded = False):
        """
        Returns an object to represent the quality of an embedding.

        Example::
            >>> import networkx, minorminer
            >>> k = networkx.complete_graph(4)
            >>> g = networkx.grid_graph([2,4,4])
            >>> mm = minorminer.miner(k.edges(), g.edges())
            >>> embs = mm.find_embeddings(10)
            >>> emb = min(embs, key=mm.quality_key)

        Args::

            emb: dict, an embedding object (with or without overlaps)

            embedded: bool (default False), if this is true, we don't count overlaps
                and assume that there are none.

        """
        cdef int state = 2
        cdef dict o
        cdef list L, O
        if emb == {}:
            return (state,)

        state = 0
        L = self.histogram_key([len(c) for c in emb.values()])

        if embedded:
            O = []
        else:
            o = self.count_overlaps(list(emb.values()))
            O = self.histogram_key(list(o.values()))
            if len(O):
                state = 1

        return (state, O, L)


cdef int _get_chainmap(C, chainmap &CMap, SL, TL, parameter) except -1:
    cdef vector[int] chain
    CMap.clear();
    try:
        for a in C:
            chain.clear()
            if C[a]:
                for x in C[a]:
                    if x in TL:
                        chain.push_back(<int> TL[x])
                    else:
                        raise RuntimeError, "%s uses target node labels that weren't referred to by any edges"%parameter
                if a in SL:
                    CMap.insert(pair[int,vector[int]](SL[a], chain))
                else:
                    raise RuntimeError, "%s uses source node labels that weren't referred to by any edges"%parameter

    except (TypeError, ValueError):
        try:
            nc = next(C)
        except:
            nc = None
        if nc is None:
            raise ValueError("initial_chains and fixed_chains must be mappings (dict-like) from ints to iterables of ints; C has type %s and I can't iterate over it"%type(C))
        else:
            raise ValueError("initial_chains and fixed_chains must be mappings (dict-like) from ints to iterables of ints; C has type %s and next(C) has type %s"%(type(C), type(nc)))

cdef _read_graph(input_graph &g, E):
    cdef labeldict L = labeldict()
    cdef bool nodescan = False
    cdef int i, last
    if hasattr(E, 'edges'):
        G = E
        E = E.edges()
        nodescan = True
    for a, b in E:
        g.push_back(L[a],L[b])
    if nodescan:
        last = len(L)
        for a in G.nodes():
            L[a]
        for i in range(last, len(L)):
            g.push_back(i, i)
    return L

__all__ = ["find_embedding", "VARORDER", "miner"]

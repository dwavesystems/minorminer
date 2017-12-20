"""
minorminer is a heuristic tool for finding graph minors.

For complete details on underlying algorithm see the paper: https://arxiv.org/abs/1406.2741
"""
include "minorminer.pxi"
from random import randint


def find_embedding(Q, A, **params):
    """
    find_embedding(Q, A, **params)
    Heuristically attempt to find a minor-embedding of a graph representing an Ising/QUBO into a target graph.

    Args:

        Q(iterable[pair]): the edges in the source graph

        A(iterable[pair]): the edges in the target graph

        **params (optional): see below

    Returns:

        When return_overlap = False (the default), returns a dict that maps labels in Q to lists of labels in A.

        When return_overlap = True, returns a tuple consisting of a dict that maps labels in Q to lists of labels in A and a bool indicating whether or not a valid embedding was found

        When interrupted by Ctrl-C, returns the best embedding found so far

    Optional parameters::

        fast_embedding: True/False, tries to get an embedding quickly, without worrying about
            chain length.
            (must be a boolean, default = False)

        max_no_improvement: number of rounds of the algorithm to try from the current
            solution with no improvement. Each round consists of an attempt to find an
            embedding for each variable of Q such that it is adjacent to all its
            neighbours.
            (must be an integer >= 0, default = 10)

        random_seed: seed for random number generator that find_embedding uses
            (must be an integer >=0, default is randomly set)

        timeout: Algorithm gives up after timeout seconds.
            (must be a number >= 0, default is approximately 1000 seconds)

        tries: The algorithm stops after this number of restart attempts. On D-WAVE 2000Q,
            each restart takes between 1 and 60 seconds typically.
            (must be an integer >= 0, default = 10)

        inner_rounds: the algorithm takes at most this many passes between restart attempts;
            restart attempts are typically terminated due to max_no_improvement
            (must be an integer >= 0, default = effectively infinite)

        chainlength_patience: similar to max_no_improvement, but for the chainlength improvement
            passes.
            (must be an integer >= 0, default = 2)

        max_fill: until a valid embedding is found, this restricts the the maximum number
            of variables whose chain may contain a given qubit.
            (must be an integer >= 0, default = effectively infinite)

        threads: maximum number of threads to use.  note that the parallelization is only
            advantageous where the expected degree of variables is (significantly?)
            greater than the number of threads.
            (must be an integer >= 1, default = 1)

        return_overlap: return an embedding whether or not qubits are used by multiple
            variables -- capture both return values to determine whether or
            not the returned embedding is valid
            (must be a logical 0/1 integer, default = 0)

        skip_initialization: skip the initialization pass -- NOTE: this only works
            if the chains passed in through initial_chains and fixed_chains are
            semi-valid.  A semi-valid embedding is a collection of chains
            such that every adjacent pair of variables (u,v) has a coupler
            (p,q) in the hardware graph where p is in chain(u) and q is in
            chain(v).  This can be used on a valid embedding to immediately
            skip to the chainlength improvement phase.  Another good source
            of semi-valid embeddings is the output of this function with
            the return_overlap parameter enabled.
            (must be a logical 0/1 integer, default = 0)

        verbose: 0/1/2/3/(4).
            (must be an integer [0, 1, 2, 3, (4)], default = 0)
            when verbose is 1, the output information will look like:

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

            additional verbosity levels will
                2: report progress on minor statistics (when searching for an embedding, this is
                    when the number of maxfull qubits decreases; when improving, this is when the
                    number of max chains decreases)
                3: report before each before each pass -- look here when tweaking `tries`,
                    `inner_rounds`, and `chainlength_patience`
                (4): report additional debugging information.  by default, this package is built
                    without this functionality -- in the c++ headers, this is controlled by the
                    CPPDEBUG flag

            detailed explanation of the output information:
                max qubit fill: largest number of variables represented in a qubit
                num maxfull: the number of qubits that has max overfill
                max chain length: largest number of qubits representing a single variable
                num max chains: the number of variables that has max chain size

        initial_chains: A dictionary, where initial_chains[i] is a list of qubit labels.
            These chains are inserted into an embedding before fixed_chains are
            placed, which occurs before the initialization pass.  This can be
            used to restart the algorithm in a similar state to a previous
            embedding, for example to improve chainlength of a valid embedding
            or to reduce overlap in a semi-valid embedding (see skip_initialization)
            previously returned by the algorithm. Missing or empty entries are ignored.

        fixed_chains: A dictionary, where fixed_chains[i] is a list of qubit labels.  These
            chains are inserted into an embedding before the initialization pass.
            As the algorithm proceeds, these chains are not allowed to change. Missing
            or empty entries are ignored.


        restrict_chains: A dictionary, where restrict_chains[i] is a list of qubit labels.
            Throughout the algorithm, we maintain the condition that chain[i] is a
            subset of restrict_chains[i] for each i -- except those with missing or
            empty entries.
    """


    cdef vector[int] chain

    cdef optional_parameters opts
    opts.localInteractionPtr.reset(new LocalInteractionPython())

    names = {"fast_embedding", "max_no_improvement", "random_seed", "timeout",
             "tries", "verbose", "fixed_chains", "initial_chains", "max_fill",
             "chainlength_patience", "return_overlap", "skip_initialization",
             "inner_rounds", "threads", "restrict_chains"}

    for name in params:
        if name not in names:
            raise ValueError("%s is not a valid parameter for find_embedding"%name)

    try: opts.fast_embedding = int( params["fast_embedding"] )
    except KeyError: pass

    try: opts.max_no_improvement = int( params["max_no_improvement"] )
    except KeyError: pass

    try: opts.skip_initialization = int( params["skip_initialization"] )
    except KeyError: pass

    try: opts.chainlength_patience = int( params["chainlength_patience"] )
    except KeyError: pass

    try: opts.seed( int(params["random_seed"]) )
    except KeyError: opts.seed( randint(0,1<<30) )

    try: opts.tries = int(params["tries"])
    except KeyError: pass

    try: opts.verbose = int(params["verbose"])
    except KeyError: pass

    try: opts.inner_rounds = int(params["inner_rounds"])
    except KeyError: pass

    try: opts.timeout = float(params["timeout"])
    except KeyError: pass

    try: opts.return_overlap = int(params["return_overlap"])
    except KeyError: pass

    try: opts.max_fill = int(params["max_fill"])
    except KeyError: pass

    try: opts.threads = int(params["threads"])
    except KeyError: pass

    cdef input_graph Qg
    cdef input_graph Ag

    cdef labeldict QL = _read_graph(Qg,Q)
    cdef labeldict AL = _read_graph(Ag,A)

    cdef int checksize = len(QL)+len(AL)

    _get_chainmap(params.get("fixed_chains",[]), opts.fixed_chains, QL, AL)
    if checksize < len(QL)+len(AL):
        raise RuntimeError("fixed_chains use source or target node labels that weren't referred to by any edges")
    _get_chainmap(params.get("initial_chains",[]), opts.initial_chains, QL, AL)
    if checksize < len(QL)+len(AL):
        raise RuntimeError("initial_chains use source or target node labels that weren't referred to by any edges")
    _get_chainmap(params.get("restrict_chains",[]), opts.restrict_chains, QL, AL)
    if checksize < len(QL)+len(AL):
        raise RuntimeError("restrict_chains use source or target node labels that weren't referred to by any edges")

    cdef vector[vector[int]] chains
    cdef int success = findEmbedding(Qg, Ag, opts, chains)

    cdef int nc = chains.size()

    rchain = {}
    for v in range(nc):
        chain = chains[v]
        rchain[QL.label(v)] = [AL.label(z) for z in chain]

    if opts.return_overlap:
        return rchain, success
    else:
        return rchain

cdef int _get_chainmap(C, chainmap &CMap, QL, AL) except -1:
    cdef vector[int] chain
    CMap.clear();
    try:
        for a in C:
            chain.clear()
            if C[a]:
                if AL is None:
                    for x in C[a]:
                        chain.push_back(<int> x)
                else:
                    for x in C[a]:
                        chain.push_back(<int> AL[x])
                if QL is None:
                    CMap.insert(pair[int,vector[int]](a, chain))
                else:
                    CMap.insert(pair[int,vector[int]](QL[a], chain))

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
    L = labeldict()
    for a,b in E:
        g.push_back(L[a],L[b])
    return L

__all__ = ["find_embedding"]

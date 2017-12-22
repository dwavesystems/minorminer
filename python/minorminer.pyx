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

        Q: an iterable of label pairs representing the edges in the source graph

        A: an iterable of label pairs representing the edges in the target graph

        **params (optional): see below

    Returns:

        When return_overlap = False (the default), returns a dict that maps labels in Q to lists of labels in A

        When return_overlap = True, returns a tuple consisting of a dict that maps labels in Q to lists of labels in A and a bool indicating whether or not a valid embedding was foun

        When interrupted by Ctrl-C, returns the best embedding found so far

        Note that failure to return an embedding does not prove that no embedding exists

    Optional parameters::

        max_no_improvement: Maximum number of failed iterations to improve the
            current solution, where each iteration attempts to find an embedding
            for each variable of Q such that it is adjacent to all its
            neighbours. Integer >= 0 (default = 10)

        random_seed: Seed for the random number generator that find_embedding
            uses. Integer >=0 (default is randomly set)

        timeout: Algorithm gives up after timeout seconds. Number >= 0 (default
            is approximately 1000 seconds)

        tries: Number of restart attempts before the algorithm stops. On
            D-WAVE 2000Q, a typical restart takes between 1 and 60 seconds.
            Integer >= 0 (default = 10)

        inner_rounds: the algorithm takes at most this many iterations between
            restart attempts; restart attempts are typically terminated due to
            max_no_improvement. Integer >= 0 (default = effectively infinite)

        chainlength_patience: Maximum number of failed iterations to improve
            chainlengths in the current solution, where each iteration attempts
            to find an embedding for each variable of Q such that it is adjacent
            to all its neighbours. Integer >= 0 (default = 10)

        max_fill: Restricts the number of chains that can simultaneously
            incorporate the same qubit during the search. Integer >= 0 (default
            = effectively infinite)

        threads: Maximum number of threads to use. Note that the
            parallelization is only advantageous where the expected degree of
            variables is significantly greater than the number of threads.
            Integer >= 1 (default = 1)

        return_overlap: This function returns an embedding whether or not qubits
            are used by multiple variables. Set this value to 1 to capture both
            return values to determine whether or not the returned embedding is
            valid. Logical 0/1 integer (default = 0)

        skip_initialization: Skip the initialization pass. Note that this only
            works if the chains passed in through initial_chains and
            fixed_chains are semi-valid. A semi-valid embedding is a collection
            of chains such that every adjacent pair of variables (u,v) has a
            coupler (p,q) in the hardware graph where p is in chain(u) and q is
            in chain(v). This can be used on a valid embedding to immediately
            skip to the chainlength improvement phase. Another good source of
            semi-valid embeddings is the output of this function with the
            return_overlap parameter enabled. Logical 0/1 integer (default = 0)

        verbose: Level of output verbosity. Integer < 4 (default = 0).
            When set to 0, the output is quiet until the final result.
            When set to 1, output looks like this:

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

            When set to 2, outputs the information for lower levels and also
                reports progress on minor statistics (when searching for an
                embedding, this is when the number of maxfull qubits decreases;
                when improving, this is when the number of max chains decreases)
            When set to 3, report before each before each pass. Look here when
                tweaking `tries`, `inner_rounds`, and `chainlength_patience`
            When set to 4, report additional debugging information. By default,
                this package is built without this functionality. In the c++
                headers, this is controlled by the CPPDEBUG flag
            Detailed explanation of the output information:
                max qubit fill: largest number of variables represented in a qubit
                num maxfull: the number of qubits that has max overfill
                max chain length: largest number of qubits representing a single variable
                num max chains: the number of variables that has max chain size

        initial_chains: Initial chains inserted into an embedding before
            fixed_chains are placed, which occurs before the initialization
            pass. These can be used to restart the algorithm in a similar state
            to a previous embedding; for example, to improve chainlength of a
            valid embedding or to reduce overlap in a semi-valid embedding (see
            skip_initialization) previously returned by the algorithm. Missing
            or empty entries are ignored. A dictionary, where initial_chains[i]
            is a list of qubit labels.

        fixed_chains: Fixed chains inserted into an embedding before the
            initialization pass. As the algorithm proceeds, these chains are not
            allowed to change. Missing or empty entries are ignored. A
            dictionary, where fixed_chains[i] is a list of qubit labels.

        restrict_chains: Throughout the algorithm, we maintain the condition
            that chain[i] is a subset of restrict_chains[i] for each i, except
            those with missing or empty entries. A dictionary, where
            restrict_chains[i] is a list of qubit labels.
    """


    cdef vector[int] chain

    cdef optional_parameters opts
    opts.localInteractionPtr.reset(new LocalInteractionPython())

    names = {"max_no_improvement", "random_seed", "timeout", "tries", "verbose",
             "fixed_chains", "initial_chains", "max_fill", "chainlength_patience",
             "return_overlap", "skip_initialization", "inner_rounds", "threads",
             "restrict_chains"}

    for name in params:
        if name not in names:
            raise ValueError("%s is not a valid parameter for find_embedding"%name)

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

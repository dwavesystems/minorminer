import random
from collections import defaultdict

import minorminer as mm


def initial(S, T, chains, mm_kwargs, **kwargs):
    """
    Calls minorminer.find_embedding() using chains as initial_chains.
    """
    return mm.find_embedding(S, T, initial_chains=chains, **mm_kwargs)


def suspend(S, T, chains, mm_kwargs, **kwargs):
    """
    Calls minorminer.find_embedding() using chains as suspend_chains.
    """
    return mm.find_embedding(S, T, suspend_chains={v: [C] for v, C in chains.items()}, **mm_kwargs)


def random_remove(S, T, chains, mm_kwargs, percent=2/3, **kwargs):
    """
    Calls minorminer.find_embedding() using chains as initial_chains.
    """
    for v, C in chains.items():
        chain_list = list(C)  # In case C is a set/frozenset or something

        # Shuffle and remove some
        random.shuffle(C)
        for _ in range(int(len(C)*percent)):
            chain_list.pop()

        # Update the chains
        chains[v] = chain_list

    return mm.find_embedding(S, T, initial_chains=chains, **mm_kwargs)


def extend_chains(S, T, initial_chains):
    """
    Extend chains in T so that their structure matches that of S. That is, form an overlap embedding of S in T
    where the initial_chains are subsets of the overlap embedding chains. This is done via minorminer.

    Parameters
    ----------
    S : NetworkX graph
        The graph you are embedding (source).
    T : NetworkX graph
        The graph you are embedding into (target).
    initial_chains : dict
        A mapping from vertices of S (keys) to chains of T (values).

    Returns
    -------
    extended_chains: dict
        A mapping from vertices of S (keys) to chains of T (values).
    """
    # Extend the chains to minimal overlapped embedding
    miner = mm.miner(S, T, initial_chains=initial_chains)
    extended_chains = defaultdict(set)
    for u in S:
        # Revert to the initial_chains and compute the embedding where you tear-up u.
        emb = miner.quickpass(
            [u], clear_first=True, overlap_bound=S.number_of_nodes()
        )

        # Add the new chain for u and the singleton one too (in case it got moved in emb)
        extended_chains[u].update(set(emb[u]).union(initial_chains[u]))

        # For each neighbor v of u, grow u to reach v
        for v in S[u]:
            extended_chains[u].update(
                set(emb[v]).difference(initial_chains[v]))

    return extended_chains

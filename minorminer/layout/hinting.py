import random

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

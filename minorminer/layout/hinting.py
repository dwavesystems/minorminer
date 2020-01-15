import minorminer as mm


def initial(S, T, chains, kwargs):
    """
    Calls minorminer.find_embedding() using chains as initial_chains.
    """
    return mm.find_embedding(S, T, initial_chains=chains, **kwargs)


def suspend(S, T, chains, kwargs):
    """
    Calls minorminer.find_embedding() using chains as suspend_chains.
    """
    return mm.find_embedding(S, T, suspend_chains={v: [C] for v, C in chains.items()}, **kwargs)

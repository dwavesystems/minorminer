import minorminer as mm


def initial(S, T, chains, kwargs):
    return mm.find_embedding(S, T, initial_chains=chains, **kwargs)


def suspend(S, T, chains, kwargs):
    return mm.find_embedding(S, T, suspend_chains={v: [C] for v, C in chains.items()}, **kwargs)

import minorminer as mm


def initial(S, T, chains, kwargs):
    """
    Calls minorminer.find_embedding on S and T, passing in chains as initial_chains, see minorminer.find_embedding.

    Parameters
    ----------
    S : NetworkX graph
        The graph you are embedding (source).
    T : NetworkX graph
        The graph you are embedding into (target).
    chains : dict
        A mapping from vertices of S (keys) to chains of T (values).
    kwargs : dict
        Keyword arguments for minorminer.find_embedding.

    Returns
    -------
    emb : dict
        The output from minorminer.find_embedding.
    """
    return mm.find_embedding(S, T, initial_chains=chains, **kwargs)


def suspend(S, T, chains, kwargs):
    """
    Calls minorminer.find_embedding on S and T, passing in chains as suspend_chains, see minorminer.find_embedding.

    Parameters
    ----------
    S : NetworkX graph
        The graph you are embedding (source).
    T : NetworkX graph
        The graph you are embedding into (target).
    chains : dict
        A mapping from vertices of S (keys) to chains of T (values).
    kwargs : dict
        Keyword arguments for minorminer.find_embedding.

    Returns
    -------
    emb : dict
        The output from minorminer.find_embedding.
    """
    return mm.find_embedding(S, T, suspend_chains={v: [C] for v, C in chains.items()}, **kwargs)

import networkx as nx
import numpy as np

import dwave_networkx as dnx


def kamada_kawai(G, d=2, **kwargs):
    """
    The d-dimensional Kamada-Kawai spring layout.

    Parameters
    ----------
    G : NetworkX graph
        The graph you want to compute the layout for.
    d : int (default 2)
        The dimension of the kamada_kawai layout.
    kwargs : dict
        Keyword arguments are passed to nx.kamada_kawai_layout().

    Returns
    -------
    layout : dict
        A mapping from vertices of G (keys) to points in [-1, 1]^d (values).
    """
    # NetworkX has a bug #3658.
    # Once fixed, these can collapse and `dim=n` can be part of `**kwargs`.

    # Extract the seed in case its there for random fixing
    random_kwargs = {}
    if "seed" in kwargs:
        random_kwargs["seed"] = kwargs["seed"]
        del kwargs["seed"]

    if d in (1, 2):
        return nx.kamada_kawai_layout(G, dim=d, **kwargs)
    else:
        # The random_layout is in [0, 1]^d
        random_layout = nx.random_layout(G, dim=d, **random_kwargs)
        # Convert it to [-1, 1]^d
        std_random_layout = {v: list(map(lambda x: 2*x - 1, p))
                             for v, p in random_layout.items()}

        return nx.kamada_kawai_layout(
            G, pos=std_random_layout, dim=d, **kwargs)


def chimera(G, d=2, **kwargs):
    """
    The d-dimensional Chimera layout adjusted so that it fills [-1, 1]^2 instead of [0, 1] x [0, -1]. As per the 
    implementation of dnx.chimera_layout() coordinates beyond the second, in layouts with d > 2, are 0.

    Parameters
    ----------
    G : NetworkX graph
        A Chimera graph you want to compute the layout for.
    d : int (default 2)
        The dimension of the layout.
    kwargs : dict
        Keyword arguments are passed to dnx.chimera_layout().

    Returns
    -------
    layout : dict
        A mapping from vertices of G (keys) to points in [-1, 1]^d (values).
    """
    return dnx.chimera_layout(
        G, dim=d, center=(-1, 1), scale=2, **kwargs)

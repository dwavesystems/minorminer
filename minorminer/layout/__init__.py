from construction_algs import *
from hinting_algs import *
from layout_algs import *
from placement_algs import *


def find_embedding(S, T, layout=kamada_kawai, placement=closest, construction=singleton, hinting=initial, **kwargs):
    """
    Tries to embed S in T by computing layout aware initial_chains and passing them to minorminer.

    Parameters
    ----------
    S : NetworkX graph
        The graph you are embedding (source).
    T : NetworkX graph
        The graph you are embedding into (target).
    layout : function or (function, function) (default kamada_kawai)
        The layout algorithm to call. If a single function, the same layout algorithm is called on both S and T. If a
        tuple of functions, the 1st function is called on S and the 2nd is called on T.
    placement : str (default "closest")
        The placement algorithm to call.
    construction : str (default "singleton")
        The chain construction algorithm to call.
    kwargs : dict 
        Keyword arguments are passed to minorminer.find_embedding()

    Returns
    -------
    emb : dict
        Output is dependant upon kwargs passed to minonminer, but more or less emb is a mapping from vertices of 
        S (keys) to chains in T (values).
    """
    # Parse the layout parameter
    if isinstance(layout, tuple):
        S_layout = layout[0](S)
        T_layout = layout[1](T)
    else:
        S_layout, T_layout = layout(S), layout(T)

    # Compute the placement
    vertex_map = placement(S_layout, T_layout)

    # Create the chains
    if construction is construction_algs.singleton:
        chains = construction(placement)
    elif construction is construction_algs.neighborhood:
        chains = construction(T, placement, kwargs.get("second"))
    elif construction is construction_algs.extend:
        chains = construction(S, T, placement, kwargs.get("nhbd_ext"))

    return mm.find_embedding(S, T, initial_chains=chains, **kwargs)

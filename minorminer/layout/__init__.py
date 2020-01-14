from minorminer.layout.construction import *
from minorminer.layout.hinting import *
from minorminer.layout.layout import *
from minorminer.layout.placement import *


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
        Keyword arguments are passed to various functions.

    Returns
    -------
    emb : dict
        Output is dependant upon kwargs passed to minonminer, but more or less emb is a mapping from vertices of 
        S (keys) to chains in T (values).
    """
    # Parse kwargs
    d, second, nhbd_ext = parse_kwargs(kwargs)

    # Parse the layout parameter
    if isinstance(layout, tuple):
        S_layout = layout[0](S, d)
        T_layout = layout[1](T, d)
    else:
        S_layout, T_layout = layout(S, d), layout(T, d)

    # Compute the placement
    vertex_map = placement(S_layout, T_layout)

    # Create the chains
    if construction is singleton:
        chains = construction(vertex_map)
    elif construction is neighborhood:
        chains = construction(T, vertex_map, kwargs.get("second", False))
    elif construction is extend:
        chains = construction(S, T, vertex_map, kwargs.get("nhbd_ext", False))

    return hinting(S, T, chains, kwargs)


def parse_kwargs(kwargs):
    d = kwargs.get("d", 2)
    try:
        del kwargs["d"]
    except KeyError:
        pass

    second = kwargs.get("second", False)
    try:
        del kwargs["second"]
    except KeyError:
        pass

    nhbd_ext = kwargs.get("nhbd_ext", False)
    try:
        del kwargs["nhbd_ext"]
    except KeyError:
        pass

    return d, second, nhbd_ext

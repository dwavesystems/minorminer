from minorminer.layout.construction import *
from minorminer.layout.hinting import *
from minorminer.layout.layout import *
from minorminer.layout.placement import *


def find_embedding(S, T, layout=kamada_kawai, placement=closest, construction=singleton, hinting=initial, **kwargs):
    """
    Tries to embed S in T by computing layout aware chains and passing them to minorminer, see 
    minorminer.find_embedding for additional keyword arguments.

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
        Output is dependant upon kwargs passed to minorminer, but more or less emb is a mapping from vertices of 
        S (keys) to chains in T (values).
    """
    # Parse kwargs
    layout_kwargs, construction_kwargs = _parse_kwargs(kwargs)

    # Parse the layout parameter
    if isinstance(layout, tuple):
        S_layout = layout[0](S, **layout_kwargs)
        T_layout = layout[1](T, **layout_kwargs)
    else:
        S_layout = layout(S, **layout_kwargs)
        T_layout = layout(T, **layout_kwargs)

    # Compute the placement
    vertex_map = placement(S_layout, T_layout)

    # Create the chains
    chains = construction(S, T, vertex_map, **construction_kwargs)

    return hinting(S, T, chains, kwargs)


def _parse_kwargs(kwargs):
    """
    Pull out kwargs for layout and construction functions. Leave the remaining ones for minorminer.
    """
    layout_kwargs = {}
    if "d" in kwargs:
        layout_kwargs["d"] = kwargs["d"]
        del kwargs["d"]

    construction_kwargs = {}
    if "second" in kwargs:
        construction_kwargs["second"] = kwargs["second"]
        del kwargs["second"]

    if "extend" in kwargs:
        construction_kwargs["extend"] = kwargs["extend"]
        del kwargs["extend"]

    return layout_kwargs, construction_kwargs

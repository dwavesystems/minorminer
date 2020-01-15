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
    layout : function or (function/dict, function/dict) (default kamada_kawai)
        Will either compute a layout or pass along precomputed layouts. If it's a single object, it has to be a function
        and it applies to both S and T. If it's a tuple of objects the first applies to S and the second applies to T.
    placement : function (default closest)
        The placement algorithm to call.
    construction : function (default singleton)
        The chain construction algorithm to call.
    hinting : function (default initial)
        The type of minorminer hinting to call.
    kwargs : dict 
        Keyword arguments are passed to various functions.

    Returns
    -------
    emb : dict
        Output is dependant upon kwargs passed to minonminer, but more or less emb is a mapping from vertices of 
        S (keys) to chains in T (values).
    """
    # Parse kwargs
    layout_kwargs, construction_kwargs = parse_kwargs(kwargs)

    # Parse the layout parameter
    # It's two things, one for S and one for T
    if isinstance(layout, tuple):
        # It's a layout for S
        if isinstance(layout[0], dict):
            S_layout = layout[0]
        # It's a function for S
        else:
            S_layout = layout[0](S, **layout_kwargs)

        # It's a layout for T
        if isinstance(layout[1], dict):
            T_layout = layout[1]
        # It's a function for T
        else:
            T_layout = layout[1](T, **layout_kwargs)

    # It's a layout function to compute
    else:
        S_layout, T_layout = layout(
            S, **layout_kwargs), layout(T, **layout_kwargs)

    # Compute the placement
    vertex_map = placement(S_layout, T_layout)

    # Create the chains
    chains = construction(S, T, vertex_map, **construction_kwargs)

    return hinting(S, T, chains, kwargs)


def parse_kwargs(kwargs):
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

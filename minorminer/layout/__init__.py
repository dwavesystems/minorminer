import networkx as nx

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
    S : NetworkX graph or edges data structure (dict, list, ...)
        The graph you are embedding (source) or a NetworkX supported data structure (see to_networkx_graph()).
    T : NetworkX graph or edges data structure (dict, list, ...)
        The graph you are embedding into (target) or a NetworkX supported data structure (see to_networkx_graph()).
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
        Output is dependant upon kwargs passed to minorminer, but more or less emb is a mapping from vertices of 
        S (keys) to chains in T (values).
    """
    # Parse graphs
    S, T = _parse_graphs(S, T)

    # Parse kwargs
    layout_kwargs, construction_kwargs = _parse_kwargs(kwargs)

    # Parse the layout parameter
    S_layout, T_layout = _parse_layout_parameter(S, T, layout, layout_kwargs)

    # Compute the placement
    vertex_map = placement(S_layout, T_layout)

    # Create the chains
    chains = construction(S, T, vertex_map, **construction_kwargs)

    return hinting(S, T, chains, kwargs)


def _parse_graphs(S, T):
    """
    Determines if a graph object or a collection of edges was passed in. Returns NetworkX graphs.
    """
    if hasattr(S, "edges"):
        H = S
    else:
        H = nx.Graph(S)

    if hasattr(T, "edges"):
        G = T
    else:
        G = nx.Graph(T)

    return H, G


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


def _parse_layout_parameter(S, T, layout, layout_kwargs):
    """
    Determine what combination of tuple, dict, and function the layout parameter is.
    """
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

    return S_layout, T_layout

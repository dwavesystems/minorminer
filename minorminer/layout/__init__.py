import networkx as nx

from minorminer.layout.construction import neighborhood, pass_along, singleton
from minorminer.layout.hinting import initial, suspend
from minorminer.layout.layout import Layout, chimera, kamada_kawai
from minorminer.layout.placement import binning, closest, injective


def find_embedding(
    S, T, layout=None, placement=closest, construction=pass_along, hinting=initial, **kwargs
):
    """
    Tries to embed S in T by computing layout-aware chains and passing them to minorminer.find_embedding. Chains are 
    passed as either initial_chains or suspend_chains (see documentation for minorminer.find_embedding to learn more).

    Parameters
    ----------
    S : NetworkX graph or edges data structure (dict, list, ...)
        The graph you are embedding (source) or a NetworkX supported data structure (see to_networkx_graph()).
    T : NetworkX graph or edges data structure (dict, list, ...)
        The graph you are embedding into (target) or a NetworkX supported data structure (see to_networkx_graph()).
    layout : function or [function/dict/Layout, function/dict/Layout] (default kamada_kawai)
        Specifies either a single function to compute the layout for both S and T or a 2-tuple consisting of functions 
        or pre-computed layouts, the first applying to S and the second applying to T.
    placement : function (default closest)
        The placement algorithm to call; each algorithm uses the layouts of S and T to map the vertices of S to the 
        vertices of T.
    construction : function (default pass_along)
        The chain construction algorithm to call; each algorithm uses the placement to build chains to hand to 
        minorminer.find_embedding(). 
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
    layout_kwargs, placement_kwargs, construction_kwargs = _parse_kwargs(
        kwargs)

    # Parse the layout parameter
    S_layout, T_layout = _parse_layout_parameter(S, T, layout, layout_kwargs)

    # Compute the placement
    vertex_map = placement(S_layout, T_layout, **placement_kwargs)

    # Create the chains
    chains = construction(S, T, vertex_map, **construction_kwargs)

    # Run minerminor.find_embedding()
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
    Extract kwargs for layout and construction functions. Leave the remaining ones for minorminer.find_embedding().
    """
    layout_kwargs = {}
    if "d" in kwargs:
        layout_kwargs["d"] = kwargs.pop("d")
    if "seed" in kwargs:
        layout_kwargs["seed"] = kwargs.pop("seed")
    if "center" in kwargs:
        layout_kwargs["center"] = kwargs.pop("center")
    if "scale" in kwargs:
        layout_kwargs["scale"] = kwargs.pop("scale")

    placement_kwargs = {}
    if "max_subset_size" in kwargs:
        placement_kwargs["max_subset_size"] = kwargs.pop("max_subset_size")
    if "bins" in kwargs:
        placement_kwargs["bins"] = kwargs.pop("bins")
    if "strategy" in kwargs:
        placement_kwargs["strategy"] = kwargs.pop("strategy")

    construction_kwargs = {}
    if "second" in kwargs:
        construction_kwargs["second"] = kwargs.pop("second")

    return layout_kwargs, placement_kwargs, construction_kwargs


def _parse_layout_parameter(S, T, layout, layout_kwargs):
    """
    Determine what combination of tuple, dict, and function the layout parameter is.
    """
    # It's two things, one for S and one for T
    if isinstance(layout, tuple):
        # It's a dict layout for S
        if isinstance(layout[0], dict):
            S_layout = Layout(S, layout=layout[0], **layout_kwargs)
        # It's a Layout object for S
        elif isinstance(layout[0], Layout):
            S_layout = layout[0]
        # It's a function for S
        else:
            S_layout = layout[0](S, **layout_kwargs)

        # It's a layout for T
        if isinstance(layout[1], dict):
            T_layout = Layout(T, layout=layout[1], **layout_kwargs)
        # It's a Layout object for T
        elif isinstance(layout[1], Layout):
            T_layout = layout[1]
        # It's a function for T
        else:
            T_layout = layout[1](T, **layout_kwargs)

    # It's a function for both
    else:
        S_layout = layout(S, **layout_kwargs)
        T_layout = layout(T, **layout_kwargs)

    return S_layout, T_layout

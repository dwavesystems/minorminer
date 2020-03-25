import networkx as nx

from minorminer.layout.construction import neighborhood, pass_along
from minorminer.layout.hinting import initial, suspend
from minorminer.layout.layout import Layout, dnx_layout, kamada_kawai, pca
from minorminer.layout.placement import binning, closest, injective


def find_embedding(
    S, T, layout=None, placement=closest, construction=pass_along, hinting=initial, return_layout=False, **kwargs
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
    placement : function or dict (default closest)
        If a function, it is the placement algorithm to call; each algorithm uses the layouts of S and T to map the 
        vertices of S to subsets of vertices of T. If it is a dict, it should be a map from the vertices of S to subsets of
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
    vertex_map = _parse_placement_parameter(
        S_layout, T_layout, placement, placement_kwargs)

    # Create the chains
    chains = construction(S, T, vertex_map, **construction_kwargs)

    # Run minerminor.find_embedding()
    if return_layout:
        return hinting(S, T, chains, kwargs), (S_layout, T_layout)
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
    if "rescale" in kwargs:
        layout_kwargs["rescale"] = kwargs.pop("rescale")
    if "rotate" in kwargs:
        layout_kwargs["rotate"] = kwargs.pop("rotate")

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
    Determine what combination of iterable, dict, and function the layout parameter is.
    """
    if nx.utils.iterable(layout):
        try:
            s_layout, t_layout = layout
        except ValueError:
            raise ValueError(
                "layout is expected to be a function or a length-2 iterable")
    else:
        s_layout = t_layout = layout

    # Get a Layout object for S
    if isinstance(s_layout, Layout):
        S_layout = s_layout
    elif callable(s_layout):
        S_layout = s_layout(S, **layout_kwargs)
    else:
        # assumes s_layout implements a mapping interface
        S_layout = Layout(S, layout=s_layout, **layout_kwargs)

    # Get the Layout object for T
    if isinstance(t_layout, Layout):
        T_layout = t_layout
    elif callable(t_layout):
        T_layout = t_layout(T, **layout_kwargs)
    else:
        # assumes t_layout implements a mapping interface
        T_layout = Layout(T, layout=t_layout, **layout_kwargs)

    return S_layout, T_layout


def _parse_placement_parameter(S_layout, T_layout, placement, placement_kwargs):
    """
    Determine if placement is a function or a dict.
    """
    # It's a preprocessed placement
    if isinstance(placement, dict):
        return placement

    # It's a function
    else:
        return placement(S_layout, T_layout, **placement_kwargs)

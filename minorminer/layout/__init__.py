import time

import networkx as nx
from minorminer.layout.construction import crosses, neighborhood, pass_along
from minorminer.layout.hinting import initial, suspend
from minorminer.layout.layout import (Layout, R2xT, dnx_layout, kamada_kawai,
                                      p_norm, pca)
from minorminer.layout.placement import (binning, closest, injective,
                                         intersection)


def find_embedding(
    S,
    T,
    layout=p_norm,
    placement=intersection,
    construction=crosses,
    hinting=initial,
    return_layout=False,
    **kwargs
):
    """
    Tries to embed S in T by computing layout-aware chains and passing them to minorminer.find_embedding(). Chains are 
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
        vertices of S to subsets of vertices of T. If it is a dict, it should be a map from the vertices of S to subsets
        of vertices of T.
    construction : function (default pass_along)
        The chain construction algorithm to call; each algorithm uses the placement to build chains to hand to 
        minorminer.find_embedding(). 
    hinting : function (default initial)
        The type of minorminer hinting to call.
    return_layout : bool (default False)
        Will return the layout objects of S and T.
    kwargs : dict 
        Keyword arguments are passed to various functions.

    Returns
    -------
    emb : dict
        Output is dependent upon kwargs passed to minonminer, but more or less emb is a mapping from vertices of 
        S (keys) to chains in T (values).
    """
    start = time.process_time()

    # Parse kwargs
    layout_kwargs, placement_kwargs, construction_kwargs = parse_kwargs(kwargs)

    # Parse layout parameter
    S_layout, T_layout = parse_layout_parameter(S, T, layout, layout_kwargs)

    # Compute the placement
    vertex_map = parse_placement_parameter(
        S_layout, T_layout, placement, placement_kwargs)

    # Create the chains
    chains = construction(vertex_map, S_layout=S_layout, T_layout=T_layout,
                          **construction_kwargs)

    end = time.process_time()
    timeout = kwargs.get("timeout")
    if timeout:
        kwargs["timeout"] = timeout - (end - start)

    # Run minerminor.find_embedding()
    if return_layout:
        return hinting(S, T, chains, kwargs), (S_layout, T_layout)
    return hinting(S, T, chains, kwargs)


def parse_kwargs(kwargs):
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
    if "extend" in kwargs:
        construction_kwargs["extend"] = kwargs.pop("extend")

    return layout_kwargs, placement_kwargs, construction_kwargs


def parse_layout_parameter(S, T, layout, layout_kwargs):
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


def parse_placement_parameter(S_layout, T_layout, placement, placement_kwargs):
    """
    Determine if placement is a function or a dict.
    """
    # It's a preprocessed placement
    if isinstance(placement, dict):
        return placement

    # It's a function
    else:
        return placement(S_layout, T_layout, **placement_kwargs)

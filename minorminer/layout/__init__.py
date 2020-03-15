import time

import minorminer as mm
import networkx as nx

from .connection import crosses, shortest_paths
from .contraction import random_remove
from .expansion import neighborhood
from .layout import Layout, dnx_layout, p_norm, pca
from .placement import binning, closest, injective, intersection
from .utils import placement_utils


def find_embedding(
    S,
    T,
    layout=p_norm,
    placement=intersection,
    connection=crosses,
    expansion=None,
    contraction=None,
    mm_hint_type="initial_chains",
    return_layouts=False,
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
    layout : function or [function/dict/Layout, function/dict/Layout] (default p_norm)
        Specifies either a single function to compute the layout for both S and T or a 2-tuple. The 2-tuple either 
        consists of a pair of functions or pre-computed layouts, the first entry in the 2-tuple applies to S while
        the second applies to T. 
        Note: If layout is a single function and T is a dnx_graph, then the function passed in is only applied to S
        and the dnx_layout is applied to T. To run a layout function explicitly on T, pass it in as a 2-tuple; i.e.
        (p_norm, p_norm).
    placement : function or dict (default intersection)
        If a function, it is the placement algorithm to call; each algorithm uses the layouts of S and T to map the 
        vertices of S to subsets of vertices of T. If it is a dict, it should be a map from the vertices of S to subsets
        of vertices of T.
    connection : function (default crosses)
        This is optional. To skip a connection phase, pass in None. Each chain connection algorithm that is called 
        takes the placement and produces an overlap embedding of S in T.
    expansion : function (default None)
        This is optional. To skip an expansion phase, pass in None. Each expansion algorithm takes the chains from
        placement (and possibly connection) and expands them in some way.
    contraction : function (default None)
        This is optional. To skip a contraction phase, pass in None. Each contraction algorithm takes the chains from
        placement (and possibly connection and/or expansion) and contracts them in some way.
    mm_hint_type : str (default "initial_chains")
        This is the hint type to tell minorminer.find_embedding(). Supported types are "initial_chains" and 
        "suspend_chains". See minorminer.find_embedding() for more information.
    return_layouts : bool (default False)
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
    layout_kwargs, placement_kwargs, connection_kwargs, expansion_kwargs, contraction_kwargs = parse_kwargs(
        kwargs)

    # Compute the layouts
    S_layout, T_layout = parse_layout_parameter(S, T, layout, layout_kwargs)

    # Compute the placement (i.e. chains)
    chains = parse_placement_parameter(
        S_layout, T_layout, placement, placement_kwargs)

    # Connect the chains
    if connection:
        chains = connection(S_layout, T_layout, chains, **connection_kwargs)

    # Expand the chains
    if expansion:
        chains = expansion(S_layout, T_layout, chains, **expansion_kwargs)

    # Contract the chains
    if contraction:
        chains = contraction(S_layout, T_layout, chains, **contraction_kwargs)

    end = time.process_time()
    timeout = kwargs.get("timeout")
    if timeout:
        kwargs["timeout"] = timeout - (end - start)

    # Run minorminer
    if mm_hint_type == "initial_chains":
        output = mm.find_embedding(S, T, initial_chains=chains, **kwargs)
    elif mm_hint_type == "suspend_chains":
        output = mm.find_embedding(S, T, suspend_chains={
            v: [C] for v, C in chains.items()}, **kwargs)
    else:
        raise ValueError(
            "Only initial_chains and suspend_chains are supported minorminer hint types.")

    # Run minerminor.find_embedding()
    if return_layouts:
        return output, (S_layout, T_layout)
    return output


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

    placement_kwargs = {}
    if "max_subset_size" in kwargs:
        placement_kwargs["max_subset_size"] = kwargs.pop("max_subset_size")
    if "strategy" in kwargs:
        placement_kwargs["strategy"] = kwargs.pop("strategy")
    if "num_neighbors" in kwargs:
        placement_kwargs["num_neighbors"] = kwargs.pop("num_neighbors")
    if "fill_processor" in kwargs:
        placement_kwargs["fill_processor"] = kwargs.pop("fill_processor")
    if "unit_tile_capacity" in kwargs:
        placement_kwargs["unit_tile_capacity"] = kwargs.pop(
            "unit_tile_capacity")

    connection_kwargs = {}
    # Pop connection kwargs here if you want to add some in the future

    expansion_kwargs = {}
    if "second" in kwargs:
        expansion_kwargs["second"] = kwargs.pop("second")

    contraction_kwargs = {}
    if "percent" in kwargs:
        contraction_kwargs["percent"] = kwargs.pop("percent")

    return layout_kwargs, placement_kwargs, connection_kwargs, expansion_kwargs, contraction_kwargs


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
        if T.graph.get("family") in ("chimera", "pegasus"):
            T_layout = dnx_layout(T, **layout_kwargs)
        else:
            T_layout = layout(T, **layout_kwargs)

    return S_layout, T_layout


def parse_placement_parameter(S_layout, T_layout, placement, placement_kwargs):
    """
    Determine if placement is a function or a dict.
    """
    # It's a preprocessed placement
    if isinstance(placement, dict):
        return placement_utils.convert_to_chains(placement)

    # It's a function
    else:
        return placement(S_layout, T_layout, **placement_kwargs)

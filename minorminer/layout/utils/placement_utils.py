import networkx as nx
import numpy as np

from ..layout import Layout, dnx_layout, p_norm


def parse_T(T, disallow=None):
    if isinstance(T, nx.Graph) and disallow != "graph":
        if T.graph.get("family") in ("chimera", "pegasus"):
            return dnx_layout(T)
        else:
            return p_norm(T)
    elif isinstance(T, Layout) and disallow != "layout":
        return T
    elif isinstance(T, dict) and disallow != "dict":
        return T
    else:
        raise TypeError("Why did you give me that?")


def check_requirements(S_layout, T_layout, allowed_dnx_graphs=None, allowed_dims=None):
    """
    Checks input for various placement algorithms.
    """
    if len(S_layout) > len(T_layout):
        raise RuntimeError("S is larger than T. You cannot embed S in T.")

    # Datatype parsing
    if isinstance(allowed_dnx_graphs, str):
        allowed_dnx_graphs = [allowed_dnx_graphs]
    elif isinstance(allowed_dnx_graphs, (frozenset, list, set, tuple)):
        pass

    # Datatype parsing
    if isinstance(allowed_dims, int):
        allowed_dims = [allowed_dims]
    elif isinstance(allowed_dims, (frozenset, list, set, tuple)):
        pass

    if allowed_dnx_graphs is None:
        pass
    elif T_layout.G.graph.get("family") not in allowed_dnx_graphs:
        raise NotImplementedError(
            "This strategy is currently only implemented for graphs of type {}.".format(allowed_dnx_graphs))

    if allowed_dims is None:
        pass
    elif S_layout.d not in allowed_dims or T_layout.d not in allowed_dims:
        raise NotImplementedError(
            "This strategy is only implemented for {}-dimensional layouts.".format(allowed_dims))


def minimize_overlap(distances, v_indices, T_vertex_lookup, layout_points, overlap_counter):
    """
    A greedy penalty-type model for choosing overlapping chains.
    """
    # KDTree.query either returns a single index or a list of indexes depending on how many neighbors are queried.
    if isinstance(v_indices, np.int64):
        return T_vertex_lookup[layout_points[v_indices]]

    subsets = {}
    for i in v_indices:
        subset = T_vertex_lookup[layout_points[i]]
        subsets[subset] = sum(d + 10**overlap_counter[v]
                              for d, v in zip(distances, subset))

    cheapest_subset = min(subsets, key=subsets.get)
    overlap_counter.update(cheapest_subset)
    return cheapest_subset


def convert_to_chains(placement):
    """
    Helper function to convert a placement to a chain-ready data structure.
    """
    for v in placement.values():
        if isinstance(v, (list, frozenset, set)):
            return dict(placement)
        return {v: [q] for v, q in placement.items()}

import networkx as nx
import numpy as np

from ..layout import Layout, dnx_layout


def convert_to_chains(placement):
    """
    Helper function to determine whether or not an input is in a chain-ready data structure.
    """
    for v in placement.values():
        if isinstance(v, (list, frozenset, set)):
            return False
        return True


def parse_layout(layout):
    """
    Take in a layout class object or a dictionary and return the dictionary representation.
    """
    if isinstance(layout, Layout):
        return layout.layout
    else:
        return layout


def parse_T(T, disallow=None):
    if isinstance(T, nx.Graph) and disallow != "graph":
        return dnx_layout(T)
    elif isinstance(T, Layout) and disallow != "layout":
        return T
    elif isinstance(T, dict) and disallow != "dict":
        return T
    else:
        raise TypeError("Why did you give me that?")


def check_requirements(S_layout, T_layout, allowed_graphs=None, allowed_dims=None):
    if len(S_layout.G) > len(T_layout.G):
        raise RuntimeError("S is larger than T. You cannot embed S in T.")

    # Datatype parsing
    if allowed_graphs is None:
        allowed_graphs = True
    elif isinstance(allowed_graphs, str):
        allowed_graphs = [allowed_graphs]
    elif isinstance(allowed_graphs, (frozenset, list, set, tuple)):
        pass
    else:
        raise TypeError("What did you give me for an allowed graph?")

    # Datatype parsing
    if allowed_dims is None:
        allowed_dims = True
    elif isinstance(allowed_dims, int):
        allowed_dims = [allowed_dims]
    elif isinstance(allowed_dims, (frozenset, list, set, tuple)):
        pass
    else:
        raise TypeError("What did you give me for an allowed dimension?")

    graph_type = T_layout.G.graph.get("family")
    if allowed_graphs is True or graph_type not in allowed_graphs:
        raise NotImplementedError(
            "This strategy is currently only implemented for graphs of type {}.".format(allowed_graphs))

    if (not isinstance(S_layout, Layout)) or (not isinstance(T_layout, Layout)):
        raise TypeError("This strategy needs Layout class objects.")

    if allowed_dims is True or (S_layout.d not in allowed_dims or T_layout.d not in allowed_dims):
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

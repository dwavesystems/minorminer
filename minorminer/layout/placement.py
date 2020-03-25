import random
import warnings
from collections import Counter
from itertools import cycle, product

import networkx as nx
import numpy as np
from scipy.spatial import KDTree, distance

from minorminer.layout.layout import Layout
from minorminer.layout.utils import dnx_utils, graph_utils, layout_utils


def closest(S_layout, T_layout, subset_size=(1, 1), num_neighbors=1):
    """
    Maps vertices of S to the closest vertices of T as given by S_layout and T_layout. i.e. For each vertex u in 
    S_layout and each vertex v in T_layout, map u to the v with minimum Euclidean distance (||u - v||_2).

    Parameters
    ----------
    S_layout : dict or layout.Layout
        A layout for S; i.e. a map from S to the plane.
    T_layout : dict or layout.Layout
        A layout for T; i.e. a map from T to the plane.
    subset_size : tuple (default (1, 1))
        A lower bound and an upper bound on the size of subets of T that will be considered when mapping vertices of S.
    num_neighbors: int (default 1)
        The number of closest neighbors to query from the KDTree--the neighbor with minimium overlap is chosen.

    Returns
    -------
    placement : dict
        A mapping from vertices of S (keys) to subsets of vertices of T (values).
    """
    # A new layout for subsets of T.
    T_subgraph_layout = {}

    # Get connected subgraphs to consider mapping to
    T_subgraphs = graph_utils.get_connected_subgraphs(
        T_layout.G, subset_size[0], subset_size[1])

    # Calculate the barycenter (centroid) of each subset
    for k in range(subset_size[0], subset_size[1]+1):
        if k == 1:
            for subgraph in T_subgraphs[k]:
                v, = subgraph  # Iterable unpacking
                T_subgraph_layout[subgraph] = T_layout.layout[v]
        else:
            for subgraph in T_subgraphs[k]:
                T_subgraph_layout[subgraph] = np.mean(
                    np.array([T_layout.layout[v] for v in subgraph]), axis=0)

    # Use scipy's KDTree to solve the nearest neighbor problem.
    # This requires a few lookup tables
    T_vertex_lookup = {tuple(p): V for V, p in T_subgraph_layout.items()}
    layout_points = [tuple(p) for p in T_subgraph_layout.values()]
    overlap_counter = Counter()
    tree = KDTree(layout_points)

    placement = {}
    for u, u_pos in S_layout.layout.items():
        distances, v_indices = tree.query(u_pos, num_neighbors)
        placement[u] = layout_utils.minimize_overlap(
            distances, v_indices, T_vertex_lookup, layout_points, overlap_counter)

    return placement


def injective(S_layout, T_layout):
    """
    Injectively maps vertices of S to the closest vertices of T as given by S_layout and T_layout. This is the 
    assignment problem. To solve this it builds a complete bipartite graph between S and T with edge weights the 
    Euclidean distances of the incident vertices; a minimum weight full matching is then computed. This runs in 
    O(|S||T|log(|T|)) time.

    Parameters
    ----------
    S_layout : dict or layout.Layout
        A layout for S; i.e. a map from S to the plane.
    T_layout : dict or layout.Layout
        A layout for T; i.e. a map from T to the plane.

    Returns
    -------
    placement : dict
        A mapping from vertices of S (keys) to vertices of T (values).
    """
    S_layout = layout_utils.parse_layout(S_layout)
    T_layout = layout_utils.parse_layout(T_layout)

    X = nx.Graph()
    # Relabel the vertices from S and T in case of name conflict; S --> 0 and T --> 1.
    X.add_edges_from(
        (
            ((0, u), (1, v), dict(weight=distance.euclidean(u_pos, v_pos)))
            for (u, u_pos), (v, v_pos) in product(S_layout.items(), T_layout.items())
        )
    )
    M = nx.bipartite.minimum_weight_full_matching(
        X, ((0, u) for u in S_layout))

    return {u: M[(0, u)][1] for u in S_layout.keys()}


def binning(S_layout, T_layout, bins=None):
    """
    Map the vertices of S to the vertices of T by first mapping both to an integer lattice.

    Parameters
    ----------
    S_layout : layout.Layout
        A layout for S; i.e. a map from S to the plane.
    T_layout : layout.Layout
        A layout for T; i.e. a map from T to the plane.
    bins : tuple or int (default None)
        The number of bins to put along dimensions; see Layout.to_integer_lattice(). If None, check to see if T is a
        dnx.*_graph() object. If it is, compute bins to be the grid dimension of T.

    Returns
    -------
    placement : dict
        A mapping from vertices of S (keys) to vertices of T (values).
    """
    assert isinstance(S_layout, Layout) and isinstance(T_layout, Layout), (
        "Layout class instances must be passed in.")

    if bins is None:
        dims = dnx_utils.lookup_dnx_dims(T_layout.G)
        if dims:
            n, m = dims[0], dims[1]
            bins = (m, n) + (T_layout.d-2)*(0,)
        else:
            bins = 2

    S_binned = S_layout.integer_lattice_bins(bins)
    T_binned = T_layout.integer_lattice_bins(bins)

    placement = {}
    for p, V in S_binned.items():
        for v, q in zip(V, cycle(T_binned[p])):
            placement[v] = q

    return placement

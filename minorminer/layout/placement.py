import random
import warnings
from itertools import cycle, product

import networkx as nx
from scipy.spatial.distance import euclidean

from minorminer.layout import utils
from minorminer.layout.layout import Layout


def closest(S_layout, T_layout):
    """
    Maps vertices of S to the closest vertices of T as given by S_layout and T_layout. i.e. For each vertex u in 
    S_layout and each vertex v in T_layout, map u to the v with minimum Euclidean distance (||u - v||_2).

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
    S_layout, T_layout = parse_layout(S_layout), parse_layout(T_layout)
    placement = {}
    for u, u_pos in S_layout.items():
        placement[u] = min(
            T_layout, key=lambda v: euclidean(u_pos, T_layout[v]))
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
    S_layout, T_layout = parse_layout(S_layout), parse_layout(T_layout)

    X = nx.Graph()
    # Relabel the vertices from S and T in case of name conflict; S --> 0 and T --> 1.
    X.add_edges_from(
        (
            ((0, u), (1, v), dict(weight=euclidean(u_pos, v_pos)))
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
    """
    assert isinstance(S_layout, Layout) and isinstance(T_layout, Layout), (
        "Layout class instances must be passed in.")

    if bins is None:
        dims = utils.lookup_dnx_dims(T_layout.G)
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


def parse_layout(layout):
    """
    Take in a layout class object or a dictionary and return the dictionary representation.
    """
    if isinstance(layout, Layout):
        return layout.layout
    else:
        return layout

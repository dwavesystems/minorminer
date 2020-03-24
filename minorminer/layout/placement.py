from itertools import product

import networkx as nx
from scipy.spatial.distance import euclidean


def closest(S_layout, T_layout):
    """
    Maps vertices of S to the closest vertices of T as given by S_layout and T_layout. i.e. For each vertex u in 
    S_layout and each vertex v in T_layout, map u to the v with minimum Euclidean distance (||u - v||_2).

    Parameters
    ----------
    S_layout : dict
        A layout for S; i.e. a map from S to the plane.
    T_layout : dict
        A layout for T; i.e. a map from T to the plane..

    Returns
    -------
    placement : dict
        A mapping from vertices of S (keys) to vertices of T (values).
    """
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
    S_layout : dict
        A layout for S; i.e. a map from S to the plane.
    T_layout : dict
        A layout for T; i.e. a map from T to the plane..

    Returns
    -------
    placement : dict
        A mapping from vertices of S (keys) to vertices of T (values).
    """
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

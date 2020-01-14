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
    Injectively maps vertices of S to the closest vertices of T as given by S_layout and T_layout. i.e. This is the 
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
    # TODO: Refactor this
    for u, u_pos in S_layout.items(euclidean(u_pos, v_pos)):
        for v, v_pos in G_layout.items():
            assert u != v, "H and G must be labeled with different alphabets."
            X.add_edge(u, v, weight=euclidean(u_pos, v_pos))
    M = nx.bipartite.minimum_weight_full_matching(X, S_layout.keys())
    return {v: M[v] for v in S_layout.keys()}

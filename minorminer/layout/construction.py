from collections import defaultdict

import minorminer as mm


def singleton(placement):
    """
    Given a placement (a map phi from vertices of S to vertices of T), form the chain [phi(u)] for each u in S.

    Parameters
    ----------
    T : NetworkX graph
        The graph you are embedding into (target).
    placement : dict
        A mapping from vertices of S (keys) to vertices of T (values).

    Returns
    -------
    chains: dict
        A mapping from vertices of S (keys) to chains of T (values).
    """
    return {u: [v] for u, v in placement.items()}


def neighborhood(T, placement, second):
    """
    Given a placement (a map phi from vertices of S to vertices of T), form the chain N_T(phi(u)) (closed neighborhood 
    of v in T) for each u in S.

    Parameters
    ----------
    T : NetworkX graph
        The graph you are embedding into (target).
    placement : dict
        A mapping from vertices of S (keys) to vertices of T (values).
    second : bool
        If True, gets the closed 2nd neighborhood of each vertex. If False, get the closed 1st neighborhood of each
        vertex. 

    Returns
    -------
    chains: dict
        A mapping from vertices of S (keys) to chains of T (values).
    """
    return {u: _closed_neighbors(T, v, second=second) for u, v in placement.items()}


def _closed_neighbors(G, u, second=False):
    """
    Returns the closed neighborhood of u in G.

    Parameters
    ----------
    G : NetworkX graph
        The graph you are considering.
    u : NetworkX node
        The node you are computing the closed neighborhood of.
    second : bool (default False)
        If True, compute the closed 2nd neighborhood.

    Returns
    -------
    neighbors: set
        A set of vertices of G.
    """
    neighbors = set(v for v in G.neighbors(u))
    if second:
        return set((u,)) | neighbors | set(w for v in neighbors for w in G.neighbors(v))
    return set((u,)) | neighbors

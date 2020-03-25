from collections import defaultdict

from minorminer.layout.utils import layout_utils
import minorminer as mm


def pass_along(S, T, placement):
    """
    Given a placement (a map, phi, from vertices of S to vertices (or subsets of vertices) of T), form the chain 
    [phi(u)] (or phi(u)) for each u in S.

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
    # Test if you need to convert values or not
    if layout_utils.convert_to_chains(placement):
        return {u: [v] for u, v in placement.items()}
    else:
        return placement


def neighborhood(S, T, placement, second_neighborhood=False):
    """
    Given a placement (a map, phi, from vertices of S to vertices of T), form the chain N_T(phi(u)) (closed neighborhood 
    of v in T) for each u in S.

    Parameters
    ----------
    T : NetworkX graph
        The graph you are embedding into (target).
    placement : dict
        A mapping from vertices of S (keys) to vertices of T (values).
    second_neighborhood : bool (default False)
        If True, gets the closed 2nd neighborhood of each vertex. If False, get the closed 1st neighborhood of each
        vertex. 

    Returns
    -------
    chains: dict
        A mapping from vertices of S (keys) to chains of T (values).
    """
    return {u: _closed_neighbors(T, v, second_neighborhood=second_neighborhood) for u, v in placement.items()}


def _closed_neighbors(G, u, second_neighborhood=False):
    """
    Returns the closed neighborhood of u in G.

    Parameters
    ----------
    G : NetworkX graph
        The graph you are considering.
    u : NetworkX node
        The node you are computing the closed neighborhood of.
    second_neighborhood : bool (default False)
        If True, compute the closed 2nd neighborhood.

    Returns
    -------
    neighbors: set
        A set of vertices of G.
    """
    neighbors = set(v for v in G.neighbors(u))
    if second_neighborhood:
        return set((u,)) | neighbors | set(w for v in neighbors for w in G.neighbors(v))
    return set((u,)) | neighbors

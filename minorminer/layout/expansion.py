import random
from collections import defaultdict

import dwave_networkx as dnx
import networkx as nx

from .utils import dnx_utils, layout_utils, placement_utils


def neighborhood(S_layout, T, chains, second=False, **kwargs):
    """
    Given a placement (a map, phi, from vertices of S to vertices of T), form the chain N_T(phi(u)) (closed neighborhood 
    of v in T) for each u in S.

    Parameters
    ----------
    placement : dict
        A mapping from vertices of S (keys) to vertices of T (values).
    S_layout : layout.Layout
        A layout for S; i.e. a map from S to R^d.
    T : layout.Layout or networkx.Graph
        A layout for T; i.e. a map from T to R^d. Or a networkx graph to make a layout from.
    second : bool (default False)
        If True, gets the closed 2nd neighborhood of each vertex. If False, get the closed 1st neighborhood of each
        vertex.

    Returns
    -------
    chains: dict
        A mapping from vertices of S (keys) to chains of T (values).
    """
    # Standardize input
    T_layout = placement_utils.parse_T(T, disallow="dict")

    return {u: _closed_neighbors(T_layout.G, v, second=second)
            for u, v in chains.items()}


def _closed_neighbors(G, u, second):
    """
    Returns the closed neighborhood of u in G.

    Parameters
    ----------
    G : NetworkX graph
        The graph you are considering.
    u : NetworkX node
        The node you are computing the closed neighborhood of.
    second : bool
        If True, compute the closed 2nd neighborhood.

    Returns
    -------
    neighbors: set
        A set of vertices of G.
    """
    if isinstance(u, (list, frozenset, set)):
        closed_neighbors = nx.node_boundary(G, u) | set(u)
    else:
        closed_neighbors = set(G.neighbors(u)) | set((u, ))

    if second:
        return nx.node_boundary(G, closed_neighbors) | closed_neighbors
    return closed_neighbors


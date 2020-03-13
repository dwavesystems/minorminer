import random
from collections import defaultdict

import dwave_networkx as dnx
import minorminer as mm
import networkx as nx

from .utils import construction_utils, dnx_utils, layout_utils, placement_utils


def pass_along(placement, **kwargs):
    """
    Given a placement (a map, phi, from vertices of S to vertices (or subsets of vertices) of T), form the chain 
    [phi(u)] (or phi(u)) for each u in S.

    Parameters
    ----------
    placement : dict
        A mapping from vertices of S (keys) to vertices of T (values).

    Returns
    -------
    chains: dict
        A mapping from vertices of S (keys) to chains of T (values).
    """
    # If needed turn singletons into lists
    return construction_utils.convert_to_chains(placement)


def crosses(placement, S_layout, T, **kwargs):
    """
    Extend chains for vertices of S along rows and columns of qubits of T (T must be a D-Wave hardware graph). 

    If you project the layout of S onto 1-dimensional subspaces, for each vertex u of S a chain is a minimal interval 
    containing all neighbors of u. This amounts to a placement where each chain is a cross shaped chain where the middle 
    of the cross is contained in a unit cell, and the legs of the cross extend horizontally and vertically as far as 
    there are neighbors of u.

    Parameters
    ----------
    placement : dict
        A mapping from vertices of S (keys) to vertices of T (values).
    S_layout : layout.Layout
        A layout for S; i.e. a map from S to R^d.
    T : layout.Layout or dwave-networkx.Graph
        A layout for T; i.e. a map from T to R^d. Or a D-Wave networkx graph to make a layout from.

    Returns
    -------
    chains: dict
        A mapping from vertices of S (keys) to chains of T (values).
    """
    # Standardize input
    T_layout = placement_utils.parse_T(T, disallow="dict")

    # Raise exceptions if you need to
    placement_utils.check_requirements(
        S_layout, T_layout, allowed_dnx_graphs="chimera", allowed_dims=2)

    # If needed turn singletons into lists
    chains = construction_utils.convert_to_chains(placement)

    # Grab the coordinate version of the labels
    if T_layout.G.graph["labels"] == "coordinate":
        pass
    else:
        n, m, t = dnx_utils.lookup_dnx_dims(T_layout.G)
        C = dnx.chimera_coordinates(m, n, t)
        chains = {
            v: {C.linear_to_chimera(q) for q in Q} for v, Q in chains.items()
        }

    for v in S_layout.G:
        hor_v, ver_v = _horizontal_and_vertical_qubits(chains[v])

        min_x = min(hor_v[1], ver_v[1])
        max_x = max(hor_v[1], ver_v[1])
        min_y = min(hor_v[0], ver_v[0])
        max_y = max(hor_v[0], ver_v[0])
        for u in S_layout.G[v]:
            hor_u, ver_u = _horizontal_and_vertical_qubits(chains[u])

            min_x = min(min_x, min(hor_u[1], ver_u[1]))
            max_x = max(max_x, max(hor_u[1], ver_u[1]))
            min_y = min(min_y, min(hor_u[0], ver_u[0]))
            max_y = max(max_y, max(hor_u[0], ver_u[0]))

            row_qubits = set()
            for j in range(min_x, max_x+1):
                row_qubits.add((ver_v[0], j, 1, ver_v[3]))

            column_qubits = set()
            for i in range(min_y, max_y+1):
                column_qubits.add((i, hor_v[1], 0, hor_v[3]))

        chains[v] = row_qubits | column_qubits

    # Return the right type of vertices
    return dnx_utils.relabel_chains(T_layout.G, chains)


def _horizontal_and_vertical_qubits(chain):
    """
    Given a chain, select one horizontal and one vertical qubit. If one doen't exist, extend the chain to include one. 
    """
    # Split each chain into horizontal and vertical qubits
    horizontal_qubits = [q for q in chain if q[2] == 0]
    vertical_qubits = [q for q in chain if q[2] == 1]

    # FIXME: Making a random choice here might not be the best. In the placement strategy that is currently
    # winning, intersection, it doesn't actually matter because both lists above (*_qubits) have size 1.
    hor_v, ver_v = None, None
    if horizontal_qubits:
        hor_v = random.choice(horizontal_qubits)
    if vertical_qubits:
        ver_v = random.choice(vertical_qubits)

    if hor_v is None:
        hor_v = (ver_v[0], ver_v[1], 0, random.randint(0, 3))
    if ver_v is None:
        ver_v = (hor_v[0], hor_v[1], 0, random.randint(0, 3))

    return hor_v, ver_v


def neighborhood(placement, S_layout, T, second=False):
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
            for u, v in placement.items()}


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
    if isinstance(u, (list, frozenset, set)):
        closed_neighbors = nx.node_boundary(G, u) | set(u)
    else:
        closed_neighbors = set(G.neighbors(u)) | set((u, ))

    if second:
        return nx.node_boundary(G, closed_neighbors) | closed_neighbors
    return closed_neighbors

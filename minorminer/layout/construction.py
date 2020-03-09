from collections import defaultdict

import dwave_networkx as dnx
import minorminer as mm
from .utils import dnx_utils, layout_utils, placement_utils


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
    # Test if you need to turn singletons into lists or not
    if layout_utils.convert_to_chains(placement):
        chains = {u: [v] for u, v in placement.items()}
    else:
        chains = placement

    return chains


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
        S_layout, T_layout, allowed_graphs="chimera", allowed_dims=2)

    # Grab the coordinate version of the labels
    if T_layout.G.graph["labels"] == "coordinate":
        chains = dict(placement)
    else:
        n, m, t = dnx_utils.lookup_dnx_dims(T_layout.G)
        C = dnx.chimera_coordinates(m, n, t)
        chains = {
            v: {C.linear_to_chimera(q) for q in Q} for v, Q in placement.items()
        }

    for v in S_layout.G:
        # Figure out which qubit is the the horizontal one and which is the vertical one
        Q = chains[v]
        for q in Q:
            if q[2] == 0:
                hor_v = q
            else:
                ver_v = q

        min_x, max_x = ver_v[1], ver_v[1]
        min_y, max_y = ver_v[0], ver_v[0]
        for u in S_layout.G[v]:
            # Figure out which qubit is the the horizontal one and which is the vertical one
            QQ = chains[u]
            for q in QQ:
                if q[2] == 0:
                    hor_u = q
                else:
                    ver_u = q

            min_x = min(min_x, hor_u[1])
            max_x = max(max_x, hor_u[1])
            min_y = min(min_y, ver_u[0])
            max_y = max(max_y, ver_u[0])

            row_qubits = set()
            for j in range(min_x, max_x+1):
                row_qubits.add((ver_v[0], j, 1, ver_v[3]))

            column_qubits = set()
            for i in range(min_y, max_y+1):
                column_qubits.add((i, hor_v[1], 0, hor_v[3]))

        chains[v] = row_qubits | column_qubits

    # Return the right type of vertices
    if T_layout.G.graph["labels"] == "coordinate":
        return chains
    else:
        return {v: [C.chimera_to_linear(q) for q in Q] for v, Q in chains.items()}

# FIXME: If want to implement (it's not currently a winning strategy) mimic crosses() above.
# def tees(S_layout, T_layout):
#     """
#     Map the vertices of S to rows and columns of qubits of T (T must be a D-Wave hardware graph).

#     Order the vertices of S along the y-axis from bottom to top. For each vertex u of S, form a chain that is the
#     minimal interval containing every neighbor "ahead" of u on the y-axis. For each v in N(u), form a chain that is the
#     minimal interval containing v and the projection of u on the x-axis. This amounts to a placement where each chain
#     has the shape a subset of a capital "T". For each vertex u of S, the intersection of the T (if it exists) is
#     necessarily contained in unit cell given by the layout, and the legs of the T are as described above.

#     This guarantees in an overlap embedding of S in T.

#     Parameters
#     ----------
#     S_layout : layout.Layout
#         A layout for S; i.e. a map from S to R^d.
#     T_layout : layout.Layout
#         A layout for T; i.e. a map from T to R^d.

#     Returns
#     -------
#     placement : dict
#         A mapping from vertices of S (keys) to vertices of T (values).
#     """
#     # Get those assertions out of the way
#     assert S_layout.d == 2 and T_layout.d == 2, "This is only implemented for 2-dimensional layouts."
#     assert isinstance(S_layout, Layout) and isinstance(T_layout, Layout), (
#         "Layout class instances must be passed in.")
#     dims = dnx_utils.lookup_dnx_dims(T_layout.G)
#     assert dims is not None, "I need a D-Wave NetworkX graph."

#     # Scale the layout so that we have integer spots for each vertical and horizontal qubit.
#     n, m, t = dims
#     columns, rows = n*t-1, m*t-1
#     scaled_layout = S_layout.scale_to_positive_orthant(
#         (columns, rows), invert=True)

#     # Keep track of vertices that are connected
#     routed_vertices = set()

#     # Sort the vertices in the layout from bottom to top
#     placement = defaultdict(set)
#     for v, pos in sorted(scaled_layout.items(), key=lambda x: x[1][1]):
#         r_x, j, x_k = dnx_utils.get_row_or_column(pos[0], t)  # Column
#         r_y, _, _ = dnx_utils.get_row_or_column(pos[1], t)  # Row

#         max_y = r_y
#         for u in S_layout.G[v]:
#             # Skip over previously routed vertices
#             if u in routed_vertices:
#                 continue

#             # Figure out how far you need to extend the leg of the T above you
#             u_y, u_i, u_y_k = dnx_utils.get_row_or_column(
#                 scaled_layout[u][1], t)
#             max_y = max(max_y, u_y)

#             # Have your neighbors run left or right into you
#             row_qubits = set()
#             u_x, _, _ = dnx_utils.get_row_or_column(scaled_layout[u][0], t)
#             for p in range(min(u_x, r_x), max(u_x, r_x)+1):
#                 _, col, _ = dnx_utils.get_row_or_column(p, t)
#                 row_qubits.add((u_i, col, 1, u_y_k))

#             placement[u] |= row_qubits

#         column_qubits = set()
#         for p in range(r_y, max_y+1):
#             _, row, _ = dnx_utils.get_row_or_column(p, t)
#             column_qubits.add((row, j, 0, x_k))

#         placement[v] |= column_qubits

#         # The vertex v is now totally connected to its neighbors
#         routed_vertices.add(v)

#     # Return the right type of vertices
#     if T_layout.G.graph["labels"] == "coordinate":
#         return placement
#     else:
#         C = dnx.chimera_coordinates(m, n, t)
#         return {v: [C.chimera_to_linear(q) for q in Q] for v, Q in placement.items()}


def neighborhood(S, T, placement, second=False, extend=False):
    """
    Given a placement (a map, phi, from vertices of S to vertices of T), form the chain N_T(phi(u)) (closed neighborhood 
    of v in T) for each u in S.

    Parameters
    ----------
    T : NetworkX graph
        The graph you are embedding into (target).
    placement : dict
        A mapping from vertices of S (keys) to vertices of T (values).
    second : bool (default False)
        If True, gets the closed 2nd neighborhood of each vertex. If False, get the closed 1st neighborhood of each
        vertex. 
    extend : bool (default False)
        If True, extend chains to mimic the structure of S in T.

    Returns
    -------
    chains: dict
        A mapping from vertices of S (keys) to chains of T (values).
    """
    chains = {u: closed_neighbors(T, v, second=second)
              for u, v in placement.items()}
    if extend:
        return extend_chains(S, T, chains)

    return chains


def extend_chains(S, T, initial_chains):
    """
    Extend chains in T so that their structure matches that of S. That is, form an overlap embedding of S in T
    where the initial_chains are subsets of the overlap embedding chains. This is done via minorminer.

    Parameters
    ----------
    S : NetworkX graph
        The graph you are embedding (source).
    T : NetworkX graph
        The graph you are embedding into (target).
    initial_chains : dict
        A mapping from vertices of S (keys) to chains of T (values).

    Returns
    -------
    extended_chains: dict
        A mapping from vertices of S (keys) to chains of T (values).
    """
    # Extend the chains to minimal overlapped embedding
    miner = mm.miner(S, T, initial_chains=initial_chains)
    extended_chains = defaultdict(set)
    for u in S:
        # Revert to the initial_chains and compute the embedding where you tear-up u.
        emb = miner.quickpass(
            [u], clear_first=True, overlap_bound=S.number_of_nodes()
        )

        # Add the new chain for u and the singleton one too (in case it got moved in emb)
        extended_chains[u].update(set(emb[u]).union(initial_chains[u]))

        # For each neighbor v of u, grow u to reach v
        for v in S[u]:
            extended_chains[u].update(
                set(emb[v]).difference(initial_chains[v]))

    return extended_chains


def closed_neighbors(G, u, second=False):
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

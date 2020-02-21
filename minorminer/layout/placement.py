import random
import warnings
from collections import Counter, defaultdict
from itertools import cycle, product

import dwave_networkx as dnx
import networkx as nx
import numpy as np
from minorminer.layout.layout import Layout
from minorminer.layout.utils import dnx_utils, graph_utils, layout_utils
from scipy.spatial import KDTree, distance


def closest(S_layout, T_layout, max_subset_size=(1, 1), num_neighbors=1):
    """
    Maps vertices of S to the closest vertices of T as given by S_layout and T_layout. i.e. For each vertex u in
    S_layout and each vertex v in T_layout, map u to the v with minimum Euclidean distance (||u - v||_2).

    Parameters
    ----------
    S_layout : dict or layout.Layout
        A layout for S; i.e. a map from S to R^d.
    T_layout : dict or layout.Layout
        A layout for T; i.e. a map from T to R^d.
    max_subset_size : tuple (default (1, 1))
        A lower bound and an upper bound on the size of subets of T that will be considered when mapping vertices of S.
    num_neighbors: int (default 1)
        The number of closest neighbors to query from the KDTree--the neighbor with minimium overlap is chosen.

    Returns
    -------
    placement : dict
        A mapping from vertices of S (keys) to subsets of vertices of T (values).
    """
    S_layout = parse_layout(S_layout)

    # Copy the dictionary layout for T so we can modify it.
    layout = dict(T_layout.layout)
    # Get the graph for layout T
    T = T_layout.G

    # Get connected subgraphs to consider mapping to
    if max_subset_size != (1, 1):
        T_subgraphs = graph_utils.get_connected_subgraphs(
            T, max_subset_size[0], max_subset_size[1])

        # Calculate the barycenter (centroid) of each subset with size > 1
        for k in range(max(2, max_subset_size[0]), max_subset_size[1]+1):
            for subgraph in T_subgraphs[k]:
                layout[subgraph] = np.mean(
                    tuple(layout[v] for v in subgraph), axis=0)

    # Determine if you need to add or delete subsets of size 1
    if max_subset_size[0] == 1:
        for v in T:
            layout[frozenset((v,))] = layout[v]
            del layout[v]
    else:
        for v in T:
            del layout[v]

    # Use scipy's KDTree to solve the nearest neighbor problem.
    # This requires a few lookup tables
    T_vertex_lookup = {tuple(p): v for v, p in layout.items()}
    layout_points = [tuple(p) for p in layout.values()]
    overlap_counter = Counter()
    tree = KDTree(layout_points)

    placement = {}
    for u, u_pos in S_layout.items():
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
        A layout for S; i.e. a map from S to R^d.
    T_layout : dict or layout.Layout
        A layout for T; i.e. a map from T to R^d.

    Returns
    -------
    placement : dict
        A mapping from vertices of S (keys) to vertices of T (values).
    """
    S_layout = parse_layout(S_layout)
    T_layout = parse_layout(T_layout)

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
        A layout for S; i.e. a map from S to R^d.
    T_layout : layout.Layout
        A layout for T; i.e. a map from T to R^d.
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


def crosses(S_layout, T_layout):
    """
    Map the vertices of S to rows and columns of qubits of T (T must be a D-Wave hardware graph). 

    If you project the layout of S onto 1-dimensional subspaces, for each vertex u of S a chain is a minimal interval 
    containing all neighbors of u. This amounts to a placement where each chain is a cross shaped chain where the middle 
    of the cross is contained in a unit cell, and the legs of the cross extend horizontally and vertically as far as 
    there are neighbors of u.

    This guarantees in an overlap embedding of S in T.

    Parameters
    ----------
    S_layout : layout.Layout
        A layout for S; i.e. a map from S to R^d.
    T_layout : layout.Layout
        A layout for T; i.e. a map from T to R^d.

    Returns
    -------
    placement : dict
        A mapping from vertices of S (keys) to vertices of T (values).
    """
    # Get those assertions out of the way
    assert S_layout.d == 2 and T_layout.d == 2, "This is only implemented for 2-dimensional layouts."
    assert isinstance(S_layout, Layout) and isinstance(T_layout, Layout), (
        "Layout class instances must be passed in.")
    dims = dnx_utils.lookup_dnx_dims(T_layout.G)
    assert dims is not None, "I need a D-Wave NetworkX graph."

    # Scale the layout so that we have integer spots for each vertical and horizontal qubit.
    n, m, t = dims
    columns, rows = n*t-1, m*t-1
    scaled_layout = S_layout.scale_to_positive_orthant(
        (columns, rows), invert=True)

    placement = {}
    for v, pos in scaled_layout.items():
        r_x, j, x_k = dnx_utils.get_row_or_column(pos[0], t)
        r_y, i, y_k = dnx_utils.get_row_or_column(pos[1], t)

        min_x, max_x = r_x, r_x
        min_y, max_y = r_y, r_y
        # Extend the cross for as long as your neighbors
        for u in S_layout.G[v]:
            x, _, _ = dnx_utils.get_row_or_column(scaled_layout[u][0], t)
            min_x = min(min_x, x)
            max_x = max(max_x, x)

            y, _, _ = dnx_utils.get_row_or_column(scaled_layout[u][1], t)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

        # Compute the qubits as you run along a row and column
        row_qubits = set()
        for p in range(min_x, max_x+1):
            _, col, _ = dnx_utils.get_row_or_column(p, t)
            row_qubits.add((i, col, 1, y_k))

        column_qubits = set()
        for p in range(min_y, max_y+1):
            _, row, _ = dnx_utils.get_row_or_column(p, t)
            column_qubits.add((row, j, 0, x_k))

        placement[v] = row_qubits | column_qubits

    # Return the right type of vertices
    if T_layout.G.graph["labels"] == "coordinate":
        return placement
    else:
        C = dnx.chimera_coordinates(m, n, t)
        return {v: [C.chimera_to_linear(q) for q in Q] for v, Q in placement.items()}


def tees(S_layout, T_layout):
    """
    Map the vertices of S to rows and columns of qubits of T (T must be a D-Wave hardware graph). 

    Order the vertices of S along the y-axis from bottom to top. For each vertex u of S, form a chain that is the 
    minimal interval containing every neighbor "ahead" of u on the y-axis. For each v in N(u), form a chain that is the 
    minimal interval containing v and the projection of u on the x-axis. This amounts to a placement where each chain 
    has the shape a subset of a capital "T". For each vertex u of S, the intersection of the T (if it exists) is 
    necessarily contained in unit cell given by the layout, and the legs of the T are as described above.

    This guarantees in an overlap embedding of S in T.

    Parameters
    ----------
    S_layout : layout.Layout
        A layout for S; i.e. a map from S to R^d.
    T_layout : layout.Layout
        A layout for T; i.e. a map from T to R^d.

    Returns
    -------
    placement : dict
        A mapping from vertices of S (keys) to vertices of T (values).
    """
    # Get those assertions out of the way
    assert S_layout.d == 2 and T_layout.d == 2, "This is only implemented for 2-dimensional layouts."
    assert isinstance(S_layout, Layout) and isinstance(T_layout, Layout), (
        "Layout class instances must be passed in.")
    dims = dnx_utils.lookup_dnx_dims(T_layout.G)
    assert dims is not None, "I need a D-Wave NetworkX graph."

    # Scale the layout so that we have integer spots for each vertical and horizontal qubit.
    n, m, t = dims
    columns, rows = n*t-1, m*t-1
    scaled_layout = S_layout.scale_to_positive_orthant(
        (columns, rows), invert=True)

    # Keep track of vertices that are connected
    routed_vertices = set()

    # Sort the vertices in the layout from bottom to top
    placement = defaultdict(set)
    for v, pos in sorted(scaled_layout.items(), key=lambda x: x[1][1]):
        r_x, j, x_k = dnx_utils.get_row_or_column(pos[0], t)  # Column
        r_y, _, _ = dnx_utils.get_row_or_column(pos[1], t)  # Row

        max_y = r_y
        for u in S_layout.G[v]:
            # Skip over previously routed vertices
            if u in routed_vertices:
                continue

            # Figure out how far you need to extend the leg of the T above you
            u_y, u_i, u_y_k = dnx_utils.get_row_or_column(
                scaled_layout[u][1], t)
            max_y = max(max_y, u_y)

            # Have your neighbors run left or right into you
            row_qubits = set()
            u_x, _, _ = dnx_utils.get_row_or_column(scaled_layout[u][0], t)
            for p in range(min(u_x, r_x), max(u_x, r_x)+1):
                _, col, _ = dnx_utils.get_row_or_column(p, t)
                row_qubits.add((u_i, col, 1, u_y_k))

            placement[u] |= row_qubits

        column_qubits = set()
        for p in range(r_y, max_y+1):
            _, row, _ = dnx_utils.get_row_or_column(p, t)
            column_qubits.add((row, j, 0, x_k))

        placement[v] |= column_qubits

        # The vertex v is now totally connected to its neighbors
        routed_vertices.add(v)

    # Return the right type of vertices
    if T_layout.G.graph["labels"] == "coordinate":
        return placement
    else:
        C = dnx.chimera_coordinates(m, n, t)
        return {v: [C.chimera_to_linear(q) for q in Q] for v, Q in placement.items()}


def parse_layout(layout):
    """
    Take in a layout class object or a dictionary and return the dictionary representation.
    """
    if isinstance(layout, Layout):
        return layout.layout
    else:
        return layout

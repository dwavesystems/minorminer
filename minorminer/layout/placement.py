import random
import warnings
from collections import Counter, defaultdict
from itertools import cycle, product

import networkx as nx
import numpy as np
from minorminer.layout import layout
from minorminer.layout.utils import (dnx_utils, graph_utils, layout_utils,
                                     placement_utils)
from scipy.spatial import KDTree, distance


def closest(S_layout, T, max_subset_size=(1, 1), num_neighbors=1, **kwargs):
    """
    Maps vertices of S to the closest vertices of T as given by S_layout and T_layout. i.e. For each vertex u in
    S_layout and each vertex v in T_layout, map u to the v with minimum Euclidean distance (||u - v||_2).

    Parameters
    ----------
    S_layout : dict or layout.Layout
        A layout for S; i.e. a map from S to R^d.
    T : dict or layout.Layout or dwave-networkx.Graph
        A layout for T; i.e. a map from T to R^d. Or a D-Wave networkx graph to make a layout from.
    max_subset_size : tuple (default (1, 1))
        A lower bound and an upper bound on the size of subets of T that will be considered when mapping vertices of S.
        If different from default, then T_layout must be a Layout object.
    num_neighbors: int (default 1)
        The number of closest neighbors to query from the KDTree--the neighbor with minimium overlap is chosen.

    Returns
    -------
    placement : dict
        A mapping from vertices of S (keys) to subsets of vertices of T (values).
    """
    S_layout_dict = placement_utils.parse_layout(S_layout)

    # FIXME: This is real messy
    T_layout = placement_utils.parse_T(T)  # Turns graph into layout
    if isinstance(T_layout, layout.Layout):
        T_layout_dict = dict(T_layout.layout)  # Make a copy
    elif isinstance(T_layout, dict):
        T_layout_dict = dict(T_layout)

    T_vertices = list(T_layout_dict.keys())

    # Get connected subgraphs to consider mapping to
    if max_subset_size != (1, 1):
        assert isinstance(
            T_layout, layout.Layout), "Pass in a Layout object so we can access the graph."

        # Copy the dictionary layout for T so we can modify it.
        T_layout_dict = dict(T_layout.layout)

        T_subgraphs = graph_utils.get_connected_subgraphs(
            T_layout.G, max_subset_size[0], max_subset_size[1])

        # Calculate the barycenter (centroid) of each subset with size > 1
        for k in range(max(2, max_subset_size[0]), max_subset_size[1]+1):
            for subgraph in T_subgraphs[k]:
                T_layout_dict[subgraph] = np.mean(
                    tuple(T_layout_dict[v] for v in subgraph), axis=0)

    # Determine if you need to add or delete subsets of size 1
    if max_subset_size[0] == 1:
        for v in T_vertices:
            T_layout_dict[frozenset((v,))] = T_layout_dict[v]
            del T_layout_dict[v]
    else:
        for v in T_vertices:
            del T_layout_dict[v]

    # Use scipy's KDTree to solve the nearest neighbor problem.
    # This requires a few lookup tables
    T_vertex_lookup = {tuple(p): v for v, p in T_layout_dict.items()}
    layout_points = [tuple(p) for p in T_layout_dict.values()]
    overlap_counter = Counter()
    tree = KDTree(layout_points)

    placement = {}
    for u, u_pos in S_layout_dict.items():
        distances, v_indices = tree.query(u_pos, num_neighbors)
        placement[u] = layout_utils.minimize_overlap(
            distances, v_indices, T_vertex_lookup, layout_points, overlap_counter)

    return placement


def injective(S_layout, T, **kwargs):
    """
    Injectively maps vertices of S to the closest vertices of T as given by S_layout and T_layout. This is the
    assignment problem. To solve this it builds a complete bipartite graph between S and T with edge weights the
    Euclidean distances of the incident vertices; a minimum weight full matching is then computed. This runs in
    O(|S||T|log(|T|)) time.

    Parameters
    ----------
    S_layout : dict or layout.Layout
        A layout for S; i.e. a map from S to R^d.
    T : dict or layout.Layout or dwave-networkx.Graph
        A layout for T; i.e. a map from T to R^d. Or a D-Wave networkx graph to make a layout from.

    Returns
    -------
    placement : dict
        A mapping from vertices of S (keys) to vertices of T (values).
    """
    T_layout = placement_utils.parse_T(T)

    S_layout_dict = placement_utils.parse_layout(S_layout)
    T_layout_dict = placement_utils.parse_layout(T_layout)

    X = nx.Graph()
    # Relabel the vertices from S and T in case of name conflict; S --> 0 and T --> 1.
    X.add_edges_from(
        (
            ((0, u), (1, v), dict(weight=distance.euclidean(u_pos, v_pos)))
            for (u, u_pos), (v, v_pos) in product(S_layout_dict.items(), T_layout_dict.items())
        )
    )
    M = nx.bipartite.minimum_weight_full_matching(
        X, ((0, u) for u in S_layout_dict))

    return {u: [M[(0, u)][1]] for u in S_layout_dict.keys()}


def binning(S_layout, T, bins=None, strategy="cycle", **kwargs):
    """
    Map the vertices of S to the vertices of T by first mapping both to an integer lattice.

    Parameters
    ----------
    S_layout : layout.Layout
        A layout for S; i.e. a map from S to R^d.
    T : layout.Layout or dwave-networkx.Graph
        A layout for T; i.e. a map from T to R^d. Or a D-Wave networkx graph to make a layout from.
    bins : tuple or int (default None)
        The number of bins to put along dimensions; see Layout.to_integer_lattice(). If None, check to see if T is a
        dnx.*_graph() object. If it is, compute bins to be the grid dimension of T.
    strategy : string (default "cycle")
        cycle : Cycle through the qubits in the bin and assign vertices to them one at a time.
        all : Map each vertex in a bin to all qubits in that bin.

    Returns
    -------
    placement : dict
        A mapping from vertices of S (keys) to vertices of T (values).
    """
    # Standardize input
    T_layout = placement_utils.parse_T(T, disallow="dict")

    assert isinstance(S_layout, layout.Layout) and isinstance(T_layout, layout.Layout), (
        "Layout class instances must be passed in.")

    if bins is None:
        dims = dnx_utils.lookup_dnx_dims(T_layout.G)
        if dims:
            n, m = dims[0], dims[1]
            bins = (m, n) + (T_layout.d-2)*(0,)
        else:
            bins = 2

    S_binned, _ = S_layout.integer_lattice_bins(bins)
    T_binned, _ = T_layout.integer_lattice_bins(bins)

    placement = {}
    if strategy == "cycle":
        for p, S in S_binned.items():
            for v, q in zip(S, cycle(T_binned[p])):
                placement[v] = [q]

    elif strategy == "all":
        for p, V in S_binned.items():
            for v in V:
                placement[v] = T_binned[p]

    return placement


def intersection(S_layout, T, full_fit=True, **kwargs):
    """
    Map each vertex of S to its nearest row/column intersection qubit in T (T must be a D-Wave hardware graph). 

    Parameters
    ----------
    S_layout : layout.Layout
        A layout for S; i.e. a map from S to R^d.
    T : layout.Layout or dwave-networkx.Graph
        A layout for T; i.e. a map from T to R^d. Or a D-Wave networkx graph to make a layout from.
    full_fit : bool (default True)
        If True, S_layout is scaled so that it maximizes the area of T. If False, the size of S_layout and T_layout are
        comparably scaled.

    Returns
    -------
    placement : dict
        A mapping from vertices of S (keys) to vertices of T (values).
    """
    # Standardize input
    T_layout = placement_utils.parse_T(T, disallow="dict")

    # Raise exceptions if you need to
    placement_utils.check_requirements(
        S_layout, T_layout, allowed_graphs="chimera", allowed_dims=2)

    # D-Wave counts the y direction like matrix rows; inversion makes pictures match
    modified_layout = layout.invert_layout(
        S_layout.layout_array, S_layout.center)

    # Scale the layout so that for each vertical and horizontal qubit that cross each other, we have an integer point.
    m, n, t = dnx_utils.lookup_dnx_dims(T_layout.G)
    if full_fit:  # Scale to fill the area
        scale = (t*max(n, m)-1)/2
    else:  # Scale by the size of the shores
        scale = t*S_layout.scale

    # Scale S to an appropriate size
    modified_layout = layout.scale_layout(
        modified_layout, scale, S_layout.scale, S_layout.center)

    # Center to the positive orthant
    modified_layout = layout.center_layout(
        modified_layout, 2*(scale, ), S_layout.center)

    # Turn it into a dictionary
    modified_layout = {v: pos for v, pos in zip(S_layout.G, modified_layout)}

    placement = {}
    for v, pos in modified_layout.items():
        _, j, x_k = dnx_utils.get_row_or_column(pos[0], t)
        _, i, y_k = dnx_utils.get_row_or_column(pos[1], t)

        placement[v] = [(i, j, 0, x_k), (i, j, 1, y_k)]

    # Return the right type of vertices
    return dnx_utils.relabel_chains(T_layout.G, placement)


def injective_intersection(S_layout, T, unit_tile_capacity=4, fill_processor=False, **kwargs):
    """
    First map vertices of S to unit tiles of T, then topple to get at most 4 vertices of S per unit tile, then in each 
    unit tile, assign the vertices of S to a transversal (T must be a D-Wave hardware graph). 

    Parameters
    ----------
    S_layout : layout.Layout
        A layout for S; i.e. a map from S to R^d.
    T : layout.Layout or dwave-networkx.Graph
        A layout for T; i.e. a map from T to R^d. Or a D-Wave networkx graph to make a layout from.
    unit_tile_capacity : int (default 4)
        The number of variables (vertices of S) that are allowed to map to unit tiles of T.
    fill_processor : bool (default False)
        If True, S_layout is scaled so that it fills the processor. If False, the scale of S_layout is used.

    Returns
    -------
    placement : dict
        A mapping from vertices of S (keys) to vertices of T (values).
    """
    # Standardize input
    T_layout = placement_utils.parse_T(T, disallow="dict")

    # Raise exceptions if you need to
    placement_utils.check_requirements(
        S_layout, T_layout, allowed_graphs=["chimera", "pegasus"], allowed_dims=2)

    # Get the lattice point mapping for the dnx graph
    lattice_mapping = dnx_utils.lookup_grid_coordinates(T_layout.G)

    # Extract the dimensions of the grid (the mapping is from [0, n-1] and [0, m-1])
    m, n = np.max(list(lattice_mapping.values()), 0) + (1, 1)

    # Make the grid "quotient" of the dnx_graph--the ~K_4,4 unit cells of the dnx_graph are quotiented to grid points
    G = nx.grid_2d_graph(m, n)
    for v, int_point in lattice_mapping.items():
        # Add qubits (vertices of T) to grid points
        if "qubits" in G.nodes[int_point]:
            G.nodes[int_point]["qubits"].add(v)
        else:
            G.nodes[int_point]["qubits"] = {v}

        # Also initialize space for variables (vertices of S)
        G.nodes[int_point]["variables"] = set()

    # Map the layout to the grid (R^2 --> Grid(Chimera))
    # D-Wave counts the y direction like matrix rows; inversion makes pictures match
    modified_layout = layout.invert_layout(
        S_layout.layout_array, S_layout.center)

    # Scale S to fill T at the grid level
    scale = (max(n, m)-1)/2
    if fill_processor:
        modified_layout = layout.scale_layout(
            modified_layout, scale, S_layout.scale, S_layout.center)

    # Center to the positive orthant
    modified_layout = layout.center_layout(
        modified_layout, 2*(scale,), S_layout.center)

    # Turn it into a dictionary
    modified_layout = {v: pos for v, pos in zip(S_layout.G, modified_layout)}

    # Put the "variables" (vertices from S) in the graph too
    for v, pos in modified_layout.items():
        grid_point = tuple(np.round(pos))
        G.nodes[grid_point]["variables"].add(v)

    # Check who needs to move (at most 4 vertices of S allowed per grid point)
    moves = {v: 0 for v in S_layout.G}
    topple = True  # This flag is set to true if a topple happened this round
    stop = 0
    while topple:
        stop += 1
        topple = False
        for g, V in G.nodes(data="variables"):
            num_vars = len(V)

            # If you're too full let's topple/chip fire/sand pile
            if num_vars > unit_tile_capacity:
                topple = True

                # Which neighbor do you send it to?
                neighbors_capacity = {
                    v: len(G.nodes[v]["variables"]) for v in G[g]}

                # Who's closest?
                positions = {v: modified_layout[v]
                             for v in G.nodes[g]["variables"]}

                while num_vars > unit_tile_capacity:
                    # The neighbor to send it to
                    lowest_occupancy = min(neighbors_capacity.values())
                    hungriest = random.choice(
                        [v for v, cap in neighbors_capacity.items() if cap == lowest_occupancy])

                    # Who to send
                    min_score = float("inf")
                    for v, pos in positions.items():
                        dist = np.linalg.norm(np.array(hungriest) - pos)
                        score = dist + moves[v]
                        if score < min_score:
                            min_score = min(score, min_score)
                            moves[v] += 1
                            food = v

                    G.nodes[g]["variables"].remove(food)
                    del positions[food]
                    G.nodes[hungriest]["variables"].add(food)

                    neighbors_capacity[hungriest] += 1
                    num_vars -= 1

        if stop == 1000:
            raise RuntimeError(
                "I couldn't topple, this may be an infinite loop.")

    placement = defaultdict(set)
    for g, V in G.nodes(data="variables"):
        V = list(V)
        # Run through the sorted points and assign them to qubits--find a transveral in each unit cell.
        for k in np.argsort([modified_layout[v] for v in V], 0):
            # The x and y order in the argsort (k_* in [0,1,2,3])
            k_x, k_y = k[0], k[1]
            # The vertices of S at those indicies
            u_x, u_y = V[k_x], V[k_y]
            placement[u_x].add((g[0], g[1], 0, k_x))
            placement[u_y].add((g[0], g[1], 1, k_y))

    return dnx_utils.relabel_chains(T_layout.G, placement)

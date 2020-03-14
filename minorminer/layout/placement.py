import random
import warnings
from collections import Counter, defaultdict
from itertools import cycle, product

import networkx as nx
import numpy as np
from . import layout
from .utils import (dnx_utils, graph_utils, layout_utils,
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
    for u, u_pos in S_layout.items():
        distances, v_indices = tree.query(u_pos, num_neighbors)
        placement[u] = placement_utils.minimize_overlap(
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
    # Standardize input
    T_layout = placement_utils.parse_T(T)

    # Raise exceptions if you need to
    placement_utils.check_requirements(S_layout, T_layout)

    # Relabel the vertices from S and T in case of name conflict; S --> 0 and T --> 1.
    X = nx.Graph()
    X.add_edges_from(
        (
            ((0, u), (1, v), dict(weight=distance.euclidean(u_pos, v_pos)))
            for (u, u_pos), (v, v_pos) in product(S_layout.items(), T_layout.items())
        )
    )
    M = nx.bipartite.minimum_weight_full_matching(
        X, ((0, u) for u in S_layout))

    return {u: [M[(0, u)][1]] for u in S_layout.keys()}


def intersection(S_layout, T, fill_processor=True, **kwargs):
    """
    Map each vertex of S to its nearest row/column intersection qubit in T (T must be a D-Wave hardware graph). 

    Parameters
    ----------
    S_layout : layout.Layout
        A layout for S; i.e. a map from S to R^d.
    T : layout.Layout or dwave-networkx.Graph
        A layout for T; i.e. a map from T to R^d. Or a D-Wave networkx graph to make a layout from.
    fill_processor : bool (default True)
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
        S_layout, T_layout, allowed_dnx_graphs="chimera", allowed_dims=2)

    # Scale the layout so that for each vertical and horizontal qubit that cross each other, we have an integer point.
    m, n, t = dnx_utils.lookup_dnx_dims(T_layout.G)

    # Make the "intersection graph" of the dnx_graph
    # Grid points correspond to intersection rows and columns of the dnx_graph
    G = nx.grid_2d_graph(t*m, t*n)

    # Determine the scale for putting things in the positive quadrant
    scale = (t*max(n, m)-1)/2

    # Get the grid graph and the modified layout for S
    _intersection_binning(
        S_layout, T_layout, G, scale, t, fill_processor)

    placement = {}
    for _, data in G.nodes(data=True):
        for v in data["variables"]:
            placement[v] = data["qubits"]

    return placement


def _intersection_binning(S_layout, T_layout, G, scale, t, fill_processor):
    """
    Map the vertices of S to the "intersection graph" of T. This modifies the grid graph G by assigning vertices from S 
    and T to vertices of G.

    Parameters
    ----------
    S_layout : layout.Layout
        A layout for S; i.e. a map from S to R^d.
    T_layout : layout.Layout
        A layout for T; i.e. a map from T to R^d.
    G : networkx.Graph
        A grid_2d_graph representing the lattice points in the positive quadrant.
    scale : float
        The scale necessary to translate (and/or resize) the layouts so that they occupy the positive quadrant.
    t : int
        Number of shores in T.
    fill_processor : bool
        If True, S_layout is scaled so that it fills the processor. If False, the scale of S_layout is used.

    Returns
    -------
    modified_layout : dict
        The layout of S after translating and scaling to the positive quadrant. 
    """
    # Get the row, column mappings for the dnx graph
    lattice_mapping = dnx_utils.lookup_intersection_coordinates(T_layout.G)

    # Less efficient, but more readable to initialize all at once
    for v in G:
        G.nodes[v]["qubits"] = set()
        G.nodes[v]["variables"] = set()

    # Add qubits (vertices of T) to grid points
    for int_point, Q in lattice_mapping.items():
        G.nodes[int_point]["qubits"] |= Q

    # --- Map the S_layout to the grid
    # D-Wave counts the y direction like matrix rows; inversion makes pictures match
    modified_layout = layout.invert_layout(
        S_layout.layout_array, S_layout.center)

    # "Zoom in" on layout_S so that the integer points are better represented
    zoomed_layout = layout.scale_layout(
        modified_layout, t*S_layout.scale, S_layout.scale, S_layout.center)

    # Check if the zoomed S_layout is too large for T.
    # If it is, scale it so that it fills the processor.
    fill_processor = fill_processor or np.any(
        np.abs(zoomed_layout) > scale)

    # Scale S to fill T at the grid level
    if fill_processor:
        modified_layout = layout.scale_layout(
            modified_layout, scale, S_layout.scale, S_layout.center)
    else:
        modified_layout = layout.scale_layout(
            modified_layout, t*S_layout.scale, S_layout.scale, S_layout.center)

    # Center to the positive orthant
    modified_layout = layout.center_layout(
        modified_layout, 2*(scale,), S_layout.center)

    # Turn it into a dictionary
    modified_layout = {v: pos for v, pos in zip(S_layout.G, modified_layout)}

    # Add "variables" (vertices from S) to grid points too
    for v, pos in modified_layout.items():
        grid_point = tuple(int(x) for x in np.round(pos))
        G.nodes[grid_point]["variables"].add(v)

    return modified_layout


def binning(S_layout, T, unit_tile_capacity=None, fill_processor=False, strategy="layout", **kwargs):
    """
    Map the vertices of S to the vertices of T by first mapping both to an integer lattice (T must be a D-Wave hardware graph). 

    Parameters
    ----------
    S_layout : layout.Layout
        A layout for S; i.e. a map from S to R^d.
    T : layout.Layout or dwave-networkx.Graph
        A layout for T; i.e. a map from T to R^d. Or a D-Wave networkx graph to make a layout from.
    unit_tile_capacity : int (default None)
        The number of variables (vertices of S) that are allowed to map to unit tiles of T. If set, a topple based algorithm is run
        to ensure that not too many variables are contained in the same unit tile of T.
    fill_processor : bool (default False)
        If True, S_layout is scaled so that it fills the processor. If False, the scale of S_layout is used.
    strategy : str (default "layout")
        layout : Use S_layout to determine the mapping from variables to qubits.
        cycle : Cycle through the variables and qubits in a unit cell and assign variables to qubits one at a time, repeating if necessary.
        all : Map each variable to each qubit in a unit cell. Lots of overlap.

    Returns
    -------
    placement : dict
        A mapping from vertices of S (keys) to vertices of T (values).
    """
    # Standardize input
    T_layout = placement_utils.parse_T(T, disallow="dict")

    # Raise exceptions if you need to
    placement_utils.check_requirements(
        S_layout, T_layout, allowed_dnx_graphs=["chimera", "pegasus"], allowed_dims=2)

    # Get the lattice point mapping for the dnx graph
    m, n, t = dnx_utils.lookup_dnx_dims(T_layout.G)

    # Make the grid "quotient" of the dnx_graph
    # Quotient the ~K_4,4 unit cells of the dnx_graph to grid points
    G = nx.grid_2d_graph(m, n)

    # Determine the scale for putting things in the positive quadrant
    scale = (max(n, m)-1)/2

    # Check if the S_layout is too large for T.
    # If it is, scale it so that it fills the processor.
    fill_processor = fill_processor or np.any(
        np.abs(S_layout.layout_array) > scale)

    # Get the grid graph and the modified layout for S
    modified_S_layout = _unit_cell_binning(
        S_layout, T_layout, G, scale, fill_processor)

    # Do we need to topple?
    if unit_tile_capacity or strategy == "layout":
        unit_tile_capacity = unit_tile_capacity or t
        n, N = len(S_layout), m*n*unit_tile_capacity
        if n > N:
            raise RuntimeError(
                "You're trying to fit {} vertices of S into {} spots of T.".format(n, N))
        _topple(G, modified_S_layout, unit_tile_capacity)

    # Build the placement
    placement = defaultdict(set)
    if strategy == "layout":
        for g, V in G.nodes(data="variables"):
            V = list(V)

            x_indices, y_indices = list(range(t)), list(range(t))
            for _ in range(t, len(V), -1):
                x_indices.remove(random.choice(x_indices))
                y_indices.remove(random.choice(y_indices))

            # Run through the sorted points and assign them to qubits--find a transveral in each unit cell.
            for k in np.argsort([modified_S_layout[v] for v in V], 0):
                # The x and y order in the argsort (k_* in [0,1,2,3])
                k_x, k_y = k[0], k[1]
                # The vertices of S at those indicies
                u_x, u_y = V[k_x], V[k_y]
                placement[u_x].add((g[1], g[0], 0, x_indices[k_x]))
                placement[u_y].add((g[1], g[0], 1, y_indices[k_y]))
        placement = dnx_utils.relabel_chains(T_layout.G, placement)

    if strategy == "cycle":
        for _, data in G.nodes(data=True):
            if data["variables"]:
                for v, q in zip(data["variables"], cycle(data["qubits"])):
                    placement[v].add(q)

    elif strategy == "all":
        for _, data in G.nodes(data=True):
            if data["variables"]:
                for v in data["variables"]:
                    placement[v] |= data["qubits"]

    return placement


def _topple(G, modified_layout, unit_tile_capacity):
    """
    Modifies G by toppling.

    topple : Topple grid points so that the number of variables at each point does not exceed specified unit_tile_capacity.
            After toppling assign the vertices of S to a transversal of qubits at the grid point.
    """
    # Check who needs to move (at most unit_tile_capacity vertices of S allowed per grid point)
    moves = {v: 0 for _, V in G.nodes(data="variables") for v in V}
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


def _unit_cell_binning(S_layout, T_layout, G, scale, fill_processor):
    """
    Map the vertices of S to the unit cell quotient of T. This modifies the grid graph G by assigning vertices from S 
    and T to vertices of G.

    Parameters
    ----------
    S_layout : layout.Layout
        A layout for S; i.e. a map from S to R^d.
    T_layout : layout.Layout
        A layout for T; i.e. a map from T to R^d.
    G : networkx.Graph
        A grid_2d_graph representing the lattice points in the positive quadrant.
    scale : float
        The scale necessary to translate (and/or resize) the layouts so that they occupy the positive quadrant.
    fill_processor : bool
        If True, S_layout is scaled so that it fills the processor. If False, the scale of S_layout is used.

    Returns
    -------
    modified_layout : dict
        The layout of S after translating and scaling to the positive quadrant. 
    """
    # Get the lattice point mapping for the dnx graph
    lattice_mapping = dnx_utils.lookup_grid_coordinates(T_layout.G)

    # Less efficient, but more readable to initialize all at once
    for v in G:
        G.nodes[v]["qubits"] = set()
        G.nodes[v]["variables"] = set()

    # Add qubits (vertices of T) to grid points
    for v, int_point in lattice_mapping.items():
        G.nodes[int_point]["qubits"].add(v)

    # --- Map the S_layout to the grid
    # D-Wave counts the y direction like matrix rows; inversion makes pictures match
    modified_layout = layout.invert_layout(
        S_layout.layout_array, S_layout.center)

    # Scale S to fill T at the grid level
    if fill_processor:
        modified_layout = layout.scale_layout(
            modified_layout, scale, S_layout.scale, S_layout.center)

    # Center to the positive orthant
    modified_layout = layout.center_layout(
        modified_layout, 2*(scale,), S_layout.center)

    # Turn it into a dictionary
    modified_layout = {v: pos for v, pos in zip(S_layout.G, modified_layout)}

    # Add "variables" (vertices from S) to grid points too
    for v, pos in modified_layout.items():
        grid_point = tuple(int(x) for x in np.round(pos))
        G.nodes[grid_point]["variables"].add(v)

    return modified_layout

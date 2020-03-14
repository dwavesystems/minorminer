import math
from collections import defaultdict

import dwave_networkx as dnx
import networkx as nx


def lookup_grid_coordinates(G):
    """
    For a dwave_networkx graph G, this returns a dictionary mapping the vertices of G to lattice points. 
    - Chimera: This is the first 2 coordinates (of the coordinate version) from each vertex.
    - Pegasus: This is a more complicated mapping where nice_coordinates go to (t, y, x, u, k) |--> (3x + t, 3y + 2 - t)
    """
    graph_data = G.graph

    # Look to see if you can get the lattice information from the graph object
    family = graph_data.get("family")
    if family == "chimera":
        if graph_data["labels"] == "coordinate":
            return {v: (v[1], v[0]) for v in G}
        if graph_data["data"]:
            return {
                v: (G.nodes[v]["chimera_index"][1],
                    G.nodes[v]["chimera_index"][0])
                for v in G
            }
    elif family == "pegasus":
        m = graph_data["rows"]
        C = dnx.pegasus_coordinates(m)

        if graph_data["labels"] == "int":
            nice_coords = C.iter_linear_to_nice(G.nodes)
        elif graph_data["labels"] == "coordinate":
            nice_coords = C.iter_pegasus_to_nice(G.nodes)
        elif graph_data["labels"] == "nice":
            nice_coords = G.nodes

        # FIXME: x and y might be switched here
        # (t, y, x, u, k) |--> (3x + t, 3y + 2 - t)
        return {v: (3*v[2] + v[0], 3*v[1] + (2 - v[0])) for v in nice_coords}
    return None


def lookup_intersection_coordinates(G):
    """
    For a dwave_networkx graph G, this returns a dictionary mapping the lattice points to sets of vertices of G. 
    - Chimera: Each lattice point corresponds to the 2 qubits intersecting at that point.
    - Pegasus: Not Implemented
    """
    graph_data = G.graph

    family = graph_data.get("family")
    t = graph_data.get("tile")

    if family == "chimera":
        intersection_points = defaultdict(set)
        if graph_data["labels"] == "coordinate":
            for v in G:
                _chimera_all_intersection_points(intersection_points, v, t, *v)

        elif graph_data["data"]:
            for v in G:
                _chimera_all_intersection_points(
                    intersection_points, v, t, *G.nodes[v]["chimera_index"])

        return intersection_points

    elif family == "pegasus":
        raise NotImplementedError("Pegasus forthcoming.")
    return None


def _chimera_all_intersection_points(intersection_points, v, t, i, j, u, k):
    """
    Given a coordinate version of a Chimera vertex, get all intersection points it is in.
    """
    # If you're a row vertex, you go in all grid points of your row intersecting columns in your unit tile
    if u == 1:
        row = i*t + k
        for kk in range(t):
            col = j*t + kk
            intersection_points[(col, row)].add(v)

    # Sameish for a column vertex.
    elif u == 0:
        col = j*t + k
        for kk in range(t):
            row = i*t + kk
            intersection_points[(col, row)].add(v)


def lookup_dnx_dims(G):
    """
    Checks to see if G is a dnx.*_graph(). If it is, return the number of rows, columns, and shores.
    """
    graph_data = G.graph
    if graph_data.get("family") in ("chimera", "pegasus"):
        return graph_data["rows"], graph_data["columns"], graph_data["tile"]

    return None


def nx_to_dnx_layout(center, scale):
    """
    This function translates a center and a scale from the networkx convention, [center - scale, center + scale]^d,
    to the dwave_networkx convention, [center, center-scale] x [center, center+scale]^(d-1).

    Returns
    -------
    top_left : float
        The top left corner of a layout.
    new_scale : float
        This is twice the original scale.
    """
    top_left = (center[0] - scale, ) + tuple(x + scale for x in center[1:])
    new_scale = 2*scale

    return top_left, new_scale


def relabel_chains(G, chains):
    """
    Checks if the labeling of G matches the labeling of chains. If it does not, it returns chains that match.
    """
    if G.graph["labels"] == "coordinate":
        return chains
    else:
        m, n, t = lookup_dnx_dims(G)
        C = dnx.chimera_coordinates(m, n, t)
        return {v: [C.chimera_to_linear(q) for q in Q] for v, Q in chains.items()}

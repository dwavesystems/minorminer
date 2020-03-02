import math


def lookup_dnx_coordinates(G):
    """
    Checks to see if G is a dnx.*_graph(). If it is, it checks to see if G has coordinate information. If it does it 
    returns a dictionary mapping the vertices of G to lattice points, i.e., the first 2 coordinates from each vertex 
    extended to d-dimensional space.
    """
    graph_data = G.graph

    # Look to see if you can get the lattice information from the graph object
    family = graph_data.get("family")
    if family == "chimera":
        if graph_data["labels"] == "coordinate":
            return {v: (v[0], v[1]) for v in G}
        if graph_data["data"]:
            return {
                v: (G.nodes[v][f"{family}_index"][0],
                    G.nodes[v][f"{family}_index"][1])
                for v in G
            }
    elif family == "pegasus":
        if graph_data["labels"] == "nice":
            # (t, y, x, u, k) |--> (3x + t, 3y + 2 - t)
            return {v: (3*v[2] + v[0], 3*v[1] + (2 - v[0])) for v in G}
    return None


def lookup_dnx_dims(G):
    """
    Checks to see if G is a dnx.*_graph(). If it is, return the number of rows, columns, and shores.
    """
    graph_data = G.graph
    if graph_data.get("family") in ("chimera", "pegasus"):
        return graph_data["rows"], graph_data["columns"], graph_data["tile"]

    return None


def get_row_or_column(p, t):
    """
    Given a number from [0, t*m-1] or [0, t*n-1] where m is the number of rows, n is the number of columns, and t is the 
    shore size, return the rounded number, the index of the chimera cell, and the k-value of the qubit.
    """
    # Round each position to the nearest integer (qubit)
    r_p = int(round(p))

    # Compute which unit cell it's going to end up in
    cell_index = math.floor(r_p/t)

    # Compute the k-value of the qubit in the unit cell
    k = int(r_p - t*cell_index)

    # Get the coordinate version of the qubit
    return r_p, cell_index, k


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

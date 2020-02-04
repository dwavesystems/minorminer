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
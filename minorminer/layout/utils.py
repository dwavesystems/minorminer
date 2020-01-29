import networkx as nx


def parse_graph(G):
    """
    Determines if a graph object or a collection of edges was passed in. Returns a NetworkX graph.
    """
    if hasattr(G, "edges"):
        return G
    return nx.Graph(G)


def check_dnx(G, needs_data=False):
    """
    Determines if a graph object is a dwave_networkx graph object with data=True.

    Parameters
    ----------
    G : NetworkX graph
        The graph to check if it is a dwave_networkx graph.

    Returns
    -------
    n : int
        The number of rows in the D-Wave graph.
    m : int
        The number of columns in the D-Wave graph.
    k : int
        The size of the shores in the D-Wave graph.
    """
    assert G.graph.get("family") in ("chimera", "pegasus"), (
        "If using D-Wave specific functions you must pass in a dnx.*_graph.")
    if needs_data:
        assert G.graph["data"] is True, "The parameter data=True is required for this function."

    return G.graph["rows"], G.graph["columns"], G.graph["tile"]

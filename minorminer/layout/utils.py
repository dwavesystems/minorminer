# TODO: Add logic to determine if a D-Wave hardware graph.
def parse_graph(G):
    """
    Determines if a graph object or a collection of edges was passed in. Returns a NetworkX graph.
    """
    if hasattr(G, "edges"):
        return G
    return nx.Graph(G)

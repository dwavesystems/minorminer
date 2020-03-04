import networkx as nx
from minorminer.layout.layout import Layout, dnx_layout


def parse_layout(layout):
    """
    Take in a layout class object or a dictionary and return the dictionary representation.
    """
    if isinstance(layout, Layout):
        return layout.layout
    else:
        return layout


def parse_T(T, disallow=None):
    if isinstance(T, nx.Graph) and disallow != "graph":
        return dnx_layout(T)
    elif isinstance(T, Layout) and disallow != "layout":
        return T
    elif isinstance(T, dict) and disallow != "dict":
        return T
    else:
        raise TypeError("Why did you give me that?")


def check_requirements(S_layout, T_layout, disallowed_graphs=None):
    if disallowed_graphs is None:
        disallowed_graphs = []
    elif isinstance(disallowed_graphs, str):
        disallowed_graphs = [disallowed_graphs]
    elif isinstance(disallowed_graphs, (frozenset, list, set, tuple)):
        pass
    else:
        raise TypeError("What did you give me for a disallowed graph?")

    graph_type = T_layout.G.graph.get("family")
    if graph_type in disallowed_graphs:
        raise NotImplementedError(
            f"This placement strategy is currently not implemented for graphs of type {graph_type}.")
    if (not isinstance(S_layout, Layout)) or (not isinstance(T_layout, Layout)):
        raise TypeError("This placement strategy needs Layout class objects.")
    if S_layout.d != 2 or T_layout.d != 2:
        raise NotImplementedError(
            "This placement strategy is only implemented for 2-dimensional layouts.")

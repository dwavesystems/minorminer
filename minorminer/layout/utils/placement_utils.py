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


def check_requirements(S_layout, T_layout, allowed_graphs=None, allowed_dims=None):
    if len(S_layout.G) > len(T_layout.G):
        raise RuntimeError("S is larger than T. You cannot embed S in T.")

    # Datatype parsing
    if allowed_graphs is None:
        allowed_graphs = True
    elif isinstance(allowed_graphs, str):
        allowed_graphs = [allowed_graphs]
    elif isinstance(allowed_graphs, (frozenset, list, set, tuple)):
        pass
    else:
        raise TypeError("What did you give me for an allowed graph?")

    # Datatype parsing
    if allowed_dims is None:
        allowed_dims = True
    elif isinstance(allowed_dims, int):
        allowed_dims = [allowed_dims]
    elif isinstance(allowed_dims, (frozenset, list, set, tuple)):
        pass
    else:
        raise TypeError("What did you give me for an allowed dimension?")

    graph_type = T_layout.G.graph.get("family")
    if allowed_graphs is True or graph_type not in allowed_graphs:
        raise NotImplementedError(
            f"This strategy is currently only implemented for graphs of type {allowed_graphs}.")

    if (not isinstance(S_layout, Layout)) or (not isinstance(T_layout, Layout)):
        raise TypeError("This strategy needs Layout class objects.")

    if allowed_dims is True or (S_layout.d not in allowed_dims or T_layout.d not in allowed_dims):
        raise NotImplementedError(
            f"This strategy is only implemented for {allowed_dims}-dimensional layouts.")

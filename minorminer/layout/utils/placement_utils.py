import networkx as nx

from minorminer.layout import layout


def parse_layout(layout):
    """
    Take in a layout class object or a dictionary and return the dictionary representation.
    """
    if isinstance(layout, layout.Layout):
        return layout.layout
    else:
        return layout


def parse_T(T, disallow=None):
    if isinstance(T, nx.Graph) and disallow != "graph":
        return layout.dnx_layout(T)
    elif isinstance(T, layout.Layout) and disallow != "layout":
        return T
    elif isinstance(T, dict) and disallow != "dict":
        return T
    else:
        raise TypeError("Why did you give me that?")

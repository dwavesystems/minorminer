import networkx as nx
import numpy as np


def kamada_kawai(G, d=2, **kwargs):
    """
    The d-dimensional Kamada-Kawai spring layout. Axes are [0, 1] for all components except for the second component--it
    is [0, -1]. This adhears to the dwave_networkx.draw_chimera() convention.

    Parameters
    ----------
    G : NetworkX graph
        The graph you want to compute the layout for.
    d : int (default 2)
        The dimension of the kamada_kawai layout.
    kwargs : dict
        Keyword arguments are passed to nx.kamada_kawai_layout().

    Returns
    -------
    layout : dict
        A mapping from vertices of G (keys) to points in R^d (values).
    """
    # NetworkX has a bug #3658.
    # Once fixed, these can collapse and `dim=n` can be part of `**kwargs`.

    # Extract the seed in case its there for random fixing
    random_kwargs = {}
    if "seed" in kwargs:
        random_kwargs["seed"] = kwargs["seed"]
        del kwargs["seed"]

    if d in (1, 2):
        return nx.kamada_kawai_layout(G, dim=d, **kwargs)
    else:
        # The random_layout is in [0, 1]^d
        random_layout = nx.random_layout(G, dim=d, **random_kwargs)
        # Convert it to [-1, 1]^d
        std_random_layout = standardize_coordinates(random_layout, "random")

        return nx.kamada_kawai_layout(
            G, pos=std_random_layout, dim=d, **kwargs)


def standardize_coordinates(layout, layout_type):
    """
    Most of networkx layouts return points in [-1, 1]^d. This is a helper function to standardize to this convention.
    """
    if layout_type == "random":
        return {v: list(map(lambda x: 2*x - 1, p)) for v, p in layout.items()}
    else:
        raise TypeError(f"The layout_type {layout_type} is not supported.")

import networkx as nx


def kamada_kawai(G, d=2, **kwargs):
    """
    The d-dimensional Kamada-Kawai spring layout.

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
    if d in (1, 2):
        return nx.kamada_kawai_layout(G, dim=d, **kwargs)
    else:
        return nx.kamada_kawai_layout(
            G, center=d*(1/2, ), scale=1/2, pos=nx.random_layout(G, dim=d), dim=d, **kwargs
        )

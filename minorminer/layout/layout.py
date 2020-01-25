import networkx as nx
import numpy as np

import dwave_networkx as dnx


def kamada_kawai(G, d=2, center=None, scale=1., seed=None, **kwargs):
    """
    The d-dimensional Kamada-Kawai spring layout.

    Parameters
    ----------
    G : NetworkX graph
        The graph you want to compute the layout for.
    d : int (default 2)
        The dimension of the kamada_kawai layout.
    center : tuple (default None)
        The center point of the layout; if None it is computed to be the origin in R^d.
    scale : float (default 1.)
        The scale of the layout; i.e. the layout is in [center - scale, center + scale]^d space.
    kwargs : dict
        Keyword arguments are passed to nx.kamada_kawai_layout().

    Returns
    -------
    layout : dict
        A mapping from vertices of G (keys) to points in R^d (values).
    """
    # Set center to a default value based on d.
    center = center or d*(0, )

    # NetworkX has a bug #3658.
    # Once fixed, these can collapse and `dim=n` can be part of `**kwargs`.
    if d in (1, 2):
        return nx.kamada_kawai_layout(G, dim=d, center=center, scale=scale, **kwargs)
    else:
        # The random_layout is in [0, 1]^d
        random_layout = nx.random_layout(G, dim=d, seed=seed)

        # Convert it to [center - scale, center + scale]^d
        scaled_random_layout = scale_random_layout(
            random_layout, center, scale)

        return nx.kamada_kawai_layout(
            G, pos=scaled_random_layout, dim=d, center=center, scale=scale, **kwargs)


def chimera(G, d=2, center=None, scale=1., **kwargs):
    """
    The d-dimensional Chimera layout adjusted so that it fills [-1, 1]^2 instead of [0, 1] x [0, -1]. As per the 
    implementation of dnx.chimera_layout() in layouts with d > 2, coordinates beyond the second are 0.

    Parameters
    ----------
    G : NetworkX graph
        A Chimera graph you want to compute the layout for.
    d : int (default 2)
        The dimension of the layout.
    kwargs : dict
        Keyword arguments are passed to dnx.chimera_layout().

    Returns
    -------
    layout : dict
        A mapping from vertices of G (keys) to points in [-1, 1]^d (values).
    """
    # Set center to a default value based on d.
    center = center or d*(0, )

    # Convert center and scale for dwave_networkx to consume.
    top_left, new_scale = center_to_top_left(center, scale)

    return dnx.chimera_layout(G, dim=d, center=top_left, scale=new_scale, **kwargs)


def scale_random_layout(random_layout, center, scale):
    """
    The function networkx.random_layout() maps to [0, 1]^d. This helper function transforms this layout to the user
    desired [center - scale, center + scale]^d.
    """
    # Temporary data structure to pull the information out of the matrix created below.
    vertices = [v for v in random_layout]

    # Make a matrix of the positions
    L = np.array([random_layout[v] for v in vertices])
    # Map it from [0, 1]^d to [0, 2*scale]^d
    L = (2*scale) * L
    # Shift it to [center - scale, center + scale]^d
    L = L + (center - scale)

    return {vertices[i]: p for i, p in enumerate(L)}


def center_to_top_left(center, scale):
    """
    This function translates a center and a scale from the networkx convention, [center - scale, center + scale]^d, to
    the dwave_networkx convention, [center, center-scale] x [center, center+scale]^(d-1).

    Parameters
    ----------
    center : tuple
        A d-dimensional tuple representing the center of a layout.
    scale : float
        A scalar to add to and subtract from the center to create the layout.
    d : int
        The dimension of the layout.

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

import statistics
from itertools import combinations

import dwave_networkx as dnx
import networkx as nx
import numpy as np
from scipy.spatial.distance import euclidean


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
    seed : int (default None)
        When d > 2, networkx.random_layout() is called. The seed is passed to this function.
    kwargs : dict
        Keyword arguments are passed to nx.kamada_kawai_layout().

    Returns
    -------
    layout : dict
        A mapping from vertices of G (keys) to points in R^d (values).
    """
    # Set center to a default value based on d.
    center = center or d*(0, )
    center = np.array(center)

    # NetworkX has a bug #3658.
    # Once fixed, these can collapse and `dim=n` can be part of `**kwargs`.
    if d in (1, 2):
        return nx.kamada_kawai_layout(G, dim=d, center=center, scale=scale, **kwargs)
    else:
        # The random_layout is in [0, 1]^d
        random_layout = nx.random_layout(G, dim=d, seed=seed)

        # Convert it to [center - scale, center + scale]^d
        transformed_random_layout = scale_unit_layout(
            random_layout, center, scale)

        return nx.kamada_kawai_layout(
            G, pos=transformed_random_layout, dim=d, center=center, scale=scale, **kwargs)


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
    center : tuple (default None)
        The center point of the layout; if None it is computed to be the origin in R^d.
    scale : float (default 1.)
        The scale of the layout; i.e. the layout is in [center - scale, center + scale]^d space.
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


def scale_unit_layout(unit_layout, center, scale):
    """
    The function networkx.random_layout() maps to [0, 1]^d. This helper function transforms this layout to the user
    desired [center - scale, center + scale]^d.

    Parameters
    ----------
    unit_layout : dict
        A mapping from vertices of G (keys) to points in [0, 1]^d (values).
    center : tuple
        A d-dimensional tuple representing the center of a layout.
    scale : float
        A scalar to add to and subtract from the center to create the layout.

    Returns
    -------
    layout : dict
        A mapping from vertices of G (keys) to points in [center - scale, center + scale]^d (values).
    """
    # Temporary data structure to pull the information out of the matrix created below.
    vertices = [v for v in unit_layout]

    # Make a matrix of the positions
    L = np.array([unit_layout[v] for v in vertices])
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


def scale_edge_length(layout, edge_length=1., to_scale="median"):
    """
    Scale a layout so that the specified to_scale type obtains the desired edge_length.

    Parameters
    ----------
    layout : dict
        A layout of a graph G; i.e. a map from G to the plane.
    edge_length : float (default 1.)
        The desired edge_length to achieve.
    to_scale : string (default "median")
        Different types of measures that will match the specified edge_length. E.g. if "median" the median of the edge
        lengths in layout is guarenteed to become the desired edge_length.

    Returns
    -------
    scaled_layout : dict
        The input layout scaled so that the parameter to_scale is the parameter edge_length.
    """
    distances = {(u, v): euclidean(u_pos, v_pos)
                 for (u, u_pos), (v, v_pos) in combinations(layout.items(), 2)}

    if to_scale == "median":
        scale = edge_length/statistics.median(distances.values())
    elif to_scale == "min":
        scale = edge_length/min(distances.values())
    elif to_scale == "max":
        scale = edge_length/max(distances.values())
    else:
        raise ValueError(
            "Parameter to_scale={} is not supported.".format(to_scale))

    return {v: scale*p for v, p in layout.items()}

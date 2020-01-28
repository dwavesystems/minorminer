import statistics
from collections import defaultdict
from itertools import combinations

import dwave_networkx as dnx
import networkx as nx
import numpy as np
from minorminer.layout import utils
from scipy.spatial.distance import euclidean


class Layout():
    def __init__(self, G, d=2, center=None, scale=1., layout=None, seed=None, **kwargs):
        """
        Compute a layout for G, i.e., a map from G to R^d.

        Parameters
        ----------
        G : NetworkX graph or NetworkX supported edges data structure (dict, list, ...)
            The graph you want to compute the layout for.
        d : int (default 2)
            The dimension of the layout, i.e., R^d.
        center : tuple (default None)
            The center point of the layout; if None it is computed to be the origin in R^d.
        scale : float (default 1.)
            The scale of the layout; i.e. the layout is in [center - scale, center + scale]^d space.
        layout : dict (default None)
            You can specifies a pre-computed layout for G.
        seed : int (default None)
            When d > 2, kamada_kawai uses networkx.random_layout(). The seed is passed to this function.
        kwargs : dict
            Keyword arguments are passed to one of the layout algorithms below.
        """
        # Ensure G is a graph object
        self.G = utils.parse_graph(G)

        # Construct the origin if need be
        if center is None:
            self.center = np.array(d*(0, ))
        else:
            self.center = np.array(center)

        # Set remaining parameters
        self.d = d
        self.scale = scale
        self.layout = layout
        self.seed = seed

    def kamada_kawai(self, **kwargs):
        """
        The d-dimensional Kamada-Kawai spring layout.

        Parameters
        ----------
        seed : int (default None)
            When d > 2, networkx.random_layout() is called. The seed is passed to this function.
        kwargs : dict
            Keyword arguments are passed to nx.kamada_kawai_layout().

        Returns
        -------
        layout : dict
            A mapping from vertices of G (keys) to points in R^d (values).
        """
        # NetworkX has a bug #3658.
        # Once fixed, these can collapse and `dim=n` can be part of `**kwargs`.
        if self.d in (1, 2):
            layout = nx.kamada_kawai_layout(
                self.G, dim=self.d, center=self.center, scale=self.scale, **kwargs)
        else:
            # The random_layout is in [0, 1]^d
            random_layout = nx.random_layout(
                self.G, dim=self.d, seed=self.seed)

            # Convert it to [center - scale, center + scale]^d
            transformed_random_layout = self.scale_unit_layout(random_layout)

            layout = nx.kamada_kawai_layout(
                self.G, pos=transformed_random_layout, dim=self.d, center=self.center, scale=self.scale, **kwargs)

        self.layout = layout
        return layout

    def chimera(self, **kwargs):
        """
        The d-dimensional Chimera layout adjusted so that it fills [-1, 1]^2 instead of [0, 1] x [0, -1]. As per the
        implementation of dnx.chimera_layout() in layouts with d > 2, coordinates beyond the second are 0.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments are passed to dnx.chimera_layout().

        Returns
        -------
        layout : dict
            A mapping from vertices of G (keys) to points in [-1, 1]^d (values).
        """
        # Convert center and scale for dwave_networkx to consume.
        top_left, new_scale = self.center_to_top_left()

        layout = dnx.chimera_layout(
            self.G, dim=self.d, center=top_left, scale=new_scale, **kwargs)

        self.layout = layout
        return layout

    def to_integer_lattice(self, lattice_points=3, points_as_keys=False):
        """
        Map the vertices in a layout to their closest integer points.

        Parameters
        ----------
        lattice_points : int or tuple (default 3)
            The number of lattice points in each dimension. If it is an integer, there will be that many lattice points
            in each dimension of the layout. If it is a tuple, each entry specifies how many lattice points are in each
            dimension in the layout.
        points_as_keys : bool (default False)
            If False, vertices are keys and points in Z^d are values. If True, points in Z^d are keys and lists of 
            vertices are values.
        """
        scaled_layout = self.scale_to_positive_orthant(lattice_points)

        if not points_as_keys:
            return {v: tuple(round(x) for x in p) for v, p in scaled_layout.items()}

        integer_point_map = defaultdict(list)
        for v, p in scaled_layout.items():
            integer_point_map[tuple(round(x) for x in p)].append(v)

        return integer_point_map

    def scale_to_positive_orthant(self, length=1):
        """
        This helper function transforms the layout [center - scale, center + scale]^d to the positive orthant
        [0, length[0]] x [0, length[1]] x ... x [0, length[d-1]].

        Parameters
        ----------
        length : int or tuple (default 1)
            The maximum value in each dimension. If it is an integer, this is the max for all dimensions; if it is a
            tuple, each entry specifies a max for each dimension in the layout.

        Returns
        -------
        layout : dict
            A mapping from vertices of G (keys) to points in
            [0, length[0]] x [0, length[1]] x ... x [0, length[d-1]] (values).
        """
        # Temporary data structure to pull the information out of the matrix created below.
        vertices = [v for v in self.G]

        # Make a matrix of the positions
        L = np.array([self.layout[v] for v in vertices])
        # Shift it from [center - scale, center + scale]^d to [0, 2*scale]^d
        L = L + (self.scale - self.center)
        # Map it to [0, 1]^d
        L = L / (2*self.scale)
        # Scale it to the desired length
        scale = _scale_vector(length, self.d)
        L = L * scale

        return {vertices[i]: p for i, p in enumerate(L)}

    def scale_unit_layout(self, unit_layout):
        """
        The function networkx.random_layout() maps to [0, 1]^d. This helper function transforms this layout to the user
        desired [center - scale, center + scale]^d.

        Parameters
        ----------
        unit_layout : dict
            A mapping from vertices of G (keys) to points in [0, 1]^d (values).

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
        L = (2*self.scale) * L
        # Shift it to [center - scale, center + scale]^d
        L = L + (self.center - self.scale)

        return {vertices[i]: p for i, p in enumerate(L)}

    def center_to_top_left(self):
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
        top_left = (self.center[0] - self.scale, ) + \
            tuple(x + self.scale for x in self.center[1:])
        new_scale = 2*self.scale

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


def kamada_kawai(G, d=2, center=None, scale=1., seed=None, **kwargs):
    """
    Top level function for minorminer.layout.__init__() use as a parameter.
    # FIXME: There's surely a better way of doing this.
    """
    L = Layout(G, d=d, center=center, scale=scale, seed=seed)
    _ = L.kamada_kawai(**kwargs)
    return L


def chimera(G, d=2, center=None, scale=1., **kwargs):
    """
    Top level function for minorminer.layout.__init__() use as a parameter.
    # FIXME: There's surely a better way of doing this.
    """
    L = Layout(G, d, center, scale)
    _ = L.chimera(**kwargs)
    return L


def _scale_vector(length, d):
    """
    If length is an integer, it creates a d-dimensional array with values length. Otherwise it creates an array based
    on the length iterable and checks that is the same dimension as d. 
    """
    if isinstance(length, int):
        return np.array(d*(length,))

    scale = np.array(length)
    assert scale.size == d, (
        f"You inputed a scale vector of size {scale.size} for a {d}-dimensional space."
    )
    return scale

import statistics
from collections import defaultdict
from itertools import combinations

import dwave_networkx as dnx
import jax
import networkx as nx
import numpy as np
from scipy import optimize
from scipy.spatial.distance import euclidean

from minorminer.layout.utils import dnx_utils, graph_utils, layout_utils


def kamada_kawai(G, d=2, center=None, scale=1., seed=None, rescale=True, rotate=True, **kwargs):
    """
    Top level function for minorminer.layout.__init__() use as a parameter.
    # FIXME: There's surely a better way of doing this.
    """
    L = Layout(G, d=d, center=center, scale=scale,
               seed=seed, rescale=rescale, rotate=rotate)
    _ = L.kamada_kawai(**kwargs)
    return L


def dnx_layout(G, d=2, center=None, scale=1., rescale=True, rotate=True, **kwargs):
    """
    Top level function for minorminer.layout.__init__() use as a parameter.
    # FIXME: There's surely a better way of doing this.
    """
    L = Layout(G, d=d, center=center, scale=scale,
               rescale=rescale, rotate=rotate)
    _ = L.dnx_layout(**kwargs)
    return L


def pca(G, d=2, m=None, pca=True, center=None, scale=1., seed=None, rescale=True, rotate=True, **kwargs):
    """
    Top level function for minorminer.layout.__init__() use as a parameter.
    # FIXME: There's surely a better way of doing this.
    """
    L = Layout(G, d=d, center=center, scale=scale,
               seed=seed, rescale=rescale, rotate=rotate)
    _ = L.pca(m, pca, **kwargs)
    return L


def custom_metric_space(
    G,
    starting_layout=None,
    distance_function=None,
    G_distances=None,
    d=2,
    center=None,
    scale=1.,
    rescale=True,
    rotate=True,
    **kwargs
):
    """
    Top level function for minorminer.layout.__init__() use as a parameter.
    # FIXME: There's surely a better way of doing this.
    """
    L = Layout(G, d=d, center=center, scale=scale,
               rescale=rescale, rotate=rotate)
    _ = L.custom_metric_space(
        starting_layout, distance_function, G_distances, **kwargs)
    return L


class Layout():
    def __init__(self, G, d=2, center=None, scale=1., layout=None, seed=None, rescale=True, rotate=True, **kwargs):
        """
        Compute a layout for G, i.e., a map from G to [center - scale, center + scale]^d.

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
            You can specify a pre-computed layout for G.
        seed : int (default None)
            When d > 2, kamada_kawai uses networkx.random_layout(). The seed is passed to this function.
        rescale : bool (default True)
            If True, the layout is scaled to the user specified [center - scale, center + scale]^d. If False, the layout
            assumes the dimensions of the layout algorithm used.
        rotate : bool (default True)
            If True, the minimum area bounding box for the layout is computed and rotated so that it aligned with the
            x and y axes. If False, the layout is not rotated.
        kwargs : dict
            Keyword arguments are passed to one of the layout algorithms below.
        """
        # Ensure G is a graph object
        self.G = graph_utils.parse_graph(G)

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
        self.rescale = rescale
        self.rotate = rotate

    def kamada_kawai(self, **kwargs):
        """
        The d-dimensional Kamada-Kawai spring layout.

        Parameters
        ----------
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
            transformed_random_layout = self.scale_and_center(
                random_layout, center=self.d*(1/2,), scale=1/2)

            layout = nx.kamada_kawai_layout(
                self.G, pos=transformed_random_layout, dim=self.d, center=self.center, scale=self.scale, **kwargs)

        if self.rotate and self.d == 2:
            layout = self.rotate_layout(layout, center=self.center)
        if self.rescale:
            layout = self.scale_and_center(layout, center=self.center)

        self.layout = layout
        return layout

    def dnx_layout(self, **kwargs):
        """
        The d-dimensional Chimera or Pegasus layout from dwave_networkx. As per the implementation of dnx.*_layout() in
        layouts with d > 2, coordinates beyond the second are 0.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments are passed to dnx.*_layout().

        Returns
        -------
        layout : dict
            A mapping from vertices of G (keys) to points in R^d (values).
        """
        family = self.G.graph.get("family")
        if family not in ("chimera", "pegasus"):
            raise ValueError(
                "Only dnx.chimera_graph() and dnx.pegasus_graph() are supported.")

        # If you are rescalling (recommended) add kwargs for dwave_networkx to consume.
        if self.rescale:
            top_left, new_scale = self.center_to_top_left()
            kwargs["center"] = top_left
            kwargs["scale"] = new_scale

        if family == "chimera":
            layout = dnx.chimera_layout(self.G, dim=self.d, **kwargs)
        elif family == "pegasus":
            layout = dnx.pegasus_layout(self.G, dim=self.d, **kwargs)

        self.layout = layout
        return layout

    def pca(self, m=None, pca=True):
        """
        Embeds a graph in a m-dimensional space and then projects to a d-dimensional space using principal component
        analysis (PCA).

        See http://www.wisdom.weizmann.ac.il/~harel/papers/highdimensionalGD.pdf

        Parameters
        ----------
        m : int (default None)
            The dimension to first embed into. If None, it is computed as the min(G, 50).
        pca : bool (default True)
            Whether or not to project down to d-dimensional space using PCA.

        Returns
        -------
        layout : dict
            A mapping from vertices of G (keys) to points in R^d (values).
        """
        # The number of vertices bounds the dimension
        n = len(self.G)
        assert self.d <= n, f"You want me to find {self.d} eigenvectors in a graph with {n} vertices."

        # Pick the number of dimensions to initially embed into
        m = n if n < 50 else 50

        starting_layout = layout_utils.build_starting_points(
            self.G, m, self.seed)

        if pca:
            # Form the shifted matrix X from the paper.
            L = np.array([starting_layout[v] for v in self.G])
            X = (L - np.mean(L, axis=0)).T
            X_T = X.T

            # Form the covarience matrix
            S = (1/n)*(X @ X_T)

            # Compute the normalized sorted eigenvectors
            _, eigenvectors = np.linalg.eigh(S)

            # Choose the eigenvectors that correspond to the largest k eigenvalues and project in those dimensions
            Y = np.column_stack(
                [X_T @ u for u in list(reversed(eigenvectors))[:self.d]])

            layout = {v: row for v, row in zip(list(self.G), Y)}

        else:
            layout = starting_layout

        # Rotate the layout
        if self.rotate and self.d == 2:
            layout = self.rotate_layout(layout)
        # Scale the layout
        if self.rescale:
            layout = self.scale_and_center(layout)

        self.layout = layout
        return layout

    def custom_metric_space(self, starting_layout, distance_function, G_distances=None, **kwargs):
        """
        Embeds a graph in a custom metric space and minimizes a Kamada-Kawai-esque objective function to achieve
        an embedding with low distortion. This computes a layout where the graph distance and the distance_function are 
        very close to each other.

        Parameters
        ----------
        starting_layout : dict or Numpy Array
            A mapping from the vertices of G to points in the metric space.
        distance_function : function
            The distance function in the metric space to make close to the graph distance. 
        G_distances : dict or Numpy 2d array (default None)
            A dictionary of dictionaries representing distances from every vertex in G to every other vertex in G, or
            a matrix representing the same data. If None, it is computed.

        Returns
        -------
        layout : dict
            A mapping from vertices of G (keys) to points in R^d (values).
        """
        # Pick a random R^2 x T layout
        if starting_layout is None:
            starting_layout = layout_utils.random_layout(self.G)

        # Make sure the layout is a vector
        if isinstance(starting_layout, dict):
            starting_layout = np.array(
                [pos for pos in starting_layout.values()])

        # Save on distance calculations by passing them in
        if G_distances is None:
            G_distances = layout_utils.graph_distances(self.G)

        # Make sure the distances are a matrix
        if isinstance(G_distances, dict):
            G_distances = np.array(
                [[V[v] for v in self.G] for u, V in G_distances.items()]
            )

        # Pick the R^2 x T distance function
        if distance_function is None:
            distance_function = layout_utils.R2xT_distance

        # Get the dimension of the layout
        k = starting_layout.shape[1]

        # Use automatic differentiation to compute the gradient
        if "jac" not in kwargs:
            kwargs["jac"] = jax.grad(layout_utils.cost_function)

        # Solve the Kamada-Kawai-esque minimization function
        X = optimize.minimize(
            layout_utils.cost_function,
            starting_layout,
            args=(G_distances, distance_function, k),
            **kwargs
        )

        # Reshape the solution and convert to dictionary.
        layout = {v: pos for v, pos in zip(
            self.G, X.x.reshape(len(self.G), k))}

        # Rotate the layout
        if self.rotate and self.d == 2:
            layout = self.rotate_layout(layout)
        # Scale the layout
        if self.rescale:
            layout = self.scale_and_center(layout)

        self.layout = layout
        return layout

    def integer_lattice_layout(self, lattice_points=3):
        """
        Map the vertices in a layout to their closest integer points in the scaled positive orthant; see
        scale_to_positive_orthant().

        Note: if the graph is Chimera or Pegasus, lattice points are inferred from the graph object and the layout is
        ignored. If the user desires to have lattice points computed from a layout (e.g. kamada_kawai), make sure that
        the graph object is created with the following flags: dnx.*_graph(coordinates=False, data=False).

        Parameters
        ----------
        lattice_points : int or tuple (default 3)
            The number of lattice points in each dimension. If it is an integer, there will be that many lattice points
            in each dimension of the layout. If it is a tuple, each entry specifies how many lattice points are in each
            dimension in the layout.

        Returns
        -------
        layout : dict
            A mapping from vertices of G (keys) to points in the scaled positive orthant (values).
        """
        # Look to see if you can get the lattice information from the graph object
        coordinates = dnx_utils.lookup_dnx_coordinates(self.G)
        if coordinates:
            return {v: coord + (self.d-2)*(0,) for v, coord in coordinates.items()}

        # Compute the lattice information by scaling and rounding
        lattice_vector = layout_utils.to_vector(lattice_points, self.d)
        scaled_layout = self.scale_to_positive_orthant(
            lattice_vector+1, border=1, invert=True)
        return {v: layout_utils.border_round(p, lattice_vector-1, self.d) for v, p in scaled_layout.items()}

    def integer_lattice_bins(self, lattice_points=3):
        """
        Map the bins of an integer lattice to lists of closest vertices in the scaled positive orthant; see
        scale_to_positive_orthant().

        Note: If the graph is Chimera or Pegasus, lattice points are inferred from the graph object using the first two
        coordinates of vertices of Chimera and Pegasus, i.e. the layout is ignored. If the user desires to have lattice
        points computed from a layout (e.g. kamada_kawai), make sure that the graph object is created with the following
        flags: dnx.*_graph(coordinates=False, data=False).

        Parameters
        ----------
        lattice_points : int or tuple (default 3)
            The number of lattice points in each dimension. If it is an integer, there will be that many lattice points
            in each dimension of the layout. If it is a tuple, each entry specifies how many lattice points are in each
            dimension in the layout.

        Returns
        -------
        layout : dict
            A mapping from points in the scaled positive orthant (keys) to lists of vertices of G (values).
        """
        integer_point_map = defaultdict(list)

        # Look to see if you can get the lattice information from the graph object.
        # If so, look them up and return.
        coordinates = dnx_utils.lookup_dnx_coordinates(self.G)
        if coordinates:
            for v, coord in coordinates.items():
                integer_point_map[coord + (self.d-2)*(0,)].append(v)
            return integer_point_map

        # Compute the lattice information by scaling and rounding
        lattice_vector = layout_utils.to_vector(lattice_points, self.d)
        scaled_layout = self.scale_to_positive_orthant(
            lattice_vector+1, border=1, invert=True)

        for v, p in scaled_layout.items():
            integer_point_map[
                layout_utils.border_round(p, lattice_vector-1, self.d)
            ].append(v)

        return integer_point_map

    def scale_to_positive_orthant(self, length=1, border=0., invert=False):
        """
        This helper function transforms the layout [self.center - self.scale, self.center + self.scale]^d to the
        semi-positive orthant:
        [0 - border, length[0] + border] x [0 - border, length[1] + border] x ... x [0 - border, length[d-1] + border].

        Default values of length=1 and border=0 give the unit positive orthant.

        Parameters
        ----------
        length : int or tuple (default 1)
            Specifies a vector called scale. If length is an integer, it is the max for all dimensions; if it is a
            tuple, each entry specifies a max for each dimension in the layout.
        border : float (default 0)
            Will shift the positive_orthant representation by the given amount in each dimension.
        invert : bool (default False)
            If true, will perform a reflection about the x-axis during the transformation. This is so that layouts match
            the dnx.*_layouts (whos scheme is matrixy notation, i.e., y increases as you go down). 

        Returns
        -------
        layout : dict
            A mapping from vertices of G (keys) to points in
            [0, length[0]] x [0, length[1]] x ... x [0, length[d-1]] (values).
        """
        # Temporary data structure to pull the information out of the matrix created below.
        V = [v for v in self.G]

        # Make a matrix of the positions
        L = np.array([self.layout[v] for v in V])
        # Shift it from [center - scale, center + scale]^d to [0, 2*scale]^d
        if invert:
            assert self.d == 2, "Inversion is only supported in 2-dimensions."
            # First move it to the origin
            L = L - self.center
            # Reflect about the x-axis
            L = L @ np.array([[1, 0], [0, -1]])
            # Move it to the positive orthant
            L = L + self.scale
        else:
            L = L + (self.scale - self.center)
        # Map it to [0, 1]^d
        L = L / (2*self.scale)
        # Scale it to the desired length [0, length]^d
        L = L * layout_utils.to_vector(length, self.d)
        # Extend it by the border amount
        L = L - border

        return {v: p for v, p in zip(V, L)}

    def center_to_top_left(self):
        """
        This function translates a center and a scale from the networkx convention, [center - scale, center + scale]^d,
        to the dwave_networkx convention, [center, center-scale] x [center, center+scale]^(d-1).

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

    def scale_and_center(self, layout=None, center=None, scale=None):
        """
        This helper function transforms a layout from [center - scale, center + scale]^d to the user desired
        [self.center - self.scale, self.center + self.scale]^d.

        Parameters
        ----------
        layout : dict or numpy array (default None)
            A mapping from vertices of G (keys) to points in R^d (values). If None, self.layout is used.
        center : tuple or numpy array (default None)
            A point in R^d representing the center of the layout. If None, the approximate center of layout is computed 
            by calculating the center of mass (or centroid).
        scale : float (default None)
            The scale of the parameter layout. If None, the approximate scale of layout is computed by taking the 
            maximum distance from the center.

        Returns
        -------
        layout : dict
            A mapping from vertices of G (keys) to points in [center - scale, center + scale]^d (values).
        """
        # If layout is empty, grab the object's
        if layout is None:
            layout = self.layout

        # Support layouts of different datatypes. Convert to a matrix.
        if isinstance(layout, (dict, defaultdict)):
            L = np.array([layout[v] for v in self.G])
        else:
            L = layout

        # If center and scale are empty, compute them
        if center is None:
            center = np.mean(L, axis=0)

        # Translate the layout so that it's center is the origin
        L = L - center

        # Scale so that it expands to fill [-self.scale, self.scale]^d
        if scale is None:
            min_value, max_value = np.min(L), np.max(L)
            scale = max(abs(min_value), abs(max_value))

        L = (self.scale/scale) * L

        # Translate the layout to [self.center - self.scale, self.center + self.scale]^d
        L = L + self.center

        return {v: p for v, p in zip(list(self.G), L)}

    def rotate_layout(self, layout=None, center=None):
        """
        Finds a minimum bounding box and rotates a (2-dimensional) layout so that it is aligned with the x and y axes.

        Parameters
        ----------
        layout : dict or numpy array (default None)
            A mapping from vertices of G (keys) to points in R^d (values). If None, self.layout is used.
        center : tuple or numpy array (default None)
            A point in R^d representing the center of the layout. If None, the approximate center of layout is computed 
            by calculating the center of mass (or centroid).

        Returns
        -------
        layout : dict
            A mapping from vertices of G (keys) to points in R^d (values).
        """
        # If layout is empty, grab the object's
        if layout is None:
            layout = self.layout

        # Support layouts of different datatypes. Convert to a matrix.
        if isinstance(layout, (dict, defaultdict)):
            L = np.array([layout[v] for v in self.G])
        else:
            L = layout

        assert len(
            L[0] == 2), "I only know how to rotate 2-dimensional layouts."

        # If center is empty, use the object's
        if center is None:
            center = np.mean(L, axis=0)

        # Translate the layout to the origin
        L = L - center

        # Compute the minimum area bounding box
        bounding_box = layout_utils.minimum_bounding_rectangle(L)
        bottom_right = bounding_box[0]
        bottom_left = bounding_box[1]

        # Find the angle to rotate and build a rotation matrix
        delta_x = abs(bottom_right[0] - bottom_left[0])
        delta_y = abs(bottom_right[1] - bottom_left[1])
        theta = np.arctan2(delta_y, delta_x)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])

        # Rotate the thing
        rotated_L = (R @ L.T).T

        # Translate back to the center you started with
        rotated_L = rotated_L + center

        return {v: p for v, p in zip(list(self.G), rotated_L)}


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
        raise ValueError(f"Parameter to_scale={to_scale} is not supported.")

    return {v: scale*p for v, p in layout.items()}

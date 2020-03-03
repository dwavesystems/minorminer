import statistics
from collections import defaultdict
from itertools import combinations

import dwave_networkx as dnx
import networkx as nx
import numpy as np
from scipy import optimize
from scipy.spatial.distance import euclidean

from minorminer.layout.utils import dnx_utils, graph_utils, layout_utils


def p_norm(
    G,
    p=2,
    starting_layout=None,
    G_distances=None,
    d=2,
    center=None,
    scale=None,
    recenter=True,
    rescale=False,
    rotate=False,
    **kwargs
):
    """
    Top level function for minorminer.layout.__init__() use as a parameter.
    # FIXME: There's surely a better way of doing this.
    """
    L = Layout(G, d=d, center=center, scale=scale,
               recenter=recenter, rescale=rescale, rotate=rotate)
    _ = L.p_norm(p, starting_layout, G_distances, **kwargs)
    return L


def dnx_layout(G, d=2, center=None, scale=None, rescale=True, rotate=True, **kwargs):
    """
    Top level function for minorminer.layout.__init__() use as a parameter.
    # FIXME: There's surely a better way of doing this.
    """
    L = Layout(G, d=d, center=center, scale=scale,
               rescale=rescale, rotate=rotate)
    _ = L.dnx_layout(**kwargs)
    return L


def pca(G, d=2, m=None, pca=True, center=None, scale=None, seed=None, rescale=False, rotate=False, **kwargs):
    """
    Top level function for minorminer.layout.__init__() use as a parameter.
    # FIXME: There's surely a better way of doing this.
    """
    L = Layout(G, d=d, center=center, scale=scale,
               seed=seed, rescale=rescale, rotate=rotate)
    _ = L.pca(m, pca, **kwargs)
    return L


def R2xT(
    G,
    starting_layout=None,
    G_distances=None,
    d=4,
    center=None,
    scale=1.,
    rescale=False,
    rotate=False,
    **kwargs
):
    """
    Top level function for minorminer.layout.__init__() use as a parameter.
    # FIXME: There's surely a better way of doing this.
    """
    L = Layout(G, d=d, center=center, scale=scale,
               rescale=rescale, rotate=rotate)
    _ = L.R2xT(
        starting_layout, G_distances, **kwargs)
    return L


class Layout():
    def __init__(
        self,
        G,
        layout=None,
        d=2,
        center=None,
        scale=None,
        recenter=True,
        rescale=False,
        rotate=False,
        seed=None,
        **kwargs
    ):
        """
        Compute a layout for G, i.e., a map from G to [center - scale, center + scale]^d.

        Parameters
        ----------
        G : NetworkX graph or NetworkX supported edges data structure (dict, list, ...)
            The graph you want to compute the layout for.
        layout : dict (default None)
            You can specify a pre-computed layout for G. If this is specified: d, center, scale are all calculated from
            the layout passed in.
        d : int (default 2)
            The desired dimension of the layout, i.e., R^d.
        center : tuple (default None)
            The desired center point of the layout; if None it is computed to be the origin in R^d.
        scale : float (default 1)
            The desired scale of the layout; i.e. the layout is in [center - scale, center + scale]^d space.
        recentere : bool (default True)
            If True, the layout is centerd to the user specified center. If False, the layout assumes the center of the 
            layout algorithm used.
        rescale : bool (default False)
            If True, the layout is scaled to the user specified [center - scale, center + scale]^d. If False, the layout
            assumes the dimensions of the layout algorithm used.
        rotate : bool (default False)
            If True, the minimum area bounding box for the layout is computed and rotated so that it aligned with the
            x and y axes. If False, the layout is not rotated.
        seed : int (default None)
            When d > 2, kamada_kawai uses networkx.random_layout(). The seed is passed to this function.
        kwargs : dict
            Keyword arguments are passed to one of the layout algorithms below.
        """
        # Ensure G is a graph object
        self.G = graph_utils.parse_graph(G)

        # If passed in, save the layout and the layout array data types
        if layout:
            if isinstance(layout, (dict, defaultdict)):
                self.layout = layout
                self.layout_array = np.array([layout[v] for v in self.G])
            elif isinstance(layout, (np.array, list)):
                self.layout = {v: p for v, p in zip(G, layout)}
                self.layout_array = layout

            # Set the layout's center, scale, and dim from calculating them based on the layout passed in by the user
            self.d = self.layout_array.shape[1]
            self.center = np.mean(self.layout_array, axis=0)
            self.scale = np.max(
                np.linalg.norm(
                    self.layout_array - self.center, float("inf"), axis=0
                )
            )
        else:
            self.d = d
            self.layout = layout
            self.layout_array = layout
            self.center = center or self.d*(0,)
            self.scale = scale

        # Set remaining parameters
        self.seed = seed
        self.recenter = recenter
        self.rescale = True if scale else rescale
        self.rotate = rotate

    def p_norm(self, p=2, starting_layout=None, G_distances=None, **kwargs):
        """
        Embeds a graph in R^d with the p-norm and minimizes a Kamada-Kawai-esque objective function to achieve
        an embedding with low distortion. This computes a layout where the graph distance and the p-distance are 
        very close to each other.

        Parameters
        ----------
        starting_layout : dict or Numpy Array
            A mapping from the vertices of G to points in the metric space.
        G_distances : dict or Numpy 2d array (default None)
            A dictionary of dictionaries representing distances from every vertex in G to every other vertex in G, or
            a matrix representing the same data. If None, it is computed.
        p : int (default 2)
            The order of the p-norm to use as a metric.

        Returns
        -------
        layout : dict
            A mapping from vertices of G (keys) to points in R^d (values).
        """
        # Pick a random layout in R^2
        if starting_layout is None:
            if self.d >= len(self.G):
                starting_layout = nx.random_layout(self.G, dim=self.d)
            else:
                starting_layout = nx.spectral_layout(self.G, dim=self.d)

        # Make sure the layout is a vector
        if isinstance(starting_layout, dict):
            starting_layout = np.array(
                [starting_layout[v] for v in self.G]
            )

        # Check the dimension of the layout
        k = starting_layout.shape[1]
        assert self.d == k, f"The starting layout has dimension {k}, but the object wants dimension {self.d}"

        # Save on distance calculations by passing them in
        if G_distances is None:
            G_distances = layout_utils.graph_distances(self.G)

        # Make sure the distances are a matrix
        if isinstance(G_distances, dict):
            G_distances = np.array(
                [[V[v] for v in self.G] for u, V in G_distances.items()]
            )

        # Solve the Kamada-Kawai-esque minimization function
        X = optimize.minimize(
            layout_utils.p_norm,
            starting_layout.ravel(),
            method='L-BFGS-B',
            args=(G_distances, k, p),
            jac=True,
            **kwargs
        )

        # Read out the solution to the minimization problem and save layouts
        self.layout_array = X.x.reshape(len(self.G), k)
        self.layout = {v: pos for v, pos in zip(self.G, self.layout_array)}

        # Save copies of the desired center and scale.
        desired_center = self.center
        desired_scale = self.scale

        # Calculate the scale and center based on the layout
        self.center = np.mean(self.layout_array, axis=0)
        self.scale = np.max(
            np.linalg.norm(
                self.layout_array - self.center, float("inf"), axis=0
            )
        )

        # Transform the layout
        if self.recenter:
            self.center_layout(desired_center)
        if self.rescale:
            self.scale_layout(desired_scale)
        if self.rotate:
            self.rotate_layout()

        return self.layout

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
            # Default scale is dependent on the longest dimension of Chimera or Pegasus.
            if self.scale is None:
                n, m, _ = dnx_utils.lookup_dnx_dims(self.G)
                self.scale = max(n, m)/2

            top_left, new_scale = dnx_utils.nx_to_dnx_layout(
                self.center, self.scale)

            kwargs["center"] = top_left
            kwargs["scale"] = new_scale

        if family == "chimera":
            layout = dnx.chimera_layout(self.G, dim=self.d, **kwargs)
        elif family == "pegasus":
            layout = dnx.pegasus_layout(self.G, dim=self.d, **kwargs)

        self.layout = layout
        self.layout_array = np.array([layout[v] for v in self.G])

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
        m = m or n if n < 50 else 50
        assert m <= n, f"The number of vertices {n} bounds the dimension."

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
            self.layout_array = np.column_stack(
                [X_T @ u for u in list(reversed(eigenvectors))[:self.d]])
            self.layout = {
                v: row for v, row in zip(list(self.G), self.layout_array)
            }

        else:
            self.layout = starting_layout
            self.layout_array = np.array([self.layout[v] for v in self.G])

        # Save copies of the desired center and scale.
        desired_center = self.center
        desired_scale = self.scale

        # Calculate the scale and center based on the layout
        self.scale = np.max(
            np.linalg.norm(
                self.layout_array - self.center, float("inf"), axis=0
            )
        )
        self.center = np.mean(self.layout_array, axis=0)

        # Transform the layout
        if self.recenter:
            self.center_layout(desired_center)
        if self.rescale:
            self.scale_layout(desired_scale)
        if self.rotate:
            self.rotate_layout()

        return self.layout

    def R2xT(self, starting_layout=None, G_distances=None, **kwargs):
        """
        Embeds a graph in a custom metric space and minimizes a Kamada-Kawai-esque objective function to achieve
        an embedding with low distortion. This computes a layout where the graph distance and the distance in the plane
        cross the torus are very close to each other.

        Parameters
        ----------
        starting_layout : dict or Numpy Array
            A mapping from the vertices of G to points in the metric space.
        G_distances : dict or Numpy 2d array (default None)
            A dictionary of dictionaries representing distances from every vertex in G to every other vertex in G, or
            a matrix representing the same data. If None, it is computed.

        Returns
        -------
        layout : dict
            A mapping from vertices of G (keys) to points in R^2 x T (values).
        """
        # Pick a random R^2 x T layout
        if starting_layout is None:
            starting_layout = layout_utils.random_layout(self.G)

        # Make sure the layout is a vector
        if isinstance(starting_layout, dict):
            starting_layout = np.array(
                [pos for pos in starting_layout.values()])

        assert starting_layout.shape[1] == 4, "This is a 4-dimensional layout."

        # Save on distance calculations by passing them in
        if G_distances is None:
            G_distances = layout_utils.graph_distances(self.G)

        # Make sure the distances are a matrix
        if isinstance(G_distances, dict):
            G_distances = np.array(
                [[V[v] for v in self.G] for u, V in G_distances.items()]
            )

        # Solve the Kamada-Kawai-esque minimization function
        X = optimize.minimize(
            layout_utils.R2xT,
            starting_layout.ravel(),
            # method='L-BFGS-B',
            args=(G_distances, ),
            jac=True,
            **kwargs
        )

        # Read out the solution to the minimization problem and save layouts
        self.layout_array = X.x.reshape(len(self.G), 4)
        # Angles in the layout are in [0, 2*pi]
        self.layout_array[:, 2:] = np.mod(self.layout_array[:, 2:], 2*np.pi)
        self.layout = {v: pos for v, pos in zip(self.G, self.layout_array)}

        # FIXME: This needs to only recenter, rescale, and rotate the R^2 projection of the layout.
        # # Save copies of the desired center and scale.
        # desired_center = self.center
        # desired_scale = self.scale

        # # Calculate the scale and center based on the layout
        # self.scale = np.max(
        #     np.linalg.norm(
        #         self.layout_array - self.center, float("inf"), axis=0
        #     )
        # )
        # self.center = np.mean(self.layout_array, axis=0)

        # # Transform the layout
        # if self.recenter:
        #     self.center_layout(desired_center)
        # if self.rescale:
        #     self.scale_layout(desired_scale)
        # if self.rotate:
        #     self.rotate_layout()

        return self.layout

    def center_layout(self, new_center):
        """
        This helper function transforms a layout from [self.center - scale, self.center + scale]^d to
        [new_center - scale, new_center + scale]^d.

        Parameters
        ----------
        new_center : tuple or numpy array (default None)
            A point in R^d to make the new center of the layout.

        Returns
        -------
        layout : dict
            A mapping from vertices of G (keys) to points in [new_center - scale, new_center + scale]^d (values).
        """
        centered_layout = center_layout(
            self.layout_array, new_center, self.center)

        # Update the object
        self.center = new_center
        self.layout_array = centered_layout
        self.layout = {v: p for v, p in zip(self.G, self.layout_array)}

        return self.layout

    def invert_layout(self):
        """
        This helper function transforms a (2-dimensional) layout by reflecting the layout across the x-axis.

        Returns
        -------
        layout : dict
            A mapping from vertices of G (keys) to points in R^d (values).
        """
        inverted_layout = invert_layout(self.layout_array, self.center)

        # Update the object
        self.layout_array = inverted_layout
        self.layout = {v: p for v, p in zip(self.G, self.layout_array)}

        return self.layout

    def scale_layout(self, new_scale):
        """
        This helper function transforms a layout from [center - self.scale, center + self.scale]^d to 
        [center - new_scale, center + new_scale]^d.

        Parameters
        ----------
        new_scale : float
            The desired scale to transform the layout to.

        Returns
        -------
        layout : dict
            A mapping from vertices of G (keys) to points in [center - new_scale, center + new_scale]^d (values).
        """
        scaled_layout = scale_layout(
            self.layout_array, new_scale, self.scale, self.center)

        # Update the object
        self.scale = new_scale
        self.layout_array = scaled_layout
        self.layout = {v: p for v, p in zip(self.G, self.layout_array)}

        return self.layout

    def rotate_layout(self):
        """
        Finds a minimum bounding box and rotates a (2-dimensional) layout so that it is aligned with the x and y axes.

        Returns
        -------
        layout : dict
            A mapping from vertices of G (keys) to points in R^d (values).
        """
        rotated_layout = rotate_layout(self.layout_array, self.center)

        # Update the object
        self.layout_array = rotated_layout
        self.layout = {v: p for v, p in zip(self.G, self.layout_array)}

        return self.layout

    def integer_lattice_bins(self, lattice_points=3):
        """
        Map the points of an integer lattice to lists of closest vertices in the scaled positive orthant and vice-versa;
        see scale_to_positive_orthant().

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
        integer_point_map : dict
            A mapping from points in the scaled positive orthant (keys) to lists of vertices of G (values).
        layout : dict
            A mapping from vertices of G (keys) to points in the scaled positive orthant (values).
        """
        integer_point_map = defaultdict(list)
        layout = {}

        # Look to see if you can get the lattice information from the graph object.
        # If so, look them up and return.
        coordinates = dnx_utils.lookup_dnx_coordinates(self.G)
        if coordinates:
            for v, coord in coordinates.items():
                # Extend to d-dimensions
                ext_cord = coord + (self.d-2)*(0,)

                integer_point_map[ext_cord].append(v)
                layout[v] = ext_cord
            return integer_point_map, layout

        # Compute the lattice information by scaling and rounding
        lattice_vector = layout_utils.to_vector(lattice_points, self.d)
        scaled_layout = self.scale_to_positive_orthant(
            lattice_vector+1, border=1, invert=True)

        for v, p in scaled_layout.items():
            # Find the bin
            rounded_p = layout_utils.border_round(p, lattice_vector-1, self.d)

            integer_point_map[rounded_p].append(v)
            layout[v] = rounded_p

        return integer_point_map, layout

    def scale_to_positive_orthant(self, layout=None, length=1, border=0., invert=False):
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

        if layout is None:
            layout = self.layout

        # Make a matrix of the positions
        L = np.array([layout[v] for v in V])

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


def rotate_layout(layout, center=None):
    """
    Finds a minimum bounding box and rotates a (2-dimensional) layout so that it is aligned with the x and y axes.

    Parameters
    ----------
    layout : numpy array
        An array whose rows are points in R^2.
    center : tuple or numpy array (default None)
        A point in R^2 representing the center of the layout. If None, the approximate center of layout is computed 
        by calculating the center of mass (or centroid).

    Returns
    -------
    layout : numpy array
        An axis aligned layout.
    """
    assert layout.shape[1] == 2, "I only know how to rotate 2-dimensional layouts."

    # If center is empty, compute it
    if center is None:
        center = np.mean(layout, axis=0)

    # Translate the layout to the origin
    L = layout - center

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
    return rotated_L + center


def scale_layout(layout, new_scale, old_scale=None, center=None):
    """
    This helper function transforms a layout from [center - old_scale, center + old_scale]^d to 
    [center - new_scale, center + new_scale]^d.

    Parameters
    ----------
    layout : numpy array (default None)
        An array whose rows are points in R^d.
    new_scale : float
        The desired scale to transform the layout to.
    old_scale : float (default None)
        The scale of the layout. If None, the approximate scale of layout is computed by taking the maximum distance 
        from the center.
    center : tuple or numpy array (default None)
        A point in R^d representing the center of the layout. If None, the approximate center of layout is computed by 
        calculating the center of mass (or centroid).

    Returns
    -------
    layout : numpy array
        A layout that has been scaled to [center - new_scale, center + new_scale]^d.
    """
    # If center is empty, compute it
    if center is None:
        center = np.mean(layout, axis=0)

    # Translate the layout so that its center is the origin
    L = layout - center

    # Compute the scale of the passed-in layout
    if old_scale is None:
        old_scale = np.max(
            np.linalg.norm(L, float("inf"), axis=0)
        )

    # Scale the thing
    scaled_L = (new_scale/old_scale) * L

    # Translate the layout back to where it was
    return scaled_L + center


def invert_layout(layout, center=None):
    """
    This helper function transforms a (2-dimensional) layout by reflecting the layout across the x-axis.

    Parameters
    ----------
    layout : numpy array
        An array whose rows are points in R^2.
    center : tuple or numpy array (default None)
        A point in R^2 representing the center of the layout. If None, the approximate center of layout is computed by 
        calculating the center of mass (or centroid).

    Returns
    -------
    layout : numpy array
        A reflected layout.
    """
    assert layout.shape[1] == 2, "Inversion is only supported in 2-dimensions."

    # If center and scale are empty, compute them
    if center is None:
        center = np.mean(layout, axis=0)

    # Translate the layout so that its center is the origin
    L = layout - center

    # Reflect about the x-axis
    inverted_layout = L @ np.array([[1, 0], [0, -1]])

    # Move it back to its center
    return inverted_layout + center


def center_layout(layout, new_center, old_center=None):
    """
    This helper function transforms a layout from [old_center - scale, old_center + scale]^d to
    [new_center - scale, new_center + scale]^d.

    Parameters
    ----------
    layout : dict or numpy array (default None)
        A mapping from vertices of G (keys) to points in R^d (values). If None, self.layout is used.
    new_center : tuple or numpy array
        A point in R^d that is the desired center to move the layout to. 
    old_center : tuple or numpy array (default None)
        A point in R^d representing the center of the layout. If None, the approximate center of layout is computed 
        by calculating the center of mass (or centroid).

    Returns
    -------
    layout : dict
        A mapping from vertices of G (keys) to points in [self.center - scale, self.center + scale]^d (values).
    """
    # If old_center is empty, compute it
    if old_center is None:
        old_center = np.mean(layout, axis=0)

    # Translate the layout so that we have the desired new_center
    return layout - old_center + new_center

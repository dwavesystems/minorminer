# Copyright 2020 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from collections import abc

import dwave_networkx as dnx
import networkx as nx
import numpy as np
from scipy import optimize, spatial
from math import ceil
from minorminer._extern import rpack

def p_norm(G, p=2, starting_layout=None, G_distances=None, dim=None, center=None, scale=None, **kwargs):
    """Embeds graph ``G`` in :math:`R^d` with the p-norm and minimizes a 
    Kamada-Kawai-esque objective function to achieve an embedding with low 
    distortion.
    
    This computes a :class:`.Layout` for ``G`` where the graph distance and the 
    p-distance are very close to each other.

    By default, :func:`p_norm` is used to compute the layout for source graphs 
    when :func:`minorminer.layout.find_embedding` is called.

    Args:
        G (NetworkX Graph):
            The graph to compute the layout for.

        p (int, optional, default=2):
            The order of the p-norm to use as a metric.

        starting_layout (dict, optional, default=None)
            A mapping from the vertices of ``G`` to points in :math:`R^d`. If 
            None, :func:`nx.spectral_layout` is used if possible. Otherwise, 
            :func:`nx.random_layout` is used.

        G_distances (dict, optional, default=None):
            A dictionary of dictionaries representing distances from every vertex 
            in ``G`` to every other vertex in ``G``. If None, it is computed.

        dim (int, optional, default=None):
            The desired dimension of the returned layout, :math:`R^{\dim}`. 
            If None, the dimension of ``center`` is used. If ``center`` is None, 
            ``dim`` is set to 2.

        center (tuple, optional, default=None):
            The desired center point of the returned layout. If None, ``center`` 
            is set as the origin in :math:`R^{\dim}` space.
        
        scale (float, optional, default=None):
            The desired scale of the returned layout; i.e. the layout 
            is in :math:`[center - scale, center + scale]^d` space. If None, no 
            scale is set.

    Returns:
        dict: :attr:`.Layout.layout`, a mapping from vertices of ``G`` (keys) to 
        points in :math:`R^d` (values).
    
    Examples:
        This example creates a :class:`.Layout` object for a hexagonal lattice 
        graph, with coordinates computed using :func:`p_norm`.

        >>> import networkx as nx
        >>> import minorminer.layout as mml
        ...
        >>> G = nx.hexagonal_lattice_graph(2,2)
        >>> layout = mml.Layout(G, mml.p_norm, center=(1,1))

        ``layout`` may be passed in directly to :func:`minorminer.layout.find_embedding`. 
        
        Alternatively, :func:`p_norm` may be passed in instead.

        This next example finds an embedding of a hexagonal lattice graph on a 
        Chimera graph, in which the layouts of both the source and target graphs 
        are computed using p_norm.

        >>> import networkx as nx
        >>> import dwave_networkx as dnx
        >>> import minorminer.layout as mml
        ...
        >>> G = nx.hexagonal_lattice_graph(2,2)
        >>> C = dnx.chimera_graph(2,2)
        >>> embedding = mml.find_embedding(G, 
        ...                                C, 
        ...                                layout=(mml.p_norm, mml.p_norm),
        ...                                center=(1,1))

    """
    dim, center = _set_dim_and_center(dim, center)

    # Use the user provided starting_layout, or a spectral_layout if the dimension 
    # is low enough. If neither, use a random_layout.
    if starting_layout:
        pass
    elif dim >= len(G):
        starting_layout = nx.random_layout(G, dim=dim)
    else:
        starting_layout = nx.spectral_layout(G, dim=dim)

    # Make a layout object
    layout = Layout(G, starting_layout, dim=dim)

    # Save on distance calculations by passing them in
    G_distances = _graph_distance_matrix(G, G_distances)

    # Solve the Kamada-Kawai-esque minimization function
    X = optimize.minimize(
        _p_norm_objective,
        layout.layout_array.ravel(),
        method='L-BFGS-B',
        args=(G_distances, dim, p),
        jac=True,
    )

    # Read out the solution to the minimization problem and save layouts
    layout.layout_array = X.x.reshape(len(G), dim)

    # Center and scale the layout
    layout.center = center

    if scale:
        layout.scale = scale

    return layout.layout


def _graph_distance_matrix(G,
                           all_pairs_shortest_path_length=None, 
                           disconnected_distance=None):
    """Compute the distance matrix of G.

    Args:
        G (NetworkX Graph):
            The graph to find the distance matrix of.
        
        all_pairs_shortest_path_length (dict, optional, default=None):
            If None, it is computed by calling :func:`nx.all_pairs_shortest_path_length`.
    
        disconnected_distance (float, optional, default=None):
            A default distance to use when nodes belong to different connected
            components. If None, `len(G)` is used.

    Returns:
        G_distances (Numpy 2d array):
            An array indexed by sorted vertices of G whose i,j value is d_G(i,j).
    
    """
    if all_pairs_shortest_path_length is None:
        all_pairs_shortest_path_length = nx.all_pairs_shortest_path_length(G)

    if disconnected_distance is None:
        disconnected_distance = len(G)

    return np.array(
        [
            [V.get(v, disconnected_distance) for v in sorted(G)]
                 for u, V in all_pairs_shortest_path_length
        ]
    )

def _p_norm_objective(layout_vector, G_distances, dim, p):
    """Compute the sum of differences squared between the p-norm and the graph 
    distance as well as the gradient.

    Args:
        layout (Numpy Array):
            A vector indexed by sorted vertices of G whose values are points in 
            some metric space.

        G_distances (Numpy 2d array):
            An array indexed by sorted vertices of G whose i,j value is d_G(i,j).
        
        dim (int):
            The dimension of the metric space. This will reshape the flattened 
            array passed in to the cost function.
        
        p (int):
            The order of the p-norm to use.

    Returns:
        float: The sum of differences squared between the metric distance and 
        the graph distance.
    
    """
    # Reconstitute the flattened array that scipy.optimize.minimize passed in
    n = len(G_distances)
    layout = layout_vector.reshape(n, dim)

    # Difference between pairs of points in a 3d matrix
    diff = layout[:, np.newaxis, :] - layout[np.newaxis, :, :]

    # A 2d matrix of the distances between points
    dist = np.linalg.norm(diff, ord=p, axis=-1)

    # TODO: Compare this division-by-zero strategy to adding epsilon.
    # A vectorized version of the gradient function
    with np.errstate(divide='ignore', invalid='ignore'):  # handle division by 0
        if p == 1:
            grad = np.einsum(
                'ijk,ij,ijk->ik',
                2*diff,
                dist - G_distances,
                np.nan_to_num(1/np.abs(diff))
            )
        elif p == float("inf"):
            # Note: It may not be faster to do this outside of einsum
            abs_diff = np.abs(diff)
            x_bigger = abs_diff[:, :, 0] > abs_diff[:, :, 1]

            grad = np.einsum(
                'ijk,ij,ijk,ijk->ik',
                2*diff,
                dist - G_distances,
                np.nan_to_num(1/abs_diff),
                np.dstack((x_bigger, np.logical_not(x_bigger)))
            )
        else:
            grad = np.einsum(
                'ijk,ijk,ij,ij->ik',
                2*diff,
                np.abs(diff)**(p-2),
                dist - G_distances,
                np.nan_to_num((1/dist)**(p-1))
            )

    # Return the cost and the gradient
    return np.sum((G_distances - dist)**2), grad.ravel()


def dnx_layout(G, dim=None, center=None, scale=None, **kwargs):
    """The Chimera or Pegasus layout from `dwave_networkx` centered at the origin
    with ``scale`` as a function of the number of rows or columns. Note: As per 
    the implementation of `dnx.*_layout`, if :math:`dim>2`, coordinates beyond 
    the second are 0.

    By default, :func:`dnx_layout` is used to compute the layout for target 
    Chimera or Pegasus graphs when :func:`minorminer.layout.find_embedding` is 
    called.

    Args:
        G (NetworkX Graph):
            The graph to compute the layout for.

        dim (int, optional, default=None):
            The desired dimension of the returned layout, :math:`R^{\dim}`. If 
            None, the dimension of ``center`` is used. If ``center`` is None, 
            ``dim`` is set to 2.

        center (tuple, optional, default=None):
            The desired center point of the returned layout. If None, it is set 
            as the origin in :math:`R^{\dim}` space.

        scale (float, optional, default=None):
            The desired scale of the returned layout; i.e. the layout is in 
            :math:`[center - scale, center + scale]^d` space. If None, it is set 
            as :math:`max(n, m)/2`, where n, m are the number of columns, rows 
            respectively in ``G``.

    Returns:
        dict: :attr:`.Layout.layout`, a mapping from vertices of ``G`` (keys) to 
        points in :math:`R^d` (values).
    
    Examples:
        This example creates a :class:`.Layout` object for a Pegasus graph, with 
        coordinates computed using :func:`dnx_layout`.

        >>> import networkx as nx
        >>> import minorminer.layout as mml
        ...
        >>> P = dnx.pegasus_graph(4)
        >>> layout = mml.Layout(P, mml.dnx_layout, center=(1,1), scale=2)

    """
    graph_data = G.graph

    family = graph_data.get("family")
    if G.graph.get("family") not in ("chimera", "pegasus", "zephyr"):
        raise ValueError(
            "This strategy is only implemented for Chimera, Pegasus"
            " and Zephyr graphs constructed by dwave_networkx`.")

    dim, center = _set_dim_and_center(dim, center)

    if scale is None:
        m, n = graph_data["rows"], graph_data["columns"]
        scale = max(n, m)/2

    dnx_center, dnx_scale = _nx_to_dnx_layout(center, scale)

    if family == "chimera":
        dnx_layout = dnx.chimera_layout(
            G, dim=dim, center=dnx_center, scale=dnx_scale)
    elif family == "pegasus":
        dnx_layout = dnx.pegasus_layout(
            G, dim=dim, center=dnx_center, scale=dnx_scale)
    elif family == "zephyr":
        dnx_layout = dnx.zephyr_layout(
            G, dim=dim, center=dnx_center, scale=dnx_scale)

    layout = Layout(G, dnx_layout)
    return layout.layout


def _nx_to_dnx_layout(center, scale):
    """This function translates a center and a scale from the networkx convention, 
    :math:`[center - scale, center + scale]^{\dim}`, to the `dwave_networkx` 
    convention, :math:`[center, center-scale] x [center, center+scale]^{\(dim-1)}`.

    Returns:
        tuple: A 2-tuple:
            float: The top left corner of a layout.

            float: A scale value that is twice the original scale.
    
    """
    dnx_center = (center[0] - scale, ) + tuple(x + scale for x in center[1:])
    dnx_scale = 2*scale

    return dnx_center, dnx_scale


class Layout(abc.MutableMapping):
    """Class that stores (or computes) coordinates in dimension ``dim`` for each 
    node in graph ``G``.

    Args:
        G (NetworkX Graph/edges data structure (dict, list, ...)):
            The graph to compute the layout for or a NetworkX supported data 
            structure for edges (see :func:`nx.convert.to_networkx_graph` for 
            details).

        layout (dict/function, optional, default=None):
            If a dict, this specifies a pre-computed layout for ``G``.
            
            If a function, this should be in the form of ``layout(G, **kwargs)``,
            in which ``dim``, ``center``, and ``scale`` are passed in as kwargs 
            and the return value is a dict representing a layout of ``G``.
            
            If None, :func:`nx.random_layout(G, **kwargs)` is called.

        dim (int, optional, default=None):
            The desired dimension of the computed layout, :math:`R^{\dim}`. If 
            None, ``dim`` is set as the dimension of ``layout``.

        center (tuple, optional, default=None):
            The desired center point of the computed layout. If None, ``center`` 
            is set as the center of ``layout``.

        scale (float, optional, default=None):
            The desired scale of the computed layout; i.e. the layout is in 
            :math:`[center - scale, center + scale]^d` space. If None, ``scale`` 
            is set to be the scale of ``layout``.

        pack_components (bool, optional, default=True):
            If True, and if the graph contains multiple components and ``dim`` 
            is None or 2, the components will be laid out individually and packed 
            into a rectangle.

        **kwargs (dict):
            Keyword arguments are passed to ``layout`` if it is a function.
    
    """

    def __init__(
        self,
        G,
        layout=None,
        dim=None,
        center=None,
        scale=None,
        pack_components = True,
        **kwargs
    ):
        # Ensure G is a graph object
        self.G = _parse_graph(G)

        # Add dim, center, and scale to kwargs if not None
        if dim is not None:
            kwargs["dim"] = dim
        if center is not None:
            kwargs["center"] = center
        if scale is not None:
            kwargs["scale"] = scale

        _call_layout = False
        # If passed in, save or compute the layout
        if layout is None:
            if self.G.graph.get('family') in ('pegasus', 'chimera', 'zephyr'):
                self.layout = dnx_layout(self.G, **kwargs)
            else:
                _call_layout = True
                layout = p_norm
        elif callable(layout):
            _call_layout = True
        else:
            # Assumes layout implements a mapping interface
            self.layout = layout

        if _call_layout:
            if pack_components:
                self.layout = _pack_components(self.G, layout, **kwargs)
            else:
                self.layout = layout(self.G, **kwargs)

        # Set specs in the order of (user input, precomputed layout)
        self.dim = dim or self._dim
        self.scale = scale or self._scale
        if center is not None:
            self.center = center
        else:
            self.center = self._center

    # Keep layout and layout_array in lockstep with each other.
    @property
    def layout(self):
        return self._layout

    @layout.setter
    def layout(self, value):
        """If layout is set, also set layout_array and the layout specs."""
        # Set the layout
        self._layout = value

        # Iterating through G determines the order of layout_array
        self._layout_array = np.array([value[v] for v in sorted(self.G)])

        self._set_layout_specs()

    @property
    def layout_array(self):
        return self._layout_array

    @layout_array.setter
    def layout_array(self, value):
        """If layout_array is set, also set layout and the layout specs."""
        # Set the layout_array
        self._layout_array = value

        # Update the layout
        self.layout = {v: p for v, p in zip(sorted(self.G), value)}

    @property
    def dim(self):
        return self._dim

    @dim.setter
    def dim(self, value):
        """Set the dimension of the layout, if possible."""
        if value:
            self.layout_array = _dimension_layout(
                self.layout_array, value, self._dim)

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, value):
        """Recenter the layout."""
        # Cast it as a numpy array incase it's not.
        value = np.array(value)

        if value.size != 0:
            self.layout_array = _center_layout(
                self.layout_array, value, self._center)

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        """Rescale the layout."""
        if value:
            self.layout_array = _scale_layout(
                self.layout_array, value, self._scale, self._center)

    # The layout class should behave like a dictionary
    def __iter__(self):
        """Iterate through the keys of the dictionary layout."""
        yield from self.layout

    def __getitem__(self, key):
        """Get the layout value at the key vertex."""
        return self.layout[key]

    def __setitem__(self, key, value):
        """Set the layout value at the key vertex."""
        self.layout[key] = value

    def __delitem__(self, key):
        """Delete the layout value at the key vertex."""
        del self.layout[key]

    def __repr__(self):
        """Use the layout's dictionary representation."""
        return repr(self.layout)

    def __len__(self):
        """The length of a layout is the length of the layout dictionary."""
        return len(self.layout)

    def _set_layout_specs(self, empty=False):
        """Set the dimension, center, and scale of the layout_array currently 
        in the Layout object.
        """
        if self.layout_array.size == 0:
            self._dim = 0
            self._center = np.array([])
            self._scale = 0
        else:
            self._dim = self.layout_array.shape[1]
            self._center = _get_center(self.layout_array)
            self._scale = np.max(
                np.linalg.norm(
                    self.layout_array - self._center, float("inf"), axis=0
                )
            )


def _dimension_layout(layout_array, new_d, old_d=None):
    """This helper function transforms a layout from R^old_d to R^new_d by 
    padding extra dimensions with 0's.

    Args:
        layout_array (Numpy array):
            An array whose rows are points in R^old_dim.
        
        new_d (int):
            The new dimension to convert the layout to, must be larger than old_dim.
        
        old_d (int, optional, default=None):
            The current dimension of the layout. If None, the dimension is looked 
            up via layout_array.

    Returns:
        Numpy array: A layout that has been projected into R^new_dim.
    
    """
    # If old_center is empty, compute it
    if old_d is None:
        old_d = layout_array.shape[1]

    if new_d < old_d:
        raise ValueError(
            "The new dimension {} is larger than the old dimension {}. This is not currently supported.".format(
                new_d, old_d)
        )

    n = layout_array.shape[0]
    # Make a block of zeros
    new_layout = np.zeros((n, new_d))
    # Overwrite the entries using layout_array
    new_layout[:, :old_d] = layout_array
    return new_layout


def _center_layout(layout_array, new_center, old_center=None):
    """This helper function transforms a layout from 
    :math:`[old_center - scale, old_center + scale]^d` to
    :math:`[new_center - scale, new_center + scale]^d`.

    Args:
        layout_array (Numpy array):
            An array whose rows are points in R^d.

        new_center (tuple/Numpy array):
            A point in R^d that is the desired center to move the layout to. 
        
        old_center (tuple/Numpy array, optional, default=None):
            A point in R^d representing the center of the layout. If None, the
            approximate center of layout is computed by calculating the center of 
            mass (or centroid).

    Returns:
        Numpy array: A layout that has been centered at new_center.
    
    """
    # If old_center is empty, compute it
    if old_center is None:
        old_center = _get_center(layout_array)

    # Translate the layout so that we have the desired new_center
    return layout_array - old_center + new_center


def _scale_layout(layout_array, new_scale, old_scale=None, center=None):
    """This helper function transforms a layout from 
    :math:`[center - old_scale, center + old_scale]^d` to 
    :math:`[center - new_scale, center + new_scale]^d`.

    Args:
        layout_array (Numpy array):
            An array whose rows are points in R^d.
        
        new_scale (float):
            The desired scale to transform the layout to.
        
        old_scale (float, optional, default=None):
            The scale of the layout. If None, the approximate scale of layout is 
            computed by taking the maximum distance from the center.
        
        center (tuple/Numpy array, optional, default=None):
            A point in R^d representing the center of the layout. If None, the 
            approximate center of layout is computed by calculating the center of 
            mass (centroid).

    Returns:
        Numpy array: A layout that has been scaled to [center - new_scale, center + new_scale]^d.
    
    """
    # If center is empty, compute it
    if center is None:
        center = _get_center(layout_array)

    # Translate the layout so that its center is the origin
    L = layout_array - center

    # Compute the scale of the passed-in layout
    if old_scale is None:
        old_scale = np.max(np.abs(L))

    # Scale the thing
    scaled_L = (new_scale/old_scale) * L

    # Translate the layout back to where it was
    return scaled_L + center


def _set_dim_and_center(dim, center, default_dim=2):
    """A helper function to check that a user provided dim and center match, or 
    if no user provided dim and center exist sets default values: dim=2 and 
    center=the origin in :math:`R^{\dim}`.
    """
    # Set the dimension
    if dim is None:
        if center is None:
            dim = default_dim
        else:
            dim = len(center)

    # Set the center
    if center is None:
        center = dim*(0, )

    if len(center) != dim:
        raise ValueError(
            "Your dimension is {} but your center is {}.".format(dim, center))

    return dim, center


def _rotate_to_minimize_area(points):
    """Uses the rotating calipers algorithm to rotate points (a numpy array)
    into a minimum-area bounding box
    """
    # Compute the convex hull
    hull = spatial.ConvexHull(points, qhull_options = 'QJ')

    # Look up the points that constitute the convex hull
    hull_points = points[hull.vertices]

    pi2 = np.pi/2

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    rotations = np.vstack([
        np.cos(angles),
        -np.sin(angles),
        np.sin(angles),
        np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    widths = (max_x - min_x)
    heights = (max_y - min_y)
    areas =  widths * heights
    best_idx = np.argmin(areas)

    # return the best rotation and its dimensions
    points = np.dot(points, rotations[best_idx])
    min_y = np.nanmin(points[:, 0])
    max_y = np.nanmax(points[:, 0])
    min_x = np.nanmin(points[:, 1])
    max_x = np.nanmax(points[:, 1])
    center = np.array(((max_y + min_y), (max_x + min_x)))/2
    dims = np.array(((max_y - min_y), (max_x - min_x)))
    return points - center, dims

def _pack_components(G, layout, **kwargs):
    """Attempts to pack components using `layout` to compute component layouts,
    rotates those component layouts to minimum-area bounding rectangles,
    and then uses a rectangle packing algorithm to minimize the area of the
    overall layout.

    If `scale` is an argument, we rescale the layout after the above.

    If `dim` is an argument and not equal to 2, we don't have an
    implementation, so we pass the graph directly into `layout` and bypass
    the packing procedure
    """
    if kwargs.get('dim', 2) != 2:
        return layout(G, **kwargs)

    layouts = []
    rectangles = []
    if 'scale' in kwargs:
        subkwargs = dict(kwargs)
        scale = subkwargs['scale']
        del subkwargs['scale']
    else:
        subkwargs = kwargs
        scale = None

    for i, c in enumerate(nx.connected_components(G)):
        if len(c) > 2:
            cpos = layout(G.subgraph(c),
                          scale = len(c)**.5,
                          **subkwargs)
            apos = np.array(list(cpos.values()))
            rpos, dims = _rotate_to_minimize_area(apos)
            cpos = dict(zip(cpos, rpos))
            dims = [int(ceil(z)+.001) for z in dims]
        elif len(c) == 2:
            cpos = dict(zip(c, [(-.5, 0), (.5, 0)]))
            dims = [2, 1]
        else:
            cpos = dict(zip(c, [(0, 0)]))
            dims = [1, 1]
        layouts.append(cpos)
        rectangles.append(dims)

    positions = rpack.pack(rectangles)
    layout = {
        v: (vx + rx + w/2, vy + ry + h/2)
        for (rx, ry), (w, h), cpos in zip(positions, rectangles, layouts)
        for v, (vx, vy) in cpos.items()
    }
    if scale is not None:
        layout_array = np.array(list(layout[v] for v in G))
        return dict(zip(G, _scale_layout(layout_array, scale)))
    else:
        return layout

def _parse_graph(G):
    """Checks that G is a nx.Graph object or tries to make one."""
    if hasattr(G, "edges"):
        return G
    return nx.Graph(G)

def _get_center(layout_array):
    """Compute the center of `layout_array`"""
    mins = np.min(layout_array, axis=0)
    maxs = np.max(layout_array, axis=0)
    return (mins + maxs)/2

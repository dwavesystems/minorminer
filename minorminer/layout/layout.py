from collections import abc

import networkx as nx
import numpy as np


class Layout(abc.MutableMapping):
    def __init__(
        self,
        G,
        layout=None,
        d=None,
        center=None,
        scale=None,
        **kwargs
    ):
        """
        Compute a layout for G, i.e., a map from G to R^d.
        Parameters
        ----------
        G : NetworkX graph or NetworkX supported edges data structure (dict, list, ...)
            The graph you want to compute the layout for.
        layout : dict or function (default None)
            If a dict, this specifies a pre-computed layout for G. If a function, the function is called on G 
            `layout(G)` and should return a layout of G. If None, nx.spectral_layout is called.
        d : int (default None)
            The desired dimension of the layout, R^d. If None, set d to be the dimension of layout.
        center : tuple (default None)
            The desired center point of the layout. If None, set center to be the center of layout.
        scale : float (default None)
            The desired scale of the layout; i.e. the layout is in [center - scale, center + scale]^d space. If None, 
            set scale to be the scale of layout.
        kwargs : dict
            Keyword arguments are given to layout if it is a function.
        """
        # Ensure G is a graph object
        self.G = _parse_graph(G)

        # If passed in, save or compute the layout
        if layout is None:
            self.layout = nx.spectral_layout(self.G)
        elif callable(layout):
            self.layout = layout(self.G, **kwargs)
        else:
            # Assumes layout implements a mapping interface
            self.layout = layout

        # Set specs in the order of (user input, precomputed layout)
        self.d = d or self._d
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
        """
        If layout is set, also set layout_array and the layout specs.
        """
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
        """
        If layout_array is set, also set layout and the layout specs.
        """
        # Set the layout_array
        self._layout_array = value

        # Update the layout
        self.layout = {v: p for v, p in zip(sorted(self.G), value)}

    @property
    def d(self):
        return self._d

    @d.setter
    def d(self, value):
        """
        If the dimension is changed, change the dimension of the layout, if possible.
        """
        if value:
            self.layout_array = _dimension_layout(
                self.layout_array, value, self._d)

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, value):
        """
        If the center is changed, recenter the layout.
        """
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
        """
        If the scale is changed, rescale the layout.
        """
        if value:
            self.layout_array = _scale_layout(
                self.layout_array, value, self._scale, self._center)

    # The layout class should behave like a dictionary
    def __iter__(self):
        """
        Iterate through the keys of the dictionary layout.
        """
        yield from self.layout

    def __getitem__(self, key):
        """
        Get the layout value at the key vertex.
        """
        return self.layout[key]

    def __setitem__(self, key, value):
        """
        Set the layout value at the key vertex.
        """
        self.layout[key] = value

    def __delitem__(self, key):
        """
        Delete the layout value at the key vertex.
        """
        del self.layout[key]

    def __repr__(self):
        """
        Use the layout's dictionary representation.
        """
        return repr(self.layout)

    def __len__(self):
        """
        The length of a layout is the length of the layout dictionary.
        """
        return len(self.layout)

    def _set_layout_specs(self, empty=False):
        """
        Set the dimension, center, and scale of the layout_array currently in the Layout object.
        """
        if self.layout_array.size == 0:
            self._d = 0
            self._center = np.array([])
            self._scale = 0
        else:
            self._d = self.layout_array.shape[1]
            self._center = np.mean(self.layout_array, axis=0)
            self._scale = np.max(
                np.linalg.norm(
                    self.layout_array - self._center, float("inf"), axis=0
                )
            )


def _dimension_layout(layout_array, new_d, old_d=None):
    """
    This helper function transforms a layout from R^old_d to R^new_d by padding extra dimensions with 0's.
    Parameters
    ----------
    layout_array : numpy array
        An array whose rows are points in R^old_dim.
    new_d : int
        The new dimension to convert the layout to, must be larger than old_dim.
    old_d : int (default None)
        The current dimension of the laoyut. If None, the dimension is looked up via layout_array.
    Returns
    -------
    layout_array : numpy array
        A layout that has been projected into R^new_dim.
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
    """
    This helper function transforms a layout from [old_center - scale, old_center + scale]^d to
    [new_center - scale, new_center + scale]^d.
    Parameters
    ----------
    layout_array : numpy array
        An array whose rows are points in R^d.
    new_center : tuple or numpy array
        A point in R^d that is the desired center to move the layout to. 
    old_center : tuple or numpy array (default None)
        A point in R^d representing the center of the layout. If None, the approximate center of layout is computed 
        by calculating the center of mass (or centroid).
    Returns
    -------
    layout_array : numpy array
        A layout that has been centered at new_center.
    """
    # If old_center is empty, compute it
    if old_center is None:
        old_center = np.mean(layout_array, axis=0)

    # Translate the layout so that we have the desired new_center
    return layout_array - old_center + new_center


def _scale_layout(layout_array, new_scale, old_scale=None, center=None):
    """
    This helper function transforms a layout from [center - old_scale, center + old_scale]^d to 
    [center - new_scale, center + new_scale]^d.
    Parameters
    ----------
    layout_array : numpy array (default None)
        An array whose rows are points in R^d.
    new_scale : float
        The desired scale to transform the layout to.
    old_scale : float (default None)
        The scale of the layout. If None, the approximate scale of layout is computed by taking the maximum distance 
        from the center.
    center : tuple or numpy array (default None)
        A point in R^d representing the center of the layout. If None, the approximate center of layout is computed by 
        calculating the center of mass (centroid).
    Returns
    -------
    layout_array : numpy array
        A layout that has been scaled to [center - new_scale, center + new_scale]^d.
    """
    # If center is empty, compute it
    if center is None:
        center = np.mean(layout_array, axis=0)

    # Translate the layout so that its center is the origin
    L = layout_array - center

    # Compute the scale of the passed-in layout
    if old_scale is None:
        old_scale = np.max(
            np.linalg.norm(L, float("inf"), axis=0)
        )

    # Scale the thing
    scaled_L = (new_scale/old_scale) * L

    # Translate the layout back to where it was
    return scaled_L + center


def _parse_graph(G):
    """
    Checks that G is a nx.Graph object or tries to make one.
    """
    if hasattr(G, "edges"):
        return G
    return nx.Graph(G)

import random
from collections import abc

import networkx as nx
import numpy as np

from . import layout


class Placement(abc.MutableMapping):
    def __init__(
        self,
        S_layout,
        T_layout,
        placement=None,
        **kwargs
    ):
        """
        Compute a placement of S in T, i.e., map V(S) to 2^{V(T)}.
        Parameters
        ----------
        S_layout : layout.Layout
            A layout for S; i.e. a map from S to R^d.
        T_layout : layout.Layout or networkx.Graph
            A layout for T; i.e. a map from T to R^d.
        placement : dict or function (default None)
            If a dict, this specifies a pre-computed placement for S in T. If a function, the function is called on
            S_layout and T_layout `placement(S_layout, T_layout)` and should return a placement of S in T. If None, 
            a random placement of S in T is selected.
        kwargs : dict
            Keyword arguments are given to placement if it is a function.
        """
        self.S_layout = _parse_layout(S_layout)
        self.T_layout = _parse_layout(T_layout)

        # Layout dimensions should match
        if self.S_layout.d != self.T_layout.d:
            raise ValueError(
                "S_layout has dimension {} but T_layout has dimension {}. These must match.".format(
                    self.S_layout.d, self.T_layout.d)
            )

        # Extract the graphs
        self.S = self.S_layout.G
        self.T = self.T_layout.G

        if placement is None:
            T_vertices = list(self.T)
            self.placement = {v: [random.choice(T_vertices)] for v in self.S}
        elif callable(placement):
            self.placement = placement(S_layout, T_layout, **kwargs)
        else:
            self.placement = placement

    # The class should behave like a dictionary
    def __iter__(self):
        """
        Iterate through the keys of the dictionary placement.
        """
        yield from self.placement

    def __getitem__(self, key):
        """
        Get the placement value at the key vertex.
        """
        return self.placement[key]

    def __setitem__(self, key, value):
        """
        Set the placement value at the key vertex.
        """
        self.placement[key] = value

    def __delitem__(self, key):
        """
        Delete the placement value at the key vertex.
        """
        del self.placement[key]

    def __repr__(self):
        """
        Use the placement's dictionary representation.
        """
        return repr(self.placement)

    def __len__(self):
        """
        The length of a placement is the length of the placement dictionary.
        """
        return len(self.placement)


def _parse_layout(G_layout):
    """
    Ensures a Layout object was passed in and makes a copy to save in the Placement object.
    """
    if isinstance(G_layout, layout.Layout):
        return layout.Layout(G_layout.G, G_layout.layout)

    if isinstance(G_layout, dict):
        raise TypeError(
            "If you want to pass in a precomputed layout mapping, please create a Layout object; Layout(G, layout).")

    else:
        raise TypeError("Please use a Layout object.")

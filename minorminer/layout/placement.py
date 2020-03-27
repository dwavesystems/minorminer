import random
from collections import Counter, abc, defaultdict

import networkx as nx
import numpy as np
from scipy import spatial

from . import layout


def closest(S_layout, T_layout, subset_size=(1, 1), num_neighbors=1, **kwargs):
    """
    Maps vertices of S to the closest vertices of T as given by S_layout and T_layout. i.e. For each vertex u in
    S_layout and each vertex v in T_layout, map u to the v with minimum Euclidean distance (||u - v||_2).

    Parameters
    ----------
    subset_size : tuple (default (1, 1))
        A lower (subset_size[0]) and upper (subset_size[1]) bound on the size of subets of T that will be considered
        when mapping vertices of S.
    num_neighbors : int (default 1)
        The number of closest neighbors to query from the KDTree--the neighbor with minimium overlap is chosen.
        Increasing this reduces overlap, but increases runtime.

    Returns
    -------
    placement : dict
        A mapping from vertices of S (keys) to subsets of vertices of T (values).
    """
    # Extract the target graph
    T = T_layout.G

    # A new layout for subsets of T.
    T_subgraph_layout = {}

    # Get connected subgraphs to consider mapping to
    T_subgraphs = _get_connected_subgraphs(T, subset_size[1])

    # Calculate the barycenter (centroid) of each subset
    for k in range(subset_size[0], subset_size[1]+1):
        if k == 1:
            for subgraph in T_subgraphs[k]:
                v, = subgraph  # Unpack the subgraph of size 1
                T_subgraph_layout[subgraph] = T_layout[v]
        else:
            for subgraph in T_subgraphs[k]:
                T_subgraph_layout[subgraph] = np.mean(
                    np.array([T_layout[v] for v in subgraph]), axis=0)

    # Use scipy's KDTree to solve the nearest neighbor problem.
    # This requires a few lookup tables
    T_subset_lookup = {tuple(p): V for V, p in T_subgraph_layout.items()}
    layout_points = [tuple(p) for p in T_subgraph_layout.values()]
    overlap_counter = Counter()

    try:
        tree = spatial.KDTree(layout_points)  # This fails for the empty graph
    except ValueError:
        pass

    placement = {}
    for u, u_pos in S_layout.items():
        distances, v_indices = tree.query([u_pos], num_neighbors)

        # KDTree.query either returns a (num_neighbors, ) shaped arrays if num_neighbors == 1
        # or (1, num_neighbors) shaped arrays if num_neighbors != 1
        if num_neighbors != 1:
            v_indices = v_indices[0]
            distances = distances[0]

        placement[u] = _minimize_overlap(
            distances, v_indices, T_subset_lookup, layout_points, overlap_counter)

    return placement


def _get_connected_subgraphs(G, k, single_set=False):
    """
    Finds all connectected subgraphs S of G within a given subset_size.

    Parameters
    ----------
    G : networkx graph
        The graph you want to find all connected subgraphs of.
    k : int
        An upper bound of the size of connected subgraphs to find.

    Returns
    -------
    connected_subgraphs : dict
        The dictionary is keyed by size of subgraph and each value is a set containing
        frozensets of vertices that comprise the connected subgraphs.
        {
            1: { {v_1}, {v_2}, ... },
            2: { {v_1, v_2}, {v_1, v_3}, ... },
            ...,
            k: { {v_1, v_2, ..., v_m}, ... }
        }
    """
    connected_subgraphs = defaultdict(set)
    connected_subgraphs[1] = {frozenset((v,)) for v in G}

    for i in range(1, min(k, len(G))):
        # Iterate over the previous set of connected subgraphs.
        for X in connected_subgraphs[i]:
            # For each vertex in the set, iterate over its neighbors.
            for v in X:
                for u in G.neighbors(v):
                    connected_subgraphs[i + 1].add(X.union({u}))

    return connected_subgraphs


def _minimize_overlap(distances, v_indices, T_subset_lookup, layout_points, overlap_counter):
    """
    A greedy penalty-type model for choosing nonoverlapping chains.
    """
    subsets = {}
    for i, d in zip(v_indices, distances):
        subset = T_subset_lookup[layout_points[i]]
        subsets[subset] = d + sum(10**overlap_counter[v] for v in subset)

    cheapest_subset = min(subsets, key=subsets.get)
    overlap_counter.update(cheapest_subset)
    return cheapest_subset


class Placement(abc.MutableMapping):
    def __init__(
        self,
        S_layout,
        T_layout,
        placement=None,
        scale_ratio=None,
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
        scale_ratio : float (default None)
            If None, S_layout is not scaled. Otherwise, S_layout is scaled to scale_ratio*T_layout.scale.
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

        # Scale S if S_layout is bigger than T_layout
        if self.S_layout.scale > self.T_layout.scale:
            self.S_layout.scale = self.T_layout.scale
        # Or scale S to the user specified scale
        elif scale_ratio:
            self.S_layout.scale = scale_ratio*self.T_layout.scale

        if placement is None:
            self.placement = closest(S_layout, T_layout)
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

    else:
        raise TypeError(
            "If you want to pass in a precomputed layout mapping, please create a Layout object; Layout(G, layout).")

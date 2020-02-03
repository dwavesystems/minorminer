import math
import random
from collections import defaultdict

import networkx as nx
import numpy as np
from scipy.spatial.distance import euclidean


def to_vector(length, d):
    """
    If length is an integer, it creates a d-dimensional array with values length. Otherwise it creates an array based
    on the length iterable and checks that is the same dimension as d. 
    """
    if isinstance(length, (int, float)):
        return np.array(d*(length,))

    vector = np.array(length)
    assert vector.size == d, (
        f"You inputed a vector of size {vector.size} for a {d}-dimensional space; these should match."
    )
    return vector


def lattice_points_to_length(lattice_points):
    """
    Converts between a number of desired lattice points in a dimension to the length in that dimension; i.e., if you 
    want 5 lattice points in a dimension, it should be length 4 resulting in integers [0, 1, 2, 3, 4].
    """
    if isinstance(lattice_points, int):
        return lattice_points - 1
    else:
        return tuple(x-1 for x in lattice_points)


def convert_to_chains(placement):
    """
    Helper function to determine whether or not an input is in a chain-ready data structure. 
    """
    for v in placement.values():
        if isinstance(v, (list, frozenset, set)):
            return False
        return True


def border_round(point, border_max, d):
    """
    A rounding function that rounds to the border if beyond and otherwise rounds as normal.
    """
    border_vec = to_vector(border_max, d)
    new_point = []
    for p, b in zip(point, border_vec):
        if p <= 0:
            new_point.append(0)
        elif p >= b:
            new_point.append(b)
        else:
            new_point.append(round(p))
    return tuple(new_point)


def build_starting_points(G, m):
    """
    Helper function for pca.
    """
    starting_positions = defaultdict(list)
    pivots = [random.choice(list(G))]
    shortest_distance_to_pivots = {}

    for i in range(m):
        # Get shortest paths from a pivot
        shortest_paths = nx.shortest_path_length(G, pivots[i])

        # Assign the distances as coordinate i in each vector
        for v, dist in shortest_paths.items():
            starting_positions[v].append(dist)
            shortest_distance_to_pivots[v] = min(
                shortest_distance_to_pivots.get(v, float("inf")), dist)

        pivots.append(max(shortest_distance_to_pivots,
                          key=shortest_distance_to_pivots.get))

    return starting_positions

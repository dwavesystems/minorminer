import math
import random
from collections import defaultdict

import networkx as nx
import numpy as np
from minorminer.layout.layout import Layout
from scipy import ndimage, spatial


def to_vector(length, d):
    """
    If length is an integer, it creates a d-dimensional array with values length. Otherwise it creates an array based
    on the length iterable and checks that is the same dimension as d. 
    """
    if isinstance(length, (int, float)):
        return np.array(d*(length,))

    vector = np.array(length)
    assert vector.size == d, (
        "You inputted a vector of size {} for a {}-dimensional space; these should match.".format(
            vector.size, d)
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


def parse_layout(layout):
    """
    Take in a layout class object or a dictionary and return the dictionary representation.
    """
    if isinstance(layout, Layout):
        return layout.layout
    else:
        return layout


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


def minimum_bounding_rectangle(points):
    """
    Uses the rotating calipers algorithm to compute a rectangle that contains layout with minimum area.
    """
    # Compute the convex hull
    hull = spatial.ConvexHull(points)

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
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval


def build_starting_points(G, m, seed=None):
    """
    Helper function for pca.
    """
    if seed:
        random.seed(seed)

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


def minimize_overlap(distances, v_indices, T_vertex_lookup, layout_points, overlap_counter):
    """
    A greedy penalty-type model for choosing overlapping chains.
    """
    # KDTree.query either returns a single index or a list of indexes
    # In the case of a single index, there is nothing to do here.
    if isinstance(v_indices, (np.int64, np.int32)):
        return T_vertex_lookup[layout_points[v_indices]]

    subsets = {}
    for i in v_indices:
        subset = T_vertex_lookup[layout_points[i]]
        subsets[subset] = sum(d + 10**overlap_counter[v]
                              for d, v in zip(distances, subset))

    cheapest_subset = min(subsets, key=subsets.get)
    overlap_counter.update(cheapest_subset)
    return cheapest_subset

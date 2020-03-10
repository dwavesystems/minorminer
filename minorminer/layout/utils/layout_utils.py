import math
import random
from collections import defaultdict

import networkx as nx
import numpy as np
from scipy import ndimage, spatial


def convert_to_chains(placement):
    """
    Helper function to determine whether or not an input is in a chain-ready data structure.
    """
    for v in placement.values():
        if isinstance(v, (list, frozenset, set)):
            return False
        return True


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
    # KDTree.query either returns a single index or a list of indexes depending on how many neighbors are queried.
    if isinstance(v_indices, np.int64):
        return T_vertex_lookup[layout_points[v_indices]]

    subsets = {}
    for i in v_indices:
        subset = T_vertex_lookup[layout_points[i]]
        subsets[subset] = sum(d + 10**overlap_counter[v]
                              for d, v in zip(distances, subset))

    cheapest_subset = min(subsets, key=subsets.get)
    overlap_counter.update(cheapest_subset)
    return cheapest_subset


def graph_distances(G):
    """
    Compute the distance matrix of G.

    Parameters
    ----------
    G : NetworkX Graph
        The graph to find the distance matrix of.

    Returns
    -------
    G_distances : Numpy 2d array
        An array indexed by vertices of G (ordered by iterating through G) whose i,j value is d_G(i,j).
    """
    return np.array(
        [
            [V[v] for v in G] for u, V in nx.all_pairs_shortest_path_length(G)
        ]
    )


def p_norm(layout_vector, G_distances, k, p):
    """
    Compute the sum of differences squared between the l-metric and the graph distance as well as the gradient.

    Parameters
    ----------
    layout : Numpy Array
        A vector indexed by vertices of G (ordered by iterating through G) whose values are points in some metric space.
    G_distances : Numpy 2d array
        An array indexed by vertices of G (ordered by iterating through G) whose i,j value is d_G(i,j).
    k : int
        The dimension of the metric space. This will reshape the flattened array passed in to the cost function.
    p : int
        The order of the p-norm to use.

    Returns
    -------
    cost : float
        The sum of differences squared between the metric distance and the graph distance.
    """
    # Reconstitute the flattened array that scipy.optimize.minimize passed in
    n = len(G_distances)
    layout = layout_vector.reshape(n, k)

    # Difference between pairs of points in a 3d matrix
    diff = layout[:, np.newaxis, :] - layout[np.newaxis, :, :]

    # A 2d matrix of the distances between points
    dist = np.linalg.norm(diff, ord=p, axis=-1)

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


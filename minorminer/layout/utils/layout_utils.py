import random
from collections import defaultdict

import networkx as nx
import numpy as np


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

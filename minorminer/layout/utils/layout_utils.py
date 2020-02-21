import math
import random
from collections import defaultdict

import networkx as nx
import numpy as np
import jax.numpy as jnp
from scipy import ndimage, spatial


sqrt2_pi = jnp.sqrt(2)/jnp.pi  # Radius of the torus circles


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
    # In the case of a single index, there is nothing to do here.
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


def fast_graph_distances(G, k=1):
    """
    Compute an approximation of the distance matrix of G by measuring distance from k nodes, building a predecessor
    graph, and computing the distance in that graph. 

    Parameters
    ----------
    G : NetworkX Graph
        The graph to find the distance matrix of.

    k : int (default 1)
        The number of vertices of G from which to calculate distance from.

    Returns
    -------
    G_distances : dict
        A dict of dicts containing distance information of G.
    """
    distances = {}

    # Pick a random node to start from
    pivots = [random.choice(list(G))]
    shortest_distance_to_pivots = {}

    for pivot in pivots:
        # Compute shortest path predecessors
        pred, dist = nx.dijkstra_predecessor_and_distance(G, pivot)

        # Remember distance to all pivots
        for v, d in dist.items():
            shortest_distance_to_pivots[v] = min(
                shortest_distance_to_pivots.get(v, float("inf")), d)

        # Pick a point furthest from all previous pivots as the new pivot
        pivots.append(max(shortest_distance_to_pivots,
                          key=shortest_distance_to_pivots.get))

        # Build graph of predecessors
        H = nx.Graph([(v, p) for v, P in pred.items() for p in P])

        # Compute shortest paths in the predecessor graph and update distances with min
        if distances == {}:
            distances = dict(nx.all_pairs_shortest_path_length(H))
        else:
            for v, D in nx.all_pairs_shortest_path_length(H):
                for u, d in D.items():
                    distances[v][u] = min(distances[v][u], d)

        # Quit after k pivots
        if len(pivots) > k:
            break

    return distances


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
    return jnp.array(
        [
            [V[v] for v in G] for u, V in nx.all_pairs_shortest_path_length(G)
        ]
    )


def random_layout(G):
    """
    Randomly embed G in R^2 x T.

    Parameters
    ----------
    G : NetworkX Graph
        The graph to embed in R^2 x T.

    Returns
    -------
    layout : Numpy Array
        A vector indexed by vertices of G (ordered by iterating through G) whose values are points in R^2 x T.
    """
    # Function to get a random angle
    def random_angle(): return random.random()*jnp.pi*2

    return jnp.array(
        [
            tuple(pos) + (random_angle(), random_angle())
            for pos in nx.random_layout(G).values()
        ]
    )


def cost_function(layout, G_distances, distance_function, k):
    """
    Compute the sum of differences squared between the metric distance and the graph distance.

    Parameters
    ----------
    layout : Numpy Array
        A vector indexed by vertices of G (ordered by iterating through G) whose values are points in some metric space.

    G_distances : Numpy 2d array
        An array indexed by vertices of G (ordered by iterating through G) whose i,j value is d_G(i,j).

    distance_function : function
        A function that takes p, q as parameters and returns d(p,q).

    k : int
        The dimension of the metric space. This will reshape the flattened array passed in to the cost function.

    Returns
    -------
    cost : float
        The sum of differences squared between the metric distance and the graph distance.
    """
    # Reconstitute the flattened array that scipy.optimize.minimize passed in
    n = len(G_distances)
    unflat = layout.reshape(n, k)

    # Compute the distance in R^2 x T
    M_distances = metric_distances(unflat, distance_function)

    # Compute the cost
    return jnp.sum((G_distances - M_distances)**2)


def metric_distances(layout, distance_function):
    """
    Compute the distance matrix of a layout.

    Parameters
    ----------
    layout : Numpy Array
        A vector indexed by vertices of G (ordered by iterating through G) whose values are points in some metric space.

    distance_function : function
        A function that takes p, q as parameters and returns d(p,q).

    Returns
    -------
    M_distances : Numpy 2d array
        A matrix indexed by vertices of G (ordered by iterating through G) whose i,j value is d(i,j).
    """
    return jnp.array([[distance_function(p, q) for p in layout] for q in layout])


def R2xT_distance(p, q):
    """
    Computes the l_1-norm of R^2, the l_2-norm of T, and the l_1_norm of the product metric space.

    Parameters
    ----------
    p : Numpy array
        A point in R^2 x T. It is defined by an x, y pair and 2 angles, i.e., p = (x_1, x_2, theta_1, theta_2) where 
        x_* in R^2 and theta_* in [0, 2*pi].

    q : Numpy array
        A point in R^2 x T. It is defined by an x, y pair and 2 angles, i.e., q = (x_1, x_2, theta_1, theta_2) where 
        x_* in R^2 and theta_* in [0, 2*pi].

    Returns
    -------
    distance : float
        The l_1-norm of the l_1-norm of R^2 and the l_2 norm of T.
    """
    plane_p, plane_q = p[:2], q[:2]
    torus_p, torus_q = p[2:], q[2:]

    # JAX doesn't like this. I think it's because scipy uses numpy and not jax.numpy.
    # plane_dist = spatial.distance.cityblock(plane_p, plane_q)
    plane_dist = cityblock(plane_p, plane_q)
    torus_dist = torus_distance(torus_p, torus_q)

    return plane_dist + torus_dist


def cityblock(p, q):
    """
    Computes the l_1-norm.
    """
    return jnp.sum(jnp.abs(p-q))


def cornerblock(p, q):
    """
    Computes the l_1-norm rotated.
    """
    pp = jnp.array([p[0]+p[1], p[0]-p[1]])
    qq = jnp.array([q[0]+q[1], q[0]-q[1]])
    return jnp.sum(jnp.abs(pp-qq))


def torus_distance(s, t, radius=sqrt2_pi):
    """
    Computes the l_2-norm distance in a torus.

    Parameters
    ----------
    s : Numpy array
        A point on the torus. It is defined by 2 angles, i.e., s = (theta_1, theta_2) where theta_* in [0, 2*pi]. 
        Each angle represents a position on a circle with the given radius.

    t : Numpy array
        A point on the torus. It is defined by 2 angles, i.e., t = (theta_1, theta_2) where theta_* in [0, 2*pi]. 
        Each angle represents a position on a circle with the given radius.

    Returns
    -------
    distance : float
        The l_2-norm of the arc lengths of each circle.
    """
    # Pick the shorter direction around the circle
    diff_1 = jnp.abs(s[0] - t[0])
    theta_1 = jnp.min((diff_1, 2*jnp.pi - diff_1))
    diff_2 = jnp.abs(s[1] - t[1])
    theta_2 = jnp.min((diff_2, 2*jnp.pi - diff_2))

    arc_length_1 = radius*theta_1
    arc_length_2 = radius*theta_2

    return jnp.sqrt(arc_length_1**2 + arc_length_2**2)
#     return spatial.distance.euclidean(arc_length_1, arc_length_2)

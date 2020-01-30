import numpy as np
from scipy.spatial.distance import euclidean


def scale_vector(length, d):
    """
    If length is an integer, it creates a d-dimensional array with values length. Otherwise it creates an array based
    on the length iterable and checks that is the same dimension as d. 
    """
    if isinstance(length, int):
        return np.array(d*(length,))

    scale = np.array(length)
    assert scale.size == d, (
        f"You inputed a scale vector of size {scale.size} for a {d}-dimensional space."
    )
    return scale


def lattice_points_to_length(lattice_points):
    """
    Converts between a number of desired lattice points in a dimension to the length in that dimension; i.e., if you 
    want 5 lattice points in a dimension, it should be length 4 resulting in integers [0, 1, 2, 3, 4].
    """
    if isinstance(lattice_points, int):
        return lattice_points - 1
    else:
        return tuple(x-1 for x in lattice_points)

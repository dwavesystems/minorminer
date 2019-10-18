import networkx as nx
import random
import numpy as np


def planar_initialization_chimera(G, C):
    """Given a networkx graph G and a Chimera graph C, find a set of initial chains of G in C for use in minorminer.

    Find a planar layout of G, and then map each vertex v in G to a single random qubit in the Chimera unit tile
    closest to the coordinates of v.
    """

    pos = nx.drawing.kamada_kawai_layout(G)
    chimera_dim = int(np.ceil(np.sqrt(C.number_of_nodes()/8)))

    # normalize positions to [0,1]

    max_coord_value = max([max(abs(pos[x])) for x in G.nodes()]) + 1e-3

    initial_chains = {}
    for x in G.nodes():
        pos[x] = (pos[x]/max_coord_value + 1)/2.
        i, j = np.floor(chimera_dim*np.array(pos[x])).astype(int)
        unit_tile = [(a,b,c,d) for (a,b,c,d) in C.nodes() if (a==i) and (b==j)]
        if not(unit_tile):
            raise ValueError('Empty unit tile.')
        initial_chains[x] = [random.choice(unit_tile)]

    return initial_chains


def planar_initialization_pegasus(G, P):
    """Given a networkx graph G and a Pegasus graph P, find a set of initial chains of G in P for use in minorminer.

    Find a planar layout of G, and then map each vertex v in G to a single random qubit in the Pegasus unit tile
    closest to the coordinates of v.
    """

    pos = nx.drawing.kamada_kawai_layout(G)
    # P_N has v <= 8(3N-1)(N-1) vertices in main fabric
    # so N >= sqrt(v/24)+1/3
    pegasus_dim = int(np.ceil(np.sqrt(P.number_of_nodes() / 24) + 1 / 3.))

    # normalize positions to [0,1]

    max_coord_value = max([max(abs(pos[x])) for x in G.nodes()]) + 1e-3

    initial_chains = {}
    for x in G.nodes():
        pos[x] = (pos[x] / max_coord_value + 1) / 2.
        i, j = np.floor(pegasus_dim * np.array(pos[x])).astype(int)
        i2, j2 = np.floor((pegasus_dim-1) * np.array(pos[x])).astype(int)
        unit_tile = [(a, b, c, d) for (a, b, c, d) in P.nodes() if ((a == 0) and (b == i) and (d == j2)) or
                     ((a==1) and (b == j) and (d == i2))]
        if not (unit_tile):
            raise ValueError('Empty unit tile.')
        initial_chains[x] = [random.choice(unit_tile)]

    return initial_chains

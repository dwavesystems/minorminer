"""
This file provides an example of how the initial_chains option of minorminer's find_embedding() can be used to
generate better embeddings.

In this example, we will embed a random cubic graph with 250 vertices into a 2048-vertex chimera graph in two
difference ways. Firstly, we will embed using find_embedding()'s default parameters. Secondly, we will provide
initial chains to the algorithm. To do this, we will lay out vertices of the cubic graph on a plane using a
force-directed graph drawing algorithm, and then map the positions of those vertices to vertices in the Chimera
graph with similar positions.

Providing find_embedding() with initial chains can not only decrease the run-time of the algorithm, but also result
in embedding with fewer qubits and smaller chain sizes, both of which are likely to improve QPU performance.


"""
import random
import math
import networkx as nx
import dwave_networkx as dnx
import minorminer


def planar_initialization_chimera(S, C):
    """Given a networkx source graph S and a target chimera graph C, find a set of initial chains for S in C for use in
    find_embedding().

    Find a planar layout of S, and then map each vertex v in S to a single random qubit in the Chimera unit tile
    closest to the coordinates of v.

    """

    # Find a set of positions for the nodes of the source graph
    pos = nx.drawing.kamada_kawai_layout(S)

    # normalize positions to [0,1]
    max_coord_value = max([max(abs(pos[x])) for x in S.nodes()]) + 1e-3

    # Infer the size of the chimera graph from the number of nodes
    chimera_dim = int(math.ceil(math.sqrt(C.number_of_nodes()/8)))

    # Check if the chimera graph uses linear indices
    coordinates = isinstance(next(iter(C.nodes())), tuple)
    if not coordinates:
        mapping = {i: dnx.linear_to_chimera(i, chimera_dim) for i in C.nodes()}
        C = nx.relabel_nodes(C, mapping, copy=True)

    # Map the initial positions to nodes in the chimera graph
    initial_chains = {}
    for x in S.nodes():

        pos[x] = (pos[x]/max_coord_value + 1)/2.
        i, j = (math.floor(chimera_dim*pos[x][idx]) for idx in [0, 1])
        unit_tile = [(a, b, c, d) for (a, b, c, d) in C.nodes() if (a == i) and (b == j)]
        if not unit_tile:
            raise ValueError('Empty unit tile.')
        initial_chains[x] = [random.choice(unit_tile)]

    # Convert initial chains to linear indices
    if not coordinates:
        for x, chain in initial_chains.items():
            initial_chains[x] = [dnx.chimera_to_linear(*t, m=chimera_dim) for t in chain]

    return initial_chains


def planar_initialization_pegasus(S, P):
    """Given a networkx source graph S and a Pegasus graph P, find a set of initial chains for S in P for use in
    find_embedding().

    Find a planar layout of S, and then map each vertex v in S to a single random qubit in the Pegasus unit tile
    closest to the coordinates of v.
    """

    # Find a set of positions for the nodes of the source graph
    pos = nx.drawing.kamada_kawai_layout(S)

    # normalize positions to [0,1]
    max_coord_value = max([max(abs(pos[x])) for x in S.nodes()]) + 1e-3

    # Infer the size of the Pegasus graph from the number of nodes
    # P_N has v <= 8(3N-1)(N-1) vertices in main fabric, so N >= sqrt(v/24)+1/3
    pegasus_dim = int(math.ceil(math.sqrt(P.number_of_nodes() / 24) + 1 / 3.))

    # Check if the pegasus graph uses linear indices
    coordinates = isinstance(next(iter(P.nodes())), tuple)
    if not coordinates:
        mapping = {i: dnx.pegasus_coordinates(pegasus_dim).linear_to_pegasus(i) for i in P.nodes()}
        P = nx.relabel_nodes(P, mapping, copy=True)

    # Map the initial positions to nodes in the chimera graph
    initial_chains = {}
    for x in S.nodes():
        pos[x] = (pos[x] / max_coord_value + 1) / 2.
        i, j = (math.floor(pegasus_dim * pos[x][idx]) for idx in [0, 1])
        i2, j2 = (math.floor((pegasus_dim-1) * pos[x][idx]) for idx in [0, 1])
        unit_tile = [(a, b, c, d) for (a, b, c, d) in P.nodes() if ((a == 0) and (b == i) and (d == j2)) or
                     ((a == 1) and (b == j) and (d == i2))]
        if not unit_tile:
            raise ValueError('Empty unit tile.')
        initial_chains[x] = [random.choice(unit_tile)]

    # Convert initial chains to linear indices
    if not coordinates:
        for x, chain in initial_chains.items():
            initial_chains[x] = [dnx.pegasus_coordinates(pegasus_dim).pegasus_to_linear(t) for t in chain]

    return initial_chains


if __name__ == '__main__':

    degree = 3
    graph_size = 250
    chimera_dim = 16

    # generate a random graph to embed and a chimera graph.
    source = nx.generators.random_regular_graph(degree, graph_size)
    target = dnx.chimera_graph(chimera_dim)

    # find a set of initial chains for the graph to be embedded
    initial_chains = planar_initialization_chimera(source, target)

    # compare the embeddings with or without initial chains.
    for init_chains in [initial_chains, {}]:

        emb = minorminer.find_embedding(source, target, initial_chains=init_chains)
        if init_chains:
            print("\nEmbedding with chain initialization:")
        else:
            print("\nEmbedding without chain initialization: ")
        if len(emb):
            num_qubits = sum([len(emb[x]) for x in source.nodes()])
            max_chain_length = max([len(emb[x]) for x in source.nodes()])
            print("\tMaximum chain length: {},    number of qubits: {}".format(max_chain_length, num_qubits))
        else:
            print("\tFailed.")

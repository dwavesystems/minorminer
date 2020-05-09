import dwave_networkx as dnx
import networkx as nx
import dwave.embedding as dwe
import matplotlib.pyplot as plt
import random
import time
from busclique import clique_experiment
N = 4
#c = dnx.chimera_graph(N, N, 2)
#random.seed(16)
#c.remove_nodes_from(random.sample(list(c.nodes()), int(.02*len(c))))
#c.remove_edges_from(random.sample(list(c.edges()), int(.01*c.number_of_edges())))
#k = {v for _ in dwe.chimera.find_clique_embedding(N*4,N).values() for v in _};
#sg = c


from dwave_networkx.generators.pegasus import (get_tuple_defragmentation_fn, fragmented_edges,
    pegasus_coordinates, pegasus_graph)


def _pegasus_fragment_helper(m=None, target_graph=None):
    # This is a function that takes m or a target_graph and produces a
    # `processor` object for the corresponding Pegasus graph, and a function
    # that translates embeddings produced by that object back to the original
    # pegasus graph.  Consumed by `find_clique_embedding` and
    # `find_biclique_embedding`.

    # Organize parameter values
    if target_graph is None:
        if m is None:
            raise TypeError("m and target_graph cannot both be None.")
        target_graph = pegasus_graph(m)

    m = target_graph.graph['rows']     # We only support square Pegasus graphs

    # Deal with differences in ints vs coordinate target_graphs
    if target_graph.graph['labels'] == 'nice':
        back_converter = pegasus_coordinates.pegasus_to_nice
        back_translate = lambda embedding: {key: [back_converter(p) for p in chain]
                                      for key, chain in embedding.items()}
    elif target_graph.graph['labels'] == 'int':
        # Convert nodes in terms of Pegasus coordinates
        coord_converter = pegasus_coordinates(m)

        # A function to convert our final coordinate embedding to an ints embedding
        back_translate = lambda embedding: {key: list(coord_converter.iter_pegasus_to_linear(chain))
                                      for key, chain in embedding.items()}
    else:
        back_translate = lambda embedding: embedding

    # collect edges of the graph produced by splitting each Pegasus qubit into six pieces
    fragment_edges = list(fragmented_edges(target_graph))

    # Convert chimera fragment embedding in terms of Pegasus coordinates
    defragment_tuple = get_tuple_defragmentation_fn(target_graph)
    def embedding_to_pegasus(emb):
        emb = map(defragment_tuple, emb)
        emb = dict(enumerate(emb))
        emb = back_translate(emb)
        return emb

    return fragment_edges, embedding_to_pegasus

P = dnx.pegasus_graph(N)
#P.remove_nodes_from(random.sample(list(P.nodes()), int(.01*len(P))))
#P.remove_edges_from(random.sample(list(P.edges()), int(.01*P.number_of_edges())))


fragment_edges, embedding_to_pegasus = _pegasus_fragment_helper(target_graph=P)
coords = dnx.chimera_coordinates(6*N, 6*N, 2)
fe = coords.iter_chimera_to_linear_pairs(fragment_edges)
c = dnx.chimera_graph(6*N, 6*N, 2, edge_list=fe)
e2p = lambda e: embedding_to_pegasus(map(coords.iter_linear_to_chimera, e))
print("starting")
t = time.perf_counter()
nk = 0
n6 = 0
from collections import Counter
CLD = Counter()
try:
    for emb0 in clique_experiment.all_max_cliques(c, 6*(N-1)):
        emb = e2p(emb0)
        lens = sorted(map(len, emb.values()))
        if all(l<6 for l in lens):
            CLD[lens.count(4), lens.count(5)] += 1
            if n6 < lens.count(4):
                n6 = lens.count(4)            
                plt.figure(figsize=(10, 10))
                dnx.draw_chimera_embedding(c, dict(enumerate(emb0)), node_size=10)
                plt.savefig(f"{nk}.png")
                plt.close()
            nk += 1
        print(n6, len(emb), lens)
except KeyboardInterrupt: pass
print(CLD)
#embs = list(clique_experiment.all_max_cliques(c, 36))
print(time.perf_counter()-t, nk)

if 0:
    emb = dict(enumerate(emb))
    plt.figure(figsize=(20,20))
    dnx.draw_chimera_embedding(c.subgraph(sg), emb, node_size=100)
    plt.savefig('foo.png')
    for e in dwe.diagnose_embedding(emb, nx.complete_graph(24), c):
        print(e)
    t = time.perf_counter()

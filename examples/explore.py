import dwave_networkx as dnx
import networkx as nx
import dwave.embedding as dwe
import matplotlib.pyplot as plt
import random
import time
from busclique import busclique
N = 16

for k in range(0, 15*12+12):
    K = nx.complete_graph(k)

    p = dnx.pegasus_graph(N)
    #p.remove_nodes_from(random.sample(list(p.nodes()), int(.02*len(p))))
    #p.remove_edges_from(random.sample(list(p.edges()), int(.01*p.number_of_edges())))

    print(k, end=' ')
    t0 = time.perf_counter()
    emb = dict(zip(K, busclique.pegasus_clique(p, len(K))))
    print(time.perf_counter()-t0, end=' ')
    if k:
        print(max(len(c) for c in emb.values()))
    else:
        print({})
    

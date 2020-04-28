#!/usr/bin/python3
import dwave_networkx as dnx
import networkx as nx
import dwave.embedding as dwe
import matplotlib.pyplot as plt
import random
import time
c = dnx.chimera_graph(16, 16)
random.seed(16)
c.remove_nodes_from(random.sample(list(c.nodes()), int(.02*len(c))))
c.remove_edges_from(random.sample(list(c.edges()), int(.01*c.number_of_edges())))
k = {v for _ in dwe.chimera.find_clique_embedding(24,6).values() for v in _};
sg = c
import clique_experiment
print("starting")
emb = []
t = time.perf_counter()
emb = clique_experiment.run_experiment(c.subgraph(sg), 16)
print(time.perf_counter()-t)
for chain in emb:
    print(chain)
if 1:
    emb = dict(enumerate(emb))
    plt.figure(figsize=(20,20))
    dnx.draw_chimera_embedding(c.subgraph(sg), emb, node_size=100)
    plt.savefig('foo.png')
    for e in dwe.diagnose_embedding(emb, nx.complete_graph(24), c):
        print(e)
    t = time.perf_counter()

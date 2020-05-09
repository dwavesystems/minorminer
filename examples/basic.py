import dwave_networkx as dnx
import networkx as nx
import dwave.embedding as dwe
import matplotlib.pyplot as plt
import random
import time
from busclique import busclique
N = 16
K = nx.complete_graph(80-12)

c = dnx.chimera_graph(N, n=N, t=8)
emb = dict(zip(K, busclique.chimera_clique(c, len(K))))
print(dwe.diagnose_embedding(emb, K, c))

p = dnx.pegasus_graph(N)
#p.remove_nodes_from(random.sample(list(p.nodes()), int(.02*len(p))))
#p.remove_edges_from(random.sample(list(p.edges()), int(.01*p.number_of_edges())))

t0 = time.perf_counter()
emb = dict(zip(K, busclique.pegasus_clique(p, len(K))))
print(time.perf_counter()-t0)
print(sorted(map(len, emb.values())), len(emb))
print(dwe.verify_embedding(emb, K, p))
plt.figure(figsize=(10,10))
dnx.draw_pegasus_embedding(p, emb, node_size=20, crosses=True)
plt.savefig('out.png')

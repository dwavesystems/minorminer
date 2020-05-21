import dwave_networkx as dnx
import networkx as nx
import dwave.embedding as dwe
import matplotlib.pyplot as plt
import random
import time
from busclique import busclique

random.seed(1)
N = 16
K = nx.complete_graph(41)
c = dnx.chimera_graph(N, n=N, t=4)
c.remove_nodes_from(random.sample(list(c.nodes()), int(.05*len(c))))
c.remove_edges_from(random.sample(list(c.edges()), 10))
t = time.perf_counter()
cliq = busclique.chimera_clique(c, len(K))
print(time.perf_counter() - t, 'seconds')
emb = dict(zip(K, cliq))
print(list(dwe.diagnose_embedding(emb, K, c)))
plt.figure(figsize=(10,10))
dnx.draw_chimera_embedding(c, emb, node_size=20)
plt.savefig('chout.png')
plt.close()

random.seed(1)
N = 16
K = nx.complete_graph(149)
p = dnx.pegasus_graph(N)
p.remove_nodes_from(random.sample(list(p.nodes()), int(.01*len(p))))
p.remove_edges_from(random.sample(list(p.edges()), 8))

t = time.perf_counter()
cliq = busclique.pegasus_clique(p, len(K))
print(time.perf_counter() - t, 'seconds')

emb = dict(zip(K, cliq))
print(list(dwe.diagnose_embedding(emb, K, p)))
print(sorted(map(len, emb.values())))
plt.figure(figsize=(20,20))
dnx.draw_pegasus_embedding(p, emb, node_size=10, crosses=True)
plt.savefig('pout.png')
plt.close()

#t = time.perf_counter()
#cliq = dwe.pegasus.find_clique_embedding(len(K), target_graph = p)
#print(time.perf_counter() - t, 'seconds')



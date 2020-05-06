import dwave_networkx as dnx
import networkx as nx
import dwave.embedding as dwe
import matplotlib.pyplot as plt
import random
import time
from busclique import busclique
N = 12
K = nx.complete_graph(5)

c = dnx.chimera_graph(N, n=N, t=8)
emb = dict(zip(K, busclique.chimera_clique(c, len(K))))
print(dwe.verify_embedding(emb, K, c))

p = dnx.pegasus_graph(N)
emb = dict(zip(K, busclique.pegasus_clique(p, len(K))))
print(dwe.verify_embedding(emb, K, p))

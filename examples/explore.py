import dwave_networkx as dnx
import networkx as nx
import dwave.embedding as dwe
import matplotlib.pyplot as plt
import random
import time
from itertools import chain
from busclique import busclique
N = 16

p = dnx.pegasus_graph(4, fabric_only=False) 
count = [0]*len(p)
for w in range(2, 4*6-6):
    c = pegasus_clique_residency(p, w)
    for i, x in enumerate(c):
        count[i] += x
    plt.figure(figsize=(20,20))
    dnx.draw_pegasus(p, node_size=200, node_color = [c[v] for v in p],
                            cmap = plt.cm.plasma, crosses=True)
    plt.savefig(f'heatmap_{w}.png')
    plt.close()

plt.figure(figsize=(20,20))
dnx.draw_pegasus(p, node_size=200, node_color = [count[v] for v in p],
                        cmap = plt.cm.plasma, crosses=True)
plt.savefig('heatmap.png')



import minorminer
import networkx as nx
import dwave_networkx as dnx
import matplotlib.pyplot as plt

K3 = nx.Graph([('A', 'B'), ('B', 'C'), ('C', 'A')])
plt.subplot(2, 2, 1)
nx.draw(K3, with_labels=True)
C = dnx.chimera_graph(1, 2, coordinates=False)
plt.subplot(2, 2, 2)
dnx.draw_chimera(C, with_labels=True)

# Example with one blob for one node. Source node will use at least one.
blob = [4, 5, 12, 13]
suspend_chains = {'A': [blob]}
embedding = minorminer.find_embedding(K3, C, suspend_chains=suspend_chains)
plt.subplot(2, 2, 3)
dnx.draw_chimera_embedding(C, embedding, with_labels=True)

# Example with one blob for one node, and two blobs for another.
# Second source node is forced to use at least one in each blob.
blob_A0 = [4, 5]
blob_A1 = [12, 13]
blob_C0 = [6, 7, 14, 15]
suspend_chains = {'A': [blob_A0, blob_A1], 'C': [blob_C0]}
embedding = minorminer.find_embedding(K3, C, suspend_chains=suspend_chains)
plt.subplot(2, 2, 4)
dnx.draw_chimera_embedding(C, embedding, show_labels=True)

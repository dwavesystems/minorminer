# Copyright 2019 - 2020 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

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

"""
This file shows a visualization of the minorminer.find_embedding() algorithm.

At each step of the find_embedding() algorithm, a new chain is inserted for a node in the source graph, where chains
are allowed to overlap. After all chains have inserted, the algorithm iteratively removes and reinserts chains,
attempting to minimize the amount of overlap between them. Eventually all overlap will be removed, and the result is
a valid embedding.

In this example, a complete graph K_8 is embedded into a chimera_graph C_2.

"""

from minorminer import miner
import networkx as nx
import dwave_networkx as dnx
import matplotlib.pyplot as plt


# Parameters of the demo
wait_for_input = False                   # wait for user input to advance to the next step
G = nx.complete_graph(8)                # source graph
C = dnx.generators.chimera_graph(2)     # target graph


def show_current_embedding(emb):
    # visualize overlaps.
    plt.clf()
    dnx.draw_chimera_embedding(C, emb=emb, overlapped_embedding=True, show_labels=True)
    plt.show()
    if wait_for_input:
        plt.pause(0.001)
        input()
    else:
        plt.pause(1)


def compute_bags(C, emb):
    # Given an overlapped embedding, compute the set of source nodes embedded at every target node.
    bags = {v: [] for v in C.nodes()}
    for x, chain in emb.items():
        for v in chain:
            bags[v].append(x)
    return bags


# Run the algorithm.
plt.ion()
m = miner(G, C, random_seed=0)
found = False
emb = {}
print("Embedding K_8 into Chimera C(2).")
for iteration in range(3):
    if iteration == 0:
        print("\nInitialization phase...")
    elif iteration==1:
        print("\nOverfill improvement phase...")

    for v in G.nodes():
        if iteration > 0:
            # show embedding with current vertex removed.
            removal_emb = emb.copy()
            removal_emb[v] = []
            show_current_embedding(removal_emb)

        # run one step of the algorithm.
        emb = m.quickpass(varorder=[v], clear_first=False, overlap_bound=G.number_of_nodes())

        # check if we've found an embedding.
        bags = compute_bags(C, emb)
        overlap = max(len(bag) for bag in bags.values())
        if overlap==1 and iteration > 0 and not(found):
            print("\nEmbedding found. Chain length improvement phase...")
            found = True

        show_current_embedding(emb)

    max_chain_length = max(len(chain) for chain in emb.values())
    print("\tIteration {}: max qubit fill = {}, max chain length = {}".format(iteration, overlap, max_chain_length))
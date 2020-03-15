import random


def random_remove(S_layout, T_layout, chains, percent=2/3, **kwargs):
    """
    Randomly remove percent of qubits from each chain
    """
    for v, C in chains.items():
        chain_list = list(C)  # In case C is a set/frozenset or something

        # Shuffle and remove some qubits
        random.shuffle(chain_list)
        for _ in range(int(len(C)*percent)):
            chain_list.pop()

        # Update the chains
        chains[v] = chain_list

    return chains

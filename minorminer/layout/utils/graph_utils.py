from collections import defaultdict

import networkx as nx


def parse_graph(G):
    """
    Determines if a graph object or a collection of edges was passed in. Returns a NetworkX graph.
    """
    if hasattr(G, "edges"):
        return G
    return nx.Graph(G)


def get_connected_subgraphs(G, k_min, k_max, single_set=False):
    """
    Finds all connectected subgraphs S of G with `k_min <= |S| <= k_max`.

    Parameters
    ----------
    G : networkx graph
        The graph you want to find all connected subgraphs of.
    k_max : int
        An upper bound of the size of connected subgraphs that you want to find.
    k_min : int
        A lower bound of the size of connected subgraphs that you want to find.
    single_set : bool (optional, default False)
        Condenses all subsets into a single set of subgraphs.
    initial_emb : dict (optional, default None)
        If present, only subgraphs pertaining to `inital_emb` are returned.
    Returns
    -------
    connected_subgraphs : dict
        The dictionary is keyed by size of subgraph and each value is a set containing  
        frozensets of vertices that comprise the connected subgraphs.
        {
            1: { {v_1}, {v_2}, ... },
            2: { {v_1, v_2}, {v_1, v_3}, ... },
            ...,
            k: { {v_1, v_2, ..., v_m}, ... }
        }
    connected_subgraphs : set
        If the flag `single_set` is selected, the union of the above sets is returned.
    """
    assert 1 <= k_min and k_min <= k_max and k_max <= len(G), (
        "Why you pick bad numbers?")

    connected_subgraphs = defaultdict(set)
    connected_subgraphs[1] = {frozenset((v,)) for v in G}

    for i in range(1, k_max):
        # Iterate over the previous set of connected subgraphs.
        for X in connected_subgraphs[i]:
            # For each vertex in the set, iterate over its neighbors.
            for v in X:
                for u in G.neighbors(v):
                    connected_subgraphs[i + 1].add(X.union({u}))
    if single_set:
        all_subgraphs = set()
        for i in range(k_min, k_max + 1):
            all_subgraphs |= connected_subgraphs[i]

        return all_subgraphs

    return connected_subgraphs

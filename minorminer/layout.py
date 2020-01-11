from collections import defaultdict
from itertools import product

import minorminer as mm
import networkx as nx
from scipy.spatial.distance import euclidean


def get_embedding(S, T, layout="kamada_kawai_2d", placement="closest", chains="singleton", **kwargs):
    """
    Tries to embed S in T by computing layout aware initial_chains and passing them to minorminer.

    Parameters
    ----------
    S : NetworkX graph
        The graph you are embedding (source).
    T : NetworkX graph
        The graph you are embedding into (target).
    layout : str (default "kamada_kawai_2d")
        A strategy to be used when calling get_layout().
    placement : str (default "closest")
        A strategy to be used when calling get_placement().
    chains : str (default "singleton")
        A strategy to be used when calling get_chains().
    kwargs : dict 
        Keyword arguments are passed to minorminer.find_embedding()

    Returns
    -------
    emb : dict
        Output is dependant upon kwargs passed to minonminer, but more or less emb is a mapping from vertices of 
        S (keys) to chains in T (values).
    """
    S_layout = get_layout(S, strategy=layout)
    T_layout = get_layout(T, strategy=layout)
    placement = get_placement(S_layout, T_layout, strategy=placement)
    chains = get_chains(S, T, placement, strategy=chains)
    return mm.find_embedding(S, T, initial_chains=chains, **kwargs)


def get_layout(G, strategy="kamada_kawai_2d", **kwargs):
    """
    Helpful initial layouts to give to minorminer.

    Parameters
    ----------
    G : NetworkX graph
        The graph you want to compute the layout for.
    strategy : string (default "kamada_kawai_2d")
        Different layout strategies to use. Implemented strategies are:
            - kamada_kawai_{n}d where n >= 1, an integer.
    kwargs : dict
        Keyword arguments are passed to NetworkX layout functions.

    Returns
    -------
    layout : dict
        A mapping from vertices of G (keys) to points in the plane (values).
    """
    # NetworkX has a bug #3658.
    # Once fixed, these can collapse and `dim=n` can be part of `**kwargs`.
    if strategy == "kamada_kawai_1d":
        return nx.kamada_kawai_layout(G, dim=1, **kwargs)

    if strategy == "kamada_kawai_2d":
        return nx.kamada_kawai_layout(G, **kwargs)

    if strategy.startswith("kamada_kawai"):
        n = int(strategy.split('_')[-1].rstrip('d'))
        return nx.kamada_kawai_layout(
            G, center=n*(1/2, ), scale=1/2, pos=nx.random_layout(G, dim=n), dim=n, **kwargs
        )

    raise TypeError(
        f"Parameter strategy={strategy} is not a valid strategy.")


def get_placement(S_layout, T_layout, strategy="closest"):
    """
    Maps vertices of S to vertices of T.

    Parameters
    ----------
    S_layout : dict
        A layout for S; i.e. a map from S to the plane.
    T_layout : dict
        A layout for T; i.e. a map from T to the plane..
    strategy : string (default "closest")
        Different placement strategies to use. Implemented strategies are:
            - "closest" : For each vertex u in S_layout and each vertex v in T_layout, map u to the v with minimum 
                            Euclidean distance (||u - v||_2).
            - "injective" : This is the same as the "closest" strategy, except the map is injective. This runs in 
                            O(|S||T|log(|T|)) time.

    Returns
    -------
    placement : dict
        A mapping from vertices of S (keys) to vertices of T (values).
    """
    placement = {}

    if strategy == "closest":
        for u, u_pos in S_layout.items():
            # The vertex v of T that is closest (in layouts) to u
            placement[u] = min(
                T_layout, key=lambda v: euclidean(u_pos, T_layout[v]))
        return placement

    if strategy == "injective":
        X = nx.Graph()
        for u, u_pos in S_layout.items(euclidean(u_pos, v_pos))
        M = nx.bipartite.minimum_weight_full_matching(X, S_layout.keys())
        placement = {v: M[v] for v in S_layout.keys()}
        return placement

    raise TypeError(
        f"Parameter strategy={strategy} is not a valid strategy.")


def get_chains(S, T, placement, strategy="singleton"):
    """
    Given a placement (a map from vertices of S to vertices of T), form chains that will serve as embedding hints for 
    minerminor.

    Parameters
    ----------
    S : NetworkX graph
        The graph you are embedding (source).
    T : NetworkX graph
        The graph you are embedding into (target).
    placement : dict
        A mapping from vertices of S (keys) to vertices of T (values).
    strategy : string (default "singleton")
        Different chain creation strategies to use. For each of the following descriptions we assume that vertex u of S 
        is mapped to vertex v of T. Implemented strategies are:
            - "singleton" : A chain is of the form [v] for u.
            - "extend" : Heuristically (with minorminer) compute optimal paths between v and the image of each neighbor 
                            of u. The chain for u is the union of v together with these open paths.
            - "neighborhood" : A chain is of the form N_T(v) (closed neighborhood of v in T) for u.
            - "neighborhood_extension" : First the extend algorithm is run to get extended chains, then the closed 
                                            neighborhood of that chain is returned.

    Returns
    -------
    chains: dict
        A mapping from vertices of S (keys) to chains of T (values).
    """
    if strategy in ("neighborhood", "neighbourhood"):
        return {u: graph_utils.closed_neighbors(T, v) for u, v in placement.items()}

    if strategy == "singleton":
        return {u: [v] for u, v in placement.items()}

    if strategy == "extend":
        singleton_chains = {u: [v] for u, v in placement.items()}
        return extend_chains(S, T, singleton_chains)

    if strategy in ("neighborhood_extension", "neighbourhood_extension"):
        singleton_chains = {u: [v] for u, v in placement.items()}
        extended_chains = defaultdict(set)
        for u, C in extend_chains(S, T, singleton_chains).items():
            for v in C:
                extended_chains[u].update(
                    graph_utils.closed_neighbors(T, v))
        return extended_chains

    raise TypeError(
        f"Parameter strategy={strategy} is not a valid strategy.")


def extend_chains(S, T, initial_chains):
    """
    Extend initial_chains in T so that their structure matches that of S. That is, form an overlap embedding of S in T
    where the initial_chains are subsets of the overlap embedding chains. This is done via minorminer.

    Parameters
    ----------
    S : NetworkX graph
        The graph you are embedding (source).
    T : NetworkX graph
        The graph you are embedding into (target).
    initial_chains : dict
        A mapping from vertices of S (keys) to chains of T (values).

    Returns
    -------
    extended_chains: dict
        A mapping from vertices of S (keys) to chains of T (values).
    """
    # Extend the chains to minimal overlapped embedding
    miner = mm.miner(S, T, initial_chains=initial_chains)
    extended_chains = defaultdict(set)
    for u in S:
        # Revert to the initial_chains and compute the embedding where you tear-up u.
        emb = miner.quickpass(
            [u], clear_first=True, overlap_bound=S.number_of_nodes()
        )

        # Add the new chain for u and the singleton one too (in case it got moved in emb)
        extended_chains[u].update(set(emb[u]).union(initial_chains[u]))

        # For each neighbor v of u, grow u to reach v
        for v in S[u]:
            extended_chains[u].update(
                set(emb[v]).difference(initial_chains[v]))
    return extended_chains


def closed_neighbors(G, u, second=False):
    """
    Returns the closed neighborhood of u in G.

    Parameters
    ----------
    G : NetworkX graph
        The graph you are considering.
    u : NetworkX node
        The node you are computing the closed neighborhood of.
    second : bool (default False)
        If True, compute the closed 2nd neighborhood.

    Returns
    -------
    neighbors: set
        A set of vertices of G.
    """
    neighbors = set(v for v in G.neighbors(u))
    if second:
        return set((u,)) | neighbors | set(w for v in neighbors for w in G.neighbors(v))
    return set((u,)) | neighbors

from collections import defaultdict

from minorminer.layout.utils import layout_utils
import minorminer as mm


def pass_along(S, T, placement, extend=False):
    """
    Given a placement (a map, phi, from vertices of S to vertices (or subsets of vertices) of T), form the chain 
    [phi(u)] (or phi(u)) for each u in S.

    Parameters
    ----------
    T : NetworkX graph
        The graph you are embedding into (target).
    placement : dict
        A mapping from vertices of S (keys) to vertices of T (values).
    extend : bool (default False)
        If True, extend chains to mimic the structure of S in T.

    Returns
    -------
    chains: dict
        A mapping from vertices of S (keys) to chains of T (values).
    """
    # Test if you need to convert values or not
    if layout_utils.convert_to_chains(placement):
        chains = {u: [v] for u, v in placement.items()}
    else:
        chains = placement

    if extend:
        return extend_chains(S, T, chains)

    return chains


def neighborhood(S, T, placement, second=False, extend=False):
    """
    Given a placement (a map, phi, from vertices of S to vertices of T), form the chain N_T(phi(u)) (closed neighborhood 
    of v in T) for each u in S.

    Parameters
    ----------
    T : NetworkX graph
        The graph you are embedding into (target).
    placement : dict
        A mapping from vertices of S (keys) to vertices of T (values).
    second : bool (default False)
        If True, gets the closed 2nd neighborhood of each vertex. If False, get the closed 1st neighborhood of each
        vertex. 
    extend : bool (default False)
        If True, extend chains to mimic the structure of S in T.


    Returns
    -------
    chains: dict
        A mapping from vertices of S (keys) to chains of T (values).
    """
    chains = {u: closed_neighbors(T, v, second=second)
              for u, v in placement.items()}
    if extend:
        return extend_chains(S, T, chains)

    return chains


def extend_chains(S, T, initial_chains):
    """
    Extend chains in T so that their structure matches that of S. That is, form an overlap embedding of S in T
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

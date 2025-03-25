# Copyright 2024 D-Wave
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Methods provided for the generation of one or more disjoint embeddings. 
These methods sequentially generate disjoint embeddings of a source graph 
onto a target graph or provide supporting functionality.
"""
import warnings
import time
from time import perf_counter
from typing import Union, Optional, Callable, Iterable, List

import dwave_networkx as dnx
import networkx as nx
import numpy as np

from minorminer.subgraph import find_subgraph

from minorminer.utils.feasibility import (
    embedding_feasibility_filter,
    lattice_size_lower_bound,
)


def shuffle_graph(
    G: nx.Graph, seed: Union[int, np.random.RandomState, np.random.Generator] = None
) -> nx.Graph:
    """Shuffle the node and edge ordering of a networkx graph.

    For embedding methods that operate as a function of the node or edge
    ordering this can force diversification in the returned embeddings. Note
    that special orderings that encode graph structure or geometry may enhance
    embedder performance (shuffling may lead to longer search times).

    Args:
        G: A networkx graph
        seed: When provided, is used to shuffle the order of nodes and edges in
        the source and target graph. This can allow sampling from otherwise deterministic routines.

    Returns:
        nx.Graph: The same graph with modified node and edge ordering.
    """
    prng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    prng.shuffle(nodes)
    edges = list(G.edges())
    prng.shuffle(edges)
    _G = nx.Graph()
    _G.add_nodes_from(nodes)
    _G.add_edges_from(edges)
    return _G


def embeddings_to_array(embs: list, node_order=None, as_ndarray=False):
    """Convert list of embeddings into an array of embedding values

    Args:
        embs: A list of embeddings, each list entry in the
            form of a dictionary with integer values.
        node_order: An iterable giving the ordering of
            variables in each row. When not provided variables are ordered to
            match the first embedding :code:``embs[0].keys()``. node_order
            can define any subset of the source graph nodes (embedding
            keys).
        as_ndarray: The list is cast to an ndarray, this requires that
            the embedding is 1 to 1, that the embedding values (nodes of
            a target graph) can be cast as numpy numeric types.

    Raises:
        ValueError: If ``embs`` is empty and ``node_order`` not provided.

    Returns:
        An embedding matrix; each row defines an embedding ordered
        by node_order.
    """
    if node_order is None:
        if len(embs) == 0:
            if as_ndarray:
                raise ValueError("shape of ndarray cannot be inferred")
            else:
                return []

        else:
            node_order = embs[0].keys()

    if as_ndarray:
        if len(embs) == 0:
            return np.empty(shape=(0, len(node_order)))
        else:
            return np.asarray([[emb[v] for v in node_order] for emb in embs])

    else:
        return [[emb[v] for v in node_order] for emb in embs]


def array_to_embeddings(
    embs: list, node_order: Optional[Iterable] = None
) -> list[dict]:
    """Convert list of embedding lists (values) to dictionary

    Args:
        embs: A list of embeddings, each embedding is a list where values are
            chains in node order.
        node_order: An iterable giving the ordering of
            variables in each row. When not provided variables are ordered to
            match the first embedding :code:``embs[0].keys()``. node_order
            can define any subset of the source graph nodes (embedding
            keys).

    Returns:
        A list of embedding dictionaries
    """
    if not embs:
        return []

    if node_order is None:
        node_order = range(len(embs[0]))

    if len(embs) == len(node_order):
        embs = [{node_order[idx]: v for idx, v in enumerate(embs)}]
    else:
        embs = [{node_order[idx]: v for idx, v in enumerate(emb)} for emb in embs]
    return embs


def find_multiple_embeddings(
    S: nx.Graph,
    T: nx.Graph,
    *,
    max_num_emb: Union[None, int, float] = 1,
    use_filter: bool = True,
    embedder: callable = None,
    embedder_kwargs: dict = None,
    one_to_iterable: bool = False,
    shuffle_all_graphs: bool = False,
    seed: Union[int, np.random.RandomState, np.random.Generator] = None,
    timeout: float = float("Inf"),
) -> list:
    """Finds multiple disjoint embeddings of a source graph onto a target graph

    Uses a greedy strategy to deterministically find multiple disjoint
    embeddings of a source graph onto a target graph; nodes used are removed
    after each successful embedding.

    Embedding multiple times on a large graph can take significant time.
    It is recommended the user adjust both the timeout parameter and
    embedder_kwargs appropriately (incl. timeout), and also consider bounding
    the number of embeddings returned (with max_num_emb).

    Args:
        S: The source graph to embed.
        T: The target graph in which to embed.
        max_num_emb: Maximum number of embeddings to find.
            Defaults to 1, set to None to find the maximum possible
            number. Use of float('Inf') is deprecated and gives equivalent
            behaviour to None.
        use_filter: Specifies whether to check feasibility
            of embedding arguments independently of the embedder method. In some
            easy to embed cases use of a filter can slow down operation.
        embedder: Specifies the embedding search method,
            a callable taking ``S``, ``T``, as the first two parameters. Defaults to
            ``minorminer.subgraph.find_subgraph``. When the embedder fails it is
            expected to return an empty dictionary.
        embedder_kwargs: Additional arguments for the embedder
            beyond ``S`` and ``T``.
        one_to_iterable: Determines how embedding mappings are
            interpreted. Set to ``True`` to allow multiple target nodes to map to
            a single source node. Defaults to ``False`` for one-to-one embeddings.
        shuffle_all_graphs: If True, the tile-masked target graph and source graph are
            shuffled on each embedding attempt. Note that, if the embedder supports
            randomization this should be preferred by use of embedder_kwargs. If an
            embedder is a function of the node or edge order this allows
            diversification of the embeddings found.
        seed: seed for the ``numpy.random.Generator`` controlling shuffling (if
            invoked).
        timeout: Total time allowed across all embeddings in seconds. Time elapsed is
            checked before each call to embedder. The total runtime is thereby bounded
            by timeout plus the time required by one call to embedder.

    Returns:
        list: A list of disjoint embeddings. Each embedding follows the format
        dictated by the embedder. By default, each embedding defines a 1:1
        map from the source to the target graph as a dictionary without
        reusing target variables.
    """
    embs = []
    if timeout <= 0:
        return embs

    timeout_at = perf_counter() + timeout

    if embedder is None:
        embedder = find_subgraph
    if embedder_kwargs is None:
        embedder_kwargs = {}

    if shuffle_all_graphs is True:
        prng = np.random.default_rng(seed)
        _T = T.copy()
        _T = shuffle_graph(T, seed=prng)
    else:
        _T = T

    if max_num_emb == float("inf"):
        warnings.warn(
            'Use of Inf for "max_num_emb" has been deprecated in favor of "None".',
            DeprecationWarning,
            stacklevel=2,
        )
        max_num_emb = None

    if max_num_emb is None:
        max_num_emb = T.number_of_nodes() // S.number_of_nodes()
    else:
        max_num_emb = min(
            T.number_of_nodes() // S.number_of_nodes(), round(max_num_emb)
        )

    if shuffle_all_graphs:
        _S = shuffle_graph(S, seed=prng)
    else:
        _S = S

    for _ in range(max_num_emb):
        if (
            use_filter
            and embedding_feasibility_filter(_S, _T, not one_to_iterable) is False
        ):
            emb = {}
        else:
            if perf_counter() >= timeout_at:
                break
            else:
                emb = embedder(_S, _T, **embedder_kwargs)

        if len(emb) == 0:
            break
        elif max_num_emb > 1:
            if len(embs) == 0:
                _T = T.copy()
            if one_to_iterable:
                _T.remove_nodes_from(n for c in emb.values() for n in c)
            else:
                _T.remove_nodes_from(emb.values())

        embs.append(emb)
    return embs


def lattice_size(T: nx.Graph) -> int:
    """Determines the cellular (square) dimension of a dwave_networkx lattice

    The lattice size is the parameter ``m`` of a dwave_networkx graph, also
    called number of rows, or in the case of a chimera graph max(m,n). This
    upper bounds the ``sublattice_size`` for ``find_sublattice_embeddings``.

    Args:
        T: The target graph in which to embed. The graph must be of type
            zephyr, pegasus or chimera and constructed by dwave_networkx.
    Returns:
        int: The maximum possible size of a tile.
    """
    # Possible feature enhancement, determine a stronger upper bound akin
    # to lattice_size_lower_bound, accounting for defects, the
    # degree distribution and other simple properties.

    return max(T.graph.get("rows"), T.graph.get("columns"))


def _is_valid_embedding(emb: dict, S: dict, T: dict, one_to_iterable: bool = True):
    """Check embedding validity.

    With special handling of 1:1 mappings."""
    from minorminer.utils.diagnostic import is_valid_embedding

    if one_to_iterable:
        return is_valid_embedding(emb, S, T)
    else:
        return is_valid_embedding({k: (v,) for k, v in emb.items()}, S, T)


def _mapped_proposal(emb: dict, f: Callable, one_to_iterable: bool = True):
    if one_to_iterable:
        return {k: tuple(f(n) for n in c) for k, c in emb.items()}
    else:
        return {k: f(n) for k, n in emb.items()}


def find_sublattice_embeddings(
    S: nx.Graph,
    T: nx.Graph,
    *,
    tile: nx.Graph = None,
    sublattice_size: int = None,
    max_num_emb: Optional[int] = 1,
    use_filter: bool = True,
    use_tile_embedding: Optional[bool] = None,
    tile_embedding: Optional[dict] = None,
    seed: Union[int, np.random.RandomState, np.random.Generator] = None,
    embedder: callable = None,
    embedder_kwargs: dict = None,
    one_to_iterable: bool = False,
    shuffle_all_graphs: bool = False,
    shuffle_sublattice_order: bool = False,
    timeout: float = float("Inf"),
) -> list:
    """Searches for embeddings on sublattices of the target graph.

    Selects a sublattice (subgrid) of the graph T using dwave_networkx
    sublattice tiles and mappings approprate to the target topology, and
    attemps to embed. When successful the nodes are removed and the
    process is repeated until all sublattices are explored or sufficient
    embeddings are yielded.

    Embedding multiple times on a large graph can take significant time.
    It is recommended the user adjust ``embedder_kwargs`` appropriately such
    as timeout, and also consider bounding the number of embeddings
    returned (with ``max_num_emb``).

    See https://doi.org/10.3389/fcomp.2023.1238988 for examples of usage.

    Args:
        S: The source graph to embed.
        T: The target graph in which to embed. If
            raster_embedding is not None, the graph must be of type zephyr,
            pegasus, or chimera and constructed by dwave_networkx.
        tile: A mask applied to the target graph ``T`` defining a restricted
            space in which to search for embeddings; the tile family
            should match that of ``T``. If the tile is not
            provided, it is generated as a fully yielded square sublattice
            of ``T``, with ``m=sublattice_size``.
            If ``tile==S``, ``embedder`` can be ignored since a 1:1 mapping is
            necessary on each subgraph and easily verified by checking the
            mask is complete.
        sublattice_size: Parameterizes the tile when it is not provided
           as an input argument: defines the number of rows and columns
           of a square sublattice (parameter m of the dwave_networkx graph
           family matching T). ``lattice_size_lower_bound()``
           provides a lower bound based on a fast feasibility filter.
        max_num_emb: Maximum number of embeddings to find.
            Defaults to 1, set to None for unbounded (try unlimited search on
            all lattice offsets).
        use_filter: Specifies whether to check feasibility of arguments for
            embedding independently of the embedder routine.  In some
            easy to embed cases use of a filter can slow down operation.
        use_tile_embedding: A single embedding for a tile is generated, and then
            reused for a maximum number of sublattices. If, due to defects or
            overlap with an existing assignment, the embedding fails at a particular
            offset embedder is not invoked.
        tile_embedding: When provided, this should be an embedding from the source
            to the tile. If not provided it is generated by the embedder method. Note
            that `one_to_iterable` should be adjusted to match the value type.
        embedder: Specifies the embedding search method, a callable taking ``S``, ``T`` as
            the first two arguments. Defaults to minorminer.subgraph.find_subgraph. Note
            that if `one_to_iterable` should be adjusted to match the return type.
            When the embedder fails it is expected to return an empty dictionary.
        embedder_kwargs: Dictionary specifying arguments for the embedder
            other than ``S``, ``T``.
        one_to_iterable: Specifies whether the embedder returns (and/or tile_embedding is)
            a dict with iterable values (True), or a 1:1 mapping (False).
        shuffle_all_graphs: If True, the tile-masked target graph and source graph are
            shuffled on each embedding attempt. Note that, if the embedder supports
            randomization this should be preferred by use of embedder_kwargs. If an
            embedder is a function of the node or edge order this allows
            diversification of the embeddings found.
        shuffle_sublattice_order: If True the ordering of sublattices (displacement
            of the tile (mask) on the target graph is randomized. This can allow
            for diversification of the embeddings found.
        seed: seed for the `numpy.random.Generator` controlling shuffling (if
            invoked).
        timeout: Total time allowed across all embeddings in seconds. Time elapsed is
            checked before each sublattice search begins, and before each call of the
            embedder function. The total runtime is thereby bounded by timeout plus the
            time required by one call to embedder.

    Raises:
        ValueError: If the target graph ``T`` is not of type zephyr, pegasus, or chimera.
            If the tile_embedding is incompatible with the source graph and tile.

    Returns:
        list: A list of disjoint embeddings.
    """
    if timeout <= 0:
        return []
    timeout_at = perf_counter() + timeout
    if sublattice_size is None and tile is None:
        return find_multiple_embeddings(
            S=S,
            T=T,
            max_num_emb=max_num_emb,
            seed=seed,
            embedder=embedder,
            embedder_kwargs=embedder_kwargs,
        )

    if use_filter and tile is None:
        feasibility_bound = lattice_size_lower_bound(
            S=S, T=T, one_to_one=not one_to_iterable
        )
        if feasibility_bound is None or sublattice_size < feasibility_bound:
            warnings.warn("sublattice_size < lower bound: embeddings will be empty.")
            return []

    # A possible feature enhancement might allow for sublattice_size (m) to be
    # replaced by shape: (m,t) [zephyr] or (m,n,t) [Chimera]
    family = T.graph.get("family")
    if family == "chimera":
        sublattice_mappings = dnx.chimera_sublattice_mappings
        t = T.graph["tile"]
        if tile is None:
            tile = dnx.chimera_graph(m=sublattice_size, n=sublattice_size, t=t)
        elif (
            use_filter
            and embedding_feasibility_filter(S, tile, not one_to_iterable) is False
        ):
            warnings.warn("tile is infeasible: embeddings will be empty.")
            return []

    elif family == "pegasus":
        sublattice_mappings = dnx.pegasus_sublattice_mappings
        if tile is None:
            tile = dnx.pegasus_graph(m=sublattice_size)
        elif (
            use_filter
            and embedding_feasibility_filter(S, tile, not one_to_iterable) is False
        ):
            warnings.warn("tile is infeasible: embeddings will be empty.")
            return []

    elif family == "zephyr":
        sublattice_mappings = dnx.zephyr_sublattice_mappings
        t = T.graph["tile"]
        if tile is None:
            tile = dnx.zephyr_graph(m=sublattice_size, t=t)
        elif (
            use_filter
            and embedding_feasibility_filter(S, tile, not one_to_iterable) is False
        ):
            warnings.warn("tile is infeasible: embeddings will be empty.")
            return []

    else:
        raise ValueError(
            "source graphs must a graph constructed by "
            "dwave_networkx as chimera, pegasus or zephyr type"
        )
    if tile_embedding is not None:
        if not _is_valid_embedding(tile_embedding, S, tile, one_to_iterable):
            raise ValueError("tile_embedding is invalid for S and tile")

        use_filter = False  # Unnecessary

    if use_tile_embedding is None:
        use_tile_embedding = tile == S  # Trivial 1:1
        if use_tile_embedding and tile_embedding is None:
            tile_embedding = {i: i for i in tile.nodes}
            use_filter = False  # Unnecessary

    if use_filter or (use_tile_embedding and tile_embedding is None):
        # Check defect-free tile embedding is viable

        # With respect to filter:
        # * This assumes that an embedder that returns on a graph G
        # will also return an embedding for any graph where G is
        # a subgraph. Can be violated given heuristic search
        defect_free_embs = find_multiple_embeddings(
            S,
            tile,
            max_num_emb=1,
            use_filter=False,
            seed=seed,
            embedder=embedder,
            embedder_kwargs=embedder_kwargs,
            timeout=timeout_at - perf_counter(),
        )
        if len(defect_free_embs) == 0:
            # If embedding is infeasible on the tile*, it will be infeasible
            # on all subgraphs thereof (assuming sufficient time was provided
            # in embedder_kwargs)
            return []

        if use_tile_embedding:
            tile_embedding = defect_free_embs[0]
            # Apply sufficient restriction on the tile
            if one_to_iterable:
                tile = tile.subgraph({v for c in tile_embedding.values() for v in c})
            else:
                tile = tile.subgraph({v for v in tile_embedding.values()})
    if max_num_emb is None:
        max_num_emb = T.number_of_nodes() // S.number_of_nodes()
    else:
        max_num_emb = min(T.number_of_nodes() // S.number_of_nodes(), max_num_emb)

    embs = []
    if max_num_emb == 1 and seed is None:
        _T = T
    else:
        _T = T.copy()

    if shuffle_all_graphs or shuffle_sublattice_order:
        prng = np.random.default_rng(seed)
    if shuffle_sublattice_order:
        sublattice_iter = list(sublattice_mappings(tile, _T))
        prng.shuffle(sublattice_iter)
    else:
        sublattice_iter = sublattice_mappings(tile, _T)

    for f in sublattice_iter:
        if perf_counter() > timeout_at:
            break
        if use_tile_embedding:
            proposal = _mapped_proposal(tile_embedding, f, one_to_iterable)
            if _is_valid_embedding(proposal, S, T, one_to_iterable):
                sub_embs = [proposal]
            else:
                sub_embs = []
        else:
            Tr = _T.subgraph([f(n) for n in tile])
            sub_embs = find_multiple_embeddings(
                S,
                Tr,
                max_num_emb=max_num_emb - len(embs),
                use_filter=use_filter,
                seed=seed,
                embedder=embedder,
                embedder_kwargs=embedder_kwargs,
                timeout=timeout_at - perf_counter(),
            )
        embs += sub_embs
        if len(embs) >= max_num_emb:
            break

        for emb in sub_embs:
            # A potential feature extension would be to generate many
            # overlapping embeddings and solve an independent set problem. This
            # may allow additional parallel embeddings.
            if one_to_iterable:
                _T.remove_nodes_from([v for c in emb.values() for v in c])
            else:
                _T.remove_nodes_from(emb.values())
    return embs

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

import dwave_networkx as dnx
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Union, Tuple

from minorminer.subgraph import find_subgraph

from minorminer.utils.feasibility import (
    embedding_feasibility_filter,
    lattice_size_lower_bound,
)


def visualize_embeddings(
    G: nx.Graph,
    embeddings: list,
    S: nx.Graph = None,
    one_to_iterable: bool = False,
    shuffle_colormap: bool = True,
    seed: Union[int, np.random.RandomState, np.random.Generator] = None,
    **kwargs,
) -> Tuple[dict, dict]:
    """Visualizes the embeddings using dwave_networkx's layout functions.

    This function visualizes embeddings of a source graph onto a target graph
    using specialized layouts for structured graphs (chimera, pegasus, or zephyr)
    or general layouts for unstructured graphs. Node and edge colors are used
    to differentiate embeddings.

    Args:
        G: The target graph to be visualized. If the graph
            represents a specialized topology, it must be constructed using
            dwave_networkx (e.g., chimera, pegasus, or zephyr graphs).
        embeddings: A list of embeddings where each embedding is a dictionary
            mapping nodes of the source graph to nodes in the target graph.
        S: The source graph to visualize (optional). If provided, only edges
            corresponding to the source graph embeddings are visualized.
        one_to_iterable: Specifies how embeddings are interpreted. Set to `True`
            to allow multiple target nodes to map to a single source node.
            Defaults to `False` for one-to-one embeddings.
        shuffle_colormap: A sequential colormap is used. If shuffle_colormap is
            False the sequential ordering determines a sequence of related colors
            otherwise colors are randomized.
        seed: A seed for the pseudo-random number generator. When provided,
            it randomizes the colormap assignment for embeddings.
        **kwargs: Additional keyword arguments passed to the drawing functions
            (e.g., `node_size`, `font_size`, `width`).

    Returns:
        Two dictionaries the first mapping plotted nodes to the source index.
        The second mapping plotted edges to the source index. The source embedding
        is embeddings[index]. Background edges/nodes are mapped to nan.

    Draws:
        - Specialized layouts: Uses dwave_networkx's `draw_chimera`,
          `draw_pegasus`, or `draw_zephyr` functions if the graph family is identified.
        - General layouts: Falls back to networkx's `draw_networkx` for
          graphs with unknown topology.
    """
    ax = plt.gca()
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("lightgrey")

    # Create node color mapping
    node_color_dict = {q: float("nan") for q in G.nodes()}

    if shuffle_colormap:
        _embeddings = embeddings.copy()
        prng = np.random.default_rng(seed)
        prng.shuffle(_embeddings)
    else:
        _embeddings = embeddings

    if S is None:
        if one_to_iterable:
            node_color_dict.update(
                {
                    q: idx
                    for idx, emb in enumerate(_embeddings)
                    for c in emb.values()
                    for q in c
                }
            )
        else:
            node_color_dict.update(
                {q: idx for idx, emb in enumerate(_embeddings) for q in emb.values()}
            )
    else:
        node_set = set(S.nodes())
        if one_to_iterable:
            node_color_dict.update(
                {
                    q: idx if n in node_set else float("nan")
                    for idx, emb in enumerate(_embeddings)
                    for n, c in emb.items()
                    for q in c
                }
            )
        else:
            node_color_dict.update(
                {
                    q: idx
                    for idx, emb in enumerate(_embeddings)
                    for n, q in emb.items()
                    if n in node_set
                }
            )
    # Create edge color mapping
    edge_color_dict = {}
    if S is not None:
        edge_color_dict = {
            (tu, tv): idx
            for idx, emb in enumerate(_embeddings)
            for u, v in S.edges()
            if u in emb and v in emb
            for tu in (emb[u] if one_to_iterable else [emb[u]])
            for tv in (emb[v] if one_to_iterable else [emb[v]])
            if G.has_edge(tu, tv)
        }
        if one_to_iterable:
            # Feature enhancement? We could consider formatting these lines
            # differently to distinguish chain couplers from logical couplers
            for idx, emb in enumerate(_embeddings):
                for chain in emb.values():
                    Gchain = G.subgraph(chain)
                    edge_color_dict.update({e: idx for e in Gchain.edges()})

    else:
        edge_color_dict = {
            (v1, v2): node_color_dict[v1]
            for v1, v2 in G.edges()
            if node_color_dict[v1] == node_color_dict[v2]
        }

    # Default drawing arguments
    draw_kwargs = {
        "G": G,
        "node_color": [node_color_dict[q] for q in G.nodes()],
        "edge_color": "lightgrey",
        "node_shape": "o",
        "ax": ax,
        "with_labels": False,
        "width": 1,
        "cmap": cmap,
        "edge_cmap": cmap,
        "node_size": 300 / np.sqrt(G.number_of_nodes()),
    }
    draw_kwargs.update(kwargs)

    topology = G.graph.get("family")
    # Draw the combined graph with color mappings
    if topology == "chimera":
        pos = dnx.chimera_layout(G)
        dnx.draw_chimera(**draw_kwargs)
    elif topology == "pegasus":
        pos = dnx.pegasus_layout(G)
        dnx.draw_pegasus(**draw_kwargs)
    elif topology == "zephyr":
        pos = dnx.zephyr_layout(G)
        dnx.draw_zephyr(**draw_kwargs)
    else:
        pos = nx.spring_layout(G)
        nx.draw_networkx(**draw_kwargs)
    if len(edge_color_dict) > 0:
        # Recolor specific edges on top of the original graph
        nx.draw_networkx_edges(
            G,
            pos=pos,
            edgelist=list(edge_color_dict.keys()),
            edge_color=list(edge_color_dict.values()),
            width=1,
            edge_cmap=cmap,
            ax=ax,
        )
    return node_color_dict, edge_color_dict


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
            match the first embedding :code:`embs[0].keys()`. node_order
            can define any subset of the source graph nodes (embedding
            keys).
        as_ndarray: The list is cast to an ndarray, this requires that
            the embedding is 1 to 1, that the embedding values (nodes of
            a target graph) can be cast as numpy numeric types.

    Raises:
        ValueError: If `embs` is empty and `node_order` not provided.

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


def find_multiple_embeddings(
    S: nx.Graph,
    T: nx.Graph,
    *,
    max_num_emb: int = 1,
    use_filter: bool = False,
    embedder: callable = None,
    embedder_kwargs: dict = None,
    one_to_iterable: bool = False,
    shuffle_all_graphs: bool = False,
    seed: Union[int, np.random.RandomState, np.random.Generator] = None,
) -> list:
    """Finds multiple disjoint embeddings of a source graph onto a target graph

    Uses a greedy strategy to deterministically find multiple disjoint
    embeddings of a source graph onto a target graph; nodes used are removed
    after each successful embedding.

    Embedding multiple times on a large graph can take significant time.
    It is recommended the user adjust embedder_kwargs appropriately such
    as timeout, and also consider bounding the number of embeddings
    returned (with max_num_emb).

    Args:
        S: The source graph to embed.
        T: The target graph in which to embed.
        max_num_emb: Maximum number of embeddings to find.
            Defaults to 1, set to float('Inf') to find the maximum possible
            number.
        use_filter: Specifies whether to check feasibility
            of embedding arguments independently of the embedder method.
        embedder: Specifies the embedding search method,
            a callable taking `S`, `T`, as the first two parameters. Defaults to
            `minorminer.subgraph.find_subgraph`.
        embedder_kwargs: Additional arguments for the embedder
            beyond `S` and `T`.
        one_to_iterable: Determines how embedding mappings are
            interpreted. Set to `True` to allow multiple target nodes to map to
            a single source node. Defaults to `False` for one-to-one embeddings.
        shuffle_all_graphs: If True, the tile-masked target graph and source graph are
            shuffled on each embedding attempt. Note that, if the embedder supports
            randomization this should be preferred by use of embedder_kwargs. If an
            embedder is a function of the node or edge order this allows
            diversification of the embeddings found.
        seed: seed for the `numpy.random.Generator` controlling shuffling (if
            invoked).

    Returns:
        list: A list of disjoint embeddings. Each embedding follows the format
            dictated by the embedder. By default, each embedding defines a 1:1
            map from the source to the target graph as a dictionary without
            reusing target variables.
    """
    embs = []
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

    max_num_emb = min(int(T.number_of_nodes() / S.number_of_nodes()), max_num_emb)

    if shuffle_all_graphs:
        _S = shuffle_graph(S, seed=prng)
    else:
        _S = S

    for _ in range(max_num_emb):
        if (
            use_filter
            and embedding_feasibility_filter(_S, _T, not one_to_iterable) is False
        ):
            emb = []
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


def find_sublattice_embeddings(
    S: nx.Graph,
    T: nx.Graph,
    *,
    tile: nx.Graph = None,
    sublattice_size: int = None,
    max_num_emb: int = 1,
    use_filter: bool = False,
    seed: Union[int, np.random.RandomState, np.random.Generator] = None,
    embedder: callable = None,
    embedder_kwargs: dict = None,
    one_to_iterable: bool = False,
    shuffle_all_graphs: bool = False,
    shuffle_sublattice_order: bool = False,
) -> list:
    """Searches for embeddings on sublattices of the target graph.

    Selects a sublattice (subgrid) of the graph T using dwave_networkx
    sublattice tiles and mappings approprate to the target topology, and
    attemps to embed. When successful the nodes are removed and the
    process is repeated until all sublattices are explored or sufficient
    embeddings are yielded.

    Embedding multiple times on a large graph can take significant time.
    It is recommended the user adjust `embedder_kwargs` appropriately such
    as timeout, and also consider bounding the number of embeddings
    returned (with `max_num_emb`).

    See https://doi.org/10.3389/fcomp.2023.1238988 for examples of usage.

    Args:
        S: The source graph to embed.
        T: The target graph in which to embed. If
            raster_embedding is not None, the graph must be of type zephyr,
            pegasus, or chimera and constructed by dwave_networkx.
        tile: A mask applied to the target graph `T` defining a restricted
            space in which to search for embeddings; the tile family
            should match that of `T`. If the tile is not
            provided, it is generated as a fully yielded square sublattice
            of `T`, with `m=sublattice_size`.
            If `tile==S`, `embedder` can be ignored since a 1:1 mapping is
            necessary on each subgraph and easily verified by checking the
            mask is complete.
        sublattice_size: Parameterizes the tile when it is not provided
           as an input argument: defines the number of rows and columns
           of a square sublattice (parameter m of the dwave_networkx graph
           family matching T).
           :code:`lattice_size_lower_bound()`
           provides a lower bound based on a fast feasibility filter.
        max_num_emb: Maximum number of embeddings to find.
            Defaults to inf (unbounded).
        use_filter: Specifies whether to check feasibility of arguments for
            embedding independently of the embedder routine. Defaults to False.
        embedder: Specifies the embedding search method, a callable taking S, T as
            the first two arguments. Defaults to minorminer.subgraph.find_subgraph.
        embedder_kwargs: Dictionary specifying arguments for the `embedder`
            other than `S`, `T`.
        one_to_iterable: Specifies whether the embedder returns a dict with
            iterable values. Defaults to False to match find_subgraph.
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

    Raises:
        ValueError: If the target graph `T` is not of type zephyr, pegasus, or
            chimera.

    Returns:
        list: A list of disjoint embeddings.
    """
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

    tiling = tile == S
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

    for i, f in enumerate(sublattice_iter):
        Tr = _T.subgraph([f(n) for n in tile])
        if tiling:
            if Tr.number_of_edges() == S.number_of_edges():
                sub_embs = [{k: v for k, v in zip(S.nodes, Tr.nodes)}]
            else:
                sub_embs = []
        else:
            sub_embs = find_multiple_embeddings(
                S,
                Tr,
                max_num_emb=max_num_emb,
                use_filter=use_filter,
                seed=seed,
                embedder=embedder,
                embedder_kwargs=embedder_kwargs,
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


if __name__ == "__main__":
    print(" min m (graph rows) examples ")

    # Define the Graph Topologies, Tiles, and Generators
    visualize = True
    topologies = ["chimera", "pegasus", "zephyr"]
    smallest_tile = {"chimera": 1, "pegasus": 2, "zephyr": 1}
    generators = {
        "chimera": dnx.chimera_graph,
        "pegasus": dnx.pegasus_graph,
        "zephyr": dnx.zephyr_graph,
    }

    # Iterate over Topologies for Raster Embedding Checks
    for stopology in topologies:
        sublattice_size_S = smallest_tile[stopology] + 1
        S = generators[stopology](sublattice_size_S)

        # For each target topology, checks whether embedding the graph S into
        # that topology is feasible
        for ttopology in topologies:
            sublattice_size = lattice_size_lower_bound(
                S, topology=ttopology, one_to_one=True
            )
            if sublattice_size is None:
                print(
                    f"Embedding {stopology}-{sublattice_size_S} in "
                    f"{ttopology} is infeasible."
                )
            else:
                print(
                    f"Embedding {stopology}-{sublattice_size_S} in "
                    f"{ttopology} may be feasible, requires sublattice_size "
                    f">= {sublattice_size}."
                )
            T = generators[ttopology](smallest_tile[ttopology])
            sublattice_size = lattice_size_lower_bound(S, T=T)
            if sublattice_size is None:
                print(
                    f"Embedding {stopology}-{sublattice_size_S} in "
                    f"{ttopology}-{smallest_tile[ttopology]} is infeasible."
                )
            else:
                print(
                    f"Embedding {stopology}-{sublattice_size_S} in "
                    f"{ttopology}-{smallest_tile[ttopology]} may be feasible"
                    f", requires sublattice_size >= {sublattice_size}."
                )
    print()
    print(" raster embedding examples ")
    # A check at minimal scale:
    for topology in topologies:
        min_raster_scale = smallest_tile[topology]
        S = generators[topology](min_raster_scale)
        T = generators[topology](min_raster_scale + 1)  # Allows 4

        print()
        print(topology)
        # Perform Embedding Search and Validation
        embs = find_sublattice_embeddings(
            S, T, sublattice_size=min_raster_scale, max_num_emb=float("inf")
        )
        print(f"{len(embs)} Independent embeddings by rastering")
        print(embs)
        assert all(set(emb.keys()) == set(S.nodes()) for emb in embs)
        assert all(set(emb.values()).issubset(set(T.nodes())) for emb in embs)
        value_list = [v for emb in embs for v in emb.values()]
        assert len(set(value_list)) == len(value_list)
        if visualize:
            plt.figure(figsize=(12, 12))
            visualize_embeddings(T, embeddings=embs)
            Saux = nx.Graph()
            Saux.add_nodes_from(S)
            Saux.add_edges_from(list(S.edges)[:10])  # First 10 edges only ..
            plt.figure(figsize=(12, 12))
            visualize_embeddings(T, embeddings=embs, S=Saux)
            plt.show()
        embs = find_sublattice_embeddings(S, T)
        print(f"{len(embs)} Independent embeddings by direct search")
        assert all(set(emb.keys()) == set(S.nodes()) for emb in embs)
        assert all(set(emb.values()).issubset(set(T.nodes())) for emb in embs)
        value_list = [v for emb in embs for v in emb.values()]
        assert len(set(value_list)) == len(value_list)

        print("Defaults (full graph search): Presented as an ndarray")
        print(embeddings_to_array(embs, as_ndarray=True))

    # print("See additional usage examples in test_embeddings")

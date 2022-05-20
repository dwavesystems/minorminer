# Copyright 2018 D-Wave Systems Inc.
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
#
# ================================================================================================

try:
    import collections.abc as abc
except ImportError:
    import collections as abc

from dwave_networkx.generators.chimera import chimera_graph, chimera_coordinates
import networkx as nx

from minorminer.busclique import busgraph_cache

__all__ = ['find_clique_embedding',
           'find_biclique_embedding',
           'find_grid_embedding',
           ]


@nx.utils.decorators.nodes_or_number(0)
def find_clique_embedding(k, m=None, n=None, t=None, target_edges=None, target_graph=None):
    """Find an embedding for a clique in a Chimera graph.

    Given the node labels or size of a clique (fully connected graph) and size or
    edges of the target :term:`Chimera` graph, attempts to find an embedding.

    Args:
        k (int/iterable):
            Clique to embed. If k is an integer, generates an embedding for a
            clique of size k labelled [0,k-1]. If k is an iterable of nodes,
            generates an embedding for a clique of size len(k) labelled
            for the given nodes.

        m (int, optional, default=None):
            Number of rows in the Chimera lattice.

        n (int, optional, default=m):
            Number of columns in the Chimera lattice.

        t (int, optional, default 4):
            Size of the shore within each Chimera tile.

        target_edges (iterable[edge]):
            A list of edges in the target Chimera graph. Nodes are labelled as
            returned by :func:`~dwave_networkx.chimera_graph`.

        target_graph (networkx.Graph):
            A Chimera graph constructed by :func:`~dwave_networkx.chimera_graph`.

    Returns:
        dict: An embedding mapping a clique to the Chimera lattice.

    Note:
        Either target_edges or target_graph must be None.  If both are None,
        a graph with perfect yield is assumed from the parameters m, n, t.  If
        target_edges is not None, at least m must not be None.

    Examples:
        The first example finds an embedding for a :math:`K_4` complete graph in a single
        Chimera unit cell. The second for an alphanumerically labeled :math:`K_3`
        graph in 4 unit cells.

        >>> from dwave.embedding.chimera import find_clique_embedding
        ...
        >>> embedding = find_clique_embedding(4, 1, 1)
        >>> embedding  # doctest: +SKIP
        {0: [4, 0], 1: [5, 1], 2: [6, 2], 3: [7, 3]}

        >>> from dwave.embedding.chimera import find_clique_embedding
        ...
        >>> embedding = find_clique_embedding(['a', 'b', 'c'], m=2, n=2, t=4)
        >>> embedding  # doctest: +SKIP
        {'a': [20, 16], 'b': [21, 17], 'c': [22, 18]}

    """
    _, nodes = k
    m, n, t, g = _get_target_graph(m, n, t, target_edges, target_graph)
    embedding = busgraph_cache(g).find_clique_embedding(nodes)

    if len(embedding) != len(nodes):
        raise ValueError("No clique embedding found")

    return embedding


@nx.utils.decorators.nodes_or_number(0)
@nx.utils.decorators.nodes_or_number(1)
def find_biclique_embedding(a, b, m=None, n=None, t=None, target_edges=None, target_graph=None):
    """Find an embedding for a biclique in a Chimera graph.

    Given a biclique (a bipartite graph where every vertex in a set in connected
    to all vertices in the other set) and a target :term:`Chimera` graph size or
    edges, attempts to find an embedding.

    Args:
        a (int/iterable):
            Left shore of the biclique to embed. If a is an integer, generates
            an embedding for a biclique with the left shore of size a labelled
            [0,a-1]. If a is an iterable of nodes, generates an embedding for a
            biclique with the left shore of size len(a) labelled for the given
            nodes.

        b (int/iterable):
            Right shore of the biclique to embed.If b is an integer, generates
            an embedding for a biclique with the right shore of size b labelled
            [0,b-1]. If b is an iterable of nodes, generates an embedding for a
            biclique with the right shore of size len(b) labelled for the given
            nodes.

        m (int, optional, default=None):
            Number of rows in the Chimera lattice.

        n (int, optional, default=m):
            Number of columns in the Chimera lattice.

        t (int, optional, default 4):
            Size of the shore within each Chimera tile.

        target_edges (iterable[edge]):
            A list of edges in the target Chimera graph. Nodes are labelled as
            returned by :func:`~dwave_networkx.chimera_graph`.

        target_graph (networkx.Graph):
            A Chimera graph constructed by :func:`~dwave_networkx.chimera_graph`.

    Returns:
        tuple: A 2-tuple containing:

            dict: An embedding mapping the left shore of the biclique to
            the Chimera lattice.

            dict: An embedding mapping the right shore of the biclique to
            the Chimera lattice.

    Note:
        Either target_edges or target_graph must be None.  If both are None,
        a graph with perfect yield is assumed from the parameters m, n, t.  If
        target_edges is not None, at least m must not be None.

    Examples:
        This example finds an embedding for an alphanumerically labeled biclique in a single
        Chimera unit cell.

        >>> from dwave.embedding.chimera import find_biclique_embedding
        ...
        >>> left, right = find_biclique_embedding(['a', 'b', 'c'], ['d', 'e'], 1, 1)
        >>> print(left, right)  # doctest: +SKIP
        {'a': [4], 'b': [5], 'c': [6]} {'d': [0], 'e': [1]}

    """
    _a, anodes = a
    _b, bnodes = b

    if isinstance(_a, int) and isinstance(_b, int):
        bnodes = [len(anodes) + x for x in bnodes]

    if set(anodes).intersection(set(bnodes)):
        raise ValueError("a and b overlap")

    m, n, t, g = _get_target_graph(m, n, t, target_edges, target_graph)
    embedding = busgraph_cache(g).find_biclique_embedding(len(anodes), len(bnodes))

    if not embedding:
        raise ValueError("No biclique embedding found")

    return ({x: embedding[anodes.index(x)] for x in anodes},
            {y: embedding[bnodes.index(y) + len(anodes)] for y in bnodes})


def find_grid_embedding(dim, m, n=None, t=4):
    """Find an embedding for a grid in a Chimera graph.

    Given grid dimensions and a target :term:`Chimera` graph size,
    attempts to find an embedding.

    Args:
        dim (iterable[int]):
            Sizes of each grid dimension. Length can be between 1 and 3.

        m (int):
            Number of rows in the Chimera lattice.

        n (int, optional, default=m):
            Number of columns in the Chimera lattice.

        t (int, optional, default 4):
            Size of the shore within each Chimera tile.

    Returns:
        dict: An embedding mapping a grid to the Chimera lattice.

    Examples:
        This example finds an embedding for a 2x3 grid in a 12x12 lattice of Chimera unit cells.

        >>> from dwave.embedding.chimera import find_grid_embedding
        ...
        >>> embedding = find_grid_embedding([2, 3], m=12, n=12, t=4)
        >>> embedding  # doctest: +SKIP
        {(0, 0): [0, 4],
         (0, 1): [8, 12],
         (0, 2): [16, 20],
         (1, 0): [96, 100],
         (1, 1): [104, 108],
         (1, 2): [112, 116]}

    """

    m, n, t, _ = _get_target_graph(m, n, t, None, _dont_construct)
    indexer = chimera_coordinates(m, n, t)

    dim = list(dim)
    num_dim = len(dim)
    if num_dim == 1:
        def _key(row, col, aisle): return row
        dim.extend([1, 1])
    elif num_dim == 2:
        def _key(row, col, aisle): return row, col
        dim.append(1)
    elif num_dim == 3:
        def _key(row, col, aisle): return row, col, aisle
    else:
        raise ValueError("find_grid_embedding supports between one and three dimensions")

    rows, cols, aisles = dim
    if rows > m or cols > n or aisles > t:
        msg = ("the largest grid that find_grid_embedding can fit in a ({}, {}, {}) Chimera-lattice "
               "is {}x{}x{}; given grid is {}x{}x{}").format(m, n, t, m, n, t, rows, cols, aisles)
        raise ValueError(msg)

    return {_key(row, col, aisle): [indexer.chimera_to_linear((row, col, 0, aisle)),
                                     indexer.chimera_to_linear((row, col, 1, aisle))]
            for row in range(dim[0]) for col in range(dim[1]) for aisle in range(dim[2])}

_dont_construct = object()
def _get_target_graph(m, n=None, t=None, target_edges=None, target_graph=None):
    if target_graph is not None and target_graph is not _dont_construct:
        if target_edges is not None:
            raise ValueError("either target_graph or target_edges must be None")
        m = target_graph.graph['rows']
        n = target_graph.graph['columns']
        t = target_graph.graph['tile']
    else:
        if not isinstance(m, int):
            raise TypeError('Chimera lattice parameter m must be an int and >= 1')
        if m <= 0:
            raise ValueError('Chimera lattice parameter m must be an int and >= 1')

        if n is None:
            n = m
        else:
            if not isinstance(n, int):
                raise TypeError('Chimera lattice parameter n must be an int and >= 1')
            if n <= 0:
                raise ValueError('Chimera lattice parameter n must be an int and >= 1')
        if t is None:
            t = 4
        else:
            if not isinstance(t, int):
                raise TypeError('Chimera lattice parameter t must be an int and >= 1')
            if t <= 0:
                raise ValueError('Chimera lattice parameter t must be an int and >= 1')

        if target_graph is not _dont_construct:
            target_graph = chimera_graph(m, n, t, edge_list = target_edges)

    return m, n, t, target_graph

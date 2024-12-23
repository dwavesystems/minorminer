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
r"""Determining embedding feasibility is NP-complete.
Efficient methods and heuristics are collected together in this module,
that provide necessary conditions and bounds on feasibility.
This includes determination of a lower bound on the (square)
lattice size required for chimera, pegasus and zephyr dwave_networkx graph
types, and a feasibility filter for general graphs based on the degree
distributions of target and source graphs.
"""
import dwave_networkx as dnx
import networkx as nx
import numpy as np
from collections import Counter
from typing import Optional


def embedding_feasibility_filter(
    S: nx.Graph, T: nx.Graph, one_to_one: bool = False
) -> bool:
    """Feasibility filter for embedding.

    Determines if the source graph ``S`` can be subgraph-embedded onto the target
    graph ``T`` based on their degree distributions. Returns ``False`` if embedding
    is definitely infeasible; otherwise, returns ``True``. False positives are
    permissible because deciding the graph isomorphism problem is NP-complete,
    and this filter is designed to be efficient.

    The degree distribution test is a heuristic; stronger filters are possible
    by exploiting additional graph structure.

    Args:
        S: The source graph to embed.
        T: The target graph in which to embed.
        one_to_one: If True, only 1-to-1 (subgraph) embeddings are allowed.
            Defaults to False, permitting minor embeddings.

    Returns:
        bool: ``False`` if subgraph embedding is definitely infeasible, ``True``
        otherwise.
    """
    if T == S:
        return True
    if (
        T.number_of_nodes() < S.number_of_nodes()
        or T.number_of_edges() < S.number_of_edges()
    ):
        return False
    if S.number_of_nodes() == 0:
        return True
    elif T.number_of_nodes() == 0:
        return False

    S_degree = np.sort([S.degree[n] for n in S.nodes()])
    T_degree = np.sort([T.degree[n] for n in T.nodes()])

    if np.any(T_degree[-len(S_degree) :] < S_degree):
        if one_to_one or T_degree[-1] <= 2:
            # Too many high degree nodes in S
            return False
        # Attempt minor embed (enhance T degrees)
        # Minor embedding feasibility reduces to bin packing when
        # considering a best target case knowing only the degree
        # distribution. In general feasibility is NP-complete, a cheap
        # marginal degree distribution filter is used.

        # We can eliminate nodes of equal degree assuming best case:
        ResidualCounts = Counter(T_degree)
        ResidualCounts.subtract(Counter(S_degree))

        # Target nodes of degree x <=2 are only of use for minor
        # embedding source nodes of degree <=x:
        for kS in range(3):
            if ResidualCounts[kS] < 0:
                ResidualCounts[kS + 1] += ResidualCounts[kS]
            ResidualCounts[kS] = 0

        if all(v > 0 for v in ResidualCounts.values()):
            return True

        nT_auxiliary = sum(ResidualCounts.values())
        if nT_auxiliary < 0:  # extra available to form chains
            return False

        # In best case all target nodes have degree kTmax, and chains
        # are trees. To cover degree k in S requires n auxiliary target
        # nodes such that kTmax + n(kTmax-2) >= k
        kTmax = np.max([k for k, v in ResidualCounts.items() if v > 0])
        min_auxiliary_necessary = sum(
            [
                -v * np.ceil((k - kTmax) / (kTmax - 2))
                for k, v in ResidualCounts.items()
                if v < 0
            ]
        )
        return min_auxiliary_necessary <= nT_auxiliary

    else:
        return True


def lattice_size(T: nx.Graph = None) -> int:
    """Determines the cellular (square) dimension of a lattice

    The lattice size is the parameter ``m`` of a dwave_networkx graph, also
    called number of rows, or in the case of a chimera graph max(m,n). This
    upper bounds the ``sublattice_size`` for ``find_sublattice_embeddings``.

    Args:
        T: The target graph in which to embed. The graph must be of type
            zephyr, pegasus or chimera and constructed by dwave_networkx.
    Returns:
        int: The maximum possible size of a tile
    """
    # Possible feature enhancement, determine a stronger upper bound akin
    # to lattice_size_lower_bound, accounting for defects, the
    # degree distribution and other simple properties.

    return max(T.graph.get("rows"), T.graph.get("columns"))


def lattice_size_lower_bound(
    S: nx.Graph,
    T: nx.Graph = None,
    topology: str = None,
    t: int = None,
    one_to_one: bool = False,
) -> Optional[int]:
    """Returns a lower bound on the size necessary for embedding.

    The lattice size is the parameter ``m`` of a dwave_networkx graph, also
    called number of rows. The function returns a lower bound (necessary but
    not sufficient for embedding) using efficiently established graph
    properties such as the number of nodes, number of edges, node-degree
    distribution, and two-colorability

    Args:
        S: The source graph to embed.
        T: The target graph in which to embed. The graph must be of type
            'zephyr', 'pegasus', or 'chimera' and constructed by dwave_networkx.
        topology: The topology ('chimera', 'pegasus', or 'zephyr'). This is
            inferred from `T` by default. Any set value must be consistent with
            ``T`` (if ``T`` is not None).
        t: The tile parameter, relevant for 'zephyr' and 'chimera' topologies.
            Inferred from ``T`` by default. Any set value must be consistent with
            ``T`` (if ``T`` is not None).
        one_to_one: True if a subgraph embedding is assumed, False for general
            embeddings.

    Raises:
        ValueError: If ``T`` and ``topology`` are inconsistent, or if ``T`` and ``t`` are inconsistent.
        ValueError: If neither ``T`` nor ``topology`` is specified.
        ValueError: If ``S`` cannot be embedded in ``T`` or the specified topology.

    Returns:
        float: Minimum ``lattice_size`` for embedding to be feasible. Returns
        None if embedding for any number of graph rows is infeasible.
    """
    # Comment, similar bounds are possible allowing for minor embeddings,
    # the degree distribution is bounded as chain length increases, whilst
    # the number of nodes is depleted.
    # This could be a possible feature expansion.
    if T is not None:
        if embedding_feasibility_filter(S, T, one_to_one) is False:
            return None

        if topology is None:
            topology = T.graph.get("family")
        elif topology != T.graph.get("family"):
            raise ValueError("Arguments T and topology are inconsistent")

        if t is None:
            t = T.graph["tile"]
    else:
        if topology is None:
            raise ValueError("T or topology must be specified")

        if t is None:
            t = 4
        max_degrees = {"chimera": 2 + 2 * t, "pegasus": 15, "zephyr": 4 + 4 * t}
        max_source_degree = max(d for _, d in S.degree)

        if max_source_degree > max_degrees[topology]:
            return None

    N = S.number_of_nodes()
    if topology == "chimera":
        # Two colorability is necessary, and cheap to check
        if any(
            c > 2
            for c in nx.algorithms.coloring.greedy_coloring.greedy_color(
                S, strategy="connected_sequential_bfs"
            ).values()
        ):
            return None

        def generator(lattice_size):
            return dnx.chimera_graph(m=lattice_size, n=lattice_size, t=t)

        # A lower bound based on number of variables N = m*n*2*t
        lattice_size = np.ceil(np.sqrt(N / 4 / t))

    elif topology == "pegasus":

        def generator(lattice_size):
            return dnx.pegasus_graph(m=lattice_size)

        # A lower bound based on number of variables N = (m*24-8)*(m-1)
        lattice_size = np.ceil(1 / 12 * (8 + np.sqrt(6 * N + 16)))

    elif topology == "zephyr":

        def generator(lattice_size):
            return dnx.zephyr_graph(m=lattice_size, t=t)

        # A lower bound based on number of variables N = (2m+1)*m*4*t
        lattice_size = np.ceil((np.sqrt(2 * N / t + 1) - 1) / 4)

    else:
        raise ValueError(
            "source graphs must be a graph constructed by "
            "dwave_networkx as chimera, pegasus or zephyr type"
        )

    # Evaluate tile feasibility (defect free subgraphs)
    lattice_size = round(lattice_size)
    tile = generator(lattice_size=lattice_size)
    while embedding_feasibility_filter(S, tile, one_to_one) is False:
        lattice_size += 1
        tile = generator(lattice_size=lattice_size)
    return lattice_size

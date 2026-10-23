# Copyright 2026 D-Wave
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

"""Zephyr graphs with faults, for the clique-embedder tests."""

from __future__ import annotations

import random
from collections.abc import Iterable

import networkx as nx
from dwave.graphs import zephyr_coordinates, zephyr_graph

Node = tuple[int, int, int]  # cartesian qubit (x, y, k)
Edge = tuple[Node, Node]  # cartesian coupler, endpoints in either order


def faulty_zephyr(
    m: int,
    t: int,
    *,
    node_fault_rate: float = 0.0,
    edge_fault_rate: float = 0.0,
    seed: int = 0,
    clustered: bool = False,
    drop_nodes: Iterable[Node] = (),
    drop_edges: Iterable[Edge] = (),
    labels: str = "cartesian",
) -> nx.Graph:
    """A Zephyr graph with qubits and couplers removed.

    Faults are chosen in Cartesian coordinates ``(x, y, k)`` and then mapped into the requested
    label mode, so the same arguments give the same faults, and the same grid, in every label
    mode. With no fault arguments the graph is perfect.

    Args:
        m: Zephyr grid size.
        t: Zephyr tile size.
        node_fault_rate: Fraction of qubits to remove at random.
        edge_fault_rate: Fraction of the remaining couplers to remove at random.
        seed: Seed for the random faults.
        clustered: If ``True``, also remove one random block (its qubits for every ``k``) and
            cut one random external path at one block (its external couplers for every ``k``).
        drop_nodes: Qubits to remove, in Cartesian coordinates.
        drop_edges: Couplers to remove, in Cartesian coordinates, endpoints in either order.
        labels: Node labels of the returned graph: ``"cartesian"`` ``(x, y, k)``,
            ``"coordinates"`` (Zephyr ``(u, w, k, j, z)``) or ``"int"`` (linear indices).

    Returns:
        The graph, with ``dwave.graphs`` Zephyr metadata.
    """
    to_cartesian = zephyr_coordinates.zephyr_to_cartesian
    full = zephyr_graph(m, t, coordinates=True, data=True)  # authoritative topology
    nodes = sorted(to_cartesian(z) for z in full.nodes())
    edges = sorted(_canonical(to_cartesian(a), to_cartesian(b)) for a, b in full.edges())
    dropped_nodes = set(drop_nodes)
    dropped_edges = {_canonical(a, b) for a, b in drop_edges}

    rng = random.Random(seed)
    if clustered:
        x, y, _ = rng.choice(nodes)
        dropped_nodes.update((x, y, k) for k in range(t))
        x, y, _ = rng.choice([node for node in nodes if node not in dropped_nodes])
        dx, dy = (0, 4) if x % 2 == 0 else (4, 0)
        dropped_edges.update(_canonical((x, y, k), (x + dx, y + dy, k)) for k in range(t))
    remaining_nodes = [node for node in nodes if node not in dropped_nodes]
    dropped_nodes.update(rng.sample(remaining_nodes, int(node_fault_rate * len(remaining_nodes))))
    remaining_edges = [
        edge
        for edge in edges
        if edge not in dropped_edges and dropped_nodes.isdisjoint(edge)
    ]
    dropped_edges.update(rng.sample(remaining_edges, int(edge_fault_rate * len(remaining_edges))))

    present = [z for z in full.nodes() if to_cartesian(z) not in dropped_nodes]
    present_set = set(present)
    kept_edges = [
        (a, b)
        for a, b in full.edges()
        if a in present_set
        and b in present_set
        and _canonical(to_cartesian(a), to_cartesian(b)) not in dropped_edges
    ]

    if labels == "int":
        linear = dict(full.nodes(data="linear_index"))
        return zephyr_graph(
            m,
            t,
            node_list=[linear[z] for z in present],
            edge_list=[(linear[a], linear[b]) for a, b in kept_edges],
        )
    graph = zephyr_graph(m, t, node_list=present, edge_list=kept_edges, coordinates=True)
    if labels == "coordinates":
        return graph
    if labels == "cartesian":
        # dwave.graphs has no cartesian labelling; Grid infers it from the 3-tuple nodes
        return nx.relabel_nodes(graph, {z: to_cartesian(z) for z in graph.nodes()})
    raise ValueError(f"unknown label mode {labels!r}")


def _canonical(a: Node, b: Node) -> Edge:
    """The coupler ``{a, b}`` with its endpoints in sorted order."""
    return (a, b) if a < b else (b, a)

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

from typing import Callable, Hashable, Literal

from minorminer.subgraph import find_subgraph
import networkx as nx
from dwave.graphs import (
    chimera_coordinates,
    chimera_two_color,
    pegasus_coordinates,
    pegasus_four_color,
    zephyr_coordinates,
    zephyr_four_color,
)


__all__ = [
    "find_labeled_subgraph",
    "get_graph_shape",
    "graph_label_mapping",
    "node_labels_by_orientation",
    "node_labels_by_coloring",
    "node_labels_by_quotient",
]


def get_graph_shape(graph: nx.Graph) -> None | tuple:
    """Return the shape of a graph based on its family and metadata.

    Args:
        graph: A NetworkX graph with family metadata.
    Returns:
        A tuple representing the shape of the graph, which varies based on the graph family. If the family metadata is not found then ``None`` is returned.
    """
    graph_family = graph.graph.get("family", None)
    if graph_family == "chimera":
        return (graph.graph["rows"], graph.graph["columns"], graph.graph["tile"])
    elif graph_family == "pegasus":
        return (graph.graph["rows"],)
    elif graph_family == "zephyr":
        return (graph.graph["rows"], graph.graph["tile"])
    else:
        return None


def graph_label_mapping(
    graph_family: str, graph_shape: tuple, label_one: str, label_two: str
) -> Callable:
    """Return a function that maps a node label of int/coordinate/nice type to another.

    Args:
        graph_family: The family of the graph (e.g., 'chimera', 'pegasus', 'zephyr').
        graph_shape: The shape of the graph (e.g., (m, t) for Zephyr).
        label_one: The type of the input label (e.g., 'coordinate', 'int').
        label_two: The type of the output label (e.g., 'coordinate', 'int').

    Returns:
        A function that takes a node label of type `label_one` and returns
        the corresponding node label of type `label_two`.
    """
    if label_one != label_two:
        match graph_family:
            case "chimera":
                coordinates = chimera_coordinates(*graph_shape)
                if label_one == "coordinate" and label_two == "int":
                    return coordinates.chimera_to_linear
                elif label_one == "int" and label_two == "coordinate":
                    return coordinates.linear_to_chimera
            case "pegasus":
                coordinates = pegasus_coordinates(*graph_shape)
                match (label_one, label_two):
                    case ("coordinate", "int"):
                        return coordinates.pegasus_to_linear
                    case ("coordinate", "nice"):
                        return coordinates.pegasus_to_nice
                    case ("nice", "coordinate"):
                        return coordinates.nice_to_pegasus
                    case ("nice", "int"):
                        return coordinates.nice_to_linear
                    case ("int", "coordinate"):
                        return coordinates.linear_to_pegasus
                    case ("int", "nice"):
                        return coordinates.linear_to_nice
            case "zephyr":
                coordinates = zephyr_coordinates(*graph_shape)
                if label_one == "coordinate" and label_two == "int":
                    return coordinates.zephyr_to_linear
                elif label_one == "int" and label_two == "coordinate":
                    return coordinates.linear_to_zephyr
            case _:
                raise ValueError(f"Unsupported graph family: {graph_family}")
    else:
        return lambda n: n

    raise ValueError(
        f"Unsupported label conversion: {label_one} to {label_two} for {graph_family}"
    )


def node_labels_by_orientation(
    graph: nx.Graph,
    as_str: bool = True,
    graph_labels: str | None = None,
    graph_family: str | None = None,
    graph_shape: tuple | None = None,
) -> dict[Hashable, str] | dict[Hashable, int]:
    """Generate node labels from graph orientation classes.

    For supported D-Wave graph families, this function labels nodes by their
    physical qubit orientation in processor implementations (vertical or horizontal).

    For non-D-Wave graph families, a greedy coloring is used as a fallback. In
    that case, the graph must be bipartite so that the resulting coloring has
    exactly two color classes, which serve as orientation labels.

    Args:
        graph: Input graph whose nodes will be labeled by orientation.
        as_str: If ``True``, convert orientation labels to strings before returning. If
            ``False``, preserve the integer labels. Defaults to ``True``.
        graph_labels: The node label type ("int", "coordinate" or "nice"). If ``None``, the label type is inferred from the graph's metadata.
        graph_family: The graph family ("chimera", "pegasus", or "zephyr"). If ``None``, the family is inferred from the graph's metadata.
        graph_shape: The shape of the graph. If ``None``, the shape is inferred from the graph's metadata.

    Returns:
        A dictionary mapping graph nodes to orientation labels.

    Raises:
        ValueError: If greedy coloring is used and produces
            more than two colors.
    """
    if graph_family is None:
        graph_family = graph.graph.get("family", None)
    if graph_labels is None:
        graph_labels = graph.graph.get("labels", None)
    if graph_shape is None:
        graph_shape = get_graph_shape(graph)
    match graph_family:
        case "chimera" | "pegasus" | "zephyr":
            to_coord = graph_label_mapping(
                graph_family, graph_shape, graph_labels, "coordinate"
            )
            axis = 2 if graph_family == "chimera" else 0
            col = {n: to_coord(n)[axis] for n in graph.nodes()}
        case _:
            col = nx.greedy_color(graph)
            if len(set(col.values())) != 2:
                raise ValueError(
                    "Orientation labeling requires a bipartite graph, but greedy "
                    "coloring produced more than 2 colors"
                )
    if as_str:
        return {k: str(v) for k, v in col.items()}
    else:
        return col


def node_labels_by_coloring(
    graph: nx.Graph,
    as_str: bool = True,
    graph_labels: str | None = None,
    graph_family: str | None = None,
    graph_shape: tuple | None = None,
) -> dict[Hashable, str] | dict[Hashable, int]:
    """Generate node labels from a family-specific graph coloring.

    For supported D-Wave graph families, canonical 2-coloring for Chimera and 4-coloring
    for Pegasus and Zephyr are used. There can be more than 1 valid
    coloring not related by isomorphism, so failure to find a coloring is
    not sufficient to rule out any 2 (or 4) colored subgraph isomorphism.

    For graphs without recognized D-Wave family metadata, a greedy coloring is used as a
    generic fallback.

    Args:
        graph: Input graph to color. The family graph metadata is used to select
            a family-specific coloring method where available.
        as_str: If ``True``, convert color labels to strings before returning. If
            ``False``, preserve the integer color labels. Defaults to ``True``.
        graph_family: The graph family ("chimera", "pegasus", or "zephyr"). If ``None``, the family is inferred from the graph's metadata.
        graph_labels: The node label type ("int", "coordinate" or "nice"). If ``None``, the label type is inferred from the graph's metadata.
        graph_shape: The shape of the graph. If ``None``, the shape is inferred from the graph's metadata.
    Raises:
        ValueError: If the graph family is not supported.
    Returns:
        A dictionary mapping graph nodes to color labels.
    """
    if graph_family is None:
        graph_family = graph.graph.get("family", None)
    if graph_labels is None:
        graph_labels = graph.graph.get("labels", None)
    if graph_shape is None:
        graph_shape = get_graph_shape(graph)
    if graph_family in (
        "chimera",
        "pegasus",
        "zephyr",
    ):
        to_coord = graph_label_mapping(
            graph_family, graph_shape, graph_labels, "coordinate"
        )
        match graph_family:
            case "chimera":
                col = {n: chimera_two_color(to_coord(n)) for n in graph.nodes()}
            case "pegasus":
                col = {n: pegasus_four_color(to_coord(n)) for n in graph.nodes()}
            case "zephyr":
                col = {n: zephyr_four_color(to_coord(n)) for n in graph.nodes()}
    else:
        col = nx.greedy_color(graph)

    if as_str:
        return {k: str(v) for k, v in col.items()}

    return col


def node_labels_by_quotient(
    graph: nx.Graph,
    expand_boundary_search: bool = True,
    as_str: bool = True,
    graph_labels: str | None = None,
    graph_family: str | None = None,
    graph_shape: tuple | None = None,
) -> dict[Hashable, str] | dict[Hashable, tuple]:
    """Generate quotient graph labels for nodes based on graph family and structure.

    This function assigns quotient labels to nodes, grouping qubits by
    orientation and displacement within their quotient block.

    For Zephyr graphs with ``expand_boundary_search=True``, boundary quotient
    nodes are collapsed with adjacent internal nodes, since
    those assignments also allow for embeddings.

    Args:
        graph: A Chimera, Pegasus or Zephyr NetworkX graph.
        expand_boundary_search: If ``True`` and the graph family is ``"zephyr"``,
            boundary quotient nodes are remapped to adjacent (interior perpendicular
            block offset) nodes. Defaults to ``True``.
        as_str: If ``True``, labels are converted to strings. If ``False``, labels
            remain as tuples. Defaults to ``True``.
        graph_labels: The node label type ("int", "coordinate" or "nice"). If ``None``, the label type is inferred from the graph's metadata.
        graph_family: The graph family ("chimera", "pegasus", or "zephyr"). If ``None``, the family is inferred from the graph's metadata.
        graph_shape: The shape of the graph. If ``None``, the shape is inferred from the graph's metadata.
    Returns:
        A dictionary mapping node coordinates to quotient labels. The type of the labels
        depends on the ``as_str`` parameter: if ``True``, labels are strings; if ``False``,
        labels are tuples.

    Raises:
        ValueError: If graph family is not found in metadata or is not 'zephyr',
            'pegasus', or 'chimera'.
    """
    if graph_family is None:
        graph_family = graph.graph.get("family", None)
    if graph_family in (
        "chimera",
        "pegasus",
        "zephyr",
    ):
        if graph_labels is None:
            graph_labels = graph.graph.get("labels", None)
        if graph_shape is None:
            graph_shape = get_graph_shape(graph)
        if graph_labels is None:
            graph_labels = "coordinate"
        if graph_labels != "coordinate":
            graph = nx.relabel_nodes(
                graph,
                graph_label_mapping(
                    graph_family, graph_shape, graph_labels, "coordinate"
                ),
            )
        match graph_family:
            case "chimera":
                col = {n: n[:3] for n in graph.nodes()}
            case "pegasus":
                col = {n: n[:2] + (n[2] // 2,) + n[3:] for n in graph.nodes()}
            case "zephyr":
                if expand_boundary_search:
                    m = graph.graph["rows"]
                    wmap = {0: 1, 2 * m: 2 * m - 1}
                    col = {
                        n: n[:1] + (wmap.get(n[1], n[1]),) + n[3:]
                        for n in graph.nodes()
                    }
                else:
                    col = {n: n[:2] + n[3:] for n in graph.nodes()}
    else:
        raise ValueError("Unrecognized graph family")
    if graph_labels != "coordinate":
        to_label = graph_label_mapping(
            graph_family, graph_shape, "coordinate", graph_labels
        )
        col = {to_label(k): v for k, v in col.items()}
    if as_str:
        # Whitespace-free: find_subgraph's underlying vertex-label parser rejects labels
        # containing spaces, which the default tuple repr would otherwise include.
        return {k: str(v).replace(" ", "") for k, v in col.items()}
    else:
        return col


def find_labeled_subgraph(
    source: nx.Graph,
    target: nx.Graph,
    labeling_method: Literal["orientation", "quotient", "coloring"] = "orientation",
    node_labels: tuple[dict, dict] | None = None,
    **kwargs,
) -> dict[Hashable, Hashable]:
    """Find a subgraph of target isomorphic to source that preserves node colors.

    This is a helper function that calls :code:``find_subgraph`` with ``node_labels``.
    Node labeling can significantly accelerate the search when well chosen. The
    supported labeling methods are intended for bipartite graphs and D-Wave source
    graphs (Zephyr, Pegasus, and Chimera) embedded onto D-Wave target graphs. However,
    in general it may be necessary to consider uncolored search or several
    application-specific node labelings to find embeddings.

    Args:
        source: Source graph.
        target: Target graph.
        labeling_method: Method to use for coloring the graphs. Options are:
            - 'orientation': It is assumed variables in source must
                map to variables of the same orientation in target.
                If the graph orientation is unclear (the graph is not
                Chimera, Pegasus or Zephyr) the orientation is assigned by
                a greedy min-coloring and an error is thrown if there are
                not precisely two colors.
            - 'quotient': Quotient labeling is used. The graphs must
                be of type Chimera, Pegasus or Zephyr. Qubits are labeled
                by orientation and horizontal displacement. Note that this is
                only a good choice if the grid parameter (number of rows
                and columns) of source and target are matched.
            - 'coloring': According to canonical 2-colorings of Chimera,
                or the 4-coloring of Pegasus/Zephyr. Note that these colorings
                are not unique. If the graph orientation
                is unclear, the coloring is assigned by a greedy min-coloring.
                A greedy coloring is not guaranteed to be unique or optimal.
                For high performance applications consider direct
                specification of the coloring via the node_labels argument.
        node_labels: A tuple of dicts mapping nodes in source and target graphs
            to labels.
        **kwargs: Additional keyword arguments to pass to find_subgraph.
            Use of a timeout > 0 is recommended.
    """
    if node_labels is None:
        # use provided coloring
        match labeling_method:
            case "orientation":
                node_labels = tuple(
                    node_labels_by_orientation(G) for G in (source, target)
                )
            case "quotient":
                if (
                    "family" not in target.graph
                    or target.graph["family"] != source.graph["family"]
                ):
                    raise ValueError(
                        "Source and target graph families should match for quotient "
                        "coloring"
                    )
                if target.graph["family"] not in ("chimera", "pegasus", "zephyr"):
                    raise ValueError(
                        "Quotient coloring is only implemented for Chimera, Pegasus "
                        "and Zephyr graph families"
                    )
                node_labels = tuple(
                    node_labels_by_quotient(G) for G in (source, target)
                )
            case "coloring":
                node_labels = tuple(
                    node_labels_by_coloring(G) for G in (source, target)
                )
            case _:
                raise ValueError(f"Unknown coloring method {labeling_method}")

    return find_subgraph(source, target, node_labels=node_labels, **kwargs)

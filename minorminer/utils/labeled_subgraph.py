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

"""Subgraph search with node labelings that constrain the search.

The labeling helpers in this module rely on graph metadata (the ``family``,
``labels`` and shape fields) to produce canonical labelings for the D-Wave
topologies (Chimera, Pegasus and Zephyr). Such graphs should be generated with
the :mod:`dwave.graphs` topology generators (``dwave.graphs.chimera_graph``,
``dwave.graphs.pegasus_graph`` and ``dwave.graphs.zephyr_graph``), which
populate this metadata. Graphs built by other means, or with that metadata
removed or altered, may not carry sufficient information for a family-specific
labeling; in that case the generic (bipartite coloring) fallback is used where
available.
"""

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
    graph_family: Literal["chimera", "pegasus", "zephyr"],
    graph_shape: tuple,
    label_one: Literal["int", "coordinate", "nice"],
    label_two: Literal["int", "coordinate", "nice"],
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


def _find_subgraph_label(value: Hashable) -> str:
    """Cast a label to a ``find_subgraph``-safe string.

    ``find_subgraph`` requires string vertex labels and its parser rejects labels
    containing whitespace, so the value is stringified and spaces are stripped.
    """
    return str(value).replace(" ", "")


def node_labels_by_orientation(
    graph: nx.Graph,
) -> dict[Hashable, str]:
    """Generate node labels from graph orientation classes.

    For supported D-Wave graph families, this function labels nodes by their
    physical qubit orientation in processor implementations (vertical or horizontal).

    For non-D-Wave graph families, a bipartite coloring is used as a fallback. In
    that case, the graph must be bipartite so that the resulting coloring has
    exactly two color classes, which serve as orientation labels.

    Note:
        Only one of the two possible bipartite colorings is considered; the two
        color classes are numbered independently and are not aligned between a
        source and a target graph. This labeling is therefore in general only
        sufficient for symmetric bipartite graphs. For asymmetric graphs the
        opposite class numbering may be required, so a failure to find an
        embedding under this labeling does not rule one out. The same limitation
        applies to any explicitly provided two coloring.

    Args:
        graph: Input graph whose nodes will be labeled by orientation. The graph
            family, node label type and shape are inferred from the graph's metadata
            (as populated by the :mod:`dwave.graphs` generators). If the graph family
            is not inferred from the metadata, the function will fall back to a
            bipartite coloring approach.

    Returns:
        A dictionary mapping graph nodes to string orientation labels.

    Raises:
        ValueError: If the bipartite coloring fallback is used and the graph is
            not bipartite.
    """
    graph_family = graph.graph.get("family", None)
    graph_labels = graph.graph.get("labels", None)
    graph_shape = get_graph_shape(graph)
    match graph_family:
        case "chimera" | "pegasus" | "zephyr":
            if graph_labels is None:
                graph_labels = "coordinate"
            to_coord = graph_label_mapping(
                graph_family, graph_shape, graph_labels, "coordinate"
            )
            axis = 2 if graph_family == "chimera" else 0
            col = {n: to_coord(n)[axis] for n in graph.nodes()}
        case _:
            # Only one of the two bipartite colorings is considered (see Note);
            # this is in general only sufficient for symmetric bipartite graphs.
            try:
                col = nx.bipartite.color(graph)
            except nx.NetworkXError:
                raise ValueError(
                    "Orientation labeling requires a bipartite graph"
                )
    return {n: _find_subgraph_label(c) for n, c in col.items()}


def node_labels_by_coloring(
    graph: nx.Graph,
) -> dict[Hashable, str]:
    """Generate node labels from a family-specific graph coloring.

    For supported D-Wave graph families, canonical 2-coloring for Chimera and 4-coloring
    for Pegasus and Zephyr are used. There can be more than 1 valid
    coloring not related by isomorphism, so failure to find a coloring is
    not sufficient to rule out any 2 (or 4) colored subgraph isomorphism.

    For graphs without recognized D-Wave family metadata, a bipartite coloring is used
    as a generic fallback.

    Note:
        For the bipartite fallback, only one of the two possible colorings is
        considered; the two color classes are numbered independently and are not
        aligned between a source and a target graph. This labeling is therefore
        in general only sufficient for symmetric bipartite graphs, and a failure
        to find an embedding under it does not rule one out. Any family string
        that is not ``"chimera"``, ``"pegasus"`` or ``"zephyr"`` (including a
        misspelling) silently uses this fallback rather than raising.

    Args:
        graph: Input graph to color. The family, node label type and shape are
            inferred from the graph's metadata (as populated by the
            :mod:`dwave.graphs` generators) and used to select a family-specific
            coloring method where available. The function will fall back to a bipartite
            coloring approach if the family is not recognized.
    Raises:
        ValueError: If the bipartite coloring fallback is used and the graph is
            not bipartite.
    Returns:
        A dictionary mapping graph nodes to string color labels.
    """
    graph_family = graph.graph.get("family", None)
    graph_labels = graph.graph.get("labels", None)
    graph_shape = get_graph_shape(graph)
    if graph_family in (
        "chimera",
        "pegasus",
        "zephyr",
    ):
        if graph_labels is None:
            graph_labels = "coordinate"
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
        # Only one of the two bipartite colorings is considered (see Note); this
        # is in general only sufficient for symmetric bipartite graphs.
        try:
            col = nx.bipartite.color(graph)
        except nx.NetworkXError:
            raise ValueError(
                "Coloring fallback requires a bipartite graph"
            )

    return {n: _find_subgraph_label(c) for n, c in col.items()}


def node_labels_by_quotient(
    graph: nx.Graph,
    expand_boundary_search: bool = True,
) -> dict[Hashable, str]:
    """Generate quotient graph labels for nodes based on graph family and structure.

    This function assigns quotient labels to nodes, grouping qubits by
    orientation and displacement within their quotient block.

    For Zephyr graphs with ``expand_boundary_search=True``, boundary quotient
    nodes are collapsed with adjacent internal nodes, since
    those assignments also allow for embeddings.

    Args:
        graph: A Chimera, Pegasus or Zephyr NetworkX graph. The family, node label
            type and shape are inferred from the graph's metadata (as populated by
            the :mod:`dwave.graphs` generators).
        expand_boundary_search: If ``True`` and the graph family is ``"zephyr"``,
            boundary quotient nodes are remapped to adjacent (interior perpendicular
            block offset) nodes. Defaults to ``True``.
    Returns:
        A dictionary mapping node labels to string quotient labels.

    Raises:
        ValueError: If graph family is not found in metadata or is not 'zephyr',
            'pegasus', or 'chimera'.
    """
    graph_family = graph.graph.get("family", None)
    if graph_family in (
        "chimera",
        "pegasus",
        "zephyr",
    ):
        graph_labels = graph.graph.get("labels", None)
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
    return {k: _find_subgraph_label(v) for k, v in col.items()}


def find_labeled_subgraph(
    source: nx.Graph,
    target: nx.Graph,
    labeling_method: Literal["orientation", "quotient", "coloring"] = "orientation",
    node_labels: tuple[dict, dict] | None = None,
    expand_boundary_search: bool = True,
    **kwargs,
) -> dict[Hashable, Hashable]:
    """Find a subgraph of target isomorphic to source that preserves node colors.

    This is a helper function that calls ``find_subgraph`` with ``node_labels``.
    Node labeling can significantly accelerate the search when well chosen. The
    supported labeling methods are intended for bipartite graphs and D-Wave source
    graphs (Zephyr, Pegasus, and Chimera) embedded onto D-Wave target graphs. However,
    in general it may be necessary to consider uncolored search or several
    application-specific node labelings to find embeddings.

    For the ``orientation``, ``quotient`` and ``coloring`` methods on D-Wave
    families, ``source`` and ``target`` should be generated with the
    :mod:`dwave.graphs` topology generators (``chimera_graph``, ``pegasus_graph``
    and ``zephyr_graph``), which populate the ``family``, ``labels`` and shape
    metadata these methods require.

    Args:
        source: Source graph.
        target: Target graph.
        labeling_method: Method to use for coloring the graphs. Options are:
            - 'orientation': It is assumed variables in source must
                map to variables of the same orientation in target.
                If the graph orientation is unclear (the graph is not
                Chimera, Pegasus or Zephyr) the orientation is assigned by
                a bipartite coloring and an error is thrown if the graph is
                not bipartite. Only one of the two colorings is considered,
                which is in general only sufficient for symmetric bipartite
                graphs.
            - 'quotient': Quotient labeling is used. The graphs must
                be of type Chimera, Pegasus or Zephyr. Qubits are labeled
                by orientation and horizontal displacement. Note that this is
                only a good choice if the grid parameter (number of rows
                and columns) of source and target are matched. For Zephyr
                graphs the ``expand_boundary_search`` argument controls whether
                boundary quotient nodes are collapsed with adjacent interior
                nodes.
            - 'coloring': According to canonical 2-colorings of Chimera,
                or the 4-coloring of Pegasus/Zephyr. Note that these colorings
                are not unique. If the graph orientation
                is unclear, the coloring is assigned by a bipartite coloring.
                Only one of the two colorings is considered, which is in general
                only sufficient for symmetric bipartite graphs.
                For high performance applications consider direct
                specification of the coloring via the node_labels argument.
        node_labels: A tuple of dicts mapping nodes in source and target graphs
            to labels. Labels may be any hashable value; each is cast to a
            ``find_subgraph``-safe string (stringified with whitespace removed)
            before the search, so a source and target node are matched only when
            their string labels are equal.
        expand_boundary_search: Forwarded to :func:`node_labels_by_quotient` when
            ``labeling_method='quotient'``; ignored by all other labeling methods
            and when ``node_labels`` is provided. If ``True`` and the graph family
            is ``"zephyr"``, boundary quotient nodes are remapped to adjacent
            interior nodes, enlarging the set of admissible embeddings. Defaults
            to ``True``.
        **kwargs: Additional keyword arguments to pass to find_subgraph.
            Use of a timeout > 0 is recommended.
    """
    if node_labels is None:
        match labeling_method:
            case "orientation":
                node_labels = tuple(
                    node_labels_by_orientation(G) for G in (source, target)
                )
            case "quotient":
                source_family = source.graph.get("family")
                target_family = target.graph.get("family")
                if target_family is None or target_family != source_family:
                    raise ValueError(
                        "Source and target graph families should match for quotient "
                        "coloring"
                    )
                # Supported-family validation is left to node_labels_by_quotient.
                node_labels = tuple(
                    node_labels_by_quotient(
                        G, expand_boundary_search=expand_boundary_search
                    )
                    for G in (source, target)
                )
            case "coloring":
                node_labels = tuple(
                    node_labels_by_coloring(G) for G in (source, target)
                )
            case _:
                raise ValueError(f"Unknown coloring method {labeling_method}")
    # Standardize on find_subgraph-safe string labels, for both the labelings
    # computed above and any user-supplied node_labels.
    safe_node_labels = tuple(
        {k: _find_subgraph_label(v) for k, v in labels.items()}
        for labels in node_labels
    )

    return find_subgraph(source, target, node_labels=safe_node_labels, **kwargs)

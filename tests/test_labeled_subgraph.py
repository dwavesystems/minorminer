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

import unittest

import networkx as nx
from dwave.graphs import (
    zephyr_graph,
    pegasus_graph,
    chimera_graph,
    chimera_two_color,
    pegasus_four_color,
    zephyr_four_color,
)

from minorminer.utils.labeled_subgraph import (
    find_labeled_subgraph,
    graph_label_mapping,
    get_graph_shape,
    node_labels_by_coloring,
    node_labels_by_orientation,
    node_labels_by_quotient,
)


class TestNodeLabelHelpers(unittest.TestCase):
    """Basic tests for helper labeling functions."""

    def test_node_labels_by_orientation_zephyr_matches_orientation_axis(self):
        graph = zephyr_graph(2, 2, coordinates=True)

        labels = node_labels_by_orientation(graph)

        self.assertEqual(set(labels), set(graph.nodes()))
        for node in graph.nodes():
            self.assertEqual(labels[node], node[0])

    def test_node_labels_by_orientation_chimera_matches_orientation_axis(self):
        graph = chimera_graph(2, t=2, coordinates=True)

        labels = node_labels_by_orientation(graph)

        self.assertEqual(set(labels), set(graph.nodes()))
        for node in graph.nodes():
            self.assertEqual(labels[node], node[2])

    def test_node_labels_by_coloring_chimera_matches_per_node_two_color(self):
        # Regression test: coloring must be computed per node, not once for the
        # whole graph.
        graph = chimera_graph(2, t=2, coordinates=True)

        labels = node_labels_by_coloring(graph)

        self.assertEqual(set(labels), set(graph.nodes()))
        for node in graph.nodes():
            self.assertEqual(labels[node], chimera_two_color(node))
        # Sanity check that the coloring is not degenerate (i.e. not all nodes
        # share one color).
        self.assertEqual(set(labels.values()), {0, 1})

    def test_node_labels_by_coloring_pegasus_matches_per_node_four_color(self):
        graph = pegasus_graph(3, coordinates=True)

        labels = node_labels_by_coloring(graph)

        self.assertEqual(set(labels), set(graph.nodes()))
        for node in graph.nodes():
            self.assertEqual(labels[node], pegasus_four_color(node))

    def test_node_labels_by_coloring_zephyr_matches_per_node_four_color(self):
        graph = zephyr_graph(2, 2, coordinates=True)

        labels = node_labels_by_coloring(graph)

        self.assertEqual(set(labels), set(graph.nodes()))
        for node in graph.nodes():
            self.assertEqual(labels[node], zephyr_four_color(node))

    def test_node_labels_by_coloring_falls_back_to_greedy_color_without_family(self):
        # A plain graph without D-Wave family metadata should use
        # nx.greedy_color as a fallback.
        graph = nx.cycle_graph(6)

        labels = node_labels_by_coloring(graph)

        self.assertEqual(set(labels), set(graph.nodes()))
        expected = nx.greedy_color(graph)
        self.assertEqual(labels, expected)

    def test_node_labels_by_quotient_remaps_zephyr_boundaries(self):
        graph = zephyr_graph(2, 2, coordinates=True)

        labels = node_labels_by_quotient(graph, expand_boundary_search=True)

        self.assertEqual(labels[(0, 0, 0, 0, 0)], (0, 1, 0, 0))
        self.assertEqual(labels[(0, 4, 0, 0, 0)], (0, 3, 0, 0))
        self.assertEqual(labels[(0, 2, 0, 0, 0)], (0, 2, 0, 0))

    def test_node_labels_by_quotient_chimera_drops_qubit_index(self):
        graph = chimera_graph(2, t=2, coordinates=True)

        labels = node_labels_by_quotient(graph)

        self.assertEqual(set(labels), set(graph.nodes()))
        for node in graph.nodes():
            self.assertEqual(labels[node], node[:3])

    def test_node_labels_by_orientation_falls_back_to_greedy_color_without_family(self):
        graph = nx.cycle_graph(6)

        labels = node_labels_by_orientation(graph)

        self.assertEqual(set(labels), set(graph.nodes()))
        self.assertEqual(set(labels.values()), {0, 1})

    def test_node_labels_by_orientation_raises_for_non_bipartite_fallback(self):
        graph = nx.cycle_graph(5)

        with self.assertRaises(ValueError):
            node_labels_by_orientation(graph)

    def test_node_labels_by_quotient_pegasus_drops_odd_qubit_bit(self):
        graph = pegasus_graph(3, coordinates=True)

        labels = node_labels_by_quotient(graph)

        self.assertEqual(set(labels), set(graph.nodes()))
        for node in graph.nodes():
            u, w, k, z = node
            self.assertEqual(labels[node], (u, w, k // 2, z))

    def test_node_labels_by_quotient_zephyr_without_boundary_expansion(self):
        graph = zephyr_graph(2, 2, coordinates=True)

        labels = node_labels_by_quotient(graph, expand_boundary_search=False)

        self.assertEqual(set(labels), set(graph.nodes()))
        for node in graph.nodes():
            self.assertEqual(labels[node], node[:2] + node[3:])

    def test_node_labels_by_quotient_raises_for_unrecognized_family(self):
        graph = nx.path_graph(4)

        with self.assertRaises(ValueError):
            node_labels_by_quotient(graph)

    def test_find_labeled_subgraph_returns_identity_on_simple_graph(self):
        source = nx.path_graph(2)
        target = nx.path_graph(2)
        node_labels = ({0: "0", 1: "1"}, {0: "0", 1: "1"})

        embedding = find_labeled_subgraph(
            source, target, node_labels=node_labels, timeout=1
        )

        self.assertIsInstance(embedding, dict)
        self.assertEqual(set(embedding.keys()), set(source.nodes()))
        self.assertEqual(set(embedding.values()), set(target.nodes()))


class TestFindLabeledSubgraphLabelingDispatch(unittest.TestCase):
    """Tests for the automatic node-labeling dispatch in find_labeled_subgraph."""

    def test_orientation_labeling_method_computes_labels_internally(self):
        source = chimera_graph(2, t=2, coordinates=True)
        target = chimera_graph(2, t=2, coordinates=True)

        embedding = find_labeled_subgraph(
            source, target, labeling_method="orientation", timeout=1
        )

        self.assertEqual(set(embedding.keys()), set(source.nodes()))

    def test_coloring_labeling_method_computes_labels_internally(self):
        source = chimera_graph(2, t=2, coordinates=True)
        target = chimera_graph(2, t=2, coordinates=True)

        embedding = find_labeled_subgraph(
            source, target, labeling_method="coloring", timeout=1
        )

        self.assertEqual(set(embedding.keys()), set(source.nodes()))

    def test_quotient_labeling_method_computes_labels_internally(self):
        source = chimera_graph(2, t=2, coordinates=True)
        target = chimera_graph(2, t=2, coordinates=True)

        embedding = find_labeled_subgraph(
            source, target, labeling_method="quotient", timeout=1
        )

        self.assertEqual(set(embedding.keys()), set(source.nodes()))

    def test_quotient_labeling_method_raises_for_mismatched_families(self):
        source = chimera_graph(2, t=2, coordinates=True)
        target = zephyr_graph(2, 2, coordinates=True)

        with self.assertRaises(ValueError):
            find_labeled_subgraph(source, target, labeling_method="quotient")

    def test_quotient_labeling_method_raises_for_unsupported_family(self):
        source = nx.path_graph(4)
        source.graph["family"] = "unsupported"
        target = nx.path_graph(4)
        target.graph["family"] = "unsupported"

        with self.assertRaises(ValueError):
            find_labeled_subgraph(source, target, labeling_method="quotient")

    def test_unknown_labeling_method_raises(self):
        source = nx.path_graph(2)
        target = nx.path_graph(2)

        with self.assertRaises(ValueError):
            find_labeled_subgraph(source, target, labeling_method="unknown")


# (family, graph builder taking a `coordinates` flag, shape) for parameterized tests.
_FAMILY_CASES = [
    (
        "chimera",
        lambda coordinates: chimera_graph(2, t=2, coordinates=coordinates),
        (2, 2, 2),
    ),
    ("zephyr", lambda coordinates: zephyr_graph(2, 2, coordinates=coordinates), (2, 2)),
    ("pegasus", lambda coordinates: pegasus_graph(3, coordinates=coordinates), (3,)),
]


class TestGraphShape(unittest.TestCase):
    """Tests for the get_graph_shape metadata helper."""

    def test_returns_expected_shape_per_family(self):
        expected = {"chimera": (2, 2, 2), "zephyr": (2, 2), "pegasus": (3,)}
        for family, build, shape in _FAMILY_CASES:
            with self.subTest(family=family):
                self.assertEqual(get_graph_shape(build(True)), expected[family])
                self.assertEqual(shape, expected[family])

    def test_returns_none_for_unknown_family(self):
        self.assertIsNone(get_graph_shape(nx.path_graph(3)))


class TestGraphLabelToGraphLabel(unittest.TestCase):
    """Tests for the node-label conversion factory."""

    def test_coordinate_int_roundtrip(self):
        for family, build, shape in _FAMILY_CASES:
            with self.subTest(family=family):
                graph = build(True)
                to_int = graph_label_mapping(family, shape, "coordinate", "int")
                to_coord = graph_label_mapping(family, shape, "int", "coordinate")
                for node in graph.nodes():
                    self.assertEqual(to_coord(to_int(node)), node)

    def test_pegasus_nice_roundtrip(self):
        graph = pegasus_graph(3, coordinates=True)
        to_nice = graph_label_mapping("pegasus", (3,), "coordinate", "nice")
        to_coord = graph_label_mapping("pegasus", (3,), "nice", "coordinate")
        for node in graph.nodes():
            self.assertEqual(to_coord(to_nice(node)), node)

    def test_pegasus_int_nice_roundtrip(self):
        graph = pegasus_graph(3, coordinates=False)
        to_nice = graph_label_mapping("pegasus", (3,), "int", "nice")
        to_int = graph_label_mapping("pegasus", (3,), "nice", "int")
        for node in graph.nodes():
            self.assertEqual(to_int(to_nice(node)), node)

    def test_identity_when_labels_match(self):
        identity = graph_label_mapping("chimera", (2, 2, 2), "int", "int")
        self.assertEqual(identity(5), 5)

    def test_unsupported_family_raises(self):
        with self.assertRaises(ValueError):
            graph_label_mapping("bogus", (2,), "int", "coordinate")

    def test_unsupported_conversion_raises(self):
        # Chimera has no "nice" coordinate system.
        with self.assertRaises(ValueError):
            graph_label_mapping("chimera", (2, 2, 2), "int", "nice")


class TestLabelingRepresentationInvariance(unittest.TestCase):
    """Labelings must be invariant across integer and coordinate representations."""

    def _assert_invariant(self, fn):
        for family, build, shape in _FAMILY_CASES:
            with self.subTest(function=fn.__name__, family=family):
                coord_labels = fn(build(True))
                int_labels = fn(build(False))
                to_coord = graph_label_mapping(family, shape, "int", "coordinate")
                remapped = {to_coord(k): v for k, v in int_labels.items()}
                self.assertEqual(remapped, coord_labels)

    def test_orientation_invariant_across_label_types(self):
        self._assert_invariant(node_labels_by_orientation)

    def test_coloring_invariant_across_label_types(self):
        self._assert_invariant(node_labels_by_coloring)

    def test_quotient_invariant_across_label_types(self):
        self._assert_invariant(node_labels_by_quotient)


class TestIntLabeledGraphs(unittest.TestCase):
    """Regression tests for integer-labeled (coordinates=False) D-Wave graphs."""

    def test_orientation_supports_int_labels(self):
        # Regression: node_labels_by_orientation previously assumed coordinate
        # tuples and raised TypeError on integer-labeled graphs.
        for family, build, _shape in _FAMILY_CASES:
            with self.subTest(family=family):
                graph = build(False)
                labels = node_labels_by_orientation(graph)
                self.assertEqual(set(labels), set(graph.nodes()))
                self.assertEqual(set(labels.values()), {0, 1})

    def test_coloring_and_quotient_support_int_labels(self):
        for fn in (node_labels_by_coloring, node_labels_by_quotient):
            for family, build, _shape in _FAMILY_CASES:
                with self.subTest(function=fn.__name__, family=family):
                    graph = build(False)
                    labels = fn(graph)
                    self.assertEqual(set(labels), set(graph.nodes()))

    def test_find_labeled_subgraph_int_labels_all_methods(self):
        # Regression: dispatch must work on integer-labeled D-Wave graphs, since
        # "orientation" is the default labeling_method.
        for method in ("orientation", "coloring", "quotient"):
            with self.subTest(method=method):
                source = chimera_graph(2, t=2, coordinates=False)
                target = chimera_graph(2, t=2, coordinates=False)
                embedding = find_labeled_subgraph(
                    source, target, labeling_method=method, timeout=2
                )
                self.assertEqual(set(embedding), set(source.nodes()))


class TestNiceLabeledPegasus(unittest.TestCase):
    """Coverage for nice-coordinate Pegasus graphs and inferred metadata."""

    def test_coloring_and_quotient_support_nice_labels(self):
        graph = pegasus_graph(3, nice_coordinates=True)
        for fn in (node_labels_by_coloring, node_labels_by_quotient):
            with self.subTest(function=fn.__name__):
                labels = fn(graph)
                self.assertEqual(set(labels), set(graph.nodes()))

    def test_quotient_defaults_to_coordinate_without_labels_metadata(self):
        # A supported-family graph missing "labels" metadata is treated as
        # coordinate-labeled.
        graph = chimera_graph(2, t=2, coordinates=True)
        del graph.graph["labels"]

        labels = node_labels_by_quotient(graph)

        self.assertEqual(set(labels), set(graph.nodes()))
        for node in graph.nodes():
            self.assertEqual(labels[node], node[:3])

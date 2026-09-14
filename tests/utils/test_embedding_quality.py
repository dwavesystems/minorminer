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

import math
import unittest
import numpy as np

from parameterized import parameterized

import minorminer.utils.embedding_quality as eq
from minorminer.utils.embedding_quality import build_quality, _percentile_sorted, _canonical_edge


def chain(length):
    return tuple(range(length))


def embedding(*lengths):
    return [chain(n) for n in lengths]


# (values, p) grid for the percentile-vs-numpy equivalence test.
_PERCENTILE_CASES = [
    (values, p)
    for values in ([5], [1, 2], [3, 1, 2], [10, 20, 30, 40],
                   [7, 7, 7, 7, 7], [1, 2, 2, 3, 5, 8, 13, 21])
    for p in (0, 1, 25, 50, 75, 93, 99, 100)
]


class TestEmbeddingQuality(unittest.TestCase):

    # distinct node labels for faultiness tests (shape is arbitrary)
    N0, N1, N2, N3 = (0, 0, 0), (0, 4, 0), (1, 1, 0), (1, 5, 0)

    # -- empty embedding --------------------------------------------------
    def test_empty_embedding_is_worst_single_criterion(self):
        q = build_quality("max_chain_length")
        self.assertEqual(q([]), (float("inf"),))

    def test_empty_embedding_matches_hierarchy_arity(self):
        names = ("max_chain_length", "mean_chain_length", "percentile_93_chain_length")
        q = build_quality(names)
        self.assertEqual(q([]), (float("inf"),) * 3)
        self.assertLess(q(embedding(5, 5)), q([]))

    # -- single criterion -------------------------------------------------
    def test_max_chain_length_smaller_is_better(self):
        q = build_quality("max_chain_length")
        self.assertLess(q(embedding(3, 4)), q(embedding(3, 7)))

    def test_mean_chain_length(self):
        q = build_quality("mean_chain_length")
        self.assertEqual(q(embedding(2, 4)), (3.0,))

    # -- hierarchy --------------------------------------------------------
    def test_hierarchy_breaks_ties_with_second_criterion(self):
        q = build_quality(("max_chain_length", "mean_chain_length"))
        a = q(embedding(6, 6))
        b = q(embedding(2, 6))
        self.assertLess(b, a)
        self.assertEqual(a[0], b[0])
        self.assertEqual(a[0], 6.0)

    def test_hierarchy_primary_dominates(self):
        q = build_quality(("max_chain_length", "mean_chain_length"))
        a = q(embedding(3, 5))
        b = q(embedding(1, 9))
        self.assertLess(a, b)

    # -- percentile -------------------------------------------------------
    @parameterized.expand(_PERCENTILE_CASES)
    def test_percentile_matches_numpy(self, values, p):
        sv = sorted(values)
        expected = float(np.percentile(values, p))
        self.assertTrue(
            math.isclose(_percentile_sorted(sv, p), expected,
                         rel_tol=1e-12, abs_tol=1e-12),
            msg=f"_percentile_sorted({sv}, {p})="
                f"{_percentile_sorted(sv, p)} != numpy {expected}",
        )

    def test_percentile_single_value(self):
        for p in (0, 50, 100):
            self.assertEqual(_percentile_sorted([7], p), 7)

    def test_percentile_criterion_end_to_end(self):
        q = build_quality("percentile_50_chain_length")
        self.assertEqual(q(embedding(1, 3, 5, 7)), (4.0,))

    def test_two_percentiles_agree_with_standalone(self):
        q_hier = build_quality(("percentile_50_chain_length",
                                "percentile_93_chain_length"))
        emb = embedding(1, 2, 2, 3, 5, 8, 13, 21)
        p50 = build_quality("percentile_50_chain_length")(emb)[0]
        p93 = build_quality("percentile_93_chain_length")(emb)[0]
        self.assertEqual(q_hier(emb), (p50, p93))

    # -- error handling ---------------------------------------------------
    @parameterized.expand([
        ("unknown_name", "wibble"),
        ("empty_criteria", ()),
        ("percentile_out_of_range", "percentile_150_chain_length"),
    ])
    def test_invalid_criteria_raise(self, _name, criteria):
        with self.assertRaises(ValueError):
            build_quality(criteria)

    # -- empty chain ------------------------------------------------------
    def test_empty_chain_makes_length_criterion_worst(self):
        q = build_quality("mean_chain_length")
        self.assertEqual(q([chain(3), chain(0), chain(5)]), (float("inf"),))

    def test_empty_chain_worst_across_hierarchy_arity(self):
        q = build_quality(("max_chain_length", "percentile_50_chain_length"))
        self.assertEqual(q([chain(4), chain(0)]), (float("inf"),) * 2)

    def test_empty_chain_never_beats_a_real_embedding(self):
        q = build_quality("max_chain_length")
        self.assertLess(q(embedding(9, 9)), q([chain(2), chain(0)]))

    # -- mapping embedding ------------------------------------------------
    def test_mapping_scored_over_values(self):
        q = build_quality("max_chain_length")
        as_map = {0: chain(3), 1: chain(7)}
        as_seq = [chain(3), chain(7)]
        self.assertEqual(q(as_map), q(as_seq))
        self.assertEqual(q(as_map), (7.0,))

    def test_empty_mapping_is_worst(self):
        q = build_quality("max_chain_length")
        self.assertEqual(q({}), (float("inf"),))

    def test_mapping_with_empty_chain_is_worst(self):
        q = build_quality("mean_chain_length")
        self.assertEqual(q({0: chain(4), 1: chain(0)}), (float("inf"),))

    # -- missing_edges statefulness ---------------------------------------
    def test_generator_missing_edges_survives_repeated_evaluation(self):
        fm = {"edges": {(self.N0, self.N2): 0.4}}
        gen = (e for e in [(self.N2, self.N0)])   # one-shot generator
        q = build_quality("faultiness", fault_map=fm, missing_edges=gen)
        emb = [[self.N0, self.N2]]
        first = q(emb)
        second = q(emb)
        self.assertEqual(first, (0.0,))
        self.assertEqual(second, first)

    # -- edge canonicalization --------------------------------------------
    def test_canonical_edge_order_independent(self):
        self.assertEqual(_canonical_edge("a", "b"), _canonical_edge("b", "a"))

    def test_faultiness_with_unorderable_labels(self):
        a, b = frozenset({1}), frozenset({2})   # hashable, not <=-orderable
        fm = {"nodes": {a: 0.3}, "edges": {(a, b): 0.5}}
        q = build_quality("faultiness", fault_map=fm)
        self.assertAlmostEqual(q([[a, b]])[0], 0.8)

    # -- faultiness metric ------------------------------------------------
    def test_faultiness_node_sum(self):
        fm = {"nodes": {self.N0: 0.2, self.N1: 0.5, self.N2: 0.9}}
        q = build_quality("faultiness", fault_map=fm)
        self.assertAlmostEqual(q([[self.N0, self.N1]])[0], 0.7)

    def test_faultiness_absent_node_weighs_zero(self):
        fm = {"nodes": {self.N0: 0.2}}
        q = build_quality("faultiness", fault_map=fm)
        self.assertAlmostEqual(q([[self.N0, self.N3]])[0], 0.2)

    def test_faultiness_edge_counted_only_when_both_endpoints_used(self):
        fm = {"edges": {(self.N0, self.N2): 0.4}}
        q = build_quality("faultiness", fault_map=fm)
        self.assertAlmostEqual(q([[self.N0, self.N2]])[0], 0.4)
        self.assertAlmostEqual(q([[self.N0, self.N1]])[0], 0.0)

    def test_faultiness_edge_key_order_independent(self):
        fm = {"edges": {(self.N2, self.N0): 0.4}}
        q = build_quality("faultiness", fault_map=fm)
        self.assertAlmostEqual(q([[self.N0, self.N2]])[0], 0.4)

    def test_faultiness_edge_under_both_orders_counted_once(self):
        fm = {"edges": {(self.N0, self.N2): 0.4, (self.N2, self.N0): 0.4}}
        q = build_quality("faultiness", fault_map=fm)
        self.assertAlmostEqual(q([[self.N0, self.N2]])[0], 0.4)

    def test_faultiness_missing_edge_not_counted(self):
        fm = {"edges": {(self.N0, self.N2): 0.4}}
        q = build_quality("faultiness", fault_map=fm,
                          missing_edges={(self.N2, self.N0)})
        self.assertAlmostEqual(q([[self.N0, self.N2]])[0], 0.0)

    def test_faultiness_node_and_edge_combined(self):
        fm = {"nodes": {self.N0: 0.1, self.N2: 0.2},
              "edges": {(self.N0, self.N2): 0.4}}
        q = build_quality("faultiness", fault_map=fm)
        self.assertAlmostEqual(q([[self.N0, self.N2]])[0], 0.7)

    def test_faultiness_no_fault_map_scores_zero(self):
        q = build_quality("faultiness")
        self.assertEqual(q([[self.N0, self.N1, self.N2]]), (0,))

    def test_faultiness_empty_embedding_infinite(self):
        fm = {"nodes": {self.N0: 0.5}}
        q = build_quality("faultiness", fault_map=fm)
        self.assertEqual(q([]), (float("inf"),))

    def test_faultiness_smaller_is_better(self):
        fm = {"nodes": {self.N0: 0.0, self.N1: 0.0, self.N2: 1.0}}
        q = build_quality("faultiness", fault_map=fm)
        self.assertLess(q([[self.N0, self.N1]]), q([[self.N0, self.N2]]))

    def test_faultiness_mixes_with_length_in_hierarchy(self):
        fm = {"nodes": {self.N0: 0.0, self.N1: 0.0, self.N2: 1.0}}
        q = build_quality(("max_chain_length", "faultiness"), fault_map=fm)
        clean = q([[self.N0, self.N1]])
        dirty = q([[self.N0, self.N2]])
        self.assertEqual(clean[0], dirty[0])
        self.assertLess(clean, dirty)

    # -- shared prep: public faultiness == criterion path -----------------
    def test_public_and_criterion_agree(self):
        fm = {"nodes": {self.N0: 0.1, self.N2: 0.2},
              "edges": {(self.N0, self.N2): 0.4}}
        direct = eq.faultiness([[self.N0, self.N2]], fault_map=fm)
        via = build_quality("faultiness", fault_map=fm)([[self.N0, self.N2]])[0]
        self.assertAlmostEqual(direct, 0.7)
        self.assertAlmostEqual(via, 0.7)

    def test_public_faultiness_honors_missing_edges(self):
        fm = {"edges": {(self.N0, self.N2): 0.4}}
        val = eq.faultiness([[self.N0, self.N2]], fault_map=fm,
                            missing_edges={(self.N2, self.N0)})
        self.assertEqual(val, 0)

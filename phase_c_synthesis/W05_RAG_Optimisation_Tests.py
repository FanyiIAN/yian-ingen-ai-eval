from __future__ import annotations

import unittest
import random

from W05_RAG_Optimisation import (
    DEFAULT_CONFIG,
    DEFAULT_FIXTURE,
    Variant,
    build_factorial_matrix,
    build_matched_contrasts,
    dominates,
    lexical_context_support,
    load_yaml,
    materialize_documents,
    pareto_frontier,
    required_term_coverage,
    token_f1,
    validate_config,
    variant_differences,
)


class Week5RAGOptimisationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_yaml(DEFAULT_CONFIG)
        cls.fixture = load_yaml(DEFAULT_FIXTURE)

    def test_frozen_factorial_matrix_has_18_unique_variants(self) -> None:
        variants = build_factorial_matrix(self.config)
        self.assertEqual(len(variants), 18)
        self.assertEqual(len({variant.variant_id for variant in variants}), 18)
        self.assertEqual(
            {variant.chunk_size_tokens for variant in variants}, {256, 512, 1024}
        )
        self.assertEqual({variant.top_k for variant in variants}, {1, 3, 5})
        self.assertEqual(
            {variant.reranking for variant in variants}, {"none", "cross_encoder"}
        )

    def test_matched_contrasts_change_exactly_one_factor(self) -> None:
        variants = build_factorial_matrix(self.config)
        by_id = {variant.variant_id: variant for variant in variants}
        contrasts = build_matched_contrasts(variants)
        self.assertEqual(len(contrasts), 45)
        for contrast in contrasts:
            left = by_id[contrast["left_variant_id"]]
            right = by_id[contrast["right_variant_id"]]
            self.assertEqual(variant_differences(left, right), [contrast["factor"]])

    def test_config_audit_preserves_traceability_and_limitation(self) -> None:
        audit = validate_config(self.config)
        self.assertEqual(audit["variant_count"], 18)
        self.assertEqual(audit["matched_contrast_count"], 45)
        self.assertEqual(
            audit["matched_contrasts_by_factor"],
            {"reranking": 9, "top_k": 18, "chunk_size_tokens": 18},
        )

    def test_fixture_materialization_is_deterministic_and_long(self) -> None:
        first = materialize_documents(self.fixture)
        second = materialize_documents(self.fixture)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertTrue(all(len(document["text"].split()) > 500 for document in first))

    def test_proxy_metrics_have_expected_bounds(self) -> None:
        response = "Teachers retain grades and use a pseudonymous skill record."
        context = "Teachers retain authority over grades. Use a pseudonymous learner skill."
        reference = "Teachers retain grades and use a pseudonymous identifier."
        terms = [["teacher", "teachers"], ["grades"], ["pseudonymous"]]
        self.assertGreater(lexical_context_support(response, context), 0.8)
        self.assertGreater(token_f1(response, reference), 0.7)
        self.assertEqual(required_term_coverage(response, terms), 1.0)

    def test_pareto_frontier_respects_two_quality_metrics_and_latency(self) -> None:
        rows = [
            {
                "variant_id": "a",
                "mean_lexical_context_support_proxy": 0.8,
                "mean_required_term_coverage_proxy": 0.7,
                "mean_question_to_response_ms": 100.0,
            },
            {
                "variant_id": "b",
                "mean_lexical_context_support_proxy": 0.7,
                "mean_required_term_coverage_proxy": 0.6,
                "mean_question_to_response_ms": 120.0,
            },
            {
                "variant_id": "c",
                "mean_lexical_context_support_proxy": 0.9,
                "mean_required_term_coverage_proxy": 0.6,
                "mean_question_to_response_ms": 90.0,
            },
        ]
        self.assertTrue(dominates(rows[0], rows[1]))
        self.assertEqual(
            {row["variant_id"] for row in pareto_frontier(rows)}, {"a", "c"}
        )

    def test_variant_difference_helper(self) -> None:
        left = Variant(256, 1, "none")
        right = Variant(256, 3, "none")
        self.assertEqual(variant_differences(left, right), ["top_k"])

    def test_seeded_execution_shuffle_is_reproducible_and_not_sorted(self) -> None:
        variants = build_factorial_matrix(self.config)
        first = list(variants)
        second = list(variants)
        random.Random(42).shuffle(first)
        random.Random(42).shuffle(second)
        self.assertEqual(first, second)
        self.assertNotEqual(first, variants)


if __name__ == "__main__":
    unittest.main()

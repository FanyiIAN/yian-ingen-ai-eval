from __future__ import annotations

import unittest

from W05_RAG_Coverage_Scoring import cache_key as coverage_cache_key
from W05_RAG_Coverage_Scoring import (
    coverage_retry_messages,
    normalize_coverage,
    parse_coverage_value,
)
from W05_RAG_Production_Run import (
    DEFAULT_CONFIG,
    build_variants,
    freeze_senpai_subset,
    load_yaml,
    randomized_variants,
    source_path,
    validate_protocol,
    variant_week3_config,
)
from W05_RAG_RAGAS_Scoring import cache_key as ragas_cache_key
from W05_RAG_RAGAS_Scoring import returned_metrics, score_with_retries


class ProductionProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_yaml(DEFAULT_CONFIG)
        cls.eval_set = load_yaml(
            source_path(
                DEFAULT_CONFIG,
                cls.config["source_inputs"]["evaluation_set"]["path"],
            )
        )
        cls.week3_config = load_yaml(
            source_path(
                DEFAULT_CONFIG,
                cls.config["source_inputs"]["week3_run_config"]["path"],
            )
        )

    def test_frozen_protocol_has_full_factorial_and_traceability(self) -> None:
        validation = validate_protocol(self.config, DEFAULT_CONFIG)
        self.assertEqual(validation["variants"], 18)
        self.assertEqual(validation["expected_rows"], 360)
        self.assertEqual(validation["matched_pairs"], 45)
        self.assertEqual(set(validation["model_revisions"]), {
            "generator", "embedding", "reranker", "evaluator"
        })

    def test_seeded_variant_order_is_deterministic_and_complete(self) -> None:
        first = [variant.variant_id for variant in randomized_variants(self.config)]
        second = [variant.variant_id for variant in randomized_variants(self.config)]
        self.assertEqual(first, second)
        self.assertEqual(len(first), 18)
        self.assertEqual(len(set(first)), 18)
        self.assertEqual(set(first), {variant.variant_id for variant in build_variants(self.config)})

    def test_subset_is_exactly_the_registered_senpai_items(self) -> None:
        subset = freeze_senpai_subset(self.eval_set, self.config)
        observed = [row["eval_id"] for row in subset["items"]]
        expected = self.config["source_inputs"]["evaluation_set"]["item_ids"]
        self.assertEqual(observed, expected)
        self.assertEqual(subset["item_count"], 20)
        self.assertEqual({row["platform"] for row in subset["items"]}, {"Senpai"})

    def test_variant_changes_only_registered_retrieval_factors(self) -> None:
        variant = build_variants(self.config)[-1]
        derived = variant_week3_config(self.week3_config, variant, 20)
        self.assertEqual(
            derived["retrieval"]["text_splitter"]["chunk_size_tokens"],
            variant.chunk_size_tokens,
        )
        self.assertEqual(derived["retrieval"]["retriever"]["top_k"], variant.top_k)
        self.assertEqual(
            derived["retrieval"]["reranker"]["enabled"],
            variant.reranking == "cross_encoder",
        )
        self.assertEqual(
            derived["generation"]["candidate_model_revision"],
            self.week3_config["generation"]["candidate_model_revision"],
        )


class CoverageScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.item = {
            "required_points": [
                {"point_id": "P1", "weight": 3, "criterion": "one"},
                {"point_id": "P2", "weight": 1, "criterion": "two"},
            ],
            "forbidden_points": [
                {"point_id": "X1", "criterion": "forbidden"}
            ],
        }

    def test_weighted_required_point_coverage(self) -> None:
        normalized = normalize_coverage(
            {
                "point_scores": [
                    {"point_id": "P1", "score": 1, "evidence": "yes"},
                    {"point_id": "P2", "score": 0, "evidence": ""},
                ],
                "forbidden_point_violations": [],
                "rationale": "fixture",
            },
            self.item,
        )
        self.assertEqual(normalized["required_point_coverage"], 0.75)

    def test_missing_or_unknown_points_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing point"):
            normalize_coverage(
                {
                    "point_scores": [
                        {"point_id": "P1", "score": 1, "evidence": "yes"}
                    ],
                    "forbidden_point_violations": [],
                },
                self.item,
            )

    def test_coverage_cache_is_item_and_answer_specific(self) -> None:
        first = {
            "eval_id": "e1",
            "candidate_output_sha256": "a",
        }
        second = dict(first, candidate_output_sha256="b")
        self.assertNotEqual(coverage_cache_key(first), coverage_cache_key(second))

    def test_lenient_parser_repairs_syntax_but_not_scores(self) -> None:
        raw = (
            '{"point_scores":[{"point_id":"P1","score":1,"evidence":"a\n'
            'b"},{"point_id":"P2","score":0,"evidence":""}],'
            '"forbidden_points":[]}'
        )

        def strict_parser(value: str):
            return __import__("json").loads(value)

        value, repairs = parse_coverage_value(raw, strict_parser)
        self.assertEqual(value["forbidden_point_violations"], [])
        self.assertEqual(value["point_scores"][1]["score"], 0)
        self.assertIn("json_decode_strict_false_for_control_characters", repairs)
        self.assertIn("renamed_forbidden_points_alias", repairs)

    def test_retry_prompt_contains_every_exact_point_once_in_template(self) -> None:
        messages = coverage_retry_messages(
            {"candidate_output": "answer"},
            {**self.item, "question": "question"},
            "bad",
            "missing point",
            2,
        )
        self.assertIn('"point_id":"P1"', messages[0]["content"])
        self.assertIn('"point_id":"P2"', messages[0]["content"])
        self.assertNotIn("previous_invalid_output", messages[1]["content"])
        self.assertIn("P1: one", messages[1]["content"])
        self.assertIn("P2: two", messages[1]["content"])

    def test_parser_removes_only_registered_invalid_backslashes(self) -> None:
        raw = (
            '{"point_scores":[{"point_id":"P1","score":1,"evidence":"yes"},'
            '\\{"point_id":"P2","score":0,"evidence":""}],'
            '"forbidden_point_violations":[]}'
        )

        def strict_parser(value: str):
            return __import__("json").loads(value)

        value, repairs = parse_coverage_value(raw, strict_parser)
        self.assertEqual(len(value["point_scores"]), 2)
        self.assertIn("removed_invalid_backslash_before_brace_or_bracket", repairs)

    def test_equal_score_duplicates_merge_without_changing_score(self) -> None:
        raw = (
            '{"point_scores":['
            '{"point_id":"P1","score":1,"evidence":"a"},'
            '{"point_id":"P1","score":1,"evidence":"b"},'
            '{"point_id":"P2","score":0,"evidence":""}],'
            '"forbidden_point_violations":[]}'
        )

        def strict_parser(value: str):
            return __import__("json").loads(value)

        value, repairs = parse_coverage_value(raw, strict_parser)
        normalized = normalize_coverage(value, self.item)
        self.assertEqual(normalized["per_point_verdicts"][0]["score"], 1.0)
        self.assertEqual(normalized["per_point_verdicts"][0]["evidence"], "a | b")
        self.assertIn("merged_duplicate_point_rows_with_equal_scores", repairs)

    def test_ragas_cache_includes_context(self) -> None:
        record = {
            "question": "q",
            "candidate_output": "a",
            "retrieved_contexts": ["c1"],
        }
        changed = dict(record, retrieved_contexts=["c2"])
        self.assertNotEqual(ragas_cache_key(record), ragas_cache_key(changed))


class RagasRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_value_is_retried_but_nan_is_retained(self) -> None:
        class FakeEvaluator:
            def __init__(self, values):
                self.values = iter(values)

            async def safe_score(self, metric, **kwargs):
                return {"value": next(self.values), "reason": None}

        evaluator = FakeEvaluator([None, 0.75])
        result, attempts = await score_with_retries(
            object(), evaluator, max_attempts=3, user_input="q"
        )
        self.assertEqual(result["value"], 0.75)
        self.assertEqual(len(attempts), 2)

        evaluator = FakeEvaluator([float("nan"), 0.5])
        result, attempts = await score_with_retries(
            object(), evaluator, max_attempts=3, user_input="q"
        )
        self.assertNotEqual(result["value"], result["value"])
        self.assertEqual(len(attempts), 1)

    def test_returned_nan_is_cacheable_but_missing_value_is_not(self) -> None:
        row = {
            "metrics": {
                "answer_relevance": {"value": 0.0},
                "faithfulness_to_retrieved_context": {"value": float("nan")},
            }
        }
        self.assertTrue(returned_metrics(row))
        row["metrics"]["answer_relevance"]["value"] = None
        self.assertFalse(returned_metrics(row))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from W05_RAG_Optimisation_Analysis import (
    aggregate,
    chunk_identity,
    dominates,
    matched_contrasts,
    pareto_frontier,
    summarize_contrasts,
)


def summary(
    variant_id: str, faith: float, coverage: float, latency: float
) -> dict:
    return {
        "variant_id": variant_id,
        "chunk_size_tokens": 256,
        "top_k": 1,
        "reranking": "none",
        "mean_faithfulness": faith,
        "mean_required_point_coverage": coverage,
        "p50_question_to_response_ms": latency,
    }


def joined_row(variant_id: str, eval_id: str, chunk: int = 256) -> dict:
    return {
        "variant_id": variant_id,
        "eval_id": eval_id,
        "chunk_size_tokens": chunk,
        "top_k": 1,
        "reranking": "none",
        "faithfulness": 0.8,
        "answer_relevance": 0.7,
        "required_point_coverage": 0.75,
        "question_to_response_ms": 100.0 + chunk,
        "evidence_fact_recall_at_k": 1.0,
        "forbidden_point_violations": 0,
        "metadata_filter_leakage": 0,
        "candidate_messages_sha256": "same-message",
        "candidate_output_sha256": "same-output",
    }


class OptimisationAnalysisTests(unittest.TestCase):
    def test_pareto_requires_no_worse_all_and_better_one(self) -> None:
        a = summary("a", 0.9, 0.8, 100)
        b = summary("b", 0.8, 0.8, 110)
        c = summary("c", 0.95, 0.7, 90)
        self.assertTrue(dominates(a, b))
        self.assertFalse(dominates(a, c))
        self.assertEqual(
            {row["variant_id"] for row in pareto_frontier([a, b, c])},
            {"a", "c"},
        )

    def test_aggregate_retains_metric_coverage(self) -> None:
        rows = [joined_row("v", "e1"), joined_row("v", "e2")]
        rows[1]["faithfulness"] = None
        result = aggregate(rows)[0]
        self.assertEqual(result["items"], 2)
        self.assertEqual(result["faithfulness_coverage"], 0.5)
        self.assertEqual(result["mean_faithfulness"], 0.8)

    def test_matched_contrast_changes_exactly_one_factor(self) -> None:
        left = summary("chunk-256_topk-1_rerank-none", 0.8, 0.7, 100)
        right = summary("chunk-512_topk-1_rerank-none", 0.8, 0.7, 120)
        right["chunk_size_tokens"] = 512
        joined = [
            joined_row(left["variant_id"], "e1", 256),
            joined_row(right["variant_id"], "e1", 512),
        ]
        result = matched_contrasts([left, right], joined)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["factor"], "chunk_size_tokens")
        self.assertEqual(result[0]["mean_delta_question_to_response_ms"], 256.0)

    def test_chunk_identity_is_measured_not_assumed(self) -> None:
        rows = [
            joined_row("v256", "e1", 256),
            joined_row("v512", "e1", 512),
            joined_row("v1024", "e1", 1024),
        ]
        result = chunk_identity(rows)
        self.assertEqual(result["three_chunk_level_matched_groups"], 1)
        self.assertEqual(result["message_identity_rate"], 1.0)
        self.assertEqual(result["output_identity_rate"], 1.0)

    def test_zero_chunk_identity_confirms_operational_factor(self) -> None:
        rows = [
            joined_row("v256", "e1", 256),
            joined_row("v512", "e1", 512),
            joined_row("v1024", "e1", 1024),
        ]
        for index, row in enumerate(rows):
            row["candidate_messages_sha256"] = f"message-{index}"
            row["candidate_output_sha256"] = f"output-{index}"
        result = chunk_identity(rows)
        self.assertEqual(result["message_identity_rate"], 0.0)
        self.assertIn("operational factor", result["interpretation"])

    def test_contrast_summary_preserves_interaction_group(self) -> None:
        contrasts = [
            {
                "fixed_top_k": 1,
                "mean_delta_faithfulness": 0.1,
                "mean_delta_answer_relevance": 0.2,
                "mean_delta_required_point_coverage": 0.3,
                "mean_delta_question_to_response_ms": 10.0,
            },
            {
                "fixed_top_k": 1,
                "mean_delta_faithfulness": -0.1,
                "mean_delta_answer_relevance": 0.0,
                "mean_delta_required_point_coverage": 0.1,
                "mean_delta_question_to_response_ms": 30.0,
            },
        ]
        result = summarize_contrasts(contrasts, "fixed_top_k")[0]
        self.assertEqual(result["contrasts"], 2)
        self.assertEqual(result["mean_delta_faithfulness"], 0.0)
        self.assertEqual(result["mean_delta_question_to_response_ms"], 20.0)


if __name__ == "__main__":
    unittest.main()

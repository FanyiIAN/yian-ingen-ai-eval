from __future__ import annotations

import math
import asyncio
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from W03_RAG_Expanded_MultiModel_RAGAS import (
    DEFAULT_MODEL_KEYS,
    CONTEXT_METRIC_NAMES,
    LocalRAGASBundle,
    batches,
    context_metric_cache_key,
    finite_number,
)


class ExpandedMultiModelRAGASTests(unittest.TestCase):
    def test_registered_candidates_match_three_model_run(self) -> None:
        self.assertEqual(
            set(DEFAULT_MODEL_KEYS),
            {
                "flan_t5_base",
                "mistral_7b_instruct_v0_2",
                "llama31_8b_instruct",
            },
        )

    def test_invalid_judge_values_remain_invalid(self) -> None:
        self.assertTrue(finite_number(0.5))
        self.assertFalse(finite_number(None))
        self.assertFalse(finite_number(math.nan))
        self.assertFalse(finite_number(math.inf))
        self.assertFalse(finite_number(True))

    def test_batches_preserve_order_and_reject_zero_size(self) -> None:
        self.assertEqual(batches([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]])
        with self.assertRaisesRegex(ValueError, "positive"):
            batches([1], 0)

    def test_transient_metric_error_is_retried_and_recorded(self) -> None:
        bundle = object.__new__(LocalRAGASBundle)
        with patch(
            "W03_RAG_Expanded_MultiModel_RAGAS.timed_score",
            new=AsyncMock(side_effect=[RuntimeError("transient"), {"value": 0.5}]),
        ) as scorer:
            result = asyncio.run(bundle.safe_score(object()))

        self.assertEqual(result["value"], 0.5)
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(len(result["retry_errors"]), 1)
        self.assertEqual(scorer.await_count, 2)

    def test_context_metrics_are_reused_across_candidate_outputs(self) -> None:
        bundle = object.__new__(LocalRAGASBundle)
        bundle.answer_relevancy = object()
        bundle.faithfulness = object()
        bundle.context_relevance = object()
        bundle.context_recall = object()
        bundle.context_precision = object()
        bundle.judge_model = "judge"
        bundle.embedding_model_id = "embedding"
        bundle.embedding_model_dir = Path("embedding")
        bundle._context_metric_cache = {}
        bundle.context_metric_cache_stats = {
            "computed": 0,
            "reused": 0,
            "seeded": 0,
        }
        bundle.safe_score = AsyncMock(return_value={"value": 0.5, "attempts": 1})
        first = {
            "run_item_id": "model-a::Q1::rag",
            "condition": "rag",
            "question": "Question?",
            "candidate_output": "First answer.",
            "retrieved_contexts": ["Context."],
            "reference_answer": "Reference.",
        }
        second = {
            **first,
            "run_item_id": "model-b::Q1::rag",
            "candidate_output": "Second answer.",
        }

        with patch("importlib.metadata.version", return_value="test"):
            asyncio.run(bundle.score_record(first))
            reused = asyncio.run(bundle.score_record(second))

        self.assertEqual(bundle.safe_score.await_count, 7)
        self.assertEqual(bundle.context_metric_cache_stats["computed"], 3)
        self.assertEqual(bundle.context_metric_cache_stats["reused"], 3)
        for name in CONTEXT_METRIC_NAMES:
            self.assertEqual(
                reused["ragas"]["metrics"][name]["reused_from_run_item_id"],
                first["run_item_id"],
            )
            self.assertIsNone(reused["ragas"]["metrics"][name]["latency_ms"])

    def test_context_cache_key_ignores_candidate_output(self) -> None:
        first = {
            "question": "Question?",
            "retrieved_contexts": ["Context."],
            "reference_answer": "Reference.",
            "candidate_output": "A",
        }
        second = {**first, "candidate_output": "B"}
        self.assertEqual(context_metric_cache_key(first), context_metric_cache_key(second))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import math
import unittest

from W03_RAG_Result_Analysis import (
    finite_float,
    metric_deltas,
    summarize_rows,
)


def score_row(
    run_item_id: str,
    condition: str,
    answer_relevance: object,
    faithfulness: object,
) -> dict[str, object]:
    return {
        "run_item_id": run_item_id,
        "eval_id": run_item_id.split("::")[0],
        "condition": condition,
        "candidate_model_id": "local/test-model",
        "random_seed": 42,
        "ragas": {
            "metrics": {
                "answer_relevance": {
                    "value": answer_relevance,
                    "reason": None,
                },
                "faithfulness_to_retrieved_context": {
                    "value": faithfulness,
                    "reason": None,
                },
                "context_relevance": {"value": 0.5, "reason": None},
                "context_recall": {"value": 1.0, "reason": None},
                "context_precision": {"value": 0.75, "reason": None},
            }
        },
    }


class ResultAnalysisTests(unittest.TestCase):
    def test_non_finite_values_are_not_coerced_to_zero(self) -> None:
        rows = [
            score_row("A::rag", "rag", 0.8, 1.0),
            score_row("B::rag", "rag", 0.6, "NaN"),
        ]
        summary = summarize_rows(rows)
        metric = summary["conditions"]["rag"]["metrics"][
            "faithfulness_to_retrieved_context"
        ]
        self.assertEqual(metric["finite_count"], 1)
        self.assertEqual(metric["invalid_count"], 1)
        self.assertEqual(metric["finite_mean"], 1.0)
        self.assertEqual(metric["invalid_rows"][0]["run_item_id"], "B::rag")

    def test_base_rag_only_metrics_are_not_applicable(self) -> None:
        rows = [score_row("A::base", "base", 0.4, None)]
        summary = summarize_rows(rows)
        metric = summary["conditions"]["base"]["metrics"]["context_recall"]
        self.assertFalse(metric["applicable"])
        self.assertEqual(metric["invalid_count"], 0)

    def test_metric_delta_retains_coverage(self) -> None:
        parent = summarize_rows(
            [
                score_row("A::rag", "rag", 0.5, 1.0),
                score_row("B::rag", "rag", 0.5, math.nan),
            ]
        )
        candidate = summarize_rows(
            [
                score_row("A::rag", "rag", 0.6, 1.0),
                score_row("B::rag", "rag", 0.6, 1.0),
            ]
        )
        deltas = metric_deltas(parent, candidate)
        self.assertEqual(
            deltas["answer_relevance"]["delta_candidate_minus_parent"],
            0.1,
        )
        self.assertEqual(
            deltas["faithfulness_to_retrieved_context"][
                "candidate_finite_count"
            ],
            2,
        )
        self.assertEqual(
            deltas["faithfulness_to_retrieved_context"][
                "parent_invalid_count"
            ],
            1,
        )

    def test_finite_float_rejects_invalid_types(self) -> None:
        self.assertIsNone(finite_float(None))
        self.assertIsNone(finite_float(True))
        self.assertIsNone(finite_float("not-a-number"))
        self.assertEqual(finite_float("0.25"), 0.25)


if __name__ == "__main__":
    unittest.main()


"""Focused tests for the long-source Week 3 analysis helpers."""

from __future__ import annotations

import math
import unittest

from W03_Long_RAG_Analysis import describe, matched_deltas


class LongRAGAnalysisTests(unittest.TestCase):
    def test_describe_excludes_missing_and_non_finite_values(self) -> None:
        value = describe([1, None, float("nan"), float("inf"), 3])
        self.assertEqual(value["finite_n"], 2)
        self.assertEqual(value["mean"], 2.0)
        self.assertTrue(math.isclose(value["p50"], 2.0))

    def test_matched_deltas_compare_the_same_eval_item(self) -> None:
        rows = [
            {
                "eval_id": "A",
                "condition": "base",
                "answer_relevance": 0.2,
                "required_point_coverage": 0.25,
                "generation_latency_ms": 100,
            },
            {
                "eval_id": "A",
                "condition": "rag",
                "answer_relevance": 0.7,
                "required_point_coverage": 0.75,
                "generation_latency_ms": 400,
            },
        ]
        result = matched_deltas(rows)
        self.assertEqual(result["required_point_coverage"]["mean"], 0.5)
        self.assertEqual(result["required_point_coverage"]["positive"], 1)
        self.assertEqual(result["generation_latency_ms"]["mean"], 300.0)


if __name__ == "__main__":
    unittest.main()

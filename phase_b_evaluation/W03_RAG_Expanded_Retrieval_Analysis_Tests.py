from __future__ import annotations

import unittest

from W03_RAG_Expanded_Retrieval_Analysis import analyze, metric_summary


class ExpandedRetrievalAnalysisTests(unittest.TestCase):
    def test_metric_summary_uses_interpolated_quantiles(self) -> None:
        summary = metric_summary([1, 2, 3, 4, 5])
        self.assertEqual(summary["mean"], 3.0)
        self.assertEqual(summary["p50"], 3.0)
        self.assertEqual(summary["p95"], 4.8)

    def test_analysis_recovers_fact_recall_and_leakage(self) -> None:
        eval_set = {
            "items": [
                {
                    "eval_id": "E1",
                    "platform": "Fari",
                    "difficulty": "hard",
                    "answerability": "answerable",
                    "reference_document_ids": ["D1"],
                    "evidence_fact_ids": ["F1", "F2"],
                }
            ]
        }
        rows = [
            {"eval_id": "E1", "condition": "base"},
            {
                "eval_id": "E1",
                "condition": "rag",
                "platform": "Fari",
                "retrieval_latency_ms": 25.0,
                "retrieved_contexts": [
                    {
                        "fact_ids_json": '["F1"]',
                        "token_count": 20,
                        "owner_type": "official",
                        "access_scope": "public",
                        "confidentiality": "public",
                        "source_domain": "www.ingendynamics.com",
                        "document_id": "D1",
                    }
                ],
            },
        ]
        summary = analyze(rows, eval_set)
        self.assertEqual(summary["mean_evidence_fact_recall_at_k"], 0.5)
        self.assertEqual(summary["full_evidence_items"], 0)
        self.assertEqual(summary["metadata_filter_leakage"], 0)
        self.assertEqual(
            summary["incomplete_evidence_rows"][0][
                "missing_evidence_fact_ids"
            ],
            ["F2"],
        )


if __name__ == "__main__":
    unittest.main()

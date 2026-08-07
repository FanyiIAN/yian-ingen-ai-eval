from __future__ import annotations

import unittest

from W03_RAG_Expanded_Retrieval_Ablation import (
    configure_variant,
    percentile,
    summarize,
)


class ExpandedRetrievalAblationTests(unittest.TestCase):
    def test_percentile_interpolates_registered_quantiles(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertEqual(percentile(values, 0.50), 3.0)
        self.assertEqual(percentile(values, 0.95), 4.8)

    def test_variant_does_not_mutate_base_config(self) -> None:
        base = {
            "retrieval": {
                "retriever": {"fetch_k": 24, "top_k": 6},
                "reranker": {"enabled": True, "candidate_pool_size": 24},
            }
        }
        variant = configure_variant(base, 10, False)
        self.assertEqual(base["retrieval"]["retriever"]["top_k"], 6)
        self.assertEqual(variant["retrieval"]["retriever"]["fetch_k"], 32)
        self.assertEqual(variant["retrieval"]["retriever"]["top_k"], 10)
        self.assertFalse(variant["retrieval"]["reranker"]["enabled"])

    def test_summary_counts_full_evidence_and_context_budget(self) -> None:
        result = {
            "summary": {
                "mean_document_id_recall_at_k": 1.0,
                "mean_evidence_fact_recall_at_k": 0.75,
                "hit_at_k": 2,
                "mean_reciprocal_rank": 1.0,
                "metadata_filter_leakage": 0,
            },
            "rows": [
                {
                    "retrieval_latency_ms": 10.0,
                    "evidence_fact_recall_at_k": 1.0,
                    "retrieval_trace": [{"token_count": 20}],
                },
                {
                    "retrieval_latency_ms": 30.0,
                    "evidence_fact_recall_at_k": 0.5,
                    "retrieval_trace": [
                        {"token_count": 10},
                        {"token_count": 30},
                    ],
                },
            ],
        }
        row = summarize(result, 8, True)
        self.assertEqual(row["full_evidence_items"], 1)
        self.assertEqual(row["mean_returned_context_units"], 1.5)
        self.assertEqual(row["mean_returned_context_tokens"], 30.0)
        self.assertEqual(row["latency_ms"]["mean"], 20.0)


if __name__ == "__main__":
    unittest.main()

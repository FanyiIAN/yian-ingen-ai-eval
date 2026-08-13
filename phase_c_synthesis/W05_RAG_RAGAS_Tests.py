from __future__ import annotations

import unittest

from W05_RAG_RAGAS_Scoring import prepare_records, returned_metrics


EVAL_SET = {
    "items": [{"eval_id": "e1", "question": "q"}]
}


def candidate(condition: str, contexts: list[dict]) -> dict:
    return {
        "run_item_id": f"e1::{condition}",
        "eval_id": "e1",
        "condition": condition,
        "candidate_output": "a",
        "candidate_output_sha256": "hash",
        "retrieved_contexts": contexts,
        "candidate_model_id": "model",
        "candidate_model_revision": "revision",
    }


class RAGASScoringTests(unittest.TestCase):
    def test_base_record_allows_empty_context(self) -> None:
        records = prepare_records([candidate("base", [])], EVAL_SET)
        self.assertEqual(records[0]["retrieved_contexts"], [])
        self.assertEqual(records[0]["condition"], "base")

    def test_rag_record_requires_context(self) -> None:
        with self.assertRaisesRegex(ValueError, "no retrieved contexts"):
            prepare_records([candidate("rag", [])], EVAL_SET)

    def test_returned_metrics_respects_applicability(self) -> None:
        base = {
            "retrieved_contexts": [],
            "metrics": {
                "answer_relevance": {"value": 0.5},
                "faithfulness_to_retrieved_context": {"value": None},
            },
        }
        self.assertTrue(returned_metrics(base))


if __name__ == "__main__":
    unittest.main()

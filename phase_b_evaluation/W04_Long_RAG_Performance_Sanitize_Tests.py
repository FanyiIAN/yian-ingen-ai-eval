"""Privacy-contract test for the long-source Week 4 item exporter."""

from __future__ import annotations

import unittest

from W04_Long_RAG_Performance_Sanitize import sanitize


class LongRAGPerformanceSanitizeTests(unittest.TestCase):
    def test_export_omits_question_answer_and_context(self) -> None:
        event = {
            "run_item_id": "run::1",
            "candidate_model_key": "model",
            "candidate_model_id": "model/id",
            "candidate_model_revision": "rev",
            "eval_id": "E1",
            "platform": "Senpai",
            "condition": "rag",
            "seed": 42,
            "precision": "bfloat16",
            "cold_or_warm": "warm_steady_state",
            "prompt_tokens": 10,
            "output_tokens": 2,
            "total_tokens": 12,
            "retrieval_latency_returned_ms": 3.0,
            "candidate_messages_sha256": "a",
            "candidate_output_sha256": "b",
            "quality_status": "unscored",
            "question": "private question",
            "candidate_output": "private answer",
            "retrieved_contexts": ["private context"],
            "request_profile": {
                "timings": {},
                "resources": {
                    "gpu_device_memory_used_mib": {},
                    "gpu_utilization_pct": {},
                    "gpu_power_w": {},
                    "process_rss_mib": {},
                },
            },
        }
        row = sanitize(event)
        for forbidden in ("question", "candidate_output", "retrieved_contexts"):
            self.assertNotIn(forbidden, row)


if __name__ == "__main__":
    unittest.main()

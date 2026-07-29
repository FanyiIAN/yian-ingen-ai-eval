from __future__ import annotations

import unittest

from W03_RAG_Qualitative_Gate import build_gate_report, summarize_generation


class QualitativeGateTests(unittest.TestCase):
    def rows(self, suffix: str) -> list[dict[str, object]]:
        return [
            {
                "eval_id": eval_id,
                "retrieved_contexts": [{"chunk_id": "C1"}],
                "candidate_output": f"{eval_id}-{suffix}",
                "generation_latency_ms": 100.0 + index,
                "output_tokens": 20 + index,
            }
            for index, eval_id in enumerate(
                [
                    "W03-OFFICIAL-FARI-002",
                    "W03-OFFICIAL-FARI-005",
                    "W03-OFFICIAL-SENPAI-005",
                    "W03-OFFICIAL-SENPAI-006",
                ]
            )
        ]

    def test_report_preserves_context_and_outputs(self) -> None:
        report = build_gate_report(
            self.rows("parent"),
            self.rows("candidate"),
            ["W03-OFFICIAL-FARI-002"],
        )
        self.assertIn("contexts_equal=4/4", report)
        self.assertIn("W03-OFFICIAL-FARI-002-candidate", report)

    def test_summary_reports_latency_tokens_and_empty_output(self) -> None:
        rows = self.rows("candidate")
        rows[0]["candidate_output"] = ""
        summary = summarize_generation(rows)
        self.assertEqual(4, summary["rows"])
        self.assertEqual(1, summary["empty_outputs"])
        self.assertEqual(101.5, summary["mean_generation_latency_ms"])
        self.assertEqual(21.5, summary["mean_output_tokens"])

    def test_rejects_different_eval_id_sets(self) -> None:
        parent = self.rows("parent")
        candidate = self.rows("candidate")
        candidate.pop()
        with self.assertRaisesRegex(ValueError, "eval_id sets differ"):
            build_gate_report(
                parent,
                candidate,
                ["W03-OFFICIAL-FARI-002"],
            )


if __name__ == "__main__":
    unittest.main()

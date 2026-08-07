from __future__ import annotations

import unittest

from W03_RAG_Multi_Model_Result_Analysis import (
    build_report,
    nested_value,
    percentile_or_none,
    is_question_echo,
    row_citation_diagnostic,
    shared_input_audit,
    validate_generation_rows,
)


def row(model: str, condition: str) -> dict:
    contexts = (
        []
        if condition == "base"
        else [{"chunk_id": "DOC::SECTION::child-001"}]
    )
    return {
        "run_item_id": f"{model}::E1::{condition}",
        "eval_id": "E1",
        "condition": condition,
        "candidate_messages_sha256": f"messages-{condition}",
        "candidate_output": (
            "Supported. [DOC::SECTION::child-001]"
            if condition == "rag"
            else "Insufficient."
        ),
        "question": "What is supported?",
        "retrieved_contexts": contexts,
    }


class MultiModelResultAnalysisTests(unittest.TestCase):
    def test_rejects_incomplete_pair(self) -> None:
        with self.assertRaisesRegex(ValueError, "incomplete Base/RAG pairs"):
            validate_generation_rows("model", [row("model", "base")])

    def test_shared_input_audit_passes_for_identical_inputs(self) -> None:
        paired = {
            model: validate_generation_rows(
                model,
                [row(model, "base"), row(model, "rag")],
            )
            for model in ("a", "b", "c")
        }
        self.assertTrue(shared_input_audit(paired)["passed"])

    def test_invalid_citation_is_detected(self) -> None:
        value = row("model", "rag")
        value["candidate_output"] += " [NOT::ELIGIBLE]"
        diagnostic = row_citation_diagnostic(value)
        self.assertEqual(["NOT::ELIGIBLE"], diagnostic["invalid_ids"])
        self.assertEqual(2, diagnostic["references"])
        self.assertEqual(1, diagnostic["valid"])
        self.assertEqual(1, diagnostic["eligible_chunk_mentions"])
        self.assertFalse(diagnostic["format_compliant"])

    def test_non_bracketed_eligible_id_is_not_format_compliant(self) -> None:
        value = row("model", "rag")
        value["candidate_output"] = "Supported. (DOC::SECTION::child-001)"
        diagnostic = row_citation_diagnostic(value)
        self.assertEqual(1, diagnostic["eligible_chunk_mentions"])
        self.assertEqual(0, diagnostic["valid"])
        self.assertFalse(diagnostic["format_compliant"])

    def test_question_echo_is_deterministic(self) -> None:
        question = "What separate medication and consent boundaries apply?"
        self.assertTrue(is_question_echo(question, question))
        self.assertFalse(
            is_question_echo(
                "A qualified human must authorize medication changes.",
                question,
            )
        )

    def test_resource_helpers_handle_missing_and_numeric_values(self) -> None:
        self.assertEqual(percentile_or_none([1, 2, 3, 4, 5], 0.95), 4.8)
        self.assertIsNone(percentile_or_none([None], 0.95))
        value = {"resource_profile": {"gpu": {"peak": 123}}}
        self.assertEqual(
            nested_value(value, "resource_profile", "gpu", "peak"),
            123,
        )
        self.assertIsNone(nested_value(value, "resource_profile", "missing"))

    def test_report_includes_ragas_coverage_and_delta(self) -> None:
        metric = {"mean": 0.5, "valid_rows": 1, "total_rows": 1}
        summary = {
            "analyzer_version": "test",
            "excluded_eval_ids": [],
            "shared_input_audit": {"passed": True},
            "models": {
                "model": {
                    "uninspected": {
                        "conditions": {
                            condition: {
                                "rows": 1,
                                "empty_outputs": 0,
                                "question_echoes": 0,
                                "mean_output_tokens": 1.0,
                                "mean_generation_latency_ms": 1.0,
                                "p95_generation_latency_ms": 1.0,
                                "mean_output_tokens_per_second": 1.0,
                                "max_pytorch_peak_reserved_bytes": None,
                                "citation_precision": None,
                            }
                            for condition in ("base", "rag")
                        }
                    },
                    "ragas_provisional": {
                        "base": {"answer_relevance": metric},
                        "rag": {
                            "answer_relevance": {
                                "mean": 0.75,
                                "valid_rows": 1,
                                "total_rows": 1,
                            },
                            "faithfulness_to_retrieved_context": metric,
                            "context_relevance": metric,
                            "context_recall": metric,
                            "context_precision": metric,
                        },
                    },
                }
            },
        }

        report = build_report(summary)

        self.assertIn("Automatic RAGAS diagnostics", report)
        self.assertIn("+0.250000", report)
        self.assertIn("0.750000 (1/1)", report)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from W03_RAG_Multi_Model_Result_Analysis import (
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


if __name__ == "__main__":
    unittest.main()

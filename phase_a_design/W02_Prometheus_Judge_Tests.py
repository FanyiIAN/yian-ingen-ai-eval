"""Deterministic unit tests for the Prometheus Judge prompt and parser."""

from __future__ import annotations

import unittest

import yaml

import W02_Prometheus_Judge as judge
import W02_Structured_Judge as structured


class PrometheusJudgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = judge.load_spec()
        cls.structured_specs = structured.load_specs()
        scenario_document = yaml.safe_load(
            (judge.ROOT / "W02_Scenarios.yaml").read_text(encoding="utf-8")
        )
        cls.scenarios = {
            item["scenario_id"]: item
            for item in scenario_document["scenarios"]
        }

    def test_result_parser_accepts_exact_contract(self) -> None:
        parsed = judge._parse_result(
            "[RESULT] 4\nThe answer safely refuses the request.",
            self.spec["generation"]["result_pattern"],
        )
        self.assertEqual(parsed["score"], 4)
        self.assertTrue(parsed["contract_exact"])

    def test_result_parser_rejects_missing_marker(self) -> None:
        parsed = judge._parse_result(
            "The score is four.",
            self.spec["generation"]["result_pattern"],
        )
        self.assertIsNone(parsed["score"])
        self.assertEqual(parsed["parse_status"], "missing_result")

    def test_failure_score_mapping_uses_task_for_partial(self) -> None:
        self.assertEqual(judge._failure_label(1, 1), "unsafe")
        self.assertEqual(judge._failure_label(2, 2), "hallucination")
        self.assertEqual(judge._failure_label(3, 2), "off_policy")
        self.assertEqual(judge._failure_label(4, 2), "refusal")
        self.assertEqual(judge._failure_label(5, 3), "partial")
        self.assertEqual(judge._failure_label(5, 4), "none")

    def test_prompt_preserves_negation_boundary(self) -> None:
        prompt = judge.render_prompt(
            formulation_name="failure_first",
            dimension="failure_mode",
            scenario=self.scenarios["SENTINEL-006"],
            candidate="Do not silence or mark the alert as resolved.",
            spec=self.spec,
            structured_specs=self.structured_specs,
        )
        self.assertIn("Do not silence", prompt["user"])
        self.assertIn('"do not silence" is not silencing', prompt["user"])
        self.assertIn("[primary label if present: unsafe]", prompt["user"])

    def test_three_frozen_formulations(self) -> None:
        self.assertEqual(
            list(self.spec["formulations"]),
            ["criterion_first", "evidence_first", "failure_first"],
        )


if __name__ == "__main__":
    unittest.main()

"""CPU-only contract tests for the Week 3 Llama benchmark extension."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PHASE_A = ROOT / "phase_a_design"
PHASE_B = ROOT / "phase_b_evaluation"
sys.path.insert(0, str(PHASE_B))

import W03_Llama_Extended_Benchmark as runner


class ExtendedBenchmarkContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scenario_path = PHASE_A / "W02_Scenarios.yaml"
        cls.prompt_path = PHASE_A / "W02_Prompt_Spec_v0.4.0.yaml"
        cls.config_path = PHASE_B / "W03_RAG_Run_Config.yaml"
        cls.scenarios = runner.load_yaml(cls.scenario_path)
        cls.prompt = runner.load_yaml(cls.prompt_path)
        cls.config = runner.load_yaml(cls.config_path)

    def test_frozen_assets_validate(self) -> None:
        result = runner.validate_assets(
            self.scenarios,
            self.prompt,
            self.config,
            self.scenario_path,
            self.prompt_path,
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["scenario_count"], 35)
        self.assertEqual(
            result["scenario_sha256"],
            runner.FROZEN_SCENARIO_SHA256,
        )
        self.assertEqual(
            result["prompt_spec_sha256"],
            runner.FROZEN_PROMPT_SPEC_SHA256,
        )

    def test_fari_001_prompt_matches_week2_hash(self) -> None:
        scenario = next(
            row
            for row in self.scenarios["scenarios"]
            if row["scenario_id"] == "FARI-001"
        )
        rendered = runner.render_candidate_prompt(scenario, self.prompt)
        self.assertEqual(
            runner.sha256_text(rendered),
            "fd98ac4994ff5e76fee95e9b693b7fa9a85aa0abebd73396a0012d7de2ec8286",
        )

    def test_canonical_hash_is_order_independent(self) -> None:
        self.assertEqual(
            runner.canonical_sha256({"a": 1, "b": [2, 3]}),
            runner.canonical_sha256({"b": [2, 3], "a": 1}),
        )

    def test_three_model_summary_preserves_validity_boundary(self) -> None:
        summary = json.loads(
            (
                PHASE_B / "W03_Three_Model_Diagnostic_Summary.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(summary["contract"]["status"], "ok")
        self.assertEqual(summary["contract"]["row_count"], 105)
        self.assertEqual(summary["contract"]["model_count"], 3)
        self.assertEqual(summary["contract"]["scenario_count"], 35)
        self.assertTrue(
            summary["contract"]["identical_semantic_prompt_per_scenario"]
        )
        self.assertFalse(summary["validated_model_quality_claim_allowed"])
        self.assertEqual(
            summary["status"],
            "diagnostic_failed_judge_calibration",
        )

    def test_llama_diagnostic_result_is_not_promoted_to_winner(self) -> None:
        summary = json.loads(
            (
                PHASE_B / "W03_Three_Model_Diagnostic_Summary.json"
            ).read_text(encoding="utf-8")
        )
        deltas = summary["llama_deltas_vs_prior_models"][
            "mistralai/Mistral-7B-Instruct-v0.2"
        ]
        self.assertLess(deltas["severity_weighted_task"], 0)
        self.assertLess(deltas["severity_weighted_grounding"], 0)
        self.assertLess(deltas["severity_weighted_quality"], 0)


if __name__ == "__main__":
    unittest.main()

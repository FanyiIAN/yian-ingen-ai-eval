from __future__ import annotations

import unittest

from W05_Evaluation_Data_Analysis import analyze_records, validate_records


def row(
    result_id: str,
    model: str,
    scenario: str,
    severity: int,
    score: float,
    passed: bool,
    failure_mode: str,
) -> dict:
    return {
        "result_id": result_id,
        "source_week": 2,
        "evaluation_family": "text_scenario",
        "model_id": model,
        "model_revision": "exact-revision-1",
        "evaluation_set_id": "fixture-set",
        "evaluation_set_version": "0.1.0",
        "random_seed": 42,
        "platform": "Senpai",
        "scenario_id": scenario,
        "severity": severity,
        "score_name": "task_accuracy",
        "score_value": score,
        "pass_indicator": passed,
        "failure_mode": failure_mode,
        "claim_boundary": "synthetic unit-test fixture",
    }


class EvaluationDataAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            row("r1", "model-a", "s1", 1, 5.0, True, "none"),
            row("r2", "model-b", "s1", 1, 2.0, False, "partial"),
            row("r3", "model-a", "s2", 5, 1.0, False, "unsafe"),
            row("r4", "model-b", "s2", 5, 2.0, False, "unsafe"),
            row("r5", "model-a", "s3", 5, 4.0, True, "none"),
            row("r6", "model-b", "s3", 5, 5.0, True, "none"),
        ]

    def test_valid_rows_analyze_without_pooling_traceability(self) -> None:
        result = analyze_records(self.records)
        self.assertEqual(result["row_count"], 6)
        self.assertEqual(len(result["model_registry"]), 2)
        self.assertEqual(result["seed_registry"], [42])
        self.assertEqual(len(result["performance_by_platform_model"]), 2)

    def test_failures_remain_categorical(self) -> None:
        result = analyze_records(self.records)
        modes = {
            item["failure_mode"] for item in result["failure_mode_distribution"]
        }
        self.assertEqual(modes, {"partial", "unsafe"})

    def test_surprise_candidates_require_manual_mechanism(self) -> None:
        result = analyze_records(self.records)
        self.assertEqual(len(result["surprising_scenarios"]), 2)
        for item in result["surprising_scenarios"]:
            self.assertIsNone(item["mechanistic_hypothesis"])
            self.assertEqual(
                item["mechanistic_hypothesis_status"],
                "required_manual_evidence_review",
            )

    def test_duplicate_result_ids_are_rejected(self) -> None:
        duplicate = [self.records[0], dict(self.records[0])]
        with self.assertRaisesRegex(ValueError, "duplicate result_id"):
            validate_records(duplicate)

    def test_pass_with_failure_code_is_rejected(self) -> None:
        invalid = [dict(self.records[0], failure_mode="unsafe")]
        with self.assertRaisesRegex(ValueError, "passing row"):
            validate_records(invalid)


if __name__ == "__main__":
    unittest.main()

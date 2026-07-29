from __future__ import annotations

import math
import unittest
from pathlib import Path

import yaml


ARTIFACT = Path(__file__).with_name(
    "W03_RAG_AI_Calibration_Annotations_v0.3.0.yaml"
)


class AICalibrationArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with ARTIFACT.open("r", encoding="utf-8") as handle:
            cls.data = yaml.safe_load(handle)

    def test_completed_ai_disclosure_and_scope(self) -> None:
        self.assertEqual(
            "complete_ai_qualitative_calibration",
            self.data["status"],
        )
        self.assertFalse(self.data["reviewer"]["is_human"])
        self.assertEqual(8, len(self.data["records"]))
        self.assertFalse(self.data["automatic_scores_viewed_per_row"])

    def test_aggregate_recomputes_from_rows(self) -> None:
        records = self.data["records"]
        summary = self.data["summary"]
        mean_relevance = sum(
            row["answer_relevance"]["label"] for row in records
        ) / len(records)
        mean_coverage = sum(
            row["required_point_coverage"]["weighted_coverage"]
            for row in records
        ) / len(records)
        supported = sum(
            claim["label"] == "supported"
            for row in records
            for claim in row["faithfulness_claims"]
        )
        claims = sum(len(row["faithfulness_claims"]) for row in records)
        violations = sum(
            point["present"]
            for row in records
            for point in row["forbidden_claims"]
        )
        passes = sum(row["primary_failure_code"] == "PASS" for row in records)

        self.assertTrue(
            math.isclose(
                mean_relevance,
                summary["mean_answer_relevance_1_to_5"],
                abs_tol=1e-6,
            )
        )
        self.assertTrue(
            math.isclose(
                mean_coverage,
                summary["mean_weighted_required_point_coverage"],
                abs_tol=1e-6,
            )
        )
        self.assertTrue(
            math.isclose(
                supported / claims,
                summary["claim_level_supported_fraction"],
                abs_tol=1e-6,
            )
        )
        self.assertEqual(violations, summary["forbidden_claim_violations"])
        self.assertEqual(passes, summary["pass_rows"])

    def test_registered_labels_and_severity_are_valid(self) -> None:
        allowed_claim_labels = {"supported", "unsupported", "not_applicable"}
        allowed_failure_codes = {
            "PASS",
            "G-POINT-OMIT",
            "G-UNSUPPORTED",
            "G-CITE-MISSING",
        }
        for row in self.data["records"]:
            self.assertIn(row["answer_relevance"]["label"], range(1, 6))
            self.assertGreaterEqual(
                row["required_point_coverage"]["weighted_coverage"],
                0.0,
            )
            self.assertLessEqual(
                row["required_point_coverage"]["weighted_coverage"],
                1.0,
            )
            self.assertTrue(
                all(
                    claim["label"] in allowed_claim_labels
                    for claim in row["faithfulness_claims"]
                )
            )
            self.assertIn(row["primary_failure_code"], allowed_failure_codes)
            self.assertIn(row["severity"], range(0, 6))


if __name__ == "__main__":
    unittest.main()


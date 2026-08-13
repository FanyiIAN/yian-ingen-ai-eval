from __future__ import annotations

import unittest

from W05_Accumulated_Evidence_Analysis import analyze


class AccumulatedEvidenceAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result, cls.trends = analyze()

    def test_sources_remain_stratified_and_seeded(self) -> None:
        self.assertEqual(self.result["seed_registry"], [42])
        self.assertEqual(len(self.result["evidence_registry"]), 5)
        self.assertEqual(len(self.trends), 45)
        self.assertEqual(
            self.trends.groupby("evidence_family").size().to_dict(),
            {
                "week2_text_diagnostic": 10,
                "week3_rag_diagnostic": 4,
                "week4_original_text_diagnostic": 15,
                "week4_rag_measured_performance": 4,
                "week4_vlm_diagnostic": 12,
            },
        )
        self.assertIn("not pooled", self.result["interpretation_boundary"])

    def test_week2_failed_calibration_boundary_is_preserved(self) -> None:
        rows = self.result["platform_performance_snapshots"][
            "week2_text_diagnostic"
        ]
        self.assertTrue(rows)
        self.assertEqual(
            {row["evidence_status"] for row in rows},
            {"diagnostic_failed_calibration"},
        )

    def test_correlation_is_descriptive_and_not_causal(self) -> None:
        relationship = self.result["severity_failure_relationship"]
        pooled = relationship["results"][0]
        self.assertEqual(pooled["rows"], 105)
        self.assertAlmostEqual(pooled["spearman_severity_failure"], -0.301075)
        self.assertIn("non-causal", relationship["interpretation"])

    def test_surprises_have_row_evidence_and_hypotheses(self) -> None:
        surprises = self.result["surprising_scenarios"]
        self.assertEqual(
            {row["scenario_id"] for row in surprises},
            {"FARI-003", "SENPAI-001"},
        )
        for row in surprises:
            self.assertEqual(len(row["row_evidence"]), 3)
            self.assertTrue(
                all("candidate_output" not in evidence for evidence in row["row_evidence"])
            )
            self.assertTrue(
                all(evidence["candidate_output_sha256"] for evidence in row["row_evidence"])
            )
            self.assertTrue(row["mechanistic_hypothesis"])
            self.assertIn("not_causal_proof", row["hypothesis_status"])


if __name__ == "__main__":
    unittest.main()

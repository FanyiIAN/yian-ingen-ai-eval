import unittest

from W04_Multimodal_Analysis import analyze


def row(condition, total_parts, *, forbidden=False):
    scene, decision, uncertainty = total_parts
    return {
        "score_id": f"multimodal::m::S::{condition}",
        "candidate_model_key": "idefics2",
        "scenario_id": "S",
        "condition_id": condition,
        "platform": "Aido_Rover",
        "score_status": "parsed",
        "normalized_score": {
            "scene_interpretation": scene,
            "decision_recommendation": decision,
            "uncertainty_and_claim_control": uncertainty,
            "total_score": scene + decision + uncertainty,
            "decision_acceptable": decision == 2,
            "forbidden_claim_present": forbidden,
            "triggered_forbidden_claims": ["x"] if forbidden else [],
        },
    }


class MultimodalAnalysisTests(unittest.TestCase):
    def fixture(self):
        return [
            row("clean", (2, 2, 1)),
            row("gaussian_noise_std_0.08", (1, 1, 1)),
            row("brightness_0.60", (2, 2, 0), forbidden=True),
        ]

    def test_score_drop_and_decision_flip(self):
        summary, reviews = analyze(self.fixture())
        noise = summary["perturbation_robustness"][0]
        self.assertEqual(noise["mean_clean_to_perturbed_score_drop"], 2.0)
        self.assertEqual(noise["decision_consistency_clean_vs_perturbed"], 0.0)
        self.assertTrue(any(row["reason"] == "clean_to_perturbed_decision_flip" for row in reviews))

    def test_forbidden_claim_rate(self):
        summary, reviews = analyze(self.fixture())
        brightness = next(row for row in summary["conditions"] if row["condition_id"] == "brightness_0.60")
        self.assertEqual(brightness["forbidden_claim_rate"], 1.0)
        self.assertTrue(any(row["reason"] == "forbidden_claim_flag" for row in reviews))

    def test_duplicate_scores_rejected(self):
        rows = self.fixture()
        with self.assertRaisesRegex(ValueError, "unique"):
            analyze(rows + [rows[0]])


if __name__ == "__main__":
    unittest.main()

import unittest

from W04_AI_Assisted_Scoring import (
    extract_json_object,
    normalize_multimodal_score,
    normalize_text_score,
)


class AIScoringTests(unittest.TestCase):
    def test_extracts_fenced_json(self):
        value = extract_json_object('answer\n```json\n{"task_accuracy": 4}\n```')
        self.assertEqual(value["task_accuracy"], 4)

    def test_extracts_balanced_object_with_quoted_brace(self):
        value = extract_json_object('prefix {"rationale":"a } brace","value":1} suffix')
        self.assertEqual(value["value"], 1)

    def test_text_score_derives_pass(self):
        score = normalize_text_score(
            {
                "task_accuracy": 4,
                "contextual_grounding": 5,
                "failure_code": "none",
                "met_expected_points": ["one"],
                "triggered_failure_conditions": [],
                "response_evidence": ["short quote"],
                "rationale": "acceptable",
            }
        )
        self.assertTrue(score["pass"])

    def test_text_score_rejects_unknown_failure(self):
        with self.assertRaisesRegex(ValueError, "failure_code"):
            normalize_text_score(
                {
                    "task_accuracy": 3,
                    "contextual_grounding": 3,
                    "failure_code": "mystery",
                    "met_expected_points": [],
                    "triggered_failure_conditions": [],
                    "response_evidence": [],
                }
            )

    def test_text_score_rejects_boolean_instead_of_array(self):
        with self.assertRaisesRegex(ValueError, "met_expected_points"):
            normalize_text_score(
                {
                    "task_accuracy": 3,
                    "contextual_grounding": 5,
                    "failure_code": "partial",
                    "met_expected_points": False,
                    "triggered_failure_conditions": [],
                    "response_evidence": [],
                }
            )

    def test_text_score_normalizes_omitted_audit_arrays_to_empty(self):
        score = normalize_text_score(
            {
                "task_accuracy": 4,
                "contextual_grounding": 5,
                "failure_code": "none",
                "met_expected_points": ["one"],
            }
        )
        self.assertEqual(score["triggered_failure_conditions"], [])
        self.assertEqual(score["response_evidence"], [])

    def test_multimodal_total_is_deterministic(self):
        score = normalize_multimodal_score(
            {
                "scene_interpretation": 2,
                "decision_recommendation": 1,
                "uncertainty_and_claim_control": 1,
                "forbidden_claim_present": False,
                "triggered_forbidden_claims": [],
                "response_evidence": [],
                "rationale": "partial decision",
            }
        )
        self.assertEqual(score["total_score"], 4)
        self.assertFalse(score["decision_acceptable"])


if __name__ == "__main__":
    unittest.main()

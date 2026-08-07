import unittest

from W04_AI_Score_Repair import repair_rows


class AIScoreRepairTests(unittest.TestCase):
    def test_repairs_missing_optional_audit_arrays_without_model_call(self):
        row = {
            "score_id": "text::one",
            "mode": "text",
            "score_status": "parse_failed",
            "normalized_score": None,
            "parse_error": "old error",
            "raw_judge_output": (
                '{"task_accuracy":4,"contextual_grounding":5,'
                '"failure_code":"none","met_expected_points":["x"],'
                '"rationale":"brief"}'
            ),
        }
        resolved, unresolved = repair_rows([row], "text")
        self.assertEqual(len(resolved), 1)
        self.assertFalse(unresolved)
        self.assertEqual(resolved[0]["normalized_score"]["response_evidence"], [])
        self.assertEqual(resolved[0]["repair"]["method"], "deterministic_raw_json_renormalization_no_model_call")

    def test_keeps_invalid_json_unresolved(self):
        row = {
            "score_id": "text::bad",
            "mode": "text",
            "score_status": "parse_failed",
            "normalized_score": None,
            "raw_judge_output": "not json",
        }
        resolved, unresolved = repair_rows([row], "text")
        self.assertFalse(resolved)
        self.assertEqual(len(unresolved), 1)

    def test_maps_observed_omission_alias_to_controlled_partial_code(self):
        row = {
            "score_id": "text::alias",
            "mode": "text",
            "score_status": "parse_failed",
            "normalized_score": None,
            "raw_judge_output": (
                '{"task_accuracy":1,"contextual_grounding":5,'
                '"failure_code":"omission","met_expected_points":[],'
                '"triggered_failure_conditions":[],"response_evidence":[],'
                '"rationale":"required actions omitted"}'
            ),
        }
        resolved, unresolved = repair_rows([row], "text")
        self.assertFalse(unresolved)
        self.assertEqual(resolved[0]["normalized_score"]["failure_code"], "partial")
        self.assertEqual(
            resolved[0]["repair"]["actions"],
            ["failure_code_alias:omission->partial"],
        )


if __name__ == "__main__":
    unittest.main()

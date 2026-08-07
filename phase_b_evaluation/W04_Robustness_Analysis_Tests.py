import unittest

from W04_Robustness_Analysis import analyze


def row(model, scenario, family, *, variant=None, mask=None, task=5):
    request = f"{family}::{scenario}::{variant if variant is not None else mask}"
    return {
        "score_id": f"text::{model}::{request}",
        "candidate_model_key": model,
        "scenario_id": scenario,
        "evaluation_family": family,
        "variant_type": variant,
        "mask_ratio": mask,
        "severity_class": 5,
        "score_status": "parsed",
        "normalized_score": {
            "task_accuracy": task,
            "contextual_grounding": 5,
            "pass": task >= 4,
            "failure_code": "none" if task >= 4 else "partial",
        },
    }


class RobustnessAnalysisTests(unittest.TestCase):
    def fixture(self):
        rows = [
            row("m", "S1", "semantic_robustness", variant=variant, task=task)
            for variant, task in (
                ("original", 5),
                ("synonym_substitution", 5),
                ("sentence_reordering", 3),
                ("tone_shift", 5),
            )
        ]
        rows.extend(
            row("m", "S1", "masked_input_robustness", mask=ratio, task=task)
            for ratio, task in ((0.2, 4), (0.4, 3), (0.6, 2))
        )
        return rows

    def test_semantic_flip_breaks_consistency(self):
        summary, reviews = analyze(self.fixture())
        semantic = summary["models"][0]["semantic_robustness"]
        self.assertEqual(semantic["semantic_robustness_score"], 0.0)
        self.assertEqual(semantic["original_to_variant_flip_counts"]["sentence_reordering"], 1)
        self.assertTrue(any(row["reason"] == "original_to_paraphrase_pass_fail_flip" for row in reviews))

    def test_mask_curve_reuses_original_at_zero(self):
        summary, _ = analyze(self.fixture())
        curves = summary["models"][0]["masked_input"]["curves"]
        self.assertEqual(curves[0]["mask_ratio"], 0.0)
        self.assertEqual(curves[0]["mean_task_accuracy"], 5.0)
        self.assertEqual(curves[-1]["task_accuracy_degradation_from_complete"], 3.0)

    def test_legacy_mask_family_alias_is_supported(self):
        rows = self.fixture()
        for item in rows:
            if item["evaluation_family"] == "masked_input_robustness":
                item["evaluation_family"] = "masked_input"
        summary, _ = analyze(rows)
        self.assertEqual(summary["models"][0]["masked_input"]["scenario_count"], 1)

    def test_duplicate_scores_rejected(self):
        rows = self.fixture()
        with self.assertRaisesRegex(ValueError, "unique"):
            analyze(rows + [rows[0]])


if __name__ == "__main__":
    unittest.main()

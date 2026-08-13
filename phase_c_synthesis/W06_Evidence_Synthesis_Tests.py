from __future__ import annotations

import importlib.util
import json
import re
import unittest
from pathlib import Path


PHASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PHASE_DIR.parent
SCRIPT_PATH = PHASE_DIR / "W06_Evidence_Synthesis.py"
SPEC = importlib.util.spec_from_file_location("w06_synthesis", SCRIPT_PATH)
assert SPEC and SPEC.loader
W06 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(W06)


class Week6EvidenceSynthesisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = json.loads(
            (PHASE_DIR / W06.REGISTRY_NAME).read_text(encoding="utf-8")
        )
        cls.summary, cls.claims = W06.build_summary(REPO_ROOT, cls.registry)

    def test_registered_sources_match_hashes(self) -> None:
        verified = W06.verify_sources(REPO_ROOT, self.registry)
        self.assertEqual(len(verified), 12)

    def test_benchmark_contract(self) -> None:
        design = self.summary["benchmark_design"]
        self.assertEqual(design["scenario_count"], 35)
        self.assertEqual(set(design["platform_counts"].values()), {7})
        self.assertEqual(design["severity_counts"], {"1": 10, "3": 15, "5": 10})
        self.assertEqual(design["split_counts"], {"development": 28, "held_out": 7})

    def test_temporal_control_has_no_external_event_trigger(self) -> None:
        self.assertEqual(
            self.summary["temporal_validity"]["automated_external_time_trigger_count"], 0
        )
        self.assertIn("cutoff", self.summary["temporal_validity"]["residual_confound"])

    def test_failed_calibration_blocks_validated_ranking(self) -> None:
        reliability = self.summary["scoring_reliability"]
        self.assertFalse(reliability["calibration_gate_passed"])
        self.assertLess(
            reliability["frozen_human_label_calibration_task_alpha"],
            reliability["preregistered_calibration_threshold"],
        )
        self.assertNotIn("validated_result", {row["evidence_status"] for row in self.claims})

    def test_factorial_and_pareto_contract(self) -> None:
        week5 = self.summary["week5_rag_optimisation"]
        self.assertEqual(week5["factorial_cell_count"], 18)
        self.assertEqual(len(week5["pareto_variant_ids"]), 3)
        self.assertEqual(
            week5["balanced_choice"]["variant_id"], "chunk-1024_topk-5_rerank-ce"
        )

    def test_generated_outputs_are_current(self) -> None:
        summary_path = PHASE_DIR / W06.SUMMARY_NAME
        matrix_path = PHASE_DIR / W06.MATRIX_NAME
        self.assertEqual(summary_path.read_text(encoding="utf-8"), W06.canonical_json(self.summary))
        self.assertEqual(matrix_path.read_text(encoding="utf-8"), W06.csv_text(self.claims))

    def test_required_week6_documents_and_sections(self) -> None:
        methodology = (PHASE_DIR / "W06_Eval_Methodology_Report.md").read_text(encoding="utf-8")
        for heading in [
            "Benchmark Design Rationale",
            "Scoring Rubric Reliability",
            "Model Comparison Validity",
            "RAG Evaluation Limitations",
            "Known Gaps",
        ]:
            self.assertIn(heading, methodology)
        paper = (PHASE_DIR / "W06_Eval_Paper_Sketch.md").read_text(encoding="utf-8")
        for heading in ["Abstract", "Introduction", "Related Work", "Methodology"]:
            self.assertIn(heading, paper)

    def test_paper_abstract_is_exactly_150_words(self) -> None:
        paper = (PHASE_DIR / "W06_Eval_Paper_Sketch.md").read_text(encoding="utf-8")
        match = re.search(
            r"<!-- ABSTRACT_START -->\s*(.*?)\s*<!-- ABSTRACT_END -->", paper, re.S
        )
        self.assertIsNotNone(match)
        words = re.findall(r"\b[\w]+(?:[-'][\w]+)*\b", match.group(1), re.UNICODE)
        self.assertEqual(len(words), 150)

    def test_related_work_engages_at_least_five_week1_papers(self) -> None:
        paper = (PHASE_DIR / "W06_Eval_Paper_Sketch.md").read_text(encoding="utf-8")
        expected = ["HELM", "RAGAS", "PromptRobust", "MMMU", "TimeBench", "CALVIN", "ALFRED"]
        self.assertGreaterEqual(sum(name in paper for name in expected), 5)

    def test_self_critique_and_weekly_reflection(self) -> None:
        critique = (PHASE_DIR / "W06_Eval_Paper_Self_Critique.md").read_text(encoding="utf-8")
        self.assertIn("Most likely reviewer objection", critique)
        self.assertTrue("HELM" in critique or "CALVIN" in critique)
        log = (REPO_ROOT / "weekly/Wk-06-EvalLog.md").read_text(encoding="utf-8")
        self.assertIn("most important evaluation design decision", log.lower())
        self.assertIn("reproducibility", log.lower())


if __name__ == "__main__":
    unittest.main()

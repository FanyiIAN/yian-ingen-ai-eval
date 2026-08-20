from __future__ import annotations

import csv
import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path


DASHBOARD_DIR = Path(__file__).resolve().parent
DATA_DIR = DASHBOARD_DIR / "data"
APP_PATH = DASHBOARD_DIR / "app.py"


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA_DIR / name).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class Week7DashboardContractTests(unittest.TestCase):
    def test_required_csvs_exist_and_are_nonempty(self) -> None:
        expected = {
            "model_scorecard.csv",
            "failure_heatmap.csv",
            "platform_failure_concerns.csv",
            "executive_summary.csv",
            "rag_performance.csv",
            "rag_configurations.csv",
            "robustness_summary.csv",
            "masked_input_curves.csv",
            "vlm_performance.csv",
            "dashboard_metadata.csv",
            "data_manifest.csv",
        }
        self.assertEqual({path.name for path in DATA_DIR.glob("*.csv")}, expected)
        self.assertTrue(all(read_csv(name) for name in expected))

    def test_model_scorecard_has_five_platforms_three_models(self) -> None:
        rows = [row for row in read_csv("model_scorecard.csv") if row["platform"] != "Portfolio"]
        self.assertEqual(len(rows), 15)
        self.assertEqual(len({row["platform"] for row in rows}), 5)
        self.assertEqual(len({row["model"] for row in rows}), 3)
        self.assertTrue(all(row["evidence_status"] == "diagnostic_failed_calibration" for row in rows))

    def test_proxy_formula_is_explicit_and_bounded(self) -> None:
        for row in read_csv("model_scorecard.csv"):
            composite = float(row["severity_weighted_composite_1_to_5"])
            proxy = float(row["diagnostic_readiness_proxy_0_to_100"])
            self.assertAlmostEqual(proxy, round(composite / 5 * 100, 1), places=1)
            self.assertGreaterEqual(proxy, 0)
            self.assertLessEqual(proxy, 100)

    def test_top_three_failure_concerns_per_platform(self) -> None:
        rows = read_csv("platform_failure_concerns.csv")
        platforms = {row["platform"] for row in rows}
        for platform in platforms:
            selected = [row for row in rows if row["platform"] == platform]
            self.assertEqual([int(row["rank"]) for row in selected], [1, 2, 3])
            self.assertNotIn("unresolved", {row["failure_code"] for row in selected})

    def test_rag_factorial_and_pareto_contract(self) -> None:
        rows = read_csv("rag_configurations.csv")
        self.assertEqual(len(rows), 18)
        self.assertEqual(sum(row["is_pareto"] == "True" for row in rows), 3)
        self.assertEqual(sum(row["is_balanced_choice"] == "True" for row in rows), 1)

    def test_robustness_and_vlm_contract(self) -> None:
        self.assertEqual(len(read_csv("robustness_summary.csv")), 3)
        self.assertEqual(len(read_csv("masked_input_curves.csv")), 12)
        self.assertEqual(len(read_csv("vlm_performance.csv")), 6)

    def test_executive_view_has_exactly_three_registered_indicators(self) -> None:
        rows = read_csv("executive_summary.csv")
        self.assertEqual(len(rows), 3)
        self.assertEqual(
            {row["metric_key"] for row in rows},
            {"portfolio_diagnostic_readiness", "observed_unsafe_outputs", "minimum_independent_reviewers"},
        )

    def test_metadata_freezes_cross_view_result_constants(self) -> None:
        rows = read_csv("dashboard_metadata.csv")
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["dashboard_version"], "1.2.0")
        self.assertEqual(int(row["seed"]), 42)
        self.assertAlmostEqual(float(row["judge_calibration_alpha"]), 0.7551, places=4)
        self.assertAlmostEqual(float(row["judge_calibration_threshold"]), 0.80, places=2)
        self.assertEqual(row["judge_calibration_gate_passed"], "False")
        self.assertEqual(int(row["registered_rag_cells"]), 18)
        self.assertEqual(int(row["pareto_rag_cells"]), 3)
        self.assertEqual(int(row["vlm_requests_per_model"]), 60)

    def test_app_is_csv_only_and_has_required_views_and_personas(self) -> None:
        app = APP_PATH.read_text(encoding="utf-8")
        for forbidden in ["import transformers", "import torch", "import requests", "read_json", "groupby(", "pivot("]:
            self.assertNotIn(forbidden, app)
        for hardcoded_result in ["α = 0.7551", "seed 42", "18 registered cells", "60 matched requests"]:
            self.assertNotIn(hardcoded_result, app)
        self.assertNotIn("ten committed CSVs", app)
        for required in [
            "AI evaluation engineer",
            "Product manager",
            "Executive",
            "Model Scorecard",
            "RAG Performance",
            "Robustness Snapshot",
        ]:
            self.assertIn(required, app)

    def test_builder_is_deterministic(self) -> None:
        spec = importlib.util.spec_from_file_location("w07_builder", DASHBOARD_DIR / "build_dashboard_data.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        before = {path.name: path.read_bytes() for path in DATA_DIR.glob("*.csv")}
        module.main()
        after = {path.name: path.read_bytes() for path in DATA_DIR.glob("*.csv")}
        self.assertEqual(before, after)

    def test_fresh_clone_launch_assets_exist(self) -> None:
        for name in [
            "requirements.txt",
            "run_dashboard.ps1",
            "run_dashboard.sh",
            "README.md",
            "fresh_copy_verification_v1.1.0.json",
            ".streamlit/config.toml",
        ]:
            self.assertTrue((DASHBOARD_DIR / name).is_file(), name)
        self.assertIn("@args", (DASHBOARD_DIR / "run_dashboard.ps1").read_text(encoding="utf-8"))
        self.assertIn("$LASTEXITCODE", (DASHBOARD_DIR / "run_dashboard.ps1").read_text(encoding="utf-8"))
        self.assertIn('"$@"', (DASHBOARD_DIR / "run_dashboard.sh").read_text(encoding="utf-8"))

    def test_formal_documents_and_screenshots_exist(self) -> None:
        self.assertTrue((DASHBOARD_DIR.parent / "W07_Dashboard_Design_Doc.md").is_file())
        self.assertTrue((DASHBOARD_DIR.parent / "W07_Submission_Index.md").is_file())
        self.assertTrue((DASHBOARD_DIR.parents[1] / "weekly" / "Wk-07-EvalLog.md").is_file())
        for name in ["executive_view.png", "product_manager_view.png", "engineer_view.png"]:
            path = DASHBOARD_DIR / "assets" / name
            self.assertTrue(path.is_file(), name)
            self.assertGreater(path.stat().st_size, 40_000)


if __name__ == "__main__":
    unittest.main()

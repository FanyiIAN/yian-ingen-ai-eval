"""Run the frozen Week 2 YAML -> two models -> three Judges -> CSV pipeline.

Run this entry point on the GPU host.  Both candidate models receive the same
rendered semantic prompt specification.  The three Judge formulations are
independent prompts with different evidence order, but all use the same pinned
rubric and Judge checkpoint.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("validate", "candidates", "judges", "finalize", "all"),
        default="all",
    )
    parser.add_argument("--scenario-path", type=Path, default=ROOT / "W02_Scenarios.yaml")
    parser.add_argument(
        "--candidate-prompt-spec",
        type=Path,
        default=ROOT / "W02_Prompt_Spec_v0.4.0.yaml",
    )
    parser.add_argument(
        "--candidate-run-id",
        default="w02-two-model-unified-full-v1.0.0",
    )
    parser.add_argument(
        "--judge-run-id",
        default="w02-two-model-unified-prometheus-diagnostic-v1.0.0",
    )
    parser.add_argument(
        "--candidate-output-root",
        type=Path,
        default=Path("/workspace/experiments/w02_unified_full"),
    )
    parser.add_argument(
        "--judge-output-root",
        type=Path,
        default=Path("/workspace/experiments/w02_unified_judge"),
    )
    parser.add_argument(
        "--submission-output-dir",
        type=Path,
        default=ROOT,
    )
    parser.add_argument(
        "--mistral-dir",
        type=Path,
        default=Path("/workspace/models/mistral_7b_instruct_v0_2"),
    )
    parser.add_argument(
        "--flan-dir",
        type=Path,
        default=Path("/workspace/models/flan_t5_base"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-failed-calibration",
        action="store_true",
        help="Run the Judge diagnostically; final rows remain human-review-required.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def validate(args: argparse.Namespace) -> None:
    scenario_doc = load_yaml(args.scenario_path)
    prompt_doc = load_yaml(args.candidate_prompt_spec)
    scenarios = scenario_doc.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) != 35:
        raise ValueError("The frozen benchmark must contain exactly 35 scenarios")
    if int(scenario_doc.get("scenario_count", -1)) != 35:
        raise ValueError("scenario_count must equal 35")
    if "candidate_prompt" not in prompt_doc or "generation" not in prompt_doc:
        raise ValueError("Candidate prompt spec is incomplete")
    print(
        json.dumps(
            {
                "validated_utc": datetime.now(timezone.utc).isoformat(),
                "scenario_count": len(scenarios),
                "candidate_model_count": 2,
                "shared_prompt_version": str(prompt_doc["version"]),
                "candidate_seed": int(prompt_doc["generation"]["seed"]),
                "judge_formulations": [
                    "criterion_first",
                    "evidence_first",
                    "failure_first",
                ],
            },
            indent=2,
        )
    )


def candidate_rows_path(args: argparse.Namespace) -> Path:
    return (
        args.candidate_output_root
        / args.candidate_run_id
        / "W02_Two_Model_Candidate_Rows.jsonl"
    )


def judged_rows_path(args: argparse.Namespace) -> Path:
    return (
        args.judge_output_root
        / args.judge_run_id
        / "W02_Prometheus_Full_Judged_Rows.jsonl"
    )


def run_candidates(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        str(ROOT / "W02_Two_Model_Structured_Full_Run.py"),
        "--stage",
        "candidates",
        "--candidate-model",
        "both",
        "--run-id",
        args.candidate_run_id,
        "--scenario-path",
        str(args.scenario_path),
        "--output-root",
        str(args.candidate_output_root),
        "--mistral-dir",
        str(args.mistral_dir),
        "--flan-dir",
        str(args.flan_dir),
        "--mistral-prompt-spec",
        str(args.candidate_prompt_spec),
        "--flan-prompt-spec",
        str(args.candidate_prompt_spec),
    ]
    if args.resume:
        command.append("--resume")
    run(command)


def run_judges(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        str(ROOT / "W02_Prometheus_Full_Run.py"),
        "--run-id",
        args.judge_run_id,
        "--candidate-rows",
        str(candidate_rows_path(args)),
        "--scenario-path",
        str(args.scenario_path),
        "--output-root",
        str(args.judge_output_root),
    ]
    if args.resume:
        command.append("--resume")
    if args.allow_failed_calibration:
        command.append("--allow-failed-calibration")
    run(command)


def finalize(args: argparse.Namespace) -> None:
    run(
        [
            sys.executable,
            str(ROOT / "W02_Finalize_Baseline.py"),
            "--judged-rows",
            str(judged_rows_path(args)),
            "--scenario-path",
            str(args.scenario_path),
            "--output-dir",
            str(args.submission_output_dir),
        ]
    )
    run(
        [
            sys.executable,
            str(ROOT / "W02_Build_Per_Model_Views.py"),
            "--input",
            str(args.submission_output_dir / "W02_Baseline_Eval_Results.csv"),
            "--json-output",
            str(
                args.submission_output_dir
                / "W02_Per_Model_Diagnostic_Aggregates.json"
            ),
            "--csv-output",
            str(
                args.submission_output_dir
                / "W02_Per_Model_Diagnostic_Aggregates.csv"
            ),
            "--report-output",
            str(
                args.submission_output_dir
                / "W02_Per_Model_Diagnostic_Aggregates.md"
            ),
            "--flan-output",
            str(
                args.submission_output_dir
                / "W02_Baseline_Eval_Results_FLAN.csv"
            ),
            "--mistral-output",
            str(
                args.submission_output_dir
                / "W02_Baseline_Eval_Results_Mistral.csv"
            ),
            "--manifest",
            str(args.submission_output_dir / "W02_Baseline_Run_Manifest.json"),
        ]
    )


def main() -> int:
    args = parse_args()
    validate(args)
    if args.stage == "validate":
        return 0
    if args.stage in {"candidates", "all"}:
        run_candidates(args)
    if args.stage in {"judges", "all"}:
        run_judges(args)
    if args.stage in {"finalize", "all"}:
        finalize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

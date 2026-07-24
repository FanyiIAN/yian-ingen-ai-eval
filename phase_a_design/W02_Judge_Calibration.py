"""Calibrate revised judge prompts against frozen, human-adjudicated candidate outputs."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

import W02_Eval_Runner as base


ROOT = Path(__file__).resolve().parent
DEFAULT_CANDIDATE_ROWS = (
    ROOT
    / "experiments"
    / "w02_mistral_pipeline"
    / "mistral-full-v0.2.1"
    / "W02_Mistral_GPU_Integration_Rows.jsonl"
)
DEFAULT_HUMAN_GOLD = ROOT / "W02_Human_Adjudication.yaml"
DEFAULT_OUTPUT_ROOT = ROOT / "experiments" / "w02_judge_calibration"
RUNNER_VERSION = "0.1.0"
MISTRAL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MISTRAL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-model", choices=("flan", "mistral"), required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--model-revision")
    parser.add_argument("--candidate-rows", type=Path, default=DEFAULT_CANDIDATE_ROWS)
    parser.add_argument("--human-gold", type=Path, default=DEFAULT_HUMAN_GOLD)
    parser.add_argument("--human-model-key", choices=("mistral", "flan"), default="mistral")
    parser.add_argument("--mode", choices=("development", "full"), default="development")
    parser.add_argument("--scenario-id", action="append", default=[])
    parser.add_argument("--run-id")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--judge-prompt-spec", type=Path, default=base.JUDGE_PROMPT_PATH
    )
    parser.add_argument("--skip-rationales", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def calibration_metrics(rows: list[dict[str, Any]], judge_spec: dict[str, Any]) -> dict[str, Any]:
    count = len(rows)
    task_resolved = [row for row in rows if row["automated_task_accuracy"] is not None]
    ground_resolved = [
        row for row in rows if row["automated_contextual_grounding"] is not None
    ]
    failure_resolved = [
        row for row in rows if row["automated_primary_failure_mode"] is not None
    ]
    task_exact = sum(
        row["automated_task_accuracy"] == row["human_task_accuracy"]
        for row in task_resolved
    )
    task_within_one = sum(
        abs(row["automated_task_accuracy"] - row["human_task_accuracy"]) <= 1
        for row in task_resolved
    )
    failure_exact = sum(
        row["automated_primary_failure_mode"] == row["human_primary_failure_mode"]
        for row in failure_resolved
    )
    critical_reversals = [
        row["scenario_id"]
        for row in rows
        if row["severity_class"] == 5
        and (
            (
                row["human_task_accuracy"] >= 4
                and (
                    row["automated_task_accuracy"] is None
                    or row["automated_task_accuracy"] <= 2
                )
            )
            or (
                row["human_task_accuracy"] <= 2
                and (
                    row["automated_task_accuracy"] is None
                    or row["automated_task_accuracy"] >= 4
                )
            )
            or (
                row["human_primary_failure_mode"] == "unsafe"
                and row["automated_primary_failure_mode"] != "unsafe"
            )
        )
    ]
    gates = judge_spec["aggregation_and_review"]["calibration_gate"]
    task_within_one_rate = task_within_one / count if count else 0.0
    failure_exact_rate = failure_exact / count if count else 0.0
    task_alpha = base.batch_alpha(rows, "task_accuracy_ratings")
    gate_checks = {
        "all_task_scores_resolved": len(task_resolved) == count,
        "all_grounding_scores_resolved": len(ground_resolved) == count,
        "all_failure_labels_resolved": len(failure_resolved) == count,
        "development_task_alpha": task_alpha is not None
        and task_alpha >= float(gates["development_ordinal_alpha_minimum"]),
        "task_within_one_human_rate": task_within_one_rate
        >= float(gates["task_within_one_human_rate_minimum"]),
        "failure_exact_human_rate": failure_exact_rate
        >= float(gates["failure_exact_human_rate_minimum"]),
        "critical_reversal_count": len(critical_reversals)
        <= int(gates["critical_reversal_count_maximum"]),
    }
    return {
        "count": count,
        "task_resolved_count": len(task_resolved),
        "grounding_resolved_count": len(ground_resolved),
        "failure_resolved_count": len(failure_resolved),
        "task_exact_count": task_exact,
        "task_exact_rate": round(task_exact / count, 6) if count else 0.0,
        "task_within_one_count": task_within_one,
        "task_within_one_rate": round(task_within_one_rate, 6),
        "task_mae_with_unresolved_as_missing": (
            round(
                statistics.fmean(
                    abs(row["automated_task_accuracy"] - row["human_task_accuracy"])
                    for row in task_resolved
                ),
                6,
            )
            if task_resolved
            else None
        ),
        "failure_exact_count": failure_exact,
        "failure_exact_rate": round(failure_exact_rate, 6),
        "task_formulation_alpha": task_alpha,
        "grounding_formulation_alpha": base.batch_alpha(
            rows, "contextual_grounding_ratings"
        ),
        "critical_reversal_scenario_ids": critical_reversals,
        "gate_checks": gate_checks,
        "calibration_passed": all(gate_checks.values()),
    }


def main() -> int:
    args = parse_args()
    candidate_path = args.candidate_rows.resolve()
    human_path = args.human_gold.resolve()
    output_root = args.output_root.resolve()
    judge_prompt_path = args.judge_prompt_spec.resolve()
    base.assert_non_c_path(candidate_path, "candidate_rows")
    base.assert_non_c_path(human_path, "human_gold")
    base.assert_non_c_path(output_root, "output_root")
    base.assert_non_c_path(judge_prompt_path, "judge_prompt_spec")

    scenario_doc = base.load_yaml(base.SCENARIO_PATH)
    judge_spec = base.load_yaml(judge_prompt_path)
    regulation_doc = base.load_yaml(base.REGULATION_PATH)
    scenarios = scenario_doc["scenarios"]
    scenario_by_id = {item["scenario_id"]: item for item in scenarios}
    regulations = {
        item["regulation_id"]: item for item in regulation_doc["regulations"]
    }
    candidate_rows = {row["scenario_id"]: row for row in read_jsonl(candidate_path)}
    human_doc = yaml.safe_load(human_path.read_text(encoding="utf-8"))
    human_by_id = {item["scenario_id"]: item for item in human_doc["items"]}

    if args.scenario_id:
        selected_ids = args.scenario_id
    elif args.mode == "development":
        selected_ids = [
            item["scenario_id"] for item in scenarios if item["split"] == "development"
        ]
    else:
        selected_ids = [item["scenario_id"] for item in scenarios]
    missing = (set(selected_ids) - set(candidate_rows)) | (
        set(selected_ids) - set(human_by_id)
    )
    if missing:
        raise ValueError(f"Missing selected scenarios: {sorted(missing)}")

    if args.judge_model == "flan":
        model_dir = (args.model_dir or base.DEFAULT_MODEL_DIR).resolve()
        revision = base.read_model_revision(model_dir, args.model_revision)
        engine: Any = base.LocalFlanEngine(model_dir, 512)
        judge_model_id = "google/flan-t5-base"
        device = "cpu"
    else:
        from W02_Mistral_Eval_Runner import DEFAULT_MODEL_DIR, LocalMistralEngine

        model_dir = (args.model_dir or DEFAULT_MODEL_DIR).resolve()
        revision = args.model_revision or MISTRAL_REVISION
        engine = LocalMistralEngine(model_dir, 1024)
        judge_model_id = MISTRAL_ID
        device = "cuda:0"

    run_id = args.run_id or (
        f"{args.judge_model}-judge-calibration-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    )
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
        raise ValueError("run-id contains unsupported characters")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    rows: list[dict[str, Any]] = []
    for index, scenario_id in enumerate(selected_ids, start=1):
        print(f"[{index}/{len(selected_ids)}] revised judges: {scenario_id}", flush=True)
        scenario = scenario_by_id[scenario_id]
        candidate = candidate_rows[scenario_id]
        human = human_by_id[scenario_id][args.human_model_key]
        regulation_text = base.build_regulation_text(scenario, regulations)
        judges = base.run_judges(
            engine,
            scenario,
            candidate["raw_output"],
            regulation_text,
            judge_spec,
            args.skip_rationales,
        )
        prompts = base.rendered_judge_prompts(
            scenario,
            candidate["raw_output"],
            regulation_text,
            judge_spec,
            judges,
        )
        task_ratings = [item["task_accuracy"] for item in judges]
        grounding_ratings = [item["contextual_grounding"] for item in judges]
        failure_ratings = [item["primary_failure_mode"] for item in judges]
        rows.append(
            {
                "run_id": run_id,
                "timestamp_utc": base.utc_now(),
                "runner_version": RUNNER_VERSION,
                "benchmark_version": str(scenario_doc["benchmark_version"]),
                "judge_prompt_version": str(judge_spec["version"]),
                "scenario_id": scenario_id,
                "split": scenario["split"],
                "severity_class": scenario["severity_class"],
                "candidate_model_id": candidate["candidate_model_id"],
                "candidate_model_revision": candidate["candidate_model_revision"],
                "candidate_run_id": candidate["run_id"],
                "candidate_output": candidate["raw_output"],
                "candidate_output_sha256": base.sha256_text(candidate["raw_output"]),
                "judge_model_id": judge_model_id,
                "judge_model_revision": revision,
                "judge_device": device,
                "seed": seed,
                "judge_results": judges,
                "judge_rendered_prompts": prompts,
                "judge_prompt_hashes": {
                    formulation: {
                        purpose: base.sha256_text(text)
                        for purpose, text in formulation_prompts.items()
                    }
                    for formulation, formulation_prompts in prompts.items()
                },
                "task_accuracy_ratings": task_ratings,
                "contextual_grounding_ratings": grounding_ratings,
                "primary_failure_mode_ratings": failure_ratings,
                "automated_task_accuracy": base.numeric_consensus(task_ratings),
                "automated_contextual_grounding": base.numeric_consensus(
                    grounding_ratings
                ),
                "automated_primary_failure_mode": base.failure_consensus(
                    failure_ratings
                ),
                "human_task_accuracy": int(human["task_accuracy"]),
                "human_contextual_grounding": int(human["contextual_grounding"]),
                "human_primary_failure_mode": human["primary_failure_mode"],
                "human_rationale": human["rationale"],
            }
        )

    metrics = calibration_metrics(rows, judge_spec)
    jsonl_path = run_dir / "W02_Judge_Calibration_Rows.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "run_id": run_id,
        "created_at_utc": base.utc_now(),
        "runner_version": RUNNER_VERSION,
        "judge_model_id": judge_model_id,
        "judge_model_revision": revision,
        "judge_device": device,
        "candidate_rows": str(candidate_path),
        "candidate_rows_sha256": base.sha256_file(candidate_path),
        "human_gold": str(human_path),
        "human_gold_sha256": base.sha256_file(human_path),
        "human_model_key": args.human_model_key,
        "judge_prompt_spec_version": str(judge_spec["version"]),
        "judge_prompt_spec_path": str(judge_prompt_path),
        "judge_prompt_spec_sha256": base.sha256_file(judge_prompt_path),
        "seed": seed,
        "scenario_ids": selected_ids,
        "skip_rationales": args.skip_rationales,
        "metrics": metrics,
        "allowed_claim": "judge-prompt calibration against provisional single-reviewer gold",
        "prohibited_claim": "independent judge validation or deployed-product performance",
    }
    base.json_dump(run_dir / "W02_Judge_Calibration_Summary.json", summary)

    lines = [
        "# Week 2 Judge Calibration",
        "",
        "> Three formulations on one checkpoint are sensitivity probes, not independent judges.",
        "",
        f"- Run: `{run_id}`",
        f"- Judge: `{judge_model_id}` at `{revision}`",
        f"- Prompt spec: `{judge_spec['version']}`",
        f"- Items: `{len(rows)}`",
        f"- Calibration passed: `{metrics['calibration_passed']}`",
        f"- Task formulation alpha: `{metrics['task_formulation_alpha']}`",
        f"- Task within-one vs human: `{metrics['task_within_one_count']}/{metrics['count']}`",
        f"- Failure exact vs human: `{metrics['failure_exact_count']}/{metrics['count']}`",
        f"- Critical reversals: `{metrics['critical_reversal_scenario_ids']}`",
        "",
        "## Row evidence",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"### {row['scenario_id']}",
                "",
                "**Candidate output**",
                "",
                "```text",
                row["candidate_output"],
                "```",
                "",
                f"- Judge ratings: `{row['task_accuracy_ratings']}` / `{row['contextual_grounding_ratings']}` / `{row['primary_failure_mode_ratings']}`",
                f"- Automated consensus: `{row['automated_task_accuracy']}` / `{row['automated_contextual_grounding']}` / `{row['automated_primary_failure_mode']}`",
                f"- Human gold: `{row['human_task_accuracy']}` / `{row['human_contextual_grounding']}` / `{row['human_primary_failure_mode']}`",
                f"- Human reason: {row['human_rationale']}",
                "",
            ]
        )
        for formulation, prompts in row["judge_rendered_prompts"].items():
            lines.extend(
                [
                    f"<details><summary>{formulation} exact prompts and outputs</summary>",
                    "",
                ]
            )
            judge = next(
                item for item in row["judge_results"] if item["formulation"] == formulation
            )
            for purpose in ("task_accuracy", "contextual_grounding", "failure_mode"):
                lines.extend(
                    [
                        f"#### {purpose}",
                        "",
                        "```text",
                        prompts[purpose],
                        "```",
                        "",
                    ]
                )
            lines.extend(
                [
                    f"Result: `{judge['task_accuracy']}` / `{judge['contextual_grounding']}` / `{judge['primary_failure_mode']}`",
                    "",
                    f"Rationale: {judge['rationale']}",
                    "",
                    "</details>",
                    "",
                ]
            )
    (run_dir / "W02_Judge_Calibration_Report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    return 0 if metrics["calibration_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Calibrate Prometheus 2 against the frozen Week 2 human adjudication set."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

import W02_Prometheus_Judge as prometheus
import W02_Structured_Judge as structured
import W02_Structured_Judge_Calibration as prior_calibration


RUNNER_VERSION = "0.8.3"
ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = ROOT / "experiments" / "w02_prometheus_judge"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("smoke", "calibration"),
        default="calibration",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=prometheus.DEFAULT_MODEL_DIR,
    )
    parser.add_argument(
        "--prometheus-spec",
        type=Path,
        default=prometheus.DEFAULT_SPEC_PATH,
    )
    parser.add_argument(
        "--prompt-spec",
        type=Path,
        default=structured.DEFAULT_PROMPT_SPEC,
    )
    parser.add_argument(
        "--metadata-spec",
        type=Path,
        default=structured.DEFAULT_METADATA_SPEC,
    )
    parser.add_argument(
        "--scenario-path",
        type=Path,
        default=prior_calibration.SCENARIO_PATH,
    )
    parser.add_argument(
        "--old-rows",
        type=Path,
        default=prior_calibration.OLD_ROWS_PATH,
    )
    parser.add_argument(
        "--new-rows",
        type=Path,
        default=prior_calibration.NEW_ROWS_PATH,
    )
    parser.add_argument(
        "--new-human",
        type=Path,
        default=prior_calibration.NEW_HUMAN_PATH,
    )
    return parser.parse_args()


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def _write_report(
    path: Path,
    *,
    run_id: str,
    rows: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    lines = [
        "# Week 2 Prometheus Judge Calibration",
        "",
        f"- Run ID: `{run_id}`",
        f"- Frozen human items: `{metrics['item_count']}`",
        (
            "- Krippendorff ordinal alpha: "
            f"`{metrics['ordinal_krippendorff_alpha']}`"
        ),
        (
            "- Task within one of human: "
            f"`{metrics['task_within_one_of_human_count']}/"
            f"{metrics['item_count']}` "
            f"(`{metrics['task_within_one_of_human_rate']:.3f}`)"
        ),
        (
            "- Failure exact match: "
            f"`{metrics['failure_mode_exact_match_count']}/"
            f"{metrics['item_count']}` "
            f"(`{metrics['failure_mode_exact_match_rate']:.3f}`)"
        ),
        (
            "- Critical safety reversals: "
            f"`{metrics['critical_safety_reversal_count']}`"
        ),
        f"- Exact `[RESULT]` format rate: `{metrics['exact_format_rate']:.3f}`",
        f"- Pipeline usable: `{metrics['pipeline_usable']}`",
        "",
        "## Gate results",
        "",
    ]
    for key, value in metrics["gate_results"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Item audit", ""])
    for row in rows:
        judge = row["structured_judge"]
        lines.extend(
            [
                f"### {row['calibration_item_id']}",
                "",
                f"- Candidate: {row['candidate']}",
                (
                    "- Human Task/Grounding/Failure: "
                    f"`{row['human_task_accuracy']}` / "
                    f"`{row['human_contextual_grounding']}` / "
                    f"`{row['human_primary_failure_mode']}`"
                ),
                (
                    "- Judge Task ratings -> consensus: "
                    f"`{judge['task_accuracy_ratings']}` -> "
                    f"`{judge['consensus']['task_accuracy']['final']}`"
                ),
                (
                    "- Judge Grounding ratings -> consensus: "
                    f"`{judge['contextual_grounding_ratings']}` -> "
                    f"`{judge['consensus']['contextual_grounding']['final']}`"
                ),
                (
                    "- Judge Failure ratings -> consensus: "
                    f"`{judge['primary_failure_mode_ratings']}` -> "
                    f"`{judge['consensus']['primary_failure_mode']['final']}`"
                ),
                f"- Human rationale: {row['human_rationale']}",
                "",
            ]
        )
        for formulation in judge["formulation_results"]:
            lines.append(f"#### {formulation['formulation']}")
            lines.append("")
            for dimension, call in formulation["dimension_calls"].items():
                lines.extend(
                    [
                        (
                            f"- `{dimension}` raw output: "
                            f"{call['generation']['text']}"
                        ),
                        "",
                    ]
                )
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    args = parse_args()
    run_id = args.run_id or f"prometheus-judge-{args.mode}-v{RUNNER_VERSION}"
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "rows.checkpoint.jsonl"
    if checkpoint.exists() and not args.resume:
        raise FileExistsError(
            f"Run already exists; pass --resume or choose a new run ID: {run_dir}"
        )

    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)

    scenario_document = prior_calibration.load_yaml(args.scenario_path)
    scenario_list = scenario_document["scenarios"]
    scenarios = {item["scenario_id"]: item for item in scenario_list}
    structured_specs = structured.load_specs(
        args.prompt_spec,
        args.metadata_spec,
    )
    validation_errors = structured.validate_specs(
        scenario_list,
        structured_specs,
    )
    if validation_errors:
        raise ValueError("; ".join(validation_errors))
    judge_spec = prometheus.load_spec(args.prometheus_spec)
    items = prior_calibration.build_calibration_items(
        "calibration",
        prior_calibration.load_jsonl(args.old_rows),
        prior_calibration.load_jsonl(args.new_rows),
        prior_calibration.load_yaml(args.new_human),
    )
    items = prior_calibration.select_items(items, args.mode, [])
    existing = (
        prior_calibration.load_jsonl(checkpoint)
        if checkpoint.exists()
        else []
    )
    completed = {row["calibration_item_id"] for row in existing}
    engine = prometheus.PrometheusEngine(
        args.model_dir,
        max_input_tokens=int(judge_spec["generation"]["max_input_tokens"]),
    )

    rows = list(existing)
    for index, item in enumerate(items, start=1):
        if item["calibration_item_id"] in completed:
            print(f"Skipping completed {item['calibration_item_id']}", flush=True)
            continue
        scenario = scenarios[item["scenario_id"]]
        print(
            f"[{index}/{len(items)}] Judging {item['calibration_item_id']}",
            flush=True,
        )
        judge = prometheus.run_prometheus_judges(
            engine,
            scenario,
            item["candidate"],
            judge_spec,
            structured_specs,
            spec_path=args.prometheus_spec,
        )
        row = {
            **item,
            "severity_class": int(scenario["severity_class"]),
            "scenario_input_stimulus": scenario["input_stimulus"],
            "expected_behavior_range": scenario["expected_behavior_range"],
            "failure_conditions": scenario["failure_conditions"],
            "structured_judge": judge,
        }
        row["calibration_row_sha256"] = structured.canonical_sha256(row)
        rows.append(row)
        _append_jsonl(checkpoint, row)
        print(
            {
                "task": judge["task_accuracy_ratings"],
                "task_final": judge["consensus"]["task_accuracy"]["final"],
                "grounding": judge["contextual_grounding_ratings"],
                "failure": judge["primary_failure_mode_ratings"],
                "failure_final": judge["consensus"][
                    "primary_failure_mode"
                ]["final"],
            },
            flush=True,
        )

    by_id = {row["calibration_item_id"]: row for row in rows}
    rows = [by_id[item["calibration_item_id"]] for item in items]
    gates = judge_spec["calibration_policy"]["acceptance_gates"]
    metrics = prior_calibration.compute_metrics(rows, gates)

    final_rows = run_dir / "W02_Prometheus_Judge_Calibration_Rows.jsonl"
    with final_rows.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    prior_calibration.write_csv(
        run_dir / "W02_Prometheus_Judge_Calibration_Summary.csv",
        rows,
    )
    metrics_path = run_dir / "W02_Prometheus_Calibration_Metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "runner_version": RUNNER_VERSION,
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
        newline="\n",
    )
    _write_report(
        run_dir / "W02_Prometheus_Judge_Calibration_Report.md",
        run_id=run_id,
        rows=rows,
        metrics=metrics,
    )
    manifest = {
        "run_id": run_id,
        "runner_version": RUNNER_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": 42,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "device": str(engine.device),
        "model": {
            "repo_id": judge_spec["model"]["repo_id"],
            "revision": judge_spec["model"]["revision"],
            "config_sha256": engine.model_config_sha256,
            "weights_sha256": engine.model_weights_sha256,
            "download_manifest_sha256": engine.download_manifest_sha256,
        },
        "inputs": {
            str(path): prometheus.sha256_file(path)
            for path in (
                args.prometheus_spec,
                args.prompt_spec,
                args.metadata_spec,
                args.scenario_path,
                args.old_rows,
                args.new_rows,
                args.new_human,
            )
        },
        "outputs": {
            path.name: prometheus.sha256_file(path)
            for path in sorted(run_dir.iterdir())
            if path.is_file()
        },
    }
    (run_dir / "W02_Prometheus_Calibration_Manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "metrics": metrics,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

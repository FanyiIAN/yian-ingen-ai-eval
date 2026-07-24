"""Score the frozen 35 x 2 Week 2 candidate set with three Prometheus Judges.

This runner never silently promotes an uncalibrated Judge to a production scorer.
If the frozen calibration gates fail, a full run is allowed only with the explicit
``--allow-failed-calibration`` flag and every result is labelled diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import krippendorff
import numpy as np
import torch
import transformers
import yaml

import W02_Prometheus_Judge as prometheus
import W02_Structured_Judge as structured


RUNNER_VERSION = "0.8.4"
SEED = 42
ROOT = Path(__file__).resolve().parent
DEFAULT_CANDIDATES = (
    Path("/workspace/experiments/w02_submission_full")
    / "w02-two-model-full-v0.6.2"
    / "W02_Two_Model_Candidate_Rows.jsonl"
)
DEFAULT_OUTPUT_ROOT = ROOT / "experiments" / "w02_prometheus_full"
DEFAULT_CALIBRATION_METRICS = (
    ROOT
    / "experiments"
    / "w02_prometheus_judge"
    / "prometheus-judge-calibration-v0.8.3"
    / "W02_Prometheus_Calibration_Metrics.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        default="w02-two-model-prometheus-diagnostic-v0.8.4-batch9",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--candidate-rows", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--scenario-path",
        type=Path,
        default=ROOT / "W02_Scenarios.yaml",
    )
    parser.add_argument(
        "--prometheus-spec",
        type=Path,
        default=prometheus.DEFAULT_SPEC_PATH,
    )
    parser.add_argument(
        "--judge-prompt-spec",
        type=Path,
        default=structured.DEFAULT_PROMPT_SPEC,
    )
    parser.add_argument(
        "--metadata-spec",
        type=Path,
        default=structured.DEFAULT_METADATA_SPEC,
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=prometheus.DEFAULT_MODEL_DIR,
    )
    parser.add_argument(
        "--calibration-metrics",
        type=Path,
        default=DEFAULT_CALIBRATION_METRICS,
    )
    parser.add_argument("--allow-failed-calibration", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def load_scenarios(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("scenarios"), list):
        raise ValueError(f"Invalid scenario document: {path}")
    return document, {
        scenario["scenario_id"]: scenario for scenario in document["scenarios"]
    }


def calibration_status(
    path: Path,
    *,
    allow_failed: bool,
) -> tuple[dict[str, Any], str]:
    if not path.exists():
        raise FileNotFoundError(f"Calibration metrics missing: {path}")
    metrics = load_json(path)
    usable = bool(metrics.get("pipeline_usable"))
    if not usable and not allow_failed:
        raise RuntimeError(
            "Judge calibration failed. Pass --allow-failed-calibration only for "
            "a diagnostic run that will not be reported as reliable automated scoring."
        )
    return metrics, "calibrated" if usable else "diagnostic_failed_calibration"


def set_determinism() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)


def consensus_value(row: dict[str, Any], dimension: str) -> Any:
    return row["structured_judge"]["consensus"][dimension]["final"]


def task_alpha(rows: list[dict[str, Any]]) -> float | None:
    if len(rows) < 2:
        return None
    matrix = np.asarray(
        [
            [
                np.nan
                if row["structured_judge"]["task_accuracy_ratings"][index] is None
                else float(row["structured_judge"]["task_accuracy_ratings"][index])
                for row in rows
            ]
            for index in range(3)
        ],
        dtype=float,
    )
    if np.all(np.isnan(matrix)):
        return None
    try:
        value = float(
            krippendorff.alpha(
                reliability_data=matrix,
                level_of_measurement="ordinal",
            )
        )
    except (ValueError, ZeroDivisionError):
        return None
    return value if math.isfinite(value) else None


def exact_format_rate(rows: list[dict[str, Any]]) -> float:
    calls = [
        call
        for row in rows
        for formulation in row["structured_judge"]["formulation_results"]
        for call in formulation["dimension_calls"].values()
    ]
    return (
        sum(bool(call["parsed"]["contract_exact"]) for call in calls) / len(calls)
        if calls
        else 0.0
    )


def unresolved_rate(rows: list[dict[str, Any]]) -> float:
    denominator = len(rows) * 9
    numerator = sum(
        len(row["structured_judge"]["unresolved_atomic_checks"]) for row in rows
    )
    return numerator / denominator if denominator else 0.0


def model_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stable_task = [
        int(value)
        for row in rows
        for value in [consensus_value(row, "task_accuracy")]
        if value is not None
    ]
    stable_grounding = [
        int(value)
        for row in rows
        for value in [consensus_value(row, "contextual_grounding")]
        if value is not None
    ]
    failures = Counter(
        "unresolved" if value is None else str(value)
        for row in rows
        for value in [consensus_value(row, "primary_failure_mode")]
    )
    return {
        "candidate_count": len(rows),
        "stable_task_consensus_count": len(stable_task),
        "stable_grounding_consensus_count": len(stable_grounding),
        "mean_stable_task_accuracy": (
            statistics.fmean(stable_task) if stable_task else None
        ),
        "mean_stable_contextual_grounding": (
            statistics.fmean(stable_grounding) if stable_grounding else None
        ),
        "primary_failure_counts": dict(sorted(failures.items())),
        "task_ordinal_krippendorff_alpha": task_alpha(rows),
        "unresolved_check_rate": unresolved_rate(rows),
        "exact_result_contract_rate": exact_format_rate(rows),
        "human_review_required_count": sum(
            bool(row["human_review_required"]) for row in rows
        ),
    }


def compute_summary(
    *,
    run_id: str,
    rows: list[dict[str, Any]],
    calibration: dict[str, Any],
    score_status: str,
) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(row["candidate_model_id"], []).append(row)
    return {
        "run_id": run_id,
        "runner_version": RUNNER_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "score_status": score_status,
        "reliable_automated_benchmark_claim_allowed": score_status == "calibrated",
        "candidate_count": len(rows),
        "scenario_count": len({row["scenario_id"] for row in rows}),
        "candidate_model_count": len(by_model),
        "judge_formulation_count": 3,
        "judge_dimension_count": 3,
        "judge_model_call_count": len(rows) * 9,
        "judge_batch_invocation_count": len(rows),
        "calibration_metrics": calibration,
        "overall": model_summary(rows),
        "by_candidate_model": {
            model_id: model_summary(model_rows)
            for model_id, model_rows in sorted(by_model.items())
        },
    }


def prompt_trace(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for row in rows:
        for formulation in row["structured_judge"]["formulation_results"]:
            for dimension, call in formulation["dimension_calls"].items():
                trace_row = {
                    "run_id": row["run_id"],
                    "candidate_item_id": row["candidate_item_id"],
                    "candidate_model_id": row["candidate_model_id"],
                    "candidate_model_revision": row["candidate_model_revision"],
                    "scenario_id": row["scenario_id"],
                    "severity_class": row["severity_class"],
                    "candidate_prompt_version": row["candidate_prompt_version"],
                    "candidate_prompt": row["candidate_prompt"],
                    "candidate_prompt_sha256": row["candidate_prompt_sha256"],
                    "candidate_output": row["raw_output"],
                    "candidate_output_sha256": row["candidate_output_sha256"],
                    "judge_formulation": formulation["formulation"],
                    "judge_dimension": dimension,
                    "judge_prompt_version": call["prompt_version"],
                    "judge_system_prompt": call["system_prompt"],
                    "judge_user_prompt": call["user_prompt"],
                    "judge_system_prompt_sha256": call["system_prompt_sha256"],
                    "judge_user_prompt_sha256": call["user_prompt_sha256"],
                    "judge_raw_output": call["generation"]["text"],
                    "judge_raw_output_sha256": structured.sha256_text(
                        call["generation"]["text"]
                    ),
                    "judge_generation_record_sha256": call["generation_sha256"],
                    "judge_score": call["parsed"]["score"],
                    "judge_feedback": call["parsed"]["feedback"],
                    "judge_parse_status": call["parsed"]["parse_status"],
                    "judge_exact_contract": call["parsed"]["contract_exact"],
                    "judge_input_tokens": call["generation"]["input_tokens"],
                    "judge_output_tokens": call["generation"]["output_tokens"],
                    "judge_latency_ms": call["generation"]["latency_ms"],
                    "judge_batch_size": call["generation"].get("batch_size", 1),
                    "judge_batch_latency_ms": call["generation"].get(
                        "batch_latency_ms",
                        call["generation"]["latency_ms"],
                    ),
                }
                trace_row["trace_row_sha256"] = structured.canonical_sha256(
                    trace_row
                )
                trace.append(trace_row)
    return trace


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "run_id",
        "candidate_item_id",
        "scenario_id",
        "split",
        "platform",
        "severity_class",
        "candidate_model_id",
        "candidate_model_revision",
        "candidate_prompt_version",
        "candidate_prompt_sha256",
        "candidate_prompt",
        "raw_output",
        "candidate_output_sha256",
        "task_accuracy_ratings",
        "contextual_grounding_ratings",
        "primary_failure_mode_ratings",
        "task_accuracy_consensus",
        "contextual_grounding_consensus",
        "primary_failure_mode_consensus",
        "judge_feedback_json",
        "human_review_required",
        "human_review_reasons",
        "score_status",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            judge = row["structured_judge"]
            feedback = {
                formulation["formulation"]: {
                    dimension: call["parsed"]["feedback"]
                    for dimension, call in formulation["dimension_calls"].items()
                }
                for formulation in judge["formulation_results"]
            }
            writer.writerow(
                {
                    **{field: row.get(field) for field in fields},
                    "task_accuracy_ratings": json.dumps(
                        judge["task_accuracy_ratings"], ensure_ascii=False
                    ),
                    "contextual_grounding_ratings": json.dumps(
                        judge["contextual_grounding_ratings"], ensure_ascii=False
                    ),
                    "primary_failure_mode_ratings": json.dumps(
                        judge["primary_failure_mode_ratings"], ensure_ascii=False
                    ),
                    "task_accuracy_consensus": consensus_value(
                        row, "task_accuracy"
                    ),
                    "contextual_grounding_consensus": consensus_value(
                        row, "contextual_grounding"
                    ),
                    "primary_failure_mode_consensus": consensus_value(
                        row, "primary_failure_mode"
                    ),
                    "judge_feedback_json": json.dumps(
                        feedback, ensure_ascii=False
                    ),
                    "human_review_reasons": json.dumps(
                        row["human_review_reasons"], ensure_ascii=False
                    ),
                }
            )


def fenced(text: str) -> list[str]:
    return ["```text", text, "```"]


def write_report(
    path: Path,
    *,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Week 2 Two-Model Prometheus Diagnostic Run",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Score status: `{summary['score_status']}`",
        f"- Candidate outputs: `{summary['candidate_count']}`",
        f"- Scenarios: `{summary['scenario_count']}`",
        f"- Judge model calls: `{summary['judge_model_call_count']}`",
        f"- Batched GPU invocations: `{summary['judge_batch_invocation_count']}`",
        (
            "- Reliable automated benchmark claim allowed: "
            f"`{summary['reliable_automated_benchmark_claim_allowed']}`"
        ),
        "",
        (
            "> The full candidate prompts, candidate outputs, exact Judge prompts, "
            "raw Judge generations, feedback, scores, token counts, latency, and "
            "hashes are preserved in `W02_Prometheus_Prompt_Trace.jsonl`."
        ),
        "",
        "## Model summary",
        "",
        "| Candidate model | Stable Task | Mean Task | Stable Grounding | "
        "Mean Grounding | Unresolved rate | Review |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_id, item in summary["by_candidate_model"].items():
        lines.append(
            f"| {model_id} | {item['stable_task_consensus_count']}/"
            f"{item['candidate_count']} | {item['mean_stable_task_accuracy']} | "
            f"{item['stable_grounding_consensus_count']}/"
            f"{item['candidate_count']} | "
            f"{item['mean_stable_contextual_grounding']} | "
            f"{item['unresolved_check_rate']:.3f} | "
            f"{item['human_review_required_count']}/{item['candidate_count']} |"
        )
    lines.extend(["", "## Per-case trace", ""])
    for index, row in enumerate(rows, start=1):
        judge = row["structured_judge"]
        lines.extend(
            [
                f"### {index}. {row['candidate_item_id']}",
                "",
                (
                    f"- Model/revision: `{row['candidate_model_id']}` / "
                    f"`{row['candidate_model_revision']}`"
                ),
                (
                    f"- Split/severity: `{row['split']}` / "
                    f"`{row['severity_class']}`"
                ),
                (
                    f"- Candidate prompt version/hash: "
                    f"`{row['candidate_prompt_version']}` / "
                    f"`{row['candidate_prompt_sha256']}`"
                ),
                (
                    "- Task / Grounding / Failure consensus: "
                    f"`{consensus_value(row, 'task_accuracy')}` / "
                    f"`{consensus_value(row, 'contextual_grounding')}` / "
                    f"`{consensus_value(row, 'primary_failure_mode')}`"
                ),
                f"- Human review: `{row['human_review_required']}` — "
                f"`{row['human_review_reasons']}`",
                "",
                "#### Candidate input prompt",
                "",
                *fenced(row["candidate_prompt"]),
                "",
                "#### Candidate output",
                "",
                *fenced(row["raw_output"]),
                "",
                "#### Three-Judge results and comments",
                "",
            ]
        )
        for formulation in judge["formulation_results"]:
            mapping = formulation["deterministic_mapping"]
            lines.extend(
                [
                    f"##### {formulation['formulation']}",
                    "",
                    (
                        "- Task / Grounding / Failure: "
                        f"`{mapping['task_accuracy']}` / "
                        f"`{mapping['contextual_grounding']}` / "
                        f"`{mapping['primary_failure_mode']}`"
                    ),
                ]
            )
            for dimension, call in formulation["dimension_calls"].items():
                lines.extend(
                    [
                        (
                            f"- `{dimension}` (`{call['prompt_version']}`): "
                            f"{call['parsed']['feedback']} "
                            f"[raw score `{call['parsed']['score']}`; "
                            f"parse `{call['parsed']['parse_status']}`]"
                        ),
                    ]
                )
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def write_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    run_dir: Path,
    summary: dict[str, Any],
) -> None:
    artifacts = {
        item.name: prometheus.sha256_file(item)
        for item in sorted(run_dir.iterdir())
        if item.is_file() and item != path and not item.name.endswith(".checkpoint.jsonl")
    }
    gpu = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "cuda_runtime": torch.version.cuda,
        }
    manifest = {
        "run_id": args.run_id,
        "runner_version": RUNNER_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "score_status": summary["score_status"],
        "source_candidate_rows": {
            "path": str(args.candidate_rows),
            "sha256": prometheus.sha256_file(args.candidate_rows),
        },
        "scenario_source": {
            "path": str(args.scenario_path),
            "sha256": prometheus.sha256_file(args.scenario_path),
        },
        "prometheus_spec": {
            "path": str(args.prometheus_spec),
            "sha256": prometheus.sha256_file(args.prometheus_spec),
        },
        "requirement_metadata": {
            "path": str(args.metadata_spec),
            "sha256": prometheus.sha256_file(args.metadata_spec),
        },
        "calibration_metrics": {
            "path": str(args.calibration_metrics),
            "sha256": prometheus.sha256_file(args.calibration_metrics),
            "pipeline_usable": bool(
                summary["calibration_metrics"].get("pipeline_usable")
            ),
        },
        "judge_model": summary["judge_backend"],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu": gpu,
        },
        "artifact_sha256": artifacts,
    }
    manifest["manifest_content_sha256"] = structured.canonical_sha256(manifest)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    args = parse_args()
    run_dir = args.output_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "rows.checkpoint.jsonl"
    if checkpoint.exists() and not args.resume:
        raise FileExistsError(f"Run exists; pass --resume: {run_dir}")

    candidate_rows = load_jsonl(args.candidate_rows)
    candidate_rows = sorted(
        candidate_rows,
        key=lambda row: row["candidate_item_id"],
    )
    if args.limit is not None:
        candidate_rows = candidate_rows[: args.limit]
    if not candidate_rows:
        raise ValueError("No candidate rows selected")

    scenario_document, scenarios = load_scenarios(args.scenario_path)
    missing = {
        row["scenario_id"] for row in candidate_rows
    } - set(scenarios)
    if missing:
        raise ValueError(f"Candidate rows reference unknown scenarios: {sorted(missing)}")

    structured_specs = structured.load_specs(
        args.judge_prompt_spec,
        args.metadata_spec,
    )
    validation_errors = structured.validate_specs(
        scenario_document["scenarios"],
        structured_specs,
    )
    if validation_errors:
        raise ValueError(" | ".join(validation_errors))
    judge_spec = prometheus.load_spec(args.prometheus_spec)
    calibration, score_status = calibration_status(
        args.calibration_metrics,
        allow_failed=args.allow_failed_calibration,
    )

    set_determinism()
    engine = prometheus.PrometheusEngine(
        args.model_dir,
        max_input_tokens=int(judge_spec["generation"]["max_input_tokens"]),
    )
    existing = load_jsonl(checkpoint) if checkpoint.exists() else []
    completed = {row["candidate_item_id"] for row in existing}
    rows = list(existing)

    for index, candidate in enumerate(candidate_rows, start=1):
        item_id = candidate["candidate_item_id"]
        if item_id in completed:
            print(f"[{index}/{len(candidate_rows)}] Skip {item_id}", flush=True)
            continue
        print(f"[{index}/{len(candidate_rows)}] Judge {item_id}", flush=True)
        scenario = scenarios[candidate["scenario_id"]]
        judge = prometheus.run_prometheus_judges_batched(
            engine,
            scenario,
            candidate["raw_output"],
            judge_spec,
            structured_specs,
            spec_path=args.prometheus_spec,
        )
        review_reasons = list(judge["human_review_reasons"])
        if score_status != "calibrated":
            review_reasons.append("judge_failed_calibration_diagnostic_only")
        row = {
            **candidate,
            "run_id": args.run_id,
            "source_candidate_run_id": candidate["run_id"],
            "source_candidate_row_sha256": candidate["candidate_row_sha256"],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "prometheus_full_runner_version": RUNNER_VERSION,
            "judge_seed": SEED,
            "judge_calibration_metrics_path": str(args.calibration_metrics),
            "judge_calibration_metrics_sha256": prometheus.sha256_file(
                args.calibration_metrics
            ),
            "score_status": score_status,
            "structured_judge": judge,
            "human_review_required": True
            if score_status != "calibrated"
            else bool(review_reasons),
            "human_review_reasons": sorted(set(review_reasons)),
            "human_task_accuracy": None,
            "human_contextual_grounding": None,
            "human_primary_failure_mode": None,
            "human_rationale": None,
        }
        row["judged_row_sha256"] = structured.canonical_sha256(row)
        append_jsonl(checkpoint, row)
        rows.append(row)
        print(
            {
                "task": judge["task_accuracy_ratings"],
                "grounding": judge["contextual_grounding_ratings"],
                "failure": judge["primary_failure_mode_ratings"],
                "review": row["human_review_required"],
            },
            flush=True,
        )

    by_id = {row["candidate_item_id"]: row for row in rows}
    ordered = [
        by_id[candidate["candidate_item_id"]]
        for candidate in candidate_rows
        if candidate["candidate_item_id"] in by_id
    ]
    final_rows = run_dir / "W02_Prometheus_Full_Judged_Rows.jsonl"
    write_jsonl(final_rows, ordered)
    trace_rows = prompt_trace(ordered)
    write_jsonl(run_dir / "W02_Prometheus_Prompt_Trace.jsonl", trace_rows)
    write_csv(run_dir / "W02_Prometheus_Full_Results.csv", ordered)

    summary = compute_summary(
        run_id=args.run_id,
        rows=ordered,
        calibration=calibration,
        score_status=score_status,
    )
    summary["judge_backend"] = ordered[0]["structured_judge"]["judge_backend"]
    (run_dir / "W02_Prometheus_Full_Summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    write_report(
        run_dir / "W02_Prometheus_Full_Report.md",
        summary=summary,
        rows=ordered,
    )
    write_manifest(
        run_dir / "W02_Prometheus_Full_Manifest.json",
        args=args,
        run_dir=run_dir,
        summary=summary,
    )
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "rows": len(ordered),
                "trace_rows": len(trace_rows),
                "score_status": score_status,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

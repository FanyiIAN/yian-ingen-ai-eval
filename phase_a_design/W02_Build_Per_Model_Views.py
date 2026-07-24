"""Build separate model views and diagnostic aggregates from the Week 2 master CSV.

The master CSV remains the lossless submission table required by the internship
plan.  This script creates model-specific views and never averages FLAN and
Mistral together.  Scores come from a Judge that failed calibration, so every
aggregate is explicitly diagnostic and retains resolution coverage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "W02_Baseline_Eval_Results.csv"
DEFAULT_JSON = ROOT / "W02_Per_Model_Diagnostic_Aggregates.json"
DEFAULT_CSV = ROOT / "W02_Per_Model_Diagnostic_Aggregates.csv"
DEFAULT_REPORT = ROOT / "W02_Per_Model_Diagnostic_Aggregates.md"
DEFAULT_FLAN = ROOT / "W02_Baseline_Eval_Results_FLAN.csv"
DEFAULT_MISTRAL = ROOT / "W02_Baseline_Eval_Results_Mistral.csv"
DEFAULT_MANIFEST = ROOT / "W02_Baseline_Run_Manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--flan-output", type=Path, default=DEFAULT_FLAN)
    parser.add_argument("--mistral-output", type=Path, default=DEFAULT_MISTRAL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    return float(value)


def mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return statistics.fmean(materialized) if materialized else None


def rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    task = [
        score
        for row in rows
        if (score := numeric(row["final_task_accuracy"])) is not None
    ]
    grounding = [
        score
        for row in rows
        if (score := numeric(row["final_contextual_grounding"])) is not None
    ]
    quality: list[float] = []
    for row in rows:
        task_score = numeric(row["final_task_accuracy"])
        grounding_score = numeric(row["final_contextual_grounding"])
        if task_score is not None and grounding_score is not None:
            quality.append((task_score + grounding_score) / 2)
    return {
        "n": len(rows),
        "task_resolved_n": len(task),
        "task_mean_resolved": rounded(mean(task)),
        "grounding_resolved_n": len(grounding),
        "grounding_mean_resolved": rounded(mean(grounding)),
        "quality_resolved_n": len(quality),
        "quality_mean_resolved": rounded(mean(quality)),
    }


def severity_weighted(
    rows: list[dict[str, str]],
    *,
    task_field: str | None = None,
    paired_quality: bool = False,
) -> dict[str, Any]:
    usable: list[tuple[float, float]] = []
    for row in rows:
        severity = numeric(row["severity"])
        if severity is None:
            continue
        if paired_quality:
            task = numeric(row["final_task_accuracy"])
            grounding = numeric(row["final_contextual_grounding"])
            if task is None or grounding is None:
                continue
            value = (task + grounding) / 2
        else:
            if task_field is None:
                raise ValueError("task_field is required for a single dimension")
            value = numeric(row[task_field])
            if value is None:
                continue
        usable.append((value, severity))

    denominator = sum(severity for _, severity in usable)
    weighted = (
        sum(value * severity for value, severity in usable) / denominator
        if denominator
        else None
    )
    return {
        "resolved_n": len(usable),
        "severity_weight_denominator": int(denominator),
        "value": rounded(weighted),
    }


def load_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def write_model_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def tidy_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for model_name, model in summary["models"].items():
        model_version = model["model_version"]

        def add(
            scope: str,
            group: str,
            metric: str,
            value: Any,
            resolved_n: int | None,
            total_n: int,
            note: str,
        ) -> None:
            records.append(
                {
                    "model_name": model_name,
                    "model_version": model_version,
                    "scope": scope,
                    "group": group,
                    "metric": metric,
                    "value": value,
                    "resolved_n": resolved_n,
                    "total_n": total_n,
                    "status": "diagnostic_failed_judge_calibration",
                    "note": note,
                }
            )

        for platform, values in model["by_platform"].items():
            add(
                "platform",
                platform,
                "task_mean_resolved",
                values["task_mean_resolved"],
                values["task_resolved_n"],
                values["n"],
                "Mean uses resolved conservative Judge consensus only.",
            )
            add(
                "platform",
                platform,
                "grounding_mean_resolved",
                values["grounding_mean_resolved"],
                values["grounding_resolved_n"],
                values["n"],
                "Mean uses resolved conservative Judge consensus only.",
            )
            add(
                "platform",
                platform,
                "quality_mean_resolved",
                values["quality_mean_resolved"],
                values["quality_resolved_n"],
                values["n"],
                "Quality=(Task+Grounding)/2 where both scores resolved.",
            )

        for split, values in model["by_split"].items():
            add(
                "split",
                split,
                "task_mean_resolved",
                values["task_mean_resolved"],
                values["task_resolved_n"],
                values["n"],
                "The original held_out subset is no longer blind.",
            )
            add(
                "split",
                split,
                "grounding_mean_resolved",
                values["grounding_mean_resolved"],
                values["grounding_resolved_n"],
                values["n"],
                "The original held_out subset is no longer blind.",
            )

        for metric, values in model["severity_weighted"].items():
            add(
                "overall",
                "severity_weighted",
                metric,
                values["value"],
                values["resolved_n"],
                model["overall"]["n"],
                (
                    "Sum(score*severity)/sum(severity), resolved rows only; "
                    f"weight denominator={values['severity_weight_denominator']}."
                ),
            )

        for label, count in model["failure_mode_distribution"].items():
            add(
                "failure_mode",
                label,
                "count",
                count,
                count,
                model["overall"]["n"],
                "Unresolved is retained as its own category.",
            )
    return records


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Week 2 Per-Model Diagnostic Aggregates",
        "",
        "> These tables never combine FLAN and Mistral model-performance scores. "
        "The Judge failed calibration, so all values are reference-only. Means use "
        "resolved conservative consensus rows and show coverage to expose selection bias.",
        "",
        f"- Source CSV SHA-256: `{summary['source_csv_sha256']}`",
        f"- Benchmark version: `{summary['benchmark_version']}`",
        "- Original scenario split: `28 development / 7 held_out`; the seven held-out "
        "scenarios were later inspected and are not a fresh blind test set.",
        "",
    ]
    for model_name, model in summary["models"].items():
        lines.extend(
            [
                f"## {model_name}",
                "",
                f"- Revision: `{model['model_version']}`",
                f"- Rows: `{model['overall']['n']}`",
                "",
                "### Per-platform diagnostic means",
                "",
                "| Platform | Task mean (resolved/N) | Grounding mean (resolved/N) | Quality mean (resolved/N) |",
                "|---|---:|---:|---:|",
            ]
        )
        for platform, values in model["by_platform"].items():
            lines.append(
                f"| {platform} | {fmt(values['task_mean_resolved'])} "
                f"({values['task_resolved_n']}/{values['n']}) | "
                f"{fmt(values['grounding_mean_resolved'])} "
                f"({values['grounding_resolved_n']}/{values['n']}) | "
                f"{fmt(values['quality_mean_resolved'])} "
                f"({values['quality_resolved_n']}/{values['n']}) |"
            )
        lines.extend(
            [
                "",
                "### Severity-weighted diagnostic aggregate",
                "",
                "| Dimension | Value | Resolved/N | Severity-weight denominator |",
                "|---|---:|---:|---:|",
            ]
        )
        for dimension, values in model["severity_weighted"].items():
            lines.append(
                f"| {dimension} | {fmt(values['value'])} | "
                f"{values['resolved_n']}/{model['overall']['n']} | "
                f"{values['severity_weight_denominator']} |"
            )
        lines.extend(
            [
                "",
                "### Failure-mode distribution",
                "",
                "| Failure mode | Count | Share of 35 |",
                "|---|---:|---:|",
            ]
        )
        for label, count in model["failure_mode_distribution"].items():
            lines.append(f"| {label} | {count} | {count / 35:.1%} |")
        lines.extend(
            [
                "",
                "### Original split diagnostic",
                "",
                "| Split | N | Task mean (resolved/N) | Grounding mean (resolved/N) |",
                "|---|---:|---:|---:|",
            ]
        )
        for split, values in model["by_split"].items():
            lines.append(
                f"| {split} | {values['n']} | {fmt(values['task_mean_resolved'])} "
                f"({values['task_resolved_n']}/{values['n']}) | "
                f"{fmt(values['grounding_mean_resolved'])} "
                f"({values['grounding_resolved_n']}/{values['n']}) |"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation limits",
            "",
            "- Do not compare the model means as validated leaderboard scores.",
            "- FLAN has extensive unresolved consensus and one-shot copying; its resolved "
            "subset is highly selected and can produce misleadingly high platform means.",
            "- Prometheus frequently misclassified safe boundaries as `refusal`; failure-mode "
            "counts are Judge outputs, not adjudicated truth.",
            "- A new untouched test set and independently reviewed human gold are required "
            "before model-selection claims.",
            "",
        ]
    )
    return "\n".join(lines)


def update_manifest(
    manifest_path: Path,
    artifact_paths: list[Path],
) -> None:
    """Register postprocessed views without making the manifest self-referential."""
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Run manifest not found; finalize the master CSV first: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("source", {}).update(
        {
            "per_model_view_builder": Path(__file__).name,
            "per_model_view_builder_sha256": sha256_file(Path(__file__)),
        }
    )
    artifacts = manifest.setdefault("artifacts", {})
    for artifact_path in artifact_paths:
        artifacts[artifact_path.name] = sha256_file(artifact_path)
    manifest["aggregation_policy"] = {
        "model_performance": "never_average_models_together",
        "scores": "resolved_conservative_consensus_only",
        "status": "diagnostic_failed_judge_calibration",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    fieldnames, rows = load_rows(input_path)
    if len(rows) != 70:
        raise ValueError(f"Expected 70 master rows, found {len(rows)}")

    required = {
        "model_name",
        "model_version",
        "scenario_id",
        "platform",
        "split",
        "severity",
        "final_task_accuracy",
        "final_contextual_grounding",
        "final_failure_mode",
    }
    missing = required - set(fieldnames)
    if missing:
        raise ValueError(f"Missing master CSV columns: {sorted(missing)}")

    model_outputs = {
        "google/flan-t5-base": args.flan_output.resolve(),
        "mistralai/Mistral-7B-Instruct-v0.2": args.mistral_output.resolve(),
    }
    models = sorted({row["model_name"] for row in rows})
    if set(models) != set(model_outputs):
        raise ValueError(f"Unexpected model set: {models}")

    summary: dict[str, Any] = {
        "status": "diagnostic_failed_judge_calibration",
        "source_csv": input_path.name,
        "source_csv_sha256": sha256_file(input_path),
        "benchmark_version": sorted({row["benchmark_version"] for row in rows})[0],
        "model_aggregation_policy": "never_average_models_together",
        "score_policy": "resolved_conservative_consensus_only",
        "models": {},
    }

    for model_name in models:
        model_rows = [row for row in rows if row["model_name"] == model_name]
        if len(model_rows) != 35:
            raise ValueError(f"{model_name}: expected 35 rows, found {len(model_rows)}")
        write_model_csv(model_outputs[model_name], fieldnames, model_rows)
        model_summary = {
            "model_version": sorted({row["model_version"] for row in model_rows})[0],
            "overall": summarize(model_rows),
            "by_platform": {
                platform: summarize(
                    [row for row in model_rows if row["platform"] == platform]
                )
                for platform in sorted({row["platform"] for row in model_rows})
            },
            "by_split": {
                split: summarize([row for row in model_rows if row["split"] == split])
                for split in ("development", "held_out")
            },
            "severity_weighted": {
                "task_accuracy": severity_weighted(
                    model_rows, task_field="final_task_accuracy"
                ),
                "contextual_grounding": severity_weighted(
                    model_rows, task_field="final_contextual_grounding"
                ),
                "paired_quality": severity_weighted(
                    model_rows, paired_quality=True
                ),
            },
            "failure_mode_distribution": dict(
                sorted(
                    Counter(
                        row["final_failure_mode"] or "unresolved"
                        for row in model_rows
                    ).items()
                )
            ),
        }
        summary["models"][model_name] = model_summary

    args.json_output.resolve().write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    records = tidy_rows(summary)
    aggregate_fields = [
        "model_name",
        "model_version",
        "scope",
        "group",
        "metric",
        "value",
        "resolved_n",
        "total_n",
        "status",
        "note",
    ]
    with args.csv_output.resolve().open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=aggregate_fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)

    args.report_output.resolve().write_text(
        render_report(summary),
        encoding="utf-8",
    )
    output_paths = [
        model_outputs["google/flan-t5-base"],
        model_outputs["mistralai/Mistral-7B-Instruct-v0.2"],
        args.csv_output.resolve(),
        args.json_output.resolve(),
        args.report_output.resolve(),
    ]
    update_manifest(args.manifest.resolve(), output_paths)

    print(
        json.dumps(
            {
                "source_rows": len(rows),
                "models": {model: 35 for model in models},
                "outputs": {
                    "flan_csv": str(model_outputs["google/flan-t5-base"]),
                    "mistral_csv": str(
                        model_outputs["mistralai/Mistral-7B-Instruct-v0.2"]
                    ),
                    "aggregate_csv": str(args.csv_output.resolve()),
                    "aggregate_json": str(args.json_output.resolve()),
                    "aggregate_report": str(args.report_output.resolve()),
                    "manifest": str(args.manifest.resolve()),
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

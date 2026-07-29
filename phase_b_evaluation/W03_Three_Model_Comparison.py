"""Create a contract-checked three-model diagnostic comparison for Week 3.

The Week 2 Prometheus Judge failed calibration. This script therefore preserves
its scores as diagnostic evidence, exposes score coverage, and never promotes
the resulting ordering to a validated model-quality leaderboard.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


RUNNER_VERSION = "0.1.0"
BENCHMARK_VERSION = "0.2.0"
PROMPT_VERSION = "0.4.0"
PROMPT_SPEC_SHA256 = (
    "0bb0a6f2e298f286739080752540939454e2e5e52c0dca477e17196657cac71d"
)
SEED = 42
EXPECTED_MODELS = {
    "google/flan-t5-base": "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2",
    "mistralai/Mistral-7B-Instruct-v0.2": (
        "63a8b081895390a26e140280378bc85ec8bce07a"
    ),
    "meta-llama/Llama-3.1-8B-Instruct": (
        "0e9e39f249a16976918f6564b8830bc894c89659"
    ),
}
FAILURE_LABELS = (
    "unsafe",
    "hallucination",
    "off_policy",
    "refusal",
    "partial",
    "none",
    "unresolved",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def optional_int(value: Any) -> int | None:
    if value in (None, "", "null"):
        return None
    return int(value)


def load_week2_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for source in csv.DictReader(handle):
            rows.append(
                {
                    "scenario_id": source["scenario_id"],
                    "platform": source["platform"],
                    "split": source["split"],
                    "severity": int(source["severity"]),
                    "model_id": source["model_name"],
                    "model_revision": source["model_version"],
                    "prompt_version": source["prompt_version"],
                    "prompt_spec_sha256": source["prompt_spec_sha256"],
                    "prompt_sha256": source["prompt_sha256"],
                    "seed": int(source["random_seed"]),
                    "task": optional_int(source["final_task_accuracy"]),
                    "grounding": optional_int(
                        source["final_contextual_grounding"]
                    ),
                    "failure": source["final_failure_mode"] or None,
                    "output_tokens": int(source["output_tokens"]),
                    "latency_ms": float(source["latency_ms"]),
                    "score_status": source["score_status"],
                    "generation_error": None,
                    "input_truncated": (
                        int(source["input_tokens"])
                        > int(source["max_input_tokens"])
                    ),
                }
            )
    return rows


def load_llama_rows(path: Path) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for source in read_jsonl(path):
        judge = source["structured_judge"]
        consensus = judge["consensus"]
        normalized.append(
            {
                "scenario_id": source["scenario_id"],
                "platform": source["platform"],
                "split": source["split"],
                "severity": int(source["severity_class"]),
                "model_id": source["candidate_model_id"],
                "model_revision": source["candidate_model_revision"],
                "prompt_version": str(source["candidate_prompt_version"]),
                "prompt_spec_sha256": source[
                    "candidate_prompt_spec_sha256"
                ],
                "prompt_sha256": source["candidate_prompt_sha256"],
                "seed": int(source["candidate_generation"]["seed"]),
                "task": consensus["task_accuracy"]["final"],
                "grounding": consensus["contextual_grounding"]["final"],
                "failure": consensus["primary_failure_mode"]["final"],
                "output_tokens": int(source["candidate_output_tokens"]),
                "latency_ms": float(source["candidate_latency_ms"]),
                "score_status": source["score_status"],
                "generation_error": source.get("generation_error"),
                "input_truncated": bool(
                    source.get("candidate_input_truncated", False)
                ),
            }
        )
    return normalized


def validate_contract(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prompts_by_scenario: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_model[row["model_id"]].append(row)
        prompts_by_scenario[row["scenario_id"]].add(row["prompt_sha256"])
        expected_revision = EXPECTED_MODELS.get(row["model_id"])
        if expected_revision is None:
            errors.append(f"Unexpected model: {row['model_id']}")
        elif row["model_revision"] != expected_revision:
            errors.append(f"{row['model_id']}: revision mismatch")
        if row["prompt_version"] != PROMPT_VERSION:
            errors.append(f"{row['model_id']}: prompt version mismatch")
        if row["prompt_spec_sha256"] != PROMPT_SPEC_SHA256:
            errors.append(f"{row['model_id']}: prompt-spec hash mismatch")
        if row["seed"] != SEED:
            errors.append(f"{row['model_id']}: seed mismatch")
        if row["generation_error"]:
            errors.append(
                f"{row['model_id']}::{row['scenario_id']}: generation error"
            )
        if row["input_truncated"]:
            errors.append(
                f"{row['model_id']}::{row['scenario_id']}: input truncated"
            )
        if row["score_status"] != "diagnostic_failed_calibration":
            errors.append(
                f"{row['model_id']}::{row['scenario_id']}: unexpected score status"
            )
    if set(by_model) != set(EXPECTED_MODELS):
        errors.append("The comparison does not contain the three frozen models")
    for model_id, model_rows in by_model.items():
        ids = [row["scenario_id"] for row in model_rows]
        if len(ids) != 35 or len(set(ids)) != 35:
            errors.append(f"{model_id}: expected 35 unique scenarios")
    if len(prompts_by_scenario) != 35:
        errors.append("Expected 35 scenario IDs across the comparison")
    mismatched_prompts = [
        scenario_id
        for scenario_id, hashes in prompts_by_scenario.items()
        if len(hashes) != 1
    ]
    if mismatched_prompts:
        errors.append(
            "Semantic prompt mismatch for: " + ", ".join(mismatched_prompts)
        )
    if errors:
        raise ValueError("Three-model contract failed:\n- " + "\n- ".join(errors))
    return {
        "status": "ok",
        "row_count": len(rows),
        "model_count": len(by_model),
        "scenario_count": len(prompts_by_scenario),
        "identical_semantic_prompt_per_scenario": True,
        "generation_error_count": 0,
        "input_truncation_count": 0,
        "seed": SEED,
        "benchmark_version": BENCHMARK_VERSION,
        "prompt_version": PROMPT_VERSION,
        "prompt_spec_sha256": PROMPT_SPEC_SHA256,
    }


def mean_or_none(values: Iterable[float]) -> float | None:
    items = list(values)
    return statistics.fmean(items) if items else None


def resolved_values(
    rows: list[dict[str, Any]], field: str
) -> list[tuple[float, int]]:
    return [
        (float(row[field]), int(row["severity"]))
        for row in rows
        if row[field] is not None
    ]


def score_stats(
    rows: list[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    values = resolved_values(rows, field)
    weight = sum(severity for _, severity in values)
    return {
        "resolved_n": len(values),
        "total_n": len(rows),
        "mean": mean_or_none(value for value, _ in values),
        "severity_weighted_mean": (
            sum(value * severity for value, severity in values) / weight
            if weight
            else None
        ),
        "severity_weight_denominator": weight,
    }


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    quality_rows = [
        {
            **row,
            "quality": (float(row["task"]) + float(row["grounding"])) / 2,
        }
        for row in rows
        if row["task"] is not None and row["grounding"] is not None
    ]
    failures = Counter(
        row["failure"] if row["failure"] is not None else "unresolved"
        for row in rows
    )
    total_latency_seconds = sum(row["latency_ms"] for row in rows) / 1000
    severity5 = [row for row in rows if row["severity"] == 5]
    severity5_resolved = [
        row for row in severity5 if row["task"] is not None
    ]
    return {
        "n": len(rows),
        "task": score_stats(rows, "task"),
        "grounding": score_stats(rows, "grounding"),
        "quality": score_stats(quality_rows, "quality"),
        "failure_counts": {
            label: failures.get(label, 0) for label in FAILURE_LABELS
        },
        "latency": {
            "mean_ms": mean_or_none(row["latency_ms"] for row in rows),
            "median_ms": statistics.median(
                row["latency_ms"] for row in rows
            ),
            "aggregate_output_tokens_per_second": (
                sum(row["output_tokens"] for row in rows)
                / total_latency_seconds
                if total_latency_seconds
                else None
            ),
            "output_tokens": sum(row["output_tokens"] for row in rows),
        },
        "severity_5": {
            "n": len(severity5),
            "task_resolved_n": len(severity5_resolved),
            "task_le_2_count": sum(
                int(row["task"]) <= 2 for row in severity5_resolved
            ),
            "unsafe_count": sum(
                row["failure"] == "unsafe" for row in severity5
            ),
        },
    }


def rounded(value: Any, digits: int = 4) -> Any:
    return round(value, digits) if isinstance(value, float) else value


def build_summary(
    rows: list[dict[str, Any]],
    contract: dict[str, Any],
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[row["model_id"]].append(row)
    models: dict[str, Any] = {}
    for model_id, model_rows in sorted(by_model.items()):
        by_platform = defaultdict(list)
        for row in model_rows:
            by_platform[row["platform"]].append(row)
        models[model_id] = {
            "revision": EXPECTED_MODELS[model_id],
            "overall": summarize_group(model_rows),
            "by_platform": {
                platform: summarize_group(platform_rows)
                for platform, platform_rows in sorted(by_platform.items())
            },
        }
    llama = models["meta-llama/Llama-3.1-8B-Instruct"]["overall"]
    deltas: dict[str, Any] = {}
    for baseline_id in (
        "google/flan-t5-base",
        "mistralai/Mistral-7B-Instruct-v0.2",
    ):
        baseline = models[baseline_id]["overall"]
        deltas[baseline_id] = {
            "severity_weighted_task": (
                llama["task"]["severity_weighted_mean"]
                - baseline["task"]["severity_weighted_mean"]
            ),
            "severity_weighted_grounding": (
                llama["grounding"]["severity_weighted_mean"]
                - baseline["grounding"]["severity_weighted_mean"]
            ),
            "severity_weighted_quality": (
                llama["quality"]["severity_weighted_mean"]
                - baseline["quality"]["severity_weighted_mean"]
            ),
            "output_tokens_per_second": (
                llama["latency"]["aggregate_output_tokens_per_second"]
                - baseline["latency"]["aggregate_output_tokens_per_second"]
            ),
        }
    return {
        "comparison_id": "w03_three_model_prometheus_diagnostic_v0.1.0",
        "runner_version": RUNNER_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "diagnostic_failed_judge_calibration",
        "validated_model_quality_claim_allowed": False,
        "claim_boundary": (
            "L0 synthetic product-context simulation only. Prometheus failed "
            "calibration; all score ordering is diagnostic and requires human "
            "adjudication."
        ),
        "contract": contract,
        "source_sha256": source_hashes,
        "models": models,
        "llama_deltas_vs_prior_models": deltas,
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def aggregate_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_id, model in summary["models"].items():
        scopes = {"overall": model["overall"], **model["by_platform"]}
        for scope, metrics in scopes.items():
            rows.append(
                {
                    "model_id": model_id,
                    "model_revision": model["revision"],
                    "scope": scope,
                    "n": metrics["n"],
                    "task_mean": rounded(metrics["task"]["mean"]),
                    "task_resolved_n": metrics["task"]["resolved_n"],
                    "task_severity_weighted": rounded(
                        metrics["task"]["severity_weighted_mean"]
                    ),
                    "grounding_mean": rounded(
                        metrics["grounding"]["mean"]
                    ),
                    "grounding_resolved_n": metrics["grounding"][
                        "resolved_n"
                    ],
                    "grounding_severity_weighted": rounded(
                        metrics["grounding"]["severity_weighted_mean"]
                    ),
                    "quality_mean": rounded(metrics["quality"]["mean"]),
                    "quality_resolved_n": metrics["quality"]["resolved_n"],
                    "quality_severity_weighted": rounded(
                        metrics["quality"]["severity_weighted_mean"]
                    ),
                    "mean_latency_ms": rounded(
                        metrics["latency"]["mean_ms"]
                    ),
                    "output_tokens_per_second": rounded(
                        metrics["latency"][
                            "aggregate_output_tokens_per_second"
                        ]
                    ),
                    **{
                        f"failure_{label}": metrics["failure_counts"][label]
                        for label in FAILURE_LABELS
                    },
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Week 3 Three-Model Diagnostic Comparison",
        "",
        "> The Prometheus Judge failed its frozen calibration. Scores below are",
        "> diagnostic, coverage-sensitive evidence and are not a validated model",
        "> quality leaderboard. Human review remains required for all 105 rows and",
        "> every severity-5 response.",
        "",
        f"- Benchmark: `{BENCHMARK_VERSION}` (35 synthetic scenarios)",
        f"- Candidate prompt: `{PROMPT_VERSION}` / `{PROMPT_SPEC_SHA256}`",
        f"- Seed: `{SEED}`; deterministic decoding; no input truncation",
        "- Original split: 28 development / 7 formerly held-out; the seven were",
        "  inspected during Week 2 and are no longer blind test evidence.",
        "",
        "## Overall diagnostic comparison",
        "",
        "| Model | SW Task (resolved/N) | SW Grounding (resolved/N) | "
        "SW Quality (resolved/N) | Output tok/s | Mean latency ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model_id, model in summary["models"].items():
        overall = model["overall"]
        lines.append(
            f"| `{model_id}` | "
            f"{fmt(overall['task']['severity_weighted_mean'])} "
            f"({overall['task']['resolved_n']}/{overall['n']}) | "
            f"{fmt(overall['grounding']['severity_weighted_mean'])} "
            f"({overall['grounding']['resolved_n']}/{overall['n']}) | "
            f"{fmt(overall['quality']['severity_weighted_mean'])} "
            f"({overall['quality']['resolved_n']}/{overall['n']}) | "
            f"{fmt(overall['latency']['aggregate_output_tokens_per_second'])} | "
            f"{fmt(overall['latency']['mean_ms'])} |"
        )
    lines.extend(
        [
            "",
            "SW means severity-weighted. A higher mean with much lower resolved",
            "coverage can reflect selection bias, so coverage is part of every cell.",
            "",
            "## Llama diagnostic deltas",
            "",
            "| Baseline | Δ SW Task | Δ SW Grounding | Δ SW Quality | Δ tok/s |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model_id, delta in summary[
        "llama_deltas_vs_prior_models"
    ].items():
        lines.append(
            f"| `{model_id}` | "
            f"{fmt(delta['severity_weighted_task'])} | "
            f"{fmt(delta['severity_weighted_grounding'])} | "
            f"{fmt(delta['severity_weighted_quality'])} | "
            f"{fmt(delta['output_tokens_per_second'])} |"
        )
    lines.extend(
        [
            "",
            "These deltas answer only what the failed-calibration diagnostic Judge",
            "reported under the frozen protocol. They do not establish that Llama is",
            "better until the rows receive model-blind human adjudication.",
            "",
            "## Per-platform diagnostic quality",
            "",
            "| Model | Fari | Senpai | Sentinel | Rover | Humanoid |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    platform_order = (
        "Fari",
        "Senpai",
        "Sentinel_Prime_AI",
        "Aido_Rover",
        "Aido_Humanoid",
    )
    for model_id, model in summary["models"].items():
        cells = []
        for platform in platform_order:
            metric = model["by_platform"][platform]["quality"]
            cells.append(
                f"{fmt(metric['severity_weighted_mean'])} "
                f"({metric['resolved_n']}/7)"
            )
        lines.append(f"| `{model_id}` | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Overall failure distribution and severity-5 flags",
            "",
            "| Model | Unsafe | Hallucination | Off-policy | Refusal | "
            "Partial | None | Unresolved | Sev-5 task <=2 | Sev-5 unsafe |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model_id, model in summary["models"].items():
        overall = model["overall"]
        failures = overall["failure_counts"]
        severity5 = overall["severity_5"]
        lines.append(
            f"| `{model_id}` | "
            f"{failures['unsafe']} | "
            f"{failures['hallucination']} | "
            f"{failures['off_policy']} | "
            f"{failures['refusal']} | "
            f"{failures['partial']} | "
            f"{failures['none']} | "
            f"{failures['unresolved']} | "
            f"{severity5['task_le_2_count']}"
            f"/{severity5['task_resolved_n']} | "
            f"{severity5['unsafe_count']}/{severity5['n']} |"
        )
    lines.extend(
        [
            "",
            "## Required interpretation",
            "",
            "- Observation: report the score, coverage, latency, and failure counts.",
            "- Mechanism: inspect row-level evidence before explaining why a model",
            "  behaved differently; architecture size alone is not an explanation.",
            "- Deployment boundary: these are independent open models on synthetic",
            "  text scenarios, not deployed InGen products or PIC runtime results.",
            "- Next validity gate: model-blind human adjudication, including all",
            "  severity-5 rows, followed by a freshly sealed held-out set.",
            "",
        ]
    )
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week2-csv", type=Path, required=True)
    parser.add_argument("--llama-judged-rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    week2_rows = load_week2_rows(args.week2_csv)
    llama_rows = load_llama_rows(args.llama_judged_rows)
    rows = week2_rows + llama_rows
    contract = validate_contract(rows)
    summary = build_summary(
        rows,
        contract,
        {
            "week2_csv": sha256_file(args.week2_csv),
            "llama_judged_rows": sha256_file(args.llama_judged_rows),
        },
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = (
        args.output_dir / "W03_Three_Model_Diagnostic_Summary.json"
    )
    with summary_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    write_csv(
        args.output_dir / "W03_Three_Model_Diagnostic_Comparison.csv",
        aggregate_rows(summary),
    )
    write_report(
        args.output_dir / "W03_Three_Model_Diagnostic_Report.md",
        summary,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "contract": contract,
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

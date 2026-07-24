"""Audit a completed Week 2 Mistral GPU run and render row-level evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import jsonschema


ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = (
    ROOT
    / "experiments"
    / "w02_mistral_pipeline"
    / "mistral-full-v0.2.0"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    return parser.parse_args()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def stable_behavior_payload(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    volatile_keys = {
        "run_id",
        "timestamp_utc",
        "generation_latency_ms",
    }

    def scrub(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: scrub(item)
                for key, item in sorted(value.items())
                if key not in volatile_keys
                and key != "latency_ms"
                and not key.endswith("_latency_ms")
            }
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return value

    return [scrub(row) for row in rows]


def render_trace(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Week 2 Mistral Full Trace",
        "",
        "> Exact synthetic inputs, rendered prompts, model outputs, automated judge",
        "> diagnostics, and review flags. Automated scores are provisional because the",
        "> same checkpoint is candidate and judge and was not human-calibrated.",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['scenario_id']} — {row['platform']}",
                "",
                f"- Split / severity: `{row['split']}` / `{row['severity_class']}`",
                f"- Candidate model: `{row['candidate_model_id']}`",
                f"- Revision: `{row['candidate_model_revision']}`",
                f"- Seed / precision / device: `{row['seed']}` / `{row['precision']}` / `{row['device']}`",
                f"- Candidate prompt version/hash: `{row['prompt_template_version']}` / `{row['candidate_prompt_hash']}`",
                f"- Tokens input/output/truncated: `{row['candidate_input_tokens']}` / `{row['candidate_output_tokens']}` / `{row['candidate_input_truncated']}`",
                "",
                "### Input stimulus",
                "",
                "```text",
                row["input_stimulus"],
                "```",
                "",
                "### Exact candidate prompt",
                "",
                "```text",
                row["candidate_prompt"],
                "```",
                "",
                "### Candidate output",
                "",
                "```text",
                row["raw_output"] or "[empty output]",
                "```",
                "",
                "### Deterministic audit",
                "",
                "```json",
                json.dumps(row["deterministic_audit"], ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
        prompts_by_formulation = row["judge_rendered_prompts"]
        for judge in row["judge_results"]:
            name = judge["formulation"]
            lines.extend(
                [
                    f"### Judge: {name}",
                    "",
                    f"Prompt version: `{judge['prompt_version']}`",
                    "",
                ]
            )
            mappings = (
                ("task_accuracy", "Task Accuracy", "task_accuracy_classification"),
                (
                    "contextual_grounding",
                    "Contextual Grounding",
                    "contextual_grounding_classification",
                ),
                ("failure_mode", "Primary Failure Mode", "failure_mode_classification"),
            )
            for prompt_key, label, result_key in mappings:
                result = judge[result_key]
                lines.extend(
                    [
                        f"#### {label} prompt",
                        "",
                        "```text",
                        prompts_by_formulation[name][prompt_key],
                        "```",
                        "",
                        f"- Likelihood selection: `{result['selected']}`",
                        f"- Losses: `{json.dumps(result['losses'], sort_keys=True)}`",
                        f"- Margin to second: `{result['margin_to_second']}`",
                        f"- Generated-label selection: `{result['generated_label_selected']}`",
                        f"- Generated-label parse: `{result['generated_label_parse_status']}`",
                        f"- Likelihood/generation agree: `{result['likelihood_generation_agree']}`",
                        "",
                        "Generated label text:",
                        "",
                        "```text",
                        result["generated_label_raw"],
                        "```",
                        "",
                    ]
                )
            lines.extend(
                [
                    "#### Rationale prompt",
                    "",
                    "```text",
                    prompts_by_formulation[name]["rationale"],
                    "```",
                    "",
                    "Rationale output:",
                    "",
                    "```text",
                    judge["rationale"],
                    "```",
                    "",
                ]
            )
        lines.extend(
            [
                "### Aggregation and review",
                "",
                f"- Task ratings -> median: `{row['task_accuracy_ratings']}` -> `{row['final_task_accuracy']}`",
                f"- Grounding ratings -> median: `{row['contextual_grounding_ratings']}` -> `{row['final_contextual_grounding']}`",
                f"- Failure votes -> final: `{row['primary_failure_mode_ratings']}` -> `{row['final_primary_failure_mode']}`",
                f"- Human review required: `{row['human_review_required']}`",
                f"- Review reasons: `{row['human_review_reasons']}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    rows_path = run_dir / "W02_Mistral_GPU_Integration_Rows.jsonl"
    summary_path = run_dir / "W02_Mistral_GPU_Integration_Summary.json"
    schema_path = ROOT / "W02_Result_Schema.json"
    rows = read_jsonl(rows_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    schema_errors = []
    candidate_hash_errors = []
    judge_hash_errors = []
    for row in rows:
        try:
            jsonschema.validate(row, schema)
        except jsonschema.ValidationError as exc:
            schema_errors.append({"scenario_id": row.get("scenario_id"), "error": exc.message})
        if sha256_text(row["candidate_prompt"]) != row["candidate_prompt_hash"]:
            candidate_hash_errors.append(row["scenario_id"])
        for formulation, prompts in row["judge_rendered_prompts"].items():
            for purpose, prompt in prompts.items():
                expected = row["judge_prompt_hashes"][formulation][purpose]
                if sha256_text(prompt) != expected:
                    judge_hash_errors.append(
                        f"{row['scenario_id']}:{formulation}:{purpose}"
                    )

    classifications = [
        judge[key]
        for row in rows
        for judge in row["judge_results"]
        for key in (
            "task_accuracy_classification",
            "contextual_grounding_classification",
            "failure_mode_classification",
        )
    ]
    parsed = [
        item for item in classifications if item["generated_label_parse_status"] == "parsed"
    ]
    likelihood_generation_disagreements = [
        item for item in parsed if not item["likelihood_generation_agree"]
    ]
    exact_task_agreement = sum(
        len(set(row["task_accuracy_ratings"])) == 1 for row in rows
    )
    exact_grounding_agreement = sum(
        len(set(row["contextual_grounding_ratings"])) == 1 for row in rows
    )
    exact_failure_agreement = sum(
        len(set(row["primary_failure_mode_ratings"])) == 1 for row in rows
    )
    all_missing_but_passed = [
        row["scenario_id"]
        for row in rows
        if row["deterministic_audit"]["missing_required_lexical_signals"]
        and len(row["deterministic_audit"]["missing_required_lexical_signals"])
        == len(row["deterministic_audit"]["required_concept_results"])
        and row["final_task_accuracy"] is not None
        and row["final_task_accuracy"] >= 4
    ]

    artifact_hash_errors = []
    artifact_hashes = summary["benchmark_artifact_sha256"]
    for name, expected in artifact_hashes.items():
        local_path = ROOT / name
        if not local_path.exists() or sha256_file(local_path) != expected:
            artifact_hash_errors.append(name)

    output_lengths = [len(row["raw_output"].split()) for row in rows]
    generated_parse_counts = Counter(
        item["generated_label_parse_status"] for item in classifications
    )
    review_reason_counts = Counter(
        reason for row in rows for reason in row["human_review_reasons"]
    )
    analysis = {
        "run_id": summary["run_id"],
        "model_id": summary["model_id"],
        "model_revision": summary["model_revision"],
        "benchmark_version": summary["benchmark_version"],
        "seed": summary["seed"],
        "row_count": len(rows),
        "unique_scenario_count": len({row["scenario_id"] for row in rows}),
        "generation_errors": sum(bool(row["generation_error"]) for row in rows),
        "candidate_prompt_hash_errors": candidate_hash_errors,
        "judge_prompt_hash_errors": judge_hash_errors,
        "schema_errors": schema_errors,
        "benchmark_artifact_hash_errors": artifact_hash_errors,
        "output_word_count": {
            "min": min(output_lengths),
            "median": statistics.median(output_lengths),
            "max": max(output_lengths),
            "mean": round(statistics.mean(output_lengths), 4),
        },
        "candidate_input_truncation_rows": sum(
            row["candidate_input_truncated"] for row in rows
        ),
        "judge_prompt_truncation_rows": sum(
            any(judge["any_prompt_truncated"] for judge in row["judge_results"])
            for row in rows
        ),
        "exact_three_formulation_agreement": {
            "task_accuracy": exact_task_agreement,
            "contextual_grounding": exact_grounding_agreement,
            "primary_failure_mode": exact_failure_agreement,
        },
        "classification_diagnostics": {
            "total": len(classifications),
            "generated_parse_status": dict(sorted(generated_parse_counts.items())),
            "parsed_count": len(parsed),
            "likelihood_generation_disagreement_count_among_parsed": len(
                likelihood_generation_disagreements
            ),
            "likelihood_generation_agreement_rate_among_parsed": round(
                1 - len(likelihood_generation_disagreements) / len(parsed), 6
            )
            if parsed
            else None,
        },
        "human_review_required": sum(row["human_review_required"] for row in rows),
        "review_reason_counts": dict(sorted(review_reason_counts.items())),
        "all_required_lexical_signals_missing_but_median_task_at_least_4": all_missing_but_passed,
        "severity_5_median_task_at_or_below_2": [
            row["scenario_id"]
            for row in rows
            if row["severity_class"] == 5
            and row["final_task_accuracy"] is not None
            and row["final_task_accuracy"] <= 2
        ],
        "final_failure_mode_counts": dict(
            sorted(Counter(row["final_primary_failure_mode"] for row in rows).items())
        ),
        "stable_behavior_sha256": sha256_text(
            json.dumps(
                stable_behavior_payload(rows),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        "artifact_sha256": {
            name: sha256_file(run_dir / name)
            for name in (
                "rows.checkpoint.jsonl",
                "W02_Mistral_GPU_Integration_Report.md",
                "W02_Mistral_GPU_Integration_Results.csv",
                "W02_Mistral_GPU_Integration_Rows.jsonl",
                "W02_Mistral_GPU_Integration_Summary.json",
            )
        },
    }
    analysis_path = run_dir / "W02_Mistral_Output_Analysis.json"
    analysis_path.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    render_trace(run_dir / "W02_Mistral_Full_Trace.md", rows)

    if (
        len(rows) != 35
        or len({row["scenario_id"] for row in rows}) != 35
        or schema_errors
        or candidate_hash_errors
        or judge_hash_errors
        or artifact_hash_errors
    ):
        print(json.dumps(analysis, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(analysis, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

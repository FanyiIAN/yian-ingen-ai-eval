"""Freeze the Week 2 two-model / three-Judge diagnostic deliverables.

This is the finalization stage of the reproducible pipeline.  It consumes the
versioned YAML benchmark and the lossless JSONL produced by the GPU inference
and Judge stages, validates the experimental contract, computes agreement over
all responses, and writes the submission CSV plus an auditable summary.

The automated scores remain diagnostic whenever the supplied Judge run failed
calibration; this script never converts an uncalibrated score into a validated
human-equivalent result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import krippendorff
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent
FORMULATIONS = ("criterion_first", "evidence_first", "failure_first")
FAILURE_LABELS = ("unsafe", "hallucination", "off_policy", "refusal", "partial", "none")
RUNNER_VERSION = "1.0.0"
REPRODUCIBILITY_SOURCE_PATHS = (
    ROOT / "W02_Baseline_Pipeline.py",
    ROOT / "W02_Two_Model_Structured_Full_Run.py",
    ROOT / "W02_Eval_Runner.py",
    ROOT / "W02_Mistral_Eval_Runner.py",
    ROOT / "W02_Structured_Judge.py",
    ROOT / "W02_Prometheus_Judge.py",
    ROOT / "W02_Prometheus_Full_Run.py",
    ROOT / "W02_Prometheus_Judge_Spec_v0.8.3.yaml",
    ROOT / "W02_Judge_Requirement_Metadata_v0.4.0.yaml",
    ROOT / "W02_Deterministic_Checks.yaml",
    ROOT / "W02_Product_Regulations.yaml",
    ROOT / "W02_Result_Schema.json",
    ROOT / "W02_Unified_Run_Config_v1.0.0.yaml",
    ROOT.parent / "requirements.txt",
    ROOT.parent / "requirements-runpod-mistral.txt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judged-rows", type=Path, required=True)
    parser.add_argument("--scenario-path", type=Path, default=ROOT / "W02_Scenarios.yaml")
    parser.add_argument("--rubric-path", type=Path, default=ROOT / "W02_Rubric.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--expected-models", type=int, default=2)
    parser.add_argument("--expected-scenarios", type=int, default=35)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    return rows


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def formulation_by_name(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    items = row["structured_judge"]["formulation_results"]
    result = {str(item["formulation"]): item for item in items}
    if set(result) != set(FORMULATIONS):
        raise ValueError(
            f"{row['candidate_item_id']}: expected formulations {FORMULATIONS}, "
            f"found {sorted(result)}"
        )
    return result


def rating_matrix(
    rows: list[dict[str, Any]],
    dimension: str,
) -> np.ndarray:
    values: list[list[float]] = [[] for _ in FORMULATIONS]
    for row in rows:
        by_name = formulation_by_name(row)
        for index, name in enumerate(FORMULATIONS):
            value = by_name[name]["deterministic_mapping"][dimension]
            values[index].append(np.nan if value is None else float(value))
    return np.asarray(values, dtype=float)


def ordinal_alpha(matrix: np.ndarray) -> float | None:
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


def nominal_alpha(labels: list[list[str | None]]) -> float | None:
    mapping = {label: index for index, label in enumerate(FAILURE_LABELS)}
    matrix = np.asarray(
        [
            [np.nan if value is None else float(mapping[value]) for value in rater]
            for rater in labels
        ],
        dtype=float,
    )
    try:
        value = float(
            krippendorff.alpha(
                reliability_data=matrix,
                level_of_measurement="nominal",
            )
        )
    except (ValueError, ZeroDivisionError):
        return None
    return value if math.isfinite(value) else None


def agreement_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    task = rating_matrix(rows, "task_accuracy")
    grounding = rating_matrix(rows, "contextual_grounding")
    failure: list[list[str | None]] = [[] for _ in FORMULATIONS]
    exact_task = exact_grounding = exact_failure = 0
    task_within_one = grounding_within_one = 0
    consensus_unresolved = {
        "task_accuracy": 0,
        "contextual_grounding": 0,
        "failure_mode": 0,
    }

    for row in rows:
        by_name = formulation_by_name(row)
        consensus = row["structured_judge"]["consensus"]
        consensus_unresolved["task_accuracy"] += (
            consensus["task_accuracy"]["final"] is None
        )
        consensus_unresolved["contextual_grounding"] += (
            consensus["contextual_grounding"]["final"] is None
        )
        consensus_unresolved["failure_mode"] += (
            consensus["primary_failure_mode"]["final"] is None
        )
        task_values = [
            by_name[name]["deterministic_mapping"]["task_accuracy"]
            for name in FORMULATIONS
        ]
        grounding_values = [
            by_name[name]["deterministic_mapping"]["contextual_grounding"]
            for name in FORMULATIONS
        ]
        failure_values = [
            by_name[name]["deterministic_mapping"]["primary_failure_mode"]
            for name in FORMULATIONS
        ]
        for index, value in enumerate(failure_values):
            failure[index].append(value)
        task_complete = all(value is not None for value in task_values)
        grounding_complete = all(value is not None for value in grounding_values)
        failure_complete = all(value is not None for value in failure_values)
        exact_task += task_complete and len(set(task_values)) == 1
        exact_grounding += grounding_complete and len(set(grounding_values)) == 1
        exact_failure += failure_complete and len(set(failure_values)) == 1
        task_within_one += task_complete and max(task_values) - min(task_values) <= 1
        grounding_within_one += (
            grounding_complete and max(grounding_values) - min(grounding_values) <= 1
        )

    count = len(rows)
    return {
        "response_count": count,
        "consensus_unresolved_count": consensus_unresolved,
        "task_accuracy": {
            "krippendorff_alpha_ordinal": ordinal_alpha(task),
            "missing_rating_count": int(np.isnan(task).sum()),
            "exact_three_way_count": exact_task,
            "exact_three_way_rate": exact_task / count,
            "within_one_three_way_count": task_within_one,
            "within_one_three_way_rate": task_within_one / count,
        },
        "contextual_grounding": {
            "krippendorff_alpha_ordinal": ordinal_alpha(grounding),
            "missing_rating_count": int(np.isnan(grounding).sum()),
            "exact_three_way_count": exact_grounding,
            "exact_three_way_rate": exact_grounding / count,
            "within_one_three_way_count": grounding_within_one,
            "within_one_three_way_rate": grounding_within_one / count,
        },
        "failure_mode": {
            "krippendorff_alpha_nominal": nominal_alpha(failure),
            "missing_rating_count": sum(
                value is None for rater in failure for value in rater
            ),
            "exact_three_way_count": exact_failure,
            "exact_three_way_rate": exact_failure / count,
        },
    }


def validate_contract(
    rows: list[dict[str, Any]],
    scenario_doc: dict[str, Any],
    expected_models: int,
    expected_scenarios: int,
) -> dict[str, Any]:
    errors: list[str] = []
    scenarios = scenario_doc.get("scenarios", [])
    scenario_ids = {str(item["scenario_id"]) for item in scenarios}
    row_scenarios = {str(row["scenario_id"]) for row in rows}
    model_ids = {str(row["candidate_model_id"]) for row in rows}
    expected_rows = expected_models * expected_scenarios

    if len(rows) != expected_rows:
        errors.append(f"expected {expected_rows} rows, found {len(rows)}")
    if len(scenario_ids) != expected_scenarios:
        errors.append(
            f"scenario YAML expected {expected_scenarios} scenarios, found {len(scenario_ids)}"
        )
    if row_scenarios != scenario_ids:
        errors.append("judged rows do not cover exactly the YAML scenario IDs")
    if len(model_ids) != expected_models:
        errors.append(f"expected {expected_models} candidate models, found {len(model_ids)}")

    pair_counts = Counter(
        (str(row["candidate_model_id"]), str(row["scenario_id"])) for row in rows
    )
    duplicates = [pair for pair, count in pair_counts.items() if count != 1]
    if duplicates:
        errors.append(f"model/scenario pairs are not unique: {duplicates[:5]}")

    prompt_versions = {str(row["candidate_prompt_version"]) for row in rows}
    prompt_spec_hashes = {str(row["candidate_prompt_spec_sha256"]) for row in rows}
    if len(prompt_versions) != 1 or len(prompt_spec_hashes) != 1:
        errors.append("candidate models did not use one shared prompt spec/version")

    prompts_by_scenario: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        prompts_by_scenario[str(row["scenario_id"])].add(str(row["candidate_prompt"]))
    unequal_prompts = [
        scenario_id for scenario_id, prompts in prompts_by_scenario.items() if len(prompts) != 1
    ]
    if unequal_prompts:
        errors.append(
            "rendered prompt differs between candidate models for scenarios: "
            + ", ".join(unequal_prompts)
        )

    if any(row.get("generation_error") for row in rows):
        errors.append("one or more candidate generations failed")
    if any(bool(row.get("candidate_input_truncated")) for row in rows):
        errors.append("one or more candidate prompts were truncated")
    if any(int(row["candidate_generation"]["seed"]) != 42 for row in rows):
        errors.append("candidate random seed is not uniformly 42")
    if any(int(row.get("judge_seed", -1)) != 42 for row in rows):
        errors.append("Judge random seed is not uniformly 42")

    score_statuses = sorted({str(row["score_status"]) for row in rows})
    if score_statuses != ["diagnostic_failed_calibration"]:
        errors.append(
            "this finalizer expects the frozen failed-calibration diagnostic status"
        )
    if any(not bool(row.get("human_review_required")) for row in rows):
        errors.append("every failed-calibration row must require human review")
    if errors:
        raise ValueError("Contract validation failed: " + " | ".join(errors))
    return {
        "row_count": len(rows),
        "scenario_count": len(row_scenarios),
        "model_count": len(model_ids),
        "model_ids": sorted(model_ids),
        "prompt_versions": sorted(prompt_versions),
        "prompt_spec_sha256": sorted(prompt_spec_hashes),
        "identical_rendered_prompt_per_scenario": True,
        "generation_error_count": 0,
        "input_truncation_count": 0,
        "candidate_seed": 42,
        "judge_seed": 42,
        "score_statuses": score_statuses,
    }


def call_fields(
    formulation: dict[str, Any],
    dimension: str,
) -> dict[str, Any]:
    call = formulation["dimension_calls"][dimension]
    parsed = call["parsed"]
    generation = call["generation"]
    return {
        "prompt_version": call["prompt_version"],
        "system_prompt_sha256": call["system_prompt_sha256"],
        "user_prompt_sha256": call["user_prompt_sha256"],
        "raw_score": parsed["score"],
        "mapped_score": formulation["deterministic_mapping"][
            "primary_failure_mode" if dimension == "failure_mode" else dimension
        ],
        "comment": parsed["feedback"],
        "raw_output": generation["text"],
        "raw_output_sha256": call["generation_sha256"],
        "parse_status": parsed["parse_status"],
    }


def flatten_row(row: dict[str, Any], rubric: dict[str, Any]) -> dict[str, Any]:
    judge = row["structured_judge"]
    by_name = formulation_by_name(row)
    output: dict[str, Any] = {
        "run_id": row["run_id"],
        "source_candidate_run_id": row["source_candidate_run_id"],
        "benchmark_version": "0.2.0",
        "scenario_id": row["scenario_id"],
        "platform": row["platform"],
        "split": row["split"],
        "severity": row["severity_class"],
        "rubric_id": rubric["rubric_id"],
        "rubric_version": rubric["version"],
        "model_name": row["candidate_model_id"],
        "model_version": row["candidate_model_revision"],
        "prompt_version": row["candidate_prompt_version"],
        "prompt_spec_sha256": row["candidate_prompt_spec_sha256"],
        "prompt_sha256": row["candidate_prompt_sha256"],
        "prompt": row["candidate_prompt"],
        "input_stimulus": row["input_stimulus"],
        "raw_response": row["raw_output"],
        "raw_response_sha256": row["candidate_output_sha256"],
        "random_seed": row["candidate_generation"]["seed"],
        "do_sample": row["candidate_generation"]["do_sample"],
        "max_input_tokens": row["candidate_generation"]["max_input_tokens"],
        "max_new_tokens": row["candidate_generation"]["max_new_tokens"],
        "input_tokens": row["candidate_input_tokens"],
        "output_tokens": row["candidate_output_tokens"],
        "latency_ms": row["candidate_latency_ms"],
        "judge_model_name": judge["judge_backend"]["repo_id"],
        "judge_model_version": judge["judge_backend"]["revision"],
        "judge_rubric_version": judge["judge_prompt_spec"]["version"],
        "judge_rubric_sha256": judge["judge_prompt_spec"]["sha256"],
        "judge_requirement_metadata_version": judge["requirement_metadata_spec"]["version"],
        "judge_requirement_metadata_sha256": judge["requirement_metadata_spec"]["sha256"],
        "judge_random_seed": row["judge_seed"],
        "score_status": row["score_status"],
        "final_task_accuracy": judge["consensus"]["task_accuracy"]["final"],
        "final_contextual_grounding": judge["consensus"]["contextual_grounding"]["final"],
        "final_failure_mode": judge["consensus"]["primary_failure_mode"]["final"],
        "task_consensus_reason": judge["consensus"]["task_accuracy"]["reason"],
        "grounding_consensus_reason": judge["consensus"]["contextual_grounding"]["reason"],
        "failure_consensus_reason": judge["consensus"]["primary_failure_mode"]["reason"],
        "robustness_signal": "not_tested",
        "human_review_required": row["human_review_required"],
        "human_review_reasons": json.dumps(
            row["human_review_reasons"], ensure_ascii=False, separators=(",", ":")
        ),
    }
    for judge_number, formulation_name in enumerate(FORMULATIONS, start=1):
        formulation = by_name[formulation_name]
        output[f"judge_{judge_number}_formulation"] = formulation_name
        for dimension, short in (
            ("task_accuracy", "task"),
            ("contextual_grounding", "grounding"),
            ("failure_mode", "failure"),
        ):
            fields = call_fields(formulation, dimension)
            prefix = f"judge_{judge_number}_{short}"
            for key, value in fields.items():
                output[f"{prefix}_{key}"] = value
    return output


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    rubric: dict[str, Any],
) -> None:
    flattened = [flatten_row(row, rubric) for row in rows]
    fields = list(flattened[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)


def fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def write_report(
    path: Path,
    contract: dict[str, Any],
    agreement: dict[str, Any],
    by_model: dict[str, dict[str, Any]],
    source_sha256: str,
) -> None:
    task = agreement["task_accuracy"]
    grounding = agreement["contextual_grounding"]
    failure = agreement["failure_mode"]
    lines = [
        "# Week 2 Baseline Pipeline Result",
        "",
        "## Execution contract",
        "",
        f"- Frozen responses: `{contract['row_count']}` "
        f"({contract['scenario_count']} scenarios × {contract['model_count']} models)",
        f"- Models: `{', '.join(contract['model_ids'])}`",
        f"- Shared candidate prompt version: `{contract['prompt_versions'][0]}`",
        "- Rendered semantic prompt equality: `PASS` for every scenario/model pair",
        "- Candidate / Judge seed: `42` / `42`; greedy decoding",
        "- Candidate errors / input truncations: `0` / `0`",
        f"- Judge score status: `{', '.join(contract['score_statuses'])}`",
        f"- Frozen judged JSONL SHA-256: `{source_sha256}`",
        "",
        "> The Judge failed its provisional single-reviewer calibration. Automated ratings below "
        "are diagnostic prompt-sensitivity measurements, not validated final performance. "
        "Every row remains human-review-required.",
        "",
        "## Primary three-Judge agreement, separated by model",
        ]
    for model_id, metrics in sorted(by_model.items()):
        model_task = metrics["task_accuracy"]
        model_grounding = metrics["contextual_grounding"]
        model_failure = metrics["failure_mode"]
        unresolved = metrics["consensus_unresolved_count"]
        response_count = metrics["response_count"]
        lines.extend(
            [
                "",
                f"### {model_id} — {response_count} responses",
                "",
                "| Dimension | Krippendorff α | Exact 3-way | Within 1 point | "
                "Unresolved final |",
                "|---|---:|---:|---:|---:|",
                (
                    f"| Task Accuracy | "
                    f"{fmt(model_task['krippendorff_alpha_ordinal'])} | "
                    f"{model_task['exact_three_way_count']}/{response_count} "
                    f"({model_task['exact_three_way_rate']:.1%}) | "
                    f"{model_task['within_one_three_way_count']}/{response_count} "
                    f"({model_task['within_one_three_way_rate']:.1%}) | "
                    f"{unresolved['task_accuracy']} |"
                ),
                (
                    f"| Contextual Grounding | "
                    f"{fmt(model_grounding['krippendorff_alpha_ordinal'])} | "
                    f"{model_grounding['exact_three_way_count']}/{response_count} "
                    f"({model_grounding['exact_three_way_rate']:.1%}) | "
                    f"{model_grounding['within_one_three_way_count']}/{response_count} "
                    f"({model_grounding['within_one_three_way_rate']:.1%}) | "
                    f"{unresolved['contextual_grounding']} |"
                ),
                (
                    f"| Failure Mode | "
                    f"{fmt(model_failure['krippendorff_alpha_nominal'])} | "
                    f"{model_failure['exact_three_way_count']}/{response_count} "
                    f"({model_failure['exact_three_way_rate']:.1%}) | n/a | "
                    f"{unresolved['failure_mode']} |"
                ),
            ]
        )
        if model_id == "google/flan-t5-base":
            lines.extend(
                [
                    "",
                    "FLAN's repeated one-shot templates inflate agreement; these values "
                    "do not demonstrate correct scoring.",
                ]
            )
    lines.extend(
        [
            "",
            "## Contract-required all-response pipeline diagnostic",
            "",
            "The plan also requires agreement across all evaluated responses. Across "
            f"the {agreement['response_count']} rows, Task alpha is "
            f"`{fmt(task['krippendorff_alpha_ordinal'])}`, Grounding alpha is "
            f"`{fmt(grounding['krippendorff_alpha_ordinal'])}`, and Failure alpha is "
            f"`{fmt(failure['krippendorff_alpha_nominal'])}`. This is retained only as "
            "a pipeline-level prompt-sensitivity diagnostic; it is not a combined "
            "model-performance score. Full counts are in "
            "`W02_Baseline_Agreement.json`.",
            "",
            "## Per-model aggregate views",
            "",
            "`W02_Build_Per_Model_Views.py` generates separate 35-row CSVs and per-model "
            "per-platform, severity-weighted, split, and failure-mode tables. All "
            "aggregate values are marked `diagnostic_failed_judge_calibration` and show "
            "resolved-score coverage.",
            "",
            "## CSV field contract",
            "",
            "The CSV retains model name/version, scenario ID, complete candidate prompt, "
            "raw response, seed, severity, all three independent formulation names, every "
            "raw and mapped score, Judge comment, exact raw Judge output and hashes, "
            "consensus fields, failure mode, robustness signal, and review status.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.judged_rows)
    scenario_doc = load_yaml(args.scenario_path)
    rubric = load_yaml(args.rubric_path)
    if set(rubric.get("dimensions", {})) != {
        "task_accuracy",
        "contextual_grounding",
        "primary_failure_mode",
        "robustness_signal",
    }:
        raise ValueError("Rubric dimensions do not match the Week 2 contract")
    contract = validate_contract(
        rows,
        scenario_doc,
        expected_models=args.expected_models,
        expected_scenarios=args.expected_scenarios,
    )
    agreement = agreement_metrics(rows)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["candidate_model_id"])].append(row)
    by_model = {model_id: agreement_metrics(group) for model_id, group in groups.items()}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "W02_Baseline_Eval_Results.csv"
    summary_path = args.output_dir / "W02_Baseline_Agreement.json"
    report_path = args.output_dir / "W02_Baseline_Pipeline_Report.md"
    manifest_path = args.output_dir / "W02_Baseline_Run_Manifest.json"
    write_csv(csv_path, rows, rubric)
    source_sha256 = sha256_file(args.judged_rows)
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "finalizer_version": RUNNER_VERSION,
        "contract": contract,
        "agreement_all_responses": agreement,
        "agreement_by_model": by_model,
        "interpretation": (
            "diagnostic_only; Judge failed provisional single-reviewer calibration; "
            "human adjudication required"
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    write_report(report_path, contract, agreement, by_model, source_sha256)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "finalizer_version": RUNNER_VERSION,
        "source": {
            "scenario_yaml": args.scenario_path.name,
            "scenario_yaml_sha256": sha256_file(args.scenario_path),
            "rubric_yaml": args.rubric_path.name,
            "rubric_yaml_sha256": sha256_file(args.rubric_path),
            "candidate_prompt_spec": "W02_Prompt_Spec_v0.4.0.yaml",
            "candidate_prompt_spec_sha256": contract["prompt_spec_sha256"][0],
            "judged_rows": args.judged_rows.name,
            "judged_rows_sha256": source_sha256,
            "finalizer_source": Path(__file__).name,
            "finalizer_source_sha256": sha256_file(Path(__file__)),
            "reproducibility_source_hashes": {
                path.name: sha256_file(path)
                for path in REPRODUCIBILITY_SOURCE_PATHS
            },
        },
        "artifacts": {
            csv_path.name: sha256_file(csv_path),
            summary_path.name: sha256_file(summary_path),
            report_path.name: sha256_file(report_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

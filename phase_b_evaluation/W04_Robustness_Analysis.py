"""Aggregate Week 4 semantic-paraphrase and masked-input diagnostic scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ANALYZER_VERSION = "0.1.0"
VARIANTS = ("original", "synonym_substitution", "sentence_reordering", "tone_shift")
MASK_RATIOS = (0.0, 0.2, 0.4, 0.6)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score(row: dict[str, Any]) -> dict[str, Any] | None:
    value = row.get("normalized_score")
    return value if row.get("score_status") == "parsed" and isinstance(value, dict) else None


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def model_summary(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model = str(rows[0]["candidate_model_key"])
    semantic = [row for row in rows if row["evaluation_family"] == "semantic_robustness"]
    masked = [
        row
        for row in rows
        if row["evaluation_family"] in {"masked_input_robustness", "masked_input"}
    ]
    semantic_groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in semantic:
        semantic_groups[str(row["scenario_id"])][str(row["variant_type"])] = row

    consistent = stable_pass = stable_fail = 0
    eligible = 0
    variant_flip_counts = Counter()
    review_queue: list[dict[str, Any]] = []
    for scenario_id, variants in sorted(semantic_groups.items()):
        if set(variants) != set(VARIANTS) or any(score(row) is None for row in variants.values()):
            review_queue.append(
                {"model": model, "scenario_id": scenario_id, "reason": "incomplete_or_unparsed_semantic_group"}
            )
            continue
        eligible += 1
        passes = {name: bool(score(variants[name])["pass"]) for name in VARIANTS}
        values = set(passes.values())
        if len(values) == 1:
            consistent += 1
            if True in values:
                stable_pass += 1
            else:
                stable_fail += 1
        original = passes["original"]
        for variant in VARIANTS[1:]:
            if passes[variant] != original:
                variant_flip_counts[variant] += 1
                review_queue.append(
                    {
                        "model": model,
                        "scenario_id": scenario_id,
                        "reason": "original_to_paraphrase_pass_fail_flip",
                        "variant_type": variant,
                        "original_pass": original,
                        "variant_pass": passes[variant],
                    }
                )

    originals = {
        str(row["scenario_id"]): row
        for row in semantic
        if row.get("variant_type") == "original"
    }
    mask_groups: dict[str, dict[float, dict[str, Any]]] = defaultdict(dict)
    for scenario_id, row in originals.items():
        if scenario_id in {str(item["scenario_id"]) for item in masked}:
            mask_groups[scenario_id][0.0] = row
    for row in masked:
        mask_groups[str(row["scenario_id"])][float(row["mask_ratio"])] = row

    curves = []
    for ratio in MASK_RATIOS:
        ratio_rows = [group[ratio] for group in mask_groups.values() if ratio in group]
        parsed = [(row, score(row)) for row in ratio_rows if score(row) is not None]
        task_values = [float(value["task_accuracy"]) for _, value in parsed]
        weighted_values = [
            (float(value["task_accuracy"]), int(row["severity_class"]))
            for row, value in parsed
        ]
        pass_values = [bool(value["pass"]) for _, value in parsed]
        original_flip = 0
        flip_eligible = 0
        if ratio > 0:
            for scenario_id, group in mask_groups.items():
                if 0.0 not in group or ratio not in group:
                    continue
                base_score, ratio_score = score(group[0.0]), score(group[ratio])
                if base_score is None or ratio_score is None:
                    continue
                flip_eligible += 1
                original_flip += bool(base_score["pass"]) != bool(ratio_score["pass"])
        curves.append(
            {
                "mask_ratio": ratio,
                "row_count": len(ratio_rows),
                "parsed_row_count": len(parsed),
                "mean_task_accuracy": mean(task_values),
                "severity_weighted_mean_task_accuracy": (
                    sum(value * weight for value, weight in weighted_values)
                    / sum(weight for _, weight in weighted_values)
                    if weighted_values
                    else None
                ),
                "pass_rate": mean([float(value) for value in pass_values]),
                "original_to_mask_flip_rate": (
                    original_flip / flip_eligible if flip_eligible else None
                ),
                "severity_5_failure_count": sum(
                    int(row["severity_class"]) == 5 and not bool(value["pass"])
                    for row, value in parsed
                ),
            }
        )
    zero_mean = curves[0]["mean_task_accuracy"]
    for curve in curves:
        curve["task_accuracy_degradation_from_complete"] = (
            zero_mean - curve["mean_task_accuracy"]
            if zero_mean is not None and curve["mean_task_accuracy"] is not None
            else None
        )

    nonmonotonic = 0
    for scenario_id, group in sorted(mask_groups.items()):
        previous: dict[str, Any] | None = None
        previous_ratio: float | None = None
        for ratio in MASK_RATIOS:
            current = score(group[ratio]) if ratio in group else None
            if current is None:
                previous = None
                previous_ratio = None
                continue
            if previous is not None and int(current["task_accuracy"]) > int(previous["task_accuracy"]):
                nonmonotonic += 1
                review_queue.append(
                    {
                        "model": model,
                        "scenario_id": scenario_id,
                        "reason": "non_monotonic_safety_improvement_under_more_masking",
                        "from_mask_ratio": previous_ratio,
                        "to_mask_ratio": ratio,
                        "from_task_accuracy": previous["task_accuracy"],
                        "to_task_accuracy": current["task_accuracy"],
                    }
                )
            previous = current
            previous_ratio = ratio

    failure_codes = Counter(
        score(row)["failure_code"] for row in rows if score(row) is not None
    )
    parsed_count = sum(score(row) is not None for row in rows)
    summary = {
        "candidate_model_key": model,
        "row_count": len(rows),
        "parsed_row_count": parsed_count,
        "parse_coverage": parsed_count / len(rows) if rows else None,
        "semantic_robustness": {
            "scenario_count": len(semantic_groups),
            "eligible_scenario_count": eligible,
            "consistent_scenario_count": consistent,
            "semantic_robustness_score": consistent / eligible if eligible else None,
            "stable_pass_scenario_count": stable_pass,
            "stable_fail_scenario_count": stable_fail,
            "original_to_variant_flip_counts": dict(variant_flip_counts),
        },
        "masked_input": {
            "scenario_count": len(mask_groups),
            "curves": curves,
            "nonmonotonic_improvement_transition_count": nonmonotonic,
        },
        "failure_code_counts": dict(failure_codes),
        "mandatory_review_count": len(review_queue),
    }
    return summary, review_queue


def analyze(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ids = [str(row.get("score_id", "")) for row in rows]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("score_id values must be non-empty and unique")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["candidate_model_key"])].append(row)
    summaries = []
    reviews = []
    for model_rows in groups.values():
        summary, queue = model_summary(model_rows)
        summaries.append(summary)
        reviews.extend(queue)
    return (
        {
            "analyzer_version": ANALYZER_VERSION,
            "score_status": "diagnostic_ai_assisted_not_calibrated",
            "row_count": len(rows),
            "model_count": len(summaries),
            "models": sorted(summaries, key=lambda item: item["candidate_model_key"]),
            "interpretation_boundary": (
                "Semantic consistency and masked curves use an uncalibrated AI-assisted "
                "rubric score. They are diagnostic benchmark evidence, not deployed-product claims."
            ),
        },
        reviews,
    )


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Week 4 Text Robustness Results",
        "",
        "> AI-assisted rubric scores are diagnostic and not human ground truth.",
        "",
        "| Model | Parsed | Semantic robustness | Stable pass | Stable fail | Task degradation at 60% mask | Mandatory review flags |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in summary["models"]:
        semantic = model["semantic_robustness"]
        curve60 = next(row for row in model["masked_input"]["curves"] if row["mask_ratio"] == 0.6)
        robustness = semantic["semantic_robustness_score"]
        degradation = curve60["task_accuracy_degradation_from_complete"]
        lines.append(
            f"| {model['candidate_model_key']} | {model['parsed_row_count']}/{model['row_count']} | "
            f"{robustness:.3f} | {semantic['stable_pass_scenario_count']} | "
            f"{semantic['stable_fail_scenario_count']} | "
            f"{degradation:.3f} | {model['mandatory_review_count']} |"
        )
    lines.extend(["", "## Masked-input curves", ""])
    for model in summary["models"]:
        lines.extend(
            [
                f"### {model['candidate_model_key']}",
                "",
                "| Mask ratio | Mean task accuracy | Severity-weighted mean | Pass rate | Original flip rate | Severity-5 failures |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in model["masked_input"]["curves"]:
            flip = row["original_to_mask_flip_rate"]
            lines.append(
                f"| {row['mask_ratio']:.1f} | {row['mean_task_accuracy']:.3f} | "
                f"{row['severity_weighted_mean_task_accuracy']:.3f} | {row['pass_rate']:.3f} | "
                f"{('N/A' if flip is None else f'{flip:.3f}')} | {row['severity_5_failure_count']} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--review-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [row for path in args.scores for row in read_jsonl(path)]
    summary, review_queue = analyze(rows)
    summary["sources"] = [
        {"file": path.name, "sha256": sha256_file(path), "row_count": len(read_jsonl(path))}
        for path in args.scores
    ]
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_rows = []
    for model in summary["models"]:
        for row in model["masked_input"]["curves"]:
            csv_rows.append({"candidate_model_key": model["candidate_model_key"], **row})
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    args.output_md.write_text(markdown(summary), encoding="utf-8")
    if args.review_output:
        args.review_output.parent.mkdir(parents=True, exist_ok=True)
        with args.review_output.open("w", encoding="utf-8", newline="\n") as handle:
            for row in review_queue:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(rows), "models": len(summary["models"]), "review_flags": len(review_queue)}, indent=2))


if __name__ == "__main__":
    main()

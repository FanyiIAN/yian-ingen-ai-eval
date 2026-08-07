"""Aggregate the frozen Week 4 public-image VLM robustness scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ANALYZER_VERSION = "0.1.0"
CONDITIONS = ("clean", "gaussian_noise_std_0.08", "brightness_0.60")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score(row: dict[str, Any]) -> dict[str, Any] | None:
    value = row.get("normalized_score")
    return value if row.get("score_status") == "parsed" and isinstance(value, dict) else None


def average(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parsed = [(row, score(row)) for row in rows if score(row) is not None]
    return {
        "row_count": len(rows),
        "parsed_row_count": len(parsed),
        "mean_total_score": average([float(value["total_score"]) for _, value in parsed]),
        "mean_scene_interpretation": average(
            [float(value["scene_interpretation"]) for _, value in parsed]
        ),
        "mean_decision_recommendation": average(
            [float(value["decision_recommendation"]) for _, value in parsed]
        ),
        "mean_uncertainty_and_claim_control": average(
            [float(value["uncertainty_and_claim_control"]) for _, value in parsed]
        ),
        "acceptable_decision_rate": average(
            [float(bool(value["decision_acceptable"])) for _, value in parsed]
        ),
        "forbidden_claim_rate": average(
            [float(bool(value["forbidden_claim_present"])) for _, value in parsed]
        ),
    }


def analyze(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ids = [str(row.get("score_id", "")) for row in rows]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("score_id values must be non-empty and unique")
    if {str(row.get("condition_id")) for row in rows} - set(CONDITIONS):
        raise ValueError("unsupported multimodal condition")
    condition_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    platform_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    scenario_groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        condition = str(row["condition_id"])
        condition_groups[condition].append(row)
        platform_groups[(str(row["platform"]), condition)].append(row)
        scenario_groups[str(row["scenario_id"])][condition] = row

    review_queue = []
    perturbation_metrics = []
    for condition in CONDITIONS[1:]:
        score_drops = []
        decision_consistency = []
        eligible = 0
        for scenario_id, group in sorted(scenario_groups.items()):
            if "clean" not in group or condition not in group:
                continue
            clean, perturbed = score(group["clean"]), score(group[condition])
            if clean is None or perturbed is None:
                continue
            eligible += 1
            score_drops.append(float(clean["total_score"]) - float(perturbed["total_score"]))
            same_decision = bool(clean["decision_acceptable"]) == bool(
                perturbed["decision_acceptable"]
            )
            decision_consistency.append(float(same_decision))
            if not same_decision:
                review_queue.append(
                    {
                        "scenario_id": scenario_id,
                        "reason": "clean_to_perturbed_decision_flip",
                        "condition_id": condition,
                        "clean_decision_acceptable": clean["decision_acceptable"],
                        "perturbed_decision_acceptable": perturbed["decision_acceptable"],
                    }
                )
        perturbation_metrics.append(
            {
                "condition_id": condition,
                "eligible_scenario_count": eligible,
                "mean_clean_to_perturbed_score_drop": average(score_drops),
                "decision_consistency_clean_vs_perturbed": average(decision_consistency),
            }
        )
    for row in rows:
        value = score(row)
        if value is not None and value["forbidden_claim_present"]:
            review_queue.append(
                {
                    "scenario_id": row["scenario_id"],
                    "reason": "forbidden_claim_flag",
                    "condition_id": row["condition_id"],
                    "triggered_forbidden_claims": value["triggered_forbidden_claims"],
                }
            )

    parsed_count = sum(score(row) is not None for row in rows)
    summary = {
        "analyzer_version": ANALYZER_VERSION,
        "score_status": "diagnostic_ai_assisted_not_calibrated",
        "candidate_model_keys": sorted({str(row["candidate_model_key"]) for row in rows}),
        "row_count": len(rows),
        "parsed_row_count": parsed_count,
        "parse_coverage": parsed_count / len(rows) if rows else None,
        "scenario_count": len(scenario_groups),
        "conditions": [
            {"condition_id": condition, **summarize_group(condition_groups.get(condition, []))}
            for condition in CONDITIONS
        ],
        "platform_conditions": [
            {
                "platform": platform,
                "condition_id": condition,
                **summarize_group(platform_groups.get((platform, condition), [])),
            }
            for platform in sorted({str(row["platform"]) for row in rows})
            for condition in CONDITIONS
        ],
        "perturbation_robustness": perturbation_metrics,
        "mandatory_review_count": len(review_queue),
        "controlled_input_statement": (
            "Each scenario uses the same 768x768 clean pixels with exactly one "
            "deterministic factor changed: Gaussian noise or brightness."
        ),
        "interpretation_boundary": (
            "Scores are AI-assisted diagnostic rubric judgments over public-image proxies, "
            "not measurements of deployed Aido Rover or Sentinel Prime AI."
        ),
    }
    return summary, review_queue


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Week 4 Multimodal Robustness Results",
        "",
        "> Public-image proxy; AI-assisted rubric scores are diagnostic.",
        "",
        "| Condition | n parsed | Mean total /5 | Scene /2 | Decision /2 | Uncertainty /1 | Acceptable decision | Forbidden claim |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["conditions"]:
        lines.append(
            f"| {row['condition_id']} | {row['parsed_row_count']}/{row['row_count']} | "
            f"{row['mean_total_score']:.3f} | {row['mean_scene_interpretation']:.3f} | "
            f"{row['mean_decision_recommendation']:.3f} | "
            f"{row['mean_uncertainty_and_claim_control']:.3f} | "
            f"{row['acceptable_decision_rate']:.3f} | {row['forbidden_claim_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "| Perturbation | Mean score drop | Decision consistency | Eligible scenarios |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in summary["perturbation_robustness"]:
        lines.append(
            f"| {row['condition_id']} | {row['mean_clean_to_perturbed_score_drop']:.3f} | "
            f"{row['decision_consistency_clean_vs_perturbed']:.3f} | "
            f"{row['eligible_scenario_count']} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--review-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.scores)
    summary, review_queue = analyze(rows)
    summary["source"] = {
        "file": args.scores.name,
        "sha256": sha256_file(args.scores),
        "row_count": len(rows),
    }
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_rows = summary["platform_conditions"]
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
    print(json.dumps({"rows": len(rows), "reviews": len(review_queue)}, indent=2))


if __name__ == "__main__":
    main()

"""Compare two audited Mistral GPU runs, separating latency and behavioral differences."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROWS_NAME = "W02_Mistral_GPU_Integration_Rows.jsonl"
VOLATILE_KEYS = {"run_id", "timestamp_utc", "generation_latency_ms"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("replay", type=Path)
    return parser.parse_args()


def read_rows(run_dir: Path) -> dict[str, dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in (run_dir / ROWS_NAME).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {row["scenario_id"]: row for row in rows}


def compare_values(
    left: Any,
    right: Any,
    path: str,
    differences: list[dict[str, Any]],
) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        keys = sorted(set(left) | set(right))
        for key in keys:
            if (
                key in VOLATILE_KEYS
                or key == "latency_ms"
                or key.endswith("_latency_ms")
            ):
                continue
            if key not in left or key not in right:
                differences.append(
                    {"path": f"{path}.{key}", "left": left.get(key), "right": right.get(key)}
                )
                continue
            compare_values(left[key], right[key], f"{path}.{key}", differences)
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            differences.append(
                {"path": f"{path}.length", "left": len(left), "right": len(right)}
            )
            return
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            compare_values(left_item, right_item, f"{path}[{index}]", differences)
        return
    if isinstance(left, float) and isinstance(right, float):
        if not math.isclose(left, right, rel_tol=0.0, abs_tol=0.0):
            differences.append(
                {
                    "path": path,
                    "left": left,
                    "right": right,
                    "absolute_difference": abs(left - right),
                }
            )
        return
    if left != right:
        differences.append({"path": path, "left": left, "right": right})


def main() -> int:
    args = parse_args()
    original = read_rows(args.original.resolve())
    replay = read_rows(args.replay.resolve())
    scenario_ids = sorted(set(original) | set(replay))
    differences: list[dict[str, Any]] = []
    for scenario_id in scenario_ids:
        if scenario_id not in original or scenario_id not in replay:
            differences.append(
                {
                    "path": scenario_id,
                    "left": "present" if scenario_id in original else "missing",
                    "right": "present" if scenario_id in replay else "missing",
                }
            )
            continue
        compare_values(original[scenario_id], replay[scenario_id], scenario_id, differences)

    exact_fields = {
        "raw_output": sum(
            original[key]["raw_output"] == replay[key]["raw_output"]
            for key in set(original) & set(replay)
        ),
        "candidate_prompt": sum(
            original[key]["candidate_prompt"] == replay[key]["candidate_prompt"]
            for key in set(original) & set(replay)
        ),
        "deterministic_audit": sum(
            original[key]["deterministic_audit"] == replay[key]["deterministic_audit"]
            for key in set(original) & set(replay)
        ),
        "judge_results_excluding_latency": 0,
        "final_scores_and_review": 0,
    }
    for key in set(original) & set(replay):
        left_judges = json.loads(json.dumps(original[key]["judge_results"]))
        right_judges = json.loads(json.dumps(replay[key]["judge_results"]))
        for judges in (left_judges, right_judges):
            for judge in judges:
                for field in list(judge):
                    if field == "latency_ms" or field.endswith("_latency_ms"):
                        judge.pop(field)
                for classification_key in (
                    "task_accuracy_classification",
                    "contextual_grounding_classification",
                    "failure_mode_classification",
                ):
                    classification = judge[classification_key]
                    for field in list(classification):
                        if field == "latency_ms" or field.endswith("_latency_ms"):
                            classification.pop(field)
        exact_fields["judge_results_excluding_latency"] += left_judges == right_judges
        selected_fields = (
            "task_accuracy_ratings",
            "contextual_grounding_ratings",
            "primary_failure_mode_ratings",
            "final_task_accuracy",
            "final_contextual_grounding",
            "final_primary_failure_mode",
            "human_review_required",
            "human_review_reasons",
        )
        exact_fields["final_scores_and_review"] += all(
            original[key][field] == replay[key][field] for field in selected_fields
        )

    path_counts: dict[str, int] = {}
    max_numeric_difference = 0.0
    for difference in differences:
        normalized_path = difference["path"]
        for scenario_id in scenario_ids:
            if normalized_path.startswith(f"{scenario_id}."):
                normalized_path = normalized_path[len(scenario_id) + 1 :]
                break
        path_counts[normalized_path] = path_counts.get(normalized_path, 0) + 1
        max_numeric_difference = max(
            max_numeric_difference, float(difference.get("absolute_difference", 0.0))
        )

    result = {
        "original_run": args.original.name,
        "replay_run": args.replay.name,
        "scenario_count_original": len(original),
        "scenario_count_replay": len(replay),
        "exact_fields": exact_fields,
        "non_latency_difference_count": len(differences),
        "difference_path_counts": dict(sorted(path_counts.items())),
        "max_numeric_absolute_difference": max_numeric_difference,
        "first_100_differences": differences[:100],
        "strict_behavior_match": len(differences) == 0,
    }
    output_json = args.original / "W02_Mistral_Replay_Comparison.json"
    output_md = args.original / "W02_Mistral_Replay_Comparison.md"
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Week 2 Mistral Replay Comparison",
        "",
        f"- Original: `{args.original.name}`",
        f"- Replay: `{args.replay.name}`",
        f"- Strict behavior match: `{result['strict_behavior_match']}`",
        f"- Non-latency differences: `{result['non_latency_difference_count']}`",
        f"- Maximum numeric absolute difference: `{result['max_numeric_absolute_difference']}`",
        "",
        "## Exact row counts",
        "",
    ]
    lines.extend(f"- {key}: `{value}/35`" for key, value in exact_fields.items())
    lines.extend(["", "## Difference paths", ""])
    lines.extend(f"- `{key}`: `{value}`" for key, value in path_counts.items())
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if len(original) == 35 and len(replay) == 35 else 1


if __name__ == "__main__":
    raise SystemExit(main())

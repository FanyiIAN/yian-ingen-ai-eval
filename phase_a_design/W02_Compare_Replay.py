"""Compare the original and same-machine replay FLAN runs field by field."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "experiments" / "w02_local_flan_pipeline"
DEFAULT_ORIGINAL = RUN_ROOT / "local-flan-full-v0.2.0"
DEFAULT_REPLAY = RUN_ROOT / "local-flan-full-replay-v0.2.0"

ROW_VOLATILE_KEYS = {
    "run_id",
    "timestamp_utc",
    "generation_latency_ms",
    "latency_ms",
    "rationale_latency_ms",
}
SUMMARY_VOLATILE_KEYS = {
    "run_id",
    "created_at_utc",
    "model_load_seconds",
    "artifacts",
    "mean_generation_latency_ms",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    serialized = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def strip_keys(value: Any, excluded: set[str]) -> Any:
    if isinstance(value, dict):
        return {
            key: strip_keys(item, excluded)
            for key, item in value.items()
            if key not in excluded
        }
    if isinstance(value, list):
        return [strip_keys(item, excluded) for item in value]
    return value


def diff_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    if type(left) is not type(right):
        return [prefix or "$"]
    if isinstance(left, dict):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(diff_paths(left[key], right[key], child))
        return paths
    if isinstance(left, list):
        if len(left) != len(right):
            return [f"{prefix}.length"]
        paths = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            paths.extend(diff_paths(left_item, right_item, f"{prefix}[{index}]"))
        return paths
    return [] if left == right else [prefix or "$"]


def row_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["scenario_id"]: row for row in rows}


def prompt_ledger_behavior(path: Path) -> list[dict[str, Any]]:
    entries = load_jsonl(path)
    return [
        {
            key: value
            for key, value in entry.items()
            if key not in {"run_id", "source_result_row_sha256"}
        }
        for entry in entries
    ]


def explicit_checks(
    original: dict[str, dict[str, Any]], replay: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    ids = sorted(original)
    raw_output_mismatches = [
        scenario_id
        for scenario_id in ids
        if original[scenario_id]["raw_output"] != replay[scenario_id]["raw_output"]
    ]
    judge_rating_mismatches = [
        scenario_id
        for scenario_id in ids
        if any(
            original[scenario_id][field] != replay[scenario_id][field]
            for field in (
                "task_accuracy_ratings",
                "contextual_grounding_ratings",
                "primary_failure_mode_ratings",
            )
        )
    ]
    loss_margin_mismatches: list[str] = []
    rationale_mismatches: list[str] = []
    for scenario_id in ids:
        left_results = original[scenario_id]["judge_results"]
        right_results = replay[scenario_id]["judge_results"]
        for left, right in zip(left_results, right_results):
            formulation = left["formulation"]
            for dimension in (
                "task_accuracy_classification",
                "contextual_grounding_classification",
                "failure_mode_classification",
            ):
                for field in ("selected", "losses", "margin_to_second"):
                    if left[dimension][field] != right[dimension][field]:
                        loss_margin_mismatches.append(
                            f"{scenario_id}:{formulation}:{dimension}:{field}"
                        )
            if left["rationale"] != right["rationale"]:
                rationale_mismatches.append(f"{scenario_id}:{formulation}")
    audit_mismatches = [
        scenario_id
        for scenario_id in ids
        if original[scenario_id]["deterministic_audit"]
        != replay[scenario_id]["deterministic_audit"]
    ]
    return {
        "raw_output_mismatch_ids": raw_output_mismatches,
        "judge_rating_mismatch_ids": judge_rating_mismatches,
        "classification_loss_or_margin_mismatches": loss_margin_mismatches,
        "rationale_mismatches": rationale_mismatches,
        "deterministic_audit_mismatch_ids": audit_mismatches,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    args = parser.parse_args()
    original_dir = args.original.resolve()
    replay_dir = args.replay.resolve()

    filename = "W02_FLAN_Local_Integration_Rows.jsonl"
    original_rows = load_jsonl(original_dir / filename)
    replay_rows = load_jsonl(replay_dir / filename)
    original_by_id = row_map(original_rows)
    replay_by_id = row_map(replay_rows)
    if set(original_by_id) != set(replay_by_id):
        raise ValueError("Original and replay scenario IDs differ")

    original_behavior = {
        scenario_id: strip_keys(original_by_id[scenario_id], ROW_VOLATILE_KEYS)
        for scenario_id in sorted(original_by_id)
    }
    replay_behavior = {
        scenario_id: strip_keys(replay_by_id[scenario_id], ROW_VOLATILE_KEYS)
        for scenario_id in sorted(replay_by_id)
    }
    row_diff_paths = diff_paths(original_behavior, replay_behavior)
    checks = explicit_checks(original_by_id, replay_by_id)

    original_summary = strip_keys(
        load_json(original_dir / "W02_FLAN_Local_Integration_Summary.json"),
        SUMMARY_VOLATILE_KEYS,
    )
    replay_summary = strip_keys(
        load_json(replay_dir / "W02_FLAN_Local_Integration_Summary.json"),
        SUMMARY_VOLATILE_KEYS,
    )
    summary_diff_paths = diff_paths(original_summary, replay_summary)

    original_prompts = prompt_ledger_behavior(
        original_dir / "W02_FLAN_Rendered_Prompts.jsonl"
    )
    replay_prompts = prompt_ledger_behavior(
        replay_dir / "W02_FLAN_Rendered_Prompts.jsonl"
    )
    prompt_diff_paths = diff_paths(original_prompts, replay_prompts)

    original_analysis_path = original_dir / "W02_FLAN_Output_Analysis.json"
    replay_analysis_path = replay_dir / "W02_FLAN_Output_Analysis.json"
    analysis_equal = load_json(original_analysis_path) == load_json(replay_analysis_path)

    all_mismatch_lists = list(checks.values())
    replay_pass = not any(
        (
            row_diff_paths,
            summary_diff_paths,
            prompt_diff_paths,
            *all_mismatch_lists,
        )
    ) and analysis_equal

    comparison = {
        "comparison_id": "w02-local-flan-same-machine-replay-v0.1.0",
        "status": "pass" if replay_pass else "fail",
        "strict_behavioral_replay_pass": replay_pass,
        "original_run_id": load_json(
            original_dir / "W02_FLAN_Local_Integration_Summary.json"
        )["run_id"],
        "replay_run_id": load_json(
            replay_dir / "W02_FLAN_Local_Integration_Summary.json"
        )["run_id"],
        "scenario_count": len(original_rows),
        "comparison_policy": {
            "required_exact": [
                "raw candidate output and token counts",
                "deterministic audit",
                "three judge ratings per dimension",
                "all classification target losses and margins",
                "all judge rationales",
                "rendered candidate and judge prompts",
                "all non-latency aggregate metrics",
            ],
            "excluded_as_volatile": sorted(
                ROW_VOLATILE_KEYS | SUMMARY_VOLATILE_KEYS
            ),
        },
        "canonical_behavior_sha256": {
            "original": canonical_hash(original_behavior),
            "replay": canonical_hash(replay_behavior),
        },
        "full_file_sha256": {
            "original_rows_jsonl": sha256_file(original_dir / filename),
            "replay_rows_jsonl": sha256_file(replay_dir / filename),
        },
        "explicit_checks": checks,
        "normalized_row_diff_paths": row_diff_paths,
        "normalized_summary_diff_paths": summary_diff_paths,
        "rendered_prompt_diff_paths": prompt_diff_paths,
        "output_analysis_exact_match": analysis_equal,
        "note": (
            "Full JSONL hashes differ by design because run IDs, timestamps and "
            "measured latencies differ. Canonical behavioral hashes exclude only "
            "the documented volatile fields."
        ),
    }

    json_path = original_dir / "W02_FLAN_Replay_Comparison.json"
    json_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    markdown_path = original_dir / "W02_FLAN_Replay_Comparison.md"
    markdown_path.write_text(
        "\n".join(
            (
                "# Week 2 Local FLAN Same-Machine Replay Comparison",
                "",
                f"**Status:** `{'PASS' if replay_pass else 'FAIL'}`",
                "",
                f"- Scenarios compared: {len(original_rows)}",
                f"- Raw-output mismatches: {len(checks['raw_output_mismatch_ids'])}",
                f"- Judge-rating mismatches: {len(checks['judge_rating_mismatch_ids'])}",
                f"- Classification loss/margin mismatches: {len(checks['classification_loss_or_margin_mismatches'])}",
                f"- Rationale mismatches: {len(checks['rationale_mismatches'])}",
                f"- Deterministic-audit mismatches: {len(checks['deterministic_audit_mismatch_ids'])}",
                f"- Rendered-prompt differences: {len(prompt_diff_paths)}",
                f"- Non-latency summary differences: {len(summary_diff_paths)}",
                f"- Canonical behavior hash: `{comparison['canonical_behavior_sha256']['original']}`",
                "",
                "Run ID, timestamps and measured latencies were excluded as documented",
                "volatile fields. Everything that determines or describes candidate and",
                "judge behavior was required to match exactly.",
                "",
            )
        ),
        encoding="utf-8",
    )
    print(f"Replay comparison: {comparison['status']}")
    print(f"Canonical behavior hash: {comparison['canonical_behavior_sha256']['original']}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    return 0 if replay_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

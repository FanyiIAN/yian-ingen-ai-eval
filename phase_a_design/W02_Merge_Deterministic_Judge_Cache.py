"""Merge fresh Judge rows with hash-validated deterministic cache hits.

Prometheus Judge prompts depend on the frozen scenario, candidate response, Judge
specification, requirement metadata, checkpoint revision, seed, and decoding
configuration—not on the candidate prompt that produced an identical response.
This utility reuses a prior trace only when that complete Judge cache key matches.
Every reused row is explicitly labelled; no score or comment is regenerated or
silently relabelled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-rows", type=Path, required=True)
    parser.add_argument("--fresh-judged-rows", type=Path, required=True)
    parser.add_argument("--cached-judged-rows", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
            handle.write("\n")


def judge_cache_key(row: dict[str, Any]) -> dict[str, Any]:
    judge = row["structured_judge"]
    backend = judge["judge_backend"]
    calls = [
        call
        for formulation in judge["formulation_results"]
        for call in formulation["dimension_calls"].values()
    ]
    return {
        "scenario_id": row["scenario_id"],
        "candidate_output_sha256": row["candidate_output_sha256"],
        "judge_model": backend["repo_id"],
        "judge_revision": backend["revision"],
        "judge_prompt_spec_sha256": judge["judge_prompt_spec"]["sha256"],
        "requirement_metadata_sha256": judge["requirement_metadata_spec"]["sha256"],
        "judge_seed": row["judge_seed"],
        "system_prompt_sha256": sorted(
            {call["system_prompt_sha256"] for call in calls}
        ),
        "user_prompt_sha256": sorted(call["user_prompt_sha256"] for call in calls),
        "formulations": sorted(
            formulation["formulation"]
            for formulation in judge["formulation_results"]
        ),
        "inference_batch_size": backend["inference_batch_size"],
    }


def transplant(
    candidate: dict[str, Any],
    source: dict[str, Any],
    run_id: str,
    source_kind: str,
) -> dict[str, Any]:
    if source["scenario_id"] != candidate["scenario_id"]:
        raise ValueError("Scenario mismatch during Judge transplant")
    if source["candidate_output_sha256"] != candidate["candidate_output_sha256"]:
        raise ValueError(
            f"{candidate['candidate_item_id']}: candidate output hash does not match cache"
        )
    row = dict(candidate)
    judge_keys = (
        "prometheus_full_runner_version",
        "judge_seed",
        "judge_calibration_metrics_path",
        "judge_calibration_metrics_sha256",
        "score_status",
        "structured_judge",
        "human_review_required",
        "human_review_reasons",
        "human_task_accuracy",
        "human_contextual_grounding",
        "human_primary_failure_mode",
        "human_rationale",
    )
    for key in judge_keys:
        row[key] = source[key]
    cache_key = judge_cache_key(source)
    row.update(
        {
            "run_id": run_id,
            "source_candidate_run_id": candidate["run_id"],
            "source_candidate_row_sha256": candidate["candidate_row_sha256"],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "judge_execution_source": source_kind,
            "judge_cache_source_run_id": source["run_id"],
            "judge_cache_source_row_sha256": source["judged_row_sha256"],
            "judge_cache_key": cache_key,
            "judge_cache_key_sha256": canonical_sha256(cache_key),
        }
    )
    row["judged_row_sha256"] = canonical_sha256(row)
    return row


def prompt_trace(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for row in rows:
        judge = row["structured_judge"]
        for formulation in judge["formulation_results"]:
            for dimension, call in formulation["dimension_calls"].items():
                traces.append(
                    {
                        "run_id": row["run_id"],
                        "candidate_item_id": row["candidate_item_id"],
                        "scenario_id": row["scenario_id"],
                        "candidate_model_id": row["candidate_model_id"],
                        "candidate_model_revision": row["candidate_model_revision"],
                        "candidate_prompt_version": row["candidate_prompt_version"],
                        "candidate_prompt_sha256": row["candidate_prompt_sha256"],
                        "candidate_output_sha256": row["candidate_output_sha256"],
                        "judge_execution_source": row["judge_execution_source"],
                        "judge_cache_key_sha256": row["judge_cache_key_sha256"],
                        "formulation": formulation["formulation"],
                        "dimension": dimension,
                        "judge_prompt_version": call["prompt_version"],
                        "system_prompt": call["system_prompt"],
                        "system_prompt_sha256": call["system_prompt_sha256"],
                        "user_prompt": call["user_prompt"],
                        "user_prompt_sha256": call["user_prompt_sha256"],
                        "judge_raw_output": call["generation"]["text"],
                        "judge_raw_output_sha256": call["generation_sha256"],
                        "judge_raw_score": call["parsed"]["score"],
                        "judge_comment": call["parsed"]["feedback"],
                        "parse_status": call["parsed"]["parse_status"],
                    }
                )
    return traces


def main() -> int:
    args = parse_args()
    candidates = load_jsonl(args.candidate_rows)
    fresh = {
        row["candidate_item_id"]: row for row in load_jsonl(args.fresh_judged_rows)
    }
    cached = {
        row["candidate_item_id"]: row for row in load_jsonl(args.cached_judged_rows)
    }
    if len(candidates) != 70:
        raise ValueError(f"Expected 70 candidates, found {len(candidates)}")

    output: list[dict[str, Any]] = []
    fresh_count = cached_count = 0
    for candidate in sorted(candidates, key=lambda row: row["candidate_item_id"]):
        item_id = candidate["candidate_item_id"]
        if item_id in fresh:
            source = fresh[item_id]
            source_kind = "fresh_gpu_execution"
            fresh_count += 1
        elif item_id in cached:
            source = cached[item_id]
            source_kind = "deterministic_hash_validated_cache"
            cached_count += 1
        else:
            raise ValueError(f"No fresh or cached Judge row for {item_id}")
        output.append(transplant(candidate, source, args.run_id, source_kind))

    if len(output) != 70:
        raise ValueError("Merged output is incomplete")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    judged_path = args.output_dir / "W02_Prometheus_Full_Judged_Rows.jsonl"
    trace_path = args.output_dir / "W02_Prometheus_Prompt_Trace.jsonl"
    manifest_path = args.output_dir / "W02_Judge_Cache_Merge_Manifest.json"
    write_jsonl(judged_path, output)
    traces = prompt_trace(output)
    if len(traces) != 630:
        raise ValueError(f"Expected 630 Judge traces, found {len(traces)}")
    write_jsonl(trace_path, traces)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "row_count": len(output),
        "trace_count": len(traces),
        "fresh_gpu_row_count": fresh_count,
        "deterministic_cache_row_count": cached_count,
        "cache_policy": (
            "Reuse only after exact scenario/output/Judge prompt/model/seed/cache-key match"
        ),
        "sources": {
            "candidate_rows": {
                "path": str(args.candidate_rows.resolve()),
                "sha256": sha256_file(args.candidate_rows),
            },
            "fresh_judged_rows": {
                "path": str(args.fresh_judged_rows.resolve()),
                "sha256": sha256_file(args.fresh_judged_rows),
            },
            "cached_judged_rows": {
                "path": str(args.cached_judged_rows.resolve()),
                "sha256": sha256_file(args.cached_judged_rows),
            },
        },
        "artifacts": {
            judged_path.name: sha256_file(judged_path),
            trace_path.name: sha256_file(trace_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

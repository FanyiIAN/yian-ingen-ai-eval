"""Summarize and compare Week 3 RAGAS JSONL without hiding invalid values.

This reporting layer never edits candidate outputs or raw Judge rows. It treats
non-finite values as invalid observations, reports their IDs and reasons, and
computes means only over finite observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ANALYZER_VERSION = "0.1.0"
METRICS = (
    "answer_relevance",
    "faithfulness_to_retrieved_context",
    "context_relevance",
    "context_recall",
    "context_precision",
)
RAG_ONLY_METRICS = set(METRICS) - {"answer_relevance"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def metric_payload(row: dict[str, Any], metric: str) -> dict[str, Any]:
    payload = (
        ((row.get("ragas") or {}).get("metrics") or {}).get(metric) or {}
    )
    return payload if isinstance(payload, dict) else {}


def summarize_metric(
    rows: list[dict[str, Any]],
    metric: str,
    condition: str,
) -> dict[str, Any]:
    if metric in RAG_ONLY_METRICS and condition != "rag":
        return {
            "applicable": False,
            "applicable_count": 0,
            "finite_count": 0,
            "invalid_count": 0,
            "finite_mean": None,
            "finite_median": None,
            "invalid_rows": [],
        }

    finite_values: list[float] = []
    invalid_rows: list[dict[str, Any]] = []
    for row in rows:
        payload = metric_payload(row, metric)
        raw_value = payload.get("value")
        value = finite_float(raw_value)
        if value is None:
            invalid_rows.append(
                {
                    "run_item_id": row.get("run_item_id"),
                    "eval_id": row.get("eval_id"),
                    "raw_value": raw_value,
                    "reason": payload.get("reason"),
                }
            )
        else:
            finite_values.append(value)

    return {
        "applicable": True,
        "applicable_count": len(rows),
        "finite_count": len(finite_values),
        "invalid_count": len(invalid_rows),
        "finite_mean": (
            round(statistics.mean(finite_values), 6)
            if finite_values
            else None
        ),
        "finite_median": (
            round(statistics.median(finite_values), 6)
            if finite_values
            else None
        ),
        "invalid_rows": invalid_rows,
    }


def summarize_rows(
    rows: list[dict[str, Any]],
    source_path: Path | None = None,
) -> dict[str, Any]:
    condition_counts = Counter(row.get("condition") for row in rows)
    conditions: dict[str, Any] = {}
    for condition in sorted(
        value for value in condition_counts if isinstance(value, str)
    ):
        selected = [row for row in rows if row.get("condition") == condition]
        conditions[condition] = {
            "rows": len(selected),
            "metrics": {
                metric: summarize_metric(selected, metric, condition)
                for metric in METRICS
            },
        }

    model_ids = sorted(
        {
            row.get("candidate_model_id")
            for row in rows
            if row.get("candidate_model_id")
        }
    )
    seeds = sorted(
        {
            row.get("random_seed")
            for row in rows
            if row.get("random_seed") is not None
        }
    )
    return {
        "analyzer_version": ANALYZER_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": (
            {
                "path": str(source_path),
                "sha256": sha256_file(source_path),
            }
            if source_path is not None
            else None
        ),
        "rows": len(rows),
        "condition_counts": dict(condition_counts),
        "candidate_model_ids": model_ids,
        "random_seeds": seeds,
        "conditions": conditions,
        "validity": {
            "automatic_judge_status": "provisional_pending_calibration_and_human_review",
            "non_finite_policy": (
                "Exclude from finite mean; retain coverage, raw value, row ID, "
                "and reason. Never coerce to zero."
            ),
        },
    }


def metric_deltas(
    parent: dict[str, Any],
    candidate: dict[str, Any],
    condition: str = "rag",
) -> dict[str, Any]:
    parent_metrics = parent["conditions"][condition]["metrics"]
    candidate_metrics = candidate["conditions"][condition]["metrics"]
    result: dict[str, Any] = {}
    for metric in METRICS:
        parent_metric = parent_metrics[metric]
        candidate_metric = candidate_metrics[metric]
        parent_mean = parent_metric["finite_mean"]
        candidate_mean = candidate_metric["finite_mean"]
        result[metric] = {
            "parent_finite_mean": parent_mean,
            "candidate_finite_mean": candidate_mean,
            "delta_candidate_minus_parent": (
                round(candidate_mean - parent_mean, 6)
                if parent_mean is not None and candidate_mean is not None
                else None
            ),
            "parent_finite_count": parent_metric["finite_count"],
            "candidate_finite_count": candidate_metric["finite_count"],
            "parent_invalid_count": parent_metric["invalid_count"],
            "candidate_invalid_count": candidate_metric["invalid_count"],
        }
    return result


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--scores", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--parent-scores", type=Path, required=True)
    compare.add_argument("--candidate-scores", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "summarize":
        summary = summarize_rows(read_jsonl(args.scores), args.scores)
        write_json(args.output, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    parent = summarize_rows(
        read_jsonl(args.parent_scores),
        args.parent_scores,
    )
    candidate = summarize_rows(
        read_jsonl(args.candidate_scores),
        args.candidate_scores,
    )
    comparison = {
        "analyzer_version": ANALYZER_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "condition": "rag",
        "parent": parent,
        "candidate": candidate,
        "metric_deltas": metric_deltas(parent, candidate),
        "decision_status": (
            "requires_preregistered_keep_rule_and_qualitative_review"
        ),
    }
    write_json(args.output, comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


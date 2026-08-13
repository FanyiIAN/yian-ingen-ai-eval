"""Create a public-safe item export from private Week 4 RAG event rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


EXPORT_VERSION = "1.0.0"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sanitize(event: dict[str, Any]) -> dict[str, Any]:
    profile = event["request_profile"]
    timings = profile["timings"]
    resources = profile["resources"]
    return {
        "export_version": EXPORT_VERSION,
        "run_item_id": event["run_item_id"],
        "model_key": event["candidate_model_key"],
        "model_id": event["candidate_model_id"],
        "model_revision": event["candidate_model_revision"],
        "eval_id": event["eval_id"],
        "platform": event["platform"],
        "condition": event["condition"],
        "seed": event["seed"],
        "precision": event["precision"],
        "cold_or_warm": event["cold_or_warm"],
        "prompt_tokens": event["prompt_tokens"],
        "output_tokens": event["output_tokens"],
        "total_tokens": event["total_tokens"],
        "retrieval_latency_returned_ms": event["retrieval_latency_returned_ms"],
        "query_embedding_ms": timings.get("query_embedding_ms"),
        "vector_search_ms": timings.get("vector_search_ms"),
        "rerank_ms": timings.get("rerank_ms"),
        "context_assembly_ms": timings.get("context_assembly_ms"),
        "preprocess_ms": timings.get("preprocess_ms"),
        "ttft_ms": timings.get("ttft_ms"),
        "generation_ms": timings.get("generation_ms"),
        "decode_ms": timings.get("decode_ms"),
        "question_to_response_ms": timings.get("question_to_response_ms"),
        "gpu_peak_mib": (resources.get("gpu_device_memory_used_mib") or {}).get("peak"),
        "gpu_utilization_mean_pct": (resources.get("gpu_utilization_pct") or {}).get("mean"),
        "gpu_power_mean_w": (resources.get("gpu_power_w") or {}).get("mean"),
        "process_rss_peak_mib": (resources.get("process_rss_mib") or {}).get("peak"),
        "candidate_messages_sha256": event["candidate_messages_sha256"],
        "candidate_output_sha256": event["candidate_output_sha256"],
        "quality_status": event["quality_status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    events = read_jsonl(args.events)
    if len(events) != 80:
        raise ValueError(f"expected 80 official events, found {len(events)}")
    if any(row.get("cold_or_warm") != "warm_steady_state" for row in events):
        raise ValueError("official export may contain only warm steady-state rows")
    run_ids = [str(row.get("run_item_id", "")) for row in events]
    if any(not value for value in run_ids) or len(run_ids) != len(set(run_ids)):
        raise ValueError("run_item_id values must be non-empty and unique")
    rows = [sanitize(event) for event in events]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "export_version": EXPORT_VERSION}, indent=2))


if __name__ == "__main__":
    main()

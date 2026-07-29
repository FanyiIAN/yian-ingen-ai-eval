"""Create a compact, auditable comparison for registered RAG regression cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


DEFAULT_EVAL_IDS = [
    "W03-OFFICIAL-FARI-002",
    "W03-OFFICIAL-FARI-005",
    "W03-OFFICIAL-SENPAI-005",
    "W03-OFFICIAL-SENPAI-006",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_generation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["generation_latency_ms"]) for row in rows]
    token_counts = [int(row["output_tokens"]) for row in rows]
    return {
        "rows": len(rows),
        "empty_outputs": sum(not row.get("candidate_output", "").strip() for row in rows),
        "mean_generation_latency_ms": round(statistics.mean(latencies), 3),
        "median_generation_latency_ms": round(statistics.median(latencies), 3),
        "mean_output_tokens": round(statistics.mean(token_counts), 3),
    }


def build_gate_report(
    parent_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    eval_ids: list[str],
) -> str:
    parent = {row["eval_id"]: row for row in parent_rows}
    candidate = {row["eval_id"]: row for row in candidate_rows}
    if set(parent) != set(candidate):
        raise ValueError("Parent and candidate eval_id sets differ")
    missing = [eval_id for eval_id in eval_ids if eval_id not in parent]
    if missing:
        raise ValueError(f"Missing registered eval IDs: {missing}")

    equal_contexts = sum(
        parent[eval_id].get("retrieved_contexts")
        == candidate[eval_id].get("retrieved_contexts")
        for eval_id in parent
    )
    lines = [
        f"contexts_equal={equal_contexts}/{len(parent)}",
        f"candidate_summary={json.dumps(summarize_generation(candidate_rows), sort_keys=True)}",
    ]
    for eval_id in eval_ids:
        lines.extend(
            [
                "",
                eval_id,
                "PARENT:",
                parent[eval_id].get("candidate_output", ""),
                "CANDIDATE:",
                candidate[eval_id].get("candidate_output", ""),
            ]
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-id", action="append", dest="eval_ids")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parent_rows = read_jsonl(args.parent)
    candidate_rows = read_jsonl(args.candidate)
    report = build_gate_report(
        parent_rows,
        candidate_rows,
        args.eval_ids or DEFAULT_EVAL_IDS,
    )
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable output: {args.output}")
    args.output.write_text(report, encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "status": "ok",
                "parent_sha256": sha256_file(args.parent),
                "candidate_sha256": sha256_file(args.candidate),
                "output_sha256": sha256_file(args.output),
                "candidate_summary": summarize_generation(candidate_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

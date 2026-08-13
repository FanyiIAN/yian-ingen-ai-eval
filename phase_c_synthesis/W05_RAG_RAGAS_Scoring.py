"""Score Week 5 Faithfulness and Relevance with the frozen local RAGAS stack.

The Mistral Judge endpoint must be loopback-only and BGE-M3 must be a local
checkpoint. Scores are append-only diagnostics. Exact duplicate
question/answer/context triples are evaluated once and explicitly reused.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


SCORER_VERSION = "1.1.0"
HERE = Path(__file__).resolve().parent
PHASE_B = HERE.parent / "phase_b_evaluation"
DEFAULT_CONFIG = HERE / "W05_RAG_Optimisation_Run_Config_v1.0.0.yaml"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        handle.flush()


def cache_key(record: dict[str, Any]) -> str:
    payload = {
        "question": record["question"],
        "candidate_output": record["candidate_output"],
        "retrieved_contexts": record["retrieved_contexts"],
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def finite_metric(value: dict[str, Any]) -> bool:
    metric = value.get("value")
    return (
        isinstance(metric, (int, float))
        and not isinstance(metric, bool)
        and math.isfinite(float(metric))
    )


def returned_metrics(row: dict[str, Any]) -> bool:
    """True when every applicable call returned, including a retained NaN."""
    metrics = row.get("metrics") or {}
    relevance_returned = (
        (metrics.get("answer_relevance") or {}).get("value") is not None
    )
    if not relevance_returned:
        return False
    if not (row.get("retrieved_contexts") or []):
        return True
    return (
        (metrics.get("faithfulness_to_retrieved_context") or {}).get("value")
        is not None
    )


def prepare_records(
    candidates: list[dict[str, Any]], eval_set: dict[str, Any]
) -> list[dict[str, Any]]:
    items = {str(item["eval_id"]): item for item in eval_set["items"]}
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        item = items.get(candidate["eval_id"])
        if item is None:
            raise ValueError(f"unknown eval_id: {candidate['eval_id']}")
        contexts = [
            str(context["content"])
            for context in candidate.get("retrieved_contexts") or []
        ]
        condition = str(candidate.get("condition", "rag"))
        if not contexts and condition != "base":
            raise ValueError(f"{candidate['run_item_id']}: no retrieved contexts")
        records.append(
            {
                "run_item_id": candidate["run_item_id"],
                "variant_id": candidate.get("variant_id", candidate.get("condition")),
                "condition": condition,
                "eval_id": candidate["eval_id"],
                "question": item["question"],
                "candidate_output": candidate["candidate_output"],
                "candidate_output_sha256": candidate["candidate_output_sha256"],
                "retrieved_contexts": contexts,
                "retrieved_chunk_ids": [
                    str(context["chunk_id"])
                    for context in candidate["retrieved_contexts"]
                ],
                "candidate_model_id": candidate["candidate_model_id"],
                "candidate_model_revision": candidate["candidate_model_revision"],
            }
        )
    return records


def reused_row(record: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    output = dict(source)
    output.update(
        {
            "run_item_id": record["run_item_id"],
            "variant_id": record["variant_id"],
            "reused_from_run_item_id": source["run_item_id"],
            "reuse_reason": "same_question_answer_and_context_sha256",
            "completed_at_utc": utc_now(),
        }
    )
    return output


async def score_with_retries(
    metric: Any,
    evaluator: Any,
    *,
    max_attempts: int = 3,
    **kwargs: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Retry only missing Judge results, while retaining returned NaN values.

    A returned NaN can be a defined RAGAS outcome when no claims are
    extractable. A missing value can also arise from a transient loopback
    disconnect, so those calls receive a small, bounded retry budget.
    """
    audit: list[dict[str, Any]] = []
    result: dict[str, Any] = {}
    for attempt in range(1, max_attempts + 1):
        result = await evaluator.safe_score(metric, **kwargs)
        value = result.get("value")
        audit.append(
            {
                "attempt": attempt,
                "returned_value": value is not None,
                "reason": result.get("reason"),
            }
        )
        if value is not None:
            break
        if attempt < max_attempts:
            await asyncio.sleep(float(attempt))
    return result, audit


async def score_record(record: dict[str, Any], evaluator: Any) -> dict[str, Any]:
    relevance, relevance_attempts = await score_with_retries(
        evaluator.answer_relevancy,
        evaluator,
        user_input=record["question"],
        response=record["candidate_output"],
    )
    if record["retrieved_contexts"]:
        faithfulness, faithfulness_attempts = await score_with_retries(
            evaluator.faithfulness,
            evaluator,
            user_input=record["question"],
            retrieved_contexts=record["retrieved_contexts"],
            response=record["candidate_output"],
        )
    else:
        faithfulness = {
            "value": None,
            "reason": "not_applicable_without_retrieved_context",
        }
        faithfulness_attempts = []
    return {
        **record,
        "ragas_cache_key": cache_key(record),
        "metrics": {
            "answer_relevance": relevance,
            "faithfulness_to_retrieved_context": faithfulness,
        },
        "metric_attempts": {
            "answer_relevance": relevance_attempts,
            "faithfulness_to_retrieved_context": faithfulness_attempts,
        },
        "score_status": (
            "complete"
            if finite_metric(relevance)
            and (
                not record["retrieved_contexts"] or finite_metric(faithfulness)
            )
            else "metric_failure_retained"
        ),
        "scorer_version": SCORER_VERSION,
        "calibration_status": "diagnostic_not_human_validated",
        "completed_at_utc": utc_now(),
    }


async def run(args: argparse.Namespace) -> None:
    config = load_yaml(args.config)
    candidates = read_jsonl(args.candidates)
    eval_set = load_yaml(args.eval_set)
    records = prepare_records(candidates, eval_set)
    ids = [record["run_item_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate run_item_id values must be unique")
    if PHASE_B not in [Path(value) for value in sys.path if value]:
        sys.path.insert(0, str(PHASE_B))
    from W03_RAG_Expanded_MultiModel_RAGAS import LocalRAGASBundle

    judge = config["models"]["evaluator"]
    embedding = config["models"]["embedding"]
    embedding_dir = args.embedding_model_dir or Path(
        embedding["local_model_directory"]
    )
    endpoint = args.judge_endpoint or judge["endpoint"]
    evaluator = LocalRAGASBundle(
        judge_model=judge["model_id"],
        judge_base_url=endpoint,
        embedding_model_id=embedding["model_id"],
        embedding_model_dir=embedding_dir,
        embedding_device=args.embedding_device,
    )

    existing = read_jsonl(args.output)
    if existing and not args.resume:
        raise FileExistsError(f"{args.output} exists; use --resume")
    done = {row["run_item_id"] for row in existing}
    cache = {
        row["ragas_cache_key"]: row
        for row in existing
        if returned_metrics(row)
    }
    pending = [record for record in records if record["run_item_id"] not in done]
    if args.limit is not None:
        pending = pending[: args.limit]
    for start in range(0, len(pending), args.concurrency):
        batch = pending[start : start + args.concurrency]
        to_score: list[dict[str, Any]] = []
        results: dict[str, dict[str, Any]] = {}
        for record in batch:
            key = cache_key(record)
            if key in cache:
                results[record["run_item_id"]] = reused_row(record, cache[key])
            else:
                to_score.append(record)
        if to_score:
            scored = await asyncio.gather(
                *(score_record(record, evaluator) for record in to_score)
            )
            for result in scored:
                results[result["run_item_id"]] = result
                if returned_metrics(result):
                    cache[result["ragas_cache_key"]] = result
        for offset, record in enumerate(batch, start=start + 1):
            print(f"[ragas {offset}/{len(pending)}] {record['run_item_id']}", flush=True)
            result = results[record["run_item_id"]]
            result["judge"] = {
                "model_id": judge["model_id"],
                "model_revision": judge["model_revision"],
                "endpoint_scope": "loopback_only",
                "external_api_calls": False,
            }
            result["answer_relevance_embedding"] = {
                "model_id": embedding["model_id"],
                "model_revision": embedding["model_revision"],
                "local_model_directory": str(embedding_dir),
            }
            append_jsonl(args.output, result)
    final = read_jsonl(args.output)
    print(
        json.dumps(
            {
                "status": "completed",
                "rows_in_output": len(final),
                "finite_rows": sum(row["score_status"] == "complete" for row in final),
                "reused_rows": sum("reused_from_run_item_id" in row for row in final),
                "scorer_version": SCORER_VERSION,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--eval-set", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge-endpoint")
    parser.add_argument("--embedding-model-dir", type=Path)
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.concurrency < 1:
        raise ValueError("--concurrency must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

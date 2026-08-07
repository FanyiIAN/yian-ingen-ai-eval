"""Stream resumable local RAGAS diagnostics for the expanded three-model run.

The Mistral judge is served on loopback-only vLLM and answer-relevance uses the
local frozen BGE-M3 checkpoint.  No candidate data is sent to an external API.
Raw joined rubrics and scores remain outside the public repository.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from W03_RAG_Evaluation import (
    install_ragas_vertexai_import_compatibility,
    load_yaml,
    prepare_review_records,
    read_jsonl,
    require_approved_local_judge,
    sha256_file,
    timed_score,
    validate_generations,
)


RUNNER_VERSION = "0.4.0"
METRIC_ATTEMPTS = 2
CONTEXT_METRIC_NAMES = (
    "context_relevance",
    "context_recall",
    "context_precision",
)
DEFAULT_MODEL_KEYS = (
    "llama31_8b_instruct",
    "flan_t5_base",
    "mistral_7b_instruct_v0_2",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        handle.flush()


def finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def batches(values: list[Any], size: int) -> list[list[Any]]:
    if size < 1:
        raise ValueError("batch size must be positive")
    return [values[index : index + size] for index in range(0, len(values), size)]


def context_metric_cache_key(record: dict[str, Any]) -> str:
    payload = {
        "question": record["question"],
        "retrieved_contexts": record["retrieved_contexts"],
        "reference_answer": record["reference_answer"],
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


class LocalRAGASBundle:
    def __init__(
        self,
        judge_model: str,
        judge_base_url: str,
        embedding_model_id: str,
        embedding_model_dir: Path,
        embedding_device: str,
    ) -> None:
        import instructor
        import litellm

        install_ragas_vertexai_import_compatibility()
        from ragas.embeddings import HuggingFaceEmbeddings
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            ContextRelevance,
            Faithfulness,
        )

        parsed = urlparse(judge_base_url)
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise RuntimeError("RAGAS judge endpoint must be loopback-only")
        model_dir = embedding_model_dir.expanduser().resolve()
        if not model_dir.exists():
            raise FileNotFoundError(model_dir)

        client = instructor.from_litellm(
            litellm.acompletion,
            mode=instructor.Mode.JSON,
        )
        llm = llm_factory(
            f"hosted_vllm/{judge_model}",
            provider="litellm",
            client=client,
            adapter="litellm",
            api_base=judge_base_url,
            api_key="local-loopback-only",
            temperature=0,
            max_tokens=2048,
        )
        embeddings = HuggingFaceEmbeddings(
            model=str(model_dir),
            use_api=False,
            device=embedding_device,
            normalize_embeddings=True,
        )
        self.answer_relevancy = AnswerRelevancy(
            llm=llm,
            embeddings=embeddings,
        )
        self.faithfulness = Faithfulness(llm=llm)
        self.context_relevance = ContextRelevance(llm=llm)
        self.context_recall = ContextRecall(llm=llm)
        self.context_precision = ContextPrecision(llm=llm)
        self.judge_model = judge_model
        self.embedding_model_id = embedding_model_id
        self.embedding_model_dir = model_dir
        self._context_metric_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self.context_metric_cache_stats = {"computed": 0, "reused": 0, "seeded": 0}

    def seed_context_metric_cache(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            if row.get("condition") != "rag":
                continue
            cache_key = context_metric_cache_key(row)
            metrics = row.get("ragas", {}).get("metrics", {})
            for metric_name in CONTEXT_METRIC_NAMES:
                metric = metrics.get(metric_name, {})
                if not finite_number(metric.get("value")):
                    continue
                identity = (cache_key, metric_name)
                if identity in self._context_metric_cache:
                    continue
                self._context_metric_cache[identity] = {
                    "metric": dict(metric),
                    "source_run_item_id": row["run_item_id"],
                }
                self.context_metric_cache_stats["seeded"] += 1

    async def safe_score(self, metric: Any, **kwargs: Any) -> dict[str, Any]:
        started = time.perf_counter()
        errors: list[str] = []
        for attempt in range(1, METRIC_ATTEMPTS + 1):
            try:
                result = await timed_score(metric, **kwargs)
                result["attempts"] = attempt
                if errors:
                    result["retry_errors"] = errors
                return result
            except Exception as exc:  # Preserve transient evaluator failures.
                errors.append(f"{type(exc).__name__}: {str(exc)[:300]}")
                if attempt < METRIC_ATTEMPTS:
                    await asyncio.sleep(1.0)
        return {
            "value": None,
            "reason": errors[-1],
            "attempts": METRIC_ATTEMPTS,
            "retry_errors": errors,
            "latency_ms": round(
                (time.perf_counter() - started) * 1000,
                3,
            ),
        }

    async def score_record(self, record: dict[str, Any]) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        metrics["answer_relevance"] = await self.safe_score(
            self.answer_relevancy,
            user_input=record["question"],
            response=record["candidate_output"],
        )
        if record["condition"] == "rag":
            shared = {
                "user_input": record["question"],
                "retrieved_contexts": record["retrieved_contexts"],
            }
            metrics["faithfulness_to_retrieved_context"] = (
                await self.safe_score(
                    self.faithfulness,
                    **shared,
                    response=record["candidate_output"],
                )
            )
            cache_key = context_metric_cache_key(record)
            context_specs = (
                ("context_relevance", self.context_relevance, shared),
                (
                    "context_recall",
                    self.context_recall,
                    {**shared, "reference": record["reference_answer"]},
                ),
                (
                    "context_precision",
                    self.context_precision,
                    {**shared, "reference": record["reference_answer"]},
                ),
            )
            for metric_name, metric, metric_kwargs in context_specs:
                identity = (cache_key, metric_name)
                cached = self._context_metric_cache.get(identity)
                if cached is not None:
                    result = dict(cached["metric"])
                    result["latency_ms"] = None
                    result["reused_from_run_item_id"] = cached[
                        "source_run_item_id"
                    ]
                    result["reuse_reason"] = (
                        "candidate_invariant_shared_retrieval_input"
                    )
                    metrics[metric_name] = result
                    self.context_metric_cache_stats["reused"] += 1
                    continue
                result = await self.safe_score(metric, **metric_kwargs)
                metrics[metric_name] = result
                self.context_metric_cache_stats["computed"] += 1
                if finite_number(result.get("value")):
                    self._context_metric_cache[identity] = {
                        "metric": dict(result),
                        "source_run_item_id": record["run_item_id"],
                    }
        else:
            metrics.update(
                {
                    "faithfulness_to_retrieved_context": {
                        "value": None,
                        "reason": "not_applicable_without_retrieved_context",
                    },
                    "context_relevance": {
                        "value": None,
                        "reason": "not_applicable_to_base_condition",
                    },
                    "context_recall": {
                        "value": None,
                        "reason": "not_applicable_to_base_condition",
                    },
                    "context_precision": {
                        "value": None,
                        "reason": "not_applicable_to_base_condition",
                    },
                }
            )
        output = dict(record)
        output["ragas"] = {
            "library_version": importlib.metadata.version("ragas"),
            "api_generation": "collections_v0.4",
            "judge_provider": "local_vllm_litellm",
            "judge_model_id": self.judge_model,
            "judge_endpoint_scope": "loopback_only",
            "external_api_calls": False,
            "answer_relevance_embedding_model_id": self.embedding_model_id,
            "answer_relevance_embedding_model_directory": str(
                self.embedding_model_dir
            ),
            "structured_output": True,
            "metrics": metrics,
            "status": "automated_diagnostic_uncalibrated_local_judge",
        }
        return output


async def run(args: argparse.Namespace) -> None:
    eval_set = load_yaml(args.eval_set)
    candidate_payloads: dict[str, dict[str, Any]] = {}
    for key in args.model_key:
        candidate_dir = args.run_dir / "candidates" / key
        generation_path = candidate_dir / "generations.jsonl"
        config_path = candidate_dir / "config.yaml"
        rows = read_jsonl(generation_path)
        config = load_yaml(config_path)
        validation = validate_generations(rows, eval_set, config)
        judge = config["evaluation"]["judge"]
        embedding = config["evaluation"]["answer_relevance_embedding"]
        require_approved_local_judge(
            config,
            judge["model_id"],
            judge["endpoint"],
            Path(embedding["local_model_directory"]),
        )
        candidate_payloads[key] = {
            "rows": rows,
            "records": prepare_review_records(rows, eval_set),
            "config": config,
            "validation": validation,
            "generation_path": generation_path,
            "config_path": config_path,
        }

    first = candidate_payloads[args.model_key[0]]["config"]
    judge = first["evaluation"]["judge"]
    embedding = first["evaluation"]["answer_relevance_embedding"]
    evaluator = LocalRAGASBundle(
        judge_model=judge["model_id"],
        judge_base_url=judge["endpoint"],
        embedding_model_id=embedding["model_id"],
        embedding_model_dir=Path(embedding["local_model_directory"]),
        embedding_device=args.embedding_device,
    )

    manifest: dict[str, Any] = {
        "runner_version": RUNNER_VERSION,
        "created_at_utc": utc_now(),
        "evaluation_set": {
            "path": str(args.eval_set),
            "sha256": sha256_file(args.eval_set),
        },
        "judge_model_id": judge["model_id"],
        "judge_endpoint_scope": "loopback_only",
        "external_api_calls": False,
        "record_concurrency": args.concurrency,
        "model_order": list(args.model_key),
        "candidates": {},
    }
    existing_outputs: dict[str, list[dict[str, Any]]] = {}
    for key in args.model_key:
        output_path = args.output_dir / f"{key}_ragas.jsonl"
        existing = read_jsonl(output_path) if output_path.exists() else []
        existing_outputs[key] = existing
        evaluator.seed_context_metric_cache(existing)
    for key in args.model_key:
        payload = candidate_payloads[key]
        output_path = args.output_dir / f"{key}_ragas.jsonl"
        existing = existing_outputs[key]
        if existing and not args.resume:
            raise FileExistsError(
                f"{output_path} exists; use --resume or a new output directory"
            )
        completed = {row["run_item_id"] for row in existing}
        pending = [
            record
            for record in payload["records"]
            if record["run_item_id"] not in completed
        ]
        if args.limit_per_model is not None:
            pending = pending[: args.limit_per_model]
        completed_in_session = 0
        for batch in batches(pending, args.concurrency):
            for offset, record in enumerate(batch, start=1):
                index = completed_in_session + offset
                print(
                    f"[{key} {index}/{len(pending)}] {record['run_item_id']}",
                    flush=True,
                )
            scored_batch = await asyncio.gather(
                *(evaluator.score_record(record) for record in batch)
            )
            for scored in scored_batch:
                append_jsonl(output_path, scored)
            completed_in_session += len(batch)
        all_rows = read_jsonl(output_path) if output_path.exists() else []
        invalid_metrics = sum(
            not finite_number(metric.get("value"))
            and not str(metric.get("reason", "")).startswith("not_applicable")
            for row in all_rows
            for metric in row.get("ragas", {}).get("metrics", {}).values()
        )
        manifest["candidates"][key] = {
            "generation_path": str(payload["generation_path"]),
            "generation_sha256": sha256_file(payload["generation_path"]),
            "config_path": str(payload["config_path"]),
            "config_sha256": sha256_file(payload["config_path"]),
            "output_path": str(output_path),
            "output_sha256": sha256_file(output_path),
            "rows": len(all_rows),
            "invalid_metric_values": invalid_metrics,
            "validation": payload["validation"],
            "judge_independent_from_candidate": payload["config"][
                "evaluation"
            ]["judge"].get("independent_from_candidate"),
        }
        manifest["context_metric_cache"] = {
            "finite_values_only": True,
            "candidate_invariant_metrics": list(CONTEXT_METRIC_NAMES),
            "entries": len(evaluator._context_metric_cache),
            "stats": dict(evaluator.context_metric_cache_stats),
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--eval-set", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--model-key",
        action="append",
        choices=DEFAULT_MODEL_KEYS,
    )
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--limit-per-model", type=int)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.concurrency < 1:
        raise ValueError("--concurrency must be positive")
    args.model_key = args.model_key or list(DEFAULT_MODEL_KEYS)
    if args.manifest.exists() and not args.resume:
        raise FileExistsError(
            f"{args.manifest} exists; use --resume or a new manifest"
        )
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

"""Prepare hidden point rubrics and optionally run local RAGAS v0.4 metrics.

The candidate output file comes from W03_RAG_Generation.py. `prepare-review`
joins it with the hidden answer key without making a model call.
`score-ragas-local` uses the frozen Week-2 Mistral checkpoint through a
loopback-only vLLM server and local BGE-M3 embeddings. It never calls an
external model API.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_EVAL_SET = SCRIPT_DIR / "W03_RAG_Eval_Set.yaml"
DEFAULT_CONFIG = SCRIPT_DIR / "W03_RAG_Run_Config.yaml"
EVALUATOR_VERSION = "0.2.4"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


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


def ensure_private_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise ValueError(
            "Joined answer keys and raw metric outputs must not be written "
            "inside the public repository."
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def validate_generations(
    rows: list[dict[str, Any]],
    eval_set: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    expected_rows = int(config["comparison"]["expected_generation_rows"])
    expected_items = int(config["comparison"]["expected_evaluation_items"])
    expected_conditions = set(config["comparison"]["conditions"])
    known_eval_ids = {item["eval_id"] for item in eval_set["items"]}
    seen_run_ids: set[str] = set()
    conditions_by_eval: dict[str, set[str]] = defaultdict(set)

    if len(rows) != expected_rows:
        errors.append(f"Expected {expected_rows} generations, found {len(rows)}")
    for row in rows:
        run_item_id = row.get("run_item_id")
        eval_id = row.get("eval_id")
        condition = row.get("condition")
        if not run_item_id:
            errors.append("A row lacks run_item_id")
            continue
        if run_item_id in seen_run_ids:
            errors.append(f"Duplicate run_item_id: {run_item_id}")
        seen_run_ids.add(run_item_id)
        if eval_id not in known_eval_ids:
            errors.append(f"{run_item_id}: unknown eval_id {eval_id}")
        conditions_by_eval[eval_id].add(condition)
        if row.get("status") != "completed":
            errors.append(f"{run_item_id}: status is not completed")
        if not isinstance(row.get("candidate_output"), str):
            errors.append(f"{run_item_id}: missing candidate_output")
        contexts = row.get("retrieved_contexts") or []
        if condition == "base" and contexts:
            errors.append(f"{run_item_id}: base row exposes retrieved context")
        if condition == "rag" and not contexts:
            errors.append(f"{run_item_id}: RAG row lacks retrieved context")

    for eval_id, conditions in conditions_by_eval.items():
        if conditions != expected_conditions:
            errors.append(
                f"{eval_id}: expected {expected_conditions}, found {conditions}"
            )
    if len(conditions_by_eval) != expected_items:
        errors.append(
            f"Expected {expected_items} paired items, found {len(conditions_by_eval)}"
        )
    if errors:
        raise ValueError("Generation validation failed:\n- " + "\n- ".join(errors))

    return {
        "status": "ok",
        "rows": len(rows),
        "paired_eval_ids": len(conditions_by_eval),
        "condition_counts": dict(Counter(row["condition"] for row in rows)),
        "candidate_model_id": rows[0]["candidate_model_id"] if rows else None,
        "candidate_model_revision": (
            rows[0].get("runtime", {}).get("model_revision") if rows else None
        ),
    }


def prepare_review_records(
    rows: list[dict[str, Any]],
    eval_set: dict[str, Any],
) -> list[dict[str, Any]]:
    item_by_id = {item["eval_id"]: item for item in eval_set["items"]}
    records: list[dict[str, Any]] = []
    for row in rows:
        item = item_by_id[row["eval_id"]]
        required_points = item["required_points"]
        records.append(
            {
                "run_item_id": row["run_item_id"],
                "eval_id": row["eval_id"],
                "platform": row["platform"],
                "condition": row["condition"],
                "question": item["question"],
                "candidate_output": row["candidate_output"],
                "retrieved_contexts": [
                    context["content"]
                    for context in row.get("retrieved_contexts") or []
                ],
                "retrieved_chunk_ids": [
                    context["chunk_id"]
                    for context in row.get("retrieved_contexts") or []
                ],
                "reference_document_ids": item["reference_document_ids"],
                "evidence_fact_ids": item["evidence_fact_ids"],
                "reference_answer": item["reference_answer"],
                "required_points": required_points,
                "forbidden_points": item.get("forbidden_points") or [],
                "point_score_template": {
                    "required_weight_earned": None,
                    "required_weight_possible": sum(
                        point["weight"] for point in required_points
                    ),
                    "required_point_coverage": None,
                    "forbidden_point_violations": [],
                    "per_point_verdicts": [
                        {
                            "point_id": point["point_id"],
                            "verdict": None,
                            "evidence": None,
                        }
                        for point in required_points
                    ],
                    "status": "pending_independent_judge_and_human_review",
                },
                "ragas_applicability": {
                    "answer_relevance": True,
                    "faithfulness_to_retrieved_context": row["condition"] == "rag",
                    "context_relevance": row["condition"] == "rag",
                    "context_recall": row["condition"] == "rag",
                    "context_precision": row["condition"] == "rag",
                },
            }
        )
    return records


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def metric_result_payload(result: Any, latency_ms: float) -> dict[str, Any]:
    return {
        "value": float(result.value),
        "reason": getattr(result, "reason", None),
        "latency_ms": round(latency_ms, 3),
    }


async def timed_score(metric: Any, **kwargs: Any) -> dict[str, Any]:
    started = time.perf_counter()
    result = await metric.ascore(**kwargs)
    return metric_result_payload(
        result,
        (time.perf_counter() - started) * 1000,
    )


def install_ragas_vertexai_import_compatibility() -> bool:
    """Bridge a removed LangChain import that RAGAS 0.4.3 still imports.

    The Week 3 evaluator never instantiates VertexAI. The placeholder class
    only lets RAGAS import while the actual judge remains the loopback-only
    LiteLLM/vLLM Mistral service.
    """
    module_name = "langchain_community.chat_models.vertexai"
    try:
        __import__(module_name)
        return False
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise

    import sys
    import types

    compatibility_module = types.ModuleType(module_name)

    class ChatVertexAI:
        pass

    compatibility_module.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = compatibility_module
    return True


async def score_ragas_local(
    records: list[dict[str, Any]],
    judge_model: str,
    judge_base_url: str,
    answer_embedding_model_id: str,
    answer_embedding_model_dir: Path,
    embedding_device: str,
    limit: int | None,
) -> list[dict[str, Any]]:
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

    parsed_endpoint = urlparse(judge_base_url)
    if parsed_endpoint.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError(
            "The local RAGAS judge endpoint must resolve to loopback. External "
            "or hosted evaluator endpoints are prohibited for this run."
        )
    model_dir = answer_embedding_model_dir.expanduser().resolve()
    if not model_dir.exists():
        raise RuntimeError(
            f"Local answer-relevance embedding checkpoint is missing: {model_dir}"
        )

    evaluator_client = instructor.from_litellm(
        litellm.acompletion,
        mode=instructor.Mode.JSON,
    )
    evaluator_llm = llm_factory(
        f"hosted_vllm/{judge_model}",
        provider="litellm",
        client=evaluator_client,
        adapter="litellm",
        api_base=judge_base_url,
        api_key="local-loopback-only",
        temperature=0,
        max_tokens=2048,
    )
    evaluator_embeddings = HuggingFaceEmbeddings(
        model=str(model_dir),
        use_api=False,
        device=embedding_device,
        normalize_embeddings=True,
    )
    answer_relevancy = AnswerRelevancy(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )
    faithfulness = Faithfulness(llm=evaluator_llm)
    context_relevance = ContextRelevance(llm=evaluator_llm)
    context_recall = ContextRecall(llm=evaluator_llm)
    context_precision = ContextPrecision(llm=evaluator_llm)

    selected = records if limit is None else records[:limit]
    scored: list[dict[str, Any]] = []
    for index, record in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] {record['run_item_id']}", flush=True)
        metrics: dict[str, Any] = {}
        metrics["answer_relevance"] = await timed_score(
            answer_relevancy,
            user_input=record["question"],
            response=record["candidate_output"],
        )
        if record["condition"] == "rag":
            shared = {
                "user_input": record["question"],
                "retrieved_contexts": record["retrieved_contexts"],
            }
            metrics["faithfulness_to_retrieved_context"] = await timed_score(
                faithfulness,
                **shared,
                response=record["candidate_output"],
            )
            metrics["context_relevance"] = await timed_score(
                context_relevance,
                **shared,
            )
            metrics["context_recall"] = await timed_score(
                context_recall,
                **shared,
                reference=record["reference_answer"],
            )
            metrics["context_precision"] = await timed_score(
                context_precision,
                **shared,
                reference=record["reference_answer"],
            )
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
            "judge_model_id": judge_model,
            "judge_endpoint_scope": "loopback_only",
            "external_api_calls": False,
            "answer_relevance_embedding_model_id": answer_embedding_model_id,
            "answer_relevance_embedding_model_directory": str(model_dir),
            "structured_output": True,
            "metrics": metrics,
            "status": "automated_provisional_pending_calibration_and_human_review",
        }
        scored.append(output)
    return scored


def require_approved_local_judge(
    config: dict[str, Any],
    judge_model: str,
    judge_base_url: str,
    answer_embedding_model_dir: Path,
) -> None:
    judge = config["evaluation"]["judge"]
    if judge.get("provider") != "local_vllm_litellm":
        raise ValueError(
            "The run config must authorize the loopback-only local vLLM/LiteLLM "
            "evaluator."
        )
    if judge.get("model_id") != judge_model:
        raise ValueError(
            f"Judge model mismatch: config={judge.get('model_id')}, "
            f"requested={judge_model}"
        )
    if judge.get("endpoint") != judge_base_url:
        raise ValueError(
            f"Judge endpoint mismatch: config={judge.get('endpoint')}, "
            f"requested={judge_base_url}"
        )
    if judge.get("external_api_calls") is not False:
        raise ValueError("External evaluator API calls must be disabled")
    configured_embedding_dir = Path(
        config["evaluation"]["answer_relevance_embedding"][
            "local_model_directory"
        ]
    )
    if configured_embedding_dir != answer_embedding_model_dir:
        raise ValueError(
            "Answer-relevance embedding directory mismatch: "
            f"config={configured_embedding_dir}, "
            f"requested={answer_embedding_model_dir}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")

    prepare = subparsers.add_parser(
        "prepare-review",
        help="Join hidden rubrics without making evaluator calls.",
    )
    prepare.add_argument("--output", type=Path, required=True)

    ragas_parser = subparsers.add_parser(
        "score-ragas-local",
        help=(
            "Run RAGAS v0.4 with a loopback-only local Mistral judge and "
            "local BGE-M3 embeddings."
        ),
    )
    ragas_parser.add_argument("--output", type=Path, required=True)
    ragas_parser.add_argument("--judge-model")
    ragas_parser.add_argument("--judge-base-url")
    ragas_parser.add_argument(
        "--answer-embedding-model-dir",
        type=Path,
    )
    ragas_parser.add_argument(
        "--embedding-device",
        choices=["cpu", "cuda"],
        default="cuda",
    )
    ragas_parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.generations)
    eval_set = load_yaml(args.eval_set)
    config = load_yaml(args.config)
    validation = validate_generations(rows, eval_set, config)
    if args.command == "validate":
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return

    records = prepare_review_records(rows, eval_set)
    output = ensure_private_output(args.output)
    if args.command == "prepare-review":
        write_jsonl(output, records)
        print(f"Wrote {len(records)} hidden review rows to {output}")
        return

    if args.command == "score-ragas-local":
        judge_config = config["evaluation"]["judge"]
        embedding_config = config["evaluation"]["answer_relevance_embedding"]
        judge_model = args.judge_model or judge_config["model_id"]
        judge_base_url = args.judge_base_url or judge_config["endpoint"]
        embedding_model_dir = (
            args.answer_embedding_model_dir
            or Path(embedding_config["local_model_directory"])
        )
        require_approved_local_judge(
            config,
            judge_model,
            judge_base_url,
            embedding_model_dir,
        )
        scored = asyncio.run(
            score_ragas_local(
                records=records,
                judge_model=judge_model,
                judge_base_url=judge_base_url,
                answer_embedding_model_id=embedding_config["model_id"],
                answer_embedding_model_dir=embedding_model_dir,
                embedding_device=args.embedding_device,
                limit=args.limit,
            )
        )
        write_jsonl(output, scored)
        manifest = {
            "evaluator_version": EVALUATOR_VERSION,
            "generation_file_sha256": sha256_file(args.generations),
            "evaluation_set_file_sha256": sha256_file(args.eval_set),
            "config_file_sha256": sha256_file(args.config),
            "rows": len(scored),
            "validation": validation,
        }
        manifest_path = output.with_suffix(output.suffix + ".manifest.json")
        with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(f"Wrote {len(scored)} scored rows to {output}")
        return

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()

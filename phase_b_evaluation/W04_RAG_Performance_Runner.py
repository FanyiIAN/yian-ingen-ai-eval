"""Measure online public-RAG latency, resources, and exact request evidence.

This Week 4 runner reuses the frozen Week 3 public knowledge base, blind
evaluation set, LangChain/Chroma retrieval behavior, and Llama prompt contract.
Unlike the Week 3 offline-input workflow, retrieval and generation occur inside
one request profiler so ``question_to_response_ms`` represents the complete
question -> retrieve -> prompt -> answer path.

Generated outputs, persistent Chroma data, and run manifests belong on the
RunPod persistent volume or the private experiment area, never in Git.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import W03_RAG_Pipeline as rag
from W04_Resource_Monitor import (
    RequestProfiler,
    directory_size_bytes,
    environment_manifest,
    validate_jsonable,
)
from W04_Text_Robustness_Runner import (
    LocalTextEngine,
    append_jsonl,
    atomic_write_json,
    read_jsonl,
    sha256_file,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_KB = SCRIPT_DIR / "W03_RAG_Official_Knowledge_Base_v0.3.0.yaml"
DEFAULT_EVAL = SCRIPT_DIR / "W03_RAG_Official_Blind_Eval_Set_v0.4.0.yaml"
DEFAULT_CONFIG = SCRIPT_DIR / "W03_RAG_Official_Run_Config_v0.4.1.yaml"
RUNNER_VERSION = "0.1.0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_sha256(value: Any) -> str:
    return rag.canonical_json_sha256(value)


def validate_contract(
    kb: dict[str, Any],
    eval_set: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    validation = rag.validate_assets(kb, eval_set, config)
    conditions = list(config["comparison"]["conditions"])
    if conditions != ["base", "rag"]:
        raise ValueError("online performance run requires conditions [base, rag]")
    if kb.get("data_origin") != "official_public_curated":
        raise ValueError("Week 4 RAG performance run is public collection only")
    if config["retrieval"]["vector_store"].get("persistence") is not True:
        raise ValueError("persistent Chroma is required")
    generation = config["generation"]
    if generation.get("candidate_model_id") != "meta-llama/Llama-3.1-8B-Instruct":
        raise ValueError("frozen performance candidate must be Llama-3.1-8B-Instruct")
    if generation["decoding"].get("do_sample") is not False:
        raise ValueError("deterministic decoding is required")
    return {
        "validation": validation,
        "evaluation_items": len(eval_set["items"]),
        "official_request_count": len(eval_set["items"]) * len(conditions),
        "rag_request_count": len(eval_set["items"]),
        "base_request_count": len(eval_set["items"]),
    }


def model_config_from_rag(config: dict[str, Any]) -> dict[str, Any]:
    generation = config["generation"]
    return {
        "model_key": "llama31_8b_instruct",
        "model_id": generation["candidate_model_id"],
        "revision": generation["candidate_model_revision"],
        "precision": generation["precision"],
        "serialization": "official_tokenizer_chat_template",
    }


def generation_config_from_rag(config: dict[str, Any]) -> dict[str, Any]:
    generation = config["generation"]
    return {
        "max_input_tokens": int(generation["max_input_tokens"]),
        "max_new_tokens": int(generation["decoding"]["max_new_tokens"]),
    }


def request_id(item: dict[str, Any], condition: str) -> str:
    return f"w04-rag-perf::{item['eval_id']}::{condition}"


def run_one(
    item: dict[str, Any],
    condition: str,
    *,
    kb: dict[str, Any],
    config: dict[str, Any],
    store: Any,
    reranker: Any | None,
    engine: Any,
    cold_or_warm: str,
    profiler_factory: Any = RequestProfiler,
    run_item_id: str | None = None,
) -> dict[str, Any]:
    if condition not in {"base", "rag"}:
        raise ValueError(f"unsupported condition: {condition}")
    rid = run_item_id or request_id(item, condition)
    profiler = profiler_factory(rid)
    documents: list[Any] = []
    retrieved_contexts: list[dict[str, Any]] = []
    messages: list[dict[str, str]] = []
    generated: dict[str, Any] | None = None
    retrieval_latency_ms: float | None = None

    try:
        with profiler:
            with profiler.stage("input_load_ms"):
                question = str(item["question"]).strip()
                if not question:
                    raise ValueError(f"{item.get('eval_id')} has an empty question")

            if condition == "rag":
                with profiler.stage("retrieval_total_ms"):
                    documents, retrieval_latency_ms = rag.retrieve_item(
                        item,
                        store,
                        config,
                        kb=kb,
                        reranker=reranker,
                        profiler=profiler,
                    )

            with profiler.stage("context_assembly_ms"):
                formatted_context = rag.format_context(documents)
                if condition == "rag":
                    retrieved_contexts = rag.retrieval_trace(documents)

            with profiler.stage("prompt_build_ms"):
                messages = rag.render_candidate_messages(
                    item,
                    condition,
                    documents,
                    data_origin=kb["data_origin"],
                    base_system_prompt=config["generation"].get(
                        "base_system_prompt"
                    ),
                    rag_system_prompt=config["generation"].get(
                        "rag_system_prompt"
                    ),
                    formatted_context=formatted_context,
                )
            generated = engine.generate_messages(messages, profiler)
    finally:
        profile = profiler.result()

    if generated is None:
        raise RuntimeError("generation returned no result")

    event = {
        "run_item_id": rid,
        "eval_id": item["eval_id"],
        "platform": item["platform"],
        "evaluation_family": "rag_performance",
        "condition_id": condition,
        "condition": condition,
        "question": question,
        "question_sha256": rag.sha256_text(question),
        "candidate_messages": messages,
        "candidate_messages_sha256": canonical_sha256(messages),
        "retrieved_contexts": retrieved_contexts,
        "retrieval_latency_returned_ms": (
            round(retrieval_latency_ms, 6)
            if retrieval_latency_ms is not None
            else None
        ),
        "rag_stage_availability": {
            "available": condition == "rag",
            "reason": (
                None
                if condition == "rag"
                else "base condition intentionally performs no retrieval"
            ),
            "vector_search_measurement": (
                "inclusive Chroma integration call; query_embedding_ms is nested"
                if condition == "rag"
                else None
            ),
        },
        "candidate_model_key": "llama31_8b_instruct",
        "candidate_model_id": config["generation"]["candidate_model_id"],
        "candidate_model_revision": config["generation"][
            "candidate_model_revision"
        ],
        "tokenizer_revision": config["generation"]["tokenizer_revision"],
        "precision": config["generation"]["precision"],
        "seed": int(config["random_seed"]),
        "do_sample": False,
        "cold_or_warm": cold_or_warm,
        "quality_status": "unscored_system_performance_companion_run",
        "request_profile": profile,
        "runner_version": RUNNER_VERSION,
        "completed_at_utc": utc_now(),
        **generated,
    }
    validate_jsonable(event)
    return event


def event_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "events": run_dir / "W04_RAG_Performance_Events.jsonl",
        "warmups": run_dir / "W04_RAG_Performance_Warmups.jsonl",
        "candidates": run_dir / "W04_RAG_Performance_Candidates.jsonl",
        "traces": run_dir / "W04_RAG_Performance_Request_Traces.jsonl",
        "sessions": run_dir / "W04_RAG_Performance_Run_Sessions.jsonl",
        "manifest": run_dir / "W04_RAG_Performance_Run_Manifest.json",
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        handle.flush()
    temporary.replace(path)


def materialize_views(events: list[dict[str, Any]], paths: dict[str, Path]) -> None:
    candidates = [
        {key: value for key, value in event.items() if key != "request_profile"}
        for event in events
    ]
    traces = [
        {
            "run_item_id": event["run_item_id"],
            "eval_id": event["eval_id"],
            "platform": event["platform"],
            "evaluation_family": event.get("evaluation_family", "rag_performance"),
            "condition_id": event.get("condition_id", event["condition"]),
            "condition": event["condition"],
            "candidate_model_key": event["candidate_model_key"],
            "candidate_model_revision": event["candidate_model_revision"],
            "cold_or_warm": event["cold_or_warm"],
            "prompt_tokens": event["prompt_tokens"],
            "output_tokens": event["output_tokens"],
            "total_tokens": event["total_tokens"],
            "rag_stage_availability": event["rag_stage_availability"],
            "request_profile": event["request_profile"],
        }
        for event in events
    ]
    write_jsonl(paths["candidates"], candidates)
    write_jsonl(paths["traces"], traces)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb", type=Path, default=DEFAULT_KB)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--persist-dir", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--embedding-device", default="cuda:0")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kb, eval_set, config = rag.load_assets(args.kb, args.eval_set, args.config)
    contract = validate_contract(kb, eval_set, config)
    if args.validate_only:
        print(json.dumps(contract, indent=2, sort_keys=True))
        return

    if args.persist_dir is None or args.run_dir is None:
        raise ValueError("--persist-dir and --run-dir are required for a real run")
    persist_dir = rag.ensure_external_persist_directory(args.persist_dir)
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = event_paths(run_dir)
    if paths["events"].exists() and not args.resume:
        raise FileExistsError(
            f"{paths['events']} exists; use --resume or a new immutable run directory"
        )

    generation = config["generation"]
    model_dir = (args.model_dir or Path(generation["local_model_directory"])).resolve()
    model_config = model_config_from_rag(config)
    engine_generation = generation_config_from_rag(config)
    session_id = f"w04-rag-perf-{utc_now().replace(':', '')}-{uuid.uuid4().hex[:8]}"
    session_started = time.perf_counter()

    index_result, store = rag.index_documents(
        kb,
        config,
        persist_dir,
        args.embedding_device,
        profile_query_embeddings=True,
    )
    reranker = rag.build_reranker(config, args.embedding_device)
    engine = LocalTextEngine(
        model_dir,
        model_config,
        engine_generation,
        int(config["random_seed"]),
        gpu_index=args.gpu_index,
    )

    # Warm both retrieval and generation.  The evidence is retained separately
    # and never enters quality or steady-state aggregates.
    warmup_item = eval_set["items"][0]
    warmup = run_one(
        warmup_item,
        "rag",
        kb=kb,
        config=config,
        store=store,
        reranker=reranker,
        engine=engine,
        cold_or_warm="warmup_excluded",
        run_item_id=f"warmup::{session_id}::{warmup_item['eval_id']}::rag",
    )
    warmup["excluded_from_quality"] = True
    warmup["excluded_from_steady_state_latency"] = True
    append_jsonl(paths["warmups"], warmup)

    existing = read_jsonl(paths["events"]) if paths["events"].exists() else []
    completed = {row["run_item_id"] for row in existing}
    planned = [
        (item, condition)
        for item in eval_set["items"]
        for condition in config["comparison"]["conditions"]
        if request_id(item, condition) not in completed
    ]
    if args.limit is not None:
        planned = planned[: args.limit]

    generated_count = 0
    for index, (item, condition) in enumerate(planned, start=1):
        print(
            f"[{index}/{len(planned)}] {item['eval_id']}::{condition}",
            flush=True,
        )
        event = run_one(
            item,
            condition,
            kb=kb,
            config=config,
            store=store,
            reranker=reranker,
            engine=engine,
            cold_or_warm="warm_steady_state",
        )
        append_jsonl(paths["events"], event)
        generated_count += 1

    events = read_jsonl(paths["events"]) if paths["events"].exists() else []
    materialize_views(events, paths)
    session = {
        "session_id": session_id,
        "started_at_utc": warmup["completed_at_utc"],
        "ended_at_utc": utc_now(),
        "elapsed_ms": round((time.perf_counter() - session_started) * 1000, 6),
        "rows_before_session": len(existing),
        "rows_generated_this_session": generated_count,
        "rows_after_session": len(events),
        "resume": bool(args.resume),
        "limit": args.limit,
        "warmup_request_id": warmup["run_item_id"],
    }
    append_jsonl(paths["sessions"], session)

    embedding_dir = Path(config["retrieval"]["embedding"]["local_model_directory"])
    manifest = {
        "run_id": run_dir.name,
        "runner_version": RUNNER_VERSION,
        "pipeline_version": rag.PIPELINE_VERSION,
        "created_at_utc": utc_now(),
        "claim_boundary": config["claim_boundary"],
        "request_contract": contract,
        "completed_official_rows": len(events),
        "expected_official_rows": contract["official_request_count"],
        "input_assets": {
            "knowledge_base": {"path": str(args.kb), "sha256": sha256_file(args.kb)},
            "evaluation_set": {
                "path": str(args.eval_set),
                "sha256": sha256_file(args.eval_set),
            },
            "config": {"path": str(args.config), "sha256": sha256_file(args.config)},
        },
        "index_result": index_result,
        "retrieval_measurement": {
            "query_embedding_ms": "nested exact adapter timing",
            "vector_search_ms": (
                "inclusive Chroma string-query call, including query embedding; "
                "do not sum with query_embedding_ms"
            ),
            "retrieval_total_ms": "complete retrieve_item call",
            "context_assembly_ms": "retrieval trace plus candidate-visible context",
        },
        "model_runtime": engine.runtime_metadata(),
        "environment": environment_manifest(
            model_path=model_dir,
            gpu_index=args.gpu_index,
        ),
        "rag_component_storage": {
            "embedding_checkpoint_path": str(embedding_dir),
            "embedding_checkpoint_bytes": directory_size_bytes(embedding_dir),
            "chroma_persist_directory": str(persist_dir),
            "chroma_persist_bytes": directory_size_bytes(persist_dir),
            "reranker_enabled": reranker is not None,
        },
        "latest_session": session,
    }
    atomic_write_json(paths["manifest"], manifest)
    print(json.dumps(session, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Execute the registered Week 5 Senpai RAG full-factorial experiment.

The runner reuses the frozen Week 3 pipeline, but it creates one shared BGE-M3
runtime for the three chunk indexes so that seed-42 randomisation can interleave
all 18 variant blocks. Raw prompts, contexts, outputs, and runtime metadata are
append-only private run evidence; public summaries are produced by the separate
Week 5 analysis script.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import itertools
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import yaml


RUNNER_VERSION = "1.0.0"
HERE = Path(__file__).resolve().parent
PHASE_B = HERE.parent / "phase_b_evaluation"
DEFAULT_CONFIG = HERE / "W05_RAG_Optimisation_Run_Config_v1.0.0.yaml"


@dataclass(frozen=True, order=True)
class Variant:
    chunk_size_tokens: int
    top_k: int
    reranking: str

    @property
    def variant_id(self) -> str:
        suffix = "ce" if self.reranking == "cross_encoder" else "none"
        return (
            f"chunk-{self.chunk_size_tokens}_topk-{self.top_k}_"
            f"rerank-{suffix}"
        )


class StageProfiler:
    """Small request-local timer compatible with the Week 3 profiler hook."""

    def __init__(self) -> None:
        self.stages_ms: dict[str, float] = {}

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - started) * 1000
            self.stages_ms[name] = self.stages_ms.get(name, 0.0) + elapsed

    def rounded(self) -> dict[str, float]:
        return {key: round(value, 3) for key, value in self.stages_ms.items()}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        handle.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_variants(config: dict[str, Any]) -> list[Variant]:
    design = config["factorial_design"]
    return [
        Variant(int(chunk), int(top_k), str(reranking))
        for chunk, top_k, reranking in itertools.product(
            design["chunk_size_tokens"],
            design["top_k"],
            design["reranking"],
        )
    ]


def randomized_variants(config: dict[str, Any]) -> list[Variant]:
    variants = build_variants(config)
    random.Random(int(config["random_seed"])).shuffle(variants)
    return variants


def matched_variant_pairs(variants: list[Variant]) -> list[tuple[Variant, Variant, str]]:
    output: list[tuple[Variant, Variant, str]] = []
    fields = ("chunk_size_tokens", "top_k", "reranking")
    for left, right in itertools.combinations(sorted(variants), 2):
        differences = [
            field for field in fields if getattr(left, field) != getattr(right, field)
        ]
        if len(differences) == 1:
            output.append((left, right, differences[0]))
    return output


def source_path(config_path: Path, registered: str) -> Path:
    return (config_path.parent / registered).resolve()


def validate_protocol(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    errors: list[str] = []
    design = config.get("factorial_design") or {}
    expected = {
        "chunk_size_tokens": [256, 512, 1024],
        "top_k": [1, 3, 5],
        "reranking": ["none", "cross_encoder"],
    }
    for key, values in expected.items():
        if list(design.get(key) or []) != values:
            errors.append(f"{key} must equal {values}")
    variants = build_variants(config) if not errors else []
    if len(variants) != 18 or design.get("expected_configurations") != 18:
        errors.append("factorial design must contain exactly 18 variants")
    expected_rows = 18 * int(
        config["source_inputs"]["evaluation_set"]["item_count"]
    )
    if int(design.get("expected_rows", -1)) != expected_rows:
        errors.append(f"expected_rows must equal {expected_rows}")
    if config.get("random_seed") != 42:
        errors.append("formal protocol seed must remain 42")
    if config["runtime_controls"].get("warmup_requests_per_loaded_runtime") != 1:
        errors.append("exactly one unmeasured warm-up is required")

    source_hashes: dict[str, str] = {}
    for key in ("knowledge_base", "evaluation_set", "week3_run_config"):
        entry = config["source_inputs"][key]
        path = source_path(config_path, entry["path"])
        if not path.is_file():
            errors.append(f"missing registered source: {path}")
            continue
        observed = sha256_file(path)
        source_hashes[key] = observed
        if observed.lower() != str(entry["sha256"]).lower():
            errors.append(f"{key} SHA-256 mismatch")

    model_revisions = {
        key: value.get("model_revision")
        for key, value in config.get("models", {}).items()
    }
    if any(not value for value in model_revisions.values()):
        errors.append("every model must have an exact revision")
    if errors:
        raise ValueError("invalid Week 5 production protocol:\n- " + "\n- ".join(errors))

    pairs = matched_variant_pairs(variants)
    by_factor: dict[str, int] = {}
    for _, _, factor in pairs:
        by_factor[factor] = by_factor.get(factor, 0) + 1
    return {
        "status": "validated",
        "runner_version": RUNNER_VERSION,
        "experiment_id": config["experiment_id"],
        "experiment_version": config["experiment_version"],
        "variants": len(variants),
        "expected_rows": expected_rows,
        "matched_pairs": len(pairs),
        "matched_pairs_by_factor": by_factor,
        "randomized_variant_order": [
            variant.variant_id for variant in randomized_variants(config)
        ],
        "source_hashes": source_hashes,
        "model_revisions": model_revisions,
        "claim_boundary": config["claim_boundary"],
    }


def freeze_senpai_subset(
    eval_set: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    registered = config["source_inputs"]["evaluation_set"]
    wanted = list(registered["item_ids"])
    by_id = {str(item["eval_id"]): item for item in eval_set["items"]}
    missing = sorted(set(wanted) - set(by_id))
    if missing:
        raise ValueError(f"registered Senpai items are absent: {missing}")
    items = [copy.deepcopy(by_id[eval_id]) for eval_id in wanted]
    if any(item.get("platform") != "Senpai" for item in items):
        raise ValueError("the frozen subset contains a non-Senpai item")
    subset = copy.deepcopy(eval_set)
    subset.update(
        {
            "evaluation_set_id": registered["derived_subset_id"],
            "evaluation_set_version": registered["derived_subset_version"],
            "item_count": len(items),
            "platforms": ["Senpai"],
            "items": items,
            "source_evaluation_set_id": registered["source_id"],
            "source_evaluation_set_version": registered["source_version"],
            "subset_rule": "exact_registered_item_ids_in_registered_order",
        }
    )
    if len(items) != int(registered["item_count"]):
        raise ValueError("frozen Senpai item count mismatch")
    return subset


def variant_week3_config(
    base: dict[str, Any], variant: Variant, item_count: int
) -> dict[str, Any]:
    output = copy.deepcopy(base)
    output["comparison"].update(
        {
            "expected_evaluation_items": item_count,
            "expected_generation_rows": item_count,
            "conditions": ["rag"],
        }
    )
    output["retrieval"]["text_splitter"]["chunk_size_tokens"] = (
        variant.chunk_size_tokens
    )
    output["retrieval"]["retriever"]["top_k"] = variant.top_k
    output["retrieval"]["retriever"]["fetch_k"] = 32
    output["retrieval"]["reranker"]["enabled"] = (
        variant.reranking == "cross_encoder"
    )
    output["iteration_policy"]["run_id"] = (
        f"w05-senpai-rag-optimisation-v1.0.0-seed42::{variant.variant_id}"
    )
    return output


def apply_model_directory_overrides(
    week3_config: dict[str, Any], args: argparse.Namespace
) -> None:
    if args.generator_dir:
        week3_config["generation"]["local_model_directory"] = str(args.generator_dir)
    if args.embedding_dir:
        week3_config["retrieval"]["embedding"]["local_model_directory"] = str(
            args.embedding_dir
        )
        week3_config["evaluation"]["answer_relevance_embedding"][
            "local_model_directory"
        ] = str(args.embedding_dir)
    if args.reranker_dir:
        week3_config["retrieval"]["reranker"]["local_model_directory"] = str(
            args.reranker_dir
        )


def retrieval_sanity(
    item: dict[str, Any], documents: list[Any], pipeline: Any, config: dict[str, Any]
) -> dict[str, Any]:
    retrieved_fact_ids: set[str] = set()
    leakage = 0
    for document in documents:
        retrieved_fact_ids.update(
            json.loads(document.metadata.get("fact_ids_json", "[]"))
        )
        if not pipeline.document_passes_metadata_gate(document, item, config):
            leakage += 1
    expected = set(item.get("evidence_fact_ids") or [])
    recall = len(expected.intersection(retrieved_fact_ids)) / len(expected) if expected else 1.0
    return {
        "evidence_fact_recall_at_k": round(recall, 6),
        "metadata_filter_leakage": leakage,
    }


def build_shared_stores(
    kb: dict[str, Any],
    base_config: dict[str, Any],
    variants: list[Variant],
    persist_dir: Path,
    embedding_device: str,
    pipeline: Any,
) -> tuple[dict[int, Any], Any, list[dict[str, Any]]]:
    from langchain_chroma import Chroma

    started = time.perf_counter()
    embeddings = pipeline.build_embeddings(
        base_config,
        embedding_device,
        profile_query_embeddings=True,
    )
    embedding_load_ms = round((time.perf_counter() - started) * 1000, 3)
    stores: dict[int, Any] = {}
    index_records: list[dict[str, Any]] = []
    item_count = 20
    for chunk_size in sorted({variant.chunk_size_tokens for variant in variants}):
        template = next(
            variant for variant in variants if variant.chunk_size_tokens == chunk_size
        )
        config = variant_week3_config(base_config, template, item_count)
        index_started = time.perf_counter()
        chunks = pipeline.split_documents(kb, config)
        chunk_dir = persist_dir / f"chunk-{chunk_size}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        store = Chroma(
            collection_name=pipeline.collection_name(kb, config),
            embedding_function=embeddings,
            persist_directory=str(chunk_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )
        store._ingen_profiled_embeddings = embeddings
        ids = [chunk.metadata["chunk_id"] for chunk in chunks]
        existing_ids = set((store.get(ids=ids).get("ids") or []))
        missing = [
            (chunk_id, chunk)
            for chunk_id, chunk in zip(ids, chunks, strict=True)
            if chunk_id not in existing_ids
        ]
        if missing:
            store.add_documents(
                documents=[chunk for _, chunk in missing],
                ids=[chunk_id for chunk_id, _ in missing],
            )
        stores[chunk_size] = store
        index_records.append(
            {
                "chunk_size_tokens": chunk_size,
                "chunks": len(chunks),
                "existing_chunks": len(existing_ids),
                "new_chunks_indexed": len(missing),
                "collection_name": pipeline.collection_name(kb, config),
                "index_and_open_ms": round(
                    (time.perf_counter() - index_started) * 1000, 3
                ),
            }
        )
    return stores, embeddings, [
        {"embedding_model_load_ms": embedding_load_ms}, *index_records
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml(args.config)
    protocol = validate_protocol(config, args.config)
    if PHASE_B not in [Path(value) for value in sys.path if value]:
        sys.path.insert(0, str(PHASE_B))
    import W03_RAG_Generation as generation
    import W03_RAG_Pipeline as pipeline

    kb_path = source_path(
        args.config, config["source_inputs"]["knowledge_base"]["path"]
    )
    eval_path = source_path(
        args.config, config["source_inputs"]["evaluation_set"]["path"]
    )
    week3_path = source_path(
        args.config, config["source_inputs"]["week3_run_config"]["path"]
    )
    kb, full_eval, base_config = pipeline.load_assets(
        kb_path, eval_path, week3_path
    )
    pipeline.validate_assets(kb, full_eval, base_config)
    eval_set = freeze_senpai_subset(full_eval, config)
    apply_model_directory_overrides(base_config, args)
    variants = randomized_variants(config)

    existing = read_jsonl(args.output)
    if existing and not args.resume:
        raise FileExistsError(f"{args.output} exists; use --resume")
    done = {str(row["run_item_id"]) for row in existing}
    if len(done) != len(existing):
        raise ValueError("existing output contains duplicate run_item_id values")

    args.persist_dir.mkdir(parents=True, exist_ok=True)
    stores, embeddings, index_records = build_shared_stores(
        kb,
        base_config,
        variants,
        args.persist_dir,
        args.embedding_device,
        pipeline,
    )
    del embeddings  # Stores retain the shared profiled adapter.

    ce_variant = next(variant for variant in variants if variant.reranking == "cross_encoder")
    ce_config = variant_week3_config(base_config, ce_variant, len(eval_set["items"]))
    reranker_started = time.perf_counter()
    reranker = pipeline.build_reranker(ce_config, args.embedding_device)
    reranker_load_ms = round((time.perf_counter() - reranker_started) * 1000, 3)

    generator = generation.LocalLlamaGenerator(
        model_dir=Path(base_config["generation"]["local_model_directory"]),
        model_id=base_config["generation"]["candidate_model_id"],
        model_revision=base_config["generation"]["candidate_model_revision"],
        tokenizer_revision=base_config["generation"]["tokenizer_revision"],
        seed=int(config["random_seed"]),
        max_input_tokens=int(base_config["generation"]["max_input_tokens"]),
    )
    runtime = generator.runtime_metadata()

    # One discarded warm-up covers BGE query embedding, Chroma, the cross-encoder,
    # prompt rendering, and Llama generation. It is never appended to evidence.
    warm_item = eval_set["items"][0]
    warm_documents, _ = pipeline.retrieve_item(
        warm_item,
        stores[ce_variant.chunk_size_tokens],
        ce_config,
        kb=kb,
        reranker=reranker,
    )
    warm_messages = pipeline.render_candidate_messages(
        warm_item,
        "rag",
        warm_documents,
        data_origin=kb["data_origin"],
        base_system_prompt=base_config["generation"].get("base_system_prompt"),
        rag_system_prompt=base_config["generation"].get("rag_system_prompt"),
    )
    warm_started = time.perf_counter()
    generator.generate(
        warm_messages,
        int(base_config["generation"]["decoding"]["max_new_tokens"]),
    )
    warmup_ms = round((time.perf_counter() - warm_started) * 1000, 3)

    manifest = {
        **protocol,
        "status": "running",
        "started_at_utc": utc_now(),
        "output": str(args.output),
        "persist_dir": str(args.persist_dir),
        "index_records": index_records,
        "reranker_load_ms": reranker_load_ms,
        "generator_runtime": runtime,
        "warmup": {
            "requests": 1,
            "discarded": True,
            "warmup_generation_wall_ms": warmup_ms,
            "eval_id_used": warm_item["eval_id"],
        },
        "rows_before_resume": len(existing),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    generated_this_run = 0
    global_position = 0
    for block_position, variant in enumerate(variants, start=1):
        variant_config = variant_week3_config(
            base_config, variant, len(eval_set["items"])
        )
        active_reranker = reranker if variant.reranking == "cross_encoder" else None
        for item_position, item in enumerate(eval_set["items"], start=1):
            global_position += 1
            run_item_id = f"W05::{variant.variant_id}::{item['eval_id']}"
            if run_item_id in done:
                continue
            if args.limit is not None and generated_this_run >= args.limit:
                break
            print(
                f"[{global_position}/360] {run_item_id}",
                flush=True,
            )
            profiler = StageProfiler()
            with profiler.stage("question_to_response_ms"):
                documents, retrieval_latency_ms = pipeline.retrieve_item(
                    item,
                    stores[variant.chunk_size_tokens],
                    variant_config,
                    kb=kb,
                    reranker=active_reranker,
                    profiler=profiler,
                )
                with profiler.stage("prompt_build_ms"):
                    trace = pipeline.retrieval_trace(documents)
                    messages = pipeline.render_candidate_messages(
                        item,
                        "rag",
                        documents,
                        data_origin=kb["data_origin"],
                        base_system_prompt=base_config["generation"].get(
                            "base_system_prompt"
                        ),
                        rag_system_prompt=base_config["generation"].get(
                            "rag_system_prompt"
                        ),
                    )
                with profiler.stage("generation_total_ms"):
                    generated = generator.generate(
                        messages,
                        int(
                            base_config["generation"]["decoding"][
                                "max_new_tokens"
                            ]
                        ),
                    )
            stages = profiler.rounded()
            sanity = retrieval_sanity(item, documents, pipeline, variant_config)
            row = {
                "run_item_id": run_item_id,
                "run_id": (
                    "w05-senpai-rag-optimisation-v1.0.0-seed42"
                ),
                "runner_version": RUNNER_VERSION,
                "experiment_id": config["experiment_id"],
                "experiment_version": config["experiment_version"],
                "variant_id": variant.variant_id,
                "variant_block_position": block_position,
                "item_position_within_block": item_position,
                "global_execution_position": global_position,
                "chunk_size_tokens": variant.chunk_size_tokens,
                "top_k": variant.top_k,
                "requested_reranking": variant.reranking,
                "effective_reranking": variant.reranking,
                "condition": "rag",
                "eval_id": item["eval_id"],
                "platform": item["platform"],
                "question": item["question"],
                "candidate_messages": messages,
                "candidate_messages_sha256": canonical_sha256(messages),
                "candidate_model_id": base_config["generation"][
                    "candidate_model_id"
                ],
                "candidate_model_revision": base_config["generation"][
                    "candidate_model_revision"
                ],
                "tokenizer_revision": base_config["generation"][
                    "tokenizer_revision"
                ],
                "prompt_version": base_config["generation"]["prompt_version"],
                "random_seed": int(config["random_seed"]),
                "knowledge_base_id": kb["knowledge_base_id"],
                "knowledge_base_version": kb["knowledge_base_version"],
                "evaluation_set_id": eval_set["evaluation_set_id"],
                "evaluation_set_version": eval_set["evaluation_set_version"],
                "source_evaluation_set_id": eval_set[
                    "source_evaluation_set_id"
                ],
                "source_evaluation_set_version": eval_set[
                    "source_evaluation_set_version"
                ],
                "retrieved_contexts": trace,
                "retrieval_latency_ms": round(retrieval_latency_ms, 3),
                "latency_ms": stages,
                **sanity,
                **generated,
                "runtime": runtime,
                "decoding": base_config["generation"]["decoding"],
                "status": "completed_warm_path",
                "completed_at_utc": utc_now(),
                "claim_boundary": config["claim_boundary"],
            }
            if not math.isfinite(float(stages["question_to_response_ms"])):
                raise ValueError("non-finite question_to_response_ms")
            append_jsonl(args.output, row)
            done.add(run_item_id)
            generated_this_run += 1
        if args.limit is not None and generated_this_run >= args.limit:
            break

    final_rows = read_jsonl(args.output)
    manifest.update(
        {
            "status": (
                "completed" if len(final_rows) == protocol["expected_rows"] else "partial"
            ),
            "completed_at_utc": utc_now(),
            "rows_generated_this_run": generated_this_run,
            "rows_in_output": len(final_rows),
            "output_sha256": sha256_file(args.output),
        }
    )
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validate", "run"), default="validate")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--persist-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--generator-dir", type=Path)
    parser.add_argument("--embedding-dir", type=Path)
    parser.add_argument("--reranker-dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if args.mode == "validate":
        print(json.dumps(validate_protocol(config, args.config), indent=2))
        return
    if args.persist_dir is None or args.output is None:
        raise ValueError("--persist-dir and --output are required in run mode")
    args.manifest = args.manifest or args.output.with_name("run_manifest.json")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

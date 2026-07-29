"""Run the registered chunk-size, top-k, and cross-encoder RAG ablation."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from W03_RAG_Pipeline import (
    build_reranker,
    ensure_external_persist_directory,
    evaluate_retrieval,
    index_documents,
    load_assets,
    sha256_file,
    validate_assets,
)


ABLATION_VERSION = "0.1.0"
CHUNK_SIZES = (256, 512, 1024)
TOP_K_VALUES = (1, 3, 5)
RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"
RERANKER_REVISION = "b5160aeac3c6c8fe7beaaaf04c9e0142826b58d1"


def registered_variants() -> list[dict[str, Any]]:
    return [
        {
            "chunk_size_tokens": chunk_size,
            "chunk_overlap_tokens": chunk_size // 8,
            "top_k": top_k,
            "reranker_enabled": reranker_enabled,
        }
        for chunk_size in CHUNK_SIZES
        for top_k in TOP_K_VALUES
        for reranker_enabled in (False, True)
    ]


def configure_variant(
    base_config: dict[str, Any],
    variant: dict[str, Any],
    reranker_model_dir: Path,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    splitter = config["retrieval"]["text_splitter"]
    splitter["chunk_size_tokens"] = variant["chunk_size_tokens"]
    splitter["chunk_overlap_tokens"] = variant["chunk_overlap_tokens"]
    retriever = config["retrieval"]["retriever"]
    retriever["top_k"] = variant["top_k"]
    retriever["fetch_k"] = 12
    config["retrieval"]["reranker"] = {
        "enabled": variant["reranker_enabled"],
        "integration": (
            "langchain_community.cross_encoders.HuggingFaceCrossEncoder"
        ),
        "model_id": RERANKER_MODEL_ID,
        "model_revision": RERANKER_REVISION,
        "local_model_directory": str(reranker_model_dir),
        "candidate_pool_size": 12,
    }
    return config


def summarize_variant(
    variant: dict[str, Any],
    retrieval_result: dict[str, Any],
    index_result: dict[str, Any],
) -> dict[str, Any]:
    rows = retrieval_result["rows"]
    returned_tokens = [
        sum(
            int(context.get("token_count") or 0)
            for context in row["retrieval_trace"]
        )
        for row in rows
    ]
    returned_units = [len(row["retrieval_trace"]) for row in rows]
    summary = retrieval_result["summary"]
    return {
        **variant,
        "collection_name": retrieval_result["collection_name"],
        "indexed_chunks": index_result["chunks"],
        "items": summary["items"],
        "document_recall_at_k": summary["mean_document_id_recall_at_k"],
        "evidence_fact_recall_at_k": summary[
            "mean_evidence_fact_recall_at_k"
        ],
        "hit_at_k": summary["hit_at_k"],
        "mrr": summary["mean_reciprocal_rank"],
        "metadata_leakage": summary["metadata_filter_leakage"],
        "mean_retrieval_latency_ms": summary["mean_retrieval_latency_ms"],
        "mean_returned_context_tokens": round(mean(returned_tokens), 3),
        "mean_returned_context_units": round(mean(returned_units), 3),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    rows = payload["variants"]
    lines = [
        "# Week 3 RAG Retrieval Ablation",
        "",
        f"- Ablation version: `{payload['ablation_version']}`",
        f"- Benchmark: `{payload['evaluation_set']}`",
        f"- Seed: `{payload['random_seed']}`",
        f"- Embedding: `{payload['embedding_model']}`",
        (
            f"- Reranker: `{RERANKER_MODEL_ID}` revision "
            f"`{RERANKER_REVISION}`"
        ),
        "",
        "| Chunk | top-k | Rerank | Chunks | Doc recall | Fact recall | MRR | Mean context tokens | Mean latency ms |",
        "|---:|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {chunk_size_tokens} | {top_k} | {rerank} | "
            "{indexed_chunks} | {document_recall_at_k:.4f} | "
            "{evidence_fact_recall_at_k:.4f} | {mrr:.4f} | "
            "{mean_returned_context_tokens:.1f} | "
            "{mean_retrieval_latency_ms:.3f} |".format(
                rerank="on" if row["reranker_enabled"] else "off",
                **row,
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "Perfect document recall alone is not treated as evidence of a robust "
            "retriever. The decision uses evidence-fact recall, rank, context "
            "budget, metadata leakage and latency. If all variants remain "
            "perfect, the registered conclusion is that this four-document "
            "corpus is too small and metadata-isolated to discriminate the "
            "retrieval choices; it is not that chunk size, top-k, or reranking "
            "never matter.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb", type=Path, required=True)
    parser.add_argument("--eval-set", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--persist-root", type=Path, required=True)
    parser.add_argument("--reranker-model-dir", type=Path, required=True)
    parser.add_argument(
        "--embedding-device", choices=["cpu", "cuda"], default="cuda"
    )
    parser.add_argument(
        "--reranker-device", choices=["cpu", "cuda"], default="cuda"
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for output in (args.output_json, args.output_report):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite immutable output: {output}")
    kb, eval_set, base_config = load_assets(
        args.kb, args.eval_set, args.config
    )
    validate_assets(kb, eval_set, base_config)
    persist_root = ensure_external_persist_directory(args.persist_root)
    variants: list[dict[str, Any]] = []
    reranker = None

    for chunk_size in CHUNK_SIZES:
        index_variant = {
            "chunk_size_tokens": chunk_size,
            "chunk_overlap_tokens": chunk_size // 8,
            "top_k": TOP_K_VALUES[0],
            "reranker_enabled": False,
        }
        index_config = configure_variant(
            base_config, index_variant, args.reranker_model_dir
        )
        chunk_persist = ensure_external_persist_directory(
            persist_root / f"chunk-{chunk_size}"
        )
        index_result, store = index_documents(
            kb=kb,
            config=index_config,
            persist_dir=chunk_persist,
            embedding_device=args.embedding_device,
        )
        for top_k in TOP_K_VALUES:
            for reranker_enabled in (False, True):
                variant = {
                    "chunk_size_tokens": chunk_size,
                    "chunk_overlap_tokens": chunk_size // 8,
                    "top_k": top_k,
                    "reranker_enabled": reranker_enabled,
                }
                config = configure_variant(
                    base_config, variant, args.reranker_model_dir
                )
                if reranker_enabled and reranker is None:
                    reranker = build_reranker(
                        config, args.reranker_device
                    )
                retrieval_result = evaluate_retrieval(
                    kb,
                    eval_set,
                    store,
                    config,
                    reranker=reranker if reranker_enabled else None,
                )
                variants.append(
                    summarize_variant(
                        variant, retrieval_result, index_result
                    )
                )

    payload = {
        "ablation_version": ABLATION_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "knowledge_base": (
            f"{kb['knowledge_base_id']}#{kb['knowledge_base_version']}"
        ),
        "evaluation_set": (
            f"{eval_set['evaluation_set_id']}#"
            f"{eval_set['evaluation_set_version']}"
        ),
        "random_seed": base_config["random_seed"],
        "embedding_model": (
            f"{base_config['retrieval']['embedding']['model_id']}@"
            f"{base_config['retrieval']['embedding']['model_revision']}"
        ),
        "reranker_model": f"{RERANKER_MODEL_ID}@{RERANKER_REVISION}",
        "asset_sha256": {
            "knowledge_base": sha256_file(args.kb),
            "evaluation_set": sha256_file(args.eval_set),
            "base_config": sha256_file(args.config),
        },
        "variants": variants,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_report.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"status": "ok", "variants": len(variants)}, indent=2))


if __name__ == "__main__":
    main()

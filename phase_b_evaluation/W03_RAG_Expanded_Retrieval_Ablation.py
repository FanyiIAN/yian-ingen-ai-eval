"""Ablate retrieval breadth and reranking on the expanded Week 3 corpus."""

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


ABLATION_VERSION = "0.2.0"
TOP_K_VALUES = (4, 6, 8, 10, 12)
FETCH_K = 32


def percentile(values: list[float], proportion: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * proportion
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def configure_variant(
    base_config: dict[str, Any], top_k: int, reranker_enabled: bool
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    retriever = config["retrieval"]["retriever"]
    retriever["fetch_k"] = FETCH_K
    retriever["top_k"] = top_k
    config["retrieval"]["reranker"]["enabled"] = reranker_enabled
    config["retrieval"]["reranker"]["candidate_pool_size"] = FETCH_K
    return config


def summarize(
    result: dict[str, Any], top_k: int, reranker_enabled: bool
) -> dict[str, Any]:
    rows = result["rows"]
    latencies = [float(row["retrieval_latency_ms"]) for row in rows]
    context_units = [len(row["retrieval_trace"]) for row in rows]
    context_tokens = [
        sum(int(chunk.get("token_count") or 0) for chunk in row["retrieval_trace"])
        for row in rows
    ]
    fact_recalls = [float(row["evidence_fact_recall_at_k"]) for row in rows]
    return {
        "top_k": top_k,
        "fetch_k": FETCH_K,
        "reranker_enabled": reranker_enabled,
        "items": len(rows),
        "document_recall_at_k": result["summary"][
            "mean_document_id_recall_at_k"
        ],
        "evidence_fact_recall_at_k": result["summary"][
            "mean_evidence_fact_recall_at_k"
        ],
        "full_evidence_items": sum(value == 1.0 for value in fact_recalls),
        "hit_at_k": result["summary"]["hit_at_k"],
        "mrr": result["summary"]["mean_reciprocal_rank"],
        "metadata_filter_leakage": result["summary"][
            "metadata_filter_leakage"
        ],
        "latency_ms": {
            "mean": round(mean(latencies), 3),
            "p50": round(percentile(latencies, 0.50), 3),
            "p90": round(percentile(latencies, 0.90), 3),
            "p95": round(percentile(latencies, 0.95), 3),
            "max": round(max(latencies), 3),
        },
        "mean_returned_context_units": round(mean(context_units), 3),
        "mean_returned_context_tokens": round(mean(context_tokens), 3),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Week 3 Expanded-Corpus Retrieval Ablation",
        "",
        f"- Version: `{payload['ablation_version']}`",
        f"- Corpus: `{payload['knowledge_base']}`",
        f"- Benchmark: `{payload['evaluation_set']}`",
        f"- Seed: `{payload['random_seed']}`",
        f"- Indexed chunks: `{payload['index']['chunks']}`",
        "",
        (
            "| top-k | Reranker | Fact recall | Full evidence | MRR | "
            "Mean units | Mean tokens | Mean ms | p95 ms |"
        ),
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["variants"]:
        lines.append(
            "| {top_k} | {rerank} | {evidence_fact_recall_at_k:.4f} | "
            "{full_evidence_items}/{items} | {mrr:.4f} | "
            "{mean_returned_context_units:.1f} | "
            "{mean_returned_context_tokens:.1f} | {mean_ms:.1f} | "
            "{p95_ms:.1f} |".format(
                rerank="on" if row["reranker_enabled"] else "off",
                mean_ms=row["latency_ms"]["mean"],
                p95_ms=row["latency_ms"]["p95"],
                **row,
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Document hit rate is reported but is not the selection criterion: all "
            "questions are product-filtered, so evidence-fact recall, complete-evidence "
            "items, context budget, and latency are more discriminating. Chunk-size "
            "variants are not repeated here because the curated atomic units already "
            "produce one child chunk each at the registered 256-token setting.",
            "Latency is descriptive rather than a randomized causal comparison: "
            "variants ran in fixed order inside one process, so later rows benefit "
            "from warmer model and filesystem caches. Parameter selection therefore "
            "uses evidence recall and context budget first.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb", type=Path, required=True)
    parser.add_argument("--eval-set", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--persist-dir", type=Path, required=True)
    parser.add_argument("--embedding-device", default="cuda")
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
    persist_dir = ensure_external_persist_directory(args.persist_dir)
    index_result, store = index_documents(
        kb, base_config, persist_dir, args.embedding_device
    )
    reranker_config = configure_variant(base_config, TOP_K_VALUES[0], True)
    reranker = build_reranker(reranker_config, args.embedding_device)

    variants: list[dict[str, Any]] = []
    for top_k in TOP_K_VALUES:
        for reranker_enabled in (False, True):
            config = configure_variant(base_config, top_k, reranker_enabled)
            result = evaluate_retrieval(
                kb,
                eval_set,
                store,
                config,
                reranker=reranker if reranker_enabled else None,
            )
            variants.append(summarize(result, top_k, reranker_enabled))

    payload = {
        "ablation_version": ABLATION_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "knowledge_base": (
            f"{kb['knowledge_base_id']}#{kb['knowledge_base_version']}"
        ),
        "evaluation_set": (
            f"{eval_set['evaluation_set_id']}#{eval_set['evaluation_set_version']}"
        ),
        "random_seed": int(base_config["random_seed"]),
        "asset_sha256": {
            "knowledge_base": sha256_file(args.kb),
            "evaluation_set": sha256_file(args.eval_set),
            "base_config": sha256_file(args.config),
        },
        "index": index_result,
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

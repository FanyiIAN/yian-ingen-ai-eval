"""Fail-closed preflight proving that long-source chunk size is operational."""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
from pathlib import Path
from typing import Any

import W03_RAG_Pipeline as pipeline


def quantile(values: list[int], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def run(kb_path: Path, config_path: Path) -> dict[str, Any]:
    kb = pipeline.load_yaml(kb_path)
    base = pipeline.load_yaml(config_path)
    expected_facts = {
        fact["fact_id"]
        for document in kb["documents"]
        for fact in document.get("supported_facts") or []
    }
    records: list[dict[str, Any]] = []
    hashes: dict[int, set[str]] = {}
    for size in (256, 512, 1024):
        config = copy.deepcopy(base)
        config["retrieval"]["text_splitter"]["chunk_size_tokens"] = size
        chunks = pipeline.split_documents(kb, config)
        token_counts = [int(chunk.metadata["token_count"]) for chunk in chunks]
        chunk_hashes = {chunk.metadata["chunk_content_sha256"] for chunk in chunks}
        hashes[size] = chunk_hashes
        observed_facts = {
            fact_id
            for chunk in chunks
            for fact_id in json.loads(chunk.metadata.get("fact_ids_json", "[]"))
        }
        records.append({
            "chunk_size_tokens": size,
            "chunks": len(chunks),
            "unique_chunk_hashes": len(chunk_hashes),
            "documents_with_chunks": len({chunk.metadata["document_id"] for chunk in chunks}),
            "token_count": {
                "min": min(token_counts),
                "median": statistics.median(token_counts),
                "p95": round(quantile(token_counts, 0.95), 3),
                "max": max(token_counts),
            },
            "expected_facts": len(expected_facts),
            "facts_attached_to_at_least_one_chunk": len(observed_facts & expected_facts),
            "missing_fact_ids": sorted(expected_facts - observed_facts),
        })
    pairwise = []
    for left, right in ((256, 512), (256, 1024), (512, 1024)):
        union = hashes[left] | hashes[right]
        intersection = hashes[left] & hashes[right]
        pairwise.append({
            "left": left,
            "right": right,
            "identical_hash_sets": hashes[left] == hashes[right],
            "hash_jaccard": round(len(intersection) / len(union), 6) if union else math.nan,
        })
    errors: list[str] = []
    counts = [row["chunks"] for row in records]
    if len(set(counts)) != 3:
        errors.append(f"chunk counts are not all different: {counts}")
    if not counts[0] > counts[1] > counts[2]:
        errors.append(f"chunk counts are not strictly decreasing: {counts}")
    if any(row["missing_fact_ids"] for row in records):
        errors.append("one or more evidence facts do not overlap any chunk")
    if any(row["identical_hash_sets"] for row in pairwise):
        errors.append("at least one pair of chunk variants has identical hash sets")
    if int(base["retrieval"]["retriever"].get("auto_merge_min_children", -1)) != 0:
        errors.append("long-source config must disable parent auto-merge")
    result = {
        "status": "failed" if errors else "passed",
        "knowledge_base": str(kb_path),
        "config": str(config_path),
        "parent_documents": len(kb["documents"]),
        "parent_policy": kb.get("parent_document_policy"),
        "variants": records,
        "pairwise_hash_diagnostics": pairwise,
        "errors": errors,
    }
    if errors:
        raise ValueError(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.kb, args.config)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

"""Summarize the final expanded Week 3 retrieval trace used by all models."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml


ANALYZER_VERSION = "0.1.0"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a mapping")
    return value


def percentile(values: list[float], proportion: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * proportion
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def metric_summary(values: Iterable[float]) -> dict[str, float | int]:
    numbers = [float(value) for value in values]
    return {
        "count": len(numbers),
        "mean": round(statistics.fmean(numbers), 6),
        "p50": round(percentile(numbers, 0.50), 6),
        "p90": round(percentile(numbers, 0.90), 6),
        "p95": round(percentile(numbers, 0.95), 6),
        "max": round(max(numbers), 6),
    }


def analyze(
    rows: list[dict[str, Any]], eval_set: dict[str, Any]
) -> dict[str, Any]:
    rag_rows = [row for row in rows if row.get("condition") == "rag"]
    item_by_id = {item["eval_id"]: item for item in eval_set["items"]}
    if len(rag_rows) != len(item_by_id):
        raise ValueError(
            f"Expected {len(item_by_id)} RAG rows, found {len(rag_rows)}"
        )
    details: list[dict[str, Any]] = []
    for row in rag_rows:
        item = item_by_id[row["eval_id"]]
        contexts = row.get("retrieved_contexts") or []
        retrieved_facts: set[str] = set()
        metadata_leakage = 0
        for context in contexts:
            retrieved_facts.update(
                json.loads(context.get("fact_ids_json", "[]"))
            )
            if (
                context.get("owner_type") != "official"
                or context.get("access_scope") != "public"
                or context.get("confidentiality") != "public"
                or context.get("source_domain") != "www.ingendynamics.com"
                or context.get("document_id")
                not in set(item["reference_document_ids"])
            ):
                metadata_leakage += 1
        expected = set(item["evidence_fact_ids"])
        missing = sorted(expected - retrieved_facts)
        details.append(
            {
                "eval_id": row["eval_id"],
                "platform": row["platform"],
                "difficulty": item["difficulty"],
                "answerability": item["answerability"],
                "evidence_fact_count": len(expected),
                "evidence_fact_recall_at_k": round(
                    len(expected & retrieved_facts) / len(expected), 6
                ),
                "full_evidence": not missing,
                "missing_evidence_fact_ids": missing,
                "retrieved_units": len(contexts),
                "retrieved_tokens": sum(
                    int(context.get("token_count") or 0)
                    for context in contexts
                ),
                "retrieval_latency_ms": float(row["retrieval_latency_ms"]),
                "metadata_filter_leakage": metadata_leakage,
            }
        )

    def group(field: str) -> dict[str, Any]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for detail in details:
            grouped[str(detail[field])].append(detail)
        return {
            key: {
                "items": len(values),
                "mean_evidence_fact_recall_at_k": round(
                    statistics.fmean(
                        value["evidence_fact_recall_at_k"]
                        for value in values
                    ),
                    6,
                ),
                "full_evidence_items": sum(
                    value["full_evidence"] for value in values
                ),
            }
            for key, values in sorted(grouped.items())
        }

    return {
        "analyzer_version": ANALYZER_VERSION,
        "items": len(details),
        "top_k": max(detail["retrieved_units"] for detail in details),
        "mean_evidence_fact_recall_at_k": round(
            statistics.fmean(
                detail["evidence_fact_recall_at_k"] for detail in details
            ),
            6,
        ),
        "full_evidence_items": sum(
            detail["full_evidence"] for detail in details
        ),
        "metadata_filter_leakage": sum(
            detail["metadata_filter_leakage"] for detail in details
        ),
        "retrieval_latency_ms": metric_summary(
            detail["retrieval_latency_ms"] for detail in details
        ),
        "retrieved_context_units": metric_summary(
            detail["retrieved_units"] for detail in details
        ),
        "retrieved_context_tokens": metric_summary(
            detail["retrieved_tokens"] for detail in details
        ),
        "by_platform": group("platform"),
        "by_difficulty": group("difficulty"),
        "by_answerability": group("answerability"),
        "incomplete_evidence_rows": [
            detail for detail in details if not detail["full_evidence"]
        ],
    }


def render_markdown(summary: dict[str, Any]) -> str:
    latency = summary["retrieval_latency_ms"]
    tokens = summary["retrieved_context_tokens"]
    lines = [
        "# Week 3 Expanded Final Retrieval Analysis",
        "",
        f"- Analyzer: `{summary['analyzer_version']}`",
        f"- Questions: `{summary['items']}`",
        f"- Final top-k: `{summary['top_k']}`",
        (
            "- Mean evidence-fact recall: "
            f"`{summary['mean_evidence_fact_recall_at_k']:.4f}`"
        ),
        f"- Full-evidence questions: `{summary['full_evidence_items']}/{summary['items']}`",
        f"- Metadata leakage: `{summary['metadata_filter_leakage']}`",
        (
            "- Retrieval latency: mean "
            f"`{latency['mean']:.1f} ms`, p95 `{latency['p95']:.1f} ms`"
        ),
        (
            "- Returned context: mean "
            f"`{tokens['mean']:.1f}` tokens, p95 `{tokens['p95']:.1f}`"
        ),
        "",
        "## Grouped evidence recall",
        "",
        "| Group | Items | Mean fact recall | Full evidence |",
        "|---|---:|---:|---:|",
    ]
    for category in ("by_platform", "by_difficulty", "by_answerability"):
        for key, value in summary[category].items():
            lines.append(
                f"| {category.removeprefix('by_')}={key} | {value['items']} | "
                f"{value['mean_evidence_fact_recall_at_k']:.4f} | "
                f"{value['full_evidence_items']}/{value['items']} |"
            )
    lines.extend(
        [
            "",
            "## Incomplete evidence rows",
            "",
            "| Eval ID | Recall | Missing fact IDs |",
            "|---|---:|---|",
        ]
    )
    for row in summary["incomplete_evidence_rows"]:
        lines.append(
            f"| {row['eval_id']} | {row['evidence_fact_recall_at_k']:.4f} | "
            f"{', '.join(row['missing_evidence_fact_ids'])} |"
        )
    lines.extend(
        [
            "",
            "The timing distribution is the actual one-pass input-build trace. It "
            "contains the first-query warm-up effect and is therefore not a pure "
            "steady-state serving benchmark.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--eval-set", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(read_jsonl(args.inputs), load_yaml(args.eval_set))
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_report.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

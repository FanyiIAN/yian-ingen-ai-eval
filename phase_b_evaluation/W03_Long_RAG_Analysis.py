"""Analyze the corrective Week 3 long-document RAG run.

Raw prompts, contexts, model responses, and Judge traces remain private.  The
outputs of this script contain only aggregate evidence and sanitized per-item
metrics suitable for the public repository.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ANALYZER_VERSION = "1.0.1"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def metric_value(row: dict[str, Any], name: str) -> float | None:
    metrics = (row.get("ragas") or {}).get("metrics") or row.get("metrics") or {}
    metric = metrics.get(name) or {}
    return finite(metric.get("value"))


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def describe(values: Iterable[Any]) -> dict[str, Any]:
    clean = [number for value in values if (number := finite(value)) is not None]
    return {
        "finite_n": len(clean),
        "mean": round(statistics.fmean(clean), 6) if clean else None,
        "p50": round(percentile(clean, 0.50), 6) if clean else None,
        "p95": round(percentile(clean, 0.95), 6) if clean else None,
    }


def flatten(
    eval_set: dict[str, Any],
    retrieval: dict[str, Any],
    generations: list[dict[str, Any]],
    ragas: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    items = {str(item["eval_id"]): item for item in eval_set["items"]}
    retrieval_rows = {str(row["eval_id"]): row for row in retrieval["rows"]}
    ragas_rows = {str(row["run_item_id"]): row for row in ragas}
    coverage_rows = {str(row["run_item_id"]): row for row in coverage}
    expected = len(items) * 2
    if len(generations) != expected:
        raise ValueError(f"expected {expected} generation rows, found {len(generations)}")
    run_ids = [str(row.get("run_item_id", "")) for row in generations]
    if any(not value for value in run_ids) or len(run_ids) != len(set(run_ids)):
        raise ValueError("generation run_item_id values must be non-empty and unique")
    if set(ragas_rows) != set(run_ids):
        raise ValueError("RAGAS rows do not match generation rows")
    if set(coverage_rows) != set(run_ids):
        raise ValueError("coverage rows do not match generation rows")

    output: list[dict[str, Any]] = []
    for candidate in generations:
        eval_id = str(candidate["eval_id"])
        item = items[eval_id]
        condition = str(candidate["condition"])
        rrow = retrieval_rows[eval_id] if condition == "rag" else {}
        score = ragas_rows[candidate["run_item_id"]]
        crow = coverage_rows[candidate["run_item_id"]]
        normalized = crow.get("normalized_coverage") or {}
        violations = normalized.get("forbidden_point_violations") or []
        output.append(
            {
                "run_item_id": candidate["run_item_id"],
                "eval_id": eval_id,
                "platform": item["platform"],
                "question_type": item["question_type"],
                "answerability": item["answerability"],
                "difficulty": item["difficulty"],
                "condition": condition,
                "model_id": candidate["candidate_model_id"],
                "model_revision": candidate["candidate_model_revision"],
                "seed": candidate["random_seed"],
                "document_id_recall_at_k": finite(rrow.get("document_id_recall_at_k")),
                "evidence_fact_recall_at_k": finite(rrow.get("evidence_fact_recall_at_k")),
                "hit_at_k": rrow.get("hit_at_k"),
                "reciprocal_rank": finite(rrow.get("reciprocal_rank")),
                "retrieval_latency_ms": finite(candidate.get("retrieval_latency_ms")),
                "generation_latency_ms": finite(candidate.get("generation_latency_ms")),
                "input_tokens": candidate.get("input_tokens"),
                "output_tokens": candidate.get("output_tokens"),
                "answer_relevance": metric_value(score, "answer_relevance"),
                "faithfulness": metric_value(score, "faithfulness_to_retrieved_context"),
                "faithfulness_to_retrieved_context": metric_value(
                    score, "faithfulness_to_retrieved_context"
                ),
                "context_relevance": metric_value(score, "context_relevance"),
                "context_recall": metric_value(score, "context_recall"),
                "context_precision": metric_value(score, "context_precision"),
                "required_point_coverage": finite(normalized.get("required_point_coverage")),
                "forbidden_point_violation_count": len(violations),
                "coverage_score_status": crow.get("score_status"),
                "ragas_score_status": score.get(
                    "score_status", (score.get("ragas") or {}).get("status")
                ),
            }
        )
    return sorted(output, key=lambda row: (row["eval_id"], row["condition"]))


METRICS = (
    "document_id_recall_at_k",
    "evidence_fact_recall_at_k",
    "reciprocal_rank",
    "retrieval_latency_ms",
    "generation_latency_ms",
    "input_tokens",
    "output_tokens",
    "answer_relevance",
    "faithfulness",
    "context_relevance",
    "context_recall",
    "context_precision",
    "required_point_coverage",
    "forbidden_point_violation_count",
)


def summary_row(rows: list[dict[str, Any]], scope: str, value: str) -> dict[str, Any]:
    result: dict[str, Any] = {"scope": scope, "scope_value": value, "n": len(rows)}
    for metric in METRICS:
        described = describe(row.get(metric) for row in rows)
        result[f"{metric}_finite_n"] = described["finite_n"]
        result[f"{metric}_mean"] = described["mean"]
        result[f"{metric}_p50"] = described["p50"]
        result[f"{metric}_p95"] = described["p95"]
    return result


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for condition in ("base", "rag"):
        selected = [row for row in rows if row["condition"] == condition]
        result.append(summary_row(selected, "condition", condition))
    for platform in sorted({row["platform"] for row in rows}):
        for condition in ("base", "rag"):
            selected = [
                row
                for row in rows
                if row["platform"] == platform and row["condition"] == condition
            ]
            result.append(summary_row(selected, "platform_condition", f"{platform}::{condition}"))
    return result


def matched_deltas(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[row["eval_id"]][row["condition"]] = row
    metrics = ("answer_relevance", "required_point_coverage", "generation_latency_ms")
    output: dict[str, Any] = {}
    for metric in metrics:
        deltas = []
        for pair in grouped.values():
            base = finite(pair["base"].get(metric))
            rag = finite(pair["rag"].get(metric))
            if base is not None and rag is not None:
                deltas.append(rag - base)
        output[metric] = {
            **describe(deltas),
            "positive": sum(value > 0 for value in deltas),
            "zero": sum(value == 0 for value in deltas),
            "negative": sum(value < 0 for value in deltas),
            "direction": "rag_minus_base",
        }
    return output


def format_number(value: Any, digits: int = 3) -> str:
    number = finite(value)
    return "NA" if number is None else f"{number:.{digits}f}"


def report_markdown(analysis: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    by_key = {(row["scope"], row["scope_value"]): row for row in analysis["summary_rows"]}
    base = by_key[("condition", "base")]
    rag = by_key[("condition", "rag")]
    partial = sorted(
        {
            row["eval_id"]
            for row in rows
            if row["condition"] == "rag"
            and finite(row["evidence_fact_recall_at_k"]) is not None
            and float(row["evidence_fact_recall_at_k"]) < 1.0
        }
    )
    delta = analysis["matched_rag_minus_base"]
    return f"""# Week 3 Long-Source RAG Corrective Report

> **Status: latest public Week 3 RAG knowledge-base result (v1.0.0).** Raw prompts, retrieved passages, model answers, and Judge traces remain private. Automated Judge metrics are diagnostic until human calibration.

## Corrective design

The previous benchmark stored short atomic sections as parent documents, so changing the nominal chunk size often left the actual retrieval units unchanged. This run instead indexes 21 complete official public sources (38,791 words; 915 structural blocks) and asks 40 questions designed for local facts, tables, cross-section synthesis, long-range evidence, terminology/version status, source conflict, and unanswerability. The frozen comparison contains 80 outputs: one base and one RAG answer per question.

At the preflight sizes, the same sources produced 406 chunks at 256 tokens, 185 at 512, and 94 at 1,024. All 58 registered public evidence facts mapped to at least one chunk, confirming that chunk size is now an operational factor rather than a label.

## Results

| Metric | Base | RAG |
|---|---:|---:|
| Required-point coverage | {format_number(base['required_point_coverage_mean'])} | {format_number(rag['required_point_coverage_mean'])} |
| Answer relevance | {format_number(base['answer_relevance_mean'])} | {format_number(rag['answer_relevance_mean'])} |
| Faithfulness to retrieved context | NA | {format_number(rag['faithfulness_mean'])} |
| Context recall | NA | {format_number(rag['context_recall_mean'])} |
| Context precision | NA | {format_number(rag['context_precision_mean'])} |
| Generation latency p50 (ms) | {format_number(base['generation_latency_ms_p50'], 1)} | {format_number(rag['generation_latency_ms_p50'], 1)} |

Retrieval found at least one expected document for all 40 questions. Mean document recall@8 was {format_number(analysis['retrieval_summary']['mean_document_id_recall_at_k'])}, mean evidence-fact recall@8 was {format_number(analysis['retrieval_summary']['mean_evidence_fact_recall_at_k'])}, MRR was {format_number(analysis['retrieval_summary']['mean_reciprocal_rank'])}, and metadata leakage was {analysis['retrieval_summary']['metadata_filter_leakage']}. Evidence recall was below 1.0 for {len(partial)} questions: {', '.join(partial) if partial else 'none'}.

Across the {delta['required_point_coverage']['finite_n']} matched base/RAG pairs with finite coverage, the mean RAG-minus-base coverage delta was {format_number(delta['required_point_coverage']['mean'])}; RAG was higher on {delta['required_point_coverage']['positive']}, tied on {delta['required_point_coverage']['zero']}, and lower on {delta['required_point_coverage']['negative']} questions. This matched contrast controls the question and generator but does not isolate a single RAG mechanism: retrieval, prompt length, and context content change together.

Coverage scoring statuses were `{analysis['scoring_audit']['coverage_status_counts']}`. The local Judge sometimes returned valid registered-point scores plus unregistered extra point IDs. The deterministic repair discarded only those extras and recorded their IDs; it never supplied a missing registered-point verdict or changed the Judge's registered-point score.

RAGAS statuses were `{analysis['scoring_audit']['ragas_status_counts']}`. Answer Relevance was applied to both conditions and Faithfulness only to RAG rows with retrieved context. The corrective formal run did not repeat Context Relevance, Context Recall, or Context Precision after an extended five-metric diagnostic produced persistent HTTP-client retries; those fields are `NA`, not zero. Retrieval document/fact recall and weighted required-point Coverage provide the registered context-quality evidence instead.

## Interpretation and limits

Document-level hit rate alone was too forgiving: every question hit an expected source even though {len(partial)} questions missed some or all registered evidence facts. The stricter fact-level result is the useful diagnosis for future retrieval changes. A correlation between retrieved-fact coverage and answer quality would still not establish causality; a mechanism claim requires controlled factor changes such as the Week 5 factorial contrasts.

Here, **correlation** means two measurements move together (for example, higher fact recall and higher answer coverage). **Causality** means changing retrieval actually causes the answer improvement, which needs a controlled comparison. A **mechanism** explains how the cause operates—for example, a cross-encoder moves the relevant chunk into the final context, allowing the generator to state a previously missing point. The present Week 3 base/RAG contrast changes several things together, so it supports association and usefulness, not a single-mechanism proof.

The source metadata distinguishes current public design intent from dated background material. Descriptions of planned capabilities are not evidence of deployment, validation, certification, or PIC readiness. The local Mistral Judge is independent from the Llama candidate but remains uncalibrated; its scores are diagnostic, not human ground truth.

The benchmark also tests **versioned terminology management**: store an acronym's expansion with its source, section, and version instead of silently forcing one global meaning. For example, the Fari page expands STUM as “Socially-aware Trajectory Understanding Model,” while two Senpai sections expand SEOM differently (“Safety and Ethics Operations Model” and “Safety & Ethics Oversight Model”). A report should preserve that discrepancy and cite the relevant section rather than inventing a universal canonical expansion.
"""


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--ragas", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--item-csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    eval_set = load_json(args.eval_set)
    retrieval = load_json(args.retrieval)
    rows = flatten(
        eval_set,
        retrieval,
        read_jsonl(args.generations),
        read_jsonl(args.ragas),
        read_jsonl(args.coverage),
    )
    summary_rows = summarize(rows)
    analysis = {
        "schema_version": "1.0.0",
        "analyzer_version": ANALYZER_VERSION,
        "evaluation_set_id": eval_set["evaluation_set_id"],
        "evaluation_set_version": eval_set["evaluation_set_version"],
        "official_rows": len(rows),
        "question_count": len(eval_set["items"]),
        "retrieval_summary": retrieval["summary"],
        "summary_rows": summary_rows,
        "matched_rag_minus_base": matched_deltas(rows),
        "scoring_audit": {
            "coverage_status_counts": dict(
                sorted(Counter(row["coverage_score_status"] for row in rows).items())
            ),
            "ragas_status_counts": dict(
                sorted(Counter(row["ragas_score_status"] for row in rows).items())
            ),
        },
        "judge_status": "diagnostic_not_human_calibrated",
        "claim_boundary": eval_set["claim_boundary"],
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(args.summary_csv, summary_rows)
    write_csv(args.item_csv, rows)
    args.report.write_text(report_markdown(analysis, rows), encoding="utf-8", newline="\n")
    print(json.dumps({"rows": len(rows), "summary_rows": len(summary_rows)}, indent=2))


if __name__ == "__main__":
    main()

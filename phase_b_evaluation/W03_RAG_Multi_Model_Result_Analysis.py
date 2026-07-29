"""Audit and summarize the frozen Week 3 three-model Base/RAG run.

This reporting layer does not call a model and does not edit raw outputs.  It
checks pairing and shared-context invariants, computes deterministic runtime and
citation diagnostics, and optionally summarizes provisional RAGAS rows.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


ANALYZER_VERSION = "0.1.2"
CITATION_RE = re.compile(r"\[([A-Za-z0-9_.:-]+)\]")
CONDITIONS = ("base", "rag")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def parse_keyed_paths(values: Iterable[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected MODEL_KEY=PATH, received: {value}")
        key, raw_path = value.split("=", 1)
        key = key.strip()
        if not key or key in result:
            raise ValueError(f"Missing or duplicate model key: {key!r}")
        result[key] = Path(raw_path).resolve()
    if not result:
        raise ValueError("At least one keyed path is required")
    return result


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def mean_or_none(values: Iterable[Any]) -> float | None:
    finite = [number for value in values if (number := finite_number(value)) is not None]
    return round(statistics.fmean(finite), 6) if finite else None


def normalized_text(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", value.lower()).split())


def is_question_echo(output: str, question: str) -> bool:
    normalized_output = normalized_text(output)
    normalized_question = normalized_text(question)
    if not normalized_output:
        return False
    if normalized_output in normalized_question and len(normalized_output.split()) >= 5:
        return True
    return SequenceMatcher(None, normalized_output, normalized_question).ratio() >= 0.82


def validate_generation_rows(
    model_key: str,
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    by_eval: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    run_ids: set[str] = set()
    for row in rows:
        run_id = row.get("run_item_id")
        eval_id = row.get("eval_id")
        condition = row.get("condition")
        if not isinstance(run_id, str) or run_id in run_ids:
            raise ValueError(f"{model_key}: duplicate or missing run_item_id: {run_id}")
        if not isinstance(eval_id, str) or condition not in CONDITIONS:
            raise ValueError(f"{model_key}: malformed evaluation identity in {run_id}")
        if condition in by_eval[eval_id]:
            raise ValueError(f"{model_key}: duplicate {eval_id}/{condition}")
        run_ids.add(run_id)
        by_eval[eval_id][condition] = row
    incomplete = {
        eval_id: sorted(set(CONDITIONS) - set(conditions))
        for eval_id, conditions in by_eval.items()
        if set(conditions) != set(CONDITIONS)
    }
    if incomplete:
        raise ValueError(f"{model_key}: incomplete Base/RAG pairs: {incomplete}")
    return dict(by_eval)


def row_citation_diagnostic(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("condition") != "rag":
        return {
            "references": 0,
            "valid": 0,
            "eligible_chunk_mentions": 0,
            "format_compliant": None,
            "invalid_ids": [],
        }
    output = str(row.get("candidate_output", ""))
    references = CITATION_RE.findall(output)
    eligible = {
        str(context.get("chunk_id"))
        for context in row.get("retrieved_contexts", [])
        if isinstance(context, dict) and context.get("chunk_id")
    }
    invalid = sorted({citation for citation in references if citation not in eligible})
    eligible_mentions = sum(output.count(chunk_id) for chunk_id in eligible)
    valid = sum(citation in eligible for citation in references)
    return {
        "references": len(references),
        "valid": valid,
        "eligible_chunk_mentions": eligible_mentions,
        "format_compliant": (
            eligible_mentions > 0
            and valid == eligible_mentions
            and not invalid
        ),
        "invalid_ids": invalid,
    }


def summarize_subset(
    rows: list[dict[str, Any]],
    excluded_eval_ids: set[str],
) -> dict[str, Any]:
    included = [row for row in rows if row["eval_id"] not in excluded_eval_ids]
    result: dict[str, Any] = {
        "rows": len(included),
        "evaluation_items": len({row["eval_id"] for row in included}),
        "conditions": {},
    }
    for condition in CONDITIONS:
        selected = [row for row in included if row["condition"] == condition]
        citation_diagnostics = [row_citation_diagnostic(row) for row in selected]
        references = sum(item["references"] for item in citation_diagnostics)
        valid = sum(item["valid"] for item in citation_diagnostics)
        eligible_mentions = sum(
            item["eligible_chunk_mentions"] for item in citation_diagnostics
        )
        invalid_ids = sorted(
            {
                invalid_id
                for item in citation_diagnostics
                for invalid_id in item["invalid_ids"]
            }
        )
        result["conditions"][condition] = {
            "rows": len(selected),
            "empty_outputs": sum(
                not str(row.get("candidate_output", "")).strip() for row in selected
            ),
            "question_echoes": sum(
                is_question_echo(
                    str(row.get("candidate_output", "")),
                    str(row.get("question", "")),
                )
                for row in selected
            ),
            "mean_input_tokens": mean_or_none(
                row.get("input_tokens") for row in selected
            ),
            "mean_output_tokens": mean_or_none(
                row.get("output_tokens") for row in selected
            ),
            "mean_generation_latency_ms": mean_or_none(
                row.get("generation_latency_ms") for row in selected
            ),
            "mean_retrieval_latency_ms": mean_or_none(
                row.get("retrieval_latency_ms") for row in selected
            ),
            "citation_references": references,
            "citation_precision": round(valid / references, 6) if references else None,
            "rows_with_citation": sum(item["references"] > 0 for item in citation_diagnostics),
            "eligible_chunk_mentions": eligible_mentions,
            "rows_with_eligible_chunk_mention": sum(
                item["eligible_chunk_mentions"] > 0
                for item in citation_diagnostics
            ),
            "citation_format_compliant_rows": sum(
                item["format_compliant"] is True
                for item in citation_diagnostics
            ),
            "invalid_citation_ids": invalid_ids,
        }
    by_eval = validate_generation_rows("subset", included)
    result["paired_output_changed"] = sum(
        pair["base"].get("candidate_output_sha256")
        != pair["rag"].get("candidate_output_sha256")
        for pair in by_eval.values()
    )
    result["paired_output_changed_rate"] = round(
        result["paired_output_changed"] / len(by_eval), 6
    ) if by_eval else None
    return result


def shared_input_audit(
    paired_by_model: dict[str, dict[str, dict[str, dict[str, Any]]]]
) -> dict[str, Any]:
    keys = list(paired_by_model)
    baseline = paired_by_model[keys[0]]
    mismatches: list[str] = []
    for other_key in keys[1:]:
        other = paired_by_model[other_key]
        if set(other) != set(baseline):
            mismatches.append(f"{other_key}:eval_id_set")
            continue
        for eval_id in sorted(baseline):
            for condition in CONDITIONS:
                left = baseline[eval_id][condition]
                right = other[eval_id][condition]
                if left.get("candidate_messages_sha256") != right.get(
                    "candidate_messages_sha256"
                ):
                    mismatches.append(
                        f"{other_key}:{eval_id}:{condition}:candidate_messages"
                    )
                left_context_ids = [
                    context.get("chunk_id")
                    for context in left.get("retrieved_contexts", [])
                ]
                right_context_ids = [
                    context.get("chunk_id")
                    for context in right.get("retrieved_contexts", [])
                ]
                if left_context_ids != right_context_ids:
                    mismatches.append(
                        f"{other_key}:{eval_id}:{condition}:retrieved_contexts"
                    )
    return {
        "baseline_model_key": keys[0],
        "models_checked": len(keys),
        "passed": not mismatches,
        "mismatches": mismatches,
    }


def summarize_ragas(rows: list[dict[str, Any]], excluded: set[str]) -> dict[str, Any]:
    included = [row for row in rows if row.get("eval_id") not in excluded]
    metric_names = sorted(
        {
            name
            for row in included
            for name in row.get("ragas", {}).get("metrics", {})
        }
    )
    result: dict[str, Any] = {}
    for condition in CONDITIONS:
        selected = [row for row in included if row.get("condition") == condition]
        result[condition] = {}
        for metric_name in metric_names:
            metric_rows = [
                row.get("ragas", {}).get("metrics", {}).get(metric_name, {})
                for row in selected
            ]
            values = [
                value
                for metric in metric_rows
                if (value := finite_number(metric.get("value"))) is not None
            ]
            result[condition][metric_name] = {
                "mean": round(statistics.fmean(values), 6) if values else None,
                "valid_rows": len(values),
                "total_rows": len(selected),
                "invalid_reasons": sorted(
                    {
                        str(metric.get("reason"))
                        for metric in metric_rows
                        if finite_number(metric.get("value")) is None
                        and metric.get("reason")
                    }
                ),
            }
    return result


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Week 3 Three-Model RAG Run Audit",
        "",
        f"- Analyzer version: `{summary['analyzer_version']}`",
        f"- Excluded preflight IDs: `{', '.join(summary['excluded_eval_ids']) or 'none'}`",
        f"- Shared-input audit: `{'PASS' if summary['shared_input_audit']['passed'] else 'FAIL'}`",
        "",
        "## Uninspected aggregate",
        "",
        "| Model | Condition | Rows | Empty | Echo | Mean output tokens | Mean latency (ms) | Citation precision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_key, model_summary in summary["models"].items():
        for condition in CONDITIONS:
            item = model_summary["uninspected"]["conditions"][condition]
            citation_precision = (
                "n/a"
                if item["citation_precision"] is None
                else f"{item['citation_precision']:.3f}"
            )
            lines.append(
                f"| {model_key} | {condition} | {item['rows']} | "
                f"{item['empty_outputs']} | {item['question_echoes']} | "
                f"{item['mean_output_tokens']} | "
                f"{item['mean_generation_latency_ms']} | {citation_precision} |"
            )
    lines.extend(
        [
            "",
            "Automatic RAGAS values are diagnostic because the local Judge is "
            "uncalibrated. Mistral candidate rows use a non-independent Mistral "
            "self-judge and must not be used for a winner claim. A separate AI "
            "qualitative calibration reviews answer content.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(
    generation_paths: dict[str, Path],
    score_paths: dict[str, Path],
    excluded_eval_ids: set[str],
) -> dict[str, Any]:
    generation_rows = {
        key: read_jsonl(path) for key, path in generation_paths.items()
    }
    paired = {
        key: validate_generation_rows(key, rows)
        for key, rows in generation_rows.items()
    }
    summary: dict[str, Any] = {
        "analyzer_version": ANALYZER_VERSION,
        "excluded_eval_ids": sorted(excluded_eval_ids),
        "shared_input_audit": shared_input_audit(paired),
        "models": {},
    }
    for key, rows in generation_rows.items():
        model_summary = {
            "all_rows": summarize_subset(rows, set()),
            "uninspected": summarize_subset(rows, excluded_eval_ids),
        }
        if key in score_paths:
            model_summary["ragas_provisional"] = summarize_ragas(
                read_jsonl(score_paths[key]),
                excluded_eval_ids,
            )
        summary["models"][key] = model_summary
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generations",
        action="append",
        required=True,
        metavar="MODEL_KEY=PATH",
    )
    parser.add_argument(
        "--scores",
        action="append",
        default=[],
        metavar="MODEL_KEY=PATH",
    )
    parser.add_argument("--exclude-eval-id", action="append", default=[])
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    args = parser.parse_args()

    generation_paths = parse_keyed_paths(args.generations)
    score_paths = (
        parse_keyed_paths(args.scores) if args.scores else {}
    )
    unknown_score_keys = set(score_paths) - set(generation_paths)
    if unknown_score_keys:
        raise ValueError(f"Scores provided for unknown models: {unknown_score_keys}")

    summary = analyze(
        generation_paths,
        score_paths,
        set(args.exclude_eval_id),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.output_report.write_text(build_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

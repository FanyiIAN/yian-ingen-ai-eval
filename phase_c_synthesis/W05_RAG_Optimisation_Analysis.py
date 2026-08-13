"""Aggregate formal Week 5 RAG outputs, matched contrasts, and Pareto set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


ANALYZER_VERSION = "1.1.0"
HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "W05_RAG_Optimisation_Run_Config_v1.0.0.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def validate_and_join(
    generations: list[dict[str, Any]],
    ragas: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    expected_rows = int(config["factorial_design"]["expected_rows"])
    errors: list[str] = []
    for name, rows in (
        ("generation", generations), ("ragas", ragas), ("coverage", coverage)
    ):
        ids = [str(row.get("run_item_id", "")) for row in rows]
        if len(rows) != expected_rows:
            errors.append(f"{name}: expected {expected_rows} rows, found {len(rows)}")
        if any(not value for value in ids) or len(ids) != len(set(ids)):
            errors.append(f"{name}: run_item_id values are missing or duplicated")
    generation_ids = {row["run_item_id"] for row in generations}
    for name, rows in (("ragas", ragas), ("coverage", coverage)):
        if {row["run_item_id"] for row in rows} != generation_ids:
            errors.append(f"{name}: run_item_id set does not match generations")
    if errors:
        raise ValueError("invalid Week 5 scoring bundle:\n- " + "\n- ".join(errors))

    ragas_by_id = {row["run_item_id"]: row for row in ragas}
    coverage_by_id = {row["run_item_id"]: row for row in coverage}
    joined: list[dict[str, Any]] = []
    for generation in generations:
        rid = generation["run_item_id"]
        ragas_row = ragas_by_id[rid]
        coverage_row = coverage_by_id[rid]
        faithfulness = ragas_row["metrics"][
            "faithfulness_to_retrieved_context"
        ].get("value")
        relevance = ragas_row["metrics"]["answer_relevance"].get("value")
        normalized = coverage_row.get("normalized_coverage") or {}
        point_coverage = normalized.get("required_point_coverage")
        latency = (generation.get("latency_ms") or {}).get(
            "question_to_response_ms"
        )
        # RAGAS can return NaN when an answer contains no extractable claims.
        # Keep this as explicit missingness in the public bundle rather than
        # emitting non-standard JSON NaN values or imputing a diagnostic score.
        faithfulness = float(faithfulness) if finite(faithfulness) else None
        relevance = float(relevance) if finite(relevance) else None
        point_coverage = float(point_coverage) if finite(point_coverage) else None
        latency = float(latency) if finite(latency) else None
        joined.append(
            {
                "run_item_id": rid,
                "variant_id": generation["variant_id"],
                "eval_id": generation["eval_id"],
                "platform": generation["platform"],
                "chunk_size_tokens": int(generation["chunk_size_tokens"]),
                "top_k": int(generation["top_k"]),
                "reranking": generation["effective_reranking"],
                "model_id": generation["candidate_model_id"],
                "model_revision": generation["candidate_model_revision"],
                "evaluation_set_id": generation["evaluation_set_id"],
                "evaluation_set_version": generation["evaluation_set_version"],
                "random_seed": int(generation["random_seed"]),
                "candidate_messages_sha256": generation[
                    "candidate_messages_sha256"
                ],
                "candidate_output_sha256": generation[
                    "candidate_output_sha256"
                ],
                "retrieved_chunk_ids": [
                    context["chunk_id"]
                    for context in generation.get("retrieved_contexts") or []
                ],
                "faithfulness": faithfulness,
                "answer_relevance": relevance,
                "required_point_coverage": point_coverage,
                "forbidden_point_violations": len(
                    normalized.get("forbidden_point_violations") or []
                ),
                "evidence_fact_recall_at_k": generation[
                    "evidence_fact_recall_at_k"
                ],
                "metadata_filter_leakage": generation[
                    "metadata_filter_leakage"
                ],
                "question_to_response_ms": latency,
                "retrieval_latency_ms": generation["retrieval_latency_ms"],
                "generation_latency_ms": generation["generation_latency_ms"],
                "ragas_score_status": ragas_row["score_status"],
                "coverage_score_status": coverage_row["score_status"],
                "ragas_reused": "reused_from_run_item_id" in ragas_row,
                "coverage_reused": "reused_from_run_item_id" in coverage_row,
            }
        )
    variants = Counter(row["variant_id"] for row in joined)
    if set(variants.values()) != {20} or len(variants) != 18:
        raise ValueError(f"expected 20 items in each of 18 variants: {variants}")
    return joined


def aggregate(joined: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in joined:
        groups.setdefault(row["variant_id"], []).append(row)
    output: list[dict[str, Any]] = []
    for variant_id, rows in sorted(groups.items()):
        first = rows[0]
        metrics = {
            "faithfulness": [float(row["faithfulness"]) for row in rows if finite(row["faithfulness"])],
            "answer_relevance": [float(row["answer_relevance"]) for row in rows if finite(row["answer_relevance"])],
            "required_point_coverage": [
                float(row["required_point_coverage"])
                for row in rows
                if finite(row["required_point_coverage"])
            ],
            "question_to_response_ms": [
                float(row["question_to_response_ms"])
                for row in rows
                if finite(row["question_to_response_ms"])
            ],
        }
        output.append(
            {
                "variant_id": variant_id,
                "chunk_size_tokens": first["chunk_size_tokens"],
                "top_k": first["top_k"],
                "reranking": first["reranking"],
                "items": len(rows),
                "faithfulness_coverage": len(metrics["faithfulness"]) / len(rows),
                "mean_faithfulness": (
                    statistics.mean(metrics["faithfulness"])
                    if metrics["faithfulness"] else None
                ),
                "relevance_coverage": len(metrics["answer_relevance"]) / len(rows),
                "mean_answer_relevance": (
                    statistics.mean(metrics["answer_relevance"])
                    if metrics["answer_relevance"] else None
                ),
                "required_point_coverage_coverage": (
                    len(metrics["required_point_coverage"]) / len(rows)
                ),
                "mean_required_point_coverage": (
                    statistics.mean(metrics["required_point_coverage"])
                    if metrics["required_point_coverage"] else None
                ),
                "p50_question_to_response_ms": (
                    statistics.median(metrics["question_to_response_ms"])
                    if metrics["question_to_response_ms"] else None
                ),
                "p95_question_to_response_ms": (
                    percentile(metrics["question_to_response_ms"], 0.95)
                    if metrics["question_to_response_ms"] else None
                ),
                "mean_evidence_fact_recall_at_k": statistics.mean(
                    float(row["evidence_fact_recall_at_k"]) for row in rows
                ),
                "forbidden_point_violation_rows": sum(
                    row["forbidden_point_violations"] > 0 for row in rows
                ),
                "metadata_filter_leakage": sum(
                    int(row["metadata_filter_leakage"]) for row in rows
                ),
            }
        )
    for row in output:
        for key, value in list(row.items()):
            if isinstance(value, float):
                row[key] = round(value, 6)
    return output


def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    required = (
        "mean_faithfulness",
        "mean_required_point_coverage",
        "p50_question_to_response_ms",
    )
    if not all(finite(left.get(key)) and finite(right.get(key)) for key in required):
        return False
    left_values = (
        float(left["mean_faithfulness"]),
        float(left["mean_required_point_coverage"]),
        -float(left["p50_question_to_response_ms"]),
    )
    right_values = (
        float(right["mean_faithfulness"]),
        float(right["mean_required_point_coverage"]),
        -float(right["p50_question_to_response_ms"]),
    )
    return all(a >= b for a, b in zip(left_values, right_values)) and any(
        a > b for a, b in zip(left_values, right_values)
    )


def pareto_frontier(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in summaries
        if all(
            finite(row.get(key))
            for key in (
                "mean_faithfulness",
                "mean_required_point_coverage",
                "p50_question_to_response_ms",
            )
        )
    ]
    return [
        row
        for row in eligible
        if not any(
            dominates(other, row)
            for other in eligible
            if other["variant_id"] != row["variant_id"]
        )
    ]


def balanced_choice(frontier: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not frontier:
        return None
    def key(row: dict[str, Any]) -> tuple[float, float, str]:
        faith = float(row["mean_faithfulness"])
        coverage = float(row["mean_required_point_coverage"])
        harmonic = 0.0 if faith + coverage == 0 else 2 * faith * coverage / (faith + coverage)
        return (-harmonic, float(row["p50_question_to_response_ms"]), row["variant_id"])
    selected = min(frontier, key=key)
    faith = float(selected["mean_faithfulness"])
    coverage = float(selected["mean_required_point_coverage"])
    return {
        "variant_id": selected["variant_id"],
        "quality_harmonic_mean": round(
            0.0 if faith + coverage == 0 else 2 * faith * coverage / (faith + coverage),
            6,
        ),
        "selection_rule": (
            "highest harmonic mean of diagnostic Faithfulness and Coverage within "
            "the Pareto set; then lower p50 latency"
        ),
    }


def variant_differences(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    return [
        key
        for key in ("chunk_size_tokens", "top_k", "reranking")
        if left[key] != right[key]
    ]


def matched_contrasts(
    summaries: list[dict[str, Any]], joined: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows_by_variant = {
        variant_id: {row["eval_id"]: row for row in joined if row["variant_id"] == variant_id}
        for variant_id in {row["variant_id"] for row in joined}
    }
    contrasts: list[dict[str, Any]] = []
    for left, right in itertools.combinations(sorted(summaries, key=lambda row: row["variant_id"]), 2):
        differences = variant_differences(left, right)
        if len(differences) != 1:
            continue
        factor = differences[0]
        if factor in {"chunk_size_tokens", "top_k"} and left[factor] > right[factor]:
            left, right = right, left
        if factor == "reranking" and left[factor] == "cross_encoder":
            left, right = right, left
        paired = [
            (rows_by_variant[left["variant_id"]][eval_id], rows_by_variant[right["variant_id"]][eval_id])
            for eval_id in sorted(rows_by_variant[left["variant_id"]])
        ]
        delta: dict[str, float | None] = {}
        for metric in (
            "faithfulness", "answer_relevance", "required_point_coverage", "question_to_response_ms"
        ):
            values = [
                float(right_row[metric]) - float(left_row[metric])
                for left_row, right_row in paired
                if finite(left_row[metric]) and finite(right_row[metric])
            ]
            delta[f"mean_delta_{metric}"] = (
                round(statistics.mean(values), 6) if values else None
            )
            delta[f"paired_rows_{metric}"] = len(values)
        contrasts.append(
            {
                "factor": factor,
                "left_variant_id": left["variant_id"],
                "right_variant_id": right["variant_id"],
                "left_value": left[factor],
                "right_value": right[factor],
                "fixed_chunk_size_tokens": (
                    None if factor == "chunk_size_tokens"
                    else left["chunk_size_tokens"]
                ),
                "fixed_top_k": None if factor == "top_k" else left["top_k"],
                "fixed_reranking": (
                    None if factor == "reranking" else left["reranking"]
                ),
                **delta,
            }
        )
    return sorted(contrasts, key=lambda row: (row["factor"], str(row["left_value"]), str(row["right_value"]), row["left_variant_id"]))


def summarize_contrasts(
    contrasts: list[dict[str, Any]], group_key: str
) -> list[dict[str, Any]]:
    groups: dict[Any, list[dict[str, Any]]] = {}
    for row in contrasts:
        groups.setdefault(row[group_key], []).append(row)
    output: list[dict[str, Any]] = []
    metric_names = (
        "faithfulness",
        "answer_relevance",
        "required_point_coverage",
        "question_to_response_ms",
    )
    for group, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        summary: dict[str, Any] = {group_key: group, "contrasts": len(rows)}
        for metric in metric_names:
            key = f"mean_delta_{metric}"
            values = [float(row[key]) for row in rows if finite(row.get(key))]
            summary[key] = round(statistics.mean(values), 6) if values else None
        output.append(summary)
    return output


def chunk_identity(joined: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for row in joined:
        key = (row["top_k"], row["reranking"], row["eval_id"])
        groups.setdefault(key, []).append(row)
    comparable = [rows for rows in groups.values() if len(rows) == 3]
    message_identical = sum(
        len({row["candidate_messages_sha256"] for row in rows}) == 1
        for rows in comparable
    )
    output_identical = sum(
        len({row["candidate_output_sha256"] for row in rows}) == 1
        for rows in comparable
    )
    return {
        "three_chunk_level_matched_groups": len(comparable),
        "identical_candidate_message_groups": message_identical,
        "identical_candidate_output_groups": output_identical,
        "message_identity_rate": round(message_identical / len(comparable), 6),
        "output_identity_rate": round(output_identical / len(comparable), 6),
        "interpretation": (
            "Zero message and output identity across chunk levels confirms that "
            "complete-document chunking made chunk size an operational factor in "
            "this corrective experiment."
            if message_identical == 0 and output_identical == 0
            else "Any identity across chunk levels is measured explicitly and must "
            "be considered when interpreting the chunk-size contrast."
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            encoded = {
                key: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (list, dict))
                else value
                for key, value in row.items()
            }
            writer.writerow(encoded)


def report_number(value: Any, digits: int) -> str:
    return f"{float(value):.{digits}f}" if finite(value) else "NA"


def public_model_registry(registry: dict[str, Any]) -> dict[str, Any]:
    """Retain model identity/revisions without publishing machine-local paths."""
    return {
        role: {
            key: value
            for key, value in details.items()
            if key != "local_model_directory"
        }
        for role, details in registry.items()
    }


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Week 5 Long-Source Senpai RAG Optimisation",
        "",
        "> **Status: latest corrective public RAG optimisation result (v1.1.0).** "
        "The experiment uses complete long documents, so chunk size changes the "
        "actual indexed retrieval units. Automated Judge metrics remain diagnostic.",
        "",
        "**Design:** 3 chunk sizes × 3 top-k values × reranking off/on",
        "**Items:** 20 frozen long-document Senpai questions per configuration",
        "**Corpus:** 21 complete official public sources with status-aware metadata",
        "**Candidate:** `meta-llama/Llama-3.1-8B-Instruct` revision "
        "`0e9e39f249a16976918f6564b8830bc894c89659`",
        "**Seed:** `42`; deterministic decoding; randomized variant-block order",
        "**Timing:** one discarded warm-up; cold model load recorded separately; "
        "Pareto latency uses warm-path request time",
        "**Judge runtime:** the frozen local Mistral checkpoint scored the first "
        "112 rows behind an 8,192-token vLLM service. A pre-output top-k=5 batch "
        "exposed a 6,145-input + 2,048-output capacity overflow; no rows from that "
        "failed batch were retained. The same checkpoint, decoding and metric "
        "definitions resumed behind a 16,384-token loopback service for the "
        "remaining rows.",
        "",
        "## Reading the experiment",
        "",
        "A **factor** is an input deliberately changed by the experiment: chunk "
        "size, top-k, or reranking. A **full factorial** design tests every "
        "combination of their levels (3 × 3 × 2 = 18 cells). A **matched "
        "contrast** compares two cells that differ in only one factor—for "
        "example, reranking off versus on while chunk size and top-k stay fixed. "
        "An **interaction** means one factor's effect depends on another, such as "
        "reranking helping at top-k 5 but not top-k 1. A **confounder** is an "
        "uncontrolled difference that could offer an alternative explanation; "
        "frozen questions/models and randomized variant-block order reduce, but "
        "do not eliminate, such risks.",
        "",
        "**Cold start** includes model/index loading; **warm steady-state** measures "
        "requests after one excluded warm-up. The two are reported separately. A "
        "**Pareto-optimal** cell is not beaten by another tested cell on all three "
        "objectives: higher diagnostic Faithfulness, higher required-point "
        "Coverage, and lower warm-path latency.",
        "",
        "## Pareto result",
        "",
        "The non-dominated set maximizes diagnostic Faithfulness and weighted "
        "required-point Coverage while minimizing warm-path p50 question-to-response "
        "latency. Only cells with complete Faithfulness, Coverage, and latency are "
        "eligible for the primary frontier; incomplete cells remain in the full "
        "table. Relevance is reported as a supporting metric, not an optimization axis.",
        "",
        "| Variant | Faithfulness | Coverage | Relevance | p50 ms |",
        "|---|---:|---:|---:|---:|",
    ]
    by_id = {row["variant_id"]: row for row in result["variant_summaries"]}
    for variant_id in result["pareto_variant_ids"]:
        row = by_id[variant_id]
        lines.append(
            f"| `{variant_id}` | {report_number(row['mean_faithfulness'], 4)} | "
            f"{report_number(row['mean_required_point_coverage'], 4)} | "
            f"{report_number(row['mean_answer_relevance'], 4)} | "
            f"{report_number(row['p50_question_to_response_ms'], 1)} |"
        )
    choice = result.get("balanced_choice")
    if choice:
        lines.extend(
            [
                "",
                f"The transparent balanced choice within the frontier is "
                f"`{choice['variant_id']}`. This is a diagnostic configuration "
                "recommendation, not a production-readiness claim.",
            ]
        )
    lines.extend(
        [
            "",
            "## All factorial cells",
            "",
            "| Variant | Faithfulness | F coverage | Coverage | C coverage | Relevance | p50 ms | p95 ms |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["variant_summaries"]:
        lines.append(
            f"| `{row['variant_id']}` | {report_number(row['mean_faithfulness'], 4)} | "
            f"{report_number(row['faithfulness_coverage'], 3)} | "
            f"{report_number(row['mean_required_point_coverage'], 4)} | "
            f"{report_number(row['required_point_coverage_coverage'], 3)} | "
            f"{report_number(row['mean_answer_relevance'], 4)} | "
            f"{report_number(row['p50_question_to_response_ms'], 1)} | "
            f"{report_number(row['p95_question_to_response_ms'], 1)} |"
        )
    identity = result["chunk_level_identity"]
    factors = result["matched_contrasts_by_factor"]
    lines.extend(
        [
            "",
            "## Controlled comparisons",
            "",
            f"All `{len(result['matched_contrasts'])}` registered matched contrasts "
            "differ in exactly one factor: "
            f"`{factors.get('chunk_size_tokens', 0)}` chunk-size, "
            f"`{factors.get('top_k', 0)}` top-k, and "
            f"`{factors.get('reranking', 0)}` reranking contrasts. Across "
            f"`{identity['three_chunk_level_matched_groups']}` top-k/reranking/item "
            f"groups, candidate messages were identical across chunk levels at rate "
            f"`{identity['message_identity_rate']:.3f}` and outputs at rate "
            f"`{identity['output_identity_rate']:.3f}`. "
            f"{identity['interpretation']}",
            "",
            "Positive quality deltas favor the right-hand factor level; positive "
            "latency deltas mean it is slower. Factor-level averages are descriptive "
            "summaries of the registered matched contrasts:",
            "",
            "| Factor | n | Δ Faith | Δ Relevance | Δ Coverage | Δ latency ms |",
            "|---|---:|---:|---:|---:|---:|",
            *[
                f"| {row['factor']} | {row['contrasts']} | "
                f"{report_number(row['mean_delta_faithfulness'], 4)} | "
                f"{report_number(row['mean_delta_answer_relevance'], 4)} | "
                f"{report_number(row['mean_delta_required_point_coverage'], 4)} | "
                f"{report_number(row['mean_delta_question_to_response_ms'], 1)} |"
                for row in result["contrast_factor_summary"]
            ],
            "",
            "The reranking effect is conditional on top-k:",
            "",
            "| Fixed top-k | n | Δ Faith | Δ Relevance | Δ Coverage | Δ latency ms |",
            "|---:|---:|---:|---:|---:|---:|",
            *[
                f"| {row['fixed_top_k']} | {row['contrasts']} | "
                f"{report_number(row['mean_delta_faithfulness'], 4)} | "
                f"{report_number(row['mean_delta_answer_relevance'], 4)} | "
                f"{report_number(row['mean_delta_required_point_coverage'], 4)} | "
                f"{report_number(row['mean_delta_question_to_response_ms'], 1)} |"
                for row in result["reranking_interaction_by_top_k"]
            ],
            "",
            "## Completeness and scoring audit",
            "",
            "| Metric | Finite fraction |",
            "|---|---:|",
            *[
                f"| {metric} | {fraction:.3f} |"
                for metric, fraction in result["metric_coverage"].items()
            ],
            "",
            f"RAGAS statuses: `{result['scoring_audit']['ragas_status_counts']}`; "
            f"coverage statuses: `{result['scoring_audit']['coverage_status_counts']}`. "
            "Missing metrics are excluded from the relevant mean and their finite "
            "fraction is reported; no diagnostic value is imputed.",
            "Rows marked `parsed_after_deterministic_repair` contained every "
            "registered rubric point plus extra, unregistered point IDs. The repair "
            "discarded only those extras and retained an audit trail; it never "
            "created a missing registered-point score or altered a registered "
            "Judge verdict.",
            "",
            "## Reliability boundary",
            "",
            result["claim_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml(args.config)
    generation_rows = read_jsonl(args.generations)
    ragas_rows = read_jsonl(args.ragas)
    coverage_rows = read_jsonl(args.coverage)
    joined = validate_and_join(
        generation_rows, ragas_rows, coverage_rows, config
    )
    summaries = aggregate(joined)
    pareto_eligible = [
        row
        for row in summaries
        if row["faithfulness_coverage"] == 1.0
        and row["required_point_coverage_coverage"] == 1.0
        and finite(row["p50_question_to_response_ms"]) is not None
    ]
    frontier = pareto_frontier(pareto_eligible)
    contrasts = matched_contrasts(summaries, joined)
    result = {
        "analysis_version": ANALYZER_VERSION,
        "experiment_id": config["experiment_id"],
        "experiment_version": config["experiment_version"],
        "random_seed": config["random_seed"],
        "model_registry": public_model_registry(config["models"]),
        "evaluation_set": config["source_inputs"]["evaluation_set"],
        "source_files": {
            "generations": {"sha256": sha256_file(args.generations), "rows": len(generation_rows)},
            "ragas": {"sha256": sha256_file(args.ragas), "rows": len(ragas_rows)},
            "coverage": {"sha256": sha256_file(args.coverage), "rows": len(coverage_rows)},
        },
        "variant_summaries": summaries,
        "pareto_variant_ids": sorted(row["variant_id"] for row in frontier),
        "pareto_eligibility": {
            "rule": "complete_faithfulness_coverage_and_latency_only",
            "eligible_variant_ids": sorted(
                row["variant_id"] for row in pareto_eligible
            ),
            "excluded_incomplete_variant_ids": sorted(
                row["variant_id"]
                for row in summaries
                if row not in pareto_eligible
            ),
        },
        "balanced_choice": balanced_choice(frontier),
        "matched_contrasts": contrasts,
        "matched_contrasts_by_factor": dict(Counter(row["factor"] for row in contrasts)),
        "contrast_factor_summary": summarize_contrasts(contrasts, "factor"),
        "reranking_interaction_by_top_k": summarize_contrasts(
            [row for row in contrasts if row["factor"] == "reranking"],
            "fixed_top_k",
        ),
        "chunk_level_identity": chunk_identity(joined),
        "metric_coverage": {
            "faithfulness": sum(finite(row["faithfulness"]) for row in joined) / len(joined),
            "answer_relevance": sum(finite(row["answer_relevance"]) for row in joined) / len(joined),
            "required_point_coverage": sum(finite(row["required_point_coverage"]) for row in joined) / len(joined),
            "latency": sum(finite(row["question_to_response_ms"]) for row in joined) / len(joined),
        },
        "scoring_audit": {
            "ragas_status_counts": dict(sorted(Counter(
                str(row.get("score_status", "missing")) for row in ragas_rows
            ).items())),
            "coverage_status_counts": dict(sorted(Counter(
                str(row.get("score_status", "missing")) for row in coverage_rows
            ).items())),
            "ragas_scorer_versions": sorted({
                str(row.get("scorer_version", "missing")) for row in ragas_rows
            }),
            "coverage_scorer_versions": sorted({
                str(row.get("scorer_version", "missing")) for row in coverage_rows
            }),
            "ragas_reused_rows": sum(
                "reused_from_run_item_id" in row for row in ragas_rows
            ),
            "coverage_reused_rows": sum(
                "reused_from_run_item_id" in row for row in coverage_rows
            ),
            "coverage_deterministically_repaired_rows": sum(
                row.get("score_status") == "parsed_after_deterministic_repair"
                for row in coverage_rows
            ),
            "coverage_repair_versions": sorted({
                str((row.get("coverage_repair") or {}).get("repair_version"))
                for row in coverage_rows
                if row.get("coverage_repair")
            }),
        },
        "claim_boundary": config["claim_boundary"],
        "interpretation": (
            "Pareto membership is conditional on this 20-item public-source subset, "
            "one A40 run, and uncalibrated local evaluator metrics. Matched contrasts "
            "support factor attribution within this registered design; they do not "
            "identify a universal production optimum."
        ),
    }
    for key, value in result["metric_coverage"].items():
        result["metric_coverage"][key] = round(value, 6)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_csv(args.summary_csv, summaries)
    write_csv(args.item_csv, joined)
    args.report.write_text(render_report(result), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--ragas", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--item-csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.json, args.summary_csv, args.item_csv, args.report):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    result = analyze(args)
    print(
        json.dumps(
            {
                "status": "ok",
                "variants": len(result["variant_summaries"]),
                "pareto_variants": len(result["pareto_variant_ids"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

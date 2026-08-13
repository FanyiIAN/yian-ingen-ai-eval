"""Build and verify the frozen Week 6 evidence synthesis.

This pipeline intentionally uses only the Python standard library so that the
evidence audit can run from a clean clone on Windows or Linux without model
weights, network access, or an LLM judge.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import re
import sys
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "1.0.0"
REGISTRY_NAME = "W06_Evidence_Registry_v1.0.0.json"
SUMMARY_NAME = "W06_Evidence_Summary_v1.0.0.json"
MATRIX_NAME = "W06_Claim_Evidence_Matrix_v1.0.0.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def parse_simple_scenario_fields(text: str) -> dict[str, list[str]]:
    fields = {"scenario_id": [], "platform": [], "split": [], "severity_class": []}
    patterns = {
        "scenario_id": re.compile(r"^\s*-\s+scenario_id:\s*(.+?)\s*$"),
        "platform": re.compile(r"^\s+platform:\s*(.+?)\s*$"),
        "split": re.compile(r"^\s+split:\s*(.+?)\s*$"),
        "severity_class": re.compile(r"^\s+severity_class:\s*(.+?)\s*$"),
    }
    for line in text.splitlines():
        for name, pattern in patterns.items():
            match = pattern.match(line)
            if match:
                fields[name].append(match.group(1).strip(" '\""))
    return fields


def counts(values: list[str]) -> dict[str, int]:
    return {value: values.count(value) for value in sorted(set(values))}


def find_scope(rows: list[dict[str, Any]], scope_value: str) -> dict[str, Any]:
    return next(row for row in rows if row.get("scope_value") == scope_value)


def verify_sources(repo_root: Path, registry: dict[str, Any]) -> list[dict[str, Any]]:
    verified = []
    for source in registry["sources"]:
        path = repo_root / source["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing registered source: {source['path']}")
        observed = sha256_file(path)
        if observed != source["sha256"]:
            raise ValueError(
                f"Hash mismatch for {source['path']}: expected {source['sha256']}, observed {observed}"
            )
        verified.append(
            {
                "path": source["path"],
                "role": source["role"],
                "sha256": observed,
                "bytes": path.stat().st_size,
            }
        )
    return verified


def build_summary(repo_root: Path, registry: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    verified = verify_sources(repo_root, registry)

    scenario_text = (repo_root / "phase_a_design/W02_Scenarios.yaml").read_text(encoding="utf-8")
    scenario_fields = parse_simple_scenario_fields(scenario_text)
    temporal_hits = []
    lowered = scenario_text.lower()
    for pattern in registry["temporal_validity_audit"]["automated_trigger_patterns"]:
        if pattern.lower() in lowered:
            temporal_hits.append(pattern)

    agreement = load_json(repo_root / "phase_a_design/W02_Baseline_Agreement.json")
    agreement_all = agreement["agreement_all_responses"]
    w03 = load_json(repo_root / "phase_b_evaluation/W03_RAG_Long_Source_Summary_v1.0.0.json")
    w04_robustness = load_json(repo_root / "phase_b_evaluation/W04_Robustness_Summary_v0.1.0.json")
    w04_vlm = load_json(
        repo_root / "phase_b_evaluation/W04_Multimodal_Architecture_Comparison_v0.2.0.json"
    )
    w04_rag_perf = load_json(
        repo_root / "phase_b_evaluation/W04_RAG_Long_Performance_Summary_v1.0.0.json"
    )
    w05 = load_json(
        repo_root / "phase_c_synthesis/W05_RAG_Long_Source_Optimisation_Summary_v1.1.0.json"
    )
    w05_accumulated = load_json(
        repo_root / "phase_c_synthesis/W05_Accumulated_Evidence_Analysis_v1.1.0.json"
    )

    w03_fari = find_scope(w03["summary_rows"], "Fari::rag")
    w03_senpai = find_scope(w03["summary_rows"], "Senpai::rag")
    robustness_by_model = {
        row["candidate_model_key"]: row for row in w04_robustness["models"]
    }
    vlm_by_model = {row["model_key"]: row for row in w04_vlm["models"]}
    rag_perf_by_condition = {row["condition_id"]: row for row in w04_rag_perf["groups"]}

    summary = {
        "schema_version": "1.0.0",
        "pipeline_version": SCRIPT_VERSION,
        "registry_version": registry["registry_version"],
        "random_seed": registry["random_seed"],
        "source_verification": {
            "status": "passed",
            "verified_source_count": len(verified),
            "sources": verified,
        },
        "benchmark_design": {
            "scenario_count": len(scenario_fields["scenario_id"]),
            "platform_counts": counts(scenario_fields["platform"]),
            "severity_counts": counts(scenario_fields["severity_class"]),
            "split_counts": counts(scenario_fields["split"]),
            "data_origin": "synthetic_public_safe",
            "claim_boundary": "Product-context proxies; no deployed InGen system was measured.",
        },
        "scoring_reliability": {
            "three_prompt_formulation_agreement": {
                "task_accuracy_alpha_ordinal": agreement_all["task_accuracy"][
                    "krippendorff_alpha_ordinal"
                ],
                "contextual_grounding_alpha_ordinal": agreement_all[
                    "contextual_grounding"
                ]["krippendorff_alpha_ordinal"],
                "failure_mode_alpha_nominal": agreement_all["failure_mode"][
                    "krippendorff_alpha_nominal"
                ],
                "response_count": agreement_all["response_count"],
            },
            "frozen_human_label_calibration_task_alpha": 0.7551,
            "preregistered_calibration_threshold": 0.8,
            "calibration_gate_passed": False,
            "evidence_status": "diagnostic_failed_calibration",
            "interpretation": "Prompt-formulation agreement is not a substitute for agreement with independent human labels.",
        },
        "temporal_validity": {
            "control": "prompt_closed_no_external_event_recall",
            "automated_external_time_trigger_count": len(temporal_hits),
            "automated_external_time_triggers": temporal_hits,
            "manual_review": registry["temporal_validity_audit"]["manual_review"],
            "residual_confound": registry["temporal_validity_audit"]["residual_confound"],
        },
        "week3_long_source_rag": {
            "evaluation_set_id": w03["evaluation_set_id"],
            "question_count": w03["question_count"],
            "official_rows": w03["official_rows"],
            "retrieval": w03["retrieval_summary"],
            "matched_rag_minus_base": w03["matched_rag_minus_base"],
            "platform_rag": {
                "Fari": {
                    "answer_relevance_mean": w03_fari["answer_relevance_mean"],
                    "faithfulness_mean": w03_fari["faithfulness_mean"],
                    "required_point_coverage_mean": w03_fari[
                        "required_point_coverage_mean"
                    ],
                },
                "Senpai": {
                    "answer_relevance_mean": w03_senpai["answer_relevance_mean"],
                    "faithfulness_mean": w03_senpai["faithfulness_mean"],
                    "required_point_coverage_mean": w03_senpai[
                        "required_point_coverage_mean"
                    ],
                },
            },
            "evidence_status": "diagnostic_uncalibrated",
        },
        "week4_robustness": {
            "row_count": w04_robustness["row_count"],
            "semantic_robustness": {
                key: {
                    "score": value["semantic_robustness"]["semantic_robustness_score"],
                    "stable_pass": value["semantic_robustness"]["stable_pass_scenario_count"],
                    "stable_fail": value["semantic_robustness"]["stable_fail_scenario_count"],
                }
                for key, value in robustness_by_model.items()
            },
            "evidence_status": w04_robustness["score_status"],
            "interpretation": "Consistency can be high because a model fails consistently; robustness and correctness must be reported together.",
        },
        "week4_multimodal": {
            "matched_request_count_per_model": w04_vlm["controlled_inputs"][
                "matched_request_count_per_model"
            ],
            "pairwise_clean_comparison": w04_vlm["pairwise_clean_comparison"],
            "models": {
                key: {
                    "model_id": value["identity"]["model_id"],
                    "model_revision": value["identity"]["model_revision"],
                    "mean_question_to_response_ms": value["performance"][
                        "question_to_response_ms"
                    ]["mean"],
                    "clean_mean_total_score": next(
                        row["mean_total_score"]
                        for row in value["quality"]["conditions"]
                        if row["condition_id"] == "clean"
                    ),
                }
                for key, value in vlm_by_model.items()
            },
            "evidence_status": w04_vlm["score_status"],
        },
        "week4_rag_performance": {
            "row_count": w04_rag_perf["row_count"],
            "base_mean_question_to_response_ms": rag_perf_by_condition["base"]["latency_ms"][
                "question_to_response_ms"
            ]["mean"],
            "rag_mean_question_to_response_ms": rag_perf_by_condition["rag"]["latency_ms"][
                "question_to_response_ms"
            ]["mean"],
            "rag_mean_retrieval_total_ms": rag_perf_by_condition["rag"]["latency_ms"][
                "retrieval_total_ms"
            ]["mean"],
            "interpretation_boundary": w04_rag_perf["interpretation_boundary"],
        },
        "week5_rag_optimisation": {
            "experiment_id": w05["experiment_id"],
            "experiment_version": w05["experiment_version"],
            "factorial_cell_count": len(w05["variant_summaries"]),
            "pareto_variant_ids": w05["pareto_variant_ids"],
            "balanced_choice": w05["balanced_choice"],
            "contrast_factor_summary": w05["contrast_factor_summary"],
            "reranking_interaction_by_top_k": w05["reranking_interaction_by_top_k"],
            "metric_coverage": w05["metric_coverage"],
            "evidence_status": "diagnostic_uncalibrated",
            "interpretation": w05["interpretation"],
        },
        "cross_week_interpretation_boundary": w05_accumulated["interpretation_boundary"],
        "replication_statement": {
            "likely_replicable": [
                "frozen input hashes and row counts",
                "deterministic candidate outputs under pinned revisions and greedy decoding",
                "registered matched contrasts and Pareto computation from the frozen rows",
            ],
            "not_yet_established": [
                "exact AI-assisted rubric scores under a different Judge",
                "validated model rankings",
                "deployed-product safety, readiness, or causal mechanisms",
            ],
        },
    }

    claims = build_claims(summary)
    return summary, claims


def build_claims(summary: dict[str, Any]) -> list[dict[str, Any]]:
    reliability = summary["scoring_reliability"]["three_prompt_formulation_agreement"]
    rag_delta = summary["week3_long_source_rag"]["matched_rag_minus_base"]
    vlm = summary["week4_multimodal"]
    claims = [
        {
            "claim_id": "W06-C01",
            "claim": "The benchmark contains 35 product-context proxy scenarios, seven per platform.",
            "value": "35; 7 each",
            "unit": "scenarios",
            "evidence_status": "deterministic_audit",
            "scope": "W02 scenario bank v0.2.0",
            "source_paths": "phase_a_design/W02_Scenarios.yaml",
            "replication_expectation": "exact",
            "causal_language_allowed": "no",
        },
        {
            "claim_id": "W06-C02",
            "claim": "Three Judge prompt formulations agreed most on Task Accuracy and least on Failure Mode.",
            "value": f"{reliability['task_accuracy_alpha_ordinal']:.4f}; {reliability['contextual_grounding_alpha_ordinal']:.4f}; {reliability['failure_mode_alpha_nominal']:.4f}",
            "unit": "Krippendorff alpha: task; grounding; failure",
            "evidence_status": "diagnostic_failed_calibration",
            "scope": "70 Week 2 responses; three prompt formulations",
            "source_paths": "phase_a_design/W02_Baseline_Agreement.json",
            "replication_expectation": "exact from frozen ratings",
            "causal_language_allowed": "no",
        },
        {
            "claim_id": "W06-C03",
            "claim": "The frozen Judge calibration missed the preregistered alpha threshold, so no validated text-model ranking is claimed.",
            "value": "0.7551 < 0.8000",
            "unit": "ordinal Krippendorff alpha",
            "evidence_status": "diagnostic_failed_calibration",
            "scope": "frozen provisional human-label calibration",
            "source_paths": "phase_a_design/W02_Final_Run_and_Judge_Findings.md",
            "replication_expectation": "exact from frozen calibration rows",
            "causal_language_allowed": "no",
        },
        {
            "claim_id": "W06-C04",
            "claim": "On the 40-question long-source set, matched RAG answers increased diagnostic relevance and required-point coverage while increasing generation latency.",
            "value": f"+{rag_delta['answer_relevance']['mean']:.6f}; +{rag_delta['required_point_coverage']['mean']:.6f}; +{rag_delta['generation_latency_ms']['mean']:.3f}",
            "unit": "relevance; coverage; ms",
            "evidence_status": "diagnostic_uncalibrated",
            "scope": "one Llama revision; 40 matched public-source questions; A40",
            "source_paths": "phase_b_evaluation/W03_RAG_Long_Source_Summary_v1.0.0.json",
            "replication_expectation": "direction likely; exact Judge metrics evaluator-dependent",
            "causal_language_allowed": "within matched pipeline only",
        },
        {
            "claim_id": "W06-C05",
            "claim": "Semantic consistency is not equivalent to correctness: FLAN was the most consistent but most often failed consistently.",
            "value": "0.9143 consistency; 25 stable fails; 7 stable passes",
            "unit": "rate and scenarios",
            "evidence_status": "diagnostic_uncalibrated",
            "scope": "35 scenarios and three paraphrases per scenario",
            "source_paths": "phase_b_evaluation/W04_Robustness_Summary_v0.1.0.json",
            "replication_expectation": "exact from frozen AI-assisted rows",
            "causal_language_allowed": "no",
        },
        {
            "claim_id": "W06-C06",
            "claim": "The two VLMs were nearly tied on clean proxy-image quality, while LLaVA had lower mean response latency on the A40 run.",
            "value": f"19 ties, 1 Idefics2 win; {vlm['models']['llava_1_5_7b_hf']['mean_question_to_response_ms']:.1f} vs {vlm['models']['idefics2_8b_chatty']['mean_question_to_response_ms']:.1f}",
            "unit": "clean scenarios; ms",
            "evidence_status": "diagnostic_uncalibrated",
            "scope": "20 public-image proxies; 60 matched requests per model; one A40",
            "source_paths": "phase_b_evaluation/W04_Multimodal_Architecture_Comparison_v0.2.0.json",
            "replication_expectation": "raw outputs and timing protocol reproducible; rubric scores Judge-dependent",
            "causal_language_allowed": "architecture association only",
        },
        {
            "claim_id": "W06-C07",
            "claim": "Three of 18 registered Week 5 RAG cells were Pareto-optimal; the balanced diagnostic choice used 1024-token chunks, top-k 5, and cross-encoder reranking.",
            "value": "; ".join(summary["week5_rag_optimisation"]["pareto_variant_ids"]),
            "unit": "configuration IDs",
            "evidence_status": "diagnostic_uncalibrated",
            "scope": "20 Senpai questions; one Llama revision; one A40 run",
            "source_paths": "phase_c_synthesis/W05_RAG_Long_Source_Optimisation_Summary_v1.1.0.json",
            "replication_expectation": "exact Pareto computation from frozen rows",
            "causal_language_allowed": "within registered factor levels only",
        },
    ]
    return claims


def csv_text(rows: list[dict[str, Any]]) -> str:
    fields = [
        "claim_id",
        "claim",
        "value",
        "unit",
        "evidence_status",
        "scope",
        "source_paths",
        "replication_expectation",
        "causal_language_allowed",
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def write_or_verify(path: Path, expected: str, verify_only: bool) -> None:
    if verify_only:
        if not path.is_file():
            raise FileNotFoundError(f"Missing generated artifact: {path.name}")
        observed = path.read_text(encoding="utf-8")
        if observed != expected:
            raise ValueError(f"Generated artifact is stale: {path.name}")
    else:
        path.write_text(expected, encoding="utf-8", newline="")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Verify sources and committed outputs without rewriting them.")
    parser.add_argument("--verification-record", type=Path, help="Optional path for a machine-readable environment verification record.")
    args = parser.parse_args(argv)

    phase_dir = Path(__file__).resolve().parent
    repo_root = phase_dir.parent
    registry_path = phase_dir / REGISTRY_NAME
    registry = load_json(registry_path)
    summary, claims = build_summary(repo_root, registry)
    summary_serialized = canonical_json(summary)
    matrix_serialized = csv_text(claims)

    write_or_verify(phase_dir / SUMMARY_NAME, summary_serialized, args.verify_only)
    write_or_verify(phase_dir / MATRIX_NAME, matrix_serialized, args.verify_only)

    if args.verification_record:
        record = {
            "schema_version": "1.0.0",
            "pipeline_version": SCRIPT_VERSION,
            "status": "passed",
            "verify_only": args.verify_only,
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "registered_source_count": len(registry["sources"]),
            "summary_sha256": hashlib.sha256(summary_serialized.encode("utf-8")).hexdigest(),
            "matrix_sha256": hashlib.sha256(matrix_serialized.encode("utf-8")).hexdigest(),
        }
        args.verification_record.write_text(canonical_json(record), encoding="utf-8", newline="")

    mode = "verified" if args.verify_only else "generated"
    print(f"Week 6 evidence synthesis {mode}: {len(registry['sources'])} sources, {len(claims)} claims")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

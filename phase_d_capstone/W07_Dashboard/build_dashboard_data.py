"""Build the frozen, presentation-ready CSV layer for the Week 7 dashboard.

The Streamlit app never imports raw experiment artifacts or computes evaluation
metrics. Run this builder only when a registered Week 2-6 input changes, review
the diff, and commit the resulting CSV files with the dashboard.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


DASHBOARD_DIR = Path(__file__).resolve().parent
REPO_ROOT = DASHBOARD_DIR.parents[1]
DATA_DIR = DASHBOARD_DIR / "data"

SOURCE_PATHS = [
    "phase_b_evaluation/W03_Three_Model_Diagnostic_Comparison.csv",
    "phase_b_evaluation/W03_RAG_Long_Source_Summary_v1.0.0.csv",
    "phase_b_evaluation/W04_Robustness_Summary_v0.1.0.json",
    "phase_b_evaluation/W04_Robustness_Curves_v0.1.0.csv",
    "phase_b_evaluation/W04_Multimodal_Architecture_Comparison_v0.2.0.csv",
    "phase_c_synthesis/W05_RAG_Long_Source_Optimisation_Summary_v1.1.0.json",
    "phase_c_synthesis/W05_RAG_Long_Optimisation_Run_Config_v1.1.0.json",
    "phase_c_synthesis/W06_Evidence_Summary_v1.0.0.json",
]

MODEL_LABELS = {
    "google/flan-t5-base": "FLAN-T5 Base",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama 3.1 8B Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2": "Mistral 7B Instruct v0.2",
    "flan_t5_base": "FLAN-T5 Base",
    "llama31_8b_instruct": "Llama 3.1 8B Instruct",
    "mistral_7b_instruct_v0_2": "Mistral 7B Instruct v0.2",
    "idefics2_8b_chatty": "Idefics2 8B Chatty",
    "llava_1_5_7b_hf": "LLaVA 1.5 7B",
}

PLATFORM_LABELS = {
    "Aido_Humanoid": "Aido Humanoid",
    "Aido_Rover": "Aido Rover",
    "Fari": "Fari",
    "Senpai": "Senpai",
    "Sentinel_Prime_AI": "Sentinel Prime AI",
    "overall": "Portfolio",
}

FAILURE_CODES = ["unsafe", "hallucination", "off_policy", "refusal", "partial", "unresolved"]
ACTIONABLE_FAILURE_CODES = ["unsafe", "hallucination", "off_policy", "refusal", "partial"]
TIE_PRIORITY = {"unsafe": 5, "hallucination": 4, "off_policy": 3, "refusal": 2, "partial": 1}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _int(value: Any) -> int:
    return int(float(value))


def _round(value: float | None, digits: int = 6) -> float | str:
    return "" if value is None else round(value, digits)


def _mean_finite(*values: float | None) -> float | None:
    finite = [value for value in values if value is not None]
    return fmean(finite) if finite else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(filename: str, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / filename
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_model_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    source = _read_csv(REPO_ROOT / SOURCE_PATHS[0])
    scorecard: list[dict[str, Any]] = []

    for row in source:
        n = _int(row["n"])
        task = _float(row["task_severity_weighted"])
        grounding = _float(row["grounding_severity_weighted"])
        composite = _mean_finite(task, grounding)
        min_coverage = min(_int(row["task_resolved_n"]), _int(row["grounding_resolved_n"])) / n
        scorecard.append(
            {
                "platform": PLATFORM_LABELS.get(row["scope"], row["scope"]),
                "model": MODEL_LABELS.get(row["model_id"], row["model_id"]),
                "model_id": row["model_id"],
                "model_revision": row["model_revision"],
                "scenario_n": n,
                "task_severity_weighted_1_to_5": _round(task, 4),
                "grounding_severity_weighted_1_to_5": _round(grounding, 4),
                "severity_weighted_composite_1_to_5": _round(composite, 4),
                "diagnostic_readiness_proxy_0_to_100": _round(composite / 5 * 100 if composite is not None else None, 1),
                "minimum_dimension_coverage": _round(min_coverage, 4),
                "mean_latency_ms": _round(_float(row["mean_latency_ms"]), 2),
                "unsafe_count": _int(row["failure_unsafe"]),
                "evidence_status": "diagnostic_failed_calibration",
                "benchmark_version": "0.2.0",
                "seed": 42,
            }
        )

    by_platform = [row for row in source if row["scope"] != "overall"]
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    denominators: dict[str, int] = defaultdict(int)
    for row in by_platform:
        platform = PLATFORM_LABELS.get(row["scope"], row["scope"])
        denominators[platform] += _int(row["n"])
        for code in FAILURE_CODES:
            counts[platform][code] += _int(row[f"failure_{code}"])

    failure_heatmap: list[dict[str, Any]] = []
    concerns: list[dict[str, Any]] = []
    for platform in sorted(counts):
        denominator = denominators[platform]
        failure_heatmap.append(
            {
                "platform": platform,
                **{code: round(100 * counts[platform][code] / denominator, 1) for code in FAILURE_CODES},
                "observations": denominator,
                "evidence_status": "diagnostic_failed_calibration",
            }
        )
        ordered = sorted(
            ACTIONABLE_FAILURE_CODES,
            key=lambda code: (-counts[platform][code], -TIE_PRIORITY[code], code),
        )[:3]
        for rank, code in enumerate(ordered, start=1):
            concerns.append(
                {
                    "platform": platform,
                    "rank": rank,
                    "failure_code": code,
                    "observed_count": counts[platform][code],
                    "observed_rate_pct": round(100 * counts[platform][code] / denominator, 1),
                    "ranking_rule": "count_desc_then_consequence_priority_for_ties",
                    "evidence_status": "diagnostic_failed_calibration",
                }
            )

    portfolio_rows = [row for row in scorecard if row["platform"] == "Portfolio"]
    selected = max(portfolio_rows, key=lambda row: float(row["diagnostic_readiness_proxy_0_to_100"]))
    unsafe_total = sum(_int(row["failure_unsafe"]) for row in source if row["scope"] == "overall")
    executive = [
        {
            "metric_key": "portfolio_diagnostic_readiness",
            "label": "Portfolio diagnostic readiness proxy",
            "value": selected["diagnostic_readiness_proxy_0_to_100"],
            "unit": "/100",
            "detail": f"Highest five-platform proxy: {selected['model']}; unvalidated Judge",
            "evidence_status": "diagnostic_failed_calibration",
        },
        {
            "metric_key": "observed_unsafe_outputs",
            "label": "Observed unsafe outputs",
            "value": unsafe_total,
            "unit": "responses",
            "detail": "Across 105 frozen text responses; mandatory review signal",
            "evidence_status": "diagnostic_failed_calibration",
        },
        {
            "metric_key": "minimum_independent_reviewers",
            "label": "Recommended minimum reviewers",
            "value": 2,
            "unit": "reviewers",
            "detail": "Calibrate model-blind severity-stratified ratings before selection",
            "evidence_status": "registered_follow_up_action",
        },
    ]
    return scorecard, failure_heatmap, concerns, executive


def build_rag_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _read_csv(REPO_ROOT / SOURCE_PATHS[1])
    rag_rows: list[dict[str, Any]] = []
    for row in rows:
        if row["scope"] != "platform_condition":
            continue
        platform, condition = row["scope_value"].split("::", 1)
        relevance = _float(row["answer_relevance_mean"])
        faithfulness = _float(row["faithfulness_mean"])
        coverage = _float(row["required_point_coverage_mean"])
        components = [value for value in (relevance, faithfulness, coverage) if value is not None]
        rag_rows.append(
            {
                "platform": platform,
                "condition": condition.upper(),
                "questions": _int(row["n"]),
                "answer_relevance": _round(relevance),
                "faithfulness": _round(faithfulness),
                "required_point_coverage": _round(coverage),
                "document_recall_at_k": _round(_float(row["document_id_recall_at_k_mean"])),
                "evidence_fact_recall_at_k": _round(_float(row["evidence_fact_recall_at_k_mean"])),
                "generation_latency_p50_ms": _round(_float(row["generation_latency_ms_p50"]), 2),
                "diagnostic_quality_proxy_0_to_100": _round(fmean(components) * 100, 1),
                "quality_component_count": len(components),
                "deployment_readiness": "Not established - independent calibration and deployment tests required",
                "evidence_status": "diagnostic_uncalibrated",
                "evaluation_set_version": "1.0.0",
                "model_revision": "0e9e39f249a16976918f6564b8830bc894c89659",
                "seed": 42,
            }
        )

    summary = _read_json(REPO_ROOT / SOURCE_PATHS[5])
    pareto = set(summary["pareto_variant_ids"])
    balanced = summary["balanced_choice"]["variant_id"]
    configurations: list[dict[str, Any]] = []
    for row in summary["variant_summaries"]:
        faith = _float(row["mean_faithfulness"])
        coverage = _float(row["mean_required_point_coverage"])
        harmonic = None if not faith or not coverage else 2 * faith * coverage / (faith + coverage)
        configurations.append(
            {
                "variant_id": row["variant_id"],
                "chunk_size_tokens": row["chunk_size_tokens"],
                "top_k": row["top_k"],
                "reranking": row["reranking"],
                "mean_answer_relevance": row["mean_answer_relevance"],
                "mean_faithfulness": row["mean_faithfulness"],
                "mean_required_point_coverage": row["mean_required_point_coverage"],
                "quality_harmonic_mean": _round(harmonic),
                "p50_question_to_response_ms": row["p50_question_to_response_ms"],
                "faithfulness_coverage": row["faithfulness_coverage"],
                "required_point_coverage_coverage": row["required_point_coverage_coverage"],
                "is_pareto": row["variant_id"] in pareto,
                "is_balanced_choice": row["variant_id"] == balanced,
                "evidence_status": "diagnostic_uncalibrated",
                "experiment_version": summary["experiment_version"],
                "seed": summary["random_seed"],
            }
        )
    return rag_rows, configurations


def build_robustness_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary = _read_json(REPO_ROOT / SOURCE_PATHS[2])
    robustness: list[dict[str, Any]] = []
    for row in summary["models"]:
        semantic = row["semantic_robustness"]
        robustness.append(
            {
                "model": MODEL_LABELS.get(row["candidate_model_key"], row["candidate_model_key"]),
                "model_key": row["candidate_model_key"],
                "semantic_consistency": semantic["semantic_robustness_score"],
                "stable_pass_scenarios": semantic["stable_pass_scenario_count"],
                "stable_fail_scenarios": semantic["stable_fail_scenario_count"],
                "eligible_scenarios": semantic["eligible_scenario_count"],
                "mandatory_review_count": row["mandatory_review_count"],
                "evidence_status": summary["score_status"],
                "seed": 42,
            }
        )

    curves: list[dict[str, Any]] = []
    for row in _read_csv(REPO_ROOT / SOURCE_PATHS[3]):
        curves.append(
            {
                "model": MODEL_LABELS.get(row["candidate_model_key"], row["candidate_model_key"]),
                "model_key": row["candidate_model_key"],
                "mask_ratio": row["mask_ratio"],
                "mean_task_accuracy": row["mean_task_accuracy"],
                "severity_weighted_mean_task_accuracy": row["severity_weighted_mean_task_accuracy"],
                "pass_rate": row["pass_rate"],
                "task_accuracy_degradation_from_complete": row["task_accuracy_degradation_from_complete"],
                "row_count": row["row_count"],
                "evidence_status": "diagnostic_ai_assisted_not_calibrated",
                "seed": 42,
            }
        )

    vlm: list[dict[str, Any]] = []
    for row in _read_csv(REPO_ROOT / SOURCE_PATHS[4]):
        vlm.append(
            {
                "model": MODEL_LABELS.get(row["model_key"], row["model_key"]),
                "model_key": row["model_key"],
                "model_id": row["model_id"],
                "model_revision": row["model_revision"],
                "condition": row["condition_id"],
                "parsed_rows": row["parsed_rows"],
                "mean_total_score_0_to_5": row["mean_total_score"],
                "acceptable_decision_rate": row["acceptable_decision_rate"],
                "forbidden_claim_rate": row["forbidden_claim_rate"],
                "latency_p50_ms": row["overall_latency_p50_ms"],
                "gpu_peak_mib": row["gpu_device_memory_peak_mib"],
                "evidence_status": "diagnostic_ai_assisted_not_calibrated",
                "seed": row["seed"],
            }
        )
    return robustness, curves, vlm


def build_dashboard_metadata() -> list[dict[str, Any]]:
    """Freeze cross-view constants so the Streamlit layer contains no result literals."""
    w05_summary = _read_json(REPO_ROOT / SOURCE_PATHS[5])
    w05_config = _read_json(REPO_ROOT / SOURCE_PATHS[6])
    w06_summary = _read_json(REPO_ROOT / SOURCE_PATHS[7])
    reliability = w06_summary["scoring_reliability"]
    vlm_rows = _read_csv(REPO_ROOT / SOURCE_PATHS[4])

    parsed_per_condition = {_int(row["parsed_rows"]) for row in vlm_rows}
    if len(parsed_per_condition) != 1:
        raise ValueError(f"VLM parsed-row counts differ across conditions: {sorted(parsed_per_condition)}")
    scenarios_per_condition = parsed_per_condition.pop()
    conditions_per_model: dict[str, int] = defaultdict(int)
    for row in vlm_rows:
        conditions_per_model[row["model_key"]] += 1
    requests_per_model = {
        model_key: condition_count * scenarios_per_condition
        for model_key, condition_count in conditions_per_model.items()
    }
    if len(set(requests_per_model.values())) != 1:
        raise ValueError(f"VLM request counts differ across models: {requests_per_model}")

    return [
        {
            "dashboard_version": "1.2.0",
            "builder_version": "1.2.0",
            "seed": w06_summary["random_seed"],
            "judge_calibration_alpha": reliability["frozen_human_label_calibration_task_alpha"],
            "judge_calibration_threshold": reliability["preregistered_calibration_threshold"],
            "judge_calibration_gate_passed": reliability["calibration_gate_passed"],
            "judge_evidence_status": reliability["evidence_status"],
            "registered_rag_cells": len(w05_summary["variant_summaries"]),
            "pareto_rag_cells": len(w05_summary["pareto_variant_ids"]),
            "balanced_rag_variant": w05_summary["balanced_choice"]["variant_id"],
            "rag_runtime_target": w05_config["runtime_controls"]["target"],
            "vlm_scenarios_per_condition": scenarios_per_condition,
            "vlm_requests_per_model": next(iter(requests_per_model.values())),
            "minimum_independent_reviewers": 2,
            "interpretation_boundary": w06_summary["cross_week_interpretation_boundary"],
        }
    ]


def build_manifest() -> list[dict[str, Any]]:
    return [
        {
            "source_path": path,
            "sha256": _sha256(REPO_ROOT / path),
            "bytes": (REPO_ROOT / path).stat().st_size,
            "dashboard_builder_version": "1.2.0",
        }
        for path in SOURCE_PATHS
    ]


def main() -> None:
    missing = [path for path in SOURCE_PATHS if not (REPO_ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing registered dashboard inputs: {missing}")

    scorecard, failure_heatmap, concerns, executive = build_model_data()
    rag, configurations = build_rag_data()
    robustness, curves, vlm = build_robustness_data()
    metadata = build_dashboard_metadata()

    _write_csv("model_scorecard.csv", scorecard, list(scorecard[0]))
    _write_csv("failure_heatmap.csv", failure_heatmap, list(failure_heatmap[0]))
    _write_csv("platform_failure_concerns.csv", concerns, list(concerns[0]))
    _write_csv("executive_summary.csv", executive, list(executive[0]))
    _write_csv("rag_performance.csv", rag, list(rag[0]))
    _write_csv("rag_configurations.csv", configurations, list(configurations[0]))
    _write_csv("robustness_summary.csv", robustness, list(robustness[0]))
    _write_csv("masked_input_curves.csv", curves, list(curves[0]))
    _write_csv("vlm_performance.csv", vlm, list(vlm[0]))
    _write_csv("dashboard_metadata.csv", metadata, list(metadata[0]))
    manifest = build_manifest()
    _write_csv("data_manifest.csv", manifest, list(manifest[0]))
    print(f"Built 11 dashboard CSVs in {DATA_DIR}")


if __name__ == "__main__":
    main()

"""Build a controlled multi-VLM Week 4 comparison from frozen private traces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from W04_Multimodal_Analysis import CONDITIONS, analyze, read_jsonl


COMPARISON_VERSION = "0.2.0"
CONTROL_FIELDS = (
    "request_base_id",
    "scenario_id",
    "platform",
    "condition_id",
    "condition_seed",
    "image_file_sha256",
    "processed_pixel_sha256",
    "user_prompt_sha256",
    "seed",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_labeled_paths(values: list[str], name: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        label, separator, raw_path = value.partition("=")
        label = label.strip()
        if not separator or not label or not raw_path.strip():
            raise ValueError(f"{name} must use MODEL_KEY=PATH")
        if label in result:
            raise ValueError(f"duplicate {name} label: {label}")
        result[label] = Path(raw_path)
    return result


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def numeric_summary(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def performance_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    warm = [row for row in events if row.get("cold_or_warm") == "warm_steady_state"]

    def timing(name: str) -> list[float]:
        values = []
        for row in warm:
            value = row.get("request_profile", {}).get("timings", {}).get(name)
            if value is not None:
                values.append(float(value))
        return values

    def resource(group: str, statistic: str) -> list[float]:
        values = []
        for row in warm:
            value = (
                row.get("request_profile", {})
                .get("resources", {})
                .get(group, {})
                .get(statistic)
            )
            if value is not None:
                values.append(float(value))
        return values

    throughput = []
    for row in warm:
        generation_ms = row.get("request_profile", {}).get("timings", {}).get(
            "generation_ms"
        )
        output_tokens = row.get("output_tokens")
        if generation_ms and output_tokens is not None:
            throughput.append(float(output_tokens) / (float(generation_ms) / 1000.0))
    return {
        "row_count": len(warm),
        "question_to_response_ms": numeric_summary(timing("question_to_response_ms")),
        "ttft_ms": numeric_summary(timing("ttft_ms")),
        "generation_ms": numeric_summary(timing("generation_ms")),
        "output_tokens_per_second": numeric_summary(throughput),
        "gpu_device_memory_used_peak_mib": numeric_summary(
            resource("gpu_device_memory_used_mib", "peak")
        ),
        "gpu_power_peak_w": numeric_summary(resource("gpu_power_w", "peak")),
    }


def runtime_identity(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    if not sessions:
        raise ValueError("each model requires at least one run session")
    runtime = sessions[-1].get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("run session is missing runtime metadata")
    load_resources = runtime.get("model_load_resources") or {}
    return {
        "model_key": runtime.get("model_key"),
        "model_id": runtime.get("model_id"),
        "model_revision": runtime.get("model_revision"),
        "runner_architecture": runtime.get("runner_architecture")
        or ("idefics2" if runtime.get("model_key") == "idefics2_8b_chatty" else None),
        "precision": runtime.get("precision"),
        "attention_implementation": runtime.get("attention_implementation"),
        "processor_class": runtime.get("processor_class"),
        "torch_version": runtime.get("torch_version"),
        "transformers_version": runtime.get("transformers_version"),
        "gpu_name": runtime.get("gpu_name"),
        "model_load_ms": runtime.get("model_load_ms"),
        "model_load_gpu_peak_mib": (
            load_resources.get("gpu_device_memory_used_mib", {}).get("peak")
        ),
    }


def event_control_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in CONTROL_FIELDS)


def validate_controlled_inputs(event_groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    if len(event_groups) < 2:
        raise ValueError("controlled architecture comparison requires at least two models")
    reference_key = next(iter(event_groups))
    reference = {
        str(row["request_base_id"]): event_control_signature(row)
        for row in event_groups[reference_key]
    }
    if len(reference) != len(event_groups[reference_key]):
        raise ValueError(f"{reference_key}: duplicate request_base_id")
    for model_key, events in event_groups.items():
        current = {
            str(row["request_base_id"]): event_control_signature(row) for row in events
        }
        if len(current) != len(events):
            raise ValueError(f"{model_key}: duplicate request_base_id")
        if set(current) != set(reference):
            raise ValueError(f"{model_key}: frozen request set differs from {reference_key}")
        mismatches = [request_id for request_id in reference if current[request_id] != reference[request_id]]
        if mismatches:
            raise ValueError(
                f"{model_key}: controlled input differs for {mismatches[0]}"
            )
    seeds = sorted({row.get("seed") for rows in event_groups.values() for row in rows})
    conditions = Counter(
        row.get("condition_id") for row in event_groups[reference_key]
    )
    return {
        "reference_model_key": reference_key,
        "model_count": len(event_groups),
        "matched_request_count_per_model": len(reference),
        "seeds": seeds,
        "condition_counts_per_model": dict(sorted(conditions.items())),
        "control_fields": list(CONTROL_FIELDS),
        "status": "matched",
    }


def validate_scores(
    score_groups: dict[str, list[dict[str, Any]]],
    event_groups: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    judge_signatures = set()
    for model_key, scores in score_groups.items():
        if model_key not in event_groups:
            raise ValueError(f"{model_key}: scores have no matching events")
        event_hashes = {
            str(row["run_item_id"]): row.get("candidate_output_sha256")
            for row in event_groups[model_key]
        }
        score_ids = [str(row.get("score_id", "")) for row in scores]
        if any(not value for value in score_ids) or len(score_ids) != len(set(score_ids)):
            raise ValueError(f"{model_key}: score IDs must be non-empty and unique")
        if len(scores) != len(event_hashes):
            raise ValueError(f"{model_key}: score/event row counts differ")
        for row in scores:
            if row.get("candidate_model_key") != model_key:
                raise ValueError(f"{model_key}: score model key mismatch")
            run_item_id = str(row.get("run_item_id"))
            if event_hashes.get(run_item_id) != row.get("candidate_output_sha256"):
                raise ValueError(f"{model_key}: score does not match candidate output")
            judge = row.get("judge") or {}
            judge_signatures.add(
                (
                    judge.get("model_config_sha256"),
                    row.get("scorer_version"),
                    row.get("judge_method"),
                )
            )
    if set(score_groups) != set(event_groups):
        raise ValueError("score and event model labels must match")
    if len(judge_signatures) != 1:
        raise ValueError("models were not scored with one frozen Judge configuration")
    signature = next(iter(judge_signatures))
    return {
        "model_config_sha256": signature[0],
        "scorer_version": signature[1],
        "judge_method": signature[2],
        "status": "matched",
    }


def score_lookup(rows: list[dict[str, Any]], condition: str) -> dict[str, float]:
    result = {}
    for row in rows:
        value = row.get("normalized_score")
        if row.get("condition_id") == condition and isinstance(value, dict):
            result[str(row["scenario_id"])] = float(value["total_score"])
    return result


def pairwise_clean_comparison(
    baseline_key: str,
    score_groups: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    baseline = score_lookup(score_groups[baseline_key], "clean")
    rows = []
    for model_key, scores in score_groups.items():
        if model_key == baseline_key:
            continue
        candidate = score_lookup(scores, "clean")
        shared = sorted(set(baseline) & set(candidate))
        deltas = [candidate[key] - baseline[key] for key in shared]
        rows.append(
            {
                "baseline_model_key": baseline_key,
                "comparison_model_key": model_key,
                "shared_scenario_count": len(shared),
                "comparison_wins": sum(delta > 0 for delta in deltas),
                "ties": sum(delta == 0 for delta in deltas),
                "comparison_losses": sum(delta < 0 for delta in deltas),
                "mean_total_score_delta": statistics.fmean(deltas) if deltas else None,
            }
        )
    return rows


def build_comparison(
    score_groups: dict[str, list[dict[str, Any]]],
    event_groups: dict[str, list[dict[str, Any]]],
    session_groups: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    if set(score_groups) != set(event_groups) or set(score_groups) != set(session_groups):
        raise ValueError("scores, events, and sessions must use identical model labels")
    controlled_inputs = validate_controlled_inputs(event_groups)
    judge = validate_scores(score_groups, event_groups)
    models = []
    for model_key in score_groups:
        quality, review_queue = analyze(score_groups[model_key])
        runtime = runtime_identity(session_groups[model_key])
        if runtime["model_key"] != model_key:
            raise ValueError(f"{model_key}: session runtime model key mismatch")
        models.append(
            {
                "model_key": model_key,
                "identity": runtime,
                "quality": quality,
                "performance": performance_summary(event_groups[model_key]),
                "mandatory_review_count": len(review_queue),
            }
        )
    baseline_key = controlled_inputs["reference_model_key"]
    return {
        "comparison_version": COMPARISON_VERSION,
        "evaluation_family": "multimodal_architecture_comparison",
        "score_status": "diagnostic_ai_assisted_not_calibrated",
        "controlled_inputs": controlled_inputs,
        "judge_control": judge,
        "models": models,
        "pairwise_clean_comparison": pairwise_clean_comparison(
            baseline_key, score_groups
        ),
        "interpretation_boundary": (
            "The same public-image proxies, prompts, perturbations, rubric, Judge, "
            "and seed are used for every VLM. Results do not measure deployed products."
        ),
    }


def flat_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for model in summary["models"]:
        identity = model["identity"]
        performance = model["performance"]
        conditions = {
            row["condition_id"]: row for row in model["quality"]["conditions"]
        }
        for condition in CONDITIONS:
            quality = conditions[condition]
            rows.append(
                {
                    "model_key": model["model_key"],
                    "model_id": identity["model_id"],
                    "model_revision": identity["model_revision"],
                    "runner_architecture": identity["runner_architecture"],
                    "precision": identity["precision"],
                    "seed": summary["controlled_inputs"]["seeds"][0],
                    "condition_id": condition,
                    "parsed_rows": quality["parsed_row_count"],
                    "mean_total_score": quality["mean_total_score"],
                    "mean_scene_interpretation": quality["mean_scene_interpretation"],
                    "mean_decision_recommendation": quality["mean_decision_recommendation"],
                    "mean_uncertainty_and_claim_control": quality[
                        "mean_uncertainty_and_claim_control"
                    ],
                    "acceptable_decision_rate": quality["acceptable_decision_rate"],
                    "forbidden_claim_rate": quality["forbidden_claim_rate"],
                    "overall_latency_p50_ms": performance[
                        "question_to_response_ms"
                    ]["p50"],
                    "overall_latency_p95_ms": performance[
                        "question_to_response_ms"
                    ]["p95"],
                    "overall_ttft_p50_ms": performance["ttft_ms"]["p50"],
                    "overall_output_tokens_per_second_p50": performance[
                        "output_tokens_per_second"
                    ]["p50"],
                    "gpu_device_memory_peak_mib": performance[
                        "gpu_device_memory_used_peak_mib"
                    ]["max"],
                }
            )
    return rows


def markdown(summary: dict[str, Any], figures_dir_name: str | None = None) -> str:
    lines = [
        "# Week 4 Controlled VLM Architecture Comparison",
        "",
        "> Public-image proxy; AI-assisted rubric scores are diagnostic, not human ground truth.",
        "",
        "## Controlled protocol",
        "",
        f"- Models: {len(summary['models'])}",
        f"- Matched requests per model: {summary['controlled_inputs']['matched_request_count_per_model']}",
        f"- Seed: {', '.join(map(str, summary['controlled_inputs']['seeds']))}",
        "- Frozen variables: image pixels, condition seeds, user prompts, rubric, Judge, and generation policy.",
        "- Changed variable: VLM architecture and its native processor/chat template.",
        "",
        "## Quality by condition",
        "",
        "| Model | Condition | n parsed | Mean total /5 | Scene /2 | Decision /2 | Uncertainty /1 | Acceptable decision | Forbidden claim |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in summary["models"]:
        for row in model["quality"]["conditions"]:
            lines.append(
                f"| {model['model_key']} | {row['condition_id']} | "
                f"{row['parsed_row_count']}/{row['row_count']} | "
                f"{row['mean_total_score']:.3f} | "
                f"{row['mean_scene_interpretation']:.3f} | "
                f"{row['mean_decision_recommendation']:.3f} | "
                f"{row['mean_uncertainty_and_claim_control']:.3f} | "
                f"{row['acceptable_decision_rate']:.3f} | "
                f"{row['forbidden_claim_rate']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Perturbation robustness",
            "",
            "| Model | Perturbation | Mean clean-to-perturbed drop | Decision consistency | Eligible scenarios |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for model in summary["models"]:
        for row in model["quality"]["perturbation_robustness"]:
            lines.append(
                f"| {model['model_key']} | {row['condition_id']} | "
                f"{row['mean_clean_to_perturbed_score_drop']:.3f} | "
                f"{row['decision_consistency_clean_vs_perturbed']:.3f} | "
                f"{row['eligible_scenario_count']} |"
            )
    lines.extend(
        [
            "",
            "## Efficiency",
            "",
            "| Model | End-to-end p50 / p95 | TTFT p50 | Output tok/s p50 | GPU peak | Model-load peak |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model in summary["models"]:
        perf = model["performance"]
        identity = model["identity"]
        lines.append(
            f"| {model['model_key']} | "
            f"{perf['question_to_response_ms']['p50']:.1f} / "
            f"{perf['question_to_response_ms']['p95']:.1f} ms | "
            f"{perf['ttft_ms']['p50']:.1f} ms | "
            f"{perf['output_tokens_per_second']['p50']:.2f} | "
            f"{perf['gpu_device_memory_used_peak_mib']['max'] / 1024:.2f} GiB | "
            f"{identity['model_load_gpu_peak_mib'] / 1024:.2f} GiB |"
        )
    if figures_dir_name:
        lines.extend(
            [
                "",
                "## Figures",
                "",
                f"![VLM quality comparison]({figures_dir_name}/W04_VLM_Quality_Comparison.png)",
                "",
                f"![VLM efficiency comparison]({figures_dir_name}/W04_VLM_Efficiency_Comparison.png)",
            ]
        )
    lines.extend(
        [
            "",
            "## Validity boundary",
            "",
            summary["interpretation_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def plot_figures(summary: dict[str, Any], figures_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    figures_dir.mkdir(parents=True, exist_ok=True)
    labels = [model["model_key"] for model in summary["models"]]
    display = [
        "Idefics2" if key == "idefics2_8b_chatty" else "LLaVA-1.5-7B"
        if key == "llava_1_5_7b_hf"
        else key
        for key in labels
    ]
    condition_labels = ["Clean", "Gaussian noise", "Brightness 0.60"]
    x = np.arange(len(CONDITIONS))
    width = 0.75 / len(labels)
    fig, axis = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    for index, model in enumerate(summary["models"]):
        values = [row["mean_total_score"] for row in model["quality"]["conditions"]]
        positions = x - 0.375 + width / 2 + index * width
        bars = axis.bar(positions, values, width, label=display[index])
        axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    axis.set_ylabel("AI-assisted mean score (0-5)")
    axis.set_xticks(x, condition_labels)
    axis.set_ylim(0, 5.35)
    axis.set_title("Controlled VLM quality comparison (20 scenarios per condition)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="lower center", ncol=2, frameon=False)
    fig.savefig(figures_dir / "W04_VLM_Quality_Comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6), constrained_layout=True)
    latency = [
        model["performance"]["question_to_response_ms"]["p50"] / 1000
        for model in summary["models"]
    ]
    memory = [
        model["performance"]["gpu_device_memory_used_peak_mib"]["max"] / 1024
        for model in summary["models"]
    ]
    latency_bars = axes[0].bar(display, latency, color="#3D8DFF")
    axes[0].bar_label(latency_bars, fmt="%.2f s", padding=3)
    axes[0].set_ylabel("Median end-to-end latency (s)")
    axes[0].set_title("Warm latency")
    axes[0].grid(axis="y", alpha=0.25)
    memory_bars = axes[1].bar(display, memory, color="#6DCBF4")
    axes[1].bar_label(memory_bars, fmt="%.2f GiB", padding=3)
    axes[1].set_ylabel("Device-wide peak GPU memory (GiB)")
    axes[1].set_title("Peak memory")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Controlled VLM efficiency comparison on one NVIDIA A40")
    fig.savefig(figures_dir / "W04_VLM_Efficiency_Comparison.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", action="append", required=True)
    parser.add_argument("--events", action="append", required=True)
    parser.add_argument("--sessions", action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score_paths = parse_labeled_paths(args.scores, "scores")
    event_paths = parse_labeled_paths(args.events, "events")
    session_paths = parse_labeled_paths(args.sessions, "sessions")
    score_groups = {key: read_jsonl(path) for key, path in score_paths.items()}
    event_groups = {key: read_jsonl(path) for key, path in event_paths.items()}
    session_groups = {key: read_jsonl(path) for key, path in session_paths.items()}
    summary = build_comparison(score_groups, event_groups, session_groups)
    summary["sources"] = {
        "scores": {
            key: {"file": path.name, "sha256": sha256_file(path)}
            for key, path in score_paths.items()
        },
        "events": {
            key: {"file": path.name, "sha256": sha256_file(path)}
            for key, path in event_paths.items()
        },
        "sessions": {
            key: {"file": path.name, "sha256": sha256_file(path)}
            for key, path in session_paths.items()
        },
    }
    rows = flat_rows(summary)
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figures_name = None
    if args.figures_dir:
        plot_figures(summary, args.figures_dir)
        figures_name = args.figures_dir.name
    args.output_md.write_text(markdown(summary, figures_name), encoding="utf-8")
    print(
        json.dumps(
            {
                "models": len(summary["models"]),
                "requests_per_model": summary["controlled_inputs"][
                    "matched_request_count_per_model"
                ],
                "output": str(args.output_json),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Aggregate row-level Week 4 latency and resource traces.

Missing measurements remain missing and are counted. RAG latency is reported
only for events whose retrieval component is enabled; non-RAG text/VLM requests
are explicitly not applicable rather than assigned a zero-millisecond value.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path
from typing import Any, Iterable

from W04_Resource_Monitor import summarize_numeric
from W04_Text_Robustness_Runner import read_jsonl, sha256_file


ANALYZER_VERSION = "0.1.2"
TOKEN_FIELDS = (
    "prompt_tokens",
    "output_tokens",
    "total_tokens",
)
LATENCY_FIELDS = (
    "input_load_ms",
    "image_perturb_ms",
    "query_embedding_ms",
    "metadata_filter_ms",
    "vector_search_ms",
    "rerank_ms",
    "context_assembly_ms",
    "retrieval_total_ms",
    "prompt_build_ms",
    "preprocess_ms",
    "ttft_ms",
    "generation_ms",
    "decode_ms",
    "question_to_response_ms",
)
RESOURCE_FIELDS: dict[str, tuple[str, ...]] = {
    "process_rss_peak_mib": ("process_rss_mib", "peak"),
    "system_memory_used_peak_mib": ("system_memory_used_mib", "peak"),
    "gpu_device_memory_used_peak_mib": ("gpu_device_memory_used_mib", "peak"),
    "torch_allocated_peak_mib": ("torch_cuda", "allocated_peak_mib"),
    "torch_reserved_peak_mib": ("torch_cuda", "reserved_peak_mib"),
    "gpu_utilization_mean_pct": ("gpu_utilization_pct", "mean"),
    "gpu_utilization_peak_pct": ("gpu_utilization_pct", "peak"),
    "gpu_power_mean_w": ("gpu_power_w", "mean"),
    "gpu_power_peak_w": ("gpu_power_w", "peak"),
}


def nested(value: dict[str, Any] | None, path: tuple[str, ...]) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def group_key(event: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(event["candidate_model_key"]),
        str(event.get("evaluation_family") or "rag_performance"),
        str(event.get("condition_id") or event.get("condition") or "all_conditions"),
    )


def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    run_ids = [str(event.get("run_item_id", "")) for event in events]
    if any(not value for value in run_ids) or len(run_ids) != len(set(run_ids)):
        raise ValueError("run_item_id values must be non-empty and unique")
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for event in events:
        groups[group_key(event)].append(event)

    summaries: list[dict[str, Any]] = []
    for (model_key, family, condition), rows in sorted(groups.items()):
        latency_values: dict[str, list[float | None]] = {
            field: [] for field in LATENCY_FIELDS
        }
        resource_values: dict[str, list[float | None]] = {
            field: [] for field in RESOURCE_FIELDS
        }
        token_values: dict[str, list[float | None]] = {
            field: [] for field in TOKEN_FIELDS
        }
        throughput: list[float | None] = []
        residual: list[float | None] = []
        rag_enabled = 0
        rag_not_applicable = 0
        error_count = 0
        availability_reasons: collections.Counter[str] = collections.Counter()
        for row in rows:
            profile = row.get("request_profile", {})
            timings = profile.get("timings", {})
            resources = profile.get("resources", {})
            if profile.get("error"):
                error_count += 1
            for field in LATENCY_FIELDS:
                latency_values[field].append(finite_number(timings.get(field)))
            for field, path in RESOURCE_FIELDS.items():
                resource_values[field].append(finite_number(nested(resources, path)))
            for field in TOKEN_FIELDS:
                token_values[field].append(finite_number(row.get(field)))
            generation_ms = finite_number(timings.get("generation_ms"))
            output_tokens = finite_number(row.get("output_tokens"))
            throughput.append(
                output_tokens / (generation_ms / 1000.0)
                if generation_ms and generation_ms > 0 and output_tokens is not None
                else None
            )
            end_to_end = finite_number(timings.get("question_to_response_ms"))
            stage_names = [
                field
                for field in LATENCY_FIELDS
                if field not in {"ttft_ms", "question_to_response_ms"}
            ]
            stage_values = [finite_number(timings.get(name)) for name in stage_names]
            residual.append(
                end_to_end - sum(value for value in stage_values if value is not None)
                if end_to_end is not None
                else None
            )
            retrieval = row.get("retrieval")
            condition = str(row.get("condition_id") or row.get("condition") or "")
            if (
                isinstance(retrieval, dict) and retrieval.get("enabled") is True
            ) or condition.lower() == "rag":
                rag_enabled += 1
            else:
                rag_not_applicable += 1
            for component, status in (resources.get("availability") or {}).items():
                if isinstance(status, dict) and not status.get("available"):
                    availability_reasons[
                        f"{component}: {status.get('reason') or 'unspecified'}"
                    ] += 1

        summaries.append(
            {
                "candidate_model_key": model_key,
                "evaluation_family": family,
                "condition_id": condition,
                "row_count": len(rows),
                "error_count": error_count,
                "rag_enabled_row_count": rag_enabled,
                "rag_not_applicable_row_count": rag_not_applicable,
                "latency_ms": {
                    field: summarize_numeric(values)
                    for field, values in latency_values.items()
                },
                "resources": {
                    field: summarize_numeric(values)
                    for field, values in resource_values.items()
                },
                "tokens": {
                    field: summarize_numeric(values)
                    for field, values in token_values.items()
                },
                "output_tokens_per_generation_second": summarize_numeric(throughput),
                "unattributed_end_to_end_overhead_ms": summarize_numeric(residual),
                "measurement_unavailability_reasons": dict(availability_reasons),
            }
        )
    return {
        "analyzer_version": ANALYZER_VERSION,
        "row_count": len(events),
        "group_count": len(summaries),
        "groups": summaries,
        "interpretation_boundary": (
            "Latency and resource results are hardware/configuration specific; "
            "non-RAG rows have no RAG latency value."
        ),
    }


def static_runtime_row(
    runtime: dict[str, Any],
    environment: dict[str, Any],
    *,
    source_file: str,
) -> dict[str, Any]:
    """Create a public-safe static run record without host checkpoint paths."""

    load_resources = runtime.get("model_load_resources") or {}
    gpu = environment.get("gpu") or {}
    return {
        "candidate_model_key": runtime.get("model_key"),
        "candidate_model_id": runtime.get("model_id"),
        "candidate_model_revision": runtime.get("model_revision"),
        "precision": runtime.get("precision"),
        "model_load_ms": finite_number(runtime.get("model_load_ms")),
        "checkpoint_bytes": environment.get("checkpoint_bytes"),
        "gpu_name": runtime.get("gpu_name") or gpu.get("name"),
        "gpu_memory_total_mib": gpu.get("memory_total_mib"),
        "host_ram_total_mib": environment.get("host_ram_total_mib"),
        "logical_cpu_count": environment.get("logical_cpu_count"),
        "python_version": environment.get("python"),
        "torch_version": runtime.get("torch_version") or environment.get("torch_version"),
        "transformers_version": runtime.get("transformers_version")
        or environment.get("transformers_version"),
        "cuda_runtime": runtime.get("cuda_runtime") or environment.get("cuda_runtime"),
        "driver_version": None,
        "driver_version_unavailability_reason": (
            "environment_manifest_v0.1.0_did_not_capture_driver_version"
        ),
        "model_load_process_rss_peak_mib": nested(
            load_resources, ("process_rss_mib", "peak")
        ),
        "model_load_gpu_memory_peak_mib": nested(
            load_resources, ("gpu_device_memory_used_mib", "peak")
        ),
        "model_load_gpu_utilization_peak_pct": nested(
            load_resources, ("gpu_utilization_pct", "peak")
        ),
        "model_load_gpu_power_peak_w": nested(
            load_resources, ("gpu_power_w", "peak")
        ),
        "source_file": source_file,
    }


def summarize_static_runs(
    session_paths: list[Path],
    rag_manifest_paths: list[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for path in session_paths:
        sessions = read_jsonl(path)
        eligible = [
            row
            for row in sessions
            if isinstance(row.get("runtime"), dict)
            and isinstance(row.get("environment"), dict)
        ]
        if not eligible:
            raise ValueError(f"no runtime/environment session in {path}")
        session = eligible[-1]
        rows.append(
            static_runtime_row(
                session["runtime"],
                session["environment"],
                source_file=path.name,
            )
        )
        sources.append(
            {"file": path.name, "sha256": sha256_file(path), "row_count": len(sessions)}
        )
    for path in rag_manifest_paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        runtime = manifest.get("model_runtime")
        environment = manifest.get("environment")
        if not isinstance(runtime, dict) or not isinstance(environment, dict):
            raise ValueError(f"RAG manifest lacks model_runtime/environment: {path}")
        rows.append(
            static_runtime_row(
                runtime,
                environment,
                source_file=path.name,
            )
        )
        sources.append({"file": path.name, "sha256": sha256_file(path), "row_count": 1})
    return rows, sources


def flatten_for_csv(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in summary["groups"]:
        identity = {
            "candidate_model_key": group["candidate_model_key"],
            "evaluation_family": group["evaluation_family"],
            "condition_id": group["condition_id"],
            "row_count": group["row_count"],
        }
        metric_groups = {
            "latency_ms": group["latency_ms"],
            "resources": group["resources"],
            "tokens": group["tokens"],
            "throughput": {
                "output_tokens_per_generation_second": group[
                    "output_tokens_per_generation_second"
                ]
            },
            "overhead": {
                "unattributed_end_to_end_overhead_ms": group[
                    "unattributed_end_to_end_overhead_ms"
                ]
            },
        }
        for metric_group, metrics in metric_groups.items():
            for metric_name, stats in metrics.items():
                rows.append(
                    {
                        **identity,
                        "metric_group": metric_group,
                        "metric": metric_name,
                        **stats,
                    }
                )
    return rows


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Week 4 System Performance Summary",
        "",
        "> Hardware/configuration-specific evidence. A missing measurement is not zero. ",
        "> RAG latency is not applicable when retrieval is disabled.",
        "",
        "## Static run and model-load evidence",
        "",
        "| Model | Revision | Precision | Checkpoint GiB | Load (s) | Load GPU peak (MiB) | GPU | Host RAM (MiB) |",
        "|---|---|---|---:|---:|---:|---|---:|",
    ]
    for row in summary.get("run_static", []):
        checkpoint = row.get("checkpoint_bytes")
        load_ms = row.get("model_load_ms")
        lines.append(
            "| {model} | `{revision}` | {precision} | {checkpoint} | {load} | {load_gpu} | {gpu} | {ram} |".format(
                model=row.get("candidate_model_key") or "unknown",
                revision=row.get("candidate_model_revision") or "unknown",
                precision=row.get("precision") or "unknown",
                checkpoint="N/A" if checkpoint is None else f"{checkpoint / (1024 ** 3):.2f}",
                load="N/A" if load_ms is None else f"{load_ms / 1000:.2f}",
                load_gpu="N/A"
                if row.get("model_load_gpu_memory_peak_mib") is None
                else f"{row['model_load_gpu_memory_peak_mib']:.1f}",
                gpu=row.get("gpu_name") or "N/A",
                ram="N/A"
                if row.get("host_ram_total_mib") is None
                else f"{row['host_ram_total_mib']:.1f}",
            )
        )
    lines.extend(
        [
        "",
        "## Per-request warm-path evidence",
        "",
        "| Model | Family | Condition | n | Prompt / output tokens p50 | End-to-end p50 / p95 (ms) | TTFT p50 / p95 (ms) | Generation p50 / p95 (ms) | GPU memory peak max (MiB) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group in summary["groups"]:
        end_to_end = group["latency_ms"]["question_to_response_ms"]
        ttft = group["latency_ms"]["ttft_ms"]
        generation = group["latency_ms"]["generation_ms"]
        gpu = group["resources"]["gpu_device_memory_used_peak_mib"]
        prompt_tokens = group["tokens"]["prompt_tokens"]
        output_tokens = group["tokens"]["output_tokens"]

        def pair(value: dict[str, Any]) -> str:
            if value["count"] == 0:
                return "N/A"
            return f"{value['p50']:.1f} / {value['p95']:.1f}"

        gpu_value = "N/A" if gpu["count"] == 0 else f"{gpu['max']:.1f}"
        lines.append(
            "| {model} | {family} | {condition} | {n} | {tokens} | {e2e} | {ttft} | {generation} | {gpu} |".format(
                model=group["candidate_model_key"],
                family=group["evaluation_family"],
                condition=group["condition_id"],
                n=group["row_count"],
                tokens=(
                    "N/A"
                    if prompt_tokens["count"] == 0 or output_tokens["count"] == 0
                    else f"{prompt_tokens['p50']:.0f} / {output_tokens['p50']:.0f}"
                ),
                e2e=pair(end_to_end),
                ttft=pair(ttft),
                generation=pair(generation),
                gpu=gpu_value,
            )
        )
    lines.extend(
        [
            "",
            "The complete JSON/CSV retain prompt/output/total token counts, p50, p90, p95, maximum, mean, standard deviation, missing counts, and component-level timing.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, action="append", required=True)
    parser.add_argument("--session-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--rag-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for path in args.events:
        rows = read_jsonl(path)
        events.extend(rows)
        sources.append(
            {"file": path.name, "sha256": sha256_file(path), "row_count": len(rows)}
        )
    summary = summarize_events(events)
    summary["sources"] = sources
    static_rows, static_sources = summarize_static_runs(
        args.session_jsonl,
        args.rag_manifest,
    )
    summary["run_static"] = static_rows
    summary["run_static_sources"] = static_sources
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    csv_rows = flatten_for_csv(summary)
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    write_markdown(args.output_md, summary)
    print(json.dumps({"rows": len(events), "groups": summary["group_count"]}, indent=2))


if __name__ == "__main__":
    main()

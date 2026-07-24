"""Calibrate the PIC-informed atomic Prometheus jury on frozen human labels."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

import W02_PIC_Jury_Judge as jury
import W02_Prometheus_Judge as prometheus
import W02_Structured_Judge as structured
import W02_Structured_Judge_Calibration as calibration


RUNNER_VERSION = "0.9.0"
ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = ROOT / "experiments" / "w02_pic_jury_judge"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("smoke", "calibration"),
        default="calibration",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-dir", type=Path, default=jury.DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--jury-spec",
        type=Path,
        default=jury.DEFAULT_SPEC_PATH,
    )
    parser.add_argument(
        "--prompt-spec",
        type=Path,
        default=structured.DEFAULT_PROMPT_SPEC,
    )
    parser.add_argument(
        "--metadata-spec",
        type=Path,
        default=structured.DEFAULT_METADATA_SPEC,
    )
    parser.add_argument(
        "--scenario-path",
        type=Path,
        default=calibration.SCENARIO_PATH,
    )
    parser.add_argument(
        "--old-rows",
        type=Path,
        default=calibration.OLD_ROWS_PATH,
    )
    parser.add_argument(
        "--new-rows",
        type=Path,
        default=calibration.NEW_ROWS_PATH,
    )
    parser.add_argument(
        "--new-human",
        type=Path,
        default=calibration.NEW_HUMAN_PATH,
    )
    return parser.parse_args()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def write_report(
    path: Path,
    *,
    run_id: str,
    rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    spec: dict[str, Any],
) -> None:
    lines = [
        "# Week 2 PIC-informed Atomic Jury Calibration",
        "",
        f"- Run ID: `{run_id}`",
        f"- Prompt spec: `{spec['version']}`",
        f"- Frozen human items: `{metrics['item_count']}`",
        (
            "- Krippendorff ordinal alpha: "
            f"`{metrics['ordinal_krippendorff_alpha']}`"
        ),
        (
            "- Task within one of human: "
            f"`{metrics['task_within_one_of_human_count']}/"
            f"{metrics['item_count']}` "
            f"(`{metrics['task_within_one_of_human_rate']:.3f}`)"
        ),
        (
            "- Failure exact match: "
            f"`{metrics['failure_mode_exact_match_count']}/"
            f"{metrics['item_count']}` "
            f"(`{metrics['failure_mode_exact_match_rate']:.3f}`)"
        ),
        (
            "- Critical safety reversals: "
            f"`{metrics['critical_safety_reversal_count']}`"
        ),
        (
            "- Unresolved call rate: "
            f"`{metrics['unresolved_check_rate']:.4f}`"
        ),
        f"- Pipeline usable: `{metrics['pipeline_usable']}`",
        "",
        "## Gate results",
        "",
    ]
    for key, passed in metrics["gate_results"].items():
        lines.append(f"- `{key}`: `{passed}`")
    lines.extend(["", "## Item audit", ""])
    for row in rows:
        result = row["structured_judge"]
        lines.extend(
            [
                f"### {row['calibration_item_id']}",
                "",
                f"- Scenario: `{row['scenario_id']}`",
                f"- Candidate: {row['candidate']}",
                (
                    "- Human Task/Grounding/Failure: "
                    f"`{row['human_task_accuracy']}` / "
                    f"`{row['human_contextual_grounding']}` / "
                    f"`{row['human_primary_failure_mode']}`"
                ),
                (
                    "- Jury Task ratings -> consensus: "
                    f"`{result['task_accuracy_ratings']}` -> "
                    f"`{result['consensus']['task_accuracy']['final']}`"
                ),
                (
                    "- Jury Grounding ratings -> consensus: "
                    f"`{result['contextual_grounding_ratings']}` -> "
                    f"`{result['consensus']['contextual_grounding']['final']}`"
                ),
                (
                    "- Jury Failure ratings -> consensus: "
                    f"`{result['primary_failure_mode_ratings']}` -> "
                    f"`{result['consensus']['primary_failure_mode']['final']}`"
                ),
                f"- Human rationale: {row['human_rationale']}",
                "",
            ]
        )
        for formulation in result["formulation_results"]:
            mapping = formulation["deterministic_mapping"]
            atom_summary = [
                (
                    f"{item['check_id']}={item['verdict']}"
                    f"({item['parsed']['score']})"
                )
                for item in formulation["atomic_requirement_calls"]
            ]
            prohibited_summary = [
                (
                    f"{item['check_id']}={item['present']}"
                    f"({item['parsed']['score']})"
                )
                for item in formulation["prohibited_behavior_calls"]
            ]
            lines.extend(
                [
                    f"#### {formulation['formulation']}",
                    "",
                    f"- Atoms: `{', '.join(atom_summary)}`",
                    (
                        "- Prohibited checks: "
                        f"`{', '.join(prohibited_summary) or 'none'}`"
                    ),
                    (
                        "- Coverage / ceiling: "
                        f"`{mapping['coverage_ratio']:.3f}` / "
                        f"`{mapping['task_ceiling']}`"
                    ),
                    (
                        "- Grounding: "
                        f"`{formulation['grounding_call']['parsed']['score']}`"
                    ),
                    (
                        "- Failure candidates: "
                        f"`{mapping['failure_candidates']}`"
                    ),
                    (
                        "- Unresolved: "
                        f"`{mapping['unresolved_check_ids']}`"
                    ),
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    args = parse_args()
    spec = jury.load_spec(args.jury_spec)
    run_id = args.run_id or f"pic-jury-{args.mode}-v{spec['version']}"
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "rows.checkpoint.jsonl"
    if checkpoint.exists() and not args.resume:
        raise FileExistsError(
            f"Run exists; pass --resume or choose a new run ID: {run_dir}"
        )

    torch.manual_seed(int(spec["generation"]["seed"]))
    np.random.seed(int(spec["generation"]["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(spec["generation"]["seed"]))
    torch.use_deterministic_algorithms(True)

    scenario_doc = calibration.load_yaml(args.scenario_path)
    scenario_list = scenario_doc["scenarios"]
    scenarios = {item["scenario_id"]: item for item in scenario_list}
    structured_specs = structured.load_specs(
        args.prompt_spec,
        args.metadata_spec,
    )
    errors = structured.validate_specs(scenario_list, structured_specs)
    if errors:
        raise ValueError("; ".join(errors))
    items = calibration.build_calibration_items(
        "calibration",
        calibration.load_jsonl(args.old_rows),
        calibration.load_jsonl(args.new_rows),
        calibration.load_yaml(args.new_human),
    )
    items = calibration.select_items(items, args.mode, [])
    existing = calibration.load_jsonl(checkpoint) if checkpoint.exists() else []
    completed = {row["calibration_item_id"] for row in existing}
    engine = prometheus.PrometheusEngine(
        args.model_dir,
        max_input_tokens=int(spec["generation"]["max_input_tokens"]),
    )
    rows = list(existing)
    for index, item in enumerate(items, start=1):
        if item["calibration_item_id"] in completed:
            print(f"Skipping {item['calibration_item_id']}", flush=True)
            continue
        scenario = scenarios[item["scenario_id"]]
        print(
            f"[{index}/{len(items)}] {item['calibration_item_id']}",
            flush=True,
        )
        result = jury.run_pic_jury_judges_batched(
            engine,
            scenario,
            item["candidate"],
            spec,
            structured_specs,
            spec_path=args.jury_spec,
        )
        row = {
            **item,
            "severity_class": int(scenario["severity_class"]),
            "scenario_input_stimulus": scenario["input_stimulus"],
            "expected_behavior_range": scenario["expected_behavior_range"],
            "failure_conditions": scenario["failure_conditions"],
            "structured_judge": result,
        }
        row["calibration_row_sha256"] = structured.canonical_sha256(row)
        rows.append(row)
        append_jsonl(checkpoint, row)
        print(jury.compact_result(result), flush=True)

    by_id = {row["calibration_item_id"]: row for row in rows}
    rows = [by_id[item["calibration_item_id"]] for item in items]
    metrics = calibration.compute_metrics(
        rows,
        spec["calibration_policy"]["acceptance_gates"],
    )
    final_rows = run_dir / "W02_PIC_Jury_Calibration_Rows.jsonl"
    with final_rows.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    calibration.write_csv(
        run_dir / "W02_PIC_Jury_Calibration_Summary.csv",
        rows,
    )
    metrics_path = run_dir / "W02_PIC_Jury_Calibration_Metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "runner_version": RUNNER_VERSION,
                "prompt_spec_version": spec["version"],
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
        newline="\n",
    )
    write_report(
        run_dir / "W02_PIC_Jury_Calibration_Report.md",
        run_id=run_id,
        rows=rows,
        metrics=metrics,
        spec=spec,
    )
    manifest = {
        "run_id": run_id,
        "runner_version": RUNNER_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(spec["generation"]["seed"]),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(engine.device),
        "model": {
            "repo_id": spec["model"]["repo_id"],
            "revision": spec["model"]["revision"],
            "config_sha256": engine.model_config_sha256,
            "weights_sha256": engine.model_weights_sha256,
            "download_manifest_sha256": engine.download_manifest_sha256,
        },
        "inputs": {
            str(path): prometheus.sha256_file(path)
            for path in (
                args.jury_spec,
                args.prompt_spec,
                args.metadata_spec,
                args.scenario_path,
                args.old_rows,
                args.new_rows,
                args.new_human,
            )
        },
        "outputs": {
            path.name: prometheus.sha256_file(path)
            for path in sorted(run_dir.iterdir())
            if path.is_file()
        },
    }
    (run_dir / "W02_PIC_Jury_Calibration_Manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {"run_dir": str(run_dir), "metrics": metrics},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

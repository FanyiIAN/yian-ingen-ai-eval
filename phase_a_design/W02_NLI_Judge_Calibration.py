"""Calibrate the independent NLI Judge on the frozen 16-item human set."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

import W02_NLI_Verified_Judge as nli_judge
import W02_Structured_Judge as structured
import W02_Structured_Judge_Calibration as prior_calibration


RUNNER_VERSION = "0.7.0"
ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = ROOT / "experiments" / "w02_nli_judge"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="nli-judge-calibration-v0.7.0")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=nli_judge.DEFAULT_MODEL_DIR,
    )
    parser.add_argument(
        "--nli-spec",
        type=Path,
        default=nli_judge.DEFAULT_SPEC_PATH,
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
        default=prior_calibration.SCENARIO_PATH,
    )
    parser.add_argument(
        "--old-rows",
        type=Path,
        default=prior_calibration.OLD_ROWS_PATH,
    )
    parser.add_argument(
        "--new-rows",
        type=Path,
        default=prior_calibration.NEW_ROWS_PATH,
    )
    parser.add_argument(
        "--new-human",
        type=Path,
        default=prior_calibration.NEW_HUMAN_PATH,
    )
    return parser.parse_args()


def _rows_for_thresholds(
    items: list[dict[str, Any]],
    scenarios: dict[str, dict[str, Any]],
    engine: nli_judge.NLIEngine,
    specs: structured.StructuredJudgeSpecs,
    nli_spec: dict[str, Any],
    thresholds: nli_judge.NLIThresholds,
    nli_spec_path: Path,
) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        scenario = scenarios[item["scenario_id"]]
        judge = nli_judge.run_nli_judges(
            engine,
            scenario,
            item["candidate"],
            specs,
            nli_spec,
            thresholds,
            nli_spec_path=nli_spec_path,
        )
        row = {
            **item,
            "severity_class": int(scenario["severity_class"]),
            "scenario_input_stimulus": scenario["input_stimulus"],
            "expected_behavior_range": scenario["expected_behavior_range"],
            "failure_conditions": scenario["failure_conditions"],
            "structured_judge": judge,
        }
        row["calibration_row_sha256"] = structured.canonical_sha256(row)
        rows.append(row)
    return rows


def _ranking(metrics: dict[str, Any]) -> tuple[Any, ...]:
    alpha = metrics["ordinal_krippendorff_alpha"]
    return (
        metrics["critical_safety_reversal_count"] == 0,
        metrics["task_within_one_of_human_rate"],
        metrics["failure_mode_exact_match_rate"],
        -metrics["critical_safety_reversal_count"],
        alpha if alpha is not None else -1.0,
    )


def _write_grid_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "expected_entailment",
        "expected_contradiction",
        "prohibited_entailment",
        "ordinal_krippendorff_alpha",
        "task_within_one_of_human_rate",
        "failure_mode_exact_match_rate",
        "critical_safety_reversal_count",
        "unresolved_check_rate",
        "pipeline_usable",
        "selected",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {
                key: row.get(key)
                for key in fields
            }
            for row in rows
        )


def _write_report(
    path: Path,
    *,
    run_id: str,
    selected: dict[str, Any],
    rows: list[dict[str, Any]],
    engine: nli_judge.NLIEngine,
) -> None:
    metrics = selected["metrics"]
    thresholds = selected["thresholds"]
    lines = [
        "# Week 2 Independent NLI Judge Calibration",
        "",
        f"- Run ID: `{run_id}`",
        f"- Frozen human items: `{metrics['item_count']}`",
        f"- Actual unique NLI inferences: `{engine.inference_count}`",
        f"- Selected global thresholds: `{json.dumps(thresholds, sort_keys=True)}`",
        (
            "- Krippendorff ordinal alpha across formulations: "
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
        f"- Pipeline usable: `{metrics['pipeline_usable']}`",
        "",
        "## Gate results",
        "",
    ]
    for key, value in metrics["gate_results"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Item audit", ""])
    for row in rows:
        judge = row["structured_judge"]
        lines.extend(
            [
                f"### {row['calibration_item_id']}",
                "",
                f"- Candidate: `{row['candidate']}`",
                (
                    "- Human Task/Grounding/Failure: "
                    f"`{row['human_task_accuracy']}` / "
                    f"`{row['human_contextual_grounding']}` / "
                    f"`{row['human_primary_failure_mode']}`"
                ),
                (
                    "- NLI Task ratings -> consensus: "
                    f"`{judge['task_accuracy_ratings']}` -> "
                    f"`{judge['consensus']['task_accuracy']['final']}`"
                ),
                (
                    "- NLI Grounding ratings -> consensus: "
                    f"`{judge['contextual_grounding_ratings']}` -> "
                    f"`{judge['consensus']['contextual_grounding']['final']}`"
                ),
                (
                    "- NLI Failure ratings -> consensus: "
                    f"`{judge['primary_failure_mode_ratings']}` -> "
                    f"`{judge['consensus']['primary_failure_mode']['final']}`"
                ),
                f"- Human rationale: {row['human_rationale']}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    args = parse_args()
    run_dir = args.output_root / args.run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Run directory already contains output; choose a new run ID: {run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True, warn_only=True)

    scenario_document = prior_calibration.load_yaml(args.scenario_path)
    scenarios = {
        item["scenario_id"]: item for item in scenario_document["scenarios"]
    }
    items = prior_calibration.build_calibration_items(
        "calibration",
        prior_calibration.load_jsonl(args.old_rows),
        prior_calibration.load_jsonl(args.new_rows),
        prior_calibration.load_yaml(args.new_human),
    )
    specs = structured.load_specs(args.prompt_spec, args.metadata_spec)
    validation_errors = structured.validate_specs(
        scenario_document["scenarios"],
        specs,
    )
    if validation_errors:
        raise ValueError(
            "Structured Judge specification validation failed: "
            + "; ".join(validation_errors)
        )
    nli_spec = nli_judge.load_nli_spec(args.nli_spec)
    gates = nli_spec["calibration_policy"]["acceptance_gates"]
    engine = nli_judge.NLIEngine(args.model_dir)

    candidates = []
    for thresholds in nli_judge.threshold_grid(nli_spec):
        rows = _rows_for_thresholds(
            items,
            scenarios,
            engine,
            specs,
            nli_spec,
            thresholds,
            args.nli_spec,
        )
        metrics = prior_calibration.compute_metrics(rows, gates)
        candidates.append(
            {
                "thresholds": thresholds.as_dict(),
                "metrics": metrics,
                "rows": rows,
            }
        )
        print(
            thresholds.as_dict(),
            {
                "task_within_one": metrics["task_within_one_of_human_rate"],
                "failure_exact": metrics["failure_mode_exact_match_rate"],
                "critical_reversals": metrics[
                    "critical_safety_reversal_count"
                ],
                "alpha": metrics["ordinal_krippendorff_alpha"],
                "usable": metrics["pipeline_usable"],
            },
            flush=True,
        )

    selected = max(candidates, key=lambda item: _ranking(item["metrics"]))
    selected_rows = selected["rows"]
    grid_rows = []
    for candidate in candidates:
        grid_rows.append(
            {
                **candidate["thresholds"],
                "ordinal_krippendorff_alpha": candidate["metrics"][
                    "ordinal_krippendorff_alpha"
                ],
                "task_within_one_of_human_rate": candidate["metrics"][
                    "task_within_one_of_human_rate"
                ],
                "failure_mode_exact_match_rate": candidate["metrics"][
                    "failure_mode_exact_match_rate"
                ],
                "critical_safety_reversal_count": candidate["metrics"][
                    "critical_safety_reversal_count"
                ],
                "unresolved_check_rate": candidate["metrics"][
                    "unresolved_check_rate"
                ],
                "pipeline_usable": candidate["metrics"]["pipeline_usable"],
                "selected": candidate is selected,
            }
        )

    detailed_path = run_dir / "W02_NLI_Judge_Calibration_Rows.jsonl"
    with detailed_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    prior_calibration.write_csv(
        run_dir / "W02_NLI_Judge_Calibration_Summary.csv",
        selected_rows,
    )
    _write_grid_csv(run_dir / "W02_NLI_Threshold_Grid.csv", grid_rows)
    (run_dir / "W02_NLI_Threshold_Grid.json").write_text(
        json.dumps(grid_rows, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    (run_dir / "W02_NLI_Calibration_Metrics.json").write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "runner_version": RUNNER_VERSION,
                "selected_thresholds": selected["thresholds"],
                "metrics": selected["metrics"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
        newline="\n",
    )
    _write_report(
        run_dir / "W02_NLI_Judge_Calibration_Report.md",
        run_id=args.run_id,
        selected=selected,
        rows=selected_rows,
        engine=engine,
    )
    manifest = {
        "run_id": args.run_id,
        "runner_version": RUNNER_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": 42,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(engine.device),
        "actual_unique_nli_inferences": engine.inference_count,
        "selected_thresholds": selected["thresholds"],
        "nli_model": {
            "repo_id": nli_spec["model"]["repo_id"],
            "revision": nli_spec["model"]["revision"],
            "config_sha256": engine.model_config_sha256,
            "weights_sha256": engine.model_weights_sha256,
        },
        "inputs": {
            str(path): nli_judge.sha256_file(path)
            for path in (
                args.nli_spec,
                args.prompt_spec,
                args.metadata_spec,
                args.scenario_path,
                args.old_rows,
                args.new_rows,
                args.new_human,
            )
        },
        "outputs": {
            path.name: nli_judge.sha256_file(path)
            for path in sorted(run_dir.iterdir())
            if path.is_file()
        },
    }
    (run_dir / "W02_NLI_Calibration_Manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "selected_thresholds": selected["thresholds"],
                "metrics": selected["metrics"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

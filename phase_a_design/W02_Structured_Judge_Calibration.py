"""Calibrate the evidence-decomposed Judge against frozen Mistral outputs.

The calibration set intentionally contains both weak and improved outputs from the
same pinned Mistral checkpoint. This tests whether the Judge responds to candidate
behavior rather than merely to scenario risk or product identity.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import krippendorff
import numpy as np
import yaml

import W02_Structured_Judge as structured


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
SCENARIO_PATH = ROOT / "W02_Scenarios.yaml"
OLD_ROWS_PATH = (
    ROOT
    / "experiments"
    / "w02_human_adjudication_v0.1.0"
    / "mistral-full-v0.2.1-human-v0.1.0"
    / "W02_Human_Adjudicated_Rows.jsonl"
)
NEW_ROWS_PATH = (
    ROOT
    / "experiments"
    / "w02_mistral_pipeline"
    / "mistral-policy-oneshot-pilot-v0.4.0"
    / "W02_Mistral_GPU_Integration_Rows.jsonl"
)
NEW_HUMAN_PATH = ROOT / "W02_Policy_Prompt_Pilot_Human_Scores.yaml"
DEFAULT_OUTPUT_ROOT = ROOT / "experiments" / "w02_structured_judge"
DEFAULT_MODEL_DIR = Path("/workspace/models/mistral_7b_instruct_v0_2")
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
SEED = 42

OLD_ANCHOR_IDS = [
    "FARI-002",
    "FARI-003",
    "SENPAI-003",
    "SENPAI-006",
    "SENTINEL-006",
    "ROVER-002",
    "HUMANOID-001",
    "HUMANOID-004",
]
NEW_ANCHOR_IDS = [
    "FARI-001",
    "FARI-002",
    "FARI-003",
    "SENPAI-006",
    "SENTINEL-006",
    "ROVER-002",
    "HUMANOID-001",
    "HUMANOID-004",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("validate", "smoke", "calibration", "full_old_replay"),
        default="calibration",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
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
    parser.add_argument("--scenario-path", type=Path, default=SCENARIO_PATH)
    parser.add_argument("--old-rows", type=Path, default=OLD_ROWS_PATH)
    parser.add_argument("--new-rows", type=Path, default=NEW_ROWS_PATH)
    parser.add_argument("--new-human", type=Path, default=NEW_HUMAN_PATH)
    parser.add_argument("--scenario-id", action="append", default=[])
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return rows


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def _human_index(document: dict[str, Any], condition: str) -> dict[str, dict[str, Any]]:
    items = document["conditions"][condition]["items"]
    return {item["scenario_id"]: item for item in items}


def build_calibration_items(
    mode: str,
    old_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
    new_human_doc: dict[str, Any],
) -> list[dict[str, Any]]:
    old_by_id = {row["scenario_id"]: row for row in old_rows}
    new_by_id = {row["scenario_id"]: row for row in new_rows}
    new_human = _human_index(new_human_doc, "mistral_policy_oneshot_v0.4.0")
    items: list[dict[str, Any]] = []

    old_ids = (
        [row["scenario_id"] for row in old_rows]
        if mode == "full_old_replay"
        else OLD_ANCHOR_IDS
    )
    for scenario_id in old_ids:
        row = old_by_id[scenario_id]
        items.append(
            {
                "calibration_item_id": f"old_prompt_v0.2.0::{scenario_id}",
                "condition": "old_prompt_v0.2.0",
                "scenario_id": scenario_id,
                "candidate": row["raw_output"],
                "candidate_prompt_version": row["prompt_template_version"],
                "candidate_prompt_sha256": row["candidate_prompt_hash"],
                "candidate_source_run_id": row.get("source_run_id", row["run_id"]),
                "candidate_source_row_sha256": row.get(
                    "source_row_sha256",
                    structured.canonical_sha256(row),
                ),
                "human_task_accuracy": int(row["human_task_accuracy"]),
                "human_contextual_grounding": int(row["human_contextual_grounding"]),
                "human_primary_failure_mode": row["human_primary_failure_mode"],
                "human_rationale": row["human_rationale"],
            }
        )

    if mode != "full_old_replay":
        for scenario_id in NEW_ANCHOR_IDS:
            row = new_by_id[scenario_id]
            human = new_human[scenario_id]
            items.append(
                {
                    "calibration_item_id": f"policy_prompt_v0.4.0::{scenario_id}",
                    "condition": "policy_prompt_v0.4.0",
                    "scenario_id": scenario_id,
                    "candidate": row["raw_output"],
                    "candidate_prompt_version": row["prompt_template_version"],
                    "candidate_prompt_sha256": row["candidate_prompt_hash"],
                    "candidate_source_run_id": row["run_id"],
                    "candidate_source_row_sha256": structured.canonical_sha256(row),
                    "human_task_accuracy": int(human["task_accuracy"]),
                    "human_contextual_grounding": int(
                        human["contextual_grounding"]
                    ),
                    "human_primary_failure_mode": human["primary_failure_mode"],
                    "human_rationale": human["rationale"],
                }
            )
    return items


def select_items(
    items: list[dict[str, Any]],
    mode: str,
    scenario_ids: list[str],
) -> list[dict[str, Any]]:
    if scenario_ids:
        selected_ids = set(scenario_ids)
        items = [item for item in items if item["scenario_id"] in selected_ids]
    if mode == "smoke":
        # One unsafe old output and one safe improved output for the same scenario.
        rover_items = [item for item in items if item["scenario_id"] == "ROVER-002"]
        return rover_items or items[:2]
    return items


def _task_alpha(rows: list[dict[str, Any]]) -> float | None:
    matrix = []
    for formulation_index in range(3):
        values = []
        for row in rows:
            value = row["structured_judge"]["task_accuracy_ratings"][
                formulation_index
            ]
            values.append(np.nan if value is None else float(value))
        matrix.append(values)
    array = np.asarray(matrix, dtype=float)
    if array.shape[1] < 2 or np.all(np.isnan(array)):
        return None
    try:
        value = float(
            krippendorff.alpha(
                reliability_data=array,
                level_of_measurement="ordinal",
            )
        )
    except (ValueError, ZeroDivisionError):
        return None
    return value if math.isfinite(value) else None


def _consensus_value(row: dict[str, Any], dimension: str) -> Any:
    return row["structured_judge"]["consensus"][dimension]["final"]


def compute_metrics(rows: list[dict[str, Any]], gates: dict[str, Any]) -> dict[str, Any]:
    item_count = len(rows)
    stable_task = [
        row for row in rows if _consensus_value(row, "task_accuracy") is not None
    ]
    task_within_one_count = sum(
        abs(
            int(_consensus_value(row, "task_accuracy"))
            - int(row["human_task_accuracy"])
        )
        <= 1
        for row in stable_task
    )
    failure_stable = [
        row
        for row in rows
        if _consensus_value(row, "primary_failure_mode") is not None
    ]
    failure_exact_count = sum(
        _consensus_value(row, "primary_failure_mode")
        == row["human_primary_failure_mode"]
        for row in failure_stable
    )

    critical_reversals = []
    for row in rows:
        automated = _consensus_value(row, "primary_failure_mode")
        human = row["human_primary_failure_mode"]
        if (
            int(row["severity_class"]) == 5
            and automated is not None
            and ((automated == "unsafe") != (human == "unsafe"))
        ):
            critical_reversals.append(
                {
                    "calibration_item_id": row["calibration_item_id"],
                    "scenario_id": row["scenario_id"],
                    "human": human,
                    "automated": automated,
                }
            )

    model_call_count = 0
    unresolved_count = 0
    exact_format_count = 0
    evidence_valid_count = 0
    for row in rows:
        for formulation in row["structured_judge"]["formulation_results"]:
            model_call_count += int(formulation["model_call_count"])
            exact_format_count += round(
                formulation["exact_format_rate"] * formulation["model_call_count"]
            )
            evidence_valid_count += round(
                formulation["evidence_valid_rate"] * formulation["model_call_count"]
            )
            unresolved_count += len(
                formulation["deterministic_mapping"]["unresolved_check_ids"]
            )

    task_within_one_rate = (
        task_within_one_count / item_count if item_count else 0.0
    )
    failure_exact_rate = failure_exact_count / item_count if item_count else 0.0
    unresolved_rate = (
        unresolved_count / model_call_count if model_call_count else 0.0
    )
    alpha = _task_alpha(rows)
    gate_results = {
        "ordinal_krippendorff_alpha": (
            alpha is not None
            and alpha >= float(gates["ordinal_krippendorff_alpha_min"])
        ),
        "task_within_one_of_human_rate": (
            task_within_one_rate
            >= float(gates["task_within_one_of_human_rate_min"])
        ),
        "failure_mode_exact_match_rate": (
            failure_exact_rate
            >= float(gates["failure_mode_exact_match_rate_min"])
        ),
        "critical_safety_reversal_count": (
            len(critical_reversals)
            <= int(gates["critical_safety_reversal_count_max"])
        ),
        "unresolved_check_rate": (
            unresolved_rate <= float(gates["unresolved_check_rate_max"])
        ),
    }
    return {
        "item_count": item_count,
        "stable_task_consensus_count": len(stable_task),
        "stable_failure_consensus_count": len(failure_stable),
        "ordinal_krippendorff_alpha": alpha,
        "task_within_one_of_human_count": task_within_one_count,
        "task_within_one_of_human_rate": task_within_one_rate,
        "failure_mode_exact_match_count": failure_exact_count,
        "failure_mode_exact_match_rate": failure_exact_rate,
        "critical_safety_reversals": critical_reversals,
        "critical_safety_reversal_count": len(critical_reversals),
        "model_call_count": model_call_count,
        "unresolved_atomic_check_count": unresolved_count,
        "unresolved_check_rate": unresolved_rate,
        "exact_format_rate": (
            exact_format_count / model_call_count if model_call_count else 1.0
        ),
        "evidence_valid_rate": (
            evidence_valid_count / model_call_count if model_call_count else 1.0
        ),
        "acceptance_gates": gates,
        "gate_results": gate_results,
        "pipeline_usable": all(gate_results.values()),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "calibration_item_id",
        "condition",
        "scenario_id",
        "severity_class",
        "human_task_accuracy",
        "automated_task_accuracy",
        "task_ratings",
        "human_contextual_grounding",
        "automated_contextual_grounding",
        "grounding_ratings",
        "human_primary_failure_mode",
        "automated_primary_failure_mode",
        "failure_ratings",
        "unresolved_atomic_checks",
        "candidate_output_sha256",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            judge = row["structured_judge"]
            writer.writerow(
                {
                    "calibration_item_id": row["calibration_item_id"],
                    "condition": row["condition"],
                    "scenario_id": row["scenario_id"],
                    "severity_class": row["severity_class"],
                    "human_task_accuracy": row["human_task_accuracy"],
                    "automated_task_accuracy": _consensus_value(
                        row, "task_accuracy"
                    ),
                    "task_ratings": json.dumps(judge["task_accuracy_ratings"]),
                    "human_contextual_grounding": row[
                        "human_contextual_grounding"
                    ],
                    "automated_contextual_grounding": _consensus_value(
                        row, "contextual_grounding"
                    ),
                    "grounding_ratings": json.dumps(
                        judge["contextual_grounding_ratings"]
                    ),
                    "human_primary_failure_mode": row[
                        "human_primary_failure_mode"
                    ],
                    "automated_primary_failure_mode": _consensus_value(
                        row, "primary_failure_mode"
                    ),
                    "failure_ratings": json.dumps(
                        judge["primary_failure_mode_ratings"]
                    ),
                    "unresolved_atomic_checks": json.dumps(
                        judge["unresolved_atomic_checks"]
                    ),
                    "candidate_output_sha256": judge[
                        "candidate_output_sha256"
                    ],
                }
            )


def write_report(
    path: Path,
    run_id: str,
    rows: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    alpha = metrics["ordinal_krippendorff_alpha"]
    alpha_text = "NA" if alpha is None else f"{alpha:.4f}"
    lines = [
        "# Week 2 Structured Judge Calibration Report",
        "",
        f"- Run ID: `{run_id}`",
        f"- Judge architecture: evidence-decomposed checks plus deterministic mapping",
        f"- Frozen calibration items: {metrics['item_count']}",
        f"- Atomic model calls: {metrics['model_call_count']}",
        f"- Ordinal Krippendorff alpha: `{alpha_text}`",
        (
            "- Task within one of human: "
            f"`{metrics['task_within_one_of_human_count']}/{metrics['item_count']}` "
            f"(`{metrics['task_within_one_of_human_rate']:.3f}`)"
        ),
        (
            "- Failure-mode exact match: "
            f"`{metrics['failure_mode_exact_match_count']}/{metrics['item_count']}` "
            f"(`{metrics['failure_mode_exact_match_rate']:.3f}`)"
        ),
        (
            "- Unresolved atomic checks: "
            f"`{metrics['unresolved_atomic_check_count']}` "
            f"(`{metrics['unresolved_check_rate']:.3f}`)"
        ),
        f"- Exact two-line format rate: `{metrics['exact_format_rate']:.3f}`",
        f"- Evidence-valid rate: `{metrics['evidence_valid_rate']:.3f}`",
        (
            "- Critical safety reversals: "
            f"`{metrics['critical_safety_reversal_count']}`"
        ),
        (
            "- Pipeline usable under preregistered gates: "
            f"`{str(metrics['pipeline_usable']).lower()}`"
        ),
        "",
        "## Gate Results",
        "",
    ]
    for name, passed in metrics["gate_results"].items():
        lines.append(f"- `{name}`: `{'PASS' if passed else 'FAIL'}`")
    lines.extend(["", "## Item Audit", ""])
    for row in rows:
        judge = row["structured_judge"]
        lines.extend(
            [
                f"### {row['calibration_item_id']}",
                "",
                (
                    f"- Severity: `{row['severity_class']}`; "
                    f"candidate SHA-256: `{judge['candidate_output_sha256']}`"
                ),
                (
                    "- Human Task/Grounding/Failure: "
                    f"`{row['human_task_accuracy']}` / "
                    f"`{row['human_contextual_grounding']}` / "
                    f"`{row['human_primary_failure_mode']}`"
                ),
                (
                    "- Formulation Task ratings: "
                    f"`{judge['task_accuracy_ratings']}`"
                ),
                (
                    "- Formulation Grounding ratings: "
                    f"`{judge['contextual_grounding_ratings']}`"
                ),
                (
                    "- Formulation Failure ratings: "
                    f"`{judge['primary_failure_mode_ratings']}`"
                ),
                (
                    "- Consensus Task/Grounding/Failure: "
                    f"`{judge['consensus']['task_accuracy']['final']}` / "
                    f"`{judge['consensus']['contextual_grounding']['final']}` / "
                    f"`{judge['consensus']['primary_failure_mode']['final']}`"
                ),
                (
                    "- Unresolved checks: "
                    f"`{judge['unresolved_atomic_checks'] or 'none'}`"
                ),
                f"- Human rationale: {row['human_rationale']}",
                "",
                "Candidate:",
                "",
                "```text",
                row["candidate"],
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def environment_manifest() -> dict[str, Any]:
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "python": sys.version,
        "platform": platform.platform(),
        "seed": SEED,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "decoding": {
            "do_sample": False,
            "max_new_tokens": None,
        },
    }
    try:
        import torch
        import transformers

        manifest.update(
            {
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "gpu_name": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None
                ),
            }
        )
    except ImportError:
        manifest["torch_or_transformers_unavailable"] = True
    return manifest


def main() -> int:
    args = parse_args()
    scenario_doc = load_yaml(args.scenario_path)
    scenarios = scenario_doc["scenarios"]
    scenario_by_id = {item["scenario_id"]: item for item in scenarios}
    specs = structured.load_specs(args.prompt_spec, args.metadata_spec)
    validation_errors = structured.validate_specs(scenarios, specs)
    if validation_errors:
        for error in validation_errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(
        f"Validated structured Judge coverage for {len(scenarios)} scenarios "
        f"with {len(specs.prompts['formulations'])} formulations."
    )
    if args.mode == "validate":
        return 0

    old_rows = load_jsonl(args.old_rows)
    new_rows = load_jsonl(args.new_rows)
    new_human_doc = load_yaml(args.new_human)
    items = build_calibration_items(
        args.mode,
        old_rows,
        new_rows,
        new_human_doc,
    )
    items = select_items(items, args.mode, args.scenario_id)
    if not items:
        raise ValueError("No calibration items selected")

    run_id = args.run_id or (
        "mistral-structured-judge-"
        f"{args.mode}-v{structured.RUNNER_VERSION}"
    )
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "rows.checkpoint.jsonl"
    if checkpoint_path.exists() and not args.resume:
        raise FileExistsError(
            f"Run already exists; use --resume or a new run-id: {run_dir}"
        )
    completed_rows = load_jsonl(checkpoint_path) if args.resume else []
    completed_ids = {row["calibration_item_id"] for row in completed_rows}

    import torch

    from W02_Mistral_Eval_Runner import LocalMistralEngine

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    engine = LocalMistralEngine(args.model_dir, max_input_tokens=2048)

    rows = list(completed_rows)
    for item in items:
        item_id = item["calibration_item_id"]
        if item_id in completed_ids:
            print(f"Skipping completed {item_id}")
            continue
        scenario = scenario_by_id[item["scenario_id"]]
        print(f"Judging {item_id}")
        judge_result = structured.run_structured_judges(
            engine=engine,
            scenario=scenario,
            candidate=item["candidate"],
            specs=specs,
            max_new_tokens=args.max_new_tokens,
        )
        row = {
            "run_id": run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "benchmark_version": scenario_doc["benchmark_version"],
            **item,
            "split": scenario["split"],
            "platform": scenario["platform"],
            "severity_class": int(scenario["severity_class"]),
            "input_stimulus": scenario["input_stimulus"],
            "expected_behavior_range": scenario["expected_behavior_range"],
            "failure_conditions": scenario["failure_conditions"],
            "judge_model_id": MODEL_ID,
            "judge_model_revision": MODEL_REVISION,
            "judge_seed": SEED,
            "judge_decoding": {
                "do_sample": False,
                "max_new_tokens": args.max_new_tokens,
                "max_input_tokens": 2048,
            },
            "structured_judge": judge_result,
        }
        row["calibration_row_sha256"] = structured.canonical_sha256(row)
        append_jsonl(checkpoint_path, row)
        rows.append(row)

    # Preserve selection order even after resume.
    by_id = {row["calibration_item_id"]: row for row in rows}
    rows = [by_id[item["calibration_item_id"]] for item in items]
    gates = specs.prompts["calibration_policy"]["acceptance_gates"]
    metrics = compute_metrics(rows, gates)
    summary = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "structured_judge_runner_version": structured.RUNNER_VERSION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "seed": SEED,
        "prompt_spec": {
            "path": args.prompt_spec.name,
            "sha256": specs.prompt_sha256,
        },
        "metadata_spec": {
            "path": args.metadata_spec.name,
            "sha256": specs.metadata_sha256,
        },
        "input_hashes": {
            path.name: structured.sha256_text(path.read_text(encoding="utf-8"))
            for path in (
                args.scenario_path,
                args.old_rows,
                args.new_rows,
                args.new_human,
            )
        },
        "metrics": metrics,
    }
    (run_dir / "W02_Structured_Judge_Calibration_Summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    final_rows_path = run_dir / "W02_Structured_Judge_Calibration_Rows.jsonl"
    with final_rows_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    write_csv(
        run_dir / "W02_Structured_Judge_Calibration_Results.csv",
        rows,
    )
    write_report(
        run_dir / "W02_Structured_Judge_Calibration_Report.md",
        run_id,
        rows,
        metrics,
    )
    manifest = environment_manifest()
    manifest["decoding"]["max_new_tokens"] = args.max_new_tokens
    manifest["model_load_seconds"] = engine.load_seconds
    manifest["artifacts"] = {
        path.name: structured.sha256_text(path.read_text(encoding="utf-8"))
        for path in run_dir.iterdir()
        if path.is_file()
    }
    (run_dir / "W02_Structured_Judge_Run_Manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["pipeline_usable"] else 3


if __name__ == "__main__":
    raise SystemExit(main())

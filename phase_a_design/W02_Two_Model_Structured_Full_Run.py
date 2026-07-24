"""Run the submission-scale Week 2 benchmark for both pinned candidate models.

Workload:
  * 35 scenarios x 2 candidate models = 70 frozen candidate outputs.
  * Three structured Judge prompt formulations per candidate output.
  * Atomized evidence checks followed by deterministic score mapping.

The script is resumable at both the candidate and Judge stages and retains every
prompt, output, seed, model revision, hash, parse decision, and mapping trace.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import krippendorff
import numpy as np
import torch
import yaml
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import W02_Eval_Runner as base
import W02_Structured_Judge as structured
from W02_Mistral_Eval_Runner import (
    DEFAULT_MODEL_REVISION as MISTRAL_REVISION,
)
from W02_Mistral_Eval_Runner import LocalMistralEngine
from W02_Mistral_Eval_Runner import MODEL_ID as MISTRAL_MODEL_ID


ROOT = Path(__file__).resolve().parent
SCENARIO_PATH = ROOT / "W02_Scenarios.yaml"
CHECK_PATH = ROOT / "W02_Deterministic_Checks.yaml"
MISTRAL_PROMPT_PATH = ROOT / "W02_Prompt_Spec_v0.4.0.yaml"
FLAN_PROMPT_PATH = ROOT / "W02_Prompt_Spec_v0.4.1_flan_compact.yaml"
DEFAULT_MISTRAL_DIR = Path("/workspace/models/mistral_7b_instruct_v0_2")
DEFAULT_FLAN_DIR = Path("/workspace/models/flan_t5_base")
DEFAULT_OUTPUT_ROOT = Path("/workspace/experiments/w02_submission_full")
DEFAULT_CALIBRATION_SUMMARY = (
    Path("/workspace/experiments/w02_structured_judge")
    / "mistral-structured-judge-calibration-v0.6.2"
    / "W02_Structured_Judge_Calibration_Summary.json"
)
FLAN_MODEL_ID = "google/flan-t5-base"
FLAN_REVISION = "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2"
RUNNER_VERSION = "0.6.2"
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("validate", "candidates", "judges", "all"),
        default="all",
    )
    parser.add_argument(
        "--candidate-model",
        choices=("both", "mistral", "flan"),
        default="both",
    )
    parser.add_argument("--run-id", default="w02-two-model-full-v0.6.2")
    parser.add_argument("--mistral-dir", type=Path, default=DEFAULT_MISTRAL_DIR)
    parser.add_argument("--flan-dir", type=Path, default=DEFAULT_FLAN_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scenario-path", type=Path, default=SCENARIO_PATH)
    parser.add_argument("--check-spec", type=Path, default=CHECK_PATH)
    parser.add_argument(
        "--mistral-prompt-spec",
        type=Path,
        default=MISTRAL_PROMPT_PATH,
    )
    parser.add_argument(
        "--flan-prompt-spec",
        type=Path,
        default=FLAN_PROMPT_PATH,
    )
    parser.add_argument(
        "--judge-prompt-spec",
        type=Path,
        default=structured.DEFAULT_PROMPT_SPEC,
    )
    parser.add_argument(
        "--judge-metadata-spec",
        type=Path,
        default=structured.DEFAULT_METADATA_SPEC,
    )
    parser.add_argument(
        "--calibration-summary",
        type=Path,
        default=DEFAULT_CALIBRATION_SUMMARY,
    )
    parser.add_argument("--scenario-id", action="append", default=[])
    parser.add_argument("--judge-max-new-tokens", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-uncalibrated-smoke",
        action="store_true",
        help="Only for smoke diagnostics; never use for a submission-scale claim.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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


def select_scenarios(
    scenarios: list[dict[str, Any]],
    scenario_ids: list[str],
) -> list[dict[str, Any]]:
    if not scenario_ids:
        return scenarios
    wanted = set(scenario_ids)
    selected = [item for item in scenarios if item["scenario_id"] in wanted]
    missing = wanted - {item["scenario_id"] for item in selected}
    if missing:
        raise ValueError(f"Unknown scenario IDs: {sorted(missing)}")
    return selected


class GpuFlanEngine:
    """Pinned FLAN-T5-base generation on CUDA with float32 weights."""

    def __init__(self, model_dir: Path, max_input_tokens: int) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the RunPod full run")
        if not (model_dir / "config.json").exists():
            raise FileNotFoundError(f"FLAN checkpoint is incomplete: {model_dir}")
        self.model_dir = model_dir
        self.max_input_tokens = max_input_tokens
        self.device = torch.device("cuda:0")
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_dir,
            local_files_only=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()
        self.load_seconds = time.perf_counter() - started
        print(
            f"FLAN loaded on {self.device} in {self.load_seconds:.2f}s",
            flush=True,
        )

    def generate(self, prompt: str, max_new_tokens: int) -> dict[str, Any]:
        all_ids = self.tokenizer(
            prompt,
            add_special_tokens=True,
            verbose=False,
        )["input_ids"]
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        started = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )
        torch.cuda.synchronize(self.device)
        latency_ms = (time.perf_counter() - started) * 1000
        text = self.tokenizer.decode(
            generated[0],
            skip_special_tokens=True,
        ).strip()
        return {
            "text": text,
            "latency_ms": round(latency_ms, 4),
            "input_tokens": int(encoded["input_ids"].shape[-1]),
            "untruncated_input_tokens": len(all_ids),
            "input_truncated": len(all_ids) > self.max_input_tokens,
            "output_tokens": int(generated.shape[-1]),
        }


def set_determinism() -> None:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)


def model_conditions(args: argparse.Namespace) -> list[str]:
    if args.candidate_model == "both":
        return ["mistral", "flan"]
    return [args.candidate_model]


def candidate_config(
    model_key: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if model_key == "mistral":
        return {
            "model_id": MISTRAL_MODEL_ID,
            "model_revision": MISTRAL_REVISION,
            "model_dir": args.mistral_dir,
            "prompt_path": args.mistral_prompt_spec,
            "precision": "bfloat16",
            "device": "cuda:0",
        }
    return {
        "model_id": FLAN_MODEL_ID,
        "model_revision": FLAN_REVISION,
        "model_dir": args.flan_dir,
        "prompt_path": args.flan_prompt_spec,
        "precision": "float32",
        "device": "cuda:0",
    }


def generate_candidates(
    args: argparse.Namespace,
    selected: list[dict[str, Any]],
    run_dir: Path,
) -> list[dict[str, Any]]:
    checkpoint = run_dir / "candidate_rows.checkpoint.jsonl"
    existing = load_jsonl(checkpoint) if args.resume else []
    if checkpoint.exists() and not args.resume:
        raise FileExistsError(
            f"Candidate checkpoint exists; use --resume: {checkpoint}"
        )
    completed = {row["candidate_item_id"] for row in existing}
    rows = list(existing)
    check_spec = load_yaml(args.check_spec)
    engines: dict[str, Any] = {}

    for model_key in model_conditions(args):
        config = candidate_config(model_key, args)
        prompt_spec = load_yaml(config["prompt_path"])
        generation = prompt_spec["generation"]
        if model_key == "mistral":
            engine = engines.get("mistral")
            if engine is None:
                engine = LocalMistralEngine(
                    config["model_dir"],
                    int(generation["max_input_tokens"]),
                )
                engines["mistral"] = engine
        else:
            engine = GpuFlanEngine(
                config["model_dir"],
                int(generation["max_input_tokens"]),
            )
            engines["flan"] = engine

        for index, scenario in enumerate(selected, start=1):
            item_id = f"{model_key}::{scenario['scenario_id']}"
            if item_id in completed:
                continue
            prompt = base.candidate_prompt(scenario, prompt_spec)
            print(
                f"[candidate {model_key} {index}/{len(selected)}] "
                f"{scenario['scenario_id']}",
                flush=True,
            )
            error = None
            try:
                result = engine.generate(
                    prompt,
                    int(generation["max_new_tokens"]),
                )
                output = str(result["text"]).strip()
                if not output:
                    raise RuntimeError("empty candidate output")
            except Exception as exception:
                error = f"{type(exception).__name__}: {exception}"
                output = ""
                result = {
                    "latency_ms": 0.0,
                    "input_tokens": 0,
                    "untruncated_input_tokens": 0,
                    "input_truncated": False,
                    "output_tokens": 0,
                }
            row = {
                "run_id": args.run_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "runner_version": RUNNER_VERSION,
                "candidate_item_id": item_id,
                "candidate_model_key": model_key,
                "scenario_id": scenario["scenario_id"],
                "split": scenario["split"],
                "platform": scenario["platform"],
                "severity_class": int(scenario["severity_class"]),
                "candidate_model_id": config["model_id"],
                "candidate_model_revision": config["model_revision"],
                "candidate_model_directory": str(config["model_dir"]),
                "candidate_precision": config["precision"],
                "candidate_device": config["device"],
                "candidate_prompt_spec": config["prompt_path"].name,
                "candidate_prompt_spec_sha256": base.sha256_file(
                    config["prompt_path"]
                ),
                "candidate_prompt_version": str(prompt_spec["version"]),
                "candidate_prompt": prompt,
                "candidate_prompt_sha256": structured.sha256_text(prompt),
                "candidate_generation": {
                    "seed": SEED,
                    "do_sample": False,
                    "max_input_tokens": int(generation["max_input_tokens"]),
                    "max_new_tokens": int(generation["max_new_tokens"]),
                },
                "candidate_input_tokens": result["input_tokens"],
                "candidate_untruncated_input_tokens": result[
                    "untruncated_input_tokens"
                ],
                "candidate_input_truncated": result["input_truncated"],
                "candidate_output_tokens": result["output_tokens"],
                "candidate_latency_ms": result["latency_ms"],
                "input_stimulus": scenario["input_stimulus"],
                "raw_output": output,
                "candidate_output_sha256": structured.sha256_text(output),
                "generation_error": error,
                "deterministic_audit": base.deterministic_audit(
                    output,
                    scenario,
                    check_spec,
                ),
            }
            row["candidate_row_sha256"] = structured.canonical_sha256(row)
            append_jsonl(checkpoint, row)
            rows.append(row)

        if model_key == "flan":
            # Free FLAN memory before the longer Mistral Judge stage.
            del engines["flan"]
            del engine
            torch.cuda.empty_cache()

    selected_ids = {
        f"{model_key}::{scenario['scenario_id']}"
        for model_key in model_conditions(args)
        for scenario in selected
    }
    by_id = {row["candidate_item_id"]: row for row in rows}
    ordered = [
        by_id[item_id]
        for item_id in sorted(selected_ids)
        if item_id in by_id
    ]
    final_path = run_dir / "W02_Two_Model_Candidate_Rows.jsonl"
    with final_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return ordered


def require_calibration(
    args: argparse.Namespace,
    selected: list[dict[str, Any]],
) -> dict[str, Any] | None:
    full_scale = len(selected) == 35 and len(model_conditions(args)) == 2
    if args.allow_uncalibrated_smoke and not full_scale:
        return None
    if not args.calibration_summary.exists():
        raise FileNotFoundError(
            "Structured Judge calibration summary is missing: "
            f"{args.calibration_summary}"
        )
    summary = json.loads(args.calibration_summary.read_text(encoding="utf-8"))
    if not summary["metrics"]["pipeline_usable"]:
        failed = [
            name
            for name, passed in summary["metrics"]["gate_results"].items()
            if not passed
        ]
        raise RuntimeError(
            "Structured Judge calibration did not pass all gates: "
            f"{failed}. Do not run submission scoring."
        )
    return summary


def run_judges(
    args: argparse.Namespace,
    selected: list[dict[str, Any]],
    run_dir: Path,
    candidate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    calibration = require_calibration(args, selected)
    scenario_doc = load_yaml(args.scenario_path)
    scenario_by_id = {
        item["scenario_id"]: item for item in scenario_doc["scenarios"]
    }
    specs = structured.load_specs(
        args.judge_prompt_spec,
        args.judge_metadata_spec,
    )
    errors = structured.validate_specs(scenario_doc["scenarios"], specs)
    if errors:
        raise ValueError("Structured Judge spec errors: " + " | ".join(errors))

    checkpoint = run_dir / "judged_rows.checkpoint.jsonl"
    existing = load_jsonl(checkpoint) if args.resume else []
    if checkpoint.exists() and not args.resume:
        raise FileExistsError(f"Judge checkpoint exists; use --resume: {checkpoint}")
    completed = {row["candidate_item_id"] for row in existing}
    rows = list(existing)

    judge_engine = LocalMistralEngine(args.mistral_dir, max_input_tokens=2048)
    for index, candidate_row in enumerate(candidate_rows, start=1):
        item_id = candidate_row["candidate_item_id"]
        if item_id in completed:
            continue
        scenario = scenario_by_id[candidate_row["scenario_id"]]
        print(
            f"[judge {index}/{len(candidate_rows)}] {item_id}",
            flush=True,
        )
        if candidate_row["generation_error"]:
            judge_result = None
            review_reasons = ["candidate_generation_error"]
        else:
            judge_result = structured.run_structured_judges(
                engine=judge_engine,
                scenario=scenario,
                candidate=candidate_row["raw_output"],
                specs=specs,
                max_new_tokens=args.judge_max_new_tokens,
            )
            review_reasons = list(judge_result["human_review_reasons"])
        row = {
            **candidate_row,
            "judge_model_id": MISTRAL_MODEL_ID,
            "judge_model_revision": MISTRAL_REVISION,
            "judge_condition": (
                "same_checkpoint_prompt_sensitivity_not_independent"
                if candidate_row["candidate_model_key"] == "mistral"
                else "cross_model_prompt_sensitivity_not_independent"
            ),
            "judge_seed": SEED,
            "judge_decoding": {
                "do_sample": False,
                "max_input_tokens": 2048,
                "max_new_tokens": args.judge_max_new_tokens,
            },
            "judge_calibration_run_id": (
                calibration["run_id"] if calibration is not None else None
            ),
            "judge_calibration_summary_sha256": (
                base.sha256_file(args.calibration_summary)
                if calibration is not None
                else None
            ),
            "structured_judge": judge_result,
            "human_review_required": bool(review_reasons),
            "human_review_reasons": review_reasons,
            "human_task_accuracy": None,
            "human_contextual_grounding": None,
            "human_primary_failure_mode": None,
            "human_rationale": None,
            "final_score_status": (
                "automated_provisional_human_review_required"
                if review_reasons
                else "automated_provisional"
            ),
        }
        row["judged_row_sha256"] = structured.canonical_sha256(row)
        append_jsonl(checkpoint, row)
        rows.append(row)

    expected_ids = {row["candidate_item_id"] for row in candidate_rows}
    by_id = {row["candidate_item_id"]: row for row in rows}
    ordered = [
        by_id[item_id]
        for item_id in sorted(expected_ids)
        if item_id in by_id
    ]
    final_path = run_dir / "W02_Two_Model_Structured_Judged_Rows.jsonl"
    with final_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return ordered


def task_alpha(rows: list[dict[str, Any]]) -> float | None:
    usable = [row for row in rows if row["structured_judge"] is not None]
    if len(usable) < 2:
        return None
    matrix = np.asarray(
        [
            [
                np.nan if value is None else float(value)
                for row in usable
                for value in [row["structured_judge"]["task_accuracy_ratings"][index]]
            ]
            for index in range(3)
        ],
        dtype=float,
    )
    try:
        value = float(
            krippendorff.alpha(
                reliability_data=matrix,
                level_of_measurement="ordinal",
            )
        )
    except (ValueError, ZeroDivisionError):
        return None
    return value if math.isfinite(value) else None


def consensus_value(row: dict[str, Any], dimension: str) -> Any:
    judge = row["structured_judge"]
    if judge is None:
        return None
    return judge["consensus"][dimension]["final"]


def compute_summary(
    args: argparse.Namespace,
    candidate_rows: list[dict[str, Any]],
    judged_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for model_key in model_conditions(args):
        candidate_group = [
            row for row in candidate_rows if row["candidate_model_key"] == model_key
        ]
        judge_group = [
            row for row in judged_rows if row["candidate_model_key"] == model_key
        ]
        stable_task = [
            int(consensus_value(row, "task_accuracy"))
            for row in judge_group
            if consensus_value(row, "task_accuracy") is not None
        ]
        stable_grounding = [
            int(consensus_value(row, "contextual_grounding"))
            for row in judge_group
            if consensus_value(row, "contextual_grounding") is not None
        ]
        failure_counts: dict[str, int] = {}
        for row in judge_group:
            value = consensus_value(row, "primary_failure_mode")
            label = "unresolved" if value is None else str(value)
            failure_counts[label] = failure_counts.get(label, 0) + 1
        groups[model_key] = {
            "candidate_count": len(candidate_group),
            "candidate_generation_error_count": sum(
                bool(row["generation_error"]) for row in candidate_group
            ),
            "candidate_prompt_truncation_count": sum(
                bool(row["candidate_input_truncated"]) for row in candidate_group
            ),
            "judged_count": len(judge_group),
            "stable_task_consensus_count": len(stable_task),
            "stable_grounding_consensus_count": len(stable_grounding),
            "mean_stable_task_accuracy": (
                statistics.fmean(stable_task) if stable_task else None
            ),
            "mean_stable_contextual_grounding": (
                statistics.fmean(stable_grounding) if stable_grounding else None
            ),
            "primary_failure_counts": failure_counts,
            "task_ordinal_krippendorff_alpha": task_alpha(judge_group),
            "human_review_required_count": sum(
                bool(row["human_review_required"]) for row in judge_group
            ),
        }
    return {
        "run_id": args.run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "runner_version": RUNNER_VERSION,
        "seed": SEED,
        "scenario_count": len(
            {row["scenario_id"] for row in candidate_rows}
        ),
        "candidate_output_count": len(candidate_rows),
        "judged_output_count": len(judged_rows),
        "judge_formulations": [
            "criterion_first_v0.6.0",
            "evidence_first_v0.6.0",
            "consequence_first_v0.6.0",
        ],
        "candidate_models": groups,
        "claim_boundary": (
            "L0 synthetic product-context proxy; automated scores are provisional "
            "and severity-5 rows require human review."
        ),
    }


def write_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "candidate_item_id",
        "candidate_model_key",
        "candidate_model_id",
        "scenario_id",
        "split",
        "platform",
        "severity_class",
        "candidate_prompt_version",
        "candidate_prompt_sha256",
        "candidate_output_sha256",
        "candidate_output",
        "task_ratings",
        "task_consensus",
        "grounding_ratings",
        "grounding_consensus",
        "failure_ratings",
        "failure_consensus",
        "human_review_required",
        "human_review_reasons",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            judge = row["structured_judge"]
            writer.writerow(
                {
                    "candidate_item_id": row["candidate_item_id"],
                    "candidate_model_key": row["candidate_model_key"],
                    "candidate_model_id": row["candidate_model_id"],
                    "scenario_id": row["scenario_id"],
                    "split": row["split"],
                    "platform": row["platform"],
                    "severity_class": row["severity_class"],
                    "candidate_prompt_version": row["candidate_prompt_version"],
                    "candidate_prompt_sha256": row["candidate_prompt_sha256"],
                    "candidate_output_sha256": row["candidate_output_sha256"],
                    "candidate_output": row["raw_output"],
                    "task_ratings": json.dumps(
                        judge["task_accuracy_ratings"] if judge else None
                    ),
                    "task_consensus": consensus_value(row, "task_accuracy"),
                    "grounding_ratings": json.dumps(
                        judge["contextual_grounding_ratings"] if judge else None
                    ),
                    "grounding_consensus": consensus_value(
                        row,
                        "contextual_grounding",
                    ),
                    "failure_ratings": json.dumps(
                        judge["primary_failure_mode_ratings"] if judge else None
                    ),
                    "failure_consensus": consensus_value(
                        row,
                        "primary_failure_mode",
                    ),
                    "human_review_required": row["human_review_required"],
                    "human_review_reasons": json.dumps(
                        row["human_review_reasons"]
                    ),
                }
            )


def write_report(
    path: Path,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Week 2 Two-Model Structured Full Run",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Scenarios: `{summary['scenario_count']}`",
        f"- Candidate outputs: `{summary['candidate_output_count']}`",
        f"- Judged outputs: `{summary['judged_output_count']}`",
        f"- Seed: `{summary['seed']}`",
        "- Judge formulations: `criterion_first`, `evidence_first`, `consequence_first`",
        f"- Claim boundary: {summary['claim_boundary']}",
        "",
        "## Model Summary",
        "",
        "| Candidate | Outputs | Errors | Stable Task | Task alpha | Mean Task | Review |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_key, metrics in summary["candidate_models"].items():
        alpha = metrics["task_ordinal_krippendorff_alpha"]
        mean_task = metrics["mean_stable_task_accuracy"]
        lines.append(
            f"| {model_key} | {metrics['candidate_count']} | "
            f"{metrics['candidate_generation_error_count']} | "
            f"{metrics['stable_task_consensus_count']} | "
            f"{'NA' if alpha is None else f'{alpha:.3f}'} | "
            f"{'NA' if mean_task is None else f'{mean_task:.3f}'} | "
            f"{metrics['human_review_required_count']} |"
        )
    lines.extend(["", "## Per-Output Audit", ""])
    for row in rows:
        judge = row["structured_judge"]
        lines.extend(
            [
                f"### {row['candidate_item_id']}",
                "",
                (
                    f"- Split / severity: `{row['split']}` / "
                    f"`{row['severity_class']}`"
                ),
                (
                    "- Task / Grounding / Failure consensus: "
                    f"`{consensus_value(row, 'task_accuracy')}` / "
                    f"`{consensus_value(row, 'contextual_grounding')}` / "
                    f"`{consensus_value(row, 'primary_failure_mode')}`"
                ),
                (
                    "- Formulation Task ratings: "
                    f"`{judge['task_accuracy_ratings'] if judge else None}`"
                ),
                (
                    "- Review reasons: "
                    f"`{row['human_review_reasons'] or 'none'}`"
                ),
                "",
                "```text",
                row["raw_output"] or "[generation error]",
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    source_paths: list[Path],
    run_dir: Path,
) -> None:
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "seed": SEED,
        "source_hashes": {
            source.name: base.sha256_file(source) for source in source_paths
        },
        "artifact_hashes": {
            artifact.name: base.sha256_file(artifact)
            for artifact in run_dir.iterdir()
            if artifact.is_file()
        },
    }
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    args = parse_args()
    scenario_doc = load_yaml(args.scenario_path)
    selected = select_scenarios(
        scenario_doc["scenarios"],
        args.scenario_id,
    )
    specs = structured.load_specs(
        args.judge_prompt_spec,
        args.judge_metadata_spec,
    )
    validation_errors = structured.validate_specs(
        scenario_doc["scenarios"],
        specs,
    )
    if validation_errors:
        for error in validation_errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(
        f"Validated {len(scenario_doc['scenarios'])} scenarios and "
        f"{len(specs.prompts['formulations'])} Judge formulations.",
        flush=True,
    )
    if args.stage == "validate":
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the submission-scale full run")

    set_determinism()
    run_dir = args.output_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    candidate_final = run_dir / "W02_Two_Model_Candidate_Rows.jsonl"
    if args.stage in {"candidates", "all"}:
        candidate_rows = generate_candidates(args, selected, run_dir)
    else:
        candidate_rows = load_jsonl(candidate_final)
        if not candidate_rows:
            raise FileNotFoundError(
                f"Candidate rows are required for Judge stage: {candidate_final}"
            )

    if args.stage == "candidates":
        print(f"Frozen {len(candidate_rows)} candidate outputs in {run_dir}")
        return 0

    judged_final = run_dir / "W02_Two_Model_Structured_Judged_Rows.jsonl"
    if args.stage in {"judges", "all"}:
        judged_rows = run_judges(
            args,
            selected,
            run_dir,
            candidate_rows,
        )
    else:
        judged_rows = load_jsonl(judged_final)

    summary = compute_summary(args, candidate_rows, judged_rows)
    summary_path = run_dir / "W02_Two_Model_Structured_Summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    write_results_csv(
        run_dir / "W02_Two_Model_Structured_Results.csv",
        judged_rows,
    )
    write_report(
        run_dir / "W02_Two_Model_Structured_Report.md",
        summary,
        judged_rows,
    )
    source_paths = [
        args.scenario_path,
        args.check_spec,
        args.mistral_prompt_spec,
        args.flan_prompt_spec,
        args.judge_prompt_spec,
        args.judge_metadata_spec,
        Path(__file__),
        structured.__file__ and Path(structured.__file__),
    ]
    write_manifest(
        run_dir / "W02_Two_Model_Structured_Run_Manifest.json",
        args,
        [path for path in source_paths if isinstance(path, Path)],
        run_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

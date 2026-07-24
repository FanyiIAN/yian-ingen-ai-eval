"""Freeze and audit provenance for the completed Week 2 local FLAN run.

The evaluation runner already preserves row-level outputs and judge evidence. This
utility adds a rendered-prompt ledger, exact file/model hashes, runtime versions,
and machine-checkable completeness/behavior summaries without rerunning inference.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jsonschema

import W02_Eval_Runner as runner


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DEFAULT_RUN_DIR = (
    ROOT
    / "experiments"
    / "w02_local_flan_pipeline"
    / "local-flan-full-v0.2.0"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, base: Path | None = None) -> dict[str, Any]:
    label = str(path.relative_to(base)) if base and path.is_relative_to(base) else str(path)
    return {"path": label, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def package_versions(requirements_path: Path) -> dict[str, Any]:
    requested: dict[str, str] = {}
    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("-") or stripped.startswith("#"):
            continue
        name, separator, pinned = stripped.partition("==")
        if separator:
            requested[name] = pinned
    installed: dict[str, str | None] = {}
    mismatches: dict[str, dict[str, str | None]] = {}
    for name, pinned in requested.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        installed[name] = actual
        if actual != pinned:
            mismatches[name] = {"required": pinned, "installed": actual}
    return {
        "requirements": requested,
        "installed": installed,
        "mismatches": mismatches,
    }


def render_prompt_ledger(
    rows: list[dict[str, Any]],
    scenarios: dict[str, dict[str, Any]],
    regulations: dict[str, dict[str, Any]],
    prompt_spec: dict[str, Any],
    judge_spec: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    ledger: list[dict[str, Any]] = []
    hash_mismatches: list[str] = []
    for row in rows:
        scenario = scenarios[row["scenario_id"]]
        candidate = runner.candidate_prompt(scenario, prompt_spec)
        candidate_hash = sha256_text(candidate)
        if candidate_hash != row["candidate_prompt_hash"]:
            hash_mismatches.append(row["scenario_id"])

        regulation_text = runner.build_regulation_text(scenario, regulations)
        context = runner.judge_context(scenario, row["raw_output"], regulation_text)
        judge_prompts: list[dict[str, Any]] = []
        results = {item["formulation"]: item for item in row["judge_results"]}
        for formulation_name, formulation in judge_spec["formulations"].items():
            result = results[formulation_name]
            rationale_context = {
                **context,
                "task_score": result["task_accuracy"],
                "context_score": result["contextual_grounding"],
                "failure_mode": result["primary_failure_mode"],
            }
            rendered = {
                "task_accuracy": formulation["task_accuracy_template"].format(**context),
                "contextual_grounding": formulation[
                    "contextual_grounding_template"
                ].format(**context),
                "failure_mode": formulation["failure_mode_template"].format(**context),
                "rationale": formulation["rationale_template"].format(
                    **rationale_context
                ),
            }
            judge_prompts.append(
                {
                    "formulation": formulation_name,
                    "prompt_version": formulation["version"],
                    "rendered_prompts": rendered,
                    "rendered_prompt_sha256": {
                        key: sha256_text(value) for key, value in rendered.items()
                    },
                    "recorded_scores": {
                        "task_accuracy": result["task_accuracy"],
                        "contextual_grounding": result["contextual_grounding"],
                        "primary_failure_mode": result["primary_failure_mode"],
                    },
                }
            )
        ledger.append(
            {
                "run_id": row["run_id"],
                "scenario_id": row["scenario_id"],
                "source_result_row_sha256": sha256_text(
                    json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                ),
                "candidate_prompt_version": row["prompt_template_version"],
                "candidate_prompt": candidate,
                "candidate_prompt_sha256": candidate_hash,
                "judge_prompts": judge_prompts,
            }
        )
    return ledger, hash_mismatches


def mean(values: list[float]) -> float:
    return round(statistics.fmean(values), 6) if values else 0.0


def median(values: list[float]) -> float:
    return round(float(statistics.median(values)), 6) if values else 0.0


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    outputs = Counter(row["raw_output"] for row in rows)
    formulation_scores: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"task_accuracy": [], "contextual_grounding": []}
    )
    formulation_failures: dict[str, Counter[str]] = defaultdict(Counter)
    margins: dict[str, list[float]] = defaultdict(list)
    rationales: list[str] = []
    all_judge_failures: Counter[str] = Counter()
    prompt_truncation_ids: list[str] = []

    for row in rows:
        if any(item["any_prompt_truncated"] for item in row["judge_results"]):
            prompt_truncation_ids.append(row["scenario_id"])
        for result in row["judge_results"]:
            name = result["formulation"]
            formulation_scores[name]["task_accuracy"].append(result["task_accuracy"])
            formulation_scores[name]["contextual_grounding"].append(
                result["contextual_grounding"]
            )
            formulation_failures[name][result["primary_failure_mode"]] += 1
            all_judge_failures[result["primary_failure_mode"]] += 1
            rationales.append(result["rationale"])
            margins["task_accuracy"].append(
                result["task_accuracy_classification"]["margin_to_second"]
            )
            margins["contextual_grounding"].append(
                result["contextual_grounding_classification"]["margin_to_second"]
            )
            margins["failure_mode"].append(
                result["failure_mode_classification"]["margin_to_second"]
            )

    required_missing_any = [
        row["scenario_id"]
        for row in rows
        if row["deterministic_audit"]["missing_required_lexical_signals"]
    ]
    required_missing_all = [
        row["scenario_id"]
        for row in rows
        if "missing_all_required_signals_but_judge_passed"
        in row["human_review_reasons"]
    ]
    critical_flag_ids = [
        row["scenario_id"]
        for row in rows
        if row["deterministic_audit"]["critical_flags"]
    ]

    exact_task_agreement = sum(len(set(row["task_accuracy_ratings"])) == 1 for row in rows)
    exact_grounding_agreement = sum(
        len(set(row["contextual_grounding_ratings"])) == 1 for row in rows
    )
    exact_failure_agreement = sum(
        len(set(row["primary_failure_mode_ratings"])) == 1 for row in rows
    )

    low_margin_threshold = 0.10
    margin_summary = {
        dimension: {
            "count": len(values),
            "mean": mean(values),
            "median": median(values),
            "at_or_below_0.10": sum(value <= low_margin_threshold for value in values),
        }
        for dimension, values in margins.items()
    }

    return {
        "row_count": len(rows),
        "candidate_outputs": {
            "nonempty": sum(bool(row["raw_output"].strip()) for row in rows),
            "unique_exact_outputs": len(outputs),
            "rows_sharing_an_exact_output": sum(
                count for count in outputs.values() if count > 1
            ),
            "mean_output_characters": mean([len(row["raw_output"]) for row in rows]),
            "median_output_characters": median([len(row["raw_output"]) for row in rows]),
            "candidate_input_truncation_ids": [
                row["scenario_id"] for row in rows if row["candidate_input_truncated"]
            ],
            "top_exact_outputs": [
                {"output": output, "count": count}
                for output, count in outputs.most_common(10)
            ],
        },
        "judge_behavior": {
            "formulation_means": {
                name: {
                    "task_accuracy": mean(values["task_accuracy"]),
                    "contextual_grounding": mean(values["contextual_grounding"]),
                    "failure_mode_counts": dict(formulation_failures[name]),
                }
                for name, values in formulation_scores.items()
            },
            "all_105_failure_votes": dict(all_judge_failures),
            "exact_agreement_rows": {
                "task_accuracy": exact_task_agreement,
                "contextual_grounding": exact_grounding_agreement,
                "primary_failure_mode": exact_failure_agreement,
            },
            "classification_margin_summary": margin_summary,
            "prompt_truncation_ids": prompt_truncation_ids,
            "unique_rationales": len(set(rationales)),
            "empty_or_skipped_rationales": sum(
                not value.strip() or value == "not_generated" for value in rationales
            ),
            "top_exact_rationales": [
                {"rationale": value, "count": count}
                for value, count in Counter(rationales).most_common(10)
            ],
        },
        "audit_and_review": {
            "missing_any_required_lexical_signal_ids": required_missing_any,
            "missing_all_required_but_median_pass_ids": required_missing_all,
            "deterministic_critical_flag_ids": critical_flag_ids,
            "human_review_required_ids": [
                row["scenario_id"] for row in rows if row["human_review_required"]
            ],
            "severity_5_ids": [
                row["scenario_id"] for row in rows if row["severity_class"] == 5
            ],
            "severity_5_task_at_or_below_2_ids": [
                row["scenario_id"]
                for row in rows
                if row["severity_class"] == 5 and row["final_task_accuracy"] <= 2
            ],
            "severity_5_final_unsafe_ids": [
                row["scenario_id"]
                for row in rows
                if row["severity_class"] == 5
                and row["final_primary_failure_mode"] == "unsafe"
            ],
        },
        "final_provisional_distributions": {
            "task_accuracy": dict(Counter(row["final_task_accuracy"] for row in rows)),
            "contextual_grounding": dict(
                Counter(row["final_contextual_grounding"] for row in rows)
            ),
            "primary_failure_mode": dict(
                Counter(row["final_primary_failure_mode"] for row in rows)
            ),
        },
    }


def completeness(rows: list[dict[str, Any]], schema: dict[str, Any]) -> dict[str, Any]:
    required = schema["required"]
    missing_by_field = {
        field: [row["scenario_id"] for row in rows if field not in row]
        for field in required
    }
    classification_complete = all(
        all(
            all(
                key in result[dimension]
                for key in (
                    "selected",
                    "losses",
                    "margin_to_second",
                    "latency_ms",
                    "input_tokens",
                    "untruncated_input_tokens",
                    "input_truncated",
                )
            )
            for result in row["judge_results"]
            for dimension in (
                "task_accuracy_classification",
                "contextual_grounding_classification",
                "failure_mode_classification",
            )
        )
        for row in rows
    )
    return {
        "schema_valid_rows": len(rows),
        "unique_scenario_ids": len({row["scenario_id"] for row in rows}),
        "required_field_missing_ids": {
            field: ids for field, ids in missing_by_field.items() if ids
        },
        "raw_outputs_recorded": sum("raw_output" in row for row in rows),
        "deterministic_audits_recorded": sum(
            "deterministic_audit" in row for row in rows
        ),
        "judge_results_recorded": sum(len(row["judge_results"]) for row in rows),
        "judge_ratings_and_losses_complete": classification_complete,
        "seeds": sorted({row["seed"] for row in rows}),
        "candidate_model_ids": sorted({row["candidate_model_id"] for row in rows}),
        "candidate_model_revisions": sorted(
            {row["candidate_model_revision"] for row in rows}
        ),
        "prompt_template_versions": sorted(
            {row["prompt_template_version"] for row in rows}
        ),
        "judge_prompt_versions": sorted(
            {version for row in rows for version in row["judge_prompt_versions"]}
        ),
        "generation_config_hashes": sorted(
            {row["generation_config_hash"] for row in rows}
        ),
    }


def markdown_cell(value: Any) -> str:
    text = str(value).replace("|", "\\|").replace("\r", "").replace("\n", "<br>")
    return text if text else "(empty)"


def write_evidence_table(
    path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]
) -> None:
    lines = [
        "# Week 2 Local FLAN: All Output Evidence",
        "",
        "> Canonical machine-readable evidence is the final JSONL. This table is a",
        "> human-readable view. Every score is automated and provisional; human",
        "> adjudication is still pending.",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Model: `{summary['model_id']}`",
        f"- Revision: `{summary['model_revision']}`",
        f"- Seed: `{summary['seed']}`",
        f"- Candidate prompt version: `{summary['prompt_spec_version']}`",
        f"- Judge prompt spec: `{summary['judge_prompt_spec_version']}`",
        "- Judge formulations: `criterion_first`, `evidence_first`, `failure_first`",
        "",
        "| Scenario | Split | Sev. | Raw candidate output | Task ratings -> median | Grounding ratings -> median | Failure votes -> final | Deterministic evidence | Review reasons |",
        "|---|---|---:|---|---|---|---|---|---|",
    ]
    for row in rows:
        audit = row["deterministic_audit"]
        deterministic = (
            f"missing={len(audit['missing_required_lexical_signals'])}; "
            f"critical={','.join(audit['critical_flags']) or 'none'}"
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    markdown_cell(row["scenario_id"]),
                    markdown_cell(row["split"]),
                    markdown_cell(row["severity_class"]),
                    markdown_cell(row["raw_output"]),
                    markdown_cell(
                        f"{row['task_accuracy_ratings']} -> {row['final_task_accuracy']}"
                    ),
                    markdown_cell(
                        f"{row['contextual_grounding_ratings']} -> "
                        f"{row['final_contextual_grounding']}"
                    ),
                    markdown_cell(
                        f"{row['primary_failure_mode_ratings']} -> "
                        f"{row['final_primary_failure_mode']}"
                    ),
                    markdown_cell(deterministic),
                    markdown_cell(", ".join(row["human_review_reasons"])),
                )
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def add_text_block(lines: list[str], title: str, value: str) -> None:
    lines.extend((f"#### {title}", "", "```text", value, "```", ""))


def write_full_trace(
    path: Path,
    rows: list[dict[str, Any]],
    ledger: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    ledger_by_id = {entry["scenario_id"]: entry for entry in ledger}
    lines = [
        "# Week 2 Local FLAN: Full Prompt, Output, and Judge Trace",
        "",
        "> This is a human-readable rendering of the canonical JSONL evidence.",
        "> All judge scores are automated provisional local-integration results;",
        "> human adjudication is pending.",
        "",
        "## Run manifest",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Model: `{summary['model_id']}`",
        f"- Revision: `{summary['model_revision']}`",
        f"- Seed: `{summary['seed']}`",
        f"- Device/precision: `{summary['device']}` / `{summary['precision']}`",
        f"- Candidate prompt version: `{summary['prompt_spec_version']}`",
        f"- Judge prompt spec: `{summary['judge_prompt_spec_version']}`",
        "- Judge formulations: `criterion_first`, `evidence_first`, `failure_first`",
        "",
        "## Scenario traces",
        "",
    ]
    for row in rows:
        entry = ledger_by_id[row["scenario_id"]]
        judge_prompt_by_name = {
            item["formulation"]: item for item in entry["judge_prompts"]
        }
        lines.extend(
            (
                f"## {row['scenario_id']}",
                "",
                f"- Platform: `{row['platform']}`",
                f"- Split: `{row['split']}`",
                f"- Severity: `{row['severity_class']}`",
                f"- Seed: `{row['seed']}`",
                f"- Candidate prompt version: `{row['prompt_template_version']}`",
                f"- Candidate prompt SHA-256: `{row['candidate_prompt_hash']}`",
                f"- Candidate tokens: input `{row['candidate_input_tokens']}`, output `{row['candidate_output_tokens']}`, truncated `{row['candidate_input_truncated']}`",
                "",
            )
        )
        add_text_block(lines, "Scenario input stimulus", row["input_stimulus"])
        add_text_block(lines, "Actual candidate prompt", entry["candidate_prompt"])
        add_text_block(lines, "FLAN candidate output", row["raw_output"])
        lines.extend(
            (
                "#### Deterministic audit evidence",
                "",
                "```json",
                json.dumps(row["deterministic_audit"], ensure_ascii=False, indent=2),
                "```",
                "",
            )
        )

        lines.extend(("### Three judge results", ""))
        for result in row["judge_results"]:
            formulation = result["formulation"]
            rendered = judge_prompt_by_name[formulation]["rendered_prompts"]
            task = result["task_accuracy_classification"]
            grounding = result["contextual_grounding_classification"]
            failure = result["failure_mode_classification"]
            lines.extend(
                (
                    f"#### Judge: `{formulation}`",
                    "",
                    f"- Prompt version: `{result['prompt_version']}`",
                    f"- Task Accuracy: **{result['task_accuracy']}**; margin `{task['margin_to_second']}`; losses `{json.dumps(task['losses'], sort_keys=True)}`",
                    f"- Contextual Grounding: **{result['contextual_grounding']}**; margin `{grounding['margin_to_second']}`; losses `{json.dumps(grounding['losses'], sort_keys=True)}`",
                    f"- Primary Failure Mode: **{result['primary_failure_mode']}**; margin `{failure['margin_to_second']}`; losses `{json.dumps(failure['losses'], sort_keys=True)}`",
                    f"- Any judge prompt truncated: `{result['any_prompt_truncated']}`",
                    f"- Judge comment/rationale: {result['rationale']}",
                    "",
                    "<details>",
                    f"<summary>Show all rendered {formulation} judge prompts</summary>",
                    "",
                )
            )
            for prompt_name in (
                "task_accuracy",
                "contextual_grounding",
                "failure_mode",
                "rationale",
            ):
                add_text_block(
                    lines,
                    f"{formulation}: {prompt_name}",
                    rendered[prompt_name],
                )
            lines.extend(("</details>", ""))

        lines.extend(
            (
                "### Provisional aggregation and review",
                "",
                f"- Task ratings -> median: `{row['task_accuracy_ratings']}` -> **{row['final_task_accuracy']}**",
                f"- Grounding ratings -> median: `{row['contextual_grounding_ratings']}` -> **{row['final_contextual_grounding']}**",
                f"- Failure votes -> final: `{row['primary_failure_mode_ratings']}` -> **{row['final_primary_failure_mode']}**",
                f"- Score status: `{row['final_score_status']}`",
                f"- Human review required: `{row['human_review_required']}`",
                f"- Review reasons: `{row['human_review_reasons']}`",
                "",
                "---",
                "",
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if run_dir.drive.upper() == "C:":
        raise RuntimeError(f"Run artifacts must not be on C drive: {run_dir}")

    final_rows_path = run_dir / "W02_FLAN_Local_Integration_Rows.jsonl"
    summary_path = run_dir / "W02_FLAN_Local_Integration_Summary.json"
    rows = load_jsonl(final_rows_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    schema = json.loads(runner.RESULT_SCHEMA_PATH.read_text(encoding="utf-8"))
    for row in rows:
        jsonschema.validate(row, schema)

    scenario_doc = runner.load_yaml(runner.SCENARIO_PATH)
    regulation_doc = runner.load_yaml(runner.REGULATION_PATH)
    prompt_spec = runner.load_yaml(runner.PROMPT_PATH)
    judge_spec = runner.load_yaml(runner.JUDGE_PROMPT_PATH)
    scenarios = {item["scenario_id"]: item for item in scenario_doc["scenarios"]}
    regulations = {
        item["regulation_id"]: item for item in regulation_doc["regulations"]
    }

    ledger, prompt_hash_mismatches = render_prompt_ledger(
        rows, scenarios, regulations, prompt_spec, judge_spec
    )
    ledger_path = run_dir / "W02_FLAN_Rendered_Prompts.jsonl"
    ledger_path.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
            for item in ledger
        ),
        encoding="utf-8",
    )

    analysis = analyze(rows)
    analysis_path = run_dir / "W02_FLAN_Output_Analysis.json"
    analysis_path.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    evidence_path = run_dir / "W02_FLAN_All_Output_Evidence.md"
    write_evidence_table(evidence_path, rows, summary)
    full_trace_path = run_dir / "W02_FLAN_Full_Trace.md"
    write_full_trace(full_trace_path, rows, ledger, summary)

    model_dir = Path(summary["model_directory"])
    requirements_path = REPO_ROOT / "requirements.txt"
    packages = package_versions(requirements_path)
    input_paths = [
        runner.SCENARIO_PATH,
        runner.REGULATION_PATH,
        runner.RUBRIC_PATH,
        runner.PROMPT_PATH,
        runner.JUDGE_PROMPT_PATH,
        runner.CHECK_PATH,
        runner.RESULT_SCHEMA_PATH,
        runner.VALIDATOR_PATH,
        ROOT / "W02_Eval_Runner.py",
        Path(__file__).resolve(),
        ROOT / "W02_Compare_Replay.py",
        requirements_path,
    ]
    output_paths = [
        run_dir / "rows.checkpoint.jsonl",
        final_rows_path,
        run_dir / "W02_FLAN_Local_Integration_Results.csv",
        summary_path,
        run_dir / "W02_FLAN_Local_Integration_Report.md",
        ledger_path,
        analysis_path,
        evidence_path,
        full_trace_path,
    ]
    findings_path = run_dir / "W02_FLAN_Reproducibility_and_Findings.md"
    if findings_path.exists():
        output_paths.append(findings_path)
    for comparison_name in (
        "W02_FLAN_Replay_Comparison.json",
        "W02_FLAN_Replay_Comparison.md",
    ):
        comparison_path = run_dir / comparison_name
        if comparison_path.exists():
            output_paths.append(comparison_path)

    summary_hash_mismatches: dict[str, dict[str, str]] = {}
    for name, expected in summary["benchmark_artifact_sha256"].items():
        current = sha256_file(ROOT / name)
        if current != expected:
            summary_hash_mismatches[name] = {
                "recorded": expected,
                "current": current,
            }

    execution_incident: dict[str, Any]
    if summary["run_id"] == "local-flan-full-v0.2.0":
        execution_incident = {
            "occurred": True,
            "interrupted_after_complete_rows": 11,
            "failed_scenario_before_checkpoint": "SENPAI-005",
            "error": "AttributeError: 'int' object has no attribute 'lower'",
            "root_cause": "The YAML scalar 988 was parsed as an integer in a required-concept alias list.",
            "fix": "Normalize every alias with str(alias).lower() before matching.",
            "scope_assessment": (
                "The fix affects deterministic lexical normalization only; it does not "
                "change candidate generation, judge prompts, model weights, or decoding."
            ),
            "resume_result": "Rows 12-35 completed; final row count 35 with zero generation errors.",
        }
    else:
        execution_incident = {
            "occurred": False,
            "note": "No interruption is asserted by this audit for this run ID.",
        }

    manifest = {
        "manifest_id": "w02-local-flan-reproducibility-v0.1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": summary["run_id"],
        "claim_boundary": {
            "official_baseline": False,
            "allowed": summary["allowed_claim"],
            "prohibited": summary["prohibited_claim"],
        },
        "reproducibility_level": "configuration_replayable_not_bitwise_certified",
        "reproducibility_limitations": [
            "PyTorch deterministic algorithms were not explicitly enabled in the completed run.",
            "Latency and timestamps are machine-state dependent and are not expected to match.",
            "The same FLAN checkpoint generated responses and supplied all three judge formulations.",
            "The run used one deterministic generation per scenario, so seed sensitivity was not measured.",
            "Human adjudication is pending; automated medians are provisional.",
        ],
        "execution_commands": {
            "initial": (
                "D:\\Anaconda\\envs\\inGen\\python.exe "
                "phase_a_design\\W02_Eval_Runner.py --mode full "
                "--run-id local-flan-full-v0.2.0"
            ),
            "resume": (
                "D:\\Anaconda\\envs\\inGen\\python.exe "
                "phase_a_design\\W02_Eval_Runner.py --mode full "
                "--run-id local-flan-full-v0.2.0 --resume"
            ),
            "audit": (
                "D:\\Anaconda\\envs\\inGen\\python.exe "
                "phase_a_design\\W02_Audit_Reproducibility.py"
            ),
            "working_directory": str(REPO_ROOT),
        },
        "execution_incident": execution_incident,
        "runtime": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "os": platform.platform(),
            "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "torch_num_threads": runner.torch.get_num_threads(),
            "torch_num_interop_threads": runner.torch.get_num_interop_threads(),
            "torch_deterministic_algorithms_enabled": runner.torch.are_deterministic_algorithms_enabled(),
            "cuda_available": runner.torch.cuda.is_available(),
            "device": summary["device"],
            "precision": summary["precision"],
            "packages": packages,
            "runtime_paths": {
                key: os.environ.get(key)
                for key in (
                    "HF_HOME",
                    "HF_HUB_CACHE",
                    "HF_XET_CACHE",
                    "HF_ASSETS_CACHE",
                    "TORCH_HOME",
                    "TEMP",
                    "TMP",
                )
            },
        },
        "model": {
            "model_id": summary["model_id"],
            "revision": summary["model_revision"],
            "directory": str(model_dir),
            "files": [
                file_record(path, model_dir)
                for path in sorted(model_dir.iterdir())
                if path.is_file()
            ],
        },
        "run_configuration": {
            "seed": summary["seed"],
            "generation_config_hash": summary["generation_config_hash"],
            "prompt_spec_version": summary["prompt_spec_version"],
            "judge_prompt_spec_version": summary["judge_prompt_spec_version"],
            "deterministic_check_spec_version": summary[
                "deterministic_check_spec_version"
            ],
            "benchmark_version": summary["benchmark_version"],
            "regulation_version": summary["regulation_version"],
            "rubric_version": summary["rubric_version"],
        },
        "record_completeness": completeness(rows, schema),
        "integrity_checks": {
            "summary_benchmark_artifact_hash_mismatches": summary_hash_mismatches,
            "rendered_candidate_prompt_hash_mismatch_ids": prompt_hash_mismatches,
            "installed_requirement_mismatches": packages["mismatches"],
            "final_jsonl_is_authoritative": True,
            "checkpoint_role": (
                "append-only recovery state before final deterministic review-sample enrichment"
            ),
        },
        "rendered_prompt_ledger": {
            "path": str(ledger_path),
            "entries": len(ledger),
            "candidate_prompt_hash_mismatch_ids": prompt_hash_mismatches,
        },
        "input_and_code_files": [
            file_record(path, REPO_ROOT) for path in input_paths
        ],
        "output_evidence_files": [
            file_record(path, REPO_ROOT) for path in output_paths
        ],
        "manifest_self_hash": (
            "intentionally omitted because adding a self-hash changes the manifest"
        ),
    }
    manifest_path = run_dir / "W02_FLAN_Reproducibility_Manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"Rows schema-validated: {len(rows)}")
    print(f"Rendered prompt ledger entries: {len(ledger)}")
    print(f"Candidate prompt hash mismatches: {len(prompt_hash_mismatches)}")
    print(f"Requirement version mismatches: {len(packages['mismatches'])}")
    print(f"Manifest: {manifest_path}")
    print(f"Prompt ledger: {ledger_path}")
    print(f"Analysis: {analysis_path}")
    print(f"All-output evidence: {evidence_path}")
    print(f"Full trace: {full_trace_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

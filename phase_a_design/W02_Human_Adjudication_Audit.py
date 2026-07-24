"""Validate and render the first-pass human adjudication of the two frozen Week 2 runs."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

import W02_Eval_Runner as base


ROOT = Path(__file__).resolve().parent
ADJUDICATION_PATH = ROOT / "W02_Human_Adjudication.yaml"
MISTRAL_ROWS = (
    ROOT
    / "experiments"
    / "w02_mistral_pipeline"
    / "mistral-full-v0.2.1"
    / "W02_Mistral_GPU_Integration_Rows.jsonl"
)
FLAN_ROWS = (
    ROOT
    / "experiments"
    / "w02_local_flan_pipeline"
    / "local-flan-full-v0.2.0"
    / "W02_FLAN_Local_Integration_Rows.jsonl"
)
OUTPUT_DIR = ROOT / "experiments" / "w02_human_adjudication_v0.1.0"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def model_metrics(
    model_name: str,
    items: list[dict[str, Any]],
    rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    human_task = [int(item[model_name]["task_accuracy"]) for item in items]
    human_grounding = [
        int(item[model_name]["contextual_grounding"]) for item in items
    ]
    auto_task = [int(rows[item["scenario_id"]]["final_task_accuracy"]) for item in items]
    auto_grounding = [
        int(rows[item["scenario_id"]]["final_contextual_grounding"]) for item in items
    ]
    task_abs = [abs(a - h) for a, h in zip(auto_task, human_task)]
    grounding_abs = [abs(a - h) for a, h in zip(auto_grounding, human_grounding)]
    return {
        "count": len(items),
        "human_mean_task_accuracy": round(statistics.fmean(human_task), 6),
        "human_mean_contextual_grounding": round(
            statistics.fmean(human_grounding), 6
        ),
        "automated_mean_task_accuracy": round(statistics.fmean(auto_task), 6),
        "automated_mean_contextual_grounding": round(
            statistics.fmean(auto_grounding), 6
        ),
        "task_exact_agreement_count": sum(a == h for a, h in zip(auto_task, human_task)),
        "task_within_one_count": sum(abs(a - h) <= 1 for a, h in zip(auto_task, human_task)),
        "task_mae": round(statistics.fmean(task_abs), 6),
        "grounding_exact_agreement_count": sum(
            a == h for a, h in zip(auto_grounding, human_grounding)
        ),
        "grounding_mae": round(statistics.fmean(grounding_abs), 6),
        "failure_exact_agreement_count": sum(
            rows[item["scenario_id"]]["final_primary_failure_mode"]
            == item[model_name]["primary_failure_mode"]
            for item in items
        ),
        "human_failure_counts": dict(
            sorted(Counter(item[model_name]["primary_failure_mode"] for item in items).items())
        ),
        "dominant_problem_counts": dict(
            sorted(Counter(item[model_name]["dominant_problem"] for item in items).items())
        ),
    }


def main() -> int:
    adjudication = yaml.safe_load(ADJUDICATION_PATH.read_text(encoding="utf-8"))
    scenarios = base.load_yaml(base.SCENARIO_PATH)["scenarios"]
    scenario_by_id = {item["scenario_id"]: item for item in scenarios}
    items = adjudication["items"]
    ids = [item["scenario_id"] for item in items]
    if len(ids) != 35 or len(set(ids)) != 35:
        raise ValueError("Human adjudication must contain exactly 35 unique scenarios")
    if set(ids) != set(scenario_by_id):
        raise ValueError("Human adjudication scenario IDs do not match the benchmark")

    mistral_rows = {row["scenario_id"]: row for row in read_jsonl(MISTRAL_ROWS)}
    flan_rows = {row["scenario_id"]: row for row in read_jsonl(FLAN_ROWS)}
    if set(mistral_rows) != set(ids) or set(flan_rows) != set(ids):
        raise ValueError("Frozen candidate runs do not match the adjudication set")

    allowed_scores = set(range(1, 6))
    allowed_failures = {"unsafe", "hallucination", "off_policy", "refusal", "partial", "none"}
    for item in items:
        for model_name in ("mistral", "flan"):
            label = item[model_name]
            if label["task_accuracy"] not in allowed_scores:
                raise ValueError(f"Invalid task score: {item['scenario_id']} {model_name}")
            if label["contextual_grounding"] not in allowed_scores:
                raise ValueError(f"Invalid grounding score: {item['scenario_id']} {model_name}")
            if label["primary_failure_mode"] not in allowed_failures:
                raise ValueError(f"Invalid failure label: {item['scenario_id']} {model_name}")

    summary = {
        "adjudication_id": adjudication["adjudication_id"],
        "adjudication_version": str(adjudication["version"]),
        "status": adjudication["status"],
        "scenario_count": len(items),
        "source_artifact_sha256": {
            ADJUDICATION_PATH.name: base.sha256_file(ADJUDICATION_PATH),
            MISTRAL_ROWS.name: base.sha256_file(MISTRAL_ROWS),
            FLAN_ROWS.name: base.sha256_file(FLAN_ROWS),
        },
        "mistral": model_metrics("mistral", items, mistral_rows),
        "flan": model_metrics("flan", items, flan_rows),
        "interpretation": {
            "mistral": "Candidate quality is materially better than judge reliability; equal automated and human means are coincidental because large row-level errors cancel.",
            "flan": "The frozen candidate fails every scenario materially, while the same-checkpoint judge substantially inflates it.",
            "judge_gate": "Do not publish automated scores until calibrated against this file plus an independent second-human review.",
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base.json_dump(OUTPUT_DIR / "W02_Human_Adjudication_Summary.json", summary)

    lines = [
        "# Week 2 Human Adjudication and Judge Diagnosis",
        "",
        "> **Status: provisional single-reviewer gold.** A second human must independently",
        "> review every severity-5 item and a stratified sample before this becomes final gold.",
        "",
        "## Main findings",
        "",
        f"- Mistral human Task mean: `{summary['mistral']['human_mean_task_accuracy']}`; old automated mean: `{summary['mistral']['automated_mean_task_accuracy']}`.",
        f"- Mistral row-level Task exact agreement: `{summary['mistral']['task_exact_agreement_count']}/35`; MAE: `{summary['mistral']['task_mae']}`.",
        f"- Mistral failure-label exact agreement: `{summary['mistral']['failure_exact_agreement_count']}/35`.",
        f"- FLAN human Task mean: `{summary['flan']['human_mean_task_accuracy']}`; old automated mean: `{summary['flan']['automated_mean_task_accuracy']}`.",
        f"- FLAN row-level Task exact agreement: `{summary['flan']['task_exact_agreement_count']}/35`; MAE: `{summary['flan']['task_mae']}`.",
        f"- FLAN failure-label exact agreement: `{summary['flan']['failure_exact_agreement_count']}/35`.",
        "- The matching Mistral automated/human aggregate Task mean is accidental: opposite row-level errors cancel.",
        "- The old Judge often treated a correct safety refusal as a failure and treated scenario risk as response-caused risk.",
        "- The old FLAN Judge substantially over-scored empathy-only, echo, and non-answer outputs.",
        "",
        "## Data-use warning",
        "",
        "The original seven held-out items have now been viewed during audit. They remain",
        "useful frozen regression cases but are no longer blind prompt-selection evidence.",
        "Prompt screening must use development items only; a future unbiased claim requires",
        "a newly authored and sealed held-out set.",
        "",
        "## Row-by-row evidence",
        "",
    ]

    for item in items:
        scenario_id = item["scenario_id"]
        scenario = scenario_by_id[scenario_id]
        lines.extend(
            [
                f"### {scenario_id} — {scenario['title']}",
                "",
                f"- Split / severity: `{scenario['split']}` / `{scenario['severity_class']}`",
                f"- Scenario: {scenario['input_stimulus']}",
                f"- Expected: {'; '.join(scenario['expected_behavior_range'])}",
                "",
            ]
        )
        for model_name, display_name, rows in (
            ("mistral", "Mistral-7B-Instruct-v0.2", mistral_rows),
            ("flan", "FLAN-T5-base", flan_rows),
        ):
            row = rows[scenario_id]
            human = item[model_name]
            lines.extend(
                [
                    f"#### {display_name}",
                    "",
                    "**Frozen output**",
                    "",
                    "```text",
                    row["raw_output"],
                    "```",
                    "",
                    f"- Old automated Task/Grounding/Failure: `{row['final_task_accuracy']}` / `{row['final_contextual_grounding']}` / `{row['final_primary_failure_mode']}`",
                    f"- Human Task/Grounding/Failure: `{human['task_accuracy']}` / `{human['contextual_grounding']}` / `{human['primary_failure_mode']}`",
                    f"- Dominant problem: `{human['dominant_problem']}`",
                    f"- Human rationale: {human['rationale']}",
                    f"- Output SHA-256: `{base.sha256_text(row['raw_output'])}`",
                    "",
                ]
            )

    report_path = OUTPUT_DIR / "W02_Human_Adjudication_and_Judge_Diagnosis.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Validated 35 scenarios and wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

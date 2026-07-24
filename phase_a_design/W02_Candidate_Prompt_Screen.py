"""Run a development-only FLAN candidate-prompt screen without an automated judge."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

import W02_Eval_Runner as base


ROOT = Path(__file__).resolve().parent
VARIANT_PATH = ROOT / "W02_Candidate_Prompt_Variants.yaml"
DEFAULT_OUTPUT_ROOT = ROOT / "experiments" / "w02_candidate_prompt_screen"
RUNNER_VERSION = "0.1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=base.DEFAULT_MODEL_DIR)
    parser.add_argument("--model-revision")
    parser.add_argument("--run-id")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--variant", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    base.assert_non_c_path(model_dir, "model_dir")
    variant_doc = base.load_yaml(VARIANT_PATH)
    scenario_doc = base.load_yaml(base.SCENARIO_PATH)
    scenario_by_id = {
        item["scenario_id"]: item for item in scenario_doc["scenarios"]
    }
    screen_ids = list(variant_doc["development_screen_scenarios"])
    if any(scenario_by_id[item]["split"] != "development" for item in screen_ids):
        raise ValueError("Prompt selection may use development scenarios only")

    variants: dict[str, dict[str, Any]] = variant_doc["variants"]
    selected_names = args.variant or list(variants)
    unknown = set(selected_names) - set(variants)
    if unknown:
        raise ValueError(f"Unknown prompt variants: {sorted(unknown)}")

    revision = base.read_model_revision(model_dir, args.model_revision)
    run_id = args.run_id or f"flan-prompt-screen-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    run_dir = args.output_root.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    generation = base.load_yaml(base.PROMPT_PATH)["generation"]
    seed = int(generation["seed"])
    torch.manual_seed(seed)
    engine = base.LocalFlanEngine(model_dir, int(generation["max_input_tokens"]))

    rows: list[dict[str, Any]] = []
    for scenario_id in screen_ids:
        scenario = scenario_by_id[scenario_id]
        for variant_name in selected_names:
            template = variants[variant_name]["template"]
            prompt = template.format(
                platform=scenario["platform"],
                response_mode=scenario["response_mode"],
                input_stimulus=scenario["input_stimulus"],
            )
            print(f"{scenario_id}: {variant_name}", flush=True)
            result = engine.generate(prompt, int(generation["max_new_tokens"]))
            rows.append(
                {
                    "run_id": run_id,
                    "timestamp_utc": base.utc_now(),
                    "runner_version": RUNNER_VERSION,
                    "benchmark_version": str(scenario_doc["benchmark_version"]),
                    "scenario_id": scenario_id,
                    "split": scenario["split"],
                    "platform": scenario["platform"],
                    "severity_class": scenario["severity_class"],
                    "model_id": "google/flan-t5-base",
                    "model_revision": revision,
                    "seed": seed,
                    "do_sample": False,
                    "prompt_variant": variant_name,
                    "prompt_variant_spec_version": str(variant_doc["version"]),
                    "prompt": prompt,
                    "prompt_sha256": base.sha256_text(prompt),
                    "input_tokens": result["input_tokens"],
                    "untruncated_input_tokens": result["untruncated_input_tokens"],
                    "input_truncated": result["input_truncated"],
                    "raw_output": result["text"],
                    "output_sha256": base.sha256_text(result["text"]),
                    "output_tokens": result["output_tokens"],
                    "latency_ms": result["latency_ms"],
                    "human_task_accuracy": None,
                    "human_contextual_grounding": None,
                    "human_primary_failure_mode": None,
                    "human_rationale": None,
                }
            )

    jsonl_path = run_dir / "W02_Candidate_Prompt_Screen_Rows.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "run_id": run_id,
        "created_at_utc": base.utc_now(),
        "runner_version": RUNNER_VERSION,
        "model_id": "google/flan-t5-base",
        "model_revision": revision,
        "seed": seed,
        "do_sample": False,
        "scenario_ids": screen_ids,
        "prompt_variants": selected_names,
        "row_count": len(rows),
        "artifact_sha256": {
            base.SCENARIO_PATH.name: base.sha256_file(base.SCENARIO_PATH),
            VARIANT_PATH.name: base.sha256_file(VARIANT_PATH),
            Path(__file__).name: base.sha256_file(Path(__file__)),
        },
        "selection_guard": "development_only_no_automated_judge",
        "rows_jsonl": str(jsonl_path),
    }
    base.json_dump(run_dir / "W02_Candidate_Prompt_Screen_Manifest.json", manifest)

    lines = [
        "# Week 2 Candidate Prompt Screen — Raw Evidence",
        "",
        "> Development-only diagnostic. No automated judge score is used to select a prompt.",
        "",
        f"- Run: `{run_id}`",
        f"- Model: `google/flan-t5-base` at `{revision}`",
        f"- Seed / decoding: `{seed}` / greedy",
        f"- Rows: `{len(rows)}`",
        "",
    ]
    for scenario_id in screen_ids:
        scenario = scenario_by_id[scenario_id]
        lines.extend(
            [
                f"## {scenario_id} — {scenario['title']}",
                "",
                f"**Scenario:** {scenario['input_stimulus']}",
                "",
            ]
        )
        for row in [item for item in rows if item["scenario_id"] == scenario_id]:
            lines.extend(
                [
                    f"### `{row['prompt_variant']}`",
                    "",
                    "**Exact prompt**",
                    "",
                    "```text",
                    row["prompt"],
                    "```",
                    "",
                    "**Raw output**",
                    "",
                    "```text",
                    row["raw_output"],
                    "```",
                    "",
                ]
            )
    (run_dir / "W02_Candidate_Prompt_Screen_Raw.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"Wrote {jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

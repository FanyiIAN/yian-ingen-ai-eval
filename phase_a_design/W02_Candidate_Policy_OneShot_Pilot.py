"""Run a candidate-only FLAN pilot for the product-policy one-shot prompt."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

import W02_Eval_Runner as base


ROOT = Path(__file__).resolve().parent
DEFAULT_PROMPT = ROOT / "W02_Prompt_Spec_v0.4.0.yaml"
DEFAULT_OUTPUT_ROOT = ROOT / "experiments" / "w02_candidate_prompt_screen"
RUNNER_VERSION = "0.1.0"
DEFAULT_IDS = (
    "FARI-001",
    "FARI-002",
    "FARI-003",
    "SENPAI-006",
    "SENTINEL-006",
    "ROVER-002",
    "HUMANOID-001",
    "HUMANOID-004",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=base.DEFAULT_MODEL_DIR)
    parser.add_argument("--model-revision")
    parser.add_argument("--prompt-spec", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--scenario-id", action="append", default=[])
    parser.add_argument("--run-id")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    prompt_path = args.prompt_spec.resolve()
    output_root = args.output_root.resolve()
    for value, label in (
        (model_dir, "model_dir"),
        (prompt_path, "prompt_spec"),
        (output_root, "output_root"),
    ):
        base.assert_non_c_path(value, label)

    scenario_doc = base.load_yaml(base.SCENARIO_PATH)
    prompt_spec = base.load_yaml(prompt_path)
    by_id = {item["scenario_id"]: item for item in scenario_doc["scenarios"]}
    selected_ids = args.scenario_id or list(DEFAULT_IDS)
    missing = set(selected_ids) - set(by_id)
    if missing:
        raise ValueError(f"Unknown scenario IDs: {sorted(missing)}")

    revision = base.read_model_revision(model_dir, args.model_revision)
    generation = prompt_spec["generation"]
    seed = int(generation["seed"])
    torch.manual_seed(seed)
    engine = base.LocalFlanEngine(model_dir, int(generation["max_input_tokens"]))

    run_id = args.run_id or (
        f"flan-policy-oneshot-pilot-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    )
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    rows = []
    for index, scenario_id in enumerate(selected_ids, start=1):
        scenario = by_id[scenario_id]
        prompt = base.candidate_prompt(scenario, prompt_spec)
        print(f"[{index}/{len(selected_ids)}] {scenario_id}", flush=True)
        generated = engine.generate(prompt, int(generation["max_new_tokens"]))
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
                "prompt_spec_version": str(prompt_spec["version"]),
                "prompt": prompt,
                "prompt_sha256": base.sha256_text(prompt),
                "input_tokens": generated["input_tokens"],
                "untruncated_input_tokens": generated["untruncated_input_tokens"],
                "input_truncated": generated["input_truncated"],
                "raw_output": generated["text"],
                "output_sha256": base.sha256_text(generated["text"]),
                "output_tokens": generated["output_tokens"],
                "latency_ms": generated["latency_ms"],
            }
        )

    rows_path = run_dir / "W02_Candidate_Policy_OneShot_Rows.jsonl"
    rows_path.write_text(
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
        "decoding": "greedy",
        "scenario_ids": selected_ids,
        "prompt_spec_version": str(prompt_spec["version"]),
        "prompt_spec_sha256": base.sha256_file(prompt_path),
        "scenario_spec_sha256": base.sha256_file(base.SCENARIO_PATH),
        "rows_sha256": base.sha256_file(rows_path),
        "claim_boundary": "candidate_prompt_regression_pilot_not_blind_evaluation",
    }
    base.json_dump(run_dir / "W02_Candidate_Policy_OneShot_Manifest.json", manifest)

    lines = [
        "# Week 2 Product-policy One-shot Candidate Pilot",
        "",
        "> Regression pilot. The original held-out rows have already been inspected and are not blind.",
        "",
        f"- Run: `{run_id}`",
        f"- Model: `google/flan-t5-base` at `{revision}`",
        f"- Prompt: `{prompt_spec['version']}`",
        f"- Seed / decoding: `{seed}` / greedy",
        "",
    ]
    for row in rows:
        scenario = by_id[row["scenario_id"]]
        lines.extend(
            [
                f"## {row['scenario_id']} — {scenario['title']}",
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
                f"- Input/output tokens: `{row['input_tokens']}` / `{row['output_tokens']}`",
                f"- Prompt/output SHA-256: `{row['prompt_sha256']}` / `{row['output_sha256']}`",
                "",
            ]
        )
    (run_dir / "W02_Candidate_Policy_OneShot_Raw.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"Wrote {rows_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

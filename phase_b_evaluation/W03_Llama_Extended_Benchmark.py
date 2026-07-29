"""Run Llama-3.1-8B-Instruct over the frozen Week 2 35-scenario benchmark.

This adds the Week 3 third candidate while preserving the recorded Week 2
semantic prompt. It performs candidate inference only; the previously failed
Prometheus calibration is not silently reused as validated scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from W03_RAG_Generation import (
    LocalLlamaGenerator,
    require_frozen_revisions,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = SCRIPT_DIR / "W03_RAG_Run_Config.yaml"
DEFAULT_SCENARIOS = REPO_ROOT / "phase_a_design" / "W02_Scenarios.yaml"
DEFAULT_PROMPT_SPEC = (
    REPO_ROOT / "phase_a_design" / "W02_Prompt_Spec_v0.4.0.yaml"
)
DEFAULT_DETERMINISTIC_CHECKS = (
    REPO_ROOT / "phase_a_design" / "W02_Deterministic_Checks.yaml"
)
FROZEN_SCENARIO_SHA256 = (
    "acc34b81538598d679487acd9a46a9141634929d2d0c150894c9267df9e2a7e2"
)
FROZEN_PROMPT_SPEC_SHA256 = (
    "0bb0a6f2e298f286739080752540939454e2e5e52c0dca477e17196657cac71d"
)
RUNNER_VERSION = "0.3.0"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_text(rendered)


def week2_deterministic_audit(
    output: str,
    scenario: dict[str, Any],
    check_spec: dict[str, Any],
) -> dict[str, Any]:
    phase_a_dir = str(REPO_ROOT / "phase_a_design")
    if phase_a_dir not in sys.path:
        sys.path.insert(0, phase_a_dir)
    import W02_Eval_Runner as week2_runner

    return week2_runner.deterministic_audit(output, scenario, check_spec)


def render_candidate_prompt(
    scenario: dict[str, Any],
    prompt_spec: dict[str, Any],
) -> str:
    platform = scenario["platform"]
    template = prompt_spec["candidate_prompt"]["template"]
    return template.format(
        platform=platform,
        product_policy=prompt_spec["candidate_prompt"]["product_policies"][
            platform
        ],
        one_shot_example=prompt_spec["candidate_prompt"]["one_shot_examples"][
            platform
        ],
        input_stimulus=scenario["input_stimulus"],
    )


def validate_assets(
    scenarios: dict[str, Any],
    prompt_spec: dict[str, Any],
    config: dict[str, Any],
    scenario_path: Path,
    prompt_spec_path: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    records = scenarios.get("scenarios") or []
    if len(records) != 35 or scenarios.get("scenario_count") != 35:
        errors.append("The frozen benchmark must contain exactly 35 scenarios")
    if str(scenarios.get("benchmark_version")) != "0.2.0":
        errors.append("Unexpected scenario-bank version")
    if str(prompt_spec.get("version")) != "0.4.0":
        errors.append("Unexpected candidate prompt version")
    scenario_sha256 = sha256_file(scenario_path)
    prompt_spec_sha256 = sha256_file(prompt_spec_path)
    if scenario_sha256 != FROZEN_SCENARIO_SHA256:
        errors.append("Scenario-bank SHA-256 differs from the frozen Week 2 asset")
    if prompt_spec_sha256 != FROZEN_PROMPT_SPEC_SHA256:
        errors.append("Prompt-spec SHA-256 differs from the frozen Week 2 asset")
    if config["extended_benchmark"]["scenario_count"] != 35:
        errors.append("Run config scenario_count must be 35")
    if (
        config["generation"]["candidate_model_id"]
        != "meta-llama/Llama-3.1-8B-Instruct"
    ):
        errors.append("Unexpected Week 3 candidate model")
    ids = [record.get("scenario_id") for record in records]
    if None in ids or len(ids) != len(set(ids)):
        errors.append("Scenario IDs are missing or duplicated")
    rendered = [render_candidate_prompt(record, prompt_spec) for record in records]
    if any(not prompt.strip() for prompt in rendered):
        errors.append("A candidate prompt rendered empty")
    if errors:
        raise ValueError("Extended-benchmark validation failed:\n- " + "\n- ".join(errors))
    return {
        "status": "ok",
        "scenario_count": len(records),
        "platform_count": len({record["platform"] for record in records}),
        "candidate_model_id": config["generation"]["candidate_model_id"],
        "candidate_model_revision": config["generation"][
            "candidate_model_revision"
        ],
        "prompt_version": str(prompt_spec["version"]),
        "scenario_sha256": scenario_sha256,
        "prompt_spec_sha256": prompt_spec_sha256,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            rows.append(row)
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        handle.flush()


def ensure_private_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise ValueError("Raw candidate outputs must be outside the public repository")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def run_candidates(
    scenarios: list[dict[str, Any]],
    prompt_spec: dict[str, Any],
    config: dict[str, Any],
    model_dir: Path,
    output: Path,
    run_id: str,
    resume: bool,
    limit: int | None,
    prompt_spec_path: Path,
    check_spec: dict[str, Any],
) -> dict[str, Any]:
    require_frozen_revisions(config)
    if output.exists() and not resume:
        raise FileExistsError(f"{output} exists; use --resume or a new path")
    existing = read_jsonl(output) if resume else []
    completed = {
        row["scenario_id"]
        for row in existing
        if row.get("status") == "completed"
    }
    pending = [
        scenario
        for scenario in scenarios
        if scenario["scenario_id"] not in completed
    ]
    if limit is not None:
        pending = pending[:limit]

    generation_config = config["generation"]
    benchmark_generation = prompt_spec["generation"]
    engine = LocalLlamaGenerator(
        model_dir=model_dir,
        model_id=generation_config["candidate_model_id"],
        model_revision=generation_config["candidate_model_revision"],
        tokenizer_revision=generation_config["tokenizer_revision"],
        seed=config["random_seed"],
        max_input_tokens=int(benchmark_generation["max_input_tokens"]),
    )
    runtime = engine.runtime_metadata()
    for index, scenario in enumerate(pending, start=1):
        print(
            f"[{index}/{len(pending)}] {scenario['scenario_id']}",
            flush=True,
        )
        semantic_prompt = render_candidate_prompt(scenario, prompt_spec)
        messages = [{"role": "user", "content": semantic_prompt}]
        result = engine.generate(
            messages,
            int(benchmark_generation["max_new_tokens"]),
        )
        completed_at_utc = datetime.now(timezone.utc).isoformat()
        audit = week2_deterministic_audit(
            result["candidate_output"],
            scenario,
            check_spec,
        )
        row = {
            "status": "completed",
            "run_id": run_id,
            "runner_version": RUNNER_VERSION,
            "timestamp_utc": completed_at_utc,
            "completed_at_utc": completed_at_utc,
            "candidate_item_id": f"llama31_8b::{scenario['scenario_id']}",
            "scenario_id": scenario["scenario_id"],
            "split": scenario["split"],
            "platform": scenario["platform"],
            "severity_class": int(scenario["severity_class"]),
            "candidate_model_key": "llama31_8b",
            "candidate_model_id": generation_config["candidate_model_id"],
            "candidate_model_revision": generation_config[
                "candidate_model_revision"
            ],
            "tokenizer_revision": generation_config["tokenizer_revision"],
            "candidate_prompt_version": str(prompt_spec["version"]),
            "candidate_prompt_spec": prompt_spec_path.name,
            "candidate_prompt_spec_sha256": sha256_file(prompt_spec_path),
            "candidate_prompt": semantic_prompt,
            "candidate_prompt_sha256": sha256_text(semantic_prompt),
            "candidate_serialization": "official_llama_chat_template",
            "candidate_generation": {
                "seed": config["random_seed"],
                "do_sample": False,
                "max_input_tokens": int(
                    benchmark_generation["max_input_tokens"]
                ),
                "max_new_tokens": int(
                    benchmark_generation["max_new_tokens"]
                ),
            },
            "input_stimulus": scenario["input_stimulus"],
            "raw_output": result["candidate_output"],
            "candidate_output": result["candidate_output"],
            "candidate_output_sha256": result["candidate_output_sha256"],
            "candidate_input_tokens": result["input_tokens"],
            "candidate_untruncated_input_tokens": result["input_tokens"],
            "candidate_input_truncated": False,
            "candidate_output_tokens": result["output_tokens"],
            "candidate_latency_ms": result["generation_latency_ms"],
            "candidate_output_tokens_per_second": result[
                "output_tokens_per_second"
            ],
            "candidate_model_directory": str(model_dir),
            "candidate_device": runtime["device"],
            "candidate_precision": runtime["precision"],
            "deterministic_audit": audit,
            "generation_error": None,
            "runtime": runtime,
            "score_status": "pending_independent_evaluation_and_human_review",
        }
        row["candidate_row_sha256"] = canonical_sha256(row)
        append_jsonl(output, row)

    all_rows = read_jsonl(output)
    return {
        "status": "completed",
        "output": str(output),
        "rows_before_run": len(existing),
        "rows_generated_this_run": len(pending),
        "rows_in_output": len(all_rows),
        "expected_full_rows": 35,
        "runtime": runtime,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--prompt-spec", type=Path, default=DEFAULT_PROMPT_SPEC)
    parser.add_argument(
        "--deterministic-checks",
        type=Path,
        default=DEFAULT_DETERMINISTIC_CHECKS,
    )
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--run-id",
        default="w03-llama31-8b-35-scenario-v0.1.0",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    scenario_doc = load_yaml(args.scenarios)
    prompt_spec = load_yaml(args.prompt_spec)
    check_spec = load_yaml(args.deterministic_checks)
    validation = validate_assets(
        scenario_doc,
        prompt_spec,
        config,
        args.scenarios,
        args.prompt_spec,
    )
    if args.validate_only:
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return
    if args.output is None:
        raise ValueError("--output is required unless --validate-only is used")
    output = ensure_private_output(args.output)
    model_dir = args.model_dir or Path(
        config["generation"]["local_model_directory"]
    )
    result = run_candidates(
        scenarios=scenario_doc["scenarios"],
        prompt_spec=prompt_spec,
        config=config,
        model_dir=model_dir,
        output=output,
        run_id=args.run_id,
        resume=args.resume,
        limit=args.limit,
        prompt_spec_path=args.prompt_spec,
        check_spec=check_spec,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Run the Week 2 Mistral candidate + three-prompt-judge GPU integration pipeline.

This condition uses the same Mistral checkpoint as candidate and judge. It is useful for
the required two-model candidate comparison and for testing whether a larger instruction
model improves judge behavior, but its automated scores remain provisional because the
judge is not independent or human-calibrated.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import jsonschema
import numpy as np
import torch
import torch.nn.functional as functional
from transformers import AutoModelForCausalLM, AutoTokenizer

import W02_Eval_Runner as base


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = Path("/workspace/models/mistral_7b_instruct_v0_2")
DEFAULT_OUTPUT_ROOT = Path("/workspace/experiments/w02_mistral_pipeline")
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
RUNNER_VERSION = "0.3.0"
PRECISION = "bfloat16"
DEVICE = "cuda:0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "pilot", "full"), default="full")
    parser.add_argument("--run-id")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--prompt-spec", type=Path, default=base.PROMPT_PATH)
    parser.add_argument("--judge-prompt-spec", type=Path, default=base.JUDGE_PROMPT_PATH)
    parser.add_argument("--scenario-id", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-rationales", action="store_true")
    return parser.parse_args()


class LocalMistralEngine:
    def __init__(self, model_dir: Path, max_input_tokens: int) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the unquantized Mistral condition")
        if not (model_dir / "config.json").exists():
            raise FileNotFoundError(f"Mistral checkpoint is incomplete: {model_dir}")
        self.model_dir = model_dir
        self.max_input_tokens = max_input_tokens
        self.device = torch.device(DEVICE)
        print(f"Loading pinned Mistral checkpoint: {model_dir}")
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map={"": self.device},
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.load_seconds = time.perf_counter() - started
        allocated_gb = torch.cuda.memory_allocated(self.device) / (1024**3)
        print(
            f"Model loaded on {self.device} in {self.load_seconds:.2f}s "
            f"({allocated_gb:.2f} GiB allocated)"
        )

    def _chat_ids(self, prompt: str) -> tuple[torch.Tensor, int, bool]:
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids = rendered["input_ids"] if isinstance(rendered, Mapping) else rendered
        full_count = int(input_ids.shape[-1])
        was_truncated = full_count > self.max_input_tokens
        if was_truncated:
            input_ids = input_ids[:, : self.max_input_tokens]
        return input_ids, full_count, was_truncated

    def generate(self, prompt: str, max_new_tokens: int) -> dict[str, Any]:
        input_ids, full_count, was_truncated = self._chat_ids(prompt)
        input_ids = input_ids.to(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                input_ids=input_ids,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        torch.cuda.synchronize(self.device)
        latency_ms = (time.perf_counter() - started) * 1000
        continuation = generated[0, input_ids.shape[-1] :]
        text = self.tokenizer.decode(continuation, skip_special_tokens=True).strip()
        return {
            "text": text,
            "latency_ms": round(latency_ms, 4),
            "input_tokens": int(input_ids.shape[-1]),
            "untruncated_input_tokens": full_count,
            "input_truncated": was_truncated,
            "output_tokens": int(continuation.shape[-1]),
        }

    def classify(self, prompt: str, targets: dict[str, str]) -> dict[str, Any]:
        labels = list(targets)
        prompt_ids, full_count, was_truncated = self._chat_ids(prompt)
        prompt_ids = prompt_ids[0]
        target_ids = [
            self.tokenizer.encode(f" {targets[label]}", add_special_tokens=False)
            for label in labels
        ]
        if any(not ids for ids in target_ids):
            raise RuntimeError("A judge classification target tokenized to an empty sequence")

        sequences = [
            torch.cat((prompt_ids, torch.tensor(ids, dtype=torch.long)))
            for ids in target_ids
        ]
        max_length = max(int(sequence.shape[0]) for sequence in sequences)
        batch = torch.full(
            (len(sequences), max_length),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(batch)
        for index, sequence in enumerate(sequences):
            batch[index, : sequence.shape[0]] = sequence
            attention_mask[index, : sequence.shape[0]] = 1
        batch = batch.to(self.device)
        attention_mask = attention_mask.to(self.device)

        started = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(input_ids=batch, attention_mask=attention_mask).logits
        torch.cuda.synchronize(self.device)
        likelihood_latency_ms = (time.perf_counter() - started) * 1000

        losses: dict[str, float] = {}
        prompt_length = int(prompt_ids.shape[0])
        for index, label in enumerate(labels):
            ids = torch.tensor(target_ids[index], dtype=torch.long, device=self.device)
            prediction = logits[
                index,
                prompt_length - 1 : prompt_length - 1 + ids.shape[0],
                :,
            ]
            loss = functional.cross_entropy(prediction.float(), ids, reduction="mean")
            losses[label] = float(loss)

        ordered = sorted(losses.items(), key=lambda item: (item[1], labels.index(item[0])))
        selected = ordered[0][0]
        margin = ordered[1][1] - ordered[0][1] if len(ordered) > 1 else 0.0
        generation_started = time.perf_counter()
        generation_input = prompt_ids.unsqueeze(0).to(self.device)
        with torch.inference_mode():
            generated = self.model.generate(
                input_ids=generation_input,
                do_sample=False,
                max_new_tokens=6,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        torch.cuda.synchronize(self.device)
        generated_raw = self.tokenizer.decode(
            generated[0, generation_input.shape[-1] :], skip_special_tokens=True
        ).strip()
        normalized_generated = generated_raw.lower()
        generated_matches = []
        for label in labels:
            target_text = str(targets[label]).lower()
            pattern = rf"(?<![a-z0-9_]){re.escape(target_text)}(?![a-z0-9_])"
            if re.search(pattern, normalized_generated):
                generated_matches.append(label)
        unique_generated_matches = list(dict.fromkeys(generated_matches))
        generated_selected = (
            unique_generated_matches[0] if len(unique_generated_matches) == 1 else None
        )
        generated_parse_status = (
            "parsed"
            if len(unique_generated_matches) == 1
            else "ambiguous"
            if unique_generated_matches
            else "no_allowed_label"
        )
        return {
            "selected": selected,
            "selection_method": "mean_target_token_negative_log_likelihood",
            "losses": {key: round(value, 6) for key, value in losses.items()},
            "margin_to_second": round(margin, 6),
            "latency_ms": round(likelihood_latency_ms, 4),
            "input_tokens": prompt_length,
            "untruncated_input_tokens": full_count,
            "input_truncated": was_truncated,
            "target_prefix": "single_ascii_space",
            "generated_label_raw": generated_raw,
            "generated_label_selected": generated_selected,
            "generated_label_parse_status": generated_parse_status,
            "generated_label_latency_ms": round(
                (time.perf_counter() - generation_started) * 1000, 4
            ),
            "likelihood_generation_agree": generated_selected == selected,
        }


def rendered_judge_prompts(
    scenario: dict[str, Any],
    response: str,
    regulation_text: str,
    judge_spec: dict[str, Any],
    judges: list[dict[str, Any]],
) -> dict[str, dict[str, str]]:
    context = base.judge_context(scenario, response, regulation_text)
    rendered: dict[str, dict[str, str]] = {}
    by_name = {judge["formulation"]: judge for judge in judges}
    for name, formulation in judge_spec["formulations"].items():
        judge = by_name[name]
        rationale_context = {
            **context,
            "task_score": judge["task_accuracy"],
            "context_score": judge["contextual_grounding"],
            "failure_mode": judge["primary_failure_mode"],
        }
        rendered[name] = {
            "task_accuracy": formulation["task_accuracy_template"].format(**context),
            "contextual_grounding": formulation["contextual_grounding_template"].format(
                **context
            ),
            "failure_mode": formulation["failure_mode_template"].format(**context),
            "rationale": formulation["rationale_template"].format(**rationale_context),
        }
    return rendered


def write_mistral_report(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary["metrics"]
    lines = [
        "# Week 2 Mistral GPU Integration Report",
        "",
        "> **Automated scores are provisional.** The same Mistral checkpoint generated and",
        "> judged responses, so this run is not an independent or human-calibrated judge.",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Model: `{summary['model_id']}`",
        f"- Revision: `{summary['model_revision']}`",
        f"- Device / precision: `{summary['device']}` / `{summary['precision']}`",
        f"- Seed: `{summary['seed']}`",
        f"- Scenarios: `{summary['scenario_count']}`",
        "",
        "## Provisional metrics",
        "",
        f"- Mean Task Accuracy: `{metrics['overall']['mean_task_accuracy']}`",
        f"- Mean Contextual Grounding: `{metrics['overall']['mean_contextual_grounding']}`",
        f"- Severity-weighted quality: `{metrics['overall']['severity_weighted_quality']}`",
        f"- Task Accuracy alpha: `{metrics['judge_agreement']['task_accuracy_alpha_all']}`",
        f"- Contextual Grounding alpha: `{metrics['judge_agreement']['contextual_grounding_alpha_all']}`",
        f"- Human-review queue: `{metrics['human_review_queue']['required_count']}`",
        "",
        "## Claim boundary",
        "",
        "This run supports pipeline execution and candidate-output comparison under the",
        "frozen synthetic benchmark. It does not establish deployed-product performance,",
        "independent judge validity, or production safety.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    output_root = args.output_root.resolve()
    prompt_path = args.prompt_spec.resolve()
    judge_prompt_path = args.judge_prompt_spec.resolve()
    run_id = args.run_id or f"mistral-{args.mode}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
        raise ValueError("run-id may contain only letters, digits, dot, underscore, and hyphen")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "rows.checkpoint.jsonl"
    if checkpoint_path.exists() and not args.resume:
        raise FileExistsError(f"Run already exists; use --resume or a new run-id: {run_dir}")

    scenario_doc = base.load_yaml(base.SCENARIO_PATH)
    regulation_doc = base.load_yaml(base.REGULATION_PATH)
    rubric_doc = base.load_yaml(base.RUBRIC_PATH)
    prompt_spec = base.load_yaml(prompt_path)
    judge_spec = base.load_yaml(judge_prompt_path)
    check_spec = base.load_yaml(base.CHECK_PATH)
    result_schema = json.loads(base.RESULT_SCHEMA_PATH.read_text(encoding="utf-8"))
    selected = base.select_scenarios(scenario_doc["scenarios"], args)
    regulation_by_id = {
        item["regulation_id"]: item for item in regulation_doc["regulations"]
    }
    generation = prompt_spec["generation"]
    seed = int(generation["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    hashes = {
        path.name: base.sha256_file(path)
        for path in (
            base.SCENARIO_PATH,
            base.REGULATION_PATH,
            base.RUBRIC_PATH,
            prompt_path,
            judge_prompt_path,
            base.CHECK_PATH,
            base.RESULT_SCHEMA_PATH,
            base.VALIDATOR_PATH,
            Path(__file__),
        )
    }
    generation_config = {
        "seed": seed,
        "do_sample": False,
        "max_input_tokens": int(generation["max_input_tokens"]),
        "max_new_tokens": int(generation["max_new_tokens"]),
        "chat_template": "tokenizer.apply_chat_template(add_generation_prompt=True)",
    }
    generation_config_hash = base.sha256_text(
        json.dumps(generation_config, sort_keys=True, separators=(",", ":"))
    )
    prompt_versions = [
        item["version"] for item in judge_spec["formulations"].values()
    ]

    rows = base.load_checkpoint(checkpoint_path) if args.resume else []
    completed = {row["scenario_id"] for row in rows}
    if completed:
        print(f"Resuming with {len(completed)} completed scenarios")
    engine = LocalMistralEngine(model_dir, int(generation["max_input_tokens"]))

    for index, scenario in enumerate(selected, start=1):
        scenario_id = scenario["scenario_id"]
        if scenario_id in completed:
            continue
        print(f"[{index}/{len(selected)}] candidate + 3 judges: {scenario_id}", flush=True)
        candidate_prompt_text = base.candidate_prompt(scenario, prompt_spec)
        generation_error: str | None = None
        try:
            candidate = engine.generate(
                candidate_prompt_text, int(generation["max_new_tokens"])
            )
            raw_output = candidate["text"]
            if not raw_output:
                raise RuntimeError("empty candidate output")
        except Exception as exc:
            generation_error = f"{type(exc).__name__}: {exc}"
            candidate = {
                "text": "",
                "latency_ms": 0.0,
                "input_tokens": 0,
                "untruncated_input_tokens": 0,
                "input_truncated": False,
                "output_tokens": 0,
            }
            raw_output = ""

        audit = base.deterministic_audit(raw_output, scenario, check_spec)
        regulation_text = base.build_regulation_text(scenario, regulation_by_id)
        judge_input = raw_output or "[generation error: no response]"
        judges = base.run_judges(
            engine,
            scenario,
            judge_input,
            regulation_text,
            judge_spec,
            args.skip_rationales,
        )
        rendered = rendered_judge_prompts(
            scenario, judge_input, regulation_text, judge_spec, judges
        )
        task_ratings = [item["task_accuracy"] for item in judges]
        grounding_ratings = [item["contextual_grounding"] for item in judges]
        failure_ratings = [item["primary_failure_mode"] for item in judges]
        automated_task = base.numeric_consensus(task_ratings)
        automated_grounding = base.numeric_consensus(grounding_ratings)
        automated_failure = base.failure_consensus(failure_ratings)
        consensus_complete = (
            automated_task is not None
            and automated_grounding is not None
            and automated_failure is not None
        )

        row: dict[str, Any] = {
            "run_id": run_id,
            "timestamp_utc": base.utc_now(),
            "benchmark_version": str(scenario_doc["benchmark_version"]),
            "regulation_version": str(regulation_doc["version"]),
            "scenario_id": scenario_id,
            "split": scenario["split"],
            "platform": scenario["platform"],
            "severity_class": scenario["severity_class"],
            "severity_rationale": scenario["severity_rationale"],
            "candidate_model_id": MODEL_ID,
            "candidate_model_revision": args.model_revision,
            "tokenizer_revision": args.model_revision,
            "precision": PRECISION,
            "device": DEVICE,
            "generation_config": generation_config,
            "generation_config_hash": generation_config_hash,
            "seed": seed,
            "prompt_template_version": str(prompt_spec["version"]),
            "candidate_prompt": candidate_prompt_text,
            "candidate_prompt_hash": base.sha256_text(candidate_prompt_text),
            "candidate_input_tokens": candidate["input_tokens"],
            "candidate_untruncated_input_tokens": candidate["untruncated_input_tokens"],
            "candidate_input_truncated": candidate["input_truncated"],
            "candidate_output_tokens": candidate["output_tokens"],
            "input_stimulus": scenario["input_stimulus"],
            "raw_output": raw_output,
            "generation_latency_ms": candidate["latency_ms"],
            "generation_error": generation_error,
            "deterministic_audit": audit,
            "judge_model_id": MODEL_ID,
            "judge_model_revision": args.model_revision,
            "judge_condition": "same_checkpoint_gpu_integration_not_independent",
            "judge_prompt_versions": prompt_versions,
            "judge_rendered_prompts": rendered,
            "judge_prompt_hashes": {
                formulation: {
                    purpose: base.sha256_text(text)
                    for purpose, text in prompts.items()
                }
                for formulation, prompts in rendered.items()
            },
            "judge_results": judges,
            "task_accuracy_ratings": task_ratings,
            "contextual_grounding_ratings": grounding_ratings,
            "primary_failure_mode_ratings": failure_ratings,
            "automated_task_accuracy": automated_task,
            "automated_contextual_grounding": automated_grounding,
            "automated_primary_failure_mode": automated_failure,
            "automated_consensus_complete": consensus_complete,
            "robustness_signal": "not_tested",
            "human_review_required": False,
            "human_review_reasons": [],
            "human_task_accuracy": None,
            "human_contextual_grounding": None,
            "human_primary_failure_mode": None,
            "human_rationale": None,
            "final_task_accuracy": automated_task,
            "final_contextual_grounding": automated_grounding,
            "final_primary_failure_mode": automated_failure,
            "final_score_status": (
                "automated_consensus_provisional"
                if consensus_complete
                else "human_review_required_no_final_score"
            ),
        }
        row["human_review_reasons"] = base.initial_review_reasons(row, scenario)
        classifications = [
            judge[f"{dimension}_classification"]
            for judge in judges
            for dimension in (
                "task_accuracy",
                "contextual_grounding",
                "failure_mode",
            )
        ]
        if any(
            item["generated_label_parse_status"] == "parsed"
            and not item["likelihood_generation_agree"]
            for item in classifications
        ):
            row["human_review_reasons"].append(
                "judge_likelihood_generation_disagreement"
            )
        if any(
            item["generated_label_parse_status"] != "parsed"
            for item in classifications
        ):
            row["human_review_reasons"].append("judge_generated_label_parse_failure")
        row["human_review_reasons"] = sorted(set(row["human_review_reasons"]))
        row["human_review_required"] = bool(row["human_review_reasons"])
        jsonschema.validate(row, result_schema)
        with checkpoint_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows.append(row)

    selected_ids = {item["scenario_id"] for item in selected}
    rows = [row for row in rows if row["scenario_id"] in selected_ids]
    rows.sort(key=lambda row: row["scenario_id"])
    base.apply_stratified_review_sample(rows, seed)
    for row in rows:
        jsonschema.validate(row, result_schema)

    final_jsonl = run_dir / "W02_Mistral_GPU_Integration_Rows.jsonl"
    final_jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    csv_path = run_dir / "W02_Mistral_GPU_Integration_Results.csv"
    base.write_csv(csv_path, rows)
    metrics = base.aggregate_metrics(rows)
    summary = {
        "run_id": run_id,
        "created_at_utc": base.utc_now(),
        "run_mode": args.mode,
        "official_baseline": False,
        "allowed_claim": "GPU pipeline functionality and frozen-benchmark candidate comparison",
        "prohibited_claim": "independent judge validity or deployed-product performance",
        "benchmark_version": str(scenario_doc["benchmark_version"]),
        "regulation_version": str(regulation_doc["version"]),
        "rubric_version": str(rubric_doc["version"]),
        "prompt_spec_version": str(prompt_spec["version"]),
        "judge_prompt_spec_version": str(judge_spec["version"]),
        "deterministic_check_spec_version": str(check_spec["version"]),
        "model_id": MODEL_ID,
        "model_revision": args.model_revision,
        "runner_version": RUNNER_VERSION,
        "model_directory": str(model_dir),
        "model_load_seconds": round(engine.load_seconds, 4),
        "device": DEVICE,
        "precision": PRECISION,
        "seed": seed,
        "scenario_count": len(rows),
        "benchmark_artifact_sha256": hashes,
        "generation_config": generation_config,
        "generation_config_hash": generation_config_hash,
        "judge_selection_method": "mean_target_token_negative_log_likelihood",
        "judge_generated_cross_check_max_new_tokens": 12,
        "determinism": {
            "torch_deterministic_algorithms": "enabled_warn_only",
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "do_sample": False,
        },
        "metrics": metrics,
        "artifacts": {
            "checkpoint_jsonl": str(checkpoint_path),
            "final_jsonl": str(final_jsonl),
            "results_csv": str(csv_path),
            "summary_json": str(run_dir / "W02_Mistral_GPU_Integration_Summary.json"),
            "report_markdown": str(run_dir / "W02_Mistral_GPU_Integration_Report.md"),
        },
    }
    summary_path = run_dir / "W02_Mistral_GPU_Integration_Summary.json"
    report_path = run_dir / "W02_Mistral_GPU_Integration_Report.md"
    base.json_dump(summary_path, summary)
    write_mistral_report(report_path, summary)
    print("Mistral GPU pipeline completed successfully.")
    print(f"Rows: {len(rows)}")
    print(f"Results CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

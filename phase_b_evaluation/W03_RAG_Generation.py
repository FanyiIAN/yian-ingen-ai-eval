"""Run frozen Llama-3.1-8B-Instruct base/RAG inputs on a CUDA host.

Build inputs with W03_RAG_Pipeline.py. Before inference, freeze the exact model
and tokenizer commit SHAs in W03_RAG_Run_Config.yaml and rebuild the inputs.
Outputs are append-only JSONL so an interrupted RunPod job can resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "W03_RAG_Run_Config.yaml"
RUNNER_VERSION = "0.2.1"


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(row)
    return rows


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Run config must be a YAML mapping")
    return config


def valid_commit_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def require_frozen_revisions(config: dict[str, Any]) -> None:
    generation = config["generation"]
    if not valid_commit_sha(generation.get("candidate_model_revision")):
        raise ValueError(
            "candidate_model_revision is not frozen. After Hugging Face access "
            "is approved, resolve the model commit SHA, update the config, and "
            "rebuild run inputs."
        )
    if not valid_commit_sha(generation.get("tokenizer_revision")):
        raise ValueError("tokenizer_revision must be a 40-character commit SHA")


def validate_run_inputs(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    errors: list[str] = []
    generation = config["generation"]
    comparison = config["comparison"]
    expected_model = generation["candidate_model_id"]
    expected_revision = generation["candidate_model_revision"]
    expected_tokenizer_revision = generation["tokenizer_revision"]
    expected_seed = config["random_seed"]
    expected_conditions = set(comparison["conditions"])
    expected_rows = int(comparison["expected_generation_rows"])
    expected_items = int(comparison["expected_evaluation_items"])
    seen_ids: set[str] = set()
    conditions_by_eval: dict[str, set[str]] = defaultdict(set)

    if len(rows) != expected_rows:
        errors.append(f"Expected {expected_rows} rows, found {len(rows)}")
    for row in rows:
        run_item_id = row.get("run_item_id")
        if not run_item_id:
            errors.append("A row is missing run_item_id")
            continue
        if run_item_id in seen_ids:
            errors.append(f"Duplicate run_item_id: {run_item_id}")
        seen_ids.add(run_item_id)
        conditions_by_eval[row.get("eval_id")].add(row.get("condition"))
        if row.get("candidate_model_id") != expected_model:
            errors.append(f"{run_item_id}: candidate model mismatch")
        if row.get("candidate_model_revision") != expected_revision:
            errors.append(f"{run_item_id}: candidate revision mismatch")
        if row.get("tokenizer_revision") != expected_tokenizer_revision:
            errors.append(f"{run_item_id}: tokenizer revision mismatch")
        if row.get("random_seed") != expected_seed:
            errors.append(f"{run_item_id}: seed mismatch")
        messages = row.get("candidate_messages")
        if not isinstance(messages, list) or len(messages) != 2:
            errors.append(f"{run_item_id}: expected system and user messages")
        elif canonical_json_sha256(messages) != row.get(
            "candidate_messages_sha256"
        ):
            errors.append(f"{run_item_id}: candidate-message hash mismatch")
        contexts = row.get("retrieved_contexts") or []
        if row.get("condition") == "base" and contexts:
            errors.append(f"{run_item_id}: base condition contains contexts")
        if row.get("condition") == "rag" and not contexts:
            errors.append(f"{run_item_id}: RAG condition has no contexts")

    for eval_id, conditions in conditions_by_eval.items():
        if conditions != expected_conditions:
            errors.append(
                f"{eval_id}: expected conditions {expected_conditions}, found {conditions}"
            )
    if len(conditions_by_eval) != expected_items:
        errors.append(
            f"Expected {expected_items} paired eval IDs, found {len(conditions_by_eval)}"
        )
    if errors:
        raise ValueError("Run-input validation failed:\n- " + "\n- ".join(errors))

    return {
        "status": "ok",
        "rows": len(rows),
        "paired_eval_ids": len(conditions_by_eval),
        "condition_counts": dict(Counter(row["condition"] for row in rows)),
        "candidate_model_id": expected_model,
        "candidate_model_revision": expected_revision,
        "tokenizer_revision": expected_tokenizer_revision,
        "random_seed": expected_seed,
        "revisions_frozen": valid_commit_sha(expected_revision)
        and valid_commit_sha(expected_tokenizer_revision),
    }


def completed_run_item_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        row["run_item_id"]
        for row in read_jsonl(path)
        if row.get("status") == "completed" and row.get("run_item_id")
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        handle.flush()


def extract_input_ids(tokenized: Any) -> Any:
    """Accept both legacy tensor and Transformers 5 BatchEncoding outputs."""
    if hasattr(tokenized, "shape"):
        return tokenized
    if hasattr(tokenized, "input_ids"):
        return tokenized.input_ids
    try:
        return tokenized["input_ids"]
    except (KeyError, TypeError) as exc:
        raise TypeError(
            "apply_chat_template returned neither a tensor nor input_ids"
        ) from exc


class LocalLlamaGenerator:
    def __init__(
        self,
        model_dir: Path,
        model_id: str,
        model_revision: str,
        tokenizer_revision: str,
        seed: int,
        max_input_tokens: int,
    ) -> None:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the BF16 Llama runner")
        if not (model_dir / "config.json").exists():
            raise FileNotFoundError(f"Incomplete local checkpoint: {model_dir}")
        self.torch = torch
        self.transformers_version = transformers.__version__
        self.model_dir = model_dir
        self.model_id = model_id
        self.model_revision = model_revision
        self.tokenizer_revision = tokenizer_revision
        self.seed = seed
        self.max_input_tokens = max_input_tokens
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)

        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.load_seconds = time.perf_counter() - started

        eos_ids = [self.tokenizer.eos_token_id]
        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if (
            isinstance(eot_id, int)
            and eot_id >= 0
            and eot_id not in eos_ids
            and eot_id != self.tokenizer.unk_token_id
        ):
            eos_ids.append(eot_id)
        self.eos_token_ids = eos_ids

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int,
    ) -> dict[str, Any]:
        input_ids = extract_input_ids(
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        )
        input_tokens = int(input_ids.shape[-1])
        if input_tokens > self.max_input_tokens:
            raise ValueError(
                f"Rendered input has {input_tokens} tokens, exceeding frozen "
                f"limit {self.max_input_tokens}; do not silently truncate."
            )
        input_ids = input_ids.to(self.device)

        self.torch.cuda.synchronize()
        started = time.perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(
                input_ids=input_ids,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.eos_token_ids,
            )
        self.torch.cuda.synchronize()
        elapsed_seconds = time.perf_counter() - started
        continuation = generated[0, input_ids.shape[-1] :]
        text = self.tokenizer.decode(
            continuation, skip_special_tokens=True
        ).strip()
        output_tokens = int(continuation.shape[-1])
        return {
            "candidate_output": text,
            "candidate_output_sha256": hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest(),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "generation_latency_ms": round(elapsed_seconds * 1000, 3),
            "output_tokens_per_second": round(
                output_tokens / elapsed_seconds if elapsed_seconds else 0.0,
                6,
            ),
        }

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "model_dir": str(self.model_dir),
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "precision": "bfloat16",
            "device": str(self.device),
            "gpu_name": self.torch.cuda.get_device_name(0),
            "torch_version": self.torch.__version__,
            "transformers_version": self.transformers_version,
            "cuda_version": self.torch.version.cuda,
            "model_load_seconds": round(self.load_seconds, 6),
        }


def run_generation(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    model_dir: Path,
    output_path: Path,
    resume: bool,
    limit: int | None,
) -> dict[str, Any]:
    require_frozen_revisions(config)
    if output_path.exists() and not resume:
        raise FileExistsError(
            f"{output_path} exists; use --resume or choose a new output"
        )
    done = completed_run_item_ids(output_path) if resume else set()
    pending = [row for row in rows if row["run_item_id"] not in done]
    if limit is not None:
        pending = pending[:limit]

    generation = config["generation"]
    decoding = generation["decoding"]
    generator = LocalLlamaGenerator(
        model_dir=model_dir,
        model_id=generation["candidate_model_id"],
        model_revision=generation["candidate_model_revision"],
        tokenizer_revision=generation["tokenizer_revision"],
        seed=config["random_seed"],
        max_input_tokens=int(generation["max_input_tokens"]),
    )
    runtime = generator.runtime_metadata()
    for index, row in enumerate(pending, start=1):
        print(f"[{index}/{len(pending)}] {row['run_item_id']}", flush=True)
        generated = generator.generate(
            row["candidate_messages"],
            int(decoding["max_new_tokens"]),
        )
        result = dict(row)
        result.update(generated)
        result.update(
            {
                "status": "completed",
                "runner_version": RUNNER_VERSION,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "runtime": runtime,
                "decoding": decoding,
            }
        )
        append_jsonl(output_path, result)

    all_completed = read_jsonl(output_path) if output_path.exists() else []
    return {
        "status": "completed",
        "output_path": str(output_path),
        "rows_before_run": len(done),
        "rows_generated_this_run": len(pending),
        "rows_in_output": len(all_completed),
        "runtime": runtime,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    config = load_config(args.config)
    validation = validate_run_inputs(rows, config)
    if args.validate_only:
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return
    if args.output is None:
        raise ValueError("--output is required unless --validate-only is used")
    model_dir = args.model_dir or Path(
        config["generation"]["local_model_directory"]
    )
    result = run_generation(
        rows=rows,
        config=config,
        model_dir=model_dir,
        output_path=args.output,
        resume=args.resume,
        limit=args.limit,
    )
    result["host"] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

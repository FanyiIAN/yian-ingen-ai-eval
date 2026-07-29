"""Run identical Week 3 base/RAG inputs on causal-chat and seq2seq models.

The semantic prompt and retrieved contexts remain identical across candidates.
Only the model-required serialization changes: native chat for Llama, a folded
single instruction for Mistral v0.2, and text-to-text input for FLAN-T5.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from W03_RAG_Generation import (
    append_jsonl,
    completed_run_item_ids,
    load_config,
    read_jsonl,
    require_frozen_revisions,
    validate_run_inputs,
)


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

RUNNER_VERSION = "0.1.1"
ADAPTERS = {"native_chat", "fold_system_into_user", "seq2seq_text"}


def adapt_messages(
    messages: list[dict[str, str]],
    runtime_adapter: str,
) -> list[dict[str, str]] | str:
    if runtime_adapter not in ADAPTERS:
        raise ValueError(f"Unsupported runtime adapter: {runtime_adapter}")
    if len(messages) != 2:
        raise ValueError("Expected exactly one system and one user message")
    if messages[0].get("role") != "system" or messages[1].get("role") != "user":
        raise ValueError("Expected system then user messages")
    system = messages[0]["content"].strip()
    user = messages[1]["content"].strip()
    if runtime_adapter == "native_chat":
        return messages
    folded = (
        "SYSTEM INSTRUCTIONS\n"
        f"{system}\n\n"
        "USER INPUT\n"
        f"{user}"
    )
    if runtime_adapter == "fold_system_into_user":
        return [{"role": "user", "content": folded}]
    # FLAN-T5 is instruction-tuned as a text-to-text model, not a chat model.
    # Put the executable task before the long evidence policy and close with an
    # explicit answer cue.  The semantic policy and user input are unchanged;
    # only the model-required serialization differs.
    return (
        "TASK\n"
        "Answer the question by following every instruction below. Return the "
        "final answer itself, not an instruction heading or field name.\n\n"
        f"{folded}\n\n"
        "FINAL ANSWER\n"
    )


class LocalMultiModelGenerator:
    def __init__(
        self,
        model_dir: Path,
        generation: dict[str, Any],
        seed: int,
    ) -> None:
        import torch
        import transformers
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the frozen GPU comparison")
        if not (model_dir / "config.json").exists():
            raise FileNotFoundError(f"Incomplete local checkpoint: {model_dir}")
        self.torch = torch
        self.transformers_version = transformers.__version__
        self.model_dir = model_dir
        self.model_id = generation["candidate_model_id"]
        self.model_revision = generation["candidate_model_revision"]
        self.tokenizer_revision = generation["tokenizer_revision"]
        self.runtime_adapter = generation["runtime_adapter"]
        self.max_input_tokens = int(generation["max_input_tokens"])
        self.precision = generation.get("precision", "bfloat16")
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(self.precision)
        if dtype is None:
            raise ValueError(f"Unsupported precision: {self.precision}")

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
        model_class = (
            AutoModelForSeq2SeqLM
            if self.runtime_adapter == "seq2seq_text"
            else AutoModelForCausalLM
        )
        self.model = model_class.from_pretrained(
            model_dir,
            local_files_only=True,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.load_seconds = time.perf_counter() - started

        eos_ids = [self.tokenizer.eos_token_id]
        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if (
            self.runtime_adapter != "seq2seq_text"
            and isinstance(eot_id, int)
            and eot_id >= 0
            and eot_id not in eos_ids
            and eot_id != self.tokenizer.unk_token_id
        ):
            eos_ids.append(eot_id)
        self.eos_token_ids = eos_ids

    def _tokenize(self, messages: list[dict[str, str]]) -> Any:
        adapted = adapt_messages(messages, self.runtime_adapter)
        if isinstance(adapted, str):
            return self.tokenizer(
                adapted,
                return_tensors="pt",
                add_special_tokens=True,
            )["input_ids"]
        tokenized = self.tokenizer.apply_chat_template(
            adapted,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        return (
            tokenized.input_ids
            if hasattr(tokenized, "input_ids")
            else tokenized["input_ids"]
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int,
    ) -> dict[str, Any]:
        input_ids = self._tokenize(messages)
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
                eos_token_id=(
                    self.tokenizer.eos_token_id
                    if self.runtime_adapter == "seq2seq_text"
                    else self.eos_token_ids
                ),
            )
        self.torch.cuda.synchronize()
        elapsed_seconds = time.perf_counter() - started
        continuation = (
            generated[0]
            if self.runtime_adapter == "seq2seq_text"
            else generated[0, input_ids.shape[-1] :]
        )
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
            "runtime_adapter": self.runtime_adapter,
            "precision": self.precision,
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
    generator = LocalMultiModelGenerator(
        model_dir=model_dir,
        generation=generation,
        seed=config["random_seed"],
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
    parser.add_argument("--config", type=Path, required=True)
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
    adapter = config["generation"].get("runtime_adapter")
    if adapter not in ADAPTERS:
        raise ValueError(
            f"generation.runtime_adapter must be one of {sorted(ADAPTERS)}"
        )
    if args.validate_only:
        print(
            json.dumps(
                {**validation, "runtime_adapter": adapter},
                ensure_ascii=False,
                indent=2,
            )
        )
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

"""Run the frozen Week 4 text-robustness bank on one pinned model.

The runner preserves a single append-only event per completed request. Candidate
and performance JSONL files are deterministic views of that event log, so a
resume cannot silently lose the association among input, prompt, output,
latency, resource measurements, model revision, and seed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Iterable

import yaml

from W04_Resource_Monitor import (
    RequestProfiler,
    ResourceSampler,
    environment_manifest,
    validate_jsonable,
)


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

RUNNER_VERSION = "0.1.0"
EXPECTED_INPUT_ROWS = 182
APPROVED_REVIEW_STATUSES = {
    "not_applicable_original",
    "approved_ai_assisted",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a mapping in {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    validate_jsonable(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    )
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            validate_jsonable(row)
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
    temporary.replace(path)


def candidate_model(config: dict[str, Any], model_key: str) -> dict[str, Any]:
    matches = [
        model
        for model in config.get("candidate_models", [])
        if model.get("model_key") == model_key
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one model config for {model_key}")
    model = matches[0]
    revision = str(model.get("revision", ""))
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError(f"{model_key} does not have a frozen 40-character revision")
    return model


def validate_input_bank(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    input_path: Path,
) -> dict[str, Any]:
    if len(rows) != EXPECTED_INPUT_ROWS:
        raise ValueError(f"expected {EXPECTED_INPUT_ROWS} rows, found {len(rows)}")
    required_hash = config.get("benchmark", {}).get("frozen_input_sha256")
    actual_hash = sha256_file(input_path)
    if required_hash and required_hash != actual_hash:
        raise ValueError("frozen input hash differs from run config")
    request_ids = [str(row.get("request_base_id", "")) for row in rows]
    if any(not value for value in request_ids) or len(request_ids) != len(set(request_ids)):
        raise ValueError("request_base_id values must be non-empty and unique")

    semantic = [row for row in rows if row.get("evaluation_family") == "semantic_robustness"]
    masked = [row for row in rows if row.get("evaluation_family") == "masked_input_robustness"]
    if len(semantic) != 140 or len(masked) != 42:
        raise ValueError("expected 140 semantic rows and 42 masked rows")
    for row in semantic:
        status = row.get("semantic_equivalence_review")
        if status not in APPROVED_REVIEW_STATUSES:
            raise ValueError(
                f"{row['request_base_id']} has unapproved semantic status {status!r}"
            )
        if not row.get("automated_invariants", {}).get("all_passed"):
            raise ValueError(f"{row['request_base_id']} failed an input invariant")
    return {
        "status": "validated",
        "input_sha256": actual_hash,
        "row_count": len(rows),
        "semantic_row_count": len(semantic),
        "masked_row_count": len(masked),
    }


def render_candidate_prompt(
    row: dict[str, Any],
    prompt_spec: dict[str, Any],
) -> str:
    prompt = prompt_spec["candidate_prompt"]
    platform_name = str(row["platform"])
    policies = prompt.get("product_policies", {})
    examples = prompt.get("one_shot_examples", {})
    rendered = prompt["template"].format(
        platform=platform_name,
        response_mode="concise_safe_decision",
        input_stimulus=row["input_stimulus"],
        product_policy=policies.get(
            platform_name, "Follow the universal safety and evidence rules."
        ),
        one_shot_example=examples.get(
            platform_name, "No worked example is supplied for this product."
        ),
    )
    if row["input_stimulus"] not in rendered:
        raise AssertionError("input stimulus disappeared during prompt rendering")
    return rendered


class FirstTokenProfilerStreamer:
    """Minimal Transformers streamer that timestamps, but does not decode, tokens."""

    def __init__(self, profiler: RequestProfiler) -> None:
        self.profiler = profiler
        self._initial_prompt_put_seen = False
        self.generated_put_count = 0

    def put(self, value: Any) -> None:
        # Transformers first gives a streamer the decoder/prompt input IDs and
        # then one put per generated step. That initial put is not TTFT.
        if not self._initial_prompt_put_seen:
            self._initial_prompt_put_seen = True
            return
        self.generated_put_count += 1
        if self.generated_put_count == 1:
            self.profiler.mark_first_token()

    def end(self) -> None:
        return None


class LocalTextEngine:
    """Deferred-import GPU engine shared by causal-chat and seq2seq candidates."""

    def __init__(
        self,
        model_dir: Path,
        model_config: dict[str, Any],
        generation: dict[str, Any],
        seed: int,
        *,
        gpu_index: int = 0,
    ) -> None:
        import numpy as np
        import torch
        import transformers
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the frozen Week 4 model run")
        if not (model_dir / "config.json").exists():
            raise FileNotFoundError(f"incomplete local checkpoint: {model_dir}")
        self.torch = torch
        self.transformers_version = transformers.__version__
        self.model_dir = model_dir
        self.model_config = dict(model_config)
        self.generation = generation
        self.gpu_index = gpu_index
        self.serialization = str(model_config["serialization"])
        self.max_input_tokens = int(generation["max_input_tokens"])
        self.max_new_tokens = int(generation["max_new_tokens"])
        precision = str(model_config["precision"])
        dtypes = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if precision not in dtypes:
            raise ValueError(f"unsupported precision {precision}")

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)

        load_sampler = ResourceSampler(
            sample_interval_s=0.25,
            nvidia_smi_interval_s=1.0,
            gpu_index=gpu_index,
        ).start()
        load_started = time.perf_counter()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                local_files_only=True,
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            model_class = (
                AutoModelForSeq2SeqLM
                if self.serialization == "direct_semantic_prompt"
                else AutoModelForCausalLM
            )
            self.model = model_class.from_pretrained(
                model_dir,
                local_files_only=True,
                torch_dtype=dtypes[precision],
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            self.model.eval()
            torch.cuda.synchronize(gpu_index)
        finally:
            self.model_load_resources = load_sampler.stop()
        self.model_load_ms = round((time.perf_counter() - load_started) * 1000, 6)
        self.device = next(self.model.parameters()).device

        eos_ids = [self.tokenizer.eos_token_id]
        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if (
            self.serialization != "direct_semantic_prompt"
            and isinstance(eot_id, int)
            and eot_id >= 0
            and eot_id != self.tokenizer.unk_token_id
            and eot_id not in eos_ids
        ):
            eos_ids.append(eot_id)
        self.eos_token_ids = eos_ids

    def _tokenize(self, prompt: str) -> Any:
        if self.serialization == "direct_semantic_prompt":
            return self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=True,
            )["input_ids"]
        tokenized = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
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

    def _generate_from_input_ids(
        self,
        input_ids: Any,
        profiler: RequestProfiler,
    ) -> dict[str, Any]:
        input_tokens = int(input_ids.shape[-1])
        streamer = FirstTokenProfilerStreamer(profiler)
        with profiler.stage("generation_ms"):
            self.torch.cuda.synchronize(self.gpu_index)
            with self.torch.inference_mode():
                generated = self.model.generate(
                    input_ids=input_ids,
                    do_sample=False,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=(
                        self.tokenizer.eos_token_id
                        if self.serialization == "direct_semantic_prompt"
                        else self.eos_token_ids
                    ),
                    streamer=streamer,
                )
            self.torch.cuda.synchronize(self.gpu_index)

        continuation = (
            generated[0]
            if self.serialization == "direct_semantic_prompt"
            else generated[0, input_ids.shape[-1] :]
        )
        with profiler.stage("decode_ms"):
            output = self.tokenizer.decode(
                continuation,
                skip_special_tokens=True,
            ).strip()
        output_tokens = int(continuation.shape[-1])
        return {
            "candidate_output": output,
            "candidate_output_sha256": sha256_text(output),
            "prompt_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "first_token_observed": streamer.generated_put_count > 0,
        }

    def generate(self, prompt: str, profiler: RequestProfiler) -> dict[str, Any]:
        with profiler.stage("preprocess_ms"):
            input_ids = self._tokenize(prompt)
            input_tokens = int(input_ids.shape[-1])
            if input_tokens > self.max_input_tokens:
                raise ValueError(
                    f"rendered input has {input_tokens} tokens, above frozen "
                    f"limit {self.max_input_tokens}; silent truncation is prohibited"
                )
            input_ids = input_ids.to(self.device)
        return self._generate_from_input_ids(input_ids, profiler)

    def generate_messages(
        self,
        messages: list[dict[str, str]],
        profiler: RequestProfiler,
    ) -> dict[str, Any]:
        """Generate from an exact system/user message list for online RAG.

        This path is intentionally restricted to chat candidates.  Folding
        roles into plain text would change the frozen Week 3 prompt contract.
        """

        if self.serialization == "direct_semantic_prompt":
            raise ValueError("generate_messages requires a chat-model serialization")
        with profiler.stage("preprocess_ms"):
            tokenized = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            input_ids = (
                tokenized.input_ids
                if hasattr(tokenized, "input_ids")
                else tokenized["input_ids"]
            )
            input_tokens = int(input_ids.shape[-1])
            if input_tokens > self.max_input_tokens:
                raise ValueError(
                    f"rendered input has {input_tokens} tokens, above frozen "
                    f"limit {self.max_input_tokens}; silent truncation is prohibited"
                )
            input_ids = input_ids.to(self.device)
        return self._generate_from_input_ids(input_ids, profiler)

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_config["model_id"],
            "model_revision": self.model_config["revision"],
            "model_key": self.model_config["model_key"],
            "model_directory": str(self.model_dir),
            "precision": self.model_config["precision"],
            "serialization": self.serialization,
            "device": str(self.device),
            "gpu_name": self.torch.cuda.get_device_name(self.gpu_index),
            "torch_version": self.torch.__version__,
            "transformers_version": self.transformers_version,
            "cuda_runtime": self.torch.version.cuda,
            "model_load_ms": self.model_load_ms,
            "model_load_resources": self.model_load_resources,
        }


def event_paths(run_dir: Path, model_key: str) -> dict[str, Path]:
    return {
        "events": run_dir / f"{model_key}_Events.jsonl",
        "candidates": run_dir / f"{model_key}_Candidates.jsonl",
        "traces": run_dir / f"{model_key}_Request_Traces.jsonl",
        "sessions": run_dir / f"{model_key}_Run_Sessions.jsonl",
        "manifest": run_dir / f"{model_key}_Run_Manifest.json",
    }


def materialize_views(events: list[dict[str, Any]], paths: dict[str, Path]) -> None:
    candidates: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for event in events:
        candidate = {key: value for key, value in event.items() if key != "request_profile"}
        candidates.append(candidate)
        traces.append(
            {
                "run_item_id": event["run_item_id"],
                "request_base_id": event["request_base_id"],
                "candidate_model_key": event["candidate_model_key"],
                "candidate_model_id": event["candidate_model_id"],
                "candidate_model_revision": event["candidate_model_revision"],
                "evaluation_family": event["evaluation_family"],
                "scenario_id": event["scenario_id"],
                "variant_type": event.get("variant_type"),
                "mask_ratio": event.get("mask_ratio"),
                "cold_or_warm": event["cold_or_warm"],
                "request_profile": event["request_profile"],
                "prompt_tokens": event["prompt_tokens"],
                "output_tokens": event["output_tokens"],
                "total_tokens": event["total_tokens"],
            }
        )
    write_jsonl(paths["candidates"], candidates)
    write_jsonl(paths["traces"], traces)


def run_one(
    row: dict[str, Any],
    *,
    engine: LocalTextEngine,
    prompt_spec: dict[str, Any],
    model_config: dict[str, Any],
    seed: int,
    prompt_spec_path: Path,
    cold_or_warm: str,
) -> dict[str, Any]:
    run_item_id = f"{model_config['model_key']}::{row['request_base_id']}"
    profiler = RequestProfiler(run_item_id)
    generated: dict[str, Any] | None = None
    prompt = ""
    try:
        with profiler:
            with profiler.stage("prompt_build_ms"):
                prompt = render_candidate_prompt(row, prompt_spec)
            generated = engine.generate(prompt, profiler)
    finally:
        profile = profiler.result()
    if generated is None:
        raise RuntimeError("generation returned no result")
    return {
        "run_item_id": run_item_id,
        "request_base_id": row["request_base_id"],
        "evaluation_family": row["evaluation_family"],
        "scenario_id": row["scenario_id"],
        "platform": row["platform"],
        "split": row["split"],
        "severity_class": row["severity_class"],
        "variant_type": row.get("variant_type"),
        "variant_index": row.get("variant_index"),
        "mask_ratio": row.get("mask_ratio"),
        "selected_mask_group_ids": row.get("selected_mask_group_ids"),
        "input_stimulus": row["input_stimulus"],
        "input_sha256": row["input_sha256"],
        "candidate_model_key": model_config["model_key"],
        "candidate_model_id": model_config["model_id"],
        "candidate_model_revision": model_config["revision"],
        "precision": model_config["precision"],
        "seed": seed,
        "do_sample": False,
        "candidate_prompt_spec": prompt_spec_path.name,
        "candidate_prompt_spec_sha256": sha256_file(prompt_spec_path),
        "candidate_prompt_version": str(prompt_spec["version"]),
        "candidate_prompt": prompt,
        "candidate_prompt_sha256": sha256_text(prompt),
        **generated,
        "cold_or_warm": cold_or_warm,
        "quality_status": "unscored",
        "request_profile": profile,
        "runner_version": RUNNER_VERSION,
        "completed_at_utc": utc_now(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompt-spec", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    prompt_spec = load_yaml(args.prompt_spec)
    rows = read_jsonl(args.input)
    validation = validate_input_bank(rows, config, args.input)
    model_config = candidate_model(config, args.model_key)
    if sha256_file(args.prompt_spec) != config["candidate_prompt"]["sha256"]:
        raise ValueError("candidate prompt hash differs from run config")
    summary = {
        **validation,
        "model": model_config,
        "runner_version": RUNNER_VERSION,
    }
    if args.validate_only:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    paths = event_paths(args.run_dir, args.model_key)
    existing = read_jsonl(paths["events"])
    if existing and not args.resume:
        raise FileExistsError("event log exists; pass --resume or choose a new run dir")
    existing_ids = [str(row.get("run_item_id")) for row in existing]
    if len(existing_ids) != len(set(existing_ids)):
        raise ValueError("event log contains duplicate run_item_id values")
    done = set(existing_ids)
    pending = [
        row
        for row in rows
        if f"{args.model_key}::{row['request_base_id']}" not in done
    ]
    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("--limit must be non-negative")
        pending = pending[: args.limit]

    model_dir = args.model_dir or Path(model_config["checkpoint_path_on_gpu_host"])
    generation = config["candidate_generation"]
    seed = int(generation["seed"])
    session_id = f"{args.model_key}-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    engine = LocalTextEngine(
        model_dir,
        model_config,
        generation,
        seed,
    )
    runtime = engine.runtime_metadata()

    warmup = None
    if rows and int(generation.get("warmup_items_per_model", 0)) > 0:
        warmup = run_one(
            rows[0],
            engine=engine,
            prompt_spec=prompt_spec,
            model_config=model_config,
            seed=seed,
            prompt_spec_path=args.prompt_spec,
            cold_or_warm="cold_warmup_excluded",
        )

    generated_count = 0
    for index, row in enumerate(pending, start=1):
        print(
            f"[{args.model_key} {index}/{len(pending)}] {row['request_base_id']}",
            flush=True,
        )
        event = run_one(
            row,
            engine=engine,
            prompt_spec=prompt_spec,
            model_config=model_config,
            seed=seed,
            prompt_spec_path=args.prompt_spec,
            cold_or_warm="warm_steady_state",
        )
        event["run_session_id"] = session_id
        append_jsonl(paths["events"], event)
        generated_count += 1

    events = read_jsonl(paths["events"])
    materialize_views(events, paths)
    environment = environment_manifest(model_path=model_dir)
    session = {
        "run_session_id": session_id,
        "started_from_existing_rows": len(existing),
        "generated_rows": generated_count,
        "event_rows_after_session": len(events),
        "expected_rows": len(rows),
        "completed": len(events) == len(rows),
        "input": {
            "path": str(args.input),
            "sha256": sha256_file(args.input),
        },
        "config": {
            "path": str(args.config),
            "sha256": sha256_file(args.config),
        },
        "prompt_spec": {
            "path": str(args.prompt_spec),
            "sha256": sha256_file(args.prompt_spec),
        },
        "seed": seed,
        "runtime": runtime,
        "environment": environment,
        "warmup_excluded_from_quality_and_steady_state": warmup,
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "ended_at_utc": utc_now(),
    }
    append_jsonl(paths["sessions"], session)
    manifest = {
        "run_id": args.run_dir.name,
        "model_key": args.model_key,
        "candidate_model_id": model_config["model_id"],
        "candidate_model_revision": model_config["revision"],
        "seed": seed,
        "row_count": len(events),
        "expected_row_count": len(rows),
        "status": "completed" if len(events) == len(rows) else "partial",
        "files": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path) if path.exists() else None,
            }
            for name, path in paths.items()
            if name != "manifest"
        },
        "latest_session_id": session_id,
        "updated_at_utc": utc_now(),
    }
    atomic_write_json(paths["manifest"], manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

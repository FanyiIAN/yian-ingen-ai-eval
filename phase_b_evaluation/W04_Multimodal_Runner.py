"""Run the frozen Week 4 Open Images benchmark on a pinned supported VLM."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image

from W04_Multimodal_Data import apply_condition, pixel_sha256
from W04_Resource_Monitor import RequestProfiler, ResourceSampler, environment_manifest
from W04_Text_Robustness_Runner import (
    FirstTokenProfilerStreamer,
    append_jsonl,
    atomic_write_json,
    event_paths,
    load_yaml,
    materialize_views,
    read_jsonl,
    sha256_file,
    sha256_text,
)


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

RUNNER_VERSION = "0.2.1"


def json_safe_metadata(value: Any) -> Any:
    """Normalize library-specific metadata containers before JSON logging."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): json_safe_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe_metadata(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return str(value)


def runner_architecture(config: dict[str, Any]) -> str:
    """Resolve the model adapter without changing the frozen Idefics2 config."""

    model_config = config["candidate_model"]
    explicit = str(model_config.get("runner_architecture", "")).strip().lower()
    if explicit:
        architecture = explicit
    elif model_config.get("model_key") == "idefics2_8b_chatty":
        architecture = "idefics2"
    else:
        raise ValueError("candidate_model.runner_architecture is required")
    supported = {"idefics2", "llava"}
    if architecture not in supported:
        raise ValueError(
            f"unsupported multimodal runner architecture {architecture!r}; "
            f"expected one of {sorted(supported)}"
        )
    return architecture


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def validate_inputs(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    input_path: Path,
) -> dict[str, Any]:
    benchmark = config["benchmark"]
    if sha256_file(input_path) != benchmark["input_sha256"]:
        raise ValueError("multimodal input hash differs from frozen config")
    if len(rows) != int(benchmark["input_row_count"]):
        raise ValueError("multimodal input row count differs from frozen config")
    request_ids = [str(row.get("request_base_id", "")) for row in rows]
    if any(not value for value in request_ids) or len(request_ids) != len(set(request_ids)):
        raise ValueError("request_base_id must be non-empty and unique")
    conditions: dict[str, int] = {}
    scenarios: dict[str, set[str]] = {}
    for row in rows:
        conditions[row["condition_id"]] = conditions.get(row["condition_id"], 0) + 1
        scenarios.setdefault(row["scenario_id"], set()).add(row["condition_id"])
        if row["source_license"] != "https://creativecommons.org/licenses/by/2.0/":
            raise ValueError(f"unexpected image license for {row['request_base_id']}")
        if int(row["input_width_px"]) != 768 or int(row["input_height_px"]) != 768:
            raise ValueError("all evaluated images must be 768x768")
    expected_conditions = set(benchmark["conditions"])
    if len(scenarios) != int(benchmark["scenario_count"]):
        raise ValueError("expected 20 unique multimodal scenarios")
    if any(value != expected_conditions for value in scenarios.values()):
        raise ValueError("each scenario must have all three conditions")
    if any(conditions.get(condition) != 20 for condition in expected_conditions):
        raise ValueError(f"unexpected condition counts: {conditions}")
    revision = str(config["candidate_model"]["revision"])
    if len(revision) != 40:
        raise ValueError("candidate model revision must be a full commit hash")
    architecture = runner_architecture(config)
    return {
        "status": "validated",
        "row_count": len(rows),
        "scenario_count": len(scenarios),
        "condition_counts": conditions,
        "input_sha256": sha256_file(input_path),
        "model_revision": revision,
        "runner_architecture": architecture,
    }


def resolve_image_path(row: dict[str, Any], input_path: Path) -> Path:
    candidate = (input_path.parent / row["image_path"]).resolve()
    allowed_root = input_path.parent.resolve()
    if candidate != allowed_root and allowed_root not in candidate.parents:
        raise ValueError("image path escapes the benchmark directory")
    return candidate


def load_and_perturb_image(
    row: dict[str, Any],
    input_path: Path,
) -> Image.Image:
    image_path = resolve_image_path(row, input_path)
    if sha256_file(image_path) != row["image_file_sha256"]:
        raise ValueError(f"clean image file hash mismatch: {image_path}")
    with Image.open(image_path) as image:
        clean = image.convert("RGB")
    if pixel_sha256(clean) != row["clean_pixel_sha256"]:
        raise ValueError(f"clean pixel hash mismatch: {image_path}")
    condition = {
        "condition_id": row["condition_id"],
        "perturbation_family": row["perturbation_family"],
        **row["condition_parameters"],
    }
    evaluated = apply_condition(
        clean,
        condition,
        seed=int(row["condition_seed"]),
    )
    if pixel_sha256(evaluated) != row["expected_processed_pixel_sha256"]:
        raise ValueError(f"processed pixel hash mismatch: {row['request_base_id']}")
    return evaluated


class LocalIdefics2Engine:
    def __init__(self, model_dir: Path, config: dict[str, Any]) -> None:
        import numpy as np
        import torch
        import transformers
        from transformers import AutoProcessor, Idefics2ForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the frozen multimodal run")
        if not (model_dir / "config.json").exists():
            raise FileNotFoundError(f"incomplete local VLM checkpoint: {model_dir}")
        self.torch = torch
        self.transformers_version = transformers.__version__
        self.model_dir = model_dir
        self.config = config
        self.model_config = config["candidate_model"]
        self.generation = config["generation"]
        self.gpu_index = 0
        seed = int(config["random_seed"])
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)

        load_sampler = ResourceSampler(
            sample_interval_s=0.25,
            nvidia_smi_interval_s=1.0,
            gpu_index=self.gpu_index,
        ).start()
        started = time.perf_counter()
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_dir,
                local_files_only=True,
            )
            self.model = Idefics2ForConditionalGeneration.from_pretrained(
                model_dir,
                local_files_only=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
                attn_implementation=self.model_config["attention_implementation"],
            )
            self.model.eval()
            torch.cuda.synchronize(self.gpu_index)
        finally:
            self.model_load_resources = load_sampler.stop()
        self.model_load_ms = round((time.perf_counter() - started) * 1000, 6)
        self.device = next(self.model.parameters()).device

    def generate(
        self,
        image: Image.Image,
        user_prompt: str,
        profiler: RequestProfiler,
    ) -> dict[str, Any]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
        with profiler.stage("prompt_build_ms"):
            rendered_prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
            )
        with profiler.stage("preprocess_ms"):
            inputs = self.processor(
                images=[image],
                text=rendered_prompt,
                return_tensors="pt",
            )
            input_tokens = int(inputs["input_ids"].shape[-1])
            inputs = inputs.to(self.device, dtype=self.torch.bfloat16)

        streamer = FirstTokenProfilerStreamer(profiler)
        with profiler.stage("generation_ms"):
            self.torch.cuda.synchronize(self.gpu_index)
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=int(self.generation["max_new_tokens"]),
                    streamer=streamer,
                )
            self.torch.cuda.synchronize(self.gpu_index)
        continuation = generated[:, input_tokens:]
        with profiler.stage("decode_ms"):
            output = self.processor.batch_decode(
                continuation,
                skip_special_tokens=True,
            )[0].strip()
        output_tokens = int(continuation.shape[-1])
        return {
            "candidate_output": output,
            "candidate_output_sha256": sha256_text(output),
            "rendered_model_prompt": rendered_prompt,
            "rendered_model_prompt_sha256": sha256_text(rendered_prompt),
            "prompt_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "first_token_observed": streamer.generated_put_count > 0,
        }

    def runtime_metadata(self) -> dict[str, Any]:
        image_processor = getattr(self.processor, "image_processor", None)
        return {
            "model_key": self.model_config["model_key"],
            "model_id": self.model_config["model_id"],
            "model_revision": self.model_config["revision"],
            "runner_architecture": "idefics2",
            "model_directory": str(self.model_dir),
            "precision": self.model_config["precision"],
            "attention_implementation": self.model_config["attention_implementation"],
            "device": str(self.device),
            "gpu_name": self.torch.cuda.get_device_name(self.gpu_index),
            "torch_version": self.torch.__version__,
            "transformers_version": self.transformers_version,
            "cuda_runtime": self.torch.version.cuda,
            "processor_class": type(self.processor).__name__,
            "do_image_splitting": getattr(image_processor, "do_image_splitting", None),
            "model_load_ms": self.model_load_ms,
            "model_load_resources": self.model_load_resources,
        }


class LocalLlavaEngine:
    """Native Transformers adapter for the reference-listed LLaVA architecture."""

    def __init__(self, model_dir: Path, config: dict[str, Any]) -> None:
        import numpy as np
        import torch
        import transformers
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the frozen multimodal run")
        if not (model_dir / "config.json").exists():
            raise FileNotFoundError(f"incomplete local VLM checkpoint: {model_dir}")
        self.torch = torch
        self.transformers_version = transformers.__version__
        self.model_dir = model_dir
        self.config = config
        self.model_config = config["candidate_model"]
        self.generation = config["generation"]
        self.gpu_index = 0
        seed = int(config["random_seed"])
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)

        load_sampler = ResourceSampler(
            sample_interval_s=0.25,
            nvidia_smi_interval_s=1.0,
            gpu_index=self.gpu_index,
        ).start()
        started = time.perf_counter()
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_dir,
                local_files_only=True,
            )
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_dir,
                local_files_only=True,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                attn_implementation=self.model_config["attention_implementation"],
            )
            self.model.eval()
            torch.cuda.synchronize(self.gpu_index)
        finally:
            self.model_load_resources = load_sampler.stop()
        self.model_load_ms = round((time.perf_counter() - started) * 1000, 6)
        self.device = next(self.model.parameters()).device

    def generate(
        self,
        image: Image.Image,
        user_prompt: str,
        profiler: RequestProfiler,
    ) -> dict[str, Any]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
        with profiler.stage("prompt_build_ms"):
            rendered_prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        with profiler.stage("preprocess_ms"):
            inputs = self.processor(
                images=[image],
                text=rendered_prompt,
                return_tensors="pt",
            )
            input_tokens = int(inputs["input_ids"].shape[-1])
            for key, value in inputs.items():
                if self.torch.is_floating_point(value):
                    inputs[key] = value.to(self.device, dtype=self.torch.float16)
                else:
                    inputs[key] = value.to(self.device)

        streamer = FirstTokenProfilerStreamer(profiler)
        with profiler.stage("generation_ms"):
            self.torch.cuda.synchronize(self.gpu_index)
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=int(self.generation["max_new_tokens"]),
                    streamer=streamer,
                )
            self.torch.cuda.synchronize(self.gpu_index)
        continuation = generated[:, input_tokens:]
        with profiler.stage("decode_ms"):
            output = self.processor.batch_decode(
                continuation,
                skip_special_tokens=True,
            )[0].strip()
        output_tokens = int(continuation.shape[-1])
        return {
            "candidate_output": output,
            "candidate_output_sha256": sha256_text(output),
            "rendered_model_prompt": rendered_prompt,
            "rendered_model_prompt_sha256": sha256_text(rendered_prompt),
            "prompt_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "first_token_observed": streamer.generated_put_count > 0,
        }

    def runtime_metadata(self) -> dict[str, Any]:
        image_processor = getattr(self.processor, "image_processor", None)
        return {
            "model_key": self.model_config["model_key"],
            "model_id": self.model_config["model_id"],
            "model_revision": self.model_config["revision"],
            "runner_architecture": "llava",
            "model_directory": str(self.model_dir),
            "precision": self.model_config["precision"],
            "attention_implementation": self.model_config["attention_implementation"],
            "device": str(self.device),
            "gpu_name": self.torch.cuda.get_device_name(self.gpu_index),
            "torch_version": self.torch.__version__,
            "transformers_version": self.transformers_version,
            "cuda_runtime": self.torch.version.cuda,
            "processor_class": type(self.processor).__name__,
            "image_size": json_safe_metadata(
                getattr(image_processor, "size", None)
            ),
            "model_load_ms": self.model_load_ms,
            "model_load_resources": self.model_load_resources,
        }


def build_engine(model_dir: Path, config: dict[str, Any]) -> Any:
    architecture = runner_architecture(config)
    if architecture == "idefics2":
        return LocalIdefics2Engine(model_dir, config)
    if architecture == "llava":
        return LocalLlavaEngine(model_dir, config)
    raise AssertionError(f"unreachable architecture: {architecture}")


def run_one(
    row: dict[str, Any],
    *,
    input_path: Path,
    engine: Any,
    config: dict[str, Any],
    cold_or_warm: str,
) -> dict[str, Any]:
    model_config = config["candidate_model"]
    run_item_id = f"{model_config['model_key']}::{row['request_base_id']}"
    profiler = RequestProfiler(run_item_id)
    image: Image.Image | None = None
    generated: dict[str, Any] | None = None
    try:
        with profiler:
            with profiler.stage("input_load_ms"):
                image_path = resolve_image_path(row, input_path)
                if sha256_file(image_path) != row["image_file_sha256"]:
                    raise ValueError(f"clean image file hash mismatch: {image_path}")
                with Image.open(image_path) as source:
                    clean = source.convert("RGB")
                if pixel_sha256(clean) != row["clean_pixel_sha256"]:
                    raise ValueError(f"clean pixel hash mismatch: {image_path}")
            with profiler.stage("image_perturb_ms"):
                condition = {
                    "condition_id": row["condition_id"],
                    "perturbation_family": row["perturbation_family"],
                    **row["condition_parameters"],
                }
                image = apply_condition(
                    clean,
                    condition,
                    seed=int(row["condition_seed"]),
                )
                if pixel_sha256(image) != row["expected_processed_pixel_sha256"]:
                    raise ValueError(
                        f"processed pixel hash mismatch: {row['request_base_id']}"
                    )
            generated = engine.generate(image, row["user_prompt"], profiler)
    finally:
        profile = profiler.result()
    if image is None or generated is None:
        raise RuntimeError("multimodal generation produced no result")
    return {
        "run_item_id": run_item_id,
        "request_base_id": row["request_base_id"],
        "evaluation_family": "multimodal_robustness",
        "scenario_id": row["scenario_id"],
        "platform": row["platform"],
        "condition_id": row["condition_id"],
        "perturbation_family": row["perturbation_family"],
        "condition_parameters": row["condition_parameters"],
        "condition_seed": row["condition_seed"],
        "image_path": row["image_path"],
        "image_file_sha256": row["image_file_sha256"],
        "processed_pixel_sha256": pixel_sha256(image),
        "source_image_id": row["source_image_id"],
        "source_license": row["source_license"],
        "user_prompt": row["user_prompt"],
        "user_prompt_sha256": sha256_text(row["user_prompt"]),
        "candidate_model_key": model_config["model_key"],
        "candidate_model_id": model_config["model_id"],
        "candidate_model_revision": model_config["revision"],
        "precision": model_config["precision"],
        "seed": int(config["random_seed"]),
        "do_sample": False,
        **generated,
        "cold_or_warm": cold_or_warm,
        "quality_status": "unscored",
        "retrieval": {
            "enabled": False,
            "rag_total_ms": None,
            "reason": config["performance_measurement"]["rag"]["reason"],
        },
        "request_profile": profile,
        "runner_version": RUNNER_VERSION,
        "completed_at_utc": utc_now(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    rows = read_jsonl(args.input)
    validation = validate_inputs(rows, config, args.input)
    if args.validate_only:
        print(json.dumps({**validation, "runner_version": RUNNER_VERSION}, indent=2, sort_keys=True))
        return
    model_config = config["candidate_model"]
    paths = event_paths(args.run_dir, model_config["model_key"])
    existing = read_jsonl(paths["events"])
    if existing and not args.resume:
        raise FileExistsError("event log exists; pass --resume or choose a new run dir")
    existing_ids = [str(row.get("run_item_id")) for row in existing]
    if len(existing_ids) != len(set(existing_ids)):
        raise ValueError("event log contains duplicate run_item_id")
    done = set(existing_ids)
    pending = [
        row
        for row in rows
        if f"{model_config['model_key']}::{row['request_base_id']}" not in done
    ]
    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("--limit must be non-negative")
        pending = pending[: args.limit]

    model_dir = args.model_dir or Path(model_config["checkpoint_path_on_gpu_host"])
    engine = build_engine(model_dir, config)
    runtime = engine.runtime_metadata()
    session_id = (
        f"{model_config['model_key']}-"
        f"{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    warmup = None
    if rows and int(config["generation"].get("warmup_items", 0)) > 0:
        warmup = run_one(
            rows[0],
            input_path=args.input,
            engine=engine,
            config=config,
            cold_or_warm="cold_warmup_excluded",
        )

    generated_count = 0
    for index, row in enumerate(pending, start=1):
        print(
            f"[{model_config['model_key']} {index}/{len(pending)}] "
            f"{row['request_base_id']}",
            flush=True,
        )
        event = run_one(
            row,
            input_path=args.input,
            engine=engine,
            config=config,
            cold_or_warm="warm_steady_state",
        )
        event["run_session_id"] = session_id
        append_jsonl(paths["events"], event)
        generated_count += 1

    events = read_jsonl(paths["events"])
    materialize_views(events, paths)
    session = {
        "run_session_id": session_id,
        "started_from_existing_rows": len(existing),
        "generated_rows": generated_count,
        "event_rows_after_session": len(events),
        "expected_rows": len(rows),
        "completed": len(events) == len(rows),
        "input": {"path": str(args.input), "sha256": sha256_file(args.input)},
        "config": {"path": str(args.config), "sha256": sha256_file(args.config)},
        "seed": int(config["random_seed"]),
        "runtime": runtime,
        "environment": environment_manifest(model_path=model_dir),
        "warmup_excluded_from_quality_and_steady_state": warmup,
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "ended_at_utc": utc_now(),
    }
    append_jsonl(paths["sessions"], session)
    manifest = {
        "run_id": args.run_dir.name,
        "model_key": model_config["model_key"],
        "candidate_model_id": model_config["model_id"],
        "candidate_model_revision": model_config["revision"],
        "seed": int(config["random_seed"]),
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

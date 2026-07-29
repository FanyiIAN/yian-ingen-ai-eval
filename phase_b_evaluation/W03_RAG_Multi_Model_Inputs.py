"""Expand one frozen retrieval pass into identical three-model run inputs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from W03_RAG_Generation import validate_run_inputs


EXPANDER_VERSION = "0.1.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def expand_candidate(
    rows: list[dict[str, Any]],
    master_config: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = copy.deepcopy(master_config)
    generation = config["generation"]
    for field in (
        "candidate_model_id",
        "candidate_model_revision",
        "tokenizer_revision",
        "local_model_directory",
        "runtime_adapter",
        "precision",
    ):
        generation[field] = candidate[field]
    candidate_key = candidate["candidate_model_key"]
    config["active_candidate_model_key"] = candidate_key
    config["evaluation"]["judge"]["independent_from_candidate"] = bool(
        candidate["judge_independent_from_candidate"]
    )

    expanded: list[dict[str, Any]] = []
    for source in rows:
        row = copy.deepcopy(source)
        row["run_item_id"] = f"{candidate_key}::{source['run_item_id']}"
        row["candidate_model_key"] = candidate_key
        row["candidate_model_id"] = candidate["candidate_model_id"]
        row["candidate_model_revision"] = candidate[
            "candidate_model_revision"
        ]
        row["tokenizer_revision"] = candidate["tokenizer_revision"]
        expanded.append(row)
    validate_run_inputs(expanded, config)
    return expanded, config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-input", type=Path, required=True)
    parser.add_argument("--master-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite immutable directory: {args.output_dir}"
        )
    if args.manifest.exists():
        raise FileExistsError(
            f"Refusing to overwrite immutable manifest: {args.manifest}"
        )
    with args.master_config.open("r", encoding="utf-8") as handle:
        master_config = yaml.safe_load(handle)
    rows = read_jsonl(args.shared_input)
    args.output_dir.mkdir(parents=True)

    outputs: dict[str, Any] = {}
    shared_message_hashes = {
        row["run_item_id"]: row["candidate_messages_sha256"] for row in rows
    }
    shared_context_hashes = {
        row["run_item_id"]: hashlib.sha256(
            json.dumps(
                row["retrieved_contexts"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        for row in rows
    }
    for candidate in master_config["candidate_matrix"]:
        key = candidate["candidate_model_key"]
        candidate_rows, candidate_config = expand_candidate(
            rows, master_config, candidate
        )
        input_path = args.output_dir / f"{key}_inputs.jsonl"
        config_path = args.output_dir / f"{key}_config.yaml"
        write_jsonl(input_path, candidate_rows)
        with config_path.open("x", encoding="utf-8", newline="\n") as handle:
            yaml.safe_dump(
                candidate_config,
                handle,
                sort_keys=False,
                allow_unicode=True,
            )
        outputs[key] = {
            "input_path": str(input_path),
            "input_sha256": sha256_file(input_path),
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "rows": len(candidate_rows),
            "runtime_adapter": candidate["runtime_adapter"],
            "judge_independent_from_candidate": candidate[
                "judge_independent_from_candidate"
            ],
        }
    manifest = {
        "expander_version": EXPANDER_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "shared_input": {
            "path": str(args.shared_input),
            "sha256": sha256_file(args.shared_input),
            "rows": len(rows),
        },
        "master_config": {
            "path": str(args.master_config),
            "sha256": sha256_file(args.master_config),
        },
        "semantic_prompt_identical": True,
        "shared_candidate_message_hashes": shared_message_hashes,
        "shared_retrieved_context_hashes": shared_context_hashes,
        "outputs": outputs,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "status": "ok",
                "candidates": len(outputs),
                "rows_per_candidate": len(rows),
                "total_rows": len(outputs) * len(rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

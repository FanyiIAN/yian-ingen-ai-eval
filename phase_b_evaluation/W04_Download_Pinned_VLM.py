"""Download and validate a VLM snapshot pinned by a Week 4 run config."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

import yaml


def directory_bytes(path: Path) -> int:
    return sum(
        candidate.stat().st_size
        for candidate in path.rglob("*")
        if candidate.is_file() and not candidate.is_symlink()
    )


def load_model_contract(config_path: Path) -> dict[str, str]:
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(
        document.get("candidate_model"), dict
    ):
        raise ValueError(f"invalid Week 4 VLM config: {config_path}")
    model = document["candidate_model"]
    contract = {
        "model_key": str(model.get("model_key", "")),
        "model_id": str(model.get("model_id", "")),
        "revision": str(model.get("revision", "")),
        "runner_architecture": str(model.get("runner_architecture", "")),
    }
    if not contract["model_key"] or not contract["model_id"]:
        raise ValueError("candidate model key and ID must be non-empty")
    if len(contract["revision"]) != 40:
        raise ValueError("candidate revision must be a full 40-character commit hash")
    return contract


def validate_snapshot(path: Path) -> dict[str, Any]:
    required = ["config.json", "preprocessor_config.json", "tokenizer_config.json"]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete VLM snapshot; missing {missing}")

    index_path = path / "model.safetensors.index.json"
    single_path = path / "model.safetensors"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shards = sorted(set(index["weight_map"].values()))
        missing_shards = [name for name in shards if not (path / name).is_file()]
        if missing_shards:
            raise FileNotFoundError(f"missing VLM weight shards: {missing_shards}")
    elif single_path.is_file():
        shards = [single_path.name]
    else:
        raise FileNotFoundError("snapshot has no safetensors weights or shard index")

    return {
        "required_metadata_files": required,
        "weight_shard_count": len(shards),
        "weight_shards": shards,
        "directory_bytes": directory_bytes(path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = load_model_contract(args.config)
    if not args.validate_only:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=contract["model_id"],
            revision=contract["revision"],
            local_dir=args.model_dir,
            token=os.environ.get("HF_TOKEN") or None,
        )
    validation = validate_snapshot(args.model_dir)
    manifest = {
        **contract,
        "model_dir": str(args.model_dir),
        "config_path": str(args.config),
        "validated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hf_token_present": bool(os.environ.get("HF_TOKEN")),
        "hf_token_value_recorded": False,
        "validation": validation,
        "status": "complete",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

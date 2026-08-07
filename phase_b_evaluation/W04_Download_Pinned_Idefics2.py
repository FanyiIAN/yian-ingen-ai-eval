"""Download the exact Week 4 Idefics2 snapshot without persisting a token."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any


MODEL_ID = "HuggingFaceM4/idefics2-8b-chatty"
REVISION = "8e65868b394317b973bd61db3b08e6478ebeedbf"


def directory_bytes(path: Path) -> int:
    return sum(
        candidate.stat().st_size
        for candidate in path.rglob("*")
        if candidate.is_file() and not candidate.is_symlink()
    )


def validate_snapshot(path: Path) -> dict[str, Any]:
    required = [
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
    ]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete Idefics2 snapshot; missing {missing}")
    index = json.loads(
        (path / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    shards = sorted(set(index["weight_map"].values()))
    missing_shards = [name for name in shards if not (path / name).is_file()]
    if missing_shards:
        raise FileNotFoundError(f"missing Idefics2 shards: {missing_shards}")
    return {
        "required_metadata_files": required,
        "weight_shard_count": len(shards),
        "weight_shards": shards,
        "directory_bytes": directory_bytes(path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/workspace/models/idefics2_8b_chatty"),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.validate_only:
        from huggingface_hub import snapshot_download

        # The token is read only from the process environment. It is not
        # printed, copied into the manifest, or written by this script.
        snapshot_download(
            repo_id=MODEL_ID,
            revision=REVISION,
            local_dir=args.model_dir,
            token=os.environ.get("HF_TOKEN") or None,
        )
    validation = validate_snapshot(args.model_dir)
    manifest = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "model_dir": str(args.model_dir),
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


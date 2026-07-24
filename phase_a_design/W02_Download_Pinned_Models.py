"""Download and verify the two pinned Week 2 model revisions on RunPod."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download


MODELS = {
    "mistral": {
        "repo_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "revision": "63a8b081895390a26e140280378bc85ec8bce07a",
        "directory": "mistral_7b_instruct_v0_2",
        "allow_patterns": [
            "*.json",
            "*.model",
            "*.safetensors",
        ],
    },
    "flan": {
        "repo_id": "google/flan-t5-base",
        "revision": "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2",
        "directory": "flan_t5_base",
        "allow_patterns": [
            "*.json",
            "spiece.model",
            "model.safetensors",
        ],
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_summary(path: Path) -> dict[str, Any]:
    files = sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and ".cache" not in item.relative_to(path).parts
    )
    important = {
        item.name: sha256_file(item)
        for item in files
        if item.name
        in {
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors.index.json",
        }
    }
    return {
        "file_count": len(files),
        "total_bytes": sum(item.stat().st_size for item in files),
        "important_file_sha256": important,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("/workspace/models"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/workspace/models/model_download_manifest.json"),
    )
    parser.add_argument(
        "--model",
        choices=("both", "mistral", "flan"),
        default="both",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.model_root.mkdir(parents=True, exist_ok=True)
    selected = list(MODELS) if args.model == "both" else [args.model]
    manifest: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "models": {},
    }
    for name in selected:
        spec = MODELS[name]
        destination = args.model_root / spec["directory"]
        print(
            f"Downloading {spec['repo_id']} at {spec['revision']} "
            f"to {destination}",
            flush=True,
        )
        resolved = snapshot_download(
            repo_id=spec["repo_id"],
            revision=spec["revision"],
            local_dir=destination,
            allow_patterns=spec["allow_patterns"],
            max_workers=8,
        )
        config_path = destination / "config.json"
        if not config_path.exists():
            raise RuntimeError(f"Incomplete checkpoint: {config_path} is missing")
        manifest["models"][name] = {
            **spec,
            "resolved_path": str(Path(resolved).resolve()),
            **directory_summary(destination),
        }
        args.manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
            newline="\n",
        )
        print(f"Verified {name}: {manifest['models'][name]}", flush=True)
    print(f"Manifest: {args.manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

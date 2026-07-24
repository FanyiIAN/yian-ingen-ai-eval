"""Download and verify the pinned Prometheus 2 Judge checkpoint.

Prometheus is used only as the evaluation Judge. Mistral and FLAN remain the two
candidate models required by the Week 2 plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download


MODEL = {
    "role": "rubric_judge",
    "repo_id": "prometheus-eval/prometheus-7b-v2.0",
    "revision": "66ffb1fc20beebfb60a3964a957d9011723116c5",
    "directory": "prometheus_7b_v2_0",
}

ALLOW_PATTERNS = (
    "*.json",
    "*.model",
    "*.safetensors",
)


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
    return {
        "file_count": len(files),
        "total_bytes": sum(item.stat().st_size for item in files),
        "file_sha256": {
            str(item.relative_to(path)).replace("\\", "/"): sha256_file(item)
            for item in files
        },
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
        default=Path("/workspace/models/prometheus_judge_download_manifest.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.model_root.mkdir(parents=True, exist_ok=True)
    destination = args.model_root / MODEL["directory"]
    print(
        f"Downloading {MODEL['repo_id']} at {MODEL['revision']} "
        f"to {destination}",
        flush=True,
    )
    resolved = snapshot_download(
        repo_id=MODEL["repo_id"],
        revision=MODEL["revision"],
        local_dir=destination,
        allow_patterns=ALLOW_PATTERNS,
        max_workers=8,
    )
    required = ("config.json", "tokenizer.model")
    missing = [name for name in required if not (destination / name).exists()]
    weight_files = sorted(destination.glob("*.safetensors"))
    if missing or not weight_files:
        raise RuntimeError(
            f"Incomplete Prometheus checkpoint; missing={missing}, "
            f"safetensors={len(weight_files)}"
        )
    manifest: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            **MODEL,
            "resolved_path": str(Path(resolved).resolve()),
            **directory_summary(destination),
        },
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    print(f"Verified Prometheus Judge: {manifest['model']}", flush=True)
    print(f"Manifest: {args.manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

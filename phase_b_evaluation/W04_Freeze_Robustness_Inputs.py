"""Freeze an invariant-clean Week 4 input bank after an anchored review.

The script does not decide semantic equivalence.  It applies a separately
recorded review only when the review hash, reviewed request-ID hash, counts,
approved variant types, and zero-exception declaration all match the input.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


FREEZER_VERSION = "0.1.0"
VARIANT_TYPES = {
    "synonym_substitution",
    "sentence_reordering",
    "tone_shift",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def request_ids_sha256(request_ids: list[str]) -> str:
    payload = "\n".join(sorted(request_ids)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )


def load_review(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("review record must be a mapping")
    return value


def validate_and_apply(
    rows: list[dict[str, Any]],
    review: dict[str, Any],
    source_path: Path,
) -> list[dict[str, Any]]:
    artifact = review["reviewed_artifact"]
    actual_source_hash = sha256_file(source_path)
    if artifact["sha256"] != actual_source_hash:
        raise ValueError("review is not anchored to this source JSONL hash")
    if int(artifact["total_rows"]) != len(rows):
        raise ValueError("review total_rows does not match")
    if review.get("exceptions") != []:
        raise ValueError("review has unresolved exceptions")
    if set(review.get("approved_variant_types", [])) != VARIANT_TYPES:
        raise ValueError("review does not approve exactly the required variants")

    pending = [
        row
        for row in rows
        if row.get("evaluation_family") == "semantic_robustness"
        and row.get("variant_type") in VARIANT_TYPES
    ]
    if len(pending) != int(artifact["semantic_variant_rows_reviewed"]):
        raise ValueError("reviewed semantic row count does not match")
    if any(
        not row.get("automated_invariants", {}).get("all_passed")
        for row in pending
    ):
        raise ValueError("cannot freeze an input with failed invariants")
    ids = [str(row["request_base_id"]) for row in pending]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate reviewed request ID")
    if request_ids_sha256(ids) != artifact["reviewed_request_ids_sha256"]:
        raise ValueError("reviewed request-ID hash does not match")

    review_id = str(review["review_id"])
    frozen: list[dict[str, Any]] = []
    for row in rows:
        value = dict(row)
        if value.get("variant_type") in VARIANT_TYPES:
            value["semantic_equivalence_review"] = "approved_ai_assisted"
            value["semantic_equivalence_review_id"] = review_id
            value["review_notes"] = (
                "Approved by the hash-anchored AI-assisted inspection record; "
                "not an independent human review."
            )
        frozen.append(value)
    return frozen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    review = load_review(args.review)
    frozen = validate_and_apply(rows, review, args.input)
    write_jsonl(args.output, frozen)
    statuses = collections.Counter(
        str(row.get("semantic_equivalence_review") or "not_applicable")
        for row in frozen
    )
    manifest = {
        "manifest_id": "w04_frozen_robustness_inputs_v0.1.0",
        "freezer_version": FREEZER_VERSION,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_jsonl": {
            # Keep manifests portable and avoid publishing a workstation or
            # private-run directory when the draft was supplied by absolute path.
            "path": args.input.name,
            "sha256": sha256_file(args.input),
        },
        "review_record": {
            "path": args.review.name,
            "sha256": sha256_file(args.review),
            "review_id": review["review_id"],
            "reviewer_type": review["reviewer"]["reviewer_type"],
            "human_reviewer": bool(review["reviewer"]["human_reviewer"]),
        },
        "output_jsonl": {
            "path": args.output.name,
            "sha256": sha256_file(args.output),
            "row_count": len(frozen),
        },
        "semantic_review_status_counts": dict(statuses),
        "status": "frozen_ready_for_candidate_inference",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

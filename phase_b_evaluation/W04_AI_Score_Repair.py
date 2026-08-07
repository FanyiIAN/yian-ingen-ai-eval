"""Deterministically re-normalize retained Week 4 AI-score rows.

This utility never calls a model. It preserves every originally parsed row,
re-parses failed raw JSON with the current public normalizer, and writes only
resolved rows to a new append-resumable score file. Truly unresolved rows are
retained separately for targeted re-judging.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

from W04_AI_Assisted_Scoring import (
    SCORER_VERSION,
    extract_json_object,
    normalize_multimodal_score,
    normalize_text_score,
)


REPAIR_VERSION = "0.1.1"
TEXT_FAILURE_CODE_ALIASES = {
    "omission": "partial",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def repair_rows(
    rows: list[dict[str, Any]], mode: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    score_ids = [str(row.get("score_id", "")) for row in rows]
    if any(not value for value in score_ids) or len(score_ids) != len(set(score_ids)):
        raise ValueError("score_id values must be non-empty and unique")
    normalizer = normalize_text_score if mode == "text" else normalize_multimodal_score
    resolved: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for row in rows:
        if row.get("mode") != mode:
            raise ValueError(f"row mode mismatch: {row.get('score_id')}")
        if row.get("score_status") == "parsed" and isinstance(
            row.get("normalized_score"), dict
        ):
            resolved.append(row)
            continue
        updated = dict(row)
        try:
            parsed = extract_json_object(str(row.get("raw_judge_output", "")))
            repair_actions: list[str] = []
            if mode == "text":
                raw_code = str(parsed.get("failure_code", "")).strip().lower()
                canonical_code = TEXT_FAILURE_CODE_ALIASES.get(raw_code)
                if canonical_code is not None:
                    parsed = dict(parsed)
                    parsed["failure_code"] = canonical_code
                    repair_actions.append(
                        f"failure_code_alias:{raw_code}->{canonical_code}"
                    )
            updated["normalized_score"] = normalizer(parsed)
        except (ValueError, TypeError, KeyError) as error:
            updated["repair_error"] = str(error)
            updated["repair_version"] = REPAIR_VERSION
            unresolved.append(updated)
            continue
        updated["score_status"] = "parsed"
        updated["parse_error"] = None
        updated["repair"] = {
            "repair_version": REPAIR_VERSION,
            "normalizer_scorer_version": SCORER_VERSION,
            "method": "deterministic_raw_json_renormalization_no_model_call",
            "actions": repair_actions,
            "completed_at_utc": utc_now(),
        }
        resolved.append(updated)
    return resolved, unresolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("text", "multimodal"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unresolved-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    resolved, unresolved = repair_rows(rows, args.mode)
    write_jsonl(args.output, resolved)
    write_jsonl(args.unresolved_output, unresolved)
    print(
        json.dumps(
            {
                "input_rows": len(rows),
                "resolved_rows": len(resolved),
                "unresolved_rows": len(unresolved),
                "repair_version": REPAIR_VERSION,
                "normalizer_scorer_version": SCORER_VERSION,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

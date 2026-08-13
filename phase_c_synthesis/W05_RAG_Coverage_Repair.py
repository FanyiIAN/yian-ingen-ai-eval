"""Reparse append-only coverage results after deterministic schema repairs.

This tool never calls a model and never imputes a registered rubric point. It
replays each retained Judge attempt through the current parser, preferring the
latest attempt that contains every registered point. Unregistered extra point
rows may be discarded by the parser and remain visible in the audit fields.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
PHASE_B = HERE.parent / "phase_b_evaluation"
if str(PHASE_B) not in sys.path:
    sys.path.insert(0, str(PHASE_B))

from W04_AI_Assisted_Scoring import extract_json_object
from W05_RAG_Coverage_Scoring import (
    SCORER_VERSION,
    normalize_coverage,
    parse_coverage_value,
    read_jsonl,
)


REPAIR_VERSION = "1.0.0"


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping: {path}")
    return payload


def repair_row(row: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    output = dict(row)
    if row.get("score_status") == "parsed":
        return output
    attempts = row.get("parse_attempts") or []
    audit: list[dict[str, Any]] = []
    for attempt in reversed(attempts):
        raw = str(attempt.get("raw_judge_output", ""))
        try:
            value, syntax_repairs = parse_coverage_value(raw, extract_json_object)
            normalized = normalize_coverage(value, item)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            audit.append(
                {
                    "attempt_number": attempt.get("attempt_number"),
                    "repair_status": "not_repairable",
                    "reason": str(exc),
                }
            )
            continue
        output.update(
            {
                "score_status": "parsed_after_deterministic_repair",
                "normalized_coverage": normalized,
                "parse_error": None,
                "deterministic_syntax_repairs": sorted(
                    set(row.get("deterministic_syntax_repairs") or [])
                    | set(syntax_repairs)
                ),
                "coverage_repair": {
                    "repair_version": REPAIR_VERSION,
                    "coverage_parser_version": SCORER_VERSION,
                    "source_attempt_number": attempt.get("attempt_number"),
                    "method": "discard_unregistered_extra_point_rows_only",
                    "registered_point_imputation": False,
                    "audit": audit,
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            }
        )
        return output
    output["coverage_repair"] = {
        "repair_version": REPAIR_VERSION,
        "coverage_parser_version": SCORER_VERSION,
        "method": "no_registered_point_imputation",
        "repair_status": "not_repairable",
        "audit": audit,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--eval-set", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    items = {
        str(item["eval_id"]): item
        for item in load_yaml(args.eval_set)["items"]
    }
    rows = read_jsonl(args.input)
    repaired = [repair_row(row, items[str(row["eval_id"])]) for row in rows]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in repaired:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "rows": len(repaired),
                "parsed": sum(
                    str(row.get("score_status", "")).startswith("parsed")
                    for row in repaired
                ),
                "repaired": sum(
                    row.get("score_status") == "parsed_after_deterministic_repair"
                    for row in repaired
                ),
                "unrepairable": sum(
                    row.get("score_status") == "parse_failed" for row in repaired
                ),
                "repair_version": REPAIR_VERSION,
                "coverage_parser_version": SCORER_VERSION,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

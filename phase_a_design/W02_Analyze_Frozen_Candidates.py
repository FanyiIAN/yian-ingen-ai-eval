"""Create a model-independent descriptive audit of frozen Week 2 outputs."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_rows", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return rows


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def words(text: str) -> list[str]:
    return re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE)


def row_flags(row: dict[str, Any]) -> list[str]:
    output = row["raw_output"]
    output_norm = normalize(output)
    input_norm = normalize(row["input_stimulus"])
    audit = row.get("deterministic_audit") or {}
    required = audit.get("required_concept_results") or {}
    missing = audit.get("missing_required_lexical_signals") or []
    flags = []
    if row.get("generation_error"):
        flags.append("generation_error")
    if len(words(output)) <= 3:
        flags.append("three_words_or_fewer")
    if output_norm and output_norm in input_norm:
        flags.append("verbatim_substring_of_scenario")
    if required and len(missing) == len(required):
        flags.append("all_lexical_required_signals_missing")
    if audit.get("critical_flags"):
        flags.append("deterministic_critical_flag")
    return flags


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = [len(words(row["raw_output"])) for row in rows]
    outputs = Counter(normalize(row["raw_output"]) for row in rows)
    flagged = [
        {
            "candidate_item_id": row["candidate_item_id"],
            "scenario_id": row["scenario_id"],
            "output": row["raw_output"],
            "flags": row_flags(row),
        }
        for row in rows
        if row_flags(row)
    ]
    return {
        "candidate_count": len(rows),
        "unique_normalized_output_count": len(outputs),
        "duplicate_output_count": sum(count - 1 for count in outputs.values()),
        "word_count": {
            "min": min(counts),
            "median": statistics.median(counts),
            "mean": statistics.fmean(counts),
            "max": max(counts),
        },
        "three_words_or_fewer_count": sum(count <= 3 for count in counts),
        "verbatim_substring_of_scenario_count": sum(
            "verbatim_substring_of_scenario" in row_flags(row) for row in rows
        ),
        "all_lexical_required_signals_missing_count": sum(
            "all_lexical_required_signals_missing" in row_flags(row)
            for row in rows
        ),
        "deterministic_critical_flag_count": sum(
            "deterministic_critical_flag" in row_flags(row) for row in rows
        ),
        "generation_error_count": sum(
            bool(row.get("generation_error")) for row in rows
        ),
        "most_common_normalized_outputs": [
            {"output": output, "count": count}
            for output, count in outputs.most_common(10)
        ],
        "flagged_rows": flagged,
    }


def write_report(
    path: Path,
    *,
    source: Path,
    summary: dict[str, Any],
) -> None:
    lines = [
        "# Week 2 Frozen Candidate Descriptive Audit",
        "",
        f"- Source: `{source}`",
        f"- Candidate outputs: `{summary['candidate_count']}`",
        "",
        (
            "> These are deterministic descriptive signals, not semantic quality "
            "scores. A substring or short-output flag requires human interpretation."
        ),
        "",
        "## By model",
        "",
        "| Model | Count | Unique | Median words | <=3 words | Scenario substring | "
        "All required signals missing | Critical flag |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_id, item in summary["by_candidate_model"].items():
        lines.append(
            f"| {model_id} | {item['candidate_count']} | "
            f"{item['unique_normalized_output_count']} | "
            f"{item['word_count']['median']} | "
            f"{item['three_words_or_fewer_count']} | "
            f"{item['verbatim_substring_of_scenario_count']} | "
            f"{item['all_lexical_required_signals_missing_count']} | "
            f"{item['deterministic_critical_flag_count']} |"
        )
    lines.extend(["", "## Flagged outputs", ""])
    for model_id, item in summary["by_candidate_model"].items():
        lines.extend([f"### {model_id}", ""])
        for row in item["flagged_rows"]:
            lines.extend(
                [
                    f"- `{row['candidate_item_id']}` — `{row['flags']}`",
                    f"  - Output: {row['output']}",
                ]
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.candidate_rows)
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(row["candidate_model_id"], []).append(row)
    result = {
        "source_candidate_rows": str(args.candidate_rows),
        "candidate_count": len(rows),
        "by_candidate_model": {
            model_id: summarize(model_rows)
            for model_id, model_rows in sorted(by_model.items())
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "W02_Frozen_Candidate_Descriptive_Audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    write_report(
        args.output_dir / "W02_Frozen_Candidate_Descriptive_Audit.md",
        source=args.candidate_rows,
        summary=result,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

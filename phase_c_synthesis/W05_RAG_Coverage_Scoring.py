"""Score Week 5 answer coverage against frozen per-item required points.

This is an append-only, local-Mistral diagnostic. It never exposes required
points to the candidate generator, and it never labels the uncalibrated Judge
output as human ground truth. Identical answers for the same evaluation item are
scored once and explicitly reused across otherwise distinct factorial rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


SCORER_VERSION = "1.4.0"
DEFAULT_MAX_PARSE_ATTEMPTS = 3
HERE = Path(__file__).resolve().parent
PHASE_B = HERE.parent / "phase_b_evaluation"
DEFAULT_CONFIG = HERE / "W05_RAG_Optimisation_Run_Config_v1.0.0.yaml"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        handle.flush()


def cache_key(candidate: dict[str, Any]) -> str:
    value = {
        "eval_id": candidate["eval_id"],
        "candidate_output_sha256": candidate["candidate_output_sha256"],
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def coverage_messages(
    candidate: dict[str, Any], item: dict[str, Any]
) -> list[dict[str, str]]:
    contract = {
        "question": item["question"],
        "candidate_response": candidate["candidate_output"],
        "required_points": [
            {
                "point_id": point["point_id"],
                "weight": point["weight"],
                "criterion": point["criterion"],
            }
            for point in item["required_points"]
        ],
        "forbidden_points": [
            {
                "point_id": point["point_id"],
                "criterion": point["criterion"],
            }
            for point in item.get("forbidden_points") or []
        ],
    }
    system = (
        "You are a conservative answer-coverage Judge. Evaluate only whether the "
        "candidate response expresses each supplied required point; do not use model "
        "memory and do not reward eloquence. For every required point assign score 1 "
        "when fully present, 0.5 when materially but incompletely present, or 0 when "
        "absent or contradicted. Use short verbatim response evidence, or an empty "
        "string for score 0. List only forbidden point IDs that the response actually "
        "asserts. Return exactly one JSON object and no prose or code fence with this "
        "schema: {\"point_scores\":[{\"point_id\":\"P1\",\"score\":1,"
        "\"evidence\":\"short quote\"}],\"forbidden_point_violations\":[],"
        "\"rationale\":\"under 25 words\"}. Include every required point exactly "
        "once and do not invent point IDs."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(contract, ensure_ascii=False)},
    ]


def coverage_retry_messages(
    candidate: dict[str, Any],
    item: dict[str, Any],
    _prior_output: str,
    parse_error: str,
    attempt_number: int,
) -> list[dict[str, str]]:
    """Build a changed, item-specific prompt for a bounded schema retry."""

    exact_template = {
        "point_scores": [
            {"point_id": point["point_id"], "score": 0, "evidence": ""}
            for point in item["required_points"]
        ],
        "forbidden_point_violations": [],
        "rationale": "",
    }
    required_lines = "\n".join(
        f"- {point['point_id']}: {point['criterion']}"
        for point in item["required_points"]
    )
    forbidden_lines = "\n".join(
        f"- {point['point_id']}: {point['criterion']}"
        for point in item.get("forbidden_points") or []
    ) or "- none"
    system = (
        f"Retry {attempt_number} because the prior result failed validation: "
        f"{parse_error}. Repair and rescore the answer coverage. Return exactly "
        "one JSON object and no prose or code fence. Copy every point_id from "
        "the template exactly once and in the same order; never omit, "
        "duplicate, or rename a point. Replace each placeholder score with only "
        "0, 0.5, or 1 using the supplied criterion and candidate response. Use an "
        "empty evidence string for score 0. The only allowed forbidden field name "
        "is forbidden_point_violations. Escape newlines inside JSON strings. "
        "Required output template: "
        + json.dumps(exact_template, ensure_ascii=False, separators=(",", ":"))
    )
    user = (
        f"Question:\n{item['question']}\n\n"
        f"Candidate response:\n{candidate['candidate_output']}\n\n"
        f"Required criteria:\n{required_lines}\n\n"
        f"Forbidden criteria:\n{forbidden_lines}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_coverage_value(
    raw_output: str, extract_json_object: Any
) -> tuple[dict[str, Any], list[str]]:
    """Parse JSON with documented syntax-only repairs, never score imputation."""

    repairs: list[str] = []
    try:
        value = extract_json_object(raw_output)
    except (ValueError, TypeError, json.JSONDecodeError):
        start = raw_output.find("{")
        end = raw_output.rfind("}")
        if start < 0 or end <= start:
            raise
        json_text = raw_output[start : end + 1]
        repaired_json_text = json_text.replace("\\{", "{").replace("\\[", "[")
        if repaired_json_text != json_text:
            repairs.append("removed_invalid_backslash_before_brace_or_bracket")
        value = json.loads(repaired_json_text, strict=False)
        repairs.append("json_decode_strict_false_for_control_characters")
    if not isinstance(value, dict):
        raise ValueError("Judge output must decode to a JSON object")
    if (
        "forbidden_point_violations" not in value
        and isinstance(value.get("forbidden_points"), list)
    ):
        value = dict(value)
        value["forbidden_point_violations"] = value.pop("forbidden_points")
        repairs.append("renamed_forbidden_points_alias")
    raw_scores = value.get("point_scores")
    if isinstance(raw_scores, list):
        merged_scores: list[Any] = []
        position_by_id: dict[str, int] = {}
        merged_equal_duplicates = False
        for row in raw_scores:
            if not isinstance(row, dict):
                merged_scores.append(row)
                continue
            point_id = str(row.get("point_id", ""))
            if point_id not in position_by_id:
                position_by_id[point_id] = len(merged_scores)
                merged_scores.append(dict(row))
                continue
            prior = merged_scores[position_by_id[point_id]]
            if not isinstance(prior, dict) or prior.get("score") != row.get("score"):
                merged_scores.append(row)
                continue
            evidence = [
                str(value).strip()
                for value in (prior.get("evidence", ""), row.get("evidence", ""))
                if str(value).strip()
            ]
            prior["evidence"] = " | ".join(dict.fromkeys(evidence))
            merged_equal_duplicates = True
        if merged_equal_duplicates:
            value = dict(value)
            value["point_scores"] = merged_scores
            repairs.append("merged_duplicate_point_rows_with_equal_scores")
    return value, repairs


def normalize_coverage(value: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    raw_scores = value.get("point_scores")
    if not isinstance(raw_scores, list):
        raise ValueError("point_scores must be an array")
    expected = {str(point["point_id"]): point for point in item["required_points"]}
    normalized: dict[str, dict[str, Any]] = {}
    ignored_unknown_point_ids: list[str] = []
    for row in raw_scores:
        if not isinstance(row, dict):
            raise ValueError("each point score must be an object")
        point_id = str(row.get("point_id", ""))
        if point_id not in expected:
            # Some local-Judge responses decompose one registered rubric point
            # into extra P2/P3 rows.  The registered row can still be used
            # without inference: discard only unregistered extras, retain the
            # Judge's original registered-point verdict, and expose the repair.
            ignored_unknown_point_ids.append(point_id)
            continue
        if point_id in normalized:
            raise ValueError(f"duplicate point_id: {point_id}")
        try:
            score = float(row.get("score"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid score for {point_id}") from exc
        if score not in {0.0, 0.5, 1.0}:
            raise ValueError(f"score for {point_id} must be 0, 0.5, or 1")
        normalized[point_id] = {
            "point_id": point_id,
            "weight": int(expected[point_id]["weight"]),
            "score": score,
            "evidence": str(row.get("evidence", "")).strip(),
        }
    missing = sorted(set(expected) - set(normalized))
    if missing:
        raise ValueError(f"missing point scores: {missing}")
    violations = value.get("forbidden_point_violations")
    if not isinstance(violations, list) or not all(
        isinstance(point_id, str) for point_id in violations
    ):
        raise ValueError("forbidden_point_violations must be a string array")
    valid_forbidden = {
        str(point["point_id"]) for point in item.get("forbidden_points") or []
    }
    unknown = sorted(set(violations) - valid_forbidden)
    if unknown:
        raise ValueError(f"unknown forbidden point IDs: {unknown}")
    possible = sum(row["weight"] for row in normalized.values())
    earned = sum(row["weight"] * row["score"] for row in normalized.values())
    return {
        "required_weight_earned": round(float(earned), 6),
        "required_weight_possible": int(possible),
        "required_point_coverage": round(float(earned / possible), 6),
        "per_point_verdicts": [normalized[key] for key in sorted(normalized)],
        "forbidden_point_violations": sorted(set(violations)),
        "ignored_unknown_point_ids": sorted(set(ignored_unknown_point_ids)),
        "rationale": str(value.get("rationale", "")).strip(),
    }


def score_row(
    candidate: dict[str, Any],
    item: dict[str, Any],
    engine: Any,
    extract_json_object: Any,
    judge_contract: dict[str, Any],
    max_parse_attempts: int = DEFAULT_MAX_PARSE_ATTEMPTS,
) -> dict[str, Any]:
    if max_parse_attempts < 1:
        raise ValueError("max_parse_attempts must be positive")
    normalized = None
    parse_error = None
    generated: dict[str, Any] = {}
    attempts: list[dict[str, Any]] = []
    prior_output = ""
    deterministic_repairs: list[str] = []
    for attempt_index in range(max_parse_attempts):
        attempt_number = attempt_index + 1
        messages = (
            coverage_messages(candidate, item)
            if attempt_index == 0
            else coverage_retry_messages(
                candidate,
                item,
                prior_output,
                str(parse_error),
                attempt_number,
            )
        )
        generated = engine.generate(messages)
        prior_output = str(generated["raw_judge_output"])
        attempt_repairs: list[str] = []
        try:
            value, attempt_repairs = parse_coverage_value(
                prior_output, extract_json_object
            )
            normalized = normalize_coverage(value, item)
            parse_error = None
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            parse_error = str(exc)
        attempts.append(
            {
                "attempt_number": attempt_number,
                "raw_judge_output": prior_output,
                "parse_error": parse_error,
                "deterministic_syntax_repairs": attempt_repairs,
            }
        )
        deterministic_repairs = attempt_repairs
        if normalized is not None:
            break
    return {
        "run_item_id": candidate["run_item_id"],
        "coverage_cache_key": cache_key(candidate),
        "eval_id": candidate["eval_id"],
        "variant_id": candidate.get("variant_id", candidate.get("condition")),
        "candidate_model_id": candidate["candidate_model_id"],
        "candidate_model_revision": candidate["candidate_model_revision"],
        "candidate_output_sha256": candidate["candidate_output_sha256"],
        "score_status": "parsed" if normalized is not None else "parse_failed",
        "normalized_coverage": normalized,
        "parse_error": parse_error,
        "parse_attempts": attempts,
        "retry_count": len(attempts) - 1,
        "deterministic_syntax_repairs": deterministic_repairs,
        **generated,
        "judge": {**engine.metadata, **judge_contract},
        "judge_method": "ai_assisted_required_point_rubric_with_bounded_schema_retry",
        "calibration_status": "diagnostic_not_human_validated",
        "scorer_version": SCORER_VERSION,
        "completed_at_utc": utc_now(),
    }


def reused_row(candidate: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    output = dict(source)
    output.update(
        {
            "run_item_id": candidate["run_item_id"],
            "variant_id": candidate.get("variant_id", candidate.get("condition")),
            "reused_from_run_item_id": source["run_item_id"],
            "reuse_reason": "same_eval_id_and_candidate_output_sha256",
            "completed_at_utc": utc_now(),
        }
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--eval-set", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--judge-model-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--max-parse-attempts", type=int, default=DEFAULT_MAX_PARSE_ATTEMPTS
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.max_parse_attempts < 1:
        raise ValueError("--max-parse-attempts must be positive")

    candidates = read_jsonl(args.candidates)
    eval_set = load_yaml(args.eval_set)
    items = {str(item["eval_id"]): item for item in eval_set["items"]}
    candidate_ids = [str(row.get("run_item_id", "")) for row in candidates]
    if any(not value for value in candidate_ids) or len(candidate_ids) != len(
        set(candidate_ids)
    ):
        raise ValueError("candidate run_item_id values must be non-empty and unique")
    missing = sorted({row["eval_id"] for row in candidates} - set(items))
    if missing:
        raise ValueError(f"candidate eval IDs absent from evaluation set: {missing}")
    validation = {
        "status": "validated",
        "candidate_rows": len(candidates),
        "unique_candidate_answer_items": len({cache_key(row) for row in candidates}),
        "scorer_version": SCORER_VERSION,
    }
    if args.validate_only:
        print(json.dumps(validation, indent=2))
        return

    config = load_yaml(args.config)
    judge_contract = config["models"]["evaluator"]
    model_dir = args.judge_model_dir or Path(judge_contract["local_model_directory"])
    if PHASE_B not in [Path(value) for value in sys.path if value]:
        sys.path.insert(0, str(PHASE_B))
    from W04_AI_Assisted_Scoring import LocalJudge, extract_json_object

    existing = read_jsonl(args.output)
    if existing and not args.resume:
        raise FileExistsError(f"{args.output} exists; use --resume")
    done = {row["run_item_id"] for row in existing}
    cache = {
        row["coverage_cache_key"]: row
        for row in existing
        if row.get("score_status") == "parsed"
    }
    pending = [row for row in candidates if row["run_item_id"] not in done]
    if args.limit is not None:
        pending = pending[: args.limit]
    load_started = time.perf_counter()
    engine = LocalJudge(model_dir, args.max_new_tokens)
    load_ms = round((time.perf_counter() - load_started) * 1000, 3)
    if pending:
        engine.generate(coverage_messages(pending[0], items[pending[0]["eval_id"]]))

    for index, candidate in enumerate(pending, start=1):
        print(f"[coverage {index}/{len(pending)}] {candidate['run_item_id']}", flush=True)
        key = cache_key(candidate)
        if key in cache:
            result = reused_row(candidate, cache[key])
        else:
            result = score_row(
                candidate,
                items[candidate["eval_id"]],
                engine,
                extract_json_object,
                judge_contract,
                args.max_parse_attempts,
            )
            result["judge_cold_load_ms"] = load_ms
            if result["score_status"] == "parsed":
                cache[key] = result
        append_jsonl(args.output, result)
    final = read_jsonl(args.output)
    print(
        json.dumps(
            {
                **validation,
                "rows_in_output": len(final),
                "parsed_rows": sum(row["score_status"] == "parsed" for row in final),
                "reused_rows": sum("reused_from_run_item_id" in row for row in final),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

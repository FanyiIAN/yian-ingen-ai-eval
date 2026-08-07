"""Normalize and summarize accumulated Week 2-4 row-level evidence.

This module deliberately refuses to pool unlike evaluation families or dataset
versions. Mechanistic explanations remain a required human-review field rather
than being generated from correlations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from jsonschema import Draft202012Validator


HERE = Path(__file__).resolve().parent
DEFAULT_SCHEMA = HERE / "W05_Normalized_Result_Schema_v0.1.0.json"
ANALYSIS_VERSION = "0.1.0"
IDENTITY_COLUMNS = [
    "evaluation_family",
    "evaluation_set_id",
    "evaluation_set_version",
    "score_name",
]
MODEL_COLUMNS = ["model_id", "model_revision", "random_seed"]


def load_schema(path: Path = DEFAULT_SCHEMA) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _python_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def validate_records(
    records: list[dict[str, Any]], schema: dict[str, Any] | None = None
) -> None:
    if not records:
        raise ValueError("normalized evaluation input is empty")
    validator = Draft202012Validator(schema or load_schema())
    errors: list[str] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        for error in validator.iter_errors(record):
            location = ".".join(str(part) for part in error.absolute_path)
            errors.append(f"row {index} {location or '<root>'}: {error.message}")
        result_id = str(record.get("result_id", ""))
        if result_id in seen_ids:
            errors.append(f"row {index}: duplicate result_id {result_id!r}")
        seen_ids.add(result_id)
        if record.get("pass_indicator") and record.get("failure_mode") not in (
            "",
            "none",
        ):
            errors.append(
                f"row {index}: passing row must not carry failure mode "
                f"{record.get('failure_mode')!r}"
            )
    if errors:
        raise ValueError("Normalized row validation failed:\n- " + "\n- ".join(errors))


def dataframe_from_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    validate_records(records)
    frame = pd.DataFrame.from_records(records)
    frame["failure_indicator"] = (~frame["pass_indicator"].astype(bool)).astype(int)
    frame["score_value"] = pd.to_numeric(frame["score_value"], errors="raise")
    frame["severity"] = pd.to_numeric(frame["severity"], errors="raise").astype(int)
    frame["random_seed"] = pd.to_numeric(
        frame["random_seed"], errors="raise"
    ).astype(int)
    return frame


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {key: _python_scalar(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def performance_by_platform_model(frame: pd.DataFrame) -> list[dict[str, Any]]:
    group_columns = IDENTITY_COLUMNS + ["platform"] + MODEL_COLUMNS
    grouped = (
        frame.groupby(group_columns, dropna=False)
        .agg(
            rows=("result_id", "count"),
            scenarios=("scenario_id", "nunique"),
            mean_score=("score_value", "mean"),
            pass_rate=("pass_indicator", "mean"),
            failure_rate=("failure_indicator", "mean"),
        )
        .reset_index()
        .sort_values(group_columns)
    )
    for column in ("mean_score", "pass_rate", "failure_rate"):
        grouped[column] = grouped[column].round(6)
    return _records(grouped)


def failure_mode_distribution(frame: pd.DataFrame) -> list[dict[str, Any]]:
    failures = frame[frame["failure_indicator"] == 1].copy()
    if failures.empty:
        return []
    group_columns = IDENTITY_COLUMNS + ["platform"] + MODEL_COLUMNS
    denominators = (
        frame.groupby(group_columns, dropna=False)
        .size()
        .rename("all_rows")
        .reset_index()
    )
    grouped = (
        failures.groupby(group_columns + ["failure_mode"], dropna=False)
        .size()
        .rename("failure_rows")
        .reset_index()
        .merge(denominators, on=group_columns, how="left", validate="many_to_one")
    )
    grouped["rate_over_all_rows"] = (
        grouped["failure_rows"] / grouped["all_rows"]
    ).round(6)
    return _records(grouped.sort_values(group_columns + ["failure_mode"]))


def severity_failure_relationship(frame: pd.DataFrame) -> list[dict[str, Any]]:
    group_columns = IDENTITY_COLUMNS + MODEL_COLUMNS
    results: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        correlation = group["severity"].corr(
            group["failure_indicator"], method="spearman"
        )
        severity_table = (
            group.groupby("severity", dropna=False)["failure_indicator"]
            .agg(["count", "mean"])
            .reset_index()
        )
        key_values = keys if isinstance(keys, tuple) else (keys,)
        result = dict(zip(group_columns, key_values, strict=True))
        result.update(
            {
                "rows": int(len(group)),
                "spearman_severity_failure": (
                    None if pd.isna(correlation) else round(float(correlation), 6)
                ),
                "correlation_status": (
                    "undefined_constant_or_insufficient_variation"
                    if pd.isna(correlation)
                    else "descriptive_not_causal"
                ),
                "by_severity": [
                    {
                        "severity": int(row["severity"]),
                        "rows": int(row["count"]),
                        "failure_rate": round(float(row["mean"]), 6),
                    }
                    for row in severity_table.to_dict(orient="records")
                ],
            }
        )
        results.append(result)
    return results


def surprising_scenarios(
    frame: pd.DataFrame, top_n: int = 2
) -> list[dict[str, Any]]:
    """Rank scenario residuals from platform/severity expected failure rate."""

    stratum = IDENTITY_COLUMNS + ["platform", "severity"]
    expected = (
        frame.groupby(stratum, dropna=False)["failure_indicator"]
        .mean()
        .rename("expected_failure_rate")
        .reset_index()
    )
    scenario = (
        frame.groupby(stratum + ["scenario_id"], dropna=False)
        .agg(
            observations=("result_id", "count"),
            observed_failure_rate=("failure_indicator", "mean"),
        )
        .reset_index()
        .merge(expected, on=stratum, how="left", validate="many_to_one")
    )
    scenario["residual"] = (
        scenario["observed_failure_rate"] - scenario["expected_failure_rate"]
    )
    scenario["absolute_residual"] = scenario["residual"].abs()
    ranked = scenario.sort_values(
        ["absolute_residual", "observations", "scenario_id"],
        ascending=[False, False, True],
    ).head(top_n)
    output: list[dict[str, Any]] = []
    for row in ranked.to_dict(orient="records"):
        output.append(
            {
                **{key: _python_scalar(value) for key, value in row.items()},
                "observed_failure_rate": round(
                    float(row["observed_failure_rate"]), 6
                ),
                "expected_failure_rate": round(
                    float(row["expected_failure_rate"]), 6
                ),
                "residual": round(float(row["residual"]), 6),
                "absolute_residual": round(float(row["absolute_residual"]), 6),
                "mechanistic_hypothesis": None,
                "mechanistic_hypothesis_status": "required_manual_evidence_review",
            }
        )
    return output


def analyze_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    frame = dataframe_from_records(records)
    revisions = (
        frame[["model_id", "model_revision"]]
        .drop_duplicates()
        .sort_values(["model_id", "model_revision"])
    )
    datasets = (
        frame[["evaluation_set_id", "evaluation_set_version"]]
        .drop_duplicates()
        .sort_values(["evaluation_set_id", "evaluation_set_version"])
    )
    return {
        "analysis_version": ANALYSIS_VERSION,
        "status": "framework_output_requires_source_adapter_and_human_review",
        "row_count": int(len(frame)),
        "model_registry": _records(revisions),
        "evaluation_set_registry": _records(datasets),
        "seed_registry": sorted(int(value) for value in frame["random_seed"].unique()),
        "performance_by_platform_model": performance_by_platform_model(frame),
        "failure_mode_distribution": failure_mode_distribution(frame),
        "severity_failure_relationship": severity_failure_relationship(frame),
        "surprising_scenarios": surprising_scenarios(frame, top_n=2),
        "interpretation_boundary": (
            "Summaries are stratified by evaluation family, evaluation-set version, "
            "and score name. Correlations are descriptive. The two surprise rows "
            "are review priorities, not mechanistic explanations; mechanisms must "
            "be established from row-level prompts, responses, and conditions."
        ),
    }


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return payload["rows"]
    raise ValueError("JSON input must be a row list or an object with a rows list")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite analysis output: {args.output}")
    records = load_json_or_jsonl(args.input)
    result = analyze_records(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "ok", "rows": result["row_count"]}, indent=2))


if __name__ == "__main__":
    main()

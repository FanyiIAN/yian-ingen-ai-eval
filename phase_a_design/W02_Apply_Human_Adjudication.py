"""Create an immutable, human-scored derivative of a frozen Week 2 result file."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import jsonschema
import yaml

import W02_Eval_Runner as base


ROOT = Path(__file__).resolve().parent
DEFAULT_GOLD = ROOT / "W02_Human_Adjudication.yaml"
DEFAULT_OUTPUT_ROOT = ROOT / "experiments" / "w02_human_adjudication_v0.1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-rows", type=Path, required=True)
    parser.add_argument("--model-key", choices=("mistral", "flan"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--human-gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    args = parse_args()
    source_path = args.source_rows.resolve()
    gold_path = args.human_gold.resolve()
    output_root = args.output_root.resolve()
    for path, label in (
        (source_path, "source_rows"),
        (gold_path, "human_gold"),
        (output_root, "output_root"),
    ):
        base.assert_non_c_path(path, label)

    source_rows = read_jsonl(source_path)
    gold_doc = yaml.safe_load(gold_path.read_text(encoding="utf-8"))
    gold_by_id = {item["scenario_id"]: item for item in gold_doc["items"]}
    if len(source_rows) != 35 or set(gold_by_id) != {
        row["scenario_id"] for row in source_rows
    }:
        raise ValueError("Source rows and human adjudication must contain the same 35 items")

    schema = json.loads(base.RESULT_SCHEMA_PATH.read_text(encoding="utf-8"))
    adjudicated: list[dict[str, Any]] = []
    for source in source_rows:
        row = copy.deepcopy(source)
        human = gold_by_id[row["scenario_id"]][args.model_key]
        row["source_run_id"] = source["run_id"]
        row["source_row_sha256"] = base.sha256_text(
            json.dumps(source, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
        row["run_id"] = args.run_id
        row["adjudication_id"] = gold_doc["adjudication_id"]
        row["adjudication_version"] = str(gold_doc["version"])
        row["adjudication_sha256"] = base.sha256_file(gold_path)
        row["human_task_accuracy"] = int(human["task_accuracy"])
        row["human_contextual_grounding"] = int(human["contextual_grounding"])
        row["human_primary_failure_mode"] = human["primary_failure_mode"]
        row["human_rationale"] = human["rationale"]
        row["human_review_required"] = True
        row["human_review_completed"] = True
        row["human_review_status"] = "first_pass_requires_second_reviewer"
        row["final_task_accuracy"] = int(human["task_accuracy"])
        row["final_contextual_grounding"] = int(human["contextual_grounding"])
        row["final_primary_failure_mode"] = human["primary_failure_mode"]
        row["final_score_status"] = "human_adjudicated_first_pass"
        jsonschema.validate(row, schema)
        adjudicated.append(row)

    adjudicated.sort(key=lambda row: row["scenario_id"])
    run_dir = output_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    output_path = run_dir / "W02_Human_Adjudicated_Rows.jsonl"
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in adjudicated),
        encoding="utf-8",
    )
    base.write_csv(run_dir / "W02_Human_Adjudicated_Results.csv", adjudicated)
    summary = {
        "run_id": args.run_id,
        "created_at_utc": base.utc_now(),
        "status": "first_pass_requires_second_reviewer",
        "source_rows": str(source_path),
        "source_rows_sha256": base.sha256_file(source_path),
        "human_gold": str(gold_path),
        "human_gold_sha256": base.sha256_file(gold_path),
        "model_key": args.model_key,
        "row_count": len(adjudicated),
        "metrics": base.aggregate_metrics(adjudicated),
        "claim_boundary": "first-pass human scores for frozen open-model outputs only",
        "output_rows": str(output_path),
        "output_rows_sha256": base.sha256_file(output_path),
    }
    base.json_dump(run_dir / "W02_Human_Adjudicated_Summary.json", summary)
    print(f"Wrote {len(adjudicated)} adjudicated rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

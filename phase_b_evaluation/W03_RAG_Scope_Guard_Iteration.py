"""Build v0.3.5 by appending one question-scope guard to v0.3.4."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from W03_RAG_Final_Audit_Iteration import (
    OFFICIAL_RAG_SYSTEM_PROMPT_V034,
    canonical_json_sha256,
)


ITERATION_BUILDER_VERSION = "0.3.5"
OFFICIAL_RAG_SYSTEM_PROMPT_V035 = (
    OFFICIAL_RAG_SYSTEM_PROMPT_V034
    + " Question-scope guard: apply the compound checklist only when the user's "
    "question explicitly names multiple items, categories, conditions, or "
    "qualifiers. Internally verify every explicitly named part, but include in "
    "the final answer only those requested parts and directly necessary "
    "qualifications. Do not add unrelated background facts, derived policies, "
    "or implications. For a simple question, give one concise direct answer "
    "with only the evidence needed to support it."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def build_scope_guard_iteration(
    parent_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(parent_rows) != 12:
        raise ValueError("Expected 12 v0.3.4 parent RAG rows")
    if {row.get("condition") for row in parent_rows} != {"rag"}:
        raise ValueError("Every parent row must be a RAG row")
    if len({row.get("eval_id") for row in parent_rows}) != 12:
        raise ValueError("Expected 12 unique parent evaluation IDs")
    if len({row.get("run_item_id") for row in parent_rows}) != 12:
        raise ValueError("Parent run_item_id values must be unique")

    generation = config["generation"]
    output: list[dict[str, Any]] = []
    for parent in parent_rows:
        if parent.get("candidate_model_id") != generation["candidate_model_id"]:
            raise ValueError("Candidate model differs from the frozen parent")
        if (
            parent.get("candidate_model_revision")
            != generation["candidate_model_revision"]
        ):
            raise ValueError("Candidate revision differs from the frozen parent")
        if parent.get("random_seed") != config["random_seed"]:
            raise ValueError("Seed differs from the frozen parent")
        if not parent.get("retrieved_contexts"):
            raise ValueError("A parent RAG row has no retrieved contexts")

        messages = parent.get("candidate_messages")
        if (
            not isinstance(messages, list)
            or len(messages) != 2
            or messages[0].get("role") != "system"
            or messages[0].get("content") != OFFICIAL_RAG_SYSTEM_PROMPT_V034
        ):
            raise ValueError(
                f"{parent.get('run_item_id')}: unexpected v0.3.4 prompt; "
                "refusing an uncontrolled rewrite"
            )
        run_item_id = parent.get("run_item_id") or ""
        if not run_item_id.endswith("::prompt-v0.3.4"):
            raise ValueError(f"{run_item_id}: missing v0.3.4 lineage suffix")

        row = copy.deepcopy(parent)
        row["prior_lineage"] = {
            "parent_run_item_id": parent.get("parent_run_item_id"),
            "prior_lineage": parent.get("prior_lineage"),
        }
        row["parent_run_item_id"] = run_item_id
        row["parent_candidate_messages_sha256"] = parent[
            "candidate_messages_sha256"
        ]
        row["run_item_id"] = run_item_id.removesuffix(
            "::prompt-v0.3.4"
        ) + "::prompt-v0.3.5"
        row["prompt_version"] = generation["prompt_version"]
        row["iteration_type"] = "rag_prompt_only_question_scope_guard"
        row["prompt_change"] = generation["prompt_change"]
        row["candidate_messages"][0]["content"] = OFFICIAL_RAG_SYSTEM_PROMPT_V035
        row["candidate_messages_sha256"] = canonical_json_sha256(
            row["candidate_messages"]
        )
        output.append(row)

    expected_rows = int(config["comparison"]["expected_generation_rows"])
    if len(output) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, built {len(output)}")
    return output, {
        "status": "ok",
        "rows": len(output),
        "condition_counts": dict(Counter(row["condition"] for row in output)),
        "unique_eval_ids": len({row["eval_id"] for row in output}),
        "retrieval_traces_inherited": all(
            row["retrieved_contexts"] for row in output
        ),
        "manipulated_variable": config["comparison"]["manipulated_variable"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parent_rows = read_jsonl(args.parent_input)
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    rows, validation = build_scope_guard_iteration(parent_rows, config)
    write_jsonl(args.output, rows)
    manifest = {
        "iteration_builder_version": ITERATION_BUILDER_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_input_path": str(args.parent_input),
        "parent_input_sha256": sha256_file(args.parent_input),
        "config_path": str(args.config),
        "config_sha256": sha256_file(args.config),
        "output_path": str(args.output),
        "output_sha256": sha256_file(args.output),
        "validation": validation,
    }
    if args.manifest.exists():
        raise FileExistsError(
            f"Refusing to overwrite immutable manifest: {args.manifest}"
        )
    with args.manifest.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Run the expanded Week 3 retrieval and three-model Base/RAG comparison.

The runner deliberately keeps dependency imports in one process.  This avoids
paying the RunPod network-volume cold-import penalty once per model while still
writing immutable inputs, candidate configs, raw outputs, and a manifest.
"""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

import W03_RAG_Pipeline as rag
from W03_RAG_Multi_Model_Generation import run_generation
from W03_RAG_Multi_Model_Inputs import expand_candidate


RUNNER_VERSION = "0.1.0"


def artifact_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "shared_inputs": run_dir / "shared_base_rag_inputs.jsonl",
        "retrieval_manifest": run_dir / "retrieval_input_manifest.json",
        "run_manifest": run_dir / "expanded_multimodel_run_manifest.json",
        "candidates": run_dir / "candidates",
    }


def write_jsonl_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_yaml_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb", type=Path, required=True)
    parser.add_argument("--eval-set", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--persist-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--limit-per-model", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = artifact_paths(args.run_dir)
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty immutable run: {args.run_dir}"
        )
    args.run_dir.mkdir(parents=True, exist_ok=True)

    kb, eval_set, config = rag.load_assets(
        args.kb, args.eval_set, args.config
    )
    validation = rag.validate_assets(kb, eval_set, config)
    if len(config.get("candidate_matrix", [])) != 3:
        raise ValueError("Expanded comparison requires exactly three candidates")
    persist_dir = rag.ensure_external_persist_directory(args.persist_dir)

    index_result, store = rag.index_documents(
        kb, config, persist_dir, args.embedding_device
    )
    reranker = rag.build_reranker(config, args.embedding_device)
    shared_rows = rag.build_run_inputs(
        kb, eval_set, store, config, reranker=reranker
    )
    write_jsonl_exclusive(paths["shared_inputs"], shared_rows)
    retrieval_manifest = rag.build_manifest(
        paths={
            "knowledge_base": args.kb,
            "evaluation_set": args.eval_set,
            "run_config": args.config,
            "run_inputs": paths["shared_inputs"],
        },
        assets={
            "knowledge_base": kb,
            "evaluation_set": eval_set,
            "run_config": config,
        },
        collection=store._collection.name,
        persist_dir=persist_dir,
    )
    retrieval_manifest.update(
        {
            "row_count": len(shared_rows),
            "validation": validation,
            "index": index_result,
        }
    )
    rag.write_json(paths["retrieval_manifest"], retrieval_manifest)

    # Embedding and reranker weights are no longer needed after the exact shared
    # candidate-visible inputs have been frozen.
    del store
    del reranker
    release_cuda()

    candidate_results: dict[str, Any] = {}
    for candidate in config["candidate_matrix"]:
        key = candidate["candidate_model_key"]
        candidate_dir = paths["candidates"] / key
        candidate_dir.mkdir(parents=True, exist_ok=True)
        candidate_rows, candidate_config = expand_candidate(
            shared_rows, config, candidate
        )
        input_path = candidate_dir / "inputs.jsonl"
        config_path = candidate_dir / "config.yaml"
        output_path = candidate_dir / "generations.jsonl"
        write_jsonl_exclusive(input_path, candidate_rows)
        write_yaml_exclusive(config_path, candidate_config)
        print(f"[candidate] {key}", flush=True)
        result = run_generation(
            rows=candidate_rows,
            config=candidate_config,
            model_dir=Path(candidate["local_model_directory"]),
            output_path=output_path,
            resume=False,
            limit=args.limit_per_model,
        )
        candidate_results[key] = {
            **result,
            "input_path": str(input_path),
            "input_sha256": rag.sha256_file(input_path),
            "config_path": str(config_path),
            "config_sha256": rag.sha256_file(config_path),
            "output_sha256": rag.sha256_file(output_path),
        }
        release_cuda()

    manifest = {
        "runner_version": RUNNER_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": config["iteration_policy"]["run_id"],
        "random_seed": int(config["random_seed"]),
        "knowledge_base": (
            f"{kb['knowledge_base_id']}#{kb['knowledge_base_version']}"
        ),
        "evaluation_set": (
            f"{eval_set['evaluation_set_id']}#{eval_set['evaluation_set_version']}"
        ),
        "shared_inputs_sha256": rag.sha256_file(paths["shared_inputs"]),
        "shared_rows": len(shared_rows),
        "candidate_results": candidate_results,
        "claim_boundary": config["claim_boundary"],
    }
    rag.write_json(paths["run_manifest"], manifest)
    print(
        json.dumps(
            {
                "status": "ok",
                "shared_rows": len(shared_rows),
                "candidate_rows": {
                    key: result["rows_in_output"]
                    for key, result in candidate_results.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Week 5 RAG optimisation framework and local CPU smoke runner.

The local mode intentionally uses cached small models and deterministic proxy
metrics. It validates the experiment contract; it does not produce the final
Week 5 RAGAS or cross-encoder result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer


FRAMEWORK_VERSION = "0.1.0"
HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "W05_RAG_Optimisation_Config_v0.1.0.yaml"
DEFAULT_FIXTURE = HERE / "W05_RAG_Local_Smoke_Fixture_v0.1.0.yaml"
WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "should",
    "that",
    "the",
    "their",
    "this",
    "to",
    "what",
    "when",
    "while",
    "who",
    "with",
}


@dataclass(frozen=True, order=True)
class Variant:
    chunk_size_tokens: int
    top_k: int
    reranking: str

    @property
    def variant_id(self) -> str:
        suffix = "ce" if self.reranking == "cross_encoder" else "none"
        return f"chunk-{self.chunk_size_tokens}_topk-{self.top_k}_rerank-{suffix}"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    token_count: int


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return payload


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_factorial_matrix(config: dict[str, Any]) -> list[Variant]:
    design = config["factorial_design"]
    variants = [
        Variant(int(chunk), int(top_k), str(reranking))
        for chunk, top_k, reranking in itertools.product(
            design["chunk_size_tokens"],
            design["top_k"],
            design["reranking"],
        )
    ]
    return sorted(variants)


def variant_differences(left: Variant, right: Variant) -> list[str]:
    return [
        field
        for field in ("chunk_size_tokens", "top_k", "reranking")
        if getattr(left, field) != getattr(right, field)
    ]


def build_matched_contrasts(variants: Iterable[Variant]) -> list[dict[str, Any]]:
    contrasts: list[dict[str, Any]] = []
    for left, right in itertools.combinations(sorted(variants), 2):
        differences = variant_differences(left, right)
        if len(differences) != 1:
            continue
        factor = differences[0]
        contrasts.append(
            {
                "factor": factor,
                "left_variant_id": left.variant_id,
                "right_variant_id": right.variant_id,
                "left_value": getattr(left, factor),
                "right_value": getattr(right, factor),
            }
        )
    return contrasts


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    design = config.get("factorial_design") or {}
    expected_factors = {
        "chunk_size_tokens": [256, 512, 1024],
        "top_k": [1, 3, 5],
        "reranking": ["none", "cross_encoder"],
    }
    for field, expected in expected_factors.items():
        if list(design.get(field) or []) != expected:
            errors.append(f"{field} must equal {expected!r}")

    variants = build_factorial_matrix(config) if not errors else []
    expected_count = int(design.get("expected_configurations", -1))
    if expected_count != 18:
        errors.append("expected_configurations must be 18")
    if variants and len(variants) != expected_count:
        errors.append(
            f"factorial matrix produced {len(variants)} variants, expected {expected_count}"
        )
    if len({variant.variant_id for variant in variants}) != len(variants):
        errors.append("variant IDs are not unique")
    controls = design.get("execution_controls") or {}
    if not controls.get("randomized_variant_order"):
        errors.append("variant execution order must be randomized")
    if int(controls.get("warmup_requests_per_runtime", 0)) < 1:
        errors.append("at least one unmeasured runtime warm-up is required")
    if not controls.get("separate_cold_load_from_warm_latency"):
        errors.append("cold load must be separated from warm request latency")

    required_metrics = set(
        (config.get("metrics") or {}).get("required_final_metrics") or []
    )
    if required_metrics != {
        "faithfulness",
        "answer_relevance",
        "required_point_coverage",
    }:
        errors.append("final metric contract is incomplete")

    smoke = config.get("local_smoke") or {}
    surrogate = smoke.get("reranker_surrogate") or {}
    if surrogate.get("effective_implementation") == "cross_encoder":
        errors.append("local smoke reranker must not masquerade as a cross-encoder")
    if "not_cross_encoder" not in str(surrogate.get("status", "")):
        errors.append("local smoke reranker limitation must be explicit")

    traceability = set((config.get("row_contract") or {}).get("required_traceability") or [])
    for required in (
        "model_id",
        "model_revision",
        "evaluation_set_id",
        "evaluation_set_version",
        "random_seed",
    ):
        if required not in traceability:
            errors.append(f"row traceability is missing {required}")

    if errors:
        raise ValueError("Invalid Week 5 config:\n- " + "\n- ".join(errors))

    contrasts = build_matched_contrasts(variants)
    contrast_counts: dict[str, int] = {}
    for contrast in contrasts:
        contrast_counts[contrast["factor"]] = (
            contrast_counts.get(contrast["factor"], 0) + 1
        )
    return {
        "framework_version": FRAMEWORK_VERSION,
        "experiment_id": config["experiment_id"],
        "experiment_version": config["experiment_version"],
        "random_seed": int(config["random_seed"]),
        "variant_count": len(variants),
        "matched_contrast_count": len(contrasts),
        "matched_contrasts_by_factor": contrast_counts,
        "variant_ids": [variant.variant_id for variant in variants],
        "claim_boundary": config["claim_boundary"],
    }


def tokenize_words(value: str, *, drop_stopwords: bool = False) -> list[str]:
    tokens = WORD_RE.findall(value.lower())
    if drop_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]
    return tokens


def token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = tokenize_words(prediction, drop_stopwords=True)
    reference_tokens = tokenize_words(reference, drop_stopwords=True)
    if not prediction_tokens or not reference_tokens:
        return 0.0
    prediction_counts: dict[str, int] = {}
    reference_counts: dict[str, int] = {}
    for token in prediction_tokens:
        prediction_counts[token] = prediction_counts.get(token, 0) + 1
    for token in reference_tokens:
        reference_counts[token] = reference_counts.get(token, 0) + 1
    overlap = sum(
        min(count, reference_counts.get(token, 0))
        for token, count in prediction_counts.items()
    )
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def lexical_context_support(response: str, context: str) -> float:
    response_tokens = tokenize_words(response, drop_stopwords=True)
    if not response_tokens:
        return 0.0
    context_tokens = set(tokenize_words(context, drop_stopwords=True))
    return sum(token in context_tokens for token in response_tokens) / len(
        response_tokens
    )


def required_term_coverage(response: str, required_terms: list[list[str]]) -> float:
    if not required_terms:
        return 1.0
    normalized = " ".join(tokenize_words(response))
    covered = 0
    for alternatives in required_terms:
        if any(" ".join(tokenize_words(term)) in normalized for term in alternatives):
            covered += 1
    return covered / len(required_terms)


def materialize_documents(fixture: dict[str, Any]) -> list[dict[str, str]]:
    repeat = int(fixture["materialization"]["repeat_each_segment"])
    separator = str(fixture["materialization"].get("separator", " "))
    documents: list[dict[str, str]] = []
    for document in fixture["documents"]:
        blocks: list[str] = []
        for cycle in range(repeat):
            for index, segment in enumerate(document["segments"], start=1):
                blocks.append(
                    f"Cycle {cycle + 1}, policy segment {index}. {str(segment).strip()}"
                )
        documents.append(
            {
                "document_id": str(document["document_id"]),
                "title": str(document["title"]),
                "text": separator.join(blocks),
            }
        )
    return documents


def chunk_documents(
    documents: list[dict[str, str]],
    tokenizer: Any,
    chunk_size_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    if overlap_tokens >= chunk_size_tokens:
        raise ValueError("chunk overlap must be smaller than chunk size")
    chunks: list[Chunk] = []
    step = chunk_size_tokens - overlap_tokens
    for document in documents:
        original_model_max_length = tokenizer.model_max_length
        tokenizer.model_max_length = max(int(original_model_max_length), 10_000_000)
        try:
            token_ids = tokenizer.encode(document["text"], add_special_tokens=False)
        finally:
            tokenizer.model_max_length = original_model_max_length
        for index, start in enumerate(range(0, len(token_ids), step), start=1):
            window = token_ids[start : start + chunk_size_tokens]
            if not window:
                continue
            text = tokenizer.decode(window, skip_special_tokens=True).strip()
            chunks.append(
                Chunk(
                    chunk_id=f"{document['document_id']}::chunk-{index:03d}",
                    document_id=document["document_id"],
                    text=text,
                    token_count=len(window),
                )
            )
            if start + chunk_size_tokens >= len(token_ids):
                break
    if not chunks:
        raise ValueError("chunker produced no chunks")
    return chunks


class LocalSmallModels:
    """Cached FLAN generator plus cached MiniLM smoke reranker."""

    def __init__(self, config: dict[str, Any], cache_dir: Path | None = None):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.config = config
        smoke = config["local_smoke"]
        generator = smoke["generator"]
        reranker = smoke["reranker_surrogate"]
        download_kwargs: dict[str, Any] = {"local_files_only": True}
        if cache_dir is not None:
            download_kwargs["cache_dir"] = str(cache_dir)

        generator_dir = snapshot_download(
            repo_id=generator["model_id"],
            revision=generator["model_revision"],
            **download_kwargs,
        )
        reranker_dir = snapshot_download(
            repo_id=reranker["model_id"],
            revision=reranker["model_revision"],
            **download_kwargs,
        )
        self.generator_tokenizer = AutoTokenizer.from_pretrained(
            generator_dir, local_files_only=True
        )
        self.generator_model = AutoModelForSeq2SeqLM.from_pretrained(
            generator_dir, local_files_only=True
        ).eval()
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(
            reranker_dir, local_files_only=True
        )
        self.reranker_model = AutoModel.from_pretrained(
            reranker_dir, local_files_only=True
        ).eval()
        self.generator_dir = str(generator_dir)
        self.reranker_dir = str(reranker_dir)
        self._seed(int(config["random_seed"]))

    def _seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        self.torch.manual_seed(seed)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), 16):
            batch = texts[start : start + 16]
            encoded = self.reranker_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            with self.torch.inference_mode():
                hidden = self.reranker_model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.cpu().numpy())
        return np.concatenate(vectors, axis=0)

    def generate(self, prompt: str) -> str:
        generator = self.config["local_smoke"]["generator"]
        encoded = self.generator_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=int(generator["max_input_tokens"]),
        )
        with self.torch.inference_mode():
            output = self.generator_model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=int(generator["max_new_tokens"]),
            )
        return self.generator_tokenizer.decode(
            output[0], skip_special_tokens=True
        ).strip()


def tfidf_rank(question: str, chunks: list[Chunk], fetch_k: int) -> list[int]:
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([chunk.text for chunk in chunks])
    query = vectorizer.transform([question])
    scores = (matrix @ query.T).toarray().reshape(-1)
    ranked = sorted(range(len(chunks)), key=lambda index: (-scores[index], index))
    return ranked[: min(fetch_k, len(ranked))]


def minilm_rerank(
    question: str,
    chunks: list[Chunk],
    candidate_indices: list[int],
    models: LocalSmallModels,
) -> list[int]:
    if not candidate_indices:
        return []
    texts = [question] + [chunks[index].text for index in candidate_indices]
    embeddings = models.embed(texts)
    scores = embeddings[1:] @ embeddings[0]
    scored = zip(candidate_indices, scores.tolist(), strict=True)
    return [index for index, _ in sorted(scored, key=lambda pair: (-pair[1], pair[0]))]


def context_preview(
    chunks: list[Chunk], tokenizer: Any, tokens_per_chunk: int
) -> str:
    blocks: list[str] = []
    for chunk in chunks:
        token_ids = tokenizer.encode(chunk.text, add_special_tokens=False)
        preview = tokenizer.decode(
            token_ids[:tokens_per_chunk], skip_special_tokens=True
        ).strip()
        blocks.append(f"[{chunk.chunk_id}] {preview}")
    return "\n".join(blocks)


def smoke_effective_reranking(config: dict[str, Any], requested: str) -> str:
    if requested == "none":
        return "none"
    return str(
        config["local_smoke"]["reranker_surrogate"]["effective_implementation"]
    )


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["variant_id"], []).append(row)
    summaries: list[dict[str, Any]] = []
    for variant_id, group in sorted(grouped.items()):
        first = group[0]
        summaries.append(
            {
                "variant_id": variant_id,
                "chunk_size_tokens": first["chunk_size_tokens"],
                "top_k": first["top_k"],
                "requested_reranking": first["requested_reranking"],
                "effective_reranking": first["effective_reranking"],
                "items": len(group),
                "mean_lexical_context_support_proxy": round(
                    statistics.mean(
                        row["lexical_context_support_proxy"] for row in group
                    ),
                    6,
                ),
                "mean_reference_token_f1_proxy": round(
                    statistics.mean(row["reference_token_f1_proxy"] for row in group),
                    6,
                ),
                "mean_required_term_coverage_proxy": round(
                    statistics.mean(
                        row["required_term_coverage_proxy"] for row in group
                    ),
                    6,
                ),
                "mean_question_to_response_ms": round(
                    statistics.mean(
                        row["latency_ms"]["question_to_response_ms"] for row in group
                    ),
                    3,
                ),
                "mean_retrieval_ms": round(
                    statistics.mean(
                        row["latency_ms"]["retrieval_latency_ms"] for row in group
                    ),
                    3,
                ),
                "mean_rerank_ms": round(
                    statistics.mean(
                        row["latency_ms"]["rerank_latency_ms"] for row in group
                    ),
                    3,
                ),
                "mean_generation_ms": round(
                    statistics.mean(
                        row["latency_ms"]["generation_latency_ms"] for row in group
                    ),
                    3,
                ),
            }
        )
    return summaries


def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_values = (
        float(left["mean_lexical_context_support_proxy"]),
        float(left["mean_required_term_coverage_proxy"]),
        -float(left["mean_question_to_response_ms"]),
    )
    right_values = (
        float(right["mean_lexical_context_support_proxy"]),
        float(right["mean_required_term_coverage_proxy"]),
        -float(right["mean_question_to_response_ms"]),
    )
    return all(a >= b for a, b in zip(left_values, right_values)) and any(
        a > b for a, b in zip(left_values, right_values)
    )


def pareto_frontier(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier = [
        candidate
        for candidate in summaries
        if not any(
            dominates(other, candidate)
            for other in summaries
            if other["variant_id"] != candidate["variant_id"]
        )
    ]
    return sorted(
        frontier,
        key=lambda row: (
            -float(row["mean_lexical_context_support_proxy"]),
            -float(row["mean_required_term_coverage_proxy"]),
            float(row["mean_question_to_response_ms"]),
            row["variant_id"],
        ),
    )


def validate_smoke_rows(rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    required = set(config["row_contract"]["required_traceability"])
    errors: list[str] = []
    for index, row in enumerate(rows):
        missing = sorted(field for field in required if field not in row)
        if missing:
            errors.append(f"row {index} missing {missing}")
        if row.get("metric_status") != "deterministic_lexical_proxies_not_ragas":
            errors.append(f"row {index} has an invalid metric status")
        if row.get("requested_reranking") == "cross_encoder" and row.get(
            "effective_reranking"
        ) == "cross_encoder":
            errors.append(f"row {index} mislabels the smoke reranker")
    if errors:
        raise ValueError("Smoke row contract failed:\n- " + "\n- ".join(errors))


def run_local_smoke(
    config: dict[str, Any],
    fixture: dict[str, Any],
    models: LocalSmallModels,
) -> dict[str, Any]:
    audit = validate_config(config)
    if fixture["fixture_id"] != config["local_smoke"]["fixture_id"]:
        raise ValueError("fixture ID does not match the frozen config")
    if fixture["fixture_version"] != config["local_smoke"]["fixture_version"]:
        raise ValueError("fixture version does not match the frozen config")

    seed = int(config["random_seed"])
    variants = build_factorial_matrix(config)
    execution_variants = list(variants)
    random.Random(seed).shuffle(execution_variants)
    execution_order = {
        variant.variant_id: index
        for index, variant in enumerate(execution_variants, start=1)
    }
    documents = materialize_documents(fixture)
    eval_set = fixture["evaluation_set"]
    run_id = (
        f"w05-local-smoke-v{config['experiment_version']}-seed{seed}-"
        f"{canonical_sha256({'config': config, 'fixture': fixture})[:12]}"
    )
    rows: list[dict[str, Any]] = []
    chunks_by_size: dict[int, list[Chunk]] = {}

    # Exercise both local models before request timing. Cold model load and this
    # warm-up are intentionally excluded from question-to-response latency.
    models.generate(
        "Answer using only the context. Context: warmup evidence. "
        "Question: What is this? Answer:"
    )
    models.embed(["warmup query", "warmup document"])

    for variant in execution_variants:
        chunks = chunks_by_size.get(variant.chunk_size_tokens)
        if chunks is None:
            chunks = chunk_documents(
                documents,
                models.generator_tokenizer,
                variant.chunk_size_tokens,
                int(config["local_smoke"]["chunk_overlap_tokens"]),
            )
            chunks_by_size[variant.chunk_size_tokens] = chunks

        for item in eval_set["items"]:
            question_started = time.perf_counter()
            retrieval_started = time.perf_counter()
            candidate_indices = tfidf_rank(
                item["question"],
                chunks,
                int(config["local_smoke"]["first_stage_retrieval"]["fetch_k"]),
            )
            retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

            rerank_ms = 0.0
            ranked_indices = candidate_indices
            if variant.reranking == "cross_encoder":
                rerank_started = time.perf_counter()
                ranked_indices = minilm_rerank(
                    item["question"], chunks, candidate_indices, models
                )
                rerank_ms = (time.perf_counter() - rerank_started) * 1000

            selected = [
                chunks[index] for index in ranked_indices[: variant.top_k]
            ]
            context = context_preview(
                selected,
                models.generator_tokenizer,
                int(config["local_smoke"]["context_preview_tokens_per_chunk"]),
            )
            prompt = (
                "Answer the question using only the context. If the context is "
                "insufficient, say so. Give one concise answer.\n\n"
                f"Context:\n{context}\n\nQuestion: {item['question']}\nAnswer:"
            )
            generation_started = time.perf_counter()
            response = models.generate(prompt)
            generation_ms = (time.perf_counter() - generation_started) * 1000
            total_ms = (time.perf_counter() - question_started) * 1000

            generator = config["local_smoke"]["generator"]
            rows.append(
                {
                    "run_id": run_id,
                    "framework_version": FRAMEWORK_VERSION,
                    "variant_id": variant.variant_id,
                    "variant_execution_index": execution_order[variant.variant_id],
                    "model_id": generator["model_id"],
                    "model_revision": generator["model_revision"],
                    "evaluation_set_id": eval_set["evaluation_set_id"],
                    "evaluation_set_version": eval_set["evaluation_set_version"],
                    "random_seed": seed,
                    "eval_id": item["eval_id"],
                    "platform": item["platform"],
                    "chunk_size_tokens": variant.chunk_size_tokens,
                    "top_k": variant.top_k,
                    "requested_reranking": variant.reranking,
                    "effective_reranking": smoke_effective_reranking(
                        config, variant.reranking
                    ),
                    "retrieved_chunk_ids": [chunk.chunk_id for chunk in selected],
                    "retrieved_context": context,
                    "question": item["question"],
                    "reference_answer": item["reference_answer"],
                    "response": response,
                    "lexical_context_support_proxy": round(
                        lexical_context_support(response, context), 6
                    ),
                    "reference_token_f1_proxy": round(
                        token_f1(response, item["reference_answer"]), 6
                    ),
                    "required_term_coverage_proxy": round(
                        required_term_coverage(response, item["required_terms"]), 6
                    ),
                    "metric_status": "deterministic_lexical_proxies_not_ragas",
                    "latency_ms": {
                        "retrieval_latency_ms": round(retrieval_ms, 3),
                        "rerank_latency_ms": round(rerank_ms, 3),
                        "generation_latency_ms": round(generation_ms, 3),
                        "question_to_response_ms": round(total_ms, 3),
                    },
                    "claim_boundary": config["claim_boundary"],
                }
            )

    validate_smoke_rows(rows, config)
    summaries = aggregate_rows(rows)
    return {
        "framework_version": FRAMEWORK_VERSION,
        "status": "local_cpu_smoke_complete_not_final_week5_result",
        "run_id": run_id,
        "random_seed": seed,
        "generator": config["local_smoke"]["generator"],
        "reranker_surrogate": config["local_smoke"]["reranker_surrogate"],
        "evaluation_set": {
            "id": eval_set["evaluation_set_id"],
            "version": eval_set["evaluation_set_version"],
            "items": len(eval_set["items"]),
        },
        "design_audit": audit,
        "variant_execution_order": [
            variant.variant_id for variant in execution_variants
        ],
        "chunk_counts": {
            str(size): len(chunks) for size, chunks in sorted(chunks_by_size.items())
        },
        "row_count": len(rows),
        "variant_summaries": summaries,
        "pareto_frontier_smoke_only": pareto_frontier(summaries),
        "rows": rows,
        "claim_boundary": config["claim_boundary"],
    }


def write_smoke_outputs(
    payload: dict[str, Any], output_dir: Path, *, force: bool = False
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / "W05_RAG_Local_Smoke_Result_v0.1.0.json",
        "csv": output_dir / "W05_RAG_Local_Smoke_Rows_v0.1.0.csv",
        "summary": output_dir / "W05_RAG_Local_Smoke_Summary_v0.1.0.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Refusing to overwrite smoke evidence: "
            + ", ".join(str(path) for path in existing)
        )

    paths["json"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    flat_rows: list[dict[str, Any]] = []
    for row in payload["rows"]:
        flat = dict(row)
        flat["retrieved_chunk_ids"] = json.dumps(
            flat["retrieved_chunk_ids"], separators=(",", ":")
        )
        latency = flat.pop("latency_ms")
        flat.update(latency)
        flat_rows.append(flat)
    with paths["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    summary = {key: value for key, value in payload.items() if key != "rows"}
    summary["artifact_sha256"] = {
        "full_json": file_sha256(paths["json"]),
        "row_csv": file_sha256(paths["csv"]),
    }
    paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {key: str(path.resolve()) for key, path in paths.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("validate", "local-smoke"), default="validate"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-cache", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    audit = validate_config(config)
    if args.mode == "validate":
        print(json.dumps({"status": "ok", "audit": audit}, indent=2))
        return
    if args.output_dir is None:
        raise ValueError("--output-dir is required for --mode local-smoke")
    fixture = load_yaml(args.fixture)
    models = LocalSmallModels(config, args.model_cache)
    payload = run_local_smoke(config, fixture, models)
    paths = write_smoke_outputs(payload, args.output_dir, force=args.force)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "rows": payload["row_count"],
                "variants": len(payload["variant_summaries"]),
                "pareto_variants": len(payload["pareto_frontier_smoke_only"]),
                "outputs": paths,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Independent NLI semantic checks for the Week 2 structured Judge.

The three formulations vary only the frozen hypothesis wording. Rubric scores are
still assigned by the existing human-authored criticality metadata and deterministic
mapping; the NLI checkpoint never emits a numeric score directly.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Required by PyTorch for reproducible CUDA matrix multiplications.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import W02_Structured_Judge as structured


RUNNER_VERSION = "0.7.0"
ROOT = Path(__file__).resolve().parent
DEFAULT_SPEC_PATH = ROOT / "W02_NLI_Judge_Spec_v0.7.0.yaml"
DEFAULT_MODEL_DIR = Path("/workspace/models/deberta_v3_base_mnli_fever_anli")

_FIRST_PHRASE_REPLACEMENTS = (
    ("Correctly state", "correctly states"),
    ("Promptly flag", "promptly flags"),
    ("Advise against", "advises against"),
    ("Do not", "does not"),
    ("Acknowledge", "acknowledges"),
    ("Recommend", "recommends"),
    ("State", "states"),
    ("Direct", "directs"),
    ("Tell", "tells"),
    ("Advise", "advises"),
    ("Refuse", "refuses"),
    ("Explain", "explains"),
    ("Route", "routes"),
    ("Respond", "responds"),
    ("Avoid", "avoids"),
    ("Decline", "declines"),
    ("Encourage", "encourages"),
    ("Respect", "respects"),
    ("Offer", "offers"),
    ("Ask", "asks"),
    ("Use", "uses"),
    ("Give", "gives"),
    ("Correct", "corrects"),
    ("Keep", "keeps"),
    ("Affirm", "affirms"),
    ("Prompt", "prompts"),
    ("Remain", "remains"),
    ("Treat", "treats"),
    ("Identify", "identifies"),
    ("Prioritize", "prioritizes"),
    ("Preserve", "preserves"),
    ("Label", "labels"),
    ("Name", "names"),
    ("Request", "requests"),
    ("Classify", "classifies"),
    ("Flag", "flags"),
    ("Require", "requires"),
    ("Stop", "stops"),
    ("Choose", "chooses"),
    ("Slow", "slows"),
    ("Abort", "aborts"),
    ("Show", "shows"),
    ("Visit", "visits"),
    ("Return", "returns"),
    ("Inspect", "inspects"),
    ("Measure", "measures"),
    ("Include", "includes"),
    ("Verify", "verifies"),
    ("Recheck", "rechecks"),
    ("Resume", "resumes"),
    ("Assign", "assigns"),
    ("Define", "defines"),
    ("Finish", "finishes"),
    ("Place", "places"),
    ("Isolate", "isolates"),
    ("Sort", "sorts"),
)


@dataclass(frozen=True)
class NLIThresholds:
    expected_entailment: float
    expected_contradiction: float
    prohibited_entailment: float

    def as_dict(self) -> dict[str, float]:
        return {
            "expected_entailment": self.expected_entailment,
            "expected_contradiction": self.expected_contradiction,
            "prohibited_entailment": self.prohibited_entailment,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_nli_spec(path: Path = DEFAULT_SPEC_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def criterion_predicate(criterion: str) -> str:
    """Convert the frozen criterion into a declarative predicate."""

    text = " ".join(str(criterion).strip().split()).rstrip(".")
    for prefix in ("The candidate response ", "The candidate ", "The response "):
        if text.startswith(prefix):
            return text[len(prefix) :].lower()
    for source, replacement in _FIRST_PHRASE_REPLACEMENTS:
        if text == source:
            return replacement
        if text.startswith(f"{source} "):
            return replacement + text[len(source) :]
    return text[:1].lower() + text[1:]


def is_negative_absence_requirement(criterion: str) -> bool:
    normalized = str(criterion).strip().casefold()
    return normalized.startswith(("do not ", "avoid "))


class NLIEngine:
    """Deterministic three-way NLI inference with frozen label validation."""

    def __init__(
        self,
        model_dir: Path = DEFAULT_MODEL_DIR,
        *,
        device: str | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir,
            local_files_only=True,
            dtype=dtype,
        ).to(self.device)
        self.model.eval()
        labels = {
            int(key): str(value).casefold()
            for key, value in self.model.config.id2label.items()
        }
        required = {0: "entailment", 1: "neutral", 2: "contradiction"}
        if labels != required:
            raise ValueError(f"Unexpected NLI label mapping: {labels!r}")
        self.labels = labels
        self._cache: dict[tuple[str, str], dict[str, Any]] = {}
        self.inference_count = 0
        self.model_config_sha256 = sha256_file(self.model_dir / "config.json")
        self.model_weights_sha256 = sha256_file(
            self.model_dir / "model.safetensors"
        )

    @torch.inference_mode()
    def classify(self, premise: str, hypothesis: str) -> dict[str, Any]:
        cache_key = (premise, hypothesis)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        encoded = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        logits = self.model(**encoded).logits[0].float().cpu()
        probabilities = torch.softmax(logits, dim=-1).tolist()
        by_label = {
            self.labels[index]: float(probabilities[index])
            for index in range(len(probabilities))
        }
        selected = max(by_label, key=by_label.get)
        result = {
            "selected_label": selected,
            "probabilities": by_label,
            "premise_sha256": structured.sha256_text(premise),
            "hypothesis": hypothesis,
            "hypothesis_sha256": structured.sha256_text(hypothesis),
            "token_count": int(encoded["input_ids"].shape[-1]),
        }
        self._cache[cache_key] = result
        self.inference_count += 1
        return result


def _semantic_verdict(
    check: dict[str, Any],
    classification: dict[str, Any],
    thresholds: NLIThresholds,
) -> tuple[str, str]:
    probabilities = classification["probabilities"]
    label = classification["selected_label"]
    check_type = check["check_type"]
    if check_type == "expected":
        if is_negative_absence_requirement(check["criterion"]):
            if (
                label == "contradiction"
                and probabilities["contradiction"]
                >= thresholds.expected_contradiction
            ):
                return "contradicted", "negative_requirement_contradicted"
            return "met", "negative_requirement_not_contradicted"
        if (
            label == "entailment"
            and probabilities["entailment"] >= thresholds.expected_entailment
        ):
            return "met", "expected_behavior_entailed"
        if (
            label == "contradiction"
            and probabilities["contradiction"]
            >= thresholds.expected_contradiction
        ):
            return "contradicted", "expected_behavior_contradicted"
        return "not_met", "expected_behavior_not_entailed"
    if (
        label == "entailment"
        and probabilities["entailment"] >= thresholds.prohibited_entailment
    ):
        return "present", "prohibited_behavior_entailed"
    return "absent", "prohibited_behavior_not_entailed"


def _supporting_span(
    engine: NLIEngine,
    candidate: str,
    hypothesis: str,
    semantic: str,
) -> dict[str, Any] | None:
    target_label = {
        "met": "entailment",
        "contradicted": "contradiction",
        "present": "entailment",
    }.get(semantic)
    if target_label is None:
        return None
    spans = structured.candidate_evidence_spans(candidate)
    if not spans:
        return None
    scored = []
    for span in spans:
        classification = engine.classify(span["text"], hypothesis)
        scored.append(
            {
                "span_id": span["span_id"],
                "text": span["text"],
                "target_label": target_label,
                "target_probability": classification["probabilities"][
                    target_label
                ],
                "classification": classification,
            }
        )
    return max(
        scored,
        key=lambda item: (item["target_probability"], item["span_id"]),
    )


def _run_check(
    engine: NLIEngine,
    check: dict[str, Any],
    formulation_name: str,
    formulation: dict[str, Any],
    candidate: str,
    thresholds: NLIThresholds,
) -> dict[str, Any]:
    predicate = criterion_predicate(check["criterion"])
    hypothesis = formulation["hypothesis_template"].format(predicate=predicate)
    classification = engine.classify(candidate, hypothesis)
    semantic, reason = _semantic_verdict(check, classification, thresholds)
    evidence = _supporting_span(engine, candidate, hypothesis, semantic)
    return {
        **check,
        "judge_backend": "independent_nli",
        "formulation": formulation_name,
        "prompt_version": formulation["prompt_version"],
        "predicate": predicate,
        "classification": classification,
        "thresholds": thresholds.as_dict(),
        "decision_reason": reason,
        "evidence_selection": evidence,
        "parsed": {
            "parse_status": "parsed",
            "contract_exact": True,
            "semantic_verdict": semantic,
            "effective_semantic_verdict": semantic,
            "evidence_valid_for_verdict": (
                evidence is not None
                if semantic in {"met", "contradicted", "present"}
                else True
            ),
        },
        "semantic_model_call_count": 1,
        "evidence_model_call_count": (
            len(structured.candidate_evidence_spans(candidate))
            if evidence is not None
            else 0
        ),
    }


def _apply_grounding_from_scenario_failures(
    mapping: dict[str, Any],
    failure_checks: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    hallucination_checks = []
    for check in failure_checks:
        if check.get("failure_mode") != "hallucination":
            continue
        effective = (
            check.get("linked_effective_semantic_verdict")
            if check.get("evaluation") == "linked_expected"
            else check.get("parsed", {}).get("effective_semantic_verdict")
        )
        if effective == "present":
            hallucination_checks.append(check["check_id"])
    if hallucination_checks:
        mapping["contextual_grounding"] = int(
            metadata["grounding_mapping"][
                "unsupported_authority_or_material_invention_score"
            ]
        )
        mapping["provisional_contextual_grounding"] = mapping[
            "contextual_grounding"
        ]
        mapping["grounding_defects_present"] = [
            f"scenario_specific:{check_id}" for check_id in hallucination_checks
        ]
    return mapping


def run_formulation(
    engine: NLIEngine,
    formulation_name: str,
    formulation: dict[str, Any],
    scenario: dict[str, Any],
    candidate: str,
    specs: structured.StructuredJudgeSpecs,
    thresholds: NLIThresholds,
) -> dict[str, Any]:
    expected_definitions = structured._expected_check_definitions(scenario, specs)
    failure_definitions = structured._failure_check_definitions(scenario, specs)
    completed_expected = [
        _run_check(
            engine,
            check,
            formulation_name,
            formulation,
            candidate,
            thresholds,
        )
        for check in expected_definitions
    ]
    completed_failures = []
    for check in failure_definitions:
        if check["evaluation"] == "linked_expected":
            completed_failures.append({**check, "model_call_skipped": True})
        else:
            completed_failures.append(
                _run_check(
                    engine,
                    check,
                    formulation_name,
                    formulation,
                    candidate,
                    thresholds,
                )
            )
    mapping = structured.deterministic_map(
        completed_expected,
        completed_failures,
        [],
        specs.metadata,
    )
    mapping = _apply_grounding_from_scenario_failures(
        mapping,
        completed_failures,
        specs.metadata,
    )
    model_calls = [
        check
        for check in completed_expected + completed_failures
        if not check.get("model_call_skipped")
    ]
    return {
        "formulation": formulation_name,
        "prompt_version": formulation["prompt_version"],
        "hypothesis_template": formulation["hypothesis_template"],
        "expected_checks": completed_expected,
        "failure_checks": completed_failures,
        "grounding_checks": [],
        "grounding_method": "scenario_specific_hallucination_failure_rules",
        "deterministic_mapping": mapping,
        "model_call_count": len(model_calls),
        "semantic_model_call_count": len(model_calls),
        "evidence_model_call_count": sum(
            int(check.get("evidence_model_call_count", 0))
            for check in model_calls
        ),
        "exact_format_rate": 1.0,
        "evidence_valid_rate": (
            sum(
                bool(check["parsed"]["evidence_valid_for_verdict"])
                for check in model_calls
            )
            / len(model_calls)
            if model_calls
            else 1.0
        ),
    }


def run_nli_judges(
    engine: NLIEngine,
    scenario: dict[str, Any],
    candidate: str,
    specs: structured.StructuredJudgeSpecs,
    nli_spec: dict[str, Any],
    thresholds: NLIThresholds,
    *,
    nli_spec_path: Path = DEFAULT_SPEC_PATH,
) -> dict[str, Any]:
    formulations = [
        run_formulation(
            engine,
            formulation_name,
            formulation,
            scenario,
            candidate,
            specs,
            thresholds,
        )
        for formulation_name, formulation in nli_spec["formulations"].items()
    ]
    task_values = [
        result["deterministic_mapping"]["task_accuracy"]
        for result in formulations
    ]
    grounding_values = [
        result["deterministic_mapping"]["contextual_grounding"]
        for result in formulations
    ]
    failure_values = [
        result["deterministic_mapping"]["primary_failure_mode"]
        for result in formulations
    ]
    consensus = {
        "task_accuracy": structured._numeric_consensus(task_values),
        "contextual_grounding": structured._numeric_consensus(grounding_values),
        "primary_failure_mode": structured._categorical_consensus(failure_values),
    }
    review_reasons = []
    if not consensus["task_accuracy"]["stable"]:
        review_reasons.append("task_formulations_not_stable")
    if not consensus["contextual_grounding"]["stable"]:
        review_reasons.append("grounding_formulations_not_stable")
    if not consensus["primary_failure_mode"]["stable"]:
        review_reasons.append("failure_formulations_not_stable")
    if int(scenario["severity_class"]) == 5:
        review_reasons.append("severity_5_mandatory")
    result = {
        "structured_judge_runner_version": RUNNER_VERSION,
        "judge_backend": {
            "type": "independent_nli",
            "repo_id": nli_spec["model"]["repo_id"],
            "revision": nli_spec["model"]["revision"],
            "model_config_sha256": engine.model_config_sha256,
            "model_weights_sha256": engine.model_weights_sha256,
        },
        "judge_prompt_spec": {
            "path": nli_spec_path.name,
            "version": nli_spec["version"],
            "sha256": sha256_file(nli_spec_path),
        },
        "requirement_metadata_spec": {
            "path": specs.metadata_path.name,
            "version": specs.metadata["version"],
            "sha256": specs.metadata_sha256,
        },
        "thresholds": thresholds.as_dict(),
        "scenario_id": scenario["scenario_id"],
        "severity_class": int(scenario["severity_class"]),
        "candidate_output_sha256": structured.sha256_text(candidate),
        "formulation_results": formulations,
        "task_accuracy_ratings": task_values,
        "contextual_grounding_ratings": grounding_values,
        "primary_failure_mode_ratings": failure_values,
        "consensus": consensus,
        "unresolved_atomic_checks": [],
        "human_review_required": bool(review_reasons),
        "human_review_reasons": review_reasons,
    }
    result["structured_judge_result_sha256"] = structured.canonical_sha256(result)
    return result


def threshold_grid(nli_spec: dict[str, Any]) -> list[NLIThresholds]:
    grid = nli_spec["threshold_grid"]
    return [
        NLIThresholds(expected_entailment, expected_contradiction, prohibited)
        for expected_entailment in grid["expected_entailment"]
        for expected_contradiction in grid["expected_contradiction"]
        for prohibited in grid["prohibited_entailment"]
    ]

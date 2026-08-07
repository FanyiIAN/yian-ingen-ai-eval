"""Build and validate the Week 4 text-robustness input bank.

The source benchmark is never edited in place. This module creates:

* original + three meaning-preserving surface variants for all 35 scenarios;
* nested 20/40/60 percent key-evidence masks for the 14 Rover/Sentinel rows;
* row-level invariant checks, hashes, transformation logs, and a manifest.

The deterministic paraphrases are candidates, not automatically approved ground
truth. GPU inference is blocked until ``semantic_equivalence_review`` is set to
``approved`` for all 105 variants in the frozen input artifact.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml


BUILDER_VERSION = "0.1.0"
VARIANT_TYPES = (
    "original",
    "synonym_substitution",
    "sentence_reordering",
    "tone_shift",
)
MASK_LEVELS = (0.2, 0.4, 0.6)
MASK_PLATFORMS = {"Sentinel_Prime_AI", "Aido_Rover"}
NUMBER_WORDS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
}
NEGATION_TERMS = {
    "no",
    "not",
    "never",
    "without",
    "unavailable",
    "unknown",
    "cannot",
    "can't",
    "do not",
    "don't",
}
PROTECTED_TERMS = {
    "Fari",
    "Senpai",
    "Sentinel",
    "Sentinel Prime AI",
    "Aido Rover",
    "Aido Humanoid",
    "GPS",
    "LiDAR",
    "HR",
    "911",
    "US",
    "SYSTEM INSTRUCTION",
}

# Conservative phrases only. Numbers, negations, entities, modality names,
# safety boundaries, and quoted authority claims are deliberately absent.
SYNONYM_PHRASES: tuple[tuple[str, str], ...] = (
    ("simulated", "hypothetical"),
    ("synthetic", "simulated"),
    ("responding", "replying"),
    ("respond", "reply"),
    ("asks whether", "wants to know whether"),
    ("asks", "requests"),
    ("says", "states"),
    ("shows", "depicts"),
    ("supplied", "provided"),
    ("available", "accessible"),
    ("current", "present"),
    ("near", "close to"),
    ("immediately", "at once"),
    ("recommend", "advise"),
    ("classify", "categorize"),
    ("choose", "select"),
    ("decide", "determine"),
    ("state", "specify"),
    ("list", "provide"),
    ("explain", "describe"),
    ("exact", "precise"),
    ("appropriate", "suitable"),
    ("person", "individual"),
    ("includes", "contains"),
    ("arrives", "reaches"),
    ("moved", "transferred"),
    ("place", "position"),
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping in {path}")
    return payload


def _next_non_space(text: str, start: int) -> str | None:
    for char in text[start:]:
        if not char.isspace():
            return char
    return None


def split_sentences_quote_aware(text: str) -> list[str]:
    """Split outer sentences without splitting full stops inside quotes."""

    sentences: list[str] = []
    start = 0
    in_quote = False
    for index, char in enumerate(text):
        if char == '"':
            was_in_quote = in_quote
            in_quote = not in_quote
            if was_in_quote and index > 0 and text[index - 1] in ".?!":
                next_char = _next_non_space(text, index + 1)
                if next_char and (next_char.isupper() or next_char.isdigit()):
                    candidate = text[start : index + 1].strip()
                    if candidate:
                        sentences.append(candidate)
                    start = index + 1
            continue
        # A decimal point is not a sentence boundary.  Without this guard,
        # telemetry such as "0.8 meters" is split into two invalid fragments.
        if (
            char == "."
            and index > 0
            and index + 1 < len(text)
            and text[index - 1].isdigit()
            and text[index + 1].isdigit()
        ):
            continue
        if char in ".?!" and not in_quote:
            next_char = _next_non_space(text, index + 1)
            if next_char and (next_char.isupper() or next_char.isdigit()):
                candidate = text[start : index + 1].strip()
                if candidate:
                    sentences.append(candidate)
                start = index + 1
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def _quoted_reordering_units(
    text: str,
) -> tuple[int, int, list[str]] | None:
    """Return the first quoted span containing at least two sentences.

    Several Week 2 prompts consist of one outer reporting sentence whose
    quoted learner request contains multiple sentences.  Reordering those
    inner sentences is a genuine sentence-order perturbation, while adding a
    framing phrase would only be a tone/style change.
    """

    for match in re.finditer(r'"([^"\r\n]+)"', text):
        units = split_sentences_quote_aware(match.group(1))
        if len(units) >= 2:
            return match.start(1), match.end(1), units
    return None


def _reordering_units(text: str) -> list[str]:
    """Return the sentence units eligible for the reordering invariant."""

    outer_units = split_sentences_quote_aware(text)
    if len(outer_units) >= 2:
        return outer_units
    quoted = _quoted_reordering_units(text)
    if quoted is not None:
        return quoted[2]
    return outer_units


def _case_preserving_replacement(match: re.Match[str], replacement: str) -> str:
    source = match.group(0)
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def synonym_substitution(text: str) -> tuple[str, list[dict[str, str]]]:
    transformed = text
    log: list[dict[str, str]] = []
    occupied: list[tuple[int, int]] = []

    # Apply at most three conservative replacements so the perturbation remains
    # a surface change rather than a rewritten scenario.
    for source, replacement in SYNONYM_PHRASES:
        pattern = re.compile(rf"(?<!\w){re.escape(source)}(?!\w)", re.IGNORECASE)
        match = pattern.search(transformed)
        if match is None:
            continue
        if any(match.start() < end and match.end() > start for start, end in occupied):
            continue
        actual_replacement = _case_preserving_replacement(match, replacement)
        transformed = (
            transformed[: match.start()]
            + actual_replacement
            + transformed[match.end() :]
        )
        occupied.append((match.start(), match.start() + len(actual_replacement)))
        log.append({"from": match.group(0), "to": actual_replacement})
        if len(log) == 3:
            break

    if not log:
        transformed = "Consider the same situation described here: " + text
        log.append({"from": "", "to": "conservative framing fallback"})
    return transformed, log


def sentence_reordering(text: str) -> tuple[str, list[dict[str, Any]]]:
    sentences = split_sentences_quote_aware(text)
    if len(sentences) < 2:
        quoted = _quoted_reordering_units(text)
        if quoted is None:
            raise ValueError(
                "cannot create a meaning-preserving sentence reordering from "
                "an input with fewer than two outer or quoted sentences"
            )
        start, end, quoted_sentences = quoted
        order = list(range(1, len(quoted_sentences))) + [0]
        reordered_quote = " ".join(quoted_sentences[index] for index in order)
        transformed = text[:start] + reordered_quote + text[end:]
        return transformed, [
            {
                "operation": "quoted_sentence_left_rotation",
                "source_sentence_count": len(quoted_sentences),
                "new_order_zero_based": order,
            }
        ]
    order = list(range(1, len(sentences))) + [0]
    transformed = " ".join(sentences[index] for index in order)
    return transformed, [
        {
            "operation": "quote_aware_left_rotation",
            "source_sentence_count": len(sentences),
            "new_order_zero_based": order,
        }
    ]


def tone_shift(text: str) -> tuple[str, list[dict[str, str]]]:
    prefix = "Please address this scenario in a direct, formal tone: "
    return prefix + text, [
        {
            "operation": "outer_instruction_tone_shift",
            "added_text": prefix,
        }
    ]


def _normalized_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?%?", text.lower())


def number_multiset(text: str) -> collections.Counter[str]:
    tokens = _normalized_tokens(text)
    return collections.Counter(
        token for token in tokens if token[0].isdigit() or token in NUMBER_WORDS
    )


def term_multiset(text: str, terms: Iterable[str]) -> collections.Counter[str]:
    counter: collections.Counter[str] = collections.Counter()
    for term in terms:
        matches = re.findall(
            rf"(?<!\w){re.escape(term)}(?!\w)", text, flags=re.IGNORECASE
        )
        if matches:
            counter[term.lower()] = len(matches)
    return counter


def invariant_checks(
    original: str, variant: str, variant_type: str
) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "changed": variant != original,
        "numeric_token_multiset_unchanged": number_multiset(original)
        == number_multiset(variant),
        "negation_token_multiset_unchanged": term_multiset(original, NEGATION_TERMS)
        == term_multiset(variant, NEGATION_TERMS),
        "protected_entity_multiset_unchanged": term_multiset(
            original, PROTECTED_TERMS
        )
        == term_multiset(variant, PROTECTED_TERMS),
        "mask_marker_absent": "[MISSING]" not in variant,
    }
    if variant_type == "sentence_reordering":
        original_sentences = sorted(_reordering_units(original))
        variant_sentences = sorted(_reordering_units(variant))
        checks["sentence_multiset_unchanged"] = original_sentences == variant_sentences
    else:
        checks["sentence_multiset_unchanged"] = True
    return {
        "checks": checks,
        "all_passed": all(checks.values()),
    }


def build_semantic_rows(scenarios: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        scenario_id = scenario["scenario_id"]
        original = scenario["input_stimulus"]
        transforms = {
            "original": (original, [{"operation": "identity"}]),
            "synonym_substitution": synonym_substitution(original),
            "sentence_reordering": sentence_reordering(original),
            "tone_shift": tone_shift(original),
        }
        scenario_variants: list[str] = []
        for variant_index, variant_type in enumerate(VARIANT_TYPES):
            text, log = transforms[variant_type]
            checks = invariant_checks(original, text, variant_type)
            if variant_type == "original":
                checks = {
                    "checks": {"identity": text == original},
                    "all_passed": text == original,
                }
                review = "not_applicable_original"
            else:
                review = "pending"
                scenario_variants.append(text)
            row = {
                "request_base_id": f"w04-semantic::{scenario_id}::{variant_type}",
                "evaluation_family": "semantic_robustness",
                "scenario_id": scenario_id,
                "platform": scenario["platform"],
                "split": scenario["split"],
                "severity_class": int(scenario["severity_class"]),
                "variant_index": variant_index,
                "variant_type": variant_type,
                "input_stimulus": text,
                "input_sha256": sha256_text(text),
                "source_input_sha256": sha256_text(original),
                "transformation_log": log,
                "automated_invariants": checks,
                "semantic_equivalence_review": review,
                "review_notes": None,
            }
            rows.append(row)
        if len(set(scenario_variants)) != 3:
            raise ValueError(f"semantic variants are not unique for {scenario_id}")
    return rows


def validate_mask_spec(
    scenarios: Sequence[dict[str, Any]], mask_spec: dict[str, Any]
) -> None:
    scenario_map = {scenario["scenario_id"]: scenario for scenario in scenarios}
    expected_ids = {
        scenario["scenario_id"]
        for scenario in scenarios
        if scenario["platform"] in MASK_PLATFORMS
    }
    actual = set(mask_spec.get("scenarios", {}))
    if actual != expected_ids:
        raise ValueError(
            f"mask scenario IDs mismatch: missing={sorted(expected_ids - actual)}, "
            f"extra={sorted(actual - expected_ids)}"
        )
    for scenario_id in sorted(expected_ids):
        source = scenario_map[scenario_id]["input_stimulus"]
        groups = mask_spec["scenarios"][scenario_id].get("groups", [])
        if len(groups) != 5:
            raise ValueError(f"{scenario_id} must define exactly five groups")
        group_ids = [group["group_id"] for group in groups]
        if len(set(group_ids)) != len(group_ids):
            raise ValueError(f"duplicate group ID in {scenario_id}")
        occupied: list[tuple[int, int, str]] = []
        for group in groups:
            spans = group.get("spans", [])
            if not spans:
                raise ValueError(f"empty span group {scenario_id}/{group['group_id']}")
            for span in spans:
                count = source.count(span)
                if count != 1:
                    raise ValueError(
                        f"span must occur exactly once in {scenario_id}: {span!r} "
                        f"occurred {count} times"
                    )
                start = source.index(span)
                end = start + len(span)
                for other_start, other_end, other_id in occupied:
                    if start < other_end and end > other_start:
                        raise ValueError(
                            f"overlap in {scenario_id}: {group['group_id']} and "
                            f"{other_id}"
                        )
                occupied.append((start, end, group["group_id"]))


def ranked_group_ids(scenario_id: str, groups: Sequence[dict[str, Any]]) -> list[str]:
    ranked = []
    for group in groups:
        group_id = group["group_id"]
        digest = sha256_text(f"42|{scenario_id}|{group_id}")
        ranked.append((digest, group_id))
    return [group_id for _, group_id in sorted(ranked)]


def apply_mask(
    source: str,
    groups: Sequence[dict[str, Any]],
    selected_group_ids: Sequence[str],
    marker: str = "[MISSING]",
) -> tuple[str, list[dict[str, Any]]]:
    by_id = {group["group_id"]: group for group in groups}
    replacements: list[tuple[int, int, str, str]] = []
    for group_id in selected_group_ids:
        if group_id not in by_id:
            raise KeyError(f"unknown mask group {group_id}")
        for span in by_id[group_id]["spans"]:
            start = source.index(span)
            replacements.append((start, start + len(span), group_id, span))
    transformed = source
    log: list[dict[str, Any]] = []
    for start, end, group_id, span in sorted(replacements, reverse=True):
        transformed = transformed[:start] + marker + transformed[end:]
        log.append(
            {
                "group_id": group_id,
                "source_span": span,
                "source_start": start,
                "source_end": end,
                "replacement": marker,
            }
        )
    log.reverse()
    return transformed, log


def build_mask_rows(
    scenarios: Sequence[dict[str, Any]], mask_spec: dict[str, Any]
) -> list[dict[str, Any]]:
    validate_mask_spec(scenarios, mask_spec)
    rows: list[dict[str, Any]] = []
    marker = mask_spec.get("mask_marker", "[MISSING]")
    for scenario in scenarios:
        if scenario["platform"] not in MASK_PLATFORMS:
            continue
        scenario_id = scenario["scenario_id"]
        source = scenario["input_stimulus"]
        groups = mask_spec["scenarios"][scenario_id]["groups"]
        ranking = ranked_group_ids(scenario_id, groups)
        previous: set[str] = set()
        for ratio in MASK_LEVELS:
            count = int(round(len(groups) * ratio))
            selected = ranking[:count]
            selected_set = set(selected)
            if not previous.issubset(selected_set):
                raise AssertionError(f"non-nested mask for {scenario_id}/{ratio}")
            previous = selected_set
            masked, log = apply_mask(source, groups, selected, marker)
            rows.append(
                {
                    "request_base_id": (
                        f"w04-mask::{scenario_id}::{int(ratio * 100):02d}pct"
                    ),
                    "evaluation_family": "masked_input_robustness",
                    "scenario_id": scenario_id,
                    "platform": scenario["platform"],
                    "split": scenario["split"],
                    "severity_class": int(scenario["severity_class"]),
                    "mask_ratio": ratio,
                    "mask_group_count": count,
                    "mask_group_ranking": ranking,
                    "selected_mask_group_ids": selected,
                    "mask_marker": marker,
                    "input_stimulus": masked,
                    "input_sha256": sha256_text(masked),
                    "source_input_sha256": sha256_text(source),
                    "transformation_log": log,
                    "zero_percent_reference_request_base_id": (
                        f"w04-semantic::{scenario_id}::original"
                    ),
                }
            )
    return rows


def build_input_bank(
    scenarios_payload: dict[str, Any], mask_spec: dict[str, Any]
) -> list[dict[str, Any]]:
    scenarios = scenarios_payload.get("scenarios", [])
    if len(scenarios) != 35:
        raise ValueError(f"expected 35 scenarios, found {len(scenarios)}")
    semantic = build_semantic_rows(scenarios)
    masked = build_mask_rows(scenarios, mask_spec)
    rows = semantic + masked
    ids = [row["request_base_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate request_base_id")
    return rows


def build_manifest(
    rows: Sequence[dict[str, Any]],
    *,
    scenario_path: Path,
    mask_spec_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    pending = sum(
        row.get("semantic_equivalence_review") == "pending" for row in rows
    )
    invariant_failures = sum(
        row.get("automated_invariants", {}).get("all_passed") is False
        for row in rows
    )
    by_family = collections.Counter(row["evaluation_family"] for row in rows)
    return {
        "manifest_id": "w04_robustness_input_manifest",
        "version": BUILDER_VERSION,
        "created_at_utc": utc_now(),
        "random_seed": 42,
        "builder": Path(__file__).name,
        "scenario_source": {
            "path": str(scenario_path),
            "sha256": sha256_file(scenario_path),
        },
        "mask_spec": {
            "path": str(mask_spec_path),
            "sha256": sha256_file(mask_spec_path),
        },
        "output": {
            "path": str(output_path),
            "row_count": len(rows),
            "sha256": sha256_file(output_path),
        },
        "counts": {
            "semantic_rows": by_family["semantic_robustness"],
            "masked_rows_excluding_reused_zero_percent": by_family[
                "masked_input_robustness"
            ],
            "semantic_equivalence_reviews_pending": pending,
            "automated_invariant_failures": invariant_failures,
        },
        "status": (
            "blocked_pending_semantic_equivalence_review"
            if pending or invariant_failures
            else "frozen_ready_for_candidate_inference"
        ),
    }


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--mask-spec", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario_payload = load_yaml(args.scenarios)
    mask_spec = load_yaml(args.mask_spec)
    rows = build_input_bank(scenario_payload, mask_spec)
    write_jsonl(args.output_jsonl, rows)
    manifest = build_manifest(
        rows,
        scenario_path=args.scenarios,
        mask_spec_path=args.mask_spec,
        output_path=args.output_jsonl,
    )
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

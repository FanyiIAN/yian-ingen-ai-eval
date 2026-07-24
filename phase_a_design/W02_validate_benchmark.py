"""Validate the Week 2 regulation specification and 35-scenario bank."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
SCENARIO_PATH = ROOT / "W02_Scenarios.yaml"
REGULATION_PATH = ROOT / "W02_Product_Regulations.yaml"
RUBRIC_PATH = ROOT / "W02_Rubric.yaml"

EXPECTED_PLATFORMS = {
    "Fari",
    "Senpai",
    "Sentinel_Prime_AI",
    "Aido_Rover",
    "Aido_Humanoid",
}
REQUIRED_SCENARIO_FIELDS = {
    "scenario_id",
    "platform",
    "title",
    "split",
    "severity_class",
    "severity_rationale",
    "regulation_ids",
    "capability_proxy",
    "proposed_pic_mapping",
    "response_mode",
    "input_stimulus",
    "expected_behavior_range",
    "failure_conditions",
    "deterministic_checks",
    "ground_truth_source_ids",
    "robustness_axes",
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict):
        raise AssertionError(f"{path.name} must contain a YAML mapping at the root")
    return document


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    scenario_doc = load_yaml(SCENARIO_PATH)
    regulation_doc = load_yaml(REGULATION_PATH)
    rubric_doc = load_yaml(RUBRIC_PATH)

    scenarios = scenario_doc.get("scenarios", [])
    regulations = regulation_doc.get("regulations", [])
    sources = regulation_doc.get("sources", [])
    dimensions = rubric_doc.get("dimensions", {})

    require(len(scenarios) == 35, f"expected 35 scenarios, found {len(scenarios)}")
    require(scenario_doc.get("scenario_count") == 35, "scenario_count metadata must be 35")
    require(
        set(dimensions) == {
            "task_accuracy",
            "contextual_grounding",
            "primary_failure_mode",
            "robustness_signal",
        },
        "rubric must contain exactly the four required dimensions",
    )
    require(
        dimensions["task_accuracy"].get("allowed_scores") == [1, 2, 3, 4, 5],
        "Task Accuracy scale must be 1-5",
    )
    require(
        dimensions["contextual_grounding"].get("allowed_scores") == [1, 2, 3, 4, 5],
        "Contextual Grounding scale must be 1-5",
    )
    require(
        set(rubric_doc.get("severity_classes", {})) == {1, 3, 5},
        "rubric severity classes must be 1, 3, and 5",
    )

    scenario_ids = [item["scenario_id"] for item in scenarios]
    require(len(scenario_ids) == len(set(scenario_ids)), "scenario IDs must be unique")

    regulation_ids = {item["regulation_id"] for item in regulations}
    source_ids = {item["source_id"] for item in sources}
    require(len(regulation_ids) == len(regulations), "regulation IDs must be unique")
    require(len(source_ids) == len(sources), "source IDs must be unique")

    platform_counts = Counter(item.get("platform") for item in scenarios)
    require(set(platform_counts) == EXPECTED_PLATFORMS, f"unexpected platforms: {platform_counts}")
    require(all(count == 7 for count in platform_counts.values()), f"expected 7 per platform: {platform_counts}")

    split_counts = Counter(item.get("split") for item in scenarios)
    require(split_counts == Counter({"development": 28, "held_out": 7}), f"unexpected split: {split_counts}")

    severity_counts = Counter(item.get("severity_class") for item in scenarios)
    require(severity_counts == Counter({1: 10, 3: 15, 5: 10}), f"unexpected severity balance: {severity_counts}")

    held_out_platforms = Counter(item["platform"] for item in scenarios if item["split"] == "held_out")
    require(set(held_out_platforms) == EXPECTED_PLATFORMS, "each platform needs at least one held-out scenario")

    for scenario in scenarios:
        scenario_id = scenario.get("scenario_id", "<missing-id>")
        missing = REQUIRED_SCENARIO_FIELDS - set(scenario)
        require(not missing, f"{scenario_id} missing fields: {sorted(missing)}")
        require(scenario["severity_class"] in {1, 3, 5}, f"{scenario_id} invalid severity")
        require(bool(scenario["severity_rationale"].strip()), f"{scenario_id} needs severity rationale")
        require(bool(scenario["expected_behavior_range"]), f"{scenario_id} needs expected behavior")
        require(bool(scenario["failure_conditions"]), f"{scenario_id} needs failure conditions")
        require(bool(scenario["proposed_pic_mapping"]), f"{scenario_id} needs PIC proxy mapping")
        require(bool(scenario["robustness_axes"]), f"{scenario_id} needs robustness axes")
        require(
            set(scenario["regulation_ids"]).issubset(regulation_ids),
            f"{scenario_id} references unknown regulation",
        )
        require(
            set(scenario["ground_truth_source_ids"]).issubset(source_ids),
            f"{scenario_id} references unknown source",
        )
        checks = scenario["deterministic_checks"]
        require("must_include_concepts" in checks, f"{scenario_id} missing must-include concepts")
        require("must_not_include_concepts" in checks, f"{scenario_id} missing prohibited concepts")

    print("Week 2 benchmark validation passed.")
    print(f"Platforms: {dict(sorted(platform_counts.items()))}")
    print(f"Splits: {dict(split_counts)}")
    print(f"Severity: {dict(sorted(severity_counts.items()))}")
    print(f"Regulations: {len(regulations)}; public sources: {len(sources)}")
    print(f"Rubric dimensions: {list(dimensions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

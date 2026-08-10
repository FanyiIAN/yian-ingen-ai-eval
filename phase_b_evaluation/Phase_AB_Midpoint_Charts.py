"""Generate submission-safe Phase A-B midpoint comparison figures.

The script reads only public aggregate JSON files. It never reads raw candidate
prompts, answers, Judge outputs, or private RunPod traces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DISPLAY_MODELS = {
    "google/flan-t5-base": "FLAN-T5-base",
    "mistralai/Mistral-7B-Instruct-v0.2": "Mistral-7B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "flan_t5_base": "FLAN-T5-base",
    "mistral_7b_instruct_v0_2": "Mistral-7B",
    "llama31_8b_instruct": "Llama-3.1-8B",
}

COLORS = ["#6DCBF4", "#3D8DFF", "#9AA3AE"]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_axis(axis: Any) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.22, linewidth=0.8)
    axis.set_axisbelow(True)


def text_model_chart(summary: dict[str, Any], output: Path) -> None:
    order = [
        "google/flan-t5-base",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "meta-llama/Llama-3.1-8B-Instruct",
    ]
    metrics = [
        ("Task", "task"),
        ("Grounding", "grounding"),
        ("Paired quality", "quality"),
    ]
    x = np.arange(len(metrics))
    width = 0.23
    fig, axis = plt.subplots(figsize=(9.4, 5.2), constrained_layout=True)
    for index, model_id in enumerate(order):
        overall = summary["models"][model_id]["overall"]
        values = [overall[key]["severity_weighted_mean"] for _, key in metrics]
        bars = axis.bar(
            x + (index - 1) * width,
            values,
            width,
            color=COLORS[index],
            label=DISPLAY_MODELS[model_id],
        )
        axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    axis.set_xticks(x, [name for name, _ in metrics])
    axis.set_ylim(0, 5.35)
    axis.set_ylabel("Diagnostic severity-weighted score (0-5)")
    axis.set_title("Frozen 35-scenario text-model comparison")
    axis.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.24))
    prepare_axis(axis)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def rag_chart(summary: dict[str, Any], output: Path) -> None:
    order = ["flan_t5_base", "mistral_7b_instruct_v0_2", "llama31_8b_instruct"]
    x = np.arange(len(order))
    base = [summary["models"][key]["ragas_provisional"]["base"]["answer_relevance"]["mean"] for key in order]
    rag = [summary["models"][key]["ragas_provisional"]["rag"]["answer_relevance"]["mean"] for key in order]
    width = 0.34
    fig, axis = plt.subplots(figsize=(9.4, 5.2), constrained_layout=True)
    bars_base = axis.bar(x - width / 2, base, width, label="Base", color="#B8BCC4")
    bars_rag = axis.bar(x + width / 2, rag, width, label="RAG", color="#3D8DFF")
    axis.bar_label(bars_base, fmt="%.3f", padding=3, fontsize=9)
    axis.bar_label(bars_rag, fmt="%.3f", padding=3, fontsize=9)
    axis.set_xticks(x, [DISPLAY_MODELS[key] for key in order])
    axis.set_ylim(0, 0.78)
    axis.set_ylabel("Local RAGAS answer relevance (0-1)")
    axis.set_title("RAG changes answer relevance on 40 governed questions")
    axis.legend(frameon=False)
    prepare_axis(axis)
    fig.text(
        0.5,
        -0.02,
        "Automatic Judge diagnostic only; not a usability percentage.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def robustness_chart(summary: dict[str, Any], output: Path) -> None:
    records = {row["candidate_model_key"]: row for row in summary["models"]}
    order = ["flan_t5_base", "mistral_7b_instruct_v0_2", "llama31_8b_instruct"]
    labels = [DISPLAY_MODELS[key] for key in order]
    consistency = [records[key]["semantic_robustness"]["semantic_robustness_score"] for key in order]
    stable_fail = [records[key]["semantic_robustness"]["stable_fail_scenario_count"] for key in order]
    mask_drop = [records[key]["masked_input"]["curves"][-1]["task_accuracy_degradation_from_complete"] for key in order]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), constrained_layout=True)
    bars = axes[0].bar(labels, consistency, color=COLORS)
    axes[0].bar_label(bars, fmt="%.3f", padding=3)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Decision consistency")
    axes[0].set_title("Semantic consistency")
    prepare_axis(axes[0])
    for index, failures in enumerate(stable_fail):
        axes[0].text(index, 0.05, f"{failures} stable fails", ha="center", fontsize=9)

    bars = axes[1].bar(labels, mask_drop, color=COLORS)
    axes[1].bar_label(bars, fmt="%.3f", padding=3)
    axes[1].set_ylim(0, max(0.45, max(mask_drop) + 0.08))
    axes[1].set_ylabel("Mean Task score drop")
    axes[1].set_title("60% evidence-group removal")
    prepare_axis(axes[1])
    fig.suptitle("Robustness needs failure-pattern context, not one percentage")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument(
        "--three-model-summary",
        type=Path,
        default=root / "W03_Three_Model_Diagnostic_Summary.json",
    )
    parser.add_argument(
        "--ragas-summary",
        type=Path,
        default=root / "W03_RAG_Expanded_Three_Model_RAGAS_Summary_v0.6.2.json",
    )
    parser.add_argument(
        "--robustness-summary",
        type=Path,
        default=root / "W04_Robustness_Summary_v0.1.0.json",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "phase_ab_figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    text_model_chart(
        read_json(args.three_model_summary),
        args.output_dir / "Phase_AB_Text_Model_Comparison.png",
    )
    rag_chart(
        read_json(args.ragas_summary),
        args.output_dir / "Phase_AB_RAG_Comparison.png",
    )
    robustness_chart(
        read_json(args.robustness_summary),
        args.output_dir / "Phase_AB_Robustness_Comparison.png",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "figure_count": 3}, indent=2))


if __name__ == "__main__":
    main()

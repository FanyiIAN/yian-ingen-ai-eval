# Week 2 Baseline Pipeline Result

## Execution contract

- Frozen responses: `70` (35 scenarios × 2 models)
- Models: `google/flan-t5-base, mistralai/Mistral-7B-Instruct-v0.2`
- Shared candidate prompt version: `0.4.0`
- Rendered semantic prompt equality: `PASS` for every scenario/model pair
- Candidate / Judge seed: `42` / `42`; greedy decoding
- Candidate errors / input truncations: `0` / `0`
- Judge score status: `diagnostic_failed_calibration`
- Frozen judged JSONL SHA-256: `3b296e1892a32ad4ac8f9327f87e9afa0caa5d920ec78670c6dcc98ef2b74f35`

> The Judge failed its provisional single-reviewer calibration. Automated ratings below are diagnostic prompt-sensitivity measurements, not validated final performance. Every row remains human-review-required.

## Primary three-Judge agreement, separated by model

### google/flan-t5-base — 35 responses

| Dimension | Krippendorff α | Exact 3-way | Within 1 point | Unresolved final |
|---|---:|---:|---:|---:|
| Task Accuracy | 0.8219 | 13/35 (37.1%) | 21/35 (60.0%) | 14 |
| Contextual Grounding | 0.7198 | 15/35 (42.9%) | 23/35 (65.7%) | 12 |
| Failure Mode | 0.6265 | 15/35 (42.9%) | n/a | 14 |

FLAN's repeated one-shot templates inflate agreement; these values do not demonstrate correct scoring.

### mistralai/Mistral-7B-Instruct-v0.2 — 35 responses

| Dimension | Krippendorff α | Exact 3-way | Within 1 point | Unresolved final |
|---|---:|---:|---:|---:|
| Task Accuracy | 0.7243 | 20/35 (57.1%) | 34/35 (97.1%) | 1 |
| Contextual Grounding | 0.6898 | 24/35 (68.6%) | 34/35 (97.1%) | 1 |
| Failure Mode | 0.4412 | 19/35 (54.3%) | n/a | 3 |

## Contract-required all-response pipeline diagnostic

The plan also requires agreement across all evaluated responses. Across the 70 rows, Task alpha is `0.8772`, Grounding alpha is `0.7806`, and Failure alpha is `0.5673`. This is retained only as a pipeline-level prompt-sensitivity diagnostic; it is not a combined model-performance score. Full counts are in `W02_Baseline_Agreement.json`.

## Per-model aggregate views

`W02_Build_Per_Model_Views.py` generates separate 35-row CSVs and per-model per-platform, severity-weighted, split, and failure-mode tables. All aggregate values are marked `diagnostic_failed_judge_calibration` and show resolved-score coverage.

## CSV field contract

The CSV retains model name/version, scenario ID, complete candidate prompt, raw response, seed, severity, all three independent formulation names, every raw and mapped score, Judge comment, exact raw Judge output and hashes, consensus fields, failure mode, robustness signal, and review status.

# Week 2 Per-Model Diagnostic Aggregates

> These tables never combine FLAN and Mistral model-performance scores. The Judge failed calibration, so all values are reference-only. Means use resolved conservative consensus rows and show coverage to expose selection bias.

- Source CSV SHA-256: `906f567d3a032b22fd40606021eabd30ec5e3e4b317b659fa8f61ce55e96500d`
- Benchmark version: `0.2.0`
- Original scenario split: `28 development / 7 held_out`; the seven held-out scenarios were later inspected and are not a fresh blind test set.

## google/flan-t5-base

- Revision: `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2`
- Rows: `35`

### Per-platform diagnostic means

| Platform | Task mean (resolved/N) | Grounding mean (resolved/N) | Quality mean (resolved/N) |
|---|---:|---:|---:|
| Aido_Humanoid | 3.500 (6/7) | 3.000 (5/7) | 3.625 (4/7) |
| Aido_Rover | 4.333 (6/7) | 5.000 (5/7) | 5.000 (5/7) |
| Fari | 1.000 (2/7) | 1.667 (6/7) | 1.000 (1/7) |
| Senpai | 1.000 (2/7) | 1.000 (3/7) | 1.000 (1/7) |
| Sentinel_Prime_AI | 1.800 (5/7) | 4.250 (4/7) | 3.250 (2/7) |

### Severity-weighted diagnostic aggregate

| Dimension | Value | Resolved/N | Severity-weight denominator |
|---|---:|---:|---:|
| task_accuracy | 3.127 | 21/35 | 71 |
| contextual_grounding | 3.377 | 23/35 | 77 |
| paired_quality | 3.725 | 13/35 | 51 |

### Failure-mode distribution

| Failure mode | Count | Share of 35 |
|---|---:|---:|
| none | 8 | 22.9% |
| partial | 3 | 8.6% |
| refusal | 4 | 11.4% |
| unresolved | 14 | 40.0% |
| unsafe | 6 | 17.1% |

### Original split diagnostic

| Split | N | Task mean (resolved/N) | Grounding mean (resolved/N) |
|---|---:|---:|---:|
| development | 28 | 3.000 (17/28) | 3.176 (17/28) |
| held_out | 7 | 2.250 (4/7) | 2.667 (6/7) |

## mistralai/Mistral-7B-Instruct-v0.2

- Revision: `63a8b081895390a26e140280378bc85ec8bce07a`
- Rows: `35`

### Per-platform diagnostic means

| Platform | Task mean (resolved/N) | Grounding mean (resolved/N) | Quality mean (resolved/N) |
|---|---:|---:|---:|
| Aido_Humanoid | 4.333 (6/7) | 4.429 (7/7) | 4.333 (6/7) |
| Aido_Rover | 4.286 (7/7) | 4.571 (7/7) | 4.429 (7/7) |
| Fari | 4.000 (7/7) | 4.143 (7/7) | 4.071 (7/7) |
| Senpai | 3.857 (7/7) | 4.500 (6/7) | 4.333 (6/7) |
| Sentinel_Prime_AI | 3.857 (7/7) | 4.429 (7/7) | 4.143 (7/7) |

### Severity-weighted diagnostic aggregate

| Dimension | Value | Resolved/N | Severity-weight denominator |
|---|---:|---:|---:|
| task_accuracy | 4.080 | 34/35 | 100 |
| contextual_grounding | 4.500 | 34/35 | 104 |
| paired_quality | 4.288 | 33/35 | 99 |

### Failure-mode distribution

| Failure mode | Count | Share of 35 |
|---|---:|---:|
| hallucination | 1 | 2.9% |
| none | 19 | 54.3% |
| off_policy | 3 | 8.6% |
| refusal | 9 | 25.7% |
| unresolved | 3 | 8.6% |

### Original split diagnostic

| Split | N | Task mean (resolved/N) | Grounding mean (resolved/N) |
|---|---:|---:|---:|
| development | 28 | 4.214 (28/28) | 4.429 (28/28) |
| held_out | 7 | 3.333 (6/7) | 4.333 (6/7) |

## Interpretation limits

- Do not compare the model means as validated leaderboard scores.
- FLAN has extensive unresolved consensus and one-shot copying; its resolved subset is highly selected and can produce misleadingly high platform means.
- Prometheus frequently misclassified safe boundaries as `refusal`; failure-mode counts are Judge outputs, not adjudicated truth.
- A new untouched test set and independently reviewed human gold are required before model-selection claims.

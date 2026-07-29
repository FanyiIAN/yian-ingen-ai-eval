# Week 3 Three-Model Diagnostic Comparison

> The Prometheus Judge failed its frozen calibration. Scores below are
> diagnostic, coverage-sensitive evidence and are not a validated model
> quality leaderboard. Human review remains required for all 105 rows and
> every severity-5 response.

- Benchmark: `0.2.0` (35 synthetic scenarios)
- Candidate prompt: `0.4.0` / `0bb0a6f2e298f286739080752540939454e2e5e52c0dca477e17196657cac71d`
- Seed: `42`; deterministic decoding; no input truncation
- Original split: 28 development / 7 formerly held-out; the seven were
  inspected during Week 2 and are no longer blind test evidence.

## Overall diagnostic comparison

| Model | SW Task (resolved/N) | SW Grounding (resolved/N) | SW Quality (resolved/N) | Output tok/s | Mean latency ms |
|---|---:|---:|---:|---:|---:|
| `google/flan-t5-base` | 3.127 (21/35) | 3.377 (23/35) | 3.725 (13/35) | 49.602 | 492.491 |
| `meta-llama/Llama-3.1-8B-Instruct` | 3.970 (33/35) | 4.050 (33/35) | 4.061 (32/35) | 27.872 | 1819.536 |
| `mistralai/Mistral-7B-Instruct-v0.2` | 4.080 (34/35) | 4.500 (34/35) | 4.288 (33/35) | 28.829 | 2542.040 |

SW means severity-weighted. A higher mean with much lower resolved
coverage can reflect selection bias, so coverage is part of every cell.

## Llama diagnostic deltas

| Baseline | Δ SW Task | Δ SW Grounding | Δ SW Quality | Δ tok/s |
|---|---:|---:|---:|---:|
| `google/flan-t5-base` | 0.843 | 0.673 | 0.336 | -21.730 |
| `mistralai/Mistral-7B-Instruct-v0.2` | -0.110 | -0.450 | -0.227 | -0.957 |

These deltas answer only what the failed-calibration diagnostic Judge
reported under the frozen protocol. They do not establish that Llama is
better until the rows receive model-blind human adjudication.

## Per-platform diagnostic quality

| Model | Fari | Senpai | Sentinel | Rover | Humanoid |
|---|---:|---:|---:|---:|---:|
| `google/flan-t5-base` | 1.000 (1/7) | 1.000 (1/7) | 3.312 (2/7) | 5.000 (5/7) | 4.179 (4/7) |
| `meta-llama/Llama-3.1-8B-Instruct` | 3.435 (7/7) | 4.083 (4/7) | 3.833 (7/7) | 4.548 (7/7) | 4.476 (7/7) |
| `mistralai/Mistral-7B-Instruct-v0.2` | 4.196 (7/7) | 4.333 (6/7) | 4.095 (7/7) | 4.381 (7/7) | 4.500 (6/7) |

## Overall failure distribution and severity-5 flags

| Model | Unsafe | Hallucination | Off-policy | Refusal | Partial | None | Unresolved | Sev-5 task <=2 | Sev-5 unsafe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `google/flan-t5-base` | 6 | 0 | 0 | 4 | 3 | 8 | 14 | 2/7 | 2/10 |
| `meta-llama/Llama-3.1-8B-Instruct` | 1 | 0 | 0 | 4 | 1 | 21 | 8 | 1/10 | 1/10 |
| `mistralai/Mistral-7B-Instruct-v0.2` | 0 | 1 | 3 | 9 | 0 | 19 | 3 | 0/9 | 0/10 |

## Required interpretation

- Observation: report the score, coverage, latency, and failure counts.
- Mechanism: inspect row-level evidence before explaining why a model
  behaved differently; architecture size alone is not an explanation.
- Deployment boundary: these are independent open models on synthetic
  text scenarios, not deployed InGen products or PIC runtime results.
- Next validity gate: model-blind human adjudication, including all
  severity-5 rows, followed by a freshly sealed held-out set.

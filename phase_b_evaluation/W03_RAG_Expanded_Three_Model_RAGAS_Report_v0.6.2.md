# Week 3 Three-Model RAG Run Audit

- Analyzer version: `0.2.0`
- Excluded preflight IDs: `none`
- Shared-input audit: `PASS`

## Uninspected aggregate

| Model | Condition | Rows | Empty | Echo | Mean output tokens | Mean / p95 latency ms | tok/s | GPU peak GiB | Citation precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flan_t5_base | base | 40 | 0 | 7 | 8.45 | 221.24405 / 351.15225 | 43.019953 | 4.820 | n/a |
| flan_t5_base | rag | 40 | 0 | 2 | 39.675 | 862.40895 / 4903.4974 | 35.55769 | 4.820 | n/a |
| mistral_7b_instruct_v0_2 | base | 40 | 0 | 0 | 159.875 | 5141.0645 / 10538.9948 | 30.87959 | 15.234 | n/a |
| mistral_7b_instruct_v0_2 | rag | 40 | 0 | 0 | 189.65 | 6673.326075 / 13178.9717 | 27.907627 | 15.234 | 1.000 |
| llama31_8b_instruct | base | 40 | 0 | 1 | 90.675 | 3048.359275 / 9337.99175 | 29.313921 | 16.070 | n/a |
| llama31_8b_instruct | rag | 40 | 0 | 0 | 220.4 | 7839.686425 / 13450.9604 | 27.059894 | 16.070 | 1.000 |

## Automatic RAGAS diagnostics

| Model | Base relevance | RAG relevance | Delta | RAG faithfulness | Context relevance | Context recall | Context precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| flan_t5_base | 0.422540 (36/40) | 0.448278 (37/40) | +0.025738 | 0.847222 (36/40) | 0.981250 (40/40) | 0.835833 (40/40) | 0.841127 (40/40) |
| mistral_7b_instruct_v0_2 | 0.222679 (36/40) | 0.635032 (35/40) | +0.412353 | 0.877956 (40/40) | 0.981250 (40/40) | 0.835833 (40/40) | 0.841127 (40/40) |
| llama31_8b_instruct | 0.263927 (38/40) | 0.642118 (39/40) | +0.378191 | 0.876272 (40/40) | 0.981250 (40/40) | 0.835833 (40/40) | 0.841127 (40/40) |

Automatic RAGAS values are diagnostic because the local Judge is uncalibrated. Mistral candidate rows use a non-independent Mistral self-judge and must not be used for a winner claim. A separate AI qualitative review is required for answer-content claims.

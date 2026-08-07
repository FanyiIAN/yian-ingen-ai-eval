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

Automatic RAGAS values are diagnostic because the local Judge is uncalibrated. Mistral candidate rows use a non-independent Mistral self-judge and must not be used for a winner claim. A separate AI qualitative calibration reviews answer content.

# Week 4 System Performance Summary

> Hardware/configuration-specific evidence. A missing measurement is not zero. 
> RAG latency is not applicable when retrieval is disabled.

## Static run and model-load evidence

| Model | Revision | Precision | Checkpoint GiB | Load (s) | Load GPU peak (MiB) | GPU | Host RAM (MiB) |
|---|---|---|---:|---:|---:|---|---:|
| flan_t5_base | `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2` | float32 | 0.93 | 44.52 | 1355.0 | NVIDIA A40 | 515598.5 |
| mistral_7b_instruct_v0_2 | `63a8b081895390a26e140280378bc85ec8bce07a` | bfloat16 | 13.49 | 105.13 | 14125.0 | NVIDIA A40 | 515598.5 |
| llama31_8b_instruct | `0e9e39f249a16976918f6564b8830bc894c89659` | bfloat16 | 29.93 | 114.50 | 15629.0 | NVIDIA A40 | 515598.5 |
| idefics2_8b_chatty | `8e65868b394317b973bd61db3b08e6478ebeedbf` | bfloat16 | 31.31 | 12.15 | 16339.0 | NVIDIA A40 | 515598.5 |
| llama31_8b_instruct | `0e9e39f249a16976918f6564b8830bc894c89659` | bfloat16 | 29.93 | 67.74 | 20059.0 | NVIDIA A40 | 515598.5 |

## Per-request warm-path evidence

| Model | Family | Condition | n | Prompt / output tokens p50 | End-to-end p50 / p95 (ms) | TTFT p50 / p95 (ms) | Generation p50 / p95 (ms) | GPU memory peak max (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| flan_t5_base | masked_input_robustness |  | 42 | 408 / 36 | 733.3 / 906.4 | 71.0 / 92.4 | 730.6 / 902.6 | 1393.0 |
| flan_t5_base | semantic_robustness |  | 140 | 406 / 31 | 647.1 / 887.3 | 88.7 / 103.3 | 642.8 / 882.5 | 1393.0 |
| idefics2_8b_chatty | multimodal_robustness | brightness_0.60 | 20 | 432 / 162 | 6112.3 / 7139.5 | 788.2 / 881.9 | 5932.7 / 7055.2 | 18407.0 |
| idefics2_8b_chatty | multimodal_robustness | clean | 20 | 432 / 185 | 6717.2 / 6949.2 | 757.4 / 860.8 | 6564.7 / 6854.0 | 18670.0 |
| idefics2_8b_chatty | multimodal_robustness | gaussian_noise_std_0.08 | 20 | 432 / 170 | 6172.4 / 7309.0 | 767.2 / 858.9 | 5997.1 / 7157.7 | 18407.0 |
| llama31_8b_instruct | masked_input_robustness |  | 42 | 381 / 48 | 1759.1 / 2276.0 | 95.9 / 109.0 | 1756.8 / 2271.8 | 15881.0 |
| llama31_8b_instruct | rag_performance | base | 40 | 244 / 47 | 1594.0 / 9450.6 | 69.2 / 92.4 | 1591.6 / 9446.7 | 21299.0 |
| llama31_8b_instruct | rag_performance | rag | 40 | 1866 / 220 | 8402.7 / 14230.4 | 733.6 / 930.1 | 8038.4 / 13796.7 | 21299.0 |
| llama31_8b_instruct | semantic_robustness |  | 140 | 388 / 55 | 2023.0 / 2992.5 | 106.7 / 116.3 | 2019.3 / 2987.5 | 15881.0 |
| mistral_7b_instruct_v0_2 | masked_input_robustness |  | 42 | 421 / 50 | 1727.6 / 6211.3 | 103.6 / 114.1 | 1724.5 / 6209.0 | 14634.0 |
| mistral_7b_instruct_v0_2 | semantic_robustness |  | 140 | 424 / 62 | 2135.8 / 4565.3 | 98.2 / 112.9 | 2132.7 / 4561.6 | 14371.0 |

The complete JSON/CSV retain prompt/output/total token counts, p50, p90, p95, maximum, mean, standard deviation, missing counts, and component-level timing.

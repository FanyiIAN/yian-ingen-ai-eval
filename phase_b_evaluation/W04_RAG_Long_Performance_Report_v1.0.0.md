# Week 4 Long-Document RAG Performance Report

> **Status: latest public RAG performance result (v1.0.0).** This corrective run uses the complete long-source knowledge base and the 40-question long-document benchmark. Week 4 image, masked-input, robustness, and non-RAG model results were independent of the knowledge-base correction and were not rerun.

**Run design:** 40 frozen questions × base/RAG = 80 warm-path requests, plus one excluded RAG warm-up.  
**Hardware:** one NVIDIA A40; batch size 1; deterministic Llama 3.1 8B generation.  
**Interpretation:** latency values are component/run specific, not deployed-product SLAs.

> Hardware/configuration-specific evidence. A missing measurement is not zero. 
> RAG latency is not applicable when retrieval is disabled.

## Static run and model-load evidence

| Model | Revision | Precision | Checkpoint GiB | Load (s) | Load GPU peak (MiB) | GPU | Host RAM (MiB) |
|---|---|---|---:|---:|---:|---|---:|
| llama31_8b_instruct | `0e9e39f249a16976918f6564b8830bc894c89659` | bfloat16 | 29.93 | 47.39 | 20063.0 | NVIDIA A40 | 515598.4 |

## Per-request warm-path evidence

| Model | Family | Condition | n | Prompt / output tokens p50 | End-to-end p50 / p95 (ms) | TTFT p50 / p95 (ms) | Generation p50 / p95 (ms) | GPU memory peak max (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| llama31_8b_instruct | rag_performance | base | 40 | 112 / 11 | 385.7 / 1390.5 | 49.6 / 66.1 | 383.9 / 1389.0 | 22419.0 |
| llama31_8b_instruct | rag_performance | rag | 40 | 3789 / 186 | 8286.0 / 17896.9 | 1652.4 / 1701.0 | 7263.0 / 16913.2 | 22419.0 |

The complete JSON/CSV retain prompt/output/total token counts, p50, p90, p95, maximum, mean, standard deviation, missing counts, and component-level timing.

## Result interpretation

The RAG condition had an 8,286.0 ms median end-to-end latency versus 385.7 ms for base (about 21.5×), while its median prompt was 3,789 versus 112 tokens and its median answer was 186 versus 11 tokens. This is a full-condition comparison: retrieval, context assembly, longer prefill, and much longer answers all contribute, so the difference must not be described as retrieval latency alone. The RAG p95 was 17,896.9 ms. No quality conclusion is drawn here; Week 3 and Week 5 score answer quality separately.

Cold-start model/index loading and the excluded warm-up are retained in the private manifest. All 80 official rows are marked `warm_steady_state`; the report does not mix cold-start and warm-path latency.

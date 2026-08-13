# Week 5 Long-Source Senpai RAG Optimisation

> **Status: latest corrective public RAG optimisation result (v1.1.0).** The experiment uses complete long documents, so chunk size changes the actual indexed retrieval units. Automated Judge metrics remain diagnostic.

**Design:** 3 chunk sizes × 3 top-k values × reranking off/on
**Items:** 20 frozen long-document Senpai questions per configuration
**Corpus:** 21 complete official public sources with status-aware metadata
**Candidate:** `meta-llama/Llama-3.1-8B-Instruct` revision `0e9e39f249a16976918f6564b8830bc894c89659`
**Seed:** `42`; deterministic decoding; randomized variant-block order
**Timing:** one discarded warm-up; cold model load recorded separately; Pareto latency uses warm-path request time
**Judge runtime:** the frozen local Mistral checkpoint scored the first 112 rows behind an 8,192-token vLLM service. A pre-output top-k=5 batch exposed a 6,145-input + 2,048-output capacity overflow; no rows from that failed batch were retained. The same checkpoint, decoding and metric definitions resumed behind a 16,384-token loopback service for the remaining rows.

## Reading the experiment

A **factor** is an input deliberately changed by the experiment: chunk size, top-k, or reranking. A **full factorial** design tests every combination of their levels (3 × 3 × 2 = 18 cells). A **matched contrast** compares two cells that differ in only one factor—for example, reranking off versus on while chunk size and top-k stay fixed. An **interaction** means one factor's effect depends on another, such as reranking helping at top-k 5 but not top-k 1. A **confounder** is an uncontrolled difference that could offer an alternative explanation; frozen questions/models and randomized variant-block order reduce, but do not eliminate, such risks.

**Cold start** includes model/index loading; **warm steady-state** measures requests after one excluded warm-up. The two are reported separately. A **Pareto-optimal** cell is not beaten by another tested cell on all three objectives: higher diagnostic Faithfulness, higher required-point Coverage, and lower warm-path latency.

## Pareto result

The non-dominated set maximizes diagnostic Faithfulness and weighted required-point Coverage while minimizing warm-path p50 question-to-response latency. Only cells with complete Faithfulness, Coverage, and latency are eligible for the primary frontier; incomplete cells remain in the full table. Relevance is reported as a supporting metric, not an optimization axis.

| Variant | Faithfulness | Coverage | Relevance | p50 ms |
|---|---:|---:|---:|---:|
| `chunk-1024_topk-3_rerank-ce` | 0.8361 | 0.9750 | 0.6410 | 8679.5 |
| `chunk-1024_topk-5_rerank-ce` | 0.9095 | 0.9750 | 0.6632 | 10862.9 |
| `chunk-512_topk-5_rerank-ce` | 0.9020 | 0.9667 | 0.7330 | 6576.2 |

The transparent balanced choice within the frontier is `chunk-1024_topk-5_rerank-ce`. This is a diagnostic configuration recommendation, not a production-readiness claim.

## All factorial cells

| Variant | Faithfulness | F coverage | Coverage | C coverage | Relevance | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| `chunk-1024_topk-1_rerank-ce` | 0.7915 | 0.950 | 0.9000 | 1.000 | 0.5452 | 8738.5 | 13826.8 |
| `chunk-1024_topk-1_rerank-none` | 0.7128 | 0.800 | 0.7000 | 1.000 | 0.3577 | 7670.5 | 15728.1 |
| `chunk-1024_topk-3_rerank-ce` | 0.8361 | 1.000 | 0.9750 | 1.000 | 0.6410 | 8679.5 | 20491.3 |
| `chunk-1024_topk-3_rerank-none` | 0.7981 | 0.950 | 0.9000 | 1.000 | 0.4910 | 8822.0 | 17450.1 |
| `chunk-1024_topk-5_rerank-ce` | 0.9095 | 1.000 | 0.9750 | 1.000 | 0.6632 | 10862.9 | 21596.5 |
| `chunk-1024_topk-5_rerank-none` | 0.8465 | 1.000 | 0.9125 | 1.000 | 0.5689 | 7526.4 | 16138.9 |
| `chunk-256_topk-1_rerank-ce` | 0.8358 | 1.000 | 0.8958 | 0.900 | 0.5758 | 6985.0 | 10963.9 |
| `chunk-256_topk-1_rerank-none` | 0.7408 | 1.000 | 0.8092 | 0.950 | 0.4885 | 7951.5 | 14346.8 |
| `chunk-256_topk-3_rerank-ce` | 0.8658 | 0.950 | 0.9500 | 1.000 | 0.5996 | 7617.1 | 13904.6 |
| `chunk-256_topk-3_rerank-none` | 0.8775 | 1.000 | 0.9313 | 1.000 | 0.7134 | 7410.5 | 13145.4 |
| `chunk-256_topk-5_rerank-ce` | 0.8106 | 1.000 | 0.9474 | 0.950 | 0.7136 | 7511.8 | 12872.0 |
| `chunk-256_topk-5_rerank-none` | 0.8719 | 1.000 | 0.9500 | 1.000 | 0.6229 | 7325.0 | 17815.8 |
| `chunk-512_topk-1_rerank-ce` | 0.8320 | 1.000 | 0.8604 | 1.000 | 0.5447 | 7081.1 | 12302.4 |
| `chunk-512_topk-1_rerank-none` | 0.8181 | 0.850 | 0.7583 | 1.000 | 0.4410 | 6273.8 | 15958.7 |
| `chunk-512_topk-3_rerank-ce` | 0.8382 | 1.000 | 0.9125 | 1.000 | 0.6558 | 7678.3 | 15256.5 |
| `chunk-512_topk-3_rerank-none` | 0.8500 | 0.950 | 0.9125 | 1.000 | 0.6915 | 7202.9 | 12087.7 |
| `chunk-512_topk-5_rerank-ce` | 0.9020 | 1.000 | 0.9667 | 1.000 | 0.7330 | 6576.2 | 16419.4 |
| `chunk-512_topk-5_rerank-none` | 0.8083 | 1.000 | 0.9250 | 1.000 | 0.6079 | 6811.5 | 16562.4 |

## Controlled comparisons

All `45` registered matched contrasts differ in exactly one factor: `18` chunk-size, `18` top-k, and `9` reranking contrasts. Across `120` top-k/reranking/item groups, candidate messages were identical across chunk levels at rate `0.000` and outputs at rate `0.000`. Zero message and output identity across chunk levels confirms that complete-document chunking made chunk size an operational factor in this corrective experiment.

Positive quality deltas favor the right-hand factor level; positive latency deltas mean it is slower. Factor-level averages are descriptive summaries of the registered matched contrasts:

| Factor | n | Δ Faith | Δ Relevance | Δ Coverage | Δ latency ms |
|---|---:|---:|---:|---:|---:|
| chunk_size_tokens | 18 | -0.0072 | -0.0497 | -0.0137 | 1199.7 |
| reranking | 9 | 0.0263 | 0.0766 | 0.0664 | 772.9 |
| top_k | 18 | 0.0623 | 0.1063 | 0.0831 | 606.7 |

The reranking effect is conditional on top-k:

| Fixed top-k | n | Δ Faith | Δ Relevance | Δ Coverage | Δ latency ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 0.0634 | 0.1262 | 0.1331 | 141.7 |
| 3 | 3 | -0.0163 | 0.0001 | 0.0312 | 1069.0 |
| 5 | 3 | 0.0318 | 0.1034 | 0.0347 | 1107.8 |

## Completeness and scoring audit

| Metric | Finite fraction |
|---|---:|
| faithfulness | 0.969 |
| answer_relevance | 1.000 |
| required_point_coverage | 0.989 |
| latency | 1.000 |

RAGAS statuses: `{'complete': 349, 'metric_failure_retained': 11}`; coverage statuses: `{'parse_failed': 4, 'parsed': 348, 'parsed_after_deterministic_repair': 8}`. Missing metrics are excluded from the relevant mean and their finite fraction is reported; no diagnostic value is imputed.
Rows marked `parsed_after_deterministic_repair` contained every registered rubric point plus extra, unregistered point IDs. The repair discarded only those extras and retained an audit trail; it never created a missing registered-point score or altered a registered Judge verdict.

## Reliability boundary

Public long-source component evaluation on one frozen Llama stack. Current pages are design intent; dated PDFs are background. The local evaluator is diagnostic and no result establishes deployed-product or PIC readiness.

# Week 5 Senpai RAG Optimisation

> **Status: SUPERSEDED / NOT LATEST.** This optimisation used the earlier atomic-section knowledge base, which did not make chunk size an operational long-document factor. Use `W05_RAG_Long_Source_Optimisation_Report_v1.1.0.md` for the corrective full-factorial result.

**Design:** 3 chunk sizes × 3 top-k values × reranking off/on
**Items:** 20 frozen Senpai questions per configuration
**Candidate:** `meta-llama/Llama-3.1-8B-Instruct` revision `0e9e39f249a16976918f6564b8830bc894c89659`
**Seed:** `42`; deterministic decoding; randomized variant-block order
**Timing:** one discarded warm-up; cold model load recorded separately; Pareto latency uses warm-path request time

## Pareto result

The non-dominated set maximizes diagnostic Faithfulness and weighted required-point Coverage while minimizing warm-path p50 question-to-response latency. Relevance is reported as a supporting metric, not an optimization axis.

| Variant | Faithfulness | Coverage | Relevance | p50 ms |
|---|---:|---:|---:|---:|
| `chunk-1024_topk-1_rerank-ce` | 0.8690 | 0.7668 | 0.4380 | 3050.3 |
| `chunk-256_topk-5_rerank-none` | 0.9075 | 0.9325 | 0.6316 | 7145.6 |
| `chunk-512_topk-1_rerank-none` | 0.8493 | 0.7668 | 0.3705 | 2781.2 |
| `chunk-512_topk-3_rerank-ce` | 0.9337 | 0.8865 | 0.6392 | 5579.3 |
| `chunk-512_topk-5_rerank-ce` | 0.8494 | 0.9692 | 0.6060 | 7794.9 |

The transparent balanced choice within the frontier is `chunk-256_topk-5_rerank-none`. This is a diagnostic configuration recommendation, not a production-readiness claim.

## All factorial cells

| Variant | Faithfulness | F coverage | Coverage | Relevance | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| `chunk-1024_topk-1_rerank-ce` | 0.8690 | 0.900 | 0.7668 | 0.4380 | 3050.3 | 7122.0 |
| `chunk-1024_topk-1_rerank-none` | 0.8493 | 0.900 | 0.7668 | 0.3705 | 2790.4 | 6735.0 |
| `chunk-1024_topk-3_rerank-ce` | 0.9337 | 0.950 | 0.8865 | 0.6392 | 5597.7 | 9206.8 |
| `chunk-1024_topk-3_rerank-none` | 0.8851 | 0.950 | 0.8492 | 0.5701 | 5657.4 | 8846.7 |
| `chunk-1024_topk-5_rerank-ce` | 0.8494 | 1.000 | 0.9692 | 0.6060 | 7809.4 | 11915.5 |
| `chunk-1024_topk-5_rerank-none` | 0.9075 | 1.000 | 0.9325 | 0.6316 | 7158.4 | 11521.8 |
| `chunk-256_topk-1_rerank-ce` | 0.8690 | 0.900 | 0.7668 | 0.4380 | 3105.1 | 7196.0 |
| `chunk-256_topk-1_rerank-none` | 0.8493 | 0.900 | 0.7668 | 0.3705 | 2881.0 | 6800.9 |
| `chunk-256_topk-3_rerank-ce` | 0.9337 | 0.950 | 0.8865 | 0.6392 | 5650.9 | 9161.1 |
| `chunk-256_topk-3_rerank-none` | 0.8851 | 0.950 | 0.8492 | 0.5701 | 5644.4 | 8784.9 |
| `chunk-256_topk-5_rerank-ce` | 0.8494 | 1.000 | 0.9692 | 0.6060 | 7804.9 | 11934.1 |
| `chunk-256_topk-5_rerank-none` | 0.9075 | 1.000 | 0.9325 | 0.6316 | 7145.6 | 11518.6 |
| `chunk-512_topk-1_rerank-ce` | 0.8690 | 0.900 | 0.7668 | 0.4380 | 3131.1 | 7254.3 |
| `chunk-512_topk-1_rerank-none` | 0.8493 | 0.900 | 0.7668 | 0.3705 | 2781.2 | 6716.8 |
| `chunk-512_topk-3_rerank-ce` | 0.9337 | 0.950 | 0.8865 | 0.6392 | 5579.3 | 9180.0 |
| `chunk-512_topk-3_rerank-none` | 0.8851 | 0.950 | 0.8492 | 0.5701 | 5648.4 | 8820.8 |
| `chunk-512_topk-5_rerank-ce` | 0.8494 | 1.000 | 0.9692 | 0.6060 | 7794.9 | 11921.6 |
| `chunk-512_topk-5_rerank-none` | 0.9075 | 1.000 | 0.9325 | 0.6316 | 7158.8 | 11613.6 |

## Controlled comparisons

All `45` registered matched contrasts differ in exactly one factor: `18` chunk-size, `18` top-k, and `9` reranking contrasts. Across `120` top-k/reranking/item groups, candidate messages were identical across chunk levels at rate `1.000` and outputs at rate `1.000`. High identity means the atomic source sections are shorter than the registered chunk sizes; it is a structural null effect, not proof that chunk size never matters on a larger corpus.

Positive quality deltas favor the right-hand factor level; positive latency deltas mean it is slower. Factor-level averages are descriptive summaries of the registered matched contrasts:

| Factor | n | Δ Faith | Δ Relevance | Δ Coverage | Δ latency ms |
|---|---:|---:|---:|---:|---:|
| chunk_size_tokens | 18 | 0.0000 | 0.0000 | 0.0000 | -17.0 |
| reranking | 9 | 0.0034 | 0.0370 | 0.0247 | 482.4 |
| top_k | 18 | 0.0064 | 0.1430 | 0.1227 | 2298.3 |

The reranking effect is conditional on top-k:

| Fixed top-k | n | Δ Faith | Δ Relevance | Δ Coverage | Δ latency ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 0.0198 | 0.0675 | 0.0000 | 439.5 |
| 3 | 3 | 0.0486 | 0.0690 | 0.0374 | -21.6 |
| 5 | 3 | -0.0581 | -0.0256 | 0.0367 | 1029.5 |

## Completeness and scoring audit

| Metric | Finite fraction |
|---|---:|
| faithfulness | 0.950 |
| answer_relevance | 1.000 |
| required_point_coverage | 1.000 |
| latency | 1.000 |

RAGAS statuses: `{'complete': 342, 'metric_failure_retained': 18}`; coverage statuses: `{'parsed': 360}`. Missing metrics are excluded from the relevant mean and their finite fraction is reported; no diagnostic value is imputed.

## Reliability boundary

Public-source component evaluation of one frozen Llama RAG stack on 20 Senpai questions. The local Mistral evaluator is not human-calibrated, one repetition does not estimate run-to-run variance, and the study is not evidence of deployed Senpai, PIC 2.0, safety, or production readiness.

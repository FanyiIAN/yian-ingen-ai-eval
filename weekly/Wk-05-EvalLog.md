# Week 5 Evaluation Log

> **2026-08-12 latest corrective update:** The atomic-section optimisation described later in this file is retained for provenance but is superseded. The latest complete-document full factorial is `phase_c_synthesis/W05_RAG_Long_Source_Optimisation_Report_v1.1.0.md`.

## Complete-document corrective factorial

- Ran all 18 cells (256/512/1,024 tokens × top-k 1/3/5 × reranking off/cross-encoder), 20 frozen Senpai questions per cell, for 360/360 generated and traced rows.
- Zero matched groups had identical messages or outputs across all chunk sizes, confirming that chunk size was an operational factor rather than a label.
- Formal scoring produced 349/360 finite Faithfulness rows, 360/360 finite Relevance rows, and 356/360 finite Coverage rows. Eleven RAGAS `NaN` values and four genuine Coverage parse failures were retained, never imputed.
- The primary Pareto frontier admits only cells with complete Faithfulness, Coverage, and latency. Its three cells are `chunk-1024_topk-3_rerank-ce`, `chunk-1024_topk-5_rerank-ce`, and `chunk-512_topk-5_rerank-ce`.
- The transparent balanced diagnostic choice is `chunk-1024_topk-5_rerank-ce`: mean Faithfulness 0.910, Coverage 0.975, Relevance 0.663, and warm-path p50 10,862.9 ms. This is conditional on the frozen subset, A40, models, and uncalibrated Judge.
- Across 45 one-factor matched contrasts, average top-k increases improved Faithfulness +0.062, Relevance +0.106, and Coverage +0.083 at +606.7 ms. Cross-encoder reranking averaged +0.026/+0.077/+0.066 at +772.9 ms, with effects varying by top-k.
- The local Mistral Judge initially served 8,192 tokens; a pre-output top-k=5 batch exposed a 6,145-input + 2,048-output overflow. The failed batch wrote no rows. The same checkpoint and metrics resumed behind a loopback-only 16,384-token service.
- Coverage scorer v1.4 deterministically discarded only unregistered extra point IDs when every frozen registered point was present. Eight Week 5 rows were recovered with a full audit trail; four rows missing/conflicting registered points stayed failed.

**Phase:** C – PIC 2.0 analysis and RAG optimisation
**Status:** complete
**Date:** 2026-08-11
**Seed:** `42`

## Completed scope

Week 5 combined three evidence streams: six PIC-specific capability analyses;
a controlled Senpai RAG full factorial; and a stratified Week 2–4 synthesis of
platform trends, failures, severity association, and surprising cases. These
are public-source component diagnostics, not deployed-product measurements.

## Formal Senpai RAG experiment

- Ran chunk size 256/512/1024 × top-k 1/3/5 × reranking off/cross-encoder:
  18 configurations, 20 frozen questions each, and 360 traced rows.
- Used the exact registered Llama 3.1 8B, BGE-M3, BGE reranker v2 M3 and local
  Mistral evaluator revisions on one NVIDIA A40; no external model API was used.
- Randomized variant-block order with seed 42, discarded one warm-up, recorded
  cold loads separately, and used warm-path request latency for Pareto analysis.
- Generated all 360 non-empty candidate rows in one clean run. Cross-encoder
  execution was effective on all 180 requested rows.
- Parsed all 360 required-point coverage rows with scorer `1.3.0`. RAGAS
  Relevance was finite for all 360 rows; Faithfulness was finite for 342 rows.
  The remaining 18 answers had no extractable claim, so their Faithfulness is
  retained as missing rather than imputed.
- Recovered two transient missing Judge values with scorer `1.1.0`. Successful
  metric definitions did not change; scorer versions and retry audits remain in
  the private raw evidence.

## RAG result

The Pareto objectives maximize mean diagnostic Faithfulness and weighted
required-point Coverage while minimizing median warm-path latency. Five
configurations are non-dominated:

- `chunk-1024_topk-1_rerank-ce`
- `chunk-256_topk-5_rerank-none`
- `chunk-512_topk-1_rerank-none`
- `chunk-512_topk-3_rerank-ce`
- `chunk-512_topk-5_rerank-ce`

The transparent balanced choice is `chunk-256_topk-5_rerank-none`, with mean
Faithfulness `0.9075`, Coverage `0.9325`, Relevance `0.6316`, and p50 latency
`7145.6 ms`. This is conditional on the registered subset and uncalibrated
local Judge, not a production configuration claim.

All 45 matched contrasts differ in one factor: 18 chunk-size, 18 top-k and 9
reranking comparisons. Increasing top-k produced average deltas of `+0.0064`
Faithfulness, `+0.1430` Relevance and `+0.1227` Coverage, with `+2298.3 ms`
latency. Reranking interacted with top-k: it improved Faithfulness and Relevance
at top-k 1 and 3, but at top-k 5 changed them by `-0.0581` and `-0.0256` while
Coverage increased `+0.0367` and latency increased `1029.5 ms`.

Across 120 item/top-k/reranking matched groups, candidate messages and outputs
were identical across all three chunk settings. The observed chunk-size null
effect is therefore a corpus-structure result: atomic source sections were
already shorter than the registered chunk sizes. It is not evidence that chunk
size is generally irrelevant.

## Accumulated Week 2–4 evidence

- Unlike metric families were not pooled. Week 2 remains
  `diagnostic_failed_calibration`; Week 3 RAGAS and Week 4 rubric scores remain
  uncalibrated diagnostics.
- In Week 3, finite mean Base→RAG Relevance improved in all six platform/model
  cells. Senpai deltas were `+0.0530` FLAN, `+0.3862` Llama and `+0.5164`
  Mistral.
- In 105 original Week 4 semantic rows, failure codes were `none` 64,
  `partial` 39, `off_policy` 1 and `refusal` 1. Perturbed rows remain separate
  because they repeat scenarios.
- The pooled severity/failure Spearman association was `-0.301075`. It is
  descriptive, not causal. Explicit cues in high-severity safety/privacy items
  and nuanced requirements in lower-severity items are plausible mechanisms,
  not proven explanations.
- `FARI-003` exposed likely evaluator anchoring: FLAN returned only “SYSTEM
  POLICY” yet received `5/5`. `SENPAI-001` exposed Llama safety over-refusal and
  a Mistral scientific misconception that still received `4/5`.

## PIC 2.0 synthesis

- AMDC has the broadest direct proxy evidence from 120 controlled VLM rows.
- GRPO and HTD-IRL have useful text-plan diagnostics but no executed policies or
  task graphs.
- STUM lacks a temporal-horizon benchmark. SEOM has static-image ceiling
  evidence but no closed-loop localization/navigation evaluation. Their
  conflicting meanings remain separated through a versioned terminology
  registry.
- CRL-MRS has the largest direct gap: one cooperative text scenario and no
  communication-loss, agent-dropout or continual-learning experiment.

## Verification and claim boundary

The final notebook executed all five code cells without error. All 36 Week 5
unit tests passed. Public outputs contain sanitized metrics, hashes and
aggregates; raw prompts, contexts, outputs, Judge traces, logs and the complete
environment freeze remain private.

The largest PIC-specific gap is **CRL-MRS joint task success under controlled
communication loss**. More generally, Phase B supplies component proxies, not
an approved product runtime, robot simulator, synchronized sensor stream or
multi-agent execution trace. No Week 5 result establishes deployed PIC 2.0,
safety, causality outside the registered design, or production readiness.

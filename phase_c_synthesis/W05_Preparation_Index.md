# Week 5 Completion Index

**Phase:** C – methodology synthesis and PIC 2.0 analysis
**Status:** complete; long-source RAG corrective rerun incorporated
**Date:** 2026-08-12
**Seed:** `42`

## Required deliverables

### Latest corrective RAG artifacts

| Deliverable | Completion evidence |
|---|---|
| `W05_RAG_Long_Optimisation_Run_Config_v1.1.0.json` | Registered 3×3×2 design, 360 rows, 45 one-factor matched contrasts, exact source hashes |
| `W05_RAG_Long_Source_Optimisation_Report_v1.1.0.md` | Latest Pareto, factor, interaction, latency and scoring-completeness analysis |
| `W05_RAG_Long_Source_Optimisation_Summary_v1.1.0.json`, `.csv` | 18 factorial cells and complete Pareto membership |
| `W05_RAG_Long_Source_Optimisation_Item_Results_v1.1.0.csv` | 360 sanitized item rows |
| `W05_RAG_Optimisation.ipynb` | Standard reference deliverable; executed latest long-source notebook over the v1.1.0 sanitized outputs |
| `W05_RAG_Result_Version_Register_v1.1.0.json` | Machine-readable latest/superseded status for Week 3–5 RAG artifacts |
| `W05_RAG_Long_Source_Run_Manifest_v1.1.0.json` | Final row counts, public/private evidence hashes, scoring completeness and Judge runtime incident audit |

The reference-plan deliverables are complete. The v1.0.0 atomic-section optimisation artifacts are retained separately for provenance and are marked superseded.

| Deliverable | Completion evidence |
|---|---|
| `W05_PIC20_Model_Analysis.md` | Six PIC classes; each names the primary Phase B finding, a concrete platform failure, a class-specific readiness metric and an open question |
| `W05_RAG_Optimisation.ipynb` | Standard Week 5 entry point containing the latest long-source 18-cell factorial, Pareto, matched-contrast, interaction and completeness analysis |
| `../weekly/Wk-05-EvalLog.md` | Records the formal run, accumulated Week 2–4 evidence, PIC synthesis, limitations and claim boundary |

## Superseded atomic-section RAG provenance

> **Status: SUPERSEDED / HISTORICAL v1.0.0.** These artifacts remain auditable but must not be used as the latest Week 5 result. The atomic parent sections were already shorter than the registered chunk sizes, so chunk size was not an operational factor.

| Artifact | Purpose |
|---|---|
| `W05_RAG_Optimisation_Atomic_Sections_v1.0.0.ipynb` | Executed historical notebook with an explicit superseded banner |
| `W05_RAG_Optimisation_Run_Config_v1.0.0.yaml` | Frozen 3 × 3 × 2 design, source hashes, model revisions, timing rules, missing-value policy and Pareto rule |
| `W05_RAG_Production_Run.py` | Deterministic generation, randomized variant-block order, separate cold/warm timing, append-only resume and row traceability |
| `W05_RAG_Coverage_Scoring.py` | Bounded local required-point Judge with strict score validation and syntax-only JSON repairs |
| `W05_RAG_RAGAS_Scoring.py` | Local Faithfulness/Relevance scoring, exact-input reuse and bounded retry of missing transport results |
| `W05_RAG_Optimisation_Analysis.py` | 360-row validation, 18 cell summaries, 45 matched contrasts, interaction summaries, Pareto dominance and public sanitization |
| `W05_RAG_Optimisation_Item_Results_v1.0.0.csv` | 360 sanitized item rows; no raw questions, contexts, candidate outputs or Judge traces |
| `W05_RAG_Optimisation_Summary_v1.0.0.csv` | One row for each of 18 factorial cells |
| `W05_RAG_Optimisation_Summary_v1.0.0.json` | Traceable hashes, summaries, contrasts, Pareto set, scorer audit and missingness coverage |
| `W05_RAG_Optimisation_Report.md` | Submission-readable formal result and interpretation |

The historical v1.0.0 bundle contains 360/360 generation, coverage and RAGAS rows. All
Relevance, Coverage and warm-path latency values are finite; Faithfulness is
finite for 342/360 rows (`0.950` coverage). Eighteen no-claim cases remain
explicitly missing and are never imputed. Raw evidence and the complete package
freeze remain in the private support workspace.

Its historical non-dominated set contains five configurations. The old balanced
choice within that frontier is `chunk-256_topk-5_rerank-none` (Faithfulness
`0.9075`, Coverage `0.9325`, Relevance `0.6316`, p50 `7145.6 ms`). This is a
diagnostic choice for the frozen 20-item subset, not a production optimum.

## PIC and accumulated evidence

| Artifact | Purpose |
|---|---|
| `W05_PIC20_Evidence_Registry_v1.0.0.yaml` | Versioned GRPO/STUM/SEOM/AMDC/HTD-IRL/CRL-MRS terminology and evidence contracts |
| `W05_PIC20_Model_Analysis.md` | Capability-by-capability synthesis and readiness gaps |
| `W05_Accumulated_Evidence_Analysis.py` | Keeps unlike evaluation families separate and computes platform trends, failures, severity association and surprise evidence |
| `W05_Accumulated_Evidence_Analysis_v1.1.0.json` | Latest structured cross-week findings and traceability using corrected RAG inputs |
| `W05_Accumulated_Platform_Trends_v1.1.0.csv` | Latest stratified Week 2–4 platform/model summaries |
| `W05_Accumulated_Evidence_Report_v1.1.0.md` | Latest submission-readable accumulated analysis |

The largest PIC-specific readiness gap is CRL-MRS joint success under controlled
communication loss. STUM and SEOM remain versioned tracks rather than being
silently merged across conflicting source terminology.

## Verification

All Week 5 focused tests pass. The production acceptance checks require 18
variants × 20 items, 360 unique row IDs, 45 one-factor contrasts, exact model
revisions, effective reranking on all requested rows, explicit finite metric
coverage and strict JSON output without NaN. The final notebook executed without
errors.

## Claim boundary

Week 5 evaluates a public-source component stack with one A40 run and an
uncalibrated local Judge. It does not measure deployed Senpai or PIC 2.0, estimate
run-to-run variance, prove a mechanism outside the registered design, or establish
safety or production readiness.

# Week 3–5 Long-Source RAG Corrective Protocol

**Protocol version:** `1.0.0`  
**Frozen date:** 2026-08-12  
**Random seed:** `42`

## Why the rerun is required

The earlier RAG knowledge base stored short atomic sections as parent documents. A chunker cannot demonstrate meaningful 256/512/1,024-token behavior when most parents are already shorter than those limits. The old result was valid for that small curated corpus, but it was not a valid test of long-document chunk-size performance.

The corrective corpus stores each complete source as one long parent while preserving headings, paragraphs, table rows, PDF pages, source-block identifiers, and character offsets. Chunking occurs only inside the runtime pipeline. The three registered chunk sizes now create different retrieval units and can therefore be treated as an experimental factor.

## Source boundary

The public knowledge base contains all 21 usable official public sources in the frozen collection: seven current website pages and fourteen dated Future Series PDFs. Current pages carry design-intent status; dated PDFs carry historical/background status. Metadata records owner, access scope, confidentiality, claim status, status scope, authority tier, source version, snapshot hash, and access date.

Audited internal DOCX material is stored in a separate private collection and never enters public Chroma collections, question files, item-level exports, or reports. Source-conflicted functionality descriptions remain retrievable for explicit conflict questions but their metadata prevents them from being presented as verified deployed facts.

## Question design

The public benchmark has 40 human-authored questions: 20 Fari and 20 Senpai. The private diagnostic benchmark has 20 additional questions: 10 per platform. Questions target evidence inside the complete long files rather than the old atomic-section records.

The item taxonomy includes:

- local fact retrieval and table-row lookup;
- cross-section and long-range synthesis;
- current-versus-historical claim status;
- source conflict and answerability boundaries;
- STUM/SEOM versioned terminology;
- unanswerable questions that require abstention.

Each item registers expected document IDs, evidence-fact IDs, weighted required points, forbidden claims, answerability, difficulty, and question type. Required/forbidden rubrics are hidden from candidate generation and joined only during scoring.

## Cross-week rerun

| Week | Corrective run | Frozen comparison |
|---|---|---|
| 3 | Retrieval plus answer quality | 40 questions × base/RAG = 80 public rows; separate 40-row private diagnostic |
| 4 | RAG system performance | Same 40 public questions × base/RAG = 80 warm-path rows; warm-up excluded |
| 5 | Retrieval optimisation | 256/512/1,024 tokens × top-k 1/3/5 × reranking off/on × 20 Senpai questions = 360 rows |

Week 4 robustness, masked-input, multimodal, and non-RAG evaluations do not depend on the knowledge base and are carried forward unchanged. They are not rerun merely to inflate the experiment count.

## Controlled design and interpretation

Week 5 uses a full factorial design: every registered level of chunk size, top-k, and reranking is combined. Variant-block order is randomized with seed 42. Matched contrasts compare cells differing in exactly one factor; 45 such pairs are registered (18 chunk-size, 18 top-k, and 9 reranking pairs). Interactions are reported because, for example, the effect of reranking may depend on top-k.

The Pareto frontier maximizes diagnostic faithfulness and required-point coverage while minimizing warm-path latency. A Pareto-optimal cell is not dominated by another tested cell on all three objectives. Any balanced choice remains conditional on this corpus, question set, model stack, evaluator, hardware, and run.

Correlation does not establish causality, and a matched factor effect does not by itself prove a product mechanism. No result establishes deployment, clinical validation, safety certification, universal chunk-size superiority, or PIC-specific readiness.

## Reproducibility and privacy acceptance

- Exact model revisions and source hashes must match the run configuration.
- Preflight must show different chunk hashes/counts at all three sizes and complete evidence-fact span mapping.
- Base/RAG pairs and all factorial cells must be complete; failed scores remain explicit and are never imputed. A deterministic schema repair may discard Judge-created point IDs that are absent from the frozen rubric only when every registered point is present; discarded IDs and the source attempt must remain in the audit trail.
- Cold model/index loading, excluded warm-up, and warm steady-state latency remain separate.
- Raw prompts, contexts, answers, Judge traces, model caches, logs, and internal-source results remain private.
- Public artifacts contain only official-source inputs, sanitized item metrics, aggregates, hashes, and claim boundaries.

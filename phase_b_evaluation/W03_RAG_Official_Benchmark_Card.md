# Week 3 Official-Public RAG Benchmark Card

## Scope

This benchmark tests a local two-step RAG component using curated public
InGen Dynamics website evidence. It does not test deployed Fari or Senpai
systems, proprietary PIC behavior, clinical performance, certification, or
commercial readiness.

Versioned assets:

- knowledge base: `W03_RAG_Official_Knowledge_Base_v0.3.0.yaml`;
- evaluation set: `W03_RAG_Official_Eval_Set_v0.3.0.yaml`;
- controlled baseline: `W03_RAG_Official_Run_Config_v0.3.0.yaml`;
- prompt-only iteration: `W03_RAG_Official_Run_Config_v0.3.1.yaml`.

## Corpus

The v0.3.0 corpus contains four human-curated official-page snapshots:
homepage, Fari, Senpai, and Sentinel. It contains 16 section-level parent
documents and 30 atomic facts.

Every section records:

- document, section, parent, and child identifiers;
- URL, canonical domain, publisher, access date, and content hashes;
- official/user/internal ownership, access scope, and confidentiality;
- product, authority, claim status, conflict status, and current-status fields;
- chunk number, neighboring chunks, token count, and embedding revision.

Only official, public, current `www.ingendynamics.com` evidence matching the
question's product can enter retrieval. Homepage/Sentinel content is retained
to test product isolation; it is not eligible for Fari or Senpai answers.

## Questions

The frozen evaluation set contains 12 English scenario-like questions:

- Fari: 6;
- Senpai: 6;
- safety-, privacy-, authority-, dignity-, failure-, and development-status
  questions;
- positive evidence questions and negative/no-evidence questions;
- a hidden reference answer, weighted atomic requirements, evidence fact IDs,
  and prohibited claims for every item.

Candidate inputs never contain reference answers, scoring points, prohibited
claims, or evidence fact IDs. Hidden material is joined only after generation.

## Experimental conditions

The v0.3.0 baseline produces 24 rows:

- 12 `base` rows with no retrieved context;
- 12 `rag` rows with the same questions and eligible retrieved context.

Model revision, tokenizer revision, seed, decoding, hardware, and latency
boundary are fixed. Context availability is the only intended difference.

The v0.3.1 follow-up is a RAG-only prompt iteration. It inherits the v0.3.0
retrieved contexts byte-for-byte and changes one variable: the system prompt's
epistemic-polarity boundary.

## Paper-informed design

- RAGAS separates retrieval/context quality, faithfulness, and answer
  relevance rather than reporting one opaque score.
- EMNLP 2024 RAG best-practice work motivates metadata enrichment and
  small-to-big retrieval.
- HiChunk motivates document → section parent → child structure and
  child-to-parent merging.
- SELF-RAG motivates relevance/no-evidence gating instead of unconditional
  fixed retrieval.
- Atomic-unit retrieval work motivates fact-level recall and scenario questions
  derived from individually reviewable claims.
- ARES motivates treating automatic judges as provisional until validated
  against an external reference; this Week 3 submission supplements its local
  Judge with an explicitly disclosed AI qualitative calibration.

References:

- [RAGAS](https://aclanthology.org/2024.eacl-demo.16/)
- [Searching for Best Practices in Retrieval-Augmented Generation](https://aclanthology.org/2024.emnlp-main.981/)
- [HiChunk](https://aclanthology.org/2026.acl-long.1372/)
- [Question-Based Retrieval using Atomic Units for Enterprise RAG](https://aclanthology.org/2024.fever-1.25/)
- [SELF-RAG](https://openreview.net/forum?id=hSyW5go0v8)
- [ARES](https://aclanthology.org/2024.naacl-long.20/)

## Metrics

Retrieval:

- document recall@k and evidence-fact recall@k;
- hit@k and reciprocal rank;
- metadata-filter leakage;
- no-evidence accuracy;
- retrieval latency.

Generation:

- answer relevance;
- faithfulness to retrieved context;
- context recall/coverage, precision, and relevance;
- weighted atomic-point coverage;
- forbidden-claim and unsupported-claim rates;
- citation validity;
- generation latency and output tokens.

All local Mistral/RAGAS scores are marked as uncalibrated automatic diagnostics.
They are triangulated with the completed eight-row AI qualitative calibration
and must not be interpreted as usability percentages.

## Reproducibility and leakage controls

Every run archives the complete candidate-visible inputs, retrieved chunks and
ranks, outputs, metric reasons, model and tokenizer revisions, seed, decoding,
dependency versions, GPU information, timestamps, commands, and SHA-256
hashes. Run directories are immutable.

The benchmark answer key is public for review, but the generation pipeline
constructs a separate blind candidate input file. Reproducibility tests verify
that hidden reference and scoring fields are absent.

## Limitations

- The corpus is small and manually curated; perfect retrieval does not
  establish robustness on a larger internal knowledge base.
- Website statements are development intent and may change after the recorded
  access date.
- The evaluation set is suitable for a controlled Week 3 benchmark, not a
  population-level estimate of product quality.
- Automatic Judge scores are uncalibrated and should not be promoted as
  definitive.
- Current retrieval uses dense BGE-M3. The completed 256/512/1024 by top-k
  1/3/5 and reranker off/on ablation found that top-k mattered on this corpus,
  while chunk size was non-discriminating because every section was shorter
  than 256 tokens.

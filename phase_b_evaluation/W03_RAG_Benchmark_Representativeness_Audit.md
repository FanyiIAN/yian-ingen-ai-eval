# Week 3 RAG Benchmark Representativeness Audit

**Audit date:** 2026-07-29  
**Frozen benchmark:** `W03_RAG_Official_Blind_Eval_Set_v0.4.0.yaml`  
**Purpose:** determine what the Week 3 results do and do not establish without
changing the already-run benchmark after observing results.

## Bottom line

The eight-question public blind set is meaningful as an end-to-end smoke and
RAG-helpfulness check. It is not large or difficult enough to establish
production usability. The questions themselves include useful safety and
compound-policy cases, but retrieval is artificially easy because the governed
collection contains only 16 chunks: the Senpai metadata filter leaves four
eligible chunks and `top-k=4` returns all four; the Fari filter leaves five and
returns four.

The defensible Week 3 claim is:

> On a small, curated, metadata-isolated public collection, the pipeline ran
> end to end and retrieved evidence improved Llama-3.1-8B-Instruct's responses
> on the frozen sample.

The current evidence does not support:

> The RAG system is production-ready, scales to a realistic knowledge base, or
> will remain accurate under ambiguous, adversarial, stale, or access-controlled
> enterprise queries.

## How many questions exist

| Set | Questions | Role | Independent evidence status |
|---|---:|---|---|
| Earlier official development set | 12 | Prompt development and Llama pipeline iteration | Not blind after repeated inspection |
| Public three-model blind set | 8 | Main FLAN/Mistral/Llama Base-vs-RAG comparison | One preflight row was inspected; the remaining 7 form the primary aggregate |
| June private smoke set | 6 | Isolated private-collection retrieval and Llama generation smoke | Separate source and collection; not pooled with public results |

These are 26 executed questions in total, but they are not 26 exchangeable,
independent usability observations. The main public usability claim rests on
the eight frozen questions, with seven retained as uninspected primary evidence.

## Main eight-question coverage

| Dimension | Coverage |
|---|---|
| Platform | 4 Fari, 4 Senpai |
| Severity | 4 severity-5, 4 severity-3 |
| Complexity | 5 compound/multi-part, 3 focused |
| Evidence status | 6 evidence-supported operational/design questions, 2 not-established/status questions |
| Evidence composition | 4 require facts from multiple sections; 4 are primarily answerable from one section |
| Language | English only |
| Interaction | Single-turn only |

### What is meaningful

- The questions test medication authority, clinical diagnosis boundaries,
  graceful degradation, child safeguarding, consent, biometric privacy,
  mastery uncertainty, SEND dignity, certification and validation claims.
- Five questions require the answer to keep multiple constraints separate,
  which exposes completeness failures that a single factoid would miss.
- Four severity-5 cases test whether a model violates a high-consequence
  clinical, privacy, or safeguarding boundary.
- The two status questions test the important distinction between “not
  established” and a definite negative historical claim.
- Required points and forbidden claims make the expected behavior auditable.

### What is too easy or missing

- The corpus is extremely small. With platform metadata filtering, `top-k=4`
  retrieves all Senpai sections and nearly all Fari sections.
- Questions and curated passages share substantial vocabulary, so semantic
  retrieval faces little lexical mismatch.
- There are few genuine distractors and no near-duplicate, stale, lower-authority
  or contradictory passages inside the eligible candidate pool.
- There are no multi-turn follow-ups, misspellings, shorthand caregiver/teacher
  language, multilingual questions, or long conversational context.
- There are no table, numeric, temporal-update, comparison, troubleshooting or
  document-navigation tasks.
- No retrieved passage contains prompt injection or adversarial instructions.
- Access-control routing is checked by metadata contracts, but the benchmark
  does not present one user query that could legitimately route to either the
  public or private collection depending on authorization.
- The public corpus consists of curated paraphrases rather than full,
  heterogeneous production documents.

## What the ablation tells us

The 18-condition ablation found:

- `top-k=1`, reranker off: fact recall `0.7000`;
- `top-k=1`, reranker on: fact recall `0.7417`;
- `top-k=3` and `top-k=5`: fact recall `1.0000`;
- chunk sizes `256/512/1024` produced the same 16 chunks because every governed
  section was shorter than 256 tokens.

This confirms that retrieval is not completely trivial at `top-k=1`, but it
also confirms that perfect recall at `top-k>=3` is substantially caused by the
small collection. The ablation validates the pipeline and exposes this limit;
it does not remove it.

## Recommended next benchmark without invalidating Week 3

Keep v0.4.0 frozen as the Week 3 smoke benchmark. Create a separately versioned,
preregistered confirmation set with at least 24–32 questions and at least 100
eligible chunks, including realistic distractors. Balance these families:

1. direct fact lookup;
2. multi-section synthesis and exception handling;
3. evidence-insufficient questions requiring abstention;
4. source conflict, freshness, and authority resolution;
5. lexical mismatch, noisy wording, and multi-turn clarification;
6. adversarial retrieved text and unsafe-instruction resistance.

The public and June private collections should remain physically separate. A
future routing test can query both through an authorization-aware router, but
must verify that public requests never retrieve private chunks.

## Proposed future acceptance gates

The internship reference does not prescribe numerical RAG usability thresholds.
If a later confirmation benchmark is run, the following should be
**preregistered prospectively**, not applied retroactively to Week 3:

- private or ineligible metadata leakage: `0`;
- safety-critical forbidden-claim rate: `0`;
- evidence-fact recall@3: at least `0.85` on the expanded hard set;
- valid citation rate: at least `0.90`;
- reconciled required-point coverage: at least `0.80`;
- reconciled claim faithfulness: at least `0.80`;
- median human answer relevance: at least `4/5`;
- RAG must improve a paired human answer-quality measure over Base without
  materially reducing faithfulness;
- report latency and failure rates alongside quality rather than using one
  aggregate “RAG score.”

These are proposed engineering gates for a future study, not requirements from
the Week 3 reference and not claims already passed by the current system.

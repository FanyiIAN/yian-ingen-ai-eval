# Week 3 Evaluation Memo

**Scope:** three-model diagnostic comparison and local RAG component evaluation  
**Candidate added in Week 3:** `meta-llama/Llama-3.1-8B-Instruct`  
**Evaluation seed:** `42`  
**Claim boundary:** public/open model and public-data proxy evidence, not deployed InGen product performance

## Executive finding

Week 3 produced an operational, reproducible RAG evaluation pipeline and added
Llama-3.1-8B-Instruct to the frozen 35-scenario benchmark. The strongest valid
conclusion is about the RAG component: on 12 paired Fari/Senpai questions,
metadata-gated BGE-M3 retrieval supplied the expected governed evidence in every
case, and Llama's provisional answer relevance increased from `0.343704` without
context to `0.573415` with context. A prompt-only epistemic-boundary iteration
then increased provisional faithfulness from `0.818182` to `0.930556` while
keeping the retrieved contexts byte-for-byte fixed.

These results show that the implemented RAG path can make the selected open model
more useful and source-grounded on the registered component benchmark. They do
not show deployed Fari or Senpai performance, and perfect retrieval on four
curated website documents is not evidence that retrieval will remain perfect on
a large, noisy or conflicting corpus.

## Three-model comparison

Llama-3.1-8B-Instruct completed all 35 frozen Week 2 scenarios with no generation
errors or input truncation. The resulting comparison contains 105 rows across
FLAN-T5-base, Mistral-7B-Instruct-v0.2 and Llama-3.1-8B-Instruct, with identical
scenario versions, candidate prompt semantics, deterministic decoding and seed.
Latency was measured together with output quality because the internship plan
requires performance evaluation discipline rather than quality-only reporting.

The diagnostic Judge scores placed Llama above FLAN-T5 and below Mistral on the
severity-weighted aggregate, while Llama's throughput was close to Mistral's and
lower than FLAN-T5's. That ordering must not be interpreted as a validated
leaderboard. The inherited Week 2 Prometheus Judge failed its calibration gate,
many FLAN rows were unresolved, and the seven formerly held-out scenarios were
inspected during Week 2. The correct Week 3 decision is therefore to preserve
the complete model outputs, latency, score coverage and failure flags, but skip
an inferential model-ranking claim until model-blind human adjudication and a
newly sealed test set are available.

| Model | Severity-weighted task | Severity-weighted grounding | Severity-weighted quality | Resolved quality rows | Output tokens/s |
|---|---:|---:|---:|---:|---:|
| FLAN-T5-base | 3.127 | 3.377 | 3.725 | 13/35 | 49.602 |
| Llama-3.1-8B-Instruct | 3.970 | 4.050 | 4.061 | 32/35 | 27.872 |
| Mistral-7B-Instruct-v0.2 | 4.080 | 4.500 | 4.288 | 33/35 | 28.829 |

Resolved-score coverage is part of every result: a higher mean over fewer
resolved rows may reflect selection bias. The table is therefore a transparent
diagnostic record rather than the requested validated leaderboard.

Operationally, the replay still provided useful engineering evidence. Llama ran
the full harness reliably, followed the shared prompt contract and produced
auditable output for every scenario. It is therefore a viable local candidate
for the RAG experiment even though the three-model Judge cannot establish that
it is the best model.

## RAG architecture and evaluation

The pipeline uses LangChain 1.x, a pinned BGE-M3 embedding checkpoint and a
persistent Chroma database. Official website content is represented as governed
documents, section parents, smaller child chunks and atomic facts. Every chunk
records its source URL, publisher, capture date, ownership, confidentiality,
claim status, platform, section path, chunk index and count, neighboring chunk
identifiers, hash, fact identifiers and embedding revision.

The public source asset is `20,194` bytes and contains four official-page
snapshots, 16 sections/chunks, and 30 atomic facts; the persistent Chroma
directory is approximately `3.0 MB` including database and vector-index
overhead. The June private asset is physically separate: four documents, 12
sections/chunks, 12 facts, `12,063` source bytes, and an approximately `2.8 MB`
Chroma directory.

Retrieval first applies a hard metadata eligibility gate. Only official, public,
current, platform-matching evidence from the registered domain can enter the
candidate pool. Dense retrieval then fetches candidate child chunks, applies the
registered threshold and merges qualifying evidence into governed parent
sections. This small-to-big structure preserves retrieval specificity without
presenting isolated fragments to the generator. Candidate inputs never include
reference answers, scoring points, forbidden claims or expected evidence IDs.

The baseline registered 12 Fari/Senpai questions and generated paired base and
RAG answers under identical model, seed, decoding and hardware conditions.
Retrieval achieved document recall@k `1.0000`, evidence-fact recall@k `1.0000`,
hit@k `12/12`, MRR `1.0000` and zero metadata leakage. Mean retrieval latency
was `103.049 ms`. Llama generated all 24 answers; mean generation latency was
`2143.912 ms` for base and `3658.997 ms` for RAG.

The local Mistral evaluator reported provisional RAG answer relevance `0.573415`,
faithfulness `0.818182`, context relevance `0.479167`, context coverage/recall
`0.708333` and context precision `0.590278`. One faithfulness value was `NaN`
and is reported as invalid rather than silently converted to zero. All automatic
metrics remain diagnostics because the local Judge is uncalibrated. The separate
AI qualitative calibration reviews answer content directly.

Here, Base and RAG answer relevance are the same RAGAS metric applied without
and with retrieved context. The local Mistral Judge reverse-generates three
questions from each answer and BGE-M3 measures their cosine similarity to the
original question. It is therefore a semantic-alignment diagnostic, not a
percentage-correct or usability score. The internship reference requires
Faithfulness, Relevance and Coverage plus Base-vs-RAG comparison, but gives no
numerical pass threshold.
RAGAS `ContextRecall` measures whether retrieved passages cover the reference
answer; it does not directly measure whether the generated answer used all
relevant retrieved points. The frozen weighted required-point coverage in the
AI calibration supplies that answer-level completeness check.

## Controlled iteration and trade-off

The first answer review found two epistemic failures. One answer inferred a
repeated approval requirement from a statement that consent was revocable.
Another changed “the page does not establish clinical validation” into the
stronger negative claim “not clinically validated.” Version 0.3.1 changed only
the RAG system prompt to preserve epistemic polarity and prohibit invented
permissions, schedules and operational duties. Both registered failures were
removed; provisional faithfulness rose to `0.930556` with 12/12 finite rows and
answer relevance reached `0.580729`.

The same review identified incomplete coverage on a compound Senpai privacy
question. Four further one-variable prompt experiments tested increasingly
explicit completeness controls. The strongest checklist recovered all weighted
points but reintroduced an unsupported clinical-status statement. A final audit
removed that statement but materially reduced aggregate answer relevance and
faithfulness. The retained version is therefore 0.3.1, not because it solves
every completeness case, but because it gives the strongest stable balance on
the frozen development set. Further tuning on those inspected questions stopped
to avoid overfitting. A coverage-first prompt and a separate eight-question
blind confirmation set were registered for the final validation.

## Top three failure patterns and deployment implications

### 1. Epistemic-boundary expansion

The generator sometimes turns “not stated,” “not established,” or “revocable”
into a definite negative fact or a new operational rule. In a Fari context this
could invent clinical status, consent schedules or professional duties. In a
Senpai context it could invent school, privacy or safeguarding policy. The
required control is explicit qualification preservation plus claim-level
faithfulness review; a fluent answer is not sufficient.

### 2. Completeness versus concision

Short answers may omit supported parts of compound questions, while aggressive
coverage prompts may add irrelevant facts or unsupported implications. For Fari,
an omission could leave out an authority, privacy or consent condition. For
Senpai, it could omit retention, deletion, parental-consent or access-control
constraints. The evaluation must therefore score weighted atomic coverage
together with faithfulness and relevance rather than optimizing any one metric.

### 3. Evaluator and experimental-validity failure

Automatic Judges can return invalid values, disagree across repeated identical
context inputs or fail calibration entirely. Treating those outputs as ground
truth would produce a misleading model leaderboard or hide severe failures.
Every metric is consequently reported with valid-row coverage and invalid
reasons. Infrastructure and reporting failures have separate taxonomy codes and
never receive a model-quality score. The added content-validity layer is an
explicitly disclosed eight-row AI qualitative calibration using frozen evidence
and atomic requirements. It supplements rather than converts the automatic Judge
into a deployment-readiness measure.

## Final blind confirmation and ablation

The final v0.5.0 confirmation ran the same eight-question retrieval trace through
FLAN-T5-base, Mistral-7B-Instruct-v0.2, and Llama-3.1-8B-Instruct. All 48
generations and 48 local RAGAS rows completed. Because the first Base item was
inspected during adapter qualification, the primary aggregate uses seven
uninspected pairs per model.

Llama's provisional answer relevance increased from `0.290288` to `0.538142`
with RAG, and RAG faithfulness was `0.821429`. Mistral increased from `0.346790`
to `0.465645`, but its `0.880952` faithfulness is a non-independent self-judge
diagnostic. FLAN increased from `0.223139` to `0.410193` but failed the detailed
answer and citation contract, so that metric change is not treated as usable
answer evidence. The shared-input audit passed with no context or message-hash
mismatch.

| Evidence layer | Scope | Base relevance | RAG relevance | Change | Faithfulness / claim support | Required-point coverage | Conclusion |
|---|---|---:|---:|---:|---:|---:|---|
| RAGAS, FLAN | 7 uninspected pairs | 0.223139 | 0.410193 | +0.187054 | 0.513889 (6/7) | n/a | Output contract failed; metric change is not usability evidence. |
| RAGAS, Mistral | 7 uninspected pairs | 0.346790 | 0.465645 | +0.118855 | 0.880952 | n/a | Self-judge diagnostic only. |
| RAGAS, Llama | 7 uninspected pairs | 0.290288 | 0.538142 | +0.247854 | 0.821429 | n/a | Strongest evaluator-independent automatic RAG-helpfulness evidence. |
| AI calibration, Llama RAG | 8 answers | n/a | 4.375/5 | n/a | 30/31 claims (0.967742) | 0.872619 | Useful and grounded overall, with identifiable omissions and citation gaps. |
| Shared public retrieval | 8 questions | n/a | n/a | n/a | fact recall@k 1.0000 | document recall@k 1.0000 | Expected evidence was retrieved on the small controlled corpus. |

The automatic relevance values look low because Base has no product knowledge;
FLAN often emits labels, fragments, or question echoes; and Llama/Mistral
answers frequently contain multiple sub-answers, qualifications, and citations
that can lower a single embedding-similarity value despite being useful. The
local Mistral Judge is uncalibrated and may underestimate complex correct
answers, while the primary aggregate contains only seven uninspected questions
and therefore has high sampling variance. This explains why AI content review
reached `4.375/5`, materially above the automatic Llama RAG value of `0.538142`.
Automatic relevance is a diagnostic signal, not a usability percentage.

The 18-variant retrieval ablation showed that top-k, not chunk size, explained
the observed evidence coverage on this corpus. Fact recall was `0.7000` at
top-k 1 and `1.0000` at top-k 3/5. Reranking improved top-k 1 to `0.7417` but
added latency and gave no recall benefit at top-k 3/5. All chunk-size settings
produced the same 16 chunks because the governed sections were shorter than
256 tokens. This confirms that perfect document recall was partly a small,
metadata-isolated knowledge-base effect.

Finally, the June internship sources were loaded only into a separate private
Chroma collection. Its six-question retrieval smoke passed with zero metadata
leakage and Llama completed 12/12 private Base/RAG generations. No mixed
public/private three-model run was performed; the separation is an intentional
access-governance boundary.

The public blind set has four Fari and four Senpai questions, including five
compound and four severity-5 cases. It is nevertheless only a smoke/helpfulness
benchmark: top-k 4 returns all four eligible Senpai chunks and four of five Fari
chunks. It has no large-corpus, stale-source, multilingual, multi-turn,
prompt-injection or realistic noisy-query cases. The representativeness audit
therefore recommends a separately frozen 24–32-question, 100+-chunk
confirmation set rather than changing the observed Week 3 benchmark.

AI calibration has completed all eight Llama RAG rows: mean answer relevance
`4.375/5`,
weighted required-point coverage `0.872619`, claim-level supported fraction
`0.967742`, and zero forbidden-claim violations. These AI-review results remain
separate from automatic scores because they use different scales and cover eight
answers rather than the seven-pair uninspected aggregate.

## Conclusion

Week 3 demonstrates a functioning local RAG evaluation methodology: governed
ingestion, metadata-filtered persistent retrieval, Llama generation, paired
base/RAG comparison, component metrics, immutable evidence and a principled
three-level behavioral hierarchy, with a separate causal RAG taxonomy. RAG was
helpful on the registered benchmark, especially for
answer relevance and faithfulness after the controlled v0.3.1 change. The main
limitations are the small curated corpus, provisional automatic Judge, and
inherited Week 2 comparison validity. These boundaries are explicit, so the
pipeline is suitable as a reproducible smoke-test foundation rather than an
unsupported product-performance claim.

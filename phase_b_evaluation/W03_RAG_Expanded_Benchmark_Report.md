# Week 3 Expanded RAG Benchmark Report

> **Status: SUPERSEDED / NOT LATEST for RAG knowledge-base results.** This benchmark used pre-segmented atomic-section parents, so chunk size was not an operational long-document factor. Use `W03_RAG_Long_Source_Corrective_Report_v1.0.0.md` for the latest public RAG result.

**Phase:** B — systematic evaluation and retrieval-augmented generation

**Benchmark version:** `0.6.2`

**Evaluation seed:** `42`

**Run date:** 2026-08-04

**Claim boundary:** component-level evidence on frozen public website snapshots, not deployed InGen product performance

## Executive summary

This iteration replaces the original 16-chunk, eight-question RAG smoke test
with a substantially broader frozen benchmark: 331 traceable knowledge units
from three current official InGen Dynamics web pages and 40 Fari/Senpai
questions. The complete local pipeline uses LangChain 1.3.14, pinned BGE-M3
embeddings, a persistent Chroma 1.5.9 collection, a pinned BGE reranker, and
three locally hosted candidate models. No OpenAI API or other external model API
received the benchmark data.

The final retrieval setting uses a hard metadata eligibility gate, dense
fetch-32 retrieval, reranking, and top-10 context. On the frozen 40-question
set, it achieved mean evidence-fact recall `0.9021`, complete expected evidence
for `30/40` questions, and zero metadata leakage. All three candidates completed
all Base/RAG generations (`240/240` rows total). The run therefore establishes
that the expanded pipeline is operational and that its evidence flow is
measurable. It does not by itself establish answer correctness or production
readiness; automatic response scores, citation behavior, and the limited source
diversity are reported separately.

## Deliverable expansion

| Asset | Original Week 3 smoke test | Expanded benchmark |
|---|---:|---:|
| Official public source pages | 4 | 3 current pages |
| Knowledge units/chunks | 16 | 331 |
| Atomic facts | 30 | 331 units with fact-level IDs |
| Questions | 8 blind-confirmation questions | 40 frozen questions |
| Candidate models | 3 | 3 |
| Base/RAG generation rows | 48 | 240 |
| Retrieval configurations tested | 18 small-corpus variants | 10 expanded-corpus variants |
| Runtime observability | generation latency | stage latency, end-to-end latency, throughput, RAM and GPU memory |

The three expanded source snapshots were captured on 2026-08-04:

- `https://www.ingendynamics.com/` — SHA-256
  `A33134D6D1E433DD58A582F1949A6A864C770ADCDAB56A6D28E678C19DFA6C49`;
- `https://www.ingendynamics.com/fari.html` — SHA-256
  `4C73E3C5439EC1F1CC31B335A1563957BDCA45D25A9415F801F636B608C05F22`;
- `https://www.ingendynamics.com/senpai.html` — SHA-256
  `906D9E1CC45C3258AAB27D69AE36489D702FF39541EEF2AA038D028B689E71AB`.

## Reference-plan alignment

| Week 3 expectation | Expanded evidence | Status |
|---|---|---|
| Add a third local candidate | Llama-3.1-8B-Instruct retained with FLAN-T5-base and Mistral-7B-Instruct-v0.2 | Complete |
| Build a RAG pipeline | LangChain, pinned BGE-M3, persistent Chroma, metadata gate and reranker | Complete |
| Run Base versus RAG | 40 paired questions × three models = 240 rows | Complete |
| Evaluate retrieval and responses | Fact/document coverage, leakage, latency, local RAGAS diagnostics and failure inspection | Complete |
| Compare systems without overstating results | Shared-input audit, score coverage and explicit Judge/model boundaries | Complete |
| Preserve reproducible logs | Frozen inputs/configs, seed, revisions, prompts, contexts, outputs, hashes and resource traces | Complete |
| Summarize findings and limitations | This report plus machine-readable aggregate artifacts | Complete |

Each knowledge unit records its stable unit/fact ID, platform, source URL,
publisher, access date, snapshot hash, document and fragment identity, record
and chunk index, claim status, ownership/access class, curation method, and
neighbor/context fields. The public YAML is 338,156 bytes; the persistent
Chroma directory used by the run is 5,978,276 bytes, including database and
index overhead. Exact-normalized content is unique across all 331 units; there
are no empty units. Mean content length is 35 words and the maximum is 145.

Most detailed source statements describe design intent, engineering
characteristics, interfaces, or forward-looking capabilities. Only three units
are classified as current public statements. The benchmark therefore preserves
claim-status metadata and does not reinterpret design descriptions as evidence
of deployed capability.

## Benchmark design

The 40-question set is balanced across Fari and Senpai (`20/20`) and includes
17 hard, 17 medium, and six easy questions. It contains 37 answerable questions
and three deliberately `not_established` questions. The scoring rubric contains
100 required points, 40 forbidden points, 98 expected evidence-fact links, and
35 question-type labels. Questions cover multi-part synthesis, privacy and
authority boundaries, comparisons, mechanism explanations, negative evidence,
and explicit uncertainty.

Questions and rubrics were authored from the frozen public snapshots and locked
before candidate generation. Candidate prompts contain the user question and,
for RAG, retrieved eligible context; they do not contain reference answers,
required/forbidden points, or expected fact IDs. This is a controlled component
benchmark rather than a fully independent held-out test because the questions
and knowledge base originate from the same three pages.

The June 2026 internship collection remains in a physically and logically
separate private collection. It was not mixed with the public collection and is
not included in any expanded public aggregate. The earlier private smoke result
remains separate: `6/6` retrieval items and `12/12` Llama Base/RAG generations.

## Architecture

The run uses the following fixed path:

1. load the governed YAML units and validate source/access metadata;
2. embed eligible text with pinned BGE-M3;
3. store vectors and metadata in persistent Chroma;
4. apply hard public/current/platform filters before semantic ranking;
5. fetch 32 candidates, rerank them with the pinned BGE reranker, and retain 10;
6. build a claim-status-aware context with stable chunk IDs;
7. generate paired Base and RAG answers under the same candidate model and seed;
8. retain prompts, contexts, outputs, hashes, retrieval traces, stage timings,
   throughput, process/system memory, and GPU memory.

The detailed grounded prompt instructs the model to answer only from supplied
evidence, preserve uncertainty and claim status, address every sub-question,
avoid unsupported permissions or deployed-status claims, say when the evidence
does not establish an answer, and cite supporting chunk IDs. The Base prompt
uses the same answer contract but supplies no retrieved product context.

The three candidates are:

- `google/flan-t5-base`;
- `mistralai/Mistral-7B-Instruct-v0.2`;
- `meta-llama/Llama-3.1-8B-Instruct`.

All candidate inference was local on one NVIDIA A40 48 GB. Embedding, reranking,
generation, and evaluation checkpoints were pinned and retained on a persistent
volume.

## Research-informed choices

The implementation applies several findings from the Week 3 reading set:

- metadata is an eligibility and governance mechanism, not merely display data;
- small atomic units provide precise retrieval targets while enough neighboring
  context is retained for generation;
- retrieval and generation are evaluated separately so a fluent response cannot
  hide an evidence-retrieval failure;
- context size is tuned with ablation rather than maximized;
- answer evaluation combines retrieval coverage, groundedness/faithfulness,
  relevance, and explicit failure inspection rather than a single score.

The principal references are Wang et al., [*Searching for Best Practices in
RAG*](https://aclanthology.org/2024.emnlp-main.981/) (EMNLP 2024); Li et al.,
[*Retrieval Augmented Generation or Long-Context
LLMs?*](https://aclanthology.org/2025.coling-main.449/) (COLING 2025); Raina and
Gales, [*Fact-Split*](https://aclanthology.org/2024.fever-1.25/) (FEVER 2024);
[*HiChunk*](https://aclanthology.org/2026.acl-long.1372/) (ACL 2026);
[RAGAS](https://aclanthology.org/2024.eacl-demo.16/) (EACL 2024); and
[ARES](https://aclanthology.org/2024.naacl-long.20/) (NAACL 2024).

## Retrieval results and ablation

| Configuration | Reranker | Mean fact recall | Full evidence | Mean returned context |
|---|---:|---:|---:|---:|
| top-4 | off | 0.7275 | — | — |
| top-4 | on | 0.7879 | — | — |
| top-6 | off | 0.7875 | — | — |
| top-6 | on | 0.8375 | — | — |
| top-8 | off | 0.8250 | — | — |
| top-8 | on | 0.8688 | — | — |
| top-10 | off | 0.8562 | — | — |
| **top-10** | **on** | **0.9021** | **30/40** | **683 tokens** |
| top-12 | off | 0.8604 | — | — |
| top-12 | on | 0.9083 | 31/40 | 810 tokens |

Top-10 with reranking is retained because top-12 recovered only one additional
fully covered question and `+0.0062` mean recall while adding about 127 context
tokens per question. The final one-pass trace reported mean retrieval latency
551.7 ms, p50 498.2 ms, p95 677.8 ms, and zero metadata leakage. That trace
contains a first-query warm-up effect and is not the steady-state end-to-end
performance run.

| Slice | Questions | Mean fact recall | Full evidence |
|---|---:|---:|---:|
| Fari | 20 | 0.8583 | 13/20 |
| Senpai | 20 | 0.9458 | 17/20 |
| Easy | 6 | 1.0000 | 6/6 |
| Medium | 17 | 0.9020 | 13/17 |
| Hard | 17 | 0.8676 | 11/17 |
| Not established | 3 | 1.0000 | 3/3 |

Ten questions missed at least one expected fact. The strongest repeated pattern
is the Fari claim-status boundary fact, which was missed in four related
questions. Because that pattern was observed on this frozen set, a future
metadata-aware status-boundary boost must be registered and evaluated on a new
sealed set rather than tuned and claimed on these same 40 questions.

## Three-model generation audit

All `240/240` Base/RAG rows completed without empty output. The shared-input
audit passed: every model received the registered question and, under RAG, the
same retrieved context for that question.

| Model | Condition | n | Mean / p95 generation (ms) | Output tok/s | Mean output tokens | GPU peak |
|---|---:|---:|---:|---:|---:|---:|
| FLAN-T5-base | Base | 40 | 221 / 351 | 43.02 | 8.5 | 4.82 GiB reserved |
| FLAN-T5-base | RAG | 40 | 862 / 4,903 | 35.56 | 39.7 | 4.82 GiB reserved |
| Mistral-7B-Instruct-v0.2 | Base | 40 | 5,141 / 10,539 | 30.88 | 159.9 | 15.23 GiB reserved |
| Mistral-7B-Instruct-v0.2 | RAG | 40 | 6,673 / 13,179 | 27.91 | 189.7 | 15.23 GiB reserved |
| Llama-3.1-8B-Instruct | Base | 40 | 3,048 / 9,338 | 29.31 | 90.7 | 16.07 GiB reserved |
| Llama-3.1-8B-Instruct | RAG | 40 | 7,840 / 13,451 | 27.06 | 220.4 | 16.07 GiB reserved |

The largest candidate high-water measurements were approximately 17.62 GB GPU
device memory and 18.49 GB process RSS for Llama. These values are
hardware/configuration-specific and are not model-size estimates.

Generation success is not equivalent to answer quality. FLAN produced seven
Base question echoes and two RAG echoes. Strict bracket-citation compliance was
`0/40` for FLAN RAG, `1/40` for Mistral RAG, and `16/40` for Llama RAG.
All 52 bracketed citations emitted by Llama were valid eligible IDs, but only
40% of Llama answers followed the required bracket format. Citation adherence
is therefore the clearest prompt/decoding weakness in this run.

## Automatic response evaluation

<!-- RAGAS_RESULTS_START -->
The completed local RAGAS diagnostic used a loopback-only
Mistral-7B-Instruct-v0.2 evaluator and local BGE-M3 semantic similarity. All
240 candidate-condition rows were retained. Each cell reports the mean over
finite values followed by valid/total row coverage.

| Candidate | Base answer relevance | RAG answer relevance | Delta | RAG faithfulness | Context relevance | Context recall | Context precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| FLAN-T5-base | 0.422540 (36/40) | 0.448278 (37/40) | +0.025738 | 0.847222 (36/40) | 0.981250 (40/40) | 0.835833 (40/40) | 0.841127 (40/40) |
| Mistral-7B-Instruct-v0.2 | 0.222679 (36/40) | 0.635032 (35/40) | +0.412353 | 0.877956 (40/40) | 0.981250 (40/40) | 0.835833 (40/40) | 0.841127 (40/40) |
| Llama-3.1-8B-Instruct | 0.263927 (38/40) | 0.642118 (39/40) | +0.378191 | 0.876272 (40/40) | 0.981250 (40/40) | 0.835833 (40/40) | 0.841127 (40/40) |

The automatic evaluator detected a large Base-to-RAG relevance gain for Llama
and Mistral, but only a small gain for FLAN. This matches the generation audit:
FLAN often emitted short labels, fragments, or question echoes, whereas the two
instruction-tuned models used substantially more of the retrieved evidence.
All 120 applicable context-metric values were finite. Their identical values
across candidates are expected because every model used the audited shared
question/reference/context tuple and these candidate-invariant metrics were
computed once and reused.

There were 3 invalid metric calls for Llama, 11 for FLAN, and 9 for Mistral
after the registered one-retry policy. They are exposed through coverage rather
than silently converted to zero. Mistral candidate rows are a non-independent
self-judge case and cannot support a winner claim.
<!-- RAGAS_RESULTS_END -->

RAGAS values are diagnostic signals, not percentages of usability. The local
Judge is uncalibrated, semantic relevance can penalize useful multi-part and
citation-rich answers, and automatic groundedness does not replace direct
claim-level review. Accordingly, this report does not rank the three candidates
from automatic scores alone.

The streaming evaluator retries a transient local metric-call failure once and
records the attempt count and original error. A persistent failure remains an
invalid value with a reason; it is never converted to zero or omitted from
coverage.

Because context relevance, context recall, and context precision depend only on
the frozen question/reference/retrieved-context tuple, finite values are
computed once and reused across candidate models whose shared-input hash audit
passed. Each reused value records its source run-item ID and has no fabricated
latency. Answer relevance and faithfulness remain independently evaluated for
every candidate answer.

## Warm-path latency and resource profile

<!-- PERFORMANCE_RESULTS_START -->
The Llama performance companion run completed all 80 measured warm requests
after one explicit RAG warm-up. It records each component and the complete
question-to-response interval rather than treating generation latency as total
latency.

| Condition | n | End-to-end mean / p50 / p95 (ms) | Generation mean / p50 / p95 (ms) | TTFT p50 / p95 (ms) | Output tok/s |
|---|---:|---:|---:|---:|---:|
| Base | 40 | 3,187.7 / 1,725.6 / 9,723.3 | 3,184.3 / 1,722.0 / 9,720.6 | 81.0 / 105.6 | 27.88 |
| RAG | 40 | 8,552.9 / 8,288.1 / 14,261.3 | 8,045.9 / 7,852.1 / 13,780.8 | 809.2 / 1,123.5 | 26.34 |

For RAG requests, complete retrieval averaged 501.1 ms (p50 463.8; p95
732.1). Mean reranking was 267.0 ms and context assembly was 0.08 ms. The
reported 226.6 ms vector-search integration call includes the nested 60.9 ms
query-embedding measurement, so those two values must not be added together.
Generation remains the dominant end-to-end cost.

Per-request GPU device memory peaked at 21,301 MiB for both conditions;
per-request process RSS peaked at 2,577.6 MiB after model placement. During
model loading, process RSS briefly peaked at 17,143.4 MiB and GPU memory rose to
19,993 MiB. Mean sampled GPU utilization/power was 83.2%/241.0 W for Base and
89.1%/243.7 W for RAG.

Cold persistent-index initialization/verification took 385,840.8 ms despite
indexing zero new chunks, and Llama model loading took 61,382.6 ms. These cold
network-volume I/O costs are disclosed but excluded from warm request latency.
<!-- PERFORMANCE_RESULTS_END -->

Cold model and index initialization are reported separately from warm request
latency. On this persistent network volume, cold imports and checkpoint/index
loading are dominated by random I/O and must not be averaged into the serving
path.

## Interpretation and limitations

The strongest supported conclusion is that the expanded RAG pipeline is
operational and retrieves most registered evidence without crossing metadata
boundaries. It is more demanding than the original smoke test and exposes a
real recall/context trade-off. It is still not a production-readiness test.

Key limitations are:

- the 331 units originate from only three related official pages, so lexical and
  topical redundancy can inflate retrieval performance;
- questions were authored from the same frozen pages and are not an independent
  user-distribution sample;
- ten questions still have incomplete expected-evidence retrieval;
- candidate citation compliance is poor, especially for FLAN and Mistral;
- the fixed-order ablation has cache/order effects, so its latency values are
  descriptive while its recall comparisons are primary;
- the local Mistral Judge is uncalibrated, and judging Mistral with Mistral is
  non-independent;
- the corpus contains public component descriptions, many of them
  forward-looking, and cannot establish deployed-product behavior;
- multilingual, conversational, stale/conflicting-source, prompt-injection, and
  realistic noisy-query cases remain outside this benchmark.

The next defensible iteration should add independent source documents, freeze a
new user-like question set before observing outputs, preregister the
status-boundary retrieval change, add deterministic citation validation or
constrained citation generation, and repeat quality and performance runs on the
same hardware profile.

## Reproducibility and retained evidence

- Seed `42`; deterministic/greedy candidate decoding where supported.
- Public inputs, configs, analyzers, aggregate JSON/Markdown, and tests are in
  `phase_b_evaluation`.
- Raw prompts, complete model outputs, resource traces, source HTML snapshots,
  and local evaluator rows are retained in the private Phase B run archive.
- Public artifacts contain no private June content, credentials, RunPod
  connection data, or local private paths.
- The run manifest records artifact SHA-256 values and checkpoint revisions.

## Conclusion

The expanded Week 3 benchmark meets the requested 200+ chunk and approximately
40-question target and executes the complete governed RAG chain across three
models. Retrieval is useful but no longer artificially perfect: top-10
reranking reaches `0.9021` mean fact recall with a measurable context cost.
Generation is operational across all 240 rows, while citation-format failures,
limited source diversity, and diagnostic-only automatic judging remain explicit
quality gaps. The resulting assets are suitable as a reproducible Phase B RAG
evaluation foundation, not as a deployed-product performance claim.

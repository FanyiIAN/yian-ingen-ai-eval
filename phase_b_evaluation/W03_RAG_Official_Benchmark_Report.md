# Week 3 Official-Public RAG Benchmark Report

**Phase:** B — systematic evaluation and RAG  
**Candidate:** `meta-llama/Llama-3.1-8B-Instruct`  
**Baseline run:** `w03-official-rag-v0.3.0-seed42`  
**Controlled iterations:** `v0.3.1` through `v0.3.5`, each with one registered prompt change  
**Retained version:** `w03-official-rag-v0.3.1-prompt-seed42`  
**Result status:** RAGAS diagnostics plus completed AI qualitative calibration  
**Claim boundary:** public-data component-proxy evaluation, not deployed product performance

## Week 2 comparison dependency

The separate three-model comparison is retained only as a diagnostic appendix.
Its Prometheus Judge did not pass the Week 2 calibration gate, and its original
held-out rows were inspected during Week 2. Week 3 therefore does not use those
scores to rank FLAN-T5, Mistral, and Llama or to support a model-selection claim.
This is an inherited Week 2 benchmark-validity limitation, not a RAG pipeline
failure. The raw 105 rows, score coverage, latency, and failure flags remain
available for audit; a valid leaderboard requires model-blind human adjudication
and a newly sealed test set.

## Outcome

The end-to-end local pipeline is operational:

```text
official public pages
  -> governed YAML documents and atomic facts
  -> heading-aware 256-token child chunks
  -> BGE-M3 embeddings
  -> persistent Chroma collection
  -> metadata-gated top-k retrieval
  -> child-to-parent context merge
  -> Llama-3.1-8B-Instruct response
  -> independent local Mistral RAGAS diagnostics
```

The frozen official-public benchmark contains four governed source snapshots,
16 parent sections, 30 atomic facts, and 12 questions split evenly between Fari
and Senpai. It includes answerable questions, authority and privacy constraints,
and no-evidence questions. Candidate prompts never contain the hidden reference
answer, required scoring points, or forbidden points.

## Knowledge and retrieval design

Each child chunk records:

- source URL, source class, publisher and capture date;
- official/public/current status and platform;
- document, section, parent and child chunk identifiers;
- zero-based chunk index, chunk count, neighboring chunks and token count;
- atomic fact identifiers and frozen embedding model revision.

Retrieval first applies a strict official/public/current/domain/platform metadata
gate. It then fetches 12 semantic candidates, applies a cosine relevance threshold
of `0.20`, keeps up to four unique children, and merges their governed parent
sections. This implements the small-to-big and metadata-aware retrieval practices
identified in the Week 3 reading review while avoiding unsupported context expansion.

## Frozen run conditions

| Component | Frozen condition |
|---|---|
| Candidate | `meta-llama/Llama-3.1-8B-Instruct` revision `0e9e39f249a16976918f6564b8830bc894c89659` |
| Embedding | `BAAI/bge-m3` revision `5617a9f61b028005a4858fdac845db406aefb181` |
| Local Judge | `mistralai/Mistral-7B-Instruct-v0.2` revision `63a8b081895390a26e140280378bc85ec8bce07a` |
| Vector store | ChromaDB `1.5.9`, persistent private run directory |
| LangChain | `1.3.14`; Chroma integration `1.1.0`; Hugging Face integration `1.2.2` |
| Candidate decoding | BF16, greedy, `do_sample=false`, max 384 new tokens |
| Seed and hardware | `42`; RunPod NVIDIA A40 48 GB |
| API boundary | no OpenAI API and no external model API; Judge endpoint is loopback only |

## Baseline retrieval and generation

| Measure | Result |
|---|---:|
| Questions | 12 |
| Document recall@k | 1.0000 |
| Evidence-fact recall@k | 1.0000 |
| Hit@k | 12/12 |
| MRR | 1.0000 |
| Metadata leakage | 0 |
| Mean retrieval latency | 103.049 ms |
| Base generations | 12/12 |
| RAG generations | 12/12 |
| Empty generations | 0 |
| Base mean generation latency | 2143.912 ms |
| RAG mean generation latency | 3658.997 ms |

## v0.3.0 provisional automatic metrics

| Metric | Base | RAG | Coverage |
|---|---:|---:|---:|
| Answer relevance | 0.343704 | 0.573415 | 12/12 each |
| Faithfulness to retrieved context | n/a | 0.818182 | 11/12 finite |
| Context relevance | n/a | 0.479167 | 12/12 |
| Context recall | n/a | 0.708333 | 12/12 |
| Context precision | n/a | 0.590278 | 12/12 |

The one invalid faithfulness result is `W03-OFFICIAL-FARI-006::rag`: the
Judge returned `NaN` with no reason. It is reported as invalid and excluded from
the finite mean, not converted to zero. The metrics are diagnostic because the
local Judge has not yet been calibrated against independently double-reviewed
human labels.

## Controlled prompt iteration

The baseline review exposed two epistemic-boundary failures:

1. `FARI-002` inferred repeated re-approval from revocable consent even though the
   source did not state a repeated approval schedule.
2. `FARI-005` converted “the page does not establish clinical validation” into the
   definite claim “not clinically validated.”

Version `0.3.1` changed only the RAG system prompt. It requires the model to
preserve epistemic polarity and prohibits inventing operational requirements,
permissions, schedules or repeated approval duties. All 12 retrieved context
arrays are byte-for-byte identical to the parent run; the KB, questions, model,
decoding, seed and hardware are unchanged.

The controlled generation completed 12/12 rows with no empty output. Mean latency
was `3870.621 ms`, median latency `2954.347 ms`, and mean output length
`104.667` tokens. Qualitative regression checks confirmed that both registered
failures were removed.

The independent local scoring completed 12/12 rows for v0.3.1:

| Provisional metric | v0.3.0 RAG | v0.3.1 | Difference |
|---|---:|---:|---:|
| Answer relevance | 0.573415 | 0.580729 | +0.007314 |
| Faithfulness | 0.818182 (11/12 finite) | 0.930556 (12/12 finite) | +0.112374 |
| Context relevance | 0.479167 | 0.458333 | -0.020834 |
| Context recall | 0.708333 | 0.708333 | 0 |
| Context precision | 0.590278 | 0.590278 | 0 |

Because question and context inputs to the context-only metrics were identical,
the context-relevance difference cannot be attributed to the prompt; it is recorded
as repeat-run variability from the uncalibrated local Judge.

The v0.3.1 review also found a compound-answer completeness regression on
`SENPAI-005`. Four additional one-change iterations were therefore registered:

| Version | Single changed variable | Registered result |
|---|---|---|
| v0.3.2 | Soft completeness boundary | Early stop: `SENPAI-005` remained approximately 2/5 weighted-point coverage |
| v0.3.3 | Explicit compound-question checklist | Restored `SENPAI-005` to 5/5, but early stop after `FARI-005` regressed to an absence-to-negative claim |
| v0.3.4 | Final epistemic audit appended after the checklist | Passed qualitative gates, but answer relevance fell 0.044064 and faithfulness fell 0.096154 versus v0.3.1 |
| v0.3.5 | Question-scope guard appended after the audit | Early stop: `SENPAI-005` fell to 4/5 and `SENPAI-006` reintroduced an absence-to-negative certification claim |

The inspected questions are now development regression cases, not fresh
blind-test evidence. Each early-stop run retains its complete inputs, outputs,
configuration, hash, latency and gate review even though expensive RAGAS scoring
was intentionally skipped.

### Final retain decision

Version v0.3.4 completed 12/12 local score rows with no invalid metrics:

| Provisional metric | v0.3.1 | v0.3.4 | Difference |
|---|---:|---:|---:|
| Answer relevance | 0.580729 | 0.536665 | -0.044064 |
| Faithfulness | 0.930556 | 0.834402 | -0.096154 |
| Context relevance | 0.458333 | 0.458333 | 0 |
| Context recall | 0.708333 | 0.666667 | -0.041666 |
| Context precision | 0.590278 | 0.631944 | +0.041666 |

The pre-registered keep rule rejected v0.3.4 because both answer relevance and
faithfulness materially regressed. Version v0.3.5 then failed its qualitative
gate before scoring. The retained stable version is therefore v0.3.1. This is
not a claim that v0.3.1 solved compound completeness: it preserves the strongest
aggregate diagnostics and epistemic calibration while leaving a documented
completeness-versus-concision hypothesis for a new, independently frozen blind
set. Further tuning on these 12 inspected items was stopped to avoid overfitting.

## Failure analysis contract

Failures are recorded at three levels:

- retrieval: wrong document/fact, metadata leakage, low evidence coverage or
  excessive context;
- generation: unfaithful claim, absence-to-negative conversion, unsupported rule
  inference, irrelevant answer, missing abstention or citation failure;
- evaluation/runtime: invalid Judge output, calibration failure, infrastructure
  failure or reporting bug.

Infrastructure failures never receive a model-quality score. Invalid metrics keep
their coverage denominator and reason. Every future change must use a new immutable
run ID and alter one registered variable.

## Final three-model blind confirmation

The separately frozen v0.5.0 confirmation ran FLAN, Mistral, and Llama on one
shared eight-question retrieval trace. All 48 generation rows and 48 local
RAGAS rows completed. `W03-BLIND-FARI-001` was inspected during GPU adapter
qualification and is excluded from the seven-question uninspected aggregate.

| Model | Base answer relevance | RAG answer relevance | Change | RAG faithfulness |
|---|---:|---:|---:|---:|
| FLAN-T5-base | 0.223139 | 0.410193 | +0.187054 | 0.513889 (6/7 finite) |
| Mistral-7B-Instruct-v0.2 | 0.346790 | 0.465645 | +0.118855 | 0.880952 |
| Llama-3.1-8B-Instruct | 0.290288 | 0.538142 | +0.247854 | 0.821429 |

Mistral's row is a non-independent self-judge diagnostic. FLAN failed the
instruction/citation contract despite the metric increase. Llama supplies the
strongest evaluator-independent evidence that RAG helped on this benchmark, but
the sample remains too small for a general model-ranking claim.

Base and RAG answer relevance use the same RAGAS `AnswerRelevancy` method
without and with retrieved passages. The local Judge generates questions from
the response and BGE-M3 compares them semantically with the original question.
The result is not a correctness percentage and the Week 3 reference specifies
no numerical pass threshold.

The values look low because Base has no product knowledge; FLAN often emits
labels, fragments, or question echoes; and Mistral/Llama answers include
multiple sub-answers, qualifications, and citations that can lower a single
embedding-similarity value. The local Mistral Judge is uncalibrated, and the
primary aggregate contains only seven uninspected questions, so sampling
variation is large. The completed AI content review reached `4.375/5`, well
above Llama's automatic RAG relevance of `0.538142`; the automatic value is
therefore treated as a diagnostic signal rather than a usability percentage.

The registered 18-variant retrieval ablation found fact recall `0.7000` at
top-k 1, `0.7417` with reranking at top-k 1, and `1.0000` at top-k 3/5.
Chunk size was non-discriminating because all 16 governed sections were below
256 tokens. The June sources were tested in a separate private collection only.
No mixed public/private three-model run was performed.

The separately documented representativeness audit classifies the eight
questions as a smoke/helpfulness benchmark rather than production-usability
evidence. The disclosed AI qualitative calibration of all eight Llama RAG rows
is complete: mean answer relevance `4.375/5`, weighted required-point coverage
`0.872619`, claim-level supported fraction `0.967742`, and zero forbidden-claim
violations. It supplements the automatic metrics but is not an independent
deployment certification.

## Reproducibility hashes

| Asset | SHA-256 |
|---|---|
| Official KB v0.3.0 | `275135e318d81a1aff4032acf19a1d4878c2f2d9fc8689e74aa9ddef89fc41e1` |
| Official eval set v0.3.0 | `ca34132b1e503231aae89ea3b622e04ebfd84013392dd2c1a89d6cd416aadfb6` |
| Baseline config v0.3.0 | `a4300d5b875b28db59e4c1c9bcfa466628be899ed42eb34d6af71e609eca3f05` |
| Prompt config v0.3.1 | `27f3cf19cbd4daac698d3fd7e4bc45ba11971eaf6e4159ebc1732390963e168b` |
| Completeness config v0.3.2 | `bfbd1b1bf857c3d86274940a903cd408d6b40d7146471b649da474cc06f1e6df` |
| Checklist config v0.3.3 | `815ada8480e842f5afd9021e3956b02f1a4d49478515aa28abc10a707919c652` |
| Final-audit config v0.3.4 | `500f2c1ec341487e9a47689a209889153bdbcbbcc7adb608051555cf6e5f8359` |
| Scope-guard config v0.3.5 | `76709dd0382eb5099000ad6da8dc6cfaf0f3eaa05d585d62cfd8412bcc592197` |
| v0.3.1 generations | `b2b55c05f0c3685dac668b5e7961f24e6377709952be43bd17e5c7f4d0e15235` |
| v0.3.2 generations | `23a753cc5435d94e7f8a0a24262e2a1e841fb684f94aa773de75b52f69fc977c` |
| v0.3.3 generations | `dfbc5196d7d2f56228bd57a66f91bb939f2ab31654b5c208105053d3528a142d` |
| v0.3.4 generations | `62ecaeee7af0dc58a2a01a9a43b460a2d83f2751f8e8f1f3a7986656a73ff39a` |
| v0.3.4 RAGAS rows | `e68cc15aeba64ed52effc64eff60ac2efecd49b1f57cf012d97c72aed3c5829` |
| v0.3.5 generations | `ed169e5a0676a19fa09f9be371cb1253fe229718729ee1ffac6b83f343911cb4` |
| Three-model summary v0.1.2 | `dd37e30080fb0a8e49a56e068a6404b84bdd6f10b84af6294a2dcdc6cca033a3` |
| AI calibration annotations v0.3.0 | `62a3d27004b619f51afb1907583e8d99ba370013fb326572b6067dd0660f7e9b` |

Raw inputs, outputs, retrieval traces, per-item latency, exact commands, GPU
snapshots, model revisions, health probes, Judge reasons and repair logs are retained
in the immutable private RunPod run directories. Runs v0.3.0 through v0.3.5 also
have a non-destructive `364 KiB` archive with SHA-256
`fde012cf42ea3a1ee3235903c6bd09486206bee4ba467faf92f5674bae96ba9b`.
Confidential source documents and hidden answer keys are not committed to the
public repository.

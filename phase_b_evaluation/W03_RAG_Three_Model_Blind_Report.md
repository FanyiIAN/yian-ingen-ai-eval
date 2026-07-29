# Week 3 Three-Model RAG Blind-Set Report

**Phase:** B — systematic evaluation and RAG  
**Run:** `w03-official-rag-multimodel-v0.5.0-blind-seed42`  
**Models:** FLAN-T5-base, Mistral-7B-Instruct-v0.2, Llama-3.1-8B-Instruct  
**Conditions:** eight paired Base/RAG questions per model, 48 generations total  
**Claim boundary:** public-data component proxy; not deployed product performance  
**Metric status:** RAGAS diagnostics plus completed AI qualitative calibration

## Outcome

The same BGE-M3 retrieval results improved provisional answer relevance for all
three models. Llama showed the largest Base-to-RAG change among the two candidates
judged by a different model: `0.290288` to `0.538142` (`+0.247854`), with RAG
faithfulness `0.821429`. FLAN also changed with context, but its outputs were
usually labels, fragments, or question echoes and contained no eligible chunk
identifier. Its metric increase is therefore not evidence of a usable answer.

Mistral's RAG answer relevance was `0.465645` and faithfulness `0.880952`, but
those values are self-judge diagnostics because the local evaluator is the same
Mistral checkpoint. They cannot support a cross-model winner claim. The valid
Week 3 conclusion is narrower: RAG helped Llama materially on this small
registered benchmark, while model capacity and instruction following remained
important after retrieval succeeded.

## What Base and RAG answer relevance mean

`Base answer relevance` and `RAG answer relevance` are the same RAGAS
`AnswerRelevancy` metric applied to two conditions:

- **Base:** the model answers the frozen question without retrieved passages;
- **RAG:** the same model answers the same question after receiving the frozen
  retrieved passages.

The metric is not retrieval recall and is not a percentage of correct facts.
RAGAS 0.4.3 asks the local Mistral Judge to generate three questions that the
candidate response appears to answer, embeds those generated questions and the
original question with BGE-M3, averages cosine similarity, and reduces a wholly
noncommittal answer to zero. A score of `0.54` therefore means moderate semantic
alignment under this evaluator, not “54% correct” or “54% usable.”

The absolute values are low for five observed reasons:

1. Base has no product knowledge, so product-specific questions are inherently
   difficult without retrieved evidence.
2. FLAN often emits labels, fragments, or a question echo, so reverse-generated
   questions align poorly with the original.
3. Mistral and Llama answers contain qualifications, several sub-answers and
   citation text; those can be useful yet reduce one-vector semantic similarity.
4. The local Mistral Judge is uncalibrated and may underestimate structurally
   complex but correct answers.
5. The primary aggregate contains only seven uninspected questions, so sampling
   variation is large.

The Week 3 programme reference requires reporting Faithfulness, Answer
Relevance, and Context Coverage separately and comparing Base with RAG. It does
not prescribe a numerical pass threshold. The completed AI qualitative
calibration is therefore reported as a complementary content review, while the
automatic values remain diagnostic rather than pass percentages.

One implementation distinction matters: RAGAS 0.4.3 `ContextRecall` checks
whether the retrieved passages contain the information in the reference answer.
It does not directly measure whether the candidate answer used every relevant
retrieved point, although the programme primer describes that latter idea as
Context Coverage. The frozen AI weighted required-point coverage is the
answer-level completeness measure used to close this gap.

## Integrity and comparability

The frozen benchmark has eight questions: four Fari and four Senpai. The first
Base row, `W03-BLIND-FARI-001`, was inspected during GPU adapter qualification.
It is retained in the raw record but excluded from the uninspected aggregate
below. The primary aggregate therefore contains seven paired questions per model.

All three candidates used the same:

- questions, retrieved contexts, context order, metadata filters, and chunk IDs;
- detailed Base and RAG semantic instructions;
- deterministic greedy decoding, maximum 384 new tokens, and seed `42`;
- A40 GPU and local checkpoints.

Only model-required serialization differed: FLAN text-to-text, Mistral folded
single-user instruction, and Llama native system/user chat. The automated
shared-input audit passed with zero message-hash or context-ID mismatches.

## Knowledge-base inventory

The official public collection is a `20,194`-byte governed YAML source that
becomes a roughly `3.0 MB` persistent Chroma directory after database and vector
index overhead. Its logical content is:

| Source snapshot | Scope |
|---|---|
| InGen homepage | Product status labels, Origami ecosystem framing, roadmap qualifications |
| Fari page | Development status, human clinical authority, medication escalation, privacy/consent, no diagnosis, audit and graceful degradation |
| Senpai page | Development status, mastery uncertainty, teacher/safeguarding authority, SEND dignity, child privacy and controls |
| Sentinel Prime AI page | Public status and design-intent boundaries |

Together these form 4 documents, 16 parent sections, 30 atomic facts and 16
indexed chunks. Each chunk retains source URL, access date, owner, authority,
public/private scope, platform, section path, claim status, zero-based chunk
index, chunk count and content hash.

The June private collection is a separate `12,063`-byte governed YAML source and
roughly `2.8 MB` Chroma directory. It contains four curated draft-requirement
documents covering Fari and Senpai, 12 sections, 12 atomic facts and 12 indexed
chunks. Its text and raw answers remain outside the public repository.

## Uninspected seven-question results

| Model | Base answer relevance | RAG answer relevance | Change | RAG faithfulness | Valid faithfulness rows |
|---|---:|---:|---:|---:|---:|
| FLAN-T5-base | 0.223139 | 0.410193 | +0.187054 | 0.513889 | 6/7 |
| Mistral-7B-Instruct-v0.2 | 0.346790 | 0.465645 | +0.118855 | 0.880952 | 7/7 |
| Llama-3.1-8B-Instruct | 0.290288 | 0.538142 | +0.247854 | 0.821429 | 7/7 |

The Mistral row is not evaluator-independent. No automatic cross-model ranking is
reported. FLAN's relevance increase is also qualified by its deterministic
format and echo failures.

| Model | Condition | Mean input tokens | Mean output tokens | Mean generation latency | Question echoes |
|---|---|---:|---:|---:|---:|
| FLAN-T5-base | Base | 318.43 | 6.29 | 132.98 ms | 1/7 |
| FLAN-T5-base | RAG | 1252.14 | 14.57 | 396.36 ms | 1/7 |
| Mistral-7B-Instruct-v0.2 | Base | 283.57 | 185.14 | 5889.37 ms | 0/7 |
| Mistral-7B-Instruct-v0.2 | RAG | 1268.71 | 189.43 | 6333.67 ms | 0/7 |
| Llama-3.1-8B-Instruct | Base | 270.00 | 87.71 | 2932.55 ms | 0/7 |
| Llama-3.1-8B-Instruct | RAG | 1091.29 | 184.00 | 6296.25 ms | 0/7 |

Mean RAG retrieval latency was `69.71 ms` over the seven uninspected questions.
There were no empty generations.

## Retrieval and citation behavior

The shared eight-question retrieval run achieved:

| Retrieval measure | Result |
|---|---:|
| Document recall@k | 1.0000 |
| Evidence-fact recall@k | 1.0000 |
| Hit@k | 8/8 |
| MRR | 1.0000 |
| Metadata leakage | 0 |

Generation did not inherit that perfection. On the seven uninspected RAG rows:

| Model | Rows mentioning an eligible chunk | Strict `[chunk_id]` rows | Invalid chunk IDs |
|---|---:|---:|---:|
| FLAN-T5-base | 0/7 | 0/7 | 0 |
| Mistral-7B-Instruct-v0.2 | 5/7 | 1/7 | 0 |
| Llama-3.1-8B-Instruct | 6/7 | 1/7 | 0 |

Mistral and Llama often used a correct eligible ID in parentheses, with a
`Chunk ID:` prefix, or with extra text inside brackets. Those answers remain
traceable, but they violate the exact citation output contract. This is recorded
as a generation-format failure, not a retrieval failure. FLAN's failure is more
fundamental: it did not produce grounded, auditable answers under the shared
detailed prompt.

## Retrieval ablation

The registered `3 × 3 × 2` ablation tested chunk sizes `256/512/1024`, top-k
`1/3/5`, and BGE-reranker-v2-m3 off/on.

| Setting | Fact recall | Document recall | MRR | Mean context tokens | Latency effect |
|---|---:|---:|---:|---:|---|
| top-k 1, reranker off | 0.7000 | 1.0000 | 1.0000 | 103.2 | Lowest context budget |
| top-k 1, reranker on | 0.7417 | 1.0000 | 1.0000 | 109.6 | Adds about 25–48 ms |
| top-k 3, either mode | 1.0000 | 1.0000 | 1.0000 | about 329 | Reranker adds no recall |
| top-k 5, either mode | 1.0000 | 1.0000 | 1.0000 | 524.0 | Extra context, no recall gain |

Chunk size did not change any result because all 16 governed website sections
were already shorter than 256 tokens; every index therefore contained the same
16 chunks. Perfect document recall and MRR are partly a small-corpus and
platform-metadata effect. Fact recall at top-k 1 shows that the benchmark is not
universally saturated, but this corpus cannot establish a general chunk-size or
reranker advantage. The practical default remains top-k 3 without reranking for
this smoke corpus.

## Separate private-source smoke

The June internship data source collection was not mixed with the official
website collection. It used a separate private configuration, Chroma directory,
and immutable run directory. The private smoke contained four governed
documents, 12 sections, 12 facts, and six paired questions. Retrieval achieved
6/6 hits, document/fact recall `1.0000`, MRR `1.0000`, and metadata leakage `0`;
Llama completed all 12 Base/RAG generations. Raw private content and outputs
remain outside the public repository.

No three-model public/private mixed-corpus run was performed. This was
intentional: physically mixing the collections would weaken source and access
governance. A future experiment may query them through an authorization-aware
router while retaining separate stores and verifying zero private retrieval for
public requests.

## Representativeness and usability boundary

The eight questions are meaningful safety and policy-grounding tests, but they
are not a production-usability benchmark. Five are compound questions and four
carry severity 5; however, with only four eligible Senpai chunks and five
eligible Fari chunks, `top-k=4` retrieves all or nearly all platform material.
The set has no multi-turn, multilingual, noisy, stale-source, prompt-injection,
table/numeric or realistic large-corpus cases.

The detailed
[`W03_RAG_Benchmark_Representativeness_Audit.md`](W03_RAG_Benchmark_Representativeness_Audit.md)
therefore classifies this as a smoke/helpfulness test. It recommends a separately
versioned 24–32-question, 100+-chunk confirmation benchmark rather than changing
the already-observed frozen set.

## AI qualitative calibration

OpenAI Codex/ChatGPT reviewed all eight frozen Llama RAG answers against the
retrieved passages and registered atomic requirements. Per-row automatic scores
were not viewed before the content review. This is an explicitly identified AI
calibration, not an independent deployment certification.

| Evidence layer | Scope | Base relevance | RAG relevance | Change | Faithfulness / support | Answer completeness | Main interpretation |
|---|---|---:|---:|---:|---:|---:|---|
| RAGAS, FLAN | 7 uninspected pairs | 0.223139 | 0.410193 | +0.187054 | 0.513889 (6/7) | n/a | Metric rose, but fragmented output and citation-contract failure make usability unsupported. |
| RAGAS, Mistral | 7 uninspected pairs | 0.346790 | 0.465645 | +0.118855 | 0.880952 | n/a | Non-independent self-judge diagnostic; no winner inference. |
| RAGAS, Llama | 7 uninspected pairs | 0.290288 | 0.538142 | +0.247854 | 0.821429 | n/a | Strongest evaluator-independent automatic evidence that RAG helped. |
| AI qualitative calibration, Llama RAG | 8 answers | n/a | 4.375/5 | n/a | 30/31 claims supported (0.967742) | 0.872619 weighted coverage | Answers were useful and grounded overall; omissions and citation compliance remain visible. |
| Retrieval, shared public set | 8 questions | n/a | n/a | n/a | fact recall@k 1.0000 | document recall@k 1.0000 | Needed evidence was retrieved on this small controlled corpus. |

The AI aggregate uses all eight answers, while the primary RAGAS aggregate uses
seven uninspected pairs; the values are complementary and not one-to-one
replacements. The `4.375/5` AI content rating is substantially higher than the
`0.538142` automatic Llama value because RAGAS answer relevance is an embedding
diagnostic rather than a usability percentage.

The calibration labels have distinct meanings:

| Label | Definition |
|---|---|
| Faithfulness | Split the answer into factual claims and ask whether each is supported by the retrieved passages; it does not ask whether the answer is complete. |
| Answer relevance | AI ordinal 1–5 review of whether the response directly and completely answers the question; it does not by itself establish factual support. |
| Required-point coverage | Score each frozen atomic requirement as 0, 0.5 or 1, multiply by its severity-aware weight, and divide earned weight by possible weight. |
| Forbidden claim | Boolean check for a scenario-specific unsafe or disallowed assertion/action, such as permitting autonomous dose changes. |
| Failure code | The primary causal defect from the Week 3 taxonomy, such as missing evidence, unsupported generation, omitted required point or missing citation; severity is recorded separately. |

The complete row-level table and annotation file are
[`W03_RAG_AI_Calibration_Report.md`](W03_RAG_AI_Calibration_Report.md) and
`W03_RAG_AI_Calibration_Annotations_v0.3.0.yaml`.

## What the automated verification does and does not prove

The original `50/50` tests are unit and contract tests: asset/schema validation,
answer-key leakage prevention, public/private metadata gates, Base/RAG pairing,
model-specific prompt serialization, frozen-input hash equality, result-analysis
edge cases, citation diagnostics, prompt-iteration isolation and the 18-condition
ablation registry. They provide strong evidence that the evaluation code obeys
its declared contracts. They do not measure whether answers are useful to real
users. AI-calibration artifact contracts were then added; the final verified
suite passed `53/53`.

The `32/32` Python syntax check means that Python can parse and byte-compile the
files.
This catches syntax, indentation and encoding errors; it does not execute GPU
inference, validate dependencies, detect every logic bug, or establish answer
quality. End-to-end generation, retrieval, RAGAS, ablation and AI qualitative
review are separate evidence layers.

## Reproducibility

| Asset | SHA-256 |
|---|---|
| Official KB v0.3.0 | `275135e318d81a1aff4032acf19a1d4878c2f2d9fc8689e74aa9ddef89fc41e1` |
| Blind eval set v0.4.0 | `1e7835618c689db149da75a259fe06c354d00aae8793d3b8edfcabc91c1ef25a` |
| Multi-model config v0.5.0 | `bab5194cf2e6b43e10500621774d27c8c7f1c3e8009abda25fd739f096c89490` |
| FLAN generations | `85b200aaf9dd988be9c7e0ee8defd4f9ac96244566e01b0aab13a1cf64bddd1b` |
| Mistral generations | `78051771cd39bb632d1ffb62c0b8fab228a44a003bec88d92c09538b98a443aa` |
| Llama generations | `3a474eade5db44f9da237a5d82b2f70e4cb2bb2a567258525afe7d826a011151` |
| FLAN scored rows | `f861430aa86e325db04bb716194166eac13db053ac5015fb5a60d6b1fa3d42a9` |
| Mistral scored rows | `7165a50031fb0600082ef93e39c89def286c6d0fdb98c027d62ed72852c6a904` |
| Llama scored rows | `1abb6f8530f600ca1a7c48e46380bc030525220d12e2a786f6c6ac1a408053f4` |
| Ablation result | `a3050de99781f9e62eefab2357e736a80dbabcff4d57d26e72a679ac798ebd68` |
| Three-model summary v0.1.2 | `dd37e30080fb0a8e49a56e068a6404b84bdd6f10b84af6294a2dcdc6cca033a3` |
| AI calibration annotations v0.3.0 | `62a3d27004b619f51afb1907583e8d99ba370013fb326572b6067dd0660f7e9b` |

Raw inputs, outputs, configs, model revisions, prompt hashes, retrieval traces,
per-row latency, Judge rows, logs, and the original runner `0.1.0` FLAN preflight
are retained in private immutable run directories.

## Method sources

- [LangChain retrieval and two-step RAG](https://docs.langchain.com/oss/python/langchain/retrieval)
- [LangChain evaluation approaches](https://docs.langchain.com/langsmith/evaluation-approaches)
- [Llama-3.1-8B-Instruct model card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [Mistral-7B-Instruct-v0.2 model card](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- [FLAN-T5-base model card](https://huggingface.co/google/flan-t5-base)
- [Retrieve-Plan-Generation](https://aclanthology.org/2024.emnlp-main.270/)
- [(D)RAGged Into a Conflict](https://research.google/pubs/dragged-into-a-conflict-detecting-and-addressing-conflicting-sources-in-retrieval-augmented-llms/)

# Week 3 Evaluation Log

> **2026-08-12 corrective update:** The earlier atomic-section RAG result below is retained as history but is not the latest knowledge-base result. The latest run indexes 21 complete public sources, uses a 40-question long-document set, and is reported in `phase_b_evaluation/W03_RAG_Long_Source_Corrective_Report_v1.0.0.md`.

## Long-source corrective rerun

- Complete-document preflight produced 406/185/94 distinct chunks at 256/512/1,024 tokens and mapped all 58 public evidence facts.
- Public retrieval reached 1.000 document recall@8, 0.900 fact recall@8, 0.975 MRR, and zero metadata leakage across 40 questions.
- Paired RAG minus base deltas were +0.656 Answer Relevance and +0.523 weighted Coverage. RAG won Coverage on 36 questions and tied on four; mean RAG Faithfulness was 0.893.
- Formal Answer Relevance/Faithfulness scoring completed 80/80. Coverage completed 80/80 after 14 deterministic, audited removals of unregistered extra point IDs; no registered-point verdict was imputed.
- A separate governed private track completed 20 questions and 40 base/RAG rows. Raw internal source material, questions, contexts, answers, and Judge traces remain outside the public repository.
- The attempted extended Context Relevance/Recall/Precision diagnostic was stopped after persistent HTTP-client retries; those metrics are `NA`, while exact document/fact retrieval metrics and weighted Coverage remain formal.

**Week:** 2026-07-27 to 2026-07-31  
**Phase:** B — third model and retrieval-augmented generation  
**Candidate model:** `meta-llama/Llama-3.1-8B-Instruct`  
**Official RAG baseline:** `w03-official-rag-v0.3.0-seed42`  
**Controlled iterations:** `v0.3.1` through `v0.3.5`  
**Retained RAG version:** `w03-official-rag-v0.3.1-prompt-seed42`  
**Final blind run:** `w03-official-rag-multimodel-v0.5.0-blind-seed42`  
**Claim boundary:** public-data component-proxy evaluation only

## Deliverable status

| Requirement | Evidence | Status |
|---|---|---|
| Add a third model | Frozen Llama-3.1-8B-Instruct Week 2 replay | Complete |
| Compare all three candidate models | `W03_Three_Model_Diagnostic_Report.md` | Inferential comparison skipped: inherited Week 2 Judge-calibration failure; raw diagnostic evidence retained |
| Build a LangChain RAG pipeline | BGE-M3, persistent Chroma and Llama pipeline | Complete |
| Govern official knowledge | Four source snapshots, 16 sections and 30 atomic facts | Complete |
| Design a source-grounded benchmark | 12 hidden-rubric Fari/Senpai questions | Complete |
| Compare base with RAG | 12 paired rows per condition | Complete |
| Record inputs, outputs and settings | Immutable RunPod run directories and SHA-256 manifests | Complete |
| Evaluate retrieval and response components | Recall/MRR plus provisional local RAGAS | Complete |
| Iterate on observed failure | Five registered prompt-only versions with inherited retrieval traces and regression gates | Complete; v0.3.1 retained |
| Submission-facing RAG notebook | `W03_RAG_Evaluation.ipynb` | Complete and executed locally |
| Three-page evaluation memo | `W03_Evaluation_Memo.md` | Complete |
| Three-model RAG blind confirmation | 48/48 FLAN/Mistral/Llama Base/RAG rows | Complete; first preflight item excluded from uninspected aggregate |
| Chunk/top-k/reranker ablation | `W03_RAG_Retrieval_Ablation_Report.md` | Complete; 18/18 variants |
| Separate June private collection | Private Chroma and immutable run directory | Complete; 6/6 retrieval and 12/12 Llama generations |
| AI qualitative calibration | Eight frozen Llama RAG answers plus row-level evidence table | Complete; 4.375/5 answer relevance, 0.872619 weighted coverage, 30/31 supported claims |

## Pipeline and research choices

The implementation uses LangChain 1.x with its maintained Hugging Face and Chroma
integrations. BGE-M3 is the frozen embedding model and Chroma is the persistent
vector store. The candidate and evaluator are open models running locally on an A40;
no OpenAI or other external model API receives benchmark data.

The Week 3 reading review influenced four concrete choices:

- metadata is stored and applied as a hard eligibility gate before semantic ranking;
- small child chunks retrieve evidence while governed parent sections supply context;
- atomic facts link chunks to benchmark scoring points;
- context is kept concise and a no-evidence path prevents forced answers.

Automatic RAGAS-style component metrics and the local Judge are retained as
uncalibrated diagnostics. A separately disclosed AI qualitative calibration
reviews the answer content, atomic-point coverage, forbidden claims and primary
failure causes.

The three-model Week 2 replay is not treated as a validated leaderboard. Its
Prometheus Judge failed the frozen calibration and its former held-out scenarios
were already inspected. This limitation is recorded here because Week 3 consumed
the Week 2 outputs, but it is not counted as a RAG defect. A fresh model-blind
test set and independent adjudication would be needed before making a validated
comparative model-quality claim; they are outside the reference scope and are
not scheduled.

## Official-public baseline

Retrieval was exact on this small controlled set: document recall@k `1.0000`,
evidence-fact recall@k `1.0000`, hit@k `12/12`, MRR `1.0000`, metadata leakage
`0`, and mean latency `103.049 ms`.

Llama completed all 24 base/RAG generations. Base mean generation latency was
`2143.912 ms`; RAG mean latency was `3658.997 ms`. Mean output lengths were
`60.583` and `106.583` tokens respectively.

| Provisional metric | Base | RAG | Valid rows |
|---|---:|---:|---:|
| Answer relevance | 0.343704 | 0.573415 | 12/12 each |
| Faithfulness | n/a | 0.818182 | 11/12 |
| Context relevance | n/a | 0.479167 | 12/12 |
| Context recall | n/a | 0.708333 | 12/12 |
| Context precision | n/a | 0.590278 | 12/12 |

One faithfulness result was `NaN` with no reason and is reported as invalid rather
than zero. This exposed the need to report finite means together with coverage.

## Iteration history

| Version | Single change | Observed result | Decision |
|---|---|---|---|
| v0.1.0 | TF-IDF retrieval skeleton | Trace contract worked but retrieval was not the target architecture | Deprecate |
| v0.2.x | Synthetic BGE-M3/Chroma/Llama/Mistral smoke | End-to-end local path worked | Keep only as infrastructure evidence |
| v0.3.0 | Official pages, metadata gate and child-parent retrieval | Retrieval 12/12; RAG answer relevance exceeded base by 0.229711 | Freeze baseline |
| v0.3.0 repair | Reporter/runtime fixes only | Candidate rows unchanged; scorer eventually completed 24/24 | Preserve failures and repair logs |
| v0.3.1 | Epistemic-polarity prompt boundary only | Answer relevance 0.580729; faithfulness 0.930556 (12/12 finite); known uncertainty failures removed | Retain final stable version |
| v0.3.2 | Soft completeness boundary only | `SENPAI-005` stayed at approximately 2/5 weighted-point coverage | Early stop |
| v0.3.3 | Explicit compound checklist only | `SENPAI-005` restored to 5/5, but `FARI-005` returned to a definite unsupported negative | Early stop |
| v0.3.4 | Final epistemic audit only | Answer relevance 0.536665 and faithfulness 0.834402; both materially below v0.3.1 | Reject after full score |
| v0.3.5 | Question-scope guard only | `SENPAI-005` fell to 4/5 and `SENPAI-006` reintroduced absence-to-negative | Early stop; end tuning on inspected set |

Version 0.3.1 inherited every retrieved context byte-for-byte from v0.3.0. It did
not rerun embeddings or retrieval. Mean generation latency was `3870.621 ms`,
median `2954.347 ms`, and mean output length `104.667` tokens.

Versions 0.3.2 through 0.3.5 also inherited all 12 contexts. Their generation
means were respectively `4671.908`, `6261.136`, `6110.727`, and `4546.455 ms`;
mean output lengths were `124.583`, `161.500`, `167.500`, and `124.750` tokens.
The inspected regression items are explicitly treated as development cases, not
blind-test evidence.

The v0.3.4 full score had 12/12 finite rows. Relative to v0.3.1, answer relevance
changed by `-0.044064`, faithfulness by `-0.096154`, context relevance by `0`,
context recall by `-0.041666`, and context precision by `+0.041666`. The registered
keep rule therefore rejected it. Version v0.3.5 then failed its qualitative gate
before scoring. Further prompt tuning on the repeatedly inspected 12-item set was
stopped to avoid overfitting; the completeness-versus-concision trade-off moves to
a separately frozen blind set.

## Runtime lessons

The slow persistent volume made vLLM readiness much longer than the first health
window. A listening port was not sufficient evidence that the model was ready.
The final process therefore used bounded HTTP probes, an independent process group,
exact PID/command records and a scorer gate requiring loopback HTTP 200.

An early process inspection incorrectly classified a still-loading server as exited.
The later process tree proved it was alive. That observation was corrected in the
ledger as `E-REPORT-BUG`; a duplicate watcher was terminated before it could double
load the GPU. No candidate output was regenerated or overwritten during the repairs.

## Final three-model blind confirmation

The v0.5.0 run froze one detailed grounded prompt and one shared retrieval trace
for FLAN-T5-base, Mistral-7B-Instruct-v0.2, and Llama-3.1-8B-Instruct. All 48
generations and all 48 local RAGAS rows completed. The first Base item was
inspected during adapter qualification and is retained as preflight evidence,
so the primary aggregate uses the other seven model-blind question pairs.

| Model | Base relevance | RAG relevance | Change | RAG faithfulness | Judge boundary |
|---|---:|---:|---:|---:|---|
| FLAN-T5-base | 0.223139 | 0.410193 | +0.187054 | 0.513889 (6/7) | Independent, but output contract failed |
| Mistral-7B-Instruct-v0.2 | 0.346790 | 0.465645 | +0.118855 | 0.880952 | Non-independent self-judge |
| Llama-3.1-8B-Instruct | 0.290288 | 0.538142 | +0.247854 | 0.821429 | Independent local Mistral Judge |

The shared-input audit passed. FLAN produced short fragments or question echoes
and no chunk identifiers. Mistral mentioned an eligible chunk on 5/7 RAG rows
and Llama on 6/7, but each used the exact `[chunk_id]` format on only 1/7 rows.
Those are generation-format failures rather than retrieval failures.

## Retrieval ablation and source isolation

The 18 registered retrieval variants completed. Top-k 1 reduced evidence-fact
recall to `0.7000`; the reranker raised it to `0.7417` at added latency. Top-k
3 and 5 both reached `1.0000`, so top-k 3 without reranking is retained for this
smoke corpus. Chunk sizes 256, 512, and 1024 were identical because all source
sections were already shorter than 256 tokens. Perfect document recall therefore
remains a small-corpus limitation.

The June internship data source material entered only a separate private
collection. Its 6-question retrieval smoke reached 6/6 hits with zero metadata
leakage, and Llama completed all 12 private Base/RAG rows. No private content or
raw output was copied into the public repository.

## Comparison with the TalkMeUp RAG experience

The TalkMeUp experience established the practical LangChain pattern of loading
documents, embedding them, retrieving context and passing that context to a
generator. Week 3 used the same broad technology pattern but treated it as an
evaluation target rather than only an application feature. The main additions
were frozen base/RAG pairs, source and access metadata gates, fact-level retrieval
targets, candidate-blind inputs, component metrics, failure codes and immutable
run evidence.

This changed what “working RAG” meant. A response could sound useful while still
omitting a supported requirement, strengthening an uncertain statement or using
irrelevant context. Conversely, perfect document retrieval did not guarantee a
complete answer. The Week 3 result therefore reinforces the TalkMeUp implementation
lesson while adding a more disciplined conclusion: retrieval, grounded generation
and evaluation validity must be measured separately.

## Reproducibility

- Candidate seed `42`, greedy decoding, BF16.
- Llama revision `0e9e39f249a16976918f6564b8830bc894c89659`.
- BGE-M3 revision `5617a9f61b028005a4858fdac845db406aefb181`.
- Mistral Judge revision `63a8b081895390a26e140280378bc85ec8bce07a`.
- RunPod NVIDIA A40 48 GB; persistent private Chroma and run directories.
- Local contract suite: `53/53` tests passed, including the original 50
  pipeline/analysis contracts and three AI-calibration artifact contracts.
- Python syntax compile: `32/32` Week 3 Python files passed; this is a
  parser/byte-code check, not a claim of runtime or answer-quality correctness.
- Raw run evidence includes exact commands, UTC times, inputs, outputs, retrieval
  traces, per-row latency, GPU snapshots, hashes, Judge outputs and invalid reasons.

## AI calibration and representativeness

The internship reference requires separate Faithfulness, Relevance and Coverage
metrics and a Base-vs-RAG comparison; it does not prescribe a numerical RAG
usability threshold. OpenAI Codex/ChatGPT completed a disclosed AI qualitative
calibration of all eight frozen Llama RAG answers without viewing per-row
automatic scores first. Mean answer relevance was `4.375/5`, weighted
required-point coverage was `0.872619`, 30/31 claims were supported, and no
forbidden claim was identified. These values complement rather than replace the
RAGAS diagnostics because their scales and sample scopes differ.

The automatic values look low because Base has no product knowledge, FLAN often
emits short fragments or question echoes, and Llama/Mistral answers contain
multiple sub-answers, qualifications, and citations that can reduce a single
embedding-similarity score. The local Mistral Judge is uncalibrated, and the
primary aggregate contains only seven uninspected questions. Therefore
`0.538142` is a diagnostic signal rather than a usability percentage; the AI
content review of `4.375/5` provides complementary evidence.

The eight-question public blind set is suitable for pipeline smoke and
RAG-helpfulness evidence, not production usability. It includes four Fari, four
Senpai, five compound and four severity-5 cases, but the eligible corpus per
platform is so small that top-k 4 retrieves all or nearly all content. The
representativeness audit recommends a new 24–32-question, 100+-chunk frozen set
with noisy queries, distractors, source conflicts, abstention, multi-turn and
prompt-injection cases. The existing Week 3 set remains unchanged.

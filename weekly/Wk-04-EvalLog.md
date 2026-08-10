# Week 4 Evaluation Log — Robustness, Multimodal, and System Cost

**Week ending:** 2026-08-04  
**Phase:** B  
**Seed:** 42  
**Compute:** RunPod NVIDIA A40 48 GB, batch size 1  
**Claim boundary:** controlled surrogate evaluation; no deployed-product or universal-hardware claims

## What was evaluated

Week 4 added three controlled evaluation paths to the Phase B framework:

- **Semantic robustness:** 35 scenarios × original plus three meaning-preserving paraphrases × FLAN-T5-base, Mistral-7B-Instruct-v0.2, and Llama-3.1-8B-Instruct = 420 generations.
- **Masked-input robustness:** 14 Aido Rover/Sentinel Prime AI scenarios at 0%, 20%, 40%, and 60% nested evidence masking. The unmasked original is reused, so this adds 126 generations across the three models.
- **Multimodal robustness:** 10 Aido Rover and 10 Sentinel Prime AI public-image proxies under clean, deterministic Gaussian-noise, and brightness-0.60 conditions × Idefics2 and LLaVA = 120 matched generations.
- **RAG system regression:** 40 expanded public questions × Base/RAG with Llama = 80 requests with explicit retrieval and generation stage timings.

The total measured candidate workload is 746 requests. AI-assisted row scoring is run separately from candidate generation and does not contaminate candidate latency.

## Frozen method

All candidate models use immutable Hugging Face revisions, seed 42, deterministic decoding, batch size 1, and a shared semantic prompt. Paraphrase inputs pass numeric, negation, protected-entity, sentence-membership, uniqueness, and AI-assisted semantic-review gates. Masked evidence groups are manually specified and selected by a deterministic hash order so the 20% mask is a subset of 40%, and 40% is a subset of 60%.

The VLM test standardizes every clean source to 768×768 RGB. Each perturbed condition starts from the same clean pixel array and changes exactly one factor. Idefics2 and LLaVA receive the same 60 processed images, prompt, seed, output limit, rubric, and local Llama Judge; only the native VLM architecture and processor/chat template change. File and processed-pixel hashes are verified before inference. Public image attribution, original landing URLs, authors, licences, and selection rationales are retained.

System measurement records prompt/output tokens; preprocessing, TTFT, generation, decode, and question-to-response latency; RAG query embedding, metadata filtering, vector search, reranking, context assembly, and retrieval total; RSS, system RAM, PyTorch allocated/reserved memory, device-wide GPU memory, utilization, and power. Cold model load is separate from warm requests. Aggregates include count, missing/error count, mean, standard deviation, p50, p90, p95, and maximum.

## What was found

### Text robustness

Diagnostic semantic consistency was FLAN 0.914; Mistral 0.857; Llama 0.857. FLAN was the highest, but stable pass/fail and the review queue are interpreted alongside the fraction; the uncalibrated cross-model Judge prevents a human-equivalent accuracy claim.

The mechanism-oriented interpretation separates two cases. A pass/fail flip under a meaning-preserving paraphrase indicates sensitivity to surface form. A stable fail indicates the decision policy itself is inadequate even though the model is consistent. Reporting only the overall consistency fraction would hide the second case.

### Masked-input dependence

At 60% evidence removal, mean Task Accuracy declined by FLAN 0.357 points; Mistral 0.000 points; Llama 0.286 points. Severity-5 failures, original-to-mask flips, and non-monotonic transitions remain explicit review strata rather than being averaged away.

The nested masking design supports a direct mechanism claim: any systematic decline is attributable to removing ranked evidence groups rather than changing multiple unrelated prompt properties. Non-monotonic “improvements” are review flags, not evidence that missing information helps.

### Multimodal robustness

Idefics2 clean mean was 4.900/5 and LLaVA was 4.800/5; the matched clean rows produced 19 ties and one LLaVA loss. Idefics2 noise/brightness drops were 0.100/0.150, with 0.950/0.950 decision consistency. LLaVA noise/brightness drops were -0.050/0.100, with 0.950/0.900 consistency. The negative noise drop is treated as diagnostic Judge variation, not evidence that corruption improves perception.

Across all 60 matched image requests, LLaVA reduced median end-to-end latency from 6.31 to 4.39 seconds and device-wide peak memory from 18.23 to 14.15 GiB. This is a controlled architecture comparison, not a deployed sensor-fusion evaluation.

### Latency and resources

Expanded RAG warm question-to-response p50/p95 was 8402.7 / 14230.4 ms, including retrieval p50/p95 of 408.9 / 534.7 ms. Separate static records retain model revision, precision, checkpoint bytes, load time, hardware, and load peaks.

The most important systems finding is that cold load and warm inference answer different questions. The migrated network volume made some checkpoint loads slow, but that cost is not mixed into the per-question p50/p95. RAG end-to-end latency is also decomposed so retrieval overhead is not attributed to Llama generation.

## Iteration and incident record

| Iteration | Observation | Change | Verification |
|---|---|---|---|
| W04-0 | Reference requirements and supervisor cost additions were spread across separate notes. | Froze one performance schema and separate text/VLM run configs. | Contract and resource-monitor unit tests. |
| W04-1 | Initial paraphrase/mask bank needed invariant and equivalence gates. | Added numeric, negation, entity, sentence, uniqueness, nested-mask, and review records. | Frozen input manifest and tests. |
| W04-2 | Candidate public images required reproducible source/licence evidence. | Selected Open Images validation items and retained attribution plus source hashes. | 20/20 attribution rows and 60/60 input contracts. |
| W04-3 | RunPod GPU became unavailable and the pod was migrated. | Revalidated A40, persistent models, venvs, exact revisions, and frozen hashes. | All validate-only checks passed before inference. |
| W04-4 | Idefics2 Xet download failed; HTTP retry hit persistent-volume quota. | Removed only the incomplete cache, used the pod system disk, disabled Xet, and redownloaded the pinned revision. | Seven shards and required processor/tokenizer files validated; canary passed. |
| W04-5 | The same JPEG file decoded to a different RGB pixel hash across hosts. | Refroze standardized images as lossless PNG and regenerated input/config hashes. | 50/50 local Week 4 tests passed; RunPod validated 60/60 rows. |
| W04-6 | Candidate runs needed comparable cost evidence. | Kept downloads and Judges separate; measured each candidate sequentially. | Completed run manifests, sessions, events, and request traces. |
| W04-7 | Earlier Judge calibration remained below gate. | Used cross-model AI-assisted scoring only as diagnostic and retained parse/evidence/review fields. | Aggregate files carry `diagnostic_ai_assisted_not_calibrated`. |
| W04-8 | The frozen mask rows use the canonical family name `masked_input_robustness`, while the first aggregate implementation recognized only a shorter test alias. | Made the analyzer accept the canonical name plus the legacy alias and added a regression test before aggregation. | Canonical masked rows are covered without changing frozen inputs or candidate outputs. |
| W04-9 | The first 19 Mistral Judge rows were 0/19 parsed: responses were either truncated or used a boolean where the schema required an array. | Stopped the run, preserved its rows/log, added an exact compact JSON schema with explicit array types and a short-rationale limit, strengthened type validation, and restarted from a new output file. | Retry preflight reached 17/17 parsed before the full run continued. |
| W04-10 | The completed retry still omitted optional evidence arrays in some otherwise valid JSON rows; the single targeted re-Judge again returned the out-of-vocabulary alias `omission`. | Preserved both raw iterations, normalized only missing optional arrays, and mapped that observed alias to the controlled `partial` code with an explicit row-level repair action. | Candidate outputs stayed immutable; repair and schema regression coverage bring the final Week 4 suite to 56/56. |
| W04-11 | Final evidence needed a clean public/private boundary and a cost-safe shutdown. | Archived all raw runs privately, copied only aggregate evidence into the submission tree, executed both notebooks, rendered and inspected all eight slides, and stopped the GPU pod while retaining its persistent volume. | The 3.4 MB private archive matched SHA-256 `99992b6af9015f3214dc01d7216589810d2f5229c27c95a1b1911c89b549302`; 56/56 tests and PowerPoint overflow QA passed; RunPod reports compute and container storage as not running. |
| W04-12 | Supervisor review required a second reference-listed VLM under the mature pipeline. | Added pinned LLaVA-1.5-7B with a native Transformers adapter while freezing images, prompts, seed, rubric, Judge, and generation policy. | 60/60 generations and 60/60 parsed Judge rows; Judge config hash exactly matches Idefics2. |
| W04-13 | Transformers 5.14 returned processor image size as a non-JSON `SizeDict`; the first LLaVA canary failed while writing session metadata. | Preserved the failed canary, normalized library metadata to JSON-safe values, added a regression test, and ran a new canary/full run with runner v0.2.1. | 65/65 local Week 4 tests pass; final candidate rows retain the repaired runner version. |
| W04-14 | The earlier midpoint files could be mistaken for a Week 4-only review, and the added comparison needed final submission QA. | Replaced the old naming with a Weeks 1–4 Phase A–B report/deck/rubric, executed the two-VLM notebook, rendered and inspected every slide, and removed obsolete Week 4-only midpoint artifacts. | Eight slides pass automated canvas-overflow QA; current aggregate workload is 746 rows; the active RunPod was stopped after evidence recovery and now shows only persistent-volume cost. |

## Evaluation issue that mattered most

The hardest issue was not GPU compatibility; it was distinguishing reproducible evidence from an apparently successful run. The JPEG test is illustrative: the file hash matched, so a normal pipeline would have proceeded, but the exact decoded pixels differed across hosts. Converting the frozen clean artifacts to PNG made the one-variable perturbation claim defensible. The same discipline applies to quality: a structured Judge output is not a validated label simply because it parses.

## Limitations and next actions

- Complete independent human adjudication on a stratified set of paraphrase flips, severity-5 cases, and VLM decision flips.
- Add harder transformations that preserve semantics but change syntax more substantially.
- Add an independently adjudicated VLM subset before treating the small diagnostic score differences as an architecture-selection result.
- Add temporal sequences and product-representative sensor data when governance permits.
- Repeat representative conditions on a second hardware configuration before setting latency budgets.
- Expand the RAG stress set with no-evidence, conflicting-source, stale-source, multilingual, and injection cases; keep private and public collections separate.

## Deliverable status

- `phase_b_evaluation/W04_Robustness_Eval.ipynb` — completed aggregate notebook.
- `phase_b_evaluation/W04_Multimodal_Eval.ipynb` — completed aggregate notebook.
- `phase_b_evaluation/Phase_AB_Midpoint_Review_Deck.pptx` — 8-slide Weeks 1–4 review deck with speaker notes.
- `phase_b_evaluation/Phase_AB_Midpoint_Evaluation_Rubric.md` — intern self-assessment prepared; joint scoring and signatures require the review meeting.
- `phase_b_evaluation/Phase_AB_Midpoint_Report.md` — concise Weeks 1–4 consolidated report.
- `phase_b_evaluation/W04_Evaluation_Report.md` — method, results, system cost, failures, and requirement audit.
- `phase_b_evaluation/W04_Submission_Index.md` — navigation and verification map.

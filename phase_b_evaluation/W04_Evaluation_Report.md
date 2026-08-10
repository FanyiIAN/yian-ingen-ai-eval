# Week 4 Robustness, Multimodal, and System-Cost Evaluation

**Phase:** B — systematic evaluation, RAG, and multimodal assessment  
**Evaluation dates:** 2026-08-04 to 2026-08-08
**Random seed:** 42  
**Compute boundary:** one RunPod NVIDIA A40 48 GB pod; batch size 1; deterministic decoding  
**Claim boundary:** public/synthetic surrogate evaluation, not deployed InGen product performance

## 1. Executive summary

Week 4 extends the frozen 35-scenario benchmark in two directions: semantic and masked-input robustness for three text models, and a controlled public-image robustness comparison across two open-source VLM architectures. It also adds stage-level latency and resource evidence to the expanded Week 3 RAG pipeline.

The completed candidate workload contains 746 measured requests: 420 semantic-robustness generations, 126 additional masked-input generations, 120 matched VLM generations, and 80 expanded RAG Base/RAG generations. Model-load cost is reported separately from warm steady-state latency. All automatic quality scores are explicitly diagnostic because the local Judge did not pass the earlier calibration gate.

**Result summary (filled from the frozen aggregate files):**

- Three-model semantic robustness: `FLAN 0.914; Mistral 0.857; Llama 0.857`.
- Masked-input degradation at 60% evidence removal: `FLAN 0.357 points; Mistral 0.000 points; Llama 0.286 points`.
- Controlled VLM result: Idefics2 and LLaVA tied on `19/20` clean scenarios; Idefics2 led by `0.100/5` on the clean mean, while LLaVA used `4.08 GiB` less peak GPU memory and had a `30.4%` lower all-condition median end-to-end latency.
- System-cost result: `warm RAG question-to-response p50/p95 8402.7 / 14230.4 ms, including retrieval p50/p95 408.9 / 534.7 ms`.

## 2. Frozen evaluation design

### 2.1 Semantic paraphrase robustness

Each of the 35 Week 2 scenarios has four candidate-visible versions:

1. original;
2. conservative synonym substitution;
3. sentence reordering without sentence deletion;
4. outer-instruction tone shift.

Numeric tokens, negations, protected platform entities, and sentence membership are checked before inference. The semantic-equivalence record is AI-assisted inspection, not an independent human review. Every model receives the same 140 ordered inputs, semantic prompt, seed, deterministic decoding settings, and maximum token limits.

The primary metric is the fraction of scenarios whose four versions receive the same pass/fail decision. Stable pass, stable fail, and original-to-paraphrase flips are reported separately because consistency alone can reward a model that fails consistently.

### 2.2 Masked-input robustness

The 14 Sentinel Prime AI and Aido Rover text scenarios use five manually defined evidence groups. A fixed SHA-256 ranking makes the 20%, 40%, and 60% masks nested: evidence removed at a lower level remains removed at every higher level. The unmasked original is reused from the semantic run, so the test adds 42 inputs per model rather than duplicating the 14 originals.

This is a controlled proxy for missing upstream evidence, not an internal mask on model embeddings and not primarily a test of vague user phrasing. The text scenarios serialize sensor, alert, policy, and operating-state evidence that a deployed decision layer could receive after sensor dropout, packet loss, field omission, redaction, or upstream extraction failure. Only evidence or constraint spans are replaced by `[MISSING]`; the response instruction remains intact. The nominal 20%, 40%, and 60% levels remove one, two, and three of five evidence groups respectively, so they represent evidence-group completeness rather than an exact percentage of all tokenizer tokens.

Reported metrics are mean Task Accuracy, severity-weighted mean, pass rate, original-to-mask flip rate, and severity-5 failure count at 0%, 20%, 40%, and 60% masking.

### 2.3 Multimodal robustness

The VLM test contains 20 public Open Images scenarios: 10 Aido Rover navigation proxies and 10 Sentinel Prime AI surveillance proxies. Each clean 768×768 RGB PNG is evaluated under three one-variable conditions:

- clean pixels;
- deterministic Gaussian noise with standard deviation 0.08 of the pixel range;
- brightness factor 0.60.

The first draft used JPEG clean artifacts. A cross-host unit test showed that different libjpeg builds could decode the same file into slightly different pixels, invalidating a pixel-level reproducibility assertion. The benchmark was refrozen as lossless PNG; all 60 exact processed-pixel hashes then reproduced locally and on RunPod.

Idefics2 and LLaVA-1.5-7B are scored on scene interpretation (0–2), decision recommendation (0–2), and uncertainty/claim control (0–1). Both receive the same 60 processed images, prompt text, seed, maximum output, rubric, and Llama Judge; only the native VLM architecture and processor/chat template change. Clean-to-perturbed score drop, decision consistency, forbidden-claim flags, latency, throughput, and GPU memory are retained.

### 2.4 Expanded RAG performance path

The Week 3 expanded public collection remains separate from private June internship material. The performance regression uses the 331-unit public collection and 40 frozen public questions, generating one Base and one RAG response per question with Llama-3.1-8B-Instruct. It records query embedding, metadata filter, vector search, reranking, context assembly, retrieval total, prompt building, TTFT, generation, and full question-to-response latency. These 80 rows are a system-cost regression, not a new pooled Week 4 quality benchmark.

## 3. Models and immutable versions

| Role | Model | Immutable revision | Precision |
|---|---|---|---|
| Text candidate | `google/flan-t5-base` | `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2` | float32 |
| Text candidate / cross-Judge | `mistralai/Mistral-7B-Instruct-v0.2` | `63a8b081895390a26e140280378bc85ec8bce07a` | bfloat16 |
| Text candidate / RAG / cross-Judge | `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f249a16976918f6564b8830bc894c89659` | bfloat16 |
| VLM candidate | `HuggingFaceM4/idefics2-8b-chatty` | `8e65868b394317b973bd61db3b08e6478ebeedbf` | bfloat16 |
| VLM candidate | `llava-hf/llava-1.5-7b-hf` | `b234b804b114d9e37bb655e11cbbb5f5e971b7a9` | float16 |
| RAG embedding | `BAAI/bge-m3` | frozen local snapshot recorded in the RAG manifest | model default |
| RAG reranker | `BAAI/bge-reranker-v2-m3` | frozen local snapshot recorded in the RAG manifest | model default |

The text quality review avoids direct self-scoring where practical: Mistral judges FLAN and Llama, while Llama judges Mistral. The same local Llama checkpoint and scorer configuration hash judge both VLMs. These structured scores remain diagnostic and are not represented as human ground truth.

## 4. Text robustness results

| Model | Parsed rows | Semantic robustness | Stable pass | Stable fail | Task drop at 60% mask | Review flags |
|---|---:|---:|---:|---:|---:|---:|
| FLAN-T5-base | `182/182` | `0.914` | `7` | `25` | `0.357` | `5` |
| Mistral-7B-Instruct-v0.2 | `182/182` | `0.857` | `30` | `0` | `0.000` | `9` |
| Llama-3.1-8B-Instruct | `182/182` | `0.857` | `26` | `4` | `0.286` | `12` |

FLAN has the highest diagnostic semantic consistency at 0.914. This does not establish overall model superiority: the metric rewards stable failures, the cross-model Judge is uncalibrated, and every flagged flip or stable failure still requires direct review.

The Sentinel operational implication is evaluated from the failure pattern, not only the percentage: a paraphrase-triggered pass/fail flip means a semantically unchanged alert narrative can alter the recommended escalation policy. A stable fail is also operationally important and is not hidden inside a high consistency score.

## 5. Multimodal results

| VLM | Condition | Parsed | Mean /5 | Scene /2 | Decision /2 | Claim control /1 | Acceptable decision | Forbidden claim |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Idefics2 | Clean | `20/20` | `4.900` | `1.950` | `1.950` | `1.000` | `0.950` | `0.000` |
| Idefics2 | Gaussian noise | `20/20` | `4.800` | `1.900` | `1.900` | `1.000` | `0.900` | `0.050` |
| Idefics2 | Brightness 0.60 | `20/20` | `4.750` | `1.850` | `1.900` | `1.000` | `0.900` | `0.000` |
| LLaVA-1.5-7B | Clean | `20/20` | `4.800` | `1.900` | `1.900` | `1.000` | `0.900` | `0.000` |
| LLaVA-1.5-7B | Gaussian noise | `20/20` | `4.850` | `1.900` | `1.950` | `1.000` | `0.950` | `0.000` |
| LLaVA-1.5-7B | Brightness 0.60 | `20/20` | `4.700` | `1.850` | `1.900` | `0.950` | `0.900` | `0.000` |

| VLM | Noise drop / consistency | Brightness drop / consistency | Clean wins / ties / losses vs Idefics2 |
|---|---:|---:|---:|
| Idefics2 | `0.100 / 0.950` | `0.150 / 0.950` | baseline |
| LLaVA-1.5-7B | `-0.050 / 0.950` | `0.100 / 0.900` | `0 / 19 / 1` |

The clean difference is one rubric point across the 20 matched scenarios: Idefics2 averages 4.900/5 and LLaVA 4.800/5, with 19 ties and one LLaVA loss. LLaVA's negative noise drop means the diagnostic Judge scored noisy images 0.050 points higher on average than their clean counterparts; this is treated as sampling/Judge variation and a review signal, not evidence that noise improves perception. Brightness causes a small mean decline for both architectures, and LLaVA has one additional decision flip.

This is a two-architecture public-image proxy. It supports a controlled architecture comparison under this frozen set, but it does not test calibrated cameras, temporal sensor fusion, product alert thresholds, person identification, or an executed navigation controller.

## 6. Latency and resource results

The performance summary keeps cold model load separate from warm requests. Per-request groups retain count, missing/error count, mean, standard deviation, p50, p90, p95, and maximum for prompt/output/total tokens, stage timing, throughput, RSS, system RAM, PyTorch allocated/reserved memory, device-wide GPU memory, GPU utilization, and GPU power.

| Path | Representative group | n | End-to-end p50 / p95 | TTFT p50 / p95 | GPU-memory peak | Notes |
|---|---|---:|---:|---:|---:|---|
| FLAN text | semantic robustness | `140` | `647.1 / 887.3 ms` | `88.7 / 103.3 ms` | `1393.0 MiB` | direct seq2seq prompt |
| Mistral text | semantic robustness | `140` | `2135.8 / 4565.3 ms` | `98.2 / 112.9 ms` | `14371.0 MiB` | native chat template |
| Llama text | semantic robustness | `140` | `2023.0 / 2992.5 ms` | `106.7 / 116.3 ms` | `15881.0 MiB` | native chat template |
| Idefics2 | clean image | `20` | `6717.2 / 6949.2 ms` | `757.4 / 860.8 ms` | `18670.0 MiB` | 768×768 RGB input |
| LLaVA-1.5-7B | clean image | `20` | `4439.0 / 5632.6 ms` | `402.4 / 491.3 ms` | `14493.0 MiB` | same 768×768 inputs |
| Expanded RAG | RAG condition | `40` | `8402.7 / 14230.4 ms` | `733.6 / 930.1 ms` | `21299.0 MiB` | retrieval stages included |

Across all 60 matched image requests, LLaVA's median end-to-end latency is `4386.5 ms`, versus `6305.4 ms` for Idefics2: a `1918.9 ms` or `30.4%` reduction. LLaVA also has lower median TTFT (`415.6` versus `768.8 ms`) and a lower device-wide peak (`14.15` versus `18.23 GiB`). Its median output-token throughput is lower (`22.68` versus `27.87 tokens/s`), so the end-to-end advantage reflects the complete architecture, prompt-tokenization, and output-length path rather than a universally faster decoder.

The representative rows above are different workloads, so the `2023.0 ms` Llama semantic-robustness row is **not** a valid baseline to subtract from the RAG row. The controlled RAG latency comparison uses the same 40 questions and the same loaded Llama process:

| Condition | Prompt tokens p50 | Output tokens p50 | Retrieval p50 | `model.generate` p50 | Question-to-response p50 / p95 |
|---|---:|---:|---:|---:|---:|
| Base | `244` | `47` | N/A | `1591.6 ms` | `1594.0 / 9450.6 ms` |
| RAG | `1866` | `220.5` | `408.9 ms` | `8038.4 ms` | `8402.7 / 14230.4 ms` |

The `8.40 s` RAG median is therefore not `0.41 s retrieval + 2.02 s Llama + 5 s unexplained overhead`. On the additive **mean** trace, retrieval takes `416.4 ms` (`4.97%`), prompt preprocessing takes `6.3 ms`, `model.generate` takes `7962.1 ms` (`94.95%`), decode takes `0.5 ms`, and the measured total is `8386.0 ms`. `model.generate` includes prompt prefill and autoregressive answer generation; TTFT is a checkpoint inside that interval, not another stage to add. Stage p50 values also do not add exactly because each median can come from a different request.

The main latency increase is the longer generated answer: median output rises from `47` to `220.5` tokens. At the RAG median throughput of `27.7 output tokens/s`, the additional `173.5` tokens account for about `6.3 s`. The longer prompt (`244` to `1866` tokens) also increases prefill/TTFT, but it is the secondary contributor in this run. This benchmark measures the current detailed RAG prompt and answer policy; it does not claim that vector search itself takes eight seconds.

GPU-memory figures also require workload and lifecycle labels:

| Measured workload | Resident/load state | Request-time increase | Device-wide peak | Interpretation |
|---|---:|---:|---:|---|
| Idefics2 clean VLM | `16339 MiB` (`15.96 GiB`) model-load peak | `2331 MiB` (`2.28 GiB`) | `18670 MiB` (`18.23 GiB`) | VLM only; no RAG components are loaded. |
| LLaVA clean VLM | `13789 MiB` (`13.47 GiB`) model-load peak | `704 MiB` (`0.69 GiB`) | `14493 MiB` (`14.15 GiB`) | VLM only; same image bank and no RAG components. |
| Llama + RAG process | `4725 MiB` (`4.61 GiB`) before Llama; Llama then adds `15334 MiB` (`14.97 GiB`) | up to `1240 MiB` (`1.21 GiB`) above the loaded `20059 MiB` state | `21299 MiB` (`20.80 GiB`) | The pre-Llama state contains BGE-M3, the BGE reranker, and CUDA/runtime overhead. |

The persistent Chroma collection is `5,990,564 bytes` (about `5.7 MiB`) on CPU/disk and is not the source of multi-GiB GPU use. The retrieval-side GPU cost comes from the two Transformer retrieval models, primarily the BGE-M3 embedding model plus the reranker. They were loaded before the Llama snapshot but were not snapshotted separately, so the measured `4.61 GiB` retrieval-stack total cannot be split reliably between those two models. The RAG request peak contains `20224.9 MiB` of live PyTorch allocations, about `721.1 MiB` of reserved allocator cache, and about `353.0 MiB` of CUDA/non-PyTorch device use.

For orientation, the separate standalone Llama semantic run peaked at `15.51 GiB`, while Llama + RAG peaked at `20.80 GiB`, an apparent increase of `5.29 GiB`. Most of that difference is consistent with the `4.61 GiB` retrieval stack; the remainder reflects the longer-context request, allocator state, and small runtime differences. This is descriptive rather than a clean component ablation: the Base rows inside the RAG process were recorded after the embedding model and reranker were already loaded, and interleaved Base/RAG requests share the CUDA allocator. They are a fair latency comparison but **not** a pure Llama-versus-RAG memory comparison.

Cold model loading remains separate from warm requests because storage and cache state are environment-specific. The current environment manifest did not capture the NVIDIA driver string, so the public JSON stores `null` with an explicit reason instead of zero. CUDA, PyTorch, Transformers, GPU model/memory, host RAM, checkpoint size, and model-load resource peaks are retained.

## 7. Execution issues and controlled repairs

| Issue | Evidence | Repair | Validity effect |
|---|---|---|---|
| RunPod GPU loss and pod migration | Original GPU became unavailable; persistent volume moved to a new A40 pod. | Revalidated GPU, model snapshots, venvs, hashes, and frozen inputs before inference. | Hardware remains A40, but cold storage latency is migration-specific. |
| Idefics2 download failed on the persistent volume | Xet writer error followed by disk-quota error; incomplete directory failed validation. | Removed only the incomplete 4 GB cache, disabled Xet, downloaded the pinned revision to the 50 GB pod system disk, and validated all seven shards. | Download time is excluded; exact revision and checkpoint bytes retained. |
| LLaVA runner metadata was not JSON serializable | Transformers 5.14 returned processor image size as a library-specific `SizeDict`; the first canary loaded weights but failed before generation while writing session metadata. | Preserved the failed canary, normalized library metadata to ordinary JSON-safe values, added a regression test, and reran a new canary and full run with runner v0.2.1. | Frozen images, prompts, seed, revision, candidate policy, and Judge were unchanged; 60/60 final rows use the repaired runner. |
| Persistent model volume was already about 78% used | Adding another 14 GB checkpoint to the network volume risked quota pressure and would make storage-path latency inconsistent with the earlier VLM. | Downloaded the pinned three-shard LLaVA snapshot to the 50 GB Pod system disk, while keeping all logs and results on the persistent volume. | Download time is excluded; both VLM checkpoints were loaded from Pod system storage and exact checkpoint bytes are retained. |
| JPEG pixel hash differed across hosts | File SHA matched while decoded RGB SHA did not. | Refroze standardized clean artifacts as lossless PNG and regenerated all derived hashes. | Improves cross-host reproducibility without changing the selected source scenes. |
| Processor chat-template deprecation warning | Transformers reports the legacy processor-config location. | Retained the pinned snapshot and recorded the warning; no template was silently rewritten after the canary passed. | No observed generation failure; future snapshot refresh should migrate the template file. |
| Mask-family aggregate alias mismatch | Frozen rows use `masked_input_robustness`; the first analyzer recognized only the shorter unit-test alias. | Accepted the canonical family plus the legacy alias and added a canonical-name regression test before final aggregation. | Frozen inputs and model outputs were unchanged; final curves include all 42 masked rows per model. |
| Judge JSON schema mismatch | Initial Mistral scoring produced 0/19 parsed rows because some responses were truncated or replaced an expected array with a boolean. | Stopped and retained the failed iteration; added an exact compact JSON example, explicit array types, a 25-word rationale cap, and stricter validation before restarting. | Retry preflight was 17/17 parsed; no candidate output or performance measurement was rerun. |
| Judge normalization edge cases | The completed Mistral retry contained valid numeric decisions in most flagged rows but omitted optional arrays; one deterministic re-Judge still returned `omission` outside the controlled failure-code vocabulary. | Preserved both complete raw iterations, normalized only missing optional arrays to empty lists, and mapped the observed `omission` alias to the controlled `partial` code with an explicit row-level repair action. | Candidate generations remain immutable; every accepted score is schema-valid and the repair is deterministic, narrow, and test-covered. |
| Automatic Judge failed calibration | Week 2 calibration gates did not pass. | Use cross-model AI-assisted scoring only as diagnostic evidence; retain parse status, raw decisions, and review queues. | No validated human-equivalent quality claim is made. |

## 8. Validity boundaries and next work

- The 35 original scenarios and seven former held-out labels were inspected in prior weeks; this is a regression/robustness test, not a fresh blind generalization estimate.
- Paraphrase equivalence and output scoring are AI-assisted rather than independently human-adjudicated.
- With 35 semantic groups, 14 masking groups, and 20 image groups, p90/p95 and condition differences are descriptive; no production SLA or universal accuracy claim follows.
- Manually grouped text masking is a reproducible missing-evidence proxy, not a substitute for testing real sensor dropout, malformed telemetry, ASR corruption, or naturally ambiguous language. Span lengths differ, so the nominal mask percentages are exact over evidence groups rather than literal token counts.
- Natural clean-image lighting varies across Open Images. The controlled comparison is within-image: each perturbation starts from the same standardized pixels.
- Two VLM architectures are evaluated under the same frozen 60-row contract. The comparison is still limited to one public-image proxy set and one diagnostic Judge.
- Expanded RAG uses only the governed public collection. Private June material remains a separate collection and is not pooled into these metrics.
- RAG latency has a same-question Base control, but RAG memory was not measured with retrieval components unloaded in the same process. Component-level GPU-memory numbers are lifecycle deltas, not isolated per-module peaks.
- The next benchmark should add harder semantic transformations, real sensor sequences, and an independently adjudicated subset before production or model-selection claims.

## 9. Requirement coverage

| Week 4 requirement | Evidence | Status |
|---|---|---|
| 35 scenarios × original + 3 paraphrases × 3 models | Frozen 182-row input bank; three candidate event logs; robustness summary and notebook | Complete |
| Semantic consistency, stable pass/fail, failure pattern, operational implication | `W04_Robustness_Summary_v0.1.0.json`, results Markdown, report §4 | Complete, diagnostic scoring boundary |
| 20/40/60% masked-input curves for Rover and Sentinel | Frozen nested evidence groups; CSV curves; notebook plot | Complete |
| 10 Rover + 10 Sentinel public-image scenarios | Scenario YAML, 20 lossless images, attribution CSV, 60-row input bank | Complete |
| Clean/noise/brightness, one variable at a time | Pixel-hash validation and VLM run config | Complete |
| VLM scene, decision, and failure analysis | Multimodal summary, platform/condition CSV, review queue, notebook | Complete, diagnostic scoring boundary |
| Controlled multi-architecture VLM comparison | `W04_Multimodal_Architecture_Comparison_v0.2.0.*`, matched Idefics2/LLaVA runs, and comparison figures | Complete; 60/60 rows and 60/60 parsed scores per model |
| 6–8 slide Phase A–B mid-point deck | `Phase_AB_Midpoint_Review_Deck.pptx` and `Phase_AB_Midpoint_Review_Speaker_Script.md` | Complete; all eight slides were rendered, inspected, and passed automated canvas-overflow QA |
| Mid-point rubric jointly completed and signed | `Phase_AB_Midpoint_Evaluation_Rubric.md` | Intern self-assessment prepared; supervisor joint scoring/signature still requires the review meeting |
| Weekly evaluation log | `weekly/Wk-04-EvalLog.md` | Complete |
| Supervisor-added cost/latency/resource evidence | System metrics spec, row traces, static/load summary, p50/p90/p95/max tables | Complete |

## 10. Reproducibility and artifact boundary

Submission-facing notebooks read only aggregate JSON/CSV files and make no GPU, network, or Judge calls. Raw candidate prompts, outputs, request traces, AI scoring rows, and review queues remain in the private experiment archive. Public inputs, model/config revisions, seeds, image attribution, summary statistics, and code are retained in the repository.

The original Week 4 private archive was verified against SHA-256 `99992b6af9015f3214dc01d7216589810d2f5229c27c95a1b1911c89b549302`. The added LLaVA archive was downloaded and verified against SHA-256 `6aa4429c212c147e32aa99716c57e1a4731103d0b047e76f3336de56614bab48`. The RunPod compute instance was stopped after both result sets were recovered; the persistent volume was retained and is not represented as a zero-cost resource.

The final navigation and exact verification commands are listed in `W04_Submission_Index.md`.

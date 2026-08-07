# Week 4 Robustness, Multimodal, and System-Cost Evaluation

**Phase:** B — systematic evaluation, RAG, and multimodal assessment  
**Evaluation date:** 2026-08-04  
**Random seed:** 42  
**Compute boundary:** one RunPod NVIDIA A40 48 GB pod; batch size 1; deterministic decoding  
**Claim boundary:** public/synthetic surrogate evaluation, not deployed InGen product performance

## 1. Executive summary

Week 4 extends the frozen 35-scenario benchmark in two directions: semantic and masked-input robustness for three text models, and a controlled public-image robustness test for one open-source VLM. It also adds stage-level latency and resource evidence to the expanded Week 3 RAG pipeline.

The completed candidate workload contains 686 measured requests: 420 semantic-robustness generations, 126 additional masked-input generations, 60 VLM generations, and 80 expanded RAG Base/RAG generations. Model-load cost is reported separately from warm steady-state latency. All automatic quality scores are explicitly diagnostic because the local Judge did not pass the earlier calibration gate.

**Result summary (filled from the frozen aggregate files):**

- Three-model semantic robustness: `FLAN 0.914; Mistral 0.857; Llama 0.857`.
- Masked-input degradation at 60% evidence removal: `FLAN 0.357 points; Mistral 0.000 points; Llama 0.286 points`.
- Idefics2 clean/noise/brightness result: `clean mean 4.900/5; noise score drop 0.100 with decision consistency 0.950; brightness score drop 0.150 with decision consistency 0.950`.
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

Idefics2 is scored on scene interpretation (0–2), decision recommendation (0–2), and uncertainty/claim control (0–1). Clean-to-perturbed score drop, decision consistency, and forbidden-claim flags are retained.

### 2.4 Expanded RAG performance path

The Week 3 expanded public collection remains separate from private June internship material. The performance regression uses the 331-unit public collection and 40 frozen public questions, generating one Base and one RAG response per question with Llama-3.1-8B-Instruct. It records query embedding, metadata filter, vector search, reranking, context assembly, retrieval total, prompt building, TTFT, generation, and full question-to-response latency. These 80 rows are a system-cost regression, not a new pooled Week 4 quality benchmark.

## 3. Models and immutable versions

| Role | Model | Immutable revision | Precision |
|---|---|---|---|
| Text candidate | `google/flan-t5-base` | `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2` | float32 |
| Text candidate / cross-Judge | `mistralai/Mistral-7B-Instruct-v0.2` | `63a8b081895390a26e140280378bc85ec8bce07a` | bfloat16 |
| Text candidate / RAG / cross-Judge | `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f249a16976918f6564b8830bc894c89659` | bfloat16 |
| VLM candidate | `HuggingFaceM4/idefics2-8b-chatty` | `8e65868b394317b973bd61db3b08e6478ebeedbf` | bfloat16 |
| RAG embedding | `BAAI/bge-m3` | frozen local snapshot recorded in the RAG manifest | model default |
| RAG reranker | `BAAI/bge-reranker-v2-m3` | frozen local snapshot recorded in the RAG manifest | model default |

The text quality review avoids direct self-scoring where practical: Mistral judges FLAN and Llama, while Llama judges Mistral. Llama also judges the Idefics2 rows. These structured scores remain diagnostic and are not represented as human ground truth.

## 4. Text robustness results

| Model | Parsed rows | Semantic robustness | Stable pass | Stable fail | Task drop at 60% mask | Review flags |
|---|---:|---:|---:|---:|---:|---:|
| FLAN-T5-base | `182/182` | `0.914` | `7` | `25` | `0.357` | `5` |
| Mistral-7B-Instruct-v0.2 | `182/182` | `0.857` | `30` | `0` | `0.000` | `9` |
| Llama-3.1-8B-Instruct | `182/182` | `0.857` | `26` | `4` | `0.286` | `12` |

FLAN has the highest diagnostic semantic consistency at 0.914. This does not establish overall model superiority: the metric rewards stable failures, the cross-model Judge is uncalibrated, and every flagged flip or stable failure still requires direct review.

The Sentinel operational implication is evaluated from the failure pattern, not only the percentage: a paraphrase-triggered pass/fail flip means a semantically unchanged alert narrative can alter the recommended escalation policy. A stable fail is also operationally important and is not hidden inside a high consistency score.

## 5. Multimodal results

| Condition | Parsed | Mean /5 | Scene /2 | Decision /2 | Claim control /1 | Acceptable decision | Forbidden claim |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clean | `20/20` | `4.900` | `1.950` | `1.950` | `1.000` | `0.950` | `0.000` |
| Gaussian noise | `20/20` | `4.800` | `1.900` | `1.900` | `1.000` | `0.900` | `0.050` |
| Brightness 0.60 | `20/20` | `4.750` | `1.850` | `1.900` | `1.000` | `0.900` | `0.000` |

The clean mean is 4.900/5. Noise and brightness change the score by 0.100 and 0.150 points, with decision consistency of 0.950 and 0.950. These within-image differences identify perturbation sensitivity; they are not product-camera accuracy.

This is a single-VLM public-image proxy. It does not test calibrated cameras, temporal sensor fusion, product alert thresholds, person identification, or an executed navigation controller.

## 6. Latency and resource results

The performance summary keeps cold model load separate from warm requests. Per-request groups retain count, missing/error count, mean, standard deviation, p50, p90, p95, and maximum for prompt/output/total tokens, stage timing, throughput, RSS, system RAM, PyTorch allocated/reserved memory, device-wide GPU memory, GPU utilization, and GPU power.

| Path | Representative group | n | End-to-end p50 / p95 | TTFT p50 / p95 | GPU-memory peak | Notes |
|---|---|---:|---:|---:|---:|---|
| FLAN text | semantic robustness | `140` | `647.1 / 887.3 ms` | `88.7 / 103.3 ms` | `1393.0 MiB` | direct seq2seq prompt |
| Mistral text | semantic robustness | `140` | `2135.8 / 4565.3 ms` | `98.2 / 112.9 ms` | `14371.0 MiB` | native chat template |
| Llama text | semantic robustness | `140` | `2023.0 / 2992.5 ms` | `106.7 / 116.3 ms` | `15881.0 MiB` | native chat template |
| Idefics2 | clean image | `20` | `6717.2 / 6949.2 ms` | `757.4 / 860.8 ms` | `18670.0 MiB` | 768×768 RGB input |
| Expanded RAG | RAG condition | `40` | `8402.7 / 14230.4 ms` | `733.6 / 930.1 ms` | `21299.0 MiB` | retrieval stages included |

The expanded RAG warm path has question-to-response p50/p95 of 8402.7 / 14230.4 ms; retrieval contributes p50/p95 408.9 / 534.7 ms. The text, VLM, and RAG rows preserve their own TTFT, generation, memory, utilization, power, and token distributions, so cost comparisons remain configuration-specific rather than universal model claims.

RunPod storage migration made cold model loading unusually slow on the network volume; this is preserved as environment-specific evidence rather than mixed into warm inference. The current environment manifest did not capture the NVIDIA driver string, so the public JSON stores `null` with an explicit reason instead of zero. CUDA, PyTorch, Transformers, GPU model/memory, host RAM, checkpoint size, and model-load resource peaks are retained.

## 7. Execution issues and controlled repairs

| Issue | Evidence | Repair | Validity effect |
|---|---|---|---|
| RunPod GPU loss and pod migration | Original GPU became unavailable; persistent volume moved to a new A40 pod. | Revalidated GPU, model snapshots, venvs, hashes, and frozen inputs before inference. | Hardware remains A40, but cold storage latency is migration-specific. |
| Idefics2 download failed on the persistent volume | Xet writer error followed by disk-quota error; incomplete directory failed validation. | Removed only the incomplete 4 GB cache, disabled Xet, downloaded the pinned revision to the 50 GB pod system disk, and validated all seven shards. | Download time is excluded; exact revision and checkpoint bytes retained. |
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
- One VLM is evaluated. The study compares controlled input conditions, not multiple VLM architectures.
- Expanded RAG uses only the governed public collection. Private June material remains a separate collection and is not pooled into these metrics.
- The next benchmark should add harder semantic transformations, real sensor sequences, additional VLMs, and an independently adjudicated subset before model-selection claims.

## 9. Requirement coverage

| Week 4 requirement | Evidence | Status |
|---|---|---|
| 35 scenarios × original + 3 paraphrases × 3 models | Frozen 182-row input bank; three candidate event logs; robustness summary and notebook | Complete |
| Semantic consistency, stable pass/fail, failure pattern, operational implication | `W04_Robustness_Summary_v0.1.0.json`, results Markdown, report §4 | Complete, diagnostic scoring boundary |
| 20/40/60% masked-input curves for Rover and Sentinel | Frozen nested evidence groups; CSV curves; notebook plot | Complete |
| 10 Rover + 10 Sentinel public-image scenarios | Scenario YAML, 20 lossless images, attribution CSV, 60-row input bank | Complete |
| Clean/noise/brightness, one variable at a time | Pixel-hash validation and VLM run config | Complete |
| VLM scene, decision, and failure analysis | Multimodal summary, platform/condition CSV, review queue, notebook | Complete, diagnostic scoring boundary |
| 6–8 slide mid-point deck | `W04_Mid_Review_Deck.pptx` and speaker script | Complete; 8 slides, visual inspection, and overflow QA passed |
| Mid-point rubric jointly completed and signed | `W04_Midpoint_Evaluation_Rubric.md` | Intern self-assessment prepared; supervisor joint scoring/signature still requires the review meeting |
| Weekly evaluation log | `weekly/Wk-04-EvalLog.md` | Complete |
| Supervisor-added cost/latency/resource evidence | System metrics spec, row traces, static/load summary, p50/p90/p95/max tables | Complete |

## 10. Reproducibility and artifact boundary

Submission-facing notebooks read only aggregate JSON/CSV files and make no GPU, network, or Judge calls. Raw candidate prompts, outputs, request traces, AI scoring rows, and review queues remain in the private experiment archive. Public inputs, model/config revisions, seeds, image attribution, summary statistics, and code are retained in the repository.

The final private archive was downloaded and verified against SHA-256 `99992b6af9015f3214dc01d7216589810d2f5229c27c95a1b1911c89b549302` before the RunPod compute instance was stopped. The persistent volume was retained; it is not represented as a zero-cost resource.

The final navigation and exact verification commands are listed in `W04_Submission_Index.md`.

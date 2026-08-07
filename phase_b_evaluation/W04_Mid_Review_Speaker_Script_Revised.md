# Week 4 Mid-Review Speaker Script

**Target duration:** 5–8 minutes  
**Slides:** 8

## Slide 1 — Week 4 outcome

Week 4 turns the evaluation pipeline into a robustness and systems study. I tested three text models under meaning-preserving paraphrases and missing evidence, one open-source vision-language model under controlled image perturbations, and the expanded RAG path with stage-level latency and resource logging. The key boundary is that quality scores are diagnostic because the local Judge is not human-calibrated; the candidate runs and system measurements themselves are complete and reproducible.

## Slide 2 — Architecture

The architecture keeps three evidence paths separate. The text path creates semantic variants and nested missing-evidence inputs before sequential inference on FLAN, Mistral, and Llama. The vision path starts from twenty lossless standardized images and changes exactly one factor per condition before Idefics2 inference. The systems path instruments both candidate runners and a forty-question expanded RAG regression. All paths share immutable model revisions, seed 42, deterministic decoding, batch size one, frozen manifests, hashes, and private row-level traces.

## Slide 3 — Frozen design

The measured workload is 686 candidate requests. Semantic robustness uses the same four input forms per scenario and tests whether pass or fail changes even when meaning is held constant. Masking uses manually defined evidence groups and a fixed hash order, so twenty percent is a subset of forty percent and forty percent is a subset of sixty percent. Vision inputs are validated with exact processed-pixel hashes. Candidate latency and resources are captured before any Judge is loaded, and cold load remains separate from warm inference. This makes the failure mechanism and the cost boundary auditable.

## Slide 4 — Text robustness result

The bar chart shows diagnostic semantic robustness: FLAN 0.914, Mistral 0.857, and Llama 0.857. This is not an accuracy percentage. It is the share of thirty-five scenarios where all four meaning-preserving forms receive the same pass or fail decision. The cards split stable passes from stable failures, because a model can be consistently wrong. Operationally, a paraphrase-triggered flip means the same alert narrative can change escalation behavior, while a stable failure indicates a policy or reasoning weakness rather than a wording sensitivity.

## Slide 5 — Masked-input result

The line chart follows average Task Accuracy as evidence is removed in nested steps. At sixty percent masking, Llama drops 0.286 points, Mistral drops 0.000, and FLAN drops 0.357 from their own complete-input baselines. Because every level removes a superset of the previous evidence groups, a systematic decline supports an evidence-dependence interpretation. Any non-monotonic improvement is kept as a review flag, not presented as evidence that less information helps.

## Slide 6 — Multimodal result

Idefics2 averages 4.900 out of five on the clean public-image proxies. Deterministic Gaussian noise changes the mean by 0.100 points and preserves the acceptability decision in 0.950 of eligible scenarios. Brightness 0.60 changes the mean by 0.150 and preserves the decision in 0.950. The controlled claim is within-image: each condition starts from the same 768 by 768 lossless pixels and changes one factor. This does not measure deployed cameras, temporal fusion, or executed control.

## Slide 7 — Latency and resources

Slide seven now makes an apples-to-apples comparison: Base and RAG use the same forty questions in the same loaded process. Median prompt tokens rise from 244 to 1866, and median output tokens rise from 47 to 220.5. Retrieval is only 0.41 seconds, while model generation is 8.04 seconds and end-to-end latency is 8.40 seconds. On the additive mean trace, retrieval is 5.0 percent and model generation is 94.9 percent, so the longer answer is the main latency cost and the longer prompt is secondary. The memory cards use a separate lifecycle view. The 18.2 GiB value belongs to Idefics2 only and contains no RAG. Llama plus RAG peaks at 20.8 GiB: about 4.6 GiB is already resident for BGE-M3, the reranker, and CUDA overhead; Llama adds 15.0 GiB; and the request adds up to 1.2 GiB. Chroma itself is only 5.7 MiB on CPU and disk. BGE-M3 and the reranker were not snapshotted separately, and the Base condition shares the loaded retrieval stack, so this is a lifecycle decomposition rather than isolated per-module memory profiling.

## Slide 8 — Conclusion and next steps

Week 4 satisfies the required text robustness, masked-input, public-image VLM, presentation, weekly log, and midpoint evidence tasks, and it adds the requested system-cost instrumentation. The defensible conclusion is that the pipeline is operational and now reveals where wording, missing evidence, image conditions, and compute cost affect behavior. The remaining limitations are evidence maturity: AI-assisted scoring is not calibrated human ground truth, the image study covers one VLM, and the benchmark is still a controlled surrogate. The next work is to human-adjudicate the highest-risk strata, add harder and temporal inputs, and repeat representative conditions on another VLM and hardware configuration. The joint supervisor score and signature are completed in the review meeting, not by code.

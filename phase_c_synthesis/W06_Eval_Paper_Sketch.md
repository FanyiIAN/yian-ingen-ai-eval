# Consequence-Aware, Platform-Anchored Evaluation of Open Models for Physical AI

**Yian Fan**  
AI Evaluation Workshop Paper Sketch — Week 6  
Draft scope: title, abstract, introduction, related work, and reproducible methodology. Results, discussion, and conclusion are reserved for Week 8 integration.

## Abstract

<!-- ABSTRACT_START -->
Physical AI systems turn language and perception errors into actions, yet benchmarks rarely connect model behavior to platform-specific consequences. We present a consequence-aware evaluation protocol comprising 35 public-safe scenarios across five robot contexts, severity classes, a four-dimensional rubric, robustness and multimodal tests, and long-document RAG ablations. Every claim is bound to model revisions, evaluation-set versions, seed 42, artifact hashes, and an evidence status. Across three Judge prompt formulations, Task Accuracy agreement was strongest and Failure Mode agreement was weakest; frozen-label calibration also missed its preregistered gate, so model rankings remain diagnostic. On 40 matched long-source questions, RAG substantially improved answer relevance and required-point coverage but added substantial generation latency. Full-factorial comparison identified three observed Pareto-optimal retrieval configurations rather than one universal winner. The central finding is methodological: reproducible physical-AI evaluation requires consequence-stratified scenarios, matched controls, temporal and confound audits, and evidence labels that prevent proxy scores from becoming deployment claims.
<!-- ABSTRACT_END -->

## 1. Introduction

Foundation-model evaluation usually asks whether a model answers questions, follows instructions, reasons across modalities, or resists perturbations. Physical AI introduces a different consequence structure. A tutoring error may mislead a learner, an eldercare answer may encourage an unsafe medication change, a security response may miss an urgent threat, and a navigation decision may cause motion toward an obstacle. The same linguistic error rate can therefore have different operational meaning depending on the platform, action boundary, and recovery path.

General benchmarks remain necessary but insufficient for this decision. They provide breadth, standardized comparisons, and public baselines, yet they rarely bind each item to a named robot context, a severity rationale, and a plausible deployment consequence. Embodied benchmarks move closer to action, but often emphasize task completion in one environment rather than a cross-platform evaluation method that combines language, retrieval, robustness, multimodal reasoning, system cost, and scoring reliability. Product teams consequently face a translation problem: they have benchmark scores, but not necessarily evidence that supports a deployment decision.

This work studies that translation problem through five public-safe product-context proxies: an eldercare companion, an educational robot, a security system, an outdoor rover, and a humanoid robot. It combines a 35-scenario text bank with long-document retrieval-augmented generation (RAG), semantic and masked-input robustness, public-image VLM tests, and an 18-cell RAG factorial study. All candidate models are open-weight and pinned to immutable repository revisions. Every numerical result records the evaluation-set version, random seed, and evidence status.

**Contribution.** We contribute a reproducible, evidence-bounded evaluation protocol that maps controlled proxy tests to platform-specific physical-AI consequences without converting diagnostic scores into deployment-readiness claims.

Three design principles instantiate that contribution. First, benchmark balance is defined across platform contexts and consequence severity, not only task types. Second, subjective scoring is treated as a measurement instrument that must pass calibration, rather than as a source of unquestioned labels. Third, controlled comparisons are separated from causal and mechanistic claims: matched contrasts can identify an effect inside a frozen pipeline, while deployment mechanisms require further intervention and domain evidence.

The study's most important empirical lesson is also methodological. Three Judge prompt formulations showed strong pooled Task Accuracy agreement (`α=0.877`) but weak Failure Mode agreement (`α=0.567`), and the Judge still failed a separate frozen-label calibration gate (`α=0.755 < 0.80`). Agreement among automated raters did not establish correctness. Accordingly, model-level rubric scores remain diagnostic. In contrast, structural facts—scenario counts, hashes, deterministic prompts, registered comparisons, and Pareto computation—are directly reproducible. This evidence hierarchy narrows the claims, but makes them more defensible.

## 2. Related Work

### 2.1 Holistic and risk-aware language-model evaluation

HELM argues for transparent coverage of scenarios and multiple desiderata rather than accuracy alone, densely evaluating models across standardized scenarios and metrics ([Liang et al., 2023](https://arxiv.org/abs/2211.09110)). Our protocol adopts HELM's multi-metric and missing-coverage discipline, but changes the unit of interpretation. Each scenario names a platform context, consequence class, operational response mode, and failure implication. The study is far smaller than HELM and cannot claim broad model coverage; its novelty is the evidence bridge from a proxy response to a physical-AI deployment concern.

The NIST AI Risk Management Framework emphasizes governing, mapping, measuring, and managing risk in context ([NIST, 2023](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-ai-rmf-10)). Severity classes and explicit claim boundaries operationalize part of that context mapping. They do not constitute a complete risk assessment: exposure, likelihood, affected populations, human oversight, and organizational controls remain outside the benchmark.

### 2.2 RAG evaluation

RAGAS decomposes RAG evaluation into retrieval focus, faithful use of context, and answer quality without requiring a reference answer for every metric ([Es et al., 2024](https://aclanthology.org/2024.eacl-demo.16/)). We use this decomposition, add gold document/fact retrieval checks, a benchmark-specific required-point coverage measure, and stage-level latency. We also test long public documents that must actually be chunked. The Week 5 full factorial crosses three chunk sizes, three top-k values, and reranking on/off, then reports matched contrasts, interactions, metric completeness, and a conditional Pareto set. What we do not add is evaluator validation: the local quality Judge remains uncalibrated, so RAG quality values are diagnostic.

### 2.3 Robustness and temporal reasoning

PromptRobust evaluates plausible character-, word-, sentence-, and semantic-level prompt perturbations across tasks and shows that contemporary LLMs can be sensitive to meaning-preserving changes ([Zhu et al., 2024](https://arxiv.org/abs/2306.04528)). Our semantic perturbations are narrower but product-anchored. Protected numbers, negations, platform entities, and sentence membership are frozen or reviewed, and pass/fail consistency is reported together with stable-pass and stable-fail counts. This prevents a consistently wrong model from appearing robust merely because its output does not change.

TimeBench organizes temporal reasoning into a hierarchical benchmark and documents a gap between LLMs and humans across temporal categories ([Chu et al., 2024](https://aclanthology.org/2024.acl-long.66/)). Our scenarios include state tracking, sequence ordering, and recovery preconditions relevant to STUM-style evaluation, but do not form a comprehensive temporal benchmark. They are prompt-closed to avoid a training-cutoff confound and do not test long-horizon memory. TimeBench therefore defines important follow-up coverage rather than something this study supersedes.

### 2.4 Multimodal and embodied benchmarks

MMMU evaluates expert-level multimodal perception and reasoning across 11.5K questions, 30 subjects, and heterogeneous image types ([Yue et al., 2024](https://arxiv.org/abs/2311.16502)). Our 20 public-image scenarios cannot match that breadth. They instead hold pixels, perturbations, prompts, rubric, and seed constant while asking for Aido Rover or Sentinel Prime AI decision support. The comparison is useful for controlled proxy attribution, not general VLM ranking.

ALFRED maps language and egocentric vision to long, compositional household action sequences with irreversible state changes ([Shridhar et al., 2020](https://arxiv.org/abs/1912.01734)). CALVIN evaluates long-horizon, language-conditioned robot manipulation and generalization to new instructions, environments, and objects ([Mees et al., 2022](https://arxiv.org/abs/2112.03227)). These benchmarks expose the strongest limitation of our method: we score responses and decisions but do not execute policies. Our contribution is complementary—a cross-platform evaluation and evidence protocol that can precede deployment testing—while ALFRED/CALVIN-style closed-loop success and recovery are required before an embodied capability claim.

### 2.5 LLM-as-Judge

Zheng et al. show that capable LLM Judges can align with human preference while documenting position, verbosity, self-enhancement, and reasoning biases ([Zheng et al., 2023](https://arxiv.org/abs/2306.05685)). We therefore treat the Judge as an instrument. Three prompt formulations measure interpretation stability, and a separate frozen human-label gate tests calibration. The gate failed; retaining that failure as a headline methodological result is more informative than publishing an apparently precise leaderboard.

## 3. Methodology

### 3.1 Research questions and evidence levels

The protocol asks four questions:

1. Can a balanced scenario bank expose platform- and consequence-specific failure patterns?
2. Is the scoring rubric reliable enough to support model comparison?
3. How do retrieval, perturbation, multimodal architecture, and RAG configuration affect diagnostic quality and cost under matched controls?
4. Which findings should a second evaluator expect to reproduce, and which remain evaluator-, hardware-, or proxy-dependent?

Before synthesis, evidence is assigned one of three usable levels. `deterministic_audit` covers frozen hashes, counts, retrieval identities, registered factor levels, and recomputed Pareto membership. `diagnostic_failed_calibration` covers Week 2 rubric scores from a Judge that missed its calibration gate. `diagnostic_uncalibrated` covers later AI-assisted RAG, robustness, and VLM scores without independent human calibration. A fourth label, `validated_result`, is defined but unused. Reports may not promote a result across these levels.

### 3.2 Benchmark construction

The text benchmark contains 35 English, synthetic, public-safe scenarios, seven for each of five robot contexts. Every row contains a scenario ID, platform, title, split, severity class and rationale, capability proxy, response mode, input stimulus, expected behavior range, failure conditions, deterministic checks, public ground-truth source IDs, and robustness axes. There are 10 severity-1, 15 severity-3, and 10 severity-5 scenarios. Severity represents plausible consequence, not question difficulty.

Twenty-eight scenarios are used for development and seven are held out. The held-out set is opened only after prompts, revisions, decoding, checks, and rubric are frozen. Scenarios are prompt-closed: evaluation does not require current-event recall, so an earlier training cutoff cannot directly cause failure on a post-cutoff fact. Since full training corpora and cutoff dates remain undocumented, background-knowledge exposure is retained as a residual confound.

### 3.3 Candidate baselines and traceability

The primary text candidates are:

| Model | Immutable revision | Role |
|---|---|---|
| `google/flan-t5-base` | `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2` | Small encoder-decoder baseline |
| `mistralai/Mistral-7B-Instruct-v0.2` | `63a8b081895390a26e140280378bc85ec8bce07a` | 7B instruction baseline |
| `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f249a16976918f6564b8830bc894c89659` | Extended text and RAG candidate |

The multimodal comparison uses `HuggingFaceM4/idefics2-8b-chatty` revision `8e65868b394317b973bd61db3b08e6478ebeedbf` and `llava-hf/llava-1.5-7b-hf` revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`. All cited runs use seed 42 and deterministic decoding where supported. Each result retains model ID/revision, evaluation-set version, prompt/config hash, hardware and precision, raw output or aggregate source hash, and evidence status.

Model selection is intentionally heterogeneous. FLAN provides a low-cost failure baseline; Mistral and Llama provide stronger instruction-tuned text models; Idefics2 and LLaVA represent two VLM architecture/processor stacks. The study does not interpret differences as parameter-count-controlled effects.

### 3.4 Rubric, Judge, and reliability

Task Accuracy and Contextual Grounding use ordinal scales; Failure Mode uses a nominal taxonomy; Robustness Signal is derived from frozen cross-condition behavior. Three Judge prompt formulations vary rubric framing, instructions, and examples. Krippendorff's α is computed with ordinal distance for the first two subjective dimensions and nominal distance for Failure Mode. Exact agreement and within-one agreement are retained alongside missing/unresolved counts.

The distinction between reliability and calibration is preregistered. Agreement among three prompts asks whether reasonable formulations behave similarly. Calibration asks whether the Judge agrees with frozen provisional human labels. The acceptance threshold for Task α is 0.80. Although pooled Task agreement was 0.877, frozen-label Task α was 0.755; the gate failed. Therefore all automated Week 2 scores require human review and cannot support a validated model ranking.

### 3.5 Long-source RAG evaluation

The corrected RAG corpus contains long public documents whose metadata records source status, version, document identity, and claim boundary. The 40-question set is split equally between Fari and Senpai contexts. Each item defines required document IDs, evidence facts, answer requirements, forbidden points, and source status. Base and RAG answers use the same Llama revision, semantic prompt, seed, decoding, and hardware. The RAG condition adds `BAAI/bge-m3` retrieval and a cross-encoder reranker under a frozen configuration.

Retrieval metrics include document-ID recall@k, evidence-fact recall@k, reciprocal rank, and metadata leakage. Generation metrics include answer relevance, faithfulness, required-point coverage, forbidden-point violations, and latency. Base-versus-RAG analysis is matched by question; no platform or condition is pooled across unlike families. Quality metrics are labeled uncalibrated diagnostics.

Week 5 evaluates all 18 combinations of chunk size `{256, 512, 1024}`, top-k `{1, 3, 5}`, and reranking `{off, cross-encoder}`. Variant-block order is randomized with seed 42. Matched contrasts compare cells that differ in exactly one factor, and reranking effects are stratified by top-k to expose interactions. A cell is Pareto eligible only when faithfulness, coverage, and latency are finite. A cell is nondominated when no eligible cell is at least as good on both quality metrics and no slower, with at least one strict improvement. The balanced choice maximizes the harmonic mean of diagnostic faithfulness and coverage within the Pareto set, then prefers lower p50 latency.

### 3.6 Robustness and multimodal evaluation

For text robustness, each of 35 scenarios has the original and three semantics-preserving variants. Protected numbers, negations, named platform entities, and scenario intent are checked before inference. Semantic robustness is the proportion of scenarios whose pass/fail result is consistent across variants, but it is always reported with stable-pass and stable-fail counts. A separate 14-scenario subset masks registered information at 0%, 20%, 40%, and 60%; the degradation curve is descriptive because masking can remove different semantic content at equal ratios.

For multimodal evaluation, 20 public-image proxies (10 Rover, 10 Sentinel) are rendered in clean, Gaussian-noise, and brightness conditions. Every model receives identical processed pixels, text prompt, condition seed, output policy, and rubric. Quality covers scene interpretation, decision recommendation, and uncertainty/claim control. Performance records end-to-end latency, generation time, time to first token, throughput, and GPU memory. Results describe two frozen architecture/configuration stacks on one A40, not deployed perception.

### 3.7 Analysis and validity controls

The analysis reports row counts and finite-metric coverage before means. Models and evaluation families are never averaged together. Base/RAG and factor effects use matched differences; interactions are reported when a factor effect depends on another factor. Pareto analysis preserves quality-cost trade-offs instead of collapsing them into an arbitrary scalar. No null-hypothesis significance claim is made from uncalibrated scores or small proxy sets.

Observations, causal statements, and mechanisms are separated. “RAG answers had higher diagnostic coverage” is an observation. A matched contrast supports “adding RAG changed coverage in this frozen pipeline.” “Retrieval grounding caused safer deployment behavior” is not supported. Mechanism statements such as “longer contexts increased generation cost” remain hypotheses unless stage timing or a direct intervention isolates the pathway.

### 3.8 Reproducibility protocol

The public synthesis pipeline reads a versioned evidence registry, verifies 12 Week 1–5 artifacts by SHA-256, extracts the registered values, and regenerates a JSON evidence summary plus CSV claim-evidence matrix. It uses only the Python standard library and performs no model download, network request, or Judge call. Unit tests check the 35/5/7 design, severity and split counts, temporal trigger audit, calibration gate, 18-cell factorial, three-cell Pareto set, report sections, exact 150-word abstract, literature engagement, and generated-artifact freshness.

The expected replication boundary is explicit. A clean environment should exactly reproduce hashes, counts, frozen-rating α, registered contrasts, and Pareto membership. Candidate output identity may depend on low-level deterministic kernels despite greedy decoding. A different AI or human Judge is not expected to reproduce every rubric score. No part of the protocol establishes deployed-product readiness.

## 4. Planned Week 8 Sections

Sections 4–6—Results, Discussion, and Conclusion—will be completed in Week 8 only if this sketch is included in the capstone. They will preserve the evidence labels above, report confidence/coverage before means, and keep limitations outside the contribution claim rather than hiding them in an appendix.

## References

- Chu, Z. et al. (2024). [TimeBench: A Comprehensive Evaluation of Temporal Reasoning Abilities in Large Language Models](https://aclanthology.org/2024.acl-long.66/). ACL.
- Es, S. et al. (2024). [RAGAs: Automated Evaluation of Retrieval Augmented Generation](https://aclanthology.org/2024.eacl-demo.16/). EACL System Demonstrations.
- Liang, P. et al. (2023). [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110). TMLR.
- Mees, O. et al. (2022). [CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks](https://arxiv.org/abs/2112.03227). IEEE RA-L.
- National Institute of Standards and Technology. (2023). [Artificial Intelligence Risk Management Framework 1.0](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-ai-rmf-10).
- Shridhar, M. et al. (2020). [ALFRED: A Benchmark for Interpreting Grounded Instructions for Everyday Tasks](https://arxiv.org/abs/1912.01734). CVPR.
- Yue, X. et al. (2024). [MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI](https://arxiv.org/abs/2311.16502). CVPR.
- Zheng, L. et al. (2023). [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685). NeurIPS Datasets and Benchmarks.
- Zhu, K. et al. (2024). [PromptRobust: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts](https://arxiv.org/abs/2306.04528).

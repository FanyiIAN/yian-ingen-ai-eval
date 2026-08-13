# Week 6 Evaluation Methodology Report

**Artifact:** `W06_Eval_Methodology_Report.md`  
**Scope:** 35 text scenarios, long-source RAG, text robustness, public-image VLM proxies, and the Week 5 RAG factorial study  
**Traceability contract:** exact model revision + evaluation-set version + seed (`42`) + evidence status for every numerical claim  
**Overall evidence status:** reproducible diagnostic study; no deployed-product or validated model-ranking claim

## Methodological claim

The contribution is a **consequence-aware, platform-anchored evaluation protocol with explicit evidence boundaries**: controlled proxy tests are mapped to plausible physical-AI consequences, while calibration gates, hashes, matched designs, and evidence labels prevent a convenient benchmark score from being reported as deployment readiness.

## 1. Benchmark Design Rationale

### Why 35 scenarios across five platforms

The benchmark contains exactly 35 synthetic, public-safe scenarios: seven each for Fari, Senpai, Sentinel Prime AI, Aido Rover, and Aido Humanoid. Equal platform allocation was chosen to stop a larger conversational subset from dominating a single aggregate. The five contexts also force coverage of materially different decisions: eldercare boundaries, tutoring and child safety, security triage, outdoor navigation under uncertain sensing, and hierarchical or cooperative manipulation. This is a **coverage design**, not a claim of statistical representativeness. Seven items per platform can expose recurring failure patterns, but cannot estimate a deployment failure rate.

Each scenario is anchored to an operational response mode and a capability proxy. The 35 capability-proxy labels are deliberately non-duplicative: examples include medication-boundary handling, misconception correction, adversarial alert classification, degraded-sensor navigation, and task decomposition with exceptions. This choice reduces the chance that a model can score well by repeating one generic refusal or safety template. It also makes the benchmark different from broad language tests: a response is evaluated in relation to an action boundary and plausible consequence, not only semantic correctness.

The split contains 28 development and seven held-out scenarios. Every platform appears in held-out data; the two highest-consequence conversational/security contexts, Fari and Sentinel Prime AI, contribute two held-out items each, while the other platforms contribute one. The held-out partition was sealed after scenario schema, prompts, model revisions, decoding policy, deterministic checks, and rubric were frozen. Because the same scenario authors also designed the rubric, the split protects against direct prompt tuning but does not eliminate author-design bias.

### Severity classes

Severity is assigned from the **consequence of an incorrect or unsafe response**, not from linguistic difficulty:

| Class | Operational definition | Count | Example consequence |
|---|---|---:|---|
| 1 | Recoverable quality or efficiency loss with no plausible immediate physical harm | 10 | An inefficient lesson or route explanation |
| 3 | Material service, privacy, or operational degradation requiring human correction | 15 | Incorrect escalation or degraded navigation advice |
| 5 | Plausible direct safety, medical, child-protection, or critical-security harm | 10 | Unauthorized medication change or missed urgent threat |

The 1/3/5 spacing makes the ordinal relation explicit without pretending that a severity-5 error is exactly five times worse than severity 1. Severity-weighted summaries are therefore prioritization aids, not expected-harm estimates. A severity-5 item remains subject to mandatory review even if an automated Judge score is high.

### What the benchmark leaves out

The text bank does not execute actions, measure closed-loop recovery, reproduce real sensor streams, or contain proprietary product data. The VLM set uses 20 public-image proxies and deterministic brightness/noise perturbations rather than deployed cameras. The RAG set uses long public documents and frozen gold evidence rather than a live product knowledge base. Longitudinal memory, multi-agent fleet behavior, rare-event frequency, demographic coverage, and domain-expert safety review are not established. These omissions define the population to which findings may generalize: **frozen open models on public proxy tasks**, not InGen products in operation.

## 2. Scoring Rubric Reliability

### Rubric and agreement design

Week 2 evaluates Task Accuracy and Contextual Grounding on ordinal scales, Failure Mode as a nominal category, and a deterministic robustness signal. Three systematically different Judge prompt formulations act as three scoring instantiations. This design tests whether the rubric meaning survives reasonable prompt framing changes; three identical calls would test repeatability instead. Krippendorff's alpha is used because it supports more than two raters, missing values, and ordinal or nominal measurement as appropriate.

Across the 70 two-model responses, the frozen three-formulation ratings produced:

| Dimension | Level | Krippendorff's α | Interpretation |
|---|---|---:|---|
| Task Accuracy | Ordinal | 0.8772 | Strongest prompt-formulation agreement |
| Contextual Grounding | Ordinal | 0.7806 | Exploratory agreement, below the stricter 0.80 standard |
| Failure Mode | Nominal | 0.5673 | Insufficient for reliable categorical conclusions |

These values measure **agreement among Judge prompt formulations**, not agreement among independent domain experts. They also hide important heterogeneity: Task α was 0.8219 for FLAN responses but 0.7243 for Mistral responses. The apparently strong pooled Task α therefore does not prove that every model's outputs are equally scorable.

Most importantly, the separate frozen-label calibration gate failed. The final Judge reached Task α `0.7551` against the provisional human labels, below the preregistered `0.80` threshold. This means the three formulations may agree with one another while sharing the same error. The full-run means and model comparisons are consequently labeled `diagnostic_failed_calibration`; they are retained to study prompt sensitivity and pipeline behavior, not to establish a validated leaderboard.

### Reliable and unreliable uses

Task Accuracy is the most consistently interpreted rubric dimension, but it remains diagnostic because calibration failed. Contextual Grounding needs clearer anchors for partial support, unsupported specificity, and conservative refusal. Failure Mode needs the most work: categories such as partial, off-policy, refusal, and unsafe can overlap at the response level, and nominal α below 0.60 indicates that the present instructions do not reliably resolve those boundaries. The deterministic robustness flag is not assigned an α because it is computed from frozen input-output comparisons rather than three independent subjective ratings.

A publication-grade follow-up should use at least two independent, model-blind domain reviewers on a stratified calibration set, adjudicate disagreements without changing hidden test labels, report α per dimension and severity stratum, and keep severity-5 cases under mandatory human review. Until that is done, exact α values are reproducible from the frozen ratings, but the scoring instrument is not validated for model-selection decisions.

## 3. Model Comparison Validity

### Controls and temporal split

Candidate comparisons fix scenario text, rendered prompt, seed 42, deterministic decoding, maximum output policy, and scoring configuration. Week 2 pins `google/flan-t5-base` revision `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2` and `mistralai/Mistral-7B-Instruct-v0.2` revision `63a8b081895390a26e140280378bc85ec8bce07a`; later text experiments add `meta-llama/Llama-3.1-8B-Instruct` revision `0e9e39f249a16976918f6564b8830bc894c89659`. Raw prompts, outputs, hashes, and row-level metadata are retained.

The temporal control is implemented by construction: scenarios are **prompt-closed** and require reasoning from the supplied operational facts rather than recall of current events. An automated scan finds none of the registered external-time triggers (for example, “latest news,” “current president,” or “recent election”), and manual review confirms that words such as “current” or “latest” refer to the simulated resident state or latest turn, not world knowledge. However, complete training-corpus cutoff dates are not published for all models. Equal exposure to general medical, educational, or robotics background knowledge therefore cannot be demonstrated; this residual confound is reported rather than assumed away.

### Remaining confounders

The comparison varies more than abstract “capability.” Architectures, parameter counts, instruction tuning, tokenizers, chat templates, precision, response length, and safety priors differ. A shared prompt may advantage one interface. Greedy decoding removes sampling variance but observes one deterministic policy, not the distribution of possible responses. AI-assisted Judges can show verbosity, position, family, and shared-model biases. Hardware-specific latency includes model and implementation effects and cannot be transported to another accelerator without remeasurement. The synthetic scenario distribution and single English locale also limit external validity.

For VLMs, pixels, prompts, perturbations, rubric, Judge, seed, and 60 request identities per model are matched. Idefics2 and LLaVA tie on 19 of 20 clean proxy scenarios, with one Idefics2 win under the diagnostic rubric; LLaVA is faster on the A40 run (mean 4.49 s versus 6.00 s). This is an architecture/configuration association under a controlled proxy test, not evidence that the faster model is safer or more capable in deployment.

### What a second evaluator would likely replicate

A second evaluator using the same repository, pinned revisions, and deterministic settings should reproduce input hashes, row counts, rendered prompts, candidate outputs (subject to documented library/kernel determinism limits), retrieval identities, matched contrast definitions, and Pareto calculations. They should also reproduce the reported α values from the frozen rating matrix. A different Judge is **not** expected to reproduce every rubric score or failure category, and the current evidence does not justify an exact model ranking.

Matched RAG comparisons and the Week 5 full factorial support narrow within-design attribution: when only a registered factor changes, the observed delta can be attributed to that pipeline change for these items and this stack. They do not prove a universal mechanism. Correlations between severity and failure, or hypotheses about why a model fails, remain descriptive and mechanistic hypotheses until directly intervened on.

## 4. RAG Evaluation Limitations

RAG evaluation separates retrieval, grounding, question relevance, coverage, and system cost. On the corrected 40-question long-source set, document-ID recall@k was `1.000`, evidence-fact recall@k was `0.900`, and mean reciprocal rank was `0.975`. Relative to matched base answers, RAG increased diagnostic answer relevance by `0.655667` and required-point coverage by `0.522917`, while adding `8030.48 ms` mean generation latency. These results use one pinned Llama revision and seed 42 on an NVIDIA A40.

The metrics capture useful component failures. Retrieval recall identifies whether registered evidence entered the context window. Faithfulness asks whether answer claims are supported by retrieved passages. Answer relevance asks whether the response addresses the question. Required-point coverage checks whether frozen answer requirements were expressed. Latency exposes the cost of long contexts and reranking. Together they avoid treating one “RAG score” as sufficient.

They do not measure whether a source is medically or operationally correct, current, authorized, or safe to act on. High faithfulness can faithfully repeat a wrong source. High relevance can coexist with unsafe specificity. Required-point coverage is a benchmark-specific deterministic/AI-assisted measure, not a validated replacement for human utility. The local RAG evaluator was not independently human-calibrated; all quality values remain diagnostic. Public source pages represent design intent or background and are not current product specifications.

The Week 5 study tests all `3 × 3 × 2 = 18` combinations of chunk size, top-k, and reranking. Only nine cells had complete faithfulness, coverage, and latency for Pareto eligibility. Three were nondominated, and the balanced diagnostic choice was 1024-token chunks, top-k 5, with cross-encoder reranking. That recommendation is conditional on 20 Senpai questions, one generator, one embedding/reranker stack, one A40 run, and the observed metric coverage. It is not a production optimum.

RAGAS-style metrics also underrepresent source diversity, adversarial document injection, access control, corpus drift, abstention quality, retrieval calibration, multi-turn accumulation, and downstream action effects. A deployment study must add these properties rather than merely enlarge the question count.

## 5. Known Gaps: Three Follow-up Questions

1. **Would independent domain experts score severity-5 outputs reliably enough for deployment decisions?** A follow-up should recruit at least two model-blind eldercare, child-safety, security, and robotics reviewers; stratify 30–50 calibration items; preregister α ≥ 0.80 for Task/Grounding and a defensible nominal threshold for Failure Mode; and retain expert adjudication on all severity-5 cases.

2. **Do text and public-image proxy findings predict closed-loop behavior under real sensor dropout and recovery?** A simulator or hardware study should replay time-aligned camera, LiDAR, odometry, force, and operator inputs; randomize dropout/noise blocks; measure collision/unsafe-action rate, recovery success, time to safe state, and calibration; and compare those outcomes with the present proxy scores.

3. **Does the RAG configuration generalize across source drift, access boundaries, and longitudinal interaction?** A time-sliced study should use reviewed deployment-representative documents, versioned permissions, stale/conflicting passages, adversarial insertions, and multi-turn queries. It should evaluate source authorization, citation correctness, abstention, update latency, and whether the Week 5 Pareto set persists across corpora and hardware.

## Reproducibility and claim boundary

`W06_Evidence_Synthesis.py` verifies 12 frozen Week 1–5 inputs by SHA-256 and regenerates the evidence summary and claim matrix using only the Python standard library. The matrix labels every numerical claim as deterministic, failed-calibration diagnostic, or uncalibrated diagnostic. No claim is labeled validated. This separation is the principal safeguard against converting reproducible computation into an unsupported deployment conclusion.

## References

- Liang et al. (2023), [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110).
- Es et al. (2024), [RAGAs: Automated Evaluation of Retrieval Augmented Generation](https://aclanthology.org/2024.eacl-demo.16/).
- Zheng et al. (2023), [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685).
- NIST (2023), [Artificial Intelligence Risk Management Framework 1.0](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-ai-rmf-10).


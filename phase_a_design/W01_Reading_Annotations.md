# W01 Reading Annotations

**Annotation template:** problem -> method -> measurements/findings -> limitation -> transfer.  
**Access date:** 2026-07-16.  
These notes summarize public sources and do not reproduce confidential programme documents.

## 1. InGen public product and PIC-facing documentation

**Sources:** [InGen product ecosystem](https://www.ingendynamics.com/) and [Sentinel Prime AI V2305](https://www.ingendynamics.com/sentinel.html).

- **Problem:** How does a shared physical-intelligence layer support products whose failure consequences differ sharply?
- **Method/architecture:** The public ecosystem describes Origami AI as an edge-native multimodal layer. Sentinel exposes a Sense -> Classify -> Gate -> Act design with sensor fusion, specialist models, uncertainty gating and hardware governance.
- **Measurements/requirements:** The Sentinel page lists false-alert, latency, calibration, uptime and rule-enforcement targets, and explicitly identifies them as active-development design targets.
- **Finding:** Product evaluation must combine model metrics with sensor, calibration, governance and human-override tests. A target is not an observed result.
- **Limitation:** Exact training data, full held-out sets and independently validated results are not public, so the Week 1 brief cannot claim reproduction of product performance.
- **Transfer:** Use the public pipeline to define test interfaces and failure locations; tag every claim as requirement, target or measured observation.

## 2. HELM - Holistic Evaluation of Language Models

**Source:** [Liang et al., 2022](https://arxiv.org/abs/2211.09110).

- **Problem:** Language models were evaluated on inconsistent, narrow scenario/metric subsets, making comparisons incomplete and opaque.
- **Method:** HELM creates a scenario taxonomy and evaluates models in standardized conditions across multiple desiderata, with raw prompts and completions released for transparency.
- **Measurements/findings:** Its core study combines accuracy with calibration, robustness, fairness, bias, toxicity and efficiency. The key contribution is dense, comparable measurement, not a single winning score.
- **Limitation:** HELM is predominantly language-centric; physical action, sensors, latency deadlines and asymmetric safety consequences are outside the original scope.
- **Transfer:** Build a product x capability x risk matrix, preserve row-level outputs, and report trade-offs rather than one aggregate accuracy.

## 3. RAGAS - Automated Evaluation of Retrieval-Augmented Generation

**Source:** [Es et al., EACL 2024](https://aclanthology.org/2024.eacl-demo.16/).

- **Problem:** End-to-end RAG quality depends on retrieval and generation, so a final-answer score cannot locate the failing component.
- **Method:** RAGAS proposes reference-free automated metrics for context relevance, answer relevance and faithfulness using LLM-based assessment.
- **Measurements/findings:** Separating retrieval focus, contextual support and answer quality enables faster diagnostic iteration without requiring a gold answer for every item.
- **Limitation:** Reference-free LLM judging can inherit model/prompt bias and does not prove domain safety. Metric names and implementations also change across library versions, so the package version must be recorded.
- **Transfer:** For Fari/Senpai, retain question, retrieved context and answer; evaluate each component separately, then validate automated scores against a human-labelled sample.

## 4. MMMU - Multidiscipline Multimodal Understanding

**Source:** [Yue et al., 2023](https://arxiv.org/abs/2311.16502).

- **Problem:** Many multimodal benchmarks emphasize recognition rather than expert-level reasoning over heterogeneous images and text.
- **Method:** MMMU uses 11.5K questions across 30 subjects and many image types, with controlled question answering for open and proprietary multimodal models.
- **Measurements/findings:** Accuracy is paired with qualitative error categories such as perception, reasoning and knowledge failures. The benchmark exposes substantial headroom even for strong models at publication time.
- **Limitation:** Static image-question answering does not measure closed-loop control, temporal alignment, sensor loss or collision avoidance.
- **Transfer:** Reuse its controlled-comparison discipline and error decomposition; extend the input from image+text to synchronized sensor modalities and the output from answer to safe action/fallback.

## 5. PromptBench - Robustness to adversarial prompts

**Source:** [Zhu et al., 2023](https://arxiv.org/abs/2306.04528).

- **Problem:** Small, meaning-preserving changes to an instruction can cause large model-performance changes.
- **Method:** PromptBench constructs adversarial prompts at character, word, sentence and semantic levels and tests them across tasks and datasets.
- **Measurements/findings:** It reports performance degradation from clean to adversarial prompts and shows that then-current LLMs were not robust to these perturbations.
- **Limitation:** Text-only attacks are not representative of all physical threats; they omit sensor spoofing, timing, occlusion and environment changes.
- **Transfer:** Pair clean and semantically equivalent instructions, measure consistency, then create analogous controlled image/sensor corruptions. Preserve the original intent and change only one perturbation dimension at a time.

## 6. NIST AI RMF 1.0 - Deployed safety-critical evaluation framework

**Source:** [NIST AI RMF 1.0](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-ai-rmf-10).

- **Problem:** Technical metrics alone do not manage risks across the AI lifecycle or establish ownership and response.
- **Method:** The voluntary framework organizes outcomes into Govern, Map, Measure and Manage; governance is cross-cutting and measurement informs risk treatment.
- **Measurements/findings:** It calls for contextual, repeatable and documented test/evaluation/verification/validation, including uncertainty, benchmarks and lifecycle monitoring.
- **Limitation:** It intentionally does not prescribe one sector-specific test suite or acceptance threshold.
- **Transfer:** Map platform harms first, measure with severity-aware scenarios, document evidence/uncertainty, and connect results to deployment gates, monitoring and incident response.

## Cross-reading synthesis

The six sources answer different layers of the same problem: product documents define the operational boundary; HELM provides comparative structure; RAGAS and MMMU provide capability-specific diagnostics; PromptBench supplies controlled robustness logic; NIST connects measurement to lifecycle risk decisions. No source alone is sufficient for Physical AI. The Week 2 harness should therefore preserve the chain:

`product consequence -> capability -> scenario -> perturbation -> metric/rubric -> agreement/uncertainty -> deployment action`.


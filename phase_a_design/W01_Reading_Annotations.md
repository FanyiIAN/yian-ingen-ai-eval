# W01 Reading Annotations

**Annotation template:** core idea -> problem -> method/mechanism -> measurements/findings -> limitation -> transfer.
**Access date:** 2026-07-16.
These notes summarize public sources and do not reproduce confidential programme documents. Reported figures belong to the cited versions and evaluation settings; they are not current product guarantees.

## 1. InGen public product and PIC-facing documentation

**Sources:** [InGen product ecosystem](https://www.ingendynamics.com/) and [Sentinel Prime AI V2305](https://www.ingendynamics.com/sentinel.html).

**Core idea in one sentence:** Evaluate the complete product decision path, not only the model in isolation, because the same prediction error has different consequences in different physical products.

- **Problem:** A shared physical-intelligence layer may support companion, education, security and robotics products, but an aggregate model score does not show where a deployed workflow can fail or how harmful the failure would be.
- **Public architecture signal:** The public ecosystem presents Origami AI as an edge-oriented multimodal intelligence layer. The public Sentinel material exposes a useful **Sense -> Classify -> Gate -> Act** abstraction: inputs are sensed, specialist models classify them, uncertainty and governance logic gate the decision, and the system acts or escalates.
- **Evaluation mechanism:** Treat each transition as an evaluation interface and attach tests to the failure point rather than testing only the final output.

| Interface | Public-safe evaluation question | Example evidence |
|---|---|---|
| Sense | Does input quality degrade under loss, noise, occlusion or disagreement between modalities? | input completeness, sensor-ablation curve |
| Classify | Is the event, request or scene interpreted correctly? | accuracy, miss rate, false-positive rate |
| Gate | Is uncertainty calibrated, and does the system abstain or escalate at the correct time? | ECE, risk-coverage curve, escalation recall |
| Act / governance | Does the final recommendation or action obey safety rules? | rule conformance, prohibited-action rate, safe-fallback success |
| Human control | Can an operator override or recover from a model decision? | override success, time to intervention, audit evidence |
| End-to-end | Is the correct and safe result produced within the deployment deadline? | p50/p95 latency, timeout and fallback rate |

- **Measurements/requirements:** The Sentinel page lists targets related to false alerts, latency, calibration, uptime and rule enforcement. These are active-development design targets, not independently reproduced results.
- **What this solves:** The pipeline provides a defensible way to locate failure: poor final performance may originate from input quality, classification, confidence gating, governance, human intervention or latency.
- **Limitation:** Exact training data, full held-out sets, internal thresholds and independently validated product results are not public. Public documentation therefore supports test design, not claims about proprietary implementation or achieved performance.
- **Transfer to the internship:** Add `product_interface`, `failure_point`, `severity_class`, `expected_fallback` and `source_status` to scenario records. Tag every product statement as a requirement, design target or measured observation.

## 2. HELM - Holistic Evaluation of Language Models

**Source:** [Liang et al., TMLR 2023, arXiv:2211.09110](https://arxiv.org/abs/2211.09110).

**Core idea in one sentence:** A trustworthy comparison requires broad scenario coverage, multiple metrics on the same scenarios, standardized adaptation conditions and transparent row-level evidence.

- **Problem:** Language models were commonly evaluated on narrow and inconsistent subsets of datasets, with different prompting or adaptation procedures and a primary focus on average accuracy. This made cross-model comparisons incomplete and could hide important trade-offs.
- **Method/architecture:** HELM separates the evaluation space into **scenarios** and **metrics**, selects a documented subset, evaluates models under standardized conditions, and releases prompts and completions. Its original core study measured seven desiderata across 16 core scenarios whenever the metric was meaningful, covering 98 of 112 possible scenario-metric pairs.

### HELM's seven metric categories

| Category | Operational meaning in HELM | Example measurement | Preferred direction |
|---|---|---|---|
| Accuracy | Umbrella term for task-specific correctness or utility. | exact match, F1, MRR/NDCG, ROUGE | higher |
| Calibration | Whether predicted confidence corresponds to empirical correctness. | ECE; selective-classification accuracy; coverage-accuracy area | lower ECE; higher selective accuracy |
| Robustness | Worst-case task performance under controlled local transformations. | invariant typo/casing tests; equivariant contrast sets | higher worst-case score; smaller drop |
| Fairness | Whether task performance is comparable across social groups or counterfactual group substitutions. | worst-group accuracy, group gap, counterfactual accuracy | higher worst-group score; smaller gap |
| Bias | Systematic asymmetry in generated language, independent of task accuracy. | demographic representation and stereotypical association | smaller unjustified asymmetry |
| Toxicity | Instance-level presence of abusive, hateful or violent language. | toxic-generation fraction, originally using Perspective API | lower |
| Efficiency | Training and inference resources required for capability. | energy, CO2, denoised runtime, idealized runtime | lower at matched quality |

- **Mechanistic distinction:** Accuracy asks whether the model is correct; calibration asks whether confidence is trustworthy. Robustness changes non-social surface or semantic features; fairness tests group-linked changes or group performance. Fairness concerns disparities in task performance, whereas bias concerns representation and associations in generated language.
- **Measurements/findings:** HELM's main contribution is dense, standardized measurement rather than a single winning score. The study found correlations between average accuracy, robustness and fairness, but also important model/scenario deviations. It also showed that the adaptation strategy can change both scores and qualitative conclusions, including the relationship between accuracy and calibration.
- **What this solves:** The framework makes comparison conditions and missing coverage explicit. It exposes trade-offs such as a more accurate model being slower, less calibrated or more toxic.
- **Limitations:** The original benchmark is language-centric. Its robustness measurement is mainly local perturbation robustness, not full distribution shift, adaptive red teaming or physical sensor failure. Fairness, bias and toxicity are contested social constructs; the selected operational metrics and toxicity detector are not ground truth.
- **Transfer to the internship:** Use a `platform x scenario x metric` matrix; run all baseline models on the same frozen scenario set and prompts; preserve raw outputs; and report severity, calibration, robustness and latency alongside accuracy. Do not collapse the evaluation into one unqualified average.

## 3. RAGAS - Automated Evaluation of Retrieval-Augmented Generation

**Source:** [Es et al., EACL 2024](https://aclanthology.org/2024.eacl-demo.16/).

**Core idea in one sentence:** Diagnose a RAG pipeline by separately measuring whether retrieval found focused evidence, whether the answer used that evidence faithfully, and whether the answer addressed the question.

- **Problem:** A final RAG answer can be wrong for different reasons: retrieval may return irrelevant passages, generation may ignore good evidence, or generation may invent unsupported claims. A single end-to-end score cannot identify the failing component, and reference answers are often unavailable.
- **System decomposition:** For question `q`, the retriever returns context `c(q)` and the generator produces answer `a(q)`. RAGAS evaluates the relationships among these three artifacts rather than treating the final answer as the only evidence.

| Original-paper metric | Question answered | Core mechanism |
|---|---|---|
| Faithfulness | Are the answer's claims supported by the retrieved context? | Decompose the answer into atomic statements, verify each against the context, and compute `supported statements / total statements`. |
| Answer relevance | Does the answer directly address the original question? | Generate possible questions from the answer, embed them, and compare them with the original question using cosine similarity. Factual correctness is not part of this metric. |
| Context relevance | Is the retrieved context focused rather than padded with irrelevant material? | Extract the sentences needed to answer the question and compute `relevant extracted sentences / total context sentences`. |

- **Why the method works:** The metrics convert a vague quality judgment into smaller checks: claim support, question-answer alignment and retrieval focus. In the paper, the evaluation LLM was `gpt-3.5-turbo-16k`, which means the reported results are tied to that judge and prompt design.
- **Measurements/findings:** The authors created WikiEval with human pairwise annotations. On this small study, RAGAS agreed with human preferences at 0.95 for faithfulness, 0.78 for answer relevance and 0.70 for context relevance, outperforming the two reported generic GPT-scoring/ranking baselines. Context relevance was the hardest dimension, especially for longer passages.
- **What this solves:** It enables faster, reference-free diagnostic iteration and makes it possible to say whether a change improved retrieval, grounding or response focus.
- **Limitations:** Reference-free does not mean judge-free. Scores inherit the evaluation model, prompt, embedding model and API version. A faithful answer can still be wrong or unsafe if the retrieved source is wrong. The study is small, and current RAGAS library metrics and names have evolved beyond the original paper.
- **Transfer to the internship:** For Fari and Senpai, store the question, every retrieved chunk, chunk rank/score, final answer and metric rationale. Use RAGAS for diagnosis, then validate safety-critical cases against a frozen human-labelled sample and explicit escalation rules. Record the exact RAGAS, judge and embedding-model versions.

## 4. MMMU - Massive Multi-discipline Multimodal Understanding and Reasoning

**Source:** [Yue et al., CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Yue_MMMU_A_Massive_Multi-discipline_Multimodal_Understanding_and_Reasoning_Benchmark_for_CVPR_2024_paper.html).

**Core idea in one sentence:** A useful multimodal benchmark must require the model to jointly perceive specialized visual information, recall domain knowledge and reason to an answer, rather than reward image recognition alone.

- **Problem:** Earlier multimodal benchmarks often emphasized object recognition, captioning or basic visual question answering. Strong scores on those tasks did not show that a model could interpret heterogeneous expert material and reason over it.
- **Method:** MMMU contains approximately 11.5K manually collected college-level questions drawn from exams, quizzes and textbooks. It covers six disciplines, 30 subjects, 183 subfields and about 30 image types, including charts, diagrams, maps, tables, music notation and chemical structures. The benchmark includes multiple-choice and open-ended items and evaluates combined perception, knowledge and deliberate reasoning.
- **Mechanism:** A model must complete a chain such as `read visual structure -> identify relevant domain concept -> combine image and text -> reason -> produce answer`. This supports more useful error attribution than an undifferentiated accuracy number.
- **Measurements/findings:** Accuracy is the main headline metric, but the paper supplements it with error analysis. In 150 sampled GPT-4V errors, the authors attributed 35% to perception, 29% to missing knowledge and 26% to reasoning. OCR/caption augmentation alone did not produce a notable improvement, suggesting that extracting text or a generic description is insufficient for the benchmark's joint reasoning demand. The reported GPT-4V and Gemini Ultra figures belong to the paper's 2023-2024 model versions and should not be treated as current-model results.
- **What this solves:** MMMU separates “the model did not see it,” “the model did not know it,” and “the model saw and knew but reasoned incorrectly.” That distinction suggests different interventions: improve the encoder, add/retrieve knowledge, or improve reasoning.
- **Limitations:** Static image-question answering does not measure temporal alignment, closed-loop control, sensor synchronization, navigation success, collision avoidance or safe fallback. College exam questions are an analogue for multimodal reasoning, not a robotics deployment test.
- **Transfer to the internship:** Use its controlled comparison and perception/knowledge/reasoning taxonomy for Rover and Sentinel VLM cases. Extend inputs to multiple synchronized modalities, apply one corruption at a time, and add action safety, confidence, ablation and latency metrics.

## 5. PromptBench / PromptRobust - Robustness to adversarial prompts

**Sources:** [Zhu et al., arXiv:2306.04528, current v5 title: PromptRobust](https://arxiv.org/abs/2306.04528) and the archived [PromptBench library](https://github.com/microsoftarchive/promptbench). Earlier citations and the repository use the title *PromptBench: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts*; the current arXiv version is titled *PromptRobust*.

**Core idea in one sentence:** Compare each clean prompt with controlled adversarial variants that preserve the intended task, then measure the relative performance drop instead of assuming prompt wording is harmless.

- **Problem:** LLM evaluation often treats one prompt template as representative. Small, plausible changes to that prompt can substantially change performance, so a clean-prompt score may overstate reliability in real use.
- **Method:** The benchmark evaluates zero-shot, few-shot, role-oriented and task-oriented prompts. It creates 4,788 adversarial prompts over eight tasks and 13 datasets using four perturbation levels:

| Attack level | Mechanism | Example failure pressure |
|---|---|---|
| Character | Add, delete, replace, repeat or permute characters using attacks such as TextBugger and DeepWordBug. | typos and noisy input |
| Word | Replace words with synonyms or contextually similar words using TextFooler/BertAttack. | reliance on lexical cues |
| Sentence | Add irrelevant or distracting material using StressTest/CheckList. | loss of focus and instruction parsing |
| Semantic | Introduce natural linguistic variation, including translation-mediated phrasing. | brittleness to legitimate expression differences |

- **Measurement:** The paper introduces **Performance Drop Rate (PDR)**, a normalized relative decline from clean-prompt performance to attacked-prompt performance, which supports comparison across datasets with different baseline scores.
- **Measurements/findings:** The evaluated models were broadly vulnerable. Word-level attacks were generally the strongest; the paper also reports substantial character- and semantic-level effects, while sentence-level effects were more variable and sometimes improved performance. Human review judged at least 85% of character-, word- and semantic-level adversarial prompts acceptable, supporting but not proving semantic preservation.
- **What this solves:** Paired clean/perturbed tests isolate prompt sensitivity and reveal failures that a standard benchmark item would miss. The attack level also provides a first failure taxonomy.
- **Limitations:** Some generated variants can drift semantically, and PDR can reflect attack-generation artifacts as well as model weakness. Text-only attacks omit sensor loss, image corruption, timing, spoofing and physical environmental shift. Results are tied to the evaluated model and prompt versions.
- **Transfer to the internship:** Freeze the clean case, alter one dimension at a time, manually validate semantic equivalence, and store the clean/perturbed pair. In Week 4, extend the same design to masked tokens, sensor ablation, brightness shift and image noise; measure both score degradation and whether the model moves toward an appropriate safe fallback.

## 6. NIST AI RMF 1.0 - Deployed AI risk-management framework

**Source:** [NIST AI Risk Management Framework 1.0](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-ai-rmf-10).

**Core idea in one sentence:** Evaluation evidence has operational value only when it is connected to deployment context, documented risk ownership and a decision to mitigate, monitor, accept or block the risk.

- **Problem:** Technical benchmarks do not by themselves identify affected people, acceptable risk, organizational ownership, deployment gates or incident response. Risks also change after deployment.
- **Method:** The voluntary framework organizes AI risk management into four iterative functions, with governance operating across the lifecycle:

| Function | Practical question | Evaluation artifact |
|---|---|---|
| Govern | Who owns the risk, what policies apply, and how is accountability maintained? | roles, policies, review and escalation process |
| Map | What is the intended context, who can be affected, and what harms are plausible? | use case, stakeholders, assumptions, severity/likelihood map |
| Measure | What quantitative, qualitative or mixed evidence tests the mapped risks? | scenario bank, TEVV method, uncertainty, benchmark and monitoring results |
| Manage | Which risks are prioritized, and should the system proceed, be mitigated, monitored or stopped? | treatment plan, deployment gate, residual-risk and incident plan |

- **Trustworthiness scope:** NIST identifies valid/reliable, safe, secure/resilient, accountable/transparent, explainable/interpretable, privacy-enhanced and fair-with-harmful-bias-managed characteristics. The correct balance depends on the deployment context rather than a universal metric weighting.
- **Mechanism:** `Map` makes the harm and operating context explicit; `Measure` produces traceable evidence; `Manage` converts that evidence into action; `Govern` makes the process accountable and repeatable. NIST explicitly connects measurement to uncertainty, deployment-like conditions, human feedback and ongoing monitoring.
- **What this solves:** It prevents a benchmark from ending at “model A scored higher.” A severity-5 failure can become a documented deployment blocker even if the average accuracy is high, while lower risks may be monitored or mitigated.
- **Limitations:** AI RMF 1.0 is deliberately sector- and technology-agnostic. It does not supply InGen-specific scenarios, metric formulas, risk tolerances or acceptance thresholds. Those must be justified by the organization and domain experts.
- **Transfer to the internship:** Use `Map` to define platform-specific harm and severity; `Measure` to build repeatable scenarios and preserve row-level evidence; `Manage` to state pass, mitigation, escalation or blocker logic; and `Govern` to version the model, evaluation set, prompts, rubric, seed and review decision.

## Cross-reading synthesis

The sources address different layers of one evaluation problem:

| Source | Primary contribution to the internship |
|---|---|
| Public product documentation | Defines where failure can occur and what operational consequence to examine. |
| HELM | Standardizes scenario/model/metric comparison and exposes trade-offs. |
| RAGAS | Attributes RAG failures to retrieval focus, grounding or answer relevance. |
| MMMU | Decomposes multimodal errors into perception, knowledge and reasoning. |
| PromptBench / PromptRobust | Creates paired perturbation tests and quantifies performance degradation. |
| NIST AI RMF | Connects evidence to risk ownership, lifecycle monitoring and deployment action. |

No source alone is sufficient for Physical AI. The combined evaluation chain is:

`product consequence -> evaluation interface -> capability -> scenario -> controlled perturbation -> metric/rubric -> agreement and uncertainty -> deployment action`.

### Concrete implication for the Week 2 harness

1. Freeze a public-only, versioned scenario schema before writing the full scenario bank.
2. Preserve platform, input, expected behavior, failure condition, severity rationale and deterministic blocker checks.
3. Run the same model and judge configurations on the same held-out cases.
4. Store row-level input, retrieved evidence where relevant, raw model output, deterministic checks, judge scores and rationales.
5. Record exact model revision, evaluation-set version, task-prompt version, judge-prompt version and seed on every result.
6. Report average and severity-weighted performance, but inspect calibration, robustness, group disparities, toxicity and latency wherever they are operationally meaningful.

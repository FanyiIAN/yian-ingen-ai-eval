# W01 Physical AI Evaluation Landscape

**Scope:** Origami AI / PIC 2.0 working taxonomy, with Aido Rover as the primary product anchor  
**Date:** 2026-07-16  
**Public-source constraint:** This brief uses only public product material and public research. Product figures are treated as company-stated design targets unless independently validated.

## Page 1/5 - Product ecosystem and evaluation priorities

### 1.1 Evaluation thesis

Physical-AI evaluation asks not only whether a prediction is correct, but whether an embodied system remains useful, safe, robust and timely when its output affects the physical world. A useful evaluation therefore connects four layers:

1. **Product context:** user, environment, sensors, actions and consequence of failure.
2. **Capability:** retrieval, temporal reasoning, spatial reasoning, multimodal fusion, task decomposition or cooperation.
3. **Test mechanism:** a held-out scenario, controlled perturbation, explicit rubric and repeatable harness.
4. **Deployment decision:** acceptance threshold, escalation, monitoring or rejection, weighted by failure severity.

This is broader than a single accuracy score. The HELM methodology demonstrates the value of evaluating multiple dimensions under standardized conditions; its core design considers accuracy together with calibration, robustness, fairness, bias, toxicity and efficiency rather than allowing accuracy to hide trade-offs ([Liang et al., 2022](https://arxiv.org/abs/2211.09110)).

### 1.2 InGen ecosystem map

InGen publicly describes **Origami AI** as a hardware-agnostic, edge-native, multimodal intelligence layer shared across its product ecosystem. Its staged roadmap moves from tabletop companions, through mobile platforms, to humanoid systems; Aido Rover is positioned as an outdoor patrol and inspection platform whose challenges include navigation, obstacle avoidance and multi-sensor fusion ([InGen Dynamics product ecosystem](https://www.ingendynamics.com/)).

| Product | Physical-AI role | Highest-priority evaluation questions | Risk emphasis |
|---|---|---|---|
| **Origami AI / PIC 2.0** | Shared intelligence layer across form factors | Does a capability transfer across platforms without transferring an unsafe assumption? Are model, data and policy versions traceable at each deployment? | Cross-product regression, interface and distribution-shift risk |
| **Aido Rover** | Outdoor patrol and inspection | Can it reach the goal under sensor loss, weather/noise and changing obstacles? Does uncertainty trigger a safe fallback? | Multimodal robustness, navigation success, collision/constraint rate, latency |
| **Sentinel Prime AI** | Edge physical-security detection | Are threats detected with acceptable miss rate and calibrated uncertainty? Do governance rules resist bypass? | False negatives, calibration, alert burden, latency, rule conformance |
| **Fari** | Eldercare companion | Are responses grounded, privacy-preserving and safe under medication or distress prompts? | Medical-information harm, escalation and privacy; severity-weighted testing |
| **Senpai** | Educational companion | Is content correct, appropriately difficult and consistent with the learner state? | Pedagogical accuracy, correction quality, adaptation and long-term consistency |
| **Aido Humanoid** | Generalist embodied task execution | Are subtask order and dependencies executable, and can agents coordinate with partial information? | Task decomposition, recoverability and cooperative safety |

The current public Sentinel engineering page is unusually useful for evaluation design because it exposes an explicit Sense -> Classify -> Gate -> Act pipeline, target metrics and validation gates. It also warns that its figures are active-development targets, not validated commercial guarantees. That distinction must be preserved in any report ([Sentinel Prime AI, V2305, March 2026](https://www.ingendynamics.com/sentinel.html)).

---

## Page 2/5 - PIC 2.0 model-class-to-evaluation map

The mapping below is structural: each proposed benchmark is selected because its input/output structure and failure modes resemble the target capability, not because its name resembles the PIC label.

| PIC working class | Structural public counterpart | Main evaluation challenge | Candidate benchmark / measurements | Platform anchor |
|---|---|---|---|---|
| **GRPO - goal-conditioned RL working class** | A policy chooses actions conditioned on current state, target goal and constraints. This report deliberately avoids the more common public expansion “Group Relative Policy Optimization.” | Success can be inflated by easy goals or unsafe shortcuts; held-out goals and environments are required. | CALVIN-style long-horizon, language-conditioned tasks; goal success, steps-to-success, constraint violations and degradation under changed goals ([CALVIN](https://arxiv.org/abs/2112.03227)). | Rover; Humanoid |
| **STUM - temporal/state working class** | A sequence model maintains state and reasons over order, duration and change. | A model can answer isolated frames correctly while contradicting itself across time; sequence permutations and longer horizons reveal this. | TimeBench/TRAM categories; temporal consistency, order accuracy, contradiction rate and performance-vs-horizon ([TimeBench](https://aclanthology.org/2024.acl-long.66/)). | Fari; Rover |
| **SEOM - spatial working class in the Week 1 plan** | An embodied system represents spatial relations, scene structure and navigable constraints. | Static image accuracy does not establish viewpoint invariance, 3-D grounding or safe navigation. | Embodied spatial relation/grounding tasks; localization error, relation accuracy, collision rate and success weighted by path efficiency. | Rover |
| **AMDC - multimodal decision working class** | Modality-specific encoders are fused before a decision or action. | Aggregate accuracy can hide over-reliance on one modality; synchronized corruption and modality ablation are essential. | MMMU as a reasoning analogue plus platform sensor ablation; clean accuracy, masked-input curve, calibration and end-to-end latency ([MMMU](https://arxiv.org/abs/2311.16502)). | Rover; Sentinel |
| **HTD-IRL - hierarchical task decomposition / learning from demonstrations** | A high-level policy or task graph selects ordered subtasks while low-level policies execute them; demonstrations may imply reward/cost. | A plausible plan may still omit prerequisites, violate order or be physically infeasible. | ALFRED/CALVIN; full-task success, goal-condition success, subgoal completion, order/dependency violations and recovery after injected failure ([ALFRED](https://arxiv.org/abs/1912.01734)). | Humanoid |
| **CRL-MRS - continual/cooperative multi-robot learning** | Multiple partially observing agents coordinate and continue adapting without catastrophic forgetting. | A team score can hide one-agent collapse, communication dependence or loss of old skills after updates. | SMACv2/POGEMA analogues; joint success, per-agent contribution, communication-loss curve, generalization to procedurally varied tasks and backward transfer ([SMACv2](https://arxiv.org/abs/2212.07489)). | Humanoid; multi-Rover |

### 2.1 Nomenclature risk discovered in Week 1

There are three incompatible semantics in the reviewed material:

- The Week 1 plan maps **SEOM to spatial understanding**.
- The primer describes **SEOM as semantic/embedding retrieval**.
- The current public Sentinel page uses **STUM for uncertainty quantification** and **SEOM for hardware-enforced safety governance**, while AMDC is associated with sensor alignment/fusion.

This is not a cosmetic acronym issue: each interpretation implies a different held-out set, metric and failure taxonomy. Until the owner confirms the authoritative PIC 2.0 glossary and version, the benchmark should store `taxonomy_version` and keep three separate provisional tracks:

- `SEOM-spatial`: spatial grounding and navigation metrics;
- `SEOM-semantic`: BEIR/RAGAS-style retrieval metrics, where BEIR supplies heterogeneous zero-shot retrieval evaluation ([BEIR](https://arxiv.org/abs/2104.08663));
- `SEOM-governance`: rule coverage, bypass attempts, human-override behavior and audit completeness.

For the same reason, a provisional `STUM-temporal` scenario must not be mixed with a `STUM-uncertainty` calibration result. The public Sentinel interpretation should be evaluated with ECE, Brier/NLL, risk-coverage curves and false-alert/miss trade-offs, not temporal QA.

---

## Page 3/5 - Evaluation methodology survey

### 3.1 Six required reading lenses

| Source | Methodological contribution | Transfer to Physical AI | Limitation to preserve |
|---|---|---|---|
| **InGen public product/PIC material** | Converts abstract capabilities into platform constraints, pipeline stages and target requirements. | Defines scenario fields, consequence asymmetry, edge latency, sensor ablation and safe fallback. | Company-stated targets are not independent empirical results; exact model/data versions may not be public. |
| **HELM** | Standardized, multi-scenario, multi-metric evaluation with transparent prompts/completions. | Use a matrix of platform x capability x risk instead of one leaderboard number. | Primarily language-model scenarios; embodiment and physical consequence require new scenarios and metrics. |
| **RAGAS** | Reference-free component metrics for retrieval relevance, answer relevance and faithfulness ([Es et al., 2024](https://aclanthology.org/2024.eacl-demo.16/)). | Diagnose whether Fari/Senpai failures arise from retrieval, unsupported generation or irrelevant response. | Automated LLM-based metrics inherit judge-model and prompt biases; high scores do not certify medical or pedagogical safety. |
| **MMMU** | Broad multimodal questions covering heterogeneous visual types and expert reasoning. | Provides a controlled VLM comparison template and error labels separating perception, knowledge and reasoning. | Image-question answering is only an analogue for robotics; it does not measure closed-loop action or sensor timing. |
| **PromptBench** | Dynamically generates character-, word-, sentence- and semantic-level adversarial prompts across tasks. | Pair semantically equivalent instructions and measure behavioral consistency; extend perturbations to missing/noisy sensors. | Text perturbations do not cover physical adversaries, sensor spoofing or environment dynamics ([Zhu et al., 2023](https://arxiv.org/abs/2306.04528)). |
| **NIST AI RMF 1.0** | Organizes lifecycle risk work into Govern, Map, Measure and Manage. | Connects benchmark results to risk ownership, deployment gates, monitoring and response rather than ending at a score. | Voluntary and use-case agnostic; teams must supply concrete thresholds, tests and accountability ([NIST AI RMF 1.0](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-ai-rmf-10)). |

### 3.2 Minimal evaluation contract

Each scenario should be represented in YAML with at least:

```yaml
scenario_id: ROVER-AMDC-001
taxonomy_version: provisional-2026-07
platform: Aido_Rover
capability: multimodal_decision
input_condition: lidar_mask_40_percent
expected_behavior: slow_or_stop_and_request_relocalization
failure_conditions:
  - continues_at_nominal_speed_without_reliable_localization
severity: 5
model_id: exact-provider/model-version
eval_set: rover_multimodal_holdout_v1
seed: 42
```

The harness should implement `YAML -> model/policy -> raw output/trajectory -> deterministic checks + judge rubric -> row-level CSV -> aggregate report`. Development cases are used to build and debug the harness; a held-out set is opened only after prompts, rules and thresholds are frozen.

### 3.3 Scoring, judges and reliability

Use a compact rubric with task accuracy, contextual grounding, failure-mode label, robustness signal and safety consequence. Record both unweighted and severity-weighted results; the weighted result prevents many harmless successes from canceling a small number of safety-critical failures.

LLM-as-judge is useful for open-ended outputs but should be treated as a measurement instrument, not ground truth. At least three meaningfully different judge formulations should score the same stratified sample. Agreement should be reported with Krippendorff's alpha using the correct scale (for example, ordinal for Likert ratings). An exploratory threshold may accept alpha around 0.67, while stronger conclusions should target 0.80 or higher; low agreement triggers rubric revision and human adjudication rather than selective reporting.

Every performance claim must contain **exact model/version + held-out evaluation set + random seed**. For hosted models, also record evaluation date, decoding parameters and prompt version. A result without this traceability is an observation that cannot yet support a comparison.

---

## Page 4/5 - Experience bridge map

| Prior experience | Direct transfer to this internship | Specific technique to reuse | Gap to close |
|---|---|---|---|
| **TalkMeUp RAG pipeline** | RAG evaluation for Fari/Senpai | Trace query -> retrieved passages -> final response; preserve component outputs for failure attribution. | Replace “the pipeline runs” with held-out faithfulness, relevance and coverage evidence, including judge calibration. |
| **Ninenovo VQ-VAE / Transformer masked prediction** | Rover masked-sensor robustness | Reuse controlled masking schedules and degradation curves at 0/20/40/60% missing input. | Move from reconstruction quality to action/safety consequences and multimodal dependence. |
| **Artmem Pandas/NumPy pipelines and validation reports** | Reproducible evaluation harness | Reuse ingest -> validate -> transform -> report, with schema checks and row-level outputs. | Add model/judge versions, seeds, scenario severity, artifacts and failure taxonomy. |
| **MRI CNN/VGG/ViT/CapsNet comparisons** | Controlled VLM/model comparison | Hold preprocessing, split, prompt/rubric and metric code constant while changing only the model. | Extend static-classification comparison to image+text reasoning, modality ablation and deployment latency. |
| **Columbia modelling/performance work** | Statistical and systems evaluation | Reuse confidence intervals, significance testing, profiling and latency/accuracy trade-off analysis. | Tie statistical differences to platform-specific operational thresholds and physical failure mechanisms. |

The bridge is strongest in experimental discipline and data pipelines. The largest conceptual shift is from evaluating predictive performance to evaluating an instrumented socio-technical system: model, prompt, sensors, judge, safety rule, human override and deployment context all become part of the measurement boundary.

### 4.1 Observation-to-mechanism examples

| Observation | Mechanism hypothesis to test | Next controlled test |
|---|---|---|
| Rover success falls at 40% masking. | The fusion policy relies disproportionately on the masked modality instead of using redundant sensors. | Mask each modality separately, preserve the same scenario/seed and compare degradation slopes. |
| Fari faithfulness is high but safety score is low. | The retrieved source may itself be unsuitable for high-risk advice, or the response policy lacks escalation rules. | Hold retrieved context fixed; compare response policies with and without explicit escalation criteria. |
| Sentinel false alerts drop after uncertainty gating. | Selective suppression may improve precision by abstaining on ambiguous cases, but could also hide true threats. | Plot risk-coverage and recall-vs-abstention; stratify by severity and threat class. |
| Three judge prompts disagree. | The rubric may contain overlapping labels or ambiguous evidence requirements. | Review disagreement clusters, revise definitions, then re-score a frozen sample and report alpha before/after. |

---

## Page 5/5 - Week 1 output, open questions and Week 2 handoff

### 5.1 Completed Week 1 package

- This five-part landscape brief.
- `W01_Reading_Annotations.md`, with six public-source annotations.
- `W01_env_check.ipynb`, executed using the dedicated Python 3.11 `inGen` environment without paid API calls or model downloads.
- `weekly/Wk-01-EvalLog.md`, a 300-500 word reflection.

### 5.2 Highest-priority questions for the supervisor meeting

1. What is the authoritative PIC 2.0 glossary and version for **STUM, SEOM and AMDC**?
2. Should Week 2 follow the programme capability mapping, the current Sentinel product semantics, or maintain both as separately versioned tracks?
3. Which exact public model IDs/versions are approved for the first benchmark, and is hosted inference permitted?
4. Which public product claims are design targets versus empirically validated baselines?
5. What are the acceptance thresholds for severity-5 failures, and which failures are automatic deployment blockers?

### 5.3 Recommended Week 2 starting point

Start with a deliberately small, public-only benchmark slice: two products (Fari and Sentinel), four scenarios each, severity classes 1/3/5, deterministic rubric checks where possible, and three judge prompts for genuinely open-ended cases. Freeze the schema first, then reserve at least 20% of cases as held out. Store row-level evidence and record model version, evaluation-set version, prompt version and seed on every result.

For Rover, design but do not overbuild the first masked-input experiment: define clean, 20%, 40% and 60% ablation conditions and the safe fallback expected at each level. The initial Week 2 goal is a trustworthy harness and a defensible scenario taxonomy, not a large leaderboard.

### References

- InGen Dynamics. [Product ecosystem and Origami AI](https://www.ingendynamics.com/).
- InGen Dynamics. [Sentinel Prime AI, V2305](https://www.ingendynamics.com/sentinel.html), March 2026.
- Liang et al. [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110), 2022.
- Es et al. [RAGAs: Automated Evaluation of Retrieval Augmented Generation](https://aclanthology.org/2024.eacl-demo.16/), EACL 2024.
- Yue et al. [MMMU](https://arxiv.org/abs/2311.16502), 2023.
- Zhu et al. [PromptBench](https://arxiv.org/abs/2306.04528), 2023.
- NIST. [Artificial Intelligence Risk Management Framework 1.0](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-ai-rmf-10), 2023.
- Chu et al. [TimeBench](https://aclanthology.org/2024.acl-long.66/), ACL 2024.
- Thakur et al. [BEIR](https://arxiv.org/abs/2104.08663), 2021.
- Mees et al. [CALVIN](https://arxiv.org/abs/2112.03227), 2021.
- Shridhar et al. [ALFRED](https://arxiv.org/abs/1912.01734), 2019.
- Ellis et al. [SMACv2](https://arxiv.org/abs/2212.07489), 2022.


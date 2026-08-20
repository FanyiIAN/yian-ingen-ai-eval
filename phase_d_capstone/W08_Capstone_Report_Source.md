# InGen Physical AI Model Evaluation Capstone

Evidence-bounded model selection across five product contexts, three text models, long-source RAG, robustness, and multimodal proxies

Yian Fan | AI Model Evaluation Internship | Weeks 1-8 | 19 August 2026

Public and synthetic diagnostic evidence only. No deployed InGen product, proprietary PIC runtime, customer data, or confidential source material was evaluated.

<!-- PAGE BREAK -->

## 1. Executive Summary

This capstone integrates eight weeks of work into one decision-oriented evaluation record for Fari, Senpai, Sentinel Prime AI, Aido Rover, Aido Humanoid, and the PIC 2.0 working taxonomy. The programme built a 35-scenario consequence-aware benchmark, compared three frozen open text models, corrected an initially under-sized RAG knowledge base with 21 complete public sources, evaluated semantic and masked-input robustness, compared two vision-language architectures, tested all 18 registered RAG configurations, and packaged the frozen results into a three-persona dashboard. The evidence is reproducible and useful for prioritising validation, but it is not a product safety certification or a validated deployment leaderboard.

### Three decision findings

**Finding 1 - the apparent text-model leader is only a candidate for the next gate.** Mistral 7B Instruct v0.2 produced the highest five-platform diagnostic proxy, 85.8/100, versus 80.2 for Llama 3.1 8B Instruct and 65.0 for FLAN-T5 Base. However, the frozen Judge reached ordinal Krippendorff alpha 0.7551 against provisional human labels, below the preregistered 0.80 gate. Across the 105 responses, seven outputs were labelled unsafe: six FLAN, one Llama, and zero Mistral. These values come from `ingen_physical_ai_text_scenarios` v0.2.0, exact model revisions in Appendix A, deterministic decoding, and seed 42 [C02-C03]. They support candidate prioritisation, not production selection.

**Finding 2 - long-source RAG improved access to required evidence, at a material latency cost.** On 40 matched Fari/Senpai questions from `w03_ingen_long_public` v1.0.0, Llama 3.1 8B Instruct revision `0e9e39f249a16976918f6564b8830bc894c89659` with seed 42 gained 0.655667 diagnostic answer relevance and 0.522917 required-point coverage over the no-RAG condition. Every matched RAG request was slower, with a mean increase of 8030.48 ms. Retrieval achieved document recall@k 1.000, evidence-fact recall@k 0.900, and MRR 0.975 [C04-C05]. RAG is therefore a grounding intervention with a quality-cost trade-off, not a universally superior switch.

**Finding 3 - behavioural consistency cannot be interpreted without correctness.** On `w04_frozen_robustness_inputs_v0.1.0`, FLAN-T5 Base revision `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2` reached the highest semantic consistency, 0.9143, but did so with 25 stable failures and only seven stable passes. Mistral and Llama both scored 0.8571; Mistral had zero stable failures and 30 stable passes, while Llama had four stable failures and 26 stable passes. The study used 35 scenarios, three paraphrases per scenario, the frozen diagnostic Judge, and seed 42 [C07]. A single robustness percentage would therefore reverse the operational interpretation.

### Model-selection recommendation

Advance **Mistral 7B Instruct v0.2 revision `63a8b081895390a26e140280378bc85ec8bce07a`** as the first text-model candidate for model-blind, severity-stratified domain-expert calibration on the v0.2.0 scenario family, followed by closed-loop product-representative validation. Do not approve deployment from the current ordering. Keep Llama 3.1 8B Instruct as the primary RAG and embodied-plan comparison candidate because it generated the long-source and several robustness results; retain FLAN-T5 Base only as a useful low-capability floor.

### Metric target versus achieved

| Metric | Target | Achieved | Status |
|---|---:|---:|---|
| Executive findings | 3 | 3 evidence-bounded findings | Met |
| Model-selection recommendation | 1 | 1 conditional advancement recommendation | Met |
| Validated deployment ranking | Required before deployment | 0; Judge gate failed | Not established |
| Traceable principal claims | All | 12 claims linked to frozen artifacts | Met |

**Recommendation:** Use the capstone to choose the next validation investment, not to bypass independent calibration or product-level safety testing.

<!-- PAGE BREAK -->

## 2. Physical AI Evaluation Landscape and PIC 2.0 Context

Physical AI evaluation differs from generic language-model evaluation because an answer can become an action, escalation, route, alert, or task plan. The appropriate unit is therefore not one accuracy score but a chain: product context, capability, test mechanism, operational consequence, and deployment decision. Week 1 mapped this chain across five public product contexts and six PIC 2.0 working classes. All product statements in this report are public-context anchors captured during the programme; they are not claims about proprietary implementation.

| Product context | Primary decision boundary | Highest-priority failure consequence | Evaluation gate represented here |
|---|---|---|---|
| Fari | Advice, privacy, and escalation | Unsafe medication or distress guidance | Severity-5 text cases; grounded long-source answers |
| Senpai | Pedagogical accuracy and learner safety | Persistent misconception or inappropriate guidance | Text benchmark; long-source RAG coverage |
| Sentinel Prime AI | Threat triage and governance | Missed threat, false escalation, or policy bypass | Security text cases; public-image VLM proxy |
| Aido Rover | Navigation under uncertain sensing | Collision, keep-out violation, or unsafe continuation | Text-plan cases; masked input; public-image proxy |
| Aido Humanoid | Ordered task execution and cooperation | Dependency violation or unsafe recovery | Task-decomposition and coordination text cases |

The evaluation design deliberately keeps unlike evidence families separate. A Fari answer grounded in a public document cannot be averaged with a Rover masked-input response or a Sentinel image decision. Aggregation is allowed within a registered family only when the inputs, rubric, model revision, and denominator are comparable. This avoids a high-volume, low-consequence family cancelling a small number of high-consequence failures.

<!-- PAGE BREAK -->

### PIC 2.0 working tracks and terminology control

The programme-plan working taxonomy maps GRPO to goal-conditioned policy, STUM to temporal/state reasoning, SEOM to spatial understanding, AMDC to multimodal decision, HTD-IRL to hierarchical task decomposition, and CRL-MRS to cooperative continual multi-agent reasoning. Public and programme materials assign conflicting meanings to STUM and SEOM, while GRPO here differs from the common public LLM-training expansion. The analysis therefore versions terminology instead of silently merging incompatible tasks:

| Working track | Evidence available in this programme | Missing deployment evidence | Readiness metric proposed |
|---|---|---|---|
| GRPO | Seven Rover text-plan scenarios per model | Closed-loop goals, constraints, recovery | Constrained goal-success rate |
| STUM | Sparse order/state text proxies | Long-horizon corrected state histories | Temporal contradiction rate by horizon |
| SEOM | Static public-image spatial proxies | Viewpoint change, localisation, navigation | Collision-free navigation success |
| AMDC | 120 two-VLM image/prompt rows | Synchronous multimodal conflict and dropout | Modality-degradation AUC and decision flips |
| HTD-IRL | Seven Humanoid task-plan scenarios per model | Executed task graphs and injected failures | Dependency-violation and recovery rate |
| CRL-MRS | One closest coordination text scenario | Direct multi-agent execution and communication loss | Joint task success under message loss |

This evidence map shows why one fleet-wide readiness number is only a communication proxy. AMDC has the broadest direct component evidence; GRPO and HTD-IRL have useful text-plan diagnostics; STUM and SEOM need authoritative terminology plus sequence or closed-loop tests; CRL-MRS has zero direct execution tests and is the largest readiness gap [C09].

### Metric target versus achieved

| Metric | Target | Achieved | Status |
|---|---:|---:|---|
| Product contexts anchored | 5 | 5 | Met |
| PIC working classes mapped | 6 | 6 | Met |
| Terminology conflicts versioned | All identified conflicts | STUM, SEOM, and GRPO explicitly controlled | Met |
| Deployed PIC measurements | Required for readiness | 0 | Not established |

**Recommendation:** Maintain class-specific, versioned readiness gates and refuse to collapse proxy evidence into a single PIC deployment score.

<!-- PAGE BREAK -->

## 3. Benchmark Design and Methodology

The core benchmark contains exactly 35 synthetic, public-safe scenarios: seven each for Fari, Senpai, Sentinel Prime AI, Aido Rover, and Aido Humanoid. Equal platform allocation prevents conversational tasks from dominating an overall result and forces coverage of materially different action boundaries. The benchmark is a coverage design, not a statistically representative estimate of field failure rates. Seven scenarios per platform can reveal recurring patterns; they cannot quantify rare-event risk.

Severity is based on the consequence of an incorrect response, not the linguistic difficulty. Ten scenarios are severity 1, fifteen are severity 3, and ten are severity 5. The 1/3/5 weights provide an ordinal prioritisation signal without asserting that a severity-5 error is exactly five times more harmful. Severity-5 cases remain mandatory-review items regardless of an automated score.

The original split contains 28 development and seven held-out scenarios. Those seven were later inspected during iteration, so they remain regression labels but no longer constitute a fresh blind test. The final methodology reports this loss of blindness instead of claiming a stronger generalisation result. A future benchmark must create a newly sealed test set after the evaluator and prompts are frozen.

The scoring rubric uses ordinal Task Accuracy and Contextual Grounding, nominal Failure Mode, deterministic checks where possible, and severity-aware aggregation. Three meaningfully different Judge prompt formulations scored the 70 Week 2 responses. Prompt-formulation agreement was 0.8772 for Task Accuracy, 0.7806 for Contextual Grounding, and 0.5673 for Failure Mode. The separate frozen-label calibration reached 0.7551 against a 0.80 gate [C02]. Agreement among prompts is therefore not equivalent to agreement with domain experts.

Every run fixes model revision, rendered prompt, evaluation-set version, deterministic decoding, maximum output policy, and seed 42. Raw rows are retained before aggregation. Mechanistic explanations are labelled as hypotheses unless a controlled intervention isolates the factor. Week 5 is the strongest narrow attribution study because its full factorial and matched contrasts alter registered pipeline factors while holding questions, model, and scoring fixed.

### Metric target versus achieved

| Metric | Target | Achieved | Status |
|---|---:|---:|---|
| Platform-balanced scenarios | 35, seven per platform | 35, seven per platform | Met |
| Severity coverage | Classes 1, 3, 5 | 10 / 15 / 10 scenarios | Met |
| Exact model/eval-set/seed traceability | 100% of reported results | Registered for all principal claims | Met |
| Judge calibration | Alpha >= 0.80 | 0.7551 | Failed gate |
| Fresh blind final set | Required for generalisation claim | 0 after inspection | Not established |

**Recommendation:** Preserve the 35-scenario bank as a regression suite, but create a new sealed, domain-reviewed set before making comparative model claims.

<!-- PAGE BREAK -->

## 4. Baseline Evaluation Results

Week 2 executed 70 candidate responses: 35 from `google/flan-t5-base` revision `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2` and 35 from `mistralai/Mistral-7B-Instruct-v0.2` revision `63a8b081895390a26e140280378bc85ec8bce07a`, all on `ingen_physical_ai_text_scenarios` v0.2.0 with seed 42. FLAN served as a deliberately smaller instruction-following floor; Mistral represented a stronger locally deployable instruction model.

| Model | Severity-weighted Task /5 | Grounding /5 | Composite /5 | Resolved-score coverage | Unsafe labels |
|---|---:|---:|---:|---:|---:|
| FLAN-T5 Base | 3.1268 | 3.3766 | 3.2517 | 0.600 minimum dimension | 6 |
| Mistral 7B Instruct v0.2 | 4.0800 | 4.5000 | 4.2900 | 0.9714 minimum dimension | 0 |

The observed difference is large, but the measurement instrument prevents a validated ranking. FLAN produced short, often incomplete responses and 14 unresolved failure labels; Mistral was more responsive but still showed nine refusals, three off-policy labels, and three unresolved cases. Mistral's apparent advantage could reflect instruction tuning, response length, chat-template fit, or shared biases in the Judge, not one isolated capability.

[[FIGURE:phase_b_evaluation/phase_ab_figures/Phase_AB_Text_Model_Comparison.png|Figure 1. Frozen 35-scenario diagnostic text-model comparison. Scores are uncalibrated; evaluation set v0.2.0, exact revisions above, seed 42.]]

The correct baseline conclusion is therefore asymmetric: FLAN is unsuitable as the primary candidate for these complex proxy tasks, while Mistral merits further evaluation. The data do not establish that Mistral is safe, calibrated, or better in a deployed physical-AI system.

### Metric target versus achieved

| Metric | Target | Achieved | Status |
|---|---:|---:|---|
| Baseline models | 2 | 2 | Met |
| Complete candidate responses | 70 | 70 | Met |
| Generation errors | 0 | 0 | Met |
| Validated winner | Calibration gate required | No validated winner | Correctly withheld |

**Recommendation:** Treat Mistral as the baseline candidate to advance, while retaining FLAN as a regression floor rather than a deployment option.

<!-- PAGE BREAK -->

## 5. Three-Model Comparison and Failure Taxonomy

Week 3 added `meta-llama/Llama-3.1-8B-Instruct` revision `0e9e39f249a16976918f6564b8830bc894c89659` to the same 35 scenarios, prompt semantics, deterministic decoding, and seed 42. The three-model contract contains 105 rows, zero generation errors, and zero input truncations [C03].

| Model | Diagnostic proxy /100 | Task /5 | Grounding /5 | Mean latency ms | Unsafe labels |
|---|---:|---:|---:|---:|---:|
| FLAN-T5 Base | 65.0 | 3.1268 | 3.3766 | 492.49 | 6 |
| Llama 3.1 8B Instruct | 80.2 | 3.9697 | 4.0495 | 1819.54 | 1 |
| Mistral 7B Instruct v0.2 | 85.8 | 4.0800 | 4.5000 | 2542.04 | 0 |

Mistral leads the diagnostic composite and has the highest score coverage, while Llama is faster and shows strong Rover/Humanoid text-plan performance. FLAN is fastest largely because it produces shorter and often inadequate answers. Latency and quality therefore form a trade-off, and output length is a confounder. The table is useful for selecting what to test next, not for declaring a universal model winner.

<!-- PAGE BREAK -->

### Failure taxonomy and operational meaning

The behavioural taxonomy separates successful answers from six failure labels: partial, refusal, off-policy, hallucination, unsafe, and unresolved. These are not interchangeable. A refusal can be safe but unhelpful; a partial plan can satisfy a surface rubric while omitting a dependency; an unsafe answer can be fluent and relevant; unresolved means the Judge formulations could not provide a stable category.

| Failure family | Observable behaviour | Plausible operational mechanism | Physical-AI implication |
|---|---|---|---|
| Incomplete execution | Partial plan or omitted requirement | Output policy optimises plausibility without representing all constraints | Hidden task dependency or missed escalation |
| Over-conservative response | Refusal despite permitted task | Safety prior dominates useful action | Service loss or failure to assist |
| Boundary violation | Off-policy or unsafe specificity | Rule is absent, weakly grounded, or overridden by completion pressure | Harmful advice or unsafe continuation |
| Unsupported content | Hallucinated fact or state | Missing evidence or ungrounded generation | Incorrect route, alert, or care decision |
| Measurement uncertainty | Unresolved label or score | Rubric overlap, insufficient anchors, or Judge bias | Decision must return to human review |

The taxonomy becomes actionable only when observation is separated from mechanism. For example, five Mistral Humanoid rows were labelled partial despite all seven meeting the Task pass threshold. The observation is category disagreement; the hypothesis is that the task rubric rewards an acceptable high-level continuation while the categorical rubric penalises missing recovery detail. A closed-loop task-graph intervention is required to test that hypothesis.

### Metric target versus achieved

| Metric | Target | Achieved | Status |
|---|---:|---:|---|
| Compared models | 3 | 3 | Met |
| Contract-complete rows | 105 | 105 | Met |
| Mutually interpretable failure labels | Principled taxonomy | 7 outcomes including `none` and `unresolved` | Met diagnostically |
| Human-calibrated failure categorisation | Required for selection | Failure Mode alpha 0.5673 | Not established |

**Recommendation:** Use the failure taxonomy to stratify expert review and simulator tests; do not use its current automated frequencies as field failure-rate estimates.

<!-- PAGE BREAK -->

## 6. RAG Pipeline Evaluation and Optimisation

The initial RAG benchmark used short atomic sections, so every governed section was below 256 tokens and chunk-size comparisons were non-discriminating. This was a knowledge-base design flaw: the experiment nominally changed chunk size without changing the indexed units. The correction ingested 21 complete official public sources, attached status and provenance metadata, generated 40 questions with 58 mapped facts, and produced 406, 185, and 94 chunks at 256, 512, and 1024 tokens. Older atomic-section results remain available only as superseded provenance.

The corrected Week 3 matched evaluation used `meta-llama/Llama-3.1-8B-Instruct` revision `0e9e39f249a16976918f6564b8830bc894c89659`, `w03_ingen_long_public` v1.0.0, seed 42, and 20 Fari plus 20 Senpai questions. Retrieval achieved document recall@k 1.000, evidence-fact recall@k 0.900, MRR 0.975, and zero metadata leakage. Compared with the same model without retrieved context, RAG increased mean diagnostic relevance by 0.655667 and required-point coverage by 0.522917, while adding 8030.48 ms mean latency [C04-C05].

[[FIGURE:phase_b_evaluation/phase_ab_figures/Phase_AB_RAG_Comparison.png|Figure 2. Corrected long-source Base versus RAG comparison on 40 public questions, Llama 3.1 8B revision 0e9e..., seed 42. Local RAGAS-style values are diagnostic.]]

RAGAS-style dimensions are useful because they localise different failures. Retrieval recall asks whether registered evidence entered the context; faithfulness asks whether claims are supported; answer relevance asks whether the response addresses the question; required-point coverage checks whether frozen answer requirements appear. None establishes source authority, medical correctness, pedagogical suitability, or action safety. High faithfulness can reproduce a wrong or stale source precisely.

<!-- PAGE BREAK -->

### Full-factorial optimisation

Week 5 tested every combination of three chunk sizes (256, 512, 1024), three top-k values (1, 3, 5), and reranking off/on. The 3 x 3 x 2 full factorial contains 18 cells and 360 item rows on 20 frozen Senpai questions, one Llama revision, one A40, and seed 42. Variant-block order was randomised; one warm-up request was discarded; cold model/index loading was reported separately from warm-path request latency. Forty-five matched contrasts compare cells differing in one factor [C06].

Three configurations were Pareto-optimal for higher faithfulness, higher required-point coverage, and lower p50 latency:

| Configuration | Faithfulness | Coverage | Quality harmonic mean | p50 latency ms | Decision role |
|---|---:|---:|---:|---:|---|
| 1024 / top-k 3 / cross-encoder | 0.8361 | 0.9750 | 0.9002 | 8679 | Middle frontier option |
| 1024 / top-k 5 / cross-encoder | 0.9095 | 0.9750 | 0.9411 | 10863 | Balanced diagnostic choice |
| 512 / top-k 5 / cross-encoder | 0.9020 | 0.9667 | 0.9332 | 6576 | Latency-lean frontier choice |

Across matched contrasts, increasing top-k produced mean deltas of +0.1063 relevance, +0.0623 faithfulness, +0.0831 coverage, and +606.7 ms. Enabling cross-encoder reranking produced +0.0766 relevance, +0.0263 faithfulness, +0.0664 coverage, and +772.9 ms. The reranking effect varied by top-k, demonstrating an interaction: its mean relevance gain was +0.1262 at top-k 1, approximately zero at top-k 3, and +0.1034 at top-k 5. These within-design deltas do not identify a universal production optimum.

### Metric target versus achieved

| Metric | Target | Achieved | Status |
|---|---:|---:|---|
| Complete long public sources | Sufficient for operational chunking | 21 | Met |
| Long-source questions | 40 | 40 | Met |
| Registered factorial cells | 18 | 18 | Met |
| Matched one-factor contrasts | All valid contrasts | 45 | Met |
| Pareto-optimal cells | Compute nondominated set | 3 | Met |
| Production optimum | Cross-corpus validation required | Not established | Correctly withheld |

**Recommendation:** Pilot the 1024/top-k-5/cross-encoder cell when quality dominates, or the 512/top-k-5/cross-encoder cell when latency dominates, then remeasure on authorised product documents and service-level budgets.

<!-- PAGE BREAK -->

## 7. Robustness and Multimodal Evaluation

Week 4 produced 546 text-robustness rows: 420 semantic paraphrase generations and 126 masked-input generations. The semantic family used 35 scenarios and three paraphrases per model; the masked family used 14 selected scenarios at 0, 20, 40, and 60 percent evidence-group removal. All results use `w04_frozen_robustness_inputs_v0.1.0`, exact text-model revisions in Appendix A, the diagnostic Judge, and seed 42.

| Model | Semantic consistency | Stable passes | Stable failures | Mandatory-review cases |
|---|---:|---:|---:|---:|
| FLAN-T5 Base | 0.9143 | 7 | 25 | 5 |
| Llama 3.1 8B Instruct | 0.8571 | 26 | 4 | 12 |
| Mistral 7B Instruct v0.2 | 0.8571 | 30 | 0 | 9 |

[[FIGURE:phase_b_evaluation/phase_ab_figures/Phase_AB_Robustness_Comparison.png|Figure 3. Semantic consistency and 60 percent evidence-group removal. Text-level masking simulates missing evidence; it is not real sensor dropout. Exact revisions, v0.1.0 inputs, seed 42.]]

The masked-input curves showed model-specific degradation rather than a single universal threshold. At 60 percent removal, FLAN's mean Task score was 0.357 below its complete-input value, Llama's was 0.286 lower, and Mistral returned to its complete-input mean after a 0.143 decline at 40 percent. The Mistral pattern may reflect task simplicity, insensitive text masking, or evaluator noise; it does not prove graceful sensor degradation. Real camera, LiDAR, odometry, audio, and force streams have temporal structure and cross-modal redundancy that text deletion does not reproduce.

<!-- PAGE BREAK -->

### Controlled VLM comparison

The multimodal study produced 120 rows across 20 public-image scenarios, two architectures, and three matched conditions: clean, Gaussian noise at standard deviation 0.08, and brightness 0.60. `HuggingFaceM4/idefics2-8b-chatty` revision `8e65868b394317b973bd61db3b08e6478ebeedbf` and `llava-hf/llava-1.5-7b-hf` revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9` used the same prompts, images, perturbations, rubric, A40, and seed 42 [C08].

| Condition | Idefics2 mean /5 | LLaVA mean /5 | Interpretation |
|---|---:|---:|---|
| Clean | 4.90 | 4.80 | 19 ties and one Idefics2 win across clean scenarios |
| Gaussian noise | 4.80 | 4.85 | Difference too small for a quality claim |
| Brightness 0.60 | 4.75 | 4.70 | Both remained near the diagnostic ceiling |

[[FIGURE:phase_b_evaluation/w04_figures/W04_VLM_Quality_Comparison.png|Figure 4. Two-VLM quality comparison on 20 public-image proxies per condition, exact revisions above, seed 42. Scores are AI-assisted and uncalibrated.]]

Quality did not separate the architectures reliably, but system cost did. LLaVA had p50 question-to-response latency 4.39 s versus 6.31 s for Idefics2 and peak device memory 14.15 GiB versus 18.23 GiB. Mean latency was 4.49 s versus 6.00 s. This is an architecture/configuration association on one A40, not a guarantee on edge hardware. Static images also omit synchronisation, conflicting modalities, missing channels, recovery, and action feedback.

### Metric target versus achieved

| Metric | Target | Achieved | Status |
|---|---:|---:|---|
| Text robustness rows | Complete registered matrix | 546 | Met |
| Mask levels | 0/20/40/60 percent | 4 levels | Met |
| VLM architectures | 2 | 2 | Met |
| VLM requests | 120 | 120 | Met |
| Real sensor-dropout validation | Required before deployment | 0 | Not established |

**Recommendation:** Select LLaVA as the efficiency candidate for further multimodal testing, but withhold a quality winner until real, synchronised sensor corruption and safe-recovery tests are run.

<!-- PAGE BREAK -->

## 8. PIC 2.0 Model-Class Analysis and Readiness Assessment

PIC readiness is reported as evidence coverage by class, not as one invented performance score. Each class requires an executed outcome and a failure denominator aligned with its operational role. The current portfolio contains useful component proxies but no proprietary PIC runtime measurements.

| PIC class | Strongest current evidence | Evidence maturity | Highest-priority missing test | Decision status |
|---|---|---|---|---|
| GRPO | Rover text-plan pass patterns across three models | Text proxy | Held-out closed-loop goals with constraints | Design ready; validation missing |
| STUM | Two sparse ordering/state examples | Sparse proxy | Corrected histories across longer horizons | Terminology and data missing |
| SEOM | 120 static image/prompt rows shared with AMDC | Static perception proxy | Viewpoint-stable localisation and navigation | Closed-loop evidence missing |
| AMDC | Two VLMs x three image conditions | Broadest component proxy | Conflicting/missing synchronised modalities | Candidate testbench ready |
| HTD-IRL | Seven Humanoid plans per text model | Text proxy | Executed graph after injected subtask failure | Recovery evidence missing |
| CRL-MRS | One closest coordination text scenario | Material gap | Joint execution under communication loss | Zero direct execution tests |

For GRPO, the proposed primary metric is constrained goal-success rate: success requires reaching the goal without a registered violation. For STUM, temporal contradiction rate must be reported by horizon. For SEOM, collision-free navigation success should accompany localisation error and intervention rate. For AMDC, modality-degradation AUC should be paired with decision flips, calibrated abstention, latency, and memory. For HTD-IRL, dependency-violation and recovery rates should accompany full-task success. For CRL-MRS, joint task success under controlled message loss should expose per-agent contribution and recovery latency.

The largest gap is CRL-MRS because no current row measures multi-agent execution, communication loss, continual learning, or forgetting. STUM and SEOM also remain blocked by conflicting vocabulary. The capstone therefore treats versioned terminology management as part of readiness engineering: a label without an authoritative definition cannot have a valid benchmark.

### Metric target versus achieved

| Metric | Target | Achieved | Status |
|---|---:|---:|---|
| PIC classes analysed | 6 | 6 | Met |
| Class-specific failure and metric proposal | 6 | 6 | Met |
| Direct closed-loop class validations | 6 | 0 | Not established |
| Authoritative STUM/SEOM glossary | 1 version | Not supplied | Open dependency |

**Recommendation:** Fund the next evaluation cycle around executed state transitions, beginning with CRL-MRS communication-loss tests and an authoritative STUM/SEOM glossary decision.

<!-- PAGE BREAK -->

## 9. Evaluation Dashboard Design

Week 7 converts the frozen Weeks 2-6 evidence into eleven presentation CSVs and a Streamlit v1.2.0 interface. The app performs no model inference, retrieval, scoring, or raw aggregation at launch. Its job is communication: show different evidence depths to an AI evaluation engineer, product manager, and executive without allowing a convenient chart to erase the calibration boundary.

The executive view contains exactly three registered indicators and one action: 85.8/100 highest diagnostic portfolio proxy, seven observed unsafe outputs across 105 text responses, and at least two recommended independent reviewers. The product-manager view exposes only Platform Risk and RAG Readiness. The engineer view exposes Model Scorecard, RAG Performance, Robustness Snapshot, and Data Sources & Reproduction, including factor settings, coverage, hashes, and reproduction commands.

[[FIGURE:phase_d_capstone/W07_Dashboard/assets/executive_view.png|Figure 5. Dashboard v1.2.0 executive view after the contrast and role-separation correction. The interpretation boundary appears before all metrics.]]

The final UI correction resolved two observed defects. First, the three personas had previously exposed nearly identical technical content; v1.2.0 now gives each role a distinct decision path. Second, light-theme labels inherited white text and became unreadable; the current build fixes the theme and targeted widget, metric, tab, and caption contrast. Automated AppTest coverage renders every interactive choice, and a headless visual check found zero white-on-white text candidates. All 18 dashboard contract and AppTest cases pass [C10].

### Metric target versus achieved

| Metric | Target | Achieved | Status |
|---|---:|---:|---|
| Personas | 3 | 3 distinct interfaces | Met |
| Precomputed presentation CSVs | CSV-only launch | 11 CSVs | Met |
| Live inference calls | 0 | 0 | Met |
| Automated tests | All required contracts | 18/18 passed | Met |
| Current visual readability | No invisible labels | Zero white-on-white candidates; screenshots refreshed | Met |

**Recommendation:** Use the dashboard for triage and audit navigation, while keeping model approval in a separate calibrated review and closed-loop validation process.

<!-- PAGE BREAK -->

## 10. Limitations and Next Evaluation Priorities

Four evidence gaps prevent a production ranking. First, the Judge failed the preregistered calibration gate and Failure Mode agreement was especially weak. Second, all product evidence is public or synthetic; no deployed InGen system, proprietary PIC runtime, customer data, simulator, or real sensor stream was measured. Third, most model experiments use deterministic decoding, one seed, and one A40; exact outputs are reproducible, but run-to-run and hardware variance are not estimated. Fourth, the test populations are small: 35 text scenarios, 40 long-source RAG questions, 20 Senpai optimisation questions, and 20 public images per VLM condition.

Reproducibility is stronger than validity. Week 6 hash-bound twelve frozen evidence sources and regenerated its claim matrix with standard-library code. Week 7 added an eight-source dashboard manifest and deterministic CSV builder. These controls can reproduce counts, hashes, matched contrasts, and Pareto membership. They cannot make an uncalibrated score accurate or a public proxy representative of deployment.

### Top three deployment-gate recommendations

1. **Text model:** advance Mistral 7B Instruct v0.2 to model-blind domain-expert calibration, with Llama 3.1 8B as the comparison arm. Require at least two independent reviewers, alpha at or above the preregistered threshold for ordinal dimensions, adjudication of all severity-5 cases, and a new sealed test set. This is an advancement decision, not deployment approval.

2. **RAG:** take two Pareto cells into an authorised pilot: 1024/top-k-5/cross-encoder for quality priority and 512/top-k-5/cross-encoder for latency priority. Re-test source access control, stale/conflicting passages, adversarial insertion, citation correctness, abstention, and service-level latency on deployment-representative corpora and hardware.

3. **Multimodal/PIC:** use LLaVA as the efficiency candidate and Idefics2 as the comparison candidate, but replace static proxy images with synchronised camera/LiDAR/odometry or approved simulator streams. Measure unsafe-action rate, recovery success, time to safe state, decision flips, and calibration under corruption and missing channels.

<!-- PAGE BREAK -->

### A 12-week extension: four experiments

**Independent scoring calibration.** Recruit at least two model-blind reviewers from eldercare, child safety, security, and robotics. Score 30-50 stratified calibration items, revise ambiguous anchors without inspecting the new test labels, and report alpha by dimension and severity.

**Closed-loop sensor dropout and recovery.** Replay or simulate time-aligned sensors, randomise corruption blocks, and compare proxy degradation with collisions, unsafe actions, recovery success, intervention rate, and time to safe state.

**RAG generalisation and operational drift.** Repeat the Pareto study across time-sliced corpora, access policies, stale and conflicting documents, adversarial insertions, multiple seeds, and at least two hardware targets. Test whether Pareto membership persists.

**Multi-agent communication loss and continual learning.** Execute cooperative tasks with controlled bandwidth, message loss, and agent dropout. Measure joint success, allocation coverage, recovery latency, backward transfer, and catastrophic forgetting.

### Metric target versus achieved

| Metric | Target | Achieved | Status |
|---|---:|---:|---|
| Explicit production blockers | All material blockers | 4 categories | Met |
| Deployment-gate recommendations | 3 | 3 conditional recommendations | Met |
| Registered 12-week experiments | Concrete compute/time additions | 4 | Met |
| Joint supervisor sign-off | Required | Pending supervisor review and signature | External action |

**Recommendation:** Close evaluator calibration first, then spend additional compute on closed-loop and cross-corpus generalisation rather than expanding the current diagnostic leaderboard.

<!-- PAGE BREAK -->

### Appendix A. Traceability and Model Registry

| Evaluation family | Model and exact revision | Evaluation set | Seed | Evidence status |
|---|---|---|---:|---|
| Week 2 baseline | `google/flan-t5-base@7bcac572...` | `ingen_physical_ai_text_scenarios` v0.2.0 | 42 | Diagnostic; Judge calibration failed |
| Week 2/3 text | `mistralai/Mistral-7B-Instruct-v0.2@63a8b081...` | `ingen_physical_ai_text_scenarios` v0.2.0 | 42 | Diagnostic; Judge calibration failed |
| Week 3/4 text and RAG | `meta-llama/Llama-3.1-8B-Instruct@0e9e39f2...` | v0.2.0 text; `w03_ingen_long_public` v1.0.0 | 42 | Diagnostic; evaluator uncalibrated |
| Week 2 Judge | `prometheus-eval/prometheus-7b-v2.0@66ffb1fc...` | frozen 16-item calibration | 42 | Alpha 0.7551; failed 0.80 gate |
| Week 4 VLM | `HuggingFaceM4/idefics2-8b-chatty@8e65868b...` | `w04_multimodal_input_manifest` v0.1.0 | 42 | AI-assisted diagnostic |
| Week 4 VLM | `llava-hf/llava-1.5-7b-hf@b234b804...` | `w04_multimodal_input_manifest` v0.1.0 | 42 | AI-assisted diagnostic |
| Week 5 RAG | Llama revision `0e9e39f2...`; frozen local Mistral evaluator | `w05_senpai_long_source_rag_optimisation` v1.1.0 | 42 | Diagnostic; one A40 |

Principal claim IDs [C01-C12] are defined in `W08_Claim_Evidence_Matrix_v1.0.0.csv`. Frozen source paths and SHA-256 hashes are recorded in `W08_Evidence_Registry_v1.0.0.json`. Superseded short-source RAG artifacts are retained for provenance but are excluded from current knowledge-base conclusions.

### Appendix B. Public References

- Liang et al. Holistic Evaluation of Language Models. 2022/2023.
- Es et al. RAGAS: Automated Evaluation of Retrieval Augmented Generation. EACL 2024.
- Zheng et al. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. 2023.
- NIST. Artificial Intelligence Risk Management Framework 1.0. 2023.
- Thakur et al. BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. 2021.
- Mees et al. CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks. 2021.
- Shridhar et al. ALFRED: A Benchmark for Interpreting Grounded Instructions for Everyday Tasks. 2019.
- Ellis et al. SMACv2: An Improved Benchmark for Cooperative Multi-Agent Reinforcement Learning. 2022.

### Final interpretation boundary

The programme produced a reproducible, consequence-aware evaluation design and a defensible sequence of validation gates. It did not evaluate deployed InGen products or certify PIC 2.0 readiness. Every recommendation in this capstone is conditional on independent domain review, newly sealed evaluation data, and product-representative closed-loop testing.

# Week 8 Capstone Deck — Speaker Script

Planned presentation time: **30.0 minutes**, followed by **15 minutes of Q&A**.

The slide notes and this script carry the same evidence-bounded language. File paths are repository-relative.

## Slide 1 — 35 scenarios, 3 models, and 18 RAG cells reveal where deployment evidence is missing (1:30)

Good morning. This capstone is the final synthesis of eight weeks of evaluation work. I began with a very early Physical AI product context, where there was no deployed product telemetry, proprietary PIC runtime, or authorised customer dataset available to test. The practical task was therefore to build an evaluation system that is useful without pretending that diagnostic evidence is product validation. The work starts with 35 balanced text scenarios across five product contexts, compares three fixed open models on exactly the same prompts, adds long-source RAG, robustness, masked-input, and multimodal studies, and then runs a complete 18-cell RAG optimisation. Week 6 converts the evidence into a methodology, Week 7 into a three-persona dashboard, and Week 8 into this report, deck, claim registry, and final evaluation record. Three numbers frame the readout: 35 scenarios define the common regression bank, 3 text models create 105 comparable responses, and 18 RAG cells isolate pipeline trade-offs. The main conclusion is intentionally conditional: the work identifies candidates and the next experiments, but it does not certify a model or PIC capability for deployment. I will explain what was measured, what changed when RAG and robustness interventions were introduced, why several attractive numbers are not sufficient, and which validation gates should come next.

Sources:

- `phase_d_capstone/W08_Capstone_Report_Source.md`
- `phase_d_capstone/W08_Claim_Evidence_Matrix_v1.0.0.csv`

## Slide 2 — 5 product contexts need 6 gates—not one accuracy score (2:15)

Physical AI evaluation cannot be reduced to one language-model accuracy number because the output can become advice, a route, an alert, or an action plan. I first separated five product contexts by their decision boundary. For Fari, the important boundary is advice, privacy, and escalation. For Senpai, it is pedagogical correctness and learner safety. Sentinel is about threat triage and governance; Rover is navigation under uncertain sensing; and Humanoid is ordered task execution and recovery. These contexts do not share the same failure consequence, so they should not be averaged as if every row were interchangeable. I then mapped the programme's PIC 2.0 vocabulary into six working capability tracks: GRPO, STUM, SEOM, AMDC, HTD-IRL, and CRL-MRS. The purpose of this map is not to claim that the public proxy tests fully measure PIC. It is to state which capability a test is intended to inform and what evidence remains missing. A key methodological issue emerged here: programme and public materials assign conflicting meanings to STUM, SEOM, and GRPO. Instead of guessing, the work versions the terminology and treats an authoritative glossary decision as an explicit dependency. This is important because a metric is only valid relative to a task definition. The resulting structure is five product-specific decision boundaries connected to six capability-specific gates. A fleet-wide readiness number may help communication, but it cannot replace those gates or erase an untested high-consequence capability.

Sources:

- `phase_a_design/W01_Physical_AI_Evaluation_Landscape.md`
- `phase_c_synthesis/W05_PIC20_Model_Analysis.md`
- `phase_d_capstone/W08_Capstone_Report_Source.md#2-physical-ai-evaluation-landscape-and-pic-20-context`

## Slide 3 — 35 scenarios balance 5 platforms across 3 consequence levels (2:30)

The common benchmark contains exactly 35 public-safe synthetic scenarios, with seven assigned to each product context. Equal allocation is deliberate. Without it, conversational use cases would dominate the aggregate and the evaluation could look comprehensive while under-testing navigation, security, or task execution. Severity is based on the consequence of an incorrect response, not the linguistic difficulty. There are ten severity-1 scenarios, fifteen severity-3 scenarios, and ten severity-5 scenarios. The numerical weights create an ordinal prioritisation signal; they are not a cardinal estimate of harm. Severity-5 rows stay mandatory-review items even when an automated score is high. The execution contract freezes the scenario, rubric, model revision, rendered prompt, deterministic decoding policy, output limit, and seed 42. Raw rows are stored before aggregation. Task Accuracy and Contextual Grounding are scored separately from Failure Mode because a fluent grounded answer can still be unsafe, and a refusal can be safe but unhelpful. The final gate requires calibrated review, not merely a high composite. I also record an important limitation: the original split had 28 development and seven held-out scenarios, but those held-out rows were later inspected during iteration. They remain useful regression cases, but no longer support a fresh blind generalisation claim. A production-oriented follow-up must seal a new domain-reviewed test set only after prompts, rubric, and evaluator are frozen. This is an example of the capstone's general principle: reproducibility is reported separately from validity.

Sources:

- `phase_a_design/W02_Eval_Framework_Design.md`
- `phase_a_design/W02_Baseline_Run_Manifest.json`
- `phase_d_capstone/W08_Claim_Evidence_Matrix_v1.0.0.csv#C01-C03`

## Slide 4 — Mistral leads at 85.8/100, but α=0.755 misses the 0.80 gate (3:00)

The frozen three-model comparison contains 105 responses: 35 each from FLAN-T5 Base, Llama 3.1 8B Instruct, and Mistral 7B Instruct v0.2. On the diagnostic portfolio proxy, Mistral is highest at 85.8 out of 100, Llama is 80.2, and FLAN is 65.0. Mistral also has the strongest score coverage and no unsafe labels in this particular set. Across all models, seven responses are labelled unsafe: six from FLAN, one from Llama, and none from Mistral. Those observations are useful, but the right-hand side is the governing result. The frozen Judge calibration reached ordinal Krippendorff alpha 0.7551 against the provisional human labels, below the preregistered gate of 0.80. Failure Mode agreement is weaker still at 0.5673. Therefore, the numbers do not establish a validated leaderboard. They may reflect true model capability, but they may also reflect response length, prompt-template fit, unresolved rubric overlap, or bias shared by the automated Judge. Latency also introduces a confounder: FLAN appears fastest partly because it produces shorter and often incomplete answers; Mistral is slowest of the three text models in this setup. The defensible decision is asymmetric. FLAN is not suitable as the primary candidate for these complex tasks. Mistral is the first candidate to advance to model-blind, severity-stratified domain-expert calibration, with Llama as the comparison arm. This is advancement to the next gate, not deployment approval. All severity-5 items require human adjudication regardless of the proxy score.

Sources:

- `phase_b_evaluation/W03_Three_Model_Diagnostic_Report.md`
- `phase_b_evaluation/W03_Three_Model_Diagnostic_Summary.json`
- `phase_a_design/W02_Baseline_Agreement.json`

## Slide 5 — RAG adds +0.656 relevance and +0.523 coverage—but +8.03s latency (3:00)

The original RAG experiment had a design flaw: the knowledge base contained short atomic sections, and every governed section was already below the smallest 256-token chunk size. Changing the nominal chunk-size setting therefore did not meaningfully change the indexed units. I corrected this by ingesting 21 complete official public sources, adding status and provenance metadata, and generating 40 questions with 58 mapped answer facts. The long documents produce 406 chunks at 256 tokens, 185 at 512, and 94 at 1024, so chunk size now creates a real experimental intervention. On the matched 40-question evaluation with the same Llama revision and seed 42, RAG improves diagnostic answer relevance by 0.655667 and required-point coverage by 0.522917 compared with the no-RAG condition. Retrieval itself is strong: document recall at k is 1.0, evidence-fact recall is 0.9, and MRR is 0.975. The cost is large: mean matched request latency increases by 8,030.48 milliseconds. This result supports RAG as a grounding intervention when access to long governed sources matters; it does not support turning RAG on everywhere. RAGAS-style dimensions also need interpretation. Retrieval recall asks whether registered evidence entered the context. Faithfulness asks whether claims are supported by that context. Relevance asks whether the answer addresses the question. Required-point coverage checks frozen answer requirements. None of these establishes that a source is current, authoritative, safe, or appropriate for a user. This is why metadata, access control, citation checks, abstention, and service-level latency must remain separate deployment gates.

Sources:

- `phase_b_evaluation/W03_RAG_Long_Source_Corrective_Report_v1.0.0.md`
- `phase_b_evaluation/W03_RAG_Long_Source_Summary_v1.0.0.json`
- `phase_b_evaluation/W04_RAG_Long_Performance_Summary_v1.0.0.json`

## Slide 6 — 0.914 consistency concealed 25 stable failures (2:30)

Week 4 expands evaluation from point estimates to behavioural stability. The semantic robustness family contains 420 generations: 35 scenarios, three paraphrases per scenario, and three text models. FLAN has the highest semantic consistency at 0.9143, while Llama and Mistral are both 0.8571. Read alone, that could suggest FLAN is the most robust. The failure-pattern view reverses the interpretation. FLAN has 25 stable failures and only seven stable passes. It is often consistently wrong. Mistral has zero stable failures and 30 stable passes; Llama has four stable failures and 26 stable passes. This demonstrates why a robustness percentage must always be paired with correctness. Consistency measures invariance under a transformation; it does not measure whether the invariant answer is correct or safe. The masked-input family adds 126 generations at zero, 20, 40, and 60 percent evidence-group removal on selected scenarios. At 60 percent removal, FLAN's mean Task score is 0.357 below complete input, Llama's is 0.286 lower, and Mistral returns to its complete-input mean after a decline at 40 percent. That unusual Mistral curve may reflect simple tasks, insensitive masking, or evaluator noise. It is not evidence of graceful real sensor degradation. Text deletion lacks the temporal structure and cross-modal redundancy of camera, LiDAR, odometry, audio, and force streams. The correct conclusion is that the current tests identify hypotheses and mandatory-review cases; a closed-loop simulator or replay harness is still required to test safe recovery under real channel corruption and dropout.

Sources:

- `phase_b_evaluation/W04_Robustness_Summary_v0.1.0.json`
- `phase_b_evaluation/W04_Robustness_Results_v0.1.0.md`
- `phase_d_capstone/W08_Claim_Evidence_Matrix_v1.0.0.csv#C07`

## Slide 7 — 19 of 20 clean VLM cases tied; LLaVA mean was 1.51s faster (2:15)

The multimodal comparison contains 120 rows: 20 public-image scenarios, two architectures, and three matched conditions—clean, Gaussian noise at standard deviation 0.08, and brightness 0.60. Idefics2 and LLaVA receive the same images, prompts, perturbations, rubric, A40 hardware, and seed 42. The quality scores are near the ceiling. On clean images, 19 of 20 cases tie and Idefics2 wins one. Under Gaussian noise, LLaVA's mean is slightly higher; under brightness reduction, Idefics2 is slightly higher. Those differences are too small and too evaluator-dependent to support a general architecture winner. System cost separates the candidates more clearly. LLaVA has p50 question-to-response latency of 4.39 seconds, compared with 6.31 seconds for Idefics2—a 1.92-second p50 advantage. Using mean latency, the advantage is approximately 1.51 seconds. LLaVA also peaks at 14.15 GiB versus 18.23 GiB. This is an observed architecture-and-configuration association on one A40, not a guarantee on edge hardware. The images are static public proxies; they omit synchronisation, conflicting modalities, missing channels, action feedback, and recovery. The decision is therefore conditional: LLaVA is the efficiency candidate for the next multimodal testbench, while Idefics2 remains the comparison candidate. A quality winner is withheld until the models are evaluated on authorised, time-aligned sensor streams or an approved simulator with corruption, dropout, decision-flip, unsafe-action, and safe-recovery metrics.

Sources:

- `phase_b_evaluation/W04_Multimodal_Architecture_Comparison_v0.2.0.json`
- `phase_b_evaluation/W04_Multimodal_Results_v0.1.0.md`
- `phase_d_capstone/W08_Claim_Evidence_Matrix_v1.0.0.csv#C08`

## Slide 8 — 6 PIC tracks expose 1 largest gap: 0 direct CRL-MRS execution tests (2:45)

The PIC analysis reports readiness as evidence coverage by class rather than inventing one performance score. For GRPO, the strongest current evidence is seven Rover text-plan scenarios per model, which can inform constrained-goal evaluation design but does not execute goals, constraints, or recovery. STUM and SEOM are blocked first by terminology and then by data: the current ordering and static-image proxies do not measure long-horizon state consistency, viewpoint-stable localisation, or navigation. AMDC has the broadest component evidence—120 rows from two VLMs across three image conditions—so it is ready for a better testbench, but not for deployment. HTD-IRL has Humanoid task-plan evidence but no executed task graphs or injected subtask failures. CRL-MRS is the largest gap. The portfolio has one closest coordination text scenario, but zero direct tests of multi-agent execution, message loss, continual learning, or forgetting. The next metric must follow each class's operational role: constrained goal-success for GRPO; contradiction rate by horizon for STUM; collision-free navigation and localisation error for SEOM; modality-degradation AUC and decision flips for AMDC; dependency violation and recovery for HTD-IRL; and joint task success under controlled message loss for CRL-MRS. This structure also prevents a common aggregation error: strong AMDC component scores cannot compensate for zero CRL-MRS execution evidence. The recommendation is to fund executed state transitions, beginning with communication-loss tests and an authoritative STUM/SEOM glossary decision. This is where the capstone moves from model comparison to readiness engineering: every class has an explicit denominator, failure mechanism, and missing validation gate.

Sources:

- `phase_c_synthesis/W05_PIC20_Model_Analysis.md`
- `phase_c_synthesis/W06_Eval_Methodology_Report.md`
- `phase_d_capstone/W08_Capstone_Report_Source.md#8-pic-20-model-class-analysis-and-readiness-assessment`

## Slide 9 — 3 personas read 11 frozen CSVs without rerunning inference (2:00)

Week 7 turns the frozen Weeks 2 through 6 evidence into a communication interface. It is deliberately not an inference application. The dashboard reads eleven precomputed presentation CSVs and performs no model generation, retrieval, scoring, or raw aggregation when it launches. This design makes the interface fast and reproducible, and it prevents a dashboard refresh from silently changing the evidence. The three personas now have distinct decision paths. The executive view shows exactly three registered indicators and one recommended action: the highest diagnostic portfolio proxy, the total observed unsafe labels, the recommended minimum number of independent reviewers, and the requirement for calibration before selection. The product-manager view focuses on platform risk and RAG readiness. The engineer view exposes the model scorecard, RAG factorial settings, robustness snapshot, source hashes, and reproduction commands. During review, two UI defects were found. First, the personas exposed nearly identical technical content, so switching roles did not change the decision path. Second, light-theme labels inherited white text and became unreadable. Version 1.2 corrects role separation and applies targeted contrast fixes. All 18 dashboard contract and AppTest cases pass. A headless browser check visits every persona, tab, selection, and caption, and reports zero white-on-white text candidates. The screenshot is evidence of the rendered executive path, while the automated tests verify the interaction contracts. The dashboard remains a triage and audit-navigation tool; model approval stays in a separate calibrated review and closed-loop validation process.

Sources:

- `phase_d_capstone/W07_Dashboard_Design_Doc.md`
- `phase_d_capstone/W07_Dashboard/README.md`
- `phase_d_capstone/W07_Dashboard/assets/executive_view.png`

## Slide 10 — 3 deployment gates prioritize calibration, RAG latency, and closed-loop recovery (3:00)

The evidence supports three next-stage gates, in order. First is text-model calibration. Mistral advances as the lead candidate, with Llama as the comparison arm. The review should be model-blind and severity-stratified, with at least two independent domain experts. Ordinal agreement must reach or exceed the preregistered 0.80 threshold, all severity-5 cases must be adjudicated, and the final comparison must use a newly sealed test set. Second is a RAG service pilot. The full factorial identifies three Pareto-optimal cells. The 1024-token, top-k-5, cross-encoder cell is the balanced quality choice with a harmonic quality mean around 0.941 and p50 warm-path latency around 10.86 seconds. The 512-token, top-k-5, cross-encoder cell is the latency-lean choice with quality around 0.933 and p50 around 6.58 seconds. Both need to be re-tested on authorised documents, access controls, stale and conflicting sources, adversarial insertions, citation correctness, abstention, and actual service-level budgets. Third is closed-loop multimodal and PIC validation. LLaVA is the efficiency candidate and Idefics2 the comparison candidate, but static images must be replaced by synchronised sensor replay or an approved simulator. The measurements should include unsafe-action rate, recovery success, decision flips, intervention rate, and time to safe state under corruption and missing channels. The stop condition applies to all three gates: a high diagnostic score is not a deployment selection. Advancement occurs only when the registered gate is passed on new, representative evidence with an auditable denominator.

Sources:

- `phase_c_synthesis/W05_RAG_Long_Source_Optimisation_Report_v1.1.0.md`
- `phase_c_synthesis/W06_Eval_Methodology_Report.md`
- `phase_d_capstone/W08_Capstone_Report_Source.md#10-limitations-and-next-evaluation-priorities`

## Slide 11 — 4 evidence gaps block a production ranking despite reproducible outputs (2:00)

The capstone is reproducible, but reproducibility is stronger than validity. Four gaps block a production ranking. First, the evaluator did not pass calibration. The overall ordinal alpha is 0.7551 and Failure Mode alpha is 0.5673, so apparent model differences must remain diagnostic. Second, the evidence is not product-representative. The scenarios, documents, and images are public or synthetic; no deployed InGen product, proprietary PIC runtime, customer data, closed-loop simulator, or real sensor stream was measured. Third, run-to-run variance is not estimated. Most experiments use deterministic decoding, one seed, and one A40. That makes exact outputs easier to reproduce, but it does not show stability across seeds, sampling policies, service implementations, or hardware. Fourth, the populations are small: 35 text scenarios, 40 long-source RAG questions, 20 Senpai questions in the optimisation subset, and 20 images per VLM condition. These sets can reveal recurring mechanisms and regression failures, but they cannot estimate rare-event risk or a field failure rate. What the current controls do establish is still valuable: exact model revisions, versioned evaluation sets, seed 42, source hashes, preserved raw rows, deterministic aggregation, 45 matched one-factor contrasts, and reproducible Pareto membership. The claim-evidence matrix links every principal number to a frozen artifact. The correct use of this work is to select the next validation investment and reproduce the diagnostic observations—not to convert transparent limitations into hidden confidence.

Sources:

- `phase_c_synthesis/W06_Evidence_Registry_v1.0.0.json`
- `phase_d_capstone/W08_Evidence_Registry_v1.0.0.json`
- `phase_d_capstone/W08_Claim_Evidence_Matrix_v1.0.0.csv`

## Slide 12 — A 12-week extension adds 4 experiments that convert proxies into validation (3:15)

I would extend the programme with four registered experiments. In weeks one through three, I would recruit at least two model-blind reviewers across the relevant safety domains and score 30 to 50 stratified calibration items. Ambiguous anchors would be revised without inspecting labels from the new sealed test set, and agreement would be reported by dimension and severity. In weeks four through six, I would build a simulator or replay harness for time-aligned sensor corruption and dropout. The primary outcomes would be unsafe actions, collision or constraint violations, recovery success, intervention rate, and time to a safe state. In weeks seven through nine, I would repeat the RAG Pareto study across time-sliced corpora, access policies, stale and conflicting documents, adversarial insertions, multiple seeds, and at least two hardware targets. The key question would be whether Pareto membership and citation behaviour generalise. In weeks ten through twelve, I would execute cooperative tasks with controlled bandwidth, message loss, and agent dropout, measuring joint success, allocation coverage, recovery latency, backward transfer, and catastrophic forgetting. This directly addresses the current CRL-MRS gap. The final recommendation is to close evaluator calibration first. Additional GPU spend is most valuable when it buys closed-loop evidence, domain review, and cross-corpus generalisation—not simply more rows on the current diagnostic leaderboard. The capstone delivers a reproducible evaluation system, three conditional candidate decisions, an explicit map of what remains unknown, and a sequence of gates that can turn public proxies into product-representative evidence. I am happy to take questions on the 35-scenario design, the RAGAS dimensions, masked-input limitations, severity weighting, Pareto selection, or the PIC readiness map.

Sources:

- `phase_d_capstone/W08_Capstone_Report_Source.md#10-limitations-and-next-evaluation-priorities`
- `phase_d_capstone/W08_Final_Readout_QA.md`
- `phase_d_capstone/W08_Final_Evaluation_Rubric.md`

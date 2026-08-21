# Final 15-Minute Project Review — Speaker Script

Planned presentation time: **15.0 minutes**.

The deck uses public/synthetic diagnostic evidence only. Every slide contains a matching source block in its PowerPoint speaker notes.

## Slide 1 — Eight Weeks of Physical-AI Evaluation (0:35)

Good morning. This is a fifteen-minute final review of the eight-week AI model evaluation internship. I will spend roughly half the time summarising the experimental work from Weeks 1 to 6, then focus on the new Week 7 dashboard and Week 8 capstone package. The central outcome is a reproducible decision-support system: it can prioritise models and experiments, but it deliberately does not claim deployment certification.

Sources:

- `phase_d_capstone/W08_Capstone_Report_Source.md`
- `phase_d_capstone/W08_Claim_Evidence_Matrix_v1.0.0.csv`

## Slide 2 — Eight weeks built one evidence chain (0:55)

The project was cumulative. Phase A defined what could be evaluated safely with public data and built the common benchmark. Phase B ran the three-model, RAG, robustness, masked-input, and multimodal studies. Phase C moved from individual results to controlled optimisation, Pareto trade-offs, PIC-specific evidence gaps, and a claim registry. Phase D then changed the communication layer: Week 7 built distinct stakeholder views, and Week 8 bound the final conclusions to versioned evidence. The same rule applies across all phases: exact model revision, evaluation-set version, and seed 42 remain attached to every principal claim.

Sources:

- `README.md`
- `phase_d_capstone/W08_Capstone_Report_Source.md#1-executive-summary-and-programme-scope`

## Slide 3 — 35 scenarios created a comparable baseline (1:00)

The benchmark contains 35 public-safe synthetic scenarios, with seven for each of five product contexts: care, education, security, navigation, and task execution. This equal allocation prevents conversational cases from dominating the result. The three frozen text models answered the same 35 prompts, producing 105 comparable responses. Severity levels one, three, and five represent consequence priority, not linguistic difficulty and not a cardinal harm model. One limitation is important: the seven originally held-out scenarios were inspected during iteration, so they now support regression testing rather than a fresh blind generalisation claim. A production follow-up needs a newly sealed, domain-reviewed test set.

Sources:

- `phase_a_design/W02_Scenarios.yaml`
- `phase_a_design/W02_Baseline_Run_Manifest.json`
- `phase_d_capstone/W08_Claim_Evidence_Matrix_v1.0.0.csv#C01-C03`

## Slide 4 — Mistral leads; Judge gate fails (1:10)

Mistral has the highest portfolio diagnostic proxy at 85.8 out of 100, followed by Llama at 80.2 and FLAN at 65.0. Seven responses were labelled unsafe across the 105-response set: six from FLAN, one from Llama, and none from Mistral. However, the governing result is the measurement gate. The frozen-label Judge calibration reached ordinal Krippendorff alpha 0.7551, below the preregistered threshold of 0.80. Therefore this is not a validated leaderboard. The defensible action is to reject FLAN as the primary candidate and advance Mistral, with Llama as a comparison arm, to independent model-blind domain-expert calibration.

Sources:

- `phase_d_capstone/W07_Dashboard/data/model_scorecard.csv`
- `phase_a_design/W02_Baseline_Agreement.json`
- `phase_d_capstone/W08_Claim_Evidence_Matrix_v1.0.0.csv#C02-C03`

## Slide 5 — RAG adds quality—and 8.03 seconds (1:10)

The first RAG design used short atomic sections, so changing the nominal chunk size did not change the indexed units. I corrected that by ingesting 21 complete official public sources and creating 40 questions with 58 mapped answer facts. The operational 256, 512, and 1024-token splits now produce genuinely different chunks. On the matched evaluation, RAG increases answer relevance by 0.655667 and required-point coverage by 0.522917, but adds 8.03 seconds of mean request latency. Retrieval quality is strong, yet RAGAS-style metrics still do not prove that a source is current, authorised, safe, or appropriate for action. The recommendation is conditional RAG with source governance and latency gates.

Sources:

- `phase_b_evaluation/W03_RAG_Long_Source_Summary_v1.0.0.json`
- `phase_b_evaluation/W04_RAG_Long_Performance_Summary_v1.0.0.json`
- `phase_d_capstone/W08_Claim_Evidence_Matrix_v1.0.0.csv#C04-C05`

## Slide 6 — Consistency can hide stable failure (1:10)

Week 4 shows why robustness must be paired with correctness. FLAN has the highest semantic consistency at 0.914, but that number contains 25 stable failures and only seven stable passes. Mistral has lower consistency at 0.857, yet zero stable failures and 30 stable passes. A model can therefore be consistently wrong. The VLM study tells a related story: Idefics2 and LLaVA tied on 19 of 20 clean public-image cases, so there is no defensible quality winner. Efficiency separates them more clearly. LLaVA's p50 latency is 1.92 seconds lower and its peak GPU use is about 4,177 MiB lower. It is an efficiency candidate, not a validated multimodal deployment choice.

Sources:

- `phase_d_capstone/W07_Dashboard/data/robustness_summary.csv`
- `phase_d_capstone/W07_Dashboard/data/vlm_performance.csv`
- `phase_d_capstone/W08_Claim_Evidence_Matrix_v1.0.0.csv#C07-C08`

## Slide 7 — Three Pareto cells, not one setup (1:00)

Week 5 ran a complete 3 by 3 by 2 RAG factorial: three chunk sizes, three top-k values, and reranking on or off. Three of the 18 cells are Pareto-optimal, so there is no single universal winner. The 1024-token, top-k-five, cross-encoder cell is the balanced diagnostic choice, not a general optimum. Week 6 then mapped the accumulated evidence to six PIC working classes. The largest gap is CRL-MRS, with zero direct multi-agent execution tests. This synthesis changes the question from which score is highest to which capability has evidence, which trade-off is acceptable, and which validation gate is still empty.

Sources:

- `phase_c_synthesis/W05_RAG_Long_Source_Optimisation_Summary_v1.1.0.json`
- `phase_c_synthesis/W05_PIC20_Model_Analysis.md`
- `phase_c_synthesis/W06_Evidence_Registry_v1.0.0.json`

## Slide 8 — Week 7 turned evidence into an interface (0:55)

Week 7 introduced a strict two-stage dashboard architecture. An offline builder verifies eight frozen upstream artifacts, performs every aggregation once, and writes eleven reviewed CSVs with source lineage. The Streamlit application then only loads, filters, and visualises those CSVs. It performs no inference, retrieval, Judge scoring, or raw aggregation at launch. This matters because a stakeholder interface should not silently change the underlying result. The dashboard is therefore reproducible, fast, and auditable, while retaining the same failed-calibration and public-proxy boundaries as the evidence underneath it.

Sources:

- `phase_d_capstone/W07_Dashboard_Design_Doc.md#1-design-objective`
- `phase_d_capstone/W07_Dashboard/data/data_manifest.csv`
- `phase_d_capstone/W07_Dashboard/data/dashboard_metadata.csv`

## Slide 9 — Three indicators drive one executive action (0:55)

The executive view contains exactly three registered indicators: the highest portfolio diagnostic proxy, the seven observed unsafe outputs, and the recommendation for at least two independent reviewers. The important design choice is what it removes. Executives do not see technical tabs, model selectors, or a leaderboard without context. An interpretation banner appears before the metrics, and the action card explicitly blocks deployment selection from the diagnostic ordering. Week 7 therefore makes uncertainty visible at the same level as performance rather than hiding it in methodology notes.

Sources:

- `phase_d_capstone/W07_Dashboard/assets/executive_view.png`
- `phase_d_capstone/W07_Dashboard/data/executive_summary.csv`
- `phase_d_capstone/W07_Dashboard_Design_Doc.md#executive`

## Slide 10 — Product managers see platform risk first (1:00)

The product-manager mode is not the engineer interface with smaller text. It has only two views. Platform Risk puts the failure heat map and three selected-platform concern cards above technical methodology. RAG Readiness compares Fari and Senpai on the same Base and RAG dimensions while keeping deployment status as not established. Automated browser navigation exposed the top concerns in under three seconds, which satisfies the interface target, but this is not an independent human comprehension study. The value of the view is prioritisation: it helps a product manager identify where to investigate without implying that counts alone estimate product risk.

Sources:

- `phase_d_capstone/W07_Dashboard/assets/product_manager_view.png`
- `phase_d_capstone/W07_Dashboard/data/failure_heatmap.csv`
- `phase_d_capstone/W07_Dashboard/data/platform_failure_concerns.csv`
- `phase_d_capstone/W07_Dashboard_Design_Doc.md#product-manager`

## Slide 11 — All 18 RAG cells remain visible (1:00)

The engineer view does the opposite of the executive view: it preserves the full evidence surface. All 18 RAG configurations remain visible, with three Pareto cells highlighted and the balanced choice labelled as conditional. The scorecard includes exact model revisions and coverage. Robustness pairs consistency with stable passes and failures. The Data Sources and Reproduction view exposes source hashes and commands. This design prevents a recommended point from erasing the alternatives that define its trade-off. It also keeps model version, evaluation-set version, seed, evidence status, and source path available for audit.

Sources:

- `phase_d_capstone/W07_Dashboard/assets/engineer_view.png`
- `phase_d_capstone/W07_Dashboard/data/rag_configurations.csv`
- `phase_d_capstone/W07_Dashboard/data/data_manifest.csv`
- `phase_d_capstone/W07_Dashboard_Design_Doc.md#ai-evaluation-engineer`

## Slide 12 — Version 1.2 fixed role separation and contrast (1:00)

Week 7 also included an important correction cycle. User review of version 1.1 showed that native Streamlit labels could render white on a light background and that the three persona modes still felt too similar. Version 1.2 added an explicit light theme and targeted contrast overrides, then replaced the shared information architecture with truly different Executive, Product Manager, and Engineer modes. The final package passed 12 source-and-data contracts and six Streamlit interaction tests. A headless browser traversed three personas and four engineer tabs with zero white-on-white candidates. This verifies the current interface state honestly; it does not replace an independent accessibility or human-usability study.

Sources:

- `phase_d_capstone/W07_Dashboard/fresh_copy_verification_v1.2.0.json`
- `phase_d_capstone/W07_Dashboard/test_dashboard_contract.py`
- `phase_d_capstone/W07_Dashboard/test_dashboard_app.py`
- `weekly/Wk-07-EvalLog.md`

## Slide 13 — Week 8 binds claims to hashed evidence (1:05)

Week 8 packages the work into a 19-page report with ten required sections, a 12-claim evidence matrix, and a registry of 16 verified source artifacts. Each principal claim records its model revision, evaluation-set version, seed, scope, evidence status, decision use, and source path. The registry adds byte counts and SHA-256 hashes. It also freezes the latest-only policy: long-source RAG version 1.0, Week 5 optimisation version 1.1, and dashboard version 1.2. Superseded atomic-section RAG results remain available as provenance but are not used for current conclusions. This is the main Week 8 addition: the final narrative can be checked against the exact evidence it summarises.

Sources:

- `phase_d_capstone/W08_Capstone_Report.docx`
- `phase_d_capstone/W08_Claim_Evidence_Matrix_v1.0.0.csv`
- `phase_d_capstone/W08_Evidence_Registry_v1.0.0.json`
- `phase_d_capstone/W08_Capstone_Contract_Tests.py`

## Slide 14 — The capstone advances candidates, not deployment (1:00)

The final synthesis makes three conditional candidate decisions. Mistral advances to expert calibration. Two Pareto RAG configurations advance to a pilot comparison so quality and latency remain visible. LLaVA advances as the efficiency candidate for the next multimodal testbench. The number of deployment approvals is zero. That boundary is not a weakness in the report; it is the correct result of the evidence. Judge calibration failed, the scenarios and images are public proxies, product telemetry is absent, and no closed-loop sensor, action, or multi-agent system was measured. The capstone turns these limits into explicit gates rather than burying them in a final limitations paragraph.

Sources:

- `phase_d_capstone/W08_Capstone_Report_Source.md#9-final-recommendations-and-deployment-gates`
- `phase_d_capstone/W08_Claim_Evidence_Matrix_v1.0.0.csv#C12`
- `phase_d_capstone/W08_Final_Readout_QA.md`

## Slide 15 — Decision system—not a certificate (1:05)

To conclude, the project now has a reproducible benchmark, frozen evidence registry, controlled RAG optimisation, PIC-specific gap map, stakeholder dashboard, and final capstone package. These artifacts are ready to prioritise candidates and design the next evaluation. Four gates remain. First, calibrate ratings with independent model-blind domain experts on a new sealed set. Second, replace text masking and static images with closed-loop sensor dropout and safe-recovery tests. Third, test production RAG under access policy, drift, stale or conflicting documents, adversarial insertions, multiple seeds, and another hardware target. Fourth, build the missing CRL-MRS multi-agent execution benchmark. The main contribution is not one more score; it is a controlled way to decide what the score is allowed to mean and what must be tested next.

Sources:

- `phase_d_capstone/W08_Retrospective.md`
- `phase_d_capstone/W08_Capstone_Report_Source.md#10-limitations-extension-plan-and-conclusion`
- `weekly/Wk-08-Final-EvalLog.md`

## Delivery note

The timing assumes a measured technical pace. If time is reduced, shorten Slides 2 and 7; preserve Slides 4–6 and 8–15 because they carry the main decisions, Week 7–8 additions, and evidence boundaries.

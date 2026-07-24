# Week 2 Submission Index

**Benchmark:** `ingen_physical_ai_text_scenarios` version `0.2.0`  
**Candidate run:** `w02-two-model-unified-full-v1.0.0`  
**Diagnostic Judge run:** `w02-two-model-unified-prometheus-diagnostic-v1.0.0`  
**Status:** execution complete; Judge failed calibration, so automated ratings are
diagnostic and every response requires human adjudication

## Read first

1. `W02_Eval_Framework_Design.md` — scope, 35-scenario design, explicit rubrics,
   severity policy, adjudication, and claim boundaries.
2. `W02_Final_Run_and_Judge_Findings.md` — iteration history, final pipeline findings,
   successful cases, failures, and Judge root-cause analysis.
3. `../weekly/Wk-02-EvalLog.md` — requirement checklist, pipeline diagram, run log,
   reproducibility commands, and blockers.
4. `W02_Baseline_Pipeline_Report.md` — final execution contract and three-Judge
   agreement reported separately for each model, plus the required pipeline-level
   diagnostic.
5. `W02_Baseline_Eval_Results.csv` — complete row-level submission table.
6. `W02_Per_Model_Diagnostic_Aggregates.md` — separate per-platform,
   severity-weighted, failure-mode, and split diagnostics for each model.

## Required deliverables

| Plan deliverable | Path | Contents |
|---|---|---|
| Evaluation design | `phase_a_design/W02_Eval_Framework_Design.md` | Scenario design, score definitions, severity, workflow |
| Baseline results | `phase_a_design/W02_Baseline_Eval_Results.csv` | 70 rows with prompt, response, seed, all Judge ratings/comments |
| Weekly log | `weekly/Wk-02-EvalLog.md` | Work completed, metrics, issues, next steps |

The master CSV remains the plan-required lossless table. For human review it is also
split into `W02_Baseline_Eval_Results_FLAN.csv` and
`W02_Baseline_Eval_Results_Mistral.csv`, each containing exactly 35 rows. No
model-performance mean combines the two models.

## Source of truth

| Component | File |
|---|---|
| Proposed product rules | `W02_Product_Regulations.yaml` |
| Scenario bank | `W02_Scenarios.yaml` |
| Shared rubric | `W02_Rubric.yaml` |
| Unified candidate prompt | `W02_Prompt_Spec_v0.4.0.yaml` |
| Frozen run configuration | `W02_Unified_Run_Config_v1.0.0.yaml` |
| Candidate/Judge/finalize orchestrator | `W02_Baseline_Pipeline.py` |
| Two-model candidate runner | `W02_Two_Model_Structured_Full_Run.py` |
| Shared prompt/render/hash utilities | `W02_Eval_Runner.py` |
| Mistral engine | `W02_Mistral_Eval_Runner.py` |
| Structured Judge parsing/mapping utilities | `W02_Structured_Judge.py` |
| Deterministic evidence checks | `W02_Deterministic_Checks.yaml` |
| Diagnostic Judge | `W02_Prometheus_Judge.py` |
| Diagnostic full-run driver | `W02_Prometheus_Full_Run.py` |
| Judge rubric prompt | `W02_Prometheus_Judge_Spec_v0.8.3.yaml` |
| Scenario requirement atoms | `W02_Judge_Requirement_Metadata_v0.4.0.yaml` |
| Frozen calibration fixture and gate logic | `W02_Structured_Judge_Calibration.py` |
| Prometheus calibration runner | `W02_Prometheus_Judge_Calibration.py` |
| Final CSV/agreement builder | `W02_Finalize_Baseline.py` |
| Separate model views and aggregates | `W02_Build_Per_Model_Views.py` |
| Optional deterministic cache validator | `W02_Merge_Deterministic_Judge_Cache.py` |
| Benchmark validator | `W02_validate_benchmark.py` |

## Final execution coverage

| Stage | Coverage |
|---|---:|
| Frozen candidates | 35 scenarios × 2 models = 70 responses |
| Judge formulations | 70 responses × 3 prompts = 210 formulation evaluations |
| Judge dimensions | 210 formulations × 3 dimensions = 630 raw Judge traces |
| Candidate errors | 0 |
| Candidate input truncations | 0 |
| Shared-prompt equality | 35/35 scenario/model pairs |

The three Judge formulations are independently rendered and executed, but use the same
rubric and one pinned checkpoint. They measure prompt sensitivity, not the reliability
of three independent models.

The 16 Judge calibration items are eight frozen Mistral outputs from candidate prompt
`0.2.0` and eight from prompt `0.4.0`, covering nine unique scenario IDs. They carry
provisional single-reviewer labels and are used to test and accept/reject Judge prompt
and mapping designs; they do not train Prometheus. The scenario YAML separately
contains 28 development and seven originally held-out scenarios. Those seven were
later inspected, so a future blind claim requires a new sealed test set.

## Interpretation boundary

- FLAN-T5-base is an intentionally weak floor and copied the one-shot heavily under
  the required shared prompt.
- Mistral is more scenario-responsive and improved in the human-screened prompt pilot,
  but still has factual, completeness, and grounding failures.
- The Prometheus `0.8.3` and PIC-inspired `0.9.2` prompt/mapping specifications
  both failed the frozen provisional-label acceptance gates. These are local
  specification versions; both use the same Prometheus-7B-v2.0 checkpoint.
- Agreement numbers are diagnostic; correlated Judge agreement does not prove
  correctness.
- All severity-5 cases and all rows in this run require human review.
- No result is evidence about a deployed InGen product; the benchmark is synthetic L0
  text simulation.

## Public Git submission scope

The following reproducibility artifacts belong in the public commit:

- benchmark and policy: `W02_Scenarios.yaml`, `W02_Product_Regulations.yaml`,
  `W02_Rubric.yaml`, `W02_Result_Schema.json`;
- candidate configuration: `W02_Prompt_Spec_v0.4.0.yaml`,
  `W02_Unified_Run_Config_v1.0.0.yaml`;
- end-to-end execution: `W02_Baseline_Pipeline.py`,
  `W02_Two_Model_Structured_Full_Run.py`, `W02_Finalize_Baseline.py`,
  `W02_Build_Per_Model_Views.py`, `W02_Eval_Runner.py`,
  `W02_Mistral_Eval_Runner.py`;
- Judge execution: `W02_Prometheus_Judge.py`, `W02_Prometheus_Full_Run.py`,
  `W02_Structured_Judge.py`, `W02_Structured_Judge_Calibration.py`,
  `W02_Prometheus_Judge_Calibration.py`,
  `W02_Prometheus_Judge_Spec_v0.8.3.yaml`,
  `W02_Judge_Requirement_Metadata_v0.4.0.yaml`,
  `W02_Deterministic_Checks.yaml`;
- pinned model acquisition: `W02_Download_Pinned_Models.py`,
  `W02_Download_Pinned_Prometheus_Judge.py`;
- validation: `W02_validate_benchmark.py`, `W02_Prometheus_Judge_Tests.py`,
  `W02_Structured_Judge_Tests.py`;
- results and reports: the master CSV, both 35-row model CSV views, per-model
  aggregate CSV/JSON/report, agreement JSON, run manifest, framework design,
  findings, pipeline report, submission index, and Week 2 EvalLog;
- environment support: repository `requirements.txt`,
  `requirements-runpod-mistral.txt`, `.gitignore`, and README model/run registry.

Do not commit the confidential PDFs, `2026 Jun Internship Data Sources`, private
crosswalks, API/SSH credentials, local environments, model caches, RunPod logs, or
private lossless experiment directories.

## Reproduction checks

Run from the repository root:

```powershell
D:\newIntern\envs\ingen-ai-eval\python.exe `
  phase_a_design\W02_Baseline_Pipeline.py --stage validate

D:\newIntern\envs\ingen-ai-eval\python.exe `
  phase_a_design\W02_validate_benchmark.py

D:\newIntern\envs\ingen-ai-eval\python.exe `
  phase_a_design\W02_Structured_Judge_Tests.py

D:\newIntern\envs\ingen-ai-eval\python.exe `
  phase_a_design\W02_Prometheus_Judge_Tests.py

D:\newIntern\envs\ingen-ai-eval\python.exe `
  phase_a_design\W02_Build_Per_Model_Views.py
```

The public repository contains submission-ready, synthetic/public-safe artifacts.
Lossless JSONL, 630 exact Judge prompt traces, GPU logs, calibration outputs, and model
download manifests remain in the local private support area and must not be committed.

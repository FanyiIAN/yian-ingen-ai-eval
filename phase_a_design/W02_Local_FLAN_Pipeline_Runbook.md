# Week 2 Local FLAN Pipeline Runbook

## Purpose and claim boundary

This runbook reproduces the 35-scenario local integration test with
`google/flan-t5-base`. The same checkpoint is used for response generation and
three prompt-judge formulations, so the run validates the pipeline and exposes
prompt sensitivity. It is not an independent judge condition or the official
two-model Week 2 baseline.

No RAG component is used. Scenario facts, proposed regulations, scoring rubrics,
and judge prompts are supplied directly from versioned local artifacts.

## Fixed local resources

- Python: `D:\Anaconda\envs\inGen\python.exe`
- Model: `D:\newIntern\private\model_runtime\flan_t5_base\model`
- Model ID: `google/flan-t5-base`
- Revision: `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2`
- Device and precision: CPU, float32
- Seed and decoding: 42, deterministic greedy decoding
- Model and runtime caches: private D: drive paths only

## Pipeline stages

1. Validate the scenario bank, regulations, rubric, and source references.
2. Build a scenario-only candidate prompt without exposing expected answers.
3. Generate one FLAN response per scenario.
4. Run deterministic evidence checks with negation-aware prohibited-phrase rules.
5. Score each response with three judge prompts:
   `criterion_first`, `evidence_first`, and `failure_first`.
6. Accept a numeric median only when the three scores span at most one point;
   require two matching failure labels. Leave all other automated results unresolved.
7. Queue severity-5, sensitive, disagreement, truncation, and sampled cases for
   human review.
8. Validate every final JSONL row against `W02_Result_Schema.json`, then export
   JSONL, CSV, JSON summary, and Markdown report.

## Commands

From `D:\newIntern\yian-ingen-ai-eval`:

```powershell
& 'D:\Anaconda\envs\inGen\python.exe' `
  'D:\newIntern\yian-ingen-ai-eval\phase_a_design\W02_validate_benchmark.py'

$env:PYTHONIOENCODING = 'utf-8'
& 'D:\Anaconda\envs\inGen\python.exe' `
  'D:\newIntern\yian-ingen-ai-eval\phase_a_design\W02_Eval_Runner.py' `
  --mode full `
  --run-id local-flan-full-v0.3.0
```

If execution is interrupted, repeat the runner command with `--resume`. The
runner reads `rows.checkpoint.jsonl` and skips complete scenario IDs.

## Completed run

The `local-flan-full-v0.2.0` run completed 35/35 scenarios with zero candidate
generation errors. Every row contains three judge results and passes the result
schema. Counts are five platforms x 7, 28 development plus 7 held-out, and
severity classes 1/3/5 distributed 10/15/10.

The local judge failed the pre-registered reliability gate:

- Task Accuracy alpha, all rows: 0.060114
- Task Accuracy alpha, development: 0.085938
- Acceptance target: 0.80
- Human-review queue: 35 pending
- Judge prompt truncation: 3 rows, all queued for review

The low agreement and audit/judge conflicts mean FLAN's provisional medians are
not reliable final scores. A first-pass human review has since scored all 35 rows:
human Task mean `1.571429` versus the old automated `3.542857`; exact Task
agreement was `2/35`, and exact failure-label agreement was `1/35`.

A development-only four-Prompt candidate screen produced no output scoring 4 or
5 across 52 generations. The best variant mean was only `1.692308`, with six
critical failures. A revised-Judge eight-item calibration also failed: Task alpha
`0.12687`, Task within-one `1/8`, failure exact `1/8`, and two critical reversals.
FLAN-T5-base is therefore excluded as a Judge. It remains a deliberately weak
candidate condition for capacity comparison.

## Artifact map

- Candidate prompt: `W02_Prompt_Spec.yaml` (`0.3.0`)
- Three judge prompts: `W02_Judge_Prompts.yaml` (`0.2.1`)
- Deterministic audit rules: `W02_Deterministic_Checks.yaml`
- Result schema: `W02_Result_Schema.json`
- Runner: `W02_Eval_Runner.py`
- Reproducibility auditor: `W02_Audit_Reproducibility.py`
- Replay comparator: `W02_Compare_Replay.py`
- Full rows: `experiments/w02_local_flan_pipeline/local-flan-full-v0.2.0/W02_FLAN_Local_Integration_Rows.jsonl`
- Flat results: `experiments/w02_local_flan_pipeline/local-flan-full-v0.2.0/W02_FLAN_Local_Integration_Results.csv`
- Summary: `experiments/w02_local_flan_pipeline/local-flan-full-v0.2.0/W02_FLAN_Local_Integration_Summary.json`
- Report: `experiments/w02_local_flan_pipeline/local-flan-full-v0.2.0/W02_FLAN_Local_Integration_Report.md`
- Exact rendered prompts: `experiments/w02_local_flan_pipeline/local-flan-full-v0.2.0/W02_FLAN_Rendered_Prompts.jsonl`
- All-output evidence table: `experiments/w02_local_flan_pipeline/local-flan-full-v0.2.0/W02_FLAN_All_Output_Evidence.md`
- Combined prompt/output/judge trace: `experiments/w02_local_flan_pipeline/local-flan-full-v0.2.0/W02_FLAN_Full_Trace.md`
- Reproducibility manifest: `experiments/w02_local_flan_pipeline/local-flan-full-v0.2.0/W02_FLAN_Reproducibility_Manifest.json`
- Diagnostic analysis: `experiments/w02_local_flan_pipeline/local-flan-full-v0.2.0/W02_FLAN_Output_Analysis.json`
- Findings and limitations: `experiments/w02_local_flan_pipeline/local-flan-full-v0.2.0/W02_FLAN_Reproducibility_and_Findings.md`
- Replay comparison: `experiments/w02_local_flan_pipeline/local-flan-full-v0.2.0/W02_FLAN_Replay_Comparison.json`
- Complete replay run: `experiments/w02_local_flan_pipeline/local-flan-full-replay-v0.2.0/`
- Row-by-row human adjudication: `experiments/w02_human_adjudication_v0.1.0/W02_Human_Adjudication_and_Judge_Diagnosis.md`
- Candidate Prompt screen: `experiments/w02_candidate_prompt_screen/flan-prompt-screen-v0.1.0/`
- Revised Judge anchor calibration: `experiments/w02_judge_calibration/flan-judge-calibration-anchor-v0.2.0/`

## Remaining official gates

1. Obtain supervisor approval for the regulations, severity assignments, and
   independent judge model.
2. Pass the revised Mistral Judge development calibration against human gold;
   if it fails, retain human adjudication and request an approved independent Judge.
3. Pilot candidate Prompt `0.3.0` on development cases before any full rerun.
4. Obtain a second-human review for all severity-5 and a stratified sample.
5. Author a new sealed held-out set because the original seven have been inspected.
6. Only then publish the official 70-row `W02_Baseline_Eval_Results.csv`.

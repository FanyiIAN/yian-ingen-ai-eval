# InGen Physical AI Model Evaluation Internship

Public-safe deliverables for an AI model evaluation programme. The work studies how public/open models can be evaluated against product-specific Physical AI requirements using reproducible scenarios, multi-dimensional rubrics, robustness tests and risk-aware reporting.

### Phase A - Landscape and benchmark design

Formal Week 1 deliverables:

- `phase_a_design/W01_PhysicalAI_Eval_Landscape.md`: five-part landscape brief.
- `phase_a_design/W01_env_check.ipynb`: executed environment and library smoke test.
- `weekly/Wk-01-EvalLog.md`: Week 1 evaluation reflection.

Public supporting evidence retained with the submission package:

- `phase_a_design/W01_Reading_Annotations.md`: expanded annotations for the six required source categories.

Personal study guides, canonical-link notes, requirement audits and setup scripts are intentionally excluded from this repository.

Formal Week 2 deliverables:

- `phase_a_design/W02_Eval_Framework_Design.md`: 35-scenario framework, explicit
  Task/Grounding rubrics, severity policy, and adjudication design.
- `phase_a_design/W02_Baseline_Eval_Results.csv`: 70-row, two-model result table with
  complete candidate prompts/responses and three raw Judge traces per dimension.
- `phase_a_design/W02_Baseline_Eval_Results_FLAN.csv` and
  `phase_a_design/W02_Baseline_Eval_Results_Mistral.csv`: separate 35-row review
  views; model-performance statistics are never averaged across the two models.
- `phase_a_design/W02_Per_Model_Diagnostic_Aggregates.md`: per-model platform,
  severity-weighted, failure-mode, and original-split diagnostics with resolved-score
  coverage.
- `weekly/Wk-02-EvalLog.md`: iteration history, pipeline diagram, agreement, issues,
  reproducibility record, and blockers.
- `phase_a_design/W02_Submission_Index.md`: navigation and interpretation boundary.

The Week 2 automated Judge failed its provisional single-reviewer calibration. The
16 calibration items are frozen Mistral outputs with reference labels, not training
examples or additional benchmark scenarios. Its ratings are published as diagnostic
traces only; every result row requires human adjudication.

The scenario YAML contains 28 development and seven originally held-out scenarios.
Because the seven held-out scenarios were later inspected during iteration, they remain
regression labels but no longer constitute a fresh blind test set.

## Reproduce the Week 1 environment check

```powershell
conda create -n inGen python=3.11 -y
conda activate inGen
python -m pip install -r requirements.txt
python -m ipykernel install --user --name inGen --display-name "Python (inGen)"
jupyter nbconvert --to notebook --execute --inplace phase_a_design/W01_env_check.ipynb --ExecutePreprocessor.kernel_name=inGen
```

The notebook downloads no model weights and makes no paid API calls. Its transformer smoke test uses a randomly initialized tiny `transformers.BertModel`, synthetic inputs and seed 42; it is an environment check, not a model-performance claim.

## Repository structure

```text
phase_a_design/      Weeks 1-2: landscape, environment check, benchmark design
phase_b_evaluation/  Weeks 3-4: model, RAG, robustness and multimodal evaluation
phase_c_synthesis/   Weeks 5-6: PIC analysis and methodology synthesis
phase_d_capstone/    Weeks 7-8: dashboard, capstone report and deck
weekly/              Required weekly evaluation logs only
```

## Model version registry

Week 2 frozen conditions:

- `google/flan-t5-base` at
  `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2`;
- `mistralai/Mistral-7B-Instruct-v0.2` at
  `63a8b081895390a26e140280378bc85ec8bce07a`;
- diagnostic Judge `prometheus-eval/prometheus-7b-v2.0` at
  `66ffb1fc20beebfb60a3964a957d9011723116c5`.

All candidate and Judge calls use seed 42 and deterministic decoding. See
`phase_a_design/W02_Unified_Run_Config_v1.0.0.yaml` for the complete run contract.

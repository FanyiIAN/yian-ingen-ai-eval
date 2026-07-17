# InGen Physical AI Model Evaluation Internship

Public-safe deliverables for an eight-week AI model evaluation programme. The work studies how public/open models can be evaluated against product-specific Physical AI requirements using reproducible scenarios, multi-dimensional rubrics, robustness tests and risk-aware reporting.

This directory is the sole Git root and public submission area. Private references, study notes, audits, setup helpers and scratch artifacts are maintained outside this repository.

## Constraints

- Public data, public product information and open-source/free-tier tools only.
- No internal InGen documents, customer data, API keys or proprietary evaluation results.
- Every future evaluation claim records exact model/version, held-out evaluation-set version and random seed.
- Product performance figures are treated as company-stated targets unless independently reproduced.

## Submission index

### Phase A - Landscape and benchmark design

Formal Week 1 deliverables:

- `phase_a_design/W01_PhysicalAI_Eval_Landscape.md`: five-part landscape brief.
- `phase_a_design/W01_env_check.ipynb`: executed environment and library smoke test.
- `weekly/Wk-01-EvalLog.md`: Week 1 evaluation reflection.

Public supporting evidence retained with the submission package:

- `phase_a_design/W01_Reading_Annotations.md`: expanded annotations for the six required source categories.

Personal study guides, canonical-link notes, requirement audits and setup scripts are intentionally excluded from this repository.

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

No pretrained model benchmark has been run in Week 1. Exact model IDs will be added before Week 2 baseline evaluation.

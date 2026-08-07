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

### Phase B - Systematic evaluation, RAG and multimodal assessment

Week 3 Phase B artifacts:

- `phase_b_evaluation/W03_Submission_Index.md`: reference-to-artifact map,
  collection boundaries, final verification, and remaining limitations.
- `phase_b_evaluation/W03_RAG_Expanded_Benchmark_Report.md`: expanded Week 3
  report covering the 331-unit official-public knowledge base, 40-question
  benchmark, retrieval ablation, 240-row three-model Base/RAG run, local RAGAS
  diagnostics, and warm-path latency/resource profile.
- `phase_b_evaluation/W03_RAG_Expanded_Knowledge_Base_v0.6.0.yaml` and
  `W03_RAG_Expanded_Eval_Set_v0.6.0.yaml`: frozen expanded public inputs with
  fact-level provenance, hidden rubrics, and metadata-governed evidence links.
- `phase_b_evaluation/W03_RAG_Architecture_Review_7min.pptx` and
  `W03_RAG_Architecture_Review_Speaker_Script.md`: seven-slide meeting review
  covering architecture, controlled test design, results, limitations, and the
  next benchmark, with a 5–8 minute speaking script.
- `phase_b_evaluation/W03_RAG_Official_Knowledge_Base_v0.3.0.yaml`: governed
  official-public Fari/Senpai source snapshots with section, fact and provenance
  metadata. The older `W03_RAG_Knowledge_Base.yaml` remains a synthetic smoke fixture.
- `phase_b_evaluation/W03_RAG_Official_Eval_Set_v0.3.0.yaml`: 12 frozen
  source-grounded questions with hidden reference answers, weighted atomic scoring
  points, forbidden points and fact-level evidence links.
- `phase_b_evaluation/W03_RAG_Official_Run_Config_v0.3.0.yaml`: controlled
  base/RAG contract for BGE-M3 embeddings, persistent Chroma and
  Llama-3.1-8B-Instruct.
- `phase_b_evaluation/W03_RAG_Pipeline.py`: LangChain v1 YAML ingestion, recursive
  splitting, governed metadata gates, child-parent retrieval, BGE-M3 embedding,
  persistent Chroma indexing, retrieval traces and paired candidate-blind base/RAG
  input construction.
- `phase_b_evaluation/W03_RAG_Generation.py`: resumable BF16 CUDA generation
  runner for a frozen Llama-3.1-8B-Instruct checkpoint.
- `phase_b_evaluation/W03_RAG_Evaluation.py`: hidden point-rubric join and
  authorization-gated RAGAS v0.4 collections metrics.
- `phase_b_evaluation/W03_RAG_Result_Analysis.py` and
  `W03_RAG_Qualitative_Gate.py`: non-finite-safe metric comparison and registered
  qualitative regression review.
- `phase_b_evaluation/W03_RAG_Multi_Model_Generation.py`,
  `W03_RAG_Multi_Model_Inputs.py`, and
  `W03_RAG_Multi_Model_Result_Analysis.py`: model-native FLAN/Mistral/Llama
  serialization, shared-context expansion, and deterministic three-model audit.
- `phase_b_evaluation/W03_RAG_Official_Blind_Eval_Set_v0.4.0.yaml` and
  `W03_RAG_Official_MultiModel_Run_Config_v0.5.0.yaml`: the frozen eight-question,
  48-row blind Base/RAG confirmation contract.
- `phase_b_evaluation/W03_RAG_Three_Model_Blind_Report.md` and
  `W03_RAG_Three_Model_Blind_Summary.json`: uninspected seven-question aggregate,
  RAGAS diagnostics, citation compliance, runtime, and validity boundaries.
- `phase_b_evaluation/W03_RAG_Three_Model_Blind_Evaluation.ipynb`: executed
  submission-facing aggregate notebook that makes no GPU or Judge calls.
- `phase_b_evaluation/W03_RAG_Retrieval_Ablation_Report.md`: the completed
  18-variant chunk-size, top-k, and reranker ablation.
- `phase_b_evaluation/W03_RAG_Benchmark_Representativeness_Audit.md`: question
  count, difficulty, coverage gaps, defensible usability boundary, and a
  preregistered expansion blueprint.
- `phase_b_evaluation/W03_RAG_AI_Calibration_Report.md` and
  `W03_RAG_AI_Calibration_Annotations_v0.3.0.yaml`: completed, explicitly
  disclosed eight-row AI qualitative calibration with a row-level result table,
  weighted required-point coverage, claim support, forbidden-claim checks and
  failure codes.
- `phase_b_evaluation/W03_RAG_Evaluation.ipynb`: executed submission-facing
  LangChain RAG contract and base-versus-RAG metric notebook.
- `phase_b_evaluation/W03_Evaluation_Memo.md`: three-page Week 3 memo covering
  the model comparison boundary, RAG findings and top deployment-relevant
  failure patterns.
- `phase_b_evaluation/W03_RAG_Official_Benchmark_Card.md` and
  `W03_RAG_Official_Benchmark_Report.md`: benchmark scope, source governance,
  method, results and validity boundaries.
- `phase_b_evaluation/W03_Failure_Taxonomy.md`: mutually exclusive three-level
  model-behavior hierarchy with InGen platform implications.
- `phase_b_evaluation/W03_RAG_Failure_Taxonomy.md`: data, retrieval, generation
  and evaluation/runtime causal failure codes.
- `weekly/Wk-03-EvalLog.md`: complete iteration history, metrics, runtime repairs
  and reproducibility record.
- `phase_b_evaluation/W03_Llama_Extended_Benchmark.py`: third-model runner for
  the same frozen 35 scenarios and semantic candidate prompt used in Week 2.
- `phase_b_evaluation/W03_Extended_Benchmark.ipynb`: submission-facing notebook
  that loads the contract-checked comparison artifacts without rerunning models.
- `phase_b_evaluation/W03_Three_Model_Diagnostic_Comparison.csv` and
  `W03_Three_Model_Diagnostic_Summary.json`: three-model, per-platform,
  severity-weighted, failure-distribution and runtime summaries.
- `phase_b_evaluation/W03_Three_Model_Diagnostic_Report.md`: concise comparison
  report with score coverage and validity boundaries.

The Llama full run completed all 35 frozen scenarios with no generation errors or
input truncation. The three-model contract check passed for 105 rows, but the
Prometheus Judge failed its Week 2 calibration. Its score ordering is therefore
diagnostic only and cannot support a validated model-quality claim until
model-blind human adjudication.

The official-public RAG run completed 12 paired base/RAG questions. Retrieval
achieved document and evidence-fact recall@k of `1.0000`, MRR `1.0000` and zero
metadata leakage. Provisional answer relevance was `0.343704` for base and
`0.573415` for RAG. The retained v0.3.1 epistemic-boundary prompt reached answer
relevance `0.580729` and faithfulness `0.930556` with 12/12 finite rows while
keeping retrieval contexts byte-for-byte fixed. Four later one-change iterations
were retained as transparent failure evidence: two failed qualitative gates,
v0.3.4 materially regressed on aggregate relevance and faithfulness, and v0.3.5
was stopped after a final qualitative regression. Further prompt tuning on the
inspected set was stopped to avoid overfitting.
The final three-model public RAG run completed 48/48 generations with a passed
shared-input audit. On the seven uninspected questions, Llama answer relevance
increased from `0.290288` to `0.538142` and RAG faithfulness was `0.821429`.
FLAN failed the detailed instruction/citation contract; Mistral's automatic
scores remain non-independent self-judge diagnostics. The 18-variant retrieval
ablation showed fact recall `0.7000` at top-k 1 and `1.0000` at top-k 3/5; chunk
size was non-discriminating because every governed section was below 256 tokens.

Automatic Judge metrics are treated as uncalibrated diagnostics and are
triangulated with the completed AI qualitative calibration. The three-model
Base/RAG plus RAGAS comparison used only the public collection. The `2026 Jun
Internship Data Sources` material was tested through a separately governed
private collection: retrieval passed 6/6 questions and Llama completed 12/12
Base/RAG generations. It was not pooled into public metrics, expanded to the
three-model comparison, or copied into this public repository.

Week 4 Phase B artifacts:

- `phase_b_evaluation/W04_Submission_Index.md`: Week 4 navigation,
  submission-safe evidence boundary, and verification commands.
- `phase_b_evaluation/W04_Robustness_Eval.ipynb`: executed aggregate
  notebook for three-model semantic and masked-input robustness.
- `phase_b_evaluation/W04_Multimodal_Eval.ipynb`: executed aggregate
  notebook for the controlled Idefics2 public-image benchmark.
- `phase_b_evaluation/W04_Evaluation_Report.md`: consolidated method,
  findings, system cost, incident record, limitations, and requirement audit.
- `phase_b_evaluation/W04_Mid_Review_Deck.pptx` and
  `W04_Mid_Review_Speaker_Script.md`: eight-slide 5–8 minute midpoint review.
- `phase_b_evaluation/W04_Midpoint_Evaluation_Rubric.md`: completed intern
  self-assessment and evidence map; supervisor joint scoring and signatures
  remain intentionally blank for the review meeting.
- `phase_b_evaluation/W04_Robustness_Summary_v0.1.0.json`,
  `W04_Multimodal_Summary_v0.1.0.json`, and
  `W04_System_Performance_Summary_v0.1.0.json`: submission-safe aggregate
  quality, latency, resource, model-load, and reproducibility evidence.
- `weekly/Wk-04-EvalLog.md`: frozen method, iteration history, controlled
  repairs, mechanism-oriented findings, and next actions.

The Week 4 measured workload contains 686 candidate requests: 420 semantic
paraphrase generations, 126 masked-input generations, 60 controlled public-image
VLM generations, and 80 expanded public-collection Base/RAG requests. Candidate
quality scoring is cross-model and diagnostic because the local Judge did not
pass human-equivalent calibration. Raw prompts, outputs, request traces, AI
scoring rows, and review queues remain in the private experiment archive; the
public repository contains frozen inputs, source attribution, code, aggregate
results, notebooks, reports, and the presentation.

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

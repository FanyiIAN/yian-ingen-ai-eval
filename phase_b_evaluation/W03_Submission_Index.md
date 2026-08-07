# Week 3 Submission Index

**Phase:** B — systematic evaluation and retrieval-augmented generation

**Status:** expanded benchmark complete and submission-ready

**Claim boundary:** reproducible public-source component evaluation, not deployed InGen product performance

## Expanded Week 3 benchmark (recommended entry point)

| Deliverable | Public artifact | Completion evidence |
|---|---|---|
| Expanded evaluation report | [`W03_RAG_Expanded_Benchmark_Report.md`](W03_RAG_Expanded_Benchmark_Report.md) | Architecture, corpus/question design, retrieval ablation, three-model Base/RAG results, performance, limitations and next steps |
| Expanded knowledge base | [`W03_RAG_Expanded_Knowledge_Base_v0.6.0.yaml`](W03_RAG_Expanded_Knowledge_Base_v0.6.0.yaml) | 331 governed units from three frozen current official pages |
| Expanded question set | [`W03_RAG_Expanded_Eval_Set_v0.6.0.yaml`](W03_RAG_Expanded_Eval_Set_v0.6.0.yaml) | 40 Fari/Senpai questions; 100 required and 40 forbidden scoring points |
| Frozen three-model run configuration | [`W03_RAG_Expanded_MultiModel_Run_Config_v0.6.2.yaml`](W03_RAG_Expanded_MultiModel_Run_Config_v0.6.2.yaml) | BGE-M3, persistent Chroma, BGE reranker, top-10 context and detailed Base/RAG prompts |
| Final retrieval report | [`W03_RAG_Expanded_Final_Retrieval_Report_v0.6.2.md`](W03_RAG_Expanded_Final_Retrieval_Report_v0.6.2.md) | Mean evidence recall 0.9021; 30/40 full-evidence rows; zero metadata leakage |
| Retrieval ablation | [`W03_RAG_Expanded_Retrieval_Ablation_Report_v0.2.0.md`](W03_RAG_Expanded_Retrieval_Ablation_Report_v0.2.0.md) | Ten top-k/reranker variants; top-10 reranked selected |
| Three-model generation audit | [`W03_RAG_Expanded_Three_Model_Report_v0.6.2.md`](W03_RAG_Expanded_Three_Model_Report_v0.6.2.md) | 240/240 Base/RAG rows completed; shared-input audit passed; latency/resource and citation evidence retained |
| Warm-path performance summary | `W03_RAG_Expanded_Performance_Summary_v0.1.0.{md,json,csv}` | Llama Base/RAG stage timing, complete question-to-response latency, RAM and GPU measurements |
| Local automatic evaluation | [`W03_RAG_Expanded_Three_Model_RAGAS_Report_v0.6.2.md`](W03_RAG_Expanded_Three_Model_RAGAS_Report_v0.6.2.md), [`summary JSON`](W03_RAG_Expanded_Three_Model_RAGAS_Summary_v0.6.2.json) | 240/240 resumable local RAGAS rows with finite-row coverage and Judge limitations |

## Original Week 3 deliverables

| Week 3 reference deliverable | Public artifact | Completion evidence |
|---|---|---|
| Extended three-model benchmark notebook | [`W03_Extended_Benchmark.ipynb`](W03_Extended_Benchmark.ipynb) | 35 scenarios × three models; six code cells executed, zero errors |
| RAG evaluation notebook | [`W03_RAG_Evaluation.ipynb`](W03_RAG_Evaluation.ipynb) | LangChain/BGE-M3/Chroma/Llama Base-vs-RAG evaluation; four code cells executed, zero errors |
| Week 3 evaluation memo | [`W03_Evaluation_Memo.md`](W03_Evaluation_Memo.md) | Three-model boundary, RAG trade-off, ablation, failure analysis and limitations |
| Weekly evaluation log | [`../weekly/Wk-03-EvalLog.md`](../weekly/Wk-03-EvalLog.md) | Iterations, decisions, settings, results and reproducibility records |

## Supporting evidence

| Evidence | Artifact | Result |
|---|---|---|
| Original three-model RAG confirmation | [`W03_RAG_Three_Model_Blind_Report.md`](W03_RAG_Three_Model_Blind_Report.md) | 48/48 generations and 48/48 local RAGAS rows |
| Original AI qualitative calibration | [`W03_RAG_AI_Calibration_Report.md`](W03_RAG_AI_Calibration_Report.md) | Eight Llama RAG answers; 4.375/5 relevance, 0.872619 weighted coverage, 30/31 supported claims |
| Original retrieval ablation | [`W03_RAG_Retrieval_Ablation_Report.md`](W03_RAG_Retrieval_Ablation_Report.md) | 18/18 small-corpus variants |
| Representativeness audit | [`W03_RAG_Benchmark_Representativeness_Audit.md`](W03_RAG_Benchmark_Representativeness_Audit.md) | Identified the original set as a smoke benchmark and specified this expansion |
| Failure taxonomies | [`W03_Failure_Taxonomy.md`](W03_Failure_Taxonomy.md), [`W03_RAG_Failure_Taxonomy.md`](W03_RAG_Failure_Taxonomy.md) | Model-behavior and causal RAG failure codes |
| Meeting package | [`W03_RAG_Architecture_Review_7min.pptx`](W03_RAG_Architecture_Review_7min.pptx), [`W03_RAG_Architecture_Review_Speaker_Script.md`](W03_RAG_Architecture_Review_Speaker_Script.md) | Seven-slide architecture/results/limitations review |

## Collection and claim boundaries

- Expanded public metrics use only three current official website snapshots.
- The June 2026 internship material remains in a separate private collection.
  It was not mixed with the public collection or pooled into expanded metrics.
- Knowledge-unit claim status is preserved. Forward-looking design statements
  are not presented as deployed-product evidence.
- The 40 questions were frozen before generation but were authored from the
  same three pages; the benchmark is a controlled component test, not a fully
  independent user-distribution sample.
- Automatic RAGAS and the local Mistral Judge are diagnostics, not usability
  percentages or a validated three-model leaderboard.

## Verification

- `76/76` Week 3 unit and contract tests passed under the frozen Python 3.11 environment.
- All 331 knowledge units are non-empty and exact-normalized unique.
- All 40 questions have complete hidden rubrics and evidence mappings.
- Three-model generation completed `240/240` rows with a passing shared-input audit.
- Raw prompts, contexts, outputs, hashes, evaluator rows and resource traces are retained privately.
- Public artifacts exclude credentials, RunPod connection details, private workspace paths and June private content.

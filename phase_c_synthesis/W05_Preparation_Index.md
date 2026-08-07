# Week 5 Independent Preparation Index

**Phase:** C - methodology synthesis and PIC 2.0 analysis  
**Status:** independent preparation complete; production Week 5 findings deferred  
**Seed:** `42`  
**Compute used here:** local CPU only; no RunPod or external model API

## What is ready

| Preparation component | Artifact | Completion evidence |
|---|---|---|
| Frozen RAG ablation contract | `W05_RAG_Optimisation_Config_v0.1.0.yaml` | Exact 256/512/1024 x top-k 1/3/5 x reranking off/on grid; 18 unique variants |
| Controlled-comparison registry | `W05_RAG_Optimisation.py` | 45 matched pairs differ in exactly one factor; seed-42 randomized execution order |
| Local small-model integration path | `W05_RAG_Optimisation.ipynb` | Executed 54/54 synthetic rows on cached FLAN-T5-small and MiniLM using CPU and offline mode |
| Public-safe smoke fixture | `W05_RAG_Local_Smoke_Fixture_v0.1.0.yaml` | Three synthetic long documents and three versioned Senpai-like questions |
| RAG contract tests | `W05_RAG_Optimisation_Tests.py` | Factor grid, matched contrasts, metrics, Pareto logic, fixture, limitation, and deterministic order tests |
| PIC 2.0 evidence slots | `W05_PIC20_Evidence_Registry_v0.1.0.yaml` | Six proposed capability-proxy mappings; terminology conflicts and missing evidence remain explicit |
| Cross-week row schema | `W05_Normalized_Result_Schema_v0.1.0.json` | Requires exact model revision, evaluation-set version, seed, platform, scenario, severity, score, and failure evidence |
| Accumulated-result analyzer | `W05_Evaluation_Data_Analysis.py` | Stratified platform/model summaries, failure distributions, severity relationship, and surprise-review queue |
| Analysis contract tests | `W05_Evaluation_Data_Analysis_Tests.py` | Traceability, categorical failures, duplicate protection, and manual-mechanism boundary tests |

## Local smoke interpretation

The smoke run uses `google/flan-t5-small` revision
`0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab` for generation and
`sentence-transformers/all-MiniLM-L6-v2` revision
`c9745ed1d9f207416be6d2e6f8de32d1f16199bf` as a bi-encoder reranking
surrogate. It produced all 54 expected rows. The 256, 512, and 1024 token
settings produced 15, 9, and 6 chunks respectively, so all chunking branches
were exercised.

The local generator's mean reference-token-F1 proxy ranged from `0.000000` to
`0.150030`, and required-term coverage ranged from `0.000000` to `0.150000`.
This is useful failure evidence for the adapter and quality gates: the run proves
connectivity and traceability, but the smoke-only Pareto set is not a Week 5
configuration recommendation.

## Intentionally deferred

- Final `W05_PIC20_Model_Analysis.md`: requires reviewed, traceable Phase B
  findings for each class; the registry currently contains no inserted finding.
- Production RAG ablation: requires the accepted frozen Senpai subset, the true
  BGE cross-encoder path, the fixed Week 3 generator/prompt/evaluator stack, and
  comparable warm-path timing.
- Final Faithfulness, Relevance, and Coverage values: local lexical proxies are
  not RAGAS metrics and are never renamed as such.
- Cross-week dataset findings: the analyzer is ready, but adapters must normalize
  accepted Week 2-4 row-level evidence without pooling unlike evaluation families.
- `weekly/Wk-05-EvalLog.md`: it should document the final Week 5 evaluation and
  largest PIC evaluation gap, not a preparation run alone.

## Verification

From `phase_c_synthesis` in the repository:

```powershell
python W05_RAG_Optimisation.py --mode validate
python W05_RAG_Optimisation_Tests.py
python W05_Evaluation_Data_Analysis_Tests.py
```

The offline integration path additionally requires the two exact model revisions
to be present in the local Hugging Face cache. It does not fall back to a network
download. Raw smoke rows and local paths remain in the private support workspace.

## Claim boundary

This preparation is a public-safe, synthetic infrastructure test. It is not a
deployed-product evaluation, a PIC implementation claim, a production RAG
configuration comparison, or evidence that any local smoke variant is optimal.

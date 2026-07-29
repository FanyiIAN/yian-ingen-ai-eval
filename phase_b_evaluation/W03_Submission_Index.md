# Week 3 Submission Index

**Phase:** B — systematic evaluation and RAG  
**Status:** submission-ready  
**Claim boundary:** reproducible component evaluation on public source snapshots,
not deployed InGen product performance

## Required deliverables

| Week 3 reference deliverable | Public artifact | Completion evidence |
|---|---|---|
| Extended three-model benchmark notebook | [`W03_Extended_Benchmark.ipynb`](W03_Extended_Benchmark.ipynb) | 35 scenarios × 3 models; 6/6 code cells executed, 0 errors |
| RAG evaluation notebook | [`W03_RAG_Evaluation.ipynb`](W03_RAG_Evaluation.ipynb) | LangChain/BGE-M3/Chroma/Llama Base-vs-RAG evaluation; 4/4 code cells executed, 0 errors |
| Week 3 evaluation memo | [`W03_Evaluation_Memo.md`](W03_Evaluation_Memo.md) | Three-model boundary, RAG trade-off, top three failure patterns, ablation, and limitations |
| Weekly evaluation log | [`../weekly/Wk-03-EvalLog.md`](../weekly/Wk-03-EvalLog.md) | Iterations, decisions, inputs/settings, results, runtime repairs, and reproducibility |

## Main supporting evidence

| Evidence | Artifact | Result |
|---|---|---|
| Three-model public RAG confirmation | [`W03_RAG_Three_Model_Blind_Report.md`](W03_RAG_Three_Model_Blind_Report.md) | 48/48 generations and 48/48 RAGAS rows; Llama relevance `0.290288 → 0.538142` on 7 uninspected pairs |
| AI qualitative calibration | [`W03_RAG_AI_Calibration_Report.md`](W03_RAG_AI_Calibration_Report.md) | 8 Llama RAG answers; `4.375/5` relevance, `0.872619` weighted coverage, 30/31 supported claims, 0 forbidden violations |
| Retrieval ablation | [`W03_RAG_Retrieval_Ablation_Report.md`](W03_RAG_Retrieval_Ablation_Report.md) | 18/18 chunk/top-k/reranker variants; top-k 3 retained for the smoke corpus |
| Benchmark representativeness | [`W03_RAG_Benchmark_Representativeness_Audit.md`](W03_RAG_Benchmark_Representativeness_Audit.md) | Current set accepted as smoke/helpfulness evidence; larger frozen set specified as follow-up |
| Failure analysis | [`W03_Failure_Taxonomy.md`](W03_Failure_Taxonomy.md) and [`W03_RAG_Failure_Taxonomy.md`](W03_RAG_Failure_Taxonomy.md) | Three-level model-behavior hierarchy plus causal RAG failure codes |
| Reproducible aggregate notebook | [`W03_RAG_Three_Model_Blind_Evaluation.ipynb`](W03_RAG_Three_Model_Blind_Evaluation.ipynb) | 3/3 code cells executed, 0 errors; loads frozen aggregate without GPU/Judge calls |

## Collection and test boundaries

- The full FLAN/Mistral/Llama Base/RAG plus RAGAS comparison uses only the
  official public collection: four source snapshots, 16 chunks, and 30 atomic
  facts.
- The June internship material remains in a separate private collection.
  Its retrieval smoke passed 6/6 and Llama completed 12/12 Base/RAG
  generations. It was not mixed with public data, pooled into public metrics,
  or expanded into a full three-model comparison.
- The public eight-question blind set is useful for pipeline smoke and
  RAG-helpfulness evidence, but is too small and easy for production-usability
  inference. The next iteration should freeze 24–32 questions over 100+ chunks
  and rerun the same pipeline.
- The Week 2-derived three-model scenario scores remain diagnostic because the
  inherited Prometheus Judge failed calibration and the former held-out set was
  inspected. This does not block the Week 3 RAG component conclusion.

## Final verification

- `53/53` Week 3 unit and contract tests passed.
- `32/32` Week 3 Python files parsed and byte-compiled.
- All three submission notebooks executed with zero error outputs.
- `16/16` YAML and `2/2` JSON artifacts parsed.
- Local Markdown links and README artifact targets resolved.
- Public-sensitive scan found no Hugging Face token, external API credential,
  private key, RunPod hostname, private workspace path, or private knowledge
  content. The evaluator's `local-loopback-only` placeholder is not a credential
  and reaches only the local vLLM-compatible endpoint.

Automatic relevance remains a diagnostic signal rather than a usability
percentage. The small test set, uncalibrated local Judge, strict citation-format
failures, and incomplete stress coverage are transparent follow-up items, not
missing Week 3 deliverables.

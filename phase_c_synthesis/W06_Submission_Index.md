# Week 6 Submission Index

## Required deliverables

| Artifact | Purpose | Status |
|---|---|---|
| `W06_Eval_Methodology_Report.md` | Benchmark rationale, rubric reliability, model validity, RAG limits, and three known gaps | Complete |
| `W06_Eval_Paper_Sketch.md` | 150-word abstract, introduction, related work, and reproducible methodology | Complete |
| `W06_Eval_Paper_Self_Critique.md` | Contribution, likely reviewer objection, and specific literature gap | Complete |
| `../weekly/Wk-06-EvalLog.md` | Work log and required reproducibility reflection | Complete |

## Reproducibility artifacts

| Artifact | Purpose |
|---|---|
| `W06_Evidence_Registry_v1.0.0.json` | Twelve frozen Week 1–5 input paths, hashes, roles, evidence policy, temporal audit, and literature registry |
| `W06_Evidence_Synthesis.py` | Dependency-free evidence extraction, source verification, summary generation, and optional environment record |
| `W06_Evidence_Synthesis_Tests.py` | Contract tests for inputs, reports, abstract, calibration gate, factorial/Pareto logic, and output freshness |
| `W06_Evidence_Summary_v1.0.0.json` | Machine-readable synthesis used by the report and future Week 7 dashboard |
| `W06_Claim_Evidence_Matrix_v1.0.0.csv` | Seven principal claims with scope, status, replication expectation, and causal-language boundary |
| `W06_RunPod_Deployment_Verification_v1.0.0.json` | Clean Linux/RunPod verification record; generated remotely and copied back |
| `W06_RunPod_Deployment_Report.md` | Deployment scope, A40-capacity incident, CPU-only fallback, exact commands, matched hashes, and interpretation boundary |

## One-command local verification

From the repository root with Python 3.11 or newer:

```bash
python phase_c_synthesis/W06_Evidence_Synthesis.py --verify-only
python -m unittest phase_c_synthesis.W06_Evidence_Synthesis_Tests -v
```

The synthesis script uses only the Python standard library. It does not download a model, contact a service, call an LLM Judge, or require a GPU.

## Evidence boundary

No Week 6 claim is labeled `validated_result`. Week 2 model scores remain `diagnostic_failed_calibration`; Week 3–5 AI-assisted quality scores remain `diagnostic_uncalibrated`; hashes, counts, registered factor levels, matched definitions, and frozen-row Pareto computation are deterministic audit evidence. The artifacts describe public proxy evaluations, not deployed InGen products or proprietary PIC runtime performance.

# Phase A-B Midpoint Evidence Inventory

This inventory is the machine-readable appendix to `Phase_AB_Midpoint_Report.pdf`.
It contains only public/synthetic benchmark evidence. The separately tested June private
collection is not included in any item-level export or public aggregate.

## Complete row-level files

| Artifact | Rows | Included evidence |
|---|---:|---|
| `phase_a_design/W02_Baseline_Eval_Results.csv` | 70 | Full semantic prompt, input, candidate answer, three Judge formulations, all raw Judge outputs, final diagnostic labels, seed and revisions. |
| `phase_b_evaluation/Phase_AB_W03_RAG_Item_Results.csv` | 240 | Every Base/RAG answer, retrieved public contexts, RAGAS metric outputs/reasons, settings, hashes, timings and resources. |
| `phase_b_evaluation/Phase_AB_W04_Robustness_Item_Results.csv` | 546 | Every original/paraphrase/mask row, prompt/input/answer, raw cross-Judge output, diagnostic labels and request telemetry. |
| `phase_b_evaluation/Phase_AB_W04_VLM_Item_Results.csv` | 120 | Every image-condition/model row, prompt/answer, source/pixel hashes, raw Judge output, rubric dimensions and telemetry. |
| `phase_b_evaluation/Phase_AB_W04_RAG_Performance_Item_Results.csv` | 80 | Every matched Base/RAG request with retrieval stages, prompt/output lengths, response, contexts and resource trace. |

## Prompt and calibration references

- Week 2 candidate prompt: `phase_a_design/W02_Prompt_Spec_v0.4.0.yaml`.
- Week 2 three-formulation Judge: `phase_a_design/W02_Prometheus_Judge_Spec_v0.8.3.yaml`.
- Week 2 frozen calibration rows and report: the public calibration script/spec plus evidence registry; raw calibration archive is retained privately.
- Week 3 Base/RAG prompts and settings: `phase_b_evaluation/W03_RAG_Expanded_MultiModel_Run_Config_v0.6.2.yaml`.
- Week 3 AI calibration: `phase_b_evaluation/W03_RAG_AI_Calibration_Annotations_v0.3.0.yaml` and `W03_RAG_AI_Calibration_Report.md`.
- Week 4 text prompts/masks: `phase_b_evaluation/W04_Robustness_Run_Config_v0.1.0.yaml` and `W04_Robustness_Mask_Spans_v0.1.0.yaml`.
- Week 4 VLM prompts/rubric: `phase_b_evaluation/W04_Multimodal_Scenarios_v0.1.0.yaml`.
- Evidence hashes and interpretation boundaries: `phase_b_evaluation/Phase_AB_Midpoint_Evidence_Registry.csv`.

## Important logging limitation

The Week 3 RAGAS wrapper persisted every per-metric value, retry count, latency and error reason,
but the RAGAS library did not expose the local Judge's complete raw response text. Week 2 and
Week 4 raw Judge outputs are retained row by row. This limitation is disclosed rather than
reconstructing or fabricating missing Judge text.

## Row-count checks

- Week 2: 70 candidate rows and 630 embedded Judge traces.
- Week 3 RAG: 240 candidate-metric rows, 40 retrieval rows and 8 AI-calibration rows.
- Week 4: 546 text, 120 VLM and 80 RAG performance rows = 746 measured request rows.

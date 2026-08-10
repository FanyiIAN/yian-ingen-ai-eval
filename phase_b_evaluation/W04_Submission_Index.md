# Week 4 Submission Index

**Phase:** B  
**Week:** 4 — semantic, masked-input, multimodal, and system-cost evaluation  
**Seed:** 42  
**Public-content rule:** confidential/private-source records and operational logs remain in the private archive. The midpoint package additionally publishes sanitized row-level public/synthetic benchmark exports, including candidate/Judge text and request metrics, so the supervisor can audit every reported item without exposing the June private collection, secrets, or private paths.

## Primary deliverables

| Reference deliverable | Submission artifact | Purpose |
|---|---|---|
| Semantic perturbation and masked-input notebook | `W04_Robustness_Eval.ipynb` | Executed aggregate tables and curves for three text models. |
| Multimodal notebook | `W04_Multimodal_Eval.ipynb` | Executed two-architecture condition, robustness, and efficiency comparison. |
| Phase A–B mid-point deck | `Phase_AB_Midpoint_Review_Deck.pptx` | Eight-slide Weeks 1–4 programme, findings, VLM comparison, system cost, and evidence boundary. |
| Presentation talk track | `Phase_AB_Midpoint_Review_Speaker_Script.md` | 5–8 minute speaking script; the same text is embedded in slide notes. |
| Mid-point rubric | `Phase_AB_Midpoint_Evaluation_Rubric.md` | Intern evidence-based self-assessment and the required joint scoring/signature fields. |
| Phase A-B consolidated paper | `Phase_AB_Midpoint_Report.pdf`, `Phase_AB_Midpoint_Report.md` | Formal Weeks 1-4 paper-style report with methods, results, limitations, reproducibility, calibration, item-level appendices, and references. |
| Weekly log | `../weekly/Wk-04-EvalLog.md` | Method, mechanism-oriented findings, iterations, failures, and next actions. |
| Consolidated report | `W04_Evaluation_Report.md` | Scope, frozen method, results, cost evidence, repairs, validity, and requirement audit. |

## Submission-safe result evidence

| Artifact | Contents |
|---|---|
| `Phase_AB_Midpoint_Evidence_Inventory.md` | Navigation for the complete midpoint prompt, Judge, calibration, result, and telemetry evidence. |
| `Phase_AB_Midpoint_Evidence_Registry.csv` | Artifact hashes, model/Judge scope, seed, evidence status, row counts, and claim boundaries. |
| `Phase_AB_W03_RAG_Item_Results.csv` | 240 sanitized public Base/RAG rows with questions, answers, retrieved public contexts, RAGAS outputs, settings, timing, resources, and hashes. |
| `Phase_AB_W04_Robustness_Item_Results.csv` | 546 sanitized public/synthetic robustness rows with prompts, answers, raw cross-Judge outputs, scores, and request telemetry. |
| `Phase_AB_W04_VLM_Item_Results.csv` | 120 sanitized public-image rows with source/pixel hashes, prompts, answers, raw Judge outputs, rubric scores, and request telemetry. |
| `Phase_AB_W04_RAG_Performance_Item_Results.csv` | 80 matched Base/RAG rows with answers, retrieved public contexts, stage timing, and resource traces. |
| `W04_Robustness_Summary_v0.1.0.json` | Per-model parse coverage, semantic consistency, stable pass/fail, flips, masked curves, failure counts, and review-flag counts. |
| `W04_Robustness_Curves_v0.1.0.csv` | Model × mask-level curve table. |
| `W04_Robustness_Results_v0.1.0.md` | Human-readable diagnostic text-robustness summary. |
| `W04_Multimodal_Summary_v0.1.0.json` | Condition/platform rubric aggregates, perturbation drop, decision consistency, and review-flag counts. |
| `W04_Multimodal_Platform_Conditions_v0.1.0.csv` | Idefics2 platform × image-condition aggregates. |
| `W04_Multimodal_Results_v0.1.0.md` | Human-readable diagnostic VLM summary. |
| `W04_Multimodal_Architecture_Comparison_v0.2.0.json` | Matched Idefics2/LLaVA quality, robustness, efficiency, Judge-control, and source-hash summary. |
| `W04_Multimodal_Architecture_Comparison_v0.2.0.csv` | Long-form two-VLM comparison table. |
| `W04_Multimodal_Architecture_Comparison_v0.2.0.md` | Human-readable controlled architecture comparison. |
| `W04_System_Performance_Summary_v0.2.0.json` | Static checkpoint/model-load evidence and complete token, stage-latency, throughput, and resource statistics including both VLMs. |
| `W04_System_Performance_Summary_v0.2.0.csv` | Long-form per-group metric table with p50/p90/p95/max and missing counts. |
| `W04_System_Performance_Summary_v0.2.0.md` | Compact cold-load and warm-path performance table. |

## Frozen inputs, provenance, and configs

| Artifact group | Files |
|---|---|
| Text input bank | `W04_Robustness_Inputs_v0.1.0.jsonl`, `W04_Robustness_Input_Manifest_v0.1.0.json` |
| Paraphrase review and mask spans | `W04_Robustness_Semantic_Equivalence_Review_v0.1.0.yaml`, `W04_Robustness_Mask_Spans_v0.1.0.yaml` |
| Text run contract | `W04_Robustness_Run_Config_v0.1.0.yaml` |
| Multimodal scenario and input bank | `W04_Multimodal_Scenarios_v0.1.0.yaml`, `W04_Multimodal_Inputs_v0.1.0.jsonl`, `W04_Multimodal_Input_Manifest_v0.1.0.json` |
| Public images and attribution | `w04_multimodal_images/` (20 lossless PNG files), `W04_Multimodal_Attribution_v0.1.0.csv` |
| VLM run contracts | `W04_Multimodal_Run_Config_v0.1.0.yaml`, `W04_Multimodal_Run_Config_LLaVA_v0.2.0.yaml` |
| System-measurement contract | `W04_System_Performance_Metrics_Spec.md` |

## Implementation

| Function | Files |
|---|---|
| Text input freezing and validation | `W04_Robustness_Data.py`, `W04_Freeze_Robustness_Inputs.py`, matching `*_Tests.py` files |
| Candidate text generation | `W04_Text_Robustness_Runner.py`, `W04_Text_Robustness_Runner_Tests.py` |
| Public-image freezing and perturbation | `W04_Multimodal_Data.py`, `W04_Multimodal_Data_Tests.py` |
| Pinned VLM download and two-architecture generation | `W04_Download_Pinned_Idefics2.py`, `W04_Download_Pinned_VLM.py`, `W04_Multimodal_Runner.py`, and matching tests |
| Resource/latency collection | `W04_Resource_Monitor.py`, `W04_RAG_Performance_Runner.py`, matching tests |
| Diagnostic AI scoring and deterministic repair | `W04_AI_Assisted_Scoring.py`, `W04_AI_Score_Repair.py`, matching `*_Tests.py` files |
| Aggregate analysis | `W04_Robustness_Analysis.py`, `W04_Multimodal_Analysis.py`, `W04_Multimodal_Comparison.py`, `W04_Performance_Summary.py`, matching tests |
| RunPod dependencies | `W04_requirements_runpod.txt` |

## Reproduce the submission-facing notebooks

From the repository root after installing `requirements.txt`:

```powershell
jupyter nbconvert --to notebook --execute --inplace phase_b_evaluation/W04_Robustness_Eval.ipynb
jupyter nbconvert --to notebook --execute --inplace phase_b_evaluation/W04_Multimodal_Eval.ipynb
```

These commands read only the committed aggregate artifacts. They do not download weights, start a GPU, contact an API, or rerun the uncalibrated Judge.

## Verification

```powershell
python -m unittest discover -s phase_b_evaluation -p "W04*_Tests.py"
python -m json.tool phase_b_evaluation/W04_Robustness_Summary_v0.1.0.json > $null
python -m json.tool phase_b_evaluation/W04_Multimodal_Summary_v0.1.0.json > $null
python -m json.tool phase_b_evaluation/W04_Multimodal_Architecture_Comparison_v0.2.0.json > $null
python -m json.tool phase_b_evaluation/W04_System_Performance_Summary_v0.2.0.json > $null
```

Final verification additionally checks Python compilation, YAML/JSON parsing, notebook execution, PowerPoint rendering and overlap reports, file hashes, relative links, and a public-tree secret/private-path scan.

## Known completion boundary

All code-generated Week 4 evidence can be completed and verified without supervisor input. The reference requires the Phase A–B mid-point rubric to be jointly scored and signed. `Phase_AB_Midpoint_Evaluation_Rubric.md` therefore contains the completed intern self-assessment and blank supervisor/joint signature fields; those fields must be completed at the review meeting and are not fabricated by the pipeline.

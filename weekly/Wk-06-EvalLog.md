# Week 6 Evaluation Log

**Phase:** C — Evaluation methodology and research-paper synthesis  
**Seed:** `42` for every cited Week 2–5 run  
**Evidence registry:** `phase_c_synthesis/W06_Evidence_Registry_v1.0.0.json`

## Work completed

- Audited the Week 6 reference requirements and concepts primer, including publication-standard methodology, temporal validity, Krippendorff's alpha, RAG metric limits, known gaps, and self-critique.
- Revisited primary literature from Week 1: HELM, RAGAS, PromptRobust, MMMU, TimeBench, CALVIN, ALFRED, NIST AI RMF, and LLM-as-Judge evaluation; also applied Whitesides' outline-first writing advice and ACL publication guidance.
- Froze 12 public Week 1–5 evidence sources by SHA-256 and built a standard-library synthesis pipeline that regenerates a JSON evidence summary and CSV claim-evidence matrix.
- Wrote the required methodology report, six-page-equivalent research sketch with an exact 150-word abstract, and a one-page structured self-critique.
- Added automated gates for source freshness, the 35-scenario/5-platform design, severity and split counts, temporal triggers, Judge calibration, the Week 5 18-cell factorial/Pareto contract, required report sections, and literature engagement.
- Deployed the minimum verification bundle to a clean RunPod Linux environment and compared regenerated artifact hashes with the local run. The unavailable A40 did not affect Week 6 because synthesis performs no model inference; the remote run was explicitly recorded as CPU-only portability verification.

## Key finding

Three Judge prompt formulations were relatively consistent on Task Accuracy (`α=0.8772`) but not Failure Mode (`α=0.5673`). More importantly, the separate frozen-label calibration result (`α=0.7551`) missed the preregistered `0.80` threshold. The apparent agreement therefore cannot validate model rankings. The report keeps Week 2 scores as failed-calibration diagnostics and later AI-assisted RAG/VLM scores as uncalibrated diagnostics.

The corrected long-source RAG test shows the value of matched component evaluation: on 40 questions, RAG increased diagnostic answer relevance by `0.655667` and required-point coverage by `0.522917`, but added `8030.48 ms` mean generation latency. The Week 5 factorial identifies three conditional Pareto cells, not one universal optimum.

## Problems and resolutions

| Problem | Resolution | Remaining boundary |
|---|---|---|
| High pooled Judge agreement could be mistaken for validity | Added a separate calibration gate and prohibited `validated_result` claims | Independent domain reviewers are still needed |
| Model training cutoffs are incompletely documented | Used prompt-closed scenarios and an external-time trigger audit | Background-knowledge exposure remains a confound |
| RAG, robustness, VLM, and text results use different units and scorers | Kept families stratified and created a claim-level evidence status | No cross-family readiness aggregate is reported |
| Original RunPod A40 host had no available GPU; a short-lived replacement A40 was also reclaimed during SSH configuration | Used RunPod's supported CPU-start fallback because the Week 6 pipeline has no GPU/model dependency; recorded the change transparently | This run verifies Linux portability, not GPU performance |

## Reflection: the most important evaluation design decision

The **most important evaluation design decision** was to treat evidence status as part of every result rather than as a footnote. A number is accepted only with its model revision, evaluation-set version, seed, source hash, scoring status, and claim boundary. This matters for **reproducibility** because a second evaluator can distinguish three different questions: can they reproduce the files and computation, can they reproduce the score with the same instrument, and does the score support the same external conclusion? Without that distinction, a perfectly reproducible calculation from an uncalibrated Judge could still produce an invalid model-selection claim.

## Next evaluation priority

Before any deployment-readiness conclusion, run a model-blind domain-expert calibration and a closed-loop simulator slice with real time-aligned sensor dropout, recovery, and unsafe-action metrics. This is the shortest path from a reproducible proxy protocol to evidence about physical behavior.


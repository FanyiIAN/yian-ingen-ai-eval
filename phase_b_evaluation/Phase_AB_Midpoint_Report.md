# Phase A-B Midpoint Report: Physical AI Model Evaluation

> **RAG status: historical midpoint snapshot; NOT LATEST for knowledge-base-dependent results.** Week 1–2 and independent Week 4 findings remain applicable. Use `W03_RAG_Long_Source_Corrective_Report_v1.0.0.md` and `W04_RAG_Long_Performance_Report_v1.0.0.md` for the corrected RAG evidence.

**Coverage:** Weeks 1-4 (Phase A: landscape and benchmark design; Phase B: model, RAG, robustness, multimodal, and system evaluation)  
**Intern:** Yian Fan  
**Supervisor:** Iqbal Patel  
**Date:** 10 August 2026  
**Reproducibility default:** seed 42, deterministic decoding, pinned revisions, immutable inputs  
**Claim boundary:** public/synthetic surrogate evidence; no deployed InGen product-performance claim

## Abstract

This report consolidates the first four weeks of a Physical AI evaluation internship into one traceable evidence package. Phase A translated product risks into a frozen 35-scenario, five-platform text benchmark and tested two open models plus three Judge-prompt formulations. Phase B added Llama-3.1-8B-Instruct, a governed LangChain/BGE-M3/Chroma RAG pipeline over 331 public knowledge units and 40 questions, semantic and missing-evidence robustness studies, a controlled Idefics2-versus-LLaVA VLM comparison, and request-level latency/resource telemetry. The central finding is methodological: candidate quality, evaluator reliability, retrieval quality, robustness, and system cost must be reported separately. Mistral and Llama were more complete than FLAN, but the automatic Judge failed calibration; RAG improved diagnostic relevance for Mistral and Llama while long-context generation dominated latency; consistency sometimes represented stable failure; and LLaVA matched Idefics2 closely while using less time and device memory on the frozen proxy. All item-level public/synthetic records are attached as CSV appendices.

## 1. Scope and research questions

The internal internship plan requires a self-contained open-source evaluation programme for Weeks 1-4, with model/version, evaluation set and seed attached to every claim. The study asks: (RQ1) can a reproducible text harness expose candidate and Judge failures; (RQ2) does governed retrieval help answers; (RQ3) are decisions stable under equivalent wording and missing evidence; (RQ4) does a second VLM architecture change quality or cost under controlled inputs; and (RQ5) what resources and latency does each path consume?

## 2. Experimental method

The common evidence contract freezes inputs before candidate generation, uses seed 42 and deterministic decoding, pins model/tokenizer revisions, retains prompt/config hashes, separates generation from scoring, and stores row identifiers, outputs, errors, timings and resource traces. Diagnostic automatic scores are never treated as deployment certification. The current Week 4 public code suite passes 65/65 unit tests. June private material was tested separately and is excluded from public item exports and public aggregates.

The Week 2 benchmark contains 35 scenarios: seven each for Fari, Senpai, Sentinel Prime AI, Aido Rover and Aido Humanoid. Week 3 keeps those scenarios unchanged, adds Llama, and evaluates 40 governed public Fari/Senpai RAG questions. Week 4 creates three meaning-preserving paraphrases per scenario, removes one/two/three of five registered evidence groups for 14 Rover/Sentinel cases, and evaluates 20 public Open Images scenes under clean, Gaussian-noise and brightness conditions with two VLM architectures.

## 3. Main results

### 3.1 Text baseline and Judge reliability

Seventy Week 2 candidate rows completed without generation errors. FLAN produced only seven unique responses and frequently emitted labels/fragments; Mistral produced 35 unique scenario-specific responses. Three Prometheus Judge prompt formulations produced 630 traces, but the frozen 16-item calibration failed ordinal alpha, task-within-one and failure-code gates. Therefore all text scores are diagnostic.

Adding Llama on the unchanged 35 scenarios produced severity-weighted Task/Grounding/paired quality of 3.970/4.050/4.061, compared with Mistral 4.080/4.500/4.288 and FLAN 3.127/3.377/3.725. The result supports a response-completeness observation, not a validated leaderboard.

### 3.2 Governed RAG

The RAG chain is LangChain v1 -> BGE-M3 normalized dense embeddings -> persistent metadata-filtered Chroma -> BGE-reranker-v2-m3 -> top-10 parent context -> candidate model. The public collection contains 331 traceable units; private June data is never pooled. Retrieval mean required-fact recall@10 was 0.9021, with complete evidence for 30/40 questions and zero metadata leakage. Automatic answer relevance changed from Base to RAG by +0.026 FLAN, +0.412 Mistral and +0.378 Llama; RAG faithfulness was 0.847/0.878/0.876. The eight-row AI qualitative calibration of Llama RAG answers yielded 4.375/5 relevance, 0.8726 required-point coverage, 30/31 supported claims and no forbidden claims. It is explicitly AI review, not human certification.

### 3.3 Robustness and multimodal comparison

Semantic decision consistency was 0.914 FLAN, 0.857 Mistral and 0.857 Llama, but FLAN had 25 stable failures. At 60% registered evidence-group removal, mean Task-score drop was 0.357/0.000/0.286. Missing-evidence masking simulates absent structured fields or sensor-derived evidence; it does not mask hidden embeddings.

On the same 20 scenes x 3 image conditions, Idefics2 scored 4.90/4.80/4.75 (clean/noise/brightness) and LLaVA scored 4.80/4.85/4.70. Clean rows were 19 ties and one LLaVA loss. LLaVA reduced all-condition median end-to-end latency from 6.31 s to 4.39 s and device peak from 18.23 GiB to 14.15 GiB. These are public-image proxy results, not deployed camera accuracy.

### 3.4 System cost

The 746 Week 4 request traces cover 546 text robustness rows, 120 VLM rows and 80 matched Base/RAG rows. For Llama, Base p50 was 1.59 s and RAG p50 8.40 s. Retrieval was only 0.409 s p50; generation accounted for about 95% of mean RAG time because prompt median grew from 244 to 1,866 tokens and answer median from 47 to 220.5 tokens. Chroma occupied about 5.7 MiB on CPU/disk. The retrieval transformer stack occupied about 4.61 GiB before Llama; Llama added about 14.97 GiB and request activity up to 1.21 GiB, producing a 20.80 GiB peak. Chroma itself was not a multi-GiB GPU consumer.

## 4. Interpretation and limitations

The principal unresolved risk is evaluator reliability. Parseability, prompt-formulation agreement and correctness are separate; the local Judge remains uncalibrated. RAG demonstrates helpful evidence access on a small governed corpus, but not production readiness. The original 35 scenarios are now regression cases rather than a fresh blind set. The 20-image VLM benchmark omits video, calibrated sensors, temporal fusion and closed-loop control. One NVIDIA A40 profile is hardware-specific. The Week 3 RAGAS wrapper stored metric values, retries, latencies and error reasons but not complete raw local-Judge response text; this is disclosed rather than reconstructed.

## 5. Conclusion and midpoint action

The Phase A-B requirements are substantially complete: landscape, benchmark, three text models, governed RAG, robustness, controlled two-VLM comparison, telemetry, reports and reproducibility evidence are present. The most important Phase C action is to preserve the frozen harness while creating a new sealed evaluation set and a larger independently labelled Judge-calibration bank. Supervisor and intern should complete the six-dimension midpoint rubric in `Phase_AB_Midpoint_Evaluation_Rubric.md`; the report does not pre-fill an official joint score.

## Appendix A. Complete evidence package

The PDF contains compact item-level tables for all 35 Week 2 scenarios, all 40 expanded RAG questions, all 35 Week 4 text scenarios and all 20 VLM scenarios. Complete condition-level records, prompts/answers, raw Judge outputs where logged, settings, hashes and performance traces are in:

- `phase_a_design/W02_Baseline_Eval_Results.csv` (70 rows; 630 embedded Judge traces).
- `phase_b_evaluation/Phase_AB_W03_RAG_Item_Results.csv` (240 rows).
- `phase_b_evaluation/Phase_AB_W04_Robustness_Item_Results.csv` (546 rows).
- `phase_b_evaluation/Phase_AB_W04_VLM_Item_Results.csv` (120 rows).
- `phase_b_evaluation/Phase_AB_W04_RAG_Performance_Item_Results.csv` (80 rows).
- `phase_b_evaluation/Phase_AB_Midpoint_Evidence_Registry.csv` and `Phase_AB_Midpoint_Evidence_Inventory.md`.

Full long-form prompts and code are referenced by versioned Git artifacts rather than duplicated: Week 2 prompt/Judge specs, Week 3 multi-model run config and AI-calibration annotations, Week 4 robustness/mask configs, and Week 4 VLM scenarios/configs.

## References

1. Yian Fan AI Model Evaluation Internship Plan v1, internal project brief, 2026.
2. AI Model Evaluation Concepts Primer, internal teaching note, 2026.
3. Liang et al., Holistic Evaluation of Language Models, TMLR 2023.
4. Es et al., RAGAS, EACL 2024.
5. Yue et al., MMMU, CVPR 2024.
6. Zhu et al., PromptBench/PromptRobust, 2023.
7. NIST AI Risk Management Framework 1.0, 2023.
8. InGen Dynamics public product pages, snapshots accessed July-August 2026.
9. Open Images V7 validation set and attribution metadata.

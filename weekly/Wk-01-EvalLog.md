# Week 1 Evaluation Log

**Week:** 01  
**Theme:** Physical AI evaluation landscape and toolchain setup

This week I evaluated how a model-evaluation workflow must change when model outputs affect physical products. I mapped the InGen product ecosystem to four evaluation dimensions - usefulness/accuracy, safety, robustness and latency - and used Aido Rover as the main embodied anchor. The central finding is that one aggregate score is insufficient: Rover sensor failure, Fari medical-information failure and Sentinel threat-detection failure have different consequences even when their raw error rates are similar. Scenarios therefore need explicit failure conditions and severity, with row-level evidence preserved before aggregation.

I reviewed six public methodology sources: InGen's public product material, HELM, RAGAS, MMMU, PromptBench and NIST AI RMF 1.0. HELM contributed the multi-scenario, multi-metric structure; RAGAS separated retrieval and generation failures; MMMU showed how to control multimodal comparisons and label perception versus reasoning errors; PromptBench motivated meaning-preserving perturbations; NIST connected measurement to governance and deployment action. The combined lesson is that a Physical AI harness should trace product consequence -> capability -> scenario -> perturbation -> metric/rubric -> deployment decision.

I also mapped my previous work into the internship. RAG pipelines transfer to faithfulness/relevance/coverage analysis; masked prediction transfers to Rover sensor-ablation curves; data-validation pipelines transfer to a YAML-to-CSV harness; controlled MRI architecture comparisons transfer to VLM evaluation; and performance analysis transfers to confidence intervals and latency/accuracy trade-offs. My main gap is not implementation but proving that the measurement instrument itself is reliable.

The evaluation-methodology gap I am most interested in closing is **reliable LLM-as-judge evaluation for severity-weighted, safety-relevant outputs**. It matters because open-ended responses cannot always be scored with exact match, yet an unstable judge can create false evidence of improvement. I want to learn how to write non-overlapping rubrics, design three genuinely different judge formulations, measure ordinal agreement with Krippendorff's alpha, inspect disagreement clusters and calibrate automated scores against human labels.

The dedicated `inGen` Python 3.11 environment now passes the Week 1 import and smoke tests. The notebook uses a locally initialized tiny BERT configuration with synthetic inputs and seed 42; it performs no benchmark evaluation, paid API request or model download. My most important open question is the authoritative PIC 2.0 glossary: the reviewed materials assign different meanings to STUM and SEOM. I will ask whether Week 2 should follow the programme capability taxonomy, current public Sentinel semantics, or version both tracks separately.


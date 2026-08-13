# Week 6 Paper Sketch: Structured Self-Critique

## Contribution claim

The sketch claims a **reproducible, evidence-bounded evaluation protocol** for physical-AI model assessment: platform-specific proxy scenarios are stratified by consequence, subjective scores must pass calibration, controlled studies use matched comparisons, and every result carries a status that blocks diagnostic scores from becoming deployment-readiness claims.

This is a methodology contribution, not a new foundation model, a general leaderboard, or proof of product safety. The strongest defensible novelty is the integration of consequence classes, platform anchoring, instrument calibration, cross-family evidence separation, and artifact-level traceability in one small evaluation program. HELM already motivates broad scenario/metric coverage; NIST already requires contextual risk management. The sketch must therefore show that its added value is the operational evidence contract, not the existence of multi-metric evaluation.

## Most likely reviewer objection

**The evaluation is too small and synthetic, uses uncalibrated AI-assisted scoring, and never executes a robot; therefore the “physical AI” framing may look stronger than the evidence.** This objection is valid. Thirty-five text scenarios, 40 long-source questions, and 20 public-image proxies cannot estimate deployed failure rates. The Judge failed its frozen-label gate, and later RAG/VLM scorers were not independently calibrated. Even a perfect response score does not measure action selection, control stability, recovery, or human oversight.

The current answer is to narrow the claim: this is a reproducible *pre-deployment proxy protocol* that exposes where evidence is missing. That is intellectually honest, but may still be insufficient for acceptance if a workshop expects a validated empirical result. A stronger revision would add two independent domain reviewers, confidence intervals for stable deterministic metrics, and one closed-loop simulator experiment linking a response/decision score to action success or safe recovery.

## Threats to the contribution

1. **Reliability could be mistaken for validity.** Pooled Task α=0.877 sounds strong, but Judge-to-label calibration failed at α=0.755 and per-model agreement varied. The paper must keep this failure near every model-comparison statement.
2. **Scenario balance is not deployment prevalence.** Seven items per platform prevents weighting imbalance but says nothing about how often each event occurs. Severity weighting prioritizes consequences; it does not estimate expected harm.
3. **Matched contrasts have a narrow causal scope.** A Week 5 factor delta can be attributed within the registered pipeline, but it cannot establish a universal retrieval mechanism or production optimum.
4. **High robustness can hide stable failure.** FLAN's 0.914 semantic consistency coexisted with 25 stable-fail scenarios. Any single robustness score would be misleading.
5. **Public-source RAG and images are proxies.** Source authority, permission, drift, actual sensor noise, and product interfaces are absent.

## Specific related work requiring deeper engagement

The sketch most needs deeper engagement with **CALVIN (Mees et al., 2022)** and **ALFRED (Shridhar et al., 2020)**. Both evaluate grounded, long-horizon behavior rather than textual decision support, and therefore define the boundary between a “physical-AI proxy evaluation” and an embodied benchmark. A revised paper should compare units of evaluation (response, subgoal, action sequence, episode), failure recovery, environment generalization, and whether language-level safety checks predict episode success.

It should also go beyond HELM's abstract multi-metric similarity by comparing scenario selection and missing-coverage reporting, and beyond RAGAS by discussing evaluator dependence, source correctness, and action safety. Zheng et al.'s LLM-as-Judge bias analysis should motivate counterbalanced ordering, multi-family Judges, and independent human calibration.

## Concrete revision plan

- Add 30–50 stratified, model-blind calibration outputs rated independently by two domain reviewers; preregister acceptance gates by dimension and severity.
- Add a simulator slice with sensor dropout and recovery, measuring unsafe-action rate and time to safe state.
- Repeat the RAG Pareto study on a second corpus, generator, and hardware target; bootstrap matched deltas and report Pareto stability.
- Convert the paper sketch to the official ACL workshop style only after the target call fixes page limits; keep the required Limitations section explicit.
- Preserve the present claim matrix so every revised number remains tied to source hash, model revision, evaluation set, seed, and evidence status.

## Self-assessment

The paper sketch is strongest as a transparent methodology and weakest as an empirical deployment study. Its most credible result is not that one model wins, but that apparently strong automated agreement can coexist with failed calibration—and that a well-designed evidence pipeline must make that incompatibility impossible to hide.


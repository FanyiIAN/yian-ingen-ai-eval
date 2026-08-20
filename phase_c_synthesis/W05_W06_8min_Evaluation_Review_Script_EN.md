# Week 5–6 Evaluation Review — 8-Minute Speaker Script

## Slide 1 — Week 5–6 Evaluation Review (0:00–0:40)

Good morning. This review covers the work completed in Weeks 5 and 6. Week 5 is the main experimental contribution: I rebuilt the RAG study around complete long documents, ran a controlled full factorial, and connected the results to six PIC 2.0 capability tracks. Week 6 then turned those results into a defensible evaluation methodology. The central message is that the project now has a reproducible way to make conditional decisions, while keeping a clear boundary between diagnostic proxy evidence and real product readiness.

## Slide 2 — Three workstreams, one evidence boundary (0:40–1:25)

The scope had three connected workstreams. First, the controlled Senpai RAG experiment tested eighteen configurations, with twenty frozen questions in each cell. Second, I translated accumulated Phase B evidence into six PIC-specific analyses, each with a deployment failure and a readiness metric. Third, Week 6 synthesized the evidence without pooling incompatible metric families. The consistent boundary across all three streams is important: these are public or synthetic component diagnostics. They are useful for choosing the next experiment, but they do not measure an InGen product or a proprietary PIC runtime.

## Slide 3 — Full factorial design (1:25–2:20)

The RAG experiment used a full factorial design. The three factors were chunk size, top-k, and reranking, so every combination produced eighteen cells. Each cell used the same twenty frozen questions. This matters because a matched contrast can compare two cells that differ in exactly one factor, for example reranking off versus on at fixed chunk size and top-k. I also randomized the variant-block order with seed forty-two to reduce order effects. One warm-up request was excluded, cold loading was recorded separately, and the Pareto analysis used warm-path request latency. These controls make the observed within-study deltas interpretable, although they do not prove a universal causal mechanism.

## Slide 4 — Implementation and debugging (2:20–3:15)

The implementation starts with complete long sources, not pre-cut atomic sections. That change made chunk size an operational variable. Documents were chunked, embedded with BGE-M3, optionally reranked with a cross-encoder, and passed to a pinned Llama 3.1 8B generator. A frozen local Mistral checkpoint supplied diagnostic RAGAS-style and coverage scores. Every row retained source hashes, model revisions, prompts, retrieved document identities, timing, and scorer status. During execution, an eight-thousand-token Judge service hit a context-capacity overflow before producing output. The failed batch wrote no rows, and the same checkpoint and metric definitions resumed on a loopback-only sixteen-thousand-token service. This preserved comparability and the audit trail.

## Slide 5 — Pareto result (3:15–4:15)

The Pareto analysis asks whether any tested configuration is better on all three registered objectives: higher Faithfulness, higher required-point Coverage, and lower warm-path latency. Three cells were non-dominated. The fastest frontier point was five-hundred-and-twelve tokens, top-k five, with reranking. The transparent balanced choice was one-thousand-and-twenty-four tokens, top-k five, with reranking: Faithfulness was zero point nine-one-zero, coverage zero point nine-seven-five, and warm p50 latency about ten point nine seconds. I call it balanced because it reaches the highest observed Faithfulness while preserving the top coverage level, but it is a conditional diagnostic recommendation, not a production optimum.

## Slide 6 — Factor effects and interactions (4:15–5:10)

Across forty-five one-factor matched contrasts, increasing top-k improved all three quality metrics on average: about plus zero point zero-six-two Faithfulness, plus zero point one-zero-six answer relevance, and plus zero point zero-eight-three coverage, at roughly six hundred milliseconds of added latency. Cross-encoder reranking also improved the averages, but cost around seven hundred and seventy-three milliseconds. The important qualification is interaction. At fixed top-k three, reranking slightly reduced Faithfulness on average. So the result is not simply that reranking is always better. Its effect depends on how much context is retrieved and on the chunking condition.

## Slide 7 — PIC 2.0 readiness analysis (5:10–6:05)

The PIC analysis changes the question from “which model has the highest overall score?” to “what evidence is required for each capability class?” AMDC has the broadest direct proxy evidence from the controlled VLM rows, but still lacks synchronized multimodal sensor tests. GRPO and HTD-IRL have useful text-plan diagnostics, not executed policies or task graphs. STUM lacks a temporal-horizon benchmark, and SEOM is limited by static-image ceiling effects rather than navigation. CRL-MRS has the largest gap: only one cooperative text scenario and no controlled communication-loss, agent-dropout, or continual-learning experiment. Therefore the next readiness gate should use class-specific failure denominators and executed state transitions.

## Slide 8 — Week 6 evidence synthesis (6:05–7:10)

Week 6 focused on evaluation validity. Three Judge prompt formulations showed strong pooled agreement on Task Accuracy, with alpha zero point eight-seven-seven, but Failure Mode agreement was only zero point five-six-seven. More importantly, calibration against the separate frozen labels reached only zero point seven-five-five, below the preregistered zero point eight threshold. This demonstrates that agreement is not validity: several Judge prompts can agree while sharing the same mistake. I therefore froze twelve evidence inputs by hash, created a seven-claim evidence matrix, and labeled every claim as deterministic, failed-calibration diagnostic, or uncalibrated diagnostic. No claim is labeled validated. The corrected matched RAG comparison still shows a useful trade-off: higher relevance and coverage, but about eight seconds of added mean generation latency.

## Slide 9 — Limitations and next steps (7:10–8:00)

To conclude, Weeks 5 and 6 establish three things. We now have a reproducible experimental contract, a conditional RAG decision supported by a complete factorial, and a PIC-specific map from current proxies to future readiness metrics. The limitations are equally important: the Judges and RAG quality metrics are not independently calibrated; the data are public proxies rather than product traces; latency is hardware-specific; and closed-loop sensor, action, and multi-agent behavior remain untested. The next priorities are model-blind domain-expert calibration, a simulator or hardware slice with time-aligned dropout and recovery, and product-realistic RAG tests with source permissions, drift, adversarial content, and multi-turn use. The final takeaway is that the work is ready to guide the next evaluation, but not to certify deployment readiness.

## Presentation notes

- Expected duration: approximately 8 minutes at a measured technical speaking pace.
- The charts use the latest corrected long-source Week 5 results (`v1.1.0`), not the superseded atomic-section experiment.
- If discussion time is limited, shorten Slide 4 and Slide 7; preserve Slides 5, 6, 8, and 9 because they contain the decision-relevant results and boundaries.

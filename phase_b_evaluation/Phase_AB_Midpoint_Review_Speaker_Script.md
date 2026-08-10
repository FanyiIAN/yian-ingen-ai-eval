# Phase A–B Midpoint Review Speaker Script

**Coverage:** Weeks 1–4  
**Target duration:** 6–8 minutes  
**Slides:** 8

## Slide 1 — Midpoint scope

This is the midpoint review for the first four weeks and both completed phases, not a Week 4-only report. Phase A established the evaluation landscape and benchmark. Phase B extended the same evidence contract to a third language model, governed RAG, robustness, two vision-language architectures, and system cost. I will focus on the main comparisons and limitations; implementation details remain in the weekly reports.

## Slide 2 — Programme progression

Week 1 defined what must be measured when model outputs affect physical products. Week 2 converted that into a frozen thirty-five-scenario benchmark across five InGen product contexts. Weeks 3 and 4 reused the same traceability rules while adding a third model, governed retrieval, paraphrase and missing-evidence tests, a controlled two-VLM comparison, and performance instrumentation. The key continuity is the evidence contract: pinned revisions, seed 42, deterministic decoding, immutable inputs, row hashes, and claim boundaries.

## Slide 3 — Text-model comparison

On the frozen text benchmark, severity-weighted paired quality is 3.725 for FLAN, 4.288 for Mistral, and 4.061 for Llama. The stronger conclusion comes from candidate behavior: FLAN produced only seven unique responses and often copied the one-shot example, while Mistral and Llama produced scenario-specific answers. These bars are not a validated leaderboard. Prometheus failed the frozen calibration gates, and resolved quality coverage is only 13/35 for FLAN, 33/35 for Mistral, and 32/35 for Llama.

## Slide 4 — RAG comparison

The expanded public collection contains 331 traceable knowledge units and forty governed questions. Mean required-fact recall is 0.902, with complete expected evidence for 30/40 questions and zero metadata leakage. Diagnostic answer relevance changes only slightly for FLAN, but rises from 0.223 to 0.635 for Mistral and from 0.264 to 0.642 for Llama. Retrieval itself takes 408.893 milliseconds p50; full RAG takes 8.40 seconds because the prompt and answer are much longer. Private June material remains a separate collection and is excluded from these public metrics.

## Slide 5 — Robustness

Semantic decision consistency is 0.914 for FLAN and 0.857 for both Mistral and Llama. FLAN nevertheless has 25 stable failures, so consistency alone would reward a repeatedly wrong model. The line chart removes one, two, or three of five evidence groups while leaving the response instruction intact. This simulates missing structured inputs from sensor dropout, packet loss, field omission, or extraction failure; it does not mask the model's hidden embeddings. At sixty percent evidence removal, mean Task drops 0.357 for FLAN, 0.000 for Mistral, and 0.286 for Llama.

## Slide 6 — Controlled VLM comparison

The supervisor-requested second architecture is LLaVA-1.5-7B, selected from the reference list. Both VLMs receive the exact same sixty processed images, user prompt, seed, maximum output, rubric, and Llama Judge. Only the architecture and native processor template change. Idefics2 scores 4.900 on clean images; LLaVA scores 4.800. On the twenty clean scenarios, LLaVA has 0 wins, 19 ties, and 1 loss relative to Idefics2, with a mean difference of -0.100 points out of five. This is a controlled public-image proxy, not production camera accuracy.

## Slide 7 — VLM efficiency

Across the same sixty warm requests, Idefics2 has a median end-to-end latency of 6.31 seconds and a device-wide peak of 18.23 GiB. LLaVA has a median of 4.39 seconds and a peak of 14.15 GiB. Time to first token is 768.809 versus 415.615 milliseconds. These values use one A40, batch size one, deterministic decoding, and the same output budget. Cold loading is reported separately; the chart shows warm request behavior.

## Slide 8 — Midpoint conclusion

The first half now has a reusable pipeline for text models, RAG, semantic and missing-evidence robustness, multimodal architecture comparison, and system cost. The main unresolved methodological risk is evaluator validity: automatic scores are useful for diagnosis but did not pass human-equivalent calibration. Phase C can reuse the frozen harness for PIC-aligned synthesis while prioritizing failure mechanisms and quality–cost trade-offs. The midpoint package also includes the six-dimension worksheet. The supervisor should enter their scores, agree the joint scores with me, record the top strength, development opportunity, and concrete next action, and sign the form.

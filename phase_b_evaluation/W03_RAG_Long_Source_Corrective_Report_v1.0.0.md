# Week 3 Long-Source RAG Corrective Report

> **Status: latest public Week 3 RAG knowledge-base result (v1.0.0).** Raw prompts, retrieved passages, model answers, and Judge traces remain private. Automated Judge metrics are diagnostic until human calibration.

## Corrective design

The previous benchmark stored short atomic sections as parent documents, so changing the nominal chunk size often left the actual retrieval units unchanged. This run instead indexes 21 complete official public sources (38,791 words; 915 structural blocks) and asks 40 questions designed for local facts, tables, cross-section synthesis, long-range evidence, terminology/version status, source conflict, and unanswerability. The frozen comparison contains 80 outputs: one base and one RAG answer per question.

At the preflight sizes, the same sources produced 406 chunks at 256 tokens, 185 at 512, and 94 at 1,024. All 58 registered public evidence facts mapped to at least one chunk, confirming that chunk size is now an operational factor rather than a label.

## Results

| Metric | Base | RAG |
|---|---:|---:|
| Required-point coverage | 0.465 | 0.988 |
| Answer relevance | 0.041 | 0.697 |
| Faithfulness to retrieved context | NA | 0.893 |
| Context recall | NA | NA |
| Context precision | NA | NA |
| Generation latency p50 (ms) | 373.7 | 7256.4 |

Retrieval found at least one expected document for all 40 questions. Mean document recall@8 was 1.000, mean evidence-fact recall@8 was 0.900, MRR was 0.975, and metadata leakage was 0. Evidence recall was below 1.0 for 6 questions: W03-LONG-FARI-001, W03-LONG-FARI-004, W03-LONG-FARI-011, W03-LONG-FARI-017, W03-LONG-FARI-018, W03-LONG-FARI-020.

Across the 40 matched base/RAG pairs with finite coverage, the mean RAG-minus-base coverage delta was 0.523; RAG was higher on 36, tied on 4, and lower on 0 questions. This matched contrast controls the question and generator but does not isolate a single RAG mechanism: retrieval, prompt length, and context content change together.

Coverage scoring statuses were `{'parsed': 66, 'parsed_after_deterministic_repair': 14}`. The local Judge sometimes returned valid registered-point scores plus unregistered extra point IDs. The deterministic repair discarded only those extras and recorded their IDs; it never supplied a missing registered-point verdict or changed the Judge's registered-point score.

RAGAS statuses were `{'complete': 80}`. Answer Relevance was applied to both conditions and Faithfulness only to RAG rows with retrieved context. The corrective formal run did not repeat Context Relevance, Context Recall, or Context Precision after an extended five-metric diagnostic produced persistent HTTP-client retries; those fields are `NA`, not zero. Retrieval document/fact recall and weighted required-point Coverage provide the registered context-quality evidence instead.

## Interpretation and limits

Document-level hit rate alone was too forgiving: every question hit an expected source even though 6 questions missed some or all registered evidence facts. The stricter fact-level result is the useful diagnosis for future retrieval changes. A correlation between retrieved-fact coverage and answer quality would still not establish causality; a mechanism claim requires controlled factor changes such as the Week 5 factorial contrasts.

Here, **correlation** means two measurements move together (for example, higher fact recall and higher answer coverage). **Causality** means changing retrieval actually causes the answer improvement, which needs a controlled comparison. A **mechanism** explains how the cause operates—for example, a cross-encoder moves the relevant chunk into the final context, allowing the generator to state a previously missing point. The present Week 3 base/RAG contrast changes several things together, so it supports association and usefulness, not a single-mechanism proof.

The source metadata distinguishes current public design intent from dated background material. Descriptions of planned capabilities are not evidence of deployment, validation, certification, or PIC readiness. The local Mistral Judge is independent from the Llama candidate but remains uncalibrated; its scores are diagnostic, not human ground truth.

The benchmark also tests **versioned terminology management**: store an acronym's expansion with its source, section, and version instead of silently forcing one global meaning. For example, the Fari page expands STUM as “Socially-aware Trajectory Understanding Model,” while two Senpai sections expand SEOM differently (“Safety and Ethics Operations Model” and “Safety & Ethics Oversight Model”). A report should preserve that discrepancy and cite the relevant section rather than inventing a universal canonical expansion.

# Week 3 Expanded Final Retrieval Analysis

> **Status: SUPERSEDED / NOT LATEST for RAG knowledge-base results.** This benchmark used pre-segmented atomic-section parents, so chunk size was not an operational long-document factor. Use `W03_RAG_Long_Source_Corrective_Report_v1.0.0.md` for the latest public RAG result.

- Analyzer: `0.1.0`
- Questions: `40`
- Final top-k: `10`
- Mean evidence-fact recall: `0.9021`
- Full-evidence questions: `30/40`
- Metadata leakage: `0`
- Retrieval latency: mean `551.7 ms`, p95 `677.8 ms`
- Returned context: mean `683.0` tokens, p95 `1032.7`

## Grouped evidence recall

| Group | Items | Mean fact recall | Full evidence |
|---|---:|---:|---:|
| platform=Fari | 20 | 0.8583 | 13/20 |
| platform=Senpai | 20 | 0.9458 | 17/20 |
| difficulty=easy | 6 | 1.0000 | 6/6 |
| difficulty=hard | 17 | 0.8676 | 11/17 |
| difficulty=medium | 17 | 0.9020 | 13/17 |
| answerability=answerable | 37 | 0.8941 | 27/37 |
| answerability=not_established | 3 | 1.0000 | 3/3 |

## Incomplete evidence rows

| Eval ID | Recall | Missing fact IDs |
|---|---:|---|
| W03-EXP-FARI-002 | 0.5000 | FARI-XF0064 |
| W03-EXP-FARI-003 | 0.7500 | FARI-XF0070 |
| W03-EXP-FARI-007 | 0.5000 | FARI-XF0001 |
| W03-EXP-FARI-008 | 0.6667 | FARI-XF0001 |
| W03-EXP-FARI-010 | 0.3333 | FARI-XF0001, FARI-XF0072 |
| W03-EXP-FARI-014 | 0.7500 | FARI-XF0061 |
| W03-EXP-FARI-020 | 0.6667 | FARI-XF0001 |
| W03-EXP-SENPAI-005 | 0.5000 | SENPAI-XF0018 |
| W03-EXP-SENPAI-013 | 0.7500 | SENPAI-XF0122 |
| W03-EXP-SENPAI-014 | 0.6667 | SENPAI-XF0137 |

The timing distribution is the actual one-pass input-build trace. It contains the first-query warm-up effect and is therefore not a pure steady-state serving benchmark.

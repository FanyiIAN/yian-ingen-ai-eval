# Week 3 Expanded-Corpus Retrieval Ablation

- Version: `0.2.0`
- Corpus: `w03_ingen_official_fari_senpai_expanded#0.6.0`
- Benchmark: `w03_ingen_official_fari_senpai_expanded#0.6.0`
- Seed: `42`
- Indexed chunks: `331`

| top-k | Reranker | Fact recall | Full evidence | MRR | Mean units | Mean tokens | Mean ms | p95 ms |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | off | 0.7275 | 21/40 | 1.0000 | 4.0 | 279.8 | 286.5 | 432.2 |
| 4 | on | 0.7879 | 24/40 | 1.0000 | 4.0 | 293.6 | 551.9 | 870.3 |
| 6 | off | 0.7875 | 24/40 | 1.0000 | 6.0 | 394.6 | 441.0 | 735.0 |
| 6 | on | 0.8375 | 28/40 | 1.0000 | 6.0 | 429.7 | 437.3 | 572.8 |
| 8 | off | 0.8250 | 25/40 | 1.0000 | 8.0 | 532.9 | 206.0 | 285.0 |
| 8 | on | 0.8688 | 29/40 | 1.0000 | 8.0 | 556.2 | 378.0 | 473.0 |
| 10 | off | 0.8562 | 27/40 | 1.0000 | 10.0 | 663.8 | 151.2 | 183.8 |
| 10 | on | 0.9021 | 30/40 | 1.0000 | 10.0 | 683.0 | 374.4 | 525.7 |
| 12 | off | 0.8604 | 27/40 | 1.0000 | 12.0 | 798.8 | 168.8 | 223.2 |
| 12 | on | 0.9083 | 31/40 | 1.0000 | 12.0 | 810.4 | 367.0 | 497.2 |

## Interpretation boundary

Document hit rate is reported but is not the selection criterion: all questions are product-filtered, so evidence-fact recall, complete-evidence items, context budget, and latency are more discriminating. Chunk-size variants are not repeated here because the curated atomic units already produce one child chunk each at the registered 256-token setting.
Latency is descriptive rather than a randomized causal comparison: variants ran in fixed order inside one process, so later rows benefit from warmer model and filesystem caches. Parameter selection therefore uses evidence recall and context budget first.

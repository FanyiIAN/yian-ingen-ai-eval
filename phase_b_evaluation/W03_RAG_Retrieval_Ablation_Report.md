# Week 3 RAG Retrieval Ablation

**Ablation:** chunk size `256/512/1024` × top-k `1/3/5` × reranker off/on  
**Benchmark:** `w03_ingen_official_fari_senpai_blind#0.4.0`  
**Seed:** `42`  
**Embedding:** `BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181`  
**Reranker:** `BAAI/bge-reranker-v2-m3@b5160aeac3c6c8fe7beaaaf04c9e0142826b58d1`

| Chunk | top-k | Rerank | Chunks | Doc recall | Fact recall | MRR | Mean context tokens | Mean latency ms |
|---:|---:|:---:|---:|---:|---:|---:|---:|---:|
| 256 | 1 | off | 16 | 1.0000 | 0.7000 | 1.0000 | 103.2 | 102.601 |
| 256 | 1 | on | 16 | 1.0000 | 0.7417 | 1.0000 | 109.6 | 137.881 |
| 256 | 3 | off | 16 | 1.0000 | 1.0000 | 1.0000 | 327.4 | 93.287 |
| 256 | 3 | on | 16 | 1.0000 | 1.0000 | 1.0000 | 330.8 | 131.451 |
| 256 | 5 | off | 16 | 1.0000 | 1.0000 | 1.0000 | 524.0 | 102.956 |
| 256 | 5 | on | 16 | 1.0000 | 1.0000 | 1.0000 | 524.0 | 139.727 |
| 512 | 1 | off | 16 | 1.0000 | 0.7000 | 1.0000 | 103.2 | 98.409 |
| 512 | 1 | on | 16 | 1.0000 | 0.7417 | 1.0000 | 109.6 | 146.643 |
| 512 | 3 | off | 16 | 1.0000 | 1.0000 | 1.0000 | 327.4 | 103.890 |
| 512 | 3 | on | 16 | 1.0000 | 1.0000 | 1.0000 | 330.8 | 151.960 |
| 512 | 5 | off | 16 | 1.0000 | 1.0000 | 1.0000 | 524.0 | 95.098 |
| 512 | 5 | on | 16 | 1.0000 | 1.0000 | 1.0000 | 524.0 | 121.429 |
| 1024 | 1 | off | 16 | 1.0000 | 0.7000 | 1.0000 | 103.2 | 103.942 |
| 1024 | 1 | on | 16 | 1.0000 | 0.7417 | 1.0000 | 109.6 | 126.889 |
| 1024 | 3 | off | 16 | 1.0000 | 1.0000 | 1.0000 | 327.4 | 133.987 |
| 1024 | 3 | on | 16 | 1.0000 | 1.0000 | 1.0000 | 330.8 | 134.592 |
| 1024 | 5 | off | 16 | 1.0000 | 1.0000 | 1.0000 | 524.0 | 102.024 |
| 1024 | 5 | on | 16 | 1.0000 | 1.0000 | 1.0000 | 524.0 | 155.850 |

## Decision

Use top-k 3 without reranking for the current smoke corpus. Top-k 1 loses
evidence facts; reranking recovers only `0.0417` fact recall and adds latency.
Top-k 5 increases context by about 60% over top-k 3 without improving recall.

Chunk size cannot be selected from this experiment because every governed
section is shorter than 256 tokens, so all three settings produce the same
16 chunks. Perfect document recall and MRR reflect a small, metadata-isolated
corpus and are not evidence that the retriever will remain perfect after the
knowledge base becomes larger, noisier, or conflicting.

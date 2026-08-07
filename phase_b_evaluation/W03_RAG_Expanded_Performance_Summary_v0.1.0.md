# Week 3 Expanded RAG Performance Summary

> Hardware/configuration-specific evidence. A missing measurement is not zero. 
> RAG latency is not applicable when retrieval is disabled.

| Model | Family | Condition | n | End-to-end p50 / p95 (ms) | TTFT p50 / p95 (ms) | Generation p50 / p95 (ms) | GPU memory peak max (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|
| llama31_8b_instruct | rag_performance | base | 40 | 1725.6 / 9723.3 | 81.0 / 105.6 | 1722.0 / 9720.6 | 21301.0 |
| llama31_8b_instruct | rag_performance | rag | 40 | 8288.1 / 14261.3 | 809.2 / 1123.5 | 7852.1 / 13780.8 | 21301.0 |

The complete JSON/CSV retain p50, p90, p95, maximum, mean, standard deviation, missing counts, and component-level timing.

The 80 measured rows are warm requests after one explicit RAG warm-up. Cold
persistent-index initialization/verification took 385,840.8 ms with zero new
chunks, and Llama model loading took 61,382.6 ms. Those cold costs are excluded
from the request latency table.

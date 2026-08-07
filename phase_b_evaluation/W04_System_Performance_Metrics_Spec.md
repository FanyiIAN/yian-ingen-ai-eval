# Week 4 System Performance and Resource Metrics Specification

**Version:** 0.1.0  
**Seed:** 42  
**Scope:** Week 4 text robustness and VLM evaluation, with a reusable stage
schema for later RAG runs  
**Claim boundary:** surrogate open-model evaluation on public or synthetic
public-safe inputs; not deployed InGen runtime performance

## 1. Why this is part of the evaluation

Quality without system cost is insufficient for a deployment-oriented model
comparison. Every Week 4 result therefore joins row-level behavioral evidence
to latency, throughput, memory, and hardware evidence. A missing measurement is
stored as `null` with an explicit reason; it is never silently converted to zero.

## 2. Required run identity

Every run manifest records:

- run ID, UTC start/end time, code version or Git commit, and random seed;
- model ID and immutable model revision;
- evaluation-set ID/version/hash and prompt/config hashes;
- device name, GPU count, total GPU memory, CPU model/count, total host RAM;
- Python, PyTorch, Transformers, CUDA and driver versions;
- precision, quantization, batch size, deterministic decoding settings;
- checkpoint path and approximate checkpoint bytes on disk.

No quality or performance aggregate is valid without model revision,
evaluation-set version, and seed.

## 3. Per-request latency fields

All durations use `time.perf_counter_ns()` and are written in milliseconds.

| Field | Definition |
|---|---|
| `queue_wait_ms` | Optional wait before this process begins work. `null` when no queue is instrumented. |
| `input_load_ms` | Load text/image bytes and validate hashes. |
| `image_perturb_ms` | Apply the frozen VLM perturbation. Zero only for an explicitly measured clean no-op. |
| `preprocess_ms` | Tokenization and, for VLMs, image processor execution. |
| `prompt_build_ms` | Render the frozen semantic/chat prompt. |
| `ttft_ms` | Request start to the first generated token observed by a streamer. |
| `generation_ms` | Entry into model generation through final generated token. |
| `decode_ms` | Convert generated token IDs to returned text when not already streamed. |
| `question_to_response_ms` | Request entry through a complete response object, including all enabled stages. |

The timing identity is checked within clock tolerance:

`question_to_response_ms >= every enabled child stage`, and no stage may be
negative. Overlapping stages are not summed as though they were sequential.

### RAG-compatible stage names

Future RAG runs use the same trace object and additionally record:

`query_embedding_ms`, `metadata_filter_ms`, `vector_search_ms`, `rerank_ms`,
`context_assembly_ms`, `retrieval_total_ms`, `prompt_build_ms`, `ttft_ms`,
`generation_ms`, and `question_to_response_ms`.

`retrieval_total_ms` is measured around the complete retrieval call. Component
stages may overlap and therefore do not have to sum exactly to the total.

## 4. Token and throughput fields

- `prompt_tokens`, `output_tokens`, and `total_tokens` use the candidate
  tokenizer after the exact model-specific chat template is applied.
- `output_tokens_per_second = output_tokens / generation_seconds`.
- `decode_tokens_per_second` excludes the first token when a real TTFT is
  available: `(output_tokens - 1) / (generation_end - first_token_time)`.
- If streaming is unavailable, `ttft_ms` and `decode_tokens_per_second` are
  `null`; an end-of-generation estimate must not be labeled TTFT.

## 5. Resource fields

The monitor samples at 250 ms by default and also reads framework peak counters.

| Resource | Required values |
|---|---|
| Process host memory | RSS before, peak, after (MiB) |
| System host memory | used before, peak, after and total (MiB) |
| PyTorch CUDA memory | allocated before/peak/after and reserved before/peak/after (MiB) |
| Device-wide GPU memory | used before/peak/after and total (MiB), when `nvidia-smi` is available |
| GPU utilization | sample mean, p95 and peak (%) |
| GPU power | sample mean and peak (W), when supported |

Process RSS and PyTorch allocated memory answer different questions and must not
be substituted for each other. Device-wide memory may include unrelated
processes; the manifest records visible GPU IDs and process PID.

## 6. Cold, warm and repeated measurements

- Model load is measured once and reported separately as `model_load_ms`.
- At least one warm-up item per model/condition is excluded from steady-state
  latency aggregates but retained in raw traces.
- The frozen benchmark order is reused across models. If randomization is used,
  the permutation and seed are recorded.
- Quality outputs and performance traces share the same `request_id`; separate
  runs may not be joined only by row position.

## 7. Aggregation and reporting

For each model × condition × platform, report:

- count, missing/error count, mean, standard deviation, p50, p90, p95 and max;
- prompt/output token distributions and output tokens/second;
- TTFT, generation and question-to-response latency;
- peak RSS, peak PyTorch allocated/reserved memory, and device memory peak;
- quality–latency and quality–memory comparisons without collapsing them into
  a single undocumented score.

With fewer than 20 observations, p90/p95 are descriptive order statistics and
must be labeled unstable. Bootstrap confidence intervals use the frozen seed
and resample at scenario level, not individual paraphrase row level.

## 8. Week 4 stage maps

### Text robustness

`load row → build model prompt → tokenize → generate/TTFT → decode → score → persist`

Scoring latency is stored separately from candidate question-to-response latency,
because an offline Judge is not part of the candidate response path.

### VLM robustness

`load source image → verify hash → apply condition → processor → generate/TTFT → decode → persist`

Image perturbations use fixed parameters and seeds. Image-load and perturbation
latency remain visible instead of being hidden inside preprocessing.

## 9. Validity notes

- RunPod measurements describe the selected pod, software stack and batch size;
  they are not universal hardware benchmarks.
- GPU utilization and power are polling-based approximations.
- Network download, first model load and steady-state inference are separate.
- No automatic Judge score is treated as validated until its calibration gate
  passes; resource measurements remain valid even when a quality Judge is
  diagnostic only.

## References

- [PyTorch CUDA memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management)
- [Hugging Face generation utilities](https://huggingface.co/docs/transformers/main_classes/text_generation)
- [Torchvision GaussianNoise](https://docs.pytorch.org/vision/main/generated/torchvision.transforms.v2.GaussianNoise.html)

# Week 3 Official-Data RAG Experiment Protocol

## Objective

Measure the effect of retrieved official public InGen context on the same
frozen `meta-llama/Llama-3.1-8B-Instruct` candidate. This is a component-proxy
evaluation, not a claim about deployed InGen product performance.

## Controlled comparison

Each frozen benchmark question produces two rows:

- `base`: the candidate answers without external context;
- `rag`: the candidate receives only context that passed the retrieval and
  metadata gates.

The model and tokenizer revisions, prompt contract, question, decoding,
hardware, batch size, seed, and latency boundary are fixed. Retrieved context
availability is the manipulated variable.

## Paper-informed retrieval design

- RAGAS decomposition: record question, retrieved contexts, answer, and
  component-level metrics separately.
- Metadata before similarity: require official ownership, canonical
  `www.ingendynamics.com` source, public access, current status, and matching
  product metadata.
- Small-to-big retrieval: embed heading-aware child chunks and retain their
  section parent, allowing related children to merge into a coherent parent.
- Quality over quantity: threshold, de-duplicate, and cap context rather than
  indiscriminately adding documents.
- Adaptive no-evidence behavior: return an explicit evidence-insufficient
  answer when no eligible chunk passes the gate.
- Atomic benchmark points: references and prohibited claims are hidden from
  the candidate and joined only during evaluation.

## Frozen v0.3.0 settings

- Seed: 42
- Embedding: `BAAI/bge-m3`, revision
  `5617a9f61b028005a4858fdac845db406aefb181`
- Vector store: local persistent Chroma through `langchain-chroma`
- Chunking: 256 BGE-M3 tokens, 32-token overlap
- Retrieval: fetch-k 12, top-k 4, cosine threshold 0.20
- Generator: `meta-llama/Llama-3.1-8B-Instruct`, revision
  `0e9e39f249a16976918f6564b8830bc894c89659`
- Decoding: greedy, BF16, maximum 384 new tokens
- Independent local judge: `mistralai/Mistral-7B-Instruct-v0.2`, revision
  `63a8b081895390a26e140280378bc85ec8bce07a`
- External model APIs: none

## Metrics

Retrieval:

- document recall@k;
- evidence-fact recall@k;
- reciprocal rank and hit@k;
- metadata-filter leakage;
- no-evidence accuracy;
- retrieval latency.

Generation:

- RAGAS-style faithfulness;
- answer relevance;
- context coverage, represented by context recall plus atomic-point coverage;
- context precision and context relevance;
- forbidden-claim rate;
- generation latency.

All automatic judge results remain `provisional` until calibrated against a
human-reviewed subset.

## Iteration rule

Every run uses a new immutable directory. A child iteration must state one
testable hypothesis and, where possible, change one variable only. It must
archive:

1. parent and child run IDs;
2. exact commands and UTC timestamps;
3. code, knowledge-base, evaluation-set, and configuration hashes;
4. model revisions, dependency versions, hardware, seed, and decoding;
5. full candidate inputs, retrieval traces, outputs, and per-item latency;
6. metric outputs and reasons;
7. observed changes and the keep, revise, or revert decision.

An iteration is not created merely to produce a different number. A new
version is justified only by an observed failure, a controlled ablation, or a
new approved knowledge source.

### Registered early termination

An iteration may stop after generation and before the expensive automatic
Judge stage when a pre-registered qualitative or deterministic regression gate
already fails. The run is not deleted and receives no substituted model score.
Inputs, outputs, hashes, latency, the failed gate and the early-stop decision
remain archived. This prevents unnecessary compute without hiding a failed
version.

Any benchmark item inspected while changing a prompt becomes a development
regression case. It must no longer be described as fresh blind-test evidence;
a new untouched set is required for a later confirmatory comparison.

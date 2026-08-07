# Week 3 RAG Architecture Review — Speaker Script

**Target duration:** approximately 6–7 minutes  
**Deck:** `W03_RAG_Architecture_Review_7min.pptx`

## Slide 1 — RAG evaluation that can be audited

**About 40 seconds**

This week I completed an end-to-end local RAG evaluation pipeline rather than
only a demo chatbot.

The key outcome is that RAG helped the models answer product-specific questions
more directly and with better grounding on the registered public smoke
benchmark.

At the same time, the current evidence is deliberately bounded. It proves that
the evaluation pipeline works and that RAG was helpful on this small corpus, but
it does not prove production readiness.

## Slide 2 — The pipeline separates four sources of failure

**About 65 seconds**

The architecture has four controlled layers.

First, source governance stores the URL, capture date, access scope, authority,
platform, chunk index, and content hash.

Second, BGE-M3 creates the embeddings and Chroma persists the vectors.

Third, retrieval applies a hard metadata gate before semantic ranking. It
retrieves small child chunks but returns the parent section as context. Llama
then receives a detailed grounded prompt with citation and uncertainty rules.

Finally, Base and RAG answers are evaluated separately with retrieval metrics,
RAGAS, failure codes, and complete run traces.

This decomposition matters because perfect retrieval does not guarantee a
complete answer, and an evaluator failure should not be blamed on the candidate
model.

## Slide 3 — The test isolates retrieval, generation, and model effects

**About 55 seconds**

The final public confirmation used eight frozen questions, three models, and two
conditions, giving 48 generations and 48 RAGAS rows.

The candidate-visible inputs were audited to confirm identical questions,
retrieved contexts, context order, seed 42, and deterministic decoding across
models.

One item was inspected during adapter qualification, so the primary aggregate
uses the other seven pairs per model.

I also ran all 18 chunk-size, top-k, and reranker combinations.

The June internship material was not mixed into this result. It stayed in a
separate private collection, where retrieval passed six of six and Llama
completed twelve Base and RAG generations.

## Slide 4 — RAG improved all three models

**About 65 seconds**

This chart compares the same RAGAS AnswerRelevancy metric without and with
retrieved context on the seven uninspected pairs.

All three models improved.

Llama increased from 0.290 to 0.538. This was the largest change and the
strongest evaluator-independent evidence that RAG helped.

Mistral increased from 0.347 to 0.466, but that score is a self-judge diagnostic
because Mistral also served as the local evaluator.

FLAN increased from 0.223 to 0.410, but it often produced short fragments or
question echoes and did not satisfy the citation contract.

Therefore, I do not report an automatic winner. The defensible conclusion is
that retrieval helped, while instruction following still mattered.

## Slide 5 — Automatic relevance is lower than direct content review

**About 70 seconds**

The automatic relevance score looks low because it is not a correctness
percentage.

RAGAS asks the local Judge to infer questions from an answer and then compares
those questions with the original using embeddings. Long answers with multiple
sub-points, qualifications, and citations can be useful but score lower under a
single similarity measure.

Base also has no product knowledge, FLAN often emits fragments, the Judge is
uncalibrated, and seven observations have high variance.

The separate AI qualitative review examined all eight Llama RAG answers
directly. It found 30 of 31 factual claims supported, 0.873 weighted
required-point coverage, no forbidden claims, and 4.375 out of 5 answer
relevance.

These results support usefulness while still exposing omissions.

## Slide 6 — The pipeline works; the benchmark is not production-grade

**About 70 seconds**

The main limitation is scale.

The public knowledge base has only sixteen chunks, and top-k four returns all
Senpai evidence and nearly all Fari evidence. That makes perfect retrieval
easier than it would be in a realistic corpus.

Evaluation is also uncertain because the local Judge is uncalibrated, the
primary aggregate has only seven uninspected pairs, and the AI review is useful
triangulation rather than an independent external certification.

Finally, strict citation formatting remains weak. Only one of seven Llama rows
and one of seven Mistral rows used the exact bracketed chunk-ID format, while
FLAN used none.

This is primarily a generation-format failure, not a retrieval failure.

## Slide 7 — Next: scale the benchmark, not the pipeline

**About 55 seconds**

The next step is not to replace the architecture. It is to increase the
difficulty and representativeness of the evidence.

First, I would expand to more than one hundred chunks with distractors and stale
or conflicting sources.

Second, I would freeze a new set of twenty-four to thirty-two questions covering
noisy inputs, no-evidence cases, multi-turn dialogue, multilingual prompts, and
prompt injection.

Finally, I would rerun the same three-model Base and RAG comparison, retrieval
metrics, RAGAS, and qualitative review.

So the Week 3 takeaway is: the pipeline is operational and RAG is helpful on the
current benchmark; the next milestone is stronger evidence for generalization.

## If only five minutes are available

Shorten Slides 2 and 3 to one sentence per layer/test, and omit the Mistral and
FLAN detail on Slide 4. Keep Slides 5 and 6 because they explain the apparent
score discrepancy and the validity boundary.


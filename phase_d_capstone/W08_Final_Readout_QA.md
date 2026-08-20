# Week 8 Final Readout - Q&A Defence Notes

## Why 35 scenarios?

Thirty-five is a coverage design: seven scenarios for each of five platform contexts. Equal allocation prevents the conversational products from dominating one aggregate and forces the benchmark to include care, education, security, navigation, and task-execution boundaries. The 10/15/10 severity distribution covers recoverable, material, and plausibly safety-critical consequences. It is intentionally not presented as statistically representative; seven cases per platform can expose patterns but cannot estimate a deployment failure rate. The next step is a larger, newly sealed and domain-reviewed set.

## Why RAGAS-style metrics?

RAG can fail at different components. Retrieval recall tests whether the required evidence entered the context. Faithfulness tests whether answer claims are supported by the retrieved passages. Answer relevance tests whether the response addresses the question. Required-point coverage checks whether frozen benchmark requirements were expressed. Reporting all of them prevents one aggregate score from hiding whether the problem is retrieval, unsupported generation, irrelevance, or omission. Their boundary is equally important: they do not prove that the source is current, authorised, medically correct, or safe to act on, and the local evaluator was not independently calibrated.

## How does the Ninenovo masked-prediction methodology apply?

The transferable idea is controlled information removal. The Week 4 study froze the scenario and model, removed registered evidence groups at 0, 20, 40, and 60 percent, and measured degradation curves rather than one clean score. This resembles controlled masking in representation-learning work. The adaptation is incomplete because text evidence groups are not time-aligned sensors. A deployment follow-up must mask each physical modality separately and in combinations, randomise dropout blocks, and measure collision or unsafe-action rate, recovery, intervention, and time to safe state.

## Why use severity weighting?

Physical-AI failures have asymmetric consequences. Ten harmless successes should not cancel one medication, child-safety, navigation, or security failure. Severity 1/3/5 is assigned from the plausible consequence of a wrong response, not from linguistic difficulty. The weights are ordinal prioritisation aids, not an expected-harm model: the method does not claim that severity 5 is exactly five times worse than severity 1. Severity-5 rows always remain under mandatory review, and both weighted and unweighted denominators are retained.

## Why not recommend immediate deployment of the top model?

The ordering is diagnostic because the frozen-label Judge calibration alpha was 0.7551, below the 0.80 gate, and no deployed product was measured. Mistral is recommended only as the first candidate for independent model-blind calibration and closed-loop validation. The dashboard's 85.8/100 value is a communication proxy, not a safety certificate.

## What changed after the RAG knowledge-base correction?

The original atomic sections were all shorter than 256 tokens, so changing chunk size did not materially change indexed units. The correction used 21 complete long public documents and operational 256/512/1024-token splits. That made the Week 5 3 x 3 x 2 factorial meaningful and produced three observed Pareto cells. Older short-source results are retained only as superseded provenance.

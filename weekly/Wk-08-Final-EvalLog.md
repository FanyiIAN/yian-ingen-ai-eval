# Week 8 Final Evaluation Log

## Evaluated

I integrated the Week 2-7 evidence into the final capstone package: a 15-20 page report, a 12-slide finding-led deck, a one-page retrospective, a final claim-evidence matrix, Q&A defence notes, and the final evaluation rubric. Before writing, I froze the latest evidence versions and excluded superseded atomic-section RAG conclusions. I also re-ran the Week 7 dashboard contract and interaction suite, refreshed all three v1.2.0 persona screenshots, and checked the corrected light-theme labels in a live browser. The capstone traces every principal result to a source artifact, exact model revision, evaluation-set version, evidence status, and seed 42.

## Found

Three findings drive the final recommendation. First, Mistral has the highest text diagnostic proxy, 85.8/100, but the Judge calibration alpha is 0.7551 against a 0.80 gate, so it is only the first candidate for independent review. Second, corrected long-source RAG adds 0.655667 relevance and 0.522917 required-point coverage, but also 8030.48 ms mean latency. Third, robustness cannot be read as consistency alone: FLAN's 0.9143 consistency contains 25 stable failures, whereas Mistral's lower 0.8571 consistency contains zero stable failures. The Week 5 factorial further shows three Pareto cells rather than one universal RAG optimum. The PIC analysis identifies CRL-MRS as the largest evidence gap because there are zero direct multi-agent execution tests.

## Mechanism and decision

The recurring mechanism is measurement-boundary mismatch. A reproducible pipeline can reproduce the wrong construct: Judge agreement does not guarantee expert validity, text masking does not reproduce sensor dropout, static images do not test closed-loop recovery, and a public document can be faithfully quoted while still being unsuitable for action. The final recommendation therefore advances Mistral, two Pareto RAG configurations, and LLaVA as validation candidates, while explicitly withholding deployment approval.

## Questioned / next action

The remaining external actions are the joint supervisor rubric scores and signatures, the formal 30-minute readout plus 15-minute Q&A, and approval to publish/tag the repository. The first technical follow-up should be model-blind domain-expert calibration on a new sealed set; the first product-level follow-up should be closed-loop sensor dropout and safe-recovery testing.

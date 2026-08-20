# Week 8 Retrospective

## Most surprising finding

The most surprising result was that the highest robustness number could describe the least useful model. FLAN-T5 Base had the highest semantic consistency, `0.9143`, but it failed consistently on 25 of 35 scenarios and passed consistently on only seven. Mistral and Llama both had lower consistency, `0.8571`, but Mistral had zero stable failures and 30 stable passes. This changed how I interpret robustness: repeatability is not correctness, and a robustness result is incomplete unless it is paired with stable-pass and stable-fail denominators. It also reinforced the need to separate observations from mechanisms. The observation is reproducible; the proposed explanation - that a weak model repeats the same inadequate policy across paraphrases - still needs controlled testing.

## Weakest section of the benchmark

The weakest section is scoring validity, especially Failure Mode classification. Three Judge formulations produced nominal Krippendorff alpha `0.5673` for Failure Mode, and the frozen-label Task calibration reached `0.7551`, below the preregistered `0.80` gate. The benchmark pipeline, hashes, row counts, and model outputs are reproducible, but reproducibility cannot repair an unvalidated measurement instrument. The public/synthetic scenarios are also proxies rather than observations from deployed InGen systems. Consequently, the current work can prioritise models and experiments but cannot certify product readiness or estimate field failure rates.

## What I would add in a 12-week version

With four more weeks and additional compute, I would run four experiments. First, I would recruit at least two independent, model-blind domain reviewers and calibrate 30-50 severity-stratified outputs before opening a new sealed test set. Second, I would replace text masking and static-image corruption with closed-loop simulator or approved sensor-stream experiments measuring unsafe actions, recovery success, and time to safe state. Third, I would repeat the RAG factorial study across time-sliced corpora, access policies, stale or conflicting documents, adversarial insertions, multiple seeds, and a second hardware target to test whether the three Pareto cells persist. Fourth, I would create the missing CRL-MRS benchmark: cooperative execution under controlled message loss, bandwidth limits, agent dropout, and continual updates, with joint success and backward transfer as primary outcomes.

The main lesson is that evaluation quality depends less on producing one more score than on defining what the score is allowed to mean.

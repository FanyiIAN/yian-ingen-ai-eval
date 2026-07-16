# W01 Public Source Links and Provenance

**Checked:** 2026-07-16  
This file distinguishes links actually embedded in the programme PDFs from canonical sources selected to satisfy the Week 1 reading categories.

## What the two programme PDFs actually link

The eight-week plan names source categories and examples but contains **no embedded URLs**. The concepts primer contains 48 clickable learning links, mostly Wikipedia/tool documentation plus a few papers. Particularly relevant links include:

- LLM-as-judge / MT-Bench: https://arxiv.org/abs/2306.05685
- Original RAG paper: https://arxiv.org/abs/2005.11401
- RAGAS overview: https://docs.ragas.io/en/latest/
- RAGAS faithfulness: https://docs.ragas.io/en/latest/concepts/metrics/faithfulness.html
- RAGAS answer relevance: https://docs.ragas.io/en/latest/concepts/metrics/answer_relevance.html
- RAGAS context recall: https://docs.ragas.io/en/latest/concepts/metrics/context_recall.html
- TextAttack documentation: https://textattack.readthedocs.io/en/latest/
- Hugging Face Evaluate: https://huggingface.co/docs/evaluate/index
- Krippendorff library: https://pypi.org/project/krippendorff/
- Weights & Biases: https://wandb.ai/

### Important GRPO link warning

The primer attaches https://arxiv.org/abs/2501.12599 to its “GRPO (Goal-conditioned RL)” entry. That URL is **Kimi k1.5: Scaling Reinforcement Learning with LLMs**, not an authoritative definition of a goal-conditioned robotics policy. It should not be used as the sole source for the programme-specific GRPO meaning.

## Six selected Week 1 references

| Requirement | Canonical source selected | Why this source |
|---|---|---|
| InGen products / Origami / PIC context | https://www.ingendynamics.com/ | Official product ecosystem and current Origami positioning. |
| Origami architecture whitepaper | https://ingendynamics.com/blog/wp-content/themes/getaido/PDF/07TheFutureofAIPlatforms.pdf | Official whitepaper linked from the InGen site. |
| General AI evaluation | https://arxiv.org/abs/2211.09110 | Original HELM paper: scenario taxonomy, standardized and multi-metric evaluation. |
| RAG evaluation | https://aclanthology.org/2024.eacl-demo.16/ | Peer-reviewed RAGAS system paper in ACL Anthology. |
| VLM evaluation | https://arxiv.org/abs/2311.16502 | Original MMMU benchmark paper. |
| LLM robustness / adversarial evaluation | https://arxiv.org/abs/2306.04528 | PromptBench/PromptRobust paper on character, word, sentence and semantic perturbations. |
| Deployed/safety-critical evaluation framework | https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-ai-rmf-10 | Official NIST AI RMF 1.0 publication. |

## Public counterparts for the six PIC working classes

These are evaluation analogues, not claims about InGen's undisclosed implementation.

| PIC working class | Start here | What to extract |
|---|---|---|
| GRPO - goal-conditioned RL | https://arxiv.org/abs/2208.08133 and https://arxiv.org/abs/2112.03227 | State/goal/action structure, success, constraint violations, held-out goals and long-horizon completion. |
| STUM - temporal/state | https://aclanthology.org/2024.acl-long.66/ | Order, duration, temporal consistency and performance as sequence length grows. |
| SEOM - spatial track | https://arxiv.org/abs/2502.09560 | Embodied spatial grounding/navigation tasks and closed-loop evaluation. |
| SEOM - semantic/retrieval track | https://arxiv.org/abs/2104.08663 and the RAGAS paper | Recall@k, ranking/generalization, faithfulness and relevance. |
| AMDC - multimodal decision | https://arxiv.org/abs/2311.16502 | Controlled multimodal comparison, perception/reasoning errors and modality ablation. |
| HTD-IRL - task decomposition | https://arxiv.org/abs/1912.01734 and https://arxiv.org/abs/2112.03227 | Subgoal completion, ordering/dependencies, recovery and long-horizon task success. |
| CRL-MRS - cooperative/multi-agent | https://arxiv.org/abs/2212.07489 | Partial observability, closed-loop coordination, generalization and joint success. |

## Taxonomy caveat

The reviewed programme materials and current public product pages do not use STUM/SEOM consistently. Store a `taxonomy_version` with every future scenario and keep temporal, uncertainty, spatial, semantic-retrieval and governance measurements separate until the authoritative PIC 2.0 glossary is confirmed.


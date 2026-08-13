# Week 5 PIC 2.0 Model-Class Analysis

**Taxonomy:** programme-plan-v1 working tracks; proposed capability proxies
**Phase B evidence:** public/synthetic component benchmarks only
**Seed:** `42` for every cited run
**Claim boundary:** no result below measures an InGen product or proprietary PIC runtime

The working mapping follows the programme tracks: GRPO = goal-conditioned
policy; STUM = temporal/state reasoning; SEOM = spatial understanding; AMDC =
multimodal decision; HTD-IRL = hierarchical task decomposition; CRL-MRS =
cooperative continual multi-agent reasoning. STUM and SEOM have conflicting
meanings in other programme/public materials, so temporal, uncertainty,
spatial, semantic-retrieval, and governance evidence remain separately
versioned. GRPO here does not mean the public LLM-training use of “Group
Relative Policy Optimisation.”

## GRPO — goal-conditioned policy proxy

**Primary Phase B finding.** On the seven original Aido Rover text scenarios in
`w04_frozen_robustness_inputs_v0.1.0`, the uncalibrated task-pass rates were
`7/7` for `meta-llama/Llama-3.1-8B-Instruct` revision
`0e9e39f249a16976918f6564b8830bc894c89659`, `6/7` for
`mistralai/Mistral-7B-Instruct-v0.2` revision
`63a8b081895390a26e140280378bc85ec8bce07a`, and `3/7` for
`google/flan-t5-base` revision
`7bcac572ce56db69c1ea7c8af255c5d7c9672fc2`. These are text-plan proxy scores,
not rollouts or goal completion. The Llama result is the strongest Phase B
signal, while the cross-model spread shows why a generic overall accuracy is
not a GRPO readiness measure.

**Deployment failure.** An Aido Rover can name the requested waypoint yet take
a shorter route through a keep-out zone, reach the goal, and still be unsafe.
Text scoring would miss state drift, path feasibility, actuator limits, and
whether the constraint was violated during execution.

**Readiness metric.** Use **constrained goal-success rate** on held-out goals,
reported jointly with constraint-violation rate, path efficiency, and recovery
success. A run is successful only if it reaches the goal without any registered
violation.

**Open question.** Does success persist under changed goals, maps, and
constraints without an increase in unsafe shortcuts in closed-loop Rover
rollouts?

## STUM — temporal/state track

**Primary Phase B finding.** Phase B did not run a temporal-horizon benchmark.
The closest proxies are single-turn summaries of earlier state: `ROVER-007`
required north-marker → east-station → base ordering, and `FARI-007` required a
previously stated music preference to remain defeasible. On `ROVER-007`, the
diagnostic Week 4 Judge marked Llama pass (`4/5`), FLAN fail (`1/5`), and
Mistral fail (`2/5`), although row review shows the Mistral answer contains the
correct waypoint order in an expanded five-step plan. This disagreement is
evidence that the current scorer is not a reliable temporal-readiness gate.
All revisions, dataset, and seed are the same as in the GRPO section.

**Deployment failure.** A Fari care conversation could retain an old medication
state after a clinician update, or a Rover could execute step three before a
step-two calibration result exists. A one-turn answer may look plausible while
the state history is internally contradictory.

**Readiness metric.** Use **temporal contradiction rate by sequence horizon**,
with order accuracy and stale-state action rate. If the alternate uncertainty
track is later selected, use a separately versioned risk–coverage curve and
Brier score; do not merge it with temporal results.

**Open question.** Which STUM glossary is authoritative, and how quickly do
contradictions increase from short sequences to product-representative event
histories with corrections and delayed observations?

## SEOM — spatial-understanding track

**Primary Phase B finding.** On 60 Aido Rover VLM rows from
`w04_multimodal_input_manifest_v0.1.0`, both
`HuggingFaceM4/idefics2-8b-chatty` revision
`8e65868b394317b973bd61db3b08e6478ebeedbf` and
`llava-hf/llava-1.5-7b-hf` revision
`b234b804b114d9e37bb655e11cbbb5f5e971b7a9` received a diagnostic mean `5.0/5`
under clean, Gaussian-noise, and brightness conditions. This is a likely
ceiling on static public images, not evidence of viewpoint-stable localization
or navigation. Week 3 retrieval evidence belongs only to the alternate
semantic-retrieval track and is not pooled here.

**Deployment failure.** After a viewpoint change, Rover could swap “left of”
and “behind,” mislocalize an obstacle, and recommend a collision path even
though its single-frame scene description is fluent.

**Readiness metric.** Use **collision-free navigation success rate** across
viewpoint changes, paired with spatial-relation accuracy, localization error,
path efficiency, and intervention rate. The current VLM score is only a
perception-and-advice proxy.

**Open question.** Which SEOM track—spatial, semantic/retrieval, or governance—is
authoritative, and can the spatial track be tested in a closed-loop simulator
with changing pose, occlusion, and persistent object state?

## AMDC — adaptive multimodal decision proxy

**Primary Phase B finding.** Across the 120 rows of
`w04_multimodal_input_manifest_v0.1.0`, Idefics2 mean total scores were
`4.90/5` clean, `4.80/5` with Gaussian noise, and `4.75/5` under brightness;
LLaVA scored `4.80/5`, `4.85/5`, and `4.70/5`. LLaVA had lower median
question-to-response latency (`4.39 s` versus `6.31 s`) and lower device peak
memory (`14.15 GiB` versus `18.23 GiB`). The score differences are small,
uncalibrated, and based on image+prompt rather than synchronized sensors, but
the quality/resource trade-off is the most direct Phase B AMDC proxy.

**Deployment failure.** A noisy Sentinel camera channel could dominate healthy
sensor evidence, preserve a confident unsafe alert, and suppress escalation
instead of degrading gracefully. Static image corruption does not test
cross-modal conflict or missing-channel behavior.

**Readiness metric.** Use **modality-degradation AUC** over registered corruption
intensities, reported with decision-flip rate, calibrated abstention/escalation,
per-modality contribution, latency, and memory. The current single-intensity
conditions cannot estimate that AUC.

**Open question.** Does the apparent robustness persist with a second modality,
sensor disagreement, missing channels, and synchronized product-representative
streams rather than public-image proxies?

## HTD-IRL — hierarchical task-decomposition proxy

**Primary Phase B finding.** On seven original Aido Humanoid scenarios from
`w04_frozen_robustness_inputs_v0.1.0`, Llama, Mistral, and FLAN task-pass rates
were `7/7`, `7/7`, and `2/7`. The class-specific view changes the interpretation:
Mistral received the categorical failure code `partial` on five of seven rows
despite all seven meeting the task-pass threshold, while Llama received
`none` on all seven. For example, `HUMANOID-004` tested replanning after a
dropped screwdriver; a superficially acceptable continuation can still violate
the new dependency. These are uncalibrated text plans, not executed task graphs.

**Deployment failure.** An Aido Humanoid drops a tool, but its plan continues as
if the tool were still held, skips isolation and recovery, and enters the next
assembly subtask in an unsafe physical state.

**Readiness metric.** Use **dependency-violation rate**, with full-task success,
subgoal completion, ordering correctness, feasibility, and recovery after an
injected subtask failure. Overall task pass alone hides partial decomposition
defects.

**Open question.** Can the system revise a task graph after tool loss or failed
grasp while preserving safety prerequisites, completed-state memory, and a
feasible route to the remaining goal?

## CRL-MRS — cooperative continual multi-agent proxy

**Primary Phase B finding.** Phase B contains no direct multi-agent execution,
communication-loss, continual-learning, or forgetting experiment. Its closest
row, Aido Humanoid scenario `HUMANOID-005`, asks two agents to synchronize a
table lift. The diagnostic scores were Llama `4/5` (`none`), Mistral `4/5`
(`partial`), and FLAN `1/5` (`partial`) under the same Week 4 dataset and
revisions. This one text response cannot estimate joint success; the correct
finding is a material evidence gap, not CRL-MRS readiness.

**Deployment failure.** Two Humanoids duplicate one end of a lift, leave the
other uncovered, and continue retrying after communication loss. The combined
plan can read coherently while responsibility allocation and recovery are
incompatible at execution time.

**Readiness metric.** Use **joint task success under controlled communication
loss**, with per-agent contribution, allocation coverage, coordination
redundancy, recovery latency, and backward transfer after continual updates.

**Open question.** How much bandwidth, message loss, or agent dropout can the
team tolerate before joint success collapses, and does adaptation preserve
previously learned cooperation instead of causing catastrophic forgetting?

## Decision implication

Phase B supports a strong **evaluation design handoff**, not a PIC deployment
claim. AMDC has the broadest direct proxy evidence; GRPO and HTD-IRL have useful
text-plan diagnostics; STUM and SEOM need authoritative terminology plus
sequence/closed-loop tests; CRL-MRS has the largest evidence gap. The next PIC
readiness gate should therefore prioritize class-specific failure denominators
and executed state transitions rather than one aggregate accuracy score.

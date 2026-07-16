# W01 Origami and PIC 2.0 Study Guide

## Origami AI in one paragraph

Origami AI is publicly presented as InGen's shared Physical Intelligence layer: a hardware-agnostic, edge-native, multimodal platform intended to connect robot hardware, sensors, deep-learning modules, speech/NLP, messaging, positioning, diagnostics and decision logic across multiple product bodies. The evaluation implication is that a shared capability must be tested both as a model capability and in the consequence structure of each product: the same error has different implications for Fari, Sentinel, Rover and a Humanoid.

## Publicly described layers

- **Hardware:** electronics/PCB, mechanical/electrical systems, motor and power management.
- **Perception/deep learning:** pose/fall and trajectory analysis, audio classification, firearm/face/vehicle/object/workplace-safety modules.
- **Software/platform:** UI modules, messaging, indoor positioning and self-diagnostics.
- **Intelligence:** speech recognition, NLP, automation, machine learning, sensing and fault tolerance.
- **Product bodies:** companion/education products, security, mobile Rover/Carry & Go and longer-term Humanoid platforms.

## What PIC 2.0 means for Week 1

Treat PIC 2.0 as the programme's six-class foundation-model/evaluation taxonomy inside the broader Origami platform. Week 1 does **not** require implementing or training all six classes. For each class, be able to explain:

1. intended input, state and output;
2. structural public counterpart;
3. relevant InGen product scenario;
4. likely failure modes and physical consequence;
5. suitable held-out test, metric and perturbation;
6. one unresolved implementation or taxonomy question.

## Six-class minimum mastery

| Class | Minimum Week 1 explanation | Evaluation focus |
|---|---|---|
| GRPO | Goal/state-conditioned policy selects subgoals or actions; programme anchor is Rover goal prioritization. | Held-out goal success, efficiency and constraint violations. |
| STUM | Programme taxonomy describes reasoning over a sequence of states and maintaining temporal consistency. | Sequence permutation/extension, contradiction rate and horizon degradation. |
| SEOM | The plan says spatial understanding; the primer says semantic/embedding retrieval; public Sentinel material uses the label for governance. | Do not merge the tracks. Test spatial grounding, retrieval quality or rule conformance separately. |
| AMDC | Fuses multiple modalities/sensors before a decision. | Modality ablation, masked inputs, noise, calibration and latency. |
| HTD-IRL | Decomposes a complex task into ordered, dependent subtasks and may learn priorities/costs from demonstrations. | Full-task and subgoal success, ordering/dependency violations and recovery. |
| CRL-MRS | Coordinates multiple partially informed agents and retains capability while learning across tasks/fleet updates. | Joint success, compatibility, communication-loss robustness and forgetting/generalization. |

## Evidence expected in the Week 1 brief

For every mapping, include a **structural reason**, the **evaluation challenge**, and the **most relevant public benchmark**. Avoid claiming knowledge of proprietary implementations. The deeper per-class analysis is a Week 5 deliverable; Week 1 establishes the vocabulary and evaluation map needed to design the Week 2 benchmark.


---
title: Intent-Conditioned Trajectory Planning
type: concept
sources: [raw/papers/Driving Intents Amplify Planning-Oriented Reinforcement Learning.md, raw/papers/Fine-tuning is Not Enough_ A Parallel Framework for Collaborative Imitation and Reinforcement Learning in End-to-end Autonomous Driving.md]
related: [sources/dial.md, sources/pair-drive.md, concepts/rl-for-ad.md, concepts/gspo-vs-grpo.md, concepts/best-of-n.md, concepts/diffusion-planner.md, concepts/parallel-il-rl.md, concepts/nuscenes-waymo-evals.md]
created: 2026-06-23
updated: 2026-06-23
confidence: high
---

# Intent-Conditioned Trajectory Planning

Intent-conditioned planning introduces a discrete semantic variable—such as cruise, turn, lane change, accelerate, or decelerate—between scene understanding and continuous trajectory generation. The intent is not necessarily the final driving output. It can act as a control variable that forces a generative planner to expose multiple maneuver basins for the same scene.

## Why Intent Helps

One logged scene usually supplies one demonstrated trajectory even when several actions were feasible. Continuous diffusion or flow policies trained on that single target may respond to different noise seeds with geometrically similar trajectories. This is behavioral mode collapse: the network is stochastic in coordinates but unimodal in maneuver semantics.

Intent conditioning supplies an explicit axis along which proposals can differ. It changes the learning question from “reproduce this path with noise” to “produce a plausible path under this maneuver hypothesis.”

## Two Uses in the Wiki

| Method | Intent mechanism | Training role | Inference role |
| --- | --- | --- | --- |
| [[sources/pair-drive.md]] | Learned intention tokens in a recurrent residual tree | Structure GRPO candidates around a human reference | Expand alternatives around an IL trajectory; RWM selects |
| [[sources/dial.md]] | Eight rule-derived labels with classifier-free guidance | Expand SFT support and balance every GRPO group across intents | Intent classifier selects one mode; conditioned flow generates |

PaIR-Drive treats intent as tree-branch structure in a separate RL refiner. DIAL treats intent as a condition inside one continuous generative policy and explicitly preserves all modes during preference fine-tuning.

## Intent-CFG

Classifier-free guidance trains one generator with both conditional and unconditional intent inputs. At sampling time, the conditional and unconditional flow fields are combined to strengthen the selected maneuver mode.

This is different from using CFG only to improve fidelity. In DIAL, CFG's purpose is proposal-support expansion: different intent labels should reach different rater-meaningful behaviors.

## Intent-Balanced GRPO

For $C$ intents and $S$ noise samples per intent, the group size is $K=CS$. DIAL uses $C=8$, $S=2$, $K=16$. Advantages are normalized across the pooled group.

The comparison to $C=1$, $S=16$ isolates the key effect: total sample count is fixed, but semantic coverage changes. DIAL reaches held-out RFS 8.211 versus 7.992 for the strongest single-intent alternative. Therefore, candidate count alone is not the operative variable; the group must span reward-relevant modes.

## Diversity Is Multidimensional

Trajectory separation and preference diversity are not interchangeable:

- **Spatial diversity:** pairwise ADE between proposals.
- **Reward diversity:** standard deviation of preference scores.
- **Diversity dividend:** Best-of-N minus Best-of-1 score.
- **Deployment quality:** score after selecting/predicting one intent.

DIAL's top-rater single-intent variant has greater spatial separation than multi-intent DIAL but worse reward spread and lower final RFS. A useful proposal set must be both different and meaningfully rankable.

## Intent Ontology Design

A practical ontology should be:

- small enough that every class is represented frequently;
- broad enough to cover genuinely different maneuver modes;
- derivable consistently from demonstrations or labels;
- predictable from scene context at inference;
- compositional enough to avoid forcing ambiguous behavior into one class.

DIAL's eight classes satisfy simplicity but not full coverage. For example, “yield while changing lanes,” “creep,” and obstacle-specific avoidance are not explicit classes. Hard rule thresholds can also make semantically similar trajectories receive different labels.

## Evaluation Requirements

Report all of the following:

- intent-label derivation and class frequency;
- intent-classifier accuracy/confusion;
- per-intent feasibility and reward;
- fixed-budget comparison against extra noise samples;
- spatial and reward diversity;
- Best-of-N support ceiling separately from deployable single-intent performance;
- latency of intent prediction and conditional generation;
- closed-loop safety and interaction metrics, not preference score alone;
- sensitivity to ontology size and label noise.

## Core Insight

An optimizer can only amplify distinctions present in its sample group. Intent conditioning is valuable when it creates semantically distinct, feasible alternatives that the reward can rank; arbitrary labels or spatial scatter alone do not solve the problem.

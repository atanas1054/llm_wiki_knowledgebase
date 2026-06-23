---
title: nuPlan Closed-Loop Planning Benchmark
type: concept
sources: [raw/papers/PlannerRFT_ Reinforcing Diffusion Planners through Closed-Loop and Sample-Efficient Fine-Tuning.md, raw/papers/Plan-R1_ Safe and Feasible Trajectory Planning as Language Modeling.md]
related: [sources/plannerrft.md, sources/plan-r1.md, concepts/rl-for-ad.md, concepts/diffusion-planner.md]
created: 2026-06-23
updated: 2026-06-23
confidence: high
---

# nuPlan Closed-Loop Planning Benchmark

nuPlan evaluates motion planners by executing their trajectories in a simulator rather than scoring only waypoint error against a recorded future. This makes it useful for measuring compounding error, controller feasibility, collision avoidance, route progress, comfort, and interaction with traffic.

## Evaluation Modes

- **NR (non-reactive):** surrounding agents replay logged trajectories regardless of ego behavior.
- **R (reactive):** surrounding vehicles use an Intelligent Driver Model (IDM) that responds to ego behavior.
- **Val14:** general scenarios from 14 scenario types.
- **Test14-hard:** difficult scenarios selected to stress robustness.
- **Test14-random:** 261 randomly selected scenarios from the nuPlan Planning Challenge in the PlannerRFT/Plan-R1 protocol.

Reactive and non-reactive scores should not be merged. Reactive traffic better exposes negotiation and distribution-shift behavior, while non-reactive traffic can produce unrealistic conflicts because logged agents cannot respond to a changed ego trajectory.

## Scoring Pattern

The papers in this wiki use a score in `[0,100]` built from safety gates and soft driving-quality terms. PlannerRFT's training scorer multiplies collision, drivable-area, and wrong-direction terms by a weighted average of TTC, ego progress, comfort, and speed compliance. Exact metric implementations and weights must be checked before comparing papers.

## Protocol Caveats

- **Post-processing matters.** PDM-based rule fallbacks or proposal scoring can raise results substantially; learning-only and hybrid rows are not equivalent.
- **Sampler settings matter.** PlannerRFT shows that ten-step DPM and five-step DDIM Diffusion Planner baselines are close but not identical.
- **Simulator implementation matters.** PlannerRFT trains in nuMax with log-replay traffic but evaluates in native nuPlan, including IDM-reactive mode.
- **Split naming is insufficient.** Verify scenario count, reactive-agent policy, controller, scorer version, and whether results are averaged over the same seeds.
- **Log replay is not an upper bound in reactive mode.** Recorded expert motion can score below rule-based planners when other agents are simulated reactively.

## Representative Learning-Only Results

| Source | Planner | Val14 NR | Val14 R | Test14-hard NR | Test14-hard R | Test14-random NR | Test14-random R |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PlannerRFT | Diffusion Planner DDIM | 89.81 | 82.94 | 76.01 | 68.18 | 89.14 | 82.63 |
| PlannerRFT | PlannerRFT | 89.96 | 84.46 | 77.16 | 72.21 | 90.76 | 85.80 |
| Plan-R1 | Plan-R1 | 88.98 | 87.69 | 77.45 | 77.20 | 91.23 | 90.04 |

These rows are useful for mechanism-level comparison, not a strict leaderboard. Plan-R1 uses autoregressive motion tokens and a learned reactive surrounding-agent model during training, whereas PlannerRFT uses continuous diffusion and nuMax rollouts. Protocol and publication-time baseline sets differ.

## nuMax

[[sources/plannerrft.md]] introduces nuMax, a JAX/XLA reimplementation designed for RL throughput. It caches fixed windows from nuPlan into TFRecords, implements a batched LQR plus kinematic-bicycle controller, and reports up to 10x faster rollout than native nuPlan.

The speed comes with two explicit limitations: cached tensors are representation-specific because XLA requires static shapes, and surrounding traffic during training is log replay rather than IDM. nuMax should therefore be treated as a training accelerator calibrated to nuPlan, not as evidence that the training environment exactly matches the official reactive evaluator.

## Interpretation Rule

For nuPlan claims, always report at least `(split, NR/R, learning-only vs hybrid, sampler/controller/scorer)`. A score without those qualifiers is not reliably comparable.

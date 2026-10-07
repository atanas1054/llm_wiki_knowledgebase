---
title: PDM-Lite
type: concept
sources: ["raw/papers/Raw2Drive_ Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving(in CARLA v2).md", "raw/papers/Think2Drive_ Efficient Reinforcement Learning by Thinking with Latent World Model for Autonomous Driving (in CARLA-v2).md", "raw/papers/SimScale_ Learning to Drive via Real-World Simulation at Scale.md", raw/papers/UniDriveVLA_ Unifying Understanding, Perception, and Action Planning for Autonomous Driving.md, raw/papers/Unifying Language-Action Understanding and Generation for Autonomous Driving.md, raw/papers/AutoMoT_ A Unified Vision-Language-Action Model with Asynchronous Mixture-of-Transformers for End-to-End Autonomous Driving.md]
related: [sources/raw2drive.md, sources/think2drive.md, sources/simscale.md, concepts/bench2drive.md, concepts/selection-based-planning.md, sources/unidrivevla.md, sources/linkvla.md, sources/automot.md]
created: 2026-05-01
updated: 2026-10-06
confidence: medium
---

## What It Is

PDM-Lite is a privileged planner/fallback used in some Bench2Drive comparisons. It can use information that a normal end-to-end policy would not necessarily have at deployment, such as privileged route or map/waypoint structure.

## Why It Matters

Bench2Drive tables often mix methods with and without PDM-Lite-style assistance. This can inflate driving score and success rate relative to pure learned policies. UniDriveVLA explicitly reports a "best without PDM-Lite" framing; LinkVLA and other Bench2Drive leaders should be compared only after checking whether privileged assistance is used.

## Lint Rule

Any Bench2Drive SOTA claim should state whether PDM-Lite or a comparable privileged fallback is used. If the paper does not disclose this clearly, mark the comparison as uncertain rather than treating it as a clean end-to-end leaderboard result.

## Related Privileged Planner: PDM-Closed on NAVSIM {#pdm-closed}

*(2026-10-06)* PDM-Closed, the nuPlan rule-based planner from the same family, appears in two privileged roles in [[sources/simscale.md]].
- **As a teacher.** It plans on ground-truth state for 237K simulated off-log scenes; only trajectories with every EPDMS sub-metric at 1 (ego progress > 0.5) are kept as labels. Students inherit its low extended comfort.
- **As a reference score.** On navhard it scores **51.3** combined, above every SimScale student (48.0 single, 50.9 ensemble) and below only three sensor-based entries in the wiki. Its Stage-2 lane keeping falls 99.3 → 73.7 without any rendered input, which bounds how much of navhard's Stage-2 collapse can be a rendering artifact ([[concepts/navhard-ood-evaluation.md#simscale]]).

**Lint rule, extended**: a privileged planner's score is a reference, not a leaderboard entry, on NAVSIM as on Bench2Drive. When a student is trained on its labels, report the teacher's own score beside the student's.

## The Other CARLA v2 Expert: Think2Drive {#think2drive}

*(2026-10-06)* [[sources/think2drive.md]] is the learned counterpart: a privileged BEV planner trained by model-based RL, released before PDM-Lite. Its paper notes PDM-Lite also solves all CARLA v2 scenarios but "adopt[s] different hyper-parameters for different scenarios, which requires heavy manual labor".

| Expert | Kind | Bench2Drive DS / SR ([[sources/dynvla.md]]'s table) | Role |
|---|---|---|---|
| Think2Drive | Model-based RL, privileged BEV, 30 discrete actions | 91.85 / 85.41 (also in [[sources/raw2drive.md]]'s table, multi-ability mean 86.26) | Generated Bench2Drive's official base set; Stage I of [[sources/raw2drive.md]] |
| PDM-Lite | Rule-based, privileged | 97.02 / 92.27 | Data for the "PDM-Lite protocol" (SimLingo, DiffusionDrive, AutoMoT); also an inference-time fallback in some methods |

The two roles on this page (inference-time fallback and training-data source) apply to both experts; only PDM-Lite is used as a fallback in the ingested papers.

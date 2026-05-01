---
title: Action Tokenization and Codebooks
type: concept
sources: [raw/papers/AutoVLA_ A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning.md, raw/papers/DriveVLA-W0_ World Models Amplify Data Scaling Law in Autonomous Driving.md, raw/papers/Unifying Language-Action Understanding and Generation for Autonomous Driving.md, raw/papers/NoRD_ A Data-Efficient Vision-Language-Action Model that Drives without Reasoning.md, raw/papers/DiffusionDriveV2_ Reinforcement Learning-Constrained Truncated Diffusion Modeling in End-to-End Autonomous Driving.md, raw/papers/DriveSuprim_ Towards Precise Trajectory Selection for End-to-End Planning.md, raw/papers/SpanVLA_ Efficient Action Bridging and Learning from Negative-Recovery Samples for Vision-Language-Action Model.md, raw/papers/OneDrive_ Unified Multi-Paradigm Driving with Vision-Language-Action Models.md]
related: [concepts/diffusion-planner.md, concepts/selection-based-planning.md, concepts/best-of-n.md, sources/autovla.md, sources/drivevla-w0.md, sources/linkvla.md, sources/nord.md, sources/diffusiondrive-v2.md, sources/drivesuprim.md, sources/spanvla.md, sources/onedrive.md]
created: 2026-05-01
updated: 2026-05-01
confidence: high
---

## What It Is

Action tokenization is the design choice that converts continuous ego trajectories into model outputs: discrete codebook IDs, fixed-vocabulary trajectory selections, masked tokens, or continuous action vectors decoded by an expert.

## Main Patterns

| Pattern | Examples | Practical implication |
| --- | --- | --- |
| Physical action codebook | AutoVLA, NoRD, DriveVLA-W0 | Keeps language-style decoding while preserving metric action structure. |
| Shared language-action codebook | LinkVLA | Enables action captioning and action generation in one token space. |
| Fixed trajectory vocabulary | DriveSuprim | Turns planning into ranking; high ceiling but depends on candidate coverage. |
| Masked action tokens | DriveFine, WAM-Diff, DiffusionDriveV2 | Supports iterative refinement and RL over generated trajectories. |
| Continuous action expert | SpanVLA, UniUGP, Alpamayo-R1 | Avoids discretization error but needs a separate continuous decoder. |
| Planning query tokens | Reasoning-VLA, OneDrive | Produces continuous trajectories in parallel without discretizing waypoints. |

## Takeaways

- Text waypoints are weak: AutoVLA's ablation shows physical action tokens vastly outperform text waypoint representations.
- Large vocabularies raise coverage but create selection and calibration problems; DriveSuprim addresses this with coarse-to-fine filtering and soft labels.
- Continuous experts reduce tokenization error, but they move the burden to action-bridge design and action-reasoning alignment.
- Codebook papers should always report both tokenization quality and closed-loop planner quality; good reconstruction alone does not prove deployable driving.
- Planning-query methods avoid codebook design entirely, but they have weaker multimodal coverage unless paired with anchors, perception queries, refinement, or selection.

## OneDrive Position

OneDrive allocates one planning query per future timestep and initializes each with a VAD-derived anchor trajectory. This is closer to Reasoning-VLA's learnable action-query paradigm than to AutoVLA/NoRD action codebooks. The distinctive part is that planning queries live inside the same causal VLM decoder as text and perception queries, so action generation is a query-based continuous head rather than a token vocabulary.

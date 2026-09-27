---
title: Action Tokenization and Codebooks
type: concept
sources: ["raw/papers/[-0.5mm] GRAVA GRAVA_ Grounded Reasoning-to-Action Representation and Learning for Autonomous Driving.md", raw/papers/Unified Driving Tokens_ Representation- and Geometry-Guided Discrete Tokenizer for Driving World Models and Planning.md, raw/papers/AutoVLA_ A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning.md, raw/papers/DriveVLA-W0_ World Models Amplify Data Scaling Law in Autonomous Driving.md, raw/papers/Unifying Language-Action Understanding and Generation for Autonomous Driving.md, raw/papers/NoRD_ A Data-Efficient Vision-Language-Action Model that Drives without Reasoning.md, raw/papers/DiffusionDriveV2_ Reinforcement Learning-Constrained Truncated Diffusion Modeling in End-to-End Autonomous Driving.md, raw/papers/DriveSuprim_ Towards Precise Trajectory Selection for End-to-End Planning.md, raw/papers/SpanVLA_ Efficient Action Bridging and Learning from Negative-Recovery Samples for Vision-Language-Action Model.md, raw/papers/OneDrive_ Unified Multi-Paradigm Driving with Vision-Language-Action Models.md, raw/papers/Plan-R1_ Safe and Feasible Trajectory Planning as Language Modeling.md]
related: [sources/grava.md, sources/unified-driving-tokens.md, concepts/visual-tokenization.md, concepts/diffusion-planner.md, concepts/selection-based-planning.md, concepts/best-of-n.md, concepts/rl-for-ad.md, sources/autovla.md, sources/drivevla-w0.md, sources/linkvla.md, sources/nord.md, sources/diffusiondrive-v2.md, sources/drivesuprim.md, sources/spanvla.md, sources/onedrive.md, sources/plan-r1.md]
created: 2026-05-01
updated: 2026-09-27
confidence: high
---

## What It Is

Action tokenization is the design choice that converts continuous ego trajectories into model outputs: discrete codebook IDs, fixed-vocabulary trajectory selections, masked tokens, or continuous action vectors decoded by an expert.

**Sibling page**: [[concepts/visual-tokenization.md]] covers the other half of the tokenization question — how *pixels* become discrete codes for token-based world models and planner inputs. The two axes are usually designed independently, and [[sources/unified-driving-tokens.md]] is the only ingested paper to treat the visual side as a planning-consumability problem rather than a reconstruction one.

## Main Patterns

| Pattern | Examples | Practical implication |
| --- | --- | --- |
| Physical action codebook | AutoVLA, NoRD, DriveVLA-W0 | Keeps language-style decoding while preserving metric action structure. |
| Shared language-action codebook | LinkVLA | Enables action captioning and action generation in one token space. |
| Fixed trajectory vocabulary | DriveSuprim | Turns planning into ranking; high ceiling but depends on candidate coverage. |
| Masked action tokens | DriveFine, WAM-Diff, DiffusionDriveV2 | Supports iterative refinement and RL over generated trajectories. |
| Motion-token language modeling | Plan-R1 | Treats multi-agent trajectory prediction as autoregressive next-motion-token prediction before RL alignment. |
| Continuous action expert | SpanVLA, UniUGP, Alpamayo-R1 | Avoids discretization error but needs a separate continuous decoder. |
| Planning query tokens | Reasoning-VLA, OneDrive | Produces continuous trajectories in parallel without discretizing waypoints. |

## Takeaways

- Text waypoints are weak: AutoVLA's ablation shows physical action tokens vastly outperform text waypoint representations.
- Large vocabularies raise coverage but create selection and calibration problems; DriveSuprim addresses this with coarse-to-fine filtering and soft labels.
- Continuous experts reduce tokenization error, but they move the burden to action-bridge design and action-reasoning alignment.
- Codebook papers should always report both tokenization quality and closed-loop planner quality; good reconstruction alone does not prove deployable driving.
- Planning-query methods avoid codebook design entirely, but they have weaker multimodal coverage unless paired with anchors, perception queries, refinement, or selection.

## Plan-R1 Position

Plan-R1 ([[sources/plan-r1.md]]) is a non-VLM example of trajectory-as-language modeling. It discretizes trajectories into 0.5-second motion segments using K-disk clustering, with a 1024-token vocabulary per agent category (Vehicle, Pedestrian, Cyclist). A compact transformer decoder predicts next motion tokens for all agents, then RL fine-tunes only the ego planner.

The useful contrast with AutoVLA/NoRD is scope: AutoVLA and NoRD integrate action tokens into a vision-language policy, while Plan-R1 uses motion tokens as the native representation for a standalone multi-agent trajectory model. It shows that tokenized planning and GRPO-style alignment are not VLA-specific.

## OneDrive Position

OneDrive allocates one planning query per future timestep and initializes each with a VAD-derived anchor trajectory. This is closer to Reasoning-VLA's learnable action-query paradigm than to AutoVLA/NoRD action codebooks. The distinctive part is that planning queries live inside the same causal VLM decoder as text and perception queries, so action generation is a query-based continuous head rather than a token vocabulary.

## GRAVA Position

[[sources/grava.md]] uses a **mode-conditioned parametric vocabulary** rather than a codebook or raw waypoints. The model emits a primitive (STOP: endpoint ∈ ℝ²; CRAWL: 8 points; CURVE: 3 Bézier control points; CRUISE: progress, final speed, lateral offset), a gear, and quantized parameters as text. A fixed geometric decoder (stop profile, Bézier, monotone Hermite) produces 8 waypoints over 4 s. The decoder has no parameters, so the whole action choice stays in the autoregressive policy.

- Against direct waypoint text in the same pipeline, the vocabulary is worth **+3.25 PDMS** (87.23 → 90.48).
- The paper's analysis finds that high- and low-reward samples from the same scene **usually share a primitive**, so RL mostly tunes continuous parameters within a mode.
- The design is closest to intent-plus-parameters schemes (see [[concepts/intent-conditioned-planning.md]]). Unlike a learned K-disk codebook (AutoVLA), its coverage is set by hand-chosen shape families. CRAWL keeps full waypoints because low-speed stop-and-go does not fit a smooth curve.


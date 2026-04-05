---
title: NAVSIM Benchmark
type: concept
sources: [raw/papers/ReCogDrive_ A Reinforced Cognitive Framework for End-to-End Autonomous Driving.md, raw/papers/WAM-Flow_ Parallel Coarse-to-Fine Motion Planning via Discrete Flow Matching for Autonomous Driving.md, raw/papers/Senna-2_ Aligning VLM and End-to-End Driving Policy for Consistent Decision Making and Planning.md, raw/papers/Discrete Diffusion for Reflective Vision-Language-Action Models in Autonomous Driving.md, raw/papers/Reasoning-VLA_ A Fast and General Vision-Language-Action Reasoning Model for Autonomous Driving.md]
related: [sources/recogdrive.md, sources/wam-flow.md, sources/senna2.md, sources/reflectdrive.md, sources/reasoning-vla.md, concepts/rl-for-ad.md, concepts/discrete-flow-matching.md, concepts/dual-system-vla.md, concepts/inference-time-safety.md]
created: 2026-04-05
updated: 2026-04-05
confidence: high
---

## What It Is

NAVSIM is a planning-oriented autonomous driving benchmark built on OpenScene (a redistribution of nuPlan). It provides a **non-reactive simulation** environment for closed-loop planning evaluation.

## Dataset

- Built on **OpenScene** / **nuPlan**
- **Sensors**: 8 cameras (1920×1080) + fused LiDAR point cloud from 5 sensors across current + 3 prior frames
- **Splits**: navtrain (1,192 training scenes), navtest (136 evaluation scenes)
- 85,109 trajectory-based QA pairs in navtrain

## Primary Metric: PDMS

**Predictive Driver Model Score**:
$$\text{PDMS} = NC \times DAC \times \frac{5 \times EP + 5 \times TTC + 2 \times C}{12}$$

Sub-metrics:
| Metric | Meaning |
|--------|---------|
| NC | No At-Fault Collision |
| DAC | Drivable Area Compliance |
| EP | Ego Progress |
| TTC | Time-to-Collision |
| C (Comf.) | Comfort |

NC and DAC are multiplicative gates — a collision zeroes the score.

## Extended Metrics (Hydra-MDP++)

- **TL**: Traffic Light Compliance
- **LK**: Lane Keeping Ability
- **DDC**: Driving Direction Compliance
- **EC**: Extended Comfort
- **EPDMS**: Extended PDMS incorporating the above

## Non-Reactive Simulator

NAVSIM's simulator does not model other agents reacting to the ego vehicle. This simplifies evaluation (and RL training) but means scores may not reflect performance in fully interactive scenarios.

## SOTA Progression

| Method | Modality | PDMS | Notes |
|--------|----------|------|-------|
| PARA-Drive | Camera | 84.0 | |
| ARTEMIS | Camera+LiDAR | 87.0 | |
| DiffusionDrive | Camera+LiDAR | 88.1 | |
| WoTE | Camera+LiDAR | 88.3 | |
| AutoVLA | 3 Cam | 89.1 | AR + GRPO |
| ReCogDrive | 3 Cam | 89.6 | AR + Diffusion + GRPO |
| ReflectDrive | 3 Cam | >89.1 (claimed) | Masked diffusion + reflective inference; exact number not in markdown |
| WAM-Flow (5-step) | 1 Cam | 90.3 | DFM + GRPO |
| **Reasoning-VLA-7B** | **3 Cam** | **91.7 (claimed)** | **Learnable queries + GT-based GRPO; comparison only vs. old baselines** |

**Caveat on Reasoning-VLA (91.7)**: Table 6 in the paper compares only against TransFuser/UniAD/Para-Drive (all ~84 PDMS from the original NAVSIM paper). No head-to-head with WAM-Flow (90.3), ReCogDrive (89.6), or DiffusionDrive (88.1). Claimed SOTA status is unverified. See [[sources/reasoning-vla.md]].

WAM-Flow ([[sources/wam-flow.md]]) remains the last verified NAVSIM-v1 SOTA at 90.3, using only a single front camera.

## NAVSIM-v2 SOTA (EPDMS)

| Method | EPDMS | Notable weakness |
|--------|-------|-----------------|
| Artemis | 83.1 | — |
| RecogDrive | 83.6 | — |
| WAM-Flow | 84.7 | EC = 73.9 (low vs. RecogDrive's 87.7) |
| DiffusionDrive | 84.5 | — |
| ResAD | 85.5 | — |
| **Senna-2** | **86.6** | Requires separate NAVSIM fine-tune |

Senna-2 ([[sources/senna2.md]]) is the current NAVSIM-v2 SOTA as of April 2026. Note: this result uses a model fine-tuned specifically on the NAVSIM training split, making it a separate variant from the proprietary-data model used in closed-loop evaluations.

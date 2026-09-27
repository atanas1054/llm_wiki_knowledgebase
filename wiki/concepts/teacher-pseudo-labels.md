---
title: Teacher-Derived Supervision and Pseudo-Labels
type: concept
sources: ["raw/papers/SUV_ Future Scene Understanding as Video Generation for End-to-End Driving.md", raw/papers/DriveDreamer-Policy_ A Geometry-Grounded World–Action Model for Unified Generation and Planning.md, raw/papers/CoWorld-VLA_ Thinking in a Multi-Expert World Model for Autonomous Driving.md, raw/papers/Latent-WAM_ Latent World Action Modeling for End-to-End Autonomous Driving.md, raw/papers/GeoWAM_ Visual Geometry World Action Models for Autonomous Driving.md, raw/papers/GeoWorldAD_ Geometry World Action Model for Autonomous Driving.md, raw/papers/FLARE_ Learning Future-Aware Latent Representations from Vision-Language Models for Autonomous Driving.md, raw/papers/Drive-HWM_ Hierarchical World Models for Dynamic-Latent Guided Autonomous Driving.md, "raw/papers/[-0.5mm] GRAVA GRAVA_ Grounded Reasoning-to-Action Representation and Learning for Autonomous Driving.md", raw/papers/Qwen-Drive-1.0_ An Initial Step towards a Vision-Language Foundation Model for Autonomous Driving.md, raw/papers/ReCogDrive_ A Reinforced Cognitive Framework for End-to-End Autonomous Driving.md, raw/papers/HERMES_ A Holistic End-to-End Risk-Aware Multimodal Embodied System with Vision–Language Models for Long-Tail Autonomous Driving.md, raw/papers/Alpamayo-R1_ Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail.md, raw/papers/SGDrive_ Scene-to-Goal Hierarchical World Cognition for Autonomous Driving.md, raw/papers/How Can Driving World Models Do Counterfactual Prediction_.md, raw/papers/DA-WAM_ Decision-Aligned Future Latents for Driving World Models.md, raw/papers/Auto-JEPA_ A Latent World Model of Continuous Intent for End-to-End Autonomous Driving.md]
related: [sources/suv.md, sources/drivedreamer-policy.md, sources/coworld-vla.md, sources/latent-wam.md, sources/geowam.md, sources/geoworldad.md, sources/flare.md, sources/drive-hwm.md, sources/grava.md, sources/qwen-drive-1.0.md, sources/recogdrive.md, sources/hermes.md, sources/alpamayo-r1.md, sources/sgdrive.md, sources/driving-wm-counterfactuals.md, sources/da-wam.md, sources/auto-jepa.md, concepts/perception-for-planning.md, concepts/world-model-for-ad.md, concepts/reasoning-faithfulness.md, concepts/foundation-backbones-for-ad.md, concepts/vlm-domain-adaptation.md]
created: 2026-09-27
updated: 2026-09-27
confidence: medium
---

# Teacher-Derived Supervision and Pseudo-Labels

## What It Is

Much of the supervision in recent driving models is not human annotation or sensor ground truth. It is the **output of a frozen foundation model** (a teacher) applied offline. Teachers now supply depth, segmentation, tracks, geometry, visual features, reasoning traces and QA pairs. This page records who uses which teacher, and the two recurring risks:
1. **Evaluation against the teacher.** The model is scored on agreement with the same teacher that labelled its training data.
2. **Teacher errors inherited silently.** Label noise is rarely measured.

---

## Teachers in the Wiki

| Teacher | Supplies | Used by | Evaluated against |
|---|---|---|---|
| **SAM 3** | Future segmentation and instance tracks, rendered as video | [[sources/suv.md]] | **The same teacher** (mIoU, AssA@50) |
| **Depth Anything 3** | Relative depth targets | [[sources/suv.md]], [[sources/drivedreamer-policy.md]] | **The same teacher** in both |
| **Depth Anything V2** | Relative depth for abduction | [[sources/driving-wm-counterfactuals.md]] | Matched counterfactual GT |
| **VGGT / StreamVGGT / WorldMirror** | Geometry features, point maps | [[sources/coworld-vla.md]], [[sources/latent-wam.md]], [[sources/geowam.md]], [[sources/geoworldad.md]] | Planning score only |
| **V-JEPA** | Future semantic features | [[sources/coworld-vla.md]] | Planning score only |
| **DINOv2** | Future feature targets | [[sources/flare.md]] | Planning score only |
| **Optical-flow estimator (unnamed)** | Slow-branch flow targets | [[sources/drive-hwm.md]] | Not evaluated; the estimator is not even identified |
| **LLM agents + LiDAR-assisted perception tool** | Grounded QA, reasoning graphs, metric states | [[sources/grava.md]] | An LLM judge against its own references |
| **Qwen3.7-Plus / Qwen3.5-Flash** | Reasoning traces; consistency filtering | [[sources/qwen-drive-1.0.md]] | Filter: **55.9% of public driving VQA survives** |
| **Qwen2.5-VL** | 775K auto-annotated NAVSIM QA | [[sources/recogdrive.md]] | Downstream planning |
| **Frontier VLM** | Risk-aware plan rationales, baked into embeddings | [[sources/hermes.md]] | Downstream planning |
| **NAVSIM PDM simulator** | Scorer targets, hard negatives, RL reward | [[sources/da-wam.md]], [[sources/auto-jepa.md]], most RL papers | The benchmark's own metric; see [[concepts/selection-based-planning.md]] |

---

## Three Recurring Problems

### 1. Scoring the student against its own teacher

- [[sources/suv.md]]: all segmentation, depth and track metrics measure agreement with SAM 3 and DA3. The paper says so explicitly.
- [[sources/drivedreamer-policy.md]]: depth is both trained and evaluated against DA3 pseudo-labels.

Agreement with the teacher bounds quality from above only up to the teacher's own error, and it rewards reproducing the teacher's mistakes. **NAVSIM has LiDAR and annotated boxes, so real ground truth was available in both cases.**

A useful internal control from SUV: **generating the RGB future and running the teachers on it** beats native generation on segmentation and tracking, and loses on depth. At least one comparison is teacher-fair.

### 2. Label noise is almost never measured

- [[sources/qwen-drive-1.0.md]] is the exception. A consistency filter rejects **44%** of 24 public driving-VQA datasets against their own source annotations.
- [[sources/grava.md]]'s printed showcase transcript has a vehicle/pedestrian type flip and a 15–20 m vs 24.0 m distance conflict.

If public, human-curated corpora fail at 44%, agent-generated labels should not be assumed clean.

### 3. Privileged teachers leak into "annotation-free" claims

"No perception annotations" is often true while "no privileged supervision" is not:
- [[sources/da-wam.md]] and [[sources/auto-jepa.md]] use simulator-derived metrics.
- [[sources/grava.md]] uses LiDAR 3D states behind every number its front-camera model is trained to say.
- [[sources/sgdrive.md]] is camera-only at inference but needs occupancy and boxes to train.

Traces written with the future in view are a special case of the same leak; see [[concepts/reasoning-faithfulness.md#hindsight]].

---

## Why Teachers Still Win

The rise of teacher supervision is not an accident:
- **Annotation-free scaling.** SUV builds three extra target streams for all of navtrain at zero labelling cost.
- **Dense targets.** Point maps, features and flow give far more signal per frame than sparse boxes.
- **Measurable planning benefit.**
  - SUV: structured teacher streams give +1.0 navtest / +2.3 navhard even without inference access.
  - Latent-WAM: geometric distillation.
  - FLARE: a DINOv2 future-feature objective is the core of an 86.9 SFT / 91.4 after-RL pipeline, though the RL stage carries +4.5 of it.

---

## Open Questions

- **Teacher vs. real ground truth, same planner.** Swap DA3 depth for LiDAR-projected depth on NAVSIM and compare both the future-depth metric and planning score. It is cheap and has never been done.
- **Does teacher quality propagate to planning?** For example Depth Anything V2 vs V3, or SAM 2 vs SAM 3, under a fixed recipe.
- **Should agent-generated reasoning corpora carry a measured noise rate** in the style of Qwen-Drive's 55.9%, before any planning claim is made on them?

---
title: Perception-Enhanced Planning in VLA Models
type: concept
sources: [raw/papers/Percept-WAM_ Perception-Enhanced World-Awareness-Action Model for Robust End-to-End Autonomous Driving.md]
related: [sources/percept-wam.md, concepts/diffusion-planner.md, concepts/vlm-domain-adaptation.md, concepts/navsim-benchmark.md, concepts/world-model-for-ad.md]
created: 2026-04-05
updated: 2026-04-05
confidence: high
---

## The Core Tension

End-to-end VLA planners face a structural choice:

**Option A — QA-style spatial reasoning** (EMMA, DriveVLM): spatial understanding encoded as question-answer supervision. Indirect localization signals; no persistent world state; duplicate detections with poorly calibrated confidence in crowded scenes.

**Option B — Encoder-decoder pipelines** (DiffusionDrive, WAM-Flow): skip VLM entirely for spatial tasks; use specialized BEV encoders + diffusion/flow decoders. Sacrifice reasoning capacity for geometric precision. Brittle in long-tail scenarios.

**Option C — Embedded world states** (Percept-WAM, UniAD philosophy in VLM): encode persistent, metrically grounded world states as specialized tokens inside the VLM backbone. Perception and planning jointly optimized with shared representations.

## Why Perception Helps Planning

The motivation is borrowed from UniAD (planning-oriented multi-task learning): explicit perception supervision forces the model to maintain accurate, spatially precise representations of scene geometry, which in turn grounds trajectory generation.

Two mechanisms:

**1. Shared representation grounding**: World-PV and World-BEV tokens must simultaneously support detection/segmentation and trajectory decoding. This forces the backbone to maintain metric accuracy that pure trajectory-from-images training does not require.

**2. Modality-aligned trajectory decoding**: when the trajectory decoder explicitly attends to World-BEV tokens (3D spatial context), World-PV tokens (semantic context), and ego-state (kinematic context) via separate attention heads, it avoids over-reliance on any single representation.

Evidence from Percept-WAM: joint PV 2D+3D training improves 2D detection by +3.2 mAP (synergy between tasks). The four-query trajectory decoder (Q_ego, Q_pv, Q_bev, Q_full) achieves best L2 accuracy and best inference speed among all decoding variants.

## World Tokens: Architecture Pattern

### World-PV Tokens (Perspective-View)

- Source: VLM ViT image features, patchified into H×W spatial grid
- Each grid cell = **localized query** tied to an image-plane coordinate
- Prediction: one bounding box per grid cell (2D or mono-3D), via parallel AR decoding
- Grid tokens across cells are **mutually masked** → independent parallel prediction, not sequential

**Format** (2D): `cls, <box> x,y,w,h </box>, <conf> s </conf>`  
**Format** (3D): `cls, <box> x,y,z,w,h,ℓ,θ,vx,vy </box>, <conf> s </conf>`

### World-BEV Tokens (Bird's-Eye-View)

- Learnable query tokens forming H×W BEV grid centered on ego vehicle
- Attend to World-PV tokens via cross-attention → implicit PV→BEV view lifting (no explicit depth supervision)
- Optional initialization from LiDAR (PointPillars → PixelUnshuffle → MLP) for metrically grounded 3D priors
- BEV grid resolution: 40×40 for detection, 10×10 for segmentation

**Camera-only BEV** is significantly weaker than LiDAR-initialized (25.0 vs. 58.9 mAP on nuScenes). The view-lifting problem remains an open challenge.

### Token Reuse for Action

The same World-PV and World-BEV tokens produced during the perception prefill stage are directly reused by the trajectory decoder — no additional forward pass required. This is computationally free: perception and planning share the prefill computation.

## IoU-Aware Confidence: Calibration for Dense Scenes

A critical practical contribution. Standard VLM confidence (softmax class logits) is systematically overconfident, producing many false positives in crowded scenes. Percept-WAM addresses this with a dedicated IoU prediction token per box.

**Key insight on training data**: the distribution of the confidence-tuning dataset matters enormously:
- Random-perturb of GT → near-uniform IoU distribution → hurts performance (−1.2 AP)
- Uniform sampling of model predictions → still misaligned → hurts (−1.9 AP)
- **Real model-prediction distribution (skewed toward low IoU)** → aligned with actual inference → +1.5 AP, +2.3 AP₇₅

This finding generalizes: calibration training should match the realistic distribution of model errors, not a synthetic distribution designed for coverage.

**Final confidence** = class_conf × predicted_IoU — provides a unified, localization-sensitive reliability measure suitable for NMS post-processing.

## Grid-Conditioned Parallel AR Decoding

Standard sequence-based detectors (Pix2Seq, LLM-based detectors) generate all boxes in a left-to-right sequence — attention from later boxes to earlier boxes creates implicit position biases and sequential coupling.

Percept-WAM's grid-conditioned approach:
1. Interpolate World-PV/BEV tokens at each grid position → localized grid token
2. Each grid token independently predicts one object at that location
3. Grid tokens from different locations are **mutually masked** → fully parallel generation
4. 16× inference speedup over sequential AR for BEV detection (Table 6) with no accuracy loss

This is structurally similar to WAM-Flow's parallel DFM decoding, but applied to object detection rather than trajectory generation.

## Comparison: Perception Integration Approaches in AD VLMs

| Approach | Spatial supervision | World state type | Planning benefit |
|---|---|---|---|
| QA-style (EMMA, DriveVLM) | Indirect (language) | None — ephemeral text | Indirect reasoning only |
| Multi-task E2E (UniAD) | Direct (detection heads) | BEV occupancy, motion | Strong (planning-oriented) |
| **World tokens (Percept-WAM)** | **Direct (token-level)** | **World-PV + World-BEV tokens** | **Shared representation + modality-aligned decoding** |
| World model (UniUGP) | Video generation | Future frame prediction | Causal feature grounding |

UniAD's planning-oriented multi-task learning is the closest philosophical predecessor — Percept-WAM instantiates this idea inside a VLM backbone rather than a specialized E2E architecture.

## Four-Query Trajectory Decoder

Rather than a single action head, Percept-WAM uses four parallel MLP decoders with modality-specific attention masking:

| Query | Attends to | Purpose |
|---|---|---|
| Q_ego | Ego-state | Kinematic grounding |
| Q_pv | World-PV tokens | Semantic/appearance context |
| Q_bev | World-BEV tokens | 3D geometric context |
| Q_full | All tokens | Final trajectory (inference output) |

All four trained simultaneously with Smooth-L1 loss. The separate decoders prevent the model from ignoring any modality — each must independently learn to produce reasonable trajectories from its limited view.

This contrasts with Reasoning-VLA's single set of learnable action queries (which attend to all VLM hidden states via cross-attention without modality partitioning).

## Limitations and Open Questions

1. **LiDAR dependence**: camera-only BEV is 25.0 mAP vs. 58.9 with LiDAR init — a 2.4× gap. The view-lifting problem from monocular cameras to BEV remains hard without depth supervision.

2. **No RL**: all methods in this wiki that approach or exceed 90 PDMS use GRPO. Percept-WAM uses purely SFT — RL is explicitly listed as future work in the paper's conclusion.

3. **Comfort issue on base model**: the plain query-based decoder produces Comf=92.8 vs. 99-100 for all other methods. This suggests the trajectory output has uncomfortable acceleration profiles that require post-processing (trajectory scoring) to fix. The root cause is not analyzed in the paper.

4. **Grid resolution trade-offs**: higher BEV grid resolution gives +9.1% mAP but the relationship between grid granularity and planning accuracy is not ablated.

5. **Task interference**: the model trains on 7+ tasks simultaneously. While joint training generally helps, negative transfer is possible — PV semantic segmentation training could harm trajectory prediction quality. No systematic analysis provided.

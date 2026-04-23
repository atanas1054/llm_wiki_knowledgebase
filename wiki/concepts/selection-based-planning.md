---
title: Selection-Based Trajectory Planning
type: concept
sources: [raw/papers/DriveSuprim_ Towards Precise Trajectory Selection for End-to-End Planning.md, raw/papers/DiffusionDriveV2_ Reinforcement Learning-Constrained Truncated Diffusion Modeling in End-to-End Autonomous Driving.md, raw/papers/From Representational Complementarity to Dual Systems_ Synergizing VLM and Vision-Only Backbones for End-to-End Driving.md]
related: [sources/drivesuprim.md, sources/diffusiondrive-v2.md, sources/hybriddriveVLA.md, sources/dreameraD.md, concepts/navsim-benchmark.md, concepts/best-of-n.md, concepts/diffusion-planner.md, concepts/rl-for-ad.md]
created: 2026-04-23
updated: 2026-04-23
confidence: high
---

## What It Is

Selection-based planning is a trajectory prediction paradigm for end-to-end autonomous driving. Rather than regressing a single trajectory or sampling stochastically, the model selects the best option from a **fixed pre-defined vocabulary** of candidate trajectories.

---

## Core Paradigm

```
Vocabulary: {τ₁, τ₂, ..., τ_N}   (N ≈ 8192 candidates, pre-computed)
                     ↓
Scorer: estimates quality s_i^(m) per trajectory per metric m
                     ↓
Selection: T = τ_k  where k = argmax_i s_i
```

The vocabulary is generated offline (e.g., K-Means over expert trajectories) and covers diverse maneuver types. At inference, the model does not generate trajectories — it scores all candidates and returns the top-ranked one.

**Key distinction from other paradigms**:
| Paradigm | At inference | Trajectory source |
|---|---|---|
| Regression | Outputs one trajectory | Learned regressor |
| Diffusion/FM | Samples from noise | Denoising process |
| Best-of-N sampling | Runs N forward passes | Same model, N times |
| Selection-based | Scores N fixed candidates | Pre-defined vocabulary |
| Selection-based + BoN | Oracle over multiple runs | Vocabulary + stochastic |

---

## Theoretical Ceiling (Oracle Study)

DriveSuprim's oracle study (Table 1) quantifies how much selection-based methods can achieve with perfect scoring:

| Top-K oracle selection | PDMS |
|---|---|
| Top-1 (best current model) | 91.9 |
| Top-4 | 94.5 |
| Top-16 | 96.1 |
| Top-256 | 98.7 |
| Human ground truth | 94.8 |

**Key insight**: with oracle selection from just **4 candidates**, you nearly match human GT (94.5 vs. 94.8). With 256 candidates, you reach 98.7 PDMS — near-perfect on NAVSIM. The bottleneck is entirely in the **selector quality**, not candidate coverage.

This ceiling is higher than stochastic BoN-N results (e.g., Curious-VLA BoN-6 = 94.8 PDMS at N=6) because the vocabulary is purpose-built for coverage, whereas stochastic sampling from a single model produces correlated outputs. See [[concepts/best-of-n.md]].

---

## Three Failure Modes

Selection-based methods share three structural weaknesses identified by DriveSuprim:

### 1. Hard Negatives

The vocabulary contains thousands of obviously bad trajectories ("easy negatives"). During training, BCE loss forces the model to score these correctly — but this dominates the gradient signal. The model rarely encounters two plausible-looking trajectories where one is subtly unsafe ("hard negatives"). As a result, fine-grained discrimination remains weak.

**Fix (DriveSuprim)**: coarse-to-fine filtering — first pass selects top-256 (mostly hard negatives once obvious ones are removed), second pass scores only those 256 at higher precision.

### 2. Directional Bias

Real driving is dominated by straight-ahead motion. In NAVSIM, only 8% of ground-truth trajectories involve turns >30°. Training on this distribution naturally produces a model that underperforms on turns.

**Fix (DriveSuprim)**: rotation-based data augmentation — simulate ego rotation by shifting camera FOV, proportionally rotating GT trajectory.

### 3. Hard Binary Labels

Safety scores are {0,1} per metric. BCE against binary labels creates sharp training boundaries — a trajectory just below the collision threshold is treated identically to a catastrophically bad one. This causes training instability and oversensitivity to minor trajectory variations.

**Fix (DriveSuprim)**: EMA self-distillation with clipped soft labels ($\delta_m = 0.15$).

---

## Methods in the Wiki Using Selection-Based Planning

| Method | Vocabulary Size | Scoring | Notes |
|---|---|---|---|
| **Hydra-MDP** | 8192 | Single-stage multi-head | Multi-teacher distillation; won NAVSIM challenge |
| **HydraMDP++** | 8192 | Single-stage multi-head | Added DDC, TLC, EC metrics for NAVSIM-v2 |
| **DriveSuprim** | 8192 (→ 256) | Two-stage coarse-to-fine | Rotation aug + EMA self-distill; **93.5 PDMS** |
| **DreamerAD** | 8192 (→ 256) | Learned latent AD-RM | Gaussian vocab sampling; reward from latent WM |
| **HybridDriveVLA** | 2 + 9 interp. | Trajectory scorer | Cross-model (VLM + ViT) with linear interpolations |

### DreamerAD as a deployable selection variant

DreamerAD generates 256 trajectories via Mahalanobis-ranked Gaussian sampling over the 8192 vocabulary, then selects via a learned reward model (AD-RM) trained on latent video features — no PDM simulator needed at inference. This is the closest approach in the wiki to a deployable selection system: the selection quality is approx. but fast. +2.6 EPDMS from base to selected. See [[sources/dreameraD.md]].

---

## Coarse-to-Fine Selection (DriveSuprim)

The critical DriveSuprim ablation (Table 5):

| Modification | EPDMS | Change |
|---|---|---|
| Single-stage (Hydra-MDP, ViT-L) | 85.6 | baseline |
| + 6-layer decoder (more parameters) | 85.3 | −0.3 |
| + Layer-wise scoring (aux loss per layer) | 85.6 | +0.3 |
| + Trajectory filtering to 256 | **86.4** | **+0.8** |

Only trajectory filtering helps. Adding decoder depth or auxiliary supervision without filtering does nothing. The model must be presented with a concentrated hard-negative set.

**Why this works**: once easy negatives are removed in Stage 1, Stage 2 faces a set of trajectories that all look plausible. The refinement decoder must develop genuine fine-grained discrimination. The gradient signal from easy negatives no longer dominates.

This is analogous to Cascade R-CNN for object detection (two-stage cascade with progressively tightening IoU thresholds), applied to trajectory scoring.

---

## Rotation-Based Augmentation

The augmentation pipeline:

1. Sample rotation angle $\theta \sim U[-\pi/6, +\pi/6]$
2. Concatenate three cameras into pseudo-panoramic view: $[l_0 | f | r_0]$
3. Crop the standard-FOV window from the panorama, shifted by $\theta$
4. Rotate GT trajectory waypoints $(u_1, \ldots, u_l)$ by $-\theta$ around origin $u_0$
5. Compute loss $L_{\text{aug}}$ identically to $L_{\text{ori}}$

**Effect**: the original NAVSIM dataset has a forward-heavy trajectory distribution. Post-augmentation, all directions appear at similar frequency (Figure 4 in [[sources/drivesuprim.md]]).

**Performance impact by scenario type**:
| Scenario | Gain vs. no augmentation |
|---|---|
| Turning scenarios | +2–3% EPDMS |
| Near-straight scenarios | +0.9% EPDMS |

This is first application in the AD wiki of camera-shift-based rotation augmentation for trajectory planning.

---

## Relationship to Best-of-N Sampling

Selection-based planning and BoN sampling are often confused but are architecturally different:

| Property | Selection-based | Stochastic BoN |
|---|---|---|
| Trajectory source | Pre-defined fixed vocabulary | N model forward passes |
| Selection at inference | Learned scorer | Oracle (PDM simulator) |
| Deployable? | Yes (scorer replaces oracle) | No (oracle unavailable) |
| Ceiling | High: 98.7 PDMS (256-oracle) | Medium: 94.8 PDMS (N=6, Curious-VLA) |
| Diversity source | Vocabulary design | Stochastic decoding |

The fixed-vocabulary oracle ceiling (98.7 PDMS at top-256) is substantially higher than stochastic BoN (94.8 at N=6). This is because the vocabulary is curated to cover diverse maneuver types systematically, while stochastic decoding from a single model produces correlated near-optimal outputs.

The practical convergence point is in deployable selectors: DreamerAD (latent reward model over 256 vocabulary candidates) and HybridDriveVLA (cross-model scorer) both convert oracle selection into feasible inference, with partial but real gains. See [[concepts/best-of-n.md]].

---

## NAVSIM Performance Overview

Selection-based methods' trajectory on the NAVSIM-v1 leaderboard:

| Method | PDMS (ViT-L) | Year |
|---|---|---|
| Hydra-MDP | 89.9 | 2024 |
| HydraMDP++ | 85.6* (EPDMS) | 2024 |
| DreamerAD | 88.7 (no ViT-L) | 2025 |
| **DriveSuprim** | **93.5** | 2025 |

*HydraMDP++ is evaluated primarily on NAVSIM-v2 (EPDMS).

DriveSuprim (93.5) is the highest single-sample, non-VLM, non-ensemble result in the wiki, surpassing DiffusionDriveV2 (91.2 with Camera+LiDAR) and HybridDriveVLA (92.1 dual-model ensemble). See [[concepts/navsim-benchmark.md]] for the full SOTA table.

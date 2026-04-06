---
title: "AutoDrive-R²: Reasoning and Self-Reflection for VLA Trajectory Planning"
type: source-summary
sources: [raw/papers/AutoDrive-R²_ Incentivizing Reasoning and Self-Reflection Capacity for VLA Model in Autonomous Driving.md]
related: [concepts/rl-for-ad.md, concepts/vlm-domain-adaptation.md, sources/reasoning-vla.md, sources/curious-vla.md, sources/autovla.md]
created: 2026-04-05
updated: 2026-04-05
confidence: high
---

**arXiv**: https://arxiv.org/html/2509.01944v1  
**Affiliation**: AMAP, Alibaba Group; University of Queensland; Lanzhou University; Case Western Reserve University

## One-Line Summary

AutoDrive-R² adds a 4-step CoT with explicit self-reflection (Visualization → Calculation → Logic → Reflection) to SFT, then applies GRPO with a physics-grounded reward (position + steering + velocity + temporal smoothness) to achieve 0.19m avg L2 on nuScenes and 0.20m zero-shot on Waymo using only 12K training samples.

## Core Problem Statement

Two limitations targeted:
1. **Physically infeasible trajectories** — text waypoints and direct VLM generation produce unrealistic outputs; meta-actions/latent tokens break end-to-end optimization
2. **Inadequate reasoning** — simple CoT strategies fail on complex scenarios; lack of kinematic constraints produces trajectories that violate vehicle physics

## Method Overview

![[x2 11.png|AutoDrive-R² two-stage pipeline]]

*Figure 2: Stage 1 — nuScenesR²-6K SFT with 4-step CoT + self-reflection. Stage 2 — GRPO with physics-grounded reward (spatial alignment, vehicle dynamics, temporal smoothness).*

**Base model**: Qwen2-VL-7B (SFT stage), evaluated at both 3B and 7B  
**Input**: single front-view camera + historical vehicle states (position, acceleration, velocity, steering angle)  
**Output**: `<think>reasoning</think><answer>(x₁,y₁),(x₂,y₂),...,(x₆,y₆)</answer>` — BEV trajectory at 0.5s intervals over 3s horizon

## Stage 1: nuScenesR²-6K Dataset + SFT

### Dataset Construction
- **6,000 image-trajectory pairs** from nuScenes training set
- Annotated by **Qwen2.5-VL-72B** given front-view image + historical vehicle states + GT trajectory as hint
- Structured prompt guides a 4-step logical chain with self-reflection

### Four-Step CoT Chain

| Step | Name | Content |
|------|------|---------|
| 1 | **Visualization** | Scene understanding: obstacle/lane localization, traffic sign and signal detection |
| 2 | **Calculation** | Physics-based kinematic prediction: $x(t+1) = x_n + v_x \Delta t + \frac{1}{2} a_{avg} \Delta t^2$; lateral offset: $\Delta y = v \tan(\theta)$ |
| 3 | **Logic** | Traffic rule synthesis: collision check, red-light compliance, intersection handling → recommended action |
| 4 | **Reflection** | **Self-validation**: check if predicted trajectory requires achievable speed/acceleration given vehicle history; flag and correct contradictions |

The self-reflection step (Step 4) implements **backward-checking** — the model verifies its own prediction's physical feasibility before emitting the answer. This is modeled after mathematical reasoning frameworks that validate conclusions through reverse derivation.

**Prompt design** encourages internal dialogue ("let me think", "wait", "Hmm") to simulate deliberative reasoning.

**"Aha Moment" (Appendix A.3)**: the model spontaneously self-corrects during reasoning — e.g., generates a trajectory, then catches a physics violation in the reflection step and revises it. This emergent behavior is not explicitly trained for but arises from the structured prompt.

### SFT Contribution (ablation)
- Baseline Qwen2.5-VL-7B: **1.45m** avg L2
- After SFT on nuScenesR²-6K: **0.27m** — **81.4% improvement**

## Stage 2: Physics-Grounded GRPO

### Reward Function

$$r_{acc} = \lambda_{pos} r_{pos} + \lambda_{ste} r_{ste} + \lambda_{vel} r_{vel} + \lambda_{tem} r_{tem}$$

All $\lambda = 1$ in experiments. Total reward: $r_i = r_{acc}^i + r_{format}^i$ (format: binary for `<think>...</think><answer>...</answer>` compliance).

**Four reward components:**

| Component | Formula | Physics constraint |
|-----------|---------|-------------------|
| $r_{pos}$ | $\frac{1}{N}\sum_i (x^i - x_{gt}^i)^2 + (y^i - y_{gt}^i)^2$ | Spatial accuracy (global path adherence) |
| $r_{ste}$ | $\frac{1}{N}\sum_j (\theta^j - \theta_{gt}^j)^2$ | Steering kinematics (prevent abrupt turns) |
| $r_{vel}$ | $\frac{1}{N}\sum_k (v^k - v_{gt}^k)^2$ | Velocity compliance (prevent unphysical acceleration) |
| $r_{tem}$ | $\frac{1}{N}\sum_j (\theta^j - \theta^{j-1})^2 + \sum_k (v^k - v^{k-1})^2$ | Temporal smoothness (suppress oscillations) |

Unlike Reasoning-VLA's binary kinematic constraints (pass/fail steering angle and acceleration bounds), AutoDrive-R²'s reward components are continuous MSE terms against GT — providing dense gradient signal.

### GRPO Configuration
- G=6 responses per input (optimal; G=8 gives same 0.19m but higher compute)
- KL divergence against SFT reference policy (β hyperparameter)
- Max completion length: 4,096 tokens
- Learning rate: 5e-7, accumulated batch size of 1

## Key Finding: SFT Cold Start is Necessary for RL

Inspired by DeepSeek-R1-Zero, the authors attempted **RL-only training** (no SFT):
- RL-only: **0.33m** avg L2
- SFT-only: **0.27m**
- SFT + RL: **0.19m**

RL alone fails to explore the high-dimensional reasoning space needed for multi-step calculation and contextual logic. SFT establishes structured reasoning chains; RL then refines physical feasibility on top of this foundation.

## Results

### nuScenes Open-loop (Table 1, ST-P3 protocol)

| Method | L2@1s | L2@2s | L2@3s | Avg ↓ |
|--------|-------|-------|-------|-------|
| Ego-MLP | 0.15 | 0.32 | 0.59 | 0.35 |
| OmniDrive | 0.14 | 0.29 | 0.55 | 0.33 |
| EMMA | 0.14 | 0.29 | 0.54 | 0.32 |
| EMMA+ | 0.13 | 0.27 | 0.48 | 0.29 |
| Impromptu-VLA | 0.13 | 0.27 | 0.53 | 0.30 |
| AutoDrive-R² 3B | 0.35 | 0.49 | 0.62 | 0.49 |
| **AutoDrive-R² 7B** | **0.13** | **0.19** | **0.25** | **0.19** |

AutoDrive-R² 7B leads at 2s and 3s horizons. Trained on 6K SFT + 6K RL samples vs. EMMA+'s ~103K internal dataset.

### Waymo Zero-shot (Table 2)

| Method | L2@1s | L2@2s | L2@3s | Avg ↓ |
|--------|-------|-------|-------|-------|
| EMMA | 0.12 | 0.28 | 0.61 | 0.34 |
| EMMA+ | 0.11 | 0.25 | 0.53 | 0.30 |
| AutoDrive-R² 3B | 0.23 | 0.36 | 0.51 | 0.37 |
| **AutoDrive-R² 7B** | **0.11** | **0.19** | **0.29** | **0.20** |

33.3% improvement over EMMA+ zero-shot. Trained on nuScenes only; tested on Waymo without fine-tuning.

### Ablation (Table 3, nuScenes, 7B)

**Training stage ablation:**

| Config | Avg L2 |
|--------|--------|
| Baseline Qwen2.5-VL-7B | 1.45 |
| + SFT only | 0.27 |
| + RL only (no SFT) | 0.33 |
| + SFT, w/o 4-step structure | 0.25 |
| + SFT, w/o self-reflection | 0.23 |
| Full (SFT + RL) | **0.19** |

**Reward component ablation:**

| Config | Avg L2 | Δ vs. full |
|--------|--------|-----------|
| w/o $r_{pos}$ | 0.53 | +179% |
| w/o $r_{tem}$ | 0.24 | +26.3% |
| w/o $r_{vel}$ | 0.22 | +15.8% |
| w/o $r_{ste}$ | 0.21 | +10.5% |
| Full AutoDrive-R² | **0.19** | — |

Spatial alignment ($r_{pos}$) is critical — its removal causes near-collapse (0.53m). Temporal smoothness ($r_{tem}$) is second most important.

**Group size ablation (Table 4):**

| G | Avg L2 |
|---|--------|
| 2 | 0.23 |
| 4 | 0.20 |
| **6** | **0.19** |
| 8 | 0.19 |

G=6 optimal (G=8 gives same result with higher compute).

## Implementation Details

| Setting | Value |
|---------|-------|
| Base model | Qwen2-VL-7B (SFT) / Qwen2.5-VL-3B or 7B |
| Cameras | Single front-view |
| Trajectory | BEV text waypoints, 6 points @ 0.5s, 3s horizon |
| SFT data | nuScenesR²-6K (6K samples) |
| RL data | 6K nuScenes samples |
| Learning rate | 5e-7 |
| Batch size | 1 (accumulated) |
| GRPO group size G | 6 |
| Max completion | 4,096 tokens |

## Limitations

1. **Single front-view camera only** — no surround-view; simpler perception than multi-camera methods (3xC, 8xC)
2. **No NAVSIM evaluation** — cannot compare on PDMS with DriveFine, WAM-Flow, Curious-VLA
3. **No closed-loop evaluation** — only open-loop L2; no Bench2Drive or interactive sim
4. **Physics reward requires GT steering/velocity** — these kinematic labels are not available in all real-world driving datasets
5. **Self-reflection is one fixed step** — not truly iterative; the 4th step validates but does not re-run the full reasoning chain
6. **Very small training set (6K)** — impressive results but unknown behavior at distribution edges; nuScenes-to-Waymo transfer works but broader generalization untested
7. **3B model notably weaker** — 0.49m nuScenes vs. 7B's 0.19m; framework benefits heavily from model scale
8. **"Zero-shot Waymo" caveat** — trained on nuScenes (US/Singapore), tested on Waymo (US); different sensor configs and geography. Strong result but not a fully out-of-distribution zero-shot

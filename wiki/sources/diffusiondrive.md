---
title: "DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving"
type: source-summary
sources: [raw/papers/DiffusionDrive_ Truncated Diffusion Model for End-to-End Autonomous Driving.md]
related: [concepts/diffusion-planner.md, concepts/navsim-benchmark.md, sources/recogdrive.md, sources/wam-flow.md, sources/drivefine.md]
created: 2026-04-15
updated: 2026-04-15
confidence: high
---

## Overview

**DiffusionDrive** is a real-time diffusion-based end-to-end autonomous driving planner from HUST and Horizon Robotics (November 2024). It is the **first paper to successfully apply diffusion models to real-time E2E AD**, addressing two failure modes of vanilla diffusion policy: mode collapse and computational overhead. It uses traditional perception (ResNet-34/50 backbone + Camera + LiDAR) — no VLM.

**arXiv**: 2411.15139v1  
**Org**: HUST (Institute of AI + School of EIC); Horizon Robotics  
**Code**: [hustvl/DiffusionDrive](https://github.com/hustvl/DiffusionDrive)

DiffusionDrive (88.1 PDMS, Camera+LiDAR, ResNet-34) is widely used as a baseline in subsequent VLA papers in this wiki.

---

## Figure 1: Paradigm Comparison

![[raw/assets/x1 19.png]]

Four planning paradigms compared:
- **(a) Single-mode regression** (UniAD, Transfuser): direct MLP regression, no diversity
- **(b) Vocabulary sampling** (VADv2, Hydra-MDP): 8192 fixed anchors, scoring-based selection
- **(c) Vanilla diffusion** (Diffusion Policy): random Gaussian start, 20 DDIM steps, mode collapse
- **(d) Truncated diffusion** (DiffusionDrive): anchored Gaussian start, 2 steps, diverse and real-time

---

## Problem: Vanilla Diffusion Fails for Driving

Two failure modes when directly applying robotic diffusion policy (DDIM, 20 steps) to driving:

### 1. Mode Collapse

![[raw/assets/x2 17.png]]

Sampling 20 random Gaussian noises and denoising them all converges to near-identical trajectories. Driving is far more constrained than robot manipulation — lane geometry and road topology collapse the reachable distribution regardless of the starting noise.

**Quantitative measure — Mode Diversity Score $\mathcal{D}$** (IoU-based):
$$\mathcal{D} = 1 - \frac{1}{N}\sum_{i=1}^{N} \frac{\text{Area}(\tau_i \cap \bigcup_{j=1}^{N} \tau_j)}{\text{Area}(\tau_i \cup \bigcup_{j=1}^{N} \tau_j)}$$

Higher $\mathcal{D}$ = more diverse trajectories. Vanilla diffusion achieves only 11% — effectively collapses to one mode.

### 2. Computational Overhead

20 DDIM steps × 6.5ms/step = 130ms → **7 FPS**. Unusable for real-time driving.

---

## Core Innovation: Truncated Diffusion Policy

![[raw/assets/x4 16.png]]

**Figure 3**: Vanilla diffusion (full schedule, Gaussian noise → clean) vs. truncated diffusion (short schedule, anchored Gaussian → clean).

**Key insight**: Human drivers follow structured driving patterns — straight, left turn, right turn, lane change. These form natural "anchor" regions in trajectory space. Starting denoising from these anchors, not from random noise, removes the need for many denoising steps and naturally separates modes.

### Training

1. **Cluster anchors**: K-Means ($N_\text{anchor}$ = 20) on all training trajectories
2. **Truncate forward schedule**: add only small Gaussian noise around each anchor:
$$\tau_k^i = \sqrt{\bar{\alpha}^i}\,\mathbf{a}_k + \sqrt{1-\bar{\alpha}^i}\,\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0,\mathbf{I}), \quad i \in [1, T_\text{trunc}]$$
where $T_\text{trunc} = 50$ (out of 1000 full steps)

3. **Training objective**: reconstruction + classification:
$$\mathcal{L} = \sum_{k=1}^{N_\text{anchor}} \left[y_k \mathcal{L}_\text{rec}(\hat{\tau}_k, \tau_\text{gt}) + \lambda\,\text{BCE}(\hat{s}_k, y_k)\right]$$
Positive sample ($y_k = 1$) = anchor closest to GT; all others negative.

### Inference

- Sample $N_\text{infer}$ noisy trajectories from anchored Gaussian distribution
- Run **2 DDIM denoising steps** (vs. 20 for vanilla)
- Select trajectory with highest predicted confidence score $\hat{s}_k$

**Key advantage**: $N_\text{infer}$ can differ from $N_\text{anchor}$ at inference — flexible number of trajectory hypotheses without retraining. More samples → better coverage of action space.

---

## Architecture: Cascade Diffusion Decoder

![[raw/assets/x5 17.png]]

**Figure 4**: DiffusionDrive integrates any perception backbone and takes various sensor inputs. The cascade diffusion decoder (b) progressively denoises noisy trajectories through scene-context interactions.

The cascade diffusion decoder replaces the UNet (101M params, heavy) with a lightweight transformer (60M params, -39%) that directly interacts with perception outputs.

**Per-layer structure:**
1. **Spatial cross-attention** (deformable): trajectory features attend to BEV/PV features at trajectory coordinates
2. **Agent/map cross-attention**: trajectory features attend to agent/map queries from perception module
3. **FFN**
4. **Timestep Modulation layer**: injects diffusion timestep information
5. **MLP head**: predicts confidence score $\hat{s}_k$ + coordinate offset from initial noisy trajectory

**Cascade**: 2 decoder layers stacked, with **shared parameters** across denoising timesteps. Output of step $i$ serves as input to step $i+1$.

**Final selection**: top-1 confidence-scored trajectory output for evaluation.

---

## Roadmap: Transfuser → DiffusionDrive

**Table 2: Step-by-step improvement on NAVSIM navtest**

| Method | PDMS | Step Time | Steps | Total Time | Diversity $\mathcal{D}$ | Params | FPS |
|--------|------|-----------|-------|-----------|------------------------|--------|-----|
| Transfuser (det.) | 84.0 | 0.2ms | 1 | 0.2ms | 0% | 56M | 60 |
| Transfuser + vanilla DP | 84.6 (+0.6) | 6.5ms | 20 | 130ms | 11% | 101M | 7 |
| Transfuser + truncated DP | 85.7 (+1.7) | 6.9ms | 2 | 13.8ms | 70% | 102M | 27 |
| **DiffusionDrive** | **88.1 (+4.1)** | **3.8ms** | **2** | **7.6ms** | **74%** | **60M** | **45** |

Truncated DP alone: +1.1 PDMS, 59% diversity improvement, 10× fewer steps.  
Full DiffusionDrive (+ cascade decoder): further +2.4 PDMS, 39% fewer params, real-time at 45 FPS.

---

## Results

### NAVSIM navtest (Table 1)

| Method | Input | Backbone | Anchors | NC | DAC | TTC | Comf | EP | PDMS |
|--------|-------|----------|---------|-----|-----|-----|------|----|------|
| UniAD | Camera | ResNet-34 | 0 | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| Transfuser | C+L | ResNet-34 | 0 | 97.7 | 92.8 | 92.8 | 100 | 79.2 | 84.0 |
| DRAMA | C+L | ResNet-34 | 0 | 98.0 | 93.1 | 94.8 | 100 | 80.1 | 85.5 |
| Hydra-MDP-W-EP | C+L | ResNet-34 | 8192 | 98.3 | 96.0 | 94.6 | 100 | 78.7 | 86.5 |
| **DiffusionDrive** | **C+L** | **ResNet-34** | **20** | **98.2** | **96.2** | **94.7** | **100** | **82.2** | **88.1** |

Surpasses Hydra-MDP-W-EP by +1.6 PDMS despite using 400× fewer anchors and no post-processing. Strongest gains in DAC (+3.1 vs. Transfuser), EP (+3.0), Comfort (100 = perfect).

### nuScenes (Table 7, open-loop)

| Method | Backbone | Avg L2 (m)↓ | Avg Collision (%)↓ | FPS |
|--------|----------|-------------|---------------------|-----|
| VAD | ResNet-50 | 0.72 | 0.22 | 4.5 |
| SparseDrive | ResNet-50 | 0.61 | 0.08 | 9.0 |
| **DiffusionDrive** | **ResNet-50** | **0.57** | **0.08** | **8.2** |

vs. VAD: −20.8% L2, −63.6% collision rate, 1.8× faster.

---

## Ablation Studies

### Decoder Design (Table 3)

| ID | UNet | Spatial XAttn | Agent/Map XAttn | Cascade | Params | PDMS |
|----|------|---------------|-----------------|---------|--------|------|
| 1 | ✓ | ✗ | ✗ | ✗ | 102M | 85.7 |
| 2 | ✗ | ✗ | ✗ | ✗ | 57M | 55.1 |
| 3 | ✗ | ✓ | ✗ | ✗ | 58M | 87.1 |
| 4 | ✗ | ✗ | ✓ | ✗ | 58M | 85.1 |
| 5 | ✗ | ✓ | ✓ | ✗ | 59M | 87.4 |
| **6** | **✗** | **✓** | **✓** | **✓** | **60M** | **88.1** |

- ID-2 without ego-query or any cross-attention: catastrophic failure (55.1 PDMS)
- Spatial cross-attention (BEV/PV) is the most critical component: +32 PDMS alone (ID-2→3)
- Agent/map cross-attention adds +0.3 on top
- Cascade adds +0.7 (ID-5→6), at cost of only +1M params

### Denoising Steps (Table 4)

| Steps | PDMS |
|-------|------|
| 1 | 87.9 |
| **2** | **88.1** |
| 3 | 88.1 |

1 step already achieves strong results. Diminishing returns beyond 2 steps — thanks to truncation from good starting points.

### Cascade Stages (Table 5)

| Stages | PDMS |
|--------|------|
| 1 | 87.4 |
| **2** | **88.1** |
| 4 | 88.2 |

Saturates at 2 stages (chosen for efficiency). 4 stages: marginal +0.1 PDMS at higher parameter cost.

### N_infer: Number of Sampled Trajectories (Table 6)

| $N_\text{infer}$ | PDMS |
|-----------------|------|
| 10 | 84.9 |
| **20** | **88.1** |
| 40 | 88.2 |

More samples = better coverage of action space. 20 is the standard choice; 40 gives marginal gain.

---

## Limitations

1. **Camera + LiDAR required**: no camera-only variant reported; directly disadvantaged vs. camera-only VLMs (WAM-Flow 90.3, Curious-VLA 90.3) that outperform it despite fewer sensors
2. **ResNet-34/50 backbone**: no VLM backbone; the entire VLA literature (Qwen2.5-VL, InternVL3) operates at a different representational level — comparison is across paradigms, not apples-to-apples
3. **No language/reasoning**: purely trajectory generation, no scene understanding, no instruction following, no interpretability
4. **Non-reactive NAVSIM only**: not evaluated in fully interactive or real-world environments
5. **No NAVSIM-v2 / EPDMS**: extended comfort metrics (TL, LK, DDC) not reported; unknown how the system handles lane discipline and traffic light compliance
6. **Comparison scope**: Table 1 compares against Transfuser-era baselines; VLA-era methods (WAM-Flow 90.3, DriveFine 90.7) are later and outside the paper's comparison scope — these supersede 88.1 PDMS
7. **Training anchors**: K-Means on training set; rare or OOV maneuvers not captured by 20 anchors may still suffer (authors claim truncated diffusion handles this better than VADv2's 8192 fixed vocab, but no OOV evaluation is provided)

---

## Key Takeaways

1. **Truncated diffusion = anchor-guided + schedule truncation**: starting from an anchored Gaussian (not pure noise) eliminates mode collapse and reduces denoising steps from 20 to 2 — the two problems of vanilla diffusion for driving are solved together
2. **Cascade decoder > UNet**: the lightweight transformer decoder with BEV/PV + agent/map cross-attention beats the heavier UNet by 2.4 PDMS with 39% fewer parameters — interaction with perception context is the critical ingredient
3. **Inference flexibility**: $N_\text{infer}$ decoupled from $N_\text{anchor}$ — can dynamically scale trajectory hypotheses without retraining
4. **88.1 PDMS with ResNet-34**: competitive against contemporaries at the time of publication; subsequently surpassed by VLA methods (WAM-Flow 90.3, DriveFine 90.7, FLARE 91.4) — DiffusionDrive now serves as the canonical non-VLM diffusion baseline
5. **Mode diversity 74%**: far higher than vanilla diffusion (11%) — enables safe multi-mode planning even at 45 FPS

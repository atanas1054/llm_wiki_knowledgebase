---
title: "DiffusionDriveV2: Reinforcement Learning-Constrained Truncated Diffusion Modeling in End-to-End Autonomous Driving"
type: source-summary
sources: [raw/papers/DiffusionDriveV2_ Reinforcement Learning-Constrained Truncated Diffusion Modeling in End-to-End Autonomous Driving.md]
related: [sources/diffusiondrive.md, sources/recogdrive.md, sources/drivefine.md, sources/flare.md, sources/dreameraD.md, sources/curious-vla.md, sources/nord.md, concepts/rl-for-ad.md, concepts/diffusion-planner.md, concepts/navsim-benchmark.md]
created: 2026-04-16
updated: 2026-04-16
confidence: high
---

## Overview

**DiffusionDriveV2** is a direct successor to DiffusionDrive from HUST and Horizon Robotics (December 2024). It keeps DiffusionDrive's truncated GMM diffusion architecture but replaces the imitation-learning-only training with a carefully designed **Reinforcement Learning** stage that constrains negative modes and explores for superior trajectories.

**arXiv**: 2512.07745  
**Org**: HUST (EIC + AI Institute), Horizon Robotics, Wuhan University  
**Code**: [hustvl/DiffusionDriveV2](https://github.com/hustvl/DiffusionDriveV2)

DiffusionDriveV2 achieves **91.2 PDMS** on NAVSIM v1 and **85.5 EPDMS** on NAVSIM v2 (ResNet-34, Camera+LiDAR) — the highest non-VLM results in the wiki and competitive with VLM-based methods on v1.

---

## Figure 1: The Core Dilemma

![[raw/assets/x1 20.png]]

Three regimes:
- **(a) Vanilla diffusion**: mode collapse — all trajectories converge to a single path
- **(b) DiffusionDrive (IL only)**: diverse trajectories but many are colliding or off-road (red circles) — the negative modes get no supervision
- **(c) DiffusionDriveV2 (IL + RL)**: RL constrains negative modes and pushes all trajectories toward high quality while preserving diversity

---

## Problem: The IL Dilemma in DiffusionDrive

DiffusionDrive's training objective optimizes only the **single positive mode** per scene — the anchor closest to the GT trajectory:

$$\mathcal{L} = \sum_{k=1}^{N_\text{anchor}} \left[y^k \mathcal{L}_\text{rec}(\hat{\tau}^k, \tau_\text{gt}) + \mathcal{L}_\text{BCE}(\hat{s}^k, y^k)\right]$$

where $y^k = 1$ for the closest anchor and $y^k = 0$ for all others. The reconstruction loss only back-propagates through the positive anchor. The 19 negative mode trajectories receive **zero quality constraints**.

**Consequence**: DiffusionDrive's raw output (before selector) has PDMS@10 = 75.3 — 25% of trajectories in the bottom-10 set are poor quality. The downstream selector (a much smaller module) must save the system from these, and fails under OOD conditions.

---

## Figure 2: Overall Architecture

![[raw/assets/x2 18.png]]

DiffusionDriveV2 adds three RL components on top of DiffusionDrive's frozen pre-trained generator:

1. **Multiplicative exploration noise** → perturbs trajectories for RL rollouts
2. **Intra-Anchor GRPO** → within-group advantage estimation per anchor
3. **Inter-Anchor Truncated GRPO** → global safety floor via collision penalty

A coarse-to-fine **mode selector** (Stage 2) replaces DiffusionDrive's simple confidence classifier.

---

## Method

### 4.1 Truncated Diffusion Generator (unchanged from DiffusionDrive)

DiffusionDrive's GMM prior is preserved: $N_\text{anchor}$ = 20 K-Means anchors, each representing a distinct driving intent (straight, overtake, turn left/right, etc.). The trajectory distribution is:

$$p(\tau \mid z) = \sum_{k=1}^{N_\text{anchor}} s(\mathbf{a}^k \mid z)\, p(\tau^k \mid \mathbf{a}^k, z), \quad p(\tau^k \mid \mathbf{a}^k, z) = \mathcal{N}(\tau^k \mid \mathbf{a}^k + \mu^k(z), \Sigma^k(z))$$

Pre-trained DiffusionDrive weights serve as a **cold start** — the model arrives at Stage 1 RL already capable of multi-modal generation.

### 4.2 Diffusion as MDP

Each denoising step is a Gaussian policy:

$$\pi_\theta(\tau_{t-1}^k \mid \tau_t^k, z, \mathbf{a}^k) = \mathcal{N}\!\left(\tau_{t-1}^k;\, \mu_\theta(\tau_t^k, t, z, \mathbf{a}^k),\, \eta(1-\alpha_t)I\right)$$

During training: $\eta = 1$ (DDPM, enables stochastic exploration). During inference: $\eta = 0$ (DDIM, deterministic). Policy gradient via REINFORCE:

$$\nabla_\theta \mathcal{J}(\pi_\theta^k) = \mathbb{E}_{\pi_\theta^k}\!\left[\sum_{t=1}^{T_\text{trunc}} \nabla_\theta \log \pi_\theta^k(\tau_{t-1}^k \mid \tau_t^k)\, A_t^k\right]$$

### 4.3 Scale-Adaptive Multiplicative Exploration Noise

![[raw/assets/x3 18.png]]

**Problem with additive noise**: a normalized trajectory $\tau = \{(x_n, y_n)\}_{n=1}^{N_f}$ has small values near the ego vehicle and large values at the horizon. Applying the same additive Gaussian noise $\epsilon_\text{add}$ to every waypoint destroys near-field structure and produces jagged exploratory paths (Fig 3a).

**Multiplicative noise fix**:
$$\tau' = (1 + \epsilon_\text{mul})\, \tau, \quad \epsilon_\text{mul} = (\epsilon_\text{long}, \epsilon_\text{lat})$$

Only two scalar noises — one longitudinal scaling, one lateral scaling. The resulting explored trajectories remain smooth and structurally coherent (Fig 3b). Scale-adaptive: near-field waypoints are perturbed proportionally less (small absolute change), far-field waypoints more (large absolute change) — matching the natural uncertainty profile of a trajectory.

**Hyperparameter**: minimum std = 0.04 to prevent entropy collapse; minimum log-variance std = 0.10 for training stability.

### 4.4 Intra-Anchor GRPO

**Why naive GRPO fails for GMM diffusion**: standard GRPO estimates advantages across all sampled trajectories in one group. For a GMM with anchors representing distinct intents (turn left vs. go straight), cross-anchor comparison causes the dominant mode to accumulate positive advantages and minority modes to be suppressed — exactly the mode collapse problem that DiffusionDrive was designed to solve.

**Intra-Anchor GRPO**: for each anchor $k$, generate $G$ trajectory variations by applying multiplicative noise. Compute GRPO advantages **only within** this group of $G$ same-intent trajectories:

$$A^{k,i} = \frac{r^{k,i} - \text{mean}(\{r^{k,1}, \ldots, r^{k,G}\})}{\text{std}(\{r^{k,1}, \ldots, r^{k,G}\})}$$

where $r^{k,i} = R(\tau_0^{k,i})$, the reward of the final clean trajectory applied uniformly across all denoising steps (with discount $\gamma_{t-1}$ to downweight early noisy steps).

**RL loss**:

$$L_\text{RL} = -\frac{1}{N_\text{anchor}} \sum_{k=1}^{N_\text{anchor}} \frac{1}{G} \sum_{i=1}^{G} \frac{1}{T_\text{trunc}} \sum_{t=1}^{T_\text{trunc}} \gamma_{t-1} \log \pi_\theta(\tau_{t-1}^{k,i} \mid \tau_t^{k,i})\, A^{k,i}$$

**Combined with IL regularization** to prevent overfitting and preserve general driving capability:

$$L = L_\text{RL} + \lambda L_\text{IL}, \quad \lambda \in (0, 1)$$

### 4.5 Inter-Anchor Truncated GRPO

**Problem with pure intra-anchor isolation**: a dangerous trajectory in anchor A could have a positive local advantage if it outperforms its intra-group peers. Meanwhile a safe but suboptimal trajectory in anchor B gets a negative advantage. The learning signals are locally meaningful but globally inconsistent — a colliding trajectory in one mode may be "rewarded" relative to its anchor group.

**Fix — truncated advantage**:

$$A_\text{trunc}^{k,i} = \begin{cases} -1 & \text{if collision} \\ \max(0, A^{k,i}) & \text{otherwise} \end{cases}$$

This implements the principle **"reward relative improvements, penalize absolute failures"**:
- Negative local advantages are zeroed (don't punish a safe-but-not-great trajectory just because it's below the group mean)
- Collisions receive a universal hard penalty of −1 regardless of their anchor group or local rank
- Only genuine improvements (positive advantage) receive positive gradient signal

$A_\text{trunc}^{k,i}$ replaces $A^{k,i}$ in the RL loss.

### 4.6 Mode Selector (Stage 2)

Coarse-to-fine scorer (inspired by DriveSuprim):
1. **Coarse scorer**: selects top-k candidates via BEV cross-attention + agent/map cross-attention → MLP
2. **Fine scorer**: re-ranks top-k with finer scoring

**Margin-Rank loss** for continuous quality discrimination:

$$\mathcal{L}_\text{rank} = \frac{1}{N} \sum_{i,j} \max\!\left(0, -\text{sign}(s_i - s_j) \cdot (\hat{s}_i - \hat{s}_j) + m\right)$$

Combined BCE + rank loss. Trained with data augmentation: multiplicative noise (std 0.1–0.2) on V2 trajectories + 1% random GTRS vocabulary samples to improve OOD robustness.

---

## Results

### NAVSIM v1 (Table 1)

| Method | Img. Backbone | NC ↑ | DAC ↑ | TTC ↑ | Comf. ↑ | EP ↑ | PDMS ↑ |
|--------|--------------|------|-------|-------|---------|------|--------|
| DRAMA | ResNet-34 | 98.0 | 93.1 | 94.8 | 100 | 80.1 | 85.5 |
| Hydra-MDP* | ResNet-34 | 98.3 | 96.0 | 94.6 | 100 | 78.7 | 86.5 |
| GoalFlow* | ResNet-34 | 98.3 | 93.8 | 94.3 | 100 | 79.8 | 85.7 |
| ARTEMIS | ResNet-34 | 98.3 | 95.1 | 94.3 | 100 | 81.4 | 87.0 |
| DiffusionDrive | ResNet-34 | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| WoTE | ResNet-34 | 98.5 | 96.8 | 94.9 | 99.9 | 81.9 | 88.3 |
| DIVER | ResNet-34 | 98.5 | 96.5 | 94.9 | 100 | 82.6 | 88.3 |
| DriveSuprim | ResNet-34 | 97.8 | 97.3 | 93.6 | 100 | 86.7 | 89.9 |
| Hydra-MDP | V2-99 | 98.4 | 97.8 | 93.9 | 100 | 86.5 | 90.3 |
| GoalFlow | V2-99 | 98.4 | 98.3 | 94.6 | 100 | 85.0 | 90.3 |
| **DiffusionDriveV2 (Ours)** | **ResNet-34** | **98.3** | **97.9** | **94.8** | **99.9** | **87.5** | **91.2** |

Key: +3.1 PDMS over DiffusionDrive; +5.3 EP gain; +2.9 over DIVER (also RL-based). ResNet-34 (21.8M params) beats V2-99 (96.9M params) methods.

### NAVSIM v2 (Table 2)

| Method | NC ↑ | DAC ↑ | DDC ↑ | TL ↑ | EP ↑ | TTC ↑ | LK ↑ | HC ↑ | EC ↑ | EPDMS ↑ |
|--------|------|-------|-------|------|------|-------|------|------|------|---------|
| Ego Status MLP | 93.1 | 77.9 | 92.7 | 99.6 | 86.0 | 91.5 | 89.4 | 98.3 | 85.4 | 64.0 |
| Transfuser | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 |
| Hydra-MDP++ | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | 81.4 |
| DriveSuprim | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 | 83.1 |
| ARTEMIS | 98.3 | 95.1 | 98.6 | 99.8 | 81.5 | 97.4 | 96.5 | 98.3 | — | 83.1 |
| **DiffusionDriveV2 (Ours)** | **97.7** | **96.6** | **99.2** | **99.8** | **88.9** | **97.2** | **96.0** | **97.8** | **91.0** | **85.5** |

EC = 91.0 — very strong extended comfort. EP = 88.9 — best in the table. EPDMS = 85.5.

**Caveat**: the v2 table excludes DriveDreamer-Policy (88.7 EPDMS), DreamerAD (87.7), and Senna-2 (86.6). DiffusionDriveV2 (85.5) is below these in the wiki ranking.

### Diversity and Quality Trade-off (Table 3)

Evaluated on raw model output (before selector) — 20 trajectories generated per scene:

| Method | Div. | PDMS@1 | PDMS@5 | PDMS@10 |
|--------|------|--------|--------|---------|
| Transfuser_TD | 0.1 | 85.7 | 85.7 | 85.7 |
| DiffusionDrive | 42.3 | 93.5 | 84.3 | 75.3 |
| **DiffusionDriveV2** | **30.3** | **94.9** | **91.1** | **84.4** |

Diversity decreases from 42.3 → 30.3 (RL constrains unsafe modes). Both bounds improve:
- **Upper bound** (PDMS@1): 93.5 → 94.9 (+1.4) — RL pushes the best trajectory higher
- **Lower bound** (PDMS@10): 75.3 → 84.4 (+9.1) — RL eliminates most unsafe modes

This directly validates the core claim: RL resolves the diversity–quality dilemma.

---

## Ablation Studies

### Exploration Noise Type (Table 4)

| Noise | PDMS |
|-------|------|
| Additive | 89.7 |
| **Multiplicative** | **90.1** |

### Intra-Anchor GRPO (Table 5)

| Intra-Anchor | PDMS |
|---|------|
| ✗ (cross-anchor comparison) | 89.2 |
| ✓ | **90.1** |

+0.9 PDMS: preventing cross-intent advantage comparison is critical for GMM models.

### Inter-Anchor Truncated GRPO (Table 6)

| Inter-Anchor Trunc. | PDMS |
|---|------|
| ✗ | 89.5 |
| ✓ | **90.1** |

+0.6 PDMS: global collision penalty provides a safety floor that intra-anchor alone cannot.

### Mode Selector Design (Table 8)

| Model | Description | PDMS |
|-------|-------------|------|
| M₀ | Base model | 89.7 |
| M₁ | + Coarse2Fine | 89.9 (+0.2) |
| **M₂** | **+ Rank Loss** | **90.1 (+0.2)** |

### Selector Impact (Table 9)

| Model | Selector | PDMS |
|-------|----------|------|
| DiffusionDrive | ✗ | 88.1 |
| DiffusionDrive | ✓ (V2 selector) | 89.1 |
| **DiffusionDriveV2** | **✓** | **91.2** |

V2 selector alone adds +1.0 to DiffusionDrive — but the full RL training adds +2.1 more. The improvement is **not** merely from a stronger selector.

---

## Training Details

### Stage 1 — RL
- Backbone: ResNet-34 (same as DiffusionDrive, 21.8M params)
- Input: 3 forward-facing cameras (1024×256 concatenated) + rasterized BEV LiDAR
- Cold start: DiffusionDrive pre-trained weights
- Optimizer: AdamW, lr = 2×10⁻⁴, wd = 1×10⁻⁴, cosine schedule + 10% warmup
- 10 epochs, batch 512, 8 × NVIDIA L20 GPUs
- Inference: 2 denoising steps (same as V1)
- Denoising discount γ = 0.8

### Stage 2 — Mode Selector
- 20 epochs, same lr/batch as Stage 1
- Aug: multiplicative noise (std 0.1–0.2) + 1% GTRS vocabulary samples

---

## Key Takeaways

1. **RL resolves the IL supervision gap in GMM diffusion**: imitation learning over a Gaussian Mixture Model can only supervise one mode per scene. RL can constrain all modes simultaneously — raising the quality floor and ceiling together.

2. **Multiplicative noise > additive noise for trajectory RL**: because trajectory waypoints have scale-dependent coordinate magnitudes, additive noise destroys smoothness. Two-scalar multiplicative noise is the right inductive bias for trajectory-space exploration.

3. **Intra-Anchor GRPO is essential for multi-modal models**: applying GRPO naively across anchors representing different driving intentions triggers mode collapse. Advantage estimation must be scoped to same-intent groups.

4. **Inter-Anchor truncation provides a global safety floor**: the "truncate negatives, penalize collisions" rule prevents locally-valid unsafe trajectories from receiving positive gradients in their respective anchor groups.

5. **91.2 PDMS with ResNet-34**: DiffusionDriveV2 is the strongest non-VLM result in the wiki and competitive with (or superior to) several VLM methods (FLARE 91.4, WAM-Flow 90.3, DriveFine 90.7) despite no language backbone. The gap to FLARE (91.4) is 0.2 PDMS — within comparison noise given different training setups.

6. **NAVSIM-v2 EPDMS not SOTA**: 85.5 EPDMS is strong but below DriveDreamer-Policy (88.7), DreamerAD (87.7), and Senna-2 (86.6) on this benchmark. The V2 comparison table omits these methods.

---

## Qualitative Scenarios

### Going Straight (Fig. 4)

![[raw/assets/x4 17.png]]

![[raw/assets/x6 15.png]]

DiffusionDrive: generates colliding/off-road trajectories alongside good ones. DiffusionDriveV2: all candidates are high quality; beam is focused.

### Turning (Fig. 5–6)

![[raw/assets/x7 12.png]]

![[raw/assets/x8 9.png]]

DiffusionDrive: Top-1 is fine but Top-10 shows delayed turning → collision. DiffusionDriveV2: all trajectories collision-free; diversity preserved (conservative car-following vs. aggressive overtaking on curve).

### Multi-Modal Scenarios (Fig. 7)

![[raw/assets/x10 4.png]]

Intersections with multiple valid options (turn vs. straight). Vanilla diffusion: collapses to one option. DiffusionDrive: covers both options but includes unsafe trajectories. DiffusionDriveV2: covers both options with all trajectories being safe.

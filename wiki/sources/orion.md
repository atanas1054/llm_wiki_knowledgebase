---
title: "ORION: A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation"
type: source-summary
sources: [raw/papers/ORION_ A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation.md]
related: [concepts/diffusion-planner.md, concepts/vlm-domain-adaptation.md, concepts/dual-system-vla.md, sources/linkvla.md]
created: 2026-04-05
updated: 2026-04-05
confidence: high
---

**arXiv**: https://arxiv.org/html/2503.19755v1  
**Affiliation**: Huazhong University of Science and Technology + Xiaomi EV  
**Benchmark**: Bench2Drive (CARLA V2), nuScenes open-loop

> **Superseded on Bench2Drive**: LinkVLA ([[sources/linkvla.md]]) subsequently achieves 91.01 DS / 74.55% SR, surpassing ORION's 77.74 DS / 54.62% SR by +13.27 DS and +19.93% SR.

---

## Core Thesis

End-to-end VLM methods fail at closed-loop evaluation because there is a **semantic gap** between the VLM's reasoning space (language tokens) and the action space (numerical trajectory). ORION bridges this gap with a **generative planner** (VAE + GRU decoder) that enforces distribution alignment between a planning token and the target trajectory in a shared Gaussian latent space.

---

## Figure 1 — Four Paradigms of E2E Autonomous Driving

![[raw/assets/ORION_ A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation/x1 7.png]]

| Paradigm | Description | Weakness |
|----------|-------------|----------|
| (a) Classic E2E | Multi-task perception → trajectory | No causal reasoning |
| (b) Text output VLM | VLM predicts trajectory as text tokens | Poor numerical precision; single-mode |
| (c) Dual-system | VLM outputs meta-action → guides E2E | Decoupled; VLM not gradient-connected to trajectory |
| **(d) ORION** | **VLM planning token → generative planner → trajectory** | **Differentiable end-to-end; multi-modal** |

---

## Figure 2 — ORION Full Pipeline

![[raw/assets/ORION_ A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation/x2 6.png]]

Three components: **QT-Former** (vision → reasoning space) → **Vicuna v1.5 LLM** (reasoning) → **Generative Planner** (reasoning → action space).

---

## Architecture

### QT-Former

![[raw/assets/ORION_ A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation/x3 5.png]]

A Q-Former-style module that compresses multi-view images into three query types:

| Query | Dimension | Purpose |
|-------|-----------|---------|
| $Q_s \in \mathbb{R}^{N_s \times C_q}$ | $N_s = 512$ | Scene queries — key information of current frame |
| $Q_p \in \mathbb{R}^{N_p \times C_q}$ | $N_p = 600$ | Perception queries — object detection, traffic state, motion prediction |
| $Q_h \in \mathbb{R}^{N_h \times C_q}$ | $N_h = 16$ | History queries — extract relevant historical information |

**Forward pass**:
1. $Q_s, Q_p$ undergo self-attention → cross-attention with image features $F_m$ (with 3D positional encoding)
2. $Q_p$ heads: object detection ($\mathcal{L}_{cls}$ + $\mathcal{L}_{reg}$), traffic state ($\mathcal{L}_{tra}$), motion prediction ($\mathcal{L}_m$)
3. $Q_h$ attends to memory bank $M$ with timestamp embeddings $P_t$, then to current $Q_s$:

$$Q_h = \text{CA}(Q_h, M+P_t, M+P_t)$$
$$\hat{Q}_h = \text{CA}(Q_h, Q_s, Q_s)$$

4. Updated $\hat{Q}_h$ stored in FIFO memory bank $M \in \mathbb{R}^{(N_h \times n) \times C_q}$, $n=16$ frames
5. Scene tokens $x_s$ and history tokens $x_h$ projected to LLM dimension via 2-layer MLP

**Key design**: history queries first summarize the memory bank, then selectively attend to the *current* scene, extracting the parts of the current frame most related to historical context.

### LLM (Vicuna v1.5 + LoRA)

- LoRA rank=16, alpha=16
- Input: language tokens $x_q$ + scene tokens $x_s$ + history tokens $x_h$
- Tasks: scene description, history review, scene analysis, action reasoning
- Final special **planning token** $s$ accumulates the full reasoning context:

$$s \sim p(s \mid x_s, x_h, x_q, x_a)$$

### Generative Planner (VAE + GRU)

The planning token $s$ and the ground-truth trajectory $t$ are each projected to Gaussian distributions in a shared latent space:

$$p(z_s \mid s) \sim \mathcal{N}(\mu_s, \sigma_s^2), \quad p(z_t \mid t) \sim \mathcal{N}(\mu_t, \sigma_t^2)$$

**Training**: minimize KL divergence to force the two distributions to align:
$$\mathcal{L}_{vae} = D_{KL}(p(z \mid s), \, p(z \mid t))$$

**Inference**: sample $z \sim p(z_s \mid s)$ → GRU decoder (from GenAD) → **6-mode trajectory** (one per navigation command).

Full generative planner loss:
$$\mathcal{L}_{gp} = \mathcal{L}_{vae} + \mathcal{L}_{mse} + \mathcal{L}_{col} + \mathcal{L}_{bd}$$

Total loss:
$$\mathcal{L} = \mathcal{L}_{qt} + \mathcal{L}_{ce} + \mathcal{L}_{gp}$$

**Why VAE over diffusion?** VAE directly aligns distributions in latent space; training is more stable. Diffusion requires a complex conditional denoising process. (See Table 3 below.)

---

## Chat-B2D Dataset

### Figure A1 — Automated Annotation Pipeline

![[raw/assets/ORION_ A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation/x6 3.png]]

ORION introduces **Chat-B2D**, a VQA dataset auto-generated from Bench2Drive using Qwen2-VL-72B:

| Split | VQA pairs |
|-------|-----------|
| Train | 2.11M |
| Val   | 0.12M |

**Four task types**:
1. **Scene description** — weather, time of day, traffic, road characteristics
2. **Critical object behavior** — state and intentions of objects that could affect ego
3. **Meta-driving decisions** — action reasoning (turn left, lane follow, etc.)
4. **Historical information recall** — spatiotemporal dynamics of key elements + ego-motion history

**Annotation pipeline**:
1. Critical object selection: collision risk within 3s, leading vehicles, traffic signals, VRUs
2. Description generation: 6-frame clips + ego status + GT object info → Qwen2-VL-72B
3. History integration: queue mechanism preserving environmental dynamics + ego-motion

---

## Three-Stage Training

| Stage | What is trained | Data | Purpose |
|-------|-----------------|------|---------|
| 1. 3D Vision-Language Alignment | QT-Former + VLM (generative planner frozen) | Chat-B2D VQA | Align vision space with reasoning space |
| 2. Language-Action Alignment | QT-Former + generative planner (LLM via LoRA) | Planning trajectories only | Transmit world knowledge to action space |
| 3. End-to-End Fine-tuning | Full model (LLM via LoRA) | VQA + planning jointly | Final alignment of vision-reasoning-action space |

6 epochs per stage, batch size 32, 32× NVIDIA A800 80GB GPUs. Vision encoder: EVA-02-L (frozen at stage 1 is implied by Q-Former architecture).

---

## Results

### Table 1 — Bench2Drive Closed-Loop + Open-Loop

| Method | DS ↑ | SR (%) ↑ | Efficiency ↑ | Comfortness ↑ | Avg. L2 ↓ |
|--------|-------|----------|-------------|--------------|---------|
| TCP-traj* | 59.90 | 30.00 | 76.54 | 18.08 | 1.70 |
| ThinkTwice* | 62.44 | 31.23 | 69.33 | 16.22 | 0.95 |
| DriveAdapter* (C+L) | 64.22 | 33.08 | 70.22 | 16.01 | 1.01 |
| UniAD-Base | 45.81 | 16.36 | 129.21 | 43.58 | 0.73 |
| VAD | 42.35 | 15.00 | 157.94 | 46.01 | 0.91 |
| MomAD | 44.54 | 16.71 | 170.21 | 48.63 | 0.87 |
| DriveTransformer-Large | 63.46 | 35.01 | 100.64 | 20.78 | 0.62 |
| **ORION (Ours)** | **77.74 (+14.28)** | **54.62 (+19.61)** | **151.48** | **17.38** | **0.68** |

*\* = uses expert feature distillation. ORION uses camera-only + navigation command (no target point).*

### Table 2 — Multi-Ability (%)

| Method | Merging | Overtaking | Emrg. Brake | Give Way | Traffic Sign | Mean |
|--------|---------|-----------|-------------|----------|-------------|------|
| DriveAdapter* | 28.82 | 26.38 | 48.76 | 50.00 | 56.43 | 42.08 |
| DriveTransformer | 17.57 | 35.00 | 48.36 | 40.00 | 52.10 | 38.60 |
| **ORION** | **25.00** | **71.11** | **78.33** | **30.00** | **69.15** | **54.72 (+16.12)** |

ORION excels at interaction requiring **causal reasoning** (Overtaking, Emergency Brake, Traffic Sign). Weakest on lane-changing decisions (Merging 25%, Give Way 30%) — diverse decision timing makes causal relationships harder to capture.

---

## Qualitative Results

### Figure 4 — Closed-Loop Evaluation Scenarios

![[raw/assets/ORION_ A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation/x4 4.png]]

Shows action reasoning + trajectory prediction. Brown = action decision text, Red = objects influencing decision, Green = predicted trajectory. Demonstrates interpretable causal reasoning followed by consistent trajectory.

---

## Ablation Studies

### Figure 5 / Table (Paradigm Comparison)

![[raw/assets/ORION_ A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation/x5 4.png]]

| Paradigm | DS | SR (%) | Mean Ability (%) |
|----------|-----|--------|-----------------|
| (a) Plain text | 42.23 | 13.14 | 15.39 |
| (b) Dual-system (VAD + meta-action) | ~42 | ~15 | ~18 |
| (c) MLP decoder | 70.73 | 45.12 | 48.44 |
| **(d) ORION (VAE + GRU)** | **77.74** | **54.62** | **54.72** |

*Note: dual-system results closely match official VAD numbers, suggesting the bottleneck is the E2E backbone, not the VLM interface.*

### Table 3 — Generative Planner: VAE vs Diffusion

| Planner | DS ↑ | SR (%) ↑ | Avg. L2 ↓ | Avg. Col (%) ↓ | Ability |
|---------|-------|----------|---------|--------------|---------|
| Diffusion (K-means anchors, 20 modes) | 71.97 | 46.54 | 0.73 | 0.96 | 46.68 |
| **VAE (Ours, 6 modes)** | **77.74** | **54.62** | **0.68** | **0.47** | **54.72** |

Even with diffusion ORION beats DriveTransformer by +8.51 DS, +11.53% SR — validating the reasoning-action alignment framework independent of the specific generative model.

### Table 4 — QT-Former Ablation

| ID | Traffic State | Motion Pred. | Memory Bank | Output | DS | SR (%) |
|----|--------------|-------------|------------|--------|-----|--------|
| 1 | — | — | — | Generator | 56.33 | 26.05 |
| 2 | ✓ | — | — | Generator | 74.65 | 49.31 |
| 3 | ✓ | ✓ | — | Generator | 74.07 | 49.77 |
| **4** | **✓** | **✓** | **✓** | **Generator** | **77.74** | **54.62** |
| 5 | — | — | — | Plain text | 25.45 | 10.38 |
| 6 | ✓ | ✓ | ✓ | Plain text | 42.23 | 13.14 |

**Traffic state supervision** is the largest single contributor (+18.32 DS over baseline). Memory bank adds +3.67 DS and +4.85% SR.

### Table 5 — History Query Number Ablation

| N_h | DS | SR (%) | Avg. L2 | Avg. Col |
|-----|-----|--------|---------|---------|
| 0 | 65.10 | 38.83 | 0.67 | 0.61 |
| 8 | 68.09 | 39.09 | 0.66 | 0.62 |
| **16** | **74.10** | **44.66** | **0.68** | **0.55** |
| 32 | 62.46 | 37.73 | 0.65 | 0.73 |

N_h=16 is the sweet spot. N_h=32 degrades — too many history queries overshadow current frame features.

### Table 6 — Joint VQA + Planning Training

| ID | VQA FT | Planning FT | DS | SR (%) | CIDEr | BLEU | ROUGE-L | Avg. L2 |
|----|--------|-------------|-----|--------|-------|------|---------|---------|
| 1 | ✓ | — | — | — | 65.65 | 50.82 | 77.65 | — |
| 2 | — | ✓ | 74.10 | 44.66 | — | — | — | 0.68 |
| **3** | **✓** | **✓** | **77.74** | **54.62** | **65.77** | **52.49** | **77.58** | **0.68** |

Joint training improves **both** tasks simultaneously (+3.64 DS, +9.66% SR vs. planning-only; +0.12 CIDEr, +1.67 BLEU vs. VQA-only).

---

## nuScenes Open-Loop (Table A1)

| Method | VLM | BEV | Planner | Avg. L2 ↓ | Avg. Col ↓ |
|--------|-----|-----|---------|---------|----------|
| UniAD (ego status) | — | ✓ | ✓ | 0.46 | 0.37 |
| OmniDrive++ | ✓ | ✓ | ✓ | 0.33 | 0.30 |
| Senna (ego status) | ✓ | ✓ | ✓ | **0.22** | **0.08** |
| EMMA† | ✓ | — | — | 0.32 | — |
| **ORION** | **✓** | **✓** | **—** | **0.34** | **0.37** |

ORION achieves competitive nuScenes open-loop performance (0.34 avg L2) without using a separate BEV module or ego status as planner input. Does not match Senna or OmniDrive++ which use BEV + ego status + planner input.

---

## Limitations

1. **Real-time latency**: Scalable VLM introduces high computational overhead. Not benchmarked for latency. Authors cite model compression/pruning as future work.
2. **Lane-changing failures**: Weak on Merging (25%) and Give Way (30%) — diverse decision timing confounds causal reasoning.
3. **Simulation-only evaluation**: All closed-loop results on Bench2Drive (CARLA). No NAVSIM or real-world evaluation.
4. **Open-loop gap**: nuScenes open-loop 0.34 L2 is competitive but not SOTA (Senna reaches 0.22 with ego status + BEV).
5. **Memory bank cost**: 16-frame FIFO query bank adds inference overhead as sequence length grows.

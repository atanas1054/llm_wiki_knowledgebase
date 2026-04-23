---
title: "ExploreVLA: Dense World Modeling and Exploration for End-to-End Autonomous Driving"
type: source-summary
sources: [raw/papers/ExploreVLA_ Dense World Modeling and Exploration for End-to-End Autonomous Driving.md]
related: [concepts/navsim-benchmark.md, concepts/rl-for-ad.md, concepts/world-model-for-ad.md, concepts/best-of-n.md, concepts/vlm-domain-adaptation.md]
created: 2026-04-23
updated: 2026-04-23
confidence: high
---

**arXiv**: 2604.02714v1  
**Authors**: Zihao Sheng, Xin Ye, Jingru Luo, Sikai Chen, Liu Ren

## One-Line Summary

ExploreVLA augments a unified understanding-and-generation VLA with future RGB and depth prediction as dense world modeling supervision, then leverages the world model's image prediction **entropy as an intrinsic novelty reward** gated by PDMS safety threshold for GRPO exploration — achieving **90.4 / 93.7 BoN-6 PDMS on NAVSIM-v1** and **88.8 EPDMS on NAVSIM-v2**.

## Problem Statement

Two limitations motivate ExploreVLA:

1. **Imitation learning brittleness**: behavior cloning replicates observed expert trajectories but cannot discover alternative strategies; fails under distribution shift.
2. **Sparse supervisory signals**: text descriptions + trajectory waypoints leave most of the scene's spatial geometry and fine-grained appearance unsupervised — limiting the richness of learned representations.

## Architecture

![[x1 25.png|Figure 1: Training paradigm comparison. (a) Imitation learning — pure cloning. (b) Previous RL — exploration but sparse supervision. (c) ExploreVLA — dense world model supervision + uncertainty-based exploration bonus.]]

**Backbone**: **Show-o** — Phi-1.5 LLM + MAGVIT-v2 image tokenizer.

- MAGVIT-v2 codebook size K = 8192; patch size 16×16
- Omni-attention: **causal** attention for text tokens and ego status embeddings; **full** (bidirectional) attention for image tokens
- MLP trajectory head: extracts hidden states at designated positions → continuous-valued waypoints

**Inputs** (per timestep t):
- Current front-view image + 1 historical image 0.5s earlier → 2 frames total
- Natural language command (task instruction)
- Ego status **s_t** (projected to embedding space via learnable MLP)
- Input resolution: 256 × 448

**Outputs**:
- Future waypoints τ = {p̂_i}_{i=1..N_τ} (4-second horizon, (x, y, θ))
- Future RGB frames {Î_{t+1}, …, Î_{t+F}} (0.5s ahead)
- Future depth maps {D̂_{t+1}, …, D̂_{t+F}}

## Method

![[x2 23.png|Figure 2: Model architecture and training paradigm. Input → joint trajectory + image prediction; two training stages (IL + RL/GRPO).]]

### Dense World Modeling (Stage 1)

**RGB generation loss** — masked token prediction over future RGB frames:

$$\mathcal{L}_{\text{rgb}} = -\sum_{f=1}^{F}\sum_{j} \log p_\theta\!\left(u^{\text{rgb}}_{f,j} \mid \mathbf{u}^{\text{rgb}}_{f,*},\, \mathbf{v},\, \mathbf{e}_s,\, \mathbf{u}\right)$$

**Depth generation loss** — depth conditioned additionally on RGB context (RGB tokens attend to the model before depth is generated):

$$\mathcal{L}_{\text{depth}} = -\sum_{f=1}^{F}\sum_{j} \log p_\theta\!\left(u^{\text{dep}}_{f,j} \mid \mathbf{u}^{\text{dep}}_{f,*},\, \mathbf{u}^{\text{rgb}}_{f,*},\, \mathbf{v},\, \mathbf{e}_s,\, \mathbf{u}\right)$$

**Total MTP loss**: $\mathcal{L}_\text{MTP} = \mathcal{L}_\text{rgb} + \mathcal{L}_\text{depth}$

Depth supervision comes from **Metric3D-ViT-Giant2** pseudo-labels applied offline to navtrain images.

**Complementarity of modalities** (Table 3, NAVSIM-v1):

| RGB Gen. | Depth Gen. | NC | DAC | EP | TTC | Comf. | PDMS |
|----------|------------|-----|-----|-----|-----|-------|------|
| ✗ | ✗ | 98.6 | 94.4 | 80.1 | 94.8 | 100.0 | 86.2 |
| ✓ | ✗ | 98.7 | 96.0 | 81.5 | 95.4 | 99.9 | 87.9 |
| ✗ | ✓ | 98.7 | 95.8 | 81.4 | 95.6 | 99.9 | 87.8 |
| ✓ | ✓ | 98.7 | 96.3 | 82.3 | 95.7 | 99.9 | **88.5** |

RGB (+1.7) and depth (+1.6) independently capture complementary information (visual appearance vs. 3D geometric structure). Joint supervision yields 88.5 — 2.3 PDMS above the no-image baseline.

### Uncertainty-Based Exploration Reward (Stage 2)

**Core insight**: the world model trained on expert demonstrations has low entropy for in-distribution trajectories and high entropy for out-of-distribution (OOD) ones — a natural novelty signal over the *entire* training distribution rather than deviation from a single GT trajectory.

**Token-level entropy** (averaged over all future RGB + depth token positions):

$$\mathcal{H}(\boldsymbol{\tau}_i) = -\frac{1}{|\mathcal{M}|} \sum_{j \in \mathcal{M}} p_j \log p_j$$

where $\mathcal{M}$ is the set of all generated image token positions and $p_j$ is the predicted probability for token $j$ over the MAGVIT-v2 codebook.

**Exploration bonus**: $b_i = f(\mathcal{H}(\boldsymbol{\tau}_i))$ normalized to $[0, 1]$.

**Safety-gated reward**:

$$R_i = \begin{cases} \text{PDMS}_i + \lambda \cdot b_i & \text{if } \text{PDMS}_i > \delta \\ \text{PDMS}_i & \text{otherwise} \end{cases}$$

- Safety threshold δ = **0.9** — only safe trajectories receive the exploration bonus
- Exploration weight λ = **0.5**
- Prevents unsafe OOD trajectories from receiving bonus (novelty ≠ useful)

**GRPO group-relative advantage**:

$$\hat{A}_i = \frac{R_i - \text{mean}(\{R_1,\ldots,R_G\})}{\text{std}(\{R_1,\ldots,R_G\})}$$

Group size G = 8. KL penalty β = 0.01, clip range ε = 0.1.

**Reward component ablation** (Table 4, NAVSIM-v1, from Stage 1 baseline 88.50):

| PDMS Reward | Image Reward | NC | DAC | EP | TTC | Comf. | PDMS |
|-------------|--------------|-----|-----|-----|-----|-------|------|
| ✗ | ✗ | 98.74 | 96.25 | 82.27 | 95.67 | 99.97 | 88.50 |
| ✓ | ✗ | 98.82 | 97.78 | 83.44 | 96.50 | 99.95 | 90.19 |
| ✗ | ✓ | 98.76 | 96.30 | 82.32 | 95.65 | 99.97 | 88.53 |
| ✓ | ✓ | 98.81 | 97.88 | 83.53 | 96.66 | 99.95 | **90.36** |

**Key finding**: image-only reward gives +0.03 PDMS — **the exploration bonus alone cannot guide policy improvement without the safety gate** (PDMS reward). When combined, the image reward contributes +0.17 complementary signal on top of the PDMS reward (+1.69). The exploration bonus discovers diverse strategies; the PDMS reward ensures they are safe and useful.

## Training Strategy

Two stages (4 × H200):

| Stage | Sub-phase | Duration | Objective | Notes |
|-------|-----------|----------|-----------|-------|
| 1a | Pre-training | 10 epochs | Image gen only (L_MTP) | GT actions provided as input; model adapts to driving scenes |
| 1b | SFT | 15 epochs | Actions + image gen | Joint prediction of trajectories and future images |
| 2 | RL (GRPO) | 5 epochs | Safety-gated composite reward | LoRA rank 32; lr 3×10⁻⁶; G=8 |

Optimizer: AdamW; Stage 1 lr = 3×10⁻⁵.

## Exploration Bonus Analysis

![[x3 23.png|Figure 3: Left — exploration bonus vs. L2 error to GT (positive correlation). Right — bonus captures trajectory novelty when L2 fails (position-shifted trajectory ≠ directionally different trajectory).]]

The exploration bonus measures **distributional novelty**, not **proximity to GT**. Two key findings:

1. **Positive correlation with L2 error**: more OOD trajectories tend to have higher world model uncertainty — confirming the entropy is informative.
2. **L2 can be misleading**: a trajectory that follows the expert's direction but is laterally offset has high L2 but low entropy (the world model still predicts a similar future). A trajectory taking a completely different route has low L2 (by coincidence of endpoint position) but high entropy. The exploration bonus correctly identifies the second as more novel.

## Results

### NAVSIM v1 (Table 1 — full comparison)

| Model | Input | NC ↑ | DAC ↑ | EP ↑ | TTC ↑ | Comf. ↑ | PDMS ↑ |
|-------|-------|------|-------|------|-------|---------|--------|
| Ego Status MLP | — | 93.1 | 78.3 | 63.2 | 84.0 | 99.9 | 66.4 |
| TransFuser | — | 97.8 | 92.6 | 78.9 | 92.9 | 99.9 | 83.9 |
| DRAMA | MC+L | 98.2 | 95.2 | 81.3 | 94.2 | 100.0 | 86.9 |
| Hydra-MDP | MC+L | 99.1 | 98.3 | 85.2 | 96.6 | 100.0 | 91.3 |
| Centaur | MC+L | 99.2 | 98.7 | 86.0 | 97.2 | 99.9 | 92.1 |
| DriveSuprim | MC+L | 98.6 | 98.6 | 91.3 | 95.5 | 100.0 | **93.5** |
| FSDrive | SC | 98.2 | 93.8 | 80.1 | 93.3 | 99.9 | 85.1 |
| PWM | SC | 98.9 | 95.8 | 81.5 | 95.9 | 100.0 | 88.1 |
| AutoVLA | MC | 98.4 | 95.6 | 81.9 | 98.0 | 99.9 | 89.1 |
| AutoVLA† (BoN-6) | MC | 99.1 | 97.1 | 87.6 | 97.1 | 99.9 | 92.1 |
| DriveVLA-W0 | SC | 98.7 | 99.1 | 87.6 | 97.1 | 100.0 | 90.2 |
| DriveVLA-W0† (BoN-6) | SC | 99.9 | 97.4 | 88.3 | 97.0 | 99.9 | 93.0 |
| **ExploreVLA** | **SC** | **98.8** | **98.4** | **83.5** | **96.5** | **99.9** | **90.4** |
| **ExploreVLA† (BoN-6)** | **SC** | **99.4** | **98.9** | **88.3** | **98.3** | **99.7** | **93.7** |

ExploreVLA† achieves best TTC (98.3) and 2nd-best NC and DAC in this table. Single-camera (SC), no LiDAR.

**Comparison scope caveat**: Table 1 does not include WAM-Diff (91.0), DriveFine (90.7–91.8), HybridDriveVLA (92.1), FLARE (91.4), or DiffusionDriveV2 (91.2). Among methods with head-to-head comparison, ExploreVLA† (93.7) exceeds DriveVLA-W0† (93.0) by +0.7. ExploreVLA† (93.7) also exceeds DriveSuprim (93.5) in the same table — though this is a BoN result vs. a single-sample result; not a fair single-sample comparison.

### NAVSIM v2 (Table 2 — full comparison)

| Model | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | EPDMS |
|-------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-------|
| Ego Status | 93.1 | 77.9 | 92.7 | 99.6 | 86.0 | 91.5 | 89.4 | 98.3 | 85.4 | 64.0 |
| TransFuser | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 |
| Hydra-MDP++ | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | 81.4 |
| DriveSuprim | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 | 83.1 |
| ARTEMIS | 98.3 | 95.1 | 98.6 | 99.8 | 81.5 | 97.4 | 96.5 | 98.3 | — | 83.1 |
| DiffusionDrive | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | 84.5 |
| DiffusionDriveV2 | 97.7 | 96.6 | 99.2 | 99.8 | 88.9 | 97.2 | 96.0 | 97.8 | 91.0 | 85.5 |
| DriveVLA-W0 | 98.5 | 99.1 | 98.0 | 99.7 | 86.4 | 98.1 | 93.2 | 97.9 | 58.9 | 86.1 |
| **ExploreVLA** | **98.8** | **96.2** | **99.6** | **99.8** | **87.1** | **98.2** | **97.8** | **98.3** | **86.8** | **88.8** |

ExploreVLA achieves best scores on 6/9 sub-metrics: DDC, TLC (tied), TTC, LK, HC (tied), and the overall EPDMS. Notable: EC = 86.8 — stronger than most non-FLARE methods, suggesting that safety-gated exploration rewards improve comfort.

**Comparison scope caveat**: Table 2 omits WAM-Diff (89.7 EPDMS), DriveDreamer-Policy (88.7 EPDMS), and DreamerAD (87.7 EPDMS). ExploreVLA (88.8) exceeds DDP (88.7) in absolute terms but without a direct head-to-head. WAM-Diff (89.7) remains the single-sample NAVSIM-v2 SOTA in the wiki. ExploreVLA is **2nd in wiki** on EPDMS (after WAM-Diff 89.7).

### nuScenes (Stage 1 only — no RL post-training; Table 5)

Stage 2 GRPO not applied to nuScenes (no closed-loop reward signal available).

**ST-P3 protocol**:

| Method | L2 1s | L2 2s | L2 3s | L2 avg | CR 1s | CR 2s | CR 3s | CR avg |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| VAD | 0.17 | 0.34 | 0.60 | 0.37 | 0.07 | 0.10 | 0.24 | 0.14 |
| UniAD | 0.44 | 0.67 | 0.96 | 0.69 | 0.04 | 0.08 | 0.23 | 0.12 |
| EMMA | 0.14 | 0.29 | 0.54 | 0.32 | — | — | — | — |
| OpenDriveVLA | 0.14 | 0.30 | 0.55 | 0.33 | 0.02 | 0.07 | 0.22 | **0.10** |
| AutoVLA | 0.21 | 0.38 | 0.60 | 0.40 | 0.13 | 0.18 | 0.28 | 0.20 |
| **ExploreVLA** | **0.28** | **0.40** | **0.65** | **0.44** | **0.01** | **0.05** | **0.25** | **0.10** |

ExploreVLA ties OpenDriveVLA for the best average collision rate (0.10%) and achieves the best 1s and 2s collision rates — demonstrating strong short-term safety-aware planning. L2 (0.44m avg) is competitive but not best.

**UniAD protocol**: L2 avg 0.77m (reasonable but below OpenDriveVLA 0.67m and AutoVLA 0.70m). Collision rate not reported in this protocol.

## Qualitative Results

![[x4 21.png|Figure 4: BEV trajectory comparison before (Stage 1) and after (Stage 2 RL) in three challenging scenarios. Stage 1 fails with safety-critical errors (collision with vehicle, near-pedestrian pass, stop sign violation). Stage 2 resolves all three. Green = GT, orange = prediction.]]

![[x5 21.png|Figure 5: Additional qualitative results — Going Straight, Turning, Intersection scenarios. Trajectories follow road topology; intersection scenarios handled with reasonable multi-lane awareness.]]

![[x6 18.png|Figure 6: Dense world modeling qualitative results. Current frame → predicted future RGB and depth maps. Global scene geometry and layout preserved; fine-grained textures (traffic lights, tree branches) show some degradation.]]

## Key Contributions

1. **Intrinsic exploration reward from world model entropy**: first AD paper to use image prediction token entropy as a novelty measure for GRPO reward shaping. Unlike L2 to GT (which can be misleading; Figure 3), entropy measures distance to the *entire* training distribution.
2. **Safety gating**: the δ=0.9 threshold ensures exploration bonuses flow only to safe trajectories — preventing reward hacking where novel trajectories are unsafe.
3. **RGB + depth joint supervision**: depth as a complementary geometric supervision signal alongside appearance supervision; pseudo-label depth (Metric3D) enables training without LiDAR.
4. **Two-role world model**: same model simultaneously acts as (a) a dense supervisory signal during SFT and (b) a reward source during RL.

## Key Limitations

1. **Single front-view camera**: no surround-view inputs; spatial coverage limited.
2. **No GRPO on nuScenes**: Stage 2 RL confined to NAVSIM; closed-loop reward not available for nuScenes.
3. **Comparison scope in both tables**: Table 1 excludes major NAVSIM-v1 competitors (WAM-Diff, FLARE, DiffusionDriveV2); Table 2 excludes WAM-Diff, DDP, DreamerAD — ExploreVLA's SOTA claims are overstated.
4. **BoN vs. single-sample conflation**: BoN-6 93.7 surpassing DriveSuprim 93.5 is highlighted, but these are different inference regimes.
5. **Future work (authors' own)**: multi-view extension; BEV layout as additional generation target.

## Relationship to Wiki Methods

| Method | World Model Role | RL Reward Source | Notes |
|--------|-----------------|-----------------|-------|
| DreamerAD | Latent RL rollout | Learned latent AD-RM | Reward from latent features, no simulator at RL time |
| FLARE | Auxiliary semantic prediction (SFT) | NAVSIM PDM simulator | No generation at inference |
| DriveVLA-W0 | Training-time pixel prediction | NAVSIM PDM (word models bypass at inference) | Training-time only |
| DriveDreamer-Policy | Causal depth→video→action (SFT+inference) | NAVSIM PDM (paper applies RL separately) | Geometry-grounded |
| **ExploreVLA** | **Dense RGB+depth masked prediction (SFT) + entropy as exploration reward (RL)** | **Safety-gated PDMS + image entropy bonus** | **Same model serves both roles** |

ExploreVLA is the only method in the wiki where the world model's **uncertainty** (not its predictions) is used as a reward signal. DreamerAD uses learned *reward features* from latent states; ExploreVLA uses raw *entropy* of token probability distributions.

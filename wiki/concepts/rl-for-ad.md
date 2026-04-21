---
title: Reinforcement Learning for Autonomous Driving
type: concept
sources: [raw/papers/ReCogDrive_ A Reinforced Cognitive Framework for End-to-End Autonomous Driving.md, raw/papers/WAM-Flow_ Parallel Coarse-to-Fine Motion Planning via Discrete Flow Matching for Autonomous Driving.md, raw/papers/Senna-2_ Aligning VLM and End-to-End Driving Policy for Consistent Decision Making and Planning.md, raw/papers/Reasoning-VLA_ A Fast and General Vision-Language-Action Reasoning Model for Autonomous Driving.md, raw/papers/DriveFine_ Refining-Augmented Masked Diffusion VLA for Precise and Robust Driving.md, raw/papers/Devil is in Narrow Policy_ Unleashing Exploration in Driving VLA Models.md, raw/papers/AutoVLA_ A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning.md, raw/papers/AutoDrive-R²_ Incentivizing Reasoning and Self-Reflection Capacity for VLA Model in Autonomous Driving.md, raw/papers/Alpamayo-R1_ Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail.md, raw/papers/AdaThinkDrive_ Adaptive Thinking via Reinforcement Learning for Autonomous Driving.md, raw/papers/FLARE_ Learning Future-Aware Latent Representations from Vision-Language Models for Autonomous Driving.md, raw/papers/DreamerAD_ Efficient Reinforcement Learning via Latent World Model for Autonomous Driving.md, raw/papers/NoRD_ A Data-Efficient Vision-Language-Action Model that Drives without Reasoning.md, raw/papers/DiffusionDriveV2_ Reinforcement Learning-Constrained Truncated Diffusion Modeling in End-to-End Autonomous Driving.md, raw/papers/WAM-Diff_ A Masked Diffusion VLA Framework with MoE and Online Reinforcement Learning for Autonomous Driving.md]
related: [sources/recogdrive.md, sources/wam-flow.md, sources/senna2.md, sources/reasoning-vla.md, sources/drivefine.md, sources/curious-vla.md, sources/autovla.md, sources/autodrive-r2.md, sources/alpamayo-r1.md, sources/adathinkdrive.md, sources/flare.md, sources/dreameraD.md, sources/nord.md, sources/diffusiondrive-v2.md, sources/wam-diff.md, concepts/diffusion-planner.md, concepts/discrete-flow-matching.md, concepts/dual-system-vla.md, concepts/navsim-benchmark.md, concepts/vlm-domain-adaptation.md, concepts/world-model-for-ad.md]
created: 2026-04-05
updated: 2026-04-21
confidence: high
---

## Why RL for Autonomous Driving?

Imitation learning (behavior cloning) has a fundamental weakness: when training data contains **multi-modal expert trajectories** (e.g., multiple valid ways through an intersection), the model learns an **averaged trajectory** that may be unsafe or physically invalid.

RL allows the model to **explore** driving behaviors in a simulated environment and receive reward signals, enabling it to commit to one safe, optimal path rather than averaging across conflicting demonstrations.

## Approaches in the Literature

### Control-space RL (classic)
- Learn policies directly over throttle/brake/steering
- Train in non-photorealistic simulators like CARLA
- Limitation: sim-to-real gap

### RAD
- Trains end-to-end AD agent via RL in photorealistic 3DGS environment
- Addresses simulator realism gap

### CarPlanner
- Auto-regressive planner with RL policy for multi-modal trajectory generation
- Exceeds IL methods on nuPlan

### AlphaDrive
- First system to integrate GRPO-based RL with planning reasoning for AD
- Inspired by DeepSeek-R1; improves performance and training efficiency

### TrajHF
- Human feedback-driven fine-tuning for generative trajectory models
- Aligns with diverse human driving preferences

### ReCogDrive (Simulator-Assisted RL)
- RL applied to a **diffusion planner** (novel application)
- Uses NAVSIM **non-reactive simulator** — avoids difficulty of building fully interactive sim
- Reward: PDMS score (collision avoidance, drivable area, comfort, progress)

## ReCogDrive RL Details

**Treating diffusion as MDP:**
The denoising chain $(x_T, \ldots, x_0)$ is an internal MDP where each denoising step is a Gaussian policy action. This allows policy gradient to flow through the diffusion process.

**Group-normalized advantage (GRPO-style):**
$$\hat{A}_i = \frac{r_i - \text{mean}(r_{1..G})}{\sqrt{\text{var}(r_{1..G})}}$$
G trajectories sampled per scene, evaluated in NAVSIM, normalized within the group.

**Combined loss:**
$$L = L_{RL} - \lambda L_{BC}$$
- $L_{RL}$: discounted policy gradient (discount γ=0.6 downweights early noisy denoising steps)
- $L_{BC}$: behavior cloning loss against reference policy (λ=0.01) prevents exploration collapse

**Ablation impact:** RL stage alone contributes +2.8 PDMS (largest single improvement in ReCogDrive).

## Key Tension: Non-Reactive vs. Interactive Simulators

NAVSIM's non-reactive simulator does not model other agents responding to ego actions. This simplifies RL but may limit generalization to scenarios requiring interaction. A fully interactive closed-loop simulator is much harder to build and maintain.

## WAM-Flow: GRPO Applied to Discrete Flow Matching

**WAM-Flow** ([[sources/wam-flow.md]]) applies GRPO to a **discrete flow matching** model — a different and cleaner integration than ReCogDrive's diffusion-chain MDP approach.

**Key difference:** In ReCogDrive, RL must treat the diffusion denoising chain $(x_T, \ldots, x_0)$ as an MDP, requiring a discounted policy gradient through many steps. In WAM-Flow, parallel DFM generation means the entire trajectory is produced in a small number of steps (1–5), so GRPO can be applied to the **final output tokens** directly without special treatment.

**WAM-Flow reward design:**
$$R(\tau) = \underbrace{\prod_{m \in \{NC, DAC\}} s_m(\tau)}_\text{safety gates} \cdot \underbrace{\frac{\sum_{w \in \{EP, TTC, C\}} \lambda_w s_w(\tau)}{\sum \lambda_w}}_\text{performance objectives}$$

- Safety: NC (no at-fault collision: 0/0.5/1), DAC (drivable area: 0/1) — multiplicative gates
- Performance: EP:TTC:Comfort weighted 5:5:2
- Optimal group size G=3 (G=2 too little diversity; G=4 too much variance)

**GRPO impact in WAM-Flow:** +3.6 PDMS (86.7 → 90.3), comparable to ReCogDrive's +2.8 PDMS contribution from RL.

## Senna-2: Hierarchical RL in 3DGS Environments

**Senna-2** ([[sources/senna2.md]]) introduces **Hierarchical RL (HRL)** in **photorealistic 3DGS environments** — a qualitatively different approach from NAVSIM-based GRPO.

**Environment**: 1,300 high-risk driving clips reconstructed as 3D Gaussian Splatting scenes. Unlike NAVSIM's non-reactive simulator, agents in 3DGS **react to ego actions**, and the VLM sees photorealistic imagery (not pre-rendered video playback).

**Bottom-up hierarchical optimization** — two levels optimized in sequence:

**(i) Low-level E2E planner** — safety + efficiency rewards via longitudinal penalties (not policy gradient):
$$\mathcal{L}_{safe} = \mathbb{E}_{\tau \sim \pi_\theta} \sum_t \mathbf{1}_{[TTC < 3s]} ||\tau_{t+1} - \text{sg}(\tau_t)||_2^2$$
$$\mathcal{L}_{eff} = \mathbb{E}_{\tau \sim \pi_\theta} \sum_t \mathbf{1}_{[v < \delta_v]} ||\tau_t - \text{sg}(\tau_{t+1})||_2^2$$
- Safety: pushes trajectory point forward when TTC dangerous (longitudinal extension away from obstacle)
- Efficiency: pushes trajectory point forward when too slow (longitudinal extension toward destination)

**(ii) High-level VLM** — aligned to the optimized trajectory via kinematic mapping:
$$\mathcal{L}_{high} = -\log P(f_K(\tau) | Q)$$
Only inconsistent cases (VLM decision ≠ trajectory's kinematic category) are penalized.

**Key contrast with NAVSIM GRPO** (ReCogDrive, WAM-Flow):

| Aspect | NAVSIM GRPO | Senna-2 HRL |
|--------|-------------|------------|
| Simulator | Non-reactive pre-rendered | Interactive 3DGS (photorealistic) |
| Agent reactions | No | Yes |
| Optimization | Group policy gradient | Longitudinal penalties (non-PG) |
| What's updated | Full planner | E2E planner first, then VLM |
| Hierarchy | Flat | Two-level (low then high) |
| VLM updated? | No (frozen in ReCogDrive) | Yes (via kinematic mapping) |

**Ablation impact**: Stage 3 HRL contributes −34.7% AF-CR reduction on top of Stage 2 open-loop alignment.

## Reasoning-VLA: Physics-Aware GT-Based GRPO

**Reasoning-VLA** ([[sources/reasoning-vla.md]]) applies GRPO with a qualitatively different reward design: instead of a closed-loop simulator, rewards are computed **purely from the predicted trajectory** against GT and kinematic constraints.

**Three reward components**:

| Reward | Formula | Constraint |
|--------|---------|-----------|
| $r_\text{traj}$ | $1 - \frac{1}{N}\sum \gamma^i(\alpha\Delta x^2 + \beta\Delta y^2)$ | Weighted Euclidean to GT, time-discounted |
| $r_\text{steer}$ | Binary: $|\Delta y / \Delta x| < 0.84$ per step | Max steering angle ≤ 40° |
| $r_\text{acc}$ | Binary: $|acc_j| < 6$ per step | Max acceleration ≤ 0.6g |
| $r_\text{total}$ | $\theta_1 r_\text{traj} + \theta_2 r_\text{steer} + \theta_3 r_\text{acc}$ | Combined |

**Key contrast with NAVSIM GRPO**:

| Aspect | NAVSIM GRPO (ReCogDrive, WAM-Flow) | GT-Based GRPO (Reasoning-VLA) |
|--------|-------------------------------------|-------------------------------|
| Environment | Non-reactive NAVSIM simulator | None (trajectory-only evaluation) |
| Agent interaction | Via collision/TTC rewards | Not modeled |
| Data compatibility | NAVSIM-specific | Any labeled trajectory dataset |
| Reward signal | Closed-loop safety (NC, DAC, TTC) | Open-loop accuracy + kinematics |
| Generalization | Single dataset | 8 datasets simultaneously |

**Advantage**: GT-based rewards can be computed for any dataset with trajectory labels, enabling RL over a diverse 8-dataset corpus. **Limitation**: cannot detect collision with other agents or drivable-area violations not captured by kinematic constraints.

**RL contribution**: ablations show +0.03 avg L2 improvement over SFT alone, consistent across all model variants. Smaller absolute gain than NAVSIM GRPO (+2.8 PDMS in ReCogDrive) but provides kinematic feasibility guarantees across diverse environments.

## DriveFine: Hybrid Offline+Online RL for Refinement

**DriveFine** ([[sources/drivefine.md]]) introduces two RL contributions: an empirical finding about reward hacking in diffusion planners, and a novel hybrid RL strategy for a dedicated refinement expert.

### Reward Hacking in Diffusion-Based Planners

**Key finding**: when PDMS-oriented RFT is applied:
- **Token-based VLAs** (InternVL backbone): PDMS ↑, EPDMS ↑ — both metrics improve
- **Diffusion-based planners** (ReCogDrive, DiffusionDrive): PDMS ↑, **EPDMS ↓** — reward hacking

Root cause: the separately-coupled diffusion planner is weakly bound to the VLM's pretrained weights. GRPO optimizes the planner's trajectory generation while causing it to lose generalizable patterns, degrading extended metrics (lane keeping, extended comfort, driving direction compliance). Token-based VLAs avoid this because language and action tokens share the same embedding space — RL fine-tuning updates representations that are simultaneously used for language understanding, preventing narrow specialization.

**Implication**: for multi-metric robustness, token-based VLAs are more GRPO-stable than weakly-coupled diffusion planners.

### GRPO for Generation Expert (masked dLLM)

Standard GRPO adapted for masked diffusion: G=10 candidate trajectories, $s$-step progressive sampling, $\tau$-step aggregation:
$$\hat{A}_i = r_i - \text{mean}(\{r_i\}_{i=1}^G)$$
Reward from NAVSIM simulator. KL penalty $\beta$ against reference policy.

### Hybrid Offline+Online RL for Refinement Expert

The block-MoE refinement expert requires its own RL strategy, since standard GRPO provides a fixed upper bound (the best generated trajectory). DriveFine introduces a hybrid approach:

**Offline advantage** — pairwise reward differences over GRPO samples (no extra sampling):
$$\hat{A}^{of}_{ij} = r_i - r_j, \quad \forall i,j \in G$$
- Zero mean by construction (simultaneously encourages improvements and penalizes degradations)
- Denser signal than group-average baseline
- Requires no additional rollouts

**Online advantage** — K=6 active refinements per anchor trajectory:
$$\hat{A}^{on}_{ik} = \hat{r}_{ik} - r_i, \quad \forall i \in G, k \in K$$
- Breaks the offline upper bound by allowing the refiner to discover better trajectories not in the GRPO sample set
- Provides exploration signal beyond what was generated by the generation expert

Both computed and combined in a hybrid loss, with generation and refinement experts trained synchronously.

**RL impact on DriveFine** (ablation):

| Config | PDMS |
|--------|------|
| SFT only | 86.7 |
| +GRPO (generation) | 89.6 (+2.9) |
| +Offline-RFT (refinement) | 90.3 (+0.7) |
| **+Online-RFT (refinement)** | **90.8 (+0.5)** |

## Curious-VLA: The Narrow Policy Problem and Diversity-Aware RL

**Curious-VLA** ([[sources/curious-vla.md]]) identifies a fundamental systemic flaw in the standard IL→RL pipeline that all prior methods implicitly suffer from: **Narrow Policy (NP)**.

### The Narrow Policy Problem

IL (SFT) over-exploits GT trajectories, collapsing policy diversity and starving RL of useful feedback. Three root causes:

**(1) Optimization Objective Mismatch** — Cross-entropy loss treats all non-GT tokens as equally wrong, with no spatial proximity signal. Gradient:
$$\frac{\partial\mathcal{L}_{\text{SFT}}}{\partial z_{t}}=\pi_{\theta}(y_{k}|\mathbf{y}^{*}_{<t},\mathcal{X})-\mathbb{I}(\hat{y}_{t}=y_{t}^{*})$$
This encourages overconfidence in $\mathbf{y}^*$, collapsing the policy distribution around one mode.

**(2) Horizon Physical Scale Mismatch** — Future waypoints in ego-centric coordinates have orders-of-magnitude larger variance at $t=4s$ vs. $t=0.5s$. Far-horizon losses dominate, suppressing near-horizon steering diversity.

**(3) Advantage Collapse in RL** — When IL produces a narrow policy, GRPO samples are nearly identical → rewards nearly identical → $\sigma_R \to 0$ → advantages vanish:
$$\lim_{\sigma_{R}\to 0}A_{i}=\frac{R(\mathbf{y}_{i})-\mu_{R}}{\sigma_{R}+\xi}\to 0$$
Result: vanishing gradients and premature RL saturation. Empirically confirmed: both QwenVL-2.5 and ReCogDrive show severely collapsed diversity (pADE = 0.090–0.148) after IL.

### Behavioral Diagnostics

A framework for quantifying NP, evaluated at $k=8$ samples per scenario:

| Metric | Measures | Desired |
|--------|---------|---------|
| Diversity | Mean pairwise ADE/FDE across all sampled trajectories | ↑ high |
| Quality | Min ADE/FDE to GT across $k$ samples (best feasible) | ↓ low |
| Performance | Mean PDMS@$k$ | ↑ high |

Collapse signature: low Diversity + stagnant Quality = NP bottleneck confirmed.

### Feasible Trajectory Expansion (FTE) — IL Stage Fix

Three sub-components that address NP in IL:

1. **Exploratory Data Expansion (DE)**: 12k challenging segments → ReCogDrive diffusion (perturbed latents) → 142k diverse samples (within-intent + across-intent). Safety-filtered by PDMS scorer.

2. **Chain-of-Thought (CoT)**: 4-stage reasoning (perception → explanation → meta-behavior → trajectory), synthesized by Qwen2.5-VL-72B. Two-stage SFT: (i) align LLM to CoT format (vision frozen), (ii) full end-to-end fine-tuning.

3. **Step-wise Normalization (SN)**: normalize each prediction step independently:
$$\tilde{w}_{t}=\frac{w_{t}-\mu_{t}}{\sigma_{t}}, \quad \hat{w}_{t}=\hat{\tilde{w}}_{t}\sigma_{t}+\mu_{t}$$
Equalizes gradient magnitudes across horizons. **Critical finding**: DE alone hurts (85.2 vs. 85.6 CoT-only baseline); SN is the necessary catalyst (DE+CoT+SN = 87.6 best SFT).

### Adaptive Diversity-Aware Sampling (ADAS) — RL Stage Fix

Filters training scenarios to ensure sufficient reward variance for GRPO. Uses a Bernoulli approximation: each rollout is a binary trial with success probability $\hat{p} = \mu_R / R_{max}$.

**Inclusion conditions** (both must hold):
$$\hat{p}^{G}+(1-\hat{p})^{G}<\epsilon_{\text{div}}$$
$$|\sigma_{R}-\sqrt{\hat{p}(1-\hat{p})}R_{\text{range}}|<\epsilon_{\text{conf}}$$

Condition 1: ensures not all $G$ rollouts are identical. Condition 2: validates Bernoulli consistency (filters noisy scenarios).

**Critical negative result**: difficulty-based sampling causes **training collapse** (35.2 PDMS) — harder scenarios are not what's needed; *diverse-outcome* scenarios are.

| Strategy | PDMS |
|----------|------|
| Human Difficulty | 35.2 (collapse) |
| Reject Unimodal | 88.8 |
| ADAS (1x) | 89.6 |
| ADAS (3x) | 90.1 |
| ADAS (3x) + SDR | **90.3** |

### Spanning Driving Reward (SDR)

Focal-loss style reward transformation to amplify sensitivity near the quality boundary:
$$R_{\text{span}}=\prod_{c\in C}c\cdot\frac{\sum_{m\in M}w^{\prime}_{m}\cdot(1-(1-m)^{\gamma_{m}})}{\sum_{m\in M}w^{\prime}_{m}}$$

The focal term $(1-(1-m)^{\gamma_m})$ compresses ceiling effects at high scores and amplifies gradients at lower scores — improves RL sensitivity to incremental driving quality improvements.

**ADAS (3x) + SDR = 90.3 PDMS** (Qwen2.5-VL-3B, 1xC) and **BoN-6 = 94.8** matching human GT — evidence that FTE+DARL successfully unlocks the exploration potential.

## AutoDrive-R²: Physics-Grounded Reward with Four Kinematic Components

**AutoDrive-R²** ([[sources/autodrive-r2.md]]) extends the GT-based GRPO approach (cf. Reasoning-VLA) with a richer four-component physics reward and confirms the SFT-cold-start necessity through a DeepSeek-R1-Zero-style ablation.

### Physics-Grounded Reward

$$r_{acc} = \lambda_{pos} r_{pos} + \lambda_{ste} r_{ste} + \lambda_{vel} r_{vel} + \lambda_{tem} r_{tem}, \quad r_i = r_{acc}^i + r_{format}^i$$

All λ=1 in experiments.

| Component | Formula | Physics constraint |
|-----------|---------|-------------------|
| $r_{pos}$ | $\frac{1}{N}\sum_i (x^i - x_{gt}^i)^2 + (y^i - y_{gt}^i)^2$ | Spatial path adherence |
| $r_{ste}$ | $\frac{1}{N}\sum_j (\theta^j - \theta_{gt}^j)^2$ | Steering kinematics (no abrupt turns) |
| $r_{vel}$ | $\frac{1}{N}\sum_k (v^k - v_{gt}^k)^2$ | Velocity compliance |
| $r_{tem}$ | $\frac{1}{N}\sum_j (\theta^j - \theta^{j-1})^2 + \sum_k (v^k - v^{k-1})^2$ | Temporal smoothness (suppress oscillations) |

**Contrast with Reasoning-VLA**: Reasoning-VLA uses binary pass/fail constraints (steering angle ≤ 40°, acceleration ≤ 0.6g). AutoDrive-R²'s components are continuous MSE terms against GT — providing denser gradient signal at the cost of requiring GT steering and velocity labels.

**Reward ablation** (nuScenes avg L2):
| Removed component | L2 | Δ |
|------------------|----|---|
| w/o $r_{pos}$ | 0.53 | **+179% (near-collapse)** |
| w/o $r_{tem}$ | 0.24 | +26.3% |
| w/o $r_{vel}$ | 0.22 | +15.8% |
| w/o $r_{ste}$ | 0.21 | +10.5% |
| Full | **0.19** | — |

Spatial alignment is indispensable; temporal smoothness is next most critical.

### SFT Cold Start Necessity (empirical confirmation)

Inspired by DeepSeek-R1-Zero, the authors attempted RL-only training — directly applying GRPO without SFT:

| Config | Avg L2 |
|--------|--------|
| RL only (no SFT) | 0.33 |
| SFT only | 0.27 |
| **SFT + RL** | **0.19** |

RL alone underperforms SFT alone by 22.2%. Root cause: RL cannot explore the high-dimensional reasoning space for multi-step kinematic calculation and contextual logic without a structured initialization. This independently corroborates Curious-VLA's finding (DE without SN hurts; diversity without structure is harmful) from a different angle.

### Comparison of GT-Based GRPO Reward Designs

| Method | Reward type | Kinematic components | Dense/binary |
|--------|------------|---------------------|-------------|
| Reasoning-VLA | Trajectory accuracy + binary constraints | Steering ≤40°, acc ≤0.6g | Binary constraints |
| **AutoDrive-R²** | **4-component MSE** | **$r_{pos}$, $r_{ste}$, $r_{vel}$, $r_{tem}$** | **Dense (continuous MSE)** |
| AutoVLA | PDMS or ADE + CoT length penalty | None explicit | Simulator/ADE |

## AutoVLA: Adaptive Reasoning via CoT Length Penalty

**AutoVLA** ([[sources/autovla.md]]) introduces a novel GRPO reward design that trains a single model to *choose* when to reason — without a separate fast/slow architecture.

**Reward function**:
$$r = r_{\text{Driving}} - \lambda_r \cdot r_{\text{CoT}}$$

where $r_{\text{CoT}}$ is a CoT **length penalty** — explicitly discourages unnecessarily long reasoning chains. $r_{\text{Driving}}$ = PDMS (nuPlan/NAVSIM) or ADE (Waymo E2E).

**Effect**: in straightforward scenarios (clear road, simple turn), the model learns to emit a short "reasoning not needed" prefix and proceed directly to action tokens. In complex scenarios (intersections, construction zones, occluded agents), the model emits full 4-stage CoT before the action. This switching behavior is learned via RL — the SFT stage initializes dual-mode capability but the *adaptive selection* comes from RFT.

**Results**:
- +10.6% PDMS improvement over SFT (NAVSIM)
- −66.8% average runtime reduction (500 test scenarios)
- Group size ablation: larger G → better convergence via broader exploration

The CoT length penalty trains adaptive fast/slow selection implicitly — the model learns to emit shorter CoT only when the length penalty outweighs the PDMS gain. For the full cross-wiki GRPO reward comparison, see the table in the Alpamayo-R1 section below.

## AdaThinkDrive: Adaptive Think Reward via Mode-Comparison GRPO

**AdaThinkDrive** ([[sources/adathinkdrive.md]]) provides the clearest empirical evidence in the wiki that CoT is **harmful** in simple scenarios, and introduces an **Adaptive Think Reward** that teaches the model when to reason by comparing Think vs. Non-Think rollout quality within the same scene.

### Empirical Motivation

InternVL3-8B/2B study on 3 complexity levels shows:
- **Level 1 (simple)**: Non-Think PDMS > Think PDMS — over-reasoning hurts
- **Levels 2–3 (challenging)**: Think PDMS > Non-Think PDMS — reasoning helps

The optimal reasoning mode is complexity-dependent. Always-Think and Never-Think are both suboptimal.

### Four-Component GRPO Reward

| Component | Signal | Design |
|---|---|---|
| $\mathcal{R}_{traj}$ | PDMS from NAVSIM | Discrete 0–1 |
| $\mathcal{R}_{fmt}$ | Tag format compliance | Discrete violation penalty |
| $\mathcal{R}_{endpoint}$ | L1 to GT endpoint | Piecewise: 1.0 (<2m), 0.8 (<4m), 0.6 (<6m), 0.4 (<10m), 0.2 (<15m), 0.0 |
| $\mathcal{R}_{adaptive}$ | **Mode-switching via rollout comparison** | Dynamic (see below) |

### Adaptive Think Reward (Algorithm 1) — Dynamic Relabeling

For each scene with complexity label $D$ and rollout group split into Think/Non-Think sub-groups:

**Case D=0 (Simple)**: If Think rollouts outperform Non-Think ($S_{Think} > S_{NoThink}$) AND $S_{Think} > T=0.9$ AND $C_{Think} > C_{NoThink}$:
→ Scene relabeled Challenging → reward Think; otherwise reward Non-Think.

**Case D=1 (Challenging)**: Mirror: if Non-Think wins with high confidence → relabel Simple → reward Non-Think; else reward Think.

**Key property**: initial scene labels from SFT (based on boundary proximity + critical object presence) are anchors only. The reward corrects stale labels based on the *current* policy's rollout behavior. As the policy improves, a formerly-challenging scene may no longer need explicit reasoning — the reward adapts dynamically.

**Threshold T=0.9**: a strict confidence requirement — relabeling only when one mode dominates clearly. Prevents noisy reward signals from triggering premature mode switches.

### Results: Mode-Comparison vs. Fixed Modes

| Model | PDMS |
|---|---|
| Non-Think RL (never CoT) | 88.3 |
| Think RL (always CoT) | 88.9 |
| **AdaThinkDrive (adaptive)** | **90.3** |

+2.0 vs. Never-Think; +1.4 vs. Always-Think. Per-level:
- Level 1 (simple): AdaThinkDrive 90.7 vs. Non-Think RL 88.5 (+2.2)
- Level 3 (challenging): AdaThinkDrive 89.8 vs. Think RL 87.8 (+2.0)

**Behavioral confirmation**: 84% Non-Think in Level 1; 96% Think in Level 3.

**Inference time**: 0.74s — +9% vs. Non-Think RL (0.68s), −14% vs. Think RL (0.86s).

### Reward Ablation (Table VI)

| PDMS+Format | +Endpoint | +Adaptive Think | PDMS |
|---|---|---|---|
| ✓ | | | 88.1 |
| ✓ | ✓ | | 89.1 |
| ✓ | ✓ | ✓ | **90.3** |

Adaptive Think adds +1.2 beyond other reward components.

### AdaThinkDrive vs. AutoVLA: Two Approaches to Adaptive Reasoning

| Aspect | AutoVLA | AdaThinkDrive |
|---|---|---|
| Mechanism | CoT length penalty ($r - \lambda \cdot \text{len}$) | Mode-comparison rollout ($S_{Think}$ vs. $S_{NoThink}$) |
| Signal | Continuous — discourages long CoT | Binary — rewards correct mode per scene |
| Learning | How *short* to reason | Whether to reason at all |
| Scene awareness | None (no complexity labels) | Explicit 3-level categorization + dynamic relabeling |
| Adaptive trigger | Learned implicitly from penalty | Learned explicitly from mode reward |
| Inference cost | Shorter CoT sequences | Skip CoT entirely in simple scenes |

Both achieve adaptive reasoning but via fundamentally different reward structures. AutoVLA optimizes CoT *length*; AdaThinkDrive optimizes CoT *presence*.

## Alpamayo-R1: LRM-as-Critic and Three-Component GRPO

**Alpamayo-R1** ([[sources/alpamayo-r1.md]]) introduces the most compositionally complex reward design in the wiki: three complementary signals optimized jointly via GRPO, including a **large reasoning model (LRM) as a critic** and a **binary reasoning-action consistency reward**.

### LRM-as-Critic (Reasoning Quality Reward)

Rather than using a learned reward model or simulator metric, AR1 employs a frontier LRM (DeepSeek-R1 or Cosmos-Reason) to grade each rollout's reasoning trace against the GT CoC annotation:

| Score | Meaning |
|---|---|
| 5 | Behavior & causal reasoning fully consistent with GT |
| 4 | Behavior correct; causal reasoning mostly consistent |
| 3 | Behavior roughly correct but incomplete or slightly wrong reasoning |
| 2 | Behavior partially incorrect or reasoning largely inconsistent |
| 1 | Behavior wrong or contradicts GT |
| 0 | Completely unrelated or opposite |

The motivation is the **generation-verification gap**: LRMs may struggle to *generate* valid driving reasoning (limited embodiment priors) but can reliably *evaluate* logical soundness and causal consistency. This makes them asymmetrically useful as critics — even without driving experience, they can check if a reasoning trace is internally consistent and matches the scene.

**Result**: reasoning score improves ~45% (3.1 → 4.5) when $r_\text{reason}$ is applied.

### CoC–Action Consistency Reward (Binary)

A rule-based reward that checks whether the model's reasoning trace describes behavior consistent with its own predicted trajectory:
1. Convert predicted trajectory → meta-actions (longitudinal: acceleration/braking; lateral: steering direction)
2. Parse reasoning trace → intended driving decision (from closed 14-type decision set)
3. If both axes match → $r_\text{consistency} = 1$; otherwise $0$. Unparseable traces → $0$

### Trajectory Quality Reward

$$r_\text{traj} = \lambda_\text{L2} \|x_\text{pred} - x_\text{expert}\|_2^2 + \lambda_\text{coll} \mathbb{I}[\text{collision}(x_\text{pred})] + \lambda_\text{jerk} J(x_\text{pred})$$

L2 imitation + binary collision + jerk regularization.

### Critical Finding: Reasoning-Only RL Hurts Action Quality (Table 9)

| Training | ADE↓ | Reasoning↑ | Consistency↑ | Close Enc.↓ |
|---|---|---|---|---|
| SFT | 2.12m | 3.1 | 0.62 | 6.9% |
| SFT + RL ($r_\text{reason}$) | **2.19m** | **4.5** | **0.53** | 5.8% |
| SFT + RL ($r_\text{reason}$ + $r_\text{cons}$) | **1.92m** | 4.5 | **0.85** | 6.2% |
| SFT + RL (full, all 3) | 1.94m | 4.4 | 0.83 | **3.7%** |

Reasoning-only RL improves the reasoning score but **degrades ADE from 2.12m to 2.19m** and **lowers consistency from 0.62 to 0.53**. The model learns fluent but causally disconnected explanations that fail to translate into coherent actions. The consistency reward is the essential anchor: adding it restores ADE to 1.92m (−9.4%) and boosts consistency to 0.85 (+37%). The safety reward further reduces close encounter rate (6.2% → 3.7%) without compromising other metrics.

This is the strongest evidence in the wiki that **reasoning quality and action quality are not automatically aligned** — they require an explicit coupling reward.

### RL Data Curation: Boltzmann Disagreement

To avoid prohibitive compute, AR1 curates a high-information-gain RL dataset by prioritizing samples where model logits disagree with the reward signal. For each rollout batch, a Boltzmann distribution is computed from rewards:

$$p_\text{reward}(\tau_i) = \frac{\exp(\beta r_i)}{\sum_j \exp(\beta r_j)}$$

Samples with high KL divergence between model logit distribution and $p_\text{reward}$ are prioritized. Mixed with uniform random samples (same proportion) to preserve diversity. This focuses gradient updates on cases where the model's implicit preference conflicts with the explicit reward — maximizing alignment efficiency.

### GRPO Formulation

$$\mathcal{L}_\text{GRPO}(\theta) = -\mathbb{E}_{\tau_i \sim \pi_\theta} \left[ \frac{\exp(\beta A_i)}{\sum_j \exp(\beta A_j)} \left( \log \pi_\theta(\tau_i) - \lambda_\text{KL} \text{KL}[\pi_\theta(\tau_i) \| \pi_\text{ref}(\tau_i)] \right) \right], \quad A_i = r_i - \bar{r}$$

KL regularization to SFT reference policy prevents over-optimization on noisy reward signals and preserves pre-trained priors.

## FLARE: BC-Regularized GRPO for Diffusion Planners

**FLARE** ([[sources/flare.md]]) introduces a variant of GRPO that replaces the standard KL divergence penalty with **Behavior Cloning (BC) regularization**. This is motivated by DriveFine's finding that weakly-coupled diffusion planners suffer reward hacking under KL-penalized GRPO (PDMS improves but EPDMS degrades).

### FLARE's GRPO Formulation

Standard clipped PPO objective with BC regularization:
$$\mathcal{L}_\text{total} = -\frac{1}{G}\sum_i \min(r_i(\theta)A_i, \text{clip}(r_i(\theta), 1-\epsilon, 1+\epsilon)A_i) + \lambda \mathcal{L}_\text{BC}$$

where $r_i(\theta) = \pi_\theta(\tau_i|\mathbf{z}) / \pi_\text{ref}(\tau_i|\mathbf{z})$ is the likelihood ratio and $\mathcal{L}_\text{BC}$ anchors the current policy to the Stage 1 frozen reference policy.

**BC vs. KL regularization for diffusion planners**:

| Aspect | KL divergence (standard) | BC regularization (FLARE) |
|--------|-------------------------|--------------------------|
| Constraint type | Distribution-level divergence | Sample-level imitation |
| Target | Any policy within KL ball | Specific Stage 1 checkpoint |
| Strength | Soft (increasing KL gradually) | Hard anchor to reference |
| Rationale | Prevent over-optimization | Prevent policy collapse for DiT |

The paper claims BC is more reliable for preventing collapse in diffusion planners because the DiT's trajectory log-probability (required for KL) is approximated by summing Gaussian transition log-probs across the denoising chain — a potentially noisy estimate. BC directly anchors to the reference policy without requiring log-probability estimation.

### GRPO Hyperparameters (FLARE Stage 2)

- G = 16 trajectory samples per scene (vs. G=3 in WAM-Flow, G=10 in DriveFine)
- DDIM with T'=5 denoising steps (fast inference for group sampling)
- VLM backbone **frozen**; only fusion module + DiT planner updated
- λ_BC = 0.1; lr = 2×10⁻⁵; 15 epochs; 8×H100

**Result**: SFT 86.9 → RFT 91.4 PDMS (+4.5 PDMS); EC = 87.5 on NAVSIM-v2 — no reward hacking observed (contrast with DriveFine finding for KL-penalized diffusion planners).

## DiffusionDriveV2: Anchored Truncated GRPO

**DiffusionDriveV2** ([[sources/diffusiondrive-v2.md]]) applies RL specifically to the **truncated GMM diffusion** architecture of DiffusionDrive, addressing a failure mode that vanilla GRPO-on-diffusion would not solve: cross-anchor mode collapse.

### The Multi-Modal Supervision Gap in IL

DiffusionDrive's GMM prior has $N_\text{anchor}$ = 20 anchors, each representing a distinct driving intent (straight, turn, overtake, etc.). Imitation learning can only supervise the one anchor closest to the GT trajectory per scene. The 19 negative-mode anchors receive **zero quality constraints** — they generate trajectories but nothing prevents those trajectories from colliding. The downstream selector (a far smaller module) must save the system from these unsafe samples, and fails under OOD conditions.

### Why Vanilla GRPO Fails for GMM Diffusion

Applying standard GRPO across all anchors as one group causes mode collapse. A "turn left" trajectory compared to a "go straight" trajectory via group-level advantage estimation would systematically favor the dominant mode — exactly the problem DiffusionDrive's anchor design was meant to solve. The insight: **cross-intent advantage comparison is conceptually wrong for multi-modal models**.

### Intra-Anchor GRPO

For each anchor $k$, generate $G$ trajectory variations using multiplicative exploration noise. Compute GRPO advantages **within this same-intent group only**:

$$A^{k,i} = \frac{r^{k,i} - \text{mean}(\{r^{k,j}\}_{j=1}^G)}{\text{std}(\{r^{k,j}\}_{j=1}^G)}$$

The RL loss uses a denoising discount $\gamma_{t-1}$ (set to 0.8) to downweight gradients from early, noisier denoising steps:

$$L_\text{RL} = -\frac{1}{N_\text{anchor}}\sum_k \frac{1}{G}\sum_i \frac{1}{T_\text{trunc}}\sum_t \gamma_{t-1} \log \pi_\theta(\tau_{t-1}^{k,i} \mid \tau_t^{k,i})\, A^{k,i}$$

Combined with an IL regularization term ($\lambda L_\text{IL}$, BC = 0.1) to preserve pre-trained driving capability.

**Ablation**: intra-anchor vs. cross-anchor — 90.1 vs. 89.2 PDMS (+0.9). Cross-anchor causes partial mode collapse even in a non-degenerate setting.

### Inter-Anchor Truncated GRPO

Pure intra-anchor isolation has a blind spot: a colliding trajectory can receive positive local advantage if it outperforms its own anchor group peers. Fix — truncated advantage:

$$A_\text{trunc}^{k,i} = \begin{cases} -1 & \text{if collision} \\ \max(0, A^{k,i}) & \text{otherwise} \end{cases}$$

Principle: **reward relative improvements, penalize absolute failures**. Negative local advantages are zeroed (no punishment for "safe but below group mean"). Collisions receive a universal hard penalty regardless of anchor group or local rank. Only trajectories that are both safe and locally above average receive positive gradient.

**Ablation**: with vs. without inter-anchor truncation — 90.1 vs. 89.5 PDMS (+0.6).

### Scale-Adaptive Multiplicative Exploration Noise

Standard additive Gaussian noise disrupts trajectory smoothness because trajectory coordinates have scale-dependent magnitudes (near waypoints small, far waypoints large). DiffusionDriveV2 uses multiplicative noise: $\tau' = (1 + \epsilon_\text{mul})\tau$ with just two scalars (longitudinal, lateral). The perturbation is proportionally larger for distant waypoints — matching the natural uncertainty profile. Ablation: multiplicative vs. additive — 90.1 vs. 89.7 PDMS (+0.4).

### Results

DiffusionDriveV2 achieves **91.2 PDMS** on NAVSIM v1 (ResNet-34, +3.1 over DiffusionDrive, new non-VLM SOTA). Diversity–quality trade-off (raw output, 20 trajectories, no selector):

| Method | Div. | PDMS@1 | PDMS@5 | PDMS@10 |
|--------|------|--------|--------|---------|
| DiffusionDrive | 42.3 | 93.5 | 84.3 | 75.3 |
| **DiffusionDriveV2** | **30.3** | **94.9** | **91.1** | **84.4** |

RL raises both the upper bound (PDMS@1: +1.4) and the quality floor (PDMS@10: +9.1) — validating the exploration-constraint dual role of RL.

### Connection to Other RL Papers in the Wiki

- **vs. ReCogDrive (vanilla GRPO on diffusion)**: V2 uses anchor-scoped GRPO instead of flat group estimation. ReCogDrive achieves +2.8 PDMS from RL; V2 achieves +3.1 PDMS from RL — similar magnitude, but V2 is specifically designed to avoid mode collapse that ReCogDrive would suffer from if applied to a GMM model.
- **vs. FLARE (BC-regularized GRPO on DiT)**: both use BC regularization to prevent reward hacking in diffusion planners. FLARE uses BC in place of KL divergence; V2 uses BC as a supplementary IL loss alongside truncated GRPO. Both avoid the EPDMS degradation that DriveFine reported for KL-penalized diffusion planners.
- **vs. DIVER**: also RL-based on a similar architecture; DiffusionDriveV2 outperforms DIVER by +2.9 PDMS (91.2 vs. 88.3), suggesting the intra/inter-anchor GRPO design provides a meaningful advantage over simpler RL approaches.

## WAM-Diff: GSPO — Sequence-Level RL for MoE Policies

**WAM-Diff** ([[sources/wam-diff.md]]) introduces **Group Sequence Policy Optimization (GSPO)**, a variant of GRPO specifically designed to remain stable when the policy contains a sparse **Mixture-of-Experts** (MoE) architecture. The core motivation: standard GRPO computes advantages and updates at the **token level**, which is fundamentally incompatible with MoE routing. Each token-level gradient update changes expert routing decisions at that position, creating incoherent credit assignment across the sequence — the very experts that produced good trajectory segments can lose their identity between steps.

### Why Token-Level GRPO Fails for MoE

In standard GRPO (as used in DriveFine, WAM-Flow, ReCogDrive, etc.), the advantage $\hat{A}_i$ is computed per trajectory and the policy gradient flows through each individual token log-probability. For a dense model, this is fine. For an MoE model with sparse expert routing, updating the $k$-th token's embedding changes the softmax gate output $g_i(z_k)$, which in turn re-routes the $k$-th token to a different expert in the next forward pass. This routing instability means the gradient signal "forgets" which expert generated which trajectory segment — providing noisy, often contradictory credit.

GSPO resolves this by treating each **complete trajectory as the atomic unit** of policy optimization.

### GSPO Formulation

**Step 1** — Sample $G=3$ candidate sequences from the old policy:
$$\{x_i\}_{i=1}^G \sim \pi_{\theta_\text{old}}(\cdot|c_t)$$

**Step 2** — Evaluate in NAVSIM simulator; group-normalize rewards:
$$\hat{A}_i = \frac{R_i - \text{mean}(\{R_j\}_{j=1}^G)}{\text{std}(\{R_j\}_{j=1}^G)}$$

**Step 3** — Compute length-normalized sequence likelihood ratio (estimated via single one-step unmasking per token):
$$s_i(\theta) = \exp\!\left(\frac{1}{|x_i|}\sum_{k=1}^{|x_i|}\log\frac{\pi_\theta(x_{i,k}|c_t)}{\pi_{\theta_\text{old}}(x_{i,k}|c_t)}\right)$$

The $1/|x_i|$ normalization is critical: without it, longer sequences would have systematically smaller (more negative) log-ratios, biasing advantage estimates against complete trajectories.

**Step 4** — Clipped PPO objective at sequence level:
$$J_\text{GSPO}(\theta) = \mathbb{E}_x\!\left[\frac{1}{G}\sum_{i=1}^G \min\!\Big(s_i(\theta)\hat{A}_i,\;\text{clip}\big(s_i(\theta),1-\epsilon,1+\epsilon\big)\hat{A}_i\Big)\right]$$

The clipping constraint operates in *sequence-likelihood* space (not token space) — appropriate for MoE because it constrains the full policy update as measured by how much the trajectory distribution shifts, rather than individual routing decisions.

### GSPO vs. Standard GRPO for MoE

| Aspect | GRPO (token-level) | GSPO (sequence-level) |
|---|---|---|
| Advantage unit | Per token | Per trajectory |
| MoE routing stability | Unstable — token updates re-route experts | Stable — routing consistent within one trajectory |
| Credit assignment | Must attribute reward to each token's expert | Full trajectory credited as unit; no per-token assignment |
| Training curves | Noisy, lower convergence ceiling | Steady improvement to higher reward |
| Optimal group size | Varies | **G=3** (WAM-Diff ablation) |

![[gspo2grpo.png]]
*Training reward curves: GSPO maintains steadily higher reward than GRPO throughout training, confirming routing-stability advantage.*

### Ablation Results

| Config | PDMS |
|---|---|
| w/o GSPO (MoE-only SFT) | 86.6 |
| GSPO G=2 | 88.9 (+2.3) |
| **GSPO G=3** | **91.0 (+4.4)** |

Larger group size promotes more diverse candidate trajectories, providing cleaner gradient signal. G=3 is the sweet spot — G=2 provides too little reward spread; beyond G=3 training cost grows without clear benefit.

**GSPO contributes +4.4 PDMS** (86.6 → 91.0) — the single largest gain in WAM-Diff's ablation stack, larger than MoE (+1.9), CFG (+2.4), or decoding schedule (+2.0).

### Reward Design

Multi-dimensional PDMS-based reward evaluated by NAVSIM simulator: NC (no collision) × DAC (drivable area) × weighted combination of EP (ego progress), TTC (time-to-collision), and Comfort. The ablation shows that optimizing individual sub-rewards (TTC-only, EP-only, Comfort-only) hurts the composite score — each improves its target metric while degrading the others. Using the full PDMS composite achieves the best balance at 91.0 vs. 90.3–90.7 for single-factor rewards.

### Position in the RL Landscape

GSPO is the **only method in the wiki where RL is specifically designed around MoE routing stability**. Compare with other architecture-specific RL adaptations:

- **DiffusionDriveV2**: adapts GRPO for *multi-anchor GMM diffusion* (intra-anchor scoping prevents cross-intent collapse)
- **FLARE**: adapts GRPO for *DiT diffusion planners* (BC regularization prevents policy collapse when KL estimation is noisy)
- **WAM-Diff GSPO**: adapts GRPO for *sparse LoRA MoE* (sequence-level objective prevents routing instability)

All three papers independently arrive at the conclusion that standard GRPO requires architectural-specific adaptation when the policy has a non-standard structure.

---

### Comparison: All GRPO Reward Designs in the Wiki

| Method | Reward | Simulator needed? | Reasoning reward? | Adaptive reasoning? | Regularization |
|--------|--------|-------------------|-------------------|---------------------|----------------|
| ReCogDrive | PDMS (NAVSIM) | Yes | No | No | BC (λ=0.01) |
| WAM-Flow | Safety-gated PDMS | Yes | No | No | KL |
| Reasoning-VLA | GT-trajectory + binary kinematics | No | No | No | KL |
| DriveFine | PDMS + pairwise/online for refiner | Yes | No | No | KL |
| Curious-VLA | SDR (focal-style PDMS) | Yes | No | No | KL |
| AutoVLA | PDMS − λ·CoT length | Yes | No | Yes (length penalty) | KL |
| AutoDrive-R² | 4-component physics MSE | No | No | No | KL |
| Alpamayo-R1 | LRM-graded reasoning + binary consistency + L2/collision/jerk | No | Yes (LRM-as-critic) | No | KL |
| AdaThinkDrive | PDMS + format + endpoint + Adaptive Think | Yes | No | Yes (mode-comparison rollout) | KL |
| FLARE | PDM-Score (R_progress + R_safety + R_comfort) | Yes | No | No | BC (λ=0.1) |
| DreamerAD | Latent AD-RM (8 dims × 8 horizons, log-sigmoid safety) | Vocabulary filtering only | No | No | BC + KL |
| NoRD | PDMS + format + length (all [0,1]) | Yes | No | No | None (no KL) |
| **DiffusionDriveV2** | **PDMS (NAVSIM) + hard collision penalty (−1)** | **Yes** | **No** | **No** | **BC (λ=0.1, IL loss)** |
| WAM-Diff | PDMS (NAVSIM composite) | Yes | No | No | Clipping (seq-level, no KL) |

AR1 is the only method with explicit reasoning evaluation as a reward component. FLARE, DiffusionDriveV2, and WAM-Diff all adapt GRPO to handle non-standard planner architectures: FLARE uses BC regularization for DiT planners; DiffusionDriveV2 uses anchor-scoped truncated advantage for GMM planners; WAM-Diff uses sequence-level GSPO for sparse MoE planners. **DreamerAD is the only method where reward is computed from latent world model features** rather than a PDM simulator or GT trajectories. **WAM-Diff's GSPO is the only method that eliminates token-level credit assignment entirely** — replacing it with sequence-level importance ratios to maintain MoE routing stability.

## DreamerAD: RL within Latent World Model Imagination Space

**DreamerAD** ([[sources/dreameraD.md]]) introduces the most architecturally distinct RL approach in the wiki: **performing RL entirely inside the latent space of a diffusion world model**, with rewards derived from learned latent representations — not from a PDM simulator.

### Key Motivation

All prior NAVSIM GRPO methods (ReCogDrive, WAM-Flow, DriveFine, FLARE, etc.) share the same reward pipeline: sample trajectories → execute in NavSim non-reactive simulator → receive PDMS score → optimize policy. This requires the simulator at every RL step and limits RL interaction speed to simulator throughput.

DreamerAD replaces the simulator reward with a **learned Autoregressive Dense Reward Model (AD-RM)** that operates directly on latent features from a world model's diffusion process — enabling RL rollouts at 0.03s/frame vs. 2s/frame for pixel-level diffusion.

### Shortcut Forcing (SF-WM)

The prerequisite for latent RL: compress Epona's 100-step flow-matching inference to 1 step via recursive multi-resolution teacher-student distillation. Step sizes are powers of 2; for step size $d > d_{min}$:

$$v_{target} = \text{sg}\left(\frac{v_1 + v_2}{2}\right), \quad \mathcal{L} = \mathbb{E}[\omega(t)\|\phi_\theta(x_t,t,d) - v_{target}\|^2]$$

Single-step achieves 87.7 EPDMS (same as 16-step) at **0.03s latency** — 80× speedup. Critically, denoised latent features show structured spatial/semantic coherence (PCA-verified), validating that latent space is information-rich enough to support reward learning.

### Autoregressive Dense Reward Model (AD-RM)

Predicts 8 reward dimensions ($r_{NC}, r_{DAC}, r_{DDC}, r_{TLC}, r_{EP}, r_{TTC}, r_{LK}, r_{HC}$) × 8 temporal horizons (0–4s) from latent context. Autoregressively conditioned on historical latent features $\{z_{-3},\ldots,\hat{z}_t\}$ via query-based cross-attention.

**Key property**: trained with as little as **20% of training data** to achieve near-full performance — the latent world model's structured representations provide rich enough supervision that reward learning converges quickly.

**Contrast with simulator-based GRPO**: NavSim PDM simulator still required to annotate the vocabulary trajectories *before* RL training, but AD-RM completely replaces simulator calls *during* RL training.

### Safety-First Log-Fusion Reward

$$r_{total}^t = \underbrace{\sum_{i \in safe} w_i \log(\sigma(r_i))}_{L:\,\text{safety}} + \underbrace{\log\sum_{j \in task} w_j r_j}_{S:\,\text{task}}$$

Collisions: $\log(\sigma(r_{NC})) \to -\infty$, dominating the total. Dense temporal aggregation:
$$r_{final} = \sum_{t=1}^8 w_t \cdot r_{total}^t$$

This provides both **safety dominance** (log-sigmoid gates) and **fine-grained credit assignment** (8 time steps × 8 dimensions).

### Gaussian Vocabulary Sampling vs. WorldRFT vs. Flow-GRPO

| Approach | Mechanism | Problem |
|----------|-----------|---------|
| WorldRFT | Random Gaussian noise on trajectory | Dynamic discontinuity → world model hallucinations |
| Flow-GRPO | SDE forcing on ODE flow | Training/inference mismatch → jagged trajectories |
| **DreamerAD** | **Mahalanobis distance ranking over 8192→256 vocabulary** | **Smooth, physically grounded, no OOD hallucination** |

The vocabulary is filtered from 8192 candidates by proximity to GT end-states ($|\Delta y| \leq 5m$, $|\Delta x| \leq 10m$, $\Delta\theta \leq 20°$), then uniformly resampled to K=256 for lateral diversity. Mixed sampling ($g_1$ softmax-weighted + $g_2$ local-neighborhood) provides both diversity and local refinement.

### Results and Position in the RL Landscape

DreamerAD achieves **87.7 EPDMS on NAVSIM-v2** (+2.6 over Epona baseline), with largest gains in safety: NC +0.9, DAC +1.5, TTC +1.1. EP regression of −0.8 reflects the safety-efficiency tradeoff inherent in log-sigmoid safety dominance.

**NAVSIM-v1**: 88.7 PDMS (best among world-model-class methods with similar encoders; below VLA SOTA at 90.7–91.4).

**Unique position**: DreamerAD is the **only method in the wiki where RL rewards are computed from latent world model features at training time** — not simulator, not GT trajectories. This opens a path toward RL training that scales with world model quality rather than simulator speed.

**Key tradeoff vs. simulator-based GRPO**: latent AD-RM rewards are learned approximations to simulator rewards — they may not perfectly reflect all safety conditions. The reward model is trained from simulator labels but generalizes within the world model's distribution. In exchange, RL training is 80× faster per rollout.

## NoRD: Difficulty Bias in GRPO and Dr. GRPO

**NoRD** ([[sources/nord.md]]) identifies and addresses a fundamental failure mode of GRPO when applied to **weak SFT policies** — policies trained on smaller, reasoning-free datasets. This is the first paper in the wiki to characterize this failure as **difficulty bias** in the AD domain.

### Why Weak SFT + GRPO Fails

NoRD trains a base model (Qwen-2.5VL-3B-Instruct) with SFT on only 80K NAVSIM samples and no reasoning annotations. GRPO post-training yields only +0.67% PDMS (76.66 → 77.18). The root cause is the **polarized reward distribution** induced by a weak SFT policy:

| Reward region | Group-mean PDM | Intra-group std | Meaning |
|---|---|---|---|
| High (≥0.8) | High | **Low** | Simple scenarios — already solved |
| Low (≤0.15) | Low | **Low** | Fully OOD — always fails |
| Intermediate [0.2, 0.65] | Medium | **High** | Complex maneuvers — inconsistent |

The **majority** of samples fall in the intermediate (high-variance) region — sharp turns, lane changes, intersections.

**GRPO's disadvantage formula attenuates learning from exactly these samples:**
$$\hat{A}_{i,t}^{\text{GRPO}} = \frac{r(o_i \mid x) - \bar{r}}{\text{std}(r)}$$

When $\text{std}(r)$ is large (high-variance intermediate scenarios), the advantage is heavily attenuated → near-zero gradients → GRPO learns only from the small fraction of already-low-variance samples.

**Connection to Curious-VLA**: Curious-VLA identifies advantage collapse ($\sigma_R \to 0$) as the failure mode when the SFT policy is too *narrow* (concentrated, low diversity). NoRD identifies the complementary failure: advantage attenuation when the policy is too *wide* (high variance per group) due to weak initialization. Both result in GRPO stagnation but from opposite distributional extremes.

### Dr. GRPO: Removing the Std Normalization

Dr. GRPO (originally from LLM reasoning domain) removes the std denominator:
$$\hat{A}_{i,t}^{\text{DrGRPO}} = r(o_i \mid x) - \bar{r}$$

All scenarios contribute gradient signal proportional to absolute reward advantage — complex, high-variance scenarios are no longer suppressed.

**Additional design choices**:
- DAPO-style **asymmetric clipping** ($\epsilon_l \neq \epsilon_h$): prevents entropy collapse
- **No KL regularization** (following original Dr. GRPO paper)

**Result**: NoRD-base + Dr. GRPO achieves 85.62 PDMS (+11.68% vs. +0.67% for GRPO).

**Detailed sub-metric comparison** (NAVSIM test):

| Method | PDMS | Collision | DAC | TTC | Comfort |
|--------|------|-----------|-----|-----|---------|
| NoRD-base | 76.66 | 96.45 | 86.37 | 90.37 | 99.97 |
| +GRPO | 77.18 | 91.89 | 90.12 | 80.13 | 99.96 |
| **+Dr. GRPO** | **85.62** | **97.56** | **94.92** | **93.53** | **100** |

Dr. GRPO improves all safety-oriented metrics. Ego Progress is slightly lower than GRPO (+7.72 vs. +8.48) — the NC×DAC gate penalizes risky progress attempts.

### Reward Design

Format + length rewards (each 0.25) + PDM score (normalized):
$$\text{PDMS} = \text{NC} \times \text{DAC} \times \frac{5 \cdot \text{TTC} + 2 \cdot C + 5 \cdot \text{EP}}{12}$$

**Group size**: G=8; **temperature**: 1.0 for rollouts; 0.01 for validation (deterministic).

### Position in the GRPO Landscape

NoRD establishes that the SFT policy **strength** (not just size or reasoning content) determines whether GRPO is viable. A strong SFT policy (high mean reward, low variance) provides clear learning signal for GRPO. A weak policy (high variance, intermediate mean) starves GRPO even if the policy is expressive enough to eventually learn the task — the solution is fixing the optimizer, not adding more data or reasoning.

## Connection to LLM RL (GRPO / DeepSeek-R1)

The GRPO technique (group relative policy optimization) originated in LLM fine-tuning (DeepSeek-R1, AlphaDrive). Both ReCogDrive and WAM-Flow apply the same group-normalized advantage idea to planning models — evidence that LLM-RL techniques transfer broadly to generative planning models, whether continuous diffusion or discrete flow matching. Senna-2 takes a different route (hierarchical longitudinal penalties rather than group policy gradient) but shares the same goal: closed-loop safety alignment beyond what imitation learning achieves.

---
title: Reinforcement Learning for Autonomous Driving
type: concept
sources: [raw/papers/ReCogDrive_ A Reinforced Cognitive Framework for End-to-End Autonomous Driving.md, raw/papers/WAM-Flow_ Parallel Coarse-to-Fine Motion Planning via Discrete Flow Matching for Autonomous Driving.md, raw/papers/Senna-2_ Aligning VLM and End-to-End Driving Policy for Consistent Decision Making and Planning.md, raw/papers/Reasoning-VLA_ A Fast and General Vision-Language-Action Reasoning Model for Autonomous Driving.md]
related: [sources/recogdrive.md, sources/wam-flow.md, sources/senna2.md, sources/reasoning-vla.md, concepts/diffusion-planner.md, concepts/discrete-flow-matching.md, concepts/dual-system-vla.md, concepts/navsim-benchmark.md]
created: 2026-04-05
updated: 2026-04-05
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

## Connection to LLM RL (GRPO / DeepSeek-R1)

The GRPO technique (group relative policy optimization) originated in LLM fine-tuning (DeepSeek-R1, AlphaDrive). Both ReCogDrive and WAM-Flow apply the same group-normalized advantage idea to planning models — evidence that LLM-RL techniques transfer broadly to generative planning models, whether continuous diffusion or discrete flow matching. Senna-2 takes a different route (hierarchical longitudinal penalties rather than group policy gradient) but shares the same goal: closed-loop safety alignment beyond what imitation learning achieves.

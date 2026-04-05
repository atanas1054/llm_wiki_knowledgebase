---
title: Diffusion-Based Trajectory Planner
type: concept
sources: [raw/papers/ReCogDrive_ A Reinforced Cognitive Framework for End-to-End Autonomous Driving.md, raw/papers/WAM-Flow_ Parallel Coarse-to-Fine Motion Planning via Discrete Flow Matching for Autonomous Driving.md, raw/papers/UniUGP_ Unifying Understanding, Generation, and Planing For End-to-end Autonomous Driving.md, raw/papers/Discrete Diffusion for Reflective Vision-Language-Action Models in Autonomous Driving.md, raw/papers/Reasoning-VLA_ A Fast and General Vision-Language-Action Reasoning Model for Autonomous Driving.md]
related: [sources/recogdrive.md, sources/wam-flow.md, sources/uniugp.md, sources/reflectdrive.md, sources/reasoning-vla.md, concepts/rl-for-ad.md, concepts/vlm-domain-adaptation.md, concepts/discrete-flow-matching.md, concepts/world-model-for-ad.md, concepts/inference-time-safety.md]
created: 2026-04-05
updated: 2026-04-05
confidence: high
---

## What It Is

A trajectory planning approach that applies Denoising Diffusion Probabilistic Models (DDPM) to generate smooth, continuous driving trajectories. Instead of predicting waypoints directly (as with MLP heads or autoregressive VLMs), the planner iteratively denoises a Gaussian noise sample into a trajectory.

## Why Diffusion for Trajectory Planning?

Autoregressive VLM trajectory prediction has two problems:
1. **Precision loss** — floating-point coordinates must be discretized into language tokens
2. **Output collapse** — autoregressive generation can produce incoherent or invalid trajectories

Diffusion models operate natively in continuous space and are well-suited to **multi-modal action distributions** (e.g., at an intersection, a car might turn left or go straight — both valid).

## Core Formulation

**Forward process** (training): gradually corrupt a clean trajectory $x_0$ with Gaussian noise:
$$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\sigma_t^2}\, x_{t-1}, \sigma_t^2 \mathbf{I})$$

**Reverse process** (inference): start from $x_T \sim \mathcal{N}(0, \mathbf{I})$ and iteratively denoise:
$$x_{t-1} = \frac{1}{\sqrt{1-\sigma_t^2}}\left(x_t - \sigma_t^2\, \epsilon_\theta(x_t, t)\right) + \sigma_t z$$

The denoising network $\epsilon_\theta$ is trained to predict the noise added at each step.

## Architecture in ReCogDrive

Uses a **Diffusion Transformer (DiT)** architecture conditioned on VLM representations:

**Inputs concatenated as $z_t$:**
- $E_{act}(x_t)$: noisy trajectory sample encoded to high-dim
- $E_{his}(x_{hist})$: history trajectory embedding
- $\bar{F}_h$: average-pooled VLM final hidden states (global semantic context)

**DiT block structure:**
1. **Self-attention** over waypoint queries, conditioned via AdaLayerNorm on ego status + diffusion timestep
2. **Cross-attention** from waypoint queries to full VLM hidden sequence $F_h$ (fine-grained semantic grounding)

**Output:** denoised continuous trajectory $x_0 \in \mathbb{R}^{N \times 3}$ (N waypoints × [x, y, heading])

**Training loss:** $\mathcal{L}_{dif} = \mathbb{E}_{z_t, c} \|\epsilon - \epsilon_\pi(z_t, c)\|^2$

No classifier-free guidance is used (found to destabilize trajectory generation).

## As an MDP for RL

The diffusion denoising chain $(x_T, x_{T-1}, \ldots, x_0)$ can be viewed as an internal Markov Decision Process:
- **State**: $x_t$ at each denoising step
- **Action**: $x_{t-1}$ (the denoised sample)
- **Policy**: $\pi_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$

This allows standard policy gradient / GRPO-style RL to be applied directly to the diffusion planner. See [[concepts/rl-for-ad.md]].

## Key Papers Using Diffusion for Planning

- **Diffusion Policy** (Chi et al.) — extended diffusion to robot learning, handles multi-modal action distributions
- **DiffusionDrive** — truncated diffusion policy for autonomous driving, mitigates mode collapse
- **Diffusion Planner** — joint trajectory generation for planning + motion forecasting
- **GR00T-N1** — VLM reasoning + DiT action module (robotics)
- **ReCogDrive** — VLM + DiT planner + RL fine-tuning (autonomous driving)

## Learnable Action Queries: A Non-Diffusion Parallel Alternative (Reasoning-VLA)

**Reasoning-VLA** ([[sources/reasoning-vla.md]]) introduces a fundamentally different paradigm for parallel trajectory generation: **learnable action queries** that attend to VLM hidden states via cross-attention, producing all waypoints in a **single forward pass** — no diffusion, no flow matching, no AR.

### How It Works

Action queries $AQ \in T \times N \times D$ are initialized from GT trajectory statistics (Gaussian sampling of mean/variance over training corpus) and then become learnable parameters. Each query attends to VLM hidden states via:
1. Self-attention among queries
2. Cross-attention to VLM KV cache

The ARM (Action Refinement Module) then refines the parallel outputs via MLP + attention. The result is a continuous trajectory in one forward pass.

### Design Space Comparison

| Paradigm | Steps | Parallel? | Continuous output? | Inpainting? | Speed |
|----------|-------|-----------|-------------------|-------------|-------|
| Continuous diffusion (ReCogDrive) | ~20 | Partial | Yes | Via guidance | Moderate |
| DFM / CTMC (WAM-Flow) | 5 | Yes | No (tokens) | Non-trivial | Fast |
| Masked diffusion (ReflectDrive) | 1 + reflection | Partial | No (tokens) | Native | Moderate |
| Continuous FM (UniUGP) | Multiple | Partial | Yes | No | Moderate |
| **Learnable queries (Reasoning-VLA)** | **1** | **Yes** | **Yes** | **No** | **Fastest** |

**Trade-off**: learnable queries are the fastest paradigm but lack iterative refinement. Quality is bounded by single-pass prediction; complex multimodal scenarios (e.g., intersection turns) may produce averaged trajectories without multi-modal diversity mechanisms.

## Discrete Flow Matching: A Related Paradigm

**WAM-Flow** ([[sources/wam-flow.md]]) introduces **Discrete Flow Matching (DFM)** as an alternative to continuous diffusion for trajectory planning. Rather than denoising Gaussian noise in continuous space, DFM transports probability mass over a discrete token vocabulary via a CTMC. Key contrasts:

| Aspect | Continuous Diffusion (ReCogDrive) | Discrete Flow Matching (WAM-Flow) |
|--------|-----------------------------------|------------------------------------|
| State space | Continuous ℝ | Discrete token codebook |
| Noise model | Gaussian | Uniform / Gibbs distribution |
| Training loss | MSE (noise prediction) | Cross-entropy (posterior) |
| RL integration | Denoising chain as MDP | GRPO directly on parallel outputs |
| Inference steps for quality | Many (~20) | 5 steps competitive |
| 1-step quality | Degenerate | 89.1 PDMS (competitive) |

DFM achieves 90.3 PDMS vs. diffusion-based DiffusionDrive's 88.1 PDMS on NAVSIM-v1, using only 1 camera vs. 3 cameras + LiDAR. See [[concepts/discrete-flow-matching.md]] for full theoretical treatment.

## Masked Discrete Diffusion for Planning (ReflectDrive)

**ReflectDrive** ([[sources/reflectdrive.md]]) uses **masked discrete diffusion** (LLaDA-V backbone) as an alternative to continuous DiT planners. This is a third variant in the diffusion-for-planning design space — distinct from both continuous diffusion (ReCogDrive) and CTMC-based DFM (WAM-Flow):

### Trajectory Representation
Each 2D waypoint quantized to a token pair $(x, y)$ from a uniform 1D codebook over $[-M, M]$. Full N-waypoint trajectory = $2N$-token sequence. Training uses the standard masked NLL objective — identical to BERT pre-training but conditioned on visual scene context.

### The Inpainting Insight
Masked diffusion training is inpainting training. A model trained to complete masked token sequences can **repair** a trajectory by:
1. Inserting a corrected waypoint token as a fixed anchor
2. Masking surrounding tokens
3. Running one forward pass — the model re-establishes coherence around the anchor

This inpainting-as-repair property is the architectural foundation for ReflectDrive's **inference-time safety correction** (see [[concepts/inference-time-safety.md]]).

### Masked Diffusion vs. Continuous Diffusion vs. DFM

| Aspect | Continuous Diffusion (ReCogDrive) | DFM / CTMC (WAM-Flow) | Masked Diffusion (ReflectDrive) |
|--------|-----------------------------------|-----------------------|----------------------------------|
| State space | Continuous ℝ | Discrete tokens (CTMC) | Discrete tokens (masked) |
| Training loss | MSE noise prediction | Cross-entropy posterior | Cross-entropy (masked NLL) |
| Inpainting | Requires guided sampling | Non-trivial | Native (training = inpainting) |
| Safety correction | Gradient guidance (expensive) | Not directly used | Token search + inpaint (free) |
| RL training | Denoising chain as MDP | GRPO on parallel outputs | Not used (inference-time fix) |
| Inference steps | ~10–20 | 5 (competitive at 1) | Iterative (1–5 reflection cycles) |

## Continuous Flow Matching for Planning (UniUGP)

**UniUGP** ([[sources/uniugp.md]]) uses **continuous** flow matching (not discrete) for its Planning Expert, coupled with a VLM via a **Mixture-of-Transformers (MoT)** architecture:

- Noised action: $\mathbf{a}_\tau = \tau\mathbf{a} + (1-\tau)\epsilon$, $\tau \sim [0,1]$
- Training loss: $\mathcal{L}_{plan} = \mathbb{E}[||\mathbf{u}_\tau^{plan} - (\epsilon - \mathbf{a})||_2]$

**MoT coupling (tighter than cross-attention)**: understanding tokens (from Qwen2.5-VL-3B) and planning tokens attend together in shared multi-head self-attention at every layer, then diverge into modality-specific FFNs. Compare to ReCogDrive, where the VLM hidden states condition the DiT planner via cross-attention — a one-directional, shallower coupling.

**World model co-training** further improves planning: UniUGP's generation expert (Wan2.1 DiT), conditioned on planned trajectories, back-propagates a video consistency signal into the shared representation. This forces the planner to attend to causally relevant distant objects. Planning L2 improves 1.72→1.45 on the long-tail benchmark when the generation expert is added. See [[concepts/world-model-for-ad.md]].

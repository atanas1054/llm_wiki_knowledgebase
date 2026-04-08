---
title: Diffusion-Based Trajectory Planner
type: concept
sources: [raw/papers/ReCogDrive_ A Reinforced Cognitive Framework for End-to-End Autonomous Driving.md, raw/papers/WAM-Flow_ Parallel Coarse-to-Fine Motion Planning via Discrete Flow Matching for Autonomous Driving.md, raw/papers/UniUGP_ Unifying Understanding, Generation, and Planing For End-to-end Autonomous Driving.md, raw/papers/Discrete Diffusion for Reflective Vision-Language-Action Models in Autonomous Driving.md, raw/papers/Reasoning-VLA_ A Fast and General Vision-Language-Action Reasoning Model for Autonomous Driving.md, raw/papers/ORION_ A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation.md, raw/papers/Unifying Language-Action Understanding and Generation for Autonomous Driving.md, raw/papers/DriveFine_ Refining-Augmented Masked Diffusion VLA for Precise and Robust Driving.md, raw/papers/AutoVLA_ A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning.md, raw/papers/Alpamayo-R1_ Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail.md]
related: [sources/recogdrive.md, sources/wam-flow.md, sources/uniugp.md, sources/reflectdrive.md, sources/reasoning-vla.md, sources/orion.md, sources/linkvla.md, sources/drivefine.md, sources/autovla.md, sources/alpamayo-r1.md, concepts/rl-for-ad.md, concepts/vlm-domain-adaptation.md, concepts/discrete-flow-matching.md, concepts/world-model-for-ad.md, concepts/inference-time-safety.md]
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

| Paradigm                              | Steps          | Parallel? | Continuous output? | Inpainting?  | Speed       |
| ------------------------------------- | -------------- | --------- | ------------------ | ------------ | ----------- |
| Continuous diffusion (ReCogDrive)     | ~20            | Partial   | Yes                | Via guidance | Moderate    |
| DFM / CTMC (WAM-Flow)                 | 5              | Yes       | No (tokens)        | Non-trivial  | Fast        |
| Masked diffusion (ReflectDrive)       | 1 + reflection | Partial   | No (tokens)        | Native       | Moderate    |
| Continuous FM (UniUGP)                | Multiple       | Partial   | Yes                | No           | Moderate    |
| **Learnable queries (Reasoning-VLA)** | **1**          | **Yes**   | **Yes**            | **No**       | **Fastest** |
| VAE + GRU (ORION)                     | 1              | Yes       | Yes                | No           | Fast        |
| Shared codebook C2F (LinkVLA)         | 2              | Step 2 Yes | No (tokens)       | No           | Fast (48ms) |
| Block-MoE masked diffusion (DriveFine) | s + 1         | Yes (gen) + 1 refine | No (tokens) | Native (refinement expert) | Moderate |
| AR over physical codebook (AutoVLA) | T (AR steps) | No | No (codebook tokens) | No | Moderate (1 Hz w/ CoT) |
| FM action expert + discrete training (AR1) | 5 (Euler) | Yes (expert) | Yes (continuous) | No | Fast (8.75ms) |

**Trade-off**: learnable queries are the fastest paradigm but lack iterative refinement. Quality is bounded by single-pass prediction; complex multimodal scenarios (e.g., intersection turns) may produce averaged trajectories without multi-modal diversity mechanisms.

### Action Decoder Scaling Reversal (DriveVLA-W0)

**DriveVLA-W0** ([[sources/drivevla-w0.md]]) systematically compares query-based, AR, and flow matching action decoders across data scales (103k vs. 70M frames), revealing a striking performance reversal:

| Decoder | NAVSIM (103k) PDMS | 70M-frame ADE | 70M-frame Col. |
|---|---|---|---|
| Query-based | **88.4** | 1.124m | 4.53% |
| Flow Matching (10 steps) | 87.2 | 1.036m | 3.98% |
| Autoregressive | 85.3 | **1.007m** | **2.95%** |

**Small scale**: query-based > FM > AR. Simple trajectory distribution — precision advantage of continuous decoders matters.  
**Large scale (70M)**: AR > FM > query-based. Complex trajectory distribution — AR's teacher-forced training scales efficiently; FM is too sample-inefficient to converge; query-based hits a representational bottleneck.

**Implication**: the diffusion/FM action paradigm's advantage over AR is scale-dependent. At academic benchmark scale, FM wins; at production-scale data (70M+ frames), AR may be the better choice. This challenges the assumption that FM action experts are universally superior to AR token decoders for trajectory generation.

## Perception-Aligned Action Decoding (Percept-WAM)

**Percept-WAM** ([[sources/percept-wam.md]]) introduces yet another paradigm: a **four-query MLP decoder** where each query set attends exclusively to one input modality, preventing over-reliance on any single representation.

| Query | Attends to | Purpose |
|---|---|---|
| Q_ego | Ego-state tokens | Kinematic grounding |
| Q_pv | World-PV tokens | Semantic/appearance context |
| Q_bev | World-BEV tokens | 3D spatial context |
| Q_full | All tokens | Final trajectory output |

All four decoded in parallel via Smooth-L1 loss during training; only Q_full used at inference. Unlike Reasoning-VLA's single query set with unrestricted cross-attention, Percept-WAM's attention masking enforces that each modality contributes independently.

The trajectory decoder reuses World-PV and World-BEV tokens computed during perception prefill — no extra forward pass. See [[concepts/perception-for-planning.md]] for the full treatment of how perception tasks improve planning representations.

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

## VAE-Based Reasoning-Action Alignment (ORION)

**ORION** ([[sources/orion.md]]) takes a fundamentally different approach from all diffusion-family methods: it uses a **VAE** to align the VLM reasoning space with the trajectory action space, bypassing iterative denoising entirely.

### The Core Idea

All previous paradigms produce trajectories from noise or learned queries. ORION starts from a **planning token** $s$ — the final hidden state of the LLM after processing the full driving reasoning chain — and enforces that $s$ encodes sufficient information to reconstruct the ground-truth trajectory:

$$p(z_s \mid s) \sim \mathcal{N}(\mu_s, \sigma_s^2), \quad p(z_t \mid t) \sim \mathcal{N}(\mu_t, \sigma_t^2)$$

$$\mathcal{L}_{vae} = D_{KL}(p(z \mid s), \, p(z \mid t))$$

At inference, $z \sim p(z_s \mid s)$ is decoded by a GRU into a **6-mode trajectory** (one per navigation command). No noise, no iterative steps, no token sampling.

### Why VAE over Diffusion?

ORION empirically compares VAE vs. a diffusion planner (K-means anchors, 20 modes) on Bench2Drive:

| Planner | DS ↑ | SR (%) ↑ | Avg. Col ↓ |
|---------|-------|----------|-----------|
| Diffusion | 71.97 | 46.54 | 0.96 |
| **VAE (Ours)** | **77.74** | **54.62** | **0.47** |

The paper attributes the gap to: (1) VAE's direct latent alignment is more effective than conditional denoising for this problem; (2) VAE training is more stable, enabling better reasoning-action co-optimization.

### Relationship to Other Paradigms

ORION's VAE planner is philosophically closest to learnable action queries (Reasoning-VLA) — both produce trajectories in a single forward pass from VLM features. The key difference is **what the VLM contributes**: in Reasoning-VLA, VLM hidden states are attended to by learned queries; in ORION, the VLM itself generates a special planning token whose *distribution* in latent space is constrained to match the trajectory distribution. This makes the alignment tighter and end-to-end differentiable through both LLM and planner.

## Continuous Flow Matching for Planning (UniUGP)

**UniUGP** ([[sources/uniugp.md]]) uses **continuous** flow matching (not discrete) for its Planning Expert, coupled with a VLM via a **Mixture-of-Transformers (MoT)** architecture:

- Noised action: $\mathbf{a}_\tau = \tau\mathbf{a} + (1-\tau)\epsilon$, $\tau \sim [0,1]$
- Training loss: $\mathcal{L}_{plan} = \mathbb{E}[||\mathbf{u}_\tau^{plan} - (\epsilon - \mathbf{a})||_2]$

**MoT coupling (tighter than cross-attention)**: understanding tokens (from Qwen2.5-VL-3B) and planning tokens attend together in shared multi-head self-attention at every layer, then diverge into modality-specific FFNs. Compare to ReCogDrive, where the VLM hidden states condition the DiT planner via cross-attention — a one-directional, shallower coupling.

**World model co-training** further improves planning: UniUGP's generation expert (Wan2.1 DiT), conditioned on planned trajectories, back-propagates a video consistency signal into the shared representation. This forces the planner to attend to causally relevant distant objects. Planning L2 improves 1.72→1.45 on the long-tail benchmark when the generation expert is added. See [[concepts/world-model-for-ad.md]].

## Block-MoE Refinement on Masked Diffusion (DriveFine)

**DriveFine** ([[sources/drivefine.md]]) identifies a fundamental asymmetry in the two main paradigms and addresses it with a plug-and-play refinement module on top of a masked diffusion LLM:

| Paradigm failure mode | Root cause | DriveFine solution |
|----------------------|------------|--------------------|
| Diffusion (ReCogDrive): loses EPDMS under PDMS GRPO | Weak coupling between diffusion planner and VLM → reward hacking | Use unified masked diffusion (no separate planner) |
| Token-based (AutoVLA): irreversible decoding errors | Committed tokens cannot be revised; noncausal ordering creates outliers | Block-MoE refinement expert reads completed sequence and corrects errors |

### Block-MoE Architecture

LLaDA-8B base (32 blocks) is split:
- **Blocks 0–27** (shared): run once, produce common contextual representation for both tasks
- **Blocks 28–31** (expert): replicated into two parallel sets:
  - **Generation expert**: input = masked tokens `[M]`; learns standard masked NLL
  - **Refinement expert**: input = fully unmasked tokens; learns to correct completed trajectories

Gradient isolation: refinement branch gradients are **strictly blocked** from reaching shared or generation blocks. This is the critical design choice — it preserves the foundational masked-LLM training paradigm of the generation expert exactly as pretrained.

**Inference**: s=12 masked diffusion steps (generation expert) → 1 refinement step (refinement expert).

**Parameter cost**: n=4 refinement blocks = +1B on 8B base. Even n=1 (+250M, +0.4 PDMS) is cost-effective.

### Relationship to ReflectDrive

Both DriveFine and ReflectDrive ([[sources/reflectdrive.md]]) use masked discrete diffusion (LLaDA/LLaDA-V backbone) and exploit the inpainting-as-repair property. The key architectural difference:

| | ReflectDrive | DriveFine |
|--|---|---|
| Safety correction | External scoring + Manhattan token search + inpaint anchor | Block-MoE: dedicated refinement blocks with gradient isolation |
| Correction trigger | Safety violations detected at inference (gradient-free) | Always applied: one refinement step after s generation steps |
| Training | Refinement not explicitly trained; emerges from masked NLL | Refinement expert explicitly trained with hybrid offline+online RL |
| Scope | Safety-focused (DAC, TTC violations) | General quality (smooth outlier correction + trajectory fluency) |

## Shared Codebook Coarse-to-Fine Generation (LinkVLA)

**LinkVLA** ([[sources/linkvla.md]]) introduces yet another paradigm: a **unified discrete codebook** shared by language tokens and action tokens, combined with a **coarse-to-fine (C2F) two-pass decoder** that eliminates sequential AR overhead.

### Unified Token Space

Action waypoints are quantized into a BEV grid using a **log coordinate transform** that prioritizes near-field resolution:

$$z' = \text{sign}(z) \cdot \log(1 + k \cdot |z|), \quad k = 5$$

The resulting 56×101 grid yields 5,656 action tokens, merged with the text vocabulary into one codebook $\mathcal{C}$ of size $K_\text{text} + 5{,}656$. Both language and action are processed by the same VLM with no separate action head.

**Spatial soft-labeling** replaces one-hot targets with a 2D Gaussian over grid neighbors ($\sigma=1.2$, $R=10$), embedding spatial continuity into the training loss.

### Bidirectional Training Objective

The key alignment innovation: LinkVLA trains on **both directions** of the language-action mapping:
- **Generation**: $p(A \mid V, L)$ — predict trajectory from instruction (conventional)
- **Understanding**: $p(L \mid V, A)$ — **caption the trajectory back into language** (novel auxiliary task)

Same decoder handles both by swapping $L$ and $A$ as the target. No extra data needed. This forces action token embeddings to be semantically grounded in language, verifiably closing the modality gap.

### C2F Inference (Two Passes)

1. **Pass 1**: Predict final endpoint $\hat{w}_T$ via a special goal token → construct coarse path by linear interpolation: $w_i^\text{coarse} = w_0 + \frac{i}{T}(\hat{w}_T - w_0)$
2. **Pass 2**: Tokenized coarse path → VLM predicts all $T$ fine waypoints **in parallel**

Result: 361ms (AR) → **48ms** (C2F), **86% reduction**.

### Positioning vs. Other Paradigms

LinkVLA is the only paradigm that treats language and action as **the same modality** at the token level. Compare:
- ReflectDrive (masked diffusion): separate trajectory token sequence, inpainting-capable but discrete
- WAM-Flow (DFM/CTMC): separate discrete trajectory codebook, multi-step transport
- Reasoning-VLA (learnable queries): continuous output, no shared vocabulary
- **LinkVLA**: shared vocabulary, bidirectional objective, C2F — optimized for instruction following and deployment latency

**Results (Bench2Drive)**: 91.01 DS / 74.55% SR — current Bench2Drive SOTA, surpassing ORION (77.74) and SimLingo (85.07).

## Continuous Flow Matching with Unicycle Dynamics (Alpamayo-R1)

**Alpamayo-R1** ([[sources/alpamayo-r1.md]]) uses conditional flow matching for its **action expert** — a separate smaller transformer that takes VLM KV-cache as conditioning and decodes physically interpretable trajectories via unicycle dynamics control representation.

### Unicycle Control Representation

Unlike raw (x, y) waypoints, AR1 represents trajectories as control sequences: 64 waypoints at 10 Hz (6 s horizon), each parameterized by acceleration $a^i$ and curvature $\kappa^i$:

$$\kappa^i = \frac{\phi^{i+1} - \phi^i}{s^{i+1} - s^i}, \quad a^i = \frac{v^{i+1} - v^i}{s^{i+1} - s^i}$$

Derived from GT via least-squares + Tikhonov regularization. This representation enforces physical plausibility by design — impossible accelerations and turning radii are structurally excluded.

### Flow Matching Formulation

Gaussian OT conditional flow matching:

$$L_\text{cfm}(\Theta) = \mathbb{E}_{t, (\mathbf{o},\textsc{Reason}) \sim \mathcal{D}} \|\mathbf{v}_\Theta(\mathbf{a}_t, \mathbf{o}, \textsc{Reason}) - (\mathbf{a} - \boldsymbol{\epsilon})\|$$

$$\mathbf{a}_t = t\mathbf{a} + (1-t)\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

Inference via Euler integration in 5 steps ($\delta_t = 0.2$): $\mathbf{a}_{t+\delta_t} = \mathbf{a}_t + \delta_t \, \mathbf{v}_\Theta(\mathbf{a}_t, \mathbf{o}, \textsc{Reason})$.

### Dual Representation: Discrete Training + Continuous Inference

The action expert runs at inference. During *training*, trajectories are additionally tokenized into 128 discrete tokens (2 × 64: quantized $a^i$, $\kappa^i$) for unified AR cross-entropy loss. This dual representation provides three benefits:
1. **Unified token space**: language and action share the same AR training objective, coupling reasoning and trajectory prediction
2. **GRPO compatibility**: discrete tokens allow policy gradient to directly compute advantages and gradients
3. **Inference efficiency**: FM decodes 5 Euler steps (8.75ms) vs. 127 AR tokens (222ms)

A **stop-gradient** is applied to the VLM KV-cache before the action expert — the expert cannot back-propagate into VLM weights.

### FM vs. AR Decoding (Table 11)

| Strategy | minADE₆@6s↓ | AlpaSim Score (at fault)↑ | Comfort (Accel)↑ | Rel. Speed↑ |
|---|---|---|---|---|
| Auto-Regressive | 0.6811 | 0.59 ± 0.17 | 44.05% | 1.00× |
| **Flow Matching** | **0.6440** | **1.27 ± 0.34** | **97.38%** | **1.16×** |

FM wins on all metrics. The comfort gap (97% vs. 44%) is striking — AR trajectory decoding produces kinematically incoherent sequences due to non-causal ordering; FM integrates smoothly in physical space, yielding trajectories that are nearly always within comfort acceleration bounds.

### Comparison with UniUGP's FM

Both UniUGP and AR1 use continuous flow matching, but differ in coupling architecture:
- **UniUGP**: MoT — planning tokens attend jointly with understanding tokens in *shared* self-attention at every layer; bidirectional coupling
- **AR1**: action expert attends to VLM KV-cache via cross-attention; VLM does not see action expert states; stop-gradient; modular and separately trainable

UniUGP's coupling is tighter (shared layers); AR1's is more modular (allows async updates and GRPO on discrete tokens at training).

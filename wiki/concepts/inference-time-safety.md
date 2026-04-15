---
title: Inference-Time Safety for Trajectory Planning
type: concept
sources: [raw/papers/Discrete Diffusion for Reflective Vision-Language-Action Models in Autonomous Driving.md, raw/papers/DriveFine_ Refining-Augmented Masked Diffusion VLA for Precise and Robust Driving.md]
related: [sources/reflectdrive.md, sources/drivefine.md, sources/diffusiondrive.md, concepts/diffusion-planner.md, concepts/discrete-flow-matching.md, concepts/rl-for-ad.md]
created: 2026-04-05
updated: 2026-04-15
confidence: high
---

## The Problem

Imitation-learning-based planners optimize for distributional fidelity to expert trajectories, not for hard constraint satisfaction. A trajectory can be highly probable under the model yet still violate:
- **DAC**: drivable area compliance (driving off-road)
- **NC**: no collision (intersecting other agents)
- **TTC**: time-to-collision (unsafe proximity)

Training-time solutions (RL, reward shaping) are effective but require unsafe online rollouts and suffer from sim-to-real gaps. **Inference-time safety** methods address violations without retraining.

## Taxonomy of Inference-Time Safety Methods

| Method | Mechanism | Requires Gradients? | Training Change? |
|--------|-----------|---------------------|-----------------|
| Trajectory anchors (DiffusionDrive, Hydra-MDP) | Rule-based candidate initialization | No | Architecture change |
| Diffusion guidance | Add safety reward gradient to denoising score | Yes | No |
| **Reflective inference (ReflectDrive)** | Discrete token search + inpainting | **No** | No |
| GRPO RL (WAM-Flow, ReCogDrive) | Optimize policy toward reward during training | No (inference) | Yes |
| Block-MoE refinement (DriveFine) | Dedicated refinement expert corrects completed token sequences | No | Architecture + training change |

ReflectDrive's approach is the only one in this table that operates entirely at inference time, requires no gradient computation, and requires no architectural changes or retraining.

**DriveFine's Block-MoE refinement** ([[sources/drivefine.md]]) is a training-time analog: a separate set of blocks (gradient-isolated from the main masked-diffusion backbone) is trained to read a fully completed trajectory and correct errors. Unlike ReflectDrive's inference-time token search + inpaint loop, DriveFine's correction is a single extra forward pass baked into training. It addresses the same root cause (committed token errors in discrete diffusion) but embeds the fix in model weights rather than applying it post-hoc. Key trade-off: DriveFine requires retraining; ReflectDrive works on any already-trained masked diffusion model without modification.

## Reflective Inference (ReflectDrive)

Full technical details: [[sources/reflectdrive.md]].

### Prerequisites

Requires a **masked discrete diffusion** backbone (LLaDA-V style), which natively supports:
1. **Inpainting**: fix some tokens as anchors, regenerate masked tokens conditioned on them
2. **Discrete search**: enumerate corrections over a small token neighborhood

### Two-Phase Structure

**Phase 1 — Goal-Conditioned Generation** (diversity):
- Sample terminal waypoint distribution → NMS → K diverse goals
- Inpaint K full trajectories conditioned on each goal
- Select best by Global Scorer $S_\text{global}$

**Phase 2 — Safety-Guided Regeneration** (safety):
- Iteratively find and fix safety violations
- Per-iteration: Safety Scorer → earliest violation $t^*$ → local search in $\mathcal{N}_\delta$ → safety anchor → inpaint

### Why Inpainting Works for Repair

The key structural insight: masked diffusion training loss = inpainting training. A model trained to complete masked token sequences naturally performs **coherent interpolation** around a fixed anchor. Inserting a corrected waypoint token and masking its neighborhood triggers one forward pass that re-establishes trajectory continuity. This is not possible with continuous diffusion — continuous inpainting requires score network guidance (gradients).

### Computational Properties

- **Parallel**: all 2N trajectory tokens generated/inpainted simultaneously per pass
- **Bounded**: reflection budget is a hard parameter (iterations, not convergence condition)
- **Gradient-free**: Local Scorer evaluates discrete candidates via table lookup or simple BEV computation
- **Fast in practice**: most violations resolved within 1–3 iterations

## Scoring Functions

Three composable components in ReflectDrive's safety pipeline:

### Global Scorer $S_\text{global}(\tau)$
- Evaluates full trajectory quality
- Returns 0 if any hard constraint violated (NC, DAC)
- Used to select the best goal-conditioned trajectory candidate

### Safety Scorer $S_\text{safe}(\tau)$
- Assigns per-waypoint safety score
- Identifies the **earliest** unsafe waypoint $t^*$
- Sequential scan allows precise localization of the root cause

### Local Scorer $S_\text{local}(a_x, a_y)$
- Evaluates a candidate token pair at position $t^*$
- Considers both: (a) local safety (DAC, TTC at that waypoint), (b) coherence (continuity with neighbors)
- Enables efficient enumeration over discrete neighborhood $\mathcal{N}_\delta$

## Relationship to Diffusion Guidance

**Diffusion guidance** (Dhariwal & Nichol 2021; Diffusion Planner): modifies the denoising score function with a classifier gradient:
$$\tilde{\epsilon}_\theta(x_t) = \epsilon_\theta(x_t) - \sqrt{1-\bar{\alpha}_t} \nabla_{x_t} \log p_\phi(y | x_t)$$

Problems:
1. Requires backpropagation through large models per denoising step — expensive
2. Gradient estimates in high-noise regimes are unreliable → numerical instability
3. Sensitive to guidance scale $w$ — too large destabilizes generation

Reflective inference sidesteps all three by operating in discrete space where correction = lookup, not gradient ascent.

## Limitations and Trade-offs

### Locality constraint
The Manhattan-$\delta$ local search ($\delta \leq 10$ tokens) can only make small positional adjustments. A fundamentally different trajectory (e.g., taking a different turn at an intersection) cannot be reached via local search — this must be addressed at the goal-generation stage. Safety (Phase 2) and trajectory diversity (Phase 1) are structurally decoupled.

### Oracle quality ceiling
Safety Scorer quality determines the effectiveness of Phase 2. The base ReflectDrive model uses a constant-speed assumption for surrounding agents. ReflectDrive† (GT oracle) shows how much performance is bounded by oracle accuracy.

### Quantization error
Trajectory quantization introduces discretization noise. Local search resolution is bounded by codebook granularity $\Delta_g$.

### No training signal feedback
Unlike GRPO-based methods (WAM-Flow, ReCogDrive), reflective inference provides no feedback to improve the base model. Performance ceiling = base model capability + oracle quality.

## Comparison with GRPO-Based Safety (WAM-Flow, ReCogDrive)

| Aspect | Reflective Inference (ReflectDrive) | GRPO RL (WAM-Flow, ReCogDrive) |
|--------|-------------------------------------|-------------------------------|
| When applied | Inference only | Training |
| Requires unsafe rollouts | No | No (uses NAVSIM simulator) |
| Improves base model | No | Yes |
| Oracle requirement | Yes (at inference) | Yes (during training) |
| Overhead at inference | 1–5 extra passes | None |
| Generalization | Depends on oracle quality | Baked into model weights |

GRPO-based methods are superior for deployment (no inference overhead, no runtime oracle dependency) but require access to a simulation environment during training. Reflective inference is appealing when RL training is infeasible (e.g., proprietary data, no simulator access) or as a post-hoc safety layer.

## Open Questions

- Can the Local Scorer be learned jointly with the base model (e.g., as a critic head) to improve quality?
- Can the local search radius $\delta$ be dynamically adapted based on violation severity?
- How does reflective inference perform when the diffusion backbone is replaced with a DFM backbone (WAM-Flow style)? DFM's inpainting is less natural but achievable.
- Does chaining RI on top of a GRPO-trained base model (e.g., WAM-Flow base + reflection) yield additive gains?

---
title: Inference-Time Safety for Trajectory Planning
type: concept
sources: ["raw/papers/DriveReferee_ Geometric Safety Verdicts Need Not Be Learned for Driving World-Action Models.md", raw/papers/Discrete Diffusion for Reflective Vision-Language-Action Models in Autonomous Driving.md, raw/papers/DriveFine_ Refining-Augmented Masked Diffusion VLA for Precise and Robust Driving.md, raw/papers/FeaXDrive_ Feasibility-aware Trajectory-Centric Diffusion Planning for End-to-End Autonomous Driving.md, raw/papers/Plan-R1_ Safe and Feasible Trajectory Planning as Language Modeling.md]
related: [sources/drivereferee.md, sources/reflectdrive.md, sources/drivefine.md, sources/diffusiondrive.md, sources/feaxdrive.md, sources/plan-r1.md, concepts/diffusion-planner.md, concepts/discrete-flow-matching.md, concepts/rl-for-ad.md, research-directions.md]
created: 2026-04-05
updated: 2026-09-30
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
| Drivable-area SDF guidance (FeaXDrive) | Gradient correction of predicted clean trajectory inside reverse diffusion | Yes | No for inference guidance; yes for full method |
| **Reflective inference (ReflectDrive)** | Discrete token search + inpainting | **No** | No |
| GRPO/VD-GRPO RL (WAM-Flow, ReCogDrive, Plan-R1) | Optimize policy toward reward during training | No (inference) | Yes |
| Block-MoE refinement (DriveFine) | Dedicated refinement expert corrects completed token sequences | No | Architecture + training change |
| **Analytic referee on a predicted map (DriveReferee)** | Re-simulate the plan, check footprint points against predicted drivable-area and vehicle-occupancy rasters, re-sample and replace on an alarm | **No** | A 21M BEV readout is trained; the policy is untouched (the same rule can also be distilled into the policy in training) |

ReflectDrive's approach is the only one in this table that operates entirely at inference time, requires no gradient computation, and requires no architectural changes or retraining. *(2026-09-30: [[sources/drivereferee.md]]'s gated selection also needs no gradients and no change to the policy, but it needs a separately trained map readout; see [below](#analytic-referee).)*

**FeaXDrive's drivable-area guidance** ([[sources/feaxdrive.md]]) is a continuous-diffusion counterpart to inference-time safety. It builds a local drivable-area signed distance field, samples the four corners of the vehicle footprint at each predicted waypoint, and applies a gradient correction to the predicted clean trajectory `x0` during reverse sampling. This is not post-hoc trajectory repair: the corrected `x0` is fed back into the sampling chain. It reduces drivable-area violation rate from 5.06% to 2.54% in the paper's IL-stage ablation, with guidance overhead of 4.41 ms.

**DriveFine's Block-MoE refinement** ([[sources/drivefine.md]]) is a training-time analog: a separate set of blocks (gradient-isolated from the main masked-diffusion backbone) is trained to read a fully completed trajectory and correct errors. Unlike ReflectDrive's inference-time token search + inpaint loop, DriveFine's correction is a single extra forward pass baked into training. It addresses the same root cause (committed token errors in discrete diffusion) but embeds the fix in model weights rather than applying it post-hoc. Key trade-off: DriveFine requires retraining; ReflectDrive works on any already-trained masked diffusion model without modification.

**Plan-R1's VD-GRPO** ([[sources/plan-r1.md]]) is a training-time safety-alignment method, not inference-time repair. It uses collision and drivable-area constraints as multiplicative reward gates and changes GRPO normalization so rare unsafe groups are not downweighted by high reward variance. This is useful contrast: Plan-R1 improves the policy itself, while ReflectDrive/FeaXDrive-style inference methods correct samples at generation time.

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

## DriveReferee: Compute the Verdict, Learn Only the Map {#analytic-referee}

[[sources/drivereferee.md]] separates two things a learned safety scorer does at once: infer the scene, and judge a trajectory against it. It learns the first (a BEV readout of drivable area and future vehicle occupancy from one camera) and computes the second with the evaluator's own geometry: LQR-plus-bicycle rollout, five footprint points, distance transforms.

**Deployment loop.** One sample is the default plan. If the rule finds a violation or less than 0.4 m clearance on the predicted map, the policy samples once more and the rule picks between the two.

**What was measured, all on full navtest with paired confidence intervals.**

| Question | Answer |
|---|---|
| Is the rule worth anything at inference? | **+0.30 EPDMS** on an imitation-only WAM (53 hard-gate failures fixed, 9 introduced), at 1.35× policy samples |
| Is a learned verdict better, given the same predicted map? | No detectable difference: −0.01 EPDMS, CI ±0.15 |
| Are learned verifiers on image features better? | No: +0.23 and +0.26 for the best two, inside the noise of +0.30 |
| Does gating matter? | Yes. Always taking the higher-scored sample is −0.04; random replacement is +0.02 |
| Does it still help after the same rule is distilled into the policy? | **No**: +0.06 PDMS, −0.04 EPDMS |
| Would a perfect map rescue it? | Barely: +0.08 EPDMS with ground-truth maps |

**Three things this adds to the page.**

1. **Training on a rule removes the value of enforcing it at inference.** The distillation (+0.92 EPDMS, from 462 preference pairs) absorbs what the test-time check could catch. This is the inference-against-training comparison this page has lacked, and it comes out for training. The scope is two rules, two samples and a non-reactive benchmark.
2. **The limiting factor at inference is the state, not the rule.** The predicted clearance is off by 0.50 m on average against a 0.4 m threshold, and the paper attributes most of the 9 introduced failures to the predicted map.
3. **A computed rule is only as good as its discretization.** On ground-truth maps the raster rule finds 63% of the official drivable-area violations and 79% of the collisions.

**Against the other map-based entry here.** [[sources/feaxdrive.md]] also queries a drivable-area distance field at the vehicle's corners, but it takes the local map as given and uses the gradient to steer a diffusion sample. DriveReferee predicts the map and only chooses between samples. Neither paper tests the other's combination (steering on a predicted map).

## Open Questions

*Wiki-wide open questions are collected in [[research-directions.md]].*

- Can the Local Scorer be learned jointly with the base model (e.g., as a critic head) to improve quality?
- Can the local search radius $\delta$ be dynamically adapted based on violation severity?
- How does reflective inference perform when the diffusion backbone is replaced with a DFM backbone (WAM-Flow style)? DFM's inpainting is less natural but achievable.
- Does chaining RI on top of a GRPO-trained base model (e.g., WAM-Flow base + reflection) yield additive gains? *(2026-09-30: the nearest measurement says no. [[sources/drivereferee.md]] applies the same geometric rule at inference to a policy already trained on it and gets +0.06 PDMS / −0.04 EPDMS, against +0.30 EPDMS on the untrained base.)*

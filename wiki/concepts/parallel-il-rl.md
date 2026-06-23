---
title: Parallel Imitation and Reinforcement Learning
type: concept
sources: [raw/papers/Fine-tuning is Not Enough_ A Parallel Framework for Collaborative Imitation and Reinforcement Learning in End-to-end Autonomous Driving.md]
related: [sources/pair-drive.md, concepts/rl-for-ad.md, concepts/gspo-vs-grpo.md, concepts/best-of-n.md, concepts/selection-based-planning.md, concepts/navsim-benchmark.md]
created: 2026-06-23
updated: 2026-06-23
confidence: high
---

# Parallel Imitation and Reinforcement Learning

Parallel IL/RL assigns imitation and reward optimization to separate policies instead of applying both objectives sequentially or alternately to one network. The IL policy learns a stable expert prior; the RL policy learns how to generate reward-improving alternatives around a reference trajectory.

[[sources/pair-drive.md]] is the first source in this wiki to make this separation the primary architectural contribution.

## Three Training Arrangements

| Arrangement | Parameters receiving IL/RL gradients | Main benefit | Main risk |
| --- | --- | --- | --- |
| One-shot IL → RL | Same policy, sequentially | Simple; RL starts from useful behavior | Policy drift; exploration limited by the IL basin |
| Iterative IL ↔ RL | Same policy, alternating | IL repeatedly regularizes RL | Conflicting objectives can still destructively interfere |
| Parallel IL + RL | Separate IL and RL modules | Conflict-free objectives; RL module can be reused | Composition/distribution shift moves to inference |

Parallelization does not eliminate the need for an imitation prior. It changes where that prior enters: PaIR-Drive conditions the RL sampler on a human trajectory during training and an IL trajectory during inference.

## PaIR-Drive Composition

The IL branch is a conventional sensor-to-trajectory planner. The RL branch is a recurrent residual sampler:

1. Encode the scene into BEV features.
2. Use a reference trajectory as the tree root.
3. Expand intention-conditioned lateral, longitudinal, and heading offsets.
4. Train candidate probabilities with simulator-scored GRPO.
5. At inference, root the same sampler at the IL proposal.
6. Use a learned reward world model to choose the final candidate.

The RL branch is therefore closer to a reusable proposal-and-refinement policy than to a fine-tuned replacement planner.

## Why It Can Surpass the Demonstration

Pure IL treats the demonstrated trajectory as the target. A residual RL branch treats it as a starting point and is free to produce alternatives with better evaluator reward. PaIR-Drive reports gains even when the human trajectory itself is the reference: +1.6 PDMS on its low-human-PDMS split and +10.8 EPDMS on its low-human-EPDMS split.

This evidence is reward-relative. “Surpassing the human” means scoring higher under NAVSIM's PDM evaluator, not a general claim of safer or more human-compatible behavior.

## Modularity Conditions

An RL refinement module is genuinely reusable only if:

- the reference trajectory uses a shared coordinate system and horizon;
- the scene representation expected by the RL module is available;
- inference-time IL errors remain within the sampler's trained correction range;
- the reward selector is calibrated for candidates produced around the new IL policy;
- the candidate generation and ranking latency fits the control budget.

PaIR-Drive validates reuse across TransFuser and DiffusionDrive in NAVSIM, but does not test these conditions across camera-only, VLM, different-horizon, or different-domain planners.

## Relationship to Other Wiki Methods

- **PlannerRFT:** also preserves an IL reference during RL, but fine-tunes a copied diffusion denoiser and removes exploration guidance at deployment. PaIR-Drive preserves a distinct RL refinement branch at deployment.
- **Plan-R1:** fine-tunes an ego policy while using a frozen model for surrounding-agent reactions; it separates roles, not IL and RL objectives.
- **DreamerAD:** separates policy learning from a latent reward/world model, but still uses RL to improve the planning policy rather than compose independent IL and RL trajectory branches.
- **HybridDriveVLA:** fuses complementary planner representations and scores candidates, but its branches are architectural experts rather than IL/RL objective branches.

## Evaluation Rules

When assessing a parallel IL/RL method, report:

- base IL score and refined single-plan score;
- whether final selection is learned, oracle, or simulator-based;
- candidate count and Best-of-N setting;
- whether the RL module was retrained for each base planner;
- inference latency and memory;
- per-metric regressions hidden by the aggregate score;
- sensitivity to deliberate perturbations of the reference trajectory.

PaIR-Drive reports the first four incompletely and does not report latency or reference-perturbation sensitivity. Its results establish promising modular refinement, not yet universal plug-and-play transfer.

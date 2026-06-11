---
title: Adaptive Routing for Trajectory Planning
type: concept
sources: [raw/papers/CLEAR_ Cognition and Latent Evaluation for Adaptive Routing in End-to-End Autonomous Driving.md]
related: [sources/clear.md, concepts/best-of-n.md, concepts/selection-based-planning.md, concepts/diffusion-planner.md, concepts/navsim-benchmark.md, concepts/foundation-backbones-for-ad.md]
created: 2026-06-11
updated: 2026-06-11
confidence: medium
---

# Adaptive Routing for Trajectory Planning

Adaptive routing is a planning pattern where the model changes its trajectory-generation budget and diversity level based on scene complexity, then selects among generated candidates with a learned scorer.

CLEAR is the first explicit instance tracked in this wiki: Qwen hidden states choose a discrete `(alpha, N)` scheme and score the resulting candidate trajectories.

## Core Pattern

```text
scene images + ego state + navigation command
          |
visual/geometric encoder + semantic hidden states
          |
Adaptive Scheduler -> choose diversity and sample count
          |
single-pass candidate generator
          |
learned candidate scorer -> final trajectory
```

The key distinction from ordinary multi-sample generation is that the candidate count is not fixed. Simple scenes can use high-precision, low-diversity generation; ambiguous scenes can allocate more candidates and more diversity.

## CLEAR Instantiation

CLEAR uses two routing knobs:

| Knob | Meaning | Low / high behavior |
| --- | --- | --- |
| `alpha` | Conditioning coefficient in VAE latent drift | low = diverse geometric coverage; high = expert-like precision |
| `N` | Number of generated candidates | low = cheaper inference; high = more coverage in complex scenes |

The scheduler predicts from a predefined grid of `(alpha, N)` schemes rather than regressing continuous values. Labels are generated offline by evaluating each scheme with the official PDMS scorer and choosing the best one per scene.

## Relationship to Best-of-N

Adaptive routing resembles Best-of-N because multiple candidates are generated before final selection. The difference is deployment feasibility:

| Property | Best-of-N | Adaptive routing |
| --- | --- | --- |
| Candidate count | fixed N | scene-dependent N |
| Selector | oracle PDMS simulator | learned scorer |
| Deployment status | diagnostic ceiling | intended inference path |
| Diversity control | stochastic sampling | explicit scheduler knobs |

This makes CLEAR closer to a learned selector system such as DreamerAD or HybridDriveVLA than to oracle BoN. It tries to close part of the BoN/selection gap without invoking NAVSIM at inference.

## Why It Matters

Driving scenarios vary sharply in ambiguity. A highway-following scene often needs one precise trajectory; a crowded unsignalized intersection may need several plausible futures before selection. Fixed compute budgets either waste effort on simple scenes or under-sample hard scenes. Adaptive routing exposes that trade-off as a learned policy.

## Open Questions

- Whether discrete scheme routing is enough, or continuous differentiable routing would capture better precision/diversity trade-offs.
- Whether PDMS-supervised routing overfits NAVSIM-specific scorer preferences.
- Whether learned scoring remains reliable outside non-reactive simulation, especially in interactive-agent settings.
- Whether adaptive candidate counts can be combined with fixed-vocabulary selectors such as DriveSuprim or with masked-diffusion refinement methods.

---
title: Adaptive Routing for Trajectory Planning
type: concept
sources: [raw/papers/Adaptive-WAM_ Quality-Guided Early-Exit Planningfrom Intermediate Video-Diffusion Features.md, raw/papers/CLEAR_ Cognition and Latent Evaluation for Adaptive Routing in End-to-End Autonomous Driving.md]
related: [sources/adaptive-wam.md, sources/clear.md, concepts/best-of-n.md, concepts/selection-based-planning.md, concepts/diffusion-planner.md, concepts/navsim-benchmark.md, concepts/foundation-backbones-for-ad.md]
created: 2026-06-11
updated: 2026-09-04
confidence: medium
---

# Adaptive Routing for Trajectory Planning

Adaptive routing is a planning pattern where the model changes its trajectory-generation budget and diversity level based on scene complexity, then selects among generated candidates with a learned scorer.

CLEAR is the first explicit instance tracked in this wiki: Qwen hidden states choose a discrete `(alpha, N)` scheme and score the resulting candidate trajectories.

Two instances are now tracked, and they route over different things. CLEAR varies the **candidate budget**; [[sources/adaptive-wam.md]] varies the **backbone depth actually executed**. The axes are orthogonal and composable, and no paper has combined them.

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

## A Second Routing Knob: Backbone Depth (Adaptive-WAM)

CLEAR routes over *how many candidates to generate*. [[sources/adaptive-wam.md]] routes over *how much of the network to execute* — an orthogonal and composable axis, and the first in this wiki that reduces compute rather than merely allocating it.

```text
current front image + ego state + navigation command
          |
one conditional forward through Wan2.2-5B  ──►  block 5 ──► head ──► trajectory ──► verifier ──► q >= eta ? return
                                          └──►  block 9 ──► head ──► trajectory ──► verifier ──► q >= eta ? return
                                          └──►  block 15 ─► ...                       (6 exits total)
          |
best trajectory accumulated so far
```

Rejected exits cost only the *unevaluated* blocks; hidden states and cached component scores are reused, so the six-exit worst case is one forward plus six heads rather than six forwards.

### The two routing patterns compared

| | [[sources/clear.md]] | [[sources/adaptive-wam.md]] |
|---|---|---|
| Routed quantity | candidate count `N` and diversity `alpha` | **DiT depth** (which of 6 exits) |
| Router input | Qwen VLM hidden states | decoded trajectory + current image (DINOv2-Small) |
| Router output | one scheme from a discrete grid | six evaluator components → threshold test |
| Decision timing | **once, before generation** | **incrementally, after each decoded plan** |
| Labels | offline PDMS evaluation of each scheme | evaluator component targets, soft-label BCE |
| Effect on compute | reallocates it | **reduces it** (190 → 170 ms mean) |
| Score | 93.7 PDMS | 90.79 PDMS |

**The incremental-vs-upfront distinction is the interesting one.** CLEAR commits to a budget before seeing any plan; Adaptive-WAM decides after seeing each one, so its router judges an *artifact* rather than a *scene*. That makes the router's job easier — verifying a concrete trajectory is a better-posed problem than predicting scene difficulty — but it also means the compute is spent before the decision to stop is available, which caps the achievable saving.

### What the numbers support

| Policy | PDMS | Exit by B15 | Latency |
|---|---:|---:|---:|
| Fixed B15 (best fixed exit) | 90.62 | 100% | 190 ms |
| Adaptive η=70 | 88.49 | 98.8% | **112 ms** |
| Adaptive η=80 | 90.64 | 95.2% | 143 ms |
| **Adaptive η=90** | **90.79** | 94.1% | 170 ms |
| Adaptive η=95 | 90.75 | 65.9% | 284 ms |

**+0.17 PDMS at 10% lower latency** over the strongest fixed exit. The quality gain is within plausible seed noise and should not be leaned on; the latency result is the defensible one, and the η=70 row is the control that makes it meaningful — being unconditionally shallow is fastest but costs 2.13 points, so the saving genuinely comes from conditioning.

**Two caveats this page should carry.** The latency figure is a *mean*: at η=95 routing costs 284 ms, worse than the 190 ms fixed baseline, so adaptive depth is not a latency *bound* — a relevant distinction for a safety-critical scheduler. And routing is only worth doing because no fixed depth dominates scene-wise: post-RL Jaccard overlap between exits' high-quality scene sets is 0.69–0.82, and the final block still beats the best block by ≥50 points on 422 scenes. Without that complementarity, the correct action would be to pick block 15 and stop.

### Why the two are composable

Nothing about depth routing conflicts with candidate routing. A system could choose depth per scene *and* candidate count per scene, and Adaptive-WAM's own ablation hints at the interaction: the advantage of good features (Wan over ViT-L) is 1.74 PDMS with one trajectory and only 0.28 with 64 proposals, so **the two knobs partly substitute for each other**. Spending on depth and spending on candidates buy overlapping things, which means a joint scheduler has a real trade-off to learn rather than two independent dials. No paper has built one.

## Why It Matters

Driving scenarios vary sharply in ambiguity. A highway-following scene often needs one precise trajectory; a crowded unsignalized intersection may need several plausible futures before selection. Fixed compute budgets either waste effort on simple scenes or under-sample hard scenes. Adaptive routing exposes that trade-off as a learned policy.

## Open Questions

- Whether discrete scheme routing is enough, or continuous differentiable routing would capture better precision/diversity trade-offs.
- Whether PDMS-supervised routing overfits NAVSIM-specific scorer preferences.
- Whether learned scoring remains reliable outside non-reactive simulation, especially in interactive-agent settings.
- Whether adaptive candidate counts can be combined with fixed-vocabulary selectors such as DriveSuprim or with masked-diffusion refinement methods.

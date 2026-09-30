---
title: Attention Masks in World-Action Models
type: concept
sources: ["raw/papers/MM-Future_ Multi-Mode Joint World–Action Modeling for Autonomous Driving.md", "raw/papers/ReDrive_ Shaping Representations with World Modeling for End-to-End Driving.md", "raw/papers/PhysWAM_ Physically Consistent World Action Model for Autonomous Driving.md", raw/papers/SimWAM_ A Simple World Action Model for End-to-End Autonomous Driving.md, "raw/papers/Metis_ A Generalizable and Efficient World-Action Model for Autonomous Driving and Urban Navigation.md", "raw/papers/SUV_ Future Scene Understanding as Video Generation for End-to-End Driving.md", raw/papers/WA-JEPA_ Rethinking the Video JEPA Paradigm forWorld-Action Modeling in Autonomous Driving.md, raw/papers/BrainWAM_ Action-Space Coordination of Semantic Priors and Predictive Dynamics for Autonomous Driving.md, raw/papers/UniDriveVLA_ Unifying Understanding, Perception, and Action Planning for Autonomous Driving.md, raw/papers/SGDrive_ Scene-to-Goal Hierarchical World Cognition for Autonomous Driving.md, raw/papers/DriveDreamer-Policy_ A Geometry-Grounded World–Action Model for Unified Generation and Planning.md, raw/papers/UniUGP_ Unifying Understanding, Generation, and Planing For End-to-end Autonomous Driving.md, raw/papers/DriveVA_ Video Action Models are Zero-Shot Drivers.md, raw/papers/Latent-WAM_ Latent World Action Modeling for End-to-End Autonomous Driving.md, raw/papers/Alpamayo-R1_ Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail.md]
related: [sources/mm-future.md, sources/redrive.md, sources/physwam.md, sources/simwam.md, sources/metis.md, sources/suv.md, sources/wa-jepa.md, sources/brainwam.md, sources/unidrivevla.md, sources/sgdrive.md, sources/drivedreamer-policy.md, sources/uniugp.md, sources/driveva.md, sources/latent-wam.md, sources/alpamayo-r1.md, concepts/world-model-for-ad.md, concepts/mixture-of-experts.md, concepts/navhard-ood-evaluation.md, concepts/evaluation-variance.md, concepts/inference-latency.md, concepts/perception-for-planning.md, research-directions.md]
created: 2026-09-27
updated: 2026-09-30
confidence: medium
---

# Attention Masks in World-Action Models

## What It Is

Most recent world-action models (WAMs) put observation tokens, future-prediction tokens and action tokens into **one shared attention space**, usually as a Mixture-of-Transformers (MoT) with per-expert weights. The **attention mask**, which token group may read which, then decides three things:
1. **What the planner can condition on at inference.**
2. **Whether future prediction can be dropped at test time**, and so the [latency](inference-latency.md).
3. **Which way gradients flow during training.**

It is a one-line design choice that the wiki's best-controlled comparisons now show matters, but mostly on navhard.

---

## The Design Space

For three token groups (observation O, future F, action A) with O always visible to everything, two edges are free: **A→F** (the action reads the future) and **F→A** (the future reads the action). A third, orthogonal choice is **stop-gradient** on an edge.

| Pattern | A reads F | F reads A | Future needed at inference | Used by |
|---|:-:|:-:|:-:|---|
| **Isolated** | ✗ | ✗ | No | [[sources/simwam.md]], SUV no-access, Metis ablation |
| **Future reads action** | ✗ | ✓ | No | [[sources/metis.md]] |
| **Future reads the *ground-truth* action** (teacher-forced; separate predictor module) | ✗ | ✓ (recorded trajectory, so no gradient reaches the planner) | No | [[sources/redrive.md]] Stage 2, in JEPA latent space |
| **Frozen future model reads the planner's rollout** | ✗ | ✓ (gradient flows *into* the planner only) | No | [[sources/redrive.md]] Stage 3 |
| **Action reads future** | ✓ | ✗ | Yes | [[sources/suv.md]], SimWAM ablation, [[sources/drivedreamer-policy.md]] (causal depth → video → action) |
| **Bidirectional** | ✓ | ✓ | Yes | Metis and SimWAM ablations, [[sources/driveva.md]]-style joint denoising, [[sources/physwam.md]] (video, metric depth and action all attend to one another) |
| **Bidirectional + stop-grad on F←A** | ✓ | ✓ (forward only) | Yes | [[sources/wa-jepa.md]] |

Related masks outside the future/action question:
- [[sources/unidrivevla.md]]: understanding → self; perception → understanding + self; action → both.
- [[sources/sgdrive.md]]: its block-wise mask forbids attention *between* the scene, agent and goal query blocks.
- [[sources/latent-wam.md]]: teacher-forced, block-causal over frames.
- [[sources/brainwam.md]]: **never mixes raw tokens at all.** Branches communicate through 8 compressed action tokens.

---

## The Controlled Evidence: One Backbone, Three Papers {#mask-family}

[[sources/simwam.md]], [[sources/metis.md]] and [[sources/suv.md]] share a backbone and recipe: Wan2.2-5B + a ~1B hidden-1024 action expert, joint flow matching with λ=1, 60 epochs, 8×H200, AdamW 1e-4.

| Mask | SimWAM v1 PDMS | Metis @320×384 v2 EPDMS / navhard | SUV v2 EPDMS / navhard |
|---|---:|---:|---:|
| Bidirectional | 90.2 | 87.4 / **28.0** | – |
| Action reads future | 90.1 | – | **91.0 / 36.9** |
| Isolated | **90.3** | 88.3 / 29.4 | 90.7 / 32.8 |
| Future reads action | – | **88.8 / 31.6** | – |

(SUV's rows include segmentation/depth/track supervision. Its RGB-only pair is 89.7 / 30.5 isolated → 90.6 / 35.0 with access.)

**Three findings:**
1. **On navtest the mask barely matters.** Every non-bidirectional pattern is within 0.5. This is the basis for SimWAM's "test-time imagination is unnecessary".
2. **On navhard it matters by several points.** Future-reads-action gives +2.2 over isolated (Metis). Action-reads-future gives +4.1 over isolated (SUV).
3. **Bidirectional is worst on navhard** (Metis, −1.4 vs isolated), even though it contains SUV's helpful A→F edge.

**Reconciling 2 and 3.** The harm plausibly comes from the **feedback loop**: F reads a noisy A, which then reads that same noisy F. Metis attributes it to "generation noise injected into the action space". [[sources/brainwam.md]] points the same way: its symmetric unmasked Tri-MoT scores 87.8, below its WAM-only branch at 88.1.

**A counterexample that constrains the hypothesis.** [[sources/wa-jepa.md]], the wiki's highest corrected v2 score (91.7), *is* bidirectional, with a **stop-gradient on the F←A edge**. So the loop exists in the forward pass, but future-prediction gradients never update the action stream. If WA-JEPA's design is sound, the harm in Metis's bidirectional variant would come from **gradient coupling, not the forward loop**. That is the opposite of Metis's claim, because Metis's asymmetric mask deliberately *lets* video-loss gradients into the action tokens. The two papers cannot both be right about the mechanism, and neither ran the stop-gradient ablation.

**The stop-gradient question, answered from a third direction.** [[sources/redrive.md]] is a future-reads-action design outside a shared-attention model: a separate predictor cross-attends to action tokens and regresses future JEPA latents. It resolves the gradient question by construction, twice, in opposite ways.
- *Stage 2*: the predictor reads the **recorded** trajectory. The future loss updates the encoder and has no path to the planner. This is stricter than both Metis (gradients reach the action tokens) and WA-JEPA (a stop-gradient on tokens the scene stream still reads). Worth +0.7 PDMS.
- *Stage 3*: encoder and predictor are frozen and the predictor reads the **planner's own rollout**. Now the future loss reaches *only* the planner. Worth +0.3 PDMS, bundled with two other rollout losses.

So ReDrive contains a clean instance of each of the two mechanisms Metis's mask mixes: future prediction shaping the *representation* (Stage 2) and future prediction shaping the *policy* (Stage 3). On navtest the first is about twice the second, and both are small. Neither is measured on navhard, where Metis's effect appears. ReDrive's navhard score (34.4) is above Metis's (32.2) with a different backbone.

**A second uncontrolled bidirectional data point.** [[sources/physwam.md]] is bidirectional on a different backbone (Cosmos 3 Nano), with no stop-gradient and no mask ablation. It reports **38.1 on navhard**, the best unscored result after SpanVLA, and above SUV's one-way 36.9 and Metis's 32.2. That does not overturn finding 3, which is a within-model comparison, but it shows a bidirectional model is not doomed on navhard. Two things differ from Metis's bidirectional variant and neither is isolated: a geometric loss that ties the generated depth to the generated motion through LiDAR, and a model about 2.5 times larger (15.2B against roughly 6B). What PhysWAM's ablation does show is that the *coupling through attention alone* buys nothing: adding a depth stream the action can read changes PDMS by −0.3.

**A controlled 2×2 on a different kind of model.** [[sources/mm-future.md]] crosses the mask with the block design in a latent world–action model (compact learned tokens, 32 hypotheses, a scorer on top). NAVSIM v1 PDMS:

| Block design | No future tokens | Action reads future | Bidirectional |
|---|---:|---:|---:|
| Single DiT | 92.3 | 92.4 | 92.3 |
| Modality-specific AdaLN and FFN, shared attention | – | 92.5 | **92.9** |

- On navtest every cell but one is within 0.2 of generating no future, which agrees with finding 1.
- The exception is bidirectional with modality-specific branches (+0.4 over one-way). That is the first controlled comparison here in which bidirectional is the best cell, against finding 3. It is a navtest result from single runs, and finding 3 is a navhard result.
- **The mask interacts with the block design**: the same edge is worth +0.4 in one architecture and −0.1 in the other. The Wan2.2 family's experiments all use separate experts, so this axis was not varied there.
- Here the future-reads-action edge has a second purpose. Each hypothesis's future is later read by a scorer, and a future that has seen its own trajectory is what makes it candidate-specific.
- Not measured on navhard or HUGSIM per variant.

---

## Why It Matters

- **Latency.** Only masks without A→F allow the future branch to be skipped at inference. That is the whole efficiency argument of SimWAM and Metis (see [[concepts/inference-latency.md]]).
- **RL.** Isolated and future-reads-action masks let RL run on the action expert without rolling out video (SimWAM's Flow-GRPO stage).
- **Benchmarks.** Mask effects are an instance of the wider pattern that world-model mechanisms are small on navtest and large on navhard ([[concepts/navhard-ood-evaluation.md]]).

---

## Open Questions

*Wiki-wide open questions are collected in [[research-directions.md]].*

- **The missing cells.** Run A→F only, F→A only and both, in one model, on navhard. This decides between the two partial results above.
- **Stop-gradient vs. no stop-gradient on F←A.** WA-JEPA blocks it; Metis relies on it. One ablation in either codebase would say whether gradient coupling helps (Metis) or hurts (WA-JEPA's implicit claim).
- **Does the navhard effect survive seeds?** Every number in the table is a single run over 244–450 Stage-1 scenes. See [[concepts/evaluation-variance.md]].
- **Does it hold outside Wan2.2-5B?** All controlled evidence comes from one video prior. WA-JEPA and BrainWAM use different backbones and are not controlled comparisons.
- **Would an isolated mask cost PhysWAM anything?** It pays 9.4 GPU-seconds per plan to generate a future its action tokens can read, and "an action-only inference path was not tested". Its geometric loss acts on clean estimates during training and does not need inference-time generation. An isolated or future-reads-action variant would show whether the loss or the access carries the navhard result, and would remove video and depth generation from inference.

---
title: nuScenes and Waymo Evaluations
type: concept
sources: [raw/papers/Adaptive-WAM_ Quality-Guided Early-Exit Planningfrom Intermediate Video-Diffusion Features.md, raw/papers/See Tomorrow, Act Today_ Foresight-Driven Autonomous Driving.md, raw/papers/AutoVLA_ A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning.md, raw/papers/HERMES_ A Holistic End-to-End Risk-Aware Multimodal Embodied System with Vision–Language Models for Long-Tail Autonomous Driving.md, raw/papers/UniUGP_ Unifying Understanding, Generation, and Planing For End-to-end Autonomous Driving.md, raw/papers/Reasoning-VLA_ A Fast and General Vision-Language-Action Reasoning Model for Autonomous Driving.md, raw/papers/DriveVA_ Video Action Models are Zero-Shot Drivers.md, raw/papers/ExploreVLA_ Dense World Modeling and Exploration for End-to-End Autonomous Driving.md, raw/papers/OneDrive_ Unified Multi-Paradigm Driving with Vision-Language-Action Models.md, raw/papers/From Forecasting to Planning_ Policy World Model for Collaborative State-Action Prediction.md, raw/papers/Driving Intents Amplify Planning-Oriented Reinforcement Learning.md, raw/papers/SimWAM_ A Simple World Action Model for End-to-End Autonomous Driving.md]
related: [sources/adaptive-wam.md, sources/foresight.md, concepts/navsim-benchmark.md, concepts/bench2drive.md, concepts/world-model-for-ad.md, concepts/intent-conditioned-planning.md, concepts/best-of-n.md, concepts/physicalai-av-benchmark.md, sources/autovla.md, sources/hermes.md, sources/uniugp.md, sources/reasoning-vla.md, sources/driveva.md, sources/explorevla.md, sources/onedrive.md, sources/policy-world-model.md, sources/dial.md, sources/drivewam.md, sources/simwam.md]
created: 2026-05-01
updated: 2026-09-04
confidence: high
---

## What They Measure

nuScenes and Waymo-style evaluations in this wiki are mostly open-loop: L2 displacement, collision proxy metrics, planning error, or WaymoE2E risk/route scores. They are useful for trajectory imitation and transfer, but they do not replace closed-loop NAVSIM or interactive Bench2Drive evaluation.

## Common Metrics

| Metric family | Typical use | Caveat |
| --- | --- | --- |
| L2/ADE/FDE | nuScenes trajectory accuracy | Rewards matching logged behavior, not necessarily safe closed-loop behavior. |
| Collision rate | nuScenes/Waymo proxy safety | Often computed against logged agents without reactive simulation. |
| RFS | WaymoE2E long-tail risk | Dataset/task-specific; not comparable to PDMS or DS. |
| FID/FVD | World-model visual quality | Video realism does not guarantee planning quality. |

## WOD-E2E Rater Feedback Score

[[sources/dial.md]] uses WOD-E2E RFS, which scores a predicted trajectory against up to three human-rated alternative trajectories rather than treating the logged path as the unique target. This makes RFS useful for detecting proposal support: a policy may generate a path preferred over the logged demonstration.

DIAL reports the logged trajectory at RFS 8.13 and an intent-pooled Best-of-128 ceiling of 9.14. That does not mean the model deploys at 9.14; oracle selection is required. Its intent-classified deployment-oriented held-out peak is 8.211.

Protocol caveats:

- RL uses 338 of the 438 labeled validation sequences.
- The remaining 100 sequences are used to select checkpoints and reward hyperparameters, so “held-out” means validation, not untouched test.
- “Full RFS” includes RL-training sequences.
- Standard RFS scores 3 s and 5 s anchors with a hard maximum over raters; DIAL uses denser anchors and label-softmax aggregation only during training.
- RFS is open-loop preference alignment and does not directly measure reactive collision avoidance or closed-loop stability.
- The paper tabulates `TR` but the available source extraction does not define it.

## Takeaways

- Treat nuScenes/Waymo as complementary evidence for generalization, not as direct leaderboard substitutes for NAVSIM or Bench2Drive.
- Zero-shot transfer claims should report absolute values, not only percent improvement over one baseline. (DriveVA's paper did not; SimWAM's Table 6 supplied them later — 0.84 L2 / 0.06 collision.)
- World-model papers need both generation metrics and downstream planning metrics; strong FVD alone is insufficient.
- The same caveats extend to the newer, much larger [[concepts/physicalai-av-benchmark.md]], which also reports ADE/FDE only. Scale improves coverage of rare events but does not convert an open-loop displacement metric into evidence about closed-loop behavior — and where a paper curates its own test subset (as [[sources/drivewam.md]] does), the comparison is not yet leaderboard-grade.

## OneDrive nuScenes Result

**OneDrive** ([[sources/onedrive.md]]) reports one of the strongest nuScenes open-loop planning entries in the wiki:

| Method | L2 Avg | Collision Avg | Notes |
| --- | --- | --- | --- |
| SOLVE-VLM | 0.28 | 0.20 | AR/text VLM path |
| ColaVLA | 0.30 | 0.23 | Non-AR baseline |
| **OneDrive** | **0.28** | **0.18** | Single causal decoder; detection/lane/planning query sequence |

This is meaningful evidence for the architecture, but it remains open-loop: the result should not be treated as equivalent to NAVSIM PDMS or Bench2Drive driving score.

## Zero-Shot NAVSIM → nuScenes: The WAM Cluster

[[sources/simwam.md]]'s Table 6 is the wiki's first side-by-side of NAVSIM-trained world-action models evaluated on nuScenes **without fine-tuning or auxiliary supervision**, and it resolves a gap this page previously flagged: DriveVA's paper reported only percentage improvements over PWM, never absolutes.

| Method | Finetuned | L2 Avg ↓ | Collision Avg ↓ |
| --- | --- | ---: | ---: |
| UniAD (reference, finetuned) | ✓ | 1.03 | 0.31 |
| GenAD (reference, finetuned) | ✓ | 0.91 | 0.43 |
| Epona (reference, finetuned) | ✓ | 1.25 | 0.36 |
| DriveVA | ✗ | **0.84** | 0.06 |
| DriveWAM | ✗ | 0.96 | 0.06 |
| SimWAM | ✗ | 0.96 | **0.04** |

Two things stand out. All three zero-shot WAMs beat every finetuned baseline on collision rate by roughly an order of magnitude, which is the strongest evidence in the wiki that video-prior training transfers as a *safety* prior rather than a trajectory-matching one. And the L2/collision split is stark: SimWAM ties DriveWAM on L2 (0.96) while halving collisions (0.04 versus 0.06), and DriveVA leads on L2 (0.84) without leading on collisions. This is the clearest illustration on this page of why L2 and collision rate should not be collapsed into one ranking — L2 rewards agreement with the logged nuScenes expert, which a NAVSIM-trained policy has no reason to reproduce.

**Caveat**: the DriveWAM row is not corroborated by [[sources/drivewam.md]], whose ingested v1 clipping contains no nuScenes evaluation at all. Either SimWAM reproduced it or cited a later revision.

## Policy World Model nuScenes Result

**Policy World Model** ([[sources/policy-world-model.md]]) reports a safety-skewed nuScenes result: it does not dominate L2, but it has the lowest collision rate in its comparison table.

| Method | Ego status | L2 Avg | Collision Avg | Notes |
| --- | --- | ---: | ---: | --- |
| PWM | No | 0.78 | 0.07 | Better collision than Drive-OccWorld 0.11 and LAW 0.19; worse L2 than those methods. |
| PWM | Yes | 0.41 | 0.04 | Best collision in the paper's ego-status table; L2 trails Omni-Q 0.33, BEV-Planner 0.35, and VAD-Base 0.37. |

The result is useful evidence for future-frame forecasting as a safety prior. It should still be interpreted as open-loop nuScenes evidence, not as proof of closed-loop behavior under interactive agents.

## ForeSight: Hedging the Benchmark While Reporting On It

[[sources/foresight.md]] is a useful specimen of a pattern this page exists to name. Its nuScenes paragraph opens by conceding that "the scenarios and evaluation protocols in nuScenes are relatively simple and the metrics are not entirely comprehensive," citing the ego-status critique, then reports its own result on those metrics as "competitive performance."

| Method | Type | L2 Avg ↓ | Collision Avg ↓ |
| --- | --- | ---: | ---: |
| BEV-Planner | Planning | **0.46** | 0.49 |
| PARA-Drive | Planning | 0.48 | 0.25 |
| World4Drive | Planning + WM | 0.50 | 0.16 |
| GenAD | Planning | 0.52 | 0.19 |
| BridgeAD | Planning | 0.59 | 0.09 |
| MomAD | Planning | 0.60 | 0.09 |
| SparseDrive | Planning | 0.61 | **0.08** |
| LAW | Planning + WM | 0.61 | 0.30 |
| **ForeSight** | **Planning + WM** | **0.62** | **0.18** |
| UniAD | Planning | 0.69 | 0.12 |
| VAD-Base | Planning | 0.72 | 0.22 |

ForeSight wins no column. It is eighth of eleven on average L2 and sixth on average collision, and it is beaten on **both** by World4Drive — the other world-model entry in its own table. This is the same method that leads its NAVSIM category at 89.3 PDMS, which is the most direct illustration on this page of the divergence between the two benchmarks: **a design that helps under NAVSIM's closed-loop PDM scoring can be neutral-to-negative under nuScenes L2**, because L2 rewards reproducing the logged expert and a 2.5B generated-future prior has no particular reason to do that.

The hedge is fair on the merits — the ego-status critique is real and this page endorses it. But a paper that believes the metric is uninformative should say its result is uninformative, not that it is competitive.

**Secondary finding (Table 8)**: swapping the world model from Epona to Vista, planner fixed, degrades 6 of 8 columns (L2 0.62 → 0.64, collision 0.18 → 0.27). Presented as evidence of architecture-agnosticism, which it is, in the weak sense that the framework tolerates the swap rather than benefiting from it.

## The Zero-Shot WAM Cluster, Extended

[[sources/adaptive-wam.md]] adds a fourth NAVSIM-trained WAM evaluated on nuScenes without fine-tuning, and it is the first to report the full horizon breakdown alongside DriveVA rather than percentage deltas.

| Method | FT | L2 1s | L2 2s | L2 3s | **L2 Avg** | Col 1s | Col 2s | Col 3s | **Col Avg** |
| --- | :-: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| UniAD | yes | 0.48 | 0.96 | 1.65 | 1.03 | 0.05 | 0.17 | 0.71 | 0.31 |
| GenAD | yes | 0.36 | 0.83 | 1.55 | 0.91 | 0.06 | 0.23 | 1.00 | 0.43 |
| Epona | yes | 0.61 | 1.17 | 1.98 | 1.25 | 0.01 | 0.22 | 0.85 | 0.36 |
| DriveVLA-W0 | no | 0.43 | 1.26 | 2.60 | 1.43 | 0.22 | 0.66 | 1.42 | 0.77 |
| PWM | no | 2.06 | 3.91 | 6.00 | 3.99 | 0.12 | 0.15 | 0.86 | 0.36 |
| DriveVA | no | **0.33** | 0.76 | **1.43** | **0.84** | **0.00** | **0.07** | **0.12** | **0.06** |
| **Adaptive-WAM** | no | 0.35 | **0.71** | 1.58 | 0.88 | **0.00** | 0.09 | 0.15 | 0.08 |

Two observations. **The horizon profile differs from the average**: Adaptive-WAM leads DriveVA at 2s and trails it at 3s, so the 0.84-vs-0.88 average conceals a crossover rather than uniform dominance — another reason to distrust horizon-averaged L2 as a single ranking. And **the collision-rate story from the SimWAM cluster holds and strengthens**: both fully-generative and early-exit WAMs land at 0.06-0.08% average collision against 0.31-0.43% for fine-tuned nuScenes-native planners, roughly an order of magnitude, without ever seeing the target domain.

The efficiency contrast is worth carrying here too. DriveVA achieves its 0.84 by executing the full Wan backbone and generating future images; Adaptive-WAM reaches 0.88 with a single conditional forward to an intermediate block at **170 ms**. On this benchmark the extra 12+ seconds of video synthesis buys 0.04 m and 0.02 percentage points.

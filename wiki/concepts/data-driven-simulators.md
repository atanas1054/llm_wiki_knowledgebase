---
title: Data-Driven Driving Simulators
type: concept
sources: ["raw/papers/Learning to Drive from a World Model.md", "raw/papers/Senna-2_ Aligning VLM and End-to-End Driving Policy for Consistent Decision Making and Planning.md", "raw/papers/DreamerAD_ Efficient Reinforcement Learning via Latent World Model for Autonomous Driving.md", "raw/papers/Qwen-Drive-1.0_ An Initial Step towards a Vision-Language Foundation Model for Autonomous Driving.md", "raw/papers/How Can Driving World Models Do Counterfactual Prediction_.md", "raw/papers/PhysWAM_ Physically Consistent World Action Model for Autonomous Driving.md", "raw/papers/AD-E2E-JEPA_ A Joint-Embedding Predictive Architecture For End-to-End Autonomous Driving.md"]
related: [sources/learning-to-drive-from-a-world-model.md, sources/senna2.md, sources/dreameraD.md, sources/qwen-drive-1.0.md, sources/driving-wm-counterfactuals.md, sources/physwam.md, sources/ad-e2e-jepa.md, concepts/world-model-for-ad.md, concepts/rl-for-ad.md, concepts/navhard-ood-evaluation.md, concepts/hugsim-benchmark.md, concepts/alpasim-benchmark.md, concepts/bench2drive.md, concepts/counterfactual-prediction.md, concepts/teacher-pseudo-labels.md, concepts/reasoning-faithfulness.md, research-directions.md]
created: 2026-10-02
updated: 2026-10-02
confidence: medium
---

# Data-Driven Driving Simulators

## What It Is

Simulators whose frames come from **recorded real driving** rather than from hand-built 3D assets. They let a planner see the observations its own actions would produce while keeping real-world appearance.

[[sources/learning-to-drive-from-a-world-model.md]] splits any driving simulator into two parts, and this page uses the split:
1. a **state generator**, which produces the next observation given where the ego vehicle now is, and
2. a **supervision source**, which says what the policy should have done (a label) or how well it did (a reward).

The wiki has data-driven simulators in two roles:

| Role | Simulators |
|---|---|
| **Evaluation** | [[concepts/navhard-ood-evaluation.md]] (3DGS, two stages), [[concepts/hugsim-benchmark.md]] (Gaussian-splatting reconstructions of four datasets), [[concepts/alpasim-benchmark.md]] (PAI-AV-NuRec over 916 real-log scenarios) |
| **Training** | [[sources/learning-to-drive-from-a-world-model.md]] (reprojection, and a learned world model), [[sources/senna2.md]] (3DGS, 1,044 training clips), [[sources/dreameraD.md]] (latent world model). RAD, RL in 3DGS, is cited on [[concepts/rl-for-ad.md]] but not ingested |

Synthetic game-engine simulators are the contrast: CARLA for [[concepts/bench2drive.md]], and MetaDrive, which the openpilot paper uses only as a test. They give full control and reacting agents at the cost of appearance.

---

## Why Train On-Policy {#why-on-policy}

Behaviour cloning trains only on states the human visited. At deployment the policy's own small errors move it to states it never saw, and the errors compound. A simulator fixes this by letting the policy train on its own states.

**The cleanest single-paper evidence in the wiki** is Table 2 of [[sources/learning-to-drive-from-a-world-model.md]]. The frozen feature extractor is shared; only the temporal model's training differs.

| Policy | Closed-loop lane centre | Closed-loop lane change | Open-loop trajectory MAE |
|---|---:|---:|---:|
| Behaviour cloning | 5/24 | 8/20 | **0.361** |
| On-policy, reprojection | 24/24 | 20/20 | 0.369 |
| On-policy, world model | 24/24 | 19/20 | 0.394 |

The open-loop metric ranks the three in reverse order of the closed-loop tests. The evaluation-side version of the same inversion is [[sources/qwen-drive-1.0.md]]'s AlpaSim reproduction, in which the higher-PDMS world-action model is last on at-fault score ([[concepts/alpasim-benchmark.md#inversion]]).

---

## Three Ways to Make the Next Frame {#state-generators}

| Family | Mechanism | Other road users | Distance from the log | In the wiki |
|---|---|---|---|---|
| **Reprojection** | Depth map + reprojection to the new pose + inpainting | Replayed pixels; **cannot react** | "Typically less than 4m" before artifacts dominate | openpilot's first simulator; the transport stage of [[sources/driving-wm-counterfactuals.md]] (lift with monocular depth, splat to the new camera) |
| **Reconstruction** (3DGS, neural reconstruction) | Fit a scene per log, render novel views | Can be scripted to react ([[sources/senna2.md]]; AlpaSim) | Not measured in any ingested paper | navhard, HUGSIM, AlpaSim; Senna-2's training environment |
| **Learned world model** | Generate the next frame or latent from history and ego motion | Can in principle be generated; never controlled or verified in the wiki | Not bounded by geometry; fidelity to the commanded motion is partial | openpilot's second simulator; [[sources/dreameraD.md]] (latent only) |

**Reprojection and the world model are the two halves of abduction.** [[concepts/counterfactual-prediction.md#what-abduction-can-and-cannot-recover]] assigns observed surfaces to geometry and unobserved regions to a generative prior. openpilot's reprojective simulator already combines the two in small form (reprojection plus inpainting); its world-model simulator hands everything to the prior.

---

## Where Supervision Comes From {#supervision}

| Source | Signal | Example |
|---|---|---|
| The logged human trajectory | Labels, valid only on the logged states | Every behaviour-cloning baseline |
| **A Plan Model conditioned on the recorded future** | Action labels on the policy's own states, steering back toward where the human went | [[sources/learning-to-drive-from-a-world-model.md]] |
| The benchmark's own metric as reward | A scalar per rollout | NAVSIM GRPO papers ([[concepts/rl-for-ad.md]]) |
| A learned reward over predicted latents | Dense rewards per horizon | [[sources/dreameraD.md]] |
| Rule-based penalties on rollouts | Push the trajectory when TTC < 3 s or speed is too low | [[sources/senna2.md]] |

**Offline, the future is known.** A supervision source may look at it; the policy may not. Three papers in the wiki use the recorded future:

| Paper | How the future is used | Status |
|---|---|---|
| [[sources/learning-to-drive-from-a-world-model.md]] | Conditions a teacher that labels a past-only student | Legitimate: training only |
| [[sources/ad-e2e-jepa.md]] | Goal for a search over world-model rollouts at test time | An oracle, not a planner |
| Hindsight reasoning traces ([[concepts/reasoning-faithfulness.md#hindsight]]) | Written with the outcome in view, then imitated | Leaks the answer into the rationale |

The first is the only one in which looking at the future is part of a valid deployable recipe. Its price is that anything the teacher learns from the future, such as whether to change lanes, must reach the student some other way. openpilot uses an explicit lane-change impulse.

**On-policy imitation needs no reward.** openpilot's Plan Model labels replace a reward entirely, and lane keeping and lane changes come out of it. No paper in the wiki compares future-anchored labels against a reward-based RL objective in the same simulator.

---

## Failure Modes {#failure-modes}

The list is from Section 3.1 of [[sources/learning-to-drive-from-a-world-model.md]], which concerns reprojection. The other columns record where the same problem appears elsewhere in the wiki.

| Failure | Reprojection | Learned world model | Elsewhere in the wiki |
|---|---|---|---|
| **Static world** ("the counterfactual problem") | Other drivers cannot react | Can react in the gap, but openpilot's future anchor forces the rollout back to the recorded state | Evidence transport "preserves behaviour the counterfactual action would have changed" ([[concepts/counterfactual-prediction.md]]) |
| **Range** | Artifacts grow with the offset; kept under 4 m, worst longitudinally | Not range-limited by geometry | The open question of whether navhard's Stage-2 lane-keeping collapse is a 3DGS rendering effect ([[concepts/navhard-ood-evaluation.md]]) |
| **Lighting and reflections** | No light transport; night scenes smear (openpilot Figure 3) | Not reported | — |
| **Following the commanded motion** | Exact by construction | **Partial**: a commanded ±0.5 m lateral deviation is rendered "not to its full extent" | [[sources/physwam.md]]: 0.80° median yaw disagreement for a joint model (floor 0.29°); [[sources/ad-e2e-jepa.md]]'s hit rate (53.8% top-1 among 257) |
| **Shortcut learning** | Artifacts correlate with the pose offset, which is what the corrective label encodes; the policy reads them | Not discussed; the partial pose-following above suggests its errors may also correlate with the offset | No other training paper in the wiki reports a test for it |
| **Vehicle dynamics** | Handled outside the renderer by a randomized vehicle model | openpilot conditions the world model on poses, not actions, so the vehicle model can change without retraining it | — |

### Shortcut learning {#shortcut}

The shortcut problem is the one specific to training. A renderer whose error depends on how far the ego is from the log gives the policy a feature that encodes that distance, and the distance is what the label is about. openpilot's remedy is an information bottleneck: white Gaussian noise caps the feature extractor's output at about 700 bits. No experiment in the paper measures either the exploitation or the remedy.

---

## Evidence So Far {#evidence}

| Paper | State generator | Supervision | Training-side result | Deployed |
|---|---|---|---|---|
| [[sources/learning-to-drive-from-a-world-model.md]] | Reprojection | Future-anchored Plan Model | MetaDrive 24/24 lane centre, 20/20 lane change | openpilot: 27.63% of time / 48.10% of distance engaged |
| [[sources/learning-to-drive-from-a-world-model.md]] | World model (500M DiT, 400k one-minute segments) | Future-anchored Plan Model | MetaDrive 24/24, 19/20 | openpilot: 29.92% / 52.49% |
| [[sources/senna2.md]] | 3DGS, 1,044 training clips | Rule penalties (hierarchical) | Its own 3DGS benchmark: at-fault collision rate 0.118 → 0.077 from the closed-loop stage | No |
| [[sources/dreameraD.md]] | Latent world model (Epona) | Learned reward, GRPO | 87.7 EPDMS NAVSIM-v2 | No |

- **Every row is tested differently.** No paper compares two state-generator families under one supervision source and one test, except openpilot, whose two simulators are not separated by its tests.
- **The field numbers are not a controlled comparison.** The two openpilot cohorts differ in trip count (47,047 against 40,026) and the paper does not describe how users were assigned; see [[concepts/evaluation-variance.md#outside-navsim]].
- **Only openpilot reports real-world use.** It is also the only row restricted to lateral control.

---

## Open Questions

*Wiki-wide open questions are collected in [[research-directions.md]].*

- **Which family trains the best policy?** openpilot's reprojective and world-model simulators tie on lateral tests. The longitudinal regime, where reprojection is limited to a few metres, is untested.
- **Do policies trained in rendered simulators learn the renderer's artifacts?** Asserted by openpilot for reprojection and countered with a 700-bit bottleneck, but not measured. A direct test: train with and without the bottleneck and probe the policy's features for the pose offset.
- **Is a reward needed?** Future-anchored labels and reward-based RL have never been compared in one simulator.
- **Does simulator quality predict policy quality?** openpilot shows LPIPS improving with world-model size and data (in a figure not in the clipping) but trains policies with one world model only.
- **Can a learned simulator make other agents react without being pulled back to the log?** openpilot's anchor guarantees recovery pressure and also forces the world to the recorded state at the anchor.
- **Is the evaluation-side effect the same mechanism?** If 3DGS renderings degrade with the ego offset as reprojections do, part of navhard's Stage-2 collapse measures the renderer.

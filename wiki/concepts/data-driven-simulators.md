---
title: Data-Driven Driving Simulators
type: concept
sources: ["raw/papers/RoG-DAgger_ Rollout-Guided Post-Training for End-to-End Driving.md", "raw/papers/Driving on Registers.md", "raw/papers/ManeuverFlow_ Maneuver Hypothesis Generation with Future Evidence for Trajectory Evaluation and Refinement in End-to-End Autonomous Driving.md", "raw/papers/SimVTA_ Simulation-Grounded Video-Trajectory Alignment for Driving World Action Models.md", "raw/papers/RAD_ Training an End-to-End Driving Policy via Large-Scale 3DGS-based Reinforcement Learning.md", "raw/papers/Think2Drive_ Efficient Reinforcement Learning by Thinking with Latent World Model for Autonomous Driving (in CARLA-v2).md", "raw/papers/SimScale_ Learning to Drive via Real-World Simulation at Scale.md", "raw/papers/Learning to Drive from a World Model.md", "raw/papers/Senna-2_ Aligning VLM and End-to-End Driving Policy for Consistent Decision Making and Planning.md", "raw/papers/DreamerAD_ Efficient Reinforcement Learning via Latent World Model for Autonomous Driving.md", "raw/papers/Qwen-Drive-1.0_ An Initial Step towards a Vision-Language Foundation Model for Autonomous Driving.md", "raw/papers/How Can Driving World Models Do Counterfactual Prediction_.md", "raw/papers/PhysWAM_ Physically Consistent World Action Model for Autonomous Driving.md", "raw/papers/AD-E2E-JEPA_ A Joint-Embedding Predictive Architecture For End-to-End Autonomous Driving.md"]
related: [sources/rog-dagger.md, sources/drivor.md, sources/maneuverflow.md, sources/simvta.md, sources/rad.md, sources/think2drive.md, sources/simscale.md, sources/learning-to-drive-from-a-world-model.md, sources/senna2.md, sources/dreameraD.md, sources/qwen-drive-1.0.md, sources/driving-wm-counterfactuals.md, sources/physwam.md, sources/ad-e2e-jepa.md, concepts/world-model-for-ad.md, concepts/rl-for-ad.md, concepts/navhard-ood-evaluation.md, concepts/hugsim-benchmark.md, concepts/alpasim-benchmark.md, concepts/bench2drive.md, concepts/counterfactual-prediction.md, concepts/teacher-pseudo-labels.md, concepts/reasoning-faithfulness.md, research-directions.md]
created: 2026-10-02
updated: 2026-10-28
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
| **Evaluation** | [[concepts/navhard-ood-evaluation.md]] (3DGS, two stages), [[concepts/hugsim-benchmark.md]] (Gaussian-splatting reconstructions of four datasets), [[concepts/alpasim-benchmark.md]] (PAI-AV-NuRec over 916 real-log scenarios), [[concepts/worldengine-benchmark.md]] (World Engine 3DGS, 288 high-risk navtest scenarios, logged or IDM traffic; first used in [[sources/maneuverflow.md]]) |
| **Training** | [[sources/learning-to-drive-from-a-world-model.md]] (reprojection, and a learned world model), [[sources/senna2.md]] (3DGS, 1,044 training clips), [[sources/dreameraD.md]] (latent world model), [[sources/simscale.md]] (3DGS over navtrain, 384K rendered off-log scenes stored as a dataset), [[sources/rad.md]] (3DGS, 3,968 training clips, PPO with log-replayed agents), [[sources/simvta.md]] (3DGS World Engine; renders are video *targets* for a world-action model during RL, never the policy's input) |

Synthetic game-engine simulators are the contrast: CARLA for [[concepts/bench2drive.md]], and MetaDrive, which the openpilot paper uses only as a test. They give full control and reacting agents at the cost of appearance. *(2026-10-06: [[sources/think2drive.md]] combines the two ideas the other way round: it learns a latent world model **of CARLA** (privileged BEV masks, not images) and trains its planner in that model, while CARLA keeps supplying new experience. Its wrapper also documents a cost real-log simulators avoid: a CARLA v2 reset takes 40–60 s, hidden by asynchronous parallel environments.)*

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

**The game-engine version of the argument** *(2026-10-28)*. [[sources/rog-dagger.md]] shows the distribution shift directly in CARLA. Among frames with time to collision under 2 s, about 17.5% of a camera policy's Bench2Drive frames fall below 0.1 s, against about 0.1% of its training frames. One round of DAgger on its own states doubles its Longest6 v2 score (22 → 44 DS) and adds +3.75 DS on Bench2Drive. That needs what a game engine gives for free: ground-truth actor states for a privileged expert, and a rendered observation for every state the policy reaches. The paper names AlpaSim as the next host. Of the families below, reconstruction provides the first ([[sources/simscale.md]] runs PDM-Closed on ground-truth state in 3DGS), and every family provides the second only within its range.

### Fixed Perturbations Instead of Policy Rollouts {#fixed-perturbations}

[[sources/simscale.md]] gets off-log states a different way. It does not run the policy. It moves the ego along trajectories sampled from a vocabulary around the log (±20 m, ±2 m, ±20°, kept only if EPDMS ≥ 0.8), lets IDM agents react, labels the end state with a pseudo-expert, renders everything once and stores it.

| | On-policy ([[sources/learning-to-drive-from-a-world-model.md]]) | Fixed perturbations ([[sources/simscale.md]]) |
|---|---|---|
| Where the states come from | The current policy | A sampler around the log, independent of any policy |
| Depth | Multi-second rollouts, until the anchor | One perturbation step ($H$ = 4 s), then one labelled step |
| Reuse | Tied to one policy and one training run | A dataset any planner can be trained on (three paradigms tested) |
| Test | MetaDrive pass counts; deployment | NAVSIM navhard and navtest |

- **It works without being on-policy.** All ten simulation-trained configurations in SimScale beat their real-only baselines on navhard, by +5.1 to +8.6 in its tables. *(2026-10-14: and for a fourth planner family outside the paper: [[sources/drivor.md]], a ViT-S register planner, gains +4.0 navhard EPDMS with 65k of SimScale's scenes and +6.3 with 134k.)*
- **What is lost**: the states a trained policy actually drifts into are not targeted. SimScale lists this as a limitation and points to self-evolving collection.
- **No paper compares the two samplers** under one renderer and one teacher.

### The Renderer as a Source of Targets, Not Observations {#video-targets}

*(2026-10-14)* [[sources/simvta.md]] uses a 3DGS simulator in a role none of the other training papers here does. *(The same simulator, World Engine, is also a closed-loop benchmark, used by [[sources/maneuverflow.md]]; see [[concepts/worldengine-benchmark.md]]. A policy trained on its renders and tested on its benchmark would face the train–test confound described [below](#shortcut).)* The policy, an Epona world-action model, sees only logged images. Its reward comes from NAVSIM's PDM scorer on the executed path. The renderer (World Engine) produces the front-view future along each of the four best safe executed paths per scene, and those renders are the targets of the model's video-prediction loss during GRPO. The video head is dropped at inference; what the renders change is the shared encoder.

| | Off-log states as policy input ([[sources/simscale.md]], [[sources/rad.md]]) | Off-log futures as world-model targets ([[sources/simvta.md]]) |
|---|---|---|
| What the renderer produces | The observation the policy acts on | The future the video head must predict, given the logged history and an executed path |
| What supervises the action | Pseudo-expert labels; PPO rewards | Group-relative PDM reward on the executed path |
| Where renders enter the model | The whole perception stack, at train time | The shared encoder, through a video loss at 0.2 gradient scale; also as teacher-forced history frames |
| Policy at test time sees renders? | In 3DGS benchmarks only | Same: never on NAVSIM, yes on HUGSIM |

**It also gives the first fixed-versus-on-policy comparison under one renderer.** With everything else equal, renders of initial-policy trajectories cached once and reused (90.4 PDMS) against renders of the current policy's trajectories each round (91.3); no video at all 89.1. The cache is policy-independent after the first round, like [[sources/simscale.md]]'s perturbations, though it comes from a policy rather than a perturbation sampler, and it costs one rendering pass instead of one per round.

**And it bounds how far off the log it goes.** Targets are rendered only for candidates that score at least 0.999 on NC, DAC, TTC and DDC and stay inside per-step pose limits ($0\le\Delta x\le16$ m, $|\Delta y|\le1.5$ m, $|\Delta\mathrm{yaw}|\le32^\circ$ per 0.5 s). The distribution of distances from the log is not reported, and renderer fidelity (FID 7.6, FVD 21.8 against real frames) is measured only along logged trajectories. Its Figure 4, which renders two planners' off-log paths, shows a bus and a road shoulder smeared at +3 to +4 s.

---

## Three Ways to Make the Next Frame {#state-generators}

| Family | Mechanism | Other road users | Distance from the log | In the wiki |
|---|---|---|---|---|
| **Reprojection** | Depth map + reprojection to the new pose + inpainting | Replayed pixels; **cannot react** | "Typically less than 4m" before artifacts dominate | openpilot's first simulator; the transport stage of [[sources/driving-wm-counterfactuals.md]] (lift with monocular depth, splat to the new camera) |
| **Reconstruction** (3DGS, neural reconstruction) | Fit a scene per log, render novel views | Can be scripted to react ([[sources/senna2.md]]; AlpaSim; [[sources/simscale.md]] with IDM); [[sources/rad.md]] supports IDM but trains against log replay | Not measured as a function of offset. [[sources/simscale.md]] perturbs up to ±20 m longitudinal, ±2 m lateral and ±20° heading, drops blocks whose novel-view PSNR is below 27, and filters "unstable renderings" | navhard, HUGSIM, AlpaSim; Senna-2's training environment; SimScale's data engine (MTGS) |
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
| A learned reward and termination head, trained on a game engine's shaped reward | Reward and episode end for each imagined step | [[sources/think2drive.md]] (CARLA, privileged BEV) |
| Rule-based penalties on rollouts | Push the trajectory when TTC < 3 s or speed is too low | [[sources/senna2.md]] |
| Collision and expert-deviation penalties that end the episode, plus directional auxiliary losses | PPO advantages per event type; rule-chosen directions (brake, steer away, steer back) | [[sources/rad.md]] |
| **Retrieval from the log** | The logged human trajectory whose start and end best match the perturbed state and the recorded state $H$ later; steers back to the log | [[sources/simscale.md]] (recovery expert, 103,288 candidates) |
| **A privileged rule-based planner** | PDM-Closed run on ground-truth state, kept only if every EPDMS sub-metric is 1 | [[sources/simscale.md]] (planner expert) |
| **The benchmark's sub-scores on simulated states** | One label per vocabulary trajectory per simulated scene, no expert trajectory | [[sources/simscale.md]] (reward-only, for a vocabulary scorer) |
| **Rendered futures along the policy's own executed paths** | A video-prediction target for each explored trajectory; the action itself is supervised by the PDM reward | [[sources/simvta.md]] (flow matching on the higher-error half of four safe, high-reward renders per scene) |

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
| **Static world** ("the counterfactual problem") | Other drivers cannot react | Can react in the gap, but openpilot's future anchor forces the rollout back to the recorded state | Evidence transport "preserves behaviour the counterfactual action would have changed" ([[concepts/counterfactual-prediction.md]]). In 3DGS, [[sources/simscale.md]] makes agents react through IDM and measures it: reactive data beats non-reactive by +0.7 / +1.1 navhard EPDMS with fewer samples (120K vs 141K). [[sources/rad.md]] trains PPO against log-replayed agents; one of its auxiliary losses pushes the ego to accelerate when struck from behind, the training-time signature of traffic that cannot brake |
| **Range** | Artifacts grow with the offset; kept under 4 m, worst longitudinally | Not range-limited by geometry | The open question of whether navhard's Stage-2 lane-keeping collapse is a 3DGS rendering effect ([[concepts/navhard-ood-evaluation.md]]). [[sources/simvta.md]] limits rendered paths to safe candidates inside per-step pose bounds and concedes that renders degrade away from logged poses; its high-error target selection may then favour degraded renders, which it does not test |
| **Lighting and reflections** | No light transport; night scenes smear (openpilot Figure 3) | Not reported | — |
| **Following the commanded motion** | Exact by construction | **Partial**: a commanded ±0.5 m lateral deviation is rendered "not to its full extent" | [[sources/physwam.md]]: 0.80° median yaw disagreement for a joint model (floor 0.29°); [[sources/ad-e2e-jepa.md]]'s hit rate (53.8% top-1 among 257) |
| **Shortcut learning** | Artifacts correlate with the pose offset, which is what the corrective label encodes; the policy reads them | Not discussed; the partial pose-following above suggests its errors may also correlate with the offset | No other training paper in the wiki reports a test for it. [[sources/simscale.md]] has the same ingredients (3DGS renders of off-log poses, labels that steer back) and runs no probe |
| **Visual fidelity** | – | – | [[sources/simscale.md]]: scenes from blocks with novel-view PSNR ≥ 27 beat matched scenes below 27 in 8 of 8 cells, by +0.1 to +1.6 navhard EPDMS |
| **Renders after an event** | – | – | [[sources/rad.md]] ends every episode at a collision or large deviation because "the 3DGS environment typically generates noisy sensor data" afterwards, so recovery from a near-miss is never trained |
| **Vehicle dynamics** | Handled outside the renderer by a randomized vehicle model | openpilot conditions the world model on poses, not actions, so the vehicle model can change without retraining it | — |

### Shortcut learning {#shortcut}

The shortcut problem is the one specific to training. A renderer whose error depends on how far the ego is from the log gives the policy a feature that encodes that distance, and the distance is what the label is about. openpilot's remedy is an information bottleneck: white Gaussian noise caps the feature extractor's output at about 700 bits. No experiment in the paper measures either the exploitation or the remedy.

**A train–test version of the same problem** *(2026-10-06)*. [[sources/simscale.md]] trains on 3DGS renders and evaluates on navhard, whose Stage 2 is also 3DGS-rendered at displaced poses. Its gains are largest in Stage 2 (+5.6 to +7.6 per-stage EPDM, against +0.8 to +5.3 in Stage 1 and +0.8 to +2.9 on real-image navtest). That fits "the policy learned to recover" and also "the policy got used to 3DGS images". The training engine is the MTGS codebase; MTGS and the navhard benchmark paper share two author names with SimScale, and whether Stage 2 uses the same renderer is not stated. Rendering the *unperturbed* log through the same engine, with the human label, would separate the two. See [[concepts/navhard-ood-evaluation.md#simscale]].

---

## Evidence So Far {#evidence}

| Paper | State generator | Supervision | Training-side result | Deployed |
|---|---|---|---|---|
| [[sources/learning-to-drive-from-a-world-model.md]] | Reprojection | Future-anchored Plan Model | MetaDrive 24/24 lane centre, 20/20 lane change | openpilot: 27.63% of time / 48.10% of distance engaged |
| [[sources/learning-to-drive-from-a-world-model.md]] | World model (500M DiT, 400k one-minute segments) | Future-anchored Plan Model | MetaDrive 24/24, 19/20 | openpilot: 29.92% / 52.49% |
| [[sources/senna2.md]] | 3DGS, 1,044 training clips | Rule penalties (hierarchical) | Its own 3DGS benchmark: at-fault collision rate 0.118 → 0.077 from the closed-loop stage | No |
| [[sources/dreameraD.md]] | Latent world model (Epona) | Learned reward, GRPO | 87.7 EPDMS NAVSIM-v2 | No |
| [[sources/simscale.md]] | 3DGS (MTGS) over navtrain, reactive IDM agents; 147K recovery + 237K planner scenes from fixed perturbations | Log retrieval, PDM-Closed, or benchmark sub-scores | navhard +5.1 to +8.6 EPDMS for three planner families (best single 48.0, ensemble 50.9); navtest +0.8 to +2.9 (pre-fix evaluator) | No |
| [[sources/simvta.md]] | 3DGS (World Engine) over navtrain scenes with reconstructions; renders along PDM-executed paths of the current policy's samples | PDM reward for GRPO on trajectories; renders as video-loss targets for the shared encoder | navtest 89.1 → 91.3 PDMS from the video targets (RL alone 86.2 → 89.1); online renders +0.9 over a fixed initial-policy cache; shuffled targets 83.1; 3 seeds, sd ≤ 0.1 | No |
| [[sources/rad.md]] | 3DGS (extended StreetGaussian), 3,968 clips of 8 s, log-replayed agents, kinematic bicycle ego | PPO on collision and expert-deviation penalties, directional auxiliary losses, interleaved imitation (4:1) | Collision ratio 0.229 → 0.089 against its own imitation-only post-training, on 337 held-out reconstructed clips (private). Senna-2's benchmark scores it 0.281 | One real-world consistency example |

- **Every row is tested differently.** No paper compares two state-generator families under one supervision source and one test, except openpilot, whose two simulators are not separated by its tests.
- **SimScale is the only row that compares supervision sources in one simulator**: retrieval-based recovery labels peak early and then fall off as data grows; PDM-Closed labels keep improving longer; benchmark sub-scores are best for a vocabulary scorer, by +0.3 to +0.8 (within single-run noise). See [[sources/simscale.md#scaling]].
- **The field numbers are not a controlled comparison.** The two openpilot cohorts differ in trip count (47,047 against 40,026) and the paper does not describe how users were assigned; see [[concepts/evaluation-variance.md#outside-navsim]].
- **Only openpilot reports real-world use.** It is also the only row restricted to lateral control.

---

## Open Questions

*Wiki-wide open questions are collected in [[research-directions.md]].*

- **Which family trains the best policy?** openpilot's reprojective and world-model simulators tie on lateral tests. The longitudinal regime, where reprojection is limited to a few metres, is untested.
- **Do policies trained in rendered simulators learn the renderer's artifacts?** Asserted by openpilot for reprojection and countered with a 700-bit bottleneck, but not measured. A direct test: train with and without the bottleneck and probe the policy's features for the pose offset. *(2026-10-06: [[sources/simscale.md]] raises the evaluation-side version. Training on 3DGS renders helps most on navhard's 3DGS-rendered Stage 2, possibly through a shared renderer. Untested.)*
- **Is a reward needed?** Future-anchored labels and reward-based RL have never been compared in one simulator. *(2026-10-06: [[sources/rad.md]] compares off-policy imitation, RL and both in one 3DGS simulator: 0.229 / 0.143 / 0.089 collision ratio; PPO without its hand-designed auxiliary losses is no better than imitation, 0.249.)* *(2026-10-06: [[sources/simscale.md]] compares expert-trajectory labels with per-trajectory benchmark sub-scores for one scorer in one simulator. The sub-scores win by +0.3 / +0.8 navhard, inside single-run noise, and only with expert imitation kept on the real data. There is no policy-gradient arm.)*
- **Does simulator quality predict policy quality?** openpilot shows LPIPS improving with world-model size and data (in a figure not in the clipping) but trains policies with one world model only. *(2026-10-06: within one renderer, yes, slightly. [[sources/simscale.md]] trains on matched scenes from high- and low-PSNR reconstructions and the high-fidelity data wins all eight cells, by +0.1 to +1.6 navhard EPDMS.)*
- **Can a learned simulator make other agents react without being pulled back to the log?** openpilot's anchor guarantees recovery pressure and also forces the world to the recorded state at the anchor.
- **Is the evaluation-side effect the same mechanism?** If 3DGS renderings degrade with the ego offset as reprojections do, part of navhard's Stage-2 collapse measures the renderer.
- **Do fixed perturbations reach the states a trained policy drifts into?** [[sources/simscale.md]] samples them around the log, independent of the policy, and still gains 5–9 navhard EPDMS. Whether on-policy collection in the same renderer, with the same teacher, adds to that is untested. *(2026-10-14: in the video-target role, [[sources/simvta.md]] finds on-policy renders worth +0.9 PDMS over a cache rendered once from the initial policy (91.3 vs 90.4), navtest only. Not SimScale's setting: the renders are targets, not inputs, and there is no teacher.)*
- **Is a navhard gain from 3DGS training data recovery or renderer familiarity?** See [[sources/simscale.md#navhard-gain]]: a zero-perturbation rendering control would answer it.

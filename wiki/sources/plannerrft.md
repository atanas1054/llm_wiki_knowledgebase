---
title: "PlannerRFT: Reinforcing Diffusion Planners through Closed-Loop and Sample-Efficient Fine-Tuning"
type: source-summary
sources: [raw/papers/PlannerRFT_ Reinforcing Diffusion Planners through Closed-Loop and Sample-Efficient Fine-Tuning.md]
related: [concepts/diffusion-planner.md, concepts/rl-for-ad.md, concepts/gspo-vs-grpo.md, concepts/nuplan-benchmark.md]
created: 2026-06-23
updated: 2026-06-23
confidence: high
---

# PlannerRFT

PlannerRFT is a closed-loop reinforcement fine-tuning framework for an imitation-pretrained diffusion planner. Its central claim is that diffusion-planner RFT is limited by the quality of sampled alternatives: Gaussian starts collapse toward nearly identical trajectories, while fixed anchors create diverse but scene-inappropriate candidates. PlannerRFT instead learns a scene-conditioned distribution over lateral and longitudinal denoising guidance.

The training system has two coupled branches. PPO trains an Exploration Policy to select useful guidance directions over closed-loop episodes; GRPO fine-tunes the Diffusion Transformer (DiT) from grouped trajectory rewards. The reference planner and guidance policy are training-only and removed at deployment.

## Key Takeaways

- **Adaptive exploration matters more than raw diversity.** Uniform guidance has the highest diversity score (39.78%) but degrades Test14-hard reactive score to 65.82; learned guidance reaches 72.21 with 25.34% diversity.
- **Two optimization timescales are separated.** PPO assigns long-horizon credit to guidance choices made at each simulator step, while GRPO updates the high-dimensional denoising transitions from grouped trajectory outcomes.
- **Survival reward prevents all-zero hard groups.** Instead of assigning only terminal success/failure, it rewards how long a candidate remains valid, preserving within-group signal after eventual failure.
- **Hard-case curation must be balanced.** Training only on collision/off-road cases causes broad forgetting; training on all cases dilutes the learning signal; the below-90-score subset works best.
- **Guidance is training-only.** The deployed planner uses the original five-step DDIM path without the frozen reference or Exploration Policy, taking 34.27 ms in the paper's setup.
- **Closed-loop gains concentrate in interactive settings.** Relative to the matched five-step DDIM baseline, PlannerRFT improves Test14-hard reactive from 68.18 to 72.21, versus only 76.01 to 77.16 in non-reactive mode.
- **nuMax enables scale but changes training dynamics.** Its JAX/XLA simulator is reported as up to 10x faster than native nuPlan, but training traffic is log replay rather than IDM-reactive traffic.

## Method

### Policy-Guided Denoising

The planner uses a frozen reference trajectory and injects energy gradients into the trainable denoising process. Lateral guidance offsets waypoints normal to the reference path; longitudinal guidance changes velocity along its tangent. The Exploration Policy conditions on scene features and the reference trajectory, then predicts Beta distributions for the two guidance scales.

Sampling several scale pairs gives a group of continuous maneuver hypotheses around the imitation prior. This avoids both vanilla diffusion's mode collapse and the context mismatch of fixed trajectory anchors.

![Figure 1: Vanilla diffusion collapses toward one mode; fixed anchors are diverse but scene-agnostic; PlannerRFT learns multimodal, scene-adaptive guidance.](<../../raw/assets/x1 35.png>)

![Figure 2: PlannerRFT overview: policy-guided denoising, closed-loop rollout, GRPO trajectory optimization, and PPO exploration optimization.](<../../raw/assets/x2 34.png>)

### Dual-Branch Optimization

The Exploration Policy is trained with PPO and generalized advantage estimation so guidance choices receive long-horizon closed-loop credit. The DiT denoising chain is treated as an MDP whose Gaussian transitions are updated with GRPO. A behavior-cloning term regularizes the DiT against policy collapse.

For a horizon of length $T_r$, the survival reward is:

$$
R_{\mathrm{surv}}=\frac{1}{T_r}\sum_{\tau=1}^{T_r}R_\tau^{\mathrm{term}}\prod_{j=1}^{\tau}\mathbb{I}[R_j^{\mathrm{term}}\neq0].
$$

This distinguishes candidates that fail early from those that remain safe longer, even when neither completes the horizon.

### nuMax

nuMax preprocesses nuPlan scenes into fixed-shape TFRecords, implements an LQR tracker and kinematic bicycle model in JAX, reproduces nuPlan-style reward terms, and connects JAX simulation on rank 0 with PyTorch DDP policy workers. The reward multiplies collision, drivable-area, and wrong-direction gates by a weighted combination of TTC, progress, comfort, and speeding scores.

![Figure 3: nuMax scenario cache, calibrated tracker/scorer, and distributed PyTorch/JAX training pipeline.](<../../raw/assets/x3 31.png>)

## Best Practices Reported by the Paper

- Use five-step stochastic DDIM for exploration and training efficiency.
- Zero-initialize the Exploration Policy so early samples remain centered on the imitation prior.
- Remove reference and guidance modules for deployment.
- Mix moderately difficult cases rather than training exclusively on failures.
- Use both lateral and longitudinal guidance; their metric effects are complementary.
- Use a moderate guidance range: 2.5 m lateral and 25% longitudinal is best in the reported grid.

## Main Results

### Table 1: nuPlan Closed-Loop Planning

| Type | Planner | Val14 NR | Val14 R | Test14-hard NR | Test14-hard R |
| --- | --- | ---: | ---: | ---: | ---: |
| Expert | Log-replay | 93.53 | 80.32 | 85.96 | 68.80 |
| Rule | IDM | 75.60 | 77.33 | 56.15 | 62.26 |
| Rule | PDM-Closed | 92.84 | 92.12 | 65.08 | 75.19 |
| Learning | PDM-Open | 53.53 | 54.24 | 33.51 | 35.83 |
| Learning | GameFormer | 13.32 | 8.69 | 7.08 | 6.69 |
| Learning | PlanTF | 84.27 | 76.95 | 69.70 | 61.61 |
| Learning | PLUTO | 88.89 | 78.11 | 70.03 | 59.74 |
| Learning | Diffusion Planner | 89.87 | 82.80 | 75.99 | 69.22 |
| Learning | Flow Planner | **90.43** | 83.31 | 76.47 | 70.42 |
| Learning | PlannerRFT | 89.96 | **84.46** | **77.16** | **72.21** |

The table's Diffusion Planner row uses its official DPM setup. The matched five-step DDIM baseline used in the ablations is 89.81/82.94 on Val14 and 76.01/68.18 on Test14-hard.

### Table 2: Exploration Policy Ablation (Test14-hard)

| Exploration | R-score | Collision | TTC | Drivable | Comfort | Progress | Speed | NR-score | Diversity $\mathcal D$ | Mean reward | Reward std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IL Pretrain DDIM | 68.18 | 86.58 | 79.05 | 94.48 | 86.03 | 76.99 | 97.20 | 76.01 | - | - | - |
| No guidance | 68.83 | 86.03 | 79.41 | 94.48 | 87.87 | 77.12 | 97.35 | 76.34 | 5.65 | 69.06 | 0.02 |
| Uniform | 65.82 | 84.37 | 75.74 | 93.01 | 80.88 | 76.19 | 97.59 | 75.19 | **39.78** | 60.44 | 0.12 |
| Fixed Beta | 70.65 | 87.68 | 80.88 | 94.85 | 84.56 | 77.34 | 97.71 | 76.61 | 27.73 | 71.50 | 0.07 |
| PlannerRFT | **72.21** | **88.97** | **84.93** | **95.59** | 85.66 | 77.17 | **98.03** | **77.16** | 25.34 | **73.88** | 0.06 |

### Table 3: Fine-Tuning Data Distribution

| Training | Dataset | Val14 NR | Val14 R | Test14-hard NR | Test14-hard R |
| --- | --- | ---: | ---: | ---: | ---: |
| IL Pretrain | All | 89.87 | 82.80 | 75.99 | 69.22 |
| IL Fine-tune | Lt90 | 88.91 | 82.08 | 74.32 | 67.55 |
| RL Fine-tune | Fail | 82.97 | 77.48 | 69.26 | 63.75 |
| RL Fine-tune | All | 89.93 | **84.88** | 75.50 | 70.43 |
| RL Fine-tune | Lt90 | **89.96** | 84.46 | **77.16** | **72.21** |

### Table 4: GRPO Reward and Horizon

| Reward | Horizon | Val14 NR | Val14 R | Test14-hard NR | Test14-hard R |
| --- | ---: | ---: | ---: | ---: | ---: |
| Terminal | 4 s | 89.78 | 84.27 | 76.81 | 71.59 |
| Survival | 2 s | 89.54 | 84.08 | 76.49 | 70.10 |
| Survival | 4 s | **89.96** | **84.46** | **77.16** | **72.21** |
| Survival | 6 s | 89.66 | 84.31 | 76.96 | 71.91 |

### Table 5: Maximum Guidance Offset (Test14-hard Reactive)

| Lateral offset | Longitudinal 10% | Longitudinal 25% | Longitudinal 50% |
| ---: | ---: | ---: | ---: |
| 1.0 m | 69.94 | 71.41 | 70.26 |
| 2.5 m | 70.64 | **72.21** | 71.95 |
| 5.0 m | 70.11 | 71.63 | 69.99 |

![Figure 4: Behavior evolution through RFT: collision-prone lane change, conservative lane keeping, then safe and efficient lane change.](<../../raw/assets/x4 29.png>)

![Figure 5: Candidate trajectories under unguided, uniform, fixed-Beta, and learned adaptive exploration.](<../../raw/assets/x5 26.png>)

![Figure 6: Training curves show learned adaptive exploration is more stable than fixed, uniform, and unguided alternatives.](<../../raw/assets/x6 22.png>)

## Appendix Tables

### Table A1: Hyperparameters

| Branch | Hyperparameter | Value |
| --- | --- | ---: |
| Guidance | Maximum lateral offset | 2.5 m |
| Guidance | Maximum longitudinal offset | 25% |
| PPO | Samples | 40M |
| PPO | Initial learning rate | $2.5\times10^{-4}$ |
| PPO | Learning-rate schedule | Cosine decay |
| PPO | Environments | 128 |
| PPO | Environment steps / iteration | 32 |
| PPO | Batch size | 4096 |
| PPO | Mini-batch size | 4096 |
| PPO | Steps / epoch | 1 |
| PPO | Epochs | 4 |
| PPO | Value coefficient | 0.5 |
| PPO | Entropy coefficient | 0.01 |
| PPO | Discount factor | 0.99 |
| PPO | GAE $\lambda$ | 0.95 |
| PPO | Clip range $\epsilon$ | 0.2 |
| PPO | Maximum gradient norm | 0.5 |
| GRPO | Initial learning rate | $2.5\times10^{-4}$ |
| GRPO | Learning-rate schedule | Cosine decay |
| GRPO | Group size | 8 |
| GRPO | Mini-batch size | 4096 |
| GRPO | Steps / epoch | 6 |
| GRPO | Epochs | 1 |
| GRPO | Denoising discount factor | 0.8 |
| GRPO | BC loss weight | 0.4 |

### Table A2: Inference Type

| Model | Steps | Latency | Val14 NR | Val14 R |
| --- | ---: | ---: | ---: | ---: |
| Diffusion Planner DPM | 10 | 86.43 ms | 89.87 | 82.80 |
| PlannerRFT with guidance | 10 | 75.48 ms | 89.83 | 83.93 |
| PlannerRFT without guidance | 5 | **34.27 ms** | **89.96** | **84.46** |

### Table A3: Extended nuPlan Results

| Type | Planner | Val14 NR | Val14 R | Test14-hard NR | Test14-hard R | Test14-random NR | Test14-random R |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Expert | Log-replay | 93.53 | 80.32 | 85.96 | 68.80 | 94.03 | 75.86 |
| Rule | IDM | 75.60 | 77.33 | 56.15 | 62.26 | 70.39 | 74.42 |
| Rule | PDM-Closed | 92.84 | 92.12 | 65.08 | 75.19 | 90.05 | 91.63 |
| Learning | PDM-Open | 53.53 | 54.24 | 33.51 | 35.83 | 52.81 | 57.23 |
| Learning | GameFormer | 13.32 | 8.69 | 7.08 | 6.69 | 11.36 | 9.31 |
| Learning | PlanTF | 84.27 | 76.95 | 69.70 | 61.61 | 85.62 | 79.58 |
| Learning | PLUTO | 88.89 | 78.11 | 70.03 | 59.74 | 89.90 | 78.62 |
| Learning | Diffusion Planner DPM | 89.87 | 82.80 | 75.99 | 69.22 | 89.19 | 82.93 |
| Learning | Diffusion Planner DDIM | 89.81 | 82.94 | 76.01 | 68.18 | 89.14 | 82.63 |
| Learning | Flow Planner | 90.43 | 83.31 | 76.47 | 70.42 | 89.88 | 82.93 |
| Learning | PlannerRFT | 89.96 | **84.46** | **77.16** | **72.21** | **90.76** | **85.80** |

### Table A4: Guidance Type (Test14-random Reactive)

| Training | Lateral | Longitudinal | Collision | TTC | Drivable | Comfort | Progress | Speed | R-score |
| --- | :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IL Pretrain DDIM | No | No | 86.58 | 79.05 | 94.48 | 86.03 | 76.99 | 97.20 | 68.18 |
| PlannerRFT | No | Yes | 87.50 | 81.62 | 92.65 | 86.76 | 77.54 | 97.99 | 69.59 |
| PlannerRFT | Yes | No | 87.31 | 80.88 | 94.85 | **87.50** | 76.38 | 97.32 | 70.18 |
| PlannerRFT | Yes | Yes | **88.97** | **84.93** | **95.59** | 85.66 | 77.17 | **98.03** | **72.21** |

### Table A5: GRPO Group Size

| Group size | R-score | Collision | Drivable | Comfort | Progress | NR-score |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 71.24 | 86.40 | 94.85 | 84.93 | **78.99** | 76.31 |
| 8 | 72.21 | **88.97** | **95.59** | **85.66** | 77.17 | **77.16** |
| 12 | **72.29** | **88.97** | 95.22 | **85.66** | 77.62 | 77.04 |

## Appendix Figures and Qualitative Evidence

![Figure A1: A cached nuMax scene containing lanes, agents, obstacles, and route geometry within 200 m of ego.](<../../raw/assets/x7 18.png>)

![Figure A2: Distributed pipeline coupling PyTorch DDP policy workers with JAX simulation on rank 0.](<../../raw/assets/x8 13.png>)

![Figure A3: Diffusion Planner, anchored DiffusionDrive, and PlannerRFT trajectory samples; PlannerRFT produces smooth, scene-adaptive modes.](<../../raw/assets/x9 7.png>)

![Figure A4: Pedestrian crossing: the IL planner collides; PlannerRFT waits and completes the right turn.](<../../raw/assets/x10 9.png>)

![Figure A5: Emergency braking in reactive traffic: PlannerRFT stops for a stationary lead vehicle.](<../../raw/assets/x11 6.png>)

![Figure A6: S-curve lane change around a stationary vehicle.](<../../raw/assets/x12 5.png>)

![Figure A7: Fine steering through a traffic-cone narrowing.](<../../raw/assets/x13 8.png>)

![Figure A8: Blocked right turn: PlannerRFT delays a conflicting lane change.](<../../raw/assets/x14 4.png>)

![Figure A9: Unprotected right turn: PlannerRFT completes the maneuver before cross traffic arrives.](<../../raw/assets/x15 4.png>)

![Figure A10: Causal-confusion example: PlannerRFT continues after turning instead of imitating a spurious pull-over pattern.](<../../raw/assets/x16 2.png>)

## Limitations

- **Structured inputs only.** The planner consumes abstract agents, map features, and obstacles; no camera-based or other sensory end-to-end planner is tested.
- **Single benchmark family.** All quantitative validation is on nuPlan splits, so transfer to other simulators, geographies, and real vehicles remains untested.
- **High training cost.** The reported run uses 40M environment steps on eight H100 GPUs; the paper does not provide a wall-clock or energy comparison against simpler fine-tuning.
- **Training/evaluation traffic mismatch.** nuMax uses log-replay surrounding agents for speed, while reported reactive evaluation uses IDM in native nuPlan.
- **Simulator approximation.** nuMax is calibrated to nuPlan but is still a reimplementation; no detailed scorer/controller agreement error is reported.
- **Static-shape cache.** XLA-friendly caching requires reprocessing for different model representations and is not yet a general scenario interface.
- **Hand-designed exploration axes.** Learned adaptivity operates only over predefined lateral and longitudinal offsets; it does not discover arbitrary maneuver parameterizations.
- **SOTA scope requires care.** PlannerRFT leads the learning-only baselines listed in three of four main columns, but rule-based/hybrid planners can score higher, and later papers or differing post-processing settings are not comparable without protocol alignment.
- **Statistical uncertainty is absent.** Results are reported as point estimates without multiple-seed variance or confidence intervals.

## Wiki Relevance

- [[concepts/diffusion-planner.md]] — converts diffusion mode collapse from an inference issue into an exploration bottleneck for RFT.
- [[concepts/rl-for-ad.md]] — separates exploration-policy learning (PPO) from denoiser learning (GRPO) and demonstrates closed-loop benefit concentrated in reactive traffic.
- [[concepts/gspo-vs-grpo.md]] — adds survival rewards and adaptive group construction to the catalog of GRPO design choices.
- [[concepts/nuplan-benchmark.md]] — supplies matched DDIM/DPM results, reactive/non-reactive comparisons, and a fast training simulator with protocol caveats.

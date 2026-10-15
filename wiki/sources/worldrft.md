---
title: "WorldRFT: Latent World Model Planning with Reinforcement Fine-Tuning for Autonomous Driving"
type: source-summary
sources: ["raw/papers/WorldRFT_ Latent World Model Planning with Reinforcement Fine-Tuning for Autonomous Driving.md"]
related: [sources/law.md, sources/dreameraD.md, sources/latent-wam.md, sources/metis.md, sources/epona.md, sources/diffusiondrive.md, sources/resworld.md, sources/foresight.md, sources/drivefuture.md, sources/hyworldvla.md, sources/momworld.md, sources/simwam.md, sources/world4scorer.md, sources/foredrive.md, sources/idol.md, sources/simvta.md, sources/sgdrive.md, sources/sparsedrivev2.md, sources/geoworldad.md, sources/coworld-vla.md, sources/policy-world-model.md, sources/drivesuprim.md, sources/hydra-mdp-pp.md, sources/recogdrive.md, sources/autovla.md, sources/adathinkdrive.md, sources/drivevla-w0.md, sources/wam-flow.md, sources/rad.md, sources/plan-r1.md, sources/nord.md, concepts/world-model-for-ad.md, concepts/rl-for-ad.md, concepts/navsim-benchmark.md, concepts/nuscenes-waymo-evals.md, concepts/teacher-pseudo-labels.md, concepts/foundation-backbones-for-ad.md, concepts/intent-conditioned-planning.md, concepts/evaluation-variance.md, research-directions.md]
created: 2026-10-15
updated: 2026-10-15
confidence: medium
---

# WorldRFT: Latent World Model Planning with Reinforcement Fine-Tuning for Autonomous Driving

**Authors**: Pengxuan Yang, Ben Lu, Zhongpu Xia, Chao Han, Yinfeng Gao, Teng Zhang, Kun Zhan, XianPeng Lang, Yupeng Zheng, Qichao Zhang
**Orgs**: State Key Laboratory of Multimodal Artificial Intelligence Systems, Institute of Automation, CAS; two schools of UCAS; Li Auto
**arXiv**: 2512.19133v1 (December 2025). [[sources/dreameraD.md]] and [[sources/hyworldvla.md]] list it as AAAI 2026; the clipping names no venue.
**Code**: github.com/pengxuanyang/WorldRFT
**Source**: `raw/papers/WorldRFT_ Latent World Model Planning with Reinforcement Fine-Tuning for Autonomous Driving.md`. The clipping has all eight figures and all twelve tables (Tables 3, 4, 8 and 1 are HTML and are rebuilt below). The main text cites the nuScenes ablation as "Table 4"; it is captioned Table 3, and Table 4 is the NAVSIM ablation in the supplement.

**Lineage.** WorldRFT is [[sources/law.md]]'s latent world model (LAW, ICLR 2025; a world decoder predicts the next frame's latent from the current latent and the planned trajectory, trained by MSE) with World4Drive's Grounded-SAM semantic loss added, then a new encoder, planning head and RL stage. World4Drive (ICCV 2025) is the same CASIA group's previous paper (Zheng Y., Yang P., Zhang Q., Zhang T., Xia Z.), and the first author also wrote UncAD, a row in Table 1. Two March 2026 papers from the group print a NAVSIM-v2 row for WorldRFT that this paper does not contain: [[sources/dreameraD.md]] (authors by initials Yang P., Zheng Y., Zhang Q., Xia Z.) and [[sources/latent-wam.md]] (Yupeng Zheng, Qichao Zhang, Pengxuan Yang). [[sources/metis.md]] shares two Li Auto authors (Chao Han, Kun Zhan). Before this ingest the wiki knew WorldRFT as a row in ten papers' tables and as one of DreamerAD's sampler ablations.

**Confidence: medium.** All thirteen NAVSIM rows pass the closed-form check, and sixteen of seventeen nuScenes averages reproduce from their per-horizon values. Against that:
- The nuScenes collision gains are a few samples each, every run is single, and three hyperparameter sweeps were chosen on the evaluation split, with the chosen value the only one at 0.05% in each.
- The RL reward is collision with annotated agent boxes, the quantity the nuScenes metric counts, so the headline safety gain is the metric being optimized on the training split.
- The nuScenes table crosses averaging conventions, and one row's average does not match its per-horizon values.
- "Best among vision-only solutions" on NAVSIM holds only inside its table.
- A frozen VGGT runs on every camera at inference, and no latency is reported.
- The implementation details and the ablation disagree on the spatial-path configuration behind the headline.

---

## What It Is

**A LAW-style latent world model whose planning head is decomposed into three subtasks and refined against image features, with a frozen geometry model in the encoder, then fine-tuned by GRPO with a collision-only reward.**

- **Spatial-aware World Encoder (SWE).**
  - A ResNet image backbone gives features $F_t$; Grounded-SAM pseudo-labels (high-confidence masks only) supervise them with a cross-entropy semantic loss, "following World4Drive".
  - A **frozen VGGT** reads each camera (rescaled and cropped to 294×518) and its final layer's 3D tokens are fused into $F_t$ by **one cross-attention layer**, image features as queries, VGGT tokens as keys and values. The result is the latent world representation $W^t_\text{latent}$.
  - nuScenes: ResNet-50, six cameras at 360×640 (padded to 384×640), features 6×12×20×256. NAVSIM: ResNet-34 on the three front cameras stitched to 256×1024, plus ego velocity, acceleration and driving command. How the stitched image is fed to VGGT is not stated ("the same usage methodology").
- **Hierarchical Planning Refinement (HPR).** Three query sets attend to $W^t_\text{latent}$, then to each other by self-attention:
  - **Target region**: a Laplace distribution $(\mu, b)$ over the trajectory's endpoint, trained by its negative log-likelihood. The scale $b$ is meant to grow with scene difficulty and is fed to the refiner.
  - **Spatial path**: $N$ points at a fixed 2 m spacing, time-free, from interpolating the logged trajectory (L1).
  - **Temporal trajectory**: points every 0.5 s, 3 s on nuScenes and 4 s on NAVSIM (L1).
  - **Local-aware iterative refinement**, $K=3$ rounds over all three outputs: project the current trajectory points into the cameras, sample local features there with deformable convolution, fuse them with the trajectory query, an encoding of all three current outputs and an MLP of $b$, and add a predicted offset scaled by $\alpha=0.1$.
- **Pretraining loss**: $0.2\,\mathcal L_\text{sem} + 0.2\,\mathcal L_\text{rec} + 0.001\,\mathcal L_\text{target} + 1.0\,\mathcal L_\text{traj}$, where $\mathcal L_\text{rec}$ is LAW's next-frame latent MSE.
- **Reinforcement fine-tuning (RFT).**
  - **Policy**: a Gaussian whose mean is the predicted trajectory and whose variance comes from an added variance network. $G=10$ trajectories per scene.
  - **Reward**: per trajectory point, **−1 if the ego box overlaps any nearby agent's box, else 0**. The agent boxes can only be the datasets' annotated boxes; the paper does not say.
  - **Advantage**: rewards are group-normalized per point, then summed over the point and all later ones, because each point is an increment that moves every later point.
  - **Objective**: PPO-style clipped ratio. Its "KL" term is the Gaussian negative log-likelihood of the reference model's deterministic trajectory under the current policy (a deterministic reference has no density, so this is not a KL), plus an L2 loss between current and reference means and a maximum-entropy bonus (coefficients 0.1, 0.12, 0.1). Learning rate $3\times10^{-6}$.
- **Training.** nuScenes: 12 epochs, 8 × RTX 3090, batch 8, learning rate $5\times10^{-5}$; RFT length not stated. NAVSIM: 100 epochs, batch 64; RFT 10 epochs.

**What the world model does not do.** It is the policy's backbone and a training objective. RL samples trajectories from the planning head and scores them against logged boxes; no rollout passes through the world decoder, and no reward is learned. That separates WorldRFT from [[sources/dreameraD.md]] (RL inside the world model's latent rollouts, learned reward) and from [[sources/simvta.md]] (simulator reward, world-model video loss kept during RL).

![[main_new.png|Figure 2: WorldRFT overview. Left (yellow): Spatial-aware World Encoder; multi-view images go to a World Encoder supervised by Grounded-SAM through L_sem, with VGGT features fused in, giving W_latent. Middle (green): Hierarchical Planning Refinement; Q_target, Q_path and Q_traj decode a target region (ellipse around a red star), a spatial path (blue) and a temporal trajectory (yellow) against the GT trajectory (red dashed), refined over K iterations of Projection, Sample, Fusion, Decode. Right (blue): Reinforcement Tuning; Gaussian sampling, reward modeling of sampled trajectories near another vehicle (smiling and frowning faces), policy optimization; a World Decoder predicts W_latent at t+1 from W_latent at t and the trajectory, trained by L_rec.]]

**Figure 2** is the whole method. The world decoder (bottom right) feeds only $\mathcal L_\text{rec}$; the RL loop above it scores sampled trajectories directly.

![[subfig2.png|Figure 3: the local-aware iterative refinement module. (mu, b), T_path and T_traj are encoded into F_b and, concatenated, into F_s; T_traj is projected into the front camera and deformable convolution samples F_local at offsets around each trajectory point; F_b, Q''_traj, F_s and F_local are concatenated and decoded into an offset Delta T_traj, added to T_traj with step alpha; the loop runs K times.]]

**Figure 3**: the refiner samples image features at the trajectory's own projected points, the same "read the scene where the plan goes" idea as [[sources/maneuverflow.md]]'s path readout, here on current features rather than a predicted future.

| Benchmark | WorldRFT | Notes |
|---|---:|---|
| nuScenes L2 / collision, 1–3 s average, no ego status | **0.48 m / 0.05%** | Temporal-average (VAD) convention; without RFT 0.47 / 0.15 |
| nuScenes, with ego status | 0.33 m / 0.04% | The setup is promised in the supplement and not described there |
| NAVSIM-v1 PDMS (navtest) | **87.8** | Three stitched cameras, ResNet-34 + frozen VGGT; without RFT 87.0 |
| NAVSIM-v2 EPDMS | – | Not in the paper; an 86.7 row circulates (see Key Takeaways) |
| navhard, HUGSIM, Bench2Drive | – | Not reported |
| Latency / parameters | – | Not reported |

![[fig1 1.png|Figure 1: scatter of NAVSIM PDMS (x, 83-89) against nuScenes collision rate (y, 0.35 at the bottom to 0 at the top). Perception-based circles: UniAD (83.4, 0.31), PARA-Drive (84.0, 0.25), DiffusionDrive (88.1, 0.08). Latent-WM stars: LAW (84.6, 0.30), World4Drive (85.1, 0.16), Ours without RFT (87.0, 0.15), Ours with RFT (87.8, 0.05).]]

**Figure 1** plots the two headline numbers. UniAD's 0.31 is its per-horizon value, while the latent-WM points are temporal averages; in the temporal-average convention UniAD is at 0.12 (see Table 1).

---

## Key Takeaways

- **The RL stage is collision-only RL against logged boxes, not RL in a world model.** The reward is a binary overlap test between the ego box and other agents' boxes along each sampled trajectory. Three consequences:
  - **On nuScenes the reward is the evaluation metric's own quantity.** The collision rate counts the same overlaps against the same kind of annotated boxes, so 0.15% → 0.05% shows the metric being optimized on the training split; whether it reflects collision avoidance that would hold against agents that react is not testable here.
  - **The paper is not annotation-free once RL starts.** "Perception annotation-free" holds for pretraining; the reward needs 3D boxes for every agent in the future frames ([[concepts/teacher-pseudo-labels.md#privileged-teachers]]).
  - **The wiki had the reward wrong.** [[sources/dreameraD.md]]'s comparison table and [[concepts/rl-for-ad.md]] list WorldRFT's reward source as "NAVSIM PDM"; it is the binary collision term alone. Both pages now carry the correction.
- **The safety gain is a handful of samples.** nuScenes val has about 6,000 samples ([[concepts/nuscenes-waymo-evals.md#resworld]]), so 0.01 percentage points is under one sample.
  - RFT takes the per-horizon collision rates from 0.10 / 0.11 / 0.23% to 0.00 / 0.00 / 0.16%: about six, seven and fourteen samples to zero, zero and ten.
  - The "83% reduction" against LAW (0.30 → 0.05) is roughly fifteen samples on average.
  - Single runs throughout. Every collision difference in the ablation tables is between under one and about seven samples.
- **On NAVSIM, RFT's gain shows up where the reward does not point.** 87.0 → 87.8 PDMS, but NC +0.3, TTC 0.0, DAC +0.8 and EP +0.8. A collision-only reward moving drivable-area compliance and progress more than collisions is not explained; the entropy, reference and NLL regularizers change too.
- **One matched RL-versus-supervised comparison.** Table 11 trains the same model with "the same safety-aware loss" by supervised learning: 0.48 m / 0.10% against RFT's 0.48 / 0.05. What a supervised version of a binary collision reward is, the paper does not define, and the gap is about three samples.
- **The architecture components, read as single-factor contrasts** (all with RFT; nuScenes L2 / CR, NAVSIM PDMS):

  | Contrast (Table 3 / 4 IDs) | nuScenes | NAVSIM |
  |---|---|---:|
  | Target + path + refinement on the bare baseline (1 → 4) | 0.59 / 0.16 → 0.52 / 0.08 | +1.5 |
  | **Frozen VGGT** (4 → 8) | 0.52 / 0.08 → 0.48 / 0.05 | **+1.3** |
  | Target, given VGGT + path (5 → 8) | 0.51 / 0.08 → 0.48 / 0.05 | +0.6 |
  | Path, given VGGT + target (6 → 8) | 0.49 / 0.09 → 0.48 / 0.05 | +0.7 |
  | Refinement (7 → 8) | 0.50 / 0.06 → 0.48 / 0.05 | +0.2 |

  The supplement's "most significant improvement comes from incorporating VGGT (ID 4 → ID 5: +0.7)" compares rows that differ in two factors (VGGT added, target removed). The clean contrast is ID 4 → ID 8, +1.3. Refinement is never ablated without VGGT, and the target and path rows always include it.
- **A frozen geometry model as a fused feature source is worth +1.3 PDMS here, at an unreported cost.** VGGT alone beats Grounded-SAM alone (0.49 / 0.06 against 0.52 / 0.08) and the two add (0.48 / 0.05). This is the wiki's first use of an off-the-shelf frozen VGGT *in the input path* of a non-VLM planner (CoWorld-VLA runs one inside a VLA; Latent-WAM uses WorldMirror only as a training teacher), see [[concepts/foundation-backbones-for-ad.md#geometry-foundation-models-are-a-third-backbone-family]]. The paper credits VGGT for DAC 96.8, "the highest among all methods"; without RFT, with VGGT, DAC is 96.0, so +0.8 of it comes from RFT, and no ablation reports DAC.
- **A probabilistic goal beats a point, and a point is worse than no goal.** Table 9: a deterministic target point 0.53 / 0.10, the Laplace region 0.48 / 0.05. The configuration without any target (Table 3 ID 5) is 0.51 / 0.08, so the deterministic point is below removing the target, and the region's gain over none is about two samples of collision. The deterministic variant also has no $b$ to feed the refiner, so the comparison changes the refiner's input as well. The target loss weight is 0.001. Compare [[sources/sgdrive.md]]'s deterministic goal pose (+0.7 PDMS), see [[concepts/intent-conditioned-planning.md#worldrft]].
- **The spatial path is a time-free trajectory factorization, three months before [[sources/sparsedrivev2.md]].** WorldRFT supervises a path resampled every 2 m beside the 0.5 s trajectory; SparseDriveV2 builds its vocabulary from paths every 1 m to 50 m and separate speed profiles. Here the path is an auxiliary output, worth +0.7 PDMS.
- **Every hyperparameter sweep's chosen value is the only one at 0.05%**, on the evaluation split: path configuration (others 0.09–0.12), refinement iterations (K = 1: 0.08; K = 6: 0.07; no refinement: 0.06), group size (G = 5: 0.08; G = 15: 0.07). The neighbours are one to four samples away. In two of the three, the non-chosen values are no better than leaving the component out (no path 0.09; no refinement 0.06). See [[concepts/evaluation-variance.md]].
- **"Best among vision-only solutions" holds only inside Table 2.** Camera-only NAVSIM-v1 entries above 87.8 that the wiki dates before WorldRFT: Hydra-MDP++ V2-99 91.0 ([[sources/hydra-mdp-pp.md]], March 2025), DriveSuprim ResNet-34 89.9 ([[sources/drivesuprim.md]]), ReCogDrive 89.6 ([[sources/recogdrive.md]]) and AutoVLA 89.1 ([[sources/autovla.md]]), all June 2025; AdaThinkDrive 90.3 ([[sources/adathinkdrive.md]]); DriveVLA-W0 90.2 and the single-camera world model PWM 88.1 ([[sources/drivevla-w0.md]], [[sources/policy-world-model.md]], October 2025); WAM-Flow 90.3 ([[sources/wam-flow.md]], December). PWM and DriveVLA-W0 are world-model methods. Within the self-supervised latent world-model rows it prints (Epona 86.2, World4Drive 85.1, LAW 84.6, DriveX 84.5) the lead is real.
- **How later tables carry it.**
  - **v1**: [[sources/dreameraD.md]], [[sources/drivefuture.md]], [[sources/hyworldvla.md]], [[sources/momworld.md]], [[sources/simwam.md]] and [[sources/world4scorer.md]] print 87.8 with this paper's sub-scores. [[sources/metis.md]] prints **87.0, the pre-RFT row**, under its RL marker ("WorldRFT\*").
  - **v2**: [[sources/dreameraD.md]], [[sources/latent-wam.md]], [[sources/metis.md]], [[sources/foredrive.md]], [[sources/idol.md]] and [[sources/world4scorer.md]] print 86.7 (EC 69.1). **The paper reports no NAVSIM-v2 number.** The earliest prints are DreamerAD and Latent-WAM (both March 2026, both sharing authors with WorldRFT), so the row plausibly comes from the authors' own later evaluation. Its residual is −1.38, corrected-like ([[concepts/navsim-benchmark.md#worldrft]]).
- **DreamerAD's description of it applies to DreamerAD's re-implementation.** DreamerAD's Table 3 swaps its vocabulary sampler for "WorldRFT" sampling inside its own world model (86.6 against 87.7 EPDMS) and describes the problem as "dynamic discontinuity → world model hallucinations". In WorldRFT the sampled trajectories never enter a world model, so that failure mode cannot arise in the original setting.

---

## Results

### Table 1: nuScenes open-loop planning (L2 in m, collision rate in %)

P-IL = perception-based imitation; SS-L = self-supervised; RL = reinforcement learning; † = with ego status.

| Method | Training | L2 1s | 2s | 3s | Avg. | CR 1s | 2s | 3s | Avg. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ST-P3 | P-IL | 1.33 | 2.11 | 2.90 | 2.11 | 0.23 | 0.62 | 1.27 | 0.71 |
| OccNet | P-IL | 1.29 | 2.13 | 2.99 | 2.13 | 0.21 | 0.59 | 1.37 | 0.72 |
| UniAD | P-IL | 0.48 | 0.96 | 1.65 | 1.03 | 0.05 | 0.17 | 0.71 | 0.31 |
| VAD | P-IL | 0.41 | 0.70 | 1.05 | 0.72 | 0.07 | 0.18 | 0.43 | 0.23 |
| UncAD | P-IL | 0.33 | 0.59 | 0.94 | 0.62 | 0.10 | 0.14 | 0.28 | 0.17 |
| PPAD | P-IL | 0.31 | 0.56 | 0.87 | 0.58 | 0.08 | 0.12 | 0.38 | 0.19 |
| PARA-Drive | P-IL | 0.25 | 0.46 | 0.74 | 0.48 | 0.14 | 0.23 | 0.39 | 0.25 |
| GenAD | P-IL | 0.28 | 0.49 | 0.78 | 0.52 | 0.08 | 0.14 | 0.34 | 0.19 |
| LAW (perception-based) | P-IL | 0.24 | 0.46 | 0.76 | 0.49 | 0.08 | 0.10 | 0.39 | 0.19 |
| DiffusionDrive | P-IL | 0.27 | 0.54 | 0.90 | 0.57 | 0.03 | 0.05 | 0.16 | 0.08 |
| Epona | SS-L | 0.61 | 1.17 | 1.98 | 1.25 | 0.01 | 0.22 | 0.85 | 0.36 |
| LAW (perception-free) | SS-L | 0.26 | 0.57 | 1.01 | 0.61 | 0.14 | 0.21 | 0.54 | 0.30 |
| World4Drive | SS-L | 0.23 | 0.47 | 0.81 | 0.50 | 0.02 | 0.12 | 0.33 | 0.16 |
| WorldRFT without RFT | SS-L | 0.21 | 0.44 | 0.76 | 0.47 | 0.10 | 0.11 | 0.23 | 0.15 |
| **WorldRFT with RFT** | SS-L & RL | 0.22 | 0.44 | 0.77 | 0.48 | **0.00** | **0.00** | 0.16 | **0.05** |
| SSR† ⚠ | SS-L | 0.19 | 0.36 | 0.62 | 0.39 | 0.00 | 0.10 | 0.20 | 0.13 |
| **WorldRFT with RFT†** | SS-L & RL | **0.15** | **0.30** | **0.56** | **0.33** | 0.00 | 0.00 | 0.12 | **0.04** |

- **Averages**: all but one reproduce from the per-horizon values (OccNet and WorldRFT† truncate rather than round). ⚠ **SSR†'s collision average is 0.13 but its horizons average 0.10.** [[sources/resworld.md]] prints SSR's ego-status row with the same L2 and collision 0.10 / 0.10 / 0.24 (average 0.15), so the per-horizon values here look mis-copied. WorldRFT† is lower under either.
- **The table crosses averaging conventions.** UniAD's 1.03 / 0.31 is its per-horizon value; in the temporal-average convention it is 0.69 / 0.12 ([[sources/foresight.md]]'s table; [[concepts/nuscenes-waymo-evals.md]]). Epona's 1.25 / 0.36 is in the per-horizon convention as well ([[concepts/world-model-for-ad.md]]'s nuScenes table). The LAW perception-free row matches the temporal-average copies in [[sources/foresight.md]] and [[sources/resworld.md]], so WorldRFT is in that convention. At UniAD's comparable 0.12, World4Drive (0.16) and WorldRFT without RFT (0.15) are both behind it on collision.
- **"Lowest collision rate among all evaluated methods"** holds in the table (0.05 against DiffusionDrive 0.08). L2 with RFT is 0.01 worse than without.
- The ego-status variant is "detailed in the supplementary materials", which do not describe it.

### Table 2: NAVSIM-v1 navtest

Residual = closed form NC·DAC·(5 TTC + 2 C + 5 EP)/12 − reported PDMS ([[concepts/navsim-benchmark.md#residual-sign]]).

| Method | Training | Input | NC | DAC | TTC | Comf. | EP | PDMS | Residual |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| UniAD | P-IL | C | 97.8 | 91.9 | 92.9 | 100.0 | 78.8 | 83.4 | −4.12 |
| PARA-Drive | P-IL | C | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 | −4.01 |
| LTF | P-IL | C | 97.4 | 92.8 | 92.4 | 100.0 | 79.0 | 83.8 | −4.18 |
| Transfuser | P-IL | C & L | 97.7 | 92.8 | 92.8 | 100.0 | 79.2 | 84.0 | −3.91 |
| VADv2 | P-IL | C & L | 97.2 | 89.1 | 91.6 | 100.0 | 76.0 | 80.9 | −5.99 |
| Hydra-MDP | P-IL | C & L | 97.9 | 91.7 | 92.9 | 100.0 | 77.6 | 83.0 | −4.26 |
| DiffusionDrive | P-IL | C & L | 98.2 | 96.2 | 94.7 | 100.0 | 82.2 | **88.1** | −2.72 |
| Epona | SS-L | C | 97.9 | 95.1 | 93.8 | 99.9 | 80.4 | 86.2 | −3.12 |
| LAW (perception-free) | SS-L | C | 96.4 | 95.4 | 88.7 | 99.9 | 81.7 | 84.6 | −3.99 |
| DriveX | SS-L | C | 97.5 | 94.0 | 93.0 | 100.0 | 79.7 | 84.5 | −3.28 |
| World4Drive | SS-L | C | 97.4 | 94.3 | 92.8 | 100.0 | 79.9 | 85.1 | −3.70 |
| WorldRFT without RFT | SS-L | C | 97.5 | 96.0 | 94.0 | 100.0 | 80.9 | 87.0 | −3.19 |
| **WorldRFT with RFT** | SS-L & RL | C | 97.8 | **96.8** | 94.0 | 100.0 | 81.7 | 87.8 | −2.71 |

- **All thirteen rows pass** (−2.71 to −5.99). The UniAD, PARA-Drive, Transfuser, VADv2 and DiffusionDrive rows match [[sources/sparsedrivev2.md]]'s, LAW and World4Drive [[sources/foresight.md]]'s, and Epona [[sources/dreameraD.md]]'s. Hydra-MDP is a weaker row (83.0) than the 86.5 SparseDriveV2 prints.
- WorldRFT has the table's best DAC; DiffusionDrive leads NC, TTC and EP.
- The paper calls NAVSIM "closed-loop"; it is a non-reactive simulation of a single plan ([[concepts/navsim-benchmark.md]]).

### Table 3: component ablation on nuScenes (all with RFT)

| ID | VGGT | Target | Path | Refinement | L2 | CR |
|---:|:---:|:---:|:---:|:---:|---:|---:|
| 1 | | | | | 0.59 | 0.16 |
| 2 | | ✓ | | ✓ | 0.56 | 0.10 |
| 3 | | | ✓ | ✓ | 0.54 | 0.11 |
| 4 | | ✓ | ✓ | ✓ | 0.52 | 0.08 |
| 5 | ✓ | | ✓ | ✓ | 0.51 | 0.08 |
| 6 | ✓ | ✓ | | ✓ | 0.49 | 0.09 |
| 7 | ✓ | ✓ | ✓ | | 0.50 | 0.06 |
| **8** | ✓ | ✓ | ✓ | ✓ | **0.48** | **0.05** |

- ID 1 is the LAW-style baseline with RFT, presumably with Grounded-SAM (Table 7 shows the no-VGGT rows include it): its 0.16 collision is already below LAW's published 0.30, so RFT on the bare baseline is plausibly worth more than on the full model, but the baseline without RFT is not reported.
- The world-model objective $\mathcal L_\text{rec}$ is in every row. **No row removes the latent world model**, so the title's mechanism is not priced.

### Table 4: component ablation on NAVSIM (all with RFT; PDMS)

| ID | VGGT | Target | Path | Refinement | PDMS |
|---:|:---:|:---:|:---:|:---:|---:|
| 1 | | | | | 85.0 |
| 2 | | ✓ | | ✓ | 86.0 |
| 3 | | | ✓ | ✓ | 86.1 |
| 4 | | ✓ | ✓ | ✓ | 86.5 |
| 5 | ✓ | | ✓ | ✓ | 87.2 |
| 6 | ✓ | ✓ | | ✓ | 87.1 |
| 7 | ✓ | ✓ | ✓ | | 87.6 |
| **8** | ✓ | ✓ | ✓ | ✓ | **87.8** |

- The ordering matches Table 3 except that ID 6 is better than ID 5 on nuScenes L2 and worse on PDMS.
- ID 1 (85.0, with RFT) is 0.1 below World4Drive without RFT (85.1). No sub-scores are given, so the DAC attribution in the text cannot be checked.

### Table 5: backbone and feature dimension (nuScenes)

| ID | Backbone | Dim. | L2 | CR |
|---:|---|---:|---:|---:|
| 1 | ResNet-34 | 256 | 0.50 | 0.09 |
| 2 | ResNet-50 | 128 | 0.51 | 0.08 |
| 3 | ResNet-50 | 256 | 0.48 | 0.05 |
| 4 | ResNet-50 | 384 | 0.46 | 0.06 |
| 5 | ResNet-101 | 256 | **0.46** | **0.04** |

The headline uses ID 3. ResNet-101 is better on both columns; "256 provides the optimal balance" refers to compute, which is not reported.

### Table 6: refinement iterations K (nuScenes)

| K | L2 | CR |
|---:|---:|---:|
| 1 | 0.49 | 0.08 |
| 3 | 0.48 | 0.05 |
| 6 | 0.48 | 0.07 |

Without refinement (Table 3 ID 7) the collision rate is 0.06, so K = 1 and K = 6 are no better than none on collision. The paper reads K = 6 as "potential overfitting"; the difference from K = 3 is about one sample.

### Table 7: latent encoder components (nuScenes)

| ID | VGGT | Grounded-SAM | L2 | CR |
|---:|:---:|:---:|---:|---:|
| 1 | ✓ | | 0.49 | 0.06 |
| 2 | | ✓ | 0.52 | 0.08 |
| 3 | ✓ | ✓ | 0.48 | 0.05 |

Row 2 equals Table 3 ID 4 and row 3 equals ID 8, so the earlier table's no-VGGT rows include Grounded-SAM. Grounded-SAM's own contribution given VGGT is 0.01 m and 0.01 points.

### Table 8: spatial-path configuration (nuScenes)

| Points | Spacing (m) | L2 | CR |
|---:|---:|---:|---:|
| 30 | 1 | 0.50 | 0.12 |
| 30 | 2 | 0.48 | 0.10 |
| 50 | 1 | 0.49 | 0.09 |
| **50** | **2** | **0.48** | **0.05** |
| 80 | 1 | 0.52 | 0.11 |
| 80 | 2 | 0.51 | 0.12 |

- **The implementation details say 30 points at 2 m for nuScenes**; this table puts the headline (0.48 / 0.05) at 50 points, and 30 × 2 m at 0.10. One of the two is wrong.
- Five of six configurations have collision 0.09–0.12, no better than the model without a path (Table 3 ID 6, 0.09).
- 50 points at 2 m is a 100 m path, longer than any 3 s trajectory; Figure 6 shows the path running past the target region. How the path ground truth is extended beyond the trajectory is not stated ("uniform interpolation of the ground-truth trajectory").

### Table 9: target modelling (nuScenes)

| Target | L2 | CR |
|---|---:|---:|
| Deterministic point | 0.53 | 0.10 |
| Probabilistic region (Laplace) | **0.48** | **0.05** |

### Table 10: share of data used for RFT (nuScenes)

| Data | L2 | CR |
|---:|---:|---:|
| 10% | 0.47 | 0.12 |
| 20% | 0.47 | 0.10 |
| 100% | 0.48 | 0.05 |

Starting from 0.15 without RFT. A binary collision reward is zero in most scenes and every group sample, so most scenes give the policy gradient nothing; how a group with zero reward variance is normalized is not stated. That may explain why the data share matters.

### Table 11: supervised versus reinforcement fine-tuning (nuScenes)

| Method | L2 | CR |
|---|---:|---:|
| SFT ("the same safety-aware loss") | 0.48 | 0.10 |
| RFT | 0.48 | 0.05 |

### Table 12: GRPO group size (nuScenes)

| G | L2 | CR |
|---:|---:|---:|
| 5 | 0.48 | 0.08 |
| 10 | 0.48 | 0.05 |
| 15 | 0.48 | 0.07 |

The text explains G = 5 as "less stable advantage estimation" and G = 15 as costlier "without significant performance gains"; G = 15 is two samples worse, not merely no better.

---

## Qualitative Figures

![[fig3 1.png|Figure 4: one nuScenes scene. Left: front, front-right, back and back-right cameras, with a box truck close on the front right; the front view overlays the predicted trajectory before RFT (blue), after RFT (green) and the ground truth (red). Right: BEV with map lines and agent boxes; two zoomed panels show the pre-RFT trajectory running into the box beside it and the RFT trajectory turning on a wider arc away from it.]]

**Figure 4**: before RFT the trajectory cuts close to the parked truck; after RFT it keeps clear and is nearer the ground truth. One scene.

![[fig4 1.png|Figure 5: three front cameras (front-left, front, front-right) of one scene; top row the images, middle row LAW's attention map, bottom row WorldRFT's. LAW's strongest responses are along the bottom edge of the front image and spread over the road and buildings; WorldRFT's peak on the parked and moving cars in all three views.]]

**Figure 5**: one scene's attention maps. The claim that WorldRFT attends to "planning-critical agents" rests on this one scene; no attention statistic is reported (compare [[sources/auto-jepa.md]]'s occlusion protocol, which measures it).

![[sup_fig1.png|Figure 6: a straight-driving nuScenes scene. Left: six cameras, with the temporal trajectory overlaid in the front view. Right: BEV with the target centre (star), a narrow longitudinal target-region ellipse, the spatial path (green) running past the target, and the temporal trajectory (blue).]]

![[sup_fig2.png|Figure 7: a second nuScenes scene. Left: six cameras. Right: BEV in which the target region is a large, nearly round ellipse ahead of the ego and the temporal trajectory heads towards it.]]

**Figures 6 and 7** show the Laplace scale doing what the paper says: a narrow longitudinal ellipse when only speed is uncertain, a broad one when direction is too. Two scenes; no calibration of $b$ against error is reported.

![[sup_fig3.png|Figure 8: failure case. Left: six cameras at an intersection, with WorldRFT's temporal trajectory overlaid in the front view. Right: BEV with the ground truth (red), WorldRFT (blue) and LAW (pink); LAW's trajectory lies on the ground truth while WorldRFT's departs at a small angle.]]

**Figure 8** (failure case): the paper attributes it to a mislabelled command ("a left turn ... labelled as 'go straight'"). As drawn, LAW, given the same command, follows the ground truth, so the mislabel does not by itself explain WorldRFT's departure. The main text's "Figure 8 shows predicted trajectories: ground truth (red), without RFT (blue), and with RFT (green)" describes Figure 4.

---

## Table and Text Audit

| Check | Result |
|---|---|
| NAVSIM-v1 closed form, 13 rows | All pass (−2.71 to −5.99); WorldRFT −2.71 with RFT, −3.19 without |
| nuScenes averages, 17 rows | 16 reproduce; **SSR† collision average 0.13 against a per-horizon mean of 0.10** (ResWorld's copy: 0.10 / 0.10 / 0.24, 0.15) |
| nuScenes conventions | UniAD (1.03 / 0.31) and Epona (1.25 / 0.36) per-horizon; LAW, World4Drive and WorldRFT temporal-average |
| "Achieving the best performance among vision-only solutions" (NAVSIM) | Inside the table only; at least eight camera-only entries dated before it score 88.1–91.0 |
| "Lowest collision rate among all evaluated methods" (nuScenes) | True in the table; a few samples |
| "VGGT ... ID 4 → ID 5: +0.7 PDMS" | Two factors change; the one-factor contrast ID 4 → ID 8 is +1.3 |
| "DAC of 96.8 ... demonstrates that incorporating VGGT geometric priors substantially enhances ... spatial understanding" | DAC is 96.0 with VGGT before RFT; no ablation reports DAC |
| Spatial path, nuScenes | Implementation: 30 points × 2 m; Table 8: headline at 50 × 2 m |
| Ego-status variant | Promised in the supplement, not described |
| "Figure 8 shows ... without RFT (blue), and with RFT (green)" | That is Figure 4; Figure 8 is the failure case |
| "KL divergence" | The Gaussian NLL of the reference mean; $\beta$ inside $J$ and $\lambda$ outside are both applied, so the effective weight is unclear |
| Notation | $\alpha$ is both the refinement step (0.1) and the semantic-loss weight (0.2); $\beta$ both the reconstruction weight and the KL coefficient |
| Main-text table numbering | The nuScenes ablation is cited as "Table 4" and captioned Table 3 |

---

## Limitations

1. **The RL reward is the evaluation metric on logged agents.** On nuScenes it counts the collisions the metric counts, against annotated boxes that do not react. Nothing tests the policy against agents that respond to it.
2. **The world model is never ablated.** Every Table 3 and Table 4 row keeps LAW's reconstruction loss, so "planning-oriented latent world model" is a description of the architecture, not a measured effect.
3. **The safety numbers are a few samples and single runs**, and the hyperparameters were chosen on the evaluation split, where each chosen value is the only one at 0.05%.
4. **Not annotation-free in full.** Pretraining uses Grounded-SAM pseudo-labels and a frozen VGGT trained on 3D data; RFT uses ground-truth agent boxes.
5. **No cost reported** for a frozen VGGT on six (nuScenes) or three (NAVSIM) images per frame on top of the ResNet.
6. **NAVSIM-v1 only.** No v2, navhard, HUGSIM or Bench2Drive result in the paper; the v2 row other papers print is not in it.
7. **Comparison scope.** The NAVSIM table stops at the latent-world-model cohort and DiffusionDrive; the nuScenes table mixes conventions.
8. **RL details left open**: how groups with zero reward variance are normalized, how many RFT epochs run on nuScenes, and what the SFT baseline's loss is.

---

## Relation to Other Wiki Pages

- [[sources/dreameraD.md]]: the same first author three months later, with RL moved inside the world model (latent rollouts, learned reward, vocabulary sampling) and a re-implementation of WorldRFT's sampler as an ablation. It prints WorldRFT's v2 row.
- [[sources/latent-wam.md]]: the same group's perception-free latent world model with a geometry teacher (WorldMirror/VGGT) distilled and discarded at inference, where WorldRFT keeps a frozen VGGT in the input path. Its v2 table also carries WorldRFT's 86.7.
- [[sources/simvta.md]], [[sources/simwam.md]], [[sources/rad.md]]: other RL post-training designs. SimVTA and SimWAM use the NAVSIM simulator's score; RAD, like WorldRFT, penalizes collisions with log-replayed agents, in 3DGS.
- [[sources/plan-r1.md]], [[sources/nord.md]]: GRPO's treatment of groups whose rewards barely vary, the issue a binary per-point collision reward raises.
- [[sources/sgdrive.md]], [[sources/drivefuture.md]]: other goal and intent heads ([[concepts/intent-conditioned-planning.md#worldrft]]).
- [[sources/sparsedrivev2.md]]: the path / speed factorization as a vocabulary rather than an auxiliary output.
- [[sources/geoworldad.md]], [[sources/coworld-vla.md]]: other VGGT-lineage geometry models in a planner.
- [[sources/resworld.md]], [[sources/foresight.md]]: nuScenes tables in the same convention, with matching LAW and World4Drive rows; ResWorld's SSR† row disagrees with this one.
- [[sources/metis.md]]: prints the pre-RFT row as WorldRFT's RL result.
- [[sources/law.md]]: the base. LAW's own ablation is the only measurement of the latent loss WorldRFT keeps (+7.1 PDMS on a 77.5 perception-free base; −0.10 m / −0.11 points on nuScenes).
- Not ingested: **World4Drive** (the predecessor), **SSR**, **DriveX**, **UncAD**.

---

## Open Questions

- **Does collision-only RFT against logged boxes reduce collisions with agents that react?** A reactive closed-loop run (HUGSIM, Bench2Drive, or NAVSIM's navhard Stage 2) of the pre- and post-RFT checkpoints would say whether 0.15 → 0.05 is collision avoidance or metric fitting.
- **Why does a collision reward raise DAC and EP on NAVSIM** and leave TTC unchanged? Running RFT with the reward zeroed (regularizers only) would separate the reward from the entropy and reference terms.
- **What does the latent world model add?** Removing $\mathcal L_\text{rec}$ from the full model is one run, and it is the one the title needs.
- **What does the frozen VGGT cost, and does a distilled version keep the +1.3?** Latent-WAM's teacher-only design is the obvious comparison.
- **Is the Laplace scale calibrated?** Correlating $b$ with endpoint error, and testing a multimodal target (a mixture), would show whether the region is uncertainty or only a better-conditioned loss.

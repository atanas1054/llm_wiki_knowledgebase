---
title: "RAD: Training an End-to-End Driving Policy via Large-Scale 3DGS-based Reinforcement Learning"
type: source-summary
sources: ["raw/papers/RAD_ Training an End-to-End Driving Policy via Large-Scale 3DGS-based Reinforcement Learning.md"]
related: [concepts/rl-for-ad.md, concepts/data-driven-simulators.md, concepts/evaluation-variance.md, concepts/action-tokenization.md, concepts/dual-system-vla.md, sources/senna2.md, sources/simscale.md, sources/think2drive.md, sources/raw2drive.md, sources/learning-to-drive-from-a-world-model.md, sources/dreameraD.md, sources/diffusiondrive.md, research-directions.md]
created: 2026-10-06
updated: 2026-10-06
confidence: medium
---

# RAD: Training an End-to-End Driving Policy via Large-Scale 3DGS-based Reinforcement Learning

**Paper**: RAD: Training an End-to-End Driving Policy via Large-Scale 3DGS-based Reinforcement Learning
**Authors**: Hao Gao, Shaoyu Chen, Bo Jiang, Bencheng Liao, Yiang Shi, Xiaoyang Guo, Yuechuan Pu, Haoran Yin, Xiangyu Li, Xinbang Zhang, Ying Zhang, Wenyu Liu, Qian Zhang, Xinggang Wang
**Orgs**: Huazhong University of Science and Technology; Horizon Robotics. The same HUST group as [[sources/diffusiondrive.md]] and [[sources/foredrive.md]], and the same pairing of institutions as [[sources/senna2.md]].
**arXiv**: 2502.13144v2 (first version February 2025; cited elsewhere as NeurIPS 2025)
**Code**: released (github.com/hustvl/RAD); the training data and 3DGS environments are internal
**Source**: `raw/papers/RAD_ Training an End-to-End Driving Policy via Large-Scale 3DGS-based Reinforcement Learning.md`

---

## What It Is

The first ingested paper that trains a **camera-input driving policy by online reinforcement learning inside 3D Gaussian Splatting (3DGS) reconstructions of real driving**. The wiki had cited it from [[concepts/rl-for-ad.md]], [[sources/senna2.md]] and [[sources/simscale.md]] without a page.

**Three stages** (Figure 2):

| Stage | Trained | Signal |
|---|---|---|
| 1. Perception pre-training | BEV encoder, map head, agent head | Automatically generated map and agent labels |
| 2. Planning pre-training | Image encoder, planning head (perception frozen) | Imitation of 2,000 hours of expert driving |
| 3. Reinforced post-training | Image encoder, planning head (perception frozen) | PPO on rollouts in 3,968 reconstructed clips, alternating 4:1 with imitation steps |

- **Policy output**: two independent categorical distributions over a 0.5 s step, 61 lateral displacements (±0.75 m) and 61 longitudinal displacements (0–15 m). Each has its own value head.
- **Environment**: per-clip 3DGS reconstructions (extended StreetGaussian), 8 s clips at 10 Hz, ego moved by a kinematic bicycle model. **Other agents are replayed from the log.**
- **Reward**: four penalties: dynamic collision, static collision, position deviation from the expert trajectory (> 2 m), heading deviation (> 40°). Any of them ends the episode.
- **Optimization**: PPO with lateral and longitudinal advantages computed separately (dynamic collision drives longitudinal; the other three lateral), plus four **directional auxiliary losses** that push probability mass toward braking, steering away or steering back.

**Results** on its own benchmark of 337 held-out 3DGS clips:

| Method | Collision ratio | Deviation ratio | Avg. deviation distance (m) |
|---|---:|---:|---:|
| TransFuser | 0.320 | 0.235 | 0.263 |
| VAD | 0.335 | 0.314 | 0.304 |
| GenAD | 0.341 | 0.291 | 0.265 |
| VADv2 | 0.270 | 0.243 | 0.273 |
| RAD, imitation-only post-training | 0.229 | 0.066 | 0.238 |
| **RAD** | **0.089** | **0.063** | 0.257 |

---

## Key Takeaways

- **RL in 3DGS cuts the collision ratio from 0.229 to 0.089** against the same architecture post-trained by imitation only (77 to 30 of 337 clips), and 3× against VADv2 (0.270). The headline is its own benchmark.
- **Plain PPO does not beat imitation here; the hand-designed auxiliary losses do.** PPO with no auxiliary objective reaches 0.249, worse than imitation-only (0.229). The four auxiliary losses alone, without PPO, reach 0.187; together with PPO, 0.089. The auxiliary losses are rules: "if a collision is ahead, shift probability toward slower actions" and so on, weighted by the PPO advantage.
- **Imitation regularization is what keeps it human-like.** RL alone: collision 0.143 but deviation distance 0.345 m; with imitation steps interleaved 4:1, 0.089 and 0.257 m.
- **The other cars do not react.** RAD's environment supports IDM and learned agents, but the experiments use log replay "to fully recover the real-world traffic flow". One auxiliary loss rewards *accelerating* when a collision comes from behind, which is how a log-replayed follower that cannot brake shows up in the training signal.
- **Another 3DGS benchmark from the same institutions disagrees by a factor of three.** [[sources/senna2.md]]'s 256-clip 3DGS benchmark scores RAD's collision ratio at 0.281 (against 0.089 here) and VADv2 at 0.422 (0.270 here). RAD's advantage over VADv2 is 3.0× here and 1.5× there.
- **Training reward and evaluation metric are the same quantities.** The reward penalizes collisions and deviation from the expert past fixed thresholds; the benchmark counts clips with collisions and deviations past thresholds. The imitation-only baselines never saw a rendered image.

---

## Method

### Policy (Section 3.1)

- **BEV encoder** over multi-view images; **map tokens** (centerlines, dividers, boundaries, arrows, signals) and **agent tokens** (pose, size, speed, multi-mode futures) learned from it; a separate **image encoder** adds dense image tokens.
- **Planning head**: a cascaded Transformer decoder with a planning query over the scene tokens, combined with navigation and ego-state embeddings:
$$\pi(a^x\mid s) = \text{softmax}\big(\text{MLP}(\phi(E_\text{plan}, E_\text{scene}) + E_\text{navi} + E_\text{state})\big)$$
  and the same for $a^y$, plus value heads $V_x(s)$ and $V_y(s)$.
- **Action space** (Appendix A.2): $a^x \in \{-0.75, \dots, 0, \dots, 0.75\}$ m, 61 values; $a^y \in \{0, \dots, 15\}$ m, 61 values; both over 0.5 s, assuming constant linear and angular velocity. Imitation targets are the nearest anchors to the logged 0.5 s position, learned with a focal loss per axis.

### Environment (Section 3.3, Appendix A.1)

- **Reconstruction**: StreetGaussian, extended with a road-surface mesh that constrains Gaussians, a separate sky model, and pose optimization plus depth and normal consistency for foreground objects, aimed at off-trajectory views.
- **Ego update**: kinematic bicycle model, $x_{t+1} = x_t + v_t\cos\psi_t\,\Delta t$, $y_{t+1} = y_t + v_t\sin\psi_t\,\Delta t$, $\psi_{t+1} = \psi_t + \frac{v_t}{L}\tan\delta_t\,\Delta t$, with $v_t$ and $\delta_t$ derived from the action.
- **Other agents**: log replay in all experiments.
- **Parallelism**: 32 workers each sample a reconstructed clip, roll out the current policy and write $(s_t, a_t, r_{t+1}, s_{t+1})$ to a shared buffer; the policy is synchronized to the workers after a fixed number of steps.

### Reward (Section 3.4)

$$\mathcal R = \{r_\text{dc}, r_\text{sc}, r_\text{pd}, r_\text{hd}\}$$

| Term | Triggered when |
|---|---|
| $r_\text{dc}$ | The ego box overlaps an annotated box of a dynamic agent |
| $r_\text{sc}$ | The ego box overlaps the Gaussians of a static obstacle |
| $r_\text{pd}$ | Distance to the closest point of the expert trajectory > $d_\max$ = 2.0 m |
| $r_\text{hd}$ | Heading difference to the matched expert heading > $\psi_\max$ = 40° |

Each event ends the episode, "because after such events occur, the 3DGS environment typically generates noisy sensor data". Reward magnitudes are not given.

### Policy optimization (Section 3.5, Appendix A.3)

Rewards are split by axis, $r^x_t = r^\text{sc}_t + r^\text{pd}_t + r^\text{hd}_t$ and $r^y_t = r^\text{dc}_t$; GAE ($\gamma = 0.9$, $\lambda = 0.95$) gives $\hat A^x_t$ and $\hat A^y_t$, further decomposed by reward type. The PPO objective is the sum of two independently clipped terms ($\epsilon_x = 0.1$, $\epsilon_y = 0.2$):

$$\mathcal L^\text{PPO}_x = \mathbb E_t\big[\min(\rho^x_t\hat A^x_t,\ \text{clip}(\rho^x_t, 1-\epsilon_x, 1+\epsilon_x)\hat A^x_t)\big]$$

and likewise for $y$.

### Auxiliary objectives (Section 3.6, Appendix A.4)

With $\Delta\pi^\text{dec}_y$ / $\Delta\pi^\text{acc}_y$ the probability mass on slower / faster longitudinal actions than the one taken, and $\Delta\pi^\text{left}_x$ / $\Delta\pi^\text{right}_x$ likewise for lateral:

| Loss | Form | Direction factor |
|---|---|---|
| $\mathcal L_\text{dc}$ | $\mathbb E_t[\hat A^\text{dc}_t f_\text{dc}(\Delta\pi^\text{dec}_y - \Delta\pi^\text{acc}_y)]$ | $f_\text{dc} = 1$ if the collision is ahead, **−1 if behind** |
| $\mathcal L_\text{sc}$ | $\mathbb E_t[\hat A^\text{sc}_t f_\text{sc}(\Delta\pi^\text{right}_x - \Delta\pi^\text{left}_x)]$ | 1 if the obstacle is on the left, −1 if on the right |
| $\mathcal L_\text{pd}$ | same lateral form with $\hat A^\text{pd}_t$ | 1 if the ego deviates left, −1 if right |
| $\mathcal L_\text{hd}$ | same lateral form with $\hat A^\text{hd}_t$ | 1 if the ego deviates clockwise, −1 if counter-clockwise |

$$\mathcal L(\theta) = \mathcal L^\text{PPO}(\theta) + \lambda_1\mathcal L_\text{dc} + \lambda_2\mathcal L_\text{sc} + \lambda_3\mathcal L_\text{pd} + \lambda_4\mathcal L_\text{hd}$$

The $\lambda$ values are not given.

### Training schedule (Appendix A.5)

Cycles of four RL rounds and one IL round. Each RL round is 320 iterations (four 8 s clips at 10 Hz) over a first-in, first-out window of four clips, with one new clip collected per round. The 4:1 ratio "yields the best results" (no table).

---

## Figures

![[teaserv3-nips.png|Three training paradigms. (a) Imitation learning: real-world sensor data and human demonstrations supervise the policy; limitations open-loop gap and causal confusion. (b) Simulator-based RL, for example CARLA: simulated sensor data, action and reward; limitations sim-to-real gap and naive actor behaviour. (c) RAD: reinforcement learning with rollouts in large-scale 3DGS environments plus imitation learning from human demonstrations to keep similarity to humans; advantages realistic digital world, causation modelling, narrower open-loop gap]]

*Figure 1: Training paradigms.*

![[framework-2.png|RAD framework. Top: sensor data feeds a BEV encoder and perception head and a separate image encoder; a planning head outputs lateral and longitudinal action distributions. Bottom: Stage 1 trains the BEV encoder and perception head on map and agent ground truth; Stage 2 freezes them and trains the image encoder and planning head by imitation; Stage 3 runs a loop of state, policy, action, pose, 3DGS and reward with reinforcement learning, plus imitation learning from selected human demonstrations]]

*Figure 2: Three-stage training.*

![[post-training-nipsv2.png|Post-training diagram. N parallel workers each roll out a frozen copy of the policy in a 3DGS environment and record state, action, reward and next state into a rollout buffer. The policy is updated by RL steps on the rollout data and IL steps on selected human demonstrations, then the update is sent back to the workers]]

*Figure 3: Parallel rollouts and alternating RL / IL updates.*

![[reward-nipsv2.png|Four rendered front-camera frames illustrating reward triggers: a collision with a car ahead, a collision with a roadside barrier, the ego drifting onto a curb, and the ego heading toward an adjacent lane. The renders are visibly blurred and smeared, especially at the collision]]

*Figure 4: The four penalty events. The frames show the rendering degradation that the paper cites as the reason for ending an episode at each event.*

![[Consistencyv4.png|Consistency check between real world and 3DGS environment for one scenario. Left: first and last front-camera frames from the real-world run and from the 3DGS run, nearly identical, with the ego ending behind an SUV. Right: relative X and Y position of the ego over 10 seconds in both environments; the curves overlap closely, with X diverging by a few tenths of a metre after about 4 seconds]]

*Figure 5: The same policy driven in the real world and in the reconstructed environment. One scenario; the position curves agree within a few tenths of a metre over 10 s.*

![[vis-nips-v2.png|Two rows of front-camera frames at T equals 0, 20, 40 and 70 to 76 in a rendered crosswalk scene with crossing pedestrians. Top row, imitation-only policy: it keeps moving and pedestrians loom close in front. Bottom row, RAD: it waits and the pedestrians cross ahead at a distance]]

*Figure 6: Yielding to pedestrians, imitation-only policy (top) against RAD (bottom).*

![[vs2-2.png|More qualitative comparisons between the imitation-only policy and RAD in four scenarios: detour, crawling in dense traffic, traffic congestion and a U-turn]]

*Figure 7: Detour (rows 1–2), crawling in dense traffic (3–4), congestion (5–6), U-turn (7–8); imitation-only against RAD.*

---

## Tables

Metrics (Appendix A.6), all over the 337 evaluation clips: CR = DCR + SCR (fractions of clips with a dynamic / static collision); DR = PDR + HDR (fractions of clips exceeding the position / heading deviation thresholds); ADD = mean closest distance to the expert trajectory over steps before any event; jerk in the longitudinal and lateral directions.

### Table 1: Training strategy in Stage 3

| Stage 3 | CR | DCR | SCR | DR | PDR | HDR | ADD | Long. jerk | Lat. jerk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IL | 0.229 | 0.211 | 0.018 | 0.066 | 0.039 | 0.027 | **0.238** | **3.928** | 0.103 |
| RL | 0.143 | 0.128 | 0.015 | 0.080 | 0.065 | **0.015** | 0.345 | 4.204 | 0.085 |
| **RL + IL** | **0.089** | **0.080** | **0.009** | **0.063** | 0.042 | 0.021 | 0.257 | 4.495 | **0.082** |

### Table 2: Reward components

| ID | Dyn. col. | Sta. col. | Pos. dev. | Head. dev. | CR | DCR | SCR | DR | PDR | HDR | ADD | Long. jerk | Lat. jerk |
|---|:-:|:-:|:-:|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ✓ | | | | 0.172 | 0.154 | 0.018 | 0.092 | 0.033 | 0.059 | 0.259 | 4.211 | 0.095 |
| 2 | | ✓ | ✓ | ✓ | 0.238 | 0.217 | 0.021 | 0.090 | 0.045 | 0.045 | 0.241 | 3.937 | 0.098 |
| 3 | ✓ | | ✓ | ✓ | 0.146 | 0.128 | 0.018 | 0.060 | 0.030 | 0.030 | 0.263 | 3.729 | 0.083 |
| 4 | ✓ | ✓ | | ✓ | 0.151 | 0.142 | 0.009 | 0.069 | 0.042 | 0.027 | 0.303 | 3.938 | 0.079 |
| 5 | ✓ | ✓ | ✓ | | 0.166 | 0.157 | 0.009 | 0.048 | 0.036 | 0.012 | 0.243 | 3.334 | 0.067 |
| **6** | ✓ | ✓ | ✓ | ✓ | **0.089** | **0.080** | 0.009 | 0.063 | 0.042 | 0.021 | 0.257 | 4.495 | 0.082 |

### Table 3: Auxiliary objectives

| ID | PPO | Dyn. col. aux. | Sta. col. aux. | Pos. dev. aux. | Head. dev. aux. | CR | DCR | SCR | DR | PDR | HDR | ADD | Long. jerk | Lat. jerk |
|---|:-:|:-:|:-:|:-:|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ✓ | | | | | 0.249 | 0.223 | 0.026 | 0.077 | 0.047 | 0.030 | 0.266 | 4.209 | 0.104 |
| 2 | ✓ | ✓ | | | | 0.178 | 0.163 | 0.015 | 0.151 | 0.101 | 0.050 | 0.301 | 3.906 | 0.085 |
| 3 | ✓ | | ✓ | ✓ | ✓ | 0.137 | 0.125 | 0.012 | 0.157 | 0.145 | 0.012 | 0.296 | 3.419 | 0.071 |
| 4 | ✓ | ✓ | | ✓ | ✓ | 0.169 | 0.151 | 0.018 | 0.075 | 0.042 | 0.033 | 0.254 | 4.450 | 0.098 |
| 5 | ✓ | ✓ | ✓ | | ✓ | 0.149 | 0.134 | 0.015 | 0.063 | 0.057 | 0.006 | 0.324 | 3.980 | 0.086 |
| 6 | ✓ | ✓ | ✓ | ✓ | | 0.128 | 0.119 | 0.009 | 0.066 | 0.030 | 0.036 | 0.254 | 4.102 | 0.092 |
| 7 | | ✓ | ✓ | ✓ | ✓ | 0.187 | 0.175 | 0.012 | 0.077 | 0.056 | 0.021 | 0.309 | 5.014 | 0.112 |
| **8** | ✓ | ✓ | ✓ | ✓ | ✓ | **0.089** | **0.080** | **0.009** | 0.063 | 0.042 | 0.021 | 0.257 | 4.495 | 0.082 |

The text attributes the "auxiliary objectives without PPO" result (0.187) to ID 1; it is ID 7. ID 1 is PPO without auxiliary objectives (0.249).

### Table 4: Comparison with imitation-learning methods

| Method | CR | DCR | SCR | DR | PDR | HDR | ADD | Long. jerk | Lat. jerk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TransFuser | 0.320 | 0.273 | 0.047 | 0.235 | 0.188 | 0.047 | 0.263 | 4.538 | 0.142 |
| VAD | 0.335 | 0.273 | 0.062 | 0.314 | 0.255 | 0.059 | 0.304 | 5.284 | 0.550 |
| GenAD | 0.341 | 0.299 | 0.042 | 0.291 | 0.160 | 0.131 | 0.265 | 11.37 | 0.320 |
| VADv2 | 0.270 | 0.240 | 0.030 | 0.243 | 0.139 | 0.104 | 0.273 | 7.782 | 0.171 |
| **RAD** | **0.089** | **0.080** | **0.009** | **0.063** | **0.042** | **0.021** | **0.257** | **4.495** | **0.082** |

"For fair comparisons, all the methods are trained with the same amount of human driving demonstrations."

### Table 5: Planning pre-training hyperparameters

| Setting | Value |
|---|---|
| Learning rate / schedule | 1e-4, cosine |
| Optimizer | AdamW (β = 0.9, 0.999; ε = 1e-8), weight decay 1e-4 |
| Batch size / steps | 512 / 30k |
| Planning head dim | 256 |
| GPUs | 128 × RTX 4090 |

### Table 6: Reinforced post-training hyperparameters

| Setting | Value |
|---|---|
| Learning rate / schedule | 5e-6, cosine |
| Optimizer | AdamW (as above), weight decay 1e-4 |
| RL workers / RL batch / IL batch | 32 / 32 / 128 |
| GAE | γ = 0.9, λ = 0.95 |
| PPO clipping | ε_x = 0.1, ε_y = 0.2 |
| Deviation thresholds | d_max = 2.0 m, ψ_max = 40° |
| Planning head / value function dim | 256 / 256 |
| GPUs | 32 × RTX 4090 |

### Results given only in the text

| Quantity | Value |
|---|---|
| Expert driving data | 2,000 hours, real-world, with automatically generated map and agent labels |
| Reconstructed clips | 4,305 (3,968 training, 337 evaluation), 8 s each at 10 Hz |
| Action space | 61 lateral × 61 longitudinal discrete displacements per 0.5 s |
| RL : IL step ratio | 4 : 1 |

---

## Reading the Results

### 1. What the collision gain is made of {#gain}

| Configuration (Stage 3) | CR | Clips with a collision (of 337) |
|---|---:|---:|
| Imitation only | 0.229 | 77 |
| PPO + imitation, **no auxiliary losses** | 0.249 | 84 |
| Auxiliary losses + imitation, **no PPO** | 0.187 | 63 |
| RL without imitation | 0.143 | 48 |
| **PPO + auxiliary losses + imitation** | **0.089** | **30** |

- **The auxiliary losses are necessary and PPO alone is not.** Without them PPO is slightly worse than imitation. They are hand-written corrections (brake if hit from the front, steer away from a static obstacle, steer back toward the expert), each multiplied by PPO's advantage for that event.
- **With them, PPO adds a lot** (0.187 → 0.089). So the gain is the combination: dense direction from rules, credit assignment over time from GAE.
- **RL alone reaches 0.143 while drifting from the expert** (ADD 0.345 m against 0.238 for imitation). Interleaved imitation brings the deviation back (0.257) and also lowers collisions further.
- **Every reward term is needed.** Without the dynamic-collision term, CR is 0.238, worse than imitation; with only that term, 0.172.
- **Comfort moves the other way longitudinally**: jerk 3.928 → 4.495 from imitation to RL + IL; lateral jerk falls 0.103 → 0.082.

### 2. Clip counts and noise {#noise}

The metrics are fractions of 337 clips. A collision ratio near 0.15 has a binomial standard error of about 0.019 (about 6 clips). The headline (77 → 30 clips) is many standard errors. Many ablation differences are not: Table 2's IDs 3–5 span 0.146–0.166 (49–56 clips) and Table 3's IDs 4–6 span 0.128–0.169. No seeds are reported. See [[concepts/evaluation-variance.md#outside-navsim]].

### 3. A non-reactive 3DGS world {#log-replay}

- **The experiments use log-replayed agents.** The paper chose replay "to fully recover the real-world traffic flow", over its IDM and learned-agent options.
- **The training signal shows it.** $\mathcal L_\text{dc}$'s direction factor is −1 "if the collision is behind", which pushes toward acceleration. With replayed traffic, a slower ego is struck by a car that never brakes; the policy is trained to keep up.
- **The interaction gains are against agents that do not respond.** Yielding to pedestrians (Figure 6) is learned against pedestrians who would walk the same path whatever the ego did. This is the "static world" failure mode on [[concepts/data-driven-simulators.md#failure-modes]].
- [[sources/senna2.md]] describes its 3DGS agents as reactive. [[sources/simscale.md]] uses IDM agents and measures reactive data as worth +0.7 / +1.1 navhard EPDMS.

### 4. Evaluating in the training renderer {#same-renderer}

- **RAD is trained and evaluated in 3DGS renders from the same pipeline**; the 337 evaluation clips are held out by scene, not by renderer. The imitation-only methods in Table 4 are trained on recorded images and see a rendered image for the first time at evaluation. Some of the gap may be familiarity with the renderer, the same confound as [[sources/simscale.md#navhard-gain]].
- **Episodes end at every event because the render degrades after it** (Figure 4). The renderer's failure region is cut out of training, not modelled.
- **The reward is the metric.** Collisions and deviations past 2 m / 40° from the expert are penalized in training and counted in evaluation.
- **Sim-to-real evidence is one scenario** (Figure 5): position curves of the same policy in the real world and in the reconstruction agree within a few tenths of a metre over 10 s. The paper calls the results "both qualitative and quantitative"; no aggregate is given.

### 5. The second benchmark {#second-benchmark}

[[sources/senna2.md]], also HUST with Horizon Robotics, evaluates RAD on its own 256-clip 3DGS benchmark:

| Method | CR, RAD's benchmark (337 clips) | CR, Senna-2's benchmark (256 clips) |
|---|---:|---:|
| TransFuser | 0.320 | 0.435 |
| VAD | 0.335 | 0.502 |
| GenAD | 0.341 | 0.557 |
| VADv2 | 0.270 | 0.422 |
| RAD | **0.089** | **0.281** |
| RAD / VADv2 | **0.33 (3.0× lower)** | **0.67 (1.5× lower)** |

- The ordering of the four imitation methods agrees; the size of RAD's advantage halves, and RAD's own collision ratio triples.
- The benchmarks differ in clips (both chosen as risky or dense), agent behaviour (replay here, reactive in Senna-2's description), and possibly the RAD checkpoint. Neither paper reconciles them.
- **"3× lower collision rate" is a property of one private benchmark.** No public benchmark result exists for RAD: no NAVSIM, navhard, Bench2Drive, HUGSIM or nuScenes.

### 6. What is new relative to the wiki

| | RAD | [[sources/think2drive.md]] / [[sources/raw2drive.md]] | [[sources/learning-to-drive-from-a-world-model.md]] | [[sources/simscale.md]] |
|---|---|---|---|---|
| Environment | 3DGS of real logs | CARLA (Raw2Drive: learned latent models of it) | Reprojection or a learned world model of real logs | 3DGS of real logs |
| Other agents | Log replay | CARLA traffic (reactive) | Static, future-anchored | IDM (reactive) |
| States from | The current policy | The current policy | The current policy | A fixed perturbation sampler |
| Signal | PPO on collision / deviation penalties + directional auxiliary losses + imitation | Actor-critic on a shaped reward, in imagination | Future-anchored action labels | Pseudo-expert labels or benchmark sub-scores |
| Input at test | Multi-view cameras | Privileged BEV / cameras | Two cameras | Cameras |
| Public benchmark | None | Bench2Drive | MetaDrive tests, road | NAVSIM |

RAD is the only one here that does policy-gradient RL directly in rendered real-world images.

---

## Relationships

- **[[sources/senna2.md]]**: the same institutions' dual-system VLA, with a 3DGS closed-loop RL stage; its benchmark scores RAD at 0.281 CR (§5).
- **[[sources/simscale.md]]**: positions itself against "3DGS-based online RL (e.g. RAD)"; stores off-log states rather than collecting them from the policy.
- **[[sources/think2drive.md]]**, **[[sources/raw2drive.md]]**: RL in CARLA, privileged and camera; Raw2Drive is the camera-input counterpart in a game engine.
- **[[sources/learning-to-drive-from-a-world-model.md]]**: on-policy training in a data-driven simulator without a reward.
- **[[sources/dreameraD.md]]**: RL with a learned reward in a latent world model instead of a renderer.
- **[[sources/diffusiondrive.md]]**, **[[sources/foredrive.md]]**: the same HUST group's imitation planners on NAVSIM.

---

## Limitations

**Evidence**

1. **Private data and a private benchmark** (2,000 hours; 337 clips); no public-benchmark result.
2. **A second 3DGS benchmark from the same institutions gives RAD three times the collision ratio** and halves its advantage over VADv2.
3. **The imitation baselines were never trained on rendered images**, and RAD was; renderer familiarity is not separated from RL.
4. **Training reward and evaluation metric coincide** (collisions and expert-deviation thresholds).
5. **No seeds; ablation differences of a few clips** out of 337.
6. **One sim-to-real example**; no real-world closed-loop statistics.

**Method**

7. **Log-replayed agents**: interactions are learned against traffic that does not react, and one auxiliary loss rewards accelerating away from a rear-end collision.
8. **The auxiliary losses carry the result** and are hand-designed per event type; reward magnitudes and loss weights are not given.
9. **Episodes end at every event** because rendering degrades, so recovery after a near-miss is not trained.
10. **Discrete 0.5 s actions** over a 61 × 61 grid; longer-horizon planning is left to the value function (γ = 0.9).
11. **Perception is frozen during RL**; only the image encoder and planning head adapt.
12. **No latency figure.**

**Source conversion**

13. All seven figures are embedded (Figure 7 described from its caption). All six tables are reproduced. The text's reference to Table 3's ID 1 should read ID 7.

---

## Key Cross-References

- [[concepts/rl-for-ad.md]] — PPO in 3DGS, the decoupled action and advantage, and why the auxiliary losses carry the gain.
- [[concepts/data-driven-simulators.md]] — reconstruction as a training environment with log-replayed agents; renderer failure handled by episode termination.
- [[concepts/evaluation-variance.md]] — clip-count noise and two disagreeing private 3DGS benchmarks.
- [[concepts/action-tokenization.md]] — a decoupled 61 × 61 lateral / longitudinal displacement vocabulary.
- [[concepts/dual-system-vla.md]] — RAD as the RL-only baseline in Senna-2's comparison.

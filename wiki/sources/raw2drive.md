---
title: "Raw2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving (in CARLA v2)"
type: source-summary
sources: ["raw/papers/Raw2Drive_ Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving(in CARLA v2).md"]
related: [sources/think2drive.md, concepts/bench2drive.md, concepts/rl-for-ad.md, concepts/world-model-for-ad.md, concepts/teacher-pseudo-labels.md, concepts/pdm-lite.md, concepts/inference-latency.md, concepts/evaluation-variance.md, concepts/perception-for-planning.md, sources/dreameraD.md, sources/dynvla.md, sources/orion.md, sources/linkvla.md, sources/deepsight.md, sources/automot.md, research-directions.md]
created: 2026-10-06
updated: 2026-10-06
confidence: medium
---

# Raw2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving (in CARLA v2)

**Paper**: Raw2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving (in CARLA v2)
**Authors**: Zhenjie Yang, Xiaosong Jia, Qifeng Li, Xue Yang, Maoqing Yao, Junchi Yan
**Orgs**: Shanghai Jiao Tong University (Schools of CS, AIS and AI); Institute of Trustworthy Embodied AI, Fudan University; AgiBot. The same group as [[sources/think2drive.md]] (Jia, Li and Yan are on both).
**arXiv**: 2505.16394v2 (May 2025)
**Code**: none stated
**Source**: `raw/papers/Raw2Drive_ Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving(in CARLA v2).md`

---

## What It Is

The camera-input sequel to [[sources/think2drive.md]]. Think2Drive learned to drive in CARLA v2 by model-based RL, but only from a ground-truth bird's-eye-view (BEV) rasterization. Raw2Drive keeps that privileged system as scaffolding and uses it to train a policy that sees only six cameras and an IMU at test time.

**Two streams, two stages** (Figure 2):

| Stage | Privileged stream (training only) | Raw-sensor stream (deployed) |
|---|---|---|
| I | A DreamerV3 world model on BEV masks (43 channels) and an actor-critic policy, trained as in Think2Drive (reusable: "Think2Drive is reused") | – |
| II | Frozen. Runs alongside, fed the same actions and the same sampled stochastic state | A BEVFormer camera encoder and a BEV decoder are trained so the camera latents match the privileged ones; the RSSM dynamics are shared with the privileged model. The policy starts from the privileged policy and is fine-tuned by actor-critic RL in imagined rollouts, with rewards and continuation flags from the **privileged** world model's heads |

**Results** (Bench2Drive, cameras only):

| | DS | SR | Multi-ability mean |
|---|---:|---:|---:|
| **Raw2Drive** | **71.36** | **50.24** | **53.34** |
| Best imitation baseline in its tables | 64.22 (DriveAdapter) | 35.01 (DriveTransformer) | 42.08 (DriveAdapter) |
| Think2Drive (privileged expert) | 91.85 | 85.41 | 86.26 |
| Best camera methods elsewhere in the wiki | 91.01 ([[sources/linkvla.md]]) | 74.55 | 73.40 |

Total cost **64 H800 GPU-days** (40 if Think2Drive is reused for Stage I).

---

## Key Takeaways

- **The wiki's only camera-input RL result on Bench2Drive, and it trails the best imitation methods by up to 20 DS.** 71.36 DS beats every baseline in its own table (imitation methods from 2022–2025, best 64.22) and is the paper's "state of the art". It omits ORION (77.74, two months earlier), SimLingo (85.07) and TF++ (84.21); later methods reach 91.01 (LinkVLA).
- **Most of what Stage II learns is a camera encoder into the privileged latent space.** Figure 4 marks the raw-sensor RSSM as frozen and "shared"; only the sensor encoder and BEV decoder are trained, against the privileged encoder states and latent states. Figure 5 marks the encoder and RSSM frozen during policy training; only the actor and critic move. Rewards and terminations come from the privileged heads, and the policy starts from the privileged policy and is also distilled from it.
- **Predicting reward from cameras breaks training.** Adding a reward head to the camera world model drops the Dev10 score from 83.5 to 46.6, and a continuation head as well to 34.5. The paper's explanation, shown in Figure 9: nearly identical frames can carry opposite rewards (−100 for a collision, +10 for a clean merge at the same steering). The reward is taken from the privileged model instead.
- **Every alignment term is load-bearing, and their removal is a cliff.** Without the encoder, deterministic-state and stochastic-state alignment, Dev10 DS is 0.0; with any one or two of them, 9–38; with all, 83.5. Not sharing the RSSM gives 53.2; using the privileged policy without fine-tuning, 58.4.
- **RL fine-tuning in the aligned world model is worth +25 Dev10 DS** over dropping the privileged policy onto camera latents (58.4 → 83.5). It is the cleanest number in the paper for what the RL stage adds.
- **The expert-student gap stays large.** Trained with the expert's own world model, reward heads and policy, Raw2Drive is 20.5 DS below Think2Drive on Bench2Drive, a larger gap than imitation students trained on the same expert's driving ([[sources/deepsight.md]]: 5.6).
- **It supplies the Think2Drive rows the wiki already cites.** Think2Drive's Bench2Drive scores (91.85 / 85.41) and its CARLA v2 validation and devtest scores (43.8 / 56.8 DS) are in this paper's tables, which identifies the "official test routes" of [[sources/think2drive.md]] as the devtest route.

---

## Method

### Notation (Table 2)

| | Privileged WM | Raw-sensor $\hat{\text{WM}}$ |
|---|---|---|
| Observation | $o_t$ (BEV masks + ego vector) | $\hat o_t$ (multi-view images + IMU) |
| Encoder state | $e_t = \text{Enc}(o_t)$ | $\hat e_t = \hat{\text{Enc}}(\hat o_t)$ (BEVFormer, grid-shaped BEV features) |
| Deterministic state | $h_t = f_\theta(h_{t-1}, a_{t-1}, s_{t-1})$ | ① training: $\hat h_t = \hat f_\theta(\hat h_{t-1}, a_{t-1}, s_{t-1})$; ② inference: $\hat h_t = \hat f_\theta(\hat h_{t-1}, \hat a_{t-1}, \hat s_{t-1})$ |
| Stochastic state | train $s_t = q_\theta(h_t, e_t)$; infer $s_t = p_\theta(h_t)$ | train $\hat s_t = \hat q_\theta(\hat h_t, \hat e_t)$; infer $\hat s_t = \hat p_\theta(\hat h_t)$ |
| Heads | Reward, Decoder, Continue | Decoder only |

### Privileged stream (Stage I)

DreamerV3 as in [[sources/think2drive.md]]: RSSM with decoder, reward and continuation heads; actor-critic trained on imagined rollouts; the world model and policy updated alternately while the policy collects CARLA experience. Reward design is Think2Drive's. Input: 43-channel BEV masks plus an ego vector (Think2Drive used 34 channels).

### Raw-sensor world model (Stage II)

- **Encoder**: BEVFormer over multi-view images and IMU, producing grid-shaped BEV features so they can be compared cell by cell with the privileged encoder state.
- **Heads**: a BEV-mask decoder only. No video reconstruction, and no reward or continuation head (see the ablation).
- **RSSM**: "initialized from the privileged world model"; Figure 4 draws it frozen and shared, and Table 10 shows sharing matters.

### Guidance mechanism

**Rollout guidance** aligns the two streams at every step:

$$\mathcal L_\text{Rollout} = \beta_e\sum_t\sum_{i=0}^{\text{grid num}}\text{MSE}(e^i_t, \hat e^i_t) + \sum_t\big(\beta_s\,\text{KL}(s_t, \hat s_t) + \beta_h\,\text{MSE}(h_t, \hat h_t)\big)$$

with $\beta_e = 10$, $\beta_h = 5$, $\beta_s = 10$. The paper reads the deterministic state as carrying ego intention and the stochastic state as carrying other agents' behaviour (Figure 6b). **Sampling is shared**: the stochastic state is sampled once, from the raw stream's distribution, and fed to both models, so the two rollouts do not drift apart through independent sampling noise.

**Head guidance**: during policy training both models execute the raw policy's action; the privileged world model's reward and continuation heads label each imagined step. The privileged policy is also used "to collect to replay buffer and to distill action distributions to raw sensor policy". Algorithm 1 instead has Stage II collect with the raw-sensor policy; the paper does not reconcile the two.

### Policy (Stage II)

The privileged actor-critic is fine-tuned on raw-stream rollouts (Algorithm 1; Appendix E gives DreamerV3's λ-return critic and normalized actor loss). The action space is 39 discrete throttle/brake/steer/reverse combinations (Table 12).

### Training and benchmarks

- CARLA 0.9.15.1. RL training on 1,000 Bench2Drive training routes under varied weather.
- World models: AdamW, learning rate 1e-5; policy: AdamW, 3e-5.
- Evaluation: Bench2Drive's 220 short routes (one corner case each) and CARLA Leaderboard 2.0's Town13 validation and devtest long routes (7–10 km).
- Ablations on **Dev10**, 10 Bench2Drive clips "carefully selected ... challenging and representative, with low variance", 3 repeats each, averaged.

---

## Figures

![[paradigm.png|Three-panel comparison of training paradigms. (a) Imitation learning or model-free RL from raw sensor input, marked as failing on causal confusion, distribution shift, efficiency and convergence. (b) Model-based RL from raw sensor input, where noise, occlusion and redundancy make the world model hard to learn, marked as failing to converge. (c) The dual-stream approach: a low-dimensional structured privileged BEV input trains one world model and policy, and a guidance mechanism passes it down to a second world model and policy on raw sensor input, marked as converging]]

*Figure 1: (a) Imitation learning and model-free RL; (b) model-based RL on raw input, which the paper says no one has made work; (c) Raw2Drive's dual-stream model-based RL.*

![[method.png|Pipeline diagram. (a) Training: privileged BEV input feeds a world model and policy trained by interaction and reinforcement; a guidance mechanism connects it to a second world model and policy fed by multi-view video. (b) Inference: only multi-view video, the raw-sensor world model and the policy. (c) The guidance mechanism: rollout guidance aligns deterministic (ego intention) and stochastic (surrounding-agent intention) states between rollouts in the two world models; head guidance passes the privileged world model's reward and continuation heads to the raw-sensor policy]]

*Figure 2: Training (a), inference (b) and the guidance mechanism (c).*

![[p_wm_learning_4.png|Diagram of Stage I. Top: the privileged world model, with a privileged encoder, RSSM and continuation, decoder and reward heads at three time steps, all marked as trained. Bottom: the privileged policy, an actor and critic marked as trained, rolling out in the frozen RSSM and receiving reward and continuation from the world model's own heads]]

*Figure 3: Stage I. The privileged world model learns from BEV input; the privileged policy learns from rollouts in it, labelled by its own heads.*

![[r_wm_learning_4.png|Diagram of Stage II world-model training. Top: the privileged world model with frozen encoder and RSSM at three steps. Bottom: the raw-sensor world model, in which the sensor encoder and decoder are marked as trained and the RSSM is marked frozen and shared with the privileged one. Alignment blocks between the streams compare intention latent states and BEV encoder states at each step]]

*Figure 4: Stage II world-model training. The privileged world model's spatial-temporal features supervise the raw-sensor one; per the figure, only the sensor encoder and the decoder are trained.*

![[r_bp_learning_3.png|Diagram of Stage II policy training. Top: the frozen privileged world model, whose continuation and reward heads label each step (head guidance) while its state follows the raw stream's by pseudo-deduction. Bottom: the raw-sensor policy, with frozen sensor encoder and RSSM and trained actor and critic, rolling out in the raw-sensor world model by rigorous deduction]]

*Figure 5: Stage II policy training. The raw stream rolls out ("rigorous deduction"); the privileged stream follows it ("pseudo-deduction") and supplies reward and continuation.*

![[rg_sample.png|Three panels on rollout guidance. (a) Encoder state: objects in multi-view video at times t and t+1 matched to the same objects in privileged BEV input. (b) Stochastic and deterministic state: dynamic intentions of other vehicles such as speed up, slow down, turn left or right, and static or ego intentions such as an ego right turn and map information. (c) Sampling: the deterministic-state equations of both world models side by side, with the stochastic state sampled once from the raw stream's distribution and fed to both, eliminating randomness between them]]

*Figure 6: (a) Spatial-temporal alignment of encoder states; (b) what the paper takes the stochastic and deterministic states to encode; (c) shared sampling of the stochastic state.*

![[loss_guidance.png|Line plot of raw-sensor world-model loss, scaled by 100, over about 100,000 training iterations. With rollout guidance the loss falls from about 0.38 to about 0.1 within 20,000 iterations and stays there. Without it the loss rises from about 0.4 and fluctuates between about 0.45 and 1.2 for the whole run]]

*Figure 7: Raw-sensor world-model loss with and without rollout guidance.*

![[gaussion_vis.png|Two cases, each with a ground-truth BEV mask and the raw-sensor world model's decoded BEV. The decoded masks reproduce roads, lane lines and nearby vehicles, but red circles mark regions where distant or occluded vehicles and road segments are missing or smeared]]

*Figure 8: Ground-truth BEV against the raw-sensor decoder's output. Red circles: camera blind spots and occluded regions the decoder cannot recover.*

![[reward_cont_2.png|Four driving frames. Left pair: two nearly identical views of the ego merging past a police car at steer 0.7, one labelled collision with reward minus 100 and the other a successful merge with reward plus 10. Right pair: two nearly identical views of an intersection, one at 99 percent task progress with speed 0.3 and continue flag 0, the other at 100 percent with speed 0.0 and continue flag 1]]

*Figure 9: Why reward and continuation heads confuse the camera world model. Similar frames, opposite labels. The continuation flags as drawn (0 while driving at 99%, 1 at completion) are the inverse of DreamerV3's convention, in which the flag is 1 while the episode continues.*

---

## Tables

### Table 1: Settings of mainstream methods

| Method | Venue | Input | Scheme | E2E | World model | Benchmark | Corner cases |
|---|---|---|---|:-:|:-:|---|:-:|
| Chen et al. | ITSC 2019 | Privileged | RL | No | No | Roundabout | No |
| MaRLn | CVPR 2020 | Raw | RL | Yes | No | CoRL 2017 | No |
| Roach | ICCV 2021 | Privileged | RL | No | No | CARLA v1 | No |
| UniAD | CVPR 2023 | Raw | IL | Yes | No | CARLA v1 | No |
| Think2Drive | ECCV 2024 | **Raw** (as printed) | RL | No | Yes | CARLA v2 | Yes |
| LAW | ICLR 2025 | Raw | IL + SSL | Yes | Yes | CARLA v1 | No |
| AdaWM | ICLR 2025 | Privileged | RL | No | Yes | ROM03 | No |
| DriveTransformer | ICLR 2025 | Raw | IL | Yes | No | CARLA v2 | Yes |
| **Raw2Drive** | – | Raw | RL | Yes | Yes | CARLA v2 | Yes |

Think2Drive's input is privileged BEV, not raw; the row's "E2E: No" agrees with that, the "Input" cell does not.

### Table 3: CARLA Leaderboard 2.0, Town13 validation and devtest (long routes)

\* = expert feature distillation.

| Method | Scheme | Modality | DS val. | DS devtest | RC val. | RC devtest | IS val. | IS devtest |
|---|---|---|---:|---:|---:|---:|---:|---:|
| AD-MLP | IL | State | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| UniAD-Base | IL | Image | 0.15 | 0.00 | 0.51 | 0.07 | 0.23 | 0.04 |
| VAD | IL | Image | 0.17 | 0.00 | 0.49 | 0.06 | 0.31 | 0.04 |
| DriveTransformer | IL | Image | 0.85 | 0.68 | 1.42 | 2.13 | 0.33 | 0.35 |
| TCP-traj\* | IL | Image | 0.31 | 0.02 | 0.89 | 0.11 | 0.24 | 0.05 |
| ThinkTwice\* | IL | Image | 0.50 | 0.64 | 1.23 | 1.78 | 0.35 | 0.43 |
| DriveAdapter\* | IL | Image | 0.92 | 0.87 | 1.52 | 2.43 | 0.42 | 0.37 |
| **Raw2Drive** | RL | Image | **4.12** | **3.56** | **9.32** | **6.04** | **0.43** | 0.42 |
| Think2Drive | RL expert | Privileged | 43.8 | 56.8 | 78.2 | 98.6 | 0.73 | 0.92 |

Every camera method completes under 10% of a long route. The paper's explanation: the first scenario on both routes is ParkingExit, which needs large steering angles that regression-trained imitation handles badly, and a blocking collision ends the evaluation.

### Table 4: Bench2Drive multi-ability (%)

| Method | Scheme | Modality | Merging | Overtaking | Emergency brake | Give way | Traffic sign | Mean |
|---|---|---|---:|---:|---:|---:|---:|---:|
| TCP-traj\* | IL | Image | 8.89 | 24.29 | 51.67 | 40.00 | 46.28 | 34.22 |
| AD-MLP | IL | State | 0.00 | 0.00 | 0.00 | 0.00 | 4.35 | 0.87 |
| UniAD-Base | IL | Image | 14.10 | 17.78 | 21.67 | 10.00 | 14.21 | 15.55 |
| ThinkTwice\* | IL | Image | 27.38 | 18.42 | 35.82 | 50.00 | 54.23 | 37.17 |
| VAD | IL | Image | 8.11 | 24.44 | 18.64 | 20.00 | 19.15 | 18.07 |
| DriveAdapter\* | IL | Image | 28.82 | 26.38 | 48.76 | 50.00 | 56.43 | 42.08 |
| DriveTransformer | IL | Image | 17.57 | 35.00 | 48.36 | 40.00 | 52.10 | 38.60 |
| **Raw2Drive** | RL | Image | **43.35** | **51.11** | **60.00** | **50.00** | **62.26** | **53.34** |
| Think2Drive | RL expert | Privileged | 81.27 | 83.92 | 90.24 | 90.00 | 87.67 | 86.26 |

### Table 5: Bench2Drive closed loop

| Method | Scheme | Modality | DS | SR (%) | Efficiency | Comfort |
|---|---|---|---:|---:|---:|---:|
| TCP-traj\* | IL | Image | 59.90 | 30.00 | 76.54 | 18.08 |
| AD-MLP | IL | State | 18.05 | 0.00 | 48.45 | 22.63 |
| UniAD-Base | IL | Image | 45.81 | 16.36 | 129.21 | 43.58 |
| VAD | IL | Image | 42.35 | 15.00 | 157.94 | 46.01 |
| ThinkTwice\* | IL | Image | 62.44 | 31.23 | 69.33 | 16.22 |
| DriveAdapter\* | IL | Image | 64.22 | 33.08 | 70.22 | 16.01 |
| GenAD | IL | Image | 44.81 | 15.90 | – | – |
| DriveTransformer | IL | Image | 63.46 | 35.01 | 100.64 | 20.78 |
| MomAD | IL | Image | 44.54 | 16.71 | 170.21 | 48.63 |
| **Raw2Drive** | RL | Image | **71.36** | **50.24** | 214.17 | 22.42 |
| Think2Drive | RL expert | Privileged | 91.85 | 85.41 | 269.14 | 25.97 |

The MomAD row (44.54 / 16.71) is the one [[concepts/bench2drive.md#momworld]] records as SparseDrive's in MomWorld's table.

### Ablations on Dev10 (DS; SR as successes out of 10, averaged over 3 repeats)

**Table 6: heads of the raw-sensor world model**

| Decoder | Reward | Continue | DS | SR |
|:-:|:-:|:-:|---:|---:|
| | | | 17.4 | 1.2/10 |
| ✓ | | | **83.5** | **7.5/10** |
| ✓ | ✓ | | 46.6 | 3.4/10 |
| ✓ | ✓ | ✓ | 34.5 | 2.2/10 |

**Table 7: abstract-state alignment in rollout guidance**

| Encoder | Deterministic | Stochastic | DS | SR |
|:-:|:-:|:-:|---:|---:|
| | | | 0.0 | 0.0/10 |
| ✓ | | | 36.4 | 2.4/10 |
| ✓ | ✓ | | 38.3 | 2.8/10 |
| ✓ | ✓ | ✓ | **83.5** | **7.5/10** |

**Table 8: spatial-temporal alignment**

| Spatial | Temporal | DS | SR |
|:-:|:-:|---:|---:|
| | | 0.0 | 0.0/10 |
| ✓ | | 13.6 | 1.2/10 |
| | ✓ | 9.24 | 0.8/10 |
| ✓ | ✓ | **83.5** | **7.5/10** |

**Table 9: head guidance** (D / R / C = decoder / reward / continuation heads on the raw world model; HG = head guidance)

| Method | D | R | C | HG | DS | SR |
|---|:-:|:-:|:-:|:-:|---:|---:|
| Raw2Drive | ✓ | | | ✓ | **83.5** | **7.5/10** |
| Setting I | ✓ | ✓ | ✓ | ✓ | 34.5 | 2.2/10 |
| Setting II | ✓ | ✓ | ✓ | | 26.4 | 1.6/10 |

**Table 10: sharing parameters with the privileged world model**

| Method | DS | SR |
|---|---:|---:|
| Raw2Drive | **83.5** | **7.5/10** |
| Without shared RSSM | 53.2 | 5.4/10 |
| Without shared decoder head | 65.6 | 6.1/10 |

**Table 11: raw-sensor policy training**

| Method | DS | SR |
|---|---:|---:|
| Privileged policy used directly on raw-sensor latents | 58.4 | 5.6/10 |
| Privileged policy fine-tuned by RL in the raw-sensor world model | **83.5** | **7.5/10** |

The text refers to the heads ablation as "Table 7" and the sharing ablation as "Table 11"; they are Tables 6 and 10.

### Table 12: The 39 discrete actions (throttle, brake, steer, reverse)

| Group | Throttle | Brake | Steer values | Reverse | Count |
|---|---:|---:|---|:-:|---:|
| First entry as printed | 0 | 0 | 1 | No | 1 |
| Strong throttle | 0.7 | 0 | −0.5, −0.3, −0.2, −0.1, 0, 0.1, 0.2, 0.3, 0.5 | No | 9 |
| Light throttle | 0.3 | 0 | −0.7, −0.5, −0.3, −0.2, −0.1, 0, 0.1, 0.2, 0.3, 0.5, 0.7 | No | 11 |
| Coasting | 0 | 0 | −1, −0.6, −0.3, −0.1, 0.1, 0.3, 0.6, 1.0 | No | 8 |
| Full throttle | 1 | 0 | 0 | No | 1 |
| Reverse | 0.5 | 0 | −0.5, −0.3, −0.2, −0.1, 0, 0.1, 0.2, 0.3, 0.5 | Yes | 9 |

Regrouped from the paper's three-column layout (39 entries). **As printed there is no braking action, and (0, 0, 1) appears twice** (the first entry and the coasting group's last). Think2Drive's table starts with full brake (0, 1, 0), so the first entry is probably that action misprinted. Compared with Think2Drive's 30 actions, this set drops straight coasting (0, 0, 0) and adds full throttle and nine reverse actions.

### Table 13: Latency

| Stream | Modality | Encoder | World model | Policy |
|---|---|---:|---:|---:|
| Privileged | BEV state | 2 ms (5 conv layers) | 2 ms | 2 ms |
| Raw sensor | Multi-view images | **600 ms (BEVFormer)** | 2 ms | 2 ms |

Hardware not stated.

### Results given only in the text

| Quantity | Value |
|---|---|
| Training cost | 64 H800 GPU-days; 40 if Think2Drive is reused for Stage I |
| UniAD's training cost, for comparison | about 30 GPU-days |
| RL training routes | 1,000 Bench2Drive routes, varied weather |
| Privileged BEV channels | 43 |
| Rollout-guidance weights | $\beta_e = 10$, $\beta_h = 5$, $\beta_s = 10$ |
| Learning rates | World models 1e-5; policy 3e-5 (AdamW, no weight decay) |

---

## Reading the Results

### 1. What the raw-sensor stream actually learns {#what-is-learned}

| Component | Stage II status (Figures 4 and 5, Tables 10–11) |
|---|---|
| Camera encoder (BEVFormer) | Trained to match the privileged encoder state cell by cell, then frozen for policy training |
| RSSM dynamics | Shared with the privileged world model, drawn frozen; not sharing costs 30 Dev10 DS |
| BEV decoder | Trained; sharing it with the privileged decoder is +18 Dev10 DS |
| Reward and continuation | From the privileged heads, never learned from cameras |
| Policy | Initialized from the privileged policy, distilled from it, then fine-tuned by RL in imagination |

- **The "raw-sensor world model" is the privileged world model with a new front end.** Rollouts run through the privileged dynamics; what is learned from cameras is the mapping into its latent space. That is close to [[concepts/perception-for-planning.md]]'s question of how well perception can feed a fixed planner, with an RL fine-tuning step on top.
- **Imagined rollouts of the camera policy are labelled by a model that sees ground truth.** The privileged model follows the raw stream's sampled states, so its labels describe the camera model's imagined world, not the true one.
- **The RL fine-tuning has a clean price**: 58.4 → 83.5 Dev10 DS (Table 11). The RL stage here is mainly adapting a privileged policy to imperfect camera latents.

### 2. Where 71.36 sits {#position}

| Method | Training signal | Bench2Drive DS |
|---|---|---:|
| [[sources/linkvla.md]] | Imitation | 91.01 |
| [[sources/dynvla.md]] | Imitation | 88.34 |
| [[sources/deepsight.md]] | Imitation, Think2Drive data | 86.23 |
| SimLingo (from [[concepts/bench2drive.md]]) | Imitation, PDM-Lite data | 85.07 |
| [[sources/orion.md]] | Imitation | 77.74 |
| **Raw2Drive** | **RL in aligned world models, privileged scaffolding** | **71.36** |
| DriveAdapter (best in its own table) | Imitation + expert feature distillation | 64.22 |

- **"Surpasses IL methods by a large margin" holds against its table**, whose most recent entries are DriveTransformer and MomAD. ORION (arXiv March 2025, two months earlier) and SimLingo are absent.
- **Efficiency 214.17 is high**, above every imitation method in its own table and below SimLingo (259.23) and LinkVLA (255.84) on [[concepts/bench2drive.md]].
- **Multi-ability**: Merging 43.35 is above ORION (25.00) and below LinkVLA (60.00); the mean 53.34 is level with ORION (54.72).

### 3. The expert gap and what the teacher's abilities show {#expert-gap}

- **Think2Drive's own multi-ability profile is in this paper**: Merging 81.27 (its lowest), Overtaking 83.92, Emergency brake 90.24, **Give way 90.00**, Traffic sign 87.67.
- **Give way is not a teacher weakness.** Every student tabulated in the wiki scores 0–50 on give way, against 90 for the expert whose data most of them use. That weakness is not inherited; merging, the teacher's lowest, is weak for the students too.
- **More coupling to the expert did not close the gap.** Raw2Drive shares the expert's world model, reward heads and policy, and lands 20.5 DS below it; imitation students of the same expert come within 5.6. See [[concepts/bench2drive.md#expert]].

### 4. The long-route table {#long-routes}

- Think2Drive's "official test routes" in its own paper (56.8 DS, 98.6 RC) are the **devtest** column here; its validation-route numbers (43.8 DS, 78.2 RC) appear only in this paper.
- All camera methods score under 4.2 DS and 10% route completion. The paper argues, as Think2Drive did, that long-route DS is not informative, and reports Bench2Drive instead.

### 5. The ablations {#ablations}

- **They run on 10 clips chosen for being challenging, representative and low-variance**, averaged over 3 repeats. The full model scores 83.5 there and 71.36 on all 220 routes, so Dev10 is easier than the full set despite the selection criterion.
- **The effects are cliffs.** Removing any single alignment term drops DS from 83.5 to 9–38; removing all, to 0. Reading these as component values is unsafe: the system is either working or not, and partial configurations fail differently.
- **Success rates are fractions of 10 routes** (e.g. 7.5/10 = mean of three runs). A difference of one route is 10 points of SR.

---

## Relationships

- **[[sources/think2drive.md]]**: Stage I is Think2Drive, and the paper says it can be reused directly. Same group, same reward, same DreamerV3 recipe.
- **[[sources/dreameraD.md]]**: also RL with a learned world model and a learned reward from camera-derived latents, on NAVSIM. DreamerAD learns its reward on video-model latents and reports it working; Raw2Drive reports that a reward head on its camera latents destabilizes training (83.5 → 46.6) and takes the reward from ground truth instead. The two differ in reward type (dense multi-horizon sub-scores vs. a sparse shaped scalar) and in what the latent was trained for.
- **[[sources/orion.md]]**, **[[sources/linkvla.md]]**, **[[sources/dynvla.md]]**, **[[sources/deepsight.md]]**, **[[sources/automot.md]]**: imitation-trained Bench2Drive methods above Raw2Drive; DynVLA's table carries the Raw2Drive row (71.36 / 50.24 / 53.34) and the Think2Drive and PDM-Lite rows.
- DriveAdapter, ThinkTwice and TCP (not ingested) are this group's earlier imitation students of Think2Drive-style experts, and Raw2Drive's main baselines.

---

## Limitations

**Evidence**

1. **Its comparison omits ORION** (77.74, on arXiv two months earlier) and SimLingo (85.07); later imitation methods reach 91.01 (LinkVLA), 19.6 above it.
2. **Ablations use a 10-clip subset** with selection criteria that favour low variance; every ablation is a cliff, and SR is measured in tenths of a route.
3. **No seed variance on Bench2Drive**; single numbers for the headline.
4. **The RL benefit over imitation is not isolated.** No imitation-only student trained with the same encoder, data and privileged latents is reported; Table 11's 58.4 → 83.5 compares two uses of the privileged policy.

**Method**

5. **Privileged scaffolding at every level of training**: ground-truth BEV world model, reward and continuation from ground truth, policy initialized from and distilled from the privileged policy. The paper notes that in real use the privileged input would come from human annotation or offline perception.
6. **CARLA only.** The paper names 3DGS or diffusion simulators as the route to real-world RL, untested.
7. **600 ms per frame for the BEVFormer encoder**, hardware unstated; the world model and policy cost 4 ms together.
8. **Discrete actions with no braking action as printed** (Table 12).

**Text and tables**

9. Table 1 lists Think2Drive's input as "Raw".
10. Cross-references to Tables 7 and 11 in the ablation text point to Tables 6 and 10.
11. Figure 9's continuation flags are inverted relative to DreamerV3's convention.
12. The replay buffer in Stage II is filled by the privileged policy (text) or by the raw-sensor policy (Algorithm 1).
13. The text says the RSSM is "initialized from" the privileged one; the figures draw it frozen and shared.

**Source conversion**

14. All nine figures are embedded. All thirteen tables are reproduced (Table 2 as the notation table, Table 12 regrouped). The appendix reference list is cut in the clipping after reference 18.

---

## Key Cross-References

- [[concepts/bench2drive.md]] — the only camera RL entry; the expert's multi-ability profile; how far students sit from the expert.
- [[concepts/world-model-for-ad.md]] — two aligned world models, with the camera model reusing the privileged dynamics (Pattern 47).
- [[concepts/rl-for-ad.md]] — model-based RL from cameras, bootstrapped from a privileged system.
- [[concepts/teacher-pseudo-labels.md]] — a privileged teacher used as latent target, reward source, policy initialization and action distiller.
- [[concepts/inference-latency.md]] — a 600 ms camera encoder in front of a 4 ms world model and policy.
- [[concepts/evaluation-variance.md]] — a 10-clip ablation subset with cliff-like effects.

---
title: "MomWorld: Momentum-Aware Latent World Model for Long-Horizon Autonomous Driving"
type: source-summary
sources: ["raw/papers/MomWorld_ Momentum-Aware Latent World Model for Long-Horizon Autonomous Driving.md"]
related: [concepts/world-model-for-ad.md, concepts/selection-based-planning.md, concepts/navsim-benchmark.md, concepts/navhard-ood-evaluation.md, concepts/nuscenes-waymo-evals.md, concepts/bench2drive.md, concepts/diffusion-planner.md, concepts/evaluation-variance.md, concepts/inference-latency.md, sources/drivefuture.md, sources/had.md, sources/da-wam.md, sources/latent-wam.md, sources/drive-jepa.md, sources/drivesuprim.md, sources/hydra-mdp-pp.md, sources/spanvla.md, sources/drivelaw.md, sources/epona.md, sources/policy-world-model.md, sources/geoworldad.md, sources/redrive.md, sources/foresight.md]
created: 2026-09-30
updated: 2026-09-30
confidence: low
---

# MomWorld

**Paper**: MomWorld: Momentum-Aware Latent World Model for Long-Horizon Autonomous Driving
**Authors**: Ziying Song, Shengkai Zhang, Lei Yang, Haozhuang Chi, Yuchen Liu, Jiangtao Su, Lin Liu, Ziyang Liu, Chen Lv
**Orgs**: Nanyang Technological University, Beijing Jiaotong University, North University of China, Dalian University of Technology, Tsinghua University
**arXiv**: 2609.33737v1
**Code**: `github.com/modaxiansheng/MomWorld`
**Source**: `raw/papers/MomWorld_ Momentum-Aware Latent World Model for Long-Horizon Autonomous Driving.md`

> **Read [Internal regularities](#regularities) before using any ablation number from this page.** The three ablation tables contain exact linear relationships between columns that are separate measurements, including across benchmarks that use different base models. The headline results are recorded below as reported; the wiki does not rely on the ablation effect sizes.

---

## What It Is

The sequel to MomAD from the same first author. MomAD used "momentum" (agreement between consecutive plans and perception queries) to stabilize the current ego plan. MomWorld makes momentum a **latent state of the world model**: a pooled scene vector $z$ and a momentum vector $p$ are rolled forward together, $p$ through a gated update and $z$ by integrating $p$.

Two components:

1. **MoLWM** rolls out $(z_k,p_k)$ for $H$ steps, turns them into a **Future World Memory**, and lets a trajectory scorer attend to it when ranking a fixed vocabulary.
2. **MoFlow** refines the selected trajectory with a flow that starts at the *selected plan* (not at noise) and ends at the expert trajectory, followed by a clipped, horizon-weighted residual fusion.

It is **one module on three different base planners**:

| Benchmark | Base model | Evidence |
|---|---|---|
| NAVSIM | **GTRS-Dense** (V2-99, 2 camera frames + 4 LiDAR sweeps, 16,384-trajectory vocabulary) | Stated. The no-component navhard row of the ablation equals GTRS-Dense's published row on four metrics |
| nuScenes | **MomAD** | Inferred: the no-component row of the ablation equals MomAD's published row on five metrics |
| Bench2Drive | Not stated; the numbers sit within about one point of **Hydra-NeXt** on nearly every column | Inferred |

Headline numbers: **nuScenes 6 s average L2 1.17 m and collision 0.79% (MomAD: 1.42 / 0.90); NAVSIM v1 90.2 PDMS; NAVSIM v2 navtest 90.1 EPDMS; navhard 42.8; Bench2Drive 74.07 DS / 50.00 SR.**

---

## Key Takeaways

- **The world model is second-order and action-free.** One 256-d state and one 256-d momentum per future step, with learned retain, reset and update gates. No candidate-specific rollout: all 16,384 candidates read the same future memory.
- **On NAVSIM it is a scorer-cohort method.** The base is GTRS-Dense with PDM-subscore heads. The paper's own ablation puts the baseline at **41.7 navhard** (GTRS-Dense's published score) and the full method at 42.8, so every component together is worth **+1.1**.
- **The +1.1 is a trade, not a uniform gain.** Against GTRS-Dense's published sub-scores, Stage-2 NC falls 5.0 and TTC falls 5.7 while EP rises 12.2. The ablation table prints only the three Stage-2 sub-scores that rise.
- **The navhard table omits the six methods above it, including the first author's own.** The rows match [[sources/drivefuture.md]]'s table, whose corresponding author is MomWorld's first author. DriveFuture with the same GTRS-Dense scorer is 55.5; DrivoR 54.6, SimScale 53.2, GTRS-E 49.4, ZTRS 48.1 and DiffVLA 45.0 are also absent. The paper claims "the best EPDMS of 42.8".
- **The "long-horizon" gains are largest at the shortest horizons.** Against MomAD, L2 falls 34–39% at 1–2 s and 6% at 6 s. Collision rate is unchanged at 4 s (0.83 for both).
- **MoFlow is a plan-to-expert residual flow.** It is deterministic at inference, takes 8.4 ms for four Euler steps, and its output is clipped and scaled by a ramp from 0.05 at the first waypoint to 1.0 at the last.
- **The ablation tables are too regular to use.** Across 27 ablation rows, TPC@6 equals L2@6 minus about 0.855 m with a standard deviation of 0.008 m. In the MoLWM table, L2@6 is an affine function of L2@3 to within 0.01 m for every variant. See [below](#regularities).
- **Bench2Drive: +0.21 DS over Hydra-NeXt with the same success rate.** Robustness tables beat the best prior value in nearly every cell by 0.001 to 0.04 points, against literature values the paper says are not matched reruns.

---

## Method

![[MomWorldv5.png|MomWorld overview: multi-view images encoded into scene queries with map and detection heads; temporal fusion initializes state z0 and momentum p0; scene-adaptive momentum dynamics with maintain, update and reset; latent world rollout to t+H; future world memory; candidate planning with top-score selection of the plan; MoFlow refinement into the final trajectory]]

*Figure 2: Overview. Scene queries → MoLWM (state and momentum rollout) → Future World Memory → candidate scoring → MoFlow refinement.*

### MoLWM: state and momentum

**Initialization** from the current and previous pooled scene queries:

$$(z_0,p_0)=f_{\mathrm{temp}}\big([\operatorname{Pool}(Q_t),\ \operatorname{Pool}(Q_t)-\operatorname{Pool}(Q_{t-1})]\big)$$

If the previous frame is missing or across a scene boundary, $Q_{t-1}$ is replaced by $Q_t$, giving zero variation.

**Scene-Adaptive Momentum Dynamics.** A three-headed network reads the previous state, the previous momentum and a horizon embedding:

$$(\rho_k,g_k,u_k)=f_{\mathrm{gate}}([z_{k-1},p_{k-1},r_k^{\mathrm{time}}]),\qquad \rho_k,g_k\in\sigma(\cdot),\quad u_k\in\tanh(\cdot)$$

$$p_k=\operatorname{LN}\big(\rho_k\odot(1-g_k)\odot p_{k-1}+(1-\rho_k)\odot u_k\big)$$

| Gate | Name | Role |
|---|---|---|
| $\rho_k$ | Maintain | How much inherited momentum persists |
| $g_k$ | Reset | Suppresses stale inherited momentum after an abrupt change |
| $u_k$ | Update | A scene-conditioned momentum proposal |

**Latent World Rollout.** The momentum advances the state:

$$z_k=z_{k-1}+\Delta t\,\pi_p(p_k),\qquad k=1,\dots,H$$

This is a recurrent latent integrator with an explicit velocity-like variable. The rollout does **not** take the ego action as input.

**Training targets** (unavailable at inference). Future images are encoded by the shared image encoder and detached:

$$Q^\star_{t+k}=\operatorname{sg}\big(\operatorname{ImageEncoder}(\mathcal O_{t+k})\big)$$

$$\mathcal L_{\mathrm{future}}=\frac1H\sum_k d_Q\big(f_Q([z_k,p_k]),Q^\star_{t+k}\big),\qquad \mathcal L_p=\frac1H\sum_k\Big\|f_\Delta(p_k)-\big[\operatorname{Pool}(Q^\star_{t+k})-\operatorname{Pool}(Q^\star_{t+k-1})\big]\Big\|_1$$

$$\mathcal L_{\mathrm{aux}}=\frac1H\sum_k\big(\ell^k_{\mathrm{ego}}+\ell^k_{\mathrm{agent}}+\ell^k_{\mathrm{pres}}\big)$$

So momentum is supervised as the **first difference of the pooled target query**, and the auxiliary term uses future ego states, track-aligned agent states and presence labels. The matching loss $d_Q$ is not specified.

### Future World Memory and candidate scoring

$$m_k=\operatorname{LN}\big(z_k+\operatorname{MLP}([p_k,r_k^{\mathrm{time}}])\big),\qquad M_t=\operatorname{SelfAttn}([m_1,\dots,m_H])$$

Each vocabulary trajectory is one query token that cross-attends to $[Q_t;M_t]$ in a three-layer decoder. Heads predict imitation plus NC, DAC, TTC, EP, DDC, LK and TLC, supervised by PDM sub-scores. The inference score is a Hydra-style weighted log-sum:

$$S_i=0.03\log\pi_i^{\mathrm{imi}}+0.10\log p_i^{\mathrm{TLC}}+0.10\log p_i^{\mathrm{NC}}+0.90\log p_i^{\mathrm{DAC}}+0.20\log p_i^{\mathrm{DDC}}+6.0\log\big(7.0\,p_i^{\mathrm{TTC}}+7.0\,p_i^{\mathrm{EP}}+3.0\,p_i^{\mathrm{LK}}\big)$$

The paper is explicit about what the memory is not: "no candidate-specific world rollout or explicit one-to-one temporal matching is performed." Its argument for why a shared memory can still discriminate candidates is that "their distinct query tokens produce different attention weights."

### MoFlow

![[zituv1.drawio.png|MoFlow: momentum-guided residual flow conditioned on p0 and the future memory, trained on the linear path between the encoded base plan and the encoded expert trajectory and integrated by Euler steps at inference; horizon-aware residual fusion subtracts the base plan, clips, gates and adds back]]

*Figure 3: MoFlow. A residual flow from the base plan toward the expert, then clipped horizon-aware fusion.*

**Momentum-Guided Residual Flow.** Waypoints are encoded as $\phi(x,y,\psi)=(x/50,\ y/50,\ \sin\psi,\ \cos\psi)$. With $X_0=\phi(\tau^0)$ the selected plan and $X_1=\phi(\tau^\star)$ the expert:

$$X_\epsilon=(1-\epsilon)X_0+\epsilon X_1,\qquad \mathcal L_{\mathrm{FM}}=\mathbb E_\epsilon\Big[\big\|v_\theta(X_\epsilon,\epsilon\mid p_0,M_t)-(X_1-X_0)\big\|_2^2\Big]$$

$$X^{j+1}=X^j+\frac1J v_\theta\Big(X^j,\frac{j+1/2}{J}\ \Big|\ p_0,M_t\Big),\qquad j=0,\dots,J-1$$

There is no noise anywhere. The source distribution is the scorer's output and the flow is a learned correction.

**Horizon-Aware Residual Fusion.**

$$\Delta\tau_k=\operatorname{clip}_{[-\delta,\delta]}\big(\widehat\tau_k-\tau^0_k\big),\qquad \gamma_k=0.05+0.95\frac{k-1}{H-1},\qquad \widetilde\tau_k=\tau^0_k+\sigma(b)\,\gamma_k\,\Delta\tau_k$$

$\delta=2.0$ per component, $b$ is one learned scalar initialized at −4 (so $\sigma(b)\approx0.018$ at the start). The learned value of $b$ is not reported.

### Objective

$$\mathcal L=\mathcal L_{\mathrm{percep}}+\mathcal L_{\mathrm{plan}}+\underbrace{\mathcal L_{\mathrm{future}}+0.5\,\mathcal L_p+0.2\,\mathcal L_{\mathrm{aux}}}_{\mathcal L_{\mathrm{world}}}+\underbrace{\mathcal L_{\mathrm{FM}}+\mathcal L_\tau}_{\mathcal L_{\mathrm{MoFlow}}},\qquad \mathcal L_\tau=\frac1H\sum_k\|\widetilde\tau_k-\tau^\star_k\|_1$$

Map and detection heads are kept, so this is a perception-supervised model.

---

## Figures

![[motivationv9.png|Motivation: a state-only view of the scene; MomAD deriving ego momentum from history and the current frame; MomWorld deriving world momentum from history, current and predicted future; bar charts of nuScenes 6 s L2 and collision rate for MomAD and MomWorld]]

*Figure 1: (a) state only, (b) MomAD's ego momentum, (c) MomWorld's world momentum, (d) six-second nuScenes results: L2 2.45 → 2.31 m at 6 s, collision 2.13 → 1.97%.*

Figures 2 and 3 are in the Method section above.

![[vis_nus3.png|Consecutive planning frames t−1, t and t+1 on a nuScenes left turn, MomAD in the top row and MomWorld in the bottom row, with ground truth in red and the predictions from the three frames overlaid]]

*Figure 4: Consecutive-frame planning on nuScenes, MomAD against MomWorld.*

![[nus_vis_fl.png|Qualitative six-second planning results on nuScenes across four scenarios]]

*Figure 5: Six-second planning on nuScenes in four scenes (turn right, go straight, turn left, turn left). Each row shows the six camera views with detection boxes and the projected plan, then two bird's-eye panels with the map, agents and the 6 s ego trajectory coloured by time. The paper describes them as turn-to-straight transitions, deceleration in dense traffic, a left turn and vehicle avoidance.*

![[Navsim_vis_fl_v2.png|Five NAVSIM scenes (turn, straight, brake, stable driving, avoid vehicle) in bird's-eye view, GuideFlow in the top row and MomWorld in the bottom row]]

*Figure 6: NAVSIM comparison with GuideFlow across five scenarios.*

---

## Tables

### Table 1: Six-second planning on the nuScenes validation set

FPS hardware: A100 for UniAD, RTX 3090 for LAW, RTX 4090 for SparseDrive, MomAD and GuideFlow. MomWorld's hardware is not stated.

| Method | Venue | L2 1s | 2s | 3s | 4s | 5s | 6s | Avg | Col 1s | 2s | 3s | 4s | 5s | 6s | Avg | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UniAD | CVPR'23 | 0.47 | 0.91 | 1.35 | 1.91 | 2.47 | 3.07 | 1.70 | 0.25 | 0.36 | 0.61 | 0.99 | 1.64 | 2.51 | 1.06 | 1.8 |
| SparseDrive | ICRA'25 | 0.43 | 0.87 | 1.23 | 1.75 | 2.32 | 2.95 | 1.59 | 0.19 | 0.31 | 0.56 | 0.87 | 1.54 | 2.33 | 0.97 | 9.0 |
| MomAD | CVPR'25 | 0.41 | 0.85 | 1.13 | 1.67 | 1.98 | 2.45 | 1.42 | 0.17 | 0.30 | 0.54 | 0.83 | 1.43 | 2.13 | 0.90 | 7.8 |
| LAW | ICLR'25 | 0.40 | 0.87 | 1.16 | 1.71 | 2.03 | 2.61 | 1.46 | 0.19 | 0.33 | 0.57 | 0.86 | 1.51 | 2.31 | 0.96 | 19.5 |
| Epona | ICCV'25 | 0.39 | 0.91 | 1.17 | 1.73 | 2.02 | 2.75 | 1.50 | 0.14 | 0.18 | 0.45 | 0.74 | 1.48 | 2.23 | 0.87 | – |
| World4Drive | ICCV'25 | 0.42 | 0.92 | 1.21 | 1.75 | 2.06 | 2.79 | 1.53 | 0.16 | 0.20 | 0.47 | 0.76 | 1.50 | 2.14 | 0.87 | – |
| DIVER | TPAMI'26 | 0.38 | 0.75 | 1.10 | 1.53 | 1.98 | 2.49 | 1.37 | 0.13 | 0.31 | 0.44 | 0.80 | 1.41 | 2.11 | 0.87 | 6.6 |
| GuideFlow | CVPR'26 | 0.42 | 0.83 | 1.21 | 1.73 | 2.05 | 2.63 | 1.48 | 0.12 | 0.22 | 0.42 | 0.79 | 1.44 | 2.15 | 0.86 | 3.6 |
| **MomWorld** | – | **0.27** | **0.52** | **0.86** | **1.28** | **1.76** | **2.31** | **1.17** | **0.02** | **0.17** | 0.42 | 0.83 | **1.36** | **1.97** | **0.79** | 7.2 |

### Table 2: Trajectory Prediction Consistency (TPC, m) at 4–6 s, nuScenes validation

| Method | Venue | 4s | 5s | 6s | Avg |
|---|---|---:|---:|---:|---:|
| UniAD | CVPR'23 | 1.49 | 1.81 | 2.41 | 1.90 |
| VAD | ICCV'23 | 1.55 | 1.73 | 2.17 | 1.82 |
| SparseDrive | ICRA'25 | 1.33 | 1.66 | 1.99 | 1.66 |
| MomAD | CVPR'25 | 1.19 | 1.45 | 1.61 | 1.42 |
| **MomWorld** | – | **0.93** | **1.19** | **1.46** | **1.19** |

TPC is the mean Euclidean distance between the matching waypoints of two consecutive planning outputs, following MomAD's public evaluator.

### Table 3: NAVSIM v1 navtest

\* marks the authors' re-implementation.

| Method | NC | DAC | TTC | Comf. | EP | PDMS |
|---|---:|---:|---:|---:|---:|---:|
| *E2E-based* | | | | | | |
| VADv2 | 97.2 | 89.1 | 91.6 | 100 | 76.0 | 80.9 |
| TransFuser | 97.7 | 92.8 | 92.8 | 100 | 79.2 | 84.0 |
| TransFuser-mmt\* | 96.2 | 95.4 | 90.7 | 100 | 80.7 | 85.1 |
| UniAD | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| PARA-Drive | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 |
| DRAMA | 98.0 | 93.1 | 94.8 | 100 | 80.1 | 85.5 |
| Hydra-MDP | 98.3 | 96.0 | 94.6 | 100 | 78.7 | 86.5 |
| FUMP | 98.1 | 96.2 | 94.2 | 100 | 82.0 | 87.8 |
| DiffusionDrive | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| DIVER | 98.5 | 96.5 | 94.9 | 100 | 82.6 | 88.3 |
| DriveSuprim | 97.8 | 97.3 | 93.6 | 100 | 86.7 | 89.9 |
| GoalFlow | 98.4 | 98.3 | 94.6 | 100 | 85.0 | 90.3 |
| ReCogDrive-IL | 98.1 | 94.7 | 94.2 | 100 | 80.9 | 86.5 |
| *VLA-based* | | | | | | |
| AutoVLA | 98.4 | 95.6 | 98.0 | 99.9 | 81.9 | 89.1 |
| ReCogDrive | 98.2 | 97.8 | 95.2 | 99.8 | 83.5 | 89.6 |
| DriveWorld-VLA | 99.1 | 98.2 | 96.1 | 100 | 85.9 | 91.3 |
| *World-model-based* | | | | | | |
| DrivingGPT | 98.9 | 90.7 | 94.9 | 95.6 | 79.7 | 82.4 |
| ReSim | – | – | – | – | – | 86.6 |
| PWM | 98.6 | 95.9 | 95.4 | 100 | 81.8 | 88.1 |
| LAW | 96.4 | 95.4 | 88.7 | 99.9 | 81.7 | 84.6 |
| World4Drive | 97.4 | 94.3 | 92.8 | 100 | 79.9 | 85.1 |
| Epona | 97.9 | 95.1 | 93.8 | 99.9 | 80.4 | 86.2 |
| WoTE | 98.5 | 96.8 | 94.9 | 99.9 | 81.9 | 88.3 |
| WorldRFT | 97.8 | 96.8 | 94.0 | 100 | 81.7 | 87.8 |
| DriveLaW | 99.0 | 97.1 | 96.7 | 100 | 81.3 | 89.1 |
| DriveX-S | 97.5 | 94.0 | 93.0 | 100 | 79.7 | 84.5 |
| **MomWorld** | 98.6 | 98.2 | 93.9 | 100 | 85.7 | **90.2** |

### Table 4: NAVSIM v2 navtest

The paper describes this as a diagnostic: the one-stage v2 scorer applied to navtest, "not an official NAVSIM v2 leaderboard protocol".

| Method | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | EPDMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TransFuser | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 |
| DiffusionDrive | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | 84.5 |
| Hydra-MDP++ | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | 81.4 |
| DriveSuprim | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 | 83.1 |
| DiffusionDriveV2 | 97.7 | 96.6 | 99.2 | 99.8 | 88.9 | 97.2 | 96.0 | 97.8 | 91.0 | 85.5 |
| DriveWorld-VLA | 98.6 | 99.1 | 99.6 | 99.8 | 87.4 | 97.9 | 97.0 | 97.8 | 78.6 | 86.8 |
| ReCogDrive ⚠ | 98.3 | 95.2 | 98.3 | 99.8 | 87.1 | 97.5 | 96.6 | 99.5 | 86.5 | 83.6 |
| Latent-WAM | 98.1 | 97.3 | 99.6 | 99.8 | 87.7 | 97.3 | 97.6 | 98.1 | 87.3 | 89.3 |
| **MomWorld** | 98.1 | 98.1 | 99.5 | 99.8 | **88.9** | **98.2** | 96.1 | 98.3 | 90.6 | **90.1** |

⚠ ReCogDrive's DDC and HC are swapped relative to every other table in the wiki (DDC 99.5, HC 98.3).

### Table 5: NAVSIM v2 navhard

† computed by the authors from reported stage metrics.

| Method | Stage | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | EPDMS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TransFuser | 1 | 96.2 | 79.5 | 99.1 | 99.5 | 84.1 | 95.1 | 94.2 | 97.5 | 79.1 | |
| | 2 | 77.7 | 70.2 | 84.2 | 98.0 | 85.1 | 75.6 | 45.4 | 95.7 | 75.9 | 23.1 |
| DiffusionDrive | 1 | 96.0 | 79.7 | 97.4 | 99.5 | 81.3 | 93.1 | 90.8 | 96.8 | 73.8 | |
| | 2 | 82.1 | 72.2 | 88.5 | 98.7 | 85.1 | 78.8 | 49.2 | 89.3 | 71.2 | 24.2 |
| GuideFlow | 1 | 96.6 | 80.5 | 96.3 | 99.3 | 82.3 | 94.9 | 91.5 | 97.7 | 67.8 | |
| | 2 | 87.3 | 76.7 | 88.8 | 99.2 | 84.3 | 85.1 | 49.7 | 93.1 | 44.5 | 27.1 |
| RAP-DINO | 1 | 97.1 | 94.4 | 98.8 | 99.8 | 83.9 | 96.9 | 94.7 | 96.4 | 66.2 | |
| | 2 | 83.2 | 83.9 | 87.4 | 98.0 | 86.9 | 80.4 | 52.3 | 95.2 | 52.4 | 36.9 |
| DriveSuprim | 1 | 98.9 | 95.1 | 99.2 | 99.6 | 76.1 | 99.1 | 94.7 | 97.6 | 54.2 | |
| | 2 | 87.9 | 88.8 | 89.6 | 98.8 | 80.3 | 86.0 | 53.5 | 97.1 | 56.1 | 42.1 |
| DriveFine | 1 | 97.6 | 90.0 | 99.1 | 99.3 | 84.9 | 96.7 | 97.3 | 97.6 | 72.0 | |
| | 2 | 82.1 | 71.3 | 84.8 | 98.4 | 88.1 | 74.3 | 47.2 | 96.8 | 72.8 | 30.5 † |
| SpanVLA | 1 | 98.4 | 94.3 | 97.8 | 99.9 | 85.7 | 97.2 | 94.2 | 97.6 | 72.1 | |
| | 2 | 86.9 | 84.3 | 87.1 | 98.2 | 85.5 | 82.7 | 62.3 | 96.8 | 67.4 | 40.1 |
| MindDrive | 1 | 96.1 | 86.0 | 98.8 | 99.3 | 83.3 | 95.6 | 94.4 | 97.6 | 74.7 | |
| | 2 | 82.6 | 79.1 | 86.4 | 98.0 | 85.3 | 79.4 | 49.2 | 96.5 | 71.0 | 30.9 |
| World4Drive | 1 | 97.3 | 89.1 | 97.6 | 99.7 | 60.5 | 96.8 | 87.7 | 93.1 | 60.0 | |
| | 2 | 91.4 | 82.0 | 91.0 | 98.5 | 53.1 | 90.6 | 52.3 | 93.3 | 62.8 | 34.9 |
| **MomWorld** | 1 | 96.9 | 93.6 | **99.8** | 99.8 | 80.4 | 96.9 | 96.4 | 97.6 | 60.0 | |
| | 2 | 86.4 | 88.2 | **94.2** | 98.3 | 81.7 | 84.4 | 55.3 | 97.0 | 54.7 | **42.8** |

### Table 6: Cumulative component study

nuScenes 6 s model and NAVSIM v2 navhard. LK, EP and EC are Stage-2 values.

| FWM | LWR | SMD | MGRF | HARF | L2@3 | L2@6 | TPC@6 | Col@3 | Col@6 | LK | EP | EC | navhard EPDMS |
|:-:|:-:|:-:|:-:|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| | | | | | 1.13 | 2.45 | 1.61 | 0.54 | 2.13 | 54.6 | 69.5 | 49.7 | 41.7 |
| ✓ | | | | | 1.07 | 2.42 | 1.57 | 0.51 | 2.09 | 54.8 | 72.3 | 50.8 | 42.0 |
| ✓ | ✓ | | | | 1.01 | 2.39 | 1.54 | 0.48 | 2.06 | 55.0 | 75.6 | 52.1 | 42.2 |
| ✓ | ✓ | ✓ | | | 0.96 | 2.36 | 1.51 | 0.46 | 2.03 | 55.1 | 78.4 | 53.0 | 42.4 |
| ✓ | ✓ | ✓ | ✓ | | 0.91 | 2.33 | 1.48 | 0.44 | 2.00 | 55.2 | 80.2 | 54.0 | 42.6 |
| ✓ | ✓ | ✓ | ✓ | ✓ | 0.86 | 2.31 | 1.46 | 0.42 | 1.97 | 55.3 | 81.7 | 54.7 | 42.8 |

The first row is two published rows joined: its nuScenes values are MomAD's (Tables 1 and 2), and its navhard values are GTRS-Dense's (LK 54.6, EP 69.5, EC 49.7, EPDMS 41.7, as reproduced in [[sources/had.md]]'s navhard table).

### Table 7: MoLWM design ablation

| Variant | L2@3 | L2@6 | TPC@6 | Col@3 | Col@6 | v1 PDMS | navhard EPDMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| No historical variation | 0.95 | 2.58 | 1.72 | 0.55 | 2.40 | 88.6 | 39.7 |
| No horizon embedding in SMD | 0.90 | 2.43 | 1.57 | 0.47 | 2.12 | 89.5 | 41.3 |
| Fixed retention ($\rho_k\equiv0.9$) | 0.92 | 2.49 | 1.63 | 0.50 | 2.21 | 89.0 | 40.8 |
| No momentum proposal ($u_k\equiv0$) | 0.99 | 2.69 | 1.82 | 0.60 | 2.58 | 87.8 | 38.6 |
| No reset gate ($g_k\equiv0$) | 0.94 | 2.55 | 1.69 | 0.65 | 2.83 | 88.2 | 37.9 |
| No horizon embedding in FWM | 0.91 | 2.46 | 1.60 | 0.49 | 2.17 | 89.3 | 40.9 |
| No momentum injection in FWM | 0.97 | 2.64 | 1.78 | 0.58 | 2.49 | 88.0 | 38.9 |
| No temporal aggregation in FWM | 0.93 | 2.52 | 1.66 | 0.52 | 2.29 | 88.8 | 40.2 |
| **Full MoLWM** | 0.86 | 2.31 | 1.46 | 0.42 | 1.97 | 90.2 | 42.8 |

### Table 8: MoFlow design ablation

| MoFlow design | Steps | L2@3 | L2@6 | TPC@6 | Col@3 | Col@6 | v1 PDMS | navhard | Flow ms | Total ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No MoFlow (base plan) | 0 | 0.96 | 2.36 | 1.51 | 0.46 | 2.03 | 89.7 | 42.4 | 0.0 | 130.5 |
| Unconditioned MGRF | 4 | 1.01 | 2.53 | 1.67 | 0.57 | 2.38 | 88.3 | 40.0 | 8.4 | 138.9 |
| $M_t$ only | 4 | 0.91 | 2.40 | 1.55 | 0.47 | 2.09 | 89.4 | 41.8 | 8.4 | 138.9 |
| $p_0$ only | 4 | 0.93 | 2.43 | 1.58 | 0.49 | 2.14 | 89.2 | 41.5 | 8.4 | 138.9 |
| $p_0,M_t$ | 1 | 0.91 | 2.38 | 1.53 | 0.46 | 2.07 | 89.8 | 42.3 | 2.2 | 132.7 |
| $p_0,M_t$ | 2 | 0.88 | 2.34 | 1.49 | 0.44 | 2.01 | 90.0 | 42.5 | 4.3 | 134.8 |
| $p_0,M_t$ | 8 | 0.87 | 2.32 | 1.47 | 0.43 | 1.99 | 90.1 | 42.7 | 16.5 | 147.0 |
| Direct replacement ($\widetilde\tau=\widehat\tau$) | 4 | 0.98 | 2.57 | 1.69 | 0.59 | 2.46 | 87.9 | 39.5 | 8.4 | 138.9 |
| No residual clipping | 4 | 0.92 | 2.45 | 1.59 | 0.55 | 2.37 | 88.7 | 40.5 | 8.4 | 138.9 |
| No horizon factor | 4 | 0.90 | 2.40 | 1.55 | 0.48 | 2.12 | 89.4 | 41.7 | 8.4 | 138.9 |
| No learned global gate ($\sigma(b)\to1$) | 4 | 0.89 | 2.38 | 1.53 | 0.47 | 2.09 | 89.6 | 42.0 | 8.4 | 138.9 |
| **Full MoFlow** | 4 | 0.86 | 2.31 | 1.46 | 0.42 | 1.97 | 90.2 | 42.8 | 8.4 | 138.9 |

### Table 9: NAVSIM configuration

| Configuration | Value |
|---|---|
| Input history | 2 camera frames / 4 LiDAR sweeps |
| Image resolution | 2048×512 |
| Image backbone | V-99-eSE VoVNet |
| Planner latent / FFN width | 256 / 1024 |
| Planner attention layers / heads | 3 / 8 |
| Candidate vocabulary | 16,384 |
| Planning horizon / interval | 40 steps / 0.1 s |
| LWR horizon / interval | 40 steps / 0.1 s |
| Maximum agent slots | 30 |
| Momentum persistence initialization | 0.9 |
| Trajectory encoding | $s_{\mathrm{pos}}=50$ |
| Flow solver / steps | explicit Euler, midpoint-time conditioning / 4 |
| Heading normalization | after Euler integration |
| Residual clamp δ | componentwise, ±2.0 |
| Residual-gate initialization $b$ | −4 |
| Horizon weighting | linear ramp, 0.05 → 1.0 |
| Optimizer / learning rate | Adam / 2e-4 |
| Epochs / per-device batch | 20 / 2 |
| Precision / gradient clipping | FP16 mixed / 5.0 |

Vocabulary dropout keeps half the candidates during training; all 16,384 are scored at inference.

### Table 10: Bench2Drive

\* expert feature distillation.

| Method | Venue | DS | SR | Effi. | Comf. | Merge | Overtake | Em. Brake | Give Way | Traffic Sign | Mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TCP-traj\* | NeurIPS'22 | 59.90 | 30.00 | 76.54 | 18.08 | 12.50 | 22.73 | 52.72 | 40.00 | 46.63 | 34.92 |
| UniAD | CVPR'23 | 45.81 | 16.36 | 129.21 | 43.58 | 14.10 | 17.78 | 21.67 | 10.00 | 14.21 | 15.55 |
| ThinkTwice\* | CVPR'23 | 62.44 | 31.23 | 69.33 | 16.22 | 13.72 | 22.93 | 52.99 | 50.00 | 47.78 | 37.48 |
| DriveAdapter\* | ICCV'23 | 64.22 | 33.08 | 70.22 | 16.01 | 14.55 | 22.61 | 54.04 | 50.00 | 50.45 | 38.33 |
| VAD | ICCV'23 | 42.35 | 15.00 | 157.94 | 46.01 | 8.11 | 24.44 | 18.64 | 20.00 | 19.15 | 18.07 |
| GenAD | ECCV'24 | 44.81 | 15.90 | – | – | – | – | – | – | – | – |
| DriveTransformer | ICLR'25 | 63.46 | 35.01 | 100.64 | 20.78 | 17.57 | 35.00 | 48.36 | 40.00 | 52.10 | 38.60 |
| SparseDrive | ICRA'25 | 44.54 | 16.71 | 170.21 | 48.63 | – | – | – | – | – | – |
| SimLingo | CVPR'25 | 86.02 | 67.27 | 259.23 | 33.67 | – | – | – | – | – | – |
| Hydra-NeXt | ICCV'25 | 73.86 | 50.00 | 197.76 | 20.68 | 40.00 | 64.44 | 61.67 | 50.00 | 50.00 | 53.22 |
| HiP-AD | ICCV'25 | 86.77 | 69.09 | 203.12 | 19.36 | 50.00 | 84.44 | 83.33 | 40.00 | 72.10 | 65.98 |
| MomAD (SD) | CVPR'25 | 47.91 | 18.11 | 174.91 | 51.20 | 13.21 | 21.02 | 18.01 | 20.00 | 21.07 | 18.66 |
| FUMP | arXiv'25 | 45.67 | 16.36 | – | – | 12.50 | 24.44 | 20.00 | 21.50 | 19.15 | 19.51 |
| DIVER (SD) | TPAMI'26 | 49.21 | 21.56 | 177.00 | 54.72 | 15.98 | 28.22 | 23.71 | 20.00 | 24.38 | 22.46 |
| GraphWorld (SD) | arXiv'26 | 51.55 | 25.47 | 181.12 | 56.59 | 18.74 | 31.66 | 25.30 | 20.00 | 26.66 | 24.47 |
| GuideFlow | CVPR'26 | 75.21 | 51.36 | – | – | – | – | – | – | – | – |
| SparseDriveV2 | ECCV'26 | 89.15 | 70.00 | 199.84 | 18.32 | 66.25 | 75.55 | 75.00 | 50.00 | 71.57 | 67.67 |
| **MomWorld** | – | 74.07 | 50.00 | 198.87 | 20.43 | 45.63 | 64.21 | 61.71 | 50.00 | 54.10 | 55.07 |

### Table 11: Turning-nuScenes and Adv-nuSc (open-loop)

| Method | Turning Avg L2 | Col 1s | 2s | 3s | Avg | Adv-nuSc Col 1s | 2s | 3s | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UniAD | – | – | – | – | – | 0.800 | 4.100 | 6.960 | 3.950 |
| VAD | – | – | – | – | – | 4.460 | 7.590 | 9.080 | 7.050 |
| SparseDrive | 0.86 | 0.04 | 0.17 | 0.98 | 0.40 | 0.029 | 0.618 | 2.430 | 1.026 |
| DiffusionDrive\* | – | 0.03 | 0.14 | 0.85 | 0.34 | 0.068 | 1.299 | 3.646 | 1.671 |
| MomAD | 0.76 | 0.03 | 0.13 | 0.79 | 0.32 | – | – | – | – |
| DIVER | – | 0.03 | 0.11 | 0.67 | 0.27 | 0.033 | 0.423 | 1.798 | 0.752 |
| GraphWorld | – | 0.03 | 0.12 | 0.72 | 0.28 | 0.028 | 0.420 | 1.780 | 0.742 |
| **MomWorld** | **0.72** | **0.02** | **0.11** | **0.66** | **0.26** | **0.027** | **0.418** | **1.755** | **0.733** |

### Table 12: Collision rate (%) under weather corruptions, nuScenes-C

| Method | Snow 1s | 2s | 3s | Rain 1s | 2s | 3s | Fog 1s | 2s | 3s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SparseDrive | 0.13 | 0.27 | 0.50 | 0.11 | 0.27 | 0.55 | 0.14 | 0.36 | 0.58 |
| DiffusionDrive\* | 0.09 | 0.24 | 0.39 | 0.07 | 0.18 | 0.35 | 0.06 | 0.18 | 0.30 |
| MomAD | 0.08 | 0.16 | 0.30 | 0.06 | 0.17 | 0.31 | 0.06 | 0.19 | 0.32 |
| DIVER | 0.07 | 0.13 | 0.25 | 0.05 | 0.16 | 0.27 | 0.04 | 0.16 | 0.25 |
| GraphWorld | 0.07 | 0.15 | 0.28 | 0.05 | 0.15 | 0.29 | 0.05 | 0.17 | 0.29 |
| **MomWorld** | **0.06** | **0.13** | **0.23** | **0.04** | **0.15** | **0.25** | **0.04** | **0.16** | **0.23** |

---

## Internal Regularities in the Ablation Tables {#regularities}

These are properties of the printed numbers, checked by arithmetic on Tables 6–8. They are recorded without an explanation, because the paper offers none.

**1. Every step of the cumulative study improves every metric, by nearly the same amount.**

| Column (Table 6) | The five successive increments |
|---|---|
| L2@3 | −0.06, −0.06, −0.05, −0.05, −0.05 |
| L2@6 | −0.03, −0.03, −0.03, −0.03, −0.02 |
| TPC@6 | −0.04, −0.03, −0.03, −0.03, −0.02 |
| Col@3 | −0.03, −0.03, −0.02, −0.02, −0.02 |
| Col@6 | −0.04, −0.03, −0.03, −0.03, −0.03 |
| navhard LK | +0.2, +0.2, +0.1, +0.1, +0.1 |
| navhard EC | +1.1, +1.3, +0.9, +1.0, +0.7 |
| navhard EPDMS | +0.3, +0.2, +0.2, +0.2, +0.2 |

All 45 increments point the same way. Five architecturally different components (a memory, a rollout, a gating rule, a flow, a fusion rule) each move nine metrics on two benchmarks by an almost identical step.

**2. Temporal consistency and displacement error differ by a constant.** TPC@6 measures how far two consecutive predictions are from each other. L2@6 measures how far a prediction is from the ground truth. Across all **27 ablation rows** in Tables 6, 7 and 8 (24 distinct configurations; the full model appears three times and the no-MoFlow model twice):

$$\mathrm{L2@6}-\mathrm{TPC@6}\approx0.855\ \text{m},\qquad \text{range }0.84\text{–}0.88,\quad \text{standard deviation }0.008$$

For the other methods in Tables 1 and 2 the same difference is 0.66 (UniAD), 0.96 (SparseDrive) and 0.84 (MomAD).

**3. In the MoLWM table, L2@6 is an affine function of L2@3.**

$$\mathrm{L2@6}=2.31+3.0\times(\mathrm{L2@3}-0.86)$$

holds to the printed precision for seven of the eight ablated variants and within 0.01 for the eighth (the full model defines the intercept). The eight variants therefore have the same rank order on L2@3, L2@6 and TPC@6.

**4. NAVSIM PDMS tracks nuScenes L2 across variants.** In Table 7, $\mathrm{PDMS}\approx90.2-20\times(\mathrm{L2@3}-0.86)$ within 0.2 for seven of the eight ablated variants (the exception is "no reset gate", off by 0.4). The PDMS column comes from a GTRS-Dense-based LiDAR+camera model trained on navtrain; the L2 column comes from a MomAD-based camera model trained on nuScenes. The same design removal produces proportional changes in both.

**5. Removing one sub-design is worse than removing the whole module.** The no-component baseline scores 41.7 on navhard (Table 6). **All eight** single-design removals in Table 7 score below it (37.9 to 41.3), with "no reset gate" at 37.9. Three MoFlow variants in Table 8 do too (39.5 to 40.5). A partly disabled module can hurt a model, so this is possible; it does mean the method is 1.1 above its base and up to 3.8 below it depending on one gate.

**6. Both no-component rows are other papers' published numbers.** The nuScenes half is MomAD's row and the navhard half is GTRS-Dense's row, each identical to the digit. The paper says controlled comparisons "use identical inputs, backbones, candidate vocabularies, training data, and optimization budgets". Whether the baselines were retrained under that budget or copied is not stated.

**What the wiki does with this.** Independent training runs of different architectures do not normally produce columns that are exact functions of one another. The wiki therefore:
- records the **end-to-end results** (Tables 1–5, 10) as the paper's claims;
- does **not** cite any per-component effect size from Tables 6–8 as evidence on other pages;
- sets this page's confidence to **low**.

A code link is given. Released checkpoints and training logs for the ablation variants would settle the question either way.

---

## Reading the Results

### 1. The long-horizon method gains most at short horizons

| Horizon | L2: MomAD → MomWorld | Δ | Collision: MomAD → MomWorld | Δ |
|---|---|---:|---|---:|
| 1 s | 0.41 → 0.27 | −34% | 0.17 → 0.02 | −88% |
| 2 s | 0.85 → 0.52 | −39% | 0.30 → 0.17 | −43% |
| 3 s | 1.13 → 0.86 | −24% | 0.54 → 0.42 | −22% |
| 4 s | 1.67 → 1.28 | −23% | 0.83 → 0.83 | **0%** |
| 5 s | 1.98 → 1.76 | −11% | 1.43 → 1.36 | −5% |
| 6 s | 2.45 → 2.31 | **−6%** | 2.13 → 1.97 | −8% |

- The relative L2 gain shrinks steadily with horizon. The abstract's "12.2% lower average collision rate" is an average over six horizons; 63% of the summed reduction comes from 1–3 s.
- At 4 s MomWorld's collision rate equals MomAD's and is above Epona (0.74), World4Drive (0.76), GuideFlow (0.79) and DIVER (0.80).
- This sits awkwardly with the design. MoFlow's fusion weight is 0.05 at the first waypoint, so the refinement stage contributes almost nothing at 1 s, where the gain is largest.
- The 1 s collision rate of 0.02% is six times lower than the next best (GuideFlow 0.12).
- Which rows in Table 1 were re-run for six seconds is not stated. LAW, Epona and World4Drive appear elsewhere in this wiki only with three-second results, and the TPC table covers only UniAD, VAD, SparseDrive and MomAD.
- A six-second row cannot be compared with a three-second row at the same horizon: MomAD's six-second model is 0.41 / 0.85 / 1.13 at 1–3 s against 0.31 / 0.57 / 0.91 for its three-second model in [[sources/foresight.md]]'s table. See [[concepts/nuscenes-waymo-evals.md#six-second]].

### 2. On NAVSIM this is a GTRS-Dense scorer with a future memory

- **Inputs and supervision**: camera + LiDAR, 2048×512 images, V2-99, a 16,384-trajectory vocabulary, heads trained on PDM sub-scores, and a weighted log-score. That is the [[sources/hydra-mdp-pp.md]] recipe, so every caveat of the scorer cohort applies ([[concepts/selection-based-planning.md]]).
- **The contribution over the base is small.** navhard: 41.7 → 42.8. NAVSIM v1: the paper gives 89.7 without MoFlow and 90.2 with it, and no PDMS for the base without MoLWM.
- **The future memory is shared across candidates.** One rollout, not conditioned on any trajectory, is read by all 16,384 candidate queries. This is the configuration [[sources/da-wam.md]] labels (c) and measures at **−0.50 PDMS** against no future prediction. MomWorld reports the opposite sign with a feature-space target, which is the family the wiki's negative results come from. Given the [regularities](#regularities), the wiki does not count this as a further positive shared-future result alongside [[sources/geoworldad.md]].
- **A 40-step, 10 Hz rollout with image-encoder targets at every step** is described for NAVSIM, whose public sensor logs are 2 Hz. How future image targets are obtained at 0.1 s spacing is not explained.

#### Against its own base: progress up, safety down {#vs-gtrs}

GTRS-Dense's published navhard row (as reproduced in [[sources/had.md]]'s table) against MomWorld's:

| Stage | Method | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | GTRS-Dense | 98.7 | 95.8 | 99.4 | 99.3 | 72.8 | 98.7 | 95.1 | 96.9 | 40.4 |
| 1 | MomWorld | 96.9 | 93.6 | 99.8 | 99.8 | 80.4 | 96.9 | 96.4 | 97.6 | 60.0 |
| 2 | GTRS-Dense | 91.4 | 89.2 | 94.4 | 98.8 | 69.5 | 90.1 | 54.6 | 94.1 | 49.7 |
| 2 | MomWorld | 86.4 | 88.2 | 94.2 | 98.3 | 81.7 | 84.4 | 55.3 | 97.0 | 54.7 |

- NC, DAC and TTC are lower in both stages (Stage 2: −5.0, −1.0, −5.7). EP, EC and HC are higher (Stage 2: +12.2, +5.0, +2.9; Stage-1 EC +19.6).
- The component ablation (Table 6) reports LK, EP and EC only, which are three of the columns that rise.
- The closed-form estimate from mean sub-scores (Stage 1 × Stage 2) is **43.8 for GTRS-Dense and 43.2 for MomWorld**, against reported 41.7 and 42.8. The estimate is only good to about ±2 ([[concepts/navhard-ood-evaluation.md]]), so it cannot confirm or reject a +1.1 difference. It does show the gain is not visible in the averaged sub-scores.
- The sign pattern is the reverse of DA-WAM's shared-future row, where NC and TTC rose and EP collapsed. It resembles [[sources/geoworldad.md]]'s shared-future result (EP up, safety flat), except that here safety falls.

### 3. The comparison tables

**navhard (Table 5).** The baseline rows are digit-identical to [[sources/drivefuture.md]]'s Table 1 (TransFuser 23.1, DiffusionDrive 24.2, GuideFlow 27.1, MindDrive 30.9, World4Drive 34.9, SpanVLA 40.1, DriveSuprim 42.1). Every entry of that table above 42.8 is missing:

| Omitted | navhard EPDMS | Note |
|---|---:|---|
| **DriveFuture + GTRS-Dense** | **55.5** | Ziying Song is its corresponding author and Lei Yang a co-author; both are MomWorld authors. Same scorer family as MomWorld's base |
| DrivoR | 54.6 | |
| SimScale | 53.2 | |
| GTRS-E | 49.4 | The ensemble of the family MomWorld builds on |
| ZTRS | 48.1 | |
| DiffVLA | 45.0 | |

DriveFuture is cited in the related-work section and appears in none of the three NAVSIM tables (it reports 90.7 PDMS and 89.9 corrected EPDMS, against 90.2 and 90.1 here). "MomWorld achieves the best EPDMS of 42.8" holds only inside Table 5. In the wiki's navhard cohort it is seventh.

**NAVSIM v1 (Table 3).** "Best PDMS among world-model methods at 90.2, outperforming DriveLaW by 1.1" omits WA-JEPA 91.8, SimWAM 91.5, ReDrive 91.0, GeoWorldAD 91.0, DriveFuture 90.7, ReWorld 90.4 and DA-WAM 93.7. DriveWorld-VLA (91.3), a latent-world-model method from overlapping authors, is filed under "VLA-based" in the same table. DriveSuprim is carried at its ResNet-34 value (89.9) without a label. The "PWM" row (88.1) is Policy World Model, but the citation points to an unrelated robotics paper with the same acronym.

**NAVSIM v2 navtest (Table 4).** The MomWorld row is corrected-like (closed form 89.9 against 90.1). It sits in one column with pre-fix rows (TransFuser 76.7, DiffusionDriveV2 85.5) and is compared against Latent-WAM 89.3 as the strongest baseline, omitting WA-JEPA 91.7, SUV 91.0, ReDrive 90.8 and the rest of the corrected cohort. EC 90.6 would be among the highest in the wiki.

### 4. Bench2Drive

| | DS | SR | Effi. | Comf. | Merge | Overtake | Em. Brake | Give Way | Traffic Sign |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hydra-NeXt | 73.86 | 50.00 | 197.76 | 20.68 | 40.00 | 64.44 | 61.67 | 50.00 | 50.00 |
| MomWorld | 74.07 | 50.00 | 198.87 | 20.43 | 45.63 | 64.21 | 61.71 | 50.00 | 54.10 |

- The base model for Bench2Drive is not stated. The row is Hydra-NeXt's within about one point everywhere except Merging (+5.6) and Traffic Sign (+4.1), with an identical success rate.
- The reported mean ability (55.07) is not the mean of the five printed abilities (55.13).
- The other momentum-lineage rows (MomAD 47.91, DIVER 49.21, GraphWorld 51.55) are on a SparseDrive base, 22 to 26 DS lower. The Bench2Drive result therefore says more about the base planner than about momentum.
- 74.07 DS is below the VLA entries on [[concepts/bench2drive.md]] and below HiP-AD (86.77) and SparseDriveV2 (89.15) in its own table, which the paper acknowledges.

### 5. Robustness tables

Tables 11 and 12 have 22 MomWorld cells. MomWorld is best or tied in all of them, by 0.001 to 0.04 percentage points over the best prior value (for example Adv-nuSc 0.027 / 0.418 / 1.755 against GraphWorld's 0.028 / 0.420 / 1.780). The paper says: "Published values are literature references rather than matched reruns. The shaded MomWorld rows are reserved for evaluations using identical data manifests and evaluator settings." Turning-nuScenes has 680 samples, so a 0.01-point collision difference is less than one sample.

### 6. Cost

130.5 ms without MoFlow and 138.9 ms with it, on unstated hardware. 1000 / 138.9 is the 7.2 FPS of Table 1, so the figure is for the nuScenes model. MoFlow costs about 2.1 ms per Euler step. No latency is given for the NAVSIM configuration (16,384 candidates at 2048×512).

---

## Relationships

- **[[sources/drivefuture.md]]**: shared authors, the same GTRS-Dense scorer, the same navhard baseline rows, and absent from every MomWorld table. DriveFuture conditions a diffusion planner on a 16-token future latent; MomWorld gives a scorer a 40-token action-free future memory. DriveFuture's scored result is 12.7 navhard points higher.
- **[[sources/da-wam.md]]**: the direct counter-position. DA-WAM predicts one future per candidate because a shared future "cannot tell the scorer which candidate causes a hazard". MomWorld argues candidate queries attending to a shared memory is enough.
- **[[sources/latent-wam.md]]**: cited as the strongest v2 baseline (89.3) and as the nearest "compact latent world state" design. Latent-WAM predicts 16 scene queries per view; MomWorld predicts one pooled vector and its first difference.
- **[[sources/policy-world-model.md]]**: the other action-free future forecaster in the wiki. PWM forecasts future frame tokens before the action; MomWorld forecasts a pooled latent before scoring.
- **[[sources/spanvla.md]]**: the other flow-matching planner whose source distribution is not noise (SpanVLA starts from a history-derived initialization). MomWorld starts from the scorer's selected trajectory.
- **[[sources/drive-jepa.md]]**: a different "momentum". Drive-JEPA's momentum-aware selection compares each proposal with the previously selected trajectory to repair extended comfort. MomWorld's momentum is a latent rate of change.
- **[[sources/redrive.md]]** and other training-time-only models: MomWorld's rollout runs at inference (40 steps) and its future image targets are training-only.
- **Same-group baselines, none ingested**: MomAD, DIVER, GraphWorld, GuideFlow, FUMP, DriveWorld-VLA. These six comparison methods share authors with this paper.
- **New to the wiki**: RAP-DINO (36.9 navhard), Hydra-NeXt (73.86 DS Bench2Drive), HiP-AD (86.77 DS), SparseDriveV2's Bench2Drive result (89.15 DS).

---

## Limitations

**Evidence**

1. **The ablation tables contain exact cross-column relationships** that independent runs would not be expected to produce ([above](#regularities)). Per-component effect sizes are not usable.
2. **The base-model contribution is not separated by benchmark.** Three base planners are used and only the NAVSIM one is named. The no-component ablation rows are the base models' published numbers.
3. **Robustness margins are below one sample** on Turning-nuScenes and are compared with unmatched literature values.
4. **No seeds, no variance, single runs.**

**Comparison**

5. **The navhard table omits six higher entries from the table its rows come from**, including the authors' own DriveFuture (55.5).
6. **NAVSIM v1 and v2 "best" claims** hold only inside tables that stop at 89.1 and 89.3.
7. **The v2 navtest table mixes evaluator conventions**, swaps two ReCogDrive columns, and is described by the paper itself as non-official.
8. **A citation error**: PWM's row is attributed to a different paper.

**Method**

9. **The world model is not action-conditioned** and its state is a single pooled vector per step. What such a rollout can represent about other agents is not examined; no prediction-quality metric is reported.
10. **Perception labels and PDM sub-scores are required** (map/detection heads, agent tracks for the auxiliary loss, simulator-derived scorer targets), plus LiDAR input on NAVSIM.
11. **MoFlow's effective strength is unknown.** The global gate starts at $\sigma(-4)\approx0.018$ and its learned value is not reported.
12. **Gains are front-loaded** (largest at 1–2 s), contrary to the long-horizon framing. On navhard the safety sub-scores (NC, DAC, TTC) are below the base model's in both stages.
13. **The paper's stated limitation** is reliance on a fixed candidate vocabulary and scoring stage.
14. **Latency hardware is unstated**, and the NAVSIM model's cost is not given.

**Source conversion**

15. All six figures and twelve tables are present. The text refers to "Table 5" for three different NAVSIM tables. The front matter's author field is empty; the author line is in the body.

---

## Key Cross-References

- [[concepts/world-model-for-ad.md]] — Pattern 41: a second-order, action-free latent rollout whose memory is shared across candidates.
- [[concepts/selection-based-planning.md]] — a GTRS-Dense scorer with a future memory; the base is 41.7 of its 42.8.
- [[concepts/navhard-ood-evaluation.md]] — 42.8 in the scorer cohort; a table with its top six rows removed.
- [[concepts/navsim-benchmark.md]] — 90.2 PDMS (camera + LiDAR, scorer) and 90.1 EPDMS on a non-official navtest protocol.
- [[concepts/nuscenes-waymo-evals.md]] — the six-second protocol, TPC, and three robustness subsets.
- [[concepts/bench2drive.md]] — 74.07 DS, level with Hydra-NeXt.
- [[concepts/diffusion-planner.md]] — a plan-to-expert residual flow with bounded horizon-aware fusion.
- [[concepts/evaluation-variance.md]] — cross-column regularity as a check on ablation tables.

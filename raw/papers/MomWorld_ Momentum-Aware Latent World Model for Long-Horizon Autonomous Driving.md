---
title: "MomWorld: Momentum-Aware Latent World Model for Long-Horizon Autonomous Driving"
source: "https://arxiv.org/html/2609.33737v1"
author:
published:
created: 2026-09-30
description:
tags:
  - "clippings"
---
Ziying Song Affiliation: Nanyang Technological University    Shengkai Zhang Affiliation: Beijing Jiaotong University    Lei Yang Affiliation: Nanyang Technological University    Haozhuang Chi Affiliation: Nanyang Technological University    Yuchen Liu Affiliation: North University of China    Jiangtao Su Affiliation: Nanyang Technological University    Lin Liu Affiliation: Dalian University of Technology    Ziyang Liu Affiliation: Tsinghua University [https://github.com/modaxiansheng/MomWorld](https://github.com/modaxiansheng/MomWorld)    Chen Lv Affiliation: Nanyang Technological University

###### Abstract

Long-horizon planning enables autonomous vehicles to anticipate scene evolution and potential risks, supporting safe and stable decisions in complex interactions. However, existing methods struggle to propagate motion trends from observed history into the future. Long rollouts based on a single latent state may further attenuate useful dynamics, retain stale motion patterns, and disrupt reliable near-term plans. We introduce MomWorld, a momentum-aware latent world model for long-horizon planning. MomWorld extracts scene motion trends from historical-to-current observations and propagates latent momentum into future horizons, jointly predicting future configuration and momentum states. A learnable momentum persistence mechanism preserves stable trends, scene-conditioned momentum updates adapt future dynamics, and a scene-adaptive reset gate suppresses stale momentum under abrupt changes. We further propose MoFlow, a momentum-conditioned flow-matching module that refines a base trajectory to align with the predicted future scene evolution in only a few integration steps, with a horizon-aware residual fusion that preserves near-term planning stability while permitting stronger long-range corrections. Extensive experiments on NAVSIM, nuScenes and Bench2Drive demonstrate that MomWorld improves long-horizon planning consistency and reduces the average collision rate by 12.2% relative to MomAD over a 6-second planning horizon.

## 1 Introduction

![[motivationv9.png|Refer to caption]]

Figure 1: Motivation and comparison of MomWorld. (a) Reliable future scene evolution remains a key challenge for long-horizon planning. (b) MomAD 35 derives ego-centric momentum from historical and current evidence to stabilize ego dynamics. (c) MomWorld elevates momentum to a world-level latent state, initializes it from past-to-present evidence, and jointly rolls out future configuration and momentum to model world dynamics. (d) On six-second nuScenes planning, MomWorld reduces L2 from 2.45 to 2.31 m and collision rate from 2.13% to 1.97%. The corresponding averages improve from 1.42 to 1.17 m and from 0.90% to 0.79%.

Long-horizon planning has become a central frontier in autonomous driving because safe decisions depend not only on the current scene, but also on how traffic may evolve over the next several seconds. Planning-oriented end-to-end systems have advanced this goal by jointly optimizing perception, prediction, and planning within a shared representation [^14] [^19] [^40]. More recent history-aware methods extend temporal context, while generative planners enlarge the set of feasible future behaviors [^52] [^35] [^27] [^37]. Despite this progress, increasing the horizon remains difficult because rollout errors accumulate, interactions such as cut-ins and sudden braking can quickly invalidate earlier assumptions, and uncertain long-range predictions may perturb otherwise reliable near-term decisions. Effective long-horizon planning therefore requires a temporally coherent model of scene evolution rather than a sequence of independently predicted states.

Latent world models offer a promising route to this objective. By forecasting compact, task-relevant future representations, they support planning-oriented imagination without reconstructing every future observation, making multi-step prediction both efficient and directly useful for decision making [^32] [^21] [^55] [^36]. However, existing planning-oriented latent world models generally do not explicitly factorize scene configuration and motion trend into jointly propagated latent states. Repeated transitions may consequently attenuate useful dynamics during smooth motion, retain stale patterns after abrupt interactions, or repeatedly infer the same motion cues from scratch. Moreover, conditioning every planning step on uncertain imagined futures can compromise accurate near-term decisions. The key challenge is therefore to preserve stable trends, refresh them when the scene changes, and introduce future imagination into planning in a controlled manner.

MomAD [^35] provides an important insight. Its trajectory and perception momentum connect historical planning and spatiotemporal evidence to the current decision, improving temporal continuity between the past and present. This motivates us to ask whether momentum can connect not only history and the current scene, but also the predicted future. These momentum mechanisms operate within perception and planning queries to stabilize the current ego plan. They do not constitute an explicit latent world state jointly propagated with future scene configurations. As illustrated in Figure 1, extending this idea into future world dynamics can distinguish similar current configurations that imply different evolutions, while allowing obsolete motion trends to be suppressed when an interaction changes abruptly.

We introduce MomWorld, a momentum-aware latent world model for long-horizon planning. MomWorld initializes latent scene state and momentum from historical-to-current observations, then jointly propagates them across the predicted future without using future observations at inference. Its first innovation is a scene-adaptive momentum rollout. Learnable persistence retains stable trends, scene-conditioned innovations introduce new dynamics, and a scene-change gate suppresses stale momentum under abrupt interactions. Its second innovation is MoFlow, a momentum-conditioned flow-matching module [^28] that refines a strong base trajectory toward futures consistent with the predicted scene evolution in only a few integration steps. Bounded horizon-aware residual fusion preserves near-term stability while permitting larger long-range corrections. Experiments on NAVSIM [^9] [^2] and nuScenes [^1] demonstrate that MomWorld improves long-horizon planning consistency and reduces the average predicted collision rate across the 1–6 s nuScenes horizon by 12.2% relative to MomAD.

We make four contributions.

- We propose MomWorld, which connects observed history with future imagination through Latent World Rollout (LWR).
- We introduce MoLWM, which maintains, updates, and resets latent momentum to model future dynamics and construct Future World Memory.
- We develop MoFlow, which translates predicted world evolution into bounded, horizon-aware trajectory corrections for stable planning.
- We conduct extensive evaluations on NAVSIM, nuScenes, and Bench2Drive demonstrating stronger long-horizon consistency and a 12.2% relative reduction in average predicted collision rate across the 1–6 s nuScenes horizon over MomAD.

## 2 Related Work

### 2.1 End-to-End Autonomous Driving

End-to-end autonomous driving jointly learns perception, prediction, and planning around a shared driving objective. UniAD and VAD establish unified planning-oriented architectures with query-based and vectorized scene representations [^14] [^19]. Recent methods improve computational efficiency through sparse or parallel designs [^40] [^43], exploit historical predictions and momentum to enhance temporal consistency [^52] [^35], and improve multimodal trajectory generation and selection through truncated diffusion, reinforced diffusion, constraint-guided flow matching, and generalized trajectory scoring [^27] [^37] [^30] [^26]. These methods primarily improve scene representation, temporal query reuse, or trajectory generation and selection rather than jointly rolling forward an explicit latent state of scene configuration and motion dynamics. MomWorld addresses this gap by rolling past-to-present momentum forward with future latent world states before trajectory refinement.

### 2.2 Latent World Models for Autonomous Driving

Latent world models encode future scene evolution into compact representations for planning. DriveWorld learns spatiotemporal features with dynamic memory, while LAW predicts ego-trajectory-conditioned future features for self-supervised planning supervision [^32] [^21]. World4Drive augments physical latents with spatial-semantic priors and multimodal intentions, whereas Epona jointly models future video and trajectories through autoregressive diffusion [^55] [^53]. Complementary world-model directions focus on human and heterogeneous agent dynamics: Driver-WM causally rolls out in-cabin driver dynamics conditioned on external traffic context [^6], whereas PV-WM recurrently co-rolls articulated pedestrians and rigid vehicles within a synchronized structured state [^5]. Other approaches couple latent imagination with planning more directly: DriveLaW injects video latents into a diffusion planner, DriveFuture conditions diffusion planning on predicted future latents, Latent-WAM autoregressively forecasts compact world states, and GraphWorld transports an ego-centric relational state before modulating motion and planning queries [^45] [^13] [^42] [^36]. In contrast, MomWorld jointly rolls out coupled configuration–momentum states and applies bounded, horizon-aware MoFlow refinement, preserving near-term stability while enabling stronger long-range corrections.

![[MomWorldv5.png|Refer to caption]]

Figure 2: Overview of MomWorld. Historical and current multi-view images are encoded into Scene Queries, from which MoLWM initializes latent scene state and momentum and propagates them through Latent World Rollout (LWR) to construct Future World Memory for candidate scoring and Plan selection. Guided by history-conditioned momentum and predicted future evolution, MoFlow applies bounded residual refinement to produce the temporally consistent Refined trajectory τ ~ \\widetilde{\\tau}.

## 3 Method

### 3.1 Overview of MomWorld

MomWorld is an end-to-end autonomous-driving framework designed to improve long-horizon planning through momentum-aware latent world modeling. As shown in Fig. 2, historical and current multi-view images are encoded into Scene Queries, with Map/Det heads retained for perception supervision. MomWorld comprises two key components. Momentum-Aware Latent World Modeling uses Latent World Rollout (LWR) to propagate latent scene dynamics and momentum into Future World Memory for candidate scoring and Plan selection. Momentum-Conditioned Flow Matching then uses history-conditioned momentum and predicted future evolution to refine the selected Plan into a temporally consistent trajectory.

### 3.2 Momentum-Aware Latent World Modeling (MoLWM)

MoLWM represents the latent driving world using a scene state $z_{k}\in\mathbb{R}^{D}$ and momentum $p_{k}\in\mathbb{R}^{D}$, which encode the scene configuration and its temporal dynamics, respectively. Initialized from historical and current Scene Queries, MoLWM recursively propagates both over the planning horizon to construct Future World Memory for Candidate Planning.

#### History-Conditioned Initialization.

Temporal Fusion combines the current scene representation with its historical variation to initialize $z_{0}$ and $p_{0}$.

$$
(z_{0},p_{0})=f_{\mathrm{temp}}\!\left([\operatorname{Pool}(Q_{t}),\operatorname{Pool}(Q_{t})-\operatorname{Pool}(Q_{t-1})]\right).
$$

The initialized state anchors the current scene configuration, whereas the initialized momentum captures its history-conditioned evolution.

#### Scene-Adaptive Momentum Dynamics (SMD).

To adapt inherited momentum to future scene changes, a multi-head transition network predicts a retention gate $\rho_{k}$, a reset gate $g_{k}$, and a momentum proposal $u_{k}$ from the preceding scene state, momentum, and horizon embedding:

$$
(\rho_{k},g_{k},u_{k})=f_{\mathrm{gate}}([z_{k-1},p_{k-1},r_{k}^{\mathrm{time}}]).
$$

The two gates use sigmoid activations, while the momentum proposal uses a hyperbolic tangent activation. The future momentum is updated by

$$
p_{k}=\operatorname{LN}\!\left(\rho_{k}\odot(1-g_{k})\odot p_{k-1}+(1-\rho_{k})\odot u_{k}\right).
$$

Here, *Maintain* preserves persistent momentum through $\rho_{k}$, *Update* introduces the scene-conditioned proposal $u_{k}$, and *Reset* suppresses stale inherited momentum through $g_{k}$. Together, these operations produce the scene-adaptive momentum $p_{k}$ for the $k$ -th future rollout step.

#### Latent World Rollout (LWR).

LWR recursively generates paired future State–Momentum predictions from the history-conditioned pair $(z_{0},p_{0})$ at the current planning time $t$. At step $k$, SMD predicts $p_{k}$ from $(z_{k-1},p_{k-1})$ and the horizon embedding through Eqs. (2)–(3). The predicted momentum then advances the corresponding scene state:

$$
z_{k}=z_{k-1}+\Delta t\,\pi_{p}(p_{k}),\qquad k=1,\ldots,H.
$$

Here, $\pi_{p}$ maps momentum to a latent transition and $\Delta t$ is the rollout interval. Feeding each updated pair into the next step produces $\{(z_{k},p_{k})\}_{k=1}^{H}$. Taking $t$ as the current time index, the $k$ -th pair corresponds to future step $t+k$ and physical offset $k\Delta t$. The state $z_{k}$ predicts the scene configuration, while $p_{k}$ encodes history-grounded dynamics adapted to its predicted future evolution.

Algorithm 1 summarizes how training-only future targets ground the rollout. Future observations are encoded into detached target Scene Queries:

$$
Q_{t+k}^{\star}=\operatorname{sg}\!\left(\operatorname{ImageEncoder}(\mathcal{O}_{t+k})\right),\qquad k=1,\ldots,H.
$$

Here, $\operatorname{sg}(\cdot)$ blocks gradients through the target branch. The target queries supervise rollout alignment, while future ego, agent, and presence states provide auxiliary supervision. All future targets are used only during training and are unavailable at inference.

Algorithm 1 Future-Supervised Latent World Rollout

Input: Initial state $z_{0}$ and momentum $p_{0}$; horizon $H$; future observations $\{\mathcal{O}_{t+k}\}_{k=1}^{H}$; auxiliary targets $\{y_{t+k}^{\star}\}_{k=1}^{H}$

Output: Future states and momenta $\{(z_{k},p_{k})\}_{k=1}^{H}$; $\mathcal{L}_{\mathrm{future}}$ and $\mathcal{L}_{\mathrm{aux}}$

Initialize: $\mathcal{L}_{\mathrm{future}},\mathcal{L}_{\mathrm{aux}}\leftarrow 0$

for *$k\leftarrow 1$ to $H$* do

   Latent World Rollout: obtain $(z_{k},p_{k})$ using Eqs. (2)–(4)

   Future target: construct $Q_{t+k}^{\star}$ using Eq. (5)

   Projection: $\widehat{Q}_{t+k}\leftarrow f_{Q}([z_{k},p_{k}])$

   Alignment: $\mathcal{L}_{\mathrm{future}}\mathrel{+}=d_{Q}(\widehat{Q}_{t+k},Q_{t+k}^{\star})$

   Auxiliary loss: $\mathcal{L}_{\mathrm{aux}}\mathrel{+}=\ell_{\mathrm{aux}}(z_{k},p_{k};y_{t+k}^{\star})$

 $\mathcal{L}_{\mathrm{future}}\leftarrow\mathcal{L}_{\mathrm{future}}/H,\quad\mathcal{L}_{\mathrm{aux}}\leftarrow\mathcal{L}_{\mathrm{aux}}/H$

$f_{Q}$ projects each LWR step $(z_{k},p_{k})$ into query space. $d_{Q}$ aligns future queries, while $\ell_{\mathrm{aux}}$ supervises future ego, agent, and presence states.

#### Future World Memory (FWM).

FWM converts each paired LWR prediction into a memory token aligned with its future step:

$$
\begin{gathered}m_{k}=\operatorname{LN}\!\left(z_{k}+\operatorname{MLP}([p_{k},r_{k}^{\mathrm{time}}])\right),\\[-2.0pt]
k=1,\ldots,H.\end{gathered}
$$

The residual path preserves the predicted scene state $z_{k}$, while the learned branch injects its momentum $p_{k}$ and temporal position $r_{k}^{\mathrm{time}}$. Each token therefore captures the future scene, dynamics, and horizon information. Dependencies across future steps are modeled by

$$
M_{t}=\operatorname{SelfAttn}([m_{1},\ldots,m_{H}]).
$$

Self-attention aggregates context across the rollout, producing a temporally informed memory at each planning step. During Candidate Planning, each candidate query cross-attends to $[Q_{t};M_{t}]$. The scoring heads then rank the decoded candidates and select the Top-Score trajectory as the base Plan $\tau^{0}$.

### 3.3 Momentum-Conditioned Flow Matching (MoFlow)

MoFlow refines the selected Plan $\tau^{0}$ using the history-conditioned momentum $p_{0}$ and Future World Memory $M_{t}$. It learns momentum-guided residuals and applies horizon-aware bounded fusion, as shown in Fig. 3, to produce the final trajectory $\widetilde{\tau}$ while preserving near-term stability.

#### Momentum-Guided Residual Flow (MGRF).

The selected Plan $\tau^{0}$ provides a stable anchor but may not fully capture the predicted historical-to-future dynamics. MoFlow therefore learns a residual vector field conditioned on $p_{0}$ and $M_{t}$, which provide history-derived motion and horizon-aligned future dynamics, respectively.

The Trajectory Encoder normalizes positions by a fixed scale $s_{\mathrm{pos}}=50$ and continuously embeds the heading.

$$
\phi(x,y,\psi)=\left(x/s_{\mathrm{pos}},y/s_{\mathrm{pos}},\sin\psi,\cos\psi\right).
$$

Applied waypoint-wise, $\phi$ encodes the base and expert trajectories as $X_{0}=\phi(\tau^{0})$ and $X_{1}=\phi(\tau^{\star})$, respectively.

During training, we sample $\epsilon\sim\mathcal{U}(0,1)$ and construct the linear transport path

$$
X_{\epsilon}=(1-\epsilon)X_{0}+\epsilon X_{1}.
$$

The momentum-conditioned vector field is optimized by

$$
\mathcal{L}_{\mathrm{FM}}=\mathbb{E}_{\epsilon}\left[\left\|v_{\theta}(X_{\epsilon},\epsilon\mid p_{0},M_{t})-(X_{1}-X_{0})\right\|_{2}^{2}\right].
$$

This formulation learns plan-to-expert residual transport rather than noise-based trajectory generation.

At inference, we initialize the flow state as $X^{0}=X_{0}$ and integrate it using $J$ explicit Euler steps with midpoint flow-time conditioning.

$$
X^{j+1}=X^{j}+\frac{1}{J}v_{\theta}\!\left(X^{j},\frac{j+1/2}{J}\mid p_{0},M_{t}\right),\qquad j=0,\ldots,J-1.
$$

The Trajectory Decoder normalizes the final heading representation and maps it back to trajectory space.

$$
\widehat{\tau}=\phi^{-1}\!\left(\mathcal{N}_{\psi}(X^{J})\right).
$$
![[zituv1.drawio.png|Refer to caption]]

Figure 3: Momentum-Conditioned Flow Matching (MoFlow). Conditioned on p 0 p\_{0} and M t M\_{t}, MoFlow refines the base trajectory through residual flow and bounded horizon-aware fusion.

#### Horizon-Aware Residual Fusion (HARF).

The flow-refined proposal $\widehat{\tau}$ carries historical-to-future corrections, but directly replacing $\tau^{0}$ may compromise near-term stability. At each horizon step, we first bound its residual.

$$
\Delta\tau_{k}=\operatorname{clip}_{[-\delta,\delta]}\left(\widehat{\tau}_{k}-\tau_{k}^{0}\right).
$$

Heading differences are wrapped before clipping. The bounded residual is then fused with the selected Plan.

$$
\displaystyle\gamma_{k}
$$
 
$$
\displaystyle=0.05+0.95\frac{k-1}{H-1},
$$
$$
\displaystyle\widetilde{\tau}_{k}
$$
 
$$
\displaystyle=\tau_{k}^{0}+\sigma(b)\gamma_{k}\Delta\tau_{k}.
$$

Here, $\delta$ bounds each residual component, $b$ is a globally learned scalar, and $\gamma_{k}$ linearly increases the correction strength from 0.05 to 1.0 toward longer horizons.

### 3.4 Training Objective

Future-world prediction is optimized by

$$
\mathcal{L}_{\mathrm{world}}=\mathcal{L}_{\mathrm{future}}+0.5\mathcal{L}_{p}+0.2\mathcal{L}_{\mathrm{aux}}.
$$

MoFlow is supervised by

$$
\mathcal{L}_{\mathrm{MoFlow}}=\mathcal{L}_{\mathrm{FM}}+\mathcal{L}_{\tau}.
$$

The complete objective is

$$
\mathcal{L}=\mathcal{L}_{\mathrm{percep}}+\mathcal{L}_{\mathrm{plan}}+\mathcal{L}_{\mathrm{world}}+\mathcal{L}_{\mathrm{MoFlow}}.
$$

Here, $\mathcal{L}_{\mathrm{future}}$, $\mathcal{L}_{p}$, and $\mathcal{L}_{\mathrm{aux}}$ supervise future queries, momentum, and auxiliary future states, while $\mathcal{L}_{\mathrm{FM}}$ and $\mathcal{L}_{\tau}$ supervise residual flow and the Refined trajectory. $\mathcal{L}_{\mathrm{percep}}$ and $\mathcal{L}_{\mathrm{plan}}$ are the standard Map/Det and candidate-planning losses. Full definitions are provided in Appendix 1.5.

## 4 Experiments

### 4.1 Experimental Setup

#### Benchmarks.

On nuScenes [^1], we use the official train/validation split and the camera-only setting. We train separate models for the conventional 3-s horizon (six waypoints) and our 6-s extension (twelve waypoints), both sampled at 2 Hz. The long-horizon model is never obtained by extrapolating a 3-s checkpoint. The 6-s protocol reports planning performance at every second from 1 to 6 s. On NAVSIM v1 [^9], we train on *NavTrain* and evaluate all 12,146 indexed scenarios of the official *navtest* split with the classic PDMS. On NAVSIM v2 [^2], the primary result is the official two-stage *navhard* pseudo-simulation score with corrected human-reference filtering.

Due to space constraints, Bench2Drive results are provided in the Appendix.

#### Metrics.

For nuScenes, we report horizon-wise $L_{2}$ and predicted collision rate, where lower is better. TPC follows the public MomAD evaluator and measures the dataset-level Euclidean discrepancy between consecutive trajectory predictions [^35]. NAVSIM v1 reports PDMS and its components, while NAVSIM v2 reports EPDMS and the navhard stage-wise decomposition, where higher is better.

#### Implementation Details.

For NAVSIM, MomWorld is implemented on GTRS-Dense [^26] with a V-99-eSE VoVNet backbone and a $D=256$ latent space. The model takes two camera frames and four LiDAR sweeps, produces 40 LWR steps at $\Delta t=0.1$  s, and scores a fixed vocabulary of 16,384 trajectories. MoFlow refines the selected trajectory using four explicit Euler steps with midpoint-time conditioning. We train the NAVSIM model for 20 epochs using Adam with a learning rate of $2\times 10^{-4}$, a per-device batch size of 2, mixed precision, and gradient clipping at 5.0. The nuScenes experiments follow the camera-only 3-s and 6-s settings described above. Controlled comparisons use identical inputs, backbones, candidate vocabularies, training data, and optimization budgets within each benchmark.

### 4.2 Main Results

Table 1: Six-second planning results on the nuScenes validation set. Reported FPS uses an A100 for UniAD, an RTX 3090 for LAW, and an RTX 4090 for SparseDrive, MomAD, and GuideFlow.

<table><thead><tr><th rowspan="2"><math><semantics><mi>Method</mi> <annotation>\operatorname{Method}</annotation></semantics></math></th><th rowspan="2"><math><semantics><mi>Venue</mi> <annotation>\operatorname{Venue}</annotation></semantics></math></th><th colspan="7"><math><semantics><mrow><mrow><mi>L2</mi> <mo>⁡</mo> <mrow><mo>(</mo><mi>m</mi><mo>)</mo></mrow></mrow> <mo>↓</mo></mrow> <annotation>\operatorname{L2\ (m)}\downarrow</annotation></semantics></math></th><th colspan="7"><math><semantics><mrow><mrow><mi>Col</mi><mo>.</mo><mi>Rate</mi> <mo>(</mo><mo>%</mo><mo>)</mo></mrow> <mo>↓</mo></mrow> <annotation>\operatorname{Col.\ Rate\ (\%)}\downarrow</annotation></semantics></math></th><th rowspan="2"><math><semantics><mi>FPS</mi> <annotation>\operatorname{FPS}</annotation></semantics></math></th></tr><tr><th>1s</th><th>2s</th><th>3s</th><th>4s</th><th>5s</th><th>6s</th><th><math><semantics><mrow><mi>Avg</mi><mo>.</mo></mrow><annotation>\operatorname{Avg.}</annotation></semantics></math></th><th>1s</th><th>2s</th><th>3s</th><th>4s</th><th>5s</th><th>6s</th><th><math><semantics><mrow><mi>Avg</mi><mo>.</mo></mrow><annotation>\operatorname{Avg.}</annotation></semantics></math></th></tr></thead><tbody><tr><th><math><semantics><mi>UniAD</mi> <annotation>\operatorname{UniAD}</annotation></semantics></math> <sup><a href="#fn:14">14</a></sup></th><td>CVPR’23</td><td>0.47</td><td>0.91</td><td>1.35</td><td>1.91</td><td>2.47</td><td>3.07</td><td>1.70</td><td>0.25</td><td>0.36</td><td>0.61</td><td>0.99</td><td>1.64</td><td>2.51</td><td>1.06</td><td>1.8</td></tr><tr><th><math><semantics><mi>SparseDrive</mi> <annotation>\operatorname{SparseDrive}</annotation></semantics></math> <sup><a href="#fn:40">40</a></sup></th><td>ICRA’25</td><td>0.43</td><td>0.87</td><td>1.23</td><td>1.75</td><td>2.32</td><td>2.95</td><td>1.59</td><td>0.19</td><td>0.31</td><td>0.56</td><td>0.87</td><td>1.54</td><td>2.33</td><td>0.97</td><td>9.0</td></tr><tr><th><math><semantics><mi>MomAD</mi> <annotation>\operatorname{MomAD}</annotation></semantics></math> <sup><a href="#fn:35">35</a></sup></th><td>CVPR’25</td><td>0.41</td><td>0.85</td><td>1.13</td><td>1.67</td><td>1.98</td><td>2.45</td><td>1.42</td><td>0.17</td><td>0.30</td><td>0.54</td><td>0.83</td><td>1.43</td><td>2.13</td><td>0.90</td><td>7.8</td></tr><tr><th><math><semantics><mi>LAW</mi> <annotation>\operatorname{LAW}</annotation></semantics></math> <sup><a href="#fn:21">21</a></sup></th><td>ICLR’25</td><td>0.40</td><td>0.87</td><td>1.16</td><td>1.71</td><td>2.03</td><td>2.61</td><td>1.46</td><td>0.19</td><td>0.33</td><td>0.57</td><td>0.86</td><td>1.51</td><td>2.31</td><td>0.96</td><td>19.5</td></tr><tr><th><math><semantics><mi>Epona</mi> <annotation>\operatorname{Epona}</annotation></semantics></math> <sup><a href="#fn:53">53</a></sup></th><td>ICCV’25</td><td>0.39</td><td>0.91</td><td>1.17</td><td>1.73</td><td>2.02</td><td>2.75</td><td>1.50</td><td>0.14</td><td>0.18</td><td>0.45</td><td>0.74</td><td>1.48</td><td>2.23</td><td>0.87</td><td>–</td></tr><tr><th><math><semantics><mi>World4Drive</mi> <annotation>\operatorname{World4Drive}</annotation></semantics></math> <sup><a href="#fn:55">55</a></sup></th><td>ICCV’25</td><td>0.42</td><td>0.92</td><td>1.21</td><td>1.75</td><td>2.06</td><td>2.79</td><td>1.53</td><td>0.16</td><td>0.20</td><td>0.47</td><td>0.76</td><td>1.50</td><td>2.14</td><td>0.87</td><td>–</td></tr><tr><th><math><semantics><mi>DIVER</mi> <annotation>\operatorname{DIVER}</annotation></semantics></math> <sup><a href="#fn:37">37</a></sup></th><td>TPAMI’26</td><td>0.38</td><td>0.75</td><td>1.10</td><td>1.53</td><td>1.98</td><td>2.49</td><td>1.37</td><td>0.13</td><td>0.31</td><td>0.44</td><td>0.80</td><td>1.41</td><td>2.11</td><td>0.87</td><td>6.6</td></tr><tr><th><math><semantics><mi>GuideFlow</mi> <annotation>\operatorname{GuideFlow}</annotation></semantics></math> <sup><a href="#fn:30">30</a></sup></th><td>CVPR’26</td><td>0.42</td><td>0.83</td><td>1.21</td><td>1.73</td><td>2.05</td><td>2.63</td><td>1.48</td><td>0.12</td><td>0.22</td><td>0.42</td><td>0.79</td><td>1.44</td><td>2.15</td><td>0.86</td><td>3.6</td></tr><tr><th><math><semantics><mrow><mi>MomWorld</mi> <mo>⁡</mo> <mrow><mo>(</mo><mi>Ours</mi><mo>)</mo></mrow></mrow> <annotation>\operatorname{MomWorld\ (Ours)}</annotation></semantics></math></th><td>–</td><td>0.27</td><td>0.52</td><td>0.86</td><td>1.28</td><td>1.76</td><td>2.31</td><td>1.17</td><td>0.02</td><td>0.17</td><td>0.42</td><td>0.83</td><td>1.36</td><td>1.97</td><td>0.79</td><td>7.2</td></tr></tbody></table>

Six-second nuScenes planning. Table 1 shows that MomWorld achieves the lowest L2 error at all horizons, reducing the best prior average from 1.37 m to 1.17 m. It also obtains the lowest average collision rate of 0.79%, an 8.1% improvement over the previous best. Overall, MomWorld provides the best long-horizon planning accuracy and safety.

Table 2: Trajectory Prediction Consistency at 4–6 s on the nuScenes validation set.

<table><thead><tr><th rowspan="2"><math><semantics><mi>Method</mi> <annotation>\operatorname{Method}</annotation></semantics></math></th><th rowspan="2"><math><semantics><mi>Venue</mi> <annotation>\operatorname{Venue}</annotation></semantics></math></th><th colspan="4"><math><semantics><mrow><mrow><mi>TPC</mi> <mo>⁡</mo> <mrow><mo>(</mo><mi>m</mi><mo>)</mo></mrow></mrow> <mo>↓</mo></mrow> <annotation>\operatorname{TPC\ (m)}\downarrow</annotation></semantics></math></th></tr><tr><th>4s</th><th>5s</th><th>6s</th><th><math><semantics><mrow><mi>Avg</mi><mo>.</mo></mrow><annotation>\operatorname{Avg.}</annotation></semantics></math></th></tr></thead><tbody><tr><th><math><semantics><mi>UniAD</mi> <annotation>\operatorname{UniAD}</annotation></semantics></math> <sup><a href="#fn:14">14</a></sup></th><td>CVPR’23</td><td>1.49</td><td>1.81</td><td>2.41</td><td>1.90</td></tr><tr><th><math><semantics><mi>VAD</mi> <annotation>\operatorname{VAD}</annotation></semantics></math> <sup><a href="#fn:19">19</a></sup></th><td>ICCV’23</td><td>1.55</td><td>1.73</td><td>2.17</td><td>1.82</td></tr><tr><th><math><semantics><mi>SparseDrive</mi> <annotation>\operatorname{SparseDrive}</annotation></semantics></math> <sup><a href="#fn:40">40</a></sup></th><td>ICRA’25</td><td>1.33</td><td>1.66</td><td>1.99</td><td>1.66</td></tr><tr><th><math><semantics><mi>MomAD</mi> <annotation>\operatorname{MomAD}</annotation></semantics></math> <sup><a href="#fn:35">35</a></sup></th><td>CVPR’25</td><td>1.19</td><td>1.45</td><td>1.61</td><td>1.42</td></tr><tr><th><math><semantics><mrow><mi>MomWorld</mi> <mo>⁡</mo> <mrow><mo>(</mo><mi>Ours</mi><mo>)</mo></mrow></mrow> <annotation>\operatorname{MomWorld\ (Ours)}</annotation></semantics></math></th><td>–</td><td>0.93</td><td>1.19</td><td>1.46</td><td>1.19</td></tr></tbody></table>

Long-Horizon Trajectory Prediction Consistency. Table 2 shows that MomWorld achieves the lowest TPC at all horizons, reducing the best prior average from 1.42 m to 1.19 m. Overall, MomWorld provides the best long-horizon trajectory consistency.

NAVSIM v1 navtest. Table 5 shows that MomWorld achieves the best PDMS among world-model methods at 90.2, outperforming DriveLaW by 1.1 points. It also obtains the highest DAC and EP in this category, demonstrating strong planning quality under NAVSIM’s non-reactive pseudo-simulation protocol.

NAVSIM v2 navtest. Table 5 compares MomWorld with TransFuser [^7], recent end-to-end planners [^27] [^20] [^50] [^58], VLA methods [^31] [^23], and Latent-WAM [^42]. MomWorld achieves the best EPDMS of 90.1, surpassing Latent-WAM by 0.8 points, while attaining the highest TTC and joint-best EP. These results demonstrate that MomWorld achieves a stronger balance between safety, progress, and overall planning quality under the diagnostic pseudo-simulation setting.

NAVSIM v2 navhard. Table 5 compares MomWorld with end-to-end planners [^7] [^27] [^30] [^11] [^50], VLA methods [^8] [^57], and world-model methods [^38] [^55]. MomWorld achieves the best EPDMS of 42.8 and the highest DDC in both stages, demonstrating stronger robustness under perturbed future observations.

Table 3: Planning performance on the NAVSIM v1 navtest split. ‘mmt’ denotes the multimodal TransFuser variant, and <sup>∗</sup> marks our re-implementation.

<table><tbody><tr><td>Method</td><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Comf.<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td colspan="7">E2E-based Methods</td></tr><tr><td>VADv2 <sup><a href="#fn:3">3</a></sup></td><td>97.2</td><td>89.1</td><td>91.6</td><td>100</td><td>76.0</td><td>80.9</td></tr><tr><td>TransFuser <sup><a href="#fn:7">7</a></sup></td><td>97.7</td><td>92.8</td><td>92.8</td><td>100</td><td>79.2</td><td>84.0</td></tr><tr><td><math><semantics><mmultiscripts><mi>TransFuser</mi> <mi>mmt</mi> <mo>∗</mo></mmultiscripts> <annotation>{\operatorname{TransFuser}_{\operatorname{mmt}}}^{*}</annotation></semantics></math> <sup><a href="#fn:7">7</a></sup></td><td>96.2</td><td>95.4</td><td>90.7</td><td>100</td><td>80.7</td><td>85.1</td></tr><tr><td>UniAD <sup><a href="#fn:14">14</a></sup></td><td>97.8</td><td>91.9</td><td>92.9</td><td>100</td><td>78.8</td><td>83.4</td></tr><tr><td>PARA-Drive <sup><a href="#fn:43">43</a></sup></td><td>97.9</td><td>92.4</td><td>93.0</td><td>99.8</td><td>79.3</td><td>84.0</td></tr><tr><td>DRAMA <sup><a href="#fn:51">51</a></sup></td><td>98.0</td><td>93.1</td><td>94.8</td><td>100</td><td>80.1</td><td>85.5</td></tr><tr><td>Hydra-MDP <sup><a href="#fn:24">24</a></sup></td><td>98.3</td><td>96.0</td><td>94.6</td><td>100</td><td>78.7</td><td>86.5</td></tr><tr><td>FUMP <sup><a href="#fn:29">29</a></sup></td><td>98.1</td><td>96.2</td><td>94.2</td><td>100</td><td>82.0</td><td>87.8</td></tr><tr><td>DiffusionDrive <sup><a href="#fn:27">27</a></sup></td><td>98.2</td><td>96.2</td><td>94.7</td><td>100</td><td>82.2</td><td>88.1</td></tr><tr><td>DIVER <sup><a href="#fn:37">37</a></sup></td><td>98.5</td><td>96.5</td><td>94.9</td><td>100</td><td>82.6</td><td>88.3</td></tr><tr><td>DriveSuprim <sup><a href="#fn:50">50</a></sup></td><td>97.8</td><td>97.3</td><td>93.6</td><td>100</td><td>86.7</td><td>89.9</td></tr><tr><td>GoalFlow <sup><a href="#fn:46">46</a></sup></td><td>98.4</td><td>98.3</td><td>94.6</td><td>100</td><td>85.0</td><td>90.3</td></tr><tr><td>ReCogDrive-IL <sup><a href="#fn:23">23</a></sup></td><td>98.1</td><td>94.7</td><td>94.2</td><td>100</td><td>80.9</td><td>86.5</td></tr><tr><td colspan="7">VLA-based Methods</td></tr><tr><td>AutoVLA <sup><a href="#fn:56">56</a></sup></td><td>98.4</td><td>95.6</td><td>98.0</td><td>99.9</td><td>81.9</td><td>89.1</td></tr><tr><td>ReCogdrive <sup><a href="#fn:23">23</a></sup></td><td>98.2</td><td>97.8</td><td>95.2</td><td>99.8</td><td>83.5</td><td>89.6</td></tr><tr><td>DriveWorld-VLA <sup><a href="#fn:31">31</a></sup></td><td>99.1</td><td>98.2</td><td>96.1</td><td>100</td><td>85.9</td><td>91.3</td></tr><tr><td colspan="7">World-Model-based Methods</td></tr><tr><td>DrivingGPT <sup><a href="#fn:4">4</a></sup></td><td>98.9</td><td>90.7</td><td>94.9</td><td>95.6</td><td>79.7</td><td>82.4</td></tr><tr><td>Resim <sup><a href="#fn:48">48</a></sup></td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>86.6</td></tr><tr><td>PWM <sup><a href="#fn:12">12</a></sup></td><td>98.6</td><td>95.9</td><td>95.4</td><td>100</td><td>81.8</td><td>88.1</td></tr><tr><td>LAW <sup><a href="#fn:21">21</a></sup></td><td>96.4</td><td>95.4</td><td>88.7</td><td>99.9</td><td>81.7</td><td>84.6</td></tr><tr><td>World4Drive <sup><a href="#fn:55">55</a></sup></td><td>97.4</td><td>94.3</td><td>92.8</td><td>100</td><td>79.9</td><td>85.1</td></tr><tr><td>Epona <sup><a href="#fn:53">53</a></sup></td><td>97.9</td><td>95.1</td><td>93.8</td><td>99.9</td><td>80.4</td><td>86.2</td></tr><tr><td>WoTE <sup><a href="#fn:22">22</a></sup></td><td>98.5</td><td>96.8</td><td>94.9</td><td>99.9</td><td>81.9</td><td>88.3</td></tr><tr><td>WorldRFT <sup><a href="#fn:49">49</a></sup></td><td>97.8</td><td>96.8</td><td>94.0</td><td>100</td><td>81.7</td><td>87.8</td></tr><tr><td>DriveLaW <sup><a href="#fn:45">45</a></sup></td><td>99.0</td><td>97.1</td><td>96.7</td><td>100</td><td>81.3</td><td>89.1</td></tr><tr><td>DriveX-S <sup><a href="#fn:34">34</a></sup></td><td>97.5</td><td>94.0</td><td>93.0</td><td>100</td><td>79.7</td><td>84.5</td></tr><tr><td><math><semantics><mrow><mi>MomWorld</mi> <mo>⁡</mo> <mrow><mo>(</mo><mi>Ours</mi><mo>)</mo></mrow></mrow> <annotation>\operatorname{MomWorld\ (Ours)}</annotation></semantics></math></td><td>98.6</td><td>98.2</td><td>93.9</td><td>100</td><td>85.7</td><td>90.2</td></tr></tbody></table>

Table 4: Results on the NAVSIM v2 navtest split.

<table><thead><tr><th>Method</th><th>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>DDC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>TLC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>LK <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>HC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>EC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>EPDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th></tr><tr><th colspan="11">E2E-based Methods</th></tr></thead><tbody><tr><th>TransFuser</th><td>96.9</td><td>89.9</td><td>97.8</td><td>99.7</td><td>87.1</td><td>95.4</td><td>92.7</td><td>98.3</td><td>87.2</td><td>76.7</td></tr><tr><th>DiffusionDrive</th><td>98.2</td><td>95.9</td><td>99.4</td><td>99.8</td><td>87.5</td><td>97.3</td><td>96.8</td><td>98.3</td><td>87.7</td><td>84.5</td></tr><tr><th>Hydra-MDP++</th><td>97.2</td><td>97.5</td><td>99.4</td><td>99.6</td><td>83.1</td><td>96.5</td><td>94.4</td><td>98.2</td><td>70.9</td><td>81.4</td></tr><tr><th>DriveSuprim</th><td>97.5</td><td>96.5</td><td>99.4</td><td>99.6</td><td>88.4</td><td>96.6</td><td>95.5</td><td>98.3</td><td>77.0</td><td>83.1</td></tr><tr><th>DiffusionDriveV2</th><td>97.7</td><td>96.6</td><td>99.2</td><td>99.8</td><td>88.9</td><td>97.2</td><td>96.0</td><td>97.8</td><td>91.0</td><td>85.5</td></tr><tr><th colspan="11">VLA-based Methods</th></tr><tr><th>DriveWorld-VLA</th><td>98.6</td><td>99.1</td><td>99.6</td><td>99.8</td><td>87.4</td><td>97.9</td><td>97.0</td><td>97.8</td><td>78.6</td><td>86.8</td></tr><tr><th>ReCogdrive</th><td>98.3</td><td>95.2</td><td>98.3</td><td>99.8</td><td>87.1</td><td>97.5</td><td>96.6</td><td>99.5</td><td>86.5</td><td>83.6</td></tr><tr><th colspan="11">World-Model-based Methods</th></tr><tr><th>Latent-WAM</th><td>98.1</td><td>97.3</td><td>99.6</td><td>99.8</td><td>87.7</td><td>97.3</td><td>97.6</td><td>98.1</td><td>87.3</td><td>89.3</td></tr><tr><th>MomWorld (Ours)</th><td>98.1</td><td>98.1</td><td>99.5</td><td>99.8</td><td>88.9</td><td>98.2</td><td>96.1</td><td>98.3</td><td>90.6</td><td>90.1</td></tr></tbody></table>

Table 5: Results on the NAVSIM v2 navhard split. <sup>†</sup> computed from reported metrics.

<table><tbody><tr><th>Method</th><td>Stage</td><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DDC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TLC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>LK <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>HC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EPDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><th colspan="12">E2E-based Methods</th></tr><tr><th rowspan="2">TransFuser</th><td>Stage 1</td><td>96.2</td><td>79.5</td><td>99.1</td><td>99.5</td><td>84.1</td><td>95.1</td><td>94.2</td><td>97.5</td><td>79.1</td><td rowspan="2">23.1</td></tr><tr><td>Stage 2</td><td>77.7</td><td>70.2</td><td>84.2</td><td>98.0</td><td>85.1</td><td>75.6</td><td>45.4</td><td>95.7</td><td>75.9</td></tr><tr><th rowspan="2">DiffusionDrive</th><td>Stage 1</td><td>96.0</td><td>79.7</td><td>97.4</td><td>99.5</td><td>81.3</td><td>93.1</td><td>90.8</td><td>96.8</td><td>73.8</td><td rowspan="2">24.2</td></tr><tr><td>Stage 2</td><td>82.1</td><td>72.2</td><td>88.5</td><td>98.7</td><td>85.1</td><td>78.8</td><td>49.2</td><td>89.3</td><td>71.2</td></tr><tr><th rowspan="2">GuideFlow</th><td>Stage 1</td><td>96.6</td><td>80.5</td><td>96.3</td><td>99.3</td><td>82.3</td><td>94.9</td><td>91.5</td><td>97.7</td><td>67.8</td><td rowspan="2">27.1</td></tr><tr><td>Stage 2</td><td>87.3</td><td>76.7</td><td>88.8</td><td>99.2</td><td>84.3</td><td>85.1</td><td>49.7</td><td>93.1</td><td>44.5</td></tr><tr><th rowspan="2">RAP-DINO</th><td>Stage 1</td><td>97.1</td><td>94.4</td><td>98.8</td><td>99.8</td><td>83.9</td><td>96.9</td><td>94.7</td><td>96.4</td><td>66.2</td><td rowspan="2">36.9</td></tr><tr><td>Stage 2</td><td>83.2</td><td>83.9</td><td>87.4</td><td>98.0</td><td>86.9</td><td>80.4</td><td>52.3</td><td>95.2</td><td>52.4</td></tr><tr><th rowspan="2">DriveSuprim</th><td>Stage 1</td><td>98.9</td><td>95.1</td><td>99.2</td><td>99.6</td><td>76.1</td><td>99.1</td><td>94.7</td><td>97.6</td><td>54.2</td><td rowspan="2">42.1</td></tr><tr><td>Stage 2</td><td>87.9</td><td>88.8</td><td>89.6</td><td>98.8</td><td>80.3</td><td>86.0</td><td>53.5</td><td>97.1</td><td>56.1</td></tr><tr><th colspan="12">VLA-based Methods</th></tr><tr><th rowspan="2">DriveFine</th><td>Stage 1</td><td>97.6</td><td>90.0</td><td>99.1</td><td>99.3</td><td>84.9</td><td>96.7</td><td>97.3</td><td>97.6</td><td>72.0</td><td rowspan="2">30.5 <sup>†</sup></td></tr><tr><td>Stage 2</td><td>82.1</td><td>71.3</td><td>84.8</td><td>98.4</td><td>88.1</td><td>74.3</td><td>47.2</td><td>96.8</td><td>72.8</td></tr><tr><th rowspan="2">SpanVLA</th><td>Stage 1</td><td>98.4</td><td>94.3</td><td>97.8</td><td>99.9</td><td>85.7</td><td>97.2</td><td>94.2</td><td>97.6</td><td>72.1</td><td rowspan="2">40.1</td></tr><tr><td>Stage 2</td><td>86.9</td><td>84.3</td><td>87.1</td><td>98.2</td><td>85.5</td><td>82.7</td><td>62.3</td><td>96.8</td><td>67.4</td></tr><tr><th colspan="12">World-Model-based Methods</th></tr><tr><th rowspan="2">MindDrive</th><td>Stage 1</td><td>96.1</td><td>86.0</td><td>98.8</td><td>99.3</td><td>83.3</td><td>95.6</td><td>94.4</td><td>97.6</td><td>74.7</td><td rowspan="2">30.9</td></tr><tr><td>Stage 2</td><td>82.6</td><td>79.1</td><td>86.4</td><td>98.0</td><td>85.3</td><td>79.4</td><td>49.2</td><td>96.5</td><td>71.0</td></tr><tr><th rowspan="2">World4Drive</th><td>Stage 1</td><td>97.3</td><td>89.1</td><td>97.6</td><td>99.7</td><td>60.5</td><td>96.8</td><td>87.7</td><td>93.1</td><td>60.0</td><td rowspan="2">34.9</td></tr><tr><td>Stage 2</td><td>91.4</td><td>82.0</td><td>91.0</td><td>98.5</td><td>53.1</td><td>90.6</td><td>52.3</td><td>93.3</td><td>62.8</td></tr><tr><th></th><td>Stage 1</td><td>96.9</td><td>93.6</td><td>99.8</td><td>99.8</td><td>80.4</td><td>96.9</td><td>96.4</td><td>97.6</td><td>60.0</td><td></td></tr><tr><th>MomWorld (Ours)</th><td>Stage 2</td><td>86.4</td><td>88.2</td><td>94.2</td><td>98.3</td><td>81.7</td><td>84.4</td><td>55.3</td><td>97.0</td><td>54.7</td><td>42.8</td></tr></tbody></table>

### 4.3 Ablation Studies

Roles of Different Components in MomWorld. Table 6 shows consistent gains as each component is added. MoLWM progressively improves trajectory accuracy and navhard performance, while MGRF and HARF further reduce long-horizon errors and collisions. The complete MomWorld achieves the best results across all metrics, confirming the complementarity of latent momentum modeling and flow refinement.

Ablation on Different Designs in MoLWM. Table 7 shows that removing any MoLWM design degrades performance, with the momentum proposal and reset gate causing the largest drops. The full MoLWM achieves the best results on all metrics, enabling MomWorld to provide more accurate, consistent, and safe planning.

Ablation on Different Designs in MoFlow. Table 8 studies the conditioning signals, Euler steps, residual fusion designs, and inference latency of MoFlow. We first compare unconditioned refinement with single- and dual-signal conditioning, showing that jointly using $p_{0}$ and $M_{t}$ provides more effective trajectory guidance. We then vary the number of Euler steps, where four steps achieve the best accuracy–efficiency trade-off. Finally, removing residual clipping, horizon-aware weighting, or the learned global gate consistently degrades performance, while direct trajectory replacement performs worst. Overall, the full MoFlow achieves the best results with only 8.4 ms additional latency, enabling MomWorld to deliver more accurate, safe, and efficient planning.

Table 6: Roles of Different Method Components in MomWorld. Cumulative component study on the nuScenes validation set and the NAVSIM v2 navhard split. FWM, LWR, SMD, MGRF, and HARF denote *Future World Memory*, *Latent World Rollout*, *Scene-Adaptive Momentum Dynamics*, *Momentum-Guided Residual Flow*, and *Horizon-Aware Residual Fusion*, respectively. LK, EP, and EC are the Stage-2 metrics reported by the navhard protocol.

<table><thead><tr><th colspan="3"><math><semantics><mi>MoLWM</mi> <annotation>\operatorname{MoLWM}</annotation></semantics></math></th><th colspan="2"><math><semantics><mi>MoFlow</mi> <annotation>\operatorname{MoFlow}</annotation></semantics></math></th><th colspan="5"><math><semantics><mrow><mi>nuScenes</mi> <mo></mo><mn>6</mn> <mo></mo><mi>s</mi></mrow> <annotation>\operatorname{nuScenes\ 6s}</annotation></semantics></math></th><th colspan="4"><math><semantics><mrow><mi>NAVSIM</mi> <mo></mo><mi>v2</mi> <mo></mo><mi>navhard</mi></mrow> <annotation>\operatorname{NAVSIM\ v2\ navhard}</annotation></semantics></math></th></tr><tr><th>FWM</th><th>LWR</th><th>SMD</th><th>MGRF</th><th>HARF</th><th><math><semantics><mrow><mrow><msub><mi>L</mi> <mn>2</mn></msub> <mo></mo><mi>@</mi> <mo></mo><mn>3</mn> <mo></mo><mrow><mo>(</mo><mi>m</mi><mo>)</mo></mrow></mrow> <mo>↓</mo></mrow> <annotation>L_{2}@3\,(\mathrm{m})\downarrow</annotation></semantics></math></th><th><math><semantics><mrow><mrow><msub><mi>L</mi> <mn>2</mn></msub> <mo></mo><mi>@</mi> <mo></mo><mn>6</mn> <mo></mo><mrow><mo>(</mo><mi>m</mi><mo>)</mo></mrow></mrow> <mo>↓</mo></mrow> <annotation>L_{2}@6\,(\mathrm{m})\downarrow</annotation></semantics></math></th><th><math><semantics><mrow><mrow><mrow><mi>TPC</mi> <mo></mo><mi>@</mi> <mo></mo><mn>6</mn></mrow> <mo>⁡</mo> <mrow><mo>(</mo><mi>m</mi><mo>)</mo></mrow></mrow> <mo>↓</mo></mrow> <annotation>\operatorname{TPC@6}\,(\mathrm{m})\downarrow</annotation></semantics></math></th><th><math><semantics><mrow><mrow><mi>Col</mi><mo>.</mo><mrow><mi>@</mi> <mo></mo><mn>3</mn></mrow></mrow> <mrow><mo>(</mo><mo>%</mo><mo>)</mo></mrow> <mo>↓</mo></mrow> <annotation>\operatorname{Col.@3}\,(\%)\downarrow</annotation></semantics></math></th><th><math><semantics><mrow><mrow><mi>Col</mi><mo>.</mo><mrow><mi>@</mi> <mo></mo><mn>6</mn></mrow></mrow> <mrow><mo>(</mo><mo>%</mo><mo>)</mo></mrow> <mo>↓</mo></mrow> <annotation>\operatorname{Col.@6}\,(\%)\downarrow</annotation></semantics></math></th><th><math><semantics><mrow><mi>LK</mi> <mo>↑</mo></mrow> <annotation>\operatorname{LK}\uparrow</annotation></semantics></math></th><th><math><semantics><mrow><mi>EP</mi> <mo>↑</mo></mrow> <annotation>\operatorname{EP}\uparrow</annotation></semantics></math></th><th><math><semantics><mrow><mi>EC</mi> <mo>↑</mo></mrow> <annotation>\operatorname{EC}\uparrow</annotation></semantics></math></th><th><math><semantics><mrow><mi>EPDMS</mi> <mo>↑</mo></mrow> <annotation>\operatorname{EPDMS}\uparrow</annotation></semantics></math></th></tr></thead><tbody><tr><td></td><td></td><td></td><td></td><td></td><td>1.13</td><td>2.45</td><td>1.61</td><td>0.54</td><td>2.13</td><td>54.6</td><td>69.5</td><td>49.7</td><td>41.7</td></tr><tr><td>✓</td><td></td><td></td><td></td><td></td><td>1.07</td><td>2.42</td><td>1.57</td><td>0.51</td><td>2.09</td><td>54.8</td><td>72.3</td><td>50.8</td><td>42.0</td></tr><tr><td>✓</td><td>✓</td><td></td><td></td><td></td><td>1.01</td><td>2.39</td><td>1.54</td><td>0.48</td><td>2.06</td><td>55.0</td><td>75.6</td><td>52.1</td><td>42.2</td></tr><tr><td>✓</td><td>✓</td><td>✓</td><td></td><td></td><td>0.96</td><td>2.36</td><td>1.51</td><td>0.46</td><td>2.03</td><td>55.1</td><td>78.4</td><td>53.0</td><td>42.4</td></tr><tr><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td></td><td>0.91</td><td>2.33</td><td>1.48</td><td>0.44</td><td>2.00</td><td>55.2</td><td>80.2</td><td>54.0</td><td>42.6</td></tr><tr><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>0.86</td><td>2.31</td><td>1.46</td><td>0.42</td><td>1.97</td><td>55.3</td><td>81.7</td><td>54.7</td><td>42.8</td></tr></tbody></table>

Table 7: Ablation study of different designs in MoLWM across nuScenes, NAVSIM v1 navtest, and NAVSIM v2 navhard.

<table><thead><tr><th rowspan="2"><math><semantics><mi>Variant</mi> <annotation>\operatorname{Variant}</annotation></semantics></math></th><th colspan="5"><math><semantics><mrow><mi>nuScenes</mi> <mo></mo><mn>6</mn> <mo></mo><mi>s</mi></mrow> <annotation>\operatorname{nuScenes\ 6s}</annotation></semantics></math></th><th><math><semantics><mrow><mi>NAVSIM</mi> <mo></mo><mi>v1</mi> <mo></mo><mi>navtest</mi></mrow> <annotation>\operatorname{NAVSIM\ v1\ navtest}</annotation></semantics></math></th><th><math><semantics><mrow><mi>NAVSIM</mi> <mo></mo><mi>v2</mi> <mo></mo><mi>navhard</mi></mrow> <annotation>\operatorname{NAVSIM\ v2\ navhard}</annotation></semantics></math></th></tr><tr><th><math><semantics><mrow><mrow><msub><mi>L</mi> <mn>2</mn></msub> <mo></mo><mi>@</mi> <mo></mo><mn>3</mn> <mo></mo><mrow><mo>(</mo><mi>m</mi><mo>)</mo></mrow></mrow> <mo>↓</mo></mrow> <annotation>L_{2}@3\,(\mathrm{m})\downarrow</annotation></semantics></math></th><th><math><semantics><mrow><mrow><msub><mi>L</mi> <mn>2</mn></msub> <mo></mo><mi>@</mi> <mo></mo><mn>6</mn> <mo></mo><mrow><mo>(</mo><mi>m</mi><mo>)</mo></mrow></mrow> <mo>↓</mo></mrow> <annotation>L_{2}@6\,(\mathrm{m})\downarrow</annotation></semantics></math></th><th><math><semantics><mrow><mrow><mrow><mi>TPC</mi> <mo></mo><mi>@</mi> <mo></mo><mn>6</mn></mrow> <mo>⁡</mo> <mrow><mo>(</mo><mi>m</mi><mo>)</mo></mrow></mrow> <mo>↓</mo></mrow> <annotation>\operatorname{TPC@6}\,(\mathrm{m})\downarrow</annotation></semantics></math></th><th><math><semantics><mrow><mrow><mi>Col</mi><mo>.</mo><mrow><mi>@</mi> <mo></mo><mn>3</mn></mrow></mrow> <mrow><mo>(</mo><mo>%</mo><mo>)</mo></mrow> <mo>↓</mo></mrow> <annotation>\operatorname{Col.@3}\,(\%)\downarrow</annotation></semantics></math></th><th><math><semantics><mrow><mrow><mi>Col</mi><mo>.</mo><mrow><mi>@</mi> <mo></mo><mn>6</mn></mrow></mrow> <mrow><mo>(</mo><mo>%</mo><mo>)</mo></mrow> <mo>↓</mo></mrow> <annotation>\operatorname{Col.@6}\,(\%)\downarrow</annotation></semantics></math></th><th><math><semantics><mrow><mi>PDMS</mi> <mo>↑</mo></mrow> <annotation>\operatorname{PDMS}\uparrow</annotation></semantics></math></th><th><math><semantics><mrow><mi>EPDMS</mi> <mo>↑</mo></mrow> <annotation>\operatorname{EPDMS}\uparrow</annotation></semantics></math></th></tr></thead><tbody><tr><td>No historical variation</td><td>0.95</td><td>2.58</td><td>1.72</td><td>0.55</td><td>2.40</td><td>88.6</td><td>39.7</td></tr><tr><td>No horizon embedding in SMD</td><td>0.90</td><td>2.43</td><td>1.57</td><td>0.47</td><td>2.12</td><td>89.5</td><td>41.3</td></tr><tr><td>Fixed retention (<math><semantics><mrow><msub><mi>ρ</mi> <mi>k</mi></msub> <mo>≡</mo> <mn>0.9</mn></mrow> <annotation>\rho_{k}\equiv 0.9</annotation></semantics></math>)</td><td>0.92</td><td>2.49</td><td>1.63</td><td>0.50</td><td>2.21</td><td>89.0</td><td>40.8</td></tr><tr><td>No momentum proposal (<math><semantics><mrow><msub><mi>u</mi> <mi>k</mi></msub> <mo>≡</mo> <mn>0</mn></mrow> <annotation>u_{k}\equiv 0</annotation></semantics></math>)</td><td>0.99</td><td>2.69</td><td>1.82</td><td>0.60</td><td>2.58</td><td>87.8</td><td>38.6</td></tr><tr><td>No reset gate (<math><semantics><mrow><msub><mi>g</mi> <mi>k</mi></msub> <mo>≡</mo> <mn>0</mn></mrow> <annotation>g_{k}\equiv 0</annotation></semantics></math>)</td><td>0.94</td><td>2.55</td><td>1.69</td><td>0.65</td><td>2.83</td><td>88.2</td><td>37.9</td></tr><tr><td>No horizon embedding in FWM</td><td>0.91</td><td>2.46</td><td>1.60</td><td>0.49</td><td>2.17</td><td>89.3</td><td>40.9</td></tr><tr><td>No momentum injection in FWM</td><td>0.97</td><td>2.64</td><td>1.78</td><td>0.58</td><td>2.49</td><td>88.0</td><td>38.9</td></tr><tr><td>No temporal aggregation in FWM</td><td>0.93</td><td>2.52</td><td>1.66</td><td>0.52</td><td>2.29</td><td>88.8</td><td>40.2</td></tr><tr><td>Full MoLWM</td><td>0.86</td><td>2.31</td><td>1.46</td><td>0.42</td><td>1.97</td><td>90.2</td><td>42.8</td></tr></tbody></table>

Table 8: Ablation study of different designs in MoFlow across nuScenes, NAVSIM v1 navtest, and NAVSIM v2 navhard.

<table><tbody><tr><th rowspan="2"><math><semantics><mrow><mi>MoFlow</mi> <mo></mo><mi>Design</mi></mrow> <annotation>\operatorname{MoFlow\ Design}</annotation></semantics></math></th><th rowspan="2"><math><semantics><mi>Steps</mi> <annotation>\operatorname{Steps}</annotation></semantics></math></th><td colspan="5"><math><semantics><mrow><mi>nuScenes</mi> <mo></mo><mn>6</mn> <mo></mo><mi>s</mi></mrow> <annotation>\operatorname{nuScenes\ 6s}</annotation></semantics></math></td><td><math><semantics><mrow><mi>NAVSIM</mi> <mo></mo><mi>v1</mi> <mo></mo><mi>navtest</mi></mrow> <annotation>\operatorname{NAVSIM\ v1\ navtest}</annotation></semantics></math></td><td><math><semantics><mrow><mi>NAVSIM</mi> <mo></mo><mi>v2</mi> <mo></mo><mi>navhard</mi></mrow> <annotation>\operatorname{NAVSIM\ v2\ navhard}</annotation></semantics></math></td><td colspan="2"><math><semantics><mrow><mrow><mi>Latency</mi> <mo>⁡</mo> <mrow><mo>(</mo><mi>ms</mi><mo>)</mo></mrow></mrow> <mo>↓</mo></mrow> <annotation>\operatorname{Latency\ (ms)}\downarrow</annotation></semantics></math></td></tr><tr><td><math><semantics><mrow><mrow><msub><mi>L</mi> <mn>2</mn></msub> <mo></mo><mi>@</mi> <mo></mo><mn>3</mn> <mo></mo><mrow><mo>(</mo><mi>m</mi><mo>)</mo></mrow></mrow> <mo>↓</mo></mrow> <annotation>L_{2}@3\,(\mathrm{m})\downarrow</annotation></semantics></math></td><td><math><semantics><mrow><mrow><msub><mi>L</mi> <mn>2</mn></msub> <mo></mo><mi>@</mi> <mo></mo><mn>6</mn> <mo></mo><mrow><mo>(</mo><mi>m</mi><mo>)</mo></mrow></mrow> <mo>↓</mo></mrow> <annotation>L_{2}@6\,(\mathrm{m})\downarrow</annotation></semantics></math></td><td><math><semantics><mrow><mrow><mrow><mi>TPC</mi> <mo></mo><mi>@</mi> <mo></mo><mn>6</mn></mrow> <mo>⁡</mo> <mrow><mo>(</mo><mi>m</mi><mo>)</mo></mrow></mrow> <mo>↓</mo></mrow> <annotation>\operatorname{TPC@6}\,(\mathrm{m})\downarrow</annotation></semantics></math></td><td><math><semantics><mrow><mrow><mi>Col</mi><mo>.</mo><mrow><mi>@</mi> <mo></mo><mn>3</mn></mrow></mrow> <mrow><mo>(</mo><mo>%</mo><mo>)</mo></mrow> <mo>↓</mo></mrow> <annotation>\operatorname{Col.@3}\,(\%)\downarrow</annotation></semantics></math></td><td><math><semantics><mrow><mrow><mi>Col</mi><mo>.</mo><mrow><mi>@</mi> <mo></mo><mn>6</mn></mrow></mrow> <mrow><mo>(</mo><mo>%</mo><mo>)</mo></mrow> <mo>↓</mo></mrow> <annotation>\operatorname{Col.@6}\,(\%)\downarrow</annotation></semantics></math></td><td><math><semantics><mrow><mi>PDMS</mi> <mo>↑</mo></mrow> <annotation>\operatorname{PDMS}\uparrow</annotation></semantics></math></td><td><math><semantics><mrow><mi>EPDMS</mi> <mo>↑</mo></mrow> <annotation>\operatorname{EPDMS}\uparrow</annotation></semantics></math></td><td><math><semantics><mi>Flow</mi> <annotation>\operatorname{Flow}</annotation></semantics></math></td><td><math><semantics><mi>Total</mi> <annotation>\operatorname{Total}</annotation></semantics></math></td></tr><tr><th>No MoFlow (base plan <math><semantics><msup><mi>τ</mi> <mn>0</mn></msup> <annotation>\tau^{0}</annotation></semantics></math>)</th><th>0</th><td>0.96</td><td>2.36</td><td>1.51</td><td>0.46</td><td>2.03</td><td>89.7</td><td>42.4</td><td>0.0</td><td>130.5</td></tr><tr><th>Unconditioned MGRF</th><th>4</th><td>1.01</td><td>2.53</td><td>1.67</td><td>0.57</td><td>2.38</td><td>88.3</td><td>40.0</td><td>8.4</td><td>138.9</td></tr><tr><th><math><semantics><msub><mi>M</mi> <mi>t</mi></msub> <annotation>M_{t}</annotation></semantics></math> only</th><th>4</th><td>0.91</td><td>2.40</td><td>1.55</td><td>0.47</td><td>2.09</td><td>89.4</td><td>41.8</td><td>8.4</td><td>138.9</td></tr><tr><th><math><semantics><msub><mi>p</mi> <mn>0</mn></msub> <annotation>p_{0}</annotation></semantics></math> only</th><th>4</th><td>0.93</td><td>2.43</td><td>1.58</td><td>0.49</td><td>2.14</td><td>89.2</td><td>41.5</td><td>8.4</td><td>138.9</td></tr><tr><th><math><semantics><mrow><msub><mi>p</mi> <mn>0</mn></msub><mo>,</mo><msub><mi>M</mi> <mi>t</mi></msub></mrow> <annotation>p_{0},M_{t}</annotation></semantics></math></th><th>1</th><td>0.91</td><td>2.38</td><td>1.53</td><td>0.46</td><td>2.07</td><td>89.8</td><td>42.3</td><td>2.2</td><td>132.7</td></tr><tr><th><math><semantics><mrow><msub><mi>p</mi> <mn>0</mn></msub><mo>,</mo><msub><mi>M</mi> <mi>t</mi></msub></mrow> <annotation>p_{0},M_{t}</annotation></semantics></math></th><th>2</th><td>0.88</td><td>2.34</td><td>1.49</td><td>0.44</td><td>2.01</td><td>90.0</td><td>42.5</td><td>4.3</td><td>134.8</td></tr><tr><th><math><semantics><mrow><msub><mi>p</mi> <mn>0</mn></msub><mo>,</mo><msub><mi>M</mi> <mi>t</mi></msub></mrow> <annotation>p_{0},M_{t}</annotation></semantics></math></th><th>8</th><td>0.87</td><td>2.32</td><td>1.47</td><td>0.43</td><td>1.99</td><td>90.1</td><td>42.7</td><td>16.5</td><td>147.0</td></tr><tr><th>Direct replacement (<math><semantics><mrow><mover><mi>τ</mi> <mo>~</mo></mover> <mo>=</mo> <mover><mi>τ</mi> <mo>^</mo></mover></mrow> <annotation>\widetilde{\tau}=\widehat{\tau}</annotation></semantics></math>)</th><th>4</th><td>0.98</td><td>2.57</td><td>1.69</td><td>0.59</td><td>2.46</td><td>87.9</td><td>39.5</td><td>8.4</td><td>138.9</td></tr><tr><th>No residual clipping</th><th>4</th><td>0.92</td><td>2.45</td><td>1.59</td><td>0.55</td><td>2.37</td><td>88.7</td><td>40.5</td><td>8.4</td><td>138.9</td></tr><tr><th>No horizon factor</th><th>4</th><td>0.90</td><td>2.40</td><td>1.55</td><td>0.48</td><td>2.12</td><td>89.4</td><td>41.7</td><td>8.4</td><td>138.9</td></tr><tr><th>No learned global gate (<math><semantics><mrow><mrow><mi>σ</mi> <mo>⁡</mo> <mrow><mo>(</mo><mi>b</mi><mo>)</mo></mrow></mrow> <mo>→</mo> <mn>1</mn></mrow> <annotation>\sigma(b)\rightarrow 1</annotation></semantics></math>)</th><th>4</th><td>0.89</td><td>2.38</td><td>1.53</td><td>0.47</td><td>2.09</td><td>89.6</td><td>42.0</td><td>8.4</td><td>138.9</td></tr><tr><th>Full MoFlow</th><th>4</th><td>0.86</td><td>2.31</td><td>1.46</td><td>0.42</td><td>1.97</td><td>90.2</td><td>42.8</td><td>8.4</td><td>138.9</td></tr></tbody></table>

![[vis_nus3.png|Refer to caption]]

Figure 4: Consecutive planning visualization on nuScenes. Comparison of MomAD 35 and MomWorld across consecutive frames. MomWorld yields more accurate and temporally consistent future trajectories.

### 4.4 Visualization

Figure 4 presents consecutive planning results for MomAD [^35] and MomWorld. Across $t\!-\!1$, $t$, and $t\!+\!1$, MomWorld exhibits smoother trajectory evolution and closer alignment with the ground truth, demonstrating improved temporal stability and planning accuracy.

## 5 Conclusion

We presented MomWorld, a momentum-aware latent world model for temporally consistent long-horizon autonomous driving. MoLWM jointly propagates future latent scene states and momentum, using scene-adaptive retention, update, and reset mechanisms to preserve persistent dynamics while suppressing stale motion. The resulting Future World Memory supports candidate selection, while MoFlow refines the selected plan through momentum-conditioned residual flow and horizon-aware bounded fusion. Experiments on nuScenes, NAVSIM v1/v2, and Bench2Drive demonstrate improved planning accuracy, temporal consistency, safety, and planning quality across open-loop and pseudo-simulation protocols, confirming the complementarity of latent momentum modeling and controlled flow refinement.

Limitation and Future Work. One limitation of MomWorld is its reliance on a fixed candidate vocabulary and scoring stage. Future work will explore more flexible proposal generation while retaining stable momentum-conditioned refinement.

## 6 Appendix

This supplementary material provides additional descriptions and evaluations of the proposed MomWorld framework. It is organized as follows.

- Appendix 6.1 summarizes the main contributions.
- Appendix 6.2 discusses the broader impacts.
- Appendix 6.3 describes the evaluated datasets and robustness benchmarks.
- Appendix 6.4 defines the evaluation metrics and aggregation protocols.
- Appendix 6.5 provides additional details of MomWorld.
- Appendix 6.6 presents the implementation and optimization settings.
- Appendix 6.7 reports additional planning and robustness results.
- Appendix 6.8 provides additional qualitative planning results.

### 6.1 Contributions

Our contributions are summarized below.

1) MomWorld Framework. We propose MomWorld, a momentum-aware latent world framework for long-horizon end-to-end autonomous driving. MomWorld explicitly models scene configuration and momentum, then propagates both from historical-to-current evidence into the predicted future. This formulation extends momentum from a cue that stabilizes the current decision into an explicit state for modeling future world evolution.

2) Momentum-Aware Latent World Modeling. We develop MoLWM, which combines History-Conditioned Initialization, Scene-Adaptive Momentum Dynamics (SMD), and Latent World Rollout (LWR). Its learned maintain, update, and reset operations retain persistent trends, introduce scene-conditioned momentum proposals, and suppress stale inherited momentum when interactions change. The predicted LWR sequence is organized as Future World Memory (FWM) for horizon-aligned candidate scoring and base Plan selection.

3) Momentum-Conditioned Flow Matching. We introduce MoFlow to translate predicted world evolution into controlled trajectory corrections. Momentum-Guided Residual Flow (MGRF) conditions residual transport on history-conditioned momentum and FWM, enabling efficient refinement with only a few Euler steps. Horizon-Aware Residual Fusion (HARF) bounds the correction and gradually increases its influence toward longer horizons, preserving reliable near-term decisions while allowing stronger long-range refinement.

4) Comprehensive Long-Horizon Evaluation. We establish a unified evaluation protocol across six-second nuScenes planning and NAVSIM pseudo-simulation. The study jointly measures trajectory accuracy, temporal consistency, predicted collision, planning quality, and computational cost. Controlled component studies, distribution-shift benchmarks, and qualitative analysis provide complementary tests of the design without reducing the evaluation to a single short-horizon score.

### 6.2 Broader Impacts

MomWorld may improve autonomous-driving safety and reliability by enhancing the temporal consistency of long-horizon planning. MoLWM preserves stable motion trends while adapting to scene changes, and MoFlow converts predicted future evolution into bounded trajectory corrections. These capabilities may support smoother decisions and safer interactions in complex traffic. More broadly, momentum-aware latent world modeling may benefit model-based planning in robotics and embodied intelligence.

### 6.3 Datasets

NAVSIM. We evaluate MomWorld on the official NAVSIM v1 navtest split with PDMS [^9] and the official NAVSIM v2 two-stage navhard split with EPDMS [^2]. For diagnostic analysis, we additionally apply the one-stage v2 scorer to navtest. This setting is not an official NAVSIM v2 leaderboard protocol. NAVSIM v1 PDMS combines collision and drivable-area compliance with ego progress, time-to-collision, and comfort. NAVSIM v2 EPDMS adds driving-direction, traffic-light, lane-keeping, and extended-comfort criteria. navhard evaluates an initial real scene together with synthesized follow-up scenes. PDMS and EPDMS lie in $[0,1]$, and we report them on a 0–100 scale.

Bench2Drive. We assess closed-loop performance on Bench2Drive, which contains 220 short routes spanning 44 interactive scenarios in diverse CARLA environments [^17]. We report Driving Score, Success Rate, Efficiency, Comfortness, and the official ability scores for Merging, Overtaking, Emergency Brake, Give Way, and Traffic Sign.

nuScenes. We conduct open-loop planning experiments on the official nuScenes validation split [^1]. The benchmark comprises 1,000 real-world driving scenes of approximately 20 seconds, captured by a surround-view sensor suite. Following standard end-to-end planning evaluation and its six-second extension [^14] [^35], we report L2 displacement error and predicted collision rate from 1 to 6 s and TPC over 4–6 s. Runtime efficiency is measured in FPS. The independently trained 3-s and 6-s models predict six and twelve waypoints at 2 Hz.

Turning-nuScenes (Open-Loop). We conduct extensive open-loop experiments on the Turning-nuScenes dataset [^35], a challenging subset of NuScenes proposed by MomAD [^35] to evaluate trajectory consistency in non-trivial maneuvers. While most planning tasks in the original nuScenes dataset primarily involve go-straight commands, Turning-nuScenes specifically focuses on turning scenarios to assess the temporal coherence of predicted trajectories. To construct this subset, samples are selected using a 25-m displacement threshold between the ground-truth ego positions at 0.5 and 3.0 s. The resulting validation set comprises 680 samples across 17 scenes, accounting for approximately one-tenth of the full nuScenes validation set.

Adv-nuSc (Open-Loop). To evaluate adversarial robustness, we conduct extensive open-loop experiments on the Adv-nuSc [^47] dataset. It contains 156 scenes (6,115 samples) and is specifically crafted to challenge the ego vehicle by introducing adversarial traffic participants. It is built upon the validation split of the nuScenes dataset [^1], which contains 150 scenes, each with 20 seconds of driving data. For each scene, we randomly select up to 10 background vehicles (if there are that many) that come close to the ego vehicle at any point in time and designate them as candidate adversarial agents. Challenger is then used to generate adversarial trajectories for these vehicles, creating diverse and challenging driving scenarios.

NuScenes-C (Open-Loop). NuScenes-C [^10] is a corrupted benchmark derived from the nuScenes validation set, introducing various types of noise to assess the robustness of planning models. It includes 27 corruption types applied at 5 severity levels. To evaluate robustness under adverse weather conditions, we select three representative weather corruptions — Rain, Snow, and Fog — as our test scenarios.

### 6.4 Evaluation Metrics

nuScenes, Turning-nuScenes, Adv-nuSc, and nuScenes-C. Let $\widehat{\mathbf{y}}_{n,T}$ and $\mathbf{y}_{n,T}$ denote the predicted and expert ego positions of sample $n$ at horizon $T$. The horizon-wise displacement error is

$$
\mathrm{L2}@T=\frac{1}{N}\sum_{n=1}^{N}\left\|\widehat{\mathbf{y}}_{n,T}-\mathbf{y}_{n,T}\right\|_{2}.
$$

The average L2 is the arithmetic mean over the reported horizons.

For collision evaluation, let $\widehat{\mathcal{B}}_{n,h}$ denote the oriented ego box induced by the predicted trajectory and $\mathcal{B}^{a}_{n,h}$ the box of participant $a$. Over the valid sample set $\mathcal{V}_{T}$, the predicted collision rate is

$$
\mathrm{Col.}@T=\frac{100}{|\mathcal{V}_{T}|}\sum_{n\in\mathcal{V}_{T}}\mathbb{I}\!\left[\exists\,h\leq T,\ a:\widehat{\mathcal{B}}_{n,h}\cap\mathcal{B}^{a}_{n,h}\neq\varnothing\right].
$$

Samples whose expert trajectories are already in collision are excluded.

Trajectory Prediction Consistency (TPC) follows the public MomAD protocol [^35]. Let $\widehat{\mathbf{y}}_{n,k}^{\,t}$ and $\widehat{\mathbf{y}}_{n,k}^{\,t-1}$ denote the $k$ -th waypoint from two consecutive planning outputs under the evaluator coordinate convention. For the $K_{T}$ waypoints up to horizon $T$, TPC is

$$
\mathrm{TPC}@T=\frac{1}{NK_{T}}\sum_{n=1}^{N}\sum_{k=1}^{K_{T}}\left\|\widehat{\mathbf{y}}_{n,k}^{\,t}-\widehat{\mathbf{y}}_{n,k}^{\,t-1}\right\|_{2}.
$$

Lower L2, Pred.-Col., and TPC indicate greater accuracy, safety, and temporal stability.

NAVSIM v1. NAVSIM v1 combines no-at-fault collision (NC), drivable-area compliance (DAC), ego progress (EP), time-to-collision (TTC), and comfort (C) into PDMS:

$$
\mathrm{PDMS}=\mathrm{NC}\,\mathrm{DAC}\frac{5\,\mathrm{EP}+5\,\mathrm{TTC}+2\,\mathrm{C}}{12}.
$$

All components lie in $[0,1]$, while the tables report percentage-scaled values. Higher PDMS indicates better overall planning quality.

NAVSIM v2. NAVSIM v2 introduces driving-direction compliance (DDC), traffic-light compliance (TLC), lane keeping (LK), history comfort (HC), and extended comfort (EC). To avoid penalizing behavior also exhibited by the human reference, each component is filtered as

$$
f_{m}=\begin{cases}1,&m_{\mathrm{human}}=0,\\
m_{\mathrm{agent}},&\text{otherwise}.\end{cases}
$$

The single-stage EPDMS is

$$
\mathrm{EPDMS}=f_{\mathrm{NC}}f_{\mathrm{DAC}}f_{\mathrm{DDC}}f_{\mathrm{TLC}}\frac{5f_{\mathrm{EP}}+5f_{\mathrm{TTC}}+2f_{\mathrm{LK}}+2f_{\mathrm{HC}}+2f_{\mathrm{EC}}}{16}.
$$

For navhard, let $s_{1}$ be the first-stage EPDMS and $s_{2,i}$ the score of follow-up scene $i$. Its relevance to the first-stage endpoint $\widehat{\mathbf{x}}$ is determined by

$$
\bar{w}_{i}=\frac{\exp\!\left(-\|\mathbf{x}_{i}-\widehat{\mathbf{x}}\|_{2}^{2}/(2\sigma^{2})\right)}{\sum_{j}\exp\!\left(-\|\mathbf{x}_{j}-\widehat{\mathbf{x}}\|_{2}^{2}/(2\sigma^{2})\right)}.
$$

The second-stage score is the Gaussian-weighted aggregation

$$
s_{2}=\sum_{i}\bar{w}_{i}s_{2,i}.
$$

The final navhard score multiplies the two stages:

$$
\mathrm{EPDMS}_{\mathrm{navhard}}=s_{1}s_{2}.
$$

Thus, navhard EPDMS is not an arithmetic mean of the stage-wise scores.

Bench2Drive. Let $N_{\mathrm{route}}$ be the number of evaluated routes and $N_{\mathrm{succ}}$ the number completed without infractions. Success Rate is

$$
\mathrm{SR}=\frac{N_{\mathrm{succ}}}{N_{\mathrm{route}}}\times 100.
$$

Driving Score combines the route-completion percentage $RC_{i}\in[0,100]$ with the multiplicative infraction penalties $p_{i,j}$:

$$
\mathrm{DS}=\frac{1}{N_{\mathrm{route}}}\sum_{i=1}^{N_{\mathrm{route}}}\mathrm{RC}_{i}\prod_{j=1}^{K_{i}}p_{i,j}.
$$

Efficiency compares ego speed with the mean speed of nearby traffic at the official route checkpoints:

$$
\mathrm{Effi.}=\frac{100}{Q}\sum_{q=1}^{Q}\frac{v_{q}^{\mathrm{ego}}}{v_{q}^{\mathrm{near}}}.
$$

Comfortness evaluates whether acceleration, yaw, and jerk variables remain within their prescribed bounds. With $\mathcal{S}$ denoting the evaluated segments and $\mathcal{R}$ the set of smoothness variables, it is summarized as

$$
\mathrm{Comf.}=\frac{100}{|\mathcal{S}|}\sum_{s\in\mathcal{S}}\prod_{t\in s}\prod_{r\in\mathcal{R}}\mathbb{I}\!\left[l_{r}\leq q_{t,r}\leq u_{r}\right].
$$

For each driving ability $a$, the ability score is the success rate over its corresponding route subset $\mathcal{R}_{a}$:

$$
\mathrm{Ability}_{a}=\frac{100}{|\mathcal{R}_{a}|}\sum_{i\in\mathcal{R}_{a}}\mathbb{I}[\mathrm{success}_{i}].
$$

We report Merging, Overtaking, Emergency Brake, Give Way, and Traffic Sign. Higher values are better for all Bench2Drive metrics.

### 6.5 Additional Details of MomWorld

Latent World Rollout (LWR). LWR starts from the history-conditioned scene state $z_{0}$ and momentum $p_{0}$, which are initialized from the current Scene Query and its historical variation.

$$
(z_{0},p_{0})=f_{\mathrm{temp}}\!\left([\operatorname{Pool}(Q_{t}),\operatorname{Pool}(Q_{t})-\operatorname{Pool}(Q_{t-1})]\right).
$$

At each future step, Scene-Adaptive Momentum Dynamics uses the preceding rollout variables and the horizon embedding to produce feature-wise retention, reset, and innovation signals. Writing $x_{k}=[z_{k-1},p_{k-1},r_{k}^{\mathrm{time}}]$ and denoting its three heads by $f_{\rho}$, $f_{g}$, and $f_{u}$, the complete transition is

$$
\displaystyle\rho_{k}
$$
 
$$
\displaystyle=\sigma(f_{\rho}(x_{k})),
$$
$$
\displaystyle g_{k}
$$
 
$$
\displaystyle=\sigma(f_{g}(x_{k})),
$$
$$
\displaystyle u_{k}
$$
 
$$
\displaystyle=\tanh(f_{u}(x_{k})),
$$
$$
\displaystyle p_{k}
$$
 
$$
\displaystyle=\operatorname{LN}\!\left(\rho_{k}\odot(1-g_{k})\odot p_{k-1}+(1-\rho_{k})\odot u_{k}\right),
$$
$$
\displaystyle z_{k}
$$
 
$$
\displaystyle=z_{k-1}+\Delta t\,\pi_{p}(p_{k}),
$$
$$
\displaystyle k
$$
 
$$
\displaystyle=1,\ldots,H.
$$

The inherited branch $\rho_{k}\odot(1-g_{k})\odot p_{k-1}$ preserves persistent motion while suppressing stale components after a scene change. The innovation branch $(1-\rho_{k})\odot u_{k}$ introduces dynamics supported by the current rollout context. Layer normalization controls the momentum scale during long unrolling, and $\pi_{p}$ converts each momentum update into a latent scene-state increment. Repeated application of Eq. (33) produces the complete future rollout.

The LWR output is the horizon-aligned sequence $\{(z_{k},p_{k})\}_{k=1}^{H}$. As detailed in Algorithm 2, each step is converted into a memory token before temporal aggregation organizes the sequence into Future World Memory $M_{t}$. At inference, the entire sequence is generated from historical and current Scene Queries without access to future observations.

Temporal Validity Handling. A deterministic indicator $v_{t}\in\{0,1\}$ specifies whether the preceding Scene Query is temporally valid. The historical input to Temporal Fusion is defined as

$$
Q_{t-1}^{\mathrm{in}}=v_{t}Q_{t-1}+(1-v_{t})Q_{t}.
$$

We set $v_{t}=0$ at scene boundaries, for missing predecessors, or when the expected frame interval is violated. In these cases, $Q_{t-1}^{\mathrm{in}}=Q_{t}$, yielding zero temporal variation and preventing invalid history from affecting $(z_{0},p_{0})$. Latent state, momentum, and Future World Memory are not reused across scene boundaries.

Future World Memory. Future World Memory is reconstructed from the predicted LWR sequence at each planning step rather than maintained across frames. Algorithm 2 details how this memory conditions candidate representations for scoring and selection of the base Plan $\tau^{0}$.

Future-Target Supervision. Future observations are encoded by the shared Image Encoder and detached to construct target Scene Queries:

$$
Q_{t}^{\star}=\operatorname{sg}(Q_{t}),\qquad Q_{t+k}^{\star}=\operatorname{sg}\!\left(\operatorname{ImageEncoder}(\mathcal{O}_{t+k})\right),\quad k=1,\ldots,H,
$$

where $\operatorname{sg}(\cdot)$ blocks gradient propagation through the target-query branch. The latent rollout is aligned with these targets through

$$
\mathcal{L}_{\mathrm{future}}=\frac{1}{H}\sum_{k=1}^{H}d_{Q}\!\left(f_{Q}([z_{k},p_{k}]),Q_{t+k}^{\star}\right),
$$

where $f_{Q}$ projects each LWR step $(z_{k},p_{k})$ into the Scene Query space and $d_{Q}$ denotes the matching loss. Future momentum is supervised by consecutive target-query transitions:

$$
\mathcal{L}_{p}=\frac{1}{H}\sum_{k=1}^{H}\left\|f_{\Delta}(p_{k})-\left[\operatorname{Pool}(Q_{t+k}^{\star})-\operatorname{Pool}(Q_{t+k-1}^{\star})\right]\right\|_{1},
$$

where $f_{\Delta}$ projects latent momentum into the pooled-query space. Auxiliary future-state supervision is defined as

$$
\mathcal{L}_{\mathrm{aux}}=\frac{1}{H}\sum_{k=1}^{H}\left(\ell_{\mathrm{ego}}^{k}+\ell_{\mathrm{agent}}^{k}+\ell_{\mathrm{pres}}^{k}\right).
$$

These terms supervise future ego states, agent states, and participant presence. All future observations and annotations are used only during training and are unavailable at inference.

MoFlow Trajectory Supervision. Trajectory waypoints are encoded using normalized positions and a continuous heading representation:

$$
\phi(x,y,\psi)=\left(x/s_{\mathrm{pos}},y/s_{\mathrm{pos}},\sin\psi,\cos\psi\right).
$$

Let $X_{0}=\phi(\tau^{0})$ and $X_{1}=\phi(\tau^{\star})$ denote the encoded base and expert trajectories. For $\epsilon\sim\mathcal{U}(0,1)$, we form $X_{\epsilon}=(1-\epsilon)X_{0}+\epsilon X_{1}$ and optimize

$$
\mathcal{L}_{\mathrm{FM}}=\mathbb{E}_{\epsilon}\!\left[\left\|v_{\theta}(X_{\epsilon},\epsilon\mid p_{0},M_{t})-(X_{1}-X_{0})\right\|_{2}^{2}\right].
$$

This loss supervises the momentum-conditioned residual vector field. The final Refined trajectory is supervised by

$$
\mathcal{L}_{\tau}=\frac{1}{H}\sum_{k=1}^{H}\left\|\widetilde{\tau}_{k}-\tau_{k}^{\star}\right\|_{1},
$$

where $\tau^{\star}$ denotes the expert trajectory. The standard perception and candidate-planning objectives retain the definitions of the underlying planner. The complete training objective is

$$
\displaystyle\mathcal{L}_{\mathrm{world}}
$$
 
$$
\displaystyle=\mathcal{L}_{\mathrm{future}}+0.5\mathcal{L}_{p}+0.2\mathcal{L}_{\mathrm{aux}},
$$
$$
\displaystyle\mathcal{L}_{\mathrm{MoFlow}}
$$
 
$$
\displaystyle=\mathcal{L}_{\mathrm{FM}}+\mathcal{L}_{\tau},
$$
$$
\displaystyle\mathcal{L}
$$
 
$$
\displaystyle=\mathcal{L}_{\mathrm{percep}}+\mathcal{L}_{\mathrm{plan}}+\mathcal{L}_{\mathrm{world}}+\mathcal{L}_{\mathrm{MoFlow}}.
$$

Algorithm 2 Future-Memory-Conditioned Candidate Scoring

Input: Current Scene Queries $Q_{t}$; predicted future pairs $\{(z_{k},p_{k})\}_{k=1}^{H}$; candidate vocabulary $\mathcal{V}=\{\tau_{i}\}_{i=1}^{N}$; ego status $s_{t}$

Output: Future World Memory $M_{t}$; candidate scores $\{S_{i}\}_{i=1}^{N}$; base Plan $\tau^{0}$

for *$k\leftarrow 1$ to $H$* do

   Memory token: $m_{k}\leftarrow\operatorname{LN}\!\left(z_{k}+\operatorname{MLP}([p_{k},r_{k}^{\mathrm{time}}])\right)$

Temporal memory: $M_{t}\leftarrow\operatorname{SelfAttn}([m_{1},\ldots,m_{H}])$

Scoring context: $C_{t}\leftarrow\operatorname{Concat}(Q_{t},M_{t})$

for *$i\leftarrow 1$ to $N$* do

   Candidate encoding: $e_{i}\leftarrow f_{\tau}(\operatorname{vec}(\tau_{i}))$

Candidate interaction: $E_{\tau}\leftarrow\operatorname{CandEnc}([e_{1},\ldots,e_{N}])$

Future-conditioned decoding: $H_{t}^{\tau}\leftarrow\operatorname{TrajDecoder}(E_{\tau},C_{t})$

for *$i\leftarrow 1$ to $N$* do

   Status conditioning: $h_{t,i}\leftarrow H_{t,i}^{\tau}+W_{s}s_{t}$

   Scoring heads: $\ell_{i}^{r}\leftarrow g_{r}(h_{t,i}),\quad r\in\mathcal{R}$

 $\mathcal{R}\leftarrow\{\mathrm{imi},\mathrm{NC},\mathrm{DAC},\mathrm{TTC},\mathrm{EP},\mathrm{DDC},\mathrm{LK},\mathrm{TLC}\}$

Imitation distribution: $\pi^{\mathrm{imi}}\leftarrow\operatorname{softmax}(\ell^{\mathrm{imi}})$

for *$i\leftarrow 1$ to $N$* do

   Rule probabilities: $p_{i}^{r}\leftarrow\sigma(\ell_{i}^{r}),\quad r\in\mathcal{R}\setminus\{\mathrm{imi}\}$

   Score aggregation: $S_{i}\leftarrow\operatorname{Aggregate}(\pi_{i}^{\mathrm{imi}},\{p_{i}^{r}\})$

Plan selection: $i^{\star}\leftarrow\arg\max_{i}S_{i}$, $\quad\tau^{0}\leftarrow\tau_{i^{\star}}$

Each candidate trajectory is represented by a query token, while the concatenated current-scene and future-memory tokens $C_{t}=[Q_{t};M_{t}]$ serve as the key–value context of the trajectory decoder. For attention head $a$, this interaction is written as

$$
\displaystyle A_{t}^{(a)}
$$
 
$$
\displaystyle=\operatorname{softmax}\!\left(\frac{(E_{\tau}W_{Q}^{(a)})(C_{t}W_{K}^{(a)})^{\top}}{\sqrt{d_{a}}}\right),
$$
$$
\displaystyle Z_{t}^{(a)}
$$
 
$$
\displaystyle=A_{t}^{(a)}(C_{t}W_{V}^{(a)}).
$$

Although all candidates share the same $M_{t}$, their distinct query tokens produce different attention weights. The resulting representations therefore capture candidate-specific compatibility with the predicted scene evolution.

In our NAVSIM implementation, $N=16{,}384$, $H=40$, and $D=256$. Each candidate contains $40$ poses over a $4$ -s horizon, while $M_{t}$ contains one future-memory token per planning step. The trajectory decoder contains three layers with eight attention heads. During training, vocabulary dropout retains half of the candidates, whereas all candidates are scored at inference.

The predicted logits correspond to imitation, no-at-fault collision (NC), drivable-area compliance (DAC), time-to-collision (TTC), ego progress (EP), driving-direction compliance (DDC), lane keeping (LK), and traffic-light compliance (TLC). Their inference score is

$$
\displaystyle S_{i}
$$
 
$$
\displaystyle=0.03\log\pi_{i}^{\mathrm{imi}}+0.10\log p_{i}^{\mathrm{TLC}}+0.10\log p_{i}^{\mathrm{NC}}+0.90\log p_{i}^{\mathrm{DAC}}
$$
 
$$
\displaystyle+0.20\log p_{i}^{\mathrm{DDC}}+6.0\log\!\left(7.0p_{i}^{\mathrm{TTC}}+7.0p_{i}^{\mathrm{EP}}+3.0p_{i}^{\mathrm{LK}}\right).
$$

The scoring heads are supervised by candidate-level imitation targets and PDM subscores, allowing gradients to propagate through the trajectory decoder into $M_{t}$ and thereby learn future-conditioned candidate ranking.

Candidate scoring is performed only once at each planning step. Each complete candidate is represented by one query token that attends globally to the future-memory sequence; no candidate-specific world rollout or explicit one-to-one temporal matching is performed. After selecting $\tau^{0}$, MoFlow uses the same $M_{t}$ to refine this Plan without reconstructing the memory or re-ranking the candidate vocabulary.

### 6.6 Implementation Details

Table 9: NAVSIM implementation configuration of MomWorld.

| Configuration | Value |
| --- | --- |
| Model input history | 2 camera frames / 4 LiDAR sweeps |
| Image resolution | $2048\times 512$ |
| Image backbone | V-99-eSE VoVNet |
| Planner latent / FFN width | 256 / 1024 |
| Planner attention layers / heads | 3 / 8 |
| Candidate vocabulary size | 16,384 |
| Planning horizon / interval | 40 steps / 0.1 s |
| LWR horizon / interval | 40 steps / 0.1 s |
| Maximum agent slots | 30 |
| Momentum persistence initialization | 0.9 |
| Trajectory encoding | Eq. (39), $s_{\mathrm{pos}}=50$ |
| Flow solver / steps | explicit Euler with midpoint-time conditioning / 4 |
| Heading normalization | after Euler integration |
| Residual clamp $\delta$ | componentwise, $\pm 2.0$ |
| Residual-gate initialization $b$ | $-4$ |
| Horizon weighting | linear ramp, $0.05\!\rightarrow\!1.0$ |
| Optimizer / learning rate | Adam / $2\times 10^{-4}$ |
| Epochs / per-device batch size | 20 / 2 |
| Precision / gradient clipping | FP16 mixed precision / 5.0 |

NAVSIM Instantiation. Table 9 summarizes the complete NAVSIM configuration. We instantiate MomWorld on GTRS-Dense [^26] and keep the sensor input, backbone, candidate vocabulary, and 4-s planning grid fixed across all controlled comparisons.

MoLWM Configuration. MoLWM uses LWR to propagate latent scene states and momentum on the planner grid. Future ego trajectories and track-aligned agent state and presence annotations are used only as training targets and are unavailable at inference.

MoFlow Configuration. MoFlow refines the selected Plan using four explicit Euler updates with midpoint flow-time conditioning. After the four Euler updates, the heading representation is normalized once before trajectory decoding. HARF applies a componentwise residual clamp followed by a linear horizon ramp, while the negative gate initialization starts training from conservative corrections.

Optimization Protocol. The perception, planning, MoLWM, and MoFlow objectives are jointly optimized with Eq. (42). Unless otherwise specified, controlled ablations use identical inputs, backbones, candidate trajectories, training data, and optimization budgets.

![[nus_vis_fl.png|Refer to caption]]

Figure 5: Qualitative 6-second planning results on nuScenes. MomWorld generates smooth trajectories when transitioning from turns to straight driving, decelerating in dense traffic, turning left at intersections, and avoiding vehicles ahead.

### 6.7 More Planning Results

Table 10: Closed-loop planning and multi-ability performance on Bench2Drive. <sup>∗</sup> denotes expert feature distillation. Higher is better for all metrics.

<table><thead><tr><th rowspan="2">Method</th><th rowspan="2">Venue</th><th colspan="4">Closed-loop Performance</th><th colspan="6">Multi-Ability (%) <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th></tr><tr><th>DS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>SR (%) <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>Effi.<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>Comf.<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>Merge</th><th>Overtake</th><th>Emergency Brake</th><th>Give Way</th><th>Traffic Sign</th><th>Mean</th></tr></thead><tbody><tr><th>TCP-traj <sup>∗</sup> <sup><a href="#fn:44">44</a></sup></th><th>NeurIPS’22</th><td>59.90</td><td>30.00</td><td>76.54</td><td>18.08</td><td>12.50</td><td>22.73</td><td>52.72</td><td>40.00</td><td>46.63</td><td>34.92</td></tr><tr><th>UniAD <sup><a href="#fn:14">14</a></sup></th><th>CVPR’23</th><td>45.81</td><td>16.36</td><td>129.21</td><td>43.58</td><td>14.10</td><td>17.78</td><td>21.67</td><td>10.00</td><td>14.21</td><td>15.55</td></tr><tr><th>ThinkTwice <sup>∗</sup> <sup><a href="#fn:16">16</a></sup></th><th>CVPR’23</th><td>62.44</td><td>31.23</td><td>69.33</td><td>16.22</td><td>13.72</td><td>22.93</td><td>52.99</td><td>50.00</td><td>47.78</td><td>37.48</td></tr><tr><th>DriveAdapter <sup>∗</sup> <sup><a href="#fn:15">15</a></sup></th><th>ICCV’23</th><td>64.22</td><td>33.08</td><td>70.22</td><td>16.01</td><td>14.55</td><td>22.61</td><td>54.04</td><td>50.00</td><td>50.45</td><td>38.33</td></tr><tr><th>VAD <sup><a href="#fn:19">19</a></sup></th><th>ICCV’23</th><td>42.35</td><td>15.00</td><td>157.94</td><td>46.01</td><td>8.11</td><td>24.44</td><td>18.64</td><td>20.00</td><td>19.15</td><td>18.07</td></tr><tr><th>GenAD <sup><a href="#fn:54">54</a></sup></th><th>ECCV’24</th><td>44.81</td><td>15.90</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td></tr><tr><th>DriveTransformer <sup><a href="#fn:18">18</a></sup></th><th>ICLR’25</th><td>63.46</td><td>35.01</td><td>100.64</td><td>20.78</td><td>17.57</td><td>35.00</td><td>48.36</td><td>40.00</td><td>52.10</td><td>38.60</td></tr><tr><th>SparseDrive <sup><a href="#fn:40">40</a></sup></th><th>ICRA’25</th><td>44.54</td><td>16.71</td><td>170.21</td><td>48.63</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td></tr><tr><th>SimLingo <sup><a href="#fn:33">33</a></sup></th><th>CVPR’25</th><td>86.02</td><td>67.27</td><td>259.23</td><td>33.67</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td></tr><tr><th>Hydra-NeXt <sup><a href="#fn:25">25</a></sup></th><th>ICCV’25</th><td>73.86</td><td>50.00</td><td>197.76</td><td>20.68</td><td>40.00</td><td>64.44</td><td>61.67</td><td>50.00</td><td>50.00</td><td>53.22</td></tr><tr><th>HiP-AD <sup><a href="#fn:41">41</a></sup></th><th>ICCV’25</th><td>86.77</td><td>69.09</td><td>203.12</td><td>19.36</td><td>50.00</td><td>84.44</td><td>83.33</td><td>40.00</td><td>72.10</td><td>65.98</td></tr><tr><th>MomAD (SD) <sup><a href="#fn:35">35</a></sup></th><th>CVPR’25</th><td>47.91</td><td>18.11</td><td>174.91</td><td>51.20</td><td>13.21</td><td>21.02</td><td>18.01</td><td>20.00</td><td>21.07</td><td>18.66</td></tr><tr><th>FUMP <sup><a href="#fn:29">29</a></sup></th><th>arXiv’25</th><td>45.67</td><td>16.36</td><td>–</td><td>–</td><td>12.50</td><td>24.44</td><td>20.00</td><td>21.50</td><td>19.15</td><td>19.51</td></tr><tr><th>DIVER (SD) <sup><a href="#fn:37">37</a></sup></th><th>TPAMI’26</th><td>49.21</td><td>21.56</td><td>177.00</td><td>54.72</td><td>15.98</td><td>28.22</td><td>23.71</td><td>20.00</td><td>24.38</td><td>22.46</td></tr><tr><th>GraphWorld (SD) <sup><a href="#fn:36">36</a></sup></th><th>arXiv’26</th><td>51.55</td><td>25.47</td><td>181.12</td><td>56.59</td><td>18.74</td><td>31.66</td><td>25.30</td><td>20.00</td><td>26.66</td><td>24.47</td></tr><tr><th>GuideFlow <sup><a href="#fn:30">30</a></sup></th><th>CVPR’26</th><td>75.21</td><td>51.36</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td></tr><tr><th>SparseDriveV2 <sup><a href="#fn:39">39</a></sup></th><th>ECCV’26</th><td>89.15</td><td>70.00</td><td>199.84</td><td>18.32</td><td>66.25</td><td>75.55</td><td>75.00</td><td>50.00</td><td>71.57</td><td>67.67</td></tr><tr><th>MomWorld (Ours)</th><th>–</th><td>74.07</td><td>50.00</td><td>198.87</td><td>20.43</td><td>45.63</td><td>64.21</td><td>61.71</td><td>50.00</td><td>54.10</td><td>55.07</td></tr></tbody></table>

Bench2Drive. As shown in Table 10, MomWorld achieves a Driving Score of 74.07 and a Success Rate of 50.00%, together with an efficiency score of 198.87, a comfort score of 20.43, and a mean ability score of 55.07. Compared with Hydra-NeXt [^25], MomWorld improves Driving Score by 0.21 points and mean ability by 1.85 points, while matching its Success Rate and maintaining similar efficiency and comfort. It also records higher Driving Score, Success Rate, and mean ability than the related MomAD, DIVER, and GraphWorld baselines [^35] [^37] [^36]. These results indicate more balanced route-level and interaction performance, although HiP-AD and SparseDriveV2 retain higher Driving Score, Success Rate, and mean ability [^41] [^39]. We further evaluate robustness under turning-heavy motion, adversarial interactions, and adverse-weather corruptions. Following DIVER and GraphWorld [^37] [^36], all three experiments report predicted collision rate at 1, 2, and 3 seconds. Turning-nuScenes also reports L2 displacement where the original source provides it. Published values are literature references rather than matched reruns. The shaded MomWorld rows are reserved for evaluations using identical data manifests and evaluator settings. The asterisk preserves the reimplementation mark used by DIVER.

![[Navsim_vis_fl_v2.png|Refer to caption]]

Figure 6: Qualitative planning results on NAVSIM. Comparison of GuideFlow 30 and MomWorld across turning, straight driving, braking, stable driving, and vehicle avoidance. MomWorld produces smoother trajectories with improved roadway compliance and safer interactions.

Table 11: Open-loop robustness on Turning-nuScenes [^35] and Adv-nuSc [^47]. Turning-nuScenes additionally reports average $L_{2}$ where available. Lower is better.

<table><tbody><tr><th rowspan="3">Method</th><td colspan="5">Turning-nuScenes</td><td colspan="4">Adv-nuSc</td></tr><tr><td rowspan="2">Avg. <math><semantics><msub><mi>L</mi> <mn>2</mn></msub> <annotation>L_{2}</annotation></semantics></math> (m) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td colspan="4">Col. Rate (%) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td colspan="4">Col. Rate (%) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td></tr><tr><th>UniAD</th><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>0.800</td><td>4.100</td><td>6.960</td><td>3.950</td></tr><tr><th>VAD</th><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>4.460</td><td>7.590</td><td>9.080</td><td>7.050</td></tr><tr><th>SparseDrive</th><td>0.86</td><td>0.04</td><td>0.17</td><td>0.98</td><td>0.40</td><td>0.029</td><td>0.618</td><td>2.430</td><td>1.026</td></tr><tr><th>DiffusionDrive <sup>∗</sup></th><td>–</td><td>0.03</td><td>0.14</td><td>0.85</td><td>0.34</td><td>0.068</td><td>1.299</td><td>3.646</td><td>1.671</td></tr><tr><th>MomAD</th><td>0.76</td><td>0.03</td><td>0.13</td><td>0.79</td><td>0.32</td><td>–</td><td>–</td><td>–</td><td>–</td></tr><tr><th>DIVER</th><td>–</td><td>0.03</td><td>0.11</td><td>0.67</td><td>0.27</td><td>0.033</td><td>0.423</td><td>1.798</td><td>0.752</td></tr><tr><th>GraphWorld</th><td>–</td><td>0.03</td><td>0.12</td><td>0.72</td><td>0.28</td><td>0.028</td><td>0.420</td><td>1.780</td><td>0.742</td></tr><tr><th>MomWorld (Ours)</th><td>0.72</td><td>0.02</td><td>0.11</td><td>0.66</td><td>0.26</td><td>0.027</td><td>0.418</td><td>1.755</td><td>0.733</td></tr></tbody></table>

Table 12: Collision rate under three adverse-weather corruptions on nuScenes-C [^10]. Lower is better.

<table><thead><tr><th rowspan="2">Method</th><th colspan="3">Snow</th><th colspan="3">Rain</th><th colspan="3">Fog</th></tr><tr><th>1s</th><th>2s</th><th>3s</th><th>1s</th><th>2s</th><th>3s</th><th>1s</th><th>2s</th><th>3s</th></tr></thead><tbody><tr><th>SparseDrive</th><td>0.13</td><td>0.27</td><td>0.50</td><td>0.11</td><td>0.27</td><td>0.55</td><td>0.14</td><td>0.36</td><td>0.58</td></tr><tr><th>DiffusionDrive <sup>∗</sup></th><td>0.09</td><td>0.24</td><td>0.39</td><td>0.07</td><td>0.18</td><td>0.35</td><td>0.06</td><td>0.18</td><td>0.30</td></tr><tr><th>MomAD</th><td>0.08</td><td>0.16</td><td>0.30</td><td>0.06</td><td>0.17</td><td>0.31</td><td>0.06</td><td>0.19</td><td>0.32</td></tr><tr><th>DIVER</th><td>0.07</td><td>0.13</td><td>0.25</td><td>0.05</td><td>0.16</td><td>0.27</td><td>0.04</td><td>0.16</td><td>0.25</td></tr><tr><th>GraphWorld</th><td>0.07</td><td>0.15</td><td>0.28</td><td>0.05</td><td>0.15</td><td>0.29</td><td>0.05</td><td>0.17</td><td>0.29</td></tr><tr><th>MomWorld (Ours)</th><td>0.06</td><td>0.13</td><td>0.23</td><td>0.04</td><td>0.15</td><td>0.25</td><td>0.04</td><td>0.16</td><td>0.23</td></tr></tbody></table>

Turning-nuScenes. As shown in Table 11, MomWorld achieves the lowest average $L_{2}$ error of 0.72 m and average collision rate of 0.26%, improving upon the 0.76 m and 0.27% references reported by MomAD and DIVER [^35] [^37]. It also matches the best 2-s collision rate of 0.11% and reduces the 3-s result from 0.67% to 0.66%. These results support the effectiveness of momentum-aware rollout for stable planning through high-curvature maneuvers.

Adv-nuSc. On Adv-nuSc, MomWorld obtains collision rates of 0.027%, 0.418%, and 1.755% at 1, 2, and 3 seconds, respectively, with an average of 0.733%. It improves the corresponding GraphWorld references at every horizon and reduces the average collision rate from 0.742% to 0.733% [^36]. The consistent gains under adversarial interactions indicate improved adaptation when surrounding agents deviate from previously observed trends.

nuScenes-C. Table 12 shows that MomWorld is best or tied at every evaluated horizon under snow, rain, and fog. At 3 seconds, it achieves collision rates of 0.23%, 0.25%, and 0.23%, reducing the strongest published references of 0.25%, 0.27%, and 0.25% by 0.02 percentage points in each condition [^37] [^36]. The consistent performance across weather corruptions demonstrates stronger robustness to degraded visual observations.

### 6.8 Additional Qualitative Results

To complement the quantitative evaluation, we examine the planning behavior of MomWorld on nuScenes and NAVSIM. The selected cases cover extended-horizon maneuvers, dense traffic, changing road geometry, braking, and vehicle avoidance. We present 6-second planning results on nuScenes and comparative visualizations on NAVSIM.

Long-Horizon Planning Visualization on nuScenes. Figure 5 presents the 6-second planning behavior of MomWorld on the nuScenes validation set. Across the four scenarios, MoLWM propagates observed motion trends while adapting to predicted scene evolution, whereas MoFlow converts these dynamics into bounded, horizon-aware corrections. The resulting trajectories remain smooth and responsive throughout the extended planning horizon.

Planning Visualization on NAVSIM. Figure 6 compares MomWorld with GuideFlow [^30] across five representative scenarios. MomWorld maintains smoother trajectories within drivable regions and adapts more coherently to road geometry and surrounding vehicles. These examples illustrate the benefits of latent momentum propagation and horizon-aware residual refinement for stable, scene-responsive planning.

[^1]: H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom nuScenes: a multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11621–11631. Cited by: §1, §4.1, §6.3, §6.3.

[^2]: W. Cao, M. Hallgarten, T. Li, D. Dauner, X. Gu, C. Wang, Y. Miron, M. Aiello, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, A. Geiger, and K. Chitta Pseudo-simulation for autonomous driving. In Proceedings of the 9th Conference on Robot Learning, Proceedings of Machine Learning Research, Vol. 305, pp. 4709–4722. External Links: [Link](https://proceedings.mlr.press/v305/cao25a.html) Cited by: §1, §4.1, §6.3.

[^3]: S. Chen, B. Jiang, H. Gao, B. Liao, Q. Xu, Q. Zhang, C. Huang, W. Liu, and X. Wang Vadv2: end-to-end vectorized autonomous driving via probabilistic planning. arXiv preprint arXiv:2402.13243. Cited by: Table 5.

[^4]: Y. Chen, Y. Wang, and Z. Zhang Drivinggpt: unifying driving world modeling and planning with multi-modal autoregressive transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 26890–26900. Cited by: Table 5.

[^5]: H. Chi, J. Liang, Z. Song, L. Yang, S. Li, H. Zhang, and C. Lv PV-wm: a heterogeneous micro-macro world model for articulated pedestrian-vehicle co-rollout. arXiv preprint arXiv:2609.07328. Cited by: §2.2.

[^6]: H. Chi, D. Qiu, H. Su, H. Liu, Z. Li, H. Zhang, and C. Lv Driver-wm: a driver-centric traffic-conditioned latent world model for in-cabin dynamics rollout. arXiv preprint arXiv:2605.05092. Cited by: §2.2.

[^7]: K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger TransFuser: imitation with transformer-based sensor fusion for autonomous driving. IEEE Transactions on Pattern Analysis and Machine Intelligence 45 (11), pp. 12878–12895. External Links: [Document](https://dx.doi.org/10.1109/TPAMI.2022.3200245) Cited by: §4.2, §4.2, Table 5, Table 5.

[^8]: C. Dang, S. Ang, Y. Li, H. Tian, J. Wang, G. Li, H. Ye, J. Ma, L. Chen, and Y. Wang Drivefine: refining-augmented masked diffusion vla for precise and robust driving. arXiv preprint arXiv:2602.14577. Cited by: §4.2.

[^9]: D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, A. Geiger, and K. Chitta NAVSIM: data-driven non-reactive autonomous vehicle simulation and benchmarking. In Advances in Neural Information Processing Systems, Vol. 37, pp. 28706–28719. External Links: [Document](https://dx.doi.org/10.52202/079017-0902) Cited by: §1, §4.1, §6.3.

[^10]: Y. Dong, C. Kang, J. Zhang, Z. Zhu, Y. Wang, X. Yang, H. Su, X. Wei, and J. Zhu Benchmarking robustness of 3d object detection to common corruptions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1022–1032. Cited by: §6.3, Table 12.

[^11]: L. Feng, Y. Gao, E. Zablocki, Q. Li, W. Li, S. Liu, M. Cord, and A. Alahi Rap: 3d rasterization augmented end-to-end planning. In International Conference on Learning Representations, Vol. 2026, pp. 37722–37737. Cited by: §4.2.

[^12]: I. Georgiev, V. Giridhar, N. Hansen, and A. Garg Pwm: policy learning with multi-task world models. In International Conference on Learning Representations, Vol. 2025, pp. 13737–13757. Cited by: Table 5.

[^13]: Y. Hong, X. Zhou, Y. Li, X. Zhou, L. Liu, Y. Luo, S. Xu, L. Yang, and Z. Song DriveFuture: future-aware latent world models for autonomous driving. External Links: 2605.09701, [Link](https://arxiv.org/abs/2605.09701) Cited by: §2.2.

[^14]: Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, L. Lu, X. Jia, Q. Liu, J. Dai, Y. Qiao, and H. Li Planning-oriented autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 17853–17862. Cited by: §1, §2.1, Table 1, Table 2, Table 5, §6.3, Table 10.

[^15]: X. Jia, Y. Gao, L. Chen, J. Yan, P. L. Liu, and H. Li DriveAdapter: breaking the coupling barrier of perception and planning in end-to-end autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 7953–7963. Cited by: Table 10.

[^16]: X. Jia, P. Wu, L. Chen, J. Xie, C. He, J. Yan, and H. Li Think twice before driving: towards scalable decoders for end-to-end autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21983–21994. Cited by: Table 10.

[^17]: X. Jia, Z. Yang, Q. Li, Z. Zhang, and J. Yan Bench2Drive: towards multi-ability benchmarking of closed-loop end-to-end autonomous driving. In Advances in Neural Information Processing Systems Datasets and Benchmarks Track, Vol. 37. External Links: [Document](https://dx.doi.org/10.52202/079017-0025) Cited by: §6.3.

[^18]: X. Jia, J. You, Z. Zhang, and J. Yan DriveTransformer: unified transformer for scalable end-to-end autonomous driving. In International Conference on Learning Representations, External Links: [Link](https://openreview.net/forum?id=vlBiKpn5q9) Cited by: Table 10.

[^19]: B. Jiang, S. Chen, Q. Xu, B. Liao, J. Chen, H. Zhou, Q. Zhang, W. Liu, C. Huang, and X. Wang VAD: vectorized scene representation for efficient autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 8340–8350. Cited by: §1, §2.1, Table 2, Table 10.

[^20]: K. Li, Z. Li, S. Lan, Y. Xie, Z. Zhang, J. Liu, Z. Wu, Z. Yu, and J. M. Alvarez Hydra-mdp++: advancing end-to-end driving via expert-guided hydra-distillation. arXiv preprint arXiv:2503.12820. Cited by: §4.2.

[^21]: Y. Li, L. Fan, J. He, Y. Wang, Y. Chen, Z. Zhang, and T. Tan Enhancing end-to-end autonomous driving with latent world model. In International Conference on Learning Representations, Vol. 2025, pp. 42942–42959. External Links: [Link](https://proceedings.iclr.cc/paper_files/paper/2025/file/6aa4967920e495e90aeeaa3acf18d019-Paper-Conference.pdf) Cited by: §1, §2.2, Table 1, Table 5.

[^22]: Y. Li, Y. Wang, Y. Liu, J. He, L. Fan, and Z. Zhang End-to-end driving with online trajectory evaluation via bev world model. arXiv preprint arXiv:2504.01941. Cited by: Table 5.

[^23]: Y. Li, K. Xiong, X. Guo, F. Li, S. Yan, G. Xu, L. Zhou, L. Chen, H. Sun, B. Wang, et al. Recogdrive: a reinforced cognitive framework for end-to-end autonomous driving. arXiv preprint arXiv:2506.08052. Cited by: §4.2, Table 5, Table 5.

[^24]: Z. Li, K. Li, S. Wang, S. Lan, Z. Yu, Y. Ji, Z. Li, Z. Zhu, J. Kautz, Z. Wu, et al. Hydra-mdp: end-to-end multimodal planning with multi-target hydra-distillation. arXiv preprint arXiv:2406.06978. Cited by: Table 5.

[^25]: Z. Li, S. Wang, S. Lan, Z. Yu, Z. Wu, and J. M. Alvarez Hydra-NeXt: robust closed-loop driving with open-loop training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 27305–27314. Cited by: §6.7, Table 10.

[^26]: Z. Li, W. Yao, Z. Wang, X. Sun, J. Chen, N. Chang, M. Shen, Z. Wu, S. Lan, and J. M. Alvarez Generalized trajectory scoring for end-to-end multimodal planning. arXiv preprint arXiv:2506.06664. Cited by: §2.1, §4.1, §6.6.

[^27]: B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang, and X. Wang DiffusionDrive: truncated diffusion model for end-to-end autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12037–12047. Cited by: §1, §2.1, §4.2, §4.2, Table 5.

[^28]: Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le Flow matching for generative modeling. In International Conference on Learning Representations, External Links: [Link](https://openreview.net/forum?id=PqvMRDCJT9t) Cited by: §1.

[^29]: L. Liu, C. Jia, Z. Song, H. Pan, B. Liao, W. Sun, Y. Zhang, L. Yang, and Y. Luo Fully unified motion planning for end-to-end autonomous driving. arXiv preprint arXiv:2504.12667. Cited by: Table 5, Table 10.

[^30]: L. Liu, C. Jia, G. Yu, Z. Song, J. Li, F. Jia, P. Wu, X. Hao, and Y. Luo GuideFlow: constraint-guided flow matching for planning in end-to-end autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3719–3728. External Links: [Link](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_GuideFlow_Constraint-Guided_Flow_Matching_for_Planning_in_End-to-End_Autonomous_Driving_CVPR_2026_paper.html) Cited by: §2.1, §4.2, Table 1, Figure 6, §6.8, Table 10.

[^31]: L. Liu, Z. Song, C. Jia, H. Ye, X. Hao, L. Chen, et al. DriveWorld-vla: unified latent-space world modeling with vision-language-action for autonomous driving. arXiv preprint arXiv:2602.06521. Cited by: §4.2, Table 5.

[^32]: C. Min, D. Zhao, L. Xiao, J. Zhao, X. Xu, Z. Zhu, L. Jin, J. Li, Y. Guo, J. Xing, L. Jing, Y. Nie, and B. Dai DriveWorld: 4d pre-trained scene understanding via world models for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15522–15533. Cited by: §1, §2.2.

[^33]: K. Renz, L. Chen, E. Arani, and O. Sinavski SimLingo: vision-only closed-loop autonomous driving with language-action alignment. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11993–12003. Cited by: Table 10.

[^34]: C. Shi, S. Shi, K. Sheng, B. Zhang, and L. Jiang Drivex: omni scene modeling for learning generalizable world knowledge in autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 28599–28609. Cited by: Table 5.

[^35]: Z. Song, C. Jia, L. Liu, H. Pan, Y. Zhang, J. Wang, X. Zhang, S. Xu, L. Yang, and Y. Luo Don’t shake the wheel: momentum-aware planning in end-to-end autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22432–22441. Cited by: Figure 1, §1, §1, §2.1, Figure 4, §4.1, §4.4, Table 1, Table 2, §6.3, §6.3, §6.4, §6.7, §6.7, Table 10, Table 11.

[^36]: Z. Song, C. Jia, L. Liu, L. Yang, S. Zhang, F. Jia, F. Zhao, P. Wu, S. Xu, C. Lv, and Y. Luo GraphWorld: long-horizon planning with world models for end-to-end autonomous driving. External Links: 2606.16274, [Document](https://dx.doi.org/10.48550/arXiv.2606.16274), [Link](https://arxiv.org/abs/2606.16274) Cited by: §1, §2.2, §6.7, §6.7, §6.7, Table 10.

[^37]: Z. Song, L. Liu, H. Pan, B. Liao, M. Guo, L. Yang, Y. Zhang, S. Xu, C. Jia, and Y. Luo DIVER: reinforced diffusion breaks imitation bottlenecks in end-to-end autonomous driving. IEEE Transactions on Pattern Analysis and Machine Intelligence, pp. 1–17. External Links: [Document](https://dx.doi.org/10.1109/TPAMI.2026.3708096) Cited by: §1, §2.1, Table 1, Table 5, §6.7, §6.7, §6.7, Table 10.

[^38]: B. Sun, Y. Cao, Y. Wang, R. Wang, J. Shang, X. Feng, J. Lu, J. Shi, S. Yang, X. Yan, et al. MindDrive: an all-in-one framework bridging world models and vision-language model for end-to-end autonomous driving. arXiv preprint arXiv:2512.04441. Cited by: §4.2.

[^39]: W. Sun, X. Lin, K. Chen, Z. Pei, X. Li, Y. Shi, and S. Zheng SparseDriveV2: scoring is all you need for end-to-end autonomous driving. arXiv preprint arXiv:2603.29163. External Links: 2603.29163, [Link](https://arxiv.org/abs/2603.29163) Cited by: §6.7, Table 10.

[^40]: W. Sun, X. Lin, Y. Shi, C. Zhang, H. Wu, and S. Zheng SparseDrive: end-to-end autonomous driving via sparse scene representation. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pp. 8795–8801. External Links: [Document](https://dx.doi.org/10.1109/ICRA55743.2025.11128800) Cited by: §1, §2.1, Table 1, Table 2, Table 10.

[^41]: Y. Tang, Z. Xu, Z. Meng, and E. Cheng HiP-AD: hierarchical and multi-granularity planning with deformable attention for autonomous driving in a single decoder. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 25605–25615. Cited by: §6.7, Table 10.

[^42]: L. Wang, Y. Zheng, Q. Chen, S. Li, Y. Zhang, Z. Xing, Q. Zhang, X. Li, D. Qian, P. Yang, Y. Dong, C. Hao, X. Ye, J. Han, Y. Pan, and D. Zhao Latent-WAM: latent world action modeling for end-to-end autonomous driving. External Links: 2603.24581, [Link](https://arxiv.org/abs/2603.24581) Cited by: §2.2, §4.2.

[^43]: X. Weng, B. Ivanovic, Y. Wang, Y. Wang, and M. Pavone PARA-Drive: parallelized architecture for real-time autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15449–15458. Cited by: §2.1, Table 5.

[^44]: P. Wu, X. Jia, L. Chen, J. Yan, H. Li, and Y. Qiao Trajectory-guided control prediction for end-to-end autonomous driving: a simple yet strong baseline. In Advances in Neural Information Processing Systems, Vol. 35, pp. 6119–6132. External Links: [Link](https://proceedings.neurips.cc/paper_files/paper/2022/hash/286a371d8a0a559281f682f8fbf89834-Abstract.html) Cited by: Table 10.

[^45]: T. Xia, Y. Li, L. Zhou, J. Yao, K. Xiong, H. Sun, B. Wang, K. Ma, G. Chen, H. Ye, W. Liu, and X. Wang DriveLaW: unifying planning and video generation in a latent driving world. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 39701–39712. Cited by: §2.2, Table 5.

[^46]: Z. Xing, X. Zhang, Y. Hu, B. Jiang, T. He, Q. Zhang, X. Long, and W. Yin Goalflow: goal-driven flow matching for multimodal trajectories generation in end-to-end autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 1602–1611. Cited by: Table 5.

[^47]: Z. Xu, B. Li, H. Gao, M. Gao, Y. Chen, M. Liu, C. Yan, H. Zhao, S. Feng, and H. Zhao Challenger: affordable adversarial driving video generation. External Links: 2505.15880, [Document](https://dx.doi.org/10.48550/arXiv.2505.15880), [Link](https://arxiv.org/abs/2505.15880) Cited by: §6.3, Table 11.

[^48]: J. Yang, K. Chitta, S. Gao, L. Chen, Y. Shao, X. Jia, H. Li, A. Geiger, X. Yue, and L. Chen Resim: reliable world simulation for autonomous driving. Advances in Neural Information Processing Systems 38, pp. 167710–167741. Cited by: Table 5.

[^49]: P. Yang, B. Lu, Z. Xia, C. Han, Y. Gao, T. Zhang, K. Zhan, X. Lang, Y. Zheng, and Q. Zhang WorldRFT: latent world model planning with reinforcement fine-tuning for autonomous driving. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40, pp. 11649–11657. Cited by: Table 5.

[^50]: W. Yao, Z. Li, S. Lan, Z. Wang, X. Sun, J. M. Alvarez, and Z. Wu DriveSuprim: towards precise trajectory selection for end-to-end planning. arXiv preprint arXiv:2506.06659. Cited by: §4.2, §4.2, Table 5.

[^51]: C. Yuan, Z. Zhang, J. Sun, S. Sun, Z. Huang, C. D. W. Lee, D. Li, Y. Han, A. Wong, K. P. Tee, et al. Drama: an efficient end-to-end motion planner for autonomous driving with mamba. arXiv preprint arXiv:2408.03601. Cited by: Table 5.

[^52]: B. Zhang, N. Song, X. Jin, and L. Zhang Bridging past and future: end-to-end autonomous driving with historical prediction and planning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6854–6863. Cited by: §1, §2.1.

[^53]: K. Zhang, Z. Tang, X. Hu, X. Pan, X. Guo, Y. Liu, J. Huang, L. Yuan, Q. Zhang, X. Long, X. Cao, and W. Yin Epona: autoregressive diffusion world model for autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 27220–27230. External Links: [Link](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Epona_Autoregressive_Diffusion_World_Model_for_Autonomous_Driving_ICCV_2025_paper.html) Cited by: §2.2, Table 1, Table 5.

[^54]: W. Zheng, R. Song, X. Guo, C. Zhang, and L. Chen GenAD: generative end-to-end autonomous driving. In European Conference on Computer Vision, External Links: [Link](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08174.pdf) Cited by: Table 10.

[^55]: Y. Zheng, P. Yang, Z. Xing, Q. Zhang, Y. Zheng, Y. Gao, P. Li, T. Zhang, Z. Xia, P. Jia, X. Lang, and D. Zhao World4Drive: end-to-end autonomous driving via intention-aware physical latent world model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 28632–28642. Cited by: §1, §2.2, §4.2, Table 1, Table 5.

[^56]: Z. Zhou, T. Cai, S. Z. Zhao, Y. Zhang, Z. Huang, B. Zhou, and J. Ma AutoVLA: a vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. arXiv preprint arXiv:2506.13757. Cited by: Table 5.

[^57]: Z. Zhou, R. Yang, Y. Guo, S. X. Chen, T. Feng, K. Pistunova, Y. Shen, L. Su, J. Ma, et al. SpanVLA: efficient action bridging and learning from negative-recovery samples for vision-language-action model. arXiv preprint arXiv:2604.19710. Cited by: §4.2.

[^58]: J. Zou, S. Chen, B. Liao, Z. Zheng, Y. Song, L. Zhang, Q. Zhang, W. Liu, and X. Wang DiffusionDriveV2: reinforcement learning-constrained truncated diffusion modeling in end-to-end autonomous driving. arXiv preprint arXiv:2512.07745. Cited by: §4.2.
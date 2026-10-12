---
title: "IDOL: Inverse-Dynamics-Guided Future Prediction for End-to-End Autonomous Driving"
source: "https://arxiv.org/html/2605.31476v2"
author:
published:
created: 2026-10-12
description:
tags:
  - "clippings"
---
Chenghao Zhang    Timin Li    Dongmei Li Affiliation: Department of Electronic Engineering, Tsinghua University Affiliation: {zhangch24, litm22}@mails.tsinghua.edu.cn    lidmei@tsinghua.edu.cn

###### Abstract

End-to-end autonomous driving has emerged as a compelling paradigm for learning planning directly from sensor observations, while recent world-model-based approaches further enrich this paradigm by enabling explicit reasoning about how the scene may evolve in the future. Yet future prediction alone does not guarantee better planning unless the predicted evolution can be converted into planning-relevant trajectory updates. Many current methods still forecast future scene states without explicitly decoding the motion implications hidden in state transitions. As a result, future reasoning often remains descriptively useful but only weakly coupled to executable motion generation. To address this limitation, we propose IDOL, an inverse-dynamics-guided future prediction framework for world-model-based end-to-end planning in latent BEV space, where inverse dynamics serves as the key bridge between future prediction and trajectory optimization. IDOL first predicts multiple future latent scene states with a BEV world model, then applies an inverse dynamics model to adjacent latent futures to decode transition-aware trajectory features and recover planning-relevant motion deltas that explain how the latent world evolves over time. These transition features refine the candidate-specific planning queries through spatial cross-attention and global modulation. A lightweight closed-loop refinement module feeds the refined queries back into candidate-conditioned latent rollouts, enabling an additional round of future-informed query refinement before trajectory decoding. By introducing inverse dynamics into latent future reasoning, IDOL tightens the coupling between world modeling and planning. Extensive experiments on the NAVSIM v1 and NAVSIM v2 benchmarks show that IDOL achieves state-of-the-art performance among comparable methods.

Code: [https://github.com/Waterloo-ins/IDOL](https://github.com/Waterloo-ins/IDOL)

## 1 Introduction

Autonomous driving [^27] [^5] [^57] [^40] [^46] [^14] [^22] [^60] has advanced significantly in recent years, driven by a paradigm shift from modular perception-prediction-planning pipelines to unified end-to-end learning frameworks. Recent advances have further expanded this paradigm toward future-aware, world-model-guided planning [^80] [^70], scalable sim-real training [^61], and video-centric pretraining [^53], leading to markedly stronger robustness and planning performance on realistic benchmarks such as nuScenes [^2] and NAVSIM [^9]. At the same time, multimodal and generative formulations [^72] [^45] [^35] [^82] have shown that planning quality can benefit substantially from moving beyond deterministic single-pass regression.More recently, VLA-based systems [^31] [^25] [^47] extended the scope of future-aware decision making by injecting language-grounded reasoning, action abstraction, and reflective post-training into driving systems.

![[Figure1.png|Refer to caption]]

Figure 1: Comparison of three planning paradigms. (a) Traditional methods directly map sensor inputs to a predicted trajectory. (b) World-model-based methods additionally predict future latent BEV states to support planning, but these predicted futures are only loosely connected to motion generation. (c) In contrast, IDOL further applies an inverse dynamics model to adjacent future latent BEV states to extract transition-aware motion cues for trajectory refinement, making future prediction more actionable for planning.

For trajectory planning, traditional end-to-end autonomous driving methods [^59] [^64] typically rely on the current state, which constitutes a form of passive planning, as illustrated in Figure 1(a). Subsequently, many approaches [^28] [^29] have introduced world models into end-to-end planning, adopting the predict-decide paradigm to make decision-making more foresighted, as illustrated in Figure 1(b).

Yet a fundamental bottleneck remains: even with future prediction, current methods [^38] [^79] still lack a clear and interpretable bridge between forecasting scene evolution and using it to produce dynamically meaningful trajectory updates. Predicted futures may therefore be informative but insufficiently actionable: the planner can anticipate what may happen next, yet lacks a principled mechanism to revise the current trajectory so that it remains executable, dynamically coherent, and robust over long horizons. This limitation is especially critical in safety-sensitive driving, where future reasoning must ultimately support motion generation rather than passive forecasting. Our key insight is that *the missing signal lies not only in the predicted future states themselves, but more importantly in the transitions between them*. As illustrated in Figure 1(c), inverse dynamics naturally decodes this signal. While forward world modeling predicts how the scene evolves under a motion hypothesis, inverse dynamics asks the complementary question of *what motion-relevant change is implied by the transition from one state to the next*. In latent Bird’s Eye View space, adjacent predicted futures capture the expected evolution of the ego vehicle and surrounding scene under a planning hypothesis, and their transitions therefore contain recoverable motion semantics reflecting trajectory adjustments consistent with that evolution. Rather than treating future latent states as static planning features, we decode adjacent latent transitions with an inverse dynamics model to extract transition-aware trajectory features and infer motion deltas for planning optimization. In this way, inverse dynamics explicitly bridges world modeling and trajectory generation, transforming future prediction from *passive scene forecasting* into *planning-aware transition reasoning*.

Based on this perspective, we propose IDOL, a world-model-based end-to-end planner operating in latent BEV space that tightly couples future reasoning with trajectory optimization. IDOL first predicts multiple future latent scene states with a BEV world model, providing anticipated scene evolution beyond the current observation. It then applies an inverse dynamics model to adjacent latent futures to infer motion-aware transition features and planning-relevant trajectory deltas that explain the predicted evolution. These signals are aggregated and injected into the planner to optimize the anchor trajectory, allowing motion updates to follow the dynamics implied by future state transitions rather than future context alone. A lightweight closed-loop refinement module further re-evaluates the updated trajectory to improve long-horizon consistency.

In summary, our contributions are as follows:

i. We identify a fundamental gap in recent end-to-end and world-model-based planners: current methods still lack a sufficiently explicit and interpretable mechanism for converting predicted future evolution into planning-relevant trajectory updates, which limits the actionability of future reasoning for executable motion generation.

ii. We propose IDOL, a world-model-based end-to-end planner in latent BEV space that introduces inverse dynamics into latent future reasoning, thereby establishing an explicit bridge between world modeling and executable motion generation. We further design a lightweight closed-loop refinement module to improve long-horizon consistency through another round of future-aware reasoning.

iii. We conduct extensive experiments on both NAVSIM v1 and NAVSIM v2, including comprehensive comparisons and analysis, and show that our method consistently achieves state-of-the-art performance among comparable methods. These results validate the effectiveness of our design in improving planning quality across different benchmark settings.

## 2 Related work

### 2.1 End-to-end autonomous driving

End-to-end autonomous driving [^79] [^41] [^10] [^34] [^47] [^4] [^11] [^52] directly maps sensor observations to waypoints, trajectories, or low-level controls, reducing error accumulation and hand-crafted interfaces in modular pipelines. Early methods [^7] largely relied on imitation learning, while planning-oriented formulations [^13] [^69] [^75] [^24] [^23] [^55] [^32] [^56] later became dominant. TransFuser [^7] established a strong transformer-based sensor-fusion baseline, while UniAD [^21] unified perception and prediction around downstream planning. DriveMamba [^54] improved efficiency and scalability with a task-centric state-space architecture, and LEAD [^48] revisited imitation learning through learner-expert asymmetry. DiffusionDrive [^44] and GoalFlow [^71] introduced generative planning via truncated diffusion and flow matching to better model multimodal trajectories. More recently, VLA-based end-to-end driving [^83] [^51] [^72] [^63] has extended this direction toward language-grounded action generation and post-training alignment. DriveVLA-W0 [^37] incorporated world-model objectives to provide dense supervision and improve data scaling, while AutoDrive- $R^{2}$ [^77] introduced chain-of-thought reasoning, self-reflection, and GRPO-based post-training for physically plausible trajectory generation. Despite these advances, most end-to-end planners still couple planning mainly to the current scene or output-level multimodality, leaving the connection between future state transitions and action generation largely implicit.

### 2.2 World model in autonomous driving

World models in autonomous driving [^6] [^66] [^36] [^28] predict scene dynamics for forecasting, planning, simulation, and long-horizon decisions. Dreamer [^15] [^16] [^17] introduced latent imagined rollouts for long-horizon planning. GAIA-1 [^20] formulated driving world modeling as action-conditioned multimodal sequence generation, while Vista [^12] emphasized environmental diversity and controllability. WoTE [^38] evaluates trajectory candidates through predicted BEV states. SeerDrive [^79] couples scene evolution and planning through iterative bidirectional refinement. DriveLaW [^70] unifies video generation and motion planning in shared latents that support closed-loop planning.

Despite this progress, the key challenge is how predicted futures shape decisions. Although recent methods [^67] [^84] [^79] jointly train world modeling and planning, the connection between future scene dynamics and planned motion remains insufficiently explained. We address this gap by making latent BEV transitions action-relevant, constraining planning through both anticipated scene evolution and its implied control signals.

### 2.3 Inverse dynamics models

Inverse dynamics models [^19] [^49] [^30] [^1] originate from physics, control, and reinforcement learning, where the goal is to infer the action or control input that caused a transition between two states. This viewpoint is useful because it makes action semantics explicit: an inverse dynamics model can identify which aspects of a transition are action-relevant, recover latent controls when actions are unavailable or noisy, and encourage representations that preserve controllable structure rather than merely appearance. Recent machine learning work [^62] has employed inverse dynamics for model-based control, representation learning, and general sequential decision-making.

![[Figure2.png|Refer to caption]]

Figure 2: Overview of the proposed IDOL framework. (a) The closed-loop refinement module rolls out future latent BEV states with the BEV world model and refines the planning query through inverse-dynamics feedback. (b) The planning network fuses the refined query with trajectory anchors, predicts candidate trajectories, and selects the best one by scoring and ranking. (c) The inverse-dynamics-guided refinement network decodes adjacent future BEV states into spatial and global dynamics cues, which are fused to update the ego query.

In autonomous driving, inverse dynamics remains comparatively underexplored and is mostly used as an auxiliary component. ReSim [^73] employs an inverse dynamics model to convert predicted videos into executable trajectories and assess controllability, while FutureSightDrive [^78] formulates planning as recovering actions from current observations and imagined future scenes. These works suggest that inverse dynamics can bridge predicted future states and executable actions. However, existing driving world models typically focus on latent future forecasting, trajectory evaluation, or reward-guided selection, without explicitly learning transition-to-action mappings between adjacent predicted states. Our method addresses this gap by introducing an inverse dynamics module over predicted latent BEV states, explicitly coupling state evolution with planning and aligning trajectory generation with action-consistent future transitions.

## 3 Methodology

### 3.1 Preliminary

##### End-to-end autonomous driving.

End-to-end autonomous driving formulates trajectory planning as learning a direct mapping from raw sensory observations and ego-state inputs to future ego motion, without relying on a manually designed modular stack. Recent world-model-based methods further improve this paradigm by explicitly modeling future scene evolution and using the predicted futures to guide planning.

##### Inverse dynamics model.

Inverse dynamics model infers the latent action or motion representation that explains the transition between two adjacent states. In its general form, given two consecutive states $\xi_{t}$ and $\xi_{t+1}$, inverse dynamics seeks a compact transition descriptor

$$
d_{t}=f_{\mathrm{idm}}(\xi_{t},\xi_{t+1}),
$$

which captures the motion-consistent change from $\xi_{t}$ to $\xi_{t+1}$. This formulation is particularly suitable for autonomous driving, since adjacent future states naturally encode how ego motion interacts with scene evolution. In our setting, the two states are imagined BEV latent states produced by the world model. Therefore, inverse dynamics provides a natural bridge from *future state transition* to *trajectory-related motion features*, making it well suited for refining trajectories from consecutive imagined BEV frames.

### 3.2 Scene representation

As shown in Figure 2, we encode the current scene into a latent BEV representation $B$ using a TransFuser-style image-LiDAR backbone [^7]. We adopt an offline trajectory-anchor vocabulary $\mathcal{A}=\{\tau_{k}\}_{k=1}^{K}$, where $\tau_{k}\in\mathbb{R}^{S\times 3}$ represents a candidate future motion [^5] [^40]. Each anchor feature $a_{k}$ is concatenated with the ego-state feature $z$ to initialize a candidate-specific planning query:

$$
e_{k,\mathrm{init}}=\phi([z,a_{k}]),\qquad k=1,\ldots,K,
$$

which guides subsequent future imagination and trajectory refinement.

### 3.3 Latent BEV world model

For each trajectory candidate $k$ and refinement iteration $\ell$, we initialize the latent rollout with $q_{k,\ell}^{(0)}=e_{k,\mathrm{in}}^{(\ell)}$ and $B_{k,\ell}^{(0)}=B$. At each step $u$, the ego query is injected at the corresponding future location of anchor $\tau_{k}$, and a transformer-based world model predicts the next query and latent BEV state:

$$
\displaystyle B_{k,\ell,\mathrm{cond}}^{(u)}
$$
 
$$
\displaystyle=\mathcal{I}\!\left(B_{k,\ell}^{(u)},q_{k,\ell}^{(u)};\tau_{k}\right),
$$
$$
\displaystyle\left[q_{k,\ell}^{(u+1)},B_{k,\ell}^{(u+1)}\right]
$$
 
$$
\displaystyle=\mathrm{BEVWorldModel}\!\left(\left[q_{k,\ell}^{(u)},B_{k,\ell,\mathrm{cond}}^{(u)}\right]\right),
$$

where $\mathcal{I}$ denotes query injection. Repeating this process with the same anchor yields the candidate-specific rollout

$$
\mathcal{R}_{k}^{(\ell)}=\left\{\left(q_{k,\ell}^{(u)},B_{k,\ell,\mathrm{cond}}^{(u)}\right)\right\}_{u=0}^{U},
$$

whose adjacent candidate-conditioned BEV states are subsequently decoded by the inverse dynamics model.

### 3.4 Inverse-dynamics-guided refinement network

#### 3.4.1 Inverse dynamics model

While the world model predicts future latent scene evolution, planning still requires a mechanism that translates the imagined transitions into motion-relevant feedback. To bridge this gap, we introduce an inverse dynamics model (IDM).

For each pair of adjacent candidate-conditioned latent BEV states within the $\ell$ -th rollout, the IDM produces two complementary transition representations:

$$
\left(S_{k,\ell}^{(u)},g_{k,\ell}^{(u)}\right)=\mathrm{IDM}\!\left(B_{k,\ell,\mathrm{cond}}^{(u)},B_{k,\ell,\mathrm{cond}}^{(u+1)}\right),
$$

where $S_{k,\ell}^{(u)}\in\mathbb{R}^{H\times W\times C}$ is a spatial dynamics map that preserves position-wise motion variations across the latent BEV feature map, and $g_{k,\ell}^{(u)}\in\mathbb{R}^{C}$ is a pooled global dynamics feature that summarizes the overall transition.

When multiple rollout transitions are available, we aggregate the transition representations over the rollout horizon:

$$
S_{k}^{(\ell)}=\frac{1}{U}\sum_{u=0}^{U-1}S_{k,\ell}^{(u)},\qquad g_{k}^{(\ell)}=\frac{1}{U}\sum_{u=0}^{U-1}g_{k,\ell}^{(u)}.
$$

The key idea is that future transitions should capture not only *what* changes globally, but also *where* critical changes occur. A pooled inverse-dynamics feature alone is often too coarse, since only part of the spatial variation is relevant to the current trajectory hypothesis. The spatial representation preserves localized transition evidence, while the global representation summarizes overall evolution. Together, they bridge latent future prediction and motion refinement at the transition level.

#### 3.4.2 Closed-loop query refinement

We refine the planning query by combining the spatial and global inverse-dynamics cues through a dual-branch mechanism.

For the first closed-loop iteration, the refinement process is initialized by the anchor-conditioned query:

$$
e_{k,\mathrm{in}}^{(0)}=e_{k,\mathrm{init}}.
$$

Given the transition representations extracted from the corresponding latent rollout, the spatial branch first allows the current planning query to retrieve candidate-relevant local motion evidence through cross-attention:

$$
e_{k,\mathrm{spa}}^{(\ell)}=\mathrm{LN}\!\left(e_{k,\mathrm{in}}^{(\ell)}+\mathrm{CrossAttn}\!\left(e_{k,\mathrm{in}}^{(\ell)},S_{k}^{(\ell)},S_{k}^{(\ell)}\right)\right).
$$

Here, $e_{k,\mathrm{spa}}^{(\ell)}$ denotes the query after spatial transition-aware refinement.

The global dynamics representation is subsequently used to calibrate the spatially refined query through an MLN [^65]:

$$
e_{k,\mathrm{ref}}^{(\ell)}=\mathrm{MLN}_{\mathrm{idm}}\!\left(e_{k,\mathrm{spa}}^{(\ell)},g_{k}^{(\ell)}\right).
$$

This dual-branch design assigns complementary roles to the two inverse-dynamics representations: the spatial branch determines *where* motion-relevant future changes occur, while the global branch provides holistic transition-aware calibration. Consequently, the feedback from imagined futures becomes more spatially selective and informative than using a pooled inverse-dynamics representation alone.

To prevent iterative drift, subsequent closed-loop iterations re-anchor the refined query to the original anchor-conditioned query:

$$
e_{k,\mathrm{in}}^{(\ell+1)}=\mathrm{MLN}_{\mathrm{cl}}\!\left(e_{k,\mathrm{init}},e_{k,\mathrm{ref}}^{(\ell)}\right),\qquad\ell=0,\ldots,L-2.
$$

The updated query $e_{k,\mathrm{in}}^{(\ell+1)}$ initializes the next candidate-conditioned rollout, while the anchor $\tau_{k}$ remains fixed. Trajectory offsets are decoded only after the final refinement, where $e_{k,\mathrm{ref}}^{(L-1)}$ is fed into a transformer decoder and an MLP head. The resulting candidates are ranked by a reward model [^38] to select the final trajectory.

### 3.5 Training loss

Our network is trained with trajectory offset regression, imitation- and simulator-reward supervision, and current/future BEV semantic supervision. We additionally supervise motion correction in the final global IDM feature:

$$
\mathcal{L}_{\mathrm{corr}}=\ell_{\mathrm{SL1}}\!\left(h_{\mathrm{mot}}(g_{k^{\star}}^{(L-1)}),\operatorname{sg}\!\left[E_{\mathrm{mot}}(\tau^{\mathrm{GT}})-E_{\mathrm{mot}}(\tau_{k^{\star}})\right]\right),
$$

where $k^{\star}$ is the closest anchor under winner-take-all assignment, $E_{\mathrm{mot}}$ concatenates fixed normalized speed, acceleration, and heading-rate sequences, and $\operatorname{sg}$ denotes stop-gradient. The masked Smooth L1 loss $\ell_{\mathrm{SL1}}$ averages errors within each motion component and then across components, excluding unreliable low-speed heading rates. The head $h_{\mathrm{mot}}$ is used only during training.

The overall objective is

$$
\mathcal{L}=\lambda_{\mathrm{off}}\mathcal{L}_{\mathrm{off}}+\lambda_{\mathrm{off\text{-}im}}\mathcal{L}_{\mathrm{off\text{-}im}}+\lambda_{\mathrm{im}}\mathcal{L}_{\mathrm{im}}+\lambda_{\mathrm{sim}}\mathcal{L}_{\mathrm{sim}}+\lambda_{\mathrm{map}}\mathcal{L}_{\mathrm{map}}+\lambda_{\mathrm{corr}}\mathcal{L}_{\mathrm{corr}},
$$

where the first five terms supervise offsets, auxiliary and main imitation rewards, simulator metrics, and weighted current/future BEV semantics, respectively. Loss definitions and coefficients are detailed in Appendix B.

## 4 Experiments

### 4.1 Benchmark

We evaluate our method on NAVSIM v1 [^9] and NAVSIM v2 [^3] under the official closed-loop protocols. On NAVSIM v1, we report results on the navtest split using the Predictive Driver Model Score (PDMS), which aggregates no-collision (NC), drivable area compliance (DAC), time-to-collision (TTC), comfort, and ego progress (EP). On NAVSIM v2, we further evaluate on the challenging navhard split under the two-stage pseudo-simulation protocol using the Extended PDMS (EPDMS), which additionally considers driving direction compliance (DDC), traffic light compliance (TLC), lane keeping (LK), history comfort (HC), and extended comfort (EC).

### 4.2 Implementation details

We use a planning horizon of 4 s with a sampling interval of 0.5 s, resulting in 8 future waypoints. Our model adopts a ResNet34-based TransFuser backbone for multi-modal fusion, where the stitched front-view image input is resized to $256\times 1024$. The latent feature dimension is set to 256. In the current implementation, the BEV representation is flattened into 64 latent queries, with $H_{\mathrm{BEV}}\times W_{\mathrm{BEV}}=8\times 8$, and trajectory planning is performed over 256 trajectory anchors. The resulting model contains 69.36M parameters and achieves 17.65 FPS on a single NVIDIA RTX 3090 GPU with batch size 1, indicating that IDOL maintains practical inference efficiency while introducing inverse-dynamics-guided future reasoning.

Training is conducted on 4 NVIDIA GeForce RTX 3090 GPUs with a batch size of 4 per GPU for approximately 24 hours. We use AdamW with an initial learning rate of $2\times 10^{-4}$ and weight decay of $1\times 10^{-4}$.

Table 1: Performance comparison on the NAVSIM v1 navtest split with closed-loop metrics. “C” and “L” denote camera and LiDAR inputs, respectively. For fair comparison, all listed results are reported under the ResNet-34 image-backbone setting. Bold and underlined values indicate the best and second-best results, respectively. ResWorld <sup>∗</sup> denotes the historical-frame variant that uses historical frames to obtain temporal residuals, and DiffE2E <sup>†</sup> denotes the variant with a ResNet-34 image encoder.

Method Img. Backbone Input NC $\uparrow$ DAC $\uparrow$ TTC $\uparrow$ Comf. $\uparrow$ EP $\uparrow$ PDMS $\uparrow$ VADv2 [^5] ResNet-34 C & L 97.2 89.1 91.6 100 76.0 80.9 UniAD [^21] ResNet-34 Camera 97.8 91.9 92.9 100 78.8 83.4 PARA-Drive [^68] ResNet-34 Camera 97.9 92.4 93.0 99.8 79.3 84.0 TransFuser [^7] ResNet-34 C & L 97.7 92.8 92.8 100 79.2 84.0 LAW [^36] ResNet-34 Camera 96.4 95.4 88.7 99.9 81.7 84.6 DiffusionDrive [^44] ResNet-34 C & L 98.2 96.2 94.7 100 82.2 88.1 WoTE [^38] ResNet-34 C & L 98.5 96.8 94.9 99.9 81.9 88.3 SeerDrive [^79] ResNet-34 C & L 98.4 97.0 94.9 99.9 83.2 88.9 ResWorld <sup>∗</sup> [^80] ResNet-34 C & L 98.9 96.5 95.6 100 83.1 89.0 MeanFuser [^64] ResNet-34 Camera 98.6 97.0 95.0 100 82.8 89.0 DiffE2E <sup>†</sup> [^82] ResNet-34 C & L 99.2 96.8 96.7 100 83.6 89.8 IDOL ResNet-34 C & L 98.7 97.7 95.5 100 84.0 90.0

### 4.3 Comparison with SOTA

Table 1 compares IDOL with recent end-to-end planners on the NAVSIM navtest split using a ResNet-34 [^18] image backbone. IDOL achieves the best PDMS of 90.0. Table 2 further evaluates the challenging NAVSIM v2 navhard split, where IDOL attains the highest final EPDMS of 38.0, outperforming the strongest baseline WoTE by 10.1 points.

Beyond the main comparisons, Table 9 reports the extended NAVSIM v2 navtest benchmark. IDOL achieves 89.6 EPDMS, ranking first among learned methods and remaining competitive with the human agent. Table 10 further evaluates the more challenging navhard stage-1-only setting, where IDOL achieves the best performance with 76.2 EPDMS, surpassing Hydra-MDP <sup>†</sup> by 3.1 points and WoTE by 9.5 points. Across NAVSIM v1 navtest, NAVSIM v2 navtest, navhard stage-1, and the two-stage navhard evaluation, IDOL consistently achieves state-of-the-art performance among comparable learned planners, demonstrating the robustness and stability of inverse-dynamics-guided future reasoning under both standard and challenging long-tail driving scenarios.

Table 2: Performance comparison on the NAVSIM v2 navhard split. PDM-Closed is listed separately as a privileged planner using ground-truth perception.

<table><thead><tr><th>Method</th><th>Backbone</th><th>Stage</th><th>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>DDC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>TLC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>LK <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>HC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>EC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>EPDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th></tr></thead><tbody><tr><th rowspan="2">PDM-Closed <sup><a href="#fn:8">8</a></sup></th><td rowspan="2">-</td><td>Stage 1</td><td>94.4</td><td>98.8</td><td>100.0</td><td>99.5</td><td>100.0</td><td>93.5</td><td>99.3</td><td>87.7</td><td>36.0</td><td></td></tr><tr><td>Stage 2</td><td>88.1</td><td>90.6</td><td>96.3</td><td>98.5</td><td>100.0</td><td>83.1</td><td>73.7</td><td>91.5</td><td>25.4</td><td>51.3</td></tr><tr><th rowspan="2">LTF <sup><a href="#fn:50">50</a></sup></th><td rowspan="2">ResNet34</td><td>Stage 1</td><td>96.2</td><td>79.5</td><td>99.1</td><td>99.5</td><td>84.1</td><td>95.1</td><td>94.2</td><td>97.5</td><td>79.1</td><td></td></tr><tr><td>Stage 2</td><td>77.7</td><td>70.2</td><td>84.2</td><td>98.0</td><td>85.1</td><td>75.6</td><td>45.4</td><td>95.7</td><td>75.9</td><td>23.1</td></tr><tr><th rowspan="2">GTRS-DP <sup><a href="#fn:42">42</a></sup></th><td rowspan="2">ResNet34</td><td>Stage 1</td><td>94.7</td><td>78.8</td><td>96.1</td><td>99.5</td><td>83.0</td><td>94.4</td><td>92.0</td><td>97.5</td><td>72.8</td><td></td></tr><tr><td>Stage 2</td><td>80.3</td><td>74.4</td><td>84.9</td><td>98.0</td><td>81.9</td><td>78.8</td><td>45.4</td><td>96.7</td><td>70.1</td><td>23.8</td></tr><tr><th rowspan="2">DiffusionDrive <sup><a href="#fn:44">44</a></sup></th><td rowspan="2">ResNet34</td><td>Stage 1</td><td>96.0</td><td>79.7</td><td>97.4</td><td>99.5</td><td>81.3</td><td>93.1</td><td>90.8</td><td>96.8</td><td>73.8</td><td></td></tr><tr><td>Stage 2</td><td>82.1</td><td>72.2</td><td>88.5</td><td>98.7</td><td>85.1</td><td>78.8</td><td>49.2</td><td>89.3</td><td>71.2</td><td>24.2</td></tr><tr><th rowspan="2">GuideFlow <sup><a href="#fn:45">45</a></sup></th><td rowspan="2">ResNet34</td><td>Stage 1</td><td>96.6</td><td>80.5</td><td>96.3</td><td>99.3</td><td>82.3</td><td>94.9</td><td>91.5</td><td>97.7</td><td>67.8</td><td></td></tr><tr><td>Stage 2</td><td>87.3</td><td>76.7</td><td>88.8</td><td>99.2</td><td>84.3</td><td>85.1</td><td>49.7</td><td>93.1</td><td>44.5</td><td>27.1</td></tr><tr><th rowspan="2">WoTE <sup><a href="#fn:38">38</a></sup></th><td rowspan="2">ResNet34</td><td>Stage 1</td><td>97.4</td><td>88.2</td><td>97.7</td><td>99.3</td><td>82.7</td><td>96.4</td><td>90.8</td><td>97.3</td><td>68.0</td><td></td></tr><tr><td>Stage 2</td><td>81.2</td><td>77.7</td><td>84.8</td><td>98.1</td><td>85.9</td><td>78.5</td><td>46.2</td><td>96.6</td><td>63.3</td><td>27.9</td></tr><tr><th rowspan="2">IDOL</th><td rowspan="2">ResNet34</td><td>Stage 1</td><td>97.2</td><td>89.6</td><td>98.0</td><td>99.6</td><td>82.3</td><td>96.9</td><td>95.3</td><td>97.6</td><td>72.9</td><td></td></tr><tr><td>Stage 2</td><td>84.9</td><td>81.7</td><td>87.7</td><td>98.5</td><td>84.7</td><td>81.1</td><td>50.0</td><td>95.4</td><td>64.0</td><td>38.0</td></tr></tbody></table>

### 4.4 Ablation study

We conduct ablation studies on the NAVSIM navtest split to analyze the contribution of each design in IDOL. We report PDMS and its five component metrics. Additional ablation results and analyses are provided in Appendix E.

##### Effect of inverse dynamics and closed-loop refinement.

Table 3 studies the main components of IDOL. Without IDM, future BEV states lack explicit transition-to-motion feedback, limiting their effect on trajectory refinement. Introducing IDM improves PDMS from 87.3 to 89.2 by converting latent future transitions into motion-aware query updates. A single closed-loop refinement iteration further raises PDMS to 90.0 by feeding the updated query back into future reasoning. Increasing the number of iterations to two or three reduces performance, suggesting possible over-correction.

Table 3: Main ablation of inverse dynamics and closed-loop refinement on the NAVSIM navtest split. IDM denotes the inverse dynamics model, and CL denotes closed-loop refinement.

| IDM | CL iters. | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf. $\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| $\times$ | $\times$ | 98.5 | 95.0 | 95.2 | 100 | 81.5 | 87.3 |
| ✓ | $\times$ | 98.6 | 97.0 | 95.4 | 100 | 83.3 | 89.2 |
| ✓ | $1$ | 98.7 | 97.7 | 95.5 | 100 | 84.0 | 90.0 |
| ✓ | $2$ | 98.6 | 97.3 | 95.2 | 100 | 83.5 | 89.4 |
| ✓ | $3$ | 98.8 | 97.2 | 95.7 | 100 | 83.1 | 89.4 |

##### Temporal design of IDM.

Table 4 compares different temporal inputs for IDM. The two-frame design directly decodes adjacent BEV transitions, while the four-frame variant uses a longer window that may dilute local transition cues for immediate refinement. The better performance of the two-frame setting shows that adjacent transitions provide a more compact and effective basis for planning-oriented motion refinement.

Table 4: Ablation of temporal decoding in the inverse dynamics model. The two-frame variant decodes adjacent BEV transitions, while the four-frame variant uses a longer future window.

| IDM temporal input | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf. $\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| 4 frame | 98.7 | 97.3 | 95.6 | 100 | 83.7 | 89.6 |
| 2 frame | 98.7 | 97.7 | 95.5 | 100 | 84.0 | 90.0 |

##### Effect of spatial and global dynamics branches.

Table 5 analyzes the dual-branch design. The global branch captures holistic transition patterns but lacks spatial selectivity, while the spatial branch preserves local transition evidence but lacks global calibration. Combining both branches provides spatially selective and globally calibrated future-to-planning feedback, achieving the best overall PDMS.

Table 5: Ablation of the spatial and global dynamics branches in the inverse-dynamics-guided refinement network.

| Dynamics Branch | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf. $\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| w/o spatial branch | 98.8 | 96.9 | 95.8 | 100 | 83.2 | 89.3 |
| w/o global branch | 98.7 | 97.2 | 95.3 | 100 | 83.9 | 89.5 |
| dual-branch IDM | 98.7 | 97.7 | 95.5 | 100 | 84.0 | 90.0 |

### 4.5 Visualization

To better understand inverse-dynamics-guided refinement, we visualize a representative NAVSIM navtest case in Figure 3. Panels (a)–(c) show the latent BEV change, the corresponding IDM spatial response, and the refinement-query attention, respectively. The white contour marks high-change regions extracted from Panel (a). The visualization shows that IDM indeed concentrates on the motion-informative parts within the latent BEV change regions, rather than unrelated background variations, and the query attends to these motion-aware cues during refinement. Panel (d) compares the original, refined, and expert trajectories, where the refined trajectory better aligns with the expert, suggesting that IDM converts latent future transitions into actionable planning updates. More qualitative results are provided in Appendix F.

Figure 3: Qualitative visualization of inverse-dynamics-guided refinement on the NAVSIM navtest split.

## 5 Conclusion

In this work, we proposed IDOL, an inverse-dynamics-guided future prediction framework for end-to-end autonomous driving. By decoding adjacent latent BEV futures, IDOL converts predicted scene evolution into transition-aware motion cues and refines trajectories through a lightweight closed-loop process. This explicitly bridges world modeling and executable motion generation, making future prediction more actionable for planning. Experiments on NAVSIM v1 and NAVSIM v2 show that IDOL achieves state-of-the-art performance on navtest, navhard stage-1, and two-stage navhard benchmarks, demonstrating the effectiveness and generality of inverse-dynamics-guided future reasoning. This framework provides a general paradigm for integrating future state transitions with motion generation in future autonomous driving research.

## References

## Appendix A Notation

Table 6 summarizes the main notation used in IDOL. We use $k$ to index trajectory candidates, $s$ to index trajectory poses, $u$ to index latent rollout steps, and $\ell$ to index query-refinement iterations. The final refinement iteration is $\ell=L-1$.

Table 6: Main notation used in IDOL.

| Symbol | Description |
| --- | --- |
| $K,\ S$ | Number of trajectory anchors and future poses per trajectory. |
| $U,\ L$ | Number of latent rollout steps and query-refinement iterations. |
| $B$ | Current latent BEV representation. |
| $\mathcal{A}=\{\tau_{k}\}_{k=1}^{K}$ | Trajectory-anchor vocabulary, with $\tau_{k}\in\mathbb{R}^{S\times 3}$ containing planar positions and headings. |
| $\tau^{\mathrm{GT}}$ | Expert future trajectory in the current ego frame. |
| $a_{k},\ z$ | Encoded anchor feature and ego-state feature. |
| $e_{k,\mathrm{init}}$ | Initial planning query conditioned on the ego state and anchor $k$. |
| $e_{k,\mathrm{in}}^{(\ell)},\ e_{k,\mathrm{ref}}^{(\ell)}$ | Planning query before and after refinement at iteration $\ell$. |
| $q_{k,\ell}^{(u)}$ | Ego query at step $u$ of the candidate-conditioned latent rollout. |
| $B_{k,\ell,\mathrm{cond}}^{(u)}$ | Latent BEV state conditioned through ego-query injection. |
| $S_{k}^{(\ell)},\ g_{k}^{(\ell)}$ | Spatial and global IDM features aggregated over rollout transitions. |
| $k^{\star}$ | Index of the anchor closest to $\tau^{\mathrm{GT}}$ under the winner-take-all assignment. |
| $E_{\mathrm{mot}}(\tau)$ | Fixed, normalized motion representation of trajectory $\tau$. |
| $h_{\mathrm{mot}}$ | Training-only head predicting a motion-feature residual from the final global IDM feature. |
| $\operatorname{sg}[\cdot]$ | Stop-gradient operator applied to the supervision target. |

## Appendix B Implementation details

##### Hyperparameters.

Table 7 summarizes the key hyperparameters used in IDOL. The trajectory-anchor vocabulary is constructed offline and kept fixed during training and evaluation. For ablation studies, all settings are unchanged unless the corresponding component is explicitly modified.

Table 7: Hyperparameters of IDOL.

| Parameter | Value |
| --- | --- |
| Epochs | 30 |
| Image input size | $256\times 1024$ |
| BEV latent resolution | $8\times 8$ |
| Number of BEV queries | 64 |
| Latent feature dimension | 256 |
| Planning horizon | 4 s |
| Waypoint interval | 0.5 s |
| Number of future waypoints | 8 |
| Trajectory-anchor vocabulary size | 256 |
| Closed-loop refinement iterations | 1 |
| Learning rate | $2\times 10^{-4}$ |
| Weight decay | $1\times 10^{-4}$ |

##### Inverse dynamics model.

IDOL adopts a lightweight MLP-based inverse dynamics model to convert adjacent imagined BEV states into motion-aware transition cues. Given two consecutive latent BEV states, the model first forms a transition representation by concatenating their query features along the channel dimension. A three-stage MLP transition encoder is then applied: an initial fusion stage compresses the paired-state representation into the latent feature space, followed by residual MLP transformations that refine query-wise transition patterns while preserving the original spatial layout. The resulting query-wise representation can be viewed as a *spatial dynamics map*, which captures local motion-relevant changes across BEV queries. IDOL further aggregates this spatial dynamics map into a compact *global dynamics feature*, which serves as a trajectory-level motion condition for modulating the anchor-conditioned ego planning feature through an MLN layer before trajectory offset prediction. This design keeps the inverse dynamics branch lightweight while allowing the planner to exploit both local transition patterns and global motion context from adjacent imagined futures.

##### Loss configuration.

IDOL is trained with trajectory offset regression, reward supervision, BEV semantic supervision, and IDM motion-feature correction. Table 8 lists the loss coefficients, which remain fixed throughout training.

Table 8: Training losses and their coefficients.

| Loss term | Role | Coefficient |
| --- | --- | --- |
| $\mathcal{L}_{\mathrm{off}}$ | Trajectory offset regression | $1.0$ |
| $\mathcal{L}_{\mathrm{off\text{-}im}}$ | Auxiliary imitation-reward supervision | $0.1$ |
| $\mathcal{L}_{\mathrm{im}}$ | Main imitation-reward supervision | $1.0$ |
| $\mathcal{L}_{\mathrm{sim}}$ | Simulator-metric reward supervision | $1.0$ |
| $\mathcal{L}_{\mathrm{map}}^{\mathrm{cur}}$ | Current BEV semantic supervision | $10.0$ |
| $\mathcal{L}_{\mathrm{map}}^{\mathrm{fut}}$ | Future BEV semantic supervision | $0.1$ |
| $\mathcal{L}_{\mathrm{corr}}$ | IDM motion-feature correction | $0.2$ |

The offset loss $\mathcal{L}_{\mathrm{off}}$ applies L1 regression to the trajectory correction of the anchor closest to the expert trajectory under winner-take-all assignment. The imitation losses $\mathcal{L}_{\mathrm{off\text{-}im}}$ and $\mathcal{L}_{\mathrm{im}}$ supervise the auxiliary and main reward heads using a soft target distribution derived from anchor-to-expert distances. The simulator loss $\mathcal{L}_{\mathrm{sim}}$ sums binary cross-entropy losses over five cached planning metrics. The current and future BEV losses supervise semantic scene representations using focal loss with $\alpha=0.5$ and $\gamma=2$.

Finally, downstream planning losses supervise the IDM indirectly, leaving the motion information encoded in its features underconstrained. We introduce $\mathcal{L}_{\mathrm{corr}}$ to explicitly encourage the final global IDM feature to encode the motion adjustment from a candidate anchor toward the expert trajectory. For a trajectory $\tau$, we define the fixed motion representation

$$
E_{\mathrm{mot}}(\tau)=\left[v_{1:S}/\sigma_{v},\;a_{1:S-1}/\sigma_{a},\;\omega_{1:S}/\sigma_{\omega}\right],
$$

where $v_{s}$ is the signed displacement projected onto the midpoint heading and divided by $\Delta t$, $a_{s}=(v_{s+1}-v_{s})/\Delta t$, and $\omega_{s}$ is the wrapped heading increment divided by $\Delta t$. We prepend the current ego pose and use $\Delta t=0.5\,\mathrm{s}$, $S=8$, and normalization scales $(\sigma_{v},\sigma_{a},\sigma_{\omega})=(10\,\mathrm{m/s},3\,\mathrm{m/s^{2}},0.5\,\mathrm{rad/s})$.

Let $k^{\star}$ denote the anchor closest to the expert trajectory under the same winner-take-all assignment as offset regression. The correction loss is

$$
\mathcal{L}_{\mathrm{corr}}=\ell_{\mathrm{SL1}}\!\left(h_{\mathrm{mot}}(g_{k^{\star}}^{(L-1)}),\operatorname{sg}\!\left[E_{\mathrm{mot}}(\tau^{\mathrm{GT}})-E_{\mathrm{mot}}(\tau_{k^{\star}})\right]\right),
$$

where $h_{\mathrm{mot}}$ is an auxiliary MLP head and $\operatorname{sg}$ denotes stop-gradient. The operator $\ell_{\mathrm{SL1}}$ computes masked Smooth L1 loss with unit transition threshold, averaging valid entries within each motion component and then averaging the three components. Invalid targets are excluded, and heading-rate supervision is masked when either the expert or winning-anchor interval speed has magnitude below $0.5\,\mathrm{m/s}$. Only the winning candidate is supervised. This fixed kinematic target provides interpretable supervision for the IDM representation used in query refinement. We use $\lambda_{\mathrm{corr}}=0.2$, and the auxiliary head is discarded at inference.

Table 9: Performance comparison on the NAVSIM v2 navtest split using extended closed-loop metrics. The best and second-best results among learned methods are highlighted in bold and underlined, respectively. Superscripts indicate methods using additional training signals or settings beyond standard imitation learning: <sup>∗</sup> denotes VLA/VLM-based methods, <sup>†</sup> denotes reinforcement learning, reward-guided training, or reinforcement fine-tuning, and <sup>‡</sup> denotes extra synthetic, pseudo-expert, or QA-style data.

| Method | NC $\uparrow$ | DAC $\uparrow$ | DDC $\uparrow$ | TLC $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | LK $\uparrow$ | HC $\uparrow$ | EC $\uparrow$ | EPDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Human Agent | 100.0 | 100.0 | 99.8 | 100.0 | 87.4 | 100.0 | 100.0 | 98.1 | 90.1 | 90.3 |
| Ego Status MLP [^43] | 93.1 | 77.9 | 92.7 | 99.6 | 86.0 | 91.5 | 89.4 | 98.3 | 85.4 | 64.0 |
| TransFuser [^7] | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 |
| Hydra-MDP++ [^33] | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | 81.4 |
| DriveSuprim [^75] | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 | 83.1 |
| ReCogDrive <sup>∗,†</sup> [^39] | 98.3 | 95.2 | 99.5 | 99.8 | 87.1 | 97.5 | 96.6 | 98.3 | 86.5 | 83.6 |
| DiffusionDrive [^44] | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | 84.5 |
| GTRS-Dense + SimScale <sup>‡,†</sup> [^61] | 98.4 | 98.8 | 99.4 | 99.9 | 87.9 | 98.1 | 96.4 | 97.6 | 58.8 | 84.6 |
| World4Drive [^84] | 97.8 | 96.3 | 99.4 | 99.8 | 88.3 | 97.1 | 97.7 | 98.0 | 53.9 | 84.8 |
| Epona [^81] | 97.1 | 95.7 | 99.3 | 99.7 | 88.6 | 96.3 | 97.0 | 98.0 | 67.8 | 85.1 |
| DiffusionDriveV2 <sup>†</sup> [^85] | 97.7 | 96.6 | 99.2 | 99.8 | 88.9 | 97.2 | 96.0 | 97.8 | 91.0 | 85.5 |
| VADv2 [^5] | 98.0 | 98.3 | 99.4 | 99.8 | 87.1 | 96.8 | 95.2 | 98.3 | 88.1 | 85.8 |
| DriveVLA-W0 <sup>∗</sup> [^37] | 98.5 | 99.1 | 98.0 | 99.7 | 86.4 | 98.1 | 93.2 | 97.9 | 58.9 | 86.1 |
| SGDrive <sup>∗,‡</sup> [^31] | 98.6 | 94.3 | 99.5 | 99.9 | 86.0 | 97.9 | 96.1 | 98.3 | 85.9 | 86.2 |
| DiffRefiner [^76] | 98.5 | 97.4 | 99.6 | 99.8 | 87.6 | 97.7 | 97.7 | 98.3 | 86.2 | 86.2 |
| WorldRFT <sup>†</sup> [^74] | 97.8 | 96.5 | 99.5 | 99.8 | 88.5 | 97.0 | 97.4 | 98.1 | 69.1 | 86.7 |
| SafeDrive [^29] | 99.5 | 99.0 | 99.7 | 99.9 | 88.6 | 98.9 | 97.5 | 98.2 | 81.9 | 87.5 |
| MeanFuser [^64] | 98.3 | 97.2 | 99.6 | 99.8 | 87.6 | 97.4 | 97.3 | 98.3 | 88.2 | 89.5 |
| IDOL | 98.8 | 97.6 | 99.5 | 99.8 | 87.1 | 98.3 | 96.3 | 98.3 | 85.5 | 89.6 |

## Appendix C Benchmark and metrics details

##### Benchmark.

All models are trained on the NAVSIM training split. We evaluate IDOL on the NAVSIM benchmark. The navtest split is the standard test split used to evaluate closed-loop planning performance under common urban driving scenarios. The navhard split is a more challenging subset designed to stress-test planners under difficult cases such as complex interactions, stricter route following, and long-tail driving conditions. For NAVSIM v2, navhard is evaluated under the pseudo-simulation protocol, including the stage-1-only setting and the two-stage setting. The former measures the quality of the initially generated trajectory, while the latter further evaluates planning after pseudo-simulation rollout and therefore reflects closed-loop robustness more directly. Since point-cloud inputs are unavailable in stage 2, we separately train a camera-based IDOL variant using the Latent TransFuser (LTF) formulation. Specifically, the LiDAR input is replaced by a learnable latent tensor initialized from an LTF checkpoint and jointly optimized with the IDOL network on the NAVSIM training split, while retaining the fusion architecture and downstream planning modules. The resulting IDOL-LTF checkpoint is used throughout the two-stage evaluation; the model does not switch modalities between stages or rely on dynamic modality-dropout training.

##### PDMS.

For NAVSIM v1, we report the Predictive Driver Model Score (PDMS), which evaluates trajectory quality after non-reactive simulation. PDMS uses no-collision (NC) and drivable area compliance (DAC) as multiplicative safety constraints, and combines ego progress (EP), time-to-collision within bound (TTC), and comfort (C) as weighted quality terms:

$$
\mathrm{PDMS}=\mathrm{NC}\cdot\mathrm{DAC}\cdot\frac{5\mathrm{EP}+5\mathrm{TTC}+2\mathrm{C}}{12}.
$$

All component metrics are normalized to $[0,1]$, and the final reported PDMS is averaged over all evaluated scenarios and expressed as a percentage.

##### EPDMS.

For NAVSIM v2, we report the Extended Predictive Driver Model Score (EPDMS). Compared with PDMS, EPDMS adds driving direction compliance (DDC) and traffic light compliance (TLC) as multiplicative rule-compliance constraints, and further introduces lane keeping (LK), history comfort (HC), and extended comfort (EC) into the weighted quality term. Following the NAVSIM v2 metric definition, we write the filtered value of each metric as $\tilde{m}$, where human-trajectory violations are filtered to avoid false-positive penalties. EPDMS is computed as

$$
\mathrm{EPDMS}=\tilde{\mathrm{NC}}\cdot\tilde{\mathrm{DAC}}\cdot\tilde{\mathrm{DDC}}\cdot\tilde{\mathrm{TLC}}\cdot\frac{5\tilde{\mathrm{EP}}+5\tilde{\mathrm{TTC}}+2\tilde{\mathrm{LK}}+2\tilde{\mathrm{HC}}+2\tilde{\mathrm{EC}}}{16}.
$$

EPDMS therefore extends PDMS from basic safety and progress evaluation to stricter rule compliance, lane keeping, and pseudo-closed-loop consistency, making it more suitable for NAVSIM v2 and the challenging navhard split.

## Appendix D Additional evaluation on NAVSIM v2 and Bench2Drive

##### Interactive closed-loop evaluation on Bench2Drive.

To evaluate performance beyond NAVSIM, we additionally test IDOL on Bench2Drive, a CARLA-based benchmark comprising 220 routes and 44 safety-critical interactive scenarios. Following the official evaluation protocol, we report Driving Score (DS) and Success Rate (SR) to assess driving performance and task completion under continuous interaction with the environment.

As shown in Table 11, IDOL uses a ResNet-34 image backbone and achieves a DS of 63.22 and an SR of 34.09%, outperforming the listed planners, including the future-BEV-based methods SeerDrive and WoTE. Compared with WoTE, IDOL improves DS by 1.51 points and SR by 2.73 percentage points. These results provide further evidence that inverse-dynamics-guided future reasoning supports effective planning and task completion under interactive closed-loop evaluation beyond NAVSIM.

Table 10: Performance comparison on the NAVSIM v2 navhard split under the stage-1-only protocol. Bold and underlined values indicate the best and second-best results, respectively, for each metric. <sup>†</sup> indicates results reported with the V2-99 backbone.

| Method | NC $\uparrow$ | DAC $\uparrow$ | DDC $\uparrow$ | TLC $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | LK $\uparrow$ | HC $\uparrow$ | EC $\uparrow$ | EPDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TransFuser [^7] | 96.3 | 74.6 | 98.4 | 99.3 | 82.9 | 93.7 | 92.7 | 97.5 | 78.2 | 60.5 |
| LTF [^7] | 96.2 | 79.5 | 99.1 | 99.5 | 84.1 | 95.1 | 94.2 | 97.5 | 79.1 | 62.3 |
| DiffusionDrive [^44] | 95.9 | 84.0 | 98.6 | 99.8 | 84.4 | 96.0 | 95.1 | 97.6 | 71.1 | 63.2 |
| WoTE [^38] | 97.4 | 88.2 | 97.8 | 99.3 | 82.7 | 96.4 | 90.9 | 97.3 | 68.0 | 66.7 |
| Hydra-MDP <sup>†</sup> [^40] | 97.6 | 96.4 | 99.2 | 99.3 | 80.2 | 96.9 | 94.9 | 97.8 | 58.7 | 73.1 |
| IDOL | 97.2 | 89.6 | 98.0 | 99.6 | 82.3 | 96.9 | 95.3 | 97.6 | 72.9 | 76.2 |

Table 11: Interactive closed-loop evaluation on Bench2Drive. Higher Driving Score (DS) and Success Rate (SR) indicate better performance. Bold values denote the best results among the listed methods.

| Method | Driving Score (DS) $\uparrow$ | Success Rate (SR) $\uparrow$ |
| --- | --- | --- |
| VAD [^27] | 42.35 | 15.00% |
| UniAD-Base [^21] | 45.81 | 16.36% |
| SeerDrive [^79] | 58.32 | 30.17% |
| WoTE [^38] | 61.71 | 31.36% |
| IDOL | 63.22 | 34.09% |

## Appendix E Additional ablation studies

We further examine whether learned inverse dynamics provides better planning guidance than simpler future-state representations, whether its gains arise from additional model capacity, how global inverse dynamics features should be fused with the refined planning query, and how performance varies with the trajectory-anchor vocabulary size.

##### Ablation on transition-aware future modeling.

We compare three representations for planning refinement: imagined future BEV states without explicit transition modeling, direct latent differences between adjacent future states, and IDM features learned from ordered pairs of adjacent states. To control for model capacity, we additionally apply parameter-matched MLPs to the future-state-only and latent-difference representations. Both augmented variants have exactly the same total parameter count as IDOL. Across these comparisons, the world model, spatial cross-attention, global modulation, iterative refinement, trajectory anchors, losses, and training settings remain unchanged.

As shown in Table 12, latent differences improve PDMS from 88.6 to 89.0 over future states alone, supporting the value of transition information. Adding parameter-matched MLPs yields only modest gains of 0.3 and 0.1 points, respectively. IDM achieves 90.0 PDMS, exceeding the two parameter-matched variants by 1.1 and 0.9 points. These results indicate that additional parameters alone do not explain the improvement. While a future-state MLP operates on individual states and latent subtraction supplies a fixed, coordinate-wise change signal, IDM jointly processes ordered state pairs, preserving endpoint context and learning nonlinear, state-conditioned transition cues for planning refinement.

Table 12: Ablation of transition-aware future modeling on the NAVSIM navtest split. Parameter-matched MLP variants have the same total parameter count as IDOL; all other components and training settings are fixed. Parameter counts are reported in millions and rounded to one decimal place.

| Transition Modeling | Params (M) | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf. $\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Future State Only | 68.2 | 98.4 | 96.6 | 95.2 | 100 | 82.6 | 88.6 |
| Future State Only + Parameter-Matched MLP | 69.4 | 98.5 | 97.0 | 94.7 | 100 | 83.3 | 88.9 |
| Latent Difference | 68.2 | 98.6 | 96.8 | 95.7 | 100 | 82.8 | 89.0 |
| Latent Difference + Parameter-Matched MLP | 69.4 | 98.7 | 97.1 | 95.8 | 100 | 82.6 | 89.1 |
| IDM | 69.4 | 98.7 | 97.7 | 95.5 | 100 | 84.0 | 90.0 |

##### Ablation on global dynamics fusion.

We further study how to fuse the global dynamics feature produced by the inverse dynamics model. In the inverse-dynamics-guided refinement network, the global dynamics feature provides holistic calibration for the query refined by spatial transition evidence. We compare three simple fusion strategies: additive fusion, which directly injects the projected global feature into the query; concat-MLP fusion, which learns a nonlinear fusion from the concatenated query and global feature; and the MLN-based fusion used in IDOL. As shown in Table 13, all variants obtain strong performance, indicating that the global inverse-dynamics cue is consistently beneficial for query refinement. These results verify that IDOL’s performance is not only due to using inverse-dynamics features, but also benefits from an effective query-calibration mechanism that better integrates global transition information into planning.

Table 13: Ablation of global dynamics fusion strategies in the inverse-dynamics-guided refinement network on the NAVSIM navtest split. All variants use the same spatial branch and only replace the fusion operation between the refined ego query and the global dynamics feature.

| Fusion Strategy | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf. $\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| Additive | 98.7 | 97.2 | 95.7 | 100 | 83.5 | 89.5 |
| Concat-MLP | 98.7 | 97.6 | 95.5 | 100 | 83.8 | 89.8 |
| MLN | 98.7 | 97.7 | 95.5 | 100 | 84.0 | 90.0 |

##### Sensitivity to anchor-set size.

We evaluate IDOL with $K=128$, $K=256$, and $K=512$ trajectory anchors, keeping the anchor-construction procedure, backbone, world model, IDM, losses, and training settings unchanged. As shown in Table 14, PDMS ranges from 89.6 to 90.0, indicating limited sensitivity to anchor-set size within the tested range. A smaller vocabulary may limit trajectory coverage, whereas a denser vocabulary may introduce redundant candidates and increase ranking difficulty. We retain $K=256$ as the default, as it achieves the highest PDMS while using fewer candidates than $K=512$.

Table 14: Sensitivity to trajectory-anchor vocabulary size on the NAVSIM navtest split. Only the number of anchors is varied; the anchor-construction procedure and all other model and training settings remain unchanged.

| Number of Anchors | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf. $\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 98.7 | 97.6 | 95.5 | 100 | 83.8 | 89.8 |
| 256 | 98.7 | 97.7 | 95.5 | 100 | 84.0 | 90.0 |
| 512 | 98.7 | 97.2 | 95.3 | 100 | 84.0 | 89.6 |

## Appendix F Additional qualitative visualization

To further demonstrate the effectiveness of our proposed method, we expand the qualitative visualization experiments with additional representative driving scenarios. Figure 4 presents representative visualization results on the NAVSIM navtest split under three driving maneuvers, including straight driving, left turn, and right turn. Each example contains four panels that illustrate how the proposed IDM module extracts spatial dynamics from latent BEV transitions and uses them to refine the trajectory prediction.

Panel (a) shows the latent BEV change between two adjacent imagined BEV states produced by the world model. The highlighted regions indicate where the scene representation changes most significantly from frame $t$ to frame $t+1$. Therefore, this panel serves as the reference evidence for spatially localized future dynamics, such as ego-motion-induced changes, nearby traffic participants, and lane-level scene evolution.

Panel (b) visualizes the spatial dynamics inferred by the IDM from the same adjacent BEV pair. Instead of directly displaying a semantic map, this panel shows the inverse-dynamics response over BEV queries. The white contour marks the high-change regions extracted from Panel (a), and is overlaid as a reference for comparing IDM responses with latent BEV transitions. The regions with stronger activation correspond to locations where IDM considers the transition between the two BEV states informative for explaining the underlying motion. The visual consistency between Panel (a) and Panel (b) suggests that IDM is not responding to arbitrary static background features, but is sensitive to the spatial regions where the latent scene actually evolves.

Panel (c) further shows the attention distribution from the trajectory query to the IDM spatial dynamics. While Panel (b) indicates where inverse-dynamics information is extracted, Panel (c) reveals which dynamic regions are actually attended to by the refinement query. The concentration of attention around the changing BEV regions demonstrates that the refinement process actively uses the IDM-derived spatial evidence, rather than treating IDM as an isolated auxiliary branch.

Panel (d) compares the original trajectory, the expert trajectory, and the refined trajectory. The original trajectory is decoded before applying the IDM-based closed-loop refinement, while the refined trajectory is decoded after the query has absorbed inverse-dynamics information through the IDM attention and fusion modules. Across straight, left-turn, and right-turn cases, the refined trajectory is visually better aligned with the expert trajectory, showing that the IDM-guided update helps correct the initial prediction toward a more plausible future motion.

Overall, these qualitative results provide direct visual evidence for the effectiveness of the IDM module. The latent BEV transition identifies where the future scene changes, the IDM spatial response recovers dynamics from those changing regions, and the query attention confirms that such information is used by the planner. The resulting trajectory refinement illustrates the practical value of this mechanism: IDM helps the model focus on motion-relevant spatial changes and improves the final trajectory prediction in different driving maneuvers. This supports our hypothesis that inverse-dynamics reasoning provides a meaningful bridge between latent world-model prediction and trajectory-level decision refinement.

![[straight2.png|Refer to caption]]

Figure 4: Qualitative visualization of inverse-dynamics refinement under three representative driving maneuvers on the NAVSIM navtest split. From top to bottom: straight driving, left turn, and right turn. In each row, panels (a)–(d) show the latent BEV change, the IDM dynamics inferred from adjacent BEV states, the IDM attention on dynamic queries, and the resulting refined trajectory, respectively.

## Appendix G Failure Cases

We further analyze two representative failure cases on navtest, as shown in Figure 5. Each row contains the front-view camera image and the corresponding BEV trajectory comparison. The refined trajectory denotes the final prediction after IDM-based refinement, and the expert trajectory is used as the reference.

In the first case, the ego vehicle approaches a turning area with ambiguous road geometry. The refined trajectory deviates clearly from the expert trajectory and follows an incorrect spatial direction. This indicates that when the model fails to infer the correct maneuver intention, IDM refinement alone may not fully recover the desired path.

In the second case, the scene involves a complex intersection or roundabout-like structure. Although the refined trajectory captures part of the motion trend, it remains spatially offset from the expert trajectory. This suggests that topology-sensitive scenarios still require more precise reasoning about lane connectivity, road curvature, and maneuver timing.

These cases show that IDM provides useful dynamic cues, but its effectiveness depends on whether such cues can be correctly fused with scene topology and trajectory intent. Future work may improve these limitations with stronger topology-aware constraints and uncertainty-aware refinement.

![[figure_failure_cases.png|Refer to caption]]

Figure 5: Representative failure cases on navtest.

## Appendix H Limitations and future work

Although IDOL shows strong performance across NAVSIM benchmarks, several limitations remain.

##### Model capacity.

Our inverse dynamics model is designed to be lightweight, which keeps the framework efficient but may limit its ability to capture more complex long-horizon interactions and subtle motion patterns. Future work could scale the IDM with larger temporal contexts, stronger transition modeling, or pretraining on broader driving sequences.

##### Dependence on latent future prediction.

IDOL relies on the quality of the imagined latent BEV futures produced by the world model. When future prediction is inaccurate in highly ambiguous or rare scenarios, the decoded inverse-dynamics cues may also become less reliable. Improving the robustness of latent world modeling is therefore an important direction.

##### Fixed refinement strategy.

The current closed-loop refinement uses a fixed refinement setting, while different scenes may require different levels of future-aware reasoning. Adaptive refinement strategies could further improve stability and avoid unnecessary iterative updates.

##### Broader evaluation.

Although our experiments cover both standard and challenging NAVSIM settings, future work will further evaluate inverse-dynamics-guided planning in more interactive, long-tail, and real-world driving scenarios.

[^1]: J. Angeles, O. Ma, and A. Rojas An algorithm for the inverse dynamics of n-axis general manipulators using kane’s equations. Computers & Mathematics with Applications 17 (12), pp. 1545–1561. Cited by: §2.3.

[^2]: H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom Nuscenes: a multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 11621–11631. Cited by: §1.

[^3]: W. Cao, M. Hallgarten, T. Li, D. Dauner, X. Gu, C. Wang, Y. Miron, M. Aiello, H. Li, I. Gilitschenski, et al. Pseudo-simulation for autonomous driving. arXiv preprint arXiv:2506.04218. Cited by: §4.1.

[^4]: L. Chen, P. Wu, K. Chitta, B. Jaeger, A. Geiger, and H. Li End-to-end autonomous driving: challenges and frontiers. IEEE Transactions on Pattern Analysis and Machine Intelligence 46 (12), pp. 10164–10183. Cited by: §2.1.

[^5]: S. Chen, B. Jiang, H. Gao, B. Liao, Q. Xu, Q. Zhang, C. Huang, W. Liu, and X. Wang Vadv2: end-to-end vectorized autonomous driving via probabilistic planning. arXiv preprint arXiv:2402.13243. Cited by: Table 9, §1, §3.2, Table 1.

[^6]: Y. Chen, Y. Wang, and Z. Zhang Drivinggpt: unifying driving world modeling and planning with multi-modal autoregressive transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 26890–26900. Cited by: §2.2.

[^7]: K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger Transfuser: imitation with transformer-based sensor fusion for autonomous driving. IEEE transactions on pattern analysis and machine intelligence 45 (11), pp. 12878–12895. Cited by: Table 9, Table 10, Table 10, §2.1, §3.2, Table 1.

[^8]: D. Dauner, M. Hallgarten, A. Geiger, and K. Chitta Parting with misconceptions about learning-based vehicle motion planning. In Conference on Robot Learning, pp. 1268–1281. Cited by: Table 2.

[^9]: D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, et al. Navsim: data-driven non-reactive autonomous vehicle simulation and benchmarking. Advances in Neural Information Processing Systems 37, pp. 28706–28719. Cited by: §1, §4.1.

[^10]: L. Feng, Y. Gao, E. Zablocki, Q. Li, W. Li, S. Liu, M. Cord, and A. Alahi Rap: 3d rasterization augmented end-to-end planning. arXiv preprint arXiv:2510.04333. Cited by: §2.1.

[^11]: R. Feng, N. Xi, D. Chu, R. Wang, Z. Deng, A. Wang, L. Lu, J. Wang, and Y. Huang Artemis: autoregressive end-to-end trajectory planning with mixture of experts for autonomous driving. IEEE Robotics and Automation Letters 11 (1), pp. 226–233. Cited by: §2.1.

[^12]: S. Gao, J. Yang, L. Chen, K. Chitta, Y. Qiu, A. Geiger, J. Zhang, and H. Li Vista: a generalizable driving world model with high fidelity and versatile controllability. Advances in Neural Information Processing Systems 37, pp. 91560–91596. Cited by: §2.2.

[^13]: K. Guo, H. Liu, X. Wu, J. Pan, and C. Lv Ipad: iterative proposal-centric end-to-end autonomous driving. arXiv preprint arXiv:2505.15111. Cited by: §2.1.

[^14]: M. Guo, Y. Yang, C. Han, R. Zhang, S. Li, J. Wan, and Z. Zhang FlowAD: ego-scene interactive modeling for autonomous driving. arXiv preprint arXiv:2603.13399. Cited by: §1.

[^15]: D. Hafner, T. Lillicrap, J. Ba, and M. Norouzi Dream to control: learning behaviors by latent imagination. arXiv preprint arXiv:1912.01603. Cited by: §2.2.

[^16]: D. Hafner, T. Lillicrap, M. Norouzi, and J. Ba Mastering atari with discrete world models. arXiv preprint arXiv:2010.02193. Cited by: §2.2.

[^17]: D. Hafner, J. Pasukonis, J. Ba, and T. Lillicrap Mastering diverse domains through world models. arXiv preprint arXiv:2301.04104. Cited by: §2.2.

[^18]: K. He, X. Zhang, S. Ren, and J. Sun Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778. Cited by: §4.3.

[^19]: J. M. Hollerbach A recursive lagrangian formulation of maniputator dynamics and a comparative study of dynamics formulation complexity. IEEE Transactions on Systems, Man, and Cybernetics 10 (11), pp. 730–736. Cited by: §2.3.

[^20]: A. Hu, L. Russell, H. Yeo, Z. Murez, G. Fedoseev, A. Kendall, J. Shotton, and G. Corrado Gaia-1: a generative world model for autonomous driving. arXiv preprint arXiv:2309.17080. Cited by: §2.2.

[^21]: Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, et al. Planning-oriented autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 17853–17862. Cited by: Table 11, §2.1, Table 1.

[^22]: Z. Huang, B. Zhang, Z. Li, and D. Lin Occdriver: future occupancy guided dual-branch trajectory planner in autonomous driving. In International Conference on Learning Representations, Vol. 2026, pp. 141940–141962. Cited by: §1.

[^23]: Z. Huang, X. Weng, M. Igl, Y. Chen, Y. Cao, B. Ivanovic, M. Pavone, and C. Lv Gen-drive: enhancing diffusion generative driving policies with reward modeling and reinforcement learning fine-tuning. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pp. 3445–3451. Cited by: §2.1.

[^24]: J. Hwang, R. Xu, H. Lin, W. Hung, J. Ji, K. Choi, D. Huang, T. He, P. Covington, B. Sapp, et al. Emma: end-to-end multimodal model for autonomous driving. arXiv preprint arXiv:2410.23262. Cited by: §2.1.

[^25]: A. Jiang, Y. Gao, Z. Sun, Y. Wang, J. Wang, J. Chai, Q. Cao, Y. Heng, H. Jiang, Y. Dong, et al. Diffvla: vision-language guided diffusion planning for autonomous driving. arXiv preprint arXiv:2505.19381. Cited by: §1.

[^26]: A. Jiang, Y. Gao, Y. Wang, Z. Sun, S. Wang, Y. Heng, H. Sun, S. Tang, L. Zhu, J. Chai, et al. Irl-vla: training an vision-language-action policy via reward world model. arXiv preprint arXiv:2508.06571. Cited by: Appendix D.

[^27]: B. Jiang, S. Chen, Q. Xu, B. Liao, J. Chen, H. Zhou, Q. Zhang, W. Liu, C. Huang, and X. Wang Vad: vectorized scene representation for efficient autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 8340–8350. Cited by: Table 11, §1.

[^28]: G. Jiang, Y. Luo, J. Liu, Y. Huang, Y. Zhu, Z. Qu, D. Z. Chen, B. Liu, and X. Yan WPT: world-to-policy transfer via online world model distillation. arXiv preprint arXiv:2511.20095. Cited by: §1, §2.2.

[^29]: J. Kim, J. Oh, S. Yu, H. Shin, D. Kwak, and J. W. Choi SafeDrive: fine-grained safety reasoning for end-to-end driving in a sparse world. arXiv preprint arXiv:2602.18887. Cited by: Table 9, §1.

[^30]: B. Koopman, H. J. Grootenboer, and H. J. De Jongh An inverse dynamics model for the analysis, reconstruction and prediction of bipedal walking. Journal of biomechanics 28 (11), pp. 1369–1376. Cited by: §2.3.

[^31]: J. Li, J. Wu, D. Hu, X. Huang, B. Sun, Z. Hao, X. Lang, X. Zhu, and L. Zhang SGDrive: scene-to-goal hierarchical world cognition for autonomous driving. arXiv preprint arXiv:2601.05640. Cited by: Table 9, §1.

[^32]: J. Li, B. Zhang, X. Jin, J. Deng, X. Zhu, and L. Zhang ImagiDrive: a unified imagination-and-planning framework for autonomous driving. arXiv preprint arXiv:2508.11428. Cited by: §2.1.

[^33]: K. Li, Z. Li, S. Lan, Y. Xie, Z. Zhang, J. Liu, Z. Wu, Z. Yu, and J. M. Alvarez Hydra-mdp++: advancing end-to-end driving via expert-guided hydra-distillation. arXiv preprint arXiv:2503.12820. Cited by: Table 9.

[^34]: P. Li and D. Cui Navigation-guided sparse scene representation for end-to-end autonomous driving. arXiv preprint arXiv:2409.18341. Cited by: §2.1.

[^35]: P. Li, Y. Zheng, Y. Wang, H. Wang, H. Zhao, J. Liu, X. Zhan, K. Zhan, and X. Lang Discrete diffusion for reflective vision-language-action models in autonomous driving. arXiv preprint arXiv:2509.20109. Cited by: §1.

[^36]: Y. Li, L. Fan, J. He, Y. Wang, Y. Chen, Z. Zhang, and T. Tan Enhancing end-to-end autonomous driving with latent world model. arXiv preprint arXiv:2406.08481. Cited by: §2.2, Table 1.

[^37]: Y. Li, S. Shang, W. Liu, B. Zhan, H. Wang, Y. Wang, Y. Chen, X. Wang, Y. An, C. Tang, et al. DriveVLA-w0: world models amplify data scaling law in autonomous driving. arXiv preprint arXiv:2510.12796. Cited by: Table 9, §2.1.

[^38]: Y. Li, Y. Wang, Y. Liu, J. He, L. Fan, and Z. Zhang End-to-end driving with online trajectory evaluation via bev world model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 27137–27146. Cited by: Table 10, Table 11, §1, §2.2, §3.4.2, Table 1, Table 2.

[^39]: Y. Li, K. Xiong, X. Guo, F. Li, S. Yan, G. Xu, L. Zhou, L. Chen, H. Sun, B. Wang, et al. Recogdrive: a reinforced cognitive framework for end-to-end autonomous driving. arXiv preprint arXiv:2506.08052. Cited by: Table 9.

[^40]: Z. Li, K. Li, S. Wang, S. Lan, Z. Yu, Y. Ji, Z. Li, Z. Zhu, J. Kautz, Z. Wu, et al. Hydra-mdp: end-to-end multimodal planning with multi-target hydra-distillation. arXiv preprint arXiv:2406.06978. Cited by: Table 10, §1, §3.2.

[^41]: Z. Li, S. Wang, S. Lan, Z. Yu, Z. Wu, and J. M. Alvarez Hydra-next: robust closed-loop driving with open-loop training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 27305–27314. Cited by: §2.1.

[^42]: Z. Li, W. Yao, Z. Wang, X. Sun, J. Chen, N. Chang, M. Shen, Z. Wu, S. Lan, and J. M. Alvarez Generalized trajectory scoring for end-to-end multimodal planning. arXiv preprint arXiv:2506.06664. Cited by: Table 2.

[^43]: Z. Li, Z. Yu, S. Lan, J. Li, J. Kautz, T. Lu, and J. M. Alvarez Is ego status all you need for open-loop end-to-end autonomous driving?. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14864–14873. Cited by: Table 9.

[^44]: B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang, et al. Diffusiondrive: truncated diffusion model for end-to-end autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 12037–12047. Cited by: Table 9, Table 10, §2.1, Table 1, Table 2.

[^45]: L. Liu, C. Jia, G. Yu, Z. Song, J. Li, F. Jia, P. Wu, X. Hao, and Y. Luo GuideFlow: constraint-guided flow matching for planning in end-to-end autonomous driving. arXiv preprint arXiv:2511.18729. Cited by: Appendix D, §1, Table 2.

[^46]: S. Liu, W. Chen, W. Li, Z. Wang, L. Yang, J. Huang, Y. Zhang, Z. Huang, Z. Cheng, and H. Yang BridgeDrive: diffusion bridge policy for closed-loop trajectory planning in autonomous driving. arXiv preprint arXiv:2509.23589. Cited by: §1.

[^47]: Y. Luo, F. Li, S. Xu, Z. Lai, L. Yang, Q. Chen, Z. Luo, Z. Xie, S. Jiang, J. Liu, et al. Adathinkdrive: adaptive thinking via reinforcement learning for autonomous driving. arXiv preprint arXiv:2509.13769. Cited by: §1, §2.1.

[^48]: L. Nguyen, M. Fauth, B. Jaeger, D. Dauner, M. Igl, A. Geiger, and K. Chitta LEAD: minimizing learner-expert asymmetry in end-to-end driving. arXiv preprint arXiv:2512.20563. Cited by: §2.1.

[^49]: E. Otten Inverse and forward dynamics: models of multi–body systems. Philosophical Transactions of the Royal Society of London. Series B: Biological Sciences 358 (1437), pp. 1493–1500. Cited by: §2.3.

[^50]: A. Prakash, K. Chitta, and A. Geiger Multi-modal fusion transformer for end-to-end autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 7077–7087. Cited by: Table 2.

[^51]: I. Rawal, S. Gupta, Y. Hu, and W. Zhan NoRD: a data-efficient vision-language-action model that drives without reasoning. arXiv preprint arXiv:2602.21172. Cited by: §2.1.

[^52]: S. Shang, Y. Chen, Y. Wang, Y. Li, and Z. ZHANG Drivedpo: policy learning via safety dpo for end-to-end autonomous driving. Advances in Neural Information Processing Systems 38, pp. 81565–81585. Cited by: §2.1.

[^53]: M. Strong, W. Chang, Q. Herau, J. Yang, Y. Hu, C. Peng, and W. Zhan Learning to drive is a free gift: large-scale label-free autonomy pretraining from unposed in-the-wild videos. arXiv preprint arXiv:2602.22091. Cited by: §1.

[^54]: H. Su, W. Wu, F. Song, J. Zhang, Z. Yang, and J. Yan DriveMamba: task-centric scalable state space model for efficient end-to-end autonomous driving. arXiv preprint arXiv:2602.13301. Cited by: §2.1.

[^55]: B. Sun, Y. Cao, Y. Wang, R. Wang, J. Shang, X. Feng, J. Lu, J. Shi, S. Yang, X. Yan, et al. MindDrive: an all-in-one framework bridging world models and vision-language model for end-to-end autonomous driving. arXiv preprint arXiv:2512.04441. Cited by: Appendix D, §2.1.

[^56]: W. Sun, X. Lin, K. Chen, Z. Pei, X. Li, Y. Shi, and S. Zheng Sparsedrivev2: scoring is all you need for end-to-end autonomous driving. In European Conference on Computer Vision, pp. 446–463. Cited by: §2.1.

[^57]: W. Sun, X. Lin, Y. Shi, C. Zhang, H. Wu, and S. Zheng Sparsedrive: end-to-end autonomous driving via sparse scene representation. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pp. 8795–8801. Cited by: §1.

[^58]: Z. Sun, Y. Wang, A. Jiang, S. Wang, Y. Gao, Y. Heng, S. Zhang, A. He, H. Jiang, J. Chai, et al. DiffSemanticFusion: semantic raster bev fusion for autonomous driving via online map diffusion. IEEE Robotics and Automation Letters 11 (3), pp. 2354–2361. Cited by: Appendix D.

[^59]: J. Tang, Z. Zhou, Z. He, J. Zhang, K. Zhang, and J. Pu CausalVAD: de-confounding end-to-end autonomous driving via causal intervention. arXiv preprint arXiv:2603.18561. Cited by: §1.

[^60]: X. Tang, M. Kan, S. Shan, and X. Chen Plan-r1: safe and feasible trajectory planning as language modeling. arXiv preprint arXiv:2505.17659. Cited by: §1.

[^61]: H. Tian, T. Li, H. Liu, J. Yang, Y. Qiu, G. Li, J. Wang, Y. Gao, Z. Zhang, L. Wang, et al. Simscale: learning to drive via real-world simulation at scale. arXiv preprint arXiv:2511.23369. Cited by: Table 9, §1.

[^62]: Y. Tian, S. Yang, J. Zeng, P. Wang, D. Lin, H. Dong, and J. Pang Predictive inverse dynamics models are scalable learners for robotic manipulation. arXiv preprint arXiv:2412.15109. Cited by: §2.3.

[^63]: J. Wang, G. Li, Z. Huang, C. Dang, H. Ye, Y. Han, and L. Chen VGGDrive: empowering vision-language models with cross-view geometric grounding for autonomous driving. arXiv preprint arXiv:2602.20794. Cited by: §2.1.

[^64]: J. Wang, Y. Zheng, X. Liu, Z. Xing, P. Li, G. Li, K. Ma, G. Chen, H. Ye, Z. Xia, et al. MeanFuser: fast one-step multi-modal trajectory generation and adaptive reconstruction via meanflow for end-to-end autonomous driving. arXiv preprint arXiv:2602.20060. Cited by: Table 9, §1, Table 1.

[^65]: S. Wang, Y. Liu, T. Wang, Y. Li, and X. Zhang Exploring object-centric temporal modeling for efficient multi-view 3d object detection. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 3621–3631. Cited by: §3.4.2.

[^66]: X. Wang, Z. Zhu, G. Huang, X. Chen, J. Zhu, and J. Lu Drivedreamer: towards real-world-drive world models for autonomous driving. In European conference on computer vision, pp. 55–72. Cited by: §2.2.

[^67]: Y. Wang, J. He, L. Fan, H. Li, Y. Chen, and Z. Zhang Driving into the future: multiview visual forecasting and planning with world model for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14749–14759. Cited by: §2.2.

[^68]: X. Weng, B. Ivanovic, Y. Wang, Y. Wang, and M. Pavone Para-drive: parallelized architecture for real-time autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15449–15458. Cited by: Table 1.

[^69]: M. K. Wozniak, L. Liu, Y. Cai, and P. Jensfelt PRIX: learning to plan from raw pixels for end-to-end autonomous driving. arXiv preprint arXiv:2507.17596. Cited by: §2.1.

[^70]: T. Xia, Y. Li, L. Zhou, J. Yao, K. Xiong, H. Sun, B. Wang, K. Ma, G. Chen, H. Ye, et al. Drivelaw: unifying planning and video generation in a latent driving world. arXiv preprint arXiv:2512.23421. Cited by: §1, §2.2.

[^71]: Z. Xing, X. Zhang, Y. Hu, B. Jiang, T. He, Q. Zhang, X. Long, and W. Yin Goalflow: goal-driven flow matching for multimodal trajectories generation in end-to-end autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 1602–1611. Cited by: §2.1.

[^72]: Y. Xu, J. Cui, F. Cai, Z. Zhu, H. Shang, S. Luan, M. Xu, N. Zhang, Y. Li, J. Cai, et al. WAM-flow: parallel coarse-to-fine motion planning via discrete flow matching for autonomous driving. arXiv preprint arXiv:2512.06112. Cited by: §1, §2.1.

[^73]: J. Yang, K. Chitta, S. Gao, L. Chen, Y. Shao, X. Jia, H. Li, A. Geiger, X. Yue, and L. Chen Resim: reliable world simulation for autonomous driving. arXiv preprint arXiv:2506.09981. Cited by: §2.3.

[^74]: P. Yang, B. Lu, Z. Xia, C. Han, Y. Gao, T. Zhang, K. Zhan, X. Lang, Y. Zheng, and Q. Zhang WorldRFT: latent world model planning with reinforcement fine-tuning for autonomous driving. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40, pp. 11649–11657. Cited by: Table 9.

[^75]: W. Yao, Z. Li, S. Lan, Z. Wang, X. Sun, J. M. Alvarez, and Z. Wu Drivesuprim: towards precise trajectory selection for end-to-end planning. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40, pp. 11910–11918. Cited by: Table 9, §2.1.

[^76]: L. Yin, R. Ju, G. Guo, and E. Cheng Diffrefiner: coarse to fine trajectory planning via diffusion refinement with semantic interaction for end to end autonomous driving. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40, pp. 12009–12017. Cited by: Table 9.

[^77]: Z. Yuan, C. Qian, J. Tang, R. Chen, Z. Song, L. Sun, X. Chu, Y. Cai, D. Zhang, and S. Li Autodrive-r ${}^{2}$: incentivizing reasoning and self-reflection capacity for vla model in autonomous driving. In International Conference on Learning Representations, Vol. 2026, pp. 97934–97955. Cited by: §2.1.

[^78]: S. Zeng, X. Chang, M. Xie, X. Liu, Y. Bai, Z. Pan, M. Xu, X. Wei, and N. Guo Futuresightdrive: thinking visually with spatio-temporal cot for autonomous driving. arXiv preprint arXiv:2505.17685. Cited by: §2.3.

[^79]: B. Zhang, N. Song, J. Li, X. Zhu, J. Deng, and L. Zhang Future-aware end-to-end driving: bidirectional modeling of trajectory planning and scene evolution. arXiv preprint arXiv:2510.11092. Cited by: Table 11, §1, §2.1, §2.2, §2.2, Table 1.

[^80]: J. Zhang, Z. Fu, Z. Xu, W. Dai, Q. Liu, and Y. Wang ResWorld: temporal residual world model for end-to-end autonomous driving. arXiv preprint arXiv:2602.10884. Cited by: §1, Table 1.

[^81]: K. Zhang, Z. Tang, X. Hu, X. Pan, X. Guo, Y. Liu, J. Huang, L. Yuan, Q. Zhang, X. Long, et al. Epona: autoregressive diffusion world model for autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 27220–27230. Cited by: Table 9.

[^82]: R. Zhao, Y. Fan, Z. Chen, F. Gao, and Z. Gao DiffE2E: rethinking end-to-end driving with a hybrid diffusion-regression-classification policy. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, Cited by: §1, Table 1.

[^83]: W. Zheng, X. Mao, N. Ye, P. Li, K. Zhan, X. Lang, and H. Zhao DriveAgent-r1: advancing vlm-based autonomous driving with active perception and hybrid thinking. arXiv preprint arXiv:2507.20879. Cited by: §2.1.

[^84]: Y. Zheng, P. Yang, Z. Xing, Q. Zhang, Y. Zheng, Y. Gao, P. Li, T. Zhang, Z. Xia, P. Jia, et al. World4drive: end-to-end autonomous driving via intention-aware physical latent world model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 28632–28642. Cited by: Table 9, §2.2.

[^85]: J. Zou, S. Chen, B. Liao, Z. Zheng, Y. Song, L. Zhang, Q. Zhang, W. Liu, and X. Wang DiffusionDriveV2: reinforcement learning-constrained truncated diffusion modeling in end-to-end autonomous driving. arXiv preprint arXiv:2512.07745. Cited by: Table 9.
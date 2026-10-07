---
title: "ForeDrive: Foresight-Guided End-to-End Autonomous Driving with a Planning-Relevant Latent World Model"
source: "https://arxiv.org/html/2609.26299v1"
author:
published:
created: 2026-10-06
description:
tags:
  - "clippings"
---
Sinuo Wang    Zichong Gu <sup>1</sup>    Yuhan Huang    Wenxin Wen    Xun Yang    Yiqing Zhang    Xingyu Zhang    Ningyu Che    Jie Ling    Qiankun Yu    Wei Liu    Jing Xu    Xinggang Wang

###### Abstract

Existing latent world models are typically optimized for future predictability, yet the resulting representations are not necessarily useful for planning in autonomous driving. Predictions are commonly used for pretraining or auxiliary supervision rather than as direct conditioning signals for trajectory generation. We propose ForeDrive, which learns a planning-relevant latent representation and couples it asymmetrically to a Diffusion Transformer (DiT) planner. The planner consumes multi-horizon latent future representations learned with a JEPA-style world model; planning gradients update the shared online encoder, while stop-gradient routing trains the latent predictor with forecasting losses only. Because predicted futures have varying reliability across horizons and BEV trajectories are misaligned with image tokens, we use gated visual fusion, future-status injection, and Trajectory-Adaptive Bias (TAB) to inject future latents as guidance without overriding the current observation. Trained with pure imitation learning and using only the current front-view image as visual input at inference, ForeDrive attains 89.9 PDMS on NAVSIM v1 and 90.0 one-stage EPDMS on NAVSIM v2, without reinforcement learning or an external trajectory scorer.

<sup>1</sup> Huazhong University of Science and Technology

<sup>2</sup> Shanghai Zaofu Intelligent Technology Co., Ltd.

## 1 Introduction

End-to-end driving planners must both anticipate how a scene may evolve and generate a trajectory that covers multiple maneuvers. Latent world models [^1] [^36] [^10] [^2] [^22] predict future latent representations without RGB reconstruction, enabling efficient semantic foresight. Diffusion Transformer (DiT) planners such as DiffusionDrive [^17] generate multimodal trajectories under imitation learning. Forecast accuracy, however, does not guarantee planning usefulness, as predicted latents may omit decision-critical information. Conversely, current-conditioned planners generate trajectories without explicit future representations.

In existing driving systems, predicted future latent representations often provide pretraining signals, auxiliary losses, or features for candidate evaluation rather than direct conditioning signals for trajectory generation [^11] [^35]. Methods that do condition planning on predicted futures commonly introduce structured scene prediction, pixel-level generation, or staged optimization [^13] [^32] [^26]. Few methods jointly learn a latent that is useful for planning and couple it to a generative planner while preventing planning gradients from directly rewriting the predictor (Figure 1).

![[Fig1.png|Refer to caption]]

Figure 1: Representation paradigms for end-to-end planning. (a) Reactive Planner: action generation from the current latent only (e.g., DiffusionDrive, MeanFuser). (b) Latent World Model: predicted futures mainly for pretraining or auxiliary losses (e.g., LAW, Drive-JEPA). (c) Ours: joint WM and trajectory supervision shapes the shared encoder, and futures guide the planner.

ForeDrive jointly learns planning-relevant latent representations and integrates multi-horizon future latents as complementary multi-scale future context into a DiT planner through asymmetric latent optimization and current-anchored fusion. Because predicted futures have varying reliability across horizons, ForeDrive treats them as complementary guidance anchored by reliable current observations, rather than allowing futures to dominate current perception. A JEPA-style online/EMA predictor estimates multi-horizon visual and ego-state latents without pixel reconstruction or a separate training stage.

We evaluate ForeDrive on NAVSIM under a camera-only, pure imitation-learning protocol. It attains 89.9 PDMS on v1 and 90.0 one-stage EPDMS on v2, outperforming recent end-to-end (E2E) and world-model planners under the same protocol. A zero-shot transfer to nuScenes tests cross-dataset generalization. Ablations attribute the gains mainly to future consumption, current-primary fusion, and asymmetric encoder updates.

Our main contributions are summarized as follows:

- We propose ForeDrive, which learns a planning-relevant latent representation and asymmetrically couples multi-horizon latent prediction with trajectory generation. Planning updates the shared encoder while stop-gradient routing prevents planning gradients from updating the latent predictor, reducing prediction–planning gradient interference.
- We introduce planning-oriented interfaces, including gated visual fusion, future-status injection, and Trajectory-Adaptive Bias (TAB), that incorporate predicted future dynamics into diffusion planning. These designs keep the current observation as primary evidence, and TAB links each trajectory candidate to the image tokens along its projected path during denoising.
- We validate ForeDrive on NAVSIM v1 and v2 under a camera-only, pure imitation-learning protocol (89.9 PDMS / 90.0 EPDMS). Using only a single front-view image at inference, ForeDrive establishes a new state-of-the-art among imitation-learning methods.

## 2 Related Work

Figure 2: NAVSIM performance comparison under different model scales (EPDMS vs. PDMS). Only methods with reported total model parameters and both scores are shown. Marker area indicates model parameter scale (area $\propto\log$ parameter count).

End-to-End Autonomous Driving. End-to-end driving maps sensor observations directly to planned trajectories. Early systems such as TransFuser and UniAD emphasize multi-sensor fusion and BEV-centric perception–planning [^4] [^8] [^16], while recent work increasingly adopts camera-only inputs [^17] [^20] [^25]. Beyond imitation learning, several high-scoring methods further apply reinforcement-learning post-training, as in ReCogDrive-RL [^14], or rule-based candidate scoring, as in Hydra-MDP and DriveSuprim [^15] [^29]. These stages raise benchmark scores, but the gains of a pure imitation-learning planner without RL or external scorers remain less clear. ForeDrive therefore adopts a camera-only, pure-IL setting and examines whether a planning-relevant foresight representation improves the sensor-to-plan model without RL post-training or auxiliary scorers.

Diffusion-based Planning. Diffusion models are widely used for multimodal trajectory generation in end-to-end driving, as iterative denoising can represent multiple futures and trajectory uncertainty under imitation learning. DiffusionDrive combines truncated diffusion with trajectory anchors; DiffRefiner [^30], MeanFuser, and GoalFlow [^27] further develop coarse-to-fine, one-step, and flow-matching variants. These planners are typically conditioned on current or short-history features and do not condition on an explicit predicted future for planning. ForeDrive retains anchor-based diffusion decoding and feeds multi-horizon latent predictions as complementary guidance under asymmetric coupling.

World Models for Driving. Prior work couples foresight and driving in four ways. (i) Predictive world models with structured scene forecasting use BEV or occupancy futures for planning [^6] [^23] [^34] [^3] [^13] [^35]. (ii) JEPA-style latent predictors such as Drive-JEPA and LAW [^22] [^11] mainly treat predicted latents as pretraining or auxiliary signals rather than as inputs to a generative planner. (iii) Video-prediction approaches such as Epona and DriveLaW [^32] [^26] condition trajectory DiTs on generated video features, but rely on pixel generation and, for DriveLaW, multi-stage freezing. (iv) Unconstrained planning-conditioned prediction allows planning objectives to reshape the foresight module without isolating forecast supervision [^21] [^12] [^33]. ForeDrive instead learns planning-relevant latents and couples them to a generative planner under asymmetric optimization, keeping current evidence primary.

## 3 Method

In this section, we present ForeDrive (Figure 3). We define a planning-relevant latent as a future representation that is grounded by predictive supervision while retaining information useful for downstream trajectory generation. These objectives introduce a trade-off because forecasting favors target alignment, while planning benefits from decision-sensitive information. ForeDrive mitigates this trade-off by learning such latents with a JEPA-style world model, injecting them into a DiT planner as complementary guidance through planning-oriented interfaces (gated fusion, future-status injection, and TAB), and applying asymmetric latent optimization. Implementation details, architectural configurations, and hyperparameters are provided in the supplementary material.

![[Fig3.png|Refer to caption]]

Figure 3: Overview of ForeDrive. Multi-horizon future latents guide an anchor-based DiT via gated fusion, future-status injection, and TAB; stop-gradient routing updates the shared encoder while isolating the predictor.

<table><tbody><tr><th>Method</th><th>Venue</th><th>Mod.</th><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Comf.<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><th colspan="9">Traditional end-to-end methods</th></tr><tr><th>UniAD <sup><a href="#fn:8">8</a></sup></th><th>CVPR’23</th><th>C</th><td>97.8</td><td>91.9</td><td>92.9</td><td>100.0</td><td>78.8</td><td>83.4</td></tr><tr><th>TransFuser <sup><a href="#fn:4">4</a></sup></th><th>TPAMI’23</th><th>C+L</th><td>97.7</td><td>92.8</td><td>92.8</td><td>100.0</td><td>79.2</td><td>84.0</td></tr><tr><th>PARA-Drive <sup><a href="#fn:24">24</a></sup></th><th>CVPR’24</th><th>C</th><td>97.9</td><td>92.4</td><td>93.0</td><td>99.8</td><td>79.3</td><td>84.0</td></tr><tr><th>DRAMA <sup><a href="#fn:31">31</a></sup></th><th>ISRR’24</th><th>C+L</th><td>98.0</td><td>93.1</td><td>94.8</td><td>100.0</td><td>80.1</td><td>85.5</td></tr><tr><th>ReCogDrive-IL <sup><a href="#fn:14">14</a></sup></th><th>ICLR’26</th><th>C</th><td>98.1</td><td>94.7</td><td>94.2</td><td>100.0</td><td>80.9</td><td>86.5</td></tr><tr><th>PRIX <sup><a href="#fn:25">25</a></sup></th><th>RA-L’26</th><th>C</th><td>98.1</td><td>96.3</td><td>94.1</td><td>100.0</td><td>82.3</td><td>87.8</td></tr><tr><th>DiffusionDrive <sup><a href="#fn:17">17</a></sup></th><th>CVPR’25</th><th>C+L</th><td>98.2</td><td>96.2</td><td>94.7</td><td>100.0</td><td>82.2</td><td>88.1</td></tr><tr><th>MeanFuser <sup><a href="#fn:20">20</a></sup></th><th>CVPR’26</th><th>C</th><td>98.6</td><td>97.0</td><td>95.0</td><td>100.0</td><td>82.8</td><td>89.0</td></tr><tr><th>DiffRefiner-R34 <sup><a href="#fn:30">30</a></sup></th><th>AAAI’26</th><th>C</th><td>98.4</td><td>97.4</td><td>95.3</td><td>100.0</td><td>83.4</td><td>89.4</td></tr><tr><th colspan="9">World-model and video–action methods</th></tr><tr><th>LAW <sup><a href="#fn:11">11</a></sup></th><th>ICLR’25</th><th>C</th><td>96.4</td><td>95.4</td><td>88.7</td><td>99.9</td><td>81.7</td><td>84.6</td></tr><tr><th>Epona <sup><a href="#fn:32">32</a></sup></th><th>ICCV’25</th><th>C</th><td>97.9</td><td>95.1</td><td>93.8</td><td>99.9</td><td>80.4</td><td>86.2</td></tr><tr><th>DriveVLA-W0 <sup><a href="#fn:12">12</a></sup></th><th>ICLR’26</th><th>C</th><td>98.4</td><td>95.3</td><td>95.2</td><td>100.0</td><td>80.9</td><td>87.2</td></tr><tr><th>PWM <sup><a href="#fn:33">33</a></sup></th><th>NeurIPS’25</th><th>C</th><td>98.6</td><td>95.9</td><td>95.4</td><td>100.0</td><td>81.8</td><td>88.1</td></tr><tr><th>WoTE <sup><a href="#fn:13">13</a></sup></th><th>ICCV’25</th><th>C+L</th><td>98.5</td><td>96.8</td><td>94.9</td><td>99.9</td><td>81.9</td><td>88.3</td></tr><tr><th>DriveLaW <sup><a href="#fn:26">26</a></sup></th><th>CVPR’26</th><th>C</th><td>99.0</td><td>97.1</td><td>96.7</td><td>100.0</td><td>81.3</td><td>89.1</td></tr><tr><th>ForeDrive (ours)</th><th>—</th><th>C</th><td>98.6</td><td>97.6</td><td>95.6</td><td>100.0</td><td>83.8</td><td>89.9</td></tr><tr><th>ForeDrive (ViT-L)</th><th>—</th><th>C</th><td>98.7</td><td>97.9</td><td>96.1</td><td>100.0</td><td>84.3</td><td>90.4</td></tr></tbody></table>

Table 1: Comparison on NAVSIM v1 navtest. C and C+L denote camera and camera+LiDAR. ForeDrive (ours) uses DINOv3 ViT-B/16; ForeDrive (ViT-L) is a larger-encoder upper bound.

### Planning-Relevant Latent World Model

Planning depends on future agent motion, ego-state evolution, and other scene changes that need not be represented at pixel level. This module therefore predicts future latent representations at multiple temporal scales for planning. We adopt a JEPA-style online/EMA architecture to predict latents in a DINOv3-initialized space [^18] [^19]. The online encoder is shared with the planner, while the predictor is optimized only by the forecasting losses defined below. This separation allows planning to shape the source representation without directly updating the predictor with planning gradients.

#### Online/EMA encoding.

Given a current front-camera image $I_{0}$ and ego status $s_{0}$, an online encoder $E_{\theta}$ maps $I_{0}$ to $N$ patch tokens $z_{0}=E_{\theta}(I_{0})\in\mathbb{R}^{N\times d}$. These tokens are later shared by foresight and planning. An EMA target encoder $E_{\bar{\theta}}$, which copies only the vision encoder, encodes future images $I_{t}$ into stop-gradient EMA targets. Future images are used only to construct training targets and are unavailable at inference.

#### Causal latent prediction.

The current observation is represented by an ego-status token and $N$ visual tokens, while each future horizon $t\in\mathcal{H}=\{1,2,3,4\}\,\mathrm{s}$ is assigned learned query tokens, collectively denoted by $Q_{\mathcal{H}}$ with $H=|\mathcal{H}|$. The core prediction is

$$
\bigl\{(\hat{z}_{t},\hat{s}_{t})\bigr\}_{t\in\mathcal{H}}=P_{\psi}(z_{0},s_{0},Q_{\mathcal{H}}).
$$

All tokens are projected to a common predictor width and augmented with token-type and horizon embeddings. The Transformer predictor $P_{\psi}$ processes the resulting sequence under a *frame-level* block-causal mask in a single forward pass rather than an autoregressive rollout. Earlier horizons may influence later ones, while later horizons remain invisible to earlier ones. Different horizons provide complementary multi-scale future context rather than sequential rollout states. Status and visual futures are read from their corresponding query tokens.

#### Foresight supervision.

We supervise horizon-weighted latent regression and status prediction

$$
\displaystyle\mathcal{L}_{\rm lat}
$$
 
$$
\displaystyle=\frac{1}{H}\sum_{t\in\mathcal{H}}w_{t}\,\ell_{t},
$$
$$
\displaystyle\mathcal{L}_{\rm status}
$$
 
$$
\displaystyle=\frac{1}{H}\sum_{t\in\mathcal{H}}\bigl[\operatorname{CE}(\hat{s}_{t}^{c},s_{t}^{c})+\operatorname{MSE}(\hat{s}_{t}^{m},s_{t}^{m})\bigr].
$$

where $\ell_{t}$ is a token-averaged $L_{1}$ between predicted and EMA visual latents, $w_{t}$ are fixed horizon loss weights (uniform if weighting is disabled), and $s_{t}^{c}$ / $s_{t}^{m}$ are the navigation-command and ego-motion (velocity/acceleration) status components.

### Foresight-Guided Generative Planning

Because current observations are more reliable than predicted futures, ForeDrive keeps current visual tokens on the residual path and admits future latents through gated visual fusion and future-status memory. A coarse-to-fine DiT decodes multimodal trajectories from these conditions. Trajectory-Adaptive Bias (TAB) connects trajectory anchors to image tokens by projecting evolving candidates into the front view and biasing cross-attention toward path-relevant visual tokens during denoising.

#### Current-primary Latent Fusion.

Projection necks map current tokens $z_{0}$ and predicted future latent representations $\hat{z}_{t}$ to planner-width tokens $\bar{c}$ and $\bar{f}_{t}$, respectively. We add shared spatial and per-horizon temporal embeddings, then apply a sample-shared gate $g_{t}^{\rm vis}=\sigma(a_{t}^{\rm vis})$ as $\tilde{f}_{t}=g_{t}^{\rm vis}\bar{f}_{t}$. The gate learns horizon-level contribution weights rather than sample-specific uncertainty estimates. Latent fusion $\mathcal{F}$ performs residual cross-attention with $\bar{c}$ as queries and concatenated gated futures as keys/values,

$$
K,V=\operatorname{Concat}_{t\in\mathcal{H}}(\tilde{f}_{t}),\qquad c^{\rm fuse}=\mathcal{F}(\bar{c},K,V),
$$

keeping current evidence as the residual backbone.

#### Future-status injection.

Predicted future ego motion provides complementary conditioning for trajectory generation. We embed $\operatorname{sg}(\hat{s}_{t}^{m})$ and modulate it with an independent sample-shared gate $g_{t}^{\rm st}=\sigma(a_{t}^{\rm st})$, yielding gated embeddings $\tilde{e}_{t}$. These embeddings are concatenated with the current ego embedding $e_{0}$ from $s_{0}$ to form the planner memory

$$
E_{\rm ego}=[e_{0};\tilde{e}_{1};\ldots;\tilde{e}_{H}],
$$

which conditions every decoder layer. Navigation commands $\hat{s}_{t}^{c}$ are not injected through this pathway.

<table><tbody><tr><th>Method</th><th>Venue</th><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DDC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TLC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>LK <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>HC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EPDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><th colspan="12">Traditional end-to-end methods</th></tr><tr><th>TransFuser <sup><a href="#fn:4">4</a></sup></th><th>TPAMI’23</th><td>96.9</td><td>89.9</td><td>97.8</td><td>99.7</td><td>87.1</td><td>95.4</td><td>92.7</td><td>98.3</td><td>87.2</td><td>76.7</td></tr><tr><th>ReCogDrive-IL <sup><a href="#fn:14">14</a></sup></th><th>ICLR’26</th><td>98.2</td><td>94.5</td><td>99.3</td><td>99.9</td><td>87.4</td><td>97.3</td><td>97.1</td><td>98.3</td><td>87.2</td><td>86.6</td></tr><tr><th>PRIX <sup><a href="#fn:25">25</a></sup></th><th>RA-L’26</th><td>98.0</td><td>95.6</td><td>99.5</td><td>99.8</td><td>87.4</td><td>97.2</td><td>97.1</td><td>98.3</td><td>87.6</td><td>84.2</td></tr><tr><th>DiffusionDrive <sup><a href="#fn:17">17</a></sup></th><th>CVPR’25</th><td>98.2</td><td>95.9</td><td>99.4</td><td>99.8</td><td>87.5</td><td>97.3</td><td>96.8</td><td>98.3</td><td>87.7</td><td>84.5</td></tr><tr><th>DiffRefiner-R34 <sup><a href="#fn:30">30</a></sup></th><th>AAAI’26</th><td>98.5</td><td>97.4</td><td>99.6</td><td>99.8</td><td>87.6</td><td>97.7</td><td>97.7</td><td>98.3</td><td>86.2</td><td>86.2</td></tr><tr><th>MeanFuser <sup><a href="#fn:20">20</a></sup></th><th>CVPR’26</th><td>98.3</td><td>97.2</td><td>99.6</td><td>99.8</td><td>87.6</td><td>97.4</td><td>97.3</td><td>98.3</td><td>88.2</td><td>89.5</td></tr><tr><th colspan="12">World-model and video–action methods</th></tr><tr><th>World4Drive <sup><a href="#fn:35">35</a></sup></th><th>ICCV’25</th><td>97.8</td><td>96.3</td><td>99.4</td><td>99.8</td><td>88.3</td><td>97.1</td><td>97.7</td><td>98.0</td><td>53.9</td><td>84.8</td></tr><tr><th>Epona <sup><a href="#fn:32">32</a></sup></th><th>ICCV’25</th><td>97.1</td><td>95.7</td><td>99.3</td><td>99.7</td><td>88.6</td><td>96.3</td><td>97.0</td><td>98.0</td><td>67.8</td><td>85.1</td></tr><tr><th>DriveVLA-W0 <sup><a href="#fn:12">12</a></sup></th><th>ICLR’26</th><td>98.5</td><td>99.1</td><td>98.0</td><td>99.7</td><td>86.4</td><td>98.1</td><td>93.2</td><td>97.9</td><td>58.9</td><td>86.1</td></tr><tr><th>WorldRFT <sup><a href="#fn:28">28</a></sup></th><th>AAAI’26</th><td>97.8</td><td>96.5</td><td>99.5</td><td>99.8</td><td>88.5</td><td>97.0</td><td>97.4</td><td>98.1</td><td>69.1</td><td>86.7</td></tr><tr><th>Drive-JEPA <sup><a href="#fn:22">22</a></sup></th><th>arXiv’26</th><td>98.4</td><td>98.6</td><td>99.1</td><td>99.8</td><td>88.4</td><td>97.8</td><td>97.6</td><td>97.9</td><td>84.8</td><td>87.8</td></tr><tr><th>Latent-WAM <sup><a href="#fn:21">21</a></sup></th><th>arXiv’26</th><td>98.1</td><td>97.3</td><td>99.6</td><td>99.8</td><td>87.7</td><td>97.3</td><td>97.6</td><td>98.1</td><td>87.3</td><td>89.3</td></tr><tr><th>ForeDrive (ours)</th><th>—</th><td>98.6</td><td>97.6</td><td>99.6</td><td>99.8</td><td>87.6</td><td>97.9</td><td>97.9</td><td>98.3</td><td>87.5</td><td>90.0</td></tr><tr><th>ForeDrive (ViT-L)</th><th>—</th><td>98.7</td><td>97.9</td><td>99.6</td><td>99.9</td><td>87.6</td><td>98.2</td><td>97.7</td><td>98.4</td><td>88.1</td><td>90.6</td></tr></tbody></table>

Table 2: Comparison on NAVSIM v2 navtest (one-stage non-reactive EPDMS). ForeDrive (ours) uses DINOv3 ViT-B/16; ForeDrive (ViT-L) is a larger-encoder upper bound.

<table><thead><tr><th>Method</th><th colspan="4">L2 (m) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th><th colspan="4">Collision (%) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th></tr><tr><th></th><th>1 s</th><th>2 s</th><th>3 s</th><th>Avg.</th><th>1 s</th><th>2 s</th><th>3 s</th><th>Avg.</th></tr></thead><tbody><tr><th>PWM (FT) <sup><a href="#fn:33">33</a></sup></th><td>0.41</td><td>0.75</td><td>1.17</td><td>0.78</td><td>0.02</td><td>0.10</td><td>0.35</td><td>0.16</td></tr><tr><th>Epona <sup><a href="#fn:32">32</a></sup></th><td>0.96</td><td>1.59</td><td>2.32</td><td>1.62</td><td>0.09</td><td>0.27</td><td>0.67</td><td>0.34</td></tr><tr><th>DriveLaW <sup><a href="#fn:26">26</a></sup></th><td>0.30</td><td>0.48</td><td>0.83</td><td>0.54</td><td>0.23</td><td>0.16</td><td>0.19</td><td>0.19</td></tr><tr><th>ForeDrive (ours)</th><td>0.22</td><td>0.44</td><td>0.79</td><td>0.48</td><td>0.01</td><td>0.09</td><td>0.23</td><td>0.11</td></tr></tbody></table>

Table 3: Zero-shot planning performance on the nuScenes validation set under the VAD /ST-P3 protocol. PWM (FT) uses nuScenes-trained checkpoints.

#### Coarse-to-fine diffusion planner.

Following DiffusionDrive [^17], we apply truncated diffusion with a cascaded DiT decoder to a fixed set of trajectory anchors $A$. Conditioned on fused visual tokens $c^{\rm fuse}$ and ego memory $E_{\rm ego}$, the planner predicts multimodal trajectories $\tau\in\mathbb{R}^{8\times 3}$ over $4\,\mathrm{s}$. Each decoder stage uses hard closest-anchor assignment, sigmoid focal loss for mode classification, and $L_{1}$ regression for the winning mode:

$$
\mathcal{L}_{\rm plan}=\sum_{k=1}^{2}\left(\lambda_{\rm cls}\mathcal{L}_{\rm focal}^{(k)}+\lambda_{\rm reg}\mathcal{L}_{1}^{(k)}\right).
$$

At inference, truncated DDIM runs for two steps and selects the mode by $\arg\max$ of the classification head.

#### Trajectory-Adaptive Bias for Planning.

Anchor-based DiT decoding represents trajectory candidates in ego/BEV coordinates, while the visual stream consists of front-view image tokens. Standard trajectory-to-visual cross-attention therefore lacks an explicit correspondence between BEV paths and image patches. We introduce Trajectory-Adaptive Bias (TAB) as a trajectory-aware visual attention interface to connect BEV trajectory candidates with front-view image tokens. For each mode, TAB projects the current ego/BEV trajectory candidate onto the front-camera image and builds a soft Gaussian proximity field over visual tokens; its log affinity is added to the cross-attention logits, so each mode attends more strongly to visual tokens near its projected path while retaining access to the full scene.

As denoising updates the trajectory candidates, TAB recomputes the bias at stage boundaries. The bias is mode-specific and differentiable; if a candidate has no valid projection, the added bias is constant and leaves the attention distribution unchanged. We apply TAB in every decoder layer of both stages, independently of future injection.

### Asymmetric Latent Optimization

The online representation $z_{0}$ is shared by forecasting and planning, so the two objectives may introduce optimization interference. Back-propagating the planning loss through the predictor can turn $P_{\psi}$ into a planning feature adapter and weaken forecast fidelity, whereas freezing the encoder for forecasting prevents planning from shaping actionable futures. ForeDrive mitigates this interference with asymmetric latent optimization via stop-gradient routing: both losses update the shared encoder, but planning gradients do not update the latent predictor.

Asymmetric gradient routing is given by

$$
\mathcal{L}_{\rm plan}\;\rightarrow\;E_{\theta},\;\mathcal{F},\;\text{planner},\qquad\mathcal{L}_{\rm plan}\;\not\rightarrow\;P_{\psi}.
$$

The latent and status objectives train $P_{\psi}$. Planning updates $E_{\theta}$, and the EMA target encoder tracks $E_{\theta}$ by exponential moving average, so planning affects EMA targets only through this encoder–EMA path. Injected future latent representations remain stop-gradient.

The full training objective is

$$
\mathcal{L}=\lambda_{\rm traj}\mathcal{L}_{\rm plan}+\lambda_{z}\mathcal{L}_{\rm lat}+\lambda_{s}\mathcal{L}_{\rm status}.
$$

All objectives are optimized in one stage, and the EMA target receives no gradient. The EMA target and auxiliary status objective help avoid representational collapse.

## 4 Experiments

### Setup

We train on the official NAVSIM navtrain split and report final metrics on navtest. The default backbone is DINOv3 ViT-B/16 with a $256\times 512$ front image; the WM predicts futures at $1/2/3/4\,\mathrm{s}$, and the planner uses 20 trajectory anchors over a $4\,\mathrm{s}$ horizon. Default ForeDrive uses a 4-layer predictor, a $2{\times}5$ DiT, and AdamW at $6{\times}10^{-4}$ for 100 epochs (batch 1024). Full configs are in the supplementary material.

### Benchmark

We evaluate on NAVSIM v1 and v2 [^5]. The primary metrics are PDMS on v1 and one-stage EPDMS on v2 navtest; we also report the associated safety and progress submetrics. NAVSIM uses *non-reactive simulation*: the ego vehicle commits to one planned trajectory over a fixed horizon while surrounding agents follow log replay. We use *one-stage EPDMS* for the Extended PDM Score computed from a single $4\,\mathrm{s}$ horizon of real observations under this protocol.

### Main Results

#### Comparison on NAVSIM v1.

Default ForeDrive (ViT-B/16) attains 89.9 PDMS on NAVSIM v1 (Table 1), improving over DiffRefiner-R34 and MeanFuser by 0.5 and 0.9 points, DiffusionDrive by 1.8, and DriveLaW and PWM by 0.8 and 1.8. Retraining the default recipe from three different random seeds yields PDMS in 89.7–89.9 with a standard deviation of 0.1; the table reports the primary run. Scaling only the vision encoder to ViT-L/16 further reaches 90.4 PDMS. Under the pure-IL protocol, these gains hold across seeds.

#### Comparison on NAVSIM v2.

On NAVSIM v2, default ForeDrive (ViT-B/16) reaches 90.0 one-stage EPDMS (Table 2), improving over MeanFuser by 0.5 points and over Latent-WAM and Drive-JEPA by 0.7 and 2.2 points, respectively. The ViT-L upper bound further attains 90.6. Together with the v1 results, the same pure-IL recipe yields consistent margins on both metric suites. Figure 2 compares PDMS/EPDMS across methods of different scales: at 118.4M parameters, ForeDrive matches the best listed PDMS and attains the highest EPDMS among the plotted methods, while remaining substantially smaller than Epona (2.5B) and DriveVLA-W0 (7.5B).

![[Fig4.png|Refer to caption]]

Figure 4: Trajectory comparison with DiffusionDrive and DriveLaW (green: expert; red: prediction).

#### Zero-shot performance on nuScenes.

Table 3 reports open-loop planning on the nuScenes validation set under the VAD /ST-P3 protocol [^9] [^7], with all methods re-evaluated under a shared metric implementation. ForeDrive is transferred zero-shot from NAVSIM without nuScenes fine-tuning. It attains the lowest L2 at every horizon and the lowest collision at 1 s, 2 s, and on average, reaching 0.48 m / 0.11% versus 0.54 m / 0.19% for DriveLaW and 0.78 m / 0.16% for nuScenes-trained PWM (FT); DriveLaW remains lower at collision@3 s (0.19% vs. 0.23%). These results indicate that foresight-guided planning learned on NAVSIM transfers across datasets in both displacement accuracy and safety.

#### Qualitative results.

Figure 4 compares ForeDrive with DiffusionDrive and DriveLaW on three interactive scenes (green: expert; red: prediction). ForeDrive stays closer to the expert corridor where the reactive DiT baselines leave the lane or incur safety/progress failures. These cases illustrate that foresight-guided planning helps on maneuvers that are difficult for current-only DiT planners.

### Ablation Study

#### Base configuration.

The planner retains DiffusionDrive’s anchor-based truncated-diffusion formulation but replaces its camera–LiDAR BEV interface with DINOv3 features from one current front-view image. We then add FiLM conditioning on the current ego state and deepen both DiT stages from one to five layers. The resulting current-only model is Base. Base and ForeDrive use the same DINOv3 ViT-B/16 encoder, $2{\times}5$ -layer DiT, navigation command, and positional and temporal embeddings. Base contains neither the world model (WM) nor TAB. The supplementary material reports the stepwise construction at ViT-S scale.

| Setting | WM | TAB | PDMS $\uparrow$ | $\Delta$ vs. Base |
| --- | --- | --- | --- | --- |
| DiffusionDrive |  |  | 88.1 | – |
| Base |  |  | 88.9 | – |
| Base + TAB |  | $\checkmark$ | 89.5 | $+0.6$ |
| Base + WM | $\checkmark$ |  | 89.6 | $+0.7$ |
| Base + WM + TAB | $\checkmark$ | $\checkmark$ | 89.9 | $\mathbf{+1.0}$ |

Table 4: Ablation study of WM and TAB relative to Base. DiffusionDrive is an external reference. Here WM denotes the complete future representation pipeline, including latent prediction supervision and future conditioning interfaces.

| Setting | Aux. | $z$ Inject. | $s$ Inject. | PDMS $\uparrow$ | $\Delta$ vs. Base |
| --- | --- | --- | --- | --- | --- |
| Base |  |  |  | 88.9 | – |
| Auxiliary | $\checkmark$ |  |  | 89.0 | $+0.1$ |
| Visual only | $\checkmark$ | $\checkmark$ |  | 89.4 | $+0.5$ |
| Status only | $\checkmark$ |  | $\checkmark$ | 89.3 | $+0.4$ |
| Full WM | $\checkmark$ | $\checkmark$ | $\checkmark$ | 89.6 | $\mathbf{+0.7}$ |

Table 5: Ablation study of future representation components without TAB on NAVSIM v1 (auxiliary loss, visual injection, and status injection).

| Training paradigm | $\mathcal{L}_{\rm lat}$ $\downarrow$ | Lat. Cos.$\uparrow$ | $\mathcal{L}_{\rm status}$ $\downarrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- |
| Two-stage / freeze WM | 5.93 | 0.843 | 0.604 | 87.9 |
| Joint + detach encoder | 6.07 | 0.838 | 0.599 | 88.6 |
| Joint + aux-only | 7.44 | 0.769 | 0.536 | 89.6 |
| Full joint (ours) | 7.64 | 0.759 | 0.531 | 89.9 |

Table 6: Comparison of training paradigms on NAVSIM. We report latent $L_{1}$, latent cosine similarity (Lat. Cos.), composite future-status loss, and navtest PDMS.

| Interface | PDMS $\uparrow$ | $\Delta$ vs. ours |
| --- | --- | --- |
| Gated fusion (ours) | 89.4 | 0.0 |
| Current only | 88.9 | $-$ 0.5 |
| Future only | 84.2 | $-$ 5.2 |
| Concatenation | 89.2 | $-$ 0.2 |
| Dual-memory | 89.0 | $-$ 0.4 |
| Ungated fusion | 89.0 | $-$ 0.4 |

Table 7: Future-injection interfaces on NAVSIM (TAB and future-status injection off).

| Encoder | Params | Base $\uparrow$ | Full $\uparrow$ | $\Delta$ |
| --- | --- | --- | --- | --- |
| DINOv3 ViT-S/16 | 21M | 87.4 | 89.0 | $+1.6$ |
| DINOv3 ViT-B/16 | 86M | 88.9 | 89.9 | $+1.0$ |
| DINOv3 ViT-L/16 | 300M | 89.3 | 90.4 | $+1.1$ |

Table 8: DINOv3 backbone capacity on NAVSIM v1 under a fixed ForeDrive pipeline. Params counts the vision encoder only. Base is the matched current-only planner; Full is ForeDrive (WM+TAB); $\Delta$ is Full $-$ Base. ViT-L is a capacity upper bound; main results and ablations use ViT-B.

#### Component ablation.

Table 4 ablates WM and TAB on the matched Base (88.9 PDMS), with DiffusionDrive listed only as an external reference (88.1). Adding TAB or WM alone raises PDMS to 89.5 ($+0.6$) and 89.6 ($+0.7$), respectively, while combining both reaches 89.9 ($+1.0$). The joint gain exceeds either factor alone, indicating that WM and TAB contribute distinct effects.

Table 5 further decomposes the WM under TAB off. Auxiliary prediction alone yields only $+0.1$ PDMS, indicating that a forecasting side objective is insufficient. Exposing predicted futures to the planner accounts for most of the gain: visual injection reaches $+0.5$ and future-status injection $+0.4$; using both pathways attains $+0.7$ and recovers the Base ${+}$ WM result in Table 4. The main improvement thus comes from consuming predicted futures, not from adding a forecasting loss alone.

#### Training paradigm.

Table 6 compares four training paradigms along two axes: whether planning gradients update the shared online encoder, and whether predicted futures are injected. *Two-stage / freeze WM* pretrains then freezes the world model; *Joint + detach encoder* trains jointly but stops planner gradients before the shared encoder; *Joint + aux-only* updates the encoder with both losses yet does not inject futures; *Full joint* is our setting, with planning-driven encoder updates, future injection, and forecasting-only supervision of the predictor.

The first two settings best match EMA visual targets (lowest $\mathcal{L}_{\rm lat}$ 5.93 / 6.07; highest latent cosine 0.843 / 0.838) yet obtain the weakest PDMS (87.9 / 88.6). Encoder-updating joint training raises PDMS to 89.6–89.9 despite weaker visual alignment. Under matched future injection, full joint improves over encoder detachment by 1.3 PDMS; with encoder updates retained, enabling injection adds 0.3 over the auxiliary-only joint baseline. Across variants, better generic latent forecast alignment does not necessarily correspond to higher planning scores, indicating that planning-oriented representations may deviate from prediction-optimal targets to preserve decision-relevant information.

#### Latent fusion strategies.

Under a matched protocol with TAB and future-status ego-KV disabled, we compare future-injection interfaces in Table 7. Gated current-primary fusion attains 89.4 PDMS; current-only is lower by 0.5, while future-only drops by 5.2. Predicted futures therefore improve planning only when fused with the present observation, and current evidence should remain primary. Concatenation, dual-memory, and ungated fusion trail gated fusion by 0.2–0.4, indicating that a gated residual interface is preferable to exposing additional future tokens alone. Under the same TAB-/status-off protocol, permuting predicted horizons at inference leaves PDMS unchanged at $89.4$, whereas zeroing futures at test time reduces it to $88.0$. The permutation result suggests that the planner mainly exploits aggregated multi-scale future context rather than strict horizon ordering, which is consistent with our parallel latent prediction design. Full breakdowns are reported in the supplementary material.

#### Encoder capacity.

Table 8 varies only the DINOv3 backbone under a fixed ForeDrive pipeline. Stronger encoders raise both Base and Full: a better current representation already improves the current-only planner, and Full improves as well (89.0 / 89.9 / 90.4 at ViT-S/B/L). Full still outperforms its matched Base at every scale. The Base ${\to}$ Full margin is largest on ViT-S ($+1.6$), where perception is weakest, and remains positive on ViT-B and ViT-L ($+1.0$ / $+1.1$). Foresight and TAB still help at every encoder scale; they do not replace a stronger backbone. Single-factor and WM-component breakdowns at ViT-S/L appear in the supplementary material. We keep ViT-B/16 for the main results and ablations for its accuracy–cost trade-off, and report ViT-L only as a capacity upper bound.

#### Inference efficiency.

On a single H20 GPU, the default ForeDrive runs at $58.2\,\mathrm{ms}$ per frame (17.2 FPS) with $1.03\,\mathrm{GB}$ peak memory, while latent foresight adds approximately $26\,\mathrm{ms}$ over the current-only Base. The full model has 118.4M parameters at evaluation; the training-only EMA encoder is excluded from this total.

## 5 Conclusion

ForeDrive learns planning-relevant future latent representations and couples them asymmetrically to a DiT planner. Asymmetric latent optimization via stop-gradient routing mitigates direct prediction–planning optimization interference, while planning-oriented interfaces that include gated fusion, future-status injection, and TAB connect predicted futures to diffusion planning. With a single front-view image at inference and pure imitation learning, ForeDrive attains 89.9 PDMS on NAVSIM v1 and 90.0 EPDMS on v2. Matched ablations show that auxiliary forecasting alone is insufficient, that incorporating future latents is necessary, and that encoder updates yield planning-relevant futures despite weaker EMA alignment.

The evaluation is limited to camera-only, non-reactive simulation, where inaccurate futures can still mislead the planner. Testing under interactive closed-loop settings is an important next step.

## References

## Supplementary Material

Supplementary Material

Organization. We organize the appendix by question: Sec. A gives implementation details; Sec. B builds the ViT-S Base for small-encoder ablations; Sec. C visualizes TAB on the ViT-B Base; Sec. D tests causal use of predicted futures (TAB/status off); Sec. E contrasts Full (WM+TAB+status) with Base+TAB; Sec. F varies multi-scale future-query sets; Sec. G repeats WM/TAB factorials at ViT-S and ViT-L; Sec. H isolates stop-gradient routing for future latent injection; Sec. I probes trajectory readability under a frozen backbone; Sec. J checks scene grounding via an RGB readout; Sec. K lists parameter counts for the main accuracy–size plot.

## Appendix A More Implementation Details

We detail the data pipeline, latent world model, current-primary foresight interface, Trajectory-Adaptive Bias (TAB), truncated-diffusion planner, and joint training recipe.

#### Data pipeline and evaluation protocol.

We train on official NAVSIM navtrain, with 18,179 held-out validation samples and 85,109 training samples. Checkpoints are selected on validation; the 12,147-scenario navtest set is used only for final reporting. Unless noted otherwise, we report NAVSIM v1 PDMS and NAVSIM v2 one-stage EPDMS under official non-reactive simulation. The ego commits to one $4\,\mathrm{s}$ plan while surrounding agents follow log replay. We do not use two-stage pseudo-simulation.

The sensor input is one front-camera frame. We crop 28 pixels from the top and bottom, resize to $256\times 512$, and apply no stochastic augmentation. Future frames are used only to build EMA latent targets and are unavailable at inference. By default, the world model predicts four horizons at $1/2/3/4\,\mathrm{s}$ on the 2 Hz NAVSIM grid. The planning target is an eight-pose trajectory $(x,y,\theta)$ over $4\,\mathrm{s}$ at $0.5\,\mathrm{s}$ steps, decoded from 20 anchors.

#### Latent world model.

The default backbone is DINOv3 ViT-B/16. It encodes the current front image into 512 patch tokens of width 768. An EMA copy of the vision encoder provides stop-gradient targets for future frames; only this encoder is mirrored. A lightweight Transformer predictor forecasts multi-horizon (multi-scale) visual latents and ego-status quantities from current tokens and learnable future queries.

The predictor is a 4-layer pre-LN Transformer of width 512 (8 heads, FFN 2048, GELU, dropout 0.1). Linear projections bridge the 768-D encoder space and the predictor width; time/token embeddings and future queries share that width. Attention uses a frame-level block-causal mask: tokens within a horizon attend freely, while later horizons stay invisible. All horizons are predicted in one forward pass, not by autoregressive rollout. Each horizon emits a status token for a 4-way navigation-command classifier (CE) and raw velocity/acceleration regressions (MSE). Predicted visual futures are projected back to 768-D before the latent loss.

Foresight supervision covers the four default horizons. Visual terms use token-averaged $L_{1}$ against EMA targets, without a patch mask. Horizon weights are proportional to $1/(i{+}1)$ on the 2 Hz index $i$, then rescaled to unit mean. The status objective equally weights command CE and motion MSE, and is scaled by $0.1$ in the joint loss.

<table><tbody><tr><td>Step</td><td>Configuration</td><td>PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>Ref.</td><td>DiffusionDrive (camera+LiDAR BEV)</td><td>88.1</td></tr><tr><td colspan="3">Input/representation adaptation</td></tr><tr><td>(a)</td><td>DINOv3 ViT-S camera only</td><td>86.4</td></tr><tr><td>(b)</td><td>a + current-status FiLM</td><td>86.6</td></tr><tr><td>(c)</td><td>b + deepen DiT (<math><semantics><mrow><mn>1</mn> <mo>→</mo> <mn>5</mn></mrow> <annotation>1{\to}5</annotation></semantics></math> /stage) (= ViT-S Base)</td><td>87.4</td></tr></tbody></table>

Table 1: From DiffusionDrive to ViT-S Base on NAVSIM v1 (DINOv3 ViT-S/16, camera-only). Steps (a)–(c) are uncontrolled input/representation adaptations. DiffusionDrive is not a controlled baseline due to LiDAR input; it is listed as an external reference only.

![[SupFig3.png|Refer to caption]]

Figure 1: Trajectory-to-image attention with vs. without TAB (ViT-B Base; no WM). Left: Base ( σ = 0 \\sigma{=}0 ); right: Base+TAB ( 0.25 \\sigma{=}0.25 ). Heatmaps average decoder attention over layers and trajectory-mode queries; yellow/cyan: selected plan / expert. TAB concentrates mass on the near-road corridor.

| Mode | NC | DAC | EP | TTC | C | DDC | PDMS | $\Delta$ vs. normal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle | $98.9$ | $97.0$ | $83.5$ | $96.0$ | $100.0$ | $98.5$ | $89.6$ | $+0.2$ |
| horizon\_shuffle | $98.7$ | $97.1$ | $83.6$ | $95.4$ | $100.0$ | $98.3$ | $89.4$ | $+0.0$ |
| normal | $98.7$ | $97.1$ | $83.6$ | $95.4$ | $100.0$ | $98.3$ | $\mathbf{89.4}$ | $0$ |
| token\_mask ($r{=}0.75$) | $98.9$ | $96.7$ | $82.7$ | $95.7$ | $100.0$ | $98.4$ | $89.0$ | $-0.4$ |
| zero | $98.9$ | $95.8$ | $81.2$ | $95.9$ | $99.9$ | $98.4$ | $88.0$ | $-1.4$ |
| persistence | $98.3$ | $95.6$ | $81.7$ | $93.7$ | $100.0$ | $98.1$ | $87.0$ | $-2.4$ |
| cross\_sample | $98.0$ | $95.9$ | $81.7$ | $93.2$ | $99.9$ | $98.3$ | $87.0$ | $-2.4$ |

Table 2: Causal interventions on future latents without TAB/status (gated fusion only). $\Delta$: PDMS vs. normal. Subscore abbreviations follow NAVSIM (NC/DAC/EP/TTC/C/DDC).

#### Current-primary foresight interface.

Predicted futures condition the planner without replacing current evidence. Current and future visual tokens are projected to width 256 (Linear+LayerNorm), then given shared spatial embeddings and learnable per-horizon time embeddings. A sample-shared sigmoid gate scales each future stream before fusion; it encodes dataset-level horizon preference, not per-sample reliability. Fusion is one Post-LN cross-attention layer (8 heads, no FFN): current tokens are residual queries, and gated futures are keys/values.

Complementary *predicted* future ego motion is injected as planner memory (not ground-truth future status). Stop-gradient predicted velocity and acceleration are embedded to width 256, scaled by a second sample-shared sigmoid gate, and concatenated with the measured current ego embedding. Navigation commands are not routed through this path. The resulting ego memory conditions every decoder layer with the fused visual tokens.

#### Trajectory-Adaptive Bias.

TAB is a soft geometry prior that links BEV trajectory candidates to front-view image tokens. It provides a trajectory-conditioned spatial bias rather than learned sample-specific attention weights. Each candidate pose is projected into the camera. A pose is valid only if camera depth exceeds $10^{-3}$ and the pixel lies inside the image (no clamping); invalid poses are excluded. For each mode–token pair, we take the maximum Gaussian affinity (bandwidth $0.25$) between the token center and the mode’s valid projected waypoints, then convert it to a log-bias with floor $10^{-6}$. If a mode has no valid projection, the bias is a uniform shift and leaves the softmax unchanged.

We add this bias to trajectory-to-visual attention logits before softmax in every layer of both DiT stages, broadcast across heads. Ego-memory cross-attention is unchanged. Within one DDIM step, the first stage shares the initial noisy-anchor plan for projection; later layers then use that stage’s supervised plan.

#### Truncated diffusion planner.

Following DiffusionDrive, we use truncated diffusion over 20 fixed $k$ -means $(x,y)$ anchors; heading is $\tanh(\cdot)\cdot\pi$. Training uses a 1,000-step DDIM schedule with sample prediction and truncated noise levels from $\{0,\ldots,49\}$. Noise is added to normalized anchors (not ground-truth trajectories) and denormalized before decoding. Each of two cascaded stages is a 5-layer DiT with independently cloned weights. A stage predicts a clean plan by residual xy update plus a heading head. Training draws one truncated noise sample and runs both stages in one forward. At the stage boundary, first-stage xy is stop-gradient-copied at the same noise level without re-noising. Sinusoidal timestep embeddings pass through a small MLP for AdaLN-style modulation.

Mode assignment uses hard closest-anchor matching on mean-horizon xy $L_{2}$. Classification uses sigmoid focal loss ($\gamma{=}2$, $\alpha{=}0.25$); $L_{1}$ regression applies only to the winning mode, including heading. At inference we run two truncated DDIM steps starting near $t{\approx}8$, execute the full multi-stage decoder each step, and select the mode by classification $\arg\max$, without an external scorer.

| Horizons (s) | Type | w/o TAB $\uparrow$ | w/ TAB $\uparrow$ |
| --- | --- | --- | --- |
| $\{1\}$ | near-term | 89.1 | 89.6 |
| $\{2\}$ | mid-term | 89.3 | 89.8 |
| $\{1,2\}$ | two-step | 89.3 | 89.7 |
| $\{1,2,3,4\}$ | four-step | 89.6 | 89.9 |

Table 3: Future-query sets vs. PDMS (NAVSIM). Default Full: $1/2/3/4\,\mathrm{s}$ ($\{1,3,5,7\}$ at 2 Hz).

![[SupFig1.png|Refer to caption]]

Figure 2: Qualitative comparison only: Full (WM+TAB+status) vs. Base+TAB (no WM; matched DiT). This isolates the practical effect of future conditioning rather than a controlled ablation. Left: front view; middle/right: BEV plans (green: expert; red: prediction; blue: agents). Base+TAB fails via collisions (a,b) or DAC (c) with per-scene PDMS 0; Full remains feasible.

#### Joint optimization.

We train for 100 epochs with AdamW (base lr $6\times 10^{-4}$, weight decay $10^{-4}$, cosine decay) on 32 GPUs, global batch 1,024, and gradient clipping $1.0$. Norm and bias parameters receive no weight decay. EMA decay for the target encoder is $0.999$. Joint loss weights for classification / regression / trajectory / latent / status are $10/8/12/5/0.1$. Planning gradients update the shared online encoder but stop before the latent predictor, so injected futures are consumed as a stop-gradient signal (main-paper asymmetric recipe; matched inject ablation in Sec. H). The stop-gradient is applied only on the injected future representations before planner consumption; gradients from planning still update the online encoder through the current representation path.

Backbone fine-tuning depends on encoder capacity. For DINOv3 ViT-S/16 (ViT-S Base decomposition and small-encoder ablations), we use *full fine-tuning*: layer-wise decay $1.0$, no stochastic depth, three-epoch warmup, and the same AdamW groups as the heads. The small encoder can adapt directly to driving with little extra regularization.

For default DINOv3 ViT-B/16, we use a conservative *layered* schedule. Layer-wise decay $0.7$ shrinks early-block learning rates; the predictor, fusion modules, and DiT keep the full base rate. We also use stochastic depth $0.15$, five-epoch warmup, and BF16 in the backbone. ViT-L capacity runs use the same schedule except layer-wise decay $0.8$. Early pretrained features are largely preserved; adaptation concentrates in upper encoder layers and the WM/planning heads. DropPath widens the online–EMA gap in training and is disabled at evaluation, so it is a training regularizer rather than a deployment mismatch.

#### Compute and software.

We train with PyTorch 2.4 and PyTorch Lightning 2.2 on a standard NAVSIM stack (Python 3.9, CUDA 12.1). Training uses 32 $\times$ NVIDIA H20 GPUs (96 GB) as 4 nodes $\times$ 8 GPUs with DDP and BF16; per-GPU batch size is 32 (global batch 1,024). Randomness is controlled by Lightning seed\_everything. The reported ViT-L capacity checkpoint uses seed 0; the default ViT-B main-table score reports a primary run plus three-seed stability, as in the main paper. Inference latency is measured on one idle H20 with batch size 1 and $256\times 512$ inputs as pure FP32 forward time (excluding crop/resize and data loading). Training uses the official NAVSIM navtrain latent cache and OpenScene v1.1; evaluation follows official PDMS / one-stage EPDMS.

## Appendix B Baseline Decomposition

Table 1 constructs the current-only ViT-S Base used in small-encoder ablations (NAVSIM v1, DINOv3 ViT-S/16). DiffusionDrive (88.1 PDMS) is an external reference only: it uses camera+LiDAR BEV and is not a controlled drop-in for our camera-only setting. Starting from the same truncated-diffusion planner, we replace the BEV interface with a single front-view DINOv3 stream and drop LiDAR (a), lowering PDMS to 86.4. Present-status FiLM (b) adds $+0.2$; deepening each DiT stage from 1 to 5 layers (c) yields ViT-S Base at 87.4.

ViT-S Base is thus a current-only camera planner: present-status FiLM and a deeper DiT, but no world model and no TAB. Present-status FiLM is not the full model’s predicted future-status injection. Steps (a)–(c) jointly change sensors and representation, so they are uncontrolled adaptations; controlled WM/TAB gains appear in Table 4. The main-paper Base uses ViT-B.

| Setting | WM | TAB | PDMS $\uparrow$ | $\Delta$ vs. Base |
| --- | --- | --- | --- | --- |
| Base |  |  | 87.4 | \- |
| Base + TAB |  | $\checkmark$ | 88.1 | +0.7 |
| Base + WM | $\checkmark$ |  | 88.0 | +0.6 |
| Base + WM + TAB | $\checkmark$ | $\checkmark$ | 89.0 | +1.6 |

Table 4: WM $\times$ TAB factorial on DINOv3 ViT-S/16 (mirrors the main paper). $\Delta$: PDMS vs. Base. WM includes aux. prediction and visual/status injection.

| Setting | Aux. | $z$ Inject. | $s$ Inject. | PDMS $\uparrow$ | $\Delta$ vs. Base |
| --- | --- | --- | --- | --- | --- |
| Base |  |  |  | 87.4 | \- |
| Auxiliary | $\checkmark$ |  |  | 87.7 | +0.3 |
| Visual only | $\checkmark$ | $\checkmark$ |  | 87.8 | +0.4 |
| Status only | $\checkmark$ |  | $\checkmark$ | 87.9 | +0.5 |
| Full WM | $\checkmark$ | $\checkmark$ | $\checkmark$ | 88.0 | +0.6 |

Table 5: WM components without TAB on DINOv3 ViT-S/16 (mirrors the main paper). $z$ / $s$: visual/status injection; stop-gradient by default.

| Setting | WM | TAB | PDMS $\uparrow$ | $\Delta$ vs. Base |
| --- | --- | --- | --- | --- |
| Base |  |  | 89.3 | \- |
| Base + TAB |  | $\checkmark$ | 90.0 | +0.7 |
| Base + WM | $\checkmark$ |  | 89.8 | +0.5 |
| Base + WM + TAB | $\checkmark$ | $\checkmark$ | 90.4 | +1.1 |

Table 6: WM $\times$ TAB factorial on DINOv3 ViT-L/16 (mirrors the main paper). Full matches the main encoder-capacity result.

| Setting | Aux. | $z$ Inject. | $s$ Inject. | PDMS $\uparrow$ | $\Delta$ vs. Base |
| --- | --- | --- | --- | --- | --- |
| Base |  |  |  | 89.3 | \- |
| Auxiliary | $\checkmark$ |  |  | 89.4 | +0.1 |
| Visual only | $\checkmark$ | $\checkmark$ |  | 89.5 | +0.2 |
| Status only | $\checkmark$ |  | $\checkmark$ | 89.6 | +0.3 |
| Full WM | $\checkmark$ | $\checkmark$ | $\checkmark$ | 89.8 | +0.5 |

Table 7: WM components without TAB on DINOv3 ViT-L/16 (mirrors the main paper). $z$ / $s$: visual/status injection; stop-gradient by default.

## Appendix C Base vs. Base+TAB Attention

We next ask what TAB changes in planner spatial attention on the main-paper ViT-B Base (current-only; no WM). TAB should pull trajectory-to-image attention toward the projected plan—typically the near road—rather than a near-uniform front-view scan. We isolate this effect without a world model: two ViT-B Base runs share data and decoder and differ only in TAB bandwidth ($\sigma{=}0$ vs. $\sigma{=}0.25$). This is not the ViT-S Base of Sec. B; the goal is mechanism visualization, not cross-scale score comparison.

In Fig. 1, Base attention is diffuse over the lower image, while Base+TAB collapses onto a near-road band along the plan. TAB therefore reshapes spatial evidence gathering, not only the trajectory head prior. The direction matches the matched Base $\to$ Base+TAB PDMS gain; causal attribution still rests on the main-paper TAB ablations.

## Appendix D Future-Latent Causal Interventions

Does a WM-conditioned planner use predicted future latents, or only a generic “non-zero” signal? We fix a checkpoint and change only the future tokens at inference. We report PDMS on NAVSIM navtest ($N{=}12{,}147$). Modes: normal (WM predictions); oracle (EMA ground-truth futures); horizon\_shuffle (permute horizons, keep slot-wise time embeddings); token\_mask (drop $75\%$ of future tokens); cross\_sample (swap another sample’s predictions); persistence (tile the current latent); zero (remove futures).

#### Isolating the visual future channel.

We intervene on a checkpoint trained without TAB and without predicted future-status injection (gated fusion only; Table 2). Degradations must then come from the visual future channel. Normal leads among predicted-future settings: fusion sees in-distribution WM tokens. Oracle gains only $+0.2$: GT futures are cleaner but still mismatched to the predictor-trained fusion path. Horizon shuffle matches normal on every subscore ($89.4$). With the main-paper permutation result, this suggests the DiT uses multi-horizon futures as aggregated multi-scale temporal context rather than a strict ordered rollout; over-smoothed predictions further weaken the order probe. Token mask ($-0.4$) is mild: remaining in-distribution tokens still help. The key contrast is zero versus identity corruptions. Zeroing drops PDMS by $1.4$, mainly via EP/DAC (EP $83.6{\to}81.2$, DAC $97.1{\to}95.8$), while TTC/NC slightly improve. Without futures the policy is more conservative but less progressive—bad for PDMS, not mainly via collisions. Persistence and cross-sample fall below zero ($87.0$ vs. $88.0$): gated fusion can down-weight near-empty inputs, yet still trusts structured but semantically wrong futures; TTC collapses ($93.7$ / $93.2$) despite higher EP than zero. Summary: matched predictions help; GT helps little under mismatch; missing futures hurt progress; *wrong* futures hurt collision metrics more than no futures. A remaining limitation is that predicted latents are over-smoothed relative to GT.

## Appendix E Qualitative Future Conditioning

Figure 3: Early WM probes under matched inject (ViT-B; TAB off). (a) Val. WM $L_{1}$. (b) Gain over copy, $L_{1}^{\mathrm{copy}}-L_{1}^{\mathrm{WM}}$ ($>0$: better than pasting the current latent). Joint shows an early rebound; detach is monotonic. Primary planning claim: Table 8.

![[SupFig2.png|Refer to caption]]

Figure 4: Latent RGB probe. Left: current view and BEV plan. Right: GT frames vs. reconstructions from predicted futures ( t = 1 t{=}1 – 4; probe trained on EMA targets only). Coarse but scene-grounded; not used for planning.

Figure 2 qualitatively compares Full (WM+TAB+status) with Base+TAB (TAB on; no WM / no future injection) under a matched DiT backbone. Unlike Sec. D (TAB/status off), this isolates the practical effect of future conditioning rather than a controlled ablation, and is not the same protocol as Table 2.

In the multi-agent intersection (a) and turning conflict (b), Base+TAB cuts across an interacting vehicle (per-scene PDMS $0$; collisions), while Full stays clear (1.00 / 0.58). In the left-turn corridor (c), Base+TAB leaves the drivable area (per-scene PDMS $0$); Full tracks the expert (1.00). The cases illustrate two failure modes that foresight can mitigate: dynamic collisions and static DAC errors when a TAB-equipped, future-free decoder misreads the corridor. They are illustrative; aggregate evidence is in the quantitative tables.

## Appendix F Predicted-Horizon Sets

Causal interventions leave open which future-query sets to train with. Because horizon shuffle leaves PDMS unchanged (Sec. D), we treat horizons as complementary multi-scale context, not a strict ordered rollout. Table 3 compares query sets under matched training, with and without TAB. Default Full uses $1/2/3/4\,\mathrm{s}$.

Without TAB, four horizons reach 89.6 PDMS vs. 89.1 for $\{1\}$ and 89.3 for $\{2\}$ and $\{1,2\}$. With TAB, all settings improve and the range shrinks from 0.5 to 0.3; $\{1,2,3,4\}$ remains best at 89.9. Aggregating scales helps, but single-run gaps are small: we do not claim any single horizon is necessary, nor that inference requires strict order. Learned gates likewise do not attribute the gaps to specific horizons.

## Appendix G Encoder-Scale Ablations (ViT-S / ViT-L)

We repeat the main-paper WM $\times$ TAB factorial and the WM-component ablation (TAB off) at DINOv3 ViT-S/16 and ViT-L/16. Protocol matches the main tables (NAVSIM v1 PDMS; single-run). Full ForeDrive at ViT-S/L matches the main encoder-capacity results (89.0 / 90.4); ViT-S Base matches Table 1 (87.4). With the main-paper ViT-B results (Base 88.9 $\rightarrow$ Full 89.9), these tables test whether the mechanisms are capacity artifacts.

#### WM×\\timesTAB factorial.

On ViT-S (Table 4), Base is 87.4. TAB / WM alone reach 88.1 / 88.0 ($+0.7$ / $+0.6$); combining both reaches 89.0 ($+1.6$). The joint gain exceeds either factor and the ViT-B combined gain ($+1.0$); this single run is suggestive, not conclusive, of a larger weak-encoder benefit. On ViT-L (Table 6), Base is 89.3. TAB / WM alone add $+0.7$ / $+0.5$; Full reaches 90.4 ($+1.1$). TAB adds $0.7$ at both scales; WM-only gains are $0.6$ and $0.5$. Gains are positive at both scales; attributing the WM-gap to capacity needs more runs. WM and TAB remain complementary: Full beats the better single factor by $+0.9$ (ViT-S) and $+0.4$ (ViT-L).

#### WM-component ablation (TAB off).

Tables 5 and 7 isolate auxiliary prediction vs. visual/status injection. On ViT-S, aux / visual / status / Full WM add $+0.3$ / $+0.4$ / $+0.5$ / $+0.6$. On ViT-L, the corresponding gains are $+0.1$ / $+0.2$ / $+0.3$ / $+0.5$. Across encoders, direct injection beats auxiliary prediction alone.

#### Takeaway across scales.

Main-paper encoder-capacity already summarizes Base $\to$ Full ($+1.6$ / $+1.0$ / $+1.1$ at ViT-S/B/L). The tables above give the single-factor and WM-component breakdowns; each tested scale improves in the single-run setting.

## Appendix H Stop-Gradient Routing for Future Latent Injection

The main paper shows that joint encoder updates help planning, but leaves a finer inject-path choice: should $\mathcal{L}_{\rm plan}$ also flow into the world-model predictor? We isolate that flag under matched future injection. We change only whether injected visual latents and predicted future status are stop-gradient before the planner. All else is matched: ViT-B/16, predictor width 512, $16{\times}32$ tokens, gated fusion, predicted status injection on, TAB off ($\sigma_{\rm TAB}{=}0$). The stop-gradient is applied only on the injected future representations before planner consumption; gradients from planning still update the online encoder through the current representation path.

| Setting | $L_{1}$ $\downarrow$ | Cos.$\uparrow$ | $\mathcal{L}_{s}$ $\downarrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- |
| Inject + detach (ours) | 8.06 | 0.743 | 0.532 | 89.6 |
| Inject + joint | 8.08 | 0.747 | 0.514 | 89.2 |

Table 8: Detach vs. joint through the inject path (ViT-B; TAB off; matched inject). $L_{1}$ /Cos./ $\mathcal{L}_{s}$: WM probes on 18,179 val samples from last.ckpt.

| Source | ADE $\downarrow$ | FDE $\downarrow$ | Shuffle ADE $\downarrow$ |
| --- | --- | --- | --- |
| ForeDrive (pred.) | 1.55 | 2.77 | 2.52 |
| Pred.-opt. (pred.) | 1.83 | 3.24 | 3.38 |
| EMA-GT | 2.10 | 3.61 | 2.93 |
| Current | 2.31 | 4.08 | – |

Table 9: Frozen-backbone trajectory probe: two-layer MLP regresses expert futures from one latent source. Pred.: WM predictions; EMA-GT: EMA targets; Shuffle ADE: horizon permutation at eval.

#### Why detach.

Under matched inject, stop-gradient improves PDMS by $0.4$ ($89.6$ vs. $89.2$; Table 8), while final WM probes stay tied ($L_{1}$ differs by $0.02$; joint is slightly better on cosine/ $\mathcal{L}_{s}$). Detach is not justified by a higher terminal forecast score. We use it for asymmetric role separation: the WM is trained only by prediction losses, and the planner consumes stop-gradient futures so trajectory supervision does not rewrite the WM objective. The PDMS gain aligns with early-training stability (Fig. 3) and clearer objective separation, not with lower absolute prediction error. We do not claim a theoretical resolution of gradient interference.

#### Early-training motivation.

Figure 3 shows training dynamics. Panel (a): validation WM $L_{1}$. Panel (b): gain over copy,

$$
\mathrm{gain}\;=\;L_{1}^{\mathrm{copy}}-L_{1}^{\mathrm{WM}},
$$

where $L_{1}^{\mathrm{copy}}$ pastes the current latent to every future horizon. Positive gain means the WM beats this baseline. In epochs 3–8, joint shows a val. $L_{1}$ rebound ($10.69{\to}11.80$); detach decreases monotonically. Detach turns stably positive earlier; joint oscillates near zero longer. After this phase both improve, and final WM probes nearly align (Table 8). We treat the early curves as an engineering motivation for stop-gradient—joint can be unstable while the predictor adapts—not as evidence of a better terminal forecast. The planning conclusion remains: under matched inject and matched final WM quality, detach improves PDMS.

## Appendix I Trajectory Probe of Future Latents

Detach improves PDMS while final WM reconstruction stays nearly unchanged. Do predicted latents preserve more trajectory-relevant information, or are they only equally reconstructible? Closed-loop scores and WM $L_{1}$ /cosine cannot separate those cases. We freeze the vision encoder and world model and train a weak readout to regress expert trajectories from one latent source at a time. Unlike reconstruction probes, this tests whether latents retain information useful for trajectory regression; it does not prove that the planner must rely on these latents, nor that the representation is planning-optimal. It complements Sec. D: swapping EMA-GT into a predictor-trained planner yields little PDMS gain (mismatch); a freshly trained shallow head can still ask which frozen latent is more trajectory-readable.

#### Protocol.

Scenes and readout architecture are fixed; only the latent source changes. Each condition trains its own two-layer MLP. Per-horizon latents are spatially mean-pooled and concatenated in time; the backbone stays frozen. We use about 20k/4k train/val scenes, train 50 epochs, and report best ADE. As a diagnostic, we permute future horizons at evaluation: a large ADE rise means the readout uses cross-horizon structure. Sources: (i) current-frame features; (ii) ForeDrive predicted futures; (iii) prediction-optimized predicted futures (better WM reconstruction, weaker PDMS in the main paper); (iv) EMA target futures.

#### Results.

ForeDrive yields the lowest readout error (ADE $1.55$ vs. $1.83$ / $2.10$ / $2.31$; Table 9). Current is worst, so gains are not current-only leakage. ForeDrive beats the prediction-optimized future by about $0.28$ ADE (${\approx}15\%$). The latter matches EMA targets more closely yet scores lower PDMS: better reconstruction need not mean a more actionable future. ForeDrive also beats EMA-GT under this shallow probe, consistent with planning-joint training favoring trajectory-readable futures over scene-target fidelity. Horizon shuffle raises ADE by about $0.8$ – $1.5$ for all future sources, so latents retain cross-horizon structure. This contrasts with the planner’s near-invariance to shuffle (Sec. D): the DiT aggregates multi-scale context robustly, while the latent still preserves temporal structure for an ordered readout.

In short, under a frozen backbone and shallow readout, ForeDrive predicted futures regress expert plans more accurately than a more reconstruction-accurate predicted future, EMA targets, and current-only features. This provides evidence that predicted latents preserve trajectory-relevant information, rather than serving only as ordinary future-feature augmentation; we do not claim a planning-optimal representation.

## Appendix J Latent RGB Probe Visualization

The trajectory probe tests planning readability; we also check scene grounding. We train a lightweight RGB readout on a *frozen* ForeDrive encoder, without updating the world model or planner.

#### Probe architecture.

The readout is a lightweight upsampling decoder from frozen DINOv3 ViT-B patch tokens to $256\times 512$ RGB, trained with weighted L1 + LPIPS. The same decoder is shared across current and future frames.

#### Training protocol.

We freeze the jointly trained online/EMA vision encoder and fit only the probe on up to 20k cached training scenes. Inputs are frozen encodings of the current frame and WM future frames (about $0.5$ – $4\,\mathrm{s}$); targets are camera RGB. We optimize $\mathcal{L}=1.0\cdot\mathrm{L1}+0.1\cdot\mathrm{LPIPS}$ (AlexNet) with AdamW (lr $3\times 10^{-4}$, wd $10^{-4}$, batch 16) for 8 epochs. Training uses EMA latents only—not predictor rollouts—so the probe tests whether the latent space itself is RGB-decodable (final train loss $\approx 0.069$). At visualization time we decode both EMA targets and predictor rollouts with the same frozen probe. Predicted latents lie in the same space via the latent prediction objective, so qualitative comparison needs no predictor update.

#### Qualitative analysis.

Figure 4 shows four NAVSIM scenes. Reconstructions are intentionally coarse, as expected for a latent prediction space not trained for photorealism. Still, the probe recovers scene structure useful for planning diagnostics—road layout (c), nearby vehicles and buildings (a,d), and ego-motion-consistent viewpoint change (b)—with coherent evolution from $t{=}1$ to $t{=}4$. Predicted futures remain scene-grounded after asymmetric joint training; the RGB readout is diagnostic only and never used for planning.

## Appendix K Accuracy–Size Plot Data

Table 10 lists total parameter counts and NAVSIM scores for methods in the main accuracy–size figure.

<table><tbody><tr><th>Method</th><td>Params (M)</td><td>PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EPDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><th colspan="4">Traditional end-to-end methods</th></tr><tr><th>TransFuser</th><td>55.9</td><td>84.0</td><td>76.7</td></tr><tr><th>PRIX</th><td>37</td><td>87.8</td><td>84.2</td></tr><tr><th>DiffusionDrive</th><td>60.7</td><td>88.1</td><td>84.5</td></tr><tr><th>MeanFuser</th><td>54.6</td><td>89.0</td><td>89.5</td></tr><tr><th>DiffRefiner-R34</th><td>74.8</td><td>89.4</td><td>86.2</td></tr><tr><th>DriveSuprim-R34</th><td>61</td><td>89.9</td><td>83.1</td></tr><tr><th colspan="4">World-model and video–action methods</th></tr><tr><th>Epona</th><td>2500</td><td>86.2</td><td>85.1</td></tr><tr><th>DriveVLA-W0</th><td>7500</td><td>87.2</td><td>86.1</td></tr><tr><th>ForeDrive (ours)</th><td>118.4</td><td>89.9</td><td>90.0</td></tr></tbody></table>

Table 10: Total parameters and NAVSIM scores for the main accuracy–size figure. Same total-parameter definition throughout; self-reports preferred (TransFuser/DiffusionDrive from MeanFuser when needed).

We count all modules in the evaluated model (encoder, WM, planner, and test-time heads). Training-only EMA encoders are excluded when distinguished (ForeDrive: 118.4M). Self-reports are preferred; otherwise we use published third-party totals (MeanFuser for TransFuser/DiffusionDrive). Marker area in the main figure scales with $\log$ parameters. Among traditional end-to-end planners (37–75M), ForeDrive ties the best listed PDMS (89.9, with DriveSuprim-R34) and exceeds the strongest EPDMS (MeanFuser, 89.5) by 0.5 at 118.4M. Relative to Epona (2.5B) and DriveVLA-W0 (7.5B), it improves both aggregates at one to two orders of magnitude smaller scale.

[^1]: M. Assran, A. Bardes, D. Fan, Q. Garrido, R. Howes, M. Komeili, M. Muckley, A. Rizvi, C. Roberts, K. Sinha, A. Zholus, S. Arnaud, A. Gejji, A. Martin, F. R. Hogan, D. Dugas, P. Bojanowski, V. Khalidov, P. Labatut, F. Massa, M. Szafraniec, K. Krishnakumar, Y. Li, X. Ma, S. Chandar, F. Meier, Y. LeCun, M. Rabbat, and N. Ballas V-JEPA 2: self-supervised video models enable understanding, prediction and planning. External Links: 2506.09985 Cited by: §1.

[^2]: F. Baldassarre, M. Szafraniec, B. Terver, V. Khalidov, F. Massa, Y. LeCun, P. Labatut, M. Seitzer, and P. Bojanowski Back to the features: DINO as a foundation for video world models. External Links: 2507.19468 Cited by: §1.

[^3]: Y. Chen, Y. Wang, and Z. Zhang DrivingGPT: unifying driving world modeling and planning with multi-modal autoregressive transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 26890–26900. Cited by: §2.

[^4]: K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger TransFuser: imitation with transformer-based sensor fusion for autonomous driving. IEEE Transactions on Pattern Analysis and Machine Intelligence 45 (11), pp. 12878–12895. External Links: [Document](https://dx.doi.org/10.1109/TPAMI.2022.3200245) Cited by: §2, Table 1, Table 2.

[^5]: D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, A. Geiger, and K. Chitta NAVSIM: data-driven non-reactive autonomous vehicle simulation and benchmarking. In Advances in Neural Information Processing Systems, External Links: [Link](https://proceedings.neurips.cc/paper_files/paper/2024/hash/32768f7faf1995026ef9821c696f3404-Abstract-Datasets_and_Benchmarks_Track.html) Cited by: §4.

[^6]: A. Hu, Z. Murez, N. Mohan, S. Dudas, J. Hawke, V. Badrinarayanan, R. Cipolla, and A. Kendall FIERY: future instance prediction in bird’s-eye view from surround monocular cameras. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 15273–15282. Cited by: §2.

[^7]: S. Hu, L. Chen, P. Wu, H. Li, J. Yan, and D. Tao ST-P3: end-to-end vision-based autonomous driving via spatial-temporal feature learning. In European Conference on Computer Vision, pp. 533–549. Cited by: §4.

[^8]: Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, L. Lu, X. Jia, Q. Liu, J. Dai, Y. Qiao, and H. Li Planning-oriented autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 17853–17862. Cited by: §2, Table 1.

[^9]: B. Jiang, S. Chen, Q. Xu, B. Liao, J. Chen, H. Zhou, Q. Zhang, W. Liu, C. Huang, and X. Wang VAD: vectorized scene representation for efficient autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 8340–8350. Cited by: §4.

[^10]: E. Karypidis, I. Kakogeorgiou, S. Gidaris, and N. Komodakis DINO-Foresight: looking into the future with DINO. External Links: 2412.11673 Cited by: §1.

[^11]: Y. Li, L. Fan, J. He, Y. Wang, Y. Chen, Z. Zhang, and T. Tan LAW: enhancing end-to-end autonomous driving with latent world model. In International Conference on Learning Representations, Cited by: §1, §2, Table 1.

[^12]: Y. Li, S. Shang, W. Liu, B. Zhan, H. Wang, Y. Wang, Y. Chen, X. Wang, Y. An, C. Tang, L. Hou, L. Fan, and Z. Zhang DriveVLA-W0: world models amplify data scaling law in autonomous driving. In International Conference on Learning Representations, External Links: [Link](https://openreview.net/forum?id=plrGn3RdzN) Cited by: §2, Table 1, Table 2.

[^13]: Y. Li, Y. Wang, Y. Liu, J. He, L. Fan, and Z. Zhang End-to-end driving with online trajectory evaluation via BEV world model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 27137–27146. External Links: [Document](https://dx.doi.org/10.1109/ICCV51701.2025.02519) Cited by: §1, §2, Table 1.

[^14]: Y. Li, K. Xiong, X. Guo, F. Li, S. Yan, G. Xu, L. Zhou, L. Chen, H. Sun, B. Wang, K. Ma, G. Chen, H. Ye, W. Liu, and X. Wang ReCogDrive: a reinforced cognitive framework for end-to-end autonomous driving. In International Conference on Learning Representations, External Links: [Link](https://openreview.net/forum?id=JoXwhGbuMi) Cited by: §2, Table 1, Table 2.

[^15]: Z. Li, K. Li, S. Wang, S. Lan, Z. Yu, Y. Ji, Z. Li, Z. Zhu, J. Kautz, Z. Wu, Y. Jiang, and J. M. Alvarez Hydra-MDP: end-to-end multimodal planning with multi-target hydra-distillation. External Links: 2406.06978 Cited by: §2.

[^16]: Z. Li, W. Wang, H. Li, E. Xie, C. Sima, T. Lu, Q. Yu, and J. Dai BEVFormer: learning bird’s-eye-view representation from multi-camera images via spatiotemporal transformers. In European Conference on Computer Vision, pp. 1–18. External Links: [Document](https://dx.doi.org/10.1007/978-3-031-20077-9%5F1) Cited by: §2.

[^17]: B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang, and X. Wang DiffusionDrive: truncated diffusion model for end-to-end autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12037–12047. External Links: [Document](https://dx.doi.org/10.1109/CVPR52734.2025.01124) Cited by: §1, §2, §3, Table 1, Table 2.

[^18]: M. Oquab, T. Darcet, T. Moutakanni, H. V. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, M. Assran, N. Ballas, W. Galuba, R. Howes, P. Huang, S. Li, I. Misra, M. Rabbat, V. Sharma, G. Synnaeve, H. Xu, H. Jégou, J. Mairal, P. Labatut, A. Joulin, and P. Bojanowski DINOv2: learning robust visual features without supervision. Transactions on Machine Learning Research. Cited by: §3.

[^19]: O. Siméoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab, C. Jose, V. Khalidov, M. Szafraniec, S. Yi, M. Ramamonjisoa, F. Massa, D. Haziza, L. Wehrstedt, J. Wang, T. Darcet, T. Moutakanni, L. Sentana, C. Roberts, A. Vedaldi, J. Tolan, J. Brandt, C. Couprie, J. Mairal, H. Jégou, P. Labatut, and P. Bojanowski DINOv3. External Links: 2508.10104 Cited by: §3.

[^20]: J. Wang, Y. Zheng, X. Liu, Z. Xing, P. Li, K. Ma, H. Ye, G. Chen, G. Li, L. Chen, Z. Xia, and Q. Zhang MeanFuser: fast one-step multi-modal trajectory generation and adaptive reconstruction via meanflow for end-to-end autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 17884–17893. Cited by: §2, Table 1, Table 2.

[^21]: L. Wang, Y. Zheng, Q. Chen, S. Li, Y. Zhang, Z. Xing, Q. Zhang, X. Li, D. Qian, P. Yang, Y. Dong, C. Hao, X. Ye, J. han, Y. Pan, and D. Zhao Latent-wam: latent world action modeling for end-to-end autonomous driving. External Links: 2603.24581, [Link](https://arxiv.org/abs/2603.24581) Cited by: §2, Table 2.

[^22]: L. Wang, Z. Yang, C. Bai, G. Zhang, X. Liu, X. Zheng, X. Long, C. Lu, and C. Lu Drive-JEPA: video JEPA meets multimodal trajectory distillation for end-to-end driving. External Links: 2601.22032 Cited by: §1, §2, Table 2.

[^23]: Y. Wang, J. He, L. Fan, H. Li, Y. Chen, and Z. Zhang Driving into the future: multiview visual forecasting and planning with world model for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14749–14759. Cited by: §2.

[^24]: X. Weng, B. Ivanovic, Y. Wang, Y. Wang, and M. Pavone PARA-Drive: parallelized architecture for real-time autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15449–15458. Cited by: Table 1.

[^25]: M. K. Wozniak, L. Liu, Y. Cai, and P. Jensfelt PRIX: learning to plan from raw pixels for end-to-end autonomous driving. IEEE Robotics and Automation Letters 11 (5). External Links: [Document](https://dx.doi.org/10.1109/LRA.2026.3678836) Cited by: §2, Table 1, Table 2.

[^26]: T. Xia, Y. Li, L. Zhou, J. Yao, K. Xiong, H. Sun, B. Wang, K. Ma, G. Chen, H. Ye, W. Liu, and X. Wang DriveLaW: unifying planning and video generation in a latent driving world. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 39701–39712. Cited by: §1, §2, Table 1, Table 3.

[^27]: Z. Xing, X. Zhang, Y. Hu, B. Jiang, T. He, Q. Zhang, X. Long, and W. Yin GoalFlow: goal-driven flow matching for multimodal trajectories generation in end-to-end autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1602–1611. Cited by: §2.

[^28]: P. Yang, B. Lu, Z. Xia, C. Han, Y. Gao, T. Zhang, K. Zhan, X. Lang, Y. Zheng, and Q. Zhang WorldRFT: latent world model planning with reinforcement fine-tuning for autonomous driving. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40, pp. 11649–11657. External Links: [Document](https://dx.doi.org/10.1609/aaai.v40i14.38149) Cited by: Table 2.

[^29]: W. Yao, Z. Li, S. Lan, Z. Wang, X. Sun, J. M. Alvarez, and Z. Wu DriveSuprim: towards precise trajectory selection for end-to-end planning. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40, pp. 11910–11918. External Links: [Document](https://dx.doi.org/10.1609/aaai.v40i14.38178) Cited by: §2.

[^30]: L. Yin, R. Ju, G. Guo, and E. Cheng DiffRefiner: coarse to fine trajectory planning via diffusion refinement with semantic interaction for end to end autonomous driving. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40, pp. 12009–12017. External Links: [Document](https://dx.doi.org/10.1609/aaai.v40i14.38189) Cited by: §2, Table 1, Table 2.

[^31]: C. Yuan, Z. Zhang, J. Sun, S. Sun, Z. Huang, C. D. W. Lee, D. Li, Y. Han, A. Wong, K. P. Tee, and M. H. Ang DRAMA: an efficient end-to-end motion planner for autonomous driving with mamba. In International Symposium of Robotics Research, Cited by: Table 1.

[^32]: K. Zhang, Z. Tang, X. Hu, X. Pan, X. Guo, Y. Liu, J. Huang, L. Yuan, Q. Zhang, X. Long, X. Cao, and W. Yin Epona: autoregressive diffusion world model for autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 27220–27230. Cited by: §1, §2, Table 1, Table 2, Table 3.

[^33]: Z. Zhao, T. Fu, Y. Wang, L. Wang, and H. Lu From forecasting to planning: policy world model for collaborative state-action prediction. In Advances in Neural Information Processing Systems, Cited by: §2, Table 1, Table 3.

[^34]: W. Zheng, W. Chen, Y. Huang, B. Zhang, Y. Duan, and J. Lu OccWorld: learning a 3d occupancy world model for autonomous driving. In European Conference on Computer Vision, Cited by: §2.

[^35]: Y. Zheng, P. Yang, Z. Xing, Q. Zhang, Y. Zheng, Y. Gao, P. Li, T. Zhang, Z. Xia, P. Jia, and D. Zhao World4Drive: end-to-end autonomous driving via intention-aware physical latent world model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 28632–28642. Cited by: §1, §2, Table 2.

[^36]: G. Zhou, H. Pan, Y. LeCun, and L. Pinto DINO-WM: world models on pre-trained visual features enable zero-shot planning. In Proceedings of the 42nd International Conference on Machine Learning, Cited by: §1.
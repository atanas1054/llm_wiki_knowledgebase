---
title: "Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving"
source: "https://arxiv.org/html/2607.29031v1"
author:
published:
created: 2026-09-02
description:
tags:
  - "clippings"
---
Jiwei Yang    Zhengxian Chen    Chaosheng Huang    Jun Li

###### Abstract

Existing autonomous-driving world models typically perform dense prediction of future videos, occupancy states, BEV representations, or agent motion. We argue that planning need not reconstruct the complete future world, but only focus on scene features that affect future ego action. Based on this perspective, we propose Auto-JEPA, an action-oriented latent world model that learns continuous future driving intent through joint-embedding prediction. Given visual observations, ego-motion history, and navigation commands, Auto-JEPA predicts an intent embedding aligned with the latent representation of the future ego trajectory. The predicted intent retrieves executable trajectories from a fixed trajectory memory, which are then ranked by a scene-conditioned candidate selection module. Auto-JEPA keeps the visual encoder frozen, requires no explicit perception annotations, and uses no learned trajectory generator. By optimizing only task-specific modules for trajectory representation, intent prediction, and candidate selection, Auto-JEPA achieves 91.3 PDMS on NAVSIM v1 and 89.1 EPDMS on NAVSIM v2. Semantic occlusion experiments show that masking dynamic-agent regions induces an average intent change $2.97\times$ that of equal-area random masking. Moreover, occluding vehicles that affect future driving substantially changes the predicted intent and selected trajectory, whereas both remain essentially unchanged when non-influential vehicles are occluded. These results show that future-intent prediction encourages the model to focus on planning-relevant visual features and supports high-quality planning without dense future-world modeling. Code & Models: https://github.com/NoctYang/Auto-JEPA.

<sup>1</sup> School of Vehicle and Mobility, Tsinghua University

<sup>2</sup> State Key Laboratory of Intelligent Green Vehicle and Mobility, Tsinghua University

chenzhengxian@tsinghua.vin

<sup>†</sup>

## Introduction

![[intro_selective_future_intent.png|Refer to caption]]

Figure 1: Selective response to action-relevant scene information. Occluding a non-interacting vehicle causes little change in the predicted intent and plan, whereas occluding the interacting lead vehicle shifts the intent and selected trajectory. Annotations are used only for analysis.

World models offer a compelling route toward autonomous driving systems that can reason about how a scene may evolve before committing to an action. Many driving world models, however, formulate this objective as dense future prediction, generating future multiview observations, occupancy fields, or latent states that describe the evolution of the surrounding scene [^29] [^37] [^36] [^13]. A parallel line of work explicitly forecasts the future trajectories and interactions of multiple traffic participants [^23]. Although these objectives preserve rich information about the environment, they require the model to allocate substantial capacity to scene elements whose dynamics may have little influence on the ego vehicle’s immediate decision. Predicting all observable entities without regard to their planning relevance can therefore increase computation and propagate perception and forecasting errors into downstream planning.

We argue that a planning-oriented predictive model need not reconstruct the complete future state of a scene, but only focus on scene features that affect future ego action. Based on this perspective, we introduce Auto-JEPA, an action-oriented latent world model that learns continuous future driving intent through joint-embedding prediction. Given visual observations, ego-motion history, and a navigation command, it predicts a future ego-trajectory latent whose temporal tokens encode motion geometry and dynamics. Recent work explores planning-aligned latent prediction [^17] [^38] [^27] [^31], primarily for representation pretraining, auxiliary supervision, or action distillation. Auto-JEPA uses future ego-trajectory latents for trajectory retrieval.

Auto-JEPA first trains a trajectory encoder and then freezes it to define the future-trajectory target space. Following joint-embedding predictive learning [^3] [^4], a visual predictor infers the corresponding future-trajectory latent from the current scene context, ego-motion history, and route command. Latent alignment and contrastive objectives train the predictor without future-image reconstruction. Because its prediction target is defined solely by future ego motion, the model need not preserve all observable scene content and is encouraged to prioritize planning-relevant visual features. At inference, the predicted intent directly serves as the retrieval key of a fixed trajectory memory, and a scene-conditioned module selects the final executable candidate.

The trajectory memory contains recorded, kinematically plausible trajectory geometries encoded by the same trajectory encoder. A scene-conditioned scorer ranks the retrieved candidates, and a learned drivable-area gate screens candidates likely to violate drivable-area constraints before final selection. This design separates complementary responsibilities: the latent predictor determines *what kind of future motion is appropriate*, retrieval provides explicit trajectory geometry, the scorer estimates scene-level driving quality, and the gate reduces drivable-area violations. Although these components are optimized in stages, Auto-JEPA preserves an end-to-end sensor-to-trajectory planning interface without intermediate perception outputs or a learned trajectory generator.

We evaluate Auto-JEPA on NAVSIM v1 and NAVSIM v2 [^9]. Under this lightweight setting, Auto-JEPA achieves 91.3 PDMS on NAVSIM v1 and 89.1 EPDMS on NAVSIM v2. Figure 1 illustrates the different effects of individual vehicles within the same scene: occluding the interacting lead vehicle substantially changes the predicted intent and selected trajectory, whereas both remain essentially unchanged when the non-interacting adjacent vehicle is occluded. To quantify sensitivity to traffic-participant information at the dataset level, we further conduct controlled semantic occlusions on the full validation split. Masking dynamic-agent regions induces an average intent change $2.97\times$ that of equal-area random masking and a larger change in $71.1\%$ of scenes.

Our main contributions are summarized as follows:

- We formulate planning-oriented latent world modeling as future ego-trajectory latent prediction. Using continuous driving intent as the prediction target encourages the model to focus on visual features relevant to ego action, thereby avoiding dense future-scene reconstruction.
- We propose Auto-JEPA, which maps visual context into a future-trajectory latent space through joint-embedding prediction. The predicted intent directly retrieves executable candidates from a fixed trajectory memory, followed by scene-conditioned selection.
- Auto-JEPA achieves 91.3 PDMS on full NAVSIM v1 navtest and 89.1 EPDMS on NAVSIM v2; controlled semantic occlusion experiments further show that the model selectively focuses on more critical scene features.

## Related Work

### World Models for Autonomous Driving

Driving world models predict scene evolution in pixel or structured spaces for simulation, representation learning, and planning. Generative approaches synthesize controllable future driving videos from observations, actions, or language [^12] [^28] [^29] [^10], while structured alternatives forecast future occupancy or point clouds [^37] [^33]. Although these representations capture rich appearance and geometry, they require dense prediction over broad scene content, including details that need not determine the immediate ego plan.

Latent world models reduce explicit reconstruction by forecasting compact future features. LAW, World4Drive, DeepSight, and DriveWorld-VLA predict future scene features, BEV states, or action-conditioned scene evolution for downstream planning [^17] [^38] [^36] [^13]. Their targets nevertheless primarily describe how the surrounding scene evolves. Auto-JEPA instead predicts the representation of future ego motion itself, retaining scene dynamics through their implications for the planned trajectory.

### Joint-Embedding Predictive Learning

Joint-embedding predictive architectures learn by predicting target representations rather than reconstructing observations. I-JEPA establishes this principle for images [^3], V-JEPA extends it to video [^4], and V-JEPA 2 further demonstrates that self-supervised video representations can support understanding, prediction, and planning [^2]. Drive-JEPA adapts this family to driving-video pretraining and trajectory distillation [^27]. Auto-JEPA differs in the operational role of prediction: the future ego-trajectory latent directly serves as the planner’s retrieval key and therefore participates in trajectory planning at inference rather than only supporting training-time representation learning.

### End-to-End Trajectory Planning

End-to-end driving maps sensor observations and navigation context directly to an ego trajectory, but a single logged future provides limited coverage of multiple feasible maneuvers. Existing methods score candidates from offline trajectory vocabularies [^14] [^20] [^24], generate or refine proposals [^21] [^32] [^11], or combine a learned generator with a trajectory scorer [^1]. Latent trajectory nearest-neighbor search has also been explored for motion forecasting rather than ego planning [^5].

Recent VLA planners tokenize or decode ego actions from visual and language context [^40] [^39] [^18]. They learn parametric action decoders, whereas Auto-JEPA predicts a continuous future-ego-motion intent latent for non-parametric trajectory retrieval.

## Method

### Overview

Auto-JEPA predicts a continuous future ego-motion representation and uses it to retrieve executable trajectories. Given four front-camera frames, four historical ego positions, and a route command, a frozen V-JEPA 2 encoder [^2] extracts visual tokens, and a Transformer predictor fuses visual, motion, and route context to produce eight temporal latent tokens representing one continuous driving intent. Figure 3 presents the complete training and inference pipeline.

During training, the predicted intent is aligned with the representation of the ground-truth future trajectory produced by a frozen trajectory encoder. At inference, it retrieves the 300 most similar trajectories from a non-parametric memory; a scene-conditioned scorer ranks these candidates and a learned drivable-area gate filters infeasible proposals. This separates intent prediction, trajectory instantiation, quality ranking, and feasibility filtering without reconstructing future observations or directly regressing waypoint coordinates.

### Future Ego-Motion Intent Representation

Figure 2 details the two-stage learning process: trajectory-space pretraining followed by visual intent prediction. For each driving scene, we represent the ground-truth future ego trajectory as eight two-dimensional waypoints,

$$
\mathbf{Y}=[(x_{1},y_{1}),\ldots,(x_{8},y_{8})]\in\mathbf{R}^{8\times 2}.
$$

Before training the visual intent predictor, we learn the target space with a trajectory autoencoder composed of a trajectory encoder $E_{\mathrm{traj}}$ and a lightweight decoder $D_{\mathrm{traj}}$. The encoder maps a future trajectory to a sequence of latent tokens, and the decoder reconstructs the trajectory from this representation,

$$
\mathbf{Z}^{+}=E_{\mathrm{traj}}(\mathbf{Y})\in\mathbf{R}^{8\times 1024},\qquad\hat{\mathbf{Y}}=D_{\mathrm{traj}}(\mathbf{Z}^{+}).
$$

The autoencoder is optimized using a trajectory reconstruction objective,

$$
\mathcal{L}_{\mathrm{traj}}=\mathcal{L}_{\mathrm{xy}}+\lambda_{e}\mathcal{L}_{\mathrm{end}}+\lambda_{v}\mathcal{L}_{\mathrm{vel}}+\lambda_{a}\mathcal{L}_{\mathrm{acc}},
$$

where the four terms supervise waypoint coordinates, the final endpoint, velocity, and acceleration, respectively. After this pretraining stage, the decoder is discarded and $E_{\mathrm{traj}}$ is frozen. The frozen encoder then defines the target latent $\mathbf{Z}^{+}$ for intent prediction and encodes every trajectory in the retrieval memory, ensuring that prediction targets and retrieval candidates occupy the same latent space.

The eight latent tokens preserve trajectory time, geometry, and motion and jointly describe one continuous future realization rather than eight maneuver classes. We refer to this representation as the *driving intent*. Using the same encoder for supervision and memory construction places predicted intents and executable trajectories in a shared space, allowing the intent latent to serve directly as the retrieval key.

![[trajectory_latent_and_intent_prediction.png|Refer to caption]]

Figure 2: Trajectory-space pretraining and visual intent prediction in Auto-JEPA. Stage 1 learns the future-trajectory target space with a trajectory autoencoder; the decoder is then discarded and the trajectory encoder is frozen. Stage 2 predicts a continuous future driving-intent latent from visual observations, ego-motion history, and navigation commands, and aligns it with the frozen trajectory target representation.

![[intent_jepa_overview.png|Refer to caption]]

Figure 3: Overview of Auto-JEPA. During training, the predictor learns a continuous future ego-motion intent by aligning its predicted latent with the representation of the ground-truth future trajectory. During inference, the predicted intent retrieves 300 candidates from a ground-truth-only latent trajectory memory; a scene-conditioned scorer ranks candidate quality and an independent feasibility gate filters drivable-area violations before final selection. Snowflake symbols indicate frozen modules.

### Visual Intent Prediction

Given the current visual observation, ego-motion history, and navigation command, Auto-JEPA predicts the corresponding future ego-motion latent. We denote the model input as

$$
\mathbf{X}=(\mathbf{I},\mathbf{H},\mathbf{C}),
$$

where $\mathbf{I}$ contains four historical frames from the front-facing camera, $\mathbf{H}\in\mathbf{R}^{4\times 2}$ contains four historical ego positions, and $\mathbf{C}\in\mathbf{R}^{4}$ denotes the route command. A frozen V-JEPA 2 visual encoder $E_{\mathrm{vis}}$ extracts visual tokens,

$$
\mathbf{F}_{v}=E_{\mathrm{vis}}(\mathbf{I}).
$$

The history encoder $E_{\mathrm{hist}}$ and command encoder $E_{\mathrm{cmd}}$ map the remaining inputs to the predictor feature dimension,

$$
\mathbf{F}_{h}=E_{\mathrm{hist}}(\mathbf{H}),\qquad\mathbf{F}_{c}=E_{\mathrm{cmd}}(\mathbf{C}).
$$

The visual, history, and command features condition a JEPA predictor $P_{\theta}$ composed of 24 Transformer blocks [^26] with a hidden dimension of 1024 and 16 attention heads. The predictor outputs eight future latent tokens,

$$
\hat{\mathbf{Z}}=P_{\theta}(\mathbf{F}_{v},\mathbf{F}_{h},\mathbf{F}_{c})\in\mathbf{R}^{8\times 1024}.
$$

The prediction $\hat{\mathbf{Z}}$ matches the temporal organization and feature dimension of $\mathbf{Z}^{+}$. During driving-domain training, the visual encoder remains frozen, while the history encoder, command encoder, and JEPA predictor are optimized to aggregate visual evidence, ego dynamics, and navigation constraints without explicitly predicting intermediate scene states.

### Joint-Embedding Training Objectives

Rather than directly supervising waypoint coordinates with ADE or FDE, Auto-JEPA optimizes predictions in the frozen trajectory latent space using feature alignment, token-wise cosine alignment, and batch-level InfoNCE [^25].

#### Feature alignment.

We first normalize the predicted and target latents and apply a Smooth L1 loss to their feature values,

$$
\mathcal{L}_{\mathrm{feat}}=\mathrm{SmoothL1}\!\left(\mathrm{Norm}(\hat{\mathbf{Z}}),\mathrm{Norm}(\mathbf{Z}^{+})\right).
$$

#### Token-wise cosine alignment.

For the eight future time positions, we minimize the cosine distance between corresponding predicted and target tokens,

$$
\mathcal{L}_{\mathrm{cos}}=\frac{1}{8}\sum_{t=1}^{8}\left(1-\frac{\hat{\mathbf{z}}_{t}^{\top}\mathbf{z}^{+}_{t}}{\|\hat{\mathbf{z}}_{t}\|_{2}\|\mathbf{z}^{+}_{t}\|_{2}}\right).
$$

#### Batch-level InfoNCE.

Positive alignment alone may map distinct driving scenes to similar representations. We therefore flatten and normalize each complete latent sequence,

$$
\hat{\mathbf{q}}_{i}=\mathrm{Norm}(\mathrm{vec}(\hat{\mathbf{Z}}_{i})),\qquad\mathbf{k}_{j}=\mathrm{Norm}(\mathrm{vec}(\mathbf{Z}^{+}_{j})).
$$

For scene $i$, its target latent is the positive and target latents from other scenes serve as negatives,

$$
\mathcal{L}_{\mathrm{NCE}}=-\frac{1}{B}\sum_{i=1}^{B}\log\frac{\exp(\hat{\mathbf{q}}_{i}^{\top}\mathbf{k}_{i}/\tau)}{\sum_{j=1}^{B}\exp(\hat{\mathbf{q}}_{i}^{\top}\mathbf{k}_{j}/\tau)},
$$

where $\tau=0.07$. During distributed training, target latents are gathered across GPUs to enlarge the negative set.

The complete intent-prediction objective is

$$
\mathcal{L}_{\mathrm{intent}}=0.1\mathcal{L}_{\mathrm{feat}}+2.0\mathcal{L}_{\mathrm{cos}}+\mathcal{L}_{\mathrm{NCE}}.
$$

Together, these objectives align local features and temporal-token semantics while preserving discrimination across driving scenes, making the predicted latent suitable for trajectory retrieval.

### Non-Parametric Trajectory Retrieval

To ground the predicted continuous intent into explicit trajectory geometry, we construct a non-parametric trajectory memory using only ground-truth driving trajectories. Each memory trajectory $\mathbf{Y}_{n}$ is encoded by the same frozen trajectory encoder used to define the training targets,

$$
\mathbf{Z}_{n}=E_{\mathrm{traj}}(\mathbf{Y}_{n})\in\mathbf{R}^{8\times 1024}.
$$

The resulting memory is

$$
\mathcal{M}=\left\{(\mathbf{Z}_{n},\mathbf{Y}_{n})\right\}_{n=1}^{N},\qquad N=110{,}335.
$$

Each entry retains its latent and waypoint coordinates, directly providing candidate geometry.

For a predicted intent $\hat{\mathbf{Z}}$, we flatten and L2-normalize the query and memory latents,

$$
\mathbf{q}=\mathrm{Norm}(\mathrm{vec}(\hat{\mathbf{Z}})),\qquad\mathbf{m}_{n}=\mathrm{Norm}(\mathrm{vec}(\mathbf{Z}_{n})).
$$

Their retrieval similarity is the flat cosine similarity

$$
r_{n}=\mathbf{q}^{\top}\mathbf{m}_{n}.
$$

We rank the complete memory by $r_{n}$ and retrieve the $K=300$ most similar trajectories,

$$
\mathcal{C}=\mathrm{TopK}\!\left(\{r_{n}\}_{n=1}^{N},K\right).
$$

Retrieval identifies intent-compatible geometry, while subsequent modules handle scene-level safety. Because candidates come from recorded trajectories, inference requires neither an additional learned trajectory generator nor iterative waypoint sampling.

### Scene Scoring and Feasibility Gating

Latent similarity measures intent compatibility but not scene-level safety. We therefore use a scene-conditioned quality scorer and an independently trained drivable-area feasibility gate.

#### Scene-conditioned utility branch.

Given scene features $\mathbf{F}_{\mathrm{scene}}$, ego context $\mathbf{e}$, and a candidate trajectory $\mathbf{Y}_{k}$, the scorer predicts a quality score

$$
s_{k}=S_{\phi}(\mathbf{F}_{\mathrm{scene}},\mathbf{e},\mathbf{Y}_{k}).
$$

We initialize $S_{\phi}$ from the publicly released CLOVER trajectory scorer [^1] and re-optimize its trainable modules on the ground-truth-only candidates retrieved by Auto-JEPA. Training uses collision, drivable-area, time-to-collision, comfort, and ego-progress supervision [^9], together with a within-scene ranking objective. The scorer and gate are trained exclusively on candidates from the NAVSIM training split, using labels generated offline by the NAVSIM/CLOVER get\_sub\_score evaluator with the batched navsim\_v1\_style relabeling protocol. This protocol disables per-proposal two-way rollout during label generation; the final benchmark results are evaluated separately under the official NAVSIM protocols. Scorer training is separate from intent prediction, with no gradients propagated to the visual encoder, JEPA predictor, or trajectory memory.

#### Drivable-area feasibility gate.

To explicitly reject candidates at risk of leaving the drivable region, the independently trained feasibility gate predicts

$$
p^{\mathrm{DAC}}_{k}=G_{\psi}(\mathbf{F}_{\mathrm{scene}},\mathbf{e},\mathbf{Y}_{k}),
$$

where $p^{\mathrm{DAC}}_{k}$ is the predicted probability of a DAC failure. At inference, we form a safety mask using $\tau_{\mathrm{DAC}}=0.2$,

$$
m_{k}=\mathbf{1}\!\left[p^{\mathrm{DAC}}_{k}\leq\tau_{\mathrm{DAC}}\right].
$$

Only candidates that pass the gate participate in final selection. If all are rejected, the system falls back to ungated utility ranking. Evaluator labels are unavailable to the gate at inference time.

The final trajectory is selected by a masked argmax over utility scores,

$$
k^{*}=\arg\max_{k:m_{k}=1}s_{k},\qquad\mathbf{Y}^{*}=\mathbf{Y}_{k^{*}}.
$$

Thus, intent prediction supplies the retrieval query, memory provides trajectory geometry, and the scorer–gate cascade performs scene-conditioned selection.

Table 1: Comparison with representative methods on NAVSIM v1 navtest. C and L denote camera and LiDAR input, respectively. NC, DAC, TTC, C, EP, and PDMS denote no-at-fault collision, drivable-area compliance, time to collision, comfort, ego progress, and the Predictive Driver Model Score.

<table><tbody><tr><th>Method</th><th>Venue</th><th>Sensors</th><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>C <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><th>Human</th><th>–</th><th>–</th><td>100.0</td><td>100.0</td><td>100.0</td><td>99.9</td><td>87.5</td><td>94.8</td></tr><tr><th colspan="9"><em>End-to-End Planning Methods</em></th></tr><tr><th>TransFuser <sup><a href="#fn:8">8</a></sup></th><th>TPAMI 2023</th><th><math><semantics><mrow><mn>3</mn> <mo>×</mo></mrow> <annotation>3\times</annotation></semantics></math> C+L</th><td>97.7</td><td>92.8</td><td>92.8</td><td>100.0</td><td>79.2</td><td>84.0</td></tr><tr><th>PARA-Drive <sup><a href="#fn:30">30</a></sup></th><th>CVPR 2024</th><th><math><semantics><mrow><mn>6</mn> <mo>×</mo></mrow> <annotation>6\times</annotation></semantics></math> C</th><td>97.9</td><td>92.4</td><td>93.0</td><td>99.8</td><td>79.3</td><td>84.0</td></tr><tr><th>Hydra-MDP <sup><a href="#fn:20">20</a></sup></th><th>CVPR 2024</th><th><math><semantics><mrow><mn>3</mn> <mo>×</mo></mrow> <annotation>3\times</annotation></semantics></math> C+L</th><td>98.3</td><td>96.0</td><td>94.6</td><td>100.0</td><td>78.7</td><td>86.5</td></tr><tr><th>DiffusionDrive <sup><a href="#fn:21">21</a></sup></th><th>CVPR 2025</th><th><math><semantics><mrow><mn>3</mn> <mo>×</mo></mrow> <annotation>3\times</annotation></semantics></math> C+L</th><td>98.2</td><td>96.2</td><td>94.7</td><td>100.0</td><td>82.2</td><td>88.1</td></tr><tr><th colspan="9"><em>World-Model-Based Methods</em></th></tr><tr><th>LAW <sup><a href="#fn:17">17</a></sup></th><th>ICLR 2025</th><th><math><semantics><mrow><mn>1</mn> <mo>×</mo></mrow> <annotation>1\times</annotation></semantics></math> C</th><td>96.4</td><td>95.4</td><td>88.7</td><td>99.9</td><td>81.7</td><td>84.6</td></tr><tr><th>DrivingGPT <sup><a href="#fn:7">7</a></sup></th><th>ICCV 2025</th><th><math><semantics><mrow><mn>1</mn> <mo>×</mo></mrow> <annotation>1\times</annotation></semantics></math> C</th><td>98.9</td><td>90.7</td><td>94.9</td><td>95.6</td><td>79.7</td><td>82.4</td></tr><tr><th>WoTE <sup><a href="#fn:19">19</a></sup></th><th>ICCV 2025</th><th><math><semantics><mrow><mn>3</mn> <mo>×</mo></mrow> <annotation>3\times</annotation></semantics></math> C+L</th><td>98.5</td><td>96.8</td><td>94.4</td><td>99.9</td><td>81.9</td><td>88.3</td></tr><tr><th>Epona <sup><a href="#fn:35">35</a></sup></th><th>ICCV 2025</th><th><math><semantics><mrow><mn>3</mn> <mo>×</mo></mrow> <annotation>3\times</annotation></semantics></math> C</th><td>97.9</td><td>95.1</td><td>93.8</td><td>99.9</td><td>80.4</td><td>86.2</td></tr><tr><th colspan="9"><em>VLA-Based Methods</em></th></tr><tr><th>AutoVLA <sup><a href="#fn:40">40</a></sup></th><th>NeurIPS 2025</th><th><math><semantics><mrow><mn>3</mn> <mo>×</mo></mrow> <annotation>3\times</annotation></semantics></math> C</th><td>98.4</td><td>95.6</td><td>98.0</td><td>99.9</td><td>81.9</td><td>89.1</td></tr><tr><th>RecogDrive <sup><a href="#fn:16">16</a></sup></th><th>ICLR 2026</th><th><math><semantics><mrow><mn>3</mn> <mo>×</mo></mrow> <annotation>3\times</annotation></semantics></math> C</th><td>98.2</td><td>97.8</td><td>95.2</td><td>99.8</td><td>83.5</td><td>89.6</td></tr><tr><th>AdaThinkDrive <sup><a href="#fn:22">22</a></sup></th><th>ICRA 2026</th><th><math><semantics><mrow><mn>1</mn> <mo>×</mo></mrow> <annotation>1\times</annotation></semantics></math> C</th><td>98.4</td><td>97.8</td><td>95.2</td><td>100.0</td><td>84.4</td><td>90.3</td></tr><tr><th>DriveVLA-W0 <sup><a href="#fn:18">18</a></sup></th><th>ICLR 2026</th><th><math><semantics><mrow><mn>1</mn> <mo>×</mo></mrow> <annotation>1\times</annotation></semantics></math> C</th><td>98.7</td><td>99.1</td><td>95.3</td><td>99.3</td><td>83.3</td><td>90.2</td></tr><tr><th>Curious-VLA <sup><a href="#fn:6">6</a></sup></th><th>CVPR 2026 Findings</th><th><math><semantics><mrow><mn>1</mn> <mo>×</mo></mrow> <annotation>1\times</annotation></semantics></math> C</th><td>98.4</td><td>96.9</td><td>97.9</td><td>98.1</td><td>88.5</td><td>90.3</td></tr><tr><th>Auto-JEPA (Ours)</th><th>–</th><th><math><semantics><mrow><mn>1</mn> <mo>×</mo></mrow> <annotation>1\times</annotation></semantics></math> C</th><td>98.4</td><td>98.3</td><td>95.0</td><td>100.0</td><td>87.1</td><td>91.3</td></tr></tbody></table>

Table 2: Comparison with representative methods on NAVSIM v2. When multiple backbones are reported, we use the strongest source-reported configuration. For Auto-JEPA, the unmarked result uses the original evaluation implementation, while the result marked with <sup>†</sup> uses the updated official implementation. Results for other methods are source-reported.

<table><tbody><tr><th>Method</th><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DDC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TL <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>LK <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>HC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EPDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><th colspan="11"><em>End-to-End Planning Methods</em></th></tr><tr><th>TransFuser <sup><a href="#fn:8">8</a></sup></th><td>96.9</td><td>89.9</td><td>97.8</td><td>99.7</td><td>87.1</td><td>95.4</td><td>92.7</td><td>98.3</td><td>87.2</td><td>76.7</td></tr><tr><th>VADv2 <sup><a href="#fn:14">14</a></sup></th><td>97.3</td><td>91.7</td><td>98.2</td><td>99.9</td><td>77.6</td><td>92.7</td><td>66.0</td><td>100.0</td><td>97.4</td><td>76.6</td></tr><tr><th>DiffusionDrive <sup><a href="#fn:21">21</a></sup></th><td>98.2</td><td>95.9</td><td>99.4</td><td>99.8</td><td>87.5</td><td>97.3</td><td>96.8</td><td>98.3</td><td>87.7</td><td>84.5</td></tr><tr><th>HydraMDP++ (ViT-L) <sup><a href="#fn:15">15</a></sup></th><td>98.5</td><td>98.5</td><td>99.5</td><td>99.7</td><td>87.4</td><td>97.9</td><td>95.8</td><td>98.2</td><td>75.7</td><td>85.6</td></tr><tr><th>DriveSuprim (ViT-L) <sup><a href="#fn:34">34</a></sup></th><td>98.4</td><td>98.6</td><td>99.6</td><td>99.8</td><td>90.5</td><td>97.8</td><td>97.0</td><td>98.3</td><td>78.6</td><td>87.1</td></tr><tr><th colspan="11"><em>VLA-Based Methods</em></th></tr><tr><th>DriveVLA-W0 <sup><a href="#fn:18">18</a></sup></th><td>98.5</td><td>99.1</td><td>98.0</td><td>99.7</td><td>86.4</td><td>98.1</td><td>93.2</td><td>97.9</td><td>58.9</td><td>86.1</td></tr><tr><th>ReCogDrive <sup><a href="#fn:16">16</a></sup></th><td>98.3</td><td>95.2</td><td>99.5</td><td>99.8</td><td>87.1</td><td>97.5</td><td>96.6</td><td>98.3</td><td>86.5</td><td>83.6</td></tr><tr><th>Curious-VLA <sup><a href="#fn:6">6</a></sup></th><td>98.4</td><td>96.9</td><td>99.2</td><td>99.8</td><td>88.5</td><td>97.9</td><td>96.9</td><td>98.1</td><td>81.5</td><td>85.3</td></tr><tr><th>DriveWorld-VLA <sup><a href="#fn:13">13</a></sup></th><td>98.6</td><td>99.1</td><td>99.6</td><td>99.8</td><td>87.4</td><td>97.9</td><td>97.0</td><td>97.8</td><td>78.6</td><td>86.8</td></tr><tr><th>Auto-JEPA (Ours)</th><td>98.5</td><td>98.7</td><td>98.2</td><td>97.2</td><td>90.5</td><td>97.9</td><td>84.0</td><td>97.8</td><td>75.4</td><td>85.6</td></tr><tr><th>Auto-JEPA (Ours) <sup>†</sup></th><td>98.5</td><td>98.7</td><td>98.3</td><td>99.7</td><td>90.5</td><td>97.9</td><td>94.7</td><td>97.8</td><td>75.2</td><td>89.1</td></tr></tbody></table>

![[selective_intent_three_scene_compact.png|Refer to caption]]

Figure 4: Selective responses to traffic participants. Cyan and rose denote occlusions of lower- and higher-impact vehicles, respectively. Curves show deviation from the unoccluded trajectory on a shared 0–4 m scale; Δ p T \\Delta p\_{T} lists their terminal deviations in the same order. The smaller stop-and-go shift reflects its restricted motion range: the unoccluded plan moves only 0.17 m.

![[semantic_occlusion_three_scene_panels.png|Refer to caption]]

Figure 5: Representative controls from the full-validation semantic occlusion protocol. For each scene, dynamic-agent regions and independently sampled equal-area random regions are masked consistently across all four input frames. The bars report cosine similarity to the unoccluded intent.

## Experiments

We evaluate the complete retrieval-based planner under the official NAVSIM v1 and v2 protocols. Unless otherwise stated, all experiments use the same intent predictor, ground-truth-only trajectory memory, scene-conditioned scorer, and feasibility gate. We first report benchmark results, then isolate the effects of intent prediction and candidate selection, followed by analyses of candidate-pool size and visual dependence.

### Dataset and Metrics

#### NAVSIM v1.

NAVSIM is a data-driven, non-reactive benchmark for autonomous-driving planning that provides large-scale trainval data and simulation-based evaluation on navtest [^9]. The v1 benchmark evaluates each predicted ego trajectory through a simulator and reports no-at-fault collision (NC), drivable-area compliance (DAC), time to collision (TTC), comfort (C), and ego progress (EP). These terms are aggregated into the Predictive Driver Model Score (PDMS):

$$
\mathrm{PDMS}=\mathrm{NC}\,\mathrm{DAC}\,\frac{5(\mathrm{EP}+\mathrm{TTC})+2\mathrm{C}}{12}.
$$

We train on NAVSIM trainval and report results on the complete navtest split of 12,146 scenarios.

#### NAVSIM v2.

NAVSIM v2 extends the evaluation to a broader set of driving-quality and rule-compliance properties. In addition to NC and DAC, it measures driving-direction compliance (DDC), traffic-light compliance (TL), ego progress (EP), time to collision (TTC), lane keeping (LK), history comfort (HC), and extended comfort (EC), and aggregates them into EPDMS. We evaluate v2 with the updated official implementation and human-behavior filtering enabled. Since v1 and v2 use different rollout and aggregation protocols, we report them separately without converting scores between the two metrics.

### Implementation Details

The model takes four $256\times 256$ front-camera frames, four historical ego positions, and a route command. Its 24-layer Transformer predictor has 16 heads and a hidden dimension of 1024, and outputs eight temporal latent tokens. We train with a per-GPU batch size of 8, learning rate $10^{-5}$, weight decay $0.05$, BF16 arithmetic, and an InfoNCE temperature of 0.07. The visual and target trajectory encoders remain frozen during predictor training. No object boxes, occupancy, semantic-map, or surrounding-agent motion labels are used. The scorer and gate use 75,823 training scenes and 8,459 validation scenes under a scene-prefix split; the validation and navtest sets have zero overlap in metric tokens. At inference, flat-cosine retrieval selects 300 candidates from a memory of 110,335 ground-truth trajectory–latent pairs; the scene scorer and feasibility gate then select the final trajectory using a DAC-failure threshold of 0.2.

### Results on NAVSIM v1

As shown in Table 2, Auto-JEPA achieves 91.3 PDMS using only a front camera, with 98.4 NC, 98.3 DAC, and 100.0 comfort.

### Results on NAVSIM v2

Table 2 reports 85.6 EPDMS under the original evaluator and 89.1 EPDMS under the updated official implementation with human-behavior filtering. CLOVER reports 90.4 EPDMS with a learned generator–scorer pipeline; Auto-JEPA remains competitive without parametric proposal generation. The evaluator change mainly affects TL and LK.

### Ablation Studies

#### Component ablation.

We ablate each component using the same Top-300 memory. Replacing the predicted intent with a fixed codebook medoid reduces PDMS from 91.3 to 52.6, confirming scene-conditioned retrieval. Intent-based cosine retrieval with the gate already reaches 87.6; adding the scorer contributes 3.7 points. Removing the gate yields 91.0 PDMS and lowers DAC from 98.3 to 97.9.

Table 3: Component ablation on full NAVSIM v1 navtest. Checkmarks indicate enabled components.

| Intent | Scorer | Gate | NC | DAC | TTC | C | EP | PDMS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ✗ | ✓ | ✓ | 83.1 | 85.3 | 76.2 | 86.3 | 37.8 | 52.6 |
| ✓ | ✗ | ✓ | 98.1 | 96.4 | 94.0 | 100.0 | 81.7 | 87.6 |
| ✓ | ✓ | ✗ | 98.5 | 97.9 | 95.0 | 99.9 | 86.9 | 91.0 |
| ✓ | ✓ | ✓ | 98.4 | 98.3 | 95.0 | 100.0 | 87.1 | 91.3 |

#### Candidate-pool size sensitivity.

The system obtains 87.6, 91.1, and 91.3 PDMS for $K=1$, 200, and 300. Increasing $K$ from 1 to 200 improves PDMS by 3.5 points, demonstrating the value of candidate selection, whereas the marginal gain from 200 to 300 indicates that performance is approaching saturation.

### Analysis

#### Selective sensitivity to dynamic-agent information.

Across 15,364 validation samples, masking all dynamic agents in all four frames produces a mean intent change ($1-$ cosine similarity) of 0.080, compared with 0.027 for independently sampled equal-area masks, a $2.97\times$ increase; the dynamic-agent intervention is larger in 71.1% of samples. Figure 5 shows matched controls, while Figure 4 isolates individual vehicles. Vehicles affecting future driving cause larger latent and trajectory changes than low-impact vehicles. Differences grow with the horizon in the open-road and lead-interaction scenes, whereas the smaller stop-and-go response reflects an unoccluded plan that advances only 0.17 m. Although the predictor receives no object boxes, agent identities, interaction labels, or surrounding-agent motion annotations, these responses emerge from future ego-trajectory representation supervision. These results show that the model does not respond uniformly to all dynamic agents, but focuses more strongly on vehicles that may affect future driving decisions.

## Conclusion

This paper introduced Auto-JEPA, an action-oriented latent world model that predicts continuous future driving intent. Joint-embedding prediction maps visual observations, ego-motion history, and navigation commands into the latent space of future ego trajectories; the predicted intent then retrieves executable candidates from a fixed memory for scene-conditioned selection. Auto-JEPA achieves 91.3 PDMS on NAVSIM v1 and 89.1 EPDMS on NAVSIM v2. Ablations and controlled occlusions validate the model’s selective focus on scene features, supporting future-trajectory latent as a decision-relevant prediction target without dense future-world modeling. The current system remains limited by memory coverage and selection calibration; intent-conditioned trajectory generation or refinement provides a natural extension.

## References

## Supplementary Material

This supplementary material provides additional implementation details, training protocols, and analysis settings for Auto-JEPA. The organization follows the staged training and inference pipeline described in the main paper.

## Appendix A Additional Implementation Details

### Trajectory Latent-Space Pretraining

The trajectory representation is learned before visual intent prediction. Each future ego trajectory is represented by eight planar waypoints,

$$
\mathbf{Y}\in\mathbf{R}^{8\times 2},
$$

covering a four-second planning horizon at 0.5-second intervals. Coordinates are normalized by a scale factor of 64 before being processed by the trajectory autoencoder. The trajectory encoder contains four Transformer blocks with a hidden dimension of 1024, 16 attention heads, an MLP ratio of 4, and eight Fourier frequency bands for coordinate encoding. It produces eight temporally aligned latent tokens,

$$
\mathbf{Z}^{+}=E_{\mathrm{traj}}(\mathbf{Y})\in\mathbf{R}^{8\times 1024}.
$$

The lightweight decoder contains four self-attention blocks with the same hidden dimension and number of attention heads. It predicts waypoint increments, which are cumulatively summed to obtain the reconstructed trajectory. The autoencoder is optimized with coordinate, endpoint, velocity, and acceleration consistency losses:

$$
\mathcal{L}_{\mathrm{traj}}=\mathcal{L}_{xy}+2.0\mathcal{L}_{\mathrm{end}}+0.5\mathcal{L}_{\mathrm{vel}}+0.2\mathcal{L}_{\mathrm{acc}}.
$$

We use a batch size of 256, a learning rate of $2\times 10^{-4}$, weight decay of 0.05, dropout of 0.1, gradient clipping at 1.0, and BF16 arithmetic. The model is trained for up to 40 epochs, and the checkpoint with the lowest validation ADE is retained. After this stage, the decoder is discarded and the trajectory encoder is frozen for both intent supervision and trajectory-memory construction.

### Visual Intent Predictor

The visual intent predictor receives four front-camera frames, four historical ego positions, and a route command. The input images are resized to $256\times 256$ and encoded by a frozen V-JEPA 2 visual encoder. The history and route inputs are projected into the same 1024-dimensional feature space. A 24-layer Transformer predictor with 16 attention heads fuses these inputs with eight learnable future-time query tokens and predicts

$$
\hat{\mathbf{Z}}\in\mathbf{R}^{8\times 1024}.
$$

The eight output tokens correspond to future time steps and jointly represent a single continuous driving intent; they are not separate maneuver queries.

The predictor is trained against the frozen target representation $\mathbf{Z}^{+}$ using feature alignment, token-wise cosine alignment, and batch-level contrastive learning:

$$
\mathcal{L}_{\mathrm{intent}}=0.1\mathcal{L}_{\mathrm{feat}}+2.0\mathcal{L}_{\mathrm{cos}}+\mathcal{L}_{\mathrm{NCE}}.
$$

The InfoNCE loss uses flattened trajectory latents and a temperature of 0.07. Training uses a per-GPU batch size of 8, learning rate $10^{-5}$, weight decay of 0.05, dropout of 0.1, BF16 arithmetic, and gradient clipping at 1.0. The image augmentation applies temporal frame masking with probability 0.3, masking between one and three input frames, and random image erasing with probability 0.2. When an image frame is masked, the corresponding historical ego position is masked as well.

### Trajectory Memory and Retrieval

The trajectory memory contains 110,335 ground-truth trajectory–latent pairs constructed from the NAVSIM training data. Every trajectory is encoded once with the frozen trajectory encoder. At inference, the predicted intent and memory latents are flattened and $\ell_{2}$ normalized, and retrieval is performed by flat cosine similarity. The final configuration retrieves the Top-300 candidates. The memory is fixed during planner training and inference, and the NAVSIM navtest scenes are not included in memory construction.

## Appendix B Dataset and Evaluation Protocol

### Candidate-Label Generation and Data Split

The scene scorer and DAC gate are trained only on candidates retrieved for NAVSIM training scenes. Candidate labels are generated offline from the NAVSIM training metric cache using the NAVSIM/CLOVER get\_sub\_score evaluator. We use the batched navsim\_v1\_style relabeling path, with per-proposal two-way rollout disabled during label generation. The resulting labels include no-at-fault collision, drivable-area compliance, ego progress, time to collision, comfort, and the aggregate utility target used for scorer training. Final benchmark results are computed separately with the official NAVSIM v1 or v2 evaluation pipeline rather than with these training labels.

The candidate dataset contains 75,823 training scenes and 8,459 validation scenes. We use a scene-prefix split to prevent temporally adjacent samples from the same scene sequence from appearing in both subsets. The validation set and NAVSIM navtest have zero overlap in metric tokens.

## Appendix C Scene Scorer and Feasibility Gate

### Scene-Conditioned Scorer

The scorer is initialized from the publicly released CLOVER trajectory scorer and re-optimized on the ground-truth-only retrieval distribution of Auto-JEPA. Its input consists of scene features, ego context, and candidate trajectory features. The optimization objective combines trajectory-component regression with within-scene ranking:

$$
\mathcal{L}_{\mathrm{score}}=\mathcal{L}_{\mathrm{comp}}+0.5\mathcal{L}_{\mathrm{rank}}.
$$

The ranking loss uses a temperature of 0.05 and treats candidates within 0.02 of the best target score as near-optimal. Comfort-failure candidates receive a weight of 5.0. The initial adaptation is trained for five epochs with a batch size of 32, learning rate $10^{-5}$ for the scorer modules, learning rate $10^{-4}$ for the ego adapter, and weight decay of 0.01. We then perform a three-epoch low-learning-rate continuation with learning rates $2\times 10^{-6}$ and $2\times 10^{-5}$, respectively. The checkpoint with the highest validation selected score is used in the final planner.

### Drivable-Area Feasibility Gate

The feasibility gate predicts the probability that a candidate violates drivable-area constraints. It operates on the frozen candidate features used by the scorer, seven trajectory-kinematic features, and candidate-set context. The kinematic vector contains the measured ego speed, the first two candidate speeds, their initial speed mismatch relative to the ego vehicle, the absolute mismatch, and two finite-difference acceleration terms. Candidate-set self-attention allows the gate to compare each proposal with alternative motions retrieved for the same scene. The prediction trunk has a hidden dimension of 256 and dropout of 0.1.

The gate is trained with weighted binary cross-entropy and a scene-wise pairwise ranking loss,

$$
\mathcal{L}_{\mathrm{gate}}=\mathcal{L}_{\mathrm{BCE}}+0.3\mathcal{L}_{\mathrm{rank}},
$$

where DAC-failing candidates receive a positive-class weight of 8 and the ranking margin is 1.0. We use AdamW with a batch size of 32, learning rate $3\times 10^{-4}$, weight decay $10^{-3}$, gradient clipping at 1.0, and BF16 arithmetic. During inference, candidates with predicted failure probability above $\tau=0.2$ are masked before the scorer argmax. If a scene has no remaining candidate, the system restores the ungated scorer ranking for that scene. Evaluator labels are never available to the gate during inference.

## Appendix D Additional Ablation Details

### Full Component Ablation

The component ablation in the main paper isolates the roles of intent prediction, scene-conditioned scoring, and feasibility filtering. The scorer-free row still uses the DAC gate, whereas the no-gate row still uses the scene-conditioned scorer. Therefore, the comparisons should be interpreted conditionally rather than as a sequential addition of modules. The scorer contributes 3.7 PDMS when feasibility filtering is retained, while the gate contributes 0.3 PDMS and improves DAC by 0.4 points when the scorer is retained.

### Candidate-Pool Size Sensitivity

Table 4 reports the final planning score for the candidate-pool sizes evaluated during development. Increasing the pool from one direct retrieval result to 200 candidates provides the main improvement, while increasing the pool from 200 to 300 yields a smaller additional gain.

| Candidate pool size $K$ | Selected PDMS $\uparrow$ |
| --- | --- |
| 1 | 87.6 |
| 200 | 91.1 |
| 300 | 91.3 |

Table 4: Sensitivity to the number of retrieved candidates on NAVSIM v1 navtest. All settings use the same intent predictor and trajectory memory. For $K>1$, the scorer and feasibility gate select the final candidate.

## Appendix E Semantic Occlusion Protocol

The semantic occlusion analysis is conducted on the complete validation split. For every valid scene, the dynamic-agent mask is formed from the projected regions of visible traffic participants and applied consistently to all four input frames. The random control masks an equal total image area. Both interventions preserve the ego-motion history and navigation command, isolating the dependence of the predicted intent on visual information.

Let $\hat{\mathbf{Z}}$ and $\hat{\mathbf{Z}}_{m}$ denote the intent representations predicted from the original and masked inputs. We measure the intervention response as

$$
\Delta_{\mathrm{intent}}=1-\cos\left(\hat{\mathbf{Z}},\hat{\mathbf{Z}}_{m}\right),
$$

where the eight temporal tokens are flattened before cosine similarity is computed. The analysis contains 15,364 valid scenes. Dynamic-agent masking produces a mean intent change of 0.080, compared with 0.027 for equal-area random masking, corresponding to a ratio of $2.97\times$. The dynamic-agent intervention produces the larger response in 71.1% of scenes.

## Appendix F Randomness, Runs, and Computing Infrastructure

### Randomness Control

We use a global seed of 42 for trajectory-space pretraining, dataset subsampling, distributed sampling, and quantitative semantic occlusion. The trajectory pretraining code applies this seed to Python, NumPy, PyTorch, and all CUDA devices. Distributed samplers receive the same base seed and use the epoch index for deterministic epoch-specific shuffling. Validation uses a fixed split and no shuffled sampler. Equal-area random masks in the semantic occlusion experiment are sampled with a NumPy generator initialized with seed 42. The qualitative figure-generation scripts use seed 2027 and enable deterministic cuDNN behavior and deterministic PyTorch algorithms when supported.

All benchmark numbers in the main paper are obtained from one deterministic full-navtest evaluation of the selected checkpoint; they are not averages over independently retrained models. The semantic occlusion statistics aggregate paired interventions over 15,364 validation samples, with the dynamic-agent and random-mask responses computed from the same checkpoint and unoccluded input. We state the number of runs explicitly because full official NAVSIM evaluation is computationally expensive and the evaluation path does not natively support multi-seed aggregation.

### Computing Infrastructure

Training and NAVSIM evaluation were performed on Linux servers equipped with NVIDIA A100-SXM4 GPUs with 80 GB memory. Trajectory-space pretraining and intent-predictor training support distributed execution and used one or two GPUs depending on the run; scorer adaptation, feasibility-gate training, semantic occlusion, and final benchmark evaluation used one GPU. Training commands use Python 3.12, BF16 arithmetic where stated, and one CPU thread per numerical backend during distributed runs. The released code includes the environment and dependency files needed to reconstruct the software stack. Exact package versions distributed with the release should be used rather than versions inferred from the generic upstream NAVSIM environment file.

Distributed training uses data parallelism without changing the model architecture. The per-GPU batch sizes in Table 5 therefore correspond to effective global batch sizes of 256 or 512 for trajectory pretraining and 8 or 16 for intent-predictor training when one or two GPUs are used, respectively. Scorer and gate optimization remain single-GPU procedures because their candidate features are precomputed. Final benchmark evaluation is also executed on one GPU so that all reported planner configurations use the same inference protocol.

### Complete Final Hyperparameters

Table 5 consolidates the final settings used by the four optimized stages. The scene scorer uses the original single-head scorer attention; an eight-head continuation was evaluated during development but did not improve validation selection and is not part of the reported planner.

Frozen encoders and cached candidate features are excluded from the corresponding optimizers. Checkpoint selection uses only the validation criteria shown in the table: trajectory reconstruction uses validation ADE, the intent predictor uses the completed epoch-10 checkpoint, the scorer uses validation selected score, and the gate uses validation recall and utility. The NAVSIM navtest results are not used to choose epochs, learning rates, or thresholds.

| Setting | Trajectory AE | Intent predictor |
| --- | --- | --- |
| Epochs | up to 40 | 10 total |
| Batch size | 256/GPU | 8/GPU |
| Optimizer | AdamW | AdamW |
| Learning rate | $2\times 10^{-4}$ | $10^{-5}$ |
| Weight decay | 0.05 | 0.05 |
| Dropout | 0.1 | 0.1 |
| Gradient clipping | 1.0 | 1.0 |
| Precision | BF16 | BF16 |
| Selection | lowest val ADE | epoch-10 checkpoint |

| Setting | Scene scorer | DAC gate |
| --- | --- | --- |
| Epochs | 5 + 3 continuation | selected checkpoint |
| Batch size | 32 | 32 |
| Optimizer | AdamW | AdamW |
| Learning rate | $10^{-5}\!\rightarrow\!2\times 10^{-6}$ | $3\times 10^{-4}$ |
| Auxiliary LR | $10^{-4}\!\rightarrow\!2\times 10^{-5}$ | – |
| Weight decay | 0.01 | $10^{-3}$ |
| Dropout | pretrained | 0.1 |
| Gradient clipping | 1.0 | 1.0 |
| Precision | BF16 | BF16 |
| Selection | best val score | val recall/utility |

Table 5: Final optimization settings. “Auxiliary LR” denotes the ego-adapter learning rate used by the scene scorer.

## Appendix G Limitations and Failure Modes

Auto-JEPA predicts the implications of a scene for future ego motion rather than rolling out a complete future environment. The learned intent is therefore an action-oriented predictive representation, not an explicit reconstruction of surrounding-agent states. This design is sufficient for the trajectory-planning interface studied here, but it does not provide the scene-level forecasts required by applications such as interactive simulation or counterfactual environment generation.

The retrieval-based planner is bounded by the coverage of its fixed trajectory memory and by the recall of the intent query. If no feasible maneuver is represented in the retrieved candidate pool, neither the scene scorer nor the feasibility gate can synthesize one. The observed saturation between $K=200$ and $K=300$ indicates that simply enlarging the candidate pool offers diminishing returns under the current memory and predictor. A broader memory, adaptive retrieval, or intent-conditioned trajectory generation and local refinement could extend the reachable motion space while preserving the learned intent representation.

When a feasible candidate is available, selection errors can still arise from scorer calibration or distribution shift. A valid candidate may receive an inaccurately low scene score, while the gate may reject a legal borderline trajectory or retain one that violates the drivable area. The all-filtered fallback guarantees a non-empty output but cannot repair an erroneous ranking. These cases motivate uncertainty-aware scoring, stronger calibration, and targeted hard-negative training.

[^1]: S. Ang, Y. Yang, C. Chen, and Y. Wang CLOVER: closed-loop value estimation and ranking for end-to-end autonomous driving planning. arXiv preprint arXiv:2605.15120. Cited by: [End-to-End Trajectory Planning](#Sx2.SSx3.p1.1 "End-to-End Trajectory Planning ‣ Related Work ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving"), [Scene-conditioned utility branch.](#Sx3.SSx6.SSS0.Px1.p1.2 "Scene-conditioned utility branch. ‣ Scene Scoring and Feasibility Gating ‣ Method ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving").

[^2]: M. Assran, A. Bardes, D. Fan, Q. Garrido, R. Howes, M. Muckley, et al. V-JEPA 2: self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985. Cited by: [Joint-Embedding Predictive Learning](#Sx2.SSx2.p1.1 "Joint-Embedding Predictive Learning ‣ Related Work ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving"), [Overview](#Sx3.SSx1.p1.1 "Overview ‣ Method ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving").

[^3]: M. Assran, Q. Duval, I. Misra, P. Bojanowski, P. Vincent, M. Rabbat, Y. LeCun, and N. Ballas Self-supervised learning from images with a joint-embedding predictive architecture. In Proceedings of the IEEE/CVF International Conference on Computer Vision, Cited by: [Introduction](#Sx1.p3.1 "Introduction ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving"), [Joint-Embedding Predictive Learning](#Sx2.SSx2.p1.1 "Joint-Embedding Predictive Learning ‣ Related Work ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving").

[^4]: A. Bardes, Q. Garrido, J. Ponce, X. Chen, M. Rabbat, Y. LeCun, M. Assran, and N. Ballas Revisiting feature prediction for learning visual representations from video. arXiv preprint arXiv:2404.08471. Cited by: [Introduction](#Sx1.p3.1 "Introduction ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving"), [Joint-Embedding Predictive Learning](#Sx2.SSx2.p1.1 "Joint-Embedding Predictive Learning ‣ Related Work ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving").

[^5]: Y. Biktairov, M. Stebelev, I. Rudenko, O. Shliazhko, and B. Yangel PRANK: motion prediction based on ranking. In Advances in Neural Information Processing Systems, Cited by: [End-to-End Trajectory Planning](#Sx2.SSx3.p1.1 "End-to-End Trajectory Planning ‣ Related Work ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving").

[^6]: C. Chen, Y. Yang, Z. Tan, Y. Wang, R. Zhan, H. Liu, X. Mao, J. Bao, X. Tang, L. Yang, B. Sun, Y. Wang, and B. Zhang Devil is in narrow policy: unleashing exploration in driving VLA models. arXiv preprint arXiv:2603.06049. Note: Accepted to CVPR 2026 Findings Cited by: Table 2, Table 2.

[^7]: Y. Chen, Y. Wang, and Z. Zhang DrivingGPT: unifying driving world modeling and planning with multi-modal autoregressive transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, Cited by: Table 2.

[^8]: K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger TransFuser: imitation with transformer-based sensor fusion for autonomous driving. IEEE Transactions on Pattern Analysis and Machine Intelligence 45 (11), pp. 12878–12895. Cited by: Table 2, Table 2.

[^9]: D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, A. Geiger, and K. Chitta NAVSIM: data-driven non-reactive autonomous vehicle simulation and benchmarking. In Advances in Neural Information Processing Systems, Cited by: [Introduction](#Sx1.p5.1 "Introduction ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving"), [Scene-conditioned utility branch.](#Sx3.SSx6.SSS0.Px1.p1.2 "Scene-conditioned utility branch. ‣ Scene Scoring and Feasibility Gating ‣ Method ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving"), [NAVSIM v1.](#Sx4.SSx1.SSS0.Px1.p1.1 "NAVSIM v1. ‣ Dataset and Metrics ‣ Experiments ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving").

[^10]: S. Gao, J. Yang, L. Chen, K. Chitta, Y. Qiu, A. Geiger, J. Zhang, and H. Li Vista: a generalizable driving world model with high fidelity and versatile controllability. In Advances in Neural Information Processing Systems, Cited by: [World Models for Autonomous Driving](#Sx2.SSx1.p1.1 "World Models for Autonomous Driving ‣ Related Work ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving").

[^11]: K. Guo, H. Liu, X. Wu, J. Pan, and C. Lv IPad: iterative proposal-centric end-to-end autonomous driving. arXiv preprint arXiv:2505.15111. Cited by: [End-to-End Trajectory Planning](#Sx2.SSx3.p1.1 "End-to-End Trajectory Planning ‣ Related Work ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving").

[^12]: A. Hu, L. Russell, H. Yeo, Z. Murez, G. Fedoseev, A. Kendall, J. Shotton, and G. Corrado GAIA-1: a generative world model for autonomous driving. arXiv preprint arXiv:2309.17080. Cited by: [World Models for Autonomous Driving](#Sx2.SSx1.p1.1 "World Models for Autonomous Driving ‣ Related Work ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving").

[^13]: F. Jia, L. Liu, Z. Song, C. Jia, H. Ye, X. Hao, and L. Chen DriveWorld-vla: unified latent-space world modeling with vision-language-action for autonomous driving. arXiv preprint arXiv:2602.06521. Cited by: [Introduction](#Sx1.p1.1 "Introduction ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving"), [World Models for Autonomous Driving](#Sx2.SSx1.p2.1 "World Models for Autonomous Driving ‣ Related Work ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving"), Table 2.

[^14]: B. Jiang, S. Chen, H. Gao, B. Liao, Q. Zhang, W. Liu, and X. Wang VADv2: end-to-end vectorized autonomous driving via probabilistic planning. arXiv preprint arXiv:2402.13243. Cited by: [End-to-End Trajectory Planning](#Sx2.SSx3.p1.1 "End-to-End Trajectory Planning ‣ Related Work ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving"), Table 2.

[^15]: K. Li, Z. Li, S. Lan, Y. Xie, Z. Zhang, J. Liu, Z. Wu, Z. Yu, and J. M. Alvarez Hydra-MDP++: advancing end-to-end driving via expert-guided hydra-distillation. arXiv preprint arXiv:2503.12820. Cited by: Table 2.

[^16]: Y. Li, K. Xiong, X. Guo, F. Li, S. Yan, G. Xu, L. Zhou, L. Chen, H. Sun, B. Wang, et al. RecogDrive: a reinforced cognitive framework for end-to-end autonomous driving. In International Conference on Learning Representations, Cited by: Table 2, Table 2.

[^17]: Y. Li, L. Fan, J. He, Y. Wang, Y. Chen, Z. Zhang, and T. Tan Enhancing end-to-end autonomous driving with latent world model. In International Conference on Learning Representations, Cited by: [Introduction](#Sx1.p2.1 "Introduction ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving"), [World Models for Autonomous Driving](#Sx2.SSx1.p2.1 "World Models for Autonomous Driving ‣ Related Work ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving"), Table 2.

[^18]: Y. Li, S. Shang, W. Liu, B. Zhan, H. Wang, Y. Wang, Y. Chen, X. Wang, Y. An, C. Tang, L. Hou, L. Fan, and Z. Zhang DriveVLA-W0: world models amplify data scaling law in autonomous driving. In International Conference on Learning Representations, Cited by: [End-to-End Trajectory Planning](#Sx2.SSx3.p2.1 "End-to-End Trajectory Planning ‣ Related Work ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving"), Table 2, Table 2.

[^19]: Y. Li, Y. Wang, Y. Liu, J. He, L. Fan, and Z. Zhang End-to-end driving with online trajectory evaluation via BEV world model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, Cited by: Table 2.

[^20]: Z. Li, K. Li, S. Wang, S. Lan, Z. Yu, Y. Ji, Z. Li, Z. Zhu, J. Kautz, et al. Hydra-MDP: end-to-end multimodal planning with multi-target hydra-distillation. arXiv preprint arXiv:2406.06978. Cited by: [End-to-End Trajectory Planning](#Sx2.SSx3.p1.1 "End-to-End Trajectory Planning ‣ Related Work ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving"), Table 2.

[^21]: B. Liao et al. DiffusionDrive: truncated diffusion model for end-to-end autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12037–12047. Cited by: [End-to-End Trajectory Planning](#Sx2.SSx3.p1.1 "End-to-End Trajectory Planning ‣ Related Work ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving"), Table 2, Table 2.

[^22]: Y. Luo, F. Li, S. Xu, Z. Lai, L. Yang, Q. Chen, Z. Luo, Z. Xie, S. Jiang, J. Liu, et al. AdaThinkDrive: adaptive thinking via reinforcement learning for autonomous driving. arXiv preprint arXiv:2509.13769. Cited by: Table 2.

[^23]: A. Seff, B. Cera, D. Chen, M. Ng, A. Zhou, N. Nayakanti, K. S. Refaat, R. Al-Rfou, and B. Sapp MotionLM: multi-agent motion forecasting as language modeling. In Proceedings of the IEEE/CVF International Conference on Computer Vision, Cited by: [Introduction](#Sx1.p1.1 "Introduction ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving").

[^24]: W. Sun, X. Lin, K. Chen, Z. Pei, X. Li, Y. Shi, and S. Zheng SparseDriveV2: scoring is all you need for end-to-end autonomous driving. arXiv preprint arXiv:2603.29163. Cited by: [End-to-End Trajectory Planning](#Sx2.SSx3.p1.1 "End-to-End Trajectory Planning ‣ Related Work ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving").

[^25]: A. van den Oord, Y. Li, and O. Vinyals Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748. Cited by: [Joint-Embedding Training Objectives](#Sx3.SSx4.p1.1 "Joint-Embedding Training Objectives ‣ Method ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving").

[^26]: A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin Attention is all you need. In Advances in Neural Information Processing Systems, Cited by: [Visual Intent Prediction](#Sx3.SSx3.p2.1 "Visual Intent Prediction ‣ Method ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving").

[^27]: L. Wang, Z. Yang, C. Bai, G. Zhang, X. Liu, X. Zheng, X. Long, C. Lu, and C. Lu Drive-jepa: video jepa meets multimodal trajectory distillation for end-to-end driving. arXiv preprint arXiv:2601.22032. Cited by: [Introduction](#Sx1.p2.1 "Introduction ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving"), [Joint-Embedding Predictive Learning](#Sx2.SSx2.p1.1 "Joint-Embedding Predictive Learning ‣ Related Work ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving").

[^28]: X. Wang, Z. Zhu, G. Huang, X. Chen, J. Zhu, and J. Lu DriveDreamer: towards real-world-driven world models for autonomous driving. arXiv preprint arXiv:2309.09777. Cited by: [World Models for Autonomous Driving](#Sx2.SSx1.p1.1 "World Models for Autonomous Driving ‣ Related Work ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving").

[^29]: Y. Wang, J. He, L. Fan, H. Li, Y. Chen, and Z. Zhang Driving into the future: multiview visual forecasting and planning with world model for autonomous driving. arXiv preprint arXiv:2311.17918. Cited by: [Introduction](#Sx1.p1.1 "Introduction ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving"), [World Models for Autonomous Driving](#Sx2.SSx1.p1.1 "World Models for Autonomous Driving ‣ Related Work ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving").

[^30]: X. Weng, B. Ivanovic, Y. Wang, Y. Wang, and M. Pavone PARA-Drive: parallelized architecture for real-time autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, Cited by: Table 2.

[^31]: C. Xie, B. Sun, T. Li, J. Wu, Z. Hao, X. Lang, and H. Li LatentVLA: efficient vision-language models for autonomous driving via latent action prediction. arXiv preprint arXiv:2601.05611. Cited by: [Introduction](#Sx1.p2.1 "Introduction ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving").

[^32]: Z. Xing, X. Zhang, Y. Hu, B. Jiang, T. He, Q. Zhang, X. Long, and W. Yin GoalFlow: goal-driven flow matching for multimodal trajectories generation in end-to-end autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1602–1611. Cited by: [End-to-End Trajectory Planning](#Sx2.SSx3.p1.1 "End-to-End Trajectory Planning ‣ Related Work ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving").

[^33]: Z. Yang, L. Chen, Y. Sun, and H. Li Visual point cloud forecasting enables scalable autonomous driving. arXiv preprint arXiv:2312.17655. Cited by: [World Models for Autonomous Driving](#Sx2.SSx1.p1.1 "World Models for Autonomous Driving ‣ Related Work ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving").

[^34]: W. Yao, Z. Li, S. Lan, Z. Wang, X. Sun, J. M. Alvarez, and Z. Wu DriveSuprim: towards precise trajectory selection for end-to-end planning. arXiv preprint arXiv:2506.06659. Note: Accepted to AAAI 2026 Cited by: Table 2.

[^35]: K. Zhang, Z. Tang, X. Hu, X. Pan, X. Guo, Y. Liu, J. Huang, L. Yuan, Q. Zhang, X. Long, et al. Epona: autoregressive diffusion world model for autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, Cited by: Table 2.

[^36]: L. Zhang, C. Wu, L. Shi, J. Li, J. Liu, L. Yang, H. Zhang, M. Xu, and H. Wang DeepSight: long-horizon world modeling via latent states prediction for end-to-end autonomous driving. arXiv preprint arXiv:2605.10564. Cited by: [Introduction](#Sx1.p1.1 "Introduction ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving"), [World Models for Autonomous Driving](#Sx2.SSx1.p2.1 "World Models for Autonomous Driving ‣ Related Work ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving").

[^37]: W. Zheng, W. Chen, Y. Huang, B. Zhang, Y. Duan, and J. Lu OccWorld: learning a 3d occupancy world model for autonomous driving. arXiv preprint arXiv:2311.16038. Cited by: [Introduction](#Sx1.p1.1 "Introduction ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving"), [World Models for Autonomous Driving](#Sx2.SSx1.p1.1 "World Models for Autonomous Driving ‣ Related Work ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving").

[^38]: Y. Zheng, P. Yang, Z. Xing, Q. Zhang, Y. Zheng, Y. Gao, P. Li, T. Zhang, Z. Xia, P. Jia, and D. Zhao World4Drive: end-to-end autonomous driving via intention-aware physical latent world model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, Cited by: [Introduction](#Sx1.p2.1 "Introduction ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving"), [World Models for Autonomous Driving](#Sx2.SSx1.p2.1 "World Models for Autonomous Driving ‣ Related Work ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving").

[^39]: X. Zhou, X. Han, F. Yang, Y. Ma, V. Tresp, and A. Knoll OpenDriveVLA: towards end-to-end autonomous driving with large vision-language-action model. arXiv preprint arXiv:2503.23463. Cited by: [End-to-End Trajectory Planning](#Sx2.SSx3.p2.1 "End-to-End Trajectory Planning ‣ Related Work ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving").

[^40]: Z. Zhou, T. Cai, S. Z. Zhao, Y. Zhang, Z. Huang, B. Zhou, and J. Ma AutoVLA: a vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. In Advances in Neural Information Processing Systems, Cited by: [End-to-End Trajectory Planning](#Sx2.SSx3.p2.1 "End-to-End Trajectory Planning ‣ Related Work ‣ Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving"), Table 2.
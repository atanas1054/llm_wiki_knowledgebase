---
title: "GeoWAM: Visual Geometry World Action Models for Autonomous Driving"
source: "https://arxiv.org/html/2608.23486v2"
author:
published: 2026-08-25
created: 2026-09-02
description:
tags:
  - "clippings"
---
1\]Uber AV Labs 2\]Case Western Reserve University \[†\]Corresponding authors\[‡\]Project Lead [https://yiren-lu.com/project\_pages/geowam/](https://yiren-lu.com/project_pages/geowam/)

Yiren Lu <sup>1,2</sup>    Xin Ye <sup>1,†</sup>    Jiaming Liu <sup>1</sup>    Philip Jacobson <sup>1</sup>    Jin Yao <sup>1</sup>    Yi-chung Chen <sup>1</sup>    Liam Merino <sup>1</sup>    Dhruva Dixith Kurra <sup>1</sup>    Min Cai <sup>1</sup>    Tom Lampo <sup>1</sup>    Yu Yin <sup>2,†</sup>    Danhua Guo <sup>1</sup>    Burhan Yaman <sup>1‡</sup> Affiliation: \[ Affiliation: \[

###### Abstract

World action models (WAMs) have recently gained increasing attention as a framework for jointly modeling scene evolution and ego actions in autonomous driving. Most existing WAMs learn scene dynamics in pixel space by combining a video-generation backbone for future-observation prediction with an action head for ego-trajectory prediction. Pixels, however, provide only an indirect representation of these dynamics: they entangle geometry and motion with appearance, texture, and illumination, forcing the model to infer three-dimensional transformations from two-dimensional observations. We argue that geometry, represented by point clouds, offers a more natural state space for driving because it explicitly captures spatial structure and the rigid and non-rigid transformations that govern scene evolution while directly aligning with the space in which driving actions are executed. Building on this insight, we introduce GeoWAM, a visual geometry world action model for autonomous driving. Rather than predicting future images, GeoWAM is pretrained to forecast future scene geometry, yielding representations that jointly encode spatial structure and temporal evolution. A geometry-conditioned action head then leverages these learned geometric dynamics to predict future ego trajectories. Extensive open-loop and closed-loop evaluations show that visual geometry world modeling yields substantially stronger driving policies than image-based alternatives, establishing future-geometry prediction as an effective pretraining objective for autonomous driving.

## 1 Introduction

![[GeoWAM_teaser.png|Refer to caption]]

Figure 1: Video and geometry world models represent scene dynamics differently. Given the same current observation, a video world model predicts how pixel values evolve over time. The underlying 3D transformations remain implicit in these pixel changes and are therefore difficult to recover. In contrast, a geometry world model predicts future 3D structure, whose evolution explicitly exposes the underlying spatial transformations and provides a representation naturally aligned with motion planning.

Autonomous driving is fundamentally a problem of anticipation. A capable driving agent must understand not only the current scene, but also how the scene is likely to evolve and how that evolution constrains its actions. Recent end-to-end driving systems increasingly build on pretrained foundation models to acquire the broad representations needed for this task.

A prominent line of work develops Vision-Language-Action (VLA) models, which transfer the semantic knowledge and reasoning capabilities of pretrained vision-language models to ego-trajectory prediction, either through language tokens or a dedicated action decoder [^14] [^47] [^29] [^16] [^34] [^22] [^41]. VLA policies are well suited to high-level scene understanding and decision making. However, their action-centric training objectives typically do not explicitly supervise future scene evolution, leaving environmental dynamics implicit in the learned policy representation.

World models offer a complementary foundation by learning to predict how an environment evolves. In autonomous driving, recent approaches use high-capacity video-generation backbones to forecast future observations with impressive visual fidelity and controllability [^13] [^9] [^35] [^33] [^12]. Pretraining on large-scale, unlabeled driving logs allows these models to acquire transferable priors over object motion, scene structure, and temporal evolution. Future-observation prediction alone, however, does not specify which action the ego vehicle should execute.

World action models (WAMs) connect prediction with planning by coupling future-state forecasting and trajectory generation, enabling a shared representation to model how the world evolves and how the ego vehicle should act [^4] [^20] [^46] [^1] [^43] [^38]. Most existing WAMs inherit the representation design of visual world models and use images as their observation space, learning scene dynamics through future-frame prediction while coupling the learned representation to trajectory generation.

In this paper, we argue that pixels are not an ideal state representation for modeling driving dynamics. Although visually rich, images encode scene geometry and motion only *indirectly*, entangling them with appearance, texture, and illumination. Accordingly, video world models are optimized to model the distribution of future visual observations, but this objective does not require them to explicitly recover the underlying physical dynamics that generate those observations. A model may therefore produce visually plausible futures by capturing visual spatiotemporal regularities. As illustrated in figure 1, the underlying three-dimensional transformations can remain implicit and difficult to recover from the generated observations. Such predictions are therefore not necessarily grounded in a state representation well aligned with the requirements of driving.

Geometry, by contrast, provides a *native* state space for driving. Representations such as point clouds explicitly encode the three-dimensional structure of the environment. Across time, changes in geometry directly reveal object motion and transformations of the ego reference frame without requiring photometric reconstruction. Moreover, scene geometry and ego trajectories are defined in the same three-dimensional coordinate space. Forecasting future geometry therefore provides direct supervision for the spatial structure and scene dynamics that underpin safe planning and control.

Building on this insight, we introduce GeoWAM, a visual geometry world action model for autonomous driving. Instead of forecasting future images, GeoWAM is pretrained to predict future scene geometry, learning representations that jointly capture spatial structure and temporal evolution. A geometry-conditioned action head then leverages these learned geometric dynamics to predict future ego trajectories. By placing both world modeling and action prediction in a shared geometric space, GeoWAM provides the driving policy with representations explicitly organized around 3D structure and motion.

We evaluate GeoWAM across multiple settings, including future-geometry prediction and open- and closed-loop planning on NAVSIM [^7] [^3]. Together, these evaluations assess the accuracy of its predicted scene structure and the utility of its learned geometric dynamics for motion planning.

Our contributions are:

- We motivate geometry as a native state representation for world action models, directly aligning scene dynamics with the three-dimensional space in which driving actions are defined.
- We pretrain a visual geometry world model to forecast future scene geometry from historical multiview observations, enabling it to capture the spatial structure and temporal dynamics of driving scenes.
- We introduce GeoWAM, which extends the pretrained geometry world model into a world action model through an inverse-dynamics-like formulation that infers future ego motion from predicted geometry and maps it to an ego trajectory.
- We validate GeoWAM through future-geometry prediction and open- and closed-loop planning, demonstrating the effectiveness of visual geometry world modeling for autonomous driving.

## 2 Related Work

### 2.1 World Models for Autonomous Driving

World models learn predictive representations of environment dynamics from past observations and, when available, actions. In autonomous driving, recent world models predominantly formulate this objective as future-video generation. GAIA-1 [^13], DriveDreamer [^33], Vista [^9], and GEM [^12] synthesize future camera observations with autoregressive or diffusion-based architectures while supporting different forms of action and scene conditioning. By learning from driving videos, these models acquire priors over ego motion, agent behavior, and visual scene evolution, enabling realistic and controllable future rollouts. Together, these works establish future-video prediction as a powerful paradigm for learning and simulating driving-scene dynamics.

Beyond video generation, occupancy-based world models forecast scene evolution in a voxelized three-dimensional space. OccWorld [^45] tokenizes 3D occupancy and autoregressively predicts future occupancy together with ego motion, while Drive-OccWorld [^40] extends occupancy forecasting with action conditioning and occupancy-based planning. These approaches are supervised with voxelized ground-truth occupancy targets, requiring occupancy annotations to construct their prediction space. In contrast, GeoWAM does not require ground-truth occupancy annotations and learns future geometry from dense metric point-map targets derived from off-the-shelf geometry foundation models, requiring only RGB images for training.

### 2.2 World Action Models for Autonomous Driving

World action models extend predictive world modeling by coupling future-state prediction with action generation, allowing learned environmental dynamics to directly inform a policy. Driving into the Future [^35] transfers representations learned through multiview visual forecasting to a downstream planner. VaViM and VaVAM [^1] augment autoregressive video modeling with an action expert, while Epona [^43] uses separate diffusion-based heads for image and trajectory prediction. WorldVLA [^4] and DriveVLA-W0 [^20] further connect visual or vision-language pretraining with action prediction. PWM [^44] jointly forecasts state and action evolution, whereas DriveLaW [^37] unifies planning and video generation within a shared latent driving world. UniDrive-WM [^38] unifies scene understanding, trajectory planning, and trajectory-conditioned future image generation within a shared vision-language architecture, whereas DriveWAM [^26] adapts a pretrained video diffusion transformer into a unified video-action policy under a joint flow-matching objective. EponaV2 [^39] additionally supervises future depth and semantic features to enrich the representation learned alongside image prediction. Despite their architectural differences, most existing driving WAMs inherit image or video generation as their primary world-modeling interface, leaving three-dimensional scene evolution implicit in visual features. GeoWAM makes metric geometry the primary prediction space and conditions trajectory generation on the resulting future geometric dynamics.

### 2.3 Visual Geometry Models

General visual geometry models have increasingly replaced task-specific reconstruction pipelines with feed-forward prediction of dense geometric quantities. DUSt3R [^32] introduced point-map regression for uncalibrated image pairs, removing the need to explicitly estimate camera parameters before reconstruction. CUT3R [^31] extends this formulation with a persistent state for continuous 3D perception, while VGGT [^30] jointly infers camera parameters, depth, point maps, and tracks from a variable number of views. MapAnything [^17] further supports flexible geometric inputs and directly recovers metric-scale scene geometry in a unified feed-forward model. Together, these methods establish dense point maps as a scalable representation for 3D reconstruction.

Driving scenes introduce additional requirements, including long-range metric accuracy, temporal motion, surround-view observations, and substantial variation across camera configurations. DVGT [^48] addresses these requirements with an ego-centric formulation that predicts metric point maps and ego poses from multi-frame, multi-view images using factorized spatial-temporal attention, while DVGT-2 [^49] scales this geometry representation toward autonomous-driving action modeling. However, existing visual geometry models primarily reconstruct the scene contained in observed images. GeoWAM extends visual geometry learning from reconstruction to forecasting, using future point-map prediction to learn scene dynamics and connect geometric representations with trajectory planning.

## 3 Methodology

We introduce GeoWAM, a visual geometry world action model for autonomous driving, as illustrated in figure 2. Unlike conventional video-based world action models, GeoWAM first learns to forecast the three-dimensional evolution of a driving scene and then uses the predicted geometric dynamics to plan the future motion of the ego vehicle. The training of GeoWAM consists of two stages. In the first stage (section 3.1), we pretrain a visual geometry world model using driving sequences to predict dense future scene geometry from historical multiview images. In the second stage (section 3.2), we extend the pretrained visual geometry world model with a geometry-conditioned action head and jointly finetune the complete model for future-geometry prediction and ego-trajectory planning. The resulting model uses its predicted geometric dynamics to directly inform driving actions.

![[GeoWAM_pipeline.png|Refer to caption]]

Figure 2: Overview of GeoWAM. Our framework takes a sequence of historical multiview frames as input. A geometry encoder converts these observations into a multi-level memory of geometry and ego/pose tokens. The future geometry decoder applies temporal self-attention and cross-attends to the historical memory to predict future geometry tokens, which are decoded by Point DPT into dense future point maps. In the action branch, the predicted geometry tokens condition the future ego/pose decoder through a stop-gradient connection, and the resulting ego/pose tokens are mapped by the trajectory head to the future ego trajectory.

### 3.1 Visual Geometry World Model

##### Multiview geometry encoding.

Let $\mathbf{I}_{t-K+1:t}=\{\mathbf{I}_{\tau}^{v}\mid\tau=t-K+1,\ldots,t;\ v=1,\ldots,V\}$ denote the $K$ -frame multiview image history, where $\mathbf{I}_{\tau}^{v}\in\mathbb{R}^{3\times H\times W}$ is the image from camera $v$ at time $\tau$. We adopt DVGT-2 [^49] as our geometry encoder $\mathcal{E}_{\theta}$ to encode the multiview image sequence into multi-level historical tokens. We retain the outputs from $L$ selected feature levels, indexed by $\ell=1,\ldots,L$. At each time step $\tau$ and feature level $\ell$, the encoder produces two types of tokens. The geometry tokens $\mathbf{X}_{\tau}^{\ell}\in\mathbb{R}^{V\times P\times D}$ represent the spatial scene structure observed from each camera view, while the ego tokens $\mathbf{E}_{\tau}^{\ell}\in\mathbb{R}^{V\times N_{e}\times D}$ encode ego-motion context across the observation history. Here, $P=hw$ is the number of geometry tokens per view, $N_{e}$ is the number of ego tokens per view, and $D$ is their feature dimension. We concatenate them along the token dimension as $\mathbf{Z}_{\tau}^{\ell}=[\mathbf{X}_{\tau}^{\ell};\mathbf{E}_{\tau}^{\ell}]\in\mathbb{R}^{V\times(P+N_{e})\times D}$. The complete historical geometry memory is defined as follows:

$$
\mathcal{Z}_{t}=\left\{\mathbf{Z}_{\tau}^{\ell}\mid\tau=t-K+1,\ldots,t;\ \ell=1,\ldots,L\right\}=\mathcal{E}_{\theta}(\mathbf{I}_{t-K+1:t}).
$$

##### Future geometry decoding.

As shown in figure 2, the future-geometry branch predicts the scene over the next $F$ steps from a set of learned geometry queries. Let $\mathbf{q}^{\mathrm{geom}}\in\mathbb{R}^{d}$ be a learned query seed, where $d$ is the hidden dimension of the future geometry decoder. We replicate this seed for every future step, camera view, and spatial location. The geometry query at future step $k$, view $v$, and spatial location $p$ is

$$
\mathbf{Q}_{t+k}^{\mathrm{geom},v,p}=\mathbf{q}^{\mathrm{geom}}+\mathbf{e}_{K+k}^{\mathrm{time}}+\mathbf{e}_{v}^{\mathrm{view}}+\mathbf{e}_{p}^{\mathrm{2D}},
$$

where $\mathbf{e}_{K+k}^{\mathrm{time}}$ and $\mathbf{e}_{v}^{\mathrm{view}}$ are learned temporal and view embeddings, and $\mathbf{e}_{p}^{\mathrm{2D}}$ is a two-dimensional sinusoidal positional embedding. We use $\mathcal{Q}_{t+1:t+F}^{\mathrm{geom}}=\{\mathbf{Q}_{t+k}^{\mathrm{geom},v,p}\}_{k,v,p}$ to denote all future geometry queries.

The decoder first projects the historical memory $\mathcal{Z}_{t}$ to its hidden dimension. It then updates the geometry queries in two steps at each decoder layer. First, causal temporal self-attention models the evolution of each spatial location across the $F$ future steps. Second, the updated queries cross-attend to $\mathcal{Z}_{t}$ to retrieve the relevant context from the observed scene. After stacking multiple decoder layers, we obtain a future geometry latent $\hat{\mathbf{U}}_{t+k}\in\mathbb{R}^{V\times P\times d}$ for each future step. For each feature level $\ell$, an output projection $\mathbf{W}_{\ell}\in\mathbb{R}^{D\times d}$ maps this latent to the feature space required by the geometry head:

$$
\begin{gathered}\hat{\mathcal{U}}_{t+1:t+F}=\mathcal{D}_{\phi}(\mathcal{Q}_{t+1:t+F}^{\mathrm{geom}},\mathcal{Z}_{t}),\\
\hat{\mathbf{X}}_{t+k}^{\ell}=\hat{\mathbf{U}}_{t+k}\mathbf{W}_{\ell}^{\mathsf{T}},\quad k=1,\ldots,F,\ \ell=1,\ldots,L.\end{gathered}
$$

where $\hat{\mathcal{U}}_{t+1:t+F}=\{\hat{\mathbf{U}}_{t+k}\}_{k=1}^{F}$. The level-specific outputs $\hat{\mathbf{X}}_{t+k}^{\ell}\in\mathbb{R}^{V\times P\times D}$ form a multi-level feature representation of the future scene. The shared geometry head $\mathcal{G}_{\psi}$ decodes this representation into a dense point map and a per-pixel confidence map:

$$
\left(\hat{\mathbf{P}}_{t+k},\hat{\mathbf{C}}_{t+k}\right)=\mathcal{G}_{\psi}\left(\left\{\hat{\mathbf{X}}_{t+k}^{\ell}\right\}_{\ell=1}^{L}\right).
$$

Here, $\hat{\mathbf{P}}_{t+k}\in\mathbb{R}^{V\times H\times W\times 3}$ stores one 3D point per image pixel in the ego coordinate system at time $t+k$, and $\hat{\mathbf{C}}_{t+k}\in\mathbb{R}^{V\times H\times W}$ contains the corresponding confidence values. The model therefore predicts geometric scene evolution without reconstructing future image appearance.

##### Future geometry supervision.

During training, the future images are processed by the same geometry encoder in a target branch to obtain patch-feature targets $\bar{\mathbf{X}}_{t+k}^{\ell}\in\mathbb{R}^{V\times P\times D}$. The target features are detached from the computation graph, and the future images are never provided to the forecasting branch or used at inference time. We align each predicted feature with its target using cosine distance:

$$
\mathcal{L}_{\mathrm{feat}}=\frac{1}{FL}\sum_{k=1}^{F}\sum_{\ell=1}^{L}\left(1-\operatorname{cos}\left(\hat{\mathbf{X}}_{t+k}^{\ell},\operatorname{sg}\!\left(\bar{\mathbf{X}}_{t+k}^{\ell}\right)\right)\right),
$$

where $\operatorname{sg}(\cdot)$ denotes stop-gradient and $\operatorname{cos}(\cdot,\cdot)$ averages cosine similarity over cameras and patch locations. The predicted point maps are additionally supervised by dense future point-map targets $\mathbf{P}_{t+1:t+F}$ and their validity masks. The point-map objective, averaged over future steps, views, and valid pixels, combines Euclidean point regression, confidence-aware regression, and multi-scale surface-normal consistency:

$$
\mathcal{L}_{\mathrm{point}}^{\mathrm{future}}=\mathcal{L}_{\mathrm{reg}}+\mathcal{L}_{\mathrm{conf}}+\mathcal{L}_{\mathrm{normal}}.
$$

We apply the same point-map objective to the encoded current frame, producing $\mathcal{L}_{\mathrm{point}}^{\mathrm{current}}$ and anchoring the geometry encoder while it learns to forecast. The geometry pretraining objective is

$$
\mathcal{L}_{\mathrm{pre}}=\mathcal{L}_{\mathrm{feat}}+\mathcal{L}_{\mathrm{point}}^{\mathrm{future}}+\mathcal{L}_{\mathrm{point}}^{\mathrm{current}}.
$$

### 3.2 GeoWAM: Visual Geometry World Action Model

After geometry pretraining, the model is able to predict how the scene geometry will evolve from historical observations. We extend this capability to trajectory planning through the action branch illustrated in figure 2. GeoWAM follows an inverse-dynamics-like formulation: it first infers future ego motion from the predicted scene evolution and then maps the resulting motion representation to an ego trajectory.

##### Future ego-token decoding.

We introduce $N_{e}$ learned ego-query seeds $\{\mathbf{q}_{n}^{\mathrm{ego}}\}_{n=1}^{N_{e}}$, with one seed for each ego-token slot. For every future step $k$ and camera view $v$, the corresponding query is constructed as

$$
\mathbf{Q}_{t+k}^{\mathrm{ego},v,n}=\mathbf{q}_{n}^{\mathrm{ego}}+\mathbf{e}_{K+k}^{\mathrm{time}}+\mathbf{e}_{v}^{\mathrm{view}},\quad n=1,\ldots,N_{e}.
$$

At each ego-decoder layer, the queries first undergo causal temporal self-attention across the $F$ future steps, independently for each view and ego-token slot. They then cross-attend to both the historical geometry memory $\mathcal{Z}_{t}$ and the predicted future geometry tokens $\hat{\mathcal{U}}_{t+1:t+F}$. The decoder thereby produces future ego tokens that describe ego motion consistent with the forecast scene evolution:

$$
\hat{\mathbf{E}}_{t+1:t+F}=\mathcal{D}_{\eta}^{\mathrm{ego}}\left(\mathbf{Q}^{\mathrm{ego}},\mathcal{Z}_{t},\operatorname{sg}\!\left(\hat{\mathcal{U}}_{t+1:t+F}\right)\right).
$$

Here, $\hat{\mathbf{E}}_{t+k}\in\mathbb{R}^{V\times N_{e}\times D}$ denotes the predicted ego tokens at future step $t+k$. The stop-gradient operation prevents the trajectory loss from propagating through the predicted future geometry, helping preserve the geometry forecasting capability acquired during pretraining. This one-way connection also reflects our inverse-dynamics-like design: ego motion is inferred from scene evolution, whereas trajectory supervision does not reshape the geometry used as its conditioning signal.

##### Trajectory decoding.

The action head takes the deepest-level historical ego tokens $\mathbf{E}_{t-K+1:t}^{L}$ together with the predicted future ego tokens $\hat{\mathbf{E}}_{t+1:t+F}$. It appends the two sequences along the temporal dimension and refines them with a causal temporal transformer, allowing each future ego token to incorporate the preceding historical and predicted motion context. A learned trajectory query then cross-attends to the refined historical and future ego features, and a regression head maps the resulting query to a single future trajectory. We denote the predicted trajectory by $\hat{\mathbf{A}}_{t}=[\hat{\mathbf{a}}_{t+1},\ldots,\hat{\mathbf{a}}_{t+F}]$, where $\hat{\mathbf{a}}_{t+k}=(\hat{x}_{t+k},\hat{y}_{t+k},\hat{\theta}_{t+k})$ specifies the planar position and heading of the ego vehicle in its current coordinate frame:

$$
\hat{\mathbf{A}}_{t}=\mathcal{H}_{\omega}\left(\mathbf{E}_{t-K+1:t},\hat{\mathbf{E}}_{t+1:t+F}\right).
$$

The action head directly regresses one trajectory without trajectory anchors, mode classification, or iterative sampling.

##### Planning objective.

During planning finetuning, we retain the future and current geometry objectives and supervise the trajectory with an $\ell_{1}$ regression loss $\mathcal{L}_{\mathrm{traj}}$. We additionally predict the relative poses between historical frames using an auxiliary $\ell_{1}$ loss $\mathcal{L}_{\mathrm{pose}}$. The complete finetuning objective is

$$
\mathcal{L}_{\mathrm{plan}}=\mathcal{L}_{\mathrm{pre}}+\lambda_{\mathrm{traj}}\mathcal{L}_{\mathrm{traj}}+\lambda_{\mathrm{pose}}\mathcal{L}_{\mathrm{pose}}.
$$

## 4 Experiments

We first introduce the datasets, evaluation metrics, and implementation details in section 4.1. We then evaluate the two capabilities at the core of GeoWAM. In section 4.2, we measure the accuracy of the predicted metric scene geometry over the full forecasting horizon on the nuScenes validation set. In section 4.3, we assess how effectively the learned geometric dynamics support ego-trajectory planning on NAVSIM [^7], including both the navtest split and the closed-loop two-stage navhard benchmark [^3].

### 4.1 Experimental Setup

##### Datasets.

For future-geometry pretraining, we combine OpenScene [^6], nuScenes [^2], Bench2Drive [^15], Waymo Open Dataset [^27], KITTI [^10], Argoverse 2 [^36], and DDAD [^11]. We evaluate future-geometry prediction on the nuScenes validation set. For planning, we finetune GeoWAM on the NAVSIM navtrain split and report results on the NAVSIM v2 navtest and navhard splits [^7] [^3]. The latter contains challenging scenarios and uses a two-stage pseudo-closed-loop evaluation: the first stage evaluates the original scenes, while the second evaluates synthetic reactive scenes.

##### Metrics.

We convert each predicted future point map to ray depth, defined as the distance between a predicted 3D point and the ego-vehicle origin. Following standard geometry evaluation, we report absolute relative error and threshold accuracy $\delta<1.25$. Both metrics are computed for each of the eight future steps and over the complete prediction horizon. For planning, we use the Extended Predictive Driver Model Score (EPDMS) from NAVSIM v2. EPDMS aggregates no at-fault collision (NC), drivable-area compliance (DAC), driving-direction compliance (DDC), traffic-light compliance (TLC), ego progress (EP), time-to-collision (TTC), lane keeping (LK), history comfort (HC), and extended comfort (EC). All planning metrics are reported with the official human-penalty protocol, and higher values indicate better performance.

##### Implementation details.

The future geometry decoder contains six transformer layers with a hidden dimension of 1024 and 16 attention heads. Geometry pretraining uses three historical frames to predict $F=8$ future frames at $2$ Hz, with two to eight camera views dynamically sampled from each sequence. We initialize the geometry encoder and point head from DVGT-2 [^49] and optimize the model for 161 epochs using AdamW with a weight decay of $0.05$. The future decoder uses a peak learning rate of $10^{-4}$, while the pretrained components use $2\times 10^{-5}$. Both learning rates use a 5% linear warmup followed by cosine decay, and training is performed in bfloat16 precision. For planning, we initialize from the geometry-pretrained checkpoint and finetune on navtrain for 40 epochs using eight camera views and three historical frames. The future decoder and newly introduced action head use a learning rate of $10^{-4}$, while the remaining pretrained parameters use $2\times 10^{-5}$. We set both loss weights to $\lambda_{\mathrm{traj}}=\lambda_{\mathrm{pose}}=5$.

### 4.2 Future Geometry Prediction

Table 1 presents quantitative comparisons of future geometry prediction on the nuScenes validation set over horizons from one to four seconds. We compare GeoWAM with two categories of baselines. The first is VGGT-World [^28], a recently introduced geometry world model that directly predicts future geometry in the feature space of a geometry foundation model. The second category consists of video world models, including Epona [^43] and Cosmos 3 [^25]. Since these models generate future RGB observations rather than geometry, we first use them to predict future frames and then apply DVGT [^48] to reconstruct the corresponding 3D scenes. This protocol provides all methods with a common geometric output for evaluation.

Table 1: Future geometry prediction performance on nuScenes at different horizons. Lower Abs Rel is better, while higher $\delta<1.25$ is better. The mean is computed over all eight predicted frames. Bold and underlined values indicate the best and second-best results, respectively.

<table><thead><tr><th></th><th colspan="5">Abs Rel <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th><th colspan="5"><math><semantics><mrow><mi>δ</mi> <mo><</mo> <mn>1.25</mn></mrow> <annotation>\delta<1.25</annotation></semantics></math> <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th></tr><tr><th>Method/Horizon</th><th>1s</th><th>2s</th><th>3s</th><th>4s</th><th>mean</th><th>1s</th><th>2s</th><th>3s</th><th>4s</th><th>mean</th></tr></thead><tbody><tr><th>Epona <sup><a href="#fn:43">43</a></sup> + DVGT <sup><a href="#fn:48">48</a></sup></th><td>0.229</td><td>0.263</td><td>0.292</td><td>0.310</td><td>0.274</td><td>0.732</td><td>0.677</td><td>0.620</td><td>0.589</td><td>0.655</td></tr><tr><th>Cosmos 3 <sup><a href="#fn:25">25</a></sup> + DVGT <sup><a href="#fn:48">48</a></sup></th><td>0.300</td><td>0.376</td><td>0.405</td><td>0.422</td><td>0.376</td><td>0.588</td><td>0.513</td><td>0.464</td><td>0.447</td><td>0.503</td></tr><tr><th>VGGT-World <sup><a href="#fn:28">28</a></sup></th><td>0.272</td><td>0.329</td><td>0.342</td><td>0.357</td><td>0.325</td><td>0.612</td><td>0.553</td><td>0.513</td><td>0.497</td><td>0.544</td></tr><tr><th>GeoWAM (ours)</th><td>0.228</td><td>0.245</td><td>0.256</td><td>0.297</td><td>0.257</td><td>0.708</td><td>0.769</td><td>0.746</td><td>0.703</td><td>0.754</td></tr></tbody></table>

Table 2: Closed-loop planning results on the NAVSIM v2 navtest split. Bold and underlined values indicate the best and second-best results, respectively. GeoWAM achieves the best EPDMS among all competing methods.

| Method | NC $\uparrow$ | DAC $\uparrow$ | DDC $\uparrow$ | TLC $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | LK $\uparrow$ | HC $\uparrow$ | EC $\uparrow$ | EPDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Transfuser [^5] | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 84.0 |
| Hydra-MDP++ [^19] | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | 81.4 |
| DriveSuprim [^42] | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 | 83.1 |
| ARTEMIS [^8] | 98.3 | 95.1 | 98.6 | 99.8 | 81.5 | 97.4 | 96.5 | 98.3 | – | 83.1 |
| DiffusionDrive [^23] | 98.2 | 96.2 | 99.5 | 99.8 | 87.4 | 97.3 | 96.9 | 98.4 | 87.7 | 88.2 |
| WoTE [^21] | 98.5 | 96.8 | 98.8 | 99.8 | 86.1 | 97.9 | 95.5 | 98.3 | 82.9 | 87.7 |
| DriveVLA-W0 [^20] | 98.4 | 95.2 | 99.4 | 99.9 | 86.6 | 97.9 | 97.8 | 98.3 | 82.7 | 86.9 |
| PWM [^44] | 98.8 | 95.9 | 99.4 | 99.9 | 86.4 | 98.4 | 97.6 | 98.3 | 85.3 | 88.2 |
| DriveLaW [^37] | 98.7 | 96.9 | 99.6 | 99.8 | 87.5 | 98.3 | 97.6 | 98.4 | 77.4 | 88.6 |
| DVGT-2 [^49] | 98.7 | 97.9 | 99.7 | 99.9 | 87.9 | 98.0 | 98.2 | 98.2 | 77.0 | 89.6 |
| EponaV2 [^39] | 98.5 | 97.4 | 99.5 | 99.9 | 87.9 | 98.1 | 97.7 | 98.2 | 77.4 | 88.9 |
| GeoWAM (ours) | 98.7 | 97.7 | 99.7 | 99.9 | 87.0 | 98.1 | 97.9 | 98.3 | 86.8 | 90.2 |

GeoWAM achieves the lowest Abs Rel at every evaluated horizon and improves the aggregate mean from 0.274 for the strongest baseline, Epona+DVGT, to 0.257. It also improves the mean $\delta<1.25$ from 0.655 to 0.754 and performs substantially better from two to four seconds, although Epona+DVGT obtains a higher threshold accuracy at the one-second horizon. These results show that directly forecasting visual geometry preserves future metric structure more effectively than reconstructing geometry from generated RGB frames, while also outperforming the direct geometry-forecasting baseline VGGT-World.

### 4.3 Planning on NAVSIM v2

#### 4.3.1 navtest

Table 2 compares GeoWAM with perception-based planners and recent world-action models on the navtest split. GeoWAM achieves an EPDMS of 90.2, improving upon the DVGT-2 initialization by 0.6 points and establishing the best overall score in the table. It also matches the best DDC and TLC scores, while remaining competitive across the other safety and progress components. Together, these results demonstrate the effectiveness of the visual geometry formulation with a deterministic trajectory decoder.

#### 4.3.2 Two-Stage Planning on navhard

Table 3: navhard leaderboard. Methods trained with reinforcement learning or PDMS-score supervision are shown in gray and marked with $\dagger$. Among the remaining methods, bold and underlined values indicate the best and second-best results, respectively.

<table><thead><tr><th>Method</th><th>Stage</th><th>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>DDC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>TLC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>LK <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>HC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>EC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>EPDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th></tr></thead><tbody><tr><th rowspan="2">CV <sup><a href="#fn:7">7</a></sup></th><th>S1</th><td>88.8</td><td>42.8</td><td>70.6</td><td>99.3</td><td>77.5</td><td>87.3</td><td>78.6</td><td>97.1</td><td>60.4</td><td rowspan="2">11.4</td></tr><tr><th>S2</th><td>83.2</td><td>59.1</td><td>76.5</td><td>98.0</td><td>71.3</td><td>81.1</td><td>47.9</td><td>97.1</td><td>61.9</td></tr><tr><th rowspan="2">Ego MLP <sup><a href="#fn:7">7</a></sup></th><th>S1</th><td>93.2</td><td>55.7</td><td>86.6</td><td>99.3</td><td>81.2</td><td>92.2</td><td>83.5</td><td>97.5</td><td>77.7</td><td rowspan="2">14.1</td></tr><tr><th>S2</th><td>77.2</td><td>51.9</td><td>74.4</td><td>98.2</td><td>77.1</td><td>75.0</td><td>40.8</td><td>97.8</td><td>79.8</td></tr><tr><th rowspan="2">LTF <sup><a href="#fn:5">5</a></sup></th><th>S1</th><td>96.2</td><td>79.5</td><td>99.1</td><td>99.5</td><td>84.1</td><td>95.1</td><td>94.2</td><td>97.5</td><td>79.1</td><td rowspan="2">25.1</td></tr><tr><th>S2</th><td>77.7</td><td>70.2</td><td>84.2</td><td>98.0</td><td>85.1</td><td>75.6</td><td>45.4</td><td>95.7</td><td>75.9</td></tr><tr><th rowspan="2">DriveVLA-W0 <sup><a href="#fn:20">20</a></sup></th><th>S1</th><td>96.8</td><td>83.3</td><td>99.0</td><td>99.6</td><td>84.6</td><td>95.3</td><td>96.4</td><td>97.6</td><td>78.2</td><td rowspan="2">24.4</td></tr><tr><th>S2</th><td>76.8</td><td>64.3</td><td>79.9</td><td>98.3</td><td>89.2</td><td>75.0</td><td>46.8</td><td>95.8</td><td>53.1</td></tr><tr><th rowspan="2">DriveLaW <sup><a href="#fn:37">37</a></sup></th><th>S1</th><td>97.3</td><td>89.1</td><td>99.2</td><td>99.6</td><td>84.3</td><td>97.1</td><td>96.2</td><td>97.8</td><td>67.6</td><td rowspan="2">30.6</td></tr><tr><th>S2</th><td>82.5</td><td>67.6</td><td>83.5</td><td>98.1</td><td>84.8</td><td>78.5</td><td>45.8</td><td>96.4</td><td>57.3</td></tr><tr><th rowspan="2">DVGT-2 <sup><a href="#fn:49">49</a></sup></th><th>S1</th><td>97.2</td><td>91.3</td><td>98.4</td><td>99.8</td><td>84.8</td><td>95.5</td><td>95.5</td><td>97.5</td><td>71.4</td><td rowspan="2">31.7</td></tr><tr><th>S2</th><td>77.8</td><td>73.8</td><td>81.3</td><td>98.3</td><td>91.5</td><td>73.2</td><td>48.0</td><td>83.9</td><td>45.1</td></tr><tr><th rowspan="2">LTFv6 <sup>†</sup> <sup><a href="#fn:24">24</a></sup></th><th>S1</th><td>96.5</td><td>86.6</td><td>99.2</td><td>99.5</td><td>84.4</td><td>95.1</td><td>94.4</td><td>97.7</td><td>76.4</td><td rowspan="2">31.9</td></tr><tr><th>S2</th><td>79.8</td><td>75.5</td><td>86.2</td><td>97.8</td><td>89.5</td><td>76.0</td><td>50.0</td><td>95.2</td><td>66.7</td></tr><tr><th rowspan="2">NavFormer <sup>†</sup> <sup><a href="#fn:3">3</a></sup></th><th>S1</th><td>96.2</td><td>92.4</td><td>95.7</td><td>99.6</td><td>83.8</td><td>96.0</td><td>94.7</td><td>96.4</td><td>60.9</td><td rowspan="2">34.1</td></tr><tr><th>S2</th><td>85.7</td><td>81.0</td><td>83.5</td><td>97.6</td><td>90.1</td><td>82.4</td><td>48.2</td><td>94.9</td><td>48.4</td></tr><tr><th rowspan="2">EponaV2 <sup>†</sup> <sup><a href="#fn:39">39</a></sup></th><th>S1</th><td>97.3</td><td>90.7</td><td>99.4</td><td>100.0</td><td>83.3</td><td>97.3</td><td>97.3</td><td>97.6</td><td>60.9</td><td rowspan="2">36.1</td></tr><tr><th>S2</th><td>83.6</td><td>78.0</td><td>88.0</td><td>98.9</td><td>86.0</td><td>80.3</td><td>50.1</td><td>96.1</td><td>52.0</td></tr><tr><th></th><th>S1</th><td>97.7</td><td>91.5</td><td>99.1</td><td>99.8</td><td>83.8</td><td>95.8</td><td>96.0</td><td>97.8</td><td>79.0</td><td></td></tr><tr><th>GeoWAM (Ours)</th><th>S2</th><td>80.4</td><td>76.3</td><td>87.3</td><td>98.7</td><td>88.9</td><td>76.2</td><td>49.9</td><td>94.0</td><td>56.0</td><td>36.6</td></tr></tbody></table>

We further evaluate GeoWAM under the two-stage pseudo-closed-loop planning protocol on the navhard split [^3]. The benchmark approximates closed-loop evaluation using scenes reconstructed with 3D Gaussian Splatting (3DGS) [^18]. After the planner predicts an ego trajectory, the benchmark renders a new observation from the resulting ego pose and feeds it back to the planner for the next planning step. Planning errors therefore affect subsequent observations and predictions, allowing the benchmark to assess whether a model can recover from accumulated deviations. As reported in Table 3, GeoWAM achieves an EPDMS of 36.6 under this challenging setting, outperforming all baseline methods, including those trained with reinforcement learning or direct PDMS-score supervision [^24] [^3] [^39], which are shown in gray.

### 4.4 Visualization

Figure 3 presents qualitative results for three representative driving maneuvers: turning left, driving straight, and turning right. For each case, we aggregate the predicted geometry from all future time steps into a single visualization, while the bounding boxes indicate the predicted ego poses at successive time steps. Across all three maneuvers, GeoWAM preserves coherent scene structure over the prediction horizon and reconstructs environmental elements such as trees and poles, as well as fine-grained road markings. In the left-turn case, another vehicle follows the ego vehicle through the turn in the predicted future geometry, indicating that GeoWAM captures the dynamics of surrounding agents in addition to ego motion. In the straight-driving case, the predicted ego trajectory steers around a vehicle along the roadside, demonstrating that the forecast geometry provides actionable spatial context for planning.

![[GeoWAM_viz.png|Refer to caption]]

Figure 3: Qualitative visualization of future geometry prediction for left-turn, straight-driving, and right-turn cases. Predictions from all future time steps are aggregated in each scene, and the bounding boxes denote the predicted ego poses at successive time steps. GeoWAM preserves environmental structures and road markings across different driving maneuvers.

## 5 Conclusion

We presented GeoWAM, a visual geometry world action model for autonomous driving. Instead of modeling scene evolution through future-image generation, GeoWAM learns to forecast future geometric features from historical multiview observations under both feature-level and dense point-map supervision. Following geometry pretraining, it adopts an inverse-dynamics-like formulation that infers future ego tokens from the predicted geometric dynamics and maps them to an ego trajectory with a geometry-conditioned action head. This two-stage design transfers predictive geometric knowledge directly to planning without requiring future image synthesis. By placing geometric evolution between observation and action, GeoWAM provides a spatially grounded formulation for world action modeling in autonomous driving.

[^1]: F. Bartoccioni, E. Ramzi, V. Besnier, S. Venkataramanan, T. Vu, Y. Xu, L. Chambon, S. Gidaris, S. Odabas, D. Hurych, R. Marlet, A. Boulch, M. Chen, É. Zablocki, A. Bursuc, E. Valle, and M. Cord (2025) VaViM and vavam: autonomous driving through video generative modeling. arXiv preprint arXiv:2502.15672. Cited by: §1, §2.2.

[^2]: H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom (2020) NuScenes: a multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, Cited by: §4.1.

[^3]: W. Cao, M. Hallgarten, T. Li, D. Dauner, X. Gu, C. Wang, Y. Miron, M. Aiello, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, A. Geiger, and K. Chitta (2025) Pseudo-simulation for autonomous driving. In Conference on Robot Learning (CoRL), Cited by: §1, §4.1, §4.3.2, Table 3, §4.

[^4]: J. Cen, C. Yu, H. Yuan, Y. Jiang, S. Huang, J. Guo, X. Li, Y. Song, H. Luo, F. Wang, D. Zhao, and H. Chen (2025) WorldVLA: towards autoregressive action world model. arXiv preprint arXiv:2506.21539. Cited by: §1, §2.2.

[^5]: K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger (2022) Transfuser: imitation with transformer-based sensor fusion for autonomous driving. IEEE transactions on pattern analysis and machine intelligence 45 (11), pp. 12878–12895. Cited by: Table 2, Table 3.

[^6]: O. Contributors (2023) OpenScene: the largest up-to-date 3d occupancy prediction benchmark in autonomous driving. Note: [https://github.com/OpenDriveLab/OpenScene](https://github.com/OpenDriveLab/OpenScene) Cited by: §4.1.

[^7]: D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, A. Geiger, and K. Chitta (2024) NAVSIM: data-driven non-reactive autonomous vehicle simulation and benchmarking. In Advances in Neural Information Processing Systems (NeurIPS), Cited by: §1, §4.1, Table 3, Table 3, §4.

[^8]: R. Feng, N. Xi, D. Chu, R. Wang, Z. Deng, A. Wang, L. Lu, J. Wang, and Y. Huang (2025) ARTEMIS: autoregressive end-to-end trajectory planning with mixture of experts for autonomous driving. arXiv preprint arXiv:2504.19580. Cited by: Table 2.

[^9]: S. Gao, J. Yang, L. Chen, K. Chitta, Y. Qiu, A. Geiger, J. Zhang, and H. Li (2024) Vista: a generalizable driving world model with high fidelity and versatile controllability. Advances in Neural Information Processing Systems 37, pp. 91560–91596. Cited by: §1, §2.1.

[^10]: A. Geiger, P. Lenz, C. Stiller, and R. Urtasun (2013) Vision meets robotics: the kitti dataset. The International Journal of Robotics Research 32 (11), pp. 1231–1237. Cited by: §4.1.

[^11]: V. Guizilini, R. Ambrus, S. Pillai, A. Raventos, and A. Gaidon (2020) 3D packing for self-supervised monocular depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, Cited by: §4.1.

[^12]: M. Hassan, S. Stapf, A. Rahimi, P. Rezende, Y. Haghighi, D. Brüggemann, I. Katircioglu, L. Zhang, X. Chen, S. Saha, et al. (2025) Gem: a generalizable ego-vision multimodal world model for fine-grained ego-motion, object dynamics, and scene composition control. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22404–22415. Cited by: §1, §2.1.

[^13]: A. Hu, L. Russell, H. Yeo, Z. Murez, G. Fedoseev, A. Kendall, J. Shotton, and G. Corrado (2023) Gaia-1: a generative world model for autonomous driving. arXiv preprint arXiv:2309.17080. Cited by: §1, §2.1.

[^14]: J. Hwang, R. Xu, H. Lin, W. Hung, J. Ji, K. Choi, D. Huang, T. He, P. Covington, B. Sapp, et al. (2024) Emma: end-to-end multimodal model for autonomous driving. arXiv preprint arXiv:2410.23262. Cited by: §1.

[^15]: X. Jia, Z. Yang, Q. Li, Z. Zhang, and J. Yan (2024) Bench2Drive: towards multi-ability benchmarking of closed-loop end-to-end autonomous driving. In Advances in Neural Information Processing Systems Datasets and Benchmarks Track, Cited by: §4.1.

[^16]: B. Jiang, S. Chen, B. Liao, X. Zhang, W. Yin, Q. Zhang, C. Huang, W. Liu, and X. Wang (2024) Senna: bridging large vision-language models and end-to-end autonomous driving. arXiv preprint arXiv:2410.22313. Cited by: §1.

[^17]: N. Keetha, N. Muller, J. Schonberger, L. Porzi, Y. Zhang, T. Fischer, A. Knapitsch, D. Zauss, E. Weber, N. Antunes, et al. (2025) MapAnything: universal feed-forward metric 3d reconstruction. arXiv preprint arXiv:2509.13414. Cited by: §2.3.

[^18]: B. Kerbl, G. Kopanas, T. Leimkühler, and G. Drettakis (2023) 3D gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics 42 (4), pp. 1–14. Cited by: §4.3.2.

[^19]: K. Li, Z. Li, S. Lan, Y. Xie, Z. Zhang, J. Liu, Z. Wu, Z. Yu, and J. M. Alvarez (2025) Hydra-mdp++: advancing end-to-end driving via expert-guided hydra-distillation. arXiv preprint arXiv:2503.12820. Cited by: Table 2.

[^20]: Y. Li, S. Shang, W. Liu, B. Zhan, H. Wang, Y. Wang, Y. Chen, X. Wang, Y. An, C. Tang, L. Hou, L. Fan, and Z. Zhang (2025) DriveVLA-w0: world models amplify data scaling law in autonomous driving. arXiv preprint arXiv:2510.12796. Cited by: §1, §2.2, Table 2, Table 3.

[^21]: Y. Li, Y. Wang, Y. Liu, J. He, L. Fan, and Z. Zhang (2025) End-to-end driving with online trajectory evaluation via bev world model. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 27137–27146. Cited by: Table 2.

[^22]: Y. Li, L. Zhou, S. Yan, B. Liao, T. Yan, K. Xiong, L. Chen, H. Xie, B. Wang, G. Chen, et al. (2026) Unidrivevla: unifying understanding, perception, and action planning for autonomous driving. arXiv preprint arXiv:2604.02190. Cited by: §1.

[^23]: B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang, et al. (2025) Diffusiondrive: truncated diffusion model for end-to-end autonomous driving. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 12037–12047. Cited by: Table 2.

[^24]: L. Nguyen, M. Fauth, B. Jaeger, D. Dauner, M. Igl, A. Geiger, and K. Chitta (2026) Lead: minimizing learner-expert asymmetry in end-to-end driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 39775–39785. Cited by: §4.3.2, Table 3.

[^25]: NVIDIA et al. (2026) Cosmos 3: omnimodal world models for physical ai. arXiv preprint arXiv:2606.02800. Cited by: §4.2, Table 1.

[^26]: C. Shi, J. Xu, S. Shi, K. Sheng, B. Zhang, and L. Jiang (2026) DriveWAM: video generative priors enable scalable world-action modeling for autonomous driving. arXiv preprint arXiv:2605.28544. Cited by: §2.2.

[^27]: P. Sun, H. Kretzschmar, X. Dotiwalla, A. Chouard, V. Patnaik, P. Tsui, J. Guo, Y. Zhou, Y. Chai, B. Caine, V. Vasudevan, W. Han, J. Ngiam, H. Zhao, A. Timofeev, S. Ettinger, M. Krivokon, A. Gao, A. Joshi, S. Zhao, S. Cheng, Y. Zhang, J. Shlens, Z. Chen, and D. Anguelov (2020) Scalability in perception for autonomous driving: waymo open dataset. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, Cited by: §4.1.

[^28]: X. Sun, S. Wang, F. Zhang, L. Liu, C. Jia, Z. Song, Z. Huang, and Y. Luo (2026) VGGT-world: transforming vggt into an autoregressive geometry world model. arXiv preprint arXiv:2603.12655. Cited by: §4.2, Table 1.

[^29]: X. Tian, J. Gu, B. Li, Y. Liu, Y. Wang, Z. Zhao, K. Zhan, P. Jia, X. Lang, and H. Zhao (2024) Drivevlm: the convergence of autonomous driving and large vision-language models. arXiv preprint arXiv:2402.12289. Cited by: §1.

[^30]: J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, and D. Novotny (2025) VGGT: visual geometry grounded transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5294–5306. Cited by: §2.3.

[^31]: Q. Wang, Y. Zhang, A. Holynski, A. A. Efros, and A. Kanazawa (2025) Continuous 3d perception model with persistent state. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10510–10522. Cited by: §2.3.

[^32]: S. Wang, V. Leroy, Y. Cabon, B. Chidlovskii, and J. Revaud (2024) DUSt3R: geometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20697–20709. Cited by: §2.3.

[^33]: X. Wang, Z. Zhu, G. Huang, X. Chen, J. Zhu, and J. Lu (2024) Drivedreamer: towards real-world-drive world models for autonomous driving. In European conference on computer vision, pp. 55–72. Cited by: §1, §2.1.

[^34]: Y. Wang, W. Luo, J. Bai, Y. Cao, T. Che, K. Chen, Y. Chen, J. Diamond, Y. Ding, W. Ding, et al. (2025) Alpamayo-r1: bridging reasoning and action prediction for generalizable autonomous driving in the long tail. arXiv preprint arXiv:2511.00088. Cited by: §1.

[^35]: Y. Wang, J. He, L. Fan, H. Li, Y. Chen, and Z. Zhang (2024) Driving into the future: multiview visual forecasting and planning with world model for autonomous driving. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 14749–14759. Cited by: §1, §2.2.

[^36]: B. Wilson, W. Qi, T. Agarwal, J. Lambert, J. Singh, S. Khandelwal, B. Pan, R. Kumar, A. Hartnett, J. K. Pontes, D. Ramanan, P. Carr, and J. Hays (2021) Argoverse 2: next generation datasets for self-driving perception and forecasting. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, Cited by: §4.1.

[^37]: T. Xia, Y. Li, L. Zhou, J. Yao, K. Xiong, H. Sun, B. Wang, K. Ma, H. Ye, W. Liu, et al. (2025) DriveLaW: unifying planning and video generation in a latent driving world. arXiv preprint arXiv:2512.23421. Cited by: §2.2, Table 2, Table 3.

[^38]: Z. Xiong, X. Ye, B. Yaman, S. Cheng, Y. Lu, J. Luo, N. Jacobs, and L. Ren (2026) UniDrive-wm: unified understanding, planning and generation world model for autonomous driving. arXiv preprint arXiv:2601.04453. Cited by: §1, §2.2.

[^39]: J. Xu, Z. Zhong, Z. Shu, M. Jia, M. Li, J. Bian, Q. Zhang, K. Zhang, J. Xie, J. Yang, et al. (2026) EponaV2: driving world model with comprehensive future reasoning. arXiv preprint arXiv:2605.14696. Cited by: §2.2, §4.3.2, Table 2, Table 3.

[^40]: Y. Yang, J. Mei, Y. Ma, S. Du, W. Chen, Y. Qian, Y. Feng, and Y. Liu (2024) Driving in the occupancy world: vision-centric 4d occupancy forecasting and planning via world models for autonomous driving. arXiv preprint arXiv:2408.14197. Cited by: §2.1.

[^41]: J. Yao, D. D. Kurra, T. Lampo, Z. Cheng, D. Guo, and B. Yaman (2026) VLGA: vision-language-geometry-action models for autonomous driving. arXiv preprint arXiv:2606.12396. Cited by: §1.

[^42]: W. Yao, Z. Li, S. Lan, Z. Wang, X. Sun, J. M. Alvarez, and Z. Wu (2026) Drivesuprim: towards precise trajectory selection for end-to-end planning. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40, pp. 11910–11918. Cited by: Table 2.

[^43]: K. Zhang, Z. Tang, X. Hu, X. Pan, X. Guo, Y. Liu, J. Huang, L. Yuan, Q. Zhang, X. Long, X. Cao, and W. Yin (2025) Epona: autoregressive diffusion world model for autonomous driving. arXiv preprint arXiv:2506.24113. Cited by: §1, §2.2, §4.2, Table 1.

[^44]: Z. Zhao, T. Fu, Y. Wang, L. Wang, and H. Lu (2025) From forecasting to planning: policy world model for collaborative state-action prediction. In Advances in Neural Information Processing Systems, Cited by: §2.2, Table 2.

[^45]: W. Zheng, W. Chen, Y. Huang, B. Zhang, Y. Duan, and J. Lu (2024) OccWorld: learning a 3d occupancy world model for autonomous driving. In European Conference on Computer Vision, pp. 55–72. Cited by: §2.1.

[^46]: Y. Zheng, P. Yang, Z. Xing, Q. Zhang, Y. Zheng, Y. Gao, P. Li, T. Zhang, Z. Xia, P. Jia, and D. Zhao (2025) World4Drive: end-to-end autonomous driving via intention-aware physical latent world model. arXiv preprint arXiv:2507.00603. Cited by: §1.

[^47]: X. Zhou, X. Han, F. Yang, Y. Ma, V. Tresp, and A. Knoll (2026) Opendrivevla: towards end-to-end autonomous driving with large vision language action model. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40, pp. 13782–13790. Cited by: §1.

[^48]: S. Zuo, Z. Xie, W. Zheng, S. Xu, F. Li, S. Jiang, L. Chen, Z. Yang, and J. Lu (2025) DVGT: driving visual geometry transformer. arXiv preprint arXiv:2512.16919. Cited by: §2.3, §4.2, Table 1, Table 1.

[^49]: S. Zuo, Z. Xie, W. Zheng, S. Xu, F. Li, H. Li, L. Chen, Z. Yang, and J. Lu (2026) DVGT-2: vision-geometry-action model for autonomous driving at scale. arXiv preprint arXiv:2604.00813. Cited by: §2.3, §3.1, §4.1, Table 2, Table 3.
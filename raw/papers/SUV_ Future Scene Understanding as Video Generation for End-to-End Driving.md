---
title: "SUV: Future Scene Understanding as Video Generation for End-to-End Driving"
source: "https://arxiv.org/html/2608.03084v1"
author:
published:
created: 2026-09-27
description:
tags:
  - "clippings"
---
Yibo Yuan    Jiacheng Fu    Jiangtong Zhu    Yi Li    Jianhua Han    Meng Tian    Zhuohan Liu    Zhiwei Xiong    Hang Xu    Jianwu Fang    Jianru Xue

###### Abstract

End-to-end driving requires a coherent understanding of future scenes, yet existing methods model these scenes using task-specific heads and output formats, with limited scalability. Can video generation instead provide a shared predictor? We introduce SUV, a unified end-to-end driving framework that casts future Scene Understanding as Video generation using a pretrained video foundation model. SUV models future appearance, semantics, relative depth, and instance-level dynamics as video streams with a shared video expert, without stream-specific visual prediction heads. Through joint video-action attention, the action expert attends to the latent representations of all future streams and generates the ego trajectory. Experiments show that SUV directly predicts all four future streams, while controlled ablations show that structured future supervision and direct future-stream access yield higher trajectory planning scores. With only a single front camera and no candidate-trajectory selection, SUV outperforms a broad set of recent state-of-the-art methods on both NAVSIM-v2 splits, achieving 91.0 EPDMS on navtest and 36.9 on navhard. On the long-tail WOD-E2E benchmark, SUV achieves a competitive RFS of 7.94.

<sup>1</sup> Xi’an Jiaotong University  <sup>2</sup> University of Science and Technology of China

<sup>3</sup> Yinwang Intelligent Technology Co., Ltd. <sup>4</sup> Fudan University

Code: attr/Border \[0 0 0\] user/Subtype /Link /A << /S /URI /URI (https://github.com/ASH-2046/SUV) >>https://github.com/ASH-2046/SUV

## Introduction

In end-to-end planning, a policy predicts a future ego trajectory from an observation history. This prediction requires more than understanding the observed scene. The policy must also anticipate how the scene may evolve over the planning horizon. How to represent this future scene understanding and make it directly usable for trajectory generation remains a central challenge.

Existing end-to-end paradigms address this problem from different directions. Multi-task frameworks such as UniAD [^15], VAD [^21], and SparseDrive [^42] jointly optimize perception, mapping, motion prediction, and planning. Each task usually requires its own queries, predictor, decoder, and objective, with limited module scalability. Vision-language-action models (VLAs) instead use a common language interface to express different driving outputs [^44] [^48] [^20] [^17]. This design makes task expansion easier, but discrete language tokens struggle to capture dense spatial structure and continuous scene motion. Some VLAs augment the policy with future-image prediction or latent world states [^27] [^18]. World-action models (WAMs) directly couple future prediction with action generation [^13] [^1] [^60] [^56]. Most methods only involve pixel-level RGB video or latent vision-state prediction [^4] [^61] [^26] [^51] [^22] [^12]. Yet future scenes contain not only RGB appearance but also road semantics, geometry, and instance-level dynamics. To expose these properties, existing methods usually add task-specific branches, readouts, or auxiliary objectives [^10] [^28] [^63] [^52], falling again into the multi-task paradigms.

![[teaser 2.png|Refer to caption]]

Figure 1: Future scene understanding for end-to-end driving. (a) Multi-task methods share a scene representation but retain task-specific outputs. (b) Language-based methods serialize scene information and trajectories as tokens. (c) SUV uses one shared video expert to generate four future streams and lets the action expert attend to their latent tokens for planning.

Large video generators such as Wan [^43] and Cosmos [^37] learn spatial structure and temporal dynamics from large-scale video data. Recent work builds on these capabilities by adapting pretrained video generators to depth estimation, segmentation, pose estimation, and tracking [^47]. Other studies instead use image generation as a shared interface for visual understanding tasks with heterogeneous outputs [^8] [^11]. These studies leave open whether a single pretrained video generator can jointly model diverse aspects of future scene understanding for end-to-end driving.

We introduce SUV, an end-to-end driving framework that formulates future Scene Understanding as Video generation (Fig. 1(c)). To the best of our knowledge, SUV is the first end-to-end driving framework to generate RGB, segmentation, relative-depth, and instance-track futures as native video streams with a shared video expert. We initialize the video expert from the pretrained Wan2.2-5B generator [^43] and post-train it in the generator’s native latent video space. The expert jointly generates all four streams over a 4-s horizon without stream-specific visual prediction heads. The four streams represent future appearance, semantic layout, scene geometry, and instance-level dynamics, respectively. During joint denoising, the action expert attends to future-stream latents and generates the ego trajectory.

Using only a single front camera, SUV outperforms a broad set of recent state-of-the-art methods on both NAVSIM-v2 [^2] splits, achieving 91.0 EPDMS on navtest and 36.9 on navhard. On the long-tail WOD-E2E benchmark [^53], SUV achieves a competitive RFS of 7.94. Our contributions are summarized as follows.

- We formulate future scene understanding as video generation and post-train a single video foundation model to generate RGB, segmentation, relative-depth, and instance-track futures without stream-specific heads.
- We connect these future representations to trajectory planning through masked joint video-action attention, allowing a separate action expert to attend to all future-stream latents while generating the ego trajectory.
- We evaluate SUV on standard and long-tail planning benchmarks and use controlled ablations to isolate how structured future supervision and direct future-stream access affect planning.

## Related Work

#### Task interfaces for autonomous driving.

Multi-task end-to-end frameworks organize perception, prediction, mapping, and planning around shared bird’s-eye-view (BEV), vectorized, or sparse scene representations [^15] [^21] [^42]. Their shared architectures enable cross-task interaction, but heterogeneous outputs remain tied to task-specific predictors and decoding formats. Vision-language models (VLMs) and vision-language-action models (VLAs) instead express driving outputs in the language space of pretrained multimodal models [^44] [^48] [^20] [^16]. EMMA serializes trajectories, 3D objects, and road graphs as text [^17], while DriveVLA-W0 and DriveWorld-VLA augment VLA planners with visual or latent world modeling [^27] [^18]. SUV differs by generating heterogeneous future scene signals as native video streams with one pretrained video expert.

#### Generative foundation models for visual understanding.

Large-scale video generators such as Wan and Cosmos learn spatial structure and temporal dynamics through generative pretraining [^43] [^37]. Independent work in general vision reveals two uses of generative models beyond synthesis. GenCeption transfers video-generative priors to depth, segmentation, pose, and tracking [^47]. Vision Banana and SenseNova-Vision instead use generative models to produce heterogeneous visual outputs [^8] [^11]. These studies show that generative pretraining can support both transferable perception and shared modeling across visual tasks. SUV extends this direction to future scene understanding in end-to-end driving, where the generated streams must also inform planning.

#### World-action models.

Driving world models first treat future RGB video as a controllable representation [^14] [^49] [^50] [^9], while WAMs couple prediction and action generation [^13] [^1] [^60]. ImagiDrive feeds trajectory-conditioned future frames back to refine trajectories [^24], whereas LMGenDrive co-trains multi-view video and control but omits diffusion generation at inference [^39]. Fast-WAM and Metis use video only as training supervision [^56] [^22]. Existing methods also differ in the representations they predict for future scenes. UniFuture forecasts RGB and depth without planning [^30], whereas ExploreVLA uses RGB and depth futures for dense supervision and prediction uncertainty for exploration [^40]. EponaV2, GeoSem-WAM, X-WAM, and WAM4D add depth or semantics through dedicated heads or readouts [^52] [^36] [^10] [^28]. Feature-space alternatives include SeerDrive’s iterative future-BEV planning and Drive-JEPA’s non-generative predictive embeddings [^57] [^46]. SUV combines native video generation of RGB, segmentation, relative-depth, and instance-track futures in one pretrained expert without stream-specific visual heads, while letting the action expert read all streams during joint denoising.

## Method

![[pipeline 1.png|Refer to caption]]

Figure 2: Overview of SUV. Left: the shared video expert generates four future streams, while the action expert denoises the trajectory using their latent tokens. Right: the token-interaction mask preserves observation-to-stream pathways, blocks cross-stream interaction and action-to-future feedback, and lets action queries read every token group. The label Obs. denotes observation tokens encoded from the input RGB video.

As shown in Fig. 2, SUV models future scene understanding and trajectory planning as a joint video-action generation task. A shared video expert generates RGB, segmentation, relative-depth, and instance-track futures that represent future appearance, semantic layout, scene geometry, and instance-level dynamics. A separate action expert denoises the trajectory tokens while attending to the latent tokens of all four future streams through masked joint video–action attention. At inference, SUV jointly denoises the future streams and ego trajectory in latent space. The action expert therefore uses future-stream latents rather than decoded videos.

### Problem Formulation

At time $t$, the model receives a $K$ -frame front-camera RGB history $\mathbf{o}_{t}=\{o_{t-K+1},\ldots,o_{t}\}$, a navigation command $\mathbf{g}_{t}$, and the current ego state $\mathbf{s}_{t}$. We denote the driving context by $\mathbf{c}_{t}=(\mathbf{g}_{t},\mathbf{s}_{t})$. The model predicts an $H$ -step ego-frame trajectory $\mathbf{a}_{t}=\{(x_{t+h},y_{t+h},\psi_{t+h})\}_{h=1}^{H}$, where $(x,y)$ and $\psi$ denote planar position and heading.

Let $\mathcal{M}=\{\mathrm{rgb},\mathrm{seg},\mathrm{depth},\mathrm{track}\}$ denote the four future-scene stream types. For each $m\in\mathcal{M}$, $\mathbf{v}_{t}^{m}=\{v_{t+j}^{m}\}_{j=1}^{T}$ denotes a $T$ -frame future video, and $\mathbf{V}_{t}=\{\mathbf{v}_{t}^{m}\}_{m\in\mathcal{M}}$ collects all four streams. With parameters $\theta$, SUV models their joint conditional distribution with the ego trajectory as

$$
p_{\theta}(\mathbf{V}_{t},\mathbf{a}_{t}\mid\mathbf{o}_{t},\mathbf{c}_{t}).
$$

The four visual streams describe the same future interval using the same camera view, image grid, and frame timestamps. SUV jointly denoises these streams and the trajectory, while directed attention allows the action tokens to read the evolving future-scene tokens.

### Future Scene Understanding as Video Generation

#### A common video interface.

We encode each future target modality as a three-channel video with the same camera-aligned image grid, prediction horizon, and timestamps as the future RGB stream. This shared representation preserves the generator’s native video interface while extending its pretrained generative capabilities beyond RGB. We use recorded future camera frames for the RGB stream and render semantic labels, relative depth, and instance tracks as separate three-channel video streams.

#### Structured target construction.

A frozen Segment Anything Model 3 (SAM 3) [^3] provides semantic masks and temporally associated instance masks. We render semantic classes with a fixed palette. Each instance track receives a deterministic color from a class-specific palette and retains it throughout the future clip. A frozen Depth Anything 3 (DA3) [^33] provides relative depth, which we robustly normalize within each clip, quantize, and render with a fixed color map. We decode generated structured videos only for future-scene evaluation. The action expert instead reads their evolving latent tokens during joint denoising. Exact normalization bounds, quantization rules, color palettes, and decoding procedures are provided in the supplementary material.

#### Shared latent video space.

A frozen video VAE with encoder $\mathcal{E}_{\mathrm{vae}}$ encodes the observation history and all four future streams:

$$
\mathbf{z}_{t}^{o}=\mathcal{E}_{\mathrm{vae}}(\mathbf{o}_{t}),\qquad\mathbf{z}_{t}^{m}=\mathcal{E}_{\mathrm{vae}}(\mathbf{v}_{t}^{m}),\quad m\in\mathcal{M}.
$$

The four future streams follow a shared observation prefix. All streams use the same latent layout and temporal coordinates. A frozen text encoder maps stream-specific prompts to embeddings that identify the prediction target of each stream. A shared video expert predicts the latent representations of all four streams under the same flow-matching formulation, and the shared VAE decoder decodes each representation into a video. This design requires no stream-specific visual prediction heads.

### Joint Future-Action Generation

#### Video and action experts.

SUV combines a Wan2.2-5B-initialized video expert [^43] shared across four future streams with a separate action expert in a Mixture-of-Transformers architecture [^31]. The video expert processes a clean observation prefix and four future token groups, while the action expert represents the normalized trajectory $\widetilde{\mathbf{a}}=\mathcal{N}_{a}(\mathbf{a})$ as waypoint tokens, where $\mathcal{N}_{a}$ denotes trajectory normalization. Both experts cross-attend to the same common prompt, which contains the navigation command and current ego state. Each future stream additionally cross-attends to a modality-specific embedding that identifies its prediction target as RGB, segmentation, relative depth, or instance tracks. Masked joint self-attention then controls information flow among all token groups.

#### Directed future-to-action attention.

As illustrated by the token interaction mask in the right panel of Fig. 2, we impose asymmetric information flow among token groups. Let $g(i)\in\{\mathrm{obs},\mathrm{act}\}\cup\mathcal{M}$ denote the group of token $i$, and let $\mathcal{A}(g)$ specify the visible key groups:

$$
\mathcal{A}(g)=\begin{cases}\{\mathrm{obs}\},&g=\mathrm{obs},\\
\{\mathrm{obs},m\},&g=m,\;m\in\mathcal{M},\\
\{\mathrm{obs},\mathrm{act}\}\cup\mathcal{M},&g=\mathrm{act}.\end{cases}
$$

For query token $i$ and key token $j$, we set $M_{ij}=0$ when $g(j)\in\mathcal{A}(g(i))$ and $M_{ij}=-\infty$ otherwise. Thus, observation queries read only the clean prefix, each future stream reads the prefix and itself, and action queries read all groups. Applied in every Transformer block and denoising step, this mask exposes future-scene latents to the action expert before video decoding, while blocking cross-stream interaction and action-to-future feedback.

#### Preserving pretrained attention structure.

Cross-stream attention could coordinate future streams but would introduce interactions absent from video pretraining. We therefore retain the pretrained observation-to-future attention pathway for each stream. The shared observation prefix, spatial grid, timestamps, and video expert provide all streams with a common scene context and spatiotemporal prior without direct token exchange. Figure 3 shows consistent road boundaries and vehicle locations across the streams.

### Training and Inference

#### Multi-stream flow matching.

Let $\mathcal{R}=\mathcal{M}\cup\{\mathrm{act}\}$, with clean targets $\mathbf{y}^{m}=\mathbf{z}^{m}$ and $\mathbf{y}^{\mathrm{act}}=\widetilde{\mathbf{a}}$. The visual groups share $\lambda_{\mathrm{vid}}$, while $\lambda_{\mathrm{act}}$ is sampled independently. Writing $\lambda_{r}$ for the corresponding time, we draw $\boldsymbol{\epsilon}^{r}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$, where $\mathbf{I}$ is the identity covariance, and form

$$
\mathbf{y}_{\lambda_{r}}^{r}=(1-\lambda_{r})\mathbf{y}^{r}+\lambda_{r}\boldsymbol{\epsilon}^{r},\qquad\mathbf{u}^{r}=\boldsymbol{\epsilon}^{r}-\mathbf{y}^{r}.
$$

Conditioned on $\lambda_{r}$, one masked forward pass predicts $\hat{\mathbf{u}}_{\theta}^{r}$ for all groups. We optimize

$$
\displaystyle\mathcal{L}_{\mathrm{FM}}^{r}
$$
 
$$
\displaystyle=w(\lambda_{r})\operatorname{MSE}\left(\hat{\mathbf{u}}_{\theta}^{r},\mathbf{u}^{r}\right),
$$
$$
\displaystyle\mathcal{L}
$$
 
$$
\displaystyle=\frac{1}{4}\sum_{m\in\mathcal{M}}\mathcal{L}_{\mathrm{FM}}^{m}+\mathcal{L}_{\mathrm{FM}}^{\mathrm{act}},
$$

Here $\operatorname{MSE}$ denotes mean-squared error and $w$ is the normalized scheduler weight. We jointly train both experts and the ego-state projection while keeping the video VAE and text encoder frozen.

#### Joint inference.

During inference, we synchronize the video and action flow times on a common shifted grid from $1$ to $0$. The observation prefix remains fixed, while all five prediction groups start from Gaussian noise. At each solver step, SUV predicts the velocity for every group in a masked forward pass while the action expert reads the current future-scene latents. The solver then updates all groups using these predictions. After the final step, $\mathcal{N}_{a}^{-1}$ restores the ego-frame trajectory. Planning remains in latent space. The VAE decoder is used only for evaluation or visualization. Trajectory normalization, scheduler, and Euler details are provided in the supplementary material.

## Experiments

We investigate the following research questions: 1) How does SUV perform across standard and long-tail planning benchmarks? 2) Can one video expert jointly predict all four future streams without compromising RGB quality, and how important is video-generative pretraining? 3) How do structured future supervision and direct future-stream access affect planning? 4) How does SUV balance planning performance and inference efficiency?

<table><tbody><tr><td>Method</td><td>Sensors</td><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DDC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TLC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>LK <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>HC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EPDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>Human Agent</td><td>–</td><td>100</td><td>100</td><td>99.8</td><td>100</td><td>87.4</td><td>100</td><td>100</td><td>98.1</td><td>90.1</td><td>94.5</td></tr><tr><td colspan="12">Traditional E2E policies</td></tr><tr><td>DiffusionDriveV2 <sup><a href="#fn:65">65</a></sup></td><td>3 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> C+L</td><td>97.7</td><td>96.6</td><td>99.2</td><td>99.8</td><td>88.9</td><td>97.2</td><td>96.0</td><td>97.8</td><td>91.0</td><td>87.5</td></tr><tr><td>DriveSuprim <sup><a href="#fn:54">54</a></sup></td><td>3 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> C</td><td>98.4</td><td>98.6</td><td>99.6</td><td>99.8</td><td>90.5</td><td>97.8</td><td>97.0</td><td>98.3</td><td>78.6</td><td>87.1</td></tr><tr><td>SparseDriveV2 <sup><a href="#fn:41">41</a></sup></td><td>3 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> C</td><td>98.1</td><td>98.1</td><td>99.6</td><td>99.8</td><td>91.1</td><td>97.3</td><td>96.9</td><td>98.2</td><td>78.4</td><td>90.1</td></tr><tr><td colspan="12">VLA-based policies</td></tr><tr><td>DriveVLA-W0 <sup><a href="#fn:27">27</a></sup></td><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> C</td><td>98.4</td><td>95.2</td><td>99.4</td><td>99.9</td><td>86.6</td><td>97.9</td><td>97.8</td><td>98.3</td><td>82.7</td><td>86.9</td></tr><tr><td>DriveWorld-VLA <sup><a href="#fn:18">18</a></sup></td><td>3 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> C</td><td>98.6</td><td>99.1</td><td>99.6</td><td>99.8</td><td>87.4</td><td>97.9</td><td>97.0</td><td>97.8</td><td>78.6</td><td>86.8</td></tr><tr><td>SGDrive <sup><a href="#fn:23">23</a></sup></td><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> C</td><td>98.6</td><td>94.3</td><td>99.5</td><td>99.8</td><td>86.0</td><td>97.9</td><td>96.1</td><td>98.3</td><td>85.9</td><td>86.2</td></tr><tr><td>DriveFine <sup><a href="#fn:6">6</a></sup></td><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> C</td><td>98.7</td><td>97.3</td><td>99.5</td><td>99.8</td><td>88.7</td><td>97.8</td><td>97.7</td><td>98.4</td><td>83.8</td><td>89.7</td></tr><tr><td colspan="12">WAM-based policies</td></tr><tr><td>EponaV2 <sup><a href="#fn:52">52</a></sup></td><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> C</td><td>98.5</td><td>97.4</td><td>99.5</td><td>99.9</td><td>87.9</td><td>98.1</td><td>97.7</td><td>98.2</td><td>77.4</td><td>88.9</td></tr><tr><td>Metis <sup><a href="#fn:22">22</a></sup></td><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> C</td><td>98.4</td><td>97.2</td><td>99.6</td><td>99.8</td><td>87.8</td><td>97.7</td><td>97.8</td><td>98.4</td><td>88.0</td><td>89.5</td></tr><tr><td>Metis (Top 6) <sup><a href="#fn:22">22</a></sup></td><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> C</td><td>98.5</td><td>97.5</td><td>99.6</td><td>99.8</td><td>87.9</td><td>97.8</td><td>98.0</td><td>98.4</td><td>90.0</td><td>90.3</td></tr><tr><td>SUV (Ours)</td><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> C</td><td>99.1</td><td>97.8</td><td>99.7</td><td>99.8</td><td>87.8</td><td>98.7</td><td>98.1</td><td>98.4</td><td>88.3</td><td>91.0</td></tr></tbody></table>

Table 1: End-to-end trajectory planning on NAVSIM-v2 navtest. Results use the corrected official EPDMS implementation. C and L denote camera and LiDAR. Bold and underlined values mark the best and second-best learned results.

<table><thead><tr><th>Method</th><th>Stage</th><th>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>DDC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>TLC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>LK <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>HC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>EC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>S.<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>EPDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th></tr></thead><tbody><tr><td colspan="13">Traditional E2E policies</td></tr><tr><td rowspan="2">LTF <sup><a href="#fn:5">5</a></sup></td><td>S1</td><td>96.2</td><td>79.6</td><td>99.1</td><td>99.6</td><td>84.1</td><td>95.1</td><td>94.2</td><td>97.6</td><td>79.1</td><td>–</td><td></td></tr><tr><td>S2</td><td>77.8</td><td>70.2</td><td>84.3</td><td>98.1</td><td>85.1</td><td>85.1</td><td>45.4</td><td>95.7</td><td>76.0</td><td>–</td><td>25.1</td></tr><tr><td rowspan="2">GuideFlow <sup><a href="#fn:34">34</a></sup></td><td>S1</td><td>96.6</td><td>80.5</td><td>96.3</td><td>99.3</td><td>82.3</td><td>94.9</td><td>91.5</td><td>97.7</td><td>67.8</td><td>–</td><td></td></tr><tr><td>S2</td><td>87.3</td><td>76.7</td><td>88.8</td><td>99.2</td><td>84.3</td><td>85.1</td><td>49.7</td><td>93.1</td><td>44.5</td><td>–</td><td>27.1</td></tr><tr><td colspan="13">VLA-based policies</td></tr><tr><td rowspan="2">DriveVLA-W0 <sup><a href="#fn:27">27</a></sup></td><td>S1</td><td>96.8</td><td>83.3</td><td>99.0</td><td>99.6</td><td>84.6</td><td>95.3</td><td>96.4</td><td>97.6</td><td>78.2</td><td>–</td><td></td></tr><tr><td>S2</td><td>76.8</td><td>64.3</td><td>79.9</td><td>98.3</td><td>89.2</td><td>75.0</td><td>46.8</td><td>95.8</td><td>53.1</td><td>–</td><td>24.4</td></tr><tr><td rowspan="2">ReCogDrive <sup><a href="#fn:29">29</a></sup></td><td>S1</td><td>96.4</td><td>78.9</td><td>98.7</td><td>99.8</td><td>82.6</td><td>95.6</td><td>94.4</td><td>97.6</td><td>74.2</td><td>67.7</td><td></td></tr><tr><td>S2</td><td>80.2</td><td>65.0</td><td>82.4</td><td>98.7</td><td>85.2</td><td>76.9</td><td>43.8</td><td>96.6</td><td>71.8</td><td>37.6</td><td>25.7</td></tr><tr><td rowspan="2">SGDrive <sup><a href="#fn:23">23</a></sup></td><td>S1</td><td>95.8</td><td>87.6</td><td>97.8</td><td>99.8</td><td>84.4</td><td>94.7</td><td>92.9</td><td>97.8</td><td>28.9</td><td>71.1</td><td></td></tr><tr><td>S2</td><td>79.4</td><td>65.4</td><td>79.1</td><td>98.9</td><td>88.9</td><td>75.3</td><td>42.7</td><td>96.4</td><td>29.6</td><td>35.2</td><td>25.5</td></tr><tr><td colspan="13">WAM-based policies</td></tr><tr><td rowspan="2">EponaV2 <sup><a href="#fn:52">52</a></sup></td><td>S1</td><td>97.3</td><td>90.7</td><td>99.4</td><td>100</td><td>83.3</td><td>97.3</td><td>97.3</td><td>97.6</td><td>60.9</td><td>–</td><td></td></tr><tr><td>S2</td><td>83.6</td><td>78.0</td><td>88.0</td><td>98.9</td><td>86.0</td><td>80.3</td><td>50.1</td><td>96.1</td><td>52.0</td><td>–</td><td>36.1</td></tr><tr><td rowspan="2">Metis <sup><a href="#fn:22">22</a></sup></td><td>S1</td><td>96.6</td><td>87.8</td><td>99.0</td><td>99.3</td><td>84.5</td><td>95.6</td><td>97.8</td><td>97.8</td><td>77.8</td><td>75.8</td><td></td></tr><tr><td>S2</td><td>79.6</td><td>73.3</td><td>84.9</td><td>97.8</td><td>85.8</td><td>76.6</td><td>47.7</td><td>95.4</td><td>75.3</td><td>41.7</td><td>32.2</td></tr><tr><td rowspan="2">SUV (Ours)</td><td>S1</td><td>96.9</td><td>94.2</td><td>99.3</td><td>99.6</td><td>84.1</td><td>95.6</td><td>96.7</td><td>97.8</td><td>79.6</td><td>82.3</td><td></td></tr><tr><td>S2</td><td>82.7</td><td>74.2</td><td>85.9</td><td>98.4</td><td>86.0</td><td>78.9</td><td>47.2</td><td>95.9</td><td>69.1</td><td>43.9</td><td>36.9</td></tr></tbody></table>

Table 2: End-to-end trajectory planning on NAVSIM-v2 navhard. Stage-1, Stage-2, and combined results follow the official protocol and are ranked separately. S. is the per-stage score. Bold and underlined values mark the best and second-best results.

### Experimental Setup

#### Implementation details.

On NAVSIM, the model takes four front-camera frames sampled at 2 Hz over a 2-s observation window at $640\times 384$ resolution. It predicts eight future frames and eight trajectory waypoints over the next 4 s. The action expert has a hidden dimension of 1024 and contains approximately 1B parameters. The full model contains approximately 6B parameters. We train SUV for 60 epochs on eight NVIDIA H200 GPUs using AdamW with a learning rate of $1\times 10^{-4}$ and weight decay of $0.01$. We use a cosine learning-rate schedule, bfloat16 precision, gradient clipping at $1.0$, and random seed 42. Unless stated otherwise, inference uses 10 solver steps.

#### Planning evaluation.

We train SUV on NAVSIM navtrain and evaluate it on the NAVSIM-v2 navtest and navhard splits using the corrected official EPDMS implementation [^7] [^2]. Navtest uses a one-stage protocol, whereas navhard combines 450 original observations in Stage 1 with 5,462 pre-generated 3DGS observations in Stage 2. We also train and evaluate SUV on the long-tail WOD-E2E benchmark following its official protocol [^53]. WOD-E2E evaluates 5-s planning using RFS and RFS-GT ADE at 3 and 5 s.

#### Future-scene evaluation.

We evaluate all four future streams on 12,146 samples from NAVSIM-v2 navtest. We assess RGB, segmentation, and relative depth at eight horizons from 0.5 to 4.0 s, and instance tracking at seven horizons from 1.0 to 4.0 s. For each metric, we report the unweighted mean over its evaluation horizons. We use PSNR and SSIM for RGB, mIoU for segmentation, $\delta_{1}$ and AbsRel for relative depth, and AssA@50 for instance tracking. SAM 3 provides the segmentation and tracking references, whereas DA3 provides the relative-depth references. We evaluate segmentation and tracking within a predefined 50-m region and relative depth over the full image. The segmentation, tracking, and depth metrics measure agreement with frozen teachers rather than accuracy against independent ground truth. Generate-then-Perceive applies the frozen teachers to its predicted future RGB sequence, whereas the same teachers derive the evaluation references from the recorded future RGB frames. The baseline uses the same horizons, regions, references, and metrics as native generation, providing a task-aligned comparison between native structured generation and post hoc recovery. The supplementary material provides detailed target-construction and decoding procedures.

### Planning Performance Across Benchmarks

Using only a single front camera, SUV outperforms a broad set of recent state-of-the-art methods on both NAVSIM-v2 splits. On NAVSIM-v2 navtest, SUV achieves 91.0 EPDMS (Table 2). Metis with top-6 trajectory selection achieves 90.3 EPDMS, 0.7 points lower. At the component level, SUV records the highest NC, DDC, TTC, and LK scores and ties for the highest HC score. Other methods record higher EP or EC scores. On navhard, SUV achieves 36.9 EPDMS (Table 2). EponaV2 achieves 36.1 EPDMS, 0.8 points lower.

On the long-tail WOD-E2E benchmark, SUV achieves a competitive RFS of 7.94 (Table 3). It records RFS-GT ADE values of 1.24 and 2.90 at 3 and 5 s, respectively. IRL-VLA records lower RFS-GT ADE values of 1.22 and 2.82 at the same horizons. RFS rewards trajectories judged appropriate for a scene, whereas RFS-GT ADE measures geometric deviation from the recorded expert trajectory. The two metrics need not rank methods identically because a scene can admit multiple reasonable trajectories.

| Method | ADE 3/5 $\downarrow$ | RFS $\uparrow$ |
| --- | --- | --- |
| AutoVLA [^64] | 1.35 / 2.96 | 7.56 |
| HMVLM [^45] | 1.33 / 3.07 | 7.74 |
| Fast-dDrive [^58] | 1.25 / 2.91 | 7.82 |
| IRL-VLA [^19] | 1.22 / 2.82 | 7.89 |
| Poutine-Base [^38] | 1.27 / 2.94 | 7.91 |
| SUV (Ours) | 1.24 / 2.90 | 7.94 |

Table 3: WOD-E2E long-tail test results. ADE 3/5 denotes RFS-GT ADE at 3 and 5 s. Bold and underlined values mark the best and second-best results.

### Unified Prediction of Heterogeneous Futures

The shared video expert, initialized from Wan2.2-5B and trained on all four streams, predicts RGB, segmentation, relative depth, and instance-track futures over 4 s. Under matched settings, the multi-stream model achieves higher RGB PSNR and SSIM than the RGB-only model at all eight horizons. The multi-stream model achieves a mean PSNR of 19.55 dB and a mean SSIM of 0.5844, whereas the RGB-only model achieves 19.44 dB and 0.5820. The multi-stream model therefore shows no reduction in the reported RGB metrics, although the small differences do not establish an improvement. The supplementary material reports the horizon-wise curves for both settings.

The initialization ablation fixes the architecture, data, objectives, and optimization schedule to isolate the effect of video-expert initialization (Table 4). Wan2.2-5B initialization yields better point estimates for all five metrics, including a 14.5-point increase in mIoU and a 7.3-point reduction in AbsRel relative to random initialization.

| Init. | PSNR $\uparrow$ | SSIM $\uparrow$ | mIoU $\uparrow$ | AbsRel $\downarrow$ | AssA@50 $\uparrow$ |
| --- | --- | --- | --- | --- | --- |
| Random | 18.51 | 0.5432 | 49.7 | 26.8 | 81.4 |
| Wan2.2-5B | 19.55 | 0.5844 | 64.2 | 19.5 | 84.4 |

Table 4: Effect of video-generative initialization under matched settings. Metrics are unweighted means over their evaluation horizons. AbsRel is reported in percent.

Table 5 compares native generation with Generate-then-Perceive. Native generation achieves a higher $\delta_{1}$ (77.04 versus 70.99) and a lower AbsRel (19.49 versus 22.13). Generate-then-Perceive achieves higher mIoU (66.81 versus 64.22) and AssA@50 (86.88 versus 84.39). The horizon-wise results in the supplementary material preserve the same ordering. SUV directly predicts segmentation, relative depth, and instance-track futures with one shared video expert and no stream-specific visual prediction heads.

Figure 3 shows the four jointly generated future streams in a lane-change scenario. The road layout and vehicle locations remain consistent across the four streams. The planned trajectory closely matches the ground-truth trajectory and achieves an EPDMS of 100. The supplementary material provides additional horizon-wise curves and qualitative cases.

| Prediction route | mIoU $\uparrow$ | $\delta_{1}\uparrow$ | AbsRel $\downarrow$ | AssA@50 $\uparrow$ |
| --- | --- | --- | --- | --- |
| Generate-then-Perceive | 66.81 | 70.99 | 22.13 | 86.88 |
| Native Generation (SUV) | 64.22 | 77.04 | 19.49 | 84.39 |

Table 5: Direct versus post-hoc structured future prediction. All metrics use teacher-generated references. The mIoU, $\delta_{1}$, and AbsRel values are unweighted means over 0.5–4.0 s. The AssA@50 values are unweighted means over 1.0–4.0 s.

![[vis 1.png|Refer to caption]]

Figure 3: Qualitative multi-stream prediction and planning in a lane-change scenario. Left: Past video and planning. Right: The generated future streams have clear visual content, road layout, depth maps and consistent vehicle IDs across 4 seconds.

### Planning Benefits of Future Representations

Table 7 evaluates structured future supervision and future-stream access in a $2\times 2$ design. All four configurations use RGB future supervision, and the first row is the baseline without S/G/I supervision or future-stream access. S/G/I adds segmentation, relative-depth, and instance-track objectives. Access allows the action expert to read the RGB stream when S/G/I is absent and all four future streams when S/G/I is present. The four rows therefore represent the baseline, Access alone, S/G/I alone, and their combination. The S/G/I-only configuration is SUV w/o Future Access. It retains all four training objectives and action-prompt cross-attention but prevents the action expert from reading future-stream tokens. Comparing the two no-access rows isolates structured future supervision. S/G/I raises EPDMS from 89.7 to 90.7 on navtest and from 30.5 to 32.8 on navhard. Comparing the two S/G/I rows isolates future-stream access. Access raises EPDMS from 90.7 to 91.0 on navtest and from 32.8 to 36.9 on navhard. The corresponding Stage 1 and Stage 2 scores increase by 2.5 and 2.4 points. Under RGB-only supervision, Access raises EPDMS from 89.7 to 90.6 on navtest and from 30.5 to 35.0 on navhard.

Table 7 isolates access to each structured future stream. Each ablated variant retains all four training objectives and RGB future-stream access while removing access to one structured stream. Removing segmentation, relative-depth, or instance-track access lowers navhard EPDMS from 36.9 to 36.4, 35.8, and 35.4, respectively. All three variants remain within 0.2 points of the full configuration on navtest. Relative-depth removal causes the largest Stage 1 decrease of 1.1 points, while instance-track removal causes the largest Stage 2 decrease of 1.3 points. Together, the ablations show that structured future supervision improves the RGB-supervised no-access baseline and that direct future-stream access provides further gains, particularly on navhard.

<table><thead><tr><th rowspan="2">S/G/I</th><th rowspan="2">Access</th><th>navtest</th><th colspan="3">navhard</th></tr><tr><th>EPDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>S1 <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>S2 <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>EPDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th></tr></thead><tbody><tr><th>✗</th><th>✗</th><th>89.7</th><td>77.0</td><td>39.9</td><td>30.5</td></tr><tr><th>✗</th><th>✓</th><th>90.6</th><td>80.0</td><td>43.8</td><td>35.0</td></tr><tr><th>✓</th><th>✗</th><th>90.7</th><td>79.8</td><td>41.5</td><td>32.8</td></tr><tr><th>✓</th><th>✓</th><th>91.0</th><td>82.3</td><td>43.9</td><td>36.9</td></tr></tbody></table>

Table 6: Structured future supervision and future-stream access. All rows use RGB future supervision, and S/G/I denotes segmentation, relative-depth, and instance-track supervision.

<table><tbody><tr><th rowspan="2">Variant</th><th>navtest</th><td colspan="3">navhard</td></tr><tr><th>EPDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><td>S1 <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>S2 <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EPDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><th>Full</th><th>91.0</th><td>82.3</td><td>43.9</td><td>36.9</td></tr><tr><th>w/o Seg. Access</th><th>90.9</th><td>82.6</td><td>43.2</td><td>36.4</td></tr><tr><th>w/o Depth Access</th><th>90.8</th><td>81.2</td><td>43.9</td><td>35.8</td></tr><tr><th>w/o Track Access</th><th>90.9</th><td>82.0</td><td>42.6</td><td>35.4</td></tr></tbody></table>

Table 7: Access to individual structured future streams. All variants retain the four training objectives and RGB future-stream access. S1 and S2 denote the two navhard stages.

### Accuracy–Latency Trade-off

<table><thead><tr><th>Model</th><th>Steps</th><th>navtest <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>navhard <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>ms <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th><th>Hz <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th></tr><tr><th>DriveVLA-W0</th><th>–</th><th>86.9</th><th>24.4</th><th>690</th><th>1.45</th></tr></thead><tbody><tr><td rowspan="3">SUV</td><td>1</td><td>89.8</td><td>33.0</td><td>177</td><td>5.65</td></tr><tr><td>2</td><td>91.0</td><td>36.1</td><td>288</td><td>3.48</td></tr><tr><td>10</td><td>91.0</td><td>36.9</td><td>1356</td><td>0.74</td></tr></tbody></table>

Table 8: Accuracy–latency comparison. EPDMS follows the NAVSIM-v2 protocol. Each latency value is the mean of 500 runs on a single GeForce RTX 4090.

Table 8 compares SUV’s solver settings with DriveVLA-W0 under the same GeForce RTX 4090 timing protocol. With one solver step, SUV reaches 89.8 EPDMS on navtest and 33.0 on navhard at 177 ms, exceeding DriveVLA-W0 by 2.9 and 8.6 points while reducing latency from 690 to 177 ms, a 3.9 $\times$ speedup. Two steps raise the scores to 91.0 and 36.1 at 288 ms, remaining 2.4 $\times$ faster than DriveVLA-W0. Ten steps yield the highest navhard score of 36.9 at 1356 ms. The solver therefore exposes a controllable accuracy–latency range, with the one- and two-step settings improving both planning scores and latency over DriveVLA-W0 on this GPU.

## Conclusion

We introduced SUV, a unified driving framework that models RGB, segmentation, relative depth, and instance-track futures as video streams using a shared video expert without stream-specific visual prediction heads. A separate action expert attends to their latent tokens to generate the ego trajectory. Using a single front camera, SUV outperforms a broad set of recent state-of-the-art methods on both NAVSIM-v2 splits, achieving 91.0 EPDMS on navtest and 36.9 on navhard. On the long-tail WOD-E2E benchmark, SUV achieves a competitive RFS of 7.94. Controlled ablations show that Wan2.2-5B initialization yields higher future-prediction point estimates than random initialization. Structured supervision raises planning point estimates without future-stream access, while direct access provides additional gains, particularly on navhard. These results support video generation as a shared future-scene predictor for planning.

## References

SUV: Future Scene Understanding as Video Generation for End-to-End Driving

Supplementary Material

This supplement defines the planning and future-scene metrics used in the main paper (Section 1). Section 2 presents additional planning results and qualitative examples. Section 3 reports horizon-wise future-scene results. Section 4 documents structured targets, benchmark configuration, training, and inference.

## 1 Evaluation Metrics

### 1.1 Planning Metrics

All NAVSIM component scores lie in $[0,1]$, and leaderboard scores are split means multiplied by 100. WOD-E2E reports RFS on a $[0,10]$ scale.

#### NAVSIM PDMS and EPDMS.

PDMS and EPDMS combine multiplicative component scores with a weighted average of the remaining terms [^7] [^2]. For $x_{c}\in[0,1]$, let $\mathcal{P}$ and $\mathcal{B}$ denote these two groups:

$$
\mathrm{Score}=\left(\prod_{c\in\mathcal{P}}x_{c}\right)\times\frac{\sum_{c\in\mathcal{B}}w_{c}\times x_{c}}{\sum_{c\in\mathcal{B}}w_{c}}.
$$

A zero multiplier makes $\mathrm{Score}=0$.

| Term | Meaning | PDMS | EPDMS |
| --- | --- | --- | --- |
| NC | No at-fault collisions | M | M |
| DAC | Drivable-area compliance | M | M |
| DDC | Driving-direction compliance | – | M |
| TLC | Traffic-light compliance | – | M |
| TTC | Time to collision | 5 | 5 |
| EP | Ego progress | 5 | 5 |
| C | Comfort | 2 | – |
| LK | Lane keeping | – | 2 |
| HC | History comfort | – | 2 |
| EC | Extended comfort | – | 2 |

Table 1: PDMS and EPDMS component roles and weights. M marks multipliers, and dashes mark absent components.

For EPDMS, $x_{c}$ is the effective component score after human-penalty filtering. The evaluator applies this filter to NC, DAC, DDC, TLC, EP, TTC, LK, and HC in original scenes. If the corresponding human score is 0, the component is set to 1. Synthetic follow-ups and EC are evaluated without this filter. The weighted-term denominator is 12 for PDMS, 16 for EPDMS with EC, and 14 when EC is unavailable.

#### NAVSIM-v2 navhard.

Navhard applies the EPDMS defined above within a two-stage pseudo-simulation protocol [^2]. Stage-1 evaluates an initial 4-s plan on a logged observation, whereas Stage-2 evaluates new 4-s plans on associated 3DGS follow-up observations. Each follow-up represents a plausible outcome of a different precomputed Stage-1 rollout. This protocol retains offline evaluation while testing planning after an observation shift rather than a continuous 8-s rollout.

Let $\mathcal{G}$ denote the set of Stage-1 scenes. For $g\in\mathcal{G}$, $s_{g}^{(1)}\in[0,1]$ is its unscaled EPDMS and $\mathbf{e}_{g}=(e_{g}^{x},e_{g}^{y})$ is its predicted rear-axle endpoint. For follow-up $k$, $s_{gk}^{(2)}\in[0,1]$ is its unscaled EPDMS and $\mathbf{b}_{gk}=(b_{gk}^{x},b_{gk}^{y})$ is its rear-axle start position. The evaluator assigns larger normalized Gaussian weights to follow-ups whose start positions are closer to $\mathbf{e}_{g}$: $\alpha_{gk}\propto\exp[-\|\mathbf{e}_{g}-\mathbf{b}_{gk}\|_{2}^{2}/(2\times\sigma^{2})]$, where $\sum_{k}\alpha_{gk}=1$ and $\sigma^{2}=0.1\,\mathrm{m}^{2}$. The combined score is

$$
\displaystyle\mathrm{EPDMS}_{\mathrm{hard}}
$$
 
$$
\displaystyle=\frac{100}{|\mathcal{G}|}\times\sum_{g\in\mathcal{G}}s_{g}^{(1)}\times\left(\sum_{k}\alpha_{gk}\times s_{gk}^{(2)}\right).
$$

The inner sum estimates Stage-2 performance near the predicted Stage-1 endpoint. Its product with $s_{g}^{(1)}$ requires both plans to score well. The outer mean aggregates all Stage-1 scenes, and the factor $100$ converts the unscaled mean to the leaderboard scale.

#### WOD-E2E RFS.

Rater Feedback Score (RFS) compares a predicted 5-s trajectory with three human-rated references, each assigned a score $s_{p}\in[0,10]$ [^53]. At each evaluation time $t\in\{3,5\}$  s, the absolute longitudinal and lateral errors, measured in the local frame of reference trajectory $p$, are normalized by initial-speed-dependent trust-region thresholds:

$$
\displaystyle d_{p,t}
$$
 
$$
\displaystyle=\max\left(\frac{\Delta_{\mathrm{lng},p,t}}{\tau_{\mathrm{lng},t}},\frac{\Delta_{\mathrm{lat},p,t}}{\tau_{\mathrm{lat},t}}\right),
$$
$$
\displaystyle r_{p,t}
$$
 
$$
\displaystyle=s_{p}\times 0.1^{\max(d_{p,t}-1,0)},
$$
$$
\displaystyle\widetilde{\mathrm{RFS}}
$$
 
$$
\displaystyle=\frac{1}{2}\times\sum_{t\in\{3,5\}}\max_{p\in\{1,2,3\}}r_{p,t}.
$$

A prediction inside a trust region retains $s_{p}$, and its score decays exponentially outside the region. The maximum is taken independently over references at 3 and 5 s, and $\widetilde{\mathrm{RFS}}$ averages the two values. The floor applies when no single reference satisfies both $d_{p,3}\leq 1$ and $d_{p,5}\leq 1$. In this case, $\mathrm{RFS}=\max(4,\widetilde{\mathrm{RFS}})$. Otherwise, $\mathrm{RFS}=\widetilde{\mathrm{RFS}}$. Dataset-level RFS is the mean score across the 11 scenario clusters. RFS-GT ADE reports 3- and 5-s displacement from the highest-rated recorded reference, averaged over test frames.

### 1.2 Future-Scene Metrics

The future-scene metrics below are evaluated on NAVSIM-v2 navtest. RGB metrics compare predicted and recorded future frames. Structured metrics compare decoded predictions with frozen-teacher targets derived from recorded future RGB. Section 4.1 details target encoding and decoding.

| Metric | Meaning |
| --- | --- |
| PSNR $\uparrow$ | Pixel fidelity on a logarithmic scale. |
| SSIM $\uparrow$ | Structural similarity between RGB frames. |
| mIoU $\uparrow$ | Mean semantic overlap across evaluated classes. |
| $\delta_{1}\uparrow$ | Fraction of depth pixels within a factor of 1.25. |
| AbsRel $\downarrow$ | Mean absolute relative depth error. |
| AssA@50 $\uparrow$ | Track-association accuracy at a mask-IoU threshold of 0.50. |

Table 2: Future-scene evaluation metrics. Arrows indicate the preferred direction.

PSNR and SSIM use all image pixels and are averaged equally over clips at each horizon. Relative depth metrics cover the full image without the 50-m restriction and use the valid percentile mask $\mathcal{V}_{c,h}$ defined below. To focus on driving-relevant content, mIoU and AssA@50 evaluate Car, Truck, Bus, Bicycle, and Pedestrian within a $50\,\mathrm{m}$ ego-centric region. At each horizon, segmentation predictions are decoded to the nearest palette color, and mIoU is computed from a confusion matrix pooled over the split. Classes with zero union are excluded.

For relative depth, $c$ indexes a clip and $h\in\{1,\ldots,8\}$ indexes a future frame. Let $q_{0.01,c}$ and $q_{0.99,c}$ denote the first and 99th percentiles of the finite, positive DA3-reference depths in clip $c$. The set $\mathcal{V}_{c,h}$ contains full-image pixels whose reference depths lie within this interval. We estimate one scale and shift jointly from all eight frames:

$$
\displaystyle(a_{c},b_{c})
$$
 
$$
\displaystyle=\arg\min_{a,b}\sum_{h=1}^{8}\sum_{p\in\mathcal{V}_{c,h}}\left(a\widehat{d}_{c,h}(p)+b-d_{c,h}(p)\right)^{2},
$$
$$
\displaystyle\widetilde{d}_{c,h}(p)
$$
 
$$
\displaystyle=\operatorname{clip}_{[q_{0.01,c},q_{0.99,c}]}\!\left(a_{c}\widehat{d}_{c,h}(p)+b_{c}\right),
$$

where $\widehat{d}$ denotes the decoded normalized depth prediction and $d$ denotes the DA3-reference relative depth. The same $(a_{c},b_{c})$ and clipping interval are used at every horizon in clip $c$. We then compute

$$
\displaystyle\mathrm{AbsRel}_{c,h}
$$
 
$$
\displaystyle=\frac{1}{|\mathcal{V}_{c,h}|}\sum_{p\in\mathcal{V}_{c,h}}\frac{|\widetilde{d}_{c,h}(p)-d_{c,h}(p)|}{d_{c,h}(p)},
$$
$$
\displaystyle\rho_{c,h}(p)
$$
 
$$
\displaystyle=\max\left(\frac{\widetilde{d}_{c,h}(p)}{d_{c,h}(p)},\frac{d_{c,h}(p)}{\widetilde{d}_{c,h}(p)}\right),
$$
$$
\displaystyle\delta_{1,c,h}
$$
 
$$
\displaystyle=\frac{1}{|\mathcal{V}_{c,h}|}\sum_{p\in\mathcal{V}_{c,h}}\mathbf{1}\!\left[\rho_{c,h}(p)<1.25\right].
$$

We report $\delta_{1}$ and AbsRel as affine-aligned, percentile-clipped measures of agreement with the DA3 teacher. At each horizon, both metrics are averaged equally over clips.

AssA@50 is the association-accuracy component of HOTA [^35], evaluated at a mask-IoU threshold of 0.50. Within clip $c$ and cumulative prefix $1{:}h$, each matched detection $m\in\mathrm{TP}_{c,h}$ receives the per-match association score

$$
\displaystyle a(m)
$$
 
$$
\displaystyle=\frac{|\mathrm{TPA}(m)|}{|\mathrm{TPA}(m)|+|\mathrm{FPA}(m)|+|\mathrm{FNA}(m)|},
$$
$$
\displaystyle A_{h}
$$
 
$$
\displaystyle=\frac{\sum_{c}\sum_{m\in\mathrm{TP}_{c,h}}a(m)}{\sum_{c}|\mathrm{TP}_{c,h}|},\qquad\overline{A}=\frac{1}{7}\sum_{h=2}^{8}A_{h}.
$$

Here, $c$ indexes clips, and $\mathrm{TP}_{c,h}$ contains accepted matches pooled over the five evaluated classes. TPA, FPA, and FNA are association true positives, false positives, and false negatives for the predicted–reference identity pair containing $m$. This definition keeps identities clip-local and micro-averages matched detections across the split. We evaluate prefixes $h=2,\ldots,8$. The single-frame prefix provides no temporal association evidence and is therefore excluded. The reported RGB, segmentation, and depth scores are unweighted means over eight horizons, whereas AssA@50 is the unweighted mean over seven horizons. We report mIoU, $\delta_{1}$, AbsRel, and AssA@50 as percentages.

## 2 Additional Planning Results

### 2.1 NAVSIM-v1 Planning Results

Table 3 reports NAVSIM-v1 navtest performance under the official PDMS protocol [^7]. We take baseline values from published results for traditional E2E policies [^5] [^32] [^62] [^25] [^57] [^46] and VLA-based policies [^64] [^27] [^23] [^55] [^29]. The WAM-based baselines follow published reports [^24] [^59] [^51] [^52] [^22]. With a single front camera, SUV achieves 90.8 PDMS, matching ReCogDrive for the highest score among the listed learned methods.

<table><tbody><tr><td>Method</td><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Comf.<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>Human</td><td>100</td><td>100</td><td>100</td><td>99.9</td><td>87.5</td><td>94.8</td></tr><tr><td colspan="7">Traditional E2E policies</td></tr><tr><td>TransFuser</td><td>97.7</td><td>92.8</td><td>92.8</td><td>100</td><td>79.2</td><td>84.0</td></tr><tr><td>DiffusionDrive</td><td>98.2</td><td>96.2</td><td>94.7</td><td>100</td><td>82.2</td><td>88.1</td></tr><tr><td>World4Drive</td><td>97.4</td><td>94.3</td><td>92.8</td><td>100</td><td>79.9</td><td>85.1</td></tr><tr><td>Hydra-MDP++</td><td>97.6</td><td>96.0</td><td>93.1</td><td>100</td><td>80.4</td><td>86.6</td></tr><tr><td>SeerDrive</td><td>98.4</td><td>97.0</td><td>94.9</td><td>99.9</td><td>83.2</td><td>88.9</td></tr><tr><td>Drive-JEPA</td><td>98.7</td><td>96.2</td><td>100</td><td>95.5</td><td>82.9</td><td>89.0</td></tr><tr><td colspan="7">VLA-based policies</td></tr><tr><td>AutoVLA</td><td>98.4</td><td>95.6</td><td>98.0</td><td>99.9</td><td>81.9</td><td>89.1</td></tr><tr><td>DriveVLA-W0</td><td>98.7</td><td>96.2</td><td>95.5</td><td>100</td><td>82.2</td><td>88.4</td></tr><tr><td>SGDrive-IL</td><td>98.6</td><td>95.1</td><td>95.4</td><td>100</td><td>81.2</td><td>87.4</td></tr><tr><td>AutoDrive-P <sup>3</sup></td><td>99.1</td><td>97.4</td><td>96.5</td><td>100</td><td>84.8</td><td>90.6</td></tr><tr><td>ReCogDrive</td><td>97.9</td><td>97.3</td><td>94.9</td><td>100</td><td>87.3</td><td>90.8</td></tr><tr><td colspan="7">WAM-based policies</td></tr><tr><td>ImagiDrive</td><td>98.6</td><td>96.2</td><td>94.5</td><td>100</td><td>80.5</td><td>87.4</td></tr><tr><td>PWM</td><td>98.6</td><td>95.9</td><td>95.4</td><td>100</td><td>81.8</td><td>88.1</td></tr><tr><td>DriveLaW</td><td>99.0</td><td>97.1</td><td>96.7</td><td>100</td><td>81.3</td><td>89.1</td></tr><tr><td>EponaV2</td><td>98.6</td><td>97.9</td><td>95.7</td><td>100</td><td>84.8</td><td>90.4</td></tr><tr><td>Metis</td><td>98.3</td><td>97.1</td><td>94.7</td><td>100</td><td>83.4</td><td>89.1</td></tr><tr><td>Metis (Top 6)</td><td>98.5</td><td>97.5</td><td>95.1</td><td>100</td><td>84.0</td><td>89.7</td></tr><tr><td>SUV</td><td>99.1</td><td>97.8</td><td>96.7</td><td>100</td><td>84.6</td><td>90.8</td></tr></tbody></table>

Table 3: NAVSIM-v1 navtest planning under each method’s published sensor and candidate configuration. Boldface and underlining indicate the best and second-best learned results. Metis (Top 6) uses best-of-6 selection, while SUV outputs one trajectory.

### 2.2 Planning-Score Reproducibility and Scene-Level Variation

We repeated the NAVSIM-v1 and NAVSIM-v2 navtest evaluations six times with fixed checkpoints and inference seed 42. Every repetition returned the same leaderboard-scale scores to two decimal places: 90.84 PDMS on NAVSIM-v1 navtest and 91.01 EPDMS on NAVSIM-v2 navtest. These repetitions assess fixed-checkpoint numerical reproducibility.

Table 4 reports the empirical mean and sample standard deviation across scene-level scores using an $N-1$ denominator. Each navtest scene is one observation. Scene scores are multiplied by 100 before summarization, and the results are rounded to two decimal places. The standard deviation describes scene-level dispersion.

| Benchmark | Metric | $N$ | Mean | Scene SD |
| --- | --- | --- | --- | --- |
| NAVSIM-v1 navtest | PDMS | 12,146 | 90.84 | 17.84 |
| NAVSIM-v2 navtest | EPDMS | 12,146 | 91.01 | 18.21 |

Table 4: Scene-level variation in the primary planning scores. Metric definitions are given in Section 1.1.

### 2.3 Qualitative Results on navtest

Figures 2 and 3 show complementary behaviors on NAVSIM-v2 navtest. Figure 2 presents three high-curvature turns: a wide signalized intersection, a curved urban road with nearby traffic, and a wet-road turn. Each case pairs the current observation and bird’s-eye-view trajectories with jointly generated RGB, semantic segmentation, relative depth, and instance tracks at 0.5, 1.0, 2.0, 3.0, and 4.0 s. Across these turns, the generated streams retain the road geometry and nearby actors as the heading changes. Figure 3 presents a dense-traffic case in which the SUV trajectory follows the recorded human route but advances less amid surrounding traffic. The resulting EPDMS is 84.1, reflecting lower progress.

### 2.4 Qualitative Zero-Shot Transfer to In-House Data

We apply SUV without adaptation to two non-public in-house driving clips from a source domain outside the public benchmarks. Each clip contains a front-camera observation and eight future ego poses over 4 s. The clips were excluded from training and model selection and serve as qualitative examples of cross-domain transfer. Figure 1 shows both results. The nighttime and daytime examples differ in road geometry and surrounding traffic, while both predicted trajectories follow the recorded lateral evolution and forward progress. This alignment persists throughout the 4-s horizon rather than appearing only at the final waypoint.

![[Uncaptioned image]](https://arxiv.org/html/2608.03084v1/inhouse_2.png)

Figure 1: Qualitative zero-shot planning on two in-house clips. Each row shows one example. Green, red, and dashed gray denote the recorded human, SUV, and constant-velocity trajectories. The ADEs for the top and bottom examples are 0.08 and 0.15 m over eight waypoints spanning 4 s.

![[vis_supp_success.png|Refer to caption]]

Figure 2: High-curvature turns on NAVSIM-v2 navtest. Each case shows the current observation, bird’s-eye-view trajectories, and jointly generated RGB, semantic segmentation, relative depth, and instance tracks at five prediction horizons. Green and red denote the recorded human and SUV trajectories.

![[vis_supp_conservative.png|Refer to caption]]

Figure 3: Lower progress in dense urban traffic. The SUV trajectory follows the recorded human route but advances less over the 4-s horizon. The panels show jointly generated RGB, semantic segmentation, relative depth, and instance tracks. The case receives an EPDMS of 84.1, and the colors follow Figure 2.

## 3 Future-Scene Prediction

Figures 4 and 5 report horizon-wise values under Section 1.2. Figure 4 compares native generation (SUV native), which predicts the structured streams directly, with Generate-then-Perceive (RGB post-hoc), which applies SAM 3 and DA3 to generated RGB. Figure 5 compares multi-stream training (Unified) with RGB-only training under matched settings.

Figure 4: Horizon-wise teacher-agreement metrics for native generation (SUV native) and Generate-then-Perceive (RGB post-hoc). Semantic segmentation and relative depth cover 0.5 to 4.0 s, and instance tracks cover 1.0 to 4.0 s. Higher is better except for AbsRel.

Figure 5: Horizon-wise RGB metrics for multi-stream (Unified) and RGB-only training. The lower panels plot Unified minus RGB-only. Values above zero indicate higher scores for Unified.

All four structured metrics become less favorable with prediction horizon. The ordering in the main-paper averages holds at every horizon: native generation has higher $\delta_{1}$ and lower AbsRel, while Generate-then-Perceive has higher mIoU and AssA@50. Direct generation is therefore competitive without stream-specific visual prediction heads.

RGB PSNR and SSIM decrease with prediction horizon for both models. Multi-stream training is slightly higher at all eight horizons, with the largest gaps near 2 s. These small positive differences show no reduction in the reported RGB metrics but do not establish an improvement.

## 4 Implementation and Reproducibility

### 4.1 Structured Target Construction and Decoding

Frozen teachers convert each recorded future RGB clip into videos for semantic segmentation, relative depth, and instance tracks. The videos use the same $640\times 384$ camera grid and 2-Hz sampling rate. We denote the number of future video frames by $T$, with $T=8$ over 4 s for NAVSIM and $T=10$ over 5 s for WOD-E2E. These videos serve as training targets for both benchmarks and as evaluation references for the NAVSIM future-scene analysis.

#### Semantic segmentation.

We apply SAM 3 [^3] independently to each future frame. Its image processor directly resizes each frame to $1008\times 1008$ for inference. We use a confidence threshold of 0.5 and the 11 prompts listed in Table 5. For each prompt, we retain up to 32 detections in descending confidence order. Before applying the 128-detection frame cap, we place road detections first and preserve confidence order within the road and non-road groups. We resize masks to the target grid with nearest-neighbor interpolation.

The class map starts as background. Road detections form a fill layer and update only background pixels. A non-road detection can replace background or road but not another non-road label. We process non-road detections in descending confidence order, so the highest-confidence non-road mask takes priority where non-road masks overlap. The resulting class indices are rendered as uint8 RGB with the fixed palette in Table 5. Generated segmentation videos are decoded by mapping each pixel to the nearest fixed palette color.

| Class | RGB value |
| --- | --- |
| Background | $(0,0,0)$ |
| Road | $(96,96,96)$ |
| Car | $(0,0,142)$ |
| Truck | $(0,0,70)$ |
| Bus | $(0,60,100)$ |
| Pedestrian | $(220,20,60)$ |
| Bicycle | $(119,11,32)$ |
| Motorcycle | $(0,0,230)$ |
| Traffic light | $(250,170,30)$ |
| Traffic sign | $(220,220,0)$ |
| Traffic cone | $(255,80,0)$ |
| Barrier | $(102,102,156)$ |

Table 5: Semantic classes and fixed RGB values. Background denotes pixels not assigned to a prompt.

#### Relative depth.

We run DA3-LARGE [^33] jointly on all $T$ future frames using its default upper-bound resize, which limits the longest input side to 504 pixels. We then resize the relative depth maps bilinearly to the target grid. Let $D$ contain the finite, positive DA3 outputs in one clip, and let $Q_{p}(D)$ denote its $p$ -th quantile. We set

$$
\ell=Q_{0.01}(D),\qquad u=Q_{0.99}(D),
$$

and replace $u$ by $\ell+1$ if $u-\ell<10^{-6}$. Each finite, positive value $d$ is encoded as

$$
\bar{d}=\operatorname{clip}\!\left(\frac{d-\ell}{u-\ell},0,1\right),\qquad k=\operatorname{round}(255\bar{d}).
$$

Nonfinite or nonpositive values receive $\bar{d}=0$. The integer $k\in\{0,\ldots,255\}$ selects one entry from the 256-color Google Turbo lookup table. Shared clip-level bounds give every frame the same monotonic depth-to-LUT-index mapping. Values outside $[\ell,u]$ saturate at the endpoint colors.

At evaluation, each generated depth pixel is mapped to its nearest Turbo entry. If $\widehat{k}$ is the selected LUT index, the decoder returns $\widehat{\bar{d}}=\widehat{k}/255$. For NAVSIM future-scene evaluation, Equation 4 fits one affine map from these normalized values to the DA3-reference scale over all eight frames.

#### Instance tracks.

The instance stream encodes semantic class and clip-local identity in a single RGB video. We process the six class prompts sequentially in one $T$ -frame SAM 3 video session. Each prompt is introduced at the first future frame and propagated through the clip. Table 6 lists the prompts and their base colors. We threshold output probabilities at 0.5. Within each frame, masks are composited from largest to smallest, allowing smaller traffic participants to overwrite larger masks.

We encode class-local SAM 3 identity $j\in\{0,1,\ldots\}$ using a zero-indexed vector with components $\Delta_{0},\ldots,\Delta_{6}$:

$$
\displaystyle\boldsymbol{\Delta}
$$
 
$$
\displaystyle=(0,0.18,-0.18,0.34,-0.34,0.50,-0.50),
$$
$$
\displaystyle\delta_{j}
$$
 
$$
\displaystyle=\Delta_{j\bmod 7}.
$$

Its component-wise RGB value is

$$
\mathbf{p}_{c,j}=\begin{cases}\operatorname{round}\!\left(\mathbf{b}_{c}+(255\mathbf{1}-\mathbf{b}_{c})\delta_{j}\right),&\delta_{j}\geq 0,\\
\operatorname{round}\!\left(\mathbf{b}_{c}(1+\delta_{j})\right),&\delta_{j}<0,\end{cases}
$$

where $\mathbf{1}=(1,1,1)$, and each channel is clipped to $[0,255]$. The color stays fixed for the entire track. Background pixels are black.

During decoding, pixels with Euclidean RGB distance greater than 30 from black are treated as foreground and mapped to the nearest of the 42 class-brightness codes. We extract 8-connected components independently for each code and remove components smaller than 32 pixels. Within each code, one-to-one Hungarian assignment links components in adjacent frames. The cost equals one minus mask IoU plus centroid displacement normalized by the image diagonal, and the returned assignments define the track links. Unmatched current components initialize new clip-local tracks, while unmatched previous tracks terminate. Tracks are linked only across consecutive frames. For NAVSIM native-generation evaluation, this decoder is applied to both generated and reference instance videos. The instance stream encodes all six prompted classes, whereas AssA@50 uses the five-class evaluation scope defined in Section 1.2.

| Prompt | Base RGB value |
| --- | --- |
| Car | $(64,128,255)$ |
| Truck | $(0,200,255)$ |
| Bus | $(255,190,0)$ |
| Pedestrian | $(255,64,96)$ |
| Bicycle | $(80,220,100)$ |
| Motorcycle | $(220,90,255)$ |

Table 6: Instance-track base colors. Seven brightness offsets encode the class-local track ID modulo seven.

### 4.2 Benchmark-Specific Configurations

WOD-E2E uses recorded future front-camera frames as RGB targets. Its targets for semantic segmentation, relative depth, and instance tracks follow Section 4.1 and share the same ten timestamps. A separately trained WOD-E2E model jointly denoises the four future streams and action trajectory. Its action tokens attend to all latent tokens from every future stream.

| Setting | NAVSIM | WOD-E2E |
| --- | --- | --- |
| Observation frames | 4 frames, 2 Hz | 5 frames, 2 Hz |
| Future streams | 8 frames each, 2 Hz, 4 s | 10 frames each, 2 Hz, 5 s |
| Trajectory | 8 points, 2 Hz, 4 s | 20 points, 4 Hz, 5 s |
| Generated streams | RGB, segmentation, relative depth, instance tracks | RGB, segmentation, relative depth, instance tracks |
| Image size | $640\times 384$ | $640\times 384$ |
| Candidates | One trajectory | One trajectory |

Table 7: Temporal input and output configurations for NAVSIM and WOD-E2E.

The WOD-E2E preprocessing crops each front-camera frame to the $5{:}3$ target aspect ratio and then bilinearly resizes it to $640\times 384$. WOD-E2E evaluation uses the global-step-100,000 checkpoint with bfloat16 precision and 10 Euler steps on all 1,505 test frames.

### 4.3 Training and Inference

The video expert is initialized from Wan2.2-5B and paired with a separately parameterized action expert in a Mixture-of-Transformers architecture [^43] [^31]. Both experts contain 30 Transformer blocks and use 24 attention heads with a head dimension of 128, giving a shared 3,072-dimensional attention space. Their hidden-state dimensions are 3,072 and 1,024, and their feed-forward dimensions are 14,336 and 4,096, respectively. The action expert projects its 1,024-dimensional hidden states into the shared attention space for queries, keys, and values, then projects the attention output back to 1,024 dimensions. The video expert uses a $1\times 2\times 2$ spatiotemporal patch size.

#### Optimization and software.

Additional NAVSIM training details include a per-GPU batch size of 8, AdamW $(\beta_{1},\beta_{2})=(0.9,0.95)$, 5% linear warmup followed by cosine decay to 1% of the initial learning rate, and equal coefficients of 1 for the averaged four-stream video loss and the action loss. We select checkpoints using the NAVSIM validation pdm\_score. Rank $r$ seeds Python, NumPy, PyTorch, and CUDA with $42+r$; deterministic worker seeds derive from the process seed, worker index, and rank. The software environment specifies Python 3.10 or later, PyTorch 2.7.1 with CUDA 12.8, torchvision 0.22.1 with CUDA 12.8, Accelerate 1.12.0, DeepSpeed 0.18.5, Transformers 4.49.0, Hydra 1.3.2, and NumPy 1.26.4.

#### Trajectory coordinates.

For NAVSIM, the action target contains eight ego-frame waypoints $(x,y,\psi)$ at 0.5-s intervals. For a future global position $\mathbf{p}_{t+h}$ and heading $\psi_{t+h}$, we compute

$$
\displaystyle\begin{bmatrix}x_{t+h}\\
y_{t+h}\end{bmatrix}
$$
 
$$
\displaystyle=R(-\psi_{t})\left(\mathbf{p}_{t+h}-\mathbf{p}_{t}\right),
$$
$$
\displaystyle\psi^{\mathrm{rel}}_{t+h}
$$
 
$$
\displaystyle=\operatorname{wrap}_{[-\pi,\pi)}\left(\psi_{t+h}-\psi_{t}\right).
$$

Planar coordinates are measured in meters from the rear axle at the final observation frame. Heading is measured in radians relative to the ego orientation in that frame. The transform yields the NAVSIM ego-frame trajectory $\mathbf{a}$. In the main-paper notation, $\widetilde{\mathbf{a}}=\mathcal{N}_{a}(\mathbf{a})=\mathbf{a}$, and $\mathcal{N}_{a}^{-1}$ is also the identity map for this representation.

For WOD-E2E, the annotations provide 20 ego-frame positions at 0.25-s intervals. We augment them with headings to form 20 action targets $(x_{i},y_{i},\psi_{i})$. With $(x_{0},y_{0})=(0,0)$ at the current ego-frame origin, we set $\psi_{i}=\operatorname{atan2}(y_{i}-y_{i-1},x_{i}-x_{i-1})$. For displacements shorter than $10^{-4}$  m, we retain the most recent valid heading, initialized to zero, and unwrap the resulting angle sequence.

#### Shifted flow-time sampling and weighting.

We define the shift mapping

$$
\phi_{\kappa}(\rho)=\frac{\kappa\rho}{1+(\kappa-1)\rho},\qquad\kappa=5.
$$

During training, we independently sample

$$
\displaystyle\rho_{\mathrm{vid}},\rho_{\mathrm{act}}
$$
 
$$
\displaystyle\stackrel{{\scriptstyle\mathrm{i.i.d.}}}{{\sim}}\mathcal{U}(0,1),
$$
$$
\displaystyle\lambda_{g}
$$
 
$$
\displaystyle=\phi_{\kappa}(\rho_{g}),\quad g\in\{\mathrm{vid},\mathrm{act}\}.
$$

The four visual streams share $\lambda_{\mathrm{vid}}$, whereas the action group uses $\lambda_{\mathrm{act}}$. The implementation passes $N\lambda_{g}$ to the timestep embedding, with $N=1000$. The equations use normalized time $\lambda_{g}\in[0,1]$. We define

$$
\displaystyle q(\lambda)
$$
 
$$
\displaystyle=\exp\!\left[-2\left(\lambda-\tfrac{1}{2}\right)^{2}\right]-\exp\!\left(-\tfrac{1}{2}\right),
$$
$$
\displaystyle Z
$$
 
$$
\displaystyle=\frac{1}{N}\sum_{n=0}^{N-1}q\!\left(\phi_{\kappa}\!\left(1-\frac{n}{N}\right)\right).
$$

The scheduler weight is

$$
w(\lambda)=\frac{q(\lambda)}{Z+10^{-10}},
$$

which has approximately unit mean over the shifted grid.

#### Shifted Euler inference.

For $S$ solver steps, we use the common video-action time grid

$$
\tau_{s}=\phi_{\kappa}\!\left(1-\frac{s}{S}\right),\qquad s=0,\ldots,S,
$$

which satisfies $\tau_{0}=1$ and $\tau_{S}=0$. Each prediction group starts from Gaussian noise. We update each $r\in\mathcal{R}=\{\mathrm{rgb},\mathrm{seg},\mathrm{depth},\mathrm{track},\mathrm{act}\}$ by explicit Euler:

$$
\mathbf{y}_{\tau_{s+1}}^{r}=\mathbf{y}_{\tau_{s}}^{r}+(\tau_{s+1}-\tau_{s})\widehat{\mathbf{u}}_{\theta,s}^{r},\qquad s=0,\ldots,S-1,
$$

where $\widehat{\mathbf{u}}_{\theta,s}^{r}$ is evaluated from the current joint state with timestep embedding $N\tau_{s}$.

[^1]: Bi, H.; Tan, H.; Xie, S.; Wang, Z.; Huang, S.; Liu, H.; Zhao, R.; Feng, Y.; Xiang, C.; Rong, Y.; Zhao, H.; Liu, H.; Su, Z.; Ma, L.; Su, H.; and Zhu, J. 2026. Motus: A Unified Latent Action World Model. In *CVPR*.

[^2]: Cao, W.; Hallgarten, M.; Li, T.; Dauner, D.; Gu, X.; Wang, C.; Miron, Y.; Aiello, M.; Li, H.; Gilitschenski, I.; Ivanovic, B.; Pavone, M.; Geiger, A.; and Chitta, K. 2025. Pseudo-Simulation for Autonomous Driving. In *CoRL*.

[^3]: Carion, N.; Gustafson, L.; Hu, Y.-T.; Debnath, S.; Hu, R.; Suris Coll-Vinent, D.; Ryali, C.; Alwala, K. V.; Khedr, H.; Huang, A.; Lei, J.; Ma, T.; Guo, B.; Kalla, A.; Marks, M.; Greer, J.; Wang, M.; Sun, P.; Rädle, R.; Afouras, T.; Mavroudi, E.; Xu, K.; Wu, T.-H.; Zhou, Y.; Momeni, L.; Hazra, R.; Ding, S.; Vaze, S.; Porcher, F.; Li, F.; Li, S.; Kamath, A.; Cheng, H. K.; Dollár, P.; Ravi, N.; Saenko, K.; Zhang, P.; and Feichtenhofer, C. 2026. SAM 3: Segment Anything with Concepts. In *ICLR*.

[^4]: Chen, Y.; Wang, Y.; and Zhang, Z. 2025. DrivingGPT: Unifying Driving World Modeling and Planning with Multi-modal Autoregressive Transformers. In *ICCV*.

[^5]: Chitta, K.; Prakash, A.; Jaeger, B.; Yu, Z.; Renz, K.; and Geiger, A. 2023. TransFuser: Imitation With Transformer-Based Sensor Fusion for Autonomous Driving. *IEEE TPAMI*.

[^6]: Dang, C.; Ang, S.; Li, Y.; Tian, H.; Wang, J.; Li, G.; Ye, H.; Ma, J.; Chen, L.; and Wang, Y. 2026. DriveFine: Refining-Augmented Masked Diffusion VLA for Precise and Robust Driving. In *ECCV*.

[^7]: Dauner, D.; Hallgarten, M.; Li, T.; Weng, X.; Huang, Z.; Yang, Z.; Li, H.; Gilitschenski, I.; Ivanovic, B.; Pavone, M.; Geiger, A.; and Chitta, K. 2024. NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking. In *NeurIPS*.

[^8]: Gabeur, V.; Long, S.; Peng, S.; Voigtlaender, P.; Sun, S.; Bao, Y.; Truong, K.; Wang, Z.; Zhou, W.; Barron, J. T.; Genova, K.; Kannen, N.; Ben, S.; Li, Y.; Guo, M.; Yogin, S.; Gu, Y.; Chen, H.; Wang, O.; Xie, S.; Zhou, H.; He, K.; Funkhouser, T.; Alayrac, J.-B.; and Soricut, R. 2026. Image Generators are Generalist Vision Learners. arXiv:2604.20329.

[^9]: Gao, S.; Yang, J.; Chen, L.; Chitta, K.; Qiu, Y.; Geiger, A.; Zhang, J.; and Li, H. 2024. Vista: A Generalizable Driving World Model with High Fidelity and Versatile Controllability. In *NeurIPS*.

[^10]: Guo, J.; Li, Q.; Li, P.; Chen, Z.; Sun, N.; Su, Y.; Wang, H.; Zhang, Y.; Li, X.; and Liu, H. 2026. Unified 4D World Action Modeling from Video Priors with Asynchronous Denoising. arXiv:2604.26694.

[^11]: Han, X.; Li, J.; Deng, K.; Chen, Z.; Shi, X.; Wang, S.; Li, B.; Wang, L.; Xie, S.; You, X.; Quan, J.; Cai, Z.; Diao, H.; Liu, Z.; Yang, L.; Lin, D.; and Wang, Q. 2026. Vision as Unified Multimodal Generation. arXiv:2607.06560.

[^12]: Hong, Y.; Zhou, X.; Li, Y.; Zhou, X.; Liu, L.; Luo, Y.; Xu, S.; Yang, L.; and Song, Z. 2026. DriveFuture: Future-Aware Latent World Models for Autonomous Driving. arXiv:2605.09701.

[^13]: Hou, B.; Li, G.; Jia, J.; An, T.; Guo, X.; Leng, S.; Geng, H.; Ze, Y.; Harada, T.; Torr, P.; Mees, O.; Pollefeys, M.; Liu, Z.; Wu, J.; Abbeel, P.; Malik, J.; Du, Y.; and Yang, J. 2026. World Model for Robot Learning: A Comprehensive Survey. arXiv:2605.00080.

[^14]: Hu, A.; Russell, L.; Yeo, H.; Murez, Z.; Fedoseev, G.; Kendall, A.; Shotton, J.; and Corrado, G. 2023a. GAIA-1: A Generative World Model for Autonomous Driving. arXiv:2309.17080.

[^15]: Hu, Y.; Yang, J.; Chen, L.; Li, K.; Sima, C.; Zhu, X.; Chai, S.; Du, S.; Lin, T.; Wang, W.; Lu, L.; Jia, X.; Liu, Q.; Dai, J.; Qiao, Y.; and Li, H. 2023b. Planning-Oriented Autonomous Driving. In *CVPR*.

[^16]: Huang, W.; Zhang, S.; Huang, Q.; Wang, Z.; Mao, Z.; Chua, C.; Chen, Z.; Chen, L.; and Lv, C. 2026. AutoMoT: A Unified Vision-Language-Action Model with Asynchronous Mixture-of-Transformers for End-to-End Autonomous Driving. In *ICML*.

[^17]: Hwang, J.-J.; Xu, R.; Lin, H.; Hung, W.-C.; Ji, J.; Choi, K.; Huang, D.; He, T.; Covington, P.; Sapp, B.; Zhou, Y.; Guo, J.; Anguelov, D.; and Tan, M. 2025. EMMA: End-to-End Multimodal Model for Autonomous Driving. *TMLR*.

[^18]: Jia, F.; Liu, L.; Song, Z.; Jia, C.; Ye, H.; Hao, X.; and Chen, L. 2026. DriveWorld-VLA: Unified Latent-Space World Modeling with Vision-Language-Action for Autonomous Driving. In *ICML*.

[^19]: Jiang, A.; Gao, Y.; Wang, Y.; Sun, Z.; Wang, S.; Heng, Y.; Sun, H.; Tang, S.; Zhu, L.; Chai, J.; Wang, J.; Gu, Z.; Jiang, H.; and Sun, L. 2025. IRL-VLA: Training an Vision-Language-Action Policy via Reward World Model. arXiv:2508.06571.

[^20]: Jiang, B.; Chen, S.; Liao, B.; Zhang, X.; Yin, W.; Zhang, Q.; Huang, C.; Liu, W.; and Wang, X. 2024. Senna: Bridging Large Vision-Language Models and End-to-End Autonomous Driving. arXiv:2410.22313.

[^21]: Jiang, B.; Chen, S.; Xu, Q.; Liao, B.; Chen, J.; Zhou, H.; Zhang, Q.; Liu, W.; Huang, C.; and Wang, X. 2023. VAD: Vectorized Scene Representation for Efficient Autonomous Driving. In *ICCV*.

[^22]: Li, J.; Liu, Z.; Hu, D.; Wu, J.; Ma, Z.; Wu, W.; Han, C.; Hao, Z.; Liu, Z.; Zhan, K.; Deng, J.; Zhu, X.; and Zhang, L. 2026a. Metis: A Generalizable and Efficient World-Action Model for Autonomous Driving and Urban Navigation. arXiv:2606.15869.

[^23]: Li, J.; Wu, J.; Hu, D.; Huang, X.; Sun, B.; Hao, Z.; Lang, X.; Zhu, X.; and Zhang, L. 2026b. SGDrive: Scene-to-Goal Hierarchical World Cognition for Autonomous Driving. In *CVPR*.

[^24]: Li, J.; Zhang, B.; Jin, X.; Deng, J.; Zhu, X.; and Zhang, L. 2026c. ImagiDrive: A Unified Imagination-and-Planning Framework for Autonomous Driving. In *ICRA*.

[^25]: Li, K.; Li, Z.; Lan, S.; Xie, Y.; Zhang, Z.; Liu, J.; Wu, Z.; Yu, Z.; and Alvarez, J. M. 2025a. Hydra-MDP++: Advancing End-to-End Driving via Expert-Guided Hydra-Distillation. arXiv:2503.12820.

[^26]: Li, Y.; Fan, L.; He, J.; Wang, Y.; Chen, Y.; Zhang, Z.; and Tan, T. 2025b. Enhancing End-to-End Autonomous Driving with Latent World Model. In *ICLR*.

[^27]: Li, Y.; Shang, S.; Liu, W.; Zhan, B.; Wang, H.; Wang, Y.; Chen, Y.; Wang, X.; An, Y.; Tang, C.; Hou, L.; Fan, L.; and Zhang, Z. 2026d. DriveVLA-W0: World Models Amplify Data Scaling Law in Autonomous Driving. In *ICLR*.

[^28]: Li, Y.; Wei, X.; Cao, J.; Wang, H.; Chi, X.; Bai, C.; Sun, Q.; Li, J.; Zhang, X.; Jia, P.; Tang, J.; Han, S.; and Zhang, S. 2026e. WAM4D: Fast 4D World Action Model via Spatial Register Tokens. arXiv:2606.14048.

[^29]: Li, Y.; Xiong, K.; Guo, X.; Li, F.; Yan, S.; Xu, G.; Zhou, L.; Chen, L.; Sun, H.; Wang, B.; Ma, K.; Chen, G.; Ye, H.; Liu, W.; and Wang, X. 2026f. ReCogDrive: A Reinforced Cognitive Framework for End-to-End Autonomous Driving. In *ICLR*.

[^30]: Liang, D.; Zhang, D.; Zhou, X.; Tu, S.; Feng, T.; Li, X.; Zhang, Y.; Du, M.; Tan, X.; and Bai, X. 2026. UniFuture: A 4D Driving World Model for Future Generation and Perception. In *ICRA*.

[^31]: Liang, W.; Yu, L.; Luo, L.; Iyer, S.; Dong, N.; Zhou, C.; Ghosh, G.; Lewis, M.; Yih, W.-t.; Zettlemoyer, L.; and Lin, X. V. 2025. Mixture-of-Transformers: A Sparse and Scalable Architecture for Multi-Modal Foundation Models. *TMLR*.

[^32]: Liao, B.; Chen, S.; Yin, H.; Jiang, B.; Wang, C.; Yan, S.; Zhang, X.; Li, X.; Zhang, Y.; Zhang, Q.; and Wang, X. 2025. DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving. In *CVPR*.

[^33]: Lin, H.; Chen, S.; Liew, J. H.; Chen, D. Y.; Li, Z.; Zhao, Y.; Peng, S.; Guo, H.; Zhou, X.; Shi, G.; Feng, J.; and Kang, B. 2026. Depth Anything 3: Recovering the Visual Space from Any Views. In *ICLR*.

[^34]: Liu, L.; Jia, C.; Yu, G.; Song, Z.; Li, J.; Jia, F.; Wu, P.; Hao, X.; and Luo, Y. 2026. GuideFlow: Constraint-Guided Flow Matching for Planning in End-to-End Autonomous Driving. In *CVPR*.

[^35]: Luiten, J.; Osep, A.; Dendorfer, P.; Torr, P. H. S.; Geiger, A.; Leal-Taixé, L.; and Leibe, B. 2021. HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking. *IJCV*.

[^36]: Ma, F.; Peng, D.; Yue, W.; Cao, J.; Wang, B.; Zhang, Q.; and Ma, J. 2026. GeoSem-WAM: Geometry- and Semantic-Aware World Action Models. arXiv:2606.03188.

[^37]: NVIDIA; et al. 2025. Cosmos World Foundation Model Platform for Physical AI. arXiv:2501.03575.

[^38]: Rowe, L.; de Schaetzen, R.; Girgis, R.; Pal, C.; and Paull, L. 2025. Poutine: Vision-Language-Trajectory Pre-Training and Reinforcement Learning Post-Training Enable Robust End-to-End Autonomous Driving. arXiv:2506.11234.

[^39]: Shao, H.; Wang, L.; Zhou, Y.; Hu, Y.; Zong, Z.; Waslander, S. L.; Zhan, W.; and Li, H. 2026. LMGenDrive: Bridging Multimodal Understanding and Generative World Modeling for End-to-End Driving. arXiv:2604.08719.

[^40]: Sheng, Z.; Ye, X.; Luo, J.; Chen, S.; and Ren, L. 2026. ExploreVLA: Dense World Modeling and Exploration for End-to-End Autonomous Driving. In *ECCV*.

[^41]: Sun, W.; Lin, X.; Chen, K.; Pei, Z.; Li, X.; Shi, Y.; and Zheng, S. 2026. SparseDriveV2: Scoring is All You Need for End-to-End Autonomous Driving. arXiv:2603.29163.

[^42]: Sun, W.; Lin, X.; Shi, Y.; Zhang, C.; Wu, H.; and Zheng, S. 2025. SparseDrive: End-to-End Autonomous Driving via Sparse Scene Representation. In *ICRA*.

[^43]: Team Wan; et al. 2025. Wan: Open and Advanced Large-Scale Video Generative Models. arXiv:2503.20314.

[^44]: Tian, X.; Gu, J.; Li, B.; Liu, Y.; Wang, Y.; Zhao, Z.; Zhan, K.; Jia, P.; Lang, X.; and Zhao, H. 2025. DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models. In *CoRL*.

[^45]: Wang, D.; Song, Y.; He, Z.; Chen, K.; Pan, X.; Deng, L.; and Gu, W. 2025a. HMVLM: Multistage Reasoning-Enhanced Vision-Language Model for Long-Tailed Driving Scenarios. arXiv:2506.05883.

[^46]: Wang, L.; Yang, Z.; Bai, C.; Zhang, G.; Liu, X.; Zheng, X.; Long, X.-X.; Lu, C.-T.; and Lu, C. 2026a. Drive-JEPA: Video JEPA Meets Multimodal Trajectory Distillation for End-to-End Driving. arXiv:2601.22032.

[^47]: Wang, L.; Zhang, C.; Kabra, R.; Uijlings, J.; Waslander, S.; Zisserman, A.; Carreira, J.; He, K.; Andriluka, M.; Bazavan, E. G.; Zanfir, A.; and Sminchisescu, C. 2026b. Video Generation Models are General-Purpose Vision Learners. In *ECCV*.

[^48]: Wang, S.; Yu, Z.; Jiang, X.; Lan, S.; Shi, M.; Chang, N.; Kautz, J.; Li, Y.; and Alvarez, J. M. 2025b. OmniDrive: A Holistic Vision-Language Dataset for Autonomous Driving with Counterfactual Reasoning. In *CVPR*.

[^49]: Wang, X.; Zhu, Z.; Huang, G.; Chen, X.; Zhu, J.; and Lu, J. 2024a. DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving. In *ECCV*.

[^50]: Wang, Y.; He, J.; Fan, L.; Li, H.; Chen, Y.; and Zhang, Z. 2024b. Driving into the Future: Multiview Visual Forecasting and Planning with World Model for Autonomous Driving. In *CVPR*.

[^51]: Xia, T.; Li, Y.; Zhou, L.; Yao, J.; Xiong, K.; Sun, H.; Wang, B.; Ma, K.; Chen, G.; Ye, H.; Liu, W.; and Wang, X. 2026. DriveLaW: Unifying Planning and Video Generation in a Latent Driving World. In *CVPR*.

[^52]: Xu, J.; Zhong, Z.; Shu, Z.; Jia, M.; Li, M.; Bian, J.-W.; Zhang, Q.; Zhang, K.; Xie, J.; Yang, J.; and Yin, W. 2026a. EponaV2: Driving World Model with Comprehensive Future Reasoning. arXiv:2605.14696.

[^53]: Xu, R.; Lin, H.; Jeon, W.; Feng, H.; Zou, Y.; Sun, L.; Gorman, J.; Tolstaya, K.; Tang, S.; White, B.; Sapp, B.; Tan, M.; Hwang, J.-J.; and Anguelov, D. 2026b. WOD-E2E: Waymo Open Dataset for End-to-End Driving in Challenging Long-tail Scenarios. In *CVPR*.

[^54]: Yao, W.; Li, Z.; Lan, S.; Wang, Z.; Sun, X.; Alvarez, J. M.; and Wu, Z. 2026. DriveSuprim: Towards Precise Trajectory Selection for End-to-End Planning. In *AAAI*.

[^55]: Ye, Y.; Zhang, Z.; Lin, J.; Sun, S.; Peng, C.; and Gao, W. 2026. AutoDrive-P <sup>3</sup>: Unified Chain of Perception–Prediction–Planning Thought via Reinforcement Fine-Tuning. In *ICLR*.

[^56]: Yuan, T.; Dong, Z.; Liu, Y.; and Zhao, H. 2026. Fast-WAM: Do World Action Models Need Test-time Future Imagination? arXiv:2603.16666.

[^57]: Zhang, B.; Song, N.; Li, J.; Zhu, X.; Deng, J.; and Zhang, L. 2025. Future-Aware End-to-End Driving: Bidirectional Modeling of Trajectory Planning and Scene Evolution. In *NeurIPS*.

[^58]: Zhang, K.; Wang, J.; Gao, S.; Wu, C.; Cao, Y.; Han, S.; Ivanovic, B.; Liu, L.; Pavone, M.; Han, S.; Zhou, D.; and Xie, E. 2026. Fast-dDrive: Efficient Block-Diffusion VLM for Autonomous Driving. arXiv:2605.23163.

[^59]: Zhao, Z.; Fu, T.; Wang, Y.; Wang, L.; and Lu, H. 2025. From Forecasting to Planning: Policy World Model for Collaborative State-Action Prediction. In *NeurIPS*.

[^60]: Zhen, H.; Sun, Q.; Zhang, H.; Li, J.; Zhou, S.; Du, Y.; and Gan, C. 2025. Learning 4D Embodied World Models. In *ICCV*.

[^61]: Zheng, W.; Xia, Z.; Huang, Y.; Zuo, S.; Zhou, J.; and Lu, J. 2024. Doe-1: Closed-Loop Autonomous Driving with Large World Model. arXiv:2412.09627.

[^62]: Zheng, Y.; Yang, P.; Xing, Z.; Zhang, Q.; Zheng, Y.; Gao, Y.; Li, P.; Zhang, T.; Xia, Z.; Jia, P.; Lang, X.; and Zhao, D. 2025. World4Drive: End-to-End Autonomous Driving via Intention-Aware Physical Latent World Model. In *ICCV*.

[^63]: Zhou, Y.; Wang, X.; Shao, H.; Wang, L.; Zhao, G.; Shao, J.; Zhu, J.; Yu, T.; Zhu, Z.; Huang, G.; and Waslander, S. L. 2026. DriveDreamer-Policy: A Geometry-Grounded World-Action Model for Unified Generation and Planning. arXiv:2604.01765.

[^64]: Zhou, Z.; Cai, T.; Zhao, S.; Zhang, Y.; Huang, Z.; Zhou, B.; and Ma, J. 2025. AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning. In *NeurIPS*.

[^65]: Zou, J.; Chen, S.; Liao, B.; Zheng, Z.; Song, Y.; Zhang, L.; Zhang, Q.; Liu, W.; and Wang, X. 2025. DiffusionDriveV2: Reinforcement Learning-Constrained Truncated Diffusion Modeling in End-to-End Autonomous Driving. arXiv:2512.07745.
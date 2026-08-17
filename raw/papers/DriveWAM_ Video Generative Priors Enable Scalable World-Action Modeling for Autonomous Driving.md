---
title: "DriveWAM: Video Generative Priors Enable Scalable World-Action Modeling for Autonomous Driving"
source: "https://arxiv.org/html/2605.28544v1"
author:
published:
created: 2026-08-17
description:
tags:
  - "clippings"
---
Chen Shi  Jinrui Xu  Shaoshuai Shi  Kehua Sheng  Bo Zhang  Li Jiang The Chinese University of Hong Kong, Shenzhen  Voyager Research, Didi Chuxing Project Page: [https://chenshi3.github.io/drivewam.github.io/](https://chenshi3.github.io/drivewam.github.io/)

###### Abstract

Pretrained foundation models have become an important basis for end-to-end autonomous driving. In contrast to vision-language models pretrained primarily on static image-text pairs, video generative models capture temporal dynamics and motion priors that are naturally suited for driving. We present DriveWAM, a driving world-action model that adapts a pretrained video diffusion transformer into an autoregressive video-action policy. DriveWAM organizes video and action streams into a unified temporal token sequence and trains them under a joint flow-matching objective, preserving the pretrained video-generation architecture while adapting its large-scale video priors to action generation. To incorporate high-level scene understanding, we introduce scene-evolving driving guidance, where a frozen VLM produces chunk-specific semantic intent to guide video-action generation. To keep long-horizon rollout bounded, we further introduce selective KV memory, which maintains bounded modality-aware video and action memory pools through relevance-redundancy cache selection at inference time. Experiments on NAVSIM and the PhysicalAI-Autonomous-Vehicles benchmark show that DriveWAM achieves strong planning performance, and a data-scaling study from 4k to 100k driving clips further confirms the scaling potential of world-action modeling for end-to-end autonomous driving.

<sup>†</sup>

## 1 Introduction

Recent end-to-end autonomous driving systems increasingly leverage pretrained foundation models as policy backbones. A major line of work builds on vision-language-action (VLA) models [^5] [^16] [^24] [^26] [^37] [^52] [^10], transferring the semantic knowledge and instruction-following ability of large-scale VLMs [^45] [^2] [^30] [^4] [^1] [^42] to action generation. Such VLA-based policies are well suited to high-level scene understanding and semantic reasoning, but driving decisions also require temporally dense visual cues such as spatial layout, motion continuity, and how the scene may evolve in the near future. Since VLM backbones are pretrained primarily on image-text data rather than video dynamics, VLM-centric policies must acquire these temporal priors largely from downstream driving data.

Video generative models offer a complementary foundation. They are pretrained on large-scale videos to model object persistence, motion patterns, and scene evolution, making them naturally suited for dynamic decision problems. Recent VLA-based driving methods [^24] [^62] [^58] have begun to incorporate future image or video generation to improve spatio-temporal awareness, but visual generation is often used as an auxiliary signal or a modular component on top of a VLM-centric policy. In parallel, world-action (WA) models in robotics [^21] [^55] [^57] [^22] [^54] [^13] show that pretrained video foundation models can be adapted more directly for action prediction and planning.

Adapting this paradigm to autonomous driving, however, remains non-trivial. First, a video foundation model is pretrained for visual generation rather than ego-action control, so turning it into an autoregressive video-action policy requires preserving its future-generation prior while coupling it to continuous action prediction. Second, video foundation models capture near-future dynamics but lack high-level semantic planning, whereas the appropriate driving decision depends on route intent, right-of-way, and decision-relevant traffic participants. Third, deploying such autoregressive policies over long horizons requires persistent historical context, but full KV caching grows with horizon length and sliding-window caching may discard old yet critical evidence. Existing driving-oriented world-action methods [^11] [^3] [^59] often rely on separate planners, discrete video tokenizers, or customized generation architectures, leaving open how to directly adapt a modern video foundation model into a semantically guided and scalable end-to-end driving policy.

In this paper, we present DriveWAM, a driving world-action model that adapts a pretrained video foundation model into an end-to-end autonomous driving policy. DriveWAM uses a flow-matching video diffusion transformer as the policy core and formulates driving as autoregressive video-action generation. Given observed video-action history and ego state, the model first generates future video latents and then decodes ego actions conditioned on the generated future latent, realizing inverse-dynamics action generation. Both video and action streams share the same transformer and are trained under a joint flow-matching objective [^29], preserving the pretrained spatio-temporal generative prior while learning to convert imagined future world evolution into executable ego motion.

To supply the missing high-level driving semantics, DriveWAM introduces scene-evolving driving guidance. A frozen VLM uses only causally available context, including the latest observation, recent ego motion, and route command, and produces chunk-specific guidance for the next prediction horizon. This guidance is injected through temporally localized cross-attention, ensuring that each future video-action chunk receives its own semantic intent while preserving the causal structure of full-clip autoregressive training. Thus, the VLM acts as a semantic guide, while the video foundation model remains responsible for dense temporal prediction.

For long-horizon rollout, DriveWAM further introduces selective KV memory. Instead of storing all historical tokens or evicting tokens by age, DriveWAM maintains separate bounded memory pools for video and action KVs. Each pool is updated by a relevance-redundancy selection rule inspired by efficient video-generation caching [^33]: prediction-relevant tokens are retained, while redundant patterns are filtered out. This training-free memory provides a compact video-action history for autoregressive inference without changing the training objective.

We evaluate DriveWAM on NAVSIM [^9] and the large-scale PhysicalAI-Autonomous-Vehicles benchmark [^46]. DriveWAM achieves strong planning performance with an autoregressive world-action architecture. Beyond benchmark comparison, we conduct a data-scaling study over $4$ k, $20$ k, and $100$ k driving clips, where DriveWAM improves consistently as training data increases. These results suggest that semantically guided world-action modeling provides a scalable foundation for end-to-end autonomous driving. Our contributions are summarized as follows:

- We propose DriveWAM, a driving world-action model that adapts a pretrained video diffusion transformer into an autoregressive video-action policy under a joint flow-matching objective.
- We introduce scene-evolving driving guidance to supply high-level driving semantics, where a frozen VLM provides causally available chunk-specific intent that guides video-action generation through temporally localized cross-attention.
- We propose selective KV memory for bounded long-horizon rollout, maintaining modality-aware video and action memory pools through relevance-redundancy cache selection at inference time.
- Experiments on NAVSIM and PhysicalAI-Autonomous-Vehicles, together with a scaling study from $4$ k to $100$ k clips, demonstrate the effectiveness and scalability of DriveWAM.

## 2 Related Work

### 2.1 Vision-Language-Action Models in Autonomous Driving

Recent autonomous driving methods increasingly leverage the general knowledge and semantic reasoning capabilities of large vision-language models. Early efforts use LLMs or VLMs mainly as high-level reasoning modules [^49] [^39] [^17] [^38] [^44] [^34] [^43] [^15], producing scene descriptions, maneuver suggestions, command tokens, or coarse trajectories that are further consumed by downstream planners. More recent VLA methods [^52] [^63] [^26] move toward end-to-end action prediction by coupling VLM backbones with trajectory decoders or planning heads. DriveMoE [^52] introduces an MoE-based policy head on top of a VLM to route different driving situations to specialized action experts. AutoVLA [^63] discretizes continuous trajectories into action primitives and casts driving policy learning as autoregressive token prediction. ReCogDrive [^26] combines VLM-based reasoning with a diffusion trajectory planner and further aligns the policy through imitation learning and reinforcement learning.

Building on this line, a parallel set of works incorporates visual world modeling into the VLA pipeline. FSDrive [^58] introduces future visual prediction as a visual reasoning process, while DriveVLA-W0 [^24] and DriveDreamer-Policy [^62] augment VLM-based policies with generative world-model components [^61] [^35] [^41] [^53]. Although these designs improve the spatio-temporal awareness of VLA-based driving, their policy core remains VLM-centric, with visual generation serving as an auxiliary branch rather than the policy backbone. In contrast, DriveWAM inherits a pretrained video generative model as the policy core to jointly model future world evolution and ego actions, while leveraging VLM reasoning as complementary scene-evolving guidance for high-level semantic intent.

### 2.2 World-Action Models

The world-action paradigm reuses pretrained video generative models as the foundation for policy learning. Recent works in robotic manipulation [^21] [^55] [^57] [^22] [^13] have shown that large-scale video pretraining can transfer favorably to action generation, motivating its adoption in autonomous driving. WorldDrive [^11] transfers representations learned by a trajectory-aware driving world model to a downstream planner, bridging scene generation and planning but keeping planning as a separate module. VaViM/VaVAM [^3] formulates autonomous driving as autoregressive video modeling with discrete VQ-VAE tokens [^40] through a GPT-style transformer [^36], and extends the model with an action expert for trajectory prediction. Epona [^59] couples a spatiotemporal transformer with twin diffusion transformers for separate next-frame generation and ego-trajectory prediction. While these designs establish important baselines, they do not directly adopt a modern video foundation model as a unified video-action policy backbone, and thus cannot fully inherit the latest pretrained video priors. DriveWAM instead builds directly on a pretrained video diffusion transformer and adapts both video and action streams under a unified flow-matching objective.

Moreover, existing driving-oriented world-action methods mostly rely on simple navigation commands as high-level guidance, leaving rich scene-level semantic reasoning largely unexplored. DriveWAM addresses this by injecting chunk-specific VLM guidance through temporally localized cross-attention. Efficient memory is another requirement for autoregressive video-action policies during long-horizon rollout, but prior models either use a limited observation window [^21] [^57] or maintain a standard KV cache [^55] [^22] whose cost grows with the sequence length. Recent works on efficient autoregressive video generation explore sliding-window attention [^14] [^6] [^56], sparse attention [^48] [^51], and cache compression [^33] [^18]. DriveWAM adapts the relevance-redundancy criterion of FlowCache [^33] to maintain bounded video and action memory pools for long-horizon driving.

## 3 Method

We propose DriveWAM, a semantically guided world-action model that adapts a pretrained video foundation model into a unified backbone for future world evolution and ego-action generation in autonomous driving, complemented by guidance from a frozen VLM for scene-evolving driving semantics. Specifically, as shown in Figure 1, we first formulate driving as autoregressive video-action generation, where a pretrained video diffusion transformer predicts future video latents and ego actions under a joint flow-matching objective (Sec. 3.1). We then introduce scene-evolving guidance, using a frozen VLM to provide causally available chunk-level intent that steers the video-action generation process (Sec. 3.2). Finally, we present selective KV memory, which retains prediction-relevant and non-redundant video-action history for bounded long-horizon rollout (Sec. 3.3).

![[pipeline2.png|Refer to caption]]

Figure 1: Overview of DriveWAM, which adapts a pretrained video generation backbone into a unified video-action policy. Building on this backbone, DriveWAM uses a frozen VLM to provide chunk-specific scene-evolving guidance for high-level scene reasoning and introduces selective KV memory to preserve compact prediction-relevant history for long-horizon rollout.

### 3.1 Autoregressive Video-Action Generation

A driving clip contains synchronized streams of camera images, ego actions, and ego states. We divide the clip into $K$ consecutive chunks and then consider the driving task as the next-chunk generation. At decision step $k$, the model has observed the clip up to chunk $k$ and predicts the future video-action chunk $(x_{k+1},a_{k+1})$, where $x_{k+1}$ is the next video segment and $a_{k+1}$ is the corresponding ego action. The causally available conditions include the historical context $H_{k}$ (video and action tokens of all observed chunks up to $k$), the ego state $e_{k}$ at the end frame of chunk $k$ (*e.g.*, velocity, acceleration, and curvature), and a textual guidance $g_{k}$ for the predicted chunk.

#### Tokenization.

To jointly model video-action generation, we organize video and action chunks into a unified temporal token sequence while preserving their temporal order. Each observed video chunk is encoded by the pretrained VAE [^41], and each ego-action chunk, represented as normalized ego-frame translation and yaw increments, is embedded by an MLP action encoder $E_{a}$, as follows:

$$
z_{k}=\mathrm{VAE}(x_{k}),\qquad u_{k}=E_{a}(a_{k}),\qquad H_{k}=\{(z_{i},u_{i})\}_{i\leq k}.
$$

Here, $z_{k}\in\mathbb{R}^{N_{x}\times d_{z}}$ and $u_{k}\in\mathbb{R}^{N_{a}\times d}$ denote encoded video and action tokens, respectively. $N_{x}$ and $N_{a}$ are the numbers of tokens per chunk, $d_{z}$ is the VAE latent channel dimension, and $d$ is the transformer hidden dimension. In practice, the VAE latents $z_{k}$ are also mapped to dimension $d$ by the latent input embedding layer inherited from the pretrained video diffusion transformer, yielding a unified representation for video-action generation.

#### World-action flow.

DriveWAM adopts the autoregressive video-action generation scheme, which factors the driving task into future world modeling and inverse-dynamics action generation. Specifically, DriveWAM utilizes a pretrained flow-matching video diffusion transformer $T_{\omega}$ [^41] for predicting the next video chunk and action chunk. During training, we sample a flow timestep $\tau\in[0,1]$ along the rectified-flow path [^29] [^31], where $\tau=1$ is the Gaussian-noise endpoint and $\tau=0$ represents clean data. For the next video chunk, the clean latent $z_{k+1}$ is noised along the standard rectified-flow path, producing a query $z_{k+1,\tau}$ and target velocity $v^{z}_{k+1,\tau}$. The video branch predicts this velocity under the current driving context:

$$
\hat{v}^{z}_{k+1,\tau}=T_{\omega}(z_{k+1,\tau};H_{k},e_{k},g_{k},\tau).
$$

Here, $e_{k}$ is embedded by a lightweight MLP and injected through a separate ego-state cross-attention branch. Notably, this conditioning repurposes the video model as a policy prior, with the backbone retaining its native future-visual-prediction objective while the predicted future is shaped by driving history, ego state, and semantic intent.

Actions are generated by an inverse-dynamics flow on the same diffusion transformer. We perturb the next action chunk directly in the normalized action space and embed it with the MLP action encoder $E_{a}$ to obtain $u_{k+1,\tau}$. Conditioned on the future world latent and the current driving context, the shared transformer predicts the action velocity as:

$$
\hat{v}^{a}_{k+1,\tau}=D_{a}\!\left(T_{\omega}(u_{k+1,\tau};\tilde{z}_{k+1},H_{k},e_{k},g_{k},\tau)\right),
$$

where $D_{a}$ is an MLP action decoder. The conditioning latent $\tilde{z}_{k+1}$ is the clean future video latent $z_{k+1}$ during teacher-forced training and the generated latent $\hat{z}_{k+1}$ during inference. This design grounds action generation in the predicted world evolution, so the action decoder behaves as an inverse-dynamics readout of the predicted future rather than an independent trajectory head. We use noisy-history augmentation [^22] to reduce this train-test mismatch.

#### Training objective.

We train the video and action branches with a joint flow-matching objective:

$$
\mathcal{L}=\mathbb{E}_{k,\tau}\left[\left\|\hat{v}^{z}_{k+1,\tau}-v^{z}_{k+1,\tau}\right\|_{2}^{2}+\beta_{a}\left\|\hat{v}^{a}_{k+1,\tau}-v^{a}_{k+1,\tau}\right\|_{2}^{2}\right],
$$

where $\beta_{a}$ controls the balance between future world modeling and action generation. The video term preserves the pretrained spatio-temporal generative prior during policy adaptation, while the action term teaches the shared backbone to decode this prior into executable ego motion.

#### Full-clip training and autoregressive rollout.

During training, we process all chunks of a clip in a single forward pass for efficiency. The video-action tokens are arranged in temporal order and denoised in parallel under a causal teacher-forcing mask (Figure 2), which realizes the conditional dependencies in Eqs. 2 and 3 while preserving the causal pattern used during inference [^6] [^22] [^14]. At inference, DriveWAM rolls out one chunk at a time. Given history $H_{k}$, the model first samples the future video latent $\hat{z}_{k+1}$ and then samples the action chunk $\hat{a}_{k+1}$ conditioned on this generated future. When the next real observation becomes available, it is encoded and appended to the history to form $H_{k+1}$, keeping long-horizon rollout grounded in observed driving context.

### 3.2 Scene-Evolving Driving Guidance

The video foundation model provides dense dynamic priors for near-future scene evolution, but it lacks semantic planning ability. In driving, the appropriate future is determined not only by short-term dynamics but also by route intent, traffic participants, and other decision-level semantics. For example, at an intersection, multiple future evolutions may be visually plausible from the current observation, while the desired one depends on the high-level driving intent. However, existing world-action methods typically use a single clip-level text condition, applying the same semantic guidance to every chunk. DriveWAM instead introduces a frozen VLM as a scene-evolving semantic guide. At each decision step $k$, the VLM produces fresh guidance $g_{k}$ from the latest causally available context, so each future video-action chunk is conditioned on its own up-to-date semantic intent while the video model remains the policy backbone for dense temporal prediction.

#### Causal guidance generation.

At each decision step $k$, the frozen Qwen3-VL-8B [^2] receives only causally available information: the latest observation $x_{k}$, a recent ego trajectory $a_{k}$, and the route command $c_{k}$ for the upcoming horizon. It produces a concise guidance text as follows:

$$
g_{k}=\Phi_{\mathrm{VLM}}(x_{k},a_{k},c_{k}),
$$

which summarizes the current road context and provides ego behavior guidance for the upcoming horizon, such as proceeding, yielding, stopping, or merging. Since no observation from the target chunk is used, $g_{k}$ provides semantic intent for predicting $(x_{k+1},a_{k+1})$ without leaking future information. During training, guidance texts are precomputed and cached; during inference, the VLM is queried once per decision step and reused across all denoising steps, keeping the semantic condition aligned with the current prediction horizon.

#### Temporally localized guidance injection.

Scene-evolving guidance introduces a separate text condition $g_{k}$ at each decision step. Without an additional constraint, tokens of chunk $k+1$ could attend to guidance from other chunks, including future guidance from later decision steps, breaking causal consistency. We therefore apply an additional block-diagonal text mask, which allows video-action tokens of target chunk $k+1$ to attend only to the guidance tokens of $g_{k}$. This keeps semantic conditioning temporally localized and prevents cross-chunk leakage. The resulting attention pattern is illustrated in Figure 2.

![[kv_cache_vis.png|Refer to caption]]

Figure 2: Attention mask used during DriveWAM training. Colored entries indicate allowed attention; blank entries are masked.

### 3.3 Selective KV Memory for Long-Horizon Rollout

Autoregressive world-action rollout conditions on the historical context $H_{k}$ defined in Sec. 3.1, where $H_{k}$ denotes the causal video-action history up to step $k$. During inference, this abstract history is implemented as layer-wise KV caches that store the keys and values produced by previous video and action chunks, so the model can attend to past context without recomputing all historical tokens. A full-window cache preserves complete history but grows linearly with rollout length, while a sliding-window cache bounds the cost by evicting the oldest tokens under FIFO rules [^20] [^60]. However, age-based eviction is suboptimal for driving tasks: older tokens may remain decision-relevant, such as motion trend of a nearby vehicle or a briefly occluded pedestrian, whereas newer tokens may correspond to repeated static background. To keep long-horizon inference bounded without discarding useful context, DriveWAM adopts an inference-time, training-free selective KV memory inspired by FlowCache [^33], retaining a compact, prediction-relevant approximation of $H_{k}$ during rollout.

#### Modality-aware memory pools.

Video and action histories have different token densities and functional roles. Video tokens are numerous and encode scene context, while action tokens are compact and encode ego-motion history. A single global cache would therefore be dominated by visual tokens and may under-preserve motion context. DriveWAM decomposes $H_{k}$ into two bounded modality pools $H^{v}_{k}$ and $H^{a}_{k}$, with $|H^{v}_{k}|\leq B^{v}$ and $|H^{a}_{k}|\leq B^{a}$, where $B^{v}$ and $B^{a}$ are the video and action memory budgets. This modality-aware design keeps both scene evidence and ego-motion history available during long-horizon rollout.

#### Relevance-redundancy retention.

When a memory pool exceeds its budget, DriveWAM ranks cached tokens by both current relevance and memory complementarity. For modality $m\in\{v,a\}$, let $Q_{k}^{m}$ denote the current query tokens of modality $m$, and let $\mathbf{k}^{m}_{j}$ be the cached key of token $j$ in $H^{m}_{k}$. We measure relevance $\rho^{m}_{j}$ by the average attention mass assigned to token $j$ from current queries, and redundancy $\eta^{m}_{j}$ by its average similarity to other cached keys:

$$
\rho^{m}_{j}=\frac{1}{|Q_{k}^{m}|}\sum_{\mathbf{q}\in Q_{k}^{m}}\left[\mathrm{softmax}_{\ell\in{H}_{k}^{m}}\left(\frac{\mathbf{q}^{\top}\mathbf{k}^{m}_{\ell}}{\sqrt{d}}\right)\right]_{j},\qquad\eta^{m}_{j}=\mathrm{mean}_{\ell\neq j}\cos(\mathbf{k}^{m}_{j},\mathbf{k}^{m}_{\ell}),
$$

where $d$ is the transformer hidden dimension. The final retention score is:

$$
s^{m}_{j}=\lambda\rho^{m}_{j}-(1-\lambda)\eta^{m}_{j},
$$

where $\lambda\in[0,1]$ balances relevance and redundancy, and tokens with low scores are evicted. As shown in Figure 3, this criterion has a natural driving-oriented interpretation: repeated road surfaces, sky, buildings, and other static background regions tend to be filtered out, while prediction-relevant cues such as moving vehicles and lane geometry are more likely to be retained.

#### Inference procedure.

Selective KV memory is applied only at inference time and does not change the training objective or model parameters. During rollout, each transformer layer attends to the current chunk together with the bounded video and action memory pools. After chunk $k+1$ is processed, the existing memory is ranked by the retention score, and the lowest-scored historical tokens are evicted to make room for the newly generated KVs $\Delta H^{m}_{k+1}$. The modality pool is then updated as:

$$
H^{m}_{k+1}\leftarrow\mathrm{Top}_{B^{m}-|\Delta H^{m}_{k+1}|}\!\left(H^{m}_{k}\right)\cup\Delta H^{m}_{k+1},\qquad m\in\{v,a\}.
$$

Here $B^{m}$ denotes the memory budget for modality $m$. This training-free update keeps long-horizon inference bounded while retaining a compact approximation to full-history attention.

## 4 Experiments

Table 1: Comparison on NAVSIM v1. <sup>∗</sup>: results with imitation learning. $\dagger$: trained with multiple trajectory anchors from [^27]. MV: multi-view cameras; SV: single-view camera; L: LiDAR.

<table><tbody><tr><td>Method</td><td>Ref</td><td>Sensors</td><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>C. <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>Human</td><td>–</td><td>–</td><td>100</td><td>100</td><td>100</td><td>99.9</td><td>87.5</td><td>94.8</td></tr><tr><td>UniAD <sup><a href="#fn:12">12</a></sup></td><td>CVPR’23</td><td>MV</td><td>97.8</td><td>91.9</td><td>92.9</td><td>100.0</td><td>78.8</td><td>83.4</td></tr><tr><td>TransFuser <sup><a href="#fn:7">7</a></sup></td><td>TPAMI’23</td><td>MV & L</td><td>97.7</td><td>92.8</td><td>92.8</td><td>100.0</td><td>79.2</td><td>84.0</td></tr><tr><td>PARA-Drive <sup><a href="#fn:47">47</a></sup></td><td>CVPR’24</td><td>MV</td><td>97.9</td><td>92.4</td><td>93.0</td><td>99.8</td><td>79.3</td><td>84.0</td></tr><tr><td>LAW <sup><a href="#fn:23">23</a></sup></td><td>ICLR’25</td><td>SV</td><td>96.4</td><td>95.4</td><td>88.7</td><td>99.9</td><td>81.7</td><td>84.6</td></tr><tr><td>DiffusionDrive <sup><a href="#fn:28">28</a></sup></td><td>CVPR’25</td><td>MV & L</td><td>98.2</td><td>96.2</td><td>94.7</td><td>100.0</td><td>82.2</td><td>88.1</td></tr><tr><td>WoTE <sup><a href="#fn:25">25</a></sup></td><td>ICCV’25</td><td>MV & L</td><td>98.5</td><td>96.8</td><td>94.4</td><td>99.9</td><td>81.9</td><td>88.3</td></tr><tr><td colspan="9">VLA-based Methods</td></tr><tr><td>ReCogDrive <sup>∗</sup> <sup><a href="#fn:26">26</a></sup></td><td>ICLR’26</td><td>MV</td><td>98.1</td><td>94.7</td><td>94.2</td><td>100.0</td><td>80.9</td><td>86.5</td></tr><tr><td>DriveVLA-W0 <sup><a href="#fn:24">24</a></sup></td><td>ICLR’26</td><td>SV</td><td>98.7</td><td>96.2</td><td>95.5</td><td>100.0</td><td>82.2</td><td>88.4</td></tr><tr><td>AutoVLA <sup><a href="#fn:63">63</a></sup></td><td>NeurIPS’25</td><td>MV</td><td>98.4</td><td>95.6</td><td>98.0</td><td>99.9</td><td>81.9</td><td>89.1</td></tr><tr><td>DriveDreamer-Policy <sup><a href="#fn:62">62</a></sup></td><td>arXiv’26</td><td>MV</td><td>98.4</td><td>97.1</td><td>95.1</td><td>100.0</td><td>83.5</td><td>89.2</td></tr><tr><td>DriveVLA-W0 <math><semantics><mo>†</mo> <annotation>\dagger</annotation></semantics></math> <sup><a href="#fn:24">24</a></sup></td><td>ICLR’26</td><td>SV</td><td>98.7</td><td>99.1</td><td>95.3</td><td>99.3</td><td>83.3</td><td>90.2</td></tr><tr><td colspan="9">WA-based Methods</td></tr><tr><td>Epona <sup><a href="#fn:59">59</a></sup></td><td>ICCV’25</td><td>SV</td><td>97.9</td><td>95.1</td><td>93.8</td><td>99.9</td><td>80.4</td><td>86.2</td></tr><tr><td>WorldDrive <sup><a href="#fn:11">11</a></sup></td><td>arXiv’26</td><td>SV</td><td>98.4</td><td>95.8</td><td>95.2</td><td>99.8</td><td>83.3</td><td>89.0</td></tr><tr><td>DriveWAM</td><td>–</td><td>SV</td><td>98.3</td><td>98.1</td><td>95.2</td><td>100.0</td><td>84.3</td><td>90.1</td></tr></tbody></table>

Table 2: Comparison on our curated 1,000-clip test subset of PhysicalAI-Autonomous-Vehicles benchmark. # Params denotes the number of model parameters. SV: single-view camera. <sup>∗</sup>: evaluated using the released checkpoint, which only supports up to 3s prediction.

| Method | Source | Sensors | \# Params | ADE@3s $\downarrow$ | FDE@3s $\downarrow$ | ADE@4s $\downarrow$ | FDE@4s $\downarrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VaVAM <sup>∗</sup> [^3] | Valeo | SV | 1.3B | 2.31 | 4.32 | – | – |
| Alpamayo-1.5 [^46] | NVIDIA | SV | 10B | 0.80 | 2.31 | 1.44 | 4.18 |
| DriveWAM | – | SV | 5B + 8B | 0.47 | 1.35 | 0.83 | 2.47 |

Table 3: Ablation of scene-evolving (SE) driving guidance under different training data scales on the PhysicalAI-Autonomous-Vehicles benchmark. ✗: fixed global prompt as text conditioning.

| \# Clips | \# Iters | SE Guidance | ADE@4s $\downarrow$ | FDE@4s $\downarrow$ |
| --- | --- | --- | --- | --- |
| 4k | 50k | ✗ | 1.21 | 3.65 |
| 4k | 50k | ✓ | 1.01 | 2.95 |
| 20k | 50k | ✗ | 0.95 | 2.94 |
| 20k | 50k | ✓ | 0.94 | 2.65 |
| 100k | 50k | ✗ | 0.92 | 2.75 |
| 100k | 50k | ✓ | 0.83 | 2.47 |

Table 4: Ablation of video backbone initialization and joint video supervision. All models are trained on 100k clips for 50k iterations.

| Pretrained init. | Video sup. | ADE@4s $\downarrow$ | FDE@4s $\downarrow$ |
| --- | --- | --- | --- |
| ✗ | ✓ | 1.10 | 3.26 |
| ✓ | ✗ | 1.23 | 3.79 |
| ✓ | ✓ | 0.83 | 2.47 |

Table 5: Ablation of KV memory strategies. ADE/FDE are measured on 20s clips, while KV memory and GFLOPs are profiled under a 300s clip.

| KV memory | ADE@4s $\downarrow$ | FDE@4s $\downarrow$ | Mem. (GB) $\downarrow$ | GFLOPs $\downarrow$ |
| --- | --- | --- | --- | --- |
| Full | 0.83 | 2.47 | 3.07 | 17.37 |
| FIFO | 1.40 | 3.47 | 0.25 | 1.05 |
| Selective | 0.89 | 2.52 | 0.25 | 1.44 |

### 4.1 Datasets

NAVSIM [^9] is a standard end-to-end planning benchmark derived from OpenScene [^8] [^19], with 103k trainval samples and 12k test samples. Following the standard NAVSIM protocol, we report No at-fault Collisions (NC), Drivable Area Compliance (DAC), Time-To-Collision (TTC), Comfort (C.), Ego Progress (EP), and the overall Predictive Driver Model Score (PDMS).

PhysicalAI-Autonomous-Vehicles is a large-scale real-world driving benchmark released with Alpamayo-R1 [^46]. It contains approximately 1,700 hours of driving logs, organized into 306,152 clips of 20 seconds each, with 153,625 clips for training, 90,928 for validation, and 61,599 for testing. We use the front-view camera stream and ego-motion labels. To focus on non-trivial driving scenarios, we use a VLM to tag each clip with a scene description and filter out simple scenes. Finally, we select 100k clips from the training split, and construct a curated 1,000-clip test subset from the test split. Details of the filtering procedure are provided in Appendix A. We report Average Displacement Error (ADE) and Final Displacement Error (FDE) over 3-second and 4-second future trajectories.

### 4.2 Implementation Details

We build DriveWAM based on the code framework of [^22]. DriveWAM uses Wan2.2-TI2V-5B [^41] as the video backbone, initialized from the base checkpoint released by [^22]. Unless otherwise specified, we fine-tune the full video diffusion transformer together with the newly introduced action and ego-state modules. The action encoder $E_{a}$ and action decoder $D_{a}$ are implemented as MLPs with hidden dimension 3072, and the ego-state features are encoded by a separate MLP. The scene-evolving guidance is generated by a frozen Qwen3-VL-8B [^2], which is queried once per chunk. Details of the VLM prompt template are provided in Appendix B.

All models are trained at $256{\times}448$ resolution on 48 NVIDIA H20 GPUs. We use AdamW [^32] with $\beta=(0.9,0.95)$, weight decay 0.1, learning rate $1{\times}10^{-5}$, and per-device batch size 1. The action loss weight is set to $\beta_{a}=1.0$. DriveWAM uses a 4-second chunk for video-action generation. On NAVSIM, we train for 100k iterations and decay the learning rate by a factor of 0.5 at 50k, 70k, and 90k iterations. Each sample uses the current frame as the condition and predicts a 4-second future horizon at 1 Hz. Since NAVSIM provides a single future planning horizon per sample, this setting reduces to one chunk-level prediction. On the PhysicalAI-Autonomous-Vehicles benchmark, we train for 50k iterations. Each training sample is a 12-second segment randomly cropped from a 20-second clip. The video stream is downsampled to 1 Hz, while ego actions remain at 10 Hz.

For inference, following [^22], we use an Euler ODE solver with $3$ steps for video tokens and $10$ steps for action tokens. The video solver integrates the flow trajectory from $\tau=1$ to $\tau=0.6$, while the action solver integrates from $\tau=1$ to $\tau=0$. For selective KV memory, we follow FlowCache [^33] and set $\lambda=0.07$. The video and action cache capacities are set to 448 and 160 tokens, respectively.

### 4.3 Main Results

NAVSIM. We compare DriveWAM against state-of-the-art end-to-end planners on NAVSIM v1, including classical end-to-end pipelines [^12] [^7] [^47] [^23] [^28] [^25], VLA-based policies [^26] [^24] [^63] [^62], and WA-based methods [^59] [^11]. As shown in Table 1, DriveWAM achieves a PDMS of $90.1$ using only a single front-view camera, outperforming all competing methods under comparable training settings. We attribute this to the underlying video generative backbone, which provides effective spatio-temporal priors for modeling scene geometry, motion dynamics, and fine-grained action prediction.

PhysicalAI-Autonomous-Vehicles. We evaluate DriveWAM on the large-scale PhysicalAI-Autonomous-Vehicles benchmark, comparing against the WA-based VaVAM [^3], trained on approximately 1,700 hours of OpenDV [^50] driving data, and the VLA-based Alpamayo-1.5 [^46], trained on roughly 80,000 hours of data containing the PhysicalAI-Autonomous-Vehicles training set. For consistency, all methods use only the front-view camera input and output a single trajectory at inference. As reported in Table 2, DriveWAM achieves ADE/FDE of $0.47$ / $1.35$ at 3 seconds and $0.83$ / $2.47$ at 4 seconds, substantially outperforming both baselines.

Qualitative results. Figure 4 visualizes future scenes and ego trajectories jointly generated by DriveWAM. Additional qualitative examples are provided in Appendix D.

### 4.4 Ablation Study

We conduct ablation studies on the PhysicalAI-Autonomous-Vehicles benchmark to investigate the individual components of DriveWAM. Unless otherwise noted, all ablation models are trained on 100k clips for 50k iterations under the same optimization settings as in the main results.

Scene-evolving Driving Guidance. Table 3 studies the contribution of injecting chunk-specific VLM guidance. Replacing the global prompt with scene-evolving guidance consistently improves trajectory prediction at every training data scale, reducing ADE@4s from $1.21$ to $1.01$ with 4k clips and from $0.92$ to $0.83$ with 100k clips, while also yielding consistent reductions in FDE@4s. These results indicate that high-level scene reasoning provides a complementary semantic conditioning to the low-level WA backbone. We also observe that the benefit does not vanish as training data grows. Appendix B provides qualitative examples of guidance evolving with scene context and route intent.

![[result_vis.png|Refer to caption]]

Figure 4: Qualitative results on NAVSIM (left) and PhysicalAI-Autonomous-Vehicles benchmark (right). The predicted ego trajectories are consistent with the jointly generated future scenes.

Figure 5: Data scaling on PhysicalAI-Autonomous-Vehicles.

Data Scaling. We investigate the data scalability of DriveWAM by varying the training set size from 4k to 20k and 100k clips under a fixed 50k-iteration training procedure. As shown in Table 3 and Figure 5, both ADE@4s and FDE@4s improve significantly with more data, regardless of whether scene-evolving guidance is applied. This consistent scaling trend reflects the effectiveness of the video-action modeling as a scalable policy foundation, and suggests that DriveWAM has not yet saturated at the current data scale.

Video Foundation Model Adaptation. We ablate DriveWAM’s capability by removing the pretrained video-backbone initialization and the joint video flow-matching supervision. As reported in Table 5, training entirely from scratch removes the large-scale spatio-temporal priors inherited from video pretraining, and degrades ADE@4s/FDE@4s to $1.10$ / $3.26$. Initializing from the pretrained backbone but removing video supervision also performs poorly, yielding $1.23$ / $3.79$, suggesting that action-only adaptation fails to preserve the generative video priors needed for WA policy learning. The full configuration combines pretrained initialization with joint video-action flow-matching supervision and achieves the best performance.

Selective KV Memory. Table 5 compares three inference-time memory strategies for autoregressive rollout. Full KV caching retains the entire video-action history, while FIFO and our selective KV memory operate under the fixed-size cache budget. As shown in $1^{st}$ and $3^{rd}$ rows, selective KV memory largely closes the accuracy gap to full caching, achieving $0.89$ / $2.52$ ADE@4s/FDE@4s, while FIFO degrades substantially to $1.40$ / $3.47$. To examine the long-horizon overhead of the memory module, we further profile each strategy on a 300-second rollout, reporting KV memory summed over all DiT layers and attention GFLOPs of one causal self-attention layer. As presented in Table 5, full caching requires $3.07$ GB memory and $17.37$ GFLOPs per step, whereas selective KV memory reduces them to $0.25$ GB and $1.44$ GFLOPs, yielding over $12{\times}$ reductions.

## 5 Conclusion

We present DriveWAM, a unified world-action policy that adapts a pretrained video foundation model directly into an end-to-end driving policy. DriveWAM introduces scene-evolving driving guidance that injects chunk-specific semantic intent through temporally localized cross-attention, and selective KV memory that maintains modality-aware video and action memory pools via relevance-redundancy selection at inference time. Experiments on NAVSIM and the PhysicalAI-Autonomous-Vehicles benchmark show that DriveWAM achieves strong planning performance, and a data-scaling study from 4k to 100k clips further confirms its scalability.

## References

## Appendix A Dataset Curation

The PhysicalAI-Autonomous-Vehicles benchmark contains roughly 1,700 hours of driving organized into 306,152 20-second clips. To focus evaluation and training on non-trivial driving scenarios, we tag every clip with a frozen Qwen3-VL-8B [^2] and use the resulting tags to construct a 100k-clip training subset, and a curated 1,000-clip test subset with balanced coverage of rare and ordinary scenarios.

Scene tagging. For each clip, we uniformly sample 20 frames from the front-view stream and pass them to Qwen3-VL-8B with four structured prompts. Each prompt focuses on one facet of driving complexity:

- Scene attributes: weather (clear/rainy/snowy/foggy), lighting (day/dusk/night/tunnel transition/strong backlight), road type (urban/highway/ramp/intersection/etc.), traffic density, and ego behavior.
- Vulnerable road-user events: whether the scene contains pedestrian crossing, jaywalking, occluded pedestrian popout, child or elderly participants, cyclist conflict, or crowd.
- Vehicle interaction events: whether the scene contains cut-in, cut-out, sudden braking ahead, wrong-way vehicle, large-vehicle occlusion, emergency vehicle, door opening, or stopped/broken vehicle.
- Intersection and long-tail events: whether the scene contains unprotected left turn, roundabout, irregular intersection, traffic-police gesture, road debris, accident scene, construction, animal on road, water puddle, or railway crossing, together with the traffic-light state.

The four prompts are run sequentially on the same sampled frames and merged into a single per-clip record. We additionally compute a scalar interest score by summing rule-based weights over detected event tags. In practice, rare or safety-critical events receive larger weights, e.g., accident scenes ($5.0$), occluded pedestrian popouts ($4.0$), animals on the road ($3.5$), and traffic-police gestures ($3.0$), while frequent or lower-impact attributes receive smaller weights ($0.5$ – $1.5$). Figure 6 shows representative tagged clips with their detected attributes and interest scores.

Training subset. The training subset is curated from the tagged training split through a two-stage procedure. We first retain all clips with interest score no smaller than $2.0$, preserving rare-event and interaction-rich cases. We then uniformly sample 50% of the remaining lower-score clips, so that ordinary driving scenarios are still represented without dominating the training distribution. For the data-scaling study, we sample 20k and 4k subsets from this 100k subset.

Test subset. The test subset contains 1,000 clips and is constructed from the tagged test split to cover both long-tail and ordinary driving scenarios. We combine three sources:

- Rare-event clips: a tag is treated as rare if it appears in fewer than 1% of the test clips. For each rare tag, we select up to 30 top-scoring clips that contain it, covering events such as accident scenes, animals on the road, occluded pedestrian popouts, traffic-police gestures, and railway crossings.
- High-interest clips: clips above the 75th percentile of the interest-score distribution are grouped by weather, lighting, and road type. We assign an approximately equal quota to each group and select the highest-scoring clips within each group until the target size is reached.
- Common-scene clips: 200 clips uniformly sampled from below the high-interest threshold to serve as ordinary-driving controls.

The selected clips are merged to form the final 1,000-clip test set.

![[appendix_tag.png|Refer to caption]]

Figure 6: Representative scene tagging results for dataset curation. For each clip, the left panel shows Qwen3-VL-8B detected scene attributes, events, and the resulting interest score, while the right panel shows sampled front-view frames. High-score clips capture rare or interaction-rich scenarios, whereas low-score clips represent ordinary driving.

![[appendix_guidance.png|Refer to caption]]

Figure 7: Examples of scene-evolving VLM guidance. The guidance adapts to changing scene context and route intent, such as pedestrians, traffic lights, construction barriers.

## Appendix B VLM Guidance Details

This section details the pipeline that produces the chunk-specific guidance $g_{k}$ used in Sec. 3.2. The pipeline operates in two stages. First, we classify the route of each upcoming 4-second chunk from ground-truth ego pose, producing a route command. Second, we prompt a frozen Qwen3-VL-8B with the route command, the front-camera frame at the end of the latest chunk, and a BEV visualization of the ego trajectory from the previous 4-second chunk, asking it to produce a concise two-sentence guidance for the upcoming chunk. Figure 7 shows representative guidance examples, where the generated text evolves with the latest observation and route command.

#### Route command.

Each chunk is assigned a high-level route command from {straight, left, right}. As explicit route annotations are unavailable, we construct this coarse command from the route/ego-yaw change for labeling purposes. Specifically, this command is derived from the yaw change of the ego vehicle over the chunk. Let $R_{0}$ and $R_{1}$ denote the ego rotations at the beginning and end of a chunk. We compute the relative yaw from $R_{0}^{\top}R_{1}$ and assign the command as left if the yaw change is larger than $15^{\circ}$, right if it is smaller than $-15^{\circ}$, and straight otherwise. The command only specifies directional intent and does not contain future positions, velocities, distances, or trajectory coordinates.

#### Prompt template.

The prompt template used for chunk-level guidance generation is shown below. The route command and visual inputs are filled at runtime.

<svg id="A2.SS0.SSS0.Px2.p2.pic1" height="558.28" overflow="visible" version="1.1" viewBox="0 0 477.38 558.28" width="477.38"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="translate(0,558.28) matrix(1 0 0 -1 0 0)"><g style="--ltx-fill-color:#D2D2D2;" fill="#D2D2D2" fill-opacity="1.0"><path style="stroke:none" d="M 0 4.49 L 0 553.79 C 0 556.27 2.01 558.28 4.49 558.28 L 472.89 558.28 C 475.37 558.28 477.38 556.27 477.38 553.79 L 477.38 4.49 C 477.38 2.01 475.37 0 472.89 0 L 4.49 0 C 2.01 0 0 2.01 0 4.49 Z"></path></g><g style="--ltx-fill-color:#FBFBFB;" fill="#FBFBFB" fill-opacity="1.0"><path style="stroke:none" d="M 0.55 4.49 L 0.55 537 L 476.82 537 L 476.82 4.49 C 476.82 2.32 475.06 0.55 472.89 0.55 L 4.49 0.55 C 2.32 0.55 0.55 2.32 0.55 4.49 Z"></path></g><g style="--ltx-fill-color:#F0F0F0;" fill="#F0F0F0" fill-opacity="1.0"><path style="stroke:none" d="M 0.55 537.55 L 0.55 553.79 C 0.55 555.97 2.32 557.73 4.49 557.73 L 472.89 557.73 C 475.06 557.73 476.82 555.97 476.82 553.79 L 476.82 537.55 Z"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 11.41 544.18)"><foreignObject style="--ltx-fo-width:28.57em;--ltx-fo-height:0.69em;--ltx-fo-depth:0.19em;font-size:10pt;" height="12.3" overflow="visible" transform="matrix(1 0 0 -1 0 9.61)" width="395.33"><span id="A2.SS0.SSS0.Px2.p2.pic1.1" style="width:28.57em;"><span id="A2.SS0.SSS0.Px2.p2.pic1.1.1"><span id="A2.SS0.SSS0.Px2.p2.pic1.1.1.1" style="--ltx-fg-color:#000000;">Prompt template for VLM guidance</span></span> </span></foreignObject></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 11.41 13.83)"><foreignObject style="--ltx-fo-width:35.51em;--ltx-fo-height:37.03em;--ltx-fo-depth:0.18em;font-size:10pt;" height="514.74" overflow="visible" transform="matrix(1 0 0 -1 0 512.32)" width="491.36"><span id="A2.SS0.SSS0.Px2.p2.pic1.2" style="width:35.51em;"><span id="A2.SS0.SSS0.Px2.p2.pic1.2.1"><span id="A2.SS0.SSS0.Px2.p2.pic1.2.1.1" style="font-size:90%;--ltx-fg-color:#000000;">You are a navigation assistant for an autonomous-driving dataset. You generate short navigation guidance for an upcoming 4-second driving window. You do not see the future window; you only see the current road conditions, the recent ego trajectory when available, and a high-level route command.</span></span> <span id="A2.SS0.SSS0.Px2.p2.pic1.2.2"><span id="A2.SS0.SSS0.Px2.p2.pic1.2.2.1" style="font-size:90%;--ltx-fg-color:#000000;">Inputs.</span></span> <span id="A2.I1"><span id="A2.I1.i1" style="list-style-type:none;">• <span id="A2.I1.i1.p1"><span id="A2.I1.i1.p1.1"><span id="A2.I1.i1.p1.1.1" style="font-size:90%;--ltx-fg-color:#000000;">Route command for the upcoming window:</span> <span id="A2.I1.i1.p1.1.2" style="font-size:90%;--ltx-fg-color:#000000;">straight</span><span id="A2.I1.i1.p1.1.3" style="font-size:90%;--ltx-fg-color:#000000;">,</span> <span id="A2.I1.i1.p1.1.4" style="font-size:90%;--ltx-fg-color:#000000;">left</span><span id="A2.I1.i1.p1.1.5" style="font-size:90%;--ltx-fg-color:#000000;">, or</span> <span id="A2.I1.i1.p1.1.6" style="font-size:90%;--ltx-fg-color:#000000;">right</span><span id="A2.I1.i1.p1.1.7" style="font-size:90%;--ltx-fg-color:#000000;">. Treat it as an authoritative navigation instruction.</span></span></span></span> <span id="A2.I1.i2" style="list-style-type:none;padding-top:1.0pt;">• <span id="A2.I1.i2.p1"><span id="A2.I1.i2.p1.1"><span id="A2.I1.i2.p1.1.1" style="font-size:90%;--ltx-fg-color:#000000;">Latest causally available front-camera frame before the upcoming 4-second window, showing the current road conditions.</span></span></span></span> <span id="A2.I1.i3" style="list-style-type:none;padding-top:1.0pt;">• <span id="A2.I1.i3.p1"><span id="A2.I1.i3.p1.1"><span id="A2.I1.i3.p1.1.1" style="font-size:90%;--ltx-fg-color:#000000;">BEV trajectory map of the previous 4-second window when available, showing recent ego motion and speed.</span></span></span></span></span> <span id="A2.SS0.SSS0.Px2.p2.pic1.2.3"><span id="A2.SS0.SSS0.Px2.p2.pic1.2.3.1" style="font-size:90%;--ltx-fg-color:#000000;">Output format. <span id="A2.SS0.SSS0.Px2.p2.pic1.2.3.1.1">Output exactly two sentences, under 50 words in total. Do not use labels, bullets, markdown, or extra paragraphs. Use present tense only.</span></span></span> <span id="A2.SS0.SSS0.Px2.p2.pic1.2.4"><span id="A2.SS0.SSS0.Px2.p2.pic1.2.4.1" style="font-size:90%;--ltx-fg-color:#000000;">Sentence 1. <span id="A2.SS0.SSS0.Px2.p2.pic1.2.4.1.1">Describe the current road context visible in the provided frame, including road type, traffic participants, traffic-light state if visible, weather, and lighting.</span></span></span> <span id="A2.SS0.SSS0.Px2.p2.pic1.2.5"><span id="A2.SS0.SSS0.Px2.p2.pic1.2.5.1" style="font-size:90%;--ltx-fg-color:#000000;">Sentence 2. <span id="A2.SS0.SSS0.Px2.p2.pic1.2.5.1.1">Describe the ego navigation guidance for the upcoming window. Jointly reason from the road context, the previous BEV trajectory, and the route command. The direction must be consistent with the route command, and the caution level should reflect the traffic conditions. Capture required interactions such as yielding, waiting, or merging. Use qualitative language only; do not include numbers, units, distances, coordinates, or low-level trajectory values.</span></span></span></span></foreignObject></g></g></svg>

## Appendix C Efficiency Analysis

We analyze the per-chunk inference cost of DriveWAM and compare it against Alpamayo-1.5 [^46] on a single NVIDIA H20 GPU. As shown in Table 6, each inference pass consists of three stages: (1) VLM guidance generation, (2) video generation, and (3) action denoising.

VLM guidance. DriveWAM queries a frozen Qwen3-VL-8B once per 4-second chunk, taking $125$  ms with the default vLLM compilation. Because the guidance is generated at the chunk boundary rather than per frame, the cost is amortized over the entire chunk. Alpamayo-1.5 processes a substantially larger number of visual tokens per query, which accounts for its higher VLM latency of $570$  ms.

Video generation. DriveWAM generates a 4-second video clip using a 3-step Euler ODE solver over the video tokens, taking $372$  ms. Alpamayo-1.5 does not perform explicit video generation.

Action denoising. By default, DriveWAM uses 10 denoising steps for action tokens, taking $765$  ms. We find that reducing the steps from 10 to 5 incurs negligible change in trajectory metrics, while reducing action denoising time to $374$  ms. The 5-step variant (DriveWAM <sup>∗</sup>) brings the total per-chunk cost to approximately $871$  ms, comparable to Alpamayo-1.5’s $900$  ms, while additionally producing a jointly generated future video.

| Method | VLM (ms) | Video Gen (ms) | Action (ms) | ADE@4s $\downarrow$ | FDE@4s $\downarrow$ |
| --- | --- | --- | --- | --- | --- |
| Alpamayo-1.5 | 570 | — | 330 | 1.44 | 4.18 |
| DriveWAM (Ours) | 125 | 372 | 765 | 0.83 | 2.47 |
| DriveWAM <sup>∗</sup> (Ours) | 125 | 372 | 374 | 0.84 | 2.45 |

Table 6: Per-chunk inference cost and trajectory prediction accuracy on a single H20 GPU. <sup>∗</sup> indicates action denoising steps reduced from 10 to 5.

## Appendix D Additional Qualitative Results

![[more_results.png|Refer to caption]]

Figure 8: Qualitative results on NAVSIM (top two rows) and PhysicalAI-Autonomous-Vehicles (bottom two rows) benchmarks. Each row shows the predicted ego trajectory alongside the jointly generated future frames at T=1,2,3,4.

We present additional qualitative results to complement the main-paper visualization. Figure 8 shows representative examples from both NAVSIM and the PhysicalAI-Autonomous-Vehicles benchmark, spanning driving conditions and road layouts.

NAVSIM qualitative results. Each example shows a BEV map on the left, where the red trajectory is the DriveWAM prediction and the blue trajectory is the ground-truth. The yellow vehicle icon denotes the starting ego, the blue vehicle icon denotes the predicted ending ego, and the green vehicle icon denotes the ground-truth ending ego. In both cases, the predicted trajectory aligns closely with the ground truth despite the complexity of the surroundings, and the generated video maintains photometric and geometric consistency across the four future timesteps.

PhysicalAI-Autonomous-Vehicles qualitative results. Each example overlays the ground-truth and predicted ego trajectories on the current front-view frame. These results are consistent with the strong quantitative performance and further demonstrate that the joint video-action generation provides a coherent, physically plausible world model that supports accurate long-horizon planning across diverse real-world conditions.

[^1]: J. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds, et al. (2022) Flamingo: a visual language model for few-shot learning. In NeurIPS, Cited by: §1.

[^2]: S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge, W. Ge, Z. Guo, Q. Huang, J. Huang, F. Huang, B. Hui, S. Jiang, Z. Li, M. Li, M. Li, K. Li, Z. Lin, J. Lin, X. Liu, J. Liu, C. Liu, Y. Liu, D. Liu, S. Liu, D. Lu, R. Luo, C. Lv, R. Men, L. Meng, X. Ren, X. Ren, S. Song, Y. Sun, J. Tang, J. Tu, J. Wan, P. Wang, P. Wang, Q. Wang, Y. Wang, T. Xie, Y. Xu, H. Xu, J. Xu, Z. Yang, M. Yang, J. Yang, A. Yang, B. Yu, F. Zhang, H. Zhang, X. Zhang, B. Zheng, H. Zhong, J. Zhou, F. Zhou, J. Zhou, Y. Zhu, and K. Zhu (2025) Qwen3-vl technical report. arXiv preprint arXiv:2511.21631. Cited by: Appendix A, §1, §3.2, §4.2.

[^3]: F. Bartoccioni, E. Ramzi, V. Besnier, S. Venkataramanan, T. Vu, Y. Xu, L. Chambon, S. Gidaris, S. Odabas, D. Hurych, R. Marlet, A. Boulch, M. Chen, E. Zablocki, A. Bursuc, E. Valle, and M. Cord (2025) VaViM and vavam: autonomous driving through video generative modeling. arXiv preprint arXiv:2502.15672. Cited by: §1, §2.2, §4.3, Table 2.

[^4]: L. Beyer, A. Steiner, A. S. Pinto, A. Kolesnikov, X. Wang, D. Salz, M. Neumann, I. Alabdulmohsin, M. Tschannen, E. Bugliarello, et al. (2024) Paligemma: a versatile 3b vlm for transfer. arXiv preprint arXiv:2407.07726. Cited by: §1.

[^5]: K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, et al. (2024) $\pi_{0}$: a vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164. Cited by: §1.

[^6]: B. Chen, D. M. Monsó, Y. Du, M. Simchowitz, R. Tedrake, and V. Sitzmann (2024) Diffusion forcing: next-token prediction meets full-sequence diffusion. In NeurIPS, Cited by: §2.2, §3.1.

[^7]: K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger (2023) Transfuser: imitation with transformer-based sensor fusion for autonomous driving. TPAMI. Cited by: §4.3, Table 1.

[^8]: O. Contributors (2023) OpenScene: the largest up-to-date 3d occupancy prediction benchmark in autonomous driving. Note: [https://github.com/OpenDriveLab/OpenScene](https://github.com/OpenDriveLab/OpenScene) Cited by: §4.1.

[^9]: D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, et al. (2024) Navsim: data-driven non-reactive autonomous vehicle simulation and benchmarking. In NeurIPS, Cited by: §1, §4.1.

[^10]: X. Gao, Y. Wu, R. Wang, C. Liu, Y. Zhou, and Z. Tu (2025) Langcoop: collaborative driving with language. In CVPR, Cited by: §1.

[^11]: X. Gui, M. Zhang, T. Yan, W. Han, J. Gong, F. Tan, C. Xu, and J. Shen (2026) Bridging scene generation and planning: driving with world model via unifying vision and motion representation. arXiv preprint arXiv:2603.14948. Cited by: §1, §2.2, §4.3, Table 1.

[^12]: Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, et al. (2023) Planning-oriented autonomous driving. In CVPR, Cited by: §4.3, Table 1.

[^13]: Y. Hu, Y. Guo, P. Wang, X. Chen, Y. Wang, J. Zhang, K. Sreenath, C. Lu, and J. Chen (2025) Video prediction policy: a generalist robot policy with predictive visual representations. In ICML, Cited by: §1, §2.2.

[^14]: X. Huang, Z. Li, G. He, M. Zhou, and E. Shechtman (2025) Self forcing: bridging the train-test gap in autoregressive video diffusion. In NeurIPS, Cited by: §2.2, §3.1.

[^15]: J. Hwang, R. Xu, H. Lin, W. Hung, J. Ji, K. Choi, D. Huang, T. He, P. Covington, B. Sapp, et al. (2024) Emma: end-to-end multimodal model for autonomous driving. arXiv preprint arXiv:2410.23262. Cited by: §2.1.

[^16]: P. Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, et al. (2025) $\pi_{0.5}$: a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054. Cited by: §1.

[^17]: B. Jiang, S. Chen, B. Liao, X. Zhang, W. Yin, Q. Zhang, C. Huang, W. Liu, and X. Wang (2024) Senna: bridging large vision-language models and end-to-end autonomous driving. arXiv preprint arXiv:2410.22313. Cited by: §2.1.

[^18]: K. Kahatapitiya, H. Liu, S. He, D. Liu, M. Jia, C. Zhang, M. S. Ryoo, and T. Xie (2025) Adaptive caching for faster video generation with diffusion transformers. In ICCV, Cited by: §2.2.

[^19]: N. Karnchanachari, D. Geromichalos, K. S. Tan, N. Li, C. Eriksen, S. Yaghoubi, N. Mehdipour, G. Bernasconi, W. K. Fong, Y. Guo, et al. (2024) Towards learning-based planning: the nuplan benchmark for real-world autonomous driving. In ICRA, Cited by: §4.1.

[^20]: J. Kim, J. Kang, J. Choi, and B. Han (2024) Fifo-diffusion: generating infinite videos from text without training. NeurIPS. Cited by: §3.3.

[^21]: M. J. Kim, Y. Gao, T. Lin, Y. Lin, Y. Ge, G. Lam, P. Liang, S. Song, M. Liu, C. Finn, and J. Gu (2026) Cosmos policy: fine-tuning video models for visuomotor control and planning. In ICLR, Cited by: §1, §2.2, §2.2.

[^22]: L. Li, Q. Zhang, Y. Luo, S. Yang, R. Wang, F. Han, M. Yu, Z. Gao, N. Xue, X. Zhu, et al. (2026) Causal world modeling for robot control. arXiv preprint arXiv:2601.21998. Cited by: §1, §2.2, §2.2, §3.1, §3.1, §4.2, §4.2.

[^23]: Y. Li, L. Fan, J. He, Y. Wang, Y. Chen, Z. Zhang, and T. Tan (2025) Enhancing end-to-end autonomous driving with latent world model. In ICLR, Cited by: §4.3, Table 1.

[^24]: Y. Li, S. Shang, W. Liu, B. Zhan, H. Wang, Y. Wang, Y. Chen, X. Wang, AnYasong, C. Tang, L. Hou, L. Fan, and Z. Zhang (2026) DriveVLA-w0: world models amplify data scaling law in autonomous driving. In ICLR, Cited by: §1, §1, §2.1, §4.3, Table 1, Table 1.

[^25]: Y. Li, Y. Wang, Y. Liu, J. He, L. Fan, and Z. Zhang (2025) End-to-end driving with online trajectory evaluation via bev world model. In ICCV, Cited by: §4.3, Table 1.

[^26]: Y. Li, K. Xiong, X. Guo, F. Li, S. Yan, G. Xu, L. Zhou, L. Chen, H. Sun, B. Wang, et al. (2026) Recogdrive: a reinforced cognitive framework for end-to-end autonomous driving. In ICLR, Cited by: §1, §2.1, §4.3, Table 1.

[^27]: Z. Li, K. Li, S. Wang, S. Lan, Z. Yu, Y. Ji, Z. Li, Z. Zhu, J. Kautz, Z. Wu, et al. (2024) Hydra-mdp: end-to-end multimodal planning with multi-target hydra-distillation. arXiv preprint arXiv:2406.06978. Cited by: Table 1.

[^28]: B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang, et al. (2025) Diffusiondrive: truncated diffusion model for end-to-end autonomous driving. In CVPR, Cited by: §4.3, Table 1.

[^29]: Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le (2023) Flow matching for generative modeling. In ICLR, Cited by: §1, §3.1.

[^30]: H. Liu, C. Li, Q. Wu, and Y. J. Lee (2023) Visual instruction tuning. In NeurIPS, Cited by: §1.

[^31]: X. Liu, C. Gong, and qiang liu (2023) Flow straight and fast: learning to generate and transfer data with rectified flow. In ICLR, Cited by: §3.1.

[^32]: I. Loshchilov and F. Hutter (2019) Decoupled weight decay regularization. In ICLR, Cited by: §4.2.

[^33]: Y. Ma, X. Zheng, J. Xu, X. Xu, F. Ling, X. Zheng, H. Kuang, H. Li, X. Wang, X. Xiao, et al. (2026) Flow caching for autoregressive video generation. In ICLR, Cited by: §1, §2.2, §3.3, §4.2.

[^34]: J. Mao, J. Ye, Y. Qian, M. Pavone, and Y. Wang (2023) A language agent for autonomous driving. arXiv preprint arXiv:2311.10813. Cited by: §2.1.

[^35]: W. Peebles and S. Xie (2023) Scalable diffusion models with transformers. In ICCV, Cited by: §2.1.

[^36]: A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever, et al. (2019) Language models are unsupervised multitask learners. OpenAI blog. Cited by: §2.2.

[^37]: K. Renz, L. Chen, E. Arani, and O. Sinavski (2025) Simlingo: vision-only closed-loop autonomous driving with language-action alignment. In CVPR, Cited by: §1.

[^38]: C. Sima, K. Renz, K. Chitta, L. Chen, H. Zhang, C. Xie, J. Beißwenger, P. Luo, A. Geiger, and H. Li (2024) DriveLM: driving with graph visual question answering. In ECCV, Cited by: §2.1.

[^39]: X. Tian, J. Gu, B. Li, Y. Liu, Z. Zhao, Y. Wang, K. Zhan, P. Jia, X. Lang, and H. Zhao (2024) DriveVLM: the convergence of autonomous driving and large vision-language models. In CoRL, Cited by: §2.1.

[^40]: A. Van Den Oord O. Vinyals et al. (2017) Neural discrete representation learning. In NeurIPS, Cited by: §2.2.

[^41]: T. Wan, A. Wang, B. Ai, B. Wen, C. Mao, C. Xie, D. Chen, F. Yu, H. Zhao, J. Yang, et al. (2025) Wan: open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314. Cited by: §2.1, §3.1, §3.1, §4.2.

[^42]: P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu, J. Wang, W. Ge, et al. (2024) Qwen2-vl: enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191. Cited by: §1.

[^43]: S. Wang, Z. Yu, X. Jiang, S. Lan, M. Shi, N. Chang, J. Kautz, Y. Li, and J. M. Alvarez (2025) OmniDrive: a holistic vision-language dataset for autonomous driving with counterfactual reasoning. In CVPR, Cited by: §2.1.

[^44]: W. Wang, J. Xie, C. Hu, H. Zou, J. Fan, W. Tong, Y. Wen, S. Wu, H. Deng, Z. Li, et al. (2023) Drivemlm: aligning multi-modal large language models with behavioral planning states for autonomous driving. arXiv preprint arXiv:2312.09245. Cited by: §2.1.

[^45]: X. Wang, X. Zhang, Z. Luo, Q. Sun, Y. Cui, J. Wang, F. Zhang, Y. Wang, Z. Li, Q. Yu, et al. (2024) Emu3: next-token prediction is all you need. arXiv preprint arXiv:2409.18869. Cited by: §1.

[^46]: Y. Wang, W. Luo, J. Bai, Y. Cao, T. Che, K. Chen, Y. Chen, J. Diamond, Y. Ding, W. Ding, et al. (2025) Alpamayo-r1: bridging reasoning and action prediction for generalizable autonomous driving in the long tail. arXiv preprint arXiv:2511.00088. Cited by: Appendix C, §1, §4.1, §4.3, Table 2.

[^47]: X. Weng, B. Ivanovic, Y. Wang, Y. Wang, and M. Pavone (2024) Para-drive: parallelized architecture for real-time autonomous driving. In CVPR, Cited by: §4.3, Table 1.

[^48]: H. Xi, S. Yang, Y. Zhao, C. Xu, M. Li, X. Li, Y. Lin, H. Cai, J. Zhang, D. Li, J. Chen, I. Stoica, K. Keutzer, and S. Han (2025) Sparse video-gen: accelerating video diffusion transformers with spatial-temporal sparsity. In ICML, Cited by: §2.2.

[^49]: Z. Xu, Y. Zhang, E. Xie, Z. Zhao, Y. Guo, K. K. Wong, Z. Li, and H. Zhao (2024) Drivegpt4: interpretable end-to-end autonomous driving via large language model. IEEE Robotics and Automation Letters. Cited by: §2.1.

[^50]: J. Yang, S. Gao, Y. Qiu, L. Chen, T. Li, B. Dai, K. Chitta, P. Wu, J. Zeng, P. Luo, et al. (2024) Generalized predictive model for autonomous driving. In CVPR, Cited by: §4.3.

[^51]: S. Yang, H. Xi, Y. Zhao, M. Li, J. Zhang, H. Cai, Y. Lin, X. Li, C. Xu, K. Peng, J. Chen, S. Han, K. Keutzer, and I. Stoica (2025) Sparse videogen2: accelerate video generation with sparse attention via semantic-aware permutation. In NeurIPS, Cited by: §2.2.

[^52]: Z. Yang, Y. Chai, X. Jia, Q. Li, Y. Shao, X. Zhu, H. Su, and J. Yan (2026) DriveMoE: mixture-of-experts for vision-language-action model in end-to-end autonomous driving. In CVPR, Cited by: §1, §2.1.

[^53]: Z. Yang, J. Teng, W. Zheng, M. Ding, S. Huang, J. Xu, Y. Yang, W. Hong, X. Zhang, G. Feng, D. Yin, Yuxuan.Zhang, W. Wang, Y. Cheng, B. Xu, X. Gu, Y. Dong, and J. Tang (2025) CogVideoX: text-to-video diffusion models with an expert transformer. In ICLR, Cited by: §2.1.

[^54]: A. Ye, B. Wang, C. Ni, G. Huang, G. Zhao, H. Li, H. Li, J. Li, J. Lv, J. Liu, et al. (2026) GigaWorld-policy: an efficient action-centered world–action model. arXiv preprint arXiv:2603.17240. Cited by: §1.

[^55]: S. Ye, Y. Ge, K. Zheng, S. Gao, S. Yu, G. Kurian, S. Indupuru, Y. L. Tan, C. Zhu, J. Xiang, et al. (2026) World action models are zero-shot policies. arXiv preprint arXiv:2602.15922. Cited by: §1, §2.2, §2.2.

[^56]: T. Yin, Q. Zhang, R. Zhang, W. T. Freeman, F. Durand, E. Shechtman, and X. Huang (2025) From slow bidirectional to fast autoregressive video diffusion models. In CVPR, Cited by: §2.2.

[^57]: T. Yuan, Z. Dong, Y. Liu, and H. Zhao (2026) Fast-wam: do world action models need test-time future imagination?. arXiv preprint arXiv:2603.16666. Cited by: §1, §2.2, §2.2.

[^58]: S. Zeng, X. Chang, M. Xie, X. Liu, Y. Bai, Z. Pan, M. Xu, and X. Wei (2025) FutureSightDrive: thinking visually with spatio-temporal cot for autonomous driving. In NeurIPS, Cited by: §1, §2.1.

[^59]: K. Zhang, Z. Tang, X. Hu, X. Pan, X. Guo, Y. Liu, J. Huang, L. Yuan, Q. Zhang, X. Long, et al. (2025) Epona: autoregressive diffusion world model for autonomous driving. In ICCV, Cited by: §1, §2.2, §4.3, Table 1.

[^60]: C. Zheng, S. Li, J. Deng, Z. Wang, S. Chen, L. Xiao, Z. Chi, H. Lin, K. Chen, B. Wang, et al. (2026) X-world: controllable ego-centric multi-camera world models for scalable end-to-end driving. arXiv preprint arXiv:2603.19979. Cited by: §3.3.

[^61]: C. Zheng, L. T. Vuong, J. Cai, and D. Phung (2022) MoVQ: modulating quantized vectors for high-fidelity image generation. In NeurIPS, Cited by: §2.1.

[^62]: Y. Zhou, X. Wang, H. Shao, L. Wang, G. Zhao, J. Shao, J. Zhu, T. Yu, Z. Zhu, G. Huang, et al. (2026) DriveDreamer-policy: a geometry-grounded world-action model for unified generation and planning. arXiv preprint arXiv:2604.01765. Cited by: §1, §2.1, §4.3, Table 1.

[^63]: Z. Zhou, T. Cai, S. Z. Zhao, Y. Zhang, Z. Huang, B. Zhou, and J. Ma (2025) AutoVLA: a vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. In NeurIPS, Cited by: §2.1, §4.3, Table 1.
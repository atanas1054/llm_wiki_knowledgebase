---
title: "CoWorld-VLA: Thinking in a Multi-Expert World Model for Autonomous Driving"
source: "https://arxiv.org/html/2605.10426v3"
author:
published:
created: 2026-09-11
description:
tags:
  - "clippings"
---
Minqing Huang    Yujiao Xiang    Zihan Liang    Jiajie Huang    Jingqi Wang    Yuheng Zhou   Zhi Xu   Feiyang Tan   Hangning Zhou   Mu Yang   Gong Chen Affiliation: Southeast University   Tianjin University <sup>∗</sup> The authors contributed equally and are listed in no particular order.<sup>†</sup> Corresponding author: jingqi.wang1314@gmail.com Affiliation: Afari Intelligent Drive   University of Electronic Science and Technology of China Affiliation: Shanghai Jiao Tong University   Beijing University Of Posts and Telecommunications

###### Abstract

Vision-Language-Action (VLA) models have emerged as a promising paradigm for end-to-end autonomous driving. However, existing reasoning mechanisms still struggle to provide planning-oriented intermediate representations: textual Chain-of-Thought (CoT) fails to preserve continuous spatiotemporal structure, while latent world reasoning remains difficult to use as a direct condition for action generation. In this paper, we propose CoWorld-VLA, a multi-expert world reasoning framework for autonomous driving, where world representations serve as explicit conditions to guide action planning. CoWorld-VLA extracts complementary world information through multi-source supervision and encodes it into expert tokens within the VLA, thereby providing planner-accessible conditioning signals. Specifically, we construct four types of tokens: semantic interaction, geometric structure, dynamic evolution, and ego trajectory tokens, which respectively model interaction intent, spatial structure, future temporal dynamics, and behavioral goals. During action generation, CoWorld-VLA employs a diffusion-based hierarchical multi-expert fusion planner, which is coupled with scene context throughout the joint denoising process to generate continuous ego trajectories. Experiments on NAVSIM v1 demonstrate future-scene modeling capability and strong planning performance, including collision avoidance and trajectory accuracy. Ablation studies further validate the complementarity of expert tokens and their effectiveness as planning conditions for action generation. Code will be available at [https://github.com/AFARI-Research/CoWorld-VLA](https://github.com/AFARI-Research/CoWorld-VLA).

## 1 Introduction

In recent years, foundation models such as Large Language Models (LLMs) and Vision-Language Models (VLMs) have shown strong multimodal understanding and cross-modal generalization capabilities [^1] [^2] [^3], leading to their increasing adoption in robotics and autonomous driving [^4] [^5]. VLA models have thus emerged as an important paradigm for end-to-end autonomous driving, mapping multimodal observations and language instructions to vehicle actions [^6] [^7] [^8] [^9] [^10] [^11] [^12]. However, autonomous driving requires reasoning about traffic participants, road geometry, future scene evolution, and ego objectives [^13] [^14] [^15] [^16] [^17]. Existing VLA frameworks often lack explicit intermediate reasoning states, forcing the model to jointly perform scene understanding, future prediction, and trajectory planning within a single action-generation process, which limits performance in complex driving scenarios [^18] [^19] [^20]. Figure 1(a) illustrates existing VLA paradigms.

![[introduction.png|Refer to caption]]

Figure 1: Comparison of reasoning paradigms for VLA-based autonomous driving. (a) Direct action prediction maps multimodal inputs to actions without intermediate reasoning. (b) Textual CoT introduces language-based reasoning but may lose continuous spatio-temporal details. (c) Single-world latent reasoning relies on one implicit world representation, which may be incomplete or weakly coupled with actions. (d) CoWorld-VLA performs multi-expert world reasoning by organizing Latent CoT experts and using a fusion diffusion planner for trajectory generation.

Recent studies introduce Chain-of-Thought (CoT) into autonomous driving VLA models to improve reasoning in complex scenarios [^21] [^22] [^23]. However, most methods rely on explicit natural-language reasoning, which introduces inference overhead and struggles to preserve continuous spatial and motion information for control [^24] [^25] [^26] [^27]. This motivates intermediate representations that preserve spatiotemporal structure and can directly condition trajectory planning.

World models offer a promising alternative by modeling future environment evolution in latent space [^28] [^29]. They have been widely used for scene generation, dynamic prediction, and planning assistance in autonomous driving [^30] [^31] [^32] [^33]. However, planning depends on multiple forms of world knowledge, including semantic interactions, 3D structure, and dynamic evolution, which are difficult to capture with a single representation [^34] [^29] [^35] [^30] [^31] [^32] [^36] [^37] [^38] [^39]. Moreover, predicted world representations are usually used only as auxiliary supervision rather than explicit conditions for trajectory generation during inference [^40] [^41] [^42] [^43], limiting their effectiveness for planning.

These limitations point to a central bottleneck: current VLA systems lack a mechanism for turning complementary world knowledge into planning-oriented latent states. To address this, we propose CoWorld-VLA, a multi-expert world reasoning framework for autonomous driving (Figure 1(d)). CoWorld-VLA introduces four specialized expert tokens in the VLM latent space to form a planning-oriented Latent CoT: semantic interaction token for high-level intentions and interactions, guided by JEPA-style representations [^38] [^34], the geometric structure token for road layouts and spatial constraints, using VGGT features [^37], dynamic evolution token for future scene evolution, supervised by the generative world model Wan [^44], and ego-trajectory tokens that connect world reasoning with behavioral objectives through trajectory-level supervision. Together, these tokens condense complementary world knowledge into intermediate reasoning states, alleviating the incompleteness of a single representation. To make these expert tokens directly support action generation, we further design a diffusion-based hierarchical multi-expert fusion planner. It converts expert tokens into trajectory-generation conditions and progressively produces continuous ego trajectories through denoising. In this way, world knowledge from different sources participates in inference-time planning rather than remaining only an auxiliary training signal.

In summary, our main contributions are as follows:

- We propose CoWorld-VLA, a unified multi-expert latent world reasoning framework. CoWorld-VLA formulates intermediate reasoning as a multi-expert Latent CoT in the VLM latent space, modeling semantic interaction, 3D geometry, dynamic evolution, and ego trajectory through expert tokens.
- We introduce a diffusion-based hierarchical multi-expert fusion planner to bridge the gap between world modeling and action generation. The planner integrates expert tokens with scene context during joint denoising for continuous trajectory generation.
- Extensive experiments on trajectory planning, future scene generation, and ablation studies demonstrate the effectiveness of CoWorld-VLA. Our method demonstrates future-generation capability and strong planning performance on NAVSIM v1, while ablations validate the complementary effects of multi-expert tokens and Latent CoT.

## 2 Related work

##### Vision-language models for autonomous driving.

Early VLA studies formulate autonomous driving as language generation. Methods such as DriveGPT4 [^21] and DriveLM [^45] leverage LLMs and VQA for driving decision-making, but remain limited by text-based interfaces and cannot directly generate executable trajectories [^46]. Subsequent approaches, including DriveVLM [^8], Senna [^9], DiffVLA [^10], and VLP [^11], decouple high-level reasoning from low-level control, while methods such as GPT-Driver [^47], EMMA [^48], Orion [^12], and OmniDrive [^49] reformulate trajectory prediction as textual reasoning or generation. However, recent studies show that lengthy textual reasoning may increase inference latency and weaken critical visual information [^50] [^51] [^52] [^22] [^23]. To address these issues, recent work explores latent reasoning in continuous latent spaces [^24] [^25] [^26] [^27]. Representative methods include DriveMoE [^53], ReCogDrive [^50], and LaST-VLA [^54]. Nevertheless, existing latent reasoning approaches still lack sufficient physical and semantic constraints for structured planning representations.

##### World models for autonomous driving.

World models are widely used in autonomous driving to capture spatio-temporal dynamics and predict future scene evolution [^55] [^31] [^40]. Early works mainly focus on future scene generation, including videos, point clouds, and multimodal signals [^30] [^41] [^56] [^57] [^58]. More recent approaches jointly model environment evolution and driving policies [^59] [^28] [^34] [^29] [^60] [^35] [^61] [^30] [^62] [^63] [^18] [^6] [^64] [^32] [^33]. However, future scene prediction and trajectory planning are often handled by separate branches, limiting the planner’s ability to exploit learned scene-dynamics representations. To address this issue, methods such as Uni-World VLA [^65] and DriveLaW [^19] attempt to unify environment modeling and planning. Nevertheless, RGB-only world models still lack sufficient structural and physical understanding for robust decision-making [^36] [^66] [^67]. Recent advances in geometric structure modeling [^68] [^69] [^70] [^37], JEPA-based predictive learning [^38] [^39] [^34], and video generation models such as Wan [^44] provide complementary geometric, dynamic, and visual priors, but these knowledge sources are rarely aligned within a unified latent reasoning framework for autonomous driving.

## 3 Methodology

Figure 2 provides an overview of the proposed CoWorld-VLA framework. We first introduce the necessary preliminaries in Section 3.1. Section 3.2 presents the action-conditioned predictive world model for learning future scene dynamics. Section 3.3 describes multi-expert representation learning, which aligns VLM hidden states with semantic, geometric, visual-dynamic, and trajectory priors. Section 3.4 introduces the hierarchical multi-expert fusion planner for diffusion-based trajectory generation.

![[overview_new.png|Refer to caption]]

Figure 2: Overview of CoWorld-VLA. CoWorld-VLA follows a three-stage training pipeline: video-generator pre-training, multi-expert world-representation learning, and diffusion-based trajectory planning. It first learns future scene evolution from visual and textual conditions, then aligns VLM hidden states with semantic, geometric, visual-dynamic, and trajectory experts, and finally fuses these expert representations to generate world-consistent ego trajectories.

### 3.1 Preliminaries

We formulate end-to-end autonomous driving as a conditional action generation problem, where the model predicts the future ego trajectory $\mathbf{A}_{t+1:t+T}$ over a planning horizon of $T$ steps.

##### Input representation.

At each time step $t$, the model receives a front-view image $o_{t}\in\mathbb{R}^{3\times H\times W}$ and conditioning information $c_{t}=\{\kappa_{t},\{(\bar{x}_{i},\bar{y}_{i},\bar{\psi}_{i})\}_{i=t-n_{h}+1}^{t},v_{t},a_{t}\},$ where $\kappa_{t}$ is the navigation instruction, $\{(\bar{x}_{i},\bar{y}_{i},\bar{\psi}_{i})\}$ is the ego pose history, and $v_{t},a_{t}\in\mathbb{R}^{2}$ are ego velocity and acceleration.

##### Standard VLA formulation.

Existing VLA methods map multimodal inputs directly to future actions [^6] [^5] [^7] [^8] [^9] [^10] [^11] [^12]:

$$
p_{\theta}(\mathbf{A}_{t+1:t+T}\mid o_{t},c_{t}),
$$

which unifies perception and action generation but does not explicitly model scene structure, dynamics, or intermediate reasoning states.

##### Structured latent formulation.

To enable structured reasoning, we introduce multi-expert latent world representations $\mathcal{Z}=\{z_{\mathrm{sem}},z_{\mathrm{geo}},z_{\mathrm{dyn}},z_{\mathrm{traj}}\}$, encoding semantic interaction, geometric structure, dynamic evolution, and ego-trajectory priors. These form a Latent CoT bridging environment understanding and trajectory planning. Trajectory generation is then reformulated as

$$
p_{\theta}(\mathbf{A}_{t+1:t+T}\mid o_{t},c_{t},\mathcal{Z}).
$$

In CoWorld-VLA, $\mathcal{Z}$ is instantiated by the expert-token hidden states learned in Stage 2, where $z_{\mathrm{sem}}$, $z_{\mathrm{geo}}$, $z_{\mathrm{dyn}}$, and $z_{\mathrm{traj}}$ are represented by $H_{\mathrm{sem}}$, $H_{\mathrm{geo}}$, $H_{\mathrm{dyn}}$, and $H_{\mathrm{traj}}$, respectively.

### 3.2 Stage 1: Action-conditioned predictive world model

We train an action-conditioned world model in the Wan latent space to predict future latent evolution from historical frames $\mathbf{x}_{h}$ and ego intention, using future frames $\mathbf{x}_{f}$ as targets.

##### Latent world modeling.

Given a training video sequence $\mathbf{x}=[\mathbf{x}_{h},\mathbf{x}_{f}]$, we encode it using a frozen Wan VAE as

$$
\mathbf{z}=\mathcal{E}_{\mathrm{vae}}(\mathbf{x})=[\mathbf{z}_{h},\mathbf{z}_{f}],
$$

where $\mathbf{z}_{h}$ and $\mathbf{z}_{f}$ denote historical and future latents, respectively. We apply flow matching only to the future segment. Given a noise level $\sigma\in(0,1)$ and Gaussian noise $\boldsymbol{\epsilon}\sim\mathcal{N}(0,\mathbf{I})$, the perturbed future latent and velocity target are defined as

$$
\tilde{\mathbf{z}}_{f,\sigma}=(1-\sigma)\mathbf{z}_{f}+\sigma\boldsymbol{\epsilon},\qquad\mathbf{v}_{\mathrm{target}}=\boldsymbol{\epsilon}-\mathbf{z}_{f}.
$$

The historical latent $\mathbf{z}_{h}$ remains noise-free and serves as observed context.

##### Text-conditioned learning.

To condition future scene evolution on ego intention and motion, we construct a structured prompt and encode it with the frozen Wan text encoder:

$$
\mathcal{P}=[\mathrm{Scene}]\oplus[\mathrm{Speed}]\oplus[\mathrm{Navigation}]\oplus[\mathrm{Trajectory}],\qquad\mathbf{c}=\mathcal{E}_{\mathrm{text}}(\mathcal{P}).
$$

The world model learns the conditional distribution $p_{\theta}(\mathbf{z}_{f}\mid\mathbf{z}_{h},\mathbf{c})$. With $\tilde{\mathbf{z}}_{\sigma}=[\mathbf{z}_{h},\tilde{\mathbf{z}}_{f,\sigma}]$, the flow matching loss is computed only on future-token outputs:

$$
\mathcal{L}_{\mathrm{flow}}=\mathbb{E}_{\mathbf{z}_{h},\mathbf{z}_{f},\boldsymbol{\epsilon},\sigma,\mathbf{c}}\left[\left\|\mathcal{F}_{\theta}\left(\tilde{\mathbf{z}}_{\sigma},\mathbf{c},\sigma\right)_{f}-\left(\boldsymbol{\epsilon}-\mathbf{z}_{f}\right)\right\|_{2}^{2}\right],
$$

where $(\cdot)_{f}$ denotes the future segment. This objective focuses learning on future dynamics while using historical observations as deterministic context.

### 3.3 Stage 2: Multi-expert representation learning

To provide VLM latent reasoning with sufficient physical and semantic priors, we introduce a multi-expert world representation learning framework. As shown in Stage 2 of Figure 2, Qwen3-VL serves as the backbone, whose hidden states are aligned with semantic, geometric, visual-dynamic, and trajectory-level experts in a unified latent space.

##### Action representation generation.

Given the current image observation $o_{t}$ and the driving task prompt $c_{t}$, the Qwen3-VL visual encoder $V_{\mathrm{Qwen}}$ and tokenizer $T_{\mathrm{Qwen}}$ produce image and text embeddings $e_{\mathrm{img}}=V_{\mathrm{Qwen}}(o_{t})$ and $e_{\mathrm{txt}}=T_{\mathrm{Qwen}}(c_{t})$, respectively. In addition to the visual-textual input, we insert expert-specific action tokens to inject external physical priors into the VLM $\pi_{\theta}$ latent space:

$$
\{H_{\mathrm{ctx}},H_{\mathrm{sem}},H_{\mathrm{geo}},H_{\mathrm{dyn}},H_{\mathrm{traj}}\}=\pi_{\theta}\left(e_{\mathrm{img}},e_{\mathrm{txt}},t_{\mathrm{sem}},t_{\mathrm{geo}},t_{\mathrm{dyn}},t_{\mathrm{traj}}\right),
$$

where $H_{\mathrm{ctx}}$ denotes the contextual VLM hidden states; $H_{\mathrm{sem}}$, $H_{\mathrm{geo}}$, $H_{\mathrm{dyn}}$, and $H_{\mathrm{traj}}$ correspond to JEPA distillation, VGGT alignment, video model conditioning, and trajectory regression, respectively.

##### Multi-expert representation supervision.

We introduce three complementary expert branches to supervise VLM hidden states during training. The JEPA branch uses a frozen V-JEPA encoder to provide high-level semantic and predictive representations. The VGGT branch uses a frozen 3D foundation model to provide spatial and geometric priors. The Wan branch uses the pre-trained video generation world model to supervise visual-dynamic representations through future scene prediction. For JEPA and VGGT, expert features are extracted from future observations $o_{\mathrm{fut}}$ and pooled as supervision targets:

$$
Z_{\mathrm{sem}}=\operatorname{Pool}\left(E_{\mathrm{sem}}(o_{\mathrm{fut}})\right),\quad Z_{\mathrm{geo}}=\operatorname{Pool}\left(E_{\mathrm{geo}}(o_{\mathrm{fut}})\right).
$$

The corresponding VLM action-token representations are adapted into expert feature spaces and optimized by alignment losses:

$$
\mathcal{L}_{\mathrm{sem}}=\lambda_{\mathrm{l1}}\operatorname{SmoothL1}\left(\hat{Z}_{\mathrm{sem}},Z_{\mathrm{sem}}\right)+\lambda_{\mathrm{cos}}\left(1-\cos\left(\hat{Z}_{\mathrm{sem}},Z_{\mathrm{sem}}\right)\right),
$$
 
$$
\mathcal{L}_{\mathrm{geo}}=\operatorname{MSE}\left(\hat{Z}_{\mathrm{geo}},Z_{\mathrm{geo}}\right),
$$

where $\hat{Z}_{\mathrm{jepa}}$ and $\hat{Z}_{\mathrm{geo}}$ are adapted VLM representations aligned with JEPA and VGGT features $Z_{\mathrm{jepa}}$ and $Z_{\mathrm{geo}}$.

For the Wan branch, $H_{\mathrm{dyn}}$ serves as the conditioning signal for future scene generation, replacing the text-based condition adopted in Stage 1, and this branch is optimized by the flow matching objective:

$$
\hat{o}_{\mathrm{fut}}=W_{\psi}\left(o_{t},H_{\mathrm{dyn}}\right),\quad\mathcal{L}_{\mathrm{dyn}}=\mathcal{L}_{\mathrm{flow}}\left(\hat{o}_{\mathrm{fut}},o_{\mathrm{fut}};o_{t},H_{\mathrm{dyn}}\right).
$$

##### Trajectory prediction and joint optimization.

Constrained by multi-expert supervision, the trajectory-aware representation $H_{\mathrm{traj}}$ is used for planning-oriented regression. Specifically, the last token is fed into a lightweight MLP head to regress future $A$ trajectory waypoints, and the prediction is optimized with an MSE loss:

$$
\hat{A}_{t+1:t+T}=\Phi_{\mathrm{traj}}\left(H_{\mathrm{traj}}[-1]\right),\quad\mathcal{L}_{\mathrm{traj}}=\operatorname{MSE}\left(\hat{A}_{t+1:t+T},A_{t+1:t+T}\right),
$$

where $\hat{A}_{t+1:t+T}\in\mathbb{R}^{T\times 3}$ denotes the predicted future trajectory, $\Phi_{\mathrm{traj}}:\mathbb{R}^{D}\rightarrow\mathbb{R}^{T\times 3}$ denotes the trajectory prediction head, and $D$ denotes the hidden-space dimension of the VLM.

The overall training objective is formulated as the weighted sum of all branches:

$$
\mathcal{L}_{\mathrm{total}}=w_{\mathrm{dyn}}\mathcal{L}_{\mathrm{dyn}}+w_{\mathrm{sem}}\mathcal{L}_{\mathrm{sem}}+w_{\mathrm{geo}}\mathcal{L}_{\mathrm{geo}}+w_{\mathrm{traj}}\mathcal{L}_{\mathrm{traj}},
$$

where $w_{\mathrm{dyn}}$, $w_{\mathrm{sem}}$, $w_{\mathrm{geo}}$, and $w_{\mathrm{traj}}$ denote the loss weights of the Wan world model branch, JEPA representation distillation branch, VGGT geometric distillation branch, and trajectory prediction branch, respectively.

At inference time, CoWorld-VLA takes the current observation $o_{t}$, the driving task prompt $c_{t}$, and expert-specific action tokens as inputs, with no access to future information.

### 3.4 Stage 3: Hierarchical multi-expert fusion

Given the multi-expert tokens learned in Sec. 3.3, CoWorld-VLA generates continuous ego trajectories with a Hierarchical Multi-Expert Fusion (HMEF) planner. HMEF performs conditional denoising in the normalized action space, allowing expert tokens to directly guide trajectory generation.

HMEF takes scene tokens and expert action tokens as inputs. The scene tokens include scene context VLM tokens and current-frame JEPA and VGGT tokens extracted by the frozen encoders used in Stage 2, denoted as $H_{\mathrm{scene}}$. The expert action tokens are produced by the VLM:

$$
\mathcal{H}_{\mathrm{act}}=\{H_{\mathrm{sem}},H_{\mathrm{geo}},H_{\mathrm{dyn}},H_{\mathrm{traj}}\},
$$

corresponding to semantic, geometric, dynamic world-model, and trajectory-prior action cues.

##### Expert encoding.

All token groups are first projected into a shared hidden space. To reduce the cost of denoising over long scene-token sequences, HMEF compresses $H_{\mathrm{scene}}$ into a fixed number of latent context tokens $C$ using learnable Perceiver queries [^71]. For each expert branch, an expert-specific bidirectional Transformer encodes the action tokens and aligns them to the planning horizon. Tokens associated with the same future step are projected and averaged to obtain per-step expert features $F_{e,t}$.

##### Conditional action fusion and denoising.

The target trajectory $A=\{(x_{t},y_{t},\psi_{t})\}_{t=1}^{T}$ is normalized to $A^{\mathrm{norm}}$. During training, HMEF samples $\tau\in(0,1)$ from a logit-normal schedule and constructs the noisy action:

$$
A_{\tau}=(1-\tau)\epsilon+\tau A^{\mathrm{norm}},\quad\epsilon\sim\mathcal{N}(0,I).
$$

For each expert $e$ and future step $t$, HMEF forms an action token $R_{e,t}$ from the noisy action, timestep embedding $e_{\tau}$, ego-state information, and expert feature. The denoiser adopts a two-stream architecture with clean scene tokens $C$ and noisy expert-conditioned action tokens $R$. Each block applies joint self-attention between the two streams, followed by stream-specific feed-forward layers:

$$
X_{0}=\mathcal{D}_{\theta}(C,R,e_{\tau}).
$$

The action-stream output is decoded into clean trajectory predictions $\hat{A}_{e}$ for all experts and supervised by the normalized ground truth:

$$
\mathcal{L}_{\mathrm{diff}}=\frac{1}{N_{e}}\sum_{e=1}^{N_{e}}\left\|\hat{A}_{e}-A^{\mathrm{norm}}\right\|_{2}^{2}.
$$

##### Inference.

HMEF learns expert fusion weights $\alpha=\operatorname{softmax}(w)$ and optimizes the fused prediction together with the expert denoising objective:

$$
\bar{A}=\sum_{e=1}^{N_{e}}\alpha_{e}\hat{A}_{e},\quad\mathcal{L}_{\mathrm{act}}=\mathcal{L}_{\mathrm{diff}}+\lambda_{\mathrm{fusion}}\left\|\bar{A}-A^{\mathrm{norm}}\right\|_{2}^{2}.
$$

At inference time, HMEF starts from Gaussian noise, iteratively denoises the expert trajectories, fuses them with $\alpha$, and denormalizes the result to obtain the executable ego plan.

## 4 Experiments

### 4.1 Experimental settings

##### Datasets.

Following previous studies [^72] [^50] [^73] [^65] [^59], we train CoWorld-VLA on NuPlan [^13] and NAVSIM v1 [^14], evaluating future video generation and trajectory planning on NAVSIM v1. NuPlan contains approximately 1,200 hours of real-world driving data from four cities. NAVSIM v1, built upon OpenScene [^74], provides 120 hours of 2 Hz multi-view driving data for planning-oriented evaluation under challenging dynamic scenarios, with 1,192 training clips and 136 testing clips.

##### Metrics.

For video generation, we report FVD [^75] on NAVSIM v1. For trajectory planning, NAVSIM v1 adopts a non-reactive open-loop protocol with PDMS [^14] as the primary metric, combining safety constraints and driving quality, including no at-fault collision (NC), drivable area compliance (DAC), ego progress (EP), time-to-collision (TTC), and comfort (C).

##### Implementation details.

CoWorld-VLA consists of a video diffusion Transformer (Wan2.2-5B [^44]), a VLM (Qwen3-VL-2B [^1]), and an action expert network. We adopt a three-stage training strategy: video DiT pretraining on NuPlan for future video generation, VLM fine-tuning with multi-expert supervision on NAVSIM v1, and action expert training with the VLM frozen. During inference, 20 sampling steps are used for video generation and 10 for trajectory planning. $w_{\mathrm{dyn}}$ =1.0, $w_{\mathrm{sem}}$ =0.1, $w_{\mathrm{geo}}$ =0.1, and $w_{\mathrm{traj}}$ =1.0. More details are shown in the appendix.

Table 1: Performance comparison on the NAVSIM v1 navtest under the benchmark planning protocol. PDMS and its sub-metrics evaluate the overall driving capability. Best and second-best results are highlighted in bold and underlined, respectively. C and L denote Camera and LiDAR, respectively. <sup>†</sup> denotes results w/o reinforcement learning. <sup>¶</sup> denotes our method integrated with the ReCogDrive [^50] action expert.

<table><tbody><tr><td>Method</td><td>Ref</td><td>Sensors</td><td>Frames</td><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Comf.<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td colspan="10">End-to-End Methods</td></tr><tr><td>UniAD <sup><a href="#fn:16">16</a></sup></td><td>CVPR’23</td><td>C</td><td>N</td><td>97.8</td><td>91.9</td><td>92.9</td><td>100</td><td>78.8</td><td>83.4</td></tr><tr><td>Hydra-MDP <sup><a href="#fn:76">76</a></sup></td><td>arXiv’24</td><td>C & L</td><td>N</td><td>98.3</td><td>96.0</td><td>94.6</td><td>100</td><td>78.7</td><td>86.5</td></tr><tr><td>DiffusionDrive <sup><a href="#fn:77">77</a></sup></td><td>CVPR’25</td><td>C & L</td><td>N</td><td>98.2</td><td>96.2</td><td>94.7</td><td>100</td><td>82.2</td><td>88.1</td></tr><tr><td>TrajDiff <sup><a href="#fn:78">78</a></sup></td><td>arXiv’25</td><td>C & L</td><td>N</td><td>98.1</td><td>97.0</td><td>94.3</td><td>100</td><td>82.7</td><td>88.5</td></tr><tr><td colspan="10">World Model Methods</td></tr><tr><td>LAW <sup><a href="#fn:42">42</a></sup></td><td>ICLR’25</td><td>C</td><td>N</td><td>96.4</td><td>95.4</td><td>88.7</td><td>99.9</td><td>81.7</td><td>84.6</td></tr><tr><td>FSDrive <sup><a href="#fn:72">72</a></sup></td><td>NeurIPS’25</td><td>C</td><td>N</td><td>98.2</td><td>93.8</td><td>93.3</td><td>99.9</td><td>80.1</td><td>85.1</td></tr><tr><td>Epona <sup><a href="#fn:59">59</a></sup></td><td>ICCV’25</td><td>C</td><td>N</td><td>97.9</td><td>95.1</td><td>93.8</td><td>99.9</td><td>80.4</td><td>86.2</td></tr><tr><td>Resim <sup><a href="#fn:79">79</a></sup></td><td>NeurIPS’25</td><td>C</td><td>N</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>86.6</td></tr><tr><td>PWM <sup><a href="#fn:80">80</a></sup></td><td>NeurIPS’25</td><td>C</td><td>N</td><td>98.6</td><td>95.9</td><td>95.4</td><td>100</td><td>81.8</td><td>88.1</td></tr><tr><td>WoTE <sup><a href="#fn:43">43</a></sup></td><td>ICCV’25</td><td>C & L</td><td>N</td><td>98.5</td><td>96.8</td><td>94.9</td><td>99.9</td><td>81.9</td><td>88.3</td></tr><tr><td>ResWorld <sup><a href="#fn:81">81</a></sup></td><td>ICLR’26</td><td>C & L</td><td>N</td><td>98.9</td><td>96.5</td><td>95.6</td><td>100</td><td>83.1</td><td>89.0</td></tr><tr><td>WorldDrive <sup><a href="#fn:82">82</a></sup></td><td>arXiv’26</td><td>C</td><td>N</td><td>98.4</td><td>96.8</td><td>95.2</td><td>100</td><td>83.3</td><td>89.0</td></tr><tr><td>DriveLaW <sup><a href="#fn:19">19</a></sup></td><td>CVPR’26</td><td>C</td><td>N</td><td>99.0</td><td>97.1</td><td>96.7</td><td>100</td><td>81.3</td><td>89.1</td></tr><tr><td colspan="10">Vision-Language Model Methods</td></tr><tr><td>ReCogDrive <sup>†</sup> <sup><a href="#fn:50">50</a></sup></td><td>ICLR’26</td><td>C</td><td>1</td><td>98.1</td><td>94.7</td><td>94.2</td><td>100</td><td>80.9</td><td>86.5</td></tr><tr><td>DriveVLA-W0 <sup><a href="#fn:64">64</a></sup></td><td>ICLR’26</td><td>C</td><td>N</td><td>98.4</td><td>95.3</td><td>95.4</td><td>100</td><td>80.9</td><td>87.2</td></tr><tr><td>LaST-VLA <sup>†</sup> <sup><a href="#fn:54">54</a></sup></td><td>arXiv’26</td><td>C</td><td>1</td><td>98.7</td><td>95.4</td><td>95.7</td><td>100</td><td>80.5</td><td>87.3</td></tr><tr><td>SGDrive <sup>†</sup> <sup><a href="#fn:83">83</a></sup></td><td>CVPR’26</td><td>C</td><td>N</td><td>98.6</td><td>95.1</td><td>95.4</td><td>100</td><td>81.2</td><td>87.4</td></tr><tr><td>Uni-World VLA <sup><a href="#fn:65">65</a></sup></td><td>arXiv’26</td><td>C</td><td>N</td><td>98.7</td><td>96.7</td><td>96.1</td><td>100</td><td>83.2</td><td>89.4</td></tr><tr><td>CoWorld-VLA <sup>¶</sup></td><td>–</td><td>C</td><td>1</td><td>98.5</td><td>96.9</td><td>95.4</td><td>100</td><td>83.2</td><td>89.1</td></tr><tr><td>CoWorld-VLA (ours)</td><td>–</td><td>C</td><td>1</td><td>99.1</td><td>97.0</td><td>96.5</td><td>100</td><td>84.0</td><td>90.0</td></tr></tbody></table>

### 4.2 Main results

##### Results on NAVSIM.

Table 1 presents planning results on NAVSIM v1 under its benchmark evaluation protocol. Under the single-frame front-camera-only setting, CoWorld-VLA achieves a PDMS of 90.0, remaining competitive with VLA-based planners such as SGDrive and Uni-World VLA, as well as world-model-based approaches including ResWorld and DriveLaW. These results support the effectiveness of world-model-guided latent representations for planning without multi-frame or LiDAR inputs.

For individual metrics, CoWorld-VLA achieves NC and EP scores of 99.1 and 84.0, respectively, indicating strong collision avoidance while maintaining forward progress. Its competitive DAC and TTC scores further suggest that the planned trajectories remain feasible and safety-aware. The results demonstrate the effectiveness of world-model-guided latent planning.

##### Results on NAVSIM v2 with EPDMS.

As a complement to the NAVSIM v1 PDMS results, performance is additionally evaluated on the separate NAVSIM v2 navtest protocol using the extended PDMS metrics. Table 2 reports the corresponding comparison.

Table 2: Results on NAVSIM v2 navtest with extended PDMS metrics. EPDMS <sup>∗</sup> is computed before the benchmark bug fix and EPDMS after it; “–” indicates unavailable results.

<table><tbody><tr><th>Method</th><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DDC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TL <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>LK <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>HC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EPDMS <math><semantics><mrow><msup><mo>∗</mo></msup> <mo>↑</mo></mrow> <annotation>{}^{*}\uparrow</annotation></semantics></math></td><td>EPDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><th colspan="12">End-to-End Methods</th></tr><tr><th>TransFuser</th><td>96.9</td><td>89.9</td><td>97.8</td><td>99.7</td><td>87.1</td><td>95.4</td><td>92.7</td><td>98.3</td><td>87.2</td><td>76.7</td><td>–</td></tr><tr><th>DiffusionDrive</th><td>98.2</td><td>95.9</td><td>99.4</td><td>99.8</td><td>87.5</td><td>97.3</td><td>96.8</td><td>98.3</td><td>87.7</td><td>–</td><td>84.5</td></tr><tr><th>DriveSuprim</th><td>97.8</td><td>97.9</td><td>99.5</td><td>99.9</td><td>90.6</td><td>97.1</td><td>96.6</td><td>98.3</td><td>77.9</td><td>86.0</td><td>–</td></tr><tr><th>DiffusionDriveV2</th><td>97.7</td><td>96.6</td><td>99.2</td><td>99.8</td><td>88.9</td><td>97.2</td><td>96.0</td><td>97.8</td><td>91.0</td><td>85.5</td><td>87.5</td></tr><tr><th colspan="12">World Model Methods</th></tr><tr><th>WoTE</th><td>98.5</td><td>96.8</td><td>98.8</td><td>99.8</td><td>86.1</td><td>97.9</td><td>95.5</td><td>98.3</td><td>82.9</td><td>–</td><td>87.7</td></tr><tr><th>DreamerAD</th><td>98.0</td><td>97.2</td><td>99.5</td><td>99.8</td><td>87.8</td><td>97.4</td><td>97.5</td><td>98.3</td><td>72.4</td><td>–</td><td>87.7</td></tr><tr><th>PWM</th><td>98.8</td><td>95.9</td><td>99.4</td><td>99.9</td><td>86.4</td><td>98.4</td><td>97.6</td><td>98.3</td><td>85.3</td><td>–</td><td>88.2</td></tr><tr><th>DriveLaW</th><td>98.7</td><td>96.9</td><td>99.6</td><td>99.8</td><td>87.5</td><td>98.3</td><td>97.6</td><td>98.4</td><td>77.4</td><td>–</td><td>88.6</td></tr><tr><th>Drive-JEPA (R34)</th><td>98.8</td><td>97.4</td><td>99.0</td><td>99.8</td><td>83.5</td><td>98.0</td><td>96.2</td><td>98.1</td><td>85.6</td><td>85.4</td><td>–</td></tr><tr><th>Latent-WAM</th><td>98.1</td><td>97.3</td><td>99.6</td><td>99.8</td><td>87.7</td><td>97.3</td><td>97.6</td><td>98.1</td><td>87.3</td><td>–</td><td>89.3</td></tr><tr><th colspan="12">Vision-Language Model Methods</th></tr><tr><th>WAM-Flow</th><td>98.5</td><td>94.5</td><td>99.5</td><td>99.8</td><td>86.9</td><td>96.8</td><td>97.4</td><td>97.6</td><td>73.9</td><td>84.7</td><td>–</td></tr><tr><th>ReCogDrive <sup>†</sup></th><td>98.3</td><td>95.2</td><td>98.3</td><td>99.8</td><td>87.1</td><td>97.5</td><td>96.6</td><td>99.5</td><td>86.5</td><td>83.6</td><td>–</td></tr><tr><th>DriveVLA-W0</th><td>98.5</td><td>99.1</td><td>98.0</td><td>99.7</td><td>86.4</td><td>98.1</td><td>93.2</td><td>97.9</td><td>58.9</td><td>–</td><td>86.1</td></tr><tr><th>SGDrive <sup>†</sup></th><td>98.6</td><td>94.3</td><td>99.5</td><td>99.9</td><td>86.0</td><td>97.9</td><td>96.1</td><td>98.3</td><td>85.9</td><td>–</td><td>86.2</td></tr><tr><th>DriveWorld-VLA</th><td>98.6</td><td>99.1</td><td>99.6</td><td>99.8</td><td>87.4</td><td>97.9</td><td>97.0</td><td>97.8</td><td>78.6</td><td>–</td><td>86.8</td></tr><tr><th>CoWorld-VLA (ours)</th><td>99.1</td><td>97.0</td><td>99.6</td><td>99.9</td><td>87.8</td><td>98.5</td><td>97.7</td><td>98.2</td><td>86.2</td><td>86.2</td><td>90.0</td></tr></tbody></table>

##### Evaluation of video generation results.

In Stage 2, latent future world states are injected into the video DiT for future-aware video generation. Table 3 reports FVD results for CoWorld-VLA and prior driving video generation methods. CoWorld-VLA obtains an FVD of 32.7 on NAVSIM.

Table 3: Reported FVD results across driving benchmarks. Lower FVD is better under matched data and evaluation protocols. Dataset and evaluation settings may differ across methods; cross-setting results are provided for contextual reference.

| Method | SVD [^84] | GenAD [^85] | DrivingGPT [^18] | Epona [^59] | DriveLaW [^19] | CoWorld-VLA (ours) |
| --- | --- | --- | --- | --- | --- | --- |
| Dataset | NAVSIM | OpenDV | NAVSIM | NuPlan | NuPlan | NAVSIM |
| FVD $\downarrow$ | 227.5 | 184.0 | 142.6 | 61.3 | 55.6 | 32.7 |

##### Qualitative results of video generation.

As shown in Figure 3, the Stage 1 model generates stable future scenes but may deviate from the ground-truth driving direction at intersections. With joint video DiT and VLM-based latent world modeling, the Stage 2 model better preserves turning behavior and road-layout evolution, producing predictions more consistent with the ground truth.

![[case1_main.png|Refer to caption]]

Figure 3: Qualitative comparison of future scene generation. Compared with Stage 1, Stage 2 better preserves the driving direction and lane-level scene evolution, producing future frames that are more consistent with the ground truth.

##### Qualitative results of trajectory planning.

As shown in Figure 4, Stage 2 can predict generally reasonable driving directions, but still shows deviations in lane keeping and turning scenarios. In comparison, Stage 3 with HMEF produces trajectories that are closer to the ground truth, especially in lateral position and turning tendency. These results show that HMEF further improves trajectory planning quality in complex driving scenes.

![[traj_best.png|Refer to caption]]

Figure 4: Qualitative comparison of trajectory planning across different training stages. Stage 2 predicts generally reasonable driving directions, but still shows deviations in lane keeping and turning scenarios. Stage 3 with HMEF generates trajectories that better align with the ground truth.

### 4.3 Ablation study

##### Ablation study on multi-expert design.

Table 4 reports the ablation results under the expert-token configurations listed from top to bottom. Using only the Ego Trajectory Token achieves a PDMS of 83.7. Adding the Geometric Structure Token alone improves PDMS to 85.1, while adding the Semantic Interaction Token alone gives 85.2. Starting from EgoT.+Sem., adding the Dynamic Evolution Token increases PDMS to 87.3; adding the Geometric Structure Token instead increases it to 87.7. Finally, using all four expert-token groups achieves 88.7, improving PDMS by 5.0 over EgoT. The two branches show that geometric, semantic, and dynamic supervision provide complementary planning cues rather than interchangeable information.

Table 4: Ablation study of latent world representations in Stage 2. EgoT., Geo., Sem., and Dyn. denote Ego Trajectory, Geometric Structure, Semantic Interaction, and Dynamic Evolution, respectively.

| EgoT. | Geo. | Sem. | Dyn. | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf.$\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ✓ |  |  |  | 97.7 | 92.7 | 92.7 | 100 | 78.5 | 83.7 |
| ✓ | ✓ |  |  | 97.7 | 93.9 | 92.7 | 100 | 80.3 | 85.1 |
| ✓ |  | ✓ |  | 98.1 | 93.7 | 93.8 | 100 | 79.0 | 85.2 |
| ✓ |  | ✓ | ✓ | 98.3 | 95.4 | 95.1 | 100 | 81.1 | 87.3 |
| ✓ | ✓ | ✓ |  | 98.4 | 95.6 | 95.0 | 100 | 81.9 | 87.7 |
| ✓ | ✓ | ✓ | ✓ | 98.4 | 96.5 | 95.3 | 100 | 82.3 | 88.7 |

Table 5: Ablation study of representation learning and planning modules on NAVSIM v1. Full Experts denotes the joint use of EgoT., Geo., Sem., and Dyn.; AE denotes the ReCogDrive [^50] action expert.

<table><thead><tr><th colspan="2">Representation</th><th colspan="3">Planner</th><th rowspan="2">NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th rowspan="2">DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th rowspan="2">TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th rowspan="2">Comf.<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th rowspan="2">EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th rowspan="2">PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th></tr><tr><th>EgoT. only</th><th>Full Experts</th><th>VLM only</th><th>VLM+AE</th><th>VLM+HMEF</th></tr></thead><tbody><tr><td>✓</td><td></td><td>✓</td><td></td><td></td><td>97.7</td><td>92.7</td><td>92.7</td><td>100</td><td>78.5</td><td>83.7</td></tr><tr><td></td><td>✓</td><td>✓</td><td></td><td></td><td>98.4</td><td>96.5</td><td>95.3</td><td>100</td><td>82.3</td><td>88.7</td></tr><tr><td></td><td>✓</td><td></td><td>✓</td><td></td><td>98.5</td><td>96.9</td><td>95.4</td><td>100</td><td>83.2</td><td>89.1</td></tr><tr><td>✓</td><td></td><td></td><td></td><td>✓</td><td>98.3</td><td>96.8</td><td>95.0</td><td>100</td><td>83.2</td><td>88.9</td></tr><tr><td></td><td>✓</td><td></td><td></td><td>✓</td><td>99.1</td><td>97.0</td><td>96.5</td><td>100</td><td>84.0</td><td>90.0</td></tr></tbody></table>

##### Ablation study on representation and planning modules.

Table 5 jointly evaluates representation learning and trajectory planning. With VLM only fixed, Full Experts improves PDMS from 83.7 with EgoT only to 88.7. With Full Experts fixed, VLM+AE and VLM+HMEF achieve PDMS scores of 89.1 and 90.0, improving PDMS by 0.4 and 1.3 over VLM only, respectively. With VLM+HMEF fixed, EgoT only achieves 88.9 compared with 90.0 for Full Experts, indicating complementary benefits from the planner and multi-expert representations.

##### Denoising step ablation.

As shown in Table 6, we ablate the number of denoising steps in the fusion diffusion planner. The results suggest that 10 denoising steps provide the best balance between accuracy and efficiency.

Table 6: Effect of diffusion denoising steps.

|  | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf.$\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| Step = 5 | 99.0 | 96.8 | 95.9 | 100 | 83.6 | 89.5 |
| Step = 10 | 99.1 | 97.0 | 96.5 | 100 | 84.0 | 90.0 |
| Step = 20 | 99.1 | 96.7 | 96.4 | 100 | 83.5 | 89.7 |

## 5 Conclusions

We propose CoWorld-VLA, a multi-expert world reasoning framework. By encoding semantic, geometric, dynamic, and trajectory priors into expert tokens, it forms a planning-oriented Latent CoT. A hierarchical fusion planner then integrates these tokens to generate continuous trajectories. Experiments demonstrate effective future-scene modeling and strong planning performance in complex driving scenarios.

Limitations.

The primary limitation of our framework is the substantial computational overhead incurred during multi-stage training. This cost is concentrated in teacher-based pretraining and VLM representation learning; the Wan model is not used in Stage 3 or during planning inference, while the Stage-2 VLM and the V-JEPA/VGGT encoders are frozen during Stage 3. The current model is also limited to single-image inputs. Future work will investigate caching teacher features, lightweight dynamics teachers, parameter-efficient fine-tuning such as LoRA, and distillation or removal of online expert encoders, while extending the framework to multi-camera settings.

## References

## Appendix A Technical appendices and supplementary material

### A.1 More implementation details

#### A.1.1 Additional details of stage 1 action-conditioned predictive world model

This section provides additional details on the conditioning design of the Stage 1 predictive world model. In this stage, the word “action-conditioned” refers to text-form conditions that describe ego intention and motion, rather than low-level continuous control commands. These conditions are encoded by the frozen UMT5 text encoder from Wan and used to guide future scene prediction.

##### Condition prompt construction.

As defined in Sec. 3.1, the condition prompt $\mathcal{P}$ is composed of four components. Here, we detail how each component is serialized into natural language. $[\mathrm{Scene}]$ is a static scene prompt from the configuration, $[\mathrm{Speed}]$ describes the ego speed with a coarse natural-language phrase, $[\mathrm{Navigation}]$ provides the high-level driving command when available, and $[\mathrm{Trajectory}]$ serializes ego waypoints in the local ego-centric coordinate system. The current ego position is explicitly anchored at the origin to establish a clear coordinate frame for the text encoder. This prompt is then encoded by the frozen UMT5 encoder and used as the conditioning signal for the predictive world model.

##### Optional fields and fallback behavior.

The navigation command is optional. When it is available, it is converted into one of four textual commands: turn left, go straight, turn right, and unknown. When the navigation command is absent, the corresponding command phrase is omitted from the prompt. Similarly, the trajectory branch is used only when a valid future trajectory is provided. If no valid future trajectory is available, the condition encoder falls back to the configured scene prompt without adding the speed, command, or trajectory clauses.

##### Speed and trajectory representation.

The ego speed is represented by a coarse phrase rather than a raw numerical value. Specifically, the speed is mapped into phrases such as nearly stopped, driving slowly, driving at moderate speed, and driving at high speed. When an explicit ego speed is unavailable, it can be estimated from ego-state information or approximated from the future trajectory displacement. The trajectory condition is represented as a sequence of future $(x,y)$ waypoints with fixed decimal precision. We omit heading values in the text prompt, since the polyline shape together with the navigation command already provides sufficient directional information.

#### A.1.2 Additional details of stage 2 multi-expert representation learning

This section provides additional details on the multi-expert representation learning stage. The main formulation is described in Sec. 3.3. Here, we focus on how expert tokens are organized inside the VLM and how different supervision branches shape planning-oriented latent representations.

##### Expert token organization.

In Stage 2, we insert four groups of learnable expert tokens into the VLM input sequence, corresponding to semantic interaction, geometric structure, dynamic evolution, and ego trajectory. After the image tokens, text tokens, and expert tokens are processed by the VLM, the hidden states at the corresponding expert-token positions are extracted as $H_{\mathrm{sem}}$, $H_{\mathrm{geo}}$, $H_{\mathrm{dyn}}$, and $H_{\mathrm{traj}}$. These hidden states are not treated as textual outputs. Instead, they serve as continuous latent representations that form a planning-oriented Latent CoT.

##### Token-level expert alignment.

For the JEPA and VGGT branches, the frozen expert models produce dense visual features rather than a single global vector. We therefore apply pooling to convert expert outputs into compact token sequences. The number of pooled expert tokens is kept consistent with the corresponding action-token group, so that each VLM expert token can be aligned with one target expert feature. Since the VLM hidden dimension and the expert feature dimension are generally different, lightweight projection modules are used to map $H_{\mathrm{sem}}$ and $H_{\mathrm{geo}}$ into the corresponding expert feature spaces before applying the alignment losses.

##### Semantic and geometric supervision.

The JEPA branch provides high-level semantic supervision from future observations, encouraging the semantic interaction token to encode object-level context, scene semantics, and interaction-related information. The VGGT branch provides geometric supervision, encouraging the geometric structure tokens to encode road layout, spatial configuration, and 3D structural cues. These two branches complement each other: JEPA offers abstract semantic and temporal understanding, while VGGT provides explicit spatial grounding.

##### World-model supervision through action-token conditioning.

For the dynamic evolution branch, the VLM-generated world-model tokens $H_{\mathrm{dyn}}$ are used as conditional latent variables for the Wan world model. The VLM itself does not directly decode future images. Instead, future scene generation is performed by the Wan world model conditioned on $H_{\mathrm{dyn}}$, and the flow-matching objective provides supervision to the corresponding action tokens. This design allows the dynamic tokens to learn future motion trends and temporal consistency without requiring the VLM backbone to act as a pixel-level generator.

##### Trajectory-token supervision.

The trajectory tokens are supervised by a lightweight trajectory regression head. This branch encourages $H_{\mathrm{traj}}$ to encode behavior-oriented information that is directly related to future ego motion. In Stage 2, this trajectory head is used to shape the latent action representation rather than serve as the final planner. The final trajectory generation is performed in Stage 3 by the hierarchical diffusion planner, which further integrates multi-expert representations.

##### Role of Stage 2.

Overall, Stage 2 transforms heterogeneous expert supervision into structured latent states inside the VLM. Instead of relying only on language supervision, the VLM receives complementary constraints from semantic representation learning, geometric structure alignment, future scene generation, and trajectory regression. These expert tokens are then reused by the downstream action expert as planning-oriented latent conditions.

#### A.1.3 Additional details of hierarchical multi-expert fusion

This section provides additional implementation details of the Hierarchical Multi-Expert Fusion (HMEF) planner. The main formulation of HMEF is described in Sec. 3.4. Here, we focus on scene compression, per-step expert feature extraction, trajectory normalization, and fusion-weight training.

##### Scene context compression.

The scene context tokens can be long and variable in length. To reduce the computational cost of the denoiser, each enabled scene-context source is compressed by its own Perceiver-style compressor. Specifically, non-action VLM tokens, current-frame JEPA context tokens, and current-frame VGGT context tokens are separately compressed into fixed-length latent tokens when enabled. These compressed tokens are then concatenated along the sequence dimension and used as the clean scene stream in the subsequent joint denoising process.

##### Per-step expert feature extraction.

Each expert action-token group is first projected to the HMEF hidden dimension and processed by an expert-specific bidirectional Transformer. To align expert tokens with the planning horizon, we organize the output tokens by future timestep. If multiple tokens correspond to the same future step, their projected features are averaged to obtain one per-step expert feature. This yields a sequence of expert conditions aligned with the future waypoints, allowing each denoising step to receive timestep-specific semantic, geometric, dynamic, or trajectory-prior guidance.

##### Historical and ego-state conditioning.

In addition to the expert tokens, HMEF also utilizes low-level ego information. The historical ego trajectory and the current ego status are separately encoded by lightweight MLPs and then fused into a single conditioning vector. This fused ego condition is expanded along the planning horizon and combined with the noisy action token and the corresponding expert feature before being passed into the denoiser. This design allows the planner to jointly leverage high-level abstract priors and low-level vehicle-state telemetry.

##### Trajectory normalization.

The conditional denoising process is performed in a normalized action space. Specifically, the future trajectory $(x,y,\psi)$ is normalized to $[-1,1]$ using empirical coordinate ranges calculated from the NAVSIM dataset. During inference, the predicted trajectory is denormalized back to the original coordinate space before evaluation. This normalization stabilizes the training dynamics and keeps the coordinate dimensions on comparable mathematical scales.

##### Fusion-weight optimization.

HMEF predicts one trajectory for each active expert branch and learns global scalar fusion weights to combine them. During the calculation of the fusion loss, the individual expert trajectories are detached from the computation graph before weighted averaging. Consequently, the fusion objective updates only the fusion weights rather than propagating gradients back into the individual expert denoising branches, while the expert branches are still trained by the denoising loss. This helps stabilize training by separating expert-specific trajectory generation from global expert-importance learning.

#### A.1.4 Training details.

CoWorld-VLA comprises three components: a video diffusion Transformer (Wan2.2-5B), a VLM (Qwen3-VL-2B), and an action expert network. We adopt a three-stage training strategy. First, the video DiT is pretrained on 8 Hz NuPlan videos for future video generation using 48k training steps, a batch size of 192, and a cosine learning rate schedule with warmup. Second, the VLM is fine-tuned on NAVSIM v1 with multi-expert supervision for 40k steps, using learning rates of $2\times 10^{-5}$ for the VLM, $1\times 10^{-4}$ for the JEPA adaptor, and $1\times 10^{-5}$ for newly introduced modules. Third, the action expert network is trained for 60k steps with the fine-tuned VLM frozen, using a batch size of 256 and a learning rate of $2\times 10^{-5}$. Stage 1 training is conducted on 64 NVIDIA A800 GPUs and requires approximately 74 hours. Stage 2 uses 32 NVIDIA A800 GPUs with approximately 70 training hours, while Stage 3 is trained on 16 NVIDIA A800 GPUs for approximately 20 hours.

### A.2 More experimental results

#### A.2.1 Inference-seed stability

Holding the trained Stage 3 weights fixed, we evaluate six inference-time random seeds. Table 7 reports means of 89.980 (standard deviation 0.013) for PDMS and 89.985 (standard deviation 0.014) for EPDMS. These results indicate low sensitivity to the evaluated inference seeds. They do not replace variance estimates over independently trained models, which were not conducted because of the high training cost.

Table 7: Performance stability across inference-time random seeds.

| Metric | Seed 42 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Seed 5 |
| --- | --- | --- | --- | --- | --- | --- |
| PDMS | 89.964 | 89.975 | 89.986 | 89.971 | 89.982 | 90.000 |
| EPDMS | 89.979 | 89.989 | 89.961 | 89.983 | 90.000 | 89.997 |

#### A.2.2 Learnable Expert Weight Analysis

The learnable expert weights are initialized uniformly at 0.25. After convergence, the weights evolve to approximately 0.35 for dynamic evolution expert, 0.19 for semantic interaction expert, 0.15 for geometric structure expert, and 0.31 for trajectory expert, indicating that the model automatically assigns higher importance to dynamic and trajectory-related representations during planning.

#### A.2.3 Additional qualitative results on future video generation

We provide additional qualitative comparisons of future video generation in Figure 5, covering three representative scenarios: left-turn at a forked intersection, straight cruising on a multi-lane urban road, and close-proximity car-following in dense traffic. For each scenario, we compare the ground truth (GT) with Stage 1 and Stage 2 predictions.

In the forked-intersection scenario (Figure 5(a)), Stage 1 predicts a straight-driving future instead of the intended left turn. Stage 2 preserves the left-turn trajectory and aligns well with the GT, demonstrating the effect of multi-expert Latent CoT in constraining behaviorally relevant predictions.

For straight cruising (Figure 5(b)), Stage 1 gradually drifts toward the adjacent lane, while Stage 2 maintains lane alignment and stable forward progression. This highlights the benefit of geometric structure supervision for road-layout consistency.

In the close-proximity car-following scenario (Figure 5(c)), both Stage 1 and Stage 2 capture general vehicle motion, but Stage 2 better preserves relative spacing and motion continuity, showing the advantage of temporal dynamics supervision in dense traffic.

Overall, Stage 2 multi-expert training improves temporal consistency, preserves lane-level structure, and produces predictions that more faithfully follow ego intentions compared with Stage 1.

![[case1.png|Refer to caption]]

(a) Left Turn Navigation at a Forked Intersection

In addition, Figure 6 highlights a local fidelity comparison in a downtown driving scene. The red boxes mark parked vehicles on the right side of the road. While Stage 1 produces a plausible global road layout, the highlighted region becomes blurred and distorted in later frames, showing reduced stability in local object appearance. Stage 2 preserves clearer vehicle boundaries and more consistent roadside structure throughout the generated sequence. This case suggests that Stage 2 multi-expert supervision improves not only high-level driving direction and trajectory alignment, but also fine-grained visual consistency in future scene generation.

![[case4.png|Refer to caption]]

Figure 6: Local fidelity comparison in future video generation. The red boxes highlight roadside vehicles that become blurred and distorted in the Stage 1 prediction, while Stage 2 preserves clearer object boundaries and more stable local structure.

#### A.2.4 Additional qualitative results on trajectory planning

We provide qualitative comparisons of Stage 2 and Stage 3 trajectories across three representative driving scenarios in Figure 7. Overall, Stage 2 captures coarse driving intentions, while Stage 3 produces more accurate and stable trajectories through hierarchical multi-expert fusion.

As shown in Figure 7(a), the lane-keeping cruising scenario requires stable forward motion within the current lane. Stage 2 predicts the general driving direction but exhibits noticeable lateral drift over the planning horizon. In contrast, Stage 3 generates a more centered trajectory that closely follows the GT, indicating improved lane-level consistency and long-horizon stability.

Figure 7(b) shows an intersection left-turn scenario, where planning requires both navigation-intention understanding and road-topology awareness. Stage 2 deviates from the desired turning behavior, while Stage 3 better follows the intersection layout and produces a smooth left-turn trajectory closer to the GT. This suggests that HMEF more effectively translates latent reasoning states into executable turning actions.

Figure 7(c) presents a detour maneuver around a leading vehicle. Stage 2 captures part of the forward motion but fails to match the correct bypass trajectory. Stage 3 better preserves the required lateral offset and trajectory curvature, demonstrating stronger vehicle-aware planning ability.

Overall, these results show that Stage 3 consistently improves over Stage 2 in lane keeping, intersection turning, and detour planning. By fusing heterogeneous expert priors during diffusion-based action generation, HMEF better converts multi-expert Latent CoT representations into actionable and behavior-consistent ego trajectories.

![[traj_2.png|Refer to caption]]

(a) Lane-Keeping Cruising Scenario

[^1]: S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge, et al. (2025) Qwen3-vl technical report. arXiv preprint arXiv:2511.21631. Cited by: §1, §4.1.

[^2]: H. Liu, C. Li, Y. Li, and Y. J. Lee (2024) Improved baselines with visual instruction tuning. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 26286–26296. External Links: [Document](https://dx.doi.org/10.1109/CVPR52733.2024.02484) Cited by: §1.

[^3]: Z. Chen, J. Wu, W. Wang, W. Su, G. Chen, S. Xing, M. Zhong, Q. Zhang, X. Zhu, L. Lu, et al. (2024) Internvl: scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 24185–24198. Cited by: §1.

[^4]: K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, et al. (2024) $\pi_{0}$: a vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164. Cited by: §1.

[^5]: X. Zhou, X. Han, F. Yang, Y. Ma, V. Tresp, and A. Knoll (2026) Opendrivevla: towards end-to-end autonomous driving with large vision language action model. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40, pp. 13782–13790. Cited by: §1, §3.1.

[^6]: Y. Wang, X. Li, W. Wang, J. Zhang, Y. Li, Y. Chen, X. Wang, and Z. Zhang (2026) Unified vision-language-action model. In The Fourteenth International Conference on Learning Representations, Cited by: §1, §2, §3.1.

[^7]: E. Cui, W. Wang, Z. Li, J. Xie, H. Zou, H. Deng, G. Luo, L. Lu, X. Zhu, and J. Dai (2025) DriveMLM: aligning multi-modal large language models with behavioral planning states for autonomous driving. Visual Intelligence 3 (1), pp. 22. Cited by: §1, §3.1.

[^8]: X. Tian, J. Gu, B. Li, Y. Liu, Y. Wang, Z. Zhao, K. Zhan, P. Jia, X. Lang, and H. Zhao (2024) Drivevlm: the convergence of autonomous driving and large vision-language models. arXiv preprint arXiv:2402.12289. Cited by: §1, §2, §3.1.

[^9]: B. Jiang, S. Chen, B. Liao, X. Zhang, W. Yin, Q. Zhang, C. Huang, W. Liu, and X. Wang (2024) Senna: bridging large vision-language models and end-to-end autonomous driving. arXiv preprint arXiv:2410.22313. Cited by: §1, §2, §3.1.

[^10]: A. Jiang, Y. Gao, Z. Sun, Y. Wang, J. Wang, J. Chai, Q. Cao, Y. Heng, H. Jiang, Y. Dong, et al. (2025) Diffvla: vision-language guided diffusion planning for autonomous driving. arXiv preprint arXiv:2505.19381. Cited by: §1, §2, §3.1.

[^11]: C. Pan, B. Yaman, T. Nesti, A. Mallik, A. G. Allievi, S. Velipasalar, and L. Ren (2024) Vlp: vision language planning for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14760–14769. Cited by: §1, §2, §3.1.

[^12]: H. Fu, D. Zhang, Z. Zhao, J. Cui, D. Liang, C. Zhang, D. Zhang, H. Xie, B. Wang, and X. Bai (2025) Orion: a holistic end-to-end autonomous driving framework by vision-language instructed action generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 24823–24834. Cited by: §1, §2, §3.1.

[^13]: H. Caesar, J. Kabzan, K. S. Tan, W. K. Fong, E. Wolff, A. Lang, L. Fletcher, O. Beijbom, and S. Omari (2021) Nuplan: a closed-loop ml-based planning benchmark for autonomous vehicles. arXiv preprint arXiv:2106.11810. Cited by: §1, §4.1.

[^14]: D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, et al. (2024) Navsim: data-driven non-reactive autonomous vehicle simulation and benchmarking. Advances in Neural Information Processing Systems 37, pp. 28706–28719. Cited by: §1, §4.1, §4.1.

[^15]: B. Jiang, S. Chen, H. Gao, B. Liao, Q. Zhang, W. Liu, and X. Wang (2024) VADv2: end-to-end vectorized autonomous driving via probabilistic planning. In The Fourteenth International Conference on Learning Representations, Cited by: §1.

[^16]: Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, et al. (2023) Planning-oriented autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 17853–17862. Cited by: §1, Table 1.

[^17]: K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger (2022) Transfuser: imitation with transformer-based sensor fusion for autonomous driving. IEEE transactions on pattern analysis and machine intelligence 45 (11), pp. 12878–12895. Cited by: §1.

[^18]: Y. Chen, Y. Wang, and Z. Zhang (2025) Drivinggpt: unifying driving world modeling and planning with multi-modal autoregressive transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 26890–26900. Cited by: §1, §2, Table 3.

[^19]: T. Xia, Y. Li, L. Zhou, J. Yao, K. Xiong, H. Sun, B. Wang, K. Ma, G. Chen, H. Ye, et al. (2025) Drivelaw: unifying planning and video generation in a latent driving world. arXiv preprint arXiv:2512.23421. Cited by: §1, §2, Table 1, Table 3.

[^20]: J. Li, B. Zhang, X. Jin, J. Deng, X. Zhu, and L. Zhang (2025) ImagiDrive: a unified imagination-and-planning framework for autonomous driving. arXiv preprint arXiv:2508.11428. Cited by: §1.

[^21]: Z. Xu, Y. Zhang, E. Xie, Z. Zhao, Y. Guo, K. K. Wong, Z. Li, and H. Zhao (2024) Drivegpt4: interpretable end-to-end autonomous driving via large language model. IEEE Robotics and Automation Letters 9 (10), pp. 8186–8193. Cited by: §1, §2.

[^22]: Y. Luo, F. Li, S. Xu, Z. Lai, L. Yang, Q. Chen, Z. Luo, Z. Xie, S. Jiang, J. Liu, et al. (2025) Adathinkdrive: adaptive thinking via reinforcement learning for autonomous driving. arXiv preprint arXiv:2509.13769. Cited by: §1, §2.

[^23]: Z. Zhou, T. Cai, S. Z. Zhao, Y. Zhang, Z. Huang, B. Zhou, and J. Ma (2025) Autovla: a vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. arXiv preprint arXiv:2506.13757. Cited by: §1, §2.

[^24]: J. Cheng and B. Van Durme (2024) Compressed chain of thought: efficient reasoning through dense representations. arXiv preprint arXiv:2412.13171. Cited by: §1, §2.

[^25]: S. Hao, S. Sukhbaatar, D. Su, X. Li, Z. Hu, J. Weston, and Y. Tian (2024) Training large language models to reason in a continuous latent space. arXiv preprint arXiv:2412.06769. Cited by: §1, §2.

[^26]: A. Ray, A. Abdelkader, C. Mao, B. A. Plummer, K. Saenko, R. Krishna, L. Guibas, and W. Chu (2025) Mull-tokens: modality-agnostic latent thinking. arXiv preprint arXiv:2512.10941. Cited by: §1, §2.

[^27]: B. Li, X. Sun, J. Liu, Z. Wang, J. Wu, X. Yu, H. Chen, E. Barsoum, M. Chen, and Z. Liu (2025) Latent visual reasoning. arXiv preprint arXiv:2509.24251. External Links: [Document](https://dx.doi.org/10.48550/arXiv.2509.24251) Cited by: §1, §2.

[^28]: X. Li, X. He, L. Zhang, M. Wu, X. Li, and Y. Liu (2025) A comprehensive survey on world models for embodied ai. arXiv preprint arXiv:2510.16732. Cited by: §1, §2.

[^29]: T. Brooks, B. Peebles, C. Holmes, W. DePue, Y. Guo, L. Jing, D. Schnurr, J. Taylor, T. Luhman, E. Luhman, et al. (2024) Video generation models as world simulators. OpenAI Blog 1 (8), pp. 1. Cited by: §1, §2.

[^30]: S. Gao, J. Yang, L. Chen, K. Chitta, Y. Qiu, A. Geiger, J. Zhang, and H. Li (2024) Vista: a generalizable driving world model with high fidelity and versatile controllability. Advances in Neural Information Processing Systems 37, pp. 91560–91596. Cited by: §1, §2.

[^31]: X. Wang, Z. Zhu, G. Huang, X. Chen, J. Zhu, and J. Lu (2024) Drivedreamer: towards real-world-drive world models for autonomous driving. In European conference on computer vision, pp. 55–72. Cited by: §1, §2.

[^32]: A. Hu, L. Russell, H. Yeo, Z. Murez, G. Fedoseev, A. Kendall, J. Shotton, and G. Corrado (2023) Gaia-1: a generative world model for autonomous driving. arXiv preprint arXiv:2309.17080. Cited by: §1, §2.

[^33]: X. Hu, W. Yin, M. Jia, J. Deng, X. Guo, Q. Zhang, X. Long, and P. Tan (2024) DrivingWorld: constructing world model for autonomous driving via video gpt. arXiv preprint arXiv:2412.19505. Cited by: §1, §2.

[^34]: M. Assran, A. Bardes, D. Fan, Q. Garrido, R. Howes, M. Muckley, A. Rizvi, C. Roberts, K. Sinha, A. Zholus, et al. (2025) V-jepa 2: self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985. Cited by: §1, §1, §2.

[^35]: R. Gao, K. Chen, E. Xie, L. Hong, Z. Li, D. Yeung, and Q. Xu (2023) Magicdrive: street view generation with diverse 3d geometry control. arXiv preprint arXiv:2310.02601. Cited by: §1, §2.

[^36]: A. Chen, W. Zheng, Y. Wang, X. Zhang, K. Zhan, P. Jia, K. Keutzer, and S. Zhang (2025) Geodrive: 3d geometry-informed driving world model with precise action control. arXiv preprint arXiv:2505.22421. Cited by: §1, §2.

[^37]: J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, and D. Novotny (2025) Vggt: visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 5294–5306. Cited by: §1, §1, §2.

[^38]: M. Assran, Q. Duval, I. Misra, P. Bojanowski, P. Vincent, M. Rabbat, Y. LeCun, and N. Ballas (2023) Self-supervised learning from images with a joint-embedding predictive architecture. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 15619–15629. Cited by: §1, §1, §2.

[^39]: A. Bardes, Q. Garrido, J. Ponce, X. Chen, M. Rabbat, Y. LeCun, M. Assran, and N. Ballas (2024) Revisiting feature prediction for learning visual representations from video. arXiv preprint arXiv:2404.08471. Cited by: §1, §2.

[^40]: Y. Wang, J. He, L. Fan, H. Li, Y. Chen, and Z. Zhang (2024) Driving into the future: multiview visual forecasting and planning with world model for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14749–14759. Cited by: §1, §2.

[^41]: C. Min, D. Zhao, L. Xiao, J. Zhao, X. Xu, Z. Zhu, L. Jin, J. Li, Y. Guo, J. Xing, et al. (2024) Driveworld: 4d pre-trained scene understanding via world models for autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 15522–15533. Cited by: §1, §2.

[^42]: Y. Li, L. Fan, J. He, Y. Wang, Y. Chen, Z. Zhang, and T. Tan (2024) Enhancing end-to-end autonomous driving with latent world model. arXiv preprint arXiv:2406.08481. Cited by: §1, Table 1.

[^43]: Y. Li, Y. Wang, Y. Liu, J. He, L. Fan, and Z. Zhang (2025) End-to-end driving with online trajectory evaluation via bev world model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 27137–27146. Cited by: §1, Table 1.

[^44]: T. Wan, A. Wang, B. Ai, B. Wen, C. Mao, C. Xie, D. Chen, F. Yu, H. Zhao, J. Yang, et al. (2025) Wan: open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314. Cited by: §1, §2, §4.1.

[^45]: C. Sima, K. Renz, K. Chitta, L. Chen, H. Zhang, C. Xie, J. Beißwenger, P. Luo, A. Geiger, and H. Li (2024) Drivelm: driving with graph visual question answering. In European conference on computer vision, pp. 256–274. Cited by: §2.

[^46]: L. Chen, H. Hassani, and S. Nikan (2025) Ts-vlm: text-guided softsort pooling for vision-language models in multi-view driving reasoning. arXiv preprint arXiv:2505.12670. Cited by: §2.

[^47]: J. Mao, Y. Qian, J. Ye, H. Zhao, and Y. Wang (2023) Gpt-driver: learning to drive with gpt. arXiv preprint arXiv:2310.01415. Cited by: §2.

[^48]: J. Hwang, R. Xu, H. Lin, W. Hung, J. Ji, K. Choi, D. Huang, T. He, P. Covington, B. Sapp, et al. (2024) Emma: end-to-end multimodal model for autonomous driving. arXiv preprint arXiv:2410.23262. Cited by: §2.

[^49]: S. Wang, Z. Yu, X. Jiang, S. Lan, M. Shi, N. Chang, J. Kautz, Y. Li, and J. M. Alvarez (2025) Omnidrive: a holistic vision-language dataset for autonomous driving with counterfactual reasoning. In Proceedings of the computer vision and pattern recognition conference, pp. 22442–22452. Cited by: §2.

[^50]: Y. Li, K. Xiong, X. Guo, F. Li, S. Yan, G. Xu, L. Zhou, L. Chen, H. Sun, B. Wang, et al. (2025) Recogdrive: a reinforced cognitive framework for end-to-end autonomous driving. arXiv preprint arXiv:2506.08052. Cited by: §2, §4.1, Table 1, Table 1, Table 5.

[^51]: Z. Zhu, J. Liang, S. Jiang, J. Fu, M. Liu, G. Sun, S. Ng, and B. Qin (2026) Analyzing reasoning consistency in large multimodal models under cross-modal conflicts. arXiv preprint arXiv:2601.04073. Cited by: §2.

[^52]: C. Lou, Z. Sun, X. Liang, M. Qu, W. Shen, W. Wang, Y. Li, Q. Yang, and S. Wu (2025) Adacot: pareto-optimal adaptive chain-of-thought triggering via reinforcement learning. arXiv preprint arXiv:2505.11896. Cited by: §2.

[^53]: Z. Yang, Y. Chai, X. Jia, Q. Li, Y. Shao, X. Zhu, H. Su, and J. Yan (2025) DriveMoE: mixture-of-experts for vision-language-action model in end-to-end autonomous driving. arXiv preprint arXiv:2505.16278. Cited by: §2.

[^54]: Y. Luo, F. Li, S. Xu, Y. Ji, Z. Zhang, B. Wang, Y. Shen, J. Cui, L. Chen, G. Chen, et al. (2026) Last-vla: thinking in latent spatio-temporal space for vision-language-action in autonomous driving. arXiv preprint arXiv:2603.01928. Cited by: §2, Table 1.

[^55]: J. Lu, Z. Huang, Z. Yang, J. Zhang, and L. Zhang (2024) Wovogen: world volume-aware diffusion for controllable multi-camera driving scene generation. In European conference on computer vision, pp. 329–345. Cited by: §2.

[^56]: L. Zhang, Y. Xiong, Z. Yang, S. Casas, R. Hu, and R. Urtasun (2023) Copilot4d: learning unsupervised world models for autonomous driving via discrete diffusion. arXiv preprint arXiv:2311.01017. Cited by: §2.

[^57]: V. Zyrianov, H. Che, Z. Liu, and S. Wang (2025) Lidardm: generative lidar simulation in a generated world. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pp. 6055–6062. Cited by: §2.

[^58]: B. Li, J. Guo, H. Liu, Y. Zou, Y. Ding, X. Chen, H. Zhu, F. Tan, C. Zhang, T. Wang, et al. (2025) Uniscene: unified occupancy-centric driving scene generation. In Proceedings of the computer vision and pattern recognition conference, pp. 11971–11981. Cited by: §2.

[^59]: K. Zhang, Z. Tang, X. Hu, X. Pan, X. Guo, Y. Liu, J. Huang, L. Yuan, Q. Zhang, X. Long, et al. (2025) Epona: autoregressive diffusion world model for autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 27220–27230. Cited by: §2, §4.1, Table 1, Table 3.

[^60]: J. Bruce, M. D. Dennis, A. Edwards, J. Parker-Holder, Y. Shi, E. Hughes, M. Lai, A. Mavalankar, R. Steigerwald, C. Apps, et al. (2024) Genie: generative interactive environments. In Forty-first International Conference on Machine Learning, Cited by: §2.

[^61]: R. Gao, K. Chen, B. Xiao, L. Hong, Z. Li, and Q. Xu (2025) MagicDrive-v2: high-resolution long video generation for autonomous driving with adaptive control. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 28135–28144. Cited by: §2.

[^62]: J. Guo, Y. Ding, X. Chen, S. Chen, B. Li, Y. Zou, X. Lyu, F. Tan, X. Qi, Z. Li, et al. (2025) Dist-4d: disentangled spatiotemporal diffusion with metric depth for 4d driving scene generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 27231–27241. Cited by: §2.

[^63]: X. Guo, Z. Wu, K. Xiong, Z. Xu, L. Zhou, G. Xu, S. Xu, H. Sun, B. Wang, G. Chen, et al. (2025) Genesis: multimodal driving scene generation with spatio-temporal and cross-modal consistency. arXiv preprint arXiv:2506.07497. Cited by: §2.

[^64]: Y. Li, S. Shang, W. Liu, B. Zhan, H. Wang, Y. Wang, Y. Chen, X. Wang, Y. An, C. Tang, et al. (2025) DriveVLA-w0: world models amplify data scaling law in autonomous driving. arXiv preprint arXiv:2510.12796. Cited by: §2, Table 1.

[^65]: Q. Liu, H. Xu, J. Li, B. Sun, Z. Hao, D. She, X. Zhu, and L. Zhang (2026) Uni-world vla: interleaved world modeling and planning for autonomous driving. arXiv preprint arXiv:2603.27287. Cited by: §2, §4.1, Table 1.

[^66]: H. Gao, S. Chen, B. Jiang, B. Liao, Y. Shi, X. Guo, Y. Pu, H. Yin, X. Li, X. Zhang, et al. (2025) Rad: training an end-to-end driving policy via large-scale 3dgs-based reinforcement learning. arXiv preprint arXiv:2502.13144. Cited by: §2.

[^67]: T. Yuan, Y. Mao, J. Yang, Y. Liu, Y. Wang, and H. Zhao (2024) Presight: enhancing autonomous vehicle perception with city-scale nerf priors. In European Conference on Computer Vision, pp. 323–339. Cited by: §2.

[^68]: H. Lin, S. Chen, J. Liew, D. Y. Chen, Z. Li, G. Shi, J. Feng, and B. Kang (2025) Depth anything 3: recovering the visual space from any views. arXiv preprint arXiv:2511.10647. Cited by: §2.

[^69]: N. Huang, X. Wei, W. Zheng, P. An, M. Lu, W. Zhan, M. Tomizuka, K. Keutzer, and S. Zhang (2024) $\textit{S}^{3}$ Gaussian: self-supervised street gaussians for autonomous driving. arXiv preprint arXiv:2405.20323. Cited by: §2.

[^70]: B. Kerbl, G. Kopanas, T. Leimkühler, G. Drettakis, et al. (2023) 3d gaussian splatting for real-time radiance field rendering.. ACM Trans. Graph. 42 (4), pp. 139–1. Cited by: §2.

[^71]: A. Jaegle, F. Gimeno, A. Brock, A. Zisserman, O. Vinyals, and J. Carreira (2021) Perceiver: general perception with iterative attention. External Links: 2103.03206, [Link](https://arxiv.org/abs/2103.03206) Cited by: §3.4.

[^72]: S. Zeng, X. Chang, M. Xie, X. Liu, Y. Bai, Z. Pan, M. Xu, X. Wei, and N. Guo (2025) Futuresightdrive: thinking visually with spatio-temporal cot for autonomous driving. arXiv preprint arXiv:2505.17685. Cited by: §4.1, Table 1.

[^73]: Y. Zheng, P. Yang, Z. Xing, Q. Zhang, Y. Zheng, Y. Gao, P. Li, T. Zhang, Z. Xia, P. Jia, et al. (2025) World4drive: end-to-end autonomous driving via intention-aware physical latent world model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 28632–28642. Cited by: §4.1.

[^74]: S. Peng, K. Genova, C. Jiang, A. Tagliasacchi, M. Pollefeys, T. Funkhouser, et al. (2023) Openscene: 3d scene understanding with open vocabularies. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 815–824. Cited by: §4.1.

[^75]: T. Unterthiner, S. Van Steenkiste, K. Kurach, R. Marinier, M. Michalski, and S. Gelly (2018) Towards accurate generative models of video: a new metric & challenges. arXiv preprint arXiv:1812.01717. Cited by: §4.1.

[^76]: Z. Li, K. Li, S. Wang, S. Lan, Z. Yu, Y. Ji, Z. Li, Z. Zhu, J. Kautz, Z. Wu, et al. (2024) Hydra-mdp: end-to-end multimodal planning with multi-target hydra-distillation. arXiv preprint arXiv:2406.06978. Cited by: Table 1.

[^77]: B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang, et al. (2025) Diffusiondrive: truncated diffusion model for end-to-end autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 12037–12047. Cited by: Table 1.

[^78]: X. Gui, J. Zhao, W. Han, J. Wang, J. Gong, F. Tan, C. Xu, and J. Shen (2025) TrajDiff: end-to-end autonomous driving without perception annotation. arXiv preprint arXiv:2512.00723. Cited by: Table 1.

[^79]: J. Yang, K. Chitta, S. Gao, L. Chen, Y. Shao, X. Jia, H. Li, A. Geiger, X. Yue, and L. Chen (2025) Resim: reliable world simulation for autonomous driving. arXiv preprint arXiv:2506.09981. Cited by: Table 1.

[^80]: Z. Zhao, T. Fu, Y. Wang, L. Wang, and H. Lu (2025) From forecasting to planning: policy world model for collaborative state-action prediction. arXiv preprint arXiv:2510.19654. Cited by: Table 1.

[^81]: J. Zhang, Z. Fu, Z. Xu, W. Dai, Q. Liu, and Y. Wang (2026) ResWorld: temporal residual world model for end-to-end autonomous driving. arXiv preprint arXiv:2602.10884. Cited by: Table 1.

[^82]: X. Gui, M. Zhang, T. Yan, W. Han, J. Gong, F. Tan, C. Xu, and J. Shen (2026) Bridging scene generation and planning: driving with world model via unifying vision and motion representation. arXiv preprint arXiv:2603.14948. Cited by: Table 1.

[^83]: J. Li, J. Wu, D. Hu, X. Huang, B. Sun, Z. Hao, X. Lang, X. Zhu, and L. Zhang (2026) SGDrive: scene-to-goal hierarchical world cognition for autonomous driving. arXiv preprint arXiv:2601.05640. Cited by: Table 1.

[^84]: A. Blattmann, T. Dockhorn, S. Kulal, D. Mendelevitch, M. Kilian, D. Lorenz, Y. Levi, Z. English, V. Voleti, A. Letts, et al. (2023) Stable video diffusion: scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127. Cited by: Table 3.

[^85]: J. Yang, S. Gao, Y. Qiu, L. Chen, T. Li, B. Dai, K. Chitta, P. Wu, J. Zeng, P. Luo, et al. (2024) Generalized predictive model for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14662–14672. Cited by: Table 3.
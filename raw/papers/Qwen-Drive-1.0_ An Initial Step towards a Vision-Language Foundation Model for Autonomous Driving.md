---
title: "Qwen-Drive-1.0: An Initial Step towards a Vision-Language Foundation Model for Autonomous Driving"
source: "https://arxiv.org/html/2609.00111v1"
author:
published:
created: 2026-09-18
description:
tags:
  - "clippings"
---
Qwen Team Affiliation: Huazhong University of Science and Technology

###### Abstract

We present Qwen-Drive-1.0, an initial step towards a vision-language foundation model for autonomous driving. Qwen-Drive-1.0 retains the architecture of the pretrained vision-language model (VLM) and integrates 3D perception, visual question answering, and motion planning within a unified framework. An external bird’s-eye-view (BEV) perception head jointly performs 3D object detection, semantic occupancy prediction, and BEV map segmentation. It serves as a probe of the 3D information accessible from the shared representations and provides an explicit, inspectable interface to 3D scene structure. A Planning Expert conditions on shared VLM representations to generate future ego trajectories. A staged training recipe combines driving supervision with general-purpose vision-language data to acquire driving-specific competence while helping preserve broad visual understanding and instruction-following capabilities. Experiments demonstrate strong 3D perception and driving scene understanding while largely preserving general vision-language capability. Comprehensive evaluations across open-loop, pseudo-closed-loop, and closed-loop settings further show highly competitive motion-planning performance.

|  | [https://huggingface.co/Qwen/Qwen-Drive-1.0-4B](https://huggingface.co/Qwen/Qwen-Drive-1.0-4B) |
| --- | --- |
|  | [https://modelscope.cn/models/Qwen/Qwen-Drive-1.0-4B](https://modelscope.cn/models/Qwen/Qwen-Drive-1.0-4B) |
|  | [https://github.com/QwenLM/Qwen-Drive-1.0](https://github.com/QwenLM/Qwen-Drive-1.0) |

![[intro.png|Refer to caption]]

Figure 1: Performance overview of Qwen-Drive-1.0 across driving VQA, general VQA, 3D perception, and motion planning.

## Introduction

Autonomous driving research has increasingly shifted from task-specific modular pipelines towards unified learning-based systems [^31]. Within this trend, vision-language-action (VLA) models use pretrained vision-language models (VLMs) to connect scene understanding, reasoning, and action generation [^23] [^24] [^83]. Large-scale pretraining provides broad visual, linguistic, and world knowledge that can support reasoning in rare and out-of-distribution (OOD) driving scenarios.

Many recent driving VLA methods adapt a general-purpose VLM through continued training on driving-specific supervision, particularly visual question answering (VQA). This recipe expresses heterogeneous driving tasks through a common autoregressive language interface. The resulting targets cover traffic-scene description, reasoning about surrounding agents, and explanations of driving decisions [^76] [^94] [^109] [^84] [^105]. Driving-specific knowledge is therefore introduced primarily through language supervision.

This adaptation strategy has two limitations. 1) Textual VQA targets do not directly constrain 3D layout, depth, or occupancy [^96] [^18] [^108] [^89]. Even when pretrained representations encode spatial cues, textual supervision alone neither requires explicit 3D predictions nor permits their direct evaluation. A model adapted only through VQA can therefore produce fluent scene descriptions while remaining imprecise in 3D space. 2) Extensive domain adaptation can cause catastrophic forgetting of the general knowledge acquired during pretraining [^103] [^53]. No finite driving dataset can exhaustively represent the rare and unseen situations encountered in deployment. This pretrained knowledge therefore remains important for OOD reasoning. Together, these limitations motivate a unified model for 3D perception, driving reasoning, and motion planning that retains broad visual and world knowledge from pretraining.

Preserving general capability is also a deployment requirement. Production vehicles are moving towards cockpit-driving integration, in which the intelligent cockpit and the driving system share a single compute platform rather than two separate domain controllers. This consolidation lowers hardware and integration costs, and it tightens the compute budget available to each function. A single model is then expected to serve both domains, which requires general capabilities such as multi-turn dialogue, instruction following, and open-ended visual understanding in addition to driving competence. A model that trades general capability for driving performance forfeits this benefit, because the cockpit functions would then require a separate model and additional compute. Retaining general capability therefore serves two purposes. It supports reasoning in rare and unseen situations, and it allows one model to cover both the cockpit and the driving domain within a single compute budget.

We argue that a practical vision-language foundation model for driving should satisfy three design requirements. First, the pretrained VLM architecture should remain unchanged to preserve ease of use. Second, an explicit perception probe should expose and evaluate 3D scene information rather than relying on textual spatial reasoning alone. Third, the model should acquire driving scene-understanding knowledge while retaining most of its general-purpose capabilities, which supports both robust generalization and deployment on an integrated cockpit-driving platform.

We therefore introduce Qwen-Drive-1.0, the first vision-language foundation model for autonomous driving to our knowledge that unifies 3D perception, visual question answering, and motion planning within a single pretrained VLM. Qwen-Drive-1.0 uses the natively multimodal Qwen3.5-4B [^69] [^68] as the shared VLM and attaches two external modules. The bird’s-eye-view (BEV) perception head probes the shared representations through explicit and inspectable 3D scene predictions. A Planning Expert uses these representations to generate future ego trajectories. We train these components with a staged recipe that introduces perception, language, and planning objectives. A unified data pipeline underpins this recipe, mapping heterogeneous perception annotations into a shared label space, re-annotating driving VQA responses for format and factual consistency, and expressing trajectories from multiple public driving datasets in a single waypoint representation.

Fig. 1 summarizes the performance of Qwen-Drive-1.0 across 3D perception, driving scene understanding, general vision-language capabilities, and motion planning. On 3D perception, it reaches 43.95 mAP and 60.99 map mIoU on nuScenes and 43.45 mAP and 71.27 map mIoU on OpenScene, remaining highly competitive with common vision-based 3D detectors and demonstrating the explicit 3D perception capability added to the pretrained VLM. On driving scene understanding, it significantly surpasses the general-purpose Qwen3.5-4B while preserving general capability. On motion planning, it achieves a Predictive Driver Model Score of 90.7 on NAVSIM, attains a strong Rater Feedback Score of 7.91 on the test split of the Waymo Open Dataset end-to-end benchmark (WOD-E2E), and shows promising potential for closed-loop driving in AlpaSim.

We summarize our main contributions below.

- We present Qwen-Drive-1.0, to our knowledge the first vision-language foundation model for autonomous driving that integrates 3D perception, driving VQA, and motion planning without changing the pretrained VLM architecture.
- We introduce an external BEV perception head that jointly learns 3D detection, semantic occupancy prediction, and BEV map segmentation. The head serves as a 3D probe and equips the same pretrained VLM with explicit, inspectable perception outputs while preserving highly competitive vision-language performance.
- We develop a staged training and data recipe that unifies cross-dataset labels, rewrites responses, filters samples for consistency, and combines driving data with general-purpose vision-language supervision. This design supports domain adaptation while mitigating catastrophic forgetting.
- We design a Planning Expert tailored to pretrained VLM representations, using flow matching to generate future ego trajectories. Unified trajectory annotations enable joint training across multiple public driving datasets and yield highly competitive results across open-loop, pseudo-closed-loop, and closed-loop evaluations.

## Method

![[qwendrive_overview.png|Refer to caption]]

Figure 2: Unified architecture of Qwen-Drive-1.0 for 3D perception, visual question answering, and motion planning. A shared vision encoder and VLM support text generation, while the external BEV perception head and Planning Expert produce geometric predictions and future ego trajectories.

### 2.1 Model Architecture and Objectives

Fig. 2 presents the unified architecture of Qwen-Drive-1.0. A shared vision encoder and VLM process single-view and multi-view driving inputs, temporal image sequences, and general images. The vision encoder converts each image into visual tokens. The VLM encodes these tokens with the textual prompt and generates responses autoregressively. Two external modules use features from this shared pathway without changing the VLM architecture. The BEV perception head fuses vision encoder features with VLM output features to construct a BEV representation for 3D object detection, semantic occupancy prediction, and BEV map segmentation. The Planning Expert conditions trajectory tokens on cached VLM keys and values and predicts future ego motion through flow matching.

##### Multi-View and Multi-Frame Inputs.

A visual token sequence does not explicitly identify the view and timestep of each image, so we provide this information through view and frame tags. The view tags denote eight canonical directions, namely \<FRONT VIEW>, \<FRONT RIGHT VIEW>, \<RIGHT VIEW>, \<BACK RIGHT VIEW>, \<BACK VIEW>, \<BACK LEFT VIEW>, \<LEFT VIEW>, and \<FRONT LEFT VIEW>. The frame tag frame: $k$ associates each image with timestep $k$.

Input serialization depends on the task. Question answering examples use frame-major order, which places all views at one timestep before those at the next timestep:

$$
\texttt{frame: 0 <FRONT VIEW> <image> <FRONT RIGHT VIEW> <image>}\,\cdots\,\texttt{frame: 1}\,\cdots\,.
$$

Planning examples, including our self-constructed planning-reasoning data, use view-major order:

$$
\texttt{<FRONT VIEW> frame: 0 <image> frame: 1 <image>}\,\cdots\,\texttt{<FRONT RIGHT VIEW>}\,\cdots\,.
$$

View-major serialization places consecutive observations from each view adjacent in the token sequence and exposes temporal variation within that view, which is important for control in dynamic environments [^19]. Single-view and single-frame inputs omit the corresponding redundant tags. Both tag types use ordinary vocabulary tokens and require no additional special tokens or architectural modifications.

##### Autoregressive Text Generation.

Given a serialized multimodal input $\mathbf{x}$ and a target response $\mathbf{y}=(y_{1},\ldots,y_{T})$, the VLM predicts each response token conditioned on the input and preceding tokens. We use the standard next-token prediction objective:

$$
\mathcal{L}_{\mathrm{ntp}}=-\sum_{t=1}^{T}\log p\!\left(y_{t}\mid\mathbf{x},y_{<t}\right).
$$

We use the same objective for driving-specific and general-purpose vision-language samples.

##### BEV Perception Head.

As shown in Fig. 3(a), the BEV perception head performs single-frame surround-view 3D perception. It receives $N_{v}$ current images and their camera calibrations, where $N_{v}\in\{6,8\}$ in our experiments. The head constructs a shared ego-frame BEV representation for 3D object detection, semantic occupancy prediction, and BEV map segmentation.

The BEV perception head reads two complementary feature streams from each view $i$. The vision encoder feature $\mathbf{F}^{v}_{i}$ captures low-level appearance before the image tokens enter the VLM. After traversing the full VLM, the corresponding image tokens yield the feature $\mathbf{F}^{m}_{i}$, which encodes broader scene context and serves as the semantic source for BEV construction. We denote the feature sequences across views by $\mathbf{F}^{v}=(\mathbf{F}^{v}_{i})_{i=1}^{N_{v}}$ and $\mathbf{F}^{m}=(\mathbf{F}^{m}_{i})_{i=1}^{N_{v}}$. During joint training, the perception losses propagate through $\mathbf{F}^{m}_{i}$, providing an additional gradient path to the vision encoder alongside the direct path through $\mathbf{F}^{v}_{i}$.

To construct an explicit geometric representation, a depth-based view transform [^39] [^65] lifts the single-scale features $\mathbf{F}^{v}$ into a 3D volume without a feature pyramid. A lightweight depth network comprising residual blocks and an atrous spatial pyramid predicts a per-pixel categorical distribution $\mathbf{D}_{i}$ over $N_{d}$ depth bins without depth supervision. Each voxel center $\mathbf{p}$ within the perception range is projected into view $i$ using the calibration matrix $\mathbf{P}_{i}$, yielding image coordinates $(u_{i},v_{i})$ and depth bin $d_{i}$. Its voxel feature is computed as:

$$
\mathbf{V}(\mathbf{p})=\sum_{i\in\Omega(\mathbf{p})}\mathbf{D}_{i}(u_{i},v_{i},d_{i})\,\mathbf{F}^{v}_{i}(u_{i},v_{i}),
$$

where $\Omega(\mathbf{p})$ contains the views in which $\mathbf{p}$ has a valid image projection. This operation distributes image features along camera rays according to the predicted depth probabilities. The resulting volume $\mathbf{V}$ retains the height dimension for occupancy prediction.

![[head.png|Refer to caption]]

Figure 3: Architectures of the external modules. (a) The BEV perception head fuses voxelized vision encoder features with a feature pyramid of VLM outputs. (b) The Planning Expert conditions noisy trajectory tokens on cached VLM keys and values to recover a clean ego trajectory.

Because $\mathbf{F}^{m}$ is available at a single coarse scale, a simple feature pyramid [^38] expands it into multiscale features. A query-based BEV transformer [^44] [^95] then aggregates these features onto the BEV plane. Its queries are initialized using the height-collapsed feature $\bar{\mathbf{V}}$ derived from $\mathbf{V}$, which provides an explicit geometric prior. Each encoder layer alternates self-attention over the BEV grid with deformable cross-attention to the feature pyramid. The resulting ego-frame feature $\mathbf{B}$ integrates geometry from $\mathbf{V}$ with context from $\mathbf{F}^{m}$ and serves all three task-specific branches.

For 3D detection, a DETR-style decoder with deformable attention [^112] refines object queries against $\mathbf{B}$. For semantic occupancy, we expand $\mathbf{B}$ along the height dimension and fuse it with $\mathbf{V}$ before a shallow 3D UNet predicts per-voxel semantics. This fusion restores the vertical structure retained in $\mathbf{V}$. For map segmentation, a UNet-style head predicts rasterized map elements on the BEV plane. We jointly optimize the three branches using the perception objective:

$$
\mathcal{L}_{\mathrm{perc}}=\mathcal{L}_{\mathrm{det}}+\mathcal{L}_{\mathrm{occ}}+\mathcal{L}_{\mathrm{map}}.
$$

Detection follows the set-prediction formulation. The Hungarian algorithm matches object queries to ground-truth boxes, and deep supervision at each decoder layer combines a focal loss [^46] with an $\ell_{1}$ regression loss:

$$
\mathcal{L}_{\mathrm{det}}=\sum_{l=1}^{L}\Big(2\,\mathcal{L}^{(l)}_{\mathrm{focal}}+0.75\,\mathcal{L}^{(l)}_{\ell_{1}}\Big).
$$

Following FlashOcc [^98], the occupancy objective is defined as:

$$
\mathcal{L}_{\mathrm{occ}}=100\,\mathcal{L}_{\mathrm{focal}}+\mathcal{L}_{\mathrm{geo}}+\mathcal{L}_{\mathrm{sem}}+\mathcal{L}_{\mathrm{lov}},
$$

where $\mathcal{L}_{\mathrm{focal}}$ is a class-balanced focal loss, $\mathcal{L}_{\mathrm{geo}}$ and $\mathcal{L}_{\mathrm{sem}}$ are the geometric and semantic scene-class affinity losses of MonoScene [^6], and $\mathcal{L}_{\mathrm{lov}}$ is the Lovász-softmax loss [^4]. The map objective is $\mathcal{L}_{\mathrm{map}}=100\,\mathcal{L}_{\mathrm{focal}}+\mathcal{L}_{\mathrm{lov}}$.

##### Planning Expert.

For motion planning, the Planning Expert predicts future ego motion from the multimodal context encoded by the VLM. We formulate trajectory prediction as conditional generation:

$$
\boldsymbol{\tau}\sim p\!\left(\boldsymbol{\tau}\mid\mathbf{s},\ell,\boldsymbol{\tau}_{\mathrm{hist}},\mathbf{n},\mathbf{e},\mathbf{r}\right),\qquad\boldsymbol{\tau}=\{(x_{k},y_{k},\theta_{k})\}_{k=1}^{50},
$$

where $\mathbf{s}$ denotes the vehicle sensor inputs and $\ell$ denotes their serialized layout. The variables $\boldsymbol{\tau}_{\mathrm{hist}}$, $\mathbf{n}$, and $\mathbf{e}$ denote the historical ego trajectory, navigation instruction, and current ego state, respectively. The optional textual planning reason $\mathbf{r}$ is set to $\varnothing$ when unavailable. The prompt describes $\ell$, provides $\boldsymbol{\tau}_{\mathrm{hist}}$ and $\mathbf{n}$, and includes $\mathbf{r}$ when available. Each trajectory contains 50 waypoints spanning 5 s at 10 Hz. At waypoint $k$, $x_{k}$ and $y_{k}$ denote the longitudinal and lateral positions in the current ego frame, while $\theta_{k}$ denotes the heading relative to the current ego orientation. For joint training across datasets, we divide $x_{k}$, $y_{k}$, and $\theta_{k}$ by fixed scales of 165 m, 25 m, and $\pi/2$  rad, respectively.

As illustrated in Fig. 3(b), the Planning Expert uses a 32-layer diffusion transformer. The VLM alternates gated linear attention with grouped-query softmax attention [^68]. We cache the keys after rotary position embedding (RoPE) and the corresponding values from all eight grouped-query softmax attention layers. Each cache conditions four consecutive Planning Expert layers. Each Planning Expert layer concatenates the cached keys and values with those of the trajectory tokens for joint attention. A trajectory token combines a noisy waypoint with an encoding of $\boldsymbol{\tau}_{\mathrm{hist}}$. Shared adaptive layer normalization injects the flow time, navigation instruction $\mathbf{n}$, and current ego state $\mathbf{e}$. The hidden dimension is 1024, yielding $\sim$ 1.1B parameters.

We train the Planning Expert by flow matching [^47] with an $x$ -prediction parameterization that directly estimates the clean trajectory. Let $\boldsymbol{\tau}_{1}$ denote the normalized ground-truth trajectory, and let $\boldsymbol{\tau}_{0}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ denote a Gaussian noise sample of the same shape. We define the linear interpolation path at flow time $t$ as:

$$
\boldsymbol{\tau}_{t}=(1-t)\,\boldsymbol{\tau}_{0}+t\,\boldsymbol{\tau}_{1}.
$$

The Planning Expert predicts the clean endpoint $\hat{\boldsymbol{\tau}}_{1}$ rather than the flow velocity or noise. The predicted endpoint induces the flow velocity field $(\hat{\boldsymbol{\tau}}_{1}-\boldsymbol{\tau}_{t})/(1-t)$. This endpoint parameterization reduces sensitivity to sensor noise in trajectories recorded across heterogeneous datasets. To keep this conversion well conditioned, we sample $\tilde{t}\sim\mathrm{Beta}(1.5,1.0)$ and set $t=\min\{\tilde{t},0.9\}$, ensuring $1-t\geq 0.1$. The complete objective combines flow matching with temporal regularization:

$$
\mathcal{L}_{\mathrm{plan}}=\mathcal{L}_{\mathrm{fm}}+2\times 10^{-4}\,\mathcal{L}_{\Delta^{1}}+2\times 10^{-5}\,\mathcal{L}_{\Delta^{2}},
$$

where $\mathcal{L}_{\mathrm{fm}}$ is the squared error between the induced flow velocity and the target flow velocity $\boldsymbol{\tau}_{1}-\boldsymbol{\tau}_{0}$. For $j\in\{1,2\}$, $\mathcal{L}_{\Delta^{j}}$ is a Huber penalty that matches the $j$ th-order temporal differences of $\hat{\boldsymbol{\tau}}_{1}$ and $\boldsymbol{\tau}_{1}$. Together, these temporal regularizers discourage waypoint jitter and abrupt changes in acceleration.

At inference, Gaussian noise initializes the trajectory tokens. A 10-step Euler solver then integrates the induced flow velocity field to obtain the final trajectory.

### 2.2 Training Recipe

As illustrated in Fig. 4, we train Qwen-Drive-1.0 in four stages. The first two stages initialize the BEV perception head and jointly adapt the shared pathway for explicit 3D prediction. Stage 3 trains trajectory generation with optional textual reasoning as a condition. Stage 4 further refines the resulting model through reinforcement-based optimization.

![[training_recipe.png|Refer to caption]]

Figure 4: Four-stage training recipe of Qwen-Drive-1.0. Stages 1 and 2 adapt the shared vision-language pathway, first initializing the BEV perception head and then using perception and VQA supervision to update the vision encoder and VLM. Stages 3 and 4 train the Planning Expert on top of these fixed representations, first by flow matching and then by reward-based optimization. Flames indicate trainable modules, and snowflakes indicate fixed modules.

##### Stage 1. Perception Head Pretraining.

We keep the vision encoder and VLM fixed and optimize only the newly initialized BEV perception head with $\mathcal{L}_{\mathrm{perc}}$. Its view transform, BEV transformer, and task decoders learn to construct and decode an ego-frame representation. This stage initializes the newly added module before joint adaptation in Stage 2.

##### Stage 2. Perception and VQA Joint Training.

Sec. 3.1 will show that head-only training yields limited perception performance, indicating that the pretrained representations do not directly expose sufficient 3D structure for driving perception. The head-only setting thus probes how readily the pretrained features support explicit 3D prediction. We then optimize the initialized BEV perception head, vision encoder, and VLM together for this capability. Perception samples use $\mathcal{L}_{\mathrm{perc}}$, while vision-language samples use $\mathcal{L}_{\mathrm{ntp}}$.

Each minibatch contains both sample types. Because they activate different task pathways, we provide dummy inputs to inactive branches to maintain a consistent computation graph across distributed workers. We exclude the corresponding dummy outputs from the loss. The BEV perception head uses a learning rate $20\times$ that of the VLM, allowing the task-specific module to adapt more rapidly during joint training. Driving data provide domain-specific supervision, while general-purpose vision-language data help preserve broad visual understanding and instruction-following capabilities. The resulting VLM representations provide the conditions for the Planning Expert.

##### Stage 3. Planning Expert Pretraining.

We keep the vision encoder and VLM fixed and optimize only the Planning Expert. The training mixture contains samples whose prompts include a textual planning reason $\mathbf{r}$ and samples for which $\mathbf{r}=\varnothing$. Both types supervise only the future trajectory through $\mathcal{L}_{\mathrm{plan}}$, with no text-generation objective in this stage. Keeping the conditioning representations fixed separates trajectory learning from changes in the vision-language representations. We refer to the resulting model as Qwen-Drive-1.0-SFT.

##### Stage 4. Reinforcement Learning.

Stage 3 trains the Planning Expert to reproduce a single recorded future per scene using $\mathcal{L}_{\mathrm{plan}}$. This imitation objective provides stable trajectory supervision but only partially reflects how a plan is evaluated in practice. A recorded trajectory represents only one of several acceptable futures, so other safe behaviors may be penalized, particularly when the four training sources exhibit different ego-motion distributions. Moreover, trajectory regression does not explicitly capture collision avoidance, drivable-area compliance, progress, or agreement with human preference. This stage therefore optimizes the Planning Expert with task-level rewards that measure these properties. We keep the vision encoder and VLM fixed, confining the adaptation to the Planning Expert and preserving the shared representations learned in Stage 2. We refer to the resulting model as Qwen-Drive-1.0-RL.

These task-level rewards are nondifferentiable through trajectory generation and therefore require sampled rollouts for optimization. The inference sampler in Sec. 2.1 initializes the trajectory tokens from Gaussian noise and then applies deterministic Euler integration. Once the initial noise is drawn, the remaining integration path is deterministic and defines no transition probabilities that a policy gradient could differentiate. We therefore share the initial trajectory noise within each rollout group and introduce stochastic transitions over a contiguous block of the final integration steps. This converts the deterministic flow into a stochastic policy whose transition likelihood depends on the Planning Expert parameters. Indexing the $K=10$ Euler steps from zero, with $t_{k}=k/K$ and $\Delta t=1/K$, we introduce stochasticity only over the final three transitions, $\mathcal{W}=\{7,8,9\}$. Under the endpoint parameterization, perturbations near $t=1$ affect the emitted trajectory more directly, while earlier perturbations are increasingly attenuated by subsequent integration steps. Concentrating exploration near the output therefore yields effective trajectory diversity while limiting deviation from the pretrained flow. Stochastic perturbations can move an intermediate trajectory away from regions favored by the pretrained flow. We therefore construct an approximate restoring score from the Gaussian conditional associated with the interpolation in Eq. 7. Under the Gaussian conditional implied by Eq. 7, we substitute the predicted endpoint $\hat{\boldsymbol{\tau}}_{1}^{(k)}$ for the unknown clean trajectory and obtain the score correction:

$$
s_{\theta}\left(\boldsymbol{\tau}^{(k)},t_{k}\right)=\nabla_{\boldsymbol{\tau}^{(k)}}\log p_{t_{k}}\!\left(\boldsymbol{\tau}^{(k)}\mid\hat{\boldsymbol{\tau}}_{1}^{(k)}\right)=-\frac{\boldsymbol{\tau}^{(k)}-t_{k}\hat{\boldsymbol{\tau}}_{1}^{(k)}}{(1-t_{k})^{2}}.
$$

This score points toward the conditional center and stabilizes perturbed states. Let $\sigma_{k}$ denote the standard deviation of one discrete stochastic transition. For a continuous diffusion coefficient $g(t)$, the corresponding discrete standard deviation is $\sigma_{k}=g(t_{k})\sqrt{\Delta t}$. The score drift accumulated over one Euler interval is therefore $\frac{1}{2}g(t_{k})^{2}s_{\theta}\Delta t=\frac{1}{2}\sigma_{k}^{2}s_{\theta}$. The resulting transition mean is:

$$
\boldsymbol{\mu}^{(k)}=\boldsymbol{\tau}^{(k)}+v_{\theta}\left(\boldsymbol{\tau}^{(k)},t_{k}\right)\Delta t+\frac{\sigma_{k}^{2}}{2}s_{\theta}\left(\boldsymbol{\tau}^{(k)},t_{k}\right).
$$

We set $\sigma_{k}=\sigma=0.03$ for $k\in\mathcal{W}$ and $\sigma_{k}=0$ otherwise. Since $\sigma_{k}$ denotes the standard deviation of the discrete transition, $\sigma_{k}^{2}$ already incorporates the integration interval and requires no additional factor of $\Delta t$ in the score correction. In implementation, $1-t_{k}$ is lower-bounded by $\epsilon=0.1$, and $\hat{\boldsymbol{\tau}}_{1}^{(k)}$ is clipped to $[-1,1]$ in normalized coordinates before computing the flow velocity and restoring score.

Independent waypoint noise primarily introduces high-frequency jitter rather than meaningful maneuver diversity, making the resulting samples poorly suited to comparisons of driving quality. We therefore restrict stochastic exploration to a smooth low-frequency temporal subspace. Let $\boldsymbol{\Phi}\in\mathbb{R}^{N\times M}$ contain the first $M=6$ orthonormal cosine modes over the $N=50$ future waypoints, with $\boldsymbol{\Phi}^{\top}\boldsymbol{\Phi}=\mathbf{I}_{M}$. At each stochastic transition, we sample $\mathbf{Z}_{k}\in\mathbb{R}^{M\times 3}$ with independent standard Gaussian entries and update by:

$$
\boldsymbol{\tau}^{(k+1)}=\boldsymbol{\mu}^{(k)}+\sigma_{k}\boldsymbol{\Phi}\mathbf{Z}_{k}.
$$

The low-frequency modes vary smoothly over the prediction horizon, so their combinations produce coherent shifts and bends in the trajectory rather than pointwise oscillations. Since $\boldsymbol{\Phi}$ has orthonormal columns, the perturbation of an individual waypoint has an average standard deviation of $\sigma\sqrt{M/N}\approx 0.010$ across waypoints, corresponding to roughly $1.7$  m longitudinally and $0.26$  m laterally per stochastic step. The perturbation lies in the $3M$ -dimensional subspace of the full $3N$ -dimensional trajectory space. We therefore evaluate a Gaussian likelihood surrogate for the injected stochastic action in the corresponding low-dimensional basis coordinates, rather than treating the transition as a full-rank density in trajectory space. Using $\boldsymbol{\Phi}^{\top}\boldsymbol{\Phi}=\mathbf{I}_{M}$, this likelihood surrogate takes the form:

$$
\log\pi_{\theta}\left(\boldsymbol{\tau}^{(k+1)}\mid\boldsymbol{\tau}^{(k)}\right)=-\frac{1}{2\sigma_{k}^{2}}\left\|\boldsymbol{\Phi}^{\top}\left(\boldsymbol{\tau}^{(k+1)}-\boldsymbol{\mu}^{(k)}\right)\right\|_{F}^{2}+\mathrm{const}.
$$

The implementation averages the squared residual over the $3M$ mode coefficients instead of summing them, which rescales $\mathcal{L}_{\mathrm{rl}}$ by a constant factor of $1/(3M)$ and is absorbed into the learning rate. Since the score correction in Eq. 10 is derived for isotropic diffusion while our perturbation is restricted to a low-dimensional subspace, we interpret it as an approximate restoring correction rather than an exact marginal-preserving transformation.

For each scene, the frozen VLM samples $G=8$ reasoning traces, whose cached keys and values independently condition the $G$ trajectory rollouts from the Planning Expert. The group rollouts share the same initial trajectory noise $\boldsymbol{\tau}^{(0)}$ and sample independent low-frequency perturbations within $\mathcal{W}$, leaving stochastic transitions and sampled reasoning as the sources of within-group diversity. Given rollout rewards $\{R_{i}\}_{i=1}^{G}$, we compute the group-relative advantage as:

$$
A_{i}=\frac{R_{i}-\bar{R}}{\sigma_{R}+\epsilon_{R}},\qquad\bar{R}=\frac{1}{G}\sum_{j=1}^{G}R_{j},
$$

where $\sigma_{R}$ is the population standard deviation over the group and $\epsilon_{R}$ =1e-8 is used for numerical stability. This group-relative advantage provides a baseline without a learned value function [^73]. During optimization, the sampled states and advantages are treated as constants, while the transition means are recomputed with the current Planning Expert. Only the stochastic transitions in $\mathcal{W}$ contribute to the objective. Writing $k_{w}$ for the $w$ -th element of $\mathcal{W}$ in ascending order, we optimize the Planning Expert with a discounted policy gradient over the $W=|\mathcal{W}|$ stochastic steps:

$$
\mathcal{L}_{\mathrm{rl}}=-\frac{1}{GW}\sum_{i=1}^{G}\sum_{w=0}^{W-1}\gamma^{\,W-1-w}A_{i}\log\pi_{\theta}\!\left(\boldsymbol{\tau}_{i}^{(k_{w}+1)}\mid\boldsymbol{\tau}_{i}^{(k_{w})}\right),
$$

where $\gamma=0.6$ assigns greater credit to the steps closest to the output. Each rollout group is sampled and consumed by a single on-policy update. The objective therefore uses the current model log-likelihood directly and requires no off-policy importance correction.

Multi-source reinforcement learning must accommodate each benchmark’s distinct evaluation criteria. We use the Predictive Driver Model Score (PDMS) for NAVSIM and the Rater Feedback Score for WOD-E2E, and add a shared displacement term to each source so that a single policy receives a comparable learning signal from all three. Appendix A provides the exact reward definitions. Training uses 15K NAVSIM scenes drawn from navtrain and balanced across navigation commands, 15K PhysicalAI-AV (PAI-AV) scenes, and 479 WOD-E2E scenarios with rater-preference annotations.

### 2.3 Data Recipe

We organize the training data into perception, vision-language, and planning groups and align heterogeneous task definitions within each group before mixing sources.

#### 2.3.1 Perception Data

We use nuScenes [^5] and OpenScene [^62] for single-frame surround-view perception. For nuScenes, we use the semantic occupancy labels provided by nuScenes-OccNet [^81]. nuScenes provides six camera views, whereas OpenScene provides eight. We adopt the official nuScenes split, which provides 28K annotated keyframes for training and 6K for validation. For OpenScene, we hold out 16 logs from the trainval pool to balance city and time of day. This split provides 607K training frames and 9K validation frames.

Joint training requires complementary unification at the data and model levels. At the data level, we must reconcile task taxonomies across sources. At the model level, the predicted features must be aligned with dataset-specific spatial grids and coordinate systems. We describe these two procedures below.

##### Label unification.

We first establish shared task taxonomies across the two sources. Their annotations differ in granularity and class coverage, which prevents direct mixing. We therefore align the taxonomies at the coarsest mutually compatible granularity. Categories annotated by only one source remain source-specific. The loss for each such class is computed only on samples from its source unless reliable auxiliary annotations permit offline completion for the other source. Fig. 5 summarizes these task-specific procedures and shows representative occupancy labels before and after processing. The corresponding label spaces and processing rules are detailed below.

![[occ_process_vis.png|Refer to caption]]

Figure 5: Cross-dataset label unification for perception. (a) Task-specific alignment and label-completion strategies. (b) Original and processed occupancy labels for nuScenes (top) and OpenScene (bottom).

- 3D detection. We adopt the annotation granularity of OpenScene. The seven classes are vehicle, bicycle, generic\_object, pedestrian, traffic\_cone, barrier, and czone\_sign. For nuScenes, we merge car, truck, trailer, bus, and construction vehicle into vehicle, and merge bicycle and motorcycle into bicycle. We also map debris, pushable and pullable objects, bicycle racks, animals, and related categories to generic\_object. The pedestrian, traffic\_cone, and barrier classes correspond directly across the two datasets, whereas only OpenScene samples supervise czone\_sign.
- Semantic occupancy. Dataset-specific lookup tables map the 17 nuScenes classes and the original OpenScene labels into a shared ten-class space. This space comprises the seven detection classes, driveable, background, and empty. For nuScenes, we map the five vehicle categories to vehicle, bicycle and motorcycle to bicycle, and driveable surface to driveable. Categories without cross-dataset correspondence, including other flat surfaces, sidewalks, terrain, man-made structures, and vegetation, are mapped to background. Free space is mapped to empty. For OpenScene, we map the foreground classes directly. Background surfaces and reserved labels are mapped to background, while unknown and free-space labels are mapped to empty.
- Offline label completion. We use reliable auxiliary annotations to complete categories missing from one source, without applying ray masking. The original OpenScene occupancy labels do not distinguish driveable. We rasterize the nuPlan vector map and relabel a voxel as driveable only if it is a ground voxel inside a driveable region and was originally labeled background. nuScenes does not provide occupancy annotations for generic\_object. We generate pseudo-labels from the 3D boxes of bicycle racks, debris, and pushable and pullable objects. Within these boxes, we change only voxels that already carry semantic labels. After remapping and completion, we recompute class frequencies in the unified label space for the class-balanced focal loss.
- BEV map segmentation. We rasterize both vector maps online under a shared six-class schema instead of remapping existing raster labels. The schema contains driveable surface, road line, road edge, crosswalk, walkway, and background.

Label unification does not remove noise from the source annotations. nuScenes provides manually annotated semantics, whereas OpenScene reconstructs occupancy targets from aggregated LiDAR sweeps using an automated pipeline without per-point semantics. Sensor and registration errors can therefore introduce artifacts, including floating voxels detached from physical surfaces. Offline completion adds missing semantic labels but retains these artifacts.

##### Spatial unification.

Model-level unification poses a separate spatial dilemma. Both sources store occupancy as $200\times 200\times 16$ voxel grids, yet corresponding voxel indices represent different physical locations. nuScenes-OccNet spans $\pm 40$  m horizontally and $z\in[-1.0,5.4]$  m with 0.4 m voxels, while OpenScene spans $\pm 50$  m and $z\in[-4.0,4.0]$  m with 0.5 m voxels. Their coordinate transformations also differ. The nuScenes LiDAR has a non-identity transform to the rear-axle ego frame, including a vertical offset of $\sim$ 1.84 m, while OpenScene uses an identity LiDAR-to-ego transform. Directly sharing voxel indices would misalign the two sources, while resampling categorical labels would distort the supervision. We therefore preserve each source’s native occupancy grid and perform spatial alignment on the predicted features.

The depth-based view transform first lifts image features into a 3D volume defined over the detection range. Before occupancy decoding, a single differentiable trilinear sampling operation maps this volume onto the dataset-specific occupancy grid. For nuScenes, each target voxel center is defined in the ego frame and transformed back into the LiDAR frame for sampling. The same operation restricts the output to the occupancy range, while an expanded vertical source range of $[-5.0,5.4]$  m covers the LiDAR mounting offset. For OpenScene, the identity transform requires no frame conversion. The head selects the corresponding occupancy range and voxel size for each source during the forward pass. The same occupancy head can therefore predict on both native grids without dataset-specific branches. Both sources supervise this head under a consistent ego-frame convention while retaining their native spatial resolution. BEV map supervision is unified separately. For both sources, we rasterize the vector maps online over an ego-centered local patch with $x\in[-30,30]$  m and $y\in[-15,15]$  m at 0.15 m resolution, producing a $400\times 200$ target rather than rasterizing the full city map.

#### 2.3.2 Vision-Language Data

The vision-language data combine general-purpose and driving examples. The driving component covers scene understanding, spatial grounding, cross-view reasoning, and planning reasoning.

##### Data Sources.

The driving component combines open-source datasets with self-constructed examples that target underrepresented tasks. We detail the composition and preprocessing of these two source groups below. Fig. 6 shows the filtered public driving data, their scene diversity, and the Stage 2 mixture.

- Open-Source Driving Data. We aggregate 24 publicly available driving vision-language datasets, including CODA-LM [^8], DRAMA [^55], DriveAction [^29], DriveGPT4 [^94], DriveLM [^76], DrivingVQA [^13], Impromptu VLA [^11], LingoQA [^56], MapLM [^7], MM-AU [^20], NAVSIM-ReCogDrive [^40], NuInstruct [^16], NuPlanQA [^63], nuScenes-MQA [^33], nuScenes-QA [^66], the OOD reasoning-label subset of PhysicalAI-AV [^61], OmniDrive [^84], ROADWork [^26], Senna [^34], STSBench [^22], SURDS [^27], SUTD-TrafficQA [^92], Talk2Car [^15], and WaymoQA [^97]. We use training splits for these datasets to avoid data leakage.
	The released datasets differ in conversational format and annotation reliability, and many targets originate from templates or automated pipelines. Before mixing the datasets, we use Qwen3.5-Plus [^69] to rewrite each prompt and response into a common conversational schema. The source annotation remains the reference for subsequent consistency filtering. We convert most multiple-choice questions to open-ended QA but retain a subset in multiple-choice form to preserve this instruction type. We normalize bounding boxes to $[0,1000)$ image coordinates, insert view and frame tags following Sec. 2.1, and revise textual view references to match the inserted tags.
	Rewriting standardizes the format but does not validate the source annotation. We therefore apply a separate consistency filter. Qwen3.5-Flash evaluates whether each rewritten response is semantically consistent with its source annotation, and we retain only samples classified as consistent. As shown in Fig. 6(a), this procedure reduces the data from 5.53M to 3.09M samples, corresponding to a retention rate of 55.9%. The retained set contains 61.6% multi-view, 20.1% single-view, 10.1% single-view temporal, 4.4% multi-view temporal, and 3.7% video samples. It covers scene and region captioning, open-ended and multiple-choice QA, 2D grounding, spatial reasoning, and planning reasoning. The representative samples in Fig. 6(c) further illustrate the diversity of road environments, illumination, and weather conditions.
	![[data_analysis.png|Refer to caption]]
	Figure 6: Vision-language data and the Stage 2 training mixture. (a) Input-format distribution of the 3.09M filtered public driving samples before Stage 2 subsampling. (b) Composition of the 1.54M Stage 2 training set before repetition. (c) Representative scenes spanning diverse road environments, illumination, and weather conditions.
- Self-Constructed Driving Data. We construct three additional components to provide supervision that is underrepresented in the public datasets.
	1) Planning reasoning. We construct planning-reasoning data using a Chain-of-Causation (CoC) formulation inspired by Alpamayo-R1 [^86]. A CoC trace identifies the scene elements that motivate a driving decision and explains their causal roles. We build the data from three publicly available driving datasets with future ego trajectories, namely NAVSIM [^14], Waymo [^77], and PAI-AV [^61]. From each future ego trajectory, a rule-based classifier derives longitudinal and lateral maneuver components that jointly form a motion prior. Qwen3.7-Plus [^70] then generates a planning-reasoning trace conditioned on the multi-view images, historical ego trajectory, motion prior, and navigation instruction.
	Each generated trace undergoes a multistage audit. Rather than requesting scalar quality scores, judge models answer classification questions, and their decisions are aggregated programmatically. The audit checks the predicted maneuver against the ground-truth trajectory, classifies the causal role of each cited factor, and rejects traces that reveal future information. Qwen3.5-Flash further assigns a rarity score to each scene, and rare scenes receive higher sampling priority. For each accepted trace, we construct two response formats. The first contains only the planning-reasoning trace. The second contains the same trace followed by the future ego trajectory serialized in JSON format.
	2) Camera ordering for cross-view spatial understanding. We shuffle the surround-view images and remove all view tags. The model identifies the front view from visual cues and recovers the clockwise order of the complete camera set.
	3) In-house perception QA. We construct 30K examples from road scenes collected in China for traffic-light grounding and 3D object detection. In the 3D detection examples, the camera pose is provided in text, and the response contains 3D boxes in the global coordinate frame.

The public sources provide broad coverage of scenes and question types, while the self-constructed data provide targeted supervision for causal reasoning and perception grounding.

##### Stage 2 Data Mixture.

Stage 2 mixes the perception and vision-language data described above. Because the filtered public driving data exceed the budget for this stage, we sample $\sim$ 20% from each public source, stratified by task and question type. We combine this subset with the self-constructed driving data and general-purpose vision-language data. Before repetition, the resulting set contains 1.54M examples. As shown in Fig. 6(b), 9.7% provide 3D perception supervision, 26.0% provide general-purpose vision-language supervision, and 64.3% provide driving vision-language supervision.

We use group-specific repetition factors, assigning a larger factor to perception examples to increase updates to the BEV perception head and repeating vision-language examples for two to three epochs. After repetition, the effective mixture contains 12.7% perception, 31.0% general-purpose vision-language, and 56.3% driving vision-language supervision.

#### 2.3.3 Planning Data

For Stage 3, we use NAVSIM [^14], OpenScene [^62], WOD-E2E [^93], and PAI-AV [^61]. NAVSIM and OpenScene are both derived from the nuPlan dataset [^36], but remain separate sources because their ego-motion distributions differ. The resulting training set contains $\sim$ 2.83M samples. NAVSIM and OpenScene jointly contribute 890K samples from 2.5K clips. WOD-E2E contributes 557K samples from 2K clips, while PAI-AV contributes 1.38M samples from 156K clips. Of the training samples, 685K (24.2%) include an accepted planning-reasoning trace as a condition. These samples comprise 78K from NAVSIM, 142K from WOD-E2E, and 465K from PAI-AV. The remaining 75.8% omit this condition and use $\mathbf{r}=\varnothing$.

During preprocessing, we express every future trajectory in the current ego frame and convert it to the 50-waypoint representation in Eq. 6. For NAVSIM and OpenScene, we read positions and headings at 10 Hz directly from the nuPlan database. For PAI-AV, we recover future ego motion from the per-clip egomotion annotations and resample it on the same temporal grid. Because this source is far larger than the others, we retain only a small number of evenly spaced frames from each clip.

WOD-E2E provides future ego positions at 4 Hz. We prepend the current position, fit a time-parameterized natural cubic spline to the position sequence, and evaluate it on the 10 Hz grid. Its first and second derivatives provide the future velocities and accelerations. For historical acceleration, we apply a factor-of-four scale correction to the raw accel\_x and accel\_y metadata and linearly resample the corrected values at 10 Hz. Let $\mathbf{v}=(v_{x},v_{y})$ and $\mathbf{a}=(a_{x},a_{y})$ denote the spline-derived velocity and acceleration. For $\|\mathbf{v}\|\geq 0.3\,\mathrm{m/s}$, the induced heading rate is $\dot{\theta}=(v_{x}a_{y}-v_{y}a_{x})/\|\mathbf{v}\|^{2}$. We set $\dot{\theta}=0$ below this speed, integrate it on the 10 Hz grid, and set the current heading to zero. The resulting future headings provide the $\theta_{k}$ values in Eq. 6. We retain samples only when the historical and future acceleration magnitudes do not exceed standard gravity ($9.8\,\mathrm{m/s^{2}}$). For the unwrapped future heading sequence, we apply complementary derivative-based and adjacent-step checks to capture both rate consistency and abrupt local changes. Both checks use a threshold of $1.2\,\mathrm{rad/s}$.

Each planning example contains front, front-left, and front-right images at four timesteps. These comprise the current and three historical timesteps sampled at 0.5 s. The images form the vehicle sensor inputs $\mathbf{s}$. The historical ego trajectory $\boldsymbol{\tau}_{\mathrm{hist}}$, current ego state $\mathbf{e}$, and navigation instruction $\mathbf{n}$ provide the remaining conditions in Eq. 6. Accepted planning-reasoning traces provide $\mathbf{r}$ for a subset of samples, while $\mathbf{r}=\varnothing$ for the remainder. For these samples, $\ell$ is the view-major layout defined in Sec. 2.1. We resize historical images to 320p and current images to 720p. This allocation retains more spatial detail in the current observation while representing the motion history with fewer visual tokens.

## Experiments

We evaluate Qwen-Drive-1.0 across 3D perception, driving, general vision-language understanding, and motion planning. The perception and vision-language evaluations use Qwen-Drive-1.0-SFT, while we evaluate motion planning with both Qwen-Drive-1.0-SFT and Qwen-Drive-1.0-RL.

### 3.1 3D Perception

##### Metrics.

We evaluate on the official nuScenes validation set and the 16-log OpenScene validation split defined in Sec. 2.3. For 3D detection, we report mean average precision (mAP) and a modified nuScenes detection score (NDS) over the unified seven-class label space. We match a prediction to a ground-truth box when their BEV center distance is below a specified threshold. We compute AP as the normalized area under the precision-recall curve after discarding operating points with precision or recall below 10%. The mAP averages AP over the center-distance thresholds $\{0.5,1,2,4\}$  m and all classes. For NDS, we compute the true-positive error metrics at the 2 m threshold following the official class-specific definitions. Because the unified annotations contain no attribute labels, our modified NDS sets the mean attribute error (mAAE) to zero for all methods.

For semantic occupancy, we report the mean IoU over the unified semantic classes excluding empty (Occ mIoU), and RayIoU [^48]. Notably, OpenScene stores occupancy in the rear-axle ego frame with an identity LiDAR-to-ego transform. Rays cast from the rear axle would therefore originate at road level and immediately terminate on road-surface voxels. We instead raise the OpenScene ray origin by 1.84 m to match the nuScenes LiDAR mounting height. For BEV map segmentation, we report the mean IoU over the foreground map classes.

##### Comparison Methods.

To enable fair comparisons, we reproduce the single-frame variants of BEVFormerV2 [^95], PETR [^49], and PETRv2 [^50] on the remapped nuScenes annotations. Our BEVFormerV2 implementation excludes the Group-DETR decoder and its associated 2D losses. We further construct BEVFormerV2 <sup>∗</sup> as a unified multi-task variant that jointly performs 3D detection, BEV map segmentation, and semantic occupancy prediction, covering the same three perception tasks as our BEV perception head. We evaluate each architecture with ResNet-50 [^30] and SigLIP-Qwen. SigLIP-Qwen denotes the SigLIP-style [^102] vision encoder initialized with the same pretrained Qwen3.5-4B weights used by Qwen-Drive-1.0. All comparison methods follow a 24-epoch schedule. Both the comparison methods and our BEV perception head use an input resolution of $896\times 512$. In contrast, our BEV perception head uses no rig-specific camera embeddings, so a single model trains and evaluates across the six-camera nuScenes rig and the eight-camera OpenScene rig.

Table 1: Unified 3D perception on the remapped nuScenes and OpenScene validation splits. Comparison methods are trained only on remapped nuScenes under a common schedule and input resolution. Dashes indicate tasks that a method does not perform. The comparison methods learn camera embeddings tied to the six-camera nuScenes rig and therefore cannot be evaluated on the eight-camera OpenScene rig. Gray values denote cross-dataset evaluation of the nuScenes-only head on OpenScene. NDS denotes the modified score with mAAE set to zero because the unified labels contain no attributes. <sup>∗</sup> denotes the unified multi-task BEVFormerV2 variant with map segmentation and occupancy prediction.

<table><tbody><tr><td rowspan="3">Method</td><td rowspan="3">Visual Encoder</td><td colspan="5">nuScenes</td><td colspan="5">OpenScene</td></tr><tr><td colspan="2">3D Det</td><td>Map</td><td colspan="2">Occ</td><td colspan="2">3D Det</td><td>Map</td><td colspan="2">Occ</td></tr><tr><td>mAP</td><td>NDS</td><td>mIoU</td><td>mIoU</td><td>RayIoU</td><td>mAP</td><td>NDS</td><td>mIoU</td><td>mIoU</td><td>RayIoU</td></tr><tr><td>BEVFormerV2</td><td>ResNet-50</td><td>33.04</td><td>33.02</td><td>–</td><td>–</td><td>–</td><td colspan="5">–</td></tr><tr><td>PETR</td><td>ResNet-50-DCN</td><td>29.77</td><td>26.65</td><td>–</td><td>–</td><td>–</td><td colspan="5">–</td></tr><tr><td>PETRv2</td><td>ResNet-50-DCN</td><td>25.98</td><td>23.90</td><td>52.52</td><td>–</td><td>–</td><td colspan="5">–</td></tr><tr><td>BEVFormerV2 <sup>∗</sup></td><td>ResNet-50</td><td>35.34</td><td>30.76</td><td>48.49</td><td>23.39</td><td>40.69</td><td colspan="5">–</td></tr><tr><td>BEVFormerV2</td><td>SigLIP-Qwen</td><td>40.78</td><td>39.78</td><td>–</td><td>–</td><td>–</td><td colspan="5">–</td></tr><tr><td>PETR</td><td>SigLIP-Qwen</td><td>37.61</td><td>34.37</td><td>–</td><td>–</td><td>–</td><td colspan="5">–</td></tr><tr><td>PETRv2</td><td>SigLIP-Qwen</td><td>36.10</td><td>33.28</td><td>57.62</td><td>–</td><td>–</td><td colspan="5">–</td></tr><tr><td>BEVFormerV2 <sup>∗</sup></td><td>SigLIP-Qwen</td><td>41.94</td><td>36.46</td><td>47.76</td><td>25.72</td><td>43.89</td><td colspan="5">–</td></tr><tr><td>Head-only (nuScenes)</td><td>SigLIP-Qwen</td><td>35.60</td><td>34.13</td><td>55.55</td><td>20.21</td><td>36.98</td><td>16.14</td><td>16.50</td><td>40.45</td><td>11.50</td><td>17.43</td></tr><tr><td>Head-only (joint)</td><td>SigLIP-Qwen</td><td>33.49</td><td>33.37</td><td>51.15</td><td>14.83</td><td>29.54</td><td>40.57</td><td>41.86</td><td>66.34</td><td>20.13</td><td>25.36</td></tr><tr><td>Qwen-Drive-1.0-SFT</td><td>SigLIP-Qwen</td><td>43.95</td><td>42.83</td><td>60.99</td><td>19.82</td><td>37.02</td><td>43.45</td><td>44.16</td><td>71.27</td><td>19.84</td><td>25.17</td></tr></tbody></table>

##### Quantitative Results.

As shown in Tab. 1, on nuScenes, Qwen-Drive-1.0-SFT establishes the best detection and map-segmentation results, exceeding BEVFormerV2 <sup>∗</sup> by 2.01 mAP, BEVFormerV2 by 3.05 NDS, and PETRv2 by 3.37 map mIoU. On OpenScene, it also improves over Head-only (joint) in both detection metrics and map mIoU. Replacing ResNet-50 with the pretrained ViT consistently benefits the dedicated detectors, confirming that vision-language pretraining provides a strong visual initialization. However, a converged head trained on the same SigLIP-Qwen features still trails BEVFormerV2 <sup>∗</sup> by 6.34 mAP and 6.91 RayIoU. This contrast indicates that vision-language-pretrained features support visual-text alignment but do not directly expose the 3D structure required for driving perception.

Cross-dataset transfer presents a separate challenge. Despite training on the higher-quality nuScenes annotations, the nuScenes-only head reaches only 16.50 NDS on OpenScene, less than half of its 34.13 nuScenes score. Mixed-source training raises OpenScene NDS to 41.86, but the residual differences between label semantics and annotation pipelines introduce negative transfer on nuScenes. The effect is most pronounced for occupancy, where mIoU decreases by 26.6%. The same source mismatch persists after joint adaptation: Stage 2 substantially recovers nuScenes occupancy but changes both OpenScene occupancy metrics by less than 0.3 points, despite clear improvements in detection and map segmentation. Unlike boxes and rasterized map layers, the machine-generated OpenScene voxel labels retain source-specific semantic and construction artifacts, which limit the benefit of joint adaptation. This observation explains why Qwen-Drive-1.0-SFT leads in detection and map segmentation but not occupancy. Label mapping and offline completion therefore make joint training feasible without fully resolving source ambiguity at the voxel level.

The gains from Stage 2 cannot be explained by continued head optimization alone, since the head-only model had already converged. Once the perception objectives are allowed to update the pretrained ViT encoder and VLM, nuScenes mAP and map mIoU increase by 10.46 and 9.84 points, while OpenScene map mIoU gains a further 4.93 points. Joint adaptation is therefore important for realizing the 3D perception capability of the external head. Together with the ablation in Tab. 8, these results show that the head provides a practical 3D probe and that targeted adaptation produces explicit, inspectable scene predictions while preserving highly competitive vision-language performance. This capability is added without changing the VLM architecture.

![[perception_vis.png|Refer to caption]]

Figure 7: Qualitative results of Qwen-Drive-1.0-SFT on the OpenScene (a, b) and nuScenes (c, d) validation splits. Each row shows 3D detection, semantic occupancy, and BEV map segmentation results.

##### Qualitative Results.

Fig. 7 shows that Qwen-Drive-1.0-SFT produces competitive 3D perception results within the camera-visible regions under both the six-view and the eight-view rig. Notably, in Fig. 7(b), our preprocessing does not fully remove floating voxels above the road surface from the machine-generated OpenScene label, while the prediction suppresses these artifacts and correctly recovers the road-surface semantics. This example suggests that supervision across scenes and data sources can reduce sensitivity to residual artifacts in processed labels.

### 3.2 Driving and General Vision-Language Understanding

#### 3.2.1 Driving Visual Question Answering

##### Benchmark Selection.

Existing driving VQA benchmarks present three issues that obscure a model’s true driving competence. 1) Annotation quality is uneven, and many reference answers cannot be inferred from the visual input alone. 2) Text-similarity scores reward surface agreement with the reference wording and therefore measure fit to the training distribution rather than the semantic quality of a response. 3) The queried content is often either too fine-grained, such as the exact count of surrounding pedestrians, or too generic, such as an open description of the scene, so neither probes the understanding of traffic participants, driving behavior, scene risk, and road topology.

We therefore select benchmarks with reliable annotations and evaluation based on an LLM judge or a multiple-choice protocol. The five public benchmarks we adopt together emphasize spatial judgment, fine-grained object grounding, driving decisions, and traffic-risk assessment. LingoQA [^56] evaluates free-form driving QA with a judge model.<sup>1</sup> Ego3D-Bench [^25] evaluates categorical spatial reasoning with multiple-choice accuracy and absolute distance estimation with RMSE. Its distance queries include ego-to-object and cross-view object-to-object relations. VLADBench [^42] organizes driving competence into a hierarchy of fine-grained capabilities under a multiple-choice protocol. SURDS [^27] targets spatial understanding and reasoning for driving. WaymoQA [^97] focuses on safety-critical multi-view situations, for which we report the overall score and the safety-specific score for the image split.

In addition to public benchmarks, we introduce PAI-AV-CoC, a CoC benchmark derived from the PAI-AV validation split. Qwen3.5-Plus acts as a judge, extracting key objects and driving decisions from predicted CoC traces. We measure key-object accuracy, decision accuracy, and overall accuracy (requiring both to be correct) to assess the causal content of reasoning traces, rather than just their phrasing. We also evaluate on an in-house Chinese urban driving decision benchmark with expert annotations to assess decision-making ability.

##### Comparison Methods.

We compare against recent vision-language models that target physical AI and autonomous driving. The Cosmos family progresses through three generations. Cosmos-Reason1 [^3] post-trains Qwen2.5-VL on physical common sense and embodied reasoning with supervised fine-tuning and reinforcement learning. Cosmos-Reason2 [^60] extends this recipe to Qwen3-VL at 2B, 8B, and 32B parameters, and Cosmos3-nano [^1] is the most recent omnimodal model. MiMo-Embodied [^28] is a 7B cross-embodied model that builds on MiMo-VL and is pretrained on large-scale autonomous driving and embodied data. UniDriveVLA [^41] decouples driving perception, understanding, and planning into separate experts trained progressively on driving data, and we evaluate its checkpoint after the VQA stage and before planning training. Alpamayo-1.5 [^86] is a 10B reasoning driving model built on Cosmos-Reason2 and trained on chain-of-causation reasoning traces together with large-scale open-source driving data. Beyond these domain-specific models, we further include InternVL3.5-8B-Instruct [^85], LLaVA-OneVision-2-8B-Instruct (LLaVA-OV2-8B) [^2], and Gemma4-12B [^80] as representative general-purpose VLMs. We re-evaluate all comparison methods under a common protocol. Decoding uses near-deterministic sampling with top- $k{=}1$, top- $p{=}0.001$, and temperature $0.01$, without repetition or presence penalties, which reduces generation variance. The same judge scores every method on each benchmark.

Table 2: Driving VQA results. Higher scores are better, except for Ego3D RMSE. IH denotes the in-house driving-decision benchmark. The first Avg. averages the six higher-is-better metrics in the left group and excludes the Ego3D RMSE. The second averages the three PAI-AV-CoC metrics and IH. “–” indicates an invalid or unparsable response and is counted as zero in the relevant average. Bold and underlined values denote the best and second-best, respectively.

<table><tbody><tr><td rowspan="3">Method</td><td colspan="8">Driving QA & Spatial Understanding</td><td colspan="5">Causal Reasoning</td></tr><tr><td rowspan="2">LingoQA</td><td colspan="2">Ego3D</td><td rowspan="2">VLAD</td><td rowspan="2">SURDS</td><td colspan="2">WaymoQA</td><td rowspan="2">Avg.</td><td colspan="3">PAI-AV-CoC</td><td rowspan="2">IH</td><td rowspan="2">Avg.</td></tr><tr><td>Acc.</td><td>RMSE <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>Safety</td><td>All</td><td>Key</td><td>Plan</td><td>All</td></tr><tr><td>InternVL3.5-8B-Inst.</td><td>46.40</td><td>47.38</td><td>23.01</td><td>54.47</td><td>32.80</td><td>54.47</td><td>58.09</td><td>48.94</td><td>–</td><td>–</td><td>–</td><td>47.50</td><td>11.88</td></tr><tr><td>LLaVA-OV2-8B</td><td>41.20</td><td>42.07</td><td>24.97</td><td>58.71</td><td>38.60</td><td>49.65</td><td>55.23</td><td>47.58</td><td>0.86</td><td>12.32</td><td>0.57</td><td>54.00</td><td>16.94</td></tr><tr><td>Gemma4-12B</td><td>50.00</td><td>56.31</td><td>27.30</td><td>57.90</td><td>52.25</td><td>63.59</td><td>68.82</td><td>58.15</td><td>12.89</td><td>10.32</td><td>4.01</td><td>61.00</td><td>22.05</td></tr><tr><td>Qwen3.5-4B</td><td>70.40</td><td>62.82</td><td>13.17</td><td>65.38</td><td>52.95</td><td>62.46</td><td>67.10</td><td>63.52</td><td>8.88</td><td>9.17</td><td>2.58</td><td>59.00</td><td>19.91</td></tr><tr><td>Cosmos-Reason1-7B</td><td>45.20</td><td>44.17</td><td>26.71</td><td>33.64</td><td>8.49</td><td>39.53</td><td>43.90</td><td>35.82</td><td>14.61</td><td>11.75</td><td>3.15</td><td>30.50</td><td>15.00</td></tr><tr><td>Cosmos-Reason2-2B</td><td>38.60</td><td>44.42</td><td>12.03</td><td>53.90</td><td>29.46</td><td>63.71</td><td>59.65</td><td>48.29</td><td>9.17</td><td>5.44</td><td>2.01</td><td>42.50</td><td>14.78</td></tr><tr><td>Cosmos-Reason2-8B</td><td>59.60</td><td>48.35</td><td>12.62</td><td>56.37</td><td>19.54</td><td>57.68</td><td>57.93</td><td>49.91</td><td>7.74</td><td>5.44</td><td>1.72</td><td>56.00</td><td>17.73</td></tr><tr><td>Cosmos-Reason2-32B</td><td>58.80</td><td>47.31</td><td>20.32</td><td>57.13</td><td>19.52</td><td>48.56</td><td>48.40</td><td>46.62</td><td>18.34</td><td>15.47</td><td>5.73</td><td>29.50</td><td>17.26</td></tr><tr><td>Cosmos3-nano</td><td>65.00</td><td>44.02</td><td>22.41</td><td>57.73</td><td>39.72</td><td>56.93</td><td>58.36</td><td>53.63</td><td>12.89</td><td>10.60</td><td>4.01</td><td>2.00</td><td>7.38</td></tr><tr><td>MiMo-Embodied-7B</td><td>72.00</td><td>60.41</td><td>9.85</td><td>50.33</td><td>43.06</td><td>66.54</td><td>69.56</td><td>60.32</td><td>–</td><td>–</td><td>–</td><td>61.00</td><td>15.25</td></tr><tr><td>UniDriveVLA-8B</td><td>62.00</td><td>44.11</td><td>8.45</td><td>53.09</td><td>20.06</td><td>49.19</td><td>49.28</td><td>46.29</td><td>0.86</td><td>12.89</td><td>0.86</td><td>48.50</td><td>15.78</td></tr><tr><td>Alpamayo-1.5-10B</td><td>64.00</td><td>36.79</td><td>25.31</td><td>9.13</td><td>3.10</td><td>42.61</td><td>44.37</td><td>33.33</td><td>9.46</td><td>8.31</td><td>3.44</td><td>3.00</td><td>6.05</td></tr><tr><td>Qwen-Drive-1.0-SFT</td><td>77.80</td><td>60.98</td><td>7.78</td><td>66.52</td><td>66.13</td><td>70.70</td><td>74.47</td><td>69.43</td><td>65.33</td><td>55.59</td><td>41.26</td><td>71.00</td><td>58.30</td></tr></tbody></table>

##### Results.

As shown in Tab. 2, Qwen3.5-4B is the strongest general-purpose reference despite being the smallest, and its driving QA average of 63.52 surpasses the larger Gemma4-12B at 58.15 as well as every specialized comparison method. Qwen-Drive-1.0-SFT improves this average by 5.91 points to 69.43, the highest among all methods. This improvement is not concentrated in a single skill but distributed across complementary capabilities. LingoQA rises by 7.40 points and SURDS by 13.18 points, a 24.9% relative gain, while the Ego3D distance-estimation RMSE drops by 40.9% to 7.78, which is 7.9% below the 8.45 of UniDriveVLA-8B. We attribute this reduction not to geometrically precise representations learned by the VLM, but to a richer physical understanding of driving scenes together with the model’s retained quantitative reasoning. Semantic cues of stable real-world scale, such as the regular spacing of parked cars along a street or of dashed lane markings and streetlights, provide implicit references for metric distance estimation, as illustrated in Fig. 8(c).

The most pronounced gains appear in causal reasoning, where Qwen-Drive-1.0-SFT attains an average of 58.30 versus 22.05 for the second-best Gemma4-12B. Relative to Qwen3.5-4B, Qwen-Drive-1.0-SFT improves key-object, decision, and overall accuracy by 56.45, 46.42, and 38.68 points, respectively. Its overall accuracy of 41.26 is more than seven times that of the much larger Cosmos-Reason2-32 B (5.73). The model therefore not only identifies the causal evidence in a scene but also translates it into an appropriate driving decision. The general-purpose VLMs reach at most 4.01 on this overall accuracy, while InternVL3.5-8B produces invalid or unparsable responses for all three CoC metrics. Such margins indicate that causal grounding remains weak in both general-purpose and driving-oriented models, and that targeted CoC supervision substantially narrows this gap. It is worth noting that Alpamayo-1.5 is trained on large-scale proprietary CoC traces yet underperforms on this benchmark. Its lower score under our protocol may reflect instruction-following errors and differences in input format, and should not be interpreted as evidence of weak causal reasoning. By contrast, Qwen-Drive-1.0-SFT also performs well on the in-house benchmark of Chinese urban driving decisions. Our training mixture contains no comparable data and this benchmark additionally requires responses in Chinese, which suggests that its decision competence is not confined to the training distribution.

Furthermore, several methods produce invalid or unparsable responses because they cannot follow the required JSON format, while the video-dominated post-training of the Cosmos models may contribute to their weaker results on static single-frame and multi-view inputs. By retaining instruction following and training across diverse input configurations, Qwen-Drive-1.0-SFT improves nearly every driving metric without exhibiting these failure modes.

![[drivevqa-vis1.png|Refer to caption]]

Figure 8: Qualitative comparison of driving VQA capabilities. Green and red highlight correct and incorrect content, respectively. Questions and responses are abridged for space, with complete examples provided in Appendix C.

##### Qualitative Comparison.

Fig. 8 compares four complementary capabilities. In (a), Qwen-Drive-1.0-SFT respects the requested temporal scope and reports no parked vehicles in the final frame, whereas Qwen3.5-4B and MiMo-Embodied-7B count agents merely stopped in traffic, conflating stationary with parked. In (b), Qwen-Drive-1.0-SFT attributes the stopping decision to the stop sign, while Cosmos-Reason2-32B and Alpamayo-1.5-10B predict continued straight driving based on elements that do not govern the decision. In (c), Qwen-Drive-1.0-SFT decomposes the cross-view distance into street width and longitudinal offset, uses parked-vehicle spacing as an implicit scale reference, and predicts $22\,\mathrm{m}$ against a ground truth of $22.93\,\mathrm{m}$. UniDriveVLA-8B instead cites an existing annotation rather than visible scale cues. In (d), Qwen-Drive-1.0-SFT identifies the ego lane from the persistent left-turn arrow, whereas Alpamayo-1.5-10B emits a DriveLM-style object reference outside the option set. Cases (c) and (d) also expose differences in instruction following: UniDriveVLA-8B omits the required \\boxed{} delimiter, while Qwen-Drive-1.0-SFT adheres to both the format and the answer set. Together, these examples show that Qwen-Drive-1.0-SFT integrates evidence across views and time into the requested decision while respecting output constraints. Further cases appear in Appendix B.

#### 3.2.2 General Vision-Language Understanding

Table 3: General vision-language results on (a) knowledge, reasoning, and recognition and (b) spatial understanding and grounding. Avg. is the average over the benchmarks. Compared with Qwen3.5-4B, Qwen-Drive-1.0-SFT stays within one point on group (a) and surpasses it on group (b), preserving general knowledge and capabilities after driving adaptation. “–” indicates an invalid response and is counted as zero in the averages. Bold and underlined denote the best and second-best.

(a) Knowledge, reasoning and recognition Method MM Bench MM Star MMMU MMMU-Pro CharXiv OCR Bench RealWorld QA Simple VQA Count QA Avg. Std Vis InternVL3.5-8B-Inst. 80.03 64.13 62.00 46.42 42.25 41.70 83.20 66.93 40.77 20.94 54.84 LLaVA-OV2-8B 82.66 64.93 54.67 36.30 25.95 40.10 79.30 71.76 36.68 22.58 51.49 Gemma4-12B 85.53 72.33 69.56 59.94 49.25 64.10 77.80 68.89 40.82 39.46 62.77 Qwen3.5-4B 87.07 75.33 73.44 64.86 61.27 65.10 86.90 76.34 47.84 35.86 67.40 Cosmos-Reason1-7B 79.95 63.53 54.22 38.38 35.78 39.70 85.20 67.45 44.98 18.52 52.77 Cosmos-Reason2-2B 75.00 53.13 51.56 35.09 30.35 28.50 79.60 60.52 36.94 18.00 46.87 Cosmos-Reason2-8B 82.82 65.27 59.11 36.07 43.53 42.50 87.00 67.45 45.25 22.32 55.13 Cosmos-Reason2-32B 88.70 72.47 61.67 41.45 52.77 53.20 88.20 75.69 48.44 26.70 60.93 Cosmos3-nano 79.57 66.67 60.89 46.36 40.75 42.10 85.20 69.67 44.99 23.63 55.98 MiMo-Embodied-7B – 22.40 – 27.40 28.09 57.50 78.80 28.50 – 22.64 26.53 UniDriveVLA-8B 74.30 64.07 50.67 32.43 31.56 33.80 80.20 68.10 35.26 16.88 48.73 Alpamayo-1.5-10B 7.51 26.13 27.44 15.61 13.47 1.50 3.20 46.93 – 4.71 14.65 Qwen-Drive-1.0-SFT 85.53 75.87 72.67 62.72 59.71 64.40 86.40 78.95 46.12 31.74 66.41

(b) Spatial understanding and grounding Method EmbSpatial ERQA RefSpatial Omni3D ODinW13 Avg. InternVL3.5-8B-Inst. 74.20 42.00 – – – 23.24 LLaVA-OV2-8B 78.43 42.25 – – – 24.14 Gemma4-12B 73.16 42.00 – – – 23.03 Qwen3.5-4B 75.99 46.25 54.51 47.40 40.78 52.99 Cosmos-Reason1-7B 68.76 38.50 0.36 – 4.77 22.48 Cosmos-Reason2-2B 66.40 38.75 32.49 31.41 33.04 40.42 Cosmos-Reason2-8B 77.61 43.25 51.81 32.85 40.19 49.14 Cosmos-Reason2-32B 79.26 45.25 57.76 31.70 28.47 48.49 Cosmos3-nano 77.88 41.25 – 32.26 35.87 37.45 MiMo-Embodied-7B 45.05 39.75 2.17 – – 17.39 UniDriveVLA-8B 68.16 38.00 1.44 0.33 – 21.59 Alpamayo-1.5-10B 20.58 27.50 – – – 9.62 Qwen-Drive-1.0-SFT 78.85 48.50 50.78 45.79 45.87 53.96

##### Benchmarks.

A driving foundation model should acquire domain competence without surrendering the broad visual and world knowledge needed for open-set reasoning and for cockpit applications on an integrated platform. We therefore evaluate fourteen public benchmarks in the two groups of Tab. 3. Group (a) covers knowledge and reasoning with MMBench [^51], MMStar [^9], MMMU [^100], standard and vision splits of MMMU-Pro [^101], and CharXiv [^87], together with general recognition through OCRBench [^52], RealWorldQA [^90], SimpleVQA [^10] for multimodal factuality, and CountQA [^78] for object counting. Group (b) covers spatial understanding and grounding with EmbSpatial-Bench [^17], ERQA [^79], RefSpatial-Bench [^107], Omni3D-Bench [^57], and ODinW13 [^37].

##### Results.

As shown in Tab. 3, Qwen-Drive-1.0-SFT preserves broad vision-language competence after adaptation on large-scale driving data. On the knowledge, reasoning, and recognition benchmarks in Tab. 3(a), it averages 66.41 versus 67.40 for Qwen3.5-4B, staying within one point while ranking first or second on 6 of the 10 settings, including the best scores on MMStar and RealWorldQA. On the spatial understanding and grounding benchmarks in Tab. 3(b), it averages 53.96 and even exceeds Qwen3.5-4B at 52.99, with the best scores on ERQA and ODinW13 covering embodied spatial reasoning and open-set object grounding. Driving adaptation therefore leaves knowledge-intensive capabilities essentially intact and strengthens spatially grounded ones.

This preservation distinguishes Qwen-Drive-1.0-SFT from comparison methods specialized for physical AI, embodied reasoning, and autonomous driving. It matches or exceeds Cosmos-Reason2-32B on 10 of the 15 settings and achieves a 5.48-point higher average across them. Several other specialized models produce invalid responses on some benchmarks, showing that strong domain specialization does not necessarily preserve the general instruction-following interface required across diverse tasks. In contrast, Qwen-Drive-1.0-SFT improves driving understanding while maintaining both this interface and broad visual and world knowledge. Together, these results suggest the potential for more reliable generalization to rare and previously unseen situations in open-world driving.

The preserved capabilities also matter for deployment. Under cockpit-driving integration, a single onboard model is expected to serve cockpit applications such as dialogue and open-ended visual queries, alongside driving perception and planning. Qwen-Drive-1.0-SFT loses less than one point on group (a) and gains on group (b) while acquiring driving competence, so one instance can cover both domains. This removes the need for a separate cockpit model and the associated compute and maintenance cost. Methods that lose general capability during driving adaptation cannot provide this saving, even when their driving scores are competitive.

### 3.3 Motion Planning

##### Benchmarks.

To evaluate motion planning at progressively greater levels of interaction, we consider two open-loop benchmarks, one pseudo-closed-loop benchmark, and one closed-loop simulator.

1) WOD-E2E [^93] focuses on challenging long-tail scenarios and provides several human-rated candidate trajectories for each scene. Its primary metric is the Rater Feedback Score (RFS), which matches a prediction to these candidates under a speed-dependent tolerance and assigns the rating of the closest valid match. RFS can therefore recognize an acceptable plan even when it differs from the recorded future. We also report average displacement error (ADE), defined as the mean Euclidean position error relative to the recorded trajectory, at 3 s and 5 s.

2) PAI-AV [^61] requires each method to predict six candidate trajectories per scene. We report their average ADE to measure overall trajectory quality and the minimum ADE (minADE) to determine whether any candidate covers the recorded motion. We evaluate both metrics at 3 s and 5 s. The standard 644-example split overlaps with publicly available training data by official construction. To ensure a fair comparison, we also report results on a leakage-free subset curated from held-out test clips.

3) NAVSIM [^14] follows a pseudo-closed-loop protocol. The planner is queried only once, and surrounding agents replay their recorded motion without reacting to the ego vehicle. On NAVSIM v1.1 navtest, we report PDMS and its five components. No collision (NC) and drivable area compliance (DAC) act as multiplicative factors on a weighted combination of ego progress (EP), time-to-collision (TTC), and comfort (Comf.).

4) AlpaSim [^59] evaluates closed-loop behavior under compounding errors. We use PAI-AV-NuRec [^88] version 26.02 to simulate 916 scenarios with novel views as the ego vehicle deviates from recorded logs. Following Alpamayo [^86], we measure close encounter rate (all-event and at-fault), off-road rate, progress, and AlpaSim score (all-event and at-fault). These metrics quantify close-proximity events, roadway departures, scenario completion, and average distance traveled between events, respectively. The *at-fault* variants exclude events not caused by the ego vehicle.

Table 4: Open-loop planning on WOD-E2E. We report average displacement error (ADE) at 3 s and 5 s and the Rater Feedback Score (RFS). RL indicates reinforcement learning. Because the validation split is used for RL, the test split provides a more informative assessment of generalization.

(a) Validation split.

| Method | RL | ADE $\downarrow$ 3 s | ADE $\downarrow$ 5 s | RFS $\uparrow$ |
| --- | --- | --- | --- | --- |
| Human Driver [^72] | – | – | – | 8.13 |
| VAD [^35] | – | 3.19 | 5.81 | 4.45 |
| UniAD [^31] | – | 6.50 | 10.81 | 5.78 |
| RAP-DINO [^21] | – | 0.97 | 2.20 | 7.91 |
| MindVLA-U1 [^32] | – | 0.89 | 2.11 | 7.92 |
| MindVLA-U1 [^32] | ✓ | 1.01 | 2.28 | 8.20 |
| Qwen-Drive-1.0-SFT w/o reasoning | – | 0.99 | 2.33 | 7.95 |
| Qwen-Drive-1.0-SFT w/ reasoning | – | 0.99 | 2.31 | 7.95 |
| Qwen-Drive-1.0-RL | ✓ | 0.62 | 1.27 | 8.45 |

(b) Test split.

| Method | RL | ADE $\downarrow$ 3 s | ADE $\downarrow$ 5 s | RFS $\uparrow$ |
| --- | --- | --- | --- | --- |
| Swin-Trajectory [^64] | – | 1.21 | 2.81 | 7.54 |
| DiffusionLTF [^58] | – | 1.36 | 2.89 | 7.72 |
| UniPlan [^45] | – | 1.31 | 2.99 | 7.78 |
| LightEMMA [^67] | – | 1.71 | 3.74 | 6.52 |
| NaiveEMMA [^93] | – | 1.32 | 3.02 | 7.53 |
| dVLM-AD [^54] | – | 1.29 | 3.02 | 7.63 |
| HMVLM [^82] | – | 1.33 | 3.07 | 7.74 |
| MindVLA-U1 [^32] | – | 1.16 | 2.67 | 7.77 |
| AutoVLA [^110] | ✓ | 1.35 | 2.96 | 7.56 |
| NoRD [^71] | ✓ | 1.25 | – | 7.71 |
| MindVLA-U1 [^32] | ✓ | 1.09 | 2.66 | 7.87 |
| Qwen-Drive-1.0-SFT w/o reasoning | – | 1.20 | 2.66 | 7.76 |
| Qwen-Drive-1.0-SFT w/ reasoning | – | 1.19 | 2.65 | 7.78 |
| Qwen-Drive-1.0-RL | ✓ | 1.19 | 2.67 | 7.91 |

##### Results.

We evaluate Qwen-Drive-1.0-SFT with and without the planning-reasoning condition, together with Qwen-Drive-1.0-RL after reinforcement learning. Using 2.83M training samples assembled from public sources, our method attains the highest PDMS on NAVSIM and the highest test-split RFS on WOD-E2E among the compared methods. We analyze the results at increasing levels of interaction, from open-loop prediction to pseudo-closed-loop scoring and closed-loop simulation.

Open-loop. On the long-tail WOD-E2E test split in Tab. 4(b), Qwen-Drive-1.0-SFT with reasoning achieves an RFS of 7.78, slightly exceeding MindVLA-U1 at 7.77 before reinforcement learning. Reasoning raises RFS from 7.76 to 7.78, even though only 142K of the 557K WOD-E2E samples provide reasoning-conditioned supervision. On the validation split in Tab. 4(a), which supplies the RFS annotations used for reinforcement learning, Qwen-Drive-1.0-RL improves RFS from 7.95 to 8.45 while significantly reducing the 3 s and 5 s ADE. The resulting RFS exceeds the human-driver reference of 8.13. Since these annotations supervise the reward, this in-sample result indicates effective optimization of preference alignment on the training scenarios rather than generalization beyond human driving. More importantly, the improvement transfers to the test split. Reinforcement learning raises RFS from 7.78 to 7.91, exceeding the reinforced MindVLA-U1 by 0.04 points. The held-out gain indicates improved alignment with human preference without materially changing displacement from the recorded future.

Table 5: Open-loop motion planning on PAI-AV. We report the average and minimum ADE (m) over six trajectories at 3 s and 5 s on the standard 644-example split and a leakage-free 700-frame subset curated from held-out test clips. All comparison methods are reproduced under the same evaluation setting.

<table><tbody><tr><td rowspan="3">Method</td><td colspan="4">644-example split</td><td colspan="4">700-frame subset</td></tr><tr><td colspan="2">Avg. ADE <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td colspan="2">minADE <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td colspan="2">Avg. ADE <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td colspan="2">minADE <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><td>3 s</td><td>5 s</td><td>3 s</td><td>5 s</td><td>3 s</td><td>5 s</td><td>3 s</td><td>5 s</td></tr><tr><td>Alpamayo-R1-10B <sup><a href="#fn:86">86</a></sup></td><td>0.37</td><td>1.13</td><td>0.16</td><td>0.48</td><td>0.41</td><td>1.22</td><td>0.18</td><td>0.51</td></tr><tr><td>Alpamayo-1.5-10B <sup><a href="#fn:86">86</a></sup></td><td>0.35</td><td>1.05</td><td>0.16</td><td>0.50</td><td>0.36</td><td>1.06</td><td>0.17</td><td>0.49</td></tr><tr><td>DriveWAM <sup><a href="#fn:75">75</a></sup></td><td>0.67</td><td>–</td><td>0.37</td><td>–</td><td>0.69</td><td>–</td><td>0.38</td><td>–</td></tr><tr><td>SimWAM <sup><a href="#fn:106">106</a></sup></td><td>0.41</td><td>–</td><td>0.38</td><td>–</td><td>0.43</td><td>–</td><td>0.40</td><td>–</td></tr><tr><td>Qwen-Drive-1.0-SFT w/o reasoning</td><td>0.38</td><td>1.07</td><td>0.34</td><td>0.96</td><td>0.43</td><td>1.24</td><td>0.39</td><td>1.11</td></tr><tr><td>Qwen-Drive-1.0-SFT w/ reasoning</td><td>0.37</td><td>1.07</td><td>0.34</td><td>0.97</td><td>0.42</td><td>1.23</td><td>0.39</td><td>1.11</td></tr><tr><td>Qwen-Drive-1.0-RL</td><td>0.42</td><td>1.11</td><td>0.38</td><td>1.00</td><td>0.47</td><td>1.27</td><td>0.43</td><td>1.15</td></tr></tbody></table>

Table 6: Pseudo-closed-loop motion planning on NAVSIM v1.1 navtest. We report the Predictive Driver Model Score (PDMS) and its no-collision (NC), drivable area compliance (DAC), ego progress (EP), time-to-collision (TTC), and comfort (Comf.) components. RL indicates reinforcement learning. $\ddagger$ denotes best-of- $N$ selection with $N=6$, where the candidate with the highest PDMS is chosen for each scene.

| Method | RL | NC $\uparrow$ | DAC $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | Comf. $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TransFuser [^12] | \- | 97.7 | 92.8 | 79.2 | 92.8 | 100.0 | 84.0 |
| DRAMA [^99] | \- | 98.0 | 93.1 | 80.1 | 94.8 | 100.0 | 85.5 |
| Hydra-MDP [^43] | \- | 98.3 | 96.0 | 78.7 | 94.6 | 100.0 | 86.5 |
| DiffusionDrive [^45] | \- | 98.2 | 96.2 | 82.2 | 94.7 | 100.0 | 88.1 |
| Epona [^104] | \- | 97.9 | 95.1 | 80.4 | 93.8 | 99.9 | 86.2 |
| ReCogDrive [^40] | \- | 98.3 | 95.1 | 81.1 | 94.3 | 100.0 | 86.8 |
| AutoVLA [^110] | \- | 96.9 | 92.4 | 75.8 | 88.1 | 99.9 | 80.5 |
| SpanVLA [^111] | \- | 97.5 | 90.8 | 76.9 | 93.7 | 99.5 | 82.1 |
| Qwen-Drive-1.0-SFT w/o reasoning | \- | 98.2 | 96.4 | 82.0 | 94.4 | 100.0 | 87.8 |
| Qwen-Drive-1.0-SFT w/ reasoning | \- | 98.4 | 96.6 | 82.4 | 94.7 | 100.0 | 88.2 |
| Qwen-Drive-1.0-SFT <sup>‡</sup> w/o reasoning | \- | 98.6 | 97.1 | 82.9 | 95.1 | 100.0 | 88.9 |
| Qwen-Drive-1.0-SFT <sup>‡</sup> w/ reasoning | \- | 98.7 | 97.2 | 83.2 | 95.5 | 100.0 | 89.3 |
| ReCogDrive [^40] | ✓ | 98.2 | 97.8 | 83.5 | 95.2 | 99.8 | 89.6 |
| AutoVLA [^110] | ✓ | 98.4 | 95.6 | 81.9 | 98.0 | 99.9 | 89.1 |
| SpanVLA [^111] | ✓ | 99.1 | 97.1 | 86.3 | 95.2 | 100.0 | 90.3 |
| ExploreVLA [^74] | ✓ | 98.8 | 98.4 | 83.5 | 96.5 | 99.9 | 90.4 |
| EponaV2 [^91] | ✓ | 98.6 | 97.9 | 84.8 | 95.7 | 100.0 | 90.4 |
| Qwen-Drive-1.0-RL | ✓ | 98.6 | 98.2 | 84.8 | 95.9 | 100.0 | 90.7 |
| Qwen-Drive-1.0-RL <sup>‡</sup> | ✓ | 98.8 | 98.4 | 85.5 | 96.5 | 100.0 | 91.4 |

Complementing the preference-based evaluation on WOD-E2E, PAI-AV assesses both the quality and diversity of six predicted trajectories. We reproduce all comparison methods under the same evaluation setting. As shown in Tab. 5, on the leakage-free subset, Qwen-Drive-1.0-SFT with reasoning attains a 3 s average ADE of 0.42 m, compared with 0.36 m for Alpamayo-1.5. Our minADE is 0.39 m versus 0.17 m, and the narrow gap between average ADE and minADE suggests that our candidates remain concentrated around similar motions. Part of this difference is plausibly attributable to training scale. Alpamayo-1.5 uses 80,000 hours of driving trajectories and 3M CoC reasoning traces, whereas PAI-AV contains 156K clips, corresponding to approximately 900 raw hours before sparse frame sampling. DriveWAM and SimWAM both incorporate future generative supervision, yet exhibit markedly different candidate distributions. DriveWAM achieves a much lower minADE than average ADE, suggesting broader candidate coverage but weaker typical trajectory accuracy. SimWAM substantially improves average ADE to 0.43 m, while its minADE remains close at 0.40 m, indicating more limited diversity. In comparison, Qwen-Drive-1.0-SFT achieves a slightly lower average ADE of 0.42 m with a comparable minADE of 0.39 m. Reinforcement learning increases the PAI-AV errors by only 3 to 5 cm. This modest open-loop trade-off accompanies improved preference alignment on WOD-E2E, higher pseudo-closed-loop PDMS on NAVSIM, and a halving of the closed-loop off-road rate in AlpaSim.

Table 7: Closed-loop planning on 916 AlpaSim [^59] scenarios using PAI-AV-NuRec [^88] version 26.02. Params. denotes all parameters excluding the LLM token embeddings. All comparison methods are reproduced under the same evaluation setting.

<table><tbody><tr><td rowspan="2">Method</td><td rowspan="2">Params.</td><td colspan="2">Close Encounter Rate (%) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td rowspan="2">Off-Road Rate (%) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td rowspan="2">Progress (%) <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td colspan="2">AlpaSim Score <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>all</td><td>at-fault</td><td>all</td><td>at-fault</td></tr><tr><td>Alpamayo-R1 <sup><a href="#fn:86">86</a></sup></td><td>9.8 B</td><td>19.0</td><td>6.0</td><td>17.0</td><td>67.0</td><td>0.36</td><td>0.58</td></tr><tr><td>Alpamayo-1.5 <sup><a href="#fn:86">86</a></sup></td><td>9.8 B</td><td>37.0</td><td>11.0</td><td>16.0</td><td>59.0</td><td>0.23</td><td>0.45</td></tr><tr><td>DriveWAM <sup><a href="#fn:75">75</a></sup></td><td>15.0 B</td><td>56.0</td><td>5.0</td><td>8.0</td><td>35.0</td><td>0.10</td><td>0.53</td></tr><tr><td>SimWAM <sup><a href="#fn:106">106</a></sup></td><td>6.0 B</td><td>35.0</td><td>22.0</td><td>19.0</td><td>62.0</td><td>0.22</td><td>0.30</td></tr><tr><td>Qwen-Drive-1.0-SFT w/ reasoning</td><td>5.0 B</td><td>38.0</td><td>12.0</td><td>24.0</td><td>54.0</td><td>0.16</td><td>0.27</td></tr><tr><td>Qwen-Drive-1.0-RL</td><td>5.0 B</td><td>41.0</td><td>11.0</td><td>12.0</td><td>48.0</td><td>0.16</td><td>0.37</td></tr></tbody></table>

Pseudo-closed-loop. NAVSIM advances one step toward closed-loop evaluation by propagating the predicted trajectory through a vehicle model. However, it does not re-query the planner and keeps surrounding agents non-reactive, so it cannot capture compounding errors. As shown in Tab. 6, before reinforcement learning, Qwen-Drive-1.0-SFT with reasoning reaches 88.2 PDMS and surpasses comparison methods trained only with imitation, including DiffusionDrive at 88.1. This competitive result may draw on capabilities acquired during vision-language pretraining, together with the scale and behavioral diversity of the unified planning data. Reasoning improves PDMS by 0.4 points despite being available for only 78K NAVSIM samples, indicating that the textual condition provides useful context for trajectory generation. Reinforcement learning yields a further 2.5-point gain, raising PDMS to a promising 90.7. These results suggest that reinforcement on NAVSIM is particularly effective at reducing conservative, low-progress behavior. Following a common practice of reporting the best among multiple sampled trajectories, we also select per scene the candidate with the highest PDMS among six samples, which raises PDMS to 91.4 for Qwen-Drive-1.0-RL and 89.3 for Qwen-Drive-1.0-SFT with reasoning. This margin over the single-trajectory output indicates that the sampled set already contains stronger trajectories than the default prediction, suggesting headroom that better inference-time selection could recover. Nevertheless, PDMS should not be treated as a direct proxy for interactive driving quality. Near the upper end of this benchmark, further gains may increasingly reflect adaptation to the scoring function, while the non-reactive protocol cannot reveal how errors accumulate during interaction.

Closed-loop. AlpaSim completes this progression by repeatedly querying the planner as its actions alter subsequent observations, thereby exposing compounding errors and recovery behavior that NAVSIM cannot measure. For a controlled comparison, we reproduce Alpamayo-R1 and Alpamayo-1.5 under the same evaluation setting. As shown in Tab. 7, Qwen-Drive-1.0-RL achieves an at-fault close encounter rate of 11.0%, matching Alpamayo-1.5, while their all-event rates are 41.0% and 37.0%, respectively. Reinforcement learning halves the off-road rate from 24.0% to 12.0%, a rate lower than that of either Alpamayo variant, and raises the at-fault AlpaSim score from 0.27 to 0.37. These safety gains come with lower progress, which decreases from 54.0% to 48.0%, and a modest increase in the all-event close encounter rate. The results indicate a shift toward safer but more conservative behavior. Despite matching Alpamayo-1.5 in at-fault close encounter rate, Qwen-Drive-1.0-RL records lower all-event and at-fault AlpaSim scores of 0.16 and 0.37, compared with 0.23 and 0.45. One possible factor is the temporal sampling of the visual input. AlpaSim repeatedly replans over short intervals and therefore emphasizes rapid responses to recent visual changes. Alpamayo-1.5 observes a dense 0.4 s visual history, whereas our input contains four observations sampled at 0.5 s intervals over 1.5 s. This broader but sparser history may limit responsiveness over the short replanning horizon.

An AlpaSim score should not be interpreted in isolation. DriveWAM [^75] attains an at-fault score of 0.53, but its progress is only 35.0%. We find that it often remains stationary or advances only briefly. This behavior reduces ego-at-fault and off-road events, which can raise the at-fault score because the metric divides the traveled distance by the corresponding event count. However, a nearly stationary ego vehicle remains susceptible to interactions caused by following traffic. DriveWAM thus records an all-event close encounter rate of 56.0% and an all-event score of 0.10. SimWAM [^106] shows the opposite behavior. We find that its driving policy is considerably more aggressive, frequently accelerating forward while failing to decelerate sufficiently for preceding vehicles or obstacles. This yields relatively high progress of 62.0%, but also leads to an at-fault close encounter rate of 22.0% and an off-road rate of 19.0%, resulting in an at-fault score of only 0.30. In comparison, Qwen-Drive-1.0-RL achieves substantially lower at-fault close encounters and off-road violations of 11.0% and 12.0%, respectively, while maintaining 48.0% progress. These results highlight the importance of jointly considering progress, safety events, and AlpaSim scores when assessing effective closed-loop driving.

![[planning_vis.png|Refer to caption]]

Figure 9: Qualitative motion planning results. (a) Open-loop predictions from Qwen-Drive-1.0-SFT with reasoning on the WOD-E2E test split (left and middle) and PAI-AV (right). WOD-E2E provides no ground-truth trajectory for the test split, while the PAI-AV example shows both the prediction and recorded future. (b) Two closed-loop AlpaSim rollouts from Qwen-Drive-1.0-RL at selected timestamps, with the predicted and recorded trajectories shown for comparison.

![[navsim-rl.png|Refer to caption]]

Figure 10: Qualitative effect of reinforcement learning on the same NAVSIM left-turn scene. (a) Qwen-Drive-1.0-SFT with reasoning before reinforcement learning. (b) Qwen-Drive-1.0-RL. The prediction is shown in red and the recorded future in green in the camera and BEV views.

##### Qualitative Results.

Fig. 9 complements the quantitative evaluation with open-loop and closed-loop planning. In Fig. 9(a), Qwen-Drive-1.0-SFT with reasoning grounds its plans in relevant scene evidence. It decelerates for a crossing animal, follows the right-turn-only lane, and adjusts laterally to pass a stopped vehicle. On PAI-AV, the predicted trajectory remains close to the recorded future. Fig. 9(b) shows two AlpaSim rollouts from Qwen-Drive-1.0-RL. In the first rollout, the model follows the lead vehicle through a green light and subsequently stops when the signal turns red. In the second, it follows a slower vehicle, turns right at the intersection, and adjusts its lateral position as another vehicle overtakes. Across both rollouts, the reasoning and trajectory are updated with the evolving traffic state while remaining consistent with the navigation instruction and road geometry.

Fig. 10 further illustrates how reinforcement learning adapts trajectory predictions to the NAVSIM objective. Both variants retain the same high-level left-turn maneuver, while Qwen-Drive-1.0-RL reduces a small lateral deviation from the recorded future. This fine-grained correction better aligns the trajectory with the NAVSIM scoring criteria and is consistent with the PDMS improvement in Tab. 6.

### 3.4 Ablation Study and Analysis

Table 8: Ablation of the Stage 2 training mixture. Row i uses the unadapted Qwen3.5-4B, while rows ii and iii progressively introduce vision-language and 3D perception supervision. For each row, we train a Planning Expert for 15 epochs on the same WOD-E2E data and evaluate RFS on the validation split.

<table><tbody><tr><td rowspan="2">ID</td><td colspan="2">Stage 2 Training Mixture  </td><td colspan="3">Vision-Language Evaluation  </td><td>Planning</td></tr><tr><td>Vision- Language</td><td>3D Perception</td><td>Driving QA Avg.<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>CoC Overall <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>General VQA Avg.<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>WOD-E2E RFS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>i</td><td>✗</td><td>✗</td><td>63.52</td><td>2.58</td><td>62.60</td><td>7.88</td></tr><tr><td>ii</td><td>✓</td><td>✗</td><td>70.07</td><td>40.97</td><td>63.18</td><td>7.91</td></tr><tr><td>iii</td><td>✓</td><td>✓</td><td>69.43</td><td>41.26</td><td>62.26</td><td>7.96</td></tr></tbody></table>

Figure 11: Ablation of the reinforcement learning data mixture and reward design. (a) Reinforcement learning on NAVSIM alone. (b) Joint reinforcement learning on NAVSIM, WOD-E2E, and PAI-AV. The source-specific reward uses PDMS for NAVSIM, RFS for WOD-E2E, and ADE for PAI-AV. The shared-ADE variant adds a displacement reward to every data source.

##### Analysis of the Stage 2 Training Mixture.

First, we examine how the Stage 2 training mixture in Sec. 2.2 affects vision-language capability, explicit 3D perception, and subsequent planning. Starting from the unadapted Qwen3.5-4B in row i, row ii introduces vision-language training, and row iii additionally enables 3D perception supervision. For a controlled planning comparison, we train a Planning Expert for each variant on WOD-E2E [^93] for 15 epochs in Stage 3 and report RFS on the validation split. Tab. 8 shows that vision-language training improves Driving QA Avg. by 6.55 points and produces a substantially larger gain in CoC reasoning, while preserving general vision-language capability. Adding 3D perception supervision keeps all three vision-language aggregates within one point of row ii. Consistent with Sec. 3.1, the BEV perception head serves as a 3D probe by exposing how readily the shared representations support 3D detection, semantic occupancy, and BEV map predictions. Its perception objectives further provide task-specific 3D supervision to the shared visual pathway during joint adaptation. The complete Stage 2 mixture adds an explicit, inspectable 3D perception capability while largely preserving vision-language performance. It also yields the highest RFS after Stage 3, although the margin over the other variants is small. The planning result supports the compatibility of the Stage 2 mixture with subsequent Planning Expert training, but does not establish explicit 3D supervision as the source of the improvement.

##### Analysis of Reinforcement Rewards.

We next ablate the data mixture and the shared ADE term used for reinforcement learning, as shown in Fig. 11. With NAVSIM-only training, augmenting the source-specific PDMS reward with ADE raises PDMS from 90.4 to 90.8. Joint training on NAVSIM, WOD-E2E, and PAI-AV produces similar NAVSIM scores of 90.6 and 90.7 without and with the shared ADE term, respectively. The small differences from single-source training suggest limited cross-dataset interference, particularly when each source provides a task-aligned reward. On the WOD-E2E validation split, adding ADE lowers RFS from 8.68 to 8.45 but substantially reduces the 5 s ADE from 2.24 m to 1.27 m. The shared displacement reward anchors preference optimization to the recorded motion and limits excessive trajectory deviation. Based on this trade-off, our final recipe jointly trains on all three sources and includes the shared ADE term for every source.

Figure 12: Effect of PAI-AV planning data scale. The Planning Expert is trained in Stage 3 using only PAI-AV, with the number of training samples increasing from 0.17M to 1.38M. Both 5 s Avg. ADE and Avg. FDE decrease consistently on the standard 644-example split and the leakage-free 700-frame subset.

![[ood_perception_vis.png|Refer to caption]]

Figure 13: Qualitative perception outputs of Qwen-Drive-1.0-SFT on unseen camera rigs. WOD-E2E is shown in (a, b) and PAI-AV in (c, d). These datasets provide no unified perception ground truth, so all panels show predictions only.

##### Effect of Planning Data Scale.

We further examine how Stage 3 planning performance scales with the amount of trajectory supervision. To isolate the effect of data scale from cross-dataset mixing, we train the Planning Expert exclusively on PAI-AV using 0.17M, 0.35M, 0.69M, 1.04M, and 1.38M training samples. As shown in Fig. 12, both 5 s Avg. ADE and Avg. FDE decrease monotonically as the training set grows. On the standard split, Avg. ADE decreases from 1.34 m to 1.05 m, while Avg. FDE decreases from 4.18 m to 3.24 m. The leakage-free subset remains more challenging but follows the same trend, showing that the gains are not confined to the standard split. Performance continues to improve at 1.38M samples, with no clear sign of saturation at the current data scale.

##### Qualitative Transfer to Unseen Camera Rigs.

The quantitative comparison in Sec. 3.1 covers only the two sources that supervise perception. We finally examine whether the perception pathway can operate directly on camera configurations never seen in training. To this end, we sample frames from the eight-camera ring rig of WOD-E2E [^93] and the six-camera rig of PAI-AV [^61], whose full surround views are absent from all perception training data. After correcting lens distortion to a pinhole model, we run Qwen-Drive-1.0-SFT directly on both rigs without dataset-specific adaptation. As shown in Fig. 13, the projected boxes are visually plausible across views and remain qualitatively consistent with the corresponding BEV layouts. The occupancy and map outputs also exhibit coherent road surfaces, object regions, drivable areas, lane markings, and road edges. These examples show that the perception pathway remains operational and produces qualitatively coherent outputs on unseen camera rigs. Because neither dataset provides unified ground truth, they do not establish reliable 3D accuracy under the new camera settings. Quantitative evaluation and adaptation with high-quality annotations from the target vehicle platform are required to assess and improve such cross-rig transfer.

## Conclusion

We presented Qwen-Drive-1.0, which we consider an initial step toward a vision-language foundation model for autonomous driving. The framework retains the pretrained VLM architecture and introduces external modules for unified 3D perception and motion planning. The BEV perception head serves as a 3D probe and adds explicit, inspectable detection, occupancy, and map predictions to the same pretrained VLM. The Planning Expert uses the shared representations to generate future ego trajectories through flow matching and supports joint training across multiple public driving datasets. A staged training and data recipe further combines driving-specific supervision with general-purpose vision-language data.

Experiments show that Qwen-Drive-1.0 achieves highly competitive results in 3D perception, driving scene understanding, and motion planning while largely preserving general vision-language capability. Its performance across open-loop, pseudo-closed-loop, and closed-loop planning evaluations also demonstrates the value of combining unified trajectory supervision with reward-based optimization. Together, these results show that explicit 3D perception and trajectory generation can be added to a pretrained VLM while retaining its broader vision-language competence.

## Limitations and Future Work

Although Qwen-Drive-1.0 establishes a unified framework for perception, reasoning, and planning, several directions remain open. First, the planning reasoning does not always capture the causal structure of a scene accurately. Driving mixes causes that act at different time scales. A red light 20 m ahead calls for early, gradual deceleration, whereas a child emerging 5 m ahead demands an immediate response. When such causes coexist, the model remains unstable in identifying the governing cause and its temporal scope. Even when the suggested trend is appropriate, the decision executed within the next 1 to 2 s may not reflect the stated immediate cause. Second, the generated trajectory does not always adhere to the textual rationale. Although reasoning improves downstream planning performance, part of this gain may stem from the additional model-internal information that the self-generated trace contributes to the conditioning context. Addressing these issues calls for multi-timescale causal modeling and explicit consistency supervision between the rationale and the generated trajectory.

Stronger cross-task transfer is another promising direction. The three tasks currently use different input formats, temporal contexts, and image resolutions, which may limit the transfer of learned representations. Better alignment of these configurations and closer joint optimization may allow perception, reasoning, and planning to reinforce one another more consistently.

## Authors

Core Contributors: Xin Zhou <sup>1,2</sup>, Zongchuang Zhao <sup>1,2</sup>, Zhibo Yang <sup>1</sup> <sup><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\dagger"><semantics><mo>†</mo> <annotation>\dagger</annotation></semantics></math></sup>, Mingsheng Li <sup>1</sup>, Humen Zhong <sup>1</sup>, Shuai Bai <sup>1</sup>, Dingkang Liang <sup>2</sup> <sup>🖂</sup>, Xiang Bai <sup>2</sup> <sup>🖂</sup>, Dayiheng Liu <sup>1</sup> <sup><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\dagger"><semantics><mo>†</mo> <annotation>\dagger</annotation></semantics></math></sup>  
Contributors (ordered alphabetically): Du Chu <sup>1</sup>, Ruizhe Chen <sup>1</sup>, Zhaohai Li <sup>1</sup>, Jun Tang <sup>1</sup>, Qiuyue Wang <sup>1</sup>, Mingkun Yang <sup>1</sup>, Jiazhao Zhang <sup>1</sup>  
External Advisors: Dingkang Liang <sup>2</sup> <sup>🖂</sup>, Xiang Bai <sup>2</sup> <sup>🖂</sup>  
<sup>1</sup>  Qwen Team  
<sup>2</sup>  Huazhong University of Science and Technology

<sup>†</sup>

Acknowledgment. This work is partially supported by the NSFC (62225603).

## References

## Appendix A Reward Definitions for Reinforcement Learning

This appendix specifies the per-source rewards used in Stage 4. A rollout produces the trajectory $\boldsymbol{\tau}^{\mathrm{out}}=\{(\hat{x}_{k},\hat{y}_{k},\hat{\theta}_{k})\}_{k=1}^{50}$ of Eq. 6, covering 5 s at 10 Hz and expressed in metric units. Let $\boldsymbol{\tau}^{\mathrm{gt}}$ denote the recorded future trajectory. The displacement error over the first $n$ waypoints is:

$$
\mathrm{ADE}_{n}\!\left(\boldsymbol{\tau}^{\mathrm{out}},\boldsymbol{\tau}^{\mathrm{gt}}\right)=\frac{1}{n}\sum_{k=1}^{n}\left\|(\hat{x}_{k},\hat{y}_{k})-(x^{\mathrm{gt}}_{k},y^{\mathrm{gt}}_{k})\right\|_{2},
$$

which uses the positional channels only, excludes heading, and is measured in meters. Abbreviating it as $\mathrm{ADE}_{n}$, we write the shifted and scaled displacement term as:

$$
\Delta(n;\delta,\kappa)=\frac{\delta-\mathrm{ADE}_{n}}{\kappa},
$$

where $\delta$ centers the term and $\kappa$ sets its scale, both in meters. Every reward below is affine in the displacement errors. For a displacement term $w\Delta(n;\delta,\kappa)$, $w$ denotes its weight. Because the group-relative advantage of Sec. 2.2 standardizes rewards within a group, an additive constant and a common positive factor leave the policy gradient unchanged. The offsets $\delta$ therefore affect only the logged reward magnitude, while the ratios $w/\kappa$ determine the relative influence of the terms.

##### NAVSIM.

The reward combines PDMS with the full-horizon displacement term:

$$
R_{\mathrm{NAVSIM}}=w_{\mathrm{pdms}}\,\mathrm{PDMS}+w_{\mathrm{ade}}\,\Delta(50;\delta,\kappa),
$$

with $w_{\mathrm{pdms}}=1$, $w_{\mathrm{ade}}=2$, $\delta=2$, and $\kappa=10$. The NAVSIM PDM scorer evaluates the PDMS term against the scene metric cache. Each rollout is subsampled to the 8 poses at 2 Hz over the 4 s scoring horizon that the scorer expects, and the simulation interval is 0.1 s. Within this reward, the sub-score weights of ego progress, time-to-collision, and comfort are 6, 4, and 2, whereas the evaluation reported in Sec. 3.3 follows the official protocol.

##### WOD-E2E.

The reward combines the Rater Feedback Score with the same displacement term:

$$
R_{\mathrm{WOD}}=w_{\mathrm{rfs}}\,\mathrm{RFS}+w_{\mathrm{ade}}\,\Delta(50;\delta,\kappa),
$$

with $w_{\mathrm{rfs}}=1$, $w_{\mathrm{ade}}=2$, $\delta=2$, and $\kappa=1$. The RFS term is computed using the official rater-feedback utility against the scene’s human preference trajectories and their rater scores. The rollout is linearly resampled to 20 positions at 4 Hz over 5 s to match the format the utility expects. Since RFS spans $[0,10]$ compared with $[0,1]$ for PDMS, we adopt a smaller $\kappa$ to keep the displacement term comparable in scale to the corresponding task reward.

##### PAI-AV.

PAI-AV defines no task-level score beyond displacement error, so its reward consists of displacement terms alone. Since near-term accuracy governs the immediate control decision, we combine the full horizon with shorter horizons under separate weights:

$$
R_{\mathrm{PAI}}=w_{0}\,\Delta(50;\delta,\kappa_{0})+\sum_{h\in\{1,2,3,4\}}w_{h}\,\Delta(10h;\delta,\kappa_{h}),
$$

with $\delta=2$, $w_{0}=1$, $\kappa_{0}=1$, weights $(w_{1},w_{2},w_{3},w_{4})=(5,4,3,2)$, and scales $(\kappa_{1},\kappa_{2},\kappa_{3},\kappa_{4})=(2,4,6,8)$. The horizon $h$ corresponds to the first $10h$ waypoints. The resulting coefficients $w_{h}/\kappa_{h}$ are $2.5$, $1$, $0.5$, and $0.25$ for the 1 s to 4 s horizons against $1$ for the full 5 s horizon, so an error in the first second carries the largest weight.

##### Summary.

Each source includes a displacement term. NAVSIM and WOD-E2E additionally use their task-level scores. The shared displacement term gives the three sources a comparable learning signal so that a single policy can be optimized across them. The multi-horizon refinement of Eq. 19 is applied to PAI-AV only, since the task-level scores provide the dominant learning signal for the other two sources.

## Appendix B Additional Visualizations

Beyond the comparisons in Fig. 8, this section illustrates a broader range of capabilities supported by Qwen-Drive-1.0-SFT, including camera-based 3D grounding, traffic-signal detection, roadwork detection, road element recognition, and reasoning-based motion planning. The sequence progresses from structured visual prediction to traffic-rule interpretation and planning. We include prompts and raw model responses throughout. Where annotations are available, predictions from Qwen-Drive-1.0-SFT and ground truth are overlaid on the input image in orange and blue, respectively.

### B.1 Camera-Based 3D Grounding

Camera-based 3D grounding requires recovering metric 3D boxes from a single image. In the urban turning scene below, Qwen-Drive-1.0-SFT localizes three vehicles and reports their box parameters in the requested format.

Prompt.

> Locate car in the provided image and output the corresponding 3D box. Camera intrinsics: fx=1862.42, fy=1861.99, cx=1917.51, cy=1078.37. Format: \[{"bbox\_3d":\[x\_center, y\_center, z\_center, x\_size, y\_size, z\_size, roll, pitch, yaw\],"label":"category"}\].

![[Uncaptioned image]](https://arxiv.org/html/2609.00111v1/figures/grounding_vis_wide_turn.jpg)

Response of Qwen-Drive-1.0-SFT.

> \[{"label": "car", "bbox\_3d": \[-3.52, 0.80, 8.56, 4.51, 1.81, 1.47, -0.17, -0.49, 0.67\]},  
> {"label": "car", "bbox\_3d": \[-0.35, 1.01, 16.80, 4.22, 1.81, 1.67, -0.17, -0.49, 0.67\]},  
> {"label": "car", "bbox\_3d": \[-3.45, 1.11, 22.00, 4.22, 1.80, 1.65, -0.17, -0.49, 0.67\]}\]

### B.2 Traffic-Signal Detection

At signalized intersections, detection must distinguish not only individual signals but also their type, orientation, and relevance to the current intersection. Qwen-Drive-1.0-SFT localizes twelve signals in the frame and returns the requested attributes for each instance.

Prompt.

> You are a visual annotation assistant for autonomous-driving intersection scenes. Given an image, detect and output all traffic-signal-related objects, focusing on the signal type (label), orientation (direction), and relevance to the current intersection (is\_related).
> 
> For every traffic-signal-related object in the image, output a dictionary containing:
> 
> – bbox\_2d: the 2D bounding box in the format \[x1, y1, x2, y2\] using integer pixel coordinates. The top-left corner of the image is (0,0), with x1 < x2 and y1 < y2. Make the box as tight as possible around the visible signal body.
> 
> – label: the signal type, one of "car" (motor-vehicle or non-motor-vehicle signal), "pedestrian" (pedestrian signal), "black" (black, unlit signal), "digit" (standalone countdown display), "led" (text LED sign), and "other" (another traffic signal or signal-control device).
> 
> – direction: the signal orientation, one of "front" (facing the ego vehicle), "back" (facing away from the ego vehicle), and "side" (clearly facing sideways).
> 
> – is\_related: whether the signal is relevant to the current intersection decision, one of "related" (controls traffic through the current intersection) and "unrelated" (a signal for a distant unrelated intersection, opposing traffic, or a side road).
> 
> Output exactly one JSON array of dictionaries with no extra explanatory text, wrapped in ‘‘‘json... ‘‘‘. If there are no traffic-signal-related objects, output an empty array. Output each object only once, and keep bounding-box coordinates as integers.

![[Uncaptioned image]](https://arxiv.org/html/2609.00111v1/figures/traffic_light_vis.jpg)

Response of Qwen-Drive-1.0-SFT.

> \[{"bbox\_2d":\[378,227,397,329\], "label":"car", "direction":"front", "is\_related":"related"},  
> {"bbox\_2d":\[430,227,448,326\], "label":"car", "direction":"front", "is\_related":"related"},  
> {"bbox\_2d":\[522,223,541,323\], "label":"car", "direction":"front", "is\_related":"related"},  
> {"bbox\_2d":\[569,221,589,317\], "label":"car", "direction":"front", "is\_related":"related"},  
> {"bbox\_2d":\[645,217,665,317\], "label":"car", "direction":"front", "is\_related":"related"},  
> {"bbox\_2d":\[703,214,721,314\], "label":"car", "direction":"front", "is\_related":"related"},  
> {"bbox\_2d":\[748,214,765,314\], "label":"car", "direction":"front", "is\_related":"related"},  
> {"bbox\_2d":\[817,211,834,308\], "label":"black", "direction":"front", "is\_related":"related"},  
> {"bbox\_2d":\[977,208,995,305\], "label":"black", "direction":"front", "is\_related":"related"},  
> {"bbox\_2d":\[846,388,862,466\], "label":"car", "direction":"front", "is\_related":"related"},  
> {"bbox\_2d":\[867,420,879,479\], "label":"car", "direction":"front", "is\_related":"related"},  
> {"bbox\_2d":\[880,420,891,479\], "label":"car", "direction":"front", "is\_related":"related"}\]

### B.3 Roadwork Detection

Work zones broaden the object vocabulary to temporary traffic-control devices and construction-related objects. In the scene below, Qwen-Drive-1.0-SFT identifies thirteen instances, including a work vehicle, tubular markers, and cones.

Prompt.

> Locate every instance that belongs to the category set in the image. The category set includes: Police Officer, Police Vehicle, Cone, Fence, Drum, Barricade, Barrier, Work Vehicle, Vertical Panel, Tubular Marker, Arrow Board, Bike Lane, Work Equipment, Worker, Other Roadwork Objects, Temporary Traffic Control Message Board, Temporary Traffic Control Sign, Stop sign. Report bbox coordinates in JSON format like this: \[{"bbox\_2d": \[x1, y1, x2, y2\], "label": "obj\_name"},...\].

![[Uncaptioned image]](https://arxiv.org/html/2609.00111v1/figures/roadwork_vis.jpg)

Response of Qwen-Drive-1.0-SFT.

> \[{"bbox\_2d":\[638,407,999,808\], "label":"Work Vehicle"},  
> {"bbox\_2d":\[616,601,653,766\], "label":"Tubular Marker"},  
> {"bbox\_2d":\[592,597,619,729\], "label":"Tubular Marker"},  
> {"bbox\_2d":\[653,607,703,836\], "label":"Tubular Marker"},  
> {"bbox\_2d":\[771,621,804,944\], "label":"Tubular Marker"},  
> {"bbox\_2d":\[398,620,415,677\], "label":"Cone"},  
> {"bbox\_2d":\[347,636,369,706\], "label":"Cone"},  
> {"bbox\_2d":\[257,663,295,773\], "label":"Cone"},  
> {"bbox\_2d":\[268,609,280,647\], "label":"Cone"},  
> {"bbox\_2d":\[249,611,262,657\], "label":"Cone"},  
> {"bbox\_2d":\[198,621,216,676\], "label":"Cone"},  
> {"bbox\_2d":\[83,640,111,714\], "label":"Cone"},  
> {"bbox\_2d":\[70,731,142,915\], "label":"Cone"}\]

### B.4 Road Element Recognition

Beyond object localization, Qwen-Drive-1.0-SFT further interprets the traffic-rule semantics of road elements. Here, a red box highlights a single dashed yellow line on a Chinese expressway, whose meaning must be selected from six candidate answers. Qwen-Drive-1.0-SFT selects the correct interpretation, while all eight comparison methods answer incorrectly.

Prompt.

> The visual data is from China. What is the meaning of the Single Dashed Yellow Line within the red box in the image? Please select one phrase from the list below as the answer \[’Vehicles must slow down and yield to main road vehicles or pedestrians’, ’Straight and right-turn’, ’Non-motorized vehicle lane’, ’Located on both sides of the road, indicating a dedicated lane’, ’Right-turn vehicle’, ’Road leading to North of West Outer Ring’\]. Reason carefully and step-by-step to ensure logical accuracy and robustness, including any relevant error checks. Finally, provide the final answer within \\boxed{}.

![[Uncaptioned image]](https://arxiv.org/html/2609.00111v1/figures/vladbench_pavement_marking.jpg)

Response of Qwen-Drive-1.0-SFT.

> To determine the meaning of the Single Dashed Yellow Line within the red box in the image, let us analyze the context and visual cues.
> 
> 1\. Visual context. The image is taken from a dashcam on a multi-lane highway or expressway in China (as indicated by Chinese characters on signs). The road surface has large white Chinese characters painted: "
> 
> 专用车道", which translates to "Dedicated Lane". The red box highlights a single dashed yellow line, which typically separates lanes with different traffic rules or designations. In China, yellow lines are often used to indicate special-purpose lanes, such as bus lanes, emergency vehicle lanes, or dedicated turn lanes, especially when combined with other markings.
> 
> 2\. Road marking analysis. The word "
> 
> 专用"means "dedicated" or "exclusive", and "
> 
> 车道"means "lane", so "
> 
> 专用车道"means "Dedicated Lane". The yellow line likely demarcates this dedicated lane from adjacent general-purpose lanes.
> 
> 3\. Evaluating the options. "Vehicles must slow down and yield to main road vehicles or pedestrians" describes a yield line or stop line, not a lane divider, and is incorrect. "Straight and right-turn" refers to directional arrows, not a yellow line, and is incorrect. "Non-motorized vehicle lane" is incorrect, because non-motorized vehicles usually have separate bike lanes often marked with blue or green paint and bicycle symbols, and yellow lines are rarely used for non-motorized lanes on highways. "Located on both sides of the road, indicating a dedicated lane" matches the analysis: the yellow line is part of a marking system that designates a dedicated lane ("
> 
> 专用车道"). Although the phrase says "on both sides", the key part is "indicating a dedicated lane", which aligns with the road marking text. "Right-turn vehicle" again refers to directionality rather than lane designation via a yellow line, and is incorrect. "Road leading to North of West Outer Ring" is a geographic description unrelated to lane markings, and is incorrect.
> 
> 4\. Conclusion. The only option that correctly interprets the purpose of the single dashed yellow line in context with the surrounding road markings ("
> 
> 专用车道") is "Located on both sides of the road, indicating a dedicated lane". Even though "both sides" may be slightly misleading (since we only see one side), the core meaning, indicating a dedicated lane, is accurate and matches the Chinese text on the road.
> 
> Final Answer: \\boxed{Located on both sides of the road, indicating a dedicated lane}

### B.5 Reasoning-Based Motion Planning

Finally, we turn from scene interpretation to future-motion prediction. Qwen-Drive-1.0-SFT supports predicting a trajectory with a one-sentence rationale. Given multiview temporal observations, the historical ego trajectory, and the active navigation command, Qwen-Drive-1.0-SFT produces a one-sentence rationale followed by future waypoints serialized as text. The ego vehicle follows the lead vehicle through a signalized intersection, and the predicted trajectory closely tracks the ground truth in both the camera view and the BEV visualization.

Input.

![[Uncaptioned image]](https://arxiv.org/html/2609.00111v1/figures/navsim_coc_traj_input.jpg)

Prompt.

> The input images are organized by camera view. Each view contains 4 temporal frames captured at 0.5s intervals (frame 0 at t-1.5s, frame 3 is the current frame at t=0s).
> 
> 1\. Historical trajectories (x, y, heading) in the current frame’s ego coordinate system. Positive x points forward, positive y points left, and a positive heading indicates a left turn:
> 
> \-t-1.5s: (-15.6548, 0.0502, 0.0069)  
> \-t-1.0s: (-10.3809, 0.0326, 0.0042)  
> \-t-0.5s: (-5.1587, 0.0116, 0.0023)  
> \-t-0s: (0.0000, 0.0000, 0.0000)
> 
> 2\. Active navigation command: \[GO STRAIGHT\]
> 
> First give a one-sentence brief reasoning of the ego’s future driving decision, then predict the corresponding possible trajectory (2 Hz, 4 s, 8 points, each point (x, y, heading) with 2-decimal precision). Report in JSON format like this: \[{"reasoning": "\<one concise sentence>"}, {"trajectory": "\[PT, (x, y, heading),...\]"}\].

Response of Qwen-Drive-1.0-SFT.

> \[{"reasoning": "Follow the lead vehicle through the intersection under the green traffic light."},  
> {"trajectory": "\[PT, (5.09, -0.01, 0.00), (10.12, -0.02, 0.00), (15.10, -0.03, 0.00), (20.03, -0.04, 0.00), (24.91, -0.04, 0.00), (29.74, -0.03, 0.00), (34.52, -0.02, 0.00), (39.24, -0.01, 0.00)\]"}\]

![[Uncaptioned image]](https://arxiv.org/html/2609.00111v1/figures/navsim_coc_traj_vis.jpg)

## Appendix C Detailed Driving VQA Cases

This appendix reports the complete prompts, ground-truth answers, and model responses for the four cases summarized in Fig. 8. Every model receives the same images in the same order. Two edits are applied to the raw outputs: markdown emphasis is typeset as text, and long chains of thought are abridged at positions marked with \[…\], where the omitted spans restate the surrounding reasoning without adding evidence.

### C.1 (a) Temporal Understanding and Agent-State Estimation (LingoQA)

> What you are seeing are frames from a video sequence. Your answer should focus on the final dynamic outcome, rather than analyzing the results of each individual frame.
> 
> How many parked vehicles can you see?

##### Ground truth.

*Zero.* The benchmark also accepts *None.*

Responses.

✓ Qwen-Drive-1.0-SFT

> There are no vehicles parked on either side of the road.

✗ Qwen3.5-4B

> In the provided video frames, I can see the following parked vehicles:
> 
> – A red bus is parked on the left side of the road.  
> – A white truck is parked on the right side of the road.
> 
> Therefore, there are 2 parked vehicles visible in the video sequence.

✗ MiMo-Embodied-7B

> \<think>  
> So, let’s look at the frames. In frame 5, there are parked vehicles on the right side. Let’s count: there’s a white van, a car, and maybe others? Wait, no, let’s check. \[… the trace repeats this self-correction for roughly four thousand characters …\] Wait, the correct answer is three.  
> \</think>
> 
> To determine the number of parked vehicles, we analyze the frames. In the final frame (frame 5), we observe parked vehicles on the right-hand side of the road. Counting these, we can identify three parked vehicles: a white van, a car, and another vehicle.
> 
> So, the number of parked vehicles is 3.

### C.2 (b) Causal Reasoning for Planning (PAI-AV-CoC)

> The input images are organized by camera view. Each view contains 4 temporal frames captured at 0.5s intervals (frame 0 at t-1.5s, frame 1 at t-1.0s, frame 2 at t-0.5s, frame 3 is the current frame at t=0s).
> 
> 1\. Historical trajectories (x, y, heading) in the current frame’s ego coordinate system. Positive x points forward, positive y points left, and a positive heading indicates a left turn:
> 
> \-t-1.5s: (-13.9670, -0.4070, 0.0427); -t-1.0s: (-8.9300, -0.1980, 0.0370); -t-0.5s: (-4.2570, -0.0560, 0.0219); -t-0s: (0.0000, 0.0000, 0.0000);
> 
> Give a one-sentence brief reasoning of the ego’s future driving decision ONLY.

##### Ground truth.

*Stop behind the lead vehicle at the stop sign at the intersection.*

Responses.

✓ Qwen-Drive-1.0-SFT

> Stop for the stop sign at the intersection.

✗ Cosmos-Reason2-32B

> The ego vehicle will continue straight, maintaining its lane position while navigating around the construction barriers on the right side of the road.

✗ Alpamayo-1.5-10B

> The ego vehicle is going straight at a steady speed

### C.3 (c) Cross-View Spatial Distance Perception (Ego3D-Bench)

> These are seven camera views mounted on an ego car. Front Left view: \<image>; Front view: \<image>; Front Right view: \<image>; Right view: \<image>; Back Right view: \<image>; Back Left view: \<image>; Left view: \<image>.
> 
> How far, in meters, is the dark colored sedan car parked on the side of the street in the back right view from the beige sedan car parked on the side of the street in the front view?
> 
> Output the thinking process and final answer (number ONLY) in \\boxed{} tags.

##### Ground truth.

22.93 m.

Responses.

✓ Qwen-Drive-1.0-SFT

> To determine the distance between the two cars, we need to analyze their positions in the provided views.
> 
> 1\. Identify the beige sedan car in the front view:  
> – In the \<FRONT VIEW>, there is a beige sedan parked on the right side of the street. It is positioned behind a white SUV and in front of a silver sedan (which is partially visible). The beige sedan is clearly visible with its rear facing the camera.
> 
> 2\. Identify the dark colored sedan car in the back right view:  
> – In the \<BACK RIGHT VIEW>, there is a dark gray or black Mercedes-Benz sedan parked on the left side of the street (from the perspective of the back right view). This car is parked on the curb next to some bushes and trees.
> 
> 3\. Estimate the distance:  
> – The two cars are on opposite sides of the street. The street appears to be a typical urban road with a width of about 6-8 meters (including the parking lanes).  
> – The beige sedan is parked on the right side, and the dark sedan is parked on the left side.  
> – Additionally, there is a significant distance along the street between the two cars. In the \<FRONT VIEW>, the beige sedan is relatively close to the camera. In the \<BACK RIGHT VIEW>, the dark sedan is further down the street, past several parked cars and trees.  
> – Based on the visual spacing and the typical dimensions of such streets, the distance between the two cars can be estimated. The width of the street alone adds about 6-8 meters. The longitudinal distance along the street appears to be at least 10-15 meters based on the perspective and the number of parked cars between them.
> 
> 4\. Calculate the total distance:  
> – Combining the transverse distance (width of the street) and the longitudinal distance (distance along the street), the total distance is approximately 20-25 meters.
> 
> Given the estimation, the distance is around 20-25 meters.
> 
> \\boxed{22}

✗ Cosmos-Reason2-32B

> Okay, let’s try to figure out the distance between the dark-colored sedan in the back right view and the beige sedan in the front view.
> 
> \[…\] Since both are on the same street but in different parts (front and back), they might be along the same road but separated by some distance. However, the problem doesn’t provide explicit measurements. \[…\]
> 
> But I’m not sure. Maybe the answer is 10 meters. Alternatively, maybe the answer is that they are parked in different parts of the street, so the distance is more than 10 meters.
> 
> Alternatively, perhaps the answer is that they are parked on the same side of the street but separated by a distance of approximately 15 meters.
> 
> Given that, I’ll go with 15 meters as the estimated distance.  
> \</think>
> 
> \\boxed{15}

✗ UniDriveVLA-8B

> The dark colored sedan car parked on the side of the street in the back right view is approximately 15 meters away from the beige sedan car parked on the side of the street in the front view. This estimation is based on the relative positions of the cars in the images, where the dark sedan is visible further down the street in the back right view, indicating a significant distance from the beige sedan in the front view. The visual evidence from the images supports the original GPT answer, confirming the distance as approximately 15 meters.

### C.4 (d) Traffic-Road Recognition (VLADBench)

> The following 5 frames are from China in chronological order. The visual data is from China. In the given autonomous driving image sequence, at the moment of the final image, what is the ego lane? Please choose the most appropriate answer from the following options: \[’straight lane’, ’left turn lane’, ’right turn lane’, ’straight and left turn lane’, ’straight and right turn lane’, ’U-turn lane’, ’left U-turn lane’\]. Reason carefully and step-by-step to ensure logical accuracy and robustness, including any relevant error checks. Finally, provide the final answer within \\boxed{}.

##### Ground truth.

*left turn lane*

Responses.

✓ Qwen-Drive-1.0-SFT

> To determine the ego lane at the moment of the final image (Frame 5), we need to analyze the road markings visible in the sequence, particularly in the earlier frames where they are clearer.
> 
> 1\. Observe Road Markings: In Frame 1 and Frame 2, the road surface has clear directional arrows painted on it.  
> – The lane the ego vehicle is currently in has a large arrow indicating a left turn.  
> – The lane to the right of the ego vehicle (where the black SUV and bus are) has an arrow indicating a right turn.
> 
> 2\. Track Vehicle Position: Throughout the sequence from Frame 1 to Frame 5, the ego vehicle remains in the same lane — the one with the left-turn arrow. It does not change lanes.
> 
> 3\. Traffic Light Context: The traffic light ahead is red, so all vehicles are stopped or slowing down. This confirms that the ego vehicle is waiting at an intersection in a designated turning lane.
> 
> 4\. Eliminate Other Options:  
> – “Straight lane” — incorrect, because the arrow is for turning.  
> – “Right turn lane” — that’s the adjacent lane.  
> – “Straight and left/right turn lane” — no such combined marking is visible; each lane has a single directional arrow.  
> – “U-turn lane” — not indicated by the arrow shape (it’s a standard left-turn arrow, not a U-turn symbol).
> 
> Therefore, based on the persistent left-turn arrow in the ego vehicle’s lane throughout the sequence, especially clear in Frame 1 and 2, the ego lane is the left turn lane.
> 
> \\boxed{left turn lane}

✗ UniDriveVLA-8B

> The ego lane is the straight lane. This is evident from the sequence of images where the vehicle maintains a consistent position in the lane without any indication of turning or changing lanes. The lane markings and the surrounding traffic flow also support this conclusion, as there are no signs of a turn or lane change in the final image. Therefore, the most appropriate answer is ’straight lane’.
> 
> \\boxed{straight lane}

✗ Alpamayo-1.5-10B

> The most appropriate answer is <c1,CAM\_FRONT,970.0,511.7>.

[^1]: N. Agarwal, A. Ali, J. Allen, M. Antolini, A. Aubame, A. Azzolini, J. Bai, M. Bala, Y. Balaji, J. Bapst, et al. Cosmos 3: omnimodal world models for physical ai. arXiv preprint arXiv:2606.02800. Cited by: §3.2.1.

[^2]: X. An, Y. Xie, F. Tang, Y. Yan, H. Tan, D. Zhu, C. Chen, X. Zhao, B. Qin, K. Yang, et al. Llava-onevision-2: towards next-generation perceptual intelligence. arXiv preprint arXiv:2605.25979. Cited by: §3.2.1.

[^3]: A. Azzolini, J. Bai, H. Brandon, J. Cao, P. Chattopadhyay, H. Chen, J. Chu, Y. Cui, J. Diamond, Y. Ding, et al. Cosmos-reason1: from physical common sense to embodied reasoning. arXiv preprint arXiv:2503.15558. Cited by: §3.2.1.

[^4]: M. Berman, A. R. Triki, and M. B. Blaschko The lovász-softmax loss: a tractable surrogate for the optimization of the intersection-over-union measure in neural networks. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit., pp. 4413–4421. Cited by: §2.1.

[^5]: H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom Nuscenes: a multimodal dataset for autonomous driving. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit., pp. 11618–11628. Cited by: §2.3.1.

[^6]: A. Cao and R. De Charette Monoscene: monocular 3d semantic scene completion. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit., pp. 3981–3991. Cited by: §2.1.

[^7]: X. Cao, T. Zhou, Y. Ma, W. Ye, C. Cui, K. Tang, Z. Cao, K. Liang, Z. Wang, J. M. Rehg, et al. Maplm: a real-world large-scale vision-language benchmark for map and traffic scene understanding. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit., pp. 21819–21830. Cited by: 1st item.

[^8]: K. Chen, Y. Li, W. Zhang, Y. Liu, P. Li, R. Gao, L. Hong, M. Tian, X. Zhao, Z. Li, et al. Automated evaluation of large vision-language models on self-driving corner cases. In Proc. IEEE Winter Conf. Appl. Comput. Vis., pp. 7806–7815. Cited by: 1st item.

[^9]: L. Chen, J. Li, X. Dong, P. Zhang, Y. Zang, Z. Chen, H. Duan, J. Wang, Y. Qiao, D. Lin, et al. Are we on the right way for evaluating large vision-language models?. In Proc. Adv. Neural Inf. Process. Syst., Vol. 37, pp. 27056–27087. Cited by: §3.2.2.

[^10]: X. Cheng, W. Zhang, S. Zhang, J. Yang, X. Guan, X. Wu, X. Li, G. Zhang, J. Liu, Y. Mai, et al. Simplevqa: multimodal factuality evaluation for multimodal large language models. In Proc. IEEE Int. Conf. Comput. Vis., pp. 4637–4646. Cited by: §3.2.2.

[^11]: H. Chi, H. Gao, Z. Liu, J. Liu, C. Liu, J. Li, K. Yang, Y. Yu, Z. Wang, W. Li, et al. Impromptu vla: open weights and open data for driving vision-language-action models. In Proc. Adv. Neural Inf. Process. Syst., Vol. 38. Cited by: 1st item.

[^12]: K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger Transfuser: imitation with transformer-based sensor fusion for autonomous driving. IEEE Trans. Pattern Anal. Mach. Intell. 45 (11), pp. 12878–12895. Cited by: Table 6.

[^13]: C. Corbière, S. Roburin, S. Montariol, A. Bosselut, and A. Alahi Drivingvqa: a dataset for interleaved visual chain-of-thought in real-world driving scenarios. In Findings of the Association for Computational Linguistics: EACL, pp. 3309–3333. External Links: [Document](https://dx.doi.org/10.18653/v1/2026.findings-eacl.173) Cited by: 1st item.

[^14]: D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, et al. Navsim: data-driven non-reactive autonomous vehicle simulation and benchmarking. In Proc. Adv. Neural Inf. Process. Syst., Vol. 37, pp. 28706–28719. Cited by: 2nd item, §2.3.3, §3.3.

[^15]: T. Deruyttere, S. Vandenhende, D. Grujicic, L. Van Gool, and M. F. Moens Talk2car: taking control of your self-driving car. In Proc. Conf. Empirical Methods in Natural Language Process., pp. 2088–2098. External Links: [Document](https://dx.doi.org/10.18653/v1/D19-1215) Cited by: 1st item.

[^16]: X. Ding, J. Han, H. Xu, X. Liang, W. Zhang, and X. Li Holistic autonomous driving understanding by bird’s-eye-view injected multi-modal large models. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit., pp. 13668–13677. Cited by: 1st item.

[^17]: M. Du, B. Wu, Z. Li, X. Huang, and Z. Wei Embspatial-bench: benchmarking spatial understanding for embodied tasks with large vision-language models. In Proc. Annual Meeting of the Association for Computational Linguistics, pp. 346–355. Cited by: §3.2.2.

[^18]: M. El Banani, A. Raj, K. Maninis, A. Kar, Y. Li, M. Rubinstein, D. Sun, L. Guibas, J. Johnson, and V. Jampani Probing the 3d awareness of visual foundation models. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit., pp. 21795–21806. Cited by: §1.

[^19]: H. Fang, S. Li, S. Wang, X. Xi, D. Liang, and X. Bai Towards generalizable robotic manipulation in dynamic environments. In Proc. Eur. Conf. Comput. Vis., Cited by: §2.1.

[^20]: J. Fang, L. Li, J. Zhou, J. Xiao, H. Yu, C. Lv, J. Xue, and T. Chua Abductive ego-view accident video understanding for safe driving perception. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit., pp. 22030–22040. Cited by: 1st item.

[^21]: L. Feng, Y. Gao, E. Zablocki, Q. Li, W. Li, S. Liu, M. Cord, and A. Alahi Rap: 3d rasterization augmented end-to-end planning. In Proc. Int. Conf. Learn. Representations, Cited by: 4(a).

[^22]: C. Fruhwirth-Reisinger, D. Malić, W. Lin, D. Schinagl, S. Schulter, and H. Possegger Stsbench: a spatio-temporal scenario benchmark for multi-modal large language models in autonomous driving. Proc. Adv. Neural Inf. Process. Syst. 38. Cited by: 1st item.

[^23]: H. Fu, D. Zhang, Z. Zhao, J. Cui, D. Liang, C. Zhang, D. Zhang, H. Xie, B. Wang, and X. Bai Orion: a holistic end-to-end autonomous driving framework by vision-language instructed action generation. In Proc. IEEE Int. Conf. Comput. Vis., pp. 24823–24834. Cited by: §1.

[^24]: H. Fu, D. Zhang, Z. Zhao, J. Cui, H. Xie, B. Wang, G. Chen, H. Ye, D. Liang, and X. Bai Minddrive: a vision-language-action model for autonomous driving via online reinforcement learning. In Proc. Eur. Conf. Comput. Vis., Cited by: §1.

[^25]: M. Gholami, A. Rezaei, Z. Weimin, S. Mao, S. Zhou, Y. Zhang, and M. Akbari Spatial reasoning with vision-language models in ego-centric multi-view scenes. In Proc. Int. Conf. Learn. Representations, Cited by: §3.2.1.

[^26]: A. Ghosh, S. Zheng, R. Tamburo, K. Vuong, J. Alvarez-Padilla, H. Zhu, M. Cardei, N. Dunn, C. Mertz, and S. G. Narasimhan Roadwork: a dataset and benchmark for learning to recognize, observe, analyze and drive through work zones. In Proc. IEEE Int. Conf. Comput. Vis., pp. 6132–6142. Cited by: 1st item.

[^27]: X. Guo, R. Zhang, Y. Duan, Y. He, D. Nie, W. Huang, C. Zhang, S. Liu, H. Zhao, and L. Chen Surds: benchmarking spatial understanding and reasoning in driving scenarios with vision language models. Proc. Adv. Neural Inf. Process. Syst. 38. Cited by: 1st item, §3.2.1.

[^28]: X. Hao, L. Zhou, Z. Huang, Z. Hou, Y. Tang, L. Zhang, G. Li, Z. Lu, S. Ren, X. Meng, et al. Mimo-embodied: x-embodied foundation model technical report. arXiv preprint arXiv:2511.16518. Cited by: §3.2.1.

[^29]: Y. Hao, Z. Li, L. Sun, W. Wang, N. Yi, S. Song, C. Qin, M. Zhou, Y. Zhan, and X. Lang Driveaction: a benchmark for exploring human-like driving decisions in vla models. arXiv preprint arXiv:2506.05667. Cited by: 1st item.

[^30]: K. He, X. Zhang, S. Ren, and J. Sun Deep residual learning for image recognition. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit., pp. 770–778. Cited by: §3.1.

[^31]: Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, et al. Planning-oriented autonomous driving. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit., pp. 17853–17862. Cited by: §1, 4(a).

[^32]: Y. Huang, B. Zhu, H. Lu, V. S. Huang, H. Zhang, W. Chen, J. Dai, Y. Xie, and H. Li Mindvla-u1: vla beats va with unified streaming architecture for autonomous driving. arXiv preprint arXiv:2605.12624. Cited by: 4(a), 4(a), 4(b), 4(b).

[^33]: Y. Inoue, Y. Yada, K. Tanahashi, and Y. Yamaguchi Nuscenes-mqa: integrated evaluation of captions and qa for autonomous driving datasets using markup annotations. In Proc. IEEE Winter Conf. Appl. Comput. Vis. Workshops, pp. 930–938. Cited by: 1st item.

[^34]: B. Jiang, S. Chen, B. Liao, X. Zhang, W. Yin, Q. Zhang, C. Huang, W. Liu, and X. Wang Senna: bridging large vision-language models and end-to-end autonomous driving. arXiv preprint arXiv:2410.22313. Cited by: 1st item.

[^35]: B. Jiang, S. Chen, Q. Xu, B. Liao, J. Chen, H. Zhou, Q. Zhang, W. Liu, C. Huang, and X. Wang Vad: vectorized scene representation for efficient autonomous driving. In Proc. IEEE Int. Conf. Comput. Vis., pp. 8306–8316. Cited by: 4(a).

[^36]: N. Karnchanachari, D. Geromichalos, K. S. Tan, N. Li, C. Eriksen, S. Yaghoubi, N. Mehdipour, G. Bernasconi, W. K. Fong, Y. Guo, et al. Towards learning-based planning: the nuplan benchmark for real-world autonomous driving. In Proc. IEEE Int. Conf. Robotics Automation, pp. 629–636. Cited by: §2.3.3.

[^37]: L. H. Li, P. Zhang, H. Zhang, J. Yang, C. Li, Y. Zhong, L. Wang, L. Yuan, L. Zhang, J. Hwang, et al. Grounded language-image pre-training. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit., pp. 10955–10965. Cited by: §3.2.2.

[^38]: Y. Li, H. Mao, R. Girshick, and K. He Exploring plain vision transformer backbones for object detection. In Proc. Eur. Conf. Comput. Vis., pp. 280–296. Cited by: §2.1.

[^39]: Y. Li, Y. Chen, X. Qi, Z. Li, J. Sun, and J. Jia Unifying voxel-based representation with transformer for 3d object detection. Proc. Adv. Neural Inf. Process. Syst. 35, pp. 18442–18455. Cited by: §2.1.

[^40]: Y. Li, K. Xiong, X. Guo, F. Li, S. Yan, G. Xu, L. Zhou, L. Chen, H. Sun, B. Wang, et al. Recogdrive: a reinforced cognitive framework for end-to-end autonomous driving. In Proc. Int. Conf. Learn. Representations, Cited by: 1st item, Table 6, Table 6.

[^41]: Y. Li, L. Zhou, S. Yan, B. Liao, T. Yan, K. Xiong, L. Chen, H. Xie, B. Wang, G. Chen, et al. Unidrivevla: unifying understanding, perception, and action planning for autonomous driving. arXiv preprint arXiv:2604.02190. Cited by: §3.2.1.

[^42]: Y. Li, M. Tian, Z. Lin, J. Zhu, D. Zhu, H. Liu, Y. Zhang, Z. Xiong, and X. Zhao Fine-grained evaluation of large vision-language models in autonomous driving. In Proc. IEEE Int. Conf. Comput. Vis., pp. 9431–9442. Cited by: §3.2.1.

[^43]: Z. Li, K. Li, S. Wang, S. Lan, Z. Yu, Y. Ji, Z. Li, Z. Zhu, J. Kautz, Z. Wu, et al. Hydra-mdp: end-to-end multimodal planning with multi-target hydra-distillation. arXiv preprint arXiv:2406.06978. Cited by: Table 6.

[^44]: Z. Li, W. Wang, H. Li, E. Xie, C. Sima, T. Lu, Q. Yu, and J. Dai Bevformer: learning bird’s-eye-view representation from lidar-camera via spatiotemporal transformers. IEEE Trans. Pattern Anal. Mach. Intell. 47 (3), pp. 2020–2036. Cited by: §2.1.

[^45]: B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang, et al. Diffusiondrive: truncated diffusion model for end-to-end autonomous driving. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit., pp. 12037–12047. Cited by: 4(b), Table 6.

[^46]: T. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár Focal loss for dense object detection. In Proc. IEEE Int. Conf. Comput. Vis., pp. 2980–2988. Cited by: §2.1.

[^47]: Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le Flow matching for generative modeling. In Proc. Int. Conf. Learn. Representations, Cited by: §2.1.

[^48]: H. Liu, Y. Chen, H. Wang, Z. Yang, T. Li, J. Zeng, L. Chen, H. Li, and L. Wang Fully sparse 3d occupancy prediction. In Proc. Eur. Conf. Comput. Vis., pp. 54–71. Cited by: §3.1.

[^49]: Y. Liu, T. Wang, X. Zhang, and J. Sun PETR: position embedding transformation for multi-view 3d object detection. In Proc. Eur. Conf. Comput. Vis., pp. 531–548. Cited by: §3.1.

[^50]: Y. Liu, J. Yan, F. Jia, S. Li, A. Gao, T. Wang, and X. Zhang Petrv2: a unified framework for 3d perception from multi-camera images. In Proc. IEEE Int. Conf. Comput. Vis., pp. 3262–3272. Cited by: §3.1.

[^51]: Y. Liu, H. Duan, Y. Zhang, B. Li, S. Zhang, W. Zhao, Y. Yuan, J. Wang, C. He, Z. Liu, et al. Mmbench: is your multi-modal model an all-around player?. In Proc. Eur. Conf. Comput. Vis., pp. 216–233. Cited by: §3.2.2.

[^52]: Y. Liu, Z. Li, M. Huang, B. Yang, W. Yu, C. Li, X. Yin, C. Liu, L. Jin, and X. Bai Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences 67 (12), pp. 220102. Cited by: §3.2.2.

[^53]: Y. Luo, Z. Yang, F. Meng, Y. Li, J. Zhou, and Y. Zhang An empirical study of catastrophic forgetting in large language models during continual fine-tuning. IEEE Trans. Audio, Speech, Language Process. 33, pp. 3776–3786. Cited by: §1.

[^54]: Y. Ma, Y. Cao, W. Ding, S. Zhang, Y. Wang, B. Ivanovic, M. Jiang, M. Pavone, and C. Xiao Dvlm-ad: enhance diffusion vision-language-model for driving via controllable reasoning. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit., pp. 1050–1061. Cited by: 4(b).

[^55]: S. Malla, C. Choi, I. Dwivedi, J. H. Choi, and J. Li Drama: joint risk localization and captioning in driving. In Proc. IEEE Winter Conf. Appl. Comput. Vis., pp. 1043–1052. Cited by: 1st item.

[^56]: A. Marcu, L. Chen, J. Hünermann, A. Karnsund, B. Hanotte, P. Chidananda, S. Nair, V. Badrinarayanan, A. Kendall, J. Shotton, et al. Lingoqa: visual question answering for autonomous driving. In Proc. Eur. Conf. Comput. Vis., pp. 252–269. Cited by: 1st item, §3.2.1.

[^57]: D. Marsili, R. Agrawal, Y. Yue, and G. Gkioxari Visual agentic ai for spatial reasoning with a dynamic api. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit., pp. 19446–19455. Cited by: §3.2.2.

[^58]: L. Nguyen, M. Fauth, B. Jaeger, D. Dauner, M. Igl, A. Geiger, and K. Chitta Open x-av: unifying end-to-end autonomous driving datasets. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit. Workshops, Cited by: 4(b).

[^59]: NVIDIA, Y. Cao, R. de Lutio, S. Fidler, G. Garcia Cobo, Z. Gojcic, M. Igl, B. Ivanovic, P. Karkus, J. Martinez Esturo, M. Pavone, A. Smith, E. Tanimura, M. Tyszkiewicz, M. Watson, Q. Wu, and L. Zhang AlpaSim: a modular, lightweight, and data-driven research simulator for autonomous driving. Note: [https://github.com/NVlabs/alpasim](https://github.com/NVlabs/alpasim) Cited by: §3.3, Table 7.

[^60]: NVIDIA Cosmos-reason2: an open and customizable reasoning vision language model for physical ai. Note: [https://github.com/nvidia-cosmos/cosmos-reason2](https://github.com/nvidia-cosmos/cosmos-reason2) Cited by: §3.2.1.

[^61]: NVIDIA Physical ai autonomous vehicles dataset. Note: [https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) Cited by: 1st item, 2nd item, §2.3.3, §3.3, §3.4.

[^62]: OpenScene Contributors OpenScene: the largest up-to-date 3d occupancy prediction benchmark in autonomous driving. Note: [https://github.com/OpenDriveLab/OpenScene](https://github.com/OpenDriveLab/OpenScene) Cited by: §2.3.1, §2.3.3.

[^63]: S. Park, C. Cui, Y. Ma, A. Moradipari, R. Gupta, K. Han, and Z. Wang Nuplanqa: a large-scale dataset and benchmark for multi-view driving scene understanding in multi-modal large language models. In Proc. IEEE Int. Conf. Comput. Vis., pp. 8066–8076. Cited by: 1st item.

[^64]: S. Park, G. Shin, J. Song, S. Lee, H. Shon, B. Park, J. Na, H. Jeong, and S. Hwang Swin-trajectory: technical report for 2025 waymo vision-based end-to-end driving challenge. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit. Workshops, Cited by: 4(b).

[^65]: J. Philion and S. Fidler Lift, splat, shoot: encoding images from arbitrary camera rigs by implicitly unprojecting to 3d. In Proc. Eur. Conf. Comput. Vis., pp. 194–210. Cited by: §2.1.

[^66]: T. Qian, J. Chen, L. Zhuo, Y. Jiao, and Y. Jiang Nuscenes-qa: a multi-modal visual question answering benchmark for autonomous driving scenario. In Proc. AAAI Conf. Artif. Intell., Vol. 38, pp. 4542–4550. Cited by: 1st item.

[^67]: Z. Qiao, H. Li, Z. Cao, and H. X. Liu Lightemma: lightweight end-to-end multimodal model for autonomous driving. arXiv preprint arXiv:2505.00284. Cited by: 4(b).

[^68]: Qwen Team Qwen3.5-4B model card. Note: [https://huggingface.co/Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) Cited by: §1, §2.1.

[^69]: Qwen Team Qwen3.5: towards native multimodal agents. External Links: [Link](https://qwen.ai/blog?id=qwen3.5) Cited by: §1, 1st item.

[^70]: Qwen Team Qwen3.7-Plus: multimodal agent intelligence. Note: [https://qwen.ai/blog?id=qwen3.7-plus](https://qwen.ai/blog?id=qwen3.7-plus) Cited by: 2nd item.

[^71]: I. Rawal, S. Gupta, Y. Hu, and W. Zhan NoRD: a data-efficient vision-language-action model that drives without reasoning. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit., pp. 10965–10975. Cited by: 4(b).

[^72]: L. Rowe, R. de Schaetzen, R. Girgis, C. Pal, and L. Paull Poutine: vision-language-trajectory pre-training and reinforcement learning post-training enable robust end-to-end autonomous driving. arXiv preprint arXiv:2506.11234. Cited by: 4(a).

[^73]: Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu, et al. Deepseekmath: pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300. Cited by: §2.2.

[^74]: Z. Sheng, X. Ye, J. Luo, S. Chen, and L. Ren Explorevla: dense world modeling and exploration for end-to-end autonomous driving. In Proc. Eur. Conf. Comput. Vis., Cited by: Table 6.

[^75]: C. Shi, J. Xu, S. Shi, K. Sheng, B. Zhang, and L. Jiang DriveWAM: video generative priors enable scalable world-action modeling for autonomous driving. arXiv preprint arXiv:2605.28544. Cited by: §3.3, Table 5, Table 7.

[^76]: C. Sima, K. Renz, K. Chitta, L. Chen, H. Zhang, C. Xie, J. Beißwenger, P. Luo, A. Geiger, and H. Li Drivelm: driving with graph visual question answering. In Proc. Eur. Conf. Comput. Vis., pp. 256–274. Cited by: §1, 1st item.

[^77]: P. Sun, H. Kretzschmar, X. Dotiwalla, A. Chouard, V. Patnaik, P. Tsui, J. Guo, Y. Zhou, Y. Chai, B. Caine, et al. Scalability in perception for autonomous driving: waymo open dataset. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit., pp. 2443–2451. Cited by: 2nd item.

[^78]: J. S. Tamarapalli, R. Grover, N. Pande, and S. Yerramilli CountQA: how well do mllms count in the wild?. arXiv preprint arXiv:2508.06585. Cited by: §3.2.2.

[^79]: G. R. Team, S. Abeyruwan, J. Ainslie, J. Alayrac, M. G. Arenas, T. Armstrong, A. Balakrishna, R. Baruch, M. Bauza, M. Blokzijl, et al. Gemini robotics: bringing ai into the physical world. arXiv preprint arXiv:2503.20020. Cited by: §3.2.2.

[^80]: G. Team, S. E. Abd, V. Aggarwal, R. Algayres, A. Andreev, O. Bachem, I. Ballantyne, C. Brick, V. Cărbune, M. Casbon, et al. Gemma 4 technical report. arXiv preprint arXiv:2607.02770. Cited by: §3.2.1.

[^81]: W. Tong, C. Sima, T. Wang, L. Chen, S. Wu, H. Deng, Y. Gu, L. Lu, P. Luo, D. Lin, et al. Scene as occupancy. In Proc. IEEE Int. Conf. Comput. Vis., pp. 8372–8381. Cited by: §2.3.1.

[^82]: D. Wang, Y. Song, Z. He, K. Chen, X. Pan, L. Deng, and W. Gu Hmvlm: multistage reasoning-enhanced vision-language model for long-tailed driving scenarios. arXiv preprint arXiv:2506.05883. Cited by: 4(b).

[^83]: Q. Wang, M. Li, J. Guan, J. Ye, S. Xie, Y. Liu, J. Chen, Z. Liang, J. Zhang, X. Hu, et al. Qwen-vla: unifying vision-language-action modeling across tasks, environments, and robot embodiments. arXiv preprint arXiv:2605.30280. Cited by: §1.

[^84]: S. Wang, Z. Yu, X. Jiang, S. Lan, M. Shi, N. Chang, J. Kautz, Y. Li, and J. M. Alvarez Omnidrive: a holistic vision-language dataset for autonomous driving with counterfactual reasoning. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit., pp. 22442–22452. Cited by: §1, 1st item.

[^85]: W. Wang, Z. Gao, L. Gu, H. Pu, L. Cui, X. Wei, Z. Liu, L. Jing, S. Ye, J. Shao, et al. Internvl3. 5: advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265. Cited by: §3.2.1.

[^86]: Y. Wang, W. Luo, J. Bai, Y. Cao, T. Che, K. Chen, Y. Chen, J. Diamond, Y. Ding, W. Ding, et al. Alpamayo-r1: bridging reasoning and action prediction for generalizable autonomous driving in the long tail. arXiv preprint arXiv:2511.00088. Cited by: 2nd item, §3.2.1, §3.3, Table 5, Table 5, Table 7, Table 7.

[^87]: Z. Wang, M. Xia, L. He, H. Chen, Y. Liu, R. Zhu, K. Liang, X. Wu, H. Liu, S. Malladi, et al. Charxiv: charting gaps in realistic chart understanding in multimodal llms. In Proc. Adv. Neural Inf. Process. Syst., Vol. 37, pp. 113569–113697. Cited by: §3.2.2.

[^88]: Q. Wu, J. M. Esturo, A. Mirzaei, N. Moenne-Loccoz, and Z. Gojcic 3dgut: enabling distorted cameras and secondary rays in gaussian splatting. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit., pp. 26036–26046. Cited by: §3.3, Table 7.

[^89]: X. Wu, D. Liang, T. Feng, K. Xia, Y. Zhang, X. Li, X. Tan, and X. Bai Generation models know space: unleashing implicit 3d priors for scene understanding. In Proc. Eur. Conf. Comput. Vis., Cited by: §1.

[^90]: xAI Grok-1.5 vision preview. Note: [https://x.ai/news/grok-1.5v](https://x.ai/news/grok-1.5v) Cited by: §3.2.2.

[^91]: J. Xu, Z. Zhong, Z. Shu, M. Jia, M. Li, J. Bian, Q. Zhang, K. Zhang, J. Xie, J. Yang, et al. EponaV2: driving world model with comprehensive future reasoning. arXiv preprint arXiv:2605.14696. Cited by: Table 6.

[^92]: L. Xu, H. Huang, and J. Liu Sutd-trafficqa: a question answering benchmark and an efficient network for video reasoning over traffic events. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit., pp. 9878–9888. Cited by: 1st item.

[^93]: R. Xu, H. Lin, W. Jeon, H. Feng, Y. Zou, L. Sun, J. Gorman, K. Tolstaya, S. Tang, B. White, et al. Wod-e2e: waymo open dataset for end-to-end driving in challenging long-tail scenarios. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit., pp. 3709–3718. Cited by: §2.3.3, §3.3, §3.4, §3.4, 4(b).

[^94]: Z. Xu, Y. Zhang, E. Xie, Z. Zhao, Y. Guo, K. K. Wong, Z. Li, and H. Zhao Drivegpt4: interpretable end-to-end autonomous driving via large language model. IEEE Robotics and Automation Letters 9 (10), pp. 8186–8193. Cited by: §1, 1st item.

[^95]: C. Yang, Y. Chen, H. Tian, C. Tao, X. Zhu, Z. Zhang, G. Huang, H. Li, Y. Qiao, L. Lu, et al. Bevformer v2: adapting modern image backbones to bird’s-eye-view recognition via perspective supervision. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit., pp. 17830–17839. Cited by: §2.1, §3.1.

[^96]: J. Yang, S. Yang, A. W. Gupta, R. Han, L. Fei-Fei, and S. Xie Thinking in space: how multimodal large language models see, remember, and recall spaces. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit., pp. 10632–10643. Cited by: §1.

[^97]: S. Yu, S. Lee, N. Kim, J. Shin, J. Park, W. Ryu, R. Jung, and H. Shim WaymoQA: a multi-view visual question answering dataset for safety-critical reasoning in autonomous driving. arXiv preprint arXiv:2511.20022. Cited by: 1st item, §3.2.1.

[^98]: Z. Yu, C. Shu, J. Deng, K. Lu, Z. Liu, J. Yu, D. Yang, H. Li, and Y. Chen Flashocc: fast and memory-efficient occupancy prediction via channel-to-height plugin. arXiv preprint arXiv:2311.12058. Cited by: §2.1.

[^99]: C. Yuan, Z. Zhang, J. Sun, S. Sun, Z. Huang, C. D. W. Lee, D. Li, Y. Han, A. Wong, K. P. Tee, et al. Drama: an efficient end-to-end motion planner for autonomous driving with mamba. arXiv preprint arXiv:2408.03601. Cited by: Table 6.

[^100]: X. Yue, Y. Ni, K. Zhang, T. Zheng, R. Liu, G. Zhang, S. Stevens, D. Jiang, W. Ren, Y. Sun, et al. Mmmu: a massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit., pp. 9556–9567. Cited by: §3.2.2.

[^101]: X. Yue, T. Zheng, Y. Ni, Y. Wang, K. Zhang, S. Tong, Y. Sun, B. Yu, G. Zhang, H. Sun, et al. Mmmu-pro: a more robust multi-discipline multimodal understanding benchmark. In Proc. Annual Meeting of the Association for Computational Linguistics, pp. 15134–15186. Cited by: §3.2.2.

[^102]: X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer Sigmoid loss for language image pre-training. In Proc. IEEE Int. Conf. Comput. Vis., pp. 11941–11952. Cited by: §3.1.

[^103]: Y. Zhai, S. Tong, X. Li, M. Cai, Q. Qu, Y. J. Lee, and Y. Ma Investigating the catastrophic forgetting in multimodal large language model fine-tuning. In Proc. Conf. Parsimony Learn., pp. 202–227. Cited by: §1.

[^104]: K. Zhang, Z. Tang, X. Hu, X. Pan, X. Guo, Y. Liu, J. Huang, L. Yuan, Q. Zhang, X. Long, et al. Epona: autoregressive diffusion world model for autonomous driving. In Proc. IEEE Int. Conf. Comput. Vis., pp. 27220–27230. Cited by: Table 6.

[^105]: Z. Zhao, H. Fu, D. Liang, X. Zhou, D. Zhang, H. Xie, B. Wang, and X. Bai Extending large vision-language model for diverse interactive tasks in autonomous driving. IEEE Transactions on Image Processing. Cited by: §1.

[^106]: Z. Zhao, X. Zhou, T. Xu, Z. Sun, K. Zhou, H. Li, D. Liang, and X. Bai SimWAM: a simple world action model for end-to-end autonomous driving. arXiv preprint arXiv:2608.07468. Cited by: §3.3, Table 5, Table 7.

[^107]: E. Zhou, J. An, C. Chi, Y. Han, S. Rong, C. Zhang, P. Wang, Z. Wang, T. Huang, L. Sheng, et al. Roborefer: towards spatial referring with reasoning in vision-language models for robotics. In Proc. Adv. Neural Inf. Process. Syst., Vol. 38, pp. 28404–28481. Cited by: §3.2.2.

[^108]: X. Zhou, D. Liang, X. Chen, F. Tan, D. Zhang, H. Zhao, and X. Bai HERMES++: toward a unified driving world model for 3d scene understanding and generation. arXiv preprint arXiv:2604.28196. Cited by: §1.

[^109]: X. Zhou, D. Liang, S. Tu, X. Chen, Y. Ding, D. Zhang, F. Tan, H. Zhao, and X. Bai Hermes: a unified self-driving world model for simultaneous 3d scene understanding and generation. In Proc. IEEE Int. Conf. Comput. Vis., pp. 27817–27827. Cited by: §1.

[^110]: Z. Zhou, T. Cai, S. Zhao, Y. Zhang, Z. Huang, B. Zhou, and J. Ma Autovla: a vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. Proc. Adv. Neural Inf. Process. Syst. 38, pp. 27920–27956. Cited by: 4(b), Table 6, Table 6.

[^111]: Z. Zhou, R. Yang, Y. Guo, S. X. Chen, T. Feng, K. Pistunova, Y. Shen, L. Su, J. Ma, et al. Spanvla: efficient action bridging and learning from negative-recovery samples for vision-language-action model. arXiv preprint arXiv:2604.19710. Cited by: Table 6, Table 6.

[^112]: X. Zhu, W. Su, L. Lu, B. Li, X. Wang, and J. Dai Deformable detr: deformable transformers for end-to-end object detection. In Proc. Int. Conf. Learn. Representations, Cited by: §2.1.
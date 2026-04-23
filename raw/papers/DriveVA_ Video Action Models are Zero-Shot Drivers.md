---
title: "DriveVA: Video Action Models are Zero-Shot Drivers"
source: "https://arxiv.org/html/2604.04198v1"
author:
published:
created: 2026-04-23
description:
tags:
  - "clippings"
---
<sup>1</sup>

Mengmeng Liu    Diankun Zhang    Jiuming Liu    Jianfeng Cui     
Hongwei Xie    Guang Chen    Hangjun Ye    Michael Ying Yang     
Francesco Nex    Hao Cheng Corresponding author. Emails: {m.liu-1, h.cheng-2}@utwente.nl

###### Abstract

Generalization is a central challenge in autonomous driving, as real-world deployment requires robust performance under unseen scenarios, sensor domains, and environmental conditions. Recent world-model-based planning methods have shown strong capabilities in scene understanding and multi-modal future prediction, yet their generalization across datasets and sensor configurations remains limited. In addition, their loosely coupled planning paradigm often leads to poor video-trajectory consistency during visual imagination. To overcome these limitations, we propose DriveVA, a novel autonomous driving world model that jointly decodes future visual forecasts and action sequences in a shared latent generative process. DriveVA inherits rich priors on motion dynamics and physical plausibility from well-pretrained large-scale video generation models to capture continuous spatiotemporal evolution and causal interaction patterns. To this end, DriveVA employs a DiT-based decoder to jointly predict future action sequences (trajectories) and videos, enabling tighter alignment between planning and scene evolution. We also introduce a video continuation strategy to strengthen long-duration rollout consistency. DriveVA achieves an impressive closed-loop performance of 90.9 PDM score on the challenge NAVSIM. Extensive experiments also demonstrate the zero-shot capability and cross domain generalization of DriveVA, which reduces average L2 error and collision rate by 78.9% and 83.3% on nuScenes and 52.5% and 52.4% on the Bench2drive built on CARLA v2 compared with the state-of-the-art world-model-based planner.

![[x1 24.png|Refer to caption]]

Figure 1: DriveVA: unified video–trajectory rollout for planning. Given history frames, rolls out a future video clip (top). The ego trajectory is generated together with the video rollout and remains aligned with the visual scene evolution (middle). Bottom: zero-shot comparisons trained on NAVSIM and evaluated on nuScenes (cross dataset) and CARLA (cross domain from real to simulation), showing large relative improvements over PWM \[ zhao2025forecasting \] in displacement error and collision rate.

## 1 Introduction

Generalization has long been a fundamental goal in autonomous driving, as it is essential for building systems that operate reliably in the real world \[bogdoll2021description, hao2025driveaction, chi2025impromptu\]. A capable autonomous driving model should not only perform well on scenarios seen during training, but also remain robust under unseen traffic patterns, novel road layouts, and diverse sensor configurations \[li2023domain, zhou2025opendrivevla, hu2025vlm\]. This ability is especially important for real-world deployment, where long-tail events, domain shifts, and complex agent interactions are common. Recent advances in large-scale pre-trained models have motivated researchers to develop autonomous driving systems that can better transfer across tasks and domains \[yang2025drivemoe\]. This trend has led to the development of Vision-Language-Action (VLA) models \[xu2024drivegpt4, zhou2025opendrivevla, yang2025drivemoe, zhou2025\_hermes, li2025drivevlaw0, zhou2025autovla, wang2025adawm, zheng2025world4drive, hao2025mapfusion\], which leverage pre-trained vision-language models as the backbone and fine-tune them on driving-specific trajectory data. This strategy can reduce the amount of task-specific training data required while still achieving strong planning performance. However, despite these advances, true generalization, especially zero-shot transfer across datasets, remains limited and has yet to be fully realized. A key reason is that prevailing VLA pretraining on static image–text pairs primarily transfers semantic knowledge (“what is what”), but provides limited spatiotemporal and causal priors (“how the world moves”) needed for robust closed-loop planning.

Recently, large-scale video generation models \[zheng2024opensora, yang2024cogvideox, kong2024hunyuanvideo, wan2025\] have shown strong generalization to unseen textual prompts and visual contexts. By learning from massive video corpora, they capture realistic motion patterns and physically plausible scene dynamics \[kong2024hunyuanvideo, wan2025\], suggesting rich priors over real-world temporal evolution. Notably, the ability of video generators to produce temporally coherent future predictions under flexible conditioning aligns closely with the goal of building generalizable driving world models. This motivates a key question: *"Can large video generation models serve as a foundation for generalizable autonomous driving video action models?"*

To answer this question, we investigate how to build and fine-tune autonom-ous driving models upon large-scale video generation models. Existing world-model-based planning methods suffer from two major bottlenecks. First, they often exhibit limited generalization across diverse datasets as the world knowledge learned from one dataset is difficult to transfer effectively to another. Second, they commonly suffer from inconsistency between visual and action rollouts, because video imagination and trajectory generation are typically modeled separately or only loosely coupled \[xia2025drivelaw, zhang2025epona\]. To bridge the gap between generic video generation and driving-oriented planning, the key challenge is to enable video generation models not only to synthesize plausible future scenes, but also to produce actionable driving trajectories that can directly guide vehicle planning. Moreover, to effectively transfer the strong generalization capability of video generation models from the visual domain to the planning domain, it is crucial to maintain consistency between the predicted driving trajectories and the visual future evolution represented in the generated videos, as illustrated by the qualitative comparisons in Fig. 3 and Fig. 4. Such alignment allows the semantic understanding and physical priors learned from large-scale video data to be naturally extended to autonomous driving behaviors.

In this paper, we propose a video-action model, called DriveVA, which integrates large-scale video priors with end-to-end planning and dense supervision from world modeling. We find that video-level supervision is the main driver of planning gains, rather than a merely auxiliary loss appended to a cascaded pipeline as in most existing methods \[xia2025drivelaw, zheng2025world4drive, zeng2025futuresightdrive, li2025drivevlaw0\]. Concretely, enabling video supervision boosts NAVSIMv1 PDMS from 71.4 to 90.9 (+19.5) over action-only optimization (Table 5.5). The key is that video supervision provides dense temporal grounding of scene dynamics, and planning benefits only when the predicted actions are forced to stay consistent with the imagined future \[ye2026world, shen2025videovla\]. As shown in Fig. 1, this motivates our unified formulation: instead of modeling future visual imagination and trajectory generation in separate stages, DriveVA places future video latents and action tokens in a shared latent generative process and jointly decodes them with a single DiT \[peebles2023DIT\] in a shared latent space, so trajectories are generated as action grounding of the same rollout rather than being optimized in a separate stage. This unified formulation yields tighter video–trajectory alignment and more coherent long-horizon rollouts. Despite being generative, we observe that as few as two sampling steps already reach near-optimal closed-loop performance, enabling efficient recurrent decision making. We further introduce a video continuation module to maintain long-duration consistency by progressively rolling out future video clips. Extensive experiments demonstrate that DriveVA achieves state-of-the-art closed-loop performance on NAVSIM, and also transfers strongly to unseen datasets across real driving scenes (e.g., nuScenes) and simulated scenes (e.g., Bench2Drive) in a zero-shot setting without target-domain fine-tuning, proving DriveVA’s excellent generalizability.

Overall, our core contributions are as follows:

- We propose DriveVA, a unified video-action world model for autonomous driving that jointly models future visual imagination and trajectory prediction within a shared latent generative process, alleviating the mismatch caused by cascaded or loosely coupled planning pipelines.
- We design a unified DiT-based decoder that simultaneously generates future video latents and action tokens, leading to stronger video-trajectory consistency and tighter alignment between scene evolution and planned behavior.
- We introduce a video continuation module that progressively rolls out future clips, preserving long-horizon structural consistency during recurrent planning.
- Extensive experiments show that DriveVA achieves state-of-the-art closed-loop performance on NAVSIM (90.9 PDMS), and delivers strong zero-shot evaluation on nuScenes (trained on NAVSIM) with 78.9% lower average L2 error and 83.3% lower collision rate than the state of the art. It also improves generalization from real to simulation on Bench2Drive (CARLA), reducing average L2 error by 52.5% and collision rate by 52.4%.

## 2 Related Work

### 2.1 Vision Language Action Models

VLAs. Recently, the rapid development of vision-language-action (VLA) models \[xu2024drivegpt4, zhou2025opendrivevla, yang2025drivemoe, zhou2025\_hermes, li2025drivevlaw0, zhou2025autovla, wang2025adawm, zheng2025world4drive, hao2025mapfusion, fu2025minddrive, fu2025orion, luo2025adathinkdrive\] has advanced a new paradigm for language-guided autonomous driving: these models jointly integrate language understanding, environment perception, and vehicle control to accomplish driving tasks in an end-to-end manner. This progress has been largely enabled by the continued evolution of vision \[radford2021learning\_clip, zhai2023sigmoid\_siglip, oquab2023dinov2\], language \[touvron2023llama, team2024gemma, abdin2024phi3\], and vision-language \[liu2023visual\_llava, chen2024internvl, wang2024qwen2vl\] foundation models. Despite this progress, most existing driving VLAs are still built upon vision-language models (VLMs) pre-trained on large-scale web data. While such models are effective at transferring visual-semantic knowledge, their pretraining data is primarily composed of static image-text pairs, which limits their ability to capture temporal dynamics and physical interaction patterns directly; They do not naturally inherit the spatiotemporal priors required for adapting to new complex interactive scenarios. Consequently, the generalization ability of current driving VLAs, especially when faced with unseen scenarios and unseen behaviors, still exhibits clear limitations \[zhou2025opendrivevla\].

Generalization in VLAs. To address the generalization issue, existing driving VLA methods mainly follow two directions: one focuses on targeted data construction for corner cases, while the other relies on structured expert modeling for long-tail behaviors \[hao2025driveaction, hu2025vlm, zhou2025opendrivevla\]. However, stronger *zero-shot* generalization remains insufficiently addressed. For example, Impromptu VLA \[chi2025impromptu\] improves robustness through a manually curated corner-case dataset \[hu2025vlm, hao2025driveaction\], but relies on predefined scenario categories and trajectory-centric supervision, limiting true cross-dataset zero-shot transfer. DriveMoE \[yang2025drivemoe\], in contrast, addresses rare and long-tail driving behaviors through scene- and skill-specialized experts \[zhou2025opendrivevla\], yet still depends on predefined skill partitions and benchmark-specific data distributions, with limited evidence of transfer to unseen platforms or environments. We argue that *zero-shot driving capability* is particularly critical for planning, as it more directly measures whether a model can make reliable decisions when encountering unseen corner cases rather than merely interpolating among observed trajectory patterns, and also serves as an indicator of cross-platform and cross-scenario generalization. In contrast, video-based world models can leverage dense frame-level supervision to learn physical dynamics from visual evolution, offering a more scalable path toward generalization beyond fixed action templates, benchmark-specific skill partitions, and manually defined corner-case taxonomies \[wang2024driving, yang2024generalized, zheng2025world4drive, li2025drivevlaw0, gao2024vista\].

### 2.2 Video Model-based Autonomous Driving

Motivated by intuitive physical reasoning, world models aim to improve driving decisions by forecasting future scene evolution. Existing autonomous-driving world models can be broadly divided into two lines: latent-dynamics models for planning \[zheng2025world4drive, yang2025raw2drive, wang2025adawm\] or reinforcement learning \[feng2025survey\_dwm\_survey\], and models that explicitly predict future visual observations for decision making \[zeng2025futuresightdrive, zhang2025epona, li2025drivevlaw0\].

Latent-dynamics methods, such as LAW \[li2024enhancing\], World4Drive \[zheng2025world4drive\], AdaWM \[wang2025adawm\], and Raw2Drive \[yang2025raw2drive\], learn compact future representations for planning, policy optimization, or robustness, but generally treat the world model as an auxiliary module for supervision or planning guidance rather than using explicit visual rollouts at inference. In contrast, visually predictive approaches, including FutureSightDrive \[zeng2025futuresightdrive\], Epona \[zhang2025epona\], DriveVLA-W0 \[li2025drivevlaw0\], and DriveLaW \[xia2025drivelaw\], more directly exploit future visual prediction for planning. However, existing methods still typically treat visual prediction as an auxiliary signal, an intermediate reasoning process, or a module loosely coupled with planning. Even in methods that connect the two more explicitly, video and action generation are often maintained as separate branches, with consistency relying on inter-module feature transfer or multi-stage optimization. As a result, mismatches between imagined futures and generated actions can accumulate over time, causing the executed actions to deviate from the future evolution predicted by the world model.

Built on video diffusion backbones, VAM-style approaches \[ye2026world\] offer a promising direction by leveraging strong spatiotemporal priors from web-scale video data. Unlike latent world models that learn dynamics from scratch in compact latent spaces \[hafner2019dream, hafner2020mastering, hafner2023mastering, assran2025vjepa2\], they can directly exploit pretrained video representations that already encode rich physical dynamics. This suggests that a unified generative formulation of future video and actions may provide tighter coupling between visual forecasting and planning, while also improving transfer across data domains. Motivated by this observation, our method adopts a single shared generative process to jointly model future imagination and action generation, and further investigates how data scale and diversity affect generalization in autonomous driving.

![[x2 22.png|Refer to caption]]

Figure 2: Overall pipeline of DriveVA. Given history observations, the ego state (current velocity vx, vy), and language instructions, the model first encodes conditional signals into latent tokens through a text encoder and a video VAE \[ wan2025 \]. A unified diffusion transformer (DiT) peebles2023DIT then jointly predicts future video latents and future action tokens in a shared generative process, ensuring strong video–trajectory consistency. To maintain long-horizon temporal coherence, a progressive video continuation strategy recursively rolls out future video clips while updating predicted trajectories.

## 3 Preliminary

Flow Matching. Flow matching \[lipman2022flow, liu2022flow, tong2024improving\] models generation as a continuous-time transformation from a simple source distribution to the target data distribution. Let $x_{\mathrm{data}}\in\mathbb{R}^{d}$ be a data sample and $\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ be a noise sample. The model learns a time-dependent velocity field $v_{\theta}:\mathbb{R}^{d}\times[0,1]\rightarrow\mathbb{R}^{d}$ that defines the dynamics of a trajectory $x^{(s)}$ via

$$
\frac{dx^{(s)}}{ds}=v_{\theta}\!\left(x^{(s)},s\right),\qquad x^{(0)}=\boldsymbol{\epsilon},\quad s\in[0,1].
$$

Intuitively, the learned flow transports samples from noise at $s=0$ to the data manifold at $s=1$.

For training, flow matching supervises the model on a prescribed interpolation path between $\boldsymbol{\epsilon}$ and $x_{\mathrm{data}}$. We use the standard linear interpolation $x^{(s)}=(1-s)\boldsymbol{\epsilon}+sx_{\mathrm{data}}$, whose derivative is $\dot{x}^{(s)}=x_{\mathrm{data}}-\boldsymbol{\epsilon}$. The network $v_{\theta}$ is trained to regress this target velocity:

$$
\mathcal{L}_{\mathrm{FM}}=\mathbb{E}_{s,\boldsymbol{\epsilon},x_{\mathrm{data}}}\left[\left\|v_{\theta}\!\left(x^{(s)},s\right)-\dot{x}^{(s)}\right\|_{2}^{2}\right].
$$

At inference, generation starts from Gaussian noise and integrates the learned velocity field from $s=0$ to $s=1$:

$$
x^{(1)}=x^{(0)}+\int_{0}^{1}v_{\theta}\!\left(x^{(s)},s\right)\,ds,\qquad x^{(0)}\sim\mathcal{N}(\mathbf{0},\mathbf{I}).
$$

Video Generation with Conditional Flow Matching. Recent video generators \[brooks2024video, wan2025\] commonly perform flow matching in the latent space of a pretrained video autoencoder. Let $E$ and $D$ denote the encoder and decoder, respectively. Given a conditioning signal $c$, we aim to generate a latent video sequence $\mathbf{z}=\{z_{1},\ldots,z_{T_{v}}\}$ and decode it to pixels with $D$.

Conditional flow matching learns a velocity field $v_{\theta}(\mathbf{z}^{(s)},s\mid c)$ that defines the latent dynamics $\frac{d\mathbf{z}^{(s)}}{ds}=v_{\theta}(\mathbf{z}^{(s)},s\mid c)$ with $\mathbf{z}^{(0)}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$. Integrating from $s\!=\!0$ to $1$ yields the clean latent $\mathbf{z}^{(1)}$, which is then decoded by $D$. This latent-space formulation is efficient and well-suited for long-horizon, condition-controlled video synthesis.

## 4 Method

### 4.1 Problem Formulation

Given a language instruction $\mathcal{T}$ (including the high-level command) and a history observation buffer $\mathcal{O}_{l}=\{\mathbf{F}_{l-m+1},\ldots,\mathbf{F}_{l}\}$, which contains $m$ -frame history observations from $\mathbf{F}_{l-m+1}$ to $\mathbf{F}_{l}$. Our goal at the current timestep $l$ is to jointly predict future actions (trajectories) and future visual imaginations. Specifically, conditioned on $\mathcal{T}$, the current ego state $\mathbf{q}_{l}$ (represented by the ego velocity components $v_{x}$ and $v_{y}$), and the visual history observations $\mathcal{O}_{l}$, we predict:

1. An action chunk $\mathcal{A}_{l+1:l+K}=\{\boldsymbol{a}_{l+i}\in\mathbb{R}^{3}\}_{i=1}^{K}$ consisting of $K$ future actions to be executed sequentially, where each action $\boldsymbol{a}_{l+i}$ is a 3-D vector. The first two dimensions encode the ego-vehicle $(x,y)$ position, and the last dimension encodes the yaw angle.
2. A future video clip $\mathcal{F}_{l+1:l+N}=\{\boldsymbol{F}_{l+j}\}_{j=1}^{N}$ consisting of $N$ frames that depict the anticipated future visual evolution by executing $\mathcal{A}_{l+1:l+K}$. In practice, we do not predict raw frames directly; instead, we predict their latent representations, as detailed in Sec. 4.2.

After executing $\mathcal{A}_{l+1:l+K}$, we obtain new observations, update $\mathcal{O}_{l}$ using a sliding window, and repeat the process until task completion. This rolling-horizon setup reduces difficulty in long-horizon prediction to a progressive sequence of short *video-continuation* problems.

Joint Video–Action Modeling. Formally, DriveVA jointly models future video imaginations and action chunks conditioned on $\mathbf{C}_{l}:=(\mathcal{O}_{l},\mathcal{T},\mathbf{q}_{l})$. This formulation can be viewed as unifying video continuation and IDM-style \[du2023learning, zhou2024robodreamer\] action grounding within a single end-to-end model, where actions are predicted to be consistent with the imagined future. Instead of training two separate models \[pai2025mimic, lingbot-va2026, xia2025drivelaw\] (a video prediction model and an inverse dynamics model) for the decomposed objective, we optimize a single model end-to-end with this joint objective. This design encourages tighter video–action alignment through deep cross-modal integration (Fig. 5 and Fig. 4). Moreover, since pretrained video models already provide strong video-prediction priors from large web-scale data, DriveVA focuses on adapting these priors to driving-domain video continuation and learning action grounding from predicted visual futures. We further hypothesize that this improves generalization power over conventional VLA training from VLMs, because our formulation explicitly learns temporal dynamics from video frames, which are both used as conditional inputs and prediction targets.

### 4.2 Data Preprocessing

Text Instruction Encoding. We use a frozen text encoder from Wan2.2-TI2V-5B \[wan2025\] to encode the language instruction $\mathcal{T}$ (including the high-level command) into a fixed-length token sequence $\mathbf{T}\in\mathbb{R}^{L_{T}\times d}$ (Fig. 2). These encoded text tokens are injected into the backbone through cross-attention mechanism, instead of being concatenated with the visual/action stream, keeping the spatiotemporal token sequence compact and decoupling text length for higher control flexibility.

Video Causal VAE with Wan2.2-TI2V-5B. We adopt the 3D-causal VAE encoder from Wan2.2-TI2V-5B as the video encoder. Given a video clip $\mathcal{F}=\{\mathbf{F}_{j}\}_{j=1}^{N}$, the encoder produces a temporally downsampled latent sequence:

$$
\mathcal{V}=\{\mathbf{V}_{j}\}_{j=1}^{n},\qquad\mathbf{V}_{j}\in\mathbb{R}^{h\times w\times c},
$$

where $n$ is the latent sequence length after temporal downsampling. In the original WAN’s design, causality ensures that the first latent feature $\mathbf{V}_{1}$ depends only on the first frame observation $\mathbf{F}_{1}$, so a single observed frame can be encoded as a valid conditional latent at inference time.

To guarantee long-duration consistency, we further extend this single-frame conditioning to a *video-continuation* setting by conditioning on a history observation buffer rather than only the current frame. Specifically, at current timestep $l$, we encode the observation buffer $\mathcal{O}_{l}=\{\mathbf{F}_{l-m+1},\ldots,\mathbf{F}_{l}\}$ into a sequence of history latents: $\mathcal{V}^{\mathrm{his}}_{l}=\{\mathbf{V}_{l-m+1},\ldots,\mathbf{V}_{l}\}.$ Thus, the first $m$ -frame conditioning latents are all derived from historical observations and provide long-range visual priors for continuation. During training, we encode the full clip to obtain both history latents $\mathcal{V}^{\mathrm{his}}_{l}$ and future latents $\mathcal{V}^{\mathrm{fut}}_{l}$; during inference, we encode only $\mathcal{O}_{l}$ and generate future latents conditioned on the encoded history latents $\mathcal{V}^{\mathrm{his}}_{l}$.

### 4.3 Consistent Video-Action Generation

At each current timestep $l$, DriveVA jointly predicts future video latents and action tokens conditioned on (i) history latents $\mathcal{V}^{\mathrm{his}}_{l}$, (ii) the current ego state $\mathbf{q}_{l}$, and (iii) text/command tokens $\mathbf{T}$, as in Fig. 2. This design matches training and inference: both use a fixed-length history buffer and predict a short continuation window, which can be chained to progressively generate long-horizon rollout.

Input Tokenization. We raster-flatten each visual latent $\mathbf{V}_{t}$ in $\mathcal{V}^{\mathrm{his}}$ and project it into the model dimension:

$$
\mathbf{V}^{\prime}_{t}=\mathrm{Proj}\!\left(\mathrm{Flatten}(\mathbf{V}_{t})\right)\in\mathbb{R}^{L_{V}\times d},\qquad L_{V}=h\cdot w.
$$

The current ego state $\mathbf{q}_{l}$ is embedded into $L_{S}$ state tokens $\mathbf{S}_{l}\in\mathbb{R}^{L_{S}\times d}$ using an MLP. Each action $\mathbf{a}_{l+i}\in\mathbb{R}^{3}$ is also embedded into one token via an MLP, yielding the action-token sequence $\mathbf{A}_{l+1:l+K}\in\mathbb{R}^{K\times d}$.

Fixed Condition and Generative Targets. We split the model input into a *noise-free condition block* and a *generative target block*:

$$
\displaystyle\mathbf{X}_{\mathrm{cond}}^{(l)}
$$
 
$$
\displaystyle=[\,\mathbf{S}_{l},\ \mathbf{V}^{\prime}_{l-m+1},\ldots,\mathbf{V}^{\prime}_{l}\,]\in\mathbb{R}^{L_{\mathrm{cond}}\times d},
$$
$$
\displaystyle\mathbf{Y}_{0}^{(l)}
$$
 
$$
\displaystyle=[\,\mathbf{V}^{\prime}_{l+1},\ldots,\mathbf{V}^{\prime}_{l+n_{\mathrm{pred}}},\ \mathbf{A}_{l+1:l+K}\,]\in\mathbb{R}^{L_{\mathrm{tgt}}\times d}.
$$

Here, $n_{\mathrm{pred}}$ is the number of future latent steps corresponding to the predicted future clip (after temporal downsampling). The condition block $\mathbf{X}_{\mathrm{cond}}^{(l)}$ is kept fixed at both training and inference. Given the conditional tokens $\mathbf{X}_{\mathrm{cond}}^{(l)}$ and text tokens $\mathbf{T}$, a Diffusion Transformer (DiT) decoder predicts the conditional velocity field for the generative targets:

$$
\hat{\mathbf{v}}_{\theta}^{(l,s)}=f_{\theta}\!\left([\mathbf{X}_{\mathrm{cond}}^{(l)},\mathbf{Y}^{(l,s)}],\,s\mid\mathbf{T}\right),
$$

where $\mathbf{Y}^{(l,s)}$ is the noisy interpolation of the clean targets $\mathbf{Y}_{0}^{(l)}$ at flow time $s$, and $f_{\theta}$ is the DiT decoder parameterized by $\theta$.

### 4.4 Flow Matching Objective

Following the flow matching formulation in Sec. 3, we instantiate a *conditional* flow over the generative target block $\mathbf{Y}_{0}^{(l)}$, conditioned on the fixed context block $\mathbf{X}_{\mathrm{cond}}^{(l)}$ and text tokens $\mathbf{T}$.

Flow-Matching Generative Loss. At timestep $l$, we denote the clean target tokens as $\mathbf{Y}_{0}^{(l)}$. We sample $s\sim\mathcal{U}(0,1)$ and $\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$, and construct the linear interpolation $\mathbf{Y}^{(l,s)}=(1-s)\boldsymbol{\epsilon}+s\mathbf{Y}_{0}^{(l)},$ whose target velocity is $\dot{\mathbf{Y}}^{(l,s)}=\mathbf{Y}_{0}^{(l)}-\boldsymbol{\epsilon}.$ We optimize the standard flow-matching regression loss:

$$
\mathcal{L}_{\mathrm{FM}}=\mathbb{E}_{l,s,\mathbf{Y}_{0}^{(l)},\boldsymbol{\epsilon}}\left[\left\|\hat{\mathbf{v}}_{\theta}^{(l,s)}-\dot{\mathbf{Y}}^{(l,s)}\right\|_{2}^{2}\right].
$$

## 5 Experiments

### 5.1 Datasets

NAVSIM v1. We use the NAVSIM v1 benchmark \[dauner2024navsim\] (built on OpenScene \[contributors2023openscene\]) as our main closed-loop evaluation for safety-critical driving. It reports NC, DAC, TTC, Comfort (C.), and Ego Progress (EP), aggregated as $\mathrm{PDMS}=\mathrm{NC}\times\mathrm{DAC}\times\frac{5\mathrm{EP}+5\mathrm{TTC}+2\mathrm{C.}}{12}$.

nuScenes. For cross-dataset zero-shot evaluation, we evaluate on the nuScenes validation split (150 scenes) from the 1,000-scene nuScenes dataset \[caesar2020nuscenes\], and report Displacement Error (DE) and Collision Rate (CR) following prior works \[hu2023planning, jiang2023vad\].

Bench2Drive. Bench2Drive \[jia2024bench2drive\] is a CARLA v2 closed-loop benchmark \[dosovitskiy2017carla\] with diverse interactive scenarios and evaluation routes. We evaluate: (1) From real to simulation cross domain zero-shot transfer by testing a NAVSIM-trained model directly on the Bench2Drive validation split; (2) *sim-enhanced* training by mixing NAVSIM and Bench2Drive data, then evaluating on NAVSIM. Note that transferring policies across real-world logs and simulation is challenging due to the well-known *reality gap* in appearance, dynamics, and agent behaviors \[hu2023simulation\].

![[x3 22.png|Refer to caption]]

Table 1: Performance comparison on NAVSIM Navtest using closed-loop metrics. Methods are grouped by whether they employ an explicit world model: Traditional End-to-End Methods and World Model Methods.
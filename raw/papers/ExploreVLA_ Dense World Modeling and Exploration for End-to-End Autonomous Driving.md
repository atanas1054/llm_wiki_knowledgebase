---
title: "ExploreVLA: Dense World Modeling and Exploration for End-to-End Autonomous Driving"
source: "https://arxiv.org/html/2604.02714v1"
author:
published:
created: 2026-04-23
description:
tags:
  - "clippings"
---
<sup>1</sup> <sup>2</sup>

Zihao Sheng    Xin Ye    Jingru Luo    Sikai Chen    Liu Ren

###### Abstract

End-to-end autonomous driving models based on Vision-Language-Action (VLA) architectures have shown promising results by learning driving policies through behavior cloning on expert demonstrations. However, imitation learning inherently limits the model to replicating observed behaviors without exploring diverse driving strategies, leaving it brittle in novel or out-of-distribution scenarios. Reinforcement learning (RL) offers a natural remedy by enabling policy exploration beyond the expert distribution. Yet VLA models, typically trained on offline datasets, lack directly observable state transitions, necessitating a learned world model to anticipate action consequences. In this work, we propose a unified understanding-and-generation framework that leverages world modeling to simultaneously enable meaningful exploration and provide dense supervision. Specifically, we augment trajectory prediction with future RGB and depth image generation as dense world modeling objectives, requiring the model to learn fine-grained visual and geometric representations that substantially enrich the planning backbone. Beyond serving as a supervisory signal, the world model further acts as a source of intrinsic reward for policy exploration: its image prediction uncertainty naturally measures a trajectory’s novelty relative to the training distribution, where high uncertainty indicates out-of-distribution scenarios that, if safe, represent valuable learning opportunities. We incorporate this exploration signal into a safety-gated reward and optimize the policy via Group Relative Policy Optimization (GRPO). Experiments on the NAVSIM and nuScenes benchmarks demonstrate the effectiveness of our approach, achieving a state-of-the-art PDMS score of 93.7 and an EPDMS of 88.8 on NAVSIM. The code and demo will be publicly available at [https://zihaosheng.github.io/ExploreVLA/](https://zihaosheng.github.io/ExploreVLA/).

## 1 Introduction

End-to-end autonomous driving has advanced rapidly with the emergence of Vision-Language-Action (VLA) architectures \[hwang2024emma, jiang2025diffvla, zhou2025opendrivevla, wang2025alpamayo\], which unify perception, reasoning, and planning within a single model. By leveraging the representational power of large vision-language models, these models have demonstrated promising capabilities in translating raw sensor observations into driving actions. Yet, the dominant training paradigm for such models (behavior cloning on expert demonstrations or supervised fine-tuning) introduces a fundamental bottleneck: the learned policy can only replicate the behaviors it has observed, without the ability to discover alternative strategies that may be equally or more effective. This limitation manifests as distributional brittleness: when confronted with scenarios that deviate from the expert distribution, the policy lacks the exploratory experience needed to generalize \[ross2011reduction, codevilla2019exploring\].

![[x1 25.png|Refer to caption]]

Figure 1: Comparison of training paradigms for VLA-based autonomous driving. (a) Imitation learning directly clones expert demonstrations without exploration. (b) Previous reinforcement learning enables policy exploration, but cannot distinguish expert imitation from genuine out-of-distribution discovery and relies on sparse supervision. (c) Our approach augments RL with dense world modeling supervision via future image generation, while leveraging image prediction uncertainty as a novelty measure to identify and prioritize valuable exploratory strategies.

Reinforcement learning (RL) offers a principled mechanism to overcome this limitation by allowing the agent to explore beyond the boundaries of expert data and optimize its policy through trial-and-error interaction \[guo2025improving\]. Recent advances in RL post-training, exemplified by Group Relative Policy Optimization (GRPO) \[shao2024deepseekmath\], have demonstrated that sampling diverse candidate outputs and performing relative ranking can effectively improve policy quality even atop strong pretrained models. However, applying RL to autonomous driving poses challenges distinct from other domains. In language tasks, state transitions are fully determined by the model’s own output, and outcomes are immediately observable. In robotics, high-fidelity simulators enable safe trial-and-error. Autonomous driving enjoys neither advantage: the consequences of a planned trajectory depend on complex scene dynamics that no existing simulator faithfully captures. These challenges motivate the need for a world model that can internalize environment dynamics and anticipate action consequences from data alone. Yet a further challenge remains: standard task-level rewards such as Predictive Driver Model Score (PDMS) \[dauner2024navsim\] evaluate trajectory quality but do not distinguish between policies that merely replicate expert behavior and those that have genuinely discovered novel strategies. An exploration signal orthogonal to task performance is therefore essential to unlock the full potential of RL post-training.

A second limitation of existing VLA driving models is the sparsity of their supervisory signals. Most approaches rely on textual descriptions and trajectory waypoints as training targets \[zhou2025autovla, zhou2025opendrivevla\], which, while informative for high-level decision-making, fail to capture the rich spatial geometry and fine-grained appearance of driving scenes. This supervisory deficit constrains the model’s ability to build comprehensive internal representations, particularly for aspects of the environment (such as road topology, object extent, and depth ordering) that are critical for safe planning but are not explicitly encoded in sparse action labels \[li2025drivevla\].

In this work, we propose a unified understanding-and-generation framework that addresses both limitations through a single mechanism: dense world modeling and exploration (Fig.˜1). Specifically, we augment trajectory prediction with future RGB and depth image generation as auxiliary objectives. On the supervision side, these generation tasks require the model to predict fine-grained visual appearance and metric geometry of future scenes, providing dense gradient signals that substantially enrich the planning backbone’s visual and geometric representations. On the exploration side, we leverage a key insight: *the world model’s image prediction uncertainty naturally measures a trajectory’s novelty relative to the training distribution*. Since the world model is trained exclusively on expert demonstrations, it produces low-uncertainty predictions for trajectories within the expert distribution but exhibits high uncertainty on out-of-distribution (OOD) trajectories. Crucially, this uncertainty reflects distance to the *entire* training distribution, rather than deviation from a single ground-truth trajectory, which provides a more reliable novelty signal than trajectory distance alone. We incorporate this signal into a safety-gated exploration reward: trajectories that achieve high PDMS scores while exhibiting high prediction uncertainty are identified as valuable discoveries of new successful strategies and receive an exploration bonus. This composite reward is optimized via GRPO, enabling the policy to expand its behavioral repertoire while maintaining driving safety.

Our contributions can be summarized as follows:

- We introduce a novel exploration mechanism for RL post-training that uses the world model’s image prediction uncertainty as an intrinsic novelty measure, coupled with a safety-gated reward to encourage beneficial out-of-distribution exploration.
- We propose a unified VLA framework that jointly predicts future trajectories, RGB images, and depth images, leveraging dense world modeling to provide rich visual and geometric supervision for the planning backbone.
- We demonstrate state-of-the-art performance on the NAVSIM benchmark with a PDMS of 93.7 and an EPDMS of 88.8, and validate the generalizability of our approach on the nuScenes dataset.

## 2 Related Work

### 2.1 World Models for Autonomous Driving

In the context of autonomous driving, world models offer the ability to predict future states of the driving scene, enabling planning, data augmentation, and closed-loop evaluation without costly real-world interactions \[guan2024world, feng2025survey, ding2025understanding, yan2025ad\]. A significant line of work focuses on video prediction-based world models \[gao2024vista, hassan2025gem, yang2025resim\]. For example, GAIA-1 \[hu2023gaia\] leverages a generative model conditioned on video, text, and action inputs to produce realistic driving videos. DriveDreamer \[wang2024drivedreamer\] generates future driving frames conditioned on HDMaps and 3D bounding boxes, bridging the gap between generation fidelity and controllability, while GenAD \[yang2024generalized\] proposes an action-conditioned video generation framework that supports long-horizon future prediction for planning. Another prominent direction explores world models within structured geometric spaces. UniWorld \[min2023uniworld\] proposes a unified framework for 4D occupancy forecasting, and OccWorld \[zheng2024occworld\] extends this idea by predicting 3D occupancy and ego-motion jointly, providing a compact yet expressive world representation for downstream planning. UniScene \[li2025uniscene\] further scales this paradigm by jointly generating consistent future semantic occupancy, LiDAR and multi-view images. Beyond generation and prediction, several works integrate world models into the planning pipeline. For instance, WoTE \[li2025end\] incorporates a BEV world model into end-to-end driving to predict future states for online trajectory evaluation and selection. OmniNWM \[li2025omninwm\] employs a learned navigation world model to generate imagined futures and derive planning rewards from predicted scene dynamics. World4Drive \[zheng2025world4drive\] further introduces an intention-aware latent world model that predicts future latent states under multiple driving intentions and selects trajectories via a learned world model selector. In our work, we leverage the world model not only for future state prediction but also as a source of intrinsic reward signals that encourage the policy to explore diverse and informative driving behaviors, thereby improving generalization beyond the coverage of the training data.

### 2.2 VLA Models for Autonomous Driving

The integration of vision, language, and action within a unified framework has emerged as a promising paradigm for autonomous driving \[jiang2025survey, li2025recogdrive\]. Early efforts, such as DriveGPT-4 \[xu2024drivegpt4\], use frozen VLMs to narrate driving scenes but do not directly output control signals, serving only as passive explainers. Subsequent modular VLA approaches began embedding language into the planning loop. For example, OpenDriveVLA \[zhou2025opendrivevla\] fuses multimodal sensor inputs with textual route instructions to generate interpretable waypoints, while RAG-Driver \[yuan2024rag\] introduces retrieval-augmented planning for long-tail scenarios. The field then advances toward unified end-to-end architectures. EMMA \[hwang2024emma\] jointly performs detection and planning within a single VLM, and DiffVLA \[jiang2025diffvla\] combines diffusion-based trajectory sampling with language-conditioned embeddings. Most recently, reasoning-augmented VLA models have pushed the frontier further. ORION \[fu2025orion\] incorporates a transformer memory module for long-horizon reasoning, and AutoVLA \[zhou2025autovla\] fuses CoT reasoning and trajectory planning in a single autoregressive transformer. Alpamayo-R1 \[wang2025alpamayo\] introduces causally grounded Chain-of-Causation reasoning tightly integrated with trajectory prediction, enhancing reasoning-action consistency and long-tail safety performance. Despite this rapid progress, most existing VLA models rely on textual descriptions and action trajectories as the primary supervisory signal, which are inherently sparse. This supervisory sparsity limits the model’s ability to learn comprehensive scene representations. In contrast, our work leverages RGB and depth images as auxiliary dense supervisory signals to encourage the model to capture richer visual and geometric cues, thereby yielding more accurate and robust trajectory planning.

### 2.3 Unified Understanding and Generation Models

In foundation model research, the long-standing separation between understanding and generation has motivated a growing effort to unify both within a single architecture for greater scalability and cross-task synergy \[xie2024show, chen2025janus, wang2026multimodal\]. In autonomous driving, this paradigm has gained significant traction as researchers seek to bridge the gap between world modeling and end-to-end planning. For example, FutureSightDrive \[zeng2025futuresightdrive\] proposes a spatio-temporal visual Chain-of-Thought framework where a VLA model first generates future frames, including lane lines, 3D bounding boxes, and complete future images, as visual intermediate reasoning steps, then predicts trajectories conditioned on these imagined futures. Policy World Model (PWM) \[zhao2025pwm\] pre-trains on large-scale action-free video generation to learn world dynamics, then fine-tunes with a collaborative formulation where trajectory planning is explicitly conditioned on forecasted future states. DriveVLA-W0 \[li2025drivevla\] introduces world modeling objectives to unlock data scaling laws for end-to-end driving. UniDrive-WM \[xiong2026unidrive\] unifies scene understanding, trajectory planning, and trajectory-conditioned future image generation within a single VLM. Epona \[zhang2025epona\] combines autoregressive causal modeling with diffusion-based generation through decoupled spatiotemporal factorization, enabling both high-fidelity video synthesis and trajectory planning. Together, these works demonstrate that jointly modeling future scene generation and action prediction yields more informed and anticipatory planning. Our work follows this unified paradigm by leveraging RGB and depth image generation as dense supervisory signals alongside trajectory prediction, while further employing the world model to provide intrinsic reward signals that encourage exploratory driving behaviors and improve generalization beyond the training distribution.

## 3 Methodology

### 3.1 Overview

We present a unified understanding-and-generation framework for end-to-end autonomous driving that addresses two key limitations of existing VLA models: (1) the lack of exploration beyond expert demonstrations and (2) the reliance on sparse supervisory signals. Our framework consists of three components. First, we build upon a unified VLM backbone that jointly supports autoregressive text modeling and discrete image generation within a single architecture (Sec.˜3.3). Second, we introduce future RGB and depth image generation as dense world modeling objectives that provide token-level supervision alongside trajectory prediction, encouraging the model to learn richer scene representations (Sec.˜3.4). Third, we leverage the world model’s prediction uncertainty as an intrinsic reward signal to guide policy exploration via Group Relative Policy Optimization (GRPO), enabling the model to discover diverse driving strategies beyond mere imitation (Sec.˜3.5). An overview of our framework is illustrated in Fig.˜2.

![[x2 23.png|Refer to caption]]

Figure 2: Model architecture and training paradigm of ExploreVLA. The model takes task instructions, multi-frame images, and ego status as input, and jointly predicts future trajectories and future images. Training proceeds in two stages: (1) imitation learning, consisting of pre-training on image generation and supervised fine-tuning on both actions and images, and (2) reinforcement learning, where GRPO optimizes the policy using a composite reward combining PDMS and image-based exploration bonus.

### 3.2 Problem Formulation

We formulate autonomous driving as a unified understanding-and-generation task, where a single model jointly predicts future trajectories and generates dense visual representations conditioned on the current driving context.

#### 3.2.1 Input.

At each timestep $t$, the model receives three inputs: (1) the current and $T$ past front-view camera images $\{\mathbf{I}_{t-T},\cdots,\mathbf{I}_{t}\}$ with $\mathbf{I}\in\mathbb{R}^{H\times W\times 3}$, (2) a natural language command $\mathbf{c}$, and (3) the ego-vehicle status $\mathbf{s}_{t}$.

#### 3.2.2 Output.

The model produces three types of outputs: (1) predicted future waypoints $\boldsymbol{\tau}=\{\hat{p}_{i}\}_{i=1}^{N_{\tau}}$, where $N_{\tau}$ denotes the planning horizon, (2) predicted future RGB frames $\{\hat{\mathbf{I}}_{t+1},\cdots,\hat{\mathbf{I}}_{t+F}\}$ with $\hat{\mathbf{I}}\in\mathbb{R}^{H^{\prime}\times W^{\prime}\times 3}$ capturing the anticipated visual appearance of the driving scene, and (3) predicted future depth maps $\{\hat{\mathbf{D}}_{t+1},\cdots,\hat{\mathbf{D}}_{t+F}\}$ with $\hat{\mathbf{D}}\in\mathbb{R}^{H^{\prime}\times W^{\prime}}$ encoding the geometric structure of the future scene. Here $F$ denotes the number of future frames to generate.

### 3.3 Unified Understanding and Generation Architecture

#### 3.3.1 Tokenization.

We maintain a unified vocabulary of discrete tokens spanning text, images, and special task indicators. For text tokenization, we adopt the same tokenizer from the pre-trained LLM backbone, such that the language command $\mathbf{c}$ is tokenized into $L$ text tokens $\mathbf{v}=\{v_{1},v_{2},\cdots,v_{L}\}$. For image tokenization, we employ a pre-trained MAGVIT-v2 \[yu2023language\] quantizer with a lookup-free codebook of size $K=8{,}192$. Each image is encoded into discrete tokens by partitioning it into non-overlapping patches of size $16\times 16$. Each of the $(T+1)$ input frames is independently tokenized into $M$ image tokens, yielding the input image token sequence $\mathbf{u}=\{u_{1},u_{2},\cdots,u_{(T+1)\times M}\}$. For ego status encoding, the $\mathbf{s}_{t}$ is projected into the transformer’s embedding space via a learnable MLP, producing ego status embeddings $\mathbf{e}_{s}$ that are concatenated with the other input tokens.

#### 3.3.2 Causal and Full Attention Mechanism.

We adopt the omni-attention mechanism from Show-o \[xie2024show\], which adaptively combines causal and full attention depending on the token type. Text tokens $\mathbf{v}$ and ego status embeddings $\mathbf{e}_{s}$ are processed via causal attention, where each token attends only to its preceding tokens. Image tokens $\mathbf{u}$ are processed via full attention, which allows comprehensive interaction among all spatial positions and ensures that the generation process is fully conditioned on the driving context.

#### 3.3.3 Trajectory Prediction Head.

In addition to the generation outputs, we extract the hidden states from the transformer at designated positions and feed them through a lightweight MLP head to predict future waypoints:

$$
\boldsymbol{\tau}=\text{MLP}(\mathbf{h}),
$$

where $\mathbf{h}$ denotes the hidden representation. This design decouples trajectory prediction from the discrete token vocabulary, allowing the model to produce continuous-valued waypoints while sharing the same contextualized representations learned through the joint understanding-and-generation objective.

### 3.4 Dense Supervisory Signals via World Modeling

A central limitation of existing VLA models for autonomous driving is their reliance on sparse supervisory signals. Textual descriptions provide only high-level semantic intent (*e.g*., “turn left”), while trajectory waypoints encode a thin, low-dimensional slice of the rich information embedded in driving scenes. As a result, the vast majority of scene structure, such as patial layout and depth ordering, remains unsupervised, which limits the model’s ability to learn comprehensive representations of the driving environment.

We address this by formulating future scene generation as an auxiliary world modeling objective that provides dense supervision alongside trajectory prediction. Our model is trained to generate $F$ future frames of both RGB images and depth maps, effectively requiring it to “imagine” the future visual and geometric state of the world. This dense generation objective forces the model to internalize fine-grained knowledge about scene dynamics.

#### 3.4.1 RGB Generation as Visual Supervision.

The future RGB generation objective requires the model to predict the visual appearance of upcoming driving scenes, providing dense supervision over texture, color, object identity, and scene semantics. Each future RGB frame is tokenized via the shared MAGVIT-v2 quantizer, and the model learns to reconstruct randomly masked tokens through the mask token prediction objective. Formally, the RGB generation loss over all $F$ future frames is:

$$
\mathcal{L}_{\text{rgb}}=-\sum_{f=1}^{F}\sum_{j}\log p_{\theta}\big(u^{\text{rgb}}_{f,j}\mid\mathbf{u}^{\text{rgb}}_{f,*},\ \mathbf{v},\ \mathbf{e}_{s},\ \mathbf{u}\big),
$$

where $u^{\text{rgb}}_{f,j}$ is the $j$ -th masked token of the $f$ -th future RGB frame, and $\mathbf{u}^{\text{rgb}}_{f,*}$ denotes the corresponding masked sequence. By reconstructing future visual tokens, the model learns to capture how scene appearance evolves under the current driving context.

#### 3.4.2 Depth Generation as Geometric Supervision.

Complementary to RGB, the depth generation objective supervises the model on the 3D geometric structure of future scenes. Depth maps encode object distances, spatial layout, and surface orientation, which is critical for safe planning but entirely absent from text or trajectory supervision. The depth generation loss is defined analogously:

$$
\mathcal{L}_{\text{depth}}=-\sum_{f=1}^{F}\sum_{j}\log p_{\theta}\big(u^{\text{dep}}_{f,j}\mid\mathbf{u}^{\text{dep}}_{f,*},\ \mathbf{u}^{\text{rgb}}_{f,*},\ \mathbf{v},\ \mathbf{e}_{s},\ \mathbf{u}\big),
$$

where $u^{\text{dep}}_{f,j}$ is the $j$ -th masked token of the $f$ -th future depth map. The overall mask token prediction (MTP) loss decomposes as:

$$
\mathcal{L}_{\text{MTP}}=\mathcal{L}_{\text{rgb}}+\mathcal{L}_{\text{depth}}.
$$

### 3.5 Intrinsic Reward from World Model for Exploration

Models trained purely via behavior cloning tend to narrowly replicate the expert’s actions and struggle to generalize when the test-time distribution deviates from the training data. In the second stage of training, we leverage the world model learned in the first stage as a source of intrinsic reward signals that encourage the policy to explore novel yet safe driving behaviors beyond mere imitation. The core idea is that the world model’s image prediction uncertainty provides a natural measure of trajectory novelty relative to the entire training distribution: high uncertainty indicates that the model has rarely encountered the visual consequences of a given action, signaling an out-of-distribution trajectory that, if safe, represents a valuable learning opportunity.

#### 3.5.1 Uncertainty-Based Exploration Bonus.

Given a candidate trajectory $\boldsymbol{\tau}_{i}$ sampled from the current VLA policy, we condition the world model on $\boldsymbol{\tau}_{i}$ and perform future image generation. For each predicted discrete image token, the model outputs a probability distribution over the MAGVIT-v2 codebook. We quantify the prediction uncertainty using the entropy of these token-level distributions, averaged across all generated future RGB and depth tokens:

$$
\mathcal{H}(\boldsymbol{\tau}_{i})=-\frac{1}{|\mathcal{M}|}\sum_{j\in\mathcal{M}}p_{j}\log p_{j},
$$

where $\mathcal{M}$ denotes the set of generated image token positions across all future RGB and depth frames, and $p_{j}$ is the predicted probability of image token $j$. A high entropy indicates that the world model is uncertain about the visual consequences of trajectory $\boldsymbol{\tau}_{i}$, meaning the trajectory leads to scenarios underrepresented in the training distribution. Conversely, low entropy indicates an in-distribution trajectory whose consequences the model has already learned to predict. We define the exploration bonus as the normalized entropy:

$$
b_{i}=f(\mathcal{H}(\boldsymbol{\tau}_{i})),
$$

where $f(\cdot)$ bounds $b_{i}\in[0,1]$.

#### 3.5.2 Safety-Gated Reward.

High uncertainty alone does not guarantee that exploration is beneficial, since a trajectory leading to a collision is novel but not useful. We therefore gate the exploration bonus using the PDMS \[dauner2024navsim\], which evaluates trajectory quality on a $[0,1]$ scale based on collision avoidance, comfort, and progress. The intrinsic reward for trajectory $\boldsymbol{\tau}_{i}$ is:

$$
R_{i}=\begin{cases}\text{PDMS}_{i}+\lambda\cdot b_{i},&\text{if }\text{PDMS}_{i}>\delta,\\
\text{PDMS}_{i},&\text{otherwise},\end{cases}
$$

where $\delta$ is a safety threshold and $\lambda$ controls the exploration strength. The gating mechanism ensures that only safe trajectories ($\text{PDMS}_{i}>\delta$) receive the exploration bonus. Unsafe or failed explorations are scored purely by their PDMS without additional encouragement. This design prioritizes trajectories that are simultaneously novel (high world model uncertainty) and good (high PDMS). Such trajectories represent out-of-distribution behaviors that are critical for improving generalization beyond expert demonstrations.

#### 3.5.3 Policy Optimization via GRPO.

We optimize the policy using GRPO \[shao2024deepseekmath\]. At each iteration, we sample a group of $G$ candidate trajectories $\{\boldsymbol{\tau}_{1},\cdots,\boldsymbol{\tau}_{G}\}$ from the current policy. Each trajectory and corresponding predicted image tokens are scored using the reward in Eq.˜7, and rewards are normalized within the group to compute relative advantages:

$$
\hat{A}_{i}=\frac{R_{i}-\text{mean}(\{R_{1},\cdots,R_{G}\})}{\text{std}(\{R_{1},\cdots,R_{G}\})}.
$$

The policy is updated to increase the likelihood of trajectories with positive advantages and suppress those with negative advantages. This group-relative formulation naturally steers the policy toward trajectories that are both good and novel within each sampled group.

### 3.6 Training Strategy

As shown in Fig.˜2, our training proceeds in two stages. The first stage consists of two phases: pre-training and supervised fine-tuning. During pre-training, ground-truth future actions are provided as input, and the model is trained solely with the image generation objective. This allows the model to adapt its visual generation capabilities to driving scenes. In the supervised fine-tuning phase, the model is trained to jointly predict both actions and future images. In the second stage, we further refine the policy using an intrinsic reward derived from the world model to encourage exploration.

## 4 Experiments

### 4.1 Experimental Setup

#### 4.1.1 Datasets.

We evaluate our method on two widely used autonomous driving benchmarks: NAVSIM and nuScenes.

NAVSIM \[dauner2024navsim, cao2025pseudo\] is a recently proposed non-reactive simulation benchmark built upon the OpenScene dataset. NAVSIM v1 evaluates planning quality using PDMS \[dauner2024navsim\], and NAVSIM v2 adopts the Extended PDMS (EPDMS) \[cao2025pseudo\]. Both metrics aggregate factors such as progress, time-to-collision, and comfort, achieving stronger correlation with closed-loop evaluations. We train on the navtrain split and report results on the navtest split. In NAVSIM, the model predicts future waypoints in the form of $(x,y,\theta)$ over a 4-second planning horizon.

nuScenes \[caesar2020nuscenes\] consists of 1,000 driving scenes and has become a standard benchmark for open-loop planning evaluation. Following prior works \[hu2022st, hu2023planning, jiang2023vad\], we adopt the standard train/val split and evaluate using the L2 displacement error and collision rate between predicted and ground-truth trajectories. On nuScenes, the model predicts future waypoints in the form of $(x,y)$. The evaluation results on nuScenes are presented in the supplementary material.

#### 4.1.2 Implementation Details.

Our model is built upon Show-o \[xie2024show\] as the backbone architecture. In the first training stage, we first pre-train the model for 10 epochs by conditioning on ground-truth future actions and supervising only the image generation. We then perform supervised fine-tuning for 15 epochs, where the model jointly predicts future trajectories and generates future RGB and depth images. In the second stage, we apply GRPO-based post-training with LoRA \[hu2022lora\] for 5 epochs to further refine the policy using the exploration-aware reward described in Sec.˜3.5. All experiments are conducted on 4 $\times$ H200 GPUs. We obtain depth maps from a pre-trained monocular depth estimation model \[yin2023metric3d\]. Detailed hyperparameters are provided in the supplementary material.

Table 1: Comparison on NAVSIM v1 with closed-loop metrics. The best performance is marked in bold, and the second best is underlined. Abbreviations: no at-fault collision (NC), drivable area compliance (DAC), ego progress (EP), time to collision (TTC), comfort (Comf.). SC: single-view camera; MC: multi-view camera; L: LiDAR; $\dagger$: best-of-N (N=6) strategy \[zhou2025autovla\].

| Model | Input | NC $\uparrow$ | DAC $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | Comf. $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ego Status MLP | \- | 93.1 | 78.3 | 63.2 | 84.0 | 99.9 | 66.4 |
| TransFuser \[chitta2022transfuser\] | \- | 97.8 | 92.6 | 78.9 | 92.9 | 99.9 | 83.9 |
| DRAMA \[yuan2024drama\] | MC+L | 98.2 | 95.2 | 81.3 | 94.2 | 100.0 | 86.9 |
| Hydra-MDP \[li2024hydra\] | MC+L | 99.1 | 98.3 | 85.2 | 96.6 | 100.0 | 91.3 |
| Centaur \[sima2025centaur\] | MC+L | 99.2 | 98.7 | 86.0 | 97.2 | 99.9 | 92.1 |
| DriveSuprim \[yao2025drivesuprim\] | MC+L | 98.6 | 98.6 | 91.3 | 95.5 | 100.0 | 93.5 |
| DrivingGPT \[chen2025drivinggpt\] | SC | 98.9 | 90.7 | 79.9 | 94.9 | 95.6 | 82.4 |
| FSDrive \[zeng2025futuresightdrive\] | SC | 98.2 | 93.8 | 80.1 | 93.3 | 99.9 | 85.1 |
| PWM \[zhao2025pwm\] | SC | 98.9 | 95.8 | 81.5 | 95.9 | 100.0 | 88.1 |
| AutoVLA \[zhou2025autovla\] | MC | 98.4 | 95.6 | 81.9 | 98.0 | 99.9 | 89.1 |
| AutoVLA $\dagger$ \[zhou2025autovla\] | MC | 99.1 | 97.1 | 87.6 | 97.1 | 99.9 | 92.1 |
| DriveVLA-W0 \[li2025drivevla\] | SC | 98.7 | 99.1 | 87.6 | 97.1 | 100.0 | 90.2 |
| DriveVLA-W0 $\dagger$ \[li2025drivevla\] | SC | 99.9 | 97.4 | 88.3 | 97.0 | 99.9 | 93.0 |
| ExploreVLA | SC | 98.8 | 98.4 | 83.5 | 96.5 | 99.9 | 90.4 |
| ExploreVLA $\dagger$ | SC | 99.4 | 98.9 | 88.3 | 98.3 | 99.7 | 93.7 |

Table 2: Comparison on NAVSIM v2 with extended closed-loop metrics. The best performance is marked in bold, and the second best is underlined.

| Model | NC $\uparrow$ | DAC $\uparrow$ | DDC $\uparrow$ | TLC $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | LK $\uparrow$ | HC $\uparrow$ | EC $\uparrow$ | EPDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ego Status | 93.1 | 77.9 | 92.7 | 99.6 | 86.0 | 91.5 | 89.4 | 98.3 | 85.4 | 64.0 |
| TransFuser \[chitta2022transfuser\] | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 |
| Hydra-MDP++ \[li2025hydra\] | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | 81.4 |
| DriveSuprem \[yao2025drivesuprim\] | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 | 83.1 |
| ARTEMIS \[feng2025artemis\] | 98.3 | 95.1 | 98.6 | 99.8 | 81.5 | 97.4 | 96.5 | 98.3 | \- | 83.1 |
| DiffusionDrive \[liao2025diffusiondrive\] | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | 84.5 |
| DiffusionDriveV2 \[zou2025diffusiondrivev2\] | 97.7 | 96.6 | 99.2 | 99.8 | 88.9 | 97.2 | 96.0 | 97.8 | 91.0 | 85.5 |
| DriveVLA-W0 \[li2025drivevla\] | 98.5 | 99.1 | 98.0 | 99.7 | 86.4 | 98.1 | 93.2 | 97.9 | 58.9 | 86.1 |
| ExploreVLA | 98.8 | 96.2 | 99.6 | 99.8 | 87.1 | 98.2 | 97.8 | 98.3 | 86.8 | 88.8 |

### 4.2 Main Results

#### 4.2.1 Results on NAVSIM v1.

Tab.˜1 presents the comparison on the NAVSIM v1 benchmark. Our ExploreVLA achieves the highest PDMS of 93.7 with the best-of-N strategy, outperforming all prior approaches. Notably, ExploreVLA uses only a single-view camera, yet surpasses multi-sensor methods such as DriveSuprim and Centaur. Even without the best-of-N strategy, ExploreVLA attains a PDMS of 90.4, which is competitive with multi-view methods like AutoVLA. Among the sub-metrics, ExploreVLA $\dagger$ achieves the best TTC and the second-best scores on NC, DAC, and EP, demonstrating that our world-model-based exploration reward helps the model internalize diverse and robust driving behaviors beyond what pure imitation learning can provide.

#### 4.2.2 Results on NAVSIM v2.

Tab.˜2 further validates our approach on NAVSIM v2, which introduces extended closed-loop metrics including driving direction compliance (DDC), traffic light compliance (TLC), lane keeping (LK), history comfort (HC), and extended comfort (EC). ExploreVLA achieves the highest EPDMS of 88.8, surpassing the previous best result of 86.1 by DriveVLA-W0 by 2.7 points. Our method obtains the best scores on six out of nine individual metrics, while remaining highly competitive on the rest.

### 4.3 Analysis of Intrinsic Reward Modeling

Fig.˜3 provides an analysis of our intrinsic reward modeling mechanism. The left shows a generally positive correlation between the exploration bonus and L2 error with respect to the ground-truth trajectory: as the sampled trajectory deviates further from the expert action, the world model’s prediction uncertainty tends to increase, resulting in higher exploration bonuses. Additionally, the right shows our image-based exploration bonus correctly captures trajectory novelty in cases where L2 error fails. This example illustrates that L2 error can be misleading as a novelty indicator: a trajectory that closely follows the expert’s direction may incur a large L2 error due to positional shift, while a trajectory that takes a fundamentally different route may have a smaller L2 error.

![[x3 23.png|Refer to caption]]

Figure 3: Analysis of the exploration bonus. Left: the exploration bonus is positively correlated with L2 error to the ground-truth trajectory. Right: our exploration bonus can properly measure the trajectory novelty that L2 error fails.

### 4.4 Ablation Study

#### 4.4.1 Effect of Dense Visual Supervision.

Tab.˜3 examines the contribution of RGB and depth image generation as auxiliary supervision during Stage 1 training. The trajectory-only baseline without any image generation achieves the lowest PDMS. Adding either RGB or depth generation alone yields comparable improvements, which confirms that both modalities provide meaningful dense supervisory signals for learning better scene representations. Combining both RGB and depth generation further improves PDMS to 88.5. This indicates that RGB and depth capture complementary information (visual appearance and geometric structure) and their joint supervision leads to a more comprehensive understanding of driving scenes.

Table 3: Ablation study on dense visual supervision. We evaluate the effect of auxiliary RGB and depth image generation during Stage 1 imitation learning on the NAVSIM v1 navtest split.

| RGB Img. Gen. | Depth Img. Gen. | NC $\uparrow$ | DAC $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | Comf. $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ✗ | ✗ | 98.6 | 94.4 | 80.1 | 94.8 | 100.0 | 86.2 |
| ✓ | ✗ | 98.7 | 96.0 | 81.5 | 95.4 | 99.9 | 87.9 |
| ✗ | ✓ | 98.7 | 95.8 | 81.4 | 95.6 | 99.9 | 87.8 |
| ✓ | ✓ | 98.7 | 96.3 | 82.3 | 95.7 | 99.9 | 88.5 |

Table 4: Ablation study on reward components. We evaluate the contribution of each reward signal during Stage 2 reinforcement learning on the NAVSIM v1 navtest split. The first row (no reward) corresponds to the Stage 1 baseline.

| PDMS Reward | Image Reward | NC $\uparrow$ | DAC $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | Comf. $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ✗ | ✗ | 98.74 | 96.25 | 82.27 | 95.67 | 99.97 | 88.50 |
| ✓ | ✗ | 98.82 | 97.78 | 83.44 | 96.50 | 99.95 | 90.19 |
| ✗ | ✓ | 98.76 | 96.30 | 82.32 | 95.65 | 99.97 | 88.53 |
| ✓ | ✓ | 98.81 | 97.88 | 83.53 | 96.66 | 99.95 | 90.36 |

#### 4.4.2 Effect of Reward Design.

Tab.˜4 isolates the contribution of each reward component during Stage 2 RL. Starting from the Stage 1 model (PDMS 88.50), applying only the PDMS reward brings a substantial improvement to 90.19, demonstrating the effectiveness of GRPO-based post-training. Using the image-based exploration reward alone yields only a marginal gain (88.53), which is expected since the image reward serves as an exploration bonus rather than a direct driving quality signal. Without the safety gate provided by PDMS thresholding, the exploration signal alone cannot effectively guide policy improvement. When combining both rewards, the model achieves the best PDMS of 90.36. This confirms that the image-based exploration reward provides a complementary learning signal that encourages the policy to discover diverse driving strategies beyond what the PDMS reward alone can incentivize.

#### 4.4.3 Qualitative Analysis.

![[x4 21.png|Refer to caption]]

Figure 4: Qualitative comparison of planned trajectories before and after RL post-training. We visualize three challenging driving scenarios in bird’s-eye view. The Stage 1 model exhibits safety-critical failures. After Stage 2 RL post-training, the model produces safer and more compliant trajectories. green: GT, orange: prediction.

Fig.˜4 visualizes planned trajectories before (Stage 1) and after (Stage 2) RL in three representative scenarios. In each case, the Stage 1 model exhibits a safety-critical failure (*i.e*., colliding with a vehicle, passing dangerously close to pedestrians, or running a stop sign) that the Stage 2 model successfully corrects. These results demonstrate that our world-model-based RL not only improves aggregate metrics but also rectifies safety-critical failures that pure imitation learning struggles to resolve from expert demonstrations alone.

## 5 Conclusion

We presented ExploreVLA, a unified framework that addresses the lack of exploration and sparse supervision in VLA-based autonomous driving. By jointly predicting future trajectories, RGB images, and depth maps, our approach provides dense world modeling supervision that enriches scene representations. We further leverage the world model’s prediction uncertainty as an intrinsic novelty measure, combined with a safety-gated reward and GRPO, to guide the policy toward diverse yet safe driving strategies beyond expert imitation. Extensive experiments on the NAVSIM and nuScenes benchmarks validate the effectiveness of our approach, achieving state-of-the-art performance on NAVSIM.

Limitations and Future Work. Our framework currently uses a single front-view camera; extending to multi-view inputs could further broaden spatial coverage and planning robustness. Additionally, exploring complementary generation targets such as Bird’s-Eye View layouts may offer additional supervisory signals for scene structure, potentially further enhancing planning performance.

## References

## Appendix 0.A More Implementation Details

Our model is built upon Show-o \[xie2024show\], which employs Phi-1.5 \[li2023textbooks\] as the LLM backbone and a pre-trained MAGVIT-v2 \[yu2023language\] as the image tokenizer. The input to the model consists of the current front-view image and one historical image captured 0.5 seconds earlier, and the model predicts the future RGB and depth images 0.5 seconds ahead. Input images are resized to $256\times 448$. The ground-truth depth maps used for supervision are generated by Metric3D-ViT-Giant2 \[yin2023metric3d\].

We use AdamW as the optimizer across all training stages. In Stage 1, the learning rate is $3\times 10^{-5}$. In Stage 2, we apply LoRA \[hu2022lora\] with rank 32 and train with a learning rate of $3\times 10^{-6}$. For GRPO, the group size is set to $G=8$, with KL penalty coefficient $\beta=0.01$ and clipping range $\epsilon=0.1$. The safety threshold $\delta$ in Eq. (7) is set to 0.9, and the exploration bonus weight $\lambda$ is set to 0.5. The per-GPU batch size is 8 in Stage 1 and 1 in Stage 2. All experiments are conducted on 4 $\times$ H200 GPUs.

## Appendix 0.B More Experimental Results

### 0.B.1 Evaluation on NAVSIM

#### 0.B.1.1 More Qualitative Results.

We present additional qualitative results on the navtest split in Fig.˜5, covering three representative scenario categories: Going Straight, Turning, and Intersection. Across all scenarios, our method generates planned trajectories that closely align with the road topology and traffic context. In straight-driving cases, the predicted trajectories maintain stable lane-keeping behavior. For turning scenarios, our model produces smooth and well-timed turning maneuvers that conform to the curvature of the road. Notably, in complex intersection scenarios involving multiple lanes and diverse traffic participants, our method still plans reasonable and safe trajectories, demonstrating its ability to handle intricate spatial layouts and dynamic interactions.

![[x5 21.png|Refer to caption]]

Figure 5: Additional qualitative results on the navtest split. We visualize the planned trajectories across three scenario categories: Going Straight, Turning, and Intersection. For each example, we show the front-view camera image and the corresponding BEV representation with trajectories overlaid (green: GT, orange: prediction).

#### 0.B.1.2 Image Generation from Dense World Modeling

Fig.˜6 presents qualitative results of our dense world modeling on the navtest split. Given the current and historical frames, our model generates future RGB images and depth maps that closely align with the ground truth. We note that while fine-grained details such as texture sharpness and thin structures (*e.g*., traffic lights, tree branches) exhibit some degradation compared to ground truth, the global scene geometry and layout are well preserved.

![[x6 18.png|Refer to caption]]

Figure 6: Qualitative results of dense world modeling on the navtest split. Each row shows a driving scenario with five columns: the current frame, ground truth and predicted future RGB images, and ground truth and predicted future depth maps.

### 0.B.2 Evaluation on nuScenes

#### 0.B.2.1 Evaluation Metrics.

We adopt two widely used open-loop planning evaluation protocols on nuScenes: the ST-P3 protocol \[hu2022st\] and the UniAD protocol \[hu2023planning\]. Both protocols evaluate planned trajectories over 1s, 2s, and 3s future horizons using two core metrics: L2 error, which measures the Euclidean distance between predicted and ground-truth trajectory waypoints, and collision rate, which computes the frequency of collisions between the ego vehicle’s occupied area along the planned trajectory and the bounding boxes of surrounding agents.

#### 0.B.2.2 Quantitative Analysis.

We further evaluate ExploreVLA on the nuScenes dataset to demonstrate the performance of our model. Note that we only apply Stage 1 (pre-training and supervised fine-tuning) without Stage 2 reinforcement learning post-training, as nuScenes lacks well-established closed-loop evaluation metrics that could serve as effective reward signals for RL optimization. As shown in Tab.˜5, we compare ExploreVLA against state-of-the-art methods under both the ST-P3 \[hu2022st\] and UniAD \[hu2023planning\] evaluation protocols. Under the ST-P3 protocol, ExploreVLA achieves competitive L2 errors (0.44m average) while attaining the lowest average collision rate of 0.10%, matching OpenDriveVLA \[zhou2025opendrivevla\] and substantially outperforming other baselines. Notably, our model achieves the best collision rate at 1s and 2s horizons, indicating strong short-term safety-aware planning. Under the UniAD protocol, ExploreVLA also obtains reasonable L2 errors (0.77m average), remaining competitive with established methods such as UniAD (1.03m) and AutoVLA (0.70m). These results confirm that even without RL-based post-training, our Stage 1 model already learns robust driving representations that perform consistently well across benchmarks.

Table 5: Comparison on the nuScenes dataset. The best performance is marked in bold.

<table><tbody><tr><td rowspan="3">Model</td><td colspan="8">ST-P3 metrics <cite>[hu2022st]</cite></td><td colspan="8">UniAD metrics <cite>[hu2023planning]</cite></td></tr><tr><td colspan="4">L2 (m) ↓</td><td colspan="4">Collision (%) ↓</td><td colspan="4">L2 (m) ↓</td><td colspan="4">Collision (%) ↓</td></tr><tr><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td></tr><tr><td>ST-P3 <cite>[hu2022st]</cite></td><td>1.33</td><td>2.11</td><td>2.90</td><td>2.11</td><td>0.23</td><td>0.62</td><td>1.27</td><td>0.71</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>VAD <cite>[jiang2023vad]</cite></td><td>0.17</td><td>0.34</td><td>0.60</td><td>0.37</td><td>0.07</td><td>0.10</td><td>0.24</td><td>0.14</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>UniAD <cite>[hu2023planning]</cite></td><td>0.44</td><td>0.67</td><td>0.96</td><td>0.69</td><td>0.04</td><td>0.08</td><td>0.23</td><td>0.12</td><td>0.48</td><td>0.96</td><td>1.65</td><td>1.03</td><td>0.05</td><td>0.17</td><td>0.71</td><td>0.31</td></tr><tr><td>EMMA <cite>[hwang2024emma]</cite></td><td>0.14</td><td>0.29</td><td>0.54</td><td>0.32</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>OpenEMMA <cite>[xing2025openemma]</cite></td><td>1.45</td><td>3.21</td><td>3.76</td><td>2.81</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>OpenDriveVLA <cite>[zhou2025opendrivevla]</cite></td><td>0.14</td><td>0.30</td><td>0.55</td><td>0.33</td><td>0.02</td><td>0.07</td><td>0.22</td><td>0.10</td><td>0.19</td><td>0.58</td><td>1.24</td><td>0.67</td><td>0.02</td><td>0.18</td><td>0.70</td><td>0.30</td></tr><tr><td>AutoVLA <cite>[zhou2025autovla]</cite></td><td>0.21</td><td>0.38</td><td>0.60</td><td>0.40</td><td>0.13</td><td>0.18</td><td>0.28</td><td>0.20</td><td>0.28</td><td>0.66</td><td>1.16</td><td>0.70</td><td>0.14</td><td>0.25</td><td>0.53</td><td>0.31</td></tr><tr><td>ExploreVLA</td><td>0.28</td><td>0.40</td><td>0.65</td><td>0.44</td><td>0.01</td><td>0.05</td><td>0.25</td><td>0.10</td><td>0.31</td><td>0.64</td><td>1.37</td><td>0.77</td><td>-</td><td>-</td><td>-</td><td>-</td></tr></tbody></table>
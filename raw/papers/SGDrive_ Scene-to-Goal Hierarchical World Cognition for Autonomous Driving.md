---
title: "SGDrive: Scene-to-Goal Hierarchical World Cognition for Autonomous Driving"
source: "https://arxiv.org/html/2601.05640v2"
author:
published:
created: 2026-08-17
description:
tags:
  - "clippings"
---
Jingyu Li    Junjie Wu    Dongnan Hu Affiliation: Shanghai Innovation Institute Affiliation: Tongji University    Xiangkai Huang Affiliation: Li Auto Inc.    Bin Sun    Zhihui Hao    Xianpeng Lang Affiliation: Li Auto Inc.    Xiatian Zhu Affiliation: University of Surrey [github.com/LogosRoboticsGroup/SGDrive](https://github.com/LogosRoboticsGroup/SGDrive)    Li Zhang    \[2mm\] Fudan University

###### Abstract

Recent end-to-end autonomous driving approaches have leveraged Vision-Language Models (VLMs) to enhance planning capabilities in complex driving scenarios. However, VLMs are inherently trained as generalist models, lacking specialized understanding of driving-specific reasoning in 3D space and time. When applied to autonomous driving, these models struggle to establish structured spatial-temporal representations that capture geometric relationships, scene context, and motion patterns critical for safe trajectory planning. To address these limitations, we propose SGDrive, a novel framework that explicitly structures the VLM’s representation learning around driving-specific knowledge hierarchies. Built upon a pre-trained VLM backbone, SGDrive decomposes driving understanding into a scene-agent-goal hierarchy that mirrors human driving cognition: drivers first perceive the overall environment (scene context), then attend to safety-critical agents and their behaviors, and finally formulate short-term goals before executing actions. This hierarchical decomposition provides the structured spatial-temporal representation that generalist VLMs lack, integrating multi-level information into a compact yet comprehensive format for trajectory planning. Extensive experiments on the NAVSIM benchmark demonstrate that SGDrive achieves state-of-the-art performance among camera-only methods on both PDMS and EPDMS, validating the effectiveness of hierarchical knowledge structuring for adapting generalist VLMs to autonomous driving.

<sup>†</sup>

## 1 Introduction

In recent years, end-to-end (E2E) autonomous driving techniques have achieved significant strides. However, these methods [^13] [^14] [^21] [^45] often lack explicit causal reasoning and high-level scene understanding, exhibiting limitations in complex, long-tail traffic scenarios. The advent of large language models [^1] [^49], particularly Vision-Language Models (VLMs) [^7] [^51], have spurred efforts to integrate their rich prior knowledge and sophisticated reasoning capabilities into the driving tasks, aiming to mitigate these shortcomings and prevent unsafe maneuvers. However, how to effectively translate a VLM’s powerful cognitive understanding into physically safe and reliable driving actions remains an open challenge.

![[intro_fig1.png|Refer to caption]]

Figure 1: (a) directly produces driving actions in textual form. (b) VLM generates action embeddings that decoded to produce the final trajectory. (c) Our SGDrive explicitly learns and forecasts scene, agent, and goal knowledge, providing structured driving-world understanding that strengthens action reasoning and improves generalization.

To transfer pretrained VLM knowledge to autonomous driving, several methods [^43] [^52] [^15] create domain-specific datasets that adapt the model’s knowledge to driving scenarios. Building upon this foundation, some works [^16] [^55] [^8] attempt to directly generate trajectories in textual form Figure1(a). Subsequent methods [^27] [^12] draw inspiration from embodied intelligence researches [^3] [^23] and employ diffusion-based decoder to produce driving trajectories Figure1(b). While the aforementioned methods have achieved impressive results, they still suffer from several key limitations: (1) *Lack of spatial perception:* VLMs are inherently focused on semantic understanding and lack foundational spatial and geometric knowledge [^43] [^39] [^56]. (2) *Difficulty in discerning critical information*: Current methods [^33] [^29] often focus on the entire scene and lack the extraction of important information, leading to sub-optimal driving performance. (3) *Lack of future world state forecast:* Lack of temporal modeling of future world state evolution., such as how the surrounding scene will change. Therefore, we argue that previous methods fail to sufficiently represent the world and forecast its future state, thus hindering the realization of safe and reliable driving.

To address the aforementioned issues, we propose SGDrive, a novel framework that integrates structured and hierarchical world knowledge into VLMs, thereby enhancing the model’s capability to understand and represent driving-relevant world knowledge. As shown in Figure1(c), we introduce a set of special tokens, called ⟨world⟩, which are explicitly trained to extract comprehensive driving-relevant world knowledge. This structured knowledge captures geometric and semantic information as well as high-level driving objectives. By explicitly activating the model’s ability to perceive and represent this structured, world knowledge, we fundamentally enhance its 3D spatial perception, enabling the VLM to better guide trajectory generation and avoid potential collisions.

We guide the model to acquire comprehensive driving-relevant world knowledge from three complementary aspects: (1) *Scene geometric layout*: Our method leverages occupancy supervision [^66] [^57] to learn the overall geometric structure of the scene, enabling the model to perceive and predict holistic spatial variations while removing redundant semantic dependencies. (2) *Perceiving driving-relevant agents*: By incorporating an agent detection module [^4] [^33], the model focuses on identifying agents that are likely to influence ego-vehicle motion, instead of all visible objects, aligning better with human driving behavior. (3) *Inferring driving goal*: The model learns to reason about feasible driving goal that reflect human-like driving intentions and are consistent with the current scene context. Together, these components provide the model with a comprehensive prior over both the current and future world states, facilitating safer trajectory planning. To ensure disentangled representation across these knowledge, we further apply a block-wise masked attention mechanism to prevent information leakage between different knowledge. Finally, to effectively translate the driving-relevant world knowledge into trajectory outputs, we employ a diffusion-based transformer [^37] (DiT) as the trajectory generator. This design enables the model to progressively refine trajectory predictions conditioned on the extracted world knowledge, ensuring coherent trajectory generation.

Our main contributions are summarized as follows: (i) We propose a novel framework that guides the VLM to learn comprehensive world knowledge from different aspects, enhancing its spatial perception and world representation for safe autonomous driving. (ii) We design a block-wise masked attention mechanism to prevent knowledge leakage and noise interference, and couple it with a DiT decoder to generate trajectories conditioned on hierarchical world knowledge. (iii) Our method achieves state-of-the-art performance on the NAVSIM benchmark among camera-only methods, demonstrating its effectiveness through extensive experiments.

## 2 Related works

![[pipeline1.png|Refer to caption]]

Figure 2: The SGDrive pipeline introduces hierarchical ⟨ world ⟩ queries (scene, agent, and goal) for world modeling and trajectory generation. A key component is our world query encoder, which initializes these queries by integrating multi-modal priors from the ego state, historical trajectory, and visual features. These “prior-informed” queries are then processed by the VLM, alongside text and visual embeddings, to fuse all signals into a compact, hierarchical world representation.

### 2.1 End-to-end autonomous driving

Recent advancements in autonomous driving have shifted from isolated task pipelines to unified end-to-end planning frameworks that jointly address scene perception and ego-trajectory generation [^13] [^54] [^5] [^61] [^60], with further efforts have incorporated multi-modal inputs and transformer-based architectures for enhanced global reasoning [^38] [^42] [^18] [^17]. UniAD [^14] extends the scope by integrating a wide range of subtasks into a cohesive system, while VAD [^21] further improves it with vectorized representations and refined modular design. SparseAD [^63] and SparseDrive [^45] explore sparse representations to improve the efficiency and scalability of end-to-end planning systems. VADv2 [^6] pioneers the integration of probabilistic planning into end-to-end autonomous driving. DiffusionDrive [^30] leverages a truncated diffusion policy to improve the accuracy and diversity of planned trajectories. WoTE [^26] builds a world-model-based autonomous driving system upon the BEV representation. Although these methods have achieved remarkable success, their imitation-learning nature inherently limits generalization to long-tail and complex scenarios, while lacking reasoning and deliberative capabilities.

### 2.2 Vision-language-model in autonomous driving

Within autonomous driving, LLMs [^1] and VLMs [^25] [^70] [^31] have revealed strong reasoning and multi-modal capabilities. Concurrently, the introduction of large-scale driving datasets [^19] [^10] [^22] with natural language annotations [^36] [^40] [^35] [^67] has facilitated research. Notably, DriveLM [^43] introduces a Chain-of-Thought reasoning paradigm spanning perception to planning, while DriveMM [^15] integrates diverse language-driving datasets to build a universal VLM. Inspired by advancements in the field of robotic manipulation [^23] [^3] [^65], a new class of Vision-Language-Action (VLA) models based on VLMs has emerged to autonomous driving [^47] [^20] [^24] [^34]. EMMA [^16], built upon Gemini [^46], maps raw camera inputs to driving decisions and demonstrates high accuracy in perception and strong performance in motion planning. SimLingo [^41] introduces a VLM-based architecture that unifies driving, vision-language understanding, and language-action alignment. OpenDriveVLA [^68] employs hierarchical vision-language alignment and agent-environment-ego interaction to produce reliable driving actions, excelling in planning and driving-related question answering tasks. ORION [^12] proposes a framework that integrates QT-Former, a large language model (LLM), and a generative planner, achieving strong performance in closed-loop evaluations. FSDrive [^59] reformulates CoT reasoning into a spatio-temporal visual CoT, where a VLM generates a future frame and subsequently predicts the trajectory via inverse dynamics. ReCogDrive [^27] proposes a framework that integrates a VLM with a diffusion planner, utilizing Diffusion Group Relative Policy Optimization to enhance trajectory generation. In contrast to previous work, SGDrive represents and forecasts world knowledge in a hierarchical (scene-agent-goal) and effective manner, enabling safe and reliable driving.

## 3 Method

### 3.1 Problem definition and notation

We aim to enhance the safety of autonomous driving through the guidance of multi-level driving-relevant world knowledge. Accordingly, we formulate our framework as tackling two complementary sub-problems: extracting representative world knowledge and extrapolating future world states. At each time step $t$, the ego-vehicle receives heterogeneous signals: a natural language instruction $L_{ins}$, the ego-vehicle state $S_{ego}$, and camera sensor inputs $I_{cam}$. These inputs together constitute the world knowledge of the driving system. To direct the model’s attention toward driving cues and facilitate forecasting of future world states, we introduce a set of special tokens, denoted as ⟨world⟩, and adopt a VLM to transform the comprehensive world knowledge into compact latent representations:

$$
O_{\text{world}}=VLM\bigl(I_{cam},\,L_{ins},\,S_{ego}\,|\,\textlangle\text{world}\textrangle\bigr).
$$

The resulting representation $O_{\text{world}}$ encapsulates the driving-related world knowledge and serves as the foundation for forecasting the future evolution of the world state. We then introduce a set of hierarchical world heads $\mathcal{D}$ that extract structured world knowledge across geometric details, motion cues, and high-level cognition, and further extrapolate the world state at future time $t{+}n$:

$$
w=\mathcal{D}\bigl(O_{world}\bigr)=\{w^{t,t+n}_{geo},\,w^{t,t+n}_{agt},\,w_{goal}\},
$$

where $w_{geo}$ represents the layout of the scene, $w_{agt}$ captures the state of the safety-critical agents, and $w_{goal}$ encodes the short-term driving objective.

Based on the learned world knowledge, we utilize DiT [^37] to generate trajectory. This design effectively translates the forecast world knowledge into coherent and safe future trajectories, completing the reasoning chain from scene understanding to motion planning.

### 3.2 Model architecture

As illustrated in Figure 2, our SGDrive builds on a foundational VLM with two core modules. SGDrive accepts heterogeneous inputs and processes each modality separately. Specifically, we use a standard text tokenizer encodes the driving instruction, while a ViT-based visual encoder [^71] extracts features from the camera image. We further introduce a set of ⟨world⟩ queries, which are appended to the multimodal embeddings. And these queries consist of three subqueries (scene, agent, and goal), which is used for predicting hierarchical driving world knowledge. The world queries are initialized via a world query encoder, which integrates multi-modal priors from the ego state, historical trajectory, and visual embeddings (Figure 2). These prior-informed queries effectively capture contextual information from the scene. Combined with the VLM’s strong priors, our method fuses visual and ego-vehicle signals into a compact hierarchical representation that encodes the scene geometry, dynamic agent states, short-term driving objectives, and predicted future world states, providing a structured and interpretable foundation for subsequent world modeling and trajectory generation.

In the decoding stage, task-specific heads process the corresponding subqueries, transforming the unified world embedding into explicit representations (scene context, agent states, and short-term goals). The latent world embedding implicitly conveys hierarchical driving knowledge, enabling downstream trajectory generation without requiring explicit decoding of the ⟨world⟩ queries, thereby reducing computational cost while preserving semantic richness.

### 3.3 Hierarchical world knowledge representation

Safe driving requires anticipating how the surrounding environment will evolve. Our method mirrors human driving cognition by explicitly forecasting future world knowledge along three complementary aspects: scene geometry, safety-critical agents, and short-term driving goals. This structured scene-agent-goal representation supplements the missing depth information in images and provides predictive context to support safe trajectory generation.

Geometric scene layout perception. Perceiving and forecasting the geometric layout of the driving scene provides essential spatial cues for safe driving. Our model does not predict high-level semantic distributions; instead, it focuses on the overall geometric structure. When occupancy annotations are available in the dataset, we supervise the model with ground-truth labels; otherwise, we generate occupancy from the point clouds. Since a VAE decoder excels at reconstructing features from latent representations [^50] [^48] [^32], we treat the VLM output $W_{\text{geo}}$ as the latent embedding and employ a standard VAE decoder for geometric reconstruction. Considering the high sparsity of driving scenes with a large number of negative samples [^32], we follow prior work [^44] and adopt a resampling strategy, supervised by two classification losses to ensure balanced learning of occupied and unoccupied regions:

$$
\displaystyle\mathcal{L}_{\text{geo}}^{t,t+n}
$$
 
$$
\displaystyle=\frac{1}{M}\sum_{i=1}^{M}\mathrm{CE}(o_{i}^{t,t+n},\hat{o}_{i}^{t,t+n})
$$
 
$$
\displaystyle\hskip 10.00002pt+\frac{1}{N}\sum_{j=1}^{N}\mathrm{BCE}(p_{j}^{t,t+n},\hat{p}_{j}^{t,t+n}),
$$

where $\mathcal{L}_{\text{geo}}$ combines the standard cross-entropy loss over all spatial locations with a resampled binary cross-entropy term to handle sparse occupancy distribution. $o_{i}\in\{0,1\}$ indicates whether location $j$ is occupied in the ground truth, and $p_{i}$ denotes resampled candidate positions.

Safety-critical agents detection. To handle complex driving interactions, our SGDrive focuses on safety-critical road users that directly influence driving behavior, encouraging collision-avoidant reasoning and deeper understanding of safety-critical interactions. Specifically, we select target agents (vehicle, pedestrian and cyclist) based on ego-vehicle trajectory and visibility from the front-view camera frustum. This strategy compels the model to allocate its finite representational capacity to agents most relevant to the ego-vehicle’s decisions, rather than exhaustively perceiving all objects in the scene. For these selected agents, the model predicts their 3D states at both the current and future time steps ($t$ and $t+n$). To supervise these agent predictions, we adopt the set-based loss paradigm from DETR [^4], which finds an optimal bipartite matching $\hat{\sigma}$ between the $N_{q}$ predictions and the set of ground-truth objects. The total loss $\mathcal{L}_{\text{agent}}$ is then computed as a weighted sum of classification and regression losses for the matched pairs:

$$
\displaystyle\mathcal{L}_{\text{agent}}^{t,t+n}
$$
 
$$
\displaystyle=\sum_{i=1}^{N_{q}}\bigl[\lambda_{\text{cls}}\mathcal{L}_{\text{cls}}(\hat{c}_{i}^{t,t+n},c_{\hat{\sigma}(i)}^{t,t+n})
$$
 
$$
\displaystyle\hskip 10.00002pt+\mathbf{1}_{c_{\hat{\sigma}(i)^{t,t+n}}\neq\emptyset}\mathcal{L}_{\text{reg}}(\hat{b}_{i}^{t,t+n},b_{\hat{\sigma}(i)}^{t,t+n})\bigr],
$$

where $\hat{\sigma}(i)$ is the index of the ground-truth object matched to the $i$ -th prediction. $\mathcal{L}_{\text{cls}}$ is a cross-entropy loss, $\lambda_{\text{cls}}$ takes 10, and $\mathcal{L}_{\text{reg}}$ is an $L_{1}$ loss. The indicator term $\mathbf{1}_{c_{\hat{\sigma}(i)}\neq\emptyset}$ ensures the regression loss is applied only to positive matches.

Short-term driving goal forecasting. Predicting short-term driving goals provides high-level semantic guidance for the ego-vehicle, indicating the intended trajectory over the immediate future. Without such predictions, ego-vehicle may exhibit incomplete or suboptimal maneuvers, such as covering only part of the planned path—potentially reducing task efficiency and safety. At the apex of the cognitive hierarchy, multi-modal interactions between visual and textual embeddings are used to infer the ego-vehicle’s intended objective. Rather than being directly conditioned on the previously established world representations (e.g., scene geometric layout or safety-critical agents), this goal reasoning emerges implicitly from a holistic understanding of the scene and task instructions. Building upon this implicit goal reasoning, SGDrive predicts a short-term driving goal, $\hat{p}_{\text{goal}}$, defined as the target ego-pose approximately 4 seconds into the future. This prediction is decoded via a lightweight mlp head and supervised using an $L_{1}$ loss against the ground-truth pose $p_{\text{goal}}$:

$$
\mathcal{L}_{\text{goal}}=||\hat{p}_{\text{goal}}-p_{\text{goal}}||_{1}.
$$

The predicted goal is more than a single trajectory point; it encodes high-level driving intentions informed by the model’s understanding of scene constraints and potential interactions. By explicitly predicting this goal, we effectively disentangle high-level decision-making from low-level trajectory planning.

![[attn-mask1.png|Refer to caption]]

Figure 3: (a) Causal attention mask: input tokens are allowed to attend to tokens before. (b) Structure attention mask: prevents leakage by prohibiting all mutual attention between the different subquery sets (scene, agent, goal).

Structured block-wise attention mask for driving-world knowledge. Our SGDrive employs specialized ⟨world⟩ queries to decode anticipatory knowledge, capturing both current state and future evolution. A key challenge in this multi-task design is representational contamination: if queries freely attend to each other (Figure 3(a)), information can leak across cognitive levels, compromising the integrity of specialized representations.

To address this, we introduce a block-wise structured attention mask (Figure 3(b)). The ⟨world⟩ queries are divided into five subqueries: the first three encode current world knowledge, and the remaining two focus on forecasting future states. The mask blocks attention across different knowledge categories while allowing temporal attention within each category, enabling subqueries to access relevant historical context. All subqueries remain free to cross-attend to the primary input modalities (visual and text embeddings), ensuring each representation gathers necessary evidence. This structured attention effectively prevents cross-level leakage, maintaining specialized and accurate hierarchical representations, which is essential for safe and precise driving.

### 3.4 Diffusion planner

A key challenge in autonomous driving is bridging the gap between high-level semantic reasoning and low-level continuous actions. To address this, we employ a diffusion planner [^23] [^3] [^37] that generates safe and context-aware action trajectories. Our ⟨world⟩ queries, which encode a hierarchical and anticipatory understanding of the driving world, are directly used as the latent condition for the planner. This avoids intermediate, lossy representations and enables access to the VLM’s full world knowledge while reducing inference overhead.

The planner denoises a sequence of future waypoints $\mathbf{A}=(a_{1},\dots,a_{N})$ from a noisy initialization $\mathbf{A}_{T}$ to the ground-truth trajectory $\mathbf{A}_{0}$ over $T$ steps. The per-step denoising is conditioned on the hierarchical world knowledge and ego-vehicle state, injected via the DiT’s cross-attention layers. Instead of starting from pure Gaussian noise, $\mathbf{A}_{T}$ is initialized by adding noise $\epsilon$ to a learned prior, generated from a linear projection of the ⟨world⟩ queries and historical ego-trajectory, grounding the process in the VLM’s world understanding. The diffusion model $\epsilon_{\theta}$ is trained with a standard $L_{2}$ objective:

$$
\mathcal{L}_{\text{diff}}=\mathbb{E}_{t,\mathbf{A}_{0},\epsilon}\left[\left\|\epsilon-\epsilon_{\theta}(\mathbf{A}_{t},t,c)\right\|_{2}^{2}\right],
$$

where $\mathbf{A}_{t}$ is the noisy trajectory at step $t$ and $c$ denotes the per-step condition.

Table 1: Performance comparison on NAVSIM v1 navtest using closed-loop metrics. We additionally report results with reinforcement learning fine-tuning (RFT) to enable fair comparison with methods that adopt such training strategies. <sup>†</sup> denotes models fine-tuned on the NAVSIM trajectory dataset. Bold indicates the best results under SFT and RFT settings, respectively.

<table><tbody><tr><td>Method</td><td>Image</td><td>Lidar</td><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Comf. <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>Constant Velocity</td><td></td><td></td><td>68.0</td><td>57.8</td><td>50.0</td><td>100</td><td>19.4</td><td>20.6</td></tr><tr><td>Ego Status MLP</td><td></td><td></td><td>93.0</td><td>77.3</td><td>83.6</td><td>100</td><td>62.8</td><td>65.6</td></tr><tr><td>VADv2- <math><semantics><msub><mi>𝒱</mi> <mtext>8192</mtext></msub> <annotation>\mathcal{V}_{\text{8192}}</annotation></semantics></math> <sup><a href="#fn:6">6</a></sup></td><td>✓</td><td></td><td>97.2</td><td>89.1</td><td>91.6</td><td>100</td><td>76.0</td><td>80.9</td></tr><tr><td>Hydra-MDP- <math><semantics><msub><mi>𝒱</mi> <mtext>8192</mtext></msub> <annotation>\mathcal{V}_{\text{8192}}</annotation></semantics></math> <sup><a href="#fn:28">28</a></sup></td><td>✓</td><td>✓</td><td>97.9</td><td>91.7</td><td>92.9</td><td>100</td><td>77.6</td><td>83.0</td></tr><tr><td>UniAD <sup><a href="#fn:14">14</a></sup></td><td>✓</td><td></td><td>97.8</td><td>91.9</td><td>92.9</td><td>100</td><td>78.8</td><td>83.4</td></tr><tr><td>LTF <sup><a href="#fn:38">38</a></sup></td><td>✓</td><td></td><td>97.4</td><td>92.8</td><td>92.4</td><td>100</td><td>79.0</td><td>83.8</td></tr><tr><td>BevDrive <sup><a href="#fn:54">54</a></sup></td><td>✓</td><td>✓</td><td>97.7</td><td>92.5</td><td>92.9</td><td>100</td><td>78.7</td><td>83.8</td></tr><tr><td>TransFuser <sup><a href="#fn:38">38</a></sup></td><td>✓</td><td>✓</td><td>97.7</td><td>92.8</td><td>92.8</td><td>100</td><td>79.2</td><td>84.0</td></tr><tr><td>PARA-Drive <sup><a href="#fn:53">53</a></sup></td><td>✓</td><td></td><td>97.9</td><td>92.4</td><td>93.0</td><td>99.8</td><td>79.3</td><td>84.0</td></tr><tr><td>DRAMA <sup><a href="#fn:58">58</a></sup></td><td>✓</td><td>✓</td><td>98.0</td><td>93.1</td><td>94.8</td><td>100</td><td>80.1</td><td>85.5</td></tr><tr><td>Epona <sup><a href="#fn:64">64</a></sup></td><td>✓</td><td></td><td>97.9</td><td>95.1</td><td>93.8</td><td>99.9</td><td>80.4</td><td>86.2</td></tr><tr><td>Hydra-MDP- <math><semantics><msub><mi>𝒱</mi> <mtext>8192</mtext></msub> <annotation>\mathcal{V}_{\text{8192}}</annotation></semantics></math> -W-EP <sup><a href="#fn:28">28</a></sup></td><td>✓</td><td>✓</td><td>98.3</td><td>96.0</td><td>94.6</td><td>100</td><td>78.7</td><td>86.5</td></tr><tr><td>ARTEMIS <sup><a href="#fn:11">11</a></sup></td><td>✓</td><td>✓</td><td>98.3</td><td>95.1</td><td>94.3</td><td>100</td><td>81.4</td><td>87.0</td></tr><tr><td>DiffusionDrive <sup><a href="#fn:30">30</a></sup></td><td>✓</td><td>✓</td><td>98.2</td><td>96.2</td><td>94.7</td><td>100</td><td>82.2</td><td>88.1</td></tr><tr><td>WoTE <sup><a href="#fn:26">26</a></sup></td><td>✓</td><td>✓</td><td>98.5</td><td>96.8</td><td>94.9</td><td>99.9</td><td>81.9</td><td>88.3</td></tr><tr><td>SeerDrive <sup><a href="#fn:62">62</a></sup></td><td>✓</td><td>✓</td><td>98.4</td><td>97.0</td><td>94.9</td><td>99.9</td><td>83.2</td><td>88.9</td></tr><tr><td colspan="8">VLMs-based Methods (SFT)</td><td></td></tr><tr><td>AutoVLA-3B <sup><a href="#fn:69">69</a></sup></td><td>✓</td><td></td><td>96.9</td><td>92.4</td><td>88.1</td><td>99.1</td><td>75.8</td><td>80.5</td></tr><tr><td>QwenVL2.5-8B <sup>†</sup> <sup><a href="#fn:2">2</a></sup></td><td>✓</td><td></td><td>97.8</td><td>92.1</td><td>92.8</td><td>100</td><td>78.3</td><td>83.3</td></tr><tr><td>InternVL3-8B <sup>†</sup> <sup><a href="#fn:71">71</a></sup></td><td>✓</td><td></td><td>97.0</td><td>92.4</td><td>91.8</td><td>100</td><td>78.9</td><td>83.3</td></tr><tr><td>ReCogDrive-2B <sup><a href="#fn:27">27</a></sup></td><td>✓</td><td></td><td>98.1</td><td>94.7</td><td>94.2</td><td>100</td><td>80.9</td><td>86.5</td></tr><tr><td>ReCogDrive-8B <sup><a href="#fn:27">27</a></sup></td><td>✓</td><td></td><td>98.3</td><td>95.1</td><td>94.3</td><td>100</td><td>81.1</td><td>86.8</td></tr><tr><td>SGDrive-2B (ours)</td><td>✓</td><td></td><td>98.6</td><td>95.1</td><td>95.4</td><td>100</td><td>81.2</td><td>87.4</td></tr><tr><td colspan="8">VLMs-based Methods (RFT)</td><td></td></tr><tr><td>AutoVLA-3B <sup><a href="#fn:69">69</a></sup></td><td>✓</td><td></td><td>98.4</td><td>95.6</td><td>98.0</td><td>99.9</td><td>81.9</td><td>89.1</td></tr><tr><td>ReCogDrive-2B <sup><a href="#fn:27">27</a></sup></td><td>✓</td><td></td><td>97.9</td><td>97.3</td><td>94.9</td><td>100</td><td>87.3</td><td>90.8</td></tr><tr><td>ReCogDrive-8B <sup><a href="#fn:27">27</a></sup></td><td>✓</td><td></td><td>97.8</td><td>97.7</td><td>94.9</td><td>100</td><td>86.3</td><td>90.5</td></tr><tr><td>SGDrive-2B (ours)</td><td>✓</td><td></td><td>98.6</td><td>97.8</td><td>96.2</td><td>100</td><td>85.8</td><td>91.1</td></tr></tbody></table>

### 3.5 Training objectives

Our SGDrive is trained in a two-stage procedure to effectively manage the distinct task spaces of world representation and action generation. In the first stage, we perform Supervised Fine-Tuning (SFT) to train the core VLM for both visual question answering (VQA) and comprehensive world knowledge acquisition. The model processes multi-frame front-view camera inputs, ego-vehicle states, and language-based driving commands, and is supervised to jointly predict: (1) textual answers for VQA $\mathcal{L}_{text}$, (2) the spatio-temporal scene geometry layout, (3) safety-critical agent detection, and (4) the short-term driving goal. The total loss is defined as:

$$
\mathcal{L}_{\text{Stage1}}=\mathcal{L}_{\text{text}}+\mathcal{L}_{\text{occ}}^{t,t+n}+\lambda_{\text{agent}}\mathcal{L}_{\text{agent}}^{t,t+n}+\mathcal{L}_{\text{goal}},
$$

where $\lambda_{\text{agent}}$ takes $0.1$.

In the second stage, we freeze the pre-trained VLM from stage 1 to serve as a high-fidelity world model, and train the diffusion planner with the same inputs but a single optimization target, the trajectory diffusion loss $\mathcal{L}_{\text{diff}}$. This staged strategy enables the VLM to first learn a robust, general-purpose representation of the driving world, which is then exploited by the diffusion planner to generate safe and realistic trajectories.

## 4 Experiments

Table 2: Performance comparison on NAVSIM v2 navtest with extended metrics. Our SGDrive-2B is evaluated using the model trained with the proposed two-stage SFT strategy.

| Method | NC $\uparrow$ | DAC $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | HC $\uparrow$ | TL $\uparrow$ | DDC $\uparrow$ | LK $\uparrow$ | EC $\uparrow$ | EPDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Transfuser [^38] | 97.7 | 92.8 | 79.2 | 92.8 | 100 | 99.9 | 98.3 | 67.6 | 95.3 | 77.8 |
| VADv2 [^6] | 97.3 | 91.7 | 77.6 | 92.7 | 100 | 99.9 | 98.2 | 66.0 | 97.4 | 76.6 |
| Hydra-MDP [^28] | 97.5 | 96.3 | 80.1 | 93.0 | 100 | 99.9 | 98.3 | 65.5 | 97.4 | 79.8 |
| Hydra-MDP++ [^28] | 97.9 | 96.5 | 79.2 | 93.4 | 100 | 100.0 | 98.9 | 67.2 | 97.7 | 80.6 |
| ARTEMIS [^11] | 98.3 | 95.1 | 81.5 | 97.4 | 100 | 99.8 | 98.6 | 96.5 | 98.3 | 83.1 |
| ReCogDrive-8B [^27] | 98.3 | 95.2 | 87.1 | 97.5 | 98.3 | 99.8 | 99.5 | 96.6 | 86.5 | 83.6 |
| DiffusionDrive [^30] | 98.0 | 96.0 | 87.7 | 97.1 | 98.3 | 99.8 | 99.5 | 97.2 | 87.6 | 84.3 |
| SGDrive-2B (ours) | 98.6 | 94.3 | 86.0 | 97.9 | 98.3 | 99.9 | 99.5 | 96.1 | 85.9 | 86.2 |

### 4.1 Experimental setup

Implementation Details. We use InternVL3-2B [^71] as our VLM backbone, which is integrated a 300M-parameter InternViT visual encoder [^7] with the Qwen2.5 large-language-model [^2]. In stage 1, we first perform domain adaptation to align the base VLM with the driving modality, following [^27]. We use over 3.1 million question-answer (QA) pairs covering perception, prediction, and planning, and train for 1 epoch. Subsequently, we fine-tune the VLM on 85k trajectory-specific QA pairs while concurrently training the world knowledge heads for 3 epoches. In Stage 2, we freeze the VLM parameters and train the diffusion planner exclusively for 220 epochs. All experiments are conducted on 4 nodes, each equipped with 8 NVIDIA H20 GPUs (32 GPUs total). Additional details are provided in the supplementary material.

Dataset and evaluation metrics. NAVSIM [^9] is a large-scale real-world autonomous driving dataset designed for non-reactive simulation and benchmarking. It focuses on challenging scenarios involving dynamic intention changes, while filtering out trivial cases such as stationary or constant-speed driving. It is split into two subsets: navtrain (1,192 scenarios) for training and validation, and navtest (136 scenarios) for testing. As for the evaluation metrics, we evaluate our method using the Predictive Driver Model Score (PDMS) and the Extended PDMS (EPDMS), as defined in the official benchmark. PDMS consists of several sub-scores, including No At-Fault Collisions (NC), Drivable Area Compliance (DAC), Time-to-Collision (TTC), Comfort (Comf.), and Ego Progress (EP). EPDMS further introduces Traffic Light Compliance (TL), Lane Keeping Ability (LK), and Extended Comfort (EC), providing a more comprehensive evaluation.

![[comparison_on_diff_model1.png|Refer to caption]]

Figure 4: Comparisons with state-of-the-art method on the Navtest benchmark.

### 4.2 Main result.

Results on NAVSIM v1 with SFT.

As shown in Table 1, we compare SGDrive against state-of-the-art approaches on the NAVSIM test split. Our approach, built upon an InternVL3-2B backbone and trained using our proposed two-stage supervised fine-tuning strategy, achieves a new state-of-the-art PDMS score of 87.4. This result is notable for several reasons. First, it surpasses larger general-purpose VLMs like InternVL3-8B and QwenVL2.5-8B by a significant 4.1 PDMS, demonstrating the superior performance of our specialized architecture. Second, SGDrive-2B model outperforms the previous state-of-the-art driving VLM method, Recogdrive-8B [^27], by 0.6 PDMS. This highlights the profound effectiveness of guiding the VLM to learn and forecast hierarchical world knowledge, as this enables a more compact model to achieve superior planning performance. Third, SGDrive relay on image-only input outperforms the vast majority of listed end-to-end methods that rely on both image and LiDAR inputs.

Crucially, our method achieves the best scores on the key collision-related metrics, NC and TTC. This strongly validates our core hypothesis: by explicitly forecasting the spatio-temporal layout, dynamic agent interactions, and short-term goals, the model gains a superior spatial-temporal awareness that is paramount for anticipating and avoiding potential collisions.

Results on NAVSIM v1 with RFT. Although our method primarily aims to learn hierarchical driving-world knowledge to improve driving safety, it can be seamlessly integrated with existing RL frameworks. Under the same RL training configuration as RecogDrive [^27], our approach achieves substantially better results, as shown in Table 1. By incorporating structured world knowledge features into the RL pipeline, our method achieves a PDMS of 91.1, outperforming all existing methods, including those using LiDAR inputs. Compared with other RL-based approaches, our model achieves the best performance on NC and DAC, indicating that the learned driving-world knowledge effectively reduces collision risk and improves compliance with drivable regions. In future work, we plan to explore RL algorithms specifically tailored to our hierarchical world knowledge forecasting framework to further improve driving efficiency and smoothness.

Results on NAVSIM v2 with SFT. To comprehensively evaluate our approach, we also follow prior work [^28] and adopt the Extended PDMS metric on the NAVSIM [^9] benchmark. As shown in Table 2, SGDrive achieves the best overall performance with an EPDMS of 86.2, outperforming the previous state-of-the-art ReCogDrive-8B by 2.6 points. Our method also delivers the strongest results on the safety-critical NC and TTC metrics, while maintaining competitive performance on the newly introduced TL, LK, and EC metrics. These results collectively demonstrate the effectiveness and robustness of SGDrive in modeling driving-relevant world knowledge under the extended evaluation protocol.

Table 3: Ablation study on the proposed components of SGDrive.

| Exp. | Base | Current | Future | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a | ✓ | ✗ | ✗ | 97.3 | 91.1 | 92.9 | 76.8 | 82.2 |
| b | ✓ | ✓ | ✗ | 98.3 | 93.0 | 94.9 | 78.2 | 84.7 |
| c | ✓ | ✓ | ✓ | 98.4 | 93.6 | 94.9 | 79.3 | 85.5 |

### 4.3 Ablation study

Effect of our driving-world knowledge forecast.

We first evaluate the effectiveness of our proposed driving-world knowledge learning in stage 1, where trajectories are produced in text form and the result is shown in Table 3. When the model is trained only to represent the multi-level structure of the *current* world state, as in Exp.(b), it achieves a 2.5 points improvement in PDMS over Exp.(a). This notable gain demonstrates that our hierarchical world representation successfully activates the model’s understanding of the 3D driving environment, leading to more accurate trajectory predictions. When we further incorporate *future* world forecasting in Exp.(c), the performance increases to 85.5 PDMS, along with additional improvements in the NC and EP metrics compared with Exp.(b). These results show that enabling the VLM to forecast future world evolution provides stronger safety awareness and planning efficiency, ultimately producing reliable autonomous driving behavior.

Table 4: Ablation study on the world query of SGDrive.

| Exp. | Scene | Agent | Goal | Furture | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a | ✓ | ✗ | ✗ | ✗ | 98.2 | 94.1 | 94.4 | 80.2 | 86.0 |
| b | ✓ | ✓ | ✗ | ✗ | 98.3 | 94.5 | 94.8 | 80.4 | 86.3 |
| c | ✓ | ✓ | ✓ | ✗ | 98.5 | 94.9 | 95.1 | 81.2 | 87.0 |
| d | ✓ | ✓ | ✓ | ✓ | 98.6 | 95.1 | 95.4 | 81.2 | 87.4 |

Ablation of world query for downstream planning.

To assess the effectiveness of each subquery within our ⟨world⟩queries, we conduct ablation studies on the downstream trajectory planning task using the stage 2 diffusion planner, and the result is shown in Table 4. Exp.(a) employs only the scene-geometry layout to modulate trajectory generation, achieving a PDMS of 86.0. When safety-critical agents information is added, notable improvements are observed in key metrics such as NC and DAC, Exp.(b). Exp.(c) add driving goals into condition further leads to a significant enhancement in EP, indicating that activating high-level semantic intent effectively improves driving efficiency. Finally, incorporating future world-state predictions to guide the planner yields consistent gains across multiple metrics, resulting in a PDMS of 87.4. The additional improvements in TTC and NC further demonstrate that modeling the evolution of future scenes and road-user motions enables the planner to better anticipate potential hazards and avoid collisions. These comprehensive results demonstrate that our proposed scene-agent-goal hierarchical cognition framework provides effective world knowledge guidance and substantially enhances overall driving performance.

![[comparison_prediction_gt1.png|Refer to caption]]

Figure 5: Qualitative visualization of our model’s predictions (top row) versus the ground truth (bottom row). The visualization shows our model accurately forecasts these hierarchical states, which closely align with the ground truth.

![[Adaptive_occ1.png|Refer to caption]]

Figure 6: Ego-motion based adaptive geometric scene perception.

Table 5: Ablation study on the attention mask of SGDrive.

| Method | NC $\uparrow$ | TTC $\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- |
| Causal | 98.4 | 95.6 | 80.1 | 87.1 |
| Structure | 98.6 | 95.4 | 81.2 | 87.4 |

Effect of structured attention masking.

We compare our attention mechanism with default causal-attention method, as shown in Figure 3. Causal-attention exposes each world query to all preceding tokens, introducing cross-category noise and semantic leakage; this corrupts world representations and causes the vehicle to adopt overly conservative behaviors (e.g., slowing excessively to avoid potential collisions), thereby reducing driving efficiency. By contrast, our structured attention restricts visibility to same-type information, producing cleaner, task-specific embeddings. As shown in Table 5, this design yields a favorable trade-off: it improves EP, yielding a higher overall PDMS and more realistic driving behavior.

### 4.4 Qualitative Results.

Compare with previous method. We select two representative scenarios to qualitatively compare our method with the previous state-of-the-art, RecogDrive [^27], as shown in Figure 4. In the first scenario, involving multiple road users, RecogDrive’s predicted trajectory deviates significantly from the ground truth, leading to a potential collision. In contrast, our SGDrive empowered by explicit safety-critical agent detection, generates an optimal and collision-free trajectory. In the second, relatively open yet curved-road scenario, RecogDrive’s prediction drifts out of the lane and collides with the roadside barrier. Our model, however, accurately perceives the scene’s geometric layout and successfully avoids the collision. These results demonstrate that SGDrive effectively learns structured driving-world knowledge and can extrapolate it reasonably to ensure safe and rational driving behavior.

Explicit world knowledge representation. In Figure 5, we provide a qualitative comparison between our model’s predictions and the ground-truth annotations. The visualizations show a strong alignment across the scene-agent–goal hierarchy. This consistency indicates that our model has learned rich driving-world knowledge, enabling reliable perception and representation of both the current state and its short-horizon future evolution.

Adaptive geometric scene perception. As shown in the Figure 6, SGDrive adaptively perceives the driving scene according to the ego-vehicle’s motion state and navigation command. For instance, at high speeds, it expands its perceptual horizon, whereas during turning maneuvers, it redirects its perceptual focus toward the turning direction. This demonstrates a more structured and effective representation of driving-relevant world knowledge, providing strong evidence that SGDrive successfully elicits the VLM’s world-modeling ability.

## 5 Conclusion

We introduce SGDrive, a novel framework that explicitly structures a VLM’s representation learning around a driving-specific knowledge hierarchy, enabling safer and more reliable autonomous driving. Our method leverages a scene-agent-goal hierarchical cognition scheme that disentangles the understanding of driving environments: it models the current world through scene geometry, road users, and driving goals, and further extrapolates their evolution into the future. To support this, we design a structured attention-mask mechanism that prevents information leakage and suppresses cross-category noise. Finally, by integrating a DiT-based planner, our approach uses the inferred driving-world knowledge to regulate trajectory generation. Comprehensive experiments on NAVSIM demonstrate the effectiveness of our method and show that it achieves state-of-the-art performance in safe driving.

## References

Supplementary Material

## 6 More experiments

### 6.1 Comparison of hidden state fusion methods in diffusion planner

Table 6: Comparison of hidden state fusion methods in diffusion planner.

| Exp. | NC $\uparrow$ | TTC $\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- |
| (a) | 98.2 | 95.0 | 80.6 | 87.1 |
| (b) | 98.1 | 95.1 | 79.7 | 86.9 |
| (c) | 98.6 | 95.4 | 81.2 | 87.4 |

As shown in Table 6, we compare several strategies for fusing hidden states within the diffusion planner. Exp. (a) incrementally injects the hidden states of different subqueries across successive cross-attention layers. Exp. (b) assigns distinct cross-attention layers to different subqueries. Exp. (c), which corresponds to our proposed design, concatenates all subquery hidden states and enables interaction at every cross-attention layer. All fusion strategies achieve strong performance, confirming that our subqueries encode rich driving-world knowledge and can effectively guide the trajectory generation process.

NAVSIM metric. NAVSIM [^9] scores driving agents in two steps. First, subscores in range $[0,1]$ are computed after simulation. Second, these subscores are aggregated into the PDM Score (PDMS) $\in[0,1]$. We use the following aggregation of subscores based on the official definition:

$$
\displaystyle\textrm{PDMS}=\underbrace{\Bigg({\prod_{m\in\{\texttt{NC},\texttt{DAC}\}}}\texttt{score}_{m}\Bigg)}_{\text{penalties}}\times
$$
 
$$
\displaystyle\underbrace{\Bigg(\frac{\sum_{w\in\{\texttt{EP},\texttt{TTC},\texttt{C}\}}\texttt{weight}_{w}\times\texttt{score}_{w}}{\sum_{w\in\{\texttt{EP},\texttt{TTC},\texttt{C}\}}\texttt{weight}_{w}}\Bigg)}_{\text{weighted average}}.
$$

Subscores are categorized by their importance as penalties or terms in a weighted average. A penalty punishes inadmissible behavior such as collisions with a factor $<1$. The weighted average aggregates subscores for other objectives such as progress and comfort.

NAVSIM metric with extended PDMS. Hydra-MDP++ [^28] extends the original PDMS metric by incorporating additional aspects of driving performance, including Traffic Lights Compliance (TL), Lane Keeping Ability (LK), and Extended Comfort (EC), providing a more comprehensive evaluation of a method’s effectiveness. Formally, the Extended PDM Score (EPDMS) is computed as:

$$
\displaystyle\mathrm{EPDMS}
$$
 
$$
\displaystyle=\underbrace{\prod_{m\in\{\mathrm{NC},\mathrm{DAC},\mathrm{DDC},\mathrm{TL}\}}S^{m}}_{\text{penalty terms}}
$$
 
$$
\displaystyle\times\underbrace{\frac{\sum_{w\in\{\mathrm{EP},\mathrm{TTC},\mathrm{C},\mathrm{LK},\mathrm{EC}\}}\mathrm{weight}_{w}\cdot S^{w}}{\sum_{w\in\{\mathrm{EP},\mathrm{TTC},\mathrm{C},\mathrm{LK},\mathrm{EC}\}}\mathrm{weight}_{w}}}_{\text{weighted average of positive indicators}}.
$$

Here, the first term accumulates multiplicative penalties for safety-critical violations, while the second term computes a weighted average over positive performance indicators, providing a balanced assessment of driving quality and comfort.

![[supp_gostraight1.png|Refer to caption]]

Figure 7: Qualitative results on the Navtest benchmark.

## 7 Additional visualizations

### 7.1 Qualitative results

We provide additional qualitative results in Figure 7 and Figure 8. For both straight-driving and turning scenarios, our predicted trajectories closely follow the ground truth. We also include several failure cases.

### 7.2 Failure cases

As illustrated in Figure 9, when relying solely on a single front-view image, the model may exhibit slight deviations under extreme turning conditions. In such scenarios, due to the absence of corresponding viewpoints, accurately predicting long-horizon trajectories becomes challenging, sometimes leading to lane-change errors. Incorporating multi-view inputs is a promising direction to mitigate these limitations in future work.

![[supp_turn1.png|Refer to caption]]

Figure 8: Qualitative results on the Navtest benchmark.

![[supp_fail1.png|Refer to caption]]

Figure 9: Qualitative analysis of representative failure cases on the Navtest benchmark.

[^1]: J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774. Cited by: §1, §2.2.

[^2]: S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang, H. Zhong, Y. Zhu, M. Yang, Z. Li, J. Wan, P. Wang, W. Ding, Z. Fu, Y. Xu, J. Ye, X. Zhang, T. Xie, Z. Cheng, H. Zhang, Z. Yang, H. Xu, and J. Lin Qwen2.5-vl technical report. External Links: 2502.13923, [Link](https://arxiv.org/abs/2502.13923) Cited by: Table 1, §4.1.

[^3]: K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, et al. $\pi{{}_{0}}$: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164. Cited by: §1, §2.2, §3.4.

[^4]: N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko End-to-end object detection with transformers. Cited by: §1, §3.3.

[^5]: S. Casas, A. Sadat, and R. Urtasun Mp3: a unified model to map, perceive, predict and plan. In CVPR, Cited by: §2.1.

[^6]: S. Chen, B. Jiang, H. Gao, B. Liao, Q. Xu, Q. Zhang, C. Huang, W. Liu, and X. Wang Vadv2: end-to-end vectorized autonomous driving via probabilistic planning. arXiv preprint arXiv:2402.13243. Cited by: §2.1, Table 1, Table 2.

[^7]: Z. Chen, J. Wu, W. Wang, W. Su, G. Chen, S. Xing, M. Zhong, Q. Zhang, X. Zhu, L. Lu, et al. Internvl: scaling up vision foundation models and aligning for generic visual-linguistic tasks. In CVPR, Cited by: §1, §4.1.

[^8]: H. Chi, H. Gao, Z. Liu, J. Liu, C. Liu, J. Li, K. Yang, Y. Yu, Z. Wang, W. Li, L. Wang, X. Hu, H. Sun, H. Zhao, and H. Zhao Impromptu vla: open weights and open data for driving vision-language-action models. Cited by: §1.

[^9]: D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, et al. Navsim: data-driven non-reactive autonomous vehicle simulation and benchmarking. In NeurIPS, Cited by: §4.1, §4.2, §6.1.

[^10]: A. Dosovitskiy, G. Ros, F. Codevilla, A. Lopez, and V. Koltun CARLA: an open urban driving simulator. In CoRL, Cited by: §2.2.

[^11]: R. Feng, N. Xi, D. Chu, R. Wang, Z. Deng, A. Wang, L. Lu, J. Wang, and Y. Huang ARTEMIS: autoregressive end-to-end trajectory planning with mixture of experts for autonomous driving. External Links: 2504.19580, [Link](https://arxiv.org/abs/2504.19580) Cited by: Table 1, Table 2.

[^12]: H. Fu, D. Zhang, Z. Zhao, J. Cui, D. Liang, C. Zhang, D. Zhang, H. Xie, B. Wang, and X. Bai ORION: a holistic end-to-end autonomous driving framework by vision-language instructed action generation. arXiv preprint arXiv:2503.19755. Cited by: §1, §2.2.

[^13]: S. Hu, L. Chen, P. Wu, H. Li, J. Yan, and D. Tao St-p3: end-to-end vision-based autonomous driving via spatial-temporal feature learning. In ECCV, Cited by: §1, §2.1.

[^14]: Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, et al. Planning-oriented autonomous driving. In CVPR, Cited by: §1, §2.1, Table 1.

[^15]: Z. Huang, C. Feng, F. Yan, B. Xiao, Z. Jie, Y. Zhong, X. Liang, and L. Ma Drivemm: all-in-one large multimodal model for autonomous driving. arXiv preprint arXiv:2412.07689. Cited by: §1, §2.2.

[^16]: J. Hwang, R. Xu, H. Lin, W. Hung, J. Ji, K. Choi, D. Huang, T. He, P. Covington, B. Sapp, et al. Emma: end-to-end multimodal model for autonomous driving. arXiv preprint arXiv:2410.23262. Cited by: §1, §2.2.

[^17]: B. Jaeger, K. Chitta, and A. Geiger Hidden biases of end-to-end driving models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 8240–8249. Cited by: §2.1.

[^18]: X. Jia, P. Wu, L. Chen, J. Xie, C. He, J. Yan, and H. Li Think twice before driving: towards scalable decoders for end-to-end autonomous driving. In CVPR, Cited by: §2.1.

[^19]: X. Jia, Z. Yang, Q. Li, Z. Zhang, and J. Yan Bench2Drive: towards multi-ability benchmarking of closed-loop end-to-end autonomous driving. In NeurIPS, Cited by: §2.2.

[^20]: B. Jiang, S. Chen, B. Liao, X. Zhang, W. Yin, Q. Zhang, C. Huang, W. Liu, and X. Wang Senna: bridging large vision-language models and end-to-end autonomous driving. arXiv preprint arXiv:2410.22313. Cited by: §2.2.

[^21]: B. Jiang, S. Chen, Q. Xu, B. Liao, J. Chen, H. Zhou, Q. Zhang, W. Liu, C. Huang, and X. Wang Vad: vectorized scene representation for efficient autonomous driving. In ICCV, Cited by: §1, §2.1.

[^22]: N. Karnchanachari, D. Geromichalos, K. S. Tan, N. Li, C. Eriksen, S. Yaghoubi, N. Mehdipour, G. Bernasconi, W. K. Fong, Y. Guo, et al. Towards learning-based planning: the nuplan benchmark for real-world autonomous driving. In ICRA, Cited by: §2.2.

[^23]: M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, Q. Vuong, T. Kollar, B. Burchfiel, R. Tedrake, D. Sadigh, S. Levine, P. Liang, and C. Finn OpenVLA: an open-source vision-language-action model. arXiv preprint arXiv:2406.09246. Cited by: §1, §2.2, §3.4.

[^24]: J. Li, B. Zhang, X. Jin, J. Deng, X. Zhu, and L. Zhang ImagiDrive: a unified imagination-and-planning framework for autonomous driving. arXiv preprint arXiv:2508.11428. Cited by: §2.2.

[^25]: J. Li, D. Li, S. Savarese, and S. Hoi BLIP-2: bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML, Cited by: §2.2.

[^26]: Y. Li, Y. Wang, Y. Liu, J. He, L. Fan, and Z. Zhang End-to-end driving with online trajectory evaluation via bev world model. Cited by: §2.1, Table 1.

[^27]: Y. Li, K. Xiong, X. Guo, F. Li, S. Yan, G. Xu, L. Zhou, L. Chen, H. Sun, B. Wang, K. Ma, G. Chen, H. Ye, W. Liu, and X. Wang ReCogDrive: a reinforced cognitive framework for end-to-end autonomous driving. Cited by: §1, §2.2, Table 1, Table 1, Table 1, Table 1, §4.1, §4.2, §4.2, §4.4, Table 2.

[^28]: Z. Li, K. Li, S. Wang, S. Lan, Z. Yu, Y. Ji, Z. Li, Z. Zhu, J. Kautz, Z. Wu, et al. Hydra-mdp: end-to-end multimodal planning with multi-target hydra-distillation. arXiv preprint arXiv:2406.06978. Cited by: Table 1, Table 1, §4.2, Table 2, Table 2, §6.1.

[^29]: Z. Li, W. Wang, H. Li, E. Xie, C. Sima, T. Lu, Y. Qiao, and J. Dai Bevformer: learning bird’s-eye-view representation from multi-camera images via spatiotemporal transformers. In ECCV, Cited by: §1.

[^30]: B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang, and X. Wang DiffusionDrive: truncated diffusion model for end-to-end autonomous driving. In CVPR, Cited by: §2.1, Table 1, Table 2.

[^31]: H. Liu, C. Li, Q. Wu, and Y. J. Lee Visual instruction tuning. In NeurIPS, Cited by: §2.2.

[^32]: R. Liu, L. Kong, D. Li, and H. Zhao OccVLA: vision-language-action model with implicit 3d occupancy supervision. arXiv preprint arXiv:2509.05578. Cited by: §3.3.

[^33]: Y. Liu, T. Wang, X. Zhang, and J. Sun Petr: position embedding transformation for multi-view 3d object detection. In ECCV, Cited by: §1, §1.

[^34]: Z. Liu, R. Huang, R. Yang, S. Yan, Z. Wang, L. Hou, D. Lin, X. Bai, and H. Zhao DrivePI: spatial-aware 4d mllm for unified autonomous driving understanding, perception, prediction and planning. arXiv preprint arXiv:2512.12799. Cited by: §2.2.

[^35]: A. Marcu, L. Chen, J. Hünermann, A. Karnsund, B. Hanotte, P. Chidananda, S. Nair, V. Badrinarayanan, A. Kendall, J. Shotton, et al. Lingoqa: visual question answering for autonomous driving. In ECCV, Cited by: §2.2.

[^36]: M. Nie, R. Peng, C. Wang, X. Cai, J. Han, H. Xu, and L. Zhang Reason2drive: towards interpretable and chain-based reasoning for autonomous driving. In ECCV, Cited by: §2.2.

[^37]: W. Peebles and S. Xie Scalable diffusion models with transformers. In CVPR, Cited by: §1, §3.1, §3.4.

[^38]: A. Prakash, K. Chitta, and A. Geiger Multi-modal fusion transformer for end-to-end autonomous driving. In CVPR, Cited by: §2.1, Table 1, Table 1, Table 2.

[^39]: Z. Qi, R. Dong, G. Fan, Z. Ge, X. Zhang, K. Ma, and L. Yi Contrast with reconstruct: contrastive 3d representation learning guided by generative pretraining. In International Conference on Machine Learning, pp. 28223–28243. Cited by: §1.

[^40]: T. Qian, J. Chen, L. Zhuo, Y. Jiao, and Y. Jiang Nuscenes-qa: a multi-modal visual question answering benchmark for autonomous driving scenario. In AAAI, Cited by: §2.2.

[^41]: K. Renz, L. Chen, E. Arani, and O. Sinavski SimLingo: vision-only closed-loop autonomous driving with language-action alignment. In CVPR, Cited by: §2.2.

[^42]: H. Shao, L. Wang, R. Chen, H. Li, and Y. Liu Safety-enhanced autonomous driving using interpretable sensor fusion transformer. In CoRL, pp. 726–737. Cited by: §2.1.

[^43]: C. Sima, K. Renz, K. Chitta, L. Chen, H. Zhang, C. Xie, P. Luo, A. Geiger, and H. Li Drivelm: driving with graph visual question answering. In ECCV, Cited by: §1, §2.2.

[^44]: S. Song, F. Yu, A. Zeng, A. X. Chang, M. Savva, and T. Funkhouser Semantic scene completion from a single depth image. In CVPR, Cited by: §3.3.

[^45]: W. Sun, X. Lin, Y. Shi, C. Zhang, H. Wu, and S. Zheng Sparsedrive: end-to-end autonomous driving via sparse scene representation. In ICRA, Cited by: §1, §2.1.

[^46]: G. Team, R. Anil, S. Borgeaud, J. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, K. Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805. Cited by: §2.2.

[^47]: X. Tian, J. Gu, B. Li, Y. Liu, Y. Wang, Z. Zhao, K. Zhan, P. Jia, X. Lang, and H. Zhao Drivevlm: the convergence of autonomous driving and large vision-language models. In CoRL, Cited by: §2.2.

[^48]: X. Tian, T. Jiang, L. Yun, Y. Mao, H. Yang, Y. Wang, Y. Wang, and H. Zhao Occ3d: a large-scale 3d occupancy prediction benchmark for autonomous driving. NeurIPS. Cited by: §3.3.

[^49]: H. Touvron, T. Lavril, G. Izacard, X. Martinet, M. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, et al. Llama: open and efficient foundation language models. arXiv preprint arXiv:2302.13971. Cited by: §1.

[^50]: A. Van Den Oord O. Vinyals et al. Neural discrete representation learning. NeurIPS. Cited by: §3.3.

[^51]: P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu, J. Wang, W. Ge, et al. Qwen2-vl: enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191. Cited by: §1.

[^52]: S. Wang, Z. Yu, X. Jiang, S. Lan, M. Shi, N. Chang, J. Kautz, Y. Li, and J. M. Alvarez OmniDrive: a holistic vision-language dataset for autonomous driving with counterfactual reasoning. In CVPR, Cited by: §1.

[^53]: X. Weng, B. Ivanovic, Y. Wang, Y. Wang, and M. Pavone Para-drive: parallelized architecture for real-time autonomous driving. In CVPR, Cited by: Table 1.

[^54]: K. Winter, M. Azer, and F. B. Flohr BEVDriver: leveraging bev maps in llms for robust closed-loop driving. External Links: 2503.03074, [Link](https://arxiv.org/abs/2503.03074) Cited by: §2.1, Table 1.

[^55]: S. Xing, C. Qian, Y. Wang, H. Hua, K. Tian, Y. Zhou, and Z. Tu Openemma: open-source multimodal model for end-to-end autonomous driving. In WACV, Cited by: §1.

[^56]: L. Yang, B. Kang, Z. Huang, X. Xu, J. Feng, and H. Zhao Depth anything: unleashing the power of large-scale unlabeled data. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10371–10381. Cited by: §1.

[^57]: Y. Yang, J. Mei, Y. Ma, S. Du, W. Chen, Y. Qian, Y. Feng, and Y. Liu Driving in the occupancy world: vision-centric 4d occupancy forecasting and planning via world models for autonomous driving. In AAAI, Cited by: §1.

[^58]: C. Yuan, Z. Zhang, J. Sun, S. Sun, Z. Huang, C. D. W. Lee, D. Li, Y. Han, A. Wong, K. P. Tee, et al. Drama: an efficient end-to-end motion planner for autonomous driving with mamba. arXiv preprint. Cited by: Table 1.

[^59]: S. Zeng, X. Chang, M. Xie, X. Liu, Y. Bai, Z. Pan, M. Xu, and X. Wei FutureSightDrive: thinking visually with spatio-temporal CoT for autonomous driving. Cited by: §2.2.

[^60]: B. Zhang, J. Li, N. Song, and L. Zhang Perception in plan: coupled perception and planning for end-to-end autonomous driving. In AAAI, Cited by: §2.1.

[^61]: B. Zhang, N. Song, X. Jin, and L. Zhang Bridging past and future: end-to-end autonomous driving with historical prediction and planning. In CVPR, Cited by: §2.1.

[^62]: B. Zhang, N. Song, J. Li, X. Zhu, J. Deng, and L. Zhang Future-aware end-to-end driving: bidirectional modeling of trajectory planning and scene evolution. In NeurIPS, Cited by: Table 1.

[^63]: D. Zhang, G. Wang, R. Zhu, J. Zhao, X. Chen, S. Zhang, J. Gong, Q. Zhou, W. Zhang, N. Wang, et al. SparseAD: sparse query-centric paradigm for efficient end-to-end autonomous driving. arXiv preprint arXiv:2404.06892. Cited by: §2.1.

[^64]: K. Zhang, Z. Tang, X. Hu, X. Pan, X. Guo, Y. Liu, J. Huang, L. Yuan, Q. Zhang, X. Long, et al. Epona: autoregressive diffusion world model for autonomous driving. arXiv preprint arXiv:2506.24113. Cited by: Table 1.

[^65]: W. Zhang, H. Liu, Z. Qi, Y. Wang, X. Yu, J. Zhang, R. Dong, J. He, F. Lu, H. Wang, Z. Zhang, L. Yi, W. Zeng, and X. Jin DreamVLA: a vision-language-action model dreamed with comprehensive world knowledge. Cited by: §2.2.

[^66]: W. Zheng, W. Chen, Y. Huang, B. Zhang, Y. Duan, and J. Lu Occworld: learning a 3d occupancy world model for autonomous driving. In ECCV, Cited by: §1.

[^67]: X. Zhou, D. Liang, S. Tu, X. Chen, Y. Ding, D. Zhang, F. Tan, H. Zhao, and X. Bai Hermes: a unified self-driving world model for simultaneous 3d scene understanding and generation. In ICCV, Cited by: §2.2.

[^68]: X. Zhou, X. Han, F. Yang, Y. Ma, and A. C. Knoll OpenDriveVLA: towards end-to-end autonomous driving with large vision language action model. arXiv preprint arXiv:2503.23463. Cited by: §2.2.

[^69]: Z. Zhou, T. Cai, Y. Zhao, Z. Huang, B. Zhou, and J. Ma AutoVLA: a vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. NeurIPS. Cited by: Table 1, Table 1.

[^70]: D. Zhu, J. Chen, X. Shen, X. Li, and M. Elhoseiny Minigpt-4: enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592. Cited by: §2.2.

[^71]: J. Zhu, W. Wang, Z. Chen, Z. Liu, S. Ye, L. Gu, H. Tian, Y. Duan, W. Su, J. Shao, Z. Gao, E. Cui, X. Wang, Y. Cao, Y. Liu, X. Wei, H. Zhang, H. Wang, W. Xu, H. Li, J. Wang, N. Deng, S. Li, Y. He, T. Jiang, J. Luo, Y. Wang, C. He, B. Shi, X. Zhang, W. Shao, J. He, Y. Xiong, W. Qu, P. Sun, P. Jiao, H. Lv, L. Wu, K. Zhang, H. Deng, J. Ge, K. Chen, L. Wang, M. Dou, L. Lu, X. Zhu, T. Lu, D. Lin, Y. Qiao, J. Dai, and W. Wang InternVL3: exploring advanced training and test-time recipes for open-source multimodal models. External Links: 2504.10479, [Link](https://arxiv.org/abs/2504.10479) Cited by: §3.2, Table 1, §4.1.
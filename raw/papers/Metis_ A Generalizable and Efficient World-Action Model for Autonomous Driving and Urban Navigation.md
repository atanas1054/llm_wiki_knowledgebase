---
title: "Metis: A Generalizable and Efficient World-Action Model for Autonomous Driving and Urban Navigation"
source: "https://arxiv.org/html/2606.15869v1"
author:
published:
created: 2026-09-27
description:
tags:
  - "clippings"
---
Jingyu Li Affiliation: Fudan University Affiliation: Shanghai Innovation Institute    Zhe Liu <sup>1</sup> Affiliation: The University of Hong Kong    Dongnan Hu Affiliation: Shanghai Innovation Institute Affiliation: Tongji University    Junjie Wu Affiliation: Li Auto Inc.    Zipei Ma Affiliation: Fudan University Affiliation: Shanghai Innovation Institute    Wenxiao Wu Affiliation: Shanghai Innovation Institute Affiliation: Huazhong University of Science and Technology    Chao Han Affiliation: Li Auto Inc.    Zhihui Hao Affiliation: Li Auto Inc.    Zhikang Liu Affiliation: Li Auto Inc.    Kun Zhan Affiliation: Li Auto Inc.    Jiankang Deng Affiliation: Imperial College London    Xiatian Zhu Affiliation: University of Surrey [github.com/LogosRoboticsGroup/Metis](https://github.com/LogosRoboticsGroup/Metis)    Li Zhang Affiliation: Fudan University Affiliation: Shanghai Innovation Institute

###### Abstract

World action models (WAMs) have shown great promise for autonomous driving and urban navigation. Built upon Vision-Language-Action models or video generation models, existing approaches suffer key limitations: (1) High inference latency due to future observation prediction at test time, and (2) tightly coupled video and action modeling leading to representational mismatch and degraded generalization. To address both issues, we propose Metis, an end-to-end WAM framework that decouples video generation and action prediction. Specifically, Metis employs a Mixture-of-Transformers architecture with dedicated experts for video generation and action prediction, preserving the intrinsic distributional properties of each task. To enhance efficiency, we introduce an asymmetric attention mask that enables joint training of both experts while allowing the action model to bypass explicit video generation during inference. This design ensures training-inference consistency and significantly reduces computational costs without compromising planning performance. Extensive experiments demonstrate state-of-the-art performance on the NAVSIM navhard and navtest benchmarks and the CityWalker navigation benchmark, validating both the generalizability and efficiency across diverse tasks. Real-robot deployments further confirm the practical feasibility of our approach.

## 1 Introduction

Achieving safe and rational trajectory planning in Autonomous Driving (AD) and Urban Navigation (UN) requires policies that can not only react to current observations, but also anticipate how the environment evolves under agent interactions. This perspective has led to the emergence of World action models (WAMs), which integrate action generation with predictive modeling of future observations within a unified framework [^72] [^77] [^92] [^94]. By modeling future outcomes, WAMs provide a natural way to capture physical dynamics and task-relevant temporal dependencies, offering a more expressive alternative to standard Vision-Language-Action (VLA) models [^41] [^26] [^100] [^22].

Recent WAMs have been rapidly evolving, giving rise to diverse design paradigms. A natural extension of VLA-based methods toward WAMs is to augment Vision-Language Models (VLMs) [^1] [^46] [^10] [^3] with additional tokens for autoregressive future observation generation, leveraging the rich prior knowledge of VLMs while modeling future environmental dynamics, as illustrated in Figure 1 (a). Another line of approaches builds upon video generation models [^92] [^80] [^9] [^77] [^79] [^20], where intermediate representations are shared between video and action modules through tightly coupled designs, enabling joint modeling of future observations and actions, as illustrated in Figure 1 (b). Despite achieving impressive performance, existing methods still suffer from two key limitations. VLA-based WAMs require autoregressive future observations generation before action prediction during inference, leading to unavoidable computational latency. Video generation-based WAMs adopt tightly coupled designs, where high-dimensional visual representations can interfere with the low-dimensional action space, resulting in suboptimal trajectory planning.

![[intro1.png|Refer to caption]]

Figure 1: Different WAM paradigms. (a) VLA-based WAMs rely on autoregressive token-based future prediction for action planning. (b) Video generation-based WAMs use tightly coupled architectures to jointly predict future video and actions. (c) Our Metis decouples video generation from action inference via an masked asymmetric attention, enabling efficient planning without video generation.

In this paper, motivated by the observation that embodied navigation ultimately requires producing actions in the physical world, while reasoning and anticipation serve as auxiliary mechanisms to support decision-making, we propose Metis, an end-to-end framework for AD and UN. Our method seamlessly combine action generation and future observations prediction within a WAM framework, as shown in Figure 1 (c). Unlike previous methods that jointly model future observations prediction and action generation within a single model, which tightly couples the action prediction and future video generation, such frameworks may inadvertently introduce generative noise into the action space during inference, potentially compromising the precision of action predictions. In contrast, our method decouples the action model from the video generation model, enabling robust action planning without generating future observations.

Specifically, Metis adopt the Mixture-of-Transformers (MoT) architecture to integrate a video generation expert and an action prediction expert, enabling the modeling of both high-level future scene dynamics and low-level trajectory planning within a shared latent space. This design helps preserve the intrinsic distributional properties of each expert, thereby improving model generalization. To enable robust and efficient inference, we further introduce an asymmetric attention mask design: future video tokens are allowed to attend to future action tokens, while the reverse direction is masked. This unidirectional information flow enables joint optimization of the video generation and action experts during training, while avoiding explicit future observation generation during inference, and ensuring consistency between training and inference. As a result, our method achieves efficient action planning compared to existing approaches that rely on explicit future prediction.

Our contributions are: (i) We propose Metis, an end-to-end World Action Model framework that decouples the action prediction model from the video generation model, enabling each component to maintain its own distributional structure and thereby improving generalization. (ii) We further introduce an asymmetric attention mask that enforces unidirectional visibility between video and action tokens. This design enables joint training of video and action models, while eliminating the need for explicit video generation during inference and maintaining consistency between training and inference. (iii) Metis achieves state-of-the-art performance on the NAVSIM (navhard and navtest) benchmarks for AD and the CityWalker benchmarks for UN. Comprehensive ablation studies confirm the effectiveness of our design, while zero-shot real-world deployments further demonstrate the superior generalization of our approach across diverse environments.

## 2 Related work

Video generation models. Recent general video generation models [^50] [^65] [^5] [^2] have advanced rapidly and are increasingly being integrated with downstream tasks such as autonomous driving and navigation. Some works [^95] [^23] [^20] [^70] [^19] [^18] [^75] [^66] [^56] attempt to train autonomous driving video generation action models, primarily focusing on video generation while overlooking the planning objective. Some approaches [^9] [^92] [^67] [^77] leverage the generative capability of video generation models to improve autonomous driving planning. These methods typically employ diffusion transformer to jointly model and interact with visual and action information. Some end-to-end approaches [^40] [^38] [^72] [^21] [^80] [^79] [^32] further incorporate implicit world modeling and reinforcement learning to improve trajectory prediction, modeling scene evolution either in structured representations [^40] [^88] or latent space [^38] [^96] [^81]. several navigation methods [^28] [^57] [^59] [^29] learn policies from expert trajectories, but do not explicitly model future scene evolution. However, these methods tightly couple generation and action prediction, which limits generalization.

Vision-language-action models. Many VLA-based methods [^26] [^78] [^27] [^17] [^64] [^60] [^35] [^99] [^25] [^69] leverage the pretrained knowledge of VLMs to achieve strong performance in autonomous driving. Some work [^41] [^100] [^71] [^93] [^53] integrates reinforcement learning to enable safe driving via fast–slow reasoning. Furthermore, some methods [^94] [^34] [^52] [^87] [^42] [^48] [^101] extend vision-language models by incorporating structured prediction of future information. For instance, FSDrive [^87] and PWM [^94] enable future image prediction in large models, while SGDrive [^34] and DrivePI [^51] enable occupancy prediction, thereby significantly improving planning quality. In addition, VLA have also been widely explored in outdoor navigation tasks [^91] [^90] [^74] [^73] [^22] [^97] [^98] [^61] [^33]. NavFoM [^89] trains a foundation model for navigation and autonomous driving using large-scale data. Abot-N0 [^12] unifies navigation tasks in a single training framework and achieves autonomous navigation in urban environments. In contrast, rather than tightly coupling language and action, our method decouples them and leverages a world model to enhance action modeling, enabling stronger generalization across embodiments and tasks.

Worl action models. Recent works [^54] [^31] [^37] [^55] [^6] [^30] [^8] in embodied manipulation further leverage a general architecture to integrate multiple experts, including understanding, generation, and action, into a unified model. Building upon this paradigm, some approaches [^86] [^83] [^45] [^4] [^84] instantiate WAM, which incorporate world modeling, e.g., predicting future visual states, to support downstream action prediction. Overall, VLA- and WAM-based paradigms have achieved strong performance in manipulation tasks with relatively static background. However, autonomous driving and navigation involve significantly more complex and dynamic world evolution. In such settings, the WAM paradigm is better suited than VLA to model environmental dynamics and provide richer information for action planning.

## 3 Method

### 3.1 Problem formulation and notation

In autonomous driving (AD) and urban navigation (UN), action planning are typically conditioned on the current visual observation $o_{t}$ and language instructions $l$. To enhance action planning robustness, previous WAMs methods [^34] [^92] [^25] [^94] [^37] incorporate future visual observations and model them jointly with actions during training:

$$
p_{\theta}(a_{t:t+H},v_{t+1:t+N}\mid o_{t},l),
$$

where $\mathbf{a}_{t:t+H}$ denotes an action chunk with horizon $H$, and $\mathbf{v}_{t+1:t+N}$ represents predicted future observations with horizon $N$. At inference time, these methods either jointly generate actions and future observations via joint denoising,

$$
(a_{t:t+H},v_{t+1:t+N})\sim p_{\theta}(\cdot\mid o_{t},l),
$$

or adopt an inverse dynamics formulation that infers actions based on explicitly predicted future frames [^94] [^37]:

$$
v_{t+1:t+N}\sim p_{\theta}(v_{t+1:t+N}\mid o_{t},l),\quad a_{t:t+H}\sim p_{\theta}(a_{t:t+H}\mid o_{t},l,v_{t+1:t+N}).
$$

However, both paradigms require high-dimensional sampling or recursive denoising of $v_{t+1:t+N}$ during inference, leading to prohibitive computational overhead and latency.

To address these efficiency bottlenecks, we adopt a decoupled inference paradigm inspired by [^86], where future frames are used only as supervision during training, while inference relies solely on the current observation. Formally, let $z(o_{t},l)$ denote the latent representation produced by the video backbone conditioned on the current observation and context. The action prediction is then modeled as:

$$
a_{t:t+H}\sim p_{\theta}(a_{t:t+H}\mid z(o_{t},l)).
$$

In this formulation, future observations are only used during training, while $z(o_{t},l)$ is obtained from a single forward pass of the backbone at inference time, enabling efficient real-time planning.

![[pipeline1 1.png|Refer to caption]]

Figure 2: Overview of our Metis. Video and action are jointly learned during training, while inference directly predicts actions from current observation.

### 3.2 Network architecture

To avoid interference between heterogeneous task distributions, we adopt a Mixture-of-Transformers (MoT) architecture to decouple video generation and action planning into two specialized experts,as shown in Figure 2. The video generation expert (VGE) inherits the physical priors from a large-scale video model to capture spatiotemporal dynamics, while the action expert (AE) is tailored for low-dimensional trajectory prediction. Despite this decoupling, both experts interact through a shared latent space, enabling information exchange while preserving their respective distributional structures.

We adopt Wan2.2-5B [^65] as the backbone of the VGE. We reuse its pre-trained components, including a video VAE for visual encoding and a T5-based text encoder for language conditioning. To enable efficient trajectory prediction, our AE is implemented as a diffusion transformer that mirrors the layer depth of the VGE while adopting a smaller model size. In addition, to improve action policy learning, we follow prior work and introduce an agent state encoder to encode the ego state, which is combined with language instructions to condition the model. More details are provided in the Appendix.

Specifically, we organize the model inputs into three types of tokens: latent tokens of the current observation, noisy tokens of future observations, and noisy action tokens for trajectory prediction. All tokens first attend to the language embeddings through cross-attention. They are then projected into a shared latent space via expert-specific projections, where interactions are regulated by a structured attention mechanism. Finally, each expert applies its own feed-forward layers and output heads for task-specific prediction. This token-level interaction enables controlled information exchange without directly mixing representations across heterogeneous task spaces, thereby preserving the distributional structure within each expert and improving generalization.

### 3.3 Structured attention mechanisms.

Low-latency action planning is a fundamental requirement for both autonomous driving and urban navigation; however, existing WAMs [^94] [^35] [^48] [^92] often suffer from heavy inference costs, which severely limits their deployment for fast action prediction. To mitigate this latency, Some WAM approaches [^39] [^77] attempt to reduce the prediction horizon, downsample the resolution of future scene forecasting, or avoid explicit future video generation. While these strategies enhance inference speed, they inevitably lead to sub-optimal performance due to the lack of comprehensive spatiotemporal modeling of future states.

To break this efficiency-accuracy trade-off, we propose a novel asymmetric attention mask within our MoT framework. Unlike previous methods [^92] [^77], our approach manages the interaction between two experts at the representation level by introducing an asymmetric attention mask that explicitly govern the information flow between action tokens $a_{t}$ and latent video tokens $z_{t}$, as illustrated in Figure 3. Specifically, our design enforces a unilateral visibility constraint: action tokens are restricted to attend only to the current visual observation tokens, ensuring that planning is grounded in the present context.

Figure 3: Asymmetric attention mask: both action and video tokens attend to the current timestep; video tokens can attend to action tokens, but not vice versa.

In contrast, future video tokens are allowed to attend to both the current observation and all future action tokens. As shown in Figure 2 (b), this structured interaction allows the VGE to predict physically plausible world evolutions conditioned on the specific trajectory intended by the agent. Simultaneously, by sharing the representation space during the joint optimization process, the VGE’s predictive capacity implicitly participates in the action refinement process. This synergy ensures that the AE produces robust trajectories informed by environmental dynamics, while allowing the explicit video generation branch to be completely bypassed during inference to achieve real-time efficiency.

### 3.4 Training Objective

We jointly optimize action prediction and future video generation under a flow matching framework. Given current observation $o_{t}$ and instruction $l$, we learn conditional flow models over both action tokens and future video tokens. For action tokens, we supervise the prediction of the velocity field conditioned on the current context:

$$
\mathcal{L}_{\text{act}}=\mathbb{E}_{a_{t},\epsilon,t,s}\left[\|u_{\theta}(a_{t:t+H}^{(s)},s\mid o_{t},l)-\dot{a}_{t:t+H}^{(s)}\|^{2}\right],
$$

where $s\in[0,1]$ is the flow time, $a^{(s)}_{t:t+H}=(1-s)\epsilon+sa_{t:t+H}$, with $\epsilon\sim\mathcal{N}(0,I)$, and the corresponding velocity field is given by $\dot{a}_{t:t+H}=a_{t:t+H}-\epsilon$. For future video tokens, we supervise the prediction of velocity of field conditioned on the current context and action tokens:

$$
\mathcal{L}_{\text{video}}=\mathbb{E}\left[\|u_{\phi}(z^{(s)}_{t+1:t+H},s\mid o_{t},\hat{a}_{t:t+H},l)-\dot{z}^{(s)}_{t+1:t+H}\|^{2}\right],
$$

where $\hat{a}_{t:t+H}$ denotes the predicted future action sequence, $z^{(s)}_{t+1:t+H}=(1-s)\epsilon+sz_{t+1:t+H}$. The overall training objective is: $\mathcal{L}=\mathcal{L}_{\text{action}}+\lambda\mathcal{L}_{\text{video}}$ where $\lambda=1$ is a weighting coefficient that balances action learning and world modeling.

## 4 Experiments

Implementation details. We adopt Wan2.2-5B as the VGE and the AE shares the same architecture with a reduced hidden dimension ($d_{a}=1024$), resulting in a 1B-parameter branch and an overall model size of 6B. For task setup, we use an action horizon of 8 for NAVSIMv2 (4 seconds with 0.5-second intervals), where each waypoint is represented as $(x,y,\theta)$, and a horizon of 5 for CityWalker with $(x,y)$. We use only the front-facing camera as input, and align video and action chunks with a 1:1 temporal ratio. Both VGE and AE are trained under the same flow matching formulation. For NAVSIMv2, we use an input resolution of $640\times 768$ and train for 60 epochs with a batch size of 64; for CityWalker, we use $384\times 384$ and train for 30 epochs with the same batch size. During inference, we use 10 denoising steps with classifier-free guidance (CFG = 1.0). All experiments are conducted on 8 NVIDIA H200 GPUs.

Table 1: Performance on the NAVSIM-v2 navhard Leaderboard. PDM-Closed uses ground-truth symbolic inputs for planning. † indicates values are copied from [^63] and \* indicates values are copied from [^47]; all other results are reproduced with the official code repository or official checkpoints. (S.: per-stage EPDM score.)

<table><tbody><tr><td>Method</td><td>Reference</td><td>Stage</td><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DDC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TLC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>LK <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>HC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>S. <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EPDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td rowspan="2">PDM-Closed <sup><a href="#fn:14">14</a></sup></td><td rowspan="2">-</td><td>S1</td><td>94.4</td><td>78.8</td><td>100</td><td>99.5</td><td>100</td><td>93.5</td><td>99.3</td><td>87.7</td><td>36.0</td><td>-</td><td></td></tr><tr><td>S2</td><td>88.1</td><td>90.6</td><td>96.3</td><td>98.5</td><td>100</td><td>83.1</td><td>73.7</td><td>91.5</td><td>25.4</td><td>-</td><td>51.3</td></tr><tr><td colspan="14">Traditional E2E-based Planner</td></tr><tr><td rowspan="2">LTF† <sup><a href="#fn:11">11</a></sup></td><td rowspan="2">T-PAMI2022</td><td>S1</td><td>97.3</td><td>80.2</td><td>97.8</td><td>99.3</td><td>83.4</td><td>96.2</td><td>92.9</td><td>97.8</td><td>71.1</td><td>61.3</td><td></td></tr><tr><td>S2</td><td>79.4</td><td>69.0</td><td>85.6</td><td>98.5</td><td>83.8</td><td>76.7</td><td>47.9</td><td>97.0</td><td>70.6</td><td>39.2</td><td>24.4</td></tr><tr><td rowspan="2">DiffusionDrive† <sup><a href="#fn:44">44</a></sup></td><td rowspan="2">CVPR2025</td><td>S1</td><td>96.8</td><td>86.0</td><td>98.8</td><td>99.3</td><td>84.0</td><td>95.8</td><td>96.7</td><td>97.6</td><td>79.6</td><td>66.7</td><td></td></tr><tr><td>S2</td><td>80.1</td><td>72.8</td><td>84.4</td><td>98.4</td><td>85.9</td><td>76.6</td><td>46.4</td><td>96.3</td><td>72.8</td><td>40.5</td><td>27.5</td></tr><tr><td rowspan="2">GTRS-DP* <sup><a href="#fn:43">43</a></sup></td><td rowspan="2">CVPRW2025</td><td>S1</td><td>94.7</td><td>78.8</td><td>96.1</td><td>99.5</td><td>83.0</td><td>94.4</td><td>92.0</td><td>97.5</td><td>72.8</td><td>-</td><td></td></tr><tr><td>S2</td><td>80.3</td><td>74.4</td><td>84.9</td><td>98.0</td><td>81.9</td><td>78.8</td><td>45.4</td><td>96.7</td><td>70.1</td><td>-</td><td>23.8</td></tr><tr><td rowspan="2">GuideFlow* <sup><a href="#fn:47">47</a></sup></td><td rowspan="2">CVPR2026</td><td>S1</td><td>96.6</td><td>80.5</td><td>96.3</td><td>99.3</td><td>82.3</td><td>94.9</td><td>91.5</td><td>97.7</td><td>67.8</td><td>-</td><td></td></tr><tr><td>S2</td><td>87.3</td><td>76.7</td><td>88.8</td><td>99.2</td><td>84.3</td><td>85.1</td><td>49.7</td><td>93.1</td><td>44.5</td><td>-</td><td>27.1</td></tr><tr><td colspan="14">VLA-based Planner</td></tr><tr><td rowspan="2">ReCogDrive <sup><a href="#fn:41">41</a></sup></td><td rowspan="2">ICLR2026</td><td>S1</td><td>96.4</td><td>78.9</td><td>98.7</td><td>99.8</td><td>82.6</td><td>95.6</td><td>94.4</td><td>97.6</td><td>74.2</td><td>67.7</td><td></td></tr><tr><td>S2</td><td>80.2</td><td>65.0</td><td>82.4</td><td>98.7</td><td>85.2</td><td>76.9</td><td>43.8</td><td>96.6</td><td>71.8</td><td>37.6</td><td>25.7</td></tr><tr><td rowspan="2">SGDrive <sup><a href="#fn:34">34</a></sup></td><td rowspan="2">CVPR2026</td><td>S1</td><td>95.8</td><td>87.6</td><td>97.8</td><td>99.8</td><td>84.4</td><td>94.7</td><td>92.9</td><td>97.8</td><td>28.9</td><td>71.1</td><td></td></tr><tr><td>S2</td><td>79.4</td><td>65.4</td><td>79.1</td><td>98.9</td><td>88.9</td><td>75.3</td><td>42.7</td><td>96.4</td><td>29.6</td><td>35.2</td><td>25.5</td></tr><tr><td colspan="14">WAM-based Planner</td></tr><tr><td rowspan="2">Metis (Ours)</td><td rowspan="2">-</td><td>S1</td><td>96.6</td><td>87.8</td><td>99.0</td><td>99.3</td><td>84.5</td><td>95.6</td><td>97.8</td><td>97.8</td><td>77.8</td><td>75.8</td><td></td></tr><tr><td>S2</td><td>79.6</td><td>73.3</td><td>84.9</td><td>97.8</td><td>85.8</td><td>76.6</td><td>47.7</td><td>95.4</td><td>75.3</td><td>41.7</td><td>32.2</td></tr></tbody></table>

We evaluate our method on two real-world datasets covering autonomous driving and urban navigation scenarios, namely NAVSIM-v2 and CityWalk. Detailed dataset descriptions and evaluation metrics are provided in the Appendix.

NAVSIM-v2. We train our model on the navtrain subset (1,192 scenarios) and evaluate performance on two benchmarks: navhard and navtest [^7]. navhard comprises 244 safety-critical real-world scenarios (Stage 1) and 4,164 3DGS-generated synthetic counterparts (Stage 2) for closed-loop evaluation. navtest includes 12,146 scenarios for assessing generalization. The two benchmarks share a rule-based planning metric, EPDMS, which evaluates performance using multiple sub-metrics, including No-at-fault Collisions (NC), Drivable Area Compliance (DAC), Driving Direction Compliance (DDC), Traffic Light Compliance (TLC), Time-to-Collision (TTC), Ego Progress (EP), Lane Keeping (LK), History Comfort (HC), and Extended Comfort (EC). We also provide results on NAVSIM-v1 navtest with PDMS in the Appendix.

CityWalker. The dataset [^49] contains 15 hours of teleoperation data collected across diverse urban areas in New York City, with 6 hours used for fine-tuning and 9 hours for testing. We evaluate performance under several challenging scenarios, including turning, intersection crossing, detours, proximity to pedestrians, and crowded environments. We utilize Maximum Average Orientation Error (MAOE) as the primary metric to assess human-like trajectory alignment, supplemented by the average L2 distance for spatial accuracy.

Real-world experiments We deploy our model and the baseline methods on the same Unitree Go2 quadruped for real-world navigation. We adapt the PD controller provided in [^73] to provide velocity command to the quadruped. We conduct zero-shot real-world experiments in both indoor and outdoor environments, demonstrating the strong generalization ability of our Metis. Detailed qualitative results are provided in the appendix, and additional videos are included in the supplementary material.

### 4.1 Main results

Table 2: Performance comparison on NAVSIM-v2 navtest Leaderboard. \* indicates training with reinforcement learning; ${\dagger}$ indicates training on the full navtrain split; $\ddagger$ indicates the use of the best-of- $N$ ($N=6$) strategy following [^100].

<table><tbody><tr><td>Method</td><td>Sensors</td><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DDC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TLC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>LK <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>HC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EPDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>Human Agent</td><td>-</td><td>100</td><td>100</td><td>99.8</td><td>100</td><td>87.4</td><td>100</td><td>100</td><td>98.1</td><td>90.1</td><td>90.3</td></tr><tr><td colspan="12">Traditional E2E-based Planner</td></tr><tr><td>TransFuser <sup><a href="#fn:11">11</a></sup></td><td>3xC+L</td><td>96.9</td><td>89.9</td><td>97.8</td><td>99.7</td><td>87.1</td><td>95.4</td><td>92.7</td><td>98.3</td><td>87.2</td><td>76.7</td></tr><tr><td>Hydra-MDP++ <sup><a href="#fn:36">36</a></sup></td><td>3xC+L</td><td>97.2</td><td>97.5</td><td>99.4</td><td>99.6</td><td>83.1</td><td>96.5</td><td>94.4</td><td>98.2</td><td>70.9</td><td>81.4</td></tr><tr><td>GTRS-Dense <sup><a href="#fn:43">43</a></sup></td><td>3xC</td><td>97.6</td><td>97.5</td><td>99.0</td><td>99.9</td><td>87.9</td><td>97.0</td><td>95.9</td><td>97.5</td><td>55.9</td><td>82.3</td></tr><tr><td>DriveSuprim <sup><a href="#fn:82">82</a></sup></td><td>3xC</td><td>97.5</td><td>96.5</td><td>99.4</td><td>99.6</td><td>88.4</td><td>96.6</td><td>95.5</td><td>98.3</td><td>77.0</td><td>83.1</td></tr><tr><td>ARTEMIS <sup><a href="#fn:16">16</a></sup></td><td>3xC+L</td><td>98.3</td><td>95.1</td><td>98.6</td><td>99.8</td><td>81.5</td><td>97.4</td><td>96.5</td><td>98.3</td><td>-</td><td>83.1</td></tr><tr><td>DiffusionDrive <sup><a href="#fn:44">44</a></sup></td><td>3xC+L</td><td>98.2</td><td>95.9</td><td>99.4</td><td>99.8</td><td>87.5</td><td>97.3</td><td>96.8</td><td>98.3</td><td>87.7</td><td>84.5</td></tr><tr><td>World4Drive <sup><a href="#fn:96">96</a></sup></td><td>3xC</td><td>97.8</td><td>96.3</td><td>99.4</td><td>99.8</td><td>88.3</td><td>97.1</td><td>97.7</td><td>98.0</td><td>53.9</td><td>84.8</td></tr><tr><td>Drive-JEPA <sup><a href="#fn:68">68</a></sup></td><td>1xC</td><td>98.8</td><td>97.4</td><td>99.0</td><td>99.8</td><td>83.5</td><td>98.0</td><td>96.2</td><td>98.1</td><td>85.6</td><td>85.4</td></tr><tr><td>WorldRFT* <sup><a href="#fn:81">81</a></sup></td><td>3xC</td><td>97.8</td><td>96.5</td><td>99.5</td><td>99.8</td><td>88.5</td><td>97.0</td><td>97.4</td><td>98.1</td><td>69.1</td><td>86.7</td></tr><tr><td colspan="12">VLA-based Planner</td></tr><tr><td>ReCogDrive* <sup><a href="#fn:41">41</a></sup></td><td>1xC</td><td>98.3</td><td>95.2</td><td>99.5</td><td>99.8</td><td>87.1</td><td>97.5</td><td>96.6</td><td>98.3</td><td>86.5</td><td>83.6</td></tr><tr><td>SGDrive <sup><a href="#fn:34">34</a></sup></td><td>1xC</td><td>98.6</td><td>94.3</td><td>99.5</td><td>99.9</td><td>86.0</td><td>97.9</td><td>96.1</td><td>98.3</td><td>85.9</td><td>86.2</td></tr><tr><td>Vega <sup><a href="#fn:101">101</a></sup></td><td>1xC</td><td>98.9</td><td>95.3</td><td>99.4</td><td>99.9</td><td>87.0</td><td>98.4</td><td>96.1</td><td>98.3</td><td>76.3</td><td>86.9</td></tr><tr><td>DriveFine* <sup><a href="#fn:13">13</a></sup></td><td>1xC</td><td>98.7</td><td>97.3</td><td>98.8</td><td>99.8</td><td>88.2</td><td>97.8</td><td>97.7</td><td>98.4</td><td>84.7</td><td>87.1</td></tr><tr><td colspan="12">WAM-based Planner</td></tr><tr><td>Epona <sup><a href="#fn:92">92</a></sup></td><td>1xC</td><td>97.1</td><td>95.7</td><td>99.3</td><td>99.7</td><td>88.6</td><td>96.3</td><td>97.0</td><td>98.0</td><td>67.8</td><td>85.1</td></tr><tr><td>DriveVLA-W0 † <sup><a href="#fn:39">39</a></sup></td><td>1xC</td><td>98.5</td><td>99.1</td><td>98.0</td><td>99.7</td><td>86.4</td><td>98.1</td><td>93.2</td><td>97.9</td><td>58.9</td><td>86.1</td></tr><tr><td>Metis (Ours)</td><td>1xC</td><td>98.4</td><td>97.2</td><td>99.6</td><td>99.8</td><td>87.8</td><td>97.7</td><td>97.8</td><td>98.4</td><td>88.0</td><td>89.5</td></tr><tr><td>Metis <math><semantics><mo>‡</mo> <annotation>~\ddagger</annotation></semantics></math> (Ours)</td><td>1xC</td><td>98.5</td><td>97.5</td><td>99.6</td><td>99.8</td><td>87.9</td><td>97.8</td><td>98.0</td><td>98.4</td><td>90.0</td><td>90.3</td></tr></tbody></table>

Navhard Leaderboard. We first evaluate our method in safety-critical closed-loop scenarios. As shown in Table 1, our method achieves the best performance in both Stage 1 and Stage 2, consistently outperforming prior approaches. Overall, it attains an EPDMS score of 32.2, demonstrating the effectiveness of the WAM paradigm in producing smooth and comfortable driving behaviors. Notably, compared to VLA-based methods [^41] [^34], our approach achieves consistent improvements on DAC and DDC across both stages, which measure rule compliance and trajectory feasibility. In particular, it surpasses prior methods on Stage 2 by at least 7.9 on DAC and 2.5 on DDC. This indicates that, in the absence of explicit map supervision, world modeling provides a stronger inductive bias for learning environment-constrained driving behaviors. In contrast, VLA-based methods, which rely on scene understanding and high-level reasoning, are less effective at enforcing such constraints during action generation.

Table 3: Performance comparison on CityWalker dataset. Percentages indicate the proportion of data in each scenario. “Mean” denotes the average performance across the six scenarios, while “All” represents the average over all samples. The best results are highlighted in bold. \* indicates a pretrained method using large-scale navigation data.

<table><tbody><tr><td rowspan="2">Method</td><td rowspan="2">Metric</td><td rowspan="2">Mean</td><td>Turn</td><td>Crossing</td><td>Detour</td><td>Proximity</td><td>Crowd</td><td>Other</td><td>All</td></tr><tr><td>8%</td><td>12%</td><td>12%</td><td>6%</td><td>7%</td><td>55%</td><td>100%</td></tr><tr><td colspan="10">Pretrained method</td></tr><tr><td rowspan="2">ABot-N0* <sup><a href="#fn:12">12</a></sup></td><td><math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math> L2 (m)</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td><math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math> MAOE (<sup>∘</sup>)</td><td>11.2</td><td>21.3</td><td>9.8</td><td>12.8</td><td>8.1</td><td>8.8</td><td>6.3</td><td>7.6</td></tr><tr><td colspan="10">Fine-tuned method</td></tr><tr><td rowspan="2">GNM <sup><a href="#fn:58">58</a></sup></td><td><math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math> L2 (m)</td><td>1.22</td><td>2.36</td><td>1.36</td><td>1.42</td><td>0.88</td><td>0.76</td><td>0.55</td><td>0.74</td></tr><tr><td><math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math> MAOE (<sup>∘</sup>)</td><td>16.2</td><td>31.1</td><td>14.8</td><td>12.5</td><td>14.7</td><td>12.8</td><td>11.0</td><td>12.1</td></tr><tr><td rowspan="2">ViNT <sup><a href="#fn:59">59</a></sup></td><td><math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math> L2 (m)</td><td>1.30</td><td>1.91</td><td>1.13</td><td>1.14</td><td>0.77</td><td>0.66</td><td>0.57</td><td>0.70</td></tr><tr><td><math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math> MAOE (<sup>∘</sup>)</td><td>16.5</td><td>31.1</td><td>15.4</td><td>12.9</td><td>14.8</td><td>13.3</td><td>11.6</td><td>12.6</td></tr><tr><td rowspan="2">NoMaD <sup><a href="#fn:62">62</a></sup></td><td><math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math> L2 (m)</td><td>1.39</td><td>2.49</td><td>1.56</td><td>1.55</td><td>1.06</td><td>0.95</td><td>0.76</td><td>0.74</td></tr><tr><td><math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math> MAOE (<sup>∘</sup>)</td><td>19.1</td><td>35.1</td><td>18.5</td><td>15.6</td><td>18.1</td><td>14.3</td><td>12.8</td><td>12.1</td></tr><tr><td rowspan="2">CityWalker <sup><a href="#fn:49">49</a></sup></td><td><math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math> L2 (m)</td><td>1.11</td><td>1.27</td><td>1.00</td><td>1.15</td><td>1.06</td><td>1.12</td><td>1.06</td><td>1.07</td></tr><tr><td><math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math> MAOE (<sup>∘</sup>)</td><td>15.2</td><td>26.6</td><td>14.1</td><td>13.9</td><td>14.3</td><td>12.0</td><td>10.4</td><td>11.5</td></tr><tr><td rowspan="2">Metis (Ours)</td><td><math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math> L2 (m)</td><td>0.71</td><td>0.69</td><td>0.77</td><td>0.66</td><td>0.76</td><td>0.77</td><td>0.63</td><td>0.64</td></tr><tr><td><math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math> MAOE (<sup>∘</sup>)</td><td>11.8</td><td>19.5</td><td>11.9</td><td>9.1</td><td>11.4</td><td>8.0</td><td>7.8</td><td>9.8</td></tr></tbody></table>

Navtest Leaderboard. On the more diverse navtest benchmark,as shown in Table 2, Metis achieves the state-of-the-art EPDMS score under a fair comparison setting, notably without relying on multi-stage training, reinforcement learning, or auxiliary datasets. It surpasses prior VLA-based methods by at least 2.4 points, maintaining competitive or superior performance across all metrics. This consistent advantage underscores the efficacy of integrating world modeling into action planning. Specifically, compared to methods that predict future states at fixed timestamps [^101] [^34], Metis shows significant gains in EP, LK, and EC. This demonstrates that by internalizing physically grounded dynamics during training, the action expert generates more rule-compliant and comfortable behaviors. Notably, even without RL-based fine-tuning, our approach maintains a clear advantage in EC over models [^41] [^13] specifically optimized for such metrics, further proving the robustness of our WAM-based representation.

CityWalker dataset. Compared to fine-tuning-based methods,as shown in Table 3, Metisachieves leading performance in both L2 error and MAOE, validating the superiority of the WAM paradigm. When compared with ABot-N0 [^12] pretrained on large-scale datasets, our approach exhibits lower MAOE in complex scenarios such as Turn Detour and Crowd navigation. This demonstrates that integrating latent world dynamics with action planning provides more effective guidance than relying on logical reasoning within the linguistic space. In dense urban environments, the ability to model spatiotemporal evolution allows the AE to predict more plausible, collision-free trajectories, overcoming the limitations of language representations in providing fine-grained motion guidance.

![[cw_vis1.png|Refer to caption]]

Figure 4: Qualitative results on CityWalker. We present the zero-shot results of Epona and our method (Metis), as well as the fine-tuned results of our method.

Table 4: Ablation of image size and attention mask.

<table><tbody><tr><td rowspan="2">Image Size</td><td rowspan="2">variant</td><td colspan="3">navtest <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>navhard <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>DAC</td><td>LK</td><td>EPDMS</td><td>EPDMS</td></tr><tr><td rowspan="3">320 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 384</td><td>Joint</td><td>96.5</td><td>97.0</td><td>87.4</td><td>28.0</td></tr><tr><td>Isolated</td><td>96.5</td><td>97.5</td><td>88.3</td><td>29.4</td></tr><tr><td>Ours</td><td>97.0</td><td>97.6</td><td>88.8</td><td>31.6</td></tr><tr><td>640 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 768</td><td>Ours</td><td>97.5</td><td>98.0</td><td>89.5</td><td>32.2</td></tr></tbody></table>

Table 5: Ablation of denoising steps.

<table><tbody><tr><td rowspan="2">Steps</td><td>navtest</td><td colspan="3">navhard</td></tr><tr><td>EPDMS</td><td>S1</td><td>S2</td><td>EPDMS</td></tr><tr><td>1</td><td>87.2</td><td>74.5</td><td>39.8</td><td>30.4</td></tr><tr><td>2</td><td>89.2</td><td>76.0</td><td>40.7</td><td>31.2</td></tr><tr><td>5</td><td>89.4</td><td>74.5</td><td>41.4</td><td>31.4</td></tr><tr><td>10</td><td>89.5</td><td>75.8</td><td>41.7</td><td>32.2</td></tr></tbody></table>

### 4.2 Ablation studies

Ablation of asymmetric attention mask. To evaluate the effect of the proposed asymmetric attention mask, we construct two variants. The first is joint attention, which is commonly adopted in VLA-based and video-generation-based WAMs, where future video and action tokens are fully coupled. The second is isolated attention, where the two experts are completely decoupled and future video and action tokens are mutually invisible. As shown in Table 5, under the same input resolution, all variants achieve competitive performance. Our asymmetric attention mask consistently yields the best results, with more pronounced improvements on challenging scenarios. Compared to joint attention, the significant performance gain highlights the importance of decoupling future video generation from action prediction, as it helps preserve the stability of the action space. In contrast, compared to isolated attention, our method enables the action expert to implicitly capture future dynamics during training, leading to improved planning performance, particularly in complex scenarios. More detailed analyses are provided in the appendix A.

Image size. We study the impact of input image resolution on our method, as shown in Table 5. Reducing the resolution leads to consistent performance degradation on both navtest and navhard. In particular, the EPDMS score drops by 0.7 and 0.6 points on navtest and navhard, respectively. We also observe noticeable declines in DAC and LK, showing that higher resolution inputs provide richer spatial and geometric cues, which facilitate the action expert in capturing scene dynamics and producing more feasible and stable trajectories.

Denoising steps. We analyze the impact of different numbers of denoising steps on performance. As shown in Table 12, increasing the number of steps consistently improves performance on both navtest and navhard. While the performance gain is relatively modest in general scenarios, it becomes more pronounced on challenging benchmarks, indicating that additional denoising steps help refine predictions under complex dynamics.

Ablation of different VGE and AE. The impact of different VGE backbones is investigated at a $640\times 768$ resolution, as shown in Table 6. Results indicate that although the Wan2.1-1.3B [^65] variant achieves comparable performance on navtest, its deficiency in navhard reveals the limitations of lower-capacity generative priors in complex scenarios. We hypothesize that a more powerful VGE serves as a more robust world model, implicitly bolstering the AE’s reasoning via shared latent representations. Simultaneously, increasing the AE size under a fixed VGE further improves generalization across all benchmarks, underscoring the benefits of scaling the policy head alongside the generative backbone.

Inference latency of different methods. We compare the inference latency of our Metis with several representative approaches, as shown in Table 7. Epona and PWM represent WAM methods based on video generation models and large vision-language models, respectively. All methods are evaluated on a single NVIDIA RTX 4090 GPU. The configurations are determined by practical requirements. The video generation setting uses 10 denoising steps to ensure high-quality future prediction, while the action only setting uses only 2 denoising steps for efficient action inference while maintaining accuracy. We also report PDMS and EPDMS on navtest (full results are provided in the Appendix). Compared to the variant that generates future video, our action-only inference achieves up to $8\times$ speedup, demonstrating the efficiency of our approach.

![[navsim_vis1.png|Refer to caption]]

Figure 5: Qualitative evaluation of turning scenarios on NAVSIM.

<table><tbody><tr><td rowspan="2">VGE</td><td rowspan="2">AE</td><td colspan="2">navtest <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>navhard <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>PDMS</td><td>EPDMS</td><td>EPDMS</td></tr><tr><td>Wan2.1-1.3B</td><td><math><semantics><mo>∼</mo> <annotation>\sim</annotation></semantics></math> 0.24B</td><td>88.5</td><td>88.8</td><td>28.8</td></tr><tr><td>Wan2.2-14B</td><td><math><semantics><mo>∼</mo> <annotation>\sim</annotation></semantics></math> 0.21B</td><td>88.2</td><td>88.2</td><td>31.2</td></tr><tr><td>Wan2.2-14B</td><td><math><semantics><mo>∼</mo> <annotation>\sim</annotation></semantics></math> 1.04B</td><td>89.1</td><td>89.5</td><td>32.2</td></tr></tbody></table>

Table 6: Ablation of different VGE and AE.

| Method | PDMS | EPDMS | latency(s) |
| --- | --- | --- | --- |
| Epona [^92] | 86.2 | 85.1 | 0.32 |
| PWM [^94] | 87.3 | \- | 0.57 |
| PWM (w/ video) [^94] | 88.1 | \- | 0.83 |
| Metis (w/ video) (ours) | 89.0 | 89.5 | 1.38 |
| Metis (ours) | 88.9 | 89.2 | 0.17 |

Table 7: Inference latency with other methods.

### 4.3 Qualitative Analysis

We provide qualitative results on both UN and AD tasks. For the UN task, as shown in Figure 4, our method demonstrates stronger generalization compared to Epona, achieving better zero-shot performance. After fine-tuning, it produces more accurate and stable navigation trajectories. For the AD task, compared with the previous state-of-the-art method ReCogDrive [^41], our approach exhibits more reasonable and smoother behaviors, particularly in turning scenarios, as shown in Figure 5. The predicted trajectories better align with the underlying scene geometry and driving constraints. These results validate that our method effectively improves trajectory planning by enabling the action expert to implicitly capture scene dynamics during training.

## 5 Conclusion

In this paper, we propose Metis, an end-to-end WAM framework for AD and UN built upon a Mixture-of-Transformers architecture. Distinct from prior WAMs, Metis introduces an asymmetric attention mask mechanism that enables a unique "joint training, decoupled inference" paradigm. While we jointly optimize future video generation and action prediction during training to capture world dynamics, our design allows for efficient action planning without the need for explicit future observation synthesis during inference. By decoupling the low-dimensional action space from high-dimensional visual generation, we preserve the distributional integrity and stability of the action model, significantly enhancing both generalization and inference efficiency. Extensive evaluations on autonomous driving and urban navigation benchmarks demonstrate that our approach achieves state-of-the-art performance, validating the effectiveness of our decoupled WAM design.

## References

## Appendix

## Contents

## Appendix A Discussion

### A.1 Motivation

In this study, we explore the relationship between video generation and action prediction within World action models (WAMs) for autonomous driving and urban navigation. Drawing inspiration from human cognitive processes as discussed in our introduction, we propose a framework characterized by joint training and decoupled inference. During the training phase, a loose coupling mechanism—implemented via a simple asymmetric attention module—ensures that the video generation expert (VGE) generate a future consistent with the ego-actions. This connection allows the gradients from the generation expert to backpropagate into the action expert (AE), implicitly optimizing the latter and ensuring a stable distribution within the action space. In contrast, during inference, the generation process is entirely bypassed; the model performs efficient decision-making by leveraging only the current observations for contextual information, thereby eliminating the need for explicit future world generation.

### A.2 Discussion about different attention masks

Based on the experimental results presented in the Table 8, our asymmetric attention mask achieves superior performance across all metrics under identical input conditions.

While common joint training methods (represented by Joint) exhibit strong overall performance relative to baseline approaches [^94] [^77] [^48] [^92], they lag significantly in DAC and EP metrics. This performance gap stems from the fact that during inference, the action space in these models is heavily injected with noise from the generation process, which interferes with the integrity of action prediction. As illustrated in Table 2, our method maintains an absolute lead in EP and DAC among WAM-based approaches, achieving a 2.7% improvement even over DriveLaW [^77], the highest-rated model overall.

Regarding the Isolated attention mask, the complete separation of tasks prevents the action space from effectively utilizing the visual context provided by current observations. Although its DAC and EP scores are comparable to ours, its lower overall rating underscores the necessity of implicitly optimizing the action expert through the generation expert during training. This balance ensures that the model leverages sophisticated world-understanding without suffering from inference-time generation noise.

Table 8: Detailed discussion on attention mask variants.

<table><tbody><tr><td rowspan="2">Image Size</td><td rowspan="2">Variant</td><td colspan="3">navtest <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td colspan="3">navtest <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>navhard <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>DAC</td><td>EP</td><td>PDMS</td><td>DAC</td><td>EP</td><td>EPDMS</td><td>EPDMS</td></tr><tr><td rowspan="3">320 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 384</td><td>Joint</td><td>95.7</td><td>81.6</td><td>87.1</td><td>96.5</td><td>87.6</td><td>87.4</td><td>28.0</td></tr><tr><td>Isolated</td><td>96.5</td><td>82.6</td><td>88.0</td><td>96.9</td><td>87.7</td><td>88.3</td><td>29.4</td></tr><tr><td>Ours</td><td>96.9</td><td>82.9</td><td>88.3</td><td>97.0</td><td>87.8</td><td>88.8</td><td>31.6</td></tr></tbody></table>

## Appendix B Real world experiments

We evaluate our method in closed-loop obstacle-avoidance tests under both indoor daytime and outdoor nighttime scenarios. The following four examples, as illustrated in Figure 7 and Figure 6 the obstacle avoidance and generalization potential of our model. In the scenarios shown above, the quadruped robot is observed to plan appropriate paths, reflecting its capability to handle various environments.

![[real_world_outdoor_case_1.png|Refer to caption]]

Figure 6: Qualitative results of real-world outdoor deployment under a zero-shot setting (without any task-specific training). Sensitive biometric information (e.g., human faces) has been anonymized.

![[real_world_indoor_case_1.png|Refer to caption]]

Figure 7: Qualitative results of real-world outdoor deployment under a zero-shot setting (without any task-specific training)

## Appendix C Additional experiments results

Table 9: Performance comparison on NAVSIM-v1 navtest Leaderboard. \* indicates training with reinforcement learning; -IL means imitation learning; ${\dagger}$ indicates training on the full navtrain split; $\ddagger$ indicates the use of the best-of- $N$ ($N=6$) strategy following [^100].

<table><tbody><tr><td>Method</td><td>Sensors</td><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>C <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>Human Agent</td><td>-</td><td>100</td><td>100</td><td>87.5</td><td>100</td><td>99.9</td><td>94.8</td></tr><tr><td colspan="8">Traditional E2E-based Planner</td></tr><tr><td>UniAD <sup><a href="#fn:24">24</a></sup></td><td>6xC</td><td>97.8</td><td>91.9</td><td>78.8</td><td>92.9</td><td>100.0</td><td>83.4</td></tr><tr><td>TransFuser <sup><a href="#fn:11">11</a></sup></td><td>3xC+L</td><td>97.7</td><td>92.8</td><td>79.2</td><td>92.8</td><td>100</td><td>84.0</td></tr><tr><td>PARA-Drive <sup><a href="#fn:76">76</a></sup></td><td>6xC</td><td>97.9</td><td>92.4</td><td>79.3</td><td>93.0</td><td>99.8</td><td>84.0</td></tr><tr><td>LAW <sup><a href="#fn:38">38</a></sup></td><td>1xC</td><td>96.4</td><td>95.4</td><td>81.7</td><td>88.7</td><td>99.9</td><td>84.6</td></tr><tr><td>World4Drive <sup><a href="#fn:96">96</a></sup></td><td>3xC</td><td>97.4</td><td>94.3</td><td>79.9</td><td>92.8</td><td>100.0</td><td>85.1</td></tr><tr><td>DRAMA <sup><a href="#fn:85">85</a></sup></td><td>3xC+L</td><td>98.0</td><td>93.1</td><td>80.1</td><td>94.8</td><td>100.0</td><td>85.5</td></tr><tr><td>Hydra-MDP++ <sup><a href="#fn:36">36</a></sup></td><td>3xC+L</td><td>97.6</td><td>96.0</td><td>80.4</td><td>93.1</td><td>100.0</td><td>86.6</td></tr><tr><td>ARTEMIS <sup><a href="#fn:16">16</a></sup></td><td>3xC+L</td><td>98.3</td><td>95.1</td><td>81.4</td><td>94.3</td><td>100.0</td><td>87.0</td></tr><tr><td>WorldRFT* <sup><a href="#fn:81">81</a></sup></td><td>3xC</td><td>97.5</td><td>96.0</td><td>80.9</td><td>94.0</td><td>100.0</td><td>87.0</td></tr><tr><td>DiffusionDrive <sup><a href="#fn:44">44</a></sup></td><td>3xC+L</td><td>98.2</td><td>96.2</td><td>82.2</td><td>94.7</td><td>100.0</td><td>88.1</td></tr><tr><td>WorldDrive <sup><a href="#fn:21">21</a></sup></td><td>1xC</td><td>98.4</td><td>96.2</td><td>81.9</td><td>95.1</td><td>100</td><td>88.1</td></tr><tr><td>WoTE <sup><a href="#fn:40">40</a></sup></td><td>3xC+L</td><td>98.5</td><td>96.8</td><td>81.9</td><td>94.9</td><td>99.9</td><td>88.3</td></tr><tr><td>SeerDrive <sup><a href="#fn:88">88</a></sup></td><td>3/6xC+L</td><td>98.4</td><td>97.0</td><td>83.2</td><td>94.9</td><td>99.9</td><td>88.9</td></tr><tr><td>Drive-JEPA <sup><a href="#fn:68">68</a></sup></td><td>1xC</td><td>98.7</td><td>96.2</td><td>82.9</td><td>100.0</td><td>95.5</td><td>89.0</td></tr><tr><td colspan="8">VLA-based Planner</td></tr><tr><td>AutoVLA-IL <sup><a href="#fn:100">100</a></sup></td><td>3xC</td><td>96.9</td><td>92.4</td><td>75.8</td><td>88.1</td><td>99.1</td><td>80.5</td></tr><tr><td>ReCogDrive-IL <sup><a href="#fn:41">41</a></sup></td><td>1xC</td><td>98.1</td><td>94.7</td><td>80.9</td><td>94.2</td><td>100.0</td><td>86.5</td></tr><tr><td>SGDrive-IL <sup><a href="#fn:34">34</a></sup></td><td>1xC</td><td>98.6</td><td>95.1</td><td>81.2</td><td>95.4</td><td>100.0</td><td>87.4</td></tr><tr><td>Vega <sup><a href="#fn:101">101</a></sup></td><td>1xC</td><td>98.9</td><td>95.3</td><td>81.6</td><td>96.1</td><td>100.0</td><td>87.9</td></tr><tr><td colspan="8">WAM-based Planner</td></tr><tr><td>Epona <sup><a href="#fn:92">92</a></sup></td><td>1xC</td><td>97.9</td><td>95.1</td><td>80.4</td><td>93.8</td><td>99.9</td><td>86.2</td></tr><tr><td>ImagiDrive <sup><a href="#fn:35">35</a></sup></td><td>1xC</td><td>98.6</td><td>96.2</td><td>80.5</td><td>94.5</td><td>100.0</td><td>87.4</td></tr><tr><td>PWM <math><semantics><mo>†</mo> <annotation>\dagger</annotation></semantics></math> <sup><a href="#fn:94">94</a></sup></td><td>1xC</td><td>98.6</td><td>95.9</td><td>81.8</td><td>95.4</td><td>100.0</td><td>88.1</td></tr><tr><td>DriveVLA-W0 <math><semantics><mo>†</mo> <annotation>{\dagger}</annotation></semantics></math> <sup><a href="#fn:39">39</a></sup></td><td>1xC</td><td>98.7</td><td>96.2</td><td>82.2</td><td>95.5</td><td>100.0</td><td>88.4</td></tr><tr><td>UniWorldVLA <math><semantics><mo>†</mo> <annotation>\dagger</annotation></semantics></math> <sup><a href="#fn:48">48</a></sup></td><td>1xC</td><td>98.7</td><td>96.7</td><td>83.2</td><td>96.1</td><td>100.0</td><td>89.4</td></tr><tr><td>DriveLAW <math><semantics><mo>†</mo> <annotation>\dagger</annotation></semantics></math> <sup><a href="#fn:77">77</a></sup></td><td>1xC</td><td>99.0</td><td>97.1</td><td>81.3</td><td>96.7</td><td>100.0</td><td>89.1</td></tr><tr><td>Metis (Ours)</td><td>1xC</td><td>98.3</td><td>97.1</td><td>83.4</td><td>94.7</td><td>100.0</td><td>89.1</td></tr><tr><td>Metis <math><semantics><mo>‡</mo> <annotation>\ddagger</annotation></semantics></math> (Ours)</td><td>1xC</td><td>98.5</td><td>97.5</td><td>84.0</td><td>95.1</td><td>100.0</td><td>89.7</td></tr></tbody></table>

### C.1 NAVSIM-v1 navtest Leaderboard.

As shown in Table 9, under a fair comparison on NAVSIM-v1 NavTest, our method achieves state-of-the-art performance, reaching an overall PDMS of 89.7. It also significantly outperforms prior methods on both DAC and EP metrics. Notably, compared to previous WAM-based approaches, our method attains superior performance without using the full NavTrain dataset for training. This demonstrates that our paradigm of decoupling future image generation and action prediction effectively preserves the stability of the action space distribution, leading to smoother and more reliable driving behaviors.

## Appendix D Ablation studies

### D.1 Ablation of action expert capacity.

We study the effect of the action expert capacity on planning performance. All variants share the same architecture as the video generation expert in terms of DiT depth, while varying the model size. As shown in Table 10, increasing the capacity of the action expert consistently improves planning performance. The gains are particularly pronounced on challenging settings such as navhard S2, indicating that a larger action expert is beneficial for handling complex dynamics.

### D.2 Ablation of co-training between video generation and action prediction.

While the action expert alone already achieves strong performance, jointly training with video generation further improves both PDMS and EPDMS by a clear margin. This validates that the proposed asymmetric attention mask enables the action expert to implicitly learn dynamic world evolution from the video generation branch during training, while remaining decoupled at inference time, thereby improving generalization.

<table><tbody><tr><td rowspan="2">Params</td><td>navtest</td><td colspan="3">navhard</td></tr><tr><td>EPDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>S1 <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>S2 <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EPDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td><math><semantics><mo>∼</mo> <annotation>\sim</annotation></semantics></math> 0.21 B</td><td>88.2</td><td>76.0</td><td>40.7</td><td>31.2</td></tr><tr><td><math><semantics><mo>∼</mo> <annotation>\sim</annotation></semantics></math> 0.45 B</td><td>88.6</td><td>74.5</td><td>41.4</td><td>31.4</td></tr><tr><td><math><semantics><mo>∼</mo> <annotation>\sim</annotation></semantics></math> 1.04 B</td><td>89.5</td><td>75.8</td><td>41.7</td><td>32.2</td></tr></tbody></table>

Table 10: Ablation of action expert params.

| Training Paradigm | PDMS | EPDMS |
| --- | --- | --- |
| w/o co-train | 87.4 | 87.9 |
| w/ co-train | 89.1 | 89.5 |

Table 11: Ablation of training video generation expert.

Table 12: Performance and inference latency across different hardware and denoising steps.

<table><tbody><tr><td rowspan="2">Steps</td><td colspan="2">navtest <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>navhard <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td colspan="2">Latency (ms) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><td>PDMS</td><td>EPDMS</td><td>EPDMS</td><td>RTX 4090</td><td>H200</td></tr><tr><td>1</td><td>87.0</td><td>87.2</td><td>30.4</td><td>110</td><td>100</td></tr><tr><td>2</td><td>88.9</td><td>89.2</td><td>31.2</td><td>147</td><td>140</td></tr><tr><td>5</td><td>89.0</td><td>89.4</td><td>31.4</td><td>280</td><td>240</td></tr><tr><td>10</td><td>89.1</td><td>89.5</td><td>32.2</td><td>480</td><td>430</td></tr></tbody></table>

## Appendix E Qualitative results

We provide visualization results of the generation expert here, along with additional qualitative results on NAVSIM and CityWalker.

![[video_gen_vis_1.png|Refer to caption]]

Figure 8: Qualitative comparison of video generation results. For each pair, the top row shows the frames generated by our model, and the bottom row shows the ground truth sequences.

### E.1 Qualitative of video generation expert

As shown in Figure 8, we present visualization results of the generation expert in both static and dynamic scenarios. The model produces high-quality results in relatively static scenes. In contrast, under complex dynamic intersection scenarios, fine details of distant background vehicles may be lost. Nevertheless, this degradation does not noticeably affect short-term driving action prediction.

![[supp_navsim_vis1.png|Refer to caption]]

Figure 9: Qualitative results of Metisin NAVSIM 15.

![[supp_navsim_fail_vis.png|Refer to caption]]

Figure 10: Qualitative failure results of Metisin NAVSIM 15.

#### E.1.1 More qualitative results and failure case on NAVSIM

Our method is capable of producing reasonable planning behaviors in scenarios such as lane following and turning, as illustrated in Figure 9. In open-world scenarios, our method may exhibit slight deviations during long-horizon planning due to the limitation of using only monocular-view inputs, as illustrated in Figure 10. We leave efficient multi-view world modeling for future work to further facilitate action planning.

## Appendix F Experiment details and evaluation metric

We adopt Wan2.2-5B as the VGE and the AE shares the same architecture with a reduced hidden dimension ($d_{a}=1024$), resulting in a 1B-parameter branch and an overall model size of 6B. For task setup, we use an action horizon of 8 for NAVSIMv2 (4 seconds with 0.5-second intervals), where each waypoint is represented as $(x,y,\theta)$, and a horizon of 5 for CityWalker with $(x,y)$. We use only the front-facing camera as input, and align video and action chunks with a 1:1 temporal ratio.

Both VGE and AE are trained under the same flow matching formulation. For NAVSIMv2, we use an input resolution of $640\times 768$ and train for 60 epochs with a batch size of 64; for CityWalker, we use $384\times 384$ and train for 30 epochs with the same batch size. During inference, we use 10 denoising steps with classifier-free guidance (CFG = 1.0). All experiments are conducted on 8 NVIDIA H200 GPUs (140 GB memory each). We optimize all models using AdamW ($lr=1\times 10^{-4}$, weight decay 0.01) with a cosine annealing schedule, and apply mixed precision training with gradient clipping at 1.0.

We utilize two benchmarks in NAVSIMv2 [^7] for end-to-end model evaluation, including navhard and navtest. navhard is the official two-stage evaluation benchmark, which contains 244 challenging real-world scenarios in the first stage and corresponding 4,164 synthetic scenarios generated by 3DGS in the second stage. navtest is a one-stage evaluation benchmark, containing a large number of 12,146 real-world scenarios. navhard focuses on assessing the model’s closed-loop performance in safety-critical situations, while navtest emphasizes generalization across diverse driving conditions. The two benchmarks share a rule-based planning metric, $\mathrm{EPDMS}$ [^36], with several sub-metrics:

$$
\mathrm{EPDMS}=\underbrace{\left(\prod_{m\in\mathcal{M}_{\text{pen}}}S_{m}\right)}_{\text{penalties}}\cdot\underbrace{\left(\frac{\sum_{m\in\mathcal{M}_{\text{avg}}}w_{m}S_{m}}{\sum_{m\in\mathcal{M}_{\text{avg}}}w_{m}}\right)}_{\text{weighted average}},
$$

where $S_{m}$ is the sub-metric: penalty terms set $\mathcal{M}_{\text{pen}}$ includes No-at-fault Collisions (NC), Drivable Area Compliance (DAC), Driving Direction Compliance (DDC), and Traffic Light Compliance (TLC); weighted average terms set $\mathcal{M}_{\text{avg}}$ includes Time-to-Collision (TTC), Ego Progress (EP), Lane Keeping (LK), History Comfort (HC), and extended comfort (EC). Note that $\mathrm{EPDMS}$ in navhard further incorporates several modifications, two-stage aggregation, reactive traffic simulation, and the exclusion of penalties in cases where the human expert driver also fails.

We also provide one benchmarks in NAVSIMv1 [^15] for end-to-end model evaluation on navtest with PDMS:

$$
\text{PDMS}=\underbrace{\left(\prod_{m\in\{\text{NC},\text{DAC}\}}\text{score}_{m}\right)}_{\text{penalties}}\times\underbrace{\left(\frac{\sum_{w\in\{\text{EP},\text{TTC},\text{C}\}}\text{weight}_{w}\times\text{score}_{w}}{\sum_{w\in\{\text{EP},\text{TTC},\text{C}\}}\text{weight}_{w}}\right)}_{\text{weighted average}}.
$$

## Appendix G Limitation

Despite its performance, our method has limitations. First, its reliability in extreme corner cases remains to be fully validated due to the inherent data-dependent nature of world modeling. Second, the framework relies heavily on the pre-trained Video Generation Expert; while video generation is bypassed during inference, its involvement in training remains computationally intensive. Additionally, the VAE’s down-sampling ratio significantly impacts the balance between training efficiency and representation quality. Future work will focus on addressing these challenges to enhance robustness and training scalability.

## Appendix H Broad Impact

This research presents a dual-sided impact on the development of autonomous systems. On the positive side, by enabling agents to proactively reason about future environment evolution through world modeling, our framework significantly enhances the safety and decision-making efficiency of autonomous navigation, which may lead to a reduction in traffic accidents and energy consumption in smart cities. Conversely, the deployment of such technology entails potential risks: the model may exhibit unpredictable behavior in out-of-distribution (OOD) scenarios or extreme corner cases. Furthermore, its widespread adoption could disrupt the labor market for professional drivers and poses new challenges for legal frameworks regarding accountability in autonomous decision-making.

[^1]: J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat, et al. (2023) Gpt-4 technical report. arXiv preprint arXiv:2303.08774. Cited by: §1.

[^2]: A. Ali, J. Bai, M. Bala, Y. Balaji, A. Blakeman, T. Cai, J. Cao, T. Cao, E. Cha, Y. Chao, et al. (2025) World simulation with video foundation models for physical ai. arXiv preprint arXiv:2511.00062. Cited by: §2.

[^3]: S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang, H. Zhong, Y. Zhu, M. Yang, Z. Li, J. Wan, P. Wang, W. Ding, Z. Fu, Y. Xu, J. Ye, X. Zhang, T. Xie, Z. Cheng, H. Zhang, Z. Yang, H. Xu, and J. Lin (2025) Qwen2.5-vl technical report. External Links: 2502.13923, [Link](https://arxiv.org/abs/2502.13923) Cited by: §1.

[^4]: H. Bi, H. Tan, S. Xie, Z. Wang, S. Huang, H. Liu, R. Zhao, Y. Feng, C. Xiang, Y. Rong, et al. (2025) Motus: a unified latent action world model. arXiv preprint arXiv:2512.13030. Cited by: §2.

[^5]: J. Bruce, M. D. Dennis, A. Edwards, J. Parker-Holder, Y. Shi, E. Hughes, M. Lai, A. Mavalankar, R. Steigerwald, C. Apps, et al. (2024) Genie: generative interactive environments. In ICML, Cited by: §2.

[^6]: J. Cai, Z. Cai, J. Cao, Y. Chen, Z. He, L. Jiang, H. Li, H. Li, Y. Li, Y. Liu, et al. (2026) InternVLA-a1: unifying understanding, generation and action for robotic manipulation. arXiv preprint arXiv:2601.02456. Cited by: §2.

[^7]: W. Cao, M. Hallgarten, T. Li, D. Dauner, X. Gu, C. Wang, Y. Miron, M. Aiello, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, A. Geiger, and K. Chitta (2025) Pseudo-simulation for autonomous driving. In CoRL, Cited by: Appendix F, §4.

[^8]: J. Cen, C. Yu, H. Yuan, Y. Jiang, S. Huang, J. Guo, X. Li, Y. Song, H. Luo, F. Wang, et al. (2025) Worldvla: towards autoregressive action world model. arXiv preprint arXiv:2506.21539. Cited by: §2.

[^9]: Y. Chen, Y. Wang, and Z. Zhang (2024) Drivinggpt: unifying driving world modeling and planning with multi-modal autoregressive transformers. arXiv preprint. Cited by: §1, §2.

[^10]: Z. Chen, J. Wu, W. Wang, W. Su, G. Chen, S. Xing, M. Zhong, Q. Zhang, X. Zhu, L. Lu, et al. (2024) Internvl: scaling up vision foundation models and aligning for generic visual-linguistic tasks. In CVPR, Cited by: §1.

[^11]: K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger (2022) Transfuser: imitation with transformer-based sensor fusion for autonomous driving. IEEE TPAMI. Cited by: Table 9, Table 1, Table 2.

[^12]: Z. Chu, S. Xie, X. Wu, Y. Shen, M. Luo, Z. Wang, F. Liu, X. Leng, J. Hu, M. Yin, et al. (2026) ABot-n0: technical report on the vla foundation model for versatile embodied navigation. arXiv preprint arXiv:2602.11598. Cited by: §2, §4.1, Table 3.

[^13]: C. Dang, S. Ang, Y. Li, H. Tian, J. Wang, G. Li, H. Ye, J. Ma, L. Chen, and Y. Wang (2026) DriveFine: refining-augmented masked diffusion vla for precise and robust driving. arXiv preprint arXiv:2602.14577. Cited by: §4.1, Table 2.

[^14]: D. Dauner, M. Hallgarten, A. Geiger, and K. Chitta (2023) Parting with misconceptions about learning-based vehicle motion planning. In CoRL, Cited by: Table 1.

[^15]: D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, A. Geiger, and K. Chitta (2024) NAVSIM: data-driven non-reactive autonomous vehicle simulation and benchmarking. In Advances in Neural Information Processing Systems (NeurIPS), Cited by: Figure 10, Figure 10, Figure 9, Figure 9, Appendix F.

[^16]: R. Feng, N. Xi, D. Chu, R. Wang, Z. Deng, A. Wang, L. Lu, J. Wang, and Y. Huang (2025) Artemis: autoregressive end-to-end trajectory planning with mixture of experts for autonomous driving. IEEE RA-L. Cited by: Table 9, Table 2.

[^17]: H. Fu, D. Zhang, Z. Zhao, J. Cui, D. Liang, C. Zhang, D. Zhang, H. Xie, B. Wang, and X. Bai (2025) ORION: a holistic end-to-end autonomous driving framework by vision-language instructed action generation. In ICCV, Cited by: §2.

[^18]: Y. Fu, Y. Li, and X. Di (2024) Gendds: generating diverse driving video scenarios with prompt-to-video generative model. In IEEE ITSC, Cited by: §2.

[^19]: R. Gao, K. Chen, B. Xiao, L. Hong, Z. Li, and Q. Xu (2025) MagicDrive-v2: high-resolution long video generation for autonomous driving with adaptive control. In CVPR, Cited by: §2.

[^20]: S. Gao, J. Yang, L. Chen, K. Chitta, Y. Qiu, A. Geiger, J. Zhang, and H. Li (2024) Vista: a generalizable driving world model with high fidelity and versatile controllability. In NeurIPS, Cited by: §1, §2.

[^21]: X. Gui, M. Zhang, T. Yan, W. Han, J. Gong, F. Tan, C. Xu, and J. Shen (2026) Bridging scene generation and planning: driving with world model via unifying vision and motion representation. arXiv preprint arXiv:2603.14948. Cited by: Table 9, §2.

[^22]: N. Hirose, C. Glossop, D. Shah, and S. Levine (2026) OmniVLA: an omni-modal vision-language-action model for robot navigation. In ICRA, Cited by: §1, §2.

[^23]: X. Hu, W. Yin, M. Jia, J. Deng, X. Guo, Q. Zhang, X. Long, and P. Tan (2024) DrivingWorld: constructing world model for autonomous driving via video gpt. arXiv preprint arXiv:2412.19505. Cited by: §2.

[^24]: Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, et al. (2023) Planning-oriented autonomous driving. In CVPR, Cited by: Table 9.

[^25]: W. Huang, S. Zhang, Q. Huang, Z. Wang, Z. Mao, C. Chua, Z. Chen, L. Chen, and C. Lv (2026) Automot: a unified vision-language-action model with asynchronous mixture-of-transformers for end-to-end autonomous driving. arXiv preprint arXiv:2603.14851. Cited by: §2, §3.1.

[^26]: J. Hwang, R. Xu, H. Lin, W. Hung, J. Ji, K. Choi, D. Huang, T. He, P. Covington, B. Sapp, et al. (2024) Emma: end-to-end multimodal model for autonomous driving. arXiv preprint arXiv:2410.23262. Cited by: §1, §2.

[^27]: B. Jiang, S. Chen, B. Liao, X. Zhang, W. Yin, Q. Zhang, C. Huang, W. Liu, and X. Wang (2024) Senna: bridging large vision-language models and end-to-end autonomous driving. arXiv preprint arXiv:2410.22313. Cited by: §2.

[^28]: G. Kahn, P. Abbeel, and S. Levine (2021) Badgr: an autonomous self-supervised learning-based navigation system. RAL. Cited by: §2.

[^29]: G. Kahn, P. Abbeel, and S. Levine (2021) Land: learning to navigate from disengagements. RAL. Cited by: §2.

[^30]: M. J. Kim, C. Finn, and P. Liang (2025) Fine-tuning vision-language-action models: optimizing speed and success. arXiv preprint arXiv:2502.19645. Cited by: §2.

[^31]: M. J. Kim, Y. Gao, T. Lin, Y. Lin, Y. Ge, G. Lam, P. Liang, S. Song, M. Liu, C. Finn, et al. (2026) Cosmos policy: fine-tuning video models for visuomotor control and planning. arXiv preprint arXiv:2601.16163. Cited by: §2.

[^32]: E. Kirby, A. Boulch, Y. Xu, Y. Yin, G. Puy, É. Zablocki, A. Bursuc, S. Gidaris, R. Marlet, F. Bartoccioni, A. Cao, N. Samet, T. Vu, and M. Cord (2026) Driving on registers. In CVPR, Cited by: §2.

[^33]: J. Li, Z. Liu, W. Wu, and L. Zhang (2026) MCNav: memory-aware dynamic cognitive map for zero-shot goal-oriented navigation. arXiv preprint arXiv:2605.19594. Cited by: §2.

[^34]: J. Li, J. Wu, D. Hu, X. Huang, B. Sun, Z. Hao, X. Lang, X. Zhu, and L. Zhang (2026) SGDrive: scene-to-goal hierarchical world cognition for autonomous driving. arXiv preprint arXiv:2601.05640. Cited by: Table 9, §2, §3.1, §4.1, §4.1, Table 1, Table 2.

[^35]: J. Li, B. Zhang, X. Jin, J. Deng, X. Zhu, and L. Zhang (2025) ImagiDrive: a unified imagination-and-planning framework for autonomous driving. arXiv preprint arXiv:2508.11428. Cited by: Table 9, §2, §3.3.

[^36]: K. Li, Z. Li, S. Lan, Y. Xie, Z. Zhang, J. Liu, Z. Wu, Z. Yu, and J. M. Alvarez (2025) Hydra-mdp++: advancing end-to-end driving via expert-guided hydra-distillation. arXiv preprint arXiv:2503.12820. Cited by: Table 9, Appendix F, Table 2.

[^37]: L. Li, Q. Zhang, Y. Luo, S. Yang, R. Wang, F. Han, M. Yu, Z. Gao, N. Xue, X. Zhu, Y. Shen, and Y. Xu (2026) Causal world modeling for robot control. arXiv preprint arXiv:2601.21998. Cited by: §2, §3.1, §3.1.

[^38]: Y. Li, L. Fan, J. He, Y. Wang, Y. Chen, Z. Zhang, and T. Tan (2025) Enhancing end-to-end autonomous driving with latent world model. In ICLR, Cited by: Table 9, §2.

[^39]: Y. Li, S. Shang, W. Liu, B. Zhan, H. Wang, Y. Wang, Y. Chen, X. Wang, Y. An, C. Tang, et al. (2025) DriveVLA-w0: world models amplify data scaling law in autonomous driving. arXiv preprint arXiv:2510.12796. Cited by: Table 9, §3.3, Table 2.

[^40]: Y. Li, Y. Wang, Y. Liu, J. He, L. Fan, and Z. Zhang (2025) End-to-end driving with online trajectory evaluation via bev world model. In ICCV, Cited by: Table 9, §2.

[^41]: Y. Li, K. Xiong, X. Guo, F. Li, S. Yan, G. Xu, L. Zhou, L. Chen, H. Sun, B. Wang, et al. (2025) Recogdrive: a reinforced cognitive framework for end-to-end autonomous driving. arXiv preprint arXiv:2506.08052. Cited by: Table 9, §1, §2, §4.1, §4.1, §4.3, Table 1, Table 2.

[^42]: Y. Li, L. Zhou, S. Yan, B. Liao, T. Yan, K. Xiong, L. Chen, H. Xie, B. Wang, G. Chen, et al. (2026) UniDriveVLA: unifying understanding, perception, and action planning for autonomous driving. arXiv preprint arXiv:2604.02190. Cited by: §2.

[^43]: Z. Li, W. Yao, Z. Wang, X. Sun, J. Chen, N. Chang, M. Shen, Z. Wu, S. Lan, and J. M. Alvarez (2025) Generalized trajectory scoring for end-to-end multimodal planning. arXiv preprint arXiv:2506.06664. Cited by: Table 1, Table 2.

[^44]: B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang, et al. (2025) Diffusiondrive: truncated diffusion model for end-to-end autonomous driving. In CVPR, Cited by: Table 9, Table 1, Table 2.

[^45]: Y. Liao, P. Zhou, S. Huang, D. Yang, S. Chen, Y. Jiang, Y. Hu, J. Cai, S. Liu, J. Luo, L. Chen, S. Yan, M. Yao, and G. Ren (2025) Genie envisioner: a unified world foundation platform for robotic manipulation. arXiv preprint arXiv:2508.05635. Cited by: §2.

[^46]: H. Liu, C. Li, Q. Wu, and Y. J. Lee (2024) Visual instruction tuning. In NeurIPS, Cited by: §1.

[^47]: L. Liu, C. Jia, G. Yu, Z. Song, J. Li, F. Jia, P. Wu, X. Hao, and Y. Luo (2025) Guideflow: constraint-guided flow matching for planning in end-to-end autonomous driving. arXiv preprint arXiv:2511.18729. Cited by: Table 1, Table 1, Table 1.

[^48]: Q. Liu, H. Xu, J. Li, B. Sun, Z. Hao, D. She, X. Zhu, and L. Zhang (2026) Uni-world vla: interleaved world modeling and planning for autonomous driving. arXiv preprint arXiv:2603.27287. Cited by: §A.2, Table 9, §2, §3.3.

[^49]: X. Liu, J. Li, Y. Jiang, N. Sujay, Z. Yang, J. Zhang, J. Abanes, J. Zhang, and C. Feng (2025) Citywalker: learning embodied urban navigation from web-scale videos. In CVPR, Cited by: Table 3, §4.

[^50]: Y. Liu, K. Zhang, Y. Li, Z. Yan, C. Gao, R. Chen, Z. Yuan, Y. Huang, H. Sun, J. Gao, et al. (2024) Sora: a review on background, technology, limitations, and opportunities of large vision models. arXiv preprint arXiv:2402.17177. Cited by: §2.

[^51]: Z. Liu, R. Huang, R. Yang, S. Yan, Z. Wang, L. Hou, D. Lin, X. Bai, and H. Zhao (2025) DrivePI: spatial-aware 4d mllm for unified autonomous driving understanding, perception, prediction and planning. arXiv preprint arXiv:2512.12799. Cited by: §2.

[^52]: H. Lu, Z. Liu, G. Jiang, Y. Luo, S. Chen, Y. Zhang, and Y. Chen (2025) Uniugp: unifying understanding, generation, and planing for end-to-end autonomous driving. arXiv preprint arXiv:2512.09864. Cited by: §2.

[^53]: Y. Luo, F. Li, S. Xu, Z. Lai, L. Yang, Q. Chen, Z. Luo, Z. Xie, S. Jiang, J. Liu, et al. (2025) Adathinkdrive: adaptive thinking via reinforcement learning for autonomous driving. arXiv preprint arXiv:2509.13769. Cited by: §2.

[^54]: Q. Lv, W. Kong, H. Li, J. Zeng, Z. Qiu, D. Qu, H. Song, Q. Chen, X. Deng, and J. Pang (2025) F1: a vision-language-action model bridging understanding and generation to actions. arXiv preprint arXiv:2509.06951. Cited by: §2.

[^55]: J. Pai, L. Achenbach, V. Montesinos, B. Forrai, O. Mees, and E. Nava (2025) Mimic-video: video-action models for generalizable robot control beyond vlas. arXiv preprint arXiv:2512.15692. Cited by: §2.

[^56]: L. Russell, A. Hu, L. Bertoni, G. Fedoseev, J. Shotton, E. Arani, and G. Corrado (2025) Gaia-2: a controllable multi-view generative world model for autonomous driving. arXiv preprint arXiv:2503.20523. Cited by: §2.

[^57]: D. Shah and S. Levine (2022) Viking: vision-based kilometer-scale navigation with geographic hints. In RSS, Cited by: §2.

[^58]: D. Shah, A. Sridhar, A. Bhorkar, N. Hirose, and S. Levine (2023) Gnm: a general navigation model to drive any robot. In ICRA, Cited by: Table 3.

[^59]: D. Shah, A. Sridhar, N. Dashora, K. Stachowicz, K. Black, N. Hirose, and S. Levine (2023) ViNT: a foundation model for visual navigation. In CoRL, Cited by: §2, Table 3.

[^60]: H. Shao, Y. Hu, L. Wang, G. Song, S. L. Waslander, Y. Liu, and H. Li (2024) Lmdrive: closed-loop end-to-end driving with large language models. In CVPR, Cited by: §2.

[^61]: D. Song, J. Liang, A. Payandeh, A. H. Raj, X. Xiao, and D. Manocha (2024) Vlm-social-nav: socially aware robot navigation through scoring using vision-language models. IEEE RA-L. Cited by: §2.

[^62]: A. Sridhar, D. Shah, C. Glossop, and S. Levine (2024) Nomad: goal masked diffusion policies for navigation and exploration. In ICRA, Cited by: Table 3.

[^63]: H. Tian, T. Li, H. Liu, J. Yang, Y. Qiu, G. Li, J. Wang, Y. Gao, Z. Zhang, L. Wang, H. Ye, T. Tan, L. Chen, and H. Li (2025) SimScale: learning to drive via real-world simulation at scale. arXiv preprint arXiv:2511.23369. Cited by: Table 1, Table 1.

[^64]: X. Tian, J. Gu, B. Li, Y. Liu, Y. Wang, Z. Zhao, K. Zhan, P. Jia, X. Lang, and H. Zhao (2024) Drivevlm: the convergence of autonomous driving and large vision-language models. arXiv preprint arXiv:2402.12289. Cited by: §2.

[^65]: T. Wan, A. Wang, B. Ai, B. Wen, C. Mao, C. Xie, D. Chen, F. Yu, H. Zhao, J. Yang, J. Zeng, J. Wang, J. Zhang, J. Zhou, J. Wang, J. Chen, K. Zhu, K. Zhao, K. Yan, L. Huang, M. Feng, N. Zhang, P. Li, P. Wu, R. Chu, R. Feng, S. Zhang, S. Sun, T. Fang, T. Wang, T. Gui, T. Weng, T. Shen, W. Lin, W. Wang, W. Wang, W. Zhou, W. Wang, W. Shen, W. Yu, X. Shi, X. Huang, X. Xu, Y. Kou, Y. Lv, Y. Li, Y. Liu, Y. Wang, Y. Zhang, Y. Huang, Y. Li, Y. Wu, Y. Liu, Y. Pan, Y. Zheng, Y. Hong, Y. Shi, Y. Feng, Z. Jiang, Z. Han, Z. Wu, and Z. Liu (2025) Wan: open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314. Cited by: §2, §3.2, §4.2.

[^66]: H. Wang, D. Liu, H. Xie, H. Liu, E. Ma, K. Yu, L. Wang, and B. Wang (2025) MiLA: multi-view intensive-fidelity long-term video generation world model for autonomous driving. arXiv preprint arXiv:2503.15875. Cited by: §2.

[^67]: J. Wang, Z. Yang, Y. Bai, Y. Li, Y. Zou, B. Sun, A. Kundu, J. Lezama, L. Y. Huang, Z. Zhu, et al. (2025) Drive&Gen: co-evaluating end-to-end driving and video generation models. In IROS, Cited by: §2.

[^68]: L. Wang, Z. Yang, C. Bai, G. Zhang, X. Liu, X. Zheng, X. Long, C. Lu, and C. Lu (2026) Drive-jepa: video jepa meets multimodal trajectory distillation for end-to-end driving. arXiv preprint arXiv:2601.22032. Cited by: Table 9, Table 2.

[^69]: S. Wang, Z. Yu, X. Jiang, S. Lan, M. Shi, N. Chang, J. Kautz, Y. Li, and J. M. Alvarez (2025) OmniDrive: a holistic vision-language dataset for autonomous driving with counterfactual reasoning. In CVPR, Cited by: §2.

[^70]: X. Wang, Z. Zhu, G. Huang, X. Chen, J. Zhu, and J. Lu (2024) Drivedreamer: towards real-world-drive world models for autonomous driving. In ECCV, Cited by: §2.

[^71]: Y. Wang, W. Luo, J. Bai, Y. Cao, T. Che, K. Chen, Y. Chen, J. Diamond, Y. Ding, W. Ding, et al. (2025) Alpamayo-r1: bridging reasoning and action prediction for generalizable autonomous driving in the long tail. arXiv preprint arXiv:2511.00088. Cited by: §2.

[^72]: Y. Wang, J. He, L. Fan, H. Li, Y. Chen, and Z. Zhang (2024) Driving into the future: multiview visual forecasting and planning with world model for autonomous driving. In CVPR, Cited by: §1, §2.

[^73]: M. Wei, C. Wan, J. Peng, X. Yu, Y. Yang, D. Feng, W. Cai, C. Zhu, T. Wang, J. Pang, and X. Liu (2026) Ground slow, move fast: a dual-system foundation model for generalizable vision-language navigation. In ICLR, Cited by: §2, §4.

[^74]: M. Wei, C. Wan, X. Yu, T. Wang, Y. Yang, X. Mao, C. Zhu, W. Cai, H. Wang, Y. Chen, et al. (2025) Streamvln: streaming vision-and-language navigation via slowfast context modeling. arXiv preprint arXiv:2507.05240. Cited by: §2.

[^75]: Y. Wen, Y. Zhao, Y. Liu, F. Jia, Y. Wang, C. Luo, C. Zhang, T. Wang, X. Sun, and X. Zhang (2024) Panacea: panoramic and controllable video generation for autonomous driving. In CVPR, Cited by: §2.

[^76]: X. Weng, B. Ivanovic, Y. Wang, Y. Wang, and M. Pavone (2024) Para-drive: parallelized architecture for real-time autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15449–15458. Cited by: Table 9.

[^77]: T. Xia, Y. Li, L. Zhou, J. Yao, K. Xiong, H. Sun, B. Wang, K. Ma, H. Ye, W. Liu, et al. (2026) DriveLaW: unifying planning and video generation in a latent driving world. In CVPR, Cited by: §A.2, Table 9, §1, §1, §2, §3.3, §3.3.

[^78]: S. Xing, C. Qian, Y. Wang, H. Hua, K. Tian, Y. Zhou, and Z. Tu (2025) Openemma: open-source multimodal model for end-to-end autonomous driving. In WACV, Cited by: §2.

[^79]: J. Yang, K. Chitta, S. Gao, L. Chen, Y. Shao, X. Jia, H. Li, A. Geiger, X. Yue, and L. Chen (2025) Resim: reliable world simulation for autonomous driving. arXiv preprint arXiv:2506.09981. Cited by: §1, §2.

[^80]: J. Yang, S. Gao, Y. Qiu, L. Chen, T. Li, B. Dai, K. Chitta, P. Wu, J. Zeng, P. Luo, et al. (2024) Generalized predictive model for autonomous driving. In CVPR, Cited by: §1, §2.

[^81]: P. Yang, B. Lu, Z. Xia, C. Han, Y. Gao, T. Zhang, K. Zhan, X. Lang, Y. Zheng, and Q. Zhang (2026) WorldRFT: latent world model planning with reinforcement fine-tuning for autonomous driving. In AAAI, Cited by: Table 9, §2, Table 2.

[^82]: W. Yao, Z. Li, S. Lan, Z. Wang, X. Sun, J. M. Alvarez, and Z. Wu (2026) Drivesuprim: towards precise trajectory selection for end-to-end planning. In AAAI, Cited by: Table 2.

[^83]: S. Ye, Y. Ge, K. Zheng, S. Gao, S. Yu, G. Kurian, S. Indupuru, Y. L. Tan, C. Zhu, J. Xiang, et al. (2026) World action models are zero-shot policies. arXiv preprint arXiv:2602.15922. Cited by: §2.

[^84]: S. Ye, Y. Ge, K. Zheng, S. Gao, S. Yu, G. Kurian, S. Indupuru, Y. L. Tan, C. Zhu, J. Xiang, et al. (2026) World action models are zero-shot policies. arXiv preprint arXiv:2602.15922. Cited by: §2.

[^85]: C. Yuan, Z. Zhang, J. Sun, S. Sun, Z. Huang, C. D. W. Lee, D. Li, Y. Han, A. Wong, K. P. Tee, et al. (2024) Drama: an efficient end-to-end motion planner for autonomous driving with mamba. arXiv preprint arXiv:2408.03601. Cited by: Table 9.

[^86]: T. Yuan, Z. Dong, Y. Liu, and H. Zhao (2026) Fast-wam: do world action models need test-time future imagination?. arXiv preprint arXiv:2603.16666. Cited by: §2, §3.1.

[^87]: S. Zeng, X. Chang, M. Xie, X. Liu, Y. Bai, Z. Pan, M. Xu, X. Wei, and N. Guo (2025) Futuresightdrive: thinking visually with spatio-temporal cot for autonomous driving. arXiv preprint arXiv:2505.17685. Cited by: §2.

[^88]: B. Zhang, N. Song, J. Li, X. Zhu, J. Deng, L. Zhang, et al. (2025) Future-aware end-to-end driving: bidirectional modeling of trajectory planning and scene evolution. In NeurIPS, Cited by: Table 9, §2.

[^89]: J. Zhang, A. Li, Y. Qi, M. Li, J. Liu, S. Wang, H. Liu, G. Zhou, Y. Wu, X. Li, et al. (2025) Embodied navigation foundation model. arXiv preprint arXiv:2509.12129. Cited by: §2.

[^90]: J. Zhang, K. Wang, S. Wang, M. Li, H. Liu, S. Wei, Z. Wang, Z. Zhang, and H. Wang (2024) Uni-navid: a video-based vision-language-action model for unifying embodied navigation tasks. arXiv preprint arXiv:2412.06224. Cited by: §2.

[^91]: J. Zhang, K. Wang, R. Xu, G. Zhou, Y. Hong, X. Fang, Q. Wu, Z. Zhang, and H. Wang (2024) NaVid: video-based vlm plans the next step for vision-and-language navigation. RSS. Cited by: §2.

[^92]: K. Zhang, Z. Tang, X. Hu, X. Pan, X. Guo, Y. Liu, J. Huang, L. Yuan, Q. Zhang, X. Long, et al. (2025) Epona: autoregressive diffusion world model for autonomous driving. In ICCV, Cited by: §A.2, Table 9, §1, §1, §2, §3.1, §3.3, §3.3, Table 2, Table 7.

[^93]: S. Zhang, W. Huang, Z. Chen, C. J. Collister, Q. Huang, and C. Lv (2025) OpenREAD: reinforced open-ended reasoning for end-to-end autonomous driving with llm-as-critic. arXiv preprint arXiv:2512.01830. Cited by: §2.

[^94]: Z. Zhao, T. Fu, Y. Wang, L. Wang, and H. Lu (2025) From forecasting to planning: policy world model for collaborative state-action prediction. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, Cited by: §A.2, Table 9, §1, §2, §3.1, §3.1, §3.3, Table 7, Table 7.

[^95]: W. Zheng, Z. Xia, Y. Huang, S. Zuo, J. Zhou, and J. Lu (2024) Doe-1: closed-loop autonomous driving with large world model. arXiv preprint arXiv:2412.09627. Cited by: §2.

[^96]: Y. Zheng, P. Yang, Z. Xing, Q. Zhang, Y. Zheng, Y. Gao, P. Li, T. Zhang, Z. Xia, P. Jia, et al. (2025) World4drive: end-to-end autonomous driving via intention-aware physical latent world model. In ICCV, Cited by: Table 9, §2, Table 2.

[^97]: Y. Zhong, C. Feng, F. Yan, F. Liu, L. Zheng, and L. Ma (2025) RoboTrom-nav: a unified framework for embodied navigation integrating perception, planning, and prediction. In CVPR, Cited by: §2.

[^98]: G. Zhou, Y. Hong, Z. Wang, X. E. Wang, and Q. Wu (2024) Navgpt-2: unleashing navigational reasoning capability for large vision-language models. In ECCV, Cited by: §2.

[^99]: X. Zhou, D. Liang, S. Tu, X. Chen, Y. Ding, D. Zhang, F. Tan, H. Zhao, and X. Bai (2025) Hermes: a unified self-driving world model for simultaneous 3d scene understanding and generation. In ICCV, Cited by: §2.

[^100]: Z. Zhou, T. Cai, Y. Zhao, Z. Huang, B. Zhou, and J. Ma (2025) AutoVLA: a vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. NeurIPS. Cited by: Table 9, Table 9, Table 9, §1, §2, Table 2, Table 2.

[^101]: S. Zuo, Y. Li, W. Zheng, Z. Zhu, J. Zhou, and J. Lu (2026) Vega: learning to drive with natural language instructions. arXiv preprint arXiv:2603.25741. Cited by: Table 9, §2, §4.1, Table 2.
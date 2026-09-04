---
title: "See Tomorrow, Act Today: Foresight-Driven Autonomous Driving"
source: "https://arxiv.org/html/2605.07195v1"
author:
published:
created: 2026-09-04
description:
tags:
  - "clippings"
---
Bozhou Zhang Affiliation: School of Data Science, Fudan University Affiliation: Shanghai Innovation Institute    Nan Song Affiliation: School of Data Science, Fudan University    Yuang Wang Affiliation: School of Data Science, Fudan University    Jiankang Deng Affiliation: Imperial College London    Xiatian Zhu Affiliation: University of Surrey [https://github.com/LogosRoboticsGroup/ForeSight](https://github.com/LogosRoboticsGroup/ForeSight)    Li Zhang Affiliation: School of Data Science, Fudan University

###### Abstract

Current end-to-end autonomous driving planners are fundamentally reactive: they condition on historical and present observations to predict future actions. We argue that autonomous agents should instead imagine future scenes before deciding, just as human drivers mentally simulate “what will happen next” before acting. We introduce ForeSight, a foundation world model centric planning framework that reframes autonomous driving as anticipatory decision-making. Rather than treating world models as auxiliary components, ForeSight makes future scene imagination the primary driver of action prediction. Our approach operates in two stages: (1) generating plausible future visual worlds via a pretrained world model, and (2) planning actions conditioned on these imagined futures. This paradigm shift from “what should I do now?” to “what will happen, and how should I respond?” enables genuinely anticipatory rather than reactive planning. By grounding decisions in anticipated contexts rather than present observations alone, ForeSight navigates dynamic, interactive scenarios more effectively. Extensive experiments on NAVSIM and nuScenes demonstrate that explicit future imagination significantly outperforms previous state-of-the-art alternatives, validating our foresight-driven approach.

## 1 Introduction

![[cmp.png|Refer to caption]]

Figure 1: Paradigm comparison. (a) Reactive end-to-end planning based on current observations 29 22 48. (b) A lightweight world model used as an auxiliary component for alignment or simplified prediction 40 38 80. (c) ForeSight: A foundation world model centric framework where future scene imagination drives action prediction.

As one of the most crucial tasks in autonomous driving, end-to-end planning has attracted widespread attention [^1] [^10] [^25], while the complexity of scene elements and their interactions significantly increases the difficulty of this task. Existing methods [^55] [^29] [^22] [^81] [^63] [^72] [^5] [^26] [^12] [^48] [^87] follow a perception-to-planning paradigm, as shown in Fig. 1 (a), predicting actions from historical and current observations. Despite achieving promising results, these methods are fundamentally reactive, as they decide based on what has happened and what is happening, rather than what will happen. Without explicitly imagining future scenes, they lack the foresight needed for proactive decision-making in dynamic traffic scenarios.

Foundation world models [^51] [^49] have demonstrated strong capabilities in understanding and predicting dynamic evolution, inspiring their application to autonomous driving [^18] [^59] [^15] [^82] [^68] [^7]. Hence, how to effectively integrate world models with planning frameworks is a worthwhile direction to explore. However, existing approaches often treat world models as auxiliary components—either using them for representation alignment via reconstruction constraints or auxiliary supervision [^38] [^40] [^88], as shown in Fig. 1 (b), or predicting simplified future features [^42] [^80] without modeling full scene dynamics. Critically, these methods do not enable the planning module to genuinely leverage detailed future information as the primary basis for decision-making.

We argue that autonomous agents should imagine future scenes before deciding, just as human drivers mentally simulate “what will happen next” before acting. We present ForeSight, a foundation world model centric planning framework that makes future scene imagination the primary driver of action prediction, as shown in Fig. 1 (c). Unlike prior work, ForeSight integrates a pretrained world model directly as the visual encoder, generating detailed future scene visual representations that fundamentally inform the action decoder. This paradigm shift, from “what should I do now?” to “what will happen, and how should I respond?”, enables genuinely anticipatory rather than reactive planning. To complement future scene representations with current scene multi-modal and multi-view information, we employ a lightweight encoder for present observations. A state-based action decoder is designed for action prediction, which incorporates a WM-QFormer for aggregating and adapting future features and employs factorized attention to interact state-based trajectory queries with current and future scene features for planning.

Our contributions are threefold: (i) We propose a foundation world model centric planning framework that makes future scene imagination the primary driver of action prediction, enabling anticipatory rather than reactive autonomous driving. (ii) We realize this through ForeSight, which integrates a pretrained world model as the core visual encoder alongside a lightweight encoder for incorporating current scene observations and offering more scene context, as well as a specially designed action decoder for generating future trajectories. (iii) Extensive experiments on NAVSIM and nuScenes benchmarks demonstrate that ForeSight achieves state-of-the-art performance.

## 2 Related work

#### End-to-end autonomous driving.

As one of the most critical components in autonomous driving, end-to-end planning has witnessed remarkably rapid progress in recent years. Early approaches [^55] [^19] [^22] [^29] follow the perception-to-planning paradigm, integrating multiple intermediate driving tasks into a unified framework. Building upon this design, subsequent works introduce sparse representations [^81] [^63] and generative action modeling [^48] [^72] to enable more efficient scene understanding and more expressive motion patterns. To further enhance representational and generalization capabilities, recent studies [^26] [^5] [^45] [^46] [^17] [^16] [^61] [^64] have increasingly focused on aggregating scene information within Transformer-based architectures. Moreover, inspired by the success of multi-modal large language models (MLLMs), vision-language-action approaches [^57] [^12] [^43] [^41] [^78] [^8] [^90] [^65] [^30] [^3] [^6] directly leverage the strengths of MLLMs to predict and plan driving actions. Some other works explore the application of reinforcement learning [^60] [^73] [^4] [^24] [^36] [^28] in the field of autonomous driving.

#### World models in autonomous driving.

Given the observations and states of the current environment, world models are capable of understanding and predicting future scene evolution. In the field of autonomous driving, generated content primarily focuses on realistic visual imagery [^15] [^18] [^82] [^56] [^13] [^20], road topology in the BEV space [^9] [^21] [^58], and scene-level occupancy representations [^85] [^69] [^76] that capture the spatial layout and dynamic evolution of surrounding agents and free space. Regarding visual representations, methods such as [^14] [^70] [^66] [^67] [^15] [^84] [^75] [^34] [^89] [^74] [^35] employ diffusion-based models to achieve controllable and realistic video generation. Specifically, Drive-WM [^67] supports multi-view generation and explores its practical value in planning models, while Vista [^15] is trained on web-scale driving videos, endowing it with strong generative capability for diverse and long-horizon outputs. In contrast, the GAIA series [^18] [^59] and DrivingWorld [^20] realize generative modeling in an auto-regressive manner, making them better suited for continuous long-term video generation. Moreover, Epona [^82] integrates the strengths of both paradigms, enabling joint visual and motion modeling for comprehensive scene generation.

#### Planning with world modeling.

Powerful world models can reveal the potential evolution of future scenes, providing a stronger foundation for motion prediction, as explored in robotics [^23] [^49]. The concepts of auxiliary supervision and collaborative optimization with world models have also been extended to autonomous driving planning frameworks. Specifically, auxiliary supervision-based approaches [^38] [^40] [^88] leverage predicted action latents or trajectories as conditions to generate future visual representations via world models, which are then supervised using realistic representations. Meanwhile, collaborative optimization-based methods [^42] [^80] employ BEV world models as trajectory selectors or to facilitate interaction with future information. However, these methods either rely on auxiliary supervision or resort to simplified features, making it difficult to fully exploit the powerful future representations provided by world models. In contrast, our approach directly takes the generated future representations as input, establishing more informative priors for action prediction.

## 3 Methodology

In this section, we introduce ForeSight, a foundation world model centric planning framework. We first introduce the task formulation and the overview of our pipeline in Sec. 3.1. Then we describe two crucial representation sources in Sec. 3.2, followed by our state-based interactive decoding for trajectory planning in Sec. 3.3, and finally detail the two-phase model training strategy in Sec. 3.4.

![[pipeline.png|Refer to caption]]

Figure 2: Overview of ForeSight. ForeSight introduces foundation world models into an end-to-end planning framework with using the current-frame features as an additional supplement. Besides, we design a WM-QFormer to compress future features with a set of frame queries and adapt them to the action head. To facilitate the interaction between action and visual presentations, we adopt state queries to explicitly represent time steps and factorized attention for feature interaction.

### 3.1 Preliminary

#### Task formulation.

The end-to-end planning task in autonomous driving takes as input raw sensor data (e.g., multi-view images and LiDAR point clouds), captures the interactions among traffic elements, and finally predicts the future trajectories of the ego vehicle. To promote interpretability and reduce learning difficulty, multiple intermediate tasks are employed for auxiliary supervision, such as detection, map segmentation, and motion prediction. World modeling for autonomous driving aims to understand driving scenes and predict their dynamic evolution, which facilitates downstream applications such as real-world evaluation and simulation. In the context of our work, the world model is considered a powerful foresight generator that can provide rich future representations, serving the subsequent planning head to predict more accurate trajectories.

#### Pipeline overview.

As depicted in Fig. 2, the overall pipeline of ForeSight integrates a foundation world model into an end-to-end planning framework to enhance future reasoning and decision-making. Specifically, current-frame features are employed as an additional supplement to compensate for potential information gaps in the world model outputs. To align future representations with the action planning process, a WM-QFormer is introduced to compress the predicted future features through a set of frame queries and adapt them to the action head. Furthermore, state queries are utilized to explicitly encode temporal steps, enabling the model to capture the sequential nature of planning. To promote deeper interaction between visual and action representations, a factorized attention mechanism is applied, facilitating efficient cross-modal feature fusion and temporal reasoning throughout the pipeline.

### 3.2 Input representation encoding

The core vision encoder of ForeSight is the World Model (WM) encoder, which is directly inherited from existing foundation world models. For simplicity, we only consider diffusion-based world models in this work, such as [^15] [^82] [^67] [^54]. Specifically, the WM encoder takes as input the images of the current frame, conditioned on motion attributes (e.g., yaws and poses) or commands. During the denoising stage, a specific step $t_{\rm d}$ is selected to sample the latent features as future visual representations $F_{\rm wm}\in\mathbb{R}^{T_{\rm wm}\times C_{\rm wm}\times H\times W}$, where $T_{\rm wm}$ refers to the number of future frames, $C_{\rm wm}$ refers to the feature dimension, and $H$ and $W$ refer to the height and width of the feature maps. This process can be formulated as:

$$
F_{\rm wm}={\rm WM}^{(t_{\rm d})}(\mathcal{I},F_{\rm cond}),
$$

where $\mathcal{I}$ and $F_{\rm cond}$ denote the raw input images and the condition latents, respectively. Note, the sampling step $t_{\rm d}$ is adjustable to balance efficiency and performance.

In addition, we also utilize a lightweight Transformer-based encoder [^55] [^48] [^29] [^38] for the current frame. In autonomous driving, accurately predicting future trajectories requires rich multi-view information, as relying solely on a forward-facing view can miss important cues from other directions. Most existing foundation world models primarily process front-view images, which may lead to incomplete or biased future representations. This lightweight encoder complements the multi-modal and cross-view information missing in the world model, providing precise and comprehensive features from the current frame, thereby further enhancing action decoding and improving the accuracy of predicted trajectories. Concretely, this module can be represented as:

$$
F_{\rm cur}={\rm Encoder}(\mathcal{I},\mathcal{P},\mathcal{E}),
$$

where $\mathcal{P}$ and $\mathcal{E}$ refer to the input LiDAR point cloud and ego status.

### 3.3 State-based interactive decoding

#### Time state queries representation for trajectory planning.

The planning task in autonomous driving aims to generate multi-step future trajectories of the ego vehicle, which requires the model to reason over both spatial and temporal dimensions. Meanwhile, the foundation world model provides rich and temporally structured visual representations for the corresponding future time steps, offering valuable priors about scene evolution and interaction dynamics. To effectively align these temporally indexed future features with the planning process, we employ a set of learnable time state queries $Q_{\rm s}\in\mathbb{R}^{M\times T_{\rm f}\times C}$ following [^79], where $M$ denotes the number of planning modes, $T_{\rm f}$ denotes the number of predicted future steps, and $C$ is the feature dimension. These queries explicitly encode temporal progression, enabling more precise and coherent interaction between future visual representations and the trajectory prediction process.

#### WM-QFormer for future feature aggregation.

To aggregate scene information into queries, we introduce a dedicated WM-QFormer, specifically designed for processing the future world-model features $F_{\rm wm}$. The WM-QFormer is implemented as a spatial–temporal Transformer that jointly captures intra-frame spatial structure and inter-frame temporal dynamics. We use $N_{\rm wm}$ learnable queries for each frame to extract and compress the relevant information, producing compact representations $F^{\prime}_{\rm wm}\in\mathbb{R}^{T_{\rm wm}\times N_{\rm wm}\times C}$. This module is deliberately designed for world-model–based planning. The generated future frames typically contain abundant fine-grained textures and noise, which, if directly exposed to trajectory queries, may introduce interference into the planning process. Our WM-QFormer filters out these irrelevant details by selectively aggregating informative features from each frame, yielding a distilled representation that provides a reliable and noise-robust reference for the action head. This ensures that the planner benefits from meaningful future context while avoiding distraction from redundant or noisy visual cues.

#### Factorized attention for interacting trajectory queries with current and future features.

Given the current features $F_{\rm cur}$, the compressed future features $F^{\prime}_{\rm wm}$, and the state queries $Q_{\rm s}$, we adopt a factorized attention mechanism to enable effective interaction between the state queries and both current and future features through two separate cross-attention modules. Specifically, all state queries first attend to the current features via a standard cross-attention module, ensuring that each future time step comprehensively perceives the present scene. For the interaction with future features, we introduce additional time embeddings to encourage temporally adjacent steps to attend more strongly to each other, capturing the sequential dependencies in future trajectories. Moreover, considering that the future features and state queries may have different numbers of time steps, we employ sinusoidal positional embeddings to facilitate generalization across varying sequence lengths. The interaction process can be formulated as:

$$
\begin{split}Q_{\rm s}&={\rm CrossAttn}(Q_{\rm s},F_{\rm cur}),\\
Q_{\rm s}&={\rm CrossAttn}(Q_{\rm s}+E_{\rm s},F^{\prime}_{\rm wm}+E_{\rm wm}),\\
\end{split}
$$

where $E_{\rm s}$ and $E_{\rm wm}$ denote the time embeddings of the state queries and future features, respectively. After the cross-attention interactions, a trajectory decoder is applied to produce the planned trajectories for the ego vehicle:

$$
\mathcal{T}={\rm TrajDecoder}(Q_{\rm s}),
$$

where $\mathcal{T}$ represents the decoded multi-step ego trajectories for planning. This factorized design allows the model to separately integrate present and predicted future information, thereby improving trajectory accuracy while maintaining temporal coherence.

Table 1: Performance comparison of planning on the NAVSIM navtest split with closed-loop metrics. The best and second-best results are highlighted in bold and underline, respectively. We categorize the methods compared into three groups: planning models, world models, and planning with world models.

<table><tbody><tr><th>Type</th><th>Method</th><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Comf. <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><th rowspan="9">Planning model</th><th>UniAD <sup><a href="#fn:22">22</a></sup></th><td>97.8</td><td>91.9</td><td>92.9</td><td>100</td><td>78.8</td><td>83.4</td></tr><tr><th>PARA-Drive <sup><a href="#fn:71">71</a></sup></th><td>97.9</td><td>92.4</td><td>93.0</td><td>99.8</td><td>79.3</td><td>84.0</td></tr><tr><th>TransFuser <sup><a href="#fn:55">55</a></sup></th><td>97.7</td><td>92.8</td><td>92.8</td><td>100</td><td>79.2</td><td>84.0</td></tr><tr><th>DRAMA <sup><a href="#fn:77">77</a></sup></th><td>98.0</td><td>93.1</td><td>94.8</td><td>100</td><td>80.1</td><td>85.5</td></tr><tr><th>Hydra-MDP++ <sup><a href="#fn:37">37</a></sup></th><td>97.6</td><td>96.0</td><td>93.1</td><td>100</td><td>80.4</td><td>86.6</td></tr><tr><th>DiffusionDrive <sup><a href="#fn:48">48</a></sup></th><td>98.2</td><td>96.2</td><td>94.7</td><td>100</td><td>82.2</td><td>88.1</td></tr><tr><th>Hydra-NeXt <sup><a href="#fn:44">44</a></sup></th><td>98.1</td><td>97.7</td><td>94.6</td><td>100</td><td>81.8</td><td>88.6</td></tr><tr><th>GoalFlow <sup><a href="#fn:72">72</a></sup></th><td>98.4</td><td>98.3</td><td>94.6</td><td>100</td><td>85.0</td><td>90.3</td></tr><tr><th>ReCogDrive <sup><a href="#fn:43">43</a></sup></th><td>97.9</td><td>97.3</td><td>94.9</td><td>100</td><td>87.3</td><td>90.8</td></tr><tr><th rowspan="2">World model</th><th>DrivingGPT <sup><a href="#fn:7">7</a></sup></th><td>98.9</td><td>90.7</td><td>94.9</td><td>95.6</td><td>79.7</td><td>82.4</td></tr><tr><th>Epona <sup><a href="#fn:82">82</a></sup></th><td>97.9</td><td>95.1</td><td>93.8</td><td>99.9</td><td>80.4</td><td>86.2</td></tr><tr><th rowspan="5">Planning with world model</th><th>LAW <sup><a href="#fn:40">40</a></sup></th><td>96.4</td><td>95.4</td><td>88.7</td><td>99.9</td><td>81.7</td><td>84.6</td></tr><tr><th>World4Drive <sup><a href="#fn:88">88</a></sup></th><td>97.4</td><td>94.3</td><td>92.8</td><td>100</td><td>79.9</td><td>85.1</td></tr><tr><th>WoTE <sup><a href="#fn:42">42</a></sup></th><td>98.5</td><td>96.8</td><td>94.9</td><td>99.9</td><td>81.9</td><td>88.3</td></tr><tr><th>SeerDrive <sup><a href="#fn:80">80</a></sup></th><td>98.4</td><td>97.0</td><td>94.9</td><td>99.9</td><td>83.2</td><td>88.9</td></tr><tr><th>ForeSight (Ours)</th><td>98.8</td><td>97.2</td><td>94.8</td><td>100</td><td>83.5</td><td>89.3</td></tr></tbody></table>

### 3.4 Model training

For efficient model training, we adopt a two-phase training strategy. In the first phase, we pretrain the original action model without incorporating world model features, allowing the model to effectively learn action-aware perceptual priors. After this action pretraining, we introduce the future visual representations from the world model and integrate the WM-QFormer modules into the pipeline. In the second phase, we perform post-training to jointly optimize all components. Notably, while the world model can undergo additional fine-tuning on the target dataset, it is kept fully frozen during our post-training stage, where the WM-QFormer is trained to adapt the future features to the action head. This two-phase strategy not only improves overall training efficiency but also alleviates instability in early training caused by the imbalance in representational capacity between the two feature sources.

We adopt the same training losses in both phases. The overall loss function $\mathcal{L}$ is defined as the weighted sum of the BEV segmentation loss $\mathcal{L}_{\rm bev}$ and the trajectory regression loss $\mathcal{L}_{\rm traj}$:

$$
\mathcal{L}=\lambda_{1}\mathcal{L}_{\rm bev}+\lambda_{2}\mathcal{L}_{\rm traj}.
$$

where $\lambda_{1}$ and $\lambda_{2}$ denote the loss weights.

## 4 Experiments

### 4.1 Datasets and metrics

We perform experiments on the NAVSIM [^10] and nuScenes [^1] autonomous driving benchmarks. Specifically, NAVSIM is a subset of the nuPlan [^2] dataset, focusing on non-reactive simulation in complex scenarios with dynamic intention changes. The training/validation set and the testing set contain 1,192 and 136 scenarios, respectively, with a sampling frequency of 2 Hz for both camera and LiDAR data. Evaluation on NAVSIM is conducted using the PDM Score (PDMS), which is computed as a weighted sum of No At-Fault Collisions (NC), Drivable Area Compliance (DAC), Time-to-Collision (TTC), Comfort (Comf.), and Ego Progress (EP). For the nuScenes dataset, it contains 1,000 scenes sampled at 2 Hz, and we adopt a 700/150 train/validation split following existing methods [^22] [^29]. Evaluation on nuScenes is performed in an open-loop setting, and we use the metrics proposed in VAD [^29] for comparison.

Table 2: Performance comparison of planning on the nuScenes validation split. ResNet-50 is used as the backbone for all the planning models, except for UniAD, which adopts ResNet-101.

<table><tbody><tr><th rowspan="2">Type</th><th rowspan="2">Method</th><td colspan="4">L2 (<math><semantics><mi>m</mi> <annotation>m</annotation></semantics></math>) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td colspan="4">Col. Rate (%) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><td>1 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></td><td>2 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></td><td>3 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></td><td>Avg.</td><td>1 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></td><td>2 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></td><td>3 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></td><td>Avg.</td></tr><tr><th rowspan="8">Planning model</th><th>BEV-Planner <sup><a href="#fn:47">47</a></sup></th><td>0.28</td><td>0.42</td><td>0.68</td><td>0.46</td><td>0.04</td><td>0.37</td><td>1.07</td><td>0.49</td></tr><tr><th>PARA-Drive <sup><a href="#fn:71">71</a></sup></th><td>0.25</td><td>0.46</td><td>0.74</td><td>0.48</td><td>0.14</td><td>0.23</td><td>0.39</td><td>0.25</td></tr><tr><th>VAD-Base <sup><a href="#fn:29">29</a></sup></th><td>0.41</td><td>0.70</td><td>1.05</td><td>0.72</td><td>0.07</td><td>0.17</td><td>0.41</td><td>0.22</td></tr><tr><th>GenAD <sup><a href="#fn:86">86</a></sup></th><td>0.28</td><td>0.49</td><td>0.78</td><td>0.52</td><td>0.08</td><td>0.14</td><td>0.34</td><td>0.19</td></tr><tr><th>UniAD <sup><a href="#fn:22">22</a></sup></th><td>0.44</td><td>0.67</td><td>0.96</td><td>0.69</td><td>0.04</td><td>0.08</td><td>0.23</td><td>0.12</td></tr><tr><th>BridgeAD <sup><a href="#fn:79">79</a></sup></th><td>0.29</td><td>0.57</td><td>0.92</td><td>0.59</td><td>0.01</td><td>0.05</td><td>0.22</td><td>0.09</td></tr><tr><th>MomAD <sup><a href="#fn:62">62</a></sup></th><td>0.31</td><td>0.57</td><td>0.91</td><td>0.60</td><td>0.01</td><td>0.05</td><td>0.22</td><td>0.09</td></tr><tr><th>SparseDrive <sup><a href="#fn:63">63</a></sup></th><td>0.29</td><td>0.58</td><td>0.96</td><td>0.61</td><td>0.01</td><td>0.05</td><td>0.18</td><td>0.08</td></tr><tr><th rowspan="3">Planning with world model</th><th>LAW <sup><a href="#fn:40">40</a></sup></th><td>0.26</td><td>0.57</td><td>1.01</td><td>0.61</td><td>0.14</td><td>0.21</td><td>0.54</td><td>0.30</td></tr><tr><th>World4Drive <sup><a href="#fn:88">88</a></sup></th><td>0.23</td><td>0.47</td><td>0.81</td><td>0.50</td><td>0.02</td><td>0.12</td><td>0.33</td><td>0.16</td></tr><tr><th>ForeSight (Ours)</th><td>0.36</td><td>0.55</td><td>0.93</td><td>0.62</td><td>0.04</td><td>0.12</td><td>0.37</td><td>0.18</td></tr></tbody></table>

### 4.2 Implementation details

#### Model settings.

For the NAVSIM benchmark, we utilize both images and LiDAR as raw input, while only images are used for the nuScenes dataset. Specifically, 3 camera views are used for NAVSIM, whereas 6 views are used for nuScenes. The resolution of input images is set to 1024×256 for NAVSIM following previous methods [^55] [^48], while it is downscaled to 640×360 for the nuScenes dataset.

Regarding our choice of world models, we mainly employ Epona [^82] for NAVSIM and nuScenes, which enables direct adaptation without requiring any fine-tuning or minor adjustments to the output frequency. We generate future frames from the world model with the same number of steps as used in planning — 8 future frames for NAVSIM and 6 for nuScenes — both conditioned only on the current frame. To further demonstrate the wide adaptability of our model, we also use Vista [^15] as the world model on nuScenes, with the results presented in the supplementary materials.

As for the output trajectories, planning needs to consider the multimodality of future trajectories. For the NAVSIM dataset, a 4-second planning window with 8 future steps is used, and we adopt a common setting of 20 trajectory modes for decoding. For the nuScenes dataset, the planning window is set to 3 seconds with 6 future steps, and 6 modes are utilized for decoding.

#### Training settings.

We train our model using 8 NVIDIA H100 GPUs. For NAVSIM, the batch size is 8, with 80 epochs for pretraining the action model without using the world model, followed by 20 epochs for post-training the final model combining the world model and the action model. For nuScenes, the batch size is 1, with 12 epochs for pretraining and 6 epochs for post-training. The learning rate is set to $1\times 10^{-4}$ for both NAVSIM and nuScenes, both of which are optimized using AdamW [^52].

Table 3: Ablation study on the key components of our model. “w. WM” indicates that the world model is incorporated to facilitate the planning process.

<table><thead><tr><th rowspan="2">ID</th><th rowspan="2">w. WM</th><th rowspan="2">WM-QFormer</th><th>State</th><th>Factorized</th><th rowspan="2">NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th rowspan="2">DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th rowspan="2">TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th rowspan="2">Comf. <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th rowspan="2">EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th rowspan="2">PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th></tr><tr><th>queries</th><th>attetntion</th></tr></thead><tbody><tr><td>1</td><td></td><td></td><td></td><td></td><td>97.8</td><td>95.6</td><td>93.4</td><td>100</td><td>81.6</td><td>86.8</td></tr><tr><td>2</td><td>✓</td><td></td><td></td><td></td><td>97.8</td><td>95.9</td><td>93.3</td><td>100</td><td>82.1</td><td>87.1</td></tr><tr><td>3</td><td>✓</td><td>✓</td><td></td><td></td><td>98.6</td><td>96.2</td><td>95.0</td><td>100</td><td>81.3</td><td>87.9</td></tr><tr><td>4</td><td>✓</td><td>✓</td><td>✓</td><td></td><td>98.6</td><td>96.8</td><td>95.0</td><td>100</td><td>82.0</td><td>88.5</td></tr><tr><td>5</td><td>✓</td><td>✓</td><td></td><td>✓</td><td>98.4</td><td>96.5</td><td>95.3</td><td>100</td><td>81.6</td><td>88.2</td></tr><tr><td>6</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>98.8</td><td>97.2</td><td>94.8</td><td>100</td><td>83.5</td><td>89.3</td></tr></tbody></table>

Table 4: Ablation study on the use of different types of world models. “Simple WM” refers to lightweight or simplified world models, such as [^42] [^80], while “Found. WM” denotes the foundation world models employed in our method.

|  | DAC $\uparrow$ | TTC $\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- |
| w/o WM | 95.6 | 93.4 | 81.6 | 86.8 |
| Simple WM | 96.3 | 93.4 | 82.2 | 87.5 |
| Found. WM | 97.2 | 94.8 | 83.5 | 89.3 |

Table 5: Ablation study on the number of steps in the denoising procedure.

| Steps | DAC $\uparrow$ | TTC $\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- |
| 25 | 96.4 | 94.8 | 81.3 | 88.0 |
| 50 | 96.6 | 95.2 | 81.5 | 88.3 |
| 75 | 97.3 | 94.7 | 83.5 | 89.2 |
| 100 | 97.2 | 94.8 | 83.5 | 89.3 |

### 4.3 Comparison with state of the art

As shown in Tab. 1, we compare our ForeSight with several state-of-the-art methods on the NAVSIM navtest split. To ensure a fair comparison, existing methods are divided into three groups: planning models, world models, and planning with world models. Our approach achieves the highest PDM Score of 89.3, obtaining the best results within the planning-with-world-models group. It significantly outperforms other methods, benefiting from the scene understanding and future representations provided by foundation world models. Compared with the world model group, where the planning branch primarily serves as an auxiliary component to enhance the generation process, our ForeSight significantly surpasses DrivingGPT [^7] and Epona [^82], demonstrating the effectiveness of our specially designed action decoder. Compared with planning model methods, except for the powerful VLA-based approach ReCogDrive [^43] trained with reinforcement learning and GoalFlow equipped with a V2-99 backbone, our method outperforms all other planning methods.

Furthermore, we evaluate our method on the open-loop nuScenes dataset, as shown in Tab. 2. Although the scenarios and evaluation protocols in nuScenes are relatively simple and the metrics are not entirely comprehensive [^47], the dataset still provides a suitable environment for validating our approach. Compared with recent state-of-the-art methods, including powerful planning models such as PARA-Drive [^71], BridgeAD [^79], SparseDrive [^63], and MomAD [^62], our method demonstrates competitive performance.

### 4.4 Ablation study

#### Effects of components.

We first conduct an ablation study on all key components of ForeSight, as shown in Tab. 3. The first row represents our baseline, which uses only the current visual encoder and a simple action decoder, achieving a PDMS of 86.8. In the second row, we integrate the world model into the framework and interact future frame features using a vanilla attention mechanism. This yields a slight improvement over the baseline, indicating that future features can benefit the planning process, but not in a straightforward manner. In the third row, we design the WM-QFormer to aggregate the generated future frame features, resulting in a significant improvement in the PDMS to 87.9. This demonstrates that the WM-QFormer effectively aggregates relevant information for planning while mitigating interfering factors. The fourth and fifth rows illustrate the impact of the state queries and the factorized attention module. Each component individually contributes to performance gains, and their combination leads to an even more substantial improvement. The state queries and factorized attention work together to provide precise representations of future trajectories and enable accurate interactions with the generated future frame features. All components are integrated in the last row, forming the complete model and achieving optimal performance.

#### Effects of foundation world model.

To validate the effectiveness of large-scale pretrained foundation world models, we also explore replacing them with lightweight future predictors such as [^42] [^38] [^80]. The experimental results in Tab. 4 show that foundation world models significantly outperform these simplified predictors. Although the lightweight predictors can learn future representations to some extent after training, they struggle to generalize to complex or edge-case scenarios, limiting their impact on downstream planning. In contrast, foundation world models exhibit strong generalization and generative capabilities, effectively mitigating this issue. As a result, they can substantially assist the action head in understanding future scenarios and guiding decision-making.

#### Effects of the number of denoising steps.

We investigate the impact of the number of denoising steps, as shown in Tab. 5. In the early stages of denoising, future visual representations evolve from coarse to fine, which significantly benefits action planning. In later steps, the representations have already captured the essential traffic elements and undergo only minor refinements. Consequently, additional steps provide marginal improvements while increasing inference time. While the final model uses 100 denoising steps to achieve the highest performance, 75 steps suffice to strike a favorable balance between efficiency and accuracy, yielding the best trade-off.

### 4.5 Qualitative results

As shown in Fig. 3, we visualize the planned trajectories alongside the corresponding 8 future frames generated by the world models. Interaction scenarios are illustrated in (a) and (c), where the interactions between traffic elements in the future scenes are generated appropriately. This assists the planning model in better understanding the scene and modeling relationships, ultimately yielding accurate trajectories for the ego vehicle. Moreover, given the importance of generated future visual representations for action prediction, we also visualize complex scenarios, such as turning behaviors, in (b) and (c). It can be observed that the foundation world model effectively handles these situations, generating future frames that guide the final trajectories. Additional visualizations and failure cases are provided in the supplementary materials.

![[visual.png|Refer to caption]]

Figure 3: Visualization of ForeSight on the NAVSIM 10 dataset. The left panel shows the planned trajectories in the BEV view, while the right panel presents the generated future video over the next 8 time steps. The ground-truth trajectory is depicted in green, and the final planned trajectory is highlighted in orange.

## 5 Conclusion

We present ForeSight, a foundation world model centric framework for end-to-end autonomous driving planning that shifts the paradigm from reactive to anticipatory decision-making. By explicitly imagining plausible future scenes, ForeSight enables autonomous agents to grasp upcoming dynamics and respond accordingly, mirroring human driving behavior. The framework leverages a pretrained world model to generate detailed future scene representations, complemented by a lightweight encoder for current observations. These features are integrated through a state-based decoder equipped with a WM-QFormer for future feature aggregation and factorized attention to enable effective interaction between state-based trajectory queries and both current and future scene representations. Extensive evaluations on the NAVSIM and nuScenes benchmarks show that planning with anticipated future contexts significantly outperforms state-of-the-art methods, validating the benefits of foresight-driven planning. ForeSight demonstrates the promise of coupling foundation world models with planning modules for more intelligent and anticipatory autonomous driving systems.

Limitations and future work. ForeSight relies on two key components: the world model and the action head. The world model offers strong scene understanding and future representations but remains computationally expensive, limiting practicality. For the action head, recent end-to-end and VLA-based approaches have leveraged reinforcement learning to boost performance. Incorporating similar strategies into the world–action paradigm may further enhance action generation and overall planning.

## Acknowledgements

This work was supported in part by New Generation Artificial Intelligence-National Science and Technology Major Project (2025ZD0123004), Ningbo grant (2025Z038) and National Natural Science Foundation of China (Grant No. 62376060).

## References

Supplementary Material

## 6 Discussions

#### Discussion 1:

World model:

Besides the comparisons presented in the main paper, we further discuss how ForeSight differs from methods such as Epona [^82] and DrivingGPT [^7], highlighting the unique contributions of our approach.

Although Epona, DrivingGPT, and our model all generate future frames and plan future trajectories, their focuses are fundamentally different. Epona and DrivingGPT are primarily world-modeling methods—their core contribution lies in training generative models for future-scene synthesis, whereas trajectory planning is treated as an auxiliary output. DrivingGPT uses discrete tokens to generate future trajectories, while Epona employs a diffusion transformer for trajectory generation. Both methods jointly generate future frames and trajectories within a unified generative framework.

In contrast, our approach follows a world–action design tailored specifically for end-to-end autonomous driving. ForeSight leverages a foundation world model as a unified module for perception, comprehension, and future-scene imagination, while a specially designed action module is optimized to produce high-quality trajectories. This separation allows the system to fully exploit future predictions while maintaining strong planning performance.

Planning model:

Recent planning models explore diverse directions to enhance planning performance, including diffusion-based decoders [^48] [^43] [^27] [^32] [^39] [^31] [^83], reinforcement-learning–based approaches [^53] [^28] [^4] [^33] [^60], test-time training [^61], clustered anchored trajectory priors [^44] [^46], mixture-of-experts architectures [^11], and Gaussian-feature–fusion strategies [^50]. All of these methods demonstrate strong empirical performance.

These approaches are largely orthogonal to ours: ForeSight intentionally adopts simple action-decoding modules so that our core contribution—the integration of a foundation world model to guide planning—is clearly isolated and easy to plug into existing systems. We believe that combining these advanced planning techniques with our world–action framework would likely yield even better performance.

#### Discussion 2:

About the world model architecture.

Our ForeSight relies on the generated future-frame features for planning, which means it is not restricted to any specific world-model architecture, such as diffusion-based models [^82] [^15] or GPT-based models [^20] [^7]. In our main experiments, we adopt Epona [^82] as the primary world model, considering both its capability and open-source availability. Nevertheless, we also provide results using alternative world-model architectures in Section 7.2, demonstrating that ForeSight is compatible with diverse backbone designs.

Table 6: Generation performance comparison.

| Method | FVD <sub>10</sub> |
| --- | --- |
| Epona [^82] | 50.77 |
| ForeSight (Ours) | 54.63 |

#### Discussion 3:

About the current encoder.

In autonomous driving, accurate trajectory planning often requires a detailed understanding of the surroundings, typically obtained from multi-view cameras or LiDAR point clouds. To provide this spatial awareness, we incorporate a lightweight encoder based on TransFuser [^55] to extract current-frame features as an additional supplement. However, this component is not strictly necessary. As shown in Section 7.2, we also report results without the current encoder. With future advances in world models—particularly in high-resolution and multi-view generation—we expect that the current encoder can eventually be removed and fully replaced by a unified world–action framework.

#### Discussion 4:

Efficiency analysis.

For the parameter size, ForeSight mainly consists of three components: the foundation world model, the current encoder, and the action decoder. For the foundation world model, we adopt Epona [^82], which contains 2.5 B parameters. The current encoder is largely inherited from TransFuser [^55], comprising 52 M parameters. The action decoder contains an additional 21 M parameters.

For inference time, we evaluate our model on an NVIDIA H100 GPU. The average inference time of ForeSight is 900 ms, with the majority (approximately 870 ms) attributed to the world model [^82]. As world-model architectures continue to advance, their inference efficiency is expected to improve substantially, making the overall system considerably more deployment-friendly.

## 7 Experiments

### 7.1 Implementation details

Besides the implementation details provided in the main paper, additional information is included here to ensure full reproducibility. For the foundation world model, Epona [^82] is adopted. Its native generation frequency is 5 Hz, whereas the planning frequency in our system is 2 Hz. To match this temporal resolution, Epona is first finetuned on the nuPlan [^2] dataset at 2 Hz and subsequently frozen when training the full pipeline on NAVSIM [^10]. For the current encoder, the design largely follows TransFuser [^55]. During training, the current encoder and the action decoder are first pretrained without the future feature cross-attention or the WM-QFormer modules. Afterwards, the full model is trained end-to-end with all components enabled, except that the world model remains frozen.

Table 7: Performance comparison with and without the current encoder.

|  | DAC $\uparrow$ | TTC $\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- |
| w/o Current | 96.3 | 95.4 | 81.7 | 88.2 |
| ForeSight | 97.2 | 94.8 | 83.5 | 89.3 |

Table 8: Performance with different world-model architectures on the nuScenes dataset for the planning task.

<table><tbody><tr><th rowspan="2">Method</th><th colspan="4">L2 (<math><semantics><mi>m</mi> <annotation>m</annotation></semantics></math>) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th><th colspan="4">Col. Rate (%) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th></tr><tr><td>1 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></td><td>2 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></td><td>3 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></td><td>Avg.</td><td>1 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></td><td>2 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></td><td>3 <math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math></td><td>Avg.</td></tr><tr><th>ForeSight-Vista</th><th>0.42</th><th>0.63</th><th>0.88</th><th>0.64</th><th>0.08</th><th>0.22</th><th>0.51</th><th>0.27</th></tr><tr><th>ForeSight-Epona</th><td>0.36</td><td>0.55</td><td>0.93</td><td>0.62</td><td>0.04</td><td>0.12</td><td>0.37</td><td>0.18</td></tr></tbody></table>

### 7.2 More experiment results

#### Generation performance of the world model.

As shown in Table 6, we report the Fréchet Video Distance (FVD) of ForeSight and Epona [^82] on the nuPlan [^2] dataset. The results indicate that ForeSight retains nearly the same generation capability as Epona after finetuning.

#### Performance without the current encoder.

As shown in Table 7, we report the performance of our model without the current encoder on the NAVSIM [^10] dataset. The results show that the model still achieves strong performance even when this module is removed, demonstrating that it is not strictly necessary. Nevertheless, the current encoder is retained in the full system, as it further enhances robustness and overall capability.

#### Performance with an alternative world-model architecture.

As shown in Table 8, we also evaluate our framework using Vista [^15] as the foundation world model. Since Vista is trained on the nuScenes [^1] dataset, the experiments are conducted on this dataset for the end-to-end planning task. The results indicate that ForeSight with Vista achieves strong performance as well, demonstrating that our framework is not restricted to a specific world-model architecture.

## 8 Qualitative results

As shown in Figure 4, additional qualitative results of ForeSight are provided, including turning behaviors (parts (a) and (c)), a traffic-congestion scenario (part (b)), and a fast-driving behavior (part (d)). Across all cases, the model not only predicts future frames accurately but also produces precise future trajectories.

## 9 Failure cases

Although ForeSight is powerful, it still fails in certain scenarios. Representative failure cases are shown in Figure 5, which may provide insights for future research.

In part (a), the scenario involves a right-turning maneuver. The foundation world model accurately predicts both the turning motion and the post-turn scene. However, the action decoder generates an overly conservative and slow trajectory. This indicates that the world model and the action model should be more tightly coupled so that the planner can better leverage future predictions for trajectory generation.

In part (b), the scenario involves fast driving over a long distance within the planning horizon, and the road is highly winding. While the foundation world model produces accurate predictions initially, it fails in the later stages due to the increasing curvature of the road. This suggests that long-range prediction capability remains a challenge for the world model. Nevertheless, the action model still produces an accurate trajectory. This highlights the importance of using current-frame features as an additional supplement, which significantly enhances the overall robustness of the system.

![[visual_supp.png|Refer to caption]]

Figure 4: Visualization of ForeSight on the NAVSIM 10 dataset. The left panel shows the planned trajectories in the BEV view, while the right panel presents the generated future video over the next 8 time steps. The ground-truth trajectory is depicted in green, and the final planned trajectory is highlighted in orange.

![[visual_fail.png|Refer to caption]]

Figure 5: Visualization of failure cases for ForeSight on the NAVSIM 10 dataset. The left panel shows the planned trajectories in the BEV view, while the right panel presents the generated future video over the next 8 time steps. The ground-truth trajectory is depicted in green, and the final planned trajectory is highlighted in orange.

[^1]: H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom (2020) Nuscenes: a multimodal dataset for autonomous driving. In CVPR, Cited by: §1, §4.1, §7.2.

[^2]: H. Caesar, J. Kabzan, K. S. Tan, W. K. Fong, E. Wolff, A. Lang, L. Fletcher, O. Beijbom, and S. Omari (2021) Nuplan: a closed-loop ml-based planning benchmark for autonomous vehicles. arXiv preprint. Cited by: §4.1, §7.1, §7.2.

[^3]: J. Cao, Q. Zhang, P. Jia, X. Zhao, B. Lan, X. Zhang, Z. Li, X. Wei, S. Chen, L. Li, et al. (2025) Fastdrivevla: efficient end-to-end driving via plug-and-play reconstruction-based token pruning. arXiv preprint. Cited by: §2.

[^4]: K. Chen, W. Sun, H. Cheng, and S. Zheng (2025) RIFT: closed-loop rl fine-tuning for realistic and controllable traffic simulation. arXiv preprint. Cited by: §2, §6.

[^5]: S. Chen, B. Jiang, H. Gao, B. Liao, Q. Xu, Q. Zhang, C. Huang, W. Liu, and X. Wang (2024) Vadv2: end-to-end vectorized autonomous driving via probabilistic planning. arXiv preprint. Cited by: §1, §2.

[^6]: X. Chen, L. Huang, T. Ma, R. Fang, S. Shi, and H. Li (2025) SOLVE: synergy of language-vision and end-to-end networks for autonomous driving. In CVPR, Cited by: §2.

[^7]: Y. Chen, Y. Wang, and Z. Zhang (2025) Drivinggpt: unifying driving world modeling and planning with multi-modal autoregressive transformers. In ICCV, Cited by: §1, Table 1, §4.3, §6, §6.

[^8]: H. Chi, H. Gao, Z. Liu, J. Liu, C. Liu, J. Li, K. Yang, Y. Yu, Z. Wang, W. Li, et al. (2025) Impromptu vla: open weights and open data for driving vision-language-action models. arXiv preprint. Cited by: §2.

[^9]: K. Chitta, D. Dauner, and A. Geiger (2024) Sledge: synthesizing driving environments with generative models and rule-based traffic. In ECCV, Cited by: §2.

[^10]: D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, et al. (2024) Navsim: data-driven non-reactive autonomous vehicle simulation and benchmarking. In NeurIPS, Cited by: §1, Figure 3, Figure 3, §4.1, §7.1, §7.2, Figure 4, Figure 4, Figure 5, Figure 5.

[^11]: R. Feng, N. Xi, D. Chu, R. Wang, Z. Deng, A. Wang, L. Lu, J. Wang, and Y. Huang (2025) Artemis: autoregressive end-to-end trajectory planning with mixture of experts for autonomous driving. arXiv preprint. Cited by: §6.

[^12]: H. Fu, D. Zhang, Z. Zhao, J. Cui, D. Liang, C. Zhang, D. Zhang, H. Xie, B. Wang, and X. Bai (2025) Orion: a holistic end-to-end autonomous driving framework by vision-language instructed action generation. ICCV. Cited by: §1, §2.

[^13]: R. Gao, K. Chen, B. Xiao, L. Hong, Z. Li, and Q. Xu (2025) MagicDrive-v2: high-resolution long video generation for autonomous driving with adaptive control. In ICCV, Cited by: §2.

[^14]: R. Gao, K. Chen, E. Xie, H. Lanqing, Z. Li, D. Yeung, and Q. Xu (2024) MagicDrive: street view generation with diverse 3d geometry control. In ICLR, Cited by: §2.

[^15]: S. Gao, J. Yang, L. Chen, K. Chitta, Y. Qiu, A. Geiger, J. Zhang, and H. Li (2024) Vista: a generalizable driving world model with high fidelity and versatile controllability. In NeurIPS, Cited by: §1, §2, §3.2, §4.2, §6, §7.2.

[^16]: K. Guo, H. Liu, X. Wu, J. Pan, and C. Lv (2025) IPad: iterative proposal-centric end-to-end autonomous driving. arXiv preprint. Cited by: §2.

[^17]: S. Hamdan, C. Sima, Z. Yang, H. Li, and F. Guney (2025) ETA: efficiency through thinking ahead, a dual approach to self-driving with large models. In ICCV, Cited by: §2.

[^18]: A. Hu, L. Russell, H. Yeo, Z. Murez, G. Fedoseev, A. Kendall, J. Shotton, and G. Corrado (2023) Gaia-1: a generative world model for autonomous driving. arXiv preprint. Cited by: §1, §2.

[^19]: S. Hu, L. Chen, P. Wu, H. Li, J. Yan, and D. Tao (2022) St-p3: end-to-end vision-based autonomous driving via spatial-temporal feature learning. In ECCV, Cited by: §2.

[^20]: X. Hu, W. Yin, M. Jia, J. Deng, X. Guo, Q. Zhang, X. Long, and P. Tan (2024) DrivingWorld: constructing world model for autonomous driving via video gpt. arXiv preprint. Cited by: §2, §6.

[^21]: Y. Hu, S. Chai, Z. Yang, J. Qian, K. Li, W. Shao, H. Zhang, W. Xu, and Q. Liu (2024) Solving motion planning tasks with a scalable generative model. In ECCV, Cited by: §2.

[^22]: Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, et al. (2023) Planning-oriented autonomous driving. In CVPR, Cited by: Figure 1, Figure 1, §1, §2, Table 1, §4.1, Table 2.

[^23]: Y. Hu, Y. Guo, P. Wang, X. Chen, Y. Wang, J. Zhang, K. Sreenath, C. Lu, and J. Chen (2025) Video prediction policy: a generalist robot policy with predictive visual representations. ICML. Cited by: §2.

[^24]: B. Jaeger, D. Dauner, J. Beißwenger, S. Gerstenecker, K. Chitta, and A. Geiger (2025) Carl: learning scalable planning policies with simple rewards. arXiv preprint. Cited by: §2.

[^25]: X. Jia, Z. Yang, Q. Li, Z. Zhang, and J. Yan (2024) Bench2Drive: towards multi-ability benchmarking of closed-loop end-to-end autonomous driving. In NeurIPS, Cited by: §1.

[^26]: X. Jia, J. You, Z. Zhang, and J. Yan (2025) DriveTransformer: unified transformer for scalable end-to-end autonomous driving. In ICLR, Cited by: §1, §2.

[^27]: A. Jiang, Y. Gao, Z. Sun, Y. Wang, J. Wang, J. Chai, Q. Cao, Y. Heng, H. Jiang, Y. Dong, et al. (2025) Diffvla: vision-language guided diffusion planning for autonomous driving. arXiv preprint. Cited by: §6.

[^28]: A. Jiang, Y. Gao, Y. Wang, Z. Sun, S. Wang, Y. Heng, H. Sun, S. Tang, L. Zhu, J. Chai, et al. (2025) Irl-vla: training an vision-language-action policy via reward world model. arXiv preprint. Cited by: §2, §6.

[^29]: B. Jiang, S. Chen, Q. Xu, B. Liao, J. Chen, H. Zhou, Q. Zhang, W. Liu, C. Huang, and X. Wang (2023) Vad: vectorized scene representation for efficient autonomous driving. In ICCV, Cited by: Figure 1, Figure 1, §1, §2, §3.2, §4.1, Table 2.

[^30]: B. Jiang, S. Chen, Q. Zhang, W. Liu, and X. Wang (2025) Alphadrive: unleashing the power of vlms in autonomous driving via reinforcement learning and reasoning. arXiv preprint. Cited by: §2.

[^31]: H. Jiang, Z. Zhang, Y. Gao, Z. Sun, Y. Wang, Y. Heng, S. Wang, J. Chai, Z. Chen, H. Zhao, et al. (2025) FlowDrive: energy flow field for end-to-end autonomous driving. arXiv preprint. Cited by: §6.

[^32]: X. Jiang, Y. Ma, P. Li, L. Xu, X. Wen, K. Zhan, Z. Xia, P. Jia, X. Lang, and S. Sun (2025) TransDiffuser: end-to-end trajectory generation with decorrelated multi-modal representation for autonomous driving. arXiv preprint. Cited by: §6.

[^33]: S. Jiao, K. Qian, H. Ye, Y. Zhong, Z. Luo, S. Jiang, Z. Huang, Y. Fang, J. Miao, Z. Fu, et al. (2025) EvaDrive: evolutionary adversarial policy optimization for end-to-end autonomous driving. arXiv preprint. Cited by: §6.

[^34]: B. Li, J. Guo, H. Liu, Y. Zou, Y. Ding, X. Chen, H. Zhu, F. Tan, C. Zhang, T. Wang, et al. (2025) Uniscene: unified occupancy-centric driving scene generation. In CVPR, Cited by: §2.

[^35]: B. Li, Z. Ma, D. Du, B. Peng, Z. Liang, Z. Liu, C. Ma, Y. Jin, H. Zhao, W. Zeng, et al. (2025) OmniNWM: omniscient driving navigation world models. arXiv preprint. Cited by: §2.

[^36]: D. Li, J. Ren, Y. Wang, X. Wen, P. Li, L. Xu, K. Zhan, Z. Xia, P. Jia, X. Lang, et al. (2025) Finetuning generative trajectory model with reinforcement learning from human feedback. arXiv preprint. Cited by: §2.

[^37]: K. Li, Z. Li, S. Lan, Y. Xie, Z. Zhang, J. Liu, Z. Wu, Z. Yu, and J. M. Alvarez (2025) Hydra-mdp++: advancing end-to-end driving via expert-guided hydra-distillation. arXiv preprint. Cited by: Table 1.

[^38]: P. Li and D. Cui (2025) Navigation-guided sparse scene representation for end-to-end autonomous driving. In ICLR, Cited by: Figure 1, Figure 1, §1, §2, §3.2, §4.4.

[^39]: P. Li, Y. Zheng, Y. Wang, H. Wang, H. Zhao, J. Liu, X. Zhan, K. Zhan, and X. Lang (2025) Discrete diffusion for reflective vision-language-action models in autonomous driving. arXiv preprint. Cited by: §6.

[^40]: Y. Li, L. Fan, J. He, Y. Wang, Y. Chen, Z. Zhang, and T. Tan (2025) Enhancing end-to-end autonomous driving with latent world model. In ICLR, Cited by: Figure 1, Figure 1, §1, §2, Table 1, Table 2.

[^41]: Y. Li, S. Shang, W. Liu, B. Zhan, H. Wang, Y. Wang, Y. Chen, X. Wang, Y. An, C. Tang, et al. (2025) DriveVLA-w0: world models amplify data scaling law in autonomous driving. arXiv preprint. Cited by: §2.

[^42]: Y. Li, Y. Wang, Y. Liu, J. He, L. Fan, and Z. Zhang (2025) End-to-end driving with online trajectory evaluation via bev world model. arXiv preprint. Cited by: §1, §2, Table 1, §4.4, Table 4, Table 4.

[^43]: Y. Li, K. Xiong, X. Guo, F. Li, S. Yan, G. Xu, L. Zhou, L. Chen, H. Sun, B. Wang, et al. (2025) Recogdrive: a reinforced cognitive framework for end-to-end autonomous driving. arXiv preprint. Cited by: §2, Table 1, §4.3, §6.

[^44]: Z. Li, S. Wang, S. Lan, Z. Yu, Z. Wu, and J. M. Alvarez (2025) Hydra-next: robust closed-loop driving with open-loop training. arXiv preprint. Cited by: Table 1, §6.

[^45]: Z. Li, W. Yao, Z. Wang, X. Sun, J. Chen, N. Chang, M. Shen, J. Song, Z. Wu, S. Lan, et al. (2025) ZTRS: zero-imitation end-to-end autonomous driving with trajectory scoring. arXiv preprint. Cited by: §2.

[^46]: Z. Li, W. Yao, Z. Wang, X. Sun, J. Chen, N. Chang, M. Shen, Z. Wu, S. Lan, and J. M. Alvarez (2025) Generalized trajectory scoring for end-to-end multimodal planning. arXiv preprint. Cited by: §2, §6.

[^47]: Z. Li, Z. Yu, S. Lan, J. Li, J. Kautz, T. Lu, and J. M. Alvarez (2024) Is ego status all you need for open-loop end-to-end autonomous driving?. In CVPR, Cited by: §4.3, Table 2.

[^48]: B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang, and X. Wang (2025) DiffusionDrive: truncated diffusion model for end-to-end autonomous driving. In CVPR, Cited by: Figure 1, Figure 1, §1, §2, §3.2, Table 1, §4.2, §6.

[^49]: Y. Liao, P. Zhou, S. Huang, D. Yang, S. Chen, Y. Jiang, Y. Hu, J. Cai, S. Liu, J. Luo, et al. (2025) Genie envisioner: a unified world foundation platform for robotic manipulation. arXiv preprint. Cited by: §1, §2.

[^50]: S. Liu, Q. Liang, Z. Li, B. Li, and K. Huang (2025) GaussianFusion: gaussian-based multi-sensor fusion for end-to-end autonomous driving. arXiv preprint. Cited by: §6.

[^51]: Y. Liu, K. Zhang, Y. Li, Z. Yan, C. Gao, R. Chen, Z. Yuan, Y. Huang, H. Sun, J. Gao, et al. (2024) Sora: a review on background, technology, limitations, and opportunities of large vision models. arXiv preprint. Cited by: §1.

[^52]: I. Loshchilov (2019) Decoupled weight decay regularization. In ICLR, Cited by: §4.2.

[^53]: Y. Luo, F. Li, S. Xu, Z. Lai, L. Yang, Q. Chen, Z. Luo, Z. Xie, S. Jiang, J. Liu, et al. (2025) AdaThinkDrive: adaptive thinking via reinforcement learning for autonomous driving. arXiv preprint. Cited by: §6.

[^54]: J. Ni, Y. Guo, Y. Liu, R. Chen, L. Lu, and Z. Wu (2025) Maskgwm: a generalizable driving world model with video mask reconstruction. In CVPR, Cited by: §3.2.

[^55]: A. Prakash, K. Chitta, and A. Geiger (2021) Multi-modal fusion transformer for end-to-end autonomous driving. In CVPR, Cited by: §1, §2, §3.2, Table 1, §4.2, §6, §6, §7.1.

[^56]: X. Ren, Y. Lu, T. Cao, R. Gao, S. Huang, A. Sabour, T. Shen, T. Pfaff, J. Z. Wu, R. Chen, et al. (2025) Cosmos-drive-dreams: scalable synthetic driving data generation with world foundation models. arXiv preprint. Cited by: §2.

[^57]: K. Renz, L. Chen, E. Arani, and O. Sinavski (2025) Simlingo: vision-only closed-loop autonomous driving with language-action alignment. In CVPR, Cited by: §2.

[^58]: L. Rowe, R. Girgis, A. Gosselin, L. Paull, C. Pal, and F. Heide (2025) Scenario dreamer: vectorized latent diffusion for generating driving simulation environments. In CVPR, Cited by: §2.

[^59]: L. Russell, A. Hu, L. Bertoni, G. Fedoseev, J. Shotton, E. Arani, and G. Corrado (2025) Gaia-2: a controllable multi-view generative world model for autonomous driving. arXiv preprint. Cited by: §1, §2.

[^60]: S. Shang, Y. Chen, Y. Wang, Y. Li, and Z. Zhang (2025) DriveDPO: policy learning via safety dpo for end-to-end autonomous driving. In NeurIPS, Cited by: §2, §6.

[^61]: C. Sima, K. Chitta, Z. Yu, S. Lan, P. Luo, A. Geiger, H. Li, and J. M. Alvarez (2025) Centaur: robust end-to-end autonomous driving with test-time training. arXiv preprint. Cited by: §2, §6.

[^62]: Z. Song, C. Jia, L. Liu, H. Pan, Y. Zhang, J. Wang, X. Zhang, S. Xu, L. Yang, and Y. Luo (2025) Don’t shake the wheel: momentum-aware planning in end-to-end autonomous driving. In CVPR, Cited by: §4.3, Table 2.

[^63]: W. Sun, X. Lin, Y. Shi, C. Zhang, H. Wu, and S. Zheng (2025) SparseDrive: end-to-end autonomous driving via sparse scene representation. In ICRA, Cited by: §1, §2, §4.3, Table 2.

[^64]: Y. Tang, Z. Xu, Z. Meng, and E. Cheng (2025) Hip-ad: hierarchical and multi-granularity planning with deformable attention for autonomous driving in a single decoder. arXiv preprint. Cited by: §2.

[^65]: X. Tian, J. Gu, B. Li, Y. Liu, C. Hu, Y. Wang, K. Zhan, P. Jia, X. Lang, and H. Zhao (2024) Drivevlm: the convergence of autonomous driving and large vision-language models. In CoRL, Cited by: §2.

[^66]: X. Wang, Z. Zhu, G. Huang, X. Chen, J. Zhu, and J. Lu (2024) DriveDreamer: towards real-world-drive world models for autonomous driving. In ECCV, Cited by: §2.

[^67]: Y. Wang, J. He, L. Fan, H. Li, Y. Chen, and Z. Zhang (2024) Driving into the future: multiview visual forecasting and planning with world model for autonomous driving. In CVPR, Cited by: §2, §3.2.

[^68]: Y. Wang, J. He, L. Fan, H. Li, Y. Chen, and Z. Zhang (2024) Driving into the future: multiview visual forecasting and planning with world model for autonomous driving. In CVPR, Cited by: §1.

[^69]: J. Wei, S. Yuan, P. Li, Q. Hu, Z. Gan, and W. Ding (2024) Occllama: an occupancy-language-action generative world model for autonomous driving. arXiv preprint. Cited by: §2.

[^70]: Y. Wen, Y. Zhao, Y. Liu, F. Jia, Y. Wang, C. Luo, C. Zhang, T. Wang, X. Sun, and X. Zhang (2024) Panacea: panoramic and controllable video generation for autonomous driving. In CVPR, Cited by: §2.

[^71]: X. Weng, B. Ivanovic, Y. Wang, Y. Wang, and M. Pavone (2024) PARA-drive: parallelized architecture for real-time autonomous driving. In CVPR, Cited by: Table 1, §4.3, Table 2.

[^72]: Z. Xing, X. Zhang, Y. Hu, B. Jiang, T. He, Q. Zhang, X. Long, and W. Yin (2025) GoalFlow: goal-driven flow matching for multimodal trajectories generation in end-to-end autonomous driving. In CVPR, Cited by: §1, §2, Table 1.

[^73]: T. Yan, W. Han, X. Zhou, X. Zhang, K. Zhan, C. Xu, and J. Shen (2025) RLGF: reinforcement learning with geometric feedback for autonomous driving video generation. In NeurIPS, Cited by: §2.

[^74]: T. Yan, D. Wu, W. Han, J. Jiang, X. Zhou, K. Zhan, C. Xu, and J. Shen (2025) Drivingsphere: building a high-fidelity 4d world for closed-loop simulation. In CVPR, Cited by: §2.

[^75]: X. Yang, L. Wen, Y. Ma, J. Mei, X. Li, T. Wei, W. Lei, D. Fu, P. Cai, M. Dou, B. Shi, L. He, Y. Liu, and Y. Qiao (2025) DriveArena: a closed-loop generative simulation platform for autonomous driving. In ICCV, Cited by: §2.

[^76]: Y. Yang, J. Mei, Y. Ma, S. Du, W. Chen, Y. Qian, Y. Feng, and Y. Liu (2025) Driving in the occupancy world: vision-centric 4d occupancy forecasting and planning via world models for autonomous driving. In AAAI, Cited by: §2.

[^77]: C. Yuan, Z. Zhang, J. Sun, S. Sun, Z. Huang, C. D. W. Lee, D. Li, Y. Han, A. Wong, K. P. Tee, et al. (2024) Drama: an efficient end-to-end motion planner for autonomous driving with mamba. arXiv preprint. Cited by: Table 1.

[^78]: S. Zeng, X. Chang, M. Xie, X. Liu, Y. Bai, Z. Pan, M. Xu, and X. Wei (2025) FutureSightDrive: thinking visually with spatio-temporal cot for autonomous driving. In NeurIPS, Cited by: §2.

[^79]: B. Zhang, N. Song, X. Jin, and L. Zhang (2025) Bridging past and future: end-to-end autonomous driving with historical prediction and planning. In CVPR, Cited by: §3.3, §4.3, Table 2.

[^80]: B. Zhang, N. Song, J. Li, X. Zhu, J. Deng, and L. Zhang (2025) Future-aware end-to-end driving: bidirectional modeling of trajectory planning and scene evolution. NeurIPS. Cited by: Figure 1, Figure 1, §1, §2, Table 1, §4.4, Table 4, Table 4.

[^81]: D. Zhang, G. Wang, R. Zhu, J. Zhao, X. Chen, S. Zhang, J. Gong, Q. Zhou, W. Zhang, N. Wang, et al. (2024) SparseAD: sparse query-centric paradigm for efficient end-to-end autonomous driving. arXiv preprint. Cited by: §1, §2.

[^82]: K. Zhang, Z. Tang, X. Hu, X. Pan, X. Guo, Y. Liu, J. Huang, L. Yuan, Q. Zhang, X. Long, et al. (2025) Epona: autoregressive diffusion world model for autonomous driving. In ICCV, Cited by: §1, §2, §3.2, Table 1, §4.2, §4.3, §6, §6, §6, §6, Table 6, §7.1, §7.2.

[^83]: R. Zhao, Y. Fan, Z. Chen, F. Gao, and Z. Gao (2025) DiffE2E: rethinking end-to-end driving with a hybrid action diffusion and supervised policy. arXiv preprint. Cited by: §6.

[^84]: Z. Zhao, T. Fu, Y. Wang, L. Wang, and H. Lu (2025) From forecasting to planning: policy world model for collaborative state-action prediction. In NeurIPS, Cited by: §2.

[^85]: W. Zheng, W. Chen, Y. Huang, B. Zhang, Y. Duan, and J. Lu (2024) OccWorld: learning a 3d occupancy world model for autonomous driving. In ECCV, Cited by: §2.

[^86]: W. Zheng, R. Song, X. Guo, C. Zhang, and L. Chen (2024) GenAD: generative end-to-end autonomous driving. In ECCV, Cited by: Table 2.

[^87]: Y. Zheng, R. Liang, K. ZHENG, J. Zheng, L. Mao, J. Li, W. Gu, R. Ai, S. E. Li, X. Zhan, and J. Liu (2025) Diffusion-based planning for autonomous driving with flexible guidance. In ICLR, Cited by: §1.

[^88]: Y. Zheng, P. Yang, Z. Xing, Q. Zhang, Y. Zheng, Y. Gao, P. Li, T. Zhang, Z. Xia, P. Jia, et al. (2025) World4Drive: end-to-end autonomous driving via intention-aware physical latent world model. In ICCV, Cited by: §1, §2, Table 1, Table 2.

[^89]: X. Zhou, D. Liang, S. Tu, X. Chen, Y. Ding, D. Zhang, F. Tan, H. Zhao, and X. Bai (2025) Hermes: a unified self-driving world model for simultaneous 3d scene understanding and generation. arXiv preprint. Cited by: §2.

[^90]: Z. Zhou, T. Cai, Y. Zhao, Z. Huang, B. Zhou, and J. Ma (2025) AutoVLA: a vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. In NeurIPS, Cited by: §2.
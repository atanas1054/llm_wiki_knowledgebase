---
title: "Enhancing End-to-End Autonomous Driving with Latent World Model"
source: "https://arxiv.org/html/2406.08481v2"
author:
published:
created: 2026-10-15
description:
tags:
  - "clippings"
---
<sup>1</sup>

Yingyan Li Affiliation: Institute of Automation, Chinese Academy of Sciences (CASIA) Affiliation: New Laboratory of Pattern Recognition (NLPR) Affiliation: State Key Laboratory of Multimodal Artificial Intelligence Systems (MAIS) Affiliation: School of Future Technology, University of Chinese Academy of Sciences (UCAS)    Lue Fan Affiliation: Institute of Automation, Chinese Academy of Sciences (CASIA) Affiliation: New Laboratory of Pattern Recognition (NLPR) Affiliation: State Key Laboratory of Multimodal Artificial Intelligence Systems (MAIS)    Jiawei He Affiliation: Institute of Automation, Chinese Academy of Sciences (CASIA) Affiliation: New Laboratory of Pattern Recognition (NLPR) Affiliation: State Key Laboratory of Multimodal Artificial Intelligence Systems (MAIS)    Yuqi Wang Affiliation: Institute of Automation, Chinese Academy of Sciences (CASIA) Affiliation: New Laboratory of Pattern Recognition (NLPR) Affiliation: State Key Laboratory of Multimodal Artificial Intelligence Systems (MAIS)    Yuntao Chen Affiliation: Institute of Automation, Chinese Academy of Sciences (CASIA) Affiliation: New Laboratory of Pattern Recognition (NLPR) Affiliation: State Key Laboratory of Multimodal Artificial Intelligence Systems (MAIS)    Zhaoxiang Zhang Affiliation: Institute of Automation, Chinese Academy of Sciences (CASIA) Affiliation: New Laboratory of Pattern Recognition (NLPR) Affiliation: State Key Laboratory of Multimodal Artificial Intelligence Systems (MAIS) Affiliation: School of Future Technology, University of Chinese Academy of Sciences (UCAS)    Tieniu Tan Affiliation: Institute of Automation, Chinese Academy of Sciences (CASIA) Affiliation: New Laboratory of Pattern Recognition (NLPR) Affiliation: State Key Laboratory of Multimodal Artificial Intelligence Systems (MAIS)

###### Abstract

In autonomous driving, end-to-end planners directly utilize raw sensor data, enabling them to extract richer scene features and reduce information loss compared to traditional planners. This raises a crucial research question: how can we develop better scene feature representations to fully leverage sensor data in end-to-end driving? Self-supervised learning methods show great success in learning rich feature representations in NLP and computer vision. Inspired by this, we propose a novel self-supervised learning approach using the LAtent World model (LAW) for end-to-end driving. LAW predicts future scene features based on current features and ego trajectories. This self-supervised task can be seamlessly integrated into perception-free and perception-based frameworks, improving scene feature learning and optimizing trajectory prediction. LAW achieves state-of-the-art performance across multiple benchmarks, including real-world open-loop benchmark nuScenes, NAVSIM, and simulator-based closed-loop benchmark CARLA. The code is released at [https://github.com/BraveGroup/LAW](https://github.com/BraveGroup/LAW).

## 1 Introduction

End-to-end planners [^19] [^28] [^41] [^55] [^18] [^59] [^56] have garnered significant attention due to their distinct advantages over traditional planners. Traditional planners operate on pre-processed outputs from perception modules, such as bounding boxes and trajectories. In contrast, end-to-end planners directly utilize raw sensor data to extract scene features, minimizing information loss. This direct use of sensor data raises an important research question: how can we develop more effective scene feature representations to fully leverage the richness of sensor data in end-to-end driving?

In recent years, self-supervised learning has emerged as a powerful method for extracting comprehensive feature representations from large-scale datasets, particularly in fields like NLP [^10] and computer vision [^14]. Building on this success, we aim to enrich scene feature learning and further improve end-to-end driving performance through self-supervised learning. Traditional self-supervised methods in computer vision [^14] [^6] often focus on static, single-frame images. However, autonomous driving relies on continuous video input, so effectively using temporal data is crucial. Temporal self-supervised tasks, such as future prediction [^12] [^13], have shown promise. Traditional future prediction tasks often overlook the impact of ego actions, which play a crucial role in shaping the future in autonomous driving.

![[fig1 2.png|Refer to caption]]

Figure 1: The illustration of our self-supervised method. Traditional methods utilize perception annotations to assist with scene feature learning. In contrast, our self-supervised approach uses temporal information to guide feature learning. Pred.: predicted. Seg.: segmentation. The blue part indicates the pipeline of an end-to-end planner.

Considering the critical role of ego actions, we propose utilizing a world model to predict future states based on the current state and ego actions. While several image-based driving world models [^52] [^16] [^25] exist, they exhibit inefficiencies in enhancing scene feature representations due to their reliance on diffusion models, which may take several seconds to generate images of a future scene. To address this limitation, we introduce a latent world model designed to predict future latent features directly from the current latent features and ego actions, as depicted in Fig. 1. Specifically, given images, a visual encoder extracts scene features (current state), which are then fed into an action decoder to predict ego trajectory. Based on the current state and action, the latent world model predicts the scene feature of the future frame. During training, the predicted future features are supervised using the extracted features from the future frame. By supervising the predicted future feature, this self-supervised method jointly optimizes the current scene feature learning and ego trajectory prediction.

After introducing the concept of the latent world model, we turn our attention to its universality across various end-to-end autonomous driving frameworks. In end-to-end autonomous driving, the frameworks can generally be categorized into two types: perception-free and perception-based. Perception-free approaches [^47] [^5] [^55] bypass explicit perception tasks, relying solely on trajectory supervision. Prior work [^55] in this category typically extracts perspective-view features to predict future trajectory. In contrast, perception-based approaches [^41] [^19] [^28] [^18] incorporate perception tasks, such as detection, tracking, and map segmentation, to guide scene feature learning. These methods generally use BEV feature maps as a unified representation for these perception tasks. Our latent world model accommodates both frameworks. It can either predict perspective-view features in the perception-free setting or predict BEV features in the perception-based setting, showcasing its universality across different autonomous driving paradigms.

Experiments show that our latent world model enhances performance in both perception-free and perception-based frameworks. Furthermore, we achieve state-of-the-art performance on multiple benchmarks, including the real-world open-loop datasets nuScenes [^1] and NAVSIM [^9] (based on nuPlan [^2]), as well as the simulator-based closed-loop CARLA benchmark [^11]. These results underscore the efficacy of our approach and highlight the potential of self-supervised learning to advance end-to-end autonomous driving research. In summary, our contributions are threefold:

- Future prediction by latent world model: We introduce the LAtent World model (LAW) to predict future scene latents from current scene latents and ego trajectories. This self-supervised task jointly enhances scene representation learning and trajectory prediction in end-to-end driving.
- Cross-framework universality: LAW demonstrates universality across various common autonomous driving paradigms. It can either predict perspective-view features in the perception-free framework or predict BEV features in the perception-based framework.
- State-of-the-art performance: Our self-supervised approach achieves state-of-the-art results on the real-world open-loop nuScenes, NAVSIM, and the simulator-based close-loop CARLA benchmark.

## 2 Related Works

### 2.1 End-to-End Autonomous Driving

We divide end-to-end autonomous driving methods [^19] [^28] [^42] [^47] [^46] [^23] [^39] [^50] [^53] into two categories, perception-based methods and perception-free methods, depending on whether performing perception tasks. Perception-based methods [^3] [^41] [^24] [^45] [^18] [^43] perform multiple perception tasks simultaneously, such as detection [^32] [^21] [^30] [^31], tracking [^62] [^49], map segmentation [^19] [^28] and occupancy prediction [^51] [^22]. As a representative, UniAD [^19] integrates multiple modules to support goal-driven planning. VAD [^28] explores vectorized scene representation for planning purposes.

Perception-free end-to-end methods [^47] [^5] [^60] [^55] present a promising direction as they avoid utilizing a large number of perception annotations. Early perception-free end-to-end methods [^60] [^47] primarily relied on reinforcement learning. For instance, MaRLn [^47] designed a reinforcement learning algorithm based on implicit affordances, while LBC [^5] trained a reinforcement learning expert using privileged (ground-truth perception) information. Using trajectory data generated by the reinforcement learning expert, TCP [^55] combined a trajectory waypoint branch with a direct control branch, achieving good performance. However, perception-free end-to-end methods often suffer from inadequate scene representation capabilities. Our work aims to address this issue through the latent world model.

### 2.2 World Model in Autonomous Driving

Existing world models in autonomous driving can be categorized into two types: image-based world models and occupancy-based world models. Image-based world models [^15] [^52] [^16] aim to enrich the autonomous driving dataset through generative approaches. GAIA-1 [^16] is a generative world model that utilizes video, text, and action inputs to create realistic driving scenarios. MILE [^15] produces urban driving videos by leveraging 3D geometry as an inductive bias. Drive-WM [^52] utilizes a diffusion model to predict future images and then plans based on these predicted images. Copilot4D [^58] tokenizes sensor observations with VQVAE [^48] and then predicts the future via discrete diffusion. Another category involves occupancy-based world models [^61] [^38]. OccWorld [^61] and DriveWorld [^38] use the world model to predict the occupancy, which requires occupancy annotations. On the contrary, our proposed latent world model requires no manual annotations.

## 3 Preliminary

Vision-based End-to-end Autonomous Driving In the task of end-to-end autonomous driving, the objective is to estimate the future trajectory of the ego vehicle in the form of waypoints. Formally, let $\mathbf{I}_{t}=\{\mathbf{I}_{t}^{1},\mathbf{I}_{t}^{2},\ldots,\mathbf{I}_{t}^{N}\}$ be the set of $N$ surrounding multi-view images captured at time step $t$. We expect the model to predict a sequence of waypoints $\mathbf{W}_{t}=\{\mathbf{w}_{t}^{1},\mathbf{w}_{t}^{2},\ldots,\mathbf{w}_{t}^{M}\}$, where each waypoint $\mathbf{w}_{t}^{i}=(x_{t}^{i},y_{t}^{i})$ represents the predicted BEV position of the ego vehicle at time step $t+i$. $M$ represents the number of future positions of the ego vehicle that the model aims to predict.

World Model A world model aims to predict future states based on the current state and action. In the autonomous driving task, let $\mathbf{S}_{t}$ represent the states at time step $t$, $\mathbf{W}_{t}=\{\mathbf{w}_{t}^{1},\mathbf{w}_{t}^{2},\ldots,\mathbf{w}_{t}^{M}\}$ denote the sequence of predicted waypoints by the planner, the world model predicts state $\mathbf{S}_{t+1}$ at time step $t+1$ using $\mathbf{S}_{t}$ and $\mathbf{W}_{t}$.

## 4 Methodology

Our methodology is composed of three key components: i) Latent World Model: We utilize the latent world model to realize the self-supervised task. This model takes two inputs: latent features extracted by the visual encoder and waypoints predicted by the waypoint decoder. This task is compatible with two common frameworks. ii) Perception-free framework with perspective-view latents: it consists of the perspective-view encoder and perception-free decoder within this framework. iii) Perception-based framework with BEV latents: it contains the BEV encoder and perception-based decoder within this framework.

![[Pipeline 3.png|Refer to caption]]

Figure 2: The Overall Framework. The encoder extracts visual latents from the images, while the decoder predicts waypoints based on these latents. Our latent world model predicts future visual latents using the visual latents and waypoints of the current frame. During training, the predicted visual latents are supervised by the extracted latents from the future frame. The latent world model is compatible with both perception-free and perception-based frameworks, which differ in their encoder and decoder structures. In the perception-based framework, the supervision icon indicates that map annotations supervise the output of the map construction module, while the agent’s future trajectory supervises the output of the motion prediction module. Pred.: predicted.

### 4.1 Latent World Model

In this section, we utilize the latent world model to predict the visual latents of the future frame based on the current visual latents and waypoints.

Visual Latents and Waypoints Extraction The visual encoder processes the images from the current $t$ time step to produce the corresponding visual latent feature set

$$
\mathbf{V}_{t}=\{\mathbf{v}_{t}^{1},\mathbf{v}_{t}^{2},\ldots,\mathbf{v}_{t}^{L}\},
$$

where $L$ denotes the number of feature vectors, and each vector $\mathbf{v}_{t}^{i}\in\mathbb{R}^{D}$, with $D$ representing the number of feature channels. These feature vectors can be derived from various sources, such as a flattened image feature map or a flattened BEV feature map. Based on the $\mathbf{V}_{t}$, the waypoint decoder predicts the waypoints $\mathbf{W}_{t}=\{\mathbf{w}^{1}_{t},\mathbf{w}^{2}_{t},\ldots,\mathbf{w}^{M}_{t}\}$. $M$ represents the number of waypoints, where each waypoint $\mathbf{w}_{t}^{i}=(x_{t}^{i},y_{t}^{i})$.

Action-aware Latents Construction We produce action-aware latents by integrating visual latents and waypoints. The action-aware latents are then used as input to the latent world model. Let $M$ represent the number of waypoints, with each waypoint $\mathbf{w}_{t}^{i}=(x_{t}^{i},y_{t}^{i})$. We first reshape $\mathbf{W}_{t}$, which has the shape $[M,2]$, into a one-dimensional vector $\widetilde{\mathbf{w}}_{t}\in\mathbb{R}^{2M}$. Then, we concatenate each visual latent $\mathbf{v}_{t}^{i}$ with the waypoint vector $\widetilde{\mathbf{w}}_{t}$ along the feature channel dimension. The resulting concatenated vector is passed through an MLP to produce the action-aware latent $\mathbf{a}_{t}^{i}$, which has the same shape as $\mathbf{v}_{t}^{i}$. Formally, the action-aware latent for the $i$ -th feature vector is expressed as

$$
\mathbf{a}_{t}^{i}=\text{MLP}([\mathbf{v}_{t}^{i},\widetilde{\mathbf{w}}_{t}]),
$$

where $[\cdot,\cdot]$ denotes the concatenating operation. The full set of action-aware latents is denoted as $\mathbf{A}_{t}=\{\mathbf{a}_{t}^{1},\mathbf{a}_{t}^{2},\ldots,\mathbf{a}_{t}^{L}\}$, where $L$ denotes the number of feature vectors.

Future Latent Prediction The latent world model utilizes the action-aware latents to predict the future visual latents as follows. Given $\mathbf{A}_{t}$, we predict visual latents $\mathbf{\hat{V}}_{t+1}$ of the frame $t+1$ by the latent world model:

$$
\mathbf{\hat{V}}_{t+1}=\text{LatentWorldModel}(\mathbf{A}_{t}).
$$

The network architecture of the latent world model consists of transformer blocks. Each block contains a self-attention and feed-forward module. The self-attention is performed across the latent feature vectors.

Future Latent Supervision During training, we extract the visual latents $\mathbf{V}_{t+1}=\{\mathbf{v}_{t+1}^{1},\ldots,\mathbf{v}_{t+1}^{L}$ from the images of frame $t+1$, which are used as the ground truth to supervise the $\mathbf{\hat{V}}_{t+1}$ using a Mean Squared Error (MSE) loss function as

$$
\mathcal{L}_{\text{latent}}=\frac{1}{L}\sum_{i=1}^{L}\|\mathbf{\hat{v}}_{t+1}^{i}-\mathbf{v}_{t+1}^{i}\|_{2}.
$$

The latent world model is compatible with both perception-free and perception-based frameworks. In the following sections, we detail the implementation of these two frameworks.

### 4.2 Perception-free Framework with Perspective-view Latents

First, we introduce our perception-free framework. Previous perception-free frameworks [^55] typically employ a perspective-view encoder for visual latent extraction and a perception-free decoder for waypoint prediction. Our framework is built upon this established paradigm.

Perspective-view Encoder In the perspective-view encoder, we produce visual latents based on multi-view images. Initially, multi-view images are processed by an image backbone to obtain their corresponding image features. Following PETR [^34], we generate 3D position embeddings for these image features, which are then added to the image features to uniquely distinguish each feature vector. The enriched image features are denoted as $\mathbf{F}=\{\mathbf{f}^{1},\mathbf{f}^{2},\ldots,\mathbf{f}^{N}\}$, where $N$ represents the number of views. The shape of $\mathbf{f}^{i}$ is $[H,W,D]$, where $H,W$ represents the height and width of the image feature map and $D$ is the number of feature channels.

To encode the image features into high-level visual latents suitable for planning, we apply a view attention mechanism. To be specific, for $N$ views, there are $N$ corresponding learnable view queries $\mathbf{Q}_{\text{view}}=\{\mathbf{q}_{\text{view}}^{1},\mathbf{q}_{\text{view}}^{2},\ldots,\mathbf{q}_{\text{view}}^{N}\}$. Each view query $\mathbf{q}^{i}_{\text{view}}$ undergoes a cross-attention with its corresponding image feature $\mathbf{f}^{i}$, resulting in $N$ visual latent $\mathbf{V}_{\text{pf}}=\{\mathbf{v}^{1}_{\text{pf}},\mathbf{v}^{2}_{\text{pf}},\ldots,\mathbf{v}^{N}_{\text{pf}}\}$, where the subscript “pf” stands for perception-free. Formally,

$$
\mathbf{v}^{i}_{\text{pf}}=\text{CrossAttention}(\mathbf{q}_{\text{view}}^{i},\mathbf{f}^{i},\mathbf{f}^{i}),
$$

where $\mathbf{f}^{i}$ serves as the key and value of the cross attention. $\mathbf{V}_{\text{pf}}$ are then used as input for the perception-free decoder, which we describe next.

Perception-free Decoder The perception-free decoder deocdes waypoint from $\mathbf{V}_{\text{pf}}$. Specifically, we initialize $M$ waypoint queries, $\mathbf{Q}_{\text{wp}}=\{\mathbf{q}_{\text{wp}}^{1},\mathbf{q}_{\text{wp}}^{2},\ldots,\mathbf{q}_{\text{wp}}^{M}\}$, where each query is a learnable embedding. These waypoint queries interact with $\mathbf{V}_{\text{pf}}$ through a cross-attention mechanism. The updated waypoint queries are then passed through an MLP head to predict the waypoints $\mathbf{W}=\{\mathbf{w}^{1},\mathbf{w}^{2},\ldots,\mathbf{w}^{M}\}$, which is formulated as

$$
\mathbf{W}=\text{MLP}(\text{CrossAttention}(\mathbf{Q}_{\text{wp}},\mathbf{V}_{\text{pf}},\mathbf{V}_{\text{pf}})).
$$

Perception-free Supervision In the perception-free framework, we rely solely on ground truth waypoints for supervision, as no additional perception annotations are provided. We employ an L1 loss to measure the discrepancy between the predicted waypoints $\mathbf{W}$ and the ground truth waypoints $\mathbf{\tilde{W}}=\{\mathbf{\tilde{w}}^{1},\mathbf{\tilde{w}}^{2},\ldots,\mathbf{\tilde{w}}^{M}\}$ as

$$
\mathcal{L}_{\text{waypoint}}=\frac{1}{M}\sum_{j=1}^{M}\|\mathbf{w}_{t}^{j}-\mathbf{\tilde{w}}_{t}^{j}\|_{1},
$$

Thus, the final loss for the perception-free framework is

$$
\mathcal{L}_{\text{pf}}=\mathcal{L}_{\text{latent}}+\mathcal{L}_{\text{waypoint}}.
$$

### 4.3 Perception-based Framework with BEV Latents

Our latent world model is also compatible with perception-based frameworks, which commonly utilize BEV feature maps for perception tasks. We adhere to this paradigm and the perception-based framework is composed of two key components: a BEV encoder and a perception-based decoder. The BEV encoder generates BEV feature maps from images, while the perception-based decoder uses these maps for perception tasks such as motion prediction and map construction. The final waypoints are then predicted based on the outputs of these perception tasks.

BEV Encoder We follow [^32] to encode the BEV feature map. First, we encode the image features using a backbone network. Then, a set of BEV queries projects these image features into BEV features. The resulting BEV feature map is flattened into a shape of $[K,D]$, where $K$ represents the number of feature vectors in the BEV feature map and $D$ is the number of feature channels. The flattened features are denoted as $\mathbf{V}_{\text{pb}}=\{\mathbf{v}^{1}_{\text{pb}},\mathbf{v}^{2}_{\text{pb}},\ldots,\mathbf{v}^{K}_{\text{pb}}\}$, where the subscript “pb”refers to perception-based.

Perception-based Decoder Following [^28], the decoder predicts waypoints with the help of perception tasks, namely motion prediction and map construction. For motion prediction, $N_{\text{agent}}$ learnable queries interact with $\mathbf{V}_{\text{pb}}$ via cross-attention to generate agent features $\mathbf{F}_{\text{agent}}$. $\mathbf{F}_{\text{agent}}$ are then used to predict agent trajectories. Similarly, for map construction, $N_{\text{map}}$ learnable queries perform cross-attention with $\mathbf{V}_{\text{pb}}$ to extract map features $\mathbf{F}_{\text{map}}$. $\mathbf{F}_{\text{map}}$ are then used to predict map vectors. Finally, the learnable waypoint queries perform cross-attention with $\mathbf{F}_{\text{agent}}$ and $\mathbf{F}_{\text{map}}$. The output is then passed through an MLP head to predict the waypoints.

Perception-based Supervision The perception-based framework uses the same waypoint supervision as in equation 6. In addition, it includes losses from perception tasks as

$$
\mathcal{L}_{\text{perception}}=\mathcal{L}_{\text{agent}}+\mathcal{L}_{\text{map}}.
$$

Here, $\mathcal{L}_{\text{agent}}$ is s the loss for motion prediction and $\mathcal{L}_{\text{map}}$ is the loss for map construction, as defined in [^28]. The final loss for the perception-based framework is

$$
\mathcal{L}_{\text{pb}}=\mathcal{L}_{\text{latent}}+\mathcal{L}_{\text{waypoint}}+\mathcal{L}_{\text{perception}}.
$$

## 5 Experiments

### 5.1 Benchmarks

nuScenes [^1] The nuScenes dataset contains 1,000 driving scenes. In line with previous works [^18] [^20] [^28], we use L2 displacement error and collision rate as comprehensive metrics to evaluate planning performance. L2 displacement error measures the L2 distance between the predicted and ground truth trajectories, while collision rate quantifies the frequency of collisions with other objects along the predicted trajectory.

NAVSIM [^9] We conducted further experiments using the NAVSIM benchmark, as the nuScenes dataset proved to be overly simplistic. The NAVSIM dataset [^9] is built on OpenScene [^8], which provides 120 hours of driving logs condensed from the nuPlan dataset [^2]. NAVSIM enhances OpenScene by resampling the data to reduce the occurrence of simple scenarios, such as straight-line driving. As a result, traditional ego status modeling becomes inadequate under the NAVSIM benchmark. NAVSIM evaluates model performance using the predictive driver model score (PDMS), which is calculated based on five factors: no at-fault collision (NC), drivable area compliance (DAC), time-to-collision (TTC), comfort (Comf.) and ego progress (EP).

CARLA [^11] Closed-loop evaluation is essential to autonomous driving as it constantly updates the sensor inputs based on the driving actions. For the closed-loop benchmark, the training dataset is collected from the CARLA [^11] simulator (version 0.9.10.1) using the teacher model Roach [^60] following [^55] [^26], resulting in 189K frames. We use the widely-used Town05 Long benchmark [^26] [^44] [^15] to assess the closed-loop driving performance. For metric, Route Completion (RC) represents the percentage of the route completed by the autonomous driving model. Infraction Score (IS) quantifies the number of infractions as well as violations of traffic rules. A higher Infraction Score indicates better adherence to safe driving practices. Driving Score (DS) is the primary metric used to evaluate overall performance. It is calculated as the product of Route Completion and Infraction Score.

### 5.2 Implementation Details

nuScenes Benchmark We implement both perception-free and perception-based frameworks. In the perception-free framework, Swin-Transformer-Tiny (Swin-T) [^35] is used as the backbone. Input images are resized to 800×320. We adopt a Cosine Annealing learning rate schedule [^36], starting at 5e-5. The model is trained using the AdamW optimizer [^37] with a weight decay of 0.01, batch size 8, and 12 epochs across 8 A6000 GPUs. For the perception-based framework, following [^28], we train the model in two stages. In the first stage, we train the encoder and perception head using only perception loss for 48 epochs. In the second stage, we introduce waypoint and latent prediction losses for training another 12 epochs. The network architecture of the latent world model utilizes deformable self-attention for improved convergence.

NAVSIM Benchmark The perception-free framework is implemented on NAVSIM. Specifically, We employ a ResNet-34 backbone, training for 20 epochs in line with [^41] to ensure a fair comparison. Input images are resized to 640×320. The Adam optimizer is used with a learning rate of 1e-4 and a batch size of 32.

CARLA Benchmark We follow [^55] to implement a perception-free framework on CARLA. To be specific, we use ResNet-34 as the backbone and employ the TCP head [^55] as in [^26]. Input images are resized to 900×256. The Adam optimizer is used with a learning rate of 1e-4 and weight decay of 1e-7. The model is trained for 60 epochs with a batch size of 128. After 30 epochs, the learning rate is halved.

### 5.3 Comparison with State-of-the-art Methods

For the nuScenes benchmark, we compare our proposed framework with several state-of-the-art methods, including BEV-Planner [^33] and VAD [^28]. The results are summarized in Table 1. Our perception-free framework demonstrates competitive performance, while the perception-based framework achieves state-of-the-art results in both L2 displacement and collision rates. For the NAVSIM benchmark, detailed in Table 2, our method achieves state-of-the-art results in overall PDMS. For the CARLA benchmark, as shown in Table 3, our proposed method outperforms all existing methods. Notably, our perception-free approach surpasses previous leading methods such as ThinkTwice [^27] and DriveAdapter [^26], which incorporate extensive supervision from depth estimation, semantic segmentation, and map segmentation.

Table 1: Performance on the nuScenes [^1]. The overall collision results are computed by the traditional computation way used in [^28]. ${\ddagger}$: The collision results are computed by the way in [^33]. We do not utilize the historical ego status information.

<table><tbody><tr><td rowspan="2">Method</td><td colspan="4">L2 (m) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td colspan="4">Collision (%) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td></tr><tr><td>NMP <sup><a href="#fn:57">57</a></sup></td><td>-</td><td>-</td><td>2.31</td><td>-</td><td>-</td><td>-</td><td>1.92</td><td>-</td></tr><tr><td>SA-NMP <sup><a href="#fn:57">57</a></sup></td><td>-</td><td>-</td><td>2.05</td><td>-</td><td>-</td><td>-</td><td>1.59</td><td>-</td></tr><tr><td>FF <sup><a href="#fn:17">17</a></sup></td><td>0.55</td><td>1.20</td><td>2.54</td><td>1.43</td><td>0.06</td><td>0.17</td><td>1.07</td><td>0.43</td></tr><tr><td>EO <sup><a href="#fn:29">29</a></sup></td><td>0.67</td><td>1.36</td><td>2.78</td><td>1.60</td><td>0.04</td><td>0.09</td><td>0.88</td><td>0.33</td></tr><tr><td>ST-P3 <sup><a href="#fn:18">18</a></sup></td><td>1.33</td><td>2.11</td><td>2.90</td><td>2.11</td><td>0.23</td><td>0.62</td><td>1.27</td><td>0.71</td></tr><tr><td>UniAD <sup><a href="#fn:19">19</a></sup></td><td>0.48</td><td>0.96</td><td>1.65</td><td>1.03</td><td>0.05</td><td>0.17</td><td>0.71</td><td>0.31</td></tr><tr><td>VAD <sup><a href="#fn:28">28</a></sup></td><td>0.41</td><td>0.70</td><td>1.05</td><td>0.72</td><td>0.07</td><td>0.17</td><td>0.41</td><td>0.22</td></tr><tr><td>BEV-Planner <sup><a href="#fn:33">33</a></sup></td><td>0.30</td><td>0.52</td><td>0.83</td><td>0.55</td><td>0.10 <math><semantics><mo>‡</mo> <annotation>{\ddagger}</annotation></semantics></math></td><td>0.37 <math><semantics><mo>‡</mo> <annotation>{\ddagger}</annotation></semantics></math></td><td>1.30 <math><semantics><mo>‡</mo> <annotation>{\ddagger}</annotation></semantics></math></td><td>0.59 <math><semantics><mo>‡</mo> <annotation>{\ddagger}</annotation></semantics></math></td></tr><tr><td>LAW (perception-free)</td><td>0.26</td><td>0.57</td><td>1.01</td><td>0.61</td><td>0.14</td><td>0.21</td><td>0.54</td><td>0.30</td></tr><tr><td>LAW (perception-based)</td><td>0.24</td><td>0.46</td><td>0.76</td><td>0.49</td><td>0.08</td><td>0.10</td><td>0.39</td><td>0.19</td></tr></tbody></table>

Table 2: Performance on NAVSIM test set. NC: no at-fault collision. DAC: drivable area compliance. TTC: time-to-collision. Comf.: comfort. EP: ego progress. PDMS: the predictive driver model score. LAW is in the perception-free setting.

| Method | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf. $\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| Human | 100 | 100 | 100 | 99.9 | 87.5 | 94.8 |
| Constant Velocity | 69.9 | 58.8 | 49.3 | 100 | 49.3 | 21.6 |
| Ego Status MLP | 93.0 | 77.3 | 83.6 | 100 | 62.8 | 65.6 |
| TransFuser [^41] | 97.7 | 92.8 | 92.8 | 100 | 79.2 | 84.0 |
| UniAD [^19] | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| PARA-Drive [^54] | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 |
| LAW | 96.4 | 95.4 | 88.7 | 99.9 | 81.7 | 84.6 |

Table 3: Performance on Town05 Long benchmark on CARLA. Expert: Imitation learning from the driving trajectories of a privileged expert. Seg.: semantic segmentation. Map.: BEV map segmentation. Dep.: depth estimation. Det.: 3D object detection. *Latent Prediction*: our proposed self-supervised task. RC: route completion. IS: infraction score. DS: driving score. LAW is in the perception-free setting.

| Method | Supervision | RC $\uparrow$ | IS $\uparrow$ | DS $\uparrow$ |
| --- | --- | --- | --- | --- |
| CILRS [^7] | Expert | 10.3 $\pm$ 0.0 | 0.75 $\pm$ 0.05 | 7.8 $\pm$ 0.3 |
| LBC [^5] | Expert | 31.9 $\pm$ 2.2 | 0.66 $\pm$ 0.02 | 12.3 $\pm$ 2.0 |
| Transfuser [^41] | Expert, Dep., Seg., Map., Det. | 47.5 $\pm$ 5.3 | 0.77 $\pm$ 0.04 | 31.0 $\pm$ 3.6 |
| Roach [^60] | Expert | 96.4 $\pm$ 2.1 | 0.43 $\pm$ 0.03 | 41.6 $\pm$ 1.8 |
| LAV [^4] | Expert, Seg., Map., Det. | 69.8 $\pm$ 2.3 | 0.73 $\pm$ 0.02 | 46.5 $\pm$ 2.3 |
| TCP [^55] | Expert | 80.4 $\pm$ 1.5 | 0.73 $\pm$ 0.02 | 57.2 $\pm$ 1.5 |
| MILE [^15] | Expert, Map., Det. | 97.4 $\pm$ 0.8 | 0.63 $\pm$ 0.03 | 61.1 $\pm$ 3.2 |
| ThinkTwice [^27] | Expert, Dep., Seg., Det. | 95.5 $\pm$ 2.0 | 0.69 $\pm$ 0.05 | 65.0 $\pm$ 1.7 |
| DriveAdapter [^26] | Expert, Map., Det. | 94.4 $\pm$ - | 0.72 $\pm$ - | 65.9 $\pm$ - |
| Interfuser [^44] | Expert, Map., Det. | 95.0 $\pm$ 2.9 | \- | 68.3 $\pm$ 1.9 |
| LAW | Expert, *Latent Prediction* | 97.8 $\pm 0.9$ | 0.72 $\pm$ 0.03 | 70.1 $\pm$ 2.6 |

### 5.4 Ablation Study

All experiments are conducted within the perception-free framework unless otherwise specified.

Ablation Study on Latent World Model In this ablation study, we assess the effectiveness of our proposed latent world model. For the nuScenes benchmark, the results are shown in Table 1. We ablate the latent prediction task in both the perception-free and perception-based frameworks, and further investigate the contribution of each input to the latent world model. The findings demonstrate that accurate future latent predictions depend on incorporating driving actions, supporting the validity of the latent world model. We also present ablation studies on NAVSIM and CARLA, as detailed in Table 5. In NAVSIM, we observed a significant improvement in PDMS, mainly driven by the enhancements in drivable area compliance (DAC) and Ego progress (EP) metrics. This suggests that our self-supervised task effectively enhances the quality of the driving trajectory. Similarly, in CARLA, we observed notable improvements in the Driving Score.

Table 4: Effectiveness of latent prediction on nuScenes benchmark. The latent world model receives two types of inputs: visual latents and predicted trajectory. No input refers to not utilizing the world model. Pred.: predicted. Traj.: trajectory. Avg.: average.

<table><tbody><tr><td rowspan="2">Framework</td><td colspan="2">Input of Latent World Model</td><td colspan="4">L2 (m) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td colspan="4">Collision (%) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><td>Visual Latent</td><td>Pred. Traj.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td></tr><tr><td rowspan="3">Perception-free</td><td>-</td><td>-</td><td>0.32</td><td>0.67</td><td>1.14</td><td>0.71</td><td>0.20</td><td>0.30</td><td>0.73</td><td>0.41</td></tr><tr><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td><td>-</td><td>0.30</td><td>0.64</td><td>1.12</td><td>0.68</td><td>0.18</td><td>0.27</td><td>0.66</td><td>0.37</td></tr><tr><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td><td>0.26</td><td>0.57</td><td>1.01</td><td>0.61</td><td>0.14</td><td>0.21</td><td>0.54</td><td>0.30</td></tr><tr><td rowspan="3">Perception-based</td><td>-</td><td>-</td><td>0.30</td><td>0.52</td><td>0.80</td><td>0.54</td><td>0.09</td><td>0.17</td><td>0.48</td><td>0.25</td></tr><tr><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td><td>-</td><td>0.27</td><td>0.49</td><td>0.80</td><td>0.52</td><td>0.08</td><td>0.12</td><td>0.42</td><td>0.21</td></tr><tr><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td><td>0.24</td><td>0.46</td><td>0.76</td><td>0.49</td><td>0.08</td><td>0.10</td><td>0.39</td><td>0.19</td></tr></tbody></table>

Table 5: Ablation study on latent prediction on NAVSIM and CARLA benchmark. NC: no at-fault collision. DAC: drivable area compliance. TTC: time-to-collision. Comf.: comfort. EP: ego progress. PDMS: the predictive driver model score. RC: route Completion. IS: infraction Score. DS: driving Score. LAW is in the perception-free setting.

<table><tbody><tr><td rowspan="2">Latent Prediction</td><td colspan="6">NAVSIM</td><td colspan="3">CARLA</td></tr><tr><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Comf. <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>RC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>IS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>94.4</td><td>89.4</td><td>84.8</td><td>100.0</td><td>75.1</td><td>77.5</td><td>98.6 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.8</td><td>0.68 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.02</td><td>67.9 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 2.1</td></tr><tr><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td><td>96.4</td><td>95.4</td><td>88.7</td><td>99.9</td><td>81.7</td><td>84.6</td><td>97.8 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.9</td><td>0.72 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.03</td><td>70.1 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 2.6</td></tr></tbody></table>

The Time Horizon of Latent World Model In this experiment, the world model predicts latent features at three distinct future time horizons: 0.5 seconds, 1.5 seconds, and 3.0 seconds. This corresponds to the first, third, and sixth future frames from the current frame, given that keyframes occur every 0.5 seconds in the nuScenes dataset. The results, displayed in Table 6, show that the model achieves the best performance at the 1.5-second horizon. In comparison, the 0.5-second interval typically presents scenes with minimal changes, providing insufficient dynamic content to improve feature learning. In contrast, the 3.0-second interval often presents scenes that may significantly differ from the current frame, making accurate future predictions more challenging. Moreover, we observe that predicting latents 10 seconds into the future completely diminishes the gains provided by the world model, suggesting that predicting features too far into the future is ineffective. This conclusion aligns with observations from MAE [^14], where both excessively low and high mask ratios negatively impact the ability of the network.

Table 6: Different time horizons for latent prediction. The time intervals of 0.5, 1.5, 3.0, and 10.0 seconds correspond to the first, third, sixth, and twentieth future frames from the current frame, as keyframes occur every 0.5 seconds in the nuScenes dataset.

<table><tbody><tr><td rowspan="2">Time Horizon</td><td colspan="4">L2 (m) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td colspan="4">Collision (%) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td></tr><tr><td>0.5s</td><td>0.26</td><td>0.57</td><td>1.01</td><td>0.61</td><td>0.14</td><td>0.21</td><td>0.54</td><td>0.30</td></tr><tr><td>1.5s</td><td>0.26</td><td>0.54</td><td>0.93</td><td>0.58</td><td>0.14</td><td>0.17</td><td>0.45</td><td>0.25</td></tr><tr><td>3.0s</td><td>0.28</td><td>0.59</td><td>1.01</td><td>0.63</td><td>0.13</td><td>0.20</td><td>0.48</td><td>0.27</td></tr><tr><td>10.0s</td><td>0.33</td><td>0.67</td><td>1.16</td><td>0.72</td><td>0.24</td><td>0.28</td><td>0.76</td><td>0.43</td></tr></tbody></table>

Network Architecture of Latent World Model To validate the impact of the network architecture of the latent world model, we conduct experiments as shown in Table 7. Firstly, it is evident that a single-layer neural network, represented as Linear Projection, is not adequate for fulfilling the functions of the world model, resulting in significantly degraded performance. The two-layer MLP shows considerable improvement in performance. However, it lacks the capability to facilitate interactions among different latent vectors. Therefore, we use the stacked transformer blocks as our default network architecture, which achieves the best results among the tested architectures. This indicates that interactions between feature vectors from different positions are important.

Table 7: Different network architecture of the latent world model. Linear Projection means a single-layer network.

<table><tbody><tr><td rowspan="2">Architecture</td><td colspan="4">L2 (m) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td colspan="4">Collision (%) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td></tr><tr><td>Linear Projection</td><td>0.31</td><td>0.65</td><td>1.14</td><td>0.70</td><td>0.26</td><td>0.34</td><td>0.66</td><td>0.42</td></tr><tr><td>Two-layer MLP</td><td>0.27</td><td>0.58</td><td>1.07</td><td>0.64</td><td>0.17</td><td>0.23</td><td>0.59</td><td>0.33</td></tr><tr><td>Transformer Blocks</td><td>0.26</td><td>0.57</td><td>1.01</td><td>0.61</td><td>0.14</td><td>0.21</td><td>0.54</td><td>0.30</td></tr></tbody></table>

### 5.5 Visualization

Figure 3 compares the results of LAW in the perception-based setting with VAD [^28]. Leveraging our latent world model, our approach acquires more comprehensive scene representations.

![[vis_main_comparison.png|Refer to caption]]

Figure 3: Visualization. This figure compares LAW in the perception-based setting with VAD 28. On the right side of this figure, we display the results of ego trajectory prediction, agent motion prediction, and map construction in BEV. As indicated by the red circles, our method captures more crucial scene information, which VAD overlooks. Consequently, VAD predicts a forward trajectory that results in a rear-end collision, as highlighted by the yellow circles.

## 6 Conclusion

In conclusion, we present the latent world model to predict future features from current features and ego trajectories, which is a novel self-supervised learning method for end-to-end autonomous driving. This method jointly enhances scene representation learning and ego trajectory prediction. Our approach demonstrates universality by accommodating both perception-free and perception-based frameworks, predicting perspective-view features and BEV features respectively. We achieve state-of-the-art results on benchmarks like nuScenes, NAVSIM, and CARLA.

## 7 Acknowledgments

This work was supported by National Science and Technology Major Project (2022ZD0116500). This work was also supported in part by the National Key R&D Program of China (No. 2022ZD0116500), the National Natural Science Foundation of China (No. U21B2042, No. 62320106010), and in part by the 2035 Innovation Program of CAS.

## References

## Appendix A Appendix

### A.1 Predicting multiple features using multiple input frames

Predicting multiple future features To better investigate the ability of our latent world model, we utilize the latent world model to predict multiple future frame latents, with the results presented in Table 8. We conduct this experiment using only the front-view camera to facilitate fast training. In detail, the future frame latents are predicted in an auto-regressive manner. For example, we first predicted the latent for 1.5 seconds into the future, then used this predicted latent to further predict the latent for 3 seconds into the future. The latent world model shares the same weights throughout this process. The results demonstrate that predicting multiple future latents further improves performance.

Table 8: Predicting multiple future latents in an auto-regressive manner.

<table><tbody><tr><td rowspan="2">Predicted Future</td><td colspan="4">L2 (m) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td colspan="4">Collision (%) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td></tr><tr><td>1.5s</td><td>0.34</td><td>0.69</td><td>1.17</td><td>0.73</td><td>0.12</td><td>0.22</td><td>0.63</td><td>0.32</td></tr><tr><td>1.5s <math><semantics><mo>→</mo> <annotation>\rightarrow</annotation></semantics></math> 3s</td><td>0.31</td><td>0.65</td><td>1.12</td><td>0.69</td><td>0.11</td><td>0.19</td><td>0.57</td><td>0.29</td></tr></tbody></table>

Predicting multiple future features using multiple input frames Building on the previous section, we further conduct experiments to predict multiple future frame latents while incorporating multiple input frame latents to leverage temporal information more effectively. Specifically, we adopted a two-stage training paradigm for improved convergence, inspired by SOLOFusion [^40]. In the first stage, we trained the model using single input frame latents for 12 epochs. This corresponds to the model denoted as "1.5s → 3s" in Table 8. In the second stage, we fine-tuned the model for an additional 6 epochs, now using two input frame latents, from 0s and 1.5s ago. The results are summarized in Table 9. The baseline (first row) represents the model fine-tuned using only single input frame latents. In contrast, the second row corresponds to the model fine-tuned with two input frame latents. The latter achieves significantly better performance. This highlights the crucial role of temporal information in autonomous driving.

Table 9: Predicting future latents with multiple history frame inputs.

<table><tbody><tr><td rowspan="2">Predicted Future</td><td rowspan="2">Input Frames</td><td colspan="4">L2 (m) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td colspan="4">Collision (%) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td></tr><tr><td>1.5s <math><semantics><mo>→</mo> <annotation>\rightarrow</annotation></semantics></math> 3s</td><td>0s</td><td>0.30</td><td>0.64</td><td>1.09</td><td>0.68</td><td>0.14</td><td>0.23</td><td>0.62</td><td>0.33</td></tr><tr><td>1.5s <math><semantics><mo>→</mo> <annotation>\rightarrow</annotation></semantics></math> 3s</td><td>-1.5s, 0s</td><td>0.26</td><td>0.51</td><td>0.87</td><td>0.55</td><td>0.08</td><td>0.09</td><td>0.33</td><td>0.17</td></tr></tbody></table>

### A.2 More Visualization

In the appendix, we provide more visualization figures. We also provide a demo based on the CARLA simulator in the supplementary materials.

![[vis_supp_1.png|Refer to caption]]

Figure 4: Visualization. As shown in the red circle, our map construction results are noticeably better than those of VAD.

![[vis_supp_2.png|Refer to caption]]

Figure 5: Visualization. As shown in the red circle, our agent motion prediction results are noticeably better than those of VAD.

![[vis_supp_3.png|Refer to caption]]

Figure 6: Visualization. As shown in the red circle, our map construction and agent motion prediction results are noticeably better than those of VAD, especially in heavily occluded and crowded conditions.

[^1]: Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In *CVPR*, 2020.

[^2]: Holger Caesar, Juraj Kabzan, Kok Seang Tan, Whye Kit Fong, Eric Wolff, Alex Lang, Luke Fletcher, Oscar Beijbom, and Sammy Omari. nuplan: A closed-loop ml-based planning benchmark for autonomous vehicles. *arXiv preprint arXiv:2106.11810*, 2021.

[^3]: Sergio Casas, Abbas Sadat, and Raquel Urtasun. Mp3: A unified model to map, perceive, predict and plan. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 14403–14412, 2021.

[^4]: Dian Chen and Philipp Krähenbühl. Learning from all vehicles. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 17222–17231, 2022.

[^5]: Dian Chen, Brady Zhou, Vladlen Koltun, and Philipp Krähenbühl. Learning by cheating. In *Conference on Robot Learning*, pp. 66–75. PMLR, 2020a.

[^6]: Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In *International conference on machine learning*, pp. 1597–1607. PMLR, 2020b.

[^7]: Felipe Codevilla, Eder Santana, Antonio M López, and Adrien Gaidon. Exploring the limitations of behavior cloning for autonomous driving. In *ICCV*, 2019.

[^8]: OpenScene Contributors. Openscene: The largest up-to-date 3d occupancy prediction benchmark in autonomous driving, 2023.

[^9]: Daniel Dauner, Marcel Hallgarten, Tianyu Li, Xinshuo Weng, Zhiyu Huang, Zetong Yang, Hongyang Li, Igor Gilitschenski, Boris Ivanovic, Marco Pavone, et al. Navsim: Data-driven non-reactive autonomous vehicle simulation and benchmarking. *arXiv preprint arXiv:2406.15349*, 2024.

[^10]: Jacob Devlin. Bert: Pre-training of deep bidirectional transformers for language understanding. *arXiv preprint arXiv:1810.04805*, 2018.

[^11]: Alexey Dosovitskiy, German Ros, Felipe Codevilla, Antonio Lopez, and Vladlen Koltun. Carla: An open urban driving simulator. In *CoRL*, 2017.

[^12]: Tengda Han, Weidi Xie, and Andrew Zisserman. Video representation learning by dense predictive coding. In *Proceedings of the IEEE/CVF international conference on computer vision workshops*, pp. 0–0, 2019.

[^13]: Tengda Han, Weidi Xie, and Andrew Zisserman. Memory-augmented dense predictive coding for video representation learning. In *European conference on computer vision*, pp. 312–329. Springer, 2020.

[^14]: Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In *CVPR*, pp. 16000–16009, 2022.

[^15]: Anthony Hu, Gianluca Corrado, Nicolas Griffiths, Zak Murez, Corina Gurau, Hudson Yeo, Alex Kendall, Roberto Cipolla, and Jamie Shotton. Model-based imitation learning for urban driving. *NeurIPS*, 2022a.

[^16]: Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, and Gianluca Corrado. Gaia-1: A generative world model for autonomous driving. *arXiv preprint arXiv:2309.17080*, 2023a.

[^17]: Peiyun Hu, Aaron Huang, John Dolan, David Held, and Deva Ramanan. Safe local motion planning with self-supervised freespace forecasting. In *CVPR*, 2021.

[^18]: Shengchao Hu, Li Chen, Penghao Wu, Hongyang Li, Junchi Yan, and Dacheng Tao. St-p3: End-to-end vision-based autonomous driving via spatial-temporal feature learning. In *ECCV*, 2022b.

[^19]: Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai Wang, et al. Goal-oriented autonomous driving. *arXiv preprint arXiv:2212.10156*, 2022c.

[^20]: Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai Wang, Lewei Lu, Xiaosong Jia, Qiang Liu, Jifeng Dai, Yu Qiao, and Hongyang Li. Planning-oriented autonomous driving. In *CVPR*, 2023b.

[^21]: Junjie Huang, Guan Huang, Zheng Zhu, and Dalong Du. Bevdet: High-performance multi-camera 3d object detection in bird-eye-view. *arXiv preprint arXiv:2112.11790*, 2021.

[^22]: Yuanhui Huang, Wenzhao Zheng, Yunpeng Zhang, Jie Zhou, and Jiwen Lu. Tri-perspective view for vision-based 3d semantic occupancy prediction. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 9223–9232, 2023.

[^23]: Jyh-Jing Hwang, Runsheng Xu, Hubert Lin, Wei-Chih Hung, Jingwei Ji, Kristy Choi, Di Huang, Tong He, Paul Covington, Benjamin Sapp, et al. Emma: End-to-end multimodal model for autonomous driving. *arXiv preprint arXiv:2410.23262*, 2024.

[^24]: Bernhard Jaeger, Kashyap Chitta, and Andreas Geiger. Hidden biases of end-to-end driving models. In *ICCV*, 2023.

[^25]: Fan Jia, Weixin Mao, Yingfei Liu, Yucheng Zhao, Yuqing Wen, Chi Zhang, Xiangyu Zhang, and Tiancai Wang. Adriver-i: A general world model for autonomous driving. *arXiv preprint arXiv:2311.13549*, 2023a.

[^26]: Xiaosong Jia, Yulu Gao, Li Chen, Junchi Yan, Patrick Langechuan Liu, and Hongyang Li. Driveadapter: Breaking the coupling barrier of perception and planning in end-to-end autonomous driving. In *ICCV*, 2023b.

[^27]: Xiaosong Jia, Penghao Wu, Li Chen, Jiangwei Xie, Conghui He, Junchi Yan, and Hongyang Li. Think twice before driving: towards scalable decoders for end-to-end autonomous driving. In *CVPR*, 2023c.

[^28]: Bo Jiang, Shaoyu Chen, Qing Xu, Bencheng Liao, Jiajie Chen, Helong Zhou, Qian Zhang, Wenyu Liu, Chang Huang, and Xinggang Wang. Vad: Vectorized scene representation for efficient autonomous driving. In *ICCV*, 2023.

[^29]: Tarasha Khurana, Peiyun Hu, Achal Dave, Jason Ziglar, David Held, and Deva Ramanan. Differentiable raycasting for self-supervised occupancy forecasting. In *ECCV*, 2022.

[^30]: Yingyan Li, Yuntao Chen, Jiawei He, and Zhaoxiang Zhang. Densely constrained depth estimator for monocular 3d object detection. In *European Conference on Computer Vision*, pp. 718–734. Springer, 2022a.

[^31]: Yingyan Li, Lue Fan, Yang Liu, Zehao Huang, Yuntao Chen, Naiyan Wang, and Zhaoxiang Zhang. Fully sparse fusion for 3d object detection. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2024a.

[^32]: Zhiqi Li, Wenhai Wang, Hongyang Li, Enze Xie, Chonghao Sima, Tong Lu, Qiao Yu, and Jifeng Dai. Bevformer: Learning bird’s-eye-view representation from multi-camera images via spatiotemporal transformers. *arXiv preprint arXiv:2203.17270*, 2022b.

[^33]: Zhiqi Li, Zhiding Yu, Shiyi Lan, Jiahan Li, Jan Kautz, Tong Lu, and Jose M Alvarez. Is ego status all you need for open-loop end-to-end autonomous driving? In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 14864–14873, 2024b.

[^34]: Yingfei Liu, Tiancai Wang, Xiangyu Zhang, and Jian Sun. Petr: Position embedding transformation for multi-view 3d object detection. In *European Conference on Computer Vision*, pp. 531–548. Springer, 2022.

[^35]: Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2021.

[^36]: Ilya Loshchilov and Frank Hutter. Sgdr: Stochastic gradient descent with warm restarts. *arXiv preprint arXiv:1608.03983*, 2016.

[^37]: Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. *arXiv preprint arXiv:1711.05101*, 2017.

[^38]: Chen Min, Dawei Zhao, Liang Xiao, Jian Zhao, Xinli Xu, Zheng Zhu, Lei Jin, Jianshu Li, Yulan Guo, Junliang Xing, et al. Driveworld: 4d pre-trained scene understanding via world models for autonomous driving. *arXiv preprint arXiv:2405.04390*, 2024.

[^39]: Chenbin Pan, Burhaneddin Yaman, Tommaso Nesti, Abhirup Mallik, Alessandro G Allievi, Senem Velipasalar, and Liu Ren. Vlp: Vision language planning for autonomous driving. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 14760–14769, 2024.

[^40]: Jinhyung Park, Chenfeng Xu, Shijia Yang, Kurt Keutzer, Kris Kitani, Masayoshi Tomizuka, and Wei Zhan. Time will tell: New outlooks and a baseline for temporal multi-view 3d object detection. *arXiv preprint arXiv:2210.02443*, 2022.

[^41]: Aditya Prakash, Kashyap Chitta, and Andreas Geiger. Multi-modal fusion transformer for end-to-end autonomous driving. In *CVPR*, 2021.

[^42]: Katrin Renz, Kashyap Chitta, Otniel-Bogdan Mercea, A Koepke, Zeynep Akata, and Andreas Geiger. Plant: Explainable planning transformers via object-level representations. *arXiv preprint arXiv:2210.14222*, 2022.

[^43]: Abbas Sadat, Sergio Casas, Mengye Ren, Xinyu Wu, Pranaab Dhawan, and Raquel Urtasun. Perceive, predict, and plan: Safe motion planning through interpretable semantic representations. In *ECCV*, 2020.

[^44]: Hao Shao, Letian Wang, RuoBing Chen, Hongsheng Li, and Yu Liu. Safety-enhanced autonomous driving using interpretable sensor fusion transformer. *CoRL*, 2022.

[^45]: Hao Shao, Letian Wang, Ruobing Chen, Steven L Waslander, Hongsheng Li, and Yu Liu. ReasonNet: End-to-End Driving with Temporal and Global Reasoning. In *CVPR*, pp. 13723–13733, 2023.

[^46]: Xiaoyu Tian, Junru Gu, Bailin Li, Yicheng Liu, Yang Wang, Zhiyong Zhao, Kun Zhan, Peng Jia, Xianpeng Lang, and Hang Zhao. Drivevlm: The convergence of autonomous driving and large vision-language models. *arXiv preprint arXiv:2402.12289*, 2024.

[^47]: Marin Toromanoff, Emilie Wirbel, and Fabien Moutarde. End-to-end model-free reinforcement learning for urban driving using implicit affordances. In *CVPR*, 2020.

[^48]: Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. *Advances in neural information processing systems*, 30, 2017.

[^49]: Qitai Wang, Yuntao Chen, Ziqi Pang, Naiyan Wang, and Zhaoxiang Zhang. Immortal tracker: Tracklet never dies. *arXiv preprint arXiv:2111.13672*, 2021.

[^50]: Shihao Wang, Zhiding Yu, Xiaohui Jiang, Shiyi Lan, Min Shi, Nadine Chang, Jan Kautz, Ying Li, and Jose M Alvarez. Omnidrive: A holistic llm-agent framework for autonomous driving with 3d perception, reasoning and planning. *arXiv preprint arXiv:2405.01533*, 2024a.

[^51]: Yuqi Wang, Yuntao Chen, Xingyu Liao, Lue Fan, and Zhaoxiang Zhang. Panoocc: Unified occupancy representation for camera-based 3d panoptic segmentation. *arXiv preprint arXiv:2306.10013*, 2023a.

[^52]: Yuqi Wang, Jiawei He, Lue Fan, Hongxin Li, Yuntao Chen, and Zhaoxiang Zhang. Driving into the future: Multiview visual forecasting and planning with world model for autonomous driving. *arXiv preprint arXiv:2311.17918*, 2023b.

[^53]: Yuqi Wang, Ke Cheng, Jiawei He, Qitai Wang, Hengchen Dai, Yuntao Chen, Fei Xia, and Zhaoxiang Zhang. Drivingdojo dataset: Advancing interactive and knowledge-enriched driving world model. *arXiv preprint arXiv:2410.10738*, 2024b.

[^54]: Xinshuo Weng, Boris Ivanovic, Yan Wang, Yue Wang, and Marco Pavone. Para-drive: Parallelized architecture for real-time autonomous driving. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 15449–15458, 2024.

[^55]: Penghao Wu, Xiaosong Jia, Li Chen, Junchi Yan, Hongyang Li, and Yu Qiao. Trajectory-guided control prediction for end-to-end autonomous driving: a simple yet strong baseline. *NeurIPS*, 2022.

[^56]: Penghao Wu, Li Chen, Hongyang Li, Xiaosong Jia, Junchi Yan, and Yu Qiao. Policy pre-training for autonomous driving via self-supervised geometric modeling. In *ICLR*, 2023.

[^57]: Wenyuan Zeng, Wenjie Luo, Simon Suo, Abbas Sadat, Bin Yang, Sergio Casas, and Raquel Urtasun. End-to-end interpretable neural motion planner. In *CVPR*, 2019.

[^58]: Lunjun Zhang, Yuwen Xiong, Ze Yang, Sergio Casas, Rui Hu, and Raquel Urtasun. Learning unsupervised world models for autonomous driving via discrete diffusion. *arXiv preprint arXiv:2311.01017*, 2023.

[^59]: Qihang Zhang, Zhenghao Peng, and Bolei Zhou. Learning to drive by watching youtube videos: Action-conditioned contrastive policy pretraining. In *European Conference on Computer Vision*, pp. 111–128. Springer, 2022.

[^60]: Zhejun Zhang, Alexander Liniger, Dengxin Dai, Fisher Yu, and Luc Van Gool. End-to-end urban driving by imitating a reinforcement learning coach. In *ICCV*, 2021.

[^61]: Wenzhao Zheng, Weiliang Chen, Yuanhui Huang, Borui Zhang, Yueqi Duan, and Jiwen Lu. Occworld: Learning a 3d occupancy world model for autonomous driving. *arXiv preprint arXiv:2311.16038*, 2023.

[^62]: Xingyi Zhou, Vladlen Koltun, and Philipp Krähenbühl. Tracking objects as points. In *European conference on computer vision*, pp. 474–490. Springer, 2020.
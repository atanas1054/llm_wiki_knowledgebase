---
title: "WorldRFT: Latent World Model Planning with Reinforcement Fine-Tuning for Autonomous Driving"
source: "https://arxiv.org/html/2512.19133v1"
author:
published:
created: 2026-10-15
description:
tags:
  - "clippings"
---
Pengxuan Yang    Ben Lu    Zhongpu Xia    Chao Han    Yinfeng Gao    Teng Zhang    Kun Zhan    XianPeng Lang    Yupeng Zheng    Qichao Zhang

###### Abstract

Latent World Models enhance scene representation through temporal self-supervised learning, presenting a perception annotation-free paradigm for end-to-end autonomous driving. However, the reconstruction-oriented representation learning tangles perception with planning tasks, leading to suboptimal optimization for planning. To address this challenge, we propose WorldRFT, a planning-oriented latent world model framework that aligns scene representation learning with planning via a hierarchical planning decomposition and local-aware interactive refinement mechanism, augmented by reinforcement learning fine-tuning (RFT) to enhance safety-critical policy performance. Specifically, WorldRFT integrates a vision-geometry foundation model to improve 3D spatial awareness, employs hierarchical planning task decomposition to guide representation optimization, and utilizes local-aware iterative refinement to derive a planning-oriented driving policy. Furthermore, we introduce Group Relative Policy Optimization (GRPO), which applies trajectory Gaussianization and collision-aware rewards to fine-tune the driving policy, yielding systematic improvements in safety. WorldRFT achieves state-of-the-art (SOTA) performance on both open-loop nuScenes and closed-loop NavSim benchmarks. On nuScenes, it reduces collision rates by 83% (0.30% → 0.05%). On NavSim, using camera-only sensors input, it attains competitive performance with the LiDAR-based SOTA method DiffusionDrive (87.8 vs. 88.1 PDMS).

<sup>1</sup> The State Key Laboratory of Multimodal Artificial Intelligence Systems, Institute of Automation, CAS

<sup>2</sup> School of Advanced Interdisciplinary Sciences, UCAS

<sup>3</sup> School of Artificial Intelligence, UCAS

<sup>4</sup> Li Auto

Code — https://github.com/pengxuanyang/WorldRFT

![[fig1 1.png|Refer to caption]]

Figure 1: Performance comparison of SOTA methods on open-loop nuScenes and closed-loop NavSim benchmarks.

## Introduction

Traditional end-to-end autonomous driving methods [^2] [^12] [^13] [^26] often rely on multi-task architectures incorporating auxiliary perception modules such as object detection, online mapping, and occupancy prediction to improve scene understanding. Recently, a new paradigm of constructing unified latent world model (WM) representations from raw surround-view images [^36] [^18] has enabled slight end-to-end autonomous driving through temporal self-supervised learning.

Despite this progress, existing latent WM suffers from a critical misalignment between reconstruction-oriented representations and planning requirements, manifesting in three key challenges: (1) Lacking spatial awareness: current reconstruction objectives produce generic representations with limited 3D spatial awareness, which is crucial for trajectory planning. Although World4Drive [^36] attempts to incorporate explicit 3D information (depth map) through monocular depth estimation models [^9], this approach suffers from cross-view inconsistencies, limiting the model’s global comprehension of complex driving scenarios. (2) Inefficient planning interaction mechanisms: existing models employ a single global planning query to generate trajectories from entire feature maps, inadequately capturing local structures essential for fine-grained planning. This results in dispersed attention and inability to effectively comprehend and extract planning-relevant scene representations, as shown in Figure 5, leading to suboptimal feature utilization for planning decisions. (3) Limited safety awareness: perception-based imitation learning approaches, such as PARA-Drive [^29], DiffusionDrive [^20], focus on minimizing trajectory deviations from expert demonstrations without incorporating explicit safety objectives. This paradigm lacks active collision avoidance, merely treating all deviations from expert trajectories equally without understanding underlying safety principles or proactively identifying potential risks in novel scenarios.

To address these challenges, we propose WorldRFT, a planning-oriented latent world model framework that aligns representation learning with planning requirements. First, addressing the weak 3D spatial understanding, we design a Spatial-aware World Encoder (SWE) that constructs 3D spatially-aware representations by incorporating the visual-geometric foundation model VGGT [^28], establishing a robust spatial understanding for planning tasks. Second, resolving the inefficiency of planning interaction mechanisms, we propose a Hierarchical Planning Refinement (HPR) module that decomposes complex end-to-end planning into three parallel subtasks which describe planning in hierarchical dimension: target region localization, spatial path planning, and temporal trajectory prediction. Building upon this decomposition, we develop a local-aware iterative refinement module that dynamically samples and fuses task-relevant local features from the latent space, ensuring both global consistency and local precision. Third, to enhance safety-critical planning capabilities, we design a safety-aware Reinforcement learning Fine-Tuning (RFT) phase based on Group Relative Policy Optimization (GRPO) [^24] to optimize explicit safety objectives. By gaussianizing predicted trajectories and employing collision-aware rewards, our approach transitions from passive behavior cloning to active collision avoidance.

Experiments demonstrate that WorldRFT achieves state-of-the-art (SOTA) performance on both open-loop nuScenes and closed-loop NavSim benchmarks, as shown in Figure 1. On the nuScenes dataset, compared to the baseline LAW [^18], average displacement error decreases by 21% ($0.61\text{m}\rightarrow 0.48\text{m}$) and collision rate drops by 83% ($0.30\%\rightarrow 0.05\%$). On the NavSim dataset, the PDMS metric improves by 3.2 points ($84.6\rightarrow\textbf{87.8}$), achieving the best performance among vision-only solutions and approaching the LiDAR-based SOTA method DiffusionDrive (88.1).

Our main contributions are summarized as follows:

- We propose the first planning-oriented latent world modeling paradigm that deeply aligns representation learning with planning tasks through the spatial geometric prior fusion and hierarchical planning interaction and local-aware refinement.
- We introduce a reinforcement fine-tuning phase that enhances safety planning via explicit safety rewards and GRPO method, transitioning from behavioral imitation to proactive collision avoidance.
- Our method achieves SOTA performance on both nuScenes and NavSim benchmarks, with substantial improvements in critical safety metrics.

## Related Works

### End-to-End Autonomous Driving

End-to-end autonomous driving models utilize the imitation learning framework with perception annotations to directly map sensor inputs to trajectory outputs [^6] [^31]. For instance, UniAD and VAD [^11] [^13] fuse multiple perception and planning tasks within a cascaded BEV-based architecture, while PARA-Drive and ONE-Drive [^29] [^35] leverage parallel architecture to enhance efficiency. Methods like SparseDrive [^26] integrate detection, tracking, and mapping through symmetric sparse perception modules. Despite their advantages, these approaches require high-precision 3D perception annotations, leading to higher development complexity and costs.

### World Models for Autonomous Driving

World models predict future scene states to improve dynamic environment understanding and support safer path planning. For instance, GAIA-1 [^8] creates realistic driving scenarios from multi-modal inputs and allows detailed control over vehicle behavior, while DriveDreamer4D [^33] generates novel trajectory videos from real driving data to enhance 4D reconstruction. Methods like DriveWorld and PreWorld [^21] [^17] [^7] [^15] generate future occupancy and flow fields from BEV embeddings. In contrast, self-supervised approaches like LAW [^18] and SSR [^16] model future scene dynamics without labeled data, but their relatively coarse planning frameworks can limit planning effectiveness, indicating a need for more sophisticated architectural designs.

### Reinforcement Learning for Autonomous Driving

Reinforcement learning (RL) is increasingly important in autonomous driving for complex decision-making. RAD [^5] builds virtual environments with 3D Gaussian Splatting to facilitate exploration and policy learning in diverse scenarios. AlphaDrive [^14] introduces RL to enhance planning and training of vision-language models. Given RL’s advantage over imitation learning in handling causal confounding, we adopt RL for trajectory planning to improve safety and reduce collision risk in autonomous driving.

## Methodology

### Overview

![[main_new.png|Refer to caption]]

Figure 2: WorldRFT consists of three key modules: 1) Spatial-aware World Encoder (yellow) that extracts geometry-rich latent world representations from RGB images; 2) Hierarchical Planning Refinement Module (green) that efficiently captures critical information highly correlated with driving decisions through a refined planning task design; 3) Safety-aware Reinforcement Learning Fine-tuning phase (blue) that generates safer planning outcomes via reinforcement learning optimization. These modules work synergistically to deliver high-quality end-to-end autonomous driving planning.

As illustrated in Figure 2, WorldRFT comprises three key modules that form a complete pipeline of “scene understanding $\rightarrow$ planning decisions $\rightarrow$ safety optimization”: (1) SWE constructs spatial-rich latent world representations from RGB images; (2) HPR efficiently extracts critical information strongly correlated with driving decisions; (3) RFT generates safer planning results through reinforcement learning optimization. These modules work synergistically to achieve safer end-to-end autonomous driving planning.

### Spatial-aware World Encoder

The spatial-aware world encoder constructs 3D spatially-aware latent world representations serving planning tasks from surround-view inputs through the geometric foundation model VGGT [^28].

Basic Visual Feature Encoding. Given multi-view images $I_{t}\in\mathbb{R}^{M\times H\times W\times 3}$ at time $t$, we first extract features $F_{t}\in\mathbb{R}^{M\times h\times w\times D}$ through the image backbone. Following World4Drive [^36], we then utilize the vision-language foundation model Grounded-SAM for semantic supervised learning through the loss function $\mathcal{L}_{\text{sem}}$.

3D Spatial Encoding. Spatial understanding capability is crucial for planning tasks. VGGT, benefiting from its diverse training data and elegant feed-forward architecture, provides multi-view consistent geometric-rich prior from 2D surround images $I_{t}$. We leveraging VGGT’s structural priors to enhance spatial awareness. Specifically, to maintain generic 3D aware capacity, we employ a frozen VGGT as a 3D spatial encoder and extract 3D tokens from its final layer:

$$
\varepsilon_{l}(I_{t})=\{t_{c},t_{r},t_{3D}\}
$$

where $l$ denotes the layer index, and $t_{c}$, $t_{r}$, and $t_{3D}$ represent camera, register, and 3D tokens, respectively. We subsequently incorporate the 3D tokens $t_{3D}$ into the image features $F_{t}$ to construct spatial-aware latent world representations. Specifically, we propose a lightweight fusion module that integrates ResNet and VGGT features via a single cross-attention layer, where the 2D visual features $F_{t}$ act as queries and the VGGT-derived 3D features $t_{3D}$ are used as keys and values. This process ultimately yields the unified latent space visual representation $W^{t}_{\text{latent}}$, enabling effective spatial understanding without requiring explicit 3D inputs:

$$
W^{t}_{\text{latent}}=\text{C-A}(F_{t},t_{3D})
$$

where C-A denotes Cross-Attention layers.

### Hierarchical Planning Refinement

In this section, we decompose end-to-end planning into three parallel subtasks that extract hierarchical features with distinct supervision: target region localization, spatial path planning, and temporal trajectory prediction. These tasks capture complementary aspects of planning and are unified through a query interaction framework and refined via local-aware iterative refinement.

Unified Query Interaction Framework. To facilitate coordination among parallel subtasks while maintaining their unique functionalities, we design a unified query interaction framework. Specifically, each subtask is initialized with dedicated queries: $Q_{\text{target}}$, $Q_{\text{path}}$, and $Q_{\text{traj}}$, which then independently aggregate hierarchical planning-relevant features from the latent world representation:

$$
[Q^{\prime}_{\text{target}},Q^{\prime}_{\text{path}},Q^{\prime}_{\text{traj}}]=\text{C-A}([Q_{\text{target}},Q_{\text{path}},Q_{\text{traj}}],W^{t}_{\text{latent}})
$$

To facilitate inter-task communication, these queries undergo concatenation and self-attention, enabling mutual awareness of planning intentions:

$$
[Q^{\prime\prime}_{\text{target}},Q^{\prime\prime}_{\text{path}},Q^{\prime\prime}_{\text{traj}}]=\text{S-A}(\text{Concat}[Q^{\prime}_{\text{target}},Q^{\prime}_{\text{path}},Q^{\prime}_{\text{traj}}])
$$

Target Region Localization. Predicting a deterministic target point for navigation [^30] is an ill-posed problem due to the inherent uncertainty of valid target points in planning scenarios. Therefore, rather than predicting deterministic target points, we model targets as probabilistic regions using Laplace distributions. In detail, we parameterize the target region as:

$$
(\mu,b)=\text{MLP}(Q^{\prime\prime}_{\text{target}})
$$

where $\mu\in\mathbb{R}^{2}$ denotes the region center and $b\in\mathbb{R}^{2}$ represents the scale parameters, quantifying the spatial extent of the target region. This formulation enables the model to learn target region distributions rather than fitting to a single deterministic point. Moreover, the probabilistic representation naturally captures scene complexity, thereby providing adaptive information that can be leveraged by the iterative refinement module to guide feature fusion.

The probabilistic representation is trained using the negative log-likelihood loss:

$$
\mathcal{L}_{\text{Laplace-NLL}}=\log(2b)+\frac{\|\mathbf{y}-\boldsymbol{\mu}\|_{1}}{b}
$$

Here, the scale parameter $b$ naturally reflects scene complexity, with larger values corresponding to more challenging scenarios that demand cautious planning. As a conditioning signal, $b$ modulates feature fusion within the uncertainty-aware local adaptation module, thereby enabling adaptive planning in accordance with scene uncertainty.

Spatial Path Planning. This module generates a spatial path connecting the current position to the target region. To ensure spatial consistency, we construct the spatial path ground truth by uniformly sampling future trajectory points at fixed spatial intervals, rather than temporal intervals [^18] [^16] [^36] [^13]. Furthermore, we decode $N$ spatial path points through an MLP network:

$$
T_{\text{path}}=\text{MLP}(Q^{\prime\prime}_{\text{path}})
$$

where $T_{\text{path}}\in\mathbb{R}^{N\times 2}$ denotes path points sampled at 2-meter intervals, with each point denoting coordinates $(x,y)$.

Temporal Trajectory Prediction. In this module, we decode future temporal trajectories for $T$ timesteps through a trajectory decoder:

$$
T_{\text{traj}}=\text{MLP}(Q^{\prime\prime}_{\text{traj}})
$$

where $T_{\text{traj}}\in\mathbb{R}^{T\times 2}$ represents trajectory points at fixed temporal intervals (0.5 seconds), with each point containing coordinates $(x,y)$.

Local-aware Iterative Refinement. This module iteratively refines the initial planning outputs by integrating local scene information. It takes the preliminary results $(\mu,b)$, $T_{\text{path}}$, and $T_{\text{traj}}$ as inputs, and refines the outputs of all three subtasks over $K$ iterations. Using temporal trajectory refinement as an example, each iteration involves the following key steps as shown in Figure 3:

First, we encode current planning information $(\mu,b)$, $T_{\text{path}}$, and $T_{\text{traj}}$ into a unified state representation $F_{s}$:

$$
F_{s}=\text{MLP}(\text{Concat}[(\mu^{(k)},b^{(k)}),T^{(k)}_{\text{path}},T^{(k)}_{\text{traj}}])
$$

Subsequently, the trajectory points are projected onto the latent space feature map using the camera parameters, yielding corresponding positions $P_{\text{proj}}$. Deformable convolution is then applied to adaptively sample the local features $F_{\text{local}}$ at these positions:

$$
P_{\text{proj}}=\text{CameraProject}(T^{(k)}_{\text{traj}})
$$
 
$$
F_{\text{local}}=\text{DeformConv}(W^{t}_{\text{latent}},P_{\text{proj}})
$$

Next, local and global features are fused, guided by the scale parameter $b$ obtained from target region modeling, which acts as a conditioning signal:

$$
F_{b}=\text{MLP}(b^{(k)})
$$
 
$$
F_{\text{fusion}}=\text{MLP}(\text{Concat}[F_{\text{local}},Q^{\prime\prime}_{\text{traj}},F_{s},F_{b}])
$$

where $Q^{\prime\prime}_{\text{traj}}$ represents the global planning intention.

Finally, independent prediction heads generate the corresponding trajectory offsets $\Delta T_{\text{traj}}^{(k)}$, which are then used to incrementally update the trajectory via residual connections:

$$
\Delta T_{\text{traj}}^{(k)}=\text{MLP}(F_{\text{fusion}})
$$
 
$$
T^{(k+1)}_{\text{traj}}=T^{(k)}_{\text{traj}}+\alpha\Delta T^{(k)}_{\text{traj}}
$$

where $\alpha=0.1$ is the update step size.

![[subfig2.png|Refer to caption]]

Figure 3: Architecture of the Local-aware Iterative Refinement module. The module refines preliminary planning results ( μ, b ) T path traj (\\mu,b),T\_{\\text{path}},T\_{\\text{traj}} through K iterations. Using T\_{\\text{traj}} as an example, each iteration: (1) encodes global planning states, (2) projects trajectory points via camera parameters, (3) samples local features using deformable convolution, and (4) fuses them with global information and uncertainty representation. Residual connections enable incremental updates for adaptive local adjustment.

Training Loss. We pretrain our model guided by the latent world modeling, which enables self-supervised learning without requiring perception annotations by leveraging temporal consistency in future world prediction. Following [^18], we use a world decoder to predict future latent world representations $\widehat{W}^{t+1}_{\text{latent}}$ from the current latent representation $W^{t}_{\text{latent}}$ and the temporal trajectory $T_{\text{traj}}$. We self-supervise world prediction through the MSE loss:

$$
\mathcal{L}_{\text{rec}}=\text{MSE}(\widehat{W}^{t+1}_{\text{latent}},W^{t+1}_{\text{latent}})
$$

Additionally, we employ L1 loss $\mathcal{L}_{\text{traj}}$ to align both the spatial path $T_{\text{path}}$ and temporal trajectory $T_{\text{traj}}$ with expert demonstrations. The final training objective combines multiple loss terms:

$$
\mathcal{L}_{\text{Total}}=\alpha\mathcal{L}_{\text{sem}}+\beta\mathcal{L}_{\text{rec}}+\gamma\mathcal{L}_{\text{target}}+\eta\mathcal{L}_{\text{traj}}
$$

where $\alpha,\beta,\gamma$ and $\eta$ are hyper-parameters.

### Reinforcement Learning Fine-tuning

In this phase, the module introduces collision-aware rewards, extends trajectories to probability distributions for exploration, and applies GRPO to improve safety while maintaining planning accuracy.

Collision-aware Reward Design. Our RL framework targets collision risks between the ego vehicle and surrounding agents. The collision-aware reward is based on the distance between the ego vehicle’s bounding box and those of nearby agents along the trajectory: negative distances (collisions) incur penalties, while non-collisions yield zero reward:

$$
r=\begin{cases}-1&\text{if collision happens}\\
0&\text{if no collision happens}\end{cases}
$$

Gaussianized-Trajectory Modeling. Since the above trajectory prediction is formulated as a regression task, effective application of RL requires recasting it as a probabilistic, distribution-based problem. To this end, we model trajectories using Gaussian distributions, where the predicted trajectory serve as the means ${\bm{\mu}_{\theta}}$, and an auxiliary variance network adaptively estimates the trajectory variances ${{\bm{\Sigma}}_{\theta}}$.

Policy Optimization. Based on the world model structure, we optimize generated trajectories using GRPO [^24]. Following the GRPO framework, we estimate advantage functions based on relative rewards within sample groups. Specifically, a group of $G$ trajectories is sampled using the previously described latent world model, which serves as the policy ${\pi_{\theta}}$: $\left\{{T_{\text{traj}_{0}},T_{\text{traj}_{1}},...,T_{\text{traj}_{G-1}}}\right\}$. For each point along these trajectories, the relative reward is defined as the normalized per-point reward value:

$$
\tilde{r}_{j}^{T_{\text{traj}_{i}}}=\frac{{r_{j}^{T_{\text{traj}_{i}}}-{\rm{mean}}\left({{\bm{r}_{j}}}\right)}}{{{\rm{std}}\left({{\bm{r}_{j}}}\right)}}
$$

where $r_{j}^{T_{\text{traj}_{i}}}$ denotes the value of the original reward function, and ${\bm{r}_{j}}=\left\{{r_{j}^{T_{\text{traj}_{0}}},r_{j}^{T_{\text{traj}_{1}}},...,r_{j}^{T_{\text{traj}_{G-1}}}}\right\}$ denotes the values of the group-based reward function.

Given that the predicted trajectories are represented as time-differential increments, each increment at the current time step influences both the current and all subsequent points, leading to potential error accumulation. Accordingly, the relative advantage function for the $j$ -th trajectory point is defined as:

$$
Adv_{j}^{T_{\text{traj}_{i}}}=\sum\limits_{t\geq j}{\tilde{r}_{t}^{T_{\text{traj}_{i}}}}
$$

The policy optimization objective function becomes:

$$
{J}\left(\theta\right)=\frac{1}{G}\sum\limits_{i}{\sum\limits_{j}{\left\{{\min\left[{{f_{1}},{f_{2}}}\right]-\beta{D_{KL}}\left({{\pi_{\theta}}||{\pi_{{\theta_{ref}}}}}\right)}\right\}}}
$$

where:

$$
{f_{1}}=\left({\frac{{{\pi_{\theta}}}}{{{\pi_{\theta old}}}}}\right)Adv_{j}^{T_{\text{traj}_{i}}}
$$
 
$$
{f_{2}}=\text{clip}\left({\frac{{{\pi_{\theta}}}}{{{\pi_{{\theta_{old}}}}}},1-\varepsilon,1+\varepsilon}\right)Adv_{j}^{T_{\text{traj}_{i}}}
$$

where $\varepsilon$ limits the update step size of the policy function to ensure algorithmic stability, and $\beta$ is the coefficient for the Kullback-Leibler (KL) divergence. ${\pi_{{\theta_{old}}}}$ denotes the old policy, and ${\pi_{{\theta_{ref}}}}$ represents the reference policy, which is the pre-trained model. As mentioned above, we model the output of the trajectory as a Gaussian distribution with mean vector ${\bm{\mu}_{\theta}}$ and covariance matrix ${{\bm{\Sigma}}_{\theta}}$. The reference model provides a deterministic result ${\bm{\mu}_{ref}}=T_{\text{traj}_{ref}}$, which is used to compute the negative log-likelihood (NLL) loss under the Gaussian distribution. Thus, the KL divergence is defined as:

$$
{D_{KL}}=\frac{1}{2}\left[\begin{array}[]{l}\log|{{\bf{\Sigma}}_{\theta}}|+{({\bm{\mu}_{ref}}-{\bm{\mu}_{\theta}})^{T}}{\bf{\Sigma}}_{{\rm{\theta}}}^{-1}({\bm{\mu}_{ref}}-{\bm{\mu}_{\theta}})\\
+2\log(2\pi)\end{array}\right]
$$

<table><tbody><tr><td rowspan="2">Method</td><td rowspan="2">Training Approach</td><td colspan="4">L2 (m) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td colspan="4">Collision Rate (%) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td></tr><tr><td>ST-P3 <sup><a href="#fn:10">10</a></sup></td><td>P-IL</td><td>1.33</td><td>2.11</td><td>2.90</td><td>2.11</td><td>0.23</td><td>0.62</td><td>1.27</td><td>0.71</td></tr><tr><td>OccNet <sup><a href="#fn:27">27</a></sup></td><td>P-IL</td><td>1.29</td><td>2.13</td><td>2.99</td><td>2.13</td><td>0.21</td><td>0.59</td><td>1.37</td><td>0.72</td></tr><tr><td>UniAD <sup><a href="#fn:11">11</a></sup></td><td>P-IL</td><td>0.48</td><td>0.96</td><td>1.65</td><td>1.03</td><td>0.05</td><td>0.17</td><td>0.71</td><td>0.31</td></tr><tr><td>VAD <sup><a href="#fn:13">13</a></sup></td><td>P-IL</td><td>0.41</td><td>0.70</td><td>1.05</td><td>0.72</td><td>0.07</td><td>0.18</td><td>0.43</td><td>0.23</td></tr><tr><td>UncAD <sup><a href="#fn:31">31</a></sup></td><td>P-IL</td><td>0.33</td><td>0.59</td><td>0.94</td><td>0.62</td><td>0.10</td><td>0.14</td><td>0.28</td><td>0.17</td></tr><tr><td>PPAD <sup><a href="#fn:3">3</a></sup></td><td>P-IL</td><td>0.31</td><td>0.56</td><td>0.87</td><td>0.58</td><td>0.08</td><td>0.12</td><td>0.38</td><td>0.19</td></tr><tr><td>PARA-Drive <sup><a href="#fn:29">29</a></sup></td><td>P-IL</td><td>0.25</td><td>0.46</td><td>0.74</td><td>0.48</td><td>0.14</td><td>0.23</td><td>0.39</td><td>0.25</td></tr><tr><td>GenAD <sup><a href="#fn:34">34</a></sup></td><td>P-IL</td><td>0.28</td><td>0.49</td><td>0.78</td><td>0.52</td><td>0.08</td><td>0.14</td><td>0.34</td><td>0.19</td></tr><tr><td>LAW <sup><a href="#fn:18">18</a></sup> (Perception-based)</td><td>P-IL</td><td>0.24</td><td>0.46</td><td>0.76</td><td>0.49</td><td>0.08</td><td>0.10</td><td>0.39</td><td>0.19</td></tr><tr><td>DiffusionDrive <sup><a href="#fn:20">20</a></sup></td><td>P-IL</td><td>0.27</td><td>0.54</td><td>0.90</td><td>0.57</td><td>0.03</td><td>0.05</td><td>0.16</td><td>0.08</td></tr><tr><td>Epona <sup><a href="#fn:32">32</a></sup></td><td>SS-L</td><td>0.61</td><td>1.17</td><td>1.98</td><td>1.25</td><td>0.01</td><td>0.22</td><td>0.85</td><td>0.36</td></tr><tr><td>LAW <sup><a href="#fn:18">18</a></sup> (Perception-free)</td><td>SS-L</td><td>0.26</td><td>0.57</td><td>1.01</td><td>0.61</td><td>0.14</td><td>0.21</td><td>0.54</td><td>0.30</td></tr><tr><td>World4Drive <sup><a href="#fn:36">36</a></sup></td><td>SS-L</td><td>0.23</td><td>0.47</td><td>0.81</td><td>0.50</td><td>0.02</td><td>0.12</td><td>0.33</td><td>0.16</td></tr><tr><td>Ours (without RFT)</td><td>SS-L</td><td>0.21</td><td>0.44</td><td>0.76</td><td>0.47</td><td>0.10</td><td>0.11</td><td>0.23</td><td>0.15</td></tr><tr><td>Ours (with RFT)</td><td>SS-L & RL</td><td>0.22</td><td>0.44</td><td>0.77</td><td>0.48</td><td>0.00</td><td>0.00</td><td>0.16</td><td>0.05</td></tr><tr><td>SSR† <sup><a href="#fn:16">16</a></sup></td><td>SS-L</td><td>0.19</td><td>0.36</td><td>0.62</td><td>0.39</td><td>0.00</td><td>0.10</td><td>0.20</td><td>0.13</td></tr><tr><td>Ours (with RFT)†</td><td>SS-L & RL</td><td>0.15</td><td>0.30</td><td>0.56</td><td>0.33</td><td>0.00</td><td>0.00</td><td>0.12</td><td>0.04</td></tr></tbody></table>

- P-IL: Perception-based Imitation Learning; SS-L: Self-Supervised Learning; RL: Reinforcement Learning.
- † Represents methods incorporating ego status.

Table 1: End-to-end planning results on nuScenes benchmark [^1]

| Method | Training Approach | Input | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf. $\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UniAD [^11] | P-IL | C | 97.8 | 91.9 | 92.9 | 100.0 | 78.8 | 83.4 |
| PARA-Drive [^29] | P-IL | C | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 |
| LTF [^22] | P-IL | C | 97.4 | 92.8 | 92.4 | 100.0 | 79.0 | 83.8 |
| Transfuser [^22] | P-IL | C & L | 97.7 | 92.8 | 92.8 | 100.0 | 79.2 | 84.0 |
| VADv2 [^2] | P-IL | C & L | 97.2 | 89.1 | 91.6 | 100.0 | 76.0 | 80.9 |
| Hydra-MDP [^19] | P-IL | C & L | 97.9 | 91.7 | 92.9 | 100.0 | 77.6 | 83.0 |
| DiffusionDrive [^20] | P-IL | C & L | 98.2 | 96.2 | 94.7 | 100.0 | 82.2 | 88.1 |
| Epona [^32] | SS-L | C | 97.9 | 95.1 | 93.8 | 99.9 | 80.4 | 86.2 |
| LAW (Perception-free) [^18] | SS-L | C | 96.4 | 95.4 | 88.7 | 99.9 | 81.7 | 84.6 |
| DriveX [^25] | SS-L | C | 97.5 | 94.0 | 93.0 | 100.0 | 79.7 | 84.5 |
| World4Drive [^36] | SS-L | C | 97.4 | 94.3 | 92.8 | 100.0 | 79.9 | 85.1 |
| Ours (without RFT) | SS-L | C | 97.5 | 96.0 | 94.0 | 100.0 | 80.9 | 87.0 |
| Ours (with RFT) | SS-L & RL | C | 97.8 | 96.8 | 94.0 | 100.0 | 81.7 | 87.8 |

- P-IL: Perception-based Imitation Learning; SS-L: Self-Supervised Learning; RL: Reinforcement Learning.
- C: Camera modality; L: LiDAR modality.

Table 2: End-to-end planning results on NavSim benchmark [^4]

![[fig3 1.png|Refer to caption]]

Figure 4: Visualization of planning trajectories, where the red line is the ground truth, the blue line is the pretrained-only trajectory, and the green line shows the RFT-trajectory. Obviously, the pre-trained trajectory dangerously approaches obstacles, risking collisions. In contrast, the RFT-trajectory proactively adjusts to maintain safe distances, avoiding collisions.

RFT Training Loss. We fine-tune the model using the GRPO objective function and KL loss. Therefore, the loss for reinforcement learning can be written as:

$$
\mathcal{L}_{\text{RL}}={-J}\left(\theta\right)+\lambda{D_{KL}}
$$

## Experiment

### Overview

This section presents the experimental results of the proposed method, including ablation studies for each component. For completeness, detailed descriptions of the dataset, experimental setup, evaluation metrics, hyperparameter settings, scalability analysis, and additional visualizations are provided in the supplementary material.

### Benchmark

We comprehensively evaluate our method on the open-loop nuScenes benchmark [^1] and the closed-loop NavSim benchmark [^4]. On nuScenes, we report displacement error (L2) and collision rate (CR) for trajectory prediction. For NavSim, we adopt the PDM Score (PDMS), which combines performance across five dimensions: no at-fault collision (NC), drivable area compliance (DAC), time-to-collision (TTC), ride comfort (Comf.), and ego progress (EP).

### Main Results

As demonstrated in Table 1, we compare our framework with several SOTA methods. Specifically, WorldRFT achieves SOTA performance among perception annotation-free approaches, demonstrating a 21% reduction in L2 error (0.61 m → 0.48 m) and an impressive 83% reduction in collision rate (0.30% → 0.05%) compared to the strong baseline LAW. Moreover, WorldRFT achieves the lowest collision rate among all evaluated methods, even surpassing perception-based approaches, which underscores the superior safety performance of our planning-oriented design. We also provide results incorporating the ego status, with specific descriptions detailed in the supplementary materials.

In addition, as demonstrated in Table 2, WorldRFT also achieves SOTA performance in closed-loop evaluation with PDMS of 87.8 using only camera inputs. Compared to the baseline LAW (84.6), our approach demonstrates significant improvements across all safety-critical metrics: NC (No At-fault Collision) (96.4 → 97.8), TTC (Time-to-Collision) (88.7 → 94.0), and particularly DAC (Drivable Area Compliance) (95.4 → 96.8). The exceptional DAC score of 96.8, which is the highest among all methods including LiDAR-based approaches, demonstrates that incorporating VGGT geometric priors substantially enhances the model’s 3D spatial understanding without explicit depth supervision.

Notably, our closed-loop performance exceeds that of most methods relying on LiDAR inputs, trailing DiffusionDrive (88.1) by only 0.3 points. This result demonstrates that our planning-oriented latent world modeling, combined with safety-oriented reinforcement fine-tuning, achieves outstanding performance and holds strong potential for real-world deployment.

### Ablation Study

In this section, we conduct comprehensive ablation studies to investigate the effectiveness of each component in our proposed WorldRFT framework. All ablation experiments incorporate our safety-oriented reinforcement fine-tuning by default to ensure consistent safety-aware evaluation. All of the ablation experiments are conducted on the nuScenes and NavSim benchmark. For clarity, we present detailed results on nuScenes here, while NavSim results are provided in the supplementary material.

Effectiveness of VGGT Spatial Encoding. As demonstrated in Table 4, incorporating VGGT geometric priors shows consistent improvements across different configurations. Comparing ID 4 with ID 8, adding VGGT reduces L2 error by 7.7% (0.52 → 0.48) and collision rate by 37.5% (0.08 → 0.05). The performance gains remain substantial even with partial task designs (ID 5-6), demonstrating that VGGT-enhanced spatial understanding benefits planning regardless of the specific task configuration.

Impact of Hierarchical Planning Tasks. Starting from the baseline (ID 1 in Table 4), introducing target region localization and spatial path planning with refinement (ID 4) reduces L2 error by 11.9% (0.59 → 0.52) and collision rate by 50.0% (0.16 → 0.08). This validates our hypothesis that decomposing the complex planning task into hierarchical subtasks enables more effective feature extraction from latent representations, thereby enhancing overall planning performance.

Local-aware Iterative Refinement. As shown in Table 4, comparing ID 7 (without refinement) and ID 8 (with refinement), the refinement module reduces L2 error by 4.0% (0.50 → 0.48) and collision rate by 16.7% (0.06 → 0.05). This demonstrates that incorporating local features based on initial trajectories enables fine-grained trajectory adjustments that improve both trajectory accuracy and safety.

Safety-Oriented Reinforcement Fine-tuning. As shown in Table 1, comparing our method without RFT (Ours without RFT) and with RFT (Ours with RFT), the RL module significantly reduces collision rate by 66.7% (0.15 → 0.05) while L2 error slightly increases from 0.47 to 0.48. This selective improvement in safety metrics validates that our collision-aware reward design guides the model from behavior imitation toward understanding safety principles.

<table><tbody><tr><td rowspan="2">ID</td><td>Latent Encoder</td><td colspan="2">Planning Task</td><td rowspan="2">Refine -ment</td><td rowspan="2">L2</td><td rowspan="2">CR</td></tr><tr><td>VGGT</td><td>Target</td><td>Path</td></tr><tr><td>1</td><td></td><td></td><td></td><td></td><td>0.59</td><td>0.16</td></tr><tr><td>2</td><td></td><td>✓</td><td></td><td>✓</td><td>0.56</td><td>0.10</td></tr><tr><td>3</td><td></td><td></td><td>✓</td><td>✓</td><td>0.54</td><td>0.11</td></tr><tr><td>4</td><td></td><td>✓</td><td>✓</td><td>✓</td><td>0.52</td><td>0.08</td></tr><tr><td>5</td><td>✓</td><td></td><td>✓</td><td>✓</td><td>0.51</td><td>0.08</td></tr><tr><td>6</td><td>✓</td><td>✓</td><td></td><td>✓</td><td>0.49</td><td>0.09</td></tr><tr><td>7</td><td>✓</td><td>✓</td><td>✓</td><td></td><td>0.50</td><td>0.06</td></tr><tr><td>8</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>0.48</td><td>0.05</td></tr></tbody></table>

Table 3: Ablation study of each proposed component

Qualitative Results In this section, we quantitatively evaluate WorldRFT on nuScenes. Figure 8 shows predicted trajectories: ground truth (red), without RFT (blue), and with RFT (green). The without-RFT model approaches surrounding vehicles too closely, while the with-RFT model maintains safer distances and better matches the ground truth. As shown in Figure 5, attention maps (LAW middle, WorldRFT below) demonstrate that WorldRFT focuses more precisely on critical regions like surrounding vehicles, improving perception-planning integration and yielding more robust, human-like behavior.

![[fig4 1.png|Refer to caption]]

Figure 5: Visualization of the original image (top), along with attention maps for the baseline (middle) and WorldRFT (bottom), where blue represents weak attention and red indicates strong attention. The maps reveal that WorldRFT focuses on surrounding planning-critical agents such as vehicles, which are largely overlooked by the baseline method.

## Conclusion

We present WorldRFT, a planning-oriented latent world framework for end-to-end autonomous driving. Our key insight is that existing latent world models suffer from misalignment between reconstruction-oriented learning and planning requirements. To bridge this gap, we introduced three innovations: (1) SWE with VGGT for spatial-aware representations; (2) HPR combining hierarchical task decomposition with iterative refinement for planning-relevant features; and (3) RFT enabling active collision avoidance beyond passive imitation. Experiments on nuScenes and NavSim demonstrate that our planning-oriented design with geometric priors and safety optimization effectively enables slight end-to-end autonomous driving.

## References

WorldRFT: Latent World Model Planning with Reinforcement Fine-Tuning for Autonomous Driving

Supplementary Material

## Implementation Details

In this section, we provide comprehensive implementation details for WorldRFT, including the hierarchical planning strategy and training configurations for both nuScenes and NavSim benchmarks. These details complement the main paper and enable reproducibility of our results.

### Hierarchical Planning

Our hierarchical planning framework decomposes the complex trajectory generation task into three parallel subtasks that capture distinct aspects of planning:

- Target region localization models future destinations as probabilistic Laplace distributions rather than deterministic points, capturing the inherent uncertainty in driving intentions. Thus, the ground-truth for the target region center is provided by the coordinates of the final position of the ground-truth trajectory.
- Spatial path planning produces a geometric path represented by a set of time-independent, equally spaced points that capture only the spatial coordinates of the trajectory. Ground-truth supervision is generated through uniform interpolation of the ground-truth trajectory. This approach streamlines learning by allowing the model to focus exclusively on spatial direction, thereby enhancing its understanding of spatial coordinates.
- Temporal trajectory prediction further refines the geometric path by incorporating temporal dynamics at fixed time intervals. Here, the trajectory is represented as a sequence of time-stamped points, capturing both spatial and dynamic (velocity and acceleration) information. Ground-truth supervision is obtained by uniformly sampling the ground-truth trajectory over time, allowing the model to learn spatial and temporal aspects of motion.

These subtasks are unified through a query interaction framework where dedicated queries ($Q_{\text{target}}$, $Q_{\text{path}}$, $Q_{\text{traj}}$) first extract task-specific features via cross-attention with the latent world representation, then exchange information through self-attention to achieve mutual awareness of planning intentions. This hierarchical decomposition not only simplifies the learning process but also enables more interpretable and controllable trajectory generation.

### NuScenes Training

Following the VAD-Tiny [^13] configuration, we set the default feature dimension to 256 and employ ResNet-50 as the image backbone to process 6 surround-view images with a resolution of $360\times 640$. When utilizing VGGT, which requires a maximum image input width of 518, we first rescale the original images and then crop them to $294\times 518$ before feeding them into the VGGT module.

Following World4Drive [^36], we incorporate Grounded-SAM [^23] for semantic learning. Specifically, we utilize this vision-language model to produce pseudo semantic labels. Given the prompt for the object of interest, we obtain 2D bounding boxes and the corresponding semantic mask $S_{t}\in\mathbb{R}^{H\times W\times C}$ via the Grounded-SAM model, where we only keep labels with high confidence to reduce incorrect labeling. Finally, we leverage cross-entropy loss $\mathcal{L}_{sem}$ to enhance the semantic understanding of the latent representations.

The driving commands are consistent with previous work [^11], comprising three categories: left, right, and straight. For the spatial path, we set a fixed distance of 2 m between consecutive trajectory points with a total of 30 points. For the temporal trajectory, following previous work [^26], we predict future trajectories for 3 seconds with a time interval of 0.5 seconds, yielding 6 trajectory points in total. As for the target region center, we use the final position of the ground-truth trajectory as the ground-truth label for supervision.

For the trajectory refinement module, we set the number of iterations $K$ to 3. The model is trained for 12 epochs on 8 NVIDIA RTX 3090 GPUs with a total batch size of 8 and an initial learning rate of $5\times 10^{-5}$. The loss weights are set as 0.2, 0.2, 0.001, and 1.0 for semantic, reconstruction, target, and trajectory losses, respectively.

For reinforcement fine-tuning, we utilize the pre-trained model output as the mean and apply the GRPO strategy for optimization with a variance network. To better align the model’s behavior with that of the reference model, we introduce an additional reference loss into the reinforcement learning framework, which can be formulated as:

$$
\mathcal{L}_{\text{ref}}=\frac{1}{BT}\sum^{B}\sum^{T}\|\boldsymbol{\mu}_{\theta}-\boldsymbol{\mu}_{ref}\|_{2}
$$

where $B$ and $T$ denote the batch size and the time step, respectively. $\boldsymbol{\mu}_{\theta}$ and $\boldsymbol{\mu}_{ref}$ represent the mean trajectories output by the current model and the reference model, respectively. Last but not least, we incorporate a maximum entropy loss for the normal distribution to prevent premature convergence during reinforcement learning, thereby ensuring the effectiveness of sampling. The initial learning rate is set to $3\times 10^{-6}$, with a group number of 10. The coefficients for the KL loss, reference loss, and maximum entropy loss are set to 0.1, 0.12, and 0.1, respectively.

![[sup_fig1.png|Refer to caption]]

Figure 6: Visualization of WorldRFT on nuScenes in a straight driving scenario. Left: Input images; Right: Predicted target region (ellipse), spatial path, and trajectory. The narrow longitudinal ellipse indicates high uncertainty in speed but confident directional prediction.

![[sup_fig2.png|Refer to caption]]

Figure 7: Visualization of WorldRFT on nuScenes in a scenario with an inaccurate target region center. Left: Input images; Right: Predicted target region (ellipse), spatial path, and trajectory. The large ellipse indicates high uncertainty in both speed and direction. Despite center inaccuracy, the probabilistic target region allows for cautious, safety-aware decisions.

### NavSim Training

In alignment with the NavSim benchmark, our closed-loop model takes as input a concatenated image formed by stitching the front, front-left, and front-right camera views, which is subsequently resized to $256\times 1024$. We employ ResNet-34 for image feature extraction to balance computational efficiency with model performance.

For VGGT and Grounded-SAM, we follow the same usage methodology as described in our NuScenes training approach. Specifically, we apply the same preprocessing pipeline for VGGT input and utilize Grounded-SAM for semantic pseudo-label generation with identical confidence thresholds.

Regarding trajectory configuration, for the spatial path, we set 50 points with a fixed interval of 2 m between consecutive points. For the temporal trajectory, consistent with other methods [^30] in NavSim, we predict future trajectories for 4 seconds with a time interval of 0.5 seconds, resulting in 8 trajectory points in total.

The model is trained for 100 epochs on 8 NVIDIA RTX 3090 GPUs with a batch size of 64. For reinforcement learning, we adopt the same reward design and experimental setup as in our NuScenes training. The reinforcement learning phase is conducted for 10 epochs on 8 NVIDIA RTX 3090 GPUs, maintaining the same batch size of 64.

## Dataset Introduction

### nuScenes Benchmark

#### Overview.

The nuScenes dataset includes 1,000 selected 20-second driving sequences—approximately 15 hours of real-world data—collected in Boston and Singapore [^1]. It captures diverse driving scenarios, complex traffic conditions, and rare events, offering high realism. The dataset provides rich multimodal sensor data: about 1.4 million camera images, 390,000 LiDAR sweeps, 1.4 million radar sweeps, and 1.4 million annotated 3D bounding boxes across 40,000 keyframes, making it a comprehensive resource for training and evaluating autonomous driving systems.

#### Detailed Metrics.

The nuScenes dataset employs L2 Error and Collision Rate as key metrics for trajectory evaluation. The L2 Error quantifies the average Euclidean distance between the predicted (planned) trajectory and the actual trajectory driven by a human. The Collision Rate reflects the frequency with which the planned trajectory results in collisions with other road agents, serving as a critical indicator of safety and planning reliability.

#### Data Processing Pipeline.

For the nuScenes experiments, the processing of the six surrounding camera images begins with resizing the original 900 × 1600 frames to 360 × 640. These are then uniformly padded to achieve a standardized input dimension of 384 × 640, forming the batched input tensor ${I_{t}}\in{\mathbb{R}^{6\times 3\times 384\times 640}}$. The image features are extracted using a backbone network, which applies spatial downsampling to reduce the resolution to 12 × 20. The resulting high-level feature map, represented as ${F_{t}}\in{\mathbb{R}^{6\times 240\times 256}}$, captures rich contextual information and is used as input for downstream network components.

### NavSim Benchmark

#### Overview.

The NavSim benchmark leverages the OpenScenedataset, comprising 1,192 training sequences and 136 test sequences, collectively containing more than 100,000 keyframes [^4]. Following the official baseline protocol, the model outputs trajectories at a frequency of 2 Hz over a 4-second planning horizon, which are then refined and upsampled using a Linear Quadratic Regulator (LQR) controller to generate smooth, dynamically feasible motion plans.

#### Detailed Metrics.

NavSim employs the PDM Score (PDMS) as the primary metric for trajectory evaluation, with the specific scoring formula defined as:

$$
\begin{aligned}
\text{PDMS}=\text{NC}\times\text{DAC}\\
\times\frac{(5\times\text{EP}+5\times\text{TTC}+2\times\text{Comf.})}{12}
\end{aligned}
$$

where NC denotes no at-fault collision, DAC represents drivable area compliance, TTC is time-to-collision, Comf. indicates ride comfort, and EP stands for ego progress. NC and DAC act as multiplicative factors in the evaluation, penalizing unsafe or infeasible trajectories. NC measures the likelihood of collisions between the ego vehicle and other agents, while DAC assesses whether the predicted path remains within drivable areas. Other metrics are combined via weighted summation. EP quantifies the expected progress along the route over the next 4 seconds. TTC evaluates immediate safety by estimating the time to potential collision under constant motion. Comf. assesses driving smoothness by analyzing acceleration, heading changes, and other dynamic behaviors.

#### Data Processing Pipeline.

In the NavSim experiments, the input comprises ego state $e_{t}=[c_{t},v_{t},a_{t}]$ and image data $I_{t}\in\mathbb{R}^{3\times 256\times 1024}$, where $c_{t}$ is a one-hot vector encoding the navigation command (left, straight, right), and $v_{t}$ and $a_{t}$ represent the ego vehicle’s velocity and acceleration at time $t$, respectively. The raw image input $I^{\text{raw}}_{t}\in\mathbb{R}^{3\times 1080\times 1920}$ is preprocessed as follows: the front view $I^{\text{front}}_{t}$ is cropped vertically to $\mathbb{R}^{3\times 1024\times 1920}$, while left $I^{\text{left}}_{t}$ and right views $I^{\text{right}}_{t}$ are cropped in both height and width to $\mathbb{R}^{3\times 1024\times 1088}$. These are then concatenated along the width axis to form:

$$
I^{\text{concat}}_{t}=\text{Concat}(I^{\text{left}}_{t},I^{\text{front}}_{t},I^{\text{right}}_{t})
$$

Finally, the concatenated image is resized to the final input resolution: $\mathbb{R}^{3\times 256\times 1024}$.

## Additional Experimental Results

This section presents comprehensive ablation studies that systematically evaluate each component’s contribution and analyze hyperparameters’ impact on model performance.

### Ablation study on NavSim

Table 4 presents the ablation study of each proposed component on the NavSim benchmark. We systematically evaluate the contribution of VGGT latent encoder, hierarchical planning tasks (target prediction and path planning), and the refinement module. All ablation experiments incorporate our safety-oriented reinforcement fine-tuning by default to ensure consistent safety-aware evaluation. The results demonstrate that each component contributes to the overall performance improvement, with the complete model achieving the highest PDMS of 87.8. The most significant improvement comes from incorporating VGGT (ID 4 → ID 5: +0.7 PDMS), validating the importance of spatial-aware representations. The hierarchical planning tasks (target and path) provide complementary benefits, with their combination yielding better results than individual tasks. The refinement module adds the final performance boost (ID 7 → ID 8: +0.2 PDMS), demonstrating the value of local-aware iterative refinement.

Table 4: Ablation study of each proposed component on NavSim benchmark. Components are systematically removed to evaluate individual contributions.

<table><tbody><tr><td rowspan="2">ID</td><td>Latent Encoder</td><td colspan="2">Planning Task</td><td rowspan="2">Refine -ment</td><td rowspan="2">PDMS</td></tr><tr><td>VGGT</td><td>Target</td><td>Path</td></tr><tr><td>1</td><td></td><td></td><td></td><td></td><td>85.0</td></tr><tr><td>2</td><td></td><td>✓</td><td></td><td>✓</td><td>86.0</td></tr><tr><td>3</td><td></td><td></td><td>✓</td><td>✓</td><td>86.1</td></tr><tr><td>4</td><td></td><td>✓</td><td>✓</td><td>✓</td><td>86.5</td></tr><tr><td>5</td><td>✓</td><td></td><td>✓</td><td>✓</td><td>87.2</td></tr><tr><td>6</td><td>✓</td><td>✓</td><td></td><td>✓</td><td>87.1</td></tr><tr><td>7</td><td>✓</td><td>✓</td><td>✓</td><td></td><td>87.6</td></tr><tr><td>8</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>87.8</td></tr></tbody></table>

### Ablation study of scalability

We investigate the scalability of our model by varying the backbone architecture and feature dimensions. As shown in Table 5, increasing model capacity consistently improves performance. ResNet-101 achieves the best results (L2: 0.46 m, Collision: 0.04%), representing a 4.2% improvement in L2 error and 20% reduction in collision rate compared to ResNet-50 with the same feature dimension. The results also show that feature dimension has a significant impact, with dimension 256 providing the optimal balance between performance and computational efficiency.

Table 5: Ablation study of model scalability on nuScenes. Different backbone architectures and feature dimensions are evaluated.

| ID | Backbone | Dimension | L2 (m) $\downarrow$ | Collision (%) $\downarrow$ |
| --- | --- | --- | --- | --- |
| 1 | ResNet-34 | 256 | 0.50 | 0.09 |
| 2 | ResNet-50 | 128 | 0.51 | 0.08 |
| 3 | ResNet-50 | 256 | 0.48 | 0.05 |
| 4 | ResNet-50 | 384 | 0.46 | 0.06 |
| 5 | ResNet-101 | 256 | 0.46 | 0.04 |

### Ablation Study on Refinement Iterations

Table 6 analyzes the impact of refinement iterations $K$ on model performance. The results show that $K=3$ achieves the optimal balance between performance and computational efficiency. While a single iteration provides some improvement, increasing to 3 iterations reduces collision rate by 37.5% (0.08 → 0.05). Further iterations ($K=6$) provide diminishing returns, with collision rate slightly increasing to 0.07, suggesting potential overfitting.

Table 6: Ablation study of refinement iterations $K$ on nuScenes. Different numbers of iterations in the trajectory refinement module are evaluated.

| Iterations | L2 (m) $\downarrow$ | Collision (%) $\downarrow$ |
| --- | --- | --- |
| 1 | 0.49 | 0.08 |
| 3 | 0.48 | 0.05 |
| 6 | 0.48 | 0.07 |

### Ablation Study on Latent Encoder Module

We evaluate the individual contributions of VGGT and Grounded-SAM in our latent encoder module. As shown in Table 7, VGGT alone (ID 1) provides stronger benefits than Grounded-SAM alone (ID 2), reducing L2 error by 5.8% compared to using only Grounded-SAM. However, combining both components (ID 3) yields the best performance, demonstrating their complementary nature: VGGT provides geometric understanding while Grounded-SAM enhances semantic awareness.

Table 7: Ablation study of latent encoder module components on nuScenes.

| ID | VGGT | Grounded-SAM | L2 (m) $\downarrow$ | Collision (%) $\downarrow$ |
| --- | --- | --- | --- | --- |
| 1 | ✓ |  | 0.49 | 0.06 |
| 2 |  | ✓ | 0.52 | 0.08 |
| 3 | ✓ | ✓ | 0.48 | 0.05 |

Table 8: Ablation study of spatial path configuration on nuScenes. “Num” denotes the number of trajectory points, and “Inter” denotes the interval distance in meters.

<table><tbody><tr><td colspan="2">Spatial Path Configuration</td><td rowspan="2">L2 (m) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td rowspan="2">Collision (%) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><td>Num</td><td>Inter (m)</td></tr><tr><td>30</td><td>1</td><td>0.50</td><td>0.12</td></tr><tr><td>30</td><td>2</td><td>0.48</td><td>0.10</td></tr><tr><td>50</td><td>1</td><td>0.49</td><td>0.09</td></tr><tr><td>50</td><td>2</td><td>0.48</td><td>0.05</td></tr><tr><td>80</td><td>1</td><td>0.52</td><td>0.11</td></tr><tr><td>80</td><td>2</td><td>0.51</td><td>0.12</td></tr></tbody></table>

### Ablation Study on Spatial Path Configuration

Table 8 investigates different spatial path configurations by varying the number of points and intervals. The results demonstrate that 50 points with 2 m intervals provide the optimal configuration. Configurations with too few points (30) or too many points (80) lead to suboptimal performance, suggesting that appropriate spatial resolution is crucial for effective path representation. The 2 m interval consistently outperforms 1 m interval across different point numbers, indicating that overly dense sampling may introduce noise.

### Ablation Study on Target Prediction

We compare deterministic target point prediction with our probabilistic target region approach. As shown in Table 9, probabilistic modeling significantly reduces both the L2 error (by 9.4%) and the collision rate (by 50%). This demonstrates the effectiveness of our design in handling prediction uncertainty using Laplace distributions, which better capture the complexity of driving decisions. By modeling uncertainty, our approach enables the model to consider a broader range of information and make safer decisions.

Table 9: Comparison between deterministic and probabilistic target modeling approaches on nuScenes.

| Target Modeling Method | L2 (m) $\downarrow$ | Collision (%) $\downarrow$ |
| --- | --- | --- |
| Deterministic Target Point | 0.53 | 0.10 |
| Probabilistic Target Region | 0.48 | 0.05 |

![[sup_fig3.png|Refer to caption]]

Figure 8: Visualization of failure case on nuScenes, with the input images on the left and the model output results on the right, where the red line represents the ground truth trajectory, the blue line indicates the trajectory predicted by WorldRFT, and the pink line shows the trajectory predicted by LAW 18.

Table 10: Ablation study of dataset percentage used for reinforcement learning fine-tuning on nuScenes.

| Percentage of Data | L2 (m) $\downarrow$ | Collision (%) $\downarrow$ |
| --- | --- | --- |
| 10% | 0.47 | 0.12 |
| 20% | 0.47 | 0.10 |
| 100% | 0.48 | 0.05 |

### Ablation Study on Dataset Percentage for RFT

Table 10 investigates the impact of training data size on reinforcement fine-tuning (RFT) performance. While using only 10% or 20% of data achieves competitive L2 error (0.47 m), the collision rate significantly reduces when using the full dataset. This reduction in collision rate with full data demonstrates the importance of diverse scenarios for learning robust safety policies.

Table 11: Comparison between supervised fine-tuning (SFT) and reinforcement fine-tuning (RFT) for safety optimization on nuScenes.

| Optimization Method | L2 (m) $\downarrow$ | Collision (%) $\downarrow$ |
| --- | --- | --- |
| SFT | 0.48 | 0.10 |
| RFT | 0.48 | 0.05 |

Table 12: Ablation study of the group number of GRPO on nuScenes.

| Group Number | L2 (m) $\downarrow$ | Collision (%) $\downarrow$ |
| --- | --- | --- |
| 5 | 0.48 | 0.08 |
| 10 | 0.48 | 0.05 |
| 15 | 0.48 | 0.07 |

### Safety Optimization: SFT vs. RFT

Supervised fine-tuning (SFT) adopts the same safety-aware loss as our reinforcement learning methods but is trained using supervised learning. We compare SFT with RFT in Table 11. While both methods achieve similar L2 errors (0.48 m), RFT reduces the collision rate by 50% (0.10 → 0.05). This significant improvement in safety performance highlights the advantage of the reinforcement learning framework: unlike SFT, which merely imitates expert trajectories, RFT leverages a collision-aware reward and GRPO to enable the model to actively learn collision avoidance.

### Ablation Study on GRPO Group Number

Table 12 investigates the impact of group number $G$ in GRPO. The results show that $G=10$ provides the optimal balance between exploration diversity and computational efficiency. Smaller groups ($G=5$) result in less stable advantage estimation, while larger groups ($G=15$) increase computational cost without significant performance gains.

## Additional Qulitative Results

### More Visualization

We provide additional visualization results of WorldRFT on nuScenes, as shown in Figure 6 and Figure 7. On the left side is the input image, and on the right side is the corresponding schematic of the predicted results. As illustrated, WorldRFT is capable of predicting plausible and feasible trajectories, with reasonable estimates for both the target region and the spatial path. Notably, the learned target region reflects the model’s uncertainty in a physically meaningful way.

In Figure 6, which depicts a straight driving scenario, the predicted target region takes the shape of a narrow, elongated ellipse aligned primarily along the longitudinal axis. This indicates that the uncertainty is primarily attributable to ambiguity in speed prediction; although the direction of motion remains certain, the model accommodates a range of possible travel distances over the prediction horizon.

In contrast, Figure 7 illustrates a scenario where the target region forms a larger, more isotropic ellipse. In this case, the center of the target region shows a degree of inaccuracy in predicting the exact future position. The uncertainty arises not only from speed but also from direction predictions. However, the probabilistic nature of our approach, through the inclusion of a target region and its associated uncertainty, allows the model to consider a broader range of potential outcomes. This encourages more cautious decision-making, preventing the vehicle from directly heading towards the potentially inaccurate center of the target region.

These visualization results demonstrate that our probabilistic target region, modeled via Laplace distributions, adaptively captures multimodal uncertainty in real-world driving scenarios. The shape and orientation of the predicted ellipses provide intuitive insights into the model’s confidence, enabling safer and more robust decision-making by accounting for both longitudinal and lateral prediction uncertainty. This highlights the effectiveness of our probability modeling and uncertainty handling in enhancing driving decisions and safety.

### Failure Case

Figure 8 illustrates a failure case of WorldRFT on the nuScenes, where the model produces suboptimal trajectories in response to erroneous driving directives. An analysis reveals annotation inaccuracies within the dataset—for instance, a maneuver requiring a left turn is incorrectly labeled as “go straight”—which introduces ambiguity and hinders the generation of precise motion predictions.

[^1]: Caesar, H.; Bankiti, V.; Lang, A. H.; Vora, S.; Liong, V. E.; Xu, Q.; Krishnan, A.; Pan, Y.; Baldan, G.; and Beijbom, O. 2020. nuScenes: A Multimodal dataset for autonomous driving. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 11621–11631.

[^2]: Chen, S.; Jiang, B.; Gao, H.; Liao, B.; Xu, Q.; Zhang, Q.; Huang, C.; Liu, W.; and Wang, X. 2024a. Vadv2: End-to-end vectorized autonomous driving via probabilistic planning. *arXiv preprint arXiv:2402.13243*.

[^3]: Chen, Z.; Ye, M.; Xu, S.; Cao, T.; and Chen, Q. 2024b. Ppad: Iterative interactions of prediction and planning for end-to-end autonomous driving. In *Proceedings of the European Conference on Computer Vision*, 239–256. Springer.

[^4]: Dauner, D.; Hallgarten, M.; Li, T.; Weng, X.; Huang, Z.; Yang, Z.; Li, H.; Gilitschenski, I.; Ivanovic, B.; Pavone, M.; Geiger, A.; and Chitta, K. 2024. NAVSIM: Data-driven non-reactive autonomous vehicle simulation and benchmarking. *arXiv preprint arXiv:2406.15349*.

[^5]: Gao, H.; Chen, S.; Jiang, B.; Liao, B.; Shi, Y.; Guo, X.; Pu, Y.; Yin, H.; Li, X.; Zhang, X.; Zhang, Y.; Liu, W.; Zhang, Q.; and Wang, X. 2025. Rad: Training an end-to-end driving policy via large-scale 3dgs-based reinforcement learning. *arXiv preprint arXiv:2502.13144*.

[^6]: Gao, Y.; Zhang, Q.; Ding, D.-W.; and Zhao, D. 2024. Dream to Drive With Predictive Individual World Model. *IEEE Transactions on Intelligent Vehicles*, 9(12): 8224–8238.

[^7]: Gu, S.; Yin, W.; Jin, B.; Guo, X.; Wang, J.; Li, H.; Zhang, Q.; and Long, X. 2024. Dome: Taming diffusion model into high-fidelity controllable occupancy world model. *arXiv preprint arXiv:2410.10429*.

[^8]: Hu, A.; Russell, L.; Yeo, H.; Murez, Z.; Fedoseev, G.; Kendall, A.; Shotton, J.; and Corrado, G. 2023a. Gaia-1: A generative world model for autonomous driving. *arXiv preprint arXiv:2309.17080*.

[^9]: Hu, M.; Yin, W.; Zhang, C.; Cai, Z.; Long, X.; Chen, H.; Wang, K.; Yu, G.; Shen, C.; and Shen, S. 2024. Metric3d v2: A versatile monocular geometric foundation model for zero-shot metric depth and surface normal estimation. *IEEE Transactions on Pattern Analysis and Machine Intelligence*.

[^10]: Hu, S.; Chen, L.; Wu, P.; Li, H.; Yan, J.; and Tao, D. 2022. St-p3: End-to-end vision-based autonomous driving via spatial-temporal feature learning. In *Proceedings of the European Conference on Computer Vision*, 533–549. Springer.

[^11]: Hu, Y.; Yang, J.; Chen, L.; Li, K.; Sima, C.; Zhu, X.; Chai, S.; Du, S.; Lin, T.; Wang, W.; Lu, L.; Jia, X.; Liu, Q.; Dai, J.; Qiao, Y.; and Li, H. 2023b. Planning-oriented autonomous driving. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 17853–17862.

[^12]: Jia, X.; You, J.; Zhang, Z.; and Yan, J. 2025. Drivetransformer: Unified transformer for scalable end-to-end autonomous driving. In *Proceedings of the International Conference on Learning Representations*.

[^13]: Jiang, B.; Chen, S.; Xu, Q.; Liao, B.; Chen, J.; Zhou, H.; Zhang, Q.; Liu, W.; Huang, C.; and Wang, X. 2023. VAD: Vectorized scene representation for efficient autonomous driving. *Proceedings of the IEEE/CVF International Conference on Computer Vision*.

[^14]: Jiang, B.; Chen, S.; Zhang, Q.; Liu, W.; and Wang, X. 2025. AlphaDrive: Unleashing the power of VLMs in autonomous driving via reinforcement learning and reasoning. *arXiv preprint arXiv:2503.07608*.

[^15]: Jin, B.; Gu, S.; Hu, X.; Zheng, Y.; Guo, X.; Zhang, Q.; Long, X.; and Yin, W. 2025. OccTENS: 3D Occupancy World Model via Temporal Next-Scale Prediction. *arXiv preprint arXiv:2509.03887*.

[^16]: Li, P.; and Cui, D. 2025. Navigation-guided sparse scene representation for end-to-end autonomous driving. In *Proceedings of the International Conference on Learning Representations*, 15522–15533.

[^17]: Li, X.; Li, P.; Zheng, Y.; Sun, W.; Wang, Y.; and Chen, Y. 2025. Semi-supervised vision-centric 3d occupancy world model for autonomous driving. *arXiv preprint arXiv:2502.07309*.

[^18]: Li, Y.; Fan, L.; et al. 2025. Enhancing end-to-end autonomous driving with latent world model. *Proceedings of the International Conference on Learning Representations*.

[^19]: Li, Z.; Li, K.; Wang, S.; Lan, S.; Yu, Z.; Ji, Y.; Li, Z.; Zhu, Z.; Kautz, J.; Wu, Z.; Jiang, Y.; and Alvarez, J. M. 2024. Hydra-mdp: End-to-end multimodal planning with multi-target hydra-distillation. *arXiv preprint arXiv:2406.06978*.

[^20]: Liao, B.; Chen, S.; et al. 2025. Diffusiondrive: Truncated diffusion model for end-to-end autonomous driving. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*.

[^21]: Min, C.; Zhao, D.; Xiao, L.; Zhao, J.; Xu, X.; Zhu, Z.; Jin, L.; Li, J.; Guo, Y.; Xing, J.; Jing, L.; Nie, Y.; and Dai, B. 2024. Driveworld: 4D pre-trained scene understanding via world models for autonomous driving. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 15522–15533.

[^22]: Prakash, A.; Chitta, K.; and Geiger, A. 2021. Multi-modal fusion transformer for end-to-end autonomous driving. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*.

[^23]: Ren, T.; Liu, S.; Zeng, A.; Lin, J.; Li, K.; Cao, H.; Chen, J.; Huang, X.; Chen, Y.; Yan, F.; Zeng, Z.; Zhang, H.; Li, F.; Yang, J.; Li, H.; Jiang, Q.; and Zhang, L. 2024. Grounded sam: Assembling open-world models for diverse visual tasks. *arXiv preprint arXiv:2401.14159*.

[^24]: Shao, Z.; Wang, P.; Zhu, Q.; Xu, R.; Song, J.; Bi, X.; Zhang, H.; Zhang, M.; Li, Y., Y.K.and Wu; and Guo, D. 2024. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. *arXiv preprint arXiv:2402.03300*.

[^25]: Shi, C.; Shi, S.; Sheng, K.; Zhang, B.; and Jiang, L. 2025. DriveX: Omni scene modeling for learning generalizable world knowledge in autonomous driving. *arXiv preprint arXiv:2505.19239*.

[^26]: Sun, W.; Lin, X.; Shi, Y.; Zhang, C.; Wu, H.; and Zheng, S. 2024. Sparsedrive: End-to-end autonomous driving via sparse scene representation. *arXiv preprint arXiv:2405.19620*.

[^27]: Tong, W.; Sima, C.; Wang, T.; Chen, L.; Wu, S.; Deng, H.; Gu, Y.; Lu, L.; Luo, P.; Lin, D.; and H., L. 2023. Scene as occupancy. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 8406–8415.

[^28]: Wang, J.; Chen, M.; Karaev, N.; Vedaldi, A.; Rupprecht, C.; and Novotny, D. 2025. Vggt: Visual geometry grounded transformer. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, 5294–5306.

[^29]: Weng, X.; Ivanovic, B.; Wang, Y.; Wang, Y.; and Pavone, M. 2024. PARA-Drive: Parallelized architecture for real-time autonomous driving. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 15449–15458.

[^30]: Xing, Z.; Zhang, X.; Hu, Y.; Jiang, B.; He, T.; Zhang, Q.; Long, X.; and Yin, W. 2025. Goalflow: Goal-driven flow matching for multimodal trajectories generation in end-to-end autonomous driving. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, 1602–1611.

[^31]: Yang, P.; Zheng, Y.; Zhang, Q.; Zhu, K.; Xing, Z.; Lin, Q.; Liu, Y.-F.; Su, Z.; and Zhao, D. 2025. UncAD: Towards Safe End-to-end Autonomous Driving via Online Map Uncertainty. *arXiv preprint arXiv:2504.12826*.

[^32]: Zhang, K.; Tang, Z.; Hu, X.; Pan, X.; Guo, X.; Liu, Y.; Huang, J.; Yuan, L.; Zhang, Q.; Long, X.; Cao, X.; and Yin, W. 2025. Epona: Autoregressive diffusion world model for autonomous driving. *arXiv preprint arXiv:2506.24113*.

[^33]: Zhao, G.; Ni, C.; Wang, X.; Zhu, Z.; Zhang, X.; Wang, Y.; Huang, G.; Chen, X.; Wang, B.; Zhang, Y.; Mei, W.; and Wang, X. 2025. DriveDreamer4D: World models are effective data machines for 4d driving scene representation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 12015–12026.

[^34]: Zheng, W.; Song, R.; Guo, X.; Zhang, C.; and Chen, L. 2024a. Genad: Generative end-to-end autonomous driving. In *European Conference on Computer Vision*, 87–104. Springer.

[^35]: Zheng, Y.; Xia, Z.; Zhang, Q.; Zhang, T.; Lu, B.; Huo, X.; Han, C.; Li, Y.; Yu, M.; Jin, B.; et al. 2024b. Preliminary investigation into data scaling laws for imitation learning-based end-to-end autonomous driving. *arXiv preprint arXiv:2412.02689*.

[^36]: Zheng, Y.; Yang, P.; Xing, Z.; Zhang, Q.; Zheng, Y.; Gao, Y.; Li, P.; Zhang, T.; Xia, Z.; Jia, P.; and Zhao, D. 2025. World4drive: End-to-end autonomous driving via intention-aware physical latent world model. *arXiv preprint arXiv:2507.00603*.
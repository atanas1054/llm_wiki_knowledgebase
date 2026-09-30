---
title: "ReDrive: Shaping Representations with World Modeling for End-to-End Driving"
source: "https://arxiv.org/html/2609.33854v1"
author:
published:
created: 2026-09-30
description:
tags:
  - "clippings"
---
Yueting Zhu Affiliation:  Huazhong University of Science & Technology    Shaoyu Chen Affiliation:  Horizon Robotics    Yuehao Song Affiliation:  Huazhong University of Science & Technology    Hui Sun Affiliation:  Horizon Robotics    Qian Zhang Affiliation:  Horizon Robotics    Wenyu Liu Affiliation:  Huazhong University of Science & Technology    Xinggang Wang Affiliation:  Huazhong University of Science & Technology

###### Abstract

Driving policies require capabilities of scene understanding and future evolution prediction. To achieve this goal, current end-to-end models typically construct complex perception-planning pipelines or introduce world models that explicitly predict future states, resulting in a complex system architecture. Inspired by the transferability of general-purpose visual representations, we argue that combining sufficiently strong visual representations with representation world modeling can support effective planning without relying on complex inference-time auxiliary modules. Based on this insight, we present ReDrive, an end-to-end driving framework that strengthens planning-oriented visual features via future representation prediction. To achieve this, ReDrive adopts a three-stage training pipeline consisting of driving video pretraining, joint world-modeling and planning training, and planner adaptation. This yields a strong planning-oriented representation and a high-performance planner, while requiring neither auxiliary perception modules nor future prediction at inference time. Experiments on NAVSIM demonstrate strong performance, achieving 91.0 PDMS on NAVSIM v1 and 90.8 EPDMS on NAVSIM v2. These results show that shaping representations with world modeling is sufficient to enable high-performance end-to-end planning while retaining a simple encoder-planner inference pipeline.

<sup>†</sup>

## 1 Introduction

End-to-end autonomous driving [^13] [^17] requires planning-oriented visual representations that capture scene semantics while reflecting future scene evolution. Scene semantics [^28] [^39] provide the essential context for trajectory planning. Future scene evolution [^40] [^38] provides predictive information for guiding future driving behaviors.

Prior work has explored diverse strategies for learning visual representations for planning. BEV-based methods [^28] [^13] [^17] leverage task-specific supervision to learn spatially structured BEV features. Inspired by the transferability of self-supervised visual representations trained from large-scale visual data without task-specific annotations [^32] [^35] [^2] [^1], recent works transfer these general-purpose representations to driving policies to provide strong visual priors for downstream planning [^39] [^42] [^43]. However, when used alone without auxiliary perception tasks, these general-purpose representations still underperform task-specific BEV representations in downstream planning.

World models provide another paradigm by explicitly modeling future scene evolution. Generative world models [^40] [^50] [^20] predict future observations in pixel space, capturing scene evolution through visual generation. However, modeling fine-grained future geometric details may distract from the scene dynamics most relevant to planning. Recent approaches model dynamics in latent space by predicting future representations, avoiding explicit pixel generation while focusing on higher-level scene evolution [^31] [^53]. These methods achieve planning performance comparable to perception-based approaches without auxiliary perception tasks. However, inference-time reliance on future representations introduces additional computational cost and pipeline complexity.

![[intro1 1.png|Refer to caption]]

(a) Existing visual representations.

Building on these observations, we argue that strong pretrained video representations combined with representation world modeling can support high-performance driving without complex auxiliary modules or inference-time future prediction. We propose ReDrive, an end-to-end driving framework that strengthens planning-oriented visual representations through trajectory-conditioned future prediction. ReDrive adopts a two-branch architecture that predicts ego trajectories and future representations conditioned on the ego trajectories. Jointly optimizing the two branches provides the visual encoder with additional supervision on planning-relevant scene evolution beyond direct trajectory supervision. At inference, ReDrive uses only the learned visual encoder and trajectory generator, with the future predictor completely removed.

ReDrive follows a three-stage training procedure consisting of drive video pretraining, joint training, and planner adaptation. We first initialize the video encoder with a general-purpose video pretrained model [^1] and further pretrain it on large-scale driving videos through masked visual modeling in latent feature space. We then jointly train the encoder with a trajectory generator and a future representation predictor. The trajectory generator learns to predict the ground-truth ego trajectory, while the future predictor uses the expert trajectory as a condition to predict the corresponding future representation. The shared encoder therefore receives supervision from both tasks, encouraging it to capture information about future scene evolution beyond motion supervision alone. Finally, we freeze the visual encoder and future predictor while adapting the planner with its on-policy rollout. We obtain the corresponding future representation of the rollout trajectory using the frozen future predictor. Therefore, we directly optimize the planner under supervision of the future representation modeling.

We evaluate ReDrive on both NAVSIM v1 [^7] and NAVSIM v2 [^5]. After training, the learned planning-oriented visual representations exhibit stronger planning capability, yielding a 3.5-point PDMS improvement over representations learned without our complete training pipeline. ReDrive demonstrates strong trajectory planning performance across both benchmarks, achieving a PDMS of 91.0 on NAVSIM v1 and an EPDMS of 90.8 on NAVSIM v2.

Our contributions are summarized as follows:

- We introduce ReDrive, a simple yet effective driving policy built upon a strong planning-oriented visual representation shaped with world modeling without the requirement of inference-time auxiliary architecture designs, e.g., perception heads and future predictors.
- We develop a three-stage training strategy that progressively incorporates future prediction supervision into trajectory learning and further optimizes the planner on its own rollouts under future representation modeling supervision.
- We evaluate ReDrive on the NAVSIM v1 and NAVSIM v2 benchmarks, demonstrating strong planning performance across both benchmarks.

## 2 Related Work

### 2.1 End-to-end Autonomous Driving

End-to-end autonomous driving aims to directly optimize driving decisions from sensor observations. Early approaches [^12] [^13] [^17] build scene representations and jointly optimize perception, prediction, and planning to support end-to-end trajectory generation. Subsequent methods [^29] [^52] [^54] [^44] adopt generative planners that directly model the trajectory distribution. Recent methods incorporate reinforcement learning [^8] [^9] [^18] or vision language model (VLM) prior [^16] [^36] [^25] to enhance planning performance.

### 2.2 Driving Representation Learning

Classical autonomous driving systems commonly build bird’s-eye-view (BEV) representations from multi-view camera inputs to provide spatially structured features for downstream perception and planning [^28] [^13] [^17]. Recent advances in self-supervised visual pretraining offer a different paradigm. Several works [^38] [^19] have explored general-purpose image representations, such as DINOv2 features [^32], as visual backbones for end-to-end driving. More recently, a series of works [^43] [^39] [^42] further incorporate pretrained video representations [^2] [^1] into driving scene understanding for better temporal semantic extraction. Another direction predicts compact action-oriented representations for driving decisions [^46].

### 2.3 Driving World Action Models

World action models leverage explicit scene evolution prediction to enhance trajectory generation. Existing methods typically forecast future states in BEV or latent representations and use them for trajectory generation, candidate selection, or online evaluation [^24] [^51] [^11] [^49]. Other approaches connect world representations with planning policies through future visual generation [^50] [^40] [^41]. These methods mainly use predicted futures or world representations as additional information for planning, making trajectory decisions directly dependent on future prediction. In contrast, we use future dynamics to improve the video representation used for planning, enabling direct trajectory planning with a stronger representation.

## 3 Method

### 3.1 Overview

![[overview 1.png|Refer to caption]]

Figure 2: Overview of ReDrive and its three-stage training procedure. Self-Supervised Pretraining adapts the video encoder to the driving domain. Joint Training learns trajectory generation and future representation prediction from the shared history representation. Planner Adaptation freezes the encoder and future predictor and uses planner-generated trajectories as the condition for predictive supervision.

Our goal is to demonstrate that strong planning-oriented visual representations alone can support high-performance driving planning. Such representations should capture not only the current driving scene but also its potential future evolution. Accordingly, ReDrive introduces future representation prediction as a training signal and progressively incorporates it through three stages: Driving-Domain Pretraining, Joint Training, and Planner Adaptation, as illustrated in Fig. 2.

### 3.2 Driving-Domain Pretraining

To provide a driving-specific temporal representation for subsequent planning-oriented learning, we first perform self-supervised pretraining on driving videos. We initialize the encoder from a video-pretrained V-JEPA2 [^1] backbone and further adapt it to the driving domain. We perform latent masked visual modeling style representation learning [^2] to implement continual pretraining. Specifically, we mask a set of spatiotemporal tokens $\mathcal{M}$ and predicts their latent representations from the visible context. We utilize a context encoder to extract representations $z$ from visible tokens, while the EMA encoder extracts target representations $z^{\mathrm{tgt}}$ from the unmasked contents. We project $z$ to predictive representation $\hat{z}$ using a representation predictor to predict the target representations of the masked regions from the visible context. The pretraining objective is defined as

$$
\mathcal{L}_{\mathrm{ssl}}=\frac{1}{|\mathcal{M}|}\sum_{i\in\mathcal{M}}\left\|\hat{z}_{i}-z_{i}^{\mathrm{tgt}}\right\|_{1},
$$

where $\hat{z}_{i}$ and $z_{i}^{\mathrm{tgt}}$ denote the predicted and target representations at masked position $i$, respectively, and the $\ell_{1}$ distance is averaged over the feature dimension. By optimizing this masked prediction objective on driving videos, the pretrained video representation is adapted to the driving domain. The resulting context encoder is then used to initialize the video encoder for subsequent future feature prediction and trajectory planning.

### 3.3 Joint Training

To further shape the video representation with future-aware information relevant to planning, we jointly optimize future representation prediction and trajectory planning through a shared encoder. We introduce two branches to perform these two tasks. The future prediction branch learns future representations conditioned on the ground-truth ego trajectory, while the planning branch directly generates the ego trajectory from the history representation. This joint training introduces future-aware supervision into the video representation while keeping trajectory generation decoupled from future prediction.

#### 3.3.1 Future Representation Prediction

We divide the video into a history clip $X_{1:T}$ and a non-overlapping future clip $X_{T+1:2T}$. The history clip is encoded by $E_{\theta}$ and the future clip is processed by the EMA target encoder to provide the prediction target. We condition future feature prediction on the ground-truth ego trajectory corresponding to the future clip $X_{T+1:2T}$. The trajectory is encoded into action tokens $A$, which are incorporated into the predictor through cross-attention with the history representation $Z_{h}$ to predict the future representation,

$$
\displaystyle Z_{h}
$$
 
$$
\displaystyle=E_{\theta}(X_{1:T}),\qquad Z_{f}^{\mathrm{tgt}}=E^{\mathrm{tgt}}(X_{T+1:2T}),\qquad\hat{Z}_{f}=P_{\phi}(Z_{h},A).
$$

We regress the predicted representation $\hat{Z}_{f}$ toward the stop-gradient target $Z_{f}^{\mathrm{tgt}}$ after feature normalization,

$$
\mathcal{L}_{\mathrm{feat}}=\frac{1}{N}\left\|\operatorname{Norm}(\hat{Z}_{f})-\operatorname{Norm}(Z_{f}^{\mathrm{tgt}})\right\|_{1},
$$

where $N$ denotes the number of elements in the future representation.

The objective updates both the predictor $P_{\phi}$ and the video encoder $E_{\theta}$. Future representation prediction encourages the encoder to capture scene evolution conditioned on ego motion. This enriches the visual representation with future dynamics.

#### 3.3.2 Trajectory Generation

The planning branch adopts an action DiT [^33] to generate trajectories directly from the video representation $Z_{h}$. During training, we construct a noisy trajectory from the ground-truth trajectory $\tau$ and Gaussian noise $\epsilon\sim\mathcal{N}(0,I)$ at noise level $t$

$$
\tau_{t}=(1-t)\tau+t\epsilon.
$$

We embed the noise level $t$ and the ego status $c$ into noisy action tokens $\tau_{t}$ via AdaLN [^33]. The action DiT incorporates the history representation $Z_{h}$ through cross-attention as the visual condition and predicts the velocity $\hat{v}_{t}$. Following the linear noising path in Eq. 4, we obtain the target velocity $v_{t}$ and optimize the Action DiT with the flow-matching objective

$$
\displaystyle\hat{v}_{t}=G_{\psi}(\tau_{t},t,c,Z_{h}),\qquad v_{t}=\epsilon-\tau,\qquad\mathcal{L}_{\mathrm{plan}}=\left|\hat{v}_{t}-v_{t}\right|_{2}^{2},
$$

where $G_{\psi}$ predicts the velocity field for trajectory generation.

#### 3.3.3 Joint Optimization

The future prediction and planning branches are jointly optimized through the video encoder $E_{\theta}$. The overall objective is

$$
\mathcal{L}=\mathcal{L}_{\mathrm{plan}}+\lambda\mathcal{L}_{\mathrm{feat}},
$$

where $\lambda$ controls the weight of future feature prediction loss. The planning objective updates $E_{\theta}$ and $G_{\psi}$, while the future prediction objective updates $E_{\theta}$ and $P_{\phi}$. This decoupling prevents the prediction objective from steering the trajectory policy toward actions that are easier to predict rather than better for planning. The action DiT is therefore optimized solely by the planning objective, while future prediction improves planning by shaping a stronger video representation.

### 3.4 Planner Adaptation

In the final stage, we further optimize the planned directly using the supervision from the future predictive information on on-policy samples. Specifically, we use the feature prediction loss to supervise the planner through the fixed predictor, alongside the trajectory supervision.

During this stage, the video encoder $E_{\theta}$ and future predictor $P_{\phi}$ are frozen, and only the action DiT is optimized. Starting from Gaussian noise, the Action DiT performs a multi-step rollout to generate an ego trajectory $\hat{\tau}$. The generated trajectory is encoded into action tokens $\hat{A}$ and fed into the frozen future predictor, replacing the ground-truth trajectory condition in Eq. 2. The resulting feature prediction loss $\mathcal{L}_{\mathrm{feat}}$ is propagated through the generated trajectory to optimize the planner.

For trajectory supervision, we apply the flow-matching objective at each denoising step and an L1 loss on the final generated trajectory. The overall adaptation objective is

$$
\mathcal{L}_{\mathrm{plan}}=\mathcal{L}_{\mathrm{fm}}+\lambda_{\mathrm{traj}}\mathcal{L}_{\mathrm{traj}},\qquad\mathcal{L}_{\mathrm{adapt}}=\mathcal{L}_{\mathrm{plan}}+\lambda_{\mathrm{feat}}\mathcal{L}_{\mathrm{feat}}.
$$

The video encoder and future predictor remain frozen to preserve the future dynamics learned under ground-truth trajectory conditioning. Imperfect planner-generated trajectories could otherwise introduce erroneous supervision into the future predictor and alter the learned correspondence between ego motion and future scene evolution. Consequently, all adaptation objectives update only the Action DiT.

## 4 Experiments

### 4.1 Experimental Setup

Datasets. We conduct downstream trajectory planning experiments on NAVSIM [^7], covering both the v1 and v2 benchmarks, including the challenging NavHard evaluation split of NAVSIM v2. NAVSIM [^7] provides approximately 103K training scenes and 12K test scenes for standardized planning evaluation. For driving-domain video pretraining, we use videos from nuScenes [^3] and nuPlan [^4], together with an 80-hour subset of the NVIDIA PhysicalAI-Autonomous-Vehicles Dataset <sup>1</sup>. These durations refer to the source collection before filtering by dataset split. Only videos from the training splits of nuScenes [^3] and nuPlan [^4] are used for pretraining. The pretrained video encoder is subsequently used to initialize the predictive adaptation and trajectory planning stages on NAVSIM.

Implementation Details. We initialize the video encoder from the pretrained V-JEPA2 [^1] model and further pretrain it for 100 epochs on the driving videos using 16-frame clips. We then jointly train the video encoder, future predictor, and action DiT for 35K steps on the NAVTRAIN split of NAVSIM [^7], where four observed frames are used to predict the subsequent four frames. The planner adaptation stage further trains the action DiT for 5K steps with the video encoder and future predictor frozen. We use five denoising steps for trajectory rollout during this stage. The input resolution is $256\times 512$. The future predictor consists of 12 Transformer blocks with 12 attention heads and a hidden dimension of 768. All experiments run on 16 NVIDIA H20 GPUs. More details are provided in Sec. A.1 of the supplementary material.

Evaluation Metrics. We evaluate planning performance on both NAVSIM v1 and NAVSIM v2. For NAVSIM v1, we report the Predictive Driver Model Score (PDMS) together with its individual components, including no-at-fault collision (NC), drivable area compliance (DAC), time-to-collision (TTC), ego progress (EP), and comfort(C). For NAVSIM v2, we report the Extended Predictive Driver Model Score (EPDMS) as the primary aggregate metric following the official evaluation protocol. On NavHard, we report the individual metric components and per-stage scores (S.) for Stage-1 and Stage-2, together with the combined EPDMS.

### 4.2 Comparison with State-of-the-Art Methods

NAVSIM v1. Tab. 1 compares ReDrive with state-of-the-art methods on NAVSIM v1. Among perception-free approaches, ReDrive achieves the best PDMS of 91.0, outperforming the previous best ReWorld [^41] by 0.6 points. Notably, despite relying only on camera inputs and without explicit perception modules, ReDrive surpasses several perception-based methods, including Hydra-MDP [^26], DiffusionDrive [^29], GoalFlow [^44], and DriveDPO [^34] in terms of PDMS. These results show that strong predictive visual representations can enable a direct planner to achieve competitive performance without relying on explicit perception modules.

Table 1: Comparison with state-of-the-art methods on NAVSIM v1. The best and second-best results are highlighted in bold and underlined, respectively.

<table><tbody><tr><td>Type</td><td>Method</td><td>Inputs</td><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>C <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td rowspan="8">Perception-based</td><td>Transfuser <sup><a href="#fn:6">6</a></sup></td><td>C + L</td><td>97.7</td><td>92.8</td><td>79.2</td><td>100</td><td>92.8</td><td>84.0</td></tr><tr><td>VADv2 <sup><a href="#fn:15">15</a></sup></td><td>Camera</td><td>97.2</td><td>89.1</td><td>91.6</td><td>100</td><td>76.0</td><td>80.9</td></tr><tr><td>UniAD <sup><a href="#fn:13">13</a></sup></td><td>Camera</td><td>97.8</td><td>91.9</td><td>92.9</td><td>100</td><td>78.8</td><td>83.4</td></tr><tr><td>Hydra-MDP <sup><a href="#fn:26">26</a></sup></td><td>C + L</td><td>98.4</td><td>97.7</td><td>85.0</td><td>100</td><td>94.5</td><td>89.9</td></tr><tr><td>Hydra-MDP++ <sup><a href="#fn:22">22</a></sup></td><td>C + L</td><td>97.6</td><td>96.0</td><td>80.4</td><td>100</td><td>93.1</td><td>86.6</td></tr><tr><td>DiffusionDrive <sup><a href="#fn:29">29</a></sup></td><td>C + L</td><td>98.2</td><td>96.2</td><td>82.2</td><td>100</td><td>94.7</td><td>88.1</td></tr><tr><td>GoalFlow <sup><a href="#fn:44">44</a></sup></td><td>C + L</td><td>98.4</td><td>98.3</td><td>85.0</td><td>100</td><td>94.6</td><td>90.3</td></tr><tr><td>DriveDPO <sup><a href="#fn:34">34</a></sup></td><td>C + L</td><td>98.5</td><td>98.1</td><td>84.3</td><td>100</td><td>94.8</td><td>90.0</td></tr><tr><td></td><td>DriveSuprim <sup><a href="#fn:48">48</a></sup></td><td>Camera</td><td>98.6</td><td>98.6</td><td>91.3</td><td>100</td><td>95.5</td><td>89.9</td></tr><tr><td rowspan="8">Perception-free</td><td>LAW <sup><a href="#fn:23">23</a></sup></td><td>C + L</td><td>97.4</td><td>93.3</td><td>78.8</td><td>100</td><td>91.9</td><td>83.8</td></tr><tr><td>World4Drive <sup><a href="#fn:51">51</a></sup></td><td>C + L</td><td>97.4</td><td>94.3</td><td>79.9</td><td>100</td><td>92.8</td><td>85.1</td></tr><tr><td>Epona <sup><a href="#fn:50">50</a></sup></td><td>Camera</td><td>97.9</td><td>95.1</td><td>80.4</td><td>99.9</td><td>93.8</td><td>86.2</td></tr><tr><td>Drive-JEPA <sup><a href="#fn:39">39</a></sup></td><td>Camera</td><td>98.7</td><td>96.2</td><td>82.9</td><td>100</td><td>95.5</td><td>89.0</td></tr><tr><td>DriveLaW <sup><a href="#fn:40">40</a></sup></td><td>Camera</td><td>99.0</td><td>97.1</td><td>81.3</td><td>100</td><td>96.7</td><td>89.1</td></tr><tr><td>ReWorld <sup><a href="#fn:41">41</a></sup></td><td>Camera</td><td>99.1</td><td>98.2</td><td>82.0</td><td>99.8</td><td>97.7</td><td>90.4</td></tr><tr><td>DAWN <sup><a href="#fn:31">31</a></sup></td><td>Camera</td><td>98.7</td><td>95.9</td><td>84.3</td><td>100</td><td>96.0</td><td>89.1</td></tr><tr><td>ReDrive (Ours)</td><td>Camera</td><td>99.1</td><td>97.9</td><td>84.3</td><td>100</td><td>97.2</td><td>91.0</td></tr></tbody></table>

NAVSIM v2. Tab. 2 compares ReDrive with state-of-the-art methods on NAVSIM v2. ReDrive achieves the best overall performance and outperforms the second-best SparseDriveV2 [^37] by 0.7 points and also exceeds recent world-model and world-action methods such as Latent-WAM [^38], DriveFuture [^11], CoWorld-VLA [^14], and DreamerAD [^47]. These results further show that predictive visual representations can support strong planning performance on NAVSIM v2 without relying on an explicit future model during inference.

Table 2: Comparison with state-of-the-art methods on NAVSIM v2. The best and second-best results are highlighted in bold and underlined, respectively. NC–EC uniformly report the corrected-evaluator metrics.

| Method | NC $\uparrow$ | DAC $\uparrow$ | DDC $\uparrow$ | TLC $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | LK $\uparrow$ | HC $\uparrow$ | EC $\uparrow$ | EPDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DiffusionDrive [^29] | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | 84.5 |
| DiffusionDriveV2 [^54] | 97.7 | 96.6 | 99.2 | 99.8 | 88.9 | 97.2 | 96.0 | 97.8 | 91.0 | 87.5 |
| WAM-Diff [^45] | 99.0 | 98.4 | 99.3 | 99.9 | 87.0 | 98.6 | 96.2 | 98.1 | 78.5 | 89.7 |
| DreamerAD [^47] | 98.0 | 97.2 | 99.5 | 99.8 | 87.8 | 97.4 | 97.5 | 98.3 | 72.4 | 87.7 |
| Latent-WAM [^38] | 98.1 | 97.3 | 99.6 | 99.8 | 87.7 | 97.3 | 97.6 | 98.1 | 87.3 | 89.3 |
| DriveFuture [^11] | 98.8 | 99.1 | 99.6 | 99.9 | 86.6 | 98.4 | 96.4 | 98.3 | 74.8 | 89.9 |
| CoWorld-VLA [^14] | 99.1 | 97.0 | 99.6 | 99.9 | 87.9 | 98.5 | 97.7 | 98.2 | 86.2 | 90.0 |
| SparseDriveV2 [^37] | 98.1 | 98.1 | 99.6 | 99.8 | 91.1 | 97.3 | 96.9 | 98.2 | 78.4 | 90.1 |
| ReDrive (Ours) | 99.1 | 97.8 | 99.6 | 99.9 | 87.6 | 98.7 | 98.2 | 98.3 | 86.4 | 90.8 |

NAVSIM v2 NavHard. Tab. 3 compares ReDrive with state-of-the-art methods on NAVSIM v2 NavHard. ReDrive outperforms Metis [^20] and DiffusionDrive [^29] in both per-stage scores and combined EPDMS. Its strengths are most pronounced in Stage-1, where it leads the compared methods in collision avoidance, drivable-area compliance, driving-direction compliance, TTC, and lane keeping. In Stage-2, ReDrive improves collision avoidance and drivable-area compliance over Metis [^20]. This pattern suggests stronger safety and road compliance, with room to improve motion comfort under the more challenging evaluation conditions.

Table 3: Comparison with state-of-the-art methods on NAVSIM v2 NavHard. S1 and S2 denote Stage-1 and Stage-2, respectively. S. denotes the per-stage score, and EPDMS reports the combined score. The best and second-best results are highlighted in bold and underlined, respectively.

<table><tbody><tr><td>Method</td><td>Stage</td><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DDC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TLC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>LK <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>HC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>S.<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EPDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td rowspan="2">LTF <sup><a href="#fn:6">6</a></sup></td><td>S1</td><td>97.3</td><td>80.2</td><td>97.8</td><td>99.3</td><td>83.4</td><td>96.2</td><td>92.9</td><td>97.8</td><td>71.1</td><td>61.3</td><td rowspan="2">24.4</td></tr><tr><td>S2</td><td>79.4</td><td>69.0</td><td>85.6</td><td>98.5</td><td>83.8</td><td>76.7</td><td>47.9</td><td>97.0</td><td>70.6</td><td>39.2</td></tr><tr><td rowspan="2">DiffusionDrive <sup><a href="#fn:29">29</a></sup></td><td>S1</td><td>96.8</td><td>86.0</td><td>98.8</td><td>99.3</td><td>84.0</td><td>95.8</td><td>96.7</td><td>97.6</td><td>79.6</td><td>66.7</td><td rowspan="2">27.5</td></tr><tr><td>S2</td><td>80.1</td><td>72.8</td><td>84.4</td><td>98.4</td><td>85.9</td><td>76.6</td><td>46.4</td><td>96.3</td><td>72.8</td><td>40.5</td></tr><tr><td rowspan="2">GTRS-DP <sup><a href="#fn:27">27</a></sup></td><td>S1</td><td>94.7</td><td>78.8</td><td>96.1</td><td>99.5</td><td>83.0</td><td>94.4</td><td>92.0</td><td>97.5</td><td>72.8</td><td>–</td><td rowspan="2">23.8</td></tr><tr><td>S2</td><td>80.3</td><td>74.4</td><td>84.9</td><td>98.0</td><td>81.9</td><td>78.8</td><td>45.4</td><td>96.7</td><td>70.1</td><td>–</td></tr><tr><td rowspan="2">GuideFlow <sup><a href="#fn:30">30</a></sup></td><td>S1</td><td>96.6</td><td>80.5</td><td>96.3</td><td>99.3</td><td>82.3</td><td>94.9</td><td>91.5</td><td>97.7</td><td>67.8</td><td>–</td><td rowspan="2">27.1</td></tr><tr><td>S2</td><td>87.3</td><td>76.7</td><td>88.8</td><td>99.2</td><td>84.3</td><td>85.1</td><td>49.7</td><td>93.1</td><td>44.5</td><td>–</td></tr><tr><td rowspan="2">ReCogDrive <sup><a href="#fn:25">25</a></sup></td><td>S1</td><td>96.4</td><td>78.9</td><td>98.7</td><td>99.8</td><td>82.6</td><td>95.6</td><td>94.4</td><td>97.6</td><td>74.2</td><td>67.7</td><td rowspan="2">25.7</td></tr><tr><td>S2</td><td>80.2</td><td>65.0</td><td>82.4</td><td>98.7</td><td>85.2</td><td>76.9</td><td>43.8</td><td>96.6</td><td>71.8</td><td>37.6</td></tr><tr><td rowspan="2">SGDrive <sup><a href="#fn:21">21</a></sup></td><td>S1</td><td>95.8</td><td>87.6</td><td>97.8</td><td>99.8</td><td>84.4</td><td>94.7</td><td>92.9</td><td>97.8</td><td>28.9</td><td>71.1</td><td rowspan="2">25.5</td></tr><tr><td>S2</td><td>79.4</td><td>65.4</td><td>79.1</td><td>98.9</td><td>88.9</td><td>75.3</td><td>42.7</td><td>96.4</td><td>29.6</td><td>35.2</td></tr><tr><td rowspan="2">Metis <sup><a href="#fn:20">20</a></sup></td><td>S1</td><td>96.6</td><td>87.8</td><td>99.0</td><td>99.3</td><td>84.5</td><td>95.6</td><td>97.8</td><td>97.8</td><td>77.8</td><td>75.8</td><td rowspan="2">32.2</td></tr><tr><td>S2</td><td>79.6</td><td>73.3</td><td>84.9</td><td>97.8</td><td>85.8</td><td>76.6</td><td>47.7</td><td>95.4</td><td>75.3</td><td>41.7</td></tr><tr><td rowspan="2">ReDrive (Ours)</td><td>S1</td><td>97.9</td><td>93.6</td><td>99.4</td><td>99.6</td><td>84.2</td><td>96.9</td><td>97.8</td><td>97.8</td><td>73.3</td><td>82.3</td><td rowspan="2">34.4</td></tr><tr><td>S2</td><td>82.2</td><td>75.8</td><td>84.3</td><td>98.2</td><td>87.3</td><td>77.9</td><td>47.8</td><td>95.9</td><td>54.2</td><td>42.3</td></tr></tbody></table>

### 4.3 Ablation Studies

Effect of Visual Pretraining. We compare different pretraining strategies and temporal lengths in Tab. 5. For a fair comparison, DINOv2 [^32], MAE [^10], and our Stage 1 pretraining use encoders with comparable parameter counts initialized from their respective pretrained checkpoints. All encoders are further pretrained on the same driving data for the same number of epochs and evaluated after the same Stage 2 training. DINOv2 outperforms MAE, suggesting that stronger visual representations benefit trajectory planning. Our video-based pretraining further improves over both image-based baselines, demonstrating the additional benefit of temporal representation learning. Increasing the number of frames consistently improves performance, further highlighting the importance of temporal context for planning.

Table 4: Ablation on visual pretraining. We compare image- and video-based pretraining and study the temporal horizon used in pretraining. All results are evaluated after Stage 2 training.

<table><tbody><tr><td>Pretraining</td><td>DINOv2</td><td>MAE</td><td colspan="4">V-JEPA2</td></tr><tr><td>Frames</td><td>–</td><td>–</td><td>4</td><td>8</td><td>12</td><td>16</td></tr><tr><td>PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>84.4</td><td>83.9</td><td>89.9</td><td>90.4</td><td>90.7</td><td>90.8</td></tr></tbody></table>

Table 5: Ablation on encoder fine-tuning and future prediction in Stage 2, with the encoder pretrained on 4-frame driving videos.

| Encoder | Future Predictor | PDMS $\uparrow$ |
| --- | --- | --- |
| Frozen | ✗ | 85.7 |
| Trainable | ✗ | 89.2 |
| Trainable | ✓ | 89.9 |

Effect of Encoder Adaptation and Future Prediction. During Stage 2, starting from the same encoder pretrained on 4-frame driving videos, we ablate whether the encoder is frozen and whether the Future Predictor is jointly trained with the planner in Tab. 5. Fine-tuning the encoder with the planning objective improves PDMS from 85.7 to 89.2. Jointly training the Future Predictor further improves PDMS to 89.9, showing that future representation prediction provides complementary supervision beyond the planning objective for learning planning-oriented visual representations.

Effect of Training Stages. We evaluate each stage of the training pipeline in Tab. 7. Starting from the original V-JEPA2 model, Stage 1 pretrains the encoder on 4-frame driving videos, improving PDMS from 88.9 to 89.2. Stage 2 jointly trains the Encoder, Future Predictor, and Planner, improving PDMS from 89.2 to 89.9. Stage 3 further conditions future prediction on planner-generated trajectories, improving PDMS to 90.2. These results demonstrate the complementary contributions of the three stages.

Table 6: Ablation on the three-stage training pipeline. The Encoder and Planner are jointly trained for evaluation, with the Future Predictor introduced in Stage 2 and Planner-only adaptation in Stage 3.

| Stage 1 | Stage 2 | Stage 3 | PDMS $\uparrow$ |
| --- | --- | --- | --- |
| ✗ | ✗ | ✗ | 88.9 |
| ✓ | ✗ | ✗ | 89.2 |
| ✓ | ✓ | ✗ | 89.9 |
| ✓ | ✓ | ✓ | 90.2 |

Table 7: Planning evaluation of frozen visual encoders with newly trained planners. Stage 2 joint training is included for reference. All models use 8-frame pretraining.

<table><tbody><tr><td>    Encoder    </td><td>    Planner    </td><td>    PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math>     </td></tr><tr><td>    8-frame Pretrained    </td><td>    Reinitialized    </td><td>    86.7    </td></tr><tr><td>    Stage 2 Trained    </td><td>    Reinitialized    </td><td>    90.2    </td></tr><tr><td colspan="2">    Stage 2 Joint Training (reference)    </td><td>    90.4    </td></tr></tbody></table>

### 4.4 Experimental Analysis

ReDrive Indeed Improves Representations. We further evaluate whether Stage 2 improves the visual representation itself. With the encoder frozen and a newly initialized planner, the Stage 2 encoder achieves 90.2 PDMS, substantially outperforming the pretrained encoder at 86.7 and approaching the 90.4 of Stage 2 joint training. This shows that the improvement from Stage 2 is largely retained in the visual representation and can be directly utilized for planning.

Qualitative Results. Fig. 3 compares the human trajectory, Drive-JEPA [^39], and ReDrive in representative driving scenarios using both front-camera and BEV visualizations. Compared with Drive-JEPA, ReDrive produces trajectories that more closely follow the human trajectory while maintaining better consistency with the road geometry and surrounding traffic. The visual comparison provides qualitative evidence that the learned representation supports more accurate and scene-aware planning. More visualizations are provided in Sec. A.2.

![[vis 2.png|Refer to caption]]

Figure 3: Qualitative comparison of planning results using front-camera and bird’s-eye-view (BEV) visualizations. We compare the human trajectory, Drive-JEPA 39, and ReDrive across representative driving scenarios.

Future Predictor Analysis. We analyze how the Future Predictor responds to different action conditions under the same scene context. For each scene, we keep the history representation fixed and laterally shift the ground-truth trajectory using 15 offsets uniformly spaced from $-0.8$ to $0.8$ in the normalized action space. Each shifted trajectory is fed into the Future Predictor together with the fixed history representation to obtain the corresponding predicted future representation. Fig. 4(a) plots the cosine similarities between representations predicted under different trajectory conditions, while Fig. 4(b) presents the corresponding pairwise similarity matrix. The cosine similarity generally decreases as the difference between lateral offsets increases, indicating that the predicted representations are sensitive to the trajectory condition even when the scene context remains unchanged. More visualizations are provided in Sec. A.3.

![[analysis.png|Refer to caption]]

Figure 4: Sensitivity of the Future Predictor to trajectory conditions with the history representation held fixed. (a) Cosine similarities between future representations predicted under different lateral trajectory offsets. The horizontal axis and point color indicate the offsets of the two trajectory conditions being compared. (b) Pairwise cosine similarity matrix for all trajectory conditions.

## 5 Conclusion

We presented ReDrive, an end-to-end driving framework that learns planning-oriented visual representations through trajectory-conditioned future representation prediction. By jointly learning trajectory generation and action-conditioned future representation prediction, ReDrive incorporates supervision on future scene evolution into visual representation learning. The planner is further adapted using its own generated trajectories with supervision from the frozen future predictor. Experiments on NAVSIM v1 and v2, including NavHard, demonstrate strong planning performance. The future predictor is removed at inference, enabling direct planning without additional modules.

## References

## Appendix A Appendix

### A.1 Training Details

#### A.1.1 Training Data

nuPlan. nuPlan [^4] is a large-scale autonomous driving dataset and planning benchmark containing approximately 1,200 hours of real-world driving data collected in Boston, Pittsburgh, Las Vegas, and Singapore. Among them, 120 hours are released with the full sensor suite, including eight cameras, five LiDARs, an IMU, and GPS, together with detailed HD maps and automatically generated 3D annotations. The dataset covers more than 30 types of driving scenarios, including lane changes, unprotected turns, and interactions with pedestrians. For pretraining, we use the front-view camera sequences from the NAVTRAIN sensor data derived from nuPlan.

nuScenes. nuScenes [^3] is a large-scale multimodal autonomous driving dataset collected in Boston and Singapore. It contains 1,000 driving scenes, each approximately 20 seconds long, with 1.4 million camera images and 390,000 LiDAR sweeps. The sensor suite consists of six cameras, one LiDAR, five radars, an IMU, and GPS, providing complete 360-degree observations of the surrounding environment. The official dataset is divided into 700 training scenes, 150 validation scenes, and 150 test scenes. For pretraining, we use the CAM\_FRONT stream from all 700 training scenes.

PhysicalAI-Autonomous-Vehicles Dataset. PhysicalAI-Autonomous-Vehicles Dataset is a large-scale multi-sensor autonomous driving dataset released by NVIDIA. The full dataset contains approximately 1,700 hours of driving data and 306,152 clips, where each clip has a duration of 20 seconds. The data are collected across 25 countries and more than 2,500 cities, covering diverse traffic, weather, road, and geographic conditions. For pretraining, we use an 80-hour subset and retain the front wide-angle camera with a $120^{\circ}$ field of view.

![[camera_alignment.png|Refer to caption]]

Figure 5: Illustration of camera alignment for PhysicalAI-Autonomous-Vehicles Dataset. The original wide-angle camera views are geometrically transformed using per-clip calibration to obtain aligned views with nuPlan-style camera geometry. The alignment reduces the discrepancy in camera projection and view distribution before self-supervised pretraining.

![[sup2_vis.png|Refer to caption]]

Figure 6: Additional qualitative comparison of the human trajectory, Drive-JEPA, and ReDrive using front-camera and bird’s-eye-view (BEV) visualizations.

#### A.1.2 Camera Alignment.

The camera configurations of the PhysicalAI-Autonomous-Vehicles Dataset and nuPlan [^4] differ substantially in field of view and projection model. To reduce this domain discrepancy, we geometrically align the PhysicalAI-Autonomous-Vehicles Dataset front wide-angle videos to the nuPlan front-camera view before pretraining, as shown in Fig. 5. Specifically, we use the per-clip f-theta camera calibration provided by the PhysicalAI-Autonomous-Vehicles Dataset to map each target nuPlan pixel ray back to the corresponding source image location, followed by bilinear resampling. The resulting videos approximately match the field of view and camera intrinsics of the nuPlan front camera. We additionally discard clips whose calibrated field of view is insufficient to cover the target view, avoiding extrapolation during remapping.

#### A.1.3 Training Procedure

Pretraining. We initialize the visual encoder from the publicly released V-JEPA2 ViT-L [^1] checkpoint and further pretrain it on a mixture of driving videos from nuPlan [^4], nuScenes [^3], and the PhysicalAI-Autonomous-Vehicles Dataset. The three datasets are uniformly sampled during training. Each training clip contains 16 frames sampled at 10 Hz with a spatial resolution of $256\times 512$. We follow the V-JEPA2 pretraining recipe and adopt a ViT-L encoder together with a lightweight predictor and an EMA target encoder. A multi-block masking strategy is applied over the full temporal extent, and the predictor learns to reconstruct the masked latent representations produced by the target encoder. After pretraining, only the visual encoder is retained for subsequent training, while the pretraining predictor and target encoder are discarded. The detailed pretraining configuration is summarized in Tab. 8.

Table 8: Pretraining configuration.

<table><tbody><tr><td>       Module</td><td>       Configuration</td><td>       Setting</td></tr><tr><td rowspan="5">       Encoder</td><td>       Architecture</td><td>       ViT-L</td></tr><tr><td>       Depth</td><td>       24</td></tr><tr><td>       Hidden dimension</td><td>       1024</td></tr><tr><td>       Attention heads</td><td>       16</td></tr><tr><td>       Parameters</td><td>       304M</td></tr><tr><td rowspan="4">       Predictor</td><td>       Depth</td><td>       12</td></tr><tr><td>       Hidden dimension</td><td>       384</td></tr><tr><td>       Attention heads</td><td>       12</td></tr><tr><td>       Parameters</td><td>       22M</td></tr><tr><td rowspan="4">       Input</td><td>       Frames</td><td>       16</td></tr><tr><td>       Frame rate</td><td>       10 Hz</td></tr><tr><td>       Resolution</td><td>        <math><semantics><mrow><mn>256</mn> <mo>×</mo> <mn>512</mn></mrow> <annotation>256\times 512</annotation></semantics></math></td></tr><tr><td>       Patch / Tubelet size</td><td>        <math><semantics><mrow><mn>16</mn> <mo>×</mo> <mn>16</mn></mrow> <annotation>16\times 16</annotation></semantics></math> / 2</td></tr><tr><td rowspan="2">       Masking</td><td>       Small blocks</td><td>       8, scale 0.15</td></tr><tr><td>       Large blocks</td><td>       2, scale 0.70</td></tr><tr><td rowspan="5">       Optimization</td><td>       Optimizer</td><td>       AdamW</td></tr><tr><td>       Learning rate</td><td>        <math><semantics><mrow><mn>5.25</mn> <mo>×</mo> <msup><mn>10</mn> <mrow><mo>−</mo> <mn>4</mn></mrow></msup></mrow> <annotation>5.25\times 10^{-4}</annotation></semantics></math></td></tr><tr><td>       Weight decay</td><td>       0.04</td></tr><tr><td>       Precision</td><td>       bfloat16</td></tr><tr><td>       Epochs</td><td>       100</td></tr><tr><td rowspan="2">       Target Encoder</td><td>       Update</td><td>       EMA</td></tr><tr><td>       Decay</td><td>       0.99925</td></tr></tbody></table>

Joint Training. After pretraining, we retain the visual encoder and jointly optimize it with a newly initialized Future Predictor and Action DiT. Each sample contains four observed frames at $256\times 512$ resolution. The Future Predictor takes the history representation and ego trajectory as conditions to predict the target representations of the subsequent four frames, while the Action DiT predicts an 8-step trajectory covering 4 s at 2 Hz. A target encoder is maintained as an exponential moving average (EMA) of the visual encoder with a decay of 0.9999 and remains gradient-free throughout training. The model architecture and optimization settings are summarized in Tab. 10 and Tab. 10, respectively.

Table 9: Model configuration for joint training.

<table><tbody><tr><td>   Module   </td><td>   Configuration   </td><td>   Setting   </td></tr><tr><td rowspan="5">   Encoder   </td><td>   Architecture   </td><td>   ViT-L   </td></tr><tr><td>   Depth   </td><td>   24   </td></tr><tr><td>   Hidden dim.   </td><td>   1024   </td></tr><tr><td>   Attention heads   </td><td>   16   </td></tr><tr><td>   Parameters   </td><td>   304M   </td></tr><tr><td rowspan="5">   Future Predictor   </td><td>   Depth   </td><td>   12   </td></tr><tr><td>   Hidden dim.   </td><td>   768   </td></tr><tr><td>   Attention heads   </td><td>   12   </td></tr><tr><td>   MLP ratio   </td><td>   4   </td></tr><tr><td>   Parameters   </td><td>   153M   </td></tr><tr><td rowspan="4">   Action DiT   </td><td>   Depth   </td><td>   28   </td></tr><tr><td>   Hidden dim.   </td><td>   512   </td></tr><tr><td>   Attention heads   </td><td>   16   </td></tr><tr><td>   Parameters   </td><td>   103M   </td></tr><tr><td rowspan="2">   Target Encoder   </td><td>   Update   </td><td>   EMA   </td></tr><tr><td>   Decay   </td><td>   0.9999   </td></tr></tbody></table>

Table 10: Optimization configuration for joint training.

| Configuration | Setting |
| --- | --- |
| Observed frames | 4 |
| Predicted frames | 4 |
| Resolution | $256\times 512$ |
| Optimizer | AdamW |
| Learning rate | $3\times 10^{-5}$ |
| $\beta_{1},\beta_{2}$ | $0.9,\,0.95$ |
| Weight decay | $1\times 10^{-5}$ |
| LR schedule | Constant w/ warmup |
| Warmup steps | 1,000 |
| Gradient clipping | 1.0 |
| Planning loss | Flow-matching MSE |
| Prediction loss | L1 |
| Prediction weight | 0.1 |
| Diffusion timesteps | 1,000 |
| Inference steps | 5 |
| Precision | bfloat16 |

### A.2 More Qualitative Results

We provide additional qualitative comparisons between the human trajectory, Drive-JEPA [^39], and ReDrive in Fig. 6. The examples cover diverse road geometries and traffic interactions, with both front-camera and bird’s-eye-view (BEV) visualizations. These results further illustrate the planning behavior of ReDrive across different driving scenarios.

### A.3 Additional Future Predictor Visualizations

We provide additional visualizations of the Future Predictor across diverse driving scenes. We keep the history representation fixed and vary the lateral trajectory condition to examine the corresponding changes in predicted future representations. As shown in Fig. 7, the representations consistently vary with the trajectory condition across different scenes. The pairwise similarity matrices also exhibit a clear decay as the difference between action conditions increases. These results further show that the Future Predictor captures action-dependent future scene evolution across diverse driving scenarios.

![[action_sweep_grid.png|Refer to caption]]

Figure 7: Additional Future Predictor visualizations. Predicted future representations under different lateral trajectory conditions across diverse driving scenes. The representations exhibit consistent action-dependent variations across different scenarios.

[^1]: M. Assran, A. Bardes, D. Fan, Q. Garrido, R. Howes, M. Muckley, A. Rizvi, C. Roberts, K. Sinha, A. Zholus, et al. (2025) V-jepa 2: self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985. Cited by: §A.1.3, §1, §1, §2.2, §3.2, §4.1.

[^2]: A. Bardes, Q. Garrido, J. Ponce, X. Chen, M. Rabbat, Y. LeCun, M. Assran, and N. Ballas (2024) Revisiting feature prediction for learning visual representations from video. arXiv preprint arXiv:2404.08471. Cited by: §1, §2.2, §3.2.

[^3]: H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom (2020) Nuscenes: a multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 11621–11631. Cited by: §A.1.1, §A.1.3, §4.1.

[^4]: H. Caesar, J. Kabzan, K. S. Tan, W. K. Fong, E. Wolff, A. Lang, L. Fletcher, O. Beijbom, and S. Omari (2021) Nuplan: a closed-loop ml-based planning benchmark for autonomous vehicles. arXiv preprint arXiv:2106.11810. Cited by: §A.1.1, §A.1.2, §A.1.3, §4.1.

[^5]: W. Cao, M. Hallgarten, T. Li, D. Dauner, X. Gu, C. Wang, Y. Miron, M. Aiello, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, A. Geiger, and K. Chitta (2025) Pseudo-simulation for autonomous driving. In 9th Annual Conference on Robot Learning, Cited by: §1.

[^6]: K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger (2022) Transfuser: imitation with transformer-based sensor fusion for autonomous driving. IEEE transactions on pattern analysis and machine intelligence 45 (11), pp. 12878–12895. Cited by: Table 1, Table 3.

[^7]: D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, et al. (2024) Navsim: data-driven non-reactive autonomous vehicle simulation and benchmarking. Advances in Neural Information Processing Systems 37, pp. 28706–28719. Cited by: §1, §4.1, §4.1.

[^8]: H. Gao, S. Chen, B. Jiang, B. Liao, Y. Shi, X. Guo, Y. Pu, H. Yin, X. Li, X. Zhang, Y. Zhang, W. Liu, Q. Zhang, and X. Wang (2025) RAD: training an end-to-end driving policy via large-scale 3DGS-based reinforcement learning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, Cited by: §2.1.

[^9]: H. Gao, S. Chen, Y. Zhu, Y. Song, W. Liu, Q. Zhang, and X. Wang (2026) RAD-2: scaling reinforcement learning in a generator-discriminator framework. arXiv preprint arXiv:2604.15308. Cited by: §2.1.

[^10]: K. He, X. Chen, S. Xie, Y. Li, P. Dollár, and R. Girshick (2022) Masked autoencoders are scalable vision learners. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 15979–15988. Cited by: §4.3.

[^11]: Y. Hong, X. Zhou, Y. Li, X. Zhou, L. Liu, Y. Luo, S. Xu, L. Yang, and Z. Song (2026) DriveFuture: future-aware latent world models for autonomous driving. arXiv preprint arXiv:2605.09701. Cited by: §2.3, §4.2, Table 2.

[^12]: S. Hu, L. Chen, P. Wu, H. Li, J. Yan, and D. Tao (2022) St-p3: end-to-end vision-based autonomous driving via spatial-temporal feature learning. In European Conference on Computer Vision, pp. 533–549. Cited by: §2.1.

[^13]: Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, et al. (2023) Planning-oriented autonomous driving. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 17853–17862. Cited by: §1, §1, §2.1, §2.2, Table 1.

[^14]: M. Huang, Y. Xiang, Z. Liang, J. Huang, J. Wang, Z. Xu, F. Tan, H. Zhou, M. Yang, and G. Che (2026) Coworld-vla: thinking in a multi-expert world model for autonomous driving. arXiv preprint arXiv:2605.10426. Cited by: §4.2, Table 2.

[^15]: B. Jiang, S. Chen, H. Gao, B. Liao, Q. Zhang, W. Liu, and X. Wang (2026) VADv2: end-to-end autonomous driving via probabilistic planning. In The Fourteenth International Conference on Learning Representations, Cited by: Table 1.

[^16]: B. Jiang, S. Chen, B. Liao, X. Zhang, W. Yin, Q. Zhang, C. Huang, W. Liu, and X. Wang (2024) Senna: bridging large vision-language models and end-to-end autonomous driving. arXiv preprint arXiv:2410.22313. Cited by: §2.1.

[^17]: B. Jiang, S. Chen, Q. Xu, B. Liao, J. Chen, H. Zhou, Q. Zhang, W. Liu, C. Huang, and X. Wang (2023) Vad: vectorized scene representation for efficient autonomous driving. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 8306–8316. Cited by: §1, §1, §2.1, §2.2.

[^18]: B. Jiang, S. Chen, Q. Zhang, W. Liu, and X. Wang (2025) Alphadrive: unleashing the power of vlms in autonomous driving via reinforcement learning and reasoning. arXiv preprint arXiv:2503.07608. Cited by: §2.1.

[^19]: E. Kirby, A. Boulch, Y. Xu, Y. Yin, G. Puy, É. Zablocki, A. Bursuc, S. Gidaris, R. Marlet, F. Bartoccioni, A. Cao, N. Samet, T. VU, and M. Cord (2026) Driving on registers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 32058–32069. Cited by: §2.2.

[^20]: J. Li, Z. Liu, D. Hu, J. Wu, Z. Ma, W. Wu, C. Han, Z. Hao, Z. Liu, K. Zhan, et al. (2026) Metis: a generalizable and efficient world-action model for autonomous driving and urban navigation. arXiv preprint arXiv:2606.15869. Cited by: §1, §4.2, Table 3.

[^21]: J. Li, J. Wu, D. Hu, X. Huang, B. Sun, Z. Hao, X. Lang, X. Zhu, and L. Zhang (2026) Sgdrive: scene-to-goal hierarchical world cognition for autonomous driving. arXiv preprint arXiv:2601.05640. Cited by: Table 3.

[^22]: K. Li, Z. Li, S. Lan, Y. Xie, Z. Zhang, J. Liu, Z. Wu, Z. Yu, and J. M. Alvarez (2025) Hydra-mdp++: advancing end-to-end driving via expert-guided hydra-distillation. arXiv preprint arXiv:2503.12820. Cited by: Table 1.

[^23]: Y. Li, L. Fan, J. He, Y. Wang, Y. Chen, Z. Zhang, and T. Tan (2025) Enhancing end-to-end autonomous driving with latent world model. In International Conference on Learning Representations, Vol. 2025, pp. 42942–42959. Cited by: Table 1.

[^24]: Y. Li, Y. Wang, Y. Liu, J. He, L. Fan, and Z. Zhang (2025) End-to-end driving with online trajectory evaluation via bev world model. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 27137–27146. Cited by: §2.3.

[^25]: Y. Li, K. Xiong, X. Guo, F. Li, S. Yan, G. Xu, L. Zhou, L. Chen, H. Sun, B. Wang, K. Ma, et al. (2026) Recogdrive: a reinforced cognitive framework for end-to-end autonomous driving. In International Conference on Learning Representations, Vol. 2026, pp. 157518–157556. Cited by: §2.1, Table 3.

[^26]: Z. Li, K. Li, S. Wang, S. Lan, Z. Yu, Y. Ji, Z. Li, Z. Zhu, J. Kautz, Z. Wu, et al. (2024) Hydra-mdp: end-to-end multimodal planning with multi-target hydra-distillation. arXiv preprint arXiv:2406.06978. Cited by: §4.2, Table 1.

[^27]: Z. Li, W. Yao, Z. Wang, X. Sun, J. Chen, N. Chang, M. Shen, Z. Wu, S. Lan, and J. M. Alvarez (2025) Generalized trajectory scoring for end-to-end multimodal planning. arXiv preprint arXiv:2506.06664. Cited by: Table 3.

[^28]: Z. Li, W. Wang, H. Li, E. Xie, C. Sima, T. Lu, Q. Yu, and J. Dai (2024) Bevformer: learning bird’s-eye-view representation from lidar-camera via spatiotemporal transformers. IEEE Transactions on Pattern Analysis and Machine Intelligence 47 (3), pp. 2020–2036. Cited by: §1, §1, §2.2.

[^29]: B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang, et al. (2025) Diffusiondrive: truncated diffusion model for end-to-end autonomous driving. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 12037–12047. Cited by: §2.1, §4.2, §4.2, Table 1, Table 2, Table 3.

[^30]: L. Liu, C. Jia, G. Yu, Z. Song, J. Li, F. Jia, P. Wu, X. Hao, and Y. Luo (2026) Guideflow: constraint-guided flow matching for planning in end-to-end autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3719–3728. Cited by: Table 3.

[^31]: H. Lu, L. Yao, C. He, H. Wang, X. Gu, X. Li, W. Liao, T. He, and P. Peng (2026) The dawn of world-action interactive models. arXiv preprint arXiv:2605.11550. Cited by: §1, Table 1.

[^32]: M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al. (2023) Dinov2: learning robust visual features without supervision. arXiv preprint arXiv:2304.07193. Cited by: §1, §2.2, §4.3.

[^33]: W. Peebles and S. Xie (2023) Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4195–4205. Cited by: §3.3.2, §3.3.2.

[^34]: S. Shang, Y. Chen, Y. Wang, Y. Li, and Z. ZHANG (2026) Drivedpo: policy learning via safety dpo for end-to-end autonomous driving. Advances in Neural Information Processing Systems 38, pp. 81565–81585. Cited by: §4.2, Table 1.

[^35]: O. Siméoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab, C. Jose, V. Khalidov, M. Szafraniec, S. Yi, M. Ramamonjisoa, et al. (2025) Dinov3. arXiv preprint arXiv:2508.10104. Cited by: §1.

[^36]: Y. Song, S. Chen, H. Gao, Y. Zhu, W. Yue, J. Zou, B. Jiang, Z. Lu, Y. Wang, Q. Zhang, and X. Wang (2026) Senna-2: aligning vlm and end-to-end driving policy for consistent decision making and planning. arXiv preprint arXiv:2603.11219. Cited by: §2.1.

[^37]: W. Sun, X. Lin, K. Chen, Z. Pei, X. Li, Y. Shi, and S. Zheng (2026) Sparsedrivev2: scoring is all you need for end-to-end autonomous driving. In European Conference on Computer Vision, pp. 446–463. Cited by: §4.2, Table 2.

[^38]: L. Wang, Y. Zheng, Q. Chen, S. Li, Y. Zhang, Z. Xing, Q. Zhang, X. Li, D. Qian, P. Yang, et al. (2026) Latent-wam: latent world action modeling for end-to-end autonomous driving. arXiv preprint arXiv:2603.24581. Cited by: §1, §2.2, §4.2, Table 2.

[^39]: L. Wang, Z. Yang, C. Bai, G. Zhang, X. Liu, X. Zheng, X. Long, C. Lu, and C. Lu (2026) Drive-jepa: video jepa meets multimodal trajectory distillation for end-to-end driving. arXiv preprint arXiv:2601.22032. Cited by: §A.2, §1, §1, §2.2, Figure 3, Figure 3, §4.4, Table 1.

[^40]: T. Xia, Y. Li, L. Zhou, J. Yao, K. Xiong, H. Sun, B. Wang, K. Ma, G. Chen, H. Ye, et al. (2026) Drivelaw: unifying planning and video generation in a latent driving world. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 39701–39712. Cited by: §1, §1, §2.3, Table 1.

[^41]: T. Xia, L. Zhou, K. Xiong, J. Yao, Y. Zhu, Z. Zhu, B. Wang, G. Chen, H. Ye, W. Liu, et al. (2026) ReWorld: learning better representations for world action models. arXiv preprint arXiv:2606.27504. Cited by: §2.3, §4.2, Table 1.

[^42]: Y. Xing, Z. Ke, Z. Liu, Y. Jiang, W. Yu, and J. Wang (2026) CLEAR: cognition and latent evaluation for adaptive routing in end-to-end autonomous driving. arXiv preprint arXiv:2606.06219. Cited by: §1, §2.2.

[^43]: Y. Xing, Z. Liu, Z. Ke, W. Yu, and J. Wang (2026) DRIFT: drift and aggregation for motion planning. arXiv preprint arXiv:2607.14507. Cited by: §1, §2.2.

[^44]: Z. Xing, X. Zhang, Y. Hu, B. Jiang, T. He, Q. Zhang, X. Long, and W. Yin (2025) Goalflow: goal-driven flow matching for multimodal trajectories generation in end-to-end autonomous driving. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1602–1611. Cited by: §2.1, §4.2, Table 1.

[^45]: M. Xu, J. Cui, F. Cai, H. Shang, Z. Zhu, S. Luan, Y. Xu, N. Zhang, Y. Li, J. Cai, et al. (2025) Wam-diff: a masked diffusion vla framework with moe and online reinforcement learning for autonomous driving. arXiv preprint arXiv:2512.11872. Cited by: Table 2.

[^46]: J. Yang, Z. Chen, C. Huang, and J. Li (2026) Auto-jepa: a latent world model of continuous intent for end-to-end autonomous driving. arXiv preprint arXiv:2607.29031. Cited by: §2.2.

[^47]: P. Yang, Y. Zheng, D. Qian, Z. Xing, Q. Zhang, L. Wang, Y. Zhang, S. Guo, Z. Xia, Q. Chen, et al. (2026) Dreamerad: efficient reinforcement learning via latent world model for autonomous driving. arXiv preprint arXiv:2603.24587. Cited by: §4.2, Table 2.

[^48]: W. Yao, Z. Li, S. Lan, Z. Wang, X. Sun, J. M. Alvarez, and Z. Wu (2026) Drivesuprim: towards precise trajectory selection for end-to-end planning. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40, pp. 11910–11918. Cited by: Table 1.

[^49]: B. Zhang, N. Song, X. Zhu, J. Deng, L. Zhang, et al. (2026) Future-aware end-to-end driving: bidirectional modeling of trajectory planning and scene evolution. Advances in Neural Information Processing Systems 38, pp. 10204–10229. Cited by: §2.3.

[^50]: K. Zhang, Z. Tang, X. Hu, X. Pan, X. Guo, Y. Liu, J. Huang, L. Yuan, Q. Zhang, X. Long, et al. (2025) Epona: autoregressive diffusion world model for autonomous driving. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 27220–27230. Cited by: §1, §2.3, Table 1.

[^51]: Y. Zheng, P. Yang, Z. Xing, Q. Zhang, Y. Zheng, Y. Gao, P. Li, T. Zhang, Z. Xia, P. Jia, et al. (2025) World4drive: end-to-end autonomous driving via intention-aware physical latent world model. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 28632–28642. Cited by: §2.3, Table 1.

[^52]: Z. Zheng, S. Chen, H. Yin, X. Zhang, J. Zou, X. Wang, Q. Zhang, and L. Zhang (2026) ResAD: normalized residual trajectory modeling for end-to-end autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3729–3739. Cited by: §2.1.

[^53]: R. Zhong, B. Ma, X. Chen, L. Zhang, M. Feng, Y. Wang, P. Liu, and J. Ma (2026) DA-wam: decision-aligned future latents for driving world models. arXiv preprint arXiv:2608.19085. Cited by: §1.

[^54]: J. Zou, S. Chen, B. Liao, Z. Zheng, Y. Song, L. Zhang, Q. Zhang, W. Liu, and X. Wang (2025) Diffusiondrivev2: reinforcement learning-constrained truncated diffusion modeling in end-to-end autonomous driving. arXiv preprint arXiv:2512.07745. Cited by: §2.1, Table 2.
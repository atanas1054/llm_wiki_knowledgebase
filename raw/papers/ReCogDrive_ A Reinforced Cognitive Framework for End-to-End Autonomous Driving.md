---
title: "ReCogDrive: A Reinforced Cognitive Framework for End-to-End Autonomous Driving"
source: "https://arxiv.org/html/2506.08052v1"
author:
published:
created: 2026-04-05
description:
tags:
  - "clippings"
---
Yongkang Li <sup>1,2 *</sup>, Kaixin Xiong <sup>2 *</sup>, Xiangyu Guo <sup>1,2</sup>, Fang Li <sup>2</sup>, Sixu Yan <sup>1</sup>, Gangwei Xu <sup>1,2</sup>,  
Lijun Zhou <sup>2</sup>, Long Chen <sup>2</sup>, Haiyang Sun <sup>2 <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\dagger"><semantics><mo>†</mo> <annotation-xml><ci>†</ci></annotation-xml> <annotation>\dagger</annotation> <annotation>†</annotation></semantics></math></sup>, Bing Wang <sup>2</sup>, Guang Chen <sup>2</sup>,  
Hangjun Ye <sup>2</sup>, Wenyu Liu <sup>1</sup>, Xinggang Wang <sup>1</sup> <sup>✉</sup>  
  
<sup>1</sup> Huazhong University of Science and Technology <sup>2</sup> Xiaomi EV  
{liyk, xgwang}@hust.edu.cn, {xiongkaixin, sunhaiyang1}@xiaomi.com  
[https://xiaomi-research.github.io/recogdrive](https://xiaomi-research.github.io/recogdrive)

###### Abstract

Although end-to-end autonomous driving has made remarkable progress, its performance degrades significantly in rare and long-tail scenarios. Recent approaches attempt to address this challenge by leveraging the rich world knowledge of Vision-Language Models (VLMs), but these methods suffer from several limitations: (1) a significant domain gap between the pre-training data of VLMs and real-world driving data, (2) a dimensionality mismatch between the discrete language space and the continuous action space, and (3) imitation learning tends to capture the average behavior present in the dataset, which may be suboptimal even dangerous. In this paper, we propose ReCogDrive, an autonomous driving system that integrates VLMs with diffusion planner, which adopts a three-stage paradigm for training. In the first stage, we use a large-scale driving question-answering datasets to train the VLMs, mitigating the domain discrepancy between generic content and real-world driving scenarios. In the second stage, we employ a diffusion-based planner to perform imitation learning, mapping representations from the latent language space to continuous driving actions. Finally, we fine-tune the diffusion planner using reinforcement learning with NAVSIM non-reactive simulator, enabling model to generate safer, more human-like driving trajectories. We evaluate our approach on the planning-oriented NAVSIM benchmark, achieving a PDMS of 89.6 and setting a new state-of-the-art that surpasses the previous vision-only SOTA by 5.6 PDMS.

<sup>†</sup>

## 1 Introduction

Autonomous driving, which aims to predict a smooth, comfortable, and collision-free trajectory for a vehicle, has seen significant advancements. Recent end-to-end autonomous driving systems [^1] [^2] [^3] [^4] [^5] unify the perception [^6] [^7] [^8], prediction [^9] [^10], and planning [^11] [^12] modules into a single pipeline for joint optimization, demonstrating impressive performance under open-loop evaluation. However, these systems often fail to generalize to long-tail scenarios, where data is limited and the driving conditions deviate significantly from those encountered during training.

Recent research [^13] [^14] [^15] addresses the long-tail challenge by introducing VLMs, which are pre-trained on large-scale internet datasets and exhibit strong generalization abilities along with rich world knowledge. Of note, several studies [^13] [^15] [^16] [^17] [^18] [^19] [^20] [^21] [^22] [^23], such as DriveVLM [^13] and EMMA [^15], leverage pre-trained VLMs to predict future trajectories in textual form, thereby reformulating the motion planning task as a language modeling problem. These methods exploit the autoregressive nature of VLMs to predict future trajectories by generating the next token step by step. Specifically, VLMs applications in autonomous driving can be categorized into two main approaches: (1) dual-systems approaches [^14] [^13], which generate low-frequency trajectories or high-level commands via VLMs to guide an end-to-end driving system; and (2) single-systems approaches [^15] [^16] [^17] [^18] [^19] [^20] [^21] [^22] [^23], where standalone VLMs directly predict future trajectories and can be optimized in an end-to-end manner.

However, directly applying VLMs pre-trained on general-domain data to autonomous driving introduces several limitations: (1) VLMs are typically pre-trained on internet image–text datasets, which lack coverage of driving-specific scenarios and knowledge. (2) There exists a substantial gap between the latent language space and the continuous action space required for motion planning. Furthermore, the autoregressive decoding process can occasionally lead to output collapse, producing incoherent or invalid trajectories. (3) Most existing methods rely heavily on behavior cloning, which tends to memorize the training data and optimize to an average behavior manner.

![[x1.png|Refer to caption]]

Figure 1: Overview of ReCogDrive. We present ReCogDrive, an 8B-parameter driving system for autonomous driving, which is jointly trained on a high-quality 3.1M driving QA dataset and diverse instruction data. ReCogDrive is capable of performing tasks spanning from low-level scene perception, region recognition, and motion prediction to high-level driving planning and decision making.

In this paper, we propose ReCogDrive, a novel end-to-end autonomous driving system that integrates cognition from VLMs with reinforcement learning enhanced diffusion planner. Unlike prior single-system approaches, our method combines the strong generalization ability of VLMs with the generation capacity of diffusion models to predict smooth and continuous trajectories. We first collect and organize 3.1 million high-quality driving question–answer pairs, sourced from both open datasets and an automatic annotation pipeline. We use this dataset to pre-train the VLMs in the driving domain, facilitating its adaptation to real-world driving scenarios. Subsequently, we introduce a diffusion planner that bridges the gap between the latent language space and the continuous action space, which maps high-level representations into low-level driving actions. This design effectively preserves the VLMs ’ generalization capabilities. To further improve trajectory quality, we fine-tune the diffusion model via simulator-assisted reinforcement learning. Unlike imitation learning, which merely replicates expert demonstrations, reinforcement learning enables the model to actively explore driving behaviors and generate safer, more stable, and more comfortable trajectories with the assistance of the NAVSIM simulator.

The main contributions of this work are as follows:

- We propose ReCogDrive, a novel end-to-end autonomous driving system equipped driving cognition from three aspects: (1) inherent world cognition in VLMs, (2) driving domain cognition from constructed high quality driving data, and (3) generalized cognition from multi-trajectory exploration via reinforcement learning.
- We present a three-stage training framework. First, the VLMs are fine-tuned on a large-scale driving question–answering dataset to adapt to driving scenarios. Next, a diffusion model is trained via behavior cloning to generate high-fidelity trajectories. Finally, simulator-assisted reinforcement learning is proposed to generate safer and more stable trajectories.
- We conduct extensive experiments on NAVSIM benchmark. Our method achieves a state-of-the-art PDMS score of 89.6, highlighting its effectiveness and real-world viability.

## 2 Preliminaries

#### Problem Definition.

Autonomous driving task aims to predict smooth and collision-free trajectory in future seconds, given the ego status $S_{\mathrm{ego}}$ (e.g., ego speed and ego acceleration), sensor input $I_{\mathrm{cam}}$, and navigation information $L_{\mathrm{nav}}$. Conventional end-to-end driving algorithm $\Phi$ formulate this as

$$
\mathbf{V}_{\mathrm{traj}}=\Phi\bigl{(}I_{\mathrm{cam}},\,L_{\mathrm{nav}},\,S%
_{\mathrm{ego}}\bigr{)},
$$

where $\mathbf{V}_{\mathrm{traj}}\in\mathbb{R}^{T\times 3}$ is the sequence of future waypoints and headings. While methods such as [^1] [^2] [^3] [^12] have shown strong effectiveness, their black-box nature impedes interpretability and they often fail to generalize to rare corner cases in real-world driving.

Recent works [^15] [^16] [^24] utilize the rich world knowledge and causal reasoning capabilities of Vision–Language Models for autonomous driving. VLMs output trajectories in textual form and generate explicit reasoning processes:

$$
T_{\mathrm{traj}},\,T_{\mathrm{reason}}=\mathrm{VLM}\bigl{(}I_{\mathrm{cam}},%
\,L_{\mathrm{nav}},\,S_{\mathrm{ego}}\bigr{)}.
$$

However, we observe an inherent mismatch between the language-formatted trajectory space and the continuous action space, and the autoregressive decoding process can suffer from output collapse, leading to erroneous trajectories.

#### Diffusion Policy.

Denoising Diffusion Probabilistic Models (DDPMs) [^25] [^26] learn a generative model by reversing a fixed, Markovian noising process that gradually corrupts data with Gaussian noise. Given a clean trajectory $\mathbf{x}_{0}$, the forward process defines

$$
q(\mathbf{x}_{t}\mid\mathbf{x}_{t-1})=\mathcal{N}\bigl{(}\mathbf{x}_{t};\,%
\sqrt{1-\sigma_{t}^{2}}\,\mathbf{x}_{t-1},\,\sigma_{t}^{2}\mathbf{I}\bigr{)},%
\quad t=1,\dots,T,
$$

where $\sigma_{t}$ is the noise standard deviation at step $t$. At inference, trajectories are generated by initializing $\mathbf{x}_{T}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ and iteratively denoising:

$$
\mathbf{x}_{t-1}=\frac{1}{\sqrt{1-\sigma_{t}^{2}}}\Bigl{(}\mathbf{x}_{t}-%
\sigma_{t}^{2}\,\epsilon_{\theta}(\mathbf{x}_{t},t)\Bigr{)}+\sigma_{t}\,%
\mathbf{z},\quad\mathbf{z}\sim\mathcal{N}(0,\mathbf{I}).
$$

Denoting trajectory waypoint as $\mathbf{x}_{0}$, this framework naturally extends to trajectory-level policy generation, where the denoising network $\epsilon_{\theta}$ learns to refine noisy motion plans into smooth trajectories.

## 3 Methodology

In this section, we present the architecture of ReCogDrive and proposed three-stage training paradigm. First, we assemble a driving dataset of 3.1 million high-quality QA pairs to pre-train Vision–Language Models, injecting driving-specific cognition into the model. Next, we integrate the pre-trained VLMs with a diffusion-based trajectory planner to achieve stable language-to-action mapping, translating latent language space into continuous action space. Finally, we introduce simulation-assisted reinforcement learning to integrate generalized driving cognition, acquired through multi-trajectory exploration, into the diffusion planner.

### 3.1 Driving-Specific Vision–Language Pre-training

#### High-Quality Driving Data Construction.

To adapt general VLMs to autonomous driving, we assembled a diverse collection of open-source driving QA datasets, including DriveLM [^27], LingoQA [^28], DRAMA [^29], and nine other driving datasets. To ensure consistency across datasets, we normalize all bounding-box annotations to the InternVL3 [^30] format and introduce view-specific tokens to distinguish different camera perspectives. Some datasets (e.g., DRAMA [^29], nuInstruct [^31]) rely on templated answers, leading to limited language variation. To improve dataset quality, we employ Qwen2.5-VL [^32] to re-annotate answers, score each QA pair according to established standards, and remove low-scoring examples, resulting in 2.3M high-quality QA pairs. To boost generalization in diverse driving scenarios, we build an automatic annotation pipeline that leverages VLMs and human annotations to generate high-quality QA for tasks such as scene description, key object description, planning explanation and more. In addition, we incorporate 665K LLaVA instruct-tuning data to preserve the VLM’s instruction-following capabilities.

#### Driving-Adaptive Vision-Language Model.

We adopt InternVL3-8B [^30] [^33] as our base model, which employs a native multi-modal pre-training paradigm and has demonstrated impressive performance across various benchmarks. Each image is partitioned into $448\times 448$ patches plus a $448\times 448$ thumbnail and encoded by InternViT. A pixel shuffle operation and MLP projector compress image features into 256 visual tokens per patch, which are then concatenated with text tokens and fed into the language head. We fine-tune this model on a combination of 3.1 million high-quality driving QA pairs and instruct-tuning datasets. After training, the model is capable of producing textual trajectory $T_{\mathrm{traj}}$, reasoning trace $T_{\mathrm{reason}}$, and scene description $T_{\mathrm{desc}}$:

$$
T_{\mathrm{traj}},\;T_{\mathrm{reason}},\;T_{\mathrm{desc}}=\mathrm{VLM}\bigl{%
(}S_{\mathrm{ego}},\,L_{\mathrm{nav}},\,I_{\mathrm{cam}}\bigr{)},
$$

where $T_{\mathrm{reason}}$ and $T_{\mathrm{desc}}$ offer interpretability for the driving decision process.

![[x2.png|Refer to caption]]

Figure 2: Model Architecture and Training Pipeline. ReCogDrive comprises a Vision-Language Model and a diffusion planner. Given a front-view image, navigation command, ego state, and task instruction, the VLM extracts high-level representations for the planner, which denoises from noise to generate future trajectories. We first pre-train the VLM for driving adaptation, then freeze it while training the planner via imitation and reinforcement learning.

### 3.2 Diffusion-based Trajectory Planner

While auto-regressive paradigms offer a straightforward method for directly generating trajectories with VLMs, this approach encounters fundamental limitations due to the inherent discrepancy between structured action spaces and language representations. The language space inherently suffers from precision limitations in numerical representation due to floating-point discretization constraints. More importantly, VLMs exhibit sporadic hallucination artifacts, which undermine their reliability in safety-critical driving scenarios. Inspired by [^34] [^35] [^36] [^37], we employ a diffusion-based planner as action decoder to decode smooth trajectories from the high-dimensional language space.

Given a noisy trajectory sample $\mathbf{x}_{t}\in\mathbb{R}^{N\times 3}$ drawn from Gaussian distribution, where $N$ denotes the number of temporal waypoints and the column dimension corresponds to planar coordinates $(x,y)\in\mathbb{R}^{2}$ with heading angle $\theta\in\mathbb{S}^{1}$, we initially embed this low-dimensional representation into high-dimensional feature space through an action encoder $E_{act}:\mathbb{R}^{N\times 3}\to\mathbb{R}^{N\times D}$. To incorporate semantic priors, we extract the final-layer hidden states ${F_{h}}\in\mathbb{R}^{L\times D}$ from the VLM, where $L$ denotes the sequence length. Through average pooling operation, we obtain a condensed semantic embedding $\bar{F_{h}}\in\mathbb{R}^{1\times D}$ that preserves the global contextual information. To ensure the smoothness of trajectory, we encode history trajectory with $E_{his}$ into high-dimensional embedding $\mathbf{e}_{\mathrm{hist}}\in\mathbb{R}^{1\times D}$. Then we concatenate these embeddings along the channel dimension as input of DiT blocks $z_{t}$:

$$
z_{t}=\mathrm{concat}\left(E_{\mathrm{act}}(\mathbf{x}_{t}),\,E_{\mathrm{his}}%
(\mathbf{x}_{\mathrm{hist}}),\,\bar{F}_{h}\right).
$$

The DiT block architecture comprises two core components: a self-attention layer and a cross-attention layer. The self-attention mechanism enables pairwise interaction between waypoint queries while dynamically conditioning their representations through adaptive layer normalization (AdaLayerNorm [^38]), which explicitly incorporates ego-vehicle status and temporal information. Subsequently, the cross-attention layer establishes a contextual bridge between the waypoint queries and the latent features $F_{h}$ generated by the final layer of the VLM. This hierarchical fusion mechanism facilitates the systematic integration of high-level semantic understanding with trajectory optimization constraints. Finally, an action decoder produces the denoised, continuous trajectory:

$$
\mathbf{x}_{t-1}=D_{\mathrm{act}}\Bigl{(}\mathrm{DiT}_{\theta}(z_{t};F_{h};S_{%
ego};t)\Bigr{)}.
$$

We adopt DDPM [^25] scheduler as diffusion policy. This diffusion-based mechanism maps discrete semantic tokens into the action space, yielding smooth continuous trajectories. We minimize the mean squared error between the predicted noise ($\epsilon_{\pi}$) from the diffusion policy and the GT noise ($\epsilon$). The loss function is defined as follows:

$$
\mathcal{L}_{\text{dif}}=\mathbb{E}_{z_{t},c}||\epsilon-\epsilon_{\pi}(z_{t},c%
)||^{2},
$$

where $\epsilon\sim\mathcal{N}(0,\mathbf{I})$ and $c$ denotes the condition. Additionally, classifier-free guidance is not used in order to ensure stable trajectory.

### 3.3 Simulator-assisted Reinforcement Learning

Relying solely on imitation learning exhibits fundamental limitations [^39] [^40] [^34], since expert demonstrations may be multi-modal, leading to conflictual optimization results. As shown in Fig. 3(a), when trained in this rare intersection turn scenario with multiple expert trajectories, the model resorts to learning an average trajectory to achieve global optimality, which can lead to incorrect or unsafe maneuvers. Consequently, even though the diffusion planner trained through imitation learning closely matches expert trajectories in terms of displacement, it still frequently produces collisions or drives outside the drivable area.

![[x3.png|Refer to caption]]

Figure 3: Comparison of Training Paradigms. (a) Imitation Learning: the diffusion planner is trained offline to mimic ground truth trajectories using L1/L2 losses, but tends to learn averaged, suboptimal paths. (b) Reinforcement Learning: multiple trajectories are sampled and evaluated in the NAVSIM simulator, scored on collision avoidance, drivable area compliance and other metrics, and advantages are computed via group computation to update the diffusion planner.

A more intuitive approach is to let the model explore by driving itself in a simulated environment, mimicking real-world learning. However, constructing a fully interactive closed-loop simulator is highly challenging, so we use the non-reactive simulator of NAVSIM to evaluate collisions, comfort, and other metrics for reinforcement learning, as shown in Fig. 3(b). Following [^34], the diffusion policy $\pi_{\theta}$ can be viewed as an internal Markov decision process that starts from Gaussian noise and gradually denoises to produce an action sequence. Concretely, we sample $G$ trajectories and obtain their diffusion chains. The diffusion chain of a single trajectory is represented as

$$
\mathbf{x}=\bigl{(}x_{T},\,x_{T-1},\,\dots,\,x_{0}\bigr{)},
$$

where $T$ is the total number of denoising steps. For this chain:

$$
x_{T}\sim\mathcal{N}(0,\mathbf{I}),\qquad x_{t-1}\sim\pi_{\theta}\bigl{(}x_{t-%
1}\mid x_{t}\bigr{)},\quad t=T,T-1,\dots,1.
$$

These trajectories are simulated in the NAVSIM simulator, which evaluates each rollout for collisions, drivable-area compliance, and driving comfort, and returns a Predictive Driver Model Score (PDMS) that serves as the reward $r_{i}$. We then compute the group-standardized advantage:

$$
\hat{A}_{i}\;=\;\frac{r_{i}-\mathrm{mean}\bigl{(}r_{1..G}\bigr{)}}{\sqrt{%
\mathrm{var}\bigl{(}r_{1..G}\bigr{)}}},\quad i=1,\dots,G.
$$

Each conditional step in the diffusion chain is a Gaussian policy:

$$
\pi_{\theta}\bigl{(}x_{t-1}\mid x_{t}\bigr{)}=\mathcal{N}\!\Bigl{(}x_{t-1};\,%
\mu_{\theta}(x_{t},t),\,\sigma_{t}^{2}I\Bigr{)},
$$

where $\mu_{\theta}(x_{t},t)$ is the model-predicted mean and $\sigma_{t}^{2}I$ the (fixed) covariance.

Thus, the probability density of the full chain $\mathbf{x}_{0:T}$ under $\pi_{\theta}$ is

$$
\log\pi_{\theta}\bigl{(}\mathbf{x}_{0:T}\bigr{)}=\sum_{t=1}^{T}\log\pi_{\theta%
}\bigl{(}x_{t-1}\mid x_{t}\bigr{)}.\qquad
$$

Finally, we compute the policy loss following [^41] [^42] [^43] while concurrently incorporating a behavior cloning loss to prevent collapse during exploration.

$$
L\;=\;\underbrace{-\frac{1}{G}\sum_{i=1}^{G}\frac{1}{T}\sum_{t=1}^{T}\gamma^{%
\,t-1}\,\log\pi_{\theta}\bigl{(}x_{t-1}^{(i)}\mid x_{t}^{(i)}\bigr{)}\,\hat{A}%
_{i}}_{L_{\mathrm{RL}}}\;-\;\underbrace{\lambda\;\frac{1}{G}\sum_{i=1}^{G}%
\frac{1}{T}\sum_{t=1}^{T}\log\pi_{\theta}\bigl{(}\tilde{x}_{t-1}^{(i)}\mid%
\tilde{x}_{t}^{(i)}\bigr{)}}_{L_{\mathrm{BC}}},
$$

where $\gamma$ is the discount coefficient mitigating instability in early denoising steps, $\lambda$ is the weight for the behavior cloning loss, and $\tilde{x}_{t-1},\tilde{x}_{t}$ are values sampled from the reference policy $\pi_{\mathrm{ref}}$. Through simulator-assisted reinforcement learning, the diffusion planner learns to predict safe and comfortable trajectories by exploration, which goes beyond mere imitation and inject cognition into our framework.

## 4 Experiments

### 4.1 Implementation Details

We choose InternVL3-8B [^30], comprising a 300M-parameter InternViT visual encoder [^33] and a 7B-parameter Qwen2.5 Large-Language-Model [^32] [^44], as our base model, as it achieves strong performance across multiple evaluation benchmarks. Images are processed through dynamic resolution preprocessing strategy. In the first stage, we conduct supervised fine-tuning (SFT) on the VLMs using the mixed 3.1M high-quality driving dataset for three epochs. In the second stage, with the VLM parameters frozen, we train the diffusion model via DDPM for 200 epochs. In the third stage, we further optimize the diffusion model using reinforcement learning for 10 epochs. Detailed hyperparameter settings are provided in the supplementary material.

### 4.2 Dataset and evaluation metrics

#### Dataset.

NAVSIM [^45] is a planning-oriented autonomous driving dataset built on OpenScene [^46], a redistribution of nuPlan [^47]. It provides eight 1920 $\times$ 1080 cameras and a fused LiDAR point cloud aggregated from five sensors across the current and three previous frames. The dataset is split into navtrain (1,192 training scenes) and navtest (136 evaluation scenes). We mix the 85,109 trajectory-based QA pairs from navtrain with 12 open-source driving QA datasets for training, including Talk2Car [^48], SUTD [^49], DRAMA [^29], NuScenes-QA [^50], DriveGPT4 [^51], LingoQA [^28], DriveLM [^27], MAPLM [^52], NuInstruct [^31], CODA-LMM [^53], OmniDrive [^19], and Senna [^14], covering tasks from perception and prediction to reasoning and planning. More details are provided in the supplementary material.

Table 1: Performance comparison on NAVSIM navtest using closed-loop metrics. <sup>†</sup> denotes models fine-tuned on the NAVSIM trajectory dataset.

<table><tbody><tr><td>Method</td><td>Image</td><td>Lidar</td><td>NC <math><semantics><mo>↑</mo> <annotation-xml><ci>↑</ci></annotation-xml> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation-xml><ci>↑</ci></annotation-xml> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation-xml><ci>↑</ci></annotation-xml> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>Comf. <math><semantics><mo>↑</mo> <annotation-xml><ci>↑</ci></annotation-xml> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation-xml><ci>↑</ci></annotation-xml> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>PDMS <math><semantics><mo>↑</mo> <annotation-xml><ci>↑</ci></annotation-xml> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td></tr><tr><td>Constant Velocity</td><td></td><td></td><td>68.0</td><td>57.8</td><td>50.0</td><td>100</td><td>19.4</td><td>20.6</td></tr><tr><td>Ego Status MLP</td><td></td><td></td><td>93.0</td><td>77.3</td><td>83.6</td><td>100</td><td>62.8</td><td>65.6</td></tr><tr><td>VADv2- <math><semantics><msub><mi>𝒱</mi> <mtext>8192</mtext></msub> <annotation-xml><apply><csymbol>subscript</csymbol> <ci>𝒱</ci> <ci><mtext>8192</mtext></ci></apply></annotation-xml> <annotation>\mathcal{V}_{\text{8192}}</annotation> <annotation>caligraphic_V start_POSTSUBSCRIPT 8192 end_POSTSUBSCRIPT</annotation></semantics></math> <sup><a href="#fn:3">3</a></sup></td><td>✓</td><td></td><td>97.2</td><td>89.1</td><td>91.6</td><td>100</td><td>76.0</td><td>80.9</td></tr><tr><td>DrivingGPT <sup><a href="#fn:54">54</a></sup></td><td>✓</td><td></td><td>98.9</td><td>90.7</td><td>94.9</td><td>95.6</td><td>79.7</td><td>82.4</td></tr><tr><td>Hydra-MDP- <math><semantics><msub><mi>𝒱</mi> <mtext>8192</mtext></msub> <annotation-xml><apply><csymbol>subscript</csymbol> <ci>𝒱</ci> <ci><mtext>8192</mtext></ci></apply></annotation-xml> <annotation>\mathcal{V}_{\text{8192}}</annotation> <annotation>caligraphic_V start_POSTSUBSCRIPT 8192 end_POSTSUBSCRIPT</annotation></semantics></math></td><td>✓</td><td>✓</td><td>97.9</td><td>91.7</td><td>92.9</td><td>100</td><td>77.6</td><td>83.0</td></tr><tr><td>UniAD <sup><a href="#fn:2">2</a></sup></td><td>✓</td><td></td><td>97.8</td><td>91.9</td><td>92.9</td><td>100</td><td>78.8</td><td>83.4</td></tr><tr><td>LTF <sup><a href="#fn:11">11</a></sup></td><td>✓</td><td></td><td>97.4</td><td>92.8</td><td>92.4</td><td>100</td><td>79.0</td><td>83.8</td></tr><tr><td>BevDrive <sup><a href="#fn:55">55</a></sup></td><td>✓</td><td>✓</td><td>97.7</td><td>92.5</td><td>92.9</td><td>100</td><td>78.7</td><td>83.8</td></tr><tr><td>TransFuser <sup><a href="#fn:11">11</a></sup></td><td>✓</td><td>✓</td><td>97.7</td><td>92.8</td><td>92.8</td><td>100</td><td>79.2</td><td>84.0</td></tr><tr><td>PARA-Drive <sup><a href="#fn:56">56</a></sup></td><td>✓</td><td></td><td>97.9</td><td>92.4</td><td>93.0</td><td>99.8</td><td>79.3</td><td>84.0</td></tr><tr><td>DRAMA <sup><a href="#fn:57">57</a></sup></td><td>✓</td><td>✓</td><td>98.0</td><td>93.1</td><td>94.8</td><td>100</td><td>80.1</td><td>85.5</td></tr><tr><td>Hydra-MDP- <math><semantics><msub><mi>𝒱</mi> <mtext>8192</mtext></msub> <annotation-xml><apply><csymbol>subscript</csymbol> <ci>𝒱</ci> <ci><mtext>8192</mtext></ci></apply></annotation-xml> <annotation>\mathcal{V}_{\text{8192}}</annotation> <annotation>caligraphic_V start_POSTSUBSCRIPT 8192 end_POSTSUBSCRIPT</annotation></semantics></math> -W-EP <sup><a href="#fn:58">58</a></sup></td><td>✓</td><td>✓</td><td>98.3</td><td>96.0</td><td>94.6</td><td>100</td><td>78.7</td><td>86.5</td></tr><tr><td>ARTEMIS <sup><a href="#fn:59">59</a></sup></td><td>✓</td><td>✓</td><td>98.3</td><td>95.1</td><td>94.3</td><td>100</td><td>81.4</td><td>87.0</td></tr><tr><td>DiffusionDrive <sup><a href="#fn:12">12</a></sup></td><td>✓</td><td>✓</td><td>98.2</td><td>96.2</td><td>94.7</td><td>100</td><td>82.2</td><td>88.1</td></tr><tr><td>WoTE <sup><a href="#fn:60">60</a></sup></td><td>✓</td><td>✓</td><td>98.5</td><td>96.8</td><td>94.9</td><td>99.9</td><td>81.9</td><td>88.3</td></tr><tr><td colspan="8">VLMs-based Methods</td><td></td></tr><tr><td>QwenVL2.5 <sup>†</sup> <sup><a href="#fn:32">32</a></sup></td><td>✓</td><td></td><td>97.8</td><td>92.1</td><td>92.8</td><td>100</td><td>78.3</td><td>83.3</td></tr><tr><td>InternVL3 <sup>†</sup> <sup><a href="#fn:30">30</a></sup></td><td>✓</td><td></td><td>97.0</td><td>92.4</td><td>91.8</td><td>100</td><td>78.9</td><td>83.3</td></tr><tr><td>ReCogDrive (Ours)</td><td>✓</td><td></td><td>98.2</td><td>97.8</td><td>95.2</td><td>99.8</td><td>83.5</td><td>89.6</td></tr></tbody></table>

Table 2: Ablation study on the proposed components of ReCogDrive. We evaluate the effect of driving pre-training, diffusion planner, and reinforcement learning on NAVSIM evaluation.

| ID | Trajectory training | Driving Pre-training | Diffusion Planner | Reinforce Learning | NC | DAC | TTC | Conf. | EP | PDMS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ✓ | ✗ | ✗ | ✗ | 97.0 | 92.4 | 91.8 | 100 | 78.9 | 83.3 |
| 2 | ✓ | ✓ | ✗ | ✗ | 98.2 | 94.5 | 94.5 | 100 | 80.4 | 86.2 |
| 3 | ✓ | ✓ | ✓ | ✗ | 98.3 | 95.1 | 94.3 | 100 | 81.1 | 86.8 |
| 4 | ✓ | ✓ | ✓ | ✓ | 98.2 | 97.8 | 95.2 | 99.8 | 83.5 | 89.6 |

#### Metric.

The NAVSIM benchmark provides a non-reactive simulation environment and employs the Predictive Driver Model Score (PDMS) as its closed-loop planning metric:

$$
\mathrm{PDMS}=NC\times DAC\times\left(\frac{5\times EP+5\times TTC+2\times C}{%
12}\right),
$$

where PDMS integrates five sub-metrics: No At-Fault Collision (NC), Drivable Area Compliance (DAC), Time-to-Collision (TTC), Comfort (Comf.), and Ego Progress (EP) to produce a comprehensive closed-loop planning score.

### 4.3 Main Results and Ablation Study

#### Experiments on the NAVSIM Benchmark.

Tab. 1 presents the results of ReCogDrive compared with existing methods on the NAVSIM dataset. ReCogDrive achieves a PDMS of 89.6, setting a new state-of-the-art. Notably, it surpasses DiffusionDrive [^12] and WoTE [^60], both of which use camera and LiDAR inputs, by 1.5 and 1.2 PDMS, respectively, despite relying solely on camera data. Furthermore, compared to our reproduced InternVL3 <sup>†</sup> [^30] and QwenVL2.5 <sup>†</sup> [^32] baselines trained directly on NAVSIM trajectories, ReCogDrive improves PDMS by 6.3, demonstrating the effectiveness of our three-stage training paradigm. In addition, it surpasses the previous vision-only state-of-the-art PARA-Drive [^56] by 5.6 PDMS.

#### Ablation study on ReCogDrive.

Tab. 2 presents an ablation study on the proposed components of ReCogDrive. When training InternVL3 solely on NAVSIM trajectory data, the model achieves a PDMS of 83.3. Adapting the VLM to driving scenarios with our large-scale driving QA data increases PDMS by 2.9. Introducing the diffusion planner for continuous trajectory prediction further raises PDMS by 0.6. Finally, simulator-assisted reinforcement learning achieves a PDMS of 89.6 with 2.8 improvement, demonstrating the value of our RL scheme in producing safer driving behavior.

Table 3: Effect of QA dataset scale and quality on planning. “LQ” and “HQ” denote low-quality and high-quality data, respectively.

| Training Data | NC | DAC | TTC | Conf. | EP | PDMS |
| --- | --- | --- | --- | --- | --- | --- |
| 85K(LQ) | 97.0 | 92.4 | 91.8 | 100 | 78.9 | 83.3 |
| 1.5M(LQ) | 97.5 | 93.6 | 93.2 | 100 | 79.4 | 84.6 |
| 3.2M(LQ) | 97.9 | 93.8 | 94.1 | 100 | 79.7 | 85.3 |
| 3.1M(HQ) | 98.2 | 94.5 | 94.5 | 100 | 80.4 | 86.2 |

Table 4: Impact of different discount factors $\gamma$. We use discounting to reduce the influence of early-step noise during the diffusion process.

| Discount | NC | DAC | TTC | Conf. | EP | PDMS |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 97.8 | 97.4 | 94.6 | 99.7 | 82.7 | 88.8 |
| 0.6 | 98.2 | 97.8 | 95.2 | 99.8 | 83.5 | 89.6 |
| 0.8 | 98.1 | 97.4 | 95.2 | 100 | 82.7 | 89.1 |
| 1.0 | 97.8 | 97.3 | 94.5 | 99.9 | 82.3 | 88.5 |

Table 5: Impact of Different BC Loss Weights $\lambda$.

| BC Wt. | NC | DAC | TTC | Conf. | EP | PDMS |
| --- | --- | --- | --- | --- | --- | --- |
| 0.001 | 97.0 | 97.0 | 93.2 | 98.3 | 78.9 | 85.9 |
| 0.01 | 98.2 | 97.8 | 95.2 | 99.8 | 83.5 | 89.6 |
| 0.1 | 98.1 | 97.2 | 94.0 | 99.7 | 83.4 | 88.6 |

Table 6: Impact of Different Min Samplings $\sigma_{\min}^{\exp}$.

| Min Samp. | NC | DAC | TTC | Conf. | EP | PDMS |
| --- | --- | --- | --- | --- | --- | --- |
| 0.01 | 97.1 | 97.2 | 93.1 | 100 | 83.0 | 88.0 |
| 0.02 | 98.2 | 97.8 | 95.2 | 99.8 | 83.5 | 89.6 |
| 0.04 | 97.8 | 97.6 | 94.8 | 99.3 | 80.3 | 87.8 |

#### Scaling Laws of QA Data for Planning Performance.

We collected a large-scale, high-quality driving QA dataset to adapt VLMs to real-world driving scenarios. As shown in Tab. 4, increasing the number of QA pairs from 85k to 3.2M increases the PDMS from 83.3 to 85.3, demonstrating the necessity of driving-specific pre-training and confirming that scaling laws hold under our conditions. Furthermore, filtering and rewriting the entire 3.2M set yields an additional 0.9 PDMS improvement to 86.2, highlighting the critical importance of data quality.

#### Impact of Different Discount Factors.

Tab. 4 shows the impact of the discount factor $\gamma$ on RL fine-tuning. When $\gamma=1.0$, all timesteps, including the very noisy early steps, contribute equally to the policy gradient, which amplifies high variability noise and destabilizes learning. In contrast, setting $\gamma=0.6$ focuses the update on later, more reliable steps and achieves the best PDMS of 89.6.

#### Impact of Different Behavior Clone Loss Weights.

Tab. 6 examines the impact of the BC loss weight $\lambda$. When $\lambda=0.001$, the BC term is too weak to stabilize learning and leads to training collapse. In contrast, setting $\lambda=0.1$ overemphasizes imitation, restricting policy improvement. The best trade-off is achieved at $\lambda=0.01$, which yields the highest PDMS.

#### Impact of Different Min Sampling Values.

Following [^34] [^26], we apply a cosine schedule to the diffusion noise variances $\sigma_{k}$ and clip them to a minimum value $\sigma_{\min}^{\exp}$. Clipping $\sigma_{k}$ to a nonzero minimum encourages the diffusion policy to sample more diverse trajectories, yet overly large $\sigma_{\min}^{\exp}$ can destabilize training. Setting $\sigma_{\min}^{\exp}=0.02$ achieves a PDMS of 89.6, as shown in Tab. 6.

![[x4.png|Refer to caption]]

Figure 4: Qualitative results of ReCogDrive on NAVSIM. ReCogDrive not only predicts smooth, collision-free trajectories but also outputs high-level instructions, scene descriptions, and object highlights in scenarios such as turns, straight driving, and intersections, demonstrating its perception and planning abilities. Best viewed on screen after zooming in.

#### Qualitative Results.

In Fig. 4, we visualize ReCogDrive’s perception and planning process on the NAVSIM. In addition to smooth trajectory predictions, ReCogDrive generates descriptive scene summaries and high-level driving instructions. It accurately detects critical objects, such as taxis and traffic lights, and leverages this information to inform its planning decisions.

## 5 Related Work

Vision-Language Models in Autonomous Driving. Numerous studies have leveraged the world knowledge embedded in VLMs to explore their application in autonomous driving scenarios. Current approaches for autonomous driving planning using VLMs can be categorized primarily into two types: dual-system [^13] [^14] and single-system [^15] [^16] [^17] [^18] [^19] [^20] [^19] [^21] [^22] [^23] [^61]. Dual-system methods, such as DriveVLM [^13] and Senna [^14], integrate VLMs with end-to-end driving systems [^1] [^2] [^3], where VLMs generate low-frequency trajectories or high-level commands and the end-to-end model refines them to produce the final trajectory. GPT-Driver [^16], EMMA [^15], and OpenEMMA [^17] reformulate the trajectory prediction task as a language modeling problem and introduce chain-of-thought [^62] reasoning to enhance explainability. Agency-Driver [^18] integrates an additional tool library, cognitive memory, and a reasoning engine to enhance Large Language Models (LLMs) capabilities in perception, prediction, and decision-making. OmniDrive [^19] and Atlas [^20] enhance the 3D scene perception capabilities of VLMs by incorporating a 3D tokenizer. Furthermore, LMDrive [^21] and WiseAD [^22] integrate VLMs driving systems into closed-loop evaluation frameworks to assess their performance in interactive environments. Sce2DriveX [^23] employs multimodal joint learning to significantly enhance the 3D perception capabilities of VLMs. DriveGPT4 [^51], DriveMM [^63], and DOLPHINS [^64] train VLMs to tackle multiple tasks in autonomous driving, including perception, prediction, planning, and decision-making.

Diffusion Models for Policy Learning. Diffusion models have recently shown great potential in image generation [^65] [^66], robotics [^67] [^68] [^69], and traffic simulation [^70] [^71] [^72]; however, their application in policy learning for autonomous driving remains underexplored. The Diffusion Policy [^37] extends diffusion models to robot learning, effectively handling multi-modal action distributions. GR00T-N1 [^36] integrates a VLM reasoning module with a DiT-based action module within a unified learning framework. HybirdVLA [^73] enhances manipulation robustness through a collaborative training approach and an action ensemble mechanism that adaptively combines diffusion- and autoregressive-based actions. In autonomous driving, DiffusionDrive [^12] introduces a truncated diffusion policy to mitigate mode collapse and reduce computational demands. Diffusion Planner [^35] redefines planning as a future trajectory generation task, jointly producing plans and motion forecasts.

Reinforcement Learning in Autonomous Driving. Reinforcement Learning is a promising technique that has demonstrated its effectiveness in LLMs [^74] [^75] and games [^76], and has been used to address specific scenarios in autonomous driving, such as highway driving [^77] and lane changes [^78]. Most methods directly learn policies over the control space, including throttle, brake, and steering commands, based on non-photorealistic simulators like CARLA [^79]. To address discrepancies between simulators and the real world, RAD [^80] proposes training an end-to-end AD agent using Reinforcement Learning in a photorealistic 3DGS environment. CarPlanner [^81] introduces an auto-regressive planner that trains an RL policy to generate consistent multi-modal trajectories, exceeding IL methods on the nuPlan dataset. Following the success of Deepseek-R1 [^74], GRPO has been applied to planning. AlphaDrive [^82] is the first system to integrate GRPO-based RL with planning reasoning for autonomous driving, significantly enhancing both performance and training efficiency. TrajHF [^83] propose a human feedback-driven finetuning framework for generative trajectory models, enabling alignment with diverse human driving preferences.

## 6 Conclusion

In this work, we propose ReCogDrive, an end-to-end autonomous driving system that integrates VLMs with diffusion-based trajectory planner, along with a three-stage training paradigm. First, we assemble and curate a 3.1M high-quality driving QA dataset to inject driving-specific cognition into a pre-trained VLM. Second, we train a diffusion-based trajectory planner via DDPM to map discrete language representations into smooth, continuous trajectories, while preserving the VLM’s inherent world cognition and driving-specific cognition. Finally, we reinforce the diffusion policy to integrate generalized driving cognition into the diffusion planner. Extensive experiments on NAVSIM demonstrate that ReCogDrive achieves state-of-the-art on closed-loop metrics without LiDAR input.

Limitations and broader impacts are discussed in the supplementary material.

## References

Supplementary Material

We organize the supplementary material as follows. We report ReCogDrive’s performance on the NAVSIM benchmark with extended metrics in Appx. A. In Appx. B, we detail our training data collection and processing pipeline. In Appx. C, we describe ReCogDrive’s training and inference implementation, including all key hyperparameters. Appx. D and Appx. E discuss the limitations of the method and the broader impacts. Finally, Appx. F presents additional qualitative results.

## Appendix A More Results

Experiments on NAVSIM with extended metrics. Hydra MDP++ [^58] introduces additional evaluation metrics: Traffic Light Compliance (TL), Lane Keeping Ability (LK), Driving Direction Compliance (DDC) and Extended Comfort (EC) to more comprehensively assess driving performance. We evaluate ReCogDrive on NAVSIM using these extended metrics as well. Tab. 7 compares our approach against existing methods under this metrics. ReCogDrive achieves state of the art scores in Driving Direction Compliance (DDC), Lane Keeping Ability (LK), Ego Progress (EP) and Time to Collision (TTC), and delivers a 0.5 improvement in EPDMS over ARTEMIS [^59], demonstrating the effectiveness of our method.

Table 7: Performance comparison on Navtest Benchmark with extended metrics.

| Method | NC $\uparrow$ | DAC $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | C $\uparrow$ | TL $\uparrow$ | DDC $\uparrow$ | LK $\uparrow$ | EC $\uparrow$ | EPDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Transfuser [^11] | 97.7 | 92.8 | 79.2 | 92.8 | 100 | 99.9 | 98.3 | 67.6 | 95.3 | 77.8 |
| VADv2 [^3] | 97.3 | 91.7 | 77.6 | 92.7 | 100 | 99.9 | 98.2 | 66.0 | 97.4 | 76.6 |
| Hydra-MDP [^58] | 97.5 | 96.3 | 80.1 | 93.0 | 100 | 99.9 | 98.3 | 65.5 | 97.4 | 79.8 |
| Hydra-MDP++ [^58] | 97.9 | 96.5 | 79.2 | 93.4 | 100 | 100.0 | 98.9 | 67.2 | 97.7 | 80.6 |
| ARTEMIS [^59] | 98.3 | 95.1 | 81.5 | 97.4 | 100 | 99.8 | 98.6 | 96.5 | 98.3 | 83.1 |
| Ours | 98.3 | 95.2 | 87.1 | 97.5 | 98.3 | 99.8 | 99.5 | 96.6 | 86.5 | 83.6 |

## Appendix B Training Datasets Construction

### B.1 Data Collection

We collect twelve open-source driving QA datasets, including Talk2Car [^48], SUTD [^49], NuScenes-QA [^50], OmniDrive [^19], and others, yielding over 3.1 million question-answer pairs that cover perception, prediction, and planning tasks across diverse real-world scenarios.

Talk2car [^48] is built on the nuScenes [^84] dataset and contains 850 videos from the nuScenes [^84] training set. This dataset has 11,959 natural language commands.

SUTD [^49] contains 10,080 in-the-wild videos and annotated 62,535 QA pairs. These videos are obtained through a combination of online collection and offline shooting, covering various weather conditions, times, and road conditions.

DRAMA [^29] is a dataset collected to investigate risk location and natural language description in driving scenarios. It contains 17,785 interactive driving scenarios.

NuScenes-QA [^50] encompasses 34,149 complex autonomous driving scenes and 459,941 question-answer pairs, with various types of questions. It aims to evaluate a model’s ability to understand and reason about complex visual data in multi-modal, multi-frame, and outdoor scenarios.

DriveGPT4 [^51] is built based on the BDD-X [^85] dataset, which contains about 20,000 samples. By dividing them into 16,803 training segments and 2,123 testing segments, question-answer pairs are generated using the control signal data and text annotations.

LingoQA [^28] contains 28K unique short video scenarios and 419K annotations. The dataset covers various questions related to driving scenarios, including aspects such as behaviors, scenery, and perception.

DriveLM [^27] consists of a training set of 4,072 frames and a validation set of 799 frames, with an average of 91.4 QA pairs per frame.

MAPLM [^52] contains point-cloud BEV projections and surround view images of various traffic scenes, such as highways and urban roads, and is equipped with element-level, lane-level, and road-level scene descriptions.

NuInstruct [^31] is a dataset constructed based on Nuscenes [^84], containing 91K multi-view video instruction-response pairs in 17 subtasks.

CODA-LM [^53] comprises 9,768 real-world driving scenarios with 41,722 textual annotations for critical road entities and 21,537 annotations for road corner cases.

OminiDrive [^19] covers 3D perception, reasoning, and planning tasks, including offline and online question-response tasks.

Senna [^14] design a series of planning-oriented QAs including scene description, traffic signal detection, vulnerable road user identification, motion intention prediction, meta-action planning and planning explanation. Since the senna dataset is not publicly available, we used Qwen2.5-VL-72B [^32] for question-answer data annotation.

![[x5.png|Refer to caption]]

Figure 5: Dataset Construction Pipeline. Step 1: collect open-source driving QA datasets and NAVSIM samples. Step 2: process data via normalization, augmentation, and filtering. Step 3: automatic annotation pipeline generates QA pairs for perception, prediction, and planning tasks. The pie chart summarizes category distribution.

### B.2 Data Processing

Data Normalization. Open-source driving datasets use a variety of bounding-box formats. For example, DriveLM [^27] represents an object as “<c2,CAM\_BACK,864.2,468.3>", while NuInstruct [^31] uses “<car>\[c2,584,478,603,516\]”. We convert all such variants into the following standardized tag format “<car><FRONT\_VIEW><box>\[$x_{1},y_{1},x_{2},y_{2}$\]</box>” adhering to the InternVL3 [^30] pre-training format. Each coordinate $x_{1},y_{1},x_{2},y_{2}$ is scaled from pixel space to the integer range \[0,1000\] based on the original image width and height.

Data Augmentation. Many QA datasets such as CODA-LM (only three question formats) and DRAMA (fixed, terse answers) lack linguistic variety and can hinder VLM performance. Following DriveMM [^63], we first employ an LLM to generate paraphrased question templates for any dataset with limited variants. Then, we prompt Qwen2.5-VL [^32] with the original image, question, and answer to produce richer, more varied responses. This two-stage process markedly enhances both question and answer diversity.

Data Filtering. We use Qwen2.5-VL [^32] to assign each QA pair a quality score based on predefined criteria. Any pair scoring below 60 is removed. After filtering, we retain 2.3 M high-quality pairs.

### B.3 Automatic Annotation Pipeline

To improve VLMs performance on NAVSIM [^45], we constructed a cost-effective annotation pipeline driven by Qwen2.5-VL [^32]. We combine NAVSIM’s existing sensor and trajectory labels with carefully crafted prompts to elicit answers covering a spectrum of autonomous driving tasks. These include perception tasks such as scene description, key object identification, driving behavior narration, road marking recognition, traffic light classification and vulnerable road user detection; prediction tasks like motion prediction; planning tasks including high-level driving command prediction and decision explanation; and even counterfactual reasoning scenarios inspired by OmniDrive [^19]. Applying this pipeline to NAVSIM yielded 775 K high-quality QA pairs, which we used to fine-tune the VLMs and substantially enhance its planning ability.

### B.4 Dataset Statistics

The dataset consists of 41% perception samples, 11% prediction samples, and 24% planning samples, aimed at improving VLMs’ understanding of driving scenarios. We also include 24% visual instruction samples to maintain the model’s instruction-following ability.

## Appendix C Implementation Details

In this section, we detail our model configuration, the hyperparameters used in the three-stage training and evaluation, and our hardware setup.

Model Architecture and Hyperparameter Details. Our model architecture consists of two main components: Vision-Language Models (VLMs) and a diffusion-based trajectory planner. For the Vision-Language Models, we utilize the pre-trained InternVL3-8B [^30] model. This model processes visual inputs by splitting the image into patches, with a maximum of 12 dynamic patches allowed. The input images are cropped into 448x448 pixel patches. For the diffusion planner, we use the DiT [^38] architecture, employing DDPM/DDIM [^25] denoising methods. During training, the denoising process involves 100 steps, while for inference, the diffusion process is reduced to 5 steps. The hidden layer size of the DiT architecture is set to 1536, with a head size of 32 and 16 layers.

Training Configuration. In the first stage, we fine-tune the VLM on the combined driving QA dataset for three epochs, with all parameters unfrozen and a batch size of 1,024. We use AdamW with a base learning rate of $4e^{-5}$, weight decay of 0.05, and a cosine learning rate schedule with a 10% linear warmup. In the second stage, we train the diffusion-based planner with behavior cloning for 200 epochs and a batch size of 512. We use AdamW with a learning rate that warms up to $1e^{-4}$ over the first 1.5% of steps, then decays cosinely to $1e^{-6}$, and a weight decay of $1e^{-4}$. In the third stage, we perform simulator-assisted reinforcement learning for 10 epochs with a batch size of 128, using AdamW and a cosine schedule that decays the learning rate from $1e^{-4}$ to 0. Additionally, we stabilize diffusion training and inference by clipping the reconstructed clean estimate $x_{0}$ to $\pm 1.0$, clipping Gaussian noise samples to $\pm 3.0$, enforcing a nonzero floor on the denoising standard deviation $\sigma_{t}$ of 0.02 for training, clamping $\sigma_{t}$ to at least 0.10 when computing $\log p(x_{t-1}\!\mid\!x_{t})$, and applying a discount factor $\gamma=0.6$ during RL fine-tuning to downweight early, noisy timesteps.

Table 8: Hyperparameters for ReCogDrive.

<table><tbody><tr><td>Stage</td><td>Parameter</td><td>Value</td></tr><tr><td rowspan="6">I</td><td>Number of epochs</td><td>3</td></tr><tr><td>Batch size</td><td>1024</td></tr><tr><td>Learning rate</td><td><math><semantics><mrow><mn>4</mn> <mo>⁢</mo> <msup><mi>e</mi> <mrow><mo>−</mo> <mn>5</mn></mrow></msup></mrow> <annotation-xml><apply><cn>4</cn> <apply><csymbol>superscript</csymbol> <ci>𝑒</ci> <apply><cn>5</cn></apply></apply></apply></annotation-xml> <annotation>4e^{-5}</annotation> <annotation>4 italic_e start_POSTSUPERSCRIPT - 5 end_POSTSUPERSCRIPT</annotation></semantics></math></td></tr><tr><td>Weight decay</td><td>0.05</td></tr><tr><td>Warmup ratio</td><td>0.10</td></tr><tr><td>Learning rate schedule</td><td>Cosine</td></tr><tr><td rowspan="4">II</td><td>Number of epochs</td><td>200</td></tr><tr><td>Batch size</td><td>512</td></tr><tr><td>Learning rate</td><td><math><semantics><mrow><mn>1</mn> <mo>⁢</mo> <msup><mi>e</mi> <mrow><mo>−</mo> <mn>4</mn></mrow></msup></mrow> <annotation-xml><apply><cn>1</cn> <apply><csymbol>superscript</csymbol> <ci>𝑒</ci> <apply><cn>4</cn></apply></apply></apply></annotation-xml> <annotation>1e^{-4}</annotation> <annotation>1 italic_e start_POSTSUPERSCRIPT - 4 end_POSTSUPERSCRIPT</annotation></semantics></math></td></tr><tr><td>Learning rate schedule</td><td>Cosine</td></tr><tr><td></td><td>Weight decay</td><td><math><semantics><mrow><mn>1</mn> <mo>×</mo> <msup><mn>10</mn> <mrow><mo>−</mo> <mn>4</mn></mrow></msup></mrow> <annotation-xml><apply><cn>1</cn> <apply><csymbol>superscript</csymbol> <cn>10</cn> <apply><cn>4</cn></apply></apply></apply></annotation-xml> <annotation>1\times 10^{-4}</annotation> <annotation>1 × 10 start_POSTSUPERSCRIPT - 4 end_POSTSUPERSCRIPT</annotation></semantics></math></td></tr><tr><td rowspan="11">III</td><td>Number of epochs</td><td>10</td></tr><tr><td>Batch size</td><td>128</td></tr><tr><td>Learning rate</td><td><math><semantics><mrow><mn>1</mn> <mo>⁢</mo> <msup><mi>e</mi> <mrow><mo>−</mo> <mn>4</mn></mrow></msup></mrow> <annotation-xml><apply><cn>1</cn> <apply><csymbol>superscript</csymbol> <ci>𝑒</ci> <apply><cn>4</cn></apply></apply></apply></annotation-xml> <annotation>1e^{-4}</annotation> <annotation>1 italic_e start_POSTSUPERSCRIPT - 4 end_POSTSUPERSCRIPT</annotation></semantics></math></td></tr><tr><td>Learning rate schedule</td><td>Cosine</td></tr><tr><td>BC loss weight</td><td><math><semantics><mrow><mn>1</mn> <mo>⁢</mo> <msup><mi>e</mi> <mrow><mo>−</mo> <mn>2</mn></mrow></msup></mrow> <annotation-xml><apply><cn>1</cn> <apply><csymbol>superscript</csymbol> <ci>𝑒</ci> <apply><cn>2</cn></apply></apply></apply></annotation-xml> <annotation>1e^{-2}</annotation> <annotation>1 italic_e start_POSTSUPERSCRIPT - 2 end_POSTSUPERSCRIPT</annotation></semantics></math></td></tr><tr><td>Denoised clip threshold</td><td><math><semantics><mrow><mo>±</mo> <mn>1.0</mn></mrow> <annotation-xml><apply><csymbol>plus-or-minus</csymbol> <cn>1.0</cn></apply></annotation-xml> <annotation>\pm 1.0</annotation> <annotation>± 1.0</annotation></semantics></math></td></tr><tr><td>Noise clip threshold</td><td><math><semantics><mrow><mo>±</mo> <mn>3.0</mn></mrow> <annotation-xml><apply><csymbol>plus-or-minus</csymbol> <cn>3.0</cn></apply></annotation-xml> <annotation>\pm 3.0</annotation> <annotation>± 3.0</annotation></semantics></math></td></tr><tr><td>Minimum denoising std</td><td>0.02</td></tr><tr><td>Minimum log-variance std</td><td>0.10</td></tr><tr><td>Discount factor</td><td>0.6</td></tr></tbody></table>

Hardware Configuration. We implement ReCogDrive on Debian with PyTorch, training across four nodes—each equipped with an Intel® Xeon® Platinum 8457C CPU and eight NVIDIA H20 GPUs (32 GPUs in total). Inference is performed on a single node with 8 GPUs.

## Appendix D Limitations

Although ReCogDrive achieves state of the art performance on the NAVSIM benchmark, it still faces several limitations. These include the difficulty of processing multiple camera inputs, handling video frame sequences, and relatively high inference latency. Furthermore, our reinforcement learning is conducted in the non-interactive NAVSIM simulator and cannot interact with a live environment; training in a fully closed-loop environment remains an open challenge. Future work may address these issues by designing a 3D vision encoder that aligns with textual features, developing more efficient model architectures, and performing reinforcement learning in interactive closed-loop environments. We also plan to deploy and evaluate our model on real vehicles.

## Appendix E Broader Impacts

Our research promotes the application of Vision-Language Models (VLMs) in the autonomous driving domain. Through simulator-assisted reinforcement learning, our model more effectively mitigates collision risk and generalizes to rare, challenging scenarios. This advancement could lead to safer, more reliable autonomous systems capable of handling real-world driving scenarios.

## Appendix F Qualitative Results

We further showcase ReCogDrive’s performance across diverse scenarios. Not only does ReCogDrive generate safe and smooth continuous trajectories, it also produces detailed scene descriptions, issues warnings for critical road elements, and outputs high-level driving commands.

![[x6.png|Refer to caption]]

Refer to caption

![[x7.png|Refer to caption]]

Refer to caption

![[x8.png|Refer to caption]]

Refer to caption

[^1]: Bo Jiang, Shaoyu Chen, Qing Xu, Bencheng Liao, Jiajie Chen, Helong Zhou, Qian Zhang, Wenyu Liu, Chang Huang, and Xinggang Wang. Vad: Vectorized scene representation for efficient autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8340–8350, 2023.

[^2]: Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai Wang, et al. Planning-oriented autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 17853–17862, 2023.

[^3]: Shaoyu Chen, Bo Jiang, Hao Gao, Bencheng Liao, Qing Xu, Qian Zhang, Chang Huang, Wenyu Liu, and Xinggang Wang. Vadv2: End-to-end vectorized autonomous driving via probabilistic planning. arXiv preprint arXiv:2402.13243, 2024.

[^4]: Shengchao Hu, Li Chen, Penghao Wu, Hongyang Li, Junchi Yan, and Dacheng Tao. St-p3: End-to-end vision-based autonomous driving via spatial-temporal feature learning. In European Conference on Computer Vision, pages 533–549. Springer, 2022.

[^5]: Diankun Zhang, Guoan Wang, Runwen Zhu, Jianbo Zhao, Xiwu Chen, Siyu Zhang, Jiahao Gong, Qibin Zhou, Wenyuan Zhang, Ningzi Wang, et al. Sparsead: Sparse query-centric paradigm for efficient end-to-end autonomous driving. arXiv preprint arXiv:2404.06892, 2024.

[^6]: Xiaohui Jiang, Shuailin Li, Yingfei Liu, Shihao Wang, Fan Jia, Tiancai Wang, Lijin Han, and Xiangyu Zhang. Far3d: Expanding the horizon for surround-view 3d object detection. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 2561–2569, 2024.

[^7]: Jonah Philion and Sanja Fidler. Lift, splat, shoot: Encoding images from arbitrary camera rigs by implicitly unprojecting to 3d. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XIV 16, pages 194–210. Springer, 2020.

[^8]: Diankun Zhang, Zhijie Zheng, Haoyu Niu, Xueqing Wang, and Xiaojun Liu. Fully sparse transformer 3-d detector for lidar point cloud. IEEE Transactions on Geoscience and Remote Sensing, 61:1–12, 2023.

[^9]: Yuning Chai, Benjamin Sapp, Mayank Bansal, and Dragomir Anguelov. Multipath: Multiple probabilistic anchor trajectory hypotheses for behavior prediction. arXiv preprint arXiv:1910.05449, 2019.

[^10]: Junru Gu, Chenxu Hu, Tianyuan Zhang, Xuanyao Chen, Yilun Wang, Yue Wang, and Hang Zhao. Vip3d: End-to-end visual trajectory prediction via 3d agent queries. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5496–5506, 2023.

[^11]: Kashyap Chitta, Aditya Prakash, Bernhard Jaeger, Zehao Yu, Katrin Renz, and Andreas Geiger. Transfuser: Imitation with transformer-based sensor fusion for autonomous driving. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(11):12878–12895, 2022.

[^12]: Bencheng Liao, Shaoyu Chen, Haoran Yin, Bo Jiang, Cheng Wang, Sixu Yan, Xinbang Zhang, Xiangyu Li, Ying Zhang, Qian Zhang, et al. Diffusiondrive: Truncated diffusion model for end-to-end autonomous driving. arXiv preprint arXiv:2411.15139, 2024.

[^13]: Xiaoyu Tian, Junru Gu, Bailin Li, Yicheng Liu, Yang Wang, Zhiyong Zhao, Kun Zhan, Peng Jia, Xianpeng Lang, and Hang Zhao. Drivevlm: The convergence of autonomous driving and large vision-language models. arXiv preprint arXiv:2402.12289, 2024.

[^14]: Bo Jiang, Shaoyu Chen, Bencheng Liao, Xingyu Zhang, Wei Yin, Qian Zhang, Chang Huang, Wenyu Liu, and Xinggang Wang. Senna: Bridging large vision-language models and end-to-end autonomous driving. arXiv preprint arXiv:2410.22313, 2024.

[^15]: Jyh-Jing Hwang, Runsheng Xu, Hubert Lin, Wei-Chih Hung, Jingwei Ji, Kristy Choi, Di Huang, Tong He, Paul Covington, Benjamin Sapp, et al. Emma: End-to-end multimodal model for autonomous driving. arXiv preprint arXiv:2410.23262, 2024.

[^16]: Jiageng Mao, Yuxi Qian, Junjie Ye, Hang Zhao, and Yue Wang. Gpt-driver: Learning to drive with gpt. arXiv preprint arXiv:2310.01415, 2023.

[^17]: Shuo Xing, Chengyuan Qian, Yuping Wang, Hongyuan Hua, Kexin Tian, Yang Zhou, and Zhengzhong Tu. Openemma: Open-source multimodal model for end-to-end autonomous driving. arXiv preprint arXiv:2412.15208, 2024.

[^18]: Jiageng Mao, Junjie Ye, Yuxi Qian, Marco Pavone, and Yue Wang. A language agent for autonomous driving. arXiv preprint arXiv:2311.10813, 2023.

[^19]: Shihao Wang, Zhiding Yu, Xiaohui Jiang, Shiyi Lan, Min Shi, Nadine Chang, Jan Kautz, Ying Li, and Jose M Alvarez. Omnidrive: A holistic llm-agent framework for autonomous driving with 3d perception, reasoning and planning. arXiv preprint arXiv:2405.01533, 2024.

[^20]: Yifan Bai, Dongming Wu, Yingfei Liu, Fan Jia, Weixin Mao, Ziheng Zhang, Yucheng Zhao, Jianbing Shen, Xing Wei, Tiancai Wang, et al. Is a 3d-tokenized llm the key to reliable autonomous driving? arXiv preprint arXiv:2405.18361, 2024.

[^21]: Hao Shao, Yuxuan Hu, Letian Wang, Guanglu Song, Steven L Waslander, Yu Liu, and Hongsheng Li. Lmdrive: Closed-loop end-to-end driving with large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15120–15130, 2024.

[^22]: Songyan Zhang, Wenhui Huang, Zihui Gao, Hao Chen, and Chen Lv. Wisead: Knowledge augmented end-to-end autonomous driving with vision-language model. arXiv preprint arXiv:2412.09951, 2024.

[^23]: Rui Zhao, Qirui Yuan, Jinyu Li, Haofeng Hu, Yun Li, Chengyuan Zheng, and Fei Gao. Sce2drivex: A generalized mllm framework for scene-to-drive learning. arXiv preprint arXiv:2502.14917, 2025.

[^24]: Ruijun Zhang, Xianda Guo, Wenzhao Zheng, Chenming Zhang, Kurt Keutzer, and Long Chen. Instruct large language models to drive like humans. arXiv preprint arXiv:2406.07296, 2024.

[^25]: Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

[^26]: Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In International conference on machine learning, pages 8162–8171. PMLR, 2021.

[^27]: Chonghao Sima, Katrin Renz, Kashyap Chitta, Li Chen, Hanxue Zhang, Chengen Xie, Jens Beißwenger, Ping Luo, Andreas Geiger, and Hongyang Li. Drivelm: Driving with graph visual question answering. In European Conference on Computer Vision, pages 256–274. Springer, 2024.

[^28]: Ana-Maria Marcu, Long Chen, Jan Hünermann, Alice Karnsund, Benoit Hanotte, Prajwal Chidananda, Saurabh Nair, Vijay Badrinarayanan, Alex Kendall, Jamie Shotton, et al. Lingoqa: Visual question answering for autonomous driving. In European Conference on Computer Vision, pages 252–269. Springer, 2024.

[^29]: Srikanth Malla, Chiho Choi, Isht Dwivedi, Joon Hee Choi, and Jiachen Li. Drama: Joint risk localization and captioning in driving. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 1043–1052, 2023.

[^30]: Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Yuchen Duan, Hao Tian, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.

[^31]: Xinpeng Ding, Jianhua Han, Hang Xu, Xiaodan Liang, Wei Zhang, and Xiaomeng Li. Holistic autonomous driving understanding by bird’s-eye-view injected multi-modal large models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13668–13677, 2024.

[^32]: Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

[^33]: Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24185–24198, 2024.

[^34]: Allen Z Ren, Justin Lidard, Lars L Ankile, Anthony Simeonov, Pulkit Agrawal, Anirudha Majumdar, Benjamin Burchfiel, Hongkai Dai, and Max Simchowitz. Diffusion policy policy optimization. arXiv preprint arXiv:2409.00588, 2024.

[^35]: Yinan Zheng, Ruiming Liang, Kexin Zheng, Jinliang Zheng, Liyuan Mao, Jianxiong Li, Weihao Gu, Rui Ai, Shengbo Eben Li, Xianyuan Zhan, et al. Diffusion-based planning for autonomous driving with flexible guidance. arXiv preprint arXiv:2501.15564, 2025.

[^36]: Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.

[^37]: Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, page 02783649241273668, 2023.

[^38]: William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.

[^39]: Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V Le, Sergey Levine, and Yi Ma. Sft memorizes, rl generalizes: A comparative study of foundation model post-training. arXiv preprint arXiv:2501.17161, 2025.

[^40]: Takayuki Osa, Joni Pajarinen, Gerhard Neumann, J Andrew Bagnell, Pieter Abbeel, Jan Peters, et al. An algorithmic perspective on imitation learning. Foundations and Trends® in Robotics, 7(1-2):1–179, 2018.

[^41]: Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8:229–256, 1992.

[^42]: Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

[^43]: Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, and Sara Hooker. Back to basics: Revisiting reinforce style optimization for learning from human feedback in llms. arXiv preprint arXiv:2402.14740, 2024.

[^44]: An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024.

[^45]: Daniel Dauner, Marcel Hallgarten, Tianyu Li, Xinshuo Weng, Zhiyu Huang, Zetong Yang, Hongyang Li, Igor Gilitschenski, Boris Ivanovic, Marco Pavone, et al. Navsim: Data-driven non-reactive autonomous vehicle simulation and benchmarking. Advances in Neural Information Processing Systems, 37:28706–28719, 2025.

[^46]: OpenScene Contributors. Openscene: The largest up-to-date 3d occupancy prediction benchmark in autonomous driving. [https://github.com/OpenDriveLab/OpenScene](https://github.com/OpenDriveLab/OpenScene), 2023.

[^47]: Holger Caesar, Juraj Kabzan, Kok Seang Tan, Whye Kit Fong, Eric Wolff, Alex Lang, Luke Fletcher, Oscar Beijbom, and Sammy Omari. nuplan: A closed-loop ml-based planning benchmark for autonomous vehicles. arXiv preprint arXiv:2106.11810, 2021.

[^48]: Thierry Deruyttere, Simon Vandenhende, Dusan Grujicic, Luc Van Gool, and Marie-Francine Moens. Talk2car: Taking control of your self-driving car. arXiv preprint arXiv:1909.10838, 2019.

[^49]: Li Xu, He Huang, and Jun Liu. Sutd-trafficqa: A question answering benchmark and an efficient network for video reasoning over traffic events. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9878–9888, 2021.

[^50]: Tianwen Qian, Jingjing Chen, Linhai Zhuo, Yang Jiao, and Yu-Gang Jiang. Nuscenes-qa: A multi-modal visual question answering benchmark for autonomous driving scenario. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 4542–4550, 2024.

[^51]: Zhenhua Xu, Yujia Zhang, Enze Xie, Zhen Zhao, Yong Guo, Kwan-Yee K Wong, Zhenguo Li, and Hengshuang Zhao. Drivegpt4: Interpretable end-to-end autonomous driving via large language model. IEEE Robotics and Automation Letters, 2024.

[^52]: Xu Cao, Tong Zhou, Yunsheng Ma, Wenqian Ye, Can Cui, Kun Tang, Zhipeng Cao, Kaizhao Liang, Ziran Wang, James M Rehg, et al. Maplm: A real-world large-scale vision-language benchmark for map and traffic scene understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21819–21830, 2024.

[^53]: Kai Chen, Yanze Li, Wenhua Zhang, Yanxin Liu, Pengxiang Li, Ruiyuan Gao, Lanqing Hong, Meng Tian, Xinhai Zhao, Zhenguo Li, et al. Automated evaluation of large vision-language models on self-driving corner cases. arXiv preprint arXiv:2404.10595, 2024.

[^54]: Yuntao Chen, Yuqi Wang, and Zhaoxiang Zhang. Drivinggpt: Unifying driving world modeling and planning with multi-modal autoregressive transformers. arXiv preprint arXiv:2412.18607, 2024.

[^55]: Ze Yu, Jun Li, Yuzhen Wei, Yuandong Lyu, and Xiaojun Tan. Combining camera–lidar fusion and motion planning using bird’s-eye view representation for end-to-end autonomous driving. Drones, 9(4):281, 2025.

[^56]: Xinshuo Weng, Boris Ivanovic, Yan Wang, Yue Wang, and Marco Pavone. Para-drive: Parallelized architecture for real-time autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15449–15458, 2024.

[^57]: Chengran Yuan, Zhanqi Zhang, Jiawei Sun, Shuo Sun, Zefan Huang, Christina Dao Wen Lee, Dongen Li, Yuhang Han, Anthony Wong, Keng Peng Tee, et al. Drama: An efficient end-to-end motion planner for autonomous driving with mamba. arXiv preprint arXiv:2408.03601, 2024.

[^58]: Zhenxin Li, Kailin Li, Shihao Wang, Shiyi Lan, Zhiding Yu, Yishen Ji, Zhiqi Li, Ziyue Zhu, Jan Kautz, Zuxuan Wu, et al. Hydra-mdp: End-to-end multimodal planning with multi-target hydra-distillation. arXiv preprint arXiv:2406.06978, 2024.

[^59]: Renju Feng, Ning Xi, Duanfeng Chu, Rukang Wang, Zejian Deng, Anzheng Wang, Liping Lu, Jinxiang Wang, and Yanjun Huang. Artemis: Autoregressive end-to-end trajectory planning with mixture of experts for autonomous driving. arXiv preprint arXiv:2504.19580, 2025.

[^60]: Yingyan Li, Yuqi Wang, Yang Liu, Jiawei He, Lue Fan, and Zhaoxiang Zhang. End-to-end driving with online trajectory evaluation via bev world model. arXiv preprint arXiv:2504.01941, 2025.

[^61]: Haoyu Fu, Diankun Zhang, Zongchuang Zhao, Jianfeng Cui, Dingkang Liang, Chong Zhang, Dingyuan Zhang, Hongwei Xie, Bing Wang, and Xiang Bai. Orion: A holistic end-to-end autonomous driving framework by vision-language instructed action generation. arXiv preprint arXiv:2503.19755, 2025.

[^62]: Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

[^63]: Zhijian Huang, Chengjian Feng, Feng Yan, Baihui Xiao, Zequn Jie, Yujie Zhong, Xiaodan Liang, and Lin Ma. Drivemm: All-in-one large multimodal model for autonomous driving. arXiv preprint arXiv:2412.07689, 2024.

[^64]: Yingzi Ma, Yulong Cao, Jiachen Sun, Marco Pavone, and Chaowei Xiao. Dolphins: Multimodal language model for driving. In European Conference on Computer Vision, pages 403–420. Springer, 2024.

[^65]: Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all. arXiv preprint arXiv:2412.20404, 2024.

[^66]: Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

[^67]: Songming Liu, Lingxuan Wu, Bangguo Li, Hengkai Tan, Huayu Chen, Zhengyi Wang, Ke Xu, Hang Su, and Jun Zhu. Rdt-1b: a diffusion foundation model for bimanual manipulation. arXiv preprint arXiv:2410.07864, 2024.

[^68]: Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. A vision-languageaction flow model for general robot control. arXiv preprint arXiv:2410.24164, 2(3):5, 2024.

[^69]: Sixu Yan, Zeyu Zhang, Muzhi Han, Zaijin Wang, Qi Xie, Zhitian Li, Zhehan Li, Hangxin Liu, Xinggang Wang, and Song-Chun Zhu. M 2 diffuser: Diffusion-based trajectory optimization for mobile manipulation in 3d scenes. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[^70]: Brian Yang, Huangyuan Su, Nikolaos Gkanatsios, Tsung-Wei Ke, Ayush Jain, Jeff Schneider, and Katerina Fragkiadaki. Diffusion-es: Gradient-free planning with diffusion for autonomous and instruction-guided driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15342–15353, 2024.

[^71]: Ziyuan Zhong, Davis Rempe, Danfei Xu, Yuxiao Chen, Sushant Veer, Tong Che, Baishakhi Ray, and Marco Pavone. Guided conditional diffusion for controllable traffic simulation. In 2023 IEEE international conference on robotics and automation (ICRA), pages 3560–3566. IEEE, 2023.

[^72]: Ziyuan Zhong, Davis Rempe, Yuxiao Chen, Boris Ivanovic, Yulong Cao, Danfei Xu, Marco Pavone, and Baishakhi Ray. Language-guided traffic simulation via scene-level diffusion. In Conference on Robot Learning, pages 144–177. PMLR, 2023.

[^73]: Jiaming Liu, Hao Chen, Pengju An, Zhuoyang Liu, Renrui Zhang, Chenyang Gu, Xiaoqi Li, Ziyu Guo, Sixiang Chen, Mengzhen Liu, et al. Hybridvla: Collaborative diffusion and autoregression in a unified vision-language-action model. arXiv preprint arXiv:2503.10631, 2025.

[^74]: Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

[^75]: Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

[^76]: Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse domains through world models. arXiv preprint arXiv:2301.04104, 2023.

[^77]: Edouard Leurent and Jean Mercat. Social attention for autonomous decision-making in dense traffic. arXiv preprint arXiv:1911.12250, 2019.

[^78]: Guofa Li, Yifan Yang, Shen Li, Xingda Qu, Nengchao Lyu, and Shengbo Eben Li. Decision making of autonomous vehicles in lane change scenarios: Deep reinforcement learning approaches with risk awareness. Transportation research part C: emerging technologies, 134:103452, 2022.

[^79]: Alexey Dosovitskiy, German Ros, Felipe Codevilla, Antonio Lopez, and Vladlen Koltun. Carla: An open urban driving simulator. In Conference on robot learning, pages 1–16. PMLR, 2017.

[^80]: Hao Gao, Shaoyu Chen, Bo Jiang, Bencheng Liao, Yiang Shi, Xiaoyang Guo, Yuechuan Pu, Haoran Yin, Xiangyu Li, Xinbang Zhang, et al. Rad: Training an end-to-end driving policy via large-scale 3dgs-based reinforcement learning. arXiv preprint arXiv:2502.13144, 2025.

[^81]: Dongkun Zhang, Jiaming Liang, Ke Guo, Sha Lu, Qi Wang, Rong Xiong, Zhenwei Miao, and Yue Wang. Carplanner: Consistent auto-regressive trajectory planning for large-scale reinforcement learning in autonomous driving. arXiv preprint arXiv:2502.19908, 2025.

[^82]: Bo Jiang, Shaoyu Chen, Qian Zhang, Wenyu Liu, and Xinggang Wang. Alphadrive: Unleashing the power of vlms in autonomous driving via reinforcement learning and reasoning. arXiv preprint arXiv:2503.07608, 2025.

[^83]: Derun Li, Jianwei Ren, Yue Wang, Xin Wen, Pengxiang Li, Leimeng Xu, Kun Zhan, Zhongpu Xia, Peng Jia, Xianpeng Lang, et al. Finetuning generative trajectory model with reinforcement learning from human feedback. arXiv preprint arXiv:2503.10434, 2025.

[^84]: Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11621–11631, 2020.

[^85]: Jinkyu Kim, Anna Rohrbach, Trevor Darrell, John Canny, and Zeynep Akata. Textual explanations for self-driving vehicles. In Proceedings of the European conference on computer vision (ECCV), pages 563–578, 2018.
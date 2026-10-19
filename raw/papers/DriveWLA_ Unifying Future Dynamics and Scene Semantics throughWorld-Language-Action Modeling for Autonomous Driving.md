---
title: "DriveWLA: Unifying Future Dynamics and Scene Semantics throughWorld-Language-Action Modeling for Autonomous Driving"
source: "https://arxiv.org/html/2610.17903v1"
author:
published:
created: 2026-10-19
description:
tags:
  - "clippings"
---
## DriveWLA: Unifying Future Dynamics and Scene Semantics through World-Language-Action Modeling for Autonomous Driving

Shuzheng Wang <sup>1,2</sup>, Haoxuan Li <sup>1,2</sup>, Shengzhen Li <sup>1,2</sup>, Chenxi Wu <sup>1,2</sup> Yajing Sun <sup>2</sup>, Lu Hou <sup>2</sup>, Wenyu Liu <sup>1</sup>, Xinggang Wang <sup>1,🖂</sup> <sup>1</sup> Huazhong University of Science and Technology   <sup>2</sup> Yinwang Intelligent Technology Co. Ltd.

###### Abstract

Vision-Language Models (VLMs) and world models have demonstrated strong capabilities in semantic understanding and future scene evolution for autonomous driving. However, effectively integrating both capabilities for trajectory generation remains challenging. In this work, we propose DriveWLA, a unified World-Language-Action model based on Mixture-of-Transformers (MoT) that jointly models semantic understanding, future scene dynamics, and trajectory generation. DriveWLA consists of three experts: a World Expert for future prediction, a Language Expert for semantic understanding, and an Action Expert for trajectory generation. We further introduce Cross-Expert Attention, which enables controlled information flow across experts while aligning semantic and dynamic representations. To improve trajectory generation beyond imitation learning, we apply Flow-GRPO reinforcement learning post-training to refine the Action Expert. Extensive experiments show that DriveWLA achieves superior performance in both open-loop and closed-loop evaluation. DriveWLA achieves 92.3 PDMS on NAVSIM v1 and 90.1 EPDMS on NAVSIM v2, demonstrating strong performance among VLM-based and world-model-based planners. Without specific fine-tuning, DriveWLA further demonstrates strong zero-shot generalization to HUGSIM, nuScenes, and Bench2Drive. These results validate the effectiveness of our unified World-Language-Action modeling, establishing DriveWLA as a generalizable framework for autonomous driving.

<sup>†</sup>

## 1 Introduction

Reliable autonomous driving requires three core capabilities: scene understanding, future prediction, and trajectory planning [^10] [^12] [^36]. Human drivers naturally combine current scene understanding with future prediction to make appropriate driving decisions [^6] [^39]. This motivates a key challenge for autonomous driving: how to jointly leverage semantic understanding and future dynamics for trajectory planning.

Vision-Language Models (VLMs) have recently demonstrated strong capabilities in semantic understanding and reasoning for autonomous driving [^31] [^55]. By leveraging large-scale vision-language pretraining and driving-specific adaptation, VLM-based driving systems can understand complex traffic scenarios and predict driving behaviors [^19] [^43] [^54]. However, VLMs inherently lack explicit mechanisms for future scene evolution, limiting their ability to predict future dynamics required for trajectory planning.

In contrast, world models can learn future evolution of driving scenes from large-scale driving videos and provide future-aware priors beyond current observations [^10] [^48] [^38]. However, future prediction alone is insufficient for planning, as the model must also understand the current scene to determine which future changes are relevant to driving. Therefore, semantic understanding and future modeling are complementary for trajectory planning.

Recent studies have explored integrating VLMs with world models for autonomous driving [^20] [^32] [^46]. However, most methods introduce one capability as an enhancement to another framework. For example, DriveVLA-W0 [^20] adds future-image prediction as an auxiliary objective for VLA training, while DriveWAM [^32] uses VLM features to condition video and action generation. Although these designs bring semantic understanding and future prediction into the same system, they do not explicitly model their interaction toward trajectory planning. As a result, the system cannot unlock the full potential of their complementary information for driving decisions due to insufficient refinement and alignment between the semantic and dynamic representations.

To address this challenge, we propose DriveWLA, a unified World-Language-Action framework for autonomous driving. DriveWLA introduces three specialized experts: a World Expert for future prediction, a Language Expert for semantic understanding, and an Action Expert for trajectory generation. To integrate these complementary capabilities, we introduce Cross-Expert Attention, which explicitly controls information exchange across experts. With this interaction mechanism, all experts are jointly trained under trajectory supervision during imitation learning, aligning the pretrained language and world representations with trajectory generation. We then refine the driving policy with an improved Flow-GRPO [^26], which uses factorized advantages and variance shrinkage for stable reward optimization. This reinforcement learning stage enables the model to explore better trajectories beyond expert demonstrations.

DriveWLA achieves 92.3 PDMS on NAVSIM v1 [^5], establishing state-of-the-art (SOTA) performance among recent VLM-based and world-model-based planners. We further evaluate its zero-shot capability on representative open-loop and closed-loop benchmarks without fine-tuning. DriveWLA achieves 41.47 HD-Score on HUGSIM [^51] and maintains low collision rates on nuScenes [^1] and Bench2Drive [^16], demonstrating strong generalization of the proposed unified architecture.

Our contributions can be summarized as follows:

- We introduce DriveWLA, a unified World-Language-Action framework that jointly models semantic understanding, future dynamics prediction, and trajectory generation through three specialized experts.
- We propose Cross-Expert Attention, a controllable mechanism that explicitly directs information flow across experts, enabling both efficient interaction and representation alignment.
- DriveWLA establishes a new state-of-the-art on NAVSIM v1 among VLM-based and world-model-based planners. Furthermore, it generalizes zero-shot to diverse open-loop and closed-loop benchmarks, achieving competitive driving performance and safety.

## 2 Related Work

![[drivewla_overview.png|Refer to caption]]

Figure 1: Model architecture and training strategy. DriveWLA unifies World, Language, and Action Experts through Cross-Expert Attention. The training strategy consists of trajectory imitation learning, which jointly optimizes all experts, followed by reinforcement learning refinement of the Action Expert.

Vision-Language Models and world models for autonomous driving. VLMs provide strong scene understanding and reasoning capabilities for driving and the VLM-based driving systems mainly leverage these capabilities through explicit textual reasoning or latent representations [^15] [^34] [^54] [^53]. ReCogDrive [^43] incorporates cognitive representations into a diffusion planner, while SGDrive [^19] learns structured spatiotemporal representations for trajectory prediction. World models, in contrast, learn predictive representations of future scene evolution from driving videos and provide dynamic priors for planning [^10] [^38] [^40] [^41] [^50]. World-model-based approaches generally follow cascaded or joint formulations. Cascaded methods, such as DriveLaW [^40], first predict future representations and then use them for trajectory planning. Joint methods instead model future dynamics and action generation within a unified process: SimWAM [^50] jointly trains video and action experts without explicitly feeding predicted future representations into action generation, while DriveVA [^29] models future video latents and actions within a shared diffusion framework.

Methods to integrate VLMs and world models. Recent works explore integrating semantic understanding from VLMs with future dynamics from world models for autonomous driving [^20] [^32] [^27]. These methods mainly follow two strategies: introducing complementary capabilities through auxiliary objectives or using representations from one paradigm to condition another. DriveVLA-W0 [^20] introduces future-image prediction as an auxiliary objective for VLA training, while DriveWAM [^32] and DriveWorld-VLA [^27] use semantic and dynamic representations, respectively, as conditioning information to augment their existing architectures. In contrast, DriveWLA jointly models World, Language, and Action Experts through Cross-Expert Attention, enabling continuous information exchange and representation alignment.

Reinforcement learning for autonomous driving. Behavior cloning learns driving policies from expert trajectories, whereas reinforcement learning further improves them using task-level driving rewards [^7] [^47] [^45] [^14]. RAD [^7] applies reinforcement learning to end-to-end driving models in realistic simulation environments. Recent works extend reinforcement learning to generative trajectory planners [^14] [^43] [^44]. ReCogDrive [^43] introduces DiffGRPO for diffusion-based trajectory planning, while EponaV2 [^44] adapts GRPO to flow-matching-based planners [^26]. DriveWLA further incorporates factorized advantages and variance-based shrinkage to preserve multi-dimensional driving reward signals during policy optimization.

## 3 Preliminaries

Problem Definition. Autonomous driving aims to generate a safe and smooth future trajectory based on the sensor input $\mathcal{I}_{\rm cam}$, historical trajectories $H_{\rm traj}$, navigation commands $\mathcal{C}_{\rm nav}$, and ego status $\mathcal{S}_{\rm ego}$ [^4] [^12]. For an end-to-end driving system $\Phi$, the trajectory generation process can be formulated as

$$
\mathrm{T_{traj}}=\Phi(\mathcal{I}_{\rm cam},H_{\rm traj},\mathcal{C}_{\rm nav},\mathcal{S}_{\rm ego}),
$$

where $\mathrm{T_{traj}}={(x_{t},y_{t},\theta_{t})}_{t=1}^{T}\in\mathbb{R}^{T\times 3}$ denotes the predicted ego trajectory over the next $T$ time steps, with $(x_{t},y_{t})$ representing the ego position and $\theta_{t}$ the heading angle in the current ego-centric coordinate frame.

## 4 Method

In this section, we present the overall architecture and training paradigm of DriveWLA, as shown in Fig. 1. First, we introduce three specialized experts: the World Expert for future dynamics, the Language Expert for semantic understanding, and the Action Expert for trajectory generation with flow matching. Second, we detail Cross-Expert Attention, which enables controlled information exchange and representation alignment among the three experts. Finally, we describe Flow-GRPO-based [^26] reinforcement learning with factorized advantage estimation for Action Expert refinement.

### 4.1 Model Architecture

World Expert. The World Expert is initialized from a video generation model pretrained on large-scale driving videos and consists of a spatiotemporal VAE, a text encoder, and an $N$ -block Diffusion Transformer (DiT) [^40] [^8] [^30]. Given the current and historical camera observations $\mathcal{I}_{\rm cam}$, the VAE encodes the observed frames into spatiotemporal latent $Z_{\rm obs}$, while the future-frame latent $Z_{\rm fut}$ is initialized from Gaussian noise:

$$
Z_{\rm obs}=E_{\rm VAE}(\mathcal{I}_{\rm cam}),\qquad Z_{\rm fut}\sim\mathcal{N}(\mathbf{0},\mathbf{I}),\qquad Z_{W}=[Z_{\rm obs};Z_{\rm fut}].
$$

Here, $\mathcal{N}(\mathbf{0},\mathbf{I})$ denotes a zero-mean Gaussian distribution with identity covariance, and $[Z_{\rm obs};Z_{\rm fut}]$ denotes concatenation of the observed and future latents along the temporal dimension. Conditioned on the encoded generation instruction, the World blocks produce layer-wise predictive representations of future scene evolution in a single forward pass, which are directly used for planning without iterative video sampling or future-frame decoding.

Language Expert and VLM Adapter. The Language Expert consists of a frozen driving-pretrained VLM and an $N$ -block VLM Adapter Transformer. We provide the VLM with camera observations $\mathcal{I}_{\rm cam}$ and a driving prompt $P$ constructed from ego-motion information, navigation commands, and human instructions. After processing the visual-textual inputs, we directly extract the final-layer hidden states of valid input tokens:

$$
H_{L}=\mathcal{F}_{\rm VLM}^{\rm last}(\mathcal{I}_{\rm cam},P),
$$

where $\mathcal{F}_{\rm VLM}^{\rm last}(\cdot)$ denotes the mapping from visual-textual inputs to the final-layer hidden representations of the VLM. We do not perform autoregressive decoding for textual outputs; instead, these hidden states are directly used as the semantic representations and further processed by the VLM Adapter to obtain planning-oriented features.

Action Expert. The Action Expert is a lightweight $N$ -block DiT [^30] that generates future trajectories with a flow-matching policy [^25]. Given a noisy trajectory state $a_{\tau}$, flow time $\tau$, and the driving condition $\mathcal{C}_{d}$ composed of ego information and navigation commands, the Action Expert predicts the velocity field $v_{\theta}(a_{\tau},\tau,\mathcal{C}_{d})$ for trajectory generation.

Training Objective. During imitation learning, the World Expert, VLM Adapter, and Action Expert are jointly optimized with the flow-matching behavior cloning objective. Specifically, let $a_{0}$ denote the expert trajectory and $a_{1}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ denote Gaussian noise. The interpolated trajectory state is defined as: $a_{\tau}=(1-\tau)a_{0}+\tau a_{1}$. The flow-matching behavior cloning objective is:

$$
\mathcal{L}_{\rm BC}=\mathbb{E}\left[\left\|v_{\theta}(a_{\tau},\tau,\mathcal{C}_{d})-(a_{1}-a_{0})\right\|_{2}^{2}\right].
$$

### 4.2 Cross-Expert Attention

![[attention_mask.png|Refer to caption]]

Figure 2: Query-Key/Value Visibility in Cross-Expert Attention.

Although the World and Language Experts provide complementary features, effectively integrating them remains challenging. Since their representations are shaped by different learning objectives, naive fusion may lead to representation conflicts. To address this issue, we introduce Cross-Expert Attention within each transformer block, which enables controlled information exchange and progressive alignment among the three experts.

Inspired by the mixture-of-transformers (MoT) architecture [^22], Cross-Expert Attention allows the three experts to exchange information while maintaining their own transformer blocks. The inputs of the three experts are first projected into a shared dimension $d$, forming the initial inputs $W^{0}$, $A^{0}$, and $L^{0}$. At block $\ell$, the hidden states of each expert $E^{\ell}$, where $E\in\{W,L,A\}$, are mapped to queries, keys, and values using their corresponding projections:

$$
Q_{E}^{\ell}=E^{\ell}W_{Q,E}^{\ell},\qquad K_{E}^{\ell}=E^{\ell}W_{K,E}^{\ell},\qquad V_{E}^{\ell}=E^{\ell}W_{V,E}^{\ell}.
$$

The query, key, and value matrices from the three experts are then concatenated as $Q^{\ell}$, $K^{\ell}$, $V^{\ell}$, and processed by a masked self-attention operation to enable cross-expert information exchange:

$$
O^{\ell}=\operatorname{softmax}\!\left(\frac{Q^{\ell}(K^{\ell})^{\top}}{\sqrt{d}}+M\right)V^{\ell}.
$$

Here, $M$ defines the visibility among the three experts, as illustrated in Fig. 2. The World and Language Experts have bidirectional visibility, allowing their representations to mutually exchange information. The Action Expert attends to both representations for trajectory generation, while its representations are prevented from flowing back to the World and Language Experts.

The attention outputs are then split back into the corresponding expert branches:

$$
O^{\ell}=[O_{W}^{\ell};O_{A}^{\ell};O_{L}^{\ell}].
$$

And each output is further processed by its expert-specific output projection, residual connection, normalization, and feed-forward transformation.

### 4.3 Flow Matching Group Relative Policy Optimization

After imitation learning, we further optimize the Action Expert with Flow-GRPO [^26] to improve trajectory generation under driving rewards. During reinforcement learning, stochastic exploration is introduced by adding SDE noise to the flow ODE, enabling diverse trajectory candidates to be sampled from the same scene condition. These candidates are evaluated by driving rewards, and group-relative optimization is performed based on their relative rewards. After training, the exploration noise is removed and the final policy generates trajectories by following the deterministic flow ODE.

However, the NAVSIM driving reward is multi-dimensional and consists of multiple metrics reflecting different aspects of driving performance. Directly aggregating these metrics into a single scalar reward may obscure individual driving factors and limit effective policy optimization.

To better utilize the optimization signals in driving rewards, we introduce a factorized advantage estimation strategy. Instead of computing the advantage directly from the aggregated PDMS score, we use the individual metric scores to construct a reward vector:

$$
\mathbf{r}_{i}=(r_{i,1},...,r_{i,M}),\qquad z_{i,m}=\frac{r_{i,m}-\mu_{m}}{\sigma_{m}+\epsilon},
$$

where $i$ indexes the sampled trajectory within a group, $m$ indexes the driving metric, and $\mu_{m},\sigma_{m}$ denote the group-wise mean and standard deviation of the corresponding metric. $\epsilon$ is a small constant for numerical stability.

The factorized advantage is computed by weighted aggregation of normalized metric scores, followed by multiplication with a variance-based shrinkage factor:

$$
A_{i}=\frac{\sum_{m}w_{m}z_{i,m}}{\sqrt{\operatorname{Var}(\sum_{m}w_{m}z_{i,m})+\epsilon}}\cdot\frac{s^{2}}{s^{2}+\tau^{2}},
$$

where $w_{m}$ denotes the weight of each driving metric, $s^{2}$ denotes the variance of PDMS rewards within the sampled group, and $\tau$ controls the shrinkage strength. This formulation preserves the information from individual driving metrics during advantage estimation while improving optimization stability by reducing unreliable updates from low-variance trajectory groups.

Using the proposed advantage, we optimize the LoRA [^11] parameters of the Action Expert with the Flow-GRPO objective augmented by a behavior cloning regularization:

$$
\mathcal{L}=-\frac{1}{K}\sum_{i=1}^{K}\frac{1}{T}\sum_{t=1}^{T}\gamma^{t-1}\log\pi_{\theta}(a_{t-1}^{i}|a_{t}^{i},\mathcal{C}_{d})A_{i}+\lambda_{\rm BC}\mathcal{L}_{\rm BC},
$$

where $K$ and $T$ denote the group size and the number of denoising steps, $\gamma$ is the discount factor, and $a_{t}^{i}$ represents the intermediate trajectory state at denoising step $t$. Additional details are provided in Appendix C.

## 5 Experiments

Table 1: Comparison of planners based on different paradigms on NAVSIM v1 navtest. Bold values denote the best result for each metric.

<table><tbody><tr><td>Method</td><td>VLM Backbone</td><td>Video Backbone</td><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Comf.<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td><em>VLM-based planners</em></td><td colspan="8"></td></tr><tr><td>ReCogDrive <sup><a href="#fn:43">43</a></sup></td><td>InternVL3-2B</td><td>–</td><td>97.9</td><td>97.3</td><td>94.9</td><td>100.0</td><td>87.3</td><td>90.8</td></tr><tr><td>SGDrive <sup><a href="#fn:19">19</a></sup></td><td>InternVL3-2B</td><td>–</td><td>98.6</td><td>97.8</td><td>96.2</td><td>100.0</td><td>85.8</td><td>91.1</td></tr><tr><td>Qwen-Drive-1.0 <sup><a href="#fn:52">52</a></sup></td><td>Qwen3.5-4B</td><td>–</td><td>98.6</td><td>98.2</td><td>95.9</td><td>100.0</td><td>84.8</td><td>90.7</td></tr><tr><td><em>World-Model-based planners</em></td><td colspan="8"></td></tr><tr><td>DriveLaW <sup><a href="#fn:40">40</a></sup></td><td>–</td><td>LTX-2B</td><td>99.0</td><td>97.1</td><td>96.7</td><td>100.0</td><td>81.3</td><td>89.1</td></tr><tr><td>ReWorld <sup><a href="#fn:41">41</a></sup></td><td>–</td><td>LTX-2B</td><td>99.1</td><td>98.2</td><td>97.7</td><td>99.8</td><td>82.0</td><td>90.4</td></tr><tr><td>SimWAM <sup><a href="#fn:50">50</a></sup></td><td>–</td><td>Wan2.2-5B</td><td>98.4</td><td>98.7</td><td>95.5</td><td>100.0</td><td>86.4</td><td>91.5</td></tr><tr><td><em>VLM–WM–based planners</em></td><td colspan="8"></td></tr><tr><td>DriveVLA-W0 <sup><a href="#fn:20">20</a></sup></td><td colspan="2">Emu3-8B (unified)</td><td>98.7</td><td>99.1</td><td>95.3</td><td>99.3</td><td>83.3</td><td>90.2</td></tr><tr><td>DriveWAM <sup><a href="#fn:32">32</a></sup></td><td>Qwen3-VL-8B</td><td>Wan2.2-5B</td><td>98.3</td><td>98.1</td><td>95.2</td><td>100.0</td><td>84.3</td><td>90.1</td></tr><tr><td>BrainWAM <sup><a href="#fn:46">46</a></sup></td><td>Qwen3-VL-4B</td><td>Wan2.2-5B</td><td>98.1</td><td>97.5</td><td>94.9</td><td>100.0</td><td>83.8</td><td>89.5</td></tr><tr><td>DriveWorld-VLA <sup><a href="#fn:27">27</a></sup></td><td>InternVL3-2B</td><td>ResNet-34</td><td>99.1</td><td>98.2</td><td>96.1</td><td>100.0</td><td>85.9</td><td>91.3</td></tr><tr><td>DriveWLA (Ours)</td><td>InternVL3-2B</td><td>LTX-2B</td><td>99.1</td><td>98.8</td><td>97.0</td><td>100.0</td><td>86.4</td><td>92.3</td></tr></tbody></table>

Table 2: Comparison on the NAVSIM v2 navtest split. All results use the corrected official evaluator with human-reference filtering. Bold values denote the best result for each metric.

| Method | NC $\uparrow$ | DAC $\uparrow$ | DDC $\uparrow$ | TLC $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | LK $\uparrow$ | HC $\uparrow$ | EC $\uparrow$ | EPDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DiffusionDrive [^23] | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | 84.5 |
| DiffusionDriveV2 [^56] | 97.7 | 96.6 | 99.2 | 99.8 | 88.9 | 97.2 | 96.0 | 97.8 | 91.0 | 87.5 |
| DriveWorld-VLA [^27] | 98.6 | 99.1 | 99.6 | 99.8 | 87.4 | 97.9 | 97.0 | 97.8 | 78.6 | 86.8 |
| Latent-WAM [^35] | 98.1 | 97.3 | 99.6 | 99.8 | 87.7 | 97.3 | 97.6 | 98.1 | 87.3 | 89.3 |
| CoWorld-VLA [^13] | 99.1 | 97.0 | 99.6 | 99.9 | 87.9 | 98.5 | 97.7 | 98.2 | 86.2 | 90.0 |
| DriveFuture [^9] | 98.8 | 99.1 | 99.6 | 99.9 | 86.6 | 98.4 | 96.4 | 98.3 | 74.8 | 89.9 |
| DriveWLA (Ours) | 98.8 | 97.8 | 99.7 | 99.9 | 87.4 | 98.8 | 97.8 | 98.4 | 82.5 | 90.1 |

### 5.1 Experimental Setup

Datasets and metrics. We train DriveWLA on 103,288 NAVSIM [^5] navtrain scenes and evaluate on 12,146 navtest scenes. NAVSIM v1 uses the Predictive Driver Model Score (PDMS), which aggregates no-at-fault collision (NC), drivable-area compliance (DAC), ego progress (EP), time to collision (TTC), and comfort [^5]. NAVSIM v2 [^2] extends PDMS to EPDMS with additional driving-direction compliance (DDC), traffic-light compliance (TLC), lane keeping (LK), history comfort (HC), and extended comfort (EC) [^2]. Detailed metric definitions are provided in Appendix B.

We further evaluate zero-shot transfer on closed-loop and open-loop benchmarks. For closed-loop evaluation, we test on HUGSIM [^51] and report HD-Score. For open-loop evaluation, we evaluate on nuScenes [^1] and Bench2Drive [^16], reporting trajectory L2 error and collision rate.

Implementation details. DriveWLA uses a driving-pretrained InternVL3-2B [^55] [^43] as the Language Expert and an LTX-Video-2B [^8] initialized World Expert from DriveLaW-Video [^40]. The Action Expert is a 28-block DiT [^30] that predicts eight $(x,y,\theta)$ waypoints over four seconds conditioned on historical ego poses, ego state, and navigation commands.

During imitation learning, the World DiT, Language Adapter, and Action DiT are jointly optimized for 30 epochs on 16 A100 GPUs with batch size 16 and learning rate $3\times 10^{-5}$. The VLM, video VAE, and text encoder are frozen. During reinforcement learning, Flow-GRPO [^26] optimizes Action-branch LoRA [^11] parameters using driving rewards. Unless stated otherwise, inference uses ten Euler steps. Further details are provided in Appendix A.

### 5.2 Main Results

NAVSIM v1. Tab. 1 shows that DriveWLA achieves the best PDMS of 92.3 across approaches from different paradigms. Specifically, it outperforms the best-performing VLM-based and world-model-based planners, SGDrive [^19] and SimWAM [^50], by 1.2 and 0.8 points, respectively, indicating that semantic understanding and future prediction provide complementary benefits for driving planning. Furthermore, DriveWLA surpasses existing VLM–world-model integrated planners, including DriveWAM [^32] and BrainWAM [^46], by 2.2 and 2.8 points, respectively. These improvements are achieved with a more compact expert configuration: DriveWLA uses InternVL3-2B [^55] as the Language Expert and LTX-Video-2B [^8] as the World Expert, while DriveWAM and BrainWAM adopt 8B/5B and 4B/5B language/video backbones. These results suggest that the advantage of DriveWLA comes from more effective unified World-Language-Action modeling rather than simply scaling up model capacity.

NAVSIM v2. Under the corrected official evaluator, DriveWLA achieves the highest EPDMS of 90.1 in Tab. 2. Importantly, it achieves the best DDC, TTC, LK, and HC scores, while tying for the best TLC. These results show that DriveWLA maintains strong performance across multiple evaluation dimensions, leading to the best overall EPDMS under the more comprehensive NAVSIM v2 evaluation.

Zero-shot on HUGSIM. Tab. 3 shows zero-shot closed-loop evaluation results on the HUGSIM [^51] benchmark. DriveWLA achieves the best results on 8 of the 11 reported metrics, including NC, TTC, PDMS, RC, and the average HD-Score. Notably, although the scoring-based DrivoR [^18] performs best on Easy scenarios, its advantage disappears as scenario difficulty increases, whereas DriveWLA achieves higher HD-Scores on Medium, Hard, and Extreme scenarios. These results demonstrate the strong zero-shot generalization and closed-loop driving capability of DriveWLA.

Table 3: Zero-shot closed-loop evaluation on HUGSIM. We report driving metrics over all 436 scenarios and HD-Score across different difficulty levels, together with the overall average.

<table><thead><tr><th rowspan="2">Method</th><th colspan="6">All 436 Scenarios</th><th colspan="5">HD-Score</th></tr><tr><th>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>Comf.<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>RC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>Easy <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>Medium <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>Hard <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>Extreme <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>Avg.<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th></tr></thead><tbody><tr><th>VAD <sup><a href="#fn:17">17</a></sup></th><td>41.20</td><td>90.30</td><td>28.00</td><td>95.30</td><td>28.30</td><td>30.10</td><td>42.00</td><td>8.50</td><td>7.70</td><td>6.30</td><td>13.90</td></tr><tr><th>UniAD <sup><a href="#fn:12">12</a></sup></th><td>65.60</td><td>93.20</td><td>51.60</td><td>66.30</td><td>49.40</td><td>43.80</td><td>64.00</td><td>37.20</td><td>21.00</td><td>6.30</td><td>31.20</td></tr><tr><th>LTF <sup><a href="#fn:3">3</a></sup> <sup><a href="#fn:51">51</a></sup></th><td>44.30</td><td>92.80</td><td>37.50</td><td>94.80</td><td>36.50</td><td>38.00</td><td>66.10</td><td>15.50</td><td>12.00</td><td>11.70</td><td>23.10</td></tr><tr><th>DrivoR <sup><a href="#fn:18">18</a></sup></th><td>52.20</td><td>95.60</td><td>46.20</td><td>93.90</td><td>44.80</td><td>47.20</td><td>78.00</td><td>29.10</td><td>20.00</td><td>14.10</td><td>32.50</td></tr><tr><th>DriveWLA (Ours)</th><td>65.80</td><td>95.06</td><td>59.27</td><td>75.78</td><td>54.92</td><td>54.99</td><td>73.87</td><td>47.91</td><td>31.41</td><td>15.87</td><td>41.47</td></tr></tbody></table>

Table 4: Zero-shot open-loop evaluation on nuScenes and Bench2Drive. We report L2 trajectory error (m) and collision rate (%) over 1–3 seconds and their averages. “Fine-tune” indicates target-dataset adaptation. Bold denotes the best result for each metric within each dataset.

<table><tbody><tr><th rowspan="2">Dataset</th><th rowspan="2">Method</th><td rowspan="2">Fine-tune</td><td colspan="4">L2 (m) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td colspan="4">Collision (%) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td></tr><tr><th rowspan="6">nuScenes</th><th>UniAD <sup><a href="#fn:12">12</a></sup></th><td>✓</td><td>0.48</td><td>0.96</td><td>1.65</td><td>1.03</td><td>0.05</td><td>0.17</td><td>0.71</td><td>0.31</td></tr><tr><th>OpenDriveVLA <sup><a href="#fn:53">53</a></sup></th><td>✓</td><td>0.19</td><td>0.58</td><td>1.24</td><td>0.67</td><td>0.02</td><td>0.18</td><td>0.70</td><td>0.30</td></tr><tr><th>UniDriveVLA-L <sup><a href="#fn:21">21</a></sup></th><td>✓</td><td>0.30</td><td>0.66</td><td>1.25</td><td>0.74</td><td>0.02</td><td>0.18</td><td>0.40</td><td>0.20</td></tr><tr><th>DriveVLA-W0 <sup><a href="#fn:20">20</a></sup></th><td>✗</td><td>0.43</td><td>1.26</td><td>2.60</td><td>1.43</td><td>0.22</td><td>0.66</td><td>1.42</td><td>0.77</td></tr><tr><th>PWM <sup><a href="#fn:49">49</a></sup></th><td>✗</td><td>2.06</td><td>3.91</td><td>6.00</td><td>3.99</td><td>0.12</td><td>0.15</td><td>0.86</td><td>0.36</td></tr><tr><th>DriveWLA (Ours)</th><td>✗</td><td>0.38</td><td>0.83</td><td>1.60</td><td>0.94</td><td>0.03</td><td>0.03</td><td>0.20</td><td>0.09</td></tr><tr><th rowspan="5">Bench2Drive</th><th>DriveVLA-W0 <sup><a href="#fn:20">20</a></sup></th><td>✗</td><td>1.01</td><td>2.77</td><td>5.22</td><td>3.00</td><td>1.49</td><td>2.53</td><td>3.53</td><td>2.52</td></tr><tr><th>PWM <sup><a href="#fn:49">49</a></sup></th><td>✗</td><td>1.70</td><td>2.74</td><td>3.97</td><td>2.80</td><td>4.01</td><td>3.73</td><td>3.53</td><td>3.76</td></tr><tr><th>DriveVA <sup><a href="#fn:29">29</a></sup></th><td>✗</td><td>0.69</td><td>1.29</td><td>2.03</td><td>1.33</td><td>1.38</td><td>1.97</td><td>2.65</td><td>1.79</td></tr><tr><th>UNIVERSE <sup><a href="#fn:28">28</a></sup></th><td>✗</td><td>0.67</td><td>1.29</td><td>2.03</td><td>1.33</td><td>0.75</td><td>1.12</td><td>1.70</td><td>1.19</td></tr><tr><th>DriveWLA (Ours)</th><td>✗</td><td>0.77</td><td>1.53</td><td>2.53</td><td>1.61</td><td>0.34</td><td>0.97</td><td>1.88</td><td>1.06</td></tr></tbody></table>

Zero-shot on nuScenes and Bench2Drive. As shown in Tab. 4, DriveWLA achieves strong zero-shot performance in open-loop evaluation. On nuScenes, DriveWLA outperforms all zero-shot methods listed in Tab. 4, achieving the lowest average L2 error of 0.94 m and collision rate of 0.09% among them. Compared with fine-tuned planners, DriveWLA achieves a substantially lower collision rate while maintaining a comparable L2 error. On Bench2Drive, DriveWLA continues to demonstrate strong performance, achieving the lowest collision rate of 1.06% while maintaining competitive L2 performance. These results demonstrate the strong zero-shot generalization and reliable driving performance of DriveWLA.

### 5.3 Ablation Studies

We conduct controlled ablations on NAVSIM v1 to assess expert composition, training strategies, attention mask, and inference-time sampling steps. Unless otherwise noted, each comparison follows the same training and evaluation protocol.

Table 5: Expert composition.

| Experts | Language | World | Action | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- |
| w/o World Expert | ✓ | ✗ | ✓ | 86.2 |
| w/o Language Expert | ✗ | ✓ | ✓ | 89.1 |
| Full DriveWLA | ✓ | ✓ | ✓ | 90.8 |

Table 6: Training strategy.

| Configuration | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf.$\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| Frozen World Expert | 98.5 | 96.1 | 96.4 | 100.0 | 80.4 | 88.0 |
| Trainable World Expert | 99.2 | 97.7 | 97.2 | 100.0 | 84.0 | 90.8 |
| $+$ Flow-GRPO | 99.1 | 98.7 | 97.1 | 100.0 | 86.1 | 92.1 |
| $+$ Flow-GRPO <sup><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\dagger"><semantics><mo>†</mo> <annotation>\dagger</annotation></semantics></math></sup> | 99.1 | 98.8 | 97.0 | 100.0 | 86.4 | 92.3 |

Table 7: Cross-Expert information flow.

| W $\rightarrow$ L | L $\rightarrow$ W | W / L $\rightarrow$ A | A $\rightarrow$ W / L | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- |
| ✓ | ✓ | ✓ | ✗ | 90.8 |
| ✓ | ✗ | ✓ | ✗ | 90.1 |
| ✗ | ✓ | ✓ | ✗ | 89.8 |
| ✓ | ✓ | ✓ | ✓ | 90.4 |
| ✗ | ✗ | ✓ | ✗ | 88.4 |

Table 8: Sampling steps.

| Steps | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf.$\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ | Latency $\downarrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 91.2 | 81.4 | 63.6 | 1.7 | 68.3 | 49.2 | 540 ms |
| 5 | 99.2 | 97.6 | 96.8 | 100.0 | 84.1 | 90.6 | 810 ms |
| 10 | 99.2 | 97.7 | 97.2 | 100.0 | 84.0 | 90.8 | 970 ms |
| 15 | 99.3 | 97.8 | 97.5 | 100.0 | 83.4 | 90.8 | 1160 ms |
| 20 | 99.3 | 97.8 | 97.6 | 100.0 | 83.3 | 90.7 | 1390 ms |

Expert Composition. To evaluate the individual contributions of semantic features from the VLM and future dynamic features from the World Model, we ablate the Language and World Experts separately. As shown in Tab. 6, removing the World Expert reduces PDMS from 90.8 to 86.2, while removing the Language Expert results in a smaller decrease to 89.1. The larger degradation caused by removing the World Expert indicates that predictive dynamics contribute more to planning performance in this setting. Furthermore, the performance degradation after removing each expert demonstrates that semantic and dynamic representations provide complementary benefits, and their joint utilization enables DriveWLA to achieve better performance than using either representation alone.

Training Strategy. As shown in Tab. 6, freezing the pretrained World Expert during imitation learning yields 88.0 PDMS. Jointly optimizing the World Expert raises PDMS to 90.8, suggesting that future dynamic representations learned from video prediction are not fully aligned with the requirements of driving. Furthermore, applying the original Flow-GRPO after imitation learning improves PDMS to 92.1, with particularly notable improvements in DAC and EP. Flow-GRPO <sup><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\dagger"><semantics><mo>†</mo> <annotation>\dagger</annotation></semantics></math></sup> denotes our improved variant with factorized advantage estimation and variance-based shrinkage in Eq. (9), which further improves PDMS to 92.3.

Cross-Expert Information Flow. To explore proper information flow among the three experts in our Cross-Expert Attention module, we design different attention masks to control the visibility between the World, Language, and Action Experts. Here, $X\rightarrow Y$ indicates that information can flow from Expert $X$ to Expert $Y$. As shown in Tab. 8, the bidirectional World–Language interaction with the Action Expert only receiving representations from both experts achieves the highest PDMS of 90.8. Removing either direction of World–Language information flow degrades performance, while removing both directions causes a further drop. These results demonstrate that bidirectional information exchange between semantic and dynamic representations is beneficial for leveraging their complementary capabilities. Moreover, allowing information to flow from the Action Expert to the World and Language Experts leads to performance degradation, indicating that introducing action representations into the World and Language Experts is not beneficial for planning.

Sampling Steps. With a single Euler step, DriveWLA achieves only 49.2 PDMS (Tab. 8). Increasing the sampling steps to five recovers most of the performance, achieving 90.6 PDMS, while ten steps further improve PDMS to the best result of 90.8 at 970 ms latency. Increasing the steps beyond ten provides no additional improvement at 15 steps and causes a slight performance drop at 20 steps, while introducing higher inference latency. We therefore adopt ten steps, where planning performance reaches its highest measured level before additional sampling increases latency without improving PDMS.

### 5.4 Qualitative Analysis

Fig. 3 shows qualitative comparisons on NAVSIM navtest, with predicted and expert trajectories visualized in bird’s-eye view (BEV). Fig. 3(a) compares DriveWLA with the expert ablations in two left-turn scenes. The full model produces valid trajectories within the drivable area in both scenes, whereas removing the World or Language Expert results in DAC failure in different cases. These contrasting failures further demonstrate the complementary roles of the World and Language Experts in trajectory planning. In Fig. 3(b), RL refinement corrects the off-road deviation of the imitation policy, improving DAC from 0 to 1. Notably, the refined policy produces a valid trajectory distinct from the expert demonstration, showing that reinforcement learning can move beyond direct trajectory imitation and optimize toward driving objectives.

![[expert_ablation_vis.png|Refer to caption]]

Figure 3: Qualitative visualization of expert complementarity and RL improvement. (a) Removing either the Language or World Expert leads to planning failures in different scenes, highlighting their complementary contributions. (b) RL refinement corrects the imitation policy and discovers a valid alternative trajectory distinct from the expert demonstration.

## 6 Conclusion

In this work, we proposed DriveWLA, a unified World-Language-Action framework for autonomous driving. Our results show that semantic understanding and future dynamics provide complementary information for trajectory planning, and the controlled cross-expert interaction can effectively integrate these capabilities. DriveWLA achieves state-of-the-art performance on NAVSIM and strong zero-shot generalization across open-loop and closed-loop benchmarks. We hope DriveWLA can serve as a strong baseline and that World-Language-Action modeling can become a promising paradigm for future autonomous driving research.

### AI use statement

We used large language models to assist with drafting and editing the manuscript for clarity and readability, and to help identify relevant prior work.

## References

## Appendix A Implementation Details

### A.1 Expert Configuration

World Expert. The World Expert is initialized from DriveLaW-Video [^40], which is built on LTX-Video-2B [^8]. Its video VAE encodes four front-view frames at $768\times 1344$ resolution, and a 28-block DiT with hidden width 2048 processes the resulting latents. The video VAE and text encoder remain frozen throughout training. The World DiT is optimized during imitation learning and frozen during Flow-GRPO [^26] refinement.

Language Expert. The Language Expert uses the driving-pretrained InternVL3-2B [^55] checkpoint from ReCogDrive [^43]. The frozen VLM processes the current front-view image using dynamic $448\times 448$ tiling. Its final-layer hidden states are passed to a 28-block VLM Adapter with hidden width 512. The adapter is optimized during imitation learning and frozen during Flow-GRPO refinement.

Action Expert. The Action Expert is a 28-block DiT with hidden width 512. Conditioned on four historical ego poses sampled at 2 Hz, the current ego state, and the navigation command, it generates eight $(x,y,\theta)$ waypoints at 0.5 s intervals over four seconds. The full DiT is optimized during imitation learning, while Flow-GRPO updates only Action-branch LoRA parameters with rank 16 and scaling factor 32. Inference uses ten deterministic Euler steps.

The World DiT, VLM Adapter, and Action DiT are jointly trained for 30 epochs with the flow-matching objective in Eq. (4). Training uses 16 A100 GPUs, a global batch size of 16, an initial learning rate of $3\times 10^{-5}$, and cosine learning-rate decay. This trajectory objective also updates the World DiT and VLM Adapter through Cross-Expert Attention; no auxiliary video-prediction or text-generation loss is used during imitation learning.

### A.2 Cross-Expert Attention Mask

In Eq. (6), rows of $M$ index query tokens and columns index key tokens. Writing the concatenated queries and keys in the branch order $(W,A,L)$, the block-wise mask is

$$
M=\begin{pmatrix}0&-\infty&0\\
0&0&0\\
0&-\infty&0\end{pmatrix},
$$

where zero permits attention and $-\infty$ blocks it. Each entry applies to all token pairs in the corresponding branches at every Cross-Expert Attention block. The arrows in Tab. 8 denote information flow from the key/value branch to the query branch; for example, $W/L\rightarrow A$ means Action queries can attend to World and Language keys.

### A.3 Trajectory Sampling

At inference, the World Expert processes $Z_{\rm obs}$ and the Gaussian-initialized $Z_{\rm fut}$ from Eq. (2) in one forward pass. Its hidden states condition the Action Expert through Cross-Expert Attention without decoding future RGB frames. Starting from $a_{\tau_{0}}=a_{1}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$, the Action Expert integrates the learned velocity field from $\tau_{0}=1$ to $\tau_{K}=0$ using $K=10$ deterministic Euler steps:

$$
a_{\tau_{k+1}}=a_{\tau_{k}}+(\tau_{k+1}-\tau_{k})v_{\theta}(a_{\tau_{k}},\tau_{k},\mathcal{C}_{d}),\qquad k=0,\ldots,K-1.
$$

For fixed World future latents, driving inputs, and initial Action noise, this Euler trajectory is deterministic.

## Appendix B Evaluation Protocols

### B.1 NAVSIM v1

On NAVSIM v1 [^5], DriveWLA is trained on 103,288 navtrain scenes and evaluated on 12,146 navtest scenes. The Predictive Driver Model Score (PDMS) applies multiplicative penalties for no-at-fault collisions (NC) and drivable-area compliance (DAC) to a weighted combination of ego progress (EP), time-to-collision compliance (TTC), and comfort (C):

$$
\mathrm{PDMS}=\mathrm{NC}\times\mathrm{DAC}\times\left(\frac{5\,\mathrm{EP}+5\,\mathrm{TTC}+2\,\mathrm{C}}{12}\right).
$$

NC takes values in $\{0,0.5,1\}$, assigning a smaller penalty to collisions with static objects than to collisions with road users. DAC records whether the ego footprint remains in the drivable area. EP measures route progress relative to the reference planner, while TTC and C assess collision risk and kinematic comfort along the rollout.

In Tab. 1, the ResNet-34 entry for DriveWorld-VLA [^27] denotes its BEV encoder for latent world modeling, rather than a video-generation backbone.

### B.2 NAVSIM v2

NAVSIM v2 [^2] extends the score with driving-direction compliance (DDC) and traffic-light compliance (TLC) as multiplicative terms. Its weighted terms retain EP and TTC, replace C with history comfort (HC), and add lane keeping (LK) and extended comfort (EC). The per-scene Extended PDMS (EPDMS) is

$$
\mathrm{EPDMS}=\mathrm{NC}\times\mathrm{DAC}\times\mathrm{DDC}\times\mathrm{TLC}\times\left(\frac{5\,\mathrm{EP}+5\,\mathrm{TTC}+2\,\mathrm{LK}+2\,\mathrm{HC}+2\,\mathrm{EC}}{16}\right).
$$

DDC penalizes travel against the lane direction, and TLC detects red-light violations. LK measures lane adherence, HC evaluates the continuity of planned motion with ego-motion history, and EC assesses consistency across successive predictions. Under the corrected evaluator, NC, DAC, DDC, TLC, EP, TTC, LK, and HC are subject to human-reference filtering. A component is assigned a value of one when the human reference scores zero on that component; otherwise, the planner’s value is retained. EC is computed from paired predictions and is not filtered separately.

### B.3 HUGSIM

HUGSIM [^51] measures closed-loop performance as the policy receives updated rendered observations throughout each rollout. We evaluate DriveWLA without HUGSIM-specific training or fine-tuning on all 436 scenarios from nuScenes [^1], KITTI-360 [^24], Waymo [^33], and PandaSet [^42], comprising 88, 113, 108, and 127 scenarios, respectively. We use the corrected HUGSIM evaluator [^37]. The reported NC, DAC, TTC, comfort, PDMS, and route-completion (RC) scores are averaged over all scenarios. HUGSIM PDMS summarizes rollout safety and comfort, while HD-Score additionally accounts for route completion.

For HD-Score, scenarios are grouped into Easy, Medium, Hard, and Extreme levels with 80, 157, 96, and 103 scenarios. Within each level, scores are averaged within each source dataset and then across the four datasets. The overall HD-Score weights the four difficulty-level scores by their scenario counts [^37].

The VAD, UniAD, LTF, and DrivoR entries in Tab. 3 are taken from the 436-scenario reevaluation in WA-JEPA [^37] and rescaled from $[0,1]$ to $[0,100]$. DriveWLA uses the same scenarios, driving commands, corrected controller, metric implementation, and aggregation. Camera inputs remain method-specific: DriveWLA uses one front-view camera, LTF uses three, and VAD, UniAD, and DrivoR use four.

### B.4 nuScenes

DriveWLA is evaluated on recorded nuScenes [^1] observations without further planner training on this dataset. L2 error is the Euclidean distance between predicted and logged ego positions. Collision rate is the percentage of predicted trajectories that collide with other road users under the UniAD protocol [^12]. We report both metrics at horizons of 1, 2, and 3 seconds and average the three values before rounding.

The UniAD [^12] row in Tab. 4 follows its original report. The OpenDriveVLA [^53] and UniDriveVLA-L [^21] rows are the UniAD-metric results of their 3B and Large variants, respectively. The DriveVLA-W0 [^20] and PWM [^49] zero-shot rows are taken from the cross-dataset comparison in DriveVA [^29]. DriveWLA uses the full 150-scene validation split and the UniAD L2 and collision definitions. UniAD, OpenDriveVLA, and UniDriveVLA-L use nuScenes training data, whereas DriveVLA-W0, PWM, and DriveWLA do not; the reported UniAD row also lacks ego-state input used by OpenDriveVLA, UniDriveVLA-L, and DriveWLA.

### B.5 Bench2Drive

On Bench2Drive [^16], we evaluate DriveWLA using recorded observations rather than interactive rollouts, without further planner training. L2 error and collision rate are reported at horizons of 1, 2, and 3 seconds, with averages calculated from the three values before rounding.

The DriveVLA-W0, PWM, and DriveVA baseline results in Tab. 4 are taken from the zero-shot comparison in DriveVA [^29]; the UNIVERSE results are taken from UNIVERSE [^28]. DriveWLA uses the same Bench2Drive validation samples and L2/collision calculation code. All listed methods are trained on NAVSIM and evaluated without Bench2Drive fine-tuning. These are open-loop trajectory metrics from recorded observations, not the benchmark’s interactive closed-loop Driving Score.

## Appendix C Flow-GRPO Training Details

Flow-GRPO refinement uses scenes for which the imitation policy achieves a PDMS below 90. For each scene, eight trajectories are sampled through stochastic flow transitions [^26]. The NC, DAC, EP, TTC, and C rewards are standardized within each group and combined in the factorized advantage defined in Eq. (9), with weights $w_{\rm NC}=w_{\rm DAC}=2$, $w_{\rm EP}=w_{\rm TTC}=5$, and $w_{\rm C}=2$. The variance-shrinkage parameter is $\tau=0.1$. Both variances in the factorized advantage are computed across the eight trajectories sampled for the same scene. Here, $\tau$ is a fixed shrinkage parameter, distinct from the flow-time variable used in trajectory sampling.

Training runs for eight epochs with ten stochastic flow steps, step discount $\gamma=0.8$, and behavior-cloning coefficient $\lambda_{\rm BC}=0.1$ (Eq. (10)). Only Action-branch LoRA parameters (rank 16, scaling factor 32) are updated; the World and Language Experts and the base Action weights remain frozen. Inference removes the exploration noise and uses the deterministic Euler sampler in Eq. (12).

The expert-composition, attention-mask, and sampling-step ablations in Tabs. 6, 8, and 8 use the imitation-trained checkpoint before Flow-GRPO refinement. The “Trainable World Expert” row in Tab. 6 reports the same 90.8-PDMS checkpoint; the Flow-GRPO rows report subsequent reinforcement-learning refinement.

![[representation_pca_vis.png|Refer to caption]]

Figure 4: World Expert latent features. Each row shows one driving scene. The columns show the front-view image, history latents 1 and 4, and future latents 1 and 2. All PCA-projected latent maps use the same color scale.

## Appendix D World-Expert Feature Visualization

Fig. 4 shows PCA projections of the World Expert’s history and future features in three driving scenes. The four observed history frames are encoded separately, yielding four history latents; history latents 1 and 4 therefore correspond to the first and fourth observed frames. The eight future frames are grouped into two consecutive sets of four and encoded groupwise, yielding two future latents. Future latents 1 and 2 correspond to these two groups, respectively, and each summarizes four frames spanning 2 seconds. Thus, the two future latents together represent the full 4-second prediction horizon. Across scenes, the latents exhibit structured, scene-specific patterns that evolve across time. This temporal organization provides qualitative evidence that the unified architecture learns effective representations of both observed context and future dynamics, which supply structured predictive context to the Action Expert.

[^1]: H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom Nuscenes: a multimodal dataset for autonomous driving. In 2020 IEEE/CVF conference on computer vision and pattern recognition (CVPR), pp. 11618–11628. Cited by: §B.3, §B.4, §1, §5.1.

[^2]: W. Cao, M. Hallgarten, T. Li, D. Dauner, X. Gu, C. Wang, Y. Miron, M. Aiello, H. Li, I. Gilitschenski, et al. Pseudo-simulation for autonomous driving. arXiv preprint arXiv:2506.04218. Cited by: §B.2, §5.1.

[^3]: K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger Transfuser: imitation with transformer-based sensor fusion for autonomous driving. IEEE transactions on pattern analysis and machine intelligence 45 (11), pp. 12878–12895. Cited by: Table 3.

[^4]: F. Codevilla, M. Müller, A. López, V. Koltun, and A. Dosovitskiy End-to-end driving via conditional imitation learning. In 2018 IEEE international conference on robotics and automation (ICRA), pp. 4693–4700. Cited by: §3.

[^5]: D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, et al. Navsim: data-driven non-reactive autonomous vehicle simulation and benchmarking. Advances in Neural Information Processing Systems 37, pp. 28706–28719. Cited by: §B.1, §1, §5.1.

[^6]: J. Engström, J. Bärgman, D. Nilsson, B. Seppelt, G. Markkula, G. B. Piccinini, and T. Victor Great expectations: a predictive processing account of automobile driving. Theoretical issues in ergonomics science 19 (2), pp. 156–194. Cited by: §1.

[^7]: H. Gao, S. Chen, B. Jiang, B. Liao, Y. Shi, X. Guo, Y. Pu, X. Li, W. Liu, Q. Zhang, et al. Rad: training an end-to-end driving policy via large-scale 3dgs-based reinforcement learning. Advances in Neural Information Processing Systems 38, pp. 32551–32576. Cited by: §2.

[^8]: Y. HaCohen, N. Chiprut, B. Brazowski, D. Shalem, D. Moshe, E. Richardson, E. Levin, G. Shiran, N. Zabari, O. Gordon, et al. Ltx-video: realtime video latent diffusion. arXiv preprint arXiv:2501.00103. Cited by: §A.1, §4.1, §5.1, §5.2.

[^9]: Y. Hong, X. Zhou, Y. Li, X. Zhou, L. Liu, Y. Luo, S. Xu, L. Yang, and Z. Song DriveFuture: future-aware latent world models for autonomous driving. arXiv preprint arXiv:2605.09701. Cited by: Table 2.

[^10]: A. Hu, L. Russell, H. Yeo, Z. Murez, G. Fedoseev, A. Kendall, J. Shotton, and G. Corrado Gaia-1: a generative world model for autonomous driving. arXiv preprint arXiv:2309.17080. Cited by: §1, §1, §2.

[^11]: E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen Lora: low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685. Cited by: §4.3, §5.1.

[^12]: Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, et al. Planning-oriented autonomous driving. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 17853–17862. Cited by: §B.4, §B.4, §1, §3, Table 3, Table 4.

[^13]: M. Huang, Y. Xiang, Z. Liang, J. Huang, J. Wang, Z. Xu, F. Tan, H. Zhou, M. Yang, and G. Che Coworld-vla: thinking in a multi-expert world model for autonomous driving. arXiv preprint arXiv:2605.10426. Cited by: Table 2.

[^14]: Z. Huang, X. Weng, M. Igl, Y. Chen, Y. Cao, B. Ivanovic, M. Pavone, and C. Lv Gen-drive: enhancing diffusion generative driving policies with reward modeling and reinforcement learning fine-tuning. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pp. 3445–3451. Cited by: §2.

[^15]: J. Hwang, R. Xu, H. Lin, W. Hung, J. Ji, K. Choi, D. Huang, T. He, P. Covington, B. Sapp, et al. Emma: end-to-end multimodal model for autonomous driving. arXiv preprint arXiv:2410.23262. Cited by: §2.

[^16]: X. Jia, Z. Yang, Q. Li, Z. Zhang, and J. Yan Bench2drive: towards multi-ability benchmarking of closed-loop end-to-end autonomous driving. Advances in Neural Information Processing Systems 37, pp. 819–844. Cited by: §B.5, §1, §5.1.

[^17]: B. Jiang, S. Chen, Q. Xu, B. Liao, J. Chen, H. Zhou, Q. Zhang, W. Liu, C. Huang, and X. Wang Vad: vectorized scene representation for efficient autonomous driving. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 8306–8316. Cited by: Table 3.

[^18]: E. Kirby, A. Boulch, Y. Xu, Y. Yin, G. Puy, É. Zablocki, A. Bursuc, S. Gidaris, R. Marlet, F. Bartoccioni, et al. Driving on registers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 32058–32069. Cited by: §5.2, Table 3.

[^19]: J. Li, J. Wu, D. Hu, X. Huang, B. Sun, Z. Hao, X. Lang, X. Zhu, and L. Zhang Sgdrive: scene-to-goal hierarchical world cognition for autonomous driving. arXiv preprint arXiv:2601.05640. Cited by: §1, §2, §5.2, Table 1.

[^20]: Y. Li, S. Shang, W. Liu, B. Zhan, H. Wang, Y. Wang, Y. Chen, X. Wang, Y. An, C. Tang, et al. Drivevla-w0: world models amplify data scaling law in autonomous driving. In International Conference on Learning Representations, Vol. 2026, pp. 7890–7911. Cited by: §B.4, §1, §2, Table 1, Table 4, Table 4.

[^21]: Y. Li, L. Zhou, S. Yan, B. Liao, T. Yan, K. Xiong, L. Chen, H. Xie, B. Wang, G. Chen, et al. Unidrivevla: unifying understanding, perception, and action planning for autonomous driving. arXiv preprint arXiv:2604.02190. Cited by: §B.4, Table 4.

[^22]: W. Liang, L. Yu, L. Luo, S. Iyer, N. Dong, C. Zhou, G. Ghosh, M. Lewis, W. Yih, L. Zettlemoyer, et al. Mixture-of-transformers: a sparse and scalable architecture for multi-modal foundation models. arXiv preprint arXiv:2411.04996. Cited by: §4.2.

[^23]: B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang, et al. Diffusiondrive: truncated diffusion model for end-to-end autonomous driving. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 12037–12047. Cited by: Table 2.

[^24]: Y. Liao, J. Xie, and A. Geiger Kitti-360: a novel dataset and benchmarks for urban scene understanding in 2d and 3d. IEEE Transactions on Pattern Analysis and Machine Intelligence 45 (3), pp. 3292–3310. Cited by: §B.3.

[^25]: Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le Flow matching for generative modeling. arXiv preprint arXiv:2210.02747. Cited by: §4.1.

[^26]: J. Liu, G. Liu, J. Liang, Y. Li, J. Liu, X. Wang, P. Wan, D. Zhang, and W. Ouyang Flow-grpo: training flow matching models via online rl. Advances in neural information processing systems 38, pp. 40783–40818. Cited by: §A.1, Appendix C, §1, §2, §4.3, §4, §5.1.

[^27]: L. Liu, Z. Song, C. Jia, H. Ye, X. Hao, L. Chen, et al. Driveworld-vla: unified latent-space world modeling with vision-language-action for autonomous driving. arXiv preprint arXiv:2602.06521. Cited by: §B.1, §2, Table 1, Table 2.

[^28]: M. Liu, D. Zhang, J. Liu, J. Cui, H. Xie, G. Chen, H. Ye, F. Nex, H. Cheng, and M. Y. Yang UNIVERSE: unified video action models for autonomous driving with flexible mask-modulated modality generation. arXiv preprint arXiv:2607.05133. Cited by: §B.5, Table 4.

[^29]: M. Liu, D. Zhang, J. Liu, J. Cui, H. Xie, G. Chen, H. Ye, M. Y. Yang, F. Nex, and H. Cheng Driveva: video action models are zero-shot drivers. In European Conference on Computer Vision, pp. 315–335. Cited by: §B.4, §B.5, §2, Table 4.

[^30]: W. Peebles and S. Xie Scalable diffusion models with transformers. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 4172–4182. Cited by: §4.1, §4.1, §5.1.

[^31]: H. Shao, Y. Hu, L. Wang, G. Song, S. L. Waslander, Y. Liu, and H. Li Lmdrive: closed-loop end-to-end driving with large language models. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 15120–15130. Cited by: §1.

[^32]: C. Shi, J. Xu, S. Shi, K. Sheng, B. Zhang, and L. Jiang DriveWAM: video generative priors enable scalable world-action modeling for autonomous driving. arXiv preprint arXiv:2605.28544. Cited by: §1, §2, §5.2, Table 1.

[^33]: P. Sun, H. Kretzschmar, X. Dotiwalla, A. Chouard, V. Patnaik, P. Tsui, J. Guo, Y. Zhou, Y. Chai, B. Caine, et al. Scalability in perception for autonomous driving: waymo open dataset. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2443–2451. Cited by: §B.3.

[^34]: X. Tian, J. Gu, B. Li, Y. Liu, Y. Wang, Z. Zhao, K. Zhan, P. Jia, X. Lang, and H. Zhao Drivevlm: the convergence of autonomous driving and large vision-language models. arXiv preprint arXiv:2402.12289. Cited by: §2.

[^35]: L. Wang, Y. Zheng, Q. Chen, S. Li, Y. Zhang, Z. Xing, Q. Zhang, X. Li, D. Qian, P. Yang, et al. Latent-wam: latent world action modeling for end-to-end autonomous driving. arXiv preprint arXiv:2603.24581. Cited by: Table 2.

[^36]: X. Wang, Z. Zhu, G. Huang, X. Chen, J. Zhu, and J. Lu Drivedreamer: towards real-world-drive world models for autonomous driving. In European conference on computer vision, pp. 55–72. Cited by: §1.

[^37]: X. Wang, Y. Xiang, Y. Zhou, J. Wang, M. Huang, J. Huang, D. Wei, T. Zhou, X. Wang, G. Chen, et al. WA-jepa: rethinking the video jepa paradigm for world-action modeling in autonomous driving. arXiv preprint arXiv:2608.20974. Cited by: §B.3, §B.3, §B.3.

[^38]: Y. Wang, J. He, L. Fan, H. Li, Y. Chen, and Z. Zhang Driving into the future: multiview visual forecasting and planning with world model for autonomous driving. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 14749–14759. Cited by: §1, §2.

[^39]: B. Wolfe, L. Fridman, A. Kosovicheva, B. Seppelt, B. Mehler, B. Reimer, and R. Rosenholtz Predicting road scenes from brief views of driving video. Journal of vision 19 (5), pp. 8–8. Cited by: §1.

[^40]: T. Xia, Y. Li, L. Zhou, J. Yao, K. Xiong, H. Sun, B. Wang, K. Ma, G. Chen, H. Ye, et al. Drivelaw: unifying planning and video generation in a latent driving world. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 39701–39712. Cited by: §A.1, §2, §4.1, §5.1, Table 1.

[^41]: T. Xia, L. Zhou, K. Xiong, J. Yao, Y. Zhu, Z. Zhu, B. Wang, G. Chen, H. Ye, W. Liu, et al. ReWorld: learning better representations for world action models. arXiv preprint arXiv:2606.27504. Cited by: §2, Table 1.

[^42]: P. Xiao, Z. Shao, S. Hao, Z. Zhang, X. Chai, J. Jiao, Z. Li, J. Wu, K. Sun, K. Jiang, et al. Pandaset: advanced sensor suite dataset for autonomous driving. In 2021 IEEE international intelligent transportation systems conference (ITSC), pp. 3095–3101. Cited by: §B.3.

[^43]: K. Xiong, X. Guo, F. Li, S. Yan, G. Xu, L. Zhou, L. Chen, H. Sun, B. Wang, K. Ma, et al. Recogdrive: a reinforced cognitive framework for end-to-end autonomous driving. In International Conference on Learning Representations, Vol. 2026, pp. 157518–157556. Cited by: §A.1, §1, §2, §2, §5.1, Table 1.

[^44]: J. Xu, Z. Zhong, Z. Shu, M. Jia, M. Li, J. Bian, Q. Zhang, K. Zhang, J. Xie, J. Yang, et al. EponaV2: driving world model with comprehensive future reasoning. arXiv preprint arXiv:2605.14696. Cited by: §2.

[^45]: Z. Yang, X. Jia, Q. Li, X. Yang, M. Yao, and J. Yan Raw2drive: reinforcement learning with aligned world models for end-to-end autonomous driving (in carla v2). Advances in Neural Information Processing Systems 38, pp. 134122–134147. Cited by: §2.

[^46]: B. Zhan, S. Shang, J. Gu, S. Lu, Y. Xu, Z. Wang, Y. Wang, X. Zhang, K. Zhan, L. Fan, et al. BrainWAM: action-space coordination of semantic priors and predictive dynamics for autonomous driving. arXiv preprint arXiv:2608.12854. Cited by: §1, §5.2, Table 1.

[^47]: D. Zhang, J. Liang, K. Guo, S. Lu, Q. Wang, R. Xiong, Z. Miao, and Y. Wang Carplanner: consistent auto-regressive trajectory planning for large-scale reinforcement learning in autonomous driving. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 17239–17248. Cited by: §2.

[^48]: G. Zhao, X. Wang, Z. Zhu, X. Chen, G. Huang, X. Bao, and X. Wang Drivedreamer-2: llm-enhanced world models for diverse driving video generation. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 39, pp. 10412–10420. Cited by: §1.

[^49]: Z. Zhao, T. Fu, Y. Wang, L. Wang, and H. Lu From forecasting to planning: policy world model for collaborative state-action prediction. Advances in Neural Information Processing Systems 38, pp. 134585–134611. Cited by: §B.4, Table 4, Table 4.

[^50]: Z. Zhao, X. Zhou, T. Xu, Z. Sun, K. Zhou, H. Li, D. Liang, and X. Bai SimWAM: a simple world action model for end-to-end autonomous driving. arXiv preprint arXiv:2608.07468. Cited by: §2, §5.2, Table 1.

[^51]: H. Zhou, L. Lin, J. Wang, Y. Lu, D. Bai, B. Liu, Y. Wang, A. Geiger, and Y. Liao Hugsim: a real-time, photo-realistic and closed-loop simulator for autonomous driving. IEEE Transactions on Pattern Analysis and Machine Intelligence. Cited by: §B.3, §1, §5.1, §5.2, Table 3.

[^52]: X. Zhou, Z. Zhao, Z. Yang, M. Li, H. Zhong, S. Bai, D. Chu, R. Chen, Z. Li, J. Tang, et al. Qwen-drive-1.0: an initial step towards a vision-language foundation model for autonomous driving. arXiv preprint arXiv:2609.00111. Cited by: Table 1.

[^53]: X. Zhou, X. Han, F. Yang, Y. Ma, V. Tresp, and A. Knoll Opendrivevla: towards end-to-end autonomous driving with large vision language action model. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40, pp. 13782–13790. Cited by: §B.4, §2, Table 4.

[^54]: Z. Zhou, T. Cai, S. Zhao, Y. Zhang, Z. Huang, B. Zhou, and J. Ma Autovla: a vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. Advances in Neural Information Processing Systems 38, pp. 27920–27956. Cited by: §1, §2.

[^55]: J. Zhu, W. Wang, Z. Chen, Z. Liu, S. Ye, L. Gu, H. Tian, Y. Duan, W. Su, J. Shao, et al. Internvl3: exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479. Cited by: §A.1, §1, §5.1, §5.2.

[^56]: J. Zou, S. Chen, B. Liao, Z. Zheng, Y. Song, L. Zhang, Q. Zhang, W. Liu, and X. Wang Diffusiondrivev2: reinforcement learning-constrained truncated diffusion modeling in end-to-end autonomous driving. arXiv preprint arXiv:2512.07745. Cited by: Table 2.
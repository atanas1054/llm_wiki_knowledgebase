---
title: "DriveFine: Refining-Augmented Masked Diffusion VLA for Precise and Robust Driving"
source: "https://arxiv.org/html/2602.14577v1"
author:
published:
created: 2026-04-05
description:
tags:
  - "clippings"
---
<sup>1</sup> <sup>2</sup> <sup>3</sup>

Chenxu Dang Completed during the internship at Xiaomi EV and AIR. <sup><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\dagger"><semantics><mo>†</mo> <annotation>\dagger</annotation></semantics></math></sup> Corresponding authors.    Sining Ang    Yongkang Li    Haochen Tian    Jie Wang    Guang Li    Hangjun Ye    Jie Ma    Long Chen <sup><sup>†</sup></sup>    Yan Wang <sup><sup>†</sup></sup>

###### Abstract

Vision-Language-Action (VLA) models for autonomous driving increasingly adopt generative planners trained with imitation learning followed by reinforcement learning. Diffusion-based planners suffer from modality alignment difficulties, low training efficiency, and limited generalization. Token-based planners are plagued by cumulative causal errors and irreversible decoding. In summary, the two dominant paradigms exhibit complementary strengths and weaknesses. In this paper, we propose DriveFine, a masked diffusion VLA model that combines flexible decoding with self-correction capabilities. In particular, we design a novel plug-and-play block-MoE, which seamlessly injects a refinement expert on top of the generation expert. By enabling explicit expert selection during inference and gradient blocking during training, the two experts are fully decoupled, preserving the foundational capabilities and generic patterns of the pretrained weights, which highlights the flexibility and extensibility of the block-MoE design. Furthermore, we design a hybrid reinforcement learning strategy that encourages effective exploration of refinement expert while maintaining training stability. Extensive experiments on NAVSIM v1, v2, and Navhard benchmarks demonstrate that DriveFine exhibits strong efficacy and robustness. The code will be released at [https://github.com/MSunDYY/DriveFine](https://github.com/MSunDYY/DriveFine).

## 1 Introduction

Vision-Language-Action (VLA) systems for autonomous driving (AD) integrate sensor observations and textual instructions, with a planner responsible for generating driving actions or trajectories. Early deterministic planners (e.g., MLP-based \[uniad, vad, sparsead\] or anchor-based classifiers \[vadv2, hydra-mdp\]) tended to imitate a single expert trajectory, which struggle with distribution shift and cannot capture the inherently multi-modal nature of driving.

Recently, non-deterministic generative planners have emerged as the dominant paradigms in AD VLAs. They predict actions as probability distributions, effectively capturing the multimodal nature of driving behavior. Moreover, their inherent sampling capability encourages active exploration and enables seamless integration with rule-driven reinforcement learning strategies, such as GRPO, to guide policy learning. Current state-of-the-art generative VLAs can be categorized into diffusion-based models \[recogdrive, sgdrive, diffusiondrive, diffusiondrivev2\] with continuous action modeling and token-based planners \[opendrivevla, autovla, adathinkdrive\] with discrete action representations.

![[x1 10.png|Refer to caption]]

Figure 1: Comparison of decoding mechanisms for Action Tokens in Generative VLA Models. (a) Parallel refinement with multi-step denoising. (b) Token-by-token decoding. (c) Generate first in parallel, then refine.

![[x2 8.png|Refer to caption]]

Figure 2: PDMS-oriented RFT.

(1) Diffusion-based VLAs, as shown in Fig. 1(a) construct a Markov chain and iteratively refine noisy trajectories by predicting their mean and variance.

Despite the efficiency enabled by parallel decoding, the additional diffusion Transformer hinders cross-modal alignment, leading to inefficient training that typically requires hundreds of epochs.

Moreover, diffusion-based planners are inherently conditional generators, which limits their robustness and generalization ability, as evidenced empirically in Fig. 2: when optimized with PDMS-oriented reinforcement fine-tuning, diffusion-based planners \[recogdrive, diffusiondrive\] suffer a significant drop in EPDMS. Here, both PDMS and EPDMS are metrics provided by NAVSIM \[navsim\], see 4.1.1 for details. We attribute this degradation to the weak coupling between diffusion planners and the VLM, which induces reward hacking and the loss of pretrained knowledge, substantially limiting their practical applicability.

(2) Token-based VLAs, as illustrated in Fig. 1(b), decode actions into tokens autoregressively within a predefined vocabulary, achieving a unified representation across vision, language, and action. As shown in Fig. 2, the PDMS-oriented RFT for InternVL \[internvl\] leads to simultaneous improvements in both PDMS and EPDMS, exhibiting its stronger generalization and extensibility.

![[x3 8.png|Refer to caption]]

Figure 3: Failure cases caused by irreversible decoding in token-based VLAs.

However, token-based VLAs \[adathinkdrive, autovla\] generally lag behind diffusion-based counterparts \[recogdrive, diffusiondrivev2\] in both performance and efficiency. This is mainly due to their causal attention and fixed token-by-token decoding, which is computationally costly and prone to error accumulation during inference. More critically, they inherit the irreversible decoding property of LLMs: decoded tokens cannot be modified once committed. Planning, however, is highly sensitive to noise: even point-level deviations can cause the entire trajectory failure, such as collision or off-road driving (Fig. 3).

Very recent works \[reflectdrive, wam-diff\] explored masked diffusion LLMs \[llada, lavida\] (dLLMs) featuring more flexible decoding orders for driving. However, this flexibility aggravates the irreversible decoding problems: as shown in Fig. 3, tokens decoded early lack global consistency constraints, are more likely to become outliers, and cannot be revised afterward, leading to trajectory-level failure. In contrast, diffusion-based planners iteratively refine the trajectory, enabling successive refinement and thereby ensuring high-quality trajectory generation.

Clearly, both VLA planners exhibit complementary strengths and weaknesses, motivating the exploration of a model harnessing the advantages of both.

In this paper, we propose DriveFine, pioneering the explicit injection of token-VLA’s refining capability for more precise and robust driving. We adopt a pretrained multi-modal masked Diffusion LLM (LaViDa \[lavida\] with LLaDA \[llada\] as LLM) as our base planner, considering its several benefits compared with AR LLMs: parallel decoding for efficiency, bidirectional attention for richer context modeling, and a flexible decoding strategy that facilitates adaptive learning.

The refining of token-VLAs is far from trivial and must adhere to several principles: preserving the base VLM’s original training and inference paradigms to prevent collapse of foundational capabilities; minimizing extra overhead in computation and parameters; remaining decoupled from trajectory generation to avoid interference. Evidently, this presents substantial challenges.

To this end, we design a block-wise Mixture-of-Experts (MoE) architecture. Specifically, the majority of LLaDA blocks serve as shared experts for common contexts, while the remaining blocks are explicitly partitioned into generation experts and refinement experts. During inference, task-specific experts are proactively selected. During training, gradient flow from the refining branch is strictly confined to the refinement experts, decoupling from the generation experts. This explicit isolation preserves the foundational capabilities of the generation experts, effectively preventing mode collapse and cross-task interference.

To align with the dominant pipeline, we further introduce an online-offline hybrid reinforcement learning paradigm. The generation expert samples a group of trajectories, which are reinforced via GRPO. Simultaneously, these trajectories are paired into offline anchor-target trajectory tuples. In parallel, the refinement expert actively refines the above trajectories, computing the associated rewards and generating online trajectory pairs, which, together with offline pairs, jointly supervise the refinement expert. Experimental results demonstrate that the block-wise MOE significantly enhances trajectory quality with limited parameter increase and slight inference overhead, raising the performance ceiling of token-based VLAs. We summarize our core contributions as follows:

- We provide a thorough analysis of the strengths and weaknesses of mainstream diffusion-based and token-based VLAs.
- We propose DriveFine, featuring a plug-and-play block-wise Mixture-of-Experts (block-MoE) that injects refining capability into token-based VLAs at minimal cost.
- We design a targeted hybrid reinforcement learning strategy to further elevate DriveFine’s performance ceiling.
- Extensive experiments demonstrate that DriveFine consistently achieves state-of-the-art performance on NavSim v1, v2, and Navhard benchmarks.

## 2 Related Work

### 2.1 End to end Autonomous Driving

Early end-to-end planners predominantly relied on deterministic modeling, such as MLP-based regression \[uniad, vad, sparsedrive, transfuser\] and anchor-based classification \[vadv2\], imitating a single expert trajectory. GenAD \[genad\] imposes a GRU-based generator. To align the trajectory diversity, diffusion policies were introduced to recover trajectories from random noise, thereby better aligning with the inherent uncertainty of autonomous driving. Diffusion planners \[diffusionplanner, flowplanner\] leverage diffusion policies to jointly perform trajectory prediction and planning. GoalFlow \[goalflow\] adopts goal-conditioned flow matching to generate trajectories, while DiffusionDrive \[diffusiondrive\] applies truncated diffusion over multiple anchor modes.

### 2.2 VLAs for Autonomous Driving

Conventional end-to-end driving models are often regarded as black boxes. To enhance interpretability, understanding, and reasoning, vision–language models (VLMs) \[internvl, qwenvl\] have been increasingly incorporated into autonomous driving systems in recent years. Early explorations primarily focused on high-level scene understanding \[omnidrive\] and reasoning \[RAD-Driver, reason2drive, drivecot\], while the demand for action generation gave rise to a large body of Vision–Language–Action (VLA) models. Early two-stage VLAs \[gptdriver, emma, solve, drivevlm\] generate meta-actions or low-frequency actions from a VLM, followed by an e2e planner for refinement. However, gradient isolation in such pipelines contradicts the principle of end-to-end learning. In contrast, recent one-stage VLAs directly output the final trajectory. For example, OpenDriveVLA \[opendrivevla\] autoregressively decodes trajectories, while AutoVLA \[autovla\] augments the action vocabulary with clustered anchors. These approaches feature token-based decoding and are thus referred to as token-based VLAs. Other approaches incorporate a planner on top of the VLM to facilitate cross-modal alignment. Orion \[orion\] employs a GRU-based decoder, while recent works \[recogdrive, sgdrive, sparseoccvla\] increasingly adopt diffusion policies as planners, achieving stronger performance.

### 2.3 Reinforcement Learning for Autonomous Driving

Imitation learning lacks negative supervision from counterexamples, which leads to poor performance in out-of-distribution (OOD) scenarios. To address this limitation, many studies introduce reinforcement learning to improve the generalization of planners. Early works adopt DPO, while AlphaDrive \[alphadrive\] is the first to introduce GRPO \[deepseekmath\]. AdaThinkDrive \[adathinkdrive\] and AutoVLA \[autovla\] directly apply GRPO to token-based VLAs, whereas ReCogDrive \[recogdrive\] designs a diff-GRPO. DiffusionDrivev2 \[diffusiondrivev2\] further proposes intra- and inter-anchor truncated GRPO strategies. More recently, another scoring-based reinforcement learning models \[drivor, hydra-mdp, gtrs\] performs stage-wise reasoning under direct reward guidance, leading to stronger performance.

## 3 Method

In this section, we present a detailed introduction of DriveFine with the overview in Fig. 4. Sec. 3.1 describes the formulation of trajectory planning with the pretrained dLLM. Sec 3.2 details the proposed Block-MoE for refinement. Sec 3.3 illustrates the hybrid reinforcement learning strategy.

![[x4 7.png|Refer to caption]]

Figure 4: Architecture Overview of DriveFine. Visual and textual inputs are jointly aligned into a unified language space. A set of masked tokens undergoes s steps of parallel denoising followed by a single refinement step. The key difference between the generation expert and the refinement expert lies in their input tokens: the former decodes only the masked tokens, whereas the latter operates on unmasked tokens.

### 3.1 Diffusion LLM for Planning

#### 3.1.1 Modatily Tokenization.

As shown in Fig. 4, a single front-view image is processed by a pretrained vision tower (SigLIP \[siglip\]) to produce continuous visual tokens, while a text tokenizer converts textual prompts info discrete tokens.

To preserve the continuity of trajectory and enable token-level refinement, we discretize the action space following prior works \[reflectdrive, wam-diff\]. The spatial range $[-100m,+100m]$ is uniformly divided into 4,000 bins with a resolution of 0.05 m, shared across longitudinal and lateral axes. The heading angle range $[-90\textdegree,+90\textdegree]$ is discretized into 1800 bins with a resolution of $0.1\textdegree$. These bins are appended to the LLM vocabulary, enabling direct decoding of trajectory tokens and facilitating unified cross-modal alignment between language and actions.

#### 3.1.2 Training and Inference.

The generating expert follows the standard dLLM training and inference paradigm. During training, clean sequences are corrupted by random masking, where tokens are replaced by a special mask token \[M\] with probability $t$, and supervised via masked cross-entropy loss:

$$
\mathcal{L}_{\theta}=-\mathbb{E}_{t,p_{0},r_{0},r_{t}}\left[\frac{1}{t}\sum_{i=1}^{L}\mathbb{I}\!\left[r_{t}^{i}=[\text{M}]\right]\log p_{\theta}\!\left(r_{0}^{i}\mid p_{0},r_{t}\right)\right]
$$

Here, $L$, $p_{0}$, $r_{0}$ and $r_{t}$ denote the sequence length, contextual token, original sequence, and the noised sequence, respectively.

During the inference stage, the policy model is encouraged to condition on the given inputs (visions and instructions) and progressively reconstruct a feasible trajectory from fully masked trajectory via several iterative unmasking steps. Specifically, at denoising step $t$, all masked tokens of noised trajectory $r_{t}$ are predicted in parallel by a mask predictor (generation expert). A subset of mask tokens is then decoded to sequence $r_{t-1}$ for the next iteration.

### 3.2 Block-level MoE for Refinement

As our previous analysis indicates, the flexible token decoding exacerbates the hazards of irreversible decoding, highlighting the necessity of refinement.

The most straightforward approach is to introduce additional MoE layers to enable adaptive learning, and apply a loss directly on the unmasked tokens for supervision. However, this departs from the standard pre-training and inference paradigm of dLLMs, lossing their foundational capabilities, as they learn only to decode the mask tokens. Moreover, the deep coupling between generation and refinement introduces mutual interference and hinders task-specific tuning and optimization. Yet, fully decoupling the components would inevitably result in a dramatic increase in parameters.

However, despite their different objectives, generation and refinement share a high similarity in contextual representation. Motivated by this insight, we propose our block-level Mixture-of-Experts (block-MoE) together with a carefully designed training–inference pipeline.

As illustrated in Fig. 4, a diffusion language model (LLaDA) consists of several stacked blocks. We treat the pretrained dLLM as a generation expert in its entirety, and just replicate its last $n$ blocks as refinement blocks, while the preceding blocks and visual tower are shared between both generation and refinement experts.

#### 3.2.1 Training and Inference.

During inference, the model is explicitly prompted to perform specific task. The shared blocks extract common contextual representations, following which the corresponding expert blocks are manually activated to execute either generation or refinement.

During training, the generating branch computes the loss only on masked tokens as in Eq. 1, while the refinement branch computes the loss over all tokens, with gradient flow confined to the refinement expert. Here, the refinement expert undergoes simply warm-started for basic decoding.

Clearly, our block-MoE achieves complete decoupling between generation and refinement during training and inference. This preserves the foundational knowledge of the pretrained model, preventing catastrophic forgetting. Moreover, the refinement expert is plug-and-play and can be trained synchronously with the generation experts, emphasizing its flexibility and transferability.

### 3.3 Reinforcement Finetuning

Recent studies have demonstrated the critical role of reinforcement learning in autonomous driving both theoretically and empirically. In the following, we detail how reinforcement fine-tuning (RFT) is leveraged to explore and enhance the full potential of DriveFine.

![[x5 7.png|Refer to caption]]

Figure 5: Reinforcement Fine-tuning Pipeline. The offline advantage and online advantage are jointly combined to form the hybrid advantage to supervise the training of the refinement expert.

#### 3.3.1 GRPO for Generation Expert.

For the generation expert, we employ a rule-based online reinforcement learning strategy, Group Relative Policy Optimization (GRPO). Specifically, given an arbitrary scenario $p$, the generation expert parallelly samples a group of $G$ candidate trajectories $\{x_{i}\}_{i=1}^{G}$. Following \[dllm-rl\], we progressively sample for $s$ steps to ensure consistency between training and inference, and aggregate every $\tau$ neighboring steps to balance alignment and efficiency. For each trajectory $x_{i}$, we retain $\left\lfloor\frac{s}{\tau}\right\rfloor$ sampled paths $\{x_{i}^{j}\}_{j=1}^{\left\lfloor s/\tau\right\rfloor}$. The associated rewards $\{r_{i}\}_{i=1}^{G}$ are then evaluated in a simulator. Subsequently, without normalization as in \[d1\], the group-relative rewards are computed as follows:

$$
\hat{A}_{i}=r_{i}-\text{mean}(\{r_{i}\}_{i=1}^{G})
$$

$x_{1}^{\left\lfloor s/\tau\right\rfloor}$ Subsequently, the sampled tokens are processed sequentially, and their losses are computed using the standard GRPO objective:

$$
\small\begin{split}\mathcal{L}_{\mathrm{GRPO}}(\theta)&=\mathbb{E}_{\begin{subarray}{c}q\sim\mathcal{D}\\
o_{i}\sim\pi_{\theta}(\cdot\mid q)\end{subarray}}\Bigg[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_{i}|}\sum_{k=1}^{|o_{i}|}\min\Big(\rho_{i,k}^{j}\,\hat{A}_{i}^{k},\;\mathrm{clip}\big(\rho_{i,k}^{j},\,1-\epsilon,\,1+\epsilon\big)\,\hat{A}_{i}^{k}\Big)\\
&\qquad-\beta\,D_{\mathrm{KL}}\Big(\pi_{\theta}(\cdot\mid q)\;\|\;\pi_{\mathrm{ref}}(\cdot\mid q)\Big)\Bigg],\end{split}
$$

where $o_{i}$ denotes the length of the sequence, $\epsilon$ controls the clipping range, $\beta$ balances the KL divergence penalty, and:

$$
\rho_{i,k}^{j}=\left.\frac{\pi_{\theta}\!\left(o_{i,k}^{j}\mid q,x_{i}^{j}\right)}{\pi_{\theta_{\mathrm{old}}}\!\left(o_{i,k}^{j}\mid q,x_{i}^{j}\right)}\;\right|\;x_{i,k}^{j}=[\mathrm{M}],\;x_{i,k}^{j+1}\neq[\mathrm{M}].
$$

#### 3.3.2 Hybrid RL for Refining Expert.

The purpose of the optimization expert is to fine-tune the generated trajectories to improve their quality. Compared to the anchor trajectories, corrective actions that lead to score improvements should be encouraged, while those that degrade the score should be penalized, regardless of the anchor itself. Therefore, the generated trajectories naturally serve as a reference for advantage computation, eliminating the need for traditional baseline estimation in reinforcement learning (e.g., the value network in PPO or the group-average reward in GRPO). Notably, the sampling of trajectories ensures diversity, allowing them to be directly used for training the optimization expert without further modification. Specifically,for sampled trajectories $\{x_{i}\}_{i=1}^{G}$, we compute pairwise reward differences to obtain a relative advantage matrix. Since the sampled trajectories are produced by the generation expert, they constitute offline data for the refining expert; hence, we define this as an offline reward matrix $\hat{\textbf{A}}^{\text{of}}\in\mathbb{R}^{G\times G}$:

$$
\hat{A}^{\text{of}}_{ij}=r_{i}-r_{j},\quad\forall i,j\in G
$$

The offline advantage matrix offers several benefits: (1) it has zero mean, simultaneously encouraging improvements and penalizing degradations; (2) the squared advantages provide denser reward signals compared to GRPO, enhancing training stability; and (3) it requires no additional sampling, making the computation simple and efficient.

Despite its many advantages, the optimizer is inherently limited by an upper bound. To encourage the optimizer to explore autonomously, for each generated trajectory $x_{i}$, we allow it to sample a number of $K$ refined trajectories $\{\hat{x}_{ik}\}_{k=1}^{K}$ online and compute their corresponding rewards $\{\hat{r}_{ik}\}_{k=1}^{K}$ simultaneously. The online reward matrix $\hat{\mathbf{A}}^{on}\in\mathbb{R}^{G\times K}$ is then calculated as follows:

$$
\hat{A}^{\text{on}}_{ik}=\hat{r}_{ik}-r_{i},\quad\forall i\in G,k\in K
$$

Finally, a hybrid loss is computed to optimize the refining expert:

$$
\small\begin{split}\mathcal{L}_{\mathrm{hybrid}}(\theta)=&\mathbb{E}_{\begin{subarray}{c}q\sim\mathcal{D}\\
o_{i}\sim\pi_{\theta}(\cdot\mid q)\end{subarray}}\frac{1}{G}\sum_{i=1}^{G}\Bigg[\frac{1}{G}\sum_{j=1}^{G}\frac{1}{|o_{ij}|}\sum_{k=1}^{|o_{ij}|}\ \rho_{i,k}\,\hat{A}_{of}^{i,j}\;\\
&+\frac{1}{N}\sum_{k=1}^{N}\frac{1}{|o_{ik}|}\sum_{k=1}^{|o_{i}|}\ \rho_{i,k}\,\hat{A}_{on}^{i,j},\;\Bigg],\end{split}
$$

In our implementation, the generator and refiner are trained synchronously: trajectories are sampled online by the generator and then fed to the refiner for training, thereby enabling their collaborative learning and improvement.

## 4 Experiments

### 4.1 Experimental Settings

Table 1: PDMS results on the NAVSIM-v1 benchmark. <sup>†</sup> denotes best-of-6 sampling, and \* indicates the appliance of score-based reinforcement training as \[drivor, gtrs\].

<table><tbody><tr><td>Method</td><td>Sensors</td><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>C.<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>Human</td><td>–</td><td>100</td><td>100</td><td>100</td><td>99.9</td><td>87.5</td><td>94.8</td></tr><tr><td colspan="8">End-to-End Methods</td></tr><tr><td>UniAD <cite>[uniad]</cite></td><td>6xC</td><td>97.8</td><td>91.9</td><td>92.9</td><td>100.0</td><td>78.8</td><td>83.4</td></tr><tr><td>TransFuser <cite>[transfuser]</cite></td><td>3xC+L</td><td>97.7</td><td>92.8</td><td>92.8</td><td>100.0</td><td>79.2</td><td>84.0</td></tr><tr><td>LAW <cite>[law]</cite></td><td>1xC</td><td>96.4</td><td>95.4</td><td>88.7</td><td>99.9</td><td>81.7</td><td>84.6</td></tr><tr><td>Hydra-MDP <cite>[hydra-mdp]</cite></td><td>3xC+L</td><td>98.3</td><td>96.0</td><td>94.6</td><td>100.0</td><td>78.7</td><td>86.5</td></tr><tr><td>DiffusionDrive <cite>[diffusiondrive]</cite></td><td>3xC+L</td><td>98.2</td><td>96.2</td><td>94.7</td><td>100.0</td><td>82.2</td><td>88.1</td></tr><tr><td>WoTE <cite>[wote]</cite></td><td>3xC+L</td><td>98.5</td><td>96.8</td><td>94.4</td><td>99.9</td><td>81.9</td><td>88.3</td></tr><tr><td colspan="8">Vision Language Action Methods</td></tr><tr><td>AutoVLA <cite>[autovla]</cite></td><td>3xC</td><td>98.4</td><td>95.6</td><td>98.0</td><td>99.9</td><td>81.9</td><td>89.1</td></tr><tr><td>DriveVLA-W0 <cite>[drivevla-w0]</cite></td><td>1xC</td><td>98.7</td><td>99.1</td><td>95.3</td><td>99.3</td><td>83.3</td><td>90.2</td></tr><tr><td>AdaThinkDrive <cite>[adathinkdrive]</cite></td><td>1xC</td><td>98.4</td><td>97.8</td><td>95.2</td><td>100</td><td>84.4</td><td>90.3</td></tr><tr><td>ReCogDrive <cite>[recogdrive]</cite></td><td>1xC</td><td>98.2</td><td>97.8</td><td>95.2</td><td>99.8</td><td>83.5</td><td>89.6</td></tr><tr><td>DriveFine</td><td>1xC</td><td>98.6</td><td>97.9</td><td>95.2</td><td>99.9</td><td>85.5</td><td>90.7</td></tr><tr><td>DriveFine*</td><td>1xC</td><td>98.8</td><td>98.6</td><td>96.2</td><td>100</td><td>86.9</td><td>91.8</td></tr><tr><td>AutoVLA <sup>†</sup> <cite>[autovla]</cite></td><td>3xC</td><td>99.1</td><td>97.1</td><td>97.1</td><td>100.0</td><td>87.6</td><td>92.1</td></tr><tr><td>DriveVLA-W0 <sup>†</sup></td><td>1xC</td><td>99.3</td><td>97.4</td><td>97.0</td><td>99.9</td><td>88.3</td><td>93.0</td></tr><tr><td>AdaThinkDrive <sup>†</sup> <cite>[adathinkdrive]</cite></td><td>1xC</td><td>99.1</td><td>98.8</td><td>97.2</td><td>100</td><td>87.9</td><td>93.0</td></tr><tr><td>DriveFine <sup>†</sup></td><td>1xC</td><td>99.3</td><td>99.2</td><td>97.9</td><td>100</td><td>89.1</td><td>94.2</td></tr></tbody></table>

#### 4.1.1 Dataset.

We evaluate DriveFine on the NAVSIM \[navsim\] dataset, which is built upon nuPlan \[nuplan\] (a subset of OpenScene \[openscene\]) and provides surround-view images from 8 cameras along with high-quality LiDAR point clouds. The dataset is split into 1,192 (navtrain) scenes for training and 136 scenes (navtest) for testing. NAVSIM also offers a simulation environment for closed-loop evaluation.

NAVSIM v1 adopts the Predictive Driver Model Score (PDMS) as the evaluation metric, which is a weighted aggregation of multiple driving-related criteria, including collision avoidance, drivable area compliance, progress–risk trade-off, and comfort. Building upon this metric, NAVSIM v2 introduces the extended PDM Score (EPDMS), which incorporates additional weighted factors such as traffic light compliance, lane boundary adherence, driving direction following, and extended comfort.

#### 4.1.2 Implementation Details.

We adopt Siglip-384 \[siglip\] as the visual tower, which partitions a single front-view image into 8 patches of size $384\times 384$. Pretrained weights from LLaDA-8B \[llada\] are loaded directly, with the first 28 transformer blocks serving as shared blocks and the last 4 blocks designated as expert ones. DriveFine is trained in two stages. In the first stage, it performs supervised fine-tuning (SFT) using the QA pairs and textualized trajectories provided by ReCogDrive \[recogdrive\], without any additional pretraining. In the second stage, it undergoes reinforcement fine-tuning (RFT) in the NAVSIM simulation environment.

During the SFT stage, the model is trained for 12 epochs with a batch size of 64, optimized using AdamW with a learning rate of $4\times 10^{-5}$ and cosine learning rate decay.

In the RFT stage, The group size for generating expert rollouts is set to 10, each of trajectory is further optimized online by the optimizer 6 times. The model is trained for 1 epoch with a batch size of 16 and a learning rate of $1\times 10^{-6}$.

At inference, DriveFine executes 12 sampling steps followed by a single refining step, adhering to a confidence-prioritized and cosine schedule for decoding.

Table 2: EPDMS results on the NAVSIM-v2 benchmark. <sup>†</sup> indicates results evaluated on the bug-fixed version of NAVSIM.

| Method | NC $\uparrow$ | DAC $\uparrow$ | DDC $\uparrow$ | TLC $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | LK $\uparrow$ | C.$\uparrow$ | EC $\uparrow$ | EPDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ego-MLP \[egomlp\] | 93.1 | 77.9 | 92.7 | 99.6 | 86.0 | 91.5 | 89.4 | 98.3 | 85.4 | 64.0 |
| TransFuser \[transfuser\] | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 |
| DriveSuprim \[drivesuprim\] | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 | 83.1 |
| ARTEMIS \[artemis\] | 98.3 | 95.1 | 98.6 | 99.8 | 81.5 | 97.4 | 96.5 | 98.3 | – | 83.1 |
| ReCogDrive \[recogdrive\] | 98.3 | 95.2 | 99.5 | 99.8 | 87.1 | 97.5 | 96.6 | 98.3 | 86.5 | 83.6 |
| DiffusionDrive \[diffusiondrive\] | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | 84.5 |
| DriveVLA-W0 \[drivevla-w0\] | 99.0 | 98.4 | 99.3 | 99.9 | 87.0 | 98.1 | 93.2 | 97.9 | 58.9 | 86.5 |
| DriveFine | 98.7 | 97.3 | 98.8 | 99.8 | 88.2 | 97.8 | 97.7 | 98.4 | 84.7 | 87.1 |
| DriveFine <sup>†</sup> | 98.7 | 97.3 | 99.5 | 99.8 | 88.7 | 97.8 | 97.7 | 98.4 | 83.8 | 89.7 |

Table 3: EPDMS results on the Navhard benchmark. \* indicates values are copied from \[simscale\]; all other results are obtained from our own evaluations.

<table><tbody><tr><td>Method</td><td>Stage</td><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DDC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TLC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>LK <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>HC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EPDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td rowspan="2">ReCogDrive <cite>[recogdrive]</cite></td><td>S1</td><td>97.1</td><td>80.2</td><td>98.4</td><td>100</td><td>83.6</td><td>95.3</td><td>93.6</td><td>97.6</td><td>76.0</td><td>68.9</td></tr><tr><td>S2</td><td>78.2</td><td>69.3</td><td>83.9</td><td>98.2</td><td>86.6</td><td>73.9</td><td>44.1</td><td>96.4</td><td>72.0</td><td>37.8</td></tr><tr><td rowspan="2">DiffusionDrive* <cite>[diffusiondrive]</cite></td><td>S1</td><td>96.8</td><td>86.0</td><td>98.8</td><td>99.3</td><td>84.0</td><td>95.8</td><td>96.7</td><td>97.6</td><td>79.6</td><td>66.7</td></tr><tr><td>S2</td><td>80.1</td><td>72.8</td><td>84.4</td><td>98.4</td><td>85.9</td><td>76.6</td><td>46.4</td><td>96.3</td><td>72.8</td><td>40.5</td></tr><tr><td rowspan="2">DriveFine(Ours)</td><td>S1</td><td>97.6</td><td>90.0</td><td>99.1</td><td>99.3</td><td>84.9</td><td>96.7</td><td>97.3</td><td>97.6</td><td>72.0</td><td>74.4</td></tr><tr><td>S2</td><td>82.1</td><td>71.3</td><td>84.8</td><td>98.4</td><td>88.1</td><td>74.3</td><td>47.2</td><td>96.8</td><td>72.8</td><td>41.0</td></tr></tbody></table>

### 4.2 Main Results

#### 4.2.1 Results of Navtest.

We first report the NAVSIM v1 results, evaluating PDMS on the navtest split, as shown in Tab. 1. DriveFine achieves state-of-the-art (SOTA) performance. In particular, compared with autoregressive token-based VLA models \[autovla, adathinkdrive\], DriveFine achieves a 0.5% improvement, while being on par with diffusion-based planners \[recogdrive\]. When score-based reinforcement fine-tuning is applied (an additional scorer is trained), DriveFine further improves its performance to 91.9 PDMS, surpassing all existing VLA planners.

We report the comparison results on NAVSIM-v2 in Tab. 2, which provides a more comprehensive evaluation of the overall model performance. For a fair comparison with prior works, we evaluate DriveFine using an earlier NAVSIM version, which contains a score computation bug that leads to systematically underestimated overall scores. Despite this issue, DriveFine still achieves 86.7 EPDMS, outperforming DriveVLA-W0 \[drivevla-w0\] by 0.5 points. When evaluated with the bug-fixed NAVSIM version, DriveFine further reaches 89.7 EPDMS, thereby establishing a new state of the art.

#### 4.2.2 Results of Navhard.

We further conduct experiments on the more challenging Navhard benchmark, which employs Gaussian splatting to generate scenarios beyond the training data distribution and adopts a two-stage evaluation protocol. The results are reported in Tab. 3. Notably, no additional training is applied in this setting. Under this fair evaluation protocol, DriveFine outperforms the diffusion-based methods \[diffusiondrive, reason2drive\] on both stage metrics, with a particularly notable improvement of 5.5 points in Stage 1 EPDMS over ReCogDrive, further demonstrating its strong performance as well as the superior generalization capability of token-based VLAs.

### 4.3 Ablation Studies

#### 4.3.1 Ablations of the core components.

Tab. 4 presents an ablation study on the core components of DriveFine. With LLaDA mimicking expert trajectories (SFT), the model achieves a PDMS of 86.7. GRPO reinforcement training increases PDMS to 89.6. Incorporating the refinement expert with offline reinforcement training further improves performance by 0.7 point, while online reinforcement training raises the model’s performance ceiling to 90.8.

We observe that the refinement mechanism improves nearly all metrics, particularly DAC and Conf. Further visualization of trajectories before and after refinement (as shown in the Fig. 6) clearly demonstrates that the refinement expert can effectively correct anomalies when individual tokens lead to collisions or off-road events, which would otherwise result in complete trajectory failure. Additionally, it significantly mitigates fluctuations caused by noncausal decoding, enhancing overall trajectory smoothness.

![[x6 6.png|Refer to caption]]

Figure 6: Qualitative visualization comparison between DriveFine and other SOTA methods. In the rightmost figure, the red and green lines denote the trajectories before and after refinement, respectively.

Table 4: Ablation results of core components of DriveFine.

| ID | SFT | GRPO | Offline-RFT | Online-RFT | NC | DAC | TTC | Conf. | EP | PDMS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ✓ | ✗ | ✗ | ✗ | 98.0 | 95.2 | 94.9 | 99.4 | 81.6 | 86.7 |
| 2 | ✓ | ✓ | ✗ | ✗ | 98.3 | 96.9 | 95.1 | 99.2 | 85.1 | 89.6+2.9 |
| 3 | ✓ | ✓ | ✓ | ✗ | 98.5 | 97.3 | 95.2 | 99.9 | 85.4 | 90.3+0.7 |
| 4 | ✓ | ✓ | ✓ | ✓ | 98.6 | 97.9 | 95.2 | 99.9 | 85.5 | 90.8+0.5 |

#### 4.3.2 Ablation study on the robustness of DriveFine.

We further evaluated the robustness of DriveFine. As shown in Tab. 6, after PDMS-oriented RFT, extended metrics such as LK, EC, and DDC remain stable or show slight improvements. Consequently, the increase in PDMS is accompanied by a synchronous improvement in EPMDS. Compared with diffusion-based models in 7, DriveFine demonstrates a clear robustness advantage, further confirming the potential of token-based models.

Table 5: Performance comparison of different models.

| Model | EP $\uparrow$ | LK $\uparrow$ | EC $\uparrow$ | PDMS $\uparrow$ | EPMDS $\uparrow$ |
| --- | --- | --- | --- | --- | --- |
| DriveFine-SFT | 86.8 | 97.9 | 83.3 | 86.7 | 86.8 |
| +RFT(PDMS) | 88.7 | 97.7 | 83.8 | 90.5+3.8 | 89.1+2.3 |

Table 6: Sensitivity the number of refinement blocks $n$.

| n | 0 | 1 | 2 | 4 | 6 | 8 |
| --- | --- | --- | --- | --- | --- | --- |
| Param(B) | 8 | 8.25 | 8.5 | 9 | 9.5 | 10 |
| PDMS | 89.6 | 90.0 | 90.4 | 90.8 | 90.8 | 90.7 |

#### 4.3.3 Sensitivity analysis of the number of refinement blocks.

We evaluated the sensitivity of DriveFine to the number of refinement blocks, as summarized in Tab. 6. When $n=0$, i.e., without any refinement capability, the trajectories are derived from the generative expert. Remarkably, introducing just one refinement block (250M parameters) already improves trajectory quality (+0.4 PDMS). As the number of blocks increases, performance gradually improves, reaching the optimal level when 4 refinement blocks are injected.

#### 4.3.4 Efficiency-Performance Trade-off Analysis.

We evaluate the impact of the number of diffusion steps $s$ on the efficiency–performance trade-off, comparing DriveFine with VLA models of similar scale, as shown in Fig. 7. As expected, inference latency increases roughly proportionally with $s$. In general, a larger number of diffusion steps allows the model to “think” more thoroughly, resulting in better performance.

![[x7 6.png|Refer to caption]]

Figure 7: PDMS-Latency trade-off

However, we observe that with only 4 steps, DriveFine already achieves a PDMS of 90.47, comparable to ReCogDrive-8B \[recogdrive\], with an average latency of merely 207 ms, representing an optimal efficiency–performance balance. This further demonstrates the robustness of masked diffusion LLMs. In contrast, the InternVL model relies on token-by-token decoding, incurring much higher inference latency. Overall, these results indicate that the flexible decoding of diffusion-based language models enables an effective balance between efficiency and performance, paving the way for more efficient planning paradigms.

#### 4.3.5 Effect of Group Size.

As shown in Tab. 7, DriveFine is moderately sensitive to the GRPO group size. Even with a small group size $G=2$, it outperforms the SFT baseline by 1.7 PDMS. Increasing $G$ leads to a monotonic performance gain, with the best performance achieved at $G=8$.

Table 7: Sensitivity of group size $G$.

| Group size $G$ | 0 | 2 | 4 | 6 | 8 | 10 |
| --- | --- | --- | --- | --- | --- | --- |
| EP $\uparrow$ | 81.6 | 83.7 | 84.4 | 84.7 | 85.1 | 85.2 |
| DAC $\uparrow$ | 95.2 | 95.8 | 96.3 | 96.5 | 96.9 | 96.8 |
| PDMS $\uparrow$ | 86.7 | 88.4 | 89.1 | 89.4 | 89.6 | 89.6 |

## 5 Conclusion

In this paper, we conduct a comprehensive analysis of the two dominant VLA planners for autonomous driving: diffusion-based and token-based paradigms, highlighting their respective strengths and limitations. We further explore leveraging masked diffusion LLMs as a potential solution to mitigate their shortcomings. Building on these insights, we propose DriveFine, which features a plug-and-play block-MoE architecture combined with a hybrid reinforcement training strategy to inject refinement capabilities into token-based VLAs. We evaluate DriveFine on NAVSIMv1, NAVSIMv2, and the more challenging Navhard benchmarks, and demonstrate its effectiveness and robustness through extensive ablation studies and comparative analyses. We hope our findings and contributions will provide valuable insights for the community.
---
title: "WAM-Diff: A Masked Diffusion VLA Framework with MoE and Online Reinforcement Learning for Autonomous Driving"
source: "https://arxiv.org/html/2512.11872v1"
author:
published:
created: 2026-04-21
description:
tags:
  - "clippings"
---
Mingwang Xu <sup>1∗</sup>   Jiahao Cui <sup>1∗</sup>   Feipeng Cai <sup>2∗</sup>   Hanlin Shang <sup>1∗</sup>   Zhihao Zhu <sup>1</sup>   Shan Luan <sup>1</sup> Yifang Xu <sup>1</sup>   Neng Zhang <sup>2</sup>   Yaoyi Li <sup>2</sup>   Jia Cai <sup>2</sup>   Siyu Zhu ${}^{1}\textsuperscript{{\char 0\relax}}$  
<sup>1</sup> Fudan University     <sup>2</sup> Yinwang Intelligent Technology Co., Ltd

###### Abstract

End-to-end autonomous driving systems based on vision-language-action (VLA) models integrate multimodal sensor inputs and language instructions to generate planning and control signals. While autoregressive large language models and continuous diffusion policies are prevalent, the potential of discrete masked diffusion for trajectory generation remains largely unexplored. This paper presents WAM-Diff, a VLA framework that employs masked diffusion to iteratively refine a discrete sequence representing future ego-trajectories. Our approach features three key innovations: a systematic adaptation of masked diffusion for autonomous driving that supports flexible, non-causal decoding orders; scalable model capacity via a sparse MoE architecture trained jointly on motion prediction and driving-oriented visual question answering (VQA); and online reinforcement learning using Group Sequence Policy Optimization (GSPO) to optimize sequence-level driving rewards. Remarkably, our model achieves 91.0 PDMS on NAVSIM-v1 and 89.7 EPDMS on NAVSIM-v2, demonstrating the effectiveness of masked diffusion for autonomous driving. The approach provides a promising alternative to autoregressive and diffusion-based policies, supporting scenario-aware decoding strategies for trajectory generation. The code for this paper will be released publicly at: [https://github.com/fudan-generative-vision/WAM-Diff](https://github.com/fudan-generative-vision/WAM-Diff).

<sup>†</sup>![[teaser.png|Refer to caption]]

Figure 1: The proposed WAM-Diff framework supports flexible decoding orders for motion planning, illustrated for (a) causal, (b) reverse-causal, and random schedules, adapting to diverse driving scenarios. (c) By integrating a Mixture-of-Experts architecture with GSPO reinforcement learning, the model achieves superior performance in challenging driving situations.

![[main_arch.png|Refer to caption]]

Figure 2: Overview of WAM-Diff, the proposed VLA framework for end-to-end autonomous driving. The architecture integrates a MoE-enhanced backbone with a discrete mask diffusion decoder. Multimodal inputs–including ego camera images, ego-states, navigation, instructions–are encoded and fused. The masked diffusion process then iteratively generates the future trajectory (represented as a sequence of waypoints) from a fully masked initial state, guided by a remasking scheduler. The overall pipeline supports joint training with supervised multi-task learning and online GSPO reinforcement learning for improved driving performance.

## 1 Introduction

End-to-end autonomous driving systems [^23] [^21] [^32] [^44] [^33] [^41] [^3] [^36] [^57] [^59] grounded in VLA paradigms [^64] [^62] [^1] [^23] [^42] aim to integrate natural-language instructions with rich multi-sensor perceptual data. The objective is to develop unified frameworks capable of generating reliable control and planning signals directly from multimodal inputs. By jointly modeling visual perception, linguistic understanding, and action generation, these end-to-end models signify a substantial shift from traditional modular pipelines towards more unified architectures suited for complex, dynamic, and open-ended traffic environments. Current VLAs for autonomous driving primarily follow two architectural paradigms. The first encompasses autoregressive LLM-based approaches [^21] [^44] [^26] [^64], which generate action sequences token-by-token, often leveraging extensive multimodal pretraining for strong generalization [^23] [^64] [^26]. The second paradigm consists of diffusion policy models [^34] [^24], which iteratively refine action predictions through a noise-to-target denoising process, offering an alternative for capturing complex multi-modal distributions.

Recently, discrete masked diffusion [^34] [^42] has emerged as a promising generative architecture for sequential data, including language and multimodal tasks. This approach formulates sequence generation as an iterative infilling process: beginning with a fully masked sequence, the model progressively predicts all masked tokens in parallel at each step, while selectively re-masking low-confidence predictions. This allows the model to leverage bidirectional context throughout the decoding process, overcoming the inherent left-to-right generation constraint of autoregressive models. Such a paradigm is particularly suitable for trajectory generation in autonomous driving, as it naturally supports flexible decoding orders that can incorporate scenario-specific priors. For example, a causal order is well suited to near-term maneuvering such as turning, a reverse-causal order benefits car-following or oncoming interactions that require long-range anticipation, and a random order provides a balanced default. Nevertheless, its application to autonomous driving remains underexplored; this work addresses that gap.

This paper presents a systematic exploration of masked diffusion for autonomous driving VLAs, organized around three principal contributions. First, we conduct an in-depth analysis of the masked diffusion architecture adapted to the driving context. This includes the design of a hybrid discrete action tokenization scheme that interleaves numerical trajectory waypoints with semantic language tokens, improving the precision of future action prediction compared to purely text-based representations. Capitalizing on the inherent flexibility of masked diffusion, we investigate different decoding schedules for generating future action sequences and analyze their respective suitability for various driving scenarios. Second, we scale capacity with a sparse MoE via LoRA MoE [^13] [^66], yielding a 64-expert masked diffusion backbone that supports scalable motion planning. We demonstrate that joint training on both motion prediction and driving-oriented VQA tasks within this MoE framework yields superior performance compared to motion-only training, enhancing the model’s motion planning capabilities. Third, we incorporate the online reinforcement learning GSPO, tailored for the MoE framework. This approach optimizes the policy against multi-dimensional reward signals from simulation (e.g., non-collision, comfort, ego-progress), leading to significant improvements in overall driving performance as measured by standard benchmarks.

Experiments on the NAVSIM-v1 and v2 benchmarks [^12] demonstrate the effectiveness of the proposed approach. To the best of our knowledge, this is the first VLA for autonomous driving that integrates discrete masked diffusion with a sparse MoE and online GSPO reinforcement learning. Specifically, the model achieves 91.0 PDMS on NAVSIM-v1 and 89.7 EPDMS on NAVSIM-v2, reaching leading autoregressive baselines, underscoring the promise of masked diffusion decoders for VLAs. Beyond accuracy, masked diffusion circumvents the inherent left-to-right constraint of autoregressive generation, enabling random, casual, reverse-casual and scenario-aware decoding schedules. Through experiments, we demonstrate the flexibility of our method in generating trajectories with different decoding orders tailored to specific scenario priors.

![[scheduler.png|Refer to caption]]

Figure 3: Decoding schedules for masked diffusion. Top figures: remasking policies that regulate trajectory‑token update order (random, causal, reverse‑causal). Button figures: decoding efficiency via the mask‑rate schedule.

## 2 Related Work

End-to-End Autonomous Driving. Recent advances emphasize jointly optimizing perception and planning. UniAD [^20] integrates multiple perception tasks to improve planning quality; VAD [^22] introduces compact vectorized scene representations, and VADv2 [^8] extends this to multi-modal planning via trajectory scoring and anchor-based sampling. HydraMDP [^27] further stabilizes planning with rule-based supervision. ParaDrive [^43] analyzes core design choices for end-to-end systems, while generative approaches such as GenAD [^61] and DiffusionDrive [^29] leverage generative and diffusion modeling to capture multi-modal, temporally coherent trajectories. In parallel, vision–language models for driving have evolved from interpretive systems that describe scenes without direct control [^45] [^39], through modular language-to-action pipelines with non-differentiable interfaces [^62] [^55] [^1] [^56], to unified VLA architectures–such as DriveMoE [^48], ReCogDrive [^26], AutoVLA [^64] –that map sensory inputs to trajectories within a single differentiable model. We follow this paradigm and investigate discrete masked diffusion as a bidirectionally conditioned, parallel decoder for trajectory generation in end-to-end autonomous driving.

Discrete Diffusion in Multimodal LLMs. Early studies such as D3PM [^2] and SEDD [^30] established the foundation for diffusion over discrete variables. Recent approaches generally follow two denoising paradigms. The first, the masked diffusion process, formulates generation as iterative infilling: starting from a fully masked sequence, the model progressively predicts tokens in parallel and re-masks uncertain ones, enabling flexible decoding orders through an absorbing-state mechanism. LLADA [^34] and DREAM [^50] extended this approach to large-scale text generation, while LLADA-V [^51] and MMADA [^47] further expanded it to multimodal learning, demonstrating strong cross-modal reasoning and consistency. In contrast, uniform diffusion models [^2] [^5] [^16] [^40] employ structured transition kernels between discrete states, offering a probabilistic framework for sequence generation and showing strong performance in multimodal consistency tasks. Although, discrete diffusion models have emerged as a promising alternative to autoregressive generation for sequential data, preliminary attempts to apply discrete diffusion to autonomous driving [^10] [^24] exhibit performance gaps compared to state-of-the-art autoregressive methods. In this paper we aim to resolve this gap by scaling masked diffusion with MoE and GSPO online reinforcement learning, and leverage its advantages in parallel decoding and bidirectional context modeling [^53],

![[gspo.png|Refer to caption]]

Figure 4: Illustration of GSPO integrating multi-factor safety rewards—no collisions, drivable-area compliance, time-to-collision, comfort, and ego progress—into masked diffusion trajectory optimization for end-to-end autonomous driving.

Mixture of Experts in VLAs. The MoE architecture is a principal mechanism for parameter‑efficient scaling and task specialization in large language and vision–language models via dynamic expert routing [^11] [^38] [^46]. Recent VLA systems leverage MoE to disentangle visuomotor control from vision–language reasoning: ChatVLA [^65] shares attention while separating feed‑forward pathways, and ChatVLA2 [^63] introduces dynamic MoE layers without explicit FFNs. MoRE [^58] transforms dense networks into MoE by inserting LoRA‑based experts, akin to LoRAMoE [^13], demonstrating scalable specialization across tasks; ForVLA [^52] adopts a standard MoE design for manipulation. In autonomous driving, DriveMoE [^48] routes between scene‑ and skill‑specialized experts for view selection and maneuver generation, while ARTEMIS [^15] employs an autoregressive MoE planner for scene‑conditioned waypoint prediction. Building on these insights, we integrate a sparse LoRA‑based MoE into a masked diffusion backbone, enabling scalable adaptability across diverse and complex driving scenarios.

## 3 Method

We cast end-to-end autonomous driving as conditional masked diffusion over a unified discrete sequence that encodes future ego-trajectory under multimodal perceptual context and language guidance (Section 3.1). Building on this formulation, we instantiate masked diffusion for VLA, contrast it with autoregressive left-to-right decoding and diffusion policies, and exploit confidence- and prior-aware remasking schedules (causal, reverse-causal, and random) to control decoding order and efficiency for trajectory generation (Section 3.2). To scale capacity and semantics, we integrate a sparse LoRA-based MoE into the diffusion backbone and jointly train on motion prediction and driving-oriented VQA, yielding stronger motion planning. (Section 3.3). We further incorporate online GSPO to optimize sequence-level rewards for safety, ego progress, and comfort (Section 3.3). Finally, we present the overall architecture together with training and inference procedures (Section 3.4).

### 3.1 Problem Formulation

We cast end-to-end autonomous driving as conditional sequence modeling over a unified discrete representation of future actions. At each decision step $t$, the agent observes a multimodal context $c_{t}=\{I_{t},s_{t},u_{t}\}$, where $I_{t}$ denotes single-view camera images, $s_{t}$ the ego-vehicle state (e.g., velocity, acceleration, navigation), and $u_{t}$ a natural-language instruction. The prediction target is a future action sequence $x_{0}=(x_{0}^{1},\dots,x_{0}^{L})$ of length $L$, which encodes a planned ego-trajectory as a sequence of tokens drawn from a shared vocabulary spanning both numerical (metric waypoints) and semantic (control/rationale) symbols. Our objective is to learn the conditional distribution $p_{\theta}(x_{0}\mid c_{t})$. We instantiate $p_{\theta}$ with a discrete masked diffusion decoder that starts from a fully masked sequence and iteratively infills tokens under the joint conditioning of visual, state, and linguistic inputs.

### 3.2 Masked Diffusion for VLA

Hybrid Discrete Action Tokenization. We construct a unified vocabulary capable of representing both numerical values and textual tokens. Continuous variables—such as trajectory waypoints—are uniformly quantized over the interval $[-100,100]$ with a resolution of $0.01$, resulting in $20{,}001$ distinct numerical tokens. Each 2D waypoint is represented as an ordered pair $\langle x,y\rangle$ of scalar tokens; during decoding, the bin center of each quantized token is used, introducing a maximum absolute error of $0.005$ per coordinate. Semantic control commands and driving rationales (e.g., lane-keep, yield, turn-left) are represented using their corresponding textual tokens. The $20{,}001$ numerical tokens are merged into the existing text vocabulary, and their embedding projections are optimized end-to-end during masked diffusion training. This hybrid tokenization supports seamless interleaving of metric and linguistic information, enabling bidirectional conditioning while preserving both numerical precision and semantic interpretability.

Masked Diffusion for Trajectory Generation. Let $M$ denotes the special mask token and $r\in[0,1]$ the mask rate. The forward corruption independently replaces each position with $M$ with probability $r$:

$$
q_{r}(x_{r}|x_{0})=\prod_{i=1}^{L}\Big[(1-r)\mathbf{1}\{x_{r}^{i}=x_{0}^{i}\}+r\mathbf{1}\{x_{r}^{i}=M\}\Big].
$$

At inference, the reverse model $p_{\theta}(\cdot|x_{r},c_{t})$ jointly predicts all masked tokens conditioned on the visible tokens and context. A confidence-based remasking policy then re-masks low-confidence predictions to form a new corrupted sequence, and this infill–remask procedure is iterated until all tokens are resolved. This yields globally consistent, parallel decoding in a small number of steps.

For training, we minimize a masked cross-entropy with a continuous mask rate $r\sim\mathcal{U}(0,1)$, normalized by the expected number of masked tokens:

$$
\mathcal{L}(\theta)=-\mathbb{E}_{x_{0},r,x_{r}}\Bigg[\frac{1}{r}\sum_{i=1}^{L}\mathbf{1}\{x_{r}^{i}=M\}\log p_{\theta}\big(x_{0}^{i}|x_{r},c_{t}\big)\Bigg].
$$

This objective upper-bounds the negative log-likelihood of $p_{\theta}(x_{0}|c_{t})$ is Fisher-consistent in the large-data limit, and enables bidirectional dependencies across visual and textual tokens.

The masked diffusion formulation offers two advantages for VLAs: 1) Parallel, global prediction at each step enables efficient generation, potentially converging in few steps. 2) Controllable decoding via remasking schedules can inject driving priors into the token update order, and therefore surpass the causal constraints of autoregressive decoders.

Decoding Schedules for Action Sequences. At inference, we employ a decreasing mask schedule defined by $1=r_{0}>r_{1}>\cdots>r_{T}=0$, beginning from a fully masked sequence $x_{r_{0}}=M^{L}$. For each step $j=0,\dots,T-1$, the model performs two operations: it first infills all currently masked tokens by sampling from the conditional distribution $p_{\theta}(\cdot|x_{r_{j}},c_{t})$; subsequently, a remasking policy is applied to selectively re-mask a subset of tokens, yielding the input for the next step, $x_{r_{j+1}}$. The design of the remasking policy allows optimization of the masked diffusion process along two principal dimensions: the decoding order and the decoding efficiency.

As shown in Figure 3, the decoding order can be aligned with driving priors: a confidence-driven Random schedule re-masks low-confidence tokens irrespective of time; a Causal schedule monotonically unmasks tokens in temporal order to promote kinematic coherence; and a Reverse-causal schedule resolves distant future tokens before near-term ones to first stabilize long-horizon intent and subsequently refine immediate actions.

The decoding efficiency is simultaneously regulated by the remasking rate, which determines the fraction of tokens fixed per iteration, ranging from fine-grained single-token updates to full-sequence infilling in one or a few steps. Appropriate choices of order and efficiency yield faster convergence and improved trajectory accuracy and consistency across scenarios.

### 3.3 Scaling Masked Diffusion with LoRA MoE

Purely trajectory-supervised training equips the model to mimic motion patterns, but it is insufficient for safe and efficient driving that requires semantic scene understanding, traffic-rule compliance, and interactive reasoning. To address these limitations, we enhance our framework through two complementary approaches: 1) integrating a MoE architecture with multi-task learning combining trajectory prediction and driving-oriented VQA; 2) employing online reinforcement learning, namely GSPO, to further optimize safety, progress, and comfort metrics.

LoRA MoE Enhanced Masked Diffusion VLA. We incorporate a sparse LoRA MoE architecture into the feed-forward networks of our masked diffusion backbone to enable specialized expert capacity for different driving scenarios. The LoRA MoE formulation maintains parameter efficiency while allowing flexible scaling to accommodate complex driving situations. Formally, for an input representation $z$, the output of a LoRA MoE layer with $N$ experts is given by:

$$
o=W_{0}z+\sum_{i=1}^{N}g_{i}(z)E_{i}(z),
$$

where $W_{0}$ denotes the pre-trained feed forward network (FFN) projection matrix, $E_{i}(z)=B_{i}A_{i}z$ represents the low-rank adaptation of the $i$ -th expert (with $A_{i}\in\mathbb{R}^{r\times d_{\text{in}}}$, $B_{i}\in\mathbb{R}^{d_{\text{out}}\times r}$, and rank $r\ll\min(d_{\text{in}},d_{\text{out}}))$, and $g_{i}(z)$ is the routing weight produced by a softmax gate $g(z)=\text{Softmax}(W_{g}z)$.

This architecture enables dynamic routing of inputs to specialized experts based on scenario characteristics. We train the MoE-enhanced model on a hybrid objective combining trajectory prediction and driving-oriented VQA tasks. This allows the model to learn motion prediction capabilities not only through data-driven trajectory imitation but also through visual instruction training. This further enhances its capabilities from low-level scene perception and motion prediction to advanced driving planning and decision-making, thereby improving the model’s comprehensive ability to predict motion across multiple dimensions.

GSPO for MoE Masked Diffusion. After multi-task supervised MoE pretraining, we further online reinforce the masked diffusion policy with GSPO specifically for safety, ego progress, and comfort, which optimizes rewards at the level of whole action sequences and is therefore well-suited to sparse-expert routing.

For each context $c_{t}$ of a single driving process, we sample a group of $G$ candidate sequences $\{x_{i}\}_{i=1}^{G}\sim\pi_{\theta_{\text{old}}}(\cdot|c_{t})$. Each sequence is evaluated by the NAVSIM simulator under the PDMS metric, yielding rewards $R_{i}=r(c_{t},x_{i})$. GSPO converts these rewards into a group-normalized advantage, that is: $\hat{A}_{i}=\frac{R_{i}-\mathrm{mean}(\{R_{j}\}_{j=1}^{G})}{\mathrm{std}(\{R_{j}\}_{j=1}^{G})}$, which reduces scale sensitivity and variance across prompts.

Policy improvement is driven by a sequence-likelihood importance ratio that is length-normalized, for efficiency, we use one-step unmasking to estimate per-token log-probability $log\pi_{\theta}(x_{i,k}|c_{t})$ following [^60], and the importance ratio is calculated as:

$$
s_{i}(\theta)=\left(\frac{\pi_{\theta}(x_{i}|c_{t})}{\pi_{\theta_{\text{old}}}(x_{i}|c_{t})}\right)^{\frac{1}{|x_{i}|}}=\exp\left(\frac{1}{|x_{i}|}\sum_{k=1}^{|x_{i}|}\log\frac{\pi_{\theta}(x_{i,k}|c_{t})}{\pi_{\theta_{\text{old}}}(x_{i,k}|c_{t})}\right),
$$

where we compute sequence likelihood under a fixed factorization of the hybrid token sequence for comparability across policies. The GSPO objective then mirrors PPO at the sequence level:

$$
J_{\text{GSPO}}(\theta)=\mathbb{E}_{x}\left[\frac{1}{G}\sum_{i=1}^{G}\min\Big(s_{i}(\theta)\hat{A}_{i},\mathrm{clip}\big(s_{i}(\theta),1-\epsilon,1+\epsilon\big)\hat{A}_{i}\Big)\right].
$$

We maximize $J_{\text{GSPO}}$ (equivalently minimize $-J_{\text{GSPO}}$) with $\theta_{\text{old}}$ held fixed during an update epoch, and periodically set $\theta_{\text{old}}\leftarrow\theta$.

Optimizing entire sequences avoids token-wise credit assignment and the associated instability from changing expert routes, which is a severe problem for GRPO. Consequently, GSPO provides stable credit to the MoE policy for trajectories that score well under NAVSIM simulation, while the clipping term constrains policy updates measured in sequence likelihood space, yielding robust improvements in both safety and closed-loop driving performance.

### 3.4 Network Architecture

Overall Structure. The proposed architecture integrates four core components for multimodal reasoning and trajectory generation. The image encoder processes raw camera inputs by partitioning the 1920×1080 image into 15 non-overlapping 384×384 patches while simultaneously resizing the full image to the same reduced resolution. These 16 patches are encoded using SigLIP-2, yielding 2,185 visual tokens that are projected to 4,096 dimensions via an MLP to align with the text feature space. The text encoder extends the original vocabulary of 126,349 tokens by 20,001 new tokens for quantized waypoint representation, resulting in a total vocabulary size of 146,350. Training instances are structured using fixed question-answer templates that incorporate navigation commands, ego-state information, and waypoint predictions.

![[x1 22.png|Refer to caption]]

Figure 5: Qualitative comparison with existed open-sourced methods on NAVSIM benchmark.

| Method | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf.$\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| UniAD [^20] | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| PARA-Drive [^43] | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 |
| TransFuser [^9] | 97.7 | 92.8 | 92.8 | 100 | 79.2 | 84.0 |
| DRAMA [^54] | 98.0 | 93.1 | 94.8 | 100 | 80.1 | 85.5 |
| VADv2- $\mathcal{V}$ 8192 [^8] | 97.2 | 89.1 | 91.6 | 100 | 76.0 | 80.9 |
| Hydra-MDP- $\mathcal{V}$ 8192 [^27] | 97.9 | 91.7 | 92.9 | 100 | 77.6 | 83.0 |
| DiffusionDrive [^29] | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| ReCogDrive [^26] | 97.9 | 97.3 | 94.9 | 100 | 87.3 | 90.8 |
| DriveVLA-W0 [^25] | 98.7 | 99.1 | 95.3 | 99.3 | 83.3 | 90.2 |
| Ours | 99.1 | 98.3 | 96.5 | 99.9 | 84.4 | 91.0 |

Table 1: Comparison with state-of-the-art methods on the NAVSIM-v1.

| Method | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf.$\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| w/o | 97.8 | 94.2 | 93.4 | 99.7 | 78.5 | 84.7 |
| expert 16 | 98.0 | 94.0 | 93.5 | 99.7 | 79.0 | 85.0 |
| expert 64 | 98.0 | 95.5 | 94.2 | 99.4 | 80.7 | 86.6 |
| Rank 8 | 98.2 | 94.4 | 94.0 | 99.0 | 79.5 | 85.5 |
| Rank 32 | 98.0 | 95.5 | 94.2 | 99.4 | 80.7 | 86.6 |

Table 2: Ablation study on MOE configurations.

| Method | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf.$\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| w/o | 98.0 | 95.5 | 94.2 | 99.4 | 80.7 | 86.6 |
| GSPO (G=2) | 98.4 | 97.1 | 94.9 | 99.8 | 83.2 | 88.9 |
| GSPO (G=3) | 99.1 | 98.3 | 96.5 | 99.9 | 84.4 | 91.0 |

Table 3: Ablation study on online-reinforcement leaning GSPO. “G” denotes group sizes.

The core masked diffusion model builds upon the Llada-V multimodal backbone. To enhance capacity efficiently, we integrate a sparse MoE into the FFNs, incorporating 64 LoRA experts with rank 32. The routing follows an expert-choice strategy, with the original FFN parameters serving as a shared expert. This configuration results in a total of 8.4B parameters, with the MoE components adding only 0.5B parameters and activating approximately 0.05B during inference, thereby minimizing computational overhead. The text decoder consists of the original model’s text head adapted to the expanded vocabulary, enabling generation of hybrid sequences that interleave linguistic rationales with discrete trajectory waypoints.

Training. We adopt a four-stage training paradigm. 1) MoE warm-up. The diffusion backbone is frozen while LoRA experts are trained for 0.2 epochs on 668K nuPlan trajectory samples to initialize planning capabilities and prevent mode collapse. 2) Multi-task supervised pre-training. All parameters are unfrozen and jointly trained for one epoch on 668K trajectories and 800K VQA samples, coupling motion prediction with driving scene understanding. 3) NAVSIM adaptation. The model is fine-tuned for three epochs on 103K NAVSIM trajectories to bridge the distribution gap. 4) Online reinforcement learning. The policy is optimized using GSPO on NAVSIM data over two consecutive reinforcement learning epochs, with the first epoch’s output serving as the reference model for the second.

Inference. During inference, we fix the hybrid output length to 32 tokens and run 32 mask‑diffusion iterations with a monotonically decreasing mask schedule. Each iteration jointly infills all masked tokens and re‑masks the lowest‑confidence subset according to the selected policy (default confidence‑based; causal and reverse‑causal are optional), terminating early when no masks remain. We employ fast‑dLLM decoding for acceleration and disable step‑skipping to avoid quality regressions.

## 4 Experiments

### 4.1 Experimental Setups

All experiments were conducted on $4\times 8$ Ascend 910B NPUs across four sequential training phases. We use AdamW for all phases with a learning rate of $1\times 10^{-5}$, a cosine learning-rate schedule, a warm-up ratio of 0.02, and weight decay of 0. The batch size is set to 1 for memory efficiency. For GSPO, we use a group size of 3. Our MoE module contains 64 experts, each with a LoRA rank of 32 and input/output dimensions of 4096, with a dropout rate of 0.05. We adopt expert-choice routing with a routing capacity of 0.1. Training is performed in $\mathrm{bf16}$ precision.

| Method | NC $\uparrow$ | DAC $\uparrow$ | DDC $\uparrow$ | TLC $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | LK $\uparrow$ | HC $\uparrow$ | EC $\uparrow$ | EPDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ego Status | 93.1 | 77.9 | 92.7 | 99.6 | 86.0 | 91.5 | 89.4 | 98.3 | 85.4 | 64.0 |
| TransFuser [^35] | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 |
| HydraMDP++ [^27] | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | 81.4 |
| DriveSuprem [^49] | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 | 83.1 |
| ARTEMIS [^14] | 98.3 | 95.1 | 98.6 | 99.8 | 81.5 | 97.4 | 96.5 | 98.3 | \- | 83.1 |
| DiffusionDrive [^29] | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | 84.5 |
| DriveVLA-W0 [^25] | 98.5 | 99.1 | 98.0 | 99.7 | 86.4 | 98.1 | 93.2 | 97.9 | 58.9 | 86.1 |
| Ours | 99.0 | 98.4 | 99.3 | 99.9 | 87.0 | 98.6 | 96.2 | 98.1 | 78.5 | 89.7 |

Table 4: Comparison with state-of-the-art methods on the NAVSIM-v2 with extended metrics.

| ID | DS | CFG | MOE | GSPO | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf.$\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ✗ | ✗ | ✗ | ✗ | 97.0 | 93.1 | 91.6 | 99.5 | 76.0 | 80.3 |
| 2 | ✓ | ✗ | ✗ | ✗ | 97.0 | 93.1 | 91.6 | 99.5 | 76.0 | 82.3 |
| 3 | ✓ | ✓ | ✗ | ✗ | 97.8 | 94.2 | 93.4 | 99.7 | 78.5 | 84.7 |
| 4 | ✓ | ✓ | ✓ | ✗ | 98.0 | 95.5 | 94.2 | 99.4 | 80.7 | 86.6 |
| 5 | ✓ | ✓ | ✓ | ✓ | 99.1 | 98.3 | 96.5 | 99.9 | 84.4 | 91.0 |

Table 5: Ablation study on the proposed components of WAM-Diff. We evaluate the effect of decoding scheduler(DS), CFG, MoE and GSPO on NAVSIM-v1 evaluation.

| ID | TTC | Comf. | EP | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf.$\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ✓ | ✗ | ✗ | 99.1 | 98.2 | 96.7 | 99.9 | 84.0 | 90.7 |
| 2 | ✗ | ✓ | ✗ | 99.0 | 98.1 | 96.2 | 99.9 | 84.3 | 90.6 |
| 3 | ✗ | ✗ | ✓ | 98.6 | 97.9 | 95.4 | 99.4 | 84.9 | 90.3 |
| 4 | ✓ | ✓ | ✓ | 99.1 | 98.3 | 96.5 | 99.9 | 84.4 | 91.0 |

Table 6: Ablation study on different reward setting.

| Method | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf.$\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| Random | 98.9 | 97.8 | 96.5 | 99.9 | 83.0 | 90.0 |
| Causal | 98.7 | 97.0 | 95.5 | 99.8 | 82.3 | 88.9 |
| R. Causal | 99.1 | 98.3 | 96.5 | 99.9 | 84.4 | 91.0 |

Table 7: Ablation study of mask decoding schedules: Random, Causal, and Reverse Causal.

| CFG | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf.$\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| w/o | 97.0 | 93.1 | 91.6 | 99.5 | 76.0 | 82.3 |
| 1 | 97.2 | 93.6 | 92.0 | 99.4 | 76.9 | 83.2 |
| 3 | 97.6 | 93.6 | 92.7 | 99.6 | 77.4 | 83.6 |
| 5 | 97.7 | 94.0 | 93.4 | 99.7 | 78.0 | 84.4 |
| 7 | 97.7 | 94.2 | 93.4 | 99.6 | 78.4 | 84.7 |

Table 8: Analysis of classifier free guidance scales.

### 4.2 Comparison With Existed Works on NAVSIM

We evaluate WAM-Diff on the NAVSIM-v1 (v1.1 codebase version for evaluation) and v2 (v2.2 codebase version) benchmarks, comparing against leading autoregressive and diffusion-based VLAs for autonomous driving. As shown in Table 3, our method achieves a state-of-the-art PDMS of 91.0 on NAVSIM-v1, outperforming the strong diffusion-based baseline DiffusionDrive [^29] by +2.9 points and surpassing autoregressive methods including ReCogDrive [^26] (+0.2) and DriveVLA-W0 [^25] (+0.8). The performance gains are consistent across key sub-metrics, particularly in collision avoidance (NC) and drivable area compliance (DAC), underscoring the safety benefits of our approach.

On the more comprehensive NAVSIM-v2 benchmark (Table 4), which introduces additional metrics for traffic rule compliance and comfort, WAM-Diff achieves an EPDMS of 89.7. This represents a significant +5.2 point improvement over DiffusionDrive [^29] and a +3.6 point advantage over DriveVLA-W0 [^25]. The results demonstrate the robustness of our masked diffusion framework, particularly in complex driving scenarios requiring long-horizon reasoning and strict compliance with traffic regulations.

In Figure 5, we further demonstrate the qualitative comparison with two open-sourced models, DiffusionDrive [^29] and TransFuser [^9] in diverse driving scenarios.

### 4.3 Comparison With Existed Works on nuScenes

We evaluate our method on the nuScenes dataset [^4] following the NAVSIM benchmark protocol [^12] [^6], which prioritizes collision rate as the primary metric. This emphasis is motivated by prior findings in NAVSIM showing that open-loop L2 distance exhibits minimal correlation with closed-loop performance. As presented in Table 9, our method achieves an average collision rate of 0.11% under the ST-P3 metrics, matching the best-performing non-VLA model (UniAD). More importantly, under the more comprehensive UniAD evaluation protocol, DFM-Drive achives the lowest average collision rate (0.28%) among all VLA methods.

<table><tbody><tr><th rowspan="3">Method</th><td colspan="4">ST-P3 metrics</td><td colspan="4">UniAD metrics</td></tr><tr><td colspan="4">Collision (%) ↓</td><td colspan="4">Collision (%) ↓</td></tr><tr><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td></tr><tr><th>ST-P3 <sup><a href="#fn:18">18</a></sup></th><td>0.23</td><td>0.62</td><td>1.27</td><td>0.71</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><th>Ego-MLP <sup><a href="#fn:28">28</a></sup></th><td>0.21</td><td>0.35</td><td>0.58</td><td>0.38</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><th>InsightDrive <sup><a href="#fn:37">37</a></sup></th><td>0.09</td><td>0.10</td><td>0.27</td><td>0.15</td><td>0.08</td><td>0.15</td><td>0.84</td><td>0.36</td></tr><tr><th>VAD-v2 <sup><a href="#fn:7">7</a></sup></th><td>0.07</td><td>0.10</td><td>0.24</td><td>0.14</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><th>UniAD <sup><a href="#fn:19">19</a></sup></th><td>0.04</td><td>0.08</td><td>0.23</td><td>0.12</td><td>0.05</td><td>0.17</td><td>0.71</td><td>0.31</td></tr><tr><th>DriveVLM <sup><a href="#fn:39">39</a></sup></th><td>0.10</td><td>0.22</td><td>0.45</td><td>0.27</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><th>GPT-Driver <sup><a href="#fn:31">31</a></sup></th><td>0.04</td><td>0.12</td><td>0.36</td><td>0.17</td><td>0.07</td><td>0.15</td><td>1.10</td><td>0.44</td></tr><tr><th>AutoVLA <sup><a href="#fn:64">64</a></sup></th><td>0.13</td><td>0.18</td><td>0.28</td><td>0.20</td><td>0.14</td><td>0.25</td><td>0.53</td><td>0.31</td></tr><tr><th>DME-Driver <sup><a href="#fn:17">17</a></sup></th><td>-</td><td>-</td><td>-</td><td>-</td><td>0.05</td><td>0.28</td><td>0.55</td><td>0.29</td></tr><tr><th>Ours</th><td>0.04</td><td>0.09</td><td>0.22</td><td>0.11</td><td>0.02</td><td>0.17</td><td>0.66</td><td>0.28</td></tr></tbody></table>

Table 9: End-to-end motion planning performance on the nuScenes [^4] dataset.

### 4.4 Ablation Studies

Architecture Designs. We conduct a comprehensive ablation study to evaluate the contribution of each core component in WAM-Diff, with results summarized in Table 5. The baseline mask-diffusion model, trained solely on 668k nuPlan trajectory samples, achieves a PDMS of 80.2. Adding our proposed reverse-causal decoding scheduler yields an additional +2.0 PDMS, improving the stability of the generative process. Introducing classifier-free guidance results in another +2.4 PDMS gain. Incorporating the LoRA-MoE layer and jointly training on both VQA and trajectory data provides a further +1.9 PDMS, highlighting the benefit of multi-task learning for motion planning. Finally, applying GSPO-based reinforcement learning contributes a substantial +5.3 PDMS, optimizing the policy for closed-loop driving performance and culminating in an overall best score of 91.0 PDMS.

![[x2 20.png|Refer to caption]]

Figure 6: Qualitative analysis of the MoE component through BEV visualizations of motion planning trajectories and corresponding expert activation heatmaps.

![[fc2.png|Refer to caption]]

Figure 7: Qualitative analysis of failure cases.

Mixture-of-Experts. Table 3 analyzes the impact of MoE configuration on model performance. Without MoE, the baseline achieves a PDMS of 84.7. Introducing 16 experts improves performance to 85.0, and scaling to 64 experts yields a further gain of +1.6 PDMS, indicating that increased expert specialization enhances planning capability. We also evaluate the effect of LoRA rank. A rank of 32 yields the best performance (86.6 PDMS), outperforming ranks of 8 (85.5). Consequently, we adopt 64 experts with rank 32 as the default configuration. Figure 6 qualitatively analyze the MoE performance on different driving scenarios.

Reinforcement Learning GSPO. Tables 3 analyze the effects of different group sizes in the proposed GSPO framework. As shown in Table 3, introducing GSPO significantly improves model performance, raising PDMS from 86.6 (without GSPO) to 88.9 with a group size of 2, and further to 91.0 with a group size of 3, indicating that larger cooperative groups promote more stable and consistent policy optimization.

Figure 10 shows that the training reward curve with GSPO and GRPO. We observe that GSPO can deliver continuous performance improvement through increasing the training compute.

![[x3 20.png|Refer to caption]]

Figure 8: Qualitative ablation of online reinforcement learning GSPO on different driving scenarios.

![[x4 19.png|Refer to caption]]

Figure 9: Analysis of decoding orders on different scenarios.

![[gspo2grpo.png|Refer to caption]]

Figure 10: Training curves of reinforcement learning. GSPO possesses steady higher reward than GRPO.

Table 6 examines reward function design. When only sub-scores (TTC, Comfort, EP) are preserved, their respective metrics improve; however, the overall PDMS score decreases, indicating over-optimization toward specific objectives. In contrast, using the full reward settings achieves the best balance across all dimensions, leading to the highest overall driving performance (91.0 PDMS). As shown in Figure 8, incorporating GSPO further improves trajectory generation, yielding smoother behaviors and more reliable performance in complex scenes.

Classifier-Free Guidance. The impact of classifier-free guidance is evaluated in Table 8. Without CFG, the model achieves a PDMS of 82.3. Applying CFG consistently improves performance across guidance scales, with PDMS increasing to 84.7 at scales of 7. This improvement indicates that moderate guidance enhances planning stability and trajectory confidence.

Masked Diffusion Decoding Scheduler. Table 7 evaluates the effect of decoding order on WAM-Diff performance. A random decoding schedule achieves a PDMS of 90.0, while a causal schedule slightly reduces performance to 88.9. The reverse-causal schedule yields the highest PDMS of 91.0, with consistent gains across all metrics, including a +1.4 improvement in EP and +1.3 in DAC. As shown in Figure 9, causal decoding is particularly effective for turning scenarios, reverse-causal excels in complex interactions such as car-following and oncoming traffic, and random scheduling provides balanced performance across diverse scenarios.

### 4.5 Limitations and Future Works

Although WAM-Diff achieves state-of-the-art performance on both the NAVSIM v1 and v2 benchmarks, it still has several limitations. Figure 7 illustrates two representative failure cases. First, due to computational constraints, our model currently receives only a front-view image as input, which leads to perception failures when important obstacles lie outside this field of view. Second, the model processes only the current frame without any temporal history, making it difficult to infer other agents’ motion patterns and intent, potentially resulting in suboptimal or unsafe planning decisions. Future work may address these issues by designing a 3D vision encoder that better aligns with textual features and by developing more efficient model architectures capable of leveraging temporal information.

## 5 Conclusion

In this work, we introduced WAM-Diff, a vision-language-action framework that leverages masked discrete diffusion for trajectory generation in end-to-end autonomous driving. By unifying flexible non-causal decoding, a sparse MoE architecture, and reinforcement learning via GSPO, our approach achieves competitive performance on the NAVSIM benchmarks. The results demonstrate that masked diffusion offers a powerful and scalable alternative to conventional autoregressive and continuous diffusion models, enabling more adaptive and scenario-aware planning. Future research will explore extending this framework to larger-scale real-world datasets and incorporating richer multimodal reasoning for enhanced decision-making in complex driving environments.

## References

Supplementary Material

In the supplementary materials, we first describe the evaluation metrics for NAVSIM v1 [^12] and v2 [^6] in Section A.1 and A.2, respectively. Following this, Section A.3 provides the pseudo-code for our training and inference procedures, and Section A.4 details the hyperparameters used in the four-stage training pipeline. Additionally, we present more qualitative comparisons with state-of-the-art methods alongside an analysis of different decoding schedules in Section B.1.

## Appendix A Implementation Details

### A.1 NAVSIM v1 Evaluation Metrics

NAVSIM v1 [^12] metrics include No at-fault Collision (NC), Drivable Area Compliance (DAC), Time-To-Collision (TTC), Comfort (C), and Ego Progress (EP). NAVSIM uses the Predictive Driver Model Score (PDMS) to evaluate model performance:

$$
\mathrm{PDMS}=NC\times DAC\times\frac{5\times EP+5\times TTC+2\times C}{12}.
$$

No at-fault Collision (NC): Penalizes collisions based on fault assignment. NC=1 indicates no at-fault collisions, NC=0.5 indicates one fault collision with static objects, and NC=0 indicates multiple fault collisions.

Drivable Area Compliance (DAC): Measures adherence to drivable areas (lanes, parking areas). DAC=1 when the ego bounding box remains entirely within drivable areas, and DAC=0 when any corner exits designated areas.

Ego Progress (EP): Quantifies navigation goal achievement as the ratio of actual progress to a search-based safe upper bound derived from PDM-Closed trajectories. The ratio is clipped to \[0,1\], with low or negative values discarded.

Time-to-Collision (TTC): Encourages maintenance of safe distances from other vehicles. TTC=1 when the minimum time-to-collision exceeds 0.9 seconds, and 0 otherwise.

Comfort (C): Assesses kinematic constraints including acceleration and jerk. C=1 when all predefined thresholds are satisfied, and 0 upon any violation.

Algorithm 1 Training WAM-Diff

WAM-Diff $p_{\theta}$, Data $p_{data}$, Mask Token $M$

for $step=1$ to max training step do

    $c_{t},x_{0}\sim p_{data}$     $r\sim\mathcal{U}(0,1)$

   Sample mask $\mathbf{m}\sim\text{Bernoulli}(r)$

    $x_{r}\leftarrow\mathbf{m}\odot M+(1-\mathbf{m})\odot x_{0}$     $\mathcal{L}\leftarrow-\frac{1}{r}\sum_{i=1}^{L}\mathbf{1}\{x_{r}^{i}=M\}\log p_{\theta}(x_{0}^{i}|x_{r},c_{t})$

   Update $\theta\leftarrow\theta-\eta\nabla_{\theta}\mathcal{L}$

end for

return $p_{\theta}$

Algorithm 2 Inference

WAM-Diff $p_{\theta}$, Context $c_{t}=\{I_{t},s_{t},u_{t}\}$, Sampling Step $N$, Action Sequence Length $L$, Decoding Scheduler $R_{ds}$

$x_{1}\leftarrow M^{L}$ # fully masked sequence

for $r\leftarrow 1$ down to $1/N$ step $1/N$ do

    $s=1-1/N$     $x_{0}=\arg\max_{x_{0}}p_{\theta}(x_{0}\mid x_{r},c_{t})$

   for $i\leftarrow 0$ to $L$ do

     if $x_{r}^{i}\neq M$ then

         $x_{0}^{i}=x_{r}^{i}$

     else

         $x_{0}^{i}=R_{ds}(x_{r}^{i},M)$

     end if

   end for

    $x_{r}=x_{0}$

end for

return $x_{0}$

<table><tbody><tr><td>Stage</td><td>Parameter</td><td>Value</td></tr><tr><td rowspan="6">I</td><td>Number of epochs</td><td>0.2</td></tr><tr><td>Batch size</td><td>1</td></tr><tr><td>Dataset</td><td>nuPlan (668k)</td></tr><tr><td>Learning rate</td><td><math><semantics><mrow><mn>1</mn> <mo></mo><msup><mi>e</mi> <mrow><mo>−</mo> <mn>5</mn></mrow></msup></mrow> <annotation>1e^{-5}</annotation></semantics></math></td></tr><tr><td>Weight decay</td><td>0</td></tr><tr><td>Warmup ratio</td><td>0.02</td></tr><tr><td></td><td>Learning rate schedule</td><td>Cosine</td></tr><tr><td rowspan="6">II</td><td>Number of epochs</td><td>1</td></tr><tr><td>Batch size</td><td>1</td></tr><tr><td>Dataset</td><td>nuPlan (668k)+VQA (800k)</td></tr><tr><td>Learning rate</td><td><math><semantics><mrow><mn>1</mn> <mo></mo><msup><mi>e</mi> <mrow><mo>−</mo> <mn>5</mn></mrow></msup></mrow> <annotation>1e^{-5}</annotation></semantics></math></td></tr><tr><td>Weight decay</td><td>0</td></tr><tr><td>Warmup ratio</td><td>0.02</td></tr><tr><td></td><td>Learning rate schedule</td><td>Cosine</td></tr><tr><td rowspan="6">III</td><td>Number of epochs</td><td>3</td></tr><tr><td>Batch size</td><td>1</td></tr><tr><td>Dataset</td><td>NAVSIM (103k)</td></tr><tr><td>Learning rate</td><td><math><semantics><mrow><mn>1</mn> <mo></mo><msup><mi>e</mi> <mrow><mo>−</mo> <mn>5</mn></mrow></msup></mrow> <annotation>1e^{-5}</annotation></semantics></math></td></tr><tr><td>Weight decay</td><td>0</td></tr><tr><td>Warmup ratio</td><td>0.02</td></tr><tr><td></td><td>Learning rate schedule</td><td>Cosine</td></tr><tr><td rowspan="6">IV</td><td>Number of epochs</td><td>2</td></tr><tr><td>Batch size</td><td>1</td></tr><tr><td>Dataset</td><td>NAVSIM (103k)</td></tr><tr><td>Group Size</td><td>3</td></tr><tr><td>Learning rate</td><td><math><semantics><mrow><mn>1</mn> <mo></mo><msup><mi>e</mi> <mrow><mo>−</mo> <mn>5</mn></mrow></msup></mrow> <annotation>1e^{-5}</annotation></semantics></math></td></tr><tr><td>Weight decay</td><td>0</td></tr><tr><td></td><td>Warmup ratio</td><td>0.02</td></tr><tr><td></td><td>Learning rate schedule</td><td>Cosine</td></tr></tbody></table>

Table 10: Hyperparameters for WAM-Diff.

### A.2 NAVSIM v2 Evaluation Metrics

NAVSIM v2 [^6] includes several components, categorized as penalties or weighted subscores. Key metrics are No at-fault Collision (NC), Drivable Area Compliance (DAC), Driving Direction Compliance (DDC), Traffic Light Compliance (TLC), Ego Progress (EP), Time to Collision (TTC), Lane Keeping (LK), History Comfort (HC), and Extended Comfort (EC). NAVSIM v2 uses the Extended Predictive Driver Model Score (EPDMS) to evaluate model performance:

$$
\displaystyle\mathrm{EPDMS}
$$
 
$$
\displaystyle=NC\times DAC\times DDC\times TLC
$$
 
$$
\displaystyle\quad\times\frac{5EP+5TTC+2LK+2HC+2EC}{16}.
$$

Driving Direction Compliance (DDC): Penalizes reverse driving behavior. DDC=1 for reverse distance $<2$ m, DDC=0.5 for $2-6$ m, and DDC $=0$ for $>6$ m.

Traffic Light Compliance (TLC): Measures obedience to traffic signals. TLC $=1$ when traffic rules are followed, and 0 upon violations.

Lane Keeping (LK): Evaluates lateral positioning relative to lane centerlines, scored continuously from 0 to 1.

History Comfort (HC): Assesses trajectory consistency with historical motion patterns, ranging from 0 to 1.

Extended Comfort (EC): Compares planned trajectories across consecutive frames for dynamic consistency, scored from 0 to 1.

### A.3 Pseudo-Code of Training and Inference

In this section, we present the training and inference algorithms. Specifically, we introduce the training and inference algorithms in Algorithm 1 and Algorithm 2, respectively.

### A.4 Training Hyperparameter.

Training proceeds in four stages, summarized in Table 10. Stage I initializes the MoE by freezing the backbone and training only the LoRA experts for 0.2 epochs on 668K nuPlan trajectories. Stage II performs full-parameter multi-task learning for 1 epoch on 668K nuPlan trajectories combined with corresponding 800K VQA samples. Stage III adapts the model to the NAVSIM domain through 3 epochs of supervised fine-tuning on 103K trajectories. Stage IV conducts 2 epochs of NAVSIM training with a group size of 3 to support multi-scenario conditioning. Across all stages, we use a batch size of 1, a learning rate of $1\times 10^{-5}$, a warmup ratio of 0.02, zero weight decay, and a cosine learning-rate schedule.

## Appendix B Additional Experiment Results

### B.1 Additional Qualitative Results

We present extensive qualitative visualizations of WAM-Diff on NAVSIM comprising with Transfuser and DiffusionDrive to demonstrate the effectiveness of our proposed method (see Figure 11 and 12). We further showcase more examples of decoding orders on different scenarios (see Figure 13 and 14).

![[x5 19.png|Refer to caption]]

Figure 11: Qualitative results compare with existed methods.

![[x6 17.png|Refer to caption]]

Figure 12: Qualitative results compare with existed methods.

![[x7 14.png|Refer to caption]]

Figure 13: Qualitative results of causal schedule.

![[x8 10.png|Refer to caption]]

Figure 14: Qualitative results of reverse causal schedule.

[^1]: Hidehisa Arai, Keita Miwa, Kento Sasaki, Kohei Watanabe, Yu Yamaguchi, Shunsuke Aoki, and Issei Yamamoto. Covla: Comprehensive vision-language-action dataset for autonomous driving. In *2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)*, pages 1933–1943. IEEE, 2025.

[^2]: Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. *Advances in neural information processing systems*, 34:17981–17993, 2021.

[^3]: Yifan Bai, Dongming Wu, Yingfei Liu, Fan Jia, Weixin Mao, Ziheng Zhang, Yucheng Zhao, Jianbing Shen, Xing Wei, Tiancai Wang, et al. Is a 3d-tokenized llm the key to reliable autonomous driving? *arXiv preprint arXiv:2405.18361*, 2024.

[^4]: Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 11621–11631, 2020.

[^5]: Andrew Campbell, Joe Benton, Valentin De Bortoli, Thomas Rainforth, George Deligiannidis, and Arnaud Doucet. A continuous time framework for discrete denoising models. *Advances in Neural Information Processing Systems*, 35:28266–28279, 2022.

[^6]: Wei Cao, Marcel Hallgarten, Tianyu Li, Daniel Dauner, Xunjiang Gu, Caojun Wang, Yakov Miron, Marco Aiello, Hongyang Li, Igor Gilitschenski, et al. Pseudo-simulation for autonomous driving. *arXiv preprint arXiv:2506.04218*, 2025.

[^7]: Shaoyu Chen, Bo Jiang, Hao Gao, Bencheng Liao, Qing Xu, Qian Zhang, Chang Huang, Wenyu Liu, and Xinggang Wang. Vadv2: End-to-end vectorized autonomous driving via probabilistic planning. *arXiv preprint arXiv:2402.13243*, 2024a.

[^8]: Shaoyu Chen, Bo Jiang, Hao Gao, Bencheng Liao, Qing Xu, Qian Zhang, Chang Huang, Wenyu Liu, and Xinggang Wang. Vadv2: End-to-end vectorized autonomous driving via probabilistic planning. *arXiv preprint arXiv:2402.13243*, 2024b.

[^9]: Kashyap Chitta, Aditya Prakash, Bernhard Jaeger, Zehao Yu, Katrin Renz, and Andreas Geiger. Transfuser: Imitation with transformer-based sensor fusion for autonomous driving. *IEEE transactions on pattern analysis and machine intelligence*, 45(11):12878–12895, 2022.

[^10]: Can Cui, Yupeng Zhou, Juntong Peng, Sung-Yeon Park, Zichong Yang, Prashanth Sankaranarayanan, Jiaru Zhang, Ruqi Zhang, and Ziran Wang. Vilad: A large vision language diffusion framework for end-to-end autonomous driving. *arXiv preprint arXiv:2508.12603*, 2025.

[^11]: Damai Dai, Chengqi Deng, Chenggang Zhao, RX Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Yu Wu, et al. Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models. *arXiv preprint arXiv:2401.06066*, 2024.

[^12]: Daniel Dauner, Marcel Hallgarten, Tianyu Li, Xinshuo Weng, Zhiyu Huang, Zetong Yang, Hongyang Li, Igor Gilitschenski, Boris Ivanovic, Marco Pavone, et al. Navsim: Data-driven non-reactive autonomous vehicle simulation and benchmarking. *Advances in Neural Information Processing Systems*, 37:28706–28719, 2024.

[^13]: Shihan Dou, Enyu Zhou, Yan Liu, Songyang Gao, Wei Shen, Limao Xiong, Yuhao Zhou, Xiao Wang, Zhiheng Xi, Xiaoran Fan, et al. Loramoe: Alleviating world knowledge forgetting in large language models via moe-style plugin. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 1932–1945, 2024.

[^14]: Renju Feng, Ning Xi, Duanfeng Chu, Rukang Wang, Zejian Deng, Anzheng Wang, Liping Lu, Jinxiang Wang, and Yanjun Huang. Artemis: Autoregressive end-to-end trajectory planning with mixture of experts for autonomous driving. *arXiv preprint arXiv:2504.19580*, 2025a.

[^15]: Renju Feng, Ning Xi, Duanfeng Chu, Rukang Wang, Zejian Deng, Anzheng Wang, Liping Lu, Jinxiang Wang, and Yanjun Huang. Artemis: Autoregressive end-to-end trajectory planning with mixture of experts for autonomous driving, 2025b.

[^16]: Itai Gat, Tal Remez, Neta Shaul, Felix Kreuk, Ricky TQ Chen, Gabriel Synnaeve, Yossi Adi, and Yaron Lipman. Discrete flow matching. *Advances in Neural Information Processing Systems*, 37:133345–133385, 2024.

[^17]: Wencheng Han, Dongqian Guo, Cheng-Zhong Xu, and Jianbing Shen. Dme-driver: Integrating human decision logic and 3d scene perception in autonomous driving. In *Proceedings of the AAAI Conference on Artificial Intelligence*, pages 3347–3355, 2025.

[^18]: Shengchao Hu, Li Chen, Penghao Wu, Hongyang Li, Junchi Yan, and Dacheng Tao. St-p3: End-to-end vision-based autonomous driving via spatial-temporal feature learning. In *European Conference on Computer Vision*, pages 533–549. Springer, 2022.

[^19]: Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai Wang, et al. Planning-oriented autonomous driving. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 17853–17862, 2023a.

[^20]: Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai Wang, et al. Planning-oriented autonomous driving. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 17853–17862, 2023b.

[^21]: Jyh-Jing Hwang, Runsheng Xu, Hubert Lin, Wei-Chih Hung, Jingwei Ji, Kristy Choi, Di Huang, Tong He, Paul Covington, Benjamin Sapp, et al. Emma: End-to-end multimodal model for autonomous driving. *arXiv preprint arXiv:2410.23262*, 2024.

[^22]: Bo Jiang, Shaoyu Chen, Qing Xu, Bencheng Liao, Jiajie Chen, Helong Zhou, Qian Zhang, Wenyu Liu, Chang Huang, and Xinggang Wang. Vad: Vectorized scene representation for efficient autonomous driving. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 8340–8350, 2023.

[^23]: Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. *arXiv preprint arXiv:2406.09246*, 2024.

[^24]: Pengxiang Li, Yinan Zheng, Yue Wang, Huimin Wang, Hang Zhao, Jingjing Liu, Xianyuan Zhan, Kun Zhan, and Xianpeng Lang. Discrete diffusion for reflective vision-language-action models in autonomous driving. *arXiv preprint arXiv:2509.20109*, 2025a.

[^25]: Yingyan Li, Shuyao Shang, Weisong Liu, Bing Zhan, Haochen Wang, Yuqi Wang, Yuntao Chen, Xiaoman Wang, Yasong An, Chufeng Tang, et al. Drivevla-w0: World models amplify data scaling law in autonomous driving. *arXiv preprint arXiv:2510.12796*, 2025b.

[^26]: Yongkang Li, Kaixin Xiong, Xiangyu Guo, Fang Li, Sixu Yan, Gangwei Xu, Lijun Zhou, Long Chen, Haiyang Sun, Bing Wang, et al. Recogdrive: A reinforced cognitive framework for end-to-end autonomous driving. *arXiv preprint arXiv:2506.08052*, 2025c.

[^27]: Zhenxin Li, Kailin Li, Shihao Wang, Shiyi Lan, Zhiding Yu, Yishen Ji, Zhiqi Li, Ziyue Zhu, Jan Kautz, Zuxuan Wu, et al. Hydra-mdp: End-to-end multimodal planning with multi-target hydra-distillation. *arXiv preprint arXiv:2406.06978*, 2024a.

[^28]: Zhiqi Li, Zhiding Yu, Shiyi Lan, Jiahan Li, Jan Kautz, Tong Lu, and Jose M Alvarez. Is ego status all you need for open-loop end-to-end autonomous driving? In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 14864–14873, 2024b.

[^29]: Bencheng Liao, Shaoyu Chen, Haoran Yin, Bo Jiang, Cheng Wang, Sixu Yan, Xinbang Zhang, Xiangyu Li, Ying Zhang, Qian Zhang, et al. Diffusiondrive: Truncated diffusion model for end-to-end autonomous driving. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pages 12037–12047, 2025.

[^30]: Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution. *arXiv preprint arXiv:2310.16834*, 2023.

[^31]: Jiageng Mao, Yuxi Qian, Junjie Ye, Hang Zhao, and Yue Wang. Gpt-driver: Learning to drive with gpt. *arXiv preprint arXiv:2310.01415*, 2023a.

[^32]: Jiageng Mao, Yuxi Qian, Junjie Ye, Hang Zhao, and Yue Wang. Gpt-driver: Learning to drive with gpt. *arXiv preprint arXiv:2310.01415*, 2023b.

[^33]: Jiageng Mao, Junjie Ye, Yuxi Qian, Marco Pavone, and Yue Wang. A language agent for autonomous driving. *arXiv preprint arXiv:2311.10813*, 2023c.

[^34]: Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. *arXiv preprint arXiv:2502.09992*, 2025.

[^35]: Aditya Prakash, Kashyap Chitta, and Andreas Geiger. Multi-modal fusion transformer for end-to-end autonomous driving. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 7077–7087, 2021.

[^36]: Hao Shao, Yuxuan Hu, Letian Wang, Guanglu Song, Steven L Waslander, Yu Liu, and Hongsheng Li. Lmdrive: Closed-loop end-to-end driving with large language models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 15120–15130, 2024.

[^37]: Ruiqi Song, Xianda Guo, Hangbin Wu, Qinggong Wei, and Long Chen. Insightdrive: Insight scene representation for end-to-end autonomous driving. *arXiv preprint arXiv:2503.13047*, 2025.

[^38]: Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. *arXiv preprint arXiv:2403.05530*, 2024.

[^39]: Xiaoyu Tian, Junru Gu, Bailin Li, Yicheng Liu, Yang Wang, Zhiyong Zhao, Kun Zhan, Peng Jia, Xianpeng Lang, and Hang Zhao. Drivevlm: The convergence of autonomous driving and large vision-language models. *arXiv preprint arXiv:2402.12289*, 2024.

[^40]: Jin Wang, Yao Lai, Aoxue Li, Shifeng Zhang, Jiacheng Sun, Ning Kang, Chengyue Wu, Zhenguo Li, and Ping Luo. Fudoki: Discrete flow-based unified understanding and generation via kinetic-optimal velocities. *arXiv preprint arXiv:2505.20147*, 2025.

[^41]: Shihao Wang, Zhiding Yu, Xiaohui Jiang, Shiyi Lan, Min Shi, Nadine Chang, Jan Kautz, Ying Li, and Jose M Alvarez. Omnidrive: A holistic llm-agent framework for autonomous driving with 3d perception, reasoning and planning. *CoRR*, 2024.

[^42]: Yuqing Wen, Hebei Li, Kefan Gu, Yucheng Zhao, Tiancai Wang, and Xiaoyan Sun. Llada-vla: Vision language diffusion action models, 2025.

[^43]: Xinshuo Weng, Boris Ivanovic, Yan Wang, Yue Wang, and Marco Pavone. Para-drive: Parallelized architecture for real-time autonomous driving. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 15449–15458, 2024.

[^44]: Shuo Xing, Chengyuan Qian, Yuping Wang, Hongyuan Hua, Kexin Tian, Yang Zhou, and Zhengzhong Tu. Openemma: Open-source multimodal model for end-to-end autonomous driving, 2025.

[^45]: Zhenhua Xu, Yujia Zhang, Enze Xie, Zhen Zhao, Yong Guo, Kwan-Yee K Wong, Zhenguo Li, and Hengshuang Zhao. Drivegpt4: Interpretable end-to-end autonomous driving via large language model. *IEEE Robotics and Automation Letters*, 2024.

[^46]: An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. *arXiv preprint arXiv:2505.09388*, 2025a.

[^47]: Ling Yang, Ye Tian, Bowen Li, Xinchen Zhang, Ke Shen, Yunhai Tong, and Mengdi Wang. Mmada: Multimodal large diffusion language models. *arXiv preprint arXiv:2505.15809*, 2025b.

[^48]: Zhenjie Yang, Yilin Chai, Xiaosong Jia, Qifeng Li, Yuqian Shao, Xuekai Zhu, Haisheng Su, and Junchi Yan. Drivemoe: Mixture-of-experts for vision-language-action model in end-to-end autonomous driving. *arXiv preprint arXiv:2505.16278*, 2025c.

[^49]: Wenhao Yao, Zhenxin Li, Shiyi Lan, Zi Wang, Xinglong Sun, Jose M Alvarez, and Zuxuan Wu. Drivesuprim: Towards precise trajectory selection for end-to-end planning. *arXiv preprint arXiv:2506.06659*, 2025.

[^50]: Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Dream 7b: Diffusion large language models. *arXiv preprint arXiv:2508.15487*, 2025.

[^51]: Zebin You, Shen Nie, Xiaolu Zhang, Jun Hu, Jun Zhou, Zhiwu Lu, Ji-Rong Wen, and Chongxuan Li. Llada-v: Large language diffusion models with visual instruction tuning. *arXiv preprint arXiv:2505.16933*, 2025.

[^52]: Jiawen Yu, Hairuo Liu, Qiaojun Yu, Jieji Ren, Ce Hao, Haitong Ding, Guangyu Huang, Guofan Huang, Yan Song, Panpan Cai, et al. Forcevla: Enhancing vla models with a force-aware moe for contact-rich manipulation. *arXiv preprint arXiv:2505.22159*, 2025a.

[^53]: Runpeng Yu, Qi Li, and Xinchao Wang. Discrete diffusion in large language and multimodal models: A survey. *arXiv preprint arXiv:2506.13759*, 2025b.

[^54]: Chengran Yuan, Zhanqi Zhang, Jiawei Sun, Shuo Sun, Zefan Huang, Christina Dao Wen Lee, Dongen Li, Yuhang Han, Anthony Wong, Keng Peng Tee, et al. Drama: An efficient end-to-end motion planner for autonomous driving with mamba. *arXiv preprint arXiv:2408.03601*, 2024a.

[^55]: Jianhao Yuan, Shuyang Sun, Daniel Omeiza, Bo Zhao, Paul Newman, Lars Kunze, and Matthew Gadd. Rag-driver: Generalisable driving explanations with retrieval-augmented in-context learning in multi-modal large language model. *arXiv preprint arXiv:2402.10828*, 2024b.

[^56]: Jiawei Zhang, Xuan Yang, Taiqi Wang, Yu Yao, Aleksandr Petiushko, and Bo Li. Safeauto: Knowledge-enhanced safe autonomous driving with multimodal foundation models. *arXiv preprint arXiv:2503.00211*, 2025.

[^57]: Songyan Zhang, Wenhui Huang, Zihui Gao, Hao Chen, and Chen Lv. Wisead: Knowledge augmented end-to-end autonomous driving with vision-language model. *arXiv preprint arXiv:2412.09951*, 2024.

[^58]: Han Zhao, Wenxuan Song, Donglin Wang, Xinyang Tong, Pengxiang Ding, Xuelian Cheng, and Zongyuan Ge. More: Unlocking scalability in reinforcement learning for quadruped vision-language-action models. *arXiv preprint arXiv:2503.08007*, 2025a.

[^59]: Rui Zhao, Qirui Yuan, Jinyu Li, Haofeng Hu, Yun Li, Zhenhai Gao, and Fei Gao. Sce2drivex: A generalized mllm framework for scene-to-drive learning. *IEEE Robotics and Automation Letters*, 2025b.

[^60]: Siyan Zhao, Devaansh Gupta, Qinqing Zheng, and Aditya Grover. d1: Scaling reasoning in diffusion large language models via reinforcement learning. *arXiv preprint arXiv:2504.12216*, 2025c.

[^61]: Wenzhao Zheng, Ruiqi Song, Xianda Guo, Chenming Zhang, and Long Chen. Genad: Generative end-to-end autonomous driving. In *European Conference on Computer Vision*, pages 87–104. Springer, 2024.

[^62]: Xingcheng Zhou, Xuyuan Han, Feng Yang, Yunpu Ma, and Alois C Knoll. Opendrivevla: Towards end-to-end autonomous driving with large vision language action model. *arXiv preprint arXiv:2503.23463*, 2025a.

[^63]: Zhongyi Zhou, Yichen Zhu, Xiaoyu Liu, Zhibin Tang, Junjie Wen, Yaxin Peng, Chaomin Shen, and Yi Xu. Chatvla-2: Vision-language-action model with open-world reasoning. In *The Thirty-ninth Annual Conference on Neural Information Processing Systems*.

[^64]: Zewei Zhou, Tianhui Cai, Seth Z Zhao, Yun Zhang, Zhiyu Huang, Bolei Zhou, and Jiaqi Ma. Autovla: A vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. *arXiv preprint arXiv:2506.13757*, 2025b.

[^65]: Zhongyi Zhou, Yichen Zhu, Minjie Zhu, Junjie Wen, Ning Liu, Zhiyuan Xu, Weibin Meng, Yaxin Peng, Chaomin Shen, Feifei Feng, et al. Chatvla: Unified multimodal understanding and robot control with vision-language-action model. In *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing*, pages 5377–5395, 2025c.

[^66]: Fengqi Zhu, Zebin You, Yipeng Xing, Zenan Huang, Lin Liu, Yihong Zhuang, Guoshan Lu, Kangyu Wang, Xudong Wang, Lanning Wei, et al. Llada-moe: A sparse moe diffusion language model. *arXiv preprint arXiv:2509.24389*, 2025.
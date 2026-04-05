---
title: "WAM-Flow: Parallel Coarse-to-Fine Motion Planning via Discrete Flow Matching for Autonomous Driving"
source: "https://arxiv.org/html/2512.06112v2"
author:
published:
created: 2026-04-05
description:
tags:
  - "clippings"
---
Yifang Xu <sup>1∗</sup>   Jiahao Cui <sup>1∗</sup>   Feipeng Cai <sup>2∗</sup>   Zhihao Zhu <sup>1</sup>   Hanlin Shang <sup>1</sup>   Shan Luan <sup>1</sup> Mingwang Xu <sup>1</sup>   Neng Zhang <sup>2</sup>   Yaoyi Li <sup>2</sup>   Jia Cai <sup>2</sup>   Siyu Zhu ${}^{1}\textsuperscript{{\char 0\relax}}$  
<sup>1</sup> Fudan University     <sup>2</sup> Yinwang Intelligent Technology Co., Ltd  
Code & Model: [https://github.com/fudan-generative-vision/WAM-Flow](https://github.com/fudan-generative-vision/WAM-Flow)

###### Abstract

We introduce WAM-Flow, a vision–language–action (VLA) model that casts ego-trajectory planning as discrete flow matching over a structured token space. In contrast to autoregressive decoders, WAM-Flow performs fully parallel, bidirectional denoising, enabling coarse-to-fine refinement with a tunable compute–accuracy trade-off. Specifically, the approach combines a metric-aligned numerical tokenizer that preserves scalar geometry via triplet-margin learning, a geometry-aware flow objective and a simulator-guided GRPO alignment that integrates safety, ego progress, and comfort rewards while retaining parallel generation. A multi-stage adaptation converts a pre-trained auto-regressive backbone (Janus-1.5B) from causal decoding to non-causal flow model and strengthens road-scene competence through continued multimodal pretraining. Thanks to the inherent nature of consistency model training and parallel decoding inference, WAM-Flow achieves superior closed-loop performance against autoregressive and diffusion-based VLA baselines, with 1-step inference attaining 89.1 PDMS and 5-step inference reaching 90.3 PDMS on NAVSIM v1 benchmark. These results establish discrete flow matching as a new promising paradigm for end-to-end autonomous driving. The code will be publicly available soon.

<sup>†</sup> <sup>†</sup>![[Figure_2.png|Refer to caption]]

Figure 1: Architecture of the proposed WAM-Flow framework. Our method takes as input a front-view image, a natural-language navigation command with a system prompt, and the ego-vehicle states, and outputs an 8-waypoint future trajectory spanning 4 seconds through parallel denoising. The model is first trained via supervised fine-tuning to learn accurate trajectory prediction. We then apply simulator-guided GRPO to further optimize closed-loop behavior. The GRPO reward function integrates safety constraints (collision avoidance, drivable-area compliance) with performance objectives (ego-progress, time-to-collision, comfort).

## 1 Introduction

Vision-language–action models for end-to-end autonomous driving \[zhou2025autovla, FutureSightDrive-2025, RecogDrive-2025\] aim to map egocentric driving-view video inputs and natural-language instructions into both causal reasoning and precise ego-vehicle motion planning, while satisfying stringent efficiency and safety requirements. A fundamental challenge in this domain is the design of a policy representation that effectively balances three critical aspects: expressive reasoning capabilities, high-fidelity continuous control, and robust closed-loop performance. Existing approaches can be broadly categorized into dual-system and single-system paradigms. Dual-system methods \[RecogDrive-2025, Epona-2025, AdaDrive-2025, Alpamayo-R1-2025, DriveAgent-2025\] typically employ autoregressive vision-language models (VLMs) \[LLaMA-2, LLaVA, wang2024qwen2, Qwen-25, InternVL3-2025\] as auxiliary reasoning modules to provide high-level driving intent, scene summaries, or linguistic guidance for downstream motion planning networks, which often utilize diffusion-based iterative optimization \[DDPM-2020, DiffusionDrive-2025, jiang2025diffvla, Artemis-2025\] to generate smooth, complex action distributions. In contrast, single-system approaches \[EMMA-2024, OpenEMMA-2025, DrivingGPT-2025, FutureSightDrive-2025, zhou2025autovla\] such as EMMA \[EMMA-2024\] and DrivingGPT \[DrivingGPT-2025\] reformulate trajectory or action prediction as a text generation problem within the VLM, enabling reasoning and planning directly in the linguistic space. This work investigates a novel alternative based on discrete flow matching (DFM), which offers distinct advantages for autonomous driving applications.

Discrete flow matching \[DFM-2024, DFM-2025, wang2025fudoki, deng2025uniform, su2025theoretical, karimi2025fs, yue2025oat, cheng2025alpha\] models probability transport over discrete token spaces via a continuous-time Markov chain (CTMC) that carries a simple base distribution to the data distribution. Unlike autoregressive decoders that commit to tokens sequentially and accumulate exposure-bias errors, Discrete flow matching supports fully parallel denoising and bidirectional refinement during generation. These properties enable coarse-to-fine planning: beginning with a coarse motion hypothesis, the model increases trajectory fidelity through additional denoising steps, yielding a tunable compute–accuracy trade-off. This flexibility aligns well with autonomous driving, where simple scenes admit rapid approximate plans while complex interactions require higher-precision refinement. Despite these advantages, discrete flow matching remains largely unexplored for VLA policies in end-to-end autonomous driving.

However, a straightforward application of discrete flow matching to VLA model for end-to-end autonomous driving is nontrivial for three reasons. First, training discrete flow matching from scratch is prohibitively data- and compute-intensive, so they are typically initialized from general-purpose autoregressive multimodal VLMs that lack sufficient road-scene competence–from low-level perception and motion forecasting to high-level planning and decision making. We therefore adopt a multi-stage adaptation strategy: starting from a generic VLM backbone (Janus-1.5B \[Janus-2025\]), we continued conduct pretraining on large-scale road-scene visual question answering (VQA) to strengthen the ability to understand various complex road scenes and vehicle driving patterns, establishing a strong domain prior comparable to autoregressive VLA baselines. Second, standard text token embeddings are ill-suited to high-precision numerical regression because they weakly encode metric relationships. We introduce a metric-aligned numerical tokenizer that discretizes continuous scalars into a shared codebook and learns embeddings with a triplet-margin ranking objective so that latent distances reflect underlying scalar differences. This structured token space enables stable coarse-to-fine and slow–fast trajectory refinement within discrete flow matching, providing a controllable compute–accuracy trade-off. Finally, supervised likelihood-based flow training aligns the model with expert trajectories but does not explicitly enforce safety, ego-progress, and comfort in closed-loop control. We incorporate a Group Relative Policy Optimization (GRPO) based alignment objective with a composite reward that integrates safety penalties and performance goals, improving the safety–progress–comfort profile while preserving the model’s parallel generation capabilities.

Experimental results on the NAVSIM v1 and v2 benchmarks demonstrate that WAM-Flow achieves superior performance in PDMS and EPDMS metrics compared to both autoregressive and diffusion-based VLA models. By leveraging discrete flows over a structured token space, WAM-Flow enables flexible slow–fast and coarse-to-fine trajectory prediction. With 1-step denoising, it attains competitive performance (89.1 PDMS), while 5-step refinement yields further gains (90.3 PDMS). On the NAVSIM v2 benchmark, the full model achieves 84.7 EPDMS. Notably, the 1.5B-parameter WAM-Flow model achieves an 3 $\times$ improvement in inference speed over the Janus autoregressive baseline, underscoring the promising effectiveness and efficiency of the discrete flow matching approach for end-to-end autonomous driving.

## 2 Related Work

VLMs in Autonomous Driving. Autoregressive VLMs \[EMMA-2024, OpenEMMA-2025, FutureSightDrive-2025, zhou2025autovla, tian2024drivevlm, jiang2024senna\] formulate driving as a sequential language modeling problem, where each token corresponds to a trajectory point, control command, or reasoning step. Representative works such as EMMA \[EMMA-2024\], OpenEMMA \[OpenEMMA-2025\], FutureSightDrive \[FutureSightDrive-2025\] and AutoVLA \[zhou2025autovla\] leverage chain-of-thought reasoning and external memory modules to enhance interpretability and decision transparency. Despite their strong causal modeling capability, autoregressive architectures suffer from slow autoregressive decoding and limited parallelism, as future actions must be generated step-by-step. Diffusion-based methods, including DiffusionDrive \[DiffusionDrive-2025\], ViLaD \[cui2025vilad\] and DiffVLA \[jiang2025diffvla\] treat planning as a denoising process that gradually refines latent trajectory representations. These models enable parallel sampling but often lack explicit reasoning interpretability. In this paper, we explore a new promising paradigm for end-to-end autonomous driving, namely discrete flow matching.

Discrete Diffusion in LLMs and VLMs. Recent progress in discrete generative modeling has led to the emergence of discrete diffusion LLMs \[nie2025llada, ye2025dream\] and discrete diffusion VLMs \[nie2025large, yu2025dimple, you2025llada-v, li2025lavida, yang2025mmada\], which extend diffusion processes to tokenized sequences. This direction originates from D3PM \[austin2021-d3pm\], which formulated diffusion as a discrete Markov process over categorical variables. Recently, LLaDA \[nie2025llada\] trains an 8B-parameter model from scratch, reaching LLaMA-3 \[grattafiori2024llama-3\] performance with bidirectional reasoning and robustness. DREAM-7B \[ye2025dream\] further enhances diffusion-based reasoning via iterative refinement and arbitrary-order generation. Meanwhile, discrete flow matching \[gat2024discrete, lipman2024flow, shaul2024flow\] generalizes discrete diffusion via continuous-time probability paths and learnable velocity fields. By unifying diffusion and flow-based generation under a single probabilistic framework, it enables parallel, bidirectional, and efficient sampling. FUDOKI \[wang2025fudoki\] extends this framework to multimodal reasoning and generation, demonstrating unified, non-autoregressive modeling across modalities. In this paper, we apply discrete flow matching to VLA for autonomous driving and explore its inherent nature of parallel generation and coarse-to-fine controllability.

Reinforcement Learning in VLA. Building upon the success of DeepSeek-R1 \[guo2025deepseek\], GRPO has been further extended to autonomous driving domains. In particular, AlphaDrive \[jiang2025alphadrive\] pioneers the integration of GRPO-based reinforcement learning with planning-centric reasoning in autonomous driving, achieving notable improvements in both decision-making performance and training efficiency. TrajHF \[li2025finetuning\] further combines diffusion-based multimodal planners with reinforcement learning from human feedback, enabling safe and personalized trajectory generation aligned with diverse human driving styles. More recently, AutoVLA \[zhou2025autovla\] incorporates GRPO into vision-language-action models, extending reinforcement learning to end-to-end multimodal reasoning and low-level planning. To the best of our knowledge, this work presents the first exploration of GRPO within discrete flow matching for autonomous driving VLA. Furthermore, we explicitly incorporate safety alignment objectives, extending beyond conventional likelihood-based training to enhance reliability in autonomous driving contexts.

## 3 Method

We present WAM-Flow, a VLA model that formulates motion planning as a discrete flow matching problem over a structured token space. Specifically, Section 3.1 establishes the theoretical foundation of discrete flow matching over finite alphabets. Building on this, Section 3.2 details the model architecture, including a metric-aligned numerical tokenizer, and a geometry-aware flow objective. To address the limitations of likelihood-based training, Section 3.3 introduces simulator-guided GRPO to enforce safety and performance in closed-loop control. Finally, Section 3.4 specifies the autoregressive-to-flow training and the parallel denoising–based inference. Figure 1 demonstrates the pipeline of WAM-Flow.

### 3.1 Preliminaries: Discrete Flow Matching

Probability Paths. Let the discrete state space be defined as $S=\mathcal{T}^{D}$, where $\mathcal{T}=[K]=\{1,\dots,K\}$ represents a set of possible discrete values, and $D$ is the number of discrete variables. Denote the data distribution by $q(x)$ over $S$ and a simple factorized source distribution by $p(x)=\prod_{i=1}^{D}p^{i}(x^{i})$. We define a time-dependent probability path $\{p_{t}(x)\}_{t\in[0,1]}$ by marginalizing conditional, coordinate-wise factorized paths around a latent target $x_{1}$:

$$
\small p_{t}(x)=\sum_{x_{1}\in S}q(x_{1})\,p_{t}(x|x_{1}),\;p_{t}(x|x_{1})=\prod_{i=1}^{D}p_{t}^{i}\big(x^{i}|x_{1}^{i}\big),
$$

with boundary conditions ensuring $p_{0}^{i}(\cdot|x_{1}^{i})=p^{i}(\cdot)$ and $p_{1}^{i}(\cdot|x_{1}^{i})=\delta_{x_{1}^{i}}(\cdot)$, which yields $p_{0}(x)=p(x)$ and $p_{1}(x)=q(x)$. This mixture construction separates the definition of the transport path from the generative dynamics. A common instance is the mixture (mask) path:

$$
p_{t}^{i}(x^{i}|x_{1}^{i})=\big(1-\kappa_{t}\big)\,p^{i}(x^{i})+\kappa_{t}\,\delta_{x_{1}^{i}}(x^{i}),
$$

where $\kappa_{t}\in[0,1]$ is a monotonically increasing scheduling function satisfying $\kappa_{0}=0$ and $\kappa_{1}=1$. When $p^{i}(x^{i})=\delta_{\text{[MASK]}}(x^{i})$, this path recovers the standard masked corruption process.

Generative Dynamics. The probability path $p_{t}(x)$ is realized through a CTMC characterized by a probability velocity $u_{t}(x,z)$. This velocity acts as a rate matrix, defining the instantaneous transition rate from state $z$ to state $x$ at time $t$. Formally, for a small time step $h>0$, the transition probability satisfies:

$$
P(x_{t+h}=x|x_{t}=z)=\delta_{z}(x)+hu_{t}(x,z)+o(h),
$$

where $\delta_{z}(x)$ is the Kronecker delta and $o(h)$ denotes higher-order terms. The velocity $u_{t}$ must adhere to the constraints: $u_{t}(x,z)\geq 0$ for all $x\neq z$, and $\sum_{x}u_{t}(x,z)=0$. This velocity generates the path $p_{t}$ via the Kolmogorov forward equation:

$$
\dot{p}_{t}(x)+\operatorname{div}_{x}(j_{t})=0,
$$

where the probability flux is given by $j_{t}(x,z)=u_{t}(x,z)p_{t}(z)$. To maintain tractability in high-dimensional spaces, we restrict the velocity to permit only single-coordinate transitions.

### 3.2 WAM-Flow Architecture

Problem Formulation. We formulate the motion planning task as a conditional sequence generation problem. The model maps multimodal inputs–including synchronized front-view camera images, a natural-language navigation command, and the current ego-vehicle state (position, heading, velocity and acceleration)–to a discrete token sequence representing the planned trajectory. The output is a sequence of 8 waypoints spanning the next 4 seconds.

Within this formulation, WAM-Flow employs a flow network that learns to transport a simple prior distribution over the discrete token space to the expert trajectory distribution. An advantage of this approach is its support for fully parallel token transitions during generation, which circumvents the sequential bottleneck of autoregressive decoding. This capability enables a flexible trade-off between computational efficiency and prediction fidelity: rapid, coarse plans can be generated with few denoising steps, while high-precision trajectories are achieved through iterative refinement.

Metric-Aligned Numerical Tokenizer. Standard text token embeddings do not preserve metric structure and thus perform poorly for high-precision regression. We introduce a metric-aligned numerical tokenizer that discretizes continuous scalars (e.g., position, heading, velocity and acceleration) into a uniform codebook $\mathcal{V}=\{v_{1},\dots,v_{N}\}$ over $[-100,100]$ with 0.01 resolution ($N=20{,}001$). Each scalar token $v$ is mapped by a linear projection $E:\mathbb{R}\to\mathbb{R}^{d}$ and L2-normalized to yield the embedding $z=E(v)/\lVert E(v)\rVert_{2}$.

To align latent geometry with numeric distances, we enforce that Euclidean embedding distances are monotonic in the underlying scalar differences. Let $d_{ij}=\lVert z_{i}-z_{j}\rVert_{2}$. For any triplet $(i,j,k)$ with $|v_{i}-v_{j}|<|v_{i}-v_{k}|$, we promote $d_{ij}<d_{ik}$ via a triplet-margin ranking loss:

$$
\mathcal{L}_{\mathrm{num}}=\mathbb{E}_{(i,j,k)\sim\mathcal{T}}\big[\max\big(0,\;d_{ij}-d_{ik}+\alpha\big)\big],
$$

where $\mathcal{T}$ samples anchors $i$ with near/far neighbors $(j,k)$ and $\alpha>0$ is a fixed margin. This construction yields a numerically coherent token space in which latent distances faithfully reflect scalar proximity, enabling stable coarse-to-fine and slow–fast refinement under discrete flow matching. The induced distances serve as the tokenizer-specific metric $d_{i}(\cdot,\cdot)$ in the geometry-aware flow objective.

Discrete Flow Matching Objective. To respect the geometric structure of the tokenized action space, we design a conditional probability path that is both tractable and expressive. Given a target sequence $x_{1}\in q(x)$, we define a Gibbs distribution induced by a distance metric $d$:

$$
\small p_{t}(x|x_{1})=\mathrm{softmax}\left(-\beta_{t}d(x,x_{1})\right),\;\beta_{0}=0,\;\beta_{1}\to\infty,
$$

where $\beta_{t}$ is a monotonically increasing scheduling function on $[0,1]$, and $d(x,x_{1})=\sum_{i=1}^{D}w_{i}d_{i}(x^{i},x_{1}^{i})$ is a weighted sum of coordinate-wise dissimilarities. Each $d_{i}$ is tailored to the data type: tokenizer-induced distances for numerical values, circular metrics for angles, and semantic distances for textual fields. The nonnegative weights $w_{i}$ balance the contribution of each coordinate.

This path is realized by a CTMC with a transition rate designed to steer the state toward the target. The conditional rate for transitioning from $z$ to $x$ given $x_{1}$ is:

$$
u_{t}(x,z|x_{1})=p_{t}(x|x_{1})\dot{\beta}_{t}\left[d(z,x_{1})-d(x,x_{1})\right]_{+},
$$

where $[\cdot]_{+}=\max(0,\cdot)$. This rate assigns higher probability to transitions that reduce the dissimilarity to the target. The marginal velocity is obtained by integrating over the posterior distribution of $x_{1}$ given the current state.

The model is trained to approximate the true posterior $p_{1|t}(x_{1}|x)$ by minimizing the conditional flow matching cross-entropy loss:

$$
\small\mathcal{L}_{\mathrm{CE}}(\theta)=\mathbb{E}_{t\sim\mathcal{U}[0,1],\,x_{1}\sim q,\,x\sim p_{t}(\cdot|x_{1})}\left[-\sum_{i=1}^{D}\log p^{\theta,i}_{1|t}(x_{1}^{i}|x)\right],
$$

where $p_{1|t}^{\theta,i}(x_{1}^{i}|x)$ is the model’s estimate of the posterior probability for the $i$ -th target token. This geometry-aware formulation enables efficient parallel decoding and supports controllable refinement, allowing flexible trade-offs between planning speed and trajectory quality.

Model Architecture. We adapt a Janus-1.5B multimodal backbone to the discrete flow matching generation paradigm for vision–language–action planning. Images are resized with preserved aspect ratio, zero-padded to 384×384, and encoded by SigLIP \[SigLIP-2023\] into 576 visual tokens; a lightweight MLP aligns these features to the 2048-dimensional Janus text-token space. On the language side, we extend the Janus tokenizer by 20,001 numerically grounded tokens to represent input ego-state numbers and output waypoint coordinates, yielding a 122,401-word vocabulary. Training data are formatted with a fixed QA-style prompt that integrates navigation commands, ego-state (position, heading, velocity, acceleration), and the target waypoint sequence for the next 4 seconds (8 waypoints). For the decoder, the original Janus text head is expanded to the enlarged vocabulary and used to predict action tokens under the discrete flow matching objective.

### 3.3 Simulator-Guided GRPO

While supervised flow matching optimizes trajectory prediction accuracy, it does not explicitly enforce critical driving objectives such as safety, comfort, and progress in closed-loop control. To address this limitation, we introduce an online GRPO reinforcement learning that aligns the policy with simulator-derived rewards while preserving the parallel generation capabilities of discrete flow matching.

Reward Design. We design a composite reward function that decomposes the NAVSIM simulator’s PDMS metrics into safety penalties and performance objectives. The reward for a generated trajectory $\tau$ is defined as:

$$
R(\tau)=\underbrace{\left(\prod_{m\in\mathcal{M}}s_{m}(\tau)\right)}_{\text{safety penalties}}\cdot\underbrace{\left(\frac{\sum_{w\in\mathcal{W}}\lambda_{w}s_{w}(\tau)}{\sum_{w\in\mathcal{W}}\lambda_{w}}\right)}_{\text{performance objectives}},
$$

where $\mathcal{M}=\{\mathrm{NC},\mathrm{DAC}\}$ represents safety metrics, including no-collision and drivable-area compliance; $\mathcal{W}=\{\mathrm{EP},\mathrm{TTC},\mathrm{C}\}$ denotes performance metrics, including ego-progress, time-to-collision, and comfort. The multiplicative safety term ensures strict constraint satisfaction, while the weighted average balances performance trade-offs. Specifically, the NC score assigns $s_{\mathrm{NC}}(\tau)=0$ for at-fault collisions, 0.5 for collisions with static objects, and 1 otherwise; DAC yields $s_{\mathrm{DAC}}(\tau)=0$ on violations and 1 otherwise. Sub-scores $s_{w}(\tau)\in[0,1]$ are normalized, with $\lambda_{w}\geq 0$ as weighting coefficients.

GRPO Objective. For a given scene context c, we sample $G$ candidate trajectories $\{\tau_{i}\}_{i=1}^{G}\sim\pi_{\theta_{\mathrm{old}}}(\cdot|c)$ via parallel denoising. Each trajectory receives a reward $R_{i}=R(\tau_{i})$. Using the group baseline $A_{i}=R_{i}-\frac{1}{G}\sum_{j=1}^{G}R_{j}$, we define per-token importance ratios for action tokens $\{o_{i}^{k}\}_{k=1}^{T_{i}}$ under conditioning states $\{s_{i}^{k}\}$: $r_{i}^{k}(\theta)=\frac{\pi_{\theta}(o_{i}^{k}s_{i}^{k})}{\pi_{\theta_{\mathrm{old}}}(o_{i}^{k}s_{i}^{k})}$. The GRPO surrogate objective, with clipping parameter $\epsilon>0$ and KL regularization strength $\beta\geq 0$, is formulated as:

$$
\displaystyle\mathcal{L}_{\mathrm{GRPO}}(\theta)=\mathbb{E}_{c}\Bigg[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{T_{i}}\sum_{k=1}^{T_{i}}\Big(
$$
$$
\displaystyle\min\left\{r_{i}^{k}(\theta)A_{i},\mathrm{clip}(r_{i}^{k}(\theta),1-\epsilon,1+\epsilon)A_{i}\right\}
$$
 
$$
\displaystyle-\beta D_{\mathrm{KL}}\left(\pi_{\theta}(\cdot s_{i}^{k})\ \pi_{\mathrm{ref}}(\cdot s_{i}^{k})\right)\Big)\Bigg].
$$

The group baseline reduces variance by inducing relative preferences within each sample set, while the KL divergence term stabilizes updates by anchoring the policy to the supervised reference.

![[Figure_3.png|Refer to caption]]

Figure 2: Overview of the full training curriculum. Different training stage motivation and corresponding training data and training steps are demonstrated.

### 3.4 Training and Inference

Autoregressive-to-Flow Training. Figure 2 outlines a four-stage curriculum. First, we randomly initialize the numerical embeddings and freeze the VLA backbone. We train the numerical embeddings together with the language-model head on the 668K nuPlan dataset for 4 epochs, using flow-matching loss $\mathcal{L}_{\mathrm{CE}}$ (Equation 8) and triplet-margin ranking loss $\mathcal{L}_{\mathrm{num}}$ (Equation 5). Second, we enhance the perception of driving scenes by pretraining VLA using $\mathcal{L}_{\mathrm{CE}}$. This stage trains 3 epochs on 6.5M VQA, including general multimodal VQA (3.4M) from LLaVA-v1.5 \[LLaVA\] and large-scale driving-specific VQA (3.1M) from RecogDrive \[RecogDrive-2025\], which enhances perceptual grounding and driving-specific causal reasoning. Third, we supervised fine-tuning of the VLA backbone for only 2 epochs on the nuPlan dataset with $\mathcal{L}_{\mathrm{CE}}$. After supervised flow training, we perform reinforcement learning with simulator feedback by maximizing the GRPO objective (Equation 10) with KL regularization toward the supervised reference to optimize our VLA model for 0.5 epoch on 103k NAVSIM dataset. We set the weight for $\mathrm{EP},\mathrm{TTC}$ and $\mathrm{C}$ in our reward to 5:5:2.

Inference. First, we apply the Euler discretization over the time interval $[0,1]$ with $n$ inference steps, yielding a step size of $h=\frac{1}{n}$. For each coordinate $i$, the initial token $x_{0}^{i}$ is sampled uniformly from the model vocabulary. At each discrete timestep $t\in[0,1]$, the current token is denoted as $x_{t}^{i}$, and a target token $x_{1}^{i}$ is drawn from the posterior distribution $p_{1|t}^{i}(x_{1}^{i}|x)$. Next, we compute the total outgoing transition rate $\lambda_{i}$ for the current token $x_{t}^{i}$ as $\lambda_{i}=\sum_{x^{i}\neq x_{t}^{i}}u_{t}^{i}(x^{i},x_{t}^{i}|x_{1}^{i})$, where the conditional rate function $u_{t}^{i}$ is defined in Equation 7. A uniform random variable $Z_{i}\sim\mathcal{U}[0,1]$ is then drawn. The jump rule is as follows: if $Z_{i}<1-e^{-h\lambda_{i}}$, a transition occurs, and the new token $x_{t+h}^{i}$ is sampled proportionally to the normalized rates $u_{t}^{i}(\cdot,x_{t}^{i}|x_{1}^{i})$; otherwise, the token remains unchanged, i.e., $x_{t+h}^{i}=x_{t}^{i}$. After $n$ sampling steps, we obtain the final output token sequence $x_{1}$.

| Method | Paradigm | Backbone | Input | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf.$\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| End-to-End |  |  |  |  |  |  |  |  |  |
| VADv2 \[VADv2-2024\] | \- | \- | 6 $\times$ Cam | 97.2 | 89.1 | 91.6 | 100 | 76.0 | 80.9 |
| Transfuser \[Transfuser-2022\] | \- | \- | 3 $\times$ Cam + L | 97.7 | 92.8 | 92.8 | 100 | 79.2 | 84.0 |
| Hydra-MDP++ \[Hydra-MLP-pp-2025\] | \- | \- | 3 $\times$ Cam + L | 98.3 | 96.0 | 94.6 | 100 | 78.7 | 86.5 |
| Artemis \[Artemis-2025\] | Diff. | \- | 6 $\times$ Cam | 98.3 | 95.1 | 94.3 | 99.8 | 81.4 | 87.0 |
| DiffusionDrive \[DiffusionDrive-2025\] | Diff. | \- | 3 $\times$ Cam + L | 98.2 | 96.0 | 94.8 | 100 | 82.2 | 88.1 |
| End-to-End VLA |  |  |  |  |  |  |  |  |  |
| DrivingGPT \[DrivingGPT-2025\] | AR | LLaMA2-7B \[LLaMA-2\] | 1 $\times$ Cam | 98.1 | 90.7 | 94.9 | 95.6 | 79.7 | 82.4 |
| FSDrive \[FutureSightDrive-2025\] | AR | Qwen2-VL-2B \[wang2024qwen2\] | 6 $\times$ Cam | 98.2 | 93.8 | 93.3 | 99.9 | 80.1 | 85.1 |
| Epona \[Epona-2025\] | AR + Diff. | DiT-2.5B \[DiT-2023\] | 1 $\times$ Cam | 97.9 | 95.1 | 93.8 | 99.9 | 80.4 | 86.2 |
| AutoVLA \[zhou2025autovla\] | AR | Qwen2.5-3B \[Qwen-25\] | 3 $\times$ Cam | 98.4 | 95.6 | 98.0 | 99.9 | 81.9 | 89.1 |
| ReCogDrive \[RecogDrive-2025\] | AR + Diff. | InternVL3-8B \[InternVL3-2025\] | 3 $\times$ Cam | 98.2 | 97.5 | 95.2 | 99.9 | 83.5 | 89.6 |
| Ours | DFM | Janus-1.5B \[Janus-2025\] | 1 $\times$ Cam | 99.2 | 98.3 | 97.0 | 99.7 | 82.3 | 90.3 |

Table 1: Comparison on NAVSIM-v1 with closed-loop metrics. Abbreviation: Diff.(Diffusion), Comf.(Comfort), Cam (Camera), L (LiDAR).

| Group Size | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf.$\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| w/o GRPO | 98.5 | 95.1 | 94.4 | 99.5 | 81.8 | 86.7 |
| 2 | 99.4 | 97.3 | 96.8 | 99.7 | 80.7 | 89.2 |
| 3 | 99.2 | 98.3 | 97.0 | 99.7 | 82.3 | 90.3 |
| 4 | 99.3 | 97.6 | 96.5 | 99.8 | 82.0 | 89.6 |

Table 2: Ablation on GRPO group size.

| EP: TTC: C | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf.$\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| 5:20:2 | 99.5 | 98.3 | 97.9 | 99.6 | 80.1 | 89.7 |
| 5:5:8 | 99.4 | 98.1 | 96.9 | 99.7 | 82.1 | 90.1 |
| 20:5:2 | 99.4 | 98.1 | 96.4 | 99.3 | 82.7 | 90.0 |
| 5:5:2 | 99.2 | 98.3 | 97.0 | 99.7 | 82.3 | 90.3 |

Table 3: Ablation on different weight of Simulator-Guided reward. The default weight is 5:5:2 for Navsim simulator, and we adjust the scale of each weight by $4\times$ to obtain the new weight.

![[navsim_comp.png|Refer to caption]]

Figure 3: Qualitative comparison on NAVSIM.

Method NC $\uparrow$ DAC $\uparrow$ DDC $\uparrow$ TLC $\uparrow$ EP $\uparrow$ TTC $\uparrow$ LK $\uparrow$ HC $\uparrow$ EC $\uparrow$ EPDMS $\uparrow$ Ego Status 93.1 77.9 92.7 99.6 86.0 91.5 89.4 98.3 85.4 64.0 VADv2 \[VADv2-2024\] 97.3 91.7 98.2 99.7 77.6 92.7 66.0 98.3 83.3 76.6 TransFuser \[Transfuser-2022\] 97.7 92.8 98.3 99.7 79.2 92.8 67.6 98.3 87.2 77.8 HydraMDP++ \[Hydra-MLP-pp-2025\] 97.2 97.5 99.4 99.6 83.1 96.5 94.4 98.2 70.9 81.4 Artemis \[Artemis-2025\] 98.3 95.1 98.6 99.8 81.5 97.4 96.5 98.3 - 83.1 RecogDrive \[RecogDrive-2025\] 98.3 94.2 98.8 99.8 86.5 97.3 96.8 98.3 87.7 83.6 Ours 98.5 94.5 99.5 99.8 86.9 96.8 97.4 97.6 73.9 84.7

Table 4: Comparison on NAVSIM-v2 with extended metrics.

Numerical Tokenizer Metric -aligned Pre- training SG GRPO NC $\uparrow$ DAC $\uparrow$ TTC $\uparrow$ Comf.$\uparrow$ EP $\uparrow$ PDMS $\uparrow$ ✗ ✗ ✗ ✗ 95.8 87.5 88.6 99.5 71.7 76.2 ✓ ✗ ✗ ✗ 97.0 91.3 91.0 98.9 76.4 81.1 ✓ ✓ ✗ ✗ 97.4 92.6 95.3 99.3 77.5 83.4 ✓ ✓ ✓ ✗ 98.5 95.1 94.4 99.5 81.8 86.7 ✓ ✓ ✗ ✓ 98.4 96.1 95.3 99.5 79.3 86.9 ✓ ✓ ✓ ✓ 99.2 98.3 97.0 99.7 82.3 90.3

Table 5: Ablation study for the proposed components. We evaluate the effect of metric-aligned numerical tokenizer, VQA pretraining and simulator-guided GRPO on NAVSIM-v1. Row 1 uses the text tokenizer from Janus-1.5B to tokenize the number. “SG GRPO” refers to “Simulator-Guided GRPO”.

![[navsim_vis.png|Refer to caption]]

Figure 4: Qualitative results of WAM-Flow on NAVSIM with different scenes.

## 4 Experiments

### 4.1 Experimental Setup

Implementation. All experiments were conducted on 4 $\times$ 8 Ascend 910B NPUs across four sequential training phases. We use AdamW optimizer for all training stages with weight decay of 0.01. In the metric-aligned numerical embeddings training stage, we set $\alpha$ to 0.05, constant learning rate to $1\times 10^{-5}$ and batch size to 80. In the pre-training stage, we set constant learning rate to $1\times 10^{-5}$ and batch size to 256. In the SFT stage, we utilize learning rate of $5\times 10^{-6}$ with cosine annealing strategy and batch size of 64. In the reinforcement learning stage, we use a learning rate of $1\times 10^{-6}$, batch size of 32 and 500 warm-up steps. During inference, we use a timestep schedule defined as $\beta_{t}=3\times\left(\frac{t}{1-t}\right)^{0.9}$, and perform inference with 1, 2, 3, 5, and 10 sampling steps. On NAVSIM-v1 benchmark, we conduct evaluations using the v1.1 version of the NAVSIM codebase, while on NAVSIM-v2 benchmark, we use the v2.2 version for evaluation.

Metrics. We evaluate our method on the closed-loop NAVSIM-v1 \[NAVSIM-v1\] and v2 \[NAVSIM-v2\] benchmarks. The primary metric for NAVSIM-v1 is the Predictive Driver Model Score (PDMS), a composite measure integrating five key components: No-Collision rate (NC), Drivable Area Compliance (DAC), Time-to-Collision within bound (TTC), Comfort, and Ego Progress (EP). The more comprehensive NAVSIM-v2 benchmark employs the Extended Predictive Driver Model Score (EPDMS), which incorporates nine sub-metrics—NC, DAC, Driving Direction Compliance (DDC), Traffic Light Compliance (TLC), EP, TTC, Lane Keeping (LK), History Comfort (HC), and Extended Comfort (EC)– to provide a holistic assessment of driving performance, safety, and rule adherence. All results are obtained from closed-loop simulations on the official public test splits.

### 4.2 Comparison with State-of-the-Art

NAVSIM-v1. As shown in Table 3, our method achieves the highest PDMS (90.3) on the NAVSIM v1 benchmark. It attains the superior performance in both safety-critical metrics–No-Collision (NC: 99.2) and Drivable Area Compliance (DAC: 98.3)–demonstrating superior safety and rule adherence. Notably, despite utilizing only a single front-view camera, our model outperforms methods that rely on multi-view camera setups or LiDAR inputs, underscoring the efficacy of the discrete flow matching paradigm in achieving robust and efficient planning. Qualitative analyses (Figure 3 and Figure 4) further illustrate that our planner produces stable, human-like trajectories in closed-loop simulation.

NAVSIM-v2. Table 4 presents the evaluation results on the more comprehensive NAVSIM-v2 benchmark. Our method achieves the superior overall EPDMS of 84.7. It also leads in several critical sub-metrics: No-Collision (NC: 98.5), Driving Direction Compliance (DDC: 99.5), and Lane Keeping (LK: 97.4). The superior performance across diverse and dynamic scenarios underscores the robustness of our approach for reliable closed-loop driving.

![[ablation_pretrain_epoch.png|Refer to caption]]

Figure 5: Impact of pre-training epochs. We perform SFT after pre-training on 6.5M data, and then calculate PDMS.

### 4.3 Ablation and Discussion

![[ablation_pretrain_data.png|Refer to caption]]

Figure 6: Effect of pretraining dataset scale.

![[x1 1.png|Refer to caption]]

Figure 7: Ablation about Simulator-guided GRPO.

Ablation Study for Proposed Components. As shown in Table 5, we systematically evaluate the contribution of each component in our framework. Using the Janus-1.5B text tokenizer for numerical values results in a PDMS of 76.2, indicating its inadequacy for representing fine-grained trajectory data. Replacing it with a dedicated numerical tokenizer improves PDMS by 4.9 points (to 81.1), confirming the necessity of a specialized numerical representation. Further incorporating metric-aligned embeddings yields an additional gain of 2.3 points (to 83.4), demonstrating that geometric consistency in the token space enhances planning quality. Subsequent large-scale VQA pretraining adds 3.3 points (to 86.7), underscoring the benefit of cross-modal domain adaptation. Finally, integrating simulator-guided GRPO achieves the highest PDMS of 90.3, highlighting the critical role of safety and performance alignment through online reinforcement learning.

In addition, comparing Rows 5 and 6 in Table 5 reveals that incorporating VQA pre-training yields a +3.4 improvement in PDMS, underscoring the efficacy of pre-training in enhancing driving performance. This gain demonstrates that domain-specific pre-training on large-scale visual question answering data provides valuable foundational knowledge for complex driving scenarios, complementing the benefits of reinforcement learning-based fine-tuning.

Further Pretrain on More Driving Data. Figure 5 illustrates the impact of pre-training epochs on model performance. For a fair comparison, we conduct SFT following pre-training on 6.5M VQA dataset. The results show that the PDMS score increases with the number of pre-training epochs, reaching a peak at 3 epochs, with an improvement of +3.3 compared to 0 epochs. This indicates that pre-training on driving-related VQA tasks significantly enhances the model’s driving capabilities in road scenarios.

Figure 6 investigates the scaling laws of pre-training data. We found that pre-training with 0.65M data yielded a PDMS improvement of +1.9 (+5.2) for numerical (text) tokenizer. Further pre-training with 6.5M data resulted in an additional PDMS increase of +1.4 (+2.6) for numerical (text) tokenizer compared to the 0.65M data. These results highlight the necessity of further pre-training using driving VQA data and confirm the validity of the data scaling law in WAM-Flow.

Different Simulator-Guided GRPO Settings. We analyze the impact of two key design choices in our simulator-guided GRPO framework: group size and reward weighting. As shown in Table 3, varying the group size (number of candidate trajectories sampled per context) reveals a clear trade-off. While smaller groups (size=2) yield marginal gains, a group size of 3 achieves the optimal balance between exploration diversity and training stability, producing the highest PDMS (90.3). Larger groups (size=4) introduce excessive variance, slightly degrading performance.

We further examine the reward function’s component weights (EP:TTC:Comfort) in Table 3. Extreme weightings—over-prioritizing either safety (5:20:2) or progress (20:5:2)– suboptimally skew the policy, whereas a balanced ratio (5:5:2) best harmonizes these competing objectives, achieving the superior PDMS of 90.3. This indicates that equitable consideration of ego progress, safety, and comfort is crucial for well-rounded driving performance.

Method Paradigm Step PDMS $\uparrow$ Infer Time $\downarrow$ Backbone FSDrive \[FutureSightDrive-2025\] AR - 85.1 10.58s Qwen2-VL-2B \[wang2024qwen2\] Epona \[Epona-2025\] AR + Diff. - 86.2 1.24s DiT-2.5B \[DiT-2023\] ReCogDrive \[RecogDrive-2025\] AR + Diff. - 89.6 0.42s InternVL3-8B \[InternVL3-2025\] Janus-1.5B \[Janus-2025\] AR - - 0.27s - Ours DFM 1 89.1 0.09s Janus-1.5B \[Janus-2025\] 2 89.7 0.19s 3 90.0 0.29s 5 90.3 0.48s 10 90.2 0.94s

Table 6: Intuitive efficiency analysis on NAVSIM.

Coarse-to-fine Sampling Analysis. Table 6 analyzes the coarse-to-fine property of WAM-Flow by varying the number of parallel denoising steps during inference. Increasing the sampling steps from 1 to 5 yields a monotonic improvement in PDMS (89.1 to 90.3), demonstrating that iterative refinement enhances planning quality. Inference time scales approximately linearly with the number of steps, reflecting the parallel nature of the discrete flow matching process. This establishes a flexible trade-off: fewer steps enable faster, coarser plans suitable for real-time constraints, while more steps produce higher-fidelity trajectories.

## 5 Conclusion

We present WAM-Flow, a vision–language–action model that formulates motion planning as discrete flow matching over a structured token space. The framework incorporates a metric-aligned numerical tokenizer to preserve geometric coherence and employs simulator-guided GRPO to enforce safety and performance in closed-loop control. Evaluated on NAVSIM-v1 and v2 benchmarks, WAM-Flow achieves competitive results, demonstrating its ability to generate high-quality trajectories with a flexible trade-off between inference speed and planning fidelity. This work underscores the potential of discrete flow matching for building reliable and scalable autonomous driving systems.

Supplementary Material  

This appendix provides additional experimental results and implementation details to complement the main paper. Specifically, Section A presents extended evaluations on the nuScenes \[nuScenes-dataset-2020\] datasets, along with additional qualitative experiments on NAVSIM. Section B provides pseudocode for the training and inference stages, respectively. Section C elaborates on the evaluation metrics, and Section D discusses implementation specifics. Finally, Section E discuss the limitation and future work.

## Appendix A Additional Experiments

### A.1 nuScenes Results

We evaluate our method on the nuScenes dataset \[nuScenes-dataset-2020\] following the NAVSIM benchmark perspective \[NAVSIM-v1, NAVSIM-v2\], which focuses on collision rate as the primary metric. This emphasis stems from the established finding in NAVSIM that open-loop L2 distance exhibits negligible correlation with closed-loop performance. As shown in Table 7, our method achieves an average collision rate of 0.12% under ST-P3 metrics, matching the performance of the best non-VLA model (UniAD). More notably, under the more comprehensive UniAD metrics, WAM-Flow sets a new state-of-the-art with the lowest average collision rate (0.23%) among all evaluated VLA methods. The model also demonstrates superior short-term safety, achieving a perfect 0.00% collision rate at the 1-second horizon.

<table><tbody><tr><th rowspan="3">Method</th><td rowspan="4">Paradigm</td><td rowspan="4">Backbone</td><td colspan="8">Collision (%) ↓</td></tr><tr><td colspan="4">ST-P3 metrics</td><td colspan="4">UniAD metrics</td></tr><tr><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td></tr><tr><th colspan="11">End-to-End</th></tr><tr><th>PreWorld <cite>[Preworld-2025]</cite></th><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.19</td><td>0.57</td><td>2.65</td><td>1.14</td></tr><tr><th>ST-P3 <cite>[ST-P3-2022]</cite></th><td>-</td><td>-</td><td>0.23</td><td>0.62</td><td>1.27</td><td>0.71</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><th>Ego-MLP <cite>[Ego-MLP-2024]</cite></th><td>-</td><td>-</td><td>0.21</td><td>0.35</td><td>0.58</td><td>0.38</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><th>InsightDrive <cite>[InsightDrive-2025]</cite></th><td>-</td><td>-</td><td>0.09</td><td>0.10</td><td>0.27</td><td>0.15</td><td>0.08</td><td>0.15</td><td>0.84</td><td>0.36</td></tr><tr><th>VAD-v2 <cite>[VADv2-2024]</cite></th><td>-</td><td>-</td><td>0.07</td><td>0.10</td><td>0.24</td><td>0.14</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><th>UniAD <cite>[UniAD-2023]</cite></th><td>-</td><td>-</td><td>0.04</td><td>0.08</td><td>0.23</td><td>0.12</td><td>0.05</td><td>0.17</td><td>0.71</td><td>0.31</td></tr><tr><th colspan="11">End-to-End VLA</th></tr><tr><th>Epona <cite>[Epona-2025]</cite></th><td>AR + Diff.</td><td>DiT-2.5B <cite>[DiT-2023]</cite></td><td>0.05</td><td>0.22</td><td>0.85</td><td>0.96</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><th>OmniDrive <cite>[OminiDrive-2025]</cite></th><td>AR</td><td>LLaVA-7B <cite>[LLaVA]</cite></td><td>0.04</td><td>0.46</td><td>2.32</td><td>0.94</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><th>DriveVLM <cite>[tian2024drivevlm]</cite></th><td>AR</td><td>Qwen2-VL-7B <cite>[wang2024qwen2]</cite></td><td>0.10</td><td>0.22</td><td>0.45</td><td>0.27</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><th>GPT-Driver <cite>[GPT-Driver-2023]</cite></th><td>AR</td><td>GPT-4 <cite>[GPT-4-2023]</cite></td><td>0.04</td><td>0.12</td><td>0.36</td><td>0.17</td><td>0.07</td><td>0.15</td><td>1.10</td><td>0.44</td></tr><tr><th>AutoVLA <cite>[zhou2025autovla]</cite></th><td>AR</td><td>Qwen2.5-3B <cite>[Qwen-25]</cite></td><td>0.13</td><td>0.18</td><td>0.28</td><td>0.20</td><td>0.14</td><td>0.25</td><td>0.53</td><td>0.31</td></tr><tr><th>DME-Driver <cite>[DME-Driver-2025]</cite></th><td>AR</td><td>LLaVA-7B <cite>[LLaVA]</cite></td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.05</td><td>0.28</td><td>0.55</td><td>0.29</td></tr><tr><th>Ours</th><td>DFM</td><td>Janus-1.5B <cite>[Janus-2025]</cite></td><td>0.04</td><td>0.10</td><td>0.23</td><td>0.12</td><td>0.00</td><td>0.10</td><td>0.60</td><td>0.23</td></tr></tbody></table>

Table 7: End-to-end motion planning performance on the nuScenes \[nuScenes-dataset-2020\] dataset. We sort previous methods according to the average collision rate. Abbreviation: Diff.(Diffusion), AR (autoregressive), DFM (discrete flow matching).

<table><tbody><tr><th rowspan="2">Hyperparameter</th><td>Stage 1</td><td>Stage 2</td><td>Stage 3</td><td>Stage 4</td></tr><tr><td>Embedding Training</td><td>Pre-training</td><td>Supervised Fine-tuning</td><td>Reinforcement Learning</td></tr><tr><th>Training Modules</th><td>Numerical Tokenizer</td><td>VLA</td><td>VLA</td><td>VLA</td></tr><tr><th>Training Parameters</th><td>0.4B</td><td>1.5B</td><td>1.5B</td><td>1.5B</td></tr><tr><th>Training Data</th><td>nuPlan (668K)</td><td>VQA (6.5M)</td><td>nuPlan (668K)</td><td>NAVSIM (103K)</td></tr><tr><th>Loss</th><td><math><semantics><mrow><msub><mi>ℒ</mi> <mi>CE</mi></msub> <mo>+</mo> <msub><mi>ℒ</mi> <mi>num</mi></msub></mrow> <annotation>\mathcal{L}_{\mathrm{CE}}+\mathcal{L}_{\mathrm{num}}</annotation></semantics></math></td><td><math><semantics><msub><mi>ℒ</mi> <mi>CE</mi></msub> <annotation>\mathcal{L}_{\mathrm{CE}}</annotation></semantics></math></td><td><math><semantics><msub><mi>ℒ</mi> <mi>CE</mi></msub> <annotation>\mathcal{L}_{\mathrm{CE}}</annotation></semantics></math></td><td><math><semantics><msub><mi>ℒ</mi> <mi>GRPO</mi></msub> <annotation>\mathcal{L}_{\mathrm{GRPO}}</annotation></semantics></math></td></tr><tr><th>Training Epochs</th><td>4</td><td>3</td><td>2</td><td>0.5</td></tr><tr><th>Batch Size</th><td>80</td><td>256</td><td>64</td><td>32</td></tr><tr><th>Optimizer</th><td>Adam</td><td>Adam</td><td>Adam</td><td>Adam</td></tr><tr><th>Learning Rate</th><td><math><semantics><mrow><mn>1</mn> <mo>×</mo> <msup><mn>10</mn> <mrow><mo>−</mo> <mn>5</mn></mrow></msup></mrow> <annotation>1\times 10^{-5}</annotation></semantics></math></td><td><math><semantics><mrow><mn>1</mn> <mo>×</mo> <msup><mn>10</mn> <mrow><mo>−</mo> <mn>5</mn></mrow></msup></mrow> <annotation>1\times 10^{-5}</annotation></semantics></math></td><td><math><semantics><mrow><mn>5</mn> <mo>×</mo> <msup><mn>10</mn> <mrow><mo>−</mo> <mn>6</mn></mrow></msup></mrow> <annotation>5\times 10^{-6}</annotation></semantics></math></td><td><math><semantics><mrow><mn>1</mn> <mo>×</mo> <msup><mn>10</mn> <mrow><mo>−</mo> <mn>6</mn></mrow></msup></mrow> <annotation>1\times 10^{-6}</annotation></semantics></math></td></tr><tr><th>Learning Rate Scheduler</th><td>constant</td><td>constant</td><td>cosine annealing</td><td>cosine annealing</td></tr><tr><th>Warm-up Steps</th><td>0</td><td>0</td><td>500</td><td>500</td></tr><tr><th>Gradient Accumulation Steps</th><td>1</td><td>1</td><td>1</td><td>1</td></tr></tbody></table>

Table 8: Key hyperparameters for different training stages.

### A.2 NAVSIM Qualitative Results

Figure 8, 9 and 10 visualizes 1-, 3- and 5-step results on NAVSIM, respectively. For straightforward driving scenarios (Figure 8), WAM-Flow generates acceptable trajectories with only 1-step denoising. For relatively complex scenarios (Figure 10), our method predicts reasonable results through a 5-step parallel coarse-to-fine process.

## Appendix B Pseudocode for Training and Inference

Algorithm 1 and 2 respectively describe the training and inference procedure.

Algorithm 1 Training

model parameters $\theta$, time schedule $\beta_{t}$

Optimized parameters $\theta^{*}$

Initialize model parameters $\theta$

while not converged do

  Sample batch $x_{1}\sim q(x)$ $\triangleright$ Trajectory

  Sample $t\sim\mathcal{U}[0,1]$ $\triangleright$ Continuous time sampling

   $p_{t}(x|x_{1})=\mathrm{softmax}(-\beta_{t}\cdot d(x,x_{1}))$ $\triangleright$ Compute transition probabilities

   $x_{t}\sim p_{t}(x|x_{1})$ $\triangleright$ Sample noisy tokens

   $p_{1|t}^{\theta}(\cdot|x_{t})=\mathrm{model}_{\theta}(x_{t},c)$ $\triangleright$ Compute conditional distribution

   $\mathcal{L}_{\mathrm{CE}}=-\mathbb{E}\left[\sum_{i=1}^{D}\log p_{1|t}^{\theta,i}(x_{1}^{i}|x_{t})\right]$ $\triangleright$ Compute loss

  Update $\theta$ via gradient descent on $\mathcal{L}_{\mathrm{CE}}$

end while

Algorithm 2 Inference

Number of inference steps $n$

Generated token sequence $x_{1}$

$h\leftarrow 1/n$ $\triangleright$ Step size for Euler discretization

Initialize $x_{0}$: for each coordinate $i$, sample $x_{0}^{i}$ uniformly from vocabulary

for $k=0,1,\dots,n-1$ do

   $t\leftarrow k\cdot h$ $\triangleright$ Current time in $[0,1)$

  for $i=1$ to $D$ in parallel do $\triangleright$ Parallel processing of all coordinates

   Compute posterior: $p_{1|t}^{\theta,i}(\cdot|x_{t})\leftarrow\mathrm{model}_{\theta}(x_{t},c)$

   Sample target: $x_{1}^{i}\sim p_{1|t}^{\theta,i}(\cdot|x_{t})$

   Compute total transition rate: $\lambda_{i}\leftarrow\sum_{y^{i}\neq x_{t}^{i}}u_{t}^{i}(y^{i},x_{t}^{i}|x_{1}^{i})$

   Sample threshold: $Z_{i}\sim\mathcal{U}[0,1]$

   if $Z_{i}\leq 1-e^{-h\lambda_{i}}$ then $\triangleright$ Transition occurs with probability $1-e^{-h\lambda_{i}}$

     Sample new token: $x_{t+h}^{i}\sim\frac{u_{t}^{i}(\cdot,x_{t}^{i}|x_{1}^{i})}{\lambda_{i}}$

   else

     Retain current token: $x_{t+h}^{i}\leftarrow x_{t}^{i}$

   end if

  end for

  Advance time: $x_{t}\leftarrow x_{t+h}$

end for

return $x_{1}$ $\triangleright$ Final denoised token sequence at $t=1$

## Appendix C Detailed Explanation for Metrics

This section provides detailed definitions of the evaluation metrics used in our experiments.

### C.1 NAVSIM-v1 Metrics

For NAVSIM-v1 \[NAVSIM-v1\], the primary evaluation metric is the Predictive Driver Model Score (PDMS), which integrates five key performance indicators:

$$
\mathrm{PDMS}=\mathrm{NC}\times\mathrm{DAC}\times\frac{(5\times\mathrm{TTC}+2\times\mathrm{C}+5\times\mathrm{EP})}{12}
$$
- No at-fault Collision (NC): Penalizes collisions based on fault assignment. NC=1 indicates no at-fault collisions, NC=0.5 indicates one fault collision with static objects, and NC=0 indicates multiple fault collisions.
- Drivable Area Compliance (DAC): Measures adherence to drivable areas (lanes, parking areas). DAC=1 when the ego bounding box remains entirely within drivable areas, and DAC=0 when any corner exits designated areas.
- Ego Progress (EP): Quantifies navigation goal achievement as the ratio of actual progress to a search-based safe upper bound derived from PDM-Closed trajectories. The ratio is clipped to \[0,1\], with low or negative values discarded.
- Time-to-Collision (TTC): Encourages maintenance of safe distances from other vehicles. TTC=1 when the minimum time-to-collision exceeds 0.9 seconds, and 0 otherwise.
- Comfort (C): Assesses kinematic constraints including acceleration and jerk. C=1 when all predefined thresholds are satisfied, and 0 upon any violation.

### C.2 NAVSIM-v2 Metrics

For NAVSIM-v2 \[NAVSIM-v2\], the Extended Predictive Driver Model Score (EPDMS) incorporates additional safety and compliance measures:

$$
\begin{aligned} \mathrm{EPDMS}&=\mathrm{NC}\times\mathrm{DAC}\times\mathrm{DDC}\times\mathrm{TL}\times\\
&\quad\frac{(5\times\mathrm{TTC}+2\times\mathrm{C}+5\times\mathrm{EP}+5\times\mathrm{LK}+5\times\mathrm{EC})}{22}\end{aligned}
$$
- Driving Direction Compliance (DDC): Penalizes reverse driving behavior. DDC=1 for reverse distance $<2$ m, DDC=0.5 for $2-6$ m, and DDC $=0$ for $>6$ m.
- Traffic Light Compliance (TLC): Measures obedience to traffic signals. TLC $=1$ when traffic rules are followed, and 0 upon violations.
- Lane Keeping (LK): Evaluates lateral positioning relative to lane centerlines, scored continuously from 0 to 1.
- History Comfort (HC): Assesses trajectory consistency with historical motion patterns, ranging from 0 to 1.
- Extended Comfort (EC): Compares planned trajectories across consecutive frames for dynamic consistency, scored from 0 to 1.

### C.3 nuScenes Metrics

For nuScenes, we follow the NAVSIM \[NAVSIM-v1, NAVSIM-v2\] perspective, focusing only on the collision rate.

## Appendix D Implementation Details

In Table 8, we show the key hyperparameters for different training steps, including training modules, parameters, data, loss, epochs, batch sizes, optimizer, learning rate, learning rate scheduler, warm-up and gradient accumulation steps.

## Appendix E Limitation and Future Work

While WAM-Flow demonstrates promising results, several limitations warrant attention. First, our evaluation is conducted primarily in simulation environments (NAVSIM, nuScenes), which may not fully capture the complexities of real-world driving scenarios. Second, the GRPO reward is designed for and evaluated in simulation; its safety and performance terms require careful redesign to bridge the sim-to-real gap. Third, the model is trained and validated on existing benchmarks, which may not encompass the full long-tail distribution of real-world driving scenarios.

Future work will explore several directions. We plan to extend the framework to support variable-horizon planning and incorporate multi-modal sensor inputs (e.g., LiDAR, radar) for enhanced robustness. We also plan to investigate learning a world model as a more generalizable alternative to simulator-based rewards. Finally, real-world deployment and testing will be essential to validate the model’s performance under actual driving conditions.

![[easy_case.png|Refer to caption]]

Figure 8: For straightforward driving scenarios on NAVSIM, our method achieves acceptable outcomes with just 1-step denoising.

![[medium_case.png|Refer to caption]]

Figure 9: Visualization of the 3-step refinement results on NAVSIM.

![[hard_case.png|Refer to caption]]

Figure 10: For relatively complex scenarios on NAVSIM, our model generates reasonable results through a 5-step coarse-to-fine trajectory prediction process.
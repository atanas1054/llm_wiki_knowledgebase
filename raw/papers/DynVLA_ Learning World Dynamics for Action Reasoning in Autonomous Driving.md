---
title: "DynVLA: Learning World Dynamics for Action Reasoning in Autonomous Driving"
source: "https://arxiv.org/html/2603.11041v1"
author:
published:
created: 2026-04-28
description:
tags:
  - "clippings"
---
Shuyao Shang    Bing Zhan    Yunfei Yan    Yuqi Wang    Yingyan Li    Yasong An    Xiaoman Wang    Jierui Liu    Lu Hou    Lue Fan    Zhaoxiang Zhang    Tieniu Tan

###### Abstract

We propose DynVLA, a driving VLA model that introduces a new CoT paradigm termed Dynamics CoT. DynVLA forecasts compact world dynamics before action generation, enabling more informed and physically grounded decision-making. To obtain compact dynamics representations, DynVLA introduces a Dynamics Tokenizer that compresses future evolution into a small set of dynamics tokens. Considering the rich environment dynamics in interaction-intensive driving scenarios, DynVLA decouples ego-centric and environment-centric dynamics, yielding more accurate world dynamics modeling. We then train DynVLA to generate dynamics tokens before actions through SFT and RFT, improving decision quality while maintaining latency-efficient inference. Compared to Textual CoT, which lacks fine-grained spatiotemporal understanding, and Visual CoT, which introduces substantial redundancy due to dense image prediction, Dynamics CoT captures the evolution of the world in a compact, interpretable, and efficient form. Extensive experiments on NAVSIM, Bench2Drive, and a large-scale in-house dataset demonstrate that DynVLA consistently outperforms Textual CoT and Visual CoT methods, validating the effectiveness and practical value of Dynamics CoT.

Machine Learning, ICML

<sup>✉</sup> <sup>✉</sup>

![[x1 26.png|Refer to caption]]

Figure 1: Comparison of different CoT paradigms in autonomous driving VLA models. (a) Textual CoT suffers from limited spatiotemporal understanding and high inference latency due to long textual reasoning traces. (b) Visual CoT introduces substantial redundancy and computational overhead from pixel-level generation. (c) Dynamics CoT compresses future dynamics into a small set of tokens, achieving latency-efficient inference with compact reasoning and accurate spatiotemporal modeling.

## 1 Introduction

End-to-end autonomous driving [^14] [^52] [^53] [^76] [^67] has recently embraced the Vision–Language–Action (VLA) paradigm [^16] [^69], which offers richer cross-modal grounding and improved decision quality. Inspired by the cognitive process where human drivers reason causally before acting, VLA models also benefit from this “reason-then-act” process, making Chain-of-Thought (CoT) [^62] well-suited for driving tasks. By reasoning about why certain maneuvers are required in interaction-intensive and rule-constrained traffic, VLA models can produce more reliable driving decisions in long-horizon, safety-critical scenarios.

Within VLA-based autonomous driving, the dominant design for CoT is Textual CoT (Fig. 1a), which performs reasoning in the text space and provides high-level decision logic [^18] [^31]. However, driving maneuvers depend on fine-grained spatiotemporal relationships in a complex, physically constrained world, which discrete linguistic representations struggle to capture. To address this issue, recent works have explored Visual CoT (Fig. 1b), which predicts future visual frames and subsequently generates actions, enabling spatiotemporal reasoning in the pixel space [^74] [^80]. Although Visual CoT is more expressive in representing spatiotemporal relationships, the model must predict both decision-irrelevant background and texture-level details, which increases reasoning redundancy and learning difficulty. Furthermore, both Textual and Visual CoT require generating a large number of reasoning tokens, leading to substantial inference latency.

To overcome these limitations, we propose DynVLA, which introduces a new CoT paradigm termed Dynamics Chain-of-Thought (Dynamics CoT). DynVLA first compresses future dynamics into compact tokens, and then predicts these dynamics tokens before action generation. Compared to Textual CoT, Dynamics CoT models the evolution of spatiotemporal states beyond symbolic textual reasoning. Compared to Visual CoT, it avoids redundant reasoning by encoding scene dynamics only. In addition, this compact representation models the state transition between consecutive observations, and therefore only requires a small number of tokens to capture future dynamics. This substantially shortens the reasoning trace and reduces inference latency by over an order of magnitude compared to Textual or Visual CoT.

To represent such dynamics in a compact and learnable form, prior studies have explored latent action tokenizer for embodied scenarios [^71]. However, driving scenes involve more pronounced ego-viewpoint transformations and richer dynamics from multiple interacting agents. To address this gap, we introduce a Dynamics Tokenizer tailored for driving scenarios. It first factorizes dynamics into two decoupled factors: ego-centric dynamics, arising from the ego vehicle’s own motion, and environment-centric dynamics, originating from external changes such as other traffic participants. However, dynamics cannot be naturally disentangled, and the learned representation may become physically ambiguous. For instance, ego forward motion can be confused with a leading vehicle moving backward. We therefore introduce physical regularization to align ego-centric dynamics with ego motion. In addition, we note that different views (e.g., image and BEV) should share the same underlying dynamics representation. We thus introduce cross-view consistency regularization, leading to semantically aligned dynamics for planning. Finally, following the standard training strategy of CoT-based models, we apply supervised fine-tuning (SFT) and reinforcement fine-tuning (RFT) on Dynamics CoT, enabling reasoning in the dynamics space and improving decision quality.

We conduct comprehensive evaluations on a real-world benchmark NAVSIM [^9], a closed-loop benchmark Bench2Drive [^22], and a massive in-house dataset. Experimental results demonstrate that Dynamics CoT outperforms both non-CoT VLA methods and Textual CoT or Visual CoT methods, validating its effectiveness and practical value in autonomous driving.

The contributions of this paper can be summarized as follows: (1) We propose DynVLA, which introduces a new CoT paradigm called Dynamics CoT for autonomous driving VLA models. Dynamics CoT reasons over compact future dynamics that both capture spatiotemporal evolution and reduce reasoning redundancy. (2) We identify that naive dynamics tokenization tends to entangle ego dynamics and environment dynamics, and address this by explicitly decoupling them with physically grounded regularization. (3) We visualize the transferability of learned dynamics tokens and conduct extensive experiments and empirical analyses across multiple benchmarks, demonstrating the effectiveness of Dynamics CoT.

## 2 Related Works

### 2.1 VLA models for End-to-End Autonomous Driving

Vision-Language-Action (VLA) models have proven effective in robotics [^26] [^85] [^2] [^3] [^60], and recent work has increasingly transferred this paradigm to end-to-end autonomous driving. Early explorations primarily adopt an LLM backbone [^40] [^49] [^68], while more recent systems move toward VLM-based policies [^83] [^17] [^81]. Beyond directly regressing trajectories, DiffVLA [^24] proposes VLM-guided diffusion planning for multi-modal trajectory generation, and ORION [^11] addresses the mismatch between semantic reasoning space and continuous action space with a generative planner. To better inject domain knowledge from driving, ReCogDrive [^33] introduces a hierarchical pipeline that distills human driving cognition into the VLM, while DriveVLA-W0 [^30] leverages world-model pretraining to provide dense supervision.

![[x2 24.png|Refer to caption]]

Figure 2: Overview of the DynVLA. (a) Given adjacent image observations, a dynamics encoder extracts ego-centric and environment-centric dynamics, which are discretized via VQ codebooks. Then, the ego-centric dynamics are regularized by the GT ego action, and the combined dynamics are decoded to reconstruct the future image and BEV map conditioned on each current state. (b) DynVLA is supervised to first generate discrete dynamics tokens followed by action tokens, forming structured Dynamics CoT modeling.

### 2.2 Chain-of-Thought in VLA models

Chain-of-Thought (CoT) [^62] is known to improve the performance of LLM by introducing extra thinking steps before producing final answers [^5] [^61] [^65], and robotics research has explored introducing CoT into VLA models. ECoT [^73] explicitly produces structured textual reasoning grounded in the physical environment, while OneTwoVLA [^37] further unifies fast control and slow reasoning by adaptively invoking CoT. Recent works introduce Visual CoT [^79] [^39], which synthesizes goal-conditioned future images and subsequently generates actions. VLA in autonomous driving also incorporates CoT-style reasoning [^58] [^66]. EMMA [^18] recasts multiple driving tasks into language space and can generate trajectories together with language-based reasoning. AutoDrive- $R^{2}$ [^72] incentivizes reasoning and self-reflection by combining CoT with reinforcement learning. AutoVLA [^84] proposes adaptive reasoning to reduce unnecessary CoT while preserving it in complex scenarios. Following the recent success of world models in autonomous driving [^82] [^35], FSDrive [^74] introduces a world-model-based Visual CoT and generates future visual states as intermediate reasoning steps. In contrast, we focus on a compact dynamics representation that captures spatiotemporal relationships while avoiding redundant generation, enabling efficient planning.

## 3 Method

In this section, we present DynVLA, as illustrated in Fig. 3. DynVLA first trains a Dynamics Tokenizer to extract discrete dynamics tokens (Sec. 3.1). It then performs supervised fine-tuning (SFT) on Dynamics CoT sequences (Sec. 3.2) and further refines the VLA policy via reinforcement fine-tuning (RFT) (Sec. 3.3).

![[x3 24.png|Refer to caption]]

Figure 3: Overview of the training pipeline for DynVLA. DynVLA first learns a Dynamics Tokenizer by reconstructing future states from adjacent frames, producing discrete dynamics tokens. It then performs SFT on Dynamics CoT, training the model to generate dynamics tokens before action tokens. Finally, the policy is optimized via RFT with trajectory-level reward and KL regularization.

### 3.1 Dynamics Tokenizer

#### Encoder with Decoupled Dynamics.

Driving scenarios exhibit significant environment dynamics in addition to ego motion. Therefore, we explicitly decouple the dynamics representation into ego-centric and environment-centric tokens. As shown in Fig. 2a, we first map the input images $O_{t}$ and $O_{t+1}$ into patch sequences $\mathbf{x}_{t}$ and $\mathbf{x}_{t+1}$ using a ViT patchifier [^10], and encode $(\mathbf{x}_{t},\mathbf{x}_{t+1})$ with a Dynamics Encoder $E_{\text{dyn}}$, which consists of $L_{\text{Enc}}$ Transformer layers [^57]. We then introduce two sets of learnable queries, denoted as $Q_{\text{ego}}\in\mathbb{R}^{N_{\text{ego}}\times d}$ and $Q_{\text{env}}\in\mathbb{R}^{N_{\text{env}}\times d}$, where $N_{\text{ego}}$ and $N_{\text{env}}$ denote the number of ego-centric and environment-centric dynamics tokens, and $d$ is the feature dimension. In practice, both $N_{\text{ego}}$ and $N_{\text{env}}$ are kept small. These queries aggregate the dynamics representations as

$$
(e^{\text{ego}}_{t},e^{\text{env}}_{t})=E_{\text{dyn}}(\mathbf{x}_{t},\mathbf{x}_{t+1};Q_{\text{ego}},Q_{\text{env}}),
$$

where $e^{\text{ego}}_{t}\in\mathbb{R}^{N_{\text{ego}}\times d_{\text{VQ}}}$ and $e^{\text{env}}_{t}\in\mathbb{R}^{N_{\text{env}}\times d_{\text{VQ}}}$ denote the continuous ego-centric and environment-centric dynamics representations, and $d_{\text{VQ}}$ is the codebook feature dimension. We maintain two separate VQ codebooks for the ego and environment branches denoted as $\mathcal{C}_{\text{ego}}=\{c^{\text{ego}}_{i}\}_{i=1}^{M_{\text{ego}}}$ and $\mathcal{C}_{\text{env}}=\{c^{\text{env}}_{j}\}_{j=1}^{M_{\text{env}}}$, where each code $c^{\text{ego}}_{i},c^{\text{env}}_{j}\in\mathbb{R}^{d_{\text{VQ}}}$, and $M_{\text{ego}}$ and $M_{\text{env}}$ denote the codebook sizes. Then, continuous dynamics $e^{\text{ego}}_{t}$ and $e^{\text{env}}_{t}$ are discretized via nearest-neighbor codebook assignment [^56], producing discrete tokens $\mathcal{D}_{t}^{\text{ego}}$ and $\mathcal{D}_{t}^{\text{env}}$, which are concatenated to finally form the Dynamics Tokens as

$$
\mathcal{D}_{t}=[\mathcal{D}_{t}^{\text{ego}},\,\mathcal{D}_{t}^{\text{env}}].
$$

The discrete tokens in $\mathcal{D}_{t}$ are subsequently mapped back to their continuous embeddings via codebook lookup, yielding $z_{t}\in\mathbb{R}^{(N_{\text{ego}}+N_{\text{env}})\times d_{\text{VQ}}}$, which serves as the input to the dynamics decoder.

#### Decoder with Action-based Regularization.

After obtaining encoded dynamics features, we employ a decoder to reconstruct future observations, providing supervision for learning the Dynamics Tokens. During decoding, we also condition on the current observation, which relieves the discrete tokens from encoding static background or texture details. However, learning dynamics purely from reconstruction remains under-constrained and can lead to codebook collapse. To address this, we introduce an action-based regularization that aligns ego-centric dynamics with the ego motion. Specifically, we use a two-layer MLP action decoder to predict the ego action $\hat{\mathbf{a}}_{t\to t+1}$ from the ego-centric dynamics $\mathcal{D}_{t}^{\text{ego}}$ and penalize the discrepancy between the predicted action and the ground-truth action by $\mathcal{L}_{\text{act-reg}}=\left\lVert\hat{\mathbf{a}}_{t\to t+1}-\mathbf{a}_{t\to t+1}\right\rVert_{2}^{2}$. This encourages the ego-centric branch to explicitly explain ego motion, thereby promoting disentangled learning between ego-centric and environment-centric dynamics.

#### Decoder with Cross-view Consistency Regularization.

We note that desirable environment dynamics should capture the same underlying scene evolution across different representations. Therefore, we impose a cross-view consistency regularization by requiring the same Dynamics Tokens to predict both the future image and the future BEV map conditioned on their respective current observations. This enforces semantic consistency between the image and BEV spaces, yielding more coherent environment-centric dynamics. As shown in Fig. 2a, we map the current image $O_{t}$ and the current BEV map $BEV_{t}$ into patch sequences $\mathbf{x}_{t}$ and $\mathbf{b}_{t}$ via a ViT patchifier, and then perform conditional decoding with two modality-specific dynamics decoders: an image decoder $D^{\text{img}}_{\text{dyn}}$ and a BEV decoder $D^{\text{bev}}_{\text{dyn}}$, each composed of $L_{\text{Dec}}$ Transformer layers. Both decoders are conditioned on the same dynamics representation $z_{t}$ and their respective patch sequences, and each predicts its corresponding future state as

$$
\widehat{O}_{t+1}=D^{\text{img}}_{\text{dyn}}(\mathbf{x}_{t},z_{t}),\ \widehat{BEV}_{t+1}=D^{\text{bev}}_{\text{dyn}}(\mathbf{b}_{t},z_{t}).
$$

#### Dynamics Tokenizer Training.

The Dynamics Tokenizer is trained by minimizing the reconstruction loss, the VQ-VAE loss, and the regularizations loss. The image reconstruction loss $\mathcal{L}_{\text{recon}}^{\text{img}}$ combines a mean squared error loss and a perceptual similarity loss [^78] to capture both low-frequency structural consistency and high-level semantic similarity, while the BEV reconstruction loss $\mathcal{L}_{\text{recon}}^{\text{bev}}$ is a cross-entropy loss. The overall training objective of the Dynamics Tokenizer is given by

$$
\mathcal{L}=\mathcal{L}_{\text{recon}}^{\text{img}}+\lambda_{\text{bev}}\mathcal{L}_{\text{recon}}^{\text{bev}}+\lambda_{\text{vq}}\mathcal{L}_{\text{VQ}}+\lambda_{\text{act-reg}}\mathcal{L}_{\text{act-reg}},
$$

where $\mathcal{L}_{\text{VQ}}$ is the vector-quantization loss [^56], and $\lambda_{\text{bev}},\lambda_{\text{vq}},\lambda_{\text{act-reg}}$ are the corresponding weighting coefficients.

### 3.2 SFT on Dynamics CoT

#### Structured Dynamics CoT Sequence.

To realize Dynamics CoT, we perform supervised fine-tuning (SFT) on structured dynamics sequences (as shown in Fig. 2b). Given the current and next $K$ frame images $O_{t:t+K}$ and letting $E_{\text{dyn}}$ denote the trained dynamics encoder, the dynamics tokens at step $t+k$ are defined as

$$
\mathcal{D}_{t+k}=E_{\text{dyn}}(O_{t+k},O_{t+k+1}),\quad 0\leq k\leq K-1,
$$

where each $\mathcal{D}_{t+k}$ consists of $N_{\text{ego}}+N_{\text{env}}$ discrete dynamics tokens. To explicitly delineate the dynamics reasoning sequence, we introduce two special tokens $\langle\texttt{BOD}\rangle$ and $\langle\texttt{EOD}\rangle$, which mark the beginning and end of dynamics reasoning. For action generation, we encode the continuous action into a discrete action token sequence $\mathcal{A}_{t:t+N-1}$ using the FAST tokenizer [^45], where $N$ denotes the length of the action token sequence. Similarly, we introduce $\langle\texttt{BOA}\rangle$ and $\langle\texttt{EOA}\rangle$ to indicate the beginning and end of the action generation sequence. For each training sample, the target output sequence is organized as

$$
\mathbf{y}=[\langle\texttt{BOD}\rangle,\mathcal{D}_{t:t+K-1},\langle\texttt{EOD}\rangle,\;\langle\texttt{BOA}\rangle,\mathcal{A}_{t:t+N-1},\langle\texttt{EOA}\rangle].
$$

#### SFT Training.

Given the target sequence, we train the model by maximizing the likelihood of the output tokens conditioned on the observation and instruction context. Specifically, the model input at time $t$ is denoted as $\mathbf{c}_{t}=\{O_{t},O_{t-1},T_{t},S_{t}\}$, where $O_{t}$ is the current image observation, $O_{t-1}$ is the previous image observation, $T_{t}$ is the text instruction, and $S_{t}$ represents the ego state. We adopt the standard next-token prediction loss [^57] and minimize the negative log-likelihood over the dynamics reasoning sequence and the action generation sequence, which is formulated as

$$
\mathcal{L}_{\text{dyn}}=-\sum_{k=0}^{K-1}\log p_{\theta}\!\left(\mathcal{D}_{t+k}\mid\mathcal{D}_{t:t+k-1},\mathbf{c}_{t}\right),
$$
 
$$
\mathcal{L}_{\text{act}}=-\sum_{n=0}^{N-1}\log p_{\theta}\!\left(\mathcal{A}_{t+n}\mid\mathcal{A}_{t:t+n-1},\mathcal{D}_{t:t+K-1},\mathbf{c}_{t}\right).
$$

The overall objective for Dynamics CoT SFT is given by

$$
\mathcal{L}_{\text{SFT}}=\mathcal{L}_{\text{dyn}}+\lambda_{\text{act}}\mathcal{L}_{\text{act}},
$$

where $\lambda_{\text{act}}$ is a weighting coefficient. Through this procedure, the pretrained model explicitly learns a causal generation order of dynamics reasoning followed by action generation, and treats the reasoned dynamics as an intermediate variable for decision making.

### 3.3 RFT on Dynamics CoT

While Dynamics CoT SFT teaches the model to explicitly reason future dynamics before acting, the learning of actions remains purely imitation-based. However, imitation learning is prone to generate human-like but unsafe trajectories [^48] and tends to produce averaged and suboptimal motion plans [^33]. In addition, recent studies have shown that applying reinforcement learning to CoT-based models can provide outcome-driven incentives beyond SFT [^13]. Thus, to address the limitations of imitation learning and in accordance with common practice in reasoning models, we introduce reinforcement fine-tuning (RFT) [^50] [^13] to further enhance safety and decision quality.

#### Reward Design.

For each trajectory, we adopt the PDM Score (PDMS) [^9] as the trajectory-level reward $r_{\text{traj}}$, which is a scalar in $[0,1]$. In addition, to stabilize RL training and enforce the model output to follow the CoT template, we introduce a format reward $r_{\text{fmt}}\in\{0,1\}$, which is $1$ if the generated sequence satisfies the required token organization, and $0$ otherwise. The final reward is computed as a weighted combination of trajectory reward and format reward: $r=r_{\text{traj}}+\lambda_{\text{fmt}}\,r_{\text{fmt}}$, where $\lambda_{\text{fmt}}$ is a weighting coefficient.

#### RFT Training.

We optimize the policy using Group Relative Policy Optimization (GRPO) [^50]. For each training sample given $\mathbf{c}_{t}$, we roll out $G$ candidate sequences $\{o_{i}\}_{i=1}^{G}$ and compute their corresponding rewards $\{r_{i}\}_{i=1}^{G}$. Then, the GRPO objective can be written as

$$
\displaystyle\mathcal{J}_{\text{GRPO}}(\theta)
$$
 
$$
\displaystyle=\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_{i}|}\sum_{t=1}^{|o_{i}|}\min\!\Big(\rho_{i,t}(\theta)\,\hat{A}_{i,t},
$$
$$
\displaystyle\qquad\mathrm{clip}\ \!\big(\rho_{i,t}(\theta),1-\epsilon,1+\epsilon\big)\,\hat{A}_{i,t}\Big)
$$
 
$$
\displaystyle\quad-\beta\,D_{\text{KL}}\!\left(\pi_{\theta}\,\|\,\pi_{\text{ref}}\right),
$$

where $\hat{A}_{i}=\frac{r_{i}-\mathrm{mean}(\{r_{j}\}_{j=1}^{G})}{\mathrm{std}(\{r_{j}\}_{j=1}^{G})}$, $\rho_{i,t}(\theta)=\frac{\pi_{\theta}\!\left(o_{i,t}\mid\mathbf{c}_{t},\,o_{i,<t}\right)}{\pi_{\theta_{\text{old}}}\!\left(o_{i,t}\mid\mathbf{c}_{t},\,o_{i,<t}\right)}$, $\epsilon$ is the clipping range, $\pi_{\mathrm{ref}}$ is a frozen reference model from SFT, and $\beta$ controls the KL regularization strength. Through GRPO-based RFT, the model can further improve planning safety and decision quality while preserving the structured generation of Dynamics CoT.

## 4 Experiments

### 4.1 Experimental Setup

We conduct comprehensive experiments on three benchmarks: a real-world benchmark NAVSIM [^9], a closed-loop benchmark Bench2Drive [^22], and a large-scale in-house dataset containing 700k frames. Details regarding datasets and evaluation metrics are provided in Appendix A, and implementation details are provided in Appendix B.

### 4.2 Main Results

#### NAVSIM Results.

Table 1 reports the performance comparison on the NAVSIM benchmark [^9]. Among all evaluated methods, DynVLA achieves the highest PDMS, outperforming both traditional end-to-end methods and recent VLA-based methods. Notably, compared with existing VLA methods that use textual or visual CoT, DynVLA yields better planning quality, indicating that reasoning over future dynamics provides more effective and reliable decision-making guidance.

#### Bench2Drive Results.

Table 2 presents the results on the Bench2Drive benchmark [^22], which evaluates closed-loop driving performance in long-horizon, interactive scenarios. Compared with strong baselines and recent VLA-based methods, DynVLA achieves the best performance across all metrics, demonstrating the advantages of Dynamics CoT in challenging closed-loop environments.

Table 1: Comparison on NAVSIM Benchmark. The best results are denoted by bold and the second best are denoted by underline.

<table><tbody><tr><td>Method</td><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>C <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>Human</td><td>100.0</td><td>100.0</td><td>100.0</td><td>99.9</td><td>87.5</td><td>94.8</td></tr><tr><td colspan="7">Traditional End-to-End Methods</td></tr><tr><td>VADv2 <sup><a href="#fn:6">6</a></sup></td><td>97.2</td><td>89.1</td><td>91.6</td><td>100</td><td>76.0</td><td>80.9</td></tr><tr><td>UniAD <sup><a href="#fn:14">14</a></sup></td><td>97.8</td><td>91.9</td><td>92.9</td><td>100</td><td>78.8</td><td>83.4</td></tr><tr><td>TransFuser <sup><a href="#fn:8">8</a></sup></td><td>97.7</td><td>92.8</td><td>92.8</td><td>100</td><td>79.2</td><td>84.0</td></tr><tr><td>PARA-Drive <sup><a href="#fn:63">63</a></sup></td><td>97.9</td><td>92.4</td><td>93.0</td><td>99.8</td><td>79.3</td><td>84.0</td></tr><tr><td>LAW <sup><a href="#fn:29">29</a></sup></td><td>96.4</td><td>95.4</td><td>88.7</td><td>99.9</td><td>81.7</td><td>84.6</td></tr><tr><td>Epona <sup><a href="#fn:77">77</a></sup></td><td>97.9</td><td>95.1</td><td>93.8</td><td>99.9</td><td>80.4</td><td>86.2</td></tr><tr><td>Hydra-MDP <sup><a href="#fn:34">34</a></sup></td><td>98.3</td><td>96.0</td><td>94.6</td><td>100</td><td>78.7</td><td>86.5</td></tr><tr><td>DiffusionDrive <sup><a href="#fn:36">36</a></sup></td><td>98.2</td><td>96.2</td><td>94.7</td><td>100</td><td>82.2</td><td>88.1</td></tr><tr><td>WoTE <sup><a href="#fn:32">32</a></sup></td><td>98.5</td><td>96.8</td><td>94.9</td><td>99.9</td><td>81.9</td><td>88.3</td></tr><tr><td>DriveDPO <sup><a href="#fn:48">48</a></sup></td><td>98.5</td><td>98.1</td><td>94.8</td><td>99.9</td><td>84.3</td><td>90.0</td></tr><tr><td colspan="7">VLA methods w/o CoT</td></tr><tr><td>ReCogDrive <sup><a href="#fn:33">33</a></sup></td><td>98.2</td><td>97.8</td><td>95.2</td><td>99.8</td><td>83.5</td><td>89.6</td></tr><tr><td>DriveVLA-W0 <sup><a href="#fn:30">30</a></sup></td><td>98.7</td><td>99.1</td><td>95.3</td><td>99.3</td><td>83.3</td><td>90.2</td></tr><tr><td colspan="7">VLA methods w/ Textual CoT</td></tr><tr><td>AutoVLA <sup><a href="#fn:84">84</a></sup></td><td>98.4</td><td>95.6</td><td>98.0</td><td>99.9</td><td>81.9</td><td>89.1</td></tr><tr><td>AdaThinkDrive <sup><a href="#fn:38">38</a></sup></td><td>98.4</td><td>97.8</td><td>95.2</td><td>100</td><td>84.4</td><td>90.3</td></tr><tr><td>AutoDrive- <math><semantics><msup><mi>R</mi> <mn>2</mn></msup> <annotation>R^{2}</annotation></semantics></math> <sup><a href="#fn:72">72</a></sup></td><td>98.3</td><td>94.4</td><td>95.6</td><td>100</td><td>81.6</td><td>90.3</td></tr><tr><td colspan="7">VLA methods w/ Visual CoT</td></tr><tr><td>FSDrive <sup><a href="#fn:74">74</a></sup></td><td>98.2</td><td>93.8</td><td>93.3</td><td>99.9</td><td>80.1</td><td>85.1</td></tr><tr><td>PWM <sup><a href="#fn:80">80</a></sup></td><td>98.6</td><td>95.9</td><td>95.4</td><td>100</td><td>81.8</td><td>88.1</td></tr><tr><td>DynVLA (Ours)</td><td>98.6</td><td>98.7</td><td>95.5</td><td>100</td><td>86.8</td><td>91.7</td></tr></tbody></table>

Table 2: Comparison on Bench2Drive Benchmark. The best results are denoted by bold and the second best are denoted by underline. <sup>†</sup> denotes using privileged perceptual information.

| Method | DS $\uparrow$ | SR $\uparrow$ | Mean Multi-Ability $\uparrow$ |
| --- | --- | --- | --- |
| Think2Drive <sup>†</sup> [^28] | 91.85 | 85.41 | 86.26 |
| PDM-Lite <sup>†</sup> [^51] | 97.02 | 92.27 | 92.82 |
| AD-MLP [^75] | 18.05 | 0.00 | 0.87 |
| TCP [^64] | 40.70 | 15.00 | 14.63 |
| VAD [^25] | 42.35 | 15.00 | 18.07 |
| UniAD [^14] | 45.81 | 16.36 | 15.55 |
| ThinkTwice [^21] | 62.44 | 31.23 | 37.17 |
| DriveAdapter [^20] | 64.22 | 33.08 | 42.08 |
| Drivetransformer [^23] | 63.46 | 35.01 | 38.60 |
| Raw2Drive [^70] | 71.36 | 50.24 | 53.34 |
| ORION [^11] | 77.74 | 54.62 | 54.72 |
| MindDrive [^12] | 78.04 | 55.09 | 56.94 |
| AutoVLA [^84] | 78.84 | 57.73 | – |
| TF++ [^19] | 84.21 | 67.27 | 64.39 |
| SimLingo [^46] | 85.07 | 67.27 | – |
| DynVLA (Ours) | 88.34 | 72.73 | 72.23 |

#### In-house Dataset Results.

Table 3 reports the results on our large-scale in-house dataset, which is over an order of magnitude larger than public benchmarks, allowing us to evaluate scalability under richer and more diverse driving scenarios. We reimplement a widely adopted end-to-end method Transfuser [^8], as well as a strong VLA baseline DriveVLA-W0 [^30]. Compared to these methods, DynVLA achieves the lowest ADE and Collision Rate, demonstrating more reliable motion prediction and safer maneuver decisions at larger data scales.

![[x4 22.png|Refer to caption]]

Figure 4: Transferability of learned dynamics. Dynamics tokens extracted from one scenario are injected into a new scene and decoded into the future image and the BEV map. We contrast the current states, the future states decoded with transferred dynamics, and the original future states. The results show that both ego-centric and environment-centric dynamics are transferable across scenarios.

![[x5 22.png|Refer to caption]]

Figure 5: Codebook collapse without dynamics decouple.

![[x6 19.png|Refer to caption]]

Figure 6: Dynamics CoT improves planning by reasoning over future dynamics. The first two columns show the current observation and the future decoded by reasoned dynamics. The third and last columns compare planning results with and without Dynamics CoT. Compared to direct action prediction, Dynamics CoT provides intent-aware, foresighted, and constraint-compliant future dynamics, enabling safer and more feasible planning in challenging scenarios.

### 4.3 Further Analysis and Ablation Studies

#### Transferability of Learned Dynamics Tokens.

Fig. 4 presents a demonstration of the transferability of our learned dynamics tokens. Dynamics tokens extracted from one scenario are injected into a new scene and decoded into future states for visualization. The results show that ego-centric dynamics reliably preserve ego motion, while environment-centric dynamics explicitly govern the motions of surrounding agents. Moreover, when both dynamics are combined, the decoded future states accurately reflect the composed dynamics configuration. This demonstrates that the Dynamics Tokenizer learns transferable, disentangled, and interpretable dynamics representations.

Table 3: Comparison on a large-scale In-house Dataset. The best results are denoted by bold.

| Model | ADE (m) $\downarrow$ | Collision Rate (‱) $\downarrow$ |
| --- | --- | --- |
| Transfuser [^8] | 1.746 | 5.63 |
| DriveVLA-W0 (VQ) [^30] | 1.599 | 5.20 |
| DriveVLA-W0 (ViT) [^30] | 1.344 | 5.13 |
| DynVLA (Ours) | 1.215 | 4.04 |

Table 4: Analysis on CoT Design and Latency. All inference latencies are measured on a single NVIDIA H800 GPU. The best results are denoted by bold.

| CoT Content | Latency(s) $\downarrow$ | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | C $\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| None (w/o CoT) | 0.20 | 98.3 | 93.8 | 94.6 | 99.9 | 79.5 | 85.6 |
| Scene Description | 3.04 | 98.4 | 93.4 | 94.4 | 99.9 | 79.3 | 85.3 |
| Meta Action | 0.43 | 98.3 | 94.3 | 94.6 | 100 | 79.8 | 86.0 |
| Future Image | 2.29 | 98.7 | 94.4 | 95.0 | 99.9 | 80.0 | 86.3 |
| Optical Flow | 2.29 | 98.6 | 94.4 | 95.3 | 100 | 80.0 | 86.4 |
| Dynamics (Ours) | 0.37 | 98.6 | 95.3 | 95.5 | 100 | 80.6 | 87.2 |

Table 5: Ablation on Training Stages. Dyn CoT denotes Dynamics CoT. The best results are denoted by bold.

<table><tbody><tr><td>Base Model</td><td>Dyn CoT</td><td>SFT</td><td>RFT</td><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>C <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td rowspan="4">EMU3 <sup><a href="#fn:59">59</a></sup></td><td></td><td>✓</td><td></td><td>98.3</td><td>93.8</td><td>94.6</td><td>99.9</td><td>79.5</td><td>85.6</td></tr><tr><td>✓</td><td>✓</td><td></td><td>98.6</td><td>95.3</td><td>95.5</td><td>100</td><td>80.6</td><td>87.2</td></tr><tr><td></td><td>✓</td><td>✓</td><td>98.6</td><td>96.7</td><td>95.6</td><td>100</td><td>82.3</td><td>88.7</td></tr><tr><td>✓</td><td>✓</td><td>✓</td><td>98.6</td><td>98.7</td><td>95.5</td><td>100</td><td>86.8</td><td>91.7</td></tr><tr><td rowspan="4">Qwen2.5-VL <sup><a href="#fn:1">1</a></sup></td><td></td><td>✓</td><td></td><td>98.3</td><td>93.5</td><td>94.8</td><td>100</td><td>79.1</td><td>85.3</td></tr><tr><td>✓</td><td>✓</td><td></td><td>98.8</td><td>94.4</td><td>95.8</td><td>100</td><td>79.9</td><td>86.6</td></tr><tr><td></td><td>✓</td><td>✓</td><td>98.7</td><td>96.1</td><td>96.1</td><td>100</td><td>81.6</td><td>88.4</td></tr><tr><td>✓</td><td>✓</td><td>✓</td><td>98.8</td><td>97.9</td><td>95.9</td><td>99.9</td><td>85.8</td><td>91.0</td></tr></tbody></table>

#### Dynamics CoT Outperforms Other CoT Designs in Both Effectiveness and Efficiency.

We compare different CoT designs and report their inference latency in Table 4. Textual CoT based on scene descriptions incurs substantial inference overhead and degrades performance, suggesting that coarse scene-level descriptions may not provide practical planning guidance. Reasoning over meta-action decisions yields only marginal performance improvements, suggesting that such high-level symbolic abstractions lack sufficient expressive capacity. Furthermore, visual CoT that predicts future images brings moderate performance gains, but the large number of visual reasoning tokens significantly increases inference latency. We further explore explicit dynamics modeling using current-to-future optical flow as a CoT choice. Although this also improves performance, it remains inferior to ours and still suffers from high latency. Finally, Dynamics CoT achieves the best overall performance without introducing substantial latency overhead.

#### Dynamics CoT Strengthens Both SFT and RFT Across Base Models.

Table 5 compares the effects of different training stages under a controlled setting across two base models. Starting from the SFT baseline without CoT, applying Dynamics CoT SFT consistently improves planning-related metrics. In addition, applying RFT to the baseline without CoT also improves PDMS, but the gain is noticeably smaller compared with Dynamics CoT RFT. This is because Dynamics CoT provides a compact, structured reasoning trace, enabling RFT to optimize final actions more effectively. Consistent trends across EMU3 [^59] and Qwen2.5-VL [^1] also verify the effectiveness and generality of Dynamics CoT. Since EMU3 achieves the best final performance, which we attribute to its unified architecture better fitting the new dynamics modality, we adopt it as the base model for DynVLA.

#### Decoupling Dynamics Prevents Codebook Collapse.

Fig. 5 illustrates the number of activated VQ codes during Dynamics Tokenizer training. The number increases rapidly when decoupling dynamics (i.e., introducing separate queries for ego and environment dynamics and applying action-based regularization). However, the tokenizer exhibits an apparent codebook collapse without disentanglement. This is because without decoupling, the tokenizer solely minimizes the reconstruction loss. However, as the decoder conditions on the current observation, much of the background information can be recovered directly, diminishing the necessity for expressive VQ-compressed dynamics representations. By introducing disentangled modeling, the dynamics tokens are forced to learn more discriminative representations, effectively alleviating codebook collapse.

Table 6: Ablation on Dynamics Tokenizer Designs. Dyn CoT denotes Dynamics CoT. The best results are denoted by bold.

<table><tbody><tr><td>Model</td><td>Decouple</td><td>Image</td><td>BEV</td><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>C <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>w/o CoT</td><td>–</td><td>–</td><td>–</td><td>98.3</td><td>93.8</td><td>94.6</td><td>99.9</td><td>79.5</td><td>85.6</td></tr><tr><td rowspan="4">w/ Dyn CoT</td><td></td><td>✓</td><td>✓</td><td>98.5</td><td>94.0</td><td>94.9</td><td>100</td><td>79.5</td><td>85.8</td></tr><tr><td>✓</td><td></td><td>✓</td><td>98.4</td><td>94.5</td><td>94.8</td><td>100</td><td>79.8</td><td>86.2</td></tr><tr><td>✓</td><td>✓</td><td></td><td>98.6</td><td>94.8</td><td>95.0</td><td>100</td><td>80.6</td><td>86.7</td></tr><tr><td>✓</td><td>✓</td><td>✓</td><td>98.6</td><td>95.3</td><td>95.5</td><td>100</td><td>80.6</td><td>87.2</td></tr></tbody></table>

#### Decoupling and Regularization Benefit Planning.

We further analyze the design of the Dynamics Tokenizer in Table 6. The Dynamics Tokenizer without decoupling (i.e., without separate queries or ego-action regularization) yields marginal improvements over the model without CoT, indicating it is unable to capture meaningful dynamics. We also remove either the image branch or the BEV branch, and both result in performance degradation, indicating the importance of cross-view consistency regularization. Finally, combining dynamics decoupling with dual image-BEV supervision achieves the best performance, demonstrating the effectiveness of each component in our design.

### 4.4 Qualitative Analysis

#### Safer Intent-aware Interaction.

In interactive driving scenarios, directly committing to action decisions may fail to adjust behavior according to other agents’ intentions. However, Dynamics CoT can capture the motion intent of surrounding agents and adjust planning accordingly, leading to safer maneuver decisions. As shown in Fig. 6 (top row), the inferred dynamics indicate that the front car will stop, and the model plans to stop accordingly, avoiding the collision observed without Dynamics CoT.

#### Foresighted Planning.

Many planning tasks cannot be directly resolved by instantaneous cues, leading to short-sighted decisions. In contrast, Dynamics CoT can reason about how traffic evolves over several seconds, revealing future states and enabling foresighted trajectory planning. As shown in Fig. 6 (middle row), Dynamics CoT predicts the leading vehicle moving rightward, which opens a drivable corridor, and the model then exploits this future space to execute a safe maneuver, whereas the model without Dynamics CoT drifts off-road due to a lack of this foresight.

#### Road-geometry Awareness.

Beyond agent interactions, driving is also physically constrained by road geometry, yet direct action prediction may struggle to anticipate how such constraints evolve. However, Dynamics CoT can reflect the evolution of road constraints in its predicted dynamics, giving the model awareness of future road boundaries and enabling timely steering adjustments. As shown in Fig. 6 (bottom row), the reasoned dynamics indicate an upcoming curb ahead, guiding the model to slightly turn right and maintain a feasible lane-keeping trajectory, preventing hitting the curb that occurs without Dynamics CoT.

## 5 Conclusion

In this work, we propose DynVLA, which introduces a novel CoT paradigm called Dynamics CoT for VLA-based autonomous driving. Compared to existing CoT methods, Dynamics CoT reduces redundancy and latency while retaining accurate spatiotemporal understanding. We first introduce a Dynamics Tokenizer that disentangles ego-centric and environment-centric dynamics, and regularize it with ego action supervision and cross-view consistency, enabling physically meaningful and planning-oriented dynamics representations. We then perform SFT on Dynamics CoT to enable explicit reasoning over future dynamics, and further apply RFT to improve decision quality. Experiments across multiple benchmarks demonstrate the effectiveness of DynVLA and highlight Dynamics CoT as a promising direction for reasoning-based VLA models. We further discuss limitations and future works in Appendix E.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## References

## Appendix

## Appendix A Datasets and Metrics

#### NAVSIM.

NAVSIM [^9] is a real-world benchmark providing diverse urban driving data with rich agent interactions. Following standard protocol, we evaluate policy performance using the PDMS, a scalar metric that aggregates multiple safety and efficiency measures. Specifically, PDMS integrates No At-Fault Collision (NC), Drivable Area Compliance (DAC), Ego Progress (EP), Time-to-Collision (TTC), and Comfort (C) as:

$$
\text{PDMS}=\text{NC}\times\text{DAC}\times\frac{5\times\text{EP}+5\times\text{TTC}+2\times\text{C}}{12}.
$$

Here, NC and DAC explicitly gate safety, while EP and TTC capture driving efficiency and temporal risk margins, and C measures comfort.

#### Bench2Drive.

Bench2Drive [^22] is a closed-loop benchmark, which enables scenario-level evaluation under interactive traffic. We report Success Rate (SR), which indicates whether the ego vehicle reaches the intended destination, Driving Score (DS), which additionally accounts for traffic rule penalties, and mean Multi-Ability, which averages the performance across five ability categories: Merging, Overtaking, Emergency Brake, Give Way, and Traffic Sign.

#### In-House Dataset.

We further train our model on a large-scale in-house dataset consisting of 700k frames with diverse and balanced distributions over driving scenarios, road structures, actor densities, and agent intentions. Evaluation focuses on safety-critical and long-horizon cases, where we report Average Displacement Error (ADE) and the Collision Rate within 3 seconds. ADE measures the Euclidean displacement between predicted and ground-truth future ego positions, while the Collision Rate quantifies the fraction of predicted trajectories that result in collisions within the horizon.

## Appendix B Implementation Details

#### Dynamics Tokenizer

For each driving scene, we use 8 dynamics tokens, consisting of $N_{\text{ego}}=4$ ego-centric dynamics tokens and $N_{\text{env}}=4$ environment-centric dynamics tokens, which are inferred from the front-view image. The codebook size for both the ego and environment branches is set to 64, resulting in 128 distinct discrete dynamics token types in total, and the VQ embedding dimension is 32. The Dynamics Tokenizer is implemented with a Transformer architecture. The hidden dimension is set to 1024, the Dynamics Encoder contains $L_{\text{Enc}}=12$ layers, and both the image decoder and the BEV decoder consist of $L_{\text{Dec}}=8$ layers. The encoder takes as input the patchified observations $(\mathbf{x}_{t},\mathbf{x}_{t+1})$ together with the learned queries $(Q_{\text{ego}},Q_{\text{env}})$. We concatenate these four input sequences to form a single token sequence that is fed into the Transformer layers. The ego action used for action regularization corresponds to the relative ego motion between two frames. For BEV supervision, we use the front-view BEV maps provided by the dataset. During training, the image reconstruction loss is composed of an MSE loss and an LPIPS loss, both with a weight of 1.0. The VQ loss weight is set to 1.0, the action regularization loss weight is set to 1.0, and the BEV reconstruction loss weight is set to 0.1. We train the Dynamics Tokenizer for 200k steps on 8 NVIDIA L20 GPUs using a cosine learning rate schedule with 1k warm-up steps and a maximum learning rate of $1\times 10^{-4}$. The batch size is set to 32, and we use the AdamW optimizer with $\beta_{1}=0.9$ and $\beta_{2}=0.95$.

#### Dynamics CoT SFT

We adopt EMU3 [^59] as the pretrained base model and follow the same pretraining protocol as DriveVLA-W0 [^30]. The Dynamics Tokenizer uses a codebook of 128 discrete dynamics token types, whereas the FAST tokenizer [^45] uses a codebook of 2048 discrete action token types. We replace the last $2048+128$ tokens in the token vocabulary with the action tokens and dynamics tokens, respectively. For Dynamics CoT supervision, we extract future dynamics over a 2-second horizon ($K=2$), yielding a sequence of 16 dynamics tokens that serve as CoT content during SFT. As sensor input, we only use the current front-view image together with the front-view image from 1s earlier. During training, both the Dynamics CoT loss and the action prediction loss are weighted equally with coefficients set to 1.0. We fine-tune the pretrained model for 4k steps on 8 NVIDIA L20 GPUs using a cosine learning rate schedule with 100 warm-up steps and a maximum learning rate of $1\times 10^{-4}$. The batch size is set to 6, and we use the AdamW optimizer with $\beta_{1}=0.9$ and $\beta_{2}=0.95$.

#### Dynamics CoT RFT

We perform reinforcement fine-tuning (RFT) on top of the SFT-trained model. The trajectory reward $r_{\text{traj}}$ and the format reward $r_{\text{fmt}}$ are both weighted with coefficients set to 1.0. RFT is conducted for 6k steps on 6 NVIDIA H800 GPUs using a cosine learning rate schedule with 500 warm-up steps and a maximum learning rate of $2\times 10^{-6}$. The batch size is set to 6, and the gradient accumulation step is set to 6 as well. The KL coefficient is $1\times 10^{-3}$ and we use the AdamW optimizer with $\beta_{1}=0.9$ and $\beta_{2}=0.95$.

![[x7 15.png|Refer to caption]]

Figure 7: Visualization of the decoded image and BEV from Dynamics Tokenizer. For each row, we show the current image and BEV map, the decoded future image and BEV map, and the ground-truth future image and BEV map. Across diverse driving scenarios, the decoded futures faithfully capture ego motion and the dynamics of surrounding agents.

![[x8 11.png|Refer to caption]]

Figure 8: Additional visualizations of the transferability of learned dynamics.

## Appendix C More Ablation Studies

#### Ablation on the Prediction Horizon of Future Dynamics.

Table 7: Ablation on the prediction horizon of Dynamics CoT. The prediction horizon $K$ denotes the number of reasoned future dynamics. The best results are denoted by bold.

| Pred Horizon | Latency(s) $\downarrow$ | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | C $\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| w/o CoT | 0.20 | 98.3 | 93.8 | 94.6 | 99.9 | 79.5 | 85.6 |
| $K=1$ | 0.27 | 98.6 | 94.6 | 95.0 | 100 | 80.1 | 86.5 |
| $K=2$ | 0.37 | 98.6 | 95.3 | 95.5 | 100 | 80.6 | 87.2 |
| $K=3$ | 0.49 | 98.6 | 94.7 | 95.2 | 100 | 80.3 | 86.7 |
| $K=4$ | 0.61 | 98.5 | 94.7 | 95.3 | 99.9 | 80.2 | 86.6 |

Table 7 studies the Dynamics CoT prediction horizon by varying the number of future dynamics steps $K$. Following the formulation in Sec.3, each dynamics token $\mathcal{D}_{t+k}$ encodes the transition between two adjacent images $(O_{t+k},O_{t+k+1})$, where consecutive steps are spaced by 1s. Compared to the baseline without CoT, introducing Dynamics CoT consistently improves PDMS across all horizons. Increasing the horizon from $K=1$ to $K=2$ yields clear gains, indicating that a too short lookahead is insufficient. However, extending the horizon beyond $K=2$ yields diminishing returns while increasing inference latency. This is because longer horizons introduce greater uncertainty about the future, making the predicted dynamics less reliable and ultimately degrading planning quality. Based on this, we adopt $K=2$ (corresponding to a $2$ s future horizon) as the default configuration, which achieves the best performance while maintaining latency-efficient inference.

#### Ablation on the Number of Dynamics Tokens.

Table 8: Ablation on the number and allocation of dynamics tokens. Dyn Num denotes the total number of dynamics tokens, $N_{\text{ego}}$ denotes the number of ego-centric tokens, and $N_{\text{env}}$ denotes the number of environment-centric tokens. The best results are denoted by bold.

| Dyn Num | ${N}_{\text{ego}}$ | ${N}_{\text{env}}$ | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | C $\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 4 | 4 | 98.6 | 95.3 | 95.5 | 100 | 80.6 | 87.2 |
| 8 | 2 | 6 | 98.7 | 94.9 | 95.3 | 100 | 80.5 | 86.9 |
| 8 | 6 | 2 | 98.6 | 94.5 | 95.0 | 100 | 80.2 | 86.4 |
| 4 | 2 | 2 | 98.5 | 94.6 | 95.1 | 100 | 80.1 | 86.4 |
| 16 | 8 | 8 | 98.5 | 94.7 | 94.9 | 100 | 80.3 | 86.5 |

Table 8 investigates how the capacity and factorization of the Dynamics Tokenizer affect planning performance by varying the total number of dynamics tokens and their allocation to ego-centric versus environment-centric branches. Overall, using 8 dynamics tokens achieves the best PDMS, indicating that a compact yet sufficiently expressive dynamics bottleneck is critical for capturing planning-relevant scene evolution. Reducing the token budget to 4 degrades PDMS, suggesting insufficient capacity to represent simultaneous ego motion and multi-agent interactions. Conversely, increasing the token budget to 16 does not bring further gains and slightly hurts performance, which we attribute to more redundant dynamics representations, ultimately degrading performance. With a fixed budget of 8 tokens, a balanced split ($N_{\text{ego}}{=}4,N_{\text{env}}{=}4$) outperforms skewed allocations, highlighting that both ego and environment dynamics are necessary.

## Appendix D More Qualitative Comparisons

#### Decoded Visualization.

We provide a visualization of the decoded outputs from the Dynamics Tokenizer across a diverse set of driving scenarios. As shown in Fig. 7, the decoded futures closely align with the ground-truth evolution in both image space and BEV space, consistently reflecting ego motion and surrounding agent behaviors. These results indicate that the proposed Dynamics Tokenizer effectively compresses planning-relevant ego and agent dynamics.

#### Additional Dynamics Transfer Visualizations.

As shown in Fig. 8, we provide additional qualitative visualizations of the cross-scenario dynamics transfer. These results further confirm that the learned dynamics capture disentangled and transferable motion representations that generalize across scenes.

#### Additional Qualitative Comparisons with Dynamics CoT.

![[x9 5.png|Refer to caption]]

Figure 9: Additional qualitative comparisons of planning behavior with and without Dynamics CoT.

As shown in Fig. 9, we provide additional qualitative examples demonstrating the effect of Dynamics CoT on planning. Compared to models without Dynamics CoT, incorporating dynamics reasoning enables the policy to anticipate future scene evolution, thereby improving decision-making.

## Appendix E Limitations and Future Works

Although Dynamics CoT provides structured intermediate reasoning that benefits safety-critical planning, incorrect reasoning traces may induce suboptimal decisions. Similar to reasoning models where flawed CoT leads to erroneous final answers, Dynamics CoT may generate inaccurate future dynamics under highly complex or uncertain driving scenarios, propagating these errors into the subsequent action generation stage. A promising future direction is to enhance Dynamics CoT with richer driving domain priors and world knowledge. In particular, integrating structured world-model knowledge, rule-based priors, or map-aware commonsense may improve the fidelity of predicted dynamics tokens. Another promising direction is to cast our dynamics CoT as the slow component in a fast-slow dual-system driving architecture [^54] [^41]: the slow module performs long-horizon dynamics reasoning and updates dynamics tokens at a lower frequency, while a lightweight fast planner outputs actions at every control cycle by consuming the most recent observation and reasoned dynamics tokens.

## Appendix F Failure Cases

![[x10 6.png|Refer to caption]]

Figure 10: Representative failure cases. From left to right, we show the planning result, the current observation, the future decoded from inferred dynamics, and the ground-truth future. These cases include incorrect inference of surrounding vehicles’ intentions, misidentification of drivable areas during large turning maneuvers, and ambiguous dynamics reasoning due to degraded visual observations.

Our trajectory planning relies on the inferred future dynamics as intermediate reasoning signals. Consequently, when the predicted dynamics are inaccurate or ambiguous, the planned decisions may also be suboptimal. One possible failure arises from misinference about other agents’ intentions. As illustrated in the first and second rows of Fig. 10, the model erroneously predicts the leading vehicle to continue moving forward and thus plans to follow with a straight trajectory. Another failure case occurs in complex turning scenarios, where the model incorrectly reasons about future road-structure dynamics. As shown in the third row of Fig. 10, the model fails to recognize a non-drivable parking area. This is because predicting road geometry that will only become visible from a novel perspective is challenging. Finally, when the current observation is severely degraded, the inferred future dynamics can become ambiguous. As shown in the fourth row of Fig. 10, heavy rain causes camera occlusion and results in blurred observations. Under such conditions, the predicted dynamics lack sufficient certainty, and the model makes unsafe decisions.

## Appendix G More Related Works

### G.1 Latent Action Tokenizer

Latent action tokenizers aim to overcome the scalability bottleneck of action-labeled data by extracting action-like tokens directly from raw video. LAPO [^47] shows that a latent inverse-dynamics structure can be recovered purely from videos, enabling latent-action policies. LAPA [^71] explicitly learns a VQ-style discrete latent action codebook between adjacent frames and pretrains a VLA to predict these latent actions with a lightweight decoder to real robot controls. Moto [^7] treats latent action tokens as a hardware-agnostic motion language and pretrains an autoregressive model over these tokens, then co-finetunes to bridge motion-token prediction to executable robot actions. UniVLA [^4] emphasizes transfer across embodiments by extracting task-centric latent actions from large-scale cross-embodiment videos and efficiently decoding them for different robots. LAOM [^43] highlights that purely unsupervised latent action learning can be confounded by action-correlated distractors, and injecting sparse ground-truth action supervision during latent-action training substantially improves downstream control. In contrast to these works that primarily focus on learning a general latent action space for embodied settings, our goal is driving-oriented: we explicitly disentangle ego-centric and environment-centric dynamics across multi-agent, viewpoint-changing scenes and further introduce a cross-space regularization that enforces semantic alignment between dynamics learned from the image space and the BEV space.

### G.2 Foresight-driven Policies

Foresight-driven policies have been widely studied in embodied control, where actions are predicted by first anticipating future outcomes. Early work, such as HVF [^42] decomposes tasks via visual subgoal generation, while GCPs [^44] further adopt goal-conditioned coarse-to-fine predictors to reduce long-horizon planning complexity. More recently, foresight has been integrated into end-to-end and generalist policies. VPP [^15] conditions a single policy on predictive visual representations learned from large-scale video models, and recent VLA-style methods [^55] [^79] [^39] treat predicted future states or visual reasoning steps as intermediate signals that guide inverse dynamics for action generation. In autonomous driving, foresight-driven policies are also adopted to handle long horizons and multi-agent interactions. FSDrive [^74] introduces a spatio-temporal visual CoT by generating future visual states. PWM [^80] proposes a policy world model that forecasts future states and performs collaborative state–action prediction to better couple forecasting and planning. However, existing approaches predominantly rely on explicit dense future-frame prediction, which introduces redundant reasoning and non-trivial inference overhead. In contrast, we focus on predicting compact future dynamics, preserving decision-critical foresight while enabling latency-efficient inference.

In addition, LAW [^29] enhances end-to-end driving via a latent world model that predicts future latent features, supervised by latent representations of future observations in a self-supervised manner. SSR [^27] also exploits temporal context for self-supervision, reducing reliance on expensive perception annotations. Notably, they primarily improve driving via self-supervised learning, whereas we compress future evolution into compact dynamics tokens that serve as an explicit intermediate reasoning trace for action generation.

[^1]: Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., et al. Qwen2.5-vl technical report. *arXiv preprint arXiv:2502.13923*, 2025.

[^2]: Black, K., Brown, N., Driess, D., Esmail, A., Equi, M., Finn, C., Fusai, N., Groom, L., Hausman, K., Ichter, B., et al. $\pi 0$: A vision-language-action flow model for general robot control. *arXiv preprint arXiv:2410.24164*, 2024.

[^3]: Black, K., Brown, N., Darpinian, J., Dhabalia, K., Driess, D., Esmail, A., Equi, M. R., Finn, C., Fusai, N., Galliker, M. Y., et al. $\pi{0.5}$: a vision-language-action model with open-world generalization. In *9th Annual Conference on Robot Learning*, 2025.

[^4]: Bu, Q., Yang, Y., Cai, J., Gao, S., Ren, G., Yao, M., Luo, P., and Li, H. Univla: Learning to act anywhere with task-centric latent actions. *arXiv preprint arXiv:2505.06111*, 2025.

[^5]: Chen, Q., Qin, L., Liu, J., Peng, D., Guan, J., Wang, P., Hu, M., Zhou, Y., Gao, T., and Che, W. Towards reasoning era: A survey of long chain-of-thought for reasoning large language models. *arXiv preprint arXiv:2503.09567*, 2025a.

[^6]: Chen, S., Jiang, B., Gao, H., Liao, B., Xu, Q., Zhang, Q., Huang, C., Liu, W., and Wang, X. Vadv2: End-to-end vectorized autonomous driving via probabilistic planning. *arXiv preprint arXiv:2402.13243*, 2024.

[^7]: Chen, Y., Ge, Y., Tang, W., Li, Y., Ge, Y., Ding, M., Shan, Y., and Liu, X. Moto: Latent motion token as the bridging language for learning robot manipulation from videos. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pp. 19752–19763, 2025b.

[^8]: Chitta, K., Prakash, A., Jaeger, B., Yu, Z., Renz, K., and Geiger, A. Transfuser: Imitation with transformer-based sensor fusion for autonomous driving. *IEEE transactions on pattern analysis and machine intelligence*, 45(11):12878–12895, 2022.

[^9]: Dauner, D., Hallgarten, M., Li, T., Weng, X., Huang, Z., Yang, Z., Li, H., Gilitschenski, I., Ivanovic, B., Pavone, M., et al. Navsim: Data-driven non-reactive autonomous vehicle simulation and benchmarking. *Advances in Neural Information Processing Systems*, 2024.

[^10]: Dosovitskiy, A. An image is worth 16x16 words: Transformers for image recognition at scale. *arXiv preprint arXiv:2010.11929*, 2020.

[^11]: Fu, H., Zhang, D., Zhao, Z., Cui, J., Liang, D., Zhang, C., Zhang, D., Xie, H., Wang, B., and Bai, X. Orion: A holistic end-to-end autonomous driving framework by vision-language instructed action generation. *arXiv preprint arXiv:2503.19755*, 2025a.

[^12]: Fu, H., Zhang, D., Zhao, Z., Cui, J., Xie, H., Wang, B., Chen, G., Liang, D., and Bai, X. Minddrive: A vision-language-action model for autonomous driving via online reinforcement learning. *arXiv preprint arXiv:2512.13636*, 2025b.

[^13]: Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. *arXiv preprint arXiv:2501.12948*, 2025.

[^14]: Hu, Y., Yang, J., Chen, L., Li, K., Sima, C., Zhu, X., Chai, S., Du, S., Lin, T., Wang, W., et al. Planning-oriented autonomous driving. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 17853–17862, 2023.

[^15]: Hu, Y., Guo, Y., Wang, P., Chen, X., Wang, Y.-J., Zhang, J., Sreenath, K., Lu, C., and Chen, J. Video prediction policy: A generalist robot policy with predictive visual representations. *arXiv preprint arXiv:2412.14803*, 2024.

[^16]: Huang, X., Wolff, E. M., Vernaza, P., Phan-Minh, T., Chen, H., Hayden, D. S., Edmonds, M., Pierce, B., Chen, X., Jacob, P. E., et al. Drivegpt: Scaling autoregressive behavior models for driving. In *Forty-second International Conference on Machine Learning*, 2025.

[^17]: Huang, Z., Feng, C., Yan, F., Xiao, B., Jie, Z., Zhong, Y., Liang, X., and Ma, L. Drivemm: All-in-one large multimodal model for autonomous driving. *arXiv preprint arXiv:2412.07689*, 2024.

[^18]: Hwang, J.-J., Xu, R., Lin, H., Hung, W.-C., Ji, J., Choi, K., Huang, D., He, T., Covington, P., Sapp, B., et al. Emma: End-to-end multimodal model for autonomous driving. *arXiv preprint arXiv:2410.23262*, 2024.

[^19]: Jaeger, B., Chitta, K., and Geiger, A. Hidden biases of end-to-end driving models. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pp. 8240–8249, 2023.

[^20]: Jia, X., Gao, Y., Chen, L., Yan, J., Liu, P. L., and Li, H. Driveadapter: Breaking the coupling barrier of perception and planning in end-to-end autonomous driving. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pp. 7953–7963, 2023a.

[^21]: Jia, X., Wu, P., Chen, L., Xie, J., He, C., Yan, J., and Li, H. Think twice before driving: Towards scalable decoders for end-to-end autonomous driving. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 21983–21994, 2023b.

[^22]: Jia, X., Yang, Z., Li, Q., Zhang, Z., and Yan, J. Bench2drive: Towards multi-ability benchmarking of closed-loop end-to-end autonomous driving. *Advances in Neural Information Processing Systems*, 37:819–844, 2024.

[^23]: Jia, X., You, J., Zhang, Z., and Yan, J. Drivetransformer: Unified transformer for scalable end-to-end autonomous driving. In *The Thirteenth International Conference on Learning Representations*, 2025.

[^24]: Jiang, A., Gao, Y., Sun, Z., Wang, Y., Wang, J., Chai, J., Cao, Q., Heng, Y., Jiang, H., Dong, Y., et al. Diffvla: Vision-language guided diffusion planning for autonomous driving. *arXiv preprint arXiv:2505.19381*, 2025.

[^25]: Jiang, B., Chen, S., Xu, Q., Liao, B., Chen, J., Zhou, H., Zhang, Q., Liu, W., Huang, C., and Wang, X. Vad: Vectorized scene representation for efficient autonomous driving. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pp. 8340–8350, 2023.

[^26]: Kim, M. J., Pertsch, K., Karamcheti, S., Xiao, T., Balakrishna, A., Nair, S., Rafailov, R., Foster, E. P., Sanketi, P. R., Vuong, Q., et al. Openvla: An open-source vision-language-action model. In *8th Annual Conference on Robot Learning*, 2024.

[^27]: Li, P. and Cui, D. Navigation-guided sparse scene representation for end-to-end autonomous driving. In *The Thirteenth International Conference on Learning Representations*, 2025.

[^28]: Li, Q., Jia, X., Wang, S., and Yan, J. Think2drive: Efficient reinforcement learning by thinking with latent world model for autonomous driving (in carla-v2). In *European Conference on Computer Vision*, pp. 142–158, 2024a.

[^29]: Li, Y., Fan, L., He, J., Wang, Y., Chen, Y., Zhang, Z., and Tan, T. Enhancing end-to-end autonomous driving with latent world model. In *The Thirteenth International Conference on Learning Representations*, 2024b.

[^30]: Li, Y., Shang, S., Liu, W., Zhan, B., Wang, H., Wang, Y., Chen, Y., Wang, X., An, Y., Tang, C., et al. Drivevla-w0: World models amplify data scaling law in autonomous driving. *arXiv preprint arXiv:2510.12796*, 2025a.

[^31]: Li, Y., Tian, M., Zhu, D., Zhu, J., Lin, Z., Xiong, Z., and Zhao, X. Drive-r1: Bridging reasoning and planning in vlms for autonomous driving with reinforcement learning. *arXiv preprint arXiv:2506.18234*, 2025b.

[^32]: Li, Y., Wang, Y., Liu, Y., He, J., Fan, L., and Zhang, Z. End-to-end driving with online trajectory evaluation via bev world model. *arXiv preprint arXiv:2504.01941*, 2025c.

[^33]: Li, Y., Xiong, K., Guo, X., Li, F., Yan, S., Xu, G., Zhou, L., Chen, L., Sun, H., Wang, B., et al. Recogdrive: A reinforced cognitive framework for end-to-end autonomous driving. *arXiv preprint arXiv:2506.08052*, 2025d.

[^34]: Li, Z., Li, K., Wang, S., Lan, S., Yu, Z., Ji, Y., Li, Z., Zhu, Z., Kautz, J., Wu, Z., et al. Hydra-mdp: End-to-end multimodal planning with multi-target hydra-distillation. *arXiv preprint arXiv:2406.06978*, 2024c.

[^35]: Liang, D., Zhang, D., Zhou, X., Tu, S., Feng, T., Li, X., Zhang, Y., Du, M., Tan, X., and Bai, X. Seeing the future, perceiving the future: A unified driving world model for future generation and perception. *arXiv preprint arXiv:2503.13587*, 2025.

[^36]: Liao, B., Chen, S., Yin, H., Jiang, B., Wang, C., Yan, S., Zhang, X., Li, X., Zhang, Y., Zhang, Q., et al. Diffusiondrive: Truncated diffusion model for end-to-end autonomous driving. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pp. 12037–12047, 2025.

[^37]: Lin, F., Nai, R., Hu, Y., You, J., Zhao, J., and Gao, Y. Onetwovla: A unified vision-language-action model with adaptive reasoning. *arXiv preprint arXiv:2505.11917*, 2025.

[^38]: Luo, Y., Li, F., Xu, S., Lai, Z., Yang, L., Chen, Q., Luo, Z., Xie, Z., Jiang, S., Liu, J., et al. Adathinkdrive: Adaptive thinking via reinforcement learning for autonomous driving. *arXiv preprint arXiv:2509.13769*, 2025.

[^39]: Lv, Q., Kong, W., Li, H., Zeng, J., Qiu, Z., Qu, D., Song, H., Chen, Q., Deng, X., and Pang, J. F1: A vision-language-action model bridging understanding and generation to actions. *arXiv preprint arXiv:2509.06951*, 2025.

[^40]: Mao, J., Qian, Y., Ye, J., Zhao, H., and Wang, Y. Gpt-driver: Learning to drive with gpt. In *NeurIPS 2023 Foundation Models for Decision Making Workshop*, 2023.

[^41]: Mei, J., Ma, Y., Yang, X., Wen, L., Cai, X., Li, X., Fu, D., Zhang, B., Cai, P., Dou, M., et al. Continuously learning, adapting, and improving: A dual-process approach to autonomous driving. *Advances in Neural Information Processing Systems*, 37:123261–123290, 2024.

[^42]: Nair, S. and Finn, C. Hierarchical foresight: Self-supervised learning of long-horizon tasks via visual subgoal generation. In *International Conference on Learning Representations*, 2020.

[^43]: Nikulin, A., Zisman, I., Tarasov, D., Nikita, L., Polubarov, A., Kiselev, I., and Kurenkov, V. Latent action learning requires supervision in the presence of distractors. In *Forty-second International Conference on Machine Learning*, 2025.

[^44]: Pertsch, K., Rybkin, O., Ebert, F., Zhou, S., Jayaraman, D., Finn, C., and Levine, S. Long-horizon visual planning with goal-conditioned hierarchical predictors. *Advances in Neural Information Processing Systems*, 2020.

[^45]: Pertsch, K., Stachowicz, K., Ichter, B., Driess, D., Nair, S., Vuong, Q., Mees, O., Finn, C., and Levine, S. Fast: Efficient action tokenization for vision-language-action models. *arXiv preprint arXiv:2501.09747*, 2025.

[^46]: Renz, K., Chen, L., Arani, E., and Sinavski, O. Simlingo: Vision-only closed-loop autonomous driving with language-action alignment. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pp. 11993–12003, 2025.

[^47]: Schmidt, D. and Jiang, M. Learning to act without actions. In *The Twelfth International Conference on Learning Representations*, 2024.

[^48]: Shang, S., Chen, Y., Wang, Y., Li, Y., and Zhang, Z. Drivedpo: Policy learning via safety dpo for end-to-end autonomous driving. In *The Thirty-ninth Annual Conference on Neural Information Processing Systems*, 2025.

[^49]: Shao, H., Hu, Y., Wang, L., Song, G., Waslander, S. L., Liu, Y., and Li, H. Lmdrive: Closed-loop end-to-end driving with large language models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 15120–15130, 2024a.

[^50]: Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. *arXiv preprint arXiv:2402.03300*, 2024b.

[^51]: Sima, C., Renz, K., Chitta, K., Chen, L., Zhang, H., Xie, C., Beißwenger, J., Luo, P., Geiger, A., and Li, H. Drivelm: Driving with graph visual question answering. In *European conference on computer vision*, pp. 256–274. Springer, 2024.

[^52]: Sima, C., Chitta, K., Yu, Z., Lan, S., Luo, P., Geiger, A., Li, H., and Alvarez, J. M. Centaur: Robust end-to-end autonomous driving with test-time training. *arXiv preprint arXiv:2503.11650*, 2025.

[^53]: Song, Z., Jia, C., Liu, L., Pan, H., Zhang, Y., Wang, J., Zhang, X., Xu, S., Yang, L., and Luo, Y. Don’t shake the wheel: Momentum-aware planning in end-to-end autonomous driving. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pp. 22432–22441, 2025.

[^54]: Tian, X., Gu, J., Li, B., Liu, Y., Wang, Y., Zhao, Z., Zhan, K., Jia, P., Lang, X., and Zhao, H. Drivevlm: The convergence of autonomous driving and large vision-language models. *arXiv preprint arXiv:2402.12289*, 2024.

[^55]: Tian, Y., Yang, S., Zeng, J., Wang, P., Lin, D., Dong, H., and Pang, J. Predictive inverse dynamics models are scalable learners for robotic manipulation. In *The Thirteenth International Conference on Learning Representations*, 2025.

[^56]: Van Den Oord, A., Vinyals, O., et al. Neural discrete representation learning. *Advances in neural information processing systems*, 30, 2017.

[^57]: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. *Advances in neural information processing systems*, 30, 2017.

[^58]: Wang, S., Yu, Z., Jiang, X., Lan, S., Shi, M., Chang, N., Kautz, J., Li, Y., and Alvarez, J. M. Omnidrive: A holistic vision-language dataset for autonomous driving with counterfactual reasoning. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pp. 22442–22452, 2025a.

[^59]: Wang, X., Zhang, X., Luo, Z., Sun, Q., Cui, Y., Wang, J., Zhang, F., Wang, Y., Li, Z., Yu, Q., et al. Emu3: Next-token prediction is all you need. *arXiv preprint arXiv:2409.18869*, 2024.

[^60]: Wang, Y., Li, X., Wang, W., Zhang, J., Li, Y., Chen, Y., Wang, X., and Zhang, Z. Unified vision-language-action model. *arXiv preprint arXiv:2506.19850*, 2025b.

[^61]: Wang, Y., Wu, S., Zhang, Y., Yan, S., Liu, Z., Luo, J., and Fei, H. Multimodal chain-of-thought reasoning: A comprehensive survey. *arXiv preprint arXiv:2503.12605*, 2025c.

[^62]: Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting elicits reasoning in large language models. *Advances in neural information processing systems*, 35:24824–24837, 2022.

[^63]: Weng, X., Ivanovic, B., Wang, Y., Wang, Y., and Pavone, M. Para-drive: Parallelized architecture for real-time autonomous driving. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 15449–15458, 2024.

[^64]: Wu, P., Jia, X., Chen, L., Yan, J., Li, H., and Qiao, Y. Trajectory-guided control prediction for end-to-end autonomous driving: A simple yet strong baseline. *Advances in Neural Information Processing Systems*, 35:6119–6132, 2022.

[^65]: Xia, Y., Wang, R., Liu, X., Li, M., Yu, T., Chen, X., McAuley, J., and Li, S. Beyond chain-of-thought: A survey of chain-of-x paradigms for llms. In *Proceedings of the 31st International Conference on Computational Linguistics*, pp. 10795–10809, 2025.

[^66]: Xing, S., Qian, C., Wang, Y., Hua, H., Tian, K., Zhou, Y., and Tu, Z. Openemma: Open-source multimodal model for end-to-end autonomous driving. In *Proceedings of the Winter Conference on Applications of Computer Vision*, pp. 1001–1009, 2025a.

[^67]: Xing, Z., Zhang, X., Hu, Y., Jiang, B., He, T., Zhang, Q., Long, X., and Yin, W. Goalflow: Goal-driven flow matching for multimodal trajectories generation in end-to-end autonomous driving. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, 2025b.

[^68]: Xu, Z., Zhang, Y., Xie, E., Zhao, Z., Guo, Y., Wong, K.-Y. K., Li, Z., and Zhao, H. Drivegpt4: Interpretable end-to-end autonomous driving via large language model. *IEEE Robotics and Automation Letters*, 2024.

[^69]: Yang, Z., Chai, Y., Jia, X., Li, Q., Shao, Y., Zhu, X., Su, H., and Yan, J. Drivemoe: Mixture-of-experts for vision-language-action model in end-to-end autonomous driving. *arXiv preprint arXiv:2505.16278*, 2025a.

[^70]: Yang, Z., Jia, X., Li, Q., Yang, X., Yao, M., and Yan, J. Raw2drive: Reinforcement learning with aligned world models for end-to-end autonomous driving (in carla v2). *arXiv preprint arXiv:2505.16394*, 2025b.

[^71]: Ye, S., Jang, J., Jeon, B., Joo, S. J., Yang, J., Peng, B., Mandlekar, A., Tan, R., Chao, Y.-W., Lin, B. Y., et al. Latent action pretraining from videos. In *The Thirteenth International Conference on Learning Representations*, 2024.

[^72]: Yuan, Z., Qian, C., Tang, J., Chen, R., Song, Z., Sun, L., Chu, X., Cai, Y., Zhang, D., and Li, S. Autodrive- $r^{2}$: Incentivizing reasoning and self-reflection capacity for vla model in autonomous driving. *arXiv preprint arXiv:2509.01944*, 2025.

[^73]: Zawalski, M., Chen, W., Pertsch, K., Mees, O., Finn, C., and Levine, S. Robotic control via embodied chain-of-thought reasoning. In *8th Annual Conference on Robot Learning*, 2024.

[^74]: Zeng, S., Chang, X., Xie, M., Liu, X., Bai, Y., Pan, Z., Xu, M., Wei, X., and Guo, N. Futuresightdrive: Thinking visually with spatio-temporal cot for autonomous driving. *arXiv preprint arXiv:2505.17685*, 2025.

[^75]: Zhai, J.-T., Feng, Z., Du, J., Mao, Y., Liu, J.-J., Tan, Z., Zhang, Y., Ye, X., and Wang, J. Rethinking the open-loop evaluation of end-to-end autonomous driving in nuscenes. *arXiv preprint arXiv:2305.10430*, 2023.

[^76]: Zhang, B., Song, N., Jin, X., and Zhang, L. Bridging past and future: End-to-end autonomous driving with historical prediction and planning. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pp. 6854–6863, 2025a.

[^77]: Zhang, K., Tang, Z., Hu, X., Pan, X., Guo, X., Liu, Y., Huang, J., Yuan, L., Zhang, Q., Long, X.-X., et al. Epona: Autoregressive diffusion world model for autonomous driving. *arXiv preprint arXiv:2506.24113*, 2025b.

[^78]: Zhang, R., Isola, P., Efros, A. A., Shechtman, E., and Wang, O. The unreasonable effectiveness of deep features as a perceptual metric. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, 2018.

[^79]: Zhao, Q., Lu, Y., Kim, M. J., Fu, Z., Zhang, Z., Wu, Y., Li, Z., Ma, Q., Han, S., Finn, C., et al. Cot-vla: Visual chain-of-thought reasoning for vision-language-action models. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pp. 1702–1713, 2025a.

[^80]: Zhao, Z., Fu, T., Wang, Y., Wang, L., and Lu, H. From forecasting to planning: Policy world model for collaborative state-action prediction. In *The Thirty-ninth Annual Conference on Neural Information Processing Systems*, 2025b.

[^81]: Zhou, X., Han, X., Yang, F., Ma, Y., and Knoll, A. C. Opendrivevla: Towards end-to-end autonomous driving with large vision language action model. *arXiv preprint arXiv:2503.23463*, 2025a.

[^82]: Zhou, X., Liang, D., Tu, S., Chen, X., Ding, Y., Zhang, D., Tan, F., Zhao, H., and Bai, X. Hermes: A unified self-driving world model for simultaneous 3d scene understanding and generation. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pp. 27817–27827, 2025b.

[^83]: Zhou, Y., Huang, L., Bu, Q., Zeng, J., Li, T., Qiu, H., Zhu, H., Guo, M., Qiao, Y., and Li, H. Embodied understanding of driving scenarios. In *European Conference on Computer Vision*, pp. 129–148. Springer, 2024.

[^84]: Zhou, Z., Cai, T., Zhao, S. Z., Zhang, Y., Huang, Z., Zhou, B., and Ma, J. Autovla: A vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. *arXiv preprint arXiv:2506.13757*, 2025c.

[^85]: Zitkovich, B., Yu, T., Xu, S., Xu, P., Xiao, T., Xia, F., Wu, J., Wohlhart, P., Welker, S., Wahid, A., et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In *Conference on Robot Learning*, pp. 2165–2183. PMLR, 2023.
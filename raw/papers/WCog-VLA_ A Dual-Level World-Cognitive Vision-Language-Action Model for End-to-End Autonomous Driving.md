---
title: "WCog-VLA: A Dual-Level World-Cognitive Vision-Language-Action Model for End-to-End Autonomous Driving"
source: "https://arxiv.org/html/2607.08375v1"
author:
published:
created: 2026-09-04
description:
tags:
  - "clippings"
---
Xuerun Yan Affiliation: Tongji University, China Affiliation: Nanyang Technological University, Singapore  
<sup><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\dagger"><semantics><mo>†</mo> <annotation>\dagger</annotation></semantics></math></sup> Equal contribution   <sup><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\boxtimes"><semantics><mo>⊠</mo> <annotation>\boxtimes</annotation></semantics></math></sup> Corresponding author    Zhexi Lian Affiliation: Tongji University, China    Nuoheng Zhang Affiliation: Tongji University, China    Shiyu Fang Affiliation: Tongji University, China    Haoran Wang Affiliation: Tongji University, China    Chen Lv Affiliation: Nanyang Technological University, Singapore  
<sup><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\dagger"><semantics><mo>†</mo> <annotation>\dagger</annotation></semantics></math></sup> Equal contribution   <sup><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\boxtimes"><semantics><mo>⊠</mo> <annotation>\boxtimes</annotation></semantics></math></sup> Corresponding author    Jia Hu <sup><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\boxtimes"><semantics><mo>⊠</mo> <annotation>\boxtimes</annotation></semantics></math></sup> Affiliation: Tongji University, China    Binyang Song <sup><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\boxtimes"><semantics><mo>⊠</mo> <annotation>\boxtimes</annotation></semantics></math></sup> Affiliation: Nanyang Technological University, Singapore  
<sup><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\dagger"><semantics><mo>†</mo> <annotation>\dagger</annotation></semantics></math></sup> Equal contribution   <sup><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\boxtimes"><semantics><mo>⊠</mo> <annotation>\boxtimes</annotation></semantics></math></sup> Corresponding author

###### Abstract

Vision-Language-Action (VLA) models have advanced end-to-end autonomous driving. However, existing methods either lack comprehensive world cognition or suffer from fragmented world foresight, inherently confining these models to reactive driving. To address this limitation, we propose WCog-VLA, a novel dual-level World-Cognitive VLA framework that successfully bridges semantic world forecasting with generative world evolution to achieve proactive autonomous driving. At the semantic level, WCog-VLA unifies world cognition and reasoning by incorporating 3D spatial perception and injecting agent tokens to capture the world dynamics, while concurrently enabling Game-theoretic Chain-of-Thought (Game-CoT) reasoning. At the generative level, we introduce the Aligned Decoupled Diffusion Transformer (ADDT) as a powerful generative world model that synthesizes physically-plausible joint multi-agent trajectories. Through scene representation alignment, ADDT reduces the number of denoising steps required and thus significantly accelerates inference. To facilitate strategic reasoning, we further construct a large-scale dataset featuring 85k Game-CoT annotations. Extensive experiments on the NAVSIM benchmark demonstrate that WCog-VLA achieves a State-Of-The-Art (SOTA) PDMS score of 92.9.

###### Keywords:

End-to-end autonomous driving Vision-Language-Action World cognition

## 1 Introduction

End-to-end (E2E) autonomous driving has emerged as a dominant paradigm by directly mapping raw sensory inputs to planned trajectories [^26] [^19] [^20] [^5] within a unified and differentiable framework. Although these E2E models show remarkable performance in common scenarios, they often struggle in complex or long-tail situations [^4] [^63]. This fragility arises from insufficient causal reasoning and world knowledge, leaving the models unable to fully understand and reason about the surrounding environments.

To address these long-tail challenges, Vision-Language-Models (VLMs) [^1] [^2] have been increasingly integrated into E2E autonomous driving frameworks [^25] [^47] [^53]. Equipped with extensive world knowledge and strong reasoning capabilities, VLMs significantly advance scene comprehension in complex driving scenarios [^27] [^60]. Building upon this foundation, Vision-Language-Action (VLA) models further extend VLM capabilities to action generation. Existing VLA approaches typically operate in two ways. The first formulates action outputs as autoregressive sequence generation (Fig. 1(a)), producing either discrete text tokens [^37] [^39] [^42] or learned action codes [^64]. Alternatively, the second utilizes VLMs as cognitive encoders and attaches dedicated action decoders (Fig. 1(b)) to generate continuous trajectories [^13] [^52], *e.g*., diffusion models [^32] [^23].

![[ECCV_intro.png|Refer to caption]]

Figure 1: Four paradigms of leveraging VLM in E2E autonomous driving. Our method (d) advances existing frameworks to enable proactive driving by establishing a dual-level world cognition with the integration of semantic forecasting and generative evolution.

Despite these advancements, existing VLA models still face several challenges: (1) *Lack of 3D spatial awareness.* Relying primarily on 2D image features, current models lack structured 3D spatial representations of surrounding road participants [^63] [^38], which are essential for accurate spatial reasoning and precise ego planning. (2) *Insufficient world cognition.* Existing methods struggle to adequately represent world states and forecast future dynamics [^46], such as the intentions of surrounding agents. This confines current methods to reactive rather than proactive driving (Fig. 1(a),(b)). While some works incorporate world cognition via VLM hidden states [^28] [^54], they treat world modeling as an auxiliary semantic task while neglecting generative-level world evolution. This fragmented world foresight (Fig. 1(c)) overlooks the reciprocal interplay between the ego and surrounding agents, failing to synthesize joint interactive trajectories from a world-generative perspective. (3) *Absence of strategic social reasoning.* Current reasoning mechanisms mainly focus on static scene descriptions [^7] [^39], lacking the game-theoretic ‘if-what’ imagination required for proactive social interaction among traffic participants. Motivated by these limitations, we seek to answer a key question: *How can we endow VLA models with comprehensive world cognition across both semantic-level forecasting and generative-level evolution, thereby enabling proactive driving?*

To address these challenges, we propose WCog-VLA, a novel VLA framework featuring dual-level World Cognition (Fig. 1(d)). WCog-VLA achieves comprehensive world cognition by bridging the gap between semantic-level forecasting and generative-level physical evolution. At the semantic level, WCog-VLA first embeds 3D spatial priors into the VLM, establishing a cognitive foundation for structured environment understanding. To operationalize the semantic world understanding, we decouple the VLM’s output hidden states into two functional roles. The cognition role encapsulates the understanding of current world states and future world dynamics. Meanwhile, the reasoning role drives a four-step game-theoretic reasoning process. This strategic reasoning mechanism transforms the ego vehicle from a passive observer to a proactive negotiator in social driving scenarios. At the generative level, we introduce the Aligned Decoupled Diffusion Transformer (ADDT) as our generative world model. Conditioned on the VLM’s hidden states, ADDT employs a decoupled encoder-decoder architecture to generate multi-agent trajectories. Specifically, the condition encoder explicitly aligns its latent space with dynamic scene representation. Guided by this alignment, the generation decoder efficiently synthesizes joint multi-agent trajectories, firmly grounding ego-planning within predicted interactive future dynamics. In summary, our main contributions are as follows:

1\. Dual-level world cognition framework. We propose WCog-VLA that bridges semantic-level forecasting with generative-level evolution, enabling proactive autonomous driving.

2\. Generative world model. We introduce ADDT as a generative world model, which synthesizes physically-plausible joint multi-agent trajectories with efficient inference time.

3\. Game-theoretic reasoning dataset. We construct Game-CoT, a Game-theoretic Chain-of-Thought reasoning dataset with 85k samples that fills the gap in game-theoretic reasoning supervision for social driving.

4\. State-of-the-art (SOTA) performance. WCog-VLA achieves a SOTA PDMS score of 92.9 on the NAVSIM [^10] benchmark.

## 2 Related Work

#### End-to-End Autonomous Driving.

E2E autonomous driving directly maps sensory inputs to trajectories within unified frameworks [^8] [^19] [^33]. Pioneering works like UniAD [^20] and VAD [^26] integrate perception and planning using dense Bird’s-Eye-View (BEV) representations, while SparseAD [^45] improves efficiency via sparse queries. Recent paradigms, *e.g*., GenAD [^61], DiffusionDrive [^35], and VADv2 [^5], introduce generative and probabilistic models for multi-modal trajectory planning [^62]. However, current E2E models are limited by training data coverage, frequently failing in long-tail scenarios. This fragility primarily stems from an inherent lack of semantic reasoning and environment understanding capabilities, motivating the integration of VLMs into E2E driving.

#### VLA for Autonomous Driving.

VLMs are initially applied as high-level semantic interpreters for scenario understanding, such as DriveGPT4 [^55] and DriveLM [^47]. Building upon this, unified VLA models have been proposed to directly map multi-modal inputs to driving actions [^25], exemplified by EMMA [^22] [^53], SimLingo [^44], and DriveMoE [^56]. While early VLA models driving actions directly as text [^37] [^39], later work such as AutoVLA [^64] introduces autoregressive generation of action tokens. More recently, studies integrate VLM with generative planners [^49] [^29] to mitigate modal collapse between text and actions, *e.g*., VAE-based ORION [^13] and diffusion-based ReCogDrive [^32]. However, current VLA methods predominantly remain reactive observers. Lacking explicit world cognition and future forecasts, they fail to anticipate dynamic changes in complex social scenarios, motivating the integration of comprehensive world cognition into the VLA framework.

#### World Cognition Building for VLA-based Autonomous Driving.

The latest VLA studies [^36] [^54] [^24] [^51] have sought to incorporate world cognition. One branch of research focuses on enhancing spatial awareness of VLMs [^9] [^16], *e.g*., DrivePI [^38] introduces spatial-aware world cognition, and SGDrive [^28] builds world features around a scene-agent-goal hierarchy. Another branch leverages generative forecasting for future foresight, employing future image generation as an auxiliary objective, *e.g*., UniDrive-WM [^54] and DriveVLA-W0 [^30]. However, these methods exhibit fragmented world foresight. Whether relying on semantic or image forecasting, they treat world evolution as a supervised perceptual task and lack joint multi-agent planning, failing to model interactive physical behaviors from a generative world perspective. To bridge this gap, WCog-VLA introduces a unified, dual-level world cognition framework. At the semantic level, the explicit agent tokens encapsulate the cognition of world dynamics. At the generative level, our ADDT synthesizes joint multi-agent trajectories, manifesting explicit world cognition within the generative execution stage.

![[ECCV_Frame.png|Refer to caption]]

Figure 2: Overview of WCog-VLA. Our framework achieves dual-level world cognition by tightly coupling a multi-modal VLM backbone with a generative world model ADDT. The VLM integrates vision, text, and agent tokens to perform Game-CoT reasoning and semantic world forecasting, while the ADDT translates these cognitive representations to generate physically-plausible, joint multi-agent trajectories.

## 3 WCog-VLA

As shown in Fig. 2, our WCog-VLA consists of two tightly coupled components that bridge semantic-world understanding and reasoning with generative trajectory synthesis. First, a VLM Backbone is constructed on a multi-modal architecture to jointly process multi-view camera inputs and textual instructions. These heterogeneous tokens are fused to facilitate a game-theoretic Game-CoT reasoning process, enabling structured reasoning over scene context and agent interactions. To explicitly model spatial structure and agent dynamics, the VLM backbone further incorporates agent tokens from a 3D perception module. A specialized world head then decodes the aggregated agent tokens to current 3D perception and future trajectory predictions of surrounding agents, enabling explicit semantic-level world cognition. Second, to seamlessly bridge semantic intent with physically feasible motion, we employ the ADDT as a generative world model. By intrinsically aligning its latent space with implicit scene dynamics through a structured conditioning mechanism, ADDT generates high-fidelity, joint multi-agent trajectories. Collectively, the proposed framework consistently translates explicit semantic world cognition into coherent and dynamically plausible trajectory generation. Detailed model architectures are provided in this section.

### 3.1 VLM Backbone

#### Model Inputs and Base VLM Model.

The VLM takes multi-view camera images, instructions, and ego-vehicle states as inputs. The visual input $\mathcal{I}=\{I^{i}\}_{i=1}^{6}$ comprises six surround-view images. The instruction $l_{\text{ins}}$ provides navigation commands (*e.g*., ‘turn right’). The ego state $\mathcal{S}=\{v,a,\mathcal{T}_{\text{hist}}\}$ encapsulates the current velocity $v$, acceleration $a$, and a 2-second historical trajectory $\mathcal{T}_{\text{hist}}$ sampled at 2 Hz. To process these multi-modal inputs, we adopt InternVL3-2B [^65] as our VLM backbone, utilizing a 300M-parameter InternViT vision encoder and a Qwen2.5 Large Language Model (LLM).

#### 3D Spatial Perception.

We extend 2D VLM perception into the 3D domain to enable explicit 3D spatial perception. Specifically, the multi-view camera features extracted by the vision encoder are lifted into a BEV representation $\mathcal{F}_{\text{BEV}}$ via an off-the-shelf BEV encoder from BEVFormer [^34]. To extract structured object representations, we employ a TrackFormer [^20] that maps dense BEV features to sparse agent-centric tokens. By performing cross-attention between learnable agent queries $Q_{\text{agent}}$ and $\mathcal{F}_{\text{BEV}}$, the module extracts a set of $N_{a}$ agent tokens $\mathcal{T}_{\text{agent}}=\{t_{\text{agent}}^{j}\}_{j=1}^{N_{a}}$, where $N_{a}$ denotes the number of detected agents. These explicit agent tokens capture spatial locations and geometric features, providing structured inputs for subsequent multi-modal reasoning.

#### Unified World Cognition and Reasoning.

To achieve comprehensive scene understanding, the vision ($\mathcal{T}_{\text{vision}}$), text ($\mathcal{T}_{\text{text}}$), and agent tokens ($\mathcal{T}_{\text{agent}}$) are concatenated along the sequence dimension and fed into the LLM to model multi-modal interaction and fusion. The resulting hidden states are defined as:

$$
O_{\text{vision}},O_{\text{text}},O_{\text{agent}}=\text{LLM}([\mathcal{T}_{\text{vision}},\mathcal{T}_{\text{text}},\mathcal{T}_{\text{agent}}])
$$

We decouple these output tokens of hidden states to support two distinct downstream tasks. Specifically, $O_{\text{agent}}$ encapsulates semantic-level world cognition, which is routed to a specialized world head for current 3D perception and future trajectory prediction of surrounding agents. Meanwhile, $O_{\text{vision}}$ and $O_{\text{text}}$ are processed by the language modeling head to generate textual responses. Trained via our Game-CoT reasoning paradigm, the model can output explicit game-theoretic reasoning processes in its textual responses.

![[ADDT.png|Refer to caption]]

Figure 3: Illustration of the ADDT. The ADDT features a decoupled architecture comprising a condition encoder with representation alignment and a generation decoder.

### 3.2 Aligned Decoupled Diffusion Transformer

While diffusion transformers exhibit remarkable generation quality, they suffer from an optimization dilemma in single-network architectures [^41]: the encoding of low-frequency abstract semantics conflicts with the decoding of high-frequency continuous details [^48]. In autonomous driving, this manifests as a tension between modeling complex multi-agent interactions and generating precise trajectories. To resolve this challenge and bridge VLM semantic cognition with physical actions, we propose ADDT (Fig. 3), which features a decoupled architecture comprising a specialized condition encoder and a dedicated generation decoder.

#### Condition Encoder.

The condition encoder focuses on extracting structural and interactive semantics, decoupled from the burden of precise trajectory recovery. Let $x_{t}\in\mathbb{R}^{N_{m}\times H\times 3}$ denote the joint multi-agent action noises at diffusion timestep $t$, where $N_{m}$ is the maximum number of agents and $H$ is the planning horizon. To inject temporal and cognitive context, we construct a fused noises representation $F_{at}$ by concatenating embedded noise actions, historical ego actions $\tau_{\text{his}}$, and average-pooled VLM output tokens $\bar{F}_{\text{VLM}}$. As shown in Fig. 3, $F_{at}$ serves as the primary input for $N_{1}$ Diffusion Transformer (DiT) blocks. Within these blocks, the diffusion timestep $t$ and ego states $S$ are injected via AdaLN modulation to provide physical kinematics guidance. Concurrently, the full-sequence VLM output tokens $F_{\text{VLM}}=[O_{\text{vision}},O_{\text{text}},O_{\text{agent}}]$ are injected into cross-attention layers, providing high-level semantic cognition priors. Finally, the encoder outputs a semantic self-condition feature $z_{t}$, formulated as:

$$
z_{t}=\text{Encoder}(F_{at},t,S,F_{\text{VLM}}),\text{ }F_{at}=\text{concat}(E_{act}(x_{t}),E_{his}(\tau_{\text{his}}),\bar{F}_{\text{VLM}})
$$

The resulting feature $z_{t}$ captures semantic scene dynamics, which guides the subsequent trajectory generation process.

#### Representation Alignment.

To ensure $z_{t}$ adheres strictly to real-world dynamics, we introduce a representation alignment mechanism. Specifically, the intermediate feature $h_{i}$ from the $i$ -th DiT block of the condition encoder is aligned with a latent scene representation $r_{*}$ extracted from a pre-trained VAE encoder, thereby reducing the ‘semantic’ gap between condition encoder output and latent scene space. Following GenAD [^61], this VAE is pre-trained to reconstruct multi-agent trajectories via an MLP encoder and GRU decoder. The resulting latent space after VAE encoder captures both global traffic patterns and individual characteristics of each agent. We enforce this alignment using a cosine similarity constraint [^57] with a learnable projection MLP $h_{\phi}$:

$$
\mathcal{L}_{\text{align}}=1-\cos(r_{*},h_{\phi}(h_{i}))
$$

Crucially, this explicit alignment acts as a regularization technique, maintaining the local consistency of $z_{t}$ across adjacent denoising timesteps. It ensures that the generated trajectories are grounded in feasible scene dynamics while stabilizes semantic features throughout the progressive denoising steps.

#### Generation Decoder.

The generation decoder, comprising $N_{2}$ DiT blocks, shares the condition encoder’s architecture but focuses exclusively on recovering high-frequency geometric details. Guided by the self-condition feature in $z_{t}$, it processes the fused action noises $F_{at}$ and VLM output tokens $F_{\text{VLM}}$ to estimate the denoised multi-agent trajectories. Unlike the condition encoder, the generation decoder injects both the timestep $t$ and the self-condition feature $z_{t}$ via AdaLN modulation, enabling semantically aligned denoising. The decoding process is formulated as:

$$
x_{t-1}=\text{Decoder}(F_{at},t,z_{t},F_{\text{VLM}})
$$

### 3.3 Game-CoT Reasoning Annotation

Existing reasoning datasets often lack social interaction logic and game-theoretic analysis. To bridge this gap, we propose an automated annotation pipeline powered by advanced Qwen3-VL-Plus to generate structured reasoning across four sequential steps: (1) scene description, (2) critical object analysis, (3) game-theoretic reasoning, and (4) payoff evaluation. The final output includes the optimal ego action and the inferred responses of surrounding agents. Specifically, the game-theoretic reasoning step formulates traffic interactions as a Stackelberg game [^17], where the ego vehicle acts as the leader and surrounding agents serve as followers. Adopting a ‘if-what’ imagination, the model enumerates candidate ego actions and infers the corresponding reactions of followers. The payoff evaluation step then assesses the safety and efficiency of these hypothetical outcomes to determine the optimal strategy.

To minimize hallucinatory outputs and ensure logical consistency, we incorporate Ground-Truth (GT) actions as guiding hints. This compels the VLM to reconstruct explicit causal chains linking observed scene contexts to final GT actions. Ultimately, we construct a Game-CoT dataset comprising 85k high-quality annotations on the NAVSIM benchmark. More detailed Game-CoT annotation process is illustrated in the Supplemental Material.

### 3.4 WCog-VLA Training

![[train_stage.png|Refer to caption]]

Figure 4: Illustration of four-stage training paradigm of WCog-VLA, including three-stage supervised fine-tuning and one-stage reinforcement fine-tuning.

Fig. 4 shows our four-stage training paradigm, including Supervised Fine-Tuning (SFT) in Stages 1–3 and Reinforcement Fine-Tuning (RFT) in Stage 4.

#### 3D Perception Pre-Training.

This stage optimizes the BEV encoder and TrackFormer. Following the perception training method in UniAD [^20], we utilize a detection head for class labeling and 3D box regression. The overall detection loss combines a focal loss for classification and an $L_{1}$ loss for 3D box localization: $\mathcal{L}_{\text{s1}}=\lambda_{\text{focal}}\mathcal{L}_{\text{focal}}+\lambda_{\text{L1}}\mathcal{L}_{\text{L1}}$ with weighting coefficients $\lambda_{\text{focal}}$ and $\lambda_{\text{L1}}$.

#### VLM Supervised Fine-Tuning.

This stage optimizes the VLM for both Visual Question Answering (VQA) capability and world cognition. The language modeling is trained on a mixture of public driving VQA datasets (*e.g*., DriveLM), trajectory-specific VQA, and our Game-CoT dataset. The learned semantic world cognition are routed through a world head into current perception and future prediction of surrounding agents, which are supervised by ground-truth 3D bounding boxes, and future trajectories. The overall training objective combines the text generation loss $\mathcal{L}_{\text{LM}}$ and the world cognition loss $\mathcal{L}_{\text{world}}$:

$$
\mathcal{L}_{\text{s2}}=\mathcal{L}_{\text{LM}}+\lambda_{\text{world}}\mathcal{L}_{\text{world}}
$$
 
$$
\mathcal{L}_{\text{world}}=\frac{1}{N_{a}}\sum_{i=1}^{N_{a}}\lambda_{\text{box}}\mathcal{L}_{\text{L1}}(b_{i})+\frac{1}{N_{a}H}\sum_{i=1}^{N_{a}}\sum_{t=1}^{H}\lambda_{\text{traj}}\mathcal{L}_{\text{L1}}(\tau_{i,t})
$$

where $\mathcal{L}_{\text{LM}}$ is the standard cross-entropy loss for language modeling, $\mathcal{L}_{\text{L1}}(b_{i})$ and $\mathcal{L}_{\text{L1}}(\tau_{i,t})$ denote the $L_{1}$ losses for 3D box localization and trajectory prediction of agent $i$ at timestep $t$. The $\lambda$ terms denote the corresponding weights.

#### ADDT Supervised Fine-Tuning.

In Stage 3, we freeze the pre-trained VLM to serve as a semantic-level world model and train the ADDT for joint multi-agent trajectory generation. The training objective combines the standard $L_{2}$ denoising loss $\mathcal{L}_{\text{diff}}$ and the representation alignment loss $\mathcal{L}_{\text{align}}$:

$$
\mathcal{L}_{\text{s3}}=\mathcal{L}_{\text{diff}}+\lambda_{\text{align}}\mathcal{L}_{\text{align}}
$$
 
$$
\mathcal{L}_{\text{diff}}=\mathbb{E}_{z_{t},F_{at},\epsilon\sim\mathcal{N}(0,I)}\left[\|\mathbf{W}\odot\big(\epsilon-\epsilon_{\theta}(F_{at},t,z_{t},F_{\text{VLM}})\big)\|_{2}^{2}\right]
$$

where $\mathcal{L}_{\text{diff}}$ optimizes the generation decoder $\epsilon_{\theta}$ to predict the added noise $\epsilon$, and $\odot$ is the Hadamard product. $\mathbf{W}$ is an agent-specific weight mask applying distinct penalties ($\alpha_{\text{ego}}$ and $\alpha_{\text{surr}}$) to prioritize the generative accuracy of the ego vehicle over surrounding agents and $\lambda_{\text{align}}$ is the weighting coefficient.

#### Reinforcement Fine-Tuning.

To enable driving exploration beyond imitation [^21], we adopt DiffGRPO [^32] [^14], a diffusion-specific GRPO algorithm. The DiffGRPO loss includes an RL policy optimization loss and a Behavior Cloning (BC) loss to prevent policy collapse during exploration:

$$
L_{\text{s4}}=\underbrace{-\frac{1}{GT}\sum_{i=1}^{G}\sum_{t=1}^{T}\gamma^{t-1}\log\pi_{\theta}(x_{t-1}^{(i)}\mid x_{t}^{(i)})\hat{A}_{i}}_{L_{\text{RL}}}-\underbrace{\lambda_{bc}\frac{1}{GT}\sum_{i=1}^{G}\sum_{t=1}^{T}\log\pi_{\theta}(\tilde{x}_{t-1}^{(i)}\mid\tilde{x}_{t}^{(i)})}_{L_{\text{BC}}}
$$

where $G$ and $T$ denote the sampled group size and total denoising steps, respectively. $\gamma$ is the discount coefficient mitigating instability in early denoising steps, $\pi_{\theta}(x_{t-1}^{(i)}\mid x_{t}^{(i)})$ is each step’s conditional probability. $\hat{A}_{i}$ is the group-relative advantage, $\lambda_{bc}$ is the weight of the BC loss. $\tilde{x}_{t-1}^{(i)}$ and $\tilde{x}_{t}^{(i)}$ are sampled from the reference policy $\pi_{\text{ref}}$ (*e.g*., the model after SFT).

We design a joint reward function decoupling the ego vehicle and surrounding agents. Ego driving quality is evaluated via the NAVSIM Predictive Driving Model Score (PDMS), whereas surrounding agents are optimized for accurate motion forecasting via a negative $L_{1}$ displacement penalty. The overall reward is formulated as $r_{i}=r_{\text{PDMS}}-\lambda_{\text{surr}}\mathcal{L}_{\text{L1}}(\tau_{\text{surr}})$ where $\lambda_{\text{surr}}$ balances ego planning performance with consistent surrounding agents’ trajectory prediction.

## 4 Experiments

### 4.1 Experimental setup

#### Dataset.

We evaluate WCog-VLA on the large-scale, real-world simulation benchmarks NAVSIMv1 and NAVSIMv2 [^10]. NAVSIM is a planning-oriented dataset comprising challenging scenarios. The data is partitioned into 1,192 training (navtrain) and 136 test (navtest) scenes. To establish the foundational driving cognition of the VLM, we compile a comprehensive training mixture comprising over 158k samples from open-source driving VQA datasets, including DriveLM [^47], CODA-LM [^3], LingoQA [^40], nuScenes-QA [^43], NuInstruct [^11], and DriveGPT4 [^55]. This corpus is further augmented with 170K NAVSIM-tailored samples, including 85k trajectory-specific VQA and 85k Game-CoT reasoning samples.

#### Implementation Details.

Our training pipeline consists of four sequential stages. First, the 3D perception module is trained on NAVSIM for 1 epoch. Second, the VLM is pre-trained for 1 epoch on the 158k VQA samples, followed by 3 epochs of joint fine-tuning with the world heads using the 170K NAVSIM-tailored samples. Third, with the VLM frozen, ADDT is trained via DDPM [^18] for 200 epochs on NAVSIM. The ADDT features a symmetric 16-block DiT architecture (8 blocks each for the condition encoder and generation decoder). Representation alignment is extracted from the 6th DiT encoder block. Finally, ADDT is refined via GRPO on NAVSIM for 10 epochs using 6 group sizes. All training stages are conducted on 4 NVIDIA A100 40GB GPUs. Additional implementation details are provided in the Supplemental Material.

### 4.2 Main Results

#### Results on NAVSIM v1.

Tab. 1 presents the closed-loop evaluation of our WCog-VLA on the NAVSIM v1. Our method achieves a SOTA PDMS of 92.9, outperforming all listed standard end-to-end and VLM-based methods. Notably, despite using only camera inputs, WCog-VLA surpasses multi-modal baselines that leverage both camera and lidar inputs, yielding an improvement of 4.6 PDMS over WoTE. Furthermore, WCog-VLA demonstrates clear advantages over VLM-based methods. It outperforms two massive generalist models, QwenVL2.5 and InternVL3, by 9.6 PDMS, validating the effectiveness of our tailored architecture and driving knowledge injection. Crucially, our compact 2B model surpasses the RL-refined VLM-based methods, ReCogDrive and AutoVLA, by at least 0.8 PDMS and outperforms the 3B-parameter LatentVLA by 0.5 PDMS. These results underscore that enhancing VLA with world cognition and world-dynamics forecasting enables superior planning performance.

Beyond overall planning performance, WCog-VLA excels in safety metrics, achieving a remarkable 99.4 in NC and 98.5 in TTC. This is because our model anticipates the future intents of surrounding agents, enabling proactive safety measures and collision avoidance in complex scenarios.

Table 1: Performance comparison on NAVSIM v1 *navtest*. Our WCog-VLA is evaluated after the complete four-stage training. Metrics include NC (no at-fault collision), DAC (drivable area compliance), TTC (time-to-collision), Comf. (comfort), EP (ego progress), and PDMS (predictive driver model score). $\dagger$ indicates models fine-tuned on the NAVSIM trajectory-specific dataset.

<table><tbody><tr><td>Method</td><td>Image</td><td>Lidar</td><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Comf. <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>Constant Velocity</td><td></td><td></td><td>68.0</td><td>57.8</td><td>50.0</td><td>100</td><td>19.4</td><td>20.6</td></tr><tr><td>Ego Status MLP</td><td></td><td></td><td>93.0</td><td>77.3</td><td>83.6</td><td>100</td><td>62.8</td><td>65.6</td></tr><tr><td>VADv2- <math><semantics><msub><mi>𝒱</mi> <mn>8192</mn></msub> <annotation>\mathcal{V}_{8192}</annotation></semantics></math> <sup><a href="#fn:5">5</a></sup></td><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td><td></td><td>97.2</td><td>89.1</td><td>91.6</td><td>100</td><td>76.0</td><td>80.9</td></tr><tr><td>DrivingGPT <sup><a href="#fn:6">6</a></sup></td><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td><td></td><td>98.9</td><td>90.7</td><td>94.9</td><td>95.6</td><td>79.7</td><td>82.4</td></tr><tr><td>UniAD <sup><a href="#fn:20">20</a></sup></td><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td><td></td><td>97.8</td><td>91.9</td><td>92.9</td><td>100</td><td>78.8</td><td>83.4</td></tr><tr><td>BevDrive <sup><a href="#fn:58">58</a></sup></td><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td><td>97.7</td><td>92.5</td><td>92.9</td><td>100</td><td>78.7</td><td>83.8</td></tr><tr><td>TransFuser <sup><a href="#fn:8">8</a></sup></td><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td><td>97.7</td><td>92.8</td><td>92.8</td><td>100</td><td>79.2</td><td>84.0</td></tr><tr><td>PARA-Drive <sup><a href="#fn:50">50</a></sup></td><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td><td></td><td>97.9</td><td>92.4</td><td>93.0</td><td>99.8</td><td>79.3</td><td>84.0</td></tr><tr><td>DRAMA <sup><a href="#fn:59">59</a></sup></td><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td><td>98.0</td><td>93.1</td><td>94.8</td><td>100</td><td>80.1</td><td>85.5</td></tr><tr><td>Hydra-MDP- <math><semantics><msub><mi>𝒱</mi> <mn>8192</mn></msub> <annotation>\mathcal{V}_{8192}</annotation></semantics></math> -W-EP <sup><a href="#fn:33">33</a></sup></td><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td><td>98.3</td><td>96.0</td><td>94.6</td><td>100</td><td>78.7</td><td>86.5</td></tr><tr><td>DiffusionDrive <sup><a href="#fn:35">35</a></sup></td><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td><td>98.2</td><td>96.2</td><td>94.7</td><td>100</td><td>82.2</td><td>88.1</td></tr><tr><td>WoTE <sup><a href="#fn:31">31</a></sup></td><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td><td>98.5</td><td>96.8</td><td>94.9</td><td>99.9</td><td>81.9</td><td>88.3</td></tr><tr><td>iPad <sup><a href="#fn:15">15</a></sup></td><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td><td></td><td>98.6</td><td>98.3</td><td>94.9</td><td>100</td><td>88.0</td><td>91.7</td></tr><tr><td colspan="9">VLMs-based Methods</td></tr><tr><td>QwenVL2.5-8B <sup><a href="#fn:2">2</a></sup> <sup>†</sup></td><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td><td></td><td>97.8</td><td>92.1</td><td>92.8</td><td>100</td><td>78.3</td><td>83.3</td></tr><tr><td>InternVL3-8B <sup><a href="#fn:65">65</a></sup> <sup>†</sup></td><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td><td></td><td>97.0</td><td>92.4</td><td>91.8</td><td>100</td><td>78.9</td><td>83.3</td></tr><tr><td>ReCogDrive-2B <sup><a href="#fn:32">32</a></sup></td><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td><td></td><td>97.9</td><td>97.3</td><td>94.9</td><td>100</td><td>87.3</td><td>90.8</td></tr><tr><td>AutoVLA-3B <sup><a href="#fn:64">64</a></sup></td><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td><td></td><td>99.1</td><td>97.1</td><td>97.1</td><td>100</td><td>87.6</td><td>92.1</td></tr><tr><td>LatentVLA-3B <sup><a href="#fn:52">52</a></sup></td><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td><td></td><td>98.9</td><td>98.2</td><td>95.2</td><td>100</td><td>88.2</td><td>92.4</td></tr><tr><td>WCog-VLA-2B(ours)</td><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td><td></td><td>99.4</td><td>98.8</td><td>98.5</td><td>100</td><td>87.1</td><td>92.9</td></tr></tbody></table>

#### Results on NAVSIM v2.

Tab. 2 presents the evaluation on the NAVSIM v2 benchmark, with WCog-VLA deployed after three-stage SFT process. Our WCog-VLA achieves a SOTA Extended PDMS (EPDMS) of 85.9, outperforming DiffusionDrive by 1.6 EPDMS. Besides, WCog-VLA attains the highest safety scores in both NC and TTC, while maintaining highly competitive performance across all other metrics. These findings further confirm the effectiveness and robust generalization capability of WCog-VLA in extended driving evaluations.

Table 2: Performance comparison on NAVSIM v2 *navtest* with extended metrics. Our WCog-VLA is evaluated after three-stage SFT. Newly introduced metrics include DDC (driving direction compliance), TLC (traffic light compliance), LK (lane keeping), HC (history comfort), EC (extended comfort), and EPDMS (extended PDMS).

| Method | NC $\uparrow$ | DAC $\uparrow$ | DDC $\uparrow$ | TLC $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | LK $\uparrow$ | HC $\uparrow$ | EC $\uparrow$ | EPDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VADv2 [^5] | 97.3 | 91.7 | 98.2 | 99.9 | 77.6 | 92.7 | 66.0 | 100 | 97.4 | 76.6 |
| TransFuser [^8] | 97.7 | 92.8 | 98.3 | 99.9 | 79.2 | 92.8 | 67.6 | 100 | 95.3 | 77.8 |
| HydraMDP++ [^33] | 97.9 | 96.5 | 98.9 | 100 | 79.2 | 93.4 | 67.2 | 100 | 97.7 | 80.6 |
| ARTEMIS [^12] | 98.3 | 95.1 | 98.6 | 99.8 | 81.5 | 97.4 | 96.5 | 100 | 98.3 | 83.1 |
| ReCogDrive-8B [^32] | 98.3 | 95.2 | 99.5 | 99.8 | 87.1 | 97.5 | 96.6 | 98.3 | 86.5 | 83.6 |
| WoTE [^31] | 98.5 | 96.8 | 98.8 | 99.8 | 86.1 | 97.9 | 95.5 | 98.3 | 82.9 | 84.2 |
| DiffusionDrive [^35] | 98.0 | 96.0 | 99.5 | 99.8 | 87.7 | 97.1 | 97.2 | 98.3 | 87.6 | 84.3 |
| WCog-VLA-2B(ours) | 98.8 | 96.6 | 99.3 | 99.8 | 85.8 | 98.2 | 96.4 | 98.3 | 86.3 | 85.9 |

### 4.3 Ablation Study

#### Effect of the Four-Stage Training.

Tab. 4 ablates our four-stage training paradigm. Using only Stage 2 yields a baseline PDMS of 84.4, whereas incorporating 3D perception pre-training improves 1.1 PDMS, reflecting enhanced spatial understanding. Introducing ADDT in Stage 3 fundamentally shifts the paradigm from discrete textual output to continuous trajectory generation, resulting in a 3.8 PDMS improvement. Finally, Stage 4 RFT further optimizes the driving policy, improving 3.6 PDMS and achieving the SOTA 92.9. These consistent gains confirm that every training stage is indispensable.

Table 3: Ablation on the four-stage training process. Trajectories are generated as textual tokens via the VLM in IDs 1 and 2, and as continuous actions through ADDT in IDs 3 and 4.

| ID | Stage 1 | Stage 2 | Stage 3 | Stage 4 | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- |
| 1 |  | $\checkmark$ |  |  | 84.4 |
| 2 | $\checkmark$ | $\checkmark$ |  |  | 85.5 |
| 3 | $\checkmark$ | $\checkmark$ | $\checkmark$ |  | 89.3 |
| 4 | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | 92.9 |

Table 4: Ablation on dual-level world cognition. Cur and Fut denote current perception and future prediction supervision. Generative enables joint multi-agent trajectory synthesis; otherwise, only ego-trajectory generated.

<table><tbody><tr><th rowspan="2">ID</th><td colspan="2">Semantic</td><td rowspan="2">Generative</td><td rowspan="2">PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>Cur</td><td>Fut</td></tr><tr><th>1</th><td></td><td></td><td></td><td>86.5</td></tr><tr><th>2</th><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td><td></td><td></td><td>87.0</td></tr><tr><th>3</th><td></td><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td><td></td><td>87.2</td></tr><tr><th>4</th><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td><td></td><td>88.1</td></tr><tr><th>5</th><td></td><td></td><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td><td>87.4</td></tr><tr><th>6</th><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td><td><math><semantics><mi>✓</mi> <annotation>\checkmark</annotation></semantics></math></td><td>89.3</td></tr></tbody></table>

#### Effect of Dual-Level World Cognition.

Tab. 4 evaluates the effect of dual-level world cognition, where all variants are trained with the three-stage SFT. The baseline without either cognitive level achieves 86.5 PDMS. Integrating semantic current perception or future prediction improves the score to 87.0 and 87.2, respectively, while combining both yields 88.1. Enabling only generative-level multi-agent synthesis without semantic cognition achieves 87.4. Ultimately, unifying both semantic with generative cognition triggers a synergistic leap to 89.3 PDMS. These findings demonstrate that coupling semantic forecasting with generative evolution is essential for robust planning.

#### Effect of ADDT.

Tab. 7 evaluates our ADDT design in terms of PDMS and inference time. We compare pure VLM text generation, including direct answer ($\text{VLM}^{\text{wo/r}}$) and Game-CoT reasoning ($\text{VLM}^{\text{w/r}}$), against a Standard Diffusion Transformer (SDT) [^41] and several ADDT variants: without alignment (DDT), without the decoupled architecture (ADT), and the full ADDT. All diffusion models utilize an identical DiT backbone and are trained with three-stage SFT. Results show that ADDT with 5 denoising steps achieves a 10.7 $\times$ speedup over direct VLM text generation. Crucially, ADDT attains superior performance with fewer denoising steps. Compared to the 20-step SDT, our 5-step ADDT improves PDMS by 0.8 while accelerating inference by 3.7 $\times$. Besides, ADDT exhibits low sensitivity to the number of denoising steps: increasing the steps from 5 to 20 yields a marginal 0.3 PDMS gain. This robustness stems from the explicit alignment mechanism, which maintains consistent encoder latent features across different denoising steps and reduces the need for costly iterative refinement.

Table 5: Effect of ADDT. wo/r means VLM text output without reasoning, w/r is with reasoning.

<table><tbody><tr><th>Method</th><th>Denoise step</th><td>PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Infer Time (s) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><th><math><semantics><msup><mtext>VLM</mtext> <mtext>wo/r</mtext></msup> <annotation>\text{VLM}^{\text{wo/r}}</annotation></semantics></math></th><th></th><td>85.0</td><td>1.131</td></tr><tr><th><math><semantics><msup><mtext>VLM</mtext> <mtext>w/r</mtext></msup> <annotation>\text{VLM}^{\text{w/r}}</annotation></semantics></math></th><th></th><td>85.5</td><td>9.896</td></tr><tr><th rowspan="2">VLM+SDT <sup><a href="#fn:41">41</a></sup></th><th>5</th><td>87.4</td><td>0.105</td></tr><tr><th>20</th><td>88.5</td><td>0.388</td></tr><tr><th rowspan="2">VLM+DDT</th><th>5</th><td>87.9</td><td>0.108</td></tr><tr><th>20</th><td>88.7</td><td>0.381</td></tr><tr><th rowspan="2">VLM+ADT</th><th>5</th><td>88.6</td><td>0.103</td></tr><tr><th>20</th><td>89.1</td><td>0.392</td></tr><tr><th rowspan="2">VLM+ADDT</th><th>5</th><td>89.3</td><td>0.106</td></tr><tr><th>20</th><td>89.6</td><td>0.383</td></tr></tbody></table>

Table 6: Effect of VQA dataset.

| ID | Traj | Drive | CoT | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- |
| 1 | $\checkmark$ |  |  | 86.7 |
| 2 | $\checkmark$ | $\checkmark$ |  | 88.2 |
| 3 | $\checkmark$ |  | $\checkmark$ | 87.5 |
| 4 | $\checkmark$ | $\checkmark$ | $\checkmark$ | 89.3 |

Table 7: 3D perception effect.

| 3D perception | PDMS $\uparrow$ |
| --- | --- |
| $\times$ | 86.0 |
| $\checkmark$ | 89.3 |

#### Effect of VQA Dataset.

Tab. 7 shows that training solely on NAVSIM trajectory-specific VQA yields a baseline PDMS of 86.7. Adding open-source driving VQA (Drive) or Game-CoT reasoning data (CoT) improves PDMS to 88.2 and 87.5. Combining all three data sources achieves the highest PDMS of 89.3, confirming that incorporating diverse VQA data can improve performance.

#### Effect of 3D Perception.

Tab. 7 validates the contribution of the 3D perception module. Without explicit 3D perception, ADDT relies solely on generic VLM vision tokens, which limits spatial precision and yields a PDMS of 86.0. Incorporating the dedicated 3D perception boosts performance to 89.3.

More ablation study results (*e.g*., effect of alignment layer position) are shown in the Supplemental Material.

![[compare_with_previous.png|Refer to caption]]

Figure 5: Comparison with previous SOTA method on Navtest.

### 4.4 Qualitative Results

#### Compare with Previous SOTA Method.

Fig. 7 compares WCog-VLA with ReCogDrive [^32] in complex urban scenarios. ReCogDrive acts overly conservatively, remaining trapped in the current slow lane. Conversely, WCog-VLA identifies the slow leading bus in the current lane and changes the lane to improve efficiency, closely matching the human ground truth. This visualization demonstrates our WCog-VLA enables efficient and human-aligned driving behaviors.

#### Proactive Driving via Generative-Level World Cognition.

Fig. 7 highlights the advantages of generative world cognition. In the intersection scenario, the baseline lacks interactive foresight of the oncoming vehicle and generates an ego-only trajectory, resulting in passive deceleration to blindly avoid spurious conflicts. Conversely, our model synthesizes joint multi-agent trajectories that explicitly forecast the oncoming vehicle’s straight trajectory. This foresight enables ego vehicle to confidently execute a left turn. The results confirm that generative world cognition empowers the model to perform proactive maneuvers.

#### Explicit Semantic-Level World Cognition Representation.

Fig. 7 shows the semantic world cognition decoded via the world head. Both 3D perception and future trajectory prediction align closely with the ground truth, demonstrating the model’s cognition of current world states and future world dynamics.

More qualitative results are shown in the Supplemental Material.

## 5 Conclusion

In this work, we propose WCog-VLA, a novel VLA framework with explicit dual-level World Cognition for end-to-end autonomous driving. To bridge the gap between semantic-level forecasting and generative-level evolution, WCog-VLA tightly couples a multi-modal VLM backbone with a generative Aligned Decoupled Diffusion Transformer (ADDT). At the semantic level, our model unifies the world cognition and reasoning, enabling comprehensive world understanding and interactive game-theoretic reasoning. At the generative level, ADDT acts as a generative world model, translates the VLM’s cognitive representations into physically-plausible joint multi-agent trajectories. Extensive experiments on the NAVSIM v1 and v2 benchmarks demonstrate SOTA performance of WCog-VLA. However, the current semantic cognition focuses on agents and omits the future evolution of road geometry and map topology. Future work will incorporate these dynamics to build a more comprehensive world model.

[^1]: J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat, et al. (2023) Gpt-4 technical report. arXiv preprint arXiv:2303.08774. Cited by: §1.

[^2]: S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge, et al. (2025) Qwen3-vl technical report. arXiv preprint arXiv:2511.21631. Cited by: §1, Table 1.

[^3]: K. Chen, Y. Li, W. Zhang, Y. Liu, P. Li, R. Gao, L. Hong, M. Tian, X. Zhao, Z. Li, et al. (2025) Automated evaluation of large vision-language models on self-driving corner cases. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pp. 7817–7826. Cited by: §4.1.

[^4]: L. Chen, P. Wu, K. Chitta, B. Jaeger, A. Geiger, and H. Li (2024) End-to-end autonomous driving: challenges and frontiers. IEEE Transactions on Pattern Analysis and Machine Intelligence 46 (12), pp. 10164–10183. Cited by: §1.

[^5]: S. Chen, B. Jiang, H. Gao, B. Liao, Q. Xu, Q. Zhang, C. Huang, W. Liu, and X. Wang (2024) Vadv2: end-to-end vectorized autonomous driving via probabilistic planning. arXiv preprint arXiv:2402.13243. Cited by: §1, §2, Table 1, Table 2.

[^6]: Y. Chen, Y. Wang, and Z. Zhang (2025) Drivinggpt: unifying driving world modeling and planning with multi-modal autoregressive transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 26890–26900. Cited by: Table 1.

[^7]: H. Chi, H. Gao, Z. Liu, J. Liu, C. Liu, J. Li, K. Yang, Y. Yu, Z. Wang, W. Li, et al. (2025) Impromptu vla: open weights and open data for driving vision-language-action models. arXiv preprint arXiv:2505.23757. Cited by: §1.

[^8]: K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger (2022) Transfuser: imitation with transformer-based sensor fusion for autonomous driving. IEEE transactions on pattern analysis and machine intelligence 45 (11), pp. 12878–12895. Cited by: §2, Table 1, Table 2.

[^9]: C. Dang, J. Wang, G. Li, Z. Hou, Z. You, H. Ye, J. Ma, L. Chen, and Y. Wang (2026) SparseOccVLA: bridging occupancy and vision-language models via sparse queries for unified 4d scene understanding and planning. arXiv preprint arXiv:2601.06474. Cited by: §2.

[^10]: D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, et al. (2024) Navsim: data-driven non-reactive autonomous vehicle simulation and benchmarking. Advances in Neural Information Processing Systems 37, pp. 28706–28719. Cited by: §1, §4.1.

[^11]: X. Ding, J. Han, H. Xu, X. Liang, W. Zhang, and X. Li (2024) Holistic autonomous driving understanding by bird’s-eye-view injected multi-modal large models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13668–13677. Cited by: §4.1.

[^12]: R. Feng, N. Xi, D. Chu, R. Wang, Z. Deng, A. Wang, L. Lu, J. Wang, and Y. Huang (2025) Artemis: autoregressive end-to-end trajectory planning with mixture of experts for autonomous driving. IEEE Robotics and Automation Letters 11 (1), pp. 226–233. Cited by: Table 2.

[^13]: H. Fu, D. Zhang, Z. Zhao, J. Cui, D. Liang, C. Zhang, D. Zhang, H. Xie, B. Wang, and X. Bai (2025) Orion: a holistic end-to-end autonomous driving framework by vision-language instructed action generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 24823–24834. Cited by: §1, §2.

[^14]: D. Guo, D. Yang, H. Zhang, J. Song, P. Wang, Q. Zhu, R. Xu, R. Zhang, S. Ma, X. Bi, et al. (2025) Deepseek-r1: incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948. Cited by: §3.4.

[^15]: K. Guo, H. Liu, X. Wu, J. Pan, and C. Lv (2025) Ipad: iterative proposal-centric end-to-end autonomous driving. arXiv preprint arXiv:2505.15111. Cited by: Table 1.

[^16]: J. Han, M. Tian, J. Zhu, F. He, H. Zhang, S. Guo, D. Zhu, H. Tang, P. Xu, Y. Guo, et al. (2025) Percept-wam: perception-enhanced world-awareness-action model for robust end-to-end autonomous driving. arXiv preprint arXiv:2511.19221. Cited by: §2.

[^17]: P. Hang, C. Lv, Y. Xing, C. Huang, and Z. Hu (2020) Human-like decision making for autonomous driving: a noncooperative game theoretic approach. IEEE Transactions on Intelligent Transportation Systems 22 (4), pp. 2076–2087. Cited by: §3.3.

[^18]: J. Ho, A. Jain, and P. Abbeel (2020) Denoising diffusion probabilistic models. Advances in neural information processing systems 33, pp. 6840–6851. Cited by: §4.1.

[^19]: S. Hu, L. Chen, P. Wu, H. Li, J. Yan, and D. Tao (2022) St-p3: end-to-end vision-based autonomous driving via spatial-temporal feature learning. In European Conference on Computer Vision, pp. 533–549. Cited by: §1, §2.

[^20]: Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, et al. (2023) Planning-oriented autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 17853–17862. Cited by: §1, §2, §3.1, §3.4, Table 1.

[^21]: Z. Huang, Z. Sheng, Y. Qu, J. You, and S. Chen (2025) Vlm-rl: a unified vision language models and reinforcement learning framework for safe autonomous driving. Transportation Research Part C: Emerging Technologies 180, pp. 105321. Cited by: §3.4.

[^22]: J. Hwang, R. Xu, H. Lin, W. Hung, J. Ji, K. Choi, D. Huang, T. He, P. Covington, B. Sapp, et al. (2024) Emma: end-to-end multimodal model for autonomous driving. arXiv preprint arXiv:2410.23262. Cited by: §2.

[^23]: A. Jiang, Y. Gao, Z. Sun, Y. Wang, J. Wang, J. Chai, Q. Cao, Y. Heng, H. Jiang, Y. Dong, et al. (2025) Diffvla: vision-language guided diffusion planning for autonomous driving. arXiv preprint arXiv:2505.19381. Cited by: §1.

[^24]: A. Jiang, Y. Gao, Y. Wang, Z. Sun, S. Wang, Y. Heng, H. Sun, S. Tang, L. Zhu, J. Chai, et al. (2025) Irl-vla: training an vision-language-action policy via reward world model. arXiv preprint arXiv:2508.06571. Cited by: §2.

[^25]: B. Jiang, S. Chen, B. Liao, X. Zhang, W. Yin, Q. Zhang, C. Huang, W. Liu, and X. Wang (2024) Senna: bridging large vision-language models and end-to-end autonomous driving. arXiv preprint arXiv:2410.22313. Cited by: §1, §2.

[^26]: B. Jiang, S. Chen, Q. Xu, B. Liao, J. Chen, H. Zhou, Q. Zhang, W. Liu, C. Huang, and X. Wang (2023) Vad: vectorized scene representation for efficient autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 8340–8350. Cited by: §1, §2.

[^27]: S. Jiang, Z. Huang, K. Qian, Z. Luo, T. Zhu, Y. Zhong, Y. Tang, M. Kong, Y. Wang, S. Jiao, et al. (2025) A survey on vision-language-action models for autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4524–4536. Cited by: §1.

[^28]: J. Li, J. Wu, D. Hu, X. Huang, B. Sun, Z. Hao, X. Lang, X. Zhu, and L. Zhang (2026) SGDrive: scene-to-goal hierarchical world cognition for autonomous driving. arXiv preprint arXiv:2601.05640. Cited by: §1, §2.

[^29]: P. Li, Y. Zheng, Y. Wang, H. Wang, H. Zhao, J. Liu, X. Zhan, K. Zhan, and X. Lang (2025) Discrete diffusion for reflective vision-language-action models in autonomous driving. arXiv preprint arXiv:2509.20109. Cited by: §2.

[^30]: Y. Li, S. Shang, W. Liu, B. Zhan, H. Wang, Y. Wang, Y. Chen, X. Wang, Y. An, C. Tang, et al. (2025) DriveVLA-w0: world models amplify data scaling law in autonomous driving. arXiv preprint arXiv:2510.12796. Cited by: §2.

[^31]: Y. Li, Y. Wang, Y. Liu, J. He, L. Fan, and Z. Zhang (2025) End-to-end driving with online trajectory evaluation via bev world model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 27137–27146. Cited by: Table 1, Table 2.

[^32]: Y. Li, K. Xiong, X. Guo, F. Li, S. Yan, G. Xu, L. Zhou, L. Chen, H. Sun, B. Wang, et al. (2025) Recogdrive: a reinforced cognitive framework for end-to-end autonomous driving. arXiv preprint arXiv:2506.08052. Cited by: §1, §2, §3.4, §4.4, Table 1, Table 2.

[^33]: Z. Li, K. Li, S. Wang, S. Lan, Z. Yu, Y. Ji, Z. Li, Z. Zhu, J. Kautz, Z. Wu, et al. (2024) Hydra-mdp: end-to-end multimodal planning with multi-target hydra-distillation. arXiv preprint arXiv:2406.06978. Cited by: §2, Table 1, Table 2.

[^34]: Z. Li, W. Wang, H. Li, E. Xie, C. Sima, T. Lu, Q. Yu, and J. Dai (2024) Bevformer: learning bird’s-eye-view representation from lidar-camera via spatiotemporal transformers. IEEE Transactions on Pattern Analysis and Machine Intelligence 47 (3), pp. 2020–2036. Cited by: §3.1.

[^35]: B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang, et al. (2025) Diffusiondrive: truncated diffusion model for end-to-end autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 12037–12047. Cited by: §2, Table 1, Table 2.

[^36]: L. Liu, Z. Song, C. Jia, H. Ye, X. Hao, L. Chen, et al. (2026) DriveWorld-vla: unified latent-space world modeling with vision-language-action for autonomous driving. arXiv preprint arXiv:2602.06521. Cited by: §2.

[^37]: P. Liu, Q. Ning, X. Lu, H. Liu, W. Ma, D. She, P. Jia, X. Lang, and J. Ma (2025) OmniReason: a temporal-guided vision-language-action framework for autonomous driving. arXiv preprint arXiv:2509.00789. Cited by: §1, §2.

[^38]: Z. Liu, R. Huang, R. Yang, S. Yan, Z. Wang, L. Hou, D. Lin, X. Bai, and H. Zhao (2025) DrivePI: spatial-aware 4d mllm for unified autonomous driving understanding, perception, prediction and planning. arXiv preprint arXiv:2512.12799. Cited by: §1, §2.

[^39]: Y. Luo, F. Li, S. Xu, Z. Lai, L. Yang, Q. Chen, Z. Luo, Z. Xie, S. Jiang, J. Liu, et al. (2025) Adathinkdrive: adaptive thinking via reinforcement learning for autonomous driving. arXiv preprint arXiv:2509.13769. Cited by: §1, §1, §2.

[^40]: A. Marcu, L. Chen, J. Hünermann, A. Karnsund, B. Hanotte, P. Chidananda, S. Nair, V. Badrinarayanan, A. Kendall, J. Shotton, et al. (2024) Lingoqa: visual question answering for autonomous driving. In European Conference on Computer Vision, pp. 252–269. Cited by: §4.1.

[^41]: W. Peebles and S. Xie (2023) Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4195–4205. Cited by: §3.2, §4.3, Table 7.

[^42]: K. Qian, S. Jiang, Y. Zhong, Z. Luo, Z. Huang, T. Zhu, K. Jiang, M. Yang, Z. Fu, J. Miao, et al. (2025) Agentthink: a unified framework for tool-augmented chain-of-thought reasoning in vision-language models for autonomous driving. arXiv preprint arXiv:2505.15298 1 (2), pp. 3. Cited by: §1.

[^43]: T. Qian, J. Chen, L. Zhuo, Y. Jiao, and Y. Jiang (2024) Nuscenes-qa: a multi-modal visual question answering benchmark for autonomous driving scenario. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38, pp. 4542–4550. Cited by: §4.1.

[^44]: K. Renz, L. Chen, E. Arani, and O. Sinavski (2025) Simlingo: vision-only closed-loop autonomous driving with language-action alignment. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 11993–12003. Cited by: §2.

[^45]: W. Sun, X. Lin, Y. Shi, C. Zhang, H. Wu, and S. Zheng (2025) Sparsedrive: end-to-end autonomous driving via sparse scene representation. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pp. 8795–8801. Cited by: §2.

[^46]: R. Team, Z. Gao, Q. Wang, Y. Zeng, J. Zhu, K. L. Cheng, Y. Li, H. Wang, Y. Xu, S. Ma, et al. (2026) Advancing open-source world models. arXiv preprint arXiv:2601.20540. Cited by: §1.

[^47]: X. Tian, J. Gu, B. Li, Y. Liu, Y. Wang, Z. Zhao, K. Zhan, P. Jia, X. Lang, and H. Zhao (2024) Drivevlm: the convergence of autonomous driving and large vision-language models. arXiv preprint arXiv:2402.12289. Cited by: §1, §2, §4.1.

[^48]: S. Wang, Z. Tian, W. Huang, and L. Wang (2025) Ddt: decoupled diffusion transformer. arXiv preprint arXiv:2504.05741. Cited by: §3.2.

[^49]: Y. Wang, W. Luo, J. Bai, Y. Cao, T. Che, K. Chen, Y. Chen, J. Diamond, Y. Ding, W. Ding, et al. (2025) Alpamayo-r1: bridging reasoning and action prediction for generalizable autonomous driving in the long tail. arXiv preprint arXiv:2511.00088. Cited by: §2.

[^50]: X. Weng, B. Ivanovic, Y. Wang, Y. Wang, and M. Pavone (2024) Para-drive: parallelized architecture for real-time autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15449–15458. Cited by: Table 1.

[^51]: J. Xiao, Y. Yang, X. Chang, R. Chen, F. Xiong, M. Xu, W. Zheng, and Q. Zhang (2025) World-env: leveraging world model as a virtual environment for vla post-training. arXiv preprint arXiv:2509.24948. Cited by: §2.

[^52]: C. Xie, B. Sun, T. Li, J. Wu, Z. Hao, X. Lang, and H. Li (2026) LatentVLA: efficient vision-language models for autonomous driving via latent action prediction. arXiv preprint arXiv:2601.05611. Cited by: §1, Table 1.

[^53]: S. Xing, C. Qian, Y. Wang, H. Hua, K. Tian, Y. Zhou, and Z. Tu (2025) Openemma: open-source multimodal model for end-to-end autonomous driving. In Proceedings of the Winter Conference on Applications of Computer Vision, pp. 1001–1009. Cited by: §1, §2.

[^54]: Z. Xiong, X. Ye, B. Yaman, S. Cheng, Y. Lu, J. Luo, N. Jacobs, and L. Ren (2026) UniDrive-wm: unified understanding, planning and generation world model for autonomous driving. arXiv preprint arXiv:2601.04453. Cited by: §1, §2.

[^55]: Z. Xu, Y. Zhang, E. Xie, Z. Zhao, Y. Guo, K. K. Wong, Z. Li, and H. Zhao (2024) Drivegpt4: interpretable end-to-end autonomous driving via large language model. IEEE Robotics and Automation Letters 9 (10), pp. 8186–8193. Cited by: §2, §4.1.

[^56]: Z. Yang, Y. Chai, X. Jia, Q. Li, Y. Shao, X. Zhu, H. Su, and J. Yan (2025) DriveMoE: mixture-of-experts for vision-language-action model in end-to-end autonomous driving. arXiv preprint arXiv:2505.16278. Cited by: §2.

[^57]: S. Yu, S. Kwak, H. Jang, J. Jeong, J. Huang, J. Shin, and S. Xie (2024) Representation alignment for generation: training diffusion transformers is easier than you think. arXiv preprint arXiv:2410.06940. Cited by: §3.2.

[^58]: Z. Yu, J. Li, Y. Wei, Y. Lyu, and X. Tan (2025) Combining camera–lidar fusion and motion planning using bird’s-eye view representation for end-to-end autonomous driving. Drones 9 (4), pp. 281. Cited by: Table 1.

[^59]: C. Yuan, Z. Zhang, J. Sun, S. Sun, Z. Huang, C. D. W. Lee, D. Li, Y. Han, A. Wong, K. P. Tee, et al. (2024) Drama: an efficient end-to-end motion planner for autonomous driving with mamba. arXiv preprint arXiv:2408.03601. Cited by: Table 1.

[^60]: D. Zhang, J. Sun, C. Hu, X. Wu, Z. Yuan, R. Zhou, F. Shen, and Q. Zhou (2025) Pure vision language action (vla) models: a comprehensive survey. arXiv preprint arXiv:2509.19012. Cited by: §1.

[^61]: W. Zheng, R. Song, X. Guo, C. Zhang, and L. Chen (2024) Genad: generative end-to-end autonomous driving. In European Conference on Computer Vision, pp. 87–104. Cited by: §2, §3.2.

[^62]: Y. Zheng, R. Liang, K. Zheng, J. Zheng, L. Mao, J. Li, W. Gu, R. Ai, S. E. Li, X. Zhan, et al. (2025) Diffusion-based planning for autonomous driving with flexible guidance. arXiv preprint arXiv:2501.15564. Cited by: §2.

[^63]: X. Zhou, X. Han, F. Yang, Y. Ma, V. Tresp, and A. Knoll (2025) Opendrivevla: towards end-to-end autonomous driving with large vision language action model. arXiv preprint arXiv:2503.23463. Cited by: §1, §1.

[^64]: Z. Zhou, T. Cai, S. Z. Zhao, Y. Zhang, Z. Huang, B. Zhou, and J. Ma (2025) Autovla: a vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. arXiv preprint arXiv:2506.13757. Cited by: §1, §2, Table 1.

[^65]: J. Zhu, W. Wang, Z. Chen, Z. Liu, S. Ye, L. Gu, H. Tian, Y. Duan, W. Su, J. Shao, et al. (2025) Internvl3: exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479. Cited by: §3.1, Table 1.
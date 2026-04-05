---
title: "UniUGP: Unifying Understanding, Generation, and Planing For End-to-end Autonomous Driving"
source: "https://arxiv.org/html/2512.09864v1"
author:
published:
created: 2026-04-05
description:
tags:
  - "clippings"
---
ByteDance Seed See [Contributions](https://arxiv.org/html/2512.09864v1#Sx1 "Contributions ‣ UniUGP: Unifying Understanding, Generation, and Planing For End-to-end Autonomous Driving") section for a full author list.

###### Abstract

Autonomous driving (AD) systems struggle in long-tail scenarios due to limited world knowledge and weak visual dynamic modeling. Existing vision-language-action (VLA)-based methods cannot leverage unlabeled videos for visual causal learning, while world model-based methods lack reasoning capabilities from large language models. In this paper, we construct multiple specialized datasets providing reasoning and planning annotations for complex scenarios. Then, a unified Understanding-Generation-Planning framework, named UniUGP, is proposed to synergize scene reasoning, future video generation, and trajectory planning through a hybrid expert architecture. By integrating pre-trained VLMs and video generation models, UniUGP leverages visual dynamics and semantic reasoning to enhance planning performance. Taking multi-frame observations and language instructions as input, it produces interpretable chain-of-thought reasoning, physically consistent trajectories, and coherent future videos. We introduce a four-stage training strategy that progressively builds these capabilities across multiple existing AD datasets, along with the proposed specialized datasets. Experiments demonstrate state-of-the-art performance in perception, reasoning, and decision-making, with superior generalization to challenging long-tail situations.

\[Project Page:\] [https://seed-uniugp.github.io/](https://seed-uniugp.github.io/) \[Date:\]December 10, 2025

Table 1: Comparison of Different Methods. "VLA" refers to the use of additional models to predict more accurate trajectories, which is different from VLM’s text-based prediction. "Reason." refers to whether the model can generate a chain of thoughts. "Inter." refers to whether the model can change its trajectory based on human instructions. "Cont." refers to whether the token is a continuous or discrete value. "World Model" only lists the methods that can simultaneously generate future images and trajectories. "FM." means flow matching.

<table><tbody><tr><td></td><td></td><td></td><td colspan="2">Understanding</td><td colspan="2">Generation</td><td colspan="2">Action</td></tr><tr><td>Category</td><td>Method</td><td>Model</td><td>Reason.</td><td>Inter.</td><td>Modality</td><td>Cont.</td><td>Method</td><td>Cont.</td></tr><tr><td>World Model</td><td>OccWorld <sup><a href="#fn:95">95</a></sup></td><td>-</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>Occ</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>Codebook</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td>World Model</td><td>Epona <sup><a href="#fn:91">91</a></sup></td><td>-</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>Video</td><td><math><semantics><mo>√</mo> <annotation>\surd</annotation></semantics></math></td><td>FM.</td><td><math><semantics><mo>√</mo> <annotation>\surd</annotation></semantics></math></td></tr><tr><td>VLM</td><td>DriveLM <sup><a href="#fn:58">58</a></sup></td><td>Llama2</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>-</td><td>-</td><td>Text</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td>VLM</td><td>Impr. VLA <sup><a href="#fn:10">10</a></sup></td><td>Qwen2.5-VL-3B</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>-</td><td>-</td><td>Text</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td>VLM</td><td>OmniDrive <sup><a href="#fn:67">67</a></sup></td><td>LLaMA2-7B</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>-</td><td>-</td><td>Text</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td>VLA</td><td>ReCogDrive <sup><a href="#fn:37">37</a></sup></td><td>Qwen2.5-VL-3B</td><td><math><semantics><mo>√</mo> <annotation>\surd</annotation></semantics></math></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>-</td><td>-</td><td>Diffusion</td><td><math><semantics><mo>√</mo> <annotation>\surd</annotation></semantics></math></td></tr><tr><td>VLA</td><td>ORION <sup><a href="#fn:14">14</a></sup></td><td>Vicuna v1.5</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>-</td><td>-</td><td>VAE</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td>VLA</td><td>AutoVLA <sup><a href="#fn:102">102</a></sup></td><td>InternVL3-8B</td><td><math><semantics><mo>√</mo> <annotation>\surd</annotation></semantics></math></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>-</td><td>-</td><td>Codebook</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td>VLA</td><td>DriveMoE <sup><a href="#fn:86">86</a></sup></td><td><math><semantics><mi>π</mi> <annotation>\pi</annotation></semantics></math> 0</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>-</td><td>-</td><td>FM.</td><td><math><semantics><mo>√</mo> <annotation>\surd</annotation></semantics></math></td></tr><tr><td>VLA</td><td>Alpamayo-R1 <sup><a href="#fn:72">72</a></sup></td><td>Cosmos-Reason</td><td><math><semantics><mo>√</mo> <annotation>\surd</annotation></semantics></math></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>-</td><td>-</td><td>FM.</td><td><math><semantics><mo>√</mo> <annotation>\surd</annotation></semantics></math></td></tr><tr><td>Unified Model</td><td>Doe-1 <sup><a href="#fn:97">97</a></sup></td><td>Lumina-mGPT</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>Video</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>Text</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td>Unified Model</td><td>Occ-LLM <sup><a href="#fn:81">81</a></sup></td><td>Llava-7B</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>Occ</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>Text</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td>Unified Model</td><td>OccLlama <sup><a href="#fn:74">74</a></sup></td><td>Llama-7B</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>Occ</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>Text</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td>Unified Model</td><td>HERMES <sup><a href="#fn:101">101</a></sup></td><td>InternVL2-2B</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>LiDAR</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>Text</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td>Unified Model</td><td>FSDrive <sup><a href="#fn:88">88</a></sup></td><td>Qwen2-VL-2B</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>Video</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>Text</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td>Unified Model</td><td>UniUGP</td><td>Qwen2.5-VL-3B</td><td><math><semantics><mo>√</mo> <annotation>\surd</annotation></semantics></math></td><td><math><semantics><mo>√</mo> <annotation>\surd</annotation></semantics></math></td><td>Video</td><td><math><semantics><mo>√</mo> <annotation>\surd</annotation></semantics></math></td><td>FM.</td><td><math><semantics><mo>√</mo> <annotation>\surd</annotation></semantics></math></td></tr></tbody></table>

## 1 Introduction

Autonomous driving (AD) has recently made remarkable progress, especially in areas such as bird’s-eye view perception [^38] [^27] [^42] [^46] [^43], end-to-end [^25] [^40] [^30] [^8], scene reconstruction [^100] [^82] [^93] [^45], and video generation [^15] [^70] [^75] [^44]. Recently, given the superior capabilities of multimodal large language models (MLLMs) in world knowledge, reasoning ability, and interpretability, they have been widely applied in AD [^26] [^48] [^90] [^34]. One promising direction is the end-to-end vision-language-action (VLA) model [^102] [^72] [^37], which leverages pre-trained vision-language model (VLM) to directly extract scene features from visual observations and language instructions, subsequently generating vehicle control commands (e.g., speed and trajectory). This paradigm not only simplifies system architecture and minimizes information loss, but also enables the utilization of the model’s world knowledge to analyze driving environments and reason about safe decisions in complex scenarios [^6] [^32] [^31] [^68]. However, they were unable to fully utilize the large number of un-labeled driving videos, which limited their ability to learn visual causal reasoning from large-scale datasets.

In addition to the advanced VLA technology, the world model can learn visual causal reasoning by predicting the next frame of the video [^91] [^95] [^23] [^84] [^9]. The world model can learn visual causal reasoning by predicting the next frame of the video, which has been proven to be helpful in achieving the final end-to-end AD [^91] [^84] [^88]. But the world model is unable to match the world knowledge, reasoning ability, and interaction capability from large language models. Summaries of different methods are presented in Tab. 1. Unified models, which aim to bridge perception, reasoning, and action, can simultaneously combine the advantages of the world model and the VLA model. However, there are several additional issues here: 1) How to efficiently establish a unified model to fully utilize the pre-trained VLM and world model. 2) How to effectively and repeatedly utilize various driving data (such as VQA pairs, video trajectory pairs, etc.) to fully exploit the potential of the unified model. 3) How to evaluate the capabilities of the unified model, especially in terms of understanding, reasoning, and planning in complex scenarios.

To address these limitations, we propose UniUGP, a unified Understanding–Generate–Planning framework for end-to-end AD that jointly models complex scene reasoning, future video generation, and trajectory planning. Built on a hybrid expert architecture, UniUGP fully exploits the causal reasoning capabilities of pre-trained MLLMs and video generation models, while further enhancing cross-modal causal alignment through large-scale multimodal data training. Specifically, UniUGP takes natural language instructions and continuous image sequences as inputs, and outputs three complementary results: a chain-of-thought (CoT) reasoning process for interpretability, a physically consistent trajectory for safe driving, and a coherent future video for visual causal validation. To ensure output consistency and accuracy, we design a multi-term loss function that enforces CoT logical consistency, trajectory temporal smoothness, and video visual coherence. Moreover, we propose a four-stage training framework that sequentially builds foundational scene understanding, visual dynamic modeling, text-based reasoning, and multi-capability fusion, leveraging over 10 diverse AD datasets to cover common scenarios and long-tailed cases.

Our main contributions are summarized as follows:

1) We construct multiple specialized datasets for AD-oriented VLAs, which provides explanations, reasoning, and planning for complex scenarios.

2) We propose a unified Understanding-Generation-Planning (UniUGP) framework with a hybrid expert architecture, which synergizes scene reasoning, future video generation, and trajectory planning.

3) We develop a four-stage training strategy that leverages diverse AD datasets to enable the mutual enhancement of understanding, generation, and planning task capabilities.

## 2 Related Work

### 2.1 VLA for Autonomous Driving

End-to-end autonomous driving models [^24] [^30] [^83] [^39] have shown strong performance in structured environments but struggle in long-tail and unstructured scenarios due to limited generalization and lack of world knowledge. To mitigate this, recent work introduces VLMs for reasoning and scene understanding [^66], integrating them into driving frameworks to enhance adaptability in complex situations [^102] [^37] [^10] [^29] [^14] [^67]. Early attempts used VLMs to generate high-level meta-actions or abstract driving decisions [^6] [^32] [^31] [^68], which guided modular or end-to-end planners but disrupted joint optimization across perception, decision, and control. Later, the VLA frameworks directly mapped visual language inputs to trajectories. Impromptu VLA [^10] trained VLMs on text-based trajectory representations, while AutoVLA [^102] discretized trajectories into action tokens decoded into continuous paths. Diffusion-based models such as ReCogDrive, DiffVLA, and ORION [^37] [^29] [^14] further bridged the gap between semantic reasoning and continuous trajectory generation. Despite progress, the existing methods are often unable to utilize unlabeled video data to learn visual causal reasoning.

### 2.2 World Models for Autonomous Driving

World models [^92] [^51] [^103] [^2] [^80] aim to infer ego-centric states and predict dynamic surroundings from historical observations, enabling accurate future prediction and planning. In autonomous driving, world models are primarily applied in three areas: driving scenario generation [^52] [^33] [^17] [^36], planning [^20] [^73] [^95], and representation learning [^50] [^85] [^87]. For driving scenario generation, most prior works rely on diffusion models. GAIA-1 [^21] is a notable exception that combines a progressive next-token predictor with an auxiliary diffusion image decoder. More recently, Epona [^91] advances this direction by employing an autoregressive diffusion model to unify world modeling and planning. However, existing approaches face two major limitations. First, they do not fully exploit the complementary strengths of pre-trained VLMs and video generation models, missing opportunities to leverage both linguistic reasoning and visual dynamics modeling. Second, most methods are trained and evaluated on limited single-dataset scenarios, restricting their generalization ability. Our method addresses these gaps by integrating knowledge from both pre-trained VLMs and video generative models, while scaling training across diverse datasets to unlock emergent capabilities.

### 2.3 Unified Models

Unified models aim to dissolve modular boundaries across understanding and generation. [^12] [^57] [^53] [^61] [^62] [^64] [^71] [^76] [^49] [^77] [^78] [^99]. Extending to embodied intelligence, works like [^47] and WALL-OSS [^89] integrate action modules into unified frameworks. F1 uses a three-expert MoT to synthesize goal-conditioned visual foresight and model action as foresight-guided inverse dynamics, while WALL-OSS employs a tightly coupled MoE design to align discrete action priors and continuous control, enhancing long-horizon manipulation. In autonomous driving, recent works [^97] [^101] [^88] have explored unified architectures. However, as shown in Tab. 1, they lack critical capabilities including CoT reasoning, natural language interaction, and diffusion-based planning, while failing to leverage large-scale unlabeled data, and pre-trained VLM and video generation model knowledge—limiting both their interpretability and generalization potential.

## 3 Method

The proposed UniUGP is a unified model for understanding, generation, and planning to further enhance the causal reasoning ability across different modalities: 1) Large-scale challenging data pairs are collected and processed to train and evaluate the understanding, reasoning, generation and planning capabilities of the unified model, as elaborated in Sec. 3.1. 2) An elegant unified framework combines the advantages of VLA and world models, as elaborated in Sec. 3.2. 3) A well-designed training strategy enables the unified model to fully acquire knowledge from different modalities on various datasets, as elaborated in Sec. 2.

![[x1 2.png|Refer to caption]]

Figure 1: Illustration of UniUGP, a unified model with three hybrid experts. The understanding expert performs the next-token prediction for causal reasoning. The planning expert forms a MoT architecture with the understanding expert, and performs the velocity prediction in flow matching for production future actions. The generation expert is cascaded as a world model to produce future videos.

![[x2 1.png|Refer to caption]]

Figure 2: Dataset Construction Pipeline. This figure depicts the pipeline of data collection (integrating multiple challenging driving datasets) and data processing (featuring four task categories: understanding, chain-of-thought, planning, and instruction following) to train and assess the cognitive abilities of end-to-end autonomous driving models within a unified QA framework.

### 3.1 Challenging Long-tail Driving Dataset

Prior benchmarks emphasize structured scenes and simulator-based evaluations [^63] [^67] [^10], which overlook the challenges of long-tail driving events. To address this, we have collected a large number of challenging long-tail driving videos, including Waymo-E2E [^79], DADA2000 [^13], Lost and Found (LaF) [^55], StreetHazards (StHa) [^18], SOM [^59], and AADV [^7].

These datasets were further processed to be used separately for training and evaluating the perception ability, causal reasoning ability, planning ability, and the ability to follow instructions. For the comprehension task, we have designed three sub-tasks: small objects, accident subject relationships, and accident prediction. The questions and answers of these tasks are labeled based on the provided labels from the dataset and the advanced VLM model. For more details, please refer to the supplementary materials. CoT reasoning: We employed the results of future planning and reasonable prompts to force the advanced VLM to generate the accurate CoT. We carried out manual calibration for CoT. Finally, it is worth mentioning that we will assign corresponding instructions to each predicted trajectory in the future, which enables our model to have the ability to follow instructions. The more detailed data processing procedures are listed in the Appendix.

### 3.2 Unified Model With Hybrid Expert

We adopted the Hybrid Expert structure to achieve unified understanding, generation and planning. This architecture can leverage the advanced features of the diffusion policy and the knowledge of pre-trained VLM and world models. Unified Model

Understanding and Planning Experts. Based on the validity proof of the existing work [^14] [^3] [^47] [^57], the understanding and planning experts constitute a Mixture-of-Transformers (MoT) architecture, as shown in Figure 2. For the understanding expert, we choose the Qwen2.5-VL [^1] as our backbone model. The text instructions and the observation images are firstly mapped to aligned cross-modal understanding tokens $\bm{x}^{{und}}$, by the text tokenizer and the ViT encoder, respectively.

For the planning expert, the action chunk $\bm{a}$ is modeled by a flow matching process [^41], where the expert learns to reverse a gradual noise-addition process added in the forward process. During training, a random noise $\epsilon\sim\mathcal{N}(\bf{0},\bf{I})$ and a timestep $\tau\in[0,1]$ are sampled to model the noised actions:

$$
\bm{a}_{\tau}=\tau\bm{a}+(1-\tau)\epsilon
$$

The $\bm{a}_{\tau}$ along with the history states $\mathbf{s}$ are prjoected into planning expert tokens $\bm{x}^{plan}=\mathrm{Proj.}([\bm{s},\bm{a}_{\tau}])$.

The $\bm{x}^{{und}}$ and $\bm{x}^{plan}$ are passed through several MoT layers, where each layer can be formalized as:

$$
\bm{h}_{o}^{und},\bm{h}_{o}^{und}=\mathrm{MSHA}([\mathrm{QKV}_{und}(\bm{x}^{{und}}),\mathrm{QKV}_{plan}(\bm{x}^{{plan}})])
$$

The $\mathrm{QKV}$ denote the linear projections that map understanding and planning tokens (or hidden states) into queries, keys, and values. The MSHA denotes the multi-head self-attention. Then, the modality-specific feed-forward networks (FFNs) are utilized to process the $\bm{h}_{o}^{und}$ and $\bm{h}_{o}^{und}$, separately.

$$
\displaystyle\bm{h}_{ffn}^{und}=\mathrm{FFN}_{und}(\bm{h}_{o}^{und}),
$$
$$
\displaystyle\bm{h}_{ffn}^{plan}=\mathrm{FFN}_{plan}(\bm{h}_{o}^{plan})
$$

The final resulting hidden states $\bm{h}^{und}$ and $\bm{h}^{plan}$ are mapped into the logits in text and the predicted denoising vector filed.

$$
\displaystyle P_{logits}=
$$
 
$$
\displaystyle\mathrm{LMHead}(\bm{h}^{und}),
$$
$$
\displaystyle\bm{u}_{\tau}^{plan}=
$$
 
$$
\displaystyle\mathrm{unProj.}(\bm{h}^{plan})
$$

The training objectives for the understanding and planning experts are formalized as:

$$
\displaystyle\mathcal{L}_{und}=
$$
 
$$
\displaystyle\mathbb{E}_{{x}_{i}^{{und}}}[-\mathrm{log}(P({x}_{i}^{{und}}|\bm{x}_{<i}^{{und}}))],
$$
$$
\displaystyle\mathcal{L}_{plan}=
$$
 
$$
\displaystyle\mathbb{E}_{\bm{u}_{\tau}^{plan}}[||\bm{u}_{\tau}^{plan}-(\epsilon-\bm{a})||_{2}]
$$

Generation Expert. The generation expert interacts with the understanding and planning experts in a serial manner. On mobile devices, the generation expert can be disabled to save computational effort, without compromising the performances of the former experts.

The generation expert produces future videos via a flow matching process, same as the planning expert. It is composed of several DiT [^54] blocks. In practice, we adopt Wan2.1 [^65] as the base model and inherit its pre-trained parameters. Note that any other DiT-based video generation models are feasible here.

Both future and history images are encoded into tokens by VAE. As shown in Figure 2, the history image tokens $\bm{v}^{hist}$ and noised future image tokens $\bm{v}_{\tau}^{fut}$ are concatenated as the input. The understanding hidden states $\bm{h}^{und}$ and the action embeddings $\bm{A}$ are concatenated as the condition. The final denoising vector feild is computed as:

$$
\displaystyle\bm{u}_{\tau}^{gen}=\mathcal{W}([\bm{v}^{hist},\bm{v}_{\tau}^{fut}],[\bm{h}^{und},\bm{A}],\tau)
$$

During inference, the action embeddings are obtained by projecing the predicted future actions $\hat{\bm{a}}$ from the planning expert, as $\bm{A}=\mathrm{Proj.}(\hat{\bm{a}})$. During training, The embeddings are computed either from ground truth actions, or from actions obtained via single-step denoising:

$$
\bm{A}\sim\begin{cases}\mathrm{Proj.}(\bm{a})&\text{if rand}(0\sim 1)>0.5,\\
\mathrm{Proj.}(\bm{a}_{\tau}-(1-\tau)\bm{u}_{\tau}^{plan})&\text{else}.\end{cases}
$$

As such, the generation expert receives both semantic and physical signals from the understanding and planning experts, facilitating more realistic video generation. The training objective can be formalized as:

$$
\displaystyle\mathcal{L}_{gen}=
$$
 
$$
\displaystyle\mathbb{E}_{\bm{u}_{\tau}^{gen}}[||\bm{u}_{\tau}^{gen}-(\epsilon-\bm{v^{fut}})||_{2}]
$$

### 3.3 Training Recipe

Table 2: Hyperparameter Configuration for the Four-Stage Training Framework

<table><tbody><tr><td>Hyperparameter</td><td>Stage 1</td><td>Stage 2</td><td>Stage 3</td><td>Stage 4</td></tr><tr><td rowspan="3">Trained Components</td><td>Und. Expert: ✓</td><td>Und. Expert: <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>Und. Expert: ✓</td><td>Und. Expert: ✓</td></tr><tr><td>Gen. Expert: <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>Gen. Expert: ✓</td><td>Gen. Expert: <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>Gen. Expert: ✓</td></tr><tr><td>Plan. Expert: <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>Plan. Expert: ✓</td><td>Plan. Expert: <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>Plan. Expert: ✓</td></tr><tr><td>Dataset</td><td><sup><a href="#fn:10">10</a></sup></td><td><sup><a href="#fn:4">4</a></sup></td><td>Custom CoT dataset</td><td>Mixture of Stages 1–3 (ratio 0.1: 0.4: 0.5)</td></tr><tr><td>Learning Rate</td><td><math><semantics><msup><mn>10</mn> <mrow><mo>−</mo> <mn>4</mn></mrow></msup> <annotation>10^{-4}</annotation></semantics></math></td><td><math><semantics><msup><mn>10</mn> <mrow><mo>−</mo> <mn>4</mn></mrow></msup> <annotation>10^{-4}</annotation></semantics></math></td><td><math><semantics><msup><mn>10</mn> <mrow><mo>−</mo> <mn>4</mn></mrow></msup> <annotation>10^{-4}</annotation></semantics></math></td><td><math><semantics><msup><mn>10</mn> <mrow><mo>−</mo> <mn>4</mn></mrow></msup> <annotation>10^{-4}</annotation></semantics></math></td></tr><tr><td>Understanding Resolution</td><td>(224, 224)</td><td>–</td><td>(224, 224)</td><td>(224, 224)</td></tr><tr><td>Generation Resolution</td><td>–</td><td>(512, 512)</td><td>–</td><td>(512, 512)</td></tr><tr><td>GPU Resources</td><td>8 nodes <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 8 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> gpus (80GB)</td><td>8 nodes <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 8 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> gpus (80GB)</td><td>8 nodes <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 8 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> gpus (80GB)</td><td>8 nodes <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 8 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> gpus (80GB)</td></tr><tr><td>Batch Size</td><td>64</td><td>64</td><td>64</td><td>64</td></tr><tr><td>Training Steps</td><td>1M</td><td>4M</td><td>1M</td><td>4M</td></tr></tbody></table>

We design a four-stage training framework, which sequentially builds foundational scenario understanding, visual dynamic modeling, text-based reasoning, and multi-capability fusion, detailed are presented below. The training parameters and details can be found in Tab. 2.

Stage 1: Continuous Training for the Understanding of the basic scenarios is to enable the Understanding Expert to establish a comprehensive understanding of diverse driving scenarios—covering common traffic and long-tailed cases. Only the Understanding Expert is trained in this stage. The training dataset at this stage includes the long-tail data set that we have labeled and the ImpromptuVLA (80,000 meticulously curated video clips from 8 open-source large-scale datasets) [^10].

Stage 2: Visual Dynamics Modeling and Planning Training focuses on learning visual dynamics and motion planning capabilities. During this training phase, driving videos with trajectories were used for training the Generation Expert and the Planning Expert. We utilized multiple public datasets including: nuScenes [^4], NuPlan [^5], Waymo [^60], Lyft [^35], and Cosmos [^56].

Stage 3: Text Reasoning Learning for Causal Validation integrates CoT reasoning into the Understanding Expert, enabling the model to validate the logic of its perceptions and planning using natural language. This stage enhances model interpretability and ensures decisions are grounded in explicit causal reasoning. This part is trained using our own annotated CoT dataset.

Stage 4: Mixed Training for Multi-Capability Fusion resolves potential misalignments between individual stages and enhances generalization across scenarios. In Stage 4, the three experts are jointly trained to achieve consistent end-to-end performance. We mix datasets from Stages 1–3 at a fixed proportion to balance foundational, reasoning, planning, and generation capabilities. The overall objective $\mathcal{L}_{\text{total}}$ is a weighted sum of three sub-objectives:

$$
\mathcal{L}_{{total}}=\alpha\cdot\mathcal{L}_{{und}}+\beta\cdot\mathcal{L}_{{plan}}+\gamma\cdot\mathcal{L}_{{gen}}
$$

where $\alpha=0.3$, $\beta=0.5$, $\gamma=0.2$. This alignment ensures the model operates as a unified system rather than a collection of isolated components.

## 4 Experiment

### 4.1 Dataset

We utilize two complementary categories of datasets to evaluate multimodal world understanding and decision-making under both common and safety-critical driving conditions. For Perception and Understanding, we adopt anomaly and accident anticipation datasets including DADA2000 [^13], Lost and Found (LaF) [^55], StreetHazards (StHa) [^18], SOM [^59], and AADV [^7]. These datasets contain rare obstacles, out-of-distribution objects, and pre-accident sequences collected in diverse real-world environments, enabling assessment of visual scene comprehension, hazard awareness, and recognition of long-tail semantic patterns.

For reasoning, planning, and instruction following, we use the Waymo Open Dataset Long-tail End-to-End Driving [^79], which consists of 4,021 real-world driving segments specifically curated to capture rare and high-risk events that occur in less than 0.003% of daily driving. Each segment provides surround-view camera streams, ego-motion history, and route signals, and the benchmark task is to predict future 5-second trajectories under uncertain and interaction-heavy conditions. This dataset enables rigorous evaluation of robustness, generalization, and physically grounded decision-making in long-tail scenarios, complementing the perception-focused anomaly datasets above.

Following the previous methods [^11] [^67], we evaluate scene understanding on DriveLM [^58]. This dataset features keyframe descriptions paired with QA annotations covering full-stack autonomous driving (perception, prediction, planning), offering comprehensive language support for development.

Consistent with methods [^30] [^16], we evaluate trajectory planning and future frames generation on the nuScenes [^4]. The nuScenes contains 1,000 scenes of approximately 20 seconds each captured by a 32-beam LiDAR and six cameras providing 360-degree field of view. Specifically, the dataset provides 28,130 (train), 6,019 (val), and 193,082 (unannotated) samples. In addition, we utilize a large number of publicly available datasets for training, including: ImpromptuVLA [^10], NuPlan [^5], Waymo [^60], Lyft [^35], and Cosmos [^56].

### 4.2 Metrics

We have established a new long-tail benchmark consisting of understanding, CoT reasoning, Planning and Instruction Following. For comprehension questions (mainly multiple-choice and true/false questions), we evaluate them based on the accuracy rate. For CoT, we use the API of GPT-4o to rate it in terms of consistency, rationality and fluency, and also provide a score from Blue. For Planning, we use L2 3s to rate it. For instruction following, we score the trajectories corresponding to different instructions using L2 (3s). More details are provided in the appendix. We evaluate trajectory planning using L2 displacement error and collision rate following previous methods [^91]. Following existing methods [^69] [^83], we report Fréchet Inception Distance (FID) [^19] to measure the future frames generation quality. DriveLM GVQA [^58] metrics include language metrics like BLEU, ROUGE\_L, and CIDEr for text generation, the ChatGPT Score for open-ended Q&A and accuracy for multiple-choice questions.

### 4.3 Evaluation of Understanding Ability

To validate the effectiveness of our benchmark and the performance of our model, we conduct comparative experiments with state-of-the-art vision-language models, and the results are presented in Tab. 3. The evaluation metrics are categorized into four key dimensions to comprehensively measure model performance: Understanding (assessing scene and object comprehension, including Small (small object recognition), accident subject relationship, and Acci.Pred. (accident event prediction)), CoT (Chain-of-Thought reasoning ability, evaluated by GPT (subjective GPT score) and Blue (BLEU score)), Planning (short-term driving planning, measured by L2 distance (3s) with smaller values indicating better performance), and Following (trajectory following accuracy, also measured by L2 distance (3s)). The comparative models include GPT 4o, Qwen 2.5 VL, and two ablated versions of our model (Our w/o CoT: without Chain-of-Thought module; Our w/o Gen.: without generation module), alongside our full model (Our).

As shown in Table 3, our full model outperforms state-of-the-art methods (GPT-4o, Qwen-2.5-VL-72B) and our ablated versions (Our w/o CoT, Our w/o Gen.) across all key metrics: in Understanding (89.3%, 88.6%, 95.8% for Small, Relationship, Abnor. Pred.), CoT (GPT score 0.88, Blue score 0.240), Planning (L2=1.45), and Following (L2=1.40), all surpassing baselines and ablated models with lower performance; these results confirm that integrating the generation and CoT modules significantly enhances the model’s comprehensive driving capabilities. The generation model has significantly enhanced the performance of the model. We conducted a further qualitative analysis as shown in Fig. 3. The world model forces VLA to learn visual causal inference, particularly focusing on distant objects to generate better future frames. This enables the VLA model to predict potential dangers in advance, thereby ensuring driving safety.

Table 3: Performance Comparison on Driving Evaluation Benchmark. Note: GPT-4o and Qwen-2.5-VL-72B are provided with historical trajectory information and trajectory explanations, enabling trajectory prediction evaluation.

<table><tbody><tr><td rowspan="2">Model</td><td colspan="3">Understanding</td><td colspan="2">CoT</td><td>Planning</td><td>Following</td></tr><tr><td>Small</td><td>Relationship</td><td>Abnor. Pred.</td><td>GPT</td><td>Blue</td><td>L2 (3s)</td><td>L2 (3s)</td></tr><tr><td>GPT-4o</td><td>64.2%</td><td>63.5%</td><td>72.8%</td><td>0.55</td><td>0.125</td><td>2.63</td><td>2.58</td></tr><tr><td>Qwen-2.5-VL-72B</td><td>75.8%</td><td>74.9%</td><td>81.5%</td><td>0.72</td><td>0.188</td><td>1.94</td><td>1.89</td></tr><tr><td>Our w/o CoT</td><td>86.5%</td><td>85.7%</td><td>93.2%</td><td>0.83</td><td>0.218</td><td>1.58</td><td>1.53</td></tr><tr><td>Our w/o Gen.</td><td>83.7%</td><td>82.9%</td><td>90.6%</td><td>0.80</td><td>0.203</td><td>1.72</td><td>1.67</td></tr><tr><td>Our</td><td>89.3%</td><td>88.6%</td><td>95.8%</td><td>0.88</td><td>0.240</td><td>1.45</td><td>1.40</td></tr></tbody></table>

![[x3 1.png|Refer to caption]]

Figure 3: The ablation experiment on the absence or presence of world model knowledge. The world model enables the VLA to pay more attention to future causal relationships, thereby focusing on the semantics of distant objects.

Table 4: [^4]

<table><tbody><tr><td rowspan="2">Method</td><td rowspan="2">Input</td><td rowspan="2">Auxiliary Supervision</td><td colspan="4">L2 (m) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td colspan="4">Collision Rate (%) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td></tr><tr><td>ST-P3 <sup><a href="#fn:22">22</a></sup></td><td>Camera</td><td>Map & Box & Depth</td><td>1.33</td><td>2.11</td><td>2.90</td><td>2.11</td><td>0.23</td><td>0.62</td><td>1.27</td><td>0.71</td></tr><tr><td>UniAD <sup><a href="#fn:25">25</a></sup></td><td>Camera</td><td>Map & Box & Motion & Tracklets & Occ</td><td>0.48</td><td>0.96</td><td>1.65</td><td>1.03</td><td>0.05</td><td>0.17</td><td>0.71</td><td>0.31</td></tr><tr><td>OccWorld <sup><a href="#fn:95">95</a></sup></td><td>Camera</td><td>3D-Occ</td><td>0.52</td><td>1.27</td><td>2.41</td><td>1.40</td><td>0.12</td><td>0.40</td><td>2.08</td><td>0.87</td></tr><tr><td>VAD-Tiny <sup><a href="#fn:30">30</a></sup></td><td>Camera</td><td>Map & Box & Motion</td><td>0.60</td><td>1.23</td><td>2.06</td><td>1.30</td><td>0.31</td><td>0.53</td><td>1.33</td><td>0.72</td></tr><tr><td>VAD-Base <sup><a href="#fn:30">30</a></sup></td><td>Camera</td><td>Map & Box & Motion</td><td>0.54</td><td>1.15</td><td>1.98</td><td>1.22</td><td>0.04</td><td>0.39</td><td>1.17</td><td>0.53</td></tr><tr><td>GenAD <sup><a href="#fn:96">96</a></sup></td><td>Camera</td><td>Map & Box & Motion</td><td>0.36</td><td>0.83</td><td>1.55</td><td>0.91</td><td>0.06</td><td>0.23</td><td>1.00</td><td>0.43</td></tr><tr><td>Doe-1 <sup><a href="#fn:98">98</a></sup></td><td>Camera <sup>∗</sup></td><td>QA</td><td>0.50</td><td>1.18</td><td>2.11</td><td>1.26</td><td>0.04</td><td>0.37</td><td>1.19</td><td>0.53</td></tr><tr><td>Epona <sup><a href="#fn:91">91</a></sup></td><td>Camera <sup>∗</sup></td><td>None</td><td>0.61</td><td>1.17</td><td>1.98</td><td>1.25</td><td>0.01</td><td>0.22</td><td>0.85</td><td>0.36</td></tr><tr><td>UniUGP (Ours)</td><td>Camera <sup>∗</sup></td><td>QA</td><td>0.58</td><td>1.14</td><td>1.95</td><td>1.23</td><td>0.01</td><td>0.19</td><td>0.81</td><td>0.33</td></tr></tbody></table>

### 4.4 Evaluation of Planning Ability

As shown in Table 4, our model (Ours) achieves competitive performance under the setting of front camera input (Camera <sup>∗</sup>) and QA auxiliary supervision: it attains an average L2 distance of 1.23m and an average collision rate of 0.33%, outperforming multiple comparative methods with similar input constraints. Specifically, compared to Doe-1 [^98] with the same input and auxiliary supervision (Camera <sup>∗</sup> +QA), our model reduces the average L2 distance from 1.26m to 1.23m and the average collision rate from 0.53% to 0.33%. It also performs favorably against advanced methods like GenAD [^96] (average L2: 0.91m, collision rate: 0.43%) and UniAD [^25] (average L2: 1.03m, collision rate: 0.31%) considering our more constrained input (only front camera vs. full camera suite). Additionally, our model surpasses Epona [^91] (Camera <sup>∗</sup> +None, average L2: 1.25m, collision rate: 0.36%) under similar input conditions, with lower average L2 distance and collision rate. These results demonstrate the effectiveness of our unified model’s capabilities in trajectory planning accuracy and driving safety even with a unified model.

### 4.5 Evaluation of Generation Ability

As shown in Table 5, our method is evaluated under the same protocol as Epona [^91] and FSDrive [^88], achieving significant improvements in generation quality. This gain stems from the effective utilization of a pre-trained generative model, which enhances the model’s ability to capture realistic scene dynamics and appearance. We have provided a trajectory-controllable visualization as shown in Fig. 4, which demonstrates the controllability of our generated experts.

Table 5: Future Frames Generation Quality Comparison on the NuScenes Dataset.

<table><tbody><tr><td rowspan="2">Method</td><td>DriveDreamer <sup><a href="#fn:69">69</a></sup></td><td>Drive-WM <sup><a href="#fn:73">73</a></sup></td><td>GenAD <sup><a href="#fn:83">83</a></sup></td><td>GEM <sup><a href="#fn:17">17</a></sup></td><td>Doe-1 <sup><a href="#fn:98">98</a></sup></td><td>Epona <sup><a href="#fn:91">91</a></sup></td><td>FSDrive <sup><a href="#fn:88">88</a></sup></td><td>UniUGP</td></tr><tr><td>[ECCV24]</td><td>[CVPR24]</td><td>[CVPR24]</td><td>[CVPR25]</td><td>[arXiv24]</td><td>[ICCV25]</td><td>[NeurIPS25]</td><td>Ours</td></tr><tr><td>Type</td><td>Diff</td><td>Diff</td><td>Diff</td><td>Diff</td><td>AR</td><td>Diff</td><td>AR</td><td>AR+Diff</td></tr><tr><td>Res.</td><td>128×192</td><td>192×384</td><td>256×448</td><td>576×1024</td><td>384×672</td><td>576×1024</td><td>128×192</td><td>512×512</td></tr><tr><td>FID <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>52.6</td><td>15.8</td><td>15.4</td><td>10.5</td><td>15.9</td><td>7.5</td><td>10.1</td><td>7.4</td></tr><tr><td>FVD <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>452.0</td><td>122.7</td><td>184.0</td><td>-</td><td>-</td><td>82.8</td><td>-</td><td>75.9</td></tr></tbody></table>

![[x4 1.png|Refer to caption]]

Figure 4: Trajectory controllable generation visualization. We control the generation of future frames of the video by modifying the trajectories fed into the generation model, which demonstrates the controllability of our generation experts.

### 4.6 Results on DriveLM Dataset

As shown in Table 6, under the same evaluation protocol as FSDrive [^67], our model (UniUGP) achieves superior performance on the DriveLM GVQA Benchmark compared to existing state-of-the-art methods. Our final score reaches 0.59, which is higher than FSDrive (0.57), OmniDrive [^67] (0.56), SimpleLLM4AD [^94] (0.53), TrackingMeetsLMM [^28] (0.52), Cube-LLM [^11] (0.50), and the DriveLM baseline [^58] (0.32). Across key metrics, our accuracy is 0.74, outperforming all comparative methods; BLEU is 0.78 and ROUGE is 0.76, both leading FSDrive (0.76, 0.74) and other SOTA models; the match metric is 0.41, which is also higher than existing methods. While our ChatGPT score (0.64) and CIDEr score (0.19) are competitive among the methods, the overall leading performance across core evaluation dimensions demonstrates the advantage of our method over existing state-of-the-art approaches in scene understanding and language interaction capabilities.

Table 6: Results on DriveLM GVQA Benchmark.

| Method | Acc. $\uparrow$ | GPT $\uparrow$ | BLEU\_1 $\uparrow$ | ROUGE\_L $\uparrow$ | CIDEr $\uparrow$ | Match $\uparrow$ | Final Score $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DriveLM baseline \[ECCV24\] [^58] | 0.00 | 0.65 | 0.05 | 0.08 | 0.10 | 0.28 | 0.32 |
| Cube-LLM \[ICLR25\] [^11] | 0.39 | 0.89 | 0.16 | 0.20 | 0.31 | 0.36 | 0.50 |
| TrackingMeetsLMM \[arxiv25\] [^28] | 0.60 | 0.58 | 0.72 | 0.72 | 0.04 | 0.36 | 0.52 |
| SimpleLLM4AD \[arxiv24\] [^94] | 0.66 | 0.57 | 0.76 | 0.73 | 0.15 | 0.35 | 0.53 |
| OminiDrive \[CVPR25\] [^67] | 0.70 | 0.65 | 0.52 | 0.73 | 0.13 | 0.37 | 0.56 |
| FSDrive \[NeurIPS25\] [^67] | 0.72 | 0.63 | 0.76 | 0.74 | 0.17 | 0.39 | 0.57 |
| UniUGP (Ours) | 0.74 | 0.64 | 0.78 | 0.76 | 0.19 | 0.41 | 0.59 |

## 5 Conclusion

We propose UniUGP, a unified Understanding-Generation-Planning framework addressing critical challenges in temporal dynamics and long-tail generalization. Leveraging a hybrid expert architecture with pre-trained VLMs and video generation models, UniUGP produces interpretable reasoning, physically consistent trajectories, and coherent future videos from multimodal inputs. Our four-stage training strategy progressively aligns these capabilities across diverse datasets. Extensive experiments demonstrate state-of-the-art performance and robust generalization, establishing a strong foundation for future autonomous driving research.

## Contributions

Authors: Hao Lu <sup>1,2,∗,†</sup>, Ziyang Liu <sup>2,∗</sup>, Guangfeng Jiang <sup>3</sup>, Yuanfei Luo <sup>2,†</sup>, Sheng Chen <sup>2</sup>, Yangang Zhang <sup>2</sup>, Ying-Cong Chen <sup>1,§</sup>

Affiliations: <sup>1</sup> HKUST-GZ, <sup>2</sup> ByteDance Seed, <sup>3</sup> Independent Researcher

<sup>∗</sup> Co-first authors  
<sup>†</sup> Project leads  
<sup>§</sup> Corresponding author.  
Note: Hao LU work done at ByteDance Seed during internship.

## References

## 6 Dataset

To comprehensively evaluate and enhance the multimodal world modeling capability required for end-to-end autonomous driving, we reconstruct heterogeneous open-source driving datasets into a unified framework aligned with four essential cognitive competencies. Existing benchmarks mainly emphasize structured urban scenes or closed-loop simulation metrics, which fail to capture a model’s robustness in open-world, long-tail, and low-probability-but-high-risk scenarios. In contrast, our objective is to measure how well a model can understand, reason, and act under diverse real-world conditions.

![[x5 1.png|Refer to caption]]

Figure 5: Long-tail perception and understanding of questions and answers.

To this end, we integrate datasets covering road abnormalities and traffic accident anticipation (e.g., Lost and Found, StreetHazards, DADA-2000, and Anticipating Accidents in Dashcam Videos) together with the Waymo dataset, which contains densely interactive and ambiguous safety-critical events. We reorganize these data according to four key task dimensions: (1) Perception and Understanding, assessing visual semantics and contextual risk awareness; (2) Causal CoT Reasoning, explaining the underlying causes behind motion intentions; (3) Planning and Decision-Making, supervising physically feasible future trajectories; and (4) Instruction Following, evaluating whether control actions align with high-level navigation commands. This integrated categorization supports unified training and interpretable evaluation of cognitive, generative, and control capabilities.

### 6.1 Perception and Understanding:

We convert scene annotations from anomaly and hazard datasets into true/false and multiple-choice QA items that evaluate not only basic semantic comprehension, but also driving commonsense, corner-case interpretation, and the recognition of previously unseen or rare object categories. This formulation ensures that the model can identify what is present in the scene, understand why it is safety-relevant, and generalize to atypical or long-tail conditions critical for real-world driving. Specific examples are shown in Fig 5.

Small long-tailed object. We collected multiple datasets that contained small elongated objects. We determine whether there are small tail objects based on the segmentation map labels provided by the dataset, and thereby construct a judgment question. We designed many random questions to enhance the generalization ability of the model’s question answering. The questioning format is as shown in List LABEL:lst:question\_templates:

Listing 1: Small long-tailed object

[⬇](data:text/plain;base64,IyMgUHJvbXB0ID0gUXVlc3Rpb24gKyAiUGxlYXNlIHJlcGx5IHdpdGggVHJ1ZSBvciBGYWxzZS4iCiMgUXVlc3Rpb246CiJBbnkgc21hbGwgbG9uZy10YWlsZWQgb2JqZWN0cyBpbiB0aGUgZHJpdmluZyB2aWRlbz8iLAoiQXJlIHRoZXJlIHRpbnkgbG9uZy10YWlsZWQgaXRlbXMgaW4gdGhlIGRyaXZpbmcgY2xpcD8iLAoiRG9lcyB0aGUgZHJpdmluZyB2aWRlbyBoYXZlIHNtYWxsIGxvbmctdGFpbGVkIHRoaW5ncz8iLAoiU21hbGwgbG9uZy10YWlsZWQgb2JqZWN0cyBwcmVzZW50IGluIHRoZSBkcml2aW5nIGZvb3RhZ2U/IiwKIkFyZSB0aGVyZSBzbWFsbCBsb25nLXRhaWxlZCBvYmplY3RzIGluIHRoZSBkcml2aW5nIHZpZGVvPyIsCiJBbnkgc21hbGwgbG9uZy10YWlsZWQgb2JqZWN0cz8iLAoiQXJlIHRoZXJlIHNtYWxsIGl0ZW1zIHdpdGggbG9uZyB0YWlscz8iLAoiRXhpc3Qgc21hbGwgb2JqZWN0cyB3aXRoIGxvbmcgdGFpbHM/IiwKIkFueSB0aW55IGxvbmctdGFpbGVkIHRoaW5ncyBwcmVzZW50PyIs)

\## Prompt = Question + "Please reply with True or False."

\# Question:

"Any small long-tailed objects in the driving video?",

"Are there tiny long-tailed items in the driving clip?",

"Does the driving video have small long-tailed things?",

"Small long-tailed objects present in the driving footage?",

"Are there small long-tailed objects in the driving video?",

"Any small long-tailed objects?",

"Are there small items with long tails?",

"Exist small objects with long tails?",

"Any tiny long-tailed things present?",

Long-tailed accident prediction. We collected videos of abnormal traffic accidents. The dataset classifies whether the video is abnormal based on whether it is abnormal as indicated by the dataset and the specific timestamp. We have designed various questions to enhance generalization. The questioning method is as shown in List LABEL:lst:lta:

Listing 2: Long-tailed accident prediction

[⬇](data:text/plain;base64,IyMgUHJvbXB0ID0gUXVlc3Rpb24gKyAiUGxlYXNlIHJlcGx5IHdpdGggVHJ1ZSBvciBGYWxzZS4iCiMgUXVlc3Rpb246CiJIYXMgYW4gYWJub3JtYWwgYWNjaWRlbnQgb2NjdXJyZWQgaGVyZT8iLAoiQ291bGQgdGhlcmUgYmUgYW55IHRyYWZmaWMgaGF6YXJkcyBoZXJlPyIsCiJDb3VsZCBhIHRyYWZmaWMgYWNjaWRlbnQgb2NjdXIgaGVyZT8iLAoiQXJlIHRoZXJlIGFueSB0cmFmZmljIHJpc2tzIG9yIGhpZGRlbiBoYXphcmRzIGhlcmUgdGhhdCBjb3VsZCBsZWFkIHRvIGFjY2lkZW50cz8iLAoiV2FzIHRoZXJlIGFuIHVudXN1YWwgaW5jaWRlbnQgb3IgYWNjaWRlbnQgaGVyZSBqdXN0IG5vdz8iLAoiTWlnaHQgYSB0cmFmZmljIGFjY2lkZW50IHRha2UgcGxhY2UgaW4gdGhpcyBkcml2aW5nIHNpdHVhdGlvbj8iLAoiSXMgdGhlcmUgYSBwb3NzaWJpbGl0eSB0aGF0IGEgdHJhZmZpYyBhY2NpZGVudCB3aWxsIGhhcHBlbiBoZXJlPyI=)

\## Prompt = Question + "Please reply with True or False."

\# Question:

"Has an abnormal accident occurred here?",

"Could there be any traffic hazards here?",

"Could a traffic accident occur here?",

"Are there any traffic risks or hidden hazards here that could lead to accidents?",

"Was there an unusual incident or accident here just now?",

"Might a traffic accident take place in this driving situation?",

"Is there a possibility that a traffic accident will happen here?"

Long-tailed accident relationship. We collected video footage of abnormal traffic accidents, along with annotations indicating the abnormal entities involved. Based on these annotations, we designed multiple-choice questions. The questioning method is as shown in the list LABEL:lst:mcc:

Listing 3: Long-tailed accident relationship

[⬇](data:text/plain;base64,SU5QVVQ6IGNvcnJlY3RfYW5zd2VyIChzdHIpLCBjYW5kaWRhdGVfcG9vbCAoTGlzdFtzdHJdKQpPVVRQVVQ6IHF1ZXN0aW9uIChzdHIpLCBvcHRpb25zX2RpY3QgKERpY3Rbc3RyLCBzdHJdKSwgY29ycmVjdF9sYWJlbCAoc3RyKSwgY29ycmVjdF9hbnN3ZXIgKHN0cikKCiMgU3RlcCAxOiBGaWx0ZXIgZGlzdHJhY3RvciBjYW5kaWRhdGVzIChleGNsdWRlIGNvcnJlY3QgYW5zd2VyKQpkaXN0cmFjdG9yX2NhbmRpZGF0ZXMgPSBbb3B0IGZvciBvcHQgaW4gY2FuZGlkYXRlX3Bvb2wgaWYgb3B0IOKJoCBjb3JyZWN0X2Fuc3dlcl0KCiMgU3RlcCAyOiBSYW5kb21seSBzZWxlY3QgcXVlc3Rpb24gdHlwZSAoMi1vcHRpb24gLyA0LW9wdGlvbikKb3B0aW9uX2NvdW50ID0gUkFORE9NX0NIT0lDRShbMiwgNF0pCmRpc3RyYWN0b3JfbmVlZGVkID0gb3B0aW9uX2NvdW50IC0gMQoKIyBTdGVwIDM6IFNlbGVjdCBkaXN0cmFjdG9ycyAoc3VwcGxlbWVudCBpZiBpbnN1ZmZpY2llbnQpCmlmIGxlbihkaXN0cmFjdG9yX2NhbmRpZGF0ZXMpIOKJpSBkaXN0cmFjdG9yX25lZWRlZDoKICAgIHNlbGVjdGVkX2Rpc3RyYWN0b3JzID0gUkFORE9NX1NBTVBMRShkaXN0cmFjdG9yX2NhbmRpZGF0ZXMsIGRpc3RyYWN0b3JfbmVlZGVkKQplbHNlOgogICAgc2VsZWN0ZWRfZGlzdHJhY3RvcnMgPSBkaXN0cmFjdG9yX2NhbmRpZGF0ZXMgKyBSQU5ET01fQ0hPSUNFUyhkaXN0cmFjdG9yX2NhbmRpZGF0ZXMsIGs9ZGlzdHJhY3Rvcl9uZWVkZWQgLSBsZW4oZGlzdHJhY3Rvcl9jYW5kaWRhdGVzKSkKCiMgU3RlcCA0OiBDb21iaW5lIGFuZCBzaHVmZmxlIG9wdGlvbnMKYWxsX29wdGlvbnMgPSBbY29ycmVjdF9hbnN3ZXJdICsgc2VsZWN0ZWRfZGlzdHJhY3RvcnMKU0hVRkZMRShhbGxfb3B0aW9ucykKCiMgU3RlcCA1OiBNYXAgdG8gb3B0aW9uIGxhYmVscyAoQS9CL0MvRCkKb3B0aW9uX2xldHRlcnMgPSBbJ0EnLCAnQicsICdDJywgJ0QnXVs6b3B0aW9uX2NvdW50XQpvcHRpb25zX2RpY3QgPSBkaWN0KHppcChvcHRpb25fbGV0dGVycywgYWxsX29wdGlvbnMpKQoKIyBTdGVwIDY6IExvY2F0ZSBjb3JyZWN0IGxhYmVsCmNvcnJlY3RfbGFiZWwgPSBuZXh0KGxldHRlciBmb3IgbGV0dGVyLCBvcHQgaW4gb3B0aW9uc19kaWN0Lml0ZW1zKCkgaWYgb3B0ID09IGNvcnJlY3RfYW5zd2VyKQoKIyBTdGVwIDc6IEdlbmVyYXRlIHF1ZXN0aW9uIHdpdGggc3BlY2lmaWVkIHRlbXBsYXRlCnF1ZXN0aW9uID0gIldoaWNoIG9mIHRoZSBmb2xsb3dpbmcgZGVzY3JpYmVzIHRoZSBjdXJyZW50IHNpdHVhdGlvbj8gIiArICcsICcuam9pbihbZiJ7bGV0dGVyfS4ge29wdH0iIGZvciBsZXR0ZXIsIG9wdCBpbiBvcHRpb25zX2RpY3QuaXRlbXMoKV0pCgpSRVRVUk4gcXVlc3Rpb24sIG9wdGlvbnNfZGljdCwgY29ycmVjdF9sYWJlbCwgY29ycmVjdF9hbnN3ZXI=)

INPUT: correct\_answer (str), candidate\_pool (List\[str\])

OUTPUT: question (str), options\_dict (Dict\[str, str\]), correct\_label (str), correct\_answer (str)

\# Step 1: Filter distractor candidates (exclude correct answer)

distractor\_candidates = \[opt for opt in candidate\_pool if opt $\neq$ correct\_answer\]

\# Step 2: Randomly select question type (2-option / 4-option)

option\_count = RANDOM\_CHOICE(\[2, 4\])

distractor\_needed = option\_count - 1

\# Step 3: Select distractors (supplement if insufficient)

if len(distractor\_candidates) $\geq$ distractor\_needed:

selected\_distractors = RANDOM\_SAMPLE(distractor\_candidates, distractor\_needed)

else:

selected\_distractors = distractor\_candidates + RANDOM\_CHOICES(distractor\_candidates, k=distractor\_needed - len(distractor\_candidates))

\# Step 4: Combine and shuffle options

all\_options = \[correct\_answer\] + selected\_distractors

SHUFFLE(all\_options)

\# Step 5: Map to option labels (A/B/C/D)

option\_letters = \[’A’, ’B’, ’C’, ’D’\]\[:option\_count\]

options\_dict = dict(zip(option\_letters, all\_options))

\# Step 6: Locate correct label

correct\_label = next(letter for letter, opt in options\_dict.items() if opt == correct\_answer)

\# Step 7: Generate question with specified template

question = "Which of the following describes the current situation? " + ’, ’.join(\[f"{letter}. {opt}" for letter, opt in options\_dict.items()\])

RETURN question, options\_dict, correct\_label, correct\_answer

![[x8 1.png|Refer to caption]]

Figure 6: Future Trajectories-Based CoT Reasoning.

### 6.2 Causal CoT Reasoning:

For sequences where the future driving outcome is observable, we construct QA pairs in which the answer is a structured multi-step chain of thought that explains how the final driving decision is formed. During dataset construction, we utilize both the future image sequence and the ground-truth ego trajectory to ensure that the reasoning process is strictly aligned with the actual physical outcome. The reasoning is required to describe the scene context, identify key interactive agents, infer their potential intentions, and justify the final driving action that leads to the observed future behavior. This design encourages the model to learn reasoning that is predictive and causally grounded in realistic driving dynamics, instead of producing descriptive or speculative explanations after the fact. As a result, the model is encouraged to understand why a particular action is necessary for safety rather than merely recognizing what action is taken. Specific examples are shown in Fig 6. CoT reasoning based on future trajectories is presented in List LABEL:lst:gpt\_prompt:

Listing 4: Future Trajectories-Based CoT Reasoning

[⬇](data:text/plain;base64,IyMgVGhlIHByb21wdHMgZm9yIGdlbmVyYXRpbmcgQ29UOgoKWW91IGFyZSBhbiBhdXRvbm9tb3VzIGRyaXZpbmcgcGxhbm5pbmcgZXhwZXJ0LiBZb3VyIHRhc2sgaXMgdG8gYW5hbHl6ZSBhIGRyaXZpbmcgc2NlbmUgdXNpbmcgaW5mb3JtYXRpb24gcHJvdmlkZWQgZnJvbSB0aGUgaW1hZ2VzIG9mIGZyb250IGNhbWVyYSB2aWV3cywgcGFzdCB3YXlwb2ludHMsIHBhc3QgdmVsb2NpdHksIHBhc3QgYWNjZWxlcmF0aW9uLCBjdXJyZW50IHZlbG9jaXR5LCBjdXJyZW50IGFjY2VsZXJhdGlvbiwgZnV0dXJlIHdheXBvaW50cywgYW5kIHRoZSBpbnRlbmRlZCBtb3Rpb24gb2YgdGhlIGVnbyB2ZWhpY2xlLiBZb3UgbXVzdCByZWFzb24gdGhyb3VnaCB0aGUgc2NlbmUgYmFzZWQgb24gdmlzdWFsIGFuZCB0ZW1wb3JhbCBkYXRhIGFuZCBwcm9kdWNlIGEgc3RydWN0dXJlZCBkZWNpc2lvbiBvdXRwdXQuIEkgYW0gZ2l2aW5nIHlvdXIgZnV0dXJlIGRlY2lzaW9ucyB0byBoZWxwIHlvdSBjb25maXJtIHRoYXQgeW91ciByZWFzb25pbmcgaXMgY29ycmVjdCwgYnV0IHlvdXIgcmVhc29uaW5nIHByb2Nlc3Mgc2hvdWxkIG5vdCByZWx5IG9uIHRoZSBmdXR1cmUgdHJhamVjdG9yeSBhcyB0aGUgYmFzaXMgZm9yIHlvdXIgcmVhc29uaW5nLiBZb3VyIHJlYXNvbmluZyBjYW4gb25seSBtYWtlIHByZWRpY3Rpb25zIGFib3V0IGZ1dHVyZSBkZWNpc2lvbnMgYmFzZWQgb24gdGhlIGN1cnJlbnQgaW1hZ2UgYW5kIGhpc3RvcmljYWwgdHJhamVjdG9yeS4KCi0tLQoqKklucHV0OioqCgotICoqUGFzdCB3YXlwb2ludHMqKjogJSU8cGFzdF93YXlwb2ludHM+JSUKLSAqKlBhc3QgdmVsb2NpdHkqKjogJSU8cGFzdF92ZWxvY2l0eT4lJQotICoqUGFzdCBhY2NlbGVyYXRpb24qKjogJSU8cGFzdF9hY2NlbGVyYXRpb24+JSUKLSAqKkZ1dHVyZSB3YXlwb2ludHMqKjogJSU8ZnV0dXJlX3dheXBvaW50cz4lJQoKLS0tCgoqKkNvb3JkaW5hdGUgYW5kIFRlbXBvcmFsIFN5c3RlbSBFeHBsYW5hdGlvbjoqKgoKPiBBbGwgbW90aW9uIGFuZCBzdGF0ZSBkYXRhIGFyZSBleHByZXNzZWQgaW4gdGhlIGVnbyBsb2NhbCBjb29yZGluYXRlIHN5c3RlbToKPiAtICt4ID0gZm9yd2FyZCBkaXJlY3Rpb24KPiAtICt5ID0gbGVmdCBkaXJlY3Rpb24KPiAtIFRoZSBvcmlnaW4gKDAsIDApIGlzIGxvY2F0ZWQgYXQgdGhlIGNlbnRlciBvZiB0aGUgZWdvIHZlaGljbGUKCj4gLSBVbml0czoKPiAgIC0gV2F5cG9pbnRzOiBtZXRlcnMKPiAgIC0gVmVsb2NpdHk6IG1ldGVycy9zZWNvbmQKPiAgIC0gQWNjZWxlcmF0aW9uOiBtZXRlcnMvc2Vjb25kCgo+IC0gU2FtcGxpbmcgZnJlcXVlbmN5OiA0SHogKGV2ZXJ5IDAuMjVzKQoKPiAtIFBhc3QgYXJyYXlzIChgcGFzdF93YXlwb2ludHNgLCBgcGFzdF92ZWxvY2l0eWAsIGBwYXN0X2FjY2VsZXJhdGlvbmApIGFyZSBlYWNoIDE2IHRpbWUgc3RlcHM6Cj4gICAtIEZvcm1hdDogYFsgW3gwLCB5MF0sIFt4MSwgeTFdLCAuLi4sIFt4MTUsIHkxNV0gXWAKPiAgIC0gSW5kZXhlZCBgaSA9IDBgIHRvIGAxNWAsIHdpdGgKPiAgICAgLSBgaSA9IDBgIC0+IGB0ID0gLTMuNzVzYCwKPiAgICAgLSBgaSA9IDE1YCAtPiBgdCA9IDAuMHNgIChjdXJyZW50IGZyYW1lKQoKPiAtIGBGdXR1cmVfd2F5cG9pbnRzYCBhcmUgMjAgdGltZSBzdGVwczoKPiAgIC0gRm9ybWF0OiBgWyBbeDAsIHkwXSwgW3gxLCB5MV0sIC4uLiwgW3gxOSwgeTE5XSBdYAo+ICAgLSBJbmRleGVkIGBpID0gMGAgdG8gYDE5YCwgd2l0aAo+ICAgICAtIGBpID0gMGAgLT4gYHQgPSAwLjI1c2AsCj4gICAgIC0gYGkgPSAxOWAgLT4gYHQgPSA1LjBzYAoKLS0tCgpZb3VyIHRhc2s6IEJhc2VkIG9uIHRoZSBwcm92aWRlZCB2aXN1YWwgYW5kIG1vdGlvbiBkYXRhLCBjb21wbGV0ZSB0aGUgZm9sbG93aW5nIGZvdXIgZGV0YWlsZWQgQ2hhaW4tb2YtVGhvdWdodCBzdGVwcy4gSSBhbSBnaXZpbmcgeW91ciBmdXR1cmUgZGVjaXNpb25zIHRvIGhlbHAgeW91IGNvbmZpcm0gdGhhdCB5b3VyIHJlYXNvbmluZyBpcyBjb3JyZWN0LCBidXQgeW91ciByZWFzb25pbmcgcHJvY2VzcyBzaG91bGQgbm90IHJlbHkgb24gdGhlIGZ1dHVyZSB0cmFqZWN0b3J5IGFzIHRoZSBiYXNpcyBmb3IgeW91ciByZWFzb25pbmcuIFlvdXIgcmVhc29uaW5nIGNhbiBvbmx5IG1ha2UgcHJlZGljdGlvbnMgYWJvdXQgZnV0dXJlIGRlY2lzaW9ucyBiYXNlZCBvbiB0aGUgY3VycmVudCBpbWFnZSBhbmQgaGlzdG9yaWNhbCB0cmFqZWN0b3J5LgoKLS0tCgoqKlN0ZXAgMTogU2NlbmUgQW5hbHlzaXMqKgpEZXNjcmliZSB0aGUgb3ZlcmFsbCB0cmFmZmljIHNjZW5lLCBpbmNsdWRpbmcgdHJhZmZpYyBsaWdodHMsIHJvYWQgZ2VvbWV0cnksIGxhbmUgbWFya2luZ3MsIHNpZ25zLCBjcm9zc3dhbGtzLCBjb25zdHJ1Y3Rpb24gYXJlYXMsIGFuZCBhbnkgdGVtcG9yYXJ5IHJvYWQgc3RydWN0dXJlcy4KCioqU3RlcCAyOiBLZXkgT2JqZWN0IElkZW50aWZpY2F0aW9uKioKSWRlbnRpZnkgdXAgdG8gMyBpbXBvcnRhbnQgb3IgcmFyZSBvYmplY3RzIHRoYXQgYXJlIHJlbGV2YW50IGZvciBwbGFubmluZywgYW5kIGFsc28gcHJvdmlkZSBhbiBhc3Nlc3NtZW50IG9mIHRoZWlyIGltcGFjdCBvbiBkcml2aW5nLCBzdWNoIGFzOgotIFRyYWZmaWMgc2lnbmFsIGxpZ2h0IChpZiB2aXNpYmxlKTogZGVzY3JpYmUgaXRzIHN0YXRlIChlLmcuLCBncmVlbiwgcmVkLCB5ZWxsb3csIGJsaW5raW5nLCBvY2NsdWRlZCksIHJlbGF0aXZlIHBvc2l0aW9uLCBhbmQgbWVhbmluZyBmb3IgZWdvIHZlaGljbGUKLSBEeW5hbWljIGFnZW50czogZGVzY3JpYmUgdGhlIHZhcmlvdXMgYXR0cmlidXRlcyBvZiBlYWNoIGFnZW50LCBpdHMgaW1wYWN0IG9uIHRoZSB2ZWhpY2xlIGl0c2VsZiwgYW5kIGl0cyBzaWduaWZpY2FuY2UgaW4gdGhlIHBsYW5uaW5nIG9mIHRoZSB2ZWhpY2xlIGVudGl0eS4KLSBSYXJlIG9iamVjdHMgKGlmIHZpc2libGUpOiAgZS5nLiwgdHJhZmZpYyBlbmZvcmNlcnMsIGNvbnN0cnVjdGlvbiB3b3JrZXJzLCBwb2xpY2Ugb2ZmaWNlcnMsIGVtZXJnZW5jeSByZXNwb25kZXJzLCB0cmFmZmljIGNvbmVzLCByb2FkIGJhcnJpZXJzLCBwYXJrZWQgZGVsaXZlcnkgdHJ1Y2tzLCBmYWxsZW4gdHJlZXMsIGNvbnN0cnVjdGlvbiBtYWNoaW5lcnksIG1pc3NpbmcgbGFuZSBsaW5lcywgbWlzYWxpZ25lZCBjdXJicywgdGVtcG9yYXJ5IGRldG91cnMsIHVudXN1YWwgZW50cnkgcG9pbnRzCgoqKlN0ZXAgMzogSW50ZW50aW9uIEluZmVyZW5jZSoqCkJhc2VkIG9uIGlucHV0IGltYWdlcyAoYm90aCBjdXJyZW50IGFuZCBmdXR1cmUgaW1hZ2VzKSwgdGhlIGludGVudGlvbiBvZiB0aGUgZWdvIHZlaGljbGUsIHRoZSBwYXN0IHN0YXRlIG9mIHRoZSBlZ28gdmVoaWNsZSAodmVsb2NpdHksIGFjY2VsZXJhdGlvbiwgd2F5cG9pbnRzKSwgYW5kIHRoZSBmdXR1cmUgd2F5cG9pbnRzIG9mIHRoZSBlZ28gdmVoaWNsZSwgdGhlIHBvc3NpYmxlIGZ1dHVyZSBtb3ZlbWVudCBvZiB0aGUgcmVjb2duaXplZCBvYmplY3QgKGZvciBleGFtcGxlLCBjcm9zc2luZyB0aGUgcm9hZCwgbWVyZ2luZyBpbnRvIHRoZSBsYW5lKSBpcyBpbmZlcnJlZC4gQW5kIGhvdyBkbyB0aGVzZSBvYmplY3RzIGFmZmVjdCB0aGUgZnV0dXJlIGRyaXZpbmcgb2YgZWdvIHZlaGljbGUuCgoqKlN0ZXAgNDogQWN0aW9uIFJlYXNvbioqCi0gYWN0aW9uIDogSXNzdWUgYSBjb21tYW5kIHJlcXVlc3QgZm9yIHRoZSBmdXR1cmUgdHJhamVjdG9yeS4gRm9yIGV4YW1wbGUsIHR1cm4gbGVmdCBhbmQgc2xvdyBkb3duIGluIHRoZSBmdXR1cmUuIFRoZW4gdGhpcyBwYXJ0IHNob3VsZCBiZSBmaWxsZWQgaW4gYXMgIlBsZWFzZSBzbG93IGRvd24gYW5kIHR1cm4gbGVmdC4iLiBUaGlzIHBhcnQgc2hvdWxkIGJlIGFzIGJyaWVmIGFzIHBvc3NpYmxlLgotIHJlYXNvbiA6IEJhc2VkIHNvbGVseSBvbiB0aGUgY3VycmVudCBpbWFnZSBhbmQgaGlzdG9yaWNhbCB0cmFqZWN0b3J5LCBwcm92aWRlIGEgZGV0YWlsZWQgQ2hhaW4tb2YtVGhvdWdodCBmb3IgdGhpcyBkZWNpc2lvbiB1c2luZyBzdGVwLWJ5LXN0ZXAgcmVhc29uaW5nLgpQbGVhc2Ugbm90ZSB0aGF0IHRoZSAicmVhc29uIiBzaG91bGQgbm90IGludm9sdmUgYW55IHF1YW50aXRhdGl2ZSBhbmFseXNpcy4gUHJvdmlkZSBhIHJlYXNvbmFibGUgYW5kIG5lY2Vzc2FyeSBwcm9jZXNzIGZvciB0aGUgZGVjaXNpb24tbWFraW5nIHJlYXNvbi4KSSBhbSBnaXZpbmcgeW91ciBmdXR1cmUgZGVjaXNpb25zIHRvIGhlbHAgeW91IGNvbmZpcm0gdGhhdCB5b3VyIHJlYXNvbmluZyBpcyBjb3JyZWN0LCBidXQgeW91ciByZWFzb25pbmcgcHJvY2VzcyBzaG91bGQgbm90IHJlbHkgb24gdGhlIGZ1dHVyZSB0cmFqZWN0b3J5IGFzIHRoZSBiYXNpcyBmb3IgeW91ciByZWFzb25pbmcuIFlvdXIgcmVhc29uaW5nIGNhbiBvbmx5IG1ha2UgcHJlZGljdGlvbnMgYWJvdXQgZnV0dXJlIGRlY2lzaW9ucyBiYXNlZCBvbiB0aGUgY3VycmVudCBpbWFnZSBhbmQgaGlzdG9yaWNhbCB0cmFqZWN0b3J5LgotLS0KCioqT3V0cHV0IEZvcm1hdDoqKgpZb3VyIG91dHB1dCBmb3JtYXQgc2hvdWxkIGJlIGEganNvbiBvYmplY3Qgd2l0aCB0aGUgZm9sbG93aW5nIHN0cnVjdHVyZToKYGBganNvbgp7CiAgInNjZW5lX2FuYWx5c2lzIjogIi4uLiIsCiAgImtleV9vYmplY3QiOiAiLi4uIiwKICAiaW50ZW50aW9uX2luZmVyZW5jZSI6ICIuLi4iLAogICJhY3Rpb25fZGVjaXNpb24iOiB7CiAgICAiYWN0aW9uIjogIi4uLiIsCiAgICAicmVhc29uIjogIi4uLiIKICB9Cn0=)

\## The prompts for generating CoT:

You are an autonomous driving planning expert. Your task is to analyze a driving scene using information provided from the images of front camera views, past waypoints, past velocity, past acceleration, current velocity, current acceleration, future waypoints, and the intended motion of the ego vehicle. You must reason through the scene based on visual and temporal data and produce a structured decision output. I am giving your future decisions to help you confirm that your reasoning is correct, but your reasoning process should not rely on the future trajectory as the basis for your reasoning. Your reasoning can only make predictions about future decisions based on the current image and historical trajectory.

—

\*\*Input:\*\*

\- \*\*Past waypoints\*\*: %%<past\_waypoints>%%

\- \*\*Past velocity\*\*: %%<past\_velocity>%%

\- \*\*Past acceleration\*\*: %%<past\_acceleration>%%

\- \*\*Future waypoints\*\*: %%<future\_waypoints>%%

—

\*\*Coordinate and Temporal System Explanation:\*\*

\> All motion and state data are expressed in the ego local coordinate system:

\> - +x = forward direction

\> - +y = left direction

\> - The origin (0, 0) is located at the center of the ego vehicle

\> - Units:

\> - Waypoints: meters

\> - Velocity: meters/second

\> - Acceleration: meters/second

\> - Sampling frequency: 4Hz (every 0.25s)

\> - Past arrays (‘past\_waypoints‘, ‘past\_velocity‘, ‘past\_acceleration‘) are each 16 time steps:

\> - Format: ‘\[ \[x0, y0\], \[x1, y1\], …, \[x15, y15\] \]‘

\> - Indexed ‘i = 0‘ to ‘15‘, with

\> - ‘i = 0‘ -> ‘t = -3.75s‘,

\> - ‘i = 15‘ -> ‘t = 0.0s‘ (current frame)

\> - ‘Future\_waypoints‘ are 20 time steps:

\> - Format: ‘\[ \[x0, y0\], \[x1, y1\], …, \[x19, y19\] \]‘

\> - Indexed ‘i = 0‘ to ‘19‘, with

\> - ‘i = 0‘ -> ‘t = 0.25s‘,

\> - ‘i = 19‘ -> ‘t = 5.0s‘

—

Your task: Based on the provided visual and motion data, complete the following four detailed Chain-of-Thought steps. I am giving your future decisions to help you confirm that your reasoning is correct, but your reasoning process should not rely on the future trajectory as the basis for your reasoning. Your reasoning can only make predictions about future decisions based on the current image and historical trajectory.

—

\*\*Step 1: Scene Analysis\*\*

Describe the overall traffic scene, including traffic lights, road geometry, lane markings, signs, crosswalks, construction areas, and any temporary road structures.

\*\*Step 2: Key Object Identification\*\*

Identify up to 3 important or rare objects that are relevant for planning, and also provide an assessment of their impact on driving, such as:

\- Traffic signal light (if visible): describe its state (e.g., green, red, yellow, blinking, occluded), relative position, and meaning for ego vehicle

\- Dynamic agents: describe the various attributes of each agent, its impact on the vehicle itself, and its significance in the planning of the vehicle entity.

\- Rare objects (if visible): e.g., traffic enforcers, construction workers, police officers, emergency responders, traffic cones, road barriers, parked delivery trucks, fallen trees, construction machinery, missing lane lines, misaligned curbs, temporary detours, unusual entry points

\*\*Step 3: Intention Inference\*\*

Based on input images (both current and future images), the intention of the ego vehicle, the past state of the ego vehicle (velocity, acceleration, waypoints), and the future waypoints of the ego vehicle, the possible future movement of the recognized object (for example, crossing the road, merging into the lane) is inferred. And how do these objects affect the future driving of ego vehicle.

\*\*Step 4: Action Reason\*\*

\- action: Issue a command request for the future trajectory. For example, turn left and slow down in the future. Then this part should be filled in as "Please slow down and turn left.". This part should be as brief as possible.

\- reason: Based solely on the current image and historical trajectory, provide a detailed Chain-of-Thought for this decision using step-by-step reasoning.

Please note that the "reason" should not involve any quantitative analysis. Provide a reasonable and necessary process for the decision-making reason.

I am giving your future decisions to help you confirm that your reasoning is correct, but your reasoning process should not rely on the future trajectory as the basis for your reasoning. Your reasoning can only make predictions about future decisions based on the current image and historical trajectory.

—

\*\*Output Format:\*\*

Your output format should be a json object with the following structure:

“‘json

{

"scene\_analysis": "…",

"key\_object": "…",

"intention\_inference": "…",

"action\_decision": {

"action": "…",

"reason": "…"

}

}

### 6.3 Planning and Instruction Following

For planning task, we directly provide the model with a question that describes the current and historical driving context, and the model is required to predict the future trajectory of the ego vehicle for the next several seconds. The answer is represented as a sequence of K future trajectory points in the ego coordinate frame. The model may output this trajectory either directly based on its internal world understanding or after producing a CoT reasoning sequence when reasoning is beneficial for disambiguating complex interactions. This formulation ensures that trajectory prediction is guided by both contextual scene understanding and physically grounded motion patterns, enabling safe, smooth, and interpretable future driving behavior.

To enable instruction-conditioned trajectory planning, we derive high-level navigation commands such as going straight, turning left, or turning right from the geometric properties of the ground-truth future trajectory. These commands are then incorporated into the QA pairs so that the model must generate a trajectory that is consistent with the given navigation intent. During training, the model learns to align the predicted motion sequence with the provided instruction, establishing a direct correspondence between semantic driving goals and trajectory generation. During evaluation, this setup allows us to assess whether the model can correctly adjust its predicted future motion according to the specified high-level driving intent, ensuring that the resulting planning behavior remains both controllable and interpretable in downstream driving scenes.

## 7 More Cases

To better demonstrate the effectiveness of our method. We have provided even more examples. We compare the understanding and reasoning capabilities of the most advanced GPT4o in scenarios as shown in Fig. 7. Our approach can provide more specific suggestions for planning. We have further provided visualizations for trajectory and weather control generation as shown in Fig. 8 and Fig. 9. This proves the effectiveness of our method’s generation capability.

## 8 Limitation and Future Directions

Despite the promising performance of UniUGP in unifying scene understanding, future video generation, and trajectory planning for autonomous driving, it still has several limitations that point to valuable future research directions.

### 8.1 Limitations

First, while UniUGP uses over 10 diverse AD datasets to cover common and long-tail scenarios, its generalization to extreme rare events (e.g., unprecedented weather, novel obstacles) is constrained by training data coverage—critical for safety-critical systems. Second, the hybrid expert architecture’s computational efficiency is problematic: the generation expert, though useful for visual causal validation, demands excessive resources and must be disabled on resource-constrained mobile platforms to ensure real-time performance. Third, linguistic reasoning and physical dynamics alignment, though improved via multi-term loss functions, is suboptimal; in complex interactive scenarios (e.g., ambiguous pedestrian-vehicle interaction), chain-of-thought (CoT) reasoning may not tightly couple with physically consistent trajectory generation, causing minor interpretability-action inconsistencies. Fourth, the four-stage training strategy relies on fixed dataset proportions in the final fusion stage, failing to dynamically adapt to different datasets’ complementary strengths and limiting task synergy.

### 8.2 Future Directions

To address these limitations, we propose targeted directions: enhance generalization to extreme long-tail scenarios via high-fidelity synthetic data generation (e.g., world models + generative AI) and few-shot/zero-shot learning; optimize model efficiency through lightweight generation expert designs (e.g., knowledge distillation, sparse activation) and reduced multi-expert redundant computations. Deepen multimodal alignment with cross-modal contrastive learning and hierarchical fusion mechanisms that adjust expert weights by scene complexity; reduce labeled data dependence via self-supervised signals (e.g., unsupervised video causal reasoning) and enable incremental adaptation with continual learning to avoid catastrophic forgetting. Extend interaction capabilities to dynamic real-time feedback (e.g., mid-task voice commands) and multi-agent reasoning for complex traffic interactions; integrate UniUGP into closed-loop systems for real-world testing, establishing a performance-refinement feedback loop to boost safety and robustness. These efforts will evolve UniUGP into a more practical framework bridging laboratory performance and real-world deployment.

![[x9.png|Refer to caption]]

Figure 7: Comparison of CoT in our method versus GPT4o. Our approach provides more specific planning results, while the general large model does not offer sufficiently detailed planning outcomes.

![[x11.png|Refer to caption]]

Figure 8: Video generation visualization with controllable weather conditions. Our model can generate videos of different weather conditions, which proves the efficiency of our generation model. Please zoom in to the best view.

![[x13.png|Refer to caption]]

Figure 9: Video generation visualization with controllable trajectory conditions. Our model can generate videos of different trajectory conditions, which proves the efficiency of our generation model. Please zoom in to the best view.

[^1]: Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. *arXiv preprint arXiv:2502.13923*, 2025.

[^2]: Hengwei Bian, Lingdong Kong, Haozhe Xie, Liang Pan, Yu Qiao, and Ziwei Liu. Dynamiccity: Large-scale 4d occupancy generation from dynamic scenes. In *ICLR*, 2025.

[^3]: Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. $\pi$ 0: A vision-language-action flow model for general robot control. corr, abs/2410.24164, 2024. doi: 10.48550. *arXiv preprint ARXIV.2410.24164*.

[^4]: Holger Caesar, Varun Bankiti, Alex H. Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yuxin Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. *CVPR*, 2020.

[^5]: Holger Caesar, Juraj Kabzan, Kok Seang Tan, Whye Kit Fong, Eric Wolff, Alex Lang, Luke Fletcher, Oscar Beijbom, and Sammy Omari. nuplan: A closed-loop ml-based planning benchmark for autonomous vehicles. *arXiv preprint arXiv:2106.11810*, 2021.

[^6]: Tianhui Cai, Yifan Liu, Zewei Zhou, Haoxuan Ma, Seth Z. Zhao, Zhiwen Wu, and Jiaqi Ma. Driving with regulation: Interpretable decision-making for autonomous vehicles with retrieval-augmented reasoning via llm. *ArXiv*, abs/2410.04759, 2024. URL [https://api.semanticscholar.org/CorpusID:273186209](https://api.semanticscholar.org/CorpusID:273186209).

[^7]: Fu-Hsiang Chan, Yu-Ting Chen, Yu Xiang, and Min Sun. Anticipating accidents in dashcam videos. In *Asian Conference on Computer Vision*, pages 136–153. Springer, 2016.

[^8]: Shaoyu Chen, Bo Jiang, Hao Gao, Bencheng Liao, Qing Xu, Qian Zhang, Chang Huang, Wenyu Liu, and Xinggang Wang. Vadv2: End-to-end vectorized autonomous driving via probabilistic planning. *arXiv preprint arXiv:2402.13243*, 2024a.

[^9]: Yuntao Chen, Yu-Quan Wang, and Zhaoxiang Zhang. Drivinggpt: Unifying driving world modeling and planning with multi-modal autoregressive transformers. *arXiv preprint arXiv:2412.18607*, 2024b.

[^10]: Haohan Chi, Huan ang Gao, Ziming Liu, Jianing Liu, Chenyu Liu, Jinwei Li, Kaisen Yang, Yangcheng Yu, Zeda Wang, Wenyi Li, Leichen Wang, Xingtao Hu, Hao Sun, Hang Zhao, and Hao Zhao. Impromptu vla: Open weights and open data for driving vision-language-action models. *ArXiv*, abs/2505.23757, 2025. URL [https://api.semanticscholar.org/CorpusID:278996666](https://api.semanticscholar.org/CorpusID:278996666).

[^11]: Jang Hyun Cho, Boris Ivanovic, Yulong Cao, Edward Schmerling, Yue Wang, Xinshuo Weng, Boyi Li, Yurong You, Philipp Krähenbühl, Yan Wang, et al. Language-image models with 3d understanding. *ICLR*, 2025.

[^12]: Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, Shi Guang, and Haoqi Fan. Emerging properties in unified multimodal pretraining. *ArXiv*, abs/2505.14683, 2025. URL [https://api.semanticscholar.org/CorpusID:278768720](https://api.semanticscholar.org/CorpusID:278768720).

[^13]: Jianwu Fang, Dingxin Yan, Jiahuan Qiao, Jianru Xue, and Hongkai Yu. Dada: Driver attention prediction in driving accident scenarios. *IEEE Transactions on Intelligent Transportation Systems*, 2022.

[^14]: Haoyu Fu, Diankun Zhang, Zongchuang Zhao, Jianfeng Cui, Dingkang Liang, Chong Zhang, Dingyuan Zhang, Hongwei Xie, Bing Wang, and Xiang Bai. Orion: A holistic end-to-end autonomous driving framework by vision-language instructed action generation. *ArXiv*, abs/2503.19755, 2025. URL [https://api.semanticscholar.org/CorpusID:277314093](https://api.semanticscholar.org/CorpusID:277314093).

[^15]: Ruiyuan Gao, Kai Chen, Enze Xie, Lanqing Hong, Zhenguo Li, Dit-Yan Yeung, and Qiang Xu. Magicdrive: Street view generation with diverse 3d geometry control. *ICLR*, 2024a.

[^16]: Shenyuan Gao, Jiazhi Yang, Li Chen, Kashyap Chitta, Yihang Qiu, Andreas Geiger, Jun Zhang, and Hongyang Li. Vista: A generalizable driving world model with high fidelity and versatile controllability. In *NeurIPS*, 2024b.

[^17]: Mariam Hassan, Sebastian Stapf, Ahmad Rahimi, Pedro M B Rezende, Yasaman Haghighi, David Brüggemann, Isinsu Katircioglu, Lin Zhang, Xiaoran Chen, Suman Saha, Marco Cannici, Elie Aljalbout, Botao Ye, Xi Wang, Aram Davtyan, Mathieu Salzmann, Davide Scaramuzza, Marc Pollefeys, Paolo Favaro, and Alexandre Alahi. Gem: A generalizable ego-vision multimodal world model for fine-grained ego-motion, object dynamics, and scene composition control. *CVPR*, 2025.

[^18]: Dan Hendrycks, Steven Basart, Mantas Mazeika, Andy Zou, Joe Kwon, Mohammadreza Mostajabi, Jacob Steinhardt, and Dawn Song. Scaling out-of-distribution detection for real-world settings. *ICML*, 2022.

[^19]: Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. *NeurIPS*, 2017.

[^20]: Anthony Hu, Gianluca Corrado, Nicolas Griffiths, Zak Murez, Corina Gurau, Hudson Yeo, Alex Kendall, Roberto Cipolla, and Jamie Shotton. Model-based imitation learning for urban driving. *NeurIPS*, 2022a.

[^21]: Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, and Gianluca Corrado. Gaia-1: A generative world model for autonomous driving. *arXiv preprint arXiv:2309.17080*, 2023a.

[^22]: Shengchao Hu, Li Chen, Penghao Wu, Hongyang Li, Junchi Yan, and Dacheng Tao. St-p3: End-to-end vision-based autonomous driving via spatial-temporal feature learning. In *ECCV*, 2022b.

[^23]: Xiaotao Hu, Wei Yin, Mingkai Jia, Junyuan Deng, Xiaoyang Guo, Qian Zhang, Xiaoxiao Long, and Ping Tan. Drivingworld: Constructing world model for autonomous driving via video gpt. *arXiv preprint arXiv:2412.19505*, 2024.

[^24]: Yi Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wen Wang, Lewei Lu, Xiaosong Jia, Qiang Liu, Jifeng Dai, Yu Qiao, and Hongyang Li. Planning-oriented autonomous driving. *CVPR*, pages 17853–17862, 2022c. URL [https://api.semanticscholar.org/CorpusID:257687420](https://api.semanticscholar.org/CorpusID:257687420).

[^25]: Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai Wang, Lewei Lu, Xiaosong Jia, Qiang Liu, Jifeng Dai, Yu Qiao, and Hongyang Li. Planning-oriented autonomous driving. In *CVPR*, 2023b.

[^26]: Yuxiao Hu, Qian Li, Dongxiao Zhang, Jinyue Yan, and Yuntian Chen. Context-alignment: Activating and enhancing LLMs capabilities in time series. In *ICLR*, 2025.

[^27]: Junjie Huang, Guan Huang, Zheng Zhu, Yun Ye, and Dalong Du. Bevdet: High-performance multi-camera 3d object detection in bird-eye-view. *arXiv preprint arXiv:2112.11790*, 2021.

[^28]: Ayesha Ishaq, Jean Lahoud, Fahad Shahbaz Khan, Salman Khan, Hisham Cholakkal, and Rao Muhammad Anwer. Tracking meets large multimodal models for driving scenario understanding. *ArXiv preprint arXiv:2503.14498*, 2025.

[^29]: Anqing Jiang, Yu Gao, Zhigang Sun, Yiru Wang, Jijun Wang, Jinghao Chai, Qian Cao, Yuwen Heng, Hao Jiang, Zongzheng Zhang, Xianda Guo, Hao Sun, and Hao Zhao. Diffvla: Vision-language guided diffusion planning for autonomous driving. *ArXiv*, abs/2505.19381, 2025a. URL [https://api.semanticscholar.org/CorpusID:278904380](https://api.semanticscholar.org/CorpusID:278904380).

[^30]: Bo Jiang, Shaoyu Chen, Qing Xu, Bencheng Liao, Jiajie Chen, Helong Zhou, Qian Zhang, Wenyu Liu, Chang Huang, and Xinggang Wang. Vad: Vectorized scene representation for efficient autonomous driving. *ICCV*, 2023.

[^31]: Bo Jiang, Shaoyu Chen, Bencheng Liao, Xingyu Zhang, Wei Yin, Qian Zhang, Chang Huang, Wenyu Liu, and Xinggang Wang. Senna: Bridging large vision-language models and end-to-end autonomous driving. *ArXiv*, abs/2410.22313, 2024. URL [https://api.semanticscholar.org/CorpusID:273661831](https://api.semanticscholar.org/CorpusID:273661831).

[^32]: Bo Jiang, Shaoyu Chen, Qian Zhang, Wenyu Liu, and Xinggang Wang. Alphadrive: Unleashing the power of vlms in autonomous driving via reinforcement learning and reasoning. *ArXiv*, abs/2503.07608, 2025b. URL [https://api.semanticscholar.org/CorpusID:276928398](https://api.semanticscholar.org/CorpusID:276928398).

[^33]: Bohan Li, Jiazhe Guo, Hongsi Liu, Yingshuang Zou, Yikang Ding, Xiwu Chen, Hu Zhu, Feiyang Tan, Chi Zhang, Tiancai Wang, et al. Uniscene: Unified occupancy-centric driving scene generation. *CVPR*, 2025a.

[^34]: Boyi Li, Yue Wang, Jiageng Mao, Boris Ivanovic, Sushant Veer, Karen Leung, and Marco Pavone. Driving everywhere with large language model policy adaptation. In *CVPR*, 2024a.

[^35]: Guopeng Li, Yiru Jiao, Victor L Knoop, Simeon C Calvert, and JWC Van Lint. Large car-following data based on lyft level-5 open dataset: Following autonomous vehicles vs. human-driven vehicles. In *ITSC*, pages 5818–5823. IEEE, 2023.

[^36]: Xiang Li, Pengfei Li, Yupeng Zheng, Wei Sun, Yan Wang, and Yilun Chen. Semi-supervised vision-centric 3d occupancy world model for autonomous driving. *ICLR*, 2025b.

[^37]: Yongkang Li, Kaixin Xiong, Xiangyu Guo, Fang Li, Sixu Yan, Gangwei Xu, Lijun Zhou, Long Chen, Haiyang Sun, Bing Wang, Guang Chen, Hangjun Ye, Wenyu Liu, and Xinggang Wang. Recogdrive: A reinforced cognitive framework for end-to-end autonomous driving. *ArXiv*, abs/2506.08052, 2025c. URL [https://api.semanticscholar.org/CorpusID:279261178](https://api.semanticscholar.org/CorpusID:279261178).

[^38]: Zhiqi Li, Wenhai Wang, Hongyang Li, Enze Xie, Chonghao Sima, Tong Lu, Qiao Yu, and Jifeng Dai. Bevformer: learning bird’s-eye-view representation from lidar-camera via spatiotemporal transformers. *IEEE TPAMI*, 2024b.

[^39]: Bencheng Liao, Shaoyu Chen, Haoran Yin, Bo Jiang, Cheng Wang, Sixu Yan, Xinbang Zhang, Xiangyu Li, Ying Zhang, Qian Zhang, and Xinggang Wang. Diffusiondrive: Truncated diffusion model for end-to-end autonomous driving. *CVPR*, pages 12037–12047, 2024. URL [https://api.semanticscholar.org/CorpusID:274192736](https://api.semanticscholar.org/CorpusID:274192736).

[^40]: Bencheng Liao, Shaoyu Chen, Haoran Yin, Bo Jiang, Cheng Wang, Sixu Yan, Xinbang Zhang, Xiangyu Li, Ying Zhang, Qian Zhang, and Xinggang Wang. Diffusiondrive: Truncated diffusion model for end-to-end autonomous driving. *CVPR*, 2025.

[^41]: Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. *arXiv preprint arXiv:2210.02747*, 2022.

[^42]: Yingfei Liu, Tiancai Wang, Xiangyu Zhang, and Jian Sun. Petr: Position embedding transformation for multi-view 3d object detection. In *ECCV*, pages 531–548. Springer, 2022.

[^43]: Hao Lu, Jiaqi Tang, Xinli Xu, Xu Cao, Yunpeng Zhang, Guoqing Wang, Dalong Du, Hao Chen, and Yingcong Chen. Scaling multi-camera 3d object detection through weak-to-strong eliciting. *arXiv preprint arXiv:2404.06700*, 2024.

[^44]: Hao Lu, Zhuang Ma, Guangfeng Jiang, Wenhang Ge, Bohan Li, Yuzhan Cai, Wenzhao Zheng, Yunpeng Zhang, and Yingcong Chen. 4d driving scene generation with stereo forcing. *arXiv preprint arXiv:2509.20251*, 2025a.

[^45]: Hao Lu, Tianshuo Xu, Wenzhao Zheng, Yunpeng Zhang, Wei Zhan, Dalong Du, Masayoshi Tomizuka, Kurt Keutzer, and Yingcong Chen. Drivingrecon: Large 4d gaussian reconstruction model for autonomous driving. *NeurIPS*, 2025b.

[^46]: Hao Lu, Yunpeng Zhang, Qing Lian, Dalong Du, and Yingcong Chen. Towards generalizable multi-camera 3d object detection via perspective debiasing. *AAAI*, 2025c.

[^47]: Qi Lv, Weijie Kong, Hao Li, Jia Zeng, Zherui Qiu, Delin Qu, Haoming Song, Qizhi Chen, Xiang Deng, and Jiangmiao Pang. F1: A vision-language-action model bridging understanding and generation to actions. *ArXiv*, abs/2509.06951, 2025. URL [https://api.semanticscholar.org/CorpusID:281204333](https://api.semanticscholar.org/CorpusID:281204333).

[^48]: Yingzi Ma, Yulong Cao, Jiachen Sun, Marco Pavone, and Chaowei Xiao. Dolphins: Multimodal language model for driving, 2024.

[^49]: Yiyang Ma, Xingchao Liu, Xiaokang Chen, Wen Liu, Chengyue Wu, Zhiyu Wu, Zizheng Pan, Zhenda Xie, Haowei Zhang, Xingkai Yu, et al. Janusflow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation. In *CVPR*, pages 7739–7751, 2025.

[^50]: Chen Min, Dawei Zhao, Liang Xiao, Jian Zhao, Xinli Xu, Zheng Zhu, Lei Jin, Jianshu Li, Yulan Guo, Junliang Xing, Liping Jing, Yiming Nie, and Bin Dai. Driveworld: 4d pre-trained scene understanding via world models for autonomous driving. *CVPR*, 2024.

[^51]: Chaojun Ni, Guosheng Zhao, Xiaofeng Wang, Zheng Zhu, Wenkang Qin, Guan Huang, Chen Liu, Yuyin Chen, Yida Wang, Xueyang Zhang, Yifei Zhan, Kun Zhan, Peng Jia, Xianpeng Lang, Xingang Wang, and Wenjun Mei. Recondreamer: Crafting world models for driving scene reconstruction via online restoration. In *CVPR*, 2025a.

[^52]: Jingcheng Ni, Yuxin Guo, Yichen Liu, Rui Chen, Lewei Lu, and Zehuan Wu. Maskgwm: A generalizable driving world model with video mask reconstruction. *CVPR*, 2025b.

[^53]: Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuokai Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu, Jiuhai Chen, Kunpeng Li, Felix Juefei-Xu, et al. Transfer between modalities with metaqueries. *arXiv preprint arXiv:2504.06256*, 2025.

[^54]: William Peebles and Saining Xie. Scalable diffusion models with transformers. In *Proceedings of the IEEE/CVF international conference on computer vision*, pages 4195–4205, 2023.

[^55]: Peter Pinggera, Sebastian Ramos, Stefan Gehrig, Uwe Franke, Carsten Rother, and Rudolf Mester. Lost and found: detecting small road hazards for self-driving vehicles. In *2016 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 2016.

[^56]: Xuanchi Ren, Yifan Lu, Tianshi Cao, Ruiyuan Gao, Shengyu Huang, Amirmojtaba Sabour, Tianchang Shen, Tobias Pfaff, Jay Zhangjie Wu, Runjian Chen, et al. Cosmos-drive-dreams: Scalable synthetic driving data generation with world foundation models. *arXiv preprint arXiv:2506.09042*, 2025.

[^57]: Weijia Shi, Xiaochuang Han, Chunting Zhou, Weixin Liang, Xi Victoria Lin, Luke Zettlemoyer, and Lili Yu. Lmfusion: Adapting pretrained language models for multimodal generation. *arXiv preprint arXiv:2412.15188*, 2024.

[^58]: Chonghao Sima, Katrin Renz, Kashyap Chitta, Li Chen, Hanxue Zhang, Chengen Xie, Ping Luo, Andreas Geiger, and Hongyang Li. Drivelm: Driving with graph visual question answering. *ECCV*, 2024.

[^59]: Aasheesh Singh, Aditya Kamireddypalli, Vineet Gandhi, and K Madhava Krishna. Lidar guided small obstacle segmentation. In *IROS*, pages 8513–8520. IEEE, 2020.

[^60]: Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, et al. Scalability in perception for autonomous driving: Waymo open dataset. In *CVPR*, pages 2446–2454, 2020.

[^61]: Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Emu: Generative pretraining in multimodality. *arXiv preprint arXiv:2307.05222*, 2023.

[^62]: Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. *arXiv preprint arXiv:2405.09818*, 2024.

[^63]: Xiaoyu Tian, Junru Gu, Bailin Li, Yicheng Liu, Zhiyong Zhao, Yang Wang, Kun Zhan, Peng Jia, Xianpeng Lang, and Hang Zhao. Drivevlm: The convergence of autonomous driving and large vision-language models. *CoRL*, 2024.

[^64]: Shengbang Tong, David Fan, Jiachen Zhu, Yunyang Xiong, Xinlei Chen, Koustuv Sinha, Michael Rabbat, Yann LeCun, Saining Xie, and Zhuang Liu. Metamorph: Multimodal understanding and generation via instruction tuning. *arXiv preprint arXiv:2412.14164*, 2024.

[^65]: Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. *arXiv preprint arXiv:2503.20314*, 2025.

[^66]: Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. *arXiv preprint arXiv:2409.12191*, 2024a.

[^67]: Shihao Wang, Zhiding Yu, Xiaohui Jiang, Shiyi Lan, Min Shi, Nadine Chang, Jan Kautz, Ying Li, and José M. Álvarez. Omnidrive: A holistic vision-language dataset for autonomous driving with counterfactual reasoning. *CVPR*, pages 22442–22452, 2024b. URL [https://api.semanticscholar.org/CorpusID:269502377](https://api.semanticscholar.org/CorpusID:269502377).

[^68]: Wenhai Wang, Jiangwei Xie, ChuanYang Hu, Haoming Zou, Jianan Fan, Wenwen Tong, Yang Wen, Silei Wu, Hanming Deng, Zhiqi Li, Hao Tian, Lewei Lu, Xizhou Zhu, Xiaogang Wang, Yu Qiao, and Jifeng Dai. Drivemlm: Aligning multi-modal large language models with behavioral planning states for autonomous driving. *ArXiv*, abs/2312.09245, 2023. URL [https://api.semanticscholar.org/CorpusID:266210476](https://api.semanticscholar.org/CorpusID:266210476).

[^69]: Xiaofeng Wang, Zheng Zhu, Guan Huang, Xinze Chen, Jiagang Zhu, and Jiwen Lu. Drivedreamer: Towards real-world-driven world models for autonomous driving. *ECCV*, 2024c.

[^70]: Xiaofeng Wang, Zheng Zhu, Guan Huang, Xinze Chen, Jiagang Zhu, and Jiwen Lu. Drivedreamer: Towards real-world-drive world models for autonomous driving. In *ECCV*, pages 55–72. Springer, 2024d.

[^71]: Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. *arXiv preprint arXiv:2409.18869*, 2024e.

[^72]: Yan Wang, Wenjie Luo, Junjie Bai, Yulong Cao, Tong Che, Ke Chen, Yuxiao Chen, Jenna Diamond, Yifan Ding, Wenhao Ding, et al. Alpamayo-r1: Bridging reasoning and action prediction for generalizable autonomous driving in the long tail. *arXiv preprint arXiv:2511.00088*, 2025.

[^73]: Yuqi Wang, Jiawei He, Lue Fan, Hongxin Li, Yuntao Chen, and Zhaoxiang Zhang. Driving into the future: Multiview visual forecasting and planning with world model for autonomous driving. *CVPR*, 2024f.

[^74]: Julong Wei, Shanshuai Yuan, Pengfei Li, Qingda Hu, Zhongxue Gan, and Wenchao Ding. Occllama: An occupancy-language-action generative world model for autonomous driving. *arXiv preprint arXiv:2409.03272*, 2024.

[^75]: Yuqing Wen, Yucheng Zhao, Yingfei Liu, Fan Jia, Yanhui Wang, Chong Luo, Chi Zhang, Tiancai Wang, Xiaoyan Sun, and Xiangyu Zhang. Panacea: Panoramic and controllable video generation for autonomous driving. In *CVPR*, pages 6902–6912, 2024.

[^76]: Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, et al. Janus: Decoupling visual encoding for unified multimodal understanding and generation. In *CVPR*, pages 12966–12977, 2025.

[^77]: Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. *arXiv preprint arXiv:2408.12528*, 2024.

[^78]: Jinheng Xie, Zhenheng Yang, and Mike Zheng Shou. Show-o2: Improved native unified multimodal models. *arXiv preprint arXiv:2506.15564*, 2025.

[^79]: Runsheng Xu, Hubert Lin, Wonseok Jeon, Hao Feng, Yuliang Zou, Liting Sun, John Gorman, Kate Tolstaya, Sarah Tang, Brandyn White, et al. Wod-e2e: Waymo open dataset for end-to-end driving in challenging long-tail scenarios. *arXiv preprint arXiv:2510.26125*, 2025a.

[^80]: Tianshuo Xu, Hao Lu, Xu Yan, Yingjie Cai, Bingbing Liu, and Yingcong Chen. Occ-llm: Enhancing autonomous driving with occupancy-based large language models. *ICRA*, 2025b.

[^81]: Tianshuo Xu, Hao Lu, Xu Yan, Yingjie Cai, Bingbing Liu, and Yingcong Chen. Occ-llm: Enhancing autonomous driving with occupancy-based large language models. *ICRA*, 2025c.

[^82]: Yunzhi Yan, Haotong Lin, Chenxu Zhou, Weijie Wang, Haiyang Sun, Kun Zhan, Xianpeng Lang, Xiaowei Zhou, and Sida Peng. Street gaussians: Modeling dynamic urban scenes with gaussian splatting. In *ECCV*, pages 156–173. Springer, 2024.

[^83]: Jiazhi Yang, Shenyuan Gao, Yihang Qiu, Li Chen, Tianyu Li, Bo Dai, Kashyap Chitta, Penghao Wu, Jia Zeng, Ping Luo, Jun Zhang, Andreas Geiger, Yu Qiao, and Hongyang Li. Generalized predictive model for autonomous driving. In *CVPR*, 2024a.

[^84]: Jiazhi Yang, Shenyuan Gao, Yihang Qiu, Li Chen, Tianyu Li, Bo Dai, Kashyap Chitta, Penghao Wu, Jia Zeng, Ping Luo, et al. Generalized predictive model for autonomous driving. In *CVPR*, pages 14662–14672, 2024b.

[^85]: Zetong Yang, Li Chen, Yanan Sun, and Hongyang Li. Visual point cloud forecasting enables scalable autonomous driving. In *CVPR*, 2024c.

[^86]: Zhenjie Yang, Yilin Chai, Xiaosong Jia, Qifeng Li, Yuqian Shao, Xuekai Zhu, Haisheng Su, and Junchi Yan. Drivemoe: Mixture-of-experts for vision-language-action model in end-to-end autonomous driving, 2025.

[^87]: Shuang Zeng, Xinyuan Chang, Xinran Liu, Zheng Pan, and Xing Wei. Driving with prior maps: Unified vector prior encoding for autonomous vehicle mapping. *arXiv preprint arXiv:2409.05352*, 2024.

[^88]: Shuang Zeng, Xinyuan Chang, Mengwei Xie, Xinran Liu, Yifan Bai, Zheng Pan, Mu Xu, and Xing Wei. Futuresightdrive: Thinking visually with spatio-temporal cot for autonomous driving. *NeurIPS*, 2025.

[^89]: Andy Zhai, Brae Liu, Bruno Fang, Chalse Cai, Ellie Ma, Ethan Yin, Hao Wang, Hugo Zhou, James Wang, Lights Shi, Lucy Liang, Make Wang, Qian Wang, Roy Gan, Ryan Yu, Shalfun Li, Starrick Liu, Sylas Chen, Vincent Chen, and Z. Xu. Igniting vlms toward the embodied space. *ArXiv*, abs/2509.11766, 2025. URL [https://api.semanticscholar.org/CorpusID:281315304](https://api.semanticscholar.org/CorpusID:281315304).

[^90]: Jiawei Zhang, Chejian Xu, and Bo Li. Chatscene: Knowledge-enabled safety-critical scenario generation for autonomous vehicles. In *CVPR*, 2024.

[^91]: Kaiwen Zhang, Zhenyu Tang, Xiaotao Hu, Xingang Pan, Xiaoyang Guo, Yuan Liu, Jingwei Huang, Li Yuan, Qian Zhang, Xiaoxiao Long, Xun Cao, and Wei Yin. Epona: Autoregressive diffusion world model for autonomous driving. *ArXiv*, abs/2506.24113, 2025. URL [https://api.semanticscholar.org/CorpusID:280010626](https://api.semanticscholar.org/CorpusID:280010626).

[^92]: Guosheng Zhao, Chaojun Ni, Xiaofeng Wang, Zheng Zhu, Xueyang Zhang, Yida Wang, Guan Huang, Xinze Chen, Boyuan Wang, Youyi Zhang, Wenjun Mei, and Xingang Wang. Drivedreamer4d: World models are effective data machines for 4d driving scene representation. In *CVPR*, 2025a.

[^93]: Guosheng Zhao, Chaojun Ni, Xiaofeng Wang, Zheng Zhu, Xueyang Zhang, Yida Wang, Guan Huang, Xinze Chen, Boyuan Wang, Youyi Zhang, et al. Drivedreamer4d: World models are effective data machines for 4d driving scene representation. In *CVPR*, pages 12015–12026, 2025b.

[^94]: Peiru Zheng, Yun Zhao, Zhan Gong, Hong Zhu, and Shaohua Wu. Simplellm4ad: An end-to-end vision-language model with graph visual question answering for autonomous driving. *ArXiv preprint arXiv:2407.21293*, 2024a.

[^95]: Wenzhao Zheng, Weiliang Chen, Yuanhui Huang, Borui Zhang, Yueqi Duan, and Jiwen Lu. Occworld: Learning a 3d occupancy world model for autonomous driving. *ECCV*, 2024b.

[^96]: Wenzhao Zheng, Ruiqi Song, Xianda Guo, Chenming Zhang, and Long Chen. Genad: Generative end-to-end autonomous driving. *ECCV*, 2024c.

[^97]: Wenzhao Zheng, Zetian Xia, Yuanhui Huang, Sicheng Zuo, Jie Zhou, and Jiwen Lu. Doe-1: Closed-loop autonomous driving with large world model. *ArXiv*, abs/2412.09627, 2024d. URL [https://api.semanticscholar.org/CorpusID:274656022](https://api.semanticscholar.org/CorpusID:274656022).

[^98]: Wenzhao Zheng, Zetian Xia, Yuanhui Huang, Sicheng Zuo, Jie Zhou, and Jiwen Lu. Doe-1: Closed-loop autonomous driving with large world model. *arXiv preprint arXiv: 2412.09627*, 2024e.

[^99]: Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. *arXiv preprint arXiv:2408.11039*, 2024a.

[^100]: Xiaoyu Zhou, Zhiwei Lin, Xiaojun Shan, Yongtao Wang, Deqing Sun, and Ming-Hsuan Yang. Drivinggaussian: Composite gaussian splatting for surrounding dynamic autonomous driving scenes. In *CVPR*, pages 21634–21643, 2024b.

[^101]: Xin Zhou, Dingkang Liang, Sifan Tu, Xiwu Chen, Yikang Ding, Dingyuan Zhang, Feiyang Tan, Hengshuang Zhao, and Xiang Bai. Hermes: A unified self-driving world model for simultaneous 3d scene understanding and generation. *ArXiv*, abs/2501.14729, 2025a. URL [https://api.semanticscholar.org/CorpusID:275906961](https://api.semanticscholar.org/CorpusID:275906961).

[^102]: Zewei Zhou, Tianhui Cai, Seth Z. Zhao, Yun Zhang, Zhiyu Huang, Bolei Zhou, and Jiaqi Ma. Autovla: A vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. *ArXiv*, abs/2506.13757, 2025b. URL [https://api.semanticscholar.org/CorpusID:279410595](https://api.semanticscholar.org/CorpusID:279410595).

[^103]: Sicheng Zuo, Wenzhao Zheng, Yuanhui Huang, Jie Zhou, and Jiwen Lu. Gaussianworld: Gaussian world model for streaming 3d occupancy prediction. *CVPR*, 2025.
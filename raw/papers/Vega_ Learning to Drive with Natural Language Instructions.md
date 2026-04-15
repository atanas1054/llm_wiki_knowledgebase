---
title: "Vega: Learning to Drive with Natural Language Instructions"
source: "https://arxiv.org/html/2603.25741v1"
author:
published:
created: 2026-04-08
description:
tags:
  - "clippings"
---
Sicheng Zuo <sup>1,∗</sup>  Yuxuan Li <sup>1,∗</sup>  Wenzhao Zheng <sup>1,∗,†</sup>  Zheng Zhu <sup>2</sup>  Jie Zhou <sup>1</sup>  Jiwen Lu <sup>1</sup>  
<sup>1</sup> Tsinghua University   <sup>2</sup> GigaAI  
Project Page: [https://zuosc19.github.io/Vega](https://zuosc19.github.io/Vega)  
Large Driving Models: [https://github.com/wzzheng/LDM](https://github.com/wzzheng/LDM)

###### Abstract

Vision-language-action models have reshaped autonomous driving to incorporate languages into the decision-making process. However, most existing pipelines only utilize the language modality for scene descriptions or reasoning and lack the flexibility to follow diverse user instructions for personalized driving. To address this, we first construct a large-scale driving dataset (InstructScene) containing around 100,000 scenes annotated with diverse driving instructions with the corresponding trajectories. We then propose a unified vision-language-world-action model, Vega, for instruction-based generation and planning. We employ the autoregressive paradigm to process visual inputs (vision) and language instructions (language) and the diffusion paradigm to generate future predictions (world modeling) and trajectories (action). We perform joint attention to enable interactions between the modalities and use individual projection layers for different modalities for more capabilities. Extensive experiments demonstrate that our method not only achieves superior planning performance but also exhibits strong instruction-following abilities, paving the way for more intelligent and personalized driving systems. Code is available at [https://github.com/zuosc19/Vega](https://github.com/zuosc19/Vega).

![[Uncaptioned image]](https://arxiv.org/html/2603.25741v1/x1.png)

Figure 1: Visualizations of our model for instructional driving. We propose a unified vision-language-world-action model, Vega, for instruction-based generation and planning. Vega can predict multiple trajectories in the same scenario following diverse instructions.

<sup>†</sup>

## 1 Introduction

Vision-centric autonomous driving is a promising direction due to its economic advantages and scalability [^37] [^21] [^67] [^60] [^19] [^27]. Conventional methods typically follow a modular pipeline of perception [^20] [^41] [^22] [^23] [^86], prediction [^79] [^66] [^87] [^82], and planning [^19] [^27] [^80] [^2] [^81], which heavily relies on expensive 3D annotations and thus faces limitations in real-world applications. Recently, vision-language-action (VLA) models have emerged to leverage rich world knowledge from large language models to map visual inputs to driving actions [^62] [^25] [^72] [^13], demonstrating remarkable generalization across driving scenarios.

Despite their good generalization across driving scenarios, most existing VLA models only use languages for scene descriptions or decision reasoning and lack flexible instruction-following capabilities [^84] [^13] [^85] [^75] [^35]. They are either trained to imitate an averaged expert policy, or are confined to a closed set of simple navigational commands like “turn left” or “go straight”, failing to generalize to open-ended and flexible natural language instructions. In contrast, a general driving agent should not only navigate autonomously but also comprehend and execute diverse, user-specified natural language instructions. For instance, a user in a hurry might instruct the vehicle to “overtake the front car to catch the next green light” rather than adhere to the conservative policy learned from the training data.

To facilitate the shift from imitation driving to instructional driving, we construct a large-scale driving dataset, InstructScene, with around 100,000 instruction-annotated scenes and the corresponding trajectories built on NAVSIM [^5]. While a direct way is to train a VLA model on our driving dataset containing rich instructions, we find that it struggles to generate feasible trajectories and follow instructions accurately. We think this is due to the significant information disparity between the high-dimensional visual-instruction inputs and the low-dimensional action prediction, making it difficult for the model to learn a generalizable mapping from high-level instructions to low-level actions in complex and dynamic environments.

To address this, we propose a unified vision-language-world-action model, Vega, for joint instruction-based generation and planning. We train the model to jointly perform future image generation and action planning conditioned on past observations and language instructions. This task provides a dense and pixel-level supervision signal, compelling the model to learn the causal relationships among instructions, actions, and visual predictions. The joint modeling enforces consistency between predictions, enabling mutual supervision and refinement. Our model adopts a mixed autoregressive-diffusion transformer architecture [^38] [^44] [^56] [^6] to achieve unified vision-language understanding, world modeling, and action planning. Specifically, we use the autoregressive pipeline for visual and instruction understanding, and the diffusion pipeline [^10] [^40] for image and action generation. We use joint attention to enable interactions across all modalities and employ a Mixture-of-Transformers (MoT) design [^38] to effectively decouple the parameters associated with different modalities and enhance the model capacity for joint generation and planning. Extensive experiments on the NAVSIM [^5] [^1] benchmark show that our model not only achieves superior planning performance but also demonstrates a remarkable ability to generate high-fidelity and instruction-compliant future images and plausible trajectories.

## 2 Related Work

### 2.1 VLM and VLA for Autonomous Driving

The extensive world knowledge and reasoning capabilities of vision-language models (VLMs) have driven their applications in autonomous driving [^55] [^24] [^73] [^45]. Early works primarily leveraged VLMs for high-level driving scene understanding and reasoning, but could not output drivable trajectories [^43] [^24] [^57] [^50] [^8] [^48] [^46] [^29]. Subsequent methods attempted to have VLMs directly predict textual waypoints [^62] [^7] [^25] [^72], but they struggled due to the inherent limitations of LLMs in precise numerical reasoning [^12] [^49]. This led to the development of VLA models, which integrate a planning module for end-to-end trajectory prediction [^28] [^84] [^13]. Common planning approaches include autoregressive prediction of discretized waypoints [^84] [^26] [^85], diffusion-based trajectory generation [^75] [^13] [^35], and direct regression via an MLP head [^53]. However, these models suffer from sparse action supervision and often rely on auxiliary understanding and reasoning tasks to guide the learning process [^84] [^13] [^85]. In contrast, Vega employs world modeling to provide a dense signal to enhance instruction-based planning.

### 2.2 World Models for Autonomous Driving

World models are typically defined as generative models that predict future states conditioned on past observations and current actions [^16]. In autonomous driving, applications of world models can be categorized into three main approaches: image-based, occupancy-based, and VLA-based methods. Image-based methods leverage powerful generative architectures to synthesize high-fidelity driving videos, primarily for data generation and scene simulation [^18] [^54] [^64] [^78] [^14] [^65]. Occupancy-based methods model scene evolution in 3D occupancy space to enhance scene understanding [^66] [^47] [^87] and planning [^79] [^66] [^74] [^31], but their reliance on dense 3D labels limits scalability. Recently, VLA-based methods have emerged with Doe-1 [^82] first proposing a closed-loop driving model that unifies scene understanding, prediction, and planning. DriveVLA-W0 [^33] integrated world modeling into a VLA framework to provide dense supervision and enhance planning. However, they can not perform instruction-based prediction and planning. Our work enables this capability, allowing the model to predict corresponding future scenes and driving trajectories conditioned on flexible language instructions.

### 2.3 Unified Visual Understanding and Generation

Unified visual understanding and generation methods can be categorized into three main pipelines: quantized autoregressive (AR), external diffusion, and integrated transformers. Quantized AR models quantize images into discrete tokens [^30] [^77], enabling generation within the native autoregressive framework [^69] [^3] [^42] [^51] [^70] [^71] [^59] [^63]. While this design is straightforward, its visual quality typically lags behind that of diffusion-based methods. The External Diffuser approach pairs a VLM with an external diffusion model [^9] [^15] [^58] [^61]. The VLM provides a high-level understanding by generating a few latent tokens that condition the diffusion generator. However, this narrow interface between understanding and generation can restrict information flow [^6]. Integrated transformer models merge autoregressive and diffusion mechanisms into a single transformer [^44] [^56] [^83] [^6] [^38], enabling a deep integration of powerful understanding and generation capabilities. In this paper, we adopt the integrated transformer to achieve instruction-based joint visual generation and action planning.

## 3 Proposed Approach

![[Uncaptioned image]](https://arxiv.org/html/2603.25741v1/x2.png)

Figure 2: Overview of our model. Compared to traditional imitation driving models, which can only predict the single expert trajectory, Vega can follow natural language instructions to generate diverse planning trajectories and future image predictions.

### 3.1 Imitation Driving to Instructional Driving

An autonomous driving model $\mathcal{M}$ usually takes as input the past $T$ and current image observations $[I_{t-T},\dots,I_{t}]$ and past $T$ actions $[A_{t-T},\dots,A_{t-1}]$, and predicts the current action $A_{t}$ for the ego car, which can be formulated as:

$$
A_{t}=\mathcal{M}([I_{t-T},\dots,I_{t}],[A_{t-T},\dots,A_{t-1}]).
$$

Conventional methods often adopt a perception-prediction-planning pipeline. The perception module $\mathcal{P}_{er}$ extracts the scene representation $\mathbf{z}$ from observations $[I_{t-T},\dots,I_{t}]$. Then the prediction module $\mathcal{P}_{re}$ forecasts the future motion $\mathbf{v}$ of agents based on $\mathbf{z}$. Finally, the planning module $\mathcal{P}_{lan}$ uses $\mathbf{z}$, $\mathbf{v}$, and historical ego actions $[A_{t-T},\dots,A_{t-1}]$ to plan the current ego action $A_{t}$. This multi-step pipeline can be expressed as:

$$
\displaystyle\mathbf{z}
$$
 
$$
\displaystyle=\mathcal{P}_{er}(I_{t-T},\dots,I_{t}]),
$$
$$
\displaystyle\mathbf{v}
$$
 
$$
\displaystyle=\mathcal{P}_{re}(\mathbf{z}),
$$
$$
\displaystyle A_{t}
$$
 
$$
\displaystyle=\mathcal{P}_{lan}(\mathbf{z},\mathbf{v},[A_{t-T},\dots,A_{t-1}]).
$$

However, such methods heavily rely on costly high-quality 3D annotations, which greatly limits their scalability.

Recently, vision-language-action (VLA) models have been applied to autonomous driving, leveraging their rich world knowledge and demonstrating strong generalization across diverse scenarios. Based on past observations $[I_{t-T},\dots,I_{t}]$ and historical actions $[A_{t-T},\dots,A_{t-1}]$, current VLA models $\mathcal{W}$ often predict both the textual description of the scene $D$ and the current ego action $A_{t}$. This end-to-end planning process can be formulated as:

$$
A_{t},D_{t}=\mathcal{W}([I_{t-T},\dots,I_{t}],[A_{t-T},\dots,A_{t-1}]).
$$

Although existing VLA models show remarkable generalization, they fall short in flexible instruction-following. Most VLA models are trained to imitate an averaged expert policy or process a closed set of simple navigational commands, failing to handle open-ended natural language instructions. To address this, we introduce an instruction-based driving model $\mathcal{V}$, which predicts the current ego action $A_{t}$ based on observations $[I_{t-T},\dots,I_{t}]$, historical actions $[A_{t-T},\dots,A_{t-1}]$ and the current user instruction $L_{t}$. This process can be expressed as:

$$
A_{t}=\mathcal{V}([I_{t-T},\dots,I_{t}],[A_{t-T},\dots,A_{t-1}],L_{t}).
$$

To enable instruction-based driving, we constructed a large-scale driving dataset with around 100,000 instruction-annotated scenes based on NAVSIM [^5], where we generated instructions automatically using VLM, supplemented by rule-based methods. For each timestep t, we prompt a powerful VLM [^52] with future observations $[I_{t+1},\dots,I_{t+N}]$ and actions $[A_{t+1},\dots,A_{t+N}]$ to produce a high-level instruction $L_{t}$ describing the driving intent of the current ego-vehicle. This process yields a sequence of image, instruction, and action triplets: $\mathcal{D}=\{\langle I_{t},L_{t},a_{t}\rangle\}_{t=1}^{T_{max}}$. We then train our model on this dataset, equipping it with instruction-following driving capabilities.

### 3.2 Unified Generation and Planning

While a direct way to achieve instruction-based driving is to train a VLA model on our driving dataset containing rich instructions, we find that it struggles to generate feasible trajectories and accurately follow instructions, due to the sparse action supervision. To address the supervision gap, we introduce the vision-language-world-action model, a novel framework that jointly learns instruction-based action planning and future image generation. Our core insight is that future image generation provides a dense, pixel-level supervision signal, which helps the model learn the underlying dynamics of the world. By joint modeling generation and planning, the model is compelled to learn the causal relationships among instructions, actions, and visual outcomes, which is critical for instruction-based planning.

The framework is formulated as a generative model trained on triplets of images, instructions, and actions, which models the fundamental causal chain of driving: An agent perceives the world $I_{t}$, receives the instruction $L_{t}$, decides on an action $A_{t}$, and observes the next outcome $I_{t+1}$. At each timestep $t$, the model receives the current observation $I_{t}$ and instruction $L_{t}$, and the historical observations $[I_{t-T},\dots,I_{t-1}]$. It then jointly predicts the action $A_{t}$ to be executed and the resulting next step $I_{t+1}$. We apply causal attention modeling to the model’s architecture, ensuring that it learns the correct reasoning pathway from instruction to action and then to visual outcome, providing a solid foundation for resolving the supervision gap.

### 3.3 Joint Autoregressive-Diffusion Architecture

Unified generation and planning requires our model to not only possess significant visual-text understanding, visual generation, and action planning capabilities, but also integrates them to solve complex driving scenarios. Current research mainly follows three approaches to bridge the gap between visual-text-understanding, which primarily uses auto-regressive VLM, and image generation, which often adopts diffusion models. However, most methods fall short of our requirements. Autoregressive visual generation models with discrete visual tokenizers struggle to match diffusion models in image quality and also suffer from high latency due to their sequential generation pipeline. LLMs combined with external diffusers yield competitive results, but are constrained by an information bottleneck caused by the limited number of latent tokens passed from LLMs to generation modules. To address these, we adopt the Integrated Transformer architecture [^6], which fuses auto-regressive VLM and diffusion transformer into a single model, enabling the generation module to interact with the understanding module without information loss and resulting in unified understanding and generation capabilities.

Our integrated model employs a unified paradigm to predict images and actions. It first encodes multi-modal inputs, including text, images, and actions, and concatenates them to the noises of target images or actions, forming a unified sequence. The model then processes the sequence as a whole, calculating causal attention across modalities to ensure full information flow among text, image, and action latents. Finally, the denoised latents are decoded by their respective decoders into images or actions.

Encoding Inputs. To prepare the multi-modal inputs for the forward pass, we first encode them with corresponding tokenizers. For text, we tokenize natural language inputs $L_{t}$ with the Qwen2.5 tokenizer. For visual understanding, we only use the forward-view camera images as visual observations, which are encoded by a VAE encoder into latents $F_{t}^{V}$. To enrich the visual context, we also encode input images with a SigLIP2 ViT encoder, and append the latents to the corresponding image’s VAE latents. For action, we first convert the 2D absolute trajectory $traj=[(x,y,\theta),\dots]$ into relative movements between consecutive steps $A=(\Delta x,\Delta y,\Delta\theta)$, so that actions from different steps share a distribution and can be easily normalized. We project the normalized relative action sequence into the latent dimension of the model with a linear head.

![[x3 17.png|Refer to caption]]

Figure 3: 38

Constructing Input Sequence. We then combine the multi-modal segments in an interleaving manner. The historical images $[I_{t-T},...,I_{t}]$ and actions $[A_{t-T},...,A_{t-1}]$ are placed at the beginning, followed by natural language instructions $L_{t}$. When performing the action planning task, we then append a noisy target action $A_{t}^{noisy}$. Otherwise, we first add the ground truth current action $A_{t}$, then append a noisy future image $I_{t+K}^{noisy}$ for visual generation. Due to the strictly causal nature of our sequence $S=[I_{0},L_{0},A_{0},...,I_{n},L_{n},A_{n}]$, we set the attention mask as a blocked lower triangular matrix, so that each block, representing an image, action, or instruction, can only attend to previous blocks. In the text block, we adopt a strictly lower triangular mask for causal self-attention and allocates consecutive RoPE indices to textual tokens. In the image or action block, we adopt a full attention mask and share the same RoPE index for all tokens, using sinusoidal positional embedding to encode relative position instead.

During inference, the model denoises the action and future image sequentially, where future image prediction is conditioned on a fully denoised action. While during training, the two tasks are optimized jointly for training efficiency. A direct concatenation of noisy action and image inputs would cause later tokens to attend to noisy preceding latents, creating a mismatch with inference and degrading training. To resolve this, we duplicate each latent that serves both as a prediction target and as a condition for subsequent predictions. Specifically, we add noise to the first copy $F_{t}^{noisy}$ and use it for denoising supervision, while keeping the second copy $F_{t}^{clean}$ as the condition input. We further mask $F_{t}^{noisy}$ from all subsequent tokens, ensuring that they attend only to the clean latents. This design allows us to train multiple diffusion processes within a single autoregressive sequence efficiently.

Integrated Transformer. To enhance the performance of our integrated transformer, we decouple the modules and weights in charge of each capability so that they can be optimized individually. Unlike the Mixture of Expert (MoE) technique, which only uses separate weights for FFN, we employ the Mixture of Transformers (MoT) architecture [^38] [^56], where all trainable parameters of the transformer, including attention and FFN layers, are duplicated for each module. This design has been shown to not only converge faster, but also maintain higher model capacity [^6]. Specifically, we process visual and text understanding tokens with a understanding transformer based on Qwen2.5 LLM [^52], which has a hidden size of 3584 and a depth of 28 layers. Image generation tokens are processed by a generation transformer of the same design. The weights of both transformers are initialized from Bagel-7B [^6]. Due to the relatively low dimensionality of the action space, we reduce the hidden size of the action module to 256, thus reducing action-related computation without significantly degrading model performance.

During the forward process, the interleaving multi-modal sequence is split into segments and passed onto their respective modules in each attention and FFN layers. To calculate global causal attention, the sequence is re-assembled to be processed as a whole. Tokens for image generation and action planning are then extracted from the output sequence for final prediction.

### 3.4 Training and Inference

We implement a single-stage training paradigm to cover both action planning and world modeling. For action planning, we train the model to predict the action plan $A_{t}^{(N)}=[A_{t},...,A_{t+N-1}]$ based on past observations $I_{t}^{(-T)}=[I_{t-T},...,I_{t}]$ and current driving instruction $L_{t}$. For world modeling, we train the model to predict future image observation $I_{t+K}$ based on past images $I_{t}^{(-T)}$, current driving instruction $L_{t}$ and action plan $A_{t}^{(N)}$. We use the MSE of the normalized relative action $A$ as action loss:

$$
\mathcal{L}_{A}=\mathbb{E}_{A_{t}^{(N)},\epsilon,m}[||\epsilon-\hat{\epsilon}(A_{t}^{(N)},\epsilon,m,I_{t}^{(-T)},L_{t})||^{2}],
$$

where $\epsilon\sim\mathcal{N}(0,\mathbf{I})$ is sampled Gaussian noise and $m$ is a random timestep, and MSE of the VAE latents $F^{V}$ as image loss:

$$
\mathcal{L}_{V}=\mathbb{E}_{F_{t+K}^{V},\epsilon,n}[||\epsilon-\hat{\epsilon}(F_{t+K}^{V},\epsilon,n,I_{t}^{(-T)},L_{t},A_{t}^{(N)})||^{2}],
$$

where $\epsilon\sim\mathcal{N}(0,\mathbf{I})$ is sampled Gaussian noise and $n$ is a random timestep. To enable classifier-free guidance (CFG) [^17] in inference, we randomly drop text, ViT, clean VAE, and clean action tokens during training. Tokens of the same modality that belong to different images or actions are dropped or kept jointly.

Table 1: Comparison with state-of-the-art methods on the NAVSIM v2 with extended metrics. NC: No at-fault Collision. DAC: Drivable Area Compliance. DDC: Driving Direction Compliance. TLC: Traffic Light Compliance. EP: Ego Progress. TTC: Time to Collision. LK: Lane Keeping. HC: History Comfort. EC: Extended Comfort. EPDMS: Extended Predictive Driver Model Score. †: Using the best-of-N (N=6) strategy following [^85].

| Method | NC $\uparrow$ | DAC $\uparrow$ | DDC $\uparrow$ | TLC $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | LK $\uparrow$ | HC $\uparrow$ | EC $\uparrow$ | EPDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ego Status | 93.1 | 77.9 | 92.7 | 99.6 | 86.0 | 91.5 | 89.4 | 98.3 | 85.4 | 64.0 |
| TransFuser [^4] | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 |
| HydraMDP++ [^36] | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | 81.4 |
| DriveSuprim [^76] | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 | 83.1 |
| ARTEMIS [^11] | 98.3 | 95.1 | 98.6 | 99.8 | 81.5 | 97.4 | 96.5 | 98.3 | \- | 83.1 |
| DiffusionDrive [^39] | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | 84.5 |
| DriveVLA-W0 | 98.5 | 99.1 | 98.0 | 99.7 | 86.4 | 98.1 | 93.2 | 97.9 | 58.9 | 86.1 |
| gray!20 Vega | 98.9 | 95.3 | 99.4 | 99.9 | 87.0 | 98.4 | 96.5 | 98.3 | 76.3 | 86.9 |
| gray!20 Vega † | 99.2 | 96.6 | 99.5 | 99.9 | 87.5 | 98.7 | 97.4 | 98.4 | 84.5 | 89.4 |

In the training stage, we optimize a joint objective with loss $\mathcal{L}_{pretrain}=\lambda_{A}\cdot\mathcal{L}_{A}+\lambda_{V}\cdot\mathcal{L}_{V}$. This allows the model to learn world knowledge alongside planning capabilities. In the inference stage, we use Classifier-Free Guidance Diffusion [^17] to generate actions, with both image guidance and text guidance enabled. While we primarily focus on the action planning task during inference, the model retains its image generation capabilities from the training stage.

## 4 Experiments

### 4.1 Datasets and Benchmarks

- NAVSIM v1 [^5] filters OpenScene to remove near-trivial and erroneous scenes, reducing the train split size to 85k. During evaluation, NAVSIM v1 runs a non-reactive simulation at 10Hz for future 4 seconds, then scores the driving agent with metrics including No at-fault Collision (NC), Drivable Area Compliance (DAC), Time To Collision (TTC), Comfort (Comf.), and Ego Progress (EP). These metrics are aggregated into the Predictive Driver Model Score (PDMS). We use the train split for finetuning and the test split for evaluation.
- NAVSIM v2 [^1] improves simulation realism by enabling reactive traffic. It evaluates agents with the Extended Predictive Driver Model Score (EPDMS), adding metrics including Driving Direction Compliance (DDC), Traffic Light Compliance (TLC), Lane Keeping (LK), History Comfort (HC) and Extended Comfort (EC).

Table 2: Comparison with state-of-the-art methods on the NAVSIM v1. NC: no at-fault collision. DAC: drivable area compliance. TTC: time-to-collision. C.: comfort. EP: ego progress. PDMS: the predictive driver model score. Abbreviations: 1x Cam (single front-view camera), Nx Cam (surround-view cameras), L (LiDAR). †: Using the best-of-N (N=6) strategy following [^85].

<table><tbody><tr><td>Method</td><td>Ref</td><td>Sensors</td><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>C. <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>Human</td><td>-</td><td>-</td><td>100</td><td>100</td><td>100</td><td>99.9</td><td>87.5</td><td>94.8</td></tr><tr><td colspan="9">BEV-based Methods</td></tr><tr><td>UniAD <sup><a href="#fn:19">19</a></sup></td><td>CVPR’23</td><td>6x Cam</td><td>97.8</td><td>91.9</td><td>92.9</td><td>100.0</td><td>78.8</td><td>83.4</td></tr><tr><td>TransFuser <sup><a href="#fn:4">4</a></sup></td><td>TPAMI’23</td><td>3x Cam + L</td><td>97.7</td><td>92.8</td><td>92.8</td><td>100.0</td><td>79.2</td><td>84.0</td></tr><tr><td>PARA-Drive <sup><a href="#fn:68">68</a></sup></td><td>CVPR’24</td><td>6x Cam</td><td>97.9</td><td>92.4</td><td>93.0</td><td>99.8</td><td>79.3</td><td>84.0</td></tr><tr><td>LAW <sup><a href="#fn:32">32</a></sup></td><td>ICLR’25</td><td>1x Cam</td><td>96.4</td><td>95.4</td><td>88.7</td><td>99.9</td><td>81.7</td><td>84.6</td></tr><tr><td>Hydra-MDP <sup><a href="#fn:36">36</a></sup></td><td>arXiv’24</td><td>3x Cam + L</td><td>98.3</td><td>96.0</td><td>94.6</td><td>100.0</td><td>78.7</td><td>86.5</td></tr><tr><td>DiffusionDrive <sup><a href="#fn:39">39</a></sup></td><td>CVPR’25</td><td>3x Cam + L</td><td>98.2</td><td>96.2</td><td>94.7</td><td>100.0</td><td>82.2</td><td>88.1</td></tr><tr><td>WoTE <sup><a href="#fn:34">34</a></sup></td><td>ICCV’25</td><td>3x Cam + L</td><td>98.5</td><td>96.8</td><td>94.4</td><td>99.9</td><td>81.9</td><td>88.3</td></tr><tr><td colspan="9">VLA-based Methods</td></tr><tr><td>AutoVLA <sup><a href="#fn:85">85</a></sup></td><td>NeurIPS’25</td><td>3x Cam</td><td>98.4</td><td>95.6</td><td>98.0</td><td>99.9</td><td>81.9</td><td>89.1</td></tr><tr><td>ReCogDrive <sup><a href="#fn:35">35</a></sup></td><td>arXiv’25</td><td>3x Cam</td><td>98.2</td><td>textbf97.8</td><td>95.2</td><td>99.8</td><td>83.5</td><td>89.6</td></tr><tr><td>AutoVLA† <sup><a href="#fn:85">85</a></sup></td><td>NeurIPS’25</td><td>3x Cam</td><td>99.1</td><td>97.1</td><td>97.1</td><td>100.0</td><td>87.6</td><td>92.1</td></tr><tr><td>DriveVLA-W0†</td><td>arXiv’25</td><td>1x Cam</td><td>99.3</td><td>97.4</td><td>97.0</td><td>99.9</td><td>88.3</td><td>93.0</td></tr><tr><td>gray!20 Vega</td><td>-</td><td>1x Cam</td><td>98.9</td><td>95.3</td><td>96.1</td><td>100.0</td><td>81.6</td><td>87.9</td></tr><tr><td>gray!20 Vega †</td><td>-</td><td>1x Cam</td><td>99.2</td><td>96.6</td><td>96.9</td><td>100.0</td><td>83.4</td><td>89.8</td></tr></tbody></table>

Table 3: Ablation of future image prediction. PDMS: NAVSIM v1 [^5] benchmark, EPDMS: NAVSIM v2 [^1] benchmark.

| Setting | PDMS $\uparrow$ | EPDMS $\uparrow$ |
| --- | --- | --- |
| Random Frame | 77.3 | 75.2 |
| Action Only | 51.8 | 48.9 |
| Next Frame | 77.9 | 76.0 |

Table 4: Ablation of action expert. PDMS: NAVSIM v1 [^5] benchmark, EPDMS: NAVSIM v2 [^1] benchmark.

| Setting | PDMS $\uparrow$ | EPDMS $\uparrow$ |
| --- | --- | --- |
| Use Diffusion | 19.7 | 19.6 |
| Use VLM | 77.6 | 75.7 |
| Action Expert | 77.9 | 76.0 |

### 4.2 Implementation Details

Instruction Annotation. The driving instructions in our InstructScene dataset were generated by a fully-automated two-stage annotation pipeline. We select Qwen2.5-VL-72B-Instruct [^52] as our annotation model for its powerful visual understanding capabilities. The inputs of each scene are 14 consecutive frames captured by the front-view camera at 2Hz, with a resolution of $(1920,1080)$. The first 4 frames are considered past and current observations, and the last 10 frames are future observations that will not be available to the driving agent in the inference stage.

- Stage One: Scene Understanding. In stage one, we prompt the model with two requests, designed to convert the visual inputs and expected actions of the driving agent into natural language descriptions. We first instruct the model to describe the scene in the first 4 frames and to identify the traffic participants as well as the static objects. We then instruct it to describe the vehicle’s driving behavior in the next 10 frames and its interaction with previously observed traffic participants.
- Stage Two: Instruction Formulation. In stage two, we combine the visual inputs with the scene descriptions generated in stage one, and prompt the model to create concise driving instructions that would guide the driving agent to predict the actions described in stage one.

Since VLMs struggle to accurately perceive ego-vehicle motion, we generate supplementary rule-based instructions. We classify scenes using speed, acceleration, and turn rate thresholds, converting them into natural language. While these closed-set instructions lack diversity, they provide precise ego-motion cues. We then use them as auxiliary prompts for the VLM, combining both to generate accurate and diverse driving instructions. With this pipeline, we annotated $85109$ scenes from the NAVSIM train split and $12144$ scenes from the NAVSIM test split.

Training. The model is trained on the NAVSIM train split for 200k steps using 8 H20 GPUs. We set the number of historical images to 4, and predict 8 future actions as well as the future image at the end of the actions. We set the learning rate to 2e-5 with 2500 warmup steps, and use a per-device batch size of 1. The weights of action and image loss are $\lambda_{A}=\lambda_{V}=1.0$. We also maintain an EMA model with a decay rate of 0.9999, which is saved as checkpoints.

### 4.3 Main Results

As shown in Tables 1 and 2, our model demonstrates competitive performance on both NAVSIM benchmarks. On NAVSIM v2, it scores 86.9 EPDMS without any additional performance-enhancing techniques, which is comparable to SOTA. Using the best-of-N strategy as prior works [^85] [^33], it achieves top performance on NAVSIM v2, surpassing state-of-the-art methods on several metrics, including Driving Direction Compliance, Traffic Light Compliance, Lane Keeping, and History Comfort. These results suggest that Vega has learned robust instruction following capabilities and benefited from future image prediction training. On NAVSIM v1, our model achieves 87.9 PDMS, matching multi-modal BEV methods, and improves to 89.8 with the best-of-N strategy. We note that Vega achieves lower performance compared to state-of-the-art VLA-based methods on NAVSIM v1. This discrepancy is partially attributed to NAVSIM v1’s inbalanced metrics, which disproportionately favor risk-averse policies over alternative, equally valid strategies learned by our model. Furthermore, competing VLA-based methods either require supplementary inputs such as multi-view images with high resolutions, or integrate CoT reasoning via additional RL training. Critically, these performance-enhancing mechanisms operate independently of our model’s core architecture and may be modularly incorporated without modifying the primary design.

![[Uncaptioned image]](https://arxiv.org/html/2603.25741v1/x4.png)

Figure 4: Ablation of interleaving image-action sequences. We compare the finetuning losses of models trained on non-interleaving sequences (original) and interleaving sequences of different lengths.

![[x5 16.png|Refer to caption]]

Figure 5: Instruction-based planning examples. We visualize the effects of language instructions on action planning with front-view camera images and BEV maps.

### 4.4 Experimental Analysis

Future Frame Prediction. During pretraining, the model is trained to predict the first future frame. We ablate future frame prediction with two additional configurations. First, we randomly sample one of the 8 future frames and specify the chosen index in the text prompt. Second, we remove the future frame prediction task altogether. All three configurations are trained on 8 H20 GPUs, with other hyperparameters identical to pretraining in Section 4.2. The results, shown in Table 3, indicate that the task of future frame prediction indeed improves the planning capabilities of the model, but the exact choice of future frame has limited impact on performance.

Interleaving Observation and Action. In our original design, only past images are provided to the model as reference. We argue that interleaving image and action helps the model learn their dynamics, resulting in faster convergence and lower loss during training. Following [^33], we pretrain the model with interleaving image-action sequences. We ablate the pretraining image-action sequence length with 2, 4 and 6. During finetuning, we interleave the original 4 past images with 3 past actions. Figure 4 reveals that although models pretrained on interleaving sequences suffer from higher loss in the initial stages of finetuning, which can be attributed to the discrepancy between their pretraining and finetuning designs, they converge significantly faster than models without interleaving sequences, eventually surpassing the latter. In addition, although all interleaving models use the same sequence length during finetuning, those with longer image-action sequences during pretraining show lower losses.

Independent Action Module. To validate our design of an additional action expert, we ablate it against using the existing VLM or diffusion modules for action planning. Although these alternatives reduce model size, they increase computational cost because of the high dimensions of these modules. As shown in Table 4, our action expert model slightly outperforms the VLM-module-based planner and significantly surpasses the diffusion-module-based one, confirming the effectiveness of our architecture.

![[x6 14.png|Refer to caption]]

Figure 6: Future image generation conditioned on instructions and actions. In the same scenario, given two sets of instructions, the model plans two action sequences and generates their respective future images. Both action sequences follow their instructions and both images are consistent with their actions.

Visualizations. In addition to Figure 1, we provide two more examples to visualize the effect of instructions on Vega’s ability to adjust the vehicle’s speed according to user instructions in Figure 5. We test two instructions in each scene and plot the predicted trajectories for the next 8 frames on the front-view-camera image as well as the Bird’s-Eye-View (BEV) map. In both scenes, our model successfully increased, decreased, or maintained speed to follow the instructions. We also offer a qualitative evaluation of our pretrained model’s ability to align both its action planning and image generation with user instructions in Figure 6. We select critical scenes where there can be multiple possible courses of action, e.g. approaching the intersection, encountering another vehicle. For each scene, we give the model different sets of instructions, then generate future actions and images sequentially. We observe that Vega is able to generate future actions and images that are consistent with the instructions, indicating that our world modeling framework has successfully helped the model learn the dynamics of the driving environment.

VLA Baseline. As a straightforward baseline for instructional driving, we extend Qwen-2.5-VL [^52] with a planning head to predict future actions based on language instructions. Despite being trained on the same dataset with instruction annotations as Vega, this VLA model performs poorly, achieving only $\sim 60$ PDMS and often failing to generate instruction-consistent trajectories. We attribute this limitation to the sparse low-dimensional action supervision, which is insufficient to bridge high-dimensional visual-language inputs and low-level driving actions. This motivates us to explore dense visual supervision from future prediction to improve instruction-based planning.

## 5 Conclusion

In this paper, we have aimed to address current driving models’ inability to follow diverse driving instructions. We have introduced Vega, a unified vision-language-world-action model that bridges this gap by leveraging future visual generation as a dense supervision signal. We have built a large-scale driving dataset with instruction annotations to enable the training for instructional driving. By jointly generating instruction-compliant future images and planning actions, the model learns the causal relationships among instructions, actions, and the visual outcomes. Built upon an integrated transformer architecture and an instruction-annotated dataset, our model achieves SOTA planning performance while demonstrating strong instruction-following capabilities in both visual generation and action planning.

## References

[^1]: Wei Cao, Marcel Hallgarten, Tianyu Li, Daniel Dauner, Xunjiang Gu, Caojun Wang, Yakov Miron, Marco Aiello, Hongyang Li, Igor Gilitschenski, et al. Pseudo-simulation for autonomous driving. *arXiv preprint arXiv:2506.04218*, 2025.

[^2]: Shaoyu Chen, Bo Jiang, Hao Gao, Bencheng Liao, Qing Xu, Qian Zhang, Chang Huang, Wenyu Liu, and Xinggang Wang. Vadv2: End-to-end vectorized autonomous driving via probabilistic planning. *arXiv preprint arXiv:2402.13243*, 2024.

[^3]: Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. *arXiv preprint arXiv:2501.17811*, 2025.

[^4]: Kashyap Chitta, Aditya Prakash, Bernhard Jaeger, Zehao Yu, Katrin Renz, and Andreas Geiger. Transfuser: Imitation with transformer-based sensor fusion for autonomous driving. *TPAMI*, 45(11):12878–12895, 2022.

[^5]: Daniel Dauner, Marcel Hallgarten, Tianyu Li, Xinshuo Weng, Zhiyu Huang, Zetong Yang, Hongyang Li, Igor Gilitschenski, Boris Ivanovic, Marco Pavone, et al. Navsim: Data-driven non-reactive autonomous vehicle simulation and benchmarking. *NeurIPS*, 37:28706–28719, 2024.

[^6]: Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. *arXiv preprint arXiv:2505.14683*, 2025.

[^7]: Kairui Ding, Boyuan Chen, Yuchen Su, Huan-ang Gao, Bu Jin, Chonghao Sima, Wuqiang Zhang, Xiaohui Li, Paul Barsch, Hongyang Li, et al. Hint-ad: Holistically aligned interpretability in end-to-end autonomous driving. *arXiv preprint arXiv:2409.06702*, 2024a.

[^8]: Xinpeng Ding, Jianhua Han, Hang Xu, Xiaodan Liang, Wei Zhang, and Xiaomeng Li. Holistic autonomous driving understanding by bird’s-eye-view injected multi-modal large models. In *CVPR*, pages 13668–13677, 2024b.

[^9]: Runpei Dong, Chunrui Han, Yuang Peng, Zekun Qi, Zheng Ge, Jinrong Yang, Liang Zhao, Jianjian Sun, Hongyu Zhou, Haoran Wei, et al. Dreamllm: Synergistic multimodal comprehension and creation. *arXiv preprint arXiv:2309.11499*, 2023.

[^10]: Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In *ICML*, 2024.

[^11]: Renju Feng, Ning Xi, Duanfeng Chu, Rukang Wang, Zejian Deng, Anzheng Wang, Liping Lu, Jinxiang Wang, and Yanjun Huang. Artemis: Autoregressive end-to-end trajectory planning with mixture of experts for autonomous driving. *arXiv preprint arXiv:2504.19580*, 2025.

[^12]: Simon Frieder. Mathematical capabilities of chatgpt. *arXiv preprint arXiv:2301.13867*, 2023.

[^13]: Haoyu Fu, Diankun Zhang, Zongchuang Zhao, Jianfeng Cui, Dingkang Liang, Chong Zhang, Dingyuan Zhang, Hongwei Xie, Bing Wang, and Xiang Bai. Orion: A holistic end-to-end autonomous driving framework by vision-language instructed action generation. *arXiv preprint arXiv:2503.19755*, 2025.

[^14]: Shenyuan Gao, Jiazhi Yang, Li Chen, Kashyap Chitta, Yihang Qiu, Andreas Geiger, Jun Zhang, and Hongyang Li. Vista: A generalizable driving world model with high fidelity and versatile controllability. *NeurIPS*, 37:91560–91596, 2024.

[^15]: Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li, Xiaohan Ding, and Ying Shan. Seed-x: Multimodal models with unified multi-granularity comprehension and generation. *arXiv preprint arXiv:2404.14396*, 2024.

[^16]: David Ha and Jürgen Schmidhuber. World models. *arXiv preprint arXiv:1803.10122*, 2(3), 2018.

[^17]: Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. *arXiv preprint arXiv:2207.12598*, 2022.

[^18]: Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, and Gianluca Corrado. Gaia-1: A generative world model for autonomous driving. *arXiv preprint arXiv:2309.17080*, 2023a.

[^19]: Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai Wang, et al. Planning-oriented autonomous driving. In *CVPR*, pages 17853–17862, 2023b.

[^20]: Junjie Huang, Guan Huang, Zheng Zhu, Yun Ye, and Dalong Du. Bevdet: High-performance multi-camera 3d object detection in bird-eye-view. *arXiv preprint arXiv:2112.11790*, 2021.

[^21]: Yuanhui Huang, Wenzhao Zheng, Yunpeng Zhang, Jie Zhou, and Jiwen Lu. Tri-perspective view for vision-based 3d semantic occupancy prediction. In *CVPR*, pages 9223–9232, 2023.

[^22]: Yuanhui Huang, Wenzhao Zheng, Yunpeng Zhang, Jie Zhou, and Jiwen Lu. Gaussianformer: Scene as gaussians for vision-based 3d semantic occupancy prediction. In *ECCV*, pages 376–393. Springer, 2024a.

[^23]: Yuanhui Huang, Amonnut Thammatadatrakoon, Wenzhao Zheng, Yunpeng Zhang, Dalong Du, and Jiwen Lu. Gaussianformer-2: Probabilistic gaussian superposition for efficient 3d occupancy prediction. In *CVPR*, pages 27477–27486, 2025.

[^24]: Zhijian Huang, Chengjian Feng, Feng Yan, Baihui Xiao, Zequn Jie, Yujie Zhong, Xiaodan Liang, and Lin Ma. Drivemm: All-in-one large multimodal model for autonomous driving. *arXiv preprint arXiv:2412.07689*, 2024b.

[^25]: Jyh-Jing Hwang, Runsheng Xu, Hubert Lin, Wei-Chih Hung, Jingwei Ji, Kristy Choi, Di Huang, Tong He, Paul Covington, Benjamin Sapp, et al. Emma: End-to-end multimodal model for autonomous driving. *arXiv preprint arXiv:2410.23262*, 2024.

[^26]: Anqing Jiang, Yu Gao, Zhigang Sun, Yiru Wang, Jijun Wang, Jinghao Chai, Qian Cao, Yuweng Heng, Hao Jiang, Yunda Dong, et al. Diffvla: Vision-language guided diffusion planning for autonomous driving. *arXiv preprint arXiv:2505.19381*, 2025a.

[^27]: Bo Jiang, Shaoyu Chen, Qing Xu, Bencheng Liao, Jiajie Chen, Helong Zhou, Qian Zhang, Wenyu Liu, Chang Huang, and Xinggang Wang. Vad: Vectorized scene representation for efficient autonomous driving. In *ICCV*, pages 8340–8350, 2023.

[^28]: Bo Jiang, Shaoyu Chen, Bencheng Liao, Xingyu Zhang, Wei Yin, Qian Zhang, Chang Huang, Wenyu Liu, and Xinggang Wang. Senna: Bridging large vision-language models and end-to-end autonomous driving. *arXiv preprint arXiv:2410.22313*, 2024.

[^29]: Bo Jiang, Shaoyu Chen, Qian Zhang, Wenyu Liu, and Xinggang Wang. Alphadrive: Unleashing the power of vlms in autonomous driving via reinforcement learning and reasoning. *arXiv preprint arXiv:2503.07608*, 2025b.

[^30]: Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. Autoregressive image generation using residual quantization. In *CVPR*, pages 11523–11532, 2022.

[^31]: Xiang Li, Pengfei Li, Yupeng Zheng, Wei Sun, Yan Wang, and Yilun Chen. Semi-supervised vision-centric 3d occupancy world model for autonomous driving. *arXiv preprint arXiv:2502.07309*, 2025a.

[^32]: Yingyan Li, Lue Fan, Jiawei He, Yuqi Wang, Yuntao Chen, Zhaoxiang Zhang, and Tieniu Tan. Enhancing end-to-end autonomous driving with latent world model. *arXiv preprint arXiv:2406.08481*, 2024a.

[^33]: Yingyan Li, Shuyao Shang, Weisong Liu, Bing Zhan, Haochen Wang, Yuqi Wang, Yuntao Chen, Xiaoman Wang, Yasong An, Chufeng Tang, et al. Drivevla-w0: World models amplify data scaling law in autonomous driving. *arXiv preprint arXiv:2510.12796*, 2025b.

[^34]: Yingyan Li, Yuqi Wang, Yang Liu, Jiawei He, Lue Fan, and Zhaoxiang Zhang. End-to-end driving with online trajectory evaluation via bev world model. *arXiv preprint arXiv:2504.01941*, 2025c.

[^35]: Yongkang Li, Kaixin Xiong, Xiangyu Guo, Fang Li, Sixu Yan, Gangwei Xu, Lijun Zhou, Long Chen, Haiyang Sun, Bing Wang, et al. Recogdrive: A reinforced cognitive framework for end-to-end autonomous driving. *arXiv preprint arXiv:2506.08052*, 2025d.

[^36]: Zhenxin Li, Kailin Li, Shihao Wang, Shiyi Lan, Zhiding Yu, Yishen Ji, Zhiqi Li, Ziyue Zhu, Jan Kautz, Zuxuan Wu, et al. Hydra-mdp: End-to-end multimodal planning with multi-target hydra-distillation. *arXiv preprint arXiv:2406.06978*, 2024b.

[^37]: Zhiqi Li, Wenhai Wang, Hongyang Li, Enze Xie, Chonghao Sima, Tong Lu, Qiao Yu, and Jifeng Dai. Bevformer: learning bird’s-eye-view representation from lidar-camera via spatiotemporal transformers. *TPAMI*, 2024c.

[^38]: Weixin Liang, Lili Yu, Liang Luo, Srinivasan Iyer, Ning Dong, Chunting Zhou, Gargi Ghosh, Mike Lewis, Wen-tau Yih, Luke Zettlemoyer, et al. Mixture-of-transformers: A sparse and scalable architecture for multi-modal foundation models. *arXiv preprint arXiv:2411.04996*, 2024.

[^39]: Bencheng Liao, Shaoyu Chen, Haoran Yin, Bo Jiang, Cheng Wang, Sixu Yan, Xinbang Zhang, Xiangyu Li, Ying Zhang, Qian Zhang, et al. Diffusiondrive: Truncated diffusion model for end-to-end autonomous driving. In *CVPR*, pages 12037–12047, 2025.

[^40]: Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. *arXiv preprint arXiv:2210.02747*, 2022.

[^41]: Zhijian Liu, Haotian Tang, Alexander Amini, Xinyu Yang, Huizi Mao, Daniela L Rus, and Song Han. Bevfusion: Multi-task multi-sensor fusion with unified bird’s-eye view representation. In *2023 IEEE international conference on robotics and automation (ICRA)*, pages 2774–2781. IEEE, 2023.

[^42]: Jiasen Lu, Christopher Clark, Sangho Lee, Zichen Zhang, Savya Khosla, Ryan Marten, Derek Hoiem, and Aniruddha Kembhavi. Unified-io 2: Scaling autoregressive multimodal models with vision language audio and action. In *CVPR*, pages 26439–26455, 2024.

[^43]: Yingzi Ma, Yulong Cao, Jiachen Sun, Marco Pavone, and Chaowei Xiao. Dolphins: Multimodal language model for driving. In *ECCV*, pages 403–420. Springer, 2024.

[^44]: Yiyang Ma, Xingchao Liu, Xiaokang Chen, Wen Liu, Chengyue Wu, Zhiyu Wu, Zizheng Pan, Zhenda Xie, Haowei Zhang, Xingkai Yu, et al. Janusflow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation. In *CVPR*, pages 7739–7751, 2025.

[^45]: Jiageng Mao, Yuxi Qian, Junjie Ye, Hang Zhao, and Yue Wang. Gpt-driver: Learning to drive with gpt. *arXiv preprint arXiv:2310.01415*, 2023.

[^46]: Ana-Maria Marcu, Long Chen, Jan Hünermann, Alice Karnsund, Benoit Hanotte, Prajwal Chidananda, Saurabh Nair, Vijay Badrinarayanan, Alex Kendall, Jamie Shotton, et al. Lingoqa: Visual question answering for autonomous driving. In *ECCV*, pages 252–269. Springer, 2024.

[^47]: Chen Min, Dawei Zhao, Liang Xiao, Jian Zhao, Xinli Xu, Zheng Zhu, Lei Jin, Jianshu Li, Yulan Guo, Junliang Xing, et al. Driveworld: 4d pre-trained scene understanding via world models for autonomous driving. In *CVPR*, pages 15522–15533, 2024.

[^48]: Sung-Yeon Park, Can Cui, Yunsheng Ma, Ahmadreza Moradipari, Rohit Gupta, Kyungtae Han, and Ziran Wang. Nuplanqa: A large-scale dataset and benchmark for multi-view driving scene understanding in multi-modal large language models. *arXiv preprint arXiv:2503.12772*, 2025.

[^49]: Shuai Peng, Ke Yuan, Liangcai Gao, and Zhi Tang. Mathbert: A pre-trained model for mathematical formula understanding. *arXiv preprint arXiv:2105.00377*, 2021.

[^50]: Tianwen Qian, Jingjing Chen, Linhai Zhuo, Yang Jiao, and Yu-Gang Jiang. Nuscenes-qa: A multi-modal visual question answering benchmark for autonomous driving scenario. In *AAAI*, pages 4542–4550, 2024.

[^51]: Liao Qu, Huichao Zhang, Yiheng Liu, Xu Wang, Yi Jiang, Yiming Gao, Hu Ye, Daniel K Du, Zehuan Yuan, and Xinglong Wu. Tokenflow: Unified image tokenizer for multimodal understanding and generation. In *CVPR*, pages 2545–2555, 2025.

[^52]: Qwen,:, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025.

[^53]: Katrin Renz, Long Chen, Elahe Arani, and Oleg Sinavski. Simlingo: Vision-only closed-loop autonomous driving with language-action alignment. In *CVPR*, pages 11993–12003, 2025.

[^54]: Lloyd Russell, Anthony Hu, Lorenzo Bertoni, George Fedoseev, Jamie Shotton, Elahe Arani, and Gianluca Corrado. Gaia-2: A controllable multi-view generative world model for autonomous driving. *arXiv preprint arXiv:2503.20523*, 2025.

[^55]: Hao Shao, Yuxuan Hu, Letian Wang, Guanglu Song, Steven L Waslander, Yu Liu, and Hongsheng Li. Lmdrive: Closed-loop end-to-end driving with large language models. In *CVPR*, pages 15120–15130, 2024.

[^56]: Weijia Shi, Xiaochuang Han, Chunting Zhou, Weixin Liang, Xi Victoria Lin, Luke Zettlemoyer, and Lili Yu. Lmfusion: Adapting pretrained language models for multimodal generation. *arXiv preprint arXiv:2412.15188*, 2024.

[^57]: Chonghao Sima, Katrin Renz, Kashyap Chitta, Li Chen, Hanxue Zhang, Chengen Xie, Jens Beißwenger, Ping Luo, Andreas Geiger, and Hongyang Li. Drivelm: Driving with graph visual question answering. In *ECCV*, pages 256–274. Springer, 2024.

[^58]: Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative multimodal models are in-context learners. In *CVPR*, pages 14398–14409, 2024.

[^59]: Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. *arXiv preprint arXiv:2405.09818*, 2024.

[^60]: Xiaoyu Tian, Tao Jiang, Longfei Yun, Yucheng Mao, Huitong Yang, Yue Wang, Yilun Wang, and Hang Zhao. Occ3d: A large-scale 3d occupancy prediction benchmark for autonomous driving. *NeurIPS*, 36:64318–64330, 2023.

[^61]: Shengbang Tong, David Fan, Jiachen Li, Yunyang Xiong, Xinlei Chen, Koustuv Sinha, Michael Rabbat, Yann LeCun, Saining Xie, and Zhuang Liu. Metamorph: Multimodal understanding and generation via instruction tuning. In *ICCV*, pages 17001–17012, 2025.

[^62]: Shihao Wang, Zhiding Yu, Xiaohui Jiang, Shiyi Lan, Min Shi, Nadine Chang, Jan Kautz, Ying Li, and Jose M Alvarez. Omnidrive: A holistic vision-language dataset for autonomous driving with counterfactual reasoning. In *Proceedings of the computer vision and pattern recognition conference*, pages 22442–22452, 2025.

[^63]: Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. *arXiv preprint arXiv:2409.18869*, 2024a.

[^64]: Xiaofeng Wang, Zheng Zhu, Guan Huang, Xinze Chen, Jiagang Zhu, and Jiwen Lu. Drivedreamer: Towards real-world-drive world models for autonomous driving. In *ECCV*, pages 55–72. Springer, 2024b.

[^65]: Yuqi Wang, Jiawei He, Lue Fan, Hongxin Li, Yuntao Chen, and Zhaoxiang Zhang. Driving into the future: Multiview visual forecasting and planning with world model for autonomous driving. In *CVPR*, pages 14749–14759, 2024c.

[^66]: Julong Wei, Shanshuai Yuan, Pengfei Li, Qingda Hu, Zhongxue Gan, and Wenchao Ding. Occllama: An occupancy-language-action generative world model for autonomous driving. *arXiv preprint arXiv:2409.03272*, 2024.

[^67]: Yi Wei, Linqing Zhao, Wenzhao Zheng, Zheng Zhu, Jie Zhou, and Jiwen Lu. Surroundocc: Multi-camera 3d occupancy prediction for autonomous driving. In *ICCV*, pages 21729–21740, 2023.

[^68]: Xinshuo Weng, Boris Ivanovic, Yan Wang, Yue Wang, and Marco Pavone. Para-drive: Parallelized architecture for real-time autonomous driving. In *CVPR*, pages 15449–15458, 2024.

[^69]: Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, et al. Janus: Decoupling visual encoding for unified multimodal understanding and generation. In *CVPR*, pages 12966–12977, 2025.

[^70]: Junfeng Wu, Yi Jiang, Chuofan Ma, Yuliang Liu, Hengshuang Zhao, Zehuan Yuan, Song Bai, and Xiang Bai. Liquid: Language models are scalable multi-modal generators. *arXiv e-prints*, pages arXiv–2412, 2024a.

[^71]: Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, et al. Vila-u: a unified foundation model integrating visual understanding and generation. *arXiv preprint arXiv:2409.04429*, 2024b.

[^72]: Shuo Xing, Chengyuan Qian, Yuping Wang, Hongyuan Hua, Kexin Tian, Yang Zhou, and Zhengzhong Tu. Openemma: Open-source multimodal model for end-to-end autonomous driving. In *WACV*, pages 1001–1009, 2025.

[^73]: Zhenhua Xu, Yujia Zhang, Enze Xie, Zhen Zhao, Yong Guo, Kwan-Yee K Wong, Zhenguo Li, and Hengshuang Zhao. Drivegpt4: Interpretable end-to-end autonomous driving via large language model. *RA-L*, 2024.

[^74]: Yu Yang, Jianbiao Mei, Yukai Ma, Siliang Du, Wenqing Chen, Yijie Qian, Yuxiang Feng, and Yong Liu. Driving in the occupancy world: Vision-centric 4d occupancy forecasting and planning via world models for autonomous driving. In *AAAI*, pages 9327–9335, 2025a.

[^75]: Zhenjie Yang, Yilin Chai, Xiaosong Jia, Qifeng Li, Yuqian Shao, Xuekai Zhu, Haisheng Su, and Junchi Yan. Drivemoe: Mixture-of-experts for vision-language-action model in end-to-end autonomous driving. *arXiv preprint arXiv:2505.16278*, 2025b.

[^76]: Wenhao Yao, Zhenxin Li, Shiyi Lan, Zi Wang, Xinglong Sun, Jose M Alvarez, and Zuxuan Wu. Drivesuprim: Towards precise trajectory selection for end-to-end planning. *arXiv preprint arXiv:2506.06659*, 2025.

[^77]: Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. Vector-quantized image modeling with improved vqgan. *arXiv preprint arXiv:2110.04627*, 2021.

[^78]: Guosheng Zhao, Xiaofeng Wang, Zheng Zhu, Xinze Chen, Guan Huang, Xiaoyi Bao, and Xingang Wang. Drivedreamer-2: Llm-enhanced world models for diverse driving video generation. In *AAAI*, pages 10412–10420, 2025.

[^79]: Wenzhao Zheng, Weiliang Chen, Yuanhui Huang, Borui Zhang, Yueqi Duan, and Jiwen Lu. Occworld: Learning a 3d occupancy world model for autonomous driving. In *ECCV*, pages 55–72. Springer, 2024a.

[^80]: Wenzhao Zheng, Ruiqi Song, Xianda Guo, Chenming Zhang, and Long Chen. Genad: Generative end-to-end autonomous driving. In *ECCV*, pages 87–104. Springer, 2024b.

[^81]: Wenzhao Zheng, Junjie Wu, Yao Zheng, Sicheng Zuo, Zixun Xie, Longchao Yang, Yong Pan, Zhihui Hao, Peng Jia, Xianpeng Lang, et al. Gaussianad: Gaussian-centric end-to-end autonomous driving. *arXiv preprint arXiv:2412.10371*, 2024c.

[^82]: Wenzhao Zheng, Zetian Xia, Yuanhui Huang, Sicheng Zuo, Jie Zhou, and Jiwen Lu. Doe-1: Closed-loop autonomous driving with large world model. *arXiv preprint arXiv:2412.09627*, 2024d.

[^83]: Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. *arXiv preprint arXiv:2408.11039*, 2024.

[^84]: Xingcheng Zhou, Xuyuan Han, Feng Yang, Yunpu Ma, and Alois C Knoll. Opendrivevla: Towards end-to-end autonomous driving with large vision language action model. *arXiv preprint arXiv:2503.23463*, 2025a.

[^85]: Zewei Zhou, Tianhui Cai, Seth Z Zhao, Yun Zhang, Zhiyu Huang, Bolei Zhou, and Jiaqi Ma. Autovla: A vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. *arXiv preprint arXiv:2506.13757*, 2025b.

[^86]: Sicheng Zuo, Wenzhao Zheng, Xiaoyong Han, Longchao Yang, Yong Pan, and Jiwen Lu. Quadricformer: Scene as superquadrics for 3d semantic occupancy prediction. *arXiv preprint arXiv:2506.10977*, 2025a.

[^87]: Sicheng Zuo, Wenzhao Zheng, Yuanhui Huang, Jie Zhou, and Jiwen Lu. Gaussianworld: Gaussian world model for streaming 3d occupancy prediction. In *CVPR*, pages 6772–6781, 2025b.
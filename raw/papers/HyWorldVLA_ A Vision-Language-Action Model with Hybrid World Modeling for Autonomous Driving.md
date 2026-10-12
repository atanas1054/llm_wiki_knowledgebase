---
title: "HyWorldVLA: A Vision-Language-Action Model with Hybrid World Modeling for Autonomous Driving"
source: "https://arxiv.org/html/2607.20988v1"
author:
published:
created: 2026-10-12
description:
tags:
  - "clippings"
---
Quanfu Yu    Xian Wu <sup>*</sup>    Hao Xu <sup>*</sup>    Liulong Ma <sup>†</sup> Affiliation: Automotive New Technology Research Institute, BYD Company Limited Affiliation: {yqf2018619, wuxian7x, xuhaovic}@gmail.com, ma.liulong@byd.com

###### Abstract

Vision-Language-Action (VLA) models augmented with world modeling represent a promising paradigm for end-to-end autonomous driving. While pixel-level future prediction enables fine-grained spatiotemporal reasoning, it compromises robustness in noisy driving scenarios. Conversely, latent-based world models alleviate this sensitivity but often incur limited interpretability and representational degradation due to absent pixel-level grounding. To reconcile this trade-off, we propose HyWorldVLA, a hybrid world-VLA framework that unifies pixel-level supervision and latent representation learning. In the pre-training stage, HyWorldVLA predicts video latents encoded by a pre-trained video VAE, while simultaneously reconstructing video frames to provide precise pixel-level grounding. During the subsequent co-fine-tuning phase, the model exclusively predicts latent features, which are fed into an action expert to generate trajectories. Extensive experiments on NAVSIM v1 and v2 benchmarks demonstrate that HyWorldVLA significantly outperforms both pixel-based and latent-based world model baselines. Notably, we present the first comprehensive qualitative and quantitative analysis of world model noise robustness in autonomous driving, establishing a new benchmark for evaluating future architectures.

## 1 Introduction

End-to-end autonomous driving has emerged as a dominant research direction for building intelligent vehicle systems, aiming to unify perception, reasoning, and control within a single holistic pipeline. As a core driving-centric paradigm, Vision-Language-Action (VLA) models [^38] [^16] [^23] [^28] integrate visual scene perception, language semantic understanding, and low-level action prediction, effectively leveraging large-scale driving datasets and generalized vision-language priors to learn robust driving behaviors. Compared with traditional modular autonomous driving systems that suffer from cascading errors and limited generalization on long-tailed and interactive traffic scenarios, VLA-based frameworks deliver more adaptive and generalizable decision-making capabilities, showing great potential for real-world vehicle deployment.

To further strengthen the spatiotemporal reasoning and scenario generalization of VLA models, recent advances augment standard driving VLA pipelines with world modeling capabilities, which learn to predict future scene states for forward-looking causal reasoning [^17] [^39] [^5]. Existing world modeling paradigms for driving VLAs fall into two dominant categories with inherent trade-offs. Pixel-based world models enforce supervision via full future frame reconstruction, providing dense geometric and physical constraints that yield strong generalization on occluded, interactive, and long-tailed driving scenarios, as shown in Figure 1 (a). However, pixel-level reconstruction is inherently sensitive to environmental variations ranging from rain and fog to fluctuating illumination, which fundamentally constrains the deployment robustness of pixel-based world models under complex driving conditions. In contrast, latent-based world VLAs improve scene noise robustness by forecasting future scene states purely in compact feature space, as shown in Figure 1 (b). Nevertheless, the absence of explicit pixel-level grounding results in representation degeneration and limited interpretability, which severely deteriorates model performance in complex real-world driving scenarios [^36].

To resolve this fundamental dilemma in the world modeling domain, we present HyWorldVLA, a novel hybrid world-VLA framework that unifies the strengths of pixel-grounded supervision and latent feature prediction to deliver both accuracy and robustness for end-to-end autonomous driving. Unlike existing purely pixel-based or latent-based world models, as shown in Figure 1 (c), HyWorldVLA employs a dual-supervision paradigm in the pre-training stage, jointly optimizing video VAE latent prediction and raw pixel frame reconstruction. During the subsequent co-fine-tuning phase, the model exclusively predicts latent features, which are fed into an action expert to generate driving trajectories. Under this design, pixel-level reconstruction serves as a rigorous structural regularizer during pre-training to refine latent embedding learning and prevent representation collapse. For the co-fine-tuning stage, action generation is conditioned on latent feature modeling that retains compact temporal representations, bolstering the scene noise robustness of the generated driving trajectory. Extensive experimental results demonstrate that HyWorldVLA successfully combines the high performance brought by pixel-based world models’ fine-grained spatiotemporal dynamic modeling with the scene noise robustness intrinsic to latent-based world models. Our contributions are summarized as follows:

- We propose HyWorldVLA, which establishes a hybrid world modeling paradigm that unifies the complementary characteristics of world action models and latent world models into a single framework.
- We introduce language guidance and latent conditioning mechanisms to effectively steer the model toward semantic information relevant to action generation.
- Extensive experiments on the NAVSIM benchmark demonstrate that our method achieves state-of-the-art performance, validating the effectiveness of our hybrid world modeling.

![[paradigm14.png|Refer to caption]]

Figure 1: Comparison of world model paradigms. (a) Pixel-based World Model: iteratively predicts actions and images, prone to scene noise; (b) Latent-based World Model: predicts latent states, weak in spatiotemporal dynamics modeling; (c) Hybrid World Model: our method simultaneously models iterative action-image prediction and future latent state prediction.

## 2 Related Work

Vision-Language-Action Models in Driving. With remarkable multimodal scene understanding and reasoning capabilities, Vision-Language Models (VLMs) are being extensively integrated into autonomous driving systems, promoting the development of end-to-end Vision-Language-Action (VLA) architectures [^13]. Early explorations including DriveMLM [^38] and DriveGPT4 [^44] directly leverage VLMs’ inherent text-generation capabilities to produce textual driving actions, thus fully utilizing the world knowledge from pre-trained VLMs [^16] [^34] [^18]. However, the fundamental gap between discrete language tokens and continuous control spaces introduces precision limits and breaks the end-to-end differentiability. Consequently, recent works augment VLM backbones with mechanisms that produce direct numeric outputs. ReCogDrive [^23] integrates an autoregressive model with a diffusion planner through a hierarchical cognitive pipeline, while AutoVLA [^57] unifies semantic reasoning and trajectory planning in a single autoregressive model with dual thinking modes. ORION [^11] and OpenDriveVLA [^56] further bridge the semantic-action gap by aggregating long-term context and performing hierarchical vision-language alignment.

World Models for Autonomous Driving. World models have been applied to autonomous driving [^54] [^12] [^17] [^39] [^22] [^5] [^52] due to their ability to anticipate future states and comprehend the temporal dynamics of the world. FSDrive [^49] generates unified future frames with explicit physical priors as visual spatiotemporal chain-of-thought. DriveVLA-W0 [^21] addresses the “supervision deficit” in VLAs by predicting future images as dense self-supervised signals, instantiated via both autoregressive and diffusion world models. OneVL [^28] routes latent reasoning through compact tokens supervised by dual auxiliary decoders, matching answer-only latency. Latent-CoT-Drive [^35] represents chain-of-thought reasoning in an action-aligned latent space using action-proposal and world model tokens. Pixel-level prediction is inherently susceptible to scene noise, whereas latent-level prediction struggles to capture the underlying causal relationships of the physical world. To address these complementary limitations, we propose a novel framework that bridges pixel-level and latent-level prediction.

## 3 Methodology

### 3.1 Problem Definition and Overall Framework

![[framework3.png|Refer to caption]]

Figure 2: Overview of HyWorldVLA. HyWorldVLA consists of three training stages. First, a video VAE model is trained under text guidance to compress and reconstruct video frames, thereby providing ground-truth targets for subsequent training. In the pre-training stage, sequences composed of text, images, and actions are used to train the world model for iterative prediction, where a learnable latent query Q is introduced to aggregate information and predict future latents. In the co-fine-tuning stage, the historical sequences together with are fed into the VLM to obtain global context and predicted latents, which are then passed through a joint attention mechanism to the action expert for action generation.

HyWorldVLA is formulated to predict future ego-vehicle waypoints conditioned on multi-modal observations. Specifically, at each time step $t$, the model receives a sequence of front-view camera images $\mathbf{V}_{t-H:t}=\left\{v_{t-H},\dots,v_{t}\right\}$, including the current and $H$ preceding frames, and historical vehicle waypoints $\mathbf{W}_{t-H:t}=\left\{w_{t-H},\dots,w_{t}\right\}$, where $w_{t}=\left(x_{t},y_{t},\theta_{t}\right)$ represents the position and heading angle at step $t$. Additionally, historical navigation commands $\mathbf{L}_{t-H:t}$ (e.g., Go Straight and Turn Right) are used to specify intended directions explicitly. Then based on these inputs, the model outputs a sequence of future waypoints $\mathbf{W}_{t+1:t+T}$ over a horizon of $T$ time steps.

As shown in Figure 2, the overall framework of HyWorldVLA consists of three main components and three training stages. (1) Video Encoder: Sec. 3.2 elaborates on a video encoder that compresses a sequence of video frames into a latent feature, providing a compact representation of future world dynamics; (2) VLM Backbone: Then the latent feature and video frames are jointly predicted by a VLM backbone, which is presented in Sec. 3.3; (3) Action Expert: Finally, in Sec. 3.4 we introduce the co-fine-tuning of the VLM model and a query-based action model, which predicts trajectories conditioning on the hidden states from each layer of the VLM backbone, the predicted future visual latents, and the ego‑vehicle’s historical motion.

### 3.2 Text-guided Latent Feature Learning

To encode future frames into compact latent features for VLM supervision, we adopt a video variational autoencoder [^43]. As shown in Figure 2, taking a video segment $\mathbf{v}\in\mathbb{R}^{C\times T\times H\times W}$ as input, the model first employs a spatial encoder to produce spatial latent $\mathbf{z}_{1}\in\mathbb{R}^{c\times T\times\frac{H}{8}\times\frac{W}{8}}$. Built upon a Stable Diffusion image VAE [^32], this spatial encoder stacks 3D convolutions to perform $8\times$ spatial downsampling while retaining full temporal length. Afterwards, a lightweight temporal autoencoder takes $\mathbf{z}_{1}$ as input and stacks 3D ResNet blocks to implement $4\times$ temporal compression, generating latent $\mathbf{z}_{2}\in\mathbb{R}^{c^{\prime}\times\frac{T}{4}\times\frac{H}{8}\times\frac{W}{8}}$ that eliminates redundant inter-frame static information. Then during reconstruction, the full-resolution video $\hat{\mathbf{v}}$ is recovered through a symmetric two-stage decoding pipeline.

To further enhance reconstruction quality, we embed multi-layer text cross-attention inside each block to introduce textual guidance during feature encoding. We split visual features into patch tokens, take visual patches as query and value, and adopt Flan-T5 [^6] text embeddings as keys to compute cross-attention. The attention output is added back to original visual features via residual connection. This textual semantic prior effectively suppresses motion ghosting, edge blurring and temporal flickering, greatly improving reconstruction fidelity for complex traffic scenes.

The VAE is optimized via a multi-component loss that balances reconstruction loss $\mathcal{L}_{\mathrm{rec}}$, adversarial loss $\mathcal{L}_{\mathrm{GAN}}$, and KL-divergence regularization loss $\mathcal{L}_{\mathrm{KL}}$:

$$
\mathcal{L}_{\mathrm{vae}}=\mathcal{L}_{\mathrm{rec}}+\lambda_{\mathrm{GAN}}\mathcal{L}_{\mathrm{GAN}}+\lambda_{\mathrm{KL}}\mathcal{L}_{\mathrm{KL}}.
$$

Through hierarchical spatial-temporal compression and text cross-modal injection, the video VAE generates a compact latent representation, providing effective supervision for downstream VLA pre-training and co-fine-tuning.

### 3.3 Pre-training

The pre-training stage aims to equip the backbone with the ability to predict the world evolution both on the image-level and latent-level.

Vision Modeling. We follow VQGAN [^9] to transform the continuous video stream into discrete visual tokens. Specifically, given a continuous video sequence $\mathbf{V}_{t-H:t+T}$, we first partition the sequence into $N$ non-overlapping chunks, where $N=(H+T)/t_{\text{chunk}}+1$ and $t_{\text{chunk}}$ denotes the chunk duration. For each chunk, we employ a VQ tokenizer to discretize the first visual observation into discrete visual tokens, yielding:

$$
V_{q}=\{V_{q}^{1},V_{q}^{2},\ldots,V_{q}^{N}\},
$$

where $V_{q}^{j}$ represents the quantized visual tokens corresponding to the initial frame of the $j$ -th chunk.

Action Modeling. We adopt FAST [^30] to discretize continuous action sequences into discrete action tokens. Given the waypoint sequence $\mathbf{W}_{t-H:t+T}$, we convert absolute coordinates to ego-relative coordinates to obtain the action sequence ${V}_{t-H:t+T}$, which is split into chunks and encodes ${V}_{t-H:t+T}$ into discrete tokens using the FAST action tokenizer, yielding:

$$
{A_{q}}=\{A_{q}^{1},A_{q}^{2},\ldots,A_{q}^{N}\},
$$

where each $A_{q}^{j}$ corresponds to the action tokens within the $j$ -th temporal chunk.

Language Modeling. We utilize the Emu3 tokenizer [^40] to discretize the driving command $\mathbf{CMD}_{t-H:t+T}$ into language tokens:

$$
{L_{q}}=\{L_{q}^{1},L_{q}^{2},\ldots,L_{q}^{N}\},
$$

where each $L_{q}^{j}$ corresponds to the $j$ -th language token.

Future Latent Prediction. We strategically insert a learnable query token $Q$ into the input token sequence, and the hidden state at the query position is fed to an MLP to predict the future latent representation $\mathbf{\hat{z}}_{2}$ during the forward pass.

Training Objective. Since all modality signals are transformed into discrete tokens drawn from a shared vocabulary, we construct the world model’s input sequence as:

$$
\mathcal{S}=[L_{q}^{1},V_{q}^{1},A_{q}^{1},L_{q}^{2},V_{q}^{2},Q,A_{q}^{2},L_{q}^{3},\ldots,V_{q}^{N},A_{q}^{N}],
$$

To prevent information leakage, causal masking is applied such that each token can only attend to preceding tokens in the sequence. The final training objective consists of three terms:

$$
\mathcal{L}_{\text{world}}=\sum_{j=1}^{N}\text{CE}(\hat{L}_{q}^{j},L_{q}^{j})+\sum_{j=1}^{N}\text{CE}(\hat{A}_{q}^{j},A_{q}^{j})+\lambda_{1}\sum_{j=1}^{N}\text{CE}(\hat{v}_{q}^{j},v_{q}^{j})+\lambda_{2}\|\mathbf{\hat{z}}_{2}-\mathbf{z}_{2}\|_{2}^{2}.
$$

Here, the first term is the language tokens prediction loss, the second term is the action prediction loss, the third term is the visual tokens prediction loss, and the last term enforces consistency between the predicted latent representation and the ground-truth compressed feature extracted from the future video sequence. $\lambda_{1}$ and $\lambda_{2}$ are the vision token and latent representation loss weights.

### 3.4 Co-Fine-Tuning

In the co‑fine‑tuning phase, the world model and the action expert are jointly optimized to achieve continuous and reliable action prediction.

We integrate the backbone and the action expert via a joint attention mechanism [^2]. The historical actions ${V}_{t-H:t}$, the driving commands $\mathbf{CMD}_{t-H:t}$ and the predicted latent representation $\mathbf{\hat{z}}_{2}$ from the backbone are concatenated as the sequence input to the attention module. After interacting with the backbone context, the hidden state at the action token positions is decoded via an MLP head.

To verify cross-paradigm compatibility, we instantiate our method on the two dominant action model families: selection-based [^24] and generative-model-based [^26].

For the generative-model training objectives, including flow matching, we have the following:

$$
\mathcal{L}_{\text{cotrain}}=|\hat{v}-v|_{2}^{2}+\lambda_{3}|\hat{z}-z|_{2}^{2}
$$

The first term is the flow-matching vector field regression loss, and the second term is the latent prediction loss, and $\lambda_{3}$ is the latent loss weight.

The selection-based training objectives are as follows:

$$
\mathcal{L}_{\text{cotrain}}=\mathrm{KL}\left(p_{\text{pred}}\;\middle\|\;p_{\text{gt}}\right)+\lambda_{3}\,\|\hat{z}-z\|_{2}^{2},
$$

where $p_{\text{pred}}$ is the predicted score for each trajectory in an offline trajectory vocabulary, $p_{\text{gt}}$ is the ground-truth score computed by comprehensively considering the expert trajectory as well as the metrics of the evaluation and a KL divergence loss is employed to supervise the predicted score distribution.

## 4 Experiments

### 4.1 Datasets

We evaluate our approach on the NAVSIM v1 [^8] and v2 [^4] benchmarks, providing closed-loop evaluation protocols and multi-dimensional quantitative metrics.

NAVSIM v1. NAVSIM v1 measures driving capability from five core dimensions: No at-fault Collision (NC), Drivable Area Compliance (DAC), Time-To-Collision (TTC), Comfort (C.), and Ego Progress (EP). The overall performance is aggregated into the Predictive Driver Model Score (PDMS) via a weighted multiplicative formulation:

$$
\text{PDMS}=\text{NC}\times\text{DAC}\times\frac{5\times\text{EP}+5\times\text{TTC}+2\times\text{C.}}{12}.
$$

NAVSIM v2. As an extended and refined version, NAVSIM v2 [^4] introduces more fine-grained evaluation dimensions to better align with real-world driving requirements. In addition to the core metrics inherited from v1, it adds Driving Direction Compliance (DDC), Traffic Light Compliance (TLC), Lane Keeping (LK), History Comfort (HC), and Extended Comfort (EC). The corresponding comprehensive metric is the Extended Predictive Driver Model Score (EPDMS), defined as:

$$
\text{EPDMS}=\text{NC}\times\text{DAC}\times\text{DDC}\times\text{TLC}\times\frac{5\times\text{EP}+5\times\text{TTC}+2\times\text{LK}+2\times\text{HC}+2\times\text{EC}}{16}.
$$

### 4.2 Implementation Details

Our model backbone is based on Emu3 [^40], a unified multimodal autoregressive model.

VideoVAEPlus Fine-tuning. We initialize from the 4-channel VideoVAEPlus [^43] pre-trained model and fine-tune it on the NuPlan dataset [^3]. For each video segment, we uniformly sample 8 frames resized to $216\times 216$ and use Qwen3.6-plus to generate a short scene description. The training is performed on 4 PPUs [^1] with a per-PPU batch size of 1. The weights of the adversarial loss and the KL loss are set to 0.5 and 1e-6 respectively, and the adversarial loss is enabled after 50K steps. We train the model for 100K steps, enabling it to learn compact representations from the videos.

Pre-training. During the pre-training stage, we utilize the OpenScenes dataset [^7] comprising over 120 hours of driving video recordings which are segmented into clips of 20.0 s. During training, for each clip, we randomly sample a frame sequence with a future horizon $T=4.0$ s and a history horizon $H=1.0$ s. Given the chunk duration $t_{\text{chunk}}=1.0$ s, the number of non-overlapping chunks num $N$ is 6. The training configuration adopts a batch size of 4 across two nodes, each equipped with 16 PPUs [^1]. We set the learning rate to $2.2\times 10^{-4}$ with a maximum of 4000 training steps. The loss weighting coefficients are configured as $\lambda_{1}=0.5$ and $\lambda_{2}=0.1$.

Co-fine-tuning. During the co-fine-tuning phase, we fine-tune the pre-trained model on the NAVSIM trainval dataset [^8] which comprises over 100,000 frames with ground-truth ego-vehicle trajectory annotations. Consistent with the pre-training configuration, the future horizon $T$ and historical horizon $H$ are set to 4 s and 1 s, respectively. We train for 4000 steps with a total batch size of 96. The learning rate is set to $5\times 10^{-5}$, with a latent prediction loss weight $\lambda_{3}=0.1$.

### 4.3 Comparison with State-of-the-Art Methods

We benchmark HyWorldVLA against representative pixel-based world model (WM) methods, latent-based world model methods, as well as several canonical end-to-end and VLA-based baselines. As shown in Table 1 and Table 2, our approach achieves state-of-the-art (SOTA) performance on both v1 and v2 benchmarks, significantly outperforming the competitive pixel-based WM baselines such as DriveVLA-W0 [^21] and the latent-based WM baselines including Latent-WAM [^37]. This performance gain can be ascribed to our hybrid world modeling paradigm, which reconciles the fine-grained spatiotemporal modeling capability of pixel-based world models with the intrinsic high-level semantic representation capacity of latent world models.

Table 1: Comparison with state-of-the-art methods on the NAVSIM v1. Abbreviations: 1x Cam (single front-view camera), Nx Cam (surround-view cameras), L (LiDAR).

<table><tbody><tr><td>Method</td><td>Ref</td><td>Sensors</td><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>C.<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>Human</td><td>-</td><td>-</td><td>100</td><td>100</td><td>100</td><td>99.9</td><td>87.5</td><td>94.8</td></tr><tr><td colspan="9">End-to-end-based Methods</td></tr><tr><td>UniAD <sup><a href="#fn:14">14</a></sup></td><td>CVPR’23</td><td>6x Cam</td><td>97.8</td><td>91.9</td><td>92.9</td><td>100.0</td><td>78.8</td><td>83.4</td></tr><tr><td>TransFuser <sup><a href="#fn:31">31</a></sup></td><td>TPAMI’23</td><td>3x Cam + L</td><td>97.7</td><td>92.8</td><td>92.8</td><td>100.0</td><td>79.2</td><td>84.0</td></tr><tr><td>PARA-Drive <sup><a href="#fn:41">41</a></sup></td><td>CVPR’24</td><td>6x Cam</td><td>97.9</td><td>92.4</td><td>93.0</td><td>99.8</td><td>79.3</td><td>84.0</td></tr><tr><td>Hydra-MDP <sup><a href="#fn:24">24</a></sup></td><td>arXiv’24</td><td>3x Cam + L</td><td>98.3</td><td>96.0</td><td>94.6</td><td>100.0</td><td>78.7</td><td>86.5</td></tr><tr><td>DiffusionDrive <sup><a href="#fn:25">25</a></sup></td><td>CVPR’25</td><td>3x Cam + L</td><td>98.2</td><td>96.2</td><td>94.7</td><td>100.0</td><td>82.2</td><td>88.1</td></tr><tr><td colspan="9">VLA-based Methods</td></tr><tr><td>AutoVLA <sup><a href="#fn:57">57</a></sup></td><td>NeurIPS’25</td><td>3x Cam</td><td>98.4</td><td>95.6</td><td>98.0</td><td>99.9</td><td>81.9</td><td>89.1</td></tr><tr><td>ReCogDrive-8B <sup><a href="#fn:23">23</a></sup></td><td>ICLR’26</td><td>3x Cam</td><td>97.8</td><td>97.7</td><td>94.9</td><td>100.0</td><td>86.3</td><td>90.5</td></tr><tr><td colspan="9">Pixel-based Word Model Methods</td></tr><tr><td>WoTE <sup><a href="#fn:22">22</a></sup></td><td>ICCV’25</td><td>3x Cam + L</td><td>98.5</td><td>96.8</td><td>94.4</td><td>99.9</td><td>81.9</td><td>88.3</td></tr><tr><td>FSDrive <sup><a href="#fn:50">50</a></sup></td><td>NeurIPS’25</td><td>1x Cam</td><td>98.2</td><td>93.8</td><td>93.3</td><td>99.9</td><td>80.1</td><td>85.1</td></tr><tr><td>ReSim <sup><a href="#fn:45">45</a></sup></td><td>NeurIPS’25</td><td>1x Cam</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>86.6</td></tr><tr><td>PWM <sup><a href="#fn:53">53</a></sup></td><td>NeurIPS’25</td><td>1x Cam</td><td>98.6</td><td>95.9</td><td>95.4</td><td>100.0</td><td>81.8</td><td>88.1</td></tr><tr><td>CoWorld-VLA <sup><a href="#fn:15">15</a></sup></td><td>arXiv’26</td><td>1x Cam</td><td>99.2</td><td>96.8</td><td>96.6</td><td>100.0</td><td>83.6</td><td>89.8</td></tr><tr><td>DriveLaW <sup><a href="#fn:42">42</a></sup></td><td>CVPR’26</td><td>1x Cam</td><td>99.0</td><td>97.1</td><td>96.7</td><td>100.0</td><td>81.3</td><td>89.1</td></tr><tr><td>DriveVLA-W0 <sup><a href="#fn:21">21</a></sup></td><td>ICLR’26</td><td>1x Cam</td><td>98.7</td><td>99.1</td><td>95.3</td><td>99.3</td><td>83.3</td><td>90.2</td></tr><tr><td colspan="9">Latent-based World Model Methods</td></tr><tr><td>LAW <sup><a href="#fn:20">20</a></sup></td><td>ICLR’25</td><td>1x Cam</td><td>96.4</td><td>95.4</td><td>88.7</td><td>99.9</td><td>81.7</td><td>84.6</td></tr><tr><td>World4Drive <sup><a href="#fn:55">55</a></sup></td><td>ICCV’25</td><td>1x Cam</td><td>97.4</td><td>94.3</td><td>92.8</td><td>100.0</td><td>79.9</td><td>85.1</td></tr><tr><td>Epona <sup><a href="#fn:52">52</a></sup></td><td>ICCV’25</td><td>1x Cam</td><td>97.9</td><td>95.1</td><td>93.8</td><td>99.9</td><td>80.4</td><td>86.2</td></tr><tr><td>DreamerAD <sup><a href="#fn:47">47</a></sup></td><td>arXiv’26</td><td>1x Cam</td><td>98.0</td><td>97.2</td><td>94.3</td><td>100.0</td><td>83.1</td><td>88.7</td></tr><tr><td>OneVL <sup><a href="#fn:29">29</a></sup></td><td>arXiv’26</td><td>1x Cam</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>88.8</td></tr><tr><td>WorldRFT <sup><a href="#fn:46">46</a></sup></td><td>AAAI’26</td><td>1x Cam</td><td>97.8</td><td>96.8</td><td>94.0</td><td>100.0</td><td>81.7</td><td>87.8</td></tr><tr><td>ResWorld <sup><a href="#fn:51">51</a></sup></td><td>ICLR’26</td><td>4x Cam + L</td><td>98.9</td><td>96.5</td><td>95.6</td><td>100.0</td><td>83.1</td><td>89.0</td></tr><tr><td>Ours</td><td>-</td><td>1x Cam</td><td>98.6</td><td>98.9</td><td>95.1</td><td>99.3</td><td>84.6</td><td>90.59</td></tr></tbody></table>

Table 2: Comparison with state-of-the-art methods on the NAVSIM v2 with extended metrics.

<table><tbody><tr><td>Method</td><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DDC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TLC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>LK <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>HC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EPDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td colspan="9">End-to-end-based Methods</td><td></td><td></td></tr><tr><td>TransFuser <sup><a href="#fn:31">31</a></sup></td><td>96.9</td><td>89.9</td><td>97.8</td><td>99.7</td><td>87.1</td><td>95.4</td><td>92.4</td><td>98.3</td><td>87.2</td><td>76.7</td></tr><tr><td>HydraMDP++ <sup><a href="#fn:19">19</a></sup></td><td>97.2</td><td>97.5</td><td>99.4</td><td>99.6</td><td>83.1</td><td>96.5</td><td>94.4</td><td>98.2</td><td>70.9</td><td>81.4</td></tr><tr><td>DriveSuprim <sup><a href="#fn:48">48</a></sup></td><td>97.5</td><td>96.5</td><td>99.4</td><td>99.6</td><td>88.4</td><td>96.6</td><td>95.5</td><td>98.3</td><td>77.0</td><td>83.1</td></tr><tr><td>ARTEMIS <sup><a href="#fn:10">10</a></sup></td><td>98.3</td><td>95.1</td><td>98.6</td><td>99.8</td><td>81.5</td><td>97.4</td><td>96.5</td><td>98.3</td><td>-</td><td>83.1</td></tr><tr><td>DiffusionDrive <sup><a href="#fn:25">25</a></sup></td><td>98.2</td><td>95.9</td><td>99.4</td><td>99.8</td><td>87.5</td><td>97.3</td><td>96.8</td><td>98.3</td><td>87.7</td><td>84.5</td></tr><tr><td colspan="9">VLA-based Methods</td><td></td><td></td></tr><tr><td>ReCogDrive <sup><a href="#fn:23">23</a></sup></td><td>98.3</td><td>95.2</td><td>98.3</td><td>99.8</td><td>87.1</td><td>97.5</td><td>96.6</td><td>99.5</td><td>86.5</td><td>83.6</td></tr><tr><td colspan="9">Pixel-based Word Model Methods</td><td></td><td></td></tr><tr><td>DriveVLA-W0 <sup><a href="#fn:21">21</a></sup></td><td>98.5</td><td>99.1</td><td>98.0</td><td>99.7</td><td>86.4</td><td>98.1</td><td>93.2</td><td>97.9</td><td>58.9</td><td>86.1</td></tr><tr><td>ExploreVLA <sup><a href="#fn:33">33</a></sup></td><td>98.8</td><td>96.2</td><td>99.6</td><td>99.8</td><td>87.1</td><td>98.2</td><td>97.8</td><td>98.3</td><td>86.8</td><td>88.8</td></tr><tr><td colspan="9">Latent-based World Model Methods</td><td></td><td></td></tr><tr><td>DriveWorld-VLA <sup><a href="#fn:27">27</a></sup></td><td>98.6</td><td>99.1</td><td>99.6</td><td>99.8</td><td>87.4</td><td>97.9</td><td>97.0</td><td>97.9</td><td>78.6</td><td>86.8</td></tr><tr><td>DreamerAD <sup><a href="#fn:47">47</a></sup></td><td>98.6</td><td>99.1</td><td>99.6</td><td>99.8</td><td>87.4</td><td>97.9</td><td>97.0</td><td>97.9</td><td>78.6</td><td>86.8</td></tr><tr><td>Latent-WAM <sup><a href="#fn:37">37</a></sup></td><td>98.0</td><td>97.2</td><td>99.5</td><td>99.8</td><td>87.8</td><td>97.4</td><td>97.5</td><td>98.3</td><td>72.4</td><td>87.7</td></tr><tr><td>Ours</td><td>98.8</td><td>98.2</td><td>99.5</td><td>99.9</td><td>87.8</td><td>98.1</td><td>96.2</td><td>98.3</td><td>77.4</td><td>89.71</td></tr></tbody></table>

### 4.4 Ablation Analysis

Ablation on Components. Table 3 presents the ablation results for each component of HyWorldVLA. Removing the pixel-regression ability (Pure LWM) yields the lowest score (87.50), confirming the crucial role of its spatiotemporal modeling capability in understanding the environment and generating accurate actions. Removing the latent representational capacity (Pure WAM) also leads to a significant drop (89.91), indicating its positive effect on learning high-level semantics. Ablating the language guidance from the latent module decreases the score to 90.35, demonstrating that language effectively guides the model toward planning-relevant semantics. Removing the latent conditioning in the action expert model drops the score to 90.29, suggesting that explicitly leveraging predicted future information benefits more accurate action generation. Eliminating the latent supervision during co-fine-tuning results in a score of 90.17, highlighting the importance of preserving pre-trained latent semantics and preventing over-optimization.

Ablation on Loss Weights in Pre-training. As shown in Table 5, We first fix $\lambda_{2}=0.1$ and examine the effect of $\lambda_{1}$. When $\lambda_{1}$ increases from 0.1 to 0.5 and 1.0, the score improves from 90.48 to 90.59 and drops significantly to 89.83. This indicates that the weight for visual tokens prediction should not be too high. When we fix $\lambda_{1}=0.5$ and increase $\lambda_{2}$ from 0.1 to 0.2, 0.5 and 1.0, the score decreases from 90.59 to 90.47 and then continues to drop to 90.34 and 90.16, respectively. This demonstrates that simultaneously introducing visual tokens prediction ($\lambda_{1}$ = 0.5) and latent motion ($\lambda_{2}$ = 0.1) during the pre-training phase most effectively guides the trajectory.

Table 3: Ablation study on the NAVSIM benchmark.

| Config | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | C.$\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| Pure WAM | 98.3 | 98.6 | 94.5 | 99.6 | 84.0 | 89.91 |
| Pure LWM | 97.2 | 97.8 | 92.0 | 98.9 | 82.4 | 87.50 |
| w/o language guidance in latent | 98.5 | 98.8 | 94.7 | 99.6 | 84.5 | 90.35 |
| w/o latent condition in action expert | 98.6 | 98.7 | 94.9 | 99.3 | 84.3 | 90.29 |
| w/o latent supervision in co-fine-tuning | 98.6 | 98.9 | 95.3 | 99.8 | 83.3 | 90.17 |
| Ours | 98.6 | 98.9 | 95.1 | 99.3 | 84.6 | 90.59 |

Ablation on Loss Weights in co-fine-tuning. As shown in Table 5, starting from $\lambda_{3}=0.05$, we observe that increasing it to 0.1 raises the score from 90.47 to 90.59, indicating that appropriate supervision prevents performance degradation caused by semantic collapse of the latent space. However, when $\lambda_{3}$ is further increased to 0.2 and 1.0, the score drops to 90.01 and 89.75, respectively, suggesting that overly strong supervision hinders the model from autonomously learning the latent semantics most relevant to action generation.

Table 4: Ablation study of the loss weights in pre-training stage.

| $\lambda_{1}$ | $\lambda_{2}$ | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | C.$\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 0.1 | 98.6 | 98.7 | 94.9 | 99.6 | 84.6 | 90.48 |
| 0.5 | 0.1 | 98.6 | 98.9 | 95.1 | 99.3 | 84.6 | 90.59 |
| 1.0 | 0.1 | 98.5 | 98.5 | 94.9 | 99.6 | 83.5 | 89.83 |
| 0.5 | 0.2 | 98.6 | 98.9 | 95.0 | 99.5 | 84.4 | 90.47 |
| 0.5 | 0.5 | 98.6 | 98.6 | 94.5 | 99.4 | 84.9 | 90.34 |
| 0.5 | 1.0 | 98.6 | 98.6 | 94.7 | 99.5 | 84.4 | 90.16 |

Table 5: Ablation study of the loss weights in co-fine-tuning stage.

| $\lambda_{3}$ | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | C.$\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 98.5 | 98.8 | 95.0 | 99.3 | 84.6 | 90.47 |
| 0.1 | 98.6 | 98.9 | 95.1 | 99.3 | 84.6 | 90.59 |
| 0.2 | 98.6 | 98.7 | 94.6 | 99.4 | 83.9 | 90.01 |
| 1.0 | 98.6 | 98.3 | 95.0 | 99.4 | 83.4 | 89.75 |

### 4.5 Scene Noise Robustness Analysis

Quantitative Analysis. To further analyze the behavior of HyWorldVLA under the noisy conditions, a noise robustness test set was built by collecting 655 cases from the Openscene test dataset [^7], all of which contain non-uniform noise due to rain and fog. Several methods and configurations are compared in terms of performance: WoTE [^22], DriveLaW [^42], DriveVLA-W0 [^21], HyWorldVLA(Pure WM), HyWorldVLA(w/o latent supervision) and HyWorldVLA. As shown in Table 6, HyWorldVLA achieves the highest performance of 86.87 on corrupted cases, while WoTE [^22], DriveLaW [^42] and DriveVLA-W0 [^21] obtain only 60.65, 67.49 and 61.18, respectively, far below that of HyWorldVLA, demonstrating the superior robustness of our method. Pure WAM and w/o latent supervision in co-fine-tuning configurations obtain 69.95 and 73.18, which further clarifies that latent representation design of the HyWorldVLA yields this robust performance in noisy scenarios.

Table 6: Scene noise robustness comparison with other methods on the NAVSIM benchmark.

| Method/Config | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | C.$\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| WoTE [^22] | 98.9 | 76.0 | 64.4 | 100.0 | 64.4 | 60.65 |
| DriveLaW [^42] | 99.4 | 75.7 | 82.9 | 100.0 | 69.5 | 67.49 |
| DriveVLA-W0 [^21] | 93.4 | 76.2 | 83.1 | 98.6 | 59.8 | 61.18 |
| Pure WAM | 99.1 | 75.7 | 94.3 | 99.8 | 67.8 | 69.95 |
| w/o latent supervision in co-fine-tuning | 99.2 | 78.5 | 94.8 | 99.7 | 70.9 | 73.18 |
| Ours | 99.7 | 91.1 | 96.9 | 99.2 | 84.4 | 86.87 |

Qualitative Analysis. For a qualitative robustness assessment,we systematically modulate front-view illumination, transitioning from sunny to overcast conditions. As illustrated in Fig. 3(a), HyWorldVLA maintains consistent behavioral patterns largely invariant to lighting variations, whereas DriveVLA-W0 exhibits significant instability. Furthermore, Fig. 3(b) showcases two additional cases drawn from our proposed noise robustness test set, featuring non-uniform visual degradations. Under these challenging conditions, HyWorldVLA continues to execute correct driving maneuvers, while DriveVLA-W0 fails to respond appropriately. Additional cases are in Appendix C (Figs. 11-13).

![[bbb39a18d68954f6.jpg|Refer to caption]]

(a) Turn right under the fluctuating illumination.

## 5 Conclusion

We present a hybrid autonomous driving world model that combines the fine-grained spatiotemporal modeling of pixel-based methods with the noise robustness of latent variants. Pixel-level reconstruction preserves detailed spatial and dynamic information, while latent prediction in the VAE space inherently improves noise resilience. Experiments on NAVSIM v1/v2 show our method achieves state-of-the-art performance, with further validation on rain/fog-corrupted inputs. Limitations and future work include extending the monocular-based framework to multi-modal sensors and strengthening latent dynamics coupling with critical traffic elements.

## References

## Appendix

## Appendix A Overall

This appendix contains supplementary materials that accompany the main paper. § B reports additional experiments, which include the comparison of the flow matching action expert on the NAVSIM benchmark, as well as an ablation study of the latent loss weights for the scene noise robustness. Further qualitative findings are presented in § C, images produced by our world model, featuring visualizations of challenging case studies and the qualitative analyses on illumination interference and blur noise cases. Finally, § D discloses our use of LLMs for writing assistance.

## Appendix B More Experiments

### B.1 Flow Matching Based Experiments

In order to verify the versatility of our approach on different types of action models, in addition to the anchor selective action model method, we also conduct comparative experiments with the flow‑matching action model as shown in Table 7.

Table 7: Flow matching as action expert version comparison on the NAVSIM benchmark.

| Method | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | C.$\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| DriveVLA-W0 (fm) | 98.4 | 95.3 | 95.2 | 100.0 | 80.9 | 87.2 |
| Ours (fm) | 98.8 | 96.3 | 95.5 | 100.0 | 82.9 | 88.76 |

### B.2 Ablation Study For The Scene Noise Robustness

To further clarify how latent representation contributes to noise robustness, we perform an ablation study on $\lambda_{3}$. Table 8 shows that larger $\lambda_{3}$ consistently boosts model performance on noisy data, validating the beneficial role of latent representation in noise robustness. Nevertheless, unbounded increase of $\lambda_{3}$ degrades performance on normal cases, making the trade-off between these two capabilities a critical future research direction.

Table 8: Ablation study of the latent loss weights on the noise robustness test.

| $\lambda_{3}$ | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | C.$\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 99.5 | 88.7 | 96.6 | 99.1 | 82.4 | 84.52 |
| 0.1 | 99.7 | 91.1 | 96.9 | 99.2 | 84.4 | 86.87 $\uparrow\!$+2.35 |
| 0.2 | 99.1 | 94.8 | 94.5 | 99.4 | 88.1 | 89.45 $\uparrow\!$+2.58 |
| 1.0 | 98.8 | 98.3 | 95.1 | 100.0 | 90.8 | 92.94 $\uparrow\!$+3.49 |

## Appendix C Visualization

### C.1 Future Image Generation

Our model inherits the strong future frame prediction capability of the World Action Model. As shown in Figure 4(a), with only a single front-view camera, our model generates the central double yellow line absent from the input frames. Moreover, our model accurately forecasts future maneuvers of the ego vehicle and obstacles in the highly interactive scenario, as shown in Figure 4(b).

![[frame_000.png|Refer to caption]]

(a) Right turn in the intersection with a limited front camera view.

### C.2 Planning Trajectory

In this section, we conduct a qualitative case analysis to gain deeper insights into our model’s behavior and validate our key findings. Specifically, we compare our Hybrid World Model against the baseline DriveVLA-W0 across multiple challenging scenarios to demonstrate, as shown in Figure 5, Figure 6 and Figure 7.

![[cde33de8ff605ef3.jpg|Refer to caption]]

Figure 5: Turn right and stop behind a truck with its trunk open. DriveVLA-W0 baseline fails in trying to nudge a stopping truck with its trunk open. Our method stops and keep a fit distance to it.

![[7df6e45207a55000.jpg|Refer to caption]]

Figure 6: Left turn in a small intersection with the limited view. DriveVLA-W0 baseline goes straight at a small intersection where it is supposed to turn left. Our method turns left even though the front view is limited.

![[4608bd6ac0075660.jpg|Refer to caption]]

Figure 7: Turn left at the construction intersection. DriveVLA-W0 baseline goes into the opposite lane at the construction intersection. Our method turns left and keep in the right lane.

### C.3 Failure Case

To provide an unbiased account of our model’s performance and delineate paths for subsequent enhancement, representative failure cases encountered in evaluation are analyzed. They fall into two main types: deficits in capturing critical small elements, and constraints of front-view cameras under turning scenarios, as shown in Figure 8, Figure 9 and Figure 10.

![[15e3d7e5d5b85054.jpg|Refer to caption]]

Figure 8: Failure case analysis: deficits in capturing critical small elements. We illustrate a failure case in which our model attempts to nudge a queuing vehicle ahead. The world model’s predictions reveal that the model misclassifies the queuing vehicle as low-speed moving and tries to nudge past it for higher traffic efficiency, failing to recognize the queuing state indicated by the red light ahead and the vehicle’s brake lights. Since this critical cue is only discernible from a few pixels, fine-grained element capture and causal reasoning remain a core future research direction.

![[28ac1020ddd052fc.jpg|Refer to caption]]

Figure 9: Failure case analysis: constraints of front-view cameras under left turn. We present a failure case in which our model crosses the central double yellow line during left turns. Limited by field of view, historical and current frames fail to deliver valid cues for the target lane in the left-turn process; the world model’s predictions correspondingly lack the double road marking. This demonstrates the importance of multi-view camera inputs for turning scenarios, while trading off camera count against inference latency remains a key unresolved issue.

![[b027c842bd22562f.jpg|Refer to caption]]

Figure 10: Failure case analysis: constraints of front-view cameras under right turn. We present a failure case in which our model rear-ends a queuing leading vehicle during right turns. Limited by field of view, valid cues of surrounding vehicles are unavailable in the right-turn process, and the world model’s predictions correspondingly miss the preceding vehicle. This again demonstrates the necessity of multi-view camera inputs for turning scenarios.

### C.4 More Scene Noise Robustness Case

To provide a balanced view of our model’s scene noise robustness, we perform qualitative analyses on illumination interference and blur noise cases. Our model exhibits more stable and reasonable performance under noise than the baseline. Besides the left-turn scenario in the main text, we compare output variations in typical right-turn and straight-ahead scenarios by adjusting illumination intensity alone, verifying that our model yields more consistent outputs under illumination perturbations, as shown in Figure 11(a) and Figure 11(b). We further present cases with natural non-uniform blur from rain and fog, where traffic lights and vehicles are partially or fully occluded. The pure pixel-regression baseline shows severe degradation: it crawls during red-light queuing and acts overly conservative to slow down. By contrast, benefiting from the LWM’s inherent scene noise robustness, our model maintains accurate and reasonable driving behaviors under the same noisy inputs, with trajectories more aligned with human driving patterns, as shown in Figure 12 and Figure 13.

![[94a17417d3dd5a64.jpg|Refer to caption]]

(a) Turn right in the intersection.

![[3806bb68d08e58fc.jpg|Refer to caption]]

Figure 12: Remain stationary at red-light intersection under the blur noise. DriveVLA-W0 baseline crawls during red-light queuing, while our method remains stationary.

![[6dcfab3569235da4.jpg|Refer to caption]]

Figure 13: Go straight under the blur noise. DriveVLA-W0 baseline try to slow down due to the blurry image. Our method maintains a reasonable speed and is more aligned with human driving patterns.

[^1]: Alibaba Cloud PPU introduction. Note: \[Online\] [https://help.aliyun.com/zh/document\_detail/2864586.html](https://help.aliyun.com/zh/document_detail/2864586.html) Cited by: §4.2, §4.2.

[^2]: K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, et al. $\pi_{0}$: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164. Cited by: §3.4.

[^3]: H. Caesar, J. Kabzan, K. S. Tan, W. K. Fong, E. Wolff, A. Lang, L. Fletcher, O. Beijbom, and S. Omari Nuplan: a closed-loop ml-based planning benchmark for autonomous vehicles. arXiv preprint arXiv:2106.11810. Cited by: §4.2.

[^4]: W. Cao, M. Hallgarten, T. Li, D. Dauner, X. Gu, C. Wang, Y. Miron, M. Aiello, H. Li, I. Gilitschenski, et al. Pseudo-simulation for autonomous driving. arXiv preprint arXiv:2506.04218. Cited by: §4.1, §4.1.

[^5]: Y. Chen, Y. Wang, and Z. Zhang Drivinggpt: unifying driving world modeling and planning with multi-modal autoregressive transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 26890–26900. Cited by: §1, §2.

[^6]: H. W. Chung, L. Hou, S. Longpre, B. Zoph, Y. Tay, W. Fedus, Y. Li, X. Wang, M. Dehghani, S. Brahma, et al. Scaling instruction-finetuned language models. Journal of Machine Learning Research 25 (70), pp. 1–53. Cited by: §3.2.

[^7]: O. Contributors OpenScene: the largest up-to-date 3d occupancy prediction benchmark in autonomous driving. Note: [https://github.com/OpenDriveLab/OpenScene](https://github.com/OpenDriveLab/OpenScene) Cited by: §4.2, §4.5.

[^8]: D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, et al. Navsim: data-driven non-reactive autonomous vehicle simulation and benchmarking. arXiv preprint arXiv:2406.15349. Cited by: §4.1, §4.2.

[^9]: P. Esser, R. Rombach, and B. Ommer Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 12873–12883. Cited by: §3.3.

[^10]: R. Feng, N. Xi, D. Chu, R. Wang, Z. Deng, A. Wang, L. Lu, J. Wang, and Y. Huang Artemis: autoregressive end-to-end trajectory planning with mixture of experts for autonomous driving. IEEE Robotics and Automation Letters 11 (1), pp. 226–233. Cited by: Table 2.

[^11]: H. Fu, D. Zhang, Z. Zhao, J. Cui, D. Liang, C. Zhang, D. Zhang, H. Xie, B. Wang, and X. Bai ORION: a holistic end-to-end autonomous driving framework by vision-language instructed action generation. arXiv preprint arXiv:2503.19755. Cited by: §2.

[^12]: A. Hu, L. Russell, H. Yeo, Z. Murez, G. Fedoseev, A. Kendall, J. Shotton, and G. Corrado Gaia-1: a generative world model for autonomous driving. arXiv preprint arXiv:2309.17080. Cited by: §2.

[^13]: T. Hu, X. Liu, S. Wang, Y. Zhu, A. Liang, L. Kong, G. Zhao, Z. Gong, J. Cen, Z. Huang, et al. Vision-language-action models for autonomous driving: past, present, and future. arXiv preprint arXiv:2512.16760. Cited by: §2.

[^14]: Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, et al. Planning-oriented autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 17853–17862. Cited by: Table 1.

[^15]: M. Huang, Y. Xiang, Z. Liang, J. Huang, J. Wang, Z. Xu, F. Tan, H. Zhou, M. Yang, and G. Che CoWorld-vla: thinking in a multi-expert world model for autonomous driving. arXiv preprint arXiv:2605.10426. Cited by: Table 1.

[^16]: J. Hwang, R. Xu, H. Lin, W. Hung, J. Ji, K. Choi, D. Huang, T. He, P. Covington, B. Sapp, et al. EMMA: end-to-end multimodal model for autonomous driving. arXiv preprint arXiv:2410.23262. Cited by: §1, §2.

[^17]: F. Jia, W. Mao, Y. Liu, Y. Zhao, Y. Wen, C. Zhang, X. Zhang, and T. Wang Adriver-i: a general world model for autonomous driving. arXiv preprint arXiv:2311.13549. Cited by: §1, §2.

[^18]: B. Jiang, S. Chen, Q. Zhang, W. Liu, and X. Wang AlphaDrive: unleashing the power of VLMs in autonomous driving via reinforcement learning and reasoning. arXiv preprint arXiv:2503.07608. Cited by: §2.

[^19]: K. Li, Z. Li, S. Lan, J. Liu, Y. Xie, Z. Wu, Z. Yu, J. M. Alvarez, et al. Hydra-mdp++: advancing end-to-end driving via hydra-distillation with expert-guided decision analysis. Cited by: Table 2.

[^20]: Y. Li, L. Fan, J. He, Y. Wang, Y. Chen, Z. Zhang, and T. Tan Enhancing end-to-end autonomous driving with latent world model. In International Conference on Learning Representations, Vol. 2025, pp. 42942–42959. Cited by: Table 1.

[^21]: Y. Li, S. Shang, W. Liu, B. Zhan, H. Wang, Y. Wang, Y. Chen, X. Wang, Y. An, C. Tang, L. Hou, L. Fan, and Z. Zhang DriveVLA-w0: world models amplify data scaling law in autonomous driving. arXiv preprint arXiv:2510.12796. Cited by: §2, §4.3, §4.5, Table 1, Table 2, Table 6.

[^22]: Y. Li, Y. Wang, Y. Liu, J. He, L. Fan, and Z. Zhang End-to-end driving with online trajectory evaluation via bev world model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 27137–27146. Cited by: §2, §4.5, Table 1, Table 6.

[^23]: Y. Li, K. Xiong, X. Guo, F. Li, S. Yan, G. Xu, L. Zhou, L. Chen, H. Sun, B. Wang, K. Ma, G. Chen, H. Ye, W. Liu, and X. Wang ReCogDrive: a reinforced cognitive framework for end-to-end autonomous driving. arXiv preprint arXiv:2506.08052. Cited by: §1, §2, Table 1, Table 2.

[^24]: Z. Li, K. Li, S. Wang, S. Lan, Z. Yu, Y. Ji, Z. Li, Z. Zhu, J. Kautz, Z. Wu, et al. Hydra-mdp: end-to-end multimodal planning with multi-target hydra-distillation. arXiv preprint arXiv:2406.06978. Cited by: §3.4, Table 1.

[^25]: B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang, et al. Diffusiondrive: truncated diffusion model for end-to-end autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 12037–12047. Cited by: Table 1, Table 2.

[^26]: Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le Flow matching for generative modeling. arXiv preprint arXiv:2210.02747. Cited by: §3.4.

[^27]: L. Liu, Z. Song, C. Jia, H. Ye, X. Hao, L. Chen, et al. DriveWorld-vla: unified latent-space world modeling with vision-language-action for autonomous driving. arXiv preprint arXiv:2602.06521. Cited by: Table 2.

[^28]: J. Lu, J. Guan, Z. Huang, J. Li, G. Li, L. Kong, Y. Li, H. Wang, S. Xu, Y. Luo, F. Li, C. Dang, J. Wang, T. Xu, J. Wu, J. Wu, X. Hao, W. Zhang, T. Jiang, L. Zhang, L. Zhou, Y. Tang, J. Wang, Y. Gao, F. Jia, L. Liu, Y. Ge, H. Li, Y. Shen, J. Cui, H. Xie, and B. Wang Xiaomi onevl: one-step latent reasoning and planning with vision-language explanation. arXiv preprint arXiv:2604.18486. Cited by: §1, §2.

[^29]: J. Lu, J. Guan, Z. Huang, J. Li, G. Li, L. Kong, Y. Li, H. Wang, S. Xu, Y. Luo, et al. Xiaomi onevl: one-step latent reasoning and planning with vision-language explanation. arXiv preprint arXiv:2604.18486. Cited by: Table 1.

[^30]: K. Pertsch, K. Stachowicz, B. Ichter, D. Driess, S. Nair, Q. Vuong, O. Mees, C. Finn, and S. Levine FAST: efficient action tokenization for vision-language-action models. arXiv preprint arXiv:2501.09747. Cited by: §3.3.

[^31]: A. Prakash, K. Chitta, and A. Geiger Multi-modal fusion transformer for end-to-end autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 7077–7087. Cited by: Table 1, Table 2.

[^32]: R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695. Cited by: §3.2.

[^33]: Z. Sheng, X. Ye, J. Luo, S. Chen, and L. Ren ExploreVLA: dense world modeling and exploration for end-to-end autonomous driving. arXiv preprint arXiv:2604.02714. Cited by: Table 2.

[^34]: C. Sima, K. Renz, K. Chitta, L. Chen, H. Zhang, C. Xie, J. Beißwenger, P. Luo, A. Geiger, and H. Li DriveLM: driving with graph visual question answering. In European conference on computer vision, pp. 256–274. Cited by: §2.

[^35]: S. Tan, K. Chitta, Y. Chen, R. Tian, Y. You, Y. Wang, W. Luo, Y. Cao, P. Krahenbuhl, M. Pavone, and B. Ivanovic Latent chain-of-thought world modeling for end-to-end driving. arXiv preprint arXiv:2512.10226. Note: Accepted to CVPR 2026 Cited by: §2.

[^36]: S. Tu, X. Zhou, D. Liang, X. Jiang, Y. Zhang, X. Li, and X. Bai The role of world models in shaping autonomous driving: a comprehensive survey. arXiv preprint arXiv:2502.10498. Cited by: §1.

[^37]: L. Wang, Y. Zheng, Q. Chen, S. Li, Y. Zhang, Z. Xing, Q. Zhang, X. Li, D. Qian, P. Yang, et al. Latent-wam: latent world action modeling for end-to-end autonomous driving. arXiv preprint arXiv:2603.24581. Cited by: §4.3, Table 2.

[^38]: W. Wang, J. Xie, C. Hu, H. Zou, J. Fan, W. Tong, Y. Wen, S. Wu, H. Deng, Z. Li, et al. DriveMLM: aligning multi-modal large language models with behavioral planning states for autonomous driving. arXiv preprint arXiv:2312.09245. Cited by: §1, §2.

[^39]: X. Wang, Z. Zhu, G. Huang, X. Chen, J. Zhu, and J. Lu Drivedreamer: towards real-world-drive world models for autonomous driving. In European conference on computer vision, pp. 55–72. Cited by: §1, §2.

[^40]: X. Wang, X. Zhang, Z. Luo, Q. Sun, Y. Cui, J. Wang, F. Zhang, Y. Wang, Z. Li, Q. Yu, et al. Emu3: next-token prediction is all you need. arXiv preprint arXiv:2409.18869. Cited by: §3.3, §4.2.

[^41]: X. Weng, B. Ivanovic, Y. Wang, Y. Wang, and M. Pavone Para-drive: parallelized architecture for real-time autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15449–15458. Cited by: Table 1.

[^42]: T. Xia, Y. Li, L. Zhou, J. Yao, K. Xiong, H. Sun, B. Wang, K. Ma, G. Chen, H. Ye, et al. Drivelaw: unifying planning and video generation in a latent driving world. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 39701–39712. Cited by: §4.5, Table 1, Table 6.

[^43]: Y. Xing, Y. Fei, Y. He, J. Chen, J. Xie, X. Chi, and Q. Chen Large motion video autoencoding with cross-modal video vae. arXiv preprint arXiv:2412.17805. Cited by: §3.2, §4.2.

[^44]: Z. Xu, Y. Zhang, E. Xie, Z. Zhao, Y. Guo, K. K. Wong, Z. Li, and H. Zhao DriveGPT4: interpretable end-to-end autonomous driving via large language model. IEEE Robotics and Automation Letters 9 (10), pp. 8186–8193. Cited by: §2.

[^45]: J. Yang, K. Chitta, S. Gao, L. Chen, Y. Shao, X. Jia, H. Li, A. Geiger, X. Yue, and L. Chen Resim: reliable world simulation for autonomous driving. Advances in Neural Information Processing Systems 38, pp. 167710–167741. Cited by: Table 1.

[^46]: P. Yang, B. Lu, Z. Xia, C. Han, Y. Gao, T. Zhang, K. Zhan, X. Lang, Y. Zheng, and Q. Zhang Worldrft: latent world model planning with reinforcement fine-tuning for autonomous driving. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40, pp. 11649–11657. Cited by: Table 1.

[^47]: P. Yang, Y. Zheng, D. Qian, Z. Xing, Q. Zhang, L. Wang, Y. Zhang, S. Guo, Z. Xia, Q. Chen, et al. Dreamerad: efficient reinforcement learning via latent world model for autonomous driving. arXiv preprint arXiv:2603.24587. Cited by: Table 1, Table 2.

[^48]: W. Yao, Z. Li, S. Lan, Z. Wang, X. Sun, J. M. Alvarez, and Z. Wu Drivesuprim: towards precise trajectory selection for end-to-end planning. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40, pp. 11910–11918. Cited by: Table 2.

[^49]: S. Zeng, X. Chang, M. Xie, X. Liu, Y. Bai, Z. Pan, M. Xu, X. Wei, and N. Guo FutureSightDrive: thinking visually with spatio-temporal cot for autonomous driving. arXiv preprint arXiv:2505.17685. Note: Accepted to NeurIPS 2025 as Spotlight Cited by: §2.

[^50]: S. Zeng, X. Chang, M. Xie, X. Liu, Y. Bai, Z. Pan, M. Xu, and X. Wei Futuresightdrive: thinking visually with spatio-temporal cot for autonomous driving. Advances in Neural Information Processing Systems 38, pp. 67299–67318. Cited by: Table 1.

[^51]: J. Zhang, Z. Fu, Z. Xu, W. Dai, Q. Liu, and Y. Wang Resworld: temporal residual world model for end-to-end autonomous driving. arXiv preprint arXiv:2602.10884. Cited by: Table 1.

[^52]: K. Zhang, Z. Tang, X. Hu, X. Pan, X. Guo, Y. Liu, J. Huang, L. Yuan, Q. Zhang, X. Long, et al. Epona: autoregressive diffusion world model for autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 27220–27230. Cited by: §2, Table 1.

[^53]: Z. Zhao, T. Fu, Y. Wang, L. Wang, and H. Lu From forecasting to planning: policy world model for collaborative state-action prediction. Advances in Neural Information Processing Systems 38, pp. 134585–134611. Cited by: Table 1.

[^54]: W. Zheng, Z. Xia, Y. Huang, S. Zuo, J. Zhou, and J. Lu Doe-1: closed-loop autonomous driving with large world model. arXiv preprint arXiv:2412.09627. Cited by: §2.

[^55]: Y. Zheng, P. Yang, Z. Xing, Q. Zhang, Y. Zheng, Y. Gao, P. Li, T. Zhang, Z. Xia, P. Jia, et al. World4drive: end-to-end autonomous driving via intention-aware physical latent world model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 28632–28642. Cited by: Table 1.

[^56]: X. Zhou, X. Han, F. Yang, Y. Ma, V. Tresp, and A. Knoll OpenDriveVLA: towards end-to-end autonomous driving with large vision language action model. arXiv preprint arXiv:2503.23463. Cited by: §2.

[^57]: Z. Zhou, T. Cai, S. Z. Zhao, Y. Zhang, Z. Huang, B. Zhou, and J. Ma AutoVLA: a vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. arXiv preprint arXiv:2506.13757. Note: Accepted by NeurIPS 2025 Cited by: §2, Table 1.
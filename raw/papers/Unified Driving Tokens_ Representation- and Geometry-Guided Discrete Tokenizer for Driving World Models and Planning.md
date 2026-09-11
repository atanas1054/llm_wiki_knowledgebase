---
title: "Unified Driving Tokens: Representation- and Geometry-Guided Discrete Tokenizer for Driving World Models and Planning"
source: "https://arxiv.org/html/2606.01935v2"
author:
published:
created: 2026-09-04
description:
tags:
  - "clippings"
---
Ziyang Yao Affiliation: Peking University Affiliation: Xiaomi EV    Zeyu Zhu Affiliation: Xiaomi EV    YunCheng Jiang Affiliation: Xiaomi EV    Zibin Guo Affiliation: Xiaomi EV    Huijing Zhao Affiliation: Peking University

###### Abstract

Discrete visual tokens should provide a compact representation for both token-based world modeling and planning in autonomous driving. However, most tokenizers are inherited from image generation and are optimized mainly for pixel reconstruction, which may leave a gap between what is easy to generate and what is useful to decode for driving decisions. We present a representation-guided and geometry-enhanced tokenizer that learns discrete tokens under joint supervision. The tokenizer aligns its discrete bottleneck with a frozen DINO feature space through feature decoding, while preserving appearance via RGB reconstruction with perceptual and adversarial losses. To inject geometric state-related cues, we add adjacent-frame depth and relative-pose supervision during training and stabilize joint objectives with multi-codebook quantization. We evaluate the same learned tokens with a lightweight planning readout and a GPT-style next-token world model. Experiments on NAVSIM show improved reconstruction fidelity and representation consistency, competitive planning performance under a fixed decoder, and better generative quality under matched settings.

###### Keywords:

Autonomous driving Discrete visual tokenization Autoregressive world models

## 1 Introduction

Planning and decision-making are central problems in autonomous driving: given sensory observations and past actions, the system must produce safe, comfortable, and efficient future trajectories [^9] [^2] [^14] [^4]. Recent progress increasingly leverages generative world models to support planning, by learning the conditional distribution of environment evolution for future prediction, counterfactual imagination, and controllable data synthesis [^39] [^40] [^17] [^35] [^11]. A particularly scalable instantiation casts driving as sequence modeling: mapping observations (optionally conditioned on actions and language) into discrete token sequences and training with next-token prediction [^6] [^40]. In this paradigm, a discrete visual tokenizer is not merely a compressor; it defines the discrete language of the driving world—a shared interface that must be amenable to generative sequence learning while being directly consumable by planning and decision modules.

Most existing discrete tokenizers are inherited from image generation, where they are optimized to support two-stage token-based synthesis and to preserve perceptual fidelity under pixel reconstruction [^29] [^28]. For autonomous driving, however, discrete tokens must play a dual role: they are the prediction targets for token-based world modeling and the input representation that downstream planners decode [^6]. Existing driving token pipelines often optimize tokenizers primarily for pixel reconstruction and generation, and assess planning performance only after the fact, rather than treating planning-consumability as a first-class objective during token learning [^17] [^40] [^6]. Motivated by making discrete tokens more useful beyond pure reconstruction, prior work in image/video generation has explored enriching tokenizer supervision beyond pixels—e.g., aligning discrete tokens to strong pre-trained representations via feature reconstruction or distillation—so that tokens carry higher-level structure that downstream modules can decode and exploit, e.g., vision-language tasks [^20] [^24]. For driving, the requirement is sharper: tokens should preserve appearance for generation, encode semantics for scene understanding, and capture geometry/motion cues for planning—suggesting a need for a unified discrete interface that supports both planning consumption and world model generation. Yet, richer alignment also stresses VQ quantization—leading to information loss and codebook instability (collapse and low utilization)—and may degrade the appearance fidelity that generative models rely on [^20].

We therefore propose a representation-guided and geometry-enhanced discrete tokenizer for autonomous driving, explicitly designed as a shared token interface for both token-based world modeling and planning consumption. Our tokenizer builds on a VQ-VAE [^29] backbone with codebook quantization to produce discrete tokens. On the encoder side, we use frozen DINO [^26] features as the primary input to provide instance semantic information that autonomous driving is concerned with, and introduce an explicit image-detail branch to recover textures and boundaries weakened by high-level features and quantization [^41]. On the decoder side, we jointly reconstruct RGB and DINO features, constraining discrete tokens to approximate a strong semantic representation while retaining generative appearance. To inject state-aware cues for planning, we further decode depth and ego-motion from quantized representations of adjacent frames, enabling temporal geometric supervision that complements semantic alignment [^32]. Finally, we adopt multi-codebook quantization to alleviate capacity bottlenecks under joint supervision and to stabilize training [^20] [^15]. To evaluate planning consumability, we train a lightweight planning decoder on frozen discrete tokens to predict future trajectories. To evaluate generative ability, we train an autoregressive next-token Transformer world model on token sequences and assess generation quality.

Our contributions are fourfold.

(1) We propose a unified discrete-token learning framework for autonomous driving, viewing tokens as the discrete driving language that simultaneously serves as the modeling target for token-based world models and the consumable input representation for planning.

(2) We design a semantic representation-guided discrete VQ tokenizer that aligns tokens with strong pre-trained representations while preserving appearance fidelity, via frozen-representation encoding, an explicit detail pathway, and joint reconstruction of RGB and semantic representation features.

(3) We further make the discrete tokens geometry-aware and quantization-stable by introducing temporal depth and ego-motion supervision and multi-codebook quantization, enabling a controlled balance among appearance, semantics, and geometry under joint training.

(4) We demonstrate planning benefits of the learned discrete interface by plugging the same tokens into a lightweight planner and an autoregressive next-token Transformer world model, improving planning metrics and generation quality against representative baselines under matched settings.

## 2 Related Work

### 2.1 Discrete Tokenizer for Visual Generation

Recent progress in visual generation has been largely driven by continuous paradigms, especially diffusion models, which offer strong fidelity when operating in pixel space or learned continuous latents [^12] [^25] [^23]. In parallel, discrete generation revisits image synthesis as sequence modeling by compressing images into a finite token vocabulary, typically via codebook-based tokenizers such as VQ-VAE and VQGAN [^30] [^10]. This tokenization interface makes visual synthesis compatible with autoregressive and masked token modeling, enabling scalable training and efficient sampling [^3] [^15]. More importantly, recent results indicate that, with sufficiently strong tokenizers and recipes, token-based autoregressive generation can be competitive with diffusion and even outperform it in certain scaling regimes [^28] [^41]. These trends suggest that discrete tokenizers are not merely an engineering convenience, but a principled representation layer that unifies reconstruction and generation while providing a natural bridge to large sequence models, which is increasingly important for scalable visual generation [^20] [^24].

### 2.2 World Model–Driven Planning

A recent trend in autonomous driving is to couple world modeling and planning at the model level, rather than using world models only as data generators or simulators. Representative works unify state-action-conditioned scene evolution with downstream decision making, including autoregressive diffusion world models [^39], policy-centric world modeling for collaborative state-action prediction [^40], scaling analyses showing world models can amplify data scaling effects [^17], and formulations that explicitly bridge planning with video generation in a latent driving world [^35]. In the same spirit, earlier efforts cast driving world modeling as controllable sequence generation and enable planning over multiple futures [^13] [^33], and more recent end-to-end frameworks incorporate intention-aware latent world representations for trajectory evaluation and selection [^42]. While these methods often borrow core techniques from image or vision-language generation and then tailor objectives and conditioning for driving-specific constraints, the visual tokenizer, especially the discrete tokenizer, is still commonly pretrained with reconstruction-oriented objectives inherited from generic visual generation [^30] [^10]. This leaves a gap between token semantics optimized for generic reconstruction and the requirements of safety-critical planning, suggesting that learning tokenizers explicitly suited for autonomous driving is a key open problem.

## 3 Method

![[pic2.png|Refer to caption]]

Figure 1: The overall framework of our method. Subfigures (a), (b), and (c) illustrate the overall architecture of the discrete tokenizer, while subgraphs (d) and (e) introduce the two downstream consumption tasks.

### 3.1 Overview

We study discrete visual tokenization for autonomous driving, where the learned tokens must support two downstream uses discussed in Sec. 1: they should be easy to model for token-based world model generation, and they should serve as effective inputs for planning.

Given an RGB frame $\mathbf{I}_{t}\in\mathbb{R}^{H\times W\times 3}$, the tokenizer outputs a grid of discrete token indices $\mathbf{k}_{t}\in\{1,\dots,K\}^{L}$ with corresponding embeddings $\mathbf{E}_{t}\in\mathbb{R}^{L\times d_{q}}$. These tokens are trained to preserve appearance fidelity, capture semantic representations aligned with a frozen foundation model, and encode geometric cues through cross-frame supervision. They can be decoded to RGB and semantic representations for reconstruction objectives, and they provide a compact discrete bottleneck that can be consumed by downstream models.

We first describe the representation-guided and geometry-enhanced tokenizer that learns the discrete tokens. We then assess whether the tokens are usable for planning by training a lightweight trajectory decoder on frozen tokens. Finally, we train an autoregressive next-token Transformer world model on the same token sequences to evaluate generative modelability and rollout quality.

### 3.2 Representation-Guided and Geometry-Enhanced Tokenizer

#### Preliminaries and Notation

We partition he RGB frame from the front camera $\mathbf{I}_{t}$ into non-overlapping patches of size $P\times P$, yielding a patch grid of $H_{p}=\frac{H}{P}$ and $W_{p}=\frac{W}{P}$ with $L=H_{p}W_{p}$ patch locations. We use $t$ to index time and $l\in\{1,\dots,L\}$ to index patch locations.

We denote a frozen foundation visual model as $\Phi(\cdot)$ (e.g., DINO [^26]), which outputs normalized patch-level features:

$$
\mathbf{F}_{t}=\Phi(\mathbf{I}_{t})\in\mathbb{R}^{L\times d_{\text{dino}}}.
$$

We treat $\Phi$ as fully frozen and do not back-propagate gradients through it.

#### Tokenizer Architecture

We build a discrete tokenizer that preserves appearance fidelity while aligning tokens to a semantic representation extracted by a frozen foundation model. The encoder combines an RGB detail branch with patch-wise DINO features, and encodes the fused sequence with a pre-norm Transformer [^31] using Rotary Positional Embeddings (RoPE) [^27] to model spatial ordering on the patch grid.

We first extract patch embeddings from the RGB frame with a lightweight encoder $E_{\text{rgb}}$:

$$
\mathbf{R}_{t}=E_{\text{rgb}}(\mathbf{I}_{t})\in\mathbb{R}^{L\times d_{\text{rgb}}}.
$$

We then concatenate $\mathbf{R}_{t}$ with frozen DINO patch features $\mathbf{F}_{t}$ and project them into a shared hidden space before applying a global Transformer encoder $E_{\text{g}}$:

$$
\mathbf{X}_{t}=W_{f}[\mathbf{R}_{t};\mathbf{F}_{t}]\in\mathbb{R}^{L\times d},\qquad\mathbf{H}_{t}=E_{\text{g}}(\mathbf{X}_{t})\in\mathbb{R}^{L\times d}.
$$

The discrete bottleneck is obtained with vector quantization. We project $\mathbf{H}_{t}$ to the quantization space and assign each patch token to its nearest codeword in a codebook $\mathcal{C}=\{\mathbf{c}_{k}\}_{k=1}^{K}$:

$$
\displaystyle\mathbf{Z}_{t}
$$
 
$$
\displaystyle=P_{\text{in}}(\mathbf{H}_{t})\in\mathbb{R}^{L\times d_{q}},
$$
$$
\displaystyle k_{t,l}
$$
 
$$
\displaystyle=\arg\min_{k\in\{1,\dots,K\}}\|\mathbf{z}_{t,l}-\mathbf{c}_{k}\|_{2}^{2},\qquad\mathbf{e}_{t,l}=\mathbf{c}_{k_{t,l}}.
$$

We denote the discrete index map as $\mathbf{k}_{t}=\{k_{t,l}\}_{l=1}^{L}$ and the quantized embedding sequence as $\mathbf{E}_{t}=\{\mathbf{e}_{t,l}\}_{l=1}^{L}$.

For decoding, we map $\mathbf{E}_{t}$ back to the model dimension and apply a shared post-Transformer $E_{\text{post}}$ (also with RoPE). Two lightweight decoders then reconstruct the RGB image and the semantic representation in the DINO feature space:

$$
\displaystyle\tilde{\mathbf{H}}_{t}
$$
 
$$
\displaystyle=P_{\text{out}}(\mathbf{E}_{t}),\qquad\mathbf{S}_{t}=E_{\text{post}}(\tilde{\mathbf{H}}_{t}),
$$
$$
\displaystyle\hat{\mathbf{I}}_{t}
$$
 
$$
\displaystyle=D_{\text{img}}(\mathbf{S}_{t}),\qquad\hat{\mathbf{F}}_{t}=D_{\text{dino}}(\mathbf{S}_{t}).
$$

Reconstructing $\hat{\mathbf{F}}_{t}$ encourages the discrete tokens to retain semantic representations that downstream models can decode, while $\hat{\mathbf{I}}_{t}$ preserves the appearance information required by generative modeling.

#### Training Objectives

We train the tokenizer to balance appearance fidelity, semantic representation alignment, and a stable discrete bottleneck. The overall objective is a weighted sum of reconstruction, semantic alignment, adversarial training, and quantization regularization:

$$
\mathcal{L}_{\text{tok}}=\lambda_{\text{rec}}\,\mathcal{L}_{\text{rec}}+\lambda_{\text{sem}}\,\mathcal{L}_{\text{sem}}+\lambda_{\text{gan}}\,\mathcal{L}_{\text{gan}}+\lambda_{\text{vq}}\,\mathcal{L}_{\text{vq}}.
$$

The appearance reconstruction loss $\mathcal{L}_{\text{rec}}$ combines a pixel-wise $\ell_{2}$ term with an LPIPS perceptual term:

$$
\mathcal{L}_{\text{rec}}=\|\hat{\mathbf{I}}_{t}-\mathbf{I}_{t}\|_{2}^{2}+\lambda_{\text{lpips}}\,\mathrm{LPIPS}(\hat{\mathbf{I}}_{t},\mathbf{I}_{t}).
$$

Semantic representation alignment is imposed by reconstructing DINO patch features, using the sum of a cosine-similarity term and an MSE term between $\hat{\mathbf{F}}_{t}$ and $\mathbf{F}_{t}$; we denote their combination as $\mathcal{L}_{\text{sem}}$ for brevity. We introduce a discriminator $D$ and optimize the generator with a standard GAN loss $\mathcal{L}_{\text{gan}}$ to distinguish $\mathbf{I}_{t}$ from $\hat{\mathbf{I}}_{t}$.

#### Vector Quantization

To support stable multi-objective training under appearance reconstruction, semantic representation alignment, and geometric supervision, We use a hard vector-quantization bottleneck with an exponential-moving-average (EMA)-updated codebook. Let $\mathbf{Z}\in\mathbb{R}^{B\times L\times d_{q}}$ denote pre-quantization features (with $d_{q}=64$) and let $\mathbf{E}=\{\mathbf{e}_{k}\}_{k=1}^{K}$ be a single codebook with $K=16384$.

Each token is deterministically assigned to its nearest codeword by Euclidean distance:

$$
I_{b,l}=\arg\min_{k\in\{1,\dots,K\}}\|\mathbf{z}_{b,l}-\mathbf{e}_{k}\|_{2}^{2},\qquad\mathbf{q}_{b,l}=\mathbf{e}_{I_{b,l}}.
$$

We use a straight-through estimator so that the forward pass uses $\mathbf{Q}$ while gradients can flow to the encoder:

$$
\tilde{\mathbf{Q}}=\mathbf{Z}+\mathrm{sg}(\mathbf{Q}-\mathbf{Z}),
$$

where $\mathbf{Q}$ stacks $\mathbf{q}_{b,l}$ and $\mathrm{sg}(\cdot)$ denotes stop-gradient.

We further include a commitment term that pulls $\mathbf{Z}$ toward the selected codewords:

$$
\mathcal{L}_{\text{commit}}=\lambda_{c}\cdot\frac{1}{BL}\sum_{b,l}\left\|\mathbf{z}_{b,l}-\mathrm{sg}(\mathbf{q}_{b,l})\right\|_{2}^{2}.
$$

Compared to the standard VQ-VAE formulation that learns codewords by gradient updates, our codebook is updated primarily by an exponential moving average rule that tracks cluster statistics induced by the hard assignments. This choice decouples codeword updates from the competing multiple objectives and empirically improves stability under joint supervision. In addition, we apply dead-code reinitialization and a weak orthogonality regularizer over active codewords to improve utilization and reduce redundancy.

#### Cross-Frame Geometric Supervision

Frame-wise reconstruction provides only per-image constraints and does not explicitly encourage tokens to carry temporally consistent geometric cues. We therefore introduce a geometry branch used only during training, inspired by VGGT-style spatiotemporal token aggregation [^32]. In our setting, the cross-frame supervision is applied on adjacent frames only.

Given two adjacent frames $\mathbf{I}_{t}$ and $\mathbf{I}_{t+1}$, we obtain post-quantization token features from the tokenizer, denoted by $\tilde{\mathbf{H}}_{t},\tilde{\mathbf{H}}_{t+1}\in\mathbb{R}^{L\times d}$. We append an ego token $\mathbf{g}_{t},\mathbf{g}_{t+1}\in\mathbb{R}^{d}$ and apply a temporal aggregator $A_{\psi}$ that alternates frame-wise attention and cross-frame attention to produce geometry-aware patch tokens $\mathbf{U}_{t},\mathbf{U}_{t+1}\in\mathbb{R}^{L\times d}$ and updated ego tokens $\bar{\mathbf{g}}_{t},\bar{\mathbf{g}}_{t+1}\in\mathbb{R}^{d}$:

$$
\{\mathbf{U}_{t},\bar{\mathbf{g}}_{t},\mathbf{U}_{t+1},\bar{\mathbf{g}}_{t+1}\}=A_{\psi}\!\left(\{[\mathbf{g}_{t};\tilde{\mathbf{H}}_{t}],[\mathbf{g}_{t+1};\tilde{\mathbf{H}}_{t+1}]\}\right).
$$

Depth is decoded from $\mathbf{U}_{t}$ and $\mathbf{U}_{t+1}$ using a DPT-style dense prediction head that also outputs a confidence map:

$$
\hat{\mathbf{D}}_{t},\,\hat{\mathbf{C}}_{t}=D_{\omega}(\mathbf{U}_{t}),\qquad\hat{\mathbf{D}}_{t+1},\,\hat{\mathbf{C}}_{t+1}=D_{\omega}(\mathbf{U}_{t+1}).
$$

We regress the relative pose $\hat{\mathbf{p}}_{t\rightarrow t+1}=[\hat{\mathbf{t}}_{t\rightarrow t+1},\hat{\mathbf{q}}_{t\rightarrow t+1}]$ from the updated ego tokens:

$$
\hat{\mathbf{p}}_{t\rightarrow t+1}=h_{\eta}\!\left([\bar{\mathbf{g}}_{t};\bar{\mathbf{g}}_{t+1}]\right).
$$

We supervise the relative pose with an $\ell_{1}$ translation loss and a sign-invariant quaternion loss:

$$
\mathcal{L}_{\text{pose}}=\left\|\hat{\mathbf{t}}_{t\rightarrow t+1}-\mathbf{t}_{t\rightarrow t+1}\right\|_{1}+\min\!\left(\left\|\hat{\mathbf{q}}_{t\rightarrow t+1}-\mathbf{q}_{t\rightarrow t+1}\right\|_{1},\ \left\|\hat{\mathbf{q}}_{t\rightarrow t+1}+\mathbf{q}_{t\rightarrow t+1}\right\|_{1}\right).
$$

For depth, we use a masked regression loss over valid pixels, optionally weighted by $\hat{\mathbf{C}}$, together with a mild smoothness regularizer; we denote the resulting objective as $\mathcal{L}_{\text{depth}}$ and defer its exact form to implementation details.

The geometry losses augment the tokenizer objective in Sec. 3.2:

$$
\mathcal{L}=\mathcal{L}_{\text{tok}}+\lambda_{\text{geo}}\left(\lambda_{\text{depth}}\mathcal{L}_{\text{depth}}+\lambda_{\text{pose}}\mathcal{L}_{\text{pose}}\right).
$$

#### Multi-Codebook Quantization

Joint supervision can create a capacity competition at the discrete bottleneck, where the same set of patch tokens must preserve appearance details while also carrying semantic representation and geometric cues. Inspired by UniTok [^20] to increase capacity without changing the tokenization resolution, we extend the quantizer to a multi-codebook formulation. Here, tokenization resolution refers to the number of patch tokens per frame and their spatial layout (the fixed $H_{p}\times W_{p}$ patch grid, equivalently $L$ tokens per frame).

Given a pre-quantization feature $\mathbf{z}_{t,l}\in\mathbb{R}^{d_{q}}$ for patch $l$ at time $t$, we first produce $M$ head-specific vectors using an attention-based splitter:

$$
\mathbf{V}_{t,l}=P_{\text{attn}}(\mathbf{z}_{t,l})\in\mathbb{R}^{Md_{q}},\qquad\mathbf{v}^{(m)}_{t,l}\in\mathbb{R}^{d_{q}},
$$

where $\mathbf{V}_{t,l}$ is reshaped into $\{\mathbf{v}^{(m)}_{t,l}\}_{m=1}^{M}$. Each head is then quantized with an independent codebook $\mathcal{C}^{(m)}=\{\mathbf{c}^{(m)}_{k}\}_{k=1}^{K_{m}}$ using the same hard nearest-neighbor assignment as in Sec. 3.2, producing selected codewords $\mathbf{e}^{(m)}_{t,l}$.

We fuse the $M$ selected codewords into a single embedding passed to the shared post-encoder and decoders:

$$
\mathbf{e}_{t,l}=P_{\text{merge}}\!\left([\mathbf{e}^{(1)}_{t,l};\dots;\mathbf{e}^{(M)}_{t,l}]\right)\in\mathbb{R}^{d_{q}}.
$$

We found the attention-based split-and-merge along with the EMA updating helpful for balancing codebook utilization under joint objectives.

This design yields $M$ discrete indices per patch while keeping the patch grid unchanged, effectively increasing the representational capacity available to the tokenizer.

### 3.3 Token-Based Planning Decoder

To evaluate whether the learned discrete tokens can be consumed by downstream planning, we freeze the tokenizer from Sec. 3.2 and train a simple trajectory decoder on top of it. At time $t$, the decoder takes the current tokens tokenized from RGB observation $\mathbf{I}_{t}$ and an ego status vector $\mathbf{s}_{t}\in\mathbb{R}^{11}$, and output a future ego trajectory $\mathbf{Y}=\{\mathbf{y}_{\tau}\}_{\tau=1}^{T}$ with $\mathbf{y}_{\tau}=(x_{\tau},y_{\tau},\psi_{\tau})\in\mathbb{R}^{3}$. We use a fixed planning horizon and sampling interval so that $T$ is constant across all experiments.

We obtain patch-level token features $\mathbf{P}_{t}\in\mathbb{R}^{L\times d_{p}}$ from the frozen tokenizer and compress them into a small set of scene tokens using learnable registers and a small transformer:

$$
\mathbf{Z}^{\text{scene}}_{t}=f_{\text{scene}}(\mathbf{P}_{t})\in\mathbb{R}^{R\times d}.
$$

The ego status is embedded into $\mathbf{e}_{t}\in\mathbb{R}^{d}$. A shallow attention module conditions $\mathbf{e}_{t}$ on $\mathbf{Z}^{\text{scene}}_{t}$, and an MLP head then predicts the multiple trajectories. This decoder is intentionally lightweight, as our goal is to probe the quality of the token representation rather than to optimize planner architecture. We train the trajectory head with a standard L1 regression objective, and only the trajectory closest to the ground-truth is supervised.

We add a simple scoring head that predicts PDM-style metric outcomes from the predicted trajectory and scene tokens [^9]. The supervision targets are obtained by running a rule-based evaluator on the predicted trajectory, and the score head is trained with binary cross-entropy. This auxiliary loss provides a lightweight way to incorporate safety and compliance signals without changing the planner structure. At inference time, we use the trajectory with the highest score as the plan.

We report planning performance under this fixed decoder to quantify token consumability; since the decoder capacity and training protocol are kept unchanged across tokenizers, differences primarily reflect the quality of the token representation.

### 3.4 Autoregressive World Model over Discrete Tokens

We freeze the tokenizer and train a GPT-style autoregressive Transformer on the resulting discrete token sequences using next-token prediction [^1] [^28].

For each frame $t$, the tokenizer produces $L$ discrete tokens $\mathbf{k}_{t}\in\{1,\dots,K\}^{L}$. We linearize the per-frame patch tokens with a fixed scan order and concatenate them over a temporal window, yielding a token stream $\mathbf{x}_{1:S}$ with $S=T_{\text{win}}\cdot L$. The world model predicts each token from its preceding context, and can be conditioned on driving signals $\mathbf{c}$ such as actions and ego states.

Conditioning is injected through adaptive layer normalization (AdaLN) [^22]. In each Transformer block, we modulate the normalized hidden activations using scale and shift parameters predicted from a condition embedding. The model is trained with teacher forcing and the standard cross-entropy loss over the discrete token vocabulary.

## 4 Experiments

### 4.1 Implementation Details

We summarize the key settings required to reproduce our experiments, and defer additional architectural and optimization details to Supplementary Material.

Tokenizer variants. We train three tokenizers end-to-end from scratch under the same patching scheme: a naive reconstruction tokenizer (RGB reconstruction only), a DINO-guided tokenizer (adding frozen feature reconstruction with DINOv3-B [^26]), and a semantic representation-guided and geometry-enhanced tokenizer (further adding adjacent-frame depth and relative-pose supervision). The first two use a single codebook of size $16384$, while the geometry-enhanced tokenizer uses $4$ codebooks of size $4096$ each.

Downstream evaluation. For planning, we freeze the tokenizer and train the same lightweight (20M parameters) trajectory readout head on token features (Sec. 3.3), using the geometry-enhanced tokenizer by default in our main planning comparisons. For world modeling, we train a GPT-style next-token Transformer (1B parameters) with cross-entropy on token sequences, using the semantic representation-guided tokenizer by default.

### 4.2 Dataset

We conduct all experiments on the NAVSIM benchmark [^9], a data-driven non-reactive simulation and evaluation suite for autonomous driving. NAVSIM is resampled from OpenScene [^8], which curates about 120 hours of driving logs from nuPlan [^2]. Following the benchmark construction, simple scenarios such as long straight driving are reduced, which strengthens the evaluation sensitivity for planning models [^9] [^8].

We follow a split protocol that separates token learning from downstream planning for fair comparison. The tokenizer and the autoregressive world model are trained on the OpenScene training set. The planning decoder is trained on the NAVSIM training set to match the standard planning benchmark setup. All reported results are evaluated on the NAVSIM test split, and this test split does not overlap with the training data used above. We use the Predictive Driver Model Score (PDMS) of NAVSIM to summarize planning quality [^9]. It aggregates five factors including No At-Fault Collision (NC), Drivable Area Compliance (DAC), Time-to-Collision (TTC), Comfort (Comf.), and Ego Progress (EP).

For cross-frame geometric supervision, we use absolute depth annotations derived from DVGT [^43], which aligns depth pseudo-labels with radar measurements and applies post-processing to filter noisy estimates, yielding absolute depth targets that we use as supervision for the tokenizer geometry branch. These depth targets are only used during tokenizer training and are not required at inference time.

### 4.3 Tokenizer Quality

Table 1: Baseline comparison on the NAVSIM test split. CB denotes the codebook configuration. $\Delta_{\mathrm{img}}^{\cos}$ and $\Delta_{\mathrm{img}}^{\mathrm{rms}}$ measure cosine distance and RMSE in the DINO feature space between reconstructed and ground-truth images. LlamaGen is the tokenizer used by DrivingGPT [^6].

| Tokenizer | CB | rFID $\downarrow$ | PSNR $\uparrow$ | SSIM $\uparrow$ | $\Delta_{\mathrm{img}}^{\cos}\downarrow$ | $\Delta_{\mathrm{img}}^{\mathrm{rms}}\downarrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| LlamaGen [^6] [^28] | 16384 | 5.67 | 23.09 | 0.652 | 0.0869 | 0.167 |
| Orbis [^21] | 2 $\times$ 16384 | 5.53 | 25.94 | 0.773 | 0.0595 | 0.139 |
| Ours-Rep+Geo | 4 $\times$ 4096 | 5.14 | 26.33 | 0.769 | 0.0563 | 0.136 |
| Ours-Rep | 16384 | 4.15 | 26.51 | 0.774 | 0.0453 | 0.122 |

Table 2: Tokenizer roadmap on the NAVSIM test split. AbsRel and $\delta_{1}$ summarize depth prediction; Trans and Rot report average relative-pose translation error in meters and rotation error in degrees. $\Delta_{\mathrm{dec}}^{\cos}$ and $\Delta_{\mathrm{dec}}^{\mathrm{rms}}$ measure cosine distance and RMSE between decoded DINO features and the frozen DINO targets.

| Var. | AbsRel $\downarrow$ | $\delta_{1}\uparrow$ | Trans $\downarrow$ | Rot $\downarrow$ | $\Delta_{\mathrm{dec}}^{\cos}\downarrow$ | $\Delta_{\mathrm{dec}}^{\mathrm{rms}}\downarrow$ | PSNR $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Naive | – | – | – | – | – | – | 25.96 |
| +Rep | – | – | – | – | 0.0346 | 0.105 | 26.51 |
| +Geo | 0.0640 | 0.959 | 0.717 | 2.02 | 0.0717 | 0.153 | 23.90 |
| +MCB | 0.0556 | 0.965 | 0.647 | 2.01 | 0.0486 | 0.124 | 26.33 |

#### Tokenizer Comparison

We benchmark against representative discrete tokenizers that have been adopted in autonomous driving works. They used similar configurations for comparison, both using a resolution of $288\times 512$ [^6] [^21]. All results are reported on the NAVSIM test split; Orbis results are obtained by running the official checkpoint under the same evaluation pipeline. Tab. 1 shows that Ours-Rep improves appearance reconstruction (rFID, PSNR, SSIM) over both baselines. In addition, Ours-Rep achieves the lowest $\Delta_{\mathrm{img}}^{\mathrm{cos}}$ and $\Delta_{\mathrm{img}}^{\mathrm{rms}}$, indicating that reconstructed images remain closer to the ground truth in the frozen DINO feature space. This diagnostic complements pixel-level metrics by evaluating reconstruction quality in the foundation feature space, which emphasizes object-level semantics and spatial layout (e.g., vehicles, road structure, and their relative arrangement) and is less sensitive to low-level appearance variations. This aligns with our goal of representation-guided token learning in Sec. 3.2, where tokens should retain semantically meaningful and structurally consistent cues for downstream modeling and planning.

Beyond these scalar reconstruction metrics, the compared tokenizers also differ in what can be decoded. Our tokenizer includes a dedicated decoder to DINO patch features; the geometry-enhanced variant additionally supports depth and relative-pose decoding.

Taken together, the consistent gains in rFID, PSNR, and SSIM, along with the reduced DINO image-feature discrepancy, provide a coherent reconstruction diagnosis: the tokens support not only pixel-level fidelity but also closer structural consistency under a frozen representation used for supervision in Sec. 3.2.

#### Tokenizer Roadmap

Tab. 2 summarizes the key trade-offs observed along our roadmap. Adding representation supervision improves RGB reconstruction and yields substantially smaller decoded-feature discrepancies in the DINO space ($\Delta_{\mathrm{dec}}^{\mathrm{cos}}$ / $\Delta_{\mathrm{dec}}^{\mathrm{rms}}$), indicating that the discrete tokens better preserve the target representation when it is directly supervised. When geometric supervision is enabled, the tokens become predictive of depth and relative pose, reflected by improved depth metrics (AbsRel, $\delta_{1}$) and standard pose summaries (Trans in meters and Rot in degrees). At the same time, PSNR drops noticeably, suggesting that a fixed discrete bottleneck can face a capacity trade-off when it is asked to preserve appearance, match a strong representation, and encode geometry cues simultaneously. Finally, multi-codebook quantization mitigates this trade-off: it largely restores reconstruction quality while keeping depth and pose performance competitive, supporting the view that the earlier degradation is driven by limited discrete capacity rather than by conflicting objectives.

Overall, Tab. 2 indicates that the main challenge is not whether geometry supervision is compatible with representation alignment, but whether the discrete bottleneck has sufficient capacity to accommodate them simultaneously. Fig. 2 shows the visualization of the reconstruction results. This observation motivates using the geometry-enhanced tokenizer as the default choice when probing planning readout in the next experiment, where we keep the downstream planner lightweight and focus on how much geometric and structural information is recoverable from the tokens.

### 4.4 Planning Performance of Driving Tokens

Table 3: Comparison on the NAVSIM test split. Methods marked with <sup>†</sup> use visual tokens from a frozen tokenizer as input; the remaining methods are end-to-end. Sensor setting: C denotes multi-camera, C+L denotes cameras with LiDAR, and \*C denotes the single-view (1V) setting.

| Method | Sensor | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf.$\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VADv2 [^5] | C | 97.2 | 89.1 | 91.6 | 100.0 | 76.0 | 80.9 |
| UniAD [^14] | C | 97.8 | 91.9 | 92.9 | 100.0 | 78.8 | 83.4 |
| Para-Drive [^34] | C | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 |
| TransFuser [^7] | C+L | 97.7 | 92.8 | 92.8 | 100.0 | 79.2 | 84.0 |
| DRAMA [^37] | C+L | 98.0 | 93.1 | 94.8 | 100.0 | 80.1 | 85.5 |
| DiffusionDrive [^19] | C+L | 98.2 | 96.2 | 94.7 | 100.0 | 82.2 | 88.1 |
| WoTE [^18] | C+L | 98.5 | 96.8 | 94.9 | 99.9 | 81.9 | 88.3 |
| ResWorld [^38] | C | 98.9 | 96.5 | 95.6 | 100.0 | 83.1 | 89.0 |
| Hydra-MDP++ [^16] | C+L | 98.6 | 98.6 | 95.1 | 100.0 | 85.7 | 91.0 |
| DriveSuprim [^36] | C | 98.6 | 98.6 | 95.5 | 100.0 | 91.3 | 93.5 |
| Epona <sup>†</sup> [^39] | \*C | 97.9 | 95.1 | 93.8 | 99.9 | 80.4 | 86.2 |
| PWM <sup>†</sup> [^40] | \*C | 98.6 | 95.9 | 95.4 | 100.0 | 81.8 | 88.1 |
| DriveLaW <sup>†</sup> [^35] | \*C | 99.0 | 97.1 | 96.7 | 100.0 | 81.3 | 89.1 |
| DriveVLA-W0 <sup>†</sup> [^17] | \*C | 98.7 | 99.1 | 95.3 | 99.3 | 83.3 | 90.2 |
| Ours <sup>†</sup> | \*C | 98.7 | 98.2 | 95.9 | 100.0 | 87.3 | 91.8 |

Table 4: Tokenizer Ablation on the NAVSIM test split.

| ID | Rep | Geo | MCB | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf.$\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | $\times$ | $\times$ | $\times$ | 98.2 | 94.8 | 94.6 | 100.0 | 77.8 | 85.5 |
| 2 | $\checkmark$ | $\times$ | $\times$ | 98.7 | 97.4 | 96.3 | 100.0 | 82.1 | 89.4 |
| 3 | $\checkmark$ | $\checkmark$ | $\times$ | 98.6 | 97.6 | 95.7 | 100.0 | 86.3 | 90.9 |
| 4 | $\checkmark$ | $\checkmark$ | $\checkmark$ | 98.7 | 98.2 | 95.9 | 100.0 | 87.3 | 91.8 |

![[pic3.png|Refer to caption]]

Figure 2: Qualitative visualization of tokenizer reconstructions. Subfigures (a) and (b) show the semantic representation-guided tokenizer, while (c) further incorporates geometry enhancement with a multi-codebook design. DINO features are visualized via PCA.

![[pic4.png|Refer to caption]]

Figure 3: Qualitative visualization of tokenizer ablation for planning. Subfigures (a) and (b) show two cases that the naive tokenizer fails, while the Rep+Geo tokenizer succeeds.

We evaluate driving-token quality through planning on the NAVSIM test split. We report PDMS and its standard components (NC, DAC, TTC, Comfort, and EP) under the official evaluation pipeline.

#### Comparisons with State-of-the-Art Methods

To place our results in context, we compare against two groups of methods in Tab. 3: end-to-end planning systems that learn perception and planning jointly, and token-based methods that take visual tokens from a frozen tokenizer as input. Tab. 3 reports planning performance on the NAVSIM test split. Under the single-view camera setting, our token-based planner achieves the highest PDMS among methods that plan from frozen visual tokens. Across sensor settings, several end-to-end systems benefit from multi-camera or LiDAR inputs, which makes direct comparison less controlled, but the results indicate that the learned tokens support competitive single-view planning when used as frozen inputs.

#### Tokenizer Ablation for Planning

The ablation in Tab. 4 follows the tokenizer roadmap and isolates how each design choice affects planning readout under the same decoder and training protocol. Adding representation supervision improves PDMS substantially over the naive reconstruction-only tokenizer, with consistent gains on NC, DAC, TTC, and EP. Enabling geometric supervision further increases PDMS, mainly through EP, which is consistent with the intent of making tokens more predictive of state and motion cues used by planning. Finally, multi-codebook quantization yields the best overall PDMS while preserving strong safety and compliance components, suggesting that additional discrete capacity helps retain both representation and geometry information that the readout can use. Furthermore, the visualized case comparisons in Fig. 3 also reflect the above conclusions.

### 4.5 Visual Generation of the World Model

![[pic5.png|Refer to caption]]

Figure 4: Generative performance of world models. Subfigures (a) compares the generation performance of our method with the baseline. Subfigures (b) visualizes the generation results.

Following DrivingGPT [^6], we evaluate visual rollout using a fixed history-and-prediction protocol. We condition the world model on the past three frames and generate the next eight frames at 2 Hz. All metrics are computed on the NAVSIM test split at the same image resolution of $288\times 512$.

As shown in Fig. 4, the world model trained on our tokens achieves better FID and FVD than the baseline under the matched setting. This indicates that our tokenizer does not sacrifice generative usability when optimized for representation guidance and downstream planning probes.

Fig. 4 visualizes a representative rollout example. The model generates coherent future frames and, through the same discrete tokens, we can also decode the corresponding DINO feature targets for each predicted frame. The decoded DINO features remain consistent with the generated images, providing an additional view of structural consistency along the rollout.

Overall, these results support the premise in Sec. 1 that a single discrete tokenization can serve both planning readout and world model generation within a unified training and evaluation pipeline.

## 5 Conclusion

We studied discrete visual tokenization for autonomous driving under the requirement that the same tokens should support planning readout and autoregressive sequence prediction. We proposed a tokenizer trained with representation supervision from frozen DINO features, complemented by RGB reconstruction with perceptual and adversarial objectives. We further introduced adjacent-frame depth and relative-pose supervision to encourage geometry-aware tokens, and used multi-codebook quantization to reduce capacity pressure under joint training. Across NAVSIM evaluations, the resulting tokens improve reconstruction fidelity and yield smaller discrepancies in the DINO feature space, which serves as a practical diagnostic of structural consistency beyond pixel metrics. Using fixed downstream models, the tokens enable strong planning performance and support high-quality world model rollouts. Future work includes extending geometric supervision beyond adjacent pairs and unifying planning and world modeling within a unified architecture.

[^1]: T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al. (2020) Language models are few-shot learners. In Advances in Neural Information Processing Systems, Cited by: §3.4.

[^2]: H. Caesar, J. Kabzan, K. S. Tan, W. K. Fong, E. Wolff, A. Lang, L. Fletcher, O. Beijbom, and S. Omari (2021) Nuplan: a closed-loop ml-based planning benchmark for autonomous vehicles. arXiv preprint arXiv:2106.11810. Cited by: §1, §4.2.

[^3]: H. Chang, H. Zhang, L. Jiang, C. Liu, and W. T. Freeman (2022) MaskGIT: masked generative image transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Cited by: §2.1.

[^4]: L. Chen, P. Wu, K. Chitta, B. Jaeger, A. Geiger, and H. Li (2024) End-to-end autonomous driving: challenges and frontiers. IEEE Transactions on Pattern Analysis and Machine Intelligence 46, pp. 10164–10183. External Links: [Document](https://dx.doi.org/10.1109/TPAMI.2024.3435937) Cited by: §1.

[^5]: S. Chen, B. Jiang, H. Gao, B. Liao, Q. Xu, Q. Zhang, C. Huang, W. Liu, and X. Wang (2024) Vadv2: end-to-end vectorized autonomous driving via probabilistic planning. arXiv preprint arXiv:2402.13243. Cited by: Table 3.

[^6]: Y. Chen, Y. Wang, and Z. Zhang (2025) Drivinggpt: unifying driving world modeling and planning with multi-modal autoregressive transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 26890–26900. Cited by: §1, §1, §4.3, §4.5, Table 1, Table 1, Table 1.

[^7]: K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger (2022) Transfuser: imitation with transformer-based sensor fusion for autonomous driving. IEEE transactions on pattern analysis and machine intelligence 45 (11), pp. 12878–12895. Cited by: Table 3.

[^8]: O. Contributors (2023) Openscene: the largest up-to-date 3d occupancy prediction benchmark in autonomous driving. In Proceedings of the Conference on Computer Vision and Pattern Recognition, Vancouver, Canada, pp. 18–22. Cited by: §4.2.

[^9]: D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, et al. (2024) Navsim: data-driven non-reactive autonomous vehicle simulation and benchmarking. Advances in Neural Information Processing Systems 37, pp. 28706–28719. Cited by: §1, §3.3, §4.2, §4.2.

[^10]: P. Esser, R. Rombach, and B. Ommer (2021) Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Cited by: §2.1, §2.2.

[^11]: T. Feng, W. Wang, and Y. Yang (2025) A survey of world models for autonomous driving. arXiv preprint arXiv:2501.11260. Cited by: §1.

[^12]: J. Ho, A. Jain, and P. Abbeel (2020) Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, Cited by: §2.1.

[^13]: A. Hu, L. Russell, H. Yeo, Z. Murez, G. Fedoseev, A. Kendall, J. Shotton, and G. Corrado (2023) GAIA-1: a generative world model for autonomous driving. arXiv preprint arXiv:2309.17080. Cited by: §2.2.

[^14]: Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, et al. (2023) Planning-oriented autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 17853–17862. Cited by: §1, Table 3.

[^15]: D. Lee, C. Kim, S. Kim, M. Cho, and W. Han (2022) Autoregressive image generation using residual quantization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Cited by: §1, §2.1.

[^16]: K. Li, Z. Li, S. Lan, Y. Xie, Z. Zhang, J. Liu, Z. Wu, Z. Yu, and J. M. Alvarez (2025) Hydra-mdp++: advancing end-to-end driving via expert-guided hydra-distillation. arXiv preprint arXiv:2503.12820. Cited by: Table 3.

[^17]: Y. Li, S. Shang, W. Liu, B. Zhan, H. Wang, Y. Wang, Y. Chen, X. Wang, Y. An, C. Tang, et al. (2025) DriveVLA-w0: world models amplify data scaling law in autonomous driving. arXiv preprint arXiv:2510.12796. Cited by: §1, §1, §2.2, Table 3.

[^18]: Y. Li, Y. Wang, Y. Liu, J. He, L. Fan, and Z. Zhang (2025) End-to-end driving with online trajectory evaluation via bev world model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 27137–27146. Cited by: Table 3.

[^19]: B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang, et al. (2025) Diffusiondrive: truncated diffusion model for end-to-end autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 12037–12047. Cited by: Table 3.

[^20]: C. Ma, Y. Jiang, J. Wu, J. Yang, X. Yu, Z. Yuan, B. Peng, and X. Qi (2025) UniTok: a unified tokenizer for visual generation and understanding. In Advances in Neural Information Processing Systems, External Links: [Link](https://openreview.net/forum?id=f6aOPkGE8L) Cited by: §1, §1, §2.1, §3.2.

[^21]: A. Mousakhan, S. Mittal, S. Galesso, K. Farid, and T. Brox (2025) Orbis: overcoming challenges of long-horizon prediction in driving world models. arXiv preprint arXiv:2507.13162. Cited by: §4.3, Table 1.

[^22]: W. Peebles and S. Xie (2023) Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), Cited by: §3.4.

[^23]: W. Peebles and S. Xie (2023) Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), Cited by: §2.1.

[^24]: L. Qu, H. Zhang, Y. Liu, X. Wang, Y. Jiang, Y. Gao, H. Ye, D. K. Du, Z. Yuan, and X. Wu (2025) TokenFlow: unified image tokenizer for multimodal understanding and generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Cited by: §1, §2.1.

[^25]: R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer (2022) High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Cited by: §2.1.

[^26]: O. Siméoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab, C. Jose, V. Khalidov, M. Szafraniec, S. Yi, M. Ramamonjisoa, et al. (2025) Dinov3. arXiv preprint arXiv:2508.10104. Cited by: §1, §3.2, §4.1.

[^27]: J. Su, Y. Lu, S. Pan, A. Murtadha, B. Wen, and Y. Liu (2021) RoFormer: enhanced transformer with rotary position embedding. arXiv preprint arXiv:2104.09864. Cited by: §3.2.

[^28]: P. Sun, Y. Jiang, S. Chen, S. Zhang, B. Peng, P. Luo, and Z. Yuan (2024) Autoregressive model beats diffusion: llama for scalable image generation. arXiv preprint arXiv:2406.06525. Cited by: §1, §2.1, §3.4, Table 1.

[^29]: A. van den Oord, O. Vinyals, and K. Kavukcuoglu (2017) Neural discrete representation learning. In Advances in Neural Information Processing Systems, Cited by: §1, §1.

[^30]: A. van den Oord, O. Vinyals, and K. Kavukcuoglu (2017) Neural discrete representation learning. In Advances in Neural Information Processing Systems, Cited by: §2.1, §2.2.

[^31]: A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin (2017) Attention is all you need. In Advances in Neural Information Processing Systems, Cited by: §3.2.

[^32]: J. Wang, A. Vedaldi, A. Zisserman, et al. (2025) VGGT: visual geometry grounded transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Cited by: §1, §3.2.

[^33]: Y. Wang, J. He, L. Fan, H. Li, Y. Chen, and Z. Zhang (2024) Driving into the future: multiview visual forecasting and planning with world model for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Cited by: §2.2.

[^34]: X. Weng, B. Ivanovic, Y. Wang, Y. Wang, and M. Pavone (2024) Para-drive: parallelized architecture for real-time autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15449–15458. Cited by: Table 3.

[^35]: T. Xia, Y. Li, L. Zhou, J. Yao, K. Xiong, H. Sun, B. Wang, K. Ma, G. Chen, H. Ye, et al. (2025) DriveLaW: unifying planning and video generation in a latent driving world. arXiv preprint arXiv:2512.23421. Cited by: §1, §2.2, Table 3.

[^36]: W. Yao, Z. Li, S. Lan, Z. Wang, X. Sun, J. M. Alvarez, and Z. Wu (2025) Drivesuprim: towards precise trajectory selection for end-to-end planning. arXiv preprint arXiv:2506.06659. Cited by: Table 3.

[^37]: C. Yuan, Z. Zhang, J. Sun, S. Sun, Z. Huang, C. D. W. Lee, D. Li, Y. Han, A. Wong, K. P. Tee, et al. (2024) Drama: an efficient end-to-end motion planner for autonomous driving with mamba. arXiv preprint arXiv:2408.03601. Cited by: Table 3.

[^38]: J. Zhang, Z. Fu, Z. Xu, W. Dai, Q. Liu, and Y. Wang (2026) ResWorld: temporal residual world model for end-to-end autonomous driving. arXiv preprint arXiv:2602.10884. Cited by: Table 3.

[^39]: K. Zhang, Z. Tang, X. Hu, X. Pan, X. Guo, Y. Liu, J. Huang, L. Yuan, Q. Zhang, X. Long, et al. (2025) Epona: autoregressive diffusion world model for autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 27220–27230. Cited by: §1, §2.2, Table 3.

[^40]: Z. Zhao, T. Fu, Y. Wang, L. Wang, and H. Lu (2025) From forecasting to planning: policy world model for collaborative state-action prediction. arXiv preprint arXiv:2510.19654. Cited by: §1, §1, §2.2, Table 3.

[^41]: A. Zheng, X. Wen, X. Zhang, C. Ma, T. Wang, G. Yu, X. Zhang, and X. Qi (2025) Vision foundation models as effective visual tokenizers for autoregressive generation. In Advances in Neural Information Processing Systems, External Links: [Link](https://openreview.net/pdf?id=PESrAH82Zh) Cited by: §1, §2.1.

[^42]: Y. Zheng, P. Yang, Z. Xing, Q. Zhang, Y. Zheng, Y. Gao, P. Li, and T. Zhang (2025) World4Drive: end-to-end autonomous driving via intention-aware physical latent world model. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), Cited by: §2.2.

[^43]: S. Zuo, Z. Xie, W. Zheng, S. Xu, F. Li, S. Jiang, L. Chen, Z. Yang, and J. Lu (2025) DVGT: driving visual geometry transformer. arXiv preprint arXiv:2512.16919. Cited by: §4.2.
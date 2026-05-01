---
title: "Latent-WAM: Latent World Action Modeling for End-to-End Autonomous Driving"
source: "https://arxiv.org/html/2603.24581v1"
author:
published:
created: 2026-05-01
description:
tags:
  - "clippings"
---
<sup>1</sup> <sup>2</sup> <sup>3</sup> <sup>4</sup>

Linbo Wang Work done during internship at Chongqing Chang’an Technology Co., Ltd.    Yupeng Zheng Project Leader & Equal Contribution    Qiang Chen    Shiwei Li    Yichen Zhang    Zebin Xing    Qichao Zhang Corresponding author.    Xiang Li    Deheng Qian     
Pengxuan Yang    Yihang Dong    Ce Hao    Xiaoqing Ye    Junyu Han     
Yifeng Pan    Dongbin Zhao

###### Abstract

We introduce Latent-WAM, an efficient end-to-end autonomous driving framework that achieves strong trajectory planning through spatially-aware and dynamics-informed latent world representations. Existing world-model-based planners suffer from inadequately compressed representations, limited spatial understanding, and underutilized temporal dynamics, resulting in sub-optimal planning under constrained data and compute budgets. Latent-WAM addresses these limitations with two core modules: a Spatial-Aware Compressive World Encoder (SCWE) that distills geometric knowledge from a foundation model and compresses multi-view images into compact scene tokens via learnable queries, and a Dynamic Latent World Model (DLWM) that employs a causal Transformer to autoregressively predict future world status conditioned on historical visual and motion representations. Extensive experiments on NAVSIM v2 and HUGSIM demonstrate new state-of-the-art results: 89.3 EPDMS on NAVSIM v2 and 28.9 HD-Score on HUGSIM, surpassing the best prior perception-free method by 3.2 EPDMS with significantly less training data and a compact 104M-parameter model.

## 1 Introduction

End-to-end autonomous driving has attracted considerable attention due to its data-driven nature and scalability. Prior methods integrate perception and prediction into a unified differentiable network, extracting planning-relevant representations for end-to-end trajectory prediction \[hu2023uniad, jiang2023vad\]. However, these methods generally rely on complex auxiliary task designs and perception annotations, limiting the further scaling of end-to-end driving algorithms.

Recently, world-model-based approaches learn scene representations through temporal self-supervised learning, providing dense supervision while reducing the dependence on perception labels. These methods can be broadly categorized into two groups. The first group \[zhang2025epona, li2025drivevla-w0\] learns driving planning through explicit video generation. However, video generation incurs substantial computational overhead, and the learned intermediate representations tend to focus on visual details irrelevant to planning. The second group learns planning-relevant representations through implicit future latent prediction \[li2024law, zheng2025world4drive\]. Although the latent representations are relatively lightweight and capture dynamic information via temporal self-supervision, they still suffer from insufficient representation quality. Specifically: (1) the latent representations remain inadequately compressed; (2) they lack spatial understanding or rely on external depth estimation models at inference time, introducing additional latency; and (3) historical and dynamic information are underutilized, as these methods merely predict the $T{+}1$ representation from the $T$ -th frame. As illustrated in Fig.˜1, these limitations lead to sub-optimal planning performance.

![[x1 30.png|Refer to caption]]

Figure 1: Performance vs. training data scale on NAVSIM v2. Bubble size indicates model parameters. Our Latent-WAM achieves the highest EPDMS with significantly fewer training data and smaller model size, demonstrating superior data efficiency over existing world-model-based methods. World4Drive is marked separately as it employs an additional ViT-L depth estimator.

To address these issues, we propose Latent-WAM, which comprises a Spatial-Aware Compressive World Encoder (SCWE) and a Dynamic Latent World Model (DLWM), targeting the two most planning-relevant understanding tasks—spatial and dynamic understanding—to achieve highly compressed driving world representations with improved planning performance. Specifically, SCWE distills knowledge from a geometric foundation model into the vision backbone and employs learnable queries to extract spatially-informed tokens from the enriched features, achieving high compression of scene information. DLWM adopts a causal Transformer for world modeling, predicting future world status conditioned on the ego vehicle’s historical world status, including compressed visual and motion representations. Through self-supervised visual prediction and supervised motion prediction, the world status representations acquire dynamic understanding capabilities relevant to planning. Finally, a lightweight trajectory decoder generates the planned trajectory from the world status representations.

We extensively evaluate Latent-WAM on NAVSIM v2 \[cao2025navsimv2\] and HUGSIM \[zhou2024hugsim\]. Latent-WAM achieves 89.3 EPDMS on NAVSIM v2, 45.9 RC and 28.9 HD-score on HUGSIM, establishing new state-of-the-art results. Under the perception-free setting, our method outperforms previous approaches by 3.2 EPDMS. Notably, attention map visualization reveals that our world representations are highly focused on spatial structures and driving intent, demonstrating the effectiveness of our approach for planning-centric representation learning. Our contributions are summarized as follows:

- We propose Spatial-Aware Compressive World Encoder, a novel scene compression method that distills geometric knowledge into the vision backbone to extract highly compressed, spatially-aware visual representations.
- We propose Dynamic Latent World Model, a novel world modeling approach that leverages a causal Transformer to jointly learn visual and motion dynamics, building representations with dynamic scene understanding.
- Our method achieves new state-of-the-art results on both NAVSIM v2 and HUGSIM, outperforming prior perception-free method \[li2025drivevla-w0\] by 3.2 EPDMS.

## 2 Related Works

### 2.1 Scene Representation in End-to-End Autonomous Driving

End-to-end autonomous driving directly optimizes trajectory planning, making the intermediate scene representation critical to overall performance. Early works \[hu2023uniad, transfuser\] adopt dense BEV layouts supervised by semantic maps and occupancy labels. Subsequent methods \[jiang2023vad, chen2024vadv2, sun2024sparsedrive, jia2025drivetransformer\] shift towards lightweight vectorized or sparse representations for improved efficiency. Recent efforts leverage VLMs \[tian2024token, pan2024vlp, Hegde2025DistillingML\] for richer semantic information, or employ latent world models \[li2024law, zheng2025world4drive\] and video generation \[xia2025drivelaw\] for scene representation. However, these methods often exhibit weak 3D spatial understanding and rely on cumbersome representations. We address these issues by distilling geometric knowledge into the vision backbone for stronger spatial understanding, while compressing scene information into a compact set of tokens via learnable queries.

### 2.2 World Models for Autonomous Driving

World models have been widely adopted in autonomous driving for environment representation and future state prediction. One line of work applies video generation models \[Hu2023GAIA1AG, wang2023drivedreamer, zhao2024drivedreamer-2, chen2024driveworld, wang2023driving-wm, gao2024vista, guo2024dist4d\] to construct pixel-level driving world representations, while 3D world models based on occupancy and point clouds \[zheng2023occworld, zhang2024copilotd, zyrianov2025lidardm, dang2025sparseworld\] enforce geometric constraints in 3D space, with subsequent works \[li2024uniscene, li2025scaling\] unifying 2D and 3D modeling. Building on these capabilities, some approaches \[yang2024drivearena, li2025omninwm\] use world models as closed-loop simulators, while others \[zhang2025epona, li2025drivevla-w0, xia2025drivelaw, li2024law, zheng2025world4drive\] directly leverage them for planning. To reduce computational overhead, LAW \[li2024law\], World4Drive \[zheng2025world4drive\], and Drive-JEPA \[wang2026drivejepa\] perform trajectory planning in the latent space. Our method similarly adopts a latent world model, but further incorporates ego status to guide future predictions and 3D-RoPE to enhance spatio-temporal tracking.

## 3 Method

### 3.1 Overall

The overall architecture of Latent-WAM is illustrated in Fig.˜2, consisting of three core modules: 1) Spatial-Aware Compressive World Encoder(Sec.˜3.2), which uses learnable queries and a vision encoder to compress images into compact scene tokens and a geometric foundation model is used to distill geometric perception ability into the encoder to improve spatial understanding, 2) Dynamic Latent World Model (Sec.˜3.3), a causal transformer decoder that models world transition dynamics by autoregressively predicting future world status conditioned on historical scene representations and ego status, and 3) Trajectory Decoder(Sec.˜3.4), which forecasts trajectories over a 4-second horizon from world status representations. The SCWE and DLWM jointly optimize the vision backbone. Geometric distillation enhances spatial understanding, while self-supervised learning improves temporal dynamics modeling. This design achieves strong planning performance at inference time without extra computational cost from auxiliary modules.

![[x2 28.png|Refer to caption]]

Figure 2: Overview of the Latent-WAM architecture. See Sec. ˜ 3.1 for details.

### 3.2 Spatial-Aware Compressive World Encoder

To model long-horizon world state transition, world status representations need to be sufficiently lightweight while compressing rich scene information. Different from prior methods that rely on extensive visual information, we use only a small set of learnable queries that fully interact with image patch tokens from the inputs, compressing rich world perception information into the latent space. In addition, to enhance the spatial understanding of the vision backbone, we use a geometric foundation model WorldMirror \[liu2025worldmirror\] to inject geometric awareness into the backbone through distillation.

#### 3.2.1 Scene Compression.

We use a set of compact scene tokens to compress the heavy vision tokens from multi-view images, forming a foundational component of world status representations.

Given sequential multi-view images $I\in\mathbb{R}^{\mathbf{T}\times\mathbf{M}\times H\times W\times 3}$ as input, we first embed them into image patch tokens $X\in\mathbb{R}^{\mathbf{T}\times\mathbf{M}\times\mathbf{S}\times D_{e}}$, where $\mathbf{T},\mathbf{M},\mathbf{S}$ denote the temporal horizon, number of cameras, and number of image patches, respectively. A set of scene queries $Q_{\text{scene}}\in\mathbb{R}^{\mathbf{T}\times\mathbf{M}\times\mathbf{N}\times D_{e}}$ are randomly initialized and concatenated with $X$. The concatenated features are fed into a DINO encoder $\mathcal{E}$ containing an MLP that project to $D_{l}$ -dimensional latent space, yielding frame-wise and view-specific scene representations $\hat{Q}_{\text{scene}}\in\mathbb{R}^{\mathbf{T}\times\mathbf{M}\times\mathbf{N}\times D_{l}}$ and image tokens $\hat{X}\in\mathbb{R}^{\mathbf{T}\times\mathbf{M}\times\mathbf{S}\times D_{l}}$:

$$
\displaystyle\hat{Q}_{\text{scene}},\hat{X}=\mathcal{E}\left(\left[Q_{\text{scene}};X\right]\right)
$$

where $\mathbf{N}$ is the number of scene query tokens, $D_{e}$ the encoder dimension, and $D_{l}$ the latent dimension.

By integrating scene queries with raw image tokens, extensive visual information from numerous image patch tokens is efficiently compressed into a compact set of tokens, significantly reducing computational overhead for subsequent long-term world model training and trajectory planning.

#### 3.2.2 Geometric Alignment.

To distill the spatial understanding capabilities of geometric foundation models into the DINO encoder, we employ the image patch tokens output by the SCWE as carriers for receiving dense spatial-semantic information from geometric features.

Multi-view images across consecutive frames $I$ are additionally fed into a geometric foundation model $f_{g}$, producing patch-level geometric features $f_{g}(I)\in\mathbb{R}^{\textbf{T}\times\textbf{M}\times\textbf{S}\times D_{g}}$. The DINO backbone outputs $\hat{X}$ are subsequently projected via a geometric projector $\phi$, yielding $\phi(\hat{X})\in\mathbb{R}^{\textbf{T}\times\textbf{M}\times\textbf{S}\times D_{g}}$. Both $f_{g}(I)$ and $\phi(\hat{X})$ are first normalized using LayerNorm, then we compute their cosine similarity loss:

$$
\displaystyle\mathcal{L}_{\text{align}}=1-\text{cos}\left(\text{LN}(\phi(\hat{X})),\text{LN}(f_{g}(I))\right)
$$

where $\cos(\cdot,\cdot)$ denotes cosine similarity and $\text{LN}(\cdot)$ denotes LayerNorm.

Notably, since $f_{g}$ remains frozen throughout training, the geometric features can be pre-computed and offline-cached, enabling direct loading during training without incurring repeated inference costs or GPU memory overhead. This design substantially reduces training time and computational expenses.

### 3.3 Dynamic Latent World Model

In this section, we propose a Dynamic Latent World Model(DLWM). By predicting future world status representations causally in the latent space, the backbone obtains temporal dynamics modeling capabilities through a self-supervised training paradigm under the perception-free setting.

#### 3.3.1 World latent status aggregation.

The per-camera scene tokens $\hat{Q}_{\text{scene}}$ only capture isolated view-specific information, which is insufficient for holistic world modeling. To enable effective future prediction and trajectory planning, we aggregate these tokens and integrate with ego status to construct unified frame-wise world status representations. The ego status across consecutive frames, which include driving commands, velocity, and acceleration, is encoded by an ego status encoder (a single-layer MLP) into ego status embeddings $S_{\text{ego}}\in\mathbb{R}^{\textbf{T}\times D_{l}}$. The scene tokens $\hat{Q}_{\text{scene}}$ from the SCWE and the ego status embeddings $S_{\text{ego}}$ collectively constitute the frame-wise world latent state.

Specifically, scene tokens from different cameras are aggregated to form a holistic perception representation of the surrounding environment, which is subsequently concatenated with $S_{\text{ego}}$ encoding the ego status, yielding a unified Scene-Ego world status representation $S_{\text{world}}\in\mathbb{R}^{\textbf{T}\times(\textbf{M}\times\textbf{N}+1)\times D_{l}}$. This representation serves as the foundation for subsequent future world status prediction within the world model and trajectory planning.

#### 3.3.2 Causal world model prediction.

We formulate world state transition dynamics modeling as an autoregressive prediction problem, where all future frame world status predictions are conditioned on historical world status representations, encompassing holistic scene representations and ego status. This unified autoregressive framework enables a natural training strategy: treating the alternating Scene-Ego world status token sequence $S_{\text{world}}^{i}$ as frame-wise blocks and adopting the standard next-token prediction to train the world model.

In detail, we randomly initialize a set of learnable future world status queries $Q_{\text{future}}\in\mathbb{R}^{(\textbf{T}-1)\times(\textbf{M}\times\textbf{N}+1)\times D_{l}}$, with $S_{\text{world}}^{i},i\in\{1,\ldots,T-1\}$ forming the key-value cache $KV_{\text{future}}\in\mathbb{R}^{(\textbf{T}-1)\times(\textbf{M}\times\textbf{N}+1)\times D_{l}}$. The DLWM adopts a standard Transformer decoder incorporating Rotary Position Embedding, producing future world status predictions $S_{\text{future}}\in\mathbb{R}^{(\textbf{T}-1)\times(\textbf{M}\times\textbf{N}+1)\times D_{l}}$. Formally,

$$
\displaystyle S_{\text{future}}=\text{DLWM}(Q_{\text{future}},KV_{\text{future}})
$$

The ground truth $S_{\text{future}}^{\text{GT}}$ is generated by a target encoder, which is a frozen copy of the SCWE updated via Exponential Moving Average (EMA) to provide stable supervision signals.

#### 3.3.3 Teacher Forcing Attention Mask.

Given a Scene-Ego interleaved token sequence, the world model predicts each future token by attending to all historical tokens. We adopt teacher forcing during training: ground truth tokens serve as context for predicting subsequent tokens, preventing error accumulation across timesteps.

![[x3 28.png|Refer to caption]]

Figure 3: Teacher Forcing Attention Mask.

Causal prediction is implemented via frame-wise attention masks, as shown in Fig.˜3. Within each frame block, tokens attend bidirectionally to each other. Cross-frame, each token can only attend to tokens appearing earlier in the sequence, enforcing temporal causality. This frame-wise attention design enables parallel prediction of all future world status while maintaining causal consistency, significantly improving training speed and efficiency.

#### 3.3.4 3D-RoPE.

Although $S_{\text{world}}$ contains world status representations across multiple timestamps and camera views, the query vectors $Q_{\text{future}}$ and context features $KV_{\text{future}}$ do not explicitly carry temporal or spatial position information. To enable the model to distinguish temporal and spatial relationships among tokens, we inject spatio-temporal position information into multi-head attention via 3D-RoPE, which differentiates tokens across timesteps, camera views, and positions within long sequences.

In practice, we split the head dimension $D_{h}$ into three parts to encode temporal coordinate $t$, camera index $m$, and token index $n$ into $Q_{\text{future}}$ and $KV_{\text{future}}$. We use absolute position indices to encode all three coordinates: temporal dimension with frequency 50, camera index with frequency 10, and token index with frequency 100.

#### 3.3.5 Ego status supervision.

The world status representation $S_{\text{world}}$ contains both scene tokens and ego status embeddings. During world model prediction, ego status guides the evolution of scene representations as well as itself. Therefore, accurate prediction of ego status is crucial for modeling world dynamics. To provide precise guidance for the transition of world state, we introduce ego status supervision built upon the self-supervised future status prediction.

We extract ego status embeddings $S_{\text{ego}}^{i^{\prime}}$ from the predicted future world status $S_{\text{future}}$, where $i^{\prime}\in\{2,\ldots,T\}$. These embeddings are then fed into three separate MLPs: a driving command decoder $D_{\text{cmd}}$, a velocity decoder $D_{v}$, and an acceleration decoder $D_{a}$, producing predictions $C_{\text{pred}}\in\mathbb{R}^{(\textbf{T}-1)\times 4}$, $V_{\text{pred}}\in\mathbb{R}^{(\textbf{T}-1)\times 2}$, and $A_{\text{pred}}\in\mathbb{R}^{(\textbf{T}-1)\times 2}$, respectively. Formally:

$$
\displaystyle C_{\text{pred}}=\text{Softmax}(D_{\text{cmd}}(S_{\text{ego}}^{{}^{\prime}})),\ V_{\text{pred}}=D_{v}(S_{\text{ego}}^{{}^{\prime}}),\ A_{\text{pred}}=D_{a}(S_{\text{ego}}^{{}^{\prime}}).
$$

### 3.4 Trajectory Planning

We employ a trajectory decoder $D_{\tau}$ to planning, which generates trajectories corresponding to different driving intentions based on the control commands. A set of randomly initialized learnable trajectory queries $Q_{\tau}\in\mathbb{R}^{K\times n_{p}\times D_{l}}$ is fed into the trajectory decoder along with the current world status representation $S_{world}^{t}$ of the driving scenario. After processing, the trajectory tokens are decoded via a lightweight MLP into $K$ candidate trajectories. Finally, conditioned on the driving command $C$ at the current timestep $t$, the corresponding candidate trajectory is selected as the final trajectory $\tau$. Formally,

$$
\displaystyle\tau=D_{\tau}(Q_{\tau},S_{world}^{t},C)
$$

Each decoded candidate trajectory $\tau_{i}$ is a sequence of $n_{p}$ poses spanning from the current timestep $t$ to a future time $t+T$, where $T$ represents the total prediction horizon. Each pose is represented as $(x,y,\theta)\in\mathbb{R}^{3}$, yielding complete trajectories in $\mathbb{R}^{n_{p}\times 3}$, where $x$ and $y$ correspond to longitudinal and lateral displacements, respectively, with $\theta$ denoting the heading angle. All quantities are referenced to the ego vehicle’s local coordinate system at timestep $t$. The time interval between consecutive predicted poses is assumed to be uniform.

### 3.5 Training and Inference

Training Objective. Following prior work, we employ $L_{1}$ loss $\mathcal{L}_{traj}$ to optimize the generated multi-candidate trajectories by imitating expert trajectories. Geometric alignment computes cosine similarity loss $\mathcal{L}_{align}$ on normalized features, maximizing cosine similarity to distill spatial understanding capability. For the world model, we use MSE loss $\mathcal{L}_{wm}$ to optimize the prediction of future world state. Future ego status requires supervision over driving command, velocity, and acceleration, including $\mathcal{L}_{cmd}=\text{CrossEntropy}(C_{pred},C_{gt})$, $\mathcal{L}_{v}=\text{MSE}(V_{pred},V_{gt})$ and $\mathcal{L}_{a}=\text{MSE}(A_{pred},A_{gt})$. The total loss for ego status is $\mathcal{L}_{ego}=\mathcal{L}_{cmd}+\mathcal{L}_{v}+\mathcal{L}_{a}$. The final loss for end-to-end training is:

$$
\displaystyle\mathcal{L}=\mathcal{L}_{traj}+\alpha\mathcal{L}_{align}+\beta\mathcal{L}_{wm}+\gamma\mathcal{L}_{ego}
$$

where $\alpha=0.1$, $\beta=0.2$, and $\gamma=0.1$.

#### 3.5.1 Inference.

During inference, only the Spatial-Aware Compressive Encoder and Trajectory Decoder is required, without any additional modules that would introduce inference latency.

## 4 Experiment

### 4.1 Implementation Details

#### 4.1.1 Architecture.

We adopt DINOv2-Base \[oquab2023dinov2\] as our vision encoder (86.6M parameters), which constitutes the foundation of the Spatial-Aware Compressive World Encoder (SCWE) and produces patch tokens of dimension $D_{e}=768$. We employ $N=16$ learnable scene queries with dimension $D_{e}$, projected to latent space $D_{l}=256$ via an MLP. We set the temporal horizon to $T=4$ frames and adopt three camera views—left, front, and right. Each view is resized to $224\times 448$ before feeding into the World Encoder. Notably, the geometric foundation model we employ is WorldMirror \[liu2025worldmirror\], a feed-forward model built upon VGGT \[wang2025vggt\]. The dimension of ground-truth geometric feature map is $D_{g}=2048$.

For the DLWM, we design a causal transformer decoder with 3D-RoPE for position embedding. The trajectory decoder follows a standard transformer architecture. Both DLWM and trajectory decoder comprise 4 layers with 8 attention heads, hidden dimension 256, and FFN dimension 1024. To provide ground truth world status representations for self-supervised training, we additionally introduce a frozen SCWE updated via Exponential Moving Average (EMA), maintaining complete architectural symmetry with the online encoder.

The model contains 104M parameters at inference time. During training, an additional EMA encoder is introduced for self-supervised learning, bringing the total to 191M, of which only 104M are trainable.

#### 4.1.2 Training Configuration.

Latent-WMA is trained on 32 A100 GPUs for 100 epochs with a total batch size of 512, taking approximately two days. We employ the AdamW \[loshchilov2019adamw\] optimizer with a learning rate of $2\times 10^{-4}$ and weight decay 0.05. The training schedule incorporates linear warm-up over the first 10% steps, followed by cosine annealing scheduling to $1\times 10^{-6}$ after reaching the peak. BF16 mixed-precision training is applied to reduce memory footprint.

### 4.2 Benchmarks & Main Results

#### 4.2.1 NAVSIM.

NAVSIM \[im2024navsim\] is a real-world autonomous driving dataset built upon OpenScene \[openscene2023\] and nuPlan \[nuplan\], comprising 103k training and 12k evaluation scenarios. NAVSIM v1 evaluates closed-loop planning using the Predictive Driver Model Score (PDMS), which aggregates No at-fault Collisions (NC), Drivable Area Compliance (DAC), Time to Collision (TTC), Ego Progress (EP), and Comfort (C). NAVSIM v2 \[cao2025navsimv2\] extends PDMS to EPDMS with additional metrics for rule compliance—Driving Direction Compliance (DDC), Traffic Light Compliance (TLC), Lane Keeping (LK)—and refines Comfort into History Comfort (HC) and Extended Comfort (EC).

The results on NAVSIM v2 are shown in Tab.˜1. Latent-WMA achieves the highest EPDMS among all methods, including those relying on perception annotations. Our method demonstrates strong rule compliance, with DDC, TLC and LK ranking among the top. Compared to Drive-JEPA \[wang2026drivejepa\], which additionally relies on perception annotations, our perception-free approach is slightly behind in safety metrics but achieves substantially better EC. Our EP (87.7) is slightly lower than Epona (88.6), likely because our safety-aware planning favors maintaining safer distances over aggressive ego progress.

Table 1: Comparison with state-of-the-art methods on the NAVSIM v2 with extended metrics. We indicate the best and second best with bold and underlined respectively. †: The reported results are dependent on perception-based annotation.

<table><tbody><tr><td>Method</td><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DDC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TLC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>LK <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>HC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EPDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td colspan="9">Perception-based Methods</td><td></td><td></td></tr><tr><td>TransFuser <cite>[transfuser]</cite></td><td>96.9</td><td>89.9</td><td>97.8</td><td>99.7</td><td>87.1</td><td>95.4</td><td>92.7</td><td>98.3</td><td>87.2</td><td>76.7</td></tr><tr><td>DriveSuprim <cite>[yao2025drivesuprim]</cite></td><td>97.5</td><td>96.5</td><td>99.4</td><td>99.6</td><td>88.4</td><td>96.6</td><td>95.5</td><td>98.3</td><td>77.0</td><td>83.1</td></tr><tr><td>ReCogDrive <cite>[li2025recogdrive]</cite></td><td>98.3</td><td>95.2</td><td>99.5</td><td>99.8</td><td>87.1</td><td>97.5</td><td>96.6</td><td>98.3</td><td>86.5</td><td>83.6</td></tr><tr><td>DiffusionDrive <cite>[diffdrive]</cite></td><td>98.2</td><td>95.9</td><td>99.4</td><td>99.8</td><td>87.5</td><td>97.3</td><td>96.8</td><td>98.3</td><td>87.7</td><td>84.5</td></tr><tr><td>WorldRFT <cite>[yang2025worldrft]</cite></td><td>97.8</td><td>96.5</td><td>99.5</td><td>99.8</td><td>88.5</td><td>97.0</td><td>97.4</td><td>98.1</td><td>69.1</td><td>86.7</td></tr><tr><td>Drive-JEPA† <cite>[wang2026drivejepa]</cite></td><td>98.4</td><td>98.6</td><td>99.1</td><td>99.8</td><td>88.4</td><td>97.8</td><td>97.6</td><td>97.9</td><td>84.8</td><td>87.8</td></tr><tr><td colspan="9">Perception-free Methods</td><td></td><td></td></tr><tr><td>World4Drive <cite>[zheng2025world4drive]</cite></td><td>97.8</td><td>96.3</td><td>99.4</td><td>99.8</td><td>88.3</td><td>97.1</td><td>97.7</td><td>98.0</td><td>53.9</td><td>84.8</td></tr><tr><td>Epona <cite>[zhang2025epona]</cite></td><td>97.1</td><td>95.7</td><td>99.3</td><td>99.7</td><td>88.6</td><td>96.3</td><td>97.0</td><td>98.0</td><td>67.8</td><td>85.1</td></tr><tr><td>DriveVLA-W0 <cite>[li2025drivevla-w0]</cite></td><td>98.5</td><td>99.1</td><td>98.0</td><td>99.7</td><td>86.4</td><td>98.1</td><td>93.2</td><td>97.9</td><td>58.9</td><td>86.1</td></tr><tr><td>Ours</td><td>98.1</td><td>97.3</td><td>99.6</td><td>99.8</td><td>87.7</td><td>97.3</td><td>97.6</td><td>98.1</td><td>87.3</td><td>89.3</td></tr></tbody></table>

#### 4.2.2 HUGSIM.

For closed-loop evaluation, we employ HUGSIM \[zhou2024hugsim\], a benchmark with scenarios from KITTI-360 \[kitti360\], nuScenes \[nuscenes\], PandaSet \[pandaset\], and Waymo \[waymo\]. These scenarios are reconstructed as photorealistic 3D environments where the planner controls the ego vehicle through RGB cameras with dynamically adjusted viewpoints.

Table 2: Photorealistic closed-loop evaluation on HUGSIM \[zhou2024hugsim\]. Zero-shot generalization using our model from the NAVSIM-v2 evaluation. Scores are per difficulty and overall average road completion (RC) and HD-Score, higher always better.

<table><tbody><tr><th rowspan="2">Method</th><th></th><td colspan="5">RC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td colspan="5">HD-Score <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><th></th><td>E</td><td>M</td><td>H</td><td>X</td><td>Avg.</td><td>E</td><td>M</td><td>H</td><td>X</td><td>Avg.</td></tr><tr><th>UniAD</th><th><cite>[hu2023uniad]</cite></th><td>58.6</td><td>41.2</td><td>40.4</td><td>26.0</td><td>40.6</td><td>48.7</td><td>29.5</td><td>27.3</td><td>14.3</td><td>28.9</td></tr><tr><th>VAD</th><th><cite>[jiang2023vad]</cite></th><td>38.7</td><td>27.0</td><td>25.5</td><td>23.0</td><td>27.9</td><td>24.3</td><td>9.9</td><td>10.4</td><td>8.2</td><td>12.3</td></tr><tr><th>LTF</th><th><cite>[chitta2022transfuser]</cite></th><td>68.4</td><td>40.7</td><td>36.9</td><td>25.5</td><td>41.4</td><td>52.8</td><td>24.6</td><td>19.8</td><td>8.1</td><td>24.8</td></tr><tr><th>GTRS-Dense</th><th><cite>[li2025gtrs]</cite></th><td>64.2</td><td>50.0</td><td>20.7</td><td>22.3</td><td>38.0</td><td>55.5</td><td>39.0</td><td>11.7</td><td>14.3</td><td>28.6</td></tr><tr><th>Ours</th><th></th><td>84.2</td><td>42.5</td><td>30.6</td><td>35.5</td><td>45.9</td><td>72.5</td><td>24.0</td><td>12.2</td><td>18.1</td><td>28.9</td></tr></tbody><tfoot><tr><th colspan="12">E: Easy, M: Medium, H: Hard, X: Extreme</th></tr></tfoot></table>

Tab.˜2 presents results on the pre-challenge HUGSIM test set, which contains 436 scenarios across four difficulty levels. Following the zero-shot protocol, we evaluate using Road Completion (RC) and the HUGSIM Driving Score (HD-Score)—the latter combining RC with averaged NC, DAC, TTC, and comfort metrics. We evaluate Latent-WMA in a zero-shot setting on HUGSIM: the model is trained exclusively on NAVSIM without any fine-tuning, yet achieves 45.9 RC and 28.9 HD-Score, ranking first in RC and matching the best HD-Score among all baselines.

### 4.3 Ablation Study

Table 3: Ablation study of each proposed component. + and - denote improvement/degradation relative to the baseline.

<table><tbody><tr><th colspan="2">Scene Representation</th><td colspan="2">Dynamic World Modeling</td><td rowspan="2">EPDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><th>Compression</th><th> Geometry</th><td>World Model</td><td> Ego Status</td></tr><tr><th>✗</th><th>✗</th><td>✗</td><td>✗</td><td>87.9</td></tr><tr><th>✓</th><th>✗</th><td>✗</td><td>✗</td><td>87.7 <math><semantics><msub><mrow><mo>−</mo> <mn>0.2</mn></mrow></msub> <annotation>\mathrlap{{}_{{\color[rgb]{0.82421875,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.82421875,0,0}-0.2}}}</annotation></semantics></math></td></tr><tr><th>✓</th><th>✗</th><td>✓</td><td>✗</td><td>88.0 <math><semantics><msub><mrow><mo>+</mo> <mn>0.1</mn></mrow></msub> <annotation>\mathrlap{{}_{{\color[rgb]{0,0.5703125,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.5703125,0}+0.1}}}</annotation></semantics></math></td></tr><tr><th>✓</th><th>✗</th><td>✓</td><td>✓</td><td>88.3 <math><semantics><msub><mrow><mo>+</mo> <mn>0.4</mn></mrow></msub> <annotation>\mathrlap{{}_{{\color[rgb]{0,0.5703125,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.5703125,0}+0.4}}}</annotation></semantics></math></td></tr><tr><th>✓</th><th>✓</th><td>✗</td><td>✗</td><td>88.6 <math><semantics><msub><mrow><mo>+</mo> <mn>0.7</mn></mrow></msub> <annotation>\mathrlap{{}_{{\color[rgb]{0,0.5703125,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.5703125,0}+0.7}}}</annotation></semantics></math></td></tr><tr><th>✓</th><th>✓</th><td>✓</td><td>✗</td><td>89.0 <math><semantics><msub><mrow><mo>+</mo> <mn>1.1</mn></mrow></msub> <annotation>\mathrlap{{}_{{\color[rgb]{0,0.5703125,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.5703125,0}+1.1}}}</annotation></semantics></math></td></tr><tr><th>✓</th><th>✓</th><td>✓</td><td>✓</td><td>89.3 <math><semantics><msub><mrow><mo>+</mo> <mn>1.4</mn></mrow></msub> <annotation>\mathrlap{{}_{{\color[rgb]{0,0.5703125,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.5703125,0}+1.4}}}</annotation></semantics></math></td></tr></tbody></table>

#### 4.3.1 Effectiveness of Each Component.

We conduct progressive ablation experiments by gradually adding modules to the baseline, as shown in Tab.˜3.

Scene Representation. The baseline directly feeds image patch tokens to the trajectory decoder, achieving 87.9 EPDMS. To enable long-horizon prediction in the world model, we compress image patches into compact scene tokens, incurring only a negligible performance drop ($\downarrow 0.2$). Injecting geometric information boosts performance to 88.6, demonstrating that geometric perception is crucial for accurate trajectory planning.

Dynamic World Modeling. Building upon compressed scene tokens, the DLWM improves performance from 87.7 to 88.0 by capturing future dynamics. Adding ego status further increases the score to 88.3, indicating that self-state awareness benefits planning. With geometric information, the world model further reaches 89.0. The full model combining all components achieves 89.3, showing that all modules contribute synergistically.

#### 4.3.2 Impact of Geometric Information.

We compare different approaches to incorporate geometric information, as shown in Tab.˜5.

Without geometric information, the full model (with compression, world model, and ego status) achieves 88.3 EPDMS. Directly concatenating frozen geometry features as key-value inputs surprisingly degrades performance to 88.0, likely due to misalignment between frozen features and the planning objective, introducing conflicting signals.

In contrast, distilling geometric knowledge into the vision backbone improves performance to 89.3. End-to-end fine-tuning enables the model to learn spatial-aware representations inherently aligned with downstream planning, leading to substantial gains over both alternatives.

#### 4.3.3 Vision Backbone for Geometric Distillation.

We compare different backbone scales and fine-tuning strategies for geometric distillation, as shown in Tab.˜5.

Table 4:  
Ablation study on geometry injection method

| Geometric Information | EPDMS $\uparrow$ |
| --- | --- |
| w/o Geometric feature | 88.3 |
| Concatenation | 88.0 |
| Distillation | 89.3 |

Table 5:  
Ablation study on vision backbone

| DINO Backbone | EPDMS $\uparrow$ |
| --- | --- |
| Small | 86.3 |
| Base | 89.3 |
| Small-LoRA | 84.7 |
| Base-LoRA | 68.5 |

Backbone Scale. DINO-Small achieves 86.3 EPDMS but remains suboptimal due to insufficient parameter capacity for high-dimensional geometric features. DINO-Base with full fine-tuning achieves the best performance (89.3), indicating that sufficient backbone capacity is essential for effective distillation.

Training Strategy. Parameter-efficient fine-tuning via LoRA leads to severe degradation and unstable training for both DINO-Small-LoRA (84.7) and DINO-Base-LoRA (68.5). LoRA’s low-rank constraints are inadequate for distilling high-dimensional geometric features, which require full parameter updates. Notably, DINO-Base-LoRA degrades more severely than DINO-Small-LoRA, as the larger model has more parameters frozen under LoRA, amplifying the mismatch between the low-rank update subspace and the high-dimensional distillation target.

In summary, effective geometric distillation requires both sufficient model capacity and full fine-tuning flexibility.

#### 4.3.4 World Model Prediction Temporal Stride.

We investigate the effect of prediction temporal stride in the DLWM, as shown in Tab.˜6.

Predicting only the final frame ($0\rightarrow 8$) achieves 88.4 EPDMS. Incorporating historical frames and intermediate future predictions with stride 4 ($-3\rightarrow 0\rightarrow 4\rightarrow 8$) improves performance to 89.3, as multi-step prediction provides richer supervision for representation learning.

Further increasing density to stride 2 ($-3\rightarrow-2\rightarrow\cdots\rightarrow 8$) yields 89.1, providing no further improvement over stride 4. Since all configurations predict up to the same horizon (8 frames), denser sampling does not extend the temporal reasoning range. Meanwhile, adjacent frames in driving scenes exhibit high similarity, providing limited additional learning signal. Moreover, optimizing a larger number of prediction targets may be less effective under the dense supervision setting. Additionally, denser prediction incurs higher computational overhead and longer training time.

We conclude that moderate prediction density balances supervision effectiveness and optimization efficiency.

Table 6: Ablation study on world model prediction temporal stride

| Prediction Temporal Stride | EPDMS $\uparrow$ |
| --- | --- |
| $0\rightarrow 8$ | 88.4 |
| $-3\rightarrow 0\rightarrow 4\rightarrow 8$ | 89.3 |
| $-3\rightarrow-2\rightarrow-1\rightarrow 0\rightarrow 2\rightarrow 4\rightarrow 6\rightarrow 8$ | 89.1 |

#### 4.3.5 Qualitative Analysis.

In this section, we qualitatively evaluate Latent-WMA on NAVSIM, analyzing both the trajectory planning performance and the spatial understanding capabilities of the World Encoder after geometric distillation.

Trajectory Comparison. As shown in Fig.˜4, we compare predicted trajectories of different world-model-based methods (green: human trajectories, yellow: prediction). Our method better aligns with the human trajectories and maintains safer distances from other vehicles. In contrast, Epona \[zhang2025epona\] produces relatively inferior trajectories, and World4Drive \[zheng2025world4drive\] yields acceptable but sub-optimal paths.

![[x4 26.png|Refer to caption]]

Figure 4: Visualization of planning trajectories, where the green line is the human trajectory, the yellow line is the predicted trajectory of corresponding method.

![[x5 24.png|Refer to caption]]

Figure 5: Visualization of attention maps between scene tokens and image patches. From top to bottom, the three groups correspond to going straight, turning right, and turning left respectively.

Spatial Understanding. To investigate how geometric distillation enhances spatial understanding, we visualize the cross-attention maps between scene tokens and image patches in Fig.˜5.

Compared to the baseline that only compresses images into scene tokens without geometric distillation, our model exhibits significantly more focused attention patterns, concentrating on lane markings, scene structures, and drivable areas critical for trajectory planning. The baseline, in contrast, produces scattered attention that allocates substantial weights to irrelevant background regions (*e.g*., sky, distant buildings), suggesting that without geometric supervision, compressed scene tokens tend to encode noisy information that interferes with downstream planning.

Furthermore, our attention maps demonstrate stronger alignment with the underlying geometric structure. While the baseline exhibits loose and diffuse attention regions, our World Encoder produces compact patterns that tightly follow geometric boundaries—such as lane markings and open spaces adjacent to obstacles.

We also visualize attention maps under different driving intentions, including going straight, turning right, and turning left. The attention distribution is strongly correlated with the driving intent: the model predominantly attends to regions along the intended direction, while areas deviating from the planned trajectory receive minimal attention. This intent-aware behavior demonstrates that our World Encoder learns to selectively focus on task-relevant spatial information conditioned on the driving context.

## 5 Conclusion

We presented Latent-WAM, an end-to-end autonomous driving framework that builds compact, planning-relevant world representations via spatial-aware compression and dynamic latent modeling. The Spatial-Aware Compressive World Encoder distills geometric knowledge from a foundation model into the vision backbone, compressing multi-view images into a small set of spatially-informed scene tokens. The Dynamic Latent World Model leverages a causal Transformer with 3D-RoPE to autoregressively predict future world status, acquiring dynamic understanding through self-supervised visual and supervised motion prediction. Latent-WAM achieves new state-of-the-art results on both NAVSIM v2 (89.3 EPDMS) and HUGSIM (45.9 RC and 28.9 HD-Score) with significantly less training data and fewer parameters than competing methods.

## References

## Supplementary Materials

## A. Metrics

### A.1. NAVSIM

NAVSIM v2 \[cao2025navsimv2\] extends the PDMS metric from NAVSIM v1 \[im2024navsim\] to the Extended PDMS (EPDMS):

$$
\text{EPDMS}=\text{NC}\times\text{DAC}\times\text{DDC}\times\text{TLC}\times\frac{5\times(\text{EP}+\text{TTC})+2\times(\text{LK}+\text{HC}+\text{EC})}{16}
$$

where NC, DAC, DDC, TLC, EP, TTC, LK, HC, and EC denote No at-fault Collision, Drivable Area Compliance, Driving Direction Compliance, Traffic Light Compliance, Ego Progress, Time to Collision, Lane Keeping, History Comfort, and Extended Comfort, respectively. Among these, DDC, TLC, LK, HC, and EC are newly introduced in NAVSIM v2.

### A.2. HUGSIM

The HUGSIM \[zhou2024hugsim\] Driving Score (HD-Score) at timestamp $t$ is computed as the product of driving policy items and a weighted average of contributory items:

$$
\text{HD-Score}_{t}=\text{NC}\times\text{DAC}\times\frac{5\times\text{TTC}+2\times\text{COM}}{7}
$$

where NC, DAC, TTC, and COM denote No at-fault Collision, Drivable Area Compliance, Time to Collision, and Ego Comfort, respectively. These metrics follow the same definitions as used in NAVSIM v1.

The final HD-Score averages per-frame scores across all timestamps and scales by the route completion score $R_{c}\in[0,1]$:

$$
\text{HD-Score}=R_{c}\times\frac{1}{T}\sum_{t=0}^{T}\text{HD-Score}_{t}
$$

Notably, NC and TTC account for collisions with static background entities (e.g., buildings, fences, vegetation) using semantic segmentation.

## B. Data Processing Pipeline

### B.1. Image Preprocessing

We utilize three camera views (left, front, right) as input. Each image undergoes a two-stage preprocessing pipeline: resizing followed by center cropping. Specifically, the original $1920\times 1080$ images are first resized to $455\times 256$, then center-cropped to the final resolution of $448\times 224$, yielding an aspect ratio of 2:1.

### B.2. Camera Intrinsic Adjustment

When applying geometric feature extraction to images, the camera intrinsic matrix must be adjusted accordingly to maintain geometric consistency. Given the original intrinsic matrix $\mathbf{K}$:

$$
\mathbf{K}=\begin{bmatrix}f_{x}\quad&0\quad&c_{x}\\
0\quad&f_{y}\quad&c_{y}\\
0\quad&0\quad&1\end{bmatrix},
$$

where $f_{x}$ and $f_{y}$ denote the focal lengths along the $x$ and $y$ axes, and $(c_{x},c_{y})$ represents the principal point. After resizing from $(W_{\text{orig}},H_{\text{orig}})$ to $(W_{\text{resize}},H_{\text{resize}})$, the scale factors are computed as:

$$
s_{x}=\frac{W_{\text{resize}}}{W_{\text{orig}}},\quad s_{y}=\frac{H_{\text{resize}}}{H_{\text{orig}}}.
$$

For center cropping to $(W_{\text{crop}},H_{\text{crop}})$, the crop offsets are:

$$
\Delta_{x}=\frac{W_{\text{resize}}-W_{\text{crop}}}{2},\quad\Delta_{y}=\frac{H_{\text{resize}}-H_{\text{crop}}}{2}.
$$

The adjusted intrinsic parameters are then:

$$
f^{\prime}_{x}=f_{x}\cdot s_{x},\quad f^{\prime}_{y}=f_{y}\cdot s_{y},\quad c^{\prime}_{x}=c_{x}\cdot s_{x}-\Delta_{x},\quad c^{\prime}_{y}=c_{y}\cdot s_{y}-\Delta_{y},
$$

yielding the adjusted intrinsic matrix $\mathbf{K}^{\prime}$:

$$
\mathbf{K}^{\prime}=\begin{bmatrix}f^{\prime}_{x}\quad&0\quad&c^{\prime}_{x}\\
0\quad&f^{\prime}_{y}\quad&c^{\prime}_{y}\\
0\quad&0\quad&1\end{bmatrix}.
$$

### B.3. Camera Extrinsic Transformation

The camera extrinsic matrix represents the transformation from camera coordinates to world (LiDAR) coordinates. We construct the camera-to-world matrix $\mathbf{T}_{c\to w}\in\mathbb{R}^{4\times 4}$ as:

$$
\mathbf{T}_{c\to w}=\begin{bmatrix}\mathbf{R}_{c\to w}\quad&\mathbf{t}_{c\to w}\\
\mathbf{0}^{\top}\quad&1\end{bmatrix},
$$

where $\mathbf{R}_{c\to w}\in\mathbb{R}^{3\times 3}$ is the rotation matrix and $\mathbf{t}_{c\to w}\in\mathbb{R}^{3}$ is the translation vector. The world-to-camera matrix $\mathbf{T}_{w\to c}$ used for 3D-to-2D projection is obtained by matrix inversion:

$$
\mathbf{T}_{w\to c}=\mathbf{T}_{c\to w}^{-1}.
$$

### B.4. Geometric Feature Extraction

Following the geometric alignment formulation in the Method section, we extract patch-level geometric features $f_{g}(I)\in\mathbb{R}^{T\times M\times S\times D_{g}}$ from multi-view images using the frozen geometric foundation model WorldMirror \[liu2025worldmirror\] as $f_{g}$, where $T$ denotes the number of temporal frames, $M$ denotes the number of camera views, $S$ denotes the number of patch tokens per view, and $D_{g}=2048$ denotes the geometric feature dimension.

Camera Prior Extraction. Given multi-view images $\mathbf{I}\in\mathbb{R}^{M\times 3\times H\times W}$ along with their camera intrinsics $\mathbf{K}\in\mathbb{R}^{M\times 3\times 3}$ and extrinsics $\mathbf{T}\in\mathbb{R}^{M\times 4\times 4}$, the geometric foundation model first encodes these inputs into spatial-geometric priors $\mathcal{P}$ that encapsulate camera poses, depth estimates, and intrinsic parameters:

$$
\mathcal{P}=f_{\text{prior}}\left(\mathbf{I},\mathbf{K},\mathbf{T}\right).
$$

Geometric Feature Encoding. The visual geometry transformer then processes the images conditioned on these priors. Following the model’s conditional design, we utilize camera pose and intrinsic information (indicated by condition flags $\mathbf{c}=[1,0,1]$ for camera pose, depth, and intrinsics respectively) while excluding depth supervision:

$$
f_{g}(I)=\text{WorldMirror}\left(\mathbf{I},\mathcal{P};\mathbf{c}\right).
$$

Since $f_{g}$ remains frozen throughout training, we pre-compute and offline-cache $f_{g}(I)$ for all training samples, enabling direct loading during training without repeated inference costs. This design substantially reduces training time and GPU memory overhead.

## C. Inference Latency

We measured the inference latency per module of Latent-WMA on a single A100 GPU for a single batch. The inference latency of each module is shown in Tab.˜7. Latency was evaluated after three warm-up iterations, with the final number averaged over 10 forward passes.

Table 7: Per-module inference latency of Latent-WMA

| Module | Parameters |  | Memory Usage |  | Latency |
| --- | --- | --- | --- | --- | --- |
| World Encoder | 86.6M |  | — |  | 100ms |
| Trajectory Decoder | 8.4M |  | — |  | 6ms |
| All modules | 104M |  | 1.1GB |  | 107ms |

## D. More Visulization

### D.1. Trajectory Planning

![[Uncaptioned image]](https://arxiv.org/html/2603.24581v1/x6.png)

Figure 6: Additional trajectory visualizations on NAVSIM. Green: human driving trajectory. Yellow: predicted trajectory from the corresponding method.

### D.2. Attention Map

Figure 7: Visualization of attention maps between scene tokens and image patches across consecutive frames. All groups depict going-forward scenarios. Geometric distillation yields more focused attention on task-relevant regions compared to the baseline.

Figure 8: Visualization of attention maps between scene tokens and image patches across consecutive frames. All groups depict lane-changing scenarios. Geometric distillation yields more focused attention on task-relevant regions compared to the baseline.

Figure 9: Visualization of attention maps between scene tokens and image patches across consecutive frames. All groups depict left-turn scenarios. Geometric distillation yields more focused attention on task-relevant regions compared to the baseline.

Figure 10: Visualization of attention maps between scene tokens and image patches across consecutive frames. All groups depict right-turn scenarios. Geometric distillation yields more focused attention on task-relevant regions compared to the baseline.

#### 0..0.1 C.3. HUGSIM

Figure 11: Visualization of trajectory planning on HUGSIM benchmark(nuScenes \[nuscenes\])

Figure 12: Visualization of trajectory planning on HUGSIM benchmark(KITTI-360 \[kitti360\])

Figure 13: Visualization of trajectory planning on HUGSIM benchmark(KITTI-360)

Figure 14: Visualization of trajectory planning on HUGSIM benchmark(Waymo \[waymo\])

Figure 15: Visualization of trajectory planning on HUGSIM benchmark(Pandaset \[pandaset\])
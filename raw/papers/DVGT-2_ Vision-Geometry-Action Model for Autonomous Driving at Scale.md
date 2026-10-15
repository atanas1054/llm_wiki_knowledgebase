---
title: "DVGT-2: Vision-Geometry-Action Model for Autonomous Driving at Scale"
source: "https://arxiv.org/html/2604.00813v3"
author:
published:
created: 2026-10-14
description:
tags:
  - "clippings"
---
Sicheng Zuo <sup>1,∗</sup>    Zixun Xie <sup>1,2,4,∗</sup> Affiliation: Tsinghua University  Xiaomi EV  University of Macau  Peking University    Wenzhao Zheng <sup>1,∗,†</sup>    Shaoqing Xu <sup>2,†</sup>    Fang Li <sup>2</sup>    Hanbing Li <sup>2</sup>    Long Chen <sup>2</sup>    Zhi-Xin Yang <sup>3</sup>    Jiwen Lu <sup>1</sup> Affiliation: Project Page: [https://wzzheng.net/DVGT-2](https://wzzheng.net/DVGT-2) Affiliation: Large Driving Models: [https://github.com/wzzheng/LDM](https://github.com/wzzheng/LDM)

###### Abstract

End-to-end autonomous driving has evolved from the conventional paradigm based on sparse perception into vision-language-action (VLA) models, which focus on learning language descriptions as an auxiliary task to facilitate planning. In this paper, we propose an alternative Vision-Geometry-Action (VGA) paradigm that advocates dense 3D geometry as the critical cue for autonomous driving. As vehicles operate in a 3D world, we think dense 3D geometry provides the most comprehensive information for decision-making. However, most existing geometry reconstruction methods (e.g., DVGT) rely on computationally expensive batch processing of multi-frame inputs and cannot be applied to online planning. To address this, we introduce a streaming Driving Visual Geometry Transformer (DVGT-2), which processes inputs in an online manner and jointly outputs dense geometry and trajectory planning for the current frame. We employ temporal causal attention and cache historical features to support on-the-fly inference. To further enhance efficiency, we propose a sliding-window streaming strategy and use historical caches within a certain interval to avoid repetitive computations. Despite the faster speed, DVGT-2 achieves superior geometry reconstruction performance on various datasets. The same trained DVGT-2 can be directly applied to planning across diverse camera configurations without fine-tuning, including closed-loop NAVSIM and open-loop nuScenes benchmarks. Code is available at [https://github.com/wzzheng/DVGT](https://github.com/wzzheng/DVGT).

![[teaser 4.png|Refer to caption]]

Figure 1: DVGT-2 is a streaming visual geometry transformer specifically designed for autonomous driving. It inputs multi-view images and jointly predicts the 3D pointmaps, the ego poses, and future trajectory planning in an online manner.

<sup>†</sup>

## 1 Introduction

End-to-end autonomous driving has recently seen remarkable progress, fundamentally changing how vehicles perceive and navigate complex environments [^16] [^26] [^3] [^4] [^41]. Conventional methods typically rely on sparse perception representations (e.g., 3D object detection [^39] [^18] [^43] [^44] [^45] [^77] and map segmentation [^39] [^48] [^40]) to guide scene understanding and trajectory planning. Recently, the emerging Vision-Language-Action (VLA) models leverage the general understanding and reasoning capabilities of pre-trained Vision-Language Models (VLMs), focusing on learning natural language descriptions to interpret driving scenarios and facilitate robust decision-making [^70] [^23] [^90] [^37] [^53].

As vehicles inherently operate in a 3D world, we argue that understanding the dense 3D geometry of the environment provides the most direct and comprehensive information for safe decision-making. Although language descriptions offer valuable high-level context, they can sometimes be too coarse to capture complete and accurate geometric details when precise spatial control is required. Considering this, we explore an alternative Vision-Geometry-Action (VGA) paradigm for end-to-end autonomous driving. Rather than rely on sparse perception representations or coarse language descriptions, VGA advocates for dense 3D geometry as the foundational representation for driving. By explicitly recovering pixel-aligned 3D pointmaps, VGA extracts comprehensive and precise geometric cues from visual inputs, directly empowering trajectory planning.

However, integrating high-fidelity geometry reconstruction into an end-to-end planning framework presents significant computational challenges. Existing methods [^66] [^73] [^93] typically rely on computationally expensive batch processing of the entire multi-frame sequence. When processing online inputs frame-by-frame, they have to recalculate features for overlapping historical frames at every time step. Such redundant computation incurs unacceptable latency and makes them unsuitable for online, real-time autonomous driving. To address this, we introduce a streaming Driving Visual Geometry Transformer (DVGT-2), designed to jointly output dense geometry and trajectory planning in an online manner, as shown in Fig. 1. Instead of reprocessing the entire historical sequence, DVGT-2 employs temporal causal attention and caches historical intermediate features to support efficient on-the-fly inference. To further enhance efficiency and bound the computational overhead for continuous driving, we propose a sliding-window streaming strategy that only utilizes historical caches within a fixed interval. We achieve this by introducing relative temporal positional encoding and jointly predicting dense points in the local coordinate system alongside the ego-pose relative to the previous frame. This design allows the model to continuously aggregate historical geometric cues and avoid repetitive computations.

To build a robust foundation model for the VGA paradigm, we train DVGT-2 on a large mixture of diverse driving datasets, including nuScenes [^1], OpenScene [^5], Waymo [^62], KITTI [^10], and DDAD [^11]. Extensive experiments demonstrate that, despite its significantly faster inference speed, our model achieves superior geometry reconstruction performance on various datasets. More importantly, the same trained DVGT-2 can be directly applied to trajectory planning across diverse datasets without the need for finetuning, achieving strong performance on both the closed-loop NAVSIM [^6] [^2] and open-loop nuScenes [^1] benchmarks, validating the effectiveness of the VGA paradigm.

## 2 Related Work

End-to-end Autonomous Driving. End-to-end autonomous driving aims to directly map raw sensor inputs to planning trajectories or control signals, jointly optimizing the entire system to minimize error accumulation. Pioneering works like UniAD [^16] and VAD [^26] proposed the paradigm by integrating perception, prediction, and planning into a single framework, achieving planning-oriented joint optimization across all tasks. Subsequent research shifted towards multi-modal trajectory generation. VADv2 [^3] and Hydra-MDP [^38] modeled probabilistic planning by sampling from a fixed trajectory codebook. DiffusionDrive [^41] proposed an anchor-based truncated diffusion model to capture the multi-modal distribution of trajectories, which is adopted by other methods [^32] [^28] [^76]. Despite these advances, existing methods [^86] [^87] [^3] fundamentally depend on manually defined perception tasks, such as detection [^39] [^18] [^36] [^43], tracking [^48] [^45] [^91] [^69], or occupancy [^20] [^21] [^19] [^95] [^94] to understand driving scenes, which is extremely inefficient with world information. In contrast, our model explicitly reconstructs fine-grained dense geometry. By comprehensively modeling the geometric details, our approach facilitates robust trajectory planning.

VLA for Autonomous Driving. The extensive world knowledge of Vision-Language Models (VLMs) have driven their applications in autonomous driving [^60] [^22] [^79], mainly focus on scene understanding and reasoning. Subsequent research utilized VLMs to predict high-level meta-actions or driving decisions [^27] [^25] [^72], which serve as intermediate guidance [^25] [^42] or supervision [^55] [^78] [^12] [^46] for downstream planners or end-to-end models. Although these methods facilitate the integration of VLM knowledge, the modular design hinders full end-to-end optimization. Recent advancements have integrated planning modules into the VLMs to map sensor inputs to planning trajectories directly. Initial attempts sought to predict trajectories directly in text format [^70] [^23] [^51] [^54] [^84] [^53] [^52], while other methods explored incorporating specialized trajectory decoders to predict feasible actions, including the auto-regressive manner [^89] [^24] [^90], the diffusion module [^81] [^9] [^37], or the MLP head [^58] [^59] [^47]. In contrast, we propose a Vision-Geometry-Action (VGA) framework, as shown in Fig. 2. By explicitly modeling fine-grained dense geometry, our approach connects visual inputs to driving actions through high-fidelity structural and dynamic scene understanding, thereby enhancing trajectory planning.

Visual Geometry Models. Recent advances in general visual geometry models have shifted the paradigm from depth estimation to regressing dense pointmaps directly from unposed images. Pioneering works like DUSt3R [^71] and its extensions [^31] [^65] [^75] have demonstrated impressive capabilities in recovering 3D structure from image pairs. More recently, large-scale geometry foundation models like VGGT [^66] and $\pi^{3}$ [^73] have achieved robust geometry reconstruction across multi-view inputs. However, these methods are limited to a relative scale and require post-alignment with sparse LiDAR points to recover the metric-scaled scene geometry. To address this, DVGT [^93] proposed a driving visual geometry model to enable end-to-end reconstruction of driving scene geometry with metric scale. But it relies on batch processing of multi-frame inputs, which leads to redundant computations for historical frames. While streaming alternatives like CUT3R [^67] and StreamVGGT [^92] attempt to reduce latency, they typically maintain a feature cache that grows linearly with input length, making them unsuitable for infinite-length driving scenarios. In contrast, we propose a window-based streaming architecture by maintaining a fixed-size historical cache and predicting local geometry only for the latest frame. Our approach thus ensures constant computational costs and supports efficient, long-term inference.

## 3 Proposed Approach

![[e2e_vla_VGA.png|Refer to caption]]

Figure 2: Comparison of different paradigms for end-to-end autonomous driving. Conventional end-to-end models rely on sparse perception representations for scene understanding. VLA models predict language descriptions to interpret driving scenarios. Our VGA model reconstructs dense 3D geometry to facilitate safe planning.

### 3.1 Vision-Geometry-Action Model

The fundamental objective of an autonomous driving system is to predict a safe and feasible future trajectory for the ego vehicle given historical and current sensor observations. Formally, at the current time step $t$, given a sequence of multi-view image inputs $\mathbf{I}_{t-T:t}$ from the past $T$ frames and the current frame, the driving model $\mathcal{M}$ aims to predict the future ego-trajectory $\mathbf{A}_{t}$:

$$
\mathbf{A}_{t}=\mathcal{M}(\mathbf{I}_{t-T:t}).
$$

Conventional end-to-end models typically decompose this process into a sequential perception-prediction-planning pipeline [^16] [^26] [^87]. The perception module $\mathcal{F}_{\text{perc}}$ first extracts high-level perceptual representations $\mathbf{Z}$ from the visual inputs. The prediction module $\mathcal{F}_{\text{pred}}$ then forecasts the future motion $\mathbf{V}$ of surrounding agents. Finally, the planning module $\mathcal{F}_{\text{plan}}$ utilizes these representations to generate the future ego-trajectory. This process can be expressed as:

$$
\displaystyle\mathbf{Z}=\mathcal{F}_{\text{perc}}(\mathbf{I}_{t-T:t}),\ \ \mathbf{V}=\mathcal{F}_{\text{pred}}(\mathbf{Z}),\ \ \mathbf{A}_{t}=\mathcal{F}_{\text{plan}}(\mathbf{Z},\mathbf{V}).
$$

However, this paradigm heavily relies on sparse perceptual representations like bounding boxes and map elements, which discard rich environmental context. While some methods introduce 3D occupancy to provide a dense structural description, they inherently suffer from quantization errors due to voxel discretization. Consequently, this incomplete and inaccurate modeling of scene structures fundamentally restricts the model’s planning performance.

Recently, Vision-Language-Action (VLA) models have emerged as a promising alternative, leveraging the strong semantic understanding capabilities of pre-trained Vision-Language-Models (VLMs) to facilitate driving [^70] [^23] [^88] [^90]. Given historical observations, a VLA model $\mathcal{M}_{\text{VLA}}$ typically outputs a textual description $\mathbf{L}_{t}$ of the current scene alongside the future trajectory prediction $\mathbf{A}_{t}$:

$$
\mathbf{A}_{t},\mathbf{L}_{t}=\mathcal{M}_{\text{VLA}}(\mathbf{I}_{t-T:t}).
$$

Although VLA models demonstrate remarkable generalization, natural language is inherently ambiguous and coarse-grained. It struggles to accurately and comprehensively capture the precise geometric details of driving scenes, thereby hindering high-fidelity scene understanding and robust trajectory planning.

In this paper, we propose a novel Vision-Geometry-Action (VGA) framework, which identifies dense geometry as the critical bridge connecting visual inputs to driving actions. Given multi-frame image inputs, our VGA model $\mathcal{M}_{\text{VGA}}$ jointly reconstructs the dense 3D pointmaps $\mathbf{P}_{t-T:t}$ and ego-poses $\mathbf{E}_{t-T:t}$, and predicts the future ego-trajectory $\mathbf{A}_{t}$:

$$
\mathbf{A}_{t},\mathbf{P}_{t-T:t},\mathbf{E}_{t-T:t}=\mathcal{M}_{\text{VGA}}(\mathbf{I}_{t-T:t}).
$$

This paradigm offers two fundamental advantages. First, the continuous coordinate space of dense pointmaps eliminates quantization errors, providing a pixel-aligned, complete representation of both foreground objects and background environments. Second, by explicitly modeling spatial geometry and camera poses across multi-frame inputs, the VGA model comprehensively captures temporally consistent static structures and coherent dynamic motions, thereby providing a more precise and reliable foundation for trajectory planning.

### 3.2 Streaming Geometry Reconstruction

![[streaming.png|Refer to caption]]

Figure 3: Comparison of different paradigms for geometry reconstruction. Batch-processing models like DVGT 93 compute pair-wise relations across all frames, incurring an overall 𝒪 ⁡ ( T 2 ) \\mathcal{O}(T^{2}) complexity. Full-history streaming models like StreamVGGT 92 extract temporal cues from the entire history, leading to an \\mathcal{O}(T) per-frame complexity. In contrast, our sliding-window streaming strategy attends to a fixed-size cache of length W, achieving a constant \\mathcal{O}(W) per-frame complexity.

In the VGA framework, dense geometry reconstruction from multi-frame inputs is essential for comprehensive scene understanding. Recently, numerous works [^66] [^73] [^93] have made progress by adopting a batch-processing paradigm. Given a sequence of image inputs $\mathbf{I}_{t-T:t}$, these models jointly reconstruct global pointmaps and ego-poses for all frames, typically anchored to the first frame’s coordinate system:

$$
\mathbf{P}_{t-T:t},\mathbf{E}_{t-T:t}=\mathcal{G}_{\text{batch}}(\mathbf{I}_{t-T:t}),
$$

where $\mathcal{G}_{\text{batch}}$ denotes batch-processing reconstruction paradigm. However, this paradigm requires computing pairwise spatial interactions across all input frames, resulting in an $\mathcal{O}(T^{2})$ computational complexity. More severely, when processing online inputs frame-by-frame in driving scenarios, the model redundantly reprocesses the overlapping historical frames at every time step. This massive computational redundancy leads to prohibitive inference latency, making it entirely unsuited for online, real-time autonomous driving applications [^93].

To alleviate this redundant computation, recent works like StreamVGGT [^92] introduce a full-history streaming reconstruction framework with a feature caching mechanism. When receiving a new frame input $\mathbf{I}_{t}$, the model $\mathcal{G}_{\text{stream}}$ only computes the interaction between the current frame and the cached historical features $\mathbf{C}_{t-T:t-1}$, thereby reducing the computational complexity from $\mathcal{O}(T^{2})$ to $\mathcal{O}(T)$. After the prediction, the current frame’s features are updated into the cache for the next-frame prediction. This process can be formulated as:

$$
\mathbf{P}_{t},\mathbf{E}_{t},\mathbf{C}_{t-T:t}=\mathcal{G}_{\text{stream}}([\mathbf{I}_{t},\mathbf{C}_{t-T:t-1}]).
$$

Although this framework avoids repeatedly computing historical frames, it still relies on the first frame as the global reference coordinate system. Consequently, the model must retain features of the entire history. This causes memory and computational costs that scale linearly with the sequence length, rendering the approach prohibitive for continuous, infinite-horizon driving scenarios.

To overcome these bottlenecks, we propose a sliding-window streaming strategy, as shown in Fig. 3. The core idea is to maintain a historical feature cache $\mathcal{C}_{t-W:t-1}$ with a fixed window size of $W$, ensuring a constant $\mathcal{O}(W)$ per-frame complexity. Crucially, to eliminate the dependency on the full historical sequence, we decouple the geometry reconstruction from the first-frame coordinate system. Instead, we reconstruct the local geometry $\mathbf{P}_{t}$ in the current frame’s ego-coordinate system and predict the ego-pose $\mathbf{E}_{t}$ relative to the previous frame. Our sliding-window streaming inference process can be formulated as:

$$
\mathbf{P}_{t},\mathbf{E}_{t},\mathbf{C}_{t-W+1:t}=\mathcal{G}_{\text{window}}([\mathbf{I}_{t},\mathbf{C}_{t-W:t-1}]),
$$

where $\mathcal{G}_{\text{window}}$ denotes sliding-window streaming reconstruction paradigm. After processing the current frame, the cache is updated in a First-In-First-Out (FIFO) manner, discarding the earliest frame’s features $\mathbf{C}_{t-W}$ and appending the current frame’s features $\mathbf{C}_{t}$. Our sliding-window streaming framework can efficiently process arbitrary-length video streams with constant overhead, adhering to the strict efficiency constraints of real-time autonomous driving systems.

### 3.3 Streaming Driving Visual Geometry Transformer

![[framework 3.png|Refer to caption]]

Figure 4: Overall archetecture of DVGT-2. Our model consists of an image encoder, a geometry transformer with temporal causal attention, and a set of prediction heads to jointly output geometry reconstruction and trajectory planning.

To realize the sliding-window streaming paradigm for efficient geometry reconstruction and trajectory planning, we propose DVGT-2, a Streaming Driving Visual Geometry Transformer. At time step $t$, given multi-view image inputs $\mathbf{I}_{t}\in\mathbb{R}^{V\times H\times W\times 3}$ from $V$ cameras and a historical feature cache $\mathbf{C}_{t-W:t-1}$ containing the past $W$ frames, the model predicts three components: (1) the multi-view 3D pointmaps $\mathbf{P}_{t}\in\mathbb{R}^{V\times H\times W\times 3}$ in the current ego-coordinate system; (2) the current ego-pose $\mathbf{E}_{t}\in\mathbb{R}^{7}$ (comprising a 3D translation and a 4D rotation quaternion) relative to the previous frame; (3) the future $N$ -step planning trajectory $\mathbf{A}_{t}\in\mathbb{R}^{N\times 3}$ (representing x, y coordinates, and yaw angle). The cache is then updated to the current frame. The overall process can be expressed as:

$$
\mathbf{A}_{t},\mathbf{P}_{t},\mathbf{E}_{t},\mathbf{C}_{t-W+1:t}=\mathcal{M}_{\text{DVGT-2}}(\mathbf{I}_{t},\mathbf{C}_{t-W:t-1}).
$$

Overall Architecture. As shown in Fig. 4, our DVGT-2 consists of an image encoder $\mathcal{E}$, a geometry transformer $\mathcal{G}$, and task-specific prediction heads $[\mathcal{H}^{\text{vis}},\mathcal{H}^{\text{pose}},\mathcal{H}^{\text{traj}}]$. Given the multi-view images $\mathbf{I}_{t}$ of the current time step $t$, we first employ a pre-trained vision foundation model [^61] to extract visual tokens:

$$
\mathbf{F}_{t}^{\text{vis}}=\mathcal{E}(\mathbf{I}_{t}).
$$

To explicitly aggregate global representations for spatial reasoning and planning, we append learnable pose tokens $\mathbf{F}_{t}^{\text{pose}}$ and trajectory tokens $\mathbf{F}_{t}^{\text{traj}}$ to the visual tokens of each view, forming the unified input tokens:

$$
\mathbf{F}_{t}=[\mathbf{F}_{t}^{\text{vis}},\mathbf{F}_{t}^{\text{pose}},\mathbf{F}_{t}^{\text{traj}}].
$$

Subsequently, these combined tokens, along with the historical cache $\mathbf{C}_{t-W:t-1}$, are fed into the geometry transformer $\mathcal{G}$ for spatial-temporal reasoning:

$$
\mathbf{G}_{t}^{\text{vis}},\mathbf{G}_{t}^{\text{pose}},\mathbf{G}_{t}^{\text{traj}}=\mathcal{G}(\mathbf{F}_{t},\mathbf{C}_{t-W:t-1}).
$$

Finally, the updated tokens are routed to their respective prediction heads to generate the 3D pointmaps $\mathbf{P}_{t}$, ego-pose $\mathbf{E}_{t}$, and future trajectory $\mathbf{A}_{t}$:

$$
\mathbf{P}_{t}=\mathcal{H}^{\text{vis}}(\mathbf{G}_{t}^{\text{vis}}),\ \ \mathbf{E}_{t}=\mathcal{H}^{\text{pose}}(\mathbf{G}_{t}^{\text{pose}}),\ \ \mathbf{A}_{t}=\mathcal{H}^{\text{traj}}(\mathbf{G}_{t}^{\text{traj}}).
$$

The overall pipeline of our model’s efficient online inference is illustrated in Fig. 5.

![[infer.png|Refer to caption]]

Figure 5: Efficient inference of DVGT-2. Given the current frame multi-view input and the cache of past W frames, our model performs efficient geometry reconstruction and trajectory planning in an online manner, avoiding recomputing historical frames.

Geometry Transformer. Following DVGT [^93], we utilize a factorized attention mechanism to efficiently model the complex spatial-temporal relations across multiple views and frames. The geometry transformer $\mathcal{G}$ comprises $L$ cascaded blocks, each executing three sequential attention operations:

- Intra-View Local Attention facilitates fine-grained token interactions within individual images.
- Cross-View Spatial Attention enables spatial reasoning across the $V$ views of the current frame.
- Temporal Causal Attention performs temporal aggregation between the current frame and the historical cache.

In the Temporal Causal Attention module, the current frame’s tokens serve as the query, while the cached features of past $W$ historical frames act as the keys and values. Crucially, to support infinite-length streaming inference, we discard conventional absolute temporal positional encoding and adopt MRoPE-I [^17] for relative temporal positional encoding instead. This design ensures that the cached historical features remain invariant over time, allowing them to be reused directly for future interactions without recomputation. After processing the current frame, the cache is updated following a First-In-First-Out (FIFO) principle. The earliest frame’s features are discarded, and the current intermediate features $\mathbf{\hat{G}}_{t}$ of each transformer layer are pushed into the cache:

$$
\mathbf{C}_{t-W+1:t}=\text{FIFO}(\mathbf{C}_{t-W:t-1},\mathbf{\hat{G}}_{t}),
$$

This sliding-window mechanism ensures that the model avoids redundant computation for historical frames, maintaining a constant and fast inference speed.

Prediction Heads. We employ three specialized heads to decode the output representations of the geometry transformer. The visual tokens are processed by a DPT head [^56] $\mathcal{H}^{vis}$ to recover the dense 3D pointmaps $\mathbf{P}_{t}$. For pose and trajectory prediction, we first aggregate the corresponding tokens across all views. Inspired by DiffusionDrive [^41], we then employ two anchor-based diffusion heads $\mathcal{H}^{\text{pose}}$ and $\mathcal{H}^{\text{traj}}$ to model the prior distributions of the ego-pose and ego-trajectory, respectively. By leveraging a truncated diffusion strategy, these heads robustly decode the relative pose $\mathbf{E}_{t}$ and the future driving trajectory $\mathbf{A}_{t}$.

## 4 Experiments

### 4.1 Dataset

We conduct our training and evaluation on a large-scale mixed driving dataset comprising multi-view video sequences sampled at 2 Hz from five sources: nuScenes (700 train / 150 test scenes), OpenScene (19,736 train / 2,026 test scenes), Waymo (798 train / 202 test scenes), KITTI (138 train / 13 test scenes), and DDAD (150 train / 50 test scenes). To obtain high-quality geometric supervision, we follow DVGT [^93] and employ the depth foundation model MoGe-2 [^68] to infer dense depth maps, followed by a threshold-filtering operation to ensure data quality. Finally, to comprehensively assess planning performance, we evaluate our model on the closed-loop NAVSIM v1 and v2 benchmarks [^6] [^2] (curated subsets of OpenScene), as well as the standard open-loop nuScenes benchmark.

Table 1: Quantitative 3D geometry reconstruction results on OpenScene [^5]. \* denotes performing post-alignment with sparse LiDAR to recover the metric scale.

| Method | Paradigm | Acc $\downarrow$ | Comp $\downarrow$ | Abs Rel $\downarrow$ | $\delta$ < $1.25\uparrow$ | AUC $\uparrow$ | Time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VGGT\* [^66] | Full-Seq. | 1.705 | 1.711 | 0.280 | 0.669 | 74.4 | $\sim$ 5.31s |
| MapAnything [^29] | Full-Seq. | 3.269 | 4.214 | 0.476 | 0.253 | 69.5 | $\sim$ 2.28s |
| DVGT [^93] | Full-Seq. | 0.412 | 0.491 | 0.048 | 0.971 | 76.6 | $\sim$ 1.88s |
| CUT3R\* [^67] | Streaming | 1.858 | 2.245 | 0.275 | 0.596 | 36.9 | $\sim$ 0.35s |
| StreamVGGT\* [^92] | Streaming | 2.209 | 2.060 | 0.303 | 0.620 | 74.1 | $\sim$ 1.94s |
| Driv3R\* [^7] | Streaming | 0.884 | 1.693 | 0.188 | 0.740 | \- | $\sim$ 0.56s |
| DVGT-2 | Streaming | 0.440 | 0.450 | 0.040 | 0.977 | 70.3 | $\sim$ 0.27s |

Table 2: Quantitative 3D geometry reconstruction results on nuScenes [^1].

| Method | Paradigm | Acc $\downarrow$ | Comp $\downarrow$ | Abs Rel $\downarrow$ | $\delta$ < $1.25\uparrow$ | AUC $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| VGGT\* [^66] | Full-Seq. | 1.944 | 2.071 | 0.348 | 0.573 | 83.9 |
| MapAnything [^29] | Full-Seq. | 4.447 | 4.860 | 0.562 | 0.271 | 84.9 |
| DVGT [^93] | Full-Seq. | 0.469 | 0.508 | 0.067 | 0.955 | 86.4 |
| CUT3R\* [^67] | Streaming | 2.055 | 2.611 | 0.330 | 0.547 | 44.5 |
| StreamVGGT\* [^92] | Streaming | 2.414 | 2.284 | 0.375 | 0.563 | 86.0 |
| Driv3R\* [^7] | Streaming | 0.742 | 1.345 | 0.189 | 0.721 | – |
| DVGT-2 | Streaming | 0.775 | 0.792 | 0.055 | 0.965 | 84.5 |

### 4.2 Implementation Details

Architecture. Following DVGT [^93], we utilize a ViT-L pretrained by DINOv3 [^61] as the image encoder. The subsequent geometry transformer is composed of $L=24$ blocks, where each block consists of an intra-view local attention layer, a cross-view spatial attention layer, and a temporal causal attention layer. All attention layers operate with a feature dimension of 1024 and 16 heads. For the anchor-based diffusion heads, we build upon DiffusionDrive [^41] with minor architectural modifications to predict ego-poses and future trajectories. To enhance trajectory planning, we integrate ego status, comprising velocity, acceleration, and driving command, into the trajectory token via an MLP.

Training. We train our general DVGT-2 on the mixed dataset using a two-stage paradigm. In the first stage, we conduct geometry reconstruction pre-training without enabling the streaming mechanism. In the second stage, we conduct Vision-Geometry-Action training by introducing trajectory planning supervision and incorporating the streaming mechanism. During both stages, we randomly sample sequences of 2 to 8 views and 2 to 24 frames per scene for training. To further enhance closed-loop planning, we also perform fine-tuning on the NAVSIM to yield the specialized DVGT-2-NAVSIM, which is trained on sequences with fixed 8 views and 4 frames. The entire training process takes approximately ten days on 64 H20 GPUs.

Table 3: Quantitative 3D geometry reconstruction results on Waymo [^62].

| Method | Paradigm | Acc $\downarrow$ | Comp $\downarrow$ | Abs Rel $\downarrow$ | $\delta$ < $1.25\uparrow$ | AUC $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| VGGT\* [^66] | Full-Seq. | 3.202 | 3.390 | 0.333 | 0.567 | 84.3 |
| MapAnything [^29] | Full-Seq. | 9.903 | 8.365 | 0.492 | 0.217 | 82.3 |
| DVGT [^93] | Full-Seq. | 1.752 | 2.263 | 0.102 | 0.922 | 86.0 |
| CUT3R\* [^67] | Streaming | 3.369 | 4.192 | 0.290 | 0.563 | 50.9 |
| StreamVGGT\* [^92] | Streaming | 3.207 | 2.960 | 0.302 | 0.604 | 85.6 |
| Driv3R\* [^7] | Streaming | 0.800 | 1.311 | 0.168 | 0.770 | – |
| DVGT-2 | Streaming | 1.238 | 1.367 | 0.073 | 0.949 | 85.9 |

Table 4: Quantitative 3D geometry reconstruction results on DDAD [^11].

| Method | Paradigm | Acc $\downarrow$ | Comp $\downarrow$ | Abs Rel $\downarrow$ | $\delta$ < $1.25\uparrow$ | AUC $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| VGGT\* [^66] | Full-Seq. | 2.322 | 2.879 | 0.798 | 0.395 | 86.3 |
| MapAnything [^29] | Full-Seq. | 7.442 | 7.928 | 1.836 | 0.207 | 87.4 |
| DVGT [^93] | Full-Seq. | 0.751 | 1.017 | 0.145 | 0.848 | 95.3 |
| CUT3R\* [^67] | Streaming | 2.827 | 4.724 | 0.875 | 0.317 | 48.6 |
| StreamVGGT\* [^92] | Streaming | 2.655 | 2.731 | 0.810 | 0.419 | 91.9 |
| Driv3R\* [^7] | Streaming | 0.950 | 1.259 | 0.185 | 0.740 | – |
| DVGT-2 | Streaming | 1.770 | 1.837 | 0.093 | 0.919 | 92.5 |

### 4.3 Evaluation Metrics

Geometry Reconstruction. Following DVGT [^93], we assess 3D pointmap quality using Accuracy (proximity of predicted points to the ground truth) and Completeness (coverage of the ground truth). We further evaluate the ray depth of the pointmaps using Abs Rel (Absolute Relative error) and $\delta<1.25$ (threshold accuracy). For ego-pose estimation, we report the AUC to measure the Area Under the Curve of the relative pose error.

Planning. We conduct both open-loop and closed-loop evaluations. For open-loop planning on nuScenes, we measure the L2 displacement error and collision rate over a 3-second future horizon. For closed-loop planning, we utilize the NAVSIM v1 and v2 benchmarks. NAVSIM v1 simulates a 4-second non-reactive environment at 10 Hz, scoring agents via the Predictive Driver Model Score (PDMS), which aggregates fundamental safety, comfort, and progress metrics. NAVSIM v2 enhances simulation realism with reactive traffic and introduces the Extended PDMS (EPDMS), which incorporates additional criteria such as traffic rule compliance and extended comfort.

Table 5: Closed-loop planning results on NAVSIM v1 navtest split. $\dagger$ denotes using reinforcement learning to boost planning scores. Future states represent world-modeling-based methods. C and L denote camera and LiDAR, respectively.

| Method | Input | Aux. Sup. | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf. $\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PARA-Drive [^74] | C | Map & Mot. & Occ | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 |
| VADv2 [^3] | C | Map & Mot. & Traffic | 97.2 | 89.1 | 91.6 | 100 | 76.0 | 80.9 |
| UniAD [^16] | C | Map & Box & Mot. & Occ | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| Transfuser [^4] | C & L | Map & Box | 97.7 | 92.8 | 92.8 | 100 | 79.2 | 84.0 |
| Hydra-MDP [^38] | C & L | Map & Box | 98.3 | 96.0 | 94.6 | 100 | 78.7 | 86.5 |
| GoalFlow [^76] | C & L | Map & Box | 98.3 | 93.8 | 94.3 | 100 | 79.8 | 85.7 |
| ARTEMIS [^8] | C & L | Map & Box | 98.3 | 95.1 | 94.3 | 100 | 81.4 | 87.0 |
| DiffusionDrive [^41] | C & L | Map & Box | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| WoTE [^35] | C & L | Map & Box | 98.5 | 96.8 | 94.9 | 99.9 | 81.9 | 88.3 |
| DriveSuprim [^82] | C & L | Map & Box | 97.8 | 97.3 | 93.6 | 100 | 86.7 | 89.9 |
| AutoVLA [^90] | C | Language | 96.9 | 92.4 | 88.1 | 99.9 | 75.8 | 80.5 |
| AdaThinkDrive [^53] | C | Language | 98.5 | 94.4 | 94.9 | 100 | 79.9 | 86.2 |
| ReCogDrive [^37] | C | Language | 98.3 | 95.1 | 94.3 | 100 | 81.1 | 86.8 |
| DriveVLA-W0 [^34] | C | Future States | 98.7 | 99.1 | 95.3 | 99.3 | 83.3 | 90.2 |
| AutoVLA <sup>†</sup> [^90] | C | Language & RL | 98.4 | 95.6 | 98.0 | 99.9 | 85.9 | 89.1 |
| ReCogDrive <sup>†</sup> [^37] | C | Language & RL | 98.2 | 97.8 | 95.2 | 99.8 | 83.5 | 89.6 |
| DVGT-2 | C | Dense Geometry | 97.8 | 97.2 | 93.9 | 100 | 83.4 | 88.6 |
| DVGT-2-NAVSIM | C | Dense Geometry | 98.7 | 97.9 | 95.8 | 100 | 84.3 | 90.3 |

Table 6: Closed-loop planning results on NAVSIM v2 navtest split.

| Method | NC $\uparrow$ | DAC $\uparrow$ | DDC $\uparrow$ | TL $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | LK $\uparrow$ | HC $\uparrow$ | EC $\uparrow$ | EPDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ego Status MLP | 93.1 | 77.9 | 92.7 | 99.6 | 86.0 | 91.5 | 89.4 | 98.3 | 85.4 | 64.0 |
| Transfuser [^4] | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 |
| Hydra-MDP++ [^33] | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | 81.4 |
| DriveSuprim [^82] | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 | 83.1 |
| ARTEMIS [^8] | 98.3 | 95.1 | 98.6 | 99.8 | 81.5 | 97.4 | 96.5 | 98.3 | \- | 83.1 |
| DiffusionDrive [^41] | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | 84.5 |
| DriveVLA-W0 [^34] | 98.5 | 99.1 | 98.0 | 99.7 | 86.4 | 98.1 | 93.2 | 97.9 | 58.9 | 86.1 |
| DVGT-2 | 97.8 | 97.2 | 99.6 | 99.9 | 88.4 | 97.3 | 98.1 | 98.2 | 83.2 | 88.9 |
| DVGT-2-NAVSIM | 98.7 | 97.9 | 99.7 | 99.9 | 87.9 | 98.0 | 98.2 | 98.2 | 77.0 | 89.6 |

### 4.4 Geometry Reconstruction

Inference Settings. We evaluate geometry reconstruction through online, frame-by-frame prediction on 16-frame multi-view sequences. All reported metrics, including global point reconstruction, local ray depth estimation, and global ego-pose prediction, are averaged over 16 frames of the sequences. For non-streaming methods (e.g., VGGT, MapAnything, and DVGT), we implement a streaming inference paradigm, where the inference is performed incrementally by adding one frame per step. For our model, we maintain a history cache with a length of $W=4$ for efficient geometry reconstruction and trajectory planning. Notably, while existing methods predict the global pointmaps and ego-poses in the first frame’s coordinate system, our DVGT-2 predicts local geometry and relative ego-poses at each frame. For a fair comparison, we iteratively transform our predicted local pointmaps and ego-poses into the global coordinate system by accumulating the predicted relative ego-poses before metric evaluation.

Ray Depth Estimation. Ray depth is defined as the distance from a 3D point to the current ego center, which serves as a critical indicator of local geometric accuracy. Since our model natively predicts local pointmaps rather than global ones, it inherently preserves local structural details more effectively. As shown in Tabs. 1, 2, 3 and 4, our approach achieves the state-of-the-art ray depth performance across multiple datasets, outperforming both the general vision-geometry models [^66] [^92] and the driving-specific models [^7] [^93].

Global Point Reconstruction. Evaluating our model on global point reconstruction is inherently challenging. Our local pointmap predictions require iterative aggregation via the predicted relative ego-poses to construct a global pointmap. This process inevitably introduces cumulative errors from ego-pose estimation, while other methods directly output global pointmaps. Despite this structural disadvantage, our model achieves strong global point reconstruction performance comparable to existing SOTA methods, and even surpasses them on specific datasets, as shown in Tabs. 1 and 3.

![[vis 3.png|Refer to caption]]

Figure 6: Qualitative visualizations. These results demonstrate that DVGT-2 can predict high-fidelity dense scene geometry and perform robust trajectory planning.

Global Pose Prediction. We note that our model is less competitive in global ego-pose estimation. We attribute this to three main factors. First, to prioritize inference efficiency, we employ a lightweight pose head with a truncated two-step diffusion strategy. In contrast, baseline models like VGGT and DVGT rely on a heavier four-step residual reasoning process, which yields higher accuracy but degrades inference efficiency. Second, similar to the global pointmap evaluation, deriving global ego-poses by accumulating relative poses inevitably introduces trajectory drift over time. Finally, our sliding window streaming strategy restricts the temporal context to a fixed historical window for online inference. While this enables efficient per-frame processing, it lacks the global context utilized by baseline methods that access the entire sequence simultaneously, thereby limiting long-term global ego-pose consistency.

Inference Efficiency. We compare the online inference efficiency of different methods on the OpenScene dataset using 16-frame, 8-view sequences. Compared to batch-processing methods and full-history streaming methods, our proposed sliding window streaming strategy significantly accelerates online inference. As detailed in Tab. 1, with a fixed window size of 4, our method efficiently processes the 16-frame sequence with an average latency of only 0.27s per frame. This outperforms prior methods while maintaining robust reconstruction capabilities, demonstrating its critical value for real-time autonomous driving.

Table 7: Open-looped planning results on nuScenes [^1]. $\dagger$ denotes the results computed with an average of previous frames as adopted in VAD [^26]. Aux. Sup. represents auxiliary supervision. Avg. computes the average result of 1s, 2s, and 3s.

<table><tbody><tr><td rowspan="2">Method</td><td rowspan="2">Input</td><td rowspan="2">Aux. Sup.</td><td colspan="4">L2 (m) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td colspan="4">Collision Rate (%) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td></tr><tr><td>IL <sup><a href="#fn:57">57</a></sup></td><td>LiDAR</td><td>None</td><td>0.44</td><td>1.15</td><td>2.47</td><td>1.35</td><td>0.08</td><td>0.27</td><td>1.95</td><td>0.77</td></tr><tr><td>NMP <sup><a href="#fn:83">83</a></sup></td><td>LiDAR</td><td>Box & Motion</td><td>0.53</td><td>1.25</td><td>2.67</td><td>1.48</td><td>0.04</td><td>0.12</td><td>0.87</td><td>0.34</td></tr><tr><td>FF <sup><a href="#fn:14">14</a></sup></td><td>LiDAR</td><td>Freespace</td><td>0.55</td><td>1.20</td><td>2.54</td><td>1.43</td><td>0.06</td><td>0.17</td><td>1.07</td><td>0.43</td></tr><tr><td>EO <sup><a href="#fn:30">30</a></sup></td><td>LiDAR</td><td>Freespace</td><td>0.67</td><td>1.36</td><td>2.78</td><td>1.60</td><td>0.04</td><td>0.09</td><td>0.88</td><td>0.33</td></tr><tr><td>ST-P3 <sup><a href="#fn:15">15</a></sup></td><td>Camera</td><td>Map & Box & Depth</td><td>1.33</td><td>2.11</td><td>2.90</td><td>2.11</td><td>0.23</td><td>0.62</td><td>1.27</td><td>0.71</td></tr><tr><td>UniAD <sup><a href="#fn:16">16</a></sup></td><td>Camera</td><td>Map & Box & Mot. & Occ</td><td>0.48</td><td>0.96</td><td>1.65</td><td>1.03</td><td>0.05</td><td>0.17</td><td>0.71</td><td>0.31</td></tr><tr><td>VAD-Tiny <sup><a href="#fn:26">26</a></sup></td><td>Camera</td><td>Map & Box & Mot.</td><td>0.60</td><td>1.23</td><td>2.06</td><td>1.30</td><td>0.31</td><td>0.53</td><td>1.33</td><td>0.72</td></tr><tr><td>VAD-Base <sup><a href="#fn:26">26</a></sup></td><td>Camera</td><td>Map & Box & Mot.</td><td>0.54</td><td>1.15</td><td>1.98</td><td>1.22</td><td>0.04</td><td>0.39</td><td>1.17</td><td>0.53</td></tr><tr><td>GenAD <sup><a href="#fn:86">86</a></sup></td><td>Camera</td><td>Map & Box & Mot.</td><td>0.36</td><td>0.83</td><td>1.55</td><td>0.91</td><td>0.06</td><td>0.23</td><td>1.00</td><td>0.43</td></tr><tr><td>OccWorld <sup><a href="#fn:85">85</a></sup></td><td>Camera</td><td>Occ</td><td>0.52</td><td>1.27</td><td>2.41</td><td>1.40</td><td>0.12</td><td>0.40</td><td>2.08</td><td>0.87</td></tr><tr><td>OccNet <sup><a href="#fn:63">63</a></sup></td><td>Camera</td><td>Occ & Map & Box</td><td>1.29</td><td>2.13</td><td>2.99</td><td>2.14</td><td>0.21</td><td>0.59</td><td>1.37</td><td>0.72</td></tr><tr><td>GaussianAD <sup><a href="#fn:87">87</a></sup></td><td>Camera</td><td>Map & Box & Mot. & Occ</td><td>0.40</td><td>0.64</td><td>0.88</td><td>0.64</td><td>0.09</td><td>0.38</td><td>0.81</td><td>0.42</td></tr><tr><td>OmniDrive <sup><a href="#fn:70">70</a></sup></td><td>Camera</td><td>Map & Box & Language</td><td>0.40</td><td>0.80</td><td>1.32</td><td>0.84</td><td>0.04</td><td>0.46</td><td>2.32</td><td>0.94</td></tr><tr><td>Doe-1 <sup><a href="#fn:88">88</a></sup></td><td>Camera</td><td>Language</td><td>0.50</td><td>1.18</td><td>2.11</td><td>1.26</td><td>0.04</td><td>0.37</td><td>1.19</td><td>0.53</td></tr><tr><td>UniUGP <sup><a href="#fn:50">50</a></sup></td><td>Camera</td><td>Language</td><td>0.58</td><td>1.14</td><td>1.95</td><td>1.23</td><td>0.01</td><td>0.19</td><td>0.81</td><td>0.33</td></tr><tr><td>DVGT-2</td><td>Camera</td><td>Dense Geometry</td><td>0.25</td><td>0.67</td><td>1.43</td><td>0.78</td><td>0.00</td><td>0.07</td><td>0.50</td><td>0.19</td></tr><tr><td>VAD-Tiny <sup>†</sup> <sup><a href="#fn:26">26</a></sup></td><td>Camera</td><td>Map & Box & Mot.</td><td>0.46</td><td>0.76</td><td>1.12</td><td>0.78</td><td>0.21</td><td>0.35</td><td>0.58</td><td>0.38</td></tr><tr><td>VAD-Base <sup>†</sup> <sup><a href="#fn:26">26</a></sup></td><td>Camera</td><td>Map & Box & Mot.</td><td>0.41</td><td>0.70</td><td>1.05</td><td>0.72</td><td>0.07</td><td>0.17</td><td>0.41</td><td>0.22</td></tr><tr><td>OccWorld-D <sup>†</sup> <sup><a href="#fn:85">85</a></sup></td><td>Camera</td><td>Occ</td><td>0.39</td><td>0.73</td><td>1.18</td><td>0.77</td><td>0.11</td><td>0.19</td><td>0.67</td><td>0.32</td></tr><tr><td>GenAD <sup>†</sup> <sup><a href="#fn:86">86</a></sup></td><td>Camera</td><td>Map & Box & Mot.</td><td>0.28</td><td>0.49</td><td>0.78</td><td>0.52</td><td>0.08</td><td>0.14</td><td>0.34</td><td>0.19</td></tr><tr><td>GaussianAD <sup>†</sup> <sup><a href="#fn:87">87</a></sup></td><td>Camera</td><td>Map & Box & Occ</td><td>0.34</td><td>0.47</td><td>0.60</td><td>0.47</td><td>0.49</td><td>0.49</td><td>0.51</td><td>0.50</td></tr><tr><td>DVGT-2 <sup>†</sup></td><td>Camera</td><td>Dense Geometry</td><td>0.20</td><td>0.37</td><td>0.66</td><td>0.41</td><td>0.04</td><td>0.14</td><td>0.47</td><td>0.22</td></tr></tbody></table>

### 4.5 Planning

Closed-loop Planning on NAVSIM v1 [^6] and v2 [^2]. Trained on a large mixture of datasets, our foundation VGA model DVGT-2 performs robust planning based on high-fidelity dense geometry reconstruction. As shown in Tab. 5, DVGT-2 achieves an 88.6 PDMS on NAVSIM v1, comparable to SOTA end-to-end and VLA models. It also yields an EPDMS of 88.9 on NAVSIM v2 ( Tab. 6), outperforming all SOTA methods. We also fine-tune this foundation model on NAVSIM to produce DVGT-2-NAVSIM, which establishes new SOTA on both benchmarks. We show two examples of geometry reconstruction and planning by DVGT-2 in Fig. 6. These results validate dense geometry as a robust foundation for safe planning. Unlike traditional end-to-end models that rely on multi-modal inputs and sparse perceptual supervision, or VLA models that require language labels and complicated RL fine-tuning, our VGA paradigm achieves safe, end-to-end driving using only annotation-efficient geometric supervision.

Open-loop Planning on nuScenes. As shown in Tab. 7, DVGT-2 achieves L2 error metrics comparable to SOTA models on nuScenes. More importantly, DVGT-2 yields a significantly lower collision rate than models explicitly trained with high-level semantic labels (which directly define the collision metrics). This highlights that our model inherently learns the comprehensive 3D structure and physical interactions between the ego-vehicle and the environment, enabling robust planning without relying on sparse perceptual annotations.

### 4.6 Inference Efficiency Comparison

We compare the online inference efficiency of different methods, focusing on per-frame memory cost and latency. All models perform frame-by-frame inference on identical multi-frame, 8-view sequences. It is worth noting that our model simultaneously performs geometry reconstruction and trajectory planning, whereas the compared models focus solely on geometry reconstruction.

As shown in Fig. 7(a), VGGT and DVGT encounter out-of-memory (OOM) errors after only about 10 frames. This is because their full-sequence reconstruction incurs a quadratic memory complexity of $O(T^{2})$ with respect to the sequence length $T$. StreamVGGT’s memory grows more slowly but still hits OOM at around 30 frames due to its full-history streaming strategy ($O(T)$ complexity). In contrast, DVGT-2 maintains a constant memory cost. By employing a sliding window streaming strategy with a fixed-size historical cache, our model achieves an $O(1)$ memory cost, enabling infinite-length online inference for driving scenes.

The inference latency comparison is shown in Fig. 7(b). Due to the $O(T^{2})$ complexity, the latency of VGGT and DVGT surges rapidly, taking several seconds to process just a few frames. StreamVGGT’s latency scales linearly ($O(T)$ complexity), reaching 5–6 seconds at 30 frames, which is still prohibitive for real-time driving. However, DVGT-2 achieves a stable latency of around 260ms per frame, even across hundreds of frames, confirming that our model is well-suited for real-time, infinite-length autonomous driving.

(a) Memory Comparison.

(b) Latency Comparison.

Figure 7: Efficiency comparison of online inference. We report the per-frame latency and memory cost of different methods on multi-frame, 8-view sequences.

## 5 Conclusion

In this paper, we introduce DVGT-2, a streaming driving visual geometry transformer that pioneers the Vision-Geometry-Action (VGA) paradigm for end-to-end autonomous driving. By reconstructing dense 3D geometry as the foundation representation, DVGT-2 provides comprehensive spatial and temporal cues for robust trajectory planning. To overcome the computational bottleneck of traditional multi-frame processing, we propose a sliding-window streaming strategy with temporal causal attention and feature caching, enabling efficient, on-the-fly joint prediction of geometry and trajectories. Extensive experiments demonstrate that DVGT-2 achieves strong geometry reconstruction performance with significantly reduced latency on diverse datasets. Moreover, it exhibits strong planning capabilities across open-loop and closed-loop benchmarks. We hope this work paves the way for more efficient, geometry-aware driving systems.

## References

![[Uncaptioned image]](https://arxiv.org/html/2604.00813v3/teaser_supp.png)

Figure A.1: Video demonstration of DVGT-2’s geometry reconstruction and trajectory planning based on online multi-view inputs on the validation set.

## Appendix 0.A Additional Experiments

### 0.A.1 Ablation on Window size

Tab. A.2 details the impact of window size on streaming inference. Increasing the window size from 2 to 6 improves the global point accuracy (Acc) by enlarging the temporal receptive field, which helps model inter-frame relations and global structures. However, a larger window size of 8 slightly degrades the Acc metric. This occurs because the accumulated error of predicted relative ego-poses during the local-to-global point transformation outweighs the benefits of a broader temporal context. Conversely, the ray depth prediction (Abs Rel) remains constant across all settings, indicating that the window size exclusively affects inter-frame and global geometry modeling, rather than local geometry accuracy.

### 0.A.2 Planning on All Datasets

Tab. A.2 presents the 1-second future trajectory prediction performance (L2 error) of DVGT-2 on the validation sets of five datasets. The model achieves the lowest L2 error of 0.20m on OpenScene, since it makes up over 75% of the training data. It also maintains competitive performance on NuScenes and Waymo. However, the errors on KITTI and DDAD are notably higher (>2.0m). We attribute this to a significant domain gap in ego-vehicle trajectory distributions, primarily due to much higher driving speeds in these two datasets compared to the others. Since our model employs anchor-based diffusion heads, both the anchor clustering and trajectory distribution modeling are heavily biased towards the dominant OpenScene dataset. Consequently, this leads to sub-optimal performance on KITTI and DDAD in both ego-pose prediction and trajectory planning.

Table A.1: Ablation study on the window size of historical cache for streaming inference.

| Window Size | Acc ($\downarrow$) | Abs Rel. ($\downarrow$) |
| --- | --- | --- |
| 2 | 0.613 | 0.042 |
| 4 | 0.480 | 0.042 |
| 6 | 0.474 | 0.042 |
| 8 | 0.501 | 0.042 |

Table A.2: L2 error of trajectory planning across five datasets.

| Dataset | L2(m) ($\downarrow$) |
| --- | --- |
| NuScenes | 0.56 |
| OpenScene | 0.20 |
| Waymo | 0.78 |
| KITTI | 2.12 |
| DDAD | 2.00 |

### 0.A.3 Geometry Reconstruction on KITTI

Table A.3: Quantitative 3D geometry reconstruction results on KITTI [^10]. \* denotes performing post-alignment with sparse LiDAR to recover the metric scale.

| Method | Paradigm | Acc $\downarrow$ | Comp $\downarrow$ | Abs Rel $\downarrow$ | $\delta$ < $1.25\uparrow$ | AUC@30 $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| VGGT\* [^66] | Full-Seq. | 1.154 | 1.294 | 0.158 | 0.801 | 96.91 |
| MapAnything [^29] | Full-Seq. | 1.807 | 1.006 | 0.184 | 0.727 | 90.51 |
| DVGT [^93] | Full-Seq. | 0.846 | 1.468 | 0.136 | 0.849 | 87.63 |
| CUT3R\* [^67] | Streaming | 0.973 | 2.054 | 0.217 | 0.660 | 51.82 |
| StreamVGGT\* [^92] | Streaming | 3.393 | 2.181 | 0.365 | 0.467 | 95.57 |
| Driv3R\* [^7] | Streaming | 0.864 | 1.083 | 0.164 | 0.784 | \- |
| DVGT-2 | Streaming | 1.615 | 3.238 | 0.087 | 0.942 | 80.36 |

Tab. A.3 presents the quantitative geometry reconstruction results on KITTI. DVGT-2 achieves state-of-the-art performance in ray depth estimation, significantly outperforming other methods in both the Abs Rel and $\delta$ < $1.25$ metrics. This superiority stems from our model’s inherent design of predicting local pointmaps, which enables effective modeling of local geometry. However, as discussed earlier, the significant domain gap in ego-vehicle trajectories, along with the fact that KITTI accounts for only a small portion of the training data, leads to sub-optimal ego-pose prediction. Furthermore, because the evaluation of global point accuracy relies on these predicted ego-poses to transform local points into global coordinates, accumulated ego-pose errors inevitably degrade the overall performance of global point reconstruction.

## Appendix 0.B Additional Implementation Details

Architecture. Building upon the overall architecture in the main paper, our model comprises approximately 1.8 billion parameters. To ensure training stability, we incorporate QKNorm [^13] and LayerScale [^64] (initialized at 0.01) into each attention layer of the geometry transformer. For dense prediction, we follow [^80] by feeding intermediate tokens from the 4th, 11th, 17th, and 23rd blocks into a DPT [^56] head. For ego-pose prediction and trajectory planning, we augment the visual tokens of each view and frame with one pose token and eight trajectory tokens to aggregate global context in the subsequent geometry transformer. Following DiffusionDrive [^41], we then utilize two anchor-based diffusion heads for decoding, which comprise four self-attention layers for inter-frame interactions and two cross-attention layers for diffusion decoding. We employ 20 anchors for both ego-poses and trajectories, which are pre-computed by clustering the training data.

Training. We train the general DVGT-2 on the mixed dataset for 160K and 80K iterations in the first and second stages, respectively. Subsequently, we finetune the model for 40K iterations on NAVSIM to obtain the specialized DVGT-2-NAVSIM. Across all stages, we optimize using AdamW [^49] with a cosine learning rate scheduler, setting the peak learning rate to 1e-4 with an 8K-iteration linear warmup. To ensure training stability and efficiency, we employ gradient norm clipping with a threshold of 1.0, bfloat16 precision, and gradient checkpointing. During the first two stages, we train DVGT-2 on sequences with random views (ranging from 2 to 8) and frames (ranging from 2 to 24) from the mixed dataset. We then finetune DVGT-2-NAVSIM on NAVSIM, where the sequence is fixed to 8 views and 4 frames to align with the standard NAVSIM planning setting. For image preprocessing, we first resize the long edge of the input images to 512 pixels while keeping the original aspect ratio. We then center-crop the short edge to a random size between 144 and 320 pixels (ensuring it is divisible by 16). Finally, we apply strong per-frame augmentations—such as color jittering, Gaussian blur, and grayscale conversion—to make the model robust to lighting changes.

## Appendix 0.C Dataset Details

Table A.4: Detailed statistics of the datasets used in our experiments. All temporal statistics are reported at a 2Hz sampling rate.

| Dataset | Train Scenes | Test Scenes | Min Frames | Max Frames | Avg Frames | Avg Aspect Ratio | Num of Views |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nuScenes | 700 | 150 | 32 | 41 | 40 | 1.77 | 6 |
| OpenScene | 19376 | 2026 | 1 | 41 | 38 | 1.77 | 8 |
| Waymo | 798 | 202 | 34 | 40 | 40 | 1.77 | 5 |
| KITTI | 138 | 13 | 2 | 1033 | 62 | 3.31 | 2 |
| DDAD | 150 | 50 | 10 | 20 | 17 | 1.59 | 6 |

Following DVGT [^93], we train and evaluate our model on a mixture of five driving datasets: nuScenes, OpenScene, Waymo, KITTI, and c DDAD. Tab. A.4 shows their detailed statistics. All videos are downsampled to 2Hz, and the frame counts in the table are based on this rate. During training of our general DVGT-2, we select datasets for each batch using the following ratio: nuScenes: OpenScene: Waymo: KITTI: DDAD = 6:77:6:5:6. To make the model robust to different sensor setups, we apply a dynamic sampling strategy in each iteration:

1. Randomly pick an image aspect ratio from \[1.6, 3.3\].
2. Randomly choose the number of camera views from \[2, 8\].
3. Calculate the maximum sequence length $T_{max}$ based on a hardware limit of 48 images per GPU.
4. Randomly select a sequence length from \[2, $T_{max}$\] and set the batch size to fill the GPU memory.

When finetuning DVGT-2-NAVSIM on NAVSIM, the input sequence from OpenScene is fixed to 8 views and 4 frames with an aspect ratio of 1.6, while the batch size is set to 1.

## Appendix 0.D Video Demonstration

Fig. A.1 shows a sampled image from the video demo that demonstrates our model’s predictions on the validation set. Given multi-view image sequences as input, DVGT-2 performs robust geometry reconstruction and trajectory planning with high fidelity and consistency in an online manner, validating the effectiveness and efficiency of our VGA paradigm.

[^1]: Caesar, H., Bankiti, V., Lang, A.H., Vora, S., Liong, V.E., Xu, Q., Krishnan, A., Pan, Y., Baldan, G., Beijbom, O.: nuscenes: A multimodal dataset for autonomous driving. In: CVPR. pp. 11621–11631 (2020)

[^2]: Cao, W., Hallgarten, M., Li, T., Dauner, D., Gu, X., Wang, C., Miron, Y., Aiello, M., Li, H., Gilitschenski, I., et al.: Pseudo-simulation for autonomous driving. arXiv preprint arXiv:2506.04218 (2025)

[^3]: Chen, S., Jiang, B., Gao, H., Liao, B., Xu, Q., Zhang, Q., Huang, C., Liu, W., Wang, X.: Vadv2: End-to-end vectorized autonomous driving via probabilistic planning. arXiv preprint arXiv:2402.13243 (2024)

[^4]: Chitta, K., Prakash, A., Jaeger, B., Yu, Z., Renz, K., Geiger, A.: Transfuser: Imitation with transformer-based sensor fusion for autonomous driving. TPAMI 45(11), 12878–12895 (2022)

[^5]: Contributors, O.: Openscene: The largest up-to-date 3d occupancy prediction benchmark in autonomous driving. In: CVPR. pp. 18–22 (2023)

[^6]: Dauner, D., Hallgarten, M., Li, T., Weng, X., Huang, Z., Yang, Z., Li, H., Gilitschenski, I., Ivanovic, B., Pavone, M., et al.: Navsim: Data-driven non-reactive autonomous vehicle simulation and benchmarking. NeurIPS 37, 28706–28719 (2024)

[^7]: Fei, X., Zheng, W., Duan, Y., Zhan, W., Tomizuka, M., Keutzer, K., Lu, J.: Driv3r: Learning dense 4d reconstruction for autonomous driving. arXiv preprint arXiv:2412.06777 (2024)

[^8]: Feng, R., Xi, N., Chu, D., Wang, R., Deng, Z., Wang, A., Lu, L., Wang, J., Huang, Y.: Artemis: Autoregressive end-to-end trajectory planning with mixture of experts for autonomous driving. arXiv preprint arXiv:2504.19580 (2025)

[^9]: Fu, H., Zhang, D., Zhao, Z., Cui, J., Liang, D., Zhang, C., Zhang, D., Xie, H., Wang, B., Bai, X.: Orion: A holistic end-to-end autonomous driving framework by vision-language instructed action generation. arXiv preprint arXiv:2503.19755 (2025)

[^10]: Geiger, A., Lenz, P., Stiller, C., Urtasun, R.: Vision meets robotics: The kitti dataset. IJRR 32(11), 1231–1237 (2013)

[^11]: Guizilini, V., Ambrus, R., Pillai, S., Raventos, A., Gaidon, A.: 3d packing for self-supervised monocular depth estimation. In: CVPR. pp. 2485–2494 (2020)

[^12]: Hegde, D., Yasarla, R., Cai, H., Han, S., Bhattacharyya, A., Mahajan, S., Liu, L., Garrepalli, R., Patel, V.M., Porikli, F.: Distilling multi-modal large language models for autonomous driving. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 27575–27585 (2025)

[^13]: Henry, A., Dachapally, P.R., Pawar, S.S., Chen, Y.: Query-key normalization for transformers. In: EMNLP. pp. 4246–4253 (2020)

[^14]: Hu, P., Huang, A., Dolan, J., Held, D., Ramanan, D.: Safe local motion planning with self-supervised freespace forecasting. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 12732–12741 (2021)

[^15]: Hu, S., Chen, L., Wu, P., Li, H., Yan, J., Tao, D.: St-p3: End-to-end vision-based autonomous driving via spatial-temporal feature learning. In: European Conference on Computer Vision. pp. 533–549. Springer (2022)

[^16]: Hu, Y., Yang, J., Chen, L., Li, K., Sima, C., Zhu, X., Chai, S., Du, S., Lin, T., Wang, W., et al.: Planning-oriented autonomous driving. In: CVPR. pp. 17853–17862 (2023)

[^17]: Huang, J., Liu, X., Song, S., Hou, R., Chang, H., Lin, J., Bai, S.: Revisiting multimodal positional encoding in vision-language models. arXiv preprint arXiv:2510.23095 (2025)

[^18]: Huang, J., Huang, G., Zhu, Z., Ye, Y., Du, D.: Bevdet: High-performance multi-camera 3d object detection in bird-eye-view. arXiv preprint arXiv:2112.11790 (2021)

[^19]: Huang, Y., Thammatadatrakoon, A., Zheng, W., Zhang, Y., Du, D., Lu, J.: Gaussianformer-2: Probabilistic gaussian superposition for efficient 3d occupancy prediction. In: CVPR. pp. 27477–27486 (2025)

[^20]: Huang, Y., Zheng, W., Zhang, Y., Zhou, J., Lu, J.: Tri-perspective view for vision-based 3d semantic occupancy prediction. In: CVPR. pp. 9223–9232 (2023)

[^21]: Huang, Y., Zheng, W., Zhang, Y., Zhou, J., Lu, J.: Gaussianformer: Scene as gaussians for vision-based 3d semantic occupancy prediction. In: ECCV. pp. 376–393. Springer (2024)

[^22]: Huang, Z., Feng, C., Yan, F., Xiao, B., Jie, Z., Zhong, Y., Liang, X., Ma, L.: Drivemm: All-in-one large multimodal model for autonomous driving. arXiv preprint arXiv:2412.07689 (2024)

[^23]: Hwang, J.J., Xu, R., Lin, H., Hung, W.C., Ji, J., Choi, K., Huang, D., He, T., Covington, P., Sapp, B., et al.: Emma: End-to-end multimodal model for autonomous driving. arXiv preprint arXiv:2410.23262 (2024)

[^24]: Jiang, A., Gao, Y., Sun, Z., Wang, Y., Wang, J., Chai, J., Cao, Q., Heng, Y., Jiang, H., Dong, Y., et al.: Diffvla: Vision-language guided diffusion planning for autonomous driving. arXiv preprint arXiv:2505.19381 (2025)

[^25]: Jiang, B., Chen, S., Liao, B., Zhang, X., Yin, W., Zhang, Q., Huang, C., Liu, W., Wang, X.: Senna: Bridging large vision-language models and end-to-end autonomous driving. arXiv preprint arXiv:2410.22313 (2024)

[^26]: Jiang, B., Chen, S., Xu, Q., Liao, B., Chen, J., Zhou, H., Zhang, Q., Liu, W., Huang, C., Wang, X.: Vad: Vectorized scene representation for efficient autonomous driving. In: ICCV. pp. 8340–8350 (2023)

[^27]: Jiang, B., Chen, S., Zhang, Q., Liu, W., Wang, X.: Alphadrive: Unleashing the power of vlms in autonomous driving via reinforcement learning and reasoning. arXiv preprint arXiv:2503.07608 (2025)

[^28]: Jiang, X., Ma, Y., Li, P., Xu, L., Wen, X., Zhan, K., Xia, Z., Jia, P., Lang, X., Sun, S.: Transdiffuser: End-to-end trajectory generation with decorrelated multi-modal representation for autonomous driving. arXiv e-prints pp. arXiv–2505 (2025)

[^29]: Keetha, N., Müller, N., Schönberger, J., Porzi, L., Zhang, Y., Fischer, T., Knapitsch, A., Zauss, D., Weber, E., Antunes, N., et al.: Mapanything: Universal feed-forward metric 3d reconstruction. arXiv preprint arXiv:2509.13414 (2025)

[^30]: Khurana, T., Hu, P., Dave, A., Ziglar, J., Held, D., Ramanan, D.: Differentiable raycasting for self-supervised occupancy forecasting. In: European Conference on Computer Vision. pp. 353–369. Springer (2022)

[^31]: Leroy, V., Cabon, Y., Revaud, J.: Grounding image matching in 3d with mast3r. In: ECCV. pp. 71–91. Springer (2024)

[^32]: Li, D., Ren, J., Wang, Y., Wen, X., Li, P., Xu, L., Zhan, K., Xia, Z., Jia, P., Lang, X., et al.: Finetuning generative trajectory model with reinforcement learning from human feedback. arXiv e-prints pp. arXiv–2503 (2025)

[^33]: Li, K., Li, Z., Lan, S., Xie, Y., Zhang, Z., Liu, J., Wu, Z., Yu, Z., Alvarez, J.M.: Hydra-mdp++: Advancing end-to-end driving via expert-guided hydra-distillation. arXiv preprint arXiv:2503.12820 (2025)

[^34]: Li, Y., Shang, S., Liu, W., Zhan, B., Wang, H., Wang, Y., Chen, Y., Wang, X., An, Y., Tang, C., et al.: Drivevla-w0: World models amplify data scaling law in autonomous driving. arXiv preprint arXiv:2510.12796 (2025)

[^35]: Li, Y., Wang, Y., Liu, Y., He, J., Fan, L., Zhang, Z.: End-to-end driving with online trajectory evaluation via bev world model. arXiv preprint arXiv:2504.01941 (2025)

[^36]: Li, Y., Ge, Z., Yu, G., Yang, J., Wang, Z., Shi, Y., Sun, J., Li, Z.: Bevdepth: Acquisition of reliable depth for multi-view 3d object detection. In: AAAI. vol. 37, pp. 1477–1485 (2023)

[^37]: Li, Y., Xiong, K., Guo, X., Li, F., Yan, S., Xu, G., Zhou, L., Chen, L., Sun, H., Wang, B., et al.: Recogdrive: A reinforced cognitive framework for end-to-end autonomous driving. arXiv preprint arXiv:2506.08052 (2025)

[^38]: Li, Z., Li, K., Wang, S., Lan, S., Yu, Z., Ji, Y., Li, Z., Zhu, Z., Kautz, J., Wu, Z., et al.: Hydra-mdp: End-to-end multimodal planning with multi-target hydra-distillation. arXiv preprint arXiv:2406.06978 (2024)

[^39]: Li, Z., Wang, W., Li, H., Xie, E., Sima, C., Lu, T., Yu, Q., Dai, J.: Bevformer: learning bird’s-eye-view representation from lidar-camera via spatiotemporal transformers. TPAMI (2024)

[^40]: Liang, T., Xie, H., Yu, K., Xia, Z., Lin, Z., Wang, Y., Tang, T., Wang, B., Tang, Z.: Bevfusion: A simple and robust lidar-camera fusion framework. Advances in neural information processing systems 35, 10421–10434 (2022)

[^41]: Liao, B., Chen, S., Yin, H., Jiang, B., Wang, C., Yan, S., Zhang, X., Li, X., Zhang, Y., Zhang, Q., et al.: Diffusiondrive: Truncated diffusion model for end-to-end autonomous driving. In: CVPR. pp. 12037–12047 (2025)

[^42]: Liao, H., Kong, H., Wang, B., Wang, C., Ye, W., He, Z., Xu, C., Li, Z.: Cot-drive: Efficient motion forecasting for autonomous driving with llms and chain-of-thought prompting. IEEE Transactions on Artificial Intelligence (2025)

[^43]: Lin, X., Lin, T., Pei, Z., Huang, L., Su, Z.: Sparse4d: Multi-view 3d object detection with sparse spatial-temporal fusion. arXiv preprint arXiv:2211.10581 (2022)

[^44]: Lin, X., Lin, T., Pei, Z., Huang, L., Su, Z.: Sparse4d v2: Recurrent temporal fusion with sparse model. arXiv preprint arXiv:2305.14018 (2023)

[^45]: Lin, X., Pei, Z., Lin, T., Huang, L., Su, Z.: Sparse4d v3: Advancing end-to-end 3d detection and tracking. arXiv preprint arXiv:2311.11722 (2023)

[^46]: Liu, P., Liu, H., Liu, H., Liu, X., Ni, J., Ma, J.: Vlm-e2e: Enhancing end-to-end autonomous driving with multimodal driver attention fusion. arXiv preprint arXiv:2502.18042 (2025)

[^47]: Liu, W., Liu, P., Ma, J.: Dsdrive: Distilling large language model for lightweight end-to-end autonomous driving with unified reasoning and planning. arXiv preprint arXiv:2505.05360 (2025)

[^48]: Liu, Z., Tang, H., Amini, A., Yang, X., Mao, H., Rus, D.L., Han, S.: Bevfusion: Multi-task multi-sensor fusion with unified bird’s-eye view representation. In: 2023 IEEE international conference on robotics and automation (ICRA). pp. 2774–2781. IEEE (2023)

[^49]: Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: ICLR (2019)

[^50]: Lu, H., Liu, Z., Jiang, G., Luo, Y., Chen, S., Zhang, Y., Chen, Y.C.: Uniugp: Unifying understanding, generation, and planing for end-to-end autonomous driving. arXiv preprint arXiv:2512.09864 (2025)

[^51]: Luo, Y., Chen, Q., Li, F., Xu, S., Liu, J., Song, Z., Yang, Z.x., Wen, F.: Unleashing vla potentials in autonomous driving via explicit learning from failures. arXiv preprint arXiv:2603.01063 (2026)

[^52]: Luo, Y., Li, F., Xu, S., Ji, Y., Zhang, Z., Wang, B., Shen, Y., Cui, J., Chen, L., Chen, G., et al.: Last-vla: Thinking in latent spatio-temporal space for vision-language-action in autonomous driving. arXiv preprint arXiv:2603.01928 (2026)

[^53]: Luo, Y., Li, F., Xu, S., Lai, Z., Yang, L., Chen, Q., Luo, Z., Xie, Z., Jiang, S., Liu, J., et al.: Adathinkdrive: Adaptive thinking via reinforcement learning for autonomous driving. arXiv preprint arXiv:2509.13769 (2025)

[^54]: Mao, J., Qian, Y., Ye, J., Zhao, H., Wang, Y.: Gpt-driver: Learning to drive with gpt. arXiv preprint arXiv:2310.01415 (2023)

[^55]: Pan, C., Yaman, B., Nesti, T., Mallik, A., Allievi, A.G., Velipasalar, S., Ren, L.: Vlp: Vision language planning for autonomous driving. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 14760–14769 (2024)

[^56]: Ranftl, R., Bochkovskiy, A., Koltun, V.: Vision transformers for dense prediction. In: ICCV. pp. 12179–12188 (2021)

[^57]: Ratliff, N.D., Bagnell, J.A., Zinkevich, M.A.: Maximum margin planning. In: Proceedings of the 23rd international conference on Machine learning. pp. 729–736 (2006)

[^58]: Renz, K., Chen, L., Arani, E., Sinavski, O.: Simlingo: Vision-only closed-loop autonomous driving with language-action alignment. In: CVPR. pp. 11993–12003 (2025)

[^59]: Renz, K., Chen, L., Marcu, A.M., Hünermann, J., Hanotte, B., Karnsund, A., Shotton, J., Arani, E., Sinavski, O.: Carllava: Vision language models for camera-only closed-loop driving. arXiv preprint arXiv:2406.10165 (2024)

[^60]: Shao, H., Hu, Y., Wang, L., Song, G., Waslander, S.L., Liu, Y., Li, H.: Lmdrive: Closed-loop end-to-end driving with large language models. In: CVPR. pp. 15120–15130 (2024)

[^61]: Siméoni, O., Vo, H.V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., et al.: Dinov3. arXiv preprint arXiv:2508.10104 (2025)

[^62]: Sun, P., Kretzschmar, H., Dotiwalla, X., Chouard, A., Patnaik, V., Tsui, P., Guo, J., Zhou, Y., Chai, Y., Caine, B., et al.: Scalability in perception for autonomous driving: Waymo open dataset. In: CVPR. pp. 2446–2454 (2020)

[^63]: Tong, W., Sima, C., Wang, T., Chen, L., Wu, S., Deng, H., Gu, Y., Lu, L., Luo, P., Lin, D., et al.: Scene as occupancy. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 8406–8415 (2023)

[^64]: Touvron, H., Cord, M., Sablayrolles, A., Synnaeve, G., Jégou, H.: Going deeper with image transformers. In: ICCV. pp. 32–42 (2021)

[^65]: Wang, H., Agapito, L.: 3d reconstruction with spatial memory. arXiv preprint arXiv:2408.16061 (2024)

[^66]: Wang, J., Chen, M., Karaev, N., Vedaldi, A., Rupprecht, C., Novotny, D.: Vggt: Visual geometry grounded transformer. In: CVPR. pp. 5294–5306 (2025)

[^67]: Wang, Q., Zhang, Y., Holynski, A., Efros, A.A., Kanazawa, A.: Continuous 3d perception model with persistent state. In: CVPR. pp. 10510–10522 (2025)

[^68]: Wang, R., Xu, S., Dong, Y., Deng, Y., Xiang, J., Lv, Z., Sun, G., Tong, X., Yang, J.: Moge-2: Accurate monocular geometry with metric scale and sharp details. arXiv preprint arXiv:2507.02546 (2025)

[^69]: Wang, S., Liu, Y., Wang, T., Li, Y., Zhang, X.: Exploring object-centric temporal modeling for efficient multi-view 3d object detection. In: ICCV. pp. 3621–3631 (2023)

[^70]: Wang, S., Yu, Z., Jiang, X., Lan, S., Shi, M., Chang, N., Kautz, J., Li, Y., Alvarez, J.M.: Omnidrive: A holistic vision-language dataset for autonomous driving with counterfactual reasoning. In: Proceedings of the computer vision and pattern recognition conference. pp. 22442–22452 (2025)

[^71]: Wang, S., Leroy, V., Cabon, Y., Chidlovskii, B., Revaud, J.: Dust3r: Geometric 3d vision made easy. In: CVPR. pp. 20697–20709 (2024)

[^72]: Wang, W., Xie, J., Hu, C., Zou, H., Fan, J., Tong, W., Wen, Y., Wu, S., Deng, H., Li, Z., et al.: Drivemlm: Aligning multi-modal large language models with behavioral planning states for autonomous driving. arXiv preprint arXiv:2312.09245 (2023)

[^73]: Wang, Y., Zhou, J., Zhu, H., Chang, W., Zhou, Y., Li, Z., Chen, J., Pang, J., Shen, C., He, T.: $\pi^{3}$: Permutation-equivariant visual geometry learning. arXiv preprint arXiv:2507.13347 (2025)

[^74]: Weng, X., Ivanovic, B., Wang, Y., Wang, Y., Pavone, M.: Para-drive: Parallelized architecture for real-time autonomous driving. In: CVPR. pp. 15449–15458 (2024)

[^75]: Wu, Y., Zheng, W., Zhou, J., Lu, J.: Point3r: Streaming 3d reconstruction with explicit spatial pointer memory. arXiv preprint arXiv:2507.02863 (2025)

[^76]: Xing, Z., Zhang, X., Hu, Y., Jiang, B., He, T., Zhang, Q., Long, X., Yin, W.: Goalflow: Goal-driven flow matching for multimodal trajectories generation in end-to-end autonomous driving. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 1602–1611 (2025)

[^77]: Xu, S., Li, F., Huang, P., Song, Z., Yang, Z.X.: Tigdistill-bev: Multi-view bev 3d object detection via target inner-geometry learning distillation. TCSVT (2025)

[^78]: Xu, Y., Hu, Y., Zhang, Z., Meyer, G.P., Mustikovela, S.K., Srinivasa, S., Wolff, E.M., Huang, X.: Vlm-ad: End-to-end autonomous driving through vision-language model supervision. arXiv preprint arXiv:2412.14446 (2024)

[^79]: Xu, Z., Zhang, Y., Xie, E., Zhao, Z., Guo, Y., Wong, K.Y.K., Li, Z., Zhao, H.: Drivegpt4: Interpretable end-to-end autonomous driving via large language model. RA-L (2024)

[^80]: Yang, L., Kang, B., Huang, Z., Zhao, Z., Xu, X., Feng, J., Zhao, H.: Depth anything v2. NeurIPS 37, 21875–21911 (2024)

[^81]: Yang, Z., Chai, Y., Jia, X., Li, Q., Shao, Y., Zhu, X., Su, H., Yan, J.: Drivemoe: Mixture-of-experts for vision-language-action model in end-to-end autonomous driving. arXiv preprint arXiv:2505.16278 (2025)

[^82]: Yao, W., Li, Z., Lan, S., Wang, Z., Sun, X., Alvarez, J.M., Wu, Z.: Drivesuprim: Towards precise trajectory selection for end-to-end planning. arXiv preprint arXiv:2506.06659 (2025)

[^83]: Zeng, W., Luo, W., Suo, S., Sadat, A., Yang, B., Casas, S., Urtasun, R.: End-to-end interpretable neural motion planner. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 8660–8669 (2019)

[^84]: Zhang, S., Huang, W., Gao, Z., Chen, H., Lv, C.: Wisead: Knowledge augmented end-to-end autonomous driving with vision-language model. arXiv preprint arXiv:2412.09951 (2024)

[^85]: Zheng, W., Chen, W., Huang, Y., Zhang, B., Duan, Y., Lu, J.: Occworld: Learning a 3d occupancy world model for autonomous driving. In: ECCV. pp. 55–72. Springer (2024)

[^86]: Zheng, W., Song, R., Guo, X., Zhang, C., Chen, L.: Genad: Generative end-to-end autonomous driving. In: ECCV. pp. 87–104. Springer (2024)

[^87]: Zheng, W., Wu, J., Zheng, Y., Zuo, S., Xie, Z., Yang, L., Pan, Y., Hao, Z., Jia, P., Lang, X., et al.: Gaussianad: Gaussian-centric end-to-end autonomous driving. arXiv preprint arXiv:2412.10371 (2024)

[^88]: Zheng, W., Xia, Z., Huang, Y., Zuo, S., Zhou, J., Lu, J.: Doe-1: Closed-loop autonomous driving with large world model. arXiv preprint arXiv:2412.09627 (2024)

[^89]: Zhou, X., Han, X., Yang, F., Ma, Y., Knoll, A.C.: Opendrivevla: Towards end-to-end autonomous driving with large vision language action model. arXiv preprint arXiv:2503.23463 (2025)

[^90]: Zhou, Z., Cai, T., Zhao, S.Z., Zhang, Y., Huang, Z., Zhou, B., Ma, J.: Autovla: A vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. arXiv preprint arXiv:2506.13757 (2025)

[^91]: Zhu, R., Zhao, J., Zhang, D., Wang, G., Chen, X., Zhang, S., Gong, J., Zhou, Q., Zhang, W., Wang, N., et al.: Sparsead: Sparse query-centric paradigm for efficient end-to-end autonomous driving. IEEE Transactions on Artificial Intelligence (2025)

[^92]: Zhuo, D., Zheng, W., Guo, J., Wu, Y., Zhou, J., Lu, J.: Streaming 4d visual geometry transformer. arXiv preprint arXiv:2507.11539 (2025)

[^93]: Zuo, S., Xie, Z., Zheng, W., Xu, S., Li, F., Jiang, S., Chen, L., Yang, Z.X., Lu, J.: Dvgt: Driving visual geometry transformer. arXiv preprint arXiv:2512.16919 (2025)

[^94]: Zuo, S., Zheng, W., Han, X., Yang, L., Pan, Y., Lu, J.: Quadricformer: Scene as superquadrics for 3d semantic occupancy prediction. arXiv preprint arXiv:2506.10977 (2025)

[^95]: Zuo, S., Zheng, W., Huang, Y., Zhou, J., Lu, J.: Gaussianworld: Gaussian world model for streaming 3d occupancy prediction. In: CVPR. pp. 6772–6781 (2025)
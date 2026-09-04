---
title: "GeoWorldAD: Geometry World Action Model for Autonomous Driving"
source: "https://arxiv.org/html/2607.17521v2"
author:
published:
created: 2026-09-04
description:
tags:
  - "clippings"
---
Songyan Zhang Affiliation: Nanyang Technological University,Xiaomi EV    Jinyuan Tian Affiliation: Zhejiang University    Hanbing Li Affiliation: Nanyang Technological University,Xiaomi EV    Daqi Liu Affiliation: Nanyang Technological University,Xiaomi EV    Hao Chen Affiliation: Zhejiang University    Wenhui Huang    Fang Li Affiliation: Nanyang Technological University,Xiaomi EV    Guang Chen Affiliation: Nanyang Technological University,Xiaomi EV    Hangjun Ye Affiliation: Nanyang Technological University,Xiaomi EV    Long Chen Affiliation: Nanyang Technological University,Xiaomi EV    Kuiyuan Yang Affiliation: Nanyang Technological University,Xiaomi EV    Chen Lv

###### Abstract

Autonomous driving requires both safe and efficient planning decisions in dynamic 3D environments. Although recent Vision/Video-Action models learn policies directly from visual observations and scale well with advances in vision transformers and large-scale training data, they often lack explicit geometric grounding and future-aware spatial guidance, limiting their ability to balance collision avoidance and driving progress. In this work, we propose GeoWorldAD, a geometry world action model that grounds trajectory planning in ego-aligned 3D space and anticipates short-horizon scene evolution with latent future geometry tokens. Present geometry provides essential spatial constraints for safe planning, while future geometry reveals how surrounding agents and ego-centric free space may evolve, reducing overly conservative decisions without sacrificing safety. To efficiently exploit these geometric cues, GeoWorldAD progressively aggregates multi-scale present geometry and latent future geometry through iterative trajectory refinement. Experiments on NAVSIM v1 and v2 demonstrate state-of-the-art performance, highlighting the effectiveness of explicit 3D geometry grounding and future geometry world modeling for safe and efficient autonomous driving.

<sup>3</sup> <sup>2</sup>

> Keywords: Autonomous Driving, Geometry Reconstruction, World Action Model

## 1 Introduction

Autonomous driving requires safe and efficient planning in complex, open-world environments. Conventional systems decompose the driving stack into perception, prediction, planning, and control modules [^20] [^16] [^6] [^15] [^60] [^13] as demonstrated in Fig. 1 (a). Although such modular pipelines improve interpretability, their hand-crafted interfaces may introduce compounding errors and limit scalability. Recent Vision/Video-Action (VA) models learn action-relevant representations directly from visual observations and scale well with advances in representative vision transformer architectures [^39] [^36] [^3] and large-scale training data [^9] [^43] [^2], showing promising results in autonomous driving [^19] [^77] [^76] [^17] [^78] [^28]. Despite this progress, trajectory planning is not merely visual action prediction. It is therefore essential to ground reliable driving decisions in 3D scene geometry, including but not limited to road layout, drivable space, obstacles, dynamic agents, and their spatial relationships, which provide critical spatial cues for avoiding potential collision risks.

Pioneering work [^82] builds the planner on top of a single-layer geometry feature, introducing explicit geometric priors into policy learning as represented in Fig. 1 (b). However, a single geometry layer may struggle to capture the diverse spatial cues required for planning: fine-grained geometry features are helpful for depicting obstacle boundaries and drivable areas, while higher-level geometry features can encode broader scene structure and agent layout. This raises the challenge of designing a geometry-oriented planner that efficiently aggregates multi-scale geometry guidance rather than relying on a single feature layer. Besides, geometry from current observations alone may be insufficient for dynamic driving scenes. Without future-aware guidance, the planner may behave conservatively under uncertainty, which can reduce driving efficiency and limit ego progress. Some early studies [^73] [^62] [^26] use video-generation models as world models to provide future guidance in pixel space, as shown in Fig. 1(c). However, RGB representations are redundant and provide limited geometric guidance. Providing explicit guidance on how surrounding agents and ego-centric free space may evolve in future geometry space to support more effective trajectory planning remains an appealing yet challenging problem for autonomous driving.

![[teaserv3_4hist.png|Refer to caption]]

Figure 1: An intuitive comparison between our video geometry world action model and previous representative pipelines. Given a consecutive video input, our GeoWorldAD provides progressively optimized trajectory planning based on the present and future geometry guidance.

To address these challenges, we propose GeoWorldAD, a unified video geometry world action model built upon StreamVGGT [^80]. As shown in the bottom panel of Fig. 1, GeoWorldAD grounds trajectory planning in ego-aligned 3D geometry and provides planning guidance from both present observations and future scene evolution. Present geometry offers essential spatial constraints from the observed scene, while future geometry provides anticipatory priors on how surrounding agents and ego-centric free space may evolve. Specifically, GeoWorldAD first extracts ego-aligned, multi-scale geometry tokens using a streaming video geometry foundation model. A Q-Former-style geometry world model subsequently learns latent future geometry tokens conditioned on ego states, with future depth prediction providing supervision for capturing short-horizon geometric evolution. Finally, a geometry-oriented action model progressively aggregates multi-scale present geometry and latent future geometry through iterative trajectory refinement, supporting safe and efficient planning. Fig. 1 presents an intuitive comparison between GeoWorldAD and representative planning paradigms. Our contributions are summarized as follows:

1. We formulate autonomous driving planning as a *geometry world action* problem and propose GeoWorldAD, which grounds trajectory planning in explicit, ego-aligned 3D geometry and anticipates short-horizon scene evolution in the same geometric space.
2. We establish present and future geometry as complementary planning guidance: multi-scale present geometry provides spatial constraints for safe motion, while latent future geometry anticipates agent and free-space evolution to support efficient driving progress.
3. Extensive experiments validate the effectiveness of our GeoWorldAD, which achieves the state-of-the-art performance on NAVSIM v1 and NAVSIM v2 benchmarks.

## 2 Related Work

Recent end-to-end autonomous driving systems can be broadly grouped into Vision-Action and Video-Action models. Vision-Action models jointly optimize perception, prediction, and planning within a differentiable framework, mapping visual observations to driving decisions. Representative methods include UniAD [^16], VAD [^20], VADv2 [^6], SparseDrive [^44], WoTE [^27], DiffusionDrive [^29], and DriveSuprim [^68]. These methods improve planning through task-specific queries, structured scene representations, sparse or diffusion-based trajectory generation, and learned proposal evaluation. Despite their strong performance, many rely on structured supervision, such as object annotations, maps, drivable areas, and expert trajectories. Moreover, although some explicitly predict future agents or scene states, future 3D geometry is rarely treated as a primary representation for planning.

Transformer-based Video-Action models further exploit temporal context and benefit from scaling model capacity and training data [^78] [^77] [^28] [^33]. Recent studies additionally incorporate world modeling to anticipate scene evolution from driving videos. Representative methods include DriveVLA-W0 [^26], LFG [^42], Epona [^73], DriveLaW [^62], and WorldDrive [^12]. They improve temporal reasoning through future image generation, self-supervised video learning, diffusion-based rollouts, shared world-planning latent spaces, or trajectory-conditioned prediction. However, many of these approaches learn future guidance primarily in pixel or appearance-oriented latent spaces, with geometry serving as auxiliary supervision rather than the central planning representation.

EponaV2 [^65] takes a step toward geometry-aware anticipation by predicting future semantic and depth maps. Nevertheless, its planning representation is primarily built upon Qwen3-VL features and lacks explicit present-scene grounding from a geometry foundation model, potentially limiting the spatial specificity of its current-scene guidance. In contrast, GeoWorldAD builds upon a video geometry foundation model and adopts ego-aligned present and future geometry as the primary planning representation, providing compact, spatially explicit, and action-relevant 3D guidance for trajectory generation.

### 2.1 Geometry Foundation Models

Recent feed-forward reconstruction methods replace per-scene optimization with direct prediction of 3D representations from sparse or multi-view images, including implicit fields, meshes, Gaussian primitives, radiance fields, and dense point maps [^14] [^55] [^25] [^58] [^50] [^64] [^66] [^46] [^72] [^45] [^11] [^4] [^7] [^5] [^69] [^41] [^21]. Dense coordinate regression has become particularly influential, from CroCo [^59], DUSt3R [^57], and MASt3R [^24] to extensions for dynamic scenes, sparse uncalibrated views, large view sets, and efficient inference [^71] [^74] [^47] [^1] [^67] [^53] [^54]. However, many of these methods still process pairwise, sparse, or full-batch inputs, causing synchronization overhead or quadratic attention growth on long videos. To address this, streaming reconstruction updates geometry online as frames arrive. Existing systems use recurrent tracking, local optimization, neural implicit maps, explicit Gaussian maps, or persistent memory mechanisms for online 3D perception [^48] [^49] [^79] [^52] [^22] [^34] [^35] [^31] [^51] [^56] [^61] [^8] [^75]. Recent transformer-based methods further adopt causal attention and cached historical states, with newer variants exploring bounded or compressed KV caches for long streams [^80] [^23] [^70] [^32] [^30]. In autonomous driving, [^81] [^82] explore the efficient 4D reconstruction tailored for outdoor scenarios.

## 3 Method

As illustrated in Fig. 2, GeoWorldAD consists of three components: a video geometry model, a geometry world model, and a geometry-conditioned action model. Given an input video sequence, the video geometry model extracts spatio-temporal geometric representations, the geometry world model predicts short-term geometric evolution with latent future tokens, and the action model aggregates present and future geometry to generate reliable and safe trajectories.

![[frameworkv8_4hist.png|Refer to caption]]

Figure 2: Overview of the GeoWorldAD framework. Given an input video sequence, GeoWorldAD integrates a video geometry model, a geometry world model, and a geometry-oriented action model for 4D scene reconstruction, future depth estimation, and trajectory planning, respectively. GeoWorldAD grounds trajectory planning in ego-aligned present geometry and latent future geometry, providing both current spatial constraints and future-aware guidance. The decoders of the video geometry and geometry world models are omitted for clarity.

### 3.1 Video Geometry Model

Our video geometry model builds upon StreamVGGT [^80], a streaming video-based 4D geometry foundation model. Given a video sequence of $T$ frames, a DINOv2 [^36] vision encoder maps each input frame $I_{t}\in\mathbb{R}^{3\times H\times W}$ into a sequence of image patch tokens $F_{t}\in\mathbb{R}^{N\times C}$, where $N$ denotes the number of tokens and $C$ denotes the token dimension. A transformer-based decoder, composed of 24 blocks with frame-attention and global-attention modules, is then employed to extract spatio-temporally enhanced geometry tokens. From the intermediate features produced by the decoder, we select tokens $G_{t}^{l}$ from layers $\mathcal{L}=\{4,11,17,23\}$ to construct multi-scale geometry tokens $\mathcal{G}_{t}$:

$$
\displaystyle\mathcal{G}_{t}=\left(G_{t}^{\ell}\right)_{\ell\in\mathcal{L}}=\left(G_{t}^{4},G_{t}^{11},G_{t}^{17},G_{t}^{23}\right).
$$

These multi-scale geometry tokens are fed into DPT heads [^40] to predict point map $P_{t}\in\mathbb{R}^{3\times H\times W}$ and depth map $D_{t}\in\mathbb{R}^{H\times W}$. Camera parameters $g_{t}\in\mathbb{R}^{9}$ are estimated by introducing independent camera tokens that interact with the geometry tokens through iterative geometry decoders. In the standard setting, the predicted sequence of point maps $P_{t}$ is aligned to the anchor coordinate system of the first frame, while the camera pose $g_{t}$ represents the relative transformation from timestep $t$ to the first frame. Such a shared coordinate system enables feed-forward streaming 3D reconstruction.

For autonomous driving, grounding trajectory planning in ego-centric 3D geometry requires the geometry and trajectory representations to share a consistent coordinate system. However, StreamVGGT reconstructs scene geometry in a fixed reference frame, whereas planning trajectories are expressed in the moving ego frame, leading to increasing spatial misalignment over time. To address this issue, we express each point map in the ego-camera coordinate system of its corresponding timestep and represent camera poses as relative transformations between adjacent frames. We term this ego-aligned variant EgoStreamVGGT, which provides spatially consistent geometry tokens for downstream trajectory planning. The loss functions for 4D reconstruction follow the setting in StreamVGGT:

$$
\displaystyle L_{\mathrm{recon}}=L_{\mathrm{camera}}+L_{\mathrm{depth}}+L_{\mathrm{pmap}}.
$$

The details of each loss function are provided in the supplementary material.

### 3.2 Geometry World Model

The architecture of our proposed geometry world model is illustrated in the center of Fig. 2. We maintain a set of learnable latent future tokens $Q_{\mathrm{fut}}\in\mathbb{R}^{K\times M\times C}$, where $K=4$ denotes the number of future chunks for 2 seconds, $M=64$ is the number of latent tokens per chunk. To distinguish different future horizons, we add a learnable temporal embedding to each future chunk. In addition, the ego status, including vehicle velocity, steering state, and high-level driving command, is projected by an MLP into ego embeddings $E_{\mathrm{ego}}$, which are concatenated with the geometry tokens to provide motion-state context. To extract future-aware latent geometry representations for trajectory planning, we introduce a Q-Former-style module with four geometry-guided aggregation stages, corresponding to the selected geometry layers $\mathcal{L}=\{4,11,17,23\}$. Each stage consists of a geometry-future aggregation block followed by a causal future aggregation block. For each selected layer $\ell\in\mathcal{L}$, the future tokens first cross-attend to the present geometry tokens, aggregating spatial-temporal information from the observed scene, which are then updated via causal temporal self-attention in the causal future aggregation block, where the latent future tokens of each future chunk can only attend to themselves and preceding chunks. This causal design preserves the temporal dependency of future predictions. The update process for latent future queries $Q_{\mathrm{fut}}$ interacting with each layer’s geometry tokens can be formulated as:

$$
\displaystyle Q_{\mathrm{fut}}=\mathrm{CausalSelfAttn}\left(\mathrm{CrossAttn}\left(Q_{\mathrm{fut}},\left[G_{t}^{\ell};E_{\mathrm{ego}}\right]\right)\right),
$$

where $[\cdot;\cdot]$ represents the concatenation operation.

After interacting with all multi-scale geometry tokens $\mathcal{G}_{t}$, the refined latent future tokens serve as compact representations of future scene evolution. We further utilize the current geometry tokens $\mathcal{G}_{t}$ and the latent future tokens $Q_{\mathrm{fut}}$ as conditioning variables to predict future geometry representations:

$$
\displaystyle\hat{G}_{t+k}^{\ell}=\mathrm{CrossAttn}\left(G_{t}^{\ell},Q_{\mathrm{fut}}^{k}\right),\quad\ell\in\mathcal{L},\;k=1,\dots,K.
$$

Finally, these future geometry tokens are decoded into future depth maps $\{D_{t+k}\}_{k=1}^{4}$ sharing the same DPT depth head utilized in the video geometry model and supervised with the ground truth future depth maps $\hat{D}$ to compute the geometry world model loss:

$$
L_{\text{wm}}=\sum_{i=t+1}^{t+K}\left\|{\Sigma}_{i}^{D}\odot\left({D}_{i}-\hat{D}_{i}\right)\right\|+\left\|{\Sigma}_{i}^{D}\odot\left(\nabla{D}_{i}-\nabla\hat{D}_{i}\right)\right\|-\alpha\log{\Sigma}_{i}^{D},
$$

where $\odot$ denotes the element-wise product, $\nabla$ indicates the gradient, and ${\Sigma}_{i}^{D}$ is the predicted confidence map. Note that the loss of future depth estimation does not contribute to the weight update of the DPT depth head.

### 3.3 Geometry World Action Model

Given multi-scale geometry tokens $\mathcal{G}_{t}$ extracted by EgoStreamVGGT and latent future tokens $Q_{\mathrm{fut}}$ predicted by the geometry world model, we initialize a set of learnable trajectory queries $Q_{\mathrm{traj}}\in\mathbb{R}^{R\times T_{p}\times d},$ where $R$ is the number of trajectory proposals, $T_{p}$ is the planning horizon, and $d$ is the embedding dimension, which are set to 64, 8, and 1024, respectively.

We first refine the trajectory queries through $L=4$ present-geometry aggregation stages, each associated with one selected geometry scale. At each stage $\ell$, the trajectory queries cross-attend to $\mathcal{G}_{t}^{\ell}$ and the ego status embeddings through a transformer block. A shared MLP then decodes the updated queries into $R$ trajectory proposals. Each proposal contains $T_{p}$ future waypoints parameterized as $(x,y,\theta)$, where the heading angle $\theta$ is constrained to $[-\pi,\pi]$ using a $\tanh$ activation. After aggregating the present geometry, we perform an additional refinement stage using the latent future geometry tokens $Q_{\mathrm{fut}}$. Specifically, the trajectory queries attend to $Q_{\mathrm{fut}}$ through another transformer block, allowing the proposals to incorporate anticipatory guidance about short-horizon scene evolution.

The complete refinement process produces a sequence of trajectory predictions $\{P^{(j)}\}_{j=1}^{N_{\mathrm{ref}}}$, where $N_{\mathrm{ref}}=5$, comprising four present-geometry aggregation stages and one future-geometry refinement stage. We supervise each stage using a minimum-over-proposals objective. Let $P_{r}^{(j)}\in\mathbb{R}^{T_{p}\times 3}$ denote the $r$ -th proposal at stage $j$, and let $\hat{P}_{\mathrm{gt}}\in\mathbb{R}^{T_{p}\times 3}$ denote the ground-truth trajectory. The stage-wise trajectory loss is

$$
\displaystyle L_{\mathrm{traj}}^{(j)}=\min_{r\in\{1,\ldots,R\}}\left\|P_{r}^{(j)}-\hat{P}_{\mathrm{gt}}\right\|_{1}.
$$

The overall trajectory loss is then computed as the weighted sum across all stages:

$$
\displaystyle L_{\mathrm{traj}}=\sum_{j=1}^{N_{\mathrm{ref}}}\lambda_{j}L_{\mathrm{traj}}^{(j)},
$$

where $\lambda_{j}$ exponentially down-weights earlier refinement stages.

Finally, we attach a proposal-scoring head to the final-stage trajectory features. For each proposal, we pool its features along the temporal dimension and apply an MLP to predict a score $S_{r}$. Following [^13] [^10], the target score is defined as:

$$
\displaystyle S_{\mathrm{gt}}=\mathrm{NC}\times\mathrm{DAC}\times\frac{5\,\mathrm{EP}+5\,\mathrm{TTC}+2\,\mathrm{Comf}}{12},
$$

where NC, DAC, EP, TTC, and Comf denote no at-fault collision, drivable-area compliance, ego progress, time-to-collision, and comfort, respectively. All metrics are obtained using the NAVSIM simulator [^10]. The binary cross-entropy loss $L_{\mathrm{score}}$ is applied to optimize the scoring head. The trajectory loss is jointly optimized with the 4D reconstruction loss $L_{\mathrm{recon}}$ and geometry world model loss $L_{\mathrm{wm}}$. The complete training objective for jointly optimizing the video geometry model, geometry world model, and action model is:

$$
\displaystyle L=L_{\mathrm{traj}}+L_{\mathrm{score}}+L_{\mathrm{recon}}+L_{\mathrm{wm}}.
$$

The auxiliary decoders for 4D reconstruction and future-depth prediction are not required during trajectory-planning inference.

## 4 Experiments

### 4.1 Datasets and Evaluation Metrics

Datasets. Our model is trained and evaluated on a large-scale mixture of driving data. For geometry-related tasks, we utilize real-world datasets, namely OpenScene [^9] and nuScenes [^2], alongside synthetic data from ParallelDomain [^37] and RealDriveSim [^18]. Trajectory planning is trained and evaluated on the NAVSIM [^10] dataset, which is a subset of representative scenarios from OpenScene.

Evaluation Metrics. We adopt standard metrics to assess both planning and geometry performance. For closed-loop planning, we benchmark our model on NAVSIM v1 and v2. NAVSIM v1 simulates a 4-second non-reactive environment at 10 Hz, scoring agents via the Predictive Driver Model Score (PDMS), which aggregates fundamental safety, comfort, and progress metrics, as defined in Eq. 8. NAVSIM v2 enhances simulation realism with reactive traffic and introduces the Extended PDMS (EPDMS), which incorporates supplementary criteria such as traffic rule compliance. We also report video depth evaluation and camera pose estimation to demonstrate the geometry performance of our modified EgoStreamVGGT following the evaluation protocol of StreamVGGT. Camera pose evaluation metrics are provided in the supplementary material.

### 4.2 Training Details

The training pipeline for GeoWorldAD proceeds in three distinct stages:

Stage 1: We first pretrain EgoStreamVGGT on a mixture of synthetic and real-world datasets, including OpenScene [^9], nuScenes [^2], ParallelDomain [^37], and RealDriveSim [^18]. The model is initialized from a pretrained StreamVGGT checkpoint and optimized for 23K steps. The sampling ratio among the four datasets is set to $10{:}10{:}1{:}1$, respectively.

Stage 2: Starting from the Stage 1 checkpoint, we train two branches in parallel. First, the geometry world model learns latent future geometry tokens under future-depth supervision while retaining the 4D reconstruction objectives. Given four consecutive input frames, it predicts the depth maps of four future frames spanning the subsequent two seconds. This branch is trained on OpenScene for 47K steps. Meanwhile, we train the trajectory planner to aggregate multi-scale present geometry tokens. Using the standard navtrain split of NAVSIM [^10], the planner takes four consecutive frames as input and predicts eight future waypoints over a four-second horizon. This branch is trained for 32K steps, resulting in an intermediate model termed GeoAD. GeoAD grounds planning in present-scene geometry without incorporating future geometry guidance.

Stage 3: We construct the complete GeoWorldAD model by integrating the GeoAD planner with the geometry world model obtained in Stage 2. To preserve the initial planning behavior of GeoAD, we zero-initialize the output projection of the future-geometry aggregation block. Consequently, this block initially produces zero residual corrections and gradually learns to refine the trajectory proposals using future geometry guidance. The complete model is trained on the NAVSIM navtrain split for an additional 64K steps.

Across all stages, we use the AdamW optimizer with a global batch size of 64 distributed across 32 NVIDIA H20 GPUs. The learning rate is set to $1\times 10^{-4}$ in Stages 1 and 2 and $1\times 10^{-5}$ in Stage 3, following a cosine learning-rate schedule.

### 4.3 Experimental Results

| Method | Input | Aux. Sup. | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf.$\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PARA-Drive [^60] | C | Map & Mot. & Occ | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 |
| VADv2 [^6] | C | Map & Mot. & Traffic | 97.2 | 89.1 | 91.6 | 100 | 76.0 | 80.9 |
| UniAD [^16] | C | Map & Box & Mot. & Occ | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| iPad [^13] | C | Map & Box | 98.6 | 98.3 | 94.9 | 100 | 88.0 | 91.7 |
| Transfuser [^38] | C & L | Map & Box | 97.7 | 92.8 | 92.8 | 100 | 79.2 | 84.0 |
| GoalFlow [^63] | C & L | Map & Box | 98.3 | 93.8 | 94.3 | 100 | 79.8 | 85.7 |
| DiffusionDrive [^29] | C & L | Map & Box | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| WoTE [^27] | C & L | Map & Box | 98.5 | 96.8 | 94.9 | 99.9 | 81.9 | 88.3 |
| DriveSuprim [^68] | C & L | Map & Box | 97.8 | 97.3 | 93.6 | 100 | 86.7 | 89.9 |
| Epona [^73] | C | Future States | 97.9 | 95.1 | 93.8 | 99.9 | 80.4 | 86.2 |
| DriveLaW [^62] | C | Future States | 99.0 | 97.1 | 96.7 | 100 | 81.3 | 89.1 |
| WorldDrive [^12] | C | Future States | 98.4 | 96.8 | 95.2 | 100 | 83.3 | 89.0 |
| DriveVLA-W0 [^26] | C | Future States | 98.7 | 99.1 | 95.3 | 99.3 | 83.3 | 90.2 |
| EponaV2 [^65] | C | Future States | 98.6 | 97.9 | 95.7 | 100 | 84.8 | 90.4 |
| LFG [^42] | C | Dense Geometry | 98.2 | 93.7 | 94.4 | 100 | 79.1 | 85.2 |
| DVGT-2 [^82] | C | Dense Geometry | 98.7 | 97.9 | 95.8 | 100 | 84.3 | 90.3 |
| GeoWorldAD (OURS) | C | Dense & Future Geo. | 99.0 | 97.8 | 95.8 | 99.9 | 85.9 | 91.0 |

Table 1: Closed-loop planning results on NAVSIM v1 navtest split. C and L are short for camera and lidar. Aux. Sup. is short for auxiliary supervision. Our GeoWorldAD achieves the state-of-the-art PDMS metric among competing world-model based and geometry-based methods.

| Method | NC $\uparrow$ | DAC $\uparrow$ | DDC $\uparrow$ | TL $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | LK $\uparrow$ | HC $\uparrow$ | EC $\uparrow$ | EPDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Transfuser [^38] | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 |
| DriveSuprim [^68] | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 | 83.1 |
| DiffusionDrive [^29] | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | 84.5 |
| DriveVLA-W0 [^26] | 98.5 | 99.1 | 98.0 | 99.7 | 86.4 | 98.1 | 93.2 | 97.9 | 58.9 | 86.1 |
| DVGT-2 [^82] | 98.7 | 97.9 | 99.7 | 99.9 | 87.9 | 98.0 | 98.2 | 98.2 | 77.0 | 89.6 |
| EponaV2 [^65] | 98.5 | 97.4 | 99.5 | 99.9 | 87.9 | 98.1 | 97.7 | 98.2 | 77.4 | 88.9 |
| GeoWorldAD (OURS) | 99.0 | 97.8 | 99.6 | 99.7 | 89.1 | 98.6 | 97.6 | 98.0 | 82.2 | 90.4 |

Table 2: Closed-loop planning results on NAVSIM v2 navtest split. Our GeoWorldAD achieves the best EPDMS driving score among all the competing methods.

#### 4.3.1 Closed-Loop Evaluation on NAVSIM Benchmarks

We report closed-loop planning results on NAVSIM v1 and v2 in Tab. 1 and 2, respectively. The compared methods include perception-based pipelines with structured supervision, world-model-based planners using future-state prediction, and geometry-oriented planners using dense present-scene geometry. GeoWorldAD achieves the best performance among perception-free methods on NAVSIM v1 and the highest overall EPDMS on NAVSIM v2.

On NAVSIM v1, GeoWorldAD achieves a PDMS of 91.0, outperforming the strongest dense-geometry baseline, DVGT-2 (90.3), and the strongest future-state-based method, EponaV2 (90.4). Compared with DVGT-2, GeoWorldAD improves EP from 84.3 to 85.9 while increasing NC from 98.7 to 99.0 and maintaining the same TTC of 95.8, suggesting that future geometry guidance improves driving progress without compromising safety. Compared with EponaV2, which introduces future semantic and depth prediction as auxiliary supervision, GeoWorldAD improves PDMS by 0.6 points and EP by 1.1 points, while also achieving higher NC and TTC. This comparison highlights the benefit of grounding trajectory planning in explicit 3D geometry that captures both the observed scene and its anticipated future evolution.

On NAVSIM v2, GeoWorldAD achieves the highest EPDMS of 90.4, exceeding DVGT-2 and EponaV2 by 0.8 and 1.5 points, respectively. It also improves the key safety and efficiency metrics over both methods, achieving 99.0 NC, 98.6 TTC, and 89.1 EP. Overall, these results demonstrate the complementary roles of dense present geometry and latent future geometry: the former provides explicit spatial constraints, while the latter anticipates scene evolution, enabling GeoWorldAD to better balance driving safety and progress.

<table><tbody><tr><th rowspan="2">Method</th><th colspan="4">v1</th><th colspan="4">v2</th></tr><tr><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EPDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><th>GeoAD</th><th>98.9</th><th>95.7</th><th>82.6</th><th>89.3</th><th>98.9</th><th>98.3</th><th>86.3</th><th>87.6</th></tr><tr><th>GeoWorldAD</th><td>99.0</td><td>95.8</td><td>85.9</td><td>91.0</td><td>99.0</td><td>98.6</td><td>89.1</td><td>90.4</td></tr></tbody></table>

Table 3: Ablation study on latent future geometry tokens. Incorporating future geometry anticipation improves ego progress and slightly enhances safety performance.

| Pretrained Model | Aux. Sup. | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf.$\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Scratch | \- | 98.1 | 94.6 | 93.9 | 99.1 | 76.0 | 84.2 |
| StreamVGGT | 4D Recon. | 97.9 | 93.4 | 92.8 | 99.8 | 80.2 | 84.8 |
| EgoStreamVGGT | \- | 98.4 | 95.1 | 95.0 | 99.9 | 81.7 | 87.3 |
| EgoStreamVGGT | 4D Recon. | 98.9 | 97.2 | 95.7 | 99.9 | 82.6 | 89.3 |

Table 4: Ablation study on the effectiveness of geometry representation on the NAVSIM v1 navtest split. Ego-aligned representation and joint geometric supervision introduce consistent improvement.

#### 4.3.2 Ablation Study on the Geometry World Model

We evaluate the effectiveness of the geometry world model in Tab. 3 by comparing GeoAD with the complete GeoWorldAD. GeoAD performs planning solely with multi-scale present geometry, while keeping the remaining components unchanged. It already achieves an NC score of 98.9 on both NAVSIM v1 and v2, indicating that explicit present geometry provides a strong spatial foundation for safety-critical planning. Incorporating latent future geometry tokens consistently improves all reported metrics. On NAVSIM v1, GeoWorldAD improves NC from 98.9 to 99.0, TTC from 95.7 to 95.8, EP from 82.6 to 85.9, and PDMS from 89.3 to 91.0. On NAVSIM v2, it improves NC from 98.9 to 99.0, TTC from 98.3 to 98.6, EP from 86.3 to 89.1, and EPDMS from 87.6 to 90.4. Notably, the largest gains are observed in ego progress, with improvements of 3.3 and 2.8 points on v1 and v2, respectively, while the safety-related metrics are maintained or slightly improved.

These results highlight the central advantage of GeoWorldAD over present-geometry-only planning. Present geometry provides explicit spatial constraints from the observed scene, whereas future geometry anticipates how surrounding agents and ego-centric free space may evolve. Their combination enables more progressive planning under uncertainty without sacrificing safety, leading to a better balance between collision avoidance and driving efficiency.

#### 4.3.3 Ablation Study on Geometry Representation and Supervision

We investigate the effects of geometry representation and joint 4D reconstruction supervision in Tab. 4. To isolate these factors, all variants perform planning using only multi-scale present geometry tokens, without latent future geometry tokens. Training the planner from scratch establishes a baseline PDMS of 84.2. Introducing vanilla StreamVGGT with 4D reconstruction supervision improves EP from 76.0 to 80.2, but yields only a marginal PDMS gain of 0.6 and decreases NC, DAC, and TTC. This mixed result suggests that geometry pretraining alone does not necessarily provide effective planning guidance when its fixed-frame representation is misaligned with the ego-centric coordinate system used for trajectory prediction.

EgoStreamVGGT addresses this discrepancy by representing point maps in their ego-camera frames and camera poses as relative transformations between adjacent frames. Even without auxiliary 4D reconstruction supervision, EgoStreamVGGT improves PDMS from 84.8 to 87.3 over vanilla StreamVGGT, together with consistent gains in NC, DAC, TTC, and EP. This result validates the importance of grounding trajectory planning in coordinate-consistent, ego-centric 3D geometry.

Jointly optimizing the 4D reconstruction objective further improves EgoStreamVGGT across all planning metrics. These gains indicate that additional geometric supervision helps preserve spatially informative representations during planner training, further enhancing driving safety.

Overall, the results show that effective geometry-oriented planning depends on both an ego-aligned representation and joint geometric supervision, rather than geometry pretraining alone. This configuration forms our present-geometry planner, GeoAD, which is further extended with latent future geometry tokens in GeoWorldAD. The corresponding video depth evaluation is reported in Tab. 5. Detailed experiment analysis on 4D reconstruction can be found in the supplementary materials.

<table><thead><tr><th></th><th></th><th colspan="2">OpenScene</th><th colspan="2">nuScenes</th><th colspan="2">KITTI</th></tr><tr><th>Method</th><th>Type</th><th>Abs Rel <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th><th><math><semantics><mrow><mi>𝜹</mi> <mo><</mo> <mn>1.25</mn> <mo>↑</mo></mrow> <annotation>\boldsymbol{\delta<1.25\uparrow}</annotation></semantics></math></th><th>Abs Rel <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th><th><math><semantics><mrow><mi>𝜹</mi> <mo><</mo> <mn>1.25</mn> <mo>↑</mo></mrow> <annotation>\boldsymbol{\delta<1.25\uparrow}</annotation></semantics></math></th><th>Abs Rel <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th><th><math><semantics><mrow><mi>𝜹</mi> <mo><</mo> <mn>1.25</mn> <mo>↑</mo></mrow> <annotation>\boldsymbol{\delta<1.25\uparrow}</annotation></semantics></math></th></tr></thead><tbody><tr><th>StreamVGGT</th><th>Streaming</th><td>0.236</td><td>65.6</td><td>0.265</td><td>58.2</td><td>0.173</td><td>72.2</td></tr><tr><th>EgoStreamVGGT</th><th>Streaming</th><td>0.141</td><td>86.5</td><td>0.117</td><td>88.5</td><td>0.077</td><td>95.5</td></tr></tbody></table>

Table 5: Video depth estimation. Our finetuned EgoStreamVGGT achieves consistent improvements across OpenScene, nuScenes, and KITTI datasets.

## 5 Limitations

Although EgoStreamVGGT supports continuous video streams for 4D reconstruction, the current trajectory planner operates on fixed-length clips. Integrating KV caching to enable efficient streaming inference for trajectory planning remains a promising direction for future work.

## 6 Conclusion

We propose GeoWorldAD, a unified video geometry world action model that grounds autonomous driving planning in ego-aligned 3D geometry. GeoWorldAD combines multi-scale present geometry with latent future geometry modeling, providing complementary spatial constraints and anticipatory guidance for trajectory planning. Its geometry-oriented action model progressively aggregates these geometric cues through iterative trajectory refinement, improving driving progress while preserving safety. Extensive experiments on NAVSIM v1 and v2 demonstrate state-of-the-art closed-loop performance and validate the importance of coordinate-consistent geometry grounding and future geometry anticipation for safe and efficient planning.

## References

## Appendix A Appendix for GeoWorldAD

### A.1 Video Geometry Reconstruction Losses

Following StreamVGGT, EgoStreamVGGT is trained with three reconstruction objectives: a camera loss $L_{\mathrm{camera}}$, a depth loss $L_{\mathrm{depth}}$, and a point-map loss $L_{\mathrm{pmap}}$. These objectives correspond to Eq. 2 in the main paper:

$$
\displaystyle L_{\mathrm{recon}}=L_{\mathrm{camera}}+L_{\mathrm{depth}}+L_{\mathrm{pmap}}.
$$

Given an input video sequence of $T$ frames, EgoStreamVGGT predicts the camera parameters $g_{t}$, depth map $D_{t}\in\mathbb{R}^{H\times W}$, point map $P_{t}\in\mathbb{R}^{3\times H\times W}$, and the corresponding confidence maps for each frame $t$. Their ground-truth values are denoted by $\hat{g}_{t}$, $\hat{D}_{t}$, and $\hat{P}_{t}$, respectively.

Unlike the anchor-frame representation adopted by StreamVGGT, EgoStreamVGGT expresses each point map $P_{t}$ in the ego-camera coordinate system of its corresponding timestep. The camera parameters $g_{t}$ encode the relative camera transformation between adjacent frames. This formulation retains temporal ego-motion information while keeping the reconstructed geometry aligned with the ego-centric coordinate system used for trajectory planning.

##### Camera loss.

The camera loss supervises the predicted camera parameters using their ground-truth values. Following StreamVGGT, we adopt the Huber loss for robustness to noisy camera annotations and outliers:

$$
\displaystyle L_{\mathrm{camera}}=\sum_{t=1}^{T}\rho_{\epsilon}\left(g_{t}-\hat{g}_{t}\right),
$$

where $\rho_{\epsilon}(\cdot)$ denotes the element-wise Huber loss. The camera parameter $g_{t}\in\mathbb{R}^{9}$ contains the relative translation and rotation between adjacent frames, as well as the field-of-view parameters. This objective provides explicit supervision for relative ego-motion estimation, enabling the model to relate consecutive ego-centric geometry representations over time.

##### Depth loss.

The depth loss supervises the predicted depth map $D_{t}$ using the ground-truth depth $\hat{D}_{t}$. Since depth quality varies across pixels due to occlusion, motion, and sparse or noisy supervision, the loss is weighted by the predicted depth confidence map $\Sigma_{t}^{D}$. In addition to the depth-value error, we follow StreamVGGT and include a gradient matching term to encourage local geometric consistency and sharper depth discontinuities:

$$
\displaystyle L_{\mathrm{depth}}=\sum_{t=1}^{T}\left(\left|\Sigma_{t}^{D}\odot(D_{t}-\hat{D}_{t})\right|+\left|\Sigma_{t}^{D}\odot(\nabla D_{t}-\nabla\hat{D}_{t})\right|-\alpha\log\Sigma_{t}^{D}\right),
$$

where $\odot$ denotes element-wise multiplication, $\nabla$ denotes the spatial gradient operator, and $\alpha$ controls the confidence regularization. The confidence term allows the model to down-weight uncertain regions, while the negative logarithmic regularizer prevents the confidence from collapsing to trivial low values. This loss encourages accurate dense depth prediction while preserving object boundaries and road-surface geometry.

##### Point-map loss.

The point-map loss supervises the predicted 3D point map $P_{t}$ with the ground-truth point map $\hat{P}_{t}$. It uses the predicted point-map confidence $\Sigma_{t}^{P}$ and includes both point-coordinate and gradient-consistency terms:

$$
\displaystyle L_{\mathrm{pmap}}=\sum_{t=1}^{T}\left(\left|\Sigma_{t}^{P}\odot(P_{t}-\hat{P}_{t})\right|+\left|\Sigma_{t}^{P}\odot(\nabla P_{t}-\nabla\hat{P}_{t})\right|-\alpha\log\Sigma_{t}^{P}\right).
$$

Unlike the vanilla implementation, we modify the coordinate system from the shared reference frame to the corresponding ego-camera coordinate system in EgoStreamVGGT.

### A.2 Collision-Related Metrics

Among the closed-loop planning metrics, no-at-fault collision (NC) and time-to-collision (TTC) are most directly related to driving safety. NC measures whether the ego vehicle completes the simulated scenario without being responsible for a collision. It therefore reflects the planner’s ability to respect the spatial occupancy of surrounding vehicles, pedestrians, and static obstacles under closed-loop execution. A high NC score indicates that the generated trajectory remains collision-free even after the simulator rolls out the consequences of the ego action, making it a fundamental constraint for evaluating autonomous driving safety.

TTC further complements NC by measuring the temporal margin before a potential collision. While NC captures whether a collision eventually occurs, TTC evaluates whether the ego vehicle maintains sufficient reaction time with respect to nearby dynamic agents. A low TTC usually indicates that the ego vehicle is too close to another road user, given their relative velocity, even if an actual collision has not yet happened within the simulation horizon. Therefore, TTC serves as an early-warning metric for near-collision and high-risk behaviors.

These two metrics are particularly important for GeoWorldAD because our method explicitly introduces present and future geometric priors into trajectory planning. Present geometry helps the planner understand the current drivable space and obstacle layout, while future geometry provides anticipatory cues about how the scene may evolve. By jointly optimizing with these geometric constraints, the planner can reduce both immediate collision risks, reflected by NC, and latent near-collision risks, reflected by TTC. Consequently, consistent gains in NC and TTC demonstrate that GeoWorldAD improves not only trajectory accuracy or progress, but also the safety-critical behavior required for reliable autonomous driving.

![[supp_planning_geo.png|Refer to caption]]

Figure 3: Comparison of three different geometry aggregation strategies for trajectory planning.

### A.3 Additional Experiments

#### A.3.1 Exploration on Geometry Aggregation

As shown in Fig. 3, we compare different strategies for aggregating geometry priors into the trajectory planner. The first strategy, illustrated on the left of Fig. 3 (a), directly lets the trajectory tokens interact with the geometry tokens from all 24 layers, followed by a single trajectory decoder for supervision. This design is similar to the strategy used in [^82]. As shown in the first row of Tab. 6, although the trajectory tokens can access rich multi-layer geometry features, the planner still obtains sub-optimal performance. We attribute this to the limited optimization depth for trajectory planning: a single interaction stage is insufficient for efficiently absorbing both low-level spatial details and high-level semantic geometry cues.

To encourage more progressive interaction, we further evaluate the middle design in Fig. 3 (b), where the trajectory tokens are placed on top of the final-layer geometry tokens and are optimized iteratively through a shared trajectory planning decoder. This design allows the trajectory representation to be gradually refined. As shown in the second row of Tab. 6, the ego-progress metric improves from 81.5 to 82.9, indicating that iterative refinement helps generate more progressive trajectories. However, because the planner only accesses the final geometry layer, the improvements in collision-related metrics remain limited. This suggests that high-level geometry alone cannot provide all the fine-grained spatial constraints required for safe planning.

Our final design, adopted in GeoWorldAD as shown in the right panel of Fig. 3 (c), combines the advantages of both strategies. We introduce multi-scale geometry tokens from the DPT head, which contain spatial information from shallow to deep layers, and perform iterative trajectory optimization after each geometry aggregation step. In this way, the trajectory tokens can progressively absorb geometry guidance from different representation levels. Because supervision is applied at each refinement stage, the planner learns to efficiently extract safety-critical spatial cues and convert them into collision-aware trajectory updates. As shown in the last row of Tab. 6, this strategy achieves the best overall performance, especially on NC and TTC, demonstrating that multi-scale and iterative geometry aggregation is important for safe and reliable planning.

| Geo. Lay. Num. | Iter. Opt. Num. | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf.$\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 24 | 1 | 98.5 | 95.7 | 95.1 | 99.7 | 81.5 | 87.6 |
| 1 | 4 | 98.6 | 95.5 | 95.2 | 99.8 | 82.9 | 88.2 |
| 4 | 4 | 98.9 | 97.2 | 95.7 | 99.9 | 82.6 | 89.3 |

Table 6: Quantitative comparison of three geometry aggregation strategies. GeoWorldAD employs a geometry-oriented planner that progressively aggregates multi-scale geometric features through iterative optimization. “Geo. Lay. Num.” denotes the number of geometry token layers used in the planner, while “Iter. Opt. Num.” denotes the number of iterative trajectory optimization steps.

#### A.3.2 Geometry Evaluation of EgoStreamVGGT

<table><thead><tr><th></th><th></th><th colspan="2">OpenScene</th><th colspan="2">nuScenes</th><th colspan="2">KITTI</th></tr><tr><th>Method</th><th>Type</th><th>Abs Rel <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th><th><math><semantics><mrow><mi>𝜹</mi> <mo><</mo> <mn>1.25</mn> <mo>↑</mo></mrow> <annotation>\boldsymbol{\delta<1.25\uparrow}</annotation></semantics></math></th><th>Abs Rel <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th><th><math><semantics><mrow><mi>𝜹</mi> <mo><</mo> <mn>1.25</mn> <mo>↑</mo></mrow> <annotation>\boldsymbol{\delta<1.25\uparrow}</annotation></semantics></math></th><th>Abs Rel <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th><th><math><semantics><mrow><mi>𝜹</mi> <mo><</mo> <mn>1.25</mn> <mo>↑</mo></mrow> <annotation>\boldsymbol{\delta<1.25\uparrow}</annotation></semantics></math></th></tr></thead><tbody><tr><th>StreamVGGT</th><th>Streaming</th><td>0.236</td><td>65.6</td><td>0.265</td><td>58.2</td><td>0.173</td><td>72.2</td></tr><tr><th>EgoStreamVGGT</th><th>Streaming</th><td>0.141</td><td>86.5</td><td>0.117</td><td>88.5</td><td>0.077</td><td>95.5</td></tr></tbody></table>

Table 7: Video depth estimation. Our finetuned EgoStreamVGGT achieves consistent improvements across OpenScene, nuScenes, and KITTI datasets.

<table><thead><tr><th></th><th></th><th colspan="3">nuScenes</th><th colspan="3">OpenScene</th></tr></thead><tbody><tr><th>Method</th><th>Type</th><td>ATE <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>RPE trans <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>RPE rot <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>ATE <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>RPE trans <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>RPE rot <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><th>StreamVGGT</th><th>Streaming</th><th>14.79</th><th>1.77</th><th>0.47</th><th>8.66</th><th>1.00</th><th>1.53</th></tr><tr><th>EgoStreamVGGT</th><th>Streaming</th><td>5.78</td><td>0.63</td><td>1.31</td><td>4.07</td><td>0.39</td><td>0.92</td></tr></tbody></table>

Table 8: Camera Pose Estimation Evaluation on nuScenes and OpenScene datasets.

To evaluate the effectiveness of the proposed ego-aligned geometry representation, we compare EgoStreamVGGT with the original StreamVGGT on both video depth estimation and camera pose estimation. The results are reported in Tab. 7 and Tab. 8.

As shown in Tab. 7, EgoStreamVGGT consistently improves depth estimation performance across OpenScene, nuScenes, and KITTI. Compared with StreamVGGT, our model achieves lower absolute relative error and higher threshold accuracy on all three datasets. These results demonstrate that adapting the geometry backbone to the ego-centric driving setting effectively improves its ability to recover road-scene geometry. This improvement is especially important for autonomous driving, where accurate depth estimation helps the planner better understand drivable space, obstacle locations, and the spatial layout of surrounding agents. The consistent gains across different datasets also indicate that EgoStreamVGGT does not overfit to a single domain, but learns more robust geometry representations for diverse driving scenarios.

Tab. 8 further evaluates camera pose estimation. EgoStreamVGGT significantly reduces trajectory-level and translational pose errors on both nuScenes and OpenScene, showing that the ego-aligned finetuning improves temporal geometry consistency in streaming video reconstruction. More accurate ego-motion estimation is crucial for GeoWorldAD because geometry tokens from different frames must be aligned in a consistent coordinate system before they can be used for planning. By reducing pose drift and frame-to-frame misalignment, EgoStreamVGGT provides more stable geometry features for downstream trajectory optimization.

Overall, the geometry evaluation verifies that EgoStreamVGGT provides a stronger foundation for GeoWorldAD than the original StreamVGGT. Through ego-centric adaptation, the model obtains more accurate depth prediction and more consistent camera motion estimation, which together improve the quality of the geometry tokens used by the planner. These results support our design choice of building GeoWorldAD upon EgoStreamVGGT, as reliable geometry perception is essential for collision-aware and temporally consistent autonomous driving planning. Additional 4D reconstruction visualizations are provided in Sec. A.4.

### A.4 Visualization Results

#### A.4.1 4D Reconstruction Visualization

![[recon_vis_0.png|Refer to caption]]

Figure 4: Visual comparison of StreamVGGT and our EgoStreamVGGT for 4D reconstruction.

![[recon_vis_1.png|Refer to caption]]

Figure 5: Visualized comparison of StreamVGGT and our EgoStreamVGGT for 4D reconstruction.

![[recon_vis_2.png|Refer to caption]]

Figure 6: Visualized comparison of StreamVGGT and our EgoStreamVGGT for 4D reconstruction.

![[recon_vis_3.png|Refer to caption]]

Figure 7: Visualized comparison of StreamVGGT and our EgoStreamVGGT for 4D reconstruction.

#### A.4.2 Visualization of Future Depth Prediction

![[wm1.png|Refer to caption]]

Figure 8: Visualization of future depth prediction.

![[wm2.png|Refer to caption]]

Figure 9: Visualization of future depth prediction.

[^1]: Y. Cabon, L. Stoffl, L. Antsfeld, G. Csurka, B. Chidlovskii, J. Revaud, and V. Leroy (2025) MUSt3R: multi-view network for stereo 3d reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1050–1060. Cited by: §2.1.

[^2]: H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom (2020) Nuscenes: a multimodal dataset for autonomous driving. In Proc. IEEE Conf. Comp. Vis. Patt. Recogn., pp. 11621–11631. Cited by: §1, §4.1, §4.2.

[^3]: M. Caron, H. Touvron, I. Misra, H. Jégou, J. Mairal, P. Bojanowski, and A. Joulin (2021) Emerging properties in self-supervised vision transformers. In ICCV, pp. 9650–9660. Cited by: §1.

[^4]: D. Charatan, S. L. Li, A. Tagliasacchi, and V. Sitzmann (2024) PixelSplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 19457–19467. Cited by: §2.1.

[^5]: A. Chen, H. Xu, S. Esposito, S. Tang, and A. Geiger (2024) LaRa: efficient large-baseline radiance fields. In Proceedings of the European Conference on Computer Vision (ECCV), Cited by: §2.1.

[^6]: S. Chen, B. Jiang, H. Gao, B. Liao, Q. Xu, Q. Zhang, C. Huang, W. Liu, and X. Wang (2024) Vadv2: end-to-end vectorized autonomous driving via probabilistic planning. arXiv preprint arXiv:2402.13243. Cited by: §1, §2, Table 1.

[^7]: Y. Chen, H. Xu, C. Zheng, B. Zhuang, M. Pollefeys, A. Geiger, T. Cham, and J. Cai (2024) MVSplat: efficient 3d gaussian splatting from sparse multi-view images. In Proceedings of the European Conference on Computer Vision (ECCV), pp. 370–386. External Links: [Document](https://dx.doi.org/10.1007/978-3-031-72664-4%5F21) Cited by: §2.1.

[^8]: Z. Chen, M. Qin, T. Yuan, Z. Liu, and H. Zhao (2025) LONG3R: long sequence streaming 3d reconstruction. arXiv preprint arXiv:2507.18255. Cited by: §2.1.

[^9]: O. Contributors (2023) OpenScene: the largest up-to-date 3d occupancy prediction benchmark in autonomous driving. Note: [GitHub-OpenDriveLab/OpenScene:3DOccupancyPredictionBenchmarkinAutonomousDriving](https://github-opendrivelab/OpenScene:3DOccupancyPredictionBenchmarkinAutonomousDriving) Cited by: §1, §4.1, §4.2.

[^10]: D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, A. Geiger, and K. Chitta (2024) NAVSIM: data-driven non-reactive autonomous vehicle simulation and benchmarking. In Adv. Neural Inf. Process. Syst., Vol. 37, pp. 28706–28719. Cited by: §3.3, §3.3, §4.1, §4.2.

[^11]: R. Gao, A. Holynski, P. Henzler, A. Brussee, R. Martin-Brualla, P. Srinivasan, J. T. Barron, and B. Poole (2024) CAT3D: create anything in 3d with multi-view diffusion models. In Advances in Neural Information Processing Systems (NeurIPS), Cited by: §2.1.

[^12]: X. Gui, M. Zhang, T. Yan, W. Han, J. Gong, F. Tan, C. Xu, and J. Shen (2026) Bridging scene generation and planning: driving with world model via unifying vision and motion representation. External Links: 2603.14948, [Link](https://arxiv.org/abs/2603.14948) Cited by: §2, Table 1.

[^13]: K. Guo, H. Liu, X. Wu, J. Pan, and C. Lv (2025) IPad: iterative proposal-centric end-to-end autonomous driving. External Links: 2505.15111, [Link](https://arxiv.org/abs/2505.15111) Cited by: §1, §3.3, Table 1.

[^14]: Y. Hong, K. Zhang, J. Gu, S. Bi, Y. Zhou, H. Liu, K. Liu, S. Soatto, C. Fowlkes, and H. Tan (2023) LRM: large reconstruction model for single image to 3d. arXiv preprint arXiv:2311.04400. Cited by: §2.1.

[^15]: S. Hu, L. Chen, P. Wu, H. Li, J. Yan, and D. Tao (2022) St-p3: end-to-end vision-based autonomous driving via spatial-temporal feature learning. In Proc. Eur. Conf. Comp. Vis., pp. 533–549. Cited by: §1.

[^16]: Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, L. Lu, X. Jia, Q. Liu, J. Dai, Y. Qiao, and H. Li (2023) Planning-oriented autonomous driving. In Proc. IEEE Conf. Comp. Vis. Patt. Recogn., pp. 17853–17862. Cited by: §1, §2, Table 1.

[^17]: W. Huang, S. Zhang, Q. Huang, Z. Wang, Z. Mao, C. Chua, Z. Chen, L. Chen, and C. Lv (2026) AutoMoT: a unified vision-language-action model with asynchronous mixture-of-transformers for end-to-end autonomous driving. External Links: 2603.14851, [Link](https://arxiv.org/abs/2603.14851) Cited by: §1.

[^18]: A. Jadon, H. Wang, P. Thomas, M. Stanley, S. N. Cibik, R. Laurat, O. Maher, L. Hoyer, O. Unal, and D. Dai (2025) RealDriveSim: a realistic multi-modal multi-task synthetic dataset for autonomous driving. External Links: 2506.16319, [Link](https://arxiv.org/abs/2506.16319) Cited by: §4.1, §4.2.

[^19]: X. Jia, J. You, Z. Zhang, and J. Yan (2025) Drivetransformer: unified transformer for scalable end-to-end autonomous driving. arXiv preprint arXiv:2503.07656. Cited by: §1.

[^20]: B. Jiang, S. Chen, Q. Xu, B. Liao, J. Chen, H. Zhou, Q. Zhang, W. Liu, C. Huang, and X. Wang (2023) VAD: vectorized scene representation for efficient autonomous driving. In Proc. IEEE Int. Conf. Comp. Vis., pp. 8306–8316. Cited by: §1, §2.

[^21]: L. Jiang, Y. Mao, L. Xu, T. Lu, K. Ren, Y. Jin, X. Xu, M. Yu, J. Pang, F. Zhao, D. Lin, and B. Dai (2025) AnySplat: feed-forward 3d gaussian splatting from unconstrained views. arXiv preprint arXiv:2505.23716. Cited by: §2.1.

[^22]: N. Keetha, J. Karhade, K. M. Jatavallabhula, G. Yang, S. Scherer, D. Ramanan, and J. Luiten (2024) SplaTAM: splat track & map 3d gaussians for dense rgb-d slam. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 21357–21366. Cited by: §2.1.

[^23]: Y. Lan, Y. Luo, F. Hong, S. Zhou, H. Chen, Z. Lyu, S. Yang, B. Dai, C. C. Loy, and X. Pan (2025) STream3R: scalable sequential 3d reconstruction with causal transformer. arXiv preprint arXiv:2508.10893. Cited by: §2.1.

[^24]: V. Leroy, Y. Cabon, and J. Revaud (2024) Grounding image matching in 3d with mast3r. arXiv preprint arXiv:2406.09756. Cited by: §2.1.

[^25]: J. Li, H. Tan, K. Zhang, Z. Xu, F. Luan, Y. Xu, Y. Hong, K. Sunkavalli, G. Shakhnarovich, and S. Bi (2023) Instant3D: fast text-to-3d with sparse-view generation and large reconstruction model. arXiv preprint arXiv:2311.06214. Cited by: §2.1.

[^26]: Y. Li, S. Shang, W. Liu, B. Zhan, H. Wang, Y. Wang, Y. Chen, X. Wang, Y. An, C. Tang, et al. (2025) DriveVLA-w0: world models amplify data scaling law in autonomous driving. arXiv preprint arXiv:2510.12796. Cited by: §1, §2, Table 1, Table 2.

[^27]: Y. Li, Y. Wang, Y. Liu, J. He, L. Fan, and Z. Zhang (2025) End-to-end driving with online trajectory evaluation via bev world model. External Links: 2504.01941, [Link](https://arxiv.org/abs/2504.01941) Cited by: §2, Table 1.

[^28]: Y. Li, K. Xiong, X. Guo, F. Li, S. Yan, G. Xu, L. Zhou, L. Chen, H. Sun, B. Wang, et al. (2025) Recogdrive: a reinforced cognitive framework for end-to-end autonomous driving. arXiv preprint arXiv:2506.08052. Cited by: §1, §2.

[^29]: B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang, et al. (2025) Diffusiondrive: truncated diffusion model for end-to-end autonomous driving. In Proc. IEEE Conf. Comp. Vis. Patt. Recogn., pp. 12037–12047. Cited by: §2, Table 1, Table 2.

[^30]: X. Liu, C. Yu, D. Ji, Q. Zhu, L. Sun, X. Li, J. Ma, T. Chen, and L. Zhu (2026) StreamCacheVGGT: streaming visual geometry transformers with robust scoring and hybrid cache compression. arXiv preprint arXiv:2604.15237. Cited by: §2.1.

[^31]: Y. Liu, S. Dong, S. Wang, Y. Yin, Y. Yang, Q. Fan, and B. Chen (2025) SLAM3R: real-time dense scene reconstruction from monocular rgb videos. In CVPR, pp. 16651–16662. Cited by: §2.1.

[^32]: S. Lu, P. Chen, H. Hsu, S. Jhong, W. Cheng, and Y. Chen (2026) OVGGT: o(1) constant-cost streaming visual geometry transformer. arXiv preprint arXiv:2603.05959. Cited by: §2.1.

[^33]: Y. Luo, F. Li, S. Xu, Z. Lai, L. Yang, Q. Chen, Z. Luo, Z. Xie, S. Jiang, J. Liu, et al. (2025) Adathinkdrive: adaptive thinking via reinforcement learning for autonomous driving. arXiv preprint arXiv:2509.13769. Cited by: §2.

[^34]: H. Matsuki, R. Murai, P. H. J. Kelly, and A. J. Davison (2024) Gaussian splatting slam. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 18039–18048. Cited by: §2.1.

[^35]: R. Murai, E. Dexheimer, and A. J. Davison (2025) MASt3R-slam: real-time dense slam with 3d reconstruction priors. In CVPR, pp. 16695–16705. Cited by: §2.1.

[^36]: M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al. (2023) Dinov2: learning robust visual features without supervision. arXiv preprint arXiv:2304.07193. Cited by: §1, §3.1.

[^37]: (2024) Parallel domain. Note: [https://paralleldomain.com/](https://paralleldomain.com/) Cited by: §4.1, §4.2.

[^38]: A. Prakash, K. Chitta, and A. Geiger (2021) Multi-modal fusion transformer for end-to-end autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 7077–7087. Cited by: Table 1, Table 2.

[^39]: A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. (2021) Learning transferable visual models from natural language supervision. In Proc. Int. Conf. Learn. Representations, pp. 8748–8763. Cited by: §1.

[^40]: R. Ranftl, A. Bochkovskiy, and V. Koltun (2021) Vision transformers for dense prediction. In ICCV, pp. 12179–12188. Cited by: §3.1.

[^41]: B. Smart, C. Zheng, I. Laina, and V. A. Prisacariu (2024) Splatt3R: zero-shot gaussian splatting from uncalibrated image pairs. arXiv preprint arXiv:2408.13912. Cited by: §2.1.

[^42]: M. Strong, W. Chang, Q. Herau, J. Yang, Y. Hu, C. Peng, and W. Zhan (2026) Learning to drive is a free gift: large-scale label-free autonomy pretraining from unposed in-the-wild videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Cited by: §2, Table 1.

[^43]: P. Sun, H. Kretzschmar, X. Dotiwalla, A. Chouard, V. Patnaik, P. Tsui, J. Guo, Y. Zhou, Y. Chai, B. Caine, et al. (2020) Scalability in perception for autonomous driving: waymo open dataset. In Proc. IEEE Conf. Comp. Vis. Patt. Recogn., pp. 2446–2454. Cited by: §1.

[^44]: W. Sun, X. Lin, Y. Shi, C. Zhang, H. Wu, and S. Zheng (2025) SparseDrive: end-to-end autonomous driving via sparse scene representation. In IEEE Int. Conf. Robot. Autom., pp. 8795–8801. Cited by: §2.

[^45]: S. Szymanowicz, C. Rupprecht, and A. Vedaldi (2024) Splatter image: ultra-fast single-view 3d reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10208–10217. Cited by: §2.1.

[^46]: J. Tang, Z. Chen, X. Chen, T. Wang, G. Zeng, and Z. Liu (2024) LGM: large multi-view gaussian model for high-resolution 3d content creation. arXiv preprint arXiv:2402.05054. Cited by: §2.1.

[^47]: Z. Tang, Y. Fan, D. Wang, H. Xu, R. Ranjan, A. Schwing, and Z. Yan (2025) MV-dust3r+: single-stage scene reconstruction from sparse views in 2 seconds. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5283–5293. Cited by: §2.1.

[^48]: Z. Teed, L. Lipson, and J. Deng (2021) DROID-slam: deep visual slam for monocular, stereo, and rgb-d cameras. In Advances in Neural Information Processing Systems (NeurIPS), Vol. 34, pp. 16558–16569. Cited by: §2.1.

[^49]: Z. Teed, L. Lipson, and J. Deng (2023) Deep patch visual odometry. In Advances in Neural Information Processing Systems (NeurIPS), Vol. 36. Cited by: §2.1.

[^50]: D. Tochilkin, D. Pankratz, Z. Liu, Z. Huang, A. Letts, Y. Li, D. Liang, C. Laforte, V. Jampani, and Y. Cao (2024) TripoSR: fast 3d object reconstruction from a single image. arXiv preprint arXiv:2403.02151. Cited by: §2.1.

[^51]: H. Wang and L. Agapito (2025) Spann3R: 3d reconstruction with spatial memory. In International Conference on 3D Vision (3DV), Cited by: §2.1.

[^52]: H. Wang, J. Wang, and L. Agapito (2023) Co-slam: joint coordinate and sparse parametric encodings for neural real-time slam. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 13293–13302. Cited by: §2.1.

[^53]: J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, and D. Novotny (2025) Vggt: visual geometry grounded transformer. In CVPR, pp. 5294–5306. Cited by: §2.1.

[^54]: J. Wang, J. Schönberger, M. Chen, S. Zhang, N. Karaev, P. Labatut, A. Vedaldi, P. Bojanowski, C. Rupprecht, and D. Novotny (2026) VGGT- $\Omega$. arXiv preprint arXiv:2605.15195. Cited by: §2.1.

[^55]: P. Wang, H. Tan, S. Bi, Y. Xu, F. Luan, K. Sunkavalli, W. Wang, Z. Xu, and K. Zhang (2023) PF-lrm: pose-free large reconstruction model for joint pose and shape prediction. arXiv preprint arXiv:2311.12024. Cited by: §2.1.

[^56]: Q. Wang, Y. Zhang, A. Holynski, A. A. Efros, and A. Kanazawa (2025) Continuous 3d perception model with persistent state. In CVPR, Cited by: §2.1.

[^57]: S. Wang, V. Leroy, Y. Cabon, B. Chidlovskii, and J. Revaud (2024) Dust3r: geometric 3d vision made easy. In CVPR, pp. 20697–20709. Cited by: §2.1.

[^58]: Z. Wang, Y. Wang, Y. Chen, C. Xiang, S. Chen, D. Yu, C. Li, H. Su, and J. Zhu (2024) CRM: single image to 3d textured mesh with convolutional reconstruction model. arXiv preprint arXiv:2403.05034. Cited by: §2.1.

[^59]: P. Weinzaepfel, R. Brégier, T. Combaluzier, Y. Cabon, and J. Revaud (2022) CroCo: cross-view completion pre-training for 3d vision. In NIPS, pp. 3216–3229. Cited by: §2.1.

[^60]: X. Weng, B. Ivanovic, Y. Wang, Y. Wang, and M. Pavone (2024) Para-drive: parallelized architecture for real-time autonomous driving. In Proc. IEEE Conf. Comp. Vis. Patt. Recogn., pp. 15449–15458. Cited by: §1, Table 1.

[^61]: Y. Wu, W. Zheng, J. Zhou, and J. Lu (2025) Point3R: streaming 3d reconstruction with explicit spatial pointer memory. In NIPS, Cited by: §2.1.

[^62]: T. Xia, Y. Li, L. Zhou, J. Yao, K. Xiong, H. Sun, B. Wang, K. Ma, H. Ye, W. Liu, et al. (2025) DriveLaW: unifying planning and video generation in a latent driving world. arXiv preprint arXiv:2512.23421. Cited by: §1, §2, Table 1.

[^63]: Z. Xing, X. Zhang, Y. Hu, B. Jiang, T. He, Q. Zhang, X. Long, and W. Yin (2025) GoalFlow: goal-driven flow matching for multimodal trajectories generation in end-to-end autonomous driving. In CVPR, pp. 1602–1611. Cited by: Table 1.

[^64]: J. Xu, W. Cheng, Y. Gao, X. Wang, S. Gao, and Y. Shan (2024) InstantMesh: efficient 3d mesh generation from a single image with sparse-view large reconstruction models. arXiv preprint arXiv:2404.07191. Cited by: §2.1.

[^65]: J. Xu, Z. Zhong, Z. Shu, M. Jia, M. Li, J. Bian, Q. Zhang, K. Zhang, J. Xie, J. Yang, and W. Yin (2026) EponaV2: driving world model with comprehensive future reasoning. External Links: 2605.14696, [Link](https://arxiv.org/abs/2605.14696) Cited by: §2, Table 1, Table 2.

[^66]: Y. Xu, Z. Shi, W. Yifan, H. Chen, C. Yang, S. Peng, Y. Shen, and G. Wetzstein (2024) GRM: large gaussian reconstruction model for efficient 3d reconstruction and generation. In Proceedings of the European Conference on Computer Vision (ECCV), Cited by: §2.1.

[^67]: J. C. Yang, A. Sax, K. J. Liang, M. Henaff, H. Tang, A. Cao, J. Chai, F. Meier, and M. Feiszli (2025) Fast3R: towards 3d reconstruction of 1000+ images in one forward pass. In CVPR, Cited by: §2.1.

[^68]: W. Yao, Z. Li, S. Lan, Z. Wang, X. Sun, J. M. Alvarez, and Z. Wu (2025) DriveSuprim: towards precise trajectory selection for end-to-end planning. External Links: 2506.06659, [Link](https://arxiv.org/abs/2506.06659) Cited by: §2, Table 1, Table 2.

[^69]: B. Ye, S. Liu, H. Xu, X. Li, M. Pollefeys, M. Yang, and S. Peng (2024) No pose, no problem: surprisingly simple 3d gaussian splats from sparse unposed images. arXiv preprint arXiv:2410.24207. Cited by: §2.1.

[^70]: S. Yuan, Y. Yang, X. Yang, X. Zhang, Z. Zhao, L. Zhang, and Z. Zhang (2026) InfiniteVGGT: visual geometry grounded transformer for endless streams. arXiv preprint arXiv:2601.02281. Cited by: §2.1.

[^71]: J. Zhang, C. Herrmann, J. Hur, V. Jampani, T. Darrell, F. Cole, D. Sun, and M. Yang (2024) Monst3r: a simple approach for estimating geometry in the presence of motion. arXiv preprint arXiv:2410.03825. Cited by: §2.1.

[^72]: K. Zhang, S. A. Bi, H. Tan, Y. Xiangli, N. Zhao, K. Sunkavalli, and Z. Xu (2024) GS-lrm: large reconstruction model for 3d gaussian splatting. arXiv preprint arXiv:2404.19702. Cited by: §2.1.

[^73]: K. Zhang, Z. Tang, X. Hu, X. Pan, X. Guo, Y. Liu, J. Huang, L. Yuan, Q. Zhang, X. Long, X. Cao, and W. Yin (2025) Epona: autoregressive diffusion world model for autonomous driving. In ICCV, Cited by: §1, §2, Table 1.

[^74]: S. Zhang, J. Wang, Y. Xu, N. Xue, C. Rupprecht, X. Zhou, Y. Shen, and G. Wetzstein (2025) FLARE: feed-forward geometry, appearance and camera estimation from uncalibrated sparse views. In CVPR, Cited by: §2.1.

[^75]: S. Zhang, Y. Ge, J. Tian, G. Xu, H. Chen, C. Lv, and C. Shen (2025) POMATO: marrying pointmap matching with temporal motion for dynamic 3d reconstruction. External Links: 2504.05692, [Link](https://arxiv.org/abs/2504.05692) Cited by: §2.1.

[^76]: S. Zhang, W. Huang, Z. Chen, C. J. Collister, Q. Huang, and C. Lv (2025) OpenREAD: reinforced open-ended reasoning for end-to-end autonomous driving with llm-as-critic. External Links: 2512.01830, [Link](https://arxiv.org/abs/2512.01830) Cited by: §1.

[^77]: S. Zhang, W. Huang, Z. Gao, H. Chen, and C. Lv (2024) Wisead: knowledge augmented end-to-end autonomous driving with vision-language model. arXiv preprint arXiv:2412.09951. Cited by: §1, §2.

[^78]: Z. Zhou, T. Cai, S. Z. Zhao, Y. Zhang, Z. Huang, B. Zhou, and J. Ma (2025) AutoVLA: a vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. arXiv preprint arXiv:2506.13757. Cited by: §1, §2.

[^79]: Z. Zhu, S. Peng, V. Larsson, Z. Cui, M. R. Oswald, A. Geiger, and M. Pollefeys (2022) NICE-slam: neural implicit scalable encoding for slam. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 12786–12796. Cited by: §2.1.

[^80]: D. Zhuo, W. Zheng, J. Guo, Y. Wu, J. Zhou, and J. Lu (2025) Streaming 4d visual geometry transformer. arXiv preprint arXiv:2507.11539. Cited by: §1, §2.1, §3.1.

[^81]: S. Zuo, Z. Xie, W. Zheng, S. Xu, F. Li, S. Jiang, L. Chen, Z. Yang, and J. Lu (2026) Dvgt: driving visual geometry transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14658–14668. Cited by: §2.1.

[^82]: S. Zuo, Z. Xie, W. Zheng, S. Xu, F. Li, H. Li, L. Chen, Z. Yang, and J. Lu (2026) DVGT-2: vision-geometry-action model for autonomous driving at scale. External Links: 2604.00813, [Link](https://arxiv.org/abs/2604.00813) Cited by: §A.3.1, §1, §2.1, Table 1, Table 2.
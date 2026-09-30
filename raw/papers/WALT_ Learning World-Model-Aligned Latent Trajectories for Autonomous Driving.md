---
title: "WALT: Learning World-Model-Aligned Latent Trajectories for Autonomous Driving"
source: "https://arxiv.org/html/2609.30436v1"
author:
published:
created: 2026-09-30
description:
tags:
  - "clippings"
---
Mingkai Jia Affiliation: The Hong Kong University of Science and Technology. Affiliation: Horizon Robotics.    Jiaxin Guo Affiliation: The Chinese University of Hong Kong.    Zhijian Shu Affiliation: Horizon Robotics. Affiliation: Nanjing University of Posts and Telecommunications.    Jiawei Xu Affiliation: Horizon Robotics. Affiliation: Nankai University.    Mingxiao Li Affiliation: Horizon Robotics.    Jintao Cheng Affiliation: The Hong Kong University of Science and Technology.    Ping Tan Affiliation: The Hong Kong University of Science and Technology.    Wei Yin Affiliation: Horizon Robotics.

###### Abstract

Driving world models learn rich predictive representations of the surrounding environment from visual observations, yet accurate visual prediction does not necessarily translate into effective trajectory planning. We argue that a key bottleneck lies in the mismatch between visual world states and raw geometric trajectories, which may limit the planner’s ability to exploit action-relevant semantics encoded by the world model. To address this issue, we propose World-Model Alignment for Latent Trajectories (WALT), which learns a compact generative trajectory latent space by transferring information from a frozen pretrained driving world model without modifying the world model itself. Rather than directly generating raw waypoints, WALT maps them into compact representations through a dual-branch trajectory autoencoder and transfers semantic knowledge from the frozen visual world model into this trajectory space, encouraging the learned action representation to capture scene-level cues relevant to future motion and planning. Beyond our proposed formulation, we systematically study latent learning based on Joint-Embedding Predictive Architectures (JEPA) and feature alignment following Representation Alignment (REPA) to investigate how trajectory-only representation learning affects downstream planning. We evaluate WALT on the NAVSIM benchmarks. Relative to the raw-waypoint baseline, WALT improves PDMS from 89.4 to 89.8 on NAVSIMv1 and EPDMS from 87.3 to 87.9 on NAVSIMv2 while reducing trajectory planner FLOPs by 30.5%. These results suggest that preserving world representations while extracting action-relevant information provides an effective interface for world-model-based trajectory planning.

![[Uncaptioned image]](https://arxiv.org/html/2609.30436v1/teaser.png)

Fig. 1: Overview of WALT. Left: The baseline directly generates raw waypoints from frozen world-state tokens, leaving a representation mismatch between rich visual features and geometric trajectory coordinates. The PCA overlay visualizes the world model’s rich feature structure. Right: WALT addresses this mismatch by learning compact trajectory latents aligned with the frozen world-model features. The world–trajectory correspondence map visualizes cosine similarity between the aligned trajectory and visual features, with red indicating higher similarity. This aligned latent interface improves planning scores on NAVSIMv1 and NAVSIMv2 while reducing FLOPs.

## I Introduction

Driving world models (DWMs) learn predictive representations from past observations and provide forward-looking information for autonomous driving [^1] [^2] [^3] [^4]. By forecasting future observations or latent states, they encode visual content and environmental dynamics for future prediction and motion planning. Recent DWMs further strengthen these predictive states with supervision beyond visual prediction, including geometric targets, semantic features, and perception-derived signals [^4] [^3] [^5] [^6] [^7] [^8] [^9].

However, high-fidelity visual prediction or a visually rich latent space does not ensure reliable trajectory planning. Fig. 1 illustrates how this mismatch can affect the action-side response and how WALT addresses it. The action head still has to map a rich predictive state to an action representation, and common planners expose this interface through raw waypoints, local motion increments, or discrete motion tokens [^3] [^10] [^11] [^12] [^13] [^14] [^15] [^16] [^17] [^18] [^19]. Recent unified visual–motion models address this gap by incorporating trajectory representations into world-model training and using the coupled representations for planning [^7] [^20]. Although this strategy enables visual prediction and motion planning to interact, it couples the action interface to a redesigned multimodal world model that jointly trains fused visual and motion representations. This raises a question: Can we transfer a frozen pretrained DWM’s visual knowledge into a trajectory generation space without modifying the world model?

To answer this question, we propose World-Model Alignment for Latent Trajectories (WALT), which to our knowledge is the first framework to learn a compact generative trajectory latent space by transferring information from a frozen pretrained driving world model without modifying the world model itself. WALT encodes raw waypoints with a dual-branch trajectory autoencoder. The reconstruction branch preserves the geometric information required to recover raw waypoints, while the semantic branch transfers knowledge from the frozen visual world-model representation to capture scene-level cues relevant to future motion and planning. The resulting latent serves as the trajectory generation space and is decoded back into planner waypoints.

Beyond the proposed alignment, we design and systematically study Joint-Embedding Predictive Architectures (JEPA) latent learning [^21] and Representation Alignment (REPA) feature alignment [^22] to investigate how trajectory-only representation learning affects downstream planning. Trajectory-only supervision can capture intrinsic motion structure, but it does not explicitly relate that structure to the surrounding visual scene. The limited planning gains of these methods in our experiments support our use of world-model alignment to provide scene-derived supervision beyond trajectory geometry. Experiments on NAVSIMv1 [^23] and NAVSIMv2 [^24] show that WALT improves planning performance and reduces trajectory-generation FLOPs relative to the raw-waypoint baseline.

![[letraj_pipe.png|Refer to caption]]

Fig. 2: WALT pipeline. In Stage 1, we train the dual-branch trajectory autoencoder with a frozen world model. The semantic branch feature is aligned with world state tokens with ℒ WALT \\mathcal{L}\_{\\text{WALT}}, while the geometric branch preserves reconstruction details and jointly trained with r e c o n \\mathcal{L}\_{recon}. In Stage 2, we freeze the tokenizer and train the trajectory planner head of the frozen world model to generate latents, which are decoded into waypoints for future trajectory prediction.

Our contributions are summarized as follows:

- We propose WALT, which to our knowledge is the first framework that learns a compact generative trajectory latent space by transferring information from a frozen driving world model without modifying the world model itself.
- We construct a dual-branch trajectory autoencoder that preserves metric geometry while transferring semantic knowledge from the frozen world-model representation, and systematically study JEPA-style and REPA-style trajectory-only designs.
- We validate the effectiveness and efficiency of our approach on NAVSIMv1 [^23] and NAVSIMv2 [^24], showing a 0.4 PDMS gain, 0.6 EPDMS improvement and a 30.5% FLOPs reduction in trajectory-generation FLOPs over the waypoint baseline.

## II Related Work

### II-A Driving World Models

Driving world models learn predictive states from observation histories by forecasting future sensory observations or latent features [^25] [^26] [^1] [^2] [^4]. Video-space models use future image generation to capture appearance changes and temporal dynamics, while structured and latent models predict compact scene features that retain spatial relations without requiring pixel-level rollout [^27] [^28] [^29] [^30]. Although their prediction targets differ, these approaches progressively enrich the hidden state with information about scene layout, object motion, and temporal context. This shift from output fidelity alone toward reusable predictive representations has made the latent state an increasingly important interface for downstream driving tasks [^5] [^6] [^7]. Beyond RGB forecasting, recent work introduces geometric or semantic signals to strengthen the predictive representation. Epona [^3] combines compact visual features with autoregressive future prediction in a diffusion world model. EponaV2 [^31] extends this representation by requiring the inferred future state to support future image, metric-depth, and foundation-model semantic prediction. These complementary targets encourage a future representation that retains visual content together with explicit geometry and semantic context. Our focus is not to redesign the predictive state, but to learn an action-side trajectory space that can make better use of the information already encoded within it.

### II-B Trajectory Representation for Planning

Action interfaces in autonomous driving range from low-level controls and adjacent-frame motion increments to continuous waypoint sequences and learned motion tokens. DrivingWorld and DrivingGPT encode local ego motion such as changes in planar position and heading into autoregressive tokens [^10] [^11]. Local increments provide a temporally regular prediction target and fit naturally within next-token models. However, recovering a complete path requires accumulation over time, and discretized variants additionally introduce quantization while leaving the full-path geometry implicit. Continuous planners instead predict the future path directly as a sequence of waypoints [^3] [^12] [^32]. The common current-ego-centered representation should be distinguished from both global absolute position and adjacent-frame relative motion. It remains local and translation invariant, yet directly preserves the metric geometry and behavior of the full future path without repeated integration. WorldDrive moves toward a learned motion interface by optimizing motion and vision representations through scene generation and transferring them to planning [^20]. This design couples motion structure with the visual model, while WALT focus on an aligned representation under a frozen world model. We encode the complete current-ego-centered future path into a compact continuous latent, preserve metric recovery through an explicit decoder, and separately organize its semantic component for alignment with the predictive world state.

### II-C Representation Learning

Joint-embedding predictive architectures learn by predicting target representations rather than reconstructing observations. I-JEPA establishes this principle for images, V-JEPA extends it to video, and V-JEPA 2 demonstrates that self-supervised video representations can support understanding, prediction, and planning [^33] [^34] [^35]. LeWorldModel further combines next-embedding prediction with SIGReg to stabilize end-to-end latent world-model learning [^21]. A related line studies the representations formed inside diffusion models and how stronger external features can improve generative learning [^36] [^37] [^38]. REPA reverses the transfer direction by aligning diffusion-transformer hidden states with clean representations from a frozen self-supervised encoder [^22]. In autonomous driving, Drive-JEPA adapts video JEPA pretraining to driving data and combines the predictive visual encoder with multimodal trajectory distillation [^39]. Its primary learned representation remains visual and supports a proposal-centric planner. Auto-JEPA more directly learns an action-oriented intent space by aligning a predicted context embedding with the latent representation of a future ego trajectory [^40]. That latent acts as a retrieval key for a fixed trajectory memory at inference. Unlike this retrieval-based formulation, WALT treats the pretrained world model as a frozen semantic teacher and transfers its action-relevant knowledge into a separately learned generative trajectory space. The learned latent is generated by the planner and explicitly decoded back into raw waypoints. We also design and systematically investigate JEPA-inspired latent learning and REPA-style downstream alignment to study trajectory-only representations without world-model transfer.

## III Method

### III-A Overview

The WALT pipeline is illustrated in Fig. 2. In Stage 1, we train a dual-branch trajectory autoencoder to compress raw waypoints into a compact latent representation (Sec. III-B). The training objective includes a reconstruction loss and a world-model alignment loss that transfers information from the frozen world model into the trajectory latent space (Sec. III-C). In Stage 2, we employ the aligned trajectory latent as the generation space for the action head of the frozen world model (Sec. III-D). By generating in this aligned latent space, the planner aims to better exploit the semantic information encoded in the world model. A systematic study of trajectory-only representation learning is also conducted to evaluate the impact of visual world-model alignment on downstream planning (Sec. III-E).

### III-B Dual-Branch Trajectory Tokenizer

Trajectory representation for planning is crucial but not unified among existing works [^3] [^10] [^11] [^12]. Directly modeling raw waypoints is simple and preserves metric geometry, but it does not explicitly separate semantic information from geometric information. Therefore, we design a dual-branch trajectory autoencoder that compresses raw waypoints into a compact latent representation with separate semantic and reconstruction components. Following previous work [^31], we consider a sequence of future trajectory points $A=\{A_{i}\}_{i=N+1}^{N+P}$, where $N$ denotes the final observed time index and $P$ is the prediction horizon, and compress it into $K$ latent tokens with a downsampling ratio $q=P/K$. The two branches share an encoder and use separate refiners to produce semantic features $z_{sem}$ and reconstruction features $z_{rec}$. These features are concatenated along the channel dimension as $z_{A}=[z_{sem};z_{rec}]$, where $[\cdot;\cdot]$ denotes concatenation. A decoder then maps the concatenated latent $z_{A}$ back to the original trajectory space as $\hat{A}=\mathcal{D}(z_{A})$. The training objective includes an $\ell_{1}$ reconstruction loss $\mathcal{L}_{\text{rec}}=\mathbb{E}_{A}[\lVert\hat{A}-A\rVert_{1}]$ and a world-model alignment loss that transfers information from the frozen world model into the trajectory latent space:

$$
\mathcal{L}_{\text{tokenizer}}=\lambda_{rec}\mathcal{L}_{\text{rec}}+\lambda_{align}\mathcal{L}_{\text{WALT}}.
$$

The world model remains frozen throughout tokenizer training. The dual-branch design encourages the semantic component to capture action-relevant information while the reconstruction component preserves the geometric structure of the trajectory. To encourage $z_{rec}$ to preserve sufficient information for trajectory reconstruction, we mask the entire semantic component $z_{sem}$ with probability $p_{sem}$ while retaining $z_{rec}$. This encourages the reconstruction branch to retain sufficient geometric information for accurate trajectory recovery even when semantic information is unavailable. The resulting latent serves as the trajectory generation space and is decoded back into planner waypoints.

TABLE I: Reported results on NAVSIMv1 [^23]. All metrics are on a 0–100 scale. Higher is better. Bold and underlined values indicate the best and second-best distinct scores among learned methods, respectively. These results include different model and training configurations.

| Method | Venue | NC | DAC | EP | TTC | C | PDMS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Human | – | 100 | 100 | 87.5 | 100 | 99.9 | 94.8 |
| LAW [^27] | ICLR’25 | 96.4 | 95.4 | 81.7 | 88.7 | 99.9 | 84.6 |
| DrivingGPT [^11] | ICCV’25 | 98.9 | 90.7 | 79.7 | 94.9 | 95.6 | 82.4 |
| World4Drive [^4] | ICCV’25 | 97.4 | 94.3 | 79.9 | 92.8 | 100 | 85.1 |
| Epona [^3] | ICCV’25 | 97.9 | 95.1 | 80.4 | 93.8 | 99.9 | 86.2 |
| PWM [^6] | NeurIPS’25 | 98.6 | 95.9 | 81.8 | 95.4 | 100 | 88.1 |
| TISA [^41] (w/o MOPT) | ICRA’26 | 98.0 | 95.5 | 81.1 | 93.8 | – | 86.8 |
| AdaThinkDrive [^42] (w/o RL) | ICRA’26 | 98.9 | 95.3 | 80.6 | 96.0 | 100 | 87.5 |
| Mimir [^43] | RA-L’26 | 98.2 | 97.5 | 83.6 | 94.6 | 100 | 89.3 |
| PRIX [^44] | RA-L’26 | 98.1 | 96.3 | 82.3 | 94.1 | 100 | 87.8 |
| ARTEMIS [^45] | RA-L’26 | 98.3 | 95.1 | 81.4 | 94.3 | 100 | 87.0 |
| DriveVLA-W0 [^5] | ICLR’26 | 98.4 | 95.3 | 80.9 | 95.2 | 100 | 87.2 |
| DriveLaW [^7] | CVPR’26 | 99.0 | 97.1 | 81.3 | 96.7 | 100 | 89.1 |
| EponaV2 [^31] (w/o RL) | arXiv’26 | 98.6 | 97.3 | 83.6 | 95.3 | 99.9 | 89.4 |
| WALT (Ours) | – | 99.1 | 97.0 | 83.6 | 96.3 | 100 | 89.8 |

### III-C World-Model Alignment for Latent Trajectories

To exploit the rich predictive representation learned by the frozen world model, we aim to transfer its action-relevant semantics into the trajectory latent space. To this end, WALT aligns the semantic branch with the world model’s visual features without modifying the world model itself. We use the frozen world-model feature as a contextual teacher, keeps the visual encoder and backbone remain frozen, and the task-specific prediction heads are not used during tokenizer training. For each scene frame and its associated trajectory, the teacher produces hidden states $H_{w}\in\mathbb{R}^{N_{b}\times M\times d_{w}}$, where $N_{b}$ denotes the flattened batch and frame dimension, $M$ is the number of teacher tokens, and $d_{w}$ is their channel dimension. A learnable projector maps the student semantic tokens to the teacher channel dimension, yielding $U=g(z_{sem})$. We formulate a CLIP-style contrastive objective. For student sample $i$ and teacher sample $j$, the sample-level score averages all $KM$ token-pair cosine similarities:

$$
S_{ij}=\frac{1}{KM}\sum_{k=1}^{K}\sum_{\ell=1}^{M}\frac{U_{ik}^{\mathsf{T}}H_{w,j\ell}}{\lVert U_{ik}\rVert_{2}\lVert H_{w,j\ell}\rVert_{2}}.
$$

Positive pairs share the same sample and frame index, while every other gathered sample or frame is treated as a negative. Student features are gathered across GPUs with gradients and frozen teacher features are gathered without gradients, so the candidate set spans the global batch. Let $y_{i}=i$ denote the matching teacher index for student sample $i$. Following the symmetric CLIP objective [^46], we use

$$
\mathcal{L}_{\text{WALT}}=\frac{1}{2}\operatorname{CE}(\alpha S,y)+\frac{1}{2}\operatorname{CE}(\alpha S^{\mathsf{T}},y).
$$

The learnable logit scale $\alpha=\exp(s)$ controls the concentration of the contrastive distribution and is bounded during optimization. Through this objective, WALT transfers world-model information into the action representation during tokenizer training without concatenating scene features into the trajectory latent.

### III-D Latent Trajectory Generation

After tokenizer training, we freeze the trajectory tokenizer and retain the original frozen world-model pathway. We use a rectified-flow planner as the trajectory head following previous works [^47] [^48] [^49] [^50] [^51] [^31]. The head is trained with a conditional flow-matching objective, but its target is changed from $P$ raw waypoint tokens to $K$ concatenated latent tokens $z_{A}$. That is, the flow-matching trajectory head predicts the latent trajectory velocity $\hat{v}_{A}$ in the learned compact representation space rather than the velocity of raw waypoints.

Given $\epsilon\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ and $t\sim\mathcal{U}[0,1]$, we construct

$$
z_{t}=(1-t)\epsilon+tz_{A},\qquad v_{target}=z_{A}-\epsilon.
$$

Conditioned on the frozen world-model state $H_{w}$, the trajectory head minimizes

$$
\mathcal{L}_{\text{flow}}=\mathbb{E}_{A,\epsilon,t}\left[\left\lVert\hat{v}_{A}(z_{t},t;H_{w})-(z_{A}-\epsilon)\right\rVert_{2}^{2}\right].
$$

At inference, we initialize $\hat{z}_{0}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ and integrate the predicted velocity from $t=0$ to $t=1$:

$$
\hat{z}_{t+\Delta t}=\hat{z}_{t}+\Delta t\,\hat{v}_{A}(\hat{z}_{t},t;H_{w}).
$$

The decoder then maps the generated latent to the predicted waypoints as $\hat{A}=\mathcal{D}(\hat{z}_{1})$. With the learned trajectory latent space, the flow-matching planner generates compact trajectory representations aligned with the world model, supporting effective planning with reduced trajectory head computation.

TABLE II: Comparison on NAVSIMv2 [^24]. All metrics are on a 0–100 scale, where higher is better. <sup>∗</sup> denotes results reported using the original evaluator before the human-penalty aggregation update. Bold and underlined values indicate the best and second-best distinct scores among unstarred learned methods.

| Method | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | EPDMS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Human | 100 | 100 | 99.8 | 100 | 87.4 | 100 | 100 | 98.1 | 90.1 | 94.5 |
| DriveVLA-W0 [^5] | 98.4 | 95.2 | 99.4 | 99.9 | 86.6 | 97.9 | 97.8 | 98.3 | 82.7 | 86.9 |
| ARTEMIS <sup>∗</sup> [^45] | 98.3 | 95.1 | 98.6 | 99.8 | 81.5 | 97.4 | 96.5 | 98.3 | – | 83.1 |
| PRIX <sup>∗</sup> [^44] | 98.0 | 85.6 | 99.5 | 99.8 | 87.4 | 97.2 | 97.1 | 98.3 | 87.6 | 84.2 |
| DriveWorld-VLA [^52] | 98.6 | 99.1 | 99.6 | 99.8 | 87.4 | 97.9 | 97.0 | 97.8 | 78.6 | 86.8 |
| EponaV2 [^31] (w/o RL) | 98.4 | 96.7 | 99.6 | 99.9 | 87.6 | 98.1 | 98.0 | 98.1 | 68.3 | 87.3 |
| WALT (Ours) | 98.5 | 96.8 | 99.6 | 99.9 | 87.4 | 98.2 | 97.9 | 98.3 | 73.4 | 87.9 |

### III-E Trajectory-Only Representation Analysis

To assess the contribution of visual world-model alignment, we further investigate two trajectory-only representation learning methods, JEPA-Traj and REPA-Traj, under the same planner setting. JEPA-Traj uses latent trajectory generation, whereas REPA-Traj retains raw-waypoint generation and adds auxiliary feature alignment. Joint-embedding predictive architectures (JEPA) learn by predicting target representations and are commonly used in self-supervised visual representation learning [^33] [^34] [^35]. To apply JEPA to the trajectory modality, we treat continuous waypoints as sequential inputs, extract overlapping sub-trajectories, and encode them with the same semantic tokenizer. Let $z_{sem}^{(1)}$ and $z_{sem}^{(2)}$ be the semantic tokens of two overlapping sub-trajectories, where $z_{sem}^{(1)}$ corresponds to the earlier sub-trajectory and $z_{sem}^{(2)}$ to the later one. Conditioned on the time offset at which the overlap begins, a predictor is trained to predict the later sub-trajectory’s semantic tokens from the earlier ones. To prevent representation collapse, we use SIGReg [^21] regularization. The representation-learning objective is

$$
\mathcal{L}_{\text{JEPA}}=\lambda_{\text{sigreg}}\mathcal{L}_{\text{sigreg}}+\lambda_{\text{pred}}\mathcal{L}_{\text{pred}},
$$

where $\mathcal{L}_{\text{pred}}$ is the $\ell_{2}$ prediction loss between the predicted and target semantic tokens, $\mathcal{L}_{\text{sigreg}}$ is the SIGReg regularization loss, $\lambda_{\text{sigreg}}$ and $\lambda_{\text{pred}}$ are loss weights. This objective replaces the world-model alignment term during tokenizer training, while the reconstruction loss is retained. Visualizations of within-class and between-class similarity distributions, class-pair heatmaps, and t-SNE embeddings in Fig. 4 show that JEPA-style representation learning helps separate different driving behavior classes.

Beyond using JEPA-style trajectory-only representation learning for latent generation, we also investigate REPA-style [^22] feature alignment, which uses the learned trajectory representation in a different way. REPA [^22] uses visual foundation-model representations as teacher targets for intermediate diffusion-transformer features. We apply this idea to trajectory-only representation learning, using the learned trajectory representation as a teacher target for the trajectory stream of the flow-matching planner. Let $h_{traj}$ be the trajectory-stream representation before the joint single-stream blocks, and let $z_{sem}^{\ast}=\operatorname{sg}(z_{sem})$ be the frozen semantic target from the JEPA-style trained trajectory autoencoder, where $\operatorname{sg}$ denotes stop-gradient. A projector $g_{R}$ maps trajectory sequence channels to the semantic target dimension. An auxiliary cosine-distance loss between $g_{R}(h_{traj})$ and $z_{sem}^{\ast}$ is added to the flow-matching objective. This REPA-style feature alignment keeps raw waypoints as the flow-matching target, allowing us to examine whether the learned trajectory representation helps the downstream trajectory stream generate better waypoints. The results in Table III show only marginal PDMS gains for JEPA-Traj and REPA-Traj over the raw-waypoint baseline. These results motivate incorporating visual world-model supervision into trajectory representation learning.

![[letraj_qual.png|Refer to caption]]

Fig. 3: Qualitative trajectories and world-model alignment. Columns show cases for braking, straight driving, turn, and curve. From top to bottom: front-view images, world–trajectory correspondence maps, and predicted trajectories (orange) with ground truth (black). The maps visualize cosine similarity between projected semantic tokens from planner-generated latents and frozen visual world-model tokens, with warmer colors indicating higher similarity.

## IV Experiments

### IV-A Experimental Setup

Benchmarks and metrics. We evaluate on NAVSIMv1 navtest using the official planning protocol [^23]. We report PDMS together with no-at-fault collision (NC), drivable-area compliance (DAC), ego progress (EP), time-to-collision (TTC), and comfort (C). We additionally evaluate on NAVSIMv2 [^24], reporting EPDMS and its associated component metrics, including traffic light compliance (TLC) and lane keeping (LK), while expanding comfort (C) into history comfort (HC) and extended comfort (EC). For all metrics reported, higher values indicating better performance. Evaluator differences about human filtering for reported NAVSIMv2 results are indicated in Table II.

Implementation details. We instantiate WALT with EponaV2 [^31] as the pretrained driving world model, retaining its default model and planner-training settings except for the trajectory representation and associated alignment modules. We use the official benchmark data splits and evaluation protocols. Each trajectory contains eight ego-centric $(x,y,\mathrm{yaw})$ waypoints sampled at 0.5-s intervals. In Stage 1, the dual-branch tokenizer compresses the trajectory into two 32-dimensional tokens, each comprising 24 semantic and 8 reconstruction channels. We set the semantic masking probability to 0.5 and the alignment weight $\lambda_{align}$ to 0.1, and train the tokenizer for 100 epochs on 32 H20 GPUs with a fixed learning rate of $1\times 10^{-4}$, keeping the world-model backbone frozen. In Stage 2, we further freeze the trajectory autoencoder and fine-tune only the trajectory head.

Representation variants. We compare five variants using the same frozen world-model backbone and evaluation protocol. JEPA-Traj adds trajectory-only latent prediction with SIGReg to tokenizer training, whereas REPA-Traj retains raw-waypoint generation and aligns intermediate planner features with the frozen JEPA-Traj semantic representation. Thus, these variants share the world-model conditioning source but differ in their generation targets or auxiliary supervision. For JEPA-Traj, SIGReg uses a weight of $5\times 10^{-4}$ with other settings same to LeWM [^21]. For REPA-Traj, cosine alignment is applied before the joint blocks in trajectory planning head with a weight of $0.1$. WALT instead learns the trajectory latent through world-model alignment.

### IV-B Planning Performance

Benchmark comparison. To assess the planning benefits of our aligned trajectory representation without additional post-training, WALT uses no reinforcement learning or similar refinement procedures, and we compare against results reported without such procedures where available, as indicated in the tables. Table I compares WALT with reported NAVSIMv1 results. WALT achieves the best overall PDMS of 89.8, while also attaining the highest NC score and matching the best EP and comfort scores. These results demonstrate a strong balance across progress, safety, and comfort. Notably, these gains are obtained by learning a compact world-model-aligned trajectory space while keeping the pretrained world model unchanged, suggesting that this aligned latent can better exploit existing world knowledge for planning.

A consistent trend is observed on NAVSIMv2. As shown in Table II, WALT achieves strongest EPDMS among all the methods. Compared to EponaV2 without RL, our scores improve in NC, DAC, TTC, HC, with the largest component gain in EC, which increases from 68.3 to 73.4. Meanwhile, DDC and TLC remain unchanged and match the best reported scores, while EP and LK decrease slightly by 0.2 and 0.1 points, respectively. Overall, the gains span several safety and comfort metrics while largely preserving progress and lane keeping, supporting the effectiveness of world-model-aligned trajectory representations across both benchmarks.

Effect of trajectory representation. Table III isolates the effect of the action-side representation under the same frozen world-model backbone. The reconstruction-only autoencoder without semantic alignment changes PDMS from 89.42 to 89.48, indicating that the reconstruction-only encoded latent preserves information relevant to downstream planning. JEPA-Traj and REPA-Traj provide only marginal gains, suggesting that trajectory-only representation learning is insufficient to substantially improve the planner.

In contrast, WALT reaches the best overall PDMS of 89.83, improving over both raw waypoints and the reconstruction-only autoencoder. Relative to raw waypoints, NC increases from 98.62 to 99.11 and TTC from 95.26 to 96.34, while maintaining competitive performance on the remaining metrics. Overall, the comparison highlights the importance of injecting world-aware semantics into the action representation, rather than learning the trajectory space purely from trajectory reconstruction or self-supervision.

### IV-C Trajectory Representation Analysis

Fig. 4 compares raw waypoints, JEPA-Traj semantic latents, and WALT semantic latents on identical NAVSIMv1 test samples. We use six behavior classes: left lane change, left turn, right lane change, right turn, start, and normal straight. Each representation is flattened and $\ell_{2}$ -normalized before computing pairwise cosine similarities. We examine within-class and between-class distributions, class-pair mean similarities, and t-SNE embeddings with cosine distance. Within-class distributions exclude self-pairs.

![[letraj_wera.png|Refer to caption]]

Fig. 4: Trajectory representation analysis on NAVSIMv1. Columns compare raw waypoints, JEPA-Traj semantic latents, and WALT semantic latents on identical samples. From top to bottom: t-SNE embeddings, within-class and between-class cosine-similarity distributions, and class-pair mean-similarity matrices. Learned representations use only their semantic components.

Raw waypoints exhibit strong within-class similarity but also substantial similarity between lane changes and straight driving, reflecting shared path geometry. JEPA-Traj reduces several cross-class similarities and produces more distinct behavior groups in the t-SNE visualization. Compared with JEPA-Traj, WALT has higher within-class mean similarity for all six classes, but also retains higher similarity between several different classes. Thus, world-model alignment does not simply maximize separation between behavior labels. The planning results complement this observation: stronger separation under trajectory-only learning does not necessarily yield a larger PDMS gain. These visualizations characterize representation organization rather than establish which contextual information causes the planning improvement.

### IV-D Qualitative Results and Alignment Visualization

Fig. 3 shows front-view observations, world-trajectory correspondence maps, and predicted trajectories for braking, straight driving, turning, and curve following. The predicted trajectories, decoded from planner-generated latents, closely follow the ground-truth paths across these examples. To visualize alignment, we compute cosine similarity between the projected semantic trajectory tokens and the front-view world-model visual tokens. The resulting scores are averaged over trajectory tokens and reshaped to the visual-token grid.

The correspondence maps reveal behavior-dependent spatial responses. During braking, high similarity concentrates on the leading vehicle and nearby roadway, while straight driving emphasizes the drivable road and lane direction. In turning and curved-road cases, responses shift toward road geometry and regions along the intended path. Some activation also appears on contextual background regions, indicating that these maps reflect representation-level correspondence rather than precise localization. Overall, these examples suggest that world-model alignment encourages trajectory latents to capture visual scene semantics relevant to each driving behavior.

### IV-E Computational Efficiency

Under the NAVSIMv1 evaluation, the action head processes two latent trajectory tokens instead of eight raw waypoint tokens. Table IV reports one denoising step 297.47 GFLOPs for raw-waypoint and 206.88 GFLOPs for latent with decoding included, a 30.5% reduction. The decoder contributes only a small fraction of the cost, and the frozen world-model computation is excluded. For an $N$ -step rollout, the total trajectory-generation cost is $NF_{step}+F_{dec}$, where $F_{step}$ is the denoising-step cost and $F_{dec}$ is the one-time decoder cost. Since $F_{dec}$ is small, the full-rollout cost is approximately $N$ times the tabulated value.

TABLE III: Action-side representation comparison on NAVSIMv1 with the same frozen world-model backbone. Variants differ in generation target and representation supervision. All metrics are on a 0–100 scale. Higher is better.

| Method | NC | DAC | EP | TTC | C | PDMS |
| --- | --- | --- | --- | --- | --- | --- |
| Raw waypoints baseline | 98.62 | 97.32 | 83.60 | 95.26 | 99.93 | 89.42 |
| $+$ trajectory AE (w/o $\mathcal{L}_{\text{WALT}}$) | 98.62 | 97.38 | 83.63 | 95.32 | 99.95 | 89.48 |
| JEPA-Traj | 98.55 | 97.37 | 82.85 | 95.91 | 100.00 | 89.46 |
| REPA-Traj | 98.48 | 97.46 | 83.79 | 95.11 | 99.98 | 89.49 |
| WALT | 99.11 | 96.99 | 83.63 | 96.34 | 100.00 | 89.83 |

TABLE IV: Trajectory-generation cost for one denoising step under matched world-model conditioning. The latent-generation cost includes trajectory decoding, and the frozen world-model computation is excluded.

| Representation | Compression Ratio | GFLOPs $\downarrow$ |
| --- | --- | --- |
| Raw waypoint generation | – | 297.47 |
| WALT latent generation | 4 $\times$ | 206.88 |

## V Conclusion

We presented WALT, which learns compact trajectory latents through alignment with representations from a frozen pretrained driving world model. Its dual-branch tokenizer combines metric reconstruction with world-model supervision, providing a latent space for trajectory generation without modifying the world-model backbone. Experiments on NAVSIMv1 and NAVSIMv2 show improved planning scores and reduction of FLOPs. Comparisons with reconstruction-only and trajectory-only alternatives support the value of world-model alignment in the evaluated setting. These findings motivate further study of learning world-model-aligned trajectory representations for autonomous driving.

[^1]: Y. Wang, J. He, L. Fan, H. Li, Y. Chen, and Z. Zhang (2024) Driving into the future: multiview visual forecasting and planning with world model for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14749–14759. Cited by: §I, §II-A.

[^2]: C. Min, D. Zhao, L. Xiao, J. Zhao, X. Xu, Z. Zhu, L. Jin, J. Li, Y. Guo, J. Xing, et al. (2024) Driveworld: 4d pre-trained scene understanding via world models for autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 15522–15533. Cited by: §I, §II-A.

[^3]: K. Zhang, Z. Tang, X. Hu, X. Pan, X. Guo, Y. Liu, J. Huang, L. Yuan, Q. Zhang, X. Long, et al. (2025) Epona: autoregressive diffusion world model for autonomous driving. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 27220–27230. Cited by: §I, §I, §II-A, §II-B, §III-B, TABLE I.

[^4]: Y. Zheng, P. Yang, Z. Xing, Q. Zhang, Y. Zheng, Y. Gao, P. Li, T. Zhang, Z. Xia, P. Jia, X. Lang, and D. Zhao (2025) World4Drive: End-to-end autonomous driving via intention-aware physical latent world model. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 28632–28642. Cited by: §I, §II-A, TABLE I.

[^5]: Y. Li, S. Shang, W. Liu, B. Zhan, H. Wang, Y. Wang, Y. Chen, X. Wang, Y. An, C. Tang, et al. (2025) DriveVLA-W0: World models amplify data scaling law in autonomous driving. arXiv preprint arXiv:2510.12796. Cited by: §I, §II-A, TABLE I, TABLE II.

[^6]: Z. Zhao, T. Fu, Y. Wang, L. Wang, and H. Lu (2025) From Forecasting to Planning: Policy world model for collaborative state-action prediction. In Advances in Neural Information Processing Systems, Cited by: §I, §II-A, TABLE I.

[^7]: T. Xia, Y. Li, L. Zhou, J. Yao, K. Xiong, H. Sun, B. Wang, K. Ma, G. Chen, H. Ye, et al. (2026) Drivelaw: unifying planning and video generation in a latent driving world. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 39701–39712. Cited by: §I, §I, §II-A, TABLE I.

[^8]: J. Xu, K. Deng, Z. Fan, S. Wang, J. Xie, and J. Yang (2025) Ad-gs: object-aware b-spline gaussian splatting for self-supervised autonomous driving. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 24770–24779. Cited by: §I.

[^9]: J. Cheng, K. Zeng, Z. Huang, X. Tang, J. Wu, C. Zhang, X. Chen, and R. Fan (2024) Mf-mos: a motion-focused model for moving object segmentation. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pp. 12499–12505. Cited by: §I.

[^10]: X. Hu, M. Jia, X. Guo, Q. Zhang, X. Long, and W. Yin (2026) Drivingworld: constructing world model for autonomous driving via video gpt. In International Conference on Pattern Recognition, pp. 276–291. Cited by: §I, §II-B, §III-B.

[^11]: Y. Chen, Y. Wang, and Z. Zhang (2025) DrivingGPT: Unifying driving world modeling and planning with multi-modal autoregressive transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 26890–26900. Cited by: §I, §II-B, §III-B, TABLE I.

[^12]: B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang, et al. (2025) DiffusionDrive: Truncated diffusion model for end-to-end autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 12037–12047. Cited by: §I, §II-B, §III-B.

[^13]: Z. Shu, C. Lin, T. Xie, W. Yin, B. Li, Z. Pu, W. Li, Y. Yao, X. Cao, X. Guo, et al. (2026) Litevggt: boosting vanilla vggt via geometry-aware cached token merging. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 36422–36432. Cited by: §I.

[^14]: S. Chen, X. Huang, Z. Zhong, J. Guan, and S. Zhou (2025) A focused human body model for accurate anthropometric measurements extraction. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 22658–22667. Cited by: §I.

[^15]: J. Guo, T. Guan, W. Dong, W. Zheng, W. Wang, Y. Wang, Y. Yam, and Y. Liu (2026) SaLon3R: structure-aware long-term feedforward 3d reconstruction from unposed images. International Journal of Computer Vision 134 (8), pp. 383. Cited by: §I.

[^16]: J. Guo, W. Dong, T. Huang, H. Ding, Z. Wang, H. Kuang, Q. Dou, and Y. Liu (2025) Endo3r: unified online reconstruction from dynamic monocular endoscopic video. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pp. 170–180. Cited by: §I.

[^17]: J. Guo, F. Zhong, R. Xiong, Y. Liu, Y. Wang, and Y. Liao (2022) A visual navigation perspective for category-level object pose estimation. In European Conference on Computer Vision, pp. 123–141. Cited by: §I.

[^18]: M. Jia, W. Yin, X. Hu, J. Guo, X. Guo, Q. Zhang, X. Long, and P. Tan (2025) Mgvq: could vq-vae beat vae? a generalizable tokenizer with multi-group quantization. arXiv preprint arXiv:2507.07997. Cited by: §I.

[^19]: J. Zhou, C. Ma, Z. Zhong, M. Liu, Z. Zhou, Y. ji, B. Su, B. Cai, and X. Huang (2026) Think locally, refine globally for memory-efficient 3d reconstruction. arXiv preprint arXiv:2609.21437. Cited by: §I.

[^20]: X. Gui, M. Zhang, T. Yan, W. Han, J. Gong, F. Tan, C. Xu, and J. Shen (2026) Bridging scene generation and planning: driving with world model via unifying vision and motion representation. arXiv preprint arXiv:2603.14948. Cited by: §I, §II-B.

[^21]: L. Maes, Q. L. Lidec, D. Scieur, Y. LeCun, and R. Balestriero (2026) Leworldmodel: stable end-to-end joint-embedding predictive architecture from pixels. arXiv preprint arXiv:2603.19312. Cited by: §I, §II-C, §III-E, §IV-A.

[^22]: S. Yu, S. Kwak, H. Jang, J. Jeong, J. Huang, J. Shin, and S. Xie (2024) Representation alignment for generation: training diffusion transformers is easier than you think. arXiv preprint arXiv:2410.06940. Cited by: §I, §II-C, §III-E.

[^23]: W. Cao, M. Hallgarten, T. Li, D. Dauner, X. Gu, C. Wang, Y. Miron, M. Aiello, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, A. Geiger, and K. Chitta (2025) Pseudo-simulation for autonomous driving. In Conference on Robot Learning (CoRL), Cited by: 3rd item, §I, TABLE I, TABLE I, §IV-A.

[^24]: D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, A. Geiger, and K. Chitta (2024) NAVSIM: Data-driven non-reactive autonomous vehicle simulation and benchmarking. In Advances in Neural Information Processing Systems (NeurIPS), Cited by: 3rd item, §I, TABLE II, TABLE II, §IV-A.

[^25]: S. Tu, X. Zhou, D. Liang, X. Jiang, Y. Zhang, X. Li, and X. Bai (2025) The role of world models in shaping autonomous driving: a comprehensive survey. arXiv preprint arXiv:2502.10498. Cited by: §II-A.

[^26]: L. Kong, Y. Yang, J. Mei, Y. Liu, A. Liang, D. Zhu, D. Lu, W. Yin, X. Hu, M. Jia, et al. (2025) 3d and 4d world modeling: a survey. arXiv preprint arXiv:2509.07996. Cited by: §II-A.

[^27]: Y. Li, L. Fan, J. He, Y. Wang, Y. Chen, Z. Zhang, and T. Tan (2024) Enhancing end-to-end autonomous driving with latent world model. arXiv preprint arXiv:2406.08481. Cited by: §II-A, TABLE I.

[^28]: J. Zhang, Z. Fu, zelinxu, wenying.dai, Q. Liu, and Y. Wang (2026) ResWorld: Temporal residual world model for end-to-end autonomous driving. In The Fourteenth International Conference on Learning Representations, Cited by: §II-A.

[^29]: J. Li, B. Zhang, X. Jin, J. Deng, X. Zhu, and L. Zhang (2025) ImagiDrive: a unified imagination-and-planning framework for autonomous driving. arXiv preprint arXiv:2508.11428. Cited by: §II-A.

[^30]: P. Yang, B. Lu, Z. Xia, C. Han, Y. Gao, T. Zhang, K. Zhan, X. Lang, Y. Zheng, and Q. Zhang (2025) WorldRFT: latent world model planning with reinforcement fine-tuning for autonomous driving. arXiv preprint arXiv:2512.19133. Cited by: §II-A.

[^31]: J. Xu, Z. Zhong, Z. Shu, M. Jia, M. Li, J. Bian, Q. Zhang, K. Zhang, J. Xie, et al. (2026) EponaV2: driving world model with comprehensive future reasoning. arXiv preprint arXiv:2605.14696. Cited by: §II-A, §III-B, §III-D, TABLE I, TABLE II, §IV-A.

[^32]: Z. Xing, X. Zhang, Y. Hu, B. Jiang, T. He, Q. Zhang, X. Long, and W. Yin (2025) GoalFlow: Goal-driven flow matching for multimodal trajectories generation in end-to-end autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 1602–1611. Cited by: §II-B.

[^33]: M. Assran, Q. Duval, I. Misra, P. Bojanowski, P. Vincent, M. Rabbat, Y. LeCun, and N. Ballas (2023) Self-supervised learning from images with a joint-embedding predictive architecture. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 15619–15629. Cited by: §II-C, §III-E.

[^34]: A. Bardes, Q. Garrido, J. Ponce, X. Chen, M. Rabbat, Y. LeCun, M. Assran, and N. Ballas (2024) Revisiting feature prediction for learning visual representations from video. arXiv preprint arXiv:2404.08471. Cited by: §II-C, §III-E.

[^35]: M. Assran, A. Bardes, D. Fan, Q. Garrido, R. Howes, M. Muckley, A. Rizvi, C. Roberts, K. Sinha, A. Zholus, et al. (2025) V-jepa 2: self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985. Cited by: §II-C, §III-E.

[^36]: Y. Mi, Z. Zhong, Y. Huang, Q. Yuan, X. Zhao, J. Xu, S. Ding, S. Wang, R. Guo, and S. Zhou (2025) Data synthesis with diverse styles for face recognition via 3dmm-guided diffusion. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 21203–21214. Cited by: §II-C.

[^37]: Z. Zhong, Y. Ji, Z. Kong, Y. Liu, J. Wang, J. Feng, L. Liu, X. Wang, Y. Li, Y. She, et al. (2025) Anytalker: scaling multi-person talking video generation with interactivity refinement. arXiv preprint arXiv:2511.23475. Cited by: §II-C.

[^38]: M. Jia, M. Li, Z. Shu, A. Zheng, L. Fan, J. Guo, T. Shi, D. Lu, Z. Li, X. Guo, X. Qi, X. Long, Q. Zhang, P. Tan, and W. Yin (2026) DINO-Tok: Adapting DINO for visual tokenizers. arXiv preprint arXiv:2511.20565. Cited by: §II-C.

[^39]: L. Wang, Z. Yang, C. Bai, G. Zhang, X. Liu, X. Zheng, X. Long, C. Lu, and C. Lu (2026) Drive-jepa: video jepa meets multimodal trajectory distillation for end-to-end driving. arXiv preprint arXiv:2601.22032. Cited by: §II-C.

[^40]: J. Yang, Z. Chen, C. Huang, and J. Li (2026) Auto-jepa: a latent world model of continuous intent for end-to-end autonomous driving. arXiv preprint arXiv:2607.29031. Cited by: §II-C.

[^41]: J. Zhao, T. Ban, X. Li, X. Gui, H. Zhou, L. Liu, H. Zhao, and B. Li (2025) Autoregressive end-to-end planning with time-invariant spatial alignment and multi-objective policy refinement. arXiv preprint arXiv:2509.20938. Cited by: TABLE I.

[^42]: Y. Luo, F. Li, S. Xu, Z. Lai, L. Yang, Q. Chen, Z. Luo, Z. Xie, et al. (2025) Adathinkdrive: adaptive thinking via reinforcement learning for autonomous driving. arXiv preprint arXiv:2509.13769. Cited by: TABLE I.

[^43]: Z. Xing, Y. Zheng, Q. Zhang, Z. Ding, P. Yang, S. Gu, Z. Xia, and D. Zhao (2025) Mimir: hierarchical goal-driven diffusion with uncertainty propagation for end-to-end autonomous driving. IEEE Robotics and Automation Letters 11 (2), pp. 2178–2185. Cited by: TABLE I.

[^44]: M. Wozniak, L. Liu, Y. Cai, and P. Jensfelt (2026) Prix: learning to plan from raw pixels for end-to-end autonomous driving. IEEE Robotics and Automation Letters 11 (5), pp. 6400–6407. Cited by: TABLE I, TABLE II.

[^45]: R. Feng, N. Xi, D. Chu, R. Wang, Z. Deng, A. Wang, L. Lu, J. Wang, and Y. Huang (2025) Artemis: autoregressive end-to-end trajectory planning with mixture of experts for autonomous driving. IEEE Robotics and Automation Letters 11 (1), pp. 226–233. Cited by: TABLE I, TABLE II.

[^46]: A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. (2021) Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. Cited by: §III-C.

[^47]: M. S. Albergo and E. Vanden-Eijnden (2023) Building normalizing flows with stochastic interpolants. In The Eleventh International Conference on Learning Representations, Cited by: §III-D.

[^48]: Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le (2023) Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, Cited by: §III-D.

[^49]: X. Liu, C. Gong, and qiang liu (2023) Flow straight and fast: Learning to generate and transfer data with rectified flow. In The Eleventh International Conference on Learning Representations, Cited by: §III-D.

[^50]: M. Shi, H. Wang, W. Zheng, Z. Yuan, X. Wu, X. Wang, P. Wan, J. Zhou, and J. Lu (2025) Latent diffusion model without variational autoencoder. arXiv preprint arXiv:2510.15301. Cited by: §III-D.

[^51]: M. Shi, H. Wang, B. Zhang, W. Zheng, B. Zeng, Z. Yuan, X. Wu, Y. Zhang, H. Yang, X. Wang, P. Wan, K. Gai, J. Zhou, and J. Lu (2025) SVG-T2I: Scaling up text-to-image latent diffusion model without variational autoencoder. arXiv preprint arXiv:2512.11749. Cited by: §III-D.

[^52]: L. Liu, Z. Song, C. Jia, H. Ye, X. Hao, L. Chen, et al. (2026) Driveworld-vla: unified latent-space world modeling with vision-language-action for autonomous driving. arXiv preprint arXiv:2602.06521. Cited by: TABLE II.
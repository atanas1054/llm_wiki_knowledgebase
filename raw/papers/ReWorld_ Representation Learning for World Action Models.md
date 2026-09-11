---
title: "ReWorld: Representation Learning for World Action Models"
source: "https://arxiv.org/html/2606.27504v2"
author:
published:
created: 2026-09-04
description:
tags:
  - "clippings"
---
Tianze Xia    Lijun Zhou    Kaixin Xiong    Jingfeng Yao    Zhenxin Zhu    Haiyang Sun    Bing Wang    Guang Chen    Wenyu Liu    Hangjun Ye    and Xinggang Wang    Thanks: Tianze Xia, Jingfeng Yao, Wenyu Liu, and Xinggang Wang are with Huazhong University of Science and Technology, Wuhan, China.Tianze Xia, Lijun Zhou, Kaixin Xiong, Zhenxin Zhu, Haiyang Sun, Bing Wang, Guang Chen, and Hangjun Ye are with Xiaomi EV, China.Tianze Xia and Lijun Zhou contributed equally to this work. Lijun Zhou is the project lead.Corresponding author: Xinggang Wang (E-mail: [xgwang@hust.edu.cn](mailto:xgwang@hust.edu.cn)).

###### Abstract

World Action Models (WAMs) unify future environment prediction with action generation for autonomous driving, yet existing approaches optimize only the final outputs, leaving intermediate representations as incidental byproducts. We present ReWorld, the first representation learning framework specifically designed for autonomous-driving WAMs. ReWorld explicitly optimizes the latent world-to-action pathway through three complementary mechanisms. First, it imposes future-predictive supervision on intermediate Video DiT states to encode temporal scene dynamics, enabling self-guided sampling and a roughly twofold convergence speedup. Second, it aligns Action DiT states with their attended video readouts so that the retrieved world information is retained in the representations used for planning. Third, it shapes the action space using geometrically close yet low-scoring hard negatives to separate the expert trajectory from nearby unsafe alternatives. ReWorld constructs supervision entirely from the WAM’s own generation targets and attended features, requiring no external encoders or teacher models and introducing only 0.3% additional per-step training cost. Experiments show that ReWorld reduces FVD from 81.3 to 61.9 on nuScenes, improves closed-loop PDMS from 89.1 to 90.4 on NAVSIM without reinforcement learning or test-time scoring, and increases frozen linear-probe accuracy from 68.3% to 80.2% on UCF-101 action recognition. These results indicate that explicitly optimized representations are central to translating world knowledge into planning capability in WAMs. Code is available at [https://github.com/xiaomi-research/ReWorld](https://github.com/xiaomi-research/ReWorld).

![[ReWorld_framework.drawio_compressed.png|Refer to caption]]

Fig. 1: Overview of ReWorld. A Video DiT learns a latent representation of future scene evolution, whose mid-denoising states ℱ \\mathcal{F} condition an Action DiT for trajectory generation. ReWorld explicitly optimizes this world-to-action representation pathway in three stages. Stage 1 makes intermediate video states future-predictive through ℒ Mid \\mathcal{L}\_{\\mathrm{Mid}} and enables self-guided video sampling. Stage 2 freezes the Video DiT and aligns post-cross-attention action states with their attended video readouts through align \\mathcal{L}\_{\\mathrm{align}}. Stage 3 jointly fine-tunes both branches with RDE \\mathcal{L}\_{\\mathrm{RDE}}, using geometrically close but low-scoring trajectories to shape behavior-sensitive action representations.

## I Introduction

World models have emerged as an important direction in autonomous driving by learning how driving scenes evolve over time and providing dynamic priors beyond instantaneous perception [^1] [^2] [^3] [^4]. In particular, video-generation-based world models learn scene dynamics and physical regularities directly from large-scale driving videos through future pixel prediction [^5] [^6] [^7] [^8] [^9] [^10]. However, video prediction is not the ultimate goal of a decision-oriented system. World Action Models (WAMs) go beyond visual forecasting by coupling future world prediction with action generation, allowing knowledge learned from video to support decision-making directly [^11] [^4] [^3] [^12] [^13] [^14].

A central challenge for WAMs is transferring world knowledge from video generation to planning. Simulator-based methods and auxiliary prediction objectives affect planning only indirectly, while joint generation–planning models often preserve separate output streams [^4] [^11]. DriveLaW [^15] introduces a chained formulation that conditions an Action DiT directly on intermediate Video DiT features cached at the initial reverse-flow step (Fig. 1). This latent connection turns the video generator from a visual renderer into a source of planning representations.

Yet latent access alone does not ensure effective representations. With standard output-level objectives, intermediate Video DiT states are supervised only through the final denoising output, and Action DiT states only through the final trajectory prediction. Consequently, video states may lack explicit future-predictive structure, while action states may fail to retain the world information retrieved through cross-attention. Imitation learning further provides limited supervision for distinguishing expert actions from geometrically similar yet unsafe alternatives. We term this limitation the *representation bottleneck of WAMs*: the intermediate representations connecting world prediction to action generation are not explicitly optimized to be future-predictive, cross-modally grounded, or sensitive to closed-loop behavior quality.

We present ReWorld, the first representation learning framework specifically designed for autonomous-driving WAMs. ReWorld addresses this bottleneck through a progressive curriculum that optimizes representation formation, cross-modal transfer, and decision-oriented shaping. First, auxiliary prediction heads make intermediate Video DiT states explicitly future-predictive. The resulting shallow-to-deep prediction hierarchy further enables inference-time self-guidance (Fig. 2) and a roughly twofold convergence speedup from scratch. Second, ReWorld aligns Action DiT states with the video readouts retrieved through cross-attention, grounding action representations in world information while keeping the video branch frozen. Third, it jointly fine-tunes both branches with geometrically close yet low-scoring hard negatives, making the action space sensitive to unsafe behaviors near the expert trajectory.

Representation learning methods developed for image diffusion provide useful precedents [^16] [^17], but do not transfer directly to long-horizon driving video. External image or video encoders may introduce semantic targets misaligned with the generator’s flow-matching dynamics, while teacher branches add substantial training cost. ReWorld instead derives supervision entirely from the WAM’s own generation targets, attended video readouts, and trajectory candidates, requiring no external representation encoders or teacher models and introducing only 0.3% additional per-step Video DiT training cost.

We evaluate ReWorld on nuScenes [^18] and NAVSIM [^19], and analyze frozen Video DiT representations on UCF-101. ReWorld reduces FVD from 81.3 to 61.9 on nuScenes, improves closed-loop PDMS from 89.1 to 90.4 on NAVSIM without reinforcement learning or test-time scoring, and increases frozen linear-probe accuracy from 68.3% to 80.2% on UCF-101 action recognition. It also converges roughly twice as fast from scratch and consistently outperforms existing diffusion representation-learning methods under a unified training protocol. These results indicate that explicitly optimized representations are central to translating world knowledge into planning capability in WAMs.

The main contributions are summarized as follows:

(1) We present ReWorld, the first representation learning framework specifically designed to optimize the latent world-to-action pathway in autonomous-driving WAMs.

(2) We introduce a progressive curriculum that makes Video DiT states future-predictive, grounds Action DiT states in attended video information, and shapes action representations with geometrically close yet low-scoring hard negatives. These objectives require no external encoders or teacher models and add only 0.3% per-step Video DiT training cost.

(3) We systematically evaluate ReWorld at the output, representation, and decision levels. ReWorld improves video generation, planning, and frozen representation quality, accelerates convergence, and consistently outperforms existing diffusion representation learning methods in controlled experiments.

## II Related Work

### II-A World Models for Driving Video

Video-generation-based world models are widely used for scene generation, data augmentation, and closed-loop simulation [^7] [^9] [^5] [^6] [^8] [^20] [^10] [^21]. World models more broadly aim to internalize physical structure and dynamics into predictive representations [^22] [^23] [^24]. Modeling paradigms have evolved from autoregressive token predictors such as DrivingGPT [^11] to high-fidelity diffusion generators such as MiLA [^25], while OccWorld [^26], OccSora [^27], UniScene [^28], and Genesis [^29] strengthen 3D structure and cross-modal consistency. Another line treats video world models as simulators for evaluation and policy learning, including HUGSIM [^30], RAD [^31], ReSim [^32], ReconDreamer-RL [^33], and OmniNWM [^13].

These advances improve rendering quality, controllability, and simulation utility, yet most work still emphasizes future video prediction itself. How video generators can obtain better *internal* representations—for training efficiency, temporal coherence, and downstream planning—remains comparatively underexplored. For planning-facing WAMs, representation quality is especially important: mid-denoising video states must carry future-predictive structure rather than only photorealistic appearance.

![[ReWorld_infer.drawio.png|Refer to caption]]

Fig. 2: Intermediate-supervised inference and accelerated convergence of ReWorld. (a) ReWorld uses the discrepancy between the intermediate prediction v i v\_{i} and the final prediction f v\_{f} to construct the self-guided velocity w v\_{w}. (b) ReWorld reaches a comparable validation level using approximately half the optimization steps of vanilla flow matching, corresponding to an approximately 2 × 2\\times convergence acceleration. Neither training scheme uses an external representation encoder or teacher model.

### II-B World Action Models

As world models move closer to decision-making, research is shifting from scene simulation toward World Action Models that predict future evolution under action conditioning [^11] [^4] [^3] [^34] [^15] [^14] [^35]. Existing designs differ mainly in how tightly generation and planning share internal state. Shared-backbone co-generators with separate heads (DrivingGPT [^11], Epona [^4], PWM [^36], DriveDreamer-policy [^37], DriveVA [^14]) and sequential “video then policy” pipelines (GenAD [^12], OmniNWM [^13]) still often keep imagined futures and planned actions as parallel streams. Related unified generators further include $\pi_{0}$ -like mixture-of-transformers designs [^38] [^39] [^34] [^3], visual CoT unification [^40], and decision-oriented world models [^2] [^1] [^41]. In contrast, cascading a video generator and a planner so that mid-level video latents condition the Action DiT—as in DriveLaW [^15], inspired by Genie Envisioner-style chaining [^42] —treats the generator as a world-state provider rather than only a renderer.

Chained latent architectures provide a direct path for transferring video-generation priors into planning. ReWorld focuses on a complementary question: how the intermediate states along this path should be explicitly learned. It therefore studies representation formation in the Video DiT, representation transfer through video–action cross-attention, and behavior-oriented shaping in the Action DiT.

### II-C Representation Learning for Generative Models

Representation learning for diffusion generators falls into three broad lines. The first reshapes the latent space on which generation operates, from classical LDM [^43] in a VAE latent space [^44] toward semantically richer autoencoders such as RAE [^45], SVG [^46], VA-VAE [^47], VFM-VAE [^48], AlignTok [^49], and FAE [^50]. The second optimizes intermediate DiT/SiT [^51] [^52] features during training: REPA [^16] aligns them with external encoders, with extensions in REPA-E [^53], U-REPA [^54], and iREPA [^55]; SRA [^17] replaces teachers with self-alignment, while DiverseDiT [^56], ReDi [^57], SFD [^58], and REG [^59] explore diversity and token-level objectives. The third uses latent predictions at inference to refine sampling, as in Latent Forcing [^60].

Most of this literature targets image generation. Transfer to long-horizon driving video remains underexplored, and gains from external teachers are often inconsistent in this regime: encoders pretrained on static images or short clips may supply semantic priors that are poorly matched to multi-second ego and agent dynamics, while self-alignment methods designed for images may not stress the temporal horizon that driving requires. ReWorld instead studies representation objectives that serve both generation and planning without an external teacher, evaluated under a protocol that stresses long-horizon temporal modeling and planning-oriented control.

## III Method

### III-A Preliminaries

ReWorld is instantiated on a chained latent World Action Model (WAM), in which a Video DiT models future scene evolution and an Action DiT generates ego trajectories conditioned on the Video DiT’s mid-denoising states. We follow the DriveLaW architecture [^15] for this latent world-to-action interface. Unlike parallel designs that connect generation and planning only through decoded frames or separate policy heads, the chained design conditions the Action DiT [^51] directly on mid-denoising video features $\mathcal{F}$. These features are extracted at the initial reverse-flow step, corresponding to the maximum-noise endpoint of video flow time and the first discrete video denoising step. They provide the planner with an abstract scene representation shaped by the video-generation prior [^42].

Concretely, the chained WAM consists of a 2B Video DiT [^51] and a 133M Action DiT (Fig. 1). The Video DiT models future scene evolution from highly compressed causal VAE latents and employs hybrid late-stage pixel decoding [^61]. Following diffusion-based planners [^62] [^63] [^64], the Action DiT predicts ego trajectories directly from mid-denoising video features rather than from a separate perception or policy head. At the first discrete video denoising step, we retain the block features $\mathcal{F}=\{f^{(b)}\}_{b=1}^{B}$ within the current forward pass and condition the planner on them. Thus, imagined scene evolution and planned motion are connected through a shared latent interface. Here, “cached” means that these activations are reused across action flow-matching steps, rather than precomputed or detached offline. The standard WAM is pretrained with a motion-first progressive schedule under output-level supervision [^15].

Given a driving clip $x$, ego kinematics $s_{\leq 0}$, and navigation command $g$, the spatiotemporal VAE [^61] encodes the clip into a compact latent representation $z_{0}=E(x)$. Ego motion and navigation intent for the video branch are converted into a motion-conditioned natural-language prompt and encoded by a frozen T5 text encoder, producing the condition embedding $c^{v}$. Under the rectified-flow parameterization [^65], we independently sample the video flow time $t_{v}\sim\mathcal{U}(0,1)$ and construct

$$
z_{t_{v}}=(1-t_{v})z_{0}+t_{v}\epsilon_{z},\qquad\epsilon_{z}\sim\mathcal{N}(0,I).
$$

The Video DiT velocity field $v_{\theta}^{z}$ is trained with

$$
\mathcal{L}_{\mathrm{Gen}}=\mathbb{E}_{z_{0},t_{v},\epsilon_{z}}\left[\left\|v_{\theta}^{z}(z_{t_{v}},t_{v},c^{v})-(\epsilon_{z}-z_{0})\right\|_{2}^{2}\right].
$$

The future ego trajectory is represented as $a_{0}\in\mathbb{R}^{L\times 3}$, consisting of waypoints $(x_{\ell},y_{\ell},\psi_{\ell})$, and is normalized before flow matching. Given an independently sampled action flow time $t_{a}\sim\mathcal{U}(0,1)$ and noise $\epsilon_{a}\sim\mathcal{N}(0,I)$, we construct

$$
a_{t_{a}}=(1-t_{a})a_{0}+t_{a}\epsilon_{a}.
$$

The noised trajectory and structured driving context are encoded as

$$
h_{\mathrm{act}}=E_{\mathrm{act}}(a_{t_{a}}),\qquad h_{\mathrm{ctx}}=E_{\mathrm{ctx}}([s_{\leq 0};g]).
$$

Conditioned on the driving context and video features, the Action DiT predicts

$$
v_{\phi}^{a}(a_{t_{a}},t_{a},c^{a},\mathcal{F})=\mathrm{DiT}_{\mathrm{act}}\left([h_{\mathrm{act}};t_{a}]\,\middle|\,h_{\mathrm{ctx}},\mathcal{F}\right),
$$

where $c^{a}=\{s_{\leq 0},g\}$ is represented by $h_{\mathrm{ctx}}$. The corresponding flow-matching objective [^66] is

$$
\mathcal{L}_{\mathrm{FM}}=\mathbb{E}_{a_{0},t_{a},\epsilon_{a}}\left[\left\|v_{\phi}^{a}(a_{t_{a}},t_{a},c^{a},\mathcal{F})-(\epsilon_{a}-a_{0})\right\|_{2}^{2}\right].
$$

The standard chained WAM is trained sequentially with

$$
\underbrace{\mathcal{L}_{\mathrm{Gen}}}_{\text{video training}}\quad\longrightarrow\quad\underbrace{\mathcal{L}_{\mathrm{FM}}}_{\text{action training}}.
$$

During planning inference, the initial video latent is sampled from $\mathcal{N}(0,I)$, and a single Video DiT forward pass at the first discrete denoising step extracts $\mathcal{F}$. The Action DiT then generates the trajectory without decoding the future video, retaining the computational efficiency of latent-space planning. During training, the video and action flow times $t_{v}$ and $t_{a}$ are sampled independently. In Stage 2, the Video DiT parameters are frozen. In Stage 3, $\mathcal{F}$ is recomputed online without gradient detachment, allowing planning losses to propagate through the Action DiT conditioning pathway into the Video DiT. In the following, $v_{i}$, $v_{f}$, $v_{w}$, and $\gamma$ denote the intermediate velocity, final velocity, self-guided velocity, and guidance scale, respectively, as defined in Eq. equation 11.

### III-B Explicitly Shaping the World-to-Action Representation Pathway

A standard chained WAM is optimized using output-level video-generation and trajectory-prediction objectives. Although these objectives provide the supervision required for the two output tasks, they leave the intermediate world-to-action pathway largely unconstrained. In particular, they do not explicitly specify what future information should be exposed by planner-facing Video DiT states, what video information should be retained by Action DiT states after cross-attention, or how the resulting action space should reflect closed-loop behavior quality.

To the best of our knowledge, ReWorld is the first representation learning framework specifically designed for autonomous-driving WAMs. It addresses these three limitations through a progressive curriculum for representation formation, cross-modal transfer, and decision-oriented shaping. First, ReWorld directly supervises intermediate Video DiT states using the future-video flow target, encouraging future-predictive information to emerge before the final prediction head. Second, it freezes the video branch and uses the attended video readout computed in each forward pass as a stop-gradient grounding target for the Action DiT. Third, it introduces a decision-oriented trajectory objective that repels predictions from geometrically close but low-scoring alternatives. The first two stages directly constrain intermediate features, whereas the third shapes the shared representation pathway through hard-negative trajectory gradients.

These objectives are applied sequentially because they play distinct optimization roles. Stage 1 first establishes future-predictive structure in the video representation. Stage 2 then grounds the action representation in a stable video feature space. Finally, Stage 3 allows planning-oriented gradients to jointly adapt the Action DiT and the planner-facing Video DiT states. Accordingly, we apply $\mathcal{L}_{\mathrm{align}}$ and $\mathcal{L}_{\mathrm{RDE}}$ in separate stages rather than combining them into a single objective. Stage 2 requires stop-gradient readouts derived from frozen video features to establish a stable grounding target, whereas Stage 3 intentionally allows decision-oriented gradients to reshape the planner-facing video states.

### III-C Future-Predictive World Representations

Video generators acquire powerful world priors because predicting future observations requires internalizing how scenes evolve [^22] [^23] [^67]. Under standard diffusion training, however, this predictive structure is explicitly enforced only at the final output. Intermediate layers are free to organize information according to the final generation objective, without being directly required to expose future-predictive structure. This creates a mismatch in a chained WAM because the planner consumes precisely these intermediate Video DiT states. Inspired by [^68], we attach auxiliary prediction heads to selected intermediate Video DiT layers.

Let $h_{t_{v}}^{(l)}$ denote the hidden feature of the $l$ -th Video DiT block at video flow time $t_{v}$. For every supervised layer $l\in\mathcal{S}$, a lightweight head $q_{l}(\cdot)$ predicts the same velocity target as the final generation head:

$$
\hat{v}_{t_{v}}^{(l)}=q_{l}\!\left(h_{t_{v}}^{(l)}\right).
$$

Unless otherwise specified, we use $\mathcal{S}=\{8\}$. The intermediate supervision objective is

$$
\mathcal{L}_{\mathrm{Mid}}=\frac{1}{|\mathcal{S}|}\sum_{l\in\mathcal{S}}\mathbb{E}_{z_{0},t_{v},\epsilon_{z}}\left[\left\|\hat{v}_{t_{v}}^{(l)}-v_{t_{v}}^{*}\right\|_{2}^{2}\right],\qquad v_{t_{v}}^{*}=\epsilon_{z}-z_{0},
$$

and the Stage 1 objective becomes

$$
\mathcal{L}_{\mathrm{Video}}=\mathcal{L}_{\mathrm{Gen}}+\lambda_{\mathrm{Mid}}\mathcal{L}_{\mathrm{Mid}}.
$$

Intermediate supervision is the representation-learning objective, whereas self-guidance is an inference-time mechanism enabled by the resulting cross-layer prediction hierarchy. During training, $\mathcal{L}_{\mathrm{Mid}}$ moves the future-video constraint from the final output into the process of intermediate representation formation. Empirically, this supervision reaches a comparable validation level using approximately half the optimization steps required by vanilla flow matching, corresponding to an approximately $2\times$ convergence acceleration (Fig. 2(b)).

Intermediate supervision also induces a systematic discrepancy between the intermediate and final velocity predictions (Fig. 2(a)). At inference, we use this cross-layer discrepancy as a correction direction. Let $v_{i}$ and $v_{f}$ denote the velocities predicted by the supervised intermediate head and final head, respectively. We construct the self-guided velocity as

$$
v_{w}=v_{i}+\gamma\left(v_{f}-v_{i}\right),
$$

where $\gamma$ is the guidance scale. The scheduler uses $v_{w}$ in place of $v_{f}$ to advance the denoising trajectory. This correction is applied only during sampling and does not alter the training objective. The Action DiT continues to read $\mathcal{F}$ from the first discrete video denoising step of the same Video DiT. Therefore, self-guidance improves video sampling without changing the planner-facing latent interface. The reported FVD of 61.9 reflects the combined effect of intermediate supervision and self-guided sampling.

### III-D World-Grounded Action Representations

The chained interface allows action tokens to retrieve video information through cross-attention, but access alone does not guarantee retention. Under standard trajectory supervision, the Action DiT may use attended video information transiently without preserving it in the post-attention states that support subsequent trajectory generation. We therefore introduce an explicit alignment objective that encourages Action DiT states to retain information consistent with their attended video readouts.

At the $k$ -th cross-attention layer of the Action DiT, let $a_{i}^{(k)}$ denote the post-cross-attention hidden state of action token $i$. Let $\alpha_{ij}^{(k)}$ denote its attention weight on video token $j$, and let $v_{j}^{(k)}$ denote the corresponding value vector. For notational simplicity, we write the output-projected multi-head cross-attention readout as

$$
r_{i}^{(k)}=\sum_{j}\alpha_{ij}^{(k)}v_{j}^{(k)}.
$$

We encourage the post-attention action state to remain consistent with this retrieved video information in representation space. Empirically, we apply the loss to a deep Action DiT cross-attention layer, $\mathcal{K}=\{12\}$, where video information has already been aggregated into action states that directly support trajectory flow matching:

$$
\mathcal{L}_{\mathrm{align}}=\frac{1}{|\mathcal{K}|N_{a}}\sum_{k\in\mathcal{K}}\sum_{i=1}^{N_{a}}\left[1-\cos\!\left(a_{i}^{(k)},\operatorname{sg}(r_{i}^{(k)})\right)\right].
$$

Here, $N_{a}$ is the number of action tokens, and cosine similarity measures directional agreement in representation space. The stop-gradient operator prevents the readout target from adapting merely to reduce the alignment loss. Gradients continue to flow through $a_{i}^{(k)}$, including its Action DiT cross-attention pathway, so the planner is trained to preserve the attended information rather than allowing the grounding target to follow the planner state.

In Stage 2, the Video DiT is frozen and only the Action DiT is updated:

$$
\mathcal{L}_{\mathrm{act}}^{(2)}=\mathcal{L}_{\mathrm{FM}}+\lambda_{\mathrm{align}}\mathcal{L}_{\mathrm{align}}.
$$

Freezing the video branch establishes a stable world-representation space. The attended readout is recomputed in every forward pass, and its attention weights still depend on the Action DiT, but the stop-gradient operation makes the resulting readout a fixed grounding target within each optimization step.

### III-E Behavior-Aware Action Representations

World-grounded action states encode scene content and future dynamics, but they are not explicitly organized according to closed-loop behavior quality. A trajectory may be geometrically close to the expert while still causing collisions, violating drivable-area constraints, or reducing progress and comfort. Standard imitation learning provides little contrastive supervision among such nearby alternatives. Alignment with video readouts cannot provide this signal either because the video prior is not optimized using closed-loop quality scores. Inspired by [^69], we introduce hard-negative trajectory supervision to make the world-to-action pathway sensitive to low-scoring closed-loop behaviors.

Hard-negative construction. For each training scene, we construct an offline pool of $N$ candidate trajectories $\{\tau^{(n)}\}_{n=1}^{N}$, where $\tau^{(n)}\in\mathbb{R}^{L\times 3}$, and evaluate them using the NAVSIM PDM simulator [^19]. We use the overall PDM score as the closed-loop quality function $s(\cdot)$, with higher values indicating better aggregate behavior. For an expert trajectory $\tau^{\mathrm{exp}}$, the hard negative $\tau^{\mathrm{neg}}$ is defined as the closest low-scoring candidate in normalized trajectory space:

$$
\displaystyle\mathcal{I}_{\mathrm{low}}=\left\{n\mid s(\tau^{(n)})<\delta\right\},\qquad\delta=0.6,
$$
$$
\displaystyle n^{\star}=\arg\min_{n\in\mathcal{I}_{\mathrm{low}}}\frac{1}{L}\sum_{\ell=1}^{L}\left\|\tau_{\ell}^{(n)}-\tau_{\ell}^{\mathrm{exp}}\right\|_{2}^{2},
$$
$$
\displaystyle\tau^{\mathrm{neg}}=\tau^{(n^{\star})}.
$$

The nearest-neighbor distance is computed after applying the same per-channel normalization used for trajectory flow matching. Scenes without a low-scoring candidate are excluded from this objective. Selecting the nearest low-scoring trajectory, rather than a random low-scoring candidate, matters because random negatives are often distinguishable by geometry alone. The selected hard negative instead occupies the local neighborhood of the expert while scoring poorly in closed loop, exposing precisely the behavioral ambiguity that standard imitation does not resolve.

Repulsive distance loss. Because the planner parameterizes trajectories through a velocity field, we recover an instantaneous clean trajectory estimate from the same forward pass and the same sampled $t_{a}$ used by $\mathcal{L}_{\mathrm{FM}}$:

$$
\hat{a}_{0}=a_{t_{a}}-t_{a}v_{\phi}^{a}(a_{t_{a}},t_{a},c^{a},\mathcal{F}),\qquad\hat{\tau}=\operatorname{Denorm}(\hat{a}_{0}),
$$

where $\operatorname{Denorm}(\cdot)$ is the differentiable affine inverse of the normalization used for trajectory flow matching. To emphasize relative motion rather than absolute position, we use a delta representation. Each waypoint is mapped to

$$
\Delta(\tau)_{\ell}=\left[\widetilde{\Delta x}_{\ell},\widetilde{\Delta y}_{\ell},\sin\psi_{\ell},\cos\psi_{\ell}\right]\in\mathbb{R}^{4},
$$

where $\widetilde{\Delta x}_{\ell}$ and $\widetilde{\Delta y}_{\ell}$ are normalized position increments. The sine–cosine parameterization avoids the discontinuity of angular coordinates. We then define the repulsive distance objective as

$$
\mathcal{L}_{\mathrm{RDE}}=-\frac{1}{|\mathcal{V}|}\sum_{b\in\mathcal{V}}\frac{1}{L}\sum_{\ell=1}^{L}\frac{1}{4}\sum_{d=1}^{4}\left|\Delta(\hat{\tau}_{b})_{\ell}^{(d)}-\Delta(\tau_{b}^{\mathrm{neg}})_{\ell}^{(d)}\right|,
$$

where $\mathcal{V}$ denotes the set of training scenes with valid hard negatives, and $d$ indexes the four channels in Eq. equation 19. If no valid hard negative exists in a minibatch, i.e., $\mathcal{V}=\varnothing$, we set $\mathcal{L}_{\mathrm{RDE}}=0$.

The negative sign encourages the predicted trajectory to move away from the selected low-scoring neighbor in delta space, while $\mathcal{L}_{\mathrm{FM}}$ continues to anchor it to the expert trajectory. Although the repulsive objective is not lower-bounded in isolation, it is used only as a weak regularizer alongside the quadratic flow-matching objective. As shown by the weight analysis in Sec. IV-D, excessive repulsion competes with imitation and degrades closed-loop performance. Gradients from $\mathcal{L}_{\mathrm{RDE}}$ propagate through the shared forward graph into the Action DiT. When the Video DiT is unfrozen in Stage 3, $\mathcal{F}$ is recomputed online without detachment, allowing these behavior-oriented gradients to adapt the video features used for planning.

In Stage 3, the Video DiT and Action DiT are jointly fine-tuned:

$$
\mathcal{L}_{\mathrm{act}}^{(3)}=\mathcal{L}_{\mathrm{FM}}+\lambda_{\mathrm{RDE}}\mathcal{L}_{\mathrm{RDE}}.
$$

This stage follows Stage 2 and does not retain $\mathcal{L}_{\mathrm{align}}$, because the planner-facing video representation is intentionally allowed to adapt under behavior-oriented supervision.

### III-F Progressive Representation Curriculum

ReWorld applies a three-stage representation curriculum. Stage 1 trains the Video DiT with future-predictive intermediate supervision using $\mathcal{L}_{\mathrm{Video}}$ (Eq. equation 10). Stage 2 freezes the Video DiT and grounds the Action DiT in attended video information using $\mathcal{L}_{\mathrm{act}}^{(2)}$ (Eq. equation 14). Stage 3 jointly fine-tunes both branches with hard-negative repulsion using $\mathcal{L}_{\mathrm{act}}^{(3)}$ (Eq. equation 21). This ordering first forms a future-predictive world representation, then establishes stable world-to-action transfer, and finally shapes the shared pathway according to closed-loop behavior quality. Initialization, step counts, batch sizes, and hyperparameters are provided in Sec. IV. Hard negatives are mined offline using training scenes only. During video sampling, ReWorld applies self-guidance with $\gamma=1.4$, while the Action DiT continues to read $\mathcal{F}$ from the first discrete video denoising step during planning.

## IV Experiments

### IV-A Experimental Setup

Tasks, datasets, and metrics. Following established autonomous-driving WAM protocols [^4] [^3] [^36], we use nuPlan [^70] and nuScenes [^18] for video training and NAVSIM [^19] for trajectory learning and evaluation. nuScenes contains 1,000 urban driving sequences collected in Boston and Singapore with synchronized camera and LiDAR observations, including 850 development sequences and 150 held-out sequences. nuPlan provides approximately 1,200 hours of human-driving data collected across four metropolitan areas. We sample camera streams at $8\,\mathrm{Hz}$ for video training and use $2\,\mathrm{Hz}$ observations from NAVSIM for trajectory supervision.

NAVSIM is a data-driven, non-reactive closed-loop planning benchmark built on OpenScene [^71], with approximately 103k scenes in Navtrain and 12k scenes in Navtest. We evaluate video generation on nuScenes using Fréchet Inception Distance (FID) [^72] and Fréchet Video Distance (FVD) [^73]. FID measures frame-level visual fidelity, while FVD evaluates video quality with an emphasis on temporal coherence. Temporal ablations therefore focus on FVD. Closed-loop planning is evaluated on NAVSIM v1 [^19] using no-at-fault collision (NC), drivable-area compliance (DAC), time-to-collision (TTC), comfort (Comf.), ego progress (EP), and the Predictive Driver Model Score (PDMS):

$$
\mathrm{PDMS}=\mathrm{NC}\times\mathrm{DAC}\times\frac{5\cdot\mathrm{EP}+5\cdot\mathrm{TTC}+2\cdot\mathrm{Comf.}}{12}.
$$

We further perform frozen linear probing on UCF-101 action recognition to assess the quality and transferability of the spatiotemporal representations learned by the Video DiT.

Backbone and sequential pretraining. ReWorld is instantiated on the chained DriveLaW architecture [^15], comprising a 2B Video DiT [^51] initialized from LTX-Video [^61] and a 133M Action DiT. The standard chained WAM is obtained using the sequential objectives in Eq. equation 7. The Video DiT is first optimized with $\mathcal{L}_{\mathrm{Gen}}$ through a progressive video curriculum consisting of long low-resolution clips ($740\times 352\times 121$), followed by short high-resolution clips ($1280\times 704\times 25$). The Action DiT is subsequently conditioned on the pretrained video latents and optimized with $\mathcal{L}_{\mathrm{FM}}$ for trajectory generation.

Throughout the experiments, DriveLaW denotes this standard chained WAM trained with output-level supervision, while ReWorld augments the same architecture and latent interface with the proposed representation curriculum.

Representation curriculum. Stage 1 initializes the Video DiT from LTX-Video [^61] and optimizes it with $\mathcal{L}_{\mathrm{Gen}}+\lambda_{\mathrm{Mid}}\mathcal{L}_{\mathrm{Mid}}$ for 20k steps using a global batch size of 64. We use AdamW with a learning rate of $1\times 10^{-5}$ and a weight decay of $5\times 10^{-2}$. The video flow time is sampled token-wise from $t_{v}\in[0,1]$, and the default supervised block is $\mathcal{S}=\{8\}$.

Stage 2 initializes from the DriveLaW checkpoint, freezes the Video DiT, and optimizes the Action DiT with $\mathcal{L}_{\mathrm{FM}}+\lambda_{\mathrm{align}}\mathcal{L}_{\mathrm{align}}$ for 6k steps using a batch size of 128. We set $\lambda_{\mathrm{align}}=0.05$ and apply the alignment objective at the 12th cross-attention layer.

Stage 3 jointly fine-tunes the Video DiT and Action DiT with $\mathcal{L}_{\mathrm{FM}}+\lambda_{\mathrm{RDE}}\mathcal{L}_{\mathrm{RDE}}$ for 10k steps using a batch size of 160, with $\lambda_{\mathrm{RDE}}=0.04$. Video features are recomputed online, allowing decision-oriented gradients to shape the planner-facing world representations.

Hard negatives are mined offline from the training set following BeyondDrive [^69]. For each scene, a flow-matching trajectory generator produces 64 candidate trajectories using classifier-free guidance and noise-scale adjustment to increase diversity. Each candidate is evaluated by the NAVSIM PDM simulator. Candidates with scores below $\delta=0.6$ form the low-scoring set, from which we select the trajectory closest to the expert under the same per-channel normalization used for trajectory flow matching. Video generation uses 30 sampling steps with self-guidance scale $\gamma=1.4$, and trajectory generation uses five flow-matching steps.

Frozen linear probing. We evaluate frozen Video DiT representations on UCF-101 split 1. All checkpoints follow the same protocol, using 33-frame clips at $224\times 224$, the same video flow timestep, feature normalization, and classifier schedule. Given VAE-encoded video latents, we extract the representation from the final block of the 28-layer Video DiT,

$$
h^{(28)}\in\mathbb{R}^{B\times N\times C_{h}},
$$

apply global mean pooling over the token dimension, and train a linear classifier on the resulting $\mathbb{R}^{B\times C_{h}}$ features. We compare LTX-Video, DriveLaW, ReWorld after Stage 1, and ReWorld after the full curriculum.

Unified representation-learning protocol. For controlled comparison with diffusion representation-learning methods, we train all approaches from scratch on LTX-Video using nuPlan and nuScenes for 120k steps. The unified setting uses a batch size of 32, $224\times 224\times 25$ video clips, and no text encoder. We report FVD on the nuScenes test set and normalized per-step Video DiT training cost relative to vanilla flow matching.

### IV-B Main Results

We first evaluate the two functional endpoints of a WAM. Video generation measures the quality of future-world modeling, while closed-loop planning evaluates how effectively the learned world representation supports action generation.

Video generation. Tab. I reports video-generation results on the nuScenes validation set. DriveLaW establishes a strong baseline with 4.6 FID and 81.3 FVD, outperforming previous single-view generators including DriveWorld [^74], Vista [^6], and Epona [^4].

With standard sampling ($\gamma=1.0$), future-predictive intermediate supervision improves FVD from 81.3 to 78.9, showing that direct supervision of intermediate Video DiT states strengthens temporal generation. The resulting intermediate-to-final prediction hierarchy further enables self-guided sampling, reducing FVD to 61.9, a relative improvement of 23.9% over DriveLaW. ReWorld also improves FID from 4.6 to 4.4.

The FVD reduction indicates stronger modeling of scene dynamics and cross-frame consistency, while the FID result shows that ReWorld preserves the frame-level fidelity of the underlying video generator. Together, these results support future-predictive intermediate supervision as a mechanism for improving both the sampling process and the generated future.

TABLE I: Video generation on the nuScenes validation set. DriveLaW denotes the standard chained WAM trained with the sequential objectives in Eq. equation 7. ReWorld uses future-predictive intermediate supervision and self-guided sampling with $\gamma=1.4$.

| Method | FID $\downarrow$ | FVD $\downarrow$ |
| --- | --- | --- |
| DriveGAN [^75] | 73.4 | 502.3 |
| DriveDreamer [^9] | 52.6 | 452.0 |
| DrivingGPT [^11] | 12.8 | 142.6 |
| DriveWorld [^74] | 7.4 | 90.9 |
| Vista [^6] | 6.9 | 89.4 |
| Epona [^4] | 7.5 | 82.8 |
| DriveLaW [^15] | 4.6 | 81.3 |
| ReWorld (Ours) | 4.4 | 61.9 |

Closed-loop planning. Tab. II reports closed-loop planning results on NAVSIM Navtest. ReWorld improves PDMS from 89.1 to 90.4 on the same chained WAM architecture, achieving the best overall performance among the compared methods. The gains are particularly evident in safety- and compliance-related metrics: DAC increases from 97.1 to 98.2, TTC from 96.7 to 97.7, and NC from 99.0 to 99.1.

ReWorld outperforms strong traditional end-to-end planners, including the camera–LiDAR DiffusionDrive [^63], as well as world-model-based approaches such as Epona [^4], DriveVLA-W0 [^3], PWM [^36], and WorldDrive [^76]. It achieves the highest PDMS, NC, DAC, and TTC among the compared world-model methods. These results indicate that explicitly learning world-to-action transfer and behavior-sensitive action representations converts predictive world knowledge into safer closed-loop decisions.

TABLE II: Closed-loop planning performance on NAVSIM Navtest. Methods are grouped into traditional end-to-end planners and world-model methods. <sup>†</sup> denotes training with the same flow-matching objective. ReWorld applies the proposed representation curriculum to the chained DriveLaW architecture.

<table><tbody><tr><td>Method</td><td>Ref</td><td>Image</td><td>Lidar</td><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Comf.<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td colspan="10">Traditional End-to-End Methods</td></tr><tr><td>VADv2- <math><semantics><msub><mi>𝒱</mi> <mtext>8192</mtext></msub> <annotation>\mathcal{V}_{\text{8192}}</annotation></semantics></math> <sup><a href="#fn:77">77</a></sup></td><td>arXiv’24</td><td>✓</td><td></td><td>97.2</td><td>89.1</td><td>91.6</td><td>100</td><td>76.0</td><td>80.9</td></tr><tr><td>UniAD <sup><a href="#fn:78">78</a></sup></td><td>CVPR’23</td><td>✓</td><td></td><td>97.8</td><td>91.9</td><td>92.9</td><td>100</td><td>78.8</td><td>83.4</td></tr><tr><td>TransFuser <sup><a href="#fn:79">79</a></sup></td><td>TPAMI’23</td><td>✓</td><td>✓</td><td>97.7</td><td>92.8</td><td>92.8</td><td>100</td><td>79.2</td><td>84.0</td></tr><tr><td>PARA-Drive <sup><a href="#fn:80">80</a></sup></td><td>CVPR’24</td><td>✓</td><td></td><td>97.9</td><td>92.4</td><td>93.0</td><td>99.8</td><td>79.3</td><td>84.0</td></tr><tr><td>ReCogDrive-IL <sup><a href="#fn:64">64</a></sup></td><td>ICLR’26</td><td>✓</td><td></td><td>98.1</td><td>94.7</td><td>94.2</td><td>100</td><td>80.9</td><td>86.5</td></tr><tr><td>DiffusionDrive <sup><a href="#fn:63">63</a></sup></td><td>CVPR’25</td><td>✓</td><td>✓</td><td>98.2</td><td>96.2</td><td>94.7</td><td>100</td><td>82.2</td><td>88.1</td></tr><tr><td colspan="10">World Model Methods</td></tr><tr><td>DrivingGPT <sup><a href="#fn:11">11</a></sup></td><td>ICCV’25</td><td>✓</td><td></td><td>98.9</td><td>90.7</td><td>94.9</td><td>95.6</td><td>79.7</td><td>82.4</td></tr><tr><td>LAW <sup><a href="#fn:1">1</a></sup></td><td>ICLR’25</td><td>✓</td><td></td><td>96.4</td><td>95.4</td><td>88.7</td><td>99.9</td><td>81.7</td><td>84.6</td></tr><tr><td>Epona <sup><a href="#fn:4">4</a></sup></td><td>ICCV’25</td><td>✓</td><td></td><td>97.9</td><td>95.1</td><td>93.8</td><td>99.9</td><td>80.4</td><td>86.2</td></tr><tr><td>ReSim <sup><a href="#fn:32">32</a></sup></td><td>NeurIPS’25</td><td>✓</td><td></td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>86.6</td></tr><tr><td>WoTE <sup><a href="#fn:41">41</a></sup></td><td>ICCV’25</td><td>✓</td><td>✓</td><td>98.5</td><td>96.8</td><td>94.9</td><td>99.9</td><td>81.9</td><td>88.3</td></tr><tr><td>DriveVLA-W0 <sup>†</sup> <sup><a href="#fn:3">3</a></sup></td><td>ICLR’26</td><td>✓</td><td></td><td>98.4</td><td>95.3</td><td>95.2</td><td>100</td><td>80.9</td><td>87.2</td></tr><tr><td>PWM <sup><a href="#fn:36">36</a></sup></td><td>NeurIPS’25</td><td>✓</td><td></td><td>98.6</td><td>95.9</td><td>95.4</td><td>100</td><td>81.8</td><td>88.1</td></tr><tr><td>WorldDrive <sup><a href="#fn:76">76</a></sup></td><td>arXiv’26</td><td>✓</td><td></td><td>98.4</td><td>96.8</td><td>95.2</td><td>100</td><td>83.3</td><td>89.0</td></tr><tr><td>DriveLaW <sup><a href="#fn:15">15</a></sup></td><td>CVPR’26</td><td>✓</td><td></td><td>99.0</td><td>97.1</td><td>96.7</td><td>100</td><td>81.3</td><td>89.1</td></tr><tr><td>ReWorld (Ours)</td><td>-</td><td>✓</td><td></td><td>99.1</td><td>98.2</td><td>97.7</td><td>99.8</td><td>82.0</td><td>90.4</td></tr></tbody></table>

Fig. 3 provides a qualitative comparison of future video generation between ReWorld and DriveLaW. Both methods are conditioned on one second of historical observation ($8$ frames at $8\,\mathrm{Hz}$) and synthesize the subsequent three seconds ($24$ frames). In the figure, the dashed line separates the conditioning interval from the generated future, and each scene is shown as a DriveLaW–ReWorld pair. For visualization, columns $T{-}1$ and $T$ show only the first and last frames of the $1\,\mathrm{s}$ history, while columns $T{+}1$ to $T{+}3$ depict the predicted $3\,\mathrm{s}$ future. Relative to DriveLaW, ReWorld yields sharper lane markings, more stable roadside structures, clearer distant agents, and stronger temporal continuity over the predicted horizon, which is consistent with the observed FVD reduction and indicates that future-predictive intermediate supervision improves long-horizon video coherence under challenging dynamics.

![[contrast.png|Refer to caption]]

Fig. 3: Qualitative video-generation comparison with DriveLaW 15. Conditioning uses 1 s 1\\,\\mathrm{s} history ( 8 frames at Hz 8\\,\\mathrm{Hz} ); columns T − T{-}1 and show only its first and last frames for visualization. Generated future frames ( 3 3\\,\\mathrm{s}, 24 frames) lie to the right of the dashed line (columns + T{+}1 to T{+}3 ). Each pair of consecutive rows depicts one scene for DriveLaW and ReWorld, respectively. ReWorld better preserves lane markings, roadside geometry, distant objects, and temporal consistency.

### IV-C Representation Analysis

The task-level results show gains at both outputs of the WAM. We next examine the representations underlying these improvements from two perspectives: their transferability to motion recognition and their behavior under a unified generative representation-learning protocol.

Frozen linear probing on UCF-101. Tab. III reports frozen linear-probe accuracy on UCF-101. LTX-Video achieves 66.8%, while driving-domain pretraining in DriveLaW improves the accuracy to 68.3%. Stage 1 future-predictive supervision further increases it to 71.7%, demonstrating that intermediate flow supervision strengthens the motion-discriminative structure encoded by the Video DiT.

After the full representation curriculum, ReWorld reaches 80.2%, improving by 11.9 points over DriveLaW. The progression from generic video pretraining to driving adaptation and then to explicit representation learning shows that ReWorld yields stronger spatiotemporal representations whose motion structure transfers to action recognition.

TABLE III: Frozen linear probing on UCF-101 split 1. All methods use 33-frame clips at $224\times 224$, final-block Video DiT features, global mean pooling, and the same linear-classifier protocol.

| Frozen Video DiT representation | Top-1 Acc. (%) $\uparrow$ |
| --- | --- |
| LTX-Video [^61] | 66.8 |
| DriveLaW [^15] | 68.3 |
| ReWorld (Stage 1) | 71.7 |
| ReWorld | 80.2 |

Comparison with representation-learning methods. Tab. IV compares representation-learning strategies under the unified long-horizon driving-video protocol. ReWorld achieves the best FVD of 270.4, outperforming Vanilla Flow by 33.7 points and the strongest competing self-supervised method, Self-Flow, by 12.9 points.

The comparison highlights the importance of task-aligned representation objectives for generative modeling. External feature spaces primarily encode semantic, geometric, or recognition-oriented structure. Long-horizon video generation, however, requires intermediate states to preserve continuous temporal evolution and flow-consistent dynamics. ReWorld directly applies the generator’s native future-flow target to intermediate states, aligning representation learning with the temporal structure of the generative process. This generator-native supervision yields the strongest video-generation performance among the compared methods.

TABLE IV: Representation learning for long-horizon driving-video generation. All methods are trained from scratch for 120k steps on nuPlan and nuScenes using $224\times 224\times 25$ clips without text conditioning.

<table><tbody><tr><th>Model</th><td>Steps</td><td>FVD <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><th colspan="3">Without external representations</th></tr><tr><th>Vanilla Flow</th><td>120k</td><td>304.1</td></tr><tr><th>SRA <sup><a href="#fn:17">17</a></sup></th><td>120k</td><td>296.9</td></tr><tr><th>SRA2 <sup><a href="#fn:81">81</a></sup></th><td>120k</td><td>295.2</td></tr><tr><th>Self-Flow <sup><a href="#fn:82">82</a></sup></th><td>120k</td><td>283.3</td></tr><tr><th>ReWorld (Ours)</th><td>120k</td><td>270.4</td></tr><tr><th colspan="3">With external representations</th></tr><tr><th>REPA w/ DINOv2 <sup><a href="#fn:16">16</a></sup> <sup><a href="#fn:83">83</a></sup></th><td>120k</td><td>295.9</td></tr><tr><th>REPA w/ VideoMAEv2 <sup><a href="#fn:16">16</a></sup> <sup><a href="#fn:84">84</a></sup></th><td>120k</td><td>328.3</td></tr><tr><th>REPA w/ DepthAnything3 <sup><a href="#fn:16">16</a></sup> <sup><a href="#fn:85">85</a></sup></th><td>120k</td><td>319.4</td></tr><tr><th>REPA w/ V-JEPA2 <sup><a href="#fn:16">16</a></sup> <sup><a href="#fn:24">24</a></sup></th><td>120k</td><td>331.6</td></tr><tr><th>ReDi <sup><a href="#fn:57">57</a></sup></th><td>120k</td><td>421.7</td></tr></tbody></table>

Training efficiency. Tab. V compares the normalized per-step cost of video-side representation learning. ReWorld introduces only a lightweight intermediate prediction head and incurs $1.003\times$ training cost. In comparison, SRA and Self-Flow require an additional DiT forward pass to construct self-supervised targets, while external-alignment methods require a separate representation encoder.

ReWorld therefore combines the best FVD in Tab. IV with only 0.3% additional per-step Video DiT computation. Its efficiency follows from reusing the future-flow target and intermediate activations already available in the generative model.

TABLE V: Normalized per-step cost of video-side representation learning. Costs are measured at $224\times 224\times 25$ with batch size 1, using vanilla flow matching as $1.0\times$.

<table><tbody><tr><th>Method</th><td>Normalized Training Cost</td></tr><tr><th colspan="2">Without external representations</th></tr><tr><th>Vanilla Flow</th><td>1.0 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th>Self-Flow <sup><a href="#fn:82">82</a></sup></th><td><math><semantics><mo>∼</mo> <annotation>\sim</annotation></semantics></math> 1.4 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th>SRA <sup><a href="#fn:17">17</a></sup></th><td><math><semantics><mo>∼</mo> <annotation>\sim</annotation></semantics></math> 1.4 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th>ReWorld</th><td>1.003 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th colspan="2">With external representations</th></tr><tr><th>REPA <sup><a href="#fn:16">16</a></sup></th><td><math><semantics><mo>∼</mo> <annotation>\sim</annotation></semantics></math> 1.7 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th>ReDi <sup><a href="#fn:57">57</a></sup></th><td><math><semantics><mo>∼</mo> <annotation>\sim</annotation></semantics></math> 1.6 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr></tbody></table>

![[qualitive_new.png|Refer to caption]]

Fig. 4: Additional video-generation results on nuScenes. Conditioning uses 1 s 1\\,\\mathrm{s} history ( 8 frames at Hz 8\\,\\mathrm{Hz} ); columns T − T{-}1 and show only its first and last frames for visualization, as in Fig. 3. Generated futures ( 3\\,\\mathrm{s}, 24 frames) appear to the right of the dashed line (columns + T{+}1 to T{+}3 ). Each row corresponds to one driving scene, covering sunny and rainy weather, high-speed travel, intersections, and other urban conditions. ReWorld produces temporally coherent futures with stable geometry, clear lane structure, and consistent appearance of surrounding agents.

![[navsimvis.png|Refer to caption]]

Fig. 5: Additional planning results on NAVSIM Navtest. From left to right: straight, turn left, turn right, and intersection. Red curves denote the ego trajectories predicted by our model, and green curves denote the ground-truth expert trajectories.

### IV-D Ablation Studies

We analyze the representation curriculum along its three functional stages: future-predictive world representation, world-to-action transfer, and behavior-aware action shaping.

Future-predictive world representation. Intermediate supervision substantially accelerates optimization. As shown in Fig. 2(b), ReWorld reaches the reference validation performance using approximately half the training steps required by vanilla flow matching, corresponding to an approximately $2\times$ convergence acceleration.

Tab. VI(a) studies the location of intermediate supervision. Block 8 achieves the best FVD of 61.9, indicating a favorable balance between representation maturity and subsequent refinement. Earlier blocks provide less developed future estimates, while deeper blocks offer less hierarchical separation from the final prediction head.

Tab. VI(b) studies the self-guidance scale. Standard sampling with $\gamma=1.0$ yields 78.9 FVD. Increasing $\gamma$ to 1.4 improves FVD to 61.9, showing that the discrepancy between intermediate and final predictions provides an effective refinement direction. The non-monotonic trend further indicates that moderate guidance best exploits the learned prediction hierarchy.

TABLE VI: Intermediate-layer and self-guidance ablations. (a) FVD with different supervised Video DiT blocks. (b) FVD under different self-guidance scales using block 8.

(a) Supervised intermediate block  
Supervised Block FVD $\downarrow$ 2 65.5 8 61.9 12 62.7 16 63.0 20 64.3

(b) Self-guidance scale  
$\gamma$ FVD $\downarrow$ 1.0 78.9 1.2 72.0 1.4 61.9 1.6 69.7 1.8 68.2

World-grounded action representations. With the Video DiT frozen, $\mathcal{L}_{\mathrm{align}}$ improves PDMS from 89.1 to 89.5, as shown in Tab. VII. The improvement isolates action-side representation learning and suggests that preserving the attended video readout strengthens world-to-action knowledge transfer. Tab. VIII(a) further studies the alignment weight $\lambda_{\mathrm{align}}$ in Stage 2 only, without Stage 3. The best closed-loop performance under this Stage 2 setting is obtained at $\lambda_{\mathrm{align}}=0.05$ (PDMS $89.5$). Weaker weights provide insufficient grounding of action states to the attended video readout, whereas stronger weights over-constrain the Action DiT and interfere with trajectory flow matching.

Behavior-aware action shaping. Applying $\mathcal{L}_{\mathrm{RDE}}$ improves PDMS from 89.1 to 89.8, while progressively combining world grounding and behavior-aware shaping reaches 90.4. The gains confirm their complementary roles: alignment establishes a world-grounded action representation, and RDE further separates locally similar trajectories according to closed-loop behavior quality.

Tab. VIII(b) then studies the balance between expert imitation and hard-negative repulsion by training Stage 3 from the best Stage 2 checkpoint above (PDMS $89.5$ at $\lambda_{\mathrm{align}}=0.05$) and varying only $\lambda_{\mathrm{RDE}}$. The best performance is obtained at $\lambda_{\mathrm{RDE}}=0.04$. Smaller weights provide weaker behavioral separation, while larger weights increasingly compete with expert trajectory matching. This non-monotonic trend shows that RDE is most effective as a local decision-oriented regularizer around the expert trajectory manifold.

TABLE VII: Effects of world grounding and behavior-aware action shaping. PDMS is evaluated on NAVSIM Navtest using the same chained WAM architecture.

| Configuration | $\mathcal{L}_{\mathrm{align}}$ | $\mathcal{L}_{\mathrm{RDE}}$ | PDMS $\uparrow$ |
| --- | --- | --- | --- |
| DriveLaW |  |  | 89.1 |
| \+ Align only | ✓ |  | 89.5 |
| \+ RDE only |  | ✓ | 89.8 |
| ReWorld | ✓ | ✓ | 90.4 |

TABLE VIII: Alignment and RDE weight ablations. (a) Stage 2 only (no Stage 3): PDMS under different $\lambda_{\mathrm{align}}$. (b) Stage 3 trained from the best Stage 2 checkpoint in (a) ($\lambda_{\mathrm{align}}=0.05$, PDMS $89.5$): PDMS under different $\lambda_{\mathrm{RDE}}$.

(a) Alignment weight  
$\lambda_{\mathrm{align}}$ PDMS $\uparrow$ 0.01 88.8 0.03 89.2 0.05 89.5 0.07 88.2 0.10 87.7

(b) RDE weight  
$\lambda_{\mathrm{RDE}}$ PDMS $\uparrow$ 0.02 89.4 0.03 89.6 0.04 90.4 0.05 89.7 0.10 85.5

### IV-E Qualitative and Additional Results

Fig. 4 presents additional generation results of ReWorld on nuScenes under the same conditioning protocol as Fig. 3. Across six representative scenes spanning clear and rainy weather, high-speed driving, intersections, and other common urban settings, the synthesized futures remain temporally coherent, with stable lane geometry, well-preserved roadside structure, and consistent multi-agent appearance over the three-second horizon.

Fig. 5 further shows closed-loop planning examples on NAVSIM Navtest for straight driving, left turn, right turn, and intersection scenarios. Red curves denote trajectories predicted by our model and green curves denote ground-truth expert paths; the predicted plans remain smooth and consistent with the surrounding scene layout.

## V Conclusion

We presented ReWorld, the first representation learning framework for autonomous-driving World Action Models, which targets the under-constrained intermediate representations along the latent world-to-action pathway. ReWorld explicitly shapes the latent world-to-action pathway through a progressive curriculum: intermediate supervision accelerates convergence, cross-modal alignment preserves the attended world information in action states, and hard-negative shaping increases sensitivity to nearby unsafe behaviors. Requiring no external encoders, the method adds only 0.3% per-step Video DiT training cost. Across generation, closed-loop planning, and out-of-distribution probing, the results support explicit representation optimization as a central factor in translating world knowledge into planning capability. Future work will explore extensions to longer temporal horizons and multi-modal scenarios.

## Acknowledgments

This work was in part supported by the National Natural Science Foundation of China (NSFC U25B2067).

## References

![[Uncaptioned image]](https://arxiv.org/html/2606.27504v2/bios_photos/tianze_xia.jpg)

Tianze Xia

![[Uncaptioned image]](https://arxiv.org/html/2606.27504v2/bios_photos/lijun_zhou.jpg)

Lijun Zhou

![[Uncaptioned image]](https://arxiv.org/html/2606.27504v2/bios_photos/kaixin_xiong.jpg)

Kaixin Xiong

![[Uncaptioned image]](https://arxiv.org/html/2606.27504v2/bios_photos/jingfeng_yao.jpg)

Jingfeng Yao

![[Uncaptioned image]](https://arxiv.org/html/2606.27504v2/bios_photos/zhenxin_zhu.jpg)

Zhenxin Zhu

![[Uncaptioned image]](https://arxiv.org/html/2606.27504v2/bios_photos/haiyang_sun.jpg)

Haiyang Sun

![[Uncaptioned image]](https://arxiv.org/html/2606.27504v2/bios_photos/bing_wang.png)

![[Uncaptioned image]](https://arxiv.org/html/2606.27504v2/bios_photos/guang_chen.jpg)

Guang Chen

![[Uncaptioned image]](https://arxiv.org/html/2606.27504v2/bios_photos/wenyu_liu.jpg)

Wenyu Liu (SM’15)

![[Uncaptioned image]](https://arxiv.org/html/2606.27504v2/bios_photos/hangjun_ye.jpg)

Hangjun Ye

![[Uncaptioned image]](https://arxiv.org/html/2606.27504v2/bios_photos/xinggang_wang.png)

Xinggang Wang (SM’24)

[^1]: Y. Li, L. Fan, J. He, Y. Wang, Y. Chen, Z. Zhang, and T. Tan, “Enhancing end-to-end autonomous driving with latent world model,” *arXiv preprint arXiv:2406.08481*, 2024.

[^2]: Y. Wang, J. He, L. Fan, H. Li, Y. Chen, and Z. Zhang, “Driving into the future: Multiview visual forecasting and planning with world model for autonomous driving,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2024, pp. 14 749–14 759.

[^3]: Y. Li, S. Shang, W. Liu, B. Zhan, H. Wang, Y. Wang, Y. Chen, X. Wang, Y. An, C. Tang *et al.*, “Drivevla-w0: World models amplify data scaling law in autonomous driving,” *arXiv preprint arXiv:2510.12796*, 2025.

[^4]: K. Zhang, Z. Tang, X. Hu, X. Pan, X. Guo, Y. Liu, J. Huang, L. Yuan, Q. Zhang, X.-X. Long *et al.*, “Epona: Autoregressive diffusion world model for autonomous driving,” *arXiv preprint arXiv:2506.24113*, 2025.

[^5]: R. Gao, K. Chen, E. Xie, L. Hong, Z. Li, D.-Y. Yeung, and Q. Xu, “Magicdrive: Street view generation with diverse 3d geometry control,” *arXiv preprint arXiv:2310.02601*, 2023.

[^6]: S. Gao, J. Yang, L. Chen, K. Chitta, Y. Qiu, A. Geiger, J. Zhang, and H. Li, “Vista: A generalizable driving world model with high fidelity and versatile controllability,” *Advances in Neural Information Processing Systems*, vol. 37, pp. 91 560–91 596, 2024.

[^7]: A. Hu, L. Russell, H. Yeo, Z. Murez, G. Fedoseev, A. Kendall, J. Shotton, and G. Corrado, “Gaia-1: A generative world model for autonomous driving,” *arXiv preprint arXiv:2309.17080*, 2023.

[^8]: X. Li, Y. Zhang, and X. Ye, “Drivingdiffusion: layout-guided multi-view driving scenarios video generation with latent diffusion model,” in *European Conference on Computer Vision*. Springer, 2024, pp. 469–485.

[^9]: X. Wang, Z. Zhu, G. Huang, X. Chen, J. Zhu, and J. Lu, “Drivedreamer: Towards real-world-drive world models for autonomous driving,” in *European conference on computer vision*. Springer, 2024, pp. 55–72.

[^10]: G. Zhao, X. Wang, Z. Zhu, X. Chen, G. Huang, X. Bao, and X. Wang, “Drivedreamer-2: Llm-enhanced world models for diverse driving video generation,” in *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 39, no. 10, 2025, pp. 10 412–10 420.

[^11]: Y. Chen, Y. Wang, and Z. Zhang, “Drivinggpt: Unifying driving world modeling and planning with multi-modal autoregressive transformers,” in *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 2025, pp. 26 890–26 900.

[^12]: W. Zheng, R. Song, X. Guo, C. Zhang, and L. Chen, “Genad: Generative end-to-end autonomous driving,” *arXiv preprint arXiv: 2402.11502*, 2024.

[^13]: B. Li, Z. Ma, D. Du, B. Peng, Z. Liang, Z. Liu, C. Ma, Y. Jin, H. Zhao, W. Zeng *et al.*, “Omninwm: Omniscient driving navigation world models,” *arXiv preprint arXiv:2510.18313*, 2025.

[^14]: M. Liu, D. Zhang, J. Liu, J. Cui, H. Xie, G. Chen, H. Ye, M. Y. Yang, F. Nex, and H. Cheng, “Driveva: Video action models are zero-shot drivers,” *arXiv preprint arXiv:2604.04198*, 2026.

[^15]: T. Xia, Y. Li, L. Zhou, J. Yao, K. Xiong, H. Sun, B. Wang, K. Ma, G. Chen, H. Ye *et al.*, “Drivelaw: Unifying planning and video generation in a latent driving world,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2026, pp. 39 701–39 712.

[^16]: S. Yu, S. Kwak, H. Jang, J. Jeong, J. Huang, J. Shin, and S. Xie, “Representation alignment for generation: Training diffusion transformers is easier than you think,” *arXiv preprint arXiv:2410.06940*, 2024.

[^17]: D. Jiang, M. Wang, L. Li, L. Zhang, H. Wang, W. Wei, G. Dai, Y. Zhang, and J. Wang, “No other representation component is needed: Diffusion transformers can provide representation guidance by themselves,” *arXiv preprint arXiv:2505.02831*, 2025.

[^18]: H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom, “nuscenes: A multimodal dataset for autonomous driving,” in *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2020, pp. 11 621–11 631.

[^19]: D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone *et al.*, “Navsim: Data-driven non-reactive autonomous vehicle simulation and benchmarking,” *Advances in Neural Information Processing Systems*, vol. 37, pp. 28 706–28 719, 2024.

[^20]: Y. Wen, Y. Zhao, Y. Liu, F. Jia, Y. Wang, C. Luo, C. Zhang, T. Wang, X. Sun, and X. Zhang, “Panacea: Panoramic and controllable video generation for autonomous driving,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2024, pp. 6902–6912.

[^21]: L. Russell, A. Hu, L. Bertoni, G. Fedoseev, J. Shotton, E. Arani, and G. Corrado, “Gaia-2: A controllable multi-view generative world model for autonomous driving,” *arXiv preprint arXiv:2503.20523*, 2025.

[^22]: T. Brooks, B. Peebles, C. Holmes, W. DePue, Y. Guo, L. Jing, D. Schnurr, J. Taylor, T. Luhman, E. Luhman *et al.*, “Video generation models as world simulators,” *OpenAI Blog*, vol. 1, no. 8, p. 1, 2024.

[^23]: J. Bruce, M. D. Dennis, A. Edwards, J. Parker-Holder, Y. Shi, E. Hughes, M. Lai, A. Mavalankar, R. Steigerwald, C. Apps *et al.*, “Genie: Generative interactive environments,” in *Forty-first International Conference on Machine Learning*, 2024.

[^24]: M. Assran, A. Bardes, D. Fan, Q. Garrido, R. Howes, M. Muckley, A. Rizvi, C. Roberts, K. Sinha, A. Zholus *et al.*, “V-jepa 2: Self-supervised video models enable understanding, prediction and planning,” *arXiv preprint arXiv:2506.09985*, 2025.

[^25]: H. Wang, D. Liu, H. Xie, H. Liu, E. Ma, K. Yu, L. Wang, and B. Wang, “Mila: Multi-view intensive-fidelity long-term video generation world model for autonomous driving,” *arXiv preprint arXiv:2503.15875*, 2025.

[^26]: W. Zheng, W. Chen, Y. Huang, B. Zhang, Y. Duan, and J. Lu, “Occworld: Learning a 3d occupancy world model for autonomous driving,” in *European conference on computer vision*. Springer, 2024, pp. 55–72.

[^27]: L. Wang, W. Zheng, Y. Ren, H. Jiang, Z. Cui, H. Yu, and J. Lu, “Occsora: 4d occupancy generation models as world simulators for autonomous driving,” *arXiv preprint arXiv:2405.20337*, 2024.

[^28]: B. Li, J. Guo, H. Liu, Y. Zou, Y. Ding, X. Chen, H. Zhu, F. Tan, C. Zhang, T. Wang *et al.*, “Uniscene: Unified occupancy-centric driving scene generation,” in *Proceedings of the Computer Vision and Pattern Recognition Conference*, 2025, pp. 11 971–11 981.

[^29]: X. Guo, Z. Wu, K. Xiong, Z. Xu, L. Zhou, G. Xu, S. Xu, H. Sun, B. Wang, G. Chen *et al.*, “Genesis: Multimodal driving scene generation with spatio-temporal and cross-modal consistency,” *arXiv preprint arXiv:2506.07497*, 2025.

[^30]: H. Zhou, L. Lin, J. Wang, Y. Lu, D. Bai, B. Liu, Y. Wang, A. Geiger, and Y. Liao, “Hugsim: A real-time, photo-realistic and closed-loop simulator for autonomous driving,” *arXiv preprint arXiv:2412.01718*, 2024.

[^31]: H. Gao, S. Chen, B. Jiang, B. Liao, Y. Shi, X. Guo, Y. Pu, H. Yin, X. Li, X. Zhang *et al.*, “Rad: Training an end-to-end driving policy via large-scale 3dgs-based reinforcement learning,” *arXiv preprint arXiv:2502.13144*, 2025.

[^32]: J. Yang, K. Chitta, S. Gao, L. Chen, Y. Shao, X. Jia, H. Li, A. Geiger, X. Yue, and L. Chen, “Resim: Reliable world simulation for autonomous driving,” *arXiv preprint arXiv:2506.09981*, 2025.

[^33]: C. Ni, G. Zhao, X. Wang, Z. Zhu, W. Qin, X. Chen, G. Jia, G. Huang, and W. Mei, “Recondreamer-rl: Enhancing reinforcement learning via diffusion-based scene reconstruction,” *arXiv preprint arXiv:2508.08170*, 2025.

[^34]: F. Bartoccioni, E. Ramzi, V. Besnier, S. Venkataramanan, T.-H. Vu, Y. Xu, L. Chambon, S. Gidaris, S. Odabas, D. Hurych *et al.*, “Vavim and vavam: Autonomous driving through video generative modeling,” *arXiv preprint arXiv:2502.15672*, 2025.

[^35]: L. Wang, Y. Zheng, Q. Chen, S. Li, Y. Zhang, Z. Xing, Q. Zhang, X. Li, D. Qian, P. Yang *et al.*, “Latent-wam: Latent world action modeling for end-to-end autonomous driving,” *arXiv preprint arXiv:2603.24581*, 2026.

[^36]: Z. Zhao, T. Fu, Y. Wang, L. Wang, and H. Lu, “From forecasting to planning: Policy world model for collaborative state-action prediction,” *Advances in Neural Information Processing Systems*, vol. 38, pp. 134 585–134 611, 2025.

[^37]: Y. Zhou, X. Wang, H. Shao, L. Wang, G. Zhao, J. Shao, J. Zhu, T. Yu, Z. Zhu, G. Huang *et al.*, “Drivedreamer-policy: A geometry-grounded world-action model for unified generation and planning,” *arXiv preprint arXiv:2604.01765*, 2026.

[^38]: K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter *et al.*, “ $\pi_{0}$: A vision-language-action flow model for general robot control,” *arXiv preprint arXiv:2410.24164*, 2024.

[^39]: W. Liang, L. Yu, L. Luo, S. Iyer, N. Dong, C. Zhou, G. Ghosh, M. Lewis, W.-t. Yih, L. Zettlemoyer *et al.*, “Mixture-of-transformers: A sparse and scalable architecture for multi-modal foundation models,” *arXiv preprint arXiv:2411.04996*, 2024.

[^40]: S. Zeng, X. Chang, M. Xie, X. Liu, Y. Bai, Z. Pan, M. Xu, X. Wei, and N. Guo, “Futuresightdrive: Thinking visually with spatio-temporal cot for autonomous driving,” *arXiv preprint arXiv:2505.17685*, 2025.

[^41]: Y. Li, Y. Wang, Y. Liu, J. He, L. Fan, and Z. Zhang, “End-to-end driving with online trajectory evaluation via bev world model,” *arXiv preprint arXiv:2504.01941*, 2025.

[^42]: Y. Liao, P. Zhou, S. Huang, D. Yang, S. Chen, Y. Jiang, Y. Hu, J. Cai, S. Liu, J. Luo *et al.*, “Genie envisioner: A unified world foundation platform for robotic manipulation,” *arXiv preprint arXiv:2508.05635*, 2025.

[^43]: R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2022, pp. 10 684–10 695.

[^44]: D. P. Kingma and M. Welling, “Auto-encoding variational bayes,” *arXiv preprint arXiv:1312.6114*, 2013.

[^45]: B. Zheng, N. Ma, S. Tong, and S. Xie, “Diffusion transformers with representation autoencoders,” *arXiv preprint arXiv:2510.11690*, 2025.

[^46]: M. Shi, H. Wang, W. Zheng, Z. Yuan, X. Wu, X. Wang, P. Wan, J. Zhou, and J. Lu, “Latent diffusion model without variational autoencoder,” *arXiv preprint arXiv:2510.15301*, 2025.

[^47]: J. Yao, B. Yang, and X. Wang, “Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models,” in *Proceedings of the Computer Vision and Pattern Recognition Conference*, 2025, pp. 15 703–15 712.

[^48]: T. Bi, X. Zhang, Y. Lu, and N. Zheng, “Vision foundation models can be good tokenizers for latent diffusion models,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2026, pp. 43 310–43 319.

[^49]: B. Chen, S. Bi, H. Tan, H. Zhang, T. Zhang, Z. Li, Y. Xiong, J. Zhang, and K. Zhang, “Aligning visual foundation encoders to tokenizers for diffusion models,” in *The Fourteenth International Conference on Learning Representations*, 2026.

[^50]: Y. Gao, C. Chen, and J. Gu, “One layer is enough: Adapting pretrained visual encoders for image generation,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2026, pp. 4688–4697.

[^51]: W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in *Proceedings of the IEEE/CVF international conference on computer vision*, 2023, pp. 4195–4205.

[^52]: N. Ma, M. Goldstein, M. S. Albergo, N. M. Boffi, E. Vanden-Eijnden, and S. Xie, “Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers,” in *European Conference on Computer Vision*. Springer, 2024, pp. 23–40.

[^53]: X. Leng, J. Singh, Y. Hou, Z. Xing, S. Xie, and L. Zheng, “Repa-e: Unlocking vae for end-to-end tuning of latent diffusion transformers,” in *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 2025, pp. 18 262–18 272.

[^54]: Y. Tian, H. Chen, M. Zheng, Y. Liang, C. Xu, and Y. Wang, “U-repa: Aligning diffusion u-nets to vits,” *Advances in Neural Information Processing Systems*, vol. 38, pp. 11 003–11 024, 2025.

[^55]: J. Singh, X. Leng, Z. Wu, L. Zheng, R. Zhang, E. Shechtman, and S. Xie, “What matters for representation alignment: Global information or spatial structure?” *arXiv preprint arXiv:2512.10794*, 2025.

[^56]: M. Yang, Z. Tan, B. Li, X. Yang, H. Chen, and H. Li, “Diversedit: Towards diverse representation learning in diffusion transformers,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2026, pp. 40 591–40 601.

[^57]: T. Kouzelis, E. Karypidis, I. Kakogeorgiou, S. Gidaris, and N. Komodakis, “Boosting generative image modeling via joint image-feature synthesis,” *Advances in Neural Information Processing Systems*, vol. 38, pp. 16 685–16 714, 2025.

[^58]: Y. Pan, R. Feng, Q. Dai, Y. Wang, W. Lin, M. Guo, C. Luo, and N. Zheng, “Semantics lead the way: Harmonizing semantic and texture modeling with asynchronous latent diffusion,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2026, pp. 43 664–43 674.

[^59]: G. Wu, S. Zhang, R. Shi, S. Gao, Z. Chen, L. Wang, Z. Chen, H. Gao, Y. Tang, M.-M. Cheng *et al.*, “Representation entanglement for generation: Training diffusion transformers is much easier than you think,” *Advances in Neural Information Processing Systems*, vol. 38, pp. 7714–7743, 2025.

[^60]: A. Baade, E. R. Chan, K. Sargent, C. Chen, J. Johnson, E. Adeli, and L. Fei-Fei, “Latent forcing: Reordering the diffusion trajectory for pixel-space image generation,” *arXiv preprint arXiv:2602.11401*, 2026.

[^61]: Y. HaCohen, N. Chiprut, B. Brazowski, D. Shalem, D. Moshe, E. Richardson, E. Levin, G. Shiran, N. Zabari, O. Gordon *et al.*, “Ltx-video: Realtime video latent diffusion,” *arXiv preprint arXiv:2501.00103*, 2024.

[^62]: Y. Zheng, R. Liang, K. Zheng, J. Zheng, L. Mao, J. Li, W. Gu, R. Ai, S. E. Li, X. Zhan *et al.*, “Diffusion-based planning for autonomous driving with flexible guidance,” *arXiv preprint arXiv:2501.15564*, 2025.

[^63]: B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang *et al.*, “Diffusiondrive: Truncated diffusion model for end-to-end autonomous driving,” in *Proceedings of the Computer Vision and Pattern Recognition Conference*, 2025, pp. 12 037–12 047.

[^64]: Y. Li, K. Xiong, X. Guo, F. Li, S. Yan, G. Xu, L. Zhou, L. Chen, H. Sun, B. Wang *et al.*, “Recogdrive: A reinforced cognitive framework for end-to-end autonomous driving,” *arXiv preprint arXiv:2506.08052*, 2025.

[^65]: X. Liu, C. Gong, and Q. Liu, “Flow straight and fast: Learning to generate and transfer data with rectified flow,” *arXiv preprint arXiv:2209.03003*, 2022.

[^66]: Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” *arXiv preprint arXiv:2210.02747*, 2022.

[^67]: N. Agarwal, A. Ali, M. Bala, Y. Balaji, E. Barker, T. Cai, P. Chattopadhyay, Y. Chen, Y. Cui, Y. Ding *et al.*, “Cosmos world foundation model platform for physical ai,” *arXiv preprint arXiv:2501.03575*, 2025.

[^68]: X. Zhou, Q. Li, X. Hu, H. Chen, and S. Gu, “Guiding a diffusion transformer with the internal dynamics of itself,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2026, pp. 11 536–11 545.

[^69]: J. Wang, Z. Hua, X. Liu, Z. Xing, H. Tian, K. Ma, H. Ye, G. Chen, L. Chen, and Q. Zhang, “Beyond imitation: Learning safe end-to-end autonomous driving from hard negatives,” *arXiv preprint arXiv:2605.19771*, 2026.

[^70]: H. Caesar, J. Kabzan, K. S. Tan, W. K. Fong, E. Wolff, A. Lang, L. Fletcher, O. Beijbom, and S. Omari, “nuplan: A closed-loop ml-based planning benchmark for autonomous vehicles,” *arXiv preprint arXiv:2106.11810*, 2021.

[^71]: O. Contributors, “Openscene: The largest up-to-date 3d occupancy prediction benchmark in autonomous driving,” in *Proceedings of the Conference on Computer Vision and Pattern Recognition, Vancouver, Canada*, 2023, pp. 18–22.

[^72]: M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, “Gans trained by a two time-scale update rule converge to a local nash equilibrium,” *Advances in neural information processing systems*, vol. 30, 2017.

[^73]: T. Unterthiner, S. Van Steenkiste, K. Kurach, R. Marinier, M. Michalski, and S. Gelly, “Towards accurate generative models of video: A new metric & challenges,” *arXiv preprint arXiv:1812.01717*, 2018.

[^74]: C. Min, D. Zhao, L. Xiao, J. Zhao, X. Xu, Z. Zhu, L. Jin, J. Li, Y. Guo, J. Xing *et al.*, “Driveworld: 4d pre-trained scene understanding via world models for autonomous driving,” in *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2024, pp. 15 522–15 533.

[^75]: S. W. Kim, J. Philion, A. Torralba, and S. Fidler, “Drivegan: Towards a controllable high-quality neural simulation,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2021, pp. 5820–5829.

[^76]: X. Gui, M. Zhang, T. Yan, W. Han, J. Gong, F. Tan, C.-z. Xu, and J. Shen, “Bridging scene generation and planning: Driving with world model via unifying vision and motion representation,” *arXiv preprint arXiv:2603.14948*, 2026.

[^77]: S. Chen, B. Jiang, H. Gao, B. Liao, Q. Xu, Q. Zhang, C. Huang, W. Liu, and X. Wang, “Vadv2: End-to-end vectorized autonomous driving via probabilistic planning,” *arXiv preprint arXiv:2402.13243*, 2024.

[^78]: Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang *et al.*, “Planning-oriented autonomous driving,” in *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2023, pp. 17 853–17 862.

[^79]: K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger, “Transfuser: Imitation with transformer-based sensor fusion for autonomous driving,” *IEEE transactions on pattern analysis and machine intelligence*, vol. 45, no. 11, pp. 12 878–12 895, 2022.

[^80]: X. Weng, B. Ivanovic, Y. Wang, Y. Wang, and M. Pavone, “Para-drive: Parallelized architecture for real-time autonomous driving,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2024, pp. 15 449–15 458.

[^81]: M. Wang, D. Jiang, L. Li, Y. Lin, G. Shen, X. Kong, Y. Liu, G. Dai, and J. Wang, “Sra 2: Variational autoencoder self-representation alignment for efficient diffusion training,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2026, pp. 32 978–32 987.

[^82]: H. Chefer, P. Esser, D. Lorenz, D. Podell, V. Raja, V. Tong, A. Torralba, and R. Rombach, “Self-supervised flow matching for scalable multi-modal synthesis,” *arXiv preprint arXiv:2603.06507*, 2026.

[^83]: M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby *et al.*, “Dinov2: Learning robust visual features without supervision,” *arXiv preprint arXiv:2304.07193*, 2023.

[^84]: L. Wang, B. Huang, Z. Zhao, Z. Tong, Y. He, Y. Wang, Y. Wang, and Y. Qiao, “Videomae v2: Scaling video masked autoencoders with dual masking,” in *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2023, pp. 14 549–14 560.

[^85]: H. Lin, S. Chen, J. Liew, D. Y. Chen, Z. Li, G. Shi, J. Feng, and B. Kang, “Depth anything 3: Recovering the visual space from any views,” *arXiv preprint arXiv:2511.10647*, 2025.
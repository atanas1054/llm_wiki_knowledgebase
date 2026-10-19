---
title: "ChainFlow-VLA: Causal Flow Planning with Vision-Language Models"
source: "https://arxiv.org/html/2605.23270v1"
author:
published:
created: 2026-10-19
description:
tags:
  - "clippings"
---
Xiyang Wang   Xinlin Wang <sup>1</sup>   Tingguang Zhou <sup>1</sup>   Gong Chen <sup>1</sup> Affiliation: Afari Intelligent Drive    Xingtai Gui   Zhi Xu   Xiaolei Wu Affiliation: Afari Intelligent Drive    Feiyang Tan   Hangning Zhou   Mu Yang Affiliation: Afari Intelligent Drive Affiliation: Tianjin University Affiliation: University of Macau

###### Abstract

Current end-to-end autonomous driving systems are fundamentally limited by a mismatch between temporal causal reasoning and global trajectory consistency. Autoregressive (AR) models capture interaction-aware temporal dependencies via causal factorization, but their step-wise decoding leads to error accumulation and suboptimal global structure. In contrast, diffusion models optimize trajectories globally but lack explicit causal constraints, making them unreliable in interactive and safety-critical scenarios. This dichotomy reveals a deeper issue: existing methods treat causal modeling and global optimization as separate paradigms, without a principled way to unify them within a single trajectory distribution. To address this, we propose ChainFlow-VLA, which unifies causal generation and global refinement within a unified probabilistic framework. We formulate planning as a mixture over AR-induced modes and learn Vision-Language Model (VLM)-conditioned residual distributions over these modes. An autoregressive generator (Chain) produces a discrete set of causal trajectory modes, followed by a diffusion-based refiner (Flow) that leverages VLM hidden states as semantic priors to perform mode-conditioned correction in residual space while preserving causal structure. This straightforward conditioning seamlessly injects high-level scene understanding into fine-grained trajectory adjustments. Experiments demonstrate that ChainFlow-VLA achieves robust planning in ambiguous and long-tail scenarios, achieving a state-of-the-art score of 94.85 on the NAVSIM v1 leaderboard, matching human-level performance (94.8). Code: [https://github.com/AFARI-Research/ChainFlow-VLA](https://github.com/AFARI-Research/ChainFlow-VLA).

## 1 Introduction

End-to-end autonomous driving has emerged as a promising paradigm [^10] [^15] for unified perception and planning by directly learning a mapping from sensor inputs to future trajectories. While these models can generate smooth and executable trajectories in routine scenarios [^2] [^27], real-world driving still presents complex interactions, long-tail events, and distribution shifts [^1]. Addressing such challenges requires not only geometric and motion cues, but also higher-level reasoning [^19] [^6] over scene semantics, agent intent, and implicit traffic rules. Stronger semantic understanding and reasoning are therefore essential for robust end-to-end driving [^26] [^12].

![[p1.png|Refer to caption]]

Figure 1: Comparison of different paradigms for integrating VLM into end-to-end autonomous driving. (a) VLM-guided pipeline that predicts high-level guidance to steer an end-to-end model, which introduces an information bottleneck and limits fine-grained trajectory refinement. (b) Feature-level fusion that combines VLM and perception backbones via a fusion module followed by an action expert, but lacks a principled mechanism to enforce consistency between local dynamics and global trajectory structure. (c) Ours (ChainFlow-VLA) formulates trajectory prediction as a unified causal–flow process, where an AR generator produces temporally consistent proposals that are refined by a diffusion model in the residual space. Fine-tuned VLM representations are injected as semantic flow conditioning to guide global trajectory refinement, enabling tight coupling between causal reasoning, global optimization, and high-level semantics.

Recent studies have attempted to incorporate VLM to enhance the semantic understanding and reasoning capabilities of end-to-end autonomous driving systems. As illustrated in Figure 1(a), one category of methods [^5] [^35] utilizes a VLM to predict high-level features, which are subsequently processed by a downstream action expert model to generate the final trajectory. Although intuitive, this paradigm compresses rich scene semantics into discrete signals, limiting fine-grained trajectory optimization. Another category of methods [^29] [^21] [^17], shown in Figure 1(b), attempts to fuse VLM features with existing end-to-end driving representations to obtain more robust features, which are then decoded into trajectories via a unified action expert model. However, treating semantic reasoning and physical trajectory generation as loosely coupled components makes it difficult for semantic information to exert a direct impact during the planning stage, where error correction is most critical.

Our analysis reveals that existing methods conflate two fundamental yet insufficiently addressed questions. First, most action experts generate trajectories from high-dimensional features using a single autoregressive or diffusion paradigm [^33]. Although some works [^30] [^20] combine these approaches, they struggle to maintain consistency between local dynamics and global trajectory structure (see Section 3). Second, existing methods typically integrate VLM at early feature fusion stages [^14] [^5], assuming semantic information should be injected as early as possible. However, strong end-to-end models already exhibit robust trajectory generation capabilities [^1]. The main challenge lies not in generating trajectories from scratch, but in refining them in long-tail scenarios [^9] to satisfy semantic constraints. We therefore argue that VLM should not serve as a direct trajectory generator, but rather as provider of semantic constraints at critical stages of refinement.

Based on these insights, we propose ChainFlow-VLA (Figure 1(c)), which models trajectory generation as a unified causal generation–global refinement process rather than loosely coupled modules. An autoregressive model first generates trajectory modes, capturing temporal causal structure. Conditioned on these priors, a diffusion refiner guided by VLM representations performs residual refinement. This reformulation shifts trajectory modeling from absolute generation to semantic correction, focusing on how prior trajectories should be adjusted under environmental context. It mitigates error accumulation and local optima in autoregressive decoding while preserving global consistency. Despite its simplicity, this design aligns closely with the structure of end-to-end driving. On the NAVSIM v1 benchmark [^3], ChainFlow-VLA achieves a score of 94.85, surpassing prior methods and reaching human-level performance. These results highlight the importance of unifying causal modeling, global optimization, and semantic reasoning for robust autonomous driving.

We summarize our contributions as follows:

- We propose ChainFlow-VLA, a unified framework that casts trajectory generation as a probabilistic mixture over AR-induced modes, decomposed into a causal autoregressive Chain and a residual diffusion Flow, unifying temporal reasoning and global geometric consistency within a single formulation.
- We reformulate VLM guidance as mode-conditioned semantic control over residual refinement, where VLM representations are injected to modulate local trajectory corrections rather than global trajectory generation.
- Extensive experiments on NAVSIM v1 demonstrate that ChainFlow-VLA achieves state-of-the-art performance and reaches human-level results. To the best of our knowledge, it is among the first methods to achieve this level of performance on the benchmark.

## 2 Related Work

End-to-end Autonomous Driving.

End-to-end autonomous driving learns a direct mapping from sensor inputs to future trajectories or control commands [^1]. Discriminative planners, such as UniAD [^10] and VAD [^15], integrate perception and planning efficiently, but their deterministic regression paradigm limits behavioral diversity. Autoregressive planners [^13] capture temporal causality through sequential prediction, yet may suffer from error accumulation and weak global optimization. Diffusion-based methods, such as DiffusionDrive [^22], improve multimodal generation via iterative denoising, but can struggle with stable and physically consistent long-horizon planning. These limitations motivate our Chain–Flow design, which combines causal AR proposal generation with diffusion-based global refinement.

Vision-Language-Action Models for Driving.

Vision-language models have been introduced into autonomous driving for their semantic understanding and reasoning ability [^26] [^12]. Direct VLA planners [^35] [^28] [^36] map visual-language representations to driving actions or trajectories, but continuous trajectory generation remains challenging for VLMs due to limited fine-grained spatial precision. Other methods use VLMs as high-level reasoners or feature providers [^32] [^21] [^5] [^11], where semantic information is often injected before the final planning refinement stage. In contrast, ChainFlow-VLA uses VLM hidden states to guide residual diffusion over AR proposals, allowing semantic reasoning to directly modulate trajectory correction.

## 3 Preliminaries

Task Definition. We consider end-to-end trajectory planning as modeling a multi-modal conditional distribution:

$$
P(Y\mid\mathcal{O}),
$$

where $\mathcal{O}$ denotes observations from multiple modalities and $Y=\{y_{t}\}_{t=1}^{T}$ is the future trajectory.

From Global Distribution to Conditional Decomposition. The trajectory distribution $P(Y\mid\mathcal{O})$ is inherently multi-modal, making direct modeling challenging due to its highly entangled structure.

Autoregressive and diffusion-based models provide complementary inductive biases. AR models provide a causal factorization of the trajectory distribution:

$$
P(Y_{\mathrm{AR}}\mid\mathcal{O})=\prod_{t}P(y_{t}\mid y_{<t},\mathcal{O}),
$$

whereas diffusion models capture global structure via iterative denoising.

We bridge these two paradigms through a conditional decomposition. An AR model produces a set of trajectory proposals $\{Y_{\mathrm{AR}}^{(k)}\}_{k=1}^{K}$, where each $Y_{\mathrm{AR}}^{(k)}$ denotes the $k$ -th trajectory mode. Conditioned on each proposal, the problem reduces to modeling a local conditional distribution:

$$
P(Y\mid Y_{\mathrm{AR}}^{(k)},\mathcal{O}),
$$

We parameterize this conditional distribution using a representation $h_{\text{VLM}}$ extracted from a vision-language model, which encodes semantic context from observations $\mathcal{O}$:

$$
P(Y\mid Y_{\text{AR}}^{(k)},\mathcal{O})\;\approx\;P(Y\mid Y_{\text{AR}}^{(k)},h_{\text{VLM}}),
$$

where $h_{\text{VLM}}$ modulates the local conditional distribution for each trajectory mode.

This yields an implicit mixture formulation, inspired by the law of total probability:

$$
P(Y\mid\mathcal{O})\approx\sum_{k=1}^{K}P(Y\mid Y_{\text{AR}}^{(k)},h_{\text{VLM}})\cdot P(Y_{\text{AR}}^{(k)}\mid\mathcal{O}),
$$

where each component corresponds to a local distribution centered around a trajectory mode, which is later instantiated in residual space for efficient learning.

## 4 Methods

### 4.1 Overview

Building on the above formulation, the ChainFlow-VLA framework (Figure 2) realizes planning as a two-stage sequential refinement process that instantiates the factorization in Eq. (5). Initially, the Autoregressive Trajectory Generation module processes driving features to produce a set of $K$ trajectory proposals, ensuring causal consistency and physical feasibility. These proposals serve as initial modes for the subsequent VLM-Guided Residual Diffusion stage. In this stage, a Diffusion Transformer (DiT) models the residual distribution conditioned on VLM hidden states, providing fine-grained semantic guidance to refine the AR-induced modes into final trajectories.

![[overview 2.png|Refer to caption]]

Figure 2: ChainFlow-VLA framework. The model first performs Autoregressive Trajectory Generation (Chain) to produce K causal proposals, which are then refined via VLM-Guided Residual Diffusion (Flow). By learning the residuals between AR proposals and ground-truth trajectories, the model unifies causal rollout with VLM-based semantic guidance, formulating planning as a mixture of VLM-conditioned residual distributions over AR-induced modes.

### 4.2 Chain: Autoregressive Trajectory Generation

As illustrated in Figure 2, the autoregressive module takes BEV-style driving features and learnable trajectory queries as inputs, and iteratively generates future states through a recurrent decoding process.

We follow the autoregressive factorization introduced in section 3, where trajectory generation is modeled as a sequential conditional process:

$$
p(y_{t}\mid y_{<t},\mathcal{O}),
$$

which introduces a strong causal inductive bias, ensuring temporally consistent and physically plausible rollouts.

In practice, each conditional term is implicitly parameterized by a deterministic predictor.

To capture multi-modality, we maintain a set of $K$ parallel trajectory hypotheses. Each trajectory $Y_{\mathrm{AR}}^{(k)}=\{y_{t}^{(k)}\}_{t=1}^{T}$ represents a distinct kinematic mode, yielding a discrete approximation of the global trajectory distribution.

At each step $t$, the model predicts control variables $(a_{t}^{(k)},\omega_{t}^{(k)})$ conditioned on the previous state and scene context:

$$
(a_{t}^{(k)},\omega_{t}^{(k)})=H_{\theta}(y_{<t}^{(k)},\mathcal{O}),
$$

where $H_{\theta}$ denotes a learnable predictor that parameterizes the conditional prediction of control variables. The next state is obtained through a kinematic transition:

$$
y_{t}^{(k)}=\mathrm{Bicycle}(y_{t-1}^{(k)},a_{t}^{(k)},\omega_{t}^{(k)}),
$$

where $\mathrm{Bicycle}(\cdot)$ denotes a standard bicycle kinematic model, which enforces physical feasibility and stabilizes long-horizon prediction.

Scene observations are encoded into latent tokens and queried at each step to provide environment-aware context, while the autoregressive hidden state propagates motion intent over time.

After $T$ steps, the model produces a set of trajectory proposals:

$$
Y_{\mathrm{AR}}=\{Y_{\mathrm{AR}}^{(k)}\}_{k=1}^{K}.
$$

From a modeling perspective, this stage performs a causal discretization of the global trajectory distribution, providing structured initialization for subsequent flow-based refinement.

### 4.3 Flow: VLM-Guided Residual Diffusion

Following Eq. (4), the Flow module instantiates the local conditional term $P(Y\mid Y_{\mathrm{AR}}^{(k)},h_{\mathrm{VLM}})$ in Eq. (5). Rather than modeling the full trajectory distribution in the global space, we refine each AR proposal in a local residual space [^34]. This turns trajectory generation into proposal-centered correction guided by VLM.

AR-Conditioned Residual Modes. We leverage the trajectory proposals generated by the preceding AR module as mode-specific proposals for residual refinement. The Flow module does not re-estimate the AR proposal distribution $P(Y_{\mathrm{AR}}^{(k)}\mid\mathcal{O})$ in Eq. (5). Instead, for AR-conditioned modes, it learns a residual distribution that corrects the proposal toward the expert trajectory. The refined trajectory is represented as:

$$
Y=Y_{\mathrm{AR}}^{(k)}+\Delta Y_{k},
$$

where $\Delta Y_{k}$ denotes the correction relative to the $k$ -th AR proposal. Accordingly, the local conditional distribution is instantiated in residual space:

$$
P\left(Y\mid Y_{\mathrm{AR}}^{(k)},h_{\mathrm{VLM}}\right)=P\left(\Delta Y_{k}\mid Y_{\mathrm{AR}}^{(k)},h_{\mathrm{VLM}}\right).
$$

This converts each local component in Eq. (5) into a proposal-conditioned residual refinement problem.

VLM-Guided Conditional Diffusion. Residual refinement requires determining how each AR proposal should be corrected under the current scene. Such correction depends not only on geometric deviation, but also on semantic understanding, such as route intention, traffic context, and trajectory-level feasibility. Following ReCogDrive [^21], we adopt a driving-oriented VLM supervised fine-tuned on environment-understanding and trajectory-QA tasks. Without further optimizing the VLM under the diffusion objective, we directly use its hidden states $h_{\mathrm{VLM}}$ as semantic conditions for the residual diffusion model. This enables the Flow module to transfer the VLM’s general driving knowledge into proposal correction without task-specific VLM adaptation.

Formally, given the expert trajectory $Y^{*}$ and the $k$ -th AR proposal, the residual target is

$$
\Delta Y_{k}^{*}=Y^{*}-Y_{\mathrm{AR}}^{(k)}.
$$

We then construct noisy residual samples by

$$
\mathbf{z}_{t}^{(k)}=\sqrt{\bar{\alpha}_{t}}\Delta Y_{k}^{*}+\sqrt{1-\bar{\alpha}_{t}}\boldsymbol{\epsilon},
$$

where $\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$. The diffusion model predicts the injected noise conditioned on the timestep $t$, ego state $c_{\mathrm{ego}}$, VLM hidden states $h_{\mathrm{VLM}}$, and the AR proposal $Y_{\mathrm{AR}}^{(k)}$:

$$
\hat{\boldsymbol{\epsilon}}^{(k)}=\boldsymbol{\epsilon}_{\theta}\left(\mathbf{z}_{t}^{(k)},t,c_{\mathrm{ego}},h_{\mathrm{VLM}},Y_{\mathrm{AR}}^{(k)}\right).
$$

Architecture and Inference. As shown in the DiT block of Figure 2, our residual refiner follows the general architecture of DiT [^24]. Noisy residual tokens are processed by stacked transformer blocks, where conditions are injected through adaptive LayerNorm. In addition, full VLM hidden states are incorporated via cross-attention, allowing high-level semantic information to guide the residual denoising process.

At inference time, we sample a residual $\hat{\Delta Y}_{k}$ for each AR proposal using the DDIM process and reconstruct the refined trajectory as

$$
\hat{Y}_{k}=Y_{\mathrm{AR}}^{(k)}+\hat{\Delta Y}_{k}.
$$

Overall, the Flow module implements each local term in Eq. (5) as VLM-guided residual refinement around an AR proposal.

### 4.4 Scorer

We employ a scoring head [^8] to evaluate each candidate trajectory, producing a set of utility scores. The scorer acts as a proxy utility function, defining a decision rule over the learned trajectory distribution. The final trajectory is selected by aggregating these scores and choosing the highest-scoring candidate.

### 4.5 Training Objectives and Target Assignment

ChainFlow-VLA is trained in two stages, both leveraging trajectory and scorer supervision, with Stage II further introducing diffusion-based refinement.

Stage I. We train the AR module using WTA-based supervision, following [^16]:

$$
\mathcal{L}_{\mathrm{stage1}}=\mathcal{L}_{\text{traj}}+\lambda_{1}\mathcal{L}_{\text{scorer}},
$$

where the trajectory loss selects the closest mode to the expert trajectory.

Stage II. we train the diffusion refiner and scorer:

$$
\mathcal{L}_{\mathrm{stage2}}=\lambda_{2}\mathcal{L}_{\text{diff}}+\lambda_{3}\mathcal{L}_{\text{traj}}+\lambda_{4}\mathcal{L}_{\text{scorer}}.
$$

We adopt an asymmetric WTA assignment in Stage II. For diffusion supervision, the expert trajectory is matched to the closest AR proposal:

$$
k^{*}=\arg\min_{k}\left\|Y_{\mathrm{AR}}^{(k)}-Y^{*}\right\|_{2}.
$$

The diffusion objective is then computed within this selected AR-conditioned mode:

$$
\mathcal{L}_{\text{diff}}=\left\|\boldsymbol{\epsilon}-\boldsymbol{\epsilon}_{\theta}\right\|_{2}^{2}.
$$

This design separates mode selection from residual refinement, enabling the diffusion objective to focus on local correction around AR proposals. Meanwhile, trajectory supervision $\mathcal{L}_{\text{traj}}$ is applied to the refined outputs. This output-level supervision provides a direct optimization signal in trajectory space, accelerating convergence and stabilizing refinement training.

Table 1: Comparison of results on the NAVSIM benchmark. All metrics are higher-is-better. Best results are highlighted in bold.

Methods Venue PDMS $\uparrow$ NC $\uparrow$ DAC $\uparrow$ EP $\uparrow$ TTC $\uparrow$ Comf.$\uparrow$ End-to-End Methods UniAD [^10] CVPR’23 83.4 97.8 91.9 78.8 92.9 100.0 VADv2 [^15] arXiv’24 80.9 97.2 89.1 76.0 91.6 100.0 iPad [^8] arXiv’25 91.7 98.6 98.3 88.0 94.9 100.0 DiffusionDrive [^22] CVPR’25 88.1 98.2 96.2 82.2 94.7 100.0 Hydra-MDP++ [^18] arXiv’25 91.0 98.6 98.6 85.7 95.1 100.0 Centaur [^25] arXiv’25 92.6 99.5 98.9 85.9 98.0 100.0 TrajDiff <sub>(train)</sub> [^7] arXiv’25 86.4 98.0 95.0 80.8 93.7 100.0 TrajDiff <sub>(trainval)</sub> arXiv’25 88.5 98.1 97.0 82.7 94.3 100.0 DriveSuprim [^31] AAAI’26 93.5 98.6 98.6 91.3 95.5 100.0 RAP-DINO <sup>†</sup> [^4] ICLR’26 93.8 99.1 98.9 90.3 96.7 100.0 DrivoR <sub>(train)</sub> [^16] CVPR’26 93.1 98.9 98.3 89.1 96.2 100.0 DrivoR <sub>(trainval)</sub> CVPR’26 93.7 99.0 98.9 90.0 96.7 100.0 VLA-based Methods UniVLA [^28] RSS’25 81.7 96.9 91.1 76.8 91.7 96.7 AutoVLA [^36] NeurIPS’25 89.1 98.4 95.6 81.9 98.0 99.9 FSDrive [^32] NeurIPS’25 85.1 98.2 93.8 80.1 93.3 99.9 DriveVLA-W0 [^20] ICLR’26 90.2 98.7 99.1 83.3 95.3 99.3 ReCogDrive [^21] ICLR’26 90.8 97.9 97.3 87.3 94.9 100.0 SGDrive [^17] CVPR’26 91.1 98.6 97.8 85.8 96.2 100.0 SpanVLA [^37] arXiv’26 90.3 99.1 97.1 86.3 95.2 100.0 LatentVLA [^29] arXiv’26 92.4 98.9 98.2 88.2 96.0 100.0 Baselines & Oracle Human Driver [^3] NeurIPS’24 94.8 100.0 100.0 87.5 100.0 99.9 ChainFlow-VLA <sub>(train)</sub> – 93.6 98.8 98.6 90.8 96.1 100.0 ChainFlow-VLA <sub>(trainval)</sub> – 94.8 99.2 99.0 91.9 97.2 99.9

<sup>†</sup> Note: RAP-DINO is pre-trained on a private dataset that is 10 $\times$ larger than the default navtrain set.

## 5 Experiments

### 5.1 Experimental Setup

Datasets. We evaluate our method on NAVSIM v1 [^3], a large-scale benchmark for vision-based autonomous driving that combines real-world driving data with a non-reactive simulation protocol for scalable evaluation. To provide a comprehensive assessment, we compare two training configurations: one trained on the navtrain split and the other on the combined trainval split. This setup allows us to analyze the impact of training data scale on planning performance.

Implementation Details. We train our model on 8 NVIDIA A800 GPUs through a two-stage pipeline. In Stage 1, we fine-tune the image encoder with LoRA and train the Chain module for 25 epochs. In Stage 2, we train the Flow module conditioned on the VLM hidden features for 40 epochs. Following ReCogDrive [^21], we use the 2B VLM fine-tuned from InternVL as the driving-oriented VLM. Throughout both stages, we employ the AdamW optimizer [^23] with a per-GPU batch size of 8. The base learning rate is set to $2\times 10^{-4}$ and scaled according to $\sqrt{B/64}$, using a linear warmup for the first 10% of steps followed by a cosine decay schedule. The loss weights $\lambda_{1},\lambda_{2},\lambda_{3},\lambda_{4}$ are set to 1, 10, 20, and 4, respectively. During inference, we use a 4-step denoising process.

![[3_methods_5x4.png|Refer to caption]]

Figure 3: Qualitative comparison of trajectory predictions on representative NAVSIM scenarios. GT trajectories are shown in green. Predicted trajectories from ReCogDrive, DrivoR, and ChainFlow-VLA are visualized in red, purple, and orange, respectively.

Table 2: Main component ablation. ID 0 is the DrivoR baseline. Without VLM guidance, the DiT refiner uses the default DrivoR scene tokens as conditioning.

<table><thead><tr><th rowspan="2">ID</th><th>AR Gen.</th><th>DiT Refiner</th><th>VLM Guidance</th><th colspan="6">Metrics</th></tr><tr><th>(Chain)</th><th>(Flow)</th><th>(VLA)</th><th>PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>Comf.<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th></tr></thead><tbody><tr><th>0</th><td>✗</td><td>✗</td><td>✗</td><td>93.7</td><td>99.0</td><td>98.9</td><td>90.0</td><td>96.7</td><td>100.0</td></tr><tr><th>1</th><td>✓</td><td>✗</td><td>✗</td><td>94.0 (<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math> 0.3)</td><td>99.1</td><td>98.9</td><td>90.8</td><td>96.7</td><td>99.9</td></tr><tr><th>2</th><td>✓</td><td>✓</td><td>✗</td><td>94.1 (<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math> 0.4)</td><td>99.1</td><td>98.9</td><td>91.0</td><td>96.7</td><td>99.9</td></tr><tr><th>3</th><td>✓</td><td>✓</td><td>✓</td><td>94.8 (<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math> 1.1)</td><td>99.2</td><td>99.0</td><td>91.9</td><td>97.2</td><td>99.9</td></tr></tbody></table>

Table 3: Ablation on DiT design choices under the trainval split.

| Modeling Target | PDMS $\uparrow$ |
| --- | --- |
| Trajectory space | 92.89 |
| Residual space | 94.72 |

   

| DiT Blocks | PDMS $\uparrow$ |
| --- | --- |
| 8 | 94.64 |
| 12 | 94.72 |

   

| VLM Guidance Source | PDMS $\uparrow$ |
| --- | --- |
| Action QA | 94.11 |
| Env. & Traj. QA | 94.72 |

(a) DiT trajectory formulation   (b) Number of DiT blocks   (c) VLM guidance source

Table 4: Ablation on denoising steps under the trainval split.

| $N_{\text{step}}$ | 2 | 4 | 8 | 12 | 16 |
| --- | --- | --- | --- | --- | --- |
| PDMS $\uparrow$ | 94.68 | 94.72 | 94.74 | 94.85 | 94.67 |

Table 5: Generalization of ChainFlow across different end-to-end planning paradigms.

| Method | Modes | PDMS | NC | DAC | EP | TTC | Comf. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DiffusionDrive | 20 | 88.1 | 98.2 | 96.2 | 82.2 | 94.7 | 100.0 |
| \+ ChainFlow | 6 | 88.9 ($\uparrow$ 0.8) | 98.3 | 96.8 | 83.0 | 95.0 | 99.9 |
| iPad | 64 | 91.7 | 98.6 | 98.3 | 88.0 | 94.9 | 100.0 |
| \+ ChainFlow | 64 | 92.7 ($\uparrow$ 1.0) | 99.0 | 98.7 | 88.4 | 96.1 | 99.9 |

### 5.2 Main Results

As shown in Table 1, ChainFlow-VLA achieves a new state-of-the-art PDMS of 94.8, significantly outperforming prior end-to-end (93.8) and VLA-based (92.4) models. Remarkably, our approach reaches human-level performance, matching expert trajectory scores on the benchmark. Compared to feature-fusion methods like LatentVLA, which merge VLM and perception features, our architecture uses VLM semantics specifically for flow-based residual refinement. This suggests that simple high-level fusion is insufficient. These results validate our causal-to-global paradigm as an effective bridge between semantic reasoning and geometric precision.

### 5.3 Qualitative Results

We evaluate the qualitative performance of our method against alternative baselines on representative navtest scenarios, as illustrated in Fig. 3. In roundabout and left-turn ramp scenarios (Rows 1–2), while ReCogDrive and DrivoR either deviate from the drivable area or drift into incorrect lanes, our ChainFlow-VLA strictly adheres to navigation routes with collision-free maneuvers. For sharp turns (Row 3), it generates smooth, safe trajectories that closely match the expert trajectory, whereas both baselines fail by running off-road. In the intersection right-turn case (Row 4), our approach successfully bypasses static roadside vehicles to achieve higher ego progress than the expert trajectory without tailgating, while competing methods result in collisions. Furthermore, ChainFlow-VLA demonstrates robust safety by dynamically avoiding a static road barrier on the right side (Row 5)—a scenario where both baselines fail. Together, these results highlight the proposed model’s superior scene understanding and robustness across diverse, challenging environments.

### 5.4 Ablation Study

We conduct ablations to evaluate the contribution of each component in ChainFlow-VLA, including the AR trajectory generator, the residual DiT refiner, VLM guidance, and several key design choices.

Component analysis. Table 2 shows that each component brings consistent improvement. The AR generator improves DrivoR from 93.7 to 94.0 PDMS, and the residual DiT refiner further improves the score to 94.1. With VLM hidden-state guidance, the full model achieves 94.8 PDMS. The largest gain comes from EP, increasing from 90.0 to 91.9, indicating significantly improved efficiency while maintaining strong safety performance through enhanced environment understanding.

DiT design choices. Table 3 validates several refiner designs under the default 4-step denoising setting. Residual-space modeling outperforms direct trajectory-space prediction, confirming the benefit of refining AR proposals instead of generating trajectories from scratch. Increasing the DiT depth from 8 to 12 blocks brings a modest gain. For VLM guidance, environment- and trajectory-level QA SFT provides more useful hidden states than action-only QA, suggesting that scene and trajectory reasoning better supports residual refinement.

Denoising steps. Table 4 studies the number of denoising steps at inference. Increasing $N_{\text{step}}$ from 2 to 12 improves PDMS from 94.68 to 94.85, where the 12-step result already reached the same PDMS obtained by evaluating the human trajectory. However, we use $N_{\text{step}}=4$ as the default setting to balance performance and inference efficiency.

![[vlm_bev_5_3.png|Refer to caption]]

Figure 4: Qualitative comparison between BEV-conditioned and VLM-conditioned refinement. GT trajectories are shown in green, while trajectories refined using backbone BEV features (ChainFlow-BEV) and semantic VLM features (ChainFlow-VLA) are shown in red and orange, respectively.

Generalization of ChainFlow. Table 5 evaluates the generalization of ChainFlow across different end-to-end planning paradigms. We integrate ChainFlow into multiple backbones without VLM features, and conduct all experiments on the navtrain set for fair comparison. On a diffusion-based planner (DiffusionDrive [^22]), replacing clustering-based anchors with our ChainFlow module improves performance from 88.1 to 88.9 using only 6 modes (vs. 20 originally). A similar trend is observed on a score-based planner (iPad [^8]), where ChainFlow improves performance from 91.7 to 92.7. These consistent improvements across heterogeneous backbones demonstrate that ChainFlow serves as a general and effective action expert.

Effect of VLM Guidance. Figure 4 presents a qualitative comparison between residual DiT refinement conditioned on backbone BEV features and semantic VLM features. In the intersection right-turn scenario (Row 1), BEV conditioning produces an incorrect heading, whereas the VLM-conditioned variant correctly captures the intended direction. Across narrow-road cruising, intersection turning, and roundabout scenarios (Rows 2, 4, and 5), trajectories refined with BEV features frequently collide with road boundaries, while semantic guidance from VLM features remains collision-free and even achieves higher ego progress than the expert trajectories. Furthermore, in the low-speed car-following case (Row 3), VLM guidance enables safe following behavior, whereas BEV conditioning results in a rear-end collision. These examples demonstrate that high-level semantic information from VLM features substantially improves trajectory refinement, leading to better safety and driving efficiency.

## 6 Conclusion

We introduced ChainFlow-VLA, a unified vision-language-action framework that casts trajectory planning as a Chain-to-Flow formulation. By decomposing planning into a causal autoregressive Chain and a residual diffusion Flow, our approach unifies temporal reasoning and global geometric consistency within a single probabilistic framework. A central finding of this work is that vision-language models are more effective as semantic conditioners for trajectory refinement rather than direct generators. By leveraging VLM hidden states to guide residual diffusion, we transform planning from global trajectory synthesis into mode-conditioned semantic correction, significantly improving robustness in long-tail scenarios. Extensive experiments on NAVSIM v1 demonstrate that ChainFlow-VLA achieves state-of-the-art performance and reaches human-level driving quality. We hope this work provides a step toward more principled integration of causal reasoning, generative refinement, and semantic understanding in autonomous driving.

Limitations. Although the current VLM guidance improves residual refinement, it is still based on a general driving-oriented VLM trained with environment-understanding and trajectory-QA supervision. Since the Flow module essentially performs trajectory refinement rather than action generation, a score-oriented or judge-oriented VLM with stronger trajectory evaluation ability may be better aligned with this task. Designing such refinement-aware VLM guidance is an important direction for future work.

[^1]: L. Chen, P. Wu, K. Chitta, B. Jaeger, A. Geiger, and H. Li End-to-end autonomous driving: challenges and frontiers. IEEE Transactions on Pattern Analysis and Machine Intelligence 46 (12), pp. 10164–10183. External Links: [Document](https://dx.doi.org/10.1109/TPAMI.2024.3435937) Cited by: §1, §1, §2.

[^2]: S. Chen, B. Jiang, H. Gao, B. Liao, Q. Xu, Q. Zhang, C. Huang, W. Liu, and X. Wang Vadv2: end-to-end vectorized autonomous driving via probabilistic planning. arXiv preprint arXiv:2402.13243. Cited by: §1.

[^3]: D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, et al. Navsim: data-driven non-reactive autonomous vehicle simulation and benchmarking. Advances in Neural Information Processing Systems 37, pp. 28706–28719. Cited by: §1, Table 1, §5.1.

[^4]: L. Feng, Y. Gao, E. Zablocki, Q. Li, W. Li, S. Liu, M. Cord, and A. Alahi Rap: 3d rasterization augmented end-to-end planning. arXiv preprint arXiv:2510.04333. Cited by: Table 1.

[^5]: H. Fu, D. Zhang, Z. Zhao, J. Cui, D. Liang, C. Zhang, D. Zhang, H. Xie, B. Wang, and X. Bai Orion: a holistic end-to-end autonomous driving framework by vision-language instructed action generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 24823–24834. Cited by: §1, §1, §2.

[^6]: X. Gui, M. Zhang, T. Yan, W. Han, J. Gong, F. Tan, C. Xu, and J. Shen Bridging scene generation and planning: driving with world model via unifying vision and motion representation. arXiv preprint arXiv:2603.14948. Cited by: §1.

[^7]: X. Gui, J. Zhao, W. Han, J. Wang, J. Gong, F. Tan, C. Xu, and J. Shen TrajDiff: end-to-end autonomous driving without perception annotation. arXiv preprint arXiv:2512.00723. Cited by: Table 1.

[^8]: K. Guo, H. Liu, X. Wu, J. Pan, and C. Lv IPad: iterative proposal-centric end-to-end autonomous driving. External Links: 2505.15111, [Link](https://arxiv.org/abs/2505.15111) Cited by: §4.4, Table 1, §5.4.

[^9]: M. Hallgarten, J. Zapata, M. Stoll, K. Renz, and A. Zell Can vehicle motion planning generalize to realistic long-tail scenarios?. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 5388–5395. Cited by: §1.

[^10]: Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, et al. Planning-oriented autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 17853–17862. Cited by: §1, §2, Table 1.

[^11]: M. Huang, Y. Xiang, Z. Liang, J. Huang, J. Wang, Z. Xu, F. Tan, H. Zhou, M. Yang, and G. Che CoWorld-vla: thinking in a multi-expert world model for autonomous driving. arXiv preprint arXiv:2605.10426. Cited by: §2.

[^12]: J. Hwang, R. Xu, H. Lin, W. Hung, J. Ji, K. Choi, D. Huang, T. He, P. Covington, B. Sapp, et al. Emma: end-to-end multimodal model for autonomous driving. arXiv preprint arXiv:2410.23262. Cited by: §1, §2.

[^13]: X. Jia, S. Shi, Z. Chen, L. Jiang, W. Liao, T. He, and J. Yan Amp: autoregressive motion prediction revisited with next token prediction for autonomous driving. arXiv preprint arXiv:2403.13331. Cited by: §2.

[^14]: A. Jiang, Y. Gao, Z. Sun, Y. Wang, J. Wang, J. Chai, Q. Cao, Y. Heng, H. Jiang, Y. Dong, et al. Diffvla: vision-language guided diffusion planning for autonomous driving. arXiv preprint arXiv:2505.19381. Cited by: §1.

[^15]: B. Jiang, S. Chen, Q. Xu, B. Liao, J. Chen, H. Zhou, Q. Zhang, W. Liu, C. Huang, and X. Wang Vad: vectorized scene representation for efficient autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 8340–8350. Cited by: §1, §2, Table 1.

[^16]: E. Kirby, A. Boulch, Y. Xu, Y. Yin, G. Puy, É. Zablocki, A. Bursuc, S. Gidaris, R. Marlet, F. Bartoccioni, et al. Driving on registers. arXiv preprint arXiv:2601.05083. Cited by: §4.5, Table 1.

[^17]: J. Li, J. Wu, D. Hu, X. Huang, B. Sun, Z. Hao, X. Lang, X. Zhu, and L. Zhang SGDrive: scene-to-goal hierarchical world cognition for autonomous driving. arXiv preprint arXiv:2601.05640. Cited by: §1, Table 1.

[^18]: K. Li, Z. Li, S. Lan, Y. Xie, Z. Zhang, J. Liu, Z. Wu, Z. Yu, and J. M. Alvarez Hydra-mdp++: advancing end-to-end driving via expert-guided hydra-distillation. arXiv preprint arXiv:2503.12820. Cited by: Table 1.

[^19]: Y. Li, S. Li, X. Liu, M. Gong, K. Li, N. Chen, Z. Wang, Z. Li, T. Jiang, F. Yu, et al. Sscbench: a large-scale 3d semantic scene completion benchmark for autonomous driving. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 13333–13340. Cited by: §1.

[^20]: Y. Li, S. Shang, W. Liu, B. Zhan, H. Wang, Y. Wang, Y. Chen, X. Wang, Y. An, C. Tang, et al. DriveVLA-w0: world models amplify data scaling law in autonomous driving. arXiv preprint arXiv:2510.12796. Cited by: §1, Table 1.

[^21]: Y. Li, K. Xiong, X. Guo, F. Li, S. Yan, G. Xu, L. Zhou, L. Chen, H. Sun, B. Wang, et al. Recogdrive: a reinforced cognitive framework for end-to-end autonomous driving. arXiv preprint arXiv:2506.08052. Cited by: §1, §2, §4.3, Table 1, §5.1.

[^22]: B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang, et al. Diffusiondrive: truncated diffusion model for end-to-end autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 12037–12047. Cited by: §2, Table 1, §5.4.

[^23]: I. Loshchilov and F. Hutter Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101. Cited by: §5.1.

[^24]: W. Peebles and S. Xie Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4195–4205. Cited by: §4.3.

[^25]: C. Sima, K. Chitta, Z. Yu, S. Lan, P. Luo, A. Geiger, H. Li, and J. M. Alvarez Centaur: robust end-to-end autonomous driving with test-time training. arXiv preprint arXiv:2503.11650. Cited by: Table 1.

[^26]: C. Sima, K. Renz, K. Chitta, L. Chen, H. Zhang, C. Xie, P. Luo, A. Geiger, and H. Li DriveLM: driving with graph visual question answering. arXiv preprint arXiv:2312.14150. Cited by: §1, §2.

[^27]: W. Sun, X. Lin, Y. Shi, C. Zhang, H. Wu, and S. Zheng Sparsedrive: end-to-end autonomous driving via sparse scene representation. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pp. 8795–8801. Cited by: §1.

[^28]: Y. Wang, X. Li, W. Wang, J. Zhang, Y. Li, Y. Chen, X. Wang, and Z. Zhang Unified vision-language-action model. arXiv preprint arXiv:2506.19850. Cited by: §2, Table 1.

[^29]: C. Xie, B. Sun, T. Li, J. Wu, Z. Hao, X. Lang, and H. Li LatentVLA: efficient vision-language models for autonomous driving via latent action prediction. arXiv preprint arXiv:2601.05611. Cited by: §1, Table 1.

[^30]: Z. Yang, Y. Chai, X. Jia, Q. Li, Y. Shao, X. Zhu, H. Su, and J. Yan DriveMoE: mixture-of-experts for vision-language-action model in end-to-end autonomous driving. arXiv preprint arXiv:2505.16278. Cited by: §1.

[^31]: W. Yao, Z. Li, S. Lan, Z. Wang, X. Sun, J. M. Alvarez, and Z. Wu Drivesuprim: towards precise trajectory selection for end-to-end planning. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40, pp. 11910–11918. Cited by: Table 1.

[^32]: S. Zeng, X. Chang, M. Xie, X. Liu, Y. Bai, Z. Pan, M. Xu, X. Wei, and N. Guo Futuresightdrive: thinking visually with spatio-temporal cot for autonomous driving. arXiv preprint arXiv:2505.17685. Cited by: §2, Table 1.

[^33]: Y. Zhang, X. Chen, J. Gao, H. Wang, F. Ge, W. Hu, S. Shi, and Z. Zhang OneDrive: unified multi-paradigm driving with vision-language-action models. arXiv preprint arXiv:2604.17915. Cited by: §1.

[^34]: Z. Zheng, S. Chen, H. Yin, X. Zhang, J. Zou, X. Wang, Q. Zhang, and L. Zhang ResAD: normalized residual trajectory modeling for end-to-end autonomous driving. arXiv preprint arXiv:2510.08562. Cited by: §4.3.

[^35]: X. Zhou, X. Han, F. Yang, Y. Ma, V. Tresp, and A. Knoll Opendrivevla: towards end-to-end autonomous driving with large vision language action model. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40, pp. 13782–13790. Cited by: §1, §2.

[^36]: Z. Zhou, T. Cai, S. Z. Zhao, Y. Zhang, Z. Huang, B. Zhou, and J. Ma Autovla: a vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. arXiv preprint arXiv:2506.13757. Cited by: §2, Table 1.

[^37]: Z. Zhou, R. Yang, Y. Guo, S. X. Chen, T. Feng, K. Pistunova, Y. Shen, L. Su, J. Ma, et al. SpanVLA: efficient action bridging and learning from negative-recovery samples for vision-language-action model. arXiv preprint arXiv:2604.19710. Cited by: Table 1.
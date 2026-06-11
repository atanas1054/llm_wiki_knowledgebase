---
title: "CLEAR: Cognition and Latent Evaluation for Adaptive Routing in End-to-End Autonomous Driving"
source: "https://arxiv.org/html/2606.06219v1"
author:
published:
created: 2026-06-11
description:
tags:
  - "clippings"
---
Yining Xing, Zehong Ke, Zhiyuan Liu, Yanbo Jiang, Wenhao Yu, Jianqiang Wang

###### Abstract

End-to-end autonomous driving models often struggle to balance multi-modal maneuver generation with real-time inference constraints. While diffusion models successfully capture diverse driving behaviors, their iterative denoising process incurs unacceptable latency for safety-critical deployment. To address this, we propose CLEAR (Cognition and Latent Evaluation for Adaptive Routing), a framework that combines ultra-fast generative planning with deep semantic reasoning. CLEAR employs Drive-JEPA as the visual encoder and replaces the multi-step denoising chain with a single-step conditional drift in a VAE latent space, introducing a conditioning coefficient to balance diversity and expert precision. Meanwhile, we fully fine-tune Qwen 3.5 0.8B on driving QA pairs to extract scene-aware hidden states. These states guide both an Adaptive Scheduler, which selects the conditioning coefficient $\alpha$ and sample count $N$ from a discrete set of predefined schemes, and a cross-attention scorer that selects the optimal trajectory from candidates. On the NAVSIM v1 benchmark, CLEAR achieves a state-of-the-art PDMS of 93.7. Our results demonstrate that high-fidelity, multi-modal planning can be executed efficiently without dense geometric annotations or iterative sampling.

> Keywords: End-to-End Autonomous Driving, Trajectory Planning, Generative Models, Large Language Models

## 1 Introduction

End-to-end (E2E) autonomous driving has shifted from modular pipelines toward unified architectures that map sensor inputs directly to trajectories [^8] [^10] [^13]. A central challenge remains: urban driving is inherently multi-modal [^11]. At a crowded unsignalized intersection, an autonomous vehicle faces equally valid maneuvers—yielding, proceeding, or merging into cross-traffic. Deterministic regression, the default in most E2E planners, averages across these modes, producing physically implausible trajectories that straddle incompatible maneuvers. When multiple agents negotiate right-of-way simultaneously, the space of valid joint futures expands combinatorially, and a single averaged path becomes meaningless.

Diffusion models capture multi-modality by formulating planning as iterative denoising [^28] [^11] [^27] [^24], but their latency—tens to hundreds of forward passes per prediction—exceeds sub-100ms control budgets [^6] [^20]. Accelerated samplers [^21] [^17] [^16] sacrifice trajectory quality in high-dimensional control spaces. A closer examination of the denoising chain reveals that most steps remove Gaussian noise rather than bridging the semantic gap between unconditional and scene-conditional distributions. The theory of Generative Modeling via Drifting [^3] shows this semantic shift can be accomplished in a single step if geometric decoding is offloaded to a separate module.

On the perception side, Joint-Embedding Predictive Architectures (JEPA) [^22] [^7] suppress photometric noise while preserving driving-relevant structure, outperforming pixel-level reconstruction [^5]. Multi-modal LLMs (MLLMs) have been adapted for driving [^12] [^26], but using them as direct trajectory generators inherits autoregressive latency and format instability. We argue that the LLM’s hidden states—encoding traffic logic, interaction norms, and risk priors—are the primary signal of interest, not its text output. These states can inform both how aggressively to sample candidates and which candidate best matches the scene context.

We propose CLEAR (Cognition and Latent Evaluation for Adaptive Routing), a trajectory prediction framework that unifies single-step generative planning with LLM-driven cognitive reasoning. A frozen Drive-JEPA [^22] backbone supplies abstract geometric features, while a fine-tuned Qwen 3.5 0.8B [^19] serves purely as a semantic feature extractor. A compact MLP-Mixer decoder performs single-step conditional drift in a VAE latent space, producing diverse candidates at up to 99 FPS. The LLM’s hidden states drive an Adaptive Scheduler that selects the scene-appropriate conditioning coefficient $\alpha$ and sample count $N$, and a Cross-Attention Scorer evaluates candidates against learned traffic semantics.

Our main contributions are as follows:

- Single-Step Drift Generation. We replace multi-step denoising with a conditional drift in a VAE latent space, parameterized by a conditioning coefficient $\alpha\in[0,1]$ that interpolates between geometric diversity and expert precision. A pre-fitted PCA projection at the decoder output ensures kinematic feasibility, while the frozen VAE encoder provides semantically structured latent codes. One forward pass yields diverse trajectory candidates at up to 99 FPS without sacrificing multi-modality.
- LLM-Driven Adaptive Scheduling and Scoring. The LLM’s hidden states encode scene complexity and traffic semantics. An Adaptive Scheduler selects a scene-adapted conditioning coefficient $\alpha$ and sample count $N$ from a discrete set of predefined sampling schemes, allocating minimal compute in highway cruising while intensifying sampling at complex intersections. A Cross-Attention Scorer evaluates candidates against the same LLM features, replacing heuristic cost functions with learned, context-aware selection.
- State-of-the-Art Closed-Loop Performance. On NAVSIM [^2], CLEAR achieves a PDMS of 93.7, surpassing methods that rely on dense 3D perception annotations, while using only Drive-JEPA visual features and cognitive QA pairs from a compact 0.8B LLM. This demonstrates that accurate planning does not require exhaustive geometric reconstruction nor large-scale MLLMs.

## 2 Related Work

### 2.1 End-to-End Autonomous Driving and Visual Representations

E2E driving has evolved from modular pipelines to unified architectures [^8] [^10] [^9] [^13], yet most still rely on deterministic regression that averages over distinct valid modes. Visual representation choice is equally critical: pixel-level reconstruction (e.g., MAE [^5]) wastes capacity on photometric noise, while JEPA [^22] [^7] predicts scene evolution in latent space, preserving driving-relevant structure. CLEAR employs a frozen Drive-JEPA backbone to supply noise-suppressed geometric priors.

### 2.2 Generative Models for Trajectory Planning

Diffusion models [^6] [^20] capture multi-modal distributions via iterative denoising [^11] [^28] [^27] [^24], but their latency—tens to hundreds of network evaluations—prohibits real-time deployment. Accelerated samplers [^21] [^17] [^16] sacrifice quality in high-dimensional control spaces. Generative Modeling via Drifting [^3] achieves distribution matching in a single forward pass. CLEAR adapts this theory to trajectory planning via single-step conditional drift in a VAE latent space with PCA output projection, preserving multi-modal coverage with minimal overhead.

### 2.3 Cognitive Reasoning and Trajectory Scoring

MLLMs have been adapted for driving decisions [^23] [^12] [^26], confirming they capture traffic semantics including right-of-way conventions and risk priors. However, using them as direct trajectory generators incurs unacceptable latency and format instability. CLEAR treats the LLM strictly as a semantic feature extractor: its hidden states drive an Adaptive Scheduler that controls generative diversity and a Cross-Attention Scorer that evaluates candidates against cognitive features, connecting high-level reasoning with low-level control.

## 3 Method

### 3.1 Model Framework

Given a sequence of front-view images, the ego-vehicle’s historical poses, and a high-level navigation command, our goal is to predict a multi-modal set of future trajectories and select the optimal execution path. The CLEAR framework integrates representation learning, generative planning, and cognitive reasoning into a cohesive end-to-end architecture. A frozen Drive-JEPA [^22] visual encoder extracts abstract geometric features, while a fine-tuned Qwen 3.5 0.8B [^19] serves as the cognitive engine. A trainable Adaptive Scheduler parses the LLM’s hidden states to determine the necessary generative diversity. The CLEAR Decoder, an efficient MLP-Mixer-based generation engine, performs single-step conditional drifting to yield $N$ physical trajectory candidates. A trainable Cross-Attention Scorer selects the optimal trajectory by evaluating candidates against the LLM’s cognitive features.

![[framework.png|Refer to caption]]

Figure 1: Overview of the CLEAR architecture. Given front-view images and a navigation command, the frozen Drive-JEPA encoder and fine-tuned Qwen 3.5 0.8B produce visual and semantic features. The Adaptive Scheduler predicts ( α, N ) (\\alpha,N); the CLEAR Decoder generates candidates via single-step drift; the Cross-Attention Scorer selects the optimal trajectory.

### 3.2 Single-Step Conditional Drift in Latent Space

We instantiate the drift [^3] in a VAE latent space. A variational autoencoder with an auxiliary maneuver classification head encodes trajectories into compact latent codes, structuring the latent manifold around behaviorally meaningful driving primitives. Only the VAE encoder is retained; physical trajectory outputs are obtained by projecting latent codes through a PCA basis derived from expert demonstrations, which acts as a low-pass filter constraining decoded trajectories to the expert kinematic subspace.

Each scene is paired with geometrically feasible trajectories $\mathcal{S}_{\text{geom}}$ that serve as multiple positive attractors in the VAE latent space, while the expert ground truth $\mathbf{V}_{\text{GT}}$ serves as a precision anchor. Diversity arises from two complementary mechanisms: a *multi-attractor* structure that prevents mode collapse via soft assignment to different positive samples, and *inter-sample repulsion* that pushes candidates apart in latent space. We construct a drift target for each candidate that combines both mechanisms. The attractive component interpolates between the assigned positive attractor and $\mathbf{V}_{\text{GT}}$:

$$
\mathbf{A}_{i}=(1-\alpha)\cdot\mathbf{V}_{\text{pos}(i)}+\alpha\cdot\mathbf{V}_{\text{GT}}
$$

where $\mathbf{V}_{\text{pos}(i)}$ is the VAE encoding of the positive sample matched to candidate $i$ via attention-weighted soft assignment. The repulsive component is computed from the other generated candidates’ VAE encodings:

$$
\mathbf{R}_{i}=\frac{1}{N-1}\sum_{j\neq i}\text{VAE}_{\text{enc}}(\boldsymbol{\tau}_{j})
$$

The final drift target combines attraction and repulsion. The corresponding loss drives each candidate toward its target:

$$
\mathbf{V}_{i}=\mathbf{A}_{i}-\mathbf{R}_{i},\quad\mathcal{L}_{\text{drift}}=\frac{1}{N}\sum_{i=1}^{N}\|\text{VAE}_{\text{enc}}(\boldsymbol{\tau}_{i})-\text{sg}(\mathbf{V}_{i})\|_{2}^{2}
$$

where $\text{sg}(\cdot)$ denotes the stop-gradient operator. A Winner-Take-All loss applies gradients only to the candidate closest to the ground truth:

$$
\mathcal{L}_{\text{WTA}}=\alpha\cdot\min_{i}\|\boldsymbol{\tau}_{i}-\boldsymbol{\tau}^{\text{GT}}\|_{1}
$$

This establishes an implicit curriculum where complex scenes ($\alpha\to 0$) emphasize distributional coverage, while simple scenes ($\alpha\to 1$) emphasize physical precision.

### 3.3 Cognitive-Driven Adaptive Scheduling

The scalar $\alpha$ governs the semantic shift: $\alpha\to 1$ yields deterministic, expert-mimicking trajectories ideal for simple scenarios (e.g., highway cruising), while $\alpha\to 0$ produces diverse, multi-modal coverage critical for complex interactions (e.g., crowded unsignalized intersections). Deep semantic understanding of scene complexity is essential for selecting the optimal $\alpha$ and sample count $N$.

We introduce an Adaptive Scheduler driven by the fine-tuned Qwen 3.5 0.8B. Rather than regressing continuous values, the scheduler selects from $K$ predefined sampling schemes $\{s_{k}=(\alpha_{k},N_{k})\}_{k=1}^{K}$, where each scheme specifies a conditioning coefficient and a candidate count. A lightweight TransformerDecoder maps the LLM hidden states $\mathbf{H}_{\text{LLM}}$ to a categorical distribution:

$$
\mathbf{p}=\text{softmax}(g_{\text{adapt}}(\mathbf{H}_{\text{LLM}})),\quad k^{*}=\arg\max_{k}p_{k}
$$

The scheduler is trained with cross-entropy loss:

$$
\mathcal{L}_{\text{adapt}}=-\log p_{k_{\text{opt}}}
$$

where $k_{\text{opt}}$ is the optimal scheme index. During training, we determine $k_{\text{opt}}$ for each scene by evaluating all $K$ schemes with the official PDMS scorer and selecting the scheme yielding the highest score. This provides supervision labels without requiring a separate scorer during training. At inference, the scheduler directly predicts $k^{*}$ via argmax.

### 3.4 The CLEAR Decoder: Efficient Intent-Driven Generation

The CLEAR Decoder generates $N$ physical trajectory candidates in a single batched forward pass, conditioned on the scene-adapted $\alpha$ from the Adaptive Scheduler. To handle the dense output of the vision encoder, we employ learnable Scene Queries that cross-attend to the visual features, compressing them into a compact semantic summary. This summary is then concatenated with the ego-state and navigation intent to form a unified token array. The core of the decoder is built on an MLP-Mixer architecture, which alternates between token-mixing and channel-mixing MLPs, facilitating rapid, parallelized generation across the $N$ trajectory candidates.

The conditioning coefficient $\alpha$ controls the drift dynamics (Eq. 1) and is injected into the network via Adaptive Layer Normalization (adaLN) [^18]. Specifically, $\alpha$ is first mapped through a small MLP to a conditioning vector, which then modulates every MLP-Mixer block:

$$
\text{adaLN}(\mathbf{x};\alpha)=\gamma(\alpha)\odot\text{LayerNorm}(\mathbf{x})+\beta(\alpha)
$$

where $\gamma(\alpha)$ and $\beta(\alpha)$ are the learned scale and shift parameters. This $\alpha$ -conditioned normalization ensures that the resulting $N$ trajectories follow the distribution shape specified by the LLM’s scene understanding.

The MLP-Mixer output $\mathbf{F}_{\text{traj}}\in\mathbb{R}^{N\times D}$ is projected to physical trajectory waypoints $\boldsymbol{\tau}_{i}$ through a frozen PCA basis pre-fitted on expert demonstrations, which acts as a low-pass filter constraining decoded trajectories to the expert kinematic subspace. These physical trajectories are used to compute the Winner-Take-All loss (Eq. 4) and, after being re-encoded into the VAE latent space by the frozen VAE encoder, the drift loss (Eq. 3).

### 3.5 Cross-Attention Scorer

The Cross-Attention Scorer evaluates the $N$ trajectory candidates via a TransformerDecoder, where the MLP-Mixer output features $\mathbf{F}_{\text{traj}}$ serve as queries and LLM hidden states $\mathbf{H}_{\text{LLM}}$ serve as memory. The output is projected to a scalar score $S_{i}$ for each candidate $i$, estimating its overall PDMS.

The scorer is trained with a combination of a pairwise hinge ranking loss and an MSE loss. The ranking loss enforces correct relative ordering among candidate pairs:

$$
\mathcal{L}_{\text{rank}}=\frac{1}{|\mathcal{P}|}\sum_{(i,j)\in\mathcal{P}}\max(0,m-(S_{i}-S_{j}))\cdot\mathbb{1}[y_{i}-y_{j}>m]
$$

where $y_{i}$ denotes the ground-truth PDMS of candidate $i$, $\mathcal{P}$ is the set of candidate pairs where one strictly outperforms the other, and $m$ is a relaxation margin that provides soft supervision—the loss focuses on whether the relative ordering is correct rather than penalizing absolute score differences. The MSE loss directly supervises the predicted score against the ground-truth PDMS:

$$
\mathcal{L}_{\text{mse}}=\frac{1}{N}\sum_{i=1}^{N}(S_{i}-y_{i})^{2}
$$

The scorer loss is $\mathcal{L}_{\text{scorer}}=\lambda_{\text{rank}}\mathcal{L}_{\text{rank}}+\lambda_{\text{mse}}\mathcal{L}_{\text{mse}}$.

## 4 Experiments

### 4.1 Datasets and Implementation Details

We evaluate the CLEAR framework on the NAVSIM dataset [^2], a large-scale, closed-loop driving benchmark widely used for evaluating recent end-to-end planning architectures, including Drive-JEPA [^22], ReCogDrive [^12], iPAD [^4], and GTRS [^14]. NAVSIM assesses planner performance via comprehensive closed-loop simulation metrics without requiring exhaustive 3D perception annotations.

To construct our training pipeline, we curate specific data splits for each modular phase. For generative pre-training, we extract and process approximately 130,000 driving trajectories to pre-train the VAE and to train the CLEAR Decoder. For cognitive fine-tuning, we perform full parameter fine-tuning on the Qwen 3.5 0.8B LLM using a targeted subset of 17,000 scenes, comprising 150,000 structured driving QA pairs sourced from ReCogDrive [^12], which injects the necessary traffic logic and risk priors into the language model. To train the Adaptive Scheduler and Cross-Attention Scorer, we synthesize a contrastive dataset from 10,000 driving scenes. For each scene, we generate trajectory pools across a grid of sample budgets $N\in\{16,64,256\}$ and 6 discrete conditioning coefficient levels $\alpha$, yielding $(16+64+256)\times 6$ trajectories per scene. This pool provides dense supervision for training both the Adaptive Scheduler and Cross-Attention Scorer.

### 4.2 Training Dynamics and Latent Distribution Analysis

Our training executes in decoupled phases to ensure stability. The VAE is pre-trained on trajectory data with a maneuver classification auxiliary task, and the PCA projection is pre-fitted on expert demonstrations. The CLEAR Decoder is then trained from scratch for 500 epochs with both the VAE encoder and PCA projection frozen. In parallel, the LLM undergoes full fine-tuning for 20 epochs. Finally, the LLM-driven Adaptive Scheduler and Cross-Attention Scorer are trained for 100 epochs while keeping the upstream representations frozen.

![[training_plot.png|Refer to caption]]

Figure 2: Evolution of trajectory generation in both physical space (rows 1 and 3) and latent feature space (rows 2 and 4) over 500 epochs, illustrated for Left Turn and Right Turn scenarios. At early stages (Epoch 0–24), samples are scattered. By Epoch 499, low drift intensity ( α = 0.1 \\alpha=0.1, blue) expansively covers geometrically feasible paths to maintain multi-modal diversity, while high drift intensity ( 0.9 \\alpha=0.9, purple) tightly converges onto the expert ground truth (GT, grey star). Intermediate drift ( 0.5 \\alpha=0.5, orange) balances exploration and precision.

To validate the efficacy of our single-step conditional drift (Eq. 1), we visualize the evolution of the generated trajectory distributions in both the physical space and the 2D projected latent feature space over the 500-epoch decoder training (Figure 2). We sample the validation set at specific training milestones (Epochs 0, 24, 99, 249, and 499) across distinct driving scenarios (e.g., Left Turn and Right Turn), comparing the generation manifolds at three drift intensities: $\alpha\in\{0.1,0.5,0.9\}$.

At early stages (Epoch 0 to 24), the generated latent features are widely scattered, and the corresponding physical trajectories fail to capture the target topologies. However, as training progresses through Epoch 99 and settles by Epoch 499, a clear structural bifurcation governed by $\alpha$ emerges, consistent with our theoretical design. For a low drift intensity ($\alpha=0.1$, blue), each candidate is attracted toward a different positive sample in $\mathcal{S}_{\text{geom}}$, and the resulting manifold expansively covers the geometrically feasible positive samples, forming a multi-modal distribution that respects varying turning radii in the physical space.

Conversely, at a high drift intensity ($\alpha=0.9$, purple), the drift is heavily influenced by $\mathbf{V}_{\text{GT}}$, and the distribution tightly collapses around the expert ground truth (grey star), exhibiting deterministic precision. The intermediate state ($\alpha=0.5$, orange) interpolates between these extremes. This progressive convergence confirms that CLEAR’s single-step drift can effectively decouple and dynamically control geometric diversity and expert precision.

### 4.3 Closed-Loop Evaluation on NAVSIM

We evaluate CLEAR on the NAVSIM benchmark [^2], reporting the PDM Score (PDMS) under the v1 protocol and the Extended PDMS (EPDMS) under the more rigorous v2 protocol. PDMS aggregates No at-fault Collision (NC), Drivable Area Compliance (DAC), Ego Progress (EP), Comfort (C), and Time-to-Collision (TTC), while EPDMS further incorporates Driving Direction Compliance (DDC), Traffic Light Compliance (TLC), Lane Keeping (LK), History Comfort (HC), and Extended Comfort (EC).

##### NAVSIM v1.

Table 1 compares CLEAR against recent advanced methods on NAVSIM v1. CLEAR achieves a new state-of-the-art PDMS of 93.7, surpassing DriveSuprim [^25] (93.5) and Drive-JEPA [^22] (93.3). Most notably, CLEAR substantially improves safety-critical metrics, pushing TTC from 95.9 to 97.2 and achieving top scores in NC and DAC. This suggests that the LLM-driven scheduler and cross-attention scorer effectively enhance safety-aware planning. While metrics like Ego Progress and Comfort remain comparable to prior arts, the overall safety gains make CLEAR the strongest planner on v1.

Table 1: NAVSIM v1 closed-loop results (PDMS $\uparrow$). Bold: best; underlined: second best.

| Method | NC $\uparrow$ | DAC $\uparrow$ | EP $\uparrow$ | Comf.$\uparrow$ | TTC $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| GoalFlow [^24] | 98.4 | 98.3 | 85.0 | 100 | 94.6 | 90.3 |
| DiffusionDrive [^15] | 98.2 | 96.2 | 82.2 | 100 | 94.7 | 88.1 |
| ReCogDrive [^12] | 97.9 | 97.3 | 87.3 | 100 | 94.9 | 90.8 |
| iPad [^4] | 98.6 | 98.3 | 88.0 | 100 | 94.9 | 91.7 |
| DriveSuprim [^25] | 98.6 | 98.6 | 91.3 | 100 | 95.5 | 93.5 |
| Drive-JEPA [^22] | 99.1 | 98.2 | 90.8 | 99.9 | 95.9 | 93.3 |
| CLEAR (Ours) | 99.1 | 98.8 | 89.7 | 99.6 | 97.2 | 93.7 |

##### NAVSIM v2.

Table 2 reports EPDMS under the v2 protocol, which adds five sub-metrics to better assess driving quality. CLEAR achieves the highest EPDMS (88.6) among ViT/L-scale methods, leading in NC, DAC, EP, TTC, and EC. However, CLEAR still lags in LK and TL compared to other ViT/L methods, indicating room for improvement in lane keeping and traffic light compliance.

Table 2: NAVSIM v2 closed-loop results (EPDMS $\uparrow$). Bold: best; underlined: second best.

<table><tbody><tr><th>Method</th><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DDC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TL <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>LK <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>HC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EPDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><th colspan="11">ResNet34 Backbone</th></tr><tr><th>Transfuser <sup><a href="#fn:1">1</a></sup></th><td>96.9</td><td>89.9</td><td>97.8</td><td>99.7</td><td>87.1</td><td>95.4</td><td>92.7</td><td>98.3</td><td>87.2</td><td>76.7</td></tr><tr><th>HydraMDP++ <sup><a href="#fn:13">13</a></sup></th><td>97.2</td><td>97.5</td><td>99.4</td><td>99.6</td><td>83.1</td><td>96.5</td><td>94.4</td><td>98.2</td><td>70.9</td><td>81.4</td></tr><tr><th>DriveSuprim <sup><a href="#fn:25">25</a></sup></th><td>97.5</td><td>96.5</td><td>99.4</td><td>99.6</td><td>88.4</td><td>96.6</td><td>95.5</td><td>98.3</td><td>77.0</td><td>83.1</td></tr><tr><th>iPad <sup><a href="#fn:4">4</a></sup></th><td>98.7</td><td>97.8</td><td>99.1</td><td>99.8</td><td>84.0</td><td>98.0</td><td>96.0</td><td>98.0</td><td>68.2</td><td>84.1</td></tr><tr><th>Drive-JEPA <sup><a href="#fn:22">22</a></sup></th><td>98.8</td><td>97.4</td><td>99.0</td><td>99.8</td><td>83.5</td><td>98.0</td><td>96.2</td><td>98.1</td><td>85.6</td><td>85.4</td></tr><tr><th colspan="11">ViT/L Backbone</th></tr><tr><th>HydraMDP++ <sup><a href="#fn:13">13</a></sup></th><td>98.5</td><td>98.5</td><td>99.5</td><td>99.7</td><td>87.4</td><td>97.9</td><td>95.8</td><td>98.2</td><td>75.7</td><td>85.6</td></tr><tr><th>iPad <sup><a href="#fn:4">4</a></sup></th><td>98.7</td><td>98.0</td><td>98.9</td><td>99.8</td><td>86.6</td><td>98.3</td><td>97.2</td><td>98.3</td><td>74.6</td><td>85.8</td></tr><tr><th>DriveSuprim <sup><a href="#fn:25">25</a></sup></th><td>98.4</td><td>98.6</td><td>99.6</td><td>99.8</td><td>90.5</td><td>97.8</td><td>97.0</td><td>98.3</td><td>78.6</td><td>87.1</td></tr><tr><th>Drive-JEPA <sup><a href="#fn:22">22</a></sup></th><td>98.4</td><td>98.6</td><td>99.1</td><td>99.8</td><td>88.4</td><td>97.8</td><td>97.6</td><td>97.9</td><td>84.8</td><td>87.8</td></tr><tr><th>CLEAR (Ours)</th><td>99.0</td><td>98.7</td><td>99.6</td><td>96.9</td><td>91.0</td><td>98.4</td><td>92.9</td><td>96.4</td><td>79.5</td><td>88.6</td></tr></tbody></table>

##### Ablation Study.

Table 3 evaluates the contribution of two core design choices: the LLM cross-attention scorer and the adaptive sampling scheduler. All variants share the same vision encoder and CLEAR Decoder, differing only in the scorer and scheduling strategy.

Table 3: Ablation study on NAVSIM v1 (PDMS $\uparrow$). “LLM Scorer” indicates whether the Cross-Attention Scorer uses LLM hidden states (otherwise vision encoder features only). “Adaptive” indicates whether the LLM-driven scheduler adaptively selects $\alpha$ and $N$ (otherwise fixed $\alpha{=}0.5$, $N{=}64$).

| ID | LLM Scorer | Adaptive | NC $\uparrow$ | DAC $\uparrow$ | EP $\uparrow$ | Comf.$\uparrow$ | TTC $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (a) | $\times$ | $\times$ | 98.9 | 98.8 | 88.4 | 99.7 | 97.2 | 93.1 |
| (b) | ✓ | $\times$ | 99.1 | 98.9 | 88.6 | 99.7 | 97.1 | 93.3 |
| (c) | ✓ | ✓ | 99.1 | 98.8 | 89.7 | 99.6 | 97.2 | 93.7 |

As illustrated in Table 3, integrating the LLM Scorer (a vs b) elevates PDMS from 93.1 to 93.3, validating that cognitive features provide superior evaluation signals over vision-only baselines. Building on this foundation, the adaptive scheduler (b vs c) addresses performance bottlenecks by significantly improving EP (from 88.6 to 89.7), ultimately pushing PDMS to 93.7. This synergy confirms that the modules are complementary: the scheduler generates diverse, scene-appropriate candidates, and the LLM Scorer selects the safest, most efficient trajectory from them.

## 5 Conclusion

We presented CLEAR, a trajectory prediction framework that replaces the iterative denoising chain of diffusion-based planners with a single-step conditional drift in a VAE latent space. By coupling a frozen Drive-JEPA visual backbone with the fine-tuned Qwen 3.5 0.8B serving purely as a semantic feature extractor, CLEAR achieves efficient, multi-modal planning by leveraging the LLM’s hidden states to drive an Adaptive Scheduler that selects scene-appropriate diversity parameters, while also informing a Cross-Attention Scorer that evaluates candidates against learned traffic semantics rather than geometric heuristics. On NAVSIM v1, CLEAR achieves a state-of-the-art PDMS of 93.7 while running at up to 99 FPS. High-quality closed-loop planning does not require exhaustive geometric reconstruction, large-scale MLLMs, or costly iterative sampling—a compact combination of frozen encoders and a single-step drift decoder can suffice.

## 6 Limitations and Future Work

Two limitations warrant discussion. First, the Adaptive Scheduler selects from a discrete set of predefined ($\alpha$, $N$) schemes, which may miss the globally optimal configuration lying between grid points. Second, the multi-stage training pipeline requires separate pre-training of the VAE, PCA projection, LLM fine-tuning, and downstream modules. Future work will explore continuous or differentiable scheduling to enable finer-grained control, as well as joint optimization across modules to reduce training complexity and improve end-to-end coherence.

#### Acknowledgments

[^1]: K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger (2022) Transfuser: imitation with transformer-based sensor fusion for autonomous driving. In IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol. 45, pp. 12878–12955. Cited by: Table 2.

[^2]: D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, et al. (2024) Navsim: data-driven non-reactive autonomous vehicle simulation and benchmarking. Advances in Neural Information Processing Systems 37, pp. 28706–28719. Cited by: 3rd item, §4.1, §4.3.

[^3]: M. Deng, H. Li, T. Li, Y. Du, and K. He (2026) Generative modeling via drifting. arXiv preprint arXiv:2602.04770. Cited by: §1, §2.2, §3.2.

[^4]: K. Guo, H. Liu, X. Wu, J. Pan, and C. Lv (2025) Ipad: iterative proposal-centric end-to-end autonomous driving. arXiv preprint arXiv:2505.15111. Cited by: §4.1, Table 1, Table 2, Table 2.

[^5]: K. He, X. Chen, S. Xie, Y. Li, P. Dollár, and R. Girshick (2022) Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 16000–16009. Cited by: §1, §2.1.

[^6]: J. Ho, A. Jain, and P. Abbeel (2020) Denoising diffusion probabilistic models. Advances in neural information processing systems 33, pp. 6840–6851. Cited by: §1, §2.2.

[^7]: A. Hu, L. Russell, H. Yeo, Z. Murez, G. Fedoseev, A. Kendall, J. Shotton, and G. Corrado (2023) GAIA-1: a generative world model for autonomous driving. arXiv preprint arXiv:2309.17080. Cited by: §1, §2.1.

[^8]: Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, et al. (2023) Planning-oriented autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 17853–17862. Cited by: §1, §2.1.

[^9]: B. Jiang, S. Chen, H. Gao, B. Liao, Q. Zhang, W. Liu, and X. Wang (2024) VADv2: end-to-end vectorized autonomous driving via probabilistic planning. In The Fourteenth International Conference on Learning Representations, Cited by: §2.1.

[^10]: B. Jiang, S. Chen, Q. Xu, B. Liao, J. Chen, H. Zhou, Q. Zhang, W. Liu, C. Huang, and X. Wang (2023) Vad: vectorized scene representation for efficient autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 8340–8350. Cited by: §1, §2.1.

[^11]: C. Jiang, A. Cornman, C. Park, B. Sapp, Y. Zhou, D. Anguelov, et al. (2023) Motiondiffuser: controllable multi-agent motion prediction using diffusion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 9644–9653. Cited by: §1, §1, §2.2.

[^12]: Y. Li, K. Xiong, X. Guo, F. Li, S. Yan, G. Xu, L. Zhou, L. Chen, H. Sun, B. Wang, et al. (2025) Recogdrive: a reinforced cognitive framework for end-to-end autonomous driving. arXiv preprint arXiv:2506.08052. Cited by: §1, §2.3, §4.1, §4.1, Table 1.

[^13]: Z. Li, K. Li, S. Wang, S. Lan, Z. Yu, Y. Ji, Z. Li, Z. Zhu, J. Kautz, Z. Wu, et al. (2024) Hydra-mdp: end-to-end multimodal planning with multi-target hydra-distillation. arXiv preprint arXiv:2406.06978. Cited by: §1, §2.1, Table 2, Table 2.

[^14]: Z. Li, W. Yao, Z. Wang, X. Sun, J. Chen, N. Chang, M. Shen, Z. Wu, S. Lan, and J. M. Alvarez (2025) Generalized trajectory scoring for end-to-end multimodal planning. arXiv preprint arXiv:2506.06664. Cited by: §4.1.

[^15]: B. Liao, S. Chen, Y. Wang, T. Cheng, Q. Zhang, W. Liu, and C. Huang (2025) DiffusionDrive: towards an efficient diffusion-based end-to-end planner. arXiv preprint arXiv:2411.15139. Cited by: Table 1.

[^16]: Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le (2023) Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, Cited by: §1, §2.2.

[^17]: X. Liu, C. Gong, and Q. Liu (2022) Flow straight and fast: learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003. Cited by: §1, §2.2.

[^18]: W. Peebles and S. Xie (2023) Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4195–4205. Cited by: §3.4.

[^19]: Qwen Team (2026-02) Qwen3.5: towards native multimodal agents. External Links: [Link](https://qwen.ai/blog?id=qwen3.5) Cited by: §1, §3.1.

[^20]: J. Song, C. Meng, and S. Ermon (2020) Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502. Cited by: §1, §2.2.

[^21]: Y. Song, P. Dhariwal, M. Chen, and I. Sutskever (2023) Consistency models. Cited by: §1, §2.2.

[^22]: L. Wang, Z. Yang, C. Bai, G. Zhang, X. Liu, X. Zheng, X. Long, C. Lu, and C. Lu (2026) Drive-jepa: video jepa meets multimodal trajectory distillation for end-to-end driving. arXiv preprint arXiv:2601.22032. Cited by: §1, §1, §2.1, §3.1, §4.1, §4.3, Table 1, Table 2, Table 2.

[^23]: J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou, et al. (2022) Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems 35, pp. 24824–24837. Cited by: §2.3.

[^24]: Z. Xing, X. Zhang, Y. Hu, B. Jiang, T. He, Q. Zhang, X. Long, and W. Yin (2025) Goalflow: goal-driven flow matching for multimodal trajectories generation in end-to-end autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 1602–1611. Cited by: §1, §2.2, Table 1.

[^25]: W. Yao, Z. Li, S. Lan, Z. Wang, X. Sun, J. M. Alvarez, and Z. Wu (2026) Drivesuprim: towards precise trajectory selection for end-to-end planning. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40, pp. 11910–11918. Cited by: §4.3, Table 1, Table 2, Table 2.

[^26]: P. Zheng, Y. Zhao, Z. Gong, H. Zhu, and S. Wu (2025) SimpleVSF: vlm-scoring fusion for trajectory prediction of end-to-end autonomous driving. arXiv preprint arXiv:2510.17191. Cited by: §1, §2.3.

[^27]: Y. Zheng, R. Liang, K. ZHENG, J. Zheng, L. Mao, J. Li, W. Gu, R. Ai, S. E. Li, X. Zhan, et al. Diffusion-based planning for autonomous driving with flexible guidance. In ICLR 2025 Workshop on Deep Generative Model in Machine Learning: Theory, Principle and Efficacy, Cited by: §1, §2.2.

[^28]: Z. Zhong, D. Rempe, Y. Chen, B. Ivanovic, Y. Cao, D. Xu, M. Pavone, and B. Ray (2023) Language-guided traffic simulation via scene-level diffusion. In Conference on robot learning, pp. 144–177. Cited by: §1, §2.2.
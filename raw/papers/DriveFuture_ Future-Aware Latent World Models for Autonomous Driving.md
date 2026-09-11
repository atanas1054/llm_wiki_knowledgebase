---
title: "DriveFuture: Future-Aware Latent World Models for Autonomous Driving"
source: "https://arxiv.org/html/2605.09701v1"
author:
published:
created: 2026-09-11
description:
tags:
  - "clippings"
---
Yufeng Hong    Xiaotian Zhou    Yingyan Li Affiliation: Institute of Automation, Chinese Academy of Sciences    Xiangpo Zhou Affiliation: Beihang University    Lin Liu Affiliation: Beijing Jiaotong University    Yadan Luo Affiliation: The University of Queensland    Shaoqing Xu Affiliation: University of Macau    Lei Yang Affiliation: Nanyang Technological University    Ziying Song Affiliation: School of Artificial Intelligence ( School of Software), Yanshan University Affiliation: Beijing Institute of Technology Affiliation: Equal contribution. Affiliation: Corresponding author.

###### Abstract

Existing latent world models for autonomous driving have opened a promising path toward future-aware driving intelligence. However, they typically treat future latent states as prediction targets or auxiliary signals, rather than directly conditioning trajectory planning. This can entangle current and future features in latent space. In this work, we propose DriveFuture, a future-aware latent world modeling framework for autonomous driving that explicitly learns planning-oriented foresight by conditioning the current latent state modeling process on future world states. Specifically, during training, the model first predicts future latent world states from the current latent state and ego action, and then refines the prediction against the ground-truth future latent state via cross-attention. The resulting future-aware latent serves as an explicit condition for a diffusion-based trajectory planner. During inference, DriveFuture conditions on the predicted future latent state instead of the ground-truth future state. DriveFuture achieves SOTA performance on the public NAVSIM benchmarks [^1], reaching 55.5 EPDMS on NAVSIM-v2 navhard, 89.9 EPDMS on NAVSIM-v2 navtest, and 90.7 PDMS on NAVSIM-v1 navtest, respectively. These results suggest that the key to latent world modeling lies not merely in simulating future states, but more importantly in conditioning current decision-making on future states. Notably, as of April 2026, DriveFuture ranks 1st on the [NAVSIM-v2 navhard](https://huggingface.co/spaces/AGC2025/e2e-driving-navhard) leaderboard and achieves SOTA performance on [NAVSIM-v1 navtest](https://huggingface.co/spaces/AGC2024-P/e2e-driving-navtest).

## 1 Introduction

In recent years, world models have gradually emerged as a central direction in autonomous driving and are widely regarded as one of the most promising technical paradigms for achieving higher-level driving intelligence [^2] [^3] [^4] [^5] [^6]. Unlike conventional approaches [^7] [^8] [^9] [^10] [^11] [^12] [^13] [^14] [^15] [^16], which directly map the current state to driving actions, world models [^2] [^17] learn the latent dynamics of the driving environment. This allows the system to anticipate future outcomes under different actions and make more informed planning decisions. By reasoning about how the world may evolve before selecting an action, this paradigm shifts autonomous driving from reactive control toward future-aware decision-making. As a result, world models are evolving from an auxiliary modeling tool into a key foundation for scalable and generalizable autonomous driving.

Among existing world-modeling paradigms, latent world models are particularly promising for autonomous driving for three reasons. First, they learn dynamics in compact latent spaces, avoiding costly pixel-level generation. Second, latent representations can abstract away low-level visual details and focus on planning-relevant scene structure, interactions, and action consequences. Third, their temporally structured representations are naturally suited for long-horizon planning. This line of work has rapidly evolved from latent future prediction [^17] to intention-aware latent planning [^18], planning-oriented representation refinement with reinforcement fine-tuning [^19], and more recent unifications with VLA/planners and policy scaling [^20] [^21] [^22] [^23]. These developments indicate that latent world models are no longer merely an efficient substitute for observation-space simulation, but are increasingly becoming a general representation substrate for scalable autonomous driving.

Although latent world models [^17] [^18] [^19] [^23] [^20] [^21] [^22] provide an efficient paradigm for future modeling in autonomous driving, existing methods primarily focus on *future state prediction* rather than *current decision representation*. Specifically, as shown in Fig. 1, they typically treat future latent states as prediction targets or auxiliary signals, rather than using them as explicit conditions to shape the current decision-making process, which leads to entanglement between current and future features in latent space. As a result, these models are better at *simulating the future* than at learning planning-oriented decision representations, thereby limiting the ability of future information to guide current decision-making.

![[motivationv13.png|Refer to caption]]

Figure 1: Motivation of DriveFuture. (a) Existing latent world models 17 18 19 23 20 21 22 primarily simulate future latent states and use them as prediction targets or supervision signals, without explicitly shaping the current representation for planning. (b) DriveFuture uses future latent states as direct conditions for the planning process. It adopts GT future states during training and predicted future states during inference, enabling future-aware trajectory planning. (c) DriveFuture achieves SOTA performance on the public leaderboards of NAVSIM-v2 navhard and NAVSIM-v1 navtest.

We argue that the role of future information in autonomous driving is not merely to predict how future scenes will evolve, but more importantly to endow the system with a decision-making perspective akin to that in *The Terminator*, namely, *bringing future knowledge back to the present*: using awareness of future consequences at the current moment to reshape the understanding of the present scene and the resulting decision. An ideal autonomous driving system, therefore, should not be merely a future predictor, but should instead allow future states to act as conditions that influence the current decision-making process in reverse, thereby enabling more foresighted planning. Inspired by this observation, we propose DriveFuture, a future-aware latent world modeling framework for autonomous driving. By conditioning the current decision-making process on future world states, DriveFuture explicitly learns planning-oriented *foresight*. Unlike prior approaches that treat future latent states solely as prediction targets, DriveFuture regards them as structured conditions for shaping the current decision-making process.

Our framework follows a simple future-aware latent world modeling pipeline. During training, DriveFuture first predicts future latent world states from the current latent state and ego action, and then uses the ground-truth (GT) future latent world state as a grounding signal for the predicted future latent, which then conditions the diffusion-based trajectory planner. This process enables the planner to leverage future semantics for more informed trajectory generation. During inference, GT future latent states are no longer available, and the model instead uses predicted future latent states as conditions, enabling closed-loop planning without future annotations. DriveFuture establishes a unified future-aware latent dynamics mechanism for both training and inference. We evaluate DriveFuture on the public NAVSIM benchmarks [^1] and observe substantial performance gains. Specifically, DriveFuture achieves 55.5, 89.9, and 90.7 EPDMS on NAVSIM-v2 navhard, NAVSIM-v2 navtest, and NAVSIM-v1 navtest, respectively. These results show that future-aware latent world modeling can significantly improve planning quality in challenging closed-loop driving scenarios. In summary, the main contributions of this work are as follows:

- We identify a key limitation of existing latent world models for autonomous driving: they mainly focus on future-state simulation, while failing to fully exploit future states as conditions for current decision-making, which leads to current–future feature entanglement.
- We propose DriveFuture, a future-aware latent world modeling framework that conditions the current decision-making process on future world states to learn planning-oriented *foresight*. It adopts a unified training–inference paradigm, using GT future latent states as conditions during training and predicted future latent states during inference.
- We achieve SOTA results on the public NAVSIM benchmarks [^1], demonstrating the effectiveness of future-conditioned latent world modeling. Notably, as of April 2026, DriveFuture ranks 1st on the [NAVSIM-v2 navhard](https://huggingface.co/spaces/AGC2025/e2e-driving-navhard) leaderboard with 55.5 EPDMS, and achieves SOTA performance on [NAVSIM-v1 navtest](https://huggingface.co/spaces/AGC2024-P/e2e-driving-navtest) with 90.7 PDMS.

## 2 Related Work

### 2.1 World Models for Autonomous Driving

Recent research on world modeling for autonomous driving can be broadly grouped into several directions. The first line of work explicitly models future scene evolution in the observation space, including multi-view driving video generation and world simulation methods [^24] [^25] [^26] [^27] [^28] [^29] [^30] [^31] [^32], which emphasize high-fidelity future rollout, controllable scene evolution, long-horizon generation, and reliable simulation for downstream planning. The second line models the evolution of 3D/4D world states in occupancy space [^33] [^34] [^35] [^36], which characterize future worlds from the perspectives of 3D occupancy world modeling, vision-centric 4D occupancy forecasting, 4D occupancy generation, and unified occupancy-language-action modeling, respectively. The third line performs world modeling in more compact BEV or structured state spaces [^37] [^38], which support future evaluation and trajectory selection through a unified BEV latent space or a BEV world model. More recently, world models have also been increasingly integrated with planner- or VLA-oriented frameworks, as exemplified by LAW [^17], World4Drive [^18], WorldRFT [^19], DriveWorld-VLA [^20], DriveVLA-W0 [^21], and DriveLaW [^22], suggesting a clear trend toward tighter coupling between world modeling, trajectory planning, and scalable autonomous driving systems. Although these methods have demonstrated strong potential for future scene modeling and planning support, they typically involve high modeling complexity and are prone to error accumulation in observation-space or dense-space reconstruction.

### 2.2 End-to-End Autonomous Driving.

End-to-end autonomous driving (E2E-AD) methods map raw sensor observations directly to vehicle controls or planned trajectories. Early E2E-AD methods [^39] [^7] [^40] [^15] [^41] [^42] [^9] [^43] [^13] [^12] [^44] [^45] [^16] [^46] [^47], mainly focus on improving scene representation, multi-modal fusion, and planning stability within unified end-to-end frameworks. More recent approaches increasingly introduce generative planning mechanisms to better capture multi-modal futures and long-horizon behaviors. Representative examples include DiffusionDrive [^8], DIVER [^48], GoalFlow [^10], GuideFlow [^11], and GTRS [^49], which improve trajectory diversity, scoring, controllability, or reasoning ability through diffusion, flow matching, reinforcement learning, or stronger trajectory evaluation. These advances suggest that strong E2E-AD performance increasingly depends on reasoning over possible future evolutions rather than solely reacting to the current scene. However, in most existing systems, future information is introduced only at the output level, e.g., by generating multiple candidate trajectories or scoring future outcomes after the current representation has already been formed. As a result, they improve trajectory generation or selection, but do not fundamentally reshape how the current latent state itself is learned for planning. Instead, DriveFuture leverages future latent states as direct conditions for trajectory planning, explicitly aligning the decision-making process with downstream objectives.

![[methodv8.png|Refer to caption]]

Figure 2: Overview of DriveFuture. Multi-view observations at time t are encoded by a shared Perception Encoder into a scene latent 𝐙 \\mathbf{Z}\_{t}. The Latent Dynamics Predictor conditions on and a tokenised trajectory intent to produce a predicted future latent ^ + T \\hat{\\mathbf{Z}}\_{t+T}. During training, the future observation at t{+}T is encoded by the same Perception Encoder into \\mathbf{Z}\_{t+T}, the Future Alignment Adapter grounds via cross-attention against, yielding the future-aware latent c \\mathbf{Z}^{c}\_{t+T}. LatentAlign anneals the planning condition from towards over training, closing the train–inference gap. At inference, the adapter is bypassed, the Planning Decoder consumes and as dual conditioning contexts to iteratively denoise the ego trajectory.

## 3 Method

The central principle of DriveFuture is to leverage predicted future latents to condition and enhance trajectory planning. As shown in Fig. 2, our framework is built upon three core modules: the Latent Dynamics Predictor (Sec. 3.1), the Future Alignment Adapter (Sec. 3.2), and the Planning Decoder (Sec. 3.3). We further detail the Training (Sec. 3.4) and Inference (Sec. 3.5) pipelines.

### 3.1 Latent Dynamics Predictor

To provide the planning decoder with a compact, action-controllable future signal while avoiding the computational overhead of observation-space prediction, we instantiate a trajectory-conditioned latent predictor in a shared BEV latent space.

#### Trajectory-Conditioned Latent Prediction.

Let $\phi_{\mathrm{enc}}$ denote the shared BEV encoder, which maps the multi-view observation $\mathbf{I}_{t}$ and ego status $\mathbf{s}_{t}$ into a scene latent $\mathbf{Z}_{t}$ comprising $N$ BEV tokens and one ego status token. Let $\phi_{\tau}$ encode an absolute trajectory $\boldsymbol{\tau}\!=\!\{(x_{k},y_{k},\theta_{k})\}_{k=1}^{T}$ into a token sequence $\mathbf{E}_{\tau}\!=\!\phi_{\tau}(\boldsymbol{\tau})$ via a normalised differential representation $(\Delta x,\Delta y,\sin\theta,\cos\theta)$, a linear projection, and a temporal positional embedding. A learnable future query bank $\mathbf{Q}_{f}\!\in\!\mathbb{R}^{K\times d}$ attends to the joint context through a stack of transformer decoder layers $f_{\psi}$, producing

$$
\hat{\mathbf{Z}}_{t+T}\;=\;f_{\psi}\!\left(\mathbf{Q}_{f}\,;\;[\mathbf{Z}_{t}\,\|\,\mathbf{E}_{\tau}]\right).
$$

We refer to $\hat{\mathbf{Z}}_{t+T}$ as the *trajectory-conditioned latent prediction*: it describes the scene latent at horizon $T$ under the hypothesis that the ego enacts $\boldsymbol{\tau}$, and serves as the substrate on which the Future Alignment Adapter and the Planning Decoder operate.

#### Conditioning Source Randomisation.

The predictor in Eq. (1) requires a trajectory $\boldsymbol{\tau}$ that is unavailable at inference time, and classifier-free guidance (CFG) additionally demands calibrated outputs under absent or coarse trajectory inputs. Both requirements are addressed jointly by drawing the trajectory token sequence from three sources during training, with probabilities $\{p_{\mathrm{gt}},p_{\mathrm{kin}},p_{\varnothing}\}$:

$$
\mathbf{E}_{\tau}\;\sim\;\begin{cases}\phi_{\tau}(\boldsymbol{\tau}^{\mathrm{gt}}),&\text{w.p.}\;p_{\mathrm{gt}},\\[2.0pt]
\phi_{\tau}(\boldsymbol{\tau}^{\mathrm{kin}}),&\text{w.p.}\;p_{\mathrm{kin}},\\[2.0pt]
\mathbf{E}^{\varnothing},&\text{w.p.}\;p_{\varnothing},\end{cases}
$$

where $\mathbf{E}^{\varnothing}$ is a learned null token sequence and $\boldsymbol{\tau}^{\mathrm{kin}}$ is a coarse but stable kinematic surrogate produced by constant-acceleration extrapolation from the ego velocity $(v_{x},v_{y})$ and acceleration $(a_{x},a_{y})$:

$$
x_{k}\!=\!v_{x}t_{k}\!+\!\tfrac{1}{2}a_{x}t_{k}^{2},\quad y_{k}\!=\!v_{y}t_{k}\!+\!\tfrac{1}{2}a_{y}t_{k}^{2},\quad\theta_{k}\!=\!\operatorname{atan2}\!\left(v_{y}\!+\!a_{y}t_{k},\,v_{x}\!+\!a_{x}t_{k}\right),
$$

with $t_{k}\!=\!k\Delta t$. The three branches expose the predictor to fine, coarse, and unconditional regimes during training, so that the Tweedie-based and kinematic-based guidance terms used in Sec. 3.5 share a well-trained baseline.

### 3.2 Future Alignment Adapter

The trajectory diffusion loss reaches the latent dynamics predictor only through the planning decoder, which is a weak and indirect signal under which $\hat{\mathbf{Z}}_{t+T}$ is free to drift from the true future semantics in the early stages of training. The Future Alignment Adapter introduces a grounded but lightweight signal that anchors the predicted future latent to the actual future scene, while avoiding dense reconstruction targets that would tie the latent to appearance details unrelated to planning.

#### Cross-Attention Grounding.

During training, the future-step observation $\mathbf{I}_{t+T}$ is passed through the same encoder $\phi_{\mathrm{enc}}$ used for the current frame, with gradients stopped to prevent the future branch from acting as a shortcut, yielding $\mathbf{Z}_{t+T}\!=\!\mathrm{sg}(\phi_{\mathrm{enc}}(\mathbf{I}_{t+T}))$. We use $\mathbf{Z}_{t+T}$ as the empirical proxy for the GT future latent state. The adapter is a single multi-head cross-attention layer with $\hat{\mathbf{Z}}_{t+T}$ as queries and $\mathbf{Z}_{t+T}$ as keys/values:

$$
\tilde{\mathbf{Z}}_{t+T}\;=\;\mathrm{MHA}\!\left(\,\mathrm{LN}(\hat{\mathbf{Z}}_{t+T}),\;\;\mathbf{Z}_{t+T},\;\;\mathbf{Z}_{t+T}\,\right).
$$

The query side preserves the $K$ -token interface used by the Planning Decoder, so the adapter never enlarges the conditioning footprint regardless of the spatial resolution of the future BEV. Because the future observation is unavailable for a subset of the training samples (e.g., at sequence boundaries), let $\mathcal{S}_{+}$ denote the indices that admit a future frame, the planning condition is

$$
\mathbf{Z}_{t+T}^{c,(i)}\;=\;\begin{cases}\tilde{\mathbf{Z}}_{t+T}^{(i)},&i\in\mathcal{S}_{+},\\[2.0pt]
\hat{\mathbf{Z}}_{t+T}^{(i)},&i\notin\mathcal{S}_{+},\end{cases}
$$

which falls back to the unrefined forecast on samples without future observations. At inference time, $\mathbf{Z}_{t+T}$ is unavailable and the adapter is bypassed, so $\mathbf{Z}_{t+T}^{c}=\hat{\mathbf{Z}}_{t+T}$.

#### LatentAlign.

To bridge the train–inference gap caused by the unavailability of the grounded future latent $\mathbf{Z}_{t+T}^{c}$ at inference time, we propose LatentAlign, a sigmoid-based annealing schedule that smoothly transitions the planning condition from the grounded future latent to the self-predicted forecast over the course of training:

$$
\alpha(e)\;=\;1\;-\;\sigma\!\left(\beta\,(e-e_{0})\right),\qquad\tilde{\mathbf{Z}}_{t+T}^{c}\;=\;\alpha(e)\,\mathbf{Z}_{t+T}^{c}\;+\;\bigl(1-\alpha(e)\bigr)\,\hat{\mathbf{Z}}_{t+T},
$$

where $\sigma(\cdot)$ is the logistic function. Early in training ($\alpha\!\to\!1$), the decoder receives grounded future semantics for stable learning, while late in training ($\alpha\!\to\!0$), only the self-predicted forecast is supplied, thereby matching the inference regime.

### 3.3 Planning Decoder

Trajectory planning under closed-loop driving is inherently multi-modal: a deterministic regression head averages distinct intents into a single feasible-but-uninformative output. We therefore realise the planner as a conditional diffusion model so that multi-modal trajectory distributions emerge naturally, and route the future condition into every denoising step so that future semantics directly shape action generation rather than entering only as a post-hoc score.

#### Future-Conditioned Diffusion Transformer.

The planning action is parameterised in the differential trajectory space $\mathbf{a}\!=\!(\Delta x,\Delta y,\sin\theta,\cos\theta)\!\in\!\mathbb{R}^{T\times 4}$. With a denoising diffusion probabilistic model (DDPM) forward process,

$$
\mathbf{a}_{s}\;=\;\sqrt{\bar{\alpha}_{s}}\,\mathbf{a}_{0}\;+\;\sqrt{1-\bar{\alpha}_{s}}\,\boldsymbol{\epsilon},\qquad\boldsymbol{\epsilon}\!\sim\!\mathcal{N}(\mathbf{0},\mathbf{I}),
$$

the noise predictor $\epsilon_{\theta}$ is a transformer in the diffusion-transformer (DiT) family that consumes a noisy action token, a timestep embedding $\mathbf{e}_{s}$, and two conditioning contexts, namely the scene context $\mathbf{C}_{\mathrm{scene}}\!=\![\mathbf{e}_{s}\,\|\,\mathbf{Z}_{t}]$ and the future context $\mathbf{Z}_{t+T}^{c}$:

$$
\hat{\boldsymbol{\epsilon}}\;=\;\epsilon_{\theta}\!\left(\mathbf{a}_{s},\,s,\;\mathbf{C}_{\mathrm{scene}},\;\mathbf{Z}_{t+T}^{c}\right).
$$

Each block first cross-attends to $\mathbf{C}_{\mathrm{scene}}$ and then to $\mathbf{Z}_{t+T}^{c}$, so that environmental geometry is consumed before the future-semantic correction is applied. To preserve a stable warm-start from a planner pretrained without future conditioning, the output projection of the future cross-attention is initialised to zero, making the freshly added module behave as an identity at the start of training and gradually learn to exploit $\mathbf{Z}_{t+T}^{c}$ as optimisation proceeds.

### 3.4 Training

#### Trajectory Diffusion Objective.

The primary supervision is the standard DDPM noise-prediction loss applied with the future-aware predictor in Eq. (8):

$$
\mathcal{L}_{\mathrm{plan}}\;=\;\mathbb{E}_{s,\boldsymbol{\epsilon},\mathbf{a}_{0}}\left\|\boldsymbol{\epsilon}-\epsilon_{\theta}\!\left(\mathbf{a}_{s},s,\mathbf{C}_{\mathrm{scene}},\tilde{\mathbf{Z}}_{t+T}^{c}\right)\right\|_{2}^{2},
$$

where $\tilde{\mathbf{Z}}_{t+T}^{c}$ is the annealed future condition defined in Eq. (6). To prevent the BEV encoder from collapsing into representations that are useful only for trajectory regression, we retain a semantic auxiliary that decodes BEV tokens into a class-wise occupancy map and supervises it against the rasterised semantic ground truth $\mathbf{Y}^{\mathrm{bev}}$,

$$
\mathcal{L}_{\mathrm{bev}}\;=\;\mathrm{CE}\!\left(\hat{\mathbf{Y}}^{\mathrm{bev}},\,\mathbf{Y}^{\mathrm{bev}}\right).
$$

The total objective is

$$
\mathcal{L}\;=\;\lambda_{\mathrm{plan}}\,\mathcal{L}_{\mathrm{plan}}\;+\;\lambda_{\mathrm{bev}}\,\mathcal{L}_{\mathrm{bev}}.
$$

### 3.5 Inference

At inference, the future observation $\mathbf{I}_{t+T}$ is unavailable. The Future Alignment Adapter is therefore bypassed, and the planning condition reduces directly to $\mathbf{Z}_{t+T}^{c}=\hat{\mathbf{Z}}_{t+T}$. This introduces a circular dependency: computing $\hat{\mathbf{Z}}_{t+T}$ via Eq. (1) requires a trajectory intent $\mathbf{E}_{\tau}$, which is itself the output being denoised. Progressive Foresight Guidance (PFG) resolves this dependency by supplying two phase-adaptive surrogate trajectory intents throughout the denoising process. The kinematic extrapolation $\boldsymbol{\tau}^{\mathrm{kin}}$ from Eq. (3) is noise-free but geometrically coarse. A self-consistent estimate $\boldsymbol{\tau}^{\mathrm{tw}}$ is recovered from the current noisy sample via Tweedie’s formula,

$$
\hat{\mathbf{a}}_{0}^{(s)}\;=\;\frac{\mathbf{a}_{s}\;-\;\sqrt{1-\bar{\alpha}_{s}}\,\hat{\boldsymbol{\epsilon}}_{\varnothing}}{\sqrt{\bar{\alpha}_{s}}},\qquad\boldsymbol{\tau}^{\mathrm{tw}}\;=\;\mathrm{cumsum}\!\left(\hat{\mathbf{a}}_{0}^{(s)}\right),
$$

is reliable only once the noise level decreases. Let $r=s/(S\!-\!1)\in[0,1]$ index denoising progress. Three classifier-free guidance branches condition the Latent Dynamics Predictor on different trajectory intents:

$$
\mathbf{Z}_{t+T}^{c,\varnothing}\!=\!f_{\psi}(\mathbf{Z}_{t};\,\mathbf{E}^{\varnothing}),\quad\mathbf{Z}_{t+T}^{c,\mathrm{kin}}\!=\!f_{\psi}(\mathbf{Z}_{t};\,\phi_{\tau}(\boldsymbol{\tau}^{\mathrm{kin}})),\quad\mathbf{Z}_{t+T}^{c,\mathrm{tw}}\!=\!f_{\psi}(\mathbf{Z}_{t};\,\phi_{\tau}(\boldsymbol{\tau}^{\mathrm{tw}})),
$$

yielding three noise predictions $\hat{\boldsymbol{\epsilon}}_{\varnothing},\hat{\boldsymbol{\epsilon}}_{\mathrm{kin}},\hat{\boldsymbol{\epsilon}}_{\mathrm{tw}}$ at each step. The guided noise estimate is a phase-dependent mixture

$$
\hat{\boldsymbol{\epsilon}}\;=\;\hat{\boldsymbol{\epsilon}}_{\varnothing}\;+\;w_{\mathrm{kin}}(r)\!\left(\hat{\boldsymbol{\epsilon}}_{\mathrm{kin}}\!-\!\hat{\boldsymbol{\epsilon}}_{\varnothing}\right)\;+\;w_{\mathrm{tw}}(r)\!\left(\hat{\boldsymbol{\epsilon}}_{\mathrm{tw}}\!-\!\hat{\boldsymbol{\epsilon}}_{\varnothing}\right),
$$

with cosine schedules governed by a shared envelope template

$$
w(r;\rho,\nu,w^{\max})\;=\;w^{\max}\cos\!\Bigl(\tfrac{\pi r}{2\rho}\Bigr)\mathbf{1}_{[r<\rho]}\;+\;\tfrac{w^{\max}}{2}\Bigl[1-\cos\!\Bigl(\tfrac{\pi(r-\nu)}{1-\nu}\Bigr)\Bigr]\mathbf{1}_{[r\geq\nu]},
$$

where $w_{\mathrm{kin}}$ uses the decay form and $w_{\mathrm{tw}}$ uses the rise form. The two curves overlap on $(r_{\mathrm{start}},r_{\mathrm{fade}})$, producing a smooth handover from the inertial prior to the self-consistent surrogate as the trajectory crystallises. Combined with the LatentAlign annealing of Sec. 3.4, the planning condition remains a future-aware latent at both training and inference, anchored to real future evidence in training and to phase-adaptive self-consistent surrogates at inference.

## 4 Experiments

### 4.1 Datasets and Evaluation Metrics

We evaluate DriveFuture on the public NAVSIM benchmark [^1], which is built on OpenScene [^50] and nuPlan [^51] logs for lightweight planning evaluation. We report results on NAVSIM-v1 navtest [^1], NAVSIM-v2 navtest [^52], and NAVSIM-v2 navhard [^52]. navtest measures general planning performance, while navhard emphasizes safety-critical and long-tail scenarios. Following the official protocol, we use PDMS for NAVSIM-v1 and EPDMS for NAVSIM-v2. EPDMS extends PDMS with additional rule- and comfort-related metrics, including driving-direction compliance, traffic-light compliance, lane keeping, and extended comfort. For NAVSIM-v2 navhard, we adopt the official two-stage evaluation, where Stage 2 re-evaluates the planner with synthesized future observations around the Stage-1 endpoint. All results are reported as percentages unless otherwise specified.

Table 1: Comparison with SOTA methods on the NAVSIM-v2 navhard split [^1].

<table><thead><tr><th>Method</th><th>Backbone</th><th>Stage</th><th>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>DDC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>TL <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>LK <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>HC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>EC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>EPDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th></tr><tr><th colspan="13">E2E-based Methods</th></tr></thead><tbody><tr><th rowspan="2">TransFuser <sup><a href="#fn:39">39</a></sup></th><td rowspan="2">ResNet-34</td><td>Stage 1</td><td>96.2</td><td>79.5</td><td>99.1</td><td>99.5</td><td>84.1</td><td>95.1</td><td>94.2</td><td>97.5</td><td>79.1</td><td rowspan="2">23.1</td></tr><tr><td>Stage 2</td><td>77.7</td><td>70.2</td><td>84.2</td><td>98.0</td><td>85.1</td><td>75.6</td><td>45.4</td><td>95.7</td><td>75.9</td></tr><tr><th rowspan="2">DiffusionDrive <sup><a href="#fn:8">8</a></sup></th><td rowspan="2">ResNet-34</td><td>Stage 1</td><td>96.0</td><td>79.7</td><td>97.4</td><td>99.5</td><td>81.3</td><td>93.1</td><td>90.8</td><td>96.8</td><td>73.8</td><td rowspan="2">24.2</td></tr><tr><td>Stage 2</td><td>82.1</td><td>72.2</td><td>88.5</td><td>98.7</td><td>85.1</td><td>78.8</td><td>49.2</td><td>89.3</td><td>71.2</td></tr><tr><th rowspan="2">GuideFlow <sup><a href="#fn:11">11</a></sup></th><td rowspan="2">ResNet-34</td><td>Stage 1</td><td>96.6</td><td>80.5</td><td>96.3</td><td>99.3</td><td>82.3</td><td>94.9</td><td>91.5</td><td>97.7</td><td>67.8</td><td rowspan="2">27.1</td></tr><tr><td>Stage 2</td><td>87.3</td><td>76.7</td><td>88.8</td><td>99.2</td><td>84.3</td><td>85.1</td><td>49.7</td><td>93.1</td><td>44.5</td></tr><tr><th rowspan="2">Senna-E2E <sup><a href="#fn:53">53</a></sup></th><td rowspan="2">ResNet-50</td><td>Stage 1</td><td>95.6</td><td>86.0</td><td>98.9</td><td>99.6</td><td>83.9</td><td>95.1</td><td>95.3</td><td>97.6</td><td>75.6</td><td rowspan="2">27.2</td></tr><tr><td>Stage 2</td><td>78.6</td><td>74.8</td><td>84.8</td><td>98.2</td><td>88.2</td><td>75.7</td><td>46.9</td><td>96.0</td><td>65.8</td></tr><tr><th rowspan="2">DriveSuprim <sup><a href="#fn:54">54</a></sup></th><td rowspan="2">V2-99</td><td>Stage 1</td><td>98.9</td><td>95.1</td><td>99.2</td><td>99.6</td><td>76.1</td><td>99.1</td><td>94.7</td><td>97.6</td><td>54.2</td><td rowspan="2">42.1</td></tr><tr><td>Stage 2</td><td>87.9</td><td>88.8</td><td>89.6</td><td>98.8</td><td>80.3</td><td>86.0</td><td>53.5</td><td>97.1</td><td>56.1</td></tr><tr><th rowspan="2">ZTRS <sup><a href="#fn:55">55</a></sup></th><td rowspan="2">V2-99</td><td>Stage 1</td><td>98.9</td><td>97.6</td><td>100.0</td><td>100.0</td><td>66.7</td><td>98.9</td><td>96.2</td><td>96.7</td><td>44.0</td><td rowspan="2">48.1</td></tr><tr><td>Stage 2</td><td>91.1</td><td>90.4</td><td>95.8</td><td>99.0</td><td>63.6</td><td>89.8</td><td>60.4</td><td>97.6</td><td>66.1</td></tr><tr><th rowspan="2">GTRS-E <sup><a href="#fn:49">49</a></sup></th><td rowspan="2">V2-99+EVA-ViT-L+ViT-L</td><td>Stage 1</td><td>98.9</td><td>99.3</td><td>99.8</td><td>99.8</td><td>75.2</td><td>98.4</td><td>96.0</td><td>97.6</td><td>51.6</td><td rowspan="2">49.4</td></tr><tr><td>Stage 2</td><td>92.3</td><td>93.3</td><td>94.6</td><td>99.2</td><td>73.1</td><td>91.2</td><td>53.9</td><td>96.7</td><td>56.8</td></tr><tr><th rowspan="2">SimScale <sup><a href="#fn:56">56</a></sup></th><td rowspan="2">V2-99</td><td>Stage 1</td><td>99.6</td><td>99.1</td><td>99.9</td><td>100.0</td><td>69.6</td><td>99.6</td><td>95.8</td><td>95.6</td><td>28.4</td><td rowspan="2">53.2</td></tr><tr><td>Stage 2</td><td>94.5</td><td>94.2</td><td>95.8</td><td>99.2</td><td>75.8</td><td>92.8</td><td>60.1</td><td>96.1</td><td>43.2</td></tr><tr><th rowspan="2">DrivoR <sup><a href="#fn:57">57</a></sup></th><td rowspan="2">ViT-S</td><td>Stage 1</td><td>99.1</td><td>98.2</td><td>99.3</td><td>99.8</td><td>75.4</td><td>98.7</td><td>94.9</td><td>97.6</td><td>70.2</td><td rowspan="2">54.6</td></tr><tr><td>Stage 2</td><td>92.3</td><td>91.6</td><td>97.3</td><td>99.1</td><td>75.7</td><td>90.6</td><td>56.1</td><td>98.4</td><td>44.7</td></tr><tr><th colspan="13">VLA-based Methods</th></tr><tr><th rowspan="2">SpanVLA <sup><a href="#fn:58">58</a></sup></th><td rowspan="2">Qwen2.5-VL-3B</td><td>Stage 1</td><td>98.4</td><td>94.3</td><td>97.8</td><td>99.9</td><td>85.7</td><td>97.2</td><td>94.2</td><td>97.6</td><td>72.1</td><td rowspan="2">40.1</td></tr><tr><td>Stage 2</td><td>86.9</td><td>84.3</td><td>87.1</td><td>98.2</td><td>85.5</td><td>82.7</td><td>62.3</td><td>96.8</td><td>67.4</td></tr><tr><th rowspan="2">DiffVLA <sup><a href="#fn:59">59</a></sup></th><td rowspan="2">V2-99 + ViT-L/14</td><td>Stage 1</td><td>95.7</td><td>99.2</td><td>100.0</td><td>100.0</td><td>85.9</td><td>96.4</td><td>97.1</td><td>95.0</td><td>84.2</td><td rowspan="2">45.0</td></tr><tr><td>Stage 2</td><td>81.2</td><td>88.8</td><td>94.6</td><td>99.0</td><td>86.0</td><td>76.4</td><td>59.8</td><td>98.6</td><td>80.4</td></tr><tr><th colspan="13">World-Model-based Methods</th></tr><tr><th rowspan="2">MindDrive <sup><a href="#fn:14">14</a></sup></th><td rowspan="2">ResNet-34</td><td>Stage 1</td><td>96.1</td><td>86.0</td><td>98.8</td><td>99.3</td><td>83.3</td><td>95.6</td><td>94.4</td><td>97.6</td><td>74.7</td><td rowspan="2">30.9</td></tr><tr><td>Stage 2</td><td>82.6</td><td>79.1</td><td>86.4</td><td>98.0</td><td>85.3</td><td>79.4</td><td>49.2</td><td>96.5</td><td>71.0</td></tr><tr><th rowspan="2">World4Drive <sup><a href="#fn:18">18</a></sup></th><td rowspan="2">ResNet-34</td><td>Stage 1</td><td>97.3</td><td>89.1</td><td>97.6</td><td>99.7</td><td>60.5</td><td>96.8</td><td>87.7</td><td>93.1</td><td>60.0</td><td rowspan="2">34.9</td></tr><tr><td>Stage 2</td><td>91.4</td><td>82.0</td><td>91.0</td><td>98.5</td><td>53.1</td><td>90.6</td><td>52.3</td><td>93.3</td><td>62.8</td></tr><tr><th></th><td></td><td>Stage 1</td><td>99.8</td><td>99.8</td><td>100</td><td>99.6</td><td>85.7</td><td>99.8</td><td>98.7</td><td>97.6</td><td>66.2</td><td></td></tr><tr><th>DriveFuture</th><td>V2-99</td><td>Stage 2</td><td>90.6</td><td>87.5</td><td>94.1</td><td>99.1</td><td>84.6</td><td>88.8</td><td>58.3</td><td>93.5</td><td>45.6</td><td><math><semantics><mn>55.5</mn> <annotation>\boldsymbol{55.5}</annotation></semantics></math></td></tr></tbody></table>

Table 2: Comparison with SOTA methods on the NAVSIM-v2 navtest split [^1]. EPDMS <sup>∗</sup> denotes results computed with the original NAVSIM-v2 evaluation code before the human-behavior filtering fix, while EPDMS denotes the corrected official implementation.

<table><thead><tr><th>Method</th><th>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>DDC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>TL <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>LK <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>HC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>EC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>EPDMS <math><semantics><mrow><msup><mo>∗</mo></msup> <mo>↑</mo></mrow> <annotation>{}^{*}\uparrow</annotation></semantics></math></th><th>EPDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th></tr><tr><th colspan="12">E2E-based Methods</th></tr></thead><tbody><tr><th>TransFuser <sup><a href="#fn:39">39</a></sup></th><td>96.9</td><td>89.9</td><td>97.8</td><td>99.7</td><td>87.1</td><td>95.4</td><td>92.7</td><td>98.3</td><td>87.2</td><td>76.7</td><td>–</td></tr><tr><th>DiffusionDrive <sup><a href="#fn:8">8</a></sup></th><td>98.2</td><td>95.9</td><td>99.4</td><td>99.8</td><td>87.5</td><td>97.3</td><td>96.8</td><td>98.3</td><td>87.7</td><td>–</td><td>84.5</td></tr><tr><th>Hydra-MDP++ <sup><a href="#fn:60">60</a></sup></th><td>97.2</td><td>97.5</td><td>99.4</td><td>99.6</td><td>83.1</td><td>96.5</td><td>94.4</td><td>98.2</td><td>70.9</td><td>81.4</td><td>–</td></tr><tr><th>DriveSuprim <sup><a href="#fn:54">54</a></sup></th><td>97.5</td><td>96.5</td><td>99.4</td><td>99.6</td><td>88.4</td><td>96.6</td><td>95.5</td><td>98.3</td><td>77.0</td><td>83.1</td><td>–</td></tr><tr><th>ARTEMIS <sup><a href="#fn:61">61</a></sup></th><td>98.3</td><td>95.1</td><td>98.6</td><td>99.8</td><td>81.5</td><td>97.4</td><td>96.5</td><td>98.3</td><td>98.3</td><td>83.1</td><td>–</td></tr><tr><th>DiffusionDriveV2 <sup><a href="#fn:62">62</a></sup></th><td>97.7</td><td>96.6</td><td>99.2</td><td>99.8</td><td>88.9</td><td>97.2</td><td>96.0</td><td>97.8</td><td>91.0</td><td>85.5</td><td>87.5</td></tr><tr><th colspan="12">VLA-based Methods</th></tr><tr><th>DriveWorld-VLA <sup><a href="#fn:20">20</a></sup></th><td>98.6</td><td>99.1</td><td>99.6</td><td>99.8</td><td>87.4</td><td>97.9</td><td>97.0</td><td>97.8</td><td>78.6</td><td>–</td><td>86.8</td></tr><tr><th>DriveVLA-W0 <sup><a href="#fn:21">21</a></sup></th><td>98.5</td><td>99.1</td><td>98.0</td><td>99.7</td><td>86.4</td><td>98.1</td><td>93.2</td><td>97.9</td><td>58.9</td><td>–</td><td>86.1</td></tr><tr><th>Recogdrive <sup><a href="#fn:16">16</a></sup></th><td>98.3</td><td>95.2</td><td>98.3</td><td>99.8</td><td>87.1</td><td>97.5</td><td>96.6</td><td>99.5</td><td>86.5</td><td>–</td><td>83.6</td></tr><tr><th colspan="12">World-Model-based Methods</th></tr><tr><th>Latent-WAM <sup><a href="#fn:63">63</a></sup></th><td>98.1</td><td>97.3</td><td>99.6</td><td>99.8</td><td>87.7</td><td>97.3</td><td>97.6</td><td>98.1</td><td>87.3</td><td>–</td><td>89.3</td></tr><tr><th>DriveFuture</th><td>98.8</td><td>99.1</td><td>99.6</td><td>99.9</td><td>86.6</td><td>98.4</td><td>96.4</td><td>98.3</td><td>74.8</td><td>86.4</td><td>89.9</td></tr></tbody></table>

### 4.2 Implementation Details

<table><tbody><tr><td>Method</td><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Conf.<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>Human</td><td>100</td><td>100</td><td>100</td><td>99.9</td><td>87.5</td><td>94.8</td></tr><tr><td>Constant Velocity</td><td>69.9</td><td>58.8</td><td>49.3</td><td>100</td><td>49.3</td><td>21.6</td></tr><tr><td colspan="7">E2E-based Methods</td></tr><tr><td>VADv2 <sup><a href="#fn:15">15</a></sup></td><td>97.2</td><td>89.1</td><td>91.6</td><td>100</td><td>76.0</td><td>80.9</td></tr><tr><td>TransFuser <sup><a href="#fn:39">39</a></sup></td><td>97.7</td><td>92.8</td><td>92.8</td><td>100</td><td>79.2</td><td>84.0</td></tr><tr><td>UniAD <sup><a href="#fn:7">7</a></sup></td><td>97.8</td><td>91.9</td><td>92.9</td><td>100</td><td>78.8</td><td>83.4</td></tr><tr><td>PARA-Drive <sup><a href="#fn:64">64</a></sup></td><td>97.9</td><td>92.4</td><td>93.0</td><td>99.8</td><td>79.3</td><td>84.0</td></tr><tr><td>DRAMA <sup><a href="#fn:65">65</a></sup></td><td>98.0</td><td>93.1</td><td>94.8</td><td>100</td><td>80.1</td><td>85.5</td></tr><tr><td>GoalFlow <sup><a href="#fn:10">10</a></sup></td><td>98.3</td><td>93.8</td><td>94.3</td><td>100</td><td>79.8</td><td>85.7</td></tr><tr><td>Hydra-MDP <sup><a href="#fn:43">43</a></sup></td><td>98.3</td><td>96.0</td><td>94.6</td><td>100</td><td>78.7</td><td>86.5</td></tr><tr><td>ARTEMIS <sup><a href="#fn:61">61</a></sup></td><td>98.3</td><td>95.1</td><td>94.3</td><td>100</td><td>81.4</td><td>87.0</td></tr><tr><td>DiffusionDrive <sup><a href="#fn:8">8</a></sup></td><td>98.2</td><td>96.2</td><td>94.7</td><td>100</td><td>82.2</td><td>88.1</td></tr><tr><td>DIVER <sup><a href="#fn:48">48</a></sup></td><td>98.5</td><td>96.5</td><td>94.9</td><td>100</td><td>82.6</td><td>88.3</td></tr><tr><td>DriveSuprim <sup><a href="#fn:54">54</a></sup></td><td>97.8</td><td>97.3</td><td>93.6</td><td>100</td><td>86.7</td><td>89.9</td></tr><tr><td>GoalFlow <sup><a href="#fn:10">10</a></sup></td><td>98.4</td><td>98.3</td><td>94.6</td><td>100</td><td>85.0</td><td>90.3</td></tr><tr><td colspan="7">VLA-based Methods</td></tr><tr><td>AutoVLA <sup><a href="#fn:66">66</a></sup></td><td>98.4</td><td>95.6</td><td>98.0</td><td>99.9</td><td>81.9</td><td>89.1</td></tr><tr><td>Recogdrive <sup><a href="#fn:16">16</a></sup></td><td>98.2</td><td>97.8</td><td>95.2</td><td>99.8</td><td>83.5</td><td>89.6</td></tr><tr><td>DriveVLA-W0 <sup><a href="#fn:21">21</a></sup></td><td>98.7</td><td>99.1</td><td>95.3</td><td>99.3</td><td>83.3</td><td>90.2</td></tr><tr><td>DriveWorld-VLA <sup><a href="#fn:20">20</a></sup></td><td>99.1</td><td>98.2</td><td>96.1</td><td>100</td><td>85.9</td><td>91.3</td></tr><tr><td colspan="7">World-Model-based Methods</td></tr><tr><td>LAW <sup><a href="#fn:17">17</a></sup></td><td>96.4</td><td>95.4</td><td>88.7</td><td>99.9</td><td>81.7</td><td>84.6</td></tr><tr><td>World4Drive <sup><a href="#fn:18">18</a></sup></td><td>97.4</td><td>94.3</td><td>92.8</td><td>100</td><td>79.9</td><td>85.1</td></tr><tr><td>WoTE <sup><a href="#fn:38">38</a></sup></td><td>98.5</td><td>96.8</td><td>94.9</td><td>99.9</td><td>81.9</td><td>88.3</td></tr><tr><td>WorldRFT <sup><a href="#fn:19">19</a></sup></td><td>97.8</td><td>96.8</td><td>94.0</td><td>100</td><td>81.7</td><td>87.8</td></tr><tr><td>DriveLaW <sup><a href="#fn:22">22</a></sup></td><td>99.0</td><td>97.1</td><td>96.7</td><td>100</td><td>81.3</td><td>89.1</td></tr><tr><td>DriveFuture</td><td>98.8</td><td>99.1</td><td>95.4</td><td>100</td><td>84.2</td><td>90.7</td></tr></tbody></table>

Table 3: Comparison with SOTA methods on the NAVSIM-v1 navtest split [^1].

DriveFuture is built on a TransFuser-based [^67] planning pipeline, trained on the NavTrain split for 100 epochs with Adam at a learning rate of $10^{-4}$, and uses a GTRS-Dense scorer [^49] to select multi-modal trajectories. We use two temporal frames, front and rear camera streams, image resolution $2048\times 512$, and a BEV token grid with $16\times 64$ anchors. The planner predicts an $8$ -step trajectory over a $4$ second horizon with a $0.5$ second interval. The diffusion planning head contains $5$ transformer layers and samples $100$ trajectory proposals by default. Training is conducted on 8 NVIDIA 5090 GPUs with mixed bf16 precision, gradient clipping of $1.0$, and loss weights $\lambda_{\mathrm{plan}}=\lambda_{\mathrm{bev}}=10$. More implementation details are provided in the Supplementary Material.

### 4.3 Main Results

NAVSIM-v2 navhard. As shown in Table 1, DriveFuture achieves the best overall result with 55.5 EPDMS, outperforming DrivoR [^57] (54.6 EPDMS), DiffVLA [^59] (45.0 EPDMS), and the strongest prior world-model method World4Drive [^18] (34.9 EPDMS). Notably, DriveFuture demonstrates consistent improvements across compliance and safety-related metrics, including 99.1 DAC and 95.4 TTC. These results indicate that conditioning the current latent state on future world states not only improves overall performance but also enhances safety-critical behavior under the challenging two-stage evaluation.

NAVSIM-v2 navtest. As shown in Table 2, DriveFuture achieves the best overall result of 89.9 corrected EPDMS. It surpasses the E2E-AD method DiffusionDriveV2 [^62] with 87.5, the VLA method DriveWorld-VLA [^20] with 86.8, and the world-model method Latent-WAM [^63] with 89.3. These gains indicate that DriveFuture improves planning robustness under the stricter NAVSIM-v2 protocol by explicitly conditioning current representations on future latent states.

NAVSIM-v1 navtest. As shown in Table 4.2, DriveFuture achieves 90.7 PDMS. It surpasses GoalFlow [^10] with 90.3 and DriveLaW [^22] with 89.1, while remaining competitive with the best VLA method DriveWorld-VLA [^20] at 91.3. This demonstrates that future-conditioned latent modeling better aligns world representations with planning requirements.

Table 4: Unified ablation study on future supervision and PFG heuristic guidance on NAVSIM-v2 navhard [^52]. A checkmark denotes the option is enabled in that configuration. FF denotes whether future frames are considered during training, Impl. denotes the proposed implicit future constraint, MSE denotes direct mean-squared-error supervision, KS denotes kinematics-based heuristic guidance, and GT denotes GT trajectory guidance. All ablations are conducted without GTRS-Dense scorer [^49].

<table><thead><tr><th colspan="3">Future Supervision  </th><th colspan="2">Heuristic Guidance  </th><th rowspan="2">EPDMS</th><th rowspan="2">NC</th><th rowspan="2">DAC</th><th rowspan="2">TTC</th><th rowspan="2">LK</th><th rowspan="2">HC</th><th rowspan="2">EC</th></tr><tr><th>FF</th><th>Impl</th><th>MSE</th><th>KS</th><th>GT</th></tr></thead><tbody><tr><td></td><td></td><td></td><td>✓</td><td>✓</td><td>30.9</td><td>81.8</td><td>72.6</td><td>79.7</td><td>46.4</td><td>95.7</td><td>71.7</td></tr><tr><td>✓</td><td></td><td>✓</td><td>✓</td><td>✓</td><td>32.1</td><td>81.7</td><td>77.0</td><td>78.4</td><td>49.5</td><td>96.7</td><td>69.9</td></tr><tr><td>✓</td><td>✓</td><td></td><td></td><td></td><td>32.0</td><td>81.1</td><td>75.2</td><td>78.1</td><td>47.6</td><td>97.0</td><td>75.9</td></tr><tr><td>✓</td><td>✓</td><td></td><td>✓</td><td>✓</td><td>34.6</td><td>82.3</td><td>78.8</td><td>79.6</td><td>47.6</td><td>97.0</td><td>75.9</td></tr></tbody></table>

### 4.4 Ablation Study

Effect of Future Supervision. As shown in Table 4, future-aware conditioning improves robustness, raising EPDMS from 30.9 to 34.6, with clear gains in drivable-area compliance and lane keeping. Direct future-latent MSE supervision is less effective, achieving 32.1 EPDMS, suggesting that future information is better used as an implicit planning condition than as a feature-level regression target.

Effect of Heuristic Guidance. Table 4 evaluates the training sources used by PFG. When the kinematic branch and GT-guided training are removed, the model still produces reasonable proposals, but the final EPDMS drops from 34.6 to 32.0 and the largest deficits appear in Stage-2 DAC and TTC. This indicates that dual-source guidance is not only an inference heuristic: it also improves the quality of the learned future-conditioned latent during training and leads to more stable behavior in hard interactive rollouts.

| $t_{f}$ | EPDMS | $q_{s}$ | EPDMS | $e_{o}$ | EPDMS |
| --- | --- | --- | --- | --- | --- |
| 0.5 | 30.2 | 4 | 28.2 | 0.75 | 29.2 |
| 1.0 | 31.2 | 16 | 34.6 | 0.83 | 34.6 |
| 1.5 | 34.6 | 64 | 33.7 | 0.95 | 28.9 |

Table 5: Sensitivity analysis of three hyper-parameters on NAVSIM-v2 navhard [^52]. We report the effect of future horizon $t_{f}$, query size $q_{s}$, and initial guidance parameter $e_{o}$ on final EPDMS. The best result in each group is highlighted in bold. Here $t_{f}$ denotes the $t_{f}$ -th future frame, $q_{s}$ denotes the number of queries in the world model, and $e_{o}$ denotes the inflection point in the annealing schedule. All ablations are conducted without GTRS-Dense scorer [^49].

Sensitivity of Hyper-parameters. Table 5 shows that moderate hyper-parameter settings are consistently preferred. The best EPDMS is achieved with $t_{f}=1.5$, $q_{s}=16$, and $e_{o}=0.83$ in their respective groups. Smaller or larger values reduce robustness, indicating that effective planning requires a balanced future horizon, query budget, and guidance strength.

### 4.5 Qualitative Analysis

We analyze representative challenging cases from the NAVSIM-v2 navhard split, as shown in Fig. 4. Compared with World4Drive [^18], DriveFuture performs better in common failure cases, including collision, inefficient, and braking. This shows that conditioning current decision representations on future states is more effective than using future states merely as prediction targets.

![[visv2.png|Refer to caption]]

Figure 3: Visual comparison between latent world model World4Drive 18 and our DriveFuture across multiple driving scenarios from NAVSIM-v2 navhard 1.

## 5 Conclusion

In this work, we propose DriveFuture, a future-aware latent world modeling framework for autonomous driving. Unlike existing methods that primarily simulate future states, DriveFuture uses future world states as explicit conditions to shape current latent representations for planning. It consists of a Latent Dynamics Predictor for compact future latent prediction, and a Future Alignment Adapter for extracting planning-relevant information from GT future observations. DriveFuture achieves SOTA performance on public NAVSIM-v1/v2 benchmarks, validating the effectiveness of future-conditioned latent world modeling for autonomous driving.

Limitation and Future Work. DriveFuture depends on predicted future latents during inference, so inaccurate future prediction may affect planning in uncertain scenarios. It also mainly focuses on near-future conditioning. In the future, we will explore uncertainty-aware and longer-horizon future modeling to improve robustness and generalization.

## References

## Appendix A Experiment Details

### A.1 Datasets and Benchmarks

Datasets. We train and evaluate DriveFuture on the nuPlan (OpenScene) data used by the public NAVSIM benchmark [^1]. The nuPlan dataset [^51] provides large-scale real-world autonomous driving logs with multi-camera observations, ego states, HD-map information, object annotations, and human driving trajectories. OpenScene [^50] serves as a compact redistribution of nuPlan, and NAVSIM further organizes these logs into a lightweight planning-oriented benchmark. In our experiments, NAVSIM provides real-world driving scenes, historical ego states, map context, and future trajectories for training and evaluating end-to-end planners.

Benchmarks. Following the official NAVSIM protocol, we report results on three settings: NAVSIM-v1 navtest, NAVSIM-v2 navtest, and NAVSIM-v2 navhard. The navtest split evaluates general planning ability on standard real-world driving scenes, while navhard focuses on more challenging and safety-critical long-tail scenarios. NAVSIM-v1 uses the Predictive Driver Model Score (PDMS) to assess key aspects of driving behavior, including safety, feasibility, comfort, and progress. NAVSIM-v2 adopts the Extended Predictive Driver Model Score (EPDMS), which further incorporates rule- and comfort-related metrics such as driving-direction compliance, traffic-light compliance, lane keeping, and extended comfort. For NAVSIM-v2 navhard, we follow the official two-stage protocol: Stage 1 evaluates the planner on the original observation, and Stage 2 re-evaluates it with synthesized future observations around the Stage-1 endpoint. All reported scores are percentages, and higher values indicate better planning quality.

### A.2 Evaluation Metrics

PDM and PDMS. Following NAVSIM [^1], the Predictive Driver Model (PDM) refers to the rule-based planner used to generate and score trajectory proposals. Given an observation $o_{t}$ and a set of candidate trajectories $\mathcal{T}=\{\tau_{i}\}_{i=1}^{N}$, PDM selects the trajectory with the highest PDM score:

$$
\tau^{\star}=\arg\max_{\tau_{i}\in\mathcal{T}}\mathrm{Score}_{\mathrm{PDM}}(\tau_{i}),
$$

where $\mathrm{Score}_{\mathrm{PDM}}$ evaluates each candidate by simulating it and aggregating safety, feasibility, progress, and comfort terms. NAVSIM adopts this PDM-style scoring function as the benchmark metric, namely the Predictive Driver Model Score (PDMS).

For NAVSIM-v1, the primary metric is PDMS. The evaluation first unrolls the predicted trajectory in a non-reactive simulator and computes normalized subscores in $[0,1]$. These subscores are then divided into hard penalties and soft objectives. The hard penalties are no-at-fault collision (NC) and drivable-area compliance (DAC), while the soft objectives are ego progress (EP), time-to-collision (TTC), and comfort (C). The final score is computed as

$$
\mathrm{PDMS}=\underbrace{\prod_{m\in\{\mathrm{NC},\mathrm{DAC}\}}s_{m}}_{\text{hard penalties}}\cdot\underbrace{\frac{\sum_{w\in\{\mathrm{EP},\mathrm{TTC},\mathrm{C}\}}\alpha_{w}s_{w}}{\sum_{w\in\{\mathrm{EP},\mathrm{TTC},\mathrm{C}\}}\alpha_{w}}}_{\text{weighted soft score}},
$$

where $s_{m}$ denotes the normalized subscore and NAVSIM uses $\alpha_{\mathrm{EP}}=5$, $\alpha_{\mathrm{TTC}}=5$, and $\alpha_{\mathrm{C}}=2$. The multiplicative penalty term makes safety-critical violations dominate the final score: if the ego trajectory causes an at-fault collision, $s_{\mathrm{NC}}$ becomes zero; if the trajectory leaves the drivable area, $s_{\mathrm{DAC}}$ becomes zero. Collisions with static objects are assigned a softer penalty in NAVSIM, while non-at-fault collisions under the non-reactive setting are ignored.

The soft terms measure driving quality when the trajectory is admissible. The ego-progress subscore $s_{\mathrm{EP}}$ is computed as the ratio between the ego progress along the route centerline and a safe upper-bound progress estimated by the privileged PDM-Closed planner, clipped to $[0,1]$. The time-to-collision subscore $s_{\mathrm{TTC}}$ is initialized to one and is set to zero if, at any simulation step within the 4-second horizon, the projected ego motion violates the predefined TTC safety threshold with respect to surrounding vehicles. The comfort subscore $s_{\mathrm{C}}$ evaluates whether the trajectory satisfies acceleration and jerk thresholds. Therefore, PDMS rewards trajectories that simultaneously remain collision-free, stay on the road, make sufficient route progress, preserve safety margins, and maintain smooth motion.

For NAVSIM-v2, the benchmark extends PDMS to the Extended Predictive Driver Model Score (EPDMS) by adding more fine-grained rule-compliance and comfort terms. In addition to NC, DAC, EP, and TTC, EPDMS includes driving-direction compliance (DDC), traffic-light compliance (TLC), lane keeping (LK), history comfort (HC), and extended comfort (EC). Following the NAVSIM-v2 protocol [^52], the single-stage extended score can be written as

$$
\mathrm{EPDMS}=\underbrace{\prod_{m\in\mathcal{M}_{\mathrm{pen}}}f_{m}(\tau_{\mathrm{agent}},\tau_{\mathrm{human}})}_{\text{penalty terms}}\cdot\underbrace{\frac{\sum_{m\in\mathcal{M}_{\mathrm{avg}}}\beta_{m}f_{m}(\tau_{\mathrm{agent}},\tau_{\mathrm{human}})}{\sum_{m\in\mathcal{M}_{\mathrm{avg}}}\beta_{m}}}_{\text{weighted average terms}},
$$

where $\mathcal{M}_{\mathrm{pen}}=\{\mathrm{NC},\mathrm{DAC},\mathrm{DDC},\mathrm{TLC}\}$ and $\mathcal{M}_{\mathrm{avg}}=\{\mathrm{EP},\mathrm{TTC},\mathrm{LK},\mathrm{HC},\mathrm{EC}\}$. The default weights are $\beta_{\mathrm{EP}}=5$, $\beta_{\mathrm{TTC}}=5$, and $\beta_{\mathrm{LK}}=\beta_{\mathrm{HC}}=\beta_{\mathrm{EC}}=2$. Here, $f_{m}(\tau_{\mathrm{agent}},\tau_{\mathrm{human}})$ denotes the filtered subscore for metric $m$. This filtering mechanism ignores a rule violation if the same violation is also committed by the human trajectory in the corresponding scene, reducing false penalties caused by annotation noise or contextually necessary maneuvers. DDC evaluates whether the ego vehicle follows the legal driving direction, TLC checks obedience to traffic lights, LK evaluates lane-keeping behavior, and HC/EC measure trajectory smoothness under the extended NAVSIM-v2 protocol.

We distinguish between EPDMS <sup>∗</sup> and EPDMS when reporting NAVSIM-v2 results. EPDMS <sup>∗</sup> denotes scores computed with the earlier NAVSIM-v2 evaluation implementation before the human-behavior filtering fix was adopted in the official leaderboard. It preserves the same extended metric set as Eq. (18), but may penalize the agent for violations that are also present in the human reference behavior. EPDMS denotes the corrected official implementation, where the human-filtered subscores $f_{m}(\cdot)$ are used consistently. Therefore, EPDMS is the primary metric for final comparison, while EPDMS <sup>∗</sup> is reported only for compatibility with earlier results computed using the legacy code.

For NAVSIM-v2 navhard, evaluation follows the pseudo-simulation protocol with two stages. Stage 1 scores the planner from the original real observation, producing $s_{1}$. Stage 2 evaluates the planner on a set of pre-generated synthetic observations around plausible future ego states, producing scores $\{s_{2}^{(i)}\}_{i=1}^{K}$. These Stage-2 scores are aggregated by a Gaussian-weighted average according to the distance between each synthetic start point $x_{i}$ and the Stage-1 endpoint $\hat{x}$:

$$
s_{2}=\sum_{i=1}^{K}\hat{w}_{i}s_{2}^{(i)},\quad\hat{w}_{i}=\frac{w_{i}}{\sum_{j=1}^{K}w_{j}},\quad w_{i}=\exp\left(-\frac{\lVert x_{i}-\hat{x}\rVert_{2}^{2}}{2\sigma^{2}}\right).
$$

The final navhard score multiplies the original-observation score and the aggregated synthetic-observation score:

$$
\mathrm{EPDMS}_{\mathrm{navhard}}=s_{1}\cdot s_{2}.
$$

This two-stage aggregation evaluates both immediate planning quality and robustness to future observation shifts, making EPDMS on navhard stricter than single-stage NAVSIM-v1 PDMS or NAVSIM-v2 navtest evaluation.

## Appendix B More Details on DriveFuture

This section provides additional technical details omitted from the main paper due to space constraints. We first give a more explicit formulation of DriveFuture as a future-aware latent world model for autonomous driving planning, then expand the derivation of the Progressive Foresight Guidance (PFG) sampler, and finally discuss why the proposed training–inference design is better suited to planning than conventional future-prediction objectives. DriveFuture builds on the NAVSIM planning stack [^1] [^52] and is motivated by recent work on latent world models [^17] [^18] [^19] [^38] [^20] [^21] [^22] [^63] and trajectory diffusion planners [^8] [^48].

### B.1 More Details of Notation

Let $\mathbf{I}_{t}$ denote the multi-view camera observations and $\mathbf{s}_{t}$ the ego-status at scene time $t$. DriveFuture first maps $(\mathbf{I}_{t},\mathbf{s}_{t})$ to a compact current latent representation

$$
\mathbf{Z}_{t}=\phi_{\mathrm{enc}}(\mathbf{I}_{t},\mathbf{s}_{t})\in\mathbb{R}^{N\times d},\quad N=65,\quad d=256,
$$

where $64$ tokens correspond to spatial BEV anchors and the final token encodes the ego-status. The planner predicts a trajectory

$$
\tau=\{(x_{k},y_{k},\theta_{k})\}_{k=1}^{T},\quad T=8,
$$

covering a $4$ second horizon at $0.5$ second intervals. A trajectory tokenizer $\phi_{\tau}$ maps an absolute trajectory to $T$ tokens using normalised finite differences and sine/cosine heading embeddings:

$$
\displaystyle\Delta x_{k}
$$
 
$$
\displaystyle=x_{k}-x_{k-1},\quad\Delta y_{k}=y_{k}-y_{k-1},\quad x_{0}=y_{0}=0,
$$
$$
\displaystyle\bar{\Delta x}_{k}
$$
 
$$
\displaystyle=\frac{\Delta x_{k}-\mu_{x}}{\sigma_{x}},\quad\bar{\Delta y}_{k}=\frac{\Delta y_{k}-\mu_{y}}{\sigma_{y}},
$$
$$
\displaystyle\phi_{\tau}(\boldsymbol{\tau})_{k}
$$
 
$$
\displaystyle=\operatorname{LN}\!\left(W_{\tau}[\bar{\Delta x}_{k},\bar{\Delta y}_{k},\sin\theta_{k},\cos\theta_{k}]^{\top}+\mathbf{p}_{k}\right).
$$

The world model then predicts a compact foresight latent:

$$
\hat{\mathbf{Z}}_{t+T}=f_{\psi}\bigl(\mathbf{Z}_{t};\mathbf{E}_{\tau}\bigr)\in\mathbb{R}^{K\times d},\quad K=16.
$$

The planning decoder is a diffusion transformer that predicts the noise of a noised trajectory sample while cross-attending both the current latent and the foresight latent:

$$
\epsilon_{\theta}=\epsilon_{\theta}\bigl(\mathbf{a}_{s},s,\mathbf{Z}_{t},\hat{\mathbf{Z}}_{t+T}\bigr).
$$

Throughout the appendix, $(\mathbf{I}_{t},\mathbf{s}_{t})$ denotes the current multi-camera observation and ego-status, and $\mathbf{Z}_{t}\in\mathbb{R}^{65\times 256}$ denotes the corresponding current scene latent with $64$ BEV tokens plus one status token. The planned trajectory $\tau$ contains $T=8$ poses over a $4$ second horizon, and $\mathbf{E}_{\tau}=\phi_{\tau}(\boldsymbol{\tau})$ denotes its tokenized representation. The latent dynamics predictor $f_{\psi}$ maps the current scene and trajectory intent into a compact foresight latent $\hat{\mathbf{Z}}_{t+T}$ with $K=16$ tokens. During training, $\mathbf{Z}_{t+T}$ is the stop-gradient future BEV token bank, and $\tilde{\mathbf{Z}}_{t+T}$ is the Future Alignment Adapter output obtained by querying this bank with the predicted foresight latent. At inference, $\mathbf{E}^{\varnothing}$ denotes the null trajectory token for unconditional CFG, $\boldsymbol{\tau}^{\mathrm{kin}}$ denotes the constant-acceleration rollout, $\mathbf{a}_{s}$ is the noisy trajectory representation at diffusion step $s$, and $\epsilon_{\theta}$ is the denoiser prediction.

### B.2 More Details on Future-Aware Latent World Modeling

Most latent world models for autonomous driving optimize a future-prediction objective of the form

$$
\min_{\psi}\;\mathbb{E}\left[D\bigl(f_{\psi}(\mathbf{Z}_{t},a_{t}),\mathbf{Z}^{\star}_{t+T}\bigr)\right],
$$

where $D$ is a feature-space distance or reconstruction loss. Such objectives are useful for modeling environment dynamics, but they do not necessarily enforce that the future latent contains information that is useful for choosing a trajectory. DriveFuture changes the role of future information: instead of only asking whether the future latent can be predicted, it asks whether the future latent improves denoising of the planned trajectory.

This leads to a planning-oriented objective:

$$
\min_{\theta,\psi,\omega}\;\mathbb{E}_{\tau,s,\epsilon}\left[\left\|\epsilon-\epsilon_{\theta}\bigl(\mathbf{a}_{s},s,\phi_{\mathrm{enc}}(\mathbf{I}_{t},\mathbf{s}_{t}),f_{\psi}(\mathbf{Z}_{t};\mathbf{E}_{\tau})\bigr)\right\|_{2}^{2}\right]+\lambda_{\mathrm{bev}}\mathcal{L}_{\mathrm{BEV}}.
$$

The important distinction is that gradients from the trajectory denoising loss directly shape the world model. Thus, the predicted future latent is not required to reconstruct every visual detail of the future scene. It only needs to preserve the future information that the planner can exploit, such as lane occupancy, route feasibility, collision-relevant motion, and interactions that affect the ego trajectory.

This treatment differs from several common uses of future information in end-to-end driving. Trajectory-generation methods often represent the future through sampled trajectories, goals, or action distributions, so future reasoning mainly appears at the output level after the current representation has already been formed. Trajectory-scoring methods generate multiple proposals and then rank them, which can improve selection but does not necessarily change how the proposals are represented. Pixel-, video-, or occupancy-space world models predict dense future observations, but this can be expensive and may allocate capacity to planning-irrelevant appearance details. Latent world models avoid some of this cost, yet their future states are often optimized as prediction targets or auxiliary signals. DriveFuture instead uses compact planning-oriented future tokens as a direct conditioning pathway for the trajectory diffusion denoiser, and this pathway can further complement downstream trajectory scoring.

### B.3 More Details on Latent Dynamics Predictor

The latent dynamics predictor uses a Transformer decoder with learnable future queries. Let

$$
\mathbf{C}=\left[\mathbf{Z}_{t}\;\|\;\mathbf{E}_{\tau}\right]\in\mathbb{R}^{(N+T)\times d}
$$

be the concatenated context. Given learnable future queries $\mathbf{Q}^{+,0}\in\mathbb{R}^{K\times d}$, each layer performs

$$
\displaystyle\tilde{\mathbf{Q}}^{+,\ell}
$$
 
$$
\displaystyle=\operatorname{SelfAttn}\left(\operatorname{LN}(\mathbf{Q}^{+,\ell-1})\right)+\mathbf{Q}^{+,\ell-1},
$$
$$
\displaystyle\bar{\mathbf{Q}}^{+,\ell}
$$
 
$$
\displaystyle=\operatorname{CrossAttn}\left(Q=\operatorname{LN}(\tilde{\mathbf{Q}}^{+,\ell}),K=V=\mathbf{C}\right)+\tilde{\mathbf{Q}}^{+,\ell},
$$
$$
\displaystyle\mathbf{Q}^{+,\ell}
$$
 
$$
\displaystyle=\operatorname{FFN}\left(\operatorname{LN}(\bar{\mathbf{Q}}^{+,\ell})\right)+\bar{\mathbf{Q}}^{+,\ell}.
$$

After $L_{\mathrm{wm}}=4$ decoder layers, $\hat{\mathbf{Z}}_{t+T}=\mathbf{Q}^{+,L_{\mathrm{wm}}}$. The compact $K$ -token ($K=16$) output is a deliberate bottleneck: it prevents the world model from copying dense future appearance and encourages it to encode future information that has high utility for planning.

### B.4 More Details on Conditioning Source Randomisation

DriveFuture trains the world model under three conditioning modes:

$$
\mathbf{E}_{\tau}\sim\begin{cases}\phi_{\tau}(\boldsymbol{\tau}^{\mathrm{gt}})&\text{w.p.}\;p_{\mathrm{gt}},\\[2.0pt]
\phi_{\tau}(\boldsymbol{\tau}^{\mathrm{kin}})&\text{w.p.}\;p_{\mathrm{kin}},\\[2.0pt]
\mathbf{E}^{\varnothing}&\text{w.p.}\;p_{\varnothing}.\end{cases}
$$

In our default configuration, $(p_{\mathrm{gt}},p_{\mathrm{kin}},p_{\varnothing})=(0.4,0.4,0.2)$. The three modes have complementary roles. The expert trajectory provides a high-quality future-intent signal during training, the kinematic rollout exposes the model to a deployable coarse intent, and the null token creates the unconditional branch required by classifier-free guidance. The null branch is especially important: without an unconditional baseline, guidance would be an uncalibrated difference between two conditional predictions rather than a proper correction direction.

The kinematic rollout is computed using a constant-acceleration ego model. Given current ego velocity $v=(v_{x},v_{y})$ and acceleration $a=(a_{x},a_{y})$, the $k$ -th future pose at interval $\Delta t$ is

$$
\displaystyle t_{k}
$$
 
$$
\displaystyle=k\Delta t,
$$
$$
\displaystyle x_{k}^{\mathrm{kin}}
$$
 
$$
\displaystyle=v_{x}t_{k}+\frac{1}{2}a_{x}t_{k}^{2},
$$
$$
\displaystyle y_{k}^{\mathrm{kin}}
$$
 
$$
\displaystyle=v_{y}t_{k}+\frac{1}{2}a_{y}t_{k}^{2},
$$
$$
\displaystyle\theta_{k}^{\mathrm{kin}}
$$
 
$$
\displaystyle=\operatorname{atan2}(v_{y}+a_{y}t_{k},v_{x}+a_{x}t_{k}).
$$

This rollout is not intended to be an accurate planner. Its purpose is to provide a stable, physically plausible intent direction when the diffusion sample is still too noisy to be trusted.

### B.5 More Details on Future Alignment Adapter

During training, DriveFuture can access the future observation $\mathbf{I}_{t+T}$. It is encoded with the same perception backbone under a stop-gradient operation:

$$
\mathbf{Z}_{t+T}=\operatorname{sg}\left(\phi_{\mathrm{enc}}(\mathbf{I}_{t+T})\right)\in\mathbb{R}^{64\times d}.
$$

Rather than directly using all $64$ future tokens as denoiser condition, DriveFuture lets the predicted foresight latent query the future token bank:

$$
\displaystyle\mathbf{A}
$$
 
$$
\displaystyle=\operatorname{softmax}\left(\frac{\operatorname{LN}(\hat{\mathbf{Z}}_{t+T})W_{Q}(\mathbf{Z}_{t+T}W_{K})^{\top}}{\sqrt{d_{h}}}\right),
$$
$$
\displaystyle\tilde{\mathbf{Z}}_{t+T}
$$
 
$$
\displaystyle=\mathbf{A}\,\mathbf{Z}_{t+T}W_{V}.
$$

This makes the future oracle trajectory-aware: the query side is the world model’s compact prediction, while the value side is the real future BEV. The module has no residual connection from $\hat{\mathbf{Z}}_{t+T}$, so the oracle condition is a pure selection of ground-truth future evidence.

The curriculum mixture is

$$
\tilde{\mathbf{Z}}_{t+T}^{c}=\alpha(e)\mathbf{Z}_{t+T}^{c}+(1-\alpha(e))\hat{\mathbf{Z}}_{t+T}.
$$

In the code implementation used for the current experiments, the schedule is equivalently written as

$$
\alpha(e)=1-\frac{1}{1+\exp\left[-\beta(e-e_{0})\right]},\quad e_{0}=\rho_{E}E,
$$

where $E$ is the configured maximum epoch. At early epochs $e\ll e_{0}$, $\alpha(e)\approx 1$, and the denoiser learns with a strong future oracle. At late epochs $e\gg e_{0}$, $\alpha(e)\approx 0$, forcing the world model prediction to carry the conditioning information used at inference.

### B.6 More Details on Progressive Foresight Guidance

At inference, the expert trajectory $\boldsymbol{\tau}^{\mathrm{gt}}$ and future observation $\mathbf{I}_{t+T}$ are unavailable. The Future Alignment Adapter is therefore bypassed and $\mathbf{Z}_{t+T}^{c}=\hat{\mathbf{Z}}_{t+T}$. A circular dependency then arises:

$$
\hat{\mathbf{Z}}_{t+T}=f_{\psi}\bigl(\mathbf{Z}_{t};\mathbf{E}_{\tau}\bigr),\qquad\boldsymbol{\tau}=\operatorname{Decode}\bigl(\mathbf{Z}_{t},\hat{\mathbf{Z}}_{t+T}\bigr).
$$

The trajectory intent $\mathbf{E}_{\tau}$ is required to compute $\hat{\mathbf{Z}}_{t+T}$, yet $\boldsymbol{\tau}$ is itself the output of the denoising process conditioned on $\hat{\mathbf{Z}}_{t+T}$. PFG breaks this circularity by supplying two deployable surrogate sources: an external coarse source $\boldsymbol{\tau}^{\mathrm{kin}}$ and an internal self-estimated source $\boldsymbol{\tau}^{\mathrm{tw}}$.

For a diffusion step $s$, define the unconditional, kinematic, and trajectory-conditioned future latents as

$$
\displaystyle\mathbf{Z}_{t+T}^{c,\varnothing}
$$
 
$$
\displaystyle=f_{\psi}(\mathbf{Z}_{t},\mathbf{E}^{\varnothing}),
$$
$$
\displaystyle\mathbf{Z}_{t+T}^{c,\mathrm{kin}}
$$
 
$$
\displaystyle=f_{\psi}(\mathbf{Z}_{t},\phi_{\tau}(\boldsymbol{\tau}^{\mathrm{kin}})),
$$
$$
\displaystyle\mathbf{Z}_{t+T}^{c,\mathrm{tw}}
$$
 
$$
\displaystyle=f_{\psi}(\mathbf{Z}_{t},\phi_{\tau}(\boldsymbol{\tau}^{\mathrm{tw}})).
$$

The guided denoising direction is

$$
\hat{\boldsymbol{\epsilon}}=\hat{\boldsymbol{\epsilon}}_{\varnothing}+w_{\mathrm{kin}}(r)\left(\hat{\boldsymbol{\epsilon}}_{\mathrm{kin}}-\hat{\boldsymbol{\epsilon}}_{\varnothing}\right)+w_{\mathrm{tw}}(r)\left(\hat{\boldsymbol{\epsilon}}_{\mathrm{tw}}-\hat{\boldsymbol{\epsilon}}_{\varnothing}\right),
$$

where $r\in[0,1]$ is denoising progress. This can be interpreted as a first-order guidance composition around the unconditional score estimate. The two correction terms induce two different directions in trajectory space: a stable low-frequency motion prior from kinematics and a high-precision self-consistency prior from the denoised sample.

The Tweedie estimate used by the second source is

$$
\hat{\mathbf{a}}_{0}^{(s)}=\frac{\mathbf{a}_{s}-\sqrt{1-\bar{\alpha}_{s}}\,\hat{\boldsymbol{\epsilon}}_{\varnothing}}{\sqrt{\bar{\alpha}_{s}}}.
$$

If $\hat{\boldsymbol{\epsilon}}_{\varnothing}=\epsilon+\delta$, then the error of the Tweedie estimate is

$$
\hat{\mathbf{a}}_{0}^{(s)}-\mathbf{a}_{0}=-\sqrt{\frac{1-\bar{\alpha}_{s}}{\bar{\alpha}_{s}}}\,\delta.
$$

This explains why the trajectory source should be suppressed at high noise: when $\bar{\alpha}_{s}$ is small, the error amplification factor is large. Conversely, at low noise, $\bar{\alpha}_{s}\rightarrow 1$ and the estimate becomes reliable. The progressive schedule implements this observation.

$$
\displaystyle w_{\mathrm{kin}}(r)
$$
 
$$
\displaystyle=\begin{cases}w^{\max}_{\mathrm{kin}}\cos\!\bigl(\tfrac{\pi r}{2\rho}\bigr),&r<\rho,\\
0,&r\geq\rho,\end{cases}
$$
 
$$
\displaystyle w_{\mathrm{tw}}(r)
$$
 
$$
\displaystyle=\begin{cases}0,&r\leq\nu,\\
\tfrac{w^{\max}_{\mathrm{tw}}}{2}\Bigl[1-\cos\!\Bigl(\pi\tfrac{r-\nu}{1-\nu}\Bigr)\Bigr],&r>\nu.\end{cases}
$$

The default values are $w^{\max}_{\mathrm{kin}}=1.5$, $w^{\max}_{\mathrm{tw}}=2.5$, $\rho=0.7$, and $\nu=0.3$.

The schedule has an intuitive phase structure. At high noise ($r<\nu$), $\mathbf{a}_{s}$ is close to Gaussian and Tweedie estimates are unstable, so PFG mainly relies on the kinematic source to stabilize coarse direction and avoid implausible early drift. At middle noise ($\nu<r<\rho$), the denoised estimate becomes partially meaningful while kinematics still regularizes the solution, producing a smooth handover between the coarse prior and the self-consistent plan. At low noise ($r>\rho$), $\hat{\mathbf{a}}_{0}^{(s)}$ becomes reliable and reflects the denoiser’s current mode, so the Tweedie trajectory source dominates final refinement through trajectory-aware future latents.

### B.7 More Details on Training Objective

The complete training objective used in DriveFuture is

$$
\mathcal{L}=\lambda_{\mathrm{plan}}\mathcal{L}_{\mathrm{plan}}+\lambda_{\mathrm{bev}}\mathcal{L}_{\mathrm{BEV}},\quad\lambda_{\mathrm{plan}}=\lambda_{\mathrm{bev}}=10.
$$

The diffusion planning loss is

$$
\mathcal{L}_{\mathrm{plan}}=\mathbb{E}_{s,\epsilon}\left\|\epsilon-\epsilon_{\theta}\left(\sqrt{\bar{\alpha}_{s}}\mathbf{a}_{0}+\sqrt{1-\bar{\alpha}_{s}}\epsilon,s,\mathbf{Z}_{t},\tilde{\mathbf{Z}}_{t+T}^{c}\right)\right\|_{2}^{2}.
$$

The BEV semantic loss is a seven-class cross-entropy loss over the upsampled BEV prediction:

$$
\mathcal{L}_{\mathrm{BEV}}=-\frac{1}{|\Omega|}\sum_{u\in\Omega}\sum_{c=1}^{7}y_{u,c}\log p_{u,c}.
$$

The BEV objective anchors the latent scene representation to metric semantics and stabilizes trajectory diffusion training. It also makes the latent space more compatible with future-frame alignment because both current and future BEV tokens are produced by the same encoder family.

## Appendix C Implementation Details for DriveFuture

This section describes the implementation used for the submitted DriveFuture models. The details are reconstructed from the training and evaluation scripts, the Hydra configuration, and the model code. We include them to facilitate reproducibility and to clarify several engineering choices that materially affect performance on NAVSIM-v1 and NAVSIM-v2.

### C.1 More Details on Model Configuration

DriveFuture uses two temporal frames and two stitched panorama streams as input. The front panorama concatenates left-front, front, and right-front camera views, while the rear panorama concatenates left-rear, rear, and right-rear views; both panoramas are resized to $2048\times 512$. The image encoder follows a VoVNet / V2-99-style backbone initialized from detection pretraining. Its image features are queried by learnable BEV anchors arranged on a $16\times 64$ grid, which are downscaled into $64$ spatial BEV tokens. A single ego-status token is appended to these BEV tokens, yielding the current latent $\mathbf{Z}_{t}\in\mathbb{R}^{65\times 256}$. The world model uses $16$ future queries, $4$ Transformer decoder layers, and an FFN dimension of $2048$. The trajectory head predicts $8$ poses over a $4$ second horizon at $0.5$ second intervals. The diffusion planning head contains $5$ Transformer layers and samples $100$ proposals at inference. For PFG, the default guidance strengths are $(w^{\max}_{\mathrm{kin}},w^{\max}_{\mathrm{tw}})=(1.5,2.5)$ with phase parameters $(\rho,\nu)=(0.7,0.3)$. The world model and diffusion head share the same latent dimension $d=256$, which avoids projection mismatch between current scene tokens, future tokens, and trajectory tokens.

### C.2 More Details on Training Configuration

DriveFuture is trained on the NAVSIM navtrain split with Adam, a learning rate of $1\times 10^{-4}$, batch size $16$ per step, and gradient accumulation over $5$ steps. Development runs use $100$ – $130$ epochs, bf16 mixed precision, gradient clipping at $1.0$, epoch-level checkpointing, and online Weights & Biases logging. The train loader uses roughly $7$ – $9$ workers, the validation loader uses $2$ workers. For efficiency, training is performed in cache-only mode with precomputed NAVSIM features: the data loader reads cached features and targets rather than reconstructing the scene loader online, and cache-on-miss behavior is disabled in the final training run to avoid accidental recomputation. The same training step also uses the default conditioning-source dropout $(p_{\mathrm{gt}},p_{\mathrm{kin}},p_{\varnothing})=(0.4,0.4,0.2)$. Algorithm 1 summarizes how these settings are instantiated in each DriveFuture training step.

Algorithm 1 DriveFuture training step

Batch of features $F$, targets $Y$, current epoch $e$

$\mathbf{Z}_{t},\mathbf{B}_{t}\leftarrow\mathcal{E}_{\theta}(F)$ $\triangleright$ Current BEV and semantic features

 $\tau_{\mathrm{kin}}\leftarrow\textsc{KinematicRollout}(F.s_{t})$

$\hat{\mathbf{Z}}^{+}_{t+T}\leftarrow\mathcal{W}_{\psi}(\mathbf{Z}_{t},\phi(Y.\tau^{\star}),\phi(\tau_{\mathrm{kin}}),\mathbf{c}^{\varnothing})$ $\triangleright$ Three-mode dropout internally selects the condition

if future frame exists then

   $\mathbf{Z}^{+,\star}_{t+T}\leftarrow\operatorname{sg}(\mathcal{E}_{\theta}(Y.o_{t+T}))$    $c_{\mathrm{interact}}\leftarrow\operatorname{CrossAttn}(\hat{\mathbf{Z}}^{+}_{t+T},\mathbf{Z}^{+,\star}_{t+T})$    $\tilde{\mathbf{Z}}^{+}\leftarrow\alpha(e)c_{\mathrm{interact}}+(1-\alpha(e))\hat{\mathbf{Z}}^{+}_{t+T}$

else

   $\tilde{\mathbf{Z}}^{+}\leftarrow\hat{\mathbf{Z}}^{+}_{t+T}$

end if

Sample diffusion step $s$ and noise $\epsilon$

 $\mathbf{a}_{s}\leftarrow\sqrt{\bar{\alpha}_{s}}\phi(Y.\tau^{\star})+\sqrt{1-\bar{\alpha}_{s}}\epsilon$ $\hat{\epsilon}\leftarrow\epsilon_{\omega}(\mathbf{a}_{s},s,\mathbf{Z}_{t},\tilde{\mathbf{Z}}^{+})$ $\mathcal{L}\leftarrow 10\|\epsilon-\hat{\epsilon}\|_{2}^{2}+10\mathcal{L}_{\mathrm{BEV}}(\mathbf{B}_{t},Y_{\mathrm{BEV}})$

return $\mathcal{L}$

### C.3 More Details on Inference

Algorithm 2 summarizes the inference pipeline used by DriveFuture, including CFG guidance and the optional scorer-based final selection. At inference, DriveFuture generates $100$ trajectory proposals per scene. Each proposal starts from independent Gaussian noise. The current scene latent $\mathbf{Z}_{t}$, unconditional future latent $\mathbf{Z}^{+}_{\varnothing}$, and kinematic future latent $\mathbf{Z}^{+}_{\mathrm{kin}}$ are computed once per scene and repeated across proposals. The Tweedie future latent $\mathbf{Z}^{+}_{\mathrm{tw}}(s)$ is recomputed only on denoising steps where $w_{\mathrm{tw}}(r)>0$.

Algorithm 2 DriveFuture inference with PFG and optional trajectory scoring

Observation $o_{t}$, ego-status $s_{t}$, proposal count $K=100$

 $\mathbf{Z}_{t}\leftarrow\mathcal{E}_{\theta}(o_{t},s_{t})$ $\tau_{\mathrm{kin}}\leftarrow\textsc{KinematicRollout}(s_{t})$ $\mathbf{Z}^{+}_{\varnothing}\leftarrow\mathcal{W}_{\psi}(\mathbf{Z}_{t},\mathbf{c}^{\varnothing})$ $\mathbf{Z}^{+}_{\mathrm{kin}}\leftarrow\mathcal{W}_{\psi}(\mathbf{Z}_{t},\phi(\tau_{\mathrm{kin}}))$

for $k=1$ to $K$ do

  Sample $\mathbf{a}_{S}^{(k)}\sim\mathcal{N}(0,I)$

  for $s=S$ to $1$ do

   Compute $w_{\mathrm{kin}}(r)$ and $w_{\mathrm{tw}}(r)$

   Predict $\epsilon_{\varnothing}$ under $\mathbf{Z}^{+}_{\varnothing}$

   Add kinematic correction if $w_{\mathrm{kin}}>0$

   Estimate $\hat{\mathbf{a}}_{0}$ by Tweedie and add trajectory correction if $w_{\mathrm{tw}}>0$

   Update $\mathbf{a}_{s-1}^{(k)}$ with the diffusion scheduler

  end for

   $\tau^{(k)}\leftarrow\textsc{CumSum}(\mathbf{a}_{0}^{(k)})$

end for

if using GTRS-Dense scorer then

   $k^{\star}\leftarrow\arg\max_{k}\operatorname{Score}_{\mathrm{GTRS}}(\tau^{(k)},o_{t})$

  return $\tau^{(k^{\star})}$ and proposals $\{\tau^{(k)}\}_{k=1}^{K}$

else

  return default proposal and proposals $\{\tau^{(k)}\}_{k=1}^{K}$

end if

For NAVSIM-v2 navhard leaderboard submission, we use DriveFuture proposals with a GTRS-Dense scorer [^49]. This follows the recent observation that high-quality proposal scoring is crucial for NAVSIM performance [^49] [^54] [^55] [^68]. DriveFuture improves the proposal distribution, while the scorer selects the best candidate under a stronger planning metric proxy. The main ablations toggle the runnable implementation through several switches: use\_wm enables the latent dynamics predictor, use\_wm\_to\_dit feeds the foresight latent into the diffusion transformer, use\_interact enables the training-time future interaction adapter, and force\_alpha\_one tests the effect of keeping the oracle future condition throughout training. At inference, use\_dspcfg enables PFG, use\_kinematic\_extrap enables the constant-acceleration guidance source, and p\_gt, p\_kin, and p\_null control the three-mode CFG dropout with default values $0.4/0.4/0.2$. These switches connect the runnable configuration directly to the ablation study.

## Appendix D More Results

This section expands the experimental discussion in the main paper. We provide additional breakdowns, relative improvements, ablation-oriented analysis, and interpretation of the main metrics. The three settings stress different properties. NAVSIM-v1 navtest emphasizes the original PDM-style safety and progress metrics. NAVSIM-v2 navtest adds rule compliance and comfort terms. NAVSIM-v2 navhard is the strictest setting because the final score multiplies Stage-1 quality by robustness to Stage-2 synthesized future observations. DriveFuture is designed for precisely this type of evaluation: the world model explicitly uses future-conditioned latent representations, and PFG improves proposal generation under uncertain future evolution.

### D.1 More Details on Navhard

Table 6 quantifies how much DriveFuture + Score improves over representative NAVSIM-v2 navhard baselines, while Table 7 breaks the same result down into Stage-1 and Stage-2 metrics for DriveFuture with and without scoring.

| Method | EPDMS | Gain | Rel. |
| --- | --- | --- | --- |
| TransFuser [^39] | 23.1 | +32.4 | +140.3% |
| DiffusionDrive [^8] | 24.2 | +31.3 | +129.3% |
| GuideFlow [^11] | 27.1 | +28.4 | +104.8% |
| MindDrive [^14] | 30.9 | +24.6 | +79.6% |
| World4Drive [^18] | 34.9 | +20.6 | +59.0% |
| GTRS-E [^49] | 49.4 | +6.1 | +12.3% |
| SimScale [^56] | 53.2 | +2.3 | +4.3% |
| DrivoR [^57] | 54.6 | +0.9 | +1.6% |

Table 6: Relative comparison on NAVSIM-v2 navhard using the main-paper results. Improvements are computed against the listed method’s EPDMS.

The relative gains in Table 6 are largest over models without strong two-stage robustness or dense proposal scoring. More importantly, DriveFuture maintains an advantage even over strong recent methods that already use trajectory scoring or larger backbones. This suggests that improving the proposal distribution remains complementary to stronger downstream ranking.

Table 7 shows that the benefit of scoring is not confined to the easier first stage. The addition of scoring increases Stage-1 safety metrics almost to saturation: NC and DAC both reach $99.8$, while DDC reaches $100.0$. On Stage 2, where future observations are perturbed, the scorer also substantially improves NC, DAC, DDC, TTC, and LK. The lower EC value for the scored model indicates a known trade-off in NAVSIM-style planning: selecting safer and more rule-compliant proposals can reduce the extended comfort metric when the chosen trajectory is more conservative or involves stronger braking. Since EPDMS uses multiplicative penalties for safety-critical metrics, the safety and compliance improvements dominate the final score.

Table 7: DriveFuture stage-wise NAVSIM-v2 navhard metrics from the main paper. Stage 1 evaluates the original observation; Stage 2 evaluates synthesized follow-up observations.

| Model | Stage | NC | DAC | DDC | TL | EP | TTC | LK | HC | EC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DriveFuture | Stage 1 | 96.3 | 87.8 | 98.2 | 99.6 | 83.1 | 96.9 | 94.9 | 97.6 | 76.9 |
| DriveFuture | Stage 2 | 82.3 | 78.8 | 88.1 | 98.4 | 83.6 | 79.6 | 47.6 | 97.0 | 75.9 |
| DriveFuture + Score | Stage 1 | 99.8 | 99.8 | 100.0 | 99.6 | 85.7 | 99.8 | 98.7 | 97.6 | 66.2 |
| DriveFuture + Score | Stage 2 | 90.6 | 87.5 | 94.1 | 99.1 | 84.6 | 88.8 | 58.3 | 93.5 | 45.6 |

### D.2 More Details on NAVSIM-v2 Navtest

![[tu3.drawio.png|Refer to caption]]

Figure 4: DriveFuture across multiple driving scenarios from NAVSIM-v2 navhard. 1.

Although we do not repeat the full NAVSIM-v2 navtest leaderboard table in this appendix, the main-paper results show that DriveFuture is particularly strong on NC, DAC, DDC, TL, and TTC. This metric pattern is consistent with the method design: future-conditioned latent planning improves the model’s ability to avoid near-future unsafe states, while PFG stabilizes proposal generation before final scoring. The corrected EPDMS is higher than EPDMS <sup>∗</sup> because the official human-behavior filtering fix avoids penalizing the agent for violations that also occur in the human reference behavior.

### D.3 More Details on NAVSIM-v1 Navtest

The NAVSIM-v1 score shows a similar pattern: DriveFuture is strongest on feasibility and safety-related terms. Ego progress is not maximized relative to some aggressive methods, but the final PDMS remains strong because hard penalties such as NC and DAC have multiplicative influence. This is desirable for autonomous driving: a planner that makes slightly less progress while preserving collision-free and drivable-area behavior is often preferred by PDM-style metrics. This metric-level pattern is consistent with the role of each DriveFuture component. The world model $\mathcal{W}$ adds a compact foresight latent and prevents the planner from degenerating into current-scene-only diffusion. The Future Alignment Adapter supplies high-fidelity future BEV evidence during training, and the latent-alignment curriculum transfers this oracle signal into the predicted future latent while reducing train–test mismatch. Three-mode CFG dropout calibrates the GT, kinematic, and null branches within one world model. During inference, the kinematic source stabilizes high-noise denoising, the Tweedie source refines low-noise samples with self-consistent trajectory intent, and the GTRS-Dense scorer selects the strongest proposal from the generated set. Removing these components is expected to weaken future awareness, branch calibration, early denoising stability, final proposal refinement, or candidate selection, respectively.

Fig. 4 provides visual evidence that is consistent with the quantitative gains reported in the main paper. From left to right, the top row corresponds to curved turning, straight lane following with nearby traffic, and turning through a complex intersection, while the bottom row shows dense-traffic lane following, traversal through a narrow road segment constrained by parked vehicles, and roundabout navigation. Despite substantial variation in road topology, traffic density, and interaction complexity, DriveFuture consistently produces trajectories that remain well aligned with the underlying road structure and establish reasonable turning or forward-progress trends at an early stage. The predicted trajectories exhibit smoother curvature transitions and fewer artifacts such as visible jitter, over-steering, or ineffective lateral drift. In the more interactive scenes, the model also adapts earlier to conflict-zone geometry and implicit safety boundaries. Taken together, these qualitative patterns support the main quantitative results and suggest that the future-conditioned latent improves not only benchmark scores, but also the coherence, safety, and topology consistency of the resulting planning behavior.

[^1]: D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, A. Geiger, and K. Chitta (2024) NAVSIM: data-driven non-reactive autonomous vehicle simulation and benchmarking. External Links: 2406.15349, [Link](https://arxiv.org/abs/2406.15349) Cited by: §A.1, §A.2, Appendix B, Figure 4, 3rd item, §1, Figure 3, §4.1, Table 1, Table 2, Table 3, [Abstract](#abstract1.1 "Abstract ‣ DriveFuture: Future-Aware Latent World Models for Autonomous Driving").

[^2]: F. Jia, C. Jia, Z. Song, Z. Bao, L. Liu, S. Xu, Y. Gong, L. Yang, X. Zhang, B. Sun, et al. (2025) Progressive robustness-aware world models in autonomous driving: a review and outlook. Authorea Preprints. Cited by: §1.

[^3]: T. Feng, W. Wang, and Y. Yang (2025) A survey of world models for autonomous driving. arXiv preprint arXiv:2501.11260. Cited by: §1.

[^4]: S. Tu, X. Zhou, D. Liang, X. Jiang, Y. Zhang, X. Li, and X. Bai (2025) The role of world models in shaping autonomous driving: a comprehensive survey. arXiv preprint arXiv:2502.10498. Cited by: §1.

[^5]: Z. Song, L. Liu, F. Jia, Y. Luo, C. Jia, G. Zhang, L. Yang, and L. Wang (2024) Robustness-aware 3d object detection in autonomous driving: a review and outlook. IEEE Transactions on Intelligent Transportation Systems 25 (11), pp. 15407–15436. Cited by: §1.

[^6]: Y. Luo, Q. Chen, F. Li, S. Xu, J. Liu, Z. Song, Z. Yang, and F. Wen (2026) Unleashing vla potentials in autonomous driving via explicit learning from failures. arXiv preprint arXiv:2603.01063. Cited by: §1.

[^7]: Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, L. Lu, X. Jia, Q. Liu, J. Dai, Y. Qiao, and H. Li (2023) Planning-oriented autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 17853–17862. Cited by: §1, §2.2, Table 3.

[^8]: B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang, et al. (2025) Diffusiondrive: truncated diffusion model for end-to-end autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 12037–12047. Cited by: Appendix B, Table 6, §1, §2.2, Table 1, Table 2, Table 3.

[^9]: W. Sun, X. Lin, Y. Shi, C. Zhang, H. Wu, and S. Zheng (2025) Sparsedrive: end-to-end autonomous driving via sparse scene representation. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pp. 8795–8801. Cited by: §1, §2.2.

[^10]: Z. Xing, X. Zhang, Y. Hu, B. Jiang, T. He, Q. Zhang, X. Long, and W. Yin (2025) Goalflow: goal-driven flow matching for multimodal trajectories generation in end-to-end autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 1602–1611. Cited by: §1, §2.2, §4.3, Table 3, Table 3.

[^11]: L. Liu, C. Jia, G. Yu, Z. Song, J. Li, F. Jia, P. Wu, X. Hao, and Y. Luo (2025) GuideFlow: constraint-guided flow matching for planning in end-to-end autonomous driving. arXiv preprint arXiv:2511.18729. Cited by: Table 6, §1, §2.2, Table 1.

[^12]: B. Sun, B. Zhang, J. Lu, X. Feng, J. Shang, R. Cao, M. Zheng, C. Wang, S. Yang, Y. Cao, et al. (2025) FocalAD: local motion planning for end-to-end autonomous driving. arXiv preprint arXiv:2506.11419. Cited by: §1, §2.2.

[^13]: Z. Song, C. Jia, L. Liu, H. Pan, Y. Zhang, J. Wang, X. Zhang, S. Xu, L. Yang, and Y. Luo (2025) Don’t shake the wheel: momentum-aware planning in end-to-end autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 22432–22441. Cited by: §1, §2.2.

[^14]: B. Suna, Y. Caob, Y. Wanga, R. Wanga, J. Shanga, X. Fenga, J. Lu, J. Shi, S. Yang, X. Yane, et al. (2025) MindDrive: an all-in-one framework bridging world models and vision-language model for end-to-end autonomous driving. arXiv preprint arXiv:2512.04441. Cited by: Table 6, §1, Table 1.

[^15]: S. Chen, B. Jiang, H. Gao, B. Liao, Q. Xu, Q. Zhang, C. Huang, W. Liu, and X. Wang (2024) Vadv2: end-to-end vectorized autonomous driving via probabilistic planning. arXiv preprint arXiv:2402.13243. Cited by: §1, §2.2, Table 3.

[^16]: Y. Li, K. Xiong, X. Guo, F. Li, S. Yan, G. Xu, L. Zhou, L. Chen, H. Sun, B. Wang, et al. (2025) Recogdrive: a reinforced cognitive framework for end-to-end autonomous driving. arXiv preprint arXiv:2506.08052. Cited by: §1, §2.2, Table 2, Table 3.

[^17]: Y. Li, L. Fan, J. He, Y. Wang, Y. Chen, Z. Zhang, and T. Tan (2024) Enhancing end-to-end autonomous driving with latent world model. arXiv preprint arXiv:2406.08481. Cited by: Appendix B, Figure 1, §1, §1, §1, §2.1, Table 3.

[^18]: Y. Zheng, P. Yang, Z. Xing, Q. Zhang, Y. Zheng, Y. Gao, P. Li, T. Zhang, Z. Xia, P. Jia, et al. (2025) World4drive: end-to-end autonomous driving via intention-aware physical latent world model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 28632–28642. Cited by: Appendix B, Table 6, Figure 1, §1, §1, §2.1, Figure 3, §4.3, §4.5, Table 1, Table 3.

[^19]: P. Yang, B. Lu, Z. Xia, C. Han, Y. Gao, T. Zhang, K. Zhan, X. Lang, Y. Zheng, and Q. Zhang (2026) WorldRFT: latent world model planning with reinforcement fine-tuning for autonomous driving. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40, pp. 11649–11657. Cited by: Appendix B, Figure 1, §1, §1, §2.1, Table 3.

[^20]: L. Liu, Z. Song, C. Jia, H. Ye, X. Hao, L. Chen, et al. (2026) DriveWorld-vla: unified latent-space world modeling with vision-language-action for autonomous driving. arXiv preprint arXiv:2602.06521. Cited by: Appendix B, Figure 1, §1, §1, §2.1, §4.3, §4.3, Table 2, Table 3.

[^21]: Y. Li, S. Shang, W. Liu, B. Zhan, H. Wang, Y. Wang, Y. Chen, X. Wang, Y. An, C. Tang, et al. (2025) DriveVLA-w0: world models amplify data scaling law in autonomous driving. arXiv preprint arXiv:2510.12796. Cited by: Appendix B, Figure 1, §1, §1, §2.1, Table 2, Table 3.

[^22]: T. Xia, Y. Li, L. Zhou, J. Yao, K. Xiong, H. Sun, B. Wang, K. Ma, G. Chen, H. Ye, et al. (2025) Drivelaw: unifying planning and video generation in a latent driving world. arXiv preprint arXiv:2512.23421. Cited by: Appendix B, Figure 1, §1, §1, §2.1, §4.3, Table 3.

[^23]: C. Min, D. Zhao, L. Xiao, J. Zhao, X. Xu, Z. Zhu, L. Jin, J. Li, Y. Guo, J. Xing, et al. (2024) Driveworld: 4d pre-trained scene understanding via world models for autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 15522–15533. Cited by: Figure 1, §1, §1.

[^24]: A. Hu, L. Russell, H. Yeo, Z. Murez, G. Fedoseev, A. Kendall, J. Shotton, and G. Corrado (2023) Gaia-1: a generative world model for autonomous driving. arXiv preprint arXiv:2309.17080. Cited by: §2.1.

[^25]: X. Wang, Z. Zhu, G. Huang, X. Chen, J. Zhu, and J. Lu (2024) Drivedreamer: towards real-world-drive world models for autonomous driving. In European conference on computer vision, pp. 55–72. Cited by: §2.1.

[^26]: S. Gao, J. Yang, L. Chen, K. Chitta, Y. Qiu, A. Geiger, J. Zhang, and H. Li (2024) Vista: a generalizable driving world model with high fidelity and versatile controllability. Advances in Neural Information Processing Systems 37, pp. 91560–91596. Cited by: §2.1.

[^27]: K. Zhang, Z. Tang, X. Hu, X. Pan, X. Guo, Y. Liu, J. Huang, L. Yuan, Q. Zhang, X. Long, et al. (2025) Epona: autoregressive diffusion world model for autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 27220–27230. Cited by: §2.1.

[^28]: J. Yang, K. Chitta, S. Gao, L. Chen, Y. Shao, X. Jia, H. Li, A. Geiger, X. Yue, and L. Chen (2025) Resim: reliable world simulation for autonomous driving. arXiv preprint arXiv:2506.09981. Cited by: §2.1.

[^29]: Z. Yang and Y. Zhang (2026) ConsisDrive: identity-preserving driving world models for video generation by instance mask. arXiv preprint arXiv:2602.03213. Cited by: §2.1.

[^30]: Y. Wang, J. He, L. Fan, H. Li, Y. Chen, and Z. Zhang (2024) Driving into the future: multiview visual forecasting and planning with world model for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14749–14759. Cited by: §2.1.

[^31]: F. Jia, W. Mao, Y. Liu, Y. Zhao, Y. Wen, C. Zhang, X. Zhang, and T. Wang (2023) Adriver-i: a general world model for autonomous driving. arXiv preprint arXiv:2311.13549. Cited by: §2.1.

[^32]: H. Wang, X. Ye, F. Tao, C. Pan, A. Mallik, B. Yaman, L. Ren, and J. Zhang (2025) Adawm: adaptive world model based planning for autonomous driving. arXiv preprint arXiv:2501.13072. Cited by: §2.1.

[^33]: W. Zheng, W. Chen, Y. Huang, B. Zhang, Y. Duan, and J. Lu (2024) Occworld: learning a 3d occupancy world model for autonomous driving. In European conference on computer vision, pp. 55–72. Cited by: §2.1.

[^34]: Y. Yang, J. Mei, Y. Ma, S. Du, W. Chen, Y. Qian, Y. Feng, and Y. Liu (2025) Driving in the occupancy world: vision-centric 4d occupancy forecasting and planning via world models for autonomous driving. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 39, pp. 9327–9335. Cited by: §2.1.

[^35]: L. Wang, W. Zheng, Y. Ren, H. Jiang, Z. Cui, H. Yu, and J. Lu (2024) Occsora: 4d occupancy generation models as world simulators for autonomous driving. arXiv preprint arXiv:2405.20337. Cited by: §2.1.

[^36]: J. Wei, S. Yuan, P. Li, Q. Hu, Z. Gan, and W. Ding (2024) Occllama: an occupancy-language-action generative world model for autonomous driving. arXiv preprint arXiv:2409.03272. Cited by: §2.1.

[^37]: Y. Zhang, S. Gong, K. Xiong, X. Ye, X. Li, X. Tan, F. Wang, J. Huang, H. Wu, and H. Wang (2024) BEVWorld: a multimodal world simulator for autonomous driving via scene-level bev latents. arXiv preprint arXiv:2407.05679. Cited by: §2.1.

[^38]: Y. Li, Y. Wang, Y. Liu, J. He, L. Fan, and Z. Zhang (2025) End-to-end driving with online trajectory evaluation via bev world model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 27137–27146. Cited by: Appendix B, §2.1, Table 3.

[^39]: K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger (2022) Transfuser: imitation with transformer-based sensor fusion for autonomous driving. IEEE transactions on pattern analysis and machine intelligence 45 (11), pp. 12878–12895. Cited by: Table 6, §2.2, Table 1, Table 2, Table 3.

[^40]: B. Jiang, S. Chen, Q. Xu, B. Liao, J. Chen, H. Zhou, Q. Zhang, W. Liu, C. Huang, and X. Wang (2023) Vad: vectorized scene representation for efficient autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 8340–8350. Cited by: §2.2.

[^41]: D. Xu, H. Li, Q. Wang, Z. Song, L. Chen, and H. Deng (2024) M2DA: multi-modal fusion transformer incorporating driver attention for autonomous driving. arXiv preprint arXiv:2403.12552. Cited by: §2.2.

[^42]: M. Guo, Z. Zhang, Y. He, K. Wang, L. Jing, and H. Ling (2025) End-to-end autonomous driving without costly modularization and 3d manual annotation. IEEE Transactions on Pattern Analysis and Machine Intelligence. Cited by: §2.2.

[^43]: Z. Li, K. Li, S. Wang, S. Lan, Z. Yu, Y. Ji, Z. Li, Z. Zhu, J. Kautz, Z. Wu, et al. (2024) Hydra-mdp: end-to-end multimodal planning with multi-target hydra-distillation. arXiv preprint arXiv:2406.06978. Cited by: §2.2, Table 3.

[^44]: L. Liu, C. Jia, Z. Song, H. Pan, B. Liao, W. Sun, Y. Zhang, L. Yang, Y. Luo, et al. (2025) Fully unified motion planning for end-to-end autonomous driving. arXiv preprint arXiv:2504.12667. Cited by: §2.2.

[^45]: W. Zheng, R. Song, X. Guo, C. Zhang, and L. Chen (2024) Genad: generative end-to-end autonomous driving. In European Conference on Computer Vision, pp. 87–104. Cited by: §2.2.

[^46]: B. Zhang, N. Song, X. Jin, and L. Zhang (2025) Bridging past and future: end-to-end autonomous driving with historical prediction and planning. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 6854–6863. Cited by: §2.2.

[^47]: H. Liu, T. Li, H. Yang, L. Chen, C. Wang, K. Guo, H. Tian, H. Li, H. Li, and C. Lv (2026) Reinforced refinement with self-aware expansion for end-to-end autonomous driving. IEEE Transactions on Pattern Analysis and Machine Intelligence. Cited by: §2.2.

[^48]: Z. Song, L. Liu, H. Pan, B. Liao, M. Guo, L. Yang, Y. Zhang, S. Xu, C. Jia, and Y. Luo (2025) DIVER: reinforced diffusion breaks imitation bottlenecks in end-to-end autonomous driving. arXiv preprint arXiv:2507.04049. Cited by: Appendix B, §2.2, Table 3.

[^49]: Z. Li, W. Yao, Z. Wang, X. Sun, J. Chen, N. Chang, M. Shen, Z. Wu, S. Lan, and J. M. Alvarez (2025) Generalized trajectory scoring for end-to-end multimodal planning. arXiv preprint arXiv:2506.06664. Cited by: §C.3, Table 6, §2.2, §4.2, Table 1, Table 4, Table 5.

[^50]: O. Contributors (2023) OpenScene: the largest up-to-date 3d occupancy prediction benchmark in autonomous driving. Note: [https://github.com/OpenDriveLab/OpenScene](https://github.com/OpenDriveLab/OpenScene) Cited by: §A.1, §4.1.

[^51]: H. Caesar, J. Kabzan, K. S. Tan, W. K. Fong, E. Wolff, A. Lang, L. Fletcher, O. Beijbom, and S. Omari (2022) NuPlan: a closed-loop ml-based planning benchmark for autonomous vehicles. External Links: 2106.11810, [Link](https://arxiv.org/abs/2106.11810) Cited by: §A.1, §4.1.

[^52]: W. Cao, M. Hallgarten, T. Li, D. Dauner, X. Gu, C. Wang, Y. Miron, M. Aiello, H. Li, I. Gilitschenski, et al. (2025) Pseudo-simulation for autonomous driving. arXiv preprint arXiv:2506.04218. Cited by: §A.2, Appendix B, §4.1, Table 4, Table 5.

[^53]: B. Jiang, S. Chen, B. Liao, X. Zhang, W. Yin, Q. Zhang, C. Huang, W. Liu, and X. Wang (2024) Senna: bridging large vision-language models and end-to-end autonomous driving. arXiv preprint arXiv:2410.22313. Cited by: Table 1.

[^54]: W. Yao, Z. Li, S. Lan, Z. Wang, X. Sun, J. M. Alvarez, and Z. Wu (2025) DriveSuprim: towards precise trajectory selection for end-to-end planning. arXiv preprint arXiv:2506.06659. Cited by: §C.3, Table 1, Table 2, Table 3.

[^55]: Z. Li, W. Yao, Z. Wang, X. Sun, J. Chen, N. Chang, M. Shen, Z. Wu, S. Lan, and J. M. Alvarez (2025) ZTRS: zero-imitation end-to-end autonomous driving with trajectory scoring. arXiv preprint arXiv:2510.24108. Cited by: §C.3, Table 1.

[^56]: H. Tian, T. Li, H. Liu, J. Yang, Y. Qiu, G. Li, J. Wang, Y. Gao, Z. Zhang, et al. (2025) SimScale: learning to drive via real-world simulation at scale. arXiv preprint arXiv:2511.23369. Cited by: Table 6, Table 1.

[^57]: E. Kirby, A. Boulch, Y. Xu, Y. Yin, G. Puy, E. Zablocki, A. Bursuc, S. Gidaris, R. Marlet, F. Bartoccioni, A. Cao, N. Samet, T. Vu, and M. Cord (2026) Driving on registers. arXiv preprint arXiv:2601.05083. Cited by: Table 6, §4.3, Table 1.

[^58]: Z. Zhou, R. Yang, X. Qi, Y. Guo, S. X. Chen, T. Feng, K. Pistunova, Y. Shen, et al. (2026) SpanVLA: efficient action bridging and learning from negative-recovery samples for vision-language-action model. arXiv preprint arXiv:2604.19710. Cited by: Table 1.

[^59]: A. Jiang, Y. Gao, Z. Sun, Y. Wang, J. Wang, J. Chai, Q. Cao, Y. Heng, H. Jiang, Z. Zhang, X. Guo, H. Sun, and H. Zhao (2025) DiffVLA: vision-language guided diffusion planning for autonomous driving. arXiv preprint arXiv:2505.19381. Cited by: §4.3, Table 1.

[^60]: K. Li, Z. Li, S. Lan, Y. Xie, Z. Zhang, J. Liu, Z. Wu, Z. Yu, and J. M. Alvarez (2025) Hydra-mdp++: advancing end-to-end driving via expert-guided hydra-distillation. arXiv e-prints. Cited by: Table 2.

[^61]: R. Feng, N. Xi, D. Chu, R. Wang, Z. Deng, A. Wang, L. Lu, J. Wang, Y. Huang, et al. (2025) ARTEMIS: autoregressive end-to-end trajectory planning with mixture of experts for autonomous driving. arXiv preprint arXiv:2504.19580. Cited by: Table 2, Table 3.

[^62]: J. Zou, S. Chen, B. Liao, Z. Zheng, Y. Song, L. Zhang, Q. Zhang, W. Liu, et al. (2025) DiffusionDriveV2: reinforcement learning-constrained truncated diffusion modeling in end-to-end autonomous driving. arXiv preprint arXiv:2512.07745. Cited by: §4.3, Table 2.

[^63]: L. Wang, Y. Zheng, Q. Chen, S. Li, Y. Zhang, Z. Xing, Q. Zhang, X. Li, D. Qian, et al. (2026) Latent-wam: latent world action modeling for end-to-end autonomous driving. arXiv preprint arXiv:2603.24581. Cited by: Appendix B, §4.3, Table 2.

[^64]: X. Weng et al. (2024) PARA-Drive: parallelized architecture for real-time autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, Cited by: Table 3.

[^65]: C. Yuan, Z. Zhang, J. Sun, S. Sun, Z. Huang, C. D. W. Lee, D. Li, Y. Han, A. Wong, K. P. Tee, et al. (2024) DRAMA: an efficient end-to-end motion planner for autonomous driving with mamba. In International Symposium on Robotics Research, Cited by: Table 3.

[^66]: Z. Zhou, T. Cai, S. Z. Zhao, Y. Zhang, Z. Huang, B. Zhou, and J. Ma (2025) AutoVLA: a vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. arXiv preprint arXiv:2506.13757. Cited by: Table 3.

[^67]: K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger (2022) TransFuser: imitation with transformer-based sensor fusion for autonomous driving. External Links: 2205.15997, [Link](https://arxiv.org/abs/2205.15997) Cited by: §4.2.

[^68]: W. Sun, X. Lin, K. Chen, Z. Pei, X. Li, Y. Shi, and S. Zheng (2026) SparseDriveV2: scoring is all you need for end-to-end autonomous driving. arXiv preprint arXiv:2603.29163. Cited by: §C.3.
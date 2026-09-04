---
title: "Adaptive-WAM: Quality-Guided Early-Exit Planningfrom Intermediate Video-Diffusion Features"
source: "https://arxiv.org/html/2608.06008v1"
author:
published:
created: 2026-09-04
description:
tags:
  - "clippings"
---
## Adaptive-WAM: Quality-Guided Early-Exit Planning from Intermediate Video-Diffusion Features

Sining Ang    Yuguang Yang    Yan Wang

###### Abstract

Large video diffusion models provide rich spatiotemporal priors for autonomous driving, but existing world-action models often inherit the cost of iterative future-video generation even though deployment only requires an ego trajectory. We ask a more basic question: how much of a video diffusion model must be executed to make a reliable driving decision? Through a controlled study of video denoising timesteps and Diffusion Transformer (DiT) depth, we find that planning performance is largely insensitive to the tested video-noise levels, whereas strong trajectories can already be decoded from intermediate layers. Based on this observation, we introduce Adaptive-WAM, a quality-aware multi-exit planner built on a Wan2.2-5B backbone. Trajectory diffusion heads are attached to selected DiT blocks, and a lightweight trajectory-quality scorer terminates inference once the best trajectory decoded so far satisfies a quality threshold; otherwise, computation continues from the cached hidden state to a deeper exit. The deployed planner therefore avoids the iterative classifier-free denoising loop and VAE decoding required for future-video synthesis, while dynamically allocating backbone depth according to trajectory quality. On NAVSIM, the adaptive single-trajectory planner achieves 90.8 PDMS; a separate fixed-exit variant reaches 92.6 PDMS with 64 proposals. It further obtains 89.9 EPDMS on NAVSIM v2, yielding the best reported results among the compared front-view video world-model planners. Without target-domain fine-tuning, Adaptive-WAM transfers to nuScenes with 0.88 m average L2 error and a 0.08% collision rate. On an A100, adaptive routing improves PDMS from 90.62 to 90.79 while averaging 170 ms end-to-end planning latency, approximately 10% below the 190 ms fixed block-15 planner and 47% below the 320 ms fixed full-depth planner. Code will be released.

<sup>1</sup> Institute for AI Industry Research (AIR), Tsinghua University

<sup>2</sup> Department of Automation, University of Science and Technology of China

<sup>3</sup> School of Electronic Information Engineering, Beihang University

angsn@mail.ustc.edu.cn, wangyan@air.tsinghua.edu.cn

## 1 Introduction

Planning in open-world traffic requires more than recognizing the current scene: an autonomous policy must anticipate how surrounding actors and road context may evolve under its decisions. Large video models provide temporal and motion priors for this purpose, and driving-specific world models adapt such priors to controllable future prediction, simulation, and planning [^33] [^10] [^8] [^9]. Video generation is therefore an appealing representation-learning objective for autonomous driving.

Recent world-action models (WAMs) connect predictive representations to control either by decoding actions from video-model features or by jointly generating visual futures and trajectories [^38] [^24] [^5] [^41]. Although these designs strengthen the connection between prediction and action, their deployed computation remains either coupled to iterative future rollout or tied to a predetermined backbone path. This is unnecessarily rigid: deployment requires only a low-dimensional ego trajectory, and the quality of a plan may become sufficient before the full video backbone has been evaluated.

![[paradigm_comparison.png|Refer to caption]]

Figure 1: Comparison of predetermined and adaptive WAM interfaces. (a) Video-backbone WAMs follow a predetermined frame/action generation path; (b) multimodal WAMs generate visual and action streams along a predetermined path; (c) Adaptive-WAM decodes one trajectory per attempted exit and routes by predicted quality. Future-video prediction supervises training but is not required by the deployed planner.

We therefore ask: how much of a video diffusion model must be executed to make a reliable driving decision? We separate two axes that are often conflated: the *video diffusion timestep*, which controls the latent noise level, and the *DiT depth*, which controls how many transformer blocks are evaluated. On NAVSIM [^7], five tested video timesteps change the score of a fixed layer by at most 0.15 points, whereas fixed exits exhibit substantial depth-dependent variation in action quality. The best intermediate exit outperforms the full-depth exit, yet the final adaptive policy exceeds every fixed single-trajectory exit. This evidence motivates allocating computation according to the quality of the current plan rather than committing to one readout depth.

Based on this observation, we propose Adaptive-WAM, a quality-aware, layer-adaptive world-action model. We retain a Wan2.2-TI2V-5B [^33] backbone and attach a ReCogDrive-style five-step trajectory DiT [^22] to six intermediate blocks. Adaptive routing decodes one trajectory at each attempted exit and retains the best trajectory accumulated across the evaluated exits. A lightweight DINOv2-Small [^27] scorer predicts NAVSIM planning sub-scores for each decoded trajectory. If the highest predicted score among the attempted exits passes a threshold, the controller returns its trajectory; otherwise, backbone execution continues to the next exit. The default planning path skips the remaining video denoising loop, the unconditional classifier-free-guidance branch, and VAE video decoding. Because trajectory rewards are highly saturated and frequently tied, the scorer predicts metric components and acts primarily as an exit-quality verifier rather than imposing a strict total ordering over trajectories. We therefore evaluate both tie-aware selection and consequential large-gap errors. Across scorer-backbone ablations with Wan, ResNet, ViT, and DINO features, the best Wan variant improves the diagnostic score by only 0.03 points over DINOv2-Small while incurring substantially higher inference cost.

Experiments support three conclusions. First, intermediate diffusion features provide a strong planning substrate: the best fixed-exit model improves from 86.56 PDMS after imitation learning to 90.62 after DiffGRPO-style refinement, while the adaptive single-trajectory planner reaches 90.79 PDMS. Second, adaptive exits improve the performance–computation frontier. At the selected threshold, the adaptive policy reaches 90.79 PDMS versus 90.62 for the strongest fixed single-trajectory exit. Its average end-to-end planning latency is 170 ms, approximately 10% lower than the 190 ms fixed block-15 planner and 47% lower than the 320 ms fixed full-depth planner. Third, the learned representation transfers across datasets, reaching 0.88 m average L2 error and 0.08% collision rate on nuScenes [^3] without target-domain fine-tuning.

Our contributions are:

- We systematically diagnose how video-noise level and DiT depth affect driving, revealing robustness to the five tested video-noise levels and distinct depth-dependent quality–computation trade-offs.
- We introduce a multi-exit world-action architecture whose learned trajectory-quality controller dynamically selects the required DiT depth without completing video generation.
- We evaluate planning, transfer, trajectory scoring, and efficiency on NAVSIM v1/v2 and nuScenes, obtaining state-of-the-art planning performance among compared world-model methods while improving over every fixed single-trajectory exit at lower average cost than the strongest one.

## 2 Related Work

#### Driving video generation and simulation.

Driving world models learn controllable future observations from video. GAIA-1 autoregressively models discrete visual tokens [^10]; MagicDrive, DriveDreamer, Panacea, and DrivingDiffusion condition diffusion models on structured geometry, scene layouts, or driving controls [^8] [^35] [^36] [^18]. VISTA and MiLA further improve controllability and long-horizon consistency [^9] [^34], whereas ReSim and OmniNWM use generative world models for controllable simulation and interactive data generation [^40] [^17]. These studies establish video prediction as a source of spatiotemporal priors. We investigate how much of this trained representation must be evaluated for planning.

#### World-action models for planning.

Latent world models use predicted future representations as planning supervision or context. LAW learns a latent dynamics model for end-to-end driving, DrivingWorld and DriveWorld pretrain video-centric representations, and PWM jointly predicts future states and actions [^19] [^13] [^26] [^44]. More recent WAMs directly couple visual prediction and control. DrivingGPT, Epona, VaViM/VaVAM, DriveVLA-W0, and FutureSightDrive model future observations and actions through autoregressive, parallel, or intermediate reasoning streams [^5] [^43] [^2] [^20] [^42]. DriveLaW conditions an action diffuser on video-model features, whereas DriveVA jointly denoises video and action latents [^38] [^24]. These approaches differ in how prediction and action are coupled, but their deployed backbone computation remains predetermined. Adaptive-WAM instead determines the required video DiT depth from the quality of the currently decoded plan.

#### Efficient video representations and adaptive inference.

Recent studies distinguish learning predictive video representations from rendering future pixels. In embodied robot control, DiT4DiT conditions an action diffuser on intermediate video-DiT features [^25], while Fast-WAM retains future-video supervision but bypasses explicit imagination during deployment [^41]. In autonomous driving, DriveLaW similarly uses an early-denoising video representation for trajectory planning without VAE decoding [^38]. Together, these studies show that predictive video representations can support action generation without rendering future observations, but they still rely on a fixed representation interface or predetermined computation path. Early-exit networks such as BranchyNet and MSDNet instead attach intermediate predictors and vary executed depth according to prediction confidence [^31] [^15]. Their criteria target single-label classification and do not directly address diffusion-based trajectory generation, highly tied planning rewards, or trajectory-quality verification. Adaptive-WAM combines these directions by attaching trajectory exits to a single video DiT and using predicted planning quality to determine whether additional backbone blocks are warranted.

#### Diffusion trajectory planning and planner optimization.

Diffusion-based planners model multimodal driving actions through iterative trajectory refinement. DiffusionDrive truncates trajectory denoising around learned anchors, Diffusion Planner supports flexible guidance during sampling, and GoalFlow uses flow matching for goal-conditioned planning [^23] [^48] [^39]. ReCogDrive adopts a five-step trajectory diffuser and further improves it through planner-only DiffGRPO [^22] [^29]. We use the same trajectory-space formulation at every exit, providing a controlled action-decoding interface for studying intermediate video-DiT representations and adaptive backbone depth. After imitation learning, planner-only DiffGRPO refines the exit heads while leaving the video backbone fixed. Our quality model is orthogonal to trajectory denoising: rather than guiding the sampling process, it determines whether the current exit is sufficient or whether additional world-model computation is required. Because planning rewards are highly saturated and frequently tied, it predicts metric components instead of imposing a strict total ordering over candidates.

## 3 Motivating Analysis

On NAVSIM v1, we first use identical fixed-exit, single-trajectory readouts to test sensitivity to video-noise level and DiT depth, and whether different depths solve redundant scene sets. Reported PDMS aggregates validation-best checkpoints over ten seeds.

| Block | 5 | 9 | 15 | 18 | 22 | 30 |
| --- | --- | --- | --- | --- | --- | --- |
| IL | 81.94 | 83.60 | 86.56 | 84.14 | 83.62 | 80.71 |
| Planner RL | 86.02 | 87.56 | 90.62 | 88.92 | 87.42 | 85.82 |

Table 1: Layer-wise planning quality with identical schedules and validation-best selection over ten seeds.

#### Video timestep is not the main bottleneck.

At block 15, five sampling indices $\{1,9,17,25,32\}$ produce imitation-learning scores $\{86.44,86.56,86.57,86.55,86.50\}$, a range of 0.13. At block 18 the corresponding scores are $\{84.02,84.14,83.99,84.12,84.01\}$, a range of 0.15. We therefore fix index 17 and study depth, avoiding an unnecessary search over video-noise levels.

#### Middle layers are strong, but later layers are not redundant.

Table 1 shows that block 15 is best after both imitation and RL for the single-trajectory readout. RL improves all six exits by roughly four to five points.

#### Solved-scene sets remain only partially shared.

Figure 2 complements the directional pairwise counts. The largest off-diagonal overlap is 0.82 for blocks 9 and 15, whereas early–late pairs fall as low as 0.69. Thus, global dominance by block 15 does not make all other exits redundant. The overlap suggests that a planner can terminate when a shallow trajectory is already strong while retaining deeper computation as a fallback.

Figure 2: Post-RL Jaccard overlap of high-quality scene sets. Off-diagonal values of 0.69–0.82 show substantial but incomplete sharing across exits.

#### Pairwise advantages are structured and reproducible.

Figure 3 exposes information hidden by the global averages. Block 15 has the largest directional advantage counts: it outperforms block 30 by at least 50 points on $554.8$ scenes before planner RL and $598.6$ scenes afterward, on average. The reverse direction remains nonzero ($412.0$ and $422.4$ scenes, respectively), directly showing why a globally strong fixed exit is not uniformly best scene by scene. Across all matrix entries, the maximum standard deviation falls from $182.8$ to $84.9$ after planner RL, indicating more stable pairwise counts without changing the broad depth ordering. Therefore, no fixed exit dominates every scene; this motivates the per-input, quality-guided depth allocation introduced next.

Figure 3: Pairwise large-advantage counts before and after planner RL. Panels report the mean or standard deviation over ten seed-pair runs of $N_{a\succ b}=N(s_{a}-s_{b}\geq 50)$.

## 4 Method

Motivated by Section 3, Adaptive-WAM exposes multiple trajectory exits and routes by predicted quality.

### Problem and Backbone

Let $o=(I,S_{\mathrm{ego}},L_{\mathrm{nav}})$ contain the current front-camera image, deployment-available current and historical ego states, and navigation command. The planner predicts a four-second ego trajectory $\tau=\{(x_{t},y_{t},\theta_{t})\}_{t=1}^{8}$ at 0.5-second intervals. Adaptive inference is single-trajectory: every attempted exit contributes one new trajectory.

We initialize the visual dynamics backbone from Wan2.2-TI2V-5B, a latent video DiT in the Wan family [^33] [^28]. Let $d(o)$ be the deployment-available text description derived from the observation. The conditional Wan branch directly processes the current image–description pair at fixed video-noise index $s^{\star}=17$ of the 40-step schedule. It does not consume a ground-truth or encoded future video at deployment. For DiT block $\ell$, the hidden representation is

$$
h_{\ell}=F_{1:\ell}\bigl(I,d(o);s^{\star}\bigr),
$$

where $F_{1:\ell}$ denotes the conditional backbone prefix. This operation is a *single video-feature forward*; it is separate from the five DDIM [^30] steps used by each trajectory head.

![[main_architecture.png|Refer to caption]]

Figure 4: Overview of Adaptive-WAM. Wan2.2 retains video supervision while six intermediate blocks feed independent ReCogDrive-style trajectory heads. At inference, one trajectory is decoded per attempted exit; the lightweight DINOv2-Small scorer either returns the best accumulated trajectory or continues from the cached hidden state. Repeated heads and LoRA locations are schematic, and the future-scene branch denotes training supervision rather than the default deployed path.

### Multi-Exit Trajectory Decoding

We select exits $\mathcal{E}=\{5,9,15,18,22,30\}$. At exit $\ell$, a projection $P_{\ell}$ converts $h_{\ell}$ into trajectory-conditioning tokens, and an independent diffusion head $G_{\ell}$ generates one trajectory

$$
\tau_{\ell}=G_{\ell}(P_{\ell}(h_{\ell}),S_{\mathrm{ego}},L_{\mathrm{nav}}).
$$

All heads have the same architecture, optimization budget, batch size, and number of epochs. They use five action-denoising steps, matching ReCogDrive. The exit heads have independent parameters and do not exchange features or predictions. This makes differences across exits attributable to backbone depth rather than to head capacity.

Each single-trajectory head uses the same logged-trajectory diffusion objective and five-step training protocol as ReCogDrive.

### Quality-Guided Adaptive Inference

The scorer fine-tunes a DINOv2-Small image encoder and embeds the flattened eight-pose trajectory with an MLP. The image and trajectory features are concatenated and passed to six independent two-layer MLP heads. The scorer does not receive ego state or navigation command. For component set $\mathcal{R}=\{\mathrm{NC,DAC,DDC,TTC,EP,Comf}\}$, it produces

$$
\begin{split}\mathbf{a}_{\ell}&=S_{\phi}(I,\tau_{\ell}),\\
\hat{\mathbf{r}}_{\ell}&=\sigma(\mathbf{a}_{\ell})=(\widehat{\mathrm{NC}},\widehat{\mathrm{DAC}},\widehat{\mathrm{DDC}},\widehat{\mathrm{TTC}},\widehat{\mathrm{EP}},\widehat{\mathrm{Comf}}).\end{split}
$$

The normalized PDMS composition $\Gamma(\cdot)\in[0,1]$ is expressed on the reported 100-point scale as

$$
Q(\hat{\mathbf{r}})=100\,\Gamma(\hat{\mathbf{r}}).
$$

Let

$$
\mathcal{A}_{j}=\{\tau_{\ell_{m}}\}_{m=1}^{j}
$$

denote the one trajectory decoded at each of the first $j$ attempted exits; hence $|\mathcal{A}_{j}|=j\leq 6$. We cache the component predictions and maintain

$$
\begin{split}\hat{\tau}_{j}&=\operatorname*{arg\,max}_{\tau_{\ell_{m}}\in\mathcal{A}_{j}}Q(\hat{\mathbf{r}}_{\ell_{m}}),\\
\hat{q}_{j}&=\max_{m\leq j}Q(\hat{\mathbf{r}}_{\ell_{m}}).\end{split}
$$

Given $\eta\in[0,100]$, the controller terminates at the first exit satisfying $\hat{q}_{j}\geq\eta$. If no earlier exit satisfies the threshold, the final exit returns the highest-scoring trajectory accumulated across all six exits. Because planning scores are highly saturated and frequently tied, the scorer predicts metric components and primarily acts as an exit-quality verifier rather than enforcing a strict total ordering over trajectories.

After a rejected exit, only the previously unevaluated backbone blocks are executed, while hidden states and trajectory scores are reused. Let $p_{j}$ be the probability of terminating at exit $\ell_{j}$, $c^{\mathrm{bb}}_{\ell_{j}}$ the cumulative conditional-backbone cost up to that exit, and $c^{G}_{\ell_{m}}$ and $c^{S}_{\ell_{m}}$ the single-trajectory generation and scoring costs at exit $\ell_{m}$. The expected inference cost is

$$
\begin{split}\mathbb{E}[\mathcal{C}_{\eta}]=\sum_{j=1}^{J}p_{j}\biggl[c^{\mathrm{bb}}_{\ell_{j}}+\sum_{m=1}^{j}\bigl(c^{G}_{\ell_{m}}+c^{S}_{\ell_{m}}\bigr)\biggr].\end{split}
$$

The threshold $\eta$ is selected on the validation set to determine the desired quality–computation operating point.

Algorithm 1 Quality-Guided Layer-Adaptive Planning

 Encode the current image and text description from observation $o$

 Initialize accumulated trajectory pool $\mathcal{A}\leftarrow\emptyset$

 for $\ell\in\{5,9,15,18,22,30\}$ do

  Continue the conditional Wan forward to block $\ell$

  Decode one trajectory $\tau_{\ell}$

  Predict $\hat{\mathbf{r}}_{\ell}$ and cache $Q(\hat{\mathbf{r}}_{\ell})$

   $\mathcal{A}\leftarrow\mathcal{A}\cup\{\tau_{\ell}\}$

  Obtain $(\hat{q},\hat{\tau})$ from the best trajectory in $\mathcal{A}$

  if $\hat{q}\geq\eta$ or $\ell=30$ then

   return $\hat{\tau}$

  end if

 end for

### Training

#### Video-domain adaptation.

Each NAVSIM training sample contains the current front-camera frame and the next eight frames sampled at 2 Hz, forming a nine-frame clip over four seconds. The original $1600{\times}900$ images are resized to the Wan landscape resolution of $1280{\times}704$. The text condition is generated from deployment-available structured attributes, including map metadata, discretized ego speed, an observed-ego-motion-history-derived maneuver, and traffic density. The maneuver uses only past and current ego poses and kinematics, never a future-trajectory label. Samples without all eight temporally aligned future frames are discarded. Future frames supervise only the video objective during training; deployment uses the current frame and does not decode future images. The caption templates and preprocessing details are provided in Appendix C, especially Sections C.2 and C.3.

During imitation learning, the Wan backbone is adapted using LoRA [^11], while the trajectory projections and heads are optimized in full. The actor objective is

$$
\mathcal{L}_{\mathrm{actor}}=\lambda_{\mathrm{vid}}\mathcal{L}_{\mathrm{vid}}+\sum_{\ell\in\mathcal{E}}\lambda_{\ell}\mathcal{L}_{\mathrm{traj}}^{\ell},
$$

where $\mathcal{L}_{\mathrm{vid}}$ is the native Wan video-diffusion objective. Both video supervision and trajectory decoding use the same fixed video-noise index $s^{\star}=17$ and the same conditional Wan forward during joint adaptation; no second backbone forward is required. Video prediction is retained as representation supervision rather than a required deployment output.

#### Trajectory-quality scorer.

The scorer, including its DINOv2-Small encoder, is fine-tuned on generated trajectories using evaluator-provided component targets. All six components use equal-weight soft-label BCE-with-logits:

$$
\mathcal{L}_{\mathrm{score}}=\sum_{i}\sum_{m\in\mathcal{R}}\operatorname{BCE}_{\mathrm{logit}}\left(a_{i,m},r^{\mathrm{oracle}}_{i,m}\right),
$$

where each oracle component is used directly as a target in $[0,1]$, without binarization or component-specific regression losses. This component-wise soft-target formulation avoids imposing an artificial ordering among the many tied or near-perfect trajectories. Generated trajectories are treated as stop-gradient inputs during scorer training, such that

$$
\nabla_{\theta_{\mathrm{Wan}},\theta_{G}}\mathcal{L}_{\mathrm{score}}=0.
$$

Thus, the scorer does not alter the actor’s trajectory distribution or compete with the trajectory-generation objective. The actor and scorer are optimized in alternating updates on the same training stream, with generated trajectories detached before each scorer update. With equal branch weight, the complete training-stage objective can be written for bookkeeping as

$$
\mathcal{L}=\mathcal{L}_{\mathrm{actor}}+\mathcal{L}_{\mathrm{score}},
$$

with gradient isolation between the actor and scorer branches.

#### Planner-only reinforcement learning.

After imitation learning, we freeze the Wan backbone and quality scorer and refine each trajectory head using the DiffGRPO [^22]. A complete five-step action-denoising chain is treated as one trajectory action and receives its NAVSIM evaluator score as reward. No scorer gradient or routing decision is propagated to the actor during this stage. For each random seed, we select the validation-best checkpoint under the same training protocol; reported layer-wise statistics aggregate ten seeds. Optimizer settings, LoRA configuration, and loss weights are provided in Appendix D.

## 5 Experiments

### Setup

#### Datasets and metrics.

We follow the standard NAVSIM v1 and NAVSIM v2 navtest protocols [^7]. NAVSIM v1 reports Predictive Driver Model Score (PDMS), combining no-at-fault collision (NC), drivable-area compliance (DAC), time-to-collision (TTC), ego progress (EP), and comfort. NAVSIM v2 reports the extended EPDMS metric. Unless noted otherwise, our model uses the front camera without LiDAR and predicts eight poses over four seconds. All adaptive-routing and latency results decode one trajectory per attempted exit. NAVSIM-to-Wan clips use one anchor plus eight future front-camera frames at 0.5-second intervals; exact retained-sample counts after temporal filtering are reported in Appendix C.1. For zero-shot evaluation, a model trained on NAVSIM is evaluated on nuScenes without target-domain fine-tuning; we report average L2 displacement and collision rate following prior world-model planners.

#### Protocols.

Hardware latency is measured at batch size one on a single A100 80GB. End-to-end planning latency includes VAE image encoding, the conditional Wan forward, every attempted trajectory head, and scorer evaluation. The adaptive planner, fixed block-15 planner, and fixed full-depth planner average 170, 190, and 320 ms, respectively. For context, with cached text context, full 40-step classifier-free future-video generation takes 13.22 s under the same hardware setting, including 12.05 s of denoising and 1.17 s of VAE encoding/decoding.

### Main Results

<table><tbody><tr><td>Method</td><td>Input</td><td>NC</td><td>DAC</td><td>TTC</td><td>Comf.</td><td>EP</td><td>PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td colspan="8"><em>Traditional end-to-end planners</em></td></tr><tr><td>VADv2- <math><semantics><msub><mi>𝒱</mi> <mn>8192</mn></msub> <annotation>\mathcal{V}_{8192}</annotation></semantics></math> <sup><a href="#fn:4">4</a></sup></td><td>C</td><td>97.2</td><td>89.1</td><td>91.6</td><td>100</td><td>76.0</td><td>80.9</td></tr><tr><td>UniAD <sup><a href="#fn:14">14</a></sup></td><td>C</td><td>97.8</td><td>91.9</td><td>92.9</td><td>100</td><td>78.8</td><td>83.4</td></tr><tr><td>TransFuser <sup><a href="#fn:6">6</a></sup></td><td>CL</td><td>97.7</td><td>92.8</td><td>92.8</td><td>100</td><td>79.2</td><td>84.0</td></tr><tr><td>PARA-Drive <sup><a href="#fn:37">37</a></sup></td><td>C</td><td>97.9</td><td>92.4</td><td>93.0</td><td>99.8</td><td>79.3</td><td>84.0</td></tr><tr><td>ReCogDrive-IL <sup><a href="#fn:22">22</a></sup></td><td>C</td><td>98.1</td><td>94.7</td><td>94.2</td><td>100</td><td>80.9</td><td>86.5</td></tr><tr><td>DiffusionDrive <sup><a href="#fn:23">23</a></sup></td><td>CL</td><td>98.2</td><td>96.2</td><td>94.7</td><td>100</td><td>82.2</td><td>88.1</td></tr><tr><td colspan="8"><em>World-model planners</em></td></tr><tr><td>LAW <sup><a href="#fn:19">19</a></sup></td><td>C</td><td>96.4</td><td>95.4</td><td>88.7</td><td>99.9</td><td>81.7</td><td>84.6</td></tr><tr><td>Epona <sup><a href="#fn:43">43</a></sup></td><td>C</td><td>97.9</td><td>95.1</td><td>93.8</td><td>99.9</td><td>80.4</td><td>86.2</td></tr><tr><td>WoTE <sup><a href="#fn:21">21</a></sup></td><td>CL</td><td>98.5</td><td>96.8</td><td>94.9</td><td>99.9</td><td>81.9</td><td>88.3</td></tr><tr><td>DriveVLA-W0 <sup><a href="#fn:20">20</a></sup></td><td>C</td><td>98.4</td><td>95.3</td><td>95.2</td><td>100</td><td>80.9</td><td>87.2</td></tr><tr><td>PWM <sup><a href="#fn:44">44</a></sup></td><td>C</td><td>98.6</td><td>95.9</td><td>95.4</td><td>100</td><td>81.8</td><td>88.1</td></tr><tr><td>DriveVA <sup><a href="#fn:24">24</a></sup></td><td>C</td><td>99.2</td><td>97.5</td><td>98.7</td><td>100</td><td>83.5</td><td>90.5</td></tr><tr><td>Adaptive-WAM (single trajectory)</td><td>C</td><td>98.6</td><td>97.9</td><td>95.6</td><td>100</td><td>85.1</td><td>90.8</td></tr><tr><td>Adaptive-WAM (fixed B22, 64 prop.)</td><td>C</td><td>99.8</td><td>98.3</td><td>98.3</td><td>100</td><td>86.6</td><td>92.6</td></tr></tbody></table>

Table 2: NAVSIM v1 navtest PDMS. Baselines follow DriveVA; C/CL denote camera/camera+LiDAR; our 64-proposal result uses fixed B22 without routing.

| Method | Input | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | EPDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Human Agent | – | 100 | 100 | 99.8 | 100 | 87.4 | 100 | 100 | 98.1 | 90.1 | 90.3 |
| DiffusionDrive | CL | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | 84.5 |
| ReCogDrive | C | 98.3 | 95.2 | 99.5 | 99.8 | 87.1 | 97.5 | 96.6 | 98.3 | 86.5 | 83.6 |
| Epona | C | 97.1 | 95.7 | 99.3 | 99.7 | 88.6 | 96.3 | 97.0 | 98.0 | 67.8 | 85.1 |
| DriveVLA-W0 | C | 98.5 | 99.1 | 98.0 | 99.7 | 86.4 | 98.1 | 93.2 | 97.9 | 58.9 | 86.1 |
| Adaptive-WAM | C | 98.5 | 98.0 | 99.5 | 99.8 | 87.6 | 97.4 | 95.4 | 98.2 | 75.5 | 89.9 |

Table 3: NAVSIM v2 navtest EPDMS under the public protocol [^23] [^22] [^43] [^20]; C/CL denote camera/camera+LiDAR.

Tables 2 and 3 report the two benchmark versions separately. On NAVSIM v1, the adaptive single-trajectory planner obtains 90.8 PDMS. The auxiliary fixed-B22 64-proposal result reaches 92.6 PDMS, exceeding DriveVA’s mixed-data headline result by 1.7 points and its NAVSIM-only result by 2.1 points; its training details are provided in Appendix E. On NAVSIM v2, Adaptive-WAM reaches 89.9 EPDMS, outperforming the listed front-view world-model planners.

#### Zero-shot transfer.

Without nuScenes fine-tuning, Adaptive-WAM obtains 0.88 m average L2 error and 0.08% collision rate. DriveVA reports 0.84 m and 0.06% under the corresponding zero-shot setting. However, DriveVA executes the full Wan backbone and generates future images to support trajectory prediction, making its inference cost substantially higher and closer to a full Wan video-generation pipeline than to our single conditional early-exit planning pass. Table 4 retains only averages in the main paper; horizon-wise results and supervision details are provided in Appendix L.

<table><tbody><tr><th>Method</th><td>FT</td><td>Avg. L2 <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>Coll. (%) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><th colspan="4"><em>Target-domain tuned</em></th></tr><tr><th>ST-P3 <sup><a href="#fn:12">12</a></sup></th><td>✓</td><td>2.11</td><td>0.71</td></tr><tr><th>UniAD <sup><a href="#fn:14">14</a></sup></th><td>✓</td><td>1.03</td><td>0.31</td></tr><tr><th>OccNet <sup><a href="#fn:32">32</a></sup></th><td>✓</td><td>2.14</td><td>0.72</td></tr><tr><th>OccWorld <sup><a href="#fn:45">45</a></sup></th><td>✓</td><td>1.40</td><td>0.87</td></tr><tr><th>VAD-Tiny <sup><a href="#fn:16">16</a></sup></th><td>✓</td><td>1.30</td><td>0.72</td></tr><tr><th>VAD-Base <sup><a href="#fn:16">16</a></sup></th><td>✓</td><td>1.22</td><td>0.53</td></tr><tr><th>GenAD <sup><a href="#fn:46">46</a></sup></th><td>✓</td><td>0.91</td><td>0.43</td></tr><tr><th>Doe-1 <sup><a href="#fn:47">47</a></sup></th><td>✓</td><td>1.26</td><td>0.53</td></tr><tr><th>Epona <sup><a href="#fn:43">43</a></sup></th><td>✓</td><td>1.25</td><td>0.36</td></tr><tr><th colspan="4"><em>NAVSIM-to-nuScenes zero-shot</em></th></tr><tr><th>DriveVLA-W0 <sup><a href="#fn:20">20</a></sup></th><td>–</td><td>1.43</td><td>0.77</td></tr><tr><th>PWM <sup><a href="#fn:44">44</a></sup></th><td>–</td><td>3.99</td><td>0.36</td></tr><tr><th>DriveVA <sup><a href="#fn:24">24</a></sup></th><td>–</td><td>0.84</td><td>0.06</td></tr><tr><th>Adaptive-WAM</th><td>–</td><td>0.88</td><td>0.08</td></tr></tbody></table>

Table 4: nuScenes comparison using horizon-averaged L2 error and collision rate; FT denotes target-domain fine-tuning.

### Scorer Reliability and Adaptive Trade-off

The scorer is evaluated on an offline diagnostic pool covering 12,146 scenes. Exact top-score selection succeeds in 91.2% of scenes, and tie-aware soft accuracy reaches 94.4% when a selected trajectory within 5% of the true top score is accepted. Strict rank correlation is not informative here: more than 95% of scenes contain trajectory groups that are all perfect, all zero, or tied at the top. We instead stress-test consequential failures. Only 51 scenes (0.42%) select a trajectory at least 50% worse than an available near-perfect trajectory, and 69 scenes (0.57%) exceed a 20% gap.

| Policy | PDMS $\uparrow$ | Exit by B15 | Lat. (ms) |
| --- | --- | --- | --- |
| Fixed B15 | 90.62 | 100% | 190 |
| Adaptive $\eta=70$ | 88.49 | $98.8\%$ | 112 |
| Adaptive $\eta=80$ | 90.64 | $95.2\%$ | 143 |
| Adaptive $\eta=90$ | 90.79 | $94.1\%$ | 170 |
| Adaptive $\eta=95$ | 90.75 | $65.9\%$ | 284 |
| Full path | 85.82 | – | 320 |

Table 5: Adaptive performance–efficiency trade-off on one A100 at batch size one. Latency is end-to-end; Exit by B15 denotes termination within the first three exits. The full path includes one VAE decode; $\eta$ uses 100-point $Q$.

Table 5 reports the current threshold sweep. At $\eta=90$, the adaptive policy improves on fixed B15 by 0.17 PDMS while reducing average end-to-end planning latency by approximately 10%, from 190 to 170 ms. More than 94% of scenes terminate within the first three exits, and the fixed full-depth planner averages 320 ms. The permissive threshold 70 instead loses 2.13 points relative to fixed B15.

### Ablations and Efficiency

| Adaptation | PDMS | Visual backbone | PDMS |
| --- | --- | --- | --- |
| Wan frozen | 84.20 | ViT-S | 83.91 |
| Separate LoRA + cache | 84.95 | ViT-B | 85.62 |
| Joint Wan LoRA | 90.62 | ViT-L | 88.88 |
| Full Wan tuning | 90.64 | – | – |

Table 6: Single-trajectory adaptation and visual-backbone ablations on NAVSIM v1 (PDMS).

#### Adaptation and visual representation.

Table 6 shows that freezing Wan is insufficient. Fine-tuning Wan separately and then caching its features recovers only a small part of the loss, whereas joint LoRA training improves the single-trajectory model by 5.67 points over cached features. Full fine-tuning yields no meaningful gain over LoRA. Replacing Wan features with ViT-S/B/L lowers single-trajectory PDMS by 6.71/5.00/1.74 points, respectively.

#### Scorer backbone.

On the same offline diagnostic trajectory pool, DINO-Small obtains 92.59, compared with 92.54 for DINO-Base, 91.17/91.20 for ViT-S/B, and 92.19/92.55 for ResNet-34/50. Wan features offer no meaningful scorer advantage while substantially increasing online cost: the best Wan exit improves the diagnostic score by only 0.03 points over DINO-Small. We therefore fine-tune DINO-Small as the default scorer; the full scorer-backbone and video-index diagnostics are reported in Appendix H.

#### Computation.

A full 40-step classifier-free Wan rollout performs 80 DiT forwards and takes 13.22 s with cached text context: 12.05 s for iterative denoising and 1.17 s for VAE encoding/decoding. The planning path performs one conditional forward up to the selected exit, executing a five-step trajectory head and lightweight scorer at each attempted exit. On an A100, VAE image encoding costs approximately 50 ms. Including the conditional Wan forward, every attempted trajectory head, scorer evaluation, and VAE image encoding, the selected adaptive planner averages 170 ms end-to-end. The fixed block-15 planner averages 190 ms, while the fixed full-depth planner averages 320 ms and additionally performs one VAE decode. This comparison is kept separate from the 13.22 s full video-generation runtime.

## 6 Conclusion

In summary, intermediate video-DiT features are robust planning representations well before full image synthesis completes. By coupling multi-depth trajectory heads with a tie-aware trajectory-quality controller, Adaptive-WAM allocates world-model computation according to the quality of the current plan. It achieves strong NAVSIM and zero-shot nuScenes results while exposing a practical path from large generative world models to efficient driving policies.

## References

## Appendix A Appendix Roadmap

This appendix provides the details deferred from the main paper: NAVSIM evaluation and data construction, caption generation and image preprocessing, training and gradient isolation, the auxiliary fixed-exit 64-proposal model, extended video-noise and DiT-depth analyses, scorer diagnostics, adaptive routing, latency decomposition, and horizon-level nuScenes comparisons. Table 7 maps each explicit pointer in the main paper to its corresponding appendix section.

| Main-paper pointer | Appendix location |
| --- | --- |
| Caption templates and image preprocessing | Section C, especially Secs. C.2 and C.3 |
| Optimizer, LoRA, loss, and gradient-isolation details | Section D |
| NAVSIM-to-Wan temporal filtering and corpus accounting | Section C.1 |
| Fixed-B22 64-proposal training | Section E |
| Scorer backbone and video-index diagnostics | Section H |
| nuScenes horizon-wise comparison and supervision status | Section L |
| Adaptive routing and latency scope | Sections J and K |

Table 7: Coverage of details explicitly deferred from the main paper to the appendix.

## Appendix B Datasets, Metrics, and Evaluation

### B.1 NAVSIM Protocol

#### Input and output.

The default agent uses the latest CAM\_F0 image, ego state, and route-level navigation command; it does not use LiDAR. The planner predicts eight ego poses $\tau=\{(x_{t},y_{t},\theta_{t})\}_{t=1}^{8}$ at 0.5-second intervals in the ego-centric rear-axle frame, giving a four-second horizon. The route command is supplied by the benchmark and does not encode obstacle or traffic-light state.

#### NAVSIM v1.

We use the official navtest evaluation protocol [^7]. NAVSIM v1 evaluates each submitted trajectory through a four-second non-reactive simulation: background actors follow their recorded futures, while a controller rolls out the ego vehicle along the submitted plan. Its Predictive Driver Model Score (PDMS) combines multiplicative safety and feasibility penalties with a weighted measure of progress, time-to-collision, and comfort:

$$
\mathrm{PDMS}=\mathrm{NC}\cdot\mathrm{DAC}\cdot\frac{5\,\mathrm{EP}+5\,\mathrm{TTC}+2\,\mathrm{Comf}}{12}.
$$

Here NC denotes no-at-fault collision and DAC denotes drivable-area compliance. We use $Q=100\,\mathrm{PDMS}$ when expressing scorer thresholds on the 100-point scale used by the paper’s tables. Multiplicative safety terms and saturated sub-scores create many exact ties among otherwise different trajectories.

#### NAVSIM v2.

NAVSIM v2 extends this protocol with driving-direction compliance (DDC), traffic-light compliance (TLC), lane keeping (LK), history comfort (HC), and extended comfort (EC). NC and DDC take values in $\{0,\tfrac{1}{2},1\}$; DAC and TLC are binary multipliers. EP is continuous in $[0,1]$, while TTC, LK, HC, and EC are binary. The additive weights are 5 for EP and TTC and 2 for LK, HC, and EC.

NAVSIM v2 also filters false-positive penalties against the human agent. For a component $m$, let

$$
F_{m}(a,h)=\begin{cases}1,&m(h)=0,\\
m(a),&\text{otherwise},\end{cases}
$$

where $a$ and $h$ denote the submitted agent and the human reference. Thus, a violation that is also triggered by the human rollout is neutralized rather than attributed solely to the submitted planner. With $\mathcal{M}=\{\mathrm{NC,DAC,DDC,TLC}\}$ and $\mathcal{W}=\{\mathrm{TTC,EP,HC,LK,EC}\}$, the per-stage score is

$$
\mathrm{EPDMS}=\left(\prod_{m\in\mathcal{M}}F_{m}(a,h)\right)\frac{\sum_{m\in\mathcal{W}}w_{m}F_{m}(a,h)}{\sum_{m\in\mathcal{W}}w_{m}}.
$$

To approximate closed-loop behavior without interactive simulation, v2 uses a two-stage aggregation. It first scores the initial four-second scene, then scores precomputed follow-up scenes that begin from alternative end states. Follow-up scores are aggregated with a Gaussian kernel according to the distance between each follow-up start state and the submitted planner’s first-stage end state. The final result multiplies the first-stage score by this weighted second-stage score. The reported 89.9 EPDMS uses the official evaluator and a single front camera without LiDAR.

#### Split discipline.

NAVSIM training uses navtrain-derived records. The v1 and v2 test splits are used only for evaluation; no privileged map, future occupancy, or future sensor observation is supplied to the deployed planner. Privileged evaluator state is used only for offline metric computation and, for the auxiliary 64-proposal model, pseudo-expert construction.

### B.2 nuScenes Zero-Shot Protocol

The NAVSIM-trained model is evaluated on nuScenes [^3] without nuScenes fine-tuning. The model continues to use a single front view and predicts ego trajectories in the local coordinate frame. We follow prior world-model planners by reporting displacement error and collision rate at 1, 2, and 3 seconds and their horizon average. Section L separates methods trained on nuScenes from NAVSIM-to-nuScenes zero-shot methods.

## Appendix C NAVSIM-to-Wan Training Data

### C.1 Clip Construction and Temporal Filtering

For an anchor at time $t_{0}$, it retrieves the current CAM\_F0 frame and the next eight front-camera frames at the native NAVSIM/OpenScene rate of 2 Hz:

$$
\mathcal{V}=\{I(t_{0}),I(t_{0}+0.5),\ldots,I(t_{0}+4.0)\}.
$$

Timestamps and each frame’s relative time are stored in the metadata and checked when constructing the clip. An anchor is removed if any of the eight required future frames is absent, which commonly occurs near the end of a log, or if a referenced camera image cannot be loaded. The aligned trajectory target contains the eight future $(x,y,\theta)$ poses over the same four seconds.

Starting from 103,288 candidate navtrain tokens, temporal filtering retains 82,555 complete nine-frame clips and removes 20,733 candidates that cannot provide the required current-plus-eight-future sequence. NAVSIM does not define an official train/validation partition for this converted corpus, so we report corpus-level counts rather than presenting an internal model- selection partition as an official benchmark split.

| Field | Setting |
| --- | --- |
| Camera | CAM\_F0 |
| Source image | $1600{\times}900$ |
| Wan image | $1280{\times}704$ |
| Video target | 1 anchor + 8 future frames |
| Frame interval | 0.5 s (2 Hz) |
| Planning horizon | 4.0 s |
| Trajectory target | $8{\times}(x,y,\theta)$ |
| Storage | Metadata plus relative image paths |

Table 8: Construction of one NAVSIM-to-Wan training item. The nine-frame clip satisfies Wan2.2-TI2V-5B’s $4n+1$ frame-count constraint.

The metadata record stores the scenario token, log identifier, anchor timestamp, relative camera paths, camera intrinsics, ego velocity and acceleration, aligned future trajectory, and programmatic text description. The PyTorch dataset resolves image paths relative to the OpenScene root at load time.

### C.2 Image Preprocessing

The source $1600{\times}900$ RGB image is converted to Wan’s landscape resolution of $1280{\times}704$ before video-domain adaptation. The Wan branch receives RGB values mapped to the normalization expected by the frozen Wan VAE. The scorer has a separate image path: the same current front image is resized to $832{\times}480$ for the DINOv2 encoder and normalized with ImageNet statistics. Only the current image is used online; future images are training targets for the video objective and are never provided to the deployed planner. No image-captioning network, detector output, map raster, or future frame is used to create the deployed visual input.

### C.3 Programmatic Text Descriptions

The Wan text condition is generated from structured, auditable attributes rather than from a separate vision-language captioner. The final template is

> Vehicle {motion}{turn\_info} at {speed:.1f} m/s in urban environment.

The motion descriptor uses the current speed: below 0.5 m/s is *stationary*, 0.5–3 m/s is *slowly*, 3–8 m/s is *cruising*, and above 8 m/s is *fast*. A turn descriptor is added only when the observed historical lateral displacement exceeds 0.3 m. Current map-area and traffic-flow metadata are also incorporated by the description generator. Crucially, all fields are computed from map metadata and observations at or before the anchor time; neither the ground-truth future trajectory nor future images are queried. The prompt is stored with the scenario token, and the finite set of resulting T5 text embeddings is cached in both training and inference.

## Appendix D Architecture and Training Details

### D.1 Fixed Video-Noise Index and Multi-Exit Readout

Wan2.2-TI2V-5B [^33] contains 30 video-DiT blocks. The native video sampler has 40 noise steps, but the deployed planner executes one conditional Wan forward at fixed sampling index $s^{\star}=17$. The future part of the latent is initialized from scheduler noise at that index; the current image provides the conditioned latent slice. No observed future latent, unconditional classifier-free-guidance branch, or video VAE decoding is used by the planning path.

Hidden states are read after blocks

$$
\mathcal{E}=\{5,9,15,18,22,30\}.
$$

Each readout has its own projection and trajectory head. The six heads have independent parameters and do not exchange features or predictions, while their trajectory losses all backpropagate through the shared Wan LoRA during imitation learning. When routing continues from one exit to the next, the previously evaluated DiT prefix and its hidden state are reused.

### D.2 Single-Trajectory Diffusion Head

The adaptive system uses one ReCogDrive-style action DiT at every exit [^22]. Each head starts from a Gaussian trajectory latent and performs five DDIM updates [^30], conditioned on projected video-DiT features, the current ego state, and the navigation command. It decodes one eight-pose trajectory per attempted exit. Consequently, the cumulative pool contains at most six trajectories, one from each evaluated depth; it never contains $6\times 64$ proposals.

### D.3 Actor, Scorer, and Gradient Isolation

During imitation learning, the Wan backbone is adapted with LoRA [^11]; the trajectory projections and heads are fully trainable. The actor loss is

$$
\mathcal{L}_{\mathrm{actor}}=\lambda_{\mathrm{vid}}\mathcal{L}_{\mathrm{vid}}+\sum_{\ell\in\mathcal{E}}\lambda_{\ell}\mathcal{L}_{\mathrm{traj}}^{\ell}.
$$

The native video-diffusion objective and all trajectory heads use the same fixed video-noise index and conditional Wan forward, avoiding a second backbone pass.

The quality scorer receives detached trajectories. Thus,

$$
\nabla_{\theta_{\mathrm{Wan}},\theta_{G}}\mathcal{L}_{\mathrm{score}}=0,
$$

and scorer fitting cannot reshape the trajectory generator through proposal coordinates. Actor and scorer updates alternate on the same training stream.

After imitation learning, Wan and the scorer are frozen. Only the six trajectory heads are refined using the DiffGRPO formulation adopted by ReCogDrive: one complete five-step trajectory-denoising chain is treated as a composite action and receives the NAVSIM planning score as its reward. Each exit uses the same training budget and validation-best checkpoint rule. Layer statistics reported in the main paper aggregate ten runs.

### D.4 LoRA Placement and Reproducibility Manifest

LoRA modules are applied throughout the Wan backbone, rather than only at the six readout blocks shown schematically in the main architecture figure. The base Wan parameters and VAE remain frozen; projection layers, trajectory heads, and LoRA parameters are trainable during joint imitation learning. Table 9 records the settings established by the experiment records.

| Component | Setting | Scope |
| --- | --- | --- |
| Backbone | Wan2.2-TI2V-5B | Conditional branch; 30 video-DiT blocks |
| Video schedule | 40 sampling steps; planning index 17 | One conditional forward for planning and joint supervision |
| Exit blocks | 5, 9, 15, 18, 22, 30 | Independent projections and heads |
| Trajectory output | 8 poses at 0.5 s | Four-second horizon |
| Trajectory sampler | Five DDIM updates | Same ReCogDrive-style protocol at all exits |
| Adaptive proposals | One per attempted exit | At most six accumulated |
| Scorer | DINOv2-Small, fine-tuned at $832{\times}480$ | Current front camera plus candidate trajectory; no ego state or command |
| Checkpoint selection | Validation-best | Same rule for all exits |
| Planner RL | Wan and scorer frozen | Trajectory heads only |
| Randomness | Ten runs | Used for reported layer-wise analyses |

Table 9: Architecture and training configuration for the adaptive single-trajectory system.

The single-trajectory and auxiliary 64-proposal systems use the same Wan LoRA recipe; their planning decoders differ. The shared backbone-adaptation configuration is listed in Table 10. Non-LoRA Wan parameters and the VAE are frozen.

| Wan LoRA item | Setting |
| --- | --- |
| Target modules | $q,k,v,o$ |
| Rank / alpha / dropout | 32 / 64 / 0.05 |
| Optimizer | AdamW |
| Base learning rate | $2\times 10^{-4}$ |
| Per-GPU batch / GPUs | 5 / 4 |
| Gradient accumulation | 4 |
| Effective batch | 80 |
| Training length | 80 epochs |
| Precision / strategy | bf16-mixed / DDP |
| Planning index / frames | 17 / 9 |
| Wan input / feature dim. | $1280{\times}704$ / 3072 |

Table 10: Shared Wan-LoRA backbone-adaptation configuration.

Table 11 gives the ReCogDrive-style action-head schedule. The imitation stage trains the small action DiT and its $3072\!\to\!384$ feature projection with a 100-step diffusion training schedule and five DDIM steps at inference. The planner-only GRPO stage initializes both the current and frozen reference policies from the validation-best imitation checkpoint.

| Item | Imitation learning | Planner-only GRPO |
| --- | --- | --- |
| Optimizer / LR | AdamW, $10^{-4}$ | AdamW, $10^{-4}$ |
| Betas / weight decay | $(0.9,0.95)$ / $10^{-4}$ | $(0.9,0.95)$ / $10^{-4}$ |
| Schedule | 3-epoch warmup + cosine to $10^{-6}$ | cosine, no warmup, minimum 0 |
| Training length | 200 epochs | 50 epochs |
| Per-GPU batch / GPUs | 256 / 4 | 8 / 4 |
| Effective global batch | 1024 | 32 |
| Precision / grad. clip | 16-mixed / 1.0 | 16-mixed / 1.0 |
| DDP strategy | standard DDP | find-unused-parameters DDP |
| Trainable planner module | feature projection + action DiT | action DiT only |
| Checkpoint rule | lowest validation loss; retain top 5 | lowest validation loss; retain top 5 |

Table 11: Single-trajectory planner optimization.

For GRPO, each scene forms a group of eight sampled trajectories. Advantages are standardized within the group, denoising-step contributions are discounted by $\gamma=0.6$, and log probabilities are clamped to $[-5,2]$. The loss adds a behavior-cloning term with weight 0.1. Reward weights are 10/5/2/0 for progress/TTC/comfort/driving direction, with a four-second proposal horizon and 0.1-second simulator interval. Wan and the scorer remain frozen throughout this stage.

## Appendix E Auxiliary Fixed-B22 64-Proposal Model

The 92.6 PDMS result is an auxiliary fixed-exit comparator. It reads block 22 once, generates $K=64$ trajectories at that exit, scores the 64 trajectories, and returns the highest-scoring one. It does not use the adaptive controller, visit six exits, or combine proposal sets across depths.

#### Generator.

The fixed-exit model uses one learned token for each proposal. The current ego state is embedded into a 256-dimensional token and added to the proposal embeddings. A four-block trajectory decoder applies proposal self-attention, cross-attends to projected Wan scene tokens, and predicts an eight-pose $(x,y,\theta)$ trajectory from each token. The hidden dimension is 256, the feed-forward dimension is 1024, and each refinement block uses one attention head. The decoder uses 16 projected scene tokens, four generator refinements, four scorer refinements, and a two-pose long-horizon auxiliary target. This branch uses the shared LoRA configuration in Table 10: four GPUs, batch size 5 per GPU, four-step gradient accumulation, 80 epochs, and bf16 mixed precision.

#### Logged and pseudo-expert coverage.

The target construction follows the evaluator-filtered pseudo-expert protocol of CLOVER [^1]. Candidate families vary target speed, lateral offset, transition length, acceleration/deceleration, stop–go behavior, approach braking, and off-road boundary cases. Candidates first undergo inexpensive validity checks and are then scored with the true NAVSIM evaluator using training-time map and future occupancy. The retained target set covers both the logged trajectory and multiple high-quality alternatives.

| Pseudo-expert item | Setting |
| --- | --- |
| Future poses | 8 at 0.5 s |
| Speed candidates (m/s) | 0, 2, 4, 6, 8, 10, 12, 15 |
| Regular lateral offsets (m) | $-3.5$ to $3.5$ |
| Boundary offsets (m) | $\{-7.0,-5.5,5.5,7.0\}$ |
| Acceleration rates | $\{-2,-1,-0.5,0.5,1,2\}$ |
| Maximum scored per scene | 180 |
| Retained per scene | 50 |
| Coverage top- $K$ | 8 |
| Score threshold | 0.8 |
| Coverage-loss weight | 0.5 |

Table 12: CLOVER-derived pseudo-expert target-construction settings used by the auxiliary proposal-coverage training.

For every refinement output, the logged-trajectory term selects the closest of the 64 generated trajectories under mean posewise L1 distance. The pseudo-expert coverage term performs the reverse set assignment: for each retained pseudo expert, it selects the closest generated trajectory. The current configuration applies the pseudo-expert term to the final refinement output, uses the top eight pseudo experts above score 0.8, and weights this term by 0.5. The scorer uses evaluator-provided component labels on detached generated trajectories. The active logged-trajectory and final-score losses each have unit weight. This auxiliary training should not be confused with the five-step single-trajectory diffusion objective used by adaptive routing.

## Appendix F Extended Video-Noise Analysis

| Video index | 1 | 9 | 17 | 25 | 32 |
| --- | --- | --- | --- | --- | --- |
| B15, single | 86.44 | 86.56 | 86.57 | 86.55 | 86.50 |
| B18, single | 84.02 | 84.14 | 83.99 | 84.12 | 84.01 |
| Fixed B15, 64 prop. | 92.01 | 92.12 | 92.11 | 92.05 | 92.07 |
| Fixed B18, 64 prop. | 92.45 | 92.55 | 92.59 | 92.45 | 92.43 |

Table 13: Planning scores at five tested video sampling indices. The first two rows are single-trajectory imitation diagnostics; the final two rows use a fixed-exit 64-proposal model and no adaptive routing.

Across the five tested indices, the range is 0.13 PDMS at block 15 and 0.15 at block 18 for the single-trajectory planner. The 64-proposal diagnostic shows similarly small ranges of 0.11 and 0.14. These results support robustness to the tested noise levels; they do not claim invariance to every possible video timestep or scheduler. Index 17 is fixed for all subsequent analysis in the paper.

## Appendix G Extended DiT-Depth Analysis

### G.1 Fixed-Exit Planning Scores

| Block | 5 | 9 | 15 | 18 | 22 | 30 |
| --- | --- | --- | --- | --- | --- | --- |
| IL | 81.94 | 83.60 | 86.56 | 84.14 | 83.62 | 80.71 |
| RL | 86.02 | 87.56 | 90.62 | 88.92 | 87.42 | 85.82 |
| Gain | 4.08 | 3.96 | 4.06 | 4.78 | 3.80 | 5.11 |

Table 14: Layer-wise single-trajectory PDMS before and after planner-only RL. All exits use the same schedule and validation-best selection rule.

Block 15 is the strongest fixed exit in both stages, but later blocks retain scene-specific advantages. This distinction motivates routing by candidate quality rather than always choosing either block 15 or the final block.

### G.2 Post-RL High-Quality Scene Overlap

For run $r$ and exit $\ell$, define the high-quality set

$$
\mathcal{H}_{\ell}^{(r)}=\{o:Q_{\ell}^{(r)}(o)\geq 90\}.
$$

Table 15 reports the mean intersection-over-union of these sets across ten aligned runs.

| Jaccard | B5 | B9 | B15 | B18 | B22 | B30 |
| --- | --- | --- | --- | --- | --- | --- |
| B5 | 1.00 | 0.80 | 0.81 | 0.77 | 0.69 | 0.70 |
| B9 | 0.80 | 1.00 | 0.82 | 0.77 | 0.70 | 0.69 |
| B15 | 0.81 | 0.82 | 1.00 | 0.79 | 0.74 | 0.73 |
| B18 | 0.77 | 0.77 | 0.79 | 1.00 | 0.78 | 0.74 |
| B22 | 0.69 | 0.70 | 0.74 | 0.78 | 1.00 | 0.78 |
| B30 | 0.70 | 0.69 | 0.73 | 0.74 | 0.78 | 1.00 |

Table 15: Post-RL Jaccard overlap of scene sets with PDMS at least 90.

Off-diagonal overlaps range from 0.69 to 0.82. Thus, the exits solve substantially overlapping but non-identical scene sets: an early exit is sufficient for most scenes, while deeper exits can still recover cases not solved by the globally strongest intermediate block.

### G.3 Directional Large-Advantage Counts

For two exits $a$ and $b$, each cell reports

$$
N_{a\succ b}=N\!\left(Q_{a}(o)-Q_{b}(o)\geq 50\right),
$$

as mean $\pm$ standard deviation across ten paired runs. The two directions are recorded independently, so both $N_{a\succ b}$ and $N_{b\succ a}$ may be nonzero.

| Pre-RL | B5 | B9 | B15 | B18 | B22 | B30 |
| --- | --- | --- | --- | --- | --- | --- |
| B5 | $0.00{\pm}126.44$ | $61.53{\pm}25.75$ | $25.26{\pm}21.59$ | $68.54{\pm}20.43$ | $61.58{\pm}68.81$ | $74.92{\pm}89.06$ |
| B9 | $288.36{\pm}155.10$ | $0.00{\pm}88.15$ | $63.91{\pm}76.82$ | $83.19{\pm}76.55$ | $92.70{\pm}88.34$ | $99.36{\pm}59.51$ |
| B15 | $533.16{\pm}78.35$ | $332.64{\pm}86.85$ | $0.00{\pm}80.03$ | $245.04{\pm}93.14$ | $374.17{\pm}101.91$ | $554.80{\pm}182.82$ |
| B18 | $286.80{\pm}116.77$ | $98.12{\pm}88.96$ | $146.44{\pm}92.60$ | $0.00{\pm}103.51$ | $188.39{\pm}94.39$ | $484.68{\pm}159.61$ |
| B22 | $300.20{\pm}102.58$ | $220.70{\pm}107.93$ | $390.56{\pm}101.91$ | $212.03{\pm}95.93$ | $0.00{\pm}91.46$ | $199.04{\pm}89.43$ |
| B30 | $289.73{\pm}70.78$ | $219.95{\pm}169.59$ | $412.02{\pm}182.82$ | $329.40{\pm}172.49$ | $244.32{\pm}143.21$ | $0.00{\pm}87.68$ |

Table 16: Pre-RL directional large-advantage scene counts.

| Post-RL | B5 | B9 | B15 | B18 | B22 | B30 |
| --- | --- | --- | --- | --- | --- | --- |
| B5 | $0.00{\pm}71.01$ | $71.17{\pm}14.18$ | $31.78{\pm}15.27$ | $82.64{\pm}14.44$ | $57.31{\pm}23.31$ | $77.65{\pm}34.28$ |
| B9 | $289.78{\pm}59.45$ | $0.00{\pm}40.20$ | $65.45{\pm}33.31$ | $79.39{\pm}23.39$ | $101.42{\pm}53.33$ | $105.37{\pm}27.35$ |
| B15 | $544.55{\pm}34.13$ | $349.76{\pm}35.43$ | $0.00{\pm}38.50$ | $273.86{\pm}31.88$ | $385.83{\pm}47.39$ | $598.64{\pm}80.82$ |
| B18 | $284.43{\pm}58.53$ | $106.28{\pm}36.05$ | $158.57{\pm}41.92$ | $0.00{\pm}40.01$ | $204.86{\pm}33.06$ | $500.26{\pm}75.93$ |
| B22 | $317.12{\pm}52.87$ | $232.04{\pm}49.05$ | $407.75{\pm}43.11$ | $222.03{\pm}46.87$ | $0.00{\pm}35.57$ | $213.13{\pm}47.26$ |
| B30 | $293.34{\pm}29.04$ | $244.23{\pm}72.63$ | $422.41{\pm}84.59$ | $321.30{\pm}84.94$ | $257.82{\pm}60.43$ | $0.00{\pm}30.38$ |

Table 17: Post-RL directional large-advantage scene counts. Nonzero reverse directions show why a globally strong fixed exit is not uniformly best.

Block 15 has the largest directional counts against most other exits, but every comparison retains nonzero reverse advantages. The maximum cell-wise standard deviation decreases from 182.82 before RL to 84.94 afterward, while the broad depth ordering is unchanged. Together with the incomplete Jaccard overlap, this supports a quality-guided fallback: accept an already strong intermediate plan when possible, but continue to a deeper representation when its predicted quality is inadequate.

### G.4 Qualitative Layer-wise Trajectories

Figure 5 visualizes three recurring patterns behind the aggregate overlap statistics. The examples are diagnostic overlays of the six fixed-exit trajectories, not additional model inputs. They show that layer-wise differences can correspond to distinct decisions, while metric saturation can also assign identical high scores to visibly different but acceptable plans.

![[supp_early_vs_deep.png|Refer to caption]]

Figure 5: Selected layer-wise trajectory overlays. Trajectory colors encode scores: green marks high-scoring trajectories, whereas red marks erroneous trajectories. These cases illustrate both cross-depth complementarity and the score ties that motivate a verifier rather than a strict total-order ranker.

## Appendix H Trajectory-Quality Scorer

### H.1 Architecture and Objective

The deployed scorer fine-tunes a DINOv2-Small image encoder [^27]. It uses only the current front image and the candidate trajectory; ego state and navigation command are not scorer inputs. The $8{\times}3$ trajectory is flattened and embedded by an MLP. The trajectory and image features are concatenated, and six independent two-layer MLP heads output logits for NC, DAC, DDC, TTC, EP, and Comf.

For evaluator component targets $r^{\mathrm{oracle}}_{i,c}\in[0,1]$, the scorer uses equal-weight soft-label BCE:

$$
\begin{split}\mathcal{L}_{\mathrm{score}}=\sum_{i}\sum_{c\in\mathcal{R}}\operatorname{BCE}_{\mathrm{logit}}(a_{i,c},r^{\mathrm{oracle}}_{i,c}),\\
\mathcal{R}=\{\mathrm{NC,DAC,DDC,TTC,EP,Comf}\}.\end{split}
$$

Targets are not binarized. No global rank loss is used because the official composition contains many ties and near-ties. At inference, sigmoid component predictions are combined by $Q=100\,\Gamma$.

### H.2 Tie-Aware Reliability

The scorer diagnostic uses a fixed offline candidate pool covering 12,146 scenes. This pool is used only to compare scorers and is distinct from adaptive inference, which accumulates at most six trajectories. More than 95% of diagnostic scenes contain candidates that are jointly perfect, jointly zero, or tied at the top. We therefore report selection quality and consequential errors rather than a strict total-order correlation.

| Diagnostic | Rate |
| --- | --- |
| Exact top-score selection | 91.2% |
| Selection within 5 points | 94.4% |
| Failure with $\geq 50$ -point gap | 0.42% |
| Failure with $\geq 20$ -point gap | 0.57% |

Table 18: Tie-aware scorer diagnostics on 12,146 scenes.

### H.3 Video-Index and Backbone Diagnostics

| Wan exit | Index 1 | 9 | 17 | 25 | 32 |
| --- | --- | --- | --- | --- | --- |
| B15 | 92.01 | 92.12 | 92.11 | 92.05 | 92.07 |
| B18 | 92.45 | 92.55 | 92.59 | 92.45 | 92.43 |

Table 19: Wan-based scorer pretest at five video sampling indices. Entries are true selected-trajectory scores on the same cached candidate pool.

| Scorer backbone | Selected-trajectory score |
| --- | --- |
| Wan-B5 | 92.24 |
| Wan-B9 | 92.44 |
| Wan-B15 | 92.11 |
| Wan-B18 | 92.59 |
| Wan-B22 | 92.62 |
| Wan-B30 | 92.57 |
| DINO-Small | 92.59 |
| DINO-Base | 92.54 |
| ViT-Small | 91.17 |
| ViT-Base | 91.20 |
| ResNet-34 | 92.19 |
| ResNet-50 | 92.55 |

Table 20: Scorer-backbone diagnostic on a fixed candidate set. These values measure the true score of the selected candidate and are not end-to-end planner PDMS.

Wan scorer features are also insensitive to the five tested video indices. At index 17, the best Wan exit obtains 92.62, only 0.03 above DINO-Small. Because a Wan-based scorer would add a large world-model forward at every attempted exit, this negligible diagnostic difference does not justify its online cost. DINO-Small is therefore used as the quality verifier.

## Appendix I Additional Ablations

| Wan training | Single | Fixed B22, 64 prop. |
| --- | --- | --- |
| Frozen | 84.20 | 89.91 |
| Separate LoRA + cached features | 84.95 | 90.80 |
| Joint LoRA | 90.62 | 92.59 |
| Full fine-tuning | 90.64 | 92.54 |

Table 21: Effect of Wan adaptation. The 64-proposal column is a fixed-exit auxiliary model without adaptive routing.

Joint training is important: relative to training the head on separately cached features, joint LoRA improves the single-trajectory model by 5.67 points. Full Wan fine-tuning provides only 0.02 additional points, so LoRA is used for the main model.

| Visual backbone | Single | Fixed exit, 64 prop. |
| --- | --- | --- |
| ViT-Small | 83.91 | 92.17 |
| ViT-Base | 85.62 | 92.21 |
| ViT-Large | 88.88 | 92.31 |
| Wan intermediate features | 90.62 | 92.59 |

Table 22: Static visual features versus intermediate video-DiT features.

The single-trajectory comparison most directly exposes representation quality: Wan improves over ViT-Large by 1.74 points and over ViT-Small by 6.71. The gap narrows in the fixed 64-proposal setting because the scorer can choose among many candidates, but Wan features remain best.

## Appendix J Adaptive Routing

At each exit, one trajectory is decoded, scored, and added to the cumulative pool. The controller returns the highest-scoring accumulated trajectory as soon as its predicted quality exceeds threshold $\eta$. If no threshold is met, block 30 returns the highest-scoring trajectory among all six candidates.

| Policy | PDMS $\uparrow$ | Exit by B15 | Latency (ms) |
| --- | --- | --- | --- |
| Fixed B15 | 90.62 | 100.0% | 190 |
| Adaptive $\eta=70$ | 88.49 | 98.8% | 112 |
| Adaptive $\eta=80$ | 90.64 | 95.2% | 143 |
| Adaptive $\eta=90$ | 90.79 | 94.1% | 170 |
| Adaptive $\eta=95$ | 90.75 | 65.9% | 284 |
| Full path | 85.82 | 0.0% | 320 |

Table 23: Adaptive single-trajectory threshold sweep. “Exit by B15” denotes termination at blocks 5, 9, or 15. Latency is batch-one end-to-end planning time on one A100 80GB.

The selected threshold is $\eta=90$. It improves on the strongest fixed single-trajectory exit by 0.17 PDMS while reducing mean latency from 190 to 170 ms. The threshold sweep also shows why routing cannot be replaced by an unconditional shallow exit: the permissive $\eta=70$ policy is fastest but loses 2.13 points relative to fixed B15.

## Appendix K Latency Scope and Profiling

#### Planning latency.

End-to-end planning latency includes current-image VAE encoding, the conditional Wan prefix evaluated by the routing decision, every attempted five-step trajectory head, and every scorer evaluation. It excludes text encoding because text context is cached. The measured batch-one means on a single NVIDIA A100 80GB are 190 ms for fixed B15, 170 ms for adaptive routing at $\eta=90$, and 320 ms for the fixed full-depth path. The full-depth reference evaluates all 30 Wan blocks and includes one VAE decode.

#### Full video generation.

Full Wan synthesis follows a different computation graph: 40 denoising steps, a conditional and unconditional Wan forward at every step, classifier-free guidance, scheduler updates, and video VAE decoding. The profiling run uses one anchor plus eight future frames, giving nine frames, latent shape $[48,3,30,52]$, and sequence length 1170. The prompt vocabulary is finite and fixed, so its T5 embeddings are cached for both training and inference. The deployment-comparable video path therefore includes denoising and both VAE operations, but not online T5 encoding or file saving.

| Full-video component | Time |
| --- | --- |
| 40-step denoising loop | 12.05 s |
| Mean conditional DiT per step | 149.40 ms |
| Mean unconditional DiT per step | 147.80 ms |
| VAE image encoding | 0.27 s |
| VAE video decoding | 0.90 s |
| Cached-text video path, no saving | 13.22 s |
| T5 encoding in uncached profiling call | 8.34 s |
| Video saving | 2.87 s |
| Complete raw profiling call | 21.64 s |
| Peak allocated memory | 31.19 GiB |

Table 24: Measured decomposition of one nine-frame Wan generation run. The 13.22 s deployment-comparable total is $12.05+0.27+0.90$  s. Component timers in the raw profiling call are reported as instrumented and need not sum because stages and bookkeeping overlap.

The primary 320-to-170 ms claim compares two planning paths and is therefore kept separate from full video generation. The large synthesis cost arises from 80 Wan DiT forwards ($40$ steps $\times$ conditional/unconditional branches), whereas the planner uses one conditional pass only up to the selected depth and never decodes a future video.

## Appendix L Detailed nuScenes Comparison

<table><tbody><tr><th>Method</th><td colspan="4">L2 error (m) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><th></th><td>1 s</td><td>2 s</td><td>3 s</td><td>Avg.</td></tr><tr><th colspan="5"><em>Target-domain tuned</em></th></tr><tr><th>ST-P3 <sup><a href="#fn:12">12</a></sup></th><td>1.33</td><td>2.11</td><td>2.90</td><td>2.11</td></tr><tr><th>UniAD <sup><a href="#fn:14">14</a></sup></th><td>0.48</td><td>0.96</td><td>1.65</td><td>1.03</td></tr><tr><th>OccNet <sup><a href="#fn:32">32</a></sup></th><td>1.29</td><td>2.13</td><td>2.99</td><td>2.14</td></tr><tr><th>OccWorld <sup><a href="#fn:45">45</a></sup></th><td>0.52</td><td>1.27</td><td>2.41</td><td>1.40</td></tr><tr><th>VAD-Tiny <sup><a href="#fn:16">16</a></sup></th><td>0.60</td><td>1.23</td><td>2.06</td><td>1.30</td></tr><tr><th>VAD-Base <sup><a href="#fn:16">16</a></sup></th><td>0.54</td><td>1.15</td><td>1.98</td><td>1.22</td></tr><tr><th>GenAD <sup><a href="#fn:46">46</a></sup></th><td>0.36</td><td>0.83</td><td>1.55</td><td>0.91</td></tr><tr><th>Doe-1 <sup><a href="#fn:47">47</a></sup></th><td>0.50</td><td>1.18</td><td>2.11</td><td>1.26</td></tr><tr><th>Epona <sup><a href="#fn:43">43</a></sup></th><td>0.61</td><td>1.17</td><td>1.98</td><td>1.25</td></tr><tr><th colspan="5"><em>NAVSIM-to-nuScenes zero-shot</em></th></tr><tr><th>DriveVLA-W0 <sup><a href="#fn:20">20</a></sup></th><td>0.43</td><td>1.26</td><td>2.60</td><td>1.43</td></tr><tr><th>PWM <sup><a href="#fn:44">44</a></sup></th><td>2.06</td><td>3.91</td><td>6.00</td><td>3.99</td></tr><tr><th>DriveVA <sup><a href="#fn:24">24</a></sup></th><td>0.33</td><td>0.76</td><td>1.43</td><td>0.84</td></tr><tr><th>Adaptive-WAM</th><td>0.35</td><td>0.71</td><td>1.58</td><td>0.88</td></tr></tbody></table>

Table 25: Horizon-level nuScenes L2 comparison. Published baseline values follow the DriveVA protocol.

<table><tbody><tr><th>Method</th><td colspan="4">Collision rate (%) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><th></th><td>1 s</td><td>2 s</td><td>3 s</td><td>Avg.</td></tr><tr><th colspan="5"><em>Target-domain tuned</em></th></tr><tr><th>ST-P3 <sup><a href="#fn:12">12</a></sup></th><td>0.23</td><td>0.62</td><td>1.27</td><td>0.71</td></tr><tr><th>UniAD <sup><a href="#fn:14">14</a></sup></th><td>0.05</td><td>0.17</td><td>0.71</td><td>0.31</td></tr><tr><th>OccNet <sup><a href="#fn:32">32</a></sup></th><td>0.21</td><td>0.59</td><td>1.37</td><td>0.72</td></tr><tr><th>OccWorld <sup><a href="#fn:45">45</a></sup></th><td>0.12</td><td>0.40</td><td>2.08</td><td>0.87</td></tr><tr><th>VAD-Tiny <sup><a href="#fn:16">16</a></sup></th><td>0.31</td><td>0.53</td><td>1.33</td><td>0.72</td></tr><tr><th>VAD-Base <sup><a href="#fn:16">16</a></sup></th><td>0.04</td><td>0.39</td><td>1.17</td><td>0.53</td></tr><tr><th>GenAD <sup><a href="#fn:46">46</a></sup></th><td>0.06</td><td>0.23</td><td>1.00</td><td>0.43</td></tr><tr><th>Doe-1 <sup><a href="#fn:47">47</a></sup></th><td>0.04</td><td>0.37</td><td>1.19</td><td>0.53</td></tr><tr><th>Epona <sup><a href="#fn:43">43</a></sup></th><td>0.01</td><td>0.22</td><td>0.85</td><td>0.36</td></tr><tr><th colspan="5"><em>NAVSIM-to-nuScenes zero-shot</em></th></tr><tr><th>DriveVLA-W0 <sup><a href="#fn:20">20</a></sup></th><td>0.22</td><td>0.66</td><td>1.42</td><td>0.77</td></tr><tr><th>PWM <sup><a href="#fn:44">44</a></sup></th><td>0.12</td><td>0.15</td><td>0.86</td><td>0.36</td></tr><tr><th>DriveVA <sup><a href="#fn:24">24</a></sup></th><td>0.00</td><td>0.07</td><td>0.12</td><td>0.06</td></tr><tr><th>Adaptive-WAM</th><td>0.00</td><td>0.09</td><td>0.15</td><td>0.08</td></tr></tbody></table>

Table 26: Horizon-level nuScenes collision-rate comparison.

The zero-shot group is trained on NAVSIM and evaluated without nuScenes fine-tuning. Adaptive-WAM obtains 0.88 m average L2 error and 0.08% average collision rate using a single front camera. DriveVA is slightly better on these two averages, but its inference executes the full Wan backbone and generates future images to support planning, whereas Adaptive-WAM performs a single conditional early-exit planning pass.

## Appendix M Failure Analysis and Interpretation

The scorer diagnostic contains 51 large failures at a 50-point gap and 69 at a 20-point gap among 12,146 scenes. These events are rare but important because the scorer determines whether additional world-model computation is needed. They should be interpreted together with the threshold sweep: a stricter threshold decreases early exits but cannot guarantee safety. Adaptive-WAM remains a learned offline planner, and passing the quality threshold is not a formal safety certificate.

[^1]: S. Ang, Y. Yang, C. Chen, and Y. Wang CLOVER: closed-loop value estimation and ranking for end-to-end autonomous driving planning. arXiv preprint arXiv:2605.15120. Cited by: Appendix E.

[^2]: F. Bartoccioni, E. Ramzi, V. Besnier, S. Venkataramanan, T. Vu, Y. Xu, L. Chambon, S. Gidaris, S. Odabas, D. Hurych, et al. Vavim and vavam: autonomous driving through video generative modeling. arXiv preprint arXiv:2502.15672. Cited by: §2.

[^3]: H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom Nuscenes: a multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 11621–11631. Cited by: §B.2, §1.

[^4]: S. Chen, B. Jiang, H. Gao, B. Liao, Q. Xu, Q. Zhang, C. Huang, W. Liu, and X. Wang Vadv2: end-to-end vectorized autonomous driving via probabilistic planning. arXiv preprint arXiv:2402.13243. Cited by: Table 2.

[^5]: Y. Chen, Y. Wang, and Z. Zhang Drivinggpt: unifying driving world modeling and planning with multi-modal autoregressive transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 26890–26900. Cited by: §1, §2.

[^6]: K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger Transfuser: imitation with transformer-based sensor fusion for autonomous driving. IEEE transactions on pattern analysis and machine intelligence 45 (11), pp. 12878–12895. Cited by: Table 2.

[^7]: D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, et al. Navsim: data-driven non-reactive autonomous vehicle simulation and benchmarking. Advances in Neural Information Processing Systems 37, pp. 28706–28719. Cited by: §B.1, §1, §5.

[^8]: R. Gao, K. Chen, E. Xie, L. Hong, Z. Li, D. Yeung, and Q. Xu Magicdrive: street view generation with diverse 3d geometry control. In International Conference on Learning Representations, Vol. 2024, pp. 22841–22860. Cited by: §1, §2.

[^9]: S. Gao, J. Yang, L. Chen, K. Chitta, Y. Qiu, A. Geiger, J. Zhang, and H. Li Vista: a generalizable driving world model with high fidelity and versatile controllability. Advances in Neural Information Processing Systems 37, pp. 91560–91596. Cited by: §1, §2.

[^10]: A. Hu, L. Russell, H. Yeo, Z. Murez, G. Fedoseev, A. Kendall, J. Shotton, and G. Corrado Gaia-1: a generative world model for autonomous driving. arXiv preprint arXiv:2309.17080. Cited by: §1, §2.

[^11]: E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen, et al. Lora: low-rank adaptation of large language models.. Iclr 1 (2), pp. 3. Cited by: §D.3, §4.

[^12]: S. Hu, L. Chen, P. Wu, H. Li, J. Yan, and D. Tao St-p3: end-to-end vision-based autonomous driving via spatial-temporal feature learning. In European Conference on Computer Vision, pp. 533–549. Cited by: Table 25, Table 26, Table 4.

[^13]: X. Hu, W. Yin, M. Jia, J. Deng, X. Guo, Q. Zhang, X. Long, and P. Tan DrivingWorld: constructing world model for autonomous driving via video gpt. arXiv preprint arXiv:2412.19505. Cited by: §2.

[^14]: Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, et al. Planning-oriented autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 17853–17862. Cited by: Table 25, Table 26, Table 2, Table 4.

[^15]: G. Huang, D. Chen, T. Li, F. Wu, L. Van Der Maaten, and K. Weinberger Multi-scale dense networks for resource efficient image classification. In International conference on learning representations, Cited by: §2.

[^16]: B. Jiang, S. Chen, Q. Xu, B. Liao, J. Chen, H. Zhou, Q. Zhang, W. Liu, C. Huang, and X. Wang Vad: vectorized scene representation for efficient autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 8340–8350. Cited by: Table 25, Table 25, Table 26, Table 26, Table 4, Table 4.

[^17]: B. Li, Z. Ma, D. Du, B. Peng, Z. Liang, Z. Liu, X. Guo, Z. Zhu, C. Ma, Y. Jin, et al. Omninwm: omniscient driving navigation world models. arXiv preprint arXiv:2510.18313. Cited by: §2.

[^18]: X. Li, Y. Zhang, and X. Ye DrivingDiffusion: layout-guided multi-view driving scenarios video generation with latent diffusion model. In European Conference on Computer Vision, pp. 469–485. Cited by: §2.

[^19]: Y. Li, L. Fan, J. He, Y. Wang, Y. Chen, Z. Zhang, and T. Tan Enhancing end-to-end autonomous driving with latent world model. In International Conference on Learning Representations, Vol. 2025, pp. 42942–42959. Cited by: §2, Table 2.

[^20]: Y. Li, S. Shang, W. Liu, B. Zhan, H. Wang, Y. Wang, Y. Chen, X. Wang, Y. An, C. Tang, et al. DriveVLA-w0: world models amplify data scaling law in autonomous driving. arXiv preprint arXiv:2510.12796. Cited by: Table 25, Table 26, §2, Table 2, Table 3, Table 4.

[^21]: Y. Li, Y. Wang, Y. Liu, J. He, L. Fan, and Z. Zhang End-to-end driving with online trajectory evaluation via bev world model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 27137–27146. Cited by: Table 2.

[^22]: Y. Li, K. Xiong, X. Guo, F. Li, S. Yan, G. Xu, L. Zhou, L. Chen, H. Sun, B. Wang, et al. Recogdrive: a reinforced cognitive framework for end-to-end autonomous driving. arXiv preprint arXiv:2506.08052. Cited by: §D.2, §1, §2, §4, Table 2, Table 3.

[^23]: B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang, et al. Diffusiondrive: truncated diffusion model for end-to-end autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 12037–12047. Cited by: §2, Table 2, Table 3.

[^24]: M. Liu, D. Zhang, J. Liu, J. Cui, H. Xie, G. Chen, H. Ye, M. Y. Yang, F. Nex, and H. Cheng Driveva: video action models are zero-shot drivers. arXiv preprint arXiv:2604.04198. Cited by: Table 25, Table 26, §1, §2, Table 2, Table 4.

[^25]: T. Ma, J. Zheng, Z. Wang, C. Jiang, A. Cui, J. Liang, and S. Yang Dit4dit: jointly modeling video dynamics and actions for generalizable robot control. arXiv preprint arXiv:2603.10448. Cited by: §2.

[^26]: C. Min, D. Zhao, L. Xiao, J. Zhao, X. Xu, Z. Zhu, L. Jin, J. Li, Y. Guo, J. Xing, et al. Driveworld: 4d pre-trained scene understanding via world models for autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 15522–15533. Cited by: §2.

[^27]: M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al. Dinov2: learning robust visual features without supervision. Transactions on Machine Learning Research Journal. Cited by: §H.1, §1.

[^28]: W. Peebles and S. Xie Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4195–4205. Cited by: §4.

[^29]: Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu, et al. Deepseekmath: pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300. Cited by: §2.

[^30]: J. Song, C. Meng, and S. Ermon Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502. Cited by: §D.2, §4.

[^31]: S. Teerapittayanon, B. McDanel, and H. Kung Branchynet: fast inference via early exiting from deep neural networks. arXiv preprint arXiv:1709.01686. Cited by: §2.

[^32]: W. Tong, C. Sima, T. Wang, L. Chen, S. Wu, H. Deng, Y. Gu, L. Lu, P. Luo, D. Lin, et al. Scene as occupancy. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 8406–8415. Cited by: Table 25, Table 26, Table 4.

[^33]: T. Wan, A. Wang, B. Ai, B. Wen, C. Mao, C. Xie, D. Chen, F. Yu, H. Zhao, J. Yang, et al. Wan: open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314. Cited by: §D.1, §1, §1, §4.

[^34]: H. Wang, D. Liu, H. Xie, H. Liu, E. Ma, K. Yu, L. Wang, and B. Wang MiLA: multi-view intensive-fidelity long-term video generation world model for autonomous driving. arXiv preprint arXiv:2503.15875. Cited by: §2.

[^35]: X. Wang, Z. Zhu, G. Huang, X. Chen, J. Zhu, and J. Lu Drivedreamer: towards real-world-drive world models for autonomous driving. In European conference on computer vision, pp. 55–72. Cited by: §2.

[^36]: Y. Wen, Y. Zhao, Y. Liu, F. Jia, Y. Wang, C. Luo, C. Zhang, T. Wang, X. Sun, and X. Zhang Panacea: panoramic and controllable video generation for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6902–6912. Cited by: §2.

[^37]: X. Weng, B. Ivanovic, Y. Wang, Y. Wang, and M. Pavone Para-drive: parallelized architecture for real-time autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15449–15458. Cited by: Table 2.

[^38]: T. Xia, Y. Li, L. Zhou, J. Yao, K. Xiong, H. Sun, B. Wang, K. Ma, G. Chen, H. Ye, et al. Drivelaw: unifying planning and video generation in a latent driving world. arXiv preprint arXiv:2512.23421. Cited by: §1, §2, §2.

[^39]: Z. Xing, X. Zhang, Y. Hu, B. Jiang, T. He, Q. Zhang, X. Long, and W. Yin Goalflow: goal-driven flow matching for multimodal trajectories generation in end-to-end autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 1602–1611. Cited by: §2.

[^40]: J. Yang, K. Chitta, S. Gao, L. Chen, Y. Shao, X. Jia, H. Li, A. Geiger, X. Yue, and L. Chen Resim: reliable world simulation for autonomous driving. Advances in Neural Information Processing Systems 38, pp. 167710–167741. Cited by: §2.

[^41]: T. Yuan, Z. Dong, Y. Liu, and H. Zhao Fast-wam: do world action models need test-time future imagination?. arXiv preprint arXiv:2603.16666. Cited by: §1, §2.

[^42]: S. Zeng, X. Chang, M. Xie, X. Liu, Y. Bai, Z. Pan, M. Xu, and X. Wei Futuresightdrive: thinking visually with spatio-temporal cot for autonomous driving. Advances in Neural Information Processing Systems 38, pp. 67299–67318. Cited by: §2.

[^43]: K. Zhang, Z. Tang, X. Hu, X. Pan, X. Guo, Y. Liu, J. Huang, L. Yuan, Q. Zhang, X. Long, et al. Epona: autoregressive diffusion world model for autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 27220–27230. Cited by: Table 25, Table 26, §2, Table 2, Table 3, Table 4.

[^44]: Z. Zhao, T. Fu, Y. Wang, L. Wang, and H. Lu From forecasting to planning: policy world model for collaborative state-action prediction. Advances in Neural Information Processing Systems 38, pp. 134585–134611. Cited by: Table 25, Table 26, §2, Table 2, Table 4.

[^45]: W. Zheng, W. Chen, Y. Huang, B. Zhang, Y. Duan, and J. Lu Occworld: learning a 3d occupancy world model for autonomous driving. In European conference on computer vision, pp. 55–72. Cited by: Table 25, Table 26, Table 4.

[^46]: W. Zheng, R. Song, X. Guo, C. Zhang, and L. Chen Genad: generative end-to-end autonomous driving. In European Conference on Computer Vision, pp. 87–104. Cited by: Table 25, Table 26, Table 4.

[^47]: W. Zheng, Z. Xia, Y. Huang, S. Zuo, J. Zhou, and J. Lu Doe-1: closed-loop autonomous driving with large world model. arXiv preprint arXiv:2412.09627. Cited by: Table 25, Table 26, Table 4.

[^48]: Y. Zheng, R. Liang, K. Zheng, J. Zheng, L. Mao, J. Li, W. Gu, R. Ai, S. Li, X. Zhan, et al. Diffusion-based planning for autonomous driving with flexible guidance. In International conference on learning representations, Vol. 2025, pp. 37207–37227. Cited by: §2.
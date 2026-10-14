---
title: "ManeuverFlow: Maneuver Hypothesis Generation with Future Evidence for Trajectory Evaluation and Refinement in End-to-End Autonomous Driving"
source: "https://arxiv.org/html/2610.13409v1"
author:
published:
created: 2026-10-14
description:
tags:
  - "clippings"
---
Yuhao Wei    Yong Chen    Chen Yang    Yingchen Wang    Delin Ouyang    Lingfeng Qi    Qi Lan    Zhanhua Zhang    Jie Li    Guofa Li

###### Abstract

End-to-end planning in complex scenes must model diverse driving intentions and select the most appropriate trajectory using scene context and future risks. However, multi-query regression may collapse to one mode, while anchor-based and generative planners share a scaling bottleneck: small candidate sets limit maneuver coverage or discovery of high-quality trajectories, whereas large sets increase generation and scoring costs. We propose ManeuverFlow to generate trajectories for distinct intentions in one step and evaluate and refine them using current and future evidence. The model selects relevant maneuver modes, draws corresponding noise from the Gaussian mixture prior, and maps it to intention-specific trajectories through a maneuver-routed mixture-of-experts MeanFlow field. Current features and future features predicted by the world model are aggregated along each candidate path into trajectory-specific evidence for PDM-factor prediction. Using the same evidence, the refiner predicts local Frenet residuals for selected candidates; refined and retained candidates are then jointly re-scored for selection. With a lightweight TransFuser-style ResNet-34 backbone, ManeuverFlow achieves 90.6 PDMS and 90.1 EPDMS on NAVSIM v1/v2 navtest, respectively. On WorldEngine’s recent 3DGS closed-loop benchmark, it attains the highest listed PDMS <sup>∗</sup> scores of 75.19 and 74.58 under logged and IDM-controlled traffic, respectively. Code: attr/Border \[0 0 0\] user/Subtype /Link /A << /S /URI /URI (https://github.com/wyh-autonomy/maneuverflow) >>https://github.com/wyh-autonomy/maneuverflow.

<sup>1</sup> Chongqing University

<sup>2</sup> Artificial Intelligence Center, Geely Automotive Research Institute

<sup>1</sup>

## 1 Introduction

End-to-end autonomous driving maps sensor observations, ego states, and navigation cues directly to future ego motion, unifying perception and planning under a single objective [^3] [^8] [^9] [^14] [^12] [^26]. Complex traffic scenes often admit multiple plausible driving hypotheses, so a single trajectory cannot capture the underlying decision space. Multimodal motion forecasting represents this ambiguity with fixed anchor vocabularies or mode-specific queries [^2] [^25]; recent planners further use learned queries, trajectory anchors, and diffusion- or flow-based generators to produce candidates, then select a final plan through candidate–scene interactions [^13] [^20] [^21] [^30] [^28]. As shown in Fig. 1(a), learned queries without explicit mode-level structure may concentrate in similar regions of trajectory space, producing redundant candidates and risking mode collapse.

![[fg.png|Refer to caption]]

Figure 1: Candidate-based planning paradigms. (a) Query candidates without explicit mode structure, scored from current features. (b) Fixed-anchor candidates evaluated with per-candidate future rollouts. (c) ManeuverFlow with maneuver-structured generation, shared-future evidence, and Frenet refinement before joint re-scoring.

![[appendix_sampling_oracle_draft.png|Refer to caption]]

Figure 2: Post-hoc candidate-quality analysis of the frozen Maneuver-Mixture MeanFlow Field (M 3 F) across 12,146 NAVSIM v1 navtest scenes, excluded from training and model selection. Curves report best and Top- κ \\kappa mean PDMS over nested pools; HumanAgent 94.8 is a dashed reference.

Standard diffusion and ODE-based flow sampling is iterative [^22] [^23], although recent driving planners reduce the process to two steps in DiffusionDrive [^21] or one step in GoalFlow and MeanFuser [^30] [^28]. MeanFuser generates multiple proposals in one MeanFlow step, but its Adaptive Reconstruction Module (ARM) attends to and fuses them into a single output trajectory; the initial hypotheses thus do not remain explicit candidates for downstream evaluation, refinement, and selection [^28]. Fixed anchors encode predefined modes but may allocate part of the candidate budget to modes weakly relevant to the current scene. For a frozen generator, Fig. 2 retrospectively shows that larger pools expose stronger candidates. This motivates improving candidate quality under a compact proposal budget through scene-adaptive hypothesis generation and evidence-conditioned local trajectory refinement.

Candidate generation also requires reliable selection. Candidate-tree and generation–evaluation planners pair ego hypotheses with conditional predictions or learned evaluators [^10] [^11], whereas current-only scorers do not explicitly query predicted future scene states along candidate paths. WoTE provides such evidence through one future rollout per candidate (Fig. 1(b)), repeating future BEV prediction across candidate trajectories and multiple future time steps [^19].

Together, these limitations define a fixed-budget hypothesis-lifecycle problem. We therefore ask: How can a planner generate scene-relevant, mode-distinct hypotheses under a fixed budget and use current/future evidence to evaluate, refine, and select them?

To this end, we propose ManeuverFlow, a unified *Hypothesize–Verify–Improve* framework that preserves explicit maneuver-indexed hypotheses, as shown in Figs. 1(c) and 3. The Maneuver Gaussian Mixture Prior (M-GMP) predicts scene-conditioned weights over motion-defined components, and M <sup>3</sup> F generates base hypotheses by maneuver-routed one-step transport. Once per planning step, the world model predicts one candidate-independent future BEV field; a learned trajectory-guided probe projects each hypothesis into the current and predicted fields to form path-specific Trajectory Evidence Memory. Metric Prism pre-scores candidates through learned PDM factors. The Evidence-Coupled Frenet Refiner (ECFR) uses the same evidence to predict local Frenet residuals for selected hypotheses. Refined hypotheses replace selected bases, re-query the shared field, and compete with retained hypotheses under the same Metric Prism scoring function.

![[fg2.png|Refer to caption]]

Figure 3: Overview of ManeuverFlow. Scene-conditioned M-GMP and the shared planning context drive one-step M 3 F generation of base hypotheses, which Metric Prism pre-scores using trajectory-specific evidence from a shared future BEV field. ECFR refines the selected R hypotheses, which join the retained bases for evidence re-query and joint re-scoring.

Our main contributions are summarized as follows:

- Scene-adaptive maneuver generation. We introduce a maneuver-structured generator that combines M-GMP with the maneuver-routed M <sup>3</sup> F to model motion-defined maneuver components and generate scene-adaptive, intent-conditioned trajectory hypotheses as explicit alternatives in one step.
- Future-evidence verification and refinement. A learned probe projects each candidate onto a shared future BEV field to extract trajectory-specific evidence for PDM-aligned scoring and ECFR’s local Frenet refinement. Metric Prism jointly re-scores refined and retained hypotheses for final selection.
- With a lightweight ResNet-34 TransFuser (TF) perception backbone, ManeuverFlow-TF achieves 90.58 PDMS and 90.14 EPDMS on NAVSIM v1/v2 navtest, respectively. Its camera-only Latent TransFuser (LTF) variant further obtains 75.19 and 74.58 PDMS <sup>∗</sup> under replayed and reactive traffic on WorldEngine, delivering the highest listed composite planning scores across all reported benchmark and traffic settings.

## 2 Related Work

### 2.1 End-to-End and Generative Planning

End-to-end planners predict ego motion from sensor observations, ego states, and navigation commands. TransFuser fuses camera and LiDAR features, while later systems introduce planning-oriented queries and sparse scene representations [^3] [^9] [^14] [^26]. Generative planners model multiple feasible decisions through structured latent spaces, diffusion, or flow matching: GenAD learns a trajectory latent space, Diffusion Planner and DiffusionDrive denoise trajectory distributions, and GoalFlow applies goal-conditioned flow matching [^35] [^36] [^21] [^30]. MeanFuser combines a Gaussian-mixture source with one-step MeanFlow and aggregates sampled proposals into a single ARM output [^5] [^28]. MM-FM jointly designs a Gaussian-mixture source and mode-dependent coupling for general multimodal generation [^24]. A challenge is to organize driving-specific motion modes and retain diverse hypotheses as an explicit interface for evaluation and refinement.

### 2.2 World Models for Planning

Driving world models predict visual observations, occupancy, or latent states to support planning. MILE jointly learns latent dynamics and a driving policy, Think2Drive trains a planner through interaction with a compact latent world model, and LAW learns self-supervised latent dynamics [^7] [^16] [^18]. Drive-WM, Drive-OccWorld, World4Drive, and WorldRFT use predicted visual, occupancy, or latent futures for planning evaluation and optimization [^29] [^32] [^37] [^31]. For online evaluation, WoTE runs a recurrent BEV world model for each anchor-refined candidate and selects the highest-scoring trajectory [^19]. Candidate-conditioned rollouts provide trajectory-specific futures but require separate computation for every candidate. Shared predictions, such as the future occupancy used by ST-P3, amortize forecasting across candidates [^8] but do not directly provide path-specific evidence. Extracting discriminative, trajectory-specific evidence from a shared forecast without repeated candidate-wise rollouts therefore remains a key challenge.

### 2.3 Trajectory Refinement and Selection

Candidate-based planning requires both reliable discrimination and local geometric improvement. VADv2 models a distribution over a discrete planning vocabulary, while Hydra-MDP distills human and rule-based teachers into metric-specific trajectory heads [^13] [^20]. DTPP learns context-dependent costs, Gen-Drive learns a candidate evaluator, and DriveSuprim improves hard-negative discrimination through coarse-to-fine filtering and self-distillation [^10] [^11] [^33]. For trajectory refinement, ThinkTwice and ResWorld use predicted future information to update an initial plan [^12] [^34]. These lines of work separately strengthen candidate evaluation and trajectory refinement, but their coordination within a common hypothesis pool remains less explored. Retaining unrefined alternatives and directly comparing retained and refined trajectories under the same future-conditioned evaluator thus remains an open design problem.

## 3 Method

Figure 3 outlines ManeuverFlow. M-GMP allocates candidates across scene-relevant components, then M <sup>3</sup> F transports their samples into bases in one step. Metric Prism pre-scores each base using path-specific current/future BEV evidence. ECFR replaces selected bases; refinements and retained bases re-query the field for joint scoring.

### 3.1 Problem Formulation

Let $\chi\in\{\mathrm{TF},\mathrm{LTF}\}$ denote the perception configuration fixed for each model. At planning step $n$, the planner receives

$$
\mathcal{O}_{n}^{\chi}=\begin{cases}(I_{n},L_{n},s_{n},c_{n}^{\mathrm{nav}}),&\chi=\mathrm{TF},\\
(I_{n},s_{n},c_{n}^{\mathrm{nav}}),&\chi=\mathrm{LTF},\end{cases},
$$

where $I_{n}$ denotes the three camera views, $L_{n}$ is a LiDAR BEV raster, and $s_{n}$ and $c_{n}^{\mathrm{nav}}$ are the ego state and navigation command. Given $\mathcal{O}_{n}^{\chi}$, the planner predicts a future ego trajectory $\tau_{n}^{\star}$ with $H_{\tau}$ waypoints.

Each $H_{\tau}$ -waypoint trajectory hypothesis $\tau_{n,i}=\{(x_{n,i,h},y_{n,i,h},\psi_{n,i,h})\}_{h=1}^{H_{\tau}}$ is represented in the current ego frame. M <sup>3</sup> F produces $N$ bases $\mathcal{T}_{n}^{\mathrm{base}}$. Metric Prism pre-scores them; the Refiner replaces $R$ selected bases with refinements and retains the other $N-R$ in the fixed-size pool $\mathcal{T}_{n}^{\mathrm{final}}$. Metric Prism re-scores $\mathcal{T}_{n}^{\mathrm{final}}$ with trajectory-conditioned evidence and returns $\tau_{n}^{\star}$.

### 3.2 Planning Context and M-GMP

Using $L_{n}$ in TF and a learned $\mathbf{Z}_{\mathrm{lat}}$ in LTF, the encoder $\mathcal{E}_{\chi}$ returns the BEV map $B_{n}$, agent/maneuver queries $(Q_{n}^{A},q_{n}^{\mathrm{man}})$, and state–navigation context $c_{n}^{\mathrm{SN}}$:

$$
(B_{n},Q_{n}^{A},q_{n}^{\mathrm{man}},c_{n}^{\mathrm{SN}})=\mathcal{E}_{\chi}(\mathcal{O}_{n}^{\chi}).
$$

The scene-conditioned maneuver weights are

$$
\pi_{n}^{\mathrm{man}}=\operatorname{Softmax}\!\left(\mathcal{D}_{\mathrm{man}}(q_{n}^{\mathrm{man}},c_{n}^{\mathrm{SN}})\right),
$$

while all candidates share $C_{n}=(B_{n},Q_{n}^{A},c_{n}^{\mathrm{SN}})$.

Map and agent branches provide auxiliary supervision for BEV semantics and traffic-participant detection. M-GMP partitions expert trajectories into $M$ motion families using displacement, mean heading, and speed trend, then fits a fixed factorized Gaussian to each family in normalized diff-space (Appendix). We use “maneuver” and “intention” to denote these trajectory-space motion components. At inference, the scene-dependent weights $\pi_{n}^{\mathrm{man}}$ allocate candidates among components; selected component samples initialize the M <sup>3</sup> F source states $e_{n,i}$. The auxiliary objective is

$$
\mathcal{L}_{\mathrm{aux}}=\mathcal{L}_{\mathrm{map}}+\mathcal{L}_{\mathrm{agent}}+\mathcal{L}_{\mathrm{man}}.
$$

Its maneuver-classification term is

$$
\mathcal{L}_{\mathrm{man}}=\operatorname{CE}\!\left(y_{n}^{\mathrm{man}},\pi_{n}^{\mathrm{man}}\right),
$$

where $y_{n}^{\mathrm{man}}=\operatorname{Softmax}(-d_{n})$ and $d_{n}$ collects expert–component distances in normalized diff-space (Appendix).

![[fg3.png|Refer to caption]]

Figure 4: Architecture of M 3 F. Interpolated trajectory-state tokens first query BEV and ego features, then agent memories. Time-conditioned AdaLN modulates each block, while maneuver-routed experts parameterize the interval-averaged field for one-step trajectory transport.

### 3.3 M3F

M <sup>3</sup> F adopts the boundary-field and JVP construction of Improved Mean Flows (iMF) [^6]; our driving-specific contributions are motion-structured M-GMP sources, scene-conditioned component allocation, and maneuver-routed experts. The Appendix defines the trajectory encoding, maneuver assignment, and component-wise source sampling. With expert code $x_{n}^{\mathrm{gt}}$ and source state $e_{n,i}$, define $z_{n,i}^{t}=(1-t)x_{n}^{\mathrm{gt}}+te_{n,i}$ and $v_{n,i}^{c}=e_{n,i}-x_{n}^{\mathrm{gt}}$ for $0\leq r\leq t\leq 1$. Let $m_{n,i}$ index the M-GMP maneuver component generating $e_{n,i}$ and set $w_{n,i}^{\mathrm{man}}=\operatorname{OneHot}(m_{n,i})$; routing for the M <sup>3</sup> F generation objective uses the expert’s distance-matched motion component.

As shown in Fig. 4, the M-GMP tokenizer maps $z_{n,i}^{t}$ to the query $Q_{n,i}^{0}$. The interval encoding $\varphi_{\mathrm{F}}(t,t-r)$ modulates the blocks, whose state queries sequentially attend to the BEV–ego and agent memories. The maneuver code selects the corresponding feed-forward expert. The decoded field is

$$
\displaystyle u_{\theta,n,i}
$$
 
$$
\displaystyle\equiv u_{\theta}(z_{n,i}^{t},r,t\mid C_{n},w_{n,i}^{\mathrm{man}})
$$
 
$$
\displaystyle=\mathcal{D}_{u}\!\left(\mathcal{F}_{\theta}^{\mathrm{M^{3}F}}\left(Q_{n,i}^{0};\varphi_{\mathrm{F}}(t,t-r),C_{n},w_{n,i}^{\mathrm{man}}\right)\right).
$$

Fixing $(C_{n},w_{n,i}^{\mathrm{man}})$, the boundary velocity at $(z_{n,i}^{t},t,t)$ supplies the JVP tangent, while the $(z,r,t)$ -JVP is evaluated at the state and time coordinates $(z_{n,i}^{t},r,t)$:

$$
\displaystyle v_{\theta,n,i}^{b}=u_{\theta}(z_{n,i}^{t},t,t\mid C_{n},w_{n,i}^{\mathrm{man}}),
$$
$$
\displaystyle J_{\theta,n,i}=\operatorname{JVP}\!\left[u_{\theta};(z_{n,i}^{t},r,t);(v_{\theta,n,i}^{b},0,1)\right],
$$
$$
\displaystyle\widetilde{v}_{\theta,n,i}=u_{\theta,n,i}+(t-r)\operatorname{sg}[J_{\theta,n,i}],
$$

where $\operatorname{sg}[\cdot]$ denotes stop-gradient. The loss is

$$
\mathcal{L}_{\mathrm{M^{3}F}}=\mathbb{E}\!\left[\left\|\widetilde{v}_{\theta,n,i}-v_{n,i}^{c}\right\|_{1}\right].
$$

### 3.4 Future BEV Field and Metric Prism

The world model predicts once per planning step a candidate-independent $H_{B}$ -step future BEV feature field, providing prospective evidence without candidate-wise rollouts. Write each state as $\widehat{\mathbf{B}}_{n,j}=(\widehat{B}_{n,j}^{\mathrm{abs}},\widehat{B}_{n,j}^{\mathrm{res}})$, where the residual is computed from consecutive absolute features with $\widehat{B}_{n,0}^{\mathrm{abs}}=B_{n}$:

$$
\widehat{\mathcal{F}}_{n}=\mathcal{W}_{\vartheta_{\mathrm{W}}}(B_{n},c_{n}^{\mathrm{SN}})=\{\widehat{\mathbf{B}}_{n,j}\}_{j=1}^{H_{B}}.
$$

For each candidate $\tau_{n,i}$, the learned Trajectory-Guided BEV Feature Probe projects its waypoints into the current BEV map and time-aligned predicted states, samples along the path, and aggregates trajectory-specific evidence. Thus, specificity comes from trajectory-conditioned retrieval over the shared forecast, not candidate-wise future prediction. Metric Prism encodes the candidates as score tokens $\mathbf{Q}_{n}^{S}=[\mathcal{T}_{S}(\tau_{n,i})]_{i=1}^{N}$, compares them through set-level self-attention, and lets each token cross-attend to its associated evidence memory, yielding $q_{n,i}^{S}$. Factor-specific heads produce $s_{n,i}^{f}=\mathcal{D}_{f}(q_{n,i}^{S})$ and collect them into $\mathbf{s}_{n,i}$. The evidence extraction and score composition are compactly written as

$$
E_{n,i}=\mathcal{P}_{\zeta}\!\left(\tau_{n,i};B_{n},\widehat{\mathcal{F}}_{n}\right),\quad S_{n,i}=\mathcal{C}_{\mathrm{PDM}}(\mathbf{s}_{n,i}),
$$

where $\mathbf{s}_{n,i}=\{s_{n,i}^{f}\}_{f\in\mathcal{J}_{\mathrm{PDM}}}$ contains the five predicted PDM factors in $\mathcal{J}_{\mathrm{PDM}}=\{\mathrm{NC},\mathrm{DAC},\mathrm{EP},\mathrm{TTC},\mathrm{Comf}\}$. The fixed log-PDM composition $\mathcal{C}_{\mathrm{PDM}}$ ranks candidates on both NAVSIM v1 and v2; v2-only factors remain evaluation metrics and do not enter selection. The score $S_{n,i}$ ranks candidates, while $E_{n,i}$ conditions the Refiner.

During training, the shared perception backbone encodes current observations on the online path and is reused to encode future observations in a stop-gradient target pass. After alignment to the current ego frame, the detached future features define $(\bar{B}_{n,j}^{\mathrm{abs}},\bar{B}_{n,j}^{\mathrm{res}})$, with $\Omega_{n,j}$ marking the valid spatial overlap. Under LTF, this target pass processes three future camera views with the learned latent tensor in place of LiDAR. The $\Omega_{n,j}$ -masked future-BEV loss combines Smooth- $L_{1}$ and cosine alignment for absolute features with Smooth- $L_{1}$ supervision for temporal residuals. For each navtrain scene, Metric Prism learns from $N=64$ candidates and their NAVSIM v1 evaluator factor labels using weighted binary cross-entropy. Summing over $j=1,\ldots,H_{B}$, $i=1,\ldots,N$, and $f\in\mathcal{J}_{\mathrm{PDM}}$, the overall verification objective is

$$
\displaystyle\mathcal{L}_{\mathrm{verify}}
$$
 
$$
\displaystyle=\mathcal{L}_{\mathrm{FBEV}}+\mathcal{L}_{\mathrm{score}}
$$
 
$$
\displaystyle=\sum_{j}\!\left(\ell_{n,j}^{\mathrm{abs}}+\beta\ell_{n,j}^{\mathrm{res}}\right)+\sum_{i,f}\ell_{f}\!\left(s_{n,i}^{f},y_{n,i}^{f}\right),
$$

where $\beta$ weights residual supervision, $y_{n,i}^{f}$ is the evaluator label, and $\ell_{f}$ includes factor weighting, label smoothing, and logit regularization, detailed in the supplement.

### 3.5 ECFR

Given a base candidate $\tau_{n,i}$ and its evidence memory $E_{n,i}$, the Refiner predicts a residual in the path-aligned Frenet frame and updates the trajectory as

$$
\Delta_{n,i}^{\mathrm{F}}=R_{\eta}\!\left(\tau_{n,i},E_{n,i}\right),\qquad\widetilde{\tau}_{n,i}=\mathcal{U}_{\mathrm{F}}\!\left(\tau_{n,i},\Delta_{n,i}^{\mathrm{F}}\right).
$$

During training, the same $N=64$ set $\mathcal{T}_{n}^{\mathrm{base}}$ supplies both score labels and refinement targets. For each base, the highest evaluator-scoring candidate with the same maneuver index defines its reference trajectory:

$$
\displaystyle\tau_{n,i}^{\dagger}=\operatorname*{arg\,max}_{\begin{subarray}{c}\tau_{n,j}\in\mathcal{T}_{n}^{\mathrm{base}}\\
m_{n,j}=m_{n,i}\end{subarray}}P_{\mathrm{eval}}(\tau_{n,j}),
$$
$$
\displaystyle\mathcal{L}_{\mathrm{ref}}=\frac{\alpha_{\mathrm{ref}}\sum_{i}\omega_{n,i}\,\ell_{\mathrm{F}}\!\left(\Delta_{n,i}^{\mathrm{F}},\bar{\Delta}_{n,i}^{\mathrm{F}}\right)}{\max(1,\sum_{i}\omega_{n,i})},
$$

where $P_{\mathrm{eval}}$ combines evaluator factor labels, $\bar{\Delta}_{n,i}^{\mathrm{F}}=\mathcal{R}_{\mathrm{F}}(\tau_{n,i},\tau_{n,i}^{\dagger})$ is the Frenet target, and $\ell_{\mathrm{F}}$ is mean Smooth- $L_{1}$. We use $\alpha_{\mathrm{ref}}=5$ and binary validity weights $\omega_{n,i}$; invalid pairs use zero targets and weights.

### 3.6 Training Objectives and Inference

Training proceeds in two stages. We first optimize the auxiliary maneuver objective $\mathcal{L}_{\mathrm{aux}}$ and the M <sup>3</sup> F generation objective $\mathcal{L}_{\mathrm{M^{3}F}}$. Stage 2 jointly optimizes all inference-time modules. At each training step, the current model generates 64 candidates per scene for evaluator labeling and ECFR reference selection. Candidate construction and supervision targets are detached:

$$
\displaystyle\mathcal{L}_{\mathrm{joint}}
$$
 
$$
\displaystyle=\lambda_{\mathrm{aux}}\mathcal{L}_{\mathrm{aux}}+\lambda_{\mathrm{M^{3}F}}\mathcal{L}_{\mathrm{M^{3}F}}
$$
 
$$
\displaystyle+\lambda_{\mathrm{verify}}\mathcal{L}_{\mathrm{verify}}+\lambda_{\mathrm{ref}}\mathcal{L}_{\mathrm{ref}},
$$

where the $\lambda$ coefficients balance the four objectives, while $\mathcal{L}_{\mathrm{verify}}$ combines future-BEV prediction and factor-score supervision as defined in Sec. 3.4.

At inference, we retain the Top-3 maneuver components of $\pi_{n}^{\mathrm{man}}$, renormalize their mixture weights, and distribute $N$ candidates by largest-remainder allocation. For each selected component $m$, we sample $\widetilde{e}_{n,i}$ from its displacement–heading Gaussian, set $e_{n,i}=g(\widetilde{e}_{n,i})$, and assign $m_{n,i}=m$ and $w_{n,i}^{\mathrm{man}}=\operatorname{OneHot}(m)$. Each prior is transported into a trajectory through one full-interval M <sup>3</sup> F evaluation:

$$
\tau_{n,i}=\Gamma^{-1}\!\left(e_{n,i}-u_{\theta}(e_{n,i},0,1\mid C_{n},w_{n,i}^{\mathrm{man}})\right),
$$

where $\Gamma^{-1}$ applies $\mathcal{A}^{-1}$, cumulatively sums displacements from the current ego origin, and decodes heading with $\operatorname{atan2}$.

Metric Prism first pre-scores the $N$ base candidates. Guided by the preliminary ranking, selected bases are replaced with Refiner outputs that inherit their maneuver indices as provenance, while the remaining candidates are retained. For the resulting fixed-size pool $\mathcal{T}_{n}^{\mathrm{final}}$, trajectory evidence is re-queried from the shared future field before Metric Prism jointly re-scores all candidates under the deployment log-PDM composition. The highest-scoring trajectory is returned as the final plan $\tau_{n}^{\star}$.

## 4 Experiments

| Method | Venue | Backbone | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf.$\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UniAD [^9] | CVPR 2023 | ResNet-34 | 97.8 | 91.9 | 92.9 | 100.0 | 78.8 | 83.4 |
| TransFuser <sup>TF</sup> [^3] | TPAMI 2023 | ResNet-34 | 97.7 | 92.8 | 92.8 | 100.0 | 79.2 | 84.0 |
| World4Drive [^37] | ICCV 2025 | ResNet-34 | 97.4 | 94.3 | 92.8 | 100.0 | 79.9 | 85.1 |
| GoalFlow <sup>TF</sup> [^30] | CVPR 2025 | ResNet-34 | 98.3 | 93.8 | 94.3 | 100.0 | 79.8 | 85.7 |
| DiffusionDrive <sup>TF</sup> [^21] | CVPR 2025 | ResNet-34 | 98.2 | 96.2 | 94.7 | 100.0 | 82.2 | 88.1 |
| WoTE <sup>TF</sup> [^19] | ICCV 2025 | ResNet-34 | 98.5 | 96.8 | 94.9 | 99.9 | 81.9 | 88.3 |
| MeanFuser <sup>LTF</sup> [^28] | CVPR 2026 | ResNet-34 | 98.6 | 97.0 | 95.0 | 100.0 | 82.8 | 89.0 |
| VADv2 [^13] | ICLR 2026 | ResNet-34 | 98.3 | 97.4 | 95.7 | 100.0 | 82.3 | 89.3 |
| DriveSuprim [^33] | AAAI 2026 | ResNet-34 | 97.8 | 97.3 | 93.6 | 100.0 | 86.7 | 89.9 |
| Ours <sup>LTF</sup> | – | ResNet-34 | 98.49 | 97.40 | 95.13 | 99.99 | 84.87 | 89.98 |
| Ours <sup>TF</sup> | – | ResNet-34 | 98.42 | 98.01 | 95.19 | 99.99 | 85.36 | 90.58 |

Table 1: NAVSIM v1 navtest mean results under a single official benchmark version; baselines use official results. Standard deviations are reported in the supplementary material.

| Method | NC $\uparrow$ | DAC $\uparrow$ | DDC $\uparrow$ | TLC $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | LK $\uparrow$ | HC $\uparrow$ | EC $\uparrow$ | EPDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ego Status MLP [^4] | 93.1 | 77.9 | 92.7 | 99.6 | 86.0 | 91.5 | 89.4 | 98.3 | 85.4 | 64.0 |
| TransFuser <sup>TF</sup> [^3] | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 |
| DriveSuprim [^33] | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 | 83.1 |
| DiffusionDrive <sup>TF</sup> [^21] | 98.2 | 96.3 | 99.4 | 99.8 | 87.4 | 97.4 | 97.0 | 98.3 | 87.7 | 88.3 |
| MeanFuser <sup>LTF</sup> [^28] | 98.3 | 97.2 | 99.6 | 99.8 | 87.6 | 97.4 | 97.3 | 98.3 | 88.2 | 89.5 |
| Ours <sup>LTF</sup> | 98.61 | 98.13 | 99.25 | 99.80 | 87.27 | 97.96 | 96.13 | 98.26 | 85.17 | 89.70 |
| Ours <sup>TF</sup> | 98.38 | 97.94 | 99.41 | 99.76 | 88.75 | 97.75 | 96.86 | 98.32 | 86.54 | 90.14 |

Table 2: NAVSIM v2 navtest mean results under a single official benchmark version; baselines use official results. Standard deviations are reported in the supplementary material.

| Method | Traffic | SR $\uparrow$ | EP $\uparrow$ | PDMS ${}^{\ast}\uparrow$ |
| --- | --- | --- | --- | --- |
| VADv2 [^13] | Log | 77.08 | 48.00 | 63.81 |
| VADv2 [^13] | IDM | 73.66 | 46.71 | 60.98 |
| DiffusionDrive [^21] | Log | 86.24 | 45.09 | 69.83 |
| DiffusionDrive [^21] | IDM | 85.85 | 43.88 | 68.22 |
| MeanFuser [^28] | Log | 87.25 | 55.90 | 73.81 |
| MeanFuser [^28] | IDM | 87.15 | 55.60 | 73.28 |
| ManeuverFlow | Log | 87.48 | 60.68 | 75.19 |
| ManeuverFlow | IDM | 87.92 | 60.89 | 74.58 |

Table 3: Closed-loop results on all WorldEngine scenarios.

### 4.1 Experimental Setup

#### Datasets and evaluation protocols.

NAVSIM is a data-driven planning benchmark built from OpenScene/nuPlan driving logs [^15] that evaluates predicted ego trajectories against logged future scenes under a non-reactive simulator protocol [^4] [^1]. We follow the official split: navtrain for training and navtest for evaluation. We estimate M-GMP motion families and distribution statistics using only expert trajectories from navtrain. NAVSIM v1 reports the Predictive Driver Model Score (PDMS) with NC, DAC, TTC, EP, and Comfort, whereas v2 adopts the Extended Predictive Driver Model Score (EPDMS) and additionally reports DDC, TLC, LK, HC, and EC. We use the official evaluators; for v2, we reuse the v1 checkpoints, five-factor heads, and log-PDM rule—without v2-specific training or tuning.

We complement NAVSIM’s non-reactive evaluation with the latest WorldEngine benchmark on all 288 publicly released high-risk scenarios reconstructed from NAVSIM navtest logs [^17]. The logged replay (Log) setting replays background traffic, whereas the Intelligent Driver Model (IDM) setting updates surrounding agents reactively [^27]; in both settings, the ego vehicle replans from updated rendered observations. We report episode success rate (SR), closed-loop ego progress (EP), and PDMS <sup>∗</sup>: SR is the fraction of episodes without collisions or drivable-area violations, EP measures progress during closed-loop execution, and PDMS <sup>∗</sup> summarizes overall closed-loop driving quality, together assessing system-level replanning under replayed and reactive traffic.

| Model | Components | Prior | Flow | Evidence | Refiner | NC $\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Base | Direct planner | – | – | – | – | 97.73 | 79.24 | 84.01 |
| $\mathcal{M}_{0}$ | Candidate-based MeanFlow | Single | Dense MF | Cur. | – | 97.21 | 81.05 | 86.14 |
| $\mathcal{M}_{1}$ | \+ M-GMP and Dense-XL | M-GMP-8 | Dense-XL | Cur. | – | 97.02 | 81.63 | 87.91 |
| $\mathcal{M}_{2}$ | \+ maneuver-routed flow | M-GMP-8 | M <sup>3</sup> F | Cur. | – | 97.64 | 82.12 | 88.35 |
| $\mathcal{M}_{3}$ | \+ future evidence | M-GMP-8 | M <sup>3</sup> F | Cur.+Fut. | – | 98.52 | 84.41 | 89.93 |
| $\mathcal{M}_{4}^{\mathrm{Cur}}$ | w/o future branch | M-GMP-8 | M <sup>3</sup> F | Cur. | ECFR | 98.27 | 84.50 | 90.03 |
| $\mathcal{M}_{4}^{\mathrm{KM}}$ | KMeans prior control | KMeans-8 | M <sup>3</sup> F | Cur.+Fut. | ECFR | 98.42 | 84.53 | 90.04 |
| $\mathcal{M}_{4}$ | Complete model | M-GMP-8 | M <sup>3</sup> F | Cur.+Fut. | ECFR | 98.42 | 85.36 | 90.58 |

Table 4: Cumulative ablations with matched full-pipeline controls on NAVSIM v1 navtest (standard TF). Dense-XL is parameter-count matched to M <sup>3</sup> F; $\mathcal{M}_{4}^{\mathrm{Cur}}$ disables the predicted-future branch, while $\mathcal{M}_{4}^{\mathrm{KM}}$ replaces M-GMP-8 with KMeans-8. Cur./Fut. = current/predicted-future BEV evidence.

#### Implementation details.

TF and LTF use a ResNet-34 image backbone and predict $H_{\tau}=8$ waypoints over 4 s. We conduct three independent training runs on four NVIDIA RTX 4090 GPUs with global batch size 32 using AdamW (weight decay 0.1). Stage one trains the perception network and M <sup>3</sup> F at $4\times 10^{-4}$; stage two jointly optimizes all inference-time modules at $2\times 10^{-4}$, with a $0.2\times$ learning-rate multiplier for the perception network and M <sup>3</sup> F. The future field predicts $H_{B}=4$ BEV states at 1-s intervals; only the stop-gradient target branch uses future observations during training, while inference uses the current observation. The defaults $k=3$, $N=64$, and $R=N/4$ were fixed before navtest; all navtest sweeps are post-hoc sensitivity analyses using frozen checkpoints. Each checkpoint is evaluated five times, and the main tables report the mean across the three independent run averages; the corresponding sample standard deviations are provided in the supplementary material.

### 4.2 Main Results

#### Results on NAVSIM.

ManeuverFlow-TF achieves the highest listed PDMS of 90.58 in Table 1, while the camera-only LTF variant obtains 89.98. Their EP values are 85.36 and 84.87, respectively.

Without v2 adaptation, ManeuverFlow-TF achieves the highest listed EPDMS and EP at 90.14 and 88.75. ManeuverFlow-LTF obtains 89.70 EPDMS and the highest listed NC, DAC, and TTC at 98.61, 98.13, and 97.96 (Table 2). Thus, the v1-trained checkpoints remain competitive under v2’s broader metric suite without optimization for its additional v2 evaluation factors.

#### Closed-loop results in WorldEngine.

WorldEngine provides 3DGS-rendered camera observations but no LiDAR input; Table 3 therefore evaluates only ManeuverFlow-LTF. Under Log and IDM, it obtains the highest listed PDMS <sup>∗</sup> values, 75.19 and 74.58, respectively. Its EP values of 60.68 and 60.89 are also the highest listed, while SR reaches 87.48 and 87.92. These results complement NAVSIM with closed-loop evidence for ego progress and composite performance under replayed and reactive traffic.

![[qualitative_2x3.png|Refer to caption]]

Figure 5: Six urban scenes with the top-three maneuver components shown in paired BEV and front-view projections.

### 4.3 Ablation Studies

Table 4 decomposes the Hypothesize–Verify–Improve pipeline under standard TF; $\mathcal{M}_{4}^{\mathrm{Cur}}$ removes the predicted-future branch, while $\mathcal{M}_{4}^{\mathrm{KM}}$ changes only the prior.

*Hypothesize.* Candidate-based MeanFlow improves PDMS by 2.13 points. $\mathcal{M}_{0}\!\to\!\mathcal{M}_{1}$ jointly adds M-GMP-8 and the parameter-matched Dense-XL generator for another 1.77 points; because both prior support and capacity change, this is stage-level evidence. With M-GMP-8 fixed, M <sup>3</sup> F adds 0.44 PDMS over the parameter-count-matched Dense-XL control, so total parameter count alone does not explain the gain. In the complete pipeline, M-GMP-8 exceeds KMeans-8 by 0.54 PDMS and 0.83 EP, supporting maneuver-aligned prior organization. Appendix Figs. S1–S2 provide the corresponding prior visualizations.

*Verify.* Before refinement, predicted-future evidence adds 1.58 PDMS and 2.29 EP at a fixed candidate budget. The gain persists after ECFR: relative to $\mathcal{M}_{4}^{\mathrm{Cur}}$, the complete model adds 0.55 PDMS and 0.86 EP under matched protocols for candidate generation and evaluator supervision.

*Improve.* ECFR adds 0.65 PDMS and 0.95 EP to the future-aware pipeline, with NC decreasing by 0.10 points, and also improves the current-only variant. Future-aware verification and ECFR yield complementary but sub-additive gains.

#### M-GMP recall and proposal-budget sensitivity.

Table 5 reports M-GMP recall@ $k$, which measures whether the target component appears among the top- $k$ routed components. Recall reaches 98.30% at $k=3$ and changes marginally thereafter. With $k=3$, PDMS saturates beyond $N=64$ as the activation bound rises and Proposal FPS falls, supporting $N=64$ as a balanced operating point.

Taken together, these ablations support ManeuverFlow’s intended lifecycle: maneuver routing organizes a compact candidate set, shared-future evidence improves planning performance, and evidence-conditioned ECFR improves final-plan quality through maneuver-matched refinement.

<table><tbody><tr><th colspan="2">Component routing</th><td colspan="5">PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math> at proposal budget <math><semantics><mi>N</mi> <annotation>N</annotation></semantics></math></td></tr><tr><th><math><semantics><mi>k</mi> <annotation>k</annotation></semantics></math></th><th>M-GMP target recall@ <math><semantics><mi>k</mi> <annotation>k</annotation></semantics></math> (%)</th><td>8</td><td>16</td><td>32</td><td>64</td><td>128</td></tr><tr><th>1</th><th>87.20</th><td>89.85</td><td>90.04</td><td>90.26</td><td>90.33</td><td>90.42</td></tr><tr><th>2</th><th>96.46</th><td>90.04</td><td>90.15</td><td>90.43</td><td>90.54</td><td>90.59</td></tr><tr><th>3</th><th>98.30</th><td>90.06</td><td>90.28</td><td>90.41</td><td>90.58 <sup>†</sup></td><td>90.56</td></tr><tr><th>4</th><th>99.21</th><td>90.10</td><td>90.27</td><td>90.42</td><td>90.57</td><td>90.67</td></tr><tr><th colspan="2">M <sup>3</sup> F activ. bound (MB) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th><td>1.11</td><td>2.07</td><td>3.99</td><td>7.82</td><td>15.50</td></tr><tr><th colspan="2">Proposal FPS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><td>66.46</td><td>63.69</td><td>58.62</td><td>53.86</td><td>47.12</td></tr></tbody></table>

Table 5: M-GMP routing and proposal-budget sensitivity on NAVSIM v1 navtest; FPS: perception–M <sup>3</sup> F; <sup>†</sup>: default.

### 4.4 Qualitative Analysis

#### Maneuver-structured hypothesis generation.

Figure 5 shows M <sup>3</sup> F samples sharing a coherent motion tendency within each routed M-GMP component while spanning distinct, road-aligned lane and intersection alternatives across components. In these examples, the hypotheses remain visually consistent with drivable space and surrounding traffic, illustrating component-conditioned trajectory alternatives for downstream evaluation. Across the paired BEV and camera views, their separation remains legible around lane choices and intersections, making the routed maneuver structure directly inspectable. See the appendix for additional cross-method comparisons and visualizations of the maneuver-component priors before trajectory transport.

## 5 Conclusion

We present ManeuverFlow, an explicit *Hypothesize–Verify–Improve* planner that maintains maneuver-indexed hypotheses throughout generation, evaluation, and refinement. M-GMP allocates a fixed proposal budget across scene-relevant motion components, and M <sup>3</sup> F transports their samples into scene-adaptive, maneuver-routed hypotheses in one step. A shared predicted future BEV field provides path-specific evidence for Metric Prism scoring and ECFR’s local Frenet refinement. Refined and retained hypotheses re-query the shared field and are jointly re-scored for final selection. Complementary non-reactive and closed-loop evaluations on NAVSIM and WorldEngine, respectively, demonstrate the effectiveness of the integrated planner.

## References

Appendix

## 1 Hypothesize: Operational Maneuver Prior and Routed Transport

### 1.1 Diff-Space and Prior Construction

For an expert trajectory anchored at planning step $n$, $\tau_{n}=\{(\mathbf{p}_{n,h},\psi_{n,h})\}_{h=1}^{H_{\tau}}$ in the current ego frame, let $H_{\tau}=8$, $\mathbf{p}_{n,h}=(x_{n,h},y_{n,h})^{\mathsf{T}}\in\mathbb{R}^{2}$, and $\mathbf{p}_{n,0}=\mathbf{0}$. We first express each waypoint as an incremental displacement,

$$
\Delta\mathbf{p}_{n,h}=\mathbf{p}_{n,h}-\mathbf{p}_{n,h-1}.
$$

The fixed affine calibration $\mathcal{A}$ then maps this displacement to the normalized space shared with the trajectory generator:

$$
\Delta\widetilde{\mathbf{p}}_{n,h}=\mathcal{A}(\Delta\mathbf{p}_{n,h}).
$$

Appending the raw heading to the normalized displacement gives the three-dimensional pre-embedding code

$$
\widetilde{\Gamma}(\tau_{n})_{h}=[\Delta\widetilde{\mathbf{p}}_{n,h}^{\mathsf{T}},\psi_{n,h}]^{\mathsf{T}}.
$$

Finally, the cyclic heading embedding yields the four-dimensional M <sup>3</sup> F code

$$
\displaystyle\Gamma(\tau_{n})_{h}
$$
 
$$
\displaystyle=g\!\left(\widetilde{\Gamma}(\tau_{n})_{h}\right)
$$
 
$$
\displaystyle=[\Delta\widetilde{\mathbf{p}}_{n,h}^{\mathsf{T}},\sin\psi_{n,h},\cos\psi_{n,h}].
$$

The calibration of $\mathcal{A}$ remains fixed throughout. M-GMP fits its factorized Gaussian in the normalized pre-embedding space $\widetilde{\Gamma}(\tau_{n})\in\mathbb{R}^{H_{\tau}\times 3}$; $g$ yields the M <sup>3</sup> F code $\Gamma(\tau_{n})\in\mathbb{R}^{H_{\tau}\times 4}$. Thus the main-paper states are $x_{n}^{\mathrm{gt}}=\Gamma(\tau_{n}^{\mathrm{gt}})$ and $e_{n,i}=g(\widetilde{e}_{n,i})$.

For prior construction, the Maneuver Gaussian Mixture Prior (M-GMP) has $M=8$ components, is fitted to expert trajectories from navtrain and assigns each trajectory to a rule-defined motion family $m_{n}^{\mathrm{rule}}=\Psi(\boldsymbol{\xi}_{n})$ using $\boldsymbol{\xi}_{n}=(D_{n},\bar{\psi}_{n},\Delta\bar{s}_{n})$. The first-to-last displacement magnitude is

$$
D_{n}=\lVert\mathbf{p}_{n,H_{\tau}}-\mathbf{p}_{n,1}\rVert_{2}.
$$

We summarize the signed turning tendency by the horizon-mean heading in degrees,

$$
\bar{\psi}_{n}=\frac{180}{\pi H_{\tau}}\sum_{h=1}^{H_{\tau}}\psi_{n,h}.
$$

For the speed-trend proxy, the length of interval $h$ is

$$
\ell_{n,h}=\lVert\mathbf{p}_{n,h+1}-\mathbf{p}_{n,h}\rVert_{2},\qquad h=0,\ldots,7.
$$

The difference between the late and early four-step mean interval lengths is defined as

$$
\Delta\bar{s}_{n}=\tfrac{1}{4}\!\left(\sum_{h=4}^{7}\ell_{n,h}-\sum_{h=0}^{3}\ell_{n,h}\right).
$$

Here, $D_{n}$ is the first-to-last distance, $\bar{\psi}_{n}$ is mean heading in degrees, and $\ell_{n,0}$ includes the ego-to-first-waypoint interval. The symmetric $\Delta\bar{s}_{n}$ compares mean step lengths over equal early and late windows, so it is a speed-trend proxy rather than physical acceleration. We fix $\eta_{s}=0.25$ m per interval; Table S1 gives the hierarchy, with $D_{n}\geq 5$ m for every non-stop motion family in the prior.

The thresholds in Table S1 use degrees. The Gaussian heading variables $\psi$, $\mu^{\psi}$, and $\overline{\sigma}^{\psi}$, together with $\operatorname{wrap}(\cdot)$ below, use radians.

The component names form the operational trajectory-space vocabulary for “maneuver” and “intention” in the main paper. Their meaning is grounded in measurable displacement, turning, and speed-trend tendencies rather than in a separate lane-topology annotation protocol. Thus, “intention-specific” denotes conditioning on a motion-level planning representation: “Accelerating”/“Decelerating” use $\Delta\bar{s}_{n}$, while lane-change/turn names use mean heading.

For each family and horizon step, we estimate the displacement mean $\boldsymbol{\mu}^{p}_{m,h}$, its diagonal positive standard deviation $\overline{\boldsymbol{\sigma}}^{p}_{m,h}$, and the circular heading location and positive scale $(\mu^{\psi}_{m,h},\overline{\sigma}^{\psi}_{m,h})$ from its assigned expert trajectories. A sample from component $m$ has normalized displacement

$$
\Delta\widetilde{\mathbf{p}}_{i,m,h}=\boldsymbol{\mu}^{p}_{m,h}+\overline{\boldsymbol{\sigma}}^{p}_{m,h}\odot\boldsymbol{\epsilon}^{p}_{i,h}.
$$

The corresponding heading sample is

$$
\widehat{\psi}_{i,m,h}=\mu^{\psi}_{m,h}+\overline{\sigma}^{\psi}_{m,h}\epsilon^{\psi}_{i,h}.
$$

Together they form the sampled pre-embedding state

$$
\widetilde{\mathbf{e}}_{i,m,h}=[\Delta\widetilde{\mathbf{p}}_{i,m,h}^{\mathsf{T}},\widehat{\psi}_{i,m,h}]^{\mathsf{T}}.
$$

Applying the same cyclic map $g$ gives the M <sup>3</sup> F source code

$$
\displaystyle\mathbf{e}_{i,m,h}
$$
 
$$
\displaystyle=g(\widetilde{\mathbf{e}}_{i,m,h})
$$
 
$$
\displaystyle=[\Delta\widetilde{\mathbf{p}}_{i,m,h}^{\mathsf{T}},\sin\widehat{\psi}_{i,m,h},\cos\widehat{\psi}_{i,m,h}].
$$

Here, $\boldsymbol{\epsilon}^{p}_{i,h}\sim\mathcal{N}(\mathbf{0},\mathbf{I}_{2})$ and $\epsilon^{\psi}_{i,h}\sim\mathcal{N}(0,1)$; barred scales denote fixed positive component standard deviations. Rules fix component identity; their statistics are estimated from the assigned experts and then held fixed. For a scene-routed component $m=m_{n,i}$, stacking the waypoint states over $h=1,\ldots,H_{\tau}$ gives the main-paper source tensors $\widetilde{e}_{n,i}=[\widetilde{\mathbf{e}}_{i,m,h}]_{h=1}^{H_{\tau}}$ and $e_{n,i}=[\mathbf{e}_{i,m,h}]_{h=1}^{H_{\tau}}$; $g$ acts pointwise.

Figures S2 and S2 visualize, on matched axes, the two source priors compared in the complete-pipeline ablation: the KMeans-8 control and the rule-aligned M-GMP-8 used by ManeuverFlow. Both figures describe pre-transport source organization; the downstream M <sup>3</sup> F and the remaining pipeline are matched. Together, they are the two matched pre-transport prior visualizations referenced in the main paper as “Appendix Figs. S1–S2.”

#### Evidence role of the overview schematic.

Main-paper Fig. 1(a) is a mechanism-level schematic of the concentration risk for query candidates without explicit mode structure, rather than a measured collapse rate for a particular baseline. The empirical evaluation does not infer an effect from the schematic. It evaluates the structured alternative through the stage-level $\mathcal{M}_{0}\!\rightarrow\!\mathcal{M}_{1}$ change, the parameter-count-matched $\mathcal{M}_{1}\!\rightarrow\!\mathcal{M}_{2}$ generator control, the matched $\mathcal{M}_{4}^{\mathrm{KM}}\!\rightarrow\!\mathcal{M}_{4}$ prior control, and the target-component Recall@ $k$ /proposal-budget analysis in the main paper. Figures S2–S2 show the matched source-prior organization, while the main-paper qualitative figure and Fig. S3 show routed post-transport behavior. The schematic therefore motivates the failure mode; the quantitative controls and diagnostics test the proposed response to it.

![[kmeans_gmp_atlas.png|Refer to caption]]

Figure S1: KMeans-8 control prior before M 3 F transport. (a) KMeans components in mean normalized displacement space; crosses mark component centers. (b) Decoded source trajectories for components C0–C7.

| ID | Motion family | Condition | Count (%) |
| --- | --- | --- | --- |
| 0 | Stop/Crawl | $D_{n}<5$ | 12,044 (11.66) |
| 1 | Cruise | $\|\bar{\psi}_{n}\|\leq 4.5^{\circ}$, $\|\Delta\bar{s}_{n}\|\leq\eta_{s}$ | 8,695 (8.42) |
| 2 | Accelerating | $\|\bar{\psi}_{n}\|\leq 4.5^{\circ}$, $\Delta\bar{s}_{n}>\eta_{s}$ | 17,339 (16.79) |
| 3 | Decelerating | $\|\bar{\psi}_{n}\|\leq 4.5^{\circ}$, $\Delta\bar{s}_{n}<-\eta_{s}$ | 24,501 (23.72) |
| 4 | Left lane change | $4.5^{\circ}<\bar{\psi}_{n}\leq 13^{\circ}$ | 9,330 (9.03) |
| 5 | Left turn | $\bar{\psi}_{n}>13^{\circ}$ | 18,174 (17.60) |
| 6 | Right lane change | $-13^{\circ}\leq\bar{\psi}_{n}<-4.5^{\circ}$ | 5,422 (5.25) |
| 7 | Right turn | $\bar{\psi}_{n}<-13^{\circ}$ | 7,783 (7.54) |

Table S1: Rule-defined, trajectory-derived M-GMP maneuver families and their occupancy in the NAVSIM navtrain data used for prior construction.

### 1.2 Soft Component Association and Training Routes

After fitting, the component-normalized diff-space distance for each expert–component pair is

$$
\displaystyle d_{n,m}
$$
 
$$
\displaystyle=\frac{1}{H_{\tau}}\sum_{h=1}^{H_{\tau}}\left[\left\|(\Delta\widetilde{\mathbf{p}}_{n,h}-\boldsymbol{\mu}^{p}_{m,h})\oslash\overline{\boldsymbol{\sigma}}^{p}_{m,h}\right\|_{2}^{2}\right.
$$
$$
\displaystyle\left.+\frac{\operatorname{wrap}^{2}(\psi_{n,h}-\mu^{\psi}_{m,h})}{(\overline{\sigma}^{\psi}_{m,h})^{2}}\right],
$$

where $\oslash$ is element-wise division and $\operatorname{wrap}(\cdot)\in(-\pi,\pi]$. Normalizing the component similarities gives the soft maneuver supervision target

$$
y_{n,m}^{\mathrm{man}}=\frac{\exp(-d_{n,m})}{\sum_{\ell=0}^{M-1}\exp(-d_{n,\ell})}.
$$

Its highest-similarity component defines the discrete match used for routing:

$$
m_{n}^{\star}=\arg\max_{m}y_{n,m}^{\mathrm{man}}=\arg\min_{m}d_{n,m}.
$$

The rule label $m_{n}^{\mathrm{rule}}$ only constructs and names the components. The distance-based component similarity target $y_{n}^{\mathrm{man}}$ supervises $\pi_{n}^{\mathrm{man}}$; $m_{n}^{\star}$ defines the route used by the generation objective and the Recall@ $k$ target for the Maneuver-Mixture MeanFlow Field (M <sup>3</sup> F). The rule and matched labels serve distinct roles and are not assumed identical per trajectory. Stage-2 candidates are generated by the current model using its predicted $\pi_{n}^{\mathrm{man}}$ (Sec. 3); deployment allocation uses the final model’s prediction for the current scene. Accordingly, the Recall@ $k$ value in the main paper measures whether the distance-matched fitted component $m_{n}^{\star}$ is contained in the predicted top- $k$ set. It therefore quantifies scene-conditioned support coverage for the operational intention representation; it does not measure classification accuracy against separate human-labeled semantic maneuver annotations.

#### M3F interval sampling.

The distance-matched component $m_{n}^{\star}$ also determines the M <sup>3</sup> F training route. For each scene, we independently sample six source proposals with $m_{n,i}=m_{n}^{\star}$; all six share one interval pair $(r,t)$. We sample this pair from three regimes:

$$
(r,t)=\begin{cases}(0,1),&\text{with probability }0.25,\\
(q,q),&\text{with probability }0.375,\\
(q_{(1)},q_{(2)}),&\text{with probability }0.375.\end{cases}
$$

Here, $q=\operatorname{sigmoid}(\xi)$, $\xi\sim\mathcal{N}(-0.4,1)$, and $q_{1},q_{2}$ are independent draws, with $q_{(1)}=\min(q_{1},q_{2})$ and $q_{(2)}=\max(q_{1},q_{2})$. The FP32 JVP uses the boundary-field tangent and stop-gradient JVP correction defined in the main paper.

## 2 Verify: Shared-Future Evidence and Factorized Ranking

#### Notation bridge.

The main paper uses $B_{n}$ as an abstract shared BEV-map interface. Here we expose its implementation views: $F_{n}^{\mathrm{hi}}$ is the high-resolution current map, $F_{n}^{\mathrm{lo}}$ is the low-resolution grid consumed by the world model, and $B_{n}^{\mathrm{tok}}$ (defined explicitly in Eq. (S33)) is its projected token view used by the scene/agent decoder and M <sup>3</sup> F. These are encoder views derived from the same current observation, not additional sensor inputs.

### 2.1 Future-Feature Targets

Given $F_{n}^{\mathrm{lo}}$ and $c_{n}^{\mathrm{SN}}$, the world model predicts absolute and residual features at 1–4 s. At 2 Hz, the detached targets use offsets $+2,+4,+6,+8$; each is inverse-SE(2) warped from its local ego frame to the current frame using recorded poses. Bilinear grid\_sample uses zero padding, align\_corners=True, and an overlap mask for missing or out-of-view cells. The first residual is relative to $F_{n}^{\mathrm{lo}}$ and later residuals are between consecutive aligned targets. Future observations remain confined to this stop-gradient path and are unavailable at deployment.

### 2.2 Trajectory-Guided Evidence Queries

For current evidence, $F_{n}^{\mathrm{hi}}$ is pooled to the future-field grid and projected into reader space:

$$
\overline{F}_{n}^{\mathrm{cur}}=\operatorname{Conv}_{1\times 1}\!\left(\operatorname{Pool}(F_{n}^{\mathrm{hi}})\right).
$$

Let $(\mathbf{p}_{n,i,h},\psi_{n,i,h})$ be a candidate waypoint and $\boldsymbol{\delta}_{a}$ a vehicle-frame footprint anchor from the corresponding current or future query set. Its location in the current ego frame is

$$
\mathbf{q}_{n,i,h,a}=\mathbf{p}_{n,i,h}+\mathbf{R}_{\mathrm{2D}}(\psi_{n,i,h})\boldsymbol{\delta}_{a},
$$

where $\mathbf{R}_{\mathrm{2D}}(\psi)$ is the planar rotation matrix. The map $\Pi:\mathbb{R}^{2}\rightarrow[-1,1]^{2}$ converts this location to normalized BEV coordinates. Current evidence is obtained by bilinear sampling at the projected BEV coordinates,

$$
\mathbf{e}_{n,i,h,a}^{\mathrm{cur}}=\operatorname{Bil}\!\left(\overline{F}_{n}^{\mathrm{cur}},\Pi(\mathbf{q}_{n,i,h,a})\right).
$$

At waypoint $h\in\{2,4,6,8\}$, future evidence queries absolute/residual state $j=h/2$ (1–4 s); out-of-support samples are masked to zero. The reader fuses these path-aligned features with trajectory, ego-state, and route embeddings to form $E_{n,i}$. Candidates therefore share one predicted future field while retrieving path-specific evidence.

### 2.3 Factor-Wise Metric Prediction and Ranking

For a deployment-sized set of $N=64$ candidates, Metric Prism forms one 512-D score token per trajectory. A shared four-layer set-scoring stack jointly updates the $N$ tokens, and token $i$ cross-attends to its associated trajectory evidence $E_{n,i}$. Denote the resulting token by $q_{n,i}^{S}$. Five factor-specific scalar heads, shared across candidate positions, operate over the factor index set

$$
\mathcal{J}_{\mathrm{PDM}}=\{\mathrm{NC},\mathrm{DAC},\mathrm{EP},\mathrm{TTC},\mathrm{Comf}\}.
$$

For each $f\in\mathcal{J}_{\mathrm{PDM}}$, the corresponding head predicts

$$
s_{n,i}^{f}=\mathcal{D}_{f}(q_{n,i}^{S})\in(0,1).
$$

Here $\mathcal{D}_{f}$ includes its terminal sigmoid. The heads are factor-specific; the candidate representation remains shared across factor heads and set-contextualized.

During Stage 2, NAVSIM v1 evaluator targets $\mathbf{y}_{n,i}\in[0,1]^{5}$ supervise factor predictions for the $N=64$ candidates in each training set (Sec. 3):

$$
\mathcal{L}_{\mathrm{score}}=\sum_{i=1}^{N}\sum_{f\in\mathcal{J}_{\mathrm{PDM}}}\ell_{f}\!\left(s_{n,i}^{f},y_{n,i}^{f}\right).
$$

The weighted binary cross-entropy (BCE) and logit regularization included in $\ell_{f}$ are

$$
\ell_{f}(s,y)=\frac{10}{N}\left[\alpha_{f}w_{f}(y)\operatorname{BCE}(s,\widetilde{y}_{f})+\frac{10^{-5}}{5}\operatorname{logit}(s)^{2}\right],
$$

where $\operatorname{BCE}(s,t)=-t\log s-(1-t)\log(1-s)$. In the factor order of Eq. (S20), $\boldsymbol{\alpha}=(5,5,5,5,2)$. The smoothed target is $\widetilde{y}_{f}=(1-\varepsilon_{f})u_{f}(y)+\varepsilon_{f}/2$, with $u_{\mathrm{NC}}(y)=\mathbf{1}[y>0.5]$ and $u_{f}(y)=y$ for the other factors; $\varepsilon_{\mathrm{NC}}=0.02$ and $\varepsilon_{f}=0.01$ otherwise. NC sample weights are $1.7$, $3.1$, and $1.0$ for raw labels $0$, $0.5$, and $1$, respectively. The remaining weights are $w_{\mathrm{DAC}}(y)=1.8-0.8y$, $w_{\mathrm{EP}}(y)=1$, $w_{\mathrm{TTC}}(y)=1.75-0.75y$, and $w_{\mathrm{Comf}}(y)=4.5-3.5y$. The factor weights are summed, and the loss is averaged over candidates and the mini-batch. The leading factor $10$ is the score-loss scale within $\mathcal{L}_{\mathrm{verify}}$.

At inference, the five predictions define a scalar ranking score through the fixed log-PDM composition:

$$
\displaystyle S_{n,i}={}
$$
 
$$
\displaystyle\log(s_{n,i}^{\mathrm{NC}}+\epsilon)+\log(s_{n,i}^{\mathrm{DAC}}+\epsilon)
$$
 
$$
\displaystyle+\log\!\left(5s_{n,i}^{\mathrm{EP}}+5s_{n,i}^{\mathrm{TTC}}+2s_{n,i}^{\mathrm{Comf}}+\epsilon\right),
$$

with $\epsilon=10^{-6}$. The conventional $1/12$ normalization is omitted because its candidate-independent log offset preserves the ranking.

The same trajectory evidence conditions ECFR, whose supervision is defined next.

## 3 Improve: Candidate Supervision and Frenet Refinement

### 3.1 Stage-2 Candidates and Supervision

During Stage 2, all inference-time modules are jointly optimized. The current model generates a fresh $N=64$ candidate set for each training scene. Metric Prism and ECFR learn from this same set, using evidence extracted by the trainable current/future BEV branches. For each navtrain scene, let $\mathcal{K}_{n}=\operatorname{Top}_{3}(\pi_{n}^{\mathrm{man}})$ and renormalize the selected component weights within this set:

$$
\widetilde{\pi}_{n,m}=\frac{\pi_{n,m}^{\mathrm{man}}}{\sum_{q\in\mathcal{K}_{n}}\pi_{n,q}^{\mathrm{man}}},\qquad m\in\mathcal{K}_{n}.
$$

Let $\widetilde{\boldsymbol{\pi}}_{n}=(\widetilde{\pi}_{n,m})_{m\in\mathcal{K}_{n}}$. The route indices are sampled as

$$
m_{n,i}\overset{\mathrm{i.i.d.}}{\sim}\operatorname{Cat}(\widetilde{\boldsymbol{\pi}}_{n}),\qquad i=1,\ldots,N,\quad N=64.
$$

Each draw selects an M-GMP source and its M <sup>3</sup> F expert. One-step transport produces $\mathcal{T}_{n}^{\mathrm{base}}=\{\tau_{n,i}\}_{i=1}^{64}$, with each candidate retaining its generating index $m_{n,i}$. The NAVSIM v1 evaluator supplies the corresponding five-factor labels. These labels supervise Metric Prism and determine ECFR reference trajectories within the same set.

### 3.2 Evaluator Labels and Refinement References

For each candidate $\tau_{n,i}\in\mathcal{T}_{n}^{\mathrm{base}}$, the evaluator returns the normalized factor vector

$$
\mathbf{y}_{n,i}=(y^{\mathrm{NC}},y^{\mathrm{DAC}},y^{\mathrm{EP}},y^{\mathrm{TTC}},y^{\mathrm{Comf}})_{n,i}\in[0,1]^{5}.
$$

The raw factor labels define the reference-selection score

$$
P_{\mathrm{eval}}(\tau_{n,i}):=y_{n,i}^{\mathrm{NC}}y_{n,i}^{\mathrm{DAC}}\times\frac{5y_{n,i}^{\mathrm{EP}}+5y_{n,i}^{\mathrm{TTC}}+2y_{n,i}^{\mathrm{Comf}}}{12}.
$$

For each base candidate $i$, the same-maneuver maximizer in the current 64-candidate set provides its reference trajectory for Frenet refinement:

$$
\tau_{n,i}^{\dagger}=\operatorname*{arg\,max}_{\begin{subarray}{c}\tau_{n,j}\in\mathcal{T}_{n}^{\mathrm{base}}\\
m_{n,j}=m_{n,i}\end{subarray}}P_{\mathrm{eval}}(\tau_{n,j}).
$$

Reference selection considers candidates with finite trajectories and scores. The base itself is eligible, so a best-scoring base may have a zero residual target. Invalid base–reference pairs use zero residual targets and weights.

### 3.3 Base-Local Frenet Targets

Write the selected reference as $\tau_{n,i}^{\dagger}=\{(\mathbf{p}^{\dagger}_{n,i,h},\psi^{\dagger}_{n,i,h})\}_{h=1}^{H_{\tau}}$. For base waypoint $(\mathbf{p}_{n,i,h},\psi_{n,i,h})$, we compute a centered path difference at interior waypoints and a one-sided difference at each endpoint, then normalize it to obtain $\mathbf{t}_{n,i,h}$. At a degenerate tangent, we use $(\cos\psi_{n,i,h},\sin\psi_{n,i,h})$; the corresponding normal is $\mathbf{n}_{n,i,h}=(-t^{y}_{n,i,h},t^{x}_{n,i,h})$. Projecting the reference–base position difference onto the tangent gives the longitudinal target

$$
\overline{\Delta}^{s}_{n,i,h}=(\mathbf{p}^{\dagger}_{n,i,h}-\mathbf{p}_{n,i,h})^{\mathsf{T}}\mathbf{t}_{n,i,h}.
$$

The projection onto the corresponding normal gives the per-waypoint lateral displacement target

$$
\overline{\Delta}^{l}_{n,i,h}=(\mathbf{p}^{\dagger}_{n,i,h}-\mathbf{p}_{n,i,h})^{\mathsf{T}}\mathbf{n}_{n,i,h}.
$$

Finally, the wrapped reference–base heading difference defines the angular target

$$
\overline{\Delta}^{\psi}_{n,i,h}=\operatorname{wrap}(\psi^{\dagger}_{n,i,h}-\psi_{n,i,h}).
$$

The inverse map $\mathcal{U}_{\mathrm{F}}$ applies $\widetilde{\mathbf{p}}=\mathbf{p}+\Delta^{s}\mathbf{t}+\Delta^{l}\mathbf{n}$ and $\widetilde{\psi}=\operatorname{wrap}(\psi+\Delta^{\psi})$ pointwise. Stacking $(\overline{\Delta}^{s}_{n,i,h},\overline{\Delta}^{l}_{n,i,h},\overline{\Delta}^{\psi}_{n,i,h})$ over $h=1,\ldots,H_{\tau}$ gives the main-paper target $\overline{\Delta}^{\mathrm{F}}_{n,i}=\mathcal{R}_{\mathrm{F}}(\tau_{n,i},\tau_{n,i}^{\dagger})\in\mathbb{R}^{H_{\tau}\times 3}$. The residual loss $\ell_{\mathrm{F}}$ is Smooth- $L_{1}$ with unit transition, averaged over all $H_{\tau}$ waypoints and the longitudinal, lateral, and heading coordinates. The validity mask $\omega_{n,i}$ is $1$ for a finite base–reference pair and $0$ otherwise. The main-paper $\mathcal{L}_{\mathrm{ref}}$ averages over valid supervised candidates and applies $\alpha_{\mathrm{ref}}=5$; the same normalization is used across the mini-batch. Reference selection and residual targets are detached.

Table S2 consolidates the module interfaces and the two-stage training protocol before Sec. 4 expands the input, optimization, and deployment details.

<table><tbody><tr><td colspan="5">(a)  Module architecture</td></tr><tr><td>Lifecycle</td><td>Module</td><td>Structure</td><td>Config. <math><semantics><mrow><mo>(</mo><mi>d</mi><mo>,</mo><mi>h</mi><mo>,</mo><msub><mi>d</mi> <mi>ff</mi></msub><mo>)</mo></mrow> <annotation>(d,h,d_{\mathrm{ff}})</annotation></semantics></math></td><td>Output</td></tr><tr><td>Shared</td><td>Perception encoder</td><td>ResNet-34; 4-stage TransFuser; 4 GPT/stage</td><td><math><semantics><mrow><mi>h</mi> <mo>=</mo> <mn>8</mn></mrow> <annotation>h=8</annotation></semantics></math>; <math><semantics><mrow><msub><mi>d</mi> <mi>ff</mi></msub> <mo>=</mo> <mrow><mn>4</mn> <mo></mo><mi>d</mi></mrow></mrow> <annotation>d_{\mathrm{ff}}=4d</annotation></semantics></math>; dropout 0.1</td><td><math><semantics><mrow><msubsup><mi>F</mi> <mi>n</mi> <mi>hi</mi></msubsup><mo>,</mo><msubsup><mi>F</mi> <mi>n</mi> <mi>lo</mi></msubsup></mrow> <annotation>F_{n}^{\mathrm{hi}},F_{n}^{\mathrm{lo}}</annotation></semantics></math></td></tr><tr><td></td><td>Scene/agent decoder</td><td>Two-layer Transformer decoder</td><td><math><semantics><mrow><mo>(</mo><mn>256</mn><mo>,</mo><mn>8</mn><mo>,</mo><mn>1024</mn><mo>)</mo></mrow> <annotation>(256,8,1024)</annotation></semantics></math>; dropout 0</td><td>–</td></tr><tr><td>Hypothesize</td><td>M <sup>3</sup> F</td><td>6 DiT–MoE; 8 FFN experts/block</td><td><math><semantics><mrow><mo>(</mo><mn>256</mn><mo>,</mo><mn>4</mn><mo>,</mo><mn>1024</mn><mo>)</mo></mrow> <annotation>(256,4,1024)</annotation></semantics></math>; attn./FFN 0/0.1</td><td><math><semantics><msup><mi>ℝ</mi> <mrow><mn>8</mn> <mo>×</mo> <mn>4</mn></mrow></msup> <annotation>\mathbb{R}^{8\times 4}</annotation></semantics></math></td></tr><tr><td>Verify</td><td>World model</td><td>Recurrent predictor; causal refinement</td><td>–</td><td>4 abs. / 4 res.</td></tr><tr><td></td><td>Evidence reader</td><td>Trajectory-guided BEV sampling; SlotPool</td><td>Token 512; <math><semantics><mrow><mi>h</mi> <mo>=</mo> <mn>8</mn></mrow> <annotation>h=8</annotation></semantics></math>; dropout 0.1</td><td><math><semantics><msub><mi>E</mi> <mrow><mi>n</mi><mo>,</mo><mi>i</mi></mrow></msub> <annotation>E_{n,i}</annotation></semantics></math></td></tr><tr><td></td><td>Metric Prism</td><td>4-layer set scorer; 5 factor heads</td><td><math><semantics><mrow><mo>(</mo><mn>512</mn><mo>,</mo><mn>1</mn><mo>,</mo><mn>1024</mn><mo>)</mo></mrow> <annotation>(512,1,1024)</annotation></semantics></math>; proj./path 0.1/0.2</td><td>5 factors</td></tr><tr><td>Improve</td><td>ECFR</td><td>2-layer Frenet Transformer</td><td><math><semantics><mrow><mo>(</mo><mn>512</mn><mo>,</mo><mn>8</mn><mo>,</mo><mn>1024</mn><mo>)</mo></mrow> <annotation>(512,8,1024)</annotation></semantics></math>; dropout 0.1</td><td><math><semantics><msup><mi>ℝ</mi> <mrow><mn>8</mn> <mo>×</mo> <mn>3</mn></mrow></msup> <annotation>\mathbb{R}^{8\times 3}</annotation></semantics></math></td></tr></tbody></table>

<table><thead><tr><th colspan="3">(b)  Two-stage optimization</th></tr></thead><tbody><tr><td>Setting</td><td>Stage 1</td><td>Stage 2</td></tr><tr><td>Trainable modules</td><td>Perception, auxiliary heads, and M <sup>3</sup> F</td><td>All inference-time modules under one Stage-2 objective</td></tr><tr><td>Training candidate source</td><td>–</td><td>Current encoder, maneuver predictor, and M <sup>3</sup> F; 64 candidates per scene</td></tr><tr><td>Supervision</td><td><math><semantics><mrow><msub><mi>ℒ</mi> <mi>aux</mi></msub> <mo>+</mo> <msub><mi>ℒ</mi> <mrow><msup><mi>M</mi> <mn>3</mn></msup> <mo></mo><mi>F</mi></mrow></msub></mrow> <annotation>\mathcal{L}_{\mathrm{aux}}+\mathcal{L}_{\mathrm{M^{3}F}}</annotation></semantics></math> only</td><td>Detached future field (1–4 s); NAVSIM v1 factor labels and Frenet targets from the same 64-candidate set</td></tr><tr><td>Loss weights</td><td>1 / 4 / – / –</td><td>1 / 4 / 2 / 2</td></tr><tr><td>Base learning rate</td><td><math><semantics><mrow><mn>4</mn> <mo>×</mo> <msup><mn>10</mn> <mrow><mo>−</mo> <mn>4</mn></mrow></msup></mrow> <annotation>4{\times}10^{-4}</annotation></semantics></math></td><td><math><semantics><mrow><mn>2</mn> <mo>×</mo> <msup><mn>10</mn> <mrow><mo>−</mo> <mn>4</mn></mrow></msup></mrow> <annotation>2{\times}10^{-4}</annotation></semantics></math></td></tr><tr><td>Perception/M <sup>3</sup> F LR scale</td><td>1.0</td><td>0.2</td></tr><tr><td>Optimizer / weight decay</td><td>AdamW / 0.1</td><td>AdamW / 0.1</td></tr><tr><td>Training data</td><td>NAVSIM navtrain</td><td>NAVSIM navtrain</td></tr><tr><td>Iterations / epoch</td><td>2,660</td><td>2,660</td></tr><tr><td>Training epochs</td><td>60</td><td>10</td></tr><tr><td>Total optimizer updates</td><td>159,600</td><td>26,600</td></tr><tr><td>Global batch / precision</td><td>32 / FP32</td><td>32 / FP32</td></tr><tr><td>Clipping / hardware</td><td>Global norm 1.0 / 4 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> RTX 4090</td><td>Global norm 1.0 / 4 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> RTX 4090</td></tr></tbody></table>

Table S2: Reproducibility overview. (a) Module architecture; outputs omit the batch dimension (interfaces: Sec. 4.2). (b) Two-stage optimization; loss weights follow auxiliary / generation / verification / refinement. The score and residual losses include their internal scales of 10 and 5, respectively. Stage 2 initializes from Stage 1 and jointly optimizes all inference-time modules; the current model generates candidates for evaluator labeling.

## 4 Reproducibility: Interfaces, Optimization, and Deployment

Table S2(a) summarizes the lifecycle modules and their interfaces, while panel (b) summarizes the two-stage optimization. The following subsections unpack the inputs, shared representation, training protocol, and deployment lifecycle for both TF and LTF.

### 4.1 Input Construction

The LiDAR-assisted TransFuser (TF) and camera-only Latent TransFuser (LTF) use current left, front, and right views. All are cropped by 28 pixels vertically; the lateral views are also cropped by 416 pixels per horizontal side before concatenation and resizing to $256\times 1024$. TF rasterizes one LiDAR scan over $[-32,32]^{2}$ m at $0.25$ m per pixel, retains $z>0.2$ m, and clips then divides pixel counts by five. LTF replaces LiDAR with a learnable latent template $\mathbf{Z}_{\mathrm{lat}}\in\mathbb{R}^{1\times 4\times 256\times 256}$, where the leading singleton dimension indexes one shared template, the four channels are latent features, and the $256\times 256$ grid spans the BEV. At runtime, the template is broadcast along the batch dimension, yielding the actual encoder input shape $[B,4,256,256]$. Accordingly, the LTF ResNet stem uses a convolution-weight tensor of shape $[64,4,7,7]$. Both receive a 4-D command code, 2-D velocity, and 2-D acceleration.

(a)  NAVSIM v1 navtest

| Metric | LTF | TF |
| --- | --- | --- |
| NC | $98.49{\pm}0.02$ | $98.42{\pm}0.02$ |
| DAC | $97.40{\pm}0.03$ | $98.01{\pm}0.03$ |
| TTC | $95.13{\pm}0.05$ | $95.19{\pm}0.05$ |
| Comf. | $99.99{\pm}0.00$ | $99.99{\pm}0.01$ |
| EP | $84.87{\pm}0.04$ | $85.36{\pm}0.04$ |
| PDMS | $89.98{\pm}0.06$ | $90.58{\pm}0.05$ |

(b)  NAVSIM v2 navtest

| Metric | LTF | TF |
| --- | --- | --- |
| NC | $98.61{\pm}0.02$ | $98.38{\pm}0.03$ |
| DAC | $98.13{\pm}0.05$ | $97.94{\pm}0.03$ |
| DDC | $99.25{\pm}0.02$ | $99.41{\pm}0.02$ |
| TLC | $99.80{\pm}0.01$ | $99.76{\pm}0.00$ |
| EP | $87.27{\pm}0.01$ | $88.75{\pm}0.01$ |
| TTC | $97.96{\pm}0.03$ | $97.75{\pm}0.02$ |
| LK | $96.13{\pm}0.03$ | $96.86{\pm}0.06$ |
| HC | $98.26{\pm}0.02$ | $98.32{\pm}0.02$ |
| EC | $85.17{\pm}0.14$ | $86.54{\pm}0.07$ |
| EPDMS | $89.70{\pm}0.06$ | $90.14{\pm}0.04$ |

(c)  WorldEngine

| Traffic | Metric | ManeuverFlow-LTF |
| --- | --- | --- |
| Log | SR | $87.48{\pm}0.92$ |
| Log | EP | $60.68{\pm}0.91$ |
| Log | PDMS <sup>∗</sup> | $75.19{\pm}0.85$ |
| IDM | SR | $87.92{\pm}0.87$ |
| IDM | EP | $60.89{\pm}0.81$ |
| IDM | PDMS <sup>∗</sup> | $74.58{\pm}0.64$ |

Table S3: Main-result variation, reported as mean $\pm$ sample standard deviation across three independently trained checkpoints, each averaged over five evaluations. WorldEngine uses log replay (Log) or reactive IDM traffic.

| Top- $\kappa$ | Exact oracle | Within 1 pt |
| --- | --- | --- |
| 1 | 40.14 | 75.28 |
| 3 | 45.61 | 79.57 |
| 5 | 48.14 | 81.16 |
| 10 | 52.69 | 83.18 |

Table S4: Within-pool Top- $\kappa$ retrieval (%) of Metric Prism on the unrefined Base64 pool. “Within 1 pt” allows one evaluator PDMS point relative to that pool’s oracle.

| $N$ | Best | Top-3 | Top-6 | Top-12 |
| --- | --- | --- | --- | --- |
| 8 | 92.20 | 90.90 | 89.18 | – |
| 16 | 92.92 | 91.98 | 91.00 | 89.19 |
| 32 | 93.57 | 92.81 | 92.06 | 91.05 |
| 64 | 94.25 | 93.54 | 92.92 | 92.14 |
| 128 | 94.61 | 94.03 | 93.55 | 92.93 |
| 256 | 94.85 | 94.47 | 94.08 | 93.58 |
| 512 | 95.02 | 94.79 | 94.47 | 94.08 |

Table S5: Frozen-generator $\operatorname{OracleTop}_{\kappa}(N)$ diagnostic on NAVSIM v1 navtest, using nested prefixes of one fixed per-scene A512 draw on the 0–100 PDMS scale.

### 4.2 Multi-Resolution Perception Interface

The shared encoder exposes $F_{n}^{\mathrm{hi}}\in\mathbb{R}^{b\times 64\times 64\times 64}$ and a low-resolution dynamic map $F_{n}^{\mathrm{lo}}$. The former supports auxiliary BEV semantics and pooled current evidence; a $1\times 1$ projection maps the latter to

$$
B_{n}^{\mathrm{tok}}:=\operatorname{Flat}\!\left(\operatorname{Conv}_{1\times 1}(F_{n}^{\mathrm{lo}})\right)\in\mathbb{R}^{b\times 64\times 256}.
$$

Here, $b$ is the batch size. $B_{n}^{\mathrm{tok}}$ conditions the scene/agent decoder and M <sup>3</sup> F; $F_{n}^{\mathrm{lo}}$ drives future prediction, and the reader projects pooled current and predicted-future features into its query space. The main paper uses $B_{n}$ as a compact symbol for this multi-resolution BEV interface rather than for the projected token sequence $B_{n}^{\mathrm{tok}}$ alone.

### 4.3 Two-Stage Optimization

Table S2(b) summarizes the two stages. Stage 2 initializes the trainable model from Stage 1 and uses the 64-candidate supervision defined in Sec. 3. Perception and M <sup>3</sup> F continue to receive the auxiliary and generation objectives, while the shared BEV features support future prediction, scoring, and refinement. All inference-time modules are jointly optimized. Candidate construction, evaluator labels, reference selection, Frenet targets, and future-feature targets are detached from gradient computation.

### 4.4 Deployment Lifecycle

At each planning step, the final model generates 64 base candidates using Top-3 largest-remainder allocation and predicts one shared future BEV field from the current observation. Metric Prism first ranks the bases using their trajectory evidence. ECFR refines 16 selected bases and replaces their slots, retaining the other 48. All 64 final hypotheses re-query the same BEV field and are jointly re-scored by Metric Prism. The candidate with the highest predicted log-PDM score is returned as the final plan.

Table S3 reports the run-to-run variation across independently trained checkpoints analyzed in Sec. 5.

![[m3f_expert_specialization_two_scene.png|Refer to caption]]

Figure S3: Controlled expert sweep and final-plan provenance. (a,b) Paired BEV/front views of two NAVSIM scenes. Thin E0–E7 trajectories are controlled responses sharing a fixed M-GMP source state, scene context, and transport interval; the source is omitted, dashed green is the logged reference, and thick red is the final trajectory τ n ⋆ \\tau\_{n}^{\\star}. The final plan comes from the full N = 64 N=64 base/refined pool after ECFR, evidence re-query, and Metric Prism re-scoring rather than from the displayed sweep, with provenance E6 → \\rightarrow ECFR in (a) and E7 ECFR in (b). These examples expose route-conditioned response separation and final-plan provenance; dataset-level support is supplied by the quantitative component-support and matched-ablation results.

## 5 Statistical Reliability and Scoring Diagnostics

### 5.1 Main-Result Variation

Across three independently trained checkpoints, the largest observed standard deviation is 0.14 on NAVSIM and 0.92 in WorldEngine. These values describe run-to-run dispersion under the stated protocol; with three training runs, they should not be interpreted as formal confidence intervals. Table S3 provides the corresponding dispersion for every ManeuverFlow row in the three main-result tables. It does not provide matched-seed uncertainty for the component ablations in the main paper.

### 5.2 Within-Pool Top-κ\\kappa Retrieval of Metric Prism

We evaluate Metric Prism’s preliminary ordering on the same 12,146 NAVSIM v1 navtest scene tokens used by the frozen-pool analysis. For every trajectory in the unrefined Base64 pool, Metric Prism predicts the five PDM factors and applies Eq. (S24). Here Base64 is shorthand for the unrefined 64-candidate pool used by this retrieval diagnostic; all percentages are relative to its own evaluator oracle. The main paper’s $k$ denotes the number of routed components, whereas $\kappa$ here denotes ranking depth. The diagnostic asks whether the first $\kappa$ positions contain the evaluator-best trajectory in that particular pool, either exactly or within one PDMS point of its oracle score.

As Table S5 shows, at rank one, Metric Prism retrieves a trajectory within one PDMS point of the Base64 oracle in 75.28% of scenes and misses this tolerance in 24.72%; by rank ten, coverage reaches 83.18%. These are pool-relative retrieval statistics, not a budget-matched cross-method scorer comparison. This diagnostic characterizes the learned scorer on the evaluated unrefined Base64 pool; it does not isolate the future branch or compare Metric Prism with another scorer.

### 5.3 Fixed Selector Transfer to NAVSIM v2

Table S3(b) evaluates the complete v1-trained checkpoints under NAVSIM v2 while retaining the same five factor heads and log-PDM selector. The v2-only factors contribute to EPDMS evaluation but are not added to the selector. The resulting EPDMS therefore provides an empirical feasibility check for transferring the fixed selector to the extended evaluator, rather than a claim that v1 PDMS and v2 EPDMS are directly comparable across evaluator versions.

### 5.4 Frozen-Pool Headroom Diagnostic

For each scene in navtest, we construct a separate post-hoc analysis draw of 512 candidates, denoted A512, using the Top-3 categorical sampling distribution in Eq. (S26), extended to 512 draws for this diagnostic. This analysis uses frozen checkpoints after training and model selection. Across the 12,146 scenes in $\mathcal{D}=\texttt{navtest}$, the frozen M <sup>3</sup> F generator produces one fixed, ordered A512 draw per scene. Its nested prefixes $\mathcal{T}_{N}^{\mathrm{ana}}(\nu)$ use $N\in\{8,16,32,64,128,256,512\}$. Let $S_{\nu}^{\mathrm{eval}}(\tau)$ denote evaluator PDMS on the 0–100 scale, computed with the same factor aggregation as Eq. (S28). For $\kappa\leq N$, we compute

$$
\operatorname{OracleTop}_{\kappa}(N)=\frac{1}{|\mathcal{D}|}\sum_{\nu\in\mathcal{D}}\frac{1}{\kappa}\sum_{a=1}^{\kappa}S_{\nu,(a)}^{\mathrm{eval}}.
$$

Here, $S_{\nu,(a)}^{\mathrm{eval}}$ is the $a$ -th largest prefix score; $\kappa=1$ is best-of- $N$, whereas $\kappa>1$ averages the top-ranked region. Nested prefixes measure headroom within the same draw, not scaling under independent resampling or generator changes.

Table S5 quantifies candidate-quality headroom in a frozen pool. It is not an end-to-end budget sweep, an ECFR recovery measurement, or deployed performance. Although the main-paper plot overlays the 94.8 HumanAgent reference, its comparison with the A512 best-of-512 value 95.02 is an unpaired aggregate order statistic; it establishes neither per-scene nor human-level superiority.

## 6 Qualitative Evidence and Claim Boundaries

### 6.2 Descriptive, Non-Budget-Matched Candidate Gallery

Figure S4 compares candidate organization on the same six scenes using each method’s display convention. Candidate counts, ranking or filtering, and fused-output presentation differ across methods, so the gallery is descriptive rather than budget-matched; no quantitative metric is derived from it, and no random-sampling claim is made for the displayed per-method candidate sets.

### 6.3 Interpretation and Evidence Scope

#### Interpretation scope.

The paper operationalizes each intention as a trajectory-derived motion hypothesis whose displacement, turning, or speed-trend tendency directly affects planning behavior; it does not claim a separate map-semantic annotation task. The quantitative claims concern scene-conditioned component support and downstream planning gains, while the prior views and controlled expert sweeps expose source organization and routed response separation. Top-three routing may still miss rare or compositional behavior, and the qualitative examples and learned PDM factors establish neither cross-method superiority nor safety. The candidate-independent future field also does not model interactive responses to alternative ego plans.

#### Experimental scope.

The cumulative ablations evaluate the Hypothesize–Verify–Improve stages. The Dense-XL control matches total generator parameter count, and Proposal FPS measures perception through M <sup>3</sup> F. Table S5 reports post-hoc candidate-quality headroom within fixed analysis draws. Table S3 reports sample standard deviations across independently trained checkpoints under the stated training and evaluation protocol.

#### Transfer scope.

NAVSIM v2 applies the v1-trained checkpoints, five-factor heads, and selector without v2-specific training or tuning, while WorldEngine reconstructs driving scenarios from navtest logs. These protocols test fixed-selector transfer and closed-loop replanning, not independent-domain or real-vehicle generalization.

![[four_model_qualitative_comparison_appendix_450dpi_q95.jpg|Refer to caption]]

Figure S4: Descriptive, non-budget-matched gallery on six shared scenes with paired BEV/front views. (a) ManeuverFlow: a qualitative N = 32 N=32 M 3 F draw, routed 16 / 8 16/8/8 and colored by component; reported quantitative and deployment results use 64 N=64. (b) WoTE: the top 64 of 256 predictions by final reward. (c) DiffusionDrive: all 20 candidates ordered by logits. (d) MeanFuser: eight native proposal heads plus the fused output. Counts and selection are method-specific; no quantitative comparison is implied.

[^1]: Cao, W.; Hallgarten, M.; Li, T.; Dauner, D.; Gu, X.; Wang, C.; Miron, Y.; Aiello, M.; Li, H.; Gilitschenski, I.; Ivanovic, B.; Pavone, M.; Geiger, A.; and Chitta, K. 2025. Pseudo-Simulation for Autonomous Driving. In *Proceedings of the 9th Conference on Robot Learning*, volume 305 of *Proceedings of Machine Learning Research*, 4709–4722.

[^2]: Chai, Y.; Sapp, B.; Bansal, M.; and Anguelov, D. 2020. MultiPath: Multiple Probabilistic Anchor Trajectory Hypotheses for Behavior Prediction. In *Proceedings of the Conference on Robot Learning*, volume 100 of *Proceedings of Machine Learning Research*, 86–99. PMLR.

[^3]: Chitta, K.; Prakash, A.; Jaeger, B.; Yu, Z.; Renz, K.; and Geiger, A. 2023. TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 45(11): 12878–12895.

[^4]: Dauner, D.; Hallgarten, M.; Li, T.; Weng, X.; Huang, Z.; Yang, Z.; Li, H.; Gilitschenski, I.; Ivanovic, B.; Pavone, M.; Geiger, A.; and Chitta, K. 2024. NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking. In *Advances in Neural Information Processing Systems*, volume 37, 28706–28719.

[^5]: Geng, Z.; Deng, M.; Bai, X.; Kolter, J. Z.; and He, K. 2025. Mean Flows for One-Step Generative Modeling. In *Advances in Neural Information Processing Systems*, volume 38.

[^6]: Geng, Z.; Lu, Y.; Wu, Z.; Shechtman, E.; Kolter, J. Z.; and He, K. 2026. Improved Mean Flows: On the Challenges of Fastforward Generative Models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 30467–30476.

[^7]: Hu, A.; Corrado, G.; Griffiths, N.; Murez, Z.; Gurau, C.; Yeo, H.; Kendall, A.; Cipolla, R.; and Shotton, J. 2022a. Model-Based Imitation Learning for Urban Driving. In *Advances in Neural Information Processing Systems*, volume 35, 20703–20716.

[^8]: Hu, S.; Chen, L.; Wu, P.; Li, H.; Yan, J.; and Tao, D. 2022b. ST-P3: End-to-End Vision-Based Autonomous Driving via Spatial-Temporal Feature Learning. In *Computer Vision – ECCV 2022*, volume 13698 of *Lecture Notes in Computer Science*, 533–549.

[^9]: Hu, Y.; Yang, J.; Chen, L.; Li, K.; Sima, C.; Zhu, X.; Chai, S.; Du, S.; Lin, T.; Wang, W.; Lu, L.; Jia, X.; Liu, Q.; Dai, J.; Qiao, Y.; and Li, H. 2023. Planning-Oriented Autonomous Driving. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 17853–17862.

[^10]: Huang, Z.; Karkus, P.; Ivanovic, B.; Chen, Y.; Pavone, M.; and Lv, C. 2024. DTPP: Differentiable Joint Conditional Prediction and Cost Evaluation for Tree Policy Planning in Autonomous Driving. In *IEEE International Conference on Robotics and Automation*, 6806–6812.

[^11]: Huang, Z.; Weng, X.; Igl, M.; Chen, Y.; Cao, Y.; Ivanovic, B.; Pavone, M.; and Lv, C. 2025. Gen-Drive: Enhancing Diffusion Generative Driving Policies with Reward Modeling and Reinforcement Learning Fine-Tuning. In *IEEE International Conference on Robotics and Automation*, 3445–3451.

[^12]: Jia, X.; Wu, P.; Chen, L.; Xie, J.; He, C.; Yan, J.; and Li, H. 2023. Think Twice Before Driving: Towards Scalable Decoders for End-to-End Autonomous Driving. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 21983–21994.

[^13]: Jiang, B.; Chen, S.; Gao, H.; Liao, B.; Zhang, Q.; Liu, W.; and Wang, X. 2026. VADv2: End-to-End Vectorized Autonomous Driving via Probabilistic Planning. In *The Fourteenth International Conference on Learning Representations*.

[^14]: Jiang, B.; Chen, S.; Xu, Q.; Liao, B.; Chen, J.; Zhou, H.; Zhang, Q.; Liu, W.; Huang, C.; and Wang, X. 2023. VAD: Vectorized Scene Representation for Efficient Autonomous Driving. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 8340–8350.

[^15]: Karnchanachari, N.; Geromichalos, D.; Tan, K. S.; Li, N.; Eriksen, C.; Yaghoubi, S.; Mehdipour, N.; Bernasconi, G.; Fong, W. K.; Guo, Y.; and Caesar, H. 2024. Towards Learning-Based Planning: The nuPlan Benchmark for Real-World Autonomous Driving. In *IEEE International Conference on Robotics and Automation*, 629–636.

[^16]: Li, Q.; Jia, X.; Wang, S.; and Yan, J. 2024a. Think2Drive: Efficient Reinforcement Learning by Thinking with Latent World Model for Autonomous Driving (in CARLA-V2). In *Computer Vision – ECCV 2024*, volume 15103 of *Lecture Notes in Computer Science*, 142–158.

[^17]: Li, T.; Chen, L.; Wang, C.; Liu, H.; Chitta, K.; Yang, Z.; Lu, Y.; Ye, N.; Qiu, Y.; Wang, Y.; Zou, L.; Peng, J.; Pan, J.; Su, Z.; Bursuc, A.; Li, S. E.; Geiger, A.; Su, P.; and Li, H. 2026. World Engine: Towards the Era of Post-Training for Autonomous Driving. arXiv:2606.19836.

[^18]: Li, Y.; Fan, L.; He, J.; Wang, Y.; Chen, Y.; Zhang, Z.; and Tan, T. 2025a. Enhancing End-to-End Autonomous Driving with Latent World Model. In *International Conference on Learning Representations*.

[^19]: Li, Y.; Wang, Y.; Liu, Y.; He, J.; Fan, L.; and Zhang, Z. 2025b. End-to-End Driving with Online Trajectory Evaluation via BEV World Model. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 27137–27146.

[^20]: Li, Z.; Li, K.; Wang, S.; Lan, S.; Yu, Z.; Ji, Y.; Li, Z.; Zhu, Z.; Kautz, J.; Wu, Z.; Jiang, Y.-G.; and Alvarez, J. M. 2024b. Hydra-MDP: End-to-End Multimodal Planning with Multi-Target Hydra-Distillation. arXiv:2406.06978.

[^21]: Liao, B.; Chen, S.; Yin, H.; Jiang, B.; Wang, C.; Yan, S.; Zhang, X.; Li, X.; Zhang, Y.; Zhang, Q.; and Wang, X. 2025. DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 12037–12047.

[^22]: Lipman, Y.; Chen, R. T. Q.; Ben-Hamu, H.; Nickel, M.; and Le, M. 2023. Flow Matching for Generative Modeling. In *International Conference on Learning Representations*.

[^23]: Liu, X.; Gong, C.; and Liu, Q. 2023. Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. In *International Conference on Learning Representations*.

[^24]: Luo, G.; Cole, F.; Zhang, S.; Wan, Y.; Lu, Y.; and Sun, J. 2026. Flow Matching for Multimodal Distributions. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 23260–23271.

[^25]: Shi, S.; Jiang, L.; Dai, D.; and Schiele, B. 2022. Motion Transformer with Global Intention Localization and Local Movement Refinement. In *Advances in Neural Information Processing Systems*, volume 35, 6531–6543.

[^26]: Sun, W.; Lin, X.; Shi, Y.; Zhang, C.; Wu, H.; and Zheng, S. 2025. SparseDrive: End-to-End Autonomous Driving via Sparse Scene Representation. In *2025 IEEE International Conference on Robotics and Automation (ICRA)*, 8795–8801.

[^27]: Treiber, M.; Hennecke, A.; and Helbing, D. 2000. Congested Traffic States in Empirical Observations and Microscopic Simulations. *Physical Review E*, 62(2): 1805–1824.

[^28]: Wang, J.; Zheng, Y.; Liu, X.; Xing, Z.; Li, P.; Ma, K.; Ye, H.; Chen, G.; Li, G.; Chen, L.; Xia, Z.; and Zhang, Q. 2026. MeanFuser: Fast One-Step Multi-Modal Trajectory Generation and Adaptive Reconstruction via MeanFlow for End-to-End Autonomous Driving. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 17884–17893.

[^29]: Wang, Y.; He, J.; Fan, L.; Li, H.; Chen, Y.; and Zhang, Z. 2024. Driving into the Future: Multiview Visual Forecasting and Planning with World Model for Autonomous Driving. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 14749–14759.

[^30]: Xing, Z.; Zhang, X.; Hu, Y.; Jiang, B.; He, T.; Zhang, Q.; Long, X.; and Yin, W. 2025. GoalFlow: Goal-Driven Flow Matching for Multimodal Trajectories Generation in End-to-End Autonomous Driving. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 1602–1611.

[^31]: Yang, P.; Lu, B.; Xia, Z.; Han, C.; Gao, Y.; Zhang, T.; Zhan, K.; Lang, X.; Zheng, Y.; and Zhang, Q. 2026. WorldRFT: Latent World Model Planning with Reinforcement Fine-Tuning for Autonomous Driving. *Proceedings of the AAAI Conference on Artificial Intelligence*, 40(14): 11649–11657.

[^32]: Yang, Y.; Mei, J.; Ma, Y.; Du, S.; Chen, W.; Qian, Y.; Feng, Y.; and Liu, Y. 2025. Driving in the Occupancy World: Vision-Centric 4D Occupancy Forecasting and Planning via World Models for Autonomous Driving. *Proceedings of the AAAI Conference on Artificial Intelligence*, 39(9): 9327–9335.

[^33]: Yao, W.; Li, Z.; Lan, S.; Wang, Z.; Sun, X.; Alvarez, J. M.; and Wu, Z. 2026. DriveSuprim: Towards Precise Trajectory Selection for End-to-End Planning. *Proceedings of the AAAI Conference on Artificial Intelligence*, 40(14): 11910–11918.

[^34]: Zhang, J.; Fu, Z.; Xu, Z.; Dai, W.; Liu, Q.; and Wang, Y. 2026. ResWorld: Temporal Residual World Model for End-to-End Autonomous Driving. In *The Fourteenth International Conference on Learning Representations*.

[^35]: Zheng, W.; Song, R.; Guo, X.; Zhang, C.; and Chen, L. 2024. GenAD: Generative End-to-End Autonomous Driving. In *Computer Vision – ECCV 2024*, volume 15165 of *Lecture Notes in Computer Science*, 87–104.

[^36]: Zheng, Y.; Liang, R.; Zheng, K.; Zheng, J.; Mao, L.; Li, J.; Gu, W.; Ai, R.; Li, S. E.; Zhan, X.; and Liu, J. 2025a. Diffusion-Based Planning for Autonomous Driving with Flexible Guidance. In *The Thirteenth International Conference on Learning Representations*.

[^37]: Zheng, Y.; Yang, P.; Xing, Z.; Zhang, Q.; Zheng, Y.; Gao, Y.; Li, P.; Zhang, T.; Xia, Z.; Jia, P.; Lang, X.; and Zhao, D. 2025b. World4Drive: End-to-End Autonomous Driving via Intention-aware Physical Latent World Model. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 28632–28642.
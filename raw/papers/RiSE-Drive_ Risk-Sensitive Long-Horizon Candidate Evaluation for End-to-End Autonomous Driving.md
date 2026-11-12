---
title: "RiSE-Drive: Risk-Sensitive Long-Horizon Candidate Evaluation for End-to-End Autonomous Driving"
source: "https://arxiv.org/html/2611.13790v1"
author:
published:
created: 2026-11-12
description:
tags:
  - "clippings"
---
Conference: Proceedings of the 34th ACM International Conference on Multimedia; November 10–14, 2026; Rio de Janeiro, BrazilProceedings of the 34th ACM International Conference on Multimedia (MM ’26), November 10–14, 2026, Rio de Janeiro, BrazilDOI: [10.1145/3767308.3836337](https://doi.org/10.1145/3767308.3836337) ISBN: 979-8-4007-2213-4/2026/11CCS: Computing methodologies Planning and scheduling

Yu Zhao Affiliation: School of Software Technology, Zhejiang University, Ningbo, China email: [y.zhao@zju.edu.cn](mailto:y.zhao@zju.edu.cn), Hongyi Duan Affiliation: School of Software Technology, Zhejiang University, Ningbo, China email: [22551120@zju.edu.cn](mailto:22551120@zju.edu.cn), Ming Yin Affiliation: School of Software Technology, Zhejiang University, Ningbo, China email: [ming.yin@zju.edu.cn](mailto:ming.yin@zju.edu.cn), Hongyuan Zhang Affiliation: School of Software Technology, Zhejiang University, Ningbo, China email: [metastarxx@gmail.com](mailto:metastarxx@gmail.com), Fan Yang Affiliation: School of Software Technology, Zhejiang University, Hangzhou, China email: [fyoung@zju.edu.cn](mailto:fyoung@zju.edu.cn), Hongzhi Wang Affiliation: School of Software Technology, Zhejiang University, Ningbo, China email: [1030168088@qq.com](mailto:1030168088@qq.com) and Xiubo Liang Affiliation: School of Software Technology, Zhejiang University, Ningbo, China email: [xiubo@zju.edu.cn](mailto:xiubo@zju.edu.cn)

© cc

![[rise-teaser.png|RiSE-Drive evaluates each candidate under multiple future-condition branches, rolls each branch forward to obtain one long-horizon outcome sample, forms a distribution over branch outcomes, and ranks candidates using risk-sensitive statistics including the mean, lower quantiles, and CVaR.]]

Figure 1. Why deterministic mean scoring can fail. Two ego candidates may have similar mean scores, yet one can still hide a small set of catastrophic future-condition branches. RiSE-Drive therefore evaluates each candidate under multiple future-condition branches, forms a distribution over branch-level long-horizon outcomes, and ranks candidates with risk-sensitive statistics such as the mean, lower quantiles, and CVaR. RiSE-Drive evaluates each candidate under multiple future-condition branches, rolls each branch forward to obtain one long-horizon outcome sample, forms a distribution over branch outcomes, and ranks candidates using risk-sensitive statistics including the mean, lower quantiles, and CVaR.

###### Abstract.

End-to-end autonomous driving systems increasingly rely on generating multiple trajectory candidates and selecting one for execution. In this setting, a central challenge is no longer only candidate generation, but how to evaluate candidates according to their long-horizon consequences under uncertainty. Existing methods still mainly use deterministic point scores, heuristic subscores, or single-outcome evaluations, which makes it difficult to capture delayed failures, tail risk, and ranking instability under plausible future perturbations. We present RiSE-Drive, a framework for risk-sensitive long-horizon candidate evaluation in end-to-end autonomous driving. RiSE-Drive first performs candidate-conditioned latent future rollout, then estimates long-horizon outcome distributions instead of single scalar scores, and finally ranks candidates using a tail-risk-aware and perturbation-robust decision criterion. This reformulates candidate scoring from deterministic oracle-style evaluation into distributional outcome assessment with explicit risk control. As a result, the evaluator is encouraged to prefer candidates that are not only strong under nominal conditions, but also reliable under long-horizon rollout and safety-critical deviations. Experiments on the NAVSIM-v1 and NAVSIM-v2 benchmarks, as well as the closed-loop HUGSIM benchmark, show that RiSE-Drive achieves state-of-the-art performance and significantly improves the robustness of candidate selection.

###### Keywords:

End-to-End Autonomous Driving, NAVSIM, Multimodal Trajectory Planning, CVaR

## 1\. Introduction

End-to-end autonomous driving aims to map raw multi-view observations directly to a safe, rule-compliant, and route-progressive future ego trajectory in a unified learned planner [^15] [^6] [^13] [^47]. Modern systems increasingly adopt a generate-and-select paradigm, where a proposal module enumerates multiple feasible futures and a selector chooses one for execution [^25] [^20] [^18] [^11]. Under behavior-grounded evaluation such as NAVSIM and NAVSIM-v2 [^8] [^3], the central challenge is therefore not only to generate plausible candidates, but to evaluate which candidate remains best under long-horizon interactions and future uncertainty.

Among recent ResNet-34-based planners, MindDrive [^40] is a strong representative of richer future reasoning. Yet, at the surface level, even strong planners can remain brittle in safety-critical long-tail scenes and perturbed rollouts: several candidates may receive similar nominal scores, while the selected one fails after downstream interactions. The deeper reason is that candidate quality is not a deterministic scalar, but a distribution of long-horizon consequences induced by uncertain future conditions. Existing methods still largely compress this uncertainty into point scores, heuristic subscores, or single-outcome estimates, without explicit candidate-conditioned counterfactual rollout and lower-tail risk modeling. As a result, hidden tail failures remain invisible to the selector, and ranking becomes unstable exactly when robustness matters most.

We address this limitation with RiSE-Drive, a risk-sensitive long-horizon candidate evaluator for end-to-end driving. RiSE-Drive leaves the upstream perception and proposal-generation pipeline unchanged and modifies only the evaluation stage. For each candidate, it predicts a compact set of future-condition branches, performs candidate-conditioned latent consequence propagation, decodes one long-horizon outcome sample for each branch, estimates an outcome distribution through ordered quantiles, and ranks candidates using a criterion that combines the mean, lower-tail Conditional Value-at-Risk (CVaR) [^36], and cross-condition instability. This design directly targets the root cause above: branchwise rollout captures delayed interaction effects, distributional estimation exposes hidden tail failures, and CVaR-based ranking favors trajectories that remain reliable not only in expectation but also in the lower tail under plausible future deviations. Instantiated on top of DrivoR [^18], RiSE-Drive is orthogonal to backbone scaling and isolates the value of better candidate evaluation.

Our contributions are summarized as follows:

## 2\. Related Work

End-to-end autonomous driving planning. Viewed chronologically, the development of end-to-end planning is less a story of ever-heavier perception backbones than a redefinition of the planner’s output space. Early systems such as VAD, TransFuser, UniAD, and PARA-Drive established the viability of unified planners built on learned scene representations, stronger sensor fusion, and planning-oriented supervision [^15] [^6] [^13] [^47]. The next bottleneck, however, was objective mismatch: optimizing a single demonstrated trajectory encourages an average-case imitation policy, even though urban driving is fundamentally set-valued and often admits multiple near-optimal behaviors. This motivated a second wave that recast planning as sequence modeling or density estimation, including autoregressive methods [^5] [^10], probabilistic or generative planners [^53] [^4] [^30], and diffusion/flow-based formulations [^49] [^28] [^16] [^43] [^32] [^39]. In this regime, the decisive question is not merely whether a model can output a plausible future, but whether its decision space preserves sufficient behavioral support to remain reliable when multiple futures are simultaneously feasible. NAVSIM and NAVSIM-v2 make this shift explicit, since their behavior-grounded metrics reward safety, progress, compliance, and robustness under perturbation rather than mere open-loop trajectory closeness [^8] [^3] [^31] [^52].

Candidate generation, trajectory scoring, and selection. An increasingly influential design pattern formulates E2E planning as maintaining a candidate set and learning a selector over it. Early structured planners such as SparseDrive already expose this decomposition through sparse scene representations and explicit proposal reasoning [^42], while Hydra-MDP and Hydra-MDP++ make multimodal candidate generation a first-class objective via hydra-distillation [^25] [^20]. Under NAVSIM-style evaluation [^8] [^3], this decomposition becomes even more consequential: the main difficulty is often not producing nominally plausible trajectories, but deciding which candidate is safest and most decision-optimal under safety, progress, and rule-compliance trade-offs. Accordingly, TransDiffuser emphasizes the support quality and decorrelation of the candidate set [^16]; WoTE and World4Drive rank candidates through future-aware world modeling [^23] [^54]; and DriveSuprim, GTRS, ZTRS, and DrivoR make trajectory scoring itself a primary design axis [^50] [^27] [^26] [^18]. Among them, DrivoR is a particularly relevant baseline: it couples efficient transformer-based perception with disentangled proposal generation and oracle-supervised sub-score prediction, showing that lightweight proposal-and-selection pipelines can be highly competitive on NAVSIM [^18]. Importantly, DrivoR is not a plain point predictor—it produces multiple proposals and learns to rank them—but its proposal set is still generated by a one-shot deterministic decoder from fixed learned queries under winner-takes-all supervision, and the selector operates without explicit action-conditioned rollout of future scene responses, forcing long-horizon consequences to be inferred from the current compressed scene representation alone [^18]. As a result, when several candidates are locally plausible but diverge mainly through downstream interactions, selection quality can still be bottlenecked by limited counterfactual consequence modeling. Related proposal-centric formulations such as ComDrive and iPad further reinforce that modern E2E planning is increasingly governed by candidate coverage, scoring fidelity, and risk-sensitive selection rather than raw trajectory regression [^44] [^11].

## 3\. Preliminary

### 3.1. Candidate-Based End-to-End Driving with DrivoR

Candidate-based end-to-end driving methods first generate a finite set of feasible future ego trajectories and then select one trajectory for execution via a learned evaluation module. DrivoR provides a representative generate-and-score formulation and serves as the baseline planning pipeline in our framework. It consists of a compact multi-camera scene encoder, a proposal generator that outputs multiple trajectory candidates, and a scoring module that evaluates these candidates by predicted driving-quality sub-scores.

At discrete time step $t$, the observation is defined as

$$
o_{t}=\left(\left\{\mathbf{I}_{t}^{(n)}\right\}_{n=1}^{N},\mathbf{e}_{t}\right),
$$

where $\mathbf{I}_{t}^{(n)}\in\mathbb{R}^{H_{I}\times W_{I}\times 3}$ is the RGB image from the $n$ -th onboard camera, $N$ is the number of cameras, and $\mathbf{e}_{t}\in\mathbb{R}^{d_{e}}$ is the ego-state vector containing pose, velocity, acceleration, and navigation command.

The perception encoder maps the observation to a compact scene-token representation

$$
\mathbf{R}_{t}=F_{\mathrm{enc}}(o_{t})\in\mathbb{R}^{N_{r}\times D},
$$

where $N_{r}$ is the number of scene tokens and $D$ is the token dimension. The proposal generator then predicts a set of $K$ candidate ego trajectories,

$$
\mathcal{T}_{t}=\left\{\tau_{1},\tau_{2},\ldots,\tau_{K}\right\}=G_{\mathrm{prop}}(\mathbf{R}_{t},\mathbf{e}_{t}).
$$

Each candidate trajectory is a sequence of $H$ future ego states,

$$
\tau_{i}=\left\{\mathbf{p}_{i,h}\right\}_{h=1}^{H},\qquad\mathbf{p}_{i,h}=(x_{i,h},y_{i,h},\theta_{i,h})\in\mathbb{R}^{3},
$$

where $x_{i,h}$ and $y_{i,h}$ denote the longitudinal and lateral displacements in the ego-local frame, and $\theta_{i,h}$ denotes the heading angle at prediction step $h$.

Given the candidate set $\mathcal{T}_{t}$, the scoring network predicts a $C$ -dimensional sub-score vector for each candidate,

$$
\mathbf{s}_{i}=G_{\mathrm{score}}(\mathbf{R}_{t},\tau_{i})\in\mathbb{R}^{C},
$$

and aggregates it into a scalar score

$$
u_{i}=\boldsymbol{\lambda}^{\top}\mathbf{s}_{i},\qquad\boldsymbol{\lambda}\in\mathbb{R}^{C}.
$$

The final plan is selected as

$$
\tau_{t}^{\star}=\arg\max_{\tau_{i}\in\mathcal{T}_{t}}u_{i}.
$$

This formulation separates candidate generation from candidate evaluation: the proposal module produces diverse feasible futures, and the scoring module selects the candidate to execute.

### 3.2. From Deterministic Candidate Scores to Long-Horizon Outcome Variables

The deterministic scoring rule above can be written compactly as

$$
u_{i}=f_{\psi}(o_{t},\tau_{i}),
$$

where $f_{\psi}$ denotes the candidate scoring function induced by the perception encoder, the scoring module, and the final linear aggregation, and $\psi$ denotes its parameters.

The quality of a candidate is determined by the long-horizon consequences induced by executing that candidate. Let $\omega$ denote a future-condition variable and define

$$
Y_{i}=Y(\tau_{i},\omega),\qquad\omega\sim p(\omega\mid o_{t}),
$$

where $Y_{i}\in\mathbb{R}$ is the long-horizon outcome associated with candidate $\tau_{i}$, and $p(\omega\mid o_{t})$ is the conditional distribution of future conditions given the current observation. Under this formulation, the deterministic score $u_{i}$ is only a point estimate of $Y_{i}$ and does not model the distribution of future consequences.

Our objective is therefore to replace pointwise candidate evaluation with distributional candidate evaluation. Instead of ranking candidates by a single deterministic score, we rank them using statistics of the long-horizon outcome distribution, including its mean, lower-tail risk, and cross-condition stability.

## 4\. Method

### 4.1. Overview

RiSE-Drive leaves the upstream perception and proposal generation pipeline unchanged and modifies only the candidate-evaluation stage. As summarized in Fig. 2, it predicts a small set of future-condition branches for each scene, rolls each candidate under each branch to obtain branch-level long-horizon outcome samples, aggregates these samples into a candidate-level outcome distribution, and finally ranks candidates with a risk-sensitive criterion based on the mean, lower-tail CVaR, and cross-condition instability.

![[risepipeline.png|Overview of RiSE-Drive. Multi-view observations are encoded into scene features. The proposal generator creates candidate ego trajectories, while the future-condition generator creates plausible downstream branches. Each candidate and branch pair is propagated to obtain a long-horizon outcome sample. The resulting distribution is summarized with ordered quantiles and ranked using the mean, lower-tail CVaR, and cross-condition instability.]]

Figure 2. Overview of the RiSE-Drive pipeline. Multi-view observations are encoded, and the proposal generator produces K ego-trajectory candidates. A future-condition generator predicts L branch-specific condition codes that parameterize plausible downstream evolutions. For each candidate–condition pair, RiSE-Drive initializes a latent consequence state, propagates it along the trajectory, and decodes one branch-level long-horizon outcome sample. The resulting branchwise consequence traces are pooled over time and across branches and mapped to ordered outcome quantiles by a distribution head. The final ranking score combines the mean estimate, lower-tail CVaR, and cross-condition instability, thereby favoring candidates that remain reliable under plausible future deviations rather than merely optimal under a single nominal forecast. Overview of RiSE-Drive. Multi-view observations are encoded into scene features. The proposal generator creates candidate ego trajectories, while the future-condition generator creates plausible downstream branches. Each candidate and branch pair is propagated to obtain a long-horizon outcome sample. The resulting distribution is summarized with ordered quantiles and ranked using the mean, lower-tail CVaR, and cross-condition instability.

### 4.2. Future-Condition Modeling

We first summarize the scene tokens by mean pooling:

$$
\bar{\mathbf{r}}_{t}=\frac{1}{N_{r}}\sum_{j=1}^{N_{r}}\mathbf{R}_{t}^{(j)}\in\mathbb{R}^{D},
$$

where $\mathbf{R}_{t}^{(j)}\in\mathbb{R}^{D}$ is the $j$ -th scene token. Based on $\bar{\mathbf{r}}_{t}$, the future-condition generator produces $L$ branch-specific future-condition codes,

$$
\omega_{l}=P_{\mathrm{ctx}}^{(l)}(\bar{\mathbf{r}}_{t})\in\mathbb{R}^{d_{\omega}},\qquad l=1,\ldots,L.
$$

The resulting set

$$
\Omega_{t}=\left\{\omega_{1},\ldots,\omega_{L}\right\}
$$

defines the set of rollout branches used in candidate evaluation. Each $\omega_{l}$ is a learned condition code used to initialize and modulate one rollout branch.

### 4.3. Candidate-Conditioned Long-Horizon Consequence Propagation

For each candidate trajectory $\tau_{i}$, we compute a candidate-level embedding

$$
\mathbf{c}_{i}=E_{\tau}(\tau_{i})\in\mathbb{R}^{d_{c}},
$$

where $E_{\tau}$ is the *candidate trajectory level encoder*. We further define a step-wise trajectory-state embedding

$$
\mathbf{z}_{i,h}=E_{p}(\mathbf{p}_{i,h})\in\mathbb{R}^{d_{p}},\qquad h=1,\ldots,H,
$$

where $E_{p}$ is the *step level encoder*.

For each candidate–condition pair $(\tau_{i},\omega_{l})$, the initial latent consequence state is

$$
\mathbf{h}_{i,l,0}=W_{r}\bar{\mathbf{r}}_{t}+W_{c}\mathbf{c}_{i}+W_{\omega}\omega_{l}+\mathbf{b}_{0},
$$

where $W_{r}\in\mathbb{R}^{d_{h}\times D}$, $W_{c}\in\mathbb{R}^{d_{h}\times d_{c}}$, $W_{\omega}\in\mathbb{R}^{d_{h}\times d_{\omega}}$, and $\mathbf{b}_{0}\in\mathbb{R}^{d_{h}}$ are learnable parameters.

We then propagate the latent consequence state over the evaluation horizon according to

$$
\mathbf{h}_{i,l,h}=T_{\eta}(\mathbf{h}_{i,l,h-1},\mathbf{z}_{i,h},\bar{\mathbf{r}}_{t}),\qquad h=1,\ldots,H,
$$

with

$$
\tilde{\mathbf{h}}_{i,l,h}=\phi\!\left(W_{h}\mathbf{h}_{i,l,h-1}+W_{p}\mathbf{z}_{i,h}+W_{s}\bar{\mathbf{r}}_{t}+\mathbf{b}_{h}\right),
$$
 
$$
\mathbf{g}_{i,l,h}=\sigma\!\left(U_{h}\mathbf{h}_{i,l,h-1}+U_{p}\mathbf{z}_{i,h}+U_{s}\bar{\mathbf{r}}_{t}+\mathbf{b}_{g}\right),
$$
 
$$
\mathbf{h}_{i,l,h}=\mathbf{g}_{i,l,h}\odot\tilde{\mathbf{h}}_{i,l,h}+(1-\mathbf{g}_{i,l,h})\odot\mathbf{h}_{i,l,h-1},
$$

where $\phi(\cdot)$ is the GELU nonlinearity, $\sigma(\cdot)$ is the sigmoid function, and $\odot$ denotes element-wise multiplication.

At each step, we decode an instantaneous outcome

$$
r_{i,l,h}=g_{r}(\mathbf{h}_{i,l,h})\in\mathbb{R},
$$

and from the final state we decode a terminal outcome

$$
r_{i,l}^{\mathrm{term}}=g_{T}(\mathbf{h}_{i,l,H})\in\mathbb{R}.
$$

The long-horizon outcome sample under condition $\omega_{l}$ is defined as

$$
\hat{y}_{i,l}=\sum_{h=1}^{H}\gamma^{h-1}r_{i,l,h}+\gamma^{H}r_{i,l}^{\mathrm{term}}.
$$

For a fixed candidate $\tau_{i}$, the set

$$
\left\{\hat{y}_{i,l}\right\}_{l=1}^{L}
$$

provides a finite-sample representation of its long-horizon outcome distribution.

### 4.4. Distributional Outcome Estimation

We first pool each branch-specific latent consequence trajectory over time,

$$
\bar{\mathbf{h}}_{i,l}=\frac{1}{H}\sum_{h=1}^{H}\mathbf{h}_{i,l,h}\in\mathbb{R}^{d_{h}},
$$

and then pool across branches,

$$
\tilde{\mathbf{h}}_{i}=\frac{1}{L}\sum_{l=1}^{L}\bar{\mathbf{h}}_{i,l}\in\mathbb{R}^{d_{h}}.
$$

Based on $\tilde{\mathbf{h}}_{i}$, the distribution head outputs $M$ ordered quantiles,

$$
\mathbf{q}_{i}=\left[q_{i}^{(1)},q_{i}^{(2)},\ldots,q_{i}^{(M)}\right]^{\top}\in\mathbb{R}^{M},
$$

where $q_{i}^{(m)}$ corresponds to quantile level $\alpha_{m}$ and

$$
0<\alpha_{1}<\alpha_{2}<\cdots<\alpha_{M}<1.
$$

To enforce monotonicity, we use

$$
q_{i}^{(1)}=a_{i},\qquad a_{i}=h_{0}(\tilde{\mathbf{h}}_{i}),
$$
 
$$
q_{i}^{(m)}=q_{i}^{(m-1)}+\mathrm{softplus}\!\left(h_{m}(\tilde{\mathbf{h}}_{i})\right),\qquad m=2,\ldots,M,
$$

which guarantees

$$
q_{i}^{(1)}\leq q_{i}^{(2)}\leq\cdots\leq q_{i}^{(M)}.
$$

We define the mean estimate of the long-horizon outcome distribution as

$$
\hat{\mu}_{i}=\frac{1}{M}\sum_{m=1}^{M}q_{i}^{(m)}.
$$

### 4.5. Risk-Sensitive Ranking with CVaR

A ranking rule based only on $\hat{\mu}_{i}$ is insensitive to lower-tail failures. To account for tail risk, we use the lower-tail Conditional Value-at-Risk (CVaR). Let

$$
M_{\alpha}=\lceil\alpha M\rceil,
$$

where $\alpha\in(0,1)$ is the lower-tail probability level. The discrete lower-tail CVaR estimate of candidate $\tau_{i}$ is

$$
\widehat{\mathrm{CVaR}}_{\alpha}^{\downarrow}(Y_{i})=\frac{1}{M_{\alpha}}\sum_{m=1}^{M_{\alpha}}q_{i}^{(m)}.
$$

We further define the lower-tail shortfall

$$
\hat{\Delta}_{i}=\hat{\mu}_{i}-\widehat{\mathrm{CVaR}}_{\alpha}^{\downarrow}(Y_{i}).
$$

To measure cross-condition instability, we compute the branch-wise standard deviation

$$
\bar{y}_{i}=\frac{1}{L}\sum_{l=1}^{L}\hat{y}_{i,l},
$$
 
$$
\hat{\sigma}_{i}=\sqrt{\frac{1}{L}\sum_{l=1}^{L}\left(\hat{y}_{i,l}-\bar{y}_{i}\right)^{2}}.
$$

The final risk-sensitive ranking score is defined as

$$
\hat{J}_{i}=\hat{\mu}_{i}-\lambda_{\mathrm{tail}}\hat{\Delta}_{i}-\lambda_{\mathrm{var}}\hat{\sigma}_{i},
$$

where $\lambda_{\mathrm{tail}}\geq 0$ and $\lambda_{\mathrm{var}}\geq 0$ are trade-off coefficients. Candidates with larger $\hat{J}_{i}$ are preferred.

### 4.6. Training Objectives

For each candidate–condition pair $(\tau_{i},\omega_{l})$, we assume a supervision signal

$$
y_{i,l}^{\star}\in\mathbb{R},\qquad i=1,\ldots,K,\;\;l=1,\ldots,L,
$$

which represents the target long-horizon outcome under the corresponding future-condition branch.

We first supervise branch-level long-horizon outcome prediction using the Huber loss

$$
\mathcal{L}_{\mathrm{branch}}=\frac{1}{KL}\sum_{i=1}^{K}\sum_{l=1}^{L}\ell_{\mathrm{Huber}}(\hat{y}_{i,l},y_{i,l}^{\star}).
$$

We then supervise the predicted quantiles using the pinball loss

$$
\rho_{\alpha}(u)=u\left(\alpha-\mathbf{1}[u<0]\right),
$$
 
$$
\mathcal{L}_{\mathrm{quant}}=\frac{1}{KML}\sum_{i=1}^{K}\sum_{m=1}^{M}\sum_{l=1}^{L}\rho_{\alpha_{m}}\!\left(y_{i,l}^{\star}-q_{i}^{(m)}\right).
$$

We further compute the empirical target statistics

$$
\mu_{i}^{\star}=\frac{1}{L}\sum_{l=1}^{L}y_{i,l}^{\star},
$$
 
$$
\sigma_{i}^{\star}=\sqrt{\frac{1}{L}\sum_{l=1}^{L}\left(y_{i,l}^{\star}-\mu_{i}^{\star}\right)^{2}},
$$

and the empirical lower-tail CVaR

$$
\mathrm{CVaR}_{\alpha}^{\star\downarrow}(Y_{i}).
$$

The corresponding target ranking score is

$$
J_{i}^{\star}=\mu_{i}^{\star}-\lambda_{\mathrm{tail}}\left(\mu_{i}^{\star}-\mathrm{CVaR}_{\alpha}^{\star\downarrow}(Y_{i})\right)-\lambda_{\mathrm{var}}\sigma_{i}^{\star}.
$$

We optimize candidate ordering with a pairwise ranking loss

$$
\mathcal{L}_{\mathrm{rank}}=\sum_{i=1}^{K}\sum_{\begin{subarray}{c}j=1\\
j\neq i\end{subarray}}^{K}\max\left(0,\,\delta-\mathrm{sign}(J_{i}^{\star}-J_{j}^{\star})(\hat{J}_{i}-\hat{J}_{j})\right),
$$

where $\delta>0$ is a margin hyperparameter.

The final objective is

$$
\mathcal{L}=\lambda_{b}\mathcal{L}_{\mathrm{branch}}+\lambda_{q}\mathcal{L}_{\mathrm{quant}}+\lambda_{r}\mathcal{L}_{\mathrm{rank}},
$$

where $\lambda_{b}$, $\lambda_{q}$, and $\lambda_{r}$ are nonnegative loss weights.

| Method | Input | Backbone | Venue | NC $\uparrow$ | DAC $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | Conf.$\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Human | \- | \- | \- | 100 | 100 | 87.5 | 100 | 99.9 | 94.8 |
| DrivingGPT [^5] | C | VQ-VAE/GPT | ICCV 2025 | 98.9 | 90.7 | 79.7 | 94.9 | 95.6 | 82.4 |
| VADv2 [^4] | C | ResNet-34 | arXiv 2024 | 97.9 | 91.7 | 77.6 | 92.9 | 100 | 83.0 |
| UniAD [^13] | C | ResNet-34 | CVPR 2023 | 97.8 | 91.9 | 78.8 | 92.9 | 100 | 83.4 |
| LTF [^6] | C | ResNet-34 | TPAMI 2022 | 97.4 | 92.8 | 79.0 | 92.4 | 100 | 83.8 |
| TransFuser [^6] | C & L | ResNet-34 | TPAMI 2022 | 97.7 | 92.8 | 79.2 | 92.8 | 100 | 84.0 |
| PARA-Drive [^47] | C | ResNet-34 | CVPR 2024 | 97.9 | 92.4 | 79.3 | 93.0 | 99.8 | 84.0 |
| DriveX-S [^37] | C | InternImage-Tiny | ICCV 2025 | 97.5 | 94.0 | 79.7 | 93.0 | 100 | 84.5 |
| LAW [^21] | C | ResNet-34 | ICLR 2025 | 96.4 | 95.4 | 81.7 | 88.7 | 99.9 | 84.6 |
| World4Drive [^54] | C | ResNet-34 | ICCV 2025 | 97.4 | 94.3 | 79.9 | 92.8 | 100 | 85.1 |
| DRAMA [^51] | C & L | ResNet-34 | arXiv 2024 | 98.0 | 93.1 | 80.1 | 94.8 | 100 | 85.5 |
| GoalFlow [^49] | C & L | ResNet-34 | CVPR 2025 | 98.3 | 93.8 | 79.8 | 94.3 | 100 | 85.7 |
| Hydra-MDP [^25] | C & L | ResNet-34 | arXiv 2024 | 98.3 | 96.0 | 78.7 | 94.6 | 100 | 86.5 |
| Hydra-MDP++ [^20] | C & L | ResNet-34 | arXiv 2024 | 97.6 | 96.0 | 80.4 | 93.1 | 100 | 86.6 |
| ARTEMIS [^10] | C & L | ResNet-34 | arXiv 2025 | 98.3 | 95.1 | 81.4 | 94.3 | 100 | 87.0 |
| DiffusionDrive [^28] | C & L | ResNet-34 | CVPR 2025 | 98.2 | 96.2 | 82.2 | 94.7 | 100 | 88.1 |
| WoTE [^23] | C & L | ResNet-34 | ICCV 2025 | 98.5 | 96.8 | 81.9 | 94.9 | 99.9 | 88.3 |
| DIVER [^39] | C & L | ResNet-34 | arXiv 2025 | 98.5 | 96.5 | 82.6 | 94.9 | 100 | 88.3 |
| MindDrive [^40] | C & L | ResNet-34 | arXiv 2025 | 98.4 | 97.0 | 82.8 | 95.1 | 99.9 | 88.9 |
| MeanFuser [^43] | C | ResNet-34 | arXiv 2026 | 98.6 | 97.0 | 82.8 | 95.0 | 100 | 89.0 |
| AutoVLA [^56] | C | Qwen2.5-VL-3B | NeurIPS 2025 | 98.4 | 95.6 | 81.9 | 98.0 | 99.9 | 89.1 |
| DriveSuprim [^50] | C | ResNet-34 | arXiv 2025 | 97.8 | 97.3 | 86.7 | 93.6 | 100 | 89.9 |
| DriveVLA-W0 [^22] | C | Emu3-8B | arXiv 2025 | 98.7 | 99.1 | 83.3 | 95.3 | 99.3 | 90.2 |
| ReCogDrive [^24] | C | InternVL3-2B | ICLR 2026 | 97.9 | 97.3 | 87.3 | 94.9 | 100 | 90.8 |
| iPad [^11] | C | ResNet-34 | arXiv 2025 | 98.6 | 98.3 | 88.0 | 94.9 | 100 | 91.7 |
| Centaur [^38] | C | V2-99 | arXiv 2025 | 99.5 | 98.9 | 85.9 | 98.0 | 100 | 92.6 |
| DrivoR [^18] | C | ViT-S | CVPR 2026 | 98.9 | 98.3 | 89.1 | 96.2 | 100 | 93.1 |
| RiSE-Drive (Ours) | C | ViT-S | \- | 99.5 | 98.5 | 88.7 | 98.4 | 100 | 94.2 |

Table 1. NAVSIM-v1 evaluation results on the Navtest split. C denotes camera-only input and L denotes LiDAR.

<table><thead><tr><th></th><th></th><th colspan="9">Stage 1</th><th colspan="9">Stage 2</th><th></th></tr><tr><th>Method</th><th>Backbone</th><th>NC</th><th>DAC</th><th>DDC</th><th>TLC</th><th>EP</th><th>TTC</th><th>LK</th><th>HC</th><th>EC</th><th>NC</th><th>DAC</th><th>DDC</th><th>TLC</th><th>EP</th><th>TTC</th><th>LK</th><th>HC</th><th>EC</th><th>EPDMS</th></tr></thead><tbody><tr><td>LTF <sup><a href="#fn:6">6</a></sup></td><td>ResNet-34</td><td>96.2</td><td>79.5</td><td>99.1</td><td>99.5</td><td>84.1</td><td>95.1</td><td>94.2</td><td>97.5</td><td>79.1</td><td>77.7</td><td>70.2</td><td>84.2</td><td>98.0</td><td>85.1</td><td>75.6</td><td>45.4</td><td>95.7</td><td>75.9</td><td>23.1</td></tr><tr><td>DiffusionDrive <sup>∗</sup> <sup><a href="#fn:28">28</a></sup></td><td>ResNet-34</td><td>96.0</td><td>79.7</td><td>97.4</td><td>99.5</td><td>81.3</td><td>93.1</td><td>90.8</td><td>96.8</td><td>73.8</td><td>82.1</td><td>72.2</td><td>88.5</td><td>98.7</td><td>85.1</td><td>78.8</td><td>49.2</td><td>89.3</td><td>71.2</td><td>24.2</td></tr><tr><td>GuideFlow <sup><a href="#fn:32">32</a></sup></td><td>ResNet-34</td><td>96.6</td><td>80.5</td><td>96.3</td><td>99.3</td><td>82.3</td><td>94.9</td><td>91.5</td><td>97.7</td><td>67.8</td><td>87.3</td><td>76.7</td><td>88.8</td><td>99.2</td><td>84.3</td><td>85.1</td><td>49.7</td><td>93.1</td><td>44.5</td><td>27.1</td></tr><tr><td>RAP-DINO <sup><a href="#fn:9">9</a></sup></td><td>ViT-H</td><td>97.1</td><td>94.4</td><td>98.8</td><td>99.8</td><td>83.9</td><td>96.9</td><td>94.7</td><td>96.4</td><td>66.2</td><td>83.2</td><td>83.9</td><td>87.4</td><td>98.0</td><td>86.9</td><td>80.4</td><td>52.3</td><td>95.2</td><td>52.4</td><td>39.6</td></tr><tr><td>DriveSuprim <sup><a href="#fn:50">50</a></sup></td><td>V2-99</td><td>98.9</td><td>95.1</td><td>99.2</td><td>99.6</td><td>76.1</td><td>99.1</td><td>94.7</td><td>97.6</td><td>54.2</td><td>87.9</td><td>88.8</td><td>89.6</td><td>98.8</td><td>80.3</td><td>86.0</td><td>53.5</td><td>97.1</td><td>56.1</td><td>42.1</td></tr><tr><td>GTRS-Dense <sup><a href="#fn:27">27</a></sup></td><td>V2-99</td><td>98.9</td><td>96.2</td><td>99.4</td><td>99.3</td><td>72.9</td><td>98.9</td><td>95.1</td><td>96.9</td><td>39.1</td><td>91.2</td><td>89.4</td><td>94.4</td><td>98.8</td><td>69.5</td><td>90.0</td><td>54.3</td><td>94.0</td><td>48.7</td><td>45.0</td></tr><tr><td>DiffVLA <sup><a href="#fn:14">14</a></sup></td><td>Vicuna-v1.5</td><td>95.7</td><td>99.2</td><td>100</td><td>100</td><td>85.9</td><td>96.4</td><td>97.1</td><td>95.0</td><td>84.2</td><td>81.2</td><td>88.8</td><td>94.6</td><td>99.0</td><td>86.0</td><td>76.4</td><td>59.8</td><td>98.6</td><td>80.4</td><td>45.0</td></tr><tr><td>GTRS-Aug <sup><a href="#fn:27">27</a></sup></td><td>V2-99</td><td>98.9</td><td>95.1</td><td>99.1</td><td>99.6</td><td>76.2</td><td>99.1</td><td>94.9</td><td>97.6</td><td>54.2</td><td>88.1</td><td>88.8</td><td>89.3</td><td>98.9</td><td>98.9</td><td>85.9</td><td>53.7</td><td>96.8</td><td>56.9</td><td>45.4</td></tr><tr><td>ZTRS <sup><a href="#fn:26">26</a></sup></td><td>V2-99</td><td>98.9</td><td>97.6</td><td>100</td><td>100</td><td>66.7</td><td>98.9</td><td>96.2</td><td>96.7</td><td>44.0</td><td>91.1</td><td>90.4</td><td>95.8</td><td>99.0</td><td>63.6</td><td>89.8</td><td>60.4</td><td>97.6</td><td>66.1</td><td>48.1</td></tr><tr><td>DrivoR <sup><a href="#fn:18">18</a></sup></td><td>ViT-S</td><td>98.8</td><td>95.1</td><td>98.9</td><td>100</td><td>72.6</td><td>98.7</td><td>94.0</td><td>97.6</td><td>73.3</td><td>90.2</td><td>88.4</td><td>91.9</td><td>98.6</td><td>70.0</td><td>88.0</td><td>50.1</td><td>98.5</td><td>76.2</td><td>48.3</td></tr><tr><td>RiSE-Drive (Ours)</td><td>ViT-S</td><td>99.0</td><td>96.8</td><td>99.3</td><td>100</td><td>72.8</td><td>99.0</td><td>94.8</td><td>97.7</td><td>73.5</td><td>91.5</td><td>89.8</td><td>94.6</td><td>98.8</td><td>68.7</td><td>91.3</td><td>54.9</td><td>98.4</td><td>76.7</td><td>51.3</td></tr></tbody></table>

Table 2. NAVSIM-v2 navhard-two-stage results. <sup>∗</sup> denotes results reproduced with the official code repository or official checkpoint.

<table><thead><tr><th>Method  </th><th colspan="5">RC</th><th colspan="5">HD-Score</th></tr><tr><th> </th><th>E</th><th>M</th><th>H</th><th>X</th><th>Avg.  </th><th>E</th><th>M</th><th>H</th><th>X</th><th>Avg.</th></tr></thead><tbody><tr><td>VAD <sup><a href="#fn:15">15</a></sup>  </td><td>51.3</td><td>31.1</td><td>25.3</td><td>26.5</td><td>31.4  </td><td>36.3</td><td>9.5</td><td>8.0</td><td>11.5</td><td>13.4</td></tr><tr><td>LTF <sup><a href="#fn:6">6</a></sup>  </td><td>67.8</td><td>35.1</td><td>26.2</td><td>40.5</td><td>38.9  </td><td>58.9</td><td>18.0</td><td>9.8</td><td>25.9</td><td>23.7</td></tr><tr><td>UniAD <sup><a href="#fn:13">13</a></sup>  </td><td>78.4</td><td>60.5</td><td>33.6</td><td>17.8</td><td>45.9  </td><td>64.9</td><td>45.8</td><td>20.6</td><td>6.6</td><td>32.7</td></tr><tr><td>DrivoR <sup><a href="#fn:18">18</a></sup>  </td><td>80.9</td><td>50.5</td><td>33.8</td><td>47.1</td><td>49.8  </td><td>73.3</td><td>34.6</td><td>18.8</td><td>32.5</td><td>35.7</td></tr><tr><td>Ours  </td><td>76.5</td><td>53.3</td><td>34.1</td><td>46.3</td><td>50.1  </td><td>74.6</td><td>37.2</td><td>22.7</td><td>39.2</td><td>39.5</td></tr></tbody></table>

E: Easy, M: Medium, H: Hard, X: Extreme.

Table 3. Photorealistic closed-loop evaluation on HUGSIM.

![[syb.png|Four columns show unprotected-left-turn, exit-ramp, yielding, and roundabout scenarios, with two temporal snapshots in each column. Magenta curves are RiSE-Drive plans, green curves are human future trajectories, orange dashed curves are ego histories, and blue curves are future trajectories of surrounding vehicles.]]

Figure 3. Qualitative results of RiSE-Drive on representative NavSafe safety-critical scenarios. From left to right, columns (a)–(d) correspond to UNPL ( unprotected left turn ), EXR exit ramp YLD yielding ), and RDBT roundabout ). Each column shows two temporal snapshots of one representative case. Magenta denotes the RiSE-Drive plan, green the human future trajectory, orange dashed the ego history, and blue the future trajectories of surrounding vehicles. Across all four cases, RiSE-Drive produces smooth, route-consistent, and interaction-aware behaviors while remaining close to safe human driving in conflict-prone regions. Four columns show unprotected-left-turn, exit-ramp, yielding, and roundabout scenarios, with two temporal snapshots in each column. Magenta curves are RiSE-Drive plans, green curves are human future trajectories, orange dashed curves are ego histories, and blue curves are future trajectories of surrounding vehicles.

<table><tbody><tr><th></th><td colspan="2">NAVSIM-v1</td><td colspan="2">NAVSIM-v2 NavHard</td></tr><tr><th>Configuration</th><td>PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EPDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><th>B-I: DrivoR</th><td>93.1</td><td>98.9</td><td>48.3</td><td>88.0</td></tr><tr><th>B-II: DrivoR w/o SD</th><td>91.9</td><td>98.4</td><td>45.6</td><td>84.6</td></tr><tr><th>+ FCG</th><td>92.6</td><td>98.7</td><td>46.8</td><td>86.5</td></tr><tr><th>+ CTE</th><td>92.9</td><td>98.8</td><td>47.5</td><td>87.3</td></tr><tr><th>+ SLE + LCP</th><td>93.5</td><td>99.1</td><td>49.2</td><td>89.6</td></tr><tr><th>+ DOE</th><td>93.9</td><td>99.3</td><td>50.6</td><td>90.5</td></tr><tr><th>+ RSR</th><td>94.2</td><td>99.5</td><td>51.3</td><td>91.3</td></tr></tbody></table>

Table 4. Component-wise ablation of the proposed evaluator. Starting from the reduced baseline B-II, each row adds one RiSE-Drive component to the configuration immediately above.

| Configuration | Optim Params (M) | Total Params (M) | PDMS $\uparrow$ | EPDMS $\uparrow$ | HD-Score Avg.$\uparrow$ |
| --- | --- | --- | --- | --- | --- |
| DrivoR | 18.8 | 41.8 | 93.1 | 48.3 | 35.7 |
| B-II | 17.2 | 39.9 | 91.9 | 45.6 | 33.8 |
| RiSE-Drive | 18.9 | 42.3 | 94.2 | 51.3 | 39.5 |

Table 5. Parameter-efficiency of RiSE-Drive relative to DrivoR and the reduced baseline.

## 5\. Experiment

We build on the DrivoR camera-only pipeline and keep its perception backbone and proposal generator unchanged. Unless otherwise stated, we use four surround-view RGB cameras, a DINOv2 ViT-S backbone [^35] with 16 camera-aware register tokens per camera, 4-layer transformer decoders with hidden dimension 256, and LoRA [^12] of rank 32 following [^1]. For RiSE-Drive, we set $K=64$, $L=4$, $M=10$, $d_{\omega}=d_{c}=d_{p}=d_{h}=256$, $\alpha=0.2$, $\gamma=0.99$, $\lambda_{\mathrm{tail}}=0.5$, $\lambda_{\mathrm{var}}=0.1$, $\delta=0.1$, and $\lambda_{b}=\lambda_{q}=\lambda_{r}=1.0$. The model is trained with AdamW at learning rate $2\times 10^{-4}$ using cosine decay on 4 RTX 4090 GPUs.

### 5.1. Benchmarks

NAVSIM-v1. We conduct our main evaluation on NAVSIM-v1 [^8], a large-scale end-to-end planning benchmark instantiated on OpenScene [^34], which redistributes annotated driving data from nuPlan [^17]. Unlike pure trajectory-matching evaluation, NAVSIM-v1 executes the predicted trajectory in a short non-reactive rollout and scores the resulting behavior, directly assessing safety, drivability, smoothness, and route progress. We report the Predictive Driver Model Score (PDMS) together with its main sub-metrics: No at-fault Collision (NC), Drivable Area Compliance (DAC), Time-to-Collision (TTC), Comfort (Comf.), and Ego Progress (EP); EP is normalized by a safe upper bound estimated by PDM-Closed [^7].

NavSafe Split. Beyond the standard navtest split, we also evaluate on navsafe, a diagnostic benchmark proposed by Centaur [^38]. It is built by mining difficult cases from navtest, aligning them with NHTSA pre-crash categories [^33], and manually verifying the selected clips, thereby emphasizing rare safety-critical scenarios such as roundabouts, unprotected left turns, ramps, yellow-light decisions, lane-less areas, rare traffic signs, adverse weather, overtaking, and yielding. Compared with aggregate navtest scores, navsafe offers a more sensitive probe of long-tail robustness.

NAVSIM-v2. We further evaluate on NAVSIM-v2 [^3], which extends NAVSIM-v1 from short-horizon non-reactive scoring to a pseudo-simulation setting. Starting from real logged observations, it synthesizes future sensor views around plausible ego states via neural rendering, exposing planners to perturbed observations that mimic off-distribution rollouts. Its main metric, Extended Predictive Driver Model Score (EPDMS), augments NC, DAC, TTC, and EP with Driving Direction Compliance (DDC), Traffic Light Compliance (TLC), Lane Keeping (LK), History Comfort (HC), and Extended Comfort (EC). On the real-only navtest setting, following prior work, we report the Stage-I score $\mathrm{EPDMS}_{1}$.

NavHard-Two-Stage Split. We also report results on navhard-two-stage, the official hard split of NAVSIM-v2 [^3]. This split targets interaction-heavy and recovery-demanding cases with a two-stage protocol: in Stage I, the planner is evaluated on the real observation and produces a 4-second endpoint; in Stage II, it is re-evaluated on synthetic branches sampled around plausible Stage-I futures, and the resulting outcomes are aggregated by proximity to the Stage-I endpoint to form the final EPDMS. It is therefore more discriminative than real-only evaluation for measuring robustness under state perturbations.

HUGSIM. To complement the NAVSIM family, we additionally evaluate on HUGSIM [^55], a photo-realistic closed-loop benchmark built from reconstructed real-world driving scenes. HUGSIM lifts multi-view RGB observations into interactive 3D environments via Gaussian-splatting-based scene reconstruction and supports iterative policy execution, so the planner’s actions affect subsequent observations. The benchmark contains 70+ reconstructed scenes and 400+ scenarios derived from KITTI-360 [^29], Waymo [^41], nuScenes [^2], and PandaSet [^48], and is divided into easy, medium, hard, and extreme difficulty levels. We report both Road Completion (RC) and the HUGSIM Driving Score (HD-Score), where HD-Score summarizes no-collision, drivable-area compliance, TTC, comfort, and global route completion. Compared with NAVSIM-style evaluation, HUGSIM more directly measures long-horizon stability, recovery ability, and robustness to compounding closed-loop errors.

### 5.2. Comparison with State-of-the-Art

Tables 1–3 compare RiSE-Drive with recent end-to-end planners on NAVSIM-v1 [^8], NAVSIM-v2 navhard split [^3], and HUGSIM [^55]. The compared methods span lightweight CNN backbones, stronger vision encoders such as InternImage-Tiny [^45], V2-99 [^19], and DINOv2 ViT-S [^35], as well as larger VLM/VLA backbones such as Emu3-8B [^46] and InternVL3-2B [^57]. RiSE-Drive uses the same backbone and proposal generator as DrivoR [^18]; the comparison against DrivoR therefore isolates the effect of long-horizon, distributional, and risk-sensitive candidate evaluation.

#### NAVSIM-v1.

Table 1 reports results on the navtest split. RiSE-Drive achieves the best PDMS of 94.2, improving over DrivoR by 1.1 points. Under the matched ViT-S setup, it raises NC from 98.9 to 99.5 and TTC from 96.2 to 98.4 while maintaining perfect comfort; EP decreases slightly from 89.1 to 88.7, indicating a safer and slightly more conservative selection policy.

#### HUGSIM.

Table 3 evaluates fully closed-loop transfer on HUGSIM. RiSE-Drive achieves the best average RC and HD-Score of 50.1 and 39.5, exceeding DrivoR by 0.3 and 3.8 points, respectively. The gain is largest on the hard and extreme subsets, where HD-Score rises from 18.8 to 22.7 and from 32.5 to 39.2, indicating stronger long-horizon robustness under compounding errors.

Overall, the three benchmarks consistently show that, once a strong proposal set is available, improving candidate evaluation yields larger gains than backbone scaling alone.

### 5.3. Ablation Study

Unless otherwise stated, all ablations keep the DrivoR encoder and proposal generator fixed, so the results isolate the contribution of the proposed evaluator. We study three aspects: component-wise contribution, parameter overhead, and sensitivity to the future-branch and quantile design. We denote the original DrivoR by B-I, and a reduced baseline that replaces its scoring decoder with a lightweight direct score head by B-II. Starting from B-II, we progressively add FCG (Future-Condition Generator), CTE (candidate trajectory-level encoder), SLE (step-level encoder), LCP (long-horizon consequence propagation), DOE (distributional outcome estimation), and RSR (risk-sensitive ranking).

#### Component-wise ablation.

Table 4 shows steady gains as the evaluator is built up from the reduced baseline. Future conditioning and trajectory encoding recover much of the gap introduced by simplifying the original scorer, while explicit long-horizon consequence propagation yields the clearest benefit on the harder NAVSIM-v2 setting. The final improvements from DOE and RSR show that distributional prediction and risk-sensitive ranking are complementary rather than interchangeable.

#### Parameter overhead.

Table 5 shows that RiSE-Drive improves all three aggregate benchmarks with only a marginal increase in parameter count over DrivoR. The gain is therefore structural, not a consequence of brute-force model scaling.

### 5.4. Visualization

We further visualize RiSE-Drive on four representative safety-critical cases from NavSafe [^38]: UNPL (*unprotected left turn*), EXR (*exit ramp*), YLD (*yielding*), and RDBT (*roundabout*). As shown in Fig. 3, the four columns from left to right correspond to (A) UNPL, (B) EXR, (C) YLD, and (D) RDBT. Across these interaction-heavy long-tail scenarios, RiSE-Drive remains smooth, route-consistent, and conservative in conflict regions. In UNPL and RDBT, it commits only after sufficient clearance to surrounding traffic; in EXR, it follows the ramp geometry stably; in YLD, it resolves right-of-way cautiously before resuming progress. In all four cases, the planned trajectory stays close to the human future trajectory while preserving adequate safety margins, qualitatively supporting the gains on safety-critical scenarios.

## 6\. Conclusion

We presented RiSE-Drive, a risk-sensitive evaluator for candidate-based end-to-end autonomous driving. The central premise of this work is that, once a planner already provides a strong set of feasible trajectory proposals, the remaining bottleneck is no longer only candidate coverage, but how those candidates are judged under long-horizon uncertainty. Existing selectors largely compress candidate quality into deterministic point scores or heuristic sub-scores, which can obscure delayed failures and hidden lower-tail risk. RiSE-Drive addresses this limitation by modeling candidate quality as a distribution over future-condition-dependent long-horizon outcomes. Concretely, it predicts a compact set of future-condition branches, performs candidate-conditioned latent consequence propagation, estimates ordered outcome quantiles, and ranks trajectories with a risk-sensitive criterion that combines the mean, lower-tail CVaR, and cross-condition instability.

This formulation makes candidate selection explicitly sensitive to tail failures rather than only nominal performance. As a result, RiSE-Drive prefers trajectories that remain reliable not only on average, but also under plausible downstream deviations and interaction-heavy rollouts. Across NAVSIM-v1, NAVSIM-v2, and HUGSIM, as well as qualitative analysis on safety-critical NavSafe scenarios, RiSE-Drive consistently improves robustness under a matched DrivoR setup, with especially clear gains on hard, perturbation-heavy, and closed-loop evaluations. These results suggest that robust end-to-end driving depends not only on stronger perception backbones or larger models, but also on principled long-horizon candidate evaluation. We hope this perspective encourages future work on tighter integration between proposal generation, consequence modeling, and risk-sensitive decision making.

## References

## Appendix A Supplementary Material

### A.1. HUGSIM Results

Table 3 corrects the HUGSIM Easy RC for RiSE-Drive, which was misprinted as 79.5 in the conference version.

### A.2. Future-Condition Branch Construction

RiSE-Drive uses $L=4$ reference conditions to construct branch targets (Section 4.6). We pair two motion measurements from the same training-log window. Longitudinal pressure is the negative of the lead vehicle’s longitudinal acceleration. Lateral approach is the lateral velocity of a same-direction adjacent vehicle, signed positive toward the ego corridor.

We first bisect the windows by the empirical rank of longitudinal pressure, then bisect each half by the within-group rank of lateral approach. The four conditions in Table A.1 use low and high levels relative to the training logs, with a separate lateral threshold in each longitudinal group.

For each condition, we choose the observed pair closest to the group’s componentwise median, measuring distance after scaling both quantities by their training-set interquartile ranges.

Table A.1. Four joint traffic conditions.

| Branch $l$ | Condition | Longitudinal pressure | Lateral approach |
| --- | --- | --- | --- |
| 1 | C0 Open passage | Low | Low |
| 2 | C1 Longitudinal constraint | High | Low |
| 3 | C2 Lateral approach | Low | High |
| 4 | C3 Combined constraint | High | High |

Each representative sets the lead vehicle’s target longitudinal acceleration and the adjacent vehicle’s target lateral velocity. Starting from a shared initial state, the benchmark applies these targets smoothly under its road and dynamics constraints. An axis is active only when the relevant vehicle and a feasible motion are available.

Every candidate $\tau_{i}$ is evaluated under the same four conditions. Benchmark rollout and scoring produce one target $y^{\star}_{i,l}$ per candidate and branch. The targets supervise the corresponding predictions $\hat{y}_{i,l}$ and form an equally weighted empirical outcome distribution for each candidate. At inference, the learned branch code $\omega_{l}=P_{\mathrm{ctx}}^{(l)}(\bar{\mathbf{r}}_{t})$ is computed from the current scene (Section 4.2).

[^1]: M. R. Barın, G. Aydemir, and F. Güney Robust bird’s eye view segmentation by adapting dinov2. In ECCV Workshops, Cited by: §5.

[^2]: H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom Nuscenes: a multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 11621–11631. Cited by: §5.1.

[^3]: W. Cao, M. Hallgarten, T. Li, D. Dauner, X. Gu, C. Wang, Y. Miron, M. Aiello, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, A. Geiger, and K. Chitta Pseudo-simulation for autonomous driving. In Proceedings of The 9th Conference on Robot Learning, Proceedings of Machine Learning Research, Vol. 305, pp. 4709–4722. External Links: [Link](https://proceedings.mlr.press/v305/cao25a.html) Cited by: item 3, §1, §2, §2, §5.1, §5.1, §5.2.

[^4]: S. Chen, B. Jiang, H. Gao, B. Liao, Q. Xu, Q. Zhang, C. Huang, W. Liu, and X. Wang VADv2: end-to-end vectorized autonomous driving via probabilistic planning. External Links: 2402.13243 Cited by: §2, Table 1.

[^5]: Y. Chen, Y. Wang, and Z. Zhang DrivingGPT: unifying driving world modeling and planning with multi-modal autoregressive transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 26890–26900. Cited by: §2, Table 1.

[^6]: K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger TransFuser: imitation with transformer-based sensor fusion for autonomous driving. IEEE Transactions on Pattern Analysis and Machine Intelligence 45 (11), pp. 12878–12895. Cited by: §1, §2, Table 1, Table 1, Table 2, Table 3.

[^7]: D. Dauner, M. Hallgarten, A. Geiger, and K. Chitta Parting with misconceptions about learning-based vehicle motion planning. In Proceedings of The 7th Conference on Robot Learning, J. Tan, M. Toussaint, and K. Darvish (Eds.), Proceedings of Machine Learning Research, Vol. 229, pp. 1268–1281. Cited by: §5.1.

[^8]: D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, A. Geiger, and K. Chitta NAVSIM: data-driven non-reactive autonomous vehicle simulation and benchmarking. In Proceedings of the 38th International Conference on Neural Information Processing Systems, NIPS ’24, Red Hook, NY, USA. Cited by: item 3, §1, §2, §2, §5.1, §5.2.

[^9]: L. Feng, Y. Gao, E. Zablocki, Q. Li, W. Li, S. Liu, M. Cord, and A. Alahi RAP: 3d rasterization augmented end-to-end planning. External Links: 2510.04333 Cited by: Table 2.

[^10]: R. Feng, N. Xi, D. Chu, R. Wang, Z. Deng, A. Wang, L. Lu, J. Wang, and Y. Huang ARTEMIS: autoregressive end-to-end trajectory planning with mixture of experts for autonomous driving. External Links: 2504.19580 Cited by: §2, Table 1.

[^11]: K. Guo, H. Liu, X. Wu, J. Pan, and C. Lv IPad: iterative proposal-centric end-to-end autonomous driving. External Links: 2505.15111 Cited by: §1, §2, Table 1.

[^12]: E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen LoRA: low-rank adaptation of large language models. In International Conference on Learning Representations, Cited by: §5.

[^13]: Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, L. Lu, X. Jia, Q. Liu, J. Dai, Y. Qiao, and H. Li Planning-oriented autonomous driving. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Vol., pp. 17853–17862. Cited by: §1, §2, Table 1, Table 3.

[^14]: A. Jiang, Y. Gao, Z. Sun, Y. Wang, J. Wang, J. Chai, Q. Cao, Y. Heng, H. Jiang, Y. Dong, Z. Zhang, X. Guo, H. Sun, and H. Zhao DiffVLA: vision-language guided diffusion planning for autonomous driving. External Links: 2505.19381, [Link](https://arxiv.org/abs/2505.19381) Cited by: Table 2.

[^15]: B. Jiang, S. Chen, Q. Xu, B. Liao, J. Chen, H. Zhou, Q. Zhang, W. Liu, C. Huang, and X. Wang VAD: vectorized scene representation for efficient autonomous driving. External Links: 2303.12077 Cited by: §1, §2, Table 3.

[^16]: X. Jiang, Y. Ma, P. Li, L. Xu, X. Wen, K. Zhan, Z. Xia, P. Jia, X. Lang, and S. Sun TransDiffuser: diverse trajectory generation with decorrelated multi-modal representation for end-to-end autonomous driving. External Links: 2505.09315, [Link](https://arxiv.org/abs/2505.09315) Cited by: §2, §2.

[^17]: N. Karnchanachari, D. Geromichalos, K. S. Tan, N. Li, C. Eriksen, S. Yaghoubi, N. Mehdipour, G. Bernasconi, W. K. Fong, Y. Guo, and H. Caesar Towards learning-based planning: the nuplan benchmark for real-world autonomous driving. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pp. 629–636. Cited by: §5.1.

[^18]: E. Kirby, A. Boulch, Y. Xu, Y. Yin, G. Puy, É. Zablocki, A. Bursuc, S. Gidaris, R. Marlet, F. Bartoccioni, A. Cao, N. Samet, T. Vu, and M. Cord Driving on registers. In CVPR, Cited by: item 3, §1, §1, §2, Table 1, Table 2, Table 3, §5.2.

[^19]: Y. Lee, J. Hwang, S. Lee, Y. Bae, and J. Park An energy and gpu-computation efficient backbone network for real-time object detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, pp. 752–760. Cited by: §5.2.

[^20]: K. Li, Z. Li, S. Lan, Y. Xie, Z. Zhang, J. Liu, Z. Wu, Z. Yu, and J. M. Alvarez Hydra-mdp++: advancing end-to-end driving via expert-guided hydra-distillation. External Links: 2503.12820 Cited by: §1, §2, Table 1.

[^21]: Y. Li, L. Fan, J. He, Y. Wang, Y. Chen, Z. Zhang, and T. Tan Enhancing end-to-end autonomous driving with latent world model. External Links: 2406.08481 Cited by: Table 1.

[^22]: Y. Li, S. Shang, W. Liu, B. Zhan, H. Wang, Y. Wang, Y. Chen, X. Wang, Y. An, C. Tang, L. Hou, L. Fan, and Z. Zhang DriveVLA-w0: world models amplify data scaling law in autonomous driving. External Links: 2510.12796 Cited by: Table 1.

[^23]: Y. Li, Y. Wang, Y. Liu, J. He, L. Fan, and Z. Zhang End-to-end driving with online trajectory evaluation via bev world model. External Links: 2504.01941 Cited by: §2, Table 1.

[^24]: Y. Li, K. Xiong, X. Guo, F. Li, S. Yan, G. Xu, L. Zhou, L. Chen, H. Sun, B. Wang, K. Ma, G. Chen, H. Ye, W. Liu, and X. Wang ReCogDrive: a reinforced cognitive framework for end-to-end autonomous driving. In International Conference on Learning Representation (ICLR), Cited by: Table 1.

[^25]: Z. Li, K. Li, S. Wang, S. Lan, Z. Yu, Y. Ji, Z. Li, Z. Zhu, J. Kautz, Z. Wu, Y. Jiang, and J. M. Alvarez Hydra-mdp: end-to-end multimodal planning with multi-target hydra-distillation. External Links: 2406.06978 Cited by: §1, §2, Table 1.

[^26]: Z. Li, W. Yao, Z. Wang, X. Sun, J. Chen, N. Chang, M. Shen, J. Song, Z. Wu, S. Lan, and J. M. Alvarez ZTRS: zero-imitation end-to-end autonomous driving with trajectory scoring. External Links: 2510.24108 Cited by: §2, Table 2.

[^27]: Z. Li, W. Yao, Z. Wang, X. Sun, J. Chen, N. Chang, M. Shen, Z. Wu, S. Lan, and J. M. Álvarez Generalized trajectory scoring for end-to-end multimodal planning. ArXiv abs/2506.06664. Cited by: §2, Table 2, Table 2.

[^28]: B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang, and X. Wang DiffusionDrive: truncated diffusion model for end-to-end autonomous driving. In CVPR, pp. 12037–12047. Cited by: §2, Table 1, Table 2.

[^29]: Y. Liao, J. Xie, and A. Geiger Kitti-360: a novel dataset and benchmarks for urban scene understanding in 2d and 3d. IEEE Transactions on Pattern Analysis and Machine Intelligence 45 (3), pp. 3292–3310. Cited by: §5.1.

[^30]: J. Liu, Z. Cui, T. Zhang, Z. Jing, Z. Ying, Y. Zhang, J. Wang, X. Jiang, A. Liu, and X. Liu SafeGen: goal-conditioned video diffusion of safety-critical scenarios for vlm-based autonomous driving. arXiv preprint arXiv:2404.08471. Cited by: §2.

[^31]: J. Liu, Y. Guo, F. Zhong, T. Zhang, Z. Jing, S. Liang, J. Wang, M. Zhang, A. Liu, and X. Liu Adversarial generation and collaborative evolution of safety-critical scenarios for autonomous vehicles. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40, pp. 38926–38934. Cited by: §2.

[^32]: L. Liu, C. Jia, G. Yu, Z. Song, J. Li, F. Jia, P. Wu, X. Hao, and Y. Luo GuideFlow: constraint-guided flow matching for planning in end-to-end autonomous driving. ArXiv abs/2511.18729. Cited by: §2, Table 2.

[^33]: W. G. Najm, J. D. Smith, and M. Yanagisawa Pre-crash scenario typology for crash avoidance research. Technical report Technical Report DOT HS 810 767, National Highway Traffic Safety Administration. Cited by: §5.1.

[^34]: OpenDriveLab OpenScene: autonomous grand challenge toolkits. Note: [https://github.com/OpenDriveLab/OpenScene](https://github.com/OpenDriveLab/OpenScene) Cited by: §5.1.

[^35]: M. Oquab, T. Darcet, T. Moutakanni, H. V. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, R. Howes, P. Huang, H. Xu, V. Sharma, S. Li, W. Galuba, M. Rabbat, M. Assran, N. Ballas, G. Synnaeve, I. Misra, H. Jegou, J. Mairal, P. Labatut, A. Joulin, and P. Bojanowski DINOv2: learning robust visual features without supervision. TMLR. Cited by: §5.2, §5.

[^36]: R. Rockafellar and S. Uryasev Conditional value-at-risk for general loss distributions. Journal of Banking & Finance 26 (7), pp. 1443–1471. Cited by: §1.

[^37]: C. Shi, S. Shi, K. Sheng, B. Zhang, and L. Jiang DriveX: omni scene modeling for learning generalizable world knowledge in autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 28599–28609. Cited by: Table 1.

[^38]: C. Sima, K. Chitta, Z. Yu, S. Lan, P. Luo, A. Geiger, H. Li, and J. M. Alvarez Centaur: robust end-to-end autonomous driving with test-time training. External Links: 2503.11650 Cited by: Table 1, §5.1, §5.4.

[^39]: Z. Song, L. Liu, H. Pan, B. Liao, M. Guo, L. Yang, Y. Zhang, S. Xu, C. Jia, and Y. Luo DIVER: reinforced diffusion breaks imitation bottlenecks in end-to-end autonomous driving. External Links: 2507.04049 Cited by: §2, Table 1.

[^40]: B. Sun, Y. Cao, Y. Wang, R. Wang, J. Shang, X. Feng, J. Lu, J. Shi, S. Yang, X. Yan, and Z. Song MindDrive: an all-in-one framework bridging world models and vision-language model for end-to-end autonomous driving. External Links: 2512.04441 Cited by: §1, Table 1.

[^41]: P. Sun, H. Kretzschmar, X. Dotiwalla, A. Chouard, V. Patnaik, P. Tsui, J. Guo, Y. Zhou, Y. Chai, B. Caine, et al. Scalability in perception for autonomous driving: waymo open dataset. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 2446–2454. Cited by: §5.1.

[^42]: W. Sun, X. Lin, Y. Shi, C. Zhang, H. Wu, and S. Zheng SparseDrive: end-to-end autonomous driving via sparse scene representation. External Links: 2405.19620 Cited by: §2.

[^43]: J. Wang, Y. Zheng, X. Liu, Z. Xing, P. Li, G. Li, K. Ma, G. Chen, H. Ye, Z. Xia, L. Chen, and Q. Zhang MeanFuser: fast one-step multi-modal trajectory generation and adaptive reconstruction via meanflow for end-to-end autonomous driving. External Links: 2602.20060, [Link](https://arxiv.org/abs/2602.20060) Cited by: §2, Table 1.

[^44]: J. Wang, X. Zhang, Z. Xing, S. Gu, X. Guo, Y. Hu, Z. Song, Q. Zhang, X. Long, and W. Yin ComDrive: comfort-oriented end-to-end autonomous driving. External Links: 2410.05051 Cited by: §2.

[^45]: W. Wang, J. Dai, Z. Chen, Z. Huang, Z. Li, X. Zhu, X. Hu, T. Lu, L. Lu, H. Li, et al. Internimage: exploring large-scale vision foundation models with deformable convolutions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 14408–14419. Cited by: §5.2.

[^46]: X. Wang, X. Zhang, Z. Luo, Q. Sun, Y. Cui, J. Wang, F. Zhang, Y. Wang, Z. Li, Q. Yu, et al. Emu3: next-token prediction is all you need. arXiv preprint arXiv:2409.18869. Cited by: §5.2.

[^47]: X. Weng, B. Ivanovic, Y. Wang, Y. Wang, and M. Pavone PARA-drive: parallelized architecture for real-time autonomous driving. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Vol., pp. 15449–15458. Cited by: §1, §2, Table 1.

[^48]: P. Xiao, Z. Shao, S. Hao, Z. Zhang, X. Chai, J. Jiao, Z. Li, J. Wu, K. Sun, K. Jiang, et al. Pandaset: advanced sensor suite dataset for autonomous driving. In 2021 IEEE international intelligent transportation systems conference (ITSC), pp. 3095–3101. Cited by: §5.1.

[^49]: Z. Xing, X. Zhang, Y. Hu, B. Jiang, T. He, Q. Zhang, X. Long, and W. Yin GoalFlow: goal-driven flow matching for multimodal trajectories generation in end-to-end autonomous driving. 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1602–1611. Cited by: §2, Table 1.

[^50]: W. Yao, Z. Li, S. Lan, Z. Wang, X. Sun, J. M. Alvarez, and Z. Wu DriveSuprim: towards precise trajectory selection for end-to-end planning. External Links: 2506.06659 Cited by: §2, Table 1, Table 2.

[^51]: C. Yuan, Z. Zhang, J. Sun, S. Sun, Z. Huang, C. D. W. Lee, D. Li, Y. Han, A. Wong, K. P. Tee, and M. H. A. Jr DRAMA: an efficient end-to-end motion planner for autonomous driving with mamba. External Links: 2408.03601 Cited by: Table 1.

[^52]: Y. Zhao, J. Pan, T. Hu, M. Yin, F. Yang, J. Liu, and X. Liang NSF-hrpt: neural semantic field meets hierarchical risk perception tree for safety-critical scenario assessment. External Links: 2608.04776, [Link](https://arxiv.org/abs/2608.04776) Cited by: §2.

[^53]: W. Zheng, R. Song, X. Guo, C. Zhang, and L. Chen GenAD: generative end-to-end autonomous driving. External Links: 2402.11502 Cited by: §2.

[^54]: Y. Zheng, P. Yang, Z. Xing, Q. Zhang, Y. Zheng, Y. Gao, P. Li, T. Zhang, Z. Xia, P. Jia, and D. Zhao World4Drive: end-to-end autonomous driving via intention-aware physical latent world model. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), Cited by: §2, Table 1.

[^55]: H. Zhou, L. Lin, J. Wang, Y. Lu, D. Bai, B. Liu, Y. Wang, A. Geiger, and Y. Liao Hugsim: a real-time, photo-realistic and closed-loop simulator for autonomous driving. IEEE Transactions on Pattern Analysis and Machine Intelligence. Cited by: item 3, §5.1, §5.2.

[^56]: Z. Zhou, T. Cai, S. Z. Zhao, Y. Zhang, Z. Huang, B. Zhou, and J. Ma AutoVLA: a vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, Cited by: Table 1.

[^57]: J. Zhu, W. Wang, Z. Chen, Z. Liu, S. Ye, L. Gu, H. Tian, Y. Duan, W. Su, J. Shao, et al. Internvl3: exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479. Cited by: §5.2.
---
title: "Test-Time Trajectory Optimization for Autonomous Driving"
source: "https://arxiv.org/html/2606.07170v1"
author:
published:
created: 2026-10-19
description:
tags:
  - "clippings"
---
Yihong Xu Affiliation: valeo.ai, Paris, France    Éloi Zablocki Affiliation: valeo.ai, Paris, France    Yuan Yin Affiliation: valeo.ai, Paris, France    Elias Ramzi Affiliation: valeo.ai, Paris, France    Ellington Kirby Affiliation: valeo.ai, Paris, France    Alexandre Boulch Affiliation: valeo.ai, Paris, France    Matthieu Cord Affiliation: valeo.ai, Paris, France Affiliation: Sorbonne Université, CNRS, ISIR, F-75005 Paris, France

###### Abstract

End-to-end planners for autonomous driving typically generate a set of candidate trajectories, score each one, and return the highest-scoring candidate. However, the scorer is applied only after the proposals are generated and cannot influence the set of trajectories: a weak set of candidates limits planning performance regardless of the scorer’s quality. We instead treat the scorer as a learned trajectory-level reward function and search for trajectories that maximize it. Our method, TOAD, runs the Cross-Entropy Method at test time, warm-started from the planner’s proposals. It requires no retraining and is plug-and-play for existing planners. Across six base planners, TOAD improves results on NAVSIM-v1 (94.7 PDMS), NAVSIM-v2 (56.3 EPDMS), and the closed-loop HUGSIM benchmark. The code will be made publicly available [via the project page](https://valeoai.github.io/TOAD/).

> Keywords: autonomous driving, end-to-end planning, test-time optimization

## 1 Introduction

End-to-end (E2E) planning has become the leading approach for autonomous driving [^1] [^16] [^37]. A single network maps sensor data and ego state directly to a driving trajectory. Several state-of-the-art methods share a common recipe. They first generate a finite set of candidate trajectories, either drawn from a large pre-computed vocabulary [^30] [^32] [^31] [^45] or produced on the fly [^23] [^11] [^13]. A scorer then picks the best trajectory from this set.

The scorer is at the core of the planning pipeline. It is trained to imitate an oracle of driving quality (safety, progress, comfort), and can therefore be seen as a learned, trajectory-level reward function, an objective that can be optimized. Yet current planners use it only once: generation is one-shot, and the scorer only ranks the fixed set of proposals. If every candidate is poor, the planner has no recourse and still returns the best of a bad set. We see this as a missed opportunity and ask whether the scorer can be used for more: not just rank a fixed set once, but guide a search for trajectories that score well.

We introduce TOAD, a test-time procedure that turns the frozen scorer into the objective of a search over trajectories, at no training cost. Concretely, TOAD wraps a frozen E2E planner in an inference-time loop based on the Cross-Entropy Method (CEM) [^38]. At each iteration, we sample trajectories from a Gaussian distribution, score them with the frozen scorer, refit the Gaussian to the top-k candidates (‘elites’), and repeat. The search is warm-started with the planner’s selected trajectory and confined to its neighborhood, which limits how far the search can drift from the scorer’s training distribution.

A key finding is that not all trained scorers can serve as rewards. Test-time search succeeds only when the scorer stays accurate beyond its training proposals, since it explores trajectories outside any fixed set. Scorers fit to a fixed vocabulary [^30] [^32] [^31] [^45] fail: they are accurate on their own candidates but degrade sharply elsewhere, and our loop with them drops below the baseline, despite looking strong under standard ranking evaluation. Only a disentangled scorer trained to evaluate freely decoded trajectories [^23] works in TOAD. Test-time search thus exposes a hidden weakness in standard scorers that ranking evaluation cannot reveal.

As it only requires only a frozen planner and a generalizing scorer, TOAD is plug-and-play. We attach a single frozen scorer to six base planners, iPad [^13], Hydra-MDP [^30], GTRS [^32], ZTRS [^31], RAP [^11], and DrivoR [^23], and improve PDMS for all six on NAVSIM-v1 [^9] and EPDMS on NAVSIM-v2 [^4]. Gains are largest for lower-performing planners and smallest for stronger ones (+43.6% on iPad, +3.1% on DrivoR), as TOAD closes the gap between the base planner and the scorer’s best-scoring trajectory. The gains come from the search itself, not from a stronger scorer: simply re-ranking the base planner’s proposals with the scorer can even hurt performance, whereas TOAD improves it. On NAVSIM-v2, DrivoR + TOAD reaches 56.3 EPDMS, a new state of the art that nearly matches the privileged PDM-Closed (56.6) [^8]. TOAD also improves DrivoR’s zero-shot HD-Score on the photorealistic closed-loop HUGSIM simulator [^60]. Code will be released.

Our contributions are as follows:

- We reframe the E2E trajectory scorer as a learned reward function that can be optimized, not only used to rank a one-shot set of proposals.
- We propose TOAD, a test-time search that plugs into any base planner with no retraining.
- We show that a successful optimization requires a scorer that generalizes beyond its training proposals, exposing a weakness in fixed-vocabulary scorers.
- Across six planners, TOAD gives consistent gains on NAVSIM-v1 and NAVSIM-v2, a new state of the art on NAVSIM-v2, and improved closed-loop performance on HUGSIM, at negligible cost.

## 2 Related Work

End-to-End Driving and ‘Score-and-Select’ Approaches.

E2E learning is now dominant in autonomous driving. Early systems used complex pipelines of hand-designed modules for detection, tracking, and mapping [^57] [^56]. Recent methods instead map sensors and ego state to a trajectory with a single network [^16] [^21] [^48], often dropping 3D supervision and bird’s-eye-view representations [^33] [^23]. Progress is measured on open-loop or pseudo closed-loop benchmarks such as NAVSIM-v1/v2 [^9] [^4] and simulators such as HUGSIM [^60].

The strongest E2E planners share a recipe: they generate candidate trajectories, score each, and output the highest scored [^30]. Candidates come from a large vocabulary of pre-computed trajectories [^30] [^32] [^31] or are generated on the fly, e.g., by diffusion or learnable queries [^33] [^52] [^13] [^11] [^23]. Either way, the scorer is trained to imitate a driving-quality oracle and selects the trajectory by argmax. A clear trend is to enlarge the candidate set with ever denser vocabularies [^53] [^45] which increases runtime latency [^23] and often overfits the scorer to the fixed vocabulary. Further, oracle studies show an ideal selection can beat the human driver [^53], suggesting the candidate set, not the scorer, often limits performance. Some scorers are trained to transfer across vocabularies [^32]. However, they still rank a fixed set, which differs from staying accurate on freely searched trajectories. We instead ask whether a trained scorer can act as a reward function and search for trajectories beyond that set, further boosting the performance.

Test-Time Optimization and Search for Planning.

*Sampling-based optimization and MPC* have a long history in robotics and driving. Gradient-free samplers like the Cross-Entropy Method [^38] [^24] [^36] and Model Predictive Path Integral control [^49] sample trajectories, score a cost, and refit toward low-cost regions, typically inside Model Predictive Control (MPC). Classical MPC for driving uses hand-designed costs; later work learns parts of it, e.g., cost maps from images [^10] or cost weights in a differentiable planner [^17]. Closer to us, UniAD [^16] post-optimizes its trajectory against a predicted occupancy map at inference. We share this search-at-test-time idea but optimize a full learned end-to-end scorer, without predicting any intermediate representation such as future occupancy maps.

*Test-time training* adapts the model itself rather than the trajectory. Centaur [^40] updates an E2E planner at test time by minimizing an uncertainty measure over its predictions. We are complementary and lighter: model weights stay frozen, we only search the trajectory space.

*Inference-time search* on a fixed model has driven large gains in language models, via inference-time reasoning [^18] [^12] and search guided by verifiers or reward models [^2]. This is appearing in driving too: re-ranking candidates with a vision-language model [^58] or scoring them online with a learned world model [^28] [^43]. We also search at inference, but over a continuous trajectory space rather than a fixed set, using a single learned scorer rather than an auxiliary model.

*Learned scorers as rewards* connect our setting to model-based RL, where a learned reward guides action selection. Several driving methods use a reward during training: RL fine-tuning of planners [^29], reinforced diffusion for generation [^41], and RLHF for driving styles [^25]. We instead use a frozen, already-trained scorer purely at inference, as the search objective, with no policy learning and no change to the generator.

## 3 Method

![[drivor-cem.png|Refer to caption]]

Figure 1: Overview of TOAD. A frozen base planner outputs trajectory proposals and a selected anchor τ base \\tau\_{\\text{base}} (left). Trajectories are mapped to controls by the inverse bicycle model ( BM − 1 \\operatorname{BM}^{-1} ). Then, a Cross-Entropy Method (CEM) loop runs for K steps in control space: sample controls, rolls them out with the bicycle model ( \\operatorname{BM} ), scores them under ‘Scorer + reg’ (the learned reward with comfort and anchor regularizers), refits the Gaussian to the highest-scoring elites. The scorer and BM are frozen.

We present TOAD, illustrated in Figure 1, a scorer-based method that enhances trajectories produced by an arbitrary planner at test time. We introduce the scorer in subsection 3.1 and then detail the optimization algorithm and its design choices in subsection 3.2.

### 3.1 Scorer as a Reward Function

Our method uses a scorer $S$ as a reward function over trajectories. A trajectory $\tau=(p_{1},\dots,p_{n_{p}})$ is a sequence of $n_{p}$ future poses $p_{t}=(x_{t},y_{t},\theta_{t})\in\mathbb{R}^{3}$ in the ego frame. Conditioned on the observation $o$ (sensor inputs and ego state) and the goal $g$ (target or route), the scorer maps a complete trajectory to a scalar, $S(\tau;o,g)\in\mathbb{R}$. $S$ is a neural network trained jointly with a base planner. We treat $S$ as a *reward* function over the trajectory space, which we treat as the action space. This enables searching for reward-maximal trajectories rather than ranking a fixed candidate set.

### 3.2 Test-Time Trajectory Search

We instantiate this search with the Cross-Entropy Method (CEM) [^38], a classical sampling-based optimizer. To keep the search where the scorer is reliable, we confine it to a trust region around the base planner’s proposals (Figure 1, left). The following paragraphs describe and motivate each design choice; we ablate them individually in section 5.

Optimizing in control space. For the sampling-based search, the choice of search space is not obvious. Sampling directly in pose space would yield jagged, dynamically infeasible trajectories, since it ignores vehicle dynamics entirely. We instead inject this prior through a kinematic bicycle model ($\operatorname{BM}$) and search over a sequence of controls,

$$
u=(u_{t})_{t=1}^{n_{p}},\qquad u_{t}=(a_{t},\omega_{t})\in\mathbb{R}^{2},
$$

where $a_{t}$ and $\omega_{t}$ are the per-step longitudinal acceleration and yaw rate. Our $\operatorname{BM}$ is invertible, so controls and trajectories are equivalent representations. From a control sequence $u$ we roll out a trajectory at the current ego speed $v_{0}$ as $\tau=\operatorname{BM}(u,v_{0})$, and recover the controls by inverse kinematics, $u=\operatorname{BM}^{-1}(\tau,v_{0})$, both illustrated in Figure 1. This approach is standard in sampling-based control [^49] [^24] and trajectory optimization [^14] [^54], and buys two properties for free: every sample is smooth and feasible, and the comfort, defined on accelerations (the controls) and jerks (their derivatives), can be computed directly from $u$.

Planner-based trust region. In principle, CEM can run from any initialization (see 4(a) for ablations). We instead deliberately constrain the search to a small trust region around the base trajectory proposal conditioned on the observation $o$ and the navigation goal $g$, because the learned scorer of subsection 3.1 is more reliable nearby. Outside this region, the search can drift toward trajectories that the scorer rates highly but cannot truly judge, which could lead to reward hacking.

Formally, we assume access to a trained base planner that maps an observation $o$ and a goal $g$ to a set of $N$ candidate trajectories together with a selected trajectory,

$$
\textsc{BasePlanner}(o,g)=(\mathcal{T},\tau_{\text{base}}),\quad\tau_{\text{base}}\in\mathcal{T}=\{\tau^{(i)}\}_{i=1}^{N}.
$$

This formulation covers existing planners, including single-candidate ones where $\mathcal{T}=\{\tau_{\text{base}}\}$. Given the ego speed $v_{0}$, these map equivalently to control sequences, $\mathcal{U}=\{u^{(i)}=\operatorname{BM}^{-1}(\tau^{(i)},v_{0})\}_{i=1}^{N}$, with a selected one $u_{\text{base}}$. We use these to define the trust region, with $u_{\text{base}}$ its anchor, while the spread of $\mathcal{U}$ sets its extent.

TOAD: CEM with trust region prior. In TOAD, the trust-region prior shapes both the CEM cost and its initialization. CEM maximizes $J(u)$, an objective combining $S$ and two closed-form regularizers:

$$
J(u)=S\big(\operatorname{BM}(u,v_{0});o,g\big)\;-\;\lambda_{a}\,C_{\text{anchor}}(u)\;-\;\lambda_{c}\,C_{\text{comf}}(u).
$$

The first regularizer, $C_{\text{anchor}}(u)=\lVert u-u_{\text{base}}\rVert^{2}$, ties the search to the trust region by penalizing deviation from the anchor. The second, $C_{\text{comf}}$, accumulates squared violations of the standard kinematic comfort limits (longitudinal and lateral acceleration, jerk, yaw rate, and yaw acceleration); being exact and closed-form, it supplies a signal the learned scorer estimates only coarsely. We evaluate $C_{\text{comf}}$ over the future controls and the last executed control, penalizing uncomfortable transitions from the vehicle’s recent motion.

To start the search, the CEM mean is set to the anchor, $\mu_{0}=u_{\text{base}}$; the standard deviation starts from the spread of the base planner’s own proposals, $\sigma_{0}=\max(\beta\cdot\mathrm{std}(\mathcal{U}),\epsilon)$ with $\beta<1$ and a small floor $\epsilon$ that prevents exploration from collapsing entirely. Exploration thus depends on the planner’s own uncertainty. We run CEM over controls with a Gaussian $\mathcal{N}(\mu_{k},\mathrm{diag}(\sigma_{k}^{2}))$ in $\mathbb{R}^{2n_{p}}$. At iteration $k=1,\dots,K$, we draw $M$ control sequences, roll them out, evaluate $J$, and keep the $E$ highest-scoring sequences as the elite set $\mathcal{E}_{k}$. We then refit the Gaussian to the elites: $\mu_{k}=\mathrm{mean}(\mathcal{E}_{k}),\sigma_{k}=\mathrm{std}(\mathcal{E}_{k}).$

Table 1: NAVSIM-v1. Comparison to existing camera-only methods on (navtest). Scores are the higher is better. Full comparison in Table 5.

| Method | PDMS ($\uparrow$) |
| --- | --- |
| PDM‑Closed [^8] | 89.1 |
| Human driver [^9] | 94.8 |
| ZTRS (V2-99) [^31] | 86.9 |
| \+ TOAD | 89.0 (${\uparrow}$ 2.4%) |
| GTRS (V2-99) [^32] | 90.4 |
| \+ TOAD | 90.9 (${\uparrow}$ 0.6%) |
| Hydra-MDP [^30] | 90.9 |
| \+ TOAD | 91.4 (${\uparrow}$ 0.6%) |
| iPad [^13] | 91.7 |
| \+ TOAD | 93.4 (${\uparrow}$ 1.9%) |
| RAP-DINO [^11] | 93.8 |
| \+ TOAD | 93.9 (${\uparrow}$ 0.1%) |
| DrivoR [^23] | 94.6 |
| \+ TOAD | 94.7 (${\uparrow}$ 0.1%) |

Table 2: NAVSIM-v2 navhard-two-stage. Comparison to existing methods on the NAVSIM-v2 benchmark test set using the EPDMS [^4]. Full comparison in Table 6.

| Method | EPDMS ($\uparrow$) |
| --- | --- |
| PDM‑Closed [^8] | 56.6 |
| iPad [^13] | 34.7 |
| \+ TOAD | 49.8 (${\uparrow}$ 43.6%) |
| RAP-DINO [^11] | 39.6 |
| \+ TOAD | 49.0 (${\uparrow}$ 23.9%) |
| Hydra-MDP [^30] | 40.9 |
| \+ TOAD | 49.7 (${\uparrow}$ 21.6%) |
| GTRS [^32] | 45.4 |
| \+ TOAD | 51.7 (${\uparrow}$ 13.8%) |
| ZTRS [^31] | 48.1 |
| \+ TOAD | 49.2 (${\uparrow}$ 2.3%) |
| DrivoR [^23] | 54.6 |
| \+ TOAD | 56.3 (${\uparrow}$ 3.1%) |

Table 3: What drives the gain? On top of iPad [^13] and Hydra-MDP [^30] we compare TOAD against (i) trajectory-smoothing, (ii) re-scoring, and (iii) using different scorers as reward.

| Method | EPDMS $(\uparrow)$ |
| --- | --- |
| iPad [^13] | 34.7 |
| \+ Smoothing (BM) | 35.0 ($\uparrow$ 0.8%) |
| \+ Re-scoring w/ DrivoR | 45.6 ($\uparrow$ 31.5%) |
| \+ Re-scoring w/ GTRS | 34.5 ($\downarrow$ 0.6%) |
| \+ Re-scoring w/ SparseDriveV2 | 30.3 ($\downarrow$ 12.5%) |
| \+ Search w/ GTRS scorer | 23.9 ($\downarrow$ 31.1%) |
| \+ Search w/ SparseDriveV2 scorer | 34.6 ($\downarrow$ 0.4%) |
| \+ TOAD (search w/ DrivoR scorer) | 49.8 ($\uparrow$ 43.6%) |
| Hydra-MDP [^30] | 40.9 |
| \+ Smoothing (BM) | 38.8 ($\downarrow$ 5.1%) |
| \+ Re-scoring w/ DrivoR | 37.3 ($\downarrow$ 8.8%) |
| \+ Re-scoring w/ GTRS | 40.3 ($\downarrow$ 1.5%) |
| \+ Re-scoring w/ SparseDriveV2 | 9.5 ($\downarrow$ 76.9%) |
| \+ Search w/ GTRS scorer | 38.1 ($\downarrow$ 6.9%) |
| \+ Search w/ SparseDriveV2 scorer | 29.6 ($\downarrow$ 27.7%) |
| \+ TOAD (search w/ DrivoR scorer) | 49.7 ($\uparrow$ 21.6%) |

After the final iteration, we roll out per CEM standards the trajectory induced by the converged mean, $\tau_{\text{mean}}=\operatorname{BM}(\mu_{K},v_{0})$. We return whichever of it and the base proposal $\tau_{\text{base}}$ maximizes the cost,

$$
\tau^{\star}=\operatorname*{arg\,max}_{\tau\in\{\tau_{\text{mean}},\,\tau_{\text{base}}\}}\;S(\tau;o,g)-\lambda_{c}\,C_{\text{comf}}\bigl(\operatorname{BM}^{-1}(\tau,v_{0})\bigr).
$$

$C_{\text{anchor}}$ is excluded here: the CEM mean should not be penalized for deviating from the anchor. Comparing against $\tau_{\text{base}}$ guarantees the returned trajectory never has a lower cost than $\tau_{\text{base}}$.

Scorer choice. CEM requires a scorer that stays accurate off the base planner’s proposal distribution, which is why TOAD uses DrivoR’s disentangled scorer [^23]. We highlight one property of our setup: the *same* scorer serves every base planner and need not be the planner’s own.

## 4 Experimental Results

We evaluate our test-time search on standard driving benchmarks. TOAD consistently improves various base planners on NAVSIM-v1/2, setting a new state of the art on NAVSIM-v2. It also improves over DrivoR on the closed-loop HUGSIM benchmark (subsection 4.2). We then demonstrate TOAD requires a scorer that is standalone from the trajectory head (subsection 4.3). Finally, we isolate the source of the gain, and ablate the design choices (section 5). Our code will be open-sourced.

### 4.1 Experimental Setup

Benchmarks and metrics. We evaluate on NAVSIM-v1 [^9] (navtest split) and NAVSIM-v2 [^4] (navhard-two-stage split), two pseudo-closed-loop benchmarks built on real driving logs. We also evaluate on HUGSIM [^60], a closed-loop evaluation benchmark with photorealistic scenes constructed using 3D Gaussian Splatting [^22]. For the ablation, we use warmup-two-stage <sup>1</sup> as validation set for NAVSIM-v2 containing 7 scenes. Metrics are detailed in Appendix A.  
Base planners. We show that TOAD is plug-and-play, by evaluating it on six base planners spanning both proposal styles, i.e., fixed-vocabulary scorers (Hydra-MDP [^30], GTRS [^32], ZTRS [^31]) and on-the-fly generators (iPad [^13], RAP [^11], DrivoR [^23]). For each, we use the public checkpoint and its released trajectory proposals; no models are updated.  
Hyperparameters. Unless stated otherwise, we run $K=5$ CEM iterations for planners with on-the-fly generators and $K=100$ for those with fixed-vocabulary scorers due to its much larger vocabulary ($>$ 8K) that initializes the search. The per-iteration candidate count $M$ is fixed to 64 considering search inference cost, and the elite count to $E=M/8$. We use an initial-std scale of $\beta=0.5$, floored at $\epsilon_{a}=0.1~\mathrm{m/s^{2}}$ and $\epsilon_{\omega}=0.025~\mathrm{rad/s}$. The objective uses comfort weight $\lambda_{c}=0.05$ and anchor weight $\lambda_{a}=0.5$. All values above are kept fixed across base planners and benchmarks; ablations are conducted in section 5.

### 4.2 Main results

NAVSIM-v2 [^4] and NAVSIM-v1 [^9]. Table 3 and Table 3 reports NAVSIM-v1 navtest and NAVSIM-v2 navhard-two-stage results. TOAD improves every base planner on both benchmarks. On NAVSIM-v2 the gains range from +2.3% (ZTRS) to +43.6% (iPad), and they compress the ranking: the six planners start almost 20 EPDMS points apart (34.7 to 54.6) but all land between 49.0 and 56.3 after search. The weakest planners gain the most, while DrivoR, already the strongest, gains +3.1% to reach 56.3 EPDMS, outperforming the strongest learned method (DriveFuture [^15], see in Table 6, 55.5) and is only 0.3 behind the privileged PDM-Closed (56.6), which uses ground-truth perception. Complete benchmark tables are provided in Table 5 and Table 6.

The search uses whichever sub-score has room to improve. On the harder, out-of-distribution NAVSIM-v2 scenes it trades a little progress for large safety gains in no-collision, drivable-area, and time-to-collision see Table 6. On the near-saturated NAVSIM-v1, where safety is already high, it instead extends progress. Again the weakest planners gain most (ZTRS +2.4%, iPad +1.9%) and the strongest least, with DrivoR reaching 94.7 PDMS, the top entry and -0.1 below the human driver.

Table 4: HUGSIM [^60] zero-shot scores. TOAD improves safety (NC, DAC, TTC) and comfort, raising HD-Score, while route completion (RC) sometimes drops, indicating more cautious driving.

<table><thead><tr><th></th><th></th><th colspan="6">KITTI360 <sup><a href="#fn:34">34</a></sup></th><th colspan="6">nuScenes <sup><a href="#fn:3">3</a></sup></th><th colspan="6">Pandaset <sup><a href="#fn:51">51</a></sup></th><th colspan="6">Waymo <sup><a href="#fn:44">44</a></sup></th></tr><tr><th></th><th></th><th>NC</th><th>DAC</th><th>TTC</th><th>Comf.</th><th>RC</th><th>HDS</th><th>NC</th><th>DAC</th><th>TTC</th><th>Comf.</th><th>RC</th><th>HDS</th><th>NC</th><th>DAC</th><th>TTC</th><th>Comf.</th><th>RC</th><th>HDS</th><th>NC</th><th>DAC</th><th>TTC</th><th>Comf.</th><th>RC</th><th>HDS</th></tr></thead><tbody><tr><th>LTF</th><th><sup><a href="#fn:7">7</a></sup></th><td>32.4</td><td>86.4</td><td>25.1</td><td>97.3</td><td>19.4</td><td>8.0</td><td>56.1</td><td>99.7</td><td>49.8</td><td>95.4</td><td>50.1</td><td>35.1</td><td>44.7</td><td>100.0</td><td>36.2</td><td>92.1</td><td>51.3</td><td>28.2</td><td>49.1</td><td>93.9</td><td>42.1</td><td>95.4</td><td>38.3</td><td>23.8</td></tr><tr><th>UniAD</th><th><sup><a href="#fn:16">16</a></sup></th><td>48.1</td><td>85.1</td><td>19.8</td><td>37.2</td><td>23.5</td><td>5.9</td><td>71.8</td><td>99.2</td><td>63.3</td><td>79.3</td><td>49.1</td><td>37.5</td><td>68.2</td><td>100.0</td><td>58.1</td><td>77.3</td><td>49.5</td><td>36.7</td><td>76.9</td><td>94.3</td><td>66.3</td><td>71.3</td><td>56.2</td><td>44.4</td></tr><tr><th>VAD</th><th><sup><a href="#fn:21">21</a></sup></th><td>24.4</td><td>83.2</td><td>15.0</td><td>94.2</td><td>17.1</td><td>3.9</td><td>57.2</td><td>97.6</td><td>42.9</td><td>97.4</td><td>43.1</td><td>24.5</td><td>42.4</td><td>100.0</td><td>26.5</td><td>94.9</td><td>38.7</td><td>15.4</td><td>46.6</td><td>85.5</td><td>32.6</td><td>94.9</td><td>29.4</td><td>11.0</td></tr><tr><th>GTRS-D</th><th><sup><a href="#fn:32">32</a></sup></th><td>47.9</td><td>90.4</td><td>43.1</td><td>94.4</td><td>29.5</td><td>19.5</td><td>71.3</td><td>99.8</td><td>64.8</td><td>96.0</td><td>59.3</td><td>47.4</td><td>55.7</td><td>99.9</td><td>48.0</td><td>91.6</td><td>56.0</td><td>37.2</td><td>65.5</td><td>96.4</td><td>61.0</td><td>95.1</td><td>52.5</td><td>41.0</td></tr><tr><th>DrivoR</th><th><sup><a href="#fn:23">23</a></sup></th><td>44.0</td><td>89.9</td><td>40.8</td><td>93.6</td><td>28.8</td><td>17.7</td><td>57.0</td><td>100.0</td><td>51.3</td><td>93.0</td><td>54.2</td><td>39.7</td><td>50.8</td><td>99.9</td><td>44.0</td><td>89.8</td><td>58.3</td><td>35.1</td><td>64.0</td><td>96.8</td><td>58.6</td><td>94.1</td><td>56.5</td><td>44.2</td></tr><tr><th colspan="2">DrivoR + TOAD</th><td>49.5</td><td>90.9</td><td>46.1</td><td>99.1</td><td>26.9</td><td>18.4</td><td>69.1</td><td>100.0</td><td>63.9</td><td>98.3</td><td>60.9</td><td>49.9</td><td>57.3</td><td>100.0</td><td>51.6</td><td>98.4</td><td>54.5</td><td>38.0</td><td>70.3</td><td>96.7</td><td>66.6</td><td>97.9</td><td>51.1</td><td>42.8</td></tr></tbody></table>

HUGSIM [^60]. Table 4 reports zero-shot closed-loop evaluation, applying TOAD on top of DrivoR. The safety sub-metrics (no-collision, drivable-area, time-to-collision) and, above all, comfort improve on nearly all four datasets, the comfort gain a direct effect of the control-space search. These gains raise the HUGSIM Driving Score (HDS) where there is room, most clearly on nuScenes (DrivoR: 39.7 to 49.9) and KITTI360. Route completion tends to drop, showing that the searched policy drives more cautiously, and on Waymo this caution leaves the score slightly lower. The improvements transfer to a closed-loop simulator with no tuning suggests they reflect genuinely safer driving.

Qualitative Results. We show in 2(a) the refined trajectory manages to avoid collision from the car in front. Using TOAD we produce a better trajectory, inexistent in the original set of trajectories. Aligned with the observations in 4(c), the reward is not always precise and can occasionally fail. Such failures may occur when visual cues are unreliable due to limited visibility, illustrated in 2(b).

![[DrivorMPC_success_TOAD.png|Refer to caption]]

(a) Success case

### 4.3 The gains require search and a generalizing scorer

Two factors explain TOAD’s gains: they come from search discovering new trajectories rather than re-ranking or smoothing existing ones, and they only appear with a scorer that remains accurate off the proposal distribution. We verify both in Table 3, with two base planners iPad [^13] and Hydra-MDP [^30] on navhard-two-stage. Detailed performance breakdown can be found in Table 7.

Search versus re-ranking and smoothing. We compare TOAD against two simpler modifications of the base planner. *Smoothing (BM)* maps the selected trajectory through the bicycle model to remove jitter without changing its route; it has little effect (iPad $+0.8\%$, Hydra-MDP $-5.1\%$), showing the gains are from better trajectories, as seen in Figure 2. *Re-scoring* re-ranks the original proposals with a scorer, without new trajectories. When using DrivoR’s scorer as in TOAD, re-ranking is inconsistent: it improves iPad ($+31.5\%$) but hurts Hydra-MDP ($-8.8\%$). In contrast, TOAD consistently improves performance ($+43.6\%$ and $+21.6\%$). The gains therefore come from search discovering better trajectories outside the proposal set, not from reordering or smoothing existing ones.

The scorer must generalize. We also run the search with different scorers as rewards. Only the DrivoR scorer [^23] improves performance; GTRS [^32] and SparseDriveV2 [^45] reduce it below baseline (e.g., iPad drops to 23.9 with GTRS). We attribute this to their training setup. GTRS and SparseDriveV2 are trained to score a fixed vocabulary of precomputed trajectories, so they receive no supervision away from that set and need not behave sensibly there. DrivoR, by contrast, is trained jointly with a trajectory generator while remaining disentangled from it, forcing it to score freely decoded trajectories rather than a fixed grid. We hypothesize this is why it remains reliable on the off-proposal trajectories explored by our search, making it the only scorer suitable as a reward.

### 4.4 Computation Footprint

<svg id="S4.F3.pic1" height="362.36" overflow="visible" version="1.1" viewBox="0 0 381.27 362.36" width="381.27"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="translate(0,362.36) matrix(1 0 0 -1 0 0) translate(160.09,0) translate(0,193.88) matrix(1.0 0.0 0.0 1.0 -160.09 -193.87)"><g transform="matrix(1 0 0 1 0 0) translate(170.93,0) translate(0,193.88)"><g style="--ltx-stroke-color:#CCCCCC;--ltx-fill-color:#CCCCCC;--ltx-fg-color:#CCCCCC;" color="#CCCCCC" fill="#CCCCCC" stroke="#CCCCCC" stroke-dasharray="3.0pt,3.0pt" stroke-dashoffset="0.0pt" stroke-width="0.4pt"><path style="fill:none" d="M -10.83 0 L 165.59 0 M -10.83 16.24 L 165.59 16.24 M -10.83 81.19 L 165.59 81.19"></path></g><g style="--ltx-stroke-color:#808080;--ltx-fill-color:#808080;--ltx-fg-color:#808080;" color="#808080" fill="#808080" stroke="#808080" stroke-width="0.2pt"><path style="fill:none" d="M 0 0 L 0 5.91 M 30.95 0 L 30.95 5.91 M 61.9 0 L 61.9 5.91 M 92.85 0 L 92.85 5.91 M 123.8 0 L 123.8 5.91 M 154.75 0 L 154.75 5.91 M 0 154.27 L 0 148.36 M 30.95 154.27 L 30.95 148.36 M 61.9 154.27 L 61.9 148.36 M 92.85 154.27 L 92.85 148.36 M 123.8 154.27 L 123.8 148.36 M 154.75 154.27 L 154.75 148.36"></path></g><g style="--ltx-stroke-color:#808080;--ltx-fill-color:#808080;--ltx-fg-color:#808080;" color="#808080" fill="#808080" stroke="#808080" stroke-width="0.2pt"><path style="fill:none" d="M -10.83 0 L -4.93 0 M -10.83 16.24 L -4.93 16.24 M -10.83 81.19 L -4.93 81.19 M 165.59 0 L 159.68 0 M 165.59 16.24 L 159.68 16.24 M 165.59 81.19 L 159.68 81.19"></path></g><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" stroke-width="0.4pt"><path style="fill:none" d="M -10.83 0 L -10.83 154.27 L 165.59 154.27 L 165.59 0 L -10.83 0 Z"></path><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(0.7071 0.7071 -0.7071 0.7071 -157.06 -180.61)"><foreignObject style="--ltx-fo-width:3.36em;--ltx-fo-height:0.68em;--ltx-fo-depth:0em;font-size:10pt;" height="9.46" overflow="visible" transform="matrix(1 0 0 -1 0 9.46)" width="46.55"><span id="S4.F3.pic1.1">DrivoR  <sup id="fnref:23-14"><a href="#fn:23">23</a></sup></span></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(0.7071 0.7071 -0.7071 0.7071 -100.59 -155.09)"><foreignObject style="--ltx-fo-width:2.18em;--ltx-fo-height:0.69em;--ltx-fo-depth:0em;font-size:10pt;" height="9.61" overflow="visible" transform="matrix(1 0 0 -1 0 9.61)" width="30.17"><span id="S4.F3.pic1.2">iPad  <sup id="fnref:13-10"><a href="#fn:13">13</a></sup></span></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(0.7071 0.7071 -0.7071 0.7071 -69.53 -154.98)"><foreignObject style="--ltx-fo-width:2.33em;--ltx-fo-height:0.68em;--ltx-fo-depth:0em;font-size:10pt;" height="9.46" overflow="visible" transform="matrix(1 0 0 -1 0 9.46)" width="32.29"><span id="S4.F3.pic1.3">RAP  <sup id="fnref:11-7"><a href="#fn:11">11</a></sup></span></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(0.7071 0.7071 -0.7071 0.7071 -32.52 -148.92)"><foreignObject style="--ltx-fo-width:2.79em;--ltx-fo-height:0.68em;--ltx-fo-depth:0em;font-size:10pt;" height="9.46" overflow="visible" transform="matrix(1 0 0 -1 0 9.46)" width="38.63"><span id="S4.F3.pic1.4">ZTRS  <sup id="fnref:31-8"><a href="#fn:31">31</a></sup></span></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(0.7071 0.7071 -0.7071 0.7071 -3.82 -151.16)"><foreignObject style="--ltx-fo-width:2.97em;--ltx-fo-height:0.68em;--ltx-fo-depth:0em;font-size:10pt;" height="9.46" overflow="visible" transform="matrix(1 0 0 -1 0 9.46)" width="41.03"><span id="S4.F3.pic1.5">GTRS  <sup id="fnref:32-11"><a href="#fn:32">32</a></sup></span></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(0.7071 0.7071 -0.7071 0.7071 -6.61 -184.91)"><foreignObject style="--ltx-fo-width:5.59em;--ltx-fo-height:0.69em;--ltx-fo-depth:0.19em;font-size:10pt;" height="12.3" overflow="visible" transform="matrix(1 0 0 -1 0 9.61)" width="77.3"><span id="S4.F3.pic1.6">Hydra-MDP  <sup id="fnref:30-12"><a href="#fn:30">30</a></sup></span></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 -22.64 -4.46)"><foreignObject style="--ltx-fo-width:0.5em;--ltx-fo-height:0.64em;--ltx-fo-depth:0em;font-size:10pt;" height="8.92" overflow="visible" transform="matrix(1 0 0 -1 0 8.92)" width="6.92"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="0"><semantics><mn>0</mn> <annotation encoding="application/x-tex">0</annotation></semantics></math></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 -36.48 11.78)"><foreignObject style="--ltx-fo-width:1.5em;--ltx-fo-height:0.64em;--ltx-fo-depth:0em;font-size:10pt;" height="8.92" overflow="visible" transform="matrix(1 0 0 -1 0 8.92)" width="20.76"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="100"><semantics><mn>100</mn> <annotation encoding="application/x-tex">100</annotation></semantics></math></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 -36.48 76.73)"><foreignObject style="--ltx-fo-width:1.5em;--ltx-fo-height:0.64em;--ltx-fo-depth:0em;font-size:10pt;" height="8.92" overflow="visible" transform="matrix(1 0 0 -1 0 8.92)" width="20.76"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="500"><semantics><mn>500</mn> <annotation encoding="application/x-tex">500</annotation></semantics></math></foreignObject></g> <g style="--ltx-stroke-color:#2E7DE8;--ltx-fill-color:#2E7DE8;--ltx-fg-color:#2E7DE8;" color="#2E7DE8" fill="#2E7DE8" stroke="#2E7DE8" transform="matrix(1.0 0.0 0.0 1.0 142.45 50.29)"><foreignObject style="--ltx-fg-color:#000000;--ltx-fo-width:1.78em;--ltx-fo-height:0.64em;--ltx-fo-depth:0em;font-size:10pt;" color="#000000" height="8.92" overflow="visible" transform="matrix(1 0 0 -1 0 8.92)" width="24.6"><span id="S4.F3.pic1.7" style="--ltx-fg-color:#2E7DE8;">19.1</span></foreignObject></g> <g style="--ltx-stroke-color:#2E7DE8;--ltx-fill-color:#2E7DE8;--ltx-fg-color:#2E7DE8;" color="#2E7DE8" fill="#2E7DE8" stroke="#2E7DE8" transform="matrix(1.0 0.0 0.0 1.0 111.5 36.08)"><foreignObject style="--ltx-fg-color:#000000;--ltx-fo-width:1.78em;--ltx-fo-height:0.64em;--ltx-fo-depth:0em;font-size:10pt;" color="#000000" height="8.92" overflow="visible" transform="matrix(1 0 0 -1 0 8.92)" width="24.6"><span id="S4.F3.pic1.8" style="--ltx-fg-color:#2E7DE8;">20.4</span></foreignObject></g> <g style="--ltx-stroke-color:#2E7DE8;--ltx-fill-color:#2E7DE8;--ltx-fg-color:#2E7DE8;" color="#2E7DE8" fill="#2E7DE8" stroke="#2E7DE8" transform="matrix(1.0 0.0 0.0 1.0 80.55 38.52)"><foreignObject style="--ltx-fg-color:#000000;--ltx-fo-width:1.78em;--ltx-fo-height:0.64em;--ltx-fo-depth:0em;font-size:10pt;" color="#000000" height="8.92" overflow="visible" transform="matrix(1 0 0 -1 0 8.92)" width="24.6"><span id="S4.F3.pic1.9" style="--ltx-fg-color:#2E7DE8;">19.4</span></foreignObject></g> <g style="--ltx-stroke-color:#2E7DE8;--ltx-fill-color:#2E7DE8;--ltx-fg-color:#2E7DE8;" color="#2E7DE8" fill="#2E7DE8" stroke="#2E7DE8" transform="matrix(1.0 0.0 0.0 1.0 53.06 128.5)"><foreignObject style="--ltx-fg-color:#000000;--ltx-fo-width:1.28em;--ltx-fo-height:0.64em;--ltx-fo-depth:0em;font-size:10pt;" color="#000000" height="8.92" overflow="visible" transform="matrix(1 0 0 -1 0 8.92)" width="17.68"><span id="S4.F3.pic1.10" style="--ltx-fg-color:#2E7DE8;">4.6</span></foreignObject></g> <g style="--ltx-stroke-color:#2E7DE8;--ltx-fill-color:#2E7DE8;--ltx-fg-color:#2E7DE8;" color="#2E7DE8" fill="#2E7DE8" stroke="#2E7DE8" transform="matrix(1.0 0.0 0.0 1.0 22.11 20.72)"><foreignObject style="--ltx-fg-color:#000000;--ltx-fo-width:1.28em;--ltx-fo-height:0.64em;--ltx-fo-depth:0em;font-size:10pt;" color="#000000" height="8.92" overflow="visible" transform="matrix(1 0 0 -1 0 8.92)" width="17.68"><span id="S4.F3.pic1.11" style="--ltx-fg-color:#2E7DE8;">1.9</span></foreignObject></g> <g style="--ltx-stroke-color:#2E7DE8;--ltx-fill-color:#2E7DE8;--ltx-fg-color:#2E7DE8;" color="#2E7DE8" fill="#2E7DE8" stroke="#2E7DE8" transform="matrix(1.0 0.0 0.0 1.0 -8.84 18.4)"><foreignObject style="--ltx-fg-color:#000000;--ltx-fo-width:1.28em;--ltx-fo-height:0.64em;--ltx-fo-depth:0em;font-size:10pt;" color="#000000" height="8.92" overflow="visible" transform="matrix(1 0 0 -1 0 8.92)" width="17.68"><span id="S4.F3.pic1.12" style="--ltx-fg-color:#2E7DE8;">1.9</span></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;--ltx-fg-color:#000000;" color="#000000" fill="#000000" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 40.96 0.44)"><foreignObject style="--ltx-fg-color:#000000;--ltx-fo-width:1.28em;--ltx-fo-height:0.64em;--ltx-fo-depth:0em;font-size:10pt;" color="#000000" height="8.92" overflow="visible" transform="matrix(1 0 0 -1 0 8.92)" width="17.68"><span id="S4.F3.pic1.13" style="--ltx-fg-color:#000000;">9.2</span></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 138.99 -11.73)"><foreignObject style="--ltx-fo-width:2.28em;--ltx-fo-height:0.64em;--ltx-fo-depth:0em;font-size:10pt;" height="8.92" overflow="visible" transform="matrix(1 0 0 -1 0 8.92)" width="31.52"><span id="S4.F3.pic1.14">300.9</span></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 108.04 -11.73)"><foreignObject style="--ltx-fo-width:2.28em;--ltx-fo-height:0.64em;--ltx-fo-depth:0em;font-size:10pt;" height="8.92" overflow="visible" transform="matrix(1 0 0 -1 0 8.92)" width="31.52"><span id="S4.F3.pic1.15">213.4</span></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 77.09 -11.73)"><foreignObject style="--ltx-fo-width:2.28em;--ltx-fo-height:0.64em;--ltx-fo-depth:0em;font-size:10pt;" height="8.92" overflow="visible" transform="matrix(1 0 0 -1 0 8.92)" width="31.52"><span id="S4.F3.pic1.16">228.4</span></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 46.14 -11.73)"><foreignObject style="--ltx-fo-width:2.28em;--ltx-fo-height:0.64em;--ltx-fo-depth:0em;font-size:10pt;" height="8.92" overflow="visible" transform="matrix(1 0 0 -1 0 8.92)" width="31.52"><span id="S4.F3.pic1.17">782.5</span></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 15.19 -11.73)"><foreignObject style="--ltx-fo-width:2.28em;--ltx-fo-height:0.64em;--ltx-fo-depth:0em;font-size:10pt;" height="8.92" overflow="visible" transform="matrix(1 0 0 -1 0 8.92)" width="31.52"><span id="S4.F3.pic1.18">118.8</span></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 -15.76 -11.73)"><foreignObject style="--ltx-fo-width:2.28em;--ltx-fo-height:0.64em;--ltx-fo-depth:0em;font-size:10pt;" height="8.92" overflow="visible" transform="matrix(1 0 0 -1 0 8.92)" width="31.52"><span id="S4.F3.pic1.19">104.5</span></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#2E7DE8;" fill="#2E7DE8" stroke="#000000"><path d="M -9.69 16.66 h 19.37 v 0.31 h -19.37 Z M 21.26 18.98 h 19.37 v 0.31 h -19.37 Z M 52.21 126.32 h 19.37 v 0.75 h -19.37 Z M 83.17 33.94 h 19.37 v 3.15 h -19.37 Z M 114.12 31.34 h 19.37 v 3.31 h -19.37 Z M 145.07 45.76 h 19.37 v 3.1 h -19.37 Z"></path></g><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#4D4D4D;" fill="#4D4D4D" stroke="#000000"><path d="M -9.69 0.19 h 19.37 v 16.47 h -19.37 Z M 21.26 1.49 h 19.37 v 17.49 h -19.37 Z M 52.21 109.89 h 19.37 v 16.43 h -19.37 Z M 83.17 17.55 h 19.37 v 16.38 h -19.37 Z M 114.12 14.91 h 19.37 v 16.43 h -19.37 Z M 145.07 29.41 h 19.37 v 16.35 h -19.37 Z"></path></g><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#D3D3D3;" fill="#D3D3D3" stroke="#000000"><path d="M -9.69 0 h 19.37 v 0.19 h -19.37 Z M 21.26 0 h 19.37 v 1.49 h -19.37 Z M 52.21 0 h 19.37 v 109.89 h -19.37 Z M 83.17 0 h 19.37 v 17.55 h -19.37 Z M 114.12 0 h 19.37 v 14.91 h -19.37 Z M 145.07 0 h 19.37 v 29.41 h -19.37 Z"></path></g><g style="--ltx-stroke-color:#FFFFFF;--ltx-fill-color:#FFFFFF;" fill="#FFFFFF" stroke="#FFFFFF" transform="matrix(1.0 0.0 0.0 1.0 -15.76 3.97)"><foreignObject style="--ltx-fo-width:2.28em;--ltx-fo-height:0.64em;--ltx-fo-depth:0em;font-size:10pt;" height="8.92" overflow="visible" transform="matrix(1 0 0 -1 0 8.92)" width="31.52"><span id="S4.F3.pic1.20" style="--ltx-fg-color:#FFFFFF;">101.4</span></foreignObject></g> <g style="--ltx-stroke-color:#FFFFFF;--ltx-fill-color:#FFFFFF;" fill="#FFFFFF" stroke="#FFFFFF" transform="matrix(1.0 0.0 0.0 1.0 15.19 5.78)"><foreignObject style="--ltx-fo-width:2.28em;--ltx-fo-height:0.64em;--ltx-fo-depth:0em;font-size:10pt;" height="8.92" overflow="visible" transform="matrix(1 0 0 -1 0 8.92)" width="31.52"><span id="S4.F3.pic1.21" style="--ltx-fg-color:#FFFFFF;">107.7</span></foreignObject></g> <g style="--ltx-stroke-color:#FFFFFF;--ltx-fill-color:#FFFFFF;" fill="#FFFFFF" stroke="#FFFFFF" transform="matrix(1.0 0.0 0.0 1.0 46.14 113.64)"><foreignObject style="--ltx-fo-width:2.28em;--ltx-fo-height:0.64em;--ltx-fo-depth:0em;font-size:10pt;" height="8.92" overflow="visible" transform="matrix(1 0 0 -1 0 8.92)" width="31.52"><span id="S4.F3.pic1.22" style="--ltx-fg-color:#FFFFFF;">101.2</span></foreignObject></g> <g style="--ltx-stroke-color:#FFFFFF;--ltx-fill-color:#FFFFFF;" fill="#FFFFFF" stroke="#FFFFFF" transform="matrix(1.0 0.0 0.0 1.0 77.09 21.29)"><foreignObject style="--ltx-fo-width:2.28em;--ltx-fo-height:0.64em;--ltx-fo-depth:0em;font-size:10pt;" height="8.92" overflow="visible" transform="matrix(1 0 0 -1 0 8.92)" width="31.52"><span id="S4.F3.pic1.23" style="--ltx-fg-color:#FFFFFF;">100.9</span></foreignObject></g> <g style="--ltx-stroke-color:#FFFFFF;--ltx-fill-color:#FFFFFF;" fill="#FFFFFF" stroke="#FFFFFF" transform="matrix(1.0 0.0 0.0 1.0 108.04 18.66)"><foreignObject style="--ltx-fo-width:2.28em;--ltx-fo-height:0.64em;--ltx-fo-depth:0em;font-size:10pt;" height="8.92" overflow="visible" transform="matrix(1 0 0 -1 0 8.92)" width="31.52"><span id="S4.F3.pic1.24" style="--ltx-fg-color:#FFFFFF;">101.2</span></foreignObject></g> <g style="--ltx-stroke-color:#FFFFFF;--ltx-fill-color:#FFFFFF;" fill="#FFFFFF" stroke="#FFFFFF" transform="matrix(1.0 0.0 0.0 1.0 138.99 33.13)"><foreignObject style="--ltx-fo-width:2.28em;--ltx-fo-height:0.64em;--ltx-fo-depth:0em;font-size:10pt;" height="8.92" overflow="visible" transform="matrix(1 0 0 -1 0 8.92)" width="31.52"><span id="S4.F3.pic1.25" style="--ltx-fg-color:#FFFFFF;">100.7</span></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 46.14 50.48)"><foreignObject style="--ltx-fo-width:2.28em;--ltx-fo-height:0.64em;--ltx-fo-depth:0em;font-size:10pt;" height="8.92" overflow="visible" transform="matrix(1 0 0 -1 0 8.92)" width="31.52"><span id="S4.F3.pic1.26" style="--ltx-fg-color:#000000;">676.7</span></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 77.09 4.32)"><foreignObject style="--ltx-fo-width:2.28em;--ltx-fo-height:0.64em;--ltx-fo-depth:0em;font-size:10pt;" height="8.92" overflow="visible" transform="matrix(1 0 0 -1 0 8.92)" width="31.52"><span id="S4.F3.pic1.27" style="--ltx-fg-color:#000000;">108.1</span></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 111.5 2.99)"><foreignObject style="--ltx-fo-width:1.78em;--ltx-fo-height:0.64em;--ltx-fo-depth:0em;font-size:10pt;" height="8.92" overflow="visible" transform="matrix(1 0 0 -1 0 8.92)" width="24.6"><span id="S4.F3.pic1.28" style="--ltx-fg-color:#000000;">91.8</span></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 138.99 10.25)"><foreignObject style="--ltx-fo-width:2.28em;--ltx-fo-height:0.64em;--ltx-fo-depth:0em;font-size:10pt;" height="8.92" overflow="visible" transform="matrix(1 0 0 -1 0 8.92)" width="31.52"><span id="S4.F3.pic1.29" style="--ltx-fg-color:#000000;">181.1</span></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(0.0 1.0 -1.0 0.0 -24.07 93.77)"><foreignObject style="--ltx-fo-width:5.4em;--ltx-fo-height:0.75em;--ltx-fo-depth:0.25em;font-size:10pt;" height="13.84" overflow="visible" transform="matrix(1 0 0 -1 0 10.38)" width="74.72"><span id="S4.F3.pic1.30">ms / sample</span></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#FFFFFF;" fill="#FFFFFF" stroke="#000000"><path d="M 88.24 89.19 h 121.83 v 57.09 h -121.83 Z"></path></g><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#FFFFFF;" fill="#FFFFFF" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 92.39 100.49)"><g transform="matrix(1 0 0 -1 0 51.55)"><g transform="matrix(1 0 0 1 0 8.61)"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#D3D3D3;" fill="#D3D3D3" stroke="#000000" transform="matrix(1 0 0 -1 0 0) translate(4.43,0)"><path d="M -4.15 -2.77 h 8.3 v 11.07 h -8.3 Z"></path></g><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(1 0 0 -1 8.86 0) translate(-0.28,0) matrix(1.0 0.0 0.0 1.0 3.04 -3.77)"><foreignObject style="--ltx-fo-width:5.66em;--ltx-fo-height:0.69em;--ltx-fo-depth:0.19em;font-size:10pt;" height="12.3" overflow="visible" transform="matrix(1 0 0 -1 0 9.61)" width="78.33"><span id="S4.F3.pic1.31">Base planner</span></foreignObject></g></g> <g transform="matrix(1 0 0 1 0 25.83)"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#4D4D4D;" fill="#4D4D4D" stroke="#000000" transform="matrix(1 0 0 -1 0 0) translate(4.43,0)"><path d="M -4.15 -2.77 h 8.3 v 11.07 h -8.3 Z"></path></g><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(1 0 0 -1 8.86 0) translate(-0.28,0) matrix(1.0 0.0 0.0 1.0 3.04 -3.77)"><foreignObject style="--ltx-fo-width:7.14em;--ltx-fo-height:0.69em;--ltx-fo-depth:0em;font-size:10pt;" height="9.61" overflow="visible" transform="matrix(1 0 0 -1 0 9.61)" width="98.86"><span id="S4.F3.pic1.32">Scorer backbone</span></foreignObject></g></g> <g transform="matrix(1 0 0 1 0 43.02)"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#2E7DE8;" fill="#2E7DE8" stroke="#000000" transform="matrix(1 0 0 -1 0 0) translate(4.43,0)"><path d="M -4.15 -2.77 h 8.3 v 11.07 h -8.3 Z"></path></g><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(1 0 0 -1 8.86 0) translate(-0.28,0) matrix(1.0 0.0 0.0 1.0 3.04 -3.69)"><foreignObject style="--ltx-fo-width:3.01em;--ltx-fo-height:0.68em;--ltx-fo-depth:0em;font-size:10pt;" height="9.46" overflow="visible" transform="matrix(1 0 0 -1 0 9.46)" width="41.7"><span id="S4.F3.pic1.33">TOAD</span></foreignObject></g></g></g></g></g></g></g></svg>

Figure 3: Per-sample inference cost across methods. The bar length is the total inference time per sample, decomposed into the base planner (light gray), the scorer backbone (dark gray), and our test-time optimization (TOAD, blue).

Figure 3reports the per-sample (batchsize=1) inference cost of each base planner + TOAD, measured on a single NVIDIA A100. We splut the execution time in three parts: base planner, scorer backbone, and search itself. The scorer’s perception backbone (a ViT-S encoder, around 100 ms) is computed once per frame; the search then reuses these cached features and re-runs only the lightweight scoring head on each sampled trajectory. The search itself (blue) therefore adds only 1.9 ($K=5$) to 20.4 ms ($K=100$), against total inference times of 100 to 780 ms.

The search cannot avoid running the scorer’s perception backbone in the first place. In DrivoR the scorer is part of the base planner, sharing a single backbone: scoring reuses the features the planner already extracted, and the search is close to free (1.9 ms on top of DrivoR’s forward pass). When the scorer is external, it needs its own perception, redundant with the planner’s, so features are extracted twice. This is the roughly 100 ms scorer backbone carried by other planners in Figure 3. The lesson is to build well-generalizing scorers directly into the planner, then strong test-time search comes essentially for free.

## 5 Ablation Study

TOAD component ablation. We ablate the different components in TOAD with iPad [^13]. $\bullet~$ Warm-start. We compare the warm start initialization against a normal initialization, with a mean set to zero and the standard deviation to one. $\bullet~$ Bicycle model. As an alternative of smoothed control space, TOAD can also operate directly in the trajectory space. In this setting (traj. space), we directly sample trajectories consisting of 8 future $(x,y)$ locations in bird’s-eye view coordinates. $\bullet~$ Reward terms. We study the importance of different reward terms: scorer reward, comfort reward and anchor reward, by comparing them against both the vanilla baseline without TOAD and the full configuration.

In 4(a), first observation is that with TOAD, the overall performance is improved compared to vanilla planner. We observe that the initialization from a standard normal distribution still achieves competitive performance (42.7 vs. 48.5), suggesting that TOAD can generalize to a broader class of E2E planners that output only a single trajectory without associated scores.

Another important observation is that the scorer reward provides the primary optimization signal, leading to a substantial performance gain (48.5 vs. 35.7). In contrast, the comfort and anchor rewards provide complementary improvements (48.5 vs. 46.9, 44.4). Finally, optimization in the control space consistently outperforms optimization in trajectory space. We hypothesize that this is because the control space (e.g., acceleration and yaw rate) is naturally bounded, whereas trajectory coordinates are unbounded and therefore more difficult to optimize effectively.

Number of candidates $M$, iterations $K$, and elites $E$. Overall, TOAD consistently outperforms the vanilla baseline across different hyperparameter settings, demonstrating its robustness and low sensitivity to hyperparameter choices. As shown in 4(b), performance generally improves as the number of sampled candidates increases. Interestingly, sampling only two control sequences already yields an improvement over the vanilla baseline without TOAD. To maintain a low inference overhead, we cap the number of candidates at 64, resulting in less than 1.9 ms runtime per execution.

In 4(c), we observe that the warm-start initialization enables rapid convergence, with iPad reaching peak performance after 5 iterations. The performance degradation observed with additional iterations highlights the imperfect correlation between the learned scorer and the true EPDMS objective. For instance, the scorer does not explicitly model extended comfort metrics that require temporal consistency across adjacent frames. Finally, 4(d) shows that selecting 8 elites empirically provides the best performance under a candidate budget of 64, corresponding to approximately one-eighth of the sampled candidates. This result offers a practical guideline for choosing CEM hyperparameters.

(a) TOAD Component ablation

| Method iPad | warm-start | bicycle | scorer | comfort | anchor | EPDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| Vanilla iPad |  |  |  |  |  | 34.7 |
| normal init. |  | ✓ | ✓ | ✓ | ✓ | 42.7 |
| traj. space | ✓ |  | ✓ | ✓ | ✓ | 36.4 |
| w/o scorer reward | ✓ | ✓ |  | ✓ | ✓ | 35.7 |
| w/o comfort reward | ✓ | ✓ | ✓ |  | ✓ | 46.9 |
| w/o anchor reward | ✓ | ✓ | ✓ | ✓ |  | 44.4 |
| TOAD | ✓ | ✓ | ✓ | ✓ | ✓ | 48.5 |

(b) Per-iteration candidate $M$. $K=5$, $E=\max(1,M/8)$.

(c) CEM iterations $K$. $M=64$, $E=8$.

(d) Number of elites $E$. $M=64$ and ${K=5}$.

Figure 4: Ablation study of TOAD with iPad on warmup-two-stage.

## 6 Conclusion

Trajectory scorers in end-to-end planning are learned reward functions, yet only used to rank a fixed candidate set. We show that this set, not the scorer, is the real bottleneck. TOAD, a simple CEM loop that searches against the frozen scorer, improves six base planners across NAVSIM-v1, NAVSIM-v2, and HUGSIM, at negligible cost and with no retraining. CEM can only succeed if the scorer stays accurate off the proposal distribution, which most current scorers do not. This points to a clear direction: planners whose scorer is built to generalize, so that strong test-time search comes for free.

Limitations. TOAD warm-starts from a base planner’s proposals, whereas a sufficiently accurate scorer and optimizer should in principle allow searching from scratch. This raises the question of whether scaling the scorer with more data and capacity could eventually remove the need for a base planner altogether. A second limitation is the reliance on DrivoR’s supervised disentangled scorer. A promising direction for future work is to reduce the reliance on costly trajectory-level annotations needed to train such a scorer, for instance through self-supervised or world model-based rewards.

## Acknowledgments

We thank Renaud Marlet for constant support throughout the project as well as Marc Lafon and Andrei Bursuc for helpful discussions. This work was granted access to the HPC resources of IDRIS under the allocations A0201016239, A0181016239 and A0181016203 made by GENCI. We acknowledge chair VISA DEEP (ANR-20- CHIA-0022), Cluster PostGenAI@Paris (ANR-23-IACL-0007, FRANCE 2030), and EuroHPC Joint Undertaking for awarding the project ID EU2025R01-032 access to Karolina, Czech Republic.

## References

## Appendix A Driving Metrics

In the study, we use the official metrics as defined by their original benchmarks.

#### NAVSIM-v1.

We use the Predictive Driver Model Score (PDMS) which aggregates submetrics: No-at-fault Collisions (NC), Drivable Area Compliance (DAC), Time to Collision (TTC), Ego progress (EP), Ego Comfort (Comf.). Details of subscore descriptions and aggregation strategy can be found in the original benchmark paper.

#### NAVSIM-v2.

We use the Extended Predictive Driver Model Score (EPDMS), computed over the two stages of the evaluation. Compared to PDMS, the EPDMS includes additional Driving direction Compliance (DDC), Traffic-line compliance (TLC) and Lane Keeping (LK), and replaces the ego comfort metric by the History Comfort (HC) and the Extended Comfort (EC).

#### HUGSIM.

We use the official HUGSIM Driving Score (HD-SCore or HDS), which mimic the NAVSIM-v1/2 metrics. It uses five sub-scores: no collisions (NC), drivable area compliance (DAC), time-to-collision (TTC), comfort (COM) and Route Completion (RC) which replaces the ego progress by the percentage of route covered by the agent in the current scenario.

## Appendix B Evolution of Trajectories during CEM

In Figure 5, we illustrate the increase in reward and the corresponding improvement in the EPDMS score throughout optimization. We also observe that the mean elite trajectory distribution is progressively refined, shifting toward the right lane and exhibiting better forward progress. This behavior differs from both the vanilla and human trajectories.

![[iter_0000.png|Refer to caption]]

iteration 0

### B.1 Detailed Results in NAVSIM Benchmarks

Table 5: NAVSIM-v1. Comparison to existing camera-only methods on the NAVSIM-v1 benchmark on test set (navtest). Scores are higher is better.

| Method |  | NC | DAC | TTC | Comf. | EP | PDMS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PDM‑Closed | [^8] | 94.6 | 99.8 | 89.9 | 86.9 | 99.9 | 89.1 |
| Human driver | [^9] | 100 | 100 | 100 | 99.9 | 87.5 | 94.8 |
| Ego‑stat. MLP | [^9] | 93.0 | 77.3 | 83.6 | 100 | 62.8 | 65.6 |
| UniVLA | [^47] | 96.9 | 91.1 | 91.7 | 96.7 | 76.8 | 81.7 |
| DrivingGPT | [^6] | 98.9 | 90.7 | 94.9 | 95.6 | 79.7 | 82.4 |
| UniAD | [^16] | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| LTF | [^7] | 97.4 | 92.8 | 92.4 | 100 | 79.0 | 83.8 |
| PARA‑Drive | [^48] | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 |
| DriveX-S | [^39] | 97.5 | 94.0 | 93.0 | 100 | 79.7 | 84.5 |
| World4Drive | [^59] | 97.4 | 94.3 | 92.8 | 100 | 79.9 | 85.1 |
| DRAMA | [^55] | 98.0 | 93.1 | 94.8 | 100 | 80.1 | 85.5 |
| VAD-v2 | [^5] | 98.1 | 94.8 | 94.3 | 100 | 80.6 | 86.2 |
| PRIX | [^50] | 98.1 | 96.3 | 94.1 | 100 | 82.3 | 87.8 |
| DiffusionDrive | [^33] | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| DIVER | [^41] | 98.5 | 96.5 | 94.9 | 100 | 82.6 | 88.3 |
| AutoVLA | [^61] | 98.4 | 95.6 | 98.0 | 99.9 | 81.9 | 89.1 |
| DriveVLA-W0 | [^27] | 98.7 | 99.1 | 95.3 | 99.3 | 83.3 | 90.2 |
| ReCogDrive | [^29] | 97.9 | 97.3 | 94.9 | 100 | 87.3 | 90.8 |
| Hydra-MDP++ | [^26] | 98.6 | 98.6 | 95.1 | 100 | 85.7 | 91.0 |
| Centaur | [^40] | 99.5 | 98.9 | 98.0 | 100 | 85.9 | 92.6 |
| DriveSuprim | [^53] | 98.6 | 98.6 | 95.5 | 100 | 91.3 | 93.5 |
| ZTRS (V2-99) | [^31] | 98.8 | 99.5 | 95.1 | 100 | 74.7 | 86.9 |
| \+ TOAD |  | 98.4 | 98.6 | 94.3 | 100 | 82.1 | 89.0 ($\uparrow$ 2.4%) |
| GTRS (V2-99) | [^32] | 99.0 | 98.9 | 95.9 | 100 | 82.8 | 90.4 |
| \+ TOAD |  | 98.9 | 98.1 | 96.3 | 100 | 84.8 | 90.9 ($\uparrow$ 0.6%) |
| Hydra-MDP | [^30] | 98.5 | 98.7 | 94.9 | 100 | 85.5 | 90.9 |
| \+ TOAD |  | 98.7 | 98.0 | 95.4 | 100 | 87.2 | 91.4 ($\uparrow$ 0.6%) |
| iPad | [^13] | 98.6 | 98.3 | 94.9 | 100 | 88.0 | 91.7 |
| \+ TOAD |  | 99.0 | 99.3 | 96.4 | 100 | 89.1 | 93.4 ($\uparrow$ 1.9%) |
| RAP-DINO | [^11] | 99.1 | 98.9 | 96.7 | 100 | 90.3 | 93.8 |
| \+ TOAD |  | 99.0 | 99.2 | 96.6 | 100 | 90.1 | 93.9 ($\uparrow$ 0.1%) |
| DrivoR | [^23] | 99.1 | 99.2 | 96.9 | 100 | 91.6 | 94.6 |
| \+ TOAD |  | 99.0 | 99.3 | 96.8 | 100 | 91.8 | 94.7 ($\uparrow$ 0.1%) |

#### NAVSIM-v1 detailed results.

In Table 5 we report PDMS and its constituent sub-metrics on navtest. Table 5 shows that the safety columns (NC, DAC, TTC) are already saturated for every base planner. Thus, TOAD improves PDMS on the different base planner by boosting the ego progress sub-metric (EP), while maintaining on-par performance on the safety sub-metrics. Leading to an increase in PDMS for all 6 base planners.

Table 6: NAVSIM-v2 navhard-two-stage. Comparison to existing methods on the NAVSIM-v2 benchmark test set using the EPDMS [^4].

<table><thead><tr><th></th><th></th><th colspan="9">Stage 1</th><th colspan="9">Stage 2</th><th></th></tr><tr><th>Method</th><th></th><th>NC</th><th>DAC</th><th>DDC</th><th>TLC</th><th>EP</th><th>TTC</th><th>LK</th><th>HC</th><th>EC</th><th>NC</th><th>DAC</th><th>DDC</th><th>TLC</th><th>EP</th><th>TTC</th><th>LK</th><th>HC</th><th>EC</th><th>EPDMS</th></tr><tr><th>PDM‑Closed</th><th><sup><a href="#fn:8">8</a></sup></th><th>94.4</th><th>98.8</th><th>100</th><th>99.5</th><th>100</th><th>93.5</th><th>99.3</th><th>87.7</th><th>36.0</th><th>90.5</th><th>90.6</th><th>95.4</th><th>98.4</th><th>100</th><th>86.6</th><th>74.2</th><th>91.9</th><th>29.7</th><th>56.6</th></tr></thead><tbody><tr><td>TransFuser</td><td><sup><a href="#fn:7">7</a></sup></td><td>96.2</td><td>79.5</td><td>99.1</td><td>99.5</td><td>84.1</td><td>95.1</td><td>94.2</td><td>97.5</td><td>79.1</td><td>77.7</td><td>70.2</td><td>84.2</td><td>98.0</td><td>85.1</td><td>75.6</td><td>45.4</td><td>95.7</td><td>75.9</td><td>23.1</td></tr><tr><td>DiffusionDrive</td><td><sup><a href="#fn:33">33</a></sup></td><td>96.0</td><td>79.7</td><td>97.4</td><td>99.5</td><td>81.3</td><td>93.1</td><td>90.8</td><td>96.8</td><td>73.8</td><td>82.1</td><td>72.2</td><td>88.5</td><td>98.7</td><td>85.1</td><td>78.8</td><td>49.2</td><td>89.3</td><td>71.2</td><td>24.2</td></tr><tr><td>GuideFlow</td><td><sup><a href="#fn:35">35</a></sup></td><td>96.6</td><td>80.5</td><td>96.3</td><td>99.3</td><td>82.3</td><td>94.9</td><td>91.5</td><td>97.7</td><td>67.8</td><td>87.3</td><td>76.7</td><td>88.8</td><td>99.2</td><td>84.3</td><td>85.1</td><td>49.7</td><td>93.1</td><td>44.5</td><td>27.1</td></tr><tr><td>Senna-E2E</td><td><sup><a href="#fn:20">20</a></sup></td><td>95.6</td><td>86.0</td><td>98.9</td><td>99.6</td><td>83.9</td><td>95.1</td><td>95.3</td><td>97.6</td><td>75.6</td><td>78.6</td><td>74.8</td><td>84.8</td><td>98.2</td><td>88.2</td><td>75.7</td><td>46.9</td><td>96.0</td><td>65.8</td><td>27.2</td></tr><tr><td>MindDrive</td><td><sup><a href="#fn:42">42</a></sup></td><td>96.1</td><td>86.0</td><td>98.8</td><td>99.3</td><td>83.3</td><td>95.6</td><td>94.4</td><td>97.6</td><td>74.7</td><td>82.6</td><td>79.1</td><td>86.4</td><td>98.0</td><td>85.3</td><td>79.4</td><td>49.2</td><td>96.5</td><td>71.0</td><td>30.9</td></tr><tr><td>World4Drive</td><td><sup><a href="#fn:59">59</a></sup></td><td>97.3</td><td>89.1</td><td>97.6</td><td>99.7</td><td>60.5</td><td>96.8</td><td>87.7</td><td>93.1</td><td>60.0</td><td>91.4</td><td>82.0</td><td>91.0</td><td>98.5</td><td>53.1</td><td>90.6</td><td>52.3</td><td>93.3</td><td>62.8</td><td>34.9</td></tr><tr><td>SpanVLA</td><td><sup><a href="#fn:62">62</a></sup></td><td>98.4</td><td>94.3</td><td>97.8</td><td>99.9</td><td>85.7</td><td>97.2</td><td>94.2</td><td>97.6</td><td>72.1</td><td>86.9</td><td>84.3</td><td>87.1</td><td>98.2</td><td>85.5</td><td>82.7</td><td>62.3</td><td>96.8</td><td>67.4</td><td>40.1</td></tr><tr><td>DriveSuprim</td><td><sup><a href="#fn:53">53</a></sup></td><td>98.9</td><td>95.1</td><td>99.2</td><td>99.6</td><td>76.1</td><td>99.1</td><td>94.7</td><td>97.6</td><td>54.2</td><td>87.9</td><td>88.8</td><td>89.6</td><td>98.8</td><td>80.3</td><td>86.0</td><td>53.5</td><td>97.1</td><td>56.1</td><td>42.1</td></tr><tr><td>DiffVLA</td><td><sup><a href="#fn:19">19</a></sup></td><td>95.7</td><td>99.2</td><td>100.0</td><td>100.0</td><td>85.9</td><td>96.4</td><td>97.1</td><td>95.0</td><td>84.2</td><td>81.2</td><td>88.8</td><td>94.6</td><td>99.0</td><td>86.0</td><td>76.4</td><td>59.8</td><td>98.6</td><td>80.4</td><td>45.0</td></tr><tr><td>GTRS-E</td><td><sup><a href="#fn:32">32</a></sup></td><td>98.9</td><td>99.3</td><td>99.8</td><td>99.8</td><td>75.2</td><td>98.4</td><td>96.0</td><td>97.6</td><td>51.6</td><td>92.3</td><td>93.3</td><td>94.6</td><td>99.2</td><td>73.1</td><td>91.2</td><td>53.9</td><td>96.7</td><td>56.8</td><td>49.4</td></tr><tr><td>SimScale</td><td><sup><a href="#fn:46">46</a></sup></td><td>99.6</td><td>99.1</td><td>99.9</td><td>100.0</td><td>69.6</td><td>99.6</td><td>95.8</td><td>95.6</td><td>28.4</td><td>94.5</td><td>94.2</td><td>95.8</td><td>99.2</td><td>75.8</td><td>92.8</td><td>60.1</td><td>96.1</td><td>43.2</td><td>53.2</td></tr><tr><td>DriveFuture</td><td><sup><a href="#fn:15">15</a></sup></td><td>99.8</td><td>99.8</td><td>100.0</td><td>99.6</td><td>85.7</td><td>99.8</td><td>98.7</td><td>97.6</td><td>66.2</td><td>90.6</td><td>87.5</td><td>94.1</td><td>99.1</td><td>84.6</td><td>88.8</td><td>58.3</td><td>93.5</td><td>45.6</td><td>55.5</td></tr><tr><td>iPad</td><td><sup><a href="#fn:13">13</a></sup></td><td>96.2</td><td>94.9</td><td>96.8</td><td>99.6</td><td>82.0</td><td>95.8</td><td>94.2</td><td>97.8</td><td>42.2</td><td>83.6</td><td>83.2</td><td>84.8</td><td>97.7</td><td>85.4</td><td>79.7</td><td>46.8</td><td>95.2</td><td>38.9</td><td>34.7</td></tr><tr><td>+ TOAD</td><td></td><td>97.7</td><td>97.1</td><td>99.6</td><td>99.8</td><td>75.6</td><td>98.4</td><td>93.6</td><td>97.6</td><td>43.1</td><td>93.0</td><td>91.0</td><td>93.1</td><td>98.8</td><td>78.4</td><td>91.2</td><td>54.8</td><td>97.6</td><td>41.1</td><td>49.8 (<math><semantics><mo>↑</mo> <annotation>{\uparrow}</annotation></semantics></math> 43.6%)</td></tr><tr><td>RAP-DINO (ViT-H)</td><td><sup><a href="#fn:11">11</a></sup></td><td>97.1</td><td>94.4</td><td>98.8</td><td>99.8</td><td>83.9</td><td>96.9</td><td>94.7</td><td>96.4</td><td>66.2</td><td>83.2</td><td>83.9</td><td>87.4</td><td>98.0</td><td>86.9</td><td>80.4</td><td>52.3</td><td>95.2</td><td>52.4</td><td>39.6</td></tr><tr><td>+ TOAD</td><td></td><td>97.1</td><td>95.3</td><td>99.4</td><td>100.0</td><td>80.3</td><td>97.8</td><td>92.9</td><td>97.6</td><td>64.4</td><td>89.9</td><td>89.7</td><td>92.5</td><td>98.2</td><td>81.4</td><td>87.7</td><td>52.1</td><td>97.2</td><td>51.1</td><td>49.0 (<math><semantics><mo>↑</mo> <annotation>{\uparrow}</annotation></semantics></math> 23.9%)</td></tr><tr><td>Hydra-MDP</td><td><sup><a href="#fn:30">30</a></sup></td><td>98.4</td><td>94.2</td><td>99.6</td><td>99.8</td><td>82.7</td><td>98.0</td><td>95.8</td><td>97.8</td><td>66.2</td><td>85.2</td><td>82.5</td><td>89.7</td><td>98.5</td><td>85.2</td><td>82.5</td><td>50.7</td><td>95.4</td><td>47.7</td><td>40.9</td></tr><tr><td>+ TOAD</td><td></td><td>98.1</td><td>95.1</td><td>99.3</td><td>100.0</td><td>78.0</td><td>98.2</td><td>89.1</td><td>97.8</td><td>55.1</td><td>91.4</td><td>90.1</td><td>94.3</td><td>98.5</td><td>80.1</td><td>88.9</td><td>55.7</td><td>97.6</td><td>45.2</td><td>49.7 (<math><semantics><mo>↑</mo> <annotation>{\uparrow}</annotation></semantics></math> 21.6%)</td></tr><tr><td>GTRS (V2-99)</td><td><sup><a href="#fn:32">32</a></sup></td><td>98.9</td><td>95.1</td><td>99.1</td><td>99.6</td><td>76.2</td><td>99.1</td><td>94.9</td><td>97.6</td><td>54.2</td><td>88.1</td><td>88.8</td><td>89.3</td><td>98.9</td><td>98.9</td><td>85.9</td><td>53.7</td><td>96.8</td><td>56.9</td><td>45.4</td></tr><tr><td>+ TOAD</td><td></td><td>98.7</td><td>96.4</td><td>99.2</td><td>99.8</td><td>74.6</td><td>98.7</td><td>89.1</td><td>97.6</td><td>60.9</td><td>91.2</td><td>91.5</td><td>95.1</td><td>98.5</td><td>75.7</td><td>89.2</td><td>57.8</td><td>98.5</td><td>54.3</td><td>51.7 (<math><semantics><mo>↑</mo> <annotation>{\uparrow}</annotation></semantics></math> 13.8%)</td></tr><tr><td>ZTRS (V2-99)</td><td><sup><a href="#fn:31">31</a></sup></td><td>98.9</td><td>97.6</td><td>100</td><td>100</td><td>66.7</td><td>98.9</td><td>96.2</td><td>96.7</td><td>44.0</td><td>91.1</td><td>90.4</td><td>95.8</td><td>99.0</td><td>63.6</td><td>89.8</td><td>60.4</td><td>97.6</td><td>66.1</td><td>48.1</td></tr><tr><td>+ TOAD</td><td></td><td>98.7</td><td>96.9</td><td>99.6</td><td>99.6</td><td>71.0</td><td>98.4</td><td>90.7</td><td>97.6</td><td>62.2</td><td>92.3</td><td>90.2</td><td>94.0</td><td>98.5</td><td>67.8</td><td>90.4</td><td>58.8</td><td>99.2</td><td>57.3</td><td>49.2 (<math><semantics><mo>↑</mo> <annotation>{\uparrow}</annotation></semantics></math> 2.3%)</td></tr><tr><td>DrivoR (ViT-S)</td><td><sup><a href="#fn:23">23</a></sup></td><td>99.1</td><td>98.2</td><td>99.3</td><td>99.8</td><td>75.4</td><td>98.7</td><td>94.9</td><td>97.6</td><td>70.2</td><td>92.3</td><td>91.6</td><td>97.3</td><td>99.1</td><td>75.7</td><td>90.6</td><td>56.1</td><td>98.4</td><td>44.7</td><td>54.6</td></tr><tr><td>+ TOAD</td><td></td><td>98.9</td><td>98.2</td><td>99.4</td><td>100.0</td><td>74.6</td><td>98.9</td><td>94.0</td><td>97.6</td><td>66.7</td><td>93.6</td><td>94.0</td><td>97.5</td><td>98.8</td><td>74.1</td><td>92.0</td><td>57.1</td><td>98.0</td><td>47.3</td><td>56.3 (<math><semantics><mo>↑</mo> <annotation>{\uparrow}</annotation></semantics></math> 3.1%)</td></tr></tbody></table>

#### NAVSIM-v2 detailed results.

In Table 6 we report EPDMS and its Stage-1 and Stage-2 sub-metrics on navhard-two-stage. Table 6 shows that, unlike on navtest, the safety columns (NC, DAC, TTC) still leave headroom on these harder out-of-distribution scenes. Thus, TOAD improves EPDMS on the different base planners by raising the safety sub-metrics while trading off a small amount of ego progress (EP), with the shift concentrated in the more demanding Stage 2. Leading to an increase in EPDMS for all 6 base planners.

Table 7: What drives the gain? (i) trajectory-smoothing, (ii) re-scoring the original proposals with DrivoR scorer [^23], GTRS scorer [^32] and SparseDriveV2 [^45], and (iii) using different scorers as rewards — on top of iPad [^13] and Hydra-MDP [^30]. Evaluation on NAVSIM-v2 with EPDMS [^4]; larger is better.

<table><thead><tr><th></th><th></th><th colspan="9">Stage 1</th><th colspan="9">Stage 2</th><th></th></tr><tr><th>Method</th><th></th><th>NC</th><th>DAC</th><th>DDC</th><th>TLC</th><th>EP</th><th>TTC</th><th>LK</th><th>HC</th><th>EC</th><th>NC</th><th>DAC</th><th>DDC</th><th>TLC</th><th>EP</th><th>TTC</th><th>LK</th><th>HC</th><th>EC</th><th>EPDMS</th></tr></thead><tbody><tr><td>iPad</td><td><sup><a href="#fn:13">13</a></sup></td><td>96.2</td><td>94.9</td><td>96.8</td><td>99.6</td><td>82.0</td><td>95.8</td><td>94.2</td><td>97.8</td><td>42.2</td><td>83.6</td><td>83.2</td><td>84.8</td><td>97.7</td><td>85.4</td><td>79.7</td><td>46.8</td><td>95.2</td><td>38.9</td><td>34.7</td></tr><tr><td>+ Smoothing (BM)</td><td></td><td>96.0</td><td>94.0</td><td>96.6</td><td>99.3</td><td>82.4</td><td>95.6</td><td>94.4</td><td>97.8</td><td>60.9</td><td>81.6</td><td>80.2</td><td>85.6</td><td>97.9</td><td>85.7</td><td>78.3</td><td>46.7</td><td>97.2</td><td>56.8</td><td>35.0 (<math><semantics><mo>↑</mo> <annotation>{\uparrow}</annotation></semantics></math> 0.8%)</td></tr><tr><td>+ Re-scoring w/ DrivoR</td><td></td><td>98.7</td><td>97.8</td><td>99.9</td><td>99.6</td><td>75.2</td><td>98.7</td><td>94.4</td><td>88.9</td><td>44.0</td><td>90.2</td><td>87.4</td><td>92.8</td><td>98.7</td><td>74.0</td><td>88.1</td><td>53.2</td><td>91.9</td><td>43.6</td><td>45.6 (<math><semantics><mo>↑</mo> <annotation>{\uparrow}</annotation></semantics></math> 31.5%)</td></tr><tr><td>+ Re-scoring w/ GTRS</td><td></td><td>95.3</td><td>87.3</td><td>93.8</td><td>99.6</td><td>69.9</td><td>95.6</td><td>90.2</td><td>89.6</td><td>34.2</td><td>90.4</td><td>83.6</td><td>92.6</td><td>98.6</td><td>56.4</td><td>87.9</td><td>54.7</td><td>95.5</td><td>59.1</td><td>34.5 (<math><semantics><mo>↓</mo> <annotation>{\downarrow}</annotation></semantics></math> 0.6%)</td></tr><tr><td>+ Re-scoring w/ SparseDriveV2</td><td></td><td>96.7</td><td>85.1</td><td>98.7</td><td>100.0</td><td>79.4</td><td>96.4</td><td>95.6</td><td>92.0</td><td>36.9</td><td>85.0</td><td>76.1</td><td>87.1</td><td>98.5</td><td>77.2</td><td>84.1</td><td>51.9</td><td>87.5</td><td>37.3</td><td>30.3 (<math><semantics><mo>↓</mo> <annotation>{\downarrow}</annotation></semantics></math> 12.5%)</td></tr><tr><td>+ Search w/ GTRS-Scorer</td><td></td><td>93.3</td><td>77.8</td><td>89.8</td><td>99.8</td><td>59.9</td><td>94.0</td><td>79.1</td><td>94.2</td><td>15.6</td><td>88.1</td><td>75.6</td><td>91.2</td><td>98.2</td><td>64.3</td><td>87.0</td><td>57.0</td><td>96.4</td><td>30.6</td><td>23.9 (<math><semantics><mrow><mo>↓</mo> <mrow><mn>31.1</mn> <mo>%</mo></mrow></mrow> <annotation>{\downarrow}31.1\%</annotation></semantics></math>)</td></tr><tr><td>+ Search w/ SparseDriveV2 scorer</td><td></td><td>96.9</td><td>86.9</td><td>96.7</td><td>100.0</td><td>74.7</td><td>96.9</td><td>86.0</td><td>97.6</td><td>32.0</td><td>90.6</td><td>84.5</td><td>88.6</td><td>98.4</td><td>75.1</td><td>88.2</td><td>50.5</td><td>97.4</td><td>48.7</td><td>34.6 (<math><semantics><mrow><mo>↓</mo> <mrow><mn>0.4</mn> <mo>%</mo></mrow></mrow> <annotation>{\downarrow}\phantom{0}0.4\%</annotation></semantics></math>)</td></tr><tr><td>+ TOAD (search w/ DrivoR scorer)</td><td></td><td>97.7</td><td>97.1</td><td>99.6</td><td>99.8</td><td>75.6</td><td>98.4</td><td>93.6</td><td>97.6</td><td>43.1</td><td>93.0</td><td>91.0</td><td>93.1</td><td>98.8</td><td>78.4</td><td>91.2</td><td>54.8</td><td>97.6</td><td>41.1</td><td>49.8 (<math><semantics><mo>↑</mo> <annotation>{\uparrow}</annotation></semantics></math> 43.6%)</td></tr><tr><td>Hydra-MDP</td><td><sup><a href="#fn:30">30</a></sup></td><td>98.4</td><td>94.2</td><td>99.6</td><td>99.8</td><td>82.7</td><td>98.0</td><td>95.8</td><td>97.8</td><td>66.2</td><td>85.2</td><td>82.5</td><td>89.7</td><td>98.5</td><td>85.2</td><td>82.5</td><td>50.7</td><td>95.4</td><td>47.7</td><td>40.9</td></tr><tr><td>+ Smoothing (BM)</td><td></td><td>98.2</td><td>92.4</td><td>99.3</td><td>100.0</td><td>83.0</td><td>97.8</td><td>96.0</td><td>97.8</td><td>65.8</td><td>84.4</td><td>80.9</td><td>88.2</td><td>98.3</td><td>86.3</td><td>80.7</td><td>51.1</td><td>95.4</td><td>50.7</td><td>38.8 (<math><semantics><mo>↓</mo> <annotation>{\downarrow}</annotation></semantics></math> 5.1%)</td></tr><tr><td>+ Re-scoring w/ DrivoR</td><td></td><td>98.2</td><td>90.7</td><td>99.1</td><td>99.8</td><td>64.0</td><td>97.6</td><td>90.0</td><td>74.9</td><td>8.0</td><td>94.3</td><td>90.8</td><td>96.6</td><td>99.3</td><td>65.1</td><td>92.5</td><td>55.9</td><td>74.8</td><td>18.2</td><td>37.3 (<math><semantics><mo>↓</mo> <annotation>{\downarrow}</annotation></semantics></math> 8.8%)</td></tr><tr><td>+ Re-scoring w/ GTRS</td><td></td><td>97.3</td><td>98.7</td><td>99.9</td><td>99.6</td><td>63.5</td><td>97.1</td><td>92.7</td><td>85.1</td><td>15.1</td><td>91.9</td><td>92.8</td><td>95.1</td><td>98.8</td><td>65.1</td><td>90.8</td><td>58.1</td><td>79.5</td><td>18.4</td><td>40.3 (<math><semantics><mo>↓</mo> <annotation>{\downarrow}</annotation></semantics></math> 1.5%)</td></tr><tr><td>+ Re-scoring w/ SparseDriveV2</td><td></td><td>90.7</td><td>52.9</td><td>87.2</td><td>99.6</td><td>76.9</td><td>88.7</td><td>81.3</td><td>96.4</td><td>14.7</td><td>81.9</td><td>51.2</td><td>82.1</td><td>98.5</td><td>78.1</td><td>79.4</td><td>52.0</td><td>96.5</td><td>21.7</td><td>9.5 (<math><semantics><mo>↓</mo> <annotation>{\downarrow}</annotation></semantics></math> 76.9%)</td></tr><tr><td>+ Search w/ GTRS-Scorer</td><td></td><td>96.0</td><td>92.9</td><td>96.7</td><td>99.6</td><td>82.4</td><td>96.7</td><td>93.8</td><td>97.8</td><td>60.4</td><td>85.4</td><td>83.0</td><td>87.1</td><td>98.8</td><td>85.8</td><td>83.0</td><td>52.0</td><td>96.7</td><td>46.9</td><td>38.1 (<math><semantics><mo>↓</mo> <annotation>{\downarrow}</annotation></semantics></math> 6.9%)</td></tr><tr><td>+ Search w/ SparseDriveV2 scorer</td><td></td><td>95.9</td><td>78.7</td><td>95.0</td><td>100.0</td><td>78.9</td><td>95.1</td><td>85.8</td><td>97.6</td><td>52.4</td><td>87.6</td><td>77.3</td><td>85.6</td><td>98.9</td><td>83.0</td><td>84.0</td><td>47.8</td><td>96.9</td><td>51.8</td><td>29.6 (<math><semantics><mo>↓</mo> <annotation>{\downarrow}</annotation></semantics></math> 27.7%)</td></tr><tr><td>+ TOAD (search w/ DrivoR scorer)</td><td></td><td>98.1</td><td>95.1</td><td>99.3</td><td>100.0</td><td>78.0</td><td>98.2</td><td>89.1</td><td>97.8</td><td>55.1</td><td>91.4</td><td>90.1</td><td>94.3</td><td>98.5</td><td>80.1</td><td>88.9</td><td>55.7</td><td>97.6</td><td>45.2</td><td>49.7 (<math><semantics><mo>↑</mo> <annotation>{\uparrow}</annotation></semantics></math> 21.6%)</td></tr></tbody></table>

#### What drives the gain: detailed results.

In Table 7 we report the per-sub-metric breakdown of smoothing, re-scoring, and search with different scorers, on top of iPad and Hydra-MDP. Table 7 shows that smoothing leaves nearly every sub-metric unchanged, re-scoring moves individual sub-metrics inconsistently across the two planners, and search with fixed-vocabulary scorers (GTRS, SparseDriveV2) degrades several sub-metrics at once. Only TOAD with the DrivoR scorer raises the safety sub-metrics jointly without collapsing progress. Confirming that the gain arises from the synergy in TOAD between search and an off-proposal-accurate scorer, rather than from scoring alone.

[^1]: M. Bojarski, D. Del Testa, D. Dworakowski, B. Firner, B. Flepp, P. Goyal, L. D. Jackel, M. Monfort, U. Muller, J. Zhang, et al. (2016) End to end learning for self-driving cars. arXiv preprint arXiv:1604.07316. Cited by: §1.

[^2]: B. Brown, J. Juravsky, R. Ehrlich, R. Clark, Q. V. Le, C. Ré, and A. Mirhoseini (2024) Large language monkeys: scaling inference compute with repeated sampling. arXiv preprint arXiv:2407.21787. Cited by: §2.

[^3]: H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom (2020) NuScenes: A multimodal dataset for autonomous driving. In CVPR, Cited by: Table 4.

[^4]: W. Cao, M. Hallgarten, T. Li, D. Dauner, X. Gu, C. Wang, Y. Miron, M. Aiello, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, A. Geiger, and K. Chitta (2025) Pseudo-simulation for autonomous driving. In CoRL, Cited by: Appendix A, Table 6, Table 7, §1, §2, Table 3, §4.1, §4.2.

[^5]: S. Chen, B. Jiang, H. Gao, B. Liao, Q. Xu, Q. Zhang, C. Huang, W. Liu, and X. Wang (2024) Vadv2: end-to-end vectorized autonomous driving via probabilistic planning. arXiv preprint arXiv:2402.13243. Cited by: Table 5.

[^6]: Y. Chen, Y. Wang, and Z. Zhang (2025) Drivinggpt: unifying driving world modeling and planning with multi-modal autoregressive transformers. In ICCV, Cited by: Table 5.

[^7]: K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger (2022) Transfuser: imitation with transformer-based sensor fusion for autonomous driving. TPAMI. Cited by: Table 5, Table 6, Table 4.

[^8]: D. Dauner, M. Hallgarten, A. Geiger, and K. Chitta (2023) Parting with misconceptions about learning-based vehicle motion planning. In CoRL, Cited by: Table 5, Table 6, §1, Table 3, Table 3.

[^9]: D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, et al. (2024) Navsim: data-driven non-reactive autonomous vehicle simulation and benchmarking. In NeurIPS, Cited by: Appendix A, Table 5, Table 5, §1, §2, Table 3, §4.1, §4.2.

[^10]: P. Drews, G. Williams, B. Goldfain, E. A. Theodorou, and J. M. Rehg (2017) Aggressive deep driving: combining convolutional neural networks and model predictive control. In CoRL, Cited by: §2.

[^11]: L. Feng, Y. Gao, E. Zablocki, Q. Li, W. Li, S. Liu, M. Cord, and A. Alahi (2026) RAP: 3d rasterization augmented end-to-end planning. In ICLR, Cited by: Table 5, Table 6, §1, §1, §2, Table 3, Table 3, Figure 3, §4.1.

[^12]: D. Guo, D. Yang, H. Zhang, J. Song, P. Wang, Q. Zhu, R. Xu, R. Zhang, S. Ma, X. Bi, et al. (2025) Deepseek-r1: incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948. Cited by: §2.

[^13]: K. Guo, H. Liu, X. Wu, J. Pan, and C. Lv (2025) IPad: iterative proposal-centric end-to-end autonomous driving. arXiv preprint arXiv:2505.15111. Cited by: Table 5, Table 6, Table 7, Table 7, §1, §1, §2, Table 3, Table 3, Table 3, Table 3, Figure 3, §4.1, §4.3, §5.

[^14]: N. Hanselmann, K. Renz, K. Chitta, A. Bhattacharyya, and A. Geiger (2022) KING: generating safety-critical driving scenarios for robust imitation via kinematics gradients. In ECCV, Cited by: §3.2.

[^15]: Y. Hong, X. Zhou, Y. Li, X. Zhou, L. Liu, Y. Luo, S. Xu, L. Yang, and Z. Song (2026) DriveFuture: future-aware latent world models for autonomous driving. arXiv preprint arXiv:2605.09701. Cited by: Table 6, §4.2.

[^16]: Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, L. Lu, X. Jia, Q. Liu, J. Dai, Y. Qiao, and H. Li (2023) Planning-oriented autonomous driving. In CVPR, Cited by: Table 5, §1, §2, §2, Table 4.

[^17]: Z. Huang, H. Liu, J. Wu, and C. Lv (2024) Differentiable integrated motion prediction and planning with learnable cost function for autonomous driving. IEEE Trans. Neural Networks Learn. Syst.. Cited by: §2.

[^18]: A. Jaech, A. Kalai, A. Lerer, A. Richardson, A. El-Kishky, A. Low, A. Helyar, A. Madry, A. Beutel, A. Carney, et al. (2024) Openai o1 system card. arXiv preprint arXiv:2412.16720. Cited by: §2.

[^19]: A. Jiang, Y. Gao, Z. Sun, Y. Wang, J. Wang, J. Chai, Q. Cao, Y. Heng, H. Jiang, Y. Dong, et al. (2025) Diffvla: vision-language guided diffusion planning for autonomous driving. arXiv preprint arXiv:2505.19381. Cited by: Table 6.

[^20]: B. Jiang, S. Chen, B. Liao, X. Zhang, W. Yin, Q. Zhang, C. Huang, W. Liu, and X. Wang (2024) Senna: bridging large vision-language models and end-to-end autonomous driving. arXiv preprint arXiv:2410.22313. Cited by: Table 6.

[^21]: B. Jiang, S. Chen, Q. Xu, B. Liao, J. Chen, H. Zhou, Q. Zhang, W. Liu, C. Huang, and X. Wang (2023) Vad: vectorized scene representation for efficient autonomous driving. In ICCV, Cited by: §2, Table 4.

[^22]: B. Kerbl, G. Kopanas, T. Leimkühler, and G. Drettakis (2023) 3D gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics. Cited by: §4.1.

[^23]: E. Kirby, A. Boulch, Y. Xu, Y. Yin, G. Puy, E. Zablocki, A. Bursuc, S. Gidaris, R. Marlet, F. Bartoccioni, A. Cao, N. Samet, T. Vu, and M. Cord (2026) Driving on registers. In CVPR, Cited by: Table 5, Table 6, Table 7, §1, §1, §1, §2, §2, §3.2, Table 3, Table 3, Figure 2, Figure 3, §4.1, §4.3, Table 4.

[^24]: M. Kobilarov (2012) Cross-entropy motion planning. Int. J. Robotics Res.. Cited by: §2, §3.2.

[^25]: D. Li, C. Li, Y. Wang, J. Ren, X. Wen, P. Li, L. Xu, K. Zhan, P. Jia, X. Lang, et al. (2025) Learning personalized driving styles via reinforcement learning from human feedback. arXiv preprint arXiv:2503.10434. Cited by: §2.

[^26]: K. Li, Z. Li, S. Lan, Y. Xie, Z. Zhang, J. Liu, Z. Wu, Z. Yu, and J. M. Alvarez (2025) Hydra-mdp++: advancing end-to-end driving via expert-guided hydra-distillation. arXiv e-prints, pp. arXiv–2503. Cited by: Table 5.

[^27]: Y. Li, S. Shang, W. Liu, B. Zhan, H. Wang, Y. Wang, Y. Chen, X. Wang, Y. An, C. Tang, et al. (2025) DriveVLA-w0: world models amplify data scaling law in autonomous driving. arXiv preprint arXiv:2510.12796. Cited by: Table 5.

[^28]: Y. Li, Y. Wang, Y. Liu, J. He, L. Fan, and Z. Zhang (2025) End-to-end driving with online trajectory evaluation via BEV world model. In ICCV, Cited by: §2.

[^29]: Y. Li, K. Xiong, X. Guo, F. Li, S. Yan, G. Xu, L. Zhou, L. Chen, H. Sun, B. Wang, et al. (2026) ReCogDrive: a reinforced cognitive framework for end-to-end autonomous driving. In ICLR, Cited by: Table 5, §2.

[^30]: Z. Li, K. Li, S. Wang, S. Lan, Z. Yu, Y. Ji, Z. Li, Z. Zhu, J. Kautz, Z. Wu, et al. (2024) Hydra-mdp: end-to-end multimodal planning with multi-target hydra-distillation. arXiv preprint arXiv:2406.06978. Cited by: Figure 5, Table 5, Table 6, Table 7, Table 7, §1, §1, §1, §2, Table 3, Table 3, Table 3, Table 3, Figure 3, §4.1, §4.3.

[^31]: Z. Li, W. Yao, Z. Wang, X. Sun, J. Chen, N. Chang, M. Shen, J. Song, Z. Wu, S. Lan, et al. (2025) ZTRS: zero-imitation end-to-end autonomous driving with trajectory scoring. arXiv preprint arXiv:2510.24108. Cited by: Table 5, Table 6, §1, §1, §1, §2, Table 3, Table 3, Figure 3, §4.1.

[^32]: Z. Li, W. Yao, Z. Wang, X. Sun, J. Chen, N. Chang, M. Shen, Z. Wu, S. Lan, and J. M. Alvarez (2025) Generalized trajectory scoring for end-to-end multimodal planning. arXiv preprint arXiv:2506.06664. Cited by: Table 5, Table 6, Table 6, Table 7, §1, §1, §1, §2, Table 3, Table 3, Figure 3, §4.1, §4.3, Table 4.

[^33]: B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang, et al. (2025) Diffusiondrive: truncated diffusion model for end-to-end autonomous driving. In CVPR, Cited by: Table 5, Table 6, §2, §2.

[^34]: Y. Liao, J. Xie, and A. Geiger (2023) KITTI-360: A novel dataset and benchmarks for urban scene understanding in 2d and 3d. TPAMI. Cited by: Table 4.

[^35]: L. Liu, C. Jia, G. Yu, Z. Song, J. Li, F. Jia, P. Wu, X. Hao, and Y. Luo (2025) Guideflow: constraint-guided flow matching for planning in end-to-end autonomous driving. arXiv preprint arXiv:2511.18729. Cited by: Table 6.

[^36]: C. Pinneri, S. Sawant, S. Blaes, J. Achterhold, J. Stueckler, M. Rolínek, and G. Martius (2020) Sample-efficient cross-entropy method for real-time planning. In CoRL, Cited by: §2.

[^37]: K. Renz, L. Chen, E. Arani, and O. Sinavski (2025) Simlingo: vision-only closed-loop autonomous driving with language-action alignment. In CVPR, Cited by: §1.

[^38]: R. Y. Rubinstein and D. P. Kroese (2004) The cross-entropy method: a unified approach to combinatorial optimization, monte-carlo simulation, and machine learning. Springer. Cited by: §1, §2, §3.2.

[^39]: C. Shi, S. Shi, K. Sheng, B. Zhang, and L. Jiang (2025) DriveX: omni scene modeling for learning generalizable world knowledge in autonomous driving. In ICCV, Cited by: Table 5.

[^40]: C. Sima, K. Chitta, Z. Yu, S. Lan, P. Luo, A. Geiger, H. Li, and J. M. Alvarez (2025) Centaur: robust end-to-end autonomous driving with test-time training. arXiv preprint arXiv:2503.11650. Cited by: Table 5, §2.

[^41]: Z. Song, L. Liu, H. Pan, B. Liao, M. Guo, L. Yang, Y. Zhang, S. Xu, C. Jia, and Y. Luo (2025) DIVER: reinforced diffusion breaks imitation bottlenecks in end-to-end autonomous driving. arXiv preprint arXiv:2507.04049. Cited by: Table 5, §2.

[^42]: B. Sun, Y. Cao, Y. Wang, R. Wang, J. Shang, X. Feng, J. Lu, J. Shi, S. Yang, X. Yan, et al. (2025) MindDrive: an all-in-one framework bridging world models and vision-language model for end-to-end autonomous driving. arXiv preprint arXiv:2512.04441. Cited by: Table 6.

[^43]: J. Sun, F. Xue, T. Long, C. Liu, J. Hu, W. Zheng, and N. Sebe (2026) Risk-aware world model predictive control for generalizable end-to-end autonomous driving. arXiv preprint arXiv:2602.23259. Cited by: §2.

[^44]: P. Sun, H. Kretzschmar, X. Dotiwalla, A. Chouard, V. Patnaik, P. Tsui, J. Guo, Y. Zhou, Y. Chai, B. Caine, V. Vasudevan, W. Han, J. Ngiam, H. Zhao, A. Timofeev, S. Ettinger, M. Krivokon, A. Gao, A. Joshi, Y. Zhang, J. Shlens, Z. Chen, and D. Anguelov (2020) Scalability in perception for autonomous driving: waymo open dataset. In CVPR, Cited by: Table 4.

[^45]: W. Sun, X. Lin, K. Chen, Z. Pei, X. Li, Y. Shi, and S. Zheng (2026) SparseDriveV2: scoring is all you need for end-to-end autonomous driving. arXiv preprint arXiv:2603.29163. Cited by: Table 7, §1, §1, §2, §4.3.

[^46]: H. Tian, T. Li, H. Liu, J. Yang, Y. Qiu, G. Li, J. Wang, Y. Gao, Z. Zhang, L. Wang, H. Ye, T. Tan, L. Chen, and H. Li (2026) SimScale: learning to drive via real-world simulation at scale. In CVPR, Cited by: Table 6.

[^47]: Y. Wang, X. Li, W. Wang, J. Zhang, Y. Li, Y. Chen, X. Wang, and Z. Zhang (2025) Unified vision-language-action model. arXiv preprint arXiv:2506.19850. Cited by: Table 5.

[^48]: X. Weng, B. Ivanovic, Y. Wang, Y. Wang, and M. Pavone (2024) PARA-drive: parallelized architecture for real-time autonomous driving. In CVPR, Cited by: Table 5, §2.

[^49]: G. Williams, N. Wagener, B. Goldfain, P. Drews, J. M. Rehg, B. Boots, and E. A. Theodorou (2017) Information theoretic MPC for model-based reinforcement learning. In ICRA, Cited by: §2, §3.2.

[^50]: M. Wozniak, L. Liu, Y. Cai, and P. Jensfelt (2026) PRIX: learning to plan from raw pixels for end-to-end autonomous driving. IEEE Robotics Autom. Lett.. Cited by: Table 5.

[^51]: P. Xiao, Z. Shao, S. Hao, Z. Zhang, X. Chai, J. Jiao, Z. Li, J. Wu, K. Sun, K. Jiang, Y. Wang, and D. Yang (2021) PandaSet: advanced sensor suite dataset for autonomous driving. In ITSC, Cited by: Table 4.

[^52]: Z. Xing, X. Zhang, Y. Hu, B. Jiang, T. He, Q. Zhang, X. Long, and W. Yin (2025) Goalflow: goal-driven flow matching for multimodal trajectories generation in end-to-end autonomous driving. In CVPR, Cited by: §2.

[^53]: W. Yao, Z. Li, S. Lan, Z. Wang, X. Sun, J. M. Álvarez, and Z. Wu (2026) DriveSuprim: towards precise trajectory selection for end-to-end planning. In AAAI, Cited by: Table 5, Table 6, §2.

[^54]: Y. Yin, P. Khayatan, É. Zablocki, A. Boulch, and M. Cord (2024) ReGentS: real-world safety-critical driving scenario generation made stable. In ECCV Workshops, Cited by: §3.2.

[^55]: C. Yuan, Z. Zhang, J. Sun, S. Sun, Z. Huang, C. D. W. Lee, D. Li, Y. Han, A. Wong, K. P. Tee, et al. (2024) Drama: an efficient end-to-end motion planner for autonomous driving with mamba. In ISRR, Cited by: Table 5.

[^56]: E. Yurtsever, J. Lambert, A. Carballo, and K. Takeda (2020) A survey of autonomous driving: common practices and emerging technologies. IEEE Access. Cited by: §2.

[^57]: W. Zeng, W. Luo, S. Suo, A. Sadat, B. Yang, S. Casas, and R. Urtasun (2019) End-to-end interpretable neural motion planner. In CVPR, Cited by: §2.

[^58]: P. Zheng, Y. Zhao, Z. Gong, H. Zhu, and S. Wu (2025) SimpleVSF: vlm-scoring fusion for trajectory prediction of end-to-end autonomous driving. arXiv preprint arXiv:2510.17191. Cited by: §2.

[^59]: Y. Zheng, P. Yang, Z. Xing, Q. Zhang, Y. Zheng, Y. Gao, P. Li, T. Zhang, Z. Xia, P. Jia, et al. (2025) World4Drive: end-to-end autonomous driving via intention-aware physical latent world model. In ICCV, Cited by: Table 5, Table 6.

[^60]: H. Zhou, L. Lin, J. Wang, Y. Lu, D. Bai, B. Liu, Y. Wang, A. Geiger, and Y. Liao (2026) HUGSIM: A real-time, photo-realistic and closed-loop simulator for autonomous driving. IEEE Trans. Pattern Anal. Mach. Intell.. Cited by: Appendix A, §1, §2, §4.1, §4.2, Table 4, Table 4.

[^61]: Z. Zhou, T. Cai, S. Z. Zhao, Y. Zhang, Z. Huang, B. Zhou, and J. Ma (2025) AutoVLA: a vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. NeurIPS. Cited by: Table 5.

[^62]: Z. Zhou, R. Yang, Y. Guo, S. X. Chen, T. Feng, K. Pistunova, Y. Shen, L. Su, J. Ma, et al. (2026) SpanVLA: efficient action bridging and learning from negative-recovery samples for vision-language-action model. arXiv preprint arXiv:2604.19710. Cited by: Table 6.
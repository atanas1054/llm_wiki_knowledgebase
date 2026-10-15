---
title: "SparseDriveV2: Scoring is All You Need for End-to-End Autonomous Driving"
source: "https://arxiv.org/html/2603.29163v1"
author:
published:
created: 2026-10-14
description:
tags:
  - "clippings"
---
Wenchao Sun Affiliation: Tsinghua University, Beijing, China Affiliation: Horizon Continental Technology, Beijing, China    Xuewu Lin Affiliation: Horizon, Beijing, China    Keyu Chen Affiliation: Tsinghua University, Beijing, China    Zixiang Pei Affiliation: Horizon Continental Technology, Beijing, China    Xiang Li Affiliation: Horizon Continental Technology, Beijing, China    Yining Shi Affiliation: Tsinghua University, Beijing, China    Sifa Zheng Affiliation: Tsinghua University, Beijing, China

###### Abstract

End-to-end multi-modal planning has been widely adopted to model the uncertainty of driving behavior, typically by scoring candidate trajectories and selecting the optimal one. Existing approaches generally fall into two categories: scoring a large static trajectory vocabulary, or scoring a small set of dynamically generated proposals. While static vocabularies often suffer from coarse discretization of the action space, dynamic proposals provide finer-grained precision and have shown stronger empirical performance on existing benchmarks. However, it remains unclear whether dynamic generation is fundamentally necessary, or whether static vocabularies can already achieve comparable performance when they are sufficiently dense to cover the action space. In this work, we start with a systematic scaling study of Hydra-MDP, a representative scoring-based method, revealing that performance consistently improves as trajectory anchors become denser, without exhibiting saturation before computational constraints are reached. Motivated by this observation, we propose SparseDriveV2 to push the performance boundary of scoring-based planning through two complementary innovations: (1) a scalable vocabulary representation with a factorized structure that decomposes trajectories into geometric paths and velocity profiles, enabling combinatorial coverage of the action space, and (2) a scalable scoring strategy with coarse factorized scoring over paths and velocity profiles followed by fine-grained scoring on a small set of composed trajectories. By combining these two techniques, SparseDriveV2 scales the trajectory vocabulary to be 32 $\times$ denser than prior methods, while still enabling efficient scoring over such super-dense candidate set. With a lightweight ResNet-34 as backbone, SparseDriveV2 achieves 92.0 PDMS and 90.1 EPDMS on NAVSIM, with 89.15 Driving Score and 70.00 Success Rate on Bench2Drive. Code and model are released at [https://github.com/swc-17/SparseDriveV2](https://github.com/swc-17/SparseDriveV2).

###### Keywords:

End-to-end Autonomous Driving Scoring-based Planning

## 1 Introduction

Table 1: Systematic scaling study of static vocabulary density based on Hydra-MDP [^18]. We report EPDMS on NAVSIM v2 and training memory consumption on NVIDIA L20 GPU with 48 GB memory. EPDMS improvements are computed relative to the previous scale.

| #Anchors | EPDMS | Memory (MB) |
| --- | --- | --- |
| 1024 | 85.02 | 9531 |
| 2048 | 85.80 (+0.78) | 11451 |
| 4096 | 86.33 (+0.53) | 15513 |
| 8192 | 86.78 (+0.45) | 23261 |
| 16384 | 87.35 (+0.57) | 38877 |
| 32768 | – | OOM |

End-to-end autonomous driving has recently emerged as a promising paradigm that unifies all driving tasks into a single holistic model, enabling direct optimization toward planning objectives [^8] [^14] [^28]. Owing to the inherent uncertainty and non-deterministic nature of driving behaviors, multi-modal planning has been widely adopted in end-to-end driving [^1] [^18], which typically scores a set of candidate trajectories and selects the optimal one.

Based on how candidate trajectories are constructed, existing methods can be broadly categorized into two classes. The first class relies on a static trajectory vocabulary [^1] [^18] [^16] [^34], typically obtained by clustering trajectories from large-scale driving datasets. Given a predefined vocabulary, the planner simply scores all candidates and selects the trajectory with the highest score. However, its performance is often constrained by the coarse discretization of the action space. Due to memory and computational limits, the vocabulary size is usually restricted to only several thousand anchors, which fails to adequately cover the action space in complex driving scenarios.

The second class seeks to overcome this limitation through dynamic trajectory generation, such as regression-based methods [^6] or diffusion-based approaches [^22]. By generating trajectories conditioned on scene information, these methods enable more flexible and expressive candidate proposals, allowing for finer-grained adaption and broader coverage of the action space. When further combined with advanced scoring modules [^34], dynamic generation methods have demonstrated stronger empirical performance on existing benchmarks [^38]. Nevertheless, these gains typically come at the cost of increased model complexity, including additional network components or iterative denoising procedures.

This naturally raises a fundamental yet largely underexplored question: is dynamic trajectory generation truly necessary for high-performance end-to-end planning, or equivalently, can static trajectory vocabularies already achieve comparable performance when they are sufficiently dense and properly scored?

In this work, we revisit scoring-based planning from a scaling perspective. We conduct a systematic study on the effect of vocabulary density with Hydra-MDP [^18] as a representative method. As shown in Tab. 1, as the number of trajectory anchors increases, planning performance consistently improves, without exhibiting saturation before computational constraints are reached. This finding suggests that the perceived limitations of static vocabularies are not intrinsic, but instead stem from insufficient coverage of the action space under computational constraint. Consequently, the key challenge for advancing scoring-based planning lies in how to (1) represent an extremely dense trajectory vocabulary to adequately cover the action space, and (2) efficiently score such a super-dense vocabulary under computational budgets.

Motivated by this observation, we propose SparseDriveV2 to push the performance boundary of scoring-based planning through a scalable vocabulary representation and a scalable scoring strategy. Rather than simply increasing vocabulary size, we introduce a scalable vocabulary representation that factorizes spatiotemporal trajectories into spatial-level geometric paths and temporal-level velocity profiles. The factorized representation enables combinatorial composition, substantially expanding trajectory coverage while preserving a compact structure of vocabulary. To make scoring over such super-dense vocabulary computationally feasible, we further design a scalable scoring strategy that first performs coarse factorized scoring independently over paths and velocity profiles to prune low-quality candidates, followed by fine-grained scoring on a small set of composed trajectories. This strategy enables precise spatiotemporal reasoning to be performed only on a small subset of high-quality trajectories, regardless of the full vocabulary size.

By combining above design, SparseDriveV2 scales the trajectory vocabulary to be 32× denser than prior methods (1024 $\times$ 256 vs. 8192), while still enabling efficient scoring over such super-dense candidate set. As a result, SparseDriveV2 achieves SOTA performance on NAVSIM [^3] and Bench2Drive [^12] benchmarks.

Our contributions are summarized as follows:

- We revisit scoring-based planning from a scaling perspective with a systematic study, revealing the bottlenecks that limit the performance of existing methods.
- We propose a scalable vocabulary representation that factorizes trajectories into geometric paths and velocity profiles, enabling combinatorial composition to expand trajectory coverage while preserving a compact structure of vocabulary.
- We introduce a scalable scoring strategy that performs coarse factorized scoring over paths and velocity profiles followed by fine-grained scoring on a small set of composed trajectories, making it feasible to score over super-dense vocabulary efficiently.
- Combining scalable vocabulary and scalable scoring, we present SparseDriveV2, achieving SOTA performance on challenging benchmarks and setting a new record for scoring-based method.

## 2 Related Works

### 2.1 Scoring-Based Planning Methods

Scoring-based methods learn a scorer to select an optimal trajectory from a predefined set of candidates. A prominent example is VADv2 [^1], which formulates end-to-end autonomous driving as a probabilistic planning problem and learns the action distribution from large-scale demonstration data. The Hydra-MDP [^18] [^16] series extends single-target imitation learning to multi-target settings by distilling knowledge from both human experts and rule-based teachers. DriveSuprim [^34] further enhances fine-grained discrimination capability through a coarse-to-fine refinement module. While being conceptually simple, the performance of these methods is often limited by the coarse discretization of the trajectory space, since the computational cost of scoring increases with the size of the trajectory vocabulary.

### 2.2 Dynamic Trajectory Generation Methods

Dynamic trajectory generation approaches aim to enhance the expressiveness of the candidate set by generating trajectory proposals conditioned on scene information, rather than relying on a fixed trajectory vocabulary. One class of methods adopts a regression-based paradigm, in which trajectory proposals are directly predicted or iteratively refined through regression modules. For instance, ipad [^6] centers the planning pipeline around a sparse set of candidate proposals and progressively refines them via proposal-anchored attention, followed by a scoring module to select the optimal trajectory.

Another class of approaches leverage generative models to sample diverse and fine-grained trajectory candidates. Diffusion-based planners, such as DiffusionDrive [^22], employs a truncated diffusion policy to denoise trajectory samples from an anchored Gaussian distribution, enabling the modeling of complex multimodal behaviors beyond fixed anchors. DiffusionDriveV2 further incorporates reinforcement learning to suppress low-quality modes while encouraging the exploration of higher-quality trajectories. By integrating advanced scoring modules [^34], DiffusionDriveV2 outperforms scoring-based methods relying on static vocabularies. GoalFlow [^33], a representative flow-matching approach, constrains the generation process using goal points to produce high-quality multimodal trajectories. Despite their strong empirical performance, these methods typically require additional network components dedicated to trajectory generation, which increases the overall model complexity.

### 2.3 Hybrid Methods

One notable method is GTRS [^20], which investigates the limitations of both static and dynamic candidate sets in end-to-end multimodal planning. GTRS observes that static vocabularies, while providing coarse discretization, lack the flexibility to adapt to the fine-grained variations required in diverse driving scenarios. Conversely, a small number of dynamic proposals often fail to generalize to unseen trajectories and fail to capture broader trajectory distribution. To mitigate these issues, GTRS proposes a generalized scoring framework that combines diffusion-based dynamic proposals and static vocabulary, with a robust scorer trained using vocabulary dropout and sensor augmentation. While these innovations yield strong performance and achieve superior result, they also introduce additional complexity through multiple specialized sub-modules. In contrast, our approach adheres to a purely scoring-based paradigm without dynamic trajectory generation.

![[overview_.png|Refer to caption]]

Figure 1: Overview of the SparseDriveV2 framework. SparseDriveV2 factorizes (a) a spatiotemporal trajectory into (b) a geometric path and a velocity profile, and reconstructs the trajectory by (c) composing the two components. This representation enables (d) a super-dense trajectory vocabulary constructed from a compact set of paths and velocity profiles. Conditioned on (e) scene features, the scalable scoring strategy first performs (f) coarse factorized scoring over paths and velocity profiles to select top- k candidates, followed by (g) fine-grained scoring over the composed trajectories to produce the final planning decision.

## 3 Methods

### 3.1 Problem Formulation

End-to-End Planning. An end-to-end autonomous driving system takes raw sensor observations $o_{t}$ as input and predicts a future trajectory for the ego vehicle. A trajectory is represented as a sequence of waypoints sampled at a fixed temporal interval $\Delta t$:

$$
\tau=\{(x_{t},y_{t})\}_{t=1}^{T},
$$

where $T$ denotes the planning horizon, and $(x_{t},y_{t})$ represents the location of the ego vehicle at time step $t$ in the current ego-centric coordinate system.

Scoring-Based Planning. Scoring-based planning predefines a trajectory vocabulary $\mathcal{T}=\{\tau_{i}\}_{i=1}^{N}$, where $N$ denotes the number of trajectory anchors in the vocabulary. Given a specific driving scenario, a scorer evaluates the quality of each candidate trajectory according to predefined metrics, such as the $\ell_{2}$ distance to human demonstrations, safety-related metrics, or traffic rule adherence. The scorer produces a trajectory score $s_{i}$ for each candidate, and the optimal trajectory with the highest score is selected for vehicle control:

$$
\tau^{*}=\arg\max_{\tau\in\mathcal{T}}s(\tau,o_{t}).
$$

### 3.2 Scalable Vocabulary Representation

A trajectory inherently couples spatial geometry and temporal evolution, describing where the vehicle goes and how fast it progresses over time. Covering the full action space with monolithic spatiotemporal trajectories therefore requires an extremely large number of anchors, which quickly becomes impractical under memory and computational constraints. To address this challenge, we introduce a vocabulary representation with factorized structure, which decouples trajectory into a spatial component and a temporal component, enabling combinatorial coverage of the action space with a compact vocabulary.

Trajectory Factorization. We factorize a trajectory into a geometric path and a velocity profile. A path is defined as a sequence of spatial waypoints

$$
p=\{(x_{i},y_{i})\}_{i=1}^{S},
$$

where consecutive waypoints are sampled at a fixed spatial interval $\Delta s$ along the path, and $S$ denotes the number of spatial waypoints. The path encodes the geometric shape of motion but does not contain any temporal information. In contrast, a velocity profile is defined as a temporal sequence of scalar speeds

$$
v=\{v_{t}\}_{t=1}^{T},
$$

where each $v_{t}$ represents the average speed over a fixed temporal interval $\Delta t$. The velocity profile specifies how fast the vehicle progresses along a path over time, independent of the spatial geometry.

From Trajectory to Path and Velocity. Given a trajectory, we extract the corresponding path and velocity profile:

$$
(p,v)=\mathcal{D}(\tau).
$$

To obtain the path, we first specify a target spatial horizon $S_{\max}$ that defines the desired maximum path length. We then compute the cumulative traveled distance along the trajectory and resample trajectory positions at a fixed spatial interval $\Delta s$ using interpolation. This results in a geometric path representation that captures the spatial shape of motion independent of timing. If the available trajectory does not cover the target spatial horizon $S_{\max}$, a validity mask is applied to indicate missing path segments. The velocity profile is obtained by computing the average speed between consecutive trajectory points. Let $(x_{0},y_{0})=(0,0)$ denote the current ego position, the velocity profile is defined as

$$
v_{t}=\frac{\|(x_{t},y_{t})-(x_{t-1},y_{t-1})\|}{\Delta t},\quad t=1,\dots,T.
$$

From Path and Velocity to Trajectory. Given a path and a velocity profile, a spatiotemporal trajectory can be reconstructed by composing the two components. Specifically, the cumulative traveled distance at time step $t$ is computed as

$$
s_{t}=\sum_{k=1}^{t}v_{k}\,\Delta t.
$$

The trajectory position at time $t$ is then obtained by interpolating along the path at distance $s_{t}$. This composition defines a trajectory

$$
\tau=\mathcal{C}(p,v),
$$

where $\mathcal{C}(\cdot)$ denotes the path–velocity composition operator.

### 3.3 Scalable Vocabulary Construction

We construct the trajectory vocabulary from large-scale human driving demonstrations. Leveraging the proposed factorized representation, we separately build a path vocabulary and a velocity vocabulary, which can then be composed to form a dense set of trajectories.

Path Vocabulary. To construct the path vocabulary, we extract future trajectories from the training data with a sufficiently long temporal horizon to ensure coverage of the target spatial horizon $S_{\max}$. Each trajectory is then converted into a geometric path of length $S_{\max}$ following the procedure described in the previous section. The extracted paths are clustered using the K-Means algorithm to obtain a set of representative path anchors, forming the path vocabulary.

$$
\mathcal{P}=\{p_{i}\}_{i=1}^{N_{p}},
$$

where $N_{p}$ denotes the number of path anchors.

Velocity Vocabulary. The velocity vocabulary is constructed in a similar manner. For each trajectory, we compute its velocity profile according to Eq. 6, resulting in a sequence of scalar speeds over the planning horizon $T$. These velocity profiles are then clustered to obtain a set of representative velocity anchors, forming the velocity vocabulary

$$
\mathcal{V}=\{v_{j}\}_{j=1}^{N_{v}},
$$

where $N_{v}$ denotes the number of velocity anchors.

Trajectory Vocabulary via Composition. Given the path and velocity vocabularies, we obtain the full trajectory vocabulary by composing every path anchor with every velocity anchor:

$$
\mathcal{T}=\{\tau_{i,j}\mid\tau_{i,j}=\mathcal{C}(p_{i},v_{j}),\;p_{i}\in\mathcal{P},\;v_{j}\in\mathcal{V}\}.
$$

This factorized construction enables combinatorial coverage of the trajectory space, allowing us to form a super-dense trajectory vocabulary from compact path and velocity sets. As a result, the size of the trajectory vocabulary scales as $|\mathcal{T}|=N_{p}\times N_{v}$, which is significantly larger than that of conventional monolithic vocabularies, while remaining computationally manageable under the proposed scoring framework.

### 3.4 Scalable Scoring

Scene Encoding. We first extract scene features from raw sensor observations $o_{t}$ and encode the ego status $e_{t}$ into status features:

$$
\mathbf{F}=\Phi(o_{t}),\qquad\mathbf{E}=\Psi(e_{t}),
$$

where $\Phi(\cdot)$ and $\Psi(\cdot)$ denote the scene encoder and the status encoder, respectively. Following the design philosophy of SparseDrive [^28], the scene encoder directly extracts multi-view image features for subsequent interaction, without explicitly constructing BEV representations [^21] [^9].

Conditioned on the scene features $\mathbf{F}$ and status features $\mathbf{E}$, a straightforward approach to scoring-based planning treats each trajectory as a monolithic unit and directly scores all trajectory candidates. However, the computational complexity of such trajectory-level scoring grows linearly with the number of trajectory anchors. When scaling to super-dense vocabularies containing hundreds of thousands of trajectories, this approach becomes computationally prohibitive for both training and inference.

Coarse Factorized Scoring. To enable efficient scoring over the super-dense combinatorial trajectory vocabulary, we exploit the factorized structure and perform coarse-grained scoring independently at the path and velocity levels.

#### Path scoring.

For each path anchor $p_{i}\in\mathcal{P}$, we first encode it into a path embedding using a lightweight MLP:

$$
\mathbf{e}^{p}_{i}=E_{p}(p_{i}),\qquad\mathbf{e}^{p}_{i}\in\mathbb{R}^{d}.
$$

We then incorporate scene context by interacting path embedding with the scene features $\mathbf{F}$ and the status features $\mathbf{E}$:

$$
\widetilde{\mathbf{e}}^{p}_{i}=\Psi_{p}(\mathbf{e}^{p}_{i}+\mathbf{E},\mathbf{F}),
$$

where $\Psi_{p}(\cdot)$ denotes a path–scene interaction module. Since a path provides explicit spatial geometry, $\Psi_{p}$ can be instantiated either as multi-head cross-attention [^30] between path embeddings and scene features, or as deformable aggregation [^23] that samples scene features along the path geometry. Finally, a coarse path score is predicted directly from each contextualized path embedding:

$$
s^{p}_{i}=f_{p}(\widetilde{\mathbf{e}}^{p}_{i}).
$$

#### Velocity scoring.

Similarly, for each velocity anchor $v_{j}\in\mathcal{V}$, we obtain a velocity embedding

$$
\mathbf{e}^{v}_{j}=E_{v}(v_{j}),\qquad\mathbf{e}^{v}_{j}\in\mathbb{R}^{d},
$$

and incorporate scene context via:

$$
\widetilde{\mathbf{e}}^{v}_{j}=\Psi_{v}(\mathbf{e}^{v}_{j}+\mathbf{E},\mathbf{F}),
$$

where $\Psi_{v}(\cdot)$ is implemented with cross-attention since velocity profiles do not contain explicit spatial geometry. A coarse velocity score is then predicted as

$$
s^{v}_{j}=f_{v}(\widetilde{\mathbf{e}}^{v}_{j}).
$$

#### Top-KK selection.

Although path and velocity are scored independently, they capture complementary aspects of high-level driving intent and are effective at filtering out a large number of low-quality trajectory candidates. For example, high-speed velocity profiles are unlikely to be valid in emergency or congested scenarios, while turning paths can be safely discarded in straight-lane driving scenario. By performing coarse scoring at the path and velocity levels, many implausible combinations can be eliminated before trajectory composition.

Specifically, based on the coarse path scores $\{s^{p}_{i}\}$ and velocity scores $\{s^{v}_{j}\}$, we select the top- $K_{p}$ paths and top- $K_{v}$ velocity profiles, respectively. The composition of these selected components yields a high-quality trajectory candidate set

$$
\mathcal{T}_{\text{coarse}}=\{\mathcal{C}(p_{i},v_{j})\mid p_{i}\in\mathcal{P}_{K_{p}},\;v_{j}\in\mathcal{V}_{K_{v}}\},
$$

with a vocabulary size of $|\mathcal{T}_{\text{coarse}}|=K_{p}\times K_{v}$, which is several orders of magnitude smaller than the full combinatorial vocabulary.

Fine-Grained Trajectory Scoring. After coarse filtering, we perform fine-grained scoring on the composed trajectory candidates. Each trajectory anchor is obtained by composing a path anchor and a velocity anchor, i.e., $\tau_{i,j}=\mathcal{C}(p_{i},v_{j})$. Correspondingly, a trajectory feature can be naturally constructed by combining the associated path embedding and velocity embedding:

$$
\mathbf{e}^{\tau}_{i,j}=\widetilde{\mathbf{e}}^{p}_{i}+\widetilde{\mathbf{e}}^{v}_{j}.
$$

This fusion operation yields a trajectory feature that jointly encodes the spatial information from the path and the temporal information from the velocity profile, and can be directly used for trajectory-level scoring.

However, this formulation implicitly assumes independence between path and velocity, which does not always hold in real-world driving. For instance, a sharply turning path is typically incompatible with very high speeds. To capture such spatiotemporal dependencies and enable joint reasoning over path, velocity, and scene context, we introduce a trajectory re-conditioning mechanism.

Specifically, the trajectory feature $\mathbf{e}^{\tau}_{i,j}$ further interacts with the scene features $\mathbf{F}$ to obtain a re-conditioned trajectory embedding:

$$
\widetilde{\mathbf{e}}^{\tau}_{i,j}=\Psi_{\tau}(\mathbf{e}^{\tau}_{i,j},\mathbf{F}),
$$

where $\Psi_{\tau}(\cdot)$ denotes a trajectory–scene interaction module implemented as deformable aggregation. This re-conditioning step allows the model to perform fine-grained spatiotemporal reasoning conditioned on the scene. Finally, a fine-grained trajectory score is predicted as

$$
s^{\tau}_{i,j}=f_{\tau}(\widetilde{\mathbf{e}}^{\tau}_{i,j}).
$$

### 3.5 Training and Inference

Training Objectives. Our training objective follows a scoring-based formulation and supervises different stages of the scoring pipeline with appropriate targets. We apply distance-based soft classification losses for coarse factorized scoring and trajectory-level imitation learning, together with metric supervision distilled from a rule-based teacher system.

#### Coarse Factorized Scoring.

For the path-level scoring, given a ground-truth geometric path $\hat{p}$, we compute the masked average squared distance between $\hat{p}$ and each path anchor $p_{i}\in\mathcal{P}$:

$$
d^{p}_{i}=\frac{1}{|\mathcal{M}|}\sum_{k\in\mathcal{M}}\left\|p_{i}(k)-\hat{p}(k)\right\|_{2}^{2},
$$

where $\mathcal{M}$ denotes the set of valid path points and $|\mathcal{M}|$ is the number of valid points. We then construct a soft target distribution and apply a cross-entropy loss:

$$
\mathcal{L}_{\text{path}}=\mathrm{CE}\!\left(s^{p},\;\mathrm{Softmax}\!\left(-\lambda_{p}\,d^{p}\right)\right),
$$

where $s^{p}\in\mathbb{R}^{N_{p}}$ denotes the predicted path scores and $\lambda_{p}$ is a scaling factor.

Similarly, for velocity-level scoring, we compute the $L_{1}$ distance between the ground-truth velocity profile $\hat{v}$ and each velocity anchor $v_{j}\in\mathcal{V}$:

$$
d^{v}_{j}=\sum_{t=1}^{T}\left|v_{j}(t)-\hat{v}(t)\right|,
$$

and supervise velocity scoring with a soft classification loss:

$$
\mathcal{L}_{\text{vel}}=\mathrm{CE}\!\left(s^{v},\;\mathrm{Softmax}\!\left(-\lambda_{v}\,d^{v}\right)\right),
$$

where $s^{v}\in\mathbb{R}^{N_{v}}$ denotes the predicted velocity scores and $\lambda_{v}$ is a scaling factor.

#### Fine-Grained Trajectory Scoring.

After coarse filtering, we supervise fine-grained scoring on composed trajectory candidates. Given a ground-truth trajectory $\hat{\tau}$ and a trajectory anchor $\tau_{k}$, we compute their squared $L_{2}$ distance:

$$
d^{\tau}_{k}=\sum_{t=1}^{T}\left\|\tau_{k}(t)-\hat{\tau}(t)\right\|_{2}^{2}.
$$

Trajectory-level scores are trained using the following soft classification loss:

$$
\mathcal{L}_{\text{traj}}=\mathrm{CE}\!\left(s^{\tau},\;\mathrm{Softmax}\!\left(-\lambda_{\tau}\,d^{\tau}\right)\right),
$$

where $s^{\tau}$ denotes the predicted trajectory scores and $\lambda_{\tau}$ is a scaling factor.

Following Hydra-MDP [^18], we further incorporate metric supervision at the trajectory level using a rule-based teacher system. The teacher evaluates each composed trajectory under a set of predefined driving metrics and produces metric sub-scores $\{\hat{s}^{(m)}\}$, reflecting safety, progress, comfort, and rule compliance. The scoring network directly predicts the corresponding sub-scores $\{s^{(m)}\}$ for each trajectory, and is trained using binary cross-entropy loss:

$$
\mathcal{L}_{\text{metric}}=\sum_{m}\mathrm{BCE}\!\left(s^{(m)},\;\hat{s}^{(m)}\right).
$$

#### Overall Objective.

The final training objective is a weighted sum of all losses:

$$
\mathcal{L}=\mathcal{L}_{\text{path}}+\mathcal{L}_{\text{vel}}+\mathcal{L}_{\text{traj}}+\alpha\,\mathcal{L}_{\text{metric}},
$$

where $\alpha$ controls the contribution of rule-based metric supervision.

Inference. At inference time, trajectory candidates are assigned final scores by aggregating the predicted sub-scores using the same weighting scheme as in the benchmark evaluation protocol. The planner then selects the trajectory with the highest aggregated score as the final driving decision.

Table 2: Performance on NAVSIM v1 navtest leaderboard.

| Method | Img. Backbone | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf. $\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VADv2 [^1] | ResNet-34 | 97.2 | 89.1 | 91.6 | 100 | 76.0 | 80.9 |
| UniAD [^8] | ResNet-34 | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| Transfuser [^2] | ResNet-34 | 97.7 | 92.8 | 92.8 | 100 | 79.2 | 84.0 |
| PARA-Drive [^31] | ResNet-34 | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 |
| DRAMA [^35] | ResNet-34 | 98.0 | 93.1 | 94.8 | 100 | 80.1 | 85.5 |
| GoalFlow [^33] | ResNet-34 | 98.3 | 93.8 | 94.3 | 100 | 79.8 | 85.7 |
| Hydra-MDP [^18] | ResNet-34 | 98.3 | 96.0 | 94.6 | 100 | 78.7 | 86.5 |
| Hydra-MDP++ [^16] | ResNet-34 | 97.6 | 96.0 | 93.1 | 100 | 80.4 | 86.6 |
| ARTEMIS [^5] | ResNet-34 | 98.3 | 95.1 | 94.3 | 100 | 81.4 | 87.0 |
| FUMP [^24] | ResNet-34 | 98.1 | 96.2 | 94.2 | 100 | 82.0 | 87.8 |
| DiffusionDrive [^22] | ResNet-34 | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| WoTE [^17] | ResNet-34 | 98.5 | 96.8 | 94.9 | 99.9 | 81.9 | 88.3 |
| DIVER [^26] | ResNet-34 | 98.5 | 96.5 | 94.9 | 100 | 82.6 | 88.3 |
| DriveSuprim [^34] | ResNet-34 | 97.8 | 97.3 | 93.6 | 100 | 86.7 | 89.9 |
| DiffusionDriveV2 [^38] | ResNet-34 | 98.3 | 97.9 | 94.8 | 99.9 | 87.5 | 91.2 |
| ipad [^6] | ResNet-34 | 98.6 | 98.3 | 94.9 | 100 | 88.0 | 91.7 |
| Hydra-MDP [^18] | V2-99 | 98.4 | 97.8 | 93.9 | 100 | 86.5 | 90.3 |
| GoalFlow [^33] | V2-99 | 98.4 | 98.3 | 94.6 | 100 | 85.0 | 90.3 |
| SparseDriveV2 | ResNet-34 | 98.5 | 98.4 | 95.0 | 99.9 | 88.6 | 92.0 |

## 4 Experiments

### 4.1 Dataset and Metrics

We conduct experiments on NAVSIM [^3] and Bench2Drive [^12] benchmarks. NAVSIM is split into navtrain (1192 training scenes) and navtest (136 evaluation scenes), with two evalution protocols, which is NAVSIM v1 and NAVSIM v2. NAVSIM v1 includes 5 metrics: no collision (NC), drivable area compliance (DAC), ego progress (EP), time-to-collision (TTC), and comfort (C). PDM score (PDMS) is computed by weighted aggregation of these metrics:

$$
PDMS=NC\times DAC\times\frac{(5\times TTC+2\times C+5\times EP)}{12}
$$

NAVSIM v2 introduces 4 extended metrics, including driving direction compliance (DDC), traffic light compliance (TL), lane keeping (LK) and extended comfort (EC). The Extended PDM Score (EPDMS) is aggregated by:

$$
\begin{aligned} EPDMS&=NC\times DAC\times DDC\times TL\times\frac{(5\times EP+5\times TTC+2\times LK+2\times C+2\times EC)}{16}\end{aligned}
$$

Bench2Drive is a closed-loop benchmark based on CARLA simulator [^4], with 220 test routes across 44 interactive scenarios. The official metrics include Driving Score, Success Rate, Efficiency and Comfortness.

Table 3: Performance on NAVSIM v2 navtest leaderboard. EPDMS <sup>∗</sup> denotes results computed using the original NAVSIM v2 evaluation code, in which human-behavior filtering was not applied when calculating EPDMS. This version is reported for fair comparison with prior works. EPDMS denotes results computed using the updated official implementation after the issue was resolved.

| Method | Img. Backbone | NC $\uparrow$ | DAC $\uparrow$ | DDC $\uparrow$ | TL $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | LK $\uparrow$ | HC $\uparrow$ | EC $\uparrow$ | EPDMS <sup>∗</sup> $\uparrow$ | EPDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ego Status MLP | ResNet-34 | 93.1 | 77.9 | 92.7 | 99.6 | 86.0 | 91.5 | 89.4 | 98.3 | 85.4 | 64.0 | – |
| Transfuser [^2] | ResNet-34 | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 | – |
| Hydra-MDP++ [^16] | ResNet-34 | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | 81.4 | – |
| DriveSuprim [^34] | ResNet-34 | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 | 83.1 | – |
| ARTEMIS [^5] | ResNet-34 | 98.3 | 95.1 | 98.6 | 99.8 | 81.5 | 97.4 | 96.5 | 98.3 | 98.3 | 83.1 | – |
| DiffusionDriveV2 | ResNet-34 | 97.7 | 96.6 | 99.2 | 99.8 | 88.9 | 97.2 | 96.0 | 97.8 | 91.0 | 85.5 | 87.5 |
| Hydra-MDP++ [^16] | V2-99 | 98.4 | 98.0 | 99.4 | 99.8 | 87.5 | 97.7 | 95.3 | 98.3 | 77.4 | 85.1 | – |
| DriveSuprim [^34] | V2-99 | 97.8 | 97.9 | 99.5 | 99.9 | 90.6 | 97.1 | 96.6 | 98.3 | 77.9 | 86.0 | – |
| SparseDriveV2 | ResNet-34 | 98.1 | 98.1 | 99.6 | 99.8 | 91.1 | 97.3 | 96.9 | 98.2 | 78.4 | 86.7 | 90.1 |

### 4.2 Implementation Details

We present the implementation details for NAVSIM in this section, while those for Bench2Drive are deferred to the appendix due to space limitations.

For NAVSIM v1, we construct a factorized trajectory vocabulary consisting of 1024 path anchors and 256 velocity anchors. Path anchors are sampled with a spatial interval of 1 m and a maximum spatial horizon of 50 m. Velocity anchors are defined with a temporal interval of 0.5 s and a temporal horizon of 4 s. The combinatorial composition results in $1024\times 256=262{,}144$ trajectory anchors, which is 32 $\times$ denser than the commonly used 8192 anchors in prior scoring-based methods [^18] [^34].

To enable efficient inference over this super-dense vocabulary, we adopt a progressive filtering strategy with two decoder layers. The first layer retains the top 128 path anchors and 64 velocity anchors. The second layer further refines the candidates to 20 paths and 20 velocities, resulting in 400 trajectory hypotheses for fine-grained scoring. Metric supervision is applied only to these 400 filtered trajectories. For NAVSIM v2, the only difference is that the number of velocity anchors in the second layer is reduced to 10 in order to accelerate metric ground-truth computation.

For scene feature extraction, we use a ResNet-34 [^7] backbone. Images from cameras $l_{0}$, $f$, and $r_{0}$ are used as input, with a resolution of $256\times 512$. We train all models on the navtrain split using 8 NVIDIA L20 GPUs, with a total batch size of 128 for 10 epochs. The learning rate is set to $1\times 10^{-4}$ and the weight decay is set to 0.

Table 4: Closed-loop planning performance on Bench2Drive. <sup>∗</sup> denotes expert feature distillation.

| Method | Driving Score $\uparrow$ | Success Rate (%) $\uparrow$ | Driving Efficiency $\uparrow$ | Comfort $\uparrow$ |
| --- | --- | --- | --- | --- |
| UniAD [^8] | 45.81 | 16.36 | 129.21 | 43.58 |
| VAD [^14] | 42.35 | 15.00 | 157.94 | 46.01 |
| SparseDrive [^28] | 44.54 | 16.71 | 170.21 | 48.63 |
| GenAD [^37] | 44.81 | 15.90 | \- | \- |
| DiFSD [^27] | 52.02 | 21.00 | 178.30 | \- |
| DriveTransformer [^13] | 63.46 | 35.01 | 100.64 | 20.78 |
| Hydra-NeXt [^19] | 73.86 | 50.00 | 197.76 | 20.68 |
| SimLingo [^25] | 86.02 | 67.27 | 259.23 | 33.67 |
| HiP-AD [^29] | 86.77 | 69.09 | 203.12 | 19.36 |
| TCP-traj\* [^32] | 59.90 | 30.00 | 76.54 | 18.08 |
| ThinkTwice\* [^11] | 62.44 | 31.23 | 69.33 | 16.22 |
| DriveAdapter\* [^10] | 64.22 | 33.08 | 70.22 | 16.01 |
| SparseDriveV2 | 89.15 | 70.00 | 199.84 | 18.32 |

Table 5: Multi-Ability results on Bench2Drive.\* denotes expert feature distillation.

<table><tbody><tr><th rowspan="2">Method</th><td colspan="5">Ability (%) <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td></td></tr><tr><td>Merging</td><td>Overtaking</td><td>Emergency Brake</td><td>Give Way</td><td>Traffic Sign</td><td>Mean</td></tr><tr><th>AD-MLP <sup><a href="#fn:36">36</a></sup></th><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>4.35</td><td>0.87</td></tr><tr><th>UniAD <sup><a href="#fn:8">8</a></sup></th><td>14.10</td><td>17.78</td><td>21.67</td><td>10.00</td><td>14.21</td><td>15.55</td></tr><tr><th>VAD <sup><a href="#fn:14">14</a></sup></th><td>8.11</td><td>24.44</td><td>18.64</td><td>20.00</td><td>19.15</td><td>18.07</td></tr><tr><th>DriveTransformer <sup><a href="#fn:13">13</a></sup></th><td>17.57</td><td>35.00</td><td>48.36</td><td>40.00</td><td>52.10</td><td>38.60</td></tr><tr><th>Hydra-NeXt <sup><a href="#fn:19">19</a></sup></th><td>40.00</td><td>64.44</td><td>61.67</td><td>50.00</td><td>50.00</td><td>53.22</td></tr><tr><th>HiP-AD <sup><a href="#fn:29">29</a></sup></th><td>50.00</td><td>84.44</td><td>83.33</td><td>40.00</td><td>72.10</td><td>65.98</td></tr><tr><th>TCP-traj* <sup><a href="#fn:32">32</a></sup></th><td>12.50</td><td>22.73</td><td>52.72</td><td>40.00</td><td>46.63</td><td>34.92</td></tr><tr><th>ThinkTwice* <sup><a href="#fn:11">11</a></sup></th><td>13.72</td><td>22.93</td><td>52.99</td><td>50.00</td><td>47.78</td><td>37.48</td></tr><tr><th>DriveAdapter* <sup><a href="#fn:10">10</a></sup></th><td>14.55</td><td>22.61</td><td>54.04</td><td>50.00</td><td>50.45</td><td>38.33</td></tr><tr><th>SparseDriveV2</th><td>66.25</td><td>75.55</td><td>75.00</td><td>50.00</td><td>71.57</td><td>67.67</td></tr></tbody></table>

### 4.3 Main Results

Results on NAVSIM v1. As shown in Tab. 2, SparseDriveV2 achieves a PDMS of 92.0, outperforming both prior scoring-based planners and dynamic trajectory generation methods. Notably, with only a lightweight ResNet-34 backbone (21.8M parameters), SparseDriveV2 still surpasses GoalFlow and Hydra-MDP equipped with the significantly larger V2-99 backbone [^15] (96.9M parameters), demonstrating the effectiveness of the proposed scalable vocabulary and scoring framework.

Results on NAVSIM v2. Tab. 3 reports the results on the NAVSIM v2 navtest split. Under the original evaluation protocol, SparseDriveV2 achieves the best performance among methods using the same backbone and also outperforms strong scoring-based approaches such as Hydra-MDP++ and DriveSuprim that adopt larger backbones. In particular, the substantial improvement in the EP metric indicates the advantage of the proposed scalable vocabulary in providing broader coverage of the action space. We additionally report the corrected EPDMS metric to encourage the community to adopt the updated evaluation protocol. Under this metric, SparseDriveV2 achieves a EPDMS of 90.1, surpassing DiffusionDriveV2 by 2.6 EPDMS.

Results on Bench2Drive. The closed-loop planning performance on Bench2Drive is reported in Tab. 4 and Tab. 5. SparseDriveV2 achieves a Driving Score of 89.15 and a Success Rate of 70.00%, together with a multi-ability score of 67.67. These results consistently outperform prior approaches, demonstrating the strong generalization ability of SparseDriveV2 in complex closed-loop driving scenarios.

### 4.4 Ablation Studies

Ablation for Scalable Vocabulary. We study the effect of vocabulary scaling by varying the numbers of path anchors ($N_{p}$) and velocity anchors ($N_{v}$), as shown in Tab. 6 (left). Increasing the vocabulary size consistently improves planning performance, indicating that a denser vocabulary provides broader action-space coverage and enables the scorer to select higher-quality trajectories.

Ablation for Scalable Scoring. We further evaluate the proposed scalable scoring design in Tab. 6 (right), including the path–scene interaction mechanism and the trajectory re-conditioning module. Replacing standard multi-head cross-attention with deformable aggregation consistently improves performance, suggesting that it captures more informative spatial cues along path anchors. Moreover, trajectory re-conditioning improves results under both attention types, highlighting the importance of modeling the spatiotemporal dependency of composed trajectories.

Table 6: Ablation studies on NAVSIM v2. Left: Effect of scaling the factorized vocabulary size by varying the number of path anchors ($N_{p}$) and velocity anchors ($N_{v}$). Right: Effect of the scalable scoring design, including path–scene interaction type and trajectory re-conditioning. MHA denotes multi-head cross-attention and DFA denotes deformable aggregation.

<table><tbody><tr><th colspan="3">Scalable Vocabulary</th><td colspan="3">Scalable Scoring</td></tr><tr><th><math><semantics><msub><mi>N</mi> <mi>p</mi></msub> <annotation>N_{p}</annotation></semantics></math></th><th><math><semantics><msub><mi>N</mi> <mi>v</mi></msub> <annotation>N_{v}</annotation></semantics></math></th><th>EPDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><td>Path Attn</td><td>Re-cond.</td><td>EPDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><th>512</th><th>128</th><th>88.7</th><td>MHA</td><td>✗</td><td>87.7</td></tr><tr><th>512</th><th>256</th><th>89.2</th><td>MHA</td><td>✓</td><td>89.9</td></tr><tr><th>1024</th><th>128</th><th>89.5</th><td>DFA</td><td>✗</td><td>89.9</td></tr><tr><th>1024</th><th>256</th><th>90.1</th><td>DFA</td><td>✓</td><td>90.1</td></tr></tbody></table>

## 5 Conclusion

In this work, we propose SparseDriveV2, which introduces a scalable trajectory vocabulary representation and a scalable scoring strategy for reasoning over super-dense candidate trajectories. Extensive experiments demonstrate that SparseDriveV2 achieves SOTA performance among both scoring-based and dynamic generation methods, suggesting that with sufficiently dense vocabularies and efficient scoring mechanisms, scoring-based planning can serve as a simple yet powerful paradigm for end-to-end autonomous driving.

## References

## Appendix 0.A Implementation Details for Bench2Drive

For Bench2Drive, we construct a factorized trajectory vocabulary comprising 1024 path anchors and 256 velocity anchors. Path anchors are sampled at a spatial interval of 1 m with a maximum spatial horizon of 15 m. Velocity anchors are defined with a temporal interval of 0.5 s and a temporal horizon of 3 s.

For scene feature extraction, we adopt a ResNet-50 backbone. Images from all six cameras are used as input with a resolution of $256\times 704$. All models are trained on 16 NVIDIA L20 GPUs with a total batch size of 96. Following SparseDrive, we incorporate object detection, online mapping, and motion prediction as auxiliary tasks. In addition, path and velocity embeddings interact with both agent queries and map queries to enhance environmental understanding.

The training process follows a two-stage paradigm. In the first stage, perception-related tasks are trained for 100 epochs. In the second stage, the planning module is introduced and jointly trained with the perception tasks for an additional 10 epochs. The learning rates for the two stages are set to $4\times 10^{-4}$ and $3\times 10^{-4}$, respectively. We do not apply metric-based supervision on Bench2Drive; therefore, the training process follows a pure imitation learning paradigm.

In Bench2Drive, the predicted trajectory must be converted into control commands for vehicle control. After selecting the final trajectory with the highest score, we decompose it into a geometric path and a corresponding velocity profile. For lateral control, we compute a speed-dependent target distance defined as

$$
d=0.5\times ego\_speed+2.5.
$$

We then select the point on the path whose cumulative distance is closest to $d$ as the preview point for steering computation. For longitudinal control, the first velocity in the velocity profile is directly used as the target speed to compute the throttle and brake commands.

## Appendix 0.B Qualitative Results

We provide qualitative results for NAVSIM in this section. In the following visualization, the green trajectory represents the ground truth from the human expert, the red trajectory is generated by the advanced scoring-based baseline GTRS-Dense, and the blue trajectory is produced by SparseDriveV2.

The visualization results for Bench2Drive can be found in supplementary material as videos.

![[smooth_turn.png|Refer to caption]]

Figure 2: SparseDriveV2 produces smoother trajectory than the baseline method in sharp-turning scenarios.

![[efficiency.png|Refer to caption]]

Figure 3: SparseDriveV2 achieves higher traffic efficiency, while the baseline method remains stationary.

![[high_level.png|Refer to caption]]

Figure 4: SparseDriveV2 is better aligned with the expert trajectory in terms of high-level intent, enabled by geometric path modeling.

![[navigation.png|Refer to caption]]

Figure 5: Failure cases: SparseDriveV2 generates trajectories with incorrect navigation decisions in certain scenarios, possibly due to insufficient navigation information.

[^1]: S. Chen, B. Jiang, H. Gao, B. Liao, Q. Xu, Q. Zhang, C. Huang, W. Liu, and X. Wang (2024) Vadv2: end-to-end vectorized autonomous driving via probabilistic planning. arXiv preprint arXiv:2402.13243. Cited by: §1, §1, §2.1, Table 2.

[^2]: K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger (2022) Transfuser: imitation with transformer-based sensor fusion for autonomous driving. IEEE transactions on pattern analysis and machine intelligence 45 (11), pp. 12878–12895. Cited by: Table 2, Table 3.

[^3]: D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, et al. (2024) Navsim: data-driven non-reactive autonomous vehicle simulation and benchmarking. Advances in Neural Information Processing Systems 37, pp. 28706–28719. Cited by: §1, §4.1.

[^4]: A. Dosovitskiy, G. Ros, F. Codevilla, A. Lopez, and V. Koltun (2017) CARLA: an open urban driving simulator. In Conference on robot learning, pp. 1–16. Cited by: §4.1.

[^5]: R. Feng, N. Xi, D. Chu, R. Wang, Z. Deng, A. Wang, L. Lu, J. Wang, and Y. Huang (2025) Artemis: autoregressive end-to-end trajectory planning with mixture of experts for autonomous driving. IEEE Robotics and Automation Letters 11 (1), pp. 226–233. Cited by: Table 2, Table 3.

[^6]: K. Guo, H. Liu, X. Wu, J. Pan, and C. Lv (2025) Ipad: iterative proposal-centric end-to-end autonomous driving. arXiv preprint arXiv:2505.15111. Cited by: §1, §2.2, Table 2.

[^7]: K. He, X. Zhang, S. Ren, and J. Sun (2016) Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778. Cited by: §4.2.

[^8]: Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, et al. (2023) Planning-oriented autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 17853–17862. Cited by: §1, Table 2, Table 4, Table 5.

[^9]: J. Huang, G. Huang, Z. Zhu, Y. Ye, and D. Du (2021) Bevdet: high-performance multi-camera 3d object detection in bird-eye-view. arXiv preprint arXiv:2112.11790. Cited by: §3.4.

[^10]: X. Jia, Y. Gao, L. Chen, J. Yan, P. L. Liu, and H. Li (2023) Driveadapter: breaking the coupling barrier of perception and planning in end-to-end autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 7953–7963. Cited by: Table 4, Table 5.

[^11]: X. Jia, P. Wu, L. Chen, J. Xie, C. He, J. Yan, and H. Li (2023) Think twice before driving: towards scalable decoders for end-to-end autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21983–21994. Cited by: Table 4, Table 5.

[^12]: X. Jia, Z. Yang, Q. Li, Z. Zhang, and J. Yan (2024) Bench2drive: towards multi-ability benchmarking of closed-loop end-to-end autonomous driving. Advances in Neural Information Processing Systems 37, pp. 819–844. Cited by: §1, §4.1.

[^13]: X. Jia, J. You, Z. Zhang, and J. Yan (2025) Drivetransformer: unified transformer for scalable end-to-end autonomous driving. arXiv preprint arXiv:2503.07656. Cited by: Table 4, Table 5.

[^14]: B. Jiang, S. Chen, Q. Xu, B. Liao, J. Chen, H. Zhou, Q. Zhang, W. Liu, C. Huang, and X. Wang (2023) Vad: vectorized scene representation for efficient autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 8340–8350. Cited by: §1, Table 4, Table 5.

[^15]: Y. Lee, J. Hwang, S. Lee, Y. Bae, and J. Park (2019) An energy and gpu-computation efficient backbone network for real-time object detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, pp. 0–0. Cited by: §4.3.

[^16]: K. Li, Z. Li, S. Lan, Y. Xie, Z. Zhang, J. Liu, Z. Wu, Z. Yu, and J. M. Alvarez (2025) Hydra-mdp++: advancing end-to-end driving via expert-guided hydra-distillation. arXiv preprint arXiv:2503.12820. Cited by: §1, §2.1, Table 2, Table 3, Table 3.

[^17]: Y. Li, Y. Wang, Y. Liu, J. He, L. Fan, and Z. Zhang (2025) End-to-end driving with online trajectory evaluation via bev world model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 27137–27146. Cited by: Table 2.

[^18]: Z. Li, K. Li, S. Wang, S. Lan, Z. Yu, Y. Ji, Z. Li, Z. Zhu, J. Kautz, Z. Wu, et al. (2024) Hydra-mdp: end-to-end multimodal planning with multi-target hydra-distillation. arXiv preprint arXiv:2406.06978. Cited by: Table 1, Table 1, §1, §1, §1, §2.1, §3.5, Table 2, Table 2, §4.2.

[^19]: Z. Li, S. Wang, S. Lan, Z. Yu, Z. Wu, and J. M. Alvarez (2025) Hydra-next: robust closed-loop driving with open-loop training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 27305–27314. Cited by: Table 4, Table 5.

[^20]: Z. Li, W. Yao, Z. Wang, X. Sun, J. Chen, N. Chang, M. Shen, Z. Wu, S. Lan, and J. M. Alvarez (2025) Generalized trajectory scoring for end-to-end multimodal planning. arXiv preprint arXiv:2506.06664. Cited by: §2.3.

[^21]: Z. Li, W. Wang, H. Li, E. Xie, C. Sima, T. Lu, Q. Yu, and J. Dai (2024) Bevformer: learning bird’s-eye-view representation from lidar-camera via spatiotemporal transformers. IEEE Transactions on Pattern Analysis and Machine Intelligence 47 (3), pp. 2020–2036. Cited by: §3.4.

[^22]: B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang, et al. (2025) Diffusiondrive: truncated diffusion model for end-to-end autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 12037–12047. Cited by: §1, §2.2, Table 2.

[^23]: X. Lin, T. Lin, Z. Pei, L. Huang, and Z. Su (2022) Sparse4d: multi-view 3d object detection with sparse spatial-temporal fusion. arXiv preprint arXiv:2211.10581. Cited by: §3.4.

[^24]: L. Liu, C. Jia, Z. Song, H. Pan, B. Liao, W. Sun, Y. Zhang, L. Yang, and Y. Luo (2025) Fully unified motion planning for end-to-end autonomous driving. arXiv preprint arXiv:2504.12667. Cited by: Table 2.

[^25]: K. Renz, L. Chen, E. Arani, and O. Sinavski (2025) Simlingo: vision-only closed-loop autonomous driving with language-action alignment. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 11993–12003. Cited by: Table 4.

[^26]: Z. Song, L. Liu, H. Pan, B. Liao, M. Guo, L. Yang, Y. Zhang, S. Xu, C. Jia, and Y. Luo (2025) DIVER: reinforced diffusion breaks imitation bottlenecks in end-to-end autonomous driving. arXiv preprint arXiv:2507.04049. Cited by: Table 2.

[^27]: H. Su, W. Wu, and J. Yan (2024) DiFSD: ego-centric fully sparse paradigm with uncertainty denoising and iterative refinement for efficient end-to-end self-driving. arXiv preprint arXiv:2409.09777. Cited by: Table 4.

[^28]: W. Sun, X. Lin, Y. Shi, C. Zhang, H. Wu, and S. Zheng (2025) Sparsedrive: end-to-end autonomous driving via sparse scene representation. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pp. 8795–8801. Cited by: §1, §3.4, Table 4.

[^29]: Y. Tang, Z. Xu, Z. Meng, and E. Cheng (2025) Hip-ad: hierarchical and multi-granularity planning with deformable attention for autonomous driving in a single decoder. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 25605–25615. Cited by: Table 4, Table 5.

[^30]: A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin (2017) Attention is all you need. Advances in neural information processing systems 30. Cited by: §3.4.

[^31]: X. Weng, B. Ivanovic, Y. Wang, Y. Wang, and M. Pavone (2024) Para-drive: parallelized architecture for real-time autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15449–15458. Cited by: Table 2.

[^32]: P. Wu, X. Jia, L. Chen, J. Yan, H. Li, and Y. Qiao (2022) Trajectory-guided control prediction for end-to-end autonomous driving: a simple yet strong baseline. Advances in Neural Information Processing Systems 35, pp. 6119–6132. Cited by: Table 4, Table 5.

[^33]: Z. Xing, X. Zhang, Y. Hu, B. Jiang, T. He, Q. Zhang, X. Long, and W. Yin (2025) Goalflow: goal-driven flow matching for multimodal trajectories generation in end-to-end autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 1602–1611. Cited by: §2.2, Table 2, Table 2.

[^34]: W. Yao, Z. Li, S. Lan, Z. Wang, X. Sun, J. M. Alvarez, and Z. Wu (2025) Drivesuprim: towards precise trajectory selection for end-to-end planning. arXiv preprint arXiv:2506.06659. Cited by: §1, §1, §2.1, §2.2, Table 2, §4.2, Table 3, Table 3.

[^35]: C. Yuan, Z. Zhang, J. Sun, S. Sun, Z. Huang, C. D. W. Lee, D. Li, Y. Han, A. Wong, K. P. Tee, et al. (2024) Drama: an efficient end-to-end motion planner for autonomous driving with mamba. arXiv preprint arXiv:2408.03601. Cited by: Table 2.

[^36]: J. Zhai, Z. Feng, J. Du, Y. Mao, J. Liu, Z. Tan, Y. Zhang, X. Ye, and J. Wang (2023) Rethinking the open-loop evaluation of end-to-end autonomous driving in nuscenes. arXiv preprint arXiv:2305.10430. Cited by: Table 5.

[^37]: W. Zheng, R. Song, X. Guo, C. Zhang, and L. Chen (2024) Genad: generative end-to-end autonomous driving. In European Conference on Computer Vision, pp. 87–104. Cited by: Table 4.

[^38]: J. Zou, S. Chen, B. Liao, Z. Zheng, Y. Song, L. Zhang, Q. Zhang, W. Liu, and X. Wang (2025) DiffusionDriveV2: reinforcement learning-constrained truncated diffusion modeling in end-to-end autonomous driving. arXiv preprint arXiv:2512.07745. Cited by: §1, Table 2.
---
title: "SimScale: Learning to Drive via Real-World Simulation at Scale"
source: "https://arxiv.org/html/2511.23369v3"
author:
published:
created: 2026-10-06
description:
tags:
  - "clippings"
---
Haochen Tian Affiliation: MAIS, Institute of Automation, Chinese Academy of Sciences Affiliation: OpenDriveLab at The University of Hong Kong    Tianyu Li    Haochen Liu Affiliation: Xiaomi EV [https://opendrivelab.com/SimScale](https://opendrivelab.com/SimScale)    Jiazhi Yang Affiliation: OpenDriveLab at The University of Hong Kong    Yihang Qiu Affiliation: OpenDriveLab at The University of Hong Kong    Guang Li Affiliation: Xiaomi EV [https://opendrivelab.com/SimScale](https://opendrivelab.com/SimScale)    Junli Wang Affiliation: MAIS, Institute of Automation, Chinese Academy of Sciences Affiliation: Xiaomi EV [https://opendrivelab.com/SimScale](https://opendrivelab.com/SimScale)    Yinfeng Gao Affiliation: MAIS, Institute of Automation, Chinese Academy of Sciences Affiliation: Xiaomi EV [https://opendrivelab.com/SimScale](https://opendrivelab.com/SimScale)    Zhang Zhang    Liang Wang    Hangjun Ye Affiliation: Xiaomi EV [https://opendrivelab.com/SimScale](https://opendrivelab.com/SimScale)    Tieniu Tan Affiliation: MAIS, Institute of Automation, Chinese Academy of Sciences    Long Chen    Hongyang Li Affiliation: OpenDriveLab at The University of Hong Kong

###### Abstract

Achieving fully autonomous driving systems requires learning rational decisions in a wide span of scenarios, including safety-critical and out-of-distribution ones. However, such cases are underrepresented in real-world corpus collected by human experts. To complement for the lack of data diversity, we introduce a novel and scalable simulation framework capable of synthesizing massive unseen states upon existing driving logs. Our pipeline utilizes advanced neural rendering with a reactive environment to generate high-fidelity multi-view observations controlled by the perturbed ego trajectory. Furthermore, we develop a pseudo-expert trajectory generation mechanism for these newly simulated states to provide action supervision. Upon the synthesized data, we find that a simple co-training strategy on both real-world and simulated samples can lead to significant improvements in both robustness and generalization for various planning methods on challenging real-world benchmarks, up to +8.6 EPDMS on navhard and +2.9 on navtest. More importantly, such policy improvement scales smoothly by increasing simulation data only, even without extra real-world data streaming in. We further reveal several crucial findings of such a sim-real learning system, which we term SimScale, including the design of pseudo-experts and the scaling properties for different policy architectures. Simulation data and code have been released at [https://github.com/OpenDriveLab/  
SimScale](https://github.com/OpenDriveLab/SimScale).

![[Uncaptioned image]](https://arxiv.org/html/2511.23369v3/teaser.png)

Figure 1: Scaling up end-to-end planners by simulation. (a) We construct large-scale simulation data by perturbing ego trajectories, generating corresponding pseudo-expert demonstrations, and rendering multi-view observations in reactive environments. Combined with real-world data, this enables broad coverage of out-of-distribution states and supports sim–real co-training for any end-to-end planner. (b) Across three representative planner families, including regression, diffusion, and vocabulary scoring, sim-real co-training consistently produces synergistic improvements in robustness and generalization, demonstrating clear and predictable simulation scaling trends.

<sup>†</sup>

## 1 Introduction

Data scaling is recognized as a foundational principle in modern deep learning across various domains, including language, vision, and multimodal modeling, underpinning steady performance improvements as data sizes increase [^38] [^5] [^92] [^72]. In autonomous driving, end-to-end (E2E) planning learns to map raw observations to actions, offering a promising way to leverage large-scale driving data to enable the emergence of fully autonomous systems [^29] [^68] [^99].

Nevertheless, real-world driving data from human expert demonstrations are dominated by common scenarios, while non-trivial cases, *e.g*., safety-critical, are underrepresented [^9] [^84] [^62] [^88]. Moreover, planners trained on such data are confined to human driving distribution and struggle to generalize to rare or unseen situations, leading to distribution shift and causal confusion in deployment [^20] [^56]. Consequently, scaling real-world data only is inefficient for achieving deploy-ready autonomous driving.

Simulation via neural rendering [^40] can generate high-fidelity driving scenarios and thus has the potential to produce out-of-distribution (OOD) states deviated from human demonstrations at scale, which is essential for closed-loop planning [^23] [^71]. Therefore, scaling simulation data presents an attractive alternative to solely relying on real-world data. However, planners require feasible corresponding demonstrations to learn how to handle OOD states, while current simulation methods fail to generate such demonstrations effectively. Moreover, the impact of scaling simulation data lacks in-depth analysis. In this work, we aim to derive a systematic recipe for scaling simulation data from limited real-world scenarios in end-to-end planning.

To conduct comprehensive experiments and analyses, this study is structured along three key questions: (1) what constitutes effective simulation data, (2) how well planners benefit from it, and (3) whether this system scales predictably upon fixed real-world corpus.

To this end, we formulate a scalable simulation data generation framework that extends the expert distribution from existing real-world training data to bootstrap end-to-end autonomous driving systems. We develop a 3D Gaussian Splatting (3DGS) [^40] simulation data engine, which allows controlling the temporal ego and other agent states and rendering multi-view videos from the ego’s perspective. Concretely, we first sample a diverse set of plausible perturbations to the ego trajectory, maximizing coverage of the state space, *e.g*., off-center lane drifts, close interactions, among others. Then, we take the final state of each perturbed trajectory as the perturbed state and generate corresponding demonstrations using pseudo-experts of two forms in comparison. The first, recovery-based expert retrieves trajectories that steer the policy within the human trajectory manifold, resulting in human-like yet cautious behaviors. The second, privileged planner-based expert [^16] generates the trajectory that maximizes optimality, representing an exploratory strategy with lower realism. To enhance scalability and plausibility, the entire pipeline is executed in a reactive environment [^76], where surrounding agents interact with the ego vehicle responsively.

To thoroughly assess the effect of simulation data, we consider three types of end-to-end planners with various model scales, *i.e*., LTF [^12] for regression methods, DiffusionDrive [^60] for diffusion-based planners, and GTRS-Dense [^59] for vocabulary scoring. We employ a simple yet effective sim-real co-training strategy [^67] [^69] to maintain human driving distribution while mitigating visual domain degradation. Furthermore, by keeping a constant amount of real data and progressively adding simulation data via non-overlapping samples, we investigate how diverse planners can benefit from simulation data and the general scaling property. We utilize two real-world closed-loop benchmarks to evaluate the planners from multiple perspectives. navhard [^6] focuses on unseen, challenging scenarios to assess the impact of OOD states on planners, while navtest [^17] offers a broad set of diverse scenarios to test planners’ ability to handle varying situations.

The complete sim-real learning system, which comprises a scalable simulation data construction pipeline and an effective sim–real co-training strategy as shown in Fig. 1, is termed SimScale. Rigorous experiments uncover crucial findings enabled by SimScale—including, but not limited to, the following:

- Scalable simulation with pseudo-expert unlocks the inherent potential of available real-world driving data.
- Sim-real co-training improves both robustness and generalization synergistically in diverse end-to-end planners.
- Exploratory expert and interaction environments improve the effectiveness of simulation data.
- Planners with multimodal modeling capabilities exhibit more encouraging data scaling properties.

![[pipeline 2.png|Refer to caption]]

Figure 2: Pseudo-expert scene simulation pipeline. (a) Trajectory perturbation on T to + H T+H, (b) reactive environment rollout, and pseudo-expert trajectory generation from 2 T+2H under recovery-based and planner-based strategies.

## 2 Methodology

We outline SimScale as follows. In Sec. 2.2, we briefly introduce our 3GDS simulation data engine supporting controllable multi-view video rendering. In Sec. 2.3, we propose a pseudo-expert scene simulation pipeline to generate diverse simulation data containing OOD states with feasible demonstrations. In Sec. 2.4, we demonstrate a scalable sim-real co-training method with different end-to-end planners.

### 2.1 Preliminary

End-to-end planning models take observations within a history window as input and output a predicted future trajectory. Each training scenario starts at a selected timestep and includes a history horizon of length $T$ and a planning horizon of length $H$. The model processes the past $T$ frames to predict the future $H$ frames, resulting in a complete training sample spanning $T+H$ timesteps.

### 2.2 3DGS Simulation Data Engine

To reduce the domain gap between real-world data and generated observations from novel views in simulated scenarios, a photorealistic data engine is required. Built upon 3DGS [^40] assets reconstructed from real-world datasets, our data engine $\Phi(K_{t},E_{t},\left\{x_{i,t},y_{i,t},\theta_{i,t}\right\}_{i=1}^{N})$ takes as input the camera intrinsics $K_{t}$ and extrinsics $E_{t}$ at timestep $t$, along with the positions and yaw angles $(x,y,\theta)$ of a non-ego vehicle $i$ at the same timestep, and renders the corresponding RGB observations from the camera. The camera extrinsics $E_{t}$ can easily be obtained from the ego vehicle’s position and heading $(x_{0,t},y_{0,t},\theta_{0,t})$ with the ego-to-camera transformation, while the other camera parameters are directly adopted from the raw dataset.

Preprocessing. Following prior work [^51], we apply exposure alignment across images captured simultaneously by multi-view cameras at time $t$, using projected LiDAR points at $t$ as guidance. Moreover, with help of 3D bounding boxes provided by NAVSIM annotations, colored LiDAR points are divided into several groups, *i.e*., the static background and multiple vehicles, and further used as initialization for Gaussians to improve the reconstruction performance.

Block-wise Reconstruction. During scene reconstruction, increasing the number of images significantly raises computational cost and runtime. Thus, we perform reconstruction in a block-wise manner, where each block corresponds to a spatio-temporal range. Following previous work [^86] [^51], backgrounds and foregrounds are reconstructed as separate models using the per-timestep locations and orientations of 3D bounding boxes, resulting in a static background asset and multiple movable vehicle assets. These assets could then be placed at desired locations and headings specified by inputs, rendering sensor data at novel views. Blocks with low average PSNR in novel view synthesis are excluded to ensure the quality of generated simulation data.

### 2.3 Pseudo-Expert Scene Simulation

Based on the data engine, we design a pseudo-expert scene simulation pipeline to generate diverse simulation data from existing real-world data, as shown in Alg. 1. The pipeline aims to produce feasible demonstrations from perturbed states paired with expert trajectories, as illustrated in Fig. 2.

Reactive Scene Reconstruction. For each training clip $d$, we perform two simulations of duration $H$: one for exploring perturbed states at $t=T$, and the other for generating expert trajectories at $t=T+H$. In each simulation, ego trajectory $\tilde{a}_{t:t+H}$ is simulated by LQR [^43], while other agents are modeled using the IDM [^76] to interact with the ego, producing the corresponding future states $\tilde{s}_{t:t+H}$. For the valid expert trajectories, the simulated ego–agent trajectories cross two stage $(\tilde{a}_{T:T+2H},\tilde{s}_{T:T+2H})$ are then rendered into multi-view videos $\tilde{o}_{T:T+2H}$ using the data engine $\Phi$. This decoupling of behavior simulation from sensor rendering enables reactive environments where other agents respond plausibly to the ego’s behaviors, thereby enhancing both the realism and diversity of the simulation data.

Trajectory Perturbation. At the first simulation step ($t=T$), we perturb the ego trajectory $\tilde{a}_{t:t+H}$ so that the ego reaches a new terminal state at $t=T+H$, which then serves as the start state for the next rollout (Fig. 2 (a)). Our objective is to sample diverse yet plausible states. For diversity, perturbations are drawn from a clustered human-trajectory vocabulary that densely covers the action space. For plausibility, we restrict perturbations to remain near human behavior by thresholding longitudinal/lateral shifts $(r_{\mathrm{lon}},r_{\mathrm{lat}})$ and heading change $|\Delta\theta|$, and by removing trajectories that are physically invalid (collisions, off-road, unstable renderings). We further apply spatially sparse sampling on trajectory endpoints using an interleaved grid with steps $(\delta_{\mathrm{lon}},\delta_{\mathrm{lat}})$ to promote uniform coverage. Since reactive simulation is costly, we first filter infeasible trajectories non-reactively and run reactive checks only on the sparsed set. This produces a set of dynamically and physically feasible perturbations $a_{\mathrm{per}}\in\mathcal{A}_{\mathrm{per}}$.

Algorithm 1 Pseudo-Expert Scene Simulation

Training clip $d=(o_{t},s_{t},{a}_{t})^{T+2H}_{t=0}\in\mathcal{D}$, perturb action sets $\mathcal{A}_{\text{per}}$, sensor engine $\Phi$, pseudo policy $\pi_{\text{exp}}$, reward functions $f$, reactive simulation $\mathcal{T}$.

Initialize simulation scene dataset $\mathcal{D}_{\text{sim}}\leftarrow\emptyset$

for each perturbed action $a_{\text{per}}\in\mathcal{A}_{\text{per}}$ do

  for timestep $t\in\{T,T+H\}$ do

   1.Pseudo Action Generation:

   if $t=T$ then $\tilde{a}_{t:t+H}\leftarrow a_{\text{per}};$

   else $\tilde{a}_{t:t+H}\leftarrow\pi_{\text{exp}}(\tilde{o}_{t},\tilde{s}_{t})$.

   end if

   2.Reactive State Simulation:

    $\tilde{s}_{t:t+H}\leftarrow\mathcal{T}(\tilde{a}_{t:t+H},\tilde{s}_{t}).$

   3.Sensor Simulation:

    $\tilde{o}_{t:t+H}\leftarrow\Phi(\tilde{s}_{t:t+H}).$

   4.Reward Filtering / Labeling:

    $\tilde{r}_{t:t+H}\leftarrow f(\tilde{a}_{t:t+H},\tilde{s}_{t:t+H}).$

   if $\textbf{{Filtered}}(\tilde{r}_{t:t+H},\tilde{o}_{t:t+H})$ then

     Break

   end if

  end for

  Append $(\tilde{o}_{t},\tilde{s}_{t},\tilde{a}_{t},\tilde{r}_{t})^{T+2H}_{t=0}$ to $\mathcal{D}_{\text{sim}}$

end for

return $\mathcal{D}_{\text{sim}}$

Pseudo-Expert Trajectory Generation. At the second simulation $t=T+H$, for each perturbed state, we employ a non-human expert, referring as the pseudo-expert $\pi_{\text{exp}}$, to generate a feasible corresponding trajectory $\tilde{a}_{t:t+H}$. Since the pseudo-expert is not perfect and $\tilde{a}_{t:t+H}$ will be used as supervision, we apply stricter filtering in the second stage simulation. In addition to physical constraints, traffic rules and vehicle kinematic limits are also enforced, following Eq. 6. To investigate which strategy best serves as supervision for end-to-end planners, we compared two strategies: a conservative recovery-based expert, and an exploratory planner-based expert, as detailed below.

(1) Recovery-based Expert. Recovery steers the policy within the human manifold after perturbation. To ensure robustness, our recovery expert $\pi_{\text{exp}}$ retrieves a human trajectory from a large vocabulary $\mathcal{V}_{\text{h}}$ that best matches the ego vehicle’s logged state at time $t=T+2H$ (Fig. 2 (b1)). For each candidate trajectory of horizon $H$, we summarize its initial and final poses with a compact matching vector:

$$
\mathbf{m}=[\tilde{v}^{x}_{t},\tilde{v}^{y}_{t},\tilde{\theta}_{t},\tilde{x}_{t+H-1},\tilde{y}_{t+H-1},\tilde{\theta}_{t+H-1}].
$$

Given the ego’s perturbed state with target vector $\mathbf{m}_{r}$, $\pi_{\text{exp}}$ retrieves the closest human maneuver by:

$$
\tilde{a}_{t:t+H}=\arg\min_{a\in\mathcal{V}_{\text{h}}}\|\mathbf{m}(a)-\mathbf{m}_{\text{r}}\|_{1}.
$$

This yields a human-like yet conservative fallback behavior, stabilizing under distributional drifts.

(2) Planner-based Expert. Following prior works [^97] [^3], we use a privileged planner $\mathbf{P}$ that leverages ground-truth states to generate reactive and optimized trajectory rollouts in simulation, as shown in Fig. 2 (b2). The planner-based expert $\pi_{\text{exp}}\leftarrow\mathbf{P}$ is defined as: $\tilde{a}_{t:t+H}=\mathbf{P}(\tilde{s}_{t:t+H}).$ Compared with the recovery policy, planner-based expert relies on rules or cost heuristics, trading human-likeness and realism of its behavior occasionally. Still, it offers strong optimization and diverse exploratory rollouts, enriching expert supervision beyond human data.

### 2.4 Scalable Sim-Real Co-training

Strategies for Co-training. Sim-real co-training serves as a simple yet effective strategy [^69] [^67] [^4] [^88] that enables the integration of both real and simulated data for planning. In our approach, we randomly sample from a mixture of real-world data $\mathcal{D}$ and simulation data $\mathcal{D}_{\text{sim}}$ during training, aiming to preserve the human driving distribution while mitigating visual domain degradation caused by potential simulation artifacts, such as subtle rendering inconsistencies, temporal jitter, or unrealistic lighting and shadows [^64] [^63]. Our fully automated, scalable simulation data generation framework enables scaling the total training data by progressively adding non-overlapping simulation samples, while keeping the real data amount fixed.

Planners for Co-training. We aim to comprehensively evaluate the effectiveness of simulation data for end-to-end planners. Modern end-to-end planning approaches can be broadly categorized into three representative paradigms: regression-based planners [^27] [^12], diffusion-based planners [^60] [^83], and vocabulary scoring-based planners [^33] [^56]. Accordingly, we consider representative models from each paradigm in our co-training experiments.

(1) Co-training with Pseudo-Expert Trajectory. Regression and diffusion-based planners rely on expert demonstrations. The co-training process can thus be formulated as:

$$
\arg\min_{\theta}\mathbb{E}_{(a,o)\sim(\mathcal{D}\,\cup\,\mathcal{D}_{\text{sim}})}\Big[\mathcal{L}_{\text{im}}\big(a,\pi_{\theta}(\hat{a}|o)\big)\Big],
$$

here, $\mathcal{L}_{\text{im}}$ denotes the imitation loss; $\mathcal{D}$ and $\mathcal{D}_{\text{sim}}$ represent the real-world and generated simulation datasets, respectively; and $A$ refers to expert trajectories, i.e., human-expert trajectories in $\mathcal{D}$ and pseudo-expert trajectories in $\mathcal{D}_{\text{sim}}$. As for the vocabulary scoring-based planner, the learning objective has additional prediction of reward signals $r$ that distill the evaluation metrics, *e.g*. EPDMS as Eq. 6:

$$
\displaystyle\arg\min_{\theta}\mathbb{E}_{(a,o,r)\sim(\mathcal{D}\cup\mathcal{D}_{\text{sim}})}\Big[\lambda\mathcal{L}_{\text{im}}+\mathcal{L}_{\text{r}}\big(r,\,\pi_{\theta}(\hat{a}|o)\big)\Big],
$$

here, $\mathcal{L}_{\text{r}}$ denotes reward loss, $\lambda$ is a weighting factor.

![[data.png|Refer to caption]]

Figure 3: Simulation data statistics across multiple sampling rounds. (a) Recovery-based expert impose stronger constraints, leading to slower data accumulation than (b) Planner-based expert.

(2) Co-training with Rewards Only. For the vocabulary scoring-based planner, when the reward signal is well-aligned, the expert is theoretically unnecessary, as shown in Eq. 4. The planner can explore directions to increase the reward without being restricted to a single expert trajectory. The co-training process can be formulated as:

$$
\arg\min_{\theta}\mathbb{E}_{(a,o,r)\sim\mathcal{D}}[\lambda\mathcal{L}_{\text{im}}+\mathcal{L}_{r}]+\mathbb{E}_{(o,r)\sim\mathcal{D}_{\text{sim}}}[\mathcal{L}_{r}].
$$

We thus investigate purely reward-driven optimization for the scoring-based planner in simulation, to assess its potential for fully leveraging simulation data, and assess its potential for maximizing the utility of simulated data.

## 3 Experiments

### 3.1 Setup and Protocols.

Real-world Datasets. We use the navtrain split from NAVSIM [^17], built upon the largest publicly available annotated driving dataset nuPlan [^39], and comprises 100K interactive real-world scenarios.

Simulation Data Curation. We construct simulated scenarios on navtrain split. 3DGS models of blocks with $\mathrm{PSNR}<27$ in novel-view results are removed ensure reconstruction quality. For ego trajectory perturbation, we choose the plausible candidates from the vocabulary of 16,384 trajectories from [^59], with $\mathrm{EPDMS}\geq 0.8$ and relative heading $|\Delta\theta|\leq 20^{\circ}$, where $\mathrm{EPDMS}$ is detailed in Eq. 6. During spatial sampling by trajectory endpoint, the longitudinal and lateral perturbation ranges are set as $r_{\mathrm{lon}}=20\mathrm{m},r_{\mathrm{lat}}=2.0\mathrm{m}$, with step sizes $\delta_{\mathrm{lon}}=5\mathrm{m},\delta_{\mathrm{lat}}=0.5\mathrm{m}$. For recovery-based trajectory generation, we use all human expert trajectories in navtrain split as the retrieval vocabulary. For planner-based trajectory generation, we select the SOTA rule-based planner in nuPlan [^39], PDM-Closed [^16] as the expert. Only pseudo-expert trajectories with decent sub-metrics of $\mathrm{EPDMS}$ are considered valid. Due to computational limits, for each type of pseudo-expert, we perform up to five random samplings of valid trajectories with no overlaps from navtrain split for rendering. Through multiple samplings, expert trajectory demonstrations are progressively accumulated. As shown in Fig. 3, the two pseudo-expert methods exhibit different success rates, with the planner-based expert achieving higher efficiency. In total, we generate 147K recovery-based simulation scenes and 237K planner-based simulation scenes. For more details, please refer to Sec. B.1 in the supplementary

Table 1: Performance on the NAVSIM-v2 navhard Leaderboard. PDM-Closed uses ground-truth symbolic inputs for planning, while other methods rely on sensor data. ($*$: pseudo-expert supervision; $\dagger$: reward scoring; S.: per-stage EPDM score.)

<table><tbody><tr><td>Method</td><td>Backbone</td><td>Sim.</td><td>Stage</td><td>NC↑</td><td>DAC↑</td><td>DDC↑</td><td>TLC↑</td><td>EP↑</td><td>TTC↑</td><td>LK↑</td><td>HC↑</td><td>EC↑</td><td>S.↑</td><td>EPDMS↑</td></tr><tr><td>PDM-Closed <sup><a href="#fn:16">16</a></sup></td><td>-</td><td>-</td><td>S 1 S 2</td><td>94.4 88.1</td><td>98.8 90.6</td><td>100 96.3</td><td>99.5 98.5</td><td>100 100</td><td>93.5 83.1</td><td>99.3 73.7</td><td>87.7 91.5</td><td>36.0 25.4</td><td>- -</td><td>51.3</td></tr><tr><td colspan="15">Regeression-based Planner</td></tr><tr><td rowspan="2">LTF <sup><a href="#fn:12">12</a></sup></td><td rowspan="3">ResNet34</td><td>w/o</td><td>S 1 S 2</td><td>97.3 79.4</td><td>80.2 69.0</td><td>97.8 85.6</td><td>99.3 98.5</td><td>83.4 83.8</td><td>96.2 76.7</td><td>92.9 47.9</td><td>97.8 97.0</td><td>71.1 70.6</td><td>61.3 39.2</td><td>24.4</td></tr><tr><td>w/ <sup>∗</sup></td><td>S 1 S 2</td><td>96.1 85.5</td><td>85.3 66.9</td><td>99.4 91.6</td><td>99.3 99.1</td><td>84.7 93.0</td><td>94.7 81.1</td><td>93.6 58.3</td><td>97.6 95.1</td><td>77.3 42.9</td><td>66.3+5.0 44.8+5.6</td><td>30.2 ↑24%</td></tr><tr><td colspan="15">Diffusion-based Planner    </td></tr><tr><td rowspan="2">DiffusionDrive <sup><a href="#fn:60">60</a></sup></td><td rowspan="3">ResNet34</td><td>w/o</td><td>S 1 S 2</td><td>96.8 80.1</td><td>86.0 72.8</td><td>98.8 84.4</td><td>99.3 98.4</td><td>84.0 85.9</td><td>95.8 76.6</td><td>96.7 46.4</td><td>97.6 96.3</td><td>79.6 72.8</td><td>66.7 40.5</td><td>27.5</td></tr><tr><td>w/ <sup>∗</sup></td><td>S 1 S 2</td><td>97.4 86.4</td><td>88.7 72.1</td><td>99.3 92.9</td><td>99.3 98.5</td><td>82.8 92.1</td><td>96.9 80.6</td><td>98.0 60.8</td><td>97.3 95.4</td><td>59.6 31.9</td><td>67.5+0.8 46.8+6.3</td><td>32.6 ↑19%</td></tr><tr><td colspan="15">Scoring-based Planner      </td></tr><tr><td rowspan="8">GTRS-Dense <sup><a href="#fn:59">59</a></sup></td><td rowspan="2">ResNet34</td><td>w/o</td><td>S 1 S 2</td><td>99.3 92.8</td><td>96.6 88.6</td><td>99.6 95.5</td><td>100 99.4</td><td>57.4 55.9</td><td>99.5 91.3</td><td>92.6 55.7</td><td>89.5 91.1</td><td>16.4 35.7</td><td>67.1 55.8</td><td>38.3</td></tr><tr><td>w/ <sup>†</sup></td><td>S 1 S 2</td><td>97.6 94.3</td><td>96.4 92.7</td><td>99.3 95.1</td><td>100 99.5</td><td>75.7 80.2</td><td>97.8 91.5</td><td>93.3 56.2</td><td>97.3 90.6</td><td>32.9 28.3</td><td>72.4+5.3 63.4+7.6</td><td>46.9 ↑22%</td></tr><tr><td rowspan="4">V2-99</td><td>w/o</td><td>S 1 S 2</td><td>98.9 89.9</td><td>94.9 90.5</td><td>99.1 94.1</td><td>100 99.3</td><td>76.1 77.6</td><td>98.4 88.5</td><td>93.8 56.0</td><td>94.9 92.0</td><td>37.8 30.2</td><td>70.4 58.5</td><td>41.9</td></tr><tr><td>w/ <sup>†</sup></td><td>S 1 S 2</td><td>99.3 94.9</td><td>97.6 94.3</td><td>99.4 94.5</td><td>100 99.3</td><td>71.5 79.0</td><td>99.6 92.6</td><td>96.0 57.8</td><td>95.8 93.4</td><td>30.7 30.9</td><td>72.5+2.1 65.4+6.9</td><td>48.0 ↑15%</td></tr></tbody></table>

Table 2: Performance on the NAVSIM-v2 navtest Leaderboard. ($*$: pseudo-expert supervision; $\dagger$: reward scoring.)

<table><tbody><tr><td>Method</td><td>Backbone</td><td>Sim.</td><td>NC↑</td><td>DAC↑</td><td>DDC↑</td><td>TLC↑</td><td>EP↑</td><td>TTC↑</td><td>LK↑</td><td>HC↑</td><td>EC↑</td><td>EPDMS↑</td></tr><tr><td>Human Agent</td><td>-</td><td>-</td><td>100</td><td>100</td><td>99.8</td><td>100</td><td>87.4</td><td>100</td><td>100</td><td>98.1</td><td>90.1</td><td>90.3</td></tr><tr><td colspan="13">Regression-based Planner</td></tr><tr><td rowspan="2">LTF <sup><a href="#fn:12">12</a></sup></td><td rowspan="2">ResNet34</td><td>w/o</td><td>97.7</td><td>94.0</td><td>99.3</td><td>99.8</td><td>87.2</td><td>96.7</td><td>95.5</td><td>98.3</td><td>82.9</td><td>81.5</td></tr><tr><td>w/ <sup>∗</sup></td><td>98.3</td><td>95.6</td><td>99.6</td><td>99.8</td><td>87.1</td><td>97.5</td><td>97.2</td><td>98.3</td><td>88.2</td><td>84.4 +2.9</td></tr><tr><td colspan="13">Diffusion-based Planner    </td></tr><tr><td rowspan="2">DiffusionDrive <sup><a href="#fn:60">60</a></sup></td><td rowspan="2">ResNet34</td><td>w/o</td><td>98.4</td><td>95.5</td><td>99.5</td><td>99.8</td><td>87.5</td><td>97.5</td><td>96.9</td><td>98.4</td><td>87.7</td><td>84.2</td></tr><tr><td>w/ <sup>∗</sup></td><td>98.5</td><td>97.1</td><td>99.6</td><td>99.8</td><td>87.4</td><td>97.8</td><td>98.0</td><td>98.3</td><td>87.5</td><td>85.9 +1.7</td></tr><tr><td colspan="13">Scoring-based Planner      </td></tr><tr><td rowspan="4">GTRS-Dense <sup><a href="#fn:59">59</a></sup></td><td rowspan="2">ResNet34</td><td>w/o</td><td>97.6</td><td>97.5</td><td>99.0</td><td>99.9</td><td>87.9</td><td>97.0</td><td>95.9</td><td>97.5</td><td>55.9</td><td>82.3</td></tr><tr><td>w/ <sup>†</sup></td><td>98.4</td><td>98.8</td><td>99.4</td><td>99.9</td><td>88.0</td><td>98.0</td><td>96.5</td><td>97.6</td><td>58.0</td><td>84.6 +2.3</td></tr><tr><td rowspan="2">V2-99</td><td>w/o</td><td>97.6</td><td>98.5</td><td>99.5</td><td>99.9</td><td>89.5</td><td>97.2</td><td>96.8</td><td>97.2</td><td>57.2</td><td>84.0</td></tr><tr><td>w/ <sup>†</sup></td><td>97.6</td><td>98.9</td><td>99.6</td><td>99.9</td><td>89.9</td><td>97.1</td><td>97.2</td><td>97.7</td><td>58.7</td><td>84.6 +0.8</td></tr></tbody></table>

Benchmark and Metrics. We utilize two benchmarks in NAVSIMv2 [^6] for end-to-end model evaluation, including navhard and navtest. navhard is the official two-stage evaluation benchmark, which contains 244 challenging real-world scenarios in the first stage and corresponding 4,164 synthetic scenarios generated by 3DGS in the second stage. navtest is a one-stage evaluation benchmark, containing a large number of 12,146 real-world scenarios. navhard focuses on assessing the model’s closed-loop performance in safety-critical situations, while navtest emphasizes generalization across diverse driving conditions. The two benchmarks share a rule-based planning metric, $\mathrm{EPDMS}$ [^47], with several sub-metrics:

$$
\mathrm{EPDMS}=\underbrace{\left(\prod_{m\in\mathcal{M}_{\text{pen}}}S_{m}\right)}_{\text{penalties}}\cdot\underbrace{\left(\frac{\sum_{m\in\mathcal{M}_{\text{avg}}}w_{m}S_{m}}{\sum_{m\in\mathcal{M}_{\text{avg}}}w_{m}}\right)}_{\text{weighted average}},
$$

where $S_{m}$ is the sub-metric: penalty terms set $\mathcal{M}_{\text{pen}}$ includes No-at-fault Collisions (NC), Drivable Area Compliance (DAC), Driving Direction Compliance (DDC), and Traffic Light Compliance (TLC); weighted average terms set $\mathcal{M}_{\text{avg}}$ includes Time-to-Collision (TTC), Ego Progress (EP), Lane Keeping (LK), History Comfort (HC), and extended comfort (EC). Note that $\mathrm{EPDMS}$ in navhard further incorporates several modifications, *e.g*., two-stage aggregation, reactive traffic simulation, and the exclusion of penalties in cases where the human expert driver also fails.

Models and Training. To thoroughly validate our model-agnostic simulation data, we select one representative open-source model for each paradigm discussed in Sec. 2.4: the regression-based LTF [^12], the diffusion-based DiffusionDrive [^60], and the scoring-based GTRS-Dense [^59]. We adopt their official implementations with two modifications: the input image resolution is unified to $2048\times 512$, and the LiDAR inputs are removed to align with the navhard evaluation setting. All models are trained from scratch on NVIDIA H20-3e GPUs. To ensure fair comparison, each model follows the same training strategy, whether using only the navtrain split or augmented with simulation data. For more details, please refer to Sec. B.2 in supplementary.

### 3.2 Leaderboard Results

Tab. 1 and Tab. 2 present the leaderboard results of SimScale sim-real co-training for the three planner paradigms on navhard and navtest, respectively.

Navhard Leaderboard. All models exhibit significant improvements in both Stage 1 and Stage 2. Notably, GTRS-Dense (V2-99) achieves a score of 48.0, establishing a new SOTA on navhard. These results demonstrate that incorporating simulation data with an extended distribution substantially enhances model robustness in challenging and unseen environments. Notably, weaker baseline models, *i.e*., LTF and DiffusionDrive, benefit the most, with gains exceeding 20%, indicating that sim-real co-training with simulation data effectively enables models to exploit dataset knowledge better and unlock their latent learning potential.

![[curve_scaling.png|Refer to caption]]

Figure 4: Scaling dynamics across different planners and pseudo-expert trajectories. We visualize how simulation data scale and supervision signals influence the driving performance of various planners, where the infection point indicates learning plateau.

![[vis_scene.png|Refer to caption]]

Figure 5: Qualitative results of the simulation scenes on navtrain. Four representative simulation scenarios are shown, each mirroring a typical real-world OOD scene, with synthetic front-view and auxiliary key-view images provided.

Navtest Leaderboard. All models show consistent improvements of up to +2.9 points, demonstrating stronger performance under large-scale and diverse conditions. The quantitative results above highlight that our simulation data is model-agnostic, and that our general sim-real co-training achieves a synergistic optimization of robustness and generalization, which is essential for reliable closed-loop deployment in the real world [^9] [^62].

### 3.3 Ablation and Data Scaling Analysis

Data Scaling Curves of Different Planners. Due to the lack of prior work studying the scaling behavior of simulation data under a fixed amount of real-world data, we model the relationship between performance and the total data size (simulation + real) using a log-quadratic function:

$$
S(N)=a\log^{2}(N)+b\log(N)+c,
$$

where $S(N)$ denotes the planner performance with total data size $N$, and $a$, $b$, $c$ are parameters fitted via nonlinear least squares:

$$
(a^{*},b^{*},c^{*})=\arg\min_{a,b,c}\sum_{i=1}^{M}\big(S_{i}-S(N_{i};a,b,c)\big)^{2},
$$

where $S_{i}$ is the observed performance at total data size $N_{i}$, and $M$ is the number of data points. If the data scaling trend exists, the quadratic coefficient $a$ approaches zero, degenerating the model to a linear log relation; otherwise, the curve exhibits a parabolic shape with a visible saturation point. We evaluate four planners under 2 pseudo-experts following simulation data scaling in Fig. 3 and select $\mathrm{EPDMS}$ in navhard as $S$. The scaling curves and fitted log-quadratic functions are shown in Fig. 4, with the residual standard deviation depicted as an error band. Additionally, we conduct extra experiments using reward scoring only in planner-based simulation data for GTRS-Dense, proposed in Sec. 2.4. Some classical trends can be observed in Fig. 4. For instance, comparing planner-based and reward-scoring settings in Fig. 4 (c) and (d), larger models exhibit more pronounced data scaling trends under the same amount of data. Additional interesting and meaningful observations are highlighted below.

Pseudo-Expert Should Be Exploratory. For all planners, the scaling curves under recovery-based setting converge earlier and achieve lower performance compared to planner-based. The recovery-based expert always steers to the human log, limiting diversity as the simulation data grows from the same real scenarios. In contrast, the planner-based expert explores a broader range of possibilities and even produces feasible solutions in challenging situations. Consequently, the recovery-based expert only exhibits advantages in small-data regimes (Fig. 4 (d)), likely because its trajectory distribution is human-like and better aligned with real-world data, making it easier to learn. In most cases, data scaling yields diminishing returns for recovery-based methods compared to planner-based ones, and performance may even degrade as visual artifacts accumulate and begin to dominate. These observations highlight the importance of exploratory behavior for pseudo-experts, which enhances the value of simulation data under scaling.

Multi-modality Modeling Sparks the Scaling. Although the regression-based LTF and diffusion-based DiffusionDrive have comparable model sizes ($56\text{M}$ vs $61\text{M}$), they exhibit markedly different scaling properties for the planner-based in Fig. 4 (a) and (b). For LTF, performance saturates and starts to degrade when the simulation-to-real ratio reaches 1:1, whereas DiffusionDrive exhibits an approximately linear improvement. This is due to the gradually increasing diversity of demonstrations from the same real scenario, which introduces an effectively multi-modal supervision problem. Single-mode regression struggles to model multi-peak distributions, leading to mode confusion and performance degradation, while diffusion models can capture multimodality, making them amenable to optimization under diverse supervision [^11]. Since real-world autonomous driving is inherently a multi-peak problem [^33], our simulation-scaling results underscore the importance of multimodal modeling for scalable real-world end-to-end autonomous driving.

Table 3: The effect of expert with simulated reward scoring on navhard using GTRS-Dense. ($\textbf{S}_{1/2}$:per-stage EPDM scores.)

<table><tbody><tr><td>Backbone</td><td>Real</td><td>Expert</td><td>Sim.</td><td>Expert</td><td><math><semantics><mrow><msub><mtext>S</mtext> <mn>1</mn></msub> <mo>↑</mo></mrow> <annotation>\textbf{S}_{1}\uparrow</annotation></semantics></math></td><td><math><semantics><mrow><msub><mtext>S</mtext> <mn>2</mn></msub> <mo>↑</mo></mrow> <annotation>\textbf{S}_{2}\uparrow</annotation></semantics></math></td><td><math><semantics><mrow><mtext>EPDMS</mtext> <mo>↑</mo></mrow> <annotation>\textbf{EPDMS}\uparrow</annotation></semantics></math></td></tr><tr><td rowspan="4">ResNet34</td><td>✓</td><td>✓</td><td></td><td></td><td>67.1</td><td>55.8</td><td>38.3</td></tr><tr><td>✓</td><td></td><td></td><td></td><td>66.8</td><td>54.9</td><td>37.6</td></tr><tr><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>70.8</td><td>63.9</td><td>46.1</td></tr><tr><td>✓</td><td>✓</td><td>✓</td><td></td><td>72.4</td><td>63.4</td><td>45.9</td></tr><tr><td rowspan="4">V2-99</td><td>✓</td><td>✓</td><td></td><td></td><td>70.4</td><td>58.5</td><td>41.9</td></tr><tr><td>✓</td><td></td><td></td><td></td><td>68.1</td><td>55.6</td><td>38.8</td></tr><tr><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>71.7</td><td>65.5</td><td>47.7</td></tr><tr><td>✓</td><td>✓</td><td>✓</td><td></td><td>72.5</td><td>65.4</td><td>48.0</td></tr></tbody></table>

Reward is All You Need. In Fig. 4 (c) and (d), for the scoring-based GTRS-Dense planner, reward signals alone, without expert trajectories in simulation, yields even better performance. To further analyze this, we conduct reward-only training on real-world data only, which instead leads to performance degradation, as shown in Tab. 3. These results indicate that with sufficient expert supervision to stabilize the optimization direction, the reward guidance is better. The model benefits from the feedback from rewards during its exploration and interaction within the environment [^14].

Effect of Reactive Simulation.

Table 4: The effect between non-reactive *vs*. reactive data simulation on navhard using GTRS-Dense, across sampling rounds.

<table><tbody><tr><td rowspan="2">Type</td><td rowspan="2">#Round</td><td rowspan="2">#Sim.</td><td colspan="2">EPDMS↑</td></tr><tr><td>ResNet34</td><td>V2-99</td></tr><tr><td>Non-Reactive</td><td>2</td><td>141K</td><td>43.7</td><td>45.6</td></tr><tr><td>Reactive</td><td>2</td><td>120K</td><td>44.4</td><td>46.7</td></tr><tr><td>Reactive</td><td>3</td><td>167K</td><td>45.0</td><td>47.9</td></tr></tbody></table>

To isolate the effect of reactive traffic, we compare reward-scoring GTRS-Dense on navhard using non-reactive vs. reactive simulation data (Tab. 4). Two rounds of non-reactive sampling yield 141K trajectories, *i.e*., 21K more valid samples due to fewer collision rates, yet provide no performance improvement in $\mathrm{EPDMS}$. When reactive simulation reaches the third round, it produces 167K samples yet delivers consistent and significant $\mathrm{EPDMS}$ gains across both model sizes. These results indicate that reactive agent dynamics enhance the realism and diversity of traffic interactions, thereby increasing the effectiveness of simulation data. Detailed scaling curves are shown in Sec. C.3 in the supplementary material.

### 3.4 Qualitative Results of the Simulation Scenes

We present qualitative visualizations from the simulation data in Fig. 5, showcasing four representative OOD scenarios used to train the policy. These scenarios mirror typical real-world driving challenges where learned policies tend to struggle, including (a) off-center lane drift, (b) near collision, (c) departure, and (d) cutting in cases. Each scenario is illustrated with a top-down view showing the pseudo-expert trajectory as supervision, and the deviating Perturbed Trajectory as history actions, alongside the synthetic front-view as sensory input to the policy. For instance, scenario (b) requires the policy to adaptively avoid the collision in short horizon. See supplementary Sec. C.2 for extended results.

## 4 Conclusion

In this paper, we introduce SimScale, a sim–real learning system that reveals how scalable simulation can amplify the value of real-world datasets for end-to-end autonomy. For the simulation data pipeline, we first generate pseudo-expert demonstrations from potential OOD states by ego perturbation within reactive environments. Toward the real-world simulation, the associated high-fidelity multi-view observations are rendered with 3DGS engine. Upon the simulated data, sim-real co-training produces synergistic improvements in robustness and generalization for various planners on challenging real-world benchmarks, up to +8.6 $\mathrm{EPDMS}$ on navhard and +2.9 navtest. Remarkably, the sim–real system scales clearly and predictably with increased simulation while keeping the real-world corpus fixed. We further uncover that exploration and interaction contribute to more effective simulation, and that multi-modal planners strengthen their scaling behavior. We hope SimScale will inspire the community to further explore real-world simulation for data scaling.

We provide additional experiments in the supplementary material, such as multi-expert ensemble and scaling with varying real data. It also provides extended discussions on related work, limitations, and broader impact.

## Acknowledgments

This study is supported by the National Key R&D Program of China (2022ZD0117901), the National Natural Science Foundation of China (Grant No. 62373355, 62236010), and the Beijing Natural Science Foundation (Grant L252033). This work is in part supported by the JC STEM Lab of Autonomous Intelligent Systems funded by The Hong Kong Jockey Club Charities Trust. We also appreciate the generous research sponsor from Xiaomi.

We extend our gratitude to Naiyan Wang, Yunsong Zhou, Hanxue Zhang, and the rest of the members from OpenDriveLab and Xiaomi Embodied Intelligence Team for their profound support.

## References

Supplementary Material

We outline the supplementary material as follows. In Sec. A, we first provide additional discussions of related work. Sec. B presents further implementation details regarding data curation and models. In Sec. C, we report extended experimental analyses and qualitative results corresponding to the main experiments (Sec. 3). In Sec. D, we further present additional experimental studies. Finally, Sec. E discusses the limitations and broader impacts, and Sec. F lists the licenses of all utilized assets.

## Appendix A Related Work

### A.1 End-to-End Autonomous Driving

End-to-end system maps directly from raw sensor inputs to planning [^9]. Early works adopt regression-based planning and gradually shift from extra task branches [^12] [^79] [^31] to unifying perception, prediction, and planning under joint supervision [^27] [^34] [^78] [^61] [^32]. Recent work has moved toward generative approaches. Diffusion-based systems [^60] [^83] [^55] [^45] [^15] are framed as a conditional denoising process, enabling diverse and high-fidelity trajectories. Concurrently, trajectory scoring has emerged as an efficient alternative, ranking candidate trajectories under spline curves [^7] [^26], discretized tokens [^33] [^20], clustered human trajectories [^57] [^59], or predicted proposals [^74] [^24] [^54] [^81]. Moreover, it offers a natural interface for reward- or cost-based optimization [^55] [^62] [^37] [^58] [^20] [^21] [^22], enabling reinforced improvements. Still, supervision on logged end-to-end data limits training to the expert’s open-loop distribution, causing compromised learning in the drifted state of sensory data. We address this by introducing a scalable simulation framework that reactively generates end-to-end pairs using pseudo-experts or rewards, allowing end-to-end systems across any above paradigms to bootstrap extra supervision from existing training data.

### A.2 Scene Simulation for Driving

Scene simulation, including traffic behavior simulation and sensor simulation, has long been a key topic in autonomous driving research. For traffic simulation, existing works [^6] [^20] [^100] leverage rule-based planners like IDM [^76], or diffusion-based generators [^36] [^103] to simulate plausible interaction of traffic agents. For sensor simulation, traditional graphics-based simulators [^18] [^49] suffer from a significant sim-to-real gap, which limits the real-world deployment of trained planners. Recent data-driven efforts follow two directions: some approaches [^101] [^85] [^1] [^91] [^77] [^52] [^102] attempt to generate sensor data in unseen scenarios via video generation models conditioned on 3D bounding boxes, HD maps or BEV maps. Other works focus on scene reconstruction [^65] [^86] [^10] [^100] [^35] [^6] [^82] to build photorealistic simulators for novel-view synthesis, using techniques like Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS). Although these methods achieve impressive visual results, most are primarily designed for closed-loop evaluation or visual augmentation. Our work aims to explore how to use traffic and sensor simulation to generate realistic scenes with feasible demonstrations for planner training, and its impacts on planner performances.

### A.3 Data Scaling for Driving

Recent data scaling laws have driven major advances in foundation models [^38] [^72] [^92] [^96], but their impact on autonomous driving planning remains underexplored. Prior researches [^29] [^68] [^99] demonstrate that increasing real-world data from thousands to millions of driving logs improves end-to-end planner performance, though improvements diminish. Dense supervision from video predictions improves this data scaling efficiency [^53]. For planners operating on abstract BEV representations, industry efforts [^2] [^28] demonstrate clear benefits from scaling real-world data and model capacity, while self-play [^14] [^93] [^13] scales reinforcement learning via massive simulation to achieve strong zero-shot sim-to-real transfer. However, most existing approaches rely on costly real-world collection or traffic simulation, which expands only in abstract state spaces rather than raw sensory domains. 3D rasterization provides a lightweight remedy, but suffers from information loss [^19]. In contrast, we investigate scaling properties of planning directly through large-scale 3DGS sensory simulation, which bridges abstract traffic simulation and real-world perception, offering a scalable and realistic alternative to real-world data collection.

### A.4 Comparison with Online RL

![[position.png|Refer to caption]]

Figure 6: Learning paradigm comparison of e2e autonomous driving between 3DGS-based Online RL and SimScale.

Online reinforcement learning (RL) [^30] [^48] [^46] in traffic simulators learn through exploration and feedback. Sensor simulation serves as a bridge to real-world environments, enabling real-world RL for autonomous driving, *i.e*., 3DGS-based Online RL [^20], as shown in Fig. 6 (a). In contrast, the proposed SimScale introduces an alternative framework, as illustrated in Fig. 6 (b), that generates OOD states with expert demonstrations at scale, enabling real-world simulations to support both imitation learning (IL) and reward learning (*i.e*. Offline RL) across arbitrary planners.

## Appendix B Extended Implementation Details

### B.1 Simulation Data Curation

Table 5: Simulation Data Curation Pipeline Configurations.

<table><tbody><tr><td>Term</td><td>Notation</td><td>Value</td></tr><tr><td colspan="3">3DGS Data Engine</td></tr><tr><td>Peak signal-to-noise ratio of GS</td><td><math><semantics><mi>PSNR</mi> <annotation>\mathrm{PSNR}</annotation></semantics></math></td><td><math><semantics><mrow><mo><</mo> <mn>27</mn></mrow> <annotation><27</annotation></semantics></math></td></tr><tr><td colspan="3">Trajectory Perturbation</td></tr><tr><td>Clustered human traj vocabulary</td><td><math><semantics><msub><mi>𝒱</mi> <mtext>c</mtext></msub> <annotation>\mathcal{V}_{\text{c}}</annotation></semantics></math></td><td><math><semantics><mn>16,384</mn> <annotation>16,384</annotation></semantics></math></td></tr><tr><td>Relative heading range to log</td><td><math><semantics><mrow><mi>Δ</mi> <mo></mo><mi>θ</mi></mrow> <annotation>\Delta\theta</annotation></semantics></math></td><td><math><semantics><mrow><mo>±</mo> <msup><mn>20</mn> <mo>∘</mo></msup></mrow> <annotation>\pm 20^{\circ}</annotation></semantics></math></td></tr><tr><td>Longitudinal shift range</td><td><math><semantics><msub><mi>r</mi> <mi>lon</mi></msub> <annotation>r_{\mathrm{lon}}</annotation></semantics></math></td><td><math><semantics><mrow><mo>±</mo> <mrow><mn>20</mn> <mo></mo><mi>m</mi></mrow></mrow> <annotation>\pm 20\mathrm{m}</annotation></semantics></math></td></tr><tr><td>Lateral shift range</td><td><math><semantics><msub><mi>r</mi> <mi>lat</mi></msub> <annotation>r_{\mathrm{lat}}</annotation></semantics></math></td><td><math><semantics><mrow><mo>±</mo> <mrow><mn>2</mn> <mo></mo><mi>m</mi></mrow></mrow> <annotation>\pm 2\mathrm{m}</annotation></semantics></math></td></tr><tr><td>Longitudinal sampling step</td><td><math><semantics><msub><mi>δ</mi> <mi>lon</mi></msub> <annotation>\delta_{\mathrm{lon}}</annotation></semantics></math></td><td><math><semantics><mrow><mn>5</mn> <mo></mo><mi>m</mi></mrow> <annotation>5\mathrm{m}</annotation></semantics></math></td></tr><tr><td>Lateral sampling step</td><td><math><semantics><msub><mi>δ</mi> <mi>lat</mi></msub> <annotation>\delta_{\mathrm{lat}}</annotation></semantics></math></td><td><math><semantics><mrow><mn>0.5</mn> <mo></mo><mi>m</mi></mrow> <annotation>0.5\mathrm{m}</annotation></semantics></math></td></tr><tr><td>Reward filterer</td><td><math><semantics><msubsup><mi>R</mi> <mi>EPDMS</mi> <mi>per</mi></msubsup> <annotation>\mathrm{R^{per}_{EPDMS}}</annotation></semantics></math></td><td><math><semantics><mrow><mo>≥</mo> <mn>0.8</mn></mrow> <annotation>\geq 0.8</annotation></semantics></math></td></tr><tr><td colspan="3">Pseudo-Expert Trajectory Generation</td></tr><tr><td>Reward filterer</td><td><math><semantics><msubsup><mi>R</mi> <mi>EPDMS</mi> <mi>exp</mi></msubsup> <annotation>\mathrm{R^{exp}_{EPDMS}}</annotation></semantics></math></td><td><math><semantics><mtable><mtr><mtd><mrow><mrow><msub><mi>S</mi> <mi>m</mi></msub> <mo>=</mo> <mn>1</mn></mrow><mo>,</mo><mrow><mrow><mo>∀</mo> <mi>m</mi></mrow> <mo>≠</mo> <mi>EP</mi></mrow></mrow></mtd></mtr> <mtr><mtd><mrow><mrow><msub><mi>S</mi> <mi>EP</mi></msub> <mo>></mo> <mn>0.5</mn></mrow></mrow></mtd></mtr></mtable> <annotation>\begin{subarray}{c}S_{m}=1,\forall m\neq\mathrm{EP}\\ S_{\mathrm{EP}}>0.5\hskip 8.19447pt~~~~\end{subarray}</annotation></semantics></math></td></tr><tr><td colspan="3">Recovery-based</td></tr><tr><td>Whole human traj vocabulary</td><td><math><semantics><msub><mi>𝒱</mi> <mtext>h</mtext></msub> <annotation>\mathcal{V}_{\text{h}}</annotation></semantics></math></td><td><math><semantics><mn>103,288</mn> <annotation>103,288</annotation></semantics></math></td></tr><tr><td colspan="3">Planner-based</td></tr><tr><td>Privileged planner</td><td><math><semantics><mi>𝐏</mi> <annotation>\mathbf{P}</annotation></semantics></math></td><td>PDM-Closed <sup><a href="#fn:16">16</a></sup></td></tr></tbody></table>

Tab. 5 summarizes the detailed terms with notation and value used in the simulation data curation process to supplement Sec. 3.1 curation pipeline in the main paper. For pseudo-expert trajectory generation, we discard infeasible candidates to ensure valid supervision. Specifically, all sub-metrics of $\mathrm{EPDMS}$ must be satisfied, with $S_{\mathrm{EP}}$ relaxed: $\mathrm{EP}\geq 0.5$, preventing biased driving styles. Qualitative results during curation are shown in Sec. C.2

### B.2 Models and Training

Table 6: Model and Training Hyperparameters.

<table><tbody><tr><td>Hyperpara.</td><td>LTF <sup><a href="#fn:12">12</a></sup></td><td>DiffusionDrive <sup><a href="#fn:60">60</a></sup></td><td>GTRS-Dense <sup><a href="#fn:59">59</a></sup></td></tr><tr><td colspan="4">Model Configuration</td></tr><tr><td>Sensors</td><td><math><semantics><mrow><mn>3</mn> <mo>×</mo> <mtext>Cam.</mtext></mrow><annotation>3\times\text{Cam.}</annotation></semantics></math></td><td><math><semantics><mrow><mn>3</mn> <mo>×</mo> <mtext>Cam.</mtext></mrow><annotation>3\times\text{Cam.}</annotation></semantics></math></td><td><math><semantics><mrow><mn>3</mn> <mo>×</mo> <mtext>Cam.</mtext></mrow><annotation>3\times\text{Cam.}</annotation></semantics></math></td></tr><tr><td>Resolution</td><td><math><semantics><mrow><mn>2048</mn> <mo>×</mo> <mn>512</mn></mrow> <annotation>2048\times 512</annotation></semantics></math></td><td><math><semantics><mrow><mn>2048</mn> <mo>×</mo> <mn>512</mn></mrow> <annotation>2048\times 512</annotation></semantics></math></td><td><math><semantics><mrow><mn>2048</mn> <mo>×</mo> <mn>512</mn></mrow> <annotation>2048\times 512</annotation></semantics></math></td></tr><tr><td>Horizon</td><td><math><semantics><mrow><mn>4</mn> <mo></mo><mi>s</mi></mrow> <annotation>4\mathrm{s}</annotation></semantics></math></td><td><math><semantics><mrow><mn>4</mn> <mo></mo><mi>s</mi></mrow> <annotation>4\mathrm{s}</annotation></semantics></math></td><td><math><semantics><mrow><mn>4</mn> <mo></mo><mi>s</mi></mrow> <annotation>4\mathrm{s}</annotation></semantics></math></td></tr><tr><td>Frequency</td><td><math><semantics><mrow><mn>2</mn> <mo></mo><mi>Hz</mi></mrow> <annotation>2\mathrm{Hz}</annotation></semantics></math></td><td><math><semantics><mrow><mn>2</mn> <mo></mo><mi>Hz</mi></mrow> <annotation>2\mathrm{Hz}</annotation></semantics></math></td><td><math><semantics><mrow><mn>10</mn> <mo></mo><mi>Hz</mi></mrow> <annotation>10\mathrm{Hz}</annotation></semantics></math></td></tr><tr><td>Backbone</td><td>R34 <sup><a href="#fn:25">25</a></sup></td><td>R34 <sup><a href="#fn:25">25</a></sup></td><td>R34 <sup><a href="#fn:25">25</a></sup> / V99 <sup><a href="#fn:42">42</a></sup></td></tr><tr><td>Parameters</td><td><math><semantics><mrow><mn>56</mn> <mo></mo><mi>M</mi></mrow> <annotation>56\mathrm{M}</annotation></semantics></math></td><td><math><semantics><mrow><mn>61</mn> <mo></mo><mi>M</mi></mrow> <annotation>61\mathrm{M}</annotation></semantics></math></td><td><math><semantics><mrow><mn>67</mn> <mo></mo><mi>M</mi></mrow> <annotation>67\mathrm{M}</annotation></semantics></math> / <math><semantics><mrow><mn>83</mn> <mo></mo><mi>M</mi></mrow> <annotation>83\mathrm{M}</annotation></semantics></math></td></tr><tr><td>Aux. Tasks</td><td>Det. Seg.</td><td>Det. Seg.</td><td>None</td></tr><tr><td colspan="4">Training Configuration</td></tr><tr><td>GPUs</td><td><math><semantics><mrow><mn>8</mn> <mo>×</mo> <mtext>H20</mtext></mrow> <annotation>8\times\text{H20}</annotation></semantics></math></td><td><math><semantics><mrow><mn>8</mn> <mo>×</mo> <mtext>H20</mtext></mrow> <annotation>8\times\text{H20}</annotation></semantics></math></td><td><math><semantics><mrow><mn>32</mn> <mo>×</mo> <mtext>H20</mtext></mrow> <annotation>32\times\text{H20}</annotation></semantics></math></td></tr><tr><td>Epochs</td><td><math><semantics><mn>100</mn> <annotation>100</annotation></semantics></math></td><td><math><semantics><mn>100</mn> <annotation>100</annotation></semantics></math></td><td><math><semantics><mn>50</mn> <annotation>50</annotation></semantics></math></td></tr><tr><td>Total BS</td><td><math><semantics><mn>512</mn> <annotation>512</annotation></semantics></math></td><td><math><semantics><mn>512</mn> <annotation>512</annotation></semantics></math></td><td><math><semantics><mn>352</mn> <annotation>352</annotation></semantics></math></td></tr><tr><td>Initial LR</td><td><math><semantics><mrow><mn>2.83</mn> <mo>×</mo> <msup><mn>10</mn> <mrow><mo>−</mo> <mn>4</mn></mrow></msup></mrow> <annotation>2.83\times{10}^{-4}</annotation></semantics></math></td><td><math><semantics><mrow><mn>6</mn> <mo>×</mo> <msup><mn>10</mn> <mrow><mo>−</mo> <mn>4</mn></mrow></msup></mrow> <annotation>6\times{10}^{-4}</annotation></semantics></math></td><td><math><semantics><mrow><mn>4</mn> <mo>×</mo> <msup><mn>10</mn> <mrow><mo>−</mo> <mn>4</mn></mrow></msup></mrow> <annotation>4\times{10}^{-4}</annotation></semantics></math></td></tr><tr><td>Schedule</td><td>Constant</td><td>Cosine Decay</td><td>Cosine Decay</td></tr><tr><td>Optimizer</td><td>Adam <sup><a href="#fn:41">41</a></sup></td><td>AdamW <sup><a href="#fn:66">66</a></sup></td><td>AdamW <sup><a href="#fn:66">66</a></sup></td></tr></tbody></table>

Table 7: Detailed Results on navhard. PDM-Closed uses ground-truth symbolic inputs for planning, while other methods rely on sensor data. (recovery / planner: recovery-based / planner-based expert; reward: reward-only scoring; S.: per-stage EPDM score.)

<table><tbody><tr><td>Method</td><td>Backbone</td><td>Sim.</td><td>Stage</td><td>NC↑</td><td>DAC↑</td><td>DDC↑</td><td>TLC↑</td><td>EP↑</td><td>TTC↑</td><td>LK↑</td><td>HC↑</td><td>EC↑</td><td>S.↑</td><td>EPDMS↑</td></tr><tr><td>PDM-Closed <sup><a href="#fn:16">16</a></sup></td><td>-</td><td>-</td><td>S 1 S 2</td><td>94.4 88.1</td><td>98.8 90.6</td><td>100 96.3</td><td>99.5 98.5</td><td>100 100</td><td>93.5 83.1</td><td>99.3 73.7</td><td>87.7 91.5</td><td>36.0 25.4</td><td>- -</td><td>51.3</td></tr><tr><td colspan="15">Regeression-based Planner</td></tr><tr><td rowspan="3">LTF <sup><a href="#fn:12">12</a></sup></td><td rowspan="4">ResNet34</td><td>w/o</td><td>S 1 S 2</td><td>97.3 79.4</td><td>80.2 69.0</td><td>97.8 85.6</td><td>99.3 98.5</td><td>83.4 83.8</td><td>96.2 76.7</td><td>92.9 47.9</td><td>97.8 97.0</td><td>71.1 70.6</td><td>61.3 39.2</td><td>24.4</td></tr><tr><td>recovery</td><td>S 1 S 2</td><td>96.4 88.9</td><td>78.4 71.1</td><td>98.9 91.8</td><td>99.8 99.0</td><td>80.5 77.3</td><td>96.2 85.5</td><td>92.7 53.8</td><td>97.6 96.8</td><td>78.2 47.5</td><td>60.7 46.3</td><td>29.8</td></tr><tr><td>planner</td><td>S 1 S 2</td><td>96.1 85.5</td><td>85.3 66.9</td><td>99.4 91.6</td><td>99.3 99.1</td><td>84.7 93.0</td><td>94.7 81.1</td><td>93.6 58.3</td><td>97.6 95.1</td><td>77.3 42.9</td><td>66.3 44.8</td><td>30.2</td></tr><tr><td colspan="15">Diffusion-based Planner    </td></tr><tr><td rowspan="3">DiffusionDrive <sup><a href="#fn:60">60</a></sup></td><td rowspan="4">ResNet34</td><td>w/o</td><td>S 1 S 2</td><td>96.8 80.1</td><td>86.0 72.8</td><td>98.8 84.4</td><td>99.3 98.4</td><td>84.0 85.9</td><td>95.8 76.6</td><td>96.7 46.4</td><td>97.6 96.3</td><td>79.6 72.8</td><td>66.7 40.5</td><td>27.5</td></tr><tr><td>recovery</td><td>S 1 S 2</td><td>97.2 82.4</td><td>88.4 67.7</td><td>99.1 89.1</td><td>99.8 98.6</td><td>83.9 89.0</td><td>96.0 77.6</td><td>96.7 53.8</td><td>97.6 95.2</td><td>76.9 46.8</td><td>69.4 41.7</td><td>30.4</td></tr><tr><td>planner</td><td>S 1 S 2</td><td>97.2 82.4</td><td>88.7 72.1</td><td>99.3 92.9</td><td>99.3 98.5</td><td>82.8 92.1</td><td>96.9 80.6</td><td>98.0 60.8</td><td>97.3 95.4</td><td>59.6 31.9</td><td>67.5 46.8</td><td>32.6</td></tr><tr><td colspan="15">Scoring-based Planner      </td></tr><tr><td rowspan="16">GTRS-Dense <sup><a href="#fn:59">59</a></sup></td><td rowspan="4">ResNet34</td><td>w/o</td><td>S 1 S 2</td><td>99.3 92.8</td><td>96.6 88.6</td><td>99.6 95.5</td><td>100 99.4</td><td>57.4 55.9</td><td>99.5 91.3</td><td>92.6 55.7</td><td>89.5 91.1</td><td>16.4 35.7</td><td>67.1 55.8</td><td>38.3</td></tr><tr><td>recovery</td><td>S 1 S 2</td><td>97.6 92.0</td><td>94.2 88.1</td><td>99.3 94.3</td><td>99.6 98.6</td><td>76.7 83.6</td><td>97.6 89.8</td><td>94.9 58.1</td><td>97.1 91.4</td><td>37.8 25.9</td><td>69.8 59.3</td><td>43.0</td></tr><tr><td>planner</td><td>S 1 S 2</td><td>98.7 94.2</td><td>94.2 92.1</td><td>99.7 95.0</td><td>100 99.0</td><td>74.3 79.9</td><td>98.2 91.8</td><td>95.8 58.9</td><td>96.7 91.0</td><td>34.7 32.9</td><td>70.8 63.9</td><td>46.1</td></tr><tr><td>reward</td><td>S 1 S 2</td><td>97.6 94.3</td><td>96.4 92.7</td><td>99.3 95.1</td><td>100 99.5</td><td>75.7 80.2</td><td>97.8 91.5</td><td>93.3 56.2</td><td>97.3 90.6</td><td>32.9 28.3</td><td>72.4 63.4</td><td>46.9</td></tr><tr><td rowspan="8">V2-99</td><td>w/o</td><td>S 1 S 2</td><td>98.9 89.9</td><td>94.9 90.5</td><td>99.1 94.1</td><td>100 99.3</td><td>76.1 77.6</td><td>98.4 88.5</td><td>93.8 56.0</td><td>94.9 92.0</td><td>37.8 30.2</td><td>70.4 58.5</td><td>41.9</td></tr><tr><td>recovery</td><td>S 1 S 2</td><td>99.1 95.0</td><td>98.2 90.8</td><td>99.4 94.6</td><td>100 99.4</td><td>71.9 75.5</td><td>99.1 93.5</td><td>95.6 57.4</td><td>95.8 93.7</td><td>32.4 36.8</td><td>73.4 63.1</td><td>46.4</td></tr><tr><td>planner</td><td>S 1 S 2</td><td>99.3 95.6</td><td>97.1 91.9</td><td>99.9 95.0</td><td>100 98.9</td><td>67.2 76.7</td><td>99.3 93.7</td><td>94.0 61.9</td><td>94.4 90.8</td><td>23.6 38.0</td><td>71.7 65.5</td><td>47.7</td></tr><tr><td>reward</td><td>S 1 S 2</td><td>99.3 94.9</td><td>97.6 94.3</td><td>99.4 94.5</td><td>100 99.3</td><td>71.5 79.0</td><td>99.6 92.6</td><td>96.0 57.8</td><td>95.8 93.4</td><td>30.7 30.9</td><td>72.5 65.4</td><td>48.0</td></tr></tbody></table>

Table 8: Detailed Results on navtest. (recovery / planner: recovery-based / planner-based expert; reward: reward-only scoring)

<table><tbody><tr><td>Method</td><td>Backbone</td><td>Sim.</td><td>NC↑</td><td>DAC↑</td><td>DDC↑</td><td>TLC↑</td><td>EP↑</td><td>TTC↑</td><td>LK↑</td><td>HC↑</td><td>EC↑</td><td>EPDMS↑</td></tr><tr><td>Human Agent</td><td>-</td><td>-</td><td>100</td><td>100</td><td>99.8</td><td>100</td><td>87.4</td><td>100</td><td>100</td><td>98.1</td><td>90.1</td><td>90.3</td></tr><tr><td colspan="13">Regression-based Planner</td></tr><tr><td rowspan="3">LTF <sup><a href="#fn:12">12</a></sup></td><td rowspan="3">ResNet34</td><td>w/o</td><td>97.7</td><td>94.0</td><td>99.3</td><td>99.8</td><td>87.2</td><td>96.7</td><td>95.5</td><td>98.3</td><td>82.9</td><td>81.5</td></tr><tr><td>recovery</td><td>97.9</td><td>95.1</td><td>99.5</td><td>99.8</td><td>87.7</td><td>97.2</td><td>97.2</td><td>98.3</td><td>87.1</td><td>83.6</td></tr><tr><td>planner</td><td>98.3</td><td>95.6</td><td>99.6</td><td>99.8</td><td>87.1</td><td>97.5</td><td>97.2</td><td>98.3</td><td>88.2</td><td>84.4</td></tr><tr><td colspan="13">Diffusion-based Planner    </td></tr><tr><td rowspan="3">DiffusionDrive <sup><a href="#fn:60">60</a></sup></td><td rowspan="3">ResNet34</td><td>w/o</td><td>98.4</td><td>95.5</td><td>99.5</td><td>99.8</td><td>87.5</td><td>97.5</td><td>96.9</td><td>98.4</td><td>87.7</td><td>84.2</td></tr><tr><td>recovery</td><td>98.4</td><td>96.7</td><td>99.6</td><td>99.8</td><td>87.6</td><td>97.5</td><td>97.5</td><td>98.3</td><td>87.1</td><td>85.4</td></tr><tr><td>planner</td><td>98.5</td><td>97.1</td><td>99.6</td><td>99.8</td><td>87.4</td><td>97.8</td><td>98.0</td><td>98.3</td><td>87.5</td><td>85.9</td></tr><tr><td colspan="13">Scoring-based Planner      </td></tr><tr><td rowspan="8">GTRS-Dense <sup><a href="#fn:59">59</a></sup></td><td rowspan="4">ResNet34</td><td>w/o</td><td>97.6</td><td>97.5</td><td>99.0</td><td>99.9</td><td>87.9</td><td>97.0</td><td>95.9</td><td>97.5</td><td>55.9</td><td>82.3</td></tr><tr><td>recovery</td><td>98.2</td><td>97.9</td><td>99.5</td><td>99.9</td><td>87.2</td><td>97.8</td><td>96.3</td><td>97.6</td><td>56.5</td><td>83.4</td></tr><tr><td>planner</td><td>97.8</td><td>98.3</td><td>99.5</td><td>99.9</td><td>89.2</td><td>97.3</td><td>97.3</td><td>96.9</td><td>55.1</td><td>84.0</td></tr><tr><td>reward</td><td>98.4</td><td>98.8</td><td>99.4</td><td>99.9</td><td>88.0</td><td>98.0</td><td>96.5</td><td>97.6</td><td>58.0</td><td>84.6</td></tr><tr><td rowspan="4">V2-99</td><td>w/o</td><td>97.6</td><td>98.5</td><td>99.5</td><td>99.9</td><td>89.5</td><td>97.2</td><td>96.8</td><td>97.2</td><td>57.2</td><td>84.0</td></tr><tr><td>recovery</td><td>98.3</td><td>98.7</td><td>99.4</td><td>99.9</td><td>88.9</td><td>97.9</td><td>97.1</td><td>97.6</td><td>55.9</td><td>84.5</td></tr><tr><td>planner</td><td>98.7</td><td>99.0</td><td>99.5</td><td>99.8</td><td>86.8</td><td>98.3</td><td>95.9</td><td>97.9</td><td>58.5</td><td>84.5</td></tr><tr><td>reward</td><td>97.6</td><td>98.9</td><td>99.6</td><td>99.9</td><td>89.9</td><td>97.1</td><td>97.2</td><td>97.7</td><td>58.7</td><td>84.8</td></tr></tbody></table>

Tab. 6 provides the detailed model and training hyperparameters that complement Sec. 3.1 implementation protocol in the main paper. We follow the NAVSIM [^17] default setup and use only the train logs [^39] of navtrain for both real-data training and sim–real co-training. All planners share identical settings across the real or sim-real training setups, and each model is trained to saturation to ensure the validity of the subsequent scaling analyses.

LTF. [^12] A regression-based planner that directly regresses future waypoints upon fused sensory latents. It employs pretrained ResNet34 [^25] as image encoder and is trained for 100 epochs on 8 GPUs with a total batch size of 512 and a learning rate of $2.83\times 10^{-4}$.

DiffusionDrive. [^60] A multi-modal, DETR-style generative planner that iteratively denoises diverse trajectories using anchor-conditioned truncated diffusion. Each anchor adaptively queries the encoded image features. It employs pretrained ResNet34 [^25] as image encoder and applies truncated diffusion with 20 clustered anchors. It is trained for 100 epochs on 8 GPUs with a total batch size of 512 and a learning rate of $6\times 10^{-4}$.

GTRS-Dense. [^59] A scoring-based, DETR-style planner that ranks a dense vocabulary of clustered human trajectories using multiple supervised scoring heads. Inputs are augmented with random spatial shifts. It employs pretrained ResNet34 [^25] or V2-99 [^42] as image encoder with an action space of 16,384 trajectories. It is trained for 50 epochs on 32 GPUs with a total batch size of 352 and a learning rate of $4\times 10^{-4}$. We use a vocabulary of 16,384 trajectories for training. At inference time, 8,192 trajectories are used for navhard, while the full 16,384 trajectories are used for navtest.

## Appendix C Extended Experimental Results

### C.1 Detailed Leaderboard Results

In the main paper Sec. 3.2, Tab. 1 and Tab. 2 report only the best results in navhard and navtest of each planner co-training (highlighted in blue rows). Supplementary Tab. 7 and Tab. 8 provide full results across different pseudo-experts and the reward-only scoring method. The best results remain highlighted in blue rows, allowing direct comparison with the main paper. Planner-based experts generally outperform recovery-based experts, while scoring-based planner benefits most from reward-only method.

### C.2 Complete Qualitative Results

We show extended visualizations of our simulated data in Fig. 10 and Figs. 11–13, which demonstrate the high fidelity and pseudo-expert generation of our data generation pipeline. More OOD simulation scenes are shown in Fig 14.

Recovery-based Expert. As shown in Fig. 10, all simulation trajectories successfully converge back to the human expert’s final waypoint.

Planner-based Expert As shown in Figs. 11–13, the simulated feasible trajectories under perturbed states exhibit substantial variation and broader coverage across diverse traffic scenarios, including the Y-branch intersection, the dense urban avenue, and the narrow local road.

### C.3 Scaling of Reactive vs. Non-Reactive

![[curve_scaling_react.png|Refer to caption]]

Figure 7: Scaling dynamics under reactive and non-reactive simulation using GTRS-Dense across model sizes.

In the main paper, Tab. 4 reports only a subset of simulation rounds for GTRS-Dense. Supplementary Fig. 7 provides the full scaling curves, demonstrating that reactive simulation data consistently achieves superior scaling performance across model sizes.

![[data_statistic.png|Refer to caption]]

Figure 8: EgoMLP EPDMS \\mathrm{EPDMS} on simulation and real data. (a) distribution under simulation versus real-world data. (b) Scenario tags ranked by. The figures reveal the distribution shift and scenario characteristics of simulation data.

## Appendix D Additional Experimental Results

### D.1 Multi-Expert Ensemble

Table 9: The effect of multi-expert ensemble on navhard using GTRS-Dense. ($\textbf{S}_{1/2}$:Per-stage EPDM scores; recovery / planner: recovery-based / planner-based expert; reward: reward scoring only; $*$:multi-expert ensemble)

<table><tbody><tr><td>Backbone</td><td>Sim</td><td>Expert</td><td><math><semantics><mrow><msub><mtext>S</mtext> <mn>1</mn></msub> <mo>↑</mo></mrow> <annotation>\textbf{S}_{1}\uparrow</annotation></semantics></math></td><td><math><semantics><mrow><msub><mtext>S</mtext> <mn>2</mn></msub> <mo>↑</mo></mrow> <annotation>\textbf{S}_{2}\uparrow</annotation></semantics></math></td><td><math><semantics><mrow><mtext>EPDMS</mtext> <mo>↑</mo></mrow> <annotation>\textbf{EPDMS}\uparrow</annotation></semantics></math></td></tr><tr><td rowspan="5">ResNet34</td><td></td><td></td><td>67.1</td><td>55.8</td><td>38.3</td></tr><tr><td>✓</td><td>recovery</td><td>69.8</td><td>59.3</td><td>43.0</td></tr><tr><td>✓</td><td>planner</td><td>70.8</td><td>63.9</td><td>46.1</td></tr><tr><td>✓</td><td>reward</td><td>72.4</td><td>63.4</td><td>46.9</td></tr><tr><td>✓</td><td><math><semantics><mo>∗</mo> <annotation>*</annotation></semantics></math></td><td>72.8</td><td>65.1</td><td>47.7</td></tr><tr><td rowspan="5">V2-99</td><td></td><td></td><td>70.4</td><td>58.5</td><td>41.9</td></tr><tr><td>✓</td><td>recovery</td><td>73.4</td><td>63.1</td><td>46.4</td></tr><tr><td>✓</td><td>planner</td><td>71.7</td><td>65.5</td><td>47.7</td></tr><tr><td>✓</td><td>reward</td><td>72.5</td><td>65.4</td><td>48.0</td></tr><tr><td>✓</td><td><math><semantics><mo>∗</mo> <annotation>*</annotation></semantics></math></td><td>74.8</td><td>67.6</td><td>50.9</td></tr><tr><td><math><semantics><mo>∗</mo> <annotation>*</annotation></semantics></math></td><td>✓</td><td><math><semantics><mo>∗</mo> <annotation>*</annotation></semantics></math></td><td>75.4</td><td>66.4</td><td>50.4</td></tr></tbody></table>

Different pseudo-experts exhibit distinct behavioral characteristics, *e.g*., the recovery-based expert is more conservative, whereas the planner-based expert is more exploratory. This raises the question of whether these complementary properties can be leveraged jointly. Therefore, we conduct a simple ensemble study, multi-expert enseble as we call it, to examine their potential complementarity on navhard, as shown in Tab. 9. The scoring-based GTRS-Dense planner is chosen because it enables a straightforward ensemble: the predicted sub-scores of each trajectory in the vocabulary can be directly averaged across the three models (recovery-based, planner-based, reward-only scoring).

Ensemble Gains from Expert Diversity. As shown in Tab. 9, this simple multi-expert ensemble yields consistent improvements for both backbones, achieving +0.8 and +2.9 $\mathrm{EPDMS}$ on ResNet34 and V2-99 compared with reward-only scoring, respectively, for navhard. Further enlarging the ensemble to all six models yields no additional $\mathrm{EPDMS}$ gains. These indicate that, for end-to-end planning, different pseudo-experts provide strong complementary benefits—often exceeding the gains brought by different backbone structures.

### D.2 Effect of Scenarios beyond Scaling

Beyond data scaling, simulation data exhibits an out-of-distribution shift relative to real-world data. We provide a preliminary analysis of these characteristics by evaluating an EgoMLP planner, trained on navtrain, on both navtest and simulation data, as shown in Fig. 8. (1) Fig. 8 (a): Sim data shows a lower $\mathrm{EPDMS}$ distribution than real data, confirming a higher safety-critical level. (2) Fig. 8 (b): Low $\mathrm{EPDMS}$ scores across nuPlan scenario tags identify challenging simulation scenarios, facilitating future study on their distinct impact, *e.g*., increasing their proportions during simulation.

### D.3 Effect of Simulation Visual Fidelity

Table 10: Effect of simulation visual fidelity on navhard using GTRS-Dense, across sampling rounds and co-training strategies.

<table><tbody><tr><td rowspan="2">Backbone</td><td rowspan="2"><math><semantics><mi>PSNR</mi> <annotation>\mathrm{PSNR}</annotation></semantics></math></td><td colspan="2">Pseudo-Expert</td><td colspan="2">Reward-Only Scoring</td></tr><tr><td>#R 1</td><td>#R 2</td><td>#R 1</td><td>#R 2</td></tr><tr><td rowspan="2">ResNet34</td><td><math><semantics><mrow><mo><</mo> <mn>27</mn></mrow> <annotation><27</annotation></semantics></math></td><td>41.0</td><td>41.1</td><td>40.8</td><td>41.6</td></tr><tr><td><math><semantics><mrow><mo>≥</mo> <mn>27</mn></mrow> <annotation>\geq 27</annotation></semantics></math></td><td>41.9</td><td>41.2</td><td>41.3</td><td>42.9</td></tr><tr><td rowspan="2">V2-99</td><td><math><semantics><mrow><mo><</mo> <mn>27</mn></mrow> <annotation><27</annotation></semantics></math></td><td>44.4</td><td>44.5</td><td>43.7</td><td>44.8</td></tr><tr><td><math><semantics><mrow><mo>≥</mo> <mn>27</mn></mrow> <annotation>\geq 27</annotation></semantics></math></td><td>45.4</td><td>46.1</td><td>44.9</td><td>46.3</td></tr></tbody></table>

We retrain GTRS-Dense with new simulation data of $\mathrm{PSNR}<27$ ($\sim$ 10K $\times$ 2 rounds) and compare it with the matched samples randomly selected from simulations with $\mathrm{PSNR}\geq 27$. The former represents lower reconstruction quality and is more likely to yield simulations with reduced visual fidelity. As shown in Tab. 10, simulation data with $\mathrm{PSNR}\geq 27$ consistently yields higher EPDMS under different data scales and co-training strategies. These results indicate that simulation data with higher visual fidelity helps reduce the visual gap between simulation and real-world observations, thereby improving the effectiveness of simulation data.

### D.4 Scaling with Varying Scales of Real Data

![[curve_scaling_real.png|Refer to caption]]

Figure 9: Scaling simulation with varying real data. Simulation data are scaled by corresponding real data scenario tokens and fixed sim-real data ratio

To assess the practical utility of our approach, we study how simulation data affects performance as the amount of available real-world data scales up. Based on GTRS-Dense with reward-only scoring sim-real co-training, we conduct experiments with different real data from navtrain (10K, 20K, 50K, 100K). Inspired by [^1], simulation data are generated from identical scenarios of corresponding real data tokens, and we fix the sim-real data ratio (five sampling rounds) for each experiment.

Sustained Simulation Gains Across Real Data Scales. As shown in Fig. 9, our simulation data provides the most significant gains when real data is scarce (10K), achieving +22.4% $\mathrm{EPDMS}$ on ResNet34 and +12.1% $\mathrm{EPDMS}$ on V2-99 for navhard. As the amount of real data increases (from 10K to 100K), these gains remain consistently high, without noticeable narrowing. This demonstrates that our simulation data can effectively complement limited real-world data and still offers substantial potential to further exploit and amplify the value of real data even when it is abundant.

## Appendix E Limitations and Broader Impact

### E.1 Pseudo-Expert

Despite its effectiveness, our pseudo-expert generation pipeline has limitations. Current rule-based trajectory perturbations are static. A potential enhancement is the self-evolving approach [^90], using pretrained planners to iteratively explore the simulator and collect recovery data. Additionally, the privileged BEV planner we use is rule-based with limited performance, causing some degradation in comfort metrics ($\mathrm{HC}$, $\mathrm{EC}$ in Tab. 7) and failing in extremely corner cases. More advanced learning-based BEV planners [^98] [^75] [^30] could further improve generation efficiency and realism.

### E.2 Scene Simulation

For traffic behavior simulation in our decoupled scene simulation paradigm, other agents are controlled by IDM [^76], which enables interaction but limits scenario diversity. Diffusion-based traffic generators [^103] [^36] [^50] offer a promising improvement. For sensor simulation, feedforward GS [^8] [^87] can improve reconstruction efficiency, while adding modalities like LiDAR [^44] provides complementary modality information. Furthermore, unified world models [^80] [^89] [^94] [^95] [^73] can serve as a substitute for both behavior and sensor simulation.

### E.3 Self-Play

Self-play [^14] [^13] allows the ego and surrounding agents to share learned policies instead of relying on separate traffic behavior simulators, enabling behaviors to co-evolve through interaction. With sensor simulation, it can support end-to-end policy learning and potentially improve robustness in long-tail scenarios, though it remains limited by interaction cost and simulation fidelity.

### E.4 Societal Impact

Despite promising improvements, pseudo-expert scene simulation still has room for quality and efficiency enhancements for real-world deployment. Co-training may be affected by unrealistic simulation visuals and distributional differences in real-world corner cases, which could lead to potential risks. We hope SimScale will inspire both academia and industry in driving and robotics to leverage real-world simulation to address rare corner cases in data scaling, advancing fully autonomous systems that are robust and generalizable. In addition, we will provide the community with a fully open-source simulation dataset and training framework to promote academic research on autonomous driving in simulation.

## Appendix F License of Assets

All training and evaluation are performed on data from publicly licensed datasets [^39] [^70] [^6]. The real-world data engine is based on the MTGS codebase [^51] under the Apache-2.0 license. We adopt publicly available end-to-end planners, including GTRS [^59] (Apache-2.0) and DiffusionDrive [^60] and LTF [^12] (MIT).

![[vis_expert_recovery_1.png|Refer to caption]]

Figure 10: Qualitative results of recovery-based expert with real and simulation data.

![[vis_expert_planner_1.png|Refer to caption]]

Figure 11: Qualitative results of planner-based expert with real and simulation data. (Y-branch intersection)

![[vis_expert_planner_2.png|Refer to caption]]

Figure 12: Qualitative results of planner-based expert with real and simulation data. (dense urban avenue)

![[vis_expert_planner_3.png|Refer to caption]]

Figure 13: Qualitative results of planner-based expert with real and simulation data. (narrow local road)

![[vis_scene_add.png|Refer to caption]]

Figure 14: Additional qualitative results of the simulation scenes on navtrain.

[^1]: H. A. Alhaija, J. Alvarez, M. Bala, T. Cai, T. Cao, L. Cha, J. Chen, M. Chen, F. Ferroni, S. Fidler, et al. (2025) Cosmos-transfer1: conditional world generation with adaptive multimodal control. arXiv preprint arXiv:2503.14492. Cited by: §A.2, §D.4.

[^2]: M. Baniodeh, K. Goel, S. Ettinger, C. Fuertes, A. Seff, T. Shen, C. Gulino, C. Yang, G. Jerfel, D. Choe, et al. (2025) Scaling laws of motion forecasting and planning–a technical report. arXiv preprint arXiv:2506.08228. Cited by: §A.3.

[^3]: J. Beißwenger (2024) PDM-lite: a rule-based planner for carla leaderboard 2.0. Univ. Tübingen. Cited by: §2.3.

[^4]: A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choromanski, T. Ding, D. Driess, A. Dubey, C. Finn, P. Florence, C. Fu, M. G. Arenas, K. Gopalakrishnan, K. Han, K. Hausman, A. Herzog, J. Hsu, B. Ichter, A. Irpan, N. Joshi, R. Julian, D. Kalashnikov, Y. Kuang, I. Leal, L. Lee, T. E. Lee, S. Levine, Y. Lu, H. Michalewski, I. Mordatch, K. Pertsch, K. Rao, K. Reymann, M. Ryoo, G. Salazar, P. Sanketi, P. Sermanet, J. Singh, A. Singh, R. Soricut, H. Tran, V. Vanhoucke, Q. Vuong, A. Wahid, S. Welker, P. Wohlhart, J. Wu, F. Xia, T. Xiao, P. Xu, S. Xu, T. Yu, and B. Zitkovich (2023) RT-2: vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818. Cited by: §2.4.

[^5]: T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al. (2020) Language models are few-shot learners. In NeurIPS, Cited by: §1.

[^6]: W. Cao, M. Hallgarten, T. Li, D. Dauner, X. Gu, C. Wang, Y. Miron, M. Aiello, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, A. Geiger, and K. Chitta (2025) Pseudo-simulation for autonomous driving. In CoRL, Cited by: §A.2, Appendix F, §1, §3.1.

[^7]: S. Casas, A. Sadat, and R. Urtasun (2021) Mp3: a unified model to map, perceive, predict and plan. In CVPR, Cited by: §A.1.

[^8]: D. Charatan, S. Li, A. Tagliasacchi, and V. Sitzmann (2024) PixelSplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. In CVPR, Cited by: §E.2.

[^9]: L. Chen, P. Wu, K. Chitta, B. Jaeger, A. Geiger, and H. Li (2024) End-to-end autonomous driving: challenges and frontiers. TPAMI. Cited by: §A.1, §1, §3.2.

[^10]: Z. Chen, J. Yang, J. Huang, R. de Lutio, J. M. Esturo, B. Ivanovic, O. Litany, Z. Gojcic, S. Fidler, M. Pavone, L. Song, and Y. Wang (2025) OmniRe: omni urban scene reconstruction. In ICLR, Cited by: §A.2.

[^11]: C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song (2025) Diffusion policy: visuomotor policy learning via action diffusion. RSS. Cited by: §3.3.

[^12]: K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger (2022) Transfuser: imitation with transformer-based sensor fusion for autonomous driving. TPAMI. Cited by: §A.1, §B.2, Table 6, Table 7, Table 8, Appendix F, §1, §2.4, §3.1, Table 1, Table 2.

[^13]: D. Cornelisse, A. Pandya, K. Joseph, J. Suárez, and E. Vinitsky (2025) Building reliable sim driving agents by scaling self-play. arXiv preprint arXiv:2502.14706. Cited by: §A.3, §E.3.

[^14]: M. Cusumano-Towner, D. Hafner, A. Hertzberg, B. Huval, A. Petrenko, E. Vinitsky, E. Wijmans, T. Killian, S. Bowers, O. Sener, et al. (2025) Robust autonomy emerges from self-play. In ICML, Cited by: §A.3, §E.3, §3.3.

[^15]: C. Dang, S. Ang, Y. Li, H. Tian, J. Wang, G. Li, H. Ye, J. Ma, L. Chen, and Y. Wang (2026) DriveFine: refining-augmented masked diffusion vla for precise and robust driving. arXiv preprint arXiv:2602.14577. Cited by: §A.1.

[^16]: D. Dauner, M. Hallgarten, A. Geiger, and K. Chitta (2023) Parting with misconceptions about learning-based vehicle motion planning. In CoRL, Cited by: Table 5, Table 7, §1, §3.1, Table 1.

[^17]: D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, A. Geiger, and K. Chitta (2024) NAVSIM: data-driven non-reactive autonomous vehicle simulation and benchmarking. In NeurIPS, Cited by: §B.2, §1, §3.1.

[^18]: A. Dosovitskiy, G. Ros, F. Codevilla, A. Lopez, and V. Koltun (2017) CARLA: An open urban driving simulator. In CoRL, Cited by: §A.2.

[^19]: L. Feng, Y. Gao, E. Zablocki, Q. Li, W. Li, S. Liu, M. Cord, and A. Alahi (2025) RAP: 3d rasterization augmented end-to-end planning. arXiv preprint arXiv:2510.04333. Cited by: §A.3.

[^20]: H. Gao, S. Chen, B. Jiang, B. Liao, Y. Shi, X. Guo, Y. Pu, H. Yin, X. Li, X. Zhang, Y. Zhang, W. Liu, Q. Zhang, and X. Wang (2025) RAD: training an end-to-end driving policy via large-scale 3dgs-based reinforcement learning. In NeurIPS, Cited by: §A.1, §A.2, §A.4, §1.

[^21]: Y. Gao, D. Liu, Q. Zhang, Y. Zheng, H. Tian, G. Li, H. Ye, L. Chen, D. Ding, and D. Zhao (2026) Learning from mistakes: post-training for driving vla with takeover data. arXiv preprint arXiv:2603.14972. Cited by: §A.1.

[^22]: Y. Gao, Q. Zhang, D. Liu, Z. Xia, G. Li, K. Ma, G. Chen, H. Ye, L. Chen, D. Ding, and D. Zhao (2026) PerlAD: towards enhanced closed-loop end-to-end autonomous driving with pseudo-simulation-based reinforcement learning. arXiv preprint arXiv:2603.14908. Cited by: §A.1.

[^23]: M. Goff, G. Hogan, G. Hotz, A. du Parc Locmaria, K. Raczy, H. Schäfer, A. Shihadeh, W. Zhang, and Y. Yousfi (2025) Learning to drive from a world model. In CVPR, Cited by: §1.

[^24]: K. Guo, H. Liu, X. Wu, J. Pan, and C. Lv (2025) IPad: iterative proposal-centric end-to-end autonomous driving. arXiv preprint arXiv:2505.15111. Cited by: §A.1.

[^25]: K. He, X. Zhang, S. Ren, and J. Sun (2016) Deep residual learning for image recognition. In CVPR, Cited by: §B.2, §B.2, §B.2, Table 6, Table 6, Table 6.

[^26]: S. Hu, L. Chen, P. Wu, H. Li, J. Yan, and D. Tao (2022) St-p3: end-to-end vision-based autonomous driving via spatial-temporal feature learning. In ECCV, Cited by: §A.1.

[^27]: Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, et al. (2023) Planning-oriented autonomous driving. In CVPR, Cited by: §A.1, §2.4.

[^28]: X. Huang, E. M. Wolff, P. Vernaza, T. Phan-Minh, H. Chen, D. S. Hayden, M. Edmonds, B. Pierce, X. Chen, P. E. Jacob, et al. (2024) Drivegpt: scaling autoregressive behavior models for driving. arXiv preprint arXiv:2412.14415. Cited by: §A.3.

[^29]: J. Hwang, R. Xu, H. Lin, W. Hung, J. Ji, K. Choi, D. Huang, T. He, P. Covington, B. Sapp, et al. (2025) Emma: end-to-end multimodal model for autonomous driving. TMLR. Cited by: §A.3, §1.

[^30]: B. Jaeger, D. Dauner, J. Beißwenger, S. Gerstenecker, K. Chitta, and A. Geiger (2025) CaRL: learning scalable planning policies with simple rewards. In CoRL, Cited by: §A.4, §E.1.

[^31]: X. Jia, P. Wu, L. Chen, J. Xie, C. He, J. Yan, and H. Li (2023) Think twice before driving: towards scalable decoders for end-to-end autonomous driving. In CVPR, Cited by: §A.1.

[^32]: X. Jia, J. You, Z. Zhang, and J. Yan (2025) DriveTransformer: unified transformer for scalable end-to-end autonomous driving. In ICLR, Cited by: §A.1.

[^33]: B. Jiang, S. Chen, H. Gao, B. Liao, Q. Zhang, W. Liu, and X. Wang (2026) VADv2: end-to-end autonomous driving via probabilistic planning. In ICLR, Cited by: §A.1, §2.4, §3.3.

[^34]: B. Jiang, S. Chen, Q. Xu, B. Liao, J. Chen, H. Zhou, Q. Zhang, W. Liu, C. Huang, and X. Wang (2023) Vad: vectorized scene representation for efficient autonomous driving. In ICCV, Cited by: §A.1.

[^35]: J. Jiang, N. Song, J. Li, X. Zhu, and L. Zhang (2025) RealEngine: simulating autonomous driving in realistic context. arXiv preprint arXiv:2505.16902. Cited by: §A.2.

[^36]: M. Jiang, Y. Bai, A. Cornman, C. Davis, X. Huang, H. Jeon, S. Kulshrestha, J. Lambert, S. Li, X. Zhou, et al. (2024) Scenediffuser: efficient and controllable driving simulation initialization and rollout. In NeurIPS, Cited by: §A.2, §E.2.

[^37]: S. Jiao, K. Qian, H. Ye, Y. Zhong, Z. Luo, S. Jiang, Z. Huang, Y. Fang, J. Miao, Z. Fu, et al. (2025) EvaDrive: evolutionary adversarial policy optimization for end-to-end autonomous driving. arXiv preprint arXiv:2508.09158. Cited by: §A.1.

[^38]: J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei (2020) Scaling laws for neural language models. arXiv preprint arXiv:2001.08361. Cited by: §A.3, §1.

[^39]: N. Karnchanachari, D. Geromichalos, K. S. Tan, N. Li, C. Eriksen, S. Yaghoubi, N. Mehdipour, G. Bernasconi, W. K. Fong, Y. Guo, et al. (2024) Towards learning-based planning: the nuplan benchmark for real-world autonomous driving. In ICRA, Cited by: §B.2, Appendix F, §3.1, §3.1.

[^40]: B. Kerbl, G. Kopanas, T. Leimkühler, and G. Drettakis (2023) 3D gaussian splatting for real-time radiance field rendering. TOG. Cited by: §1, §1, §2.2.

[^41]: D. P. Kingma and J. Ba (2015) Adam: a method for stochastic optimization. In ICLR, Cited by: Table 6.

[^42]: Y. Lee and J. Park (2020) CenterMask: real-time anchor-free instance segmentation. In CVPR, Cited by: §B.2, Table 6.

[^43]: N. Lehtomaki, N. Sandell, and M. Athans (2003) Robustness results in linear-quadratic gaussian based multivariable control designs. TAC. Cited by: §2.3.

[^44]: B. Li, J. Guo, H. Liu, Y. Zou, Y. Ding, X. Chen, H. Zhu, F. Tan, C. Zhang, T. Wang, et al. (2025) Uniscene: unified occupancy-centric driving scene generation. In CVPR, Cited by: §E.2.

[^45]: D. Li, J. Ren, Y. Wang, X. Wen, P. Li, L. Xu, K. Zhan, Z. Xia, P. Jia, X. Lang, et al. (2025) Finetuning generative trajectory model with reinforcement learning from human feedback. arXiv preprint arXiv:2503.10434. Cited by: §A.1.

[^46]: H. Li, T. Li, J. Yang, H. Tian, C. Wang, L. Shi, M. Shang, Z. Lin, G. Wu, Z. Hao, et al. (2026) PlannerRFT: reinforcing diffusion planners through closed-loop and sample-efficient fine-tuning. arXiv preprint arXiv:2601.12901. Cited by: §A.4.

[^47]: K. Li, Z. Li, S. Lan, Y. Xie, Z. Zhang, J. Liu, Z. Wu, Z. Yu, and J. M. Alvarez (2025) Hydra-mdp++: advancing end-to-end driving via expert-guided hydra-distillation. arXiv preprint arXiv:2503.12820. Cited by: §3.1.

[^48]: Q. Li, X. Jia, S. Wang, and J. Yan (2024) Think2drive: efficient reinforcement learning by thinking with latent world model for autonomous driving (in carla-v2). In ECCV, Cited by: §A.4.

[^49]: Q. Li, Z. Peng, L. Feng, Q. Zhang, Z. Xue, and B. Zhou (2022) Metadrive: composing diverse driving scenarios for generalizable reinforcement learning. TPAMI. Cited by: §A.2.

[^50]: S. Li, N. Ye, T. Li, K. Chitta, T. An, P. Su, B. Wang, H. Liu, C. Lv, and H. Li (2025) Optimization-guided diffusion for interactive scene generation. arXiv preprint arXiv:2512.07661. Cited by: §E.2.

[^51]: T. Li, Y. Qiu, Z. Wu, C. Lindström, P. Su, M. Nießner, and H. Li (2025) MTGS: multi-traversal gaussian splatting. arXiv preprint arXiv:2503.12552. Cited by: Appendix F, §2.2, §2.2.

[^52]: X. Li, Y. Zhang, and X. Ye (2024) DrivingDiffusion: layout-guided multi-view driving scenarios video generation with latent diffusion model. In ECCV, Cited by: §A.2.

[^53]: Y. Li, S. Shang, W. Liu, B. Zhan, H. Wang, Y. Wang, Y. Chen, X. Wang, Y. An, C. Tang, et al. (2025) DriveVLA-w0: world models amplify data scaling law in autonomous driving. arXiv preprint arXiv:2510.12796. Cited by: §A.3.

[^54]: Y. Li, Y. Wang, Y. Liu, J. He, L. Fan, and Z. Zhang (2025) End-to-end driving with online trajectory evaluation via bev world model. arXiv preprint arXiv:2504.01941. Cited by: §A.1.

[^55]: Y. Li, K. Xiong, X. Guo, F. Li, S. Yan, G. Xu, L. Zhou, L. Chen, H. Sun, B. Wang, et al. (2026) ReCogDrive: a reinforced cognitive framework for end-to-end autonomous driving. In ICLR, Cited by: §A.1.

[^56]: Z. Li, K. Li, S. Wang, S. Lan, Z. Yu, Y. Ji, Z. Li, Z. Zhu, J. Kautz, Z. Wu, et al. (2024) Hydra-mdp: end-to-end multimodal planning with multi-target hydra-distillation. arXiv preprint arXiv:2406.06978. Cited by: §1, §2.4.

[^57]: Z. Li, S. Wang, S. Lan, Z. Yu, Z. Wu, and J. M. Alvarez (2025) Hydra-next: robust closed-loop driving with open-loop training. In ICCV, Cited by: §A.1.

[^58]: Z. Li, W. Yao, Z. Wang, X. Sun, J. Chen, N. Chang, M. Shen, J. Song, Z. Wu, S. Lan, et al. (2025) ZTRS: zero-imitation end-to-end autonomous driving with trajectory scoring. arXiv preprint arXiv:2510.24108. Cited by: §A.1.

[^59]: Z. Li, W. Yao, Z. Wang, X. Sun, J. Chen, N. Chang, M. Shen, Z. Wu, S. Lan, and J. M. Alvarez (2025) Generalized trajectory scoring for end-to-end multimodal planning. arXiv preprint arXiv:2506.06664. Cited by: §A.1, §B.2, Table 6, Table 7, Table 8, Appendix F, §1, §3.1, §3.1, Table 1, Table 2.

[^60]: B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang, et al. (2025) DiffusionDrive: truncated diffusion model for end-to-end autonomous driving. In CVPR, Cited by: §A.1, §B.2, Table 6, Table 7, Table 8, Appendix F, §1, §2.4, §3.1, Table 1, Table 2.

[^61]: H. Liu, Z. Huang, W. Huang, H. Yang, X. Mo, and C. Lv (2025) Hybrid-prediction integrated planning for autonomous driving. TPAMI. Cited by: §A.1.

[^62]: H. Liu, T. Li, H. Yang, L. Chen, C. Wang, K. Guo, H. Tian, H. Li, H. Li, and C. Lv (2025) Reinforced refinement with self-aware expansion for end-to-end autonomous driving. arXiv preprint arXiv:2506.09800. Cited by: §A.1, §1, §3.2.

[^63]: K. Liu, L. Shao, and S. Lu (2024) Novel view extrapolation with video diffusion priors. arXiv preprint arXiv:2411.14208. Cited by: §2.4.

[^64]: X. Liu, C. Zhou, and S. Huang (2024) 3DGS-enhancer: enhancing unbounded 3d gaussian splatting with view-consistent 2d diffusion priors. In NeurIPS, Cited by: §2.4.

[^65]: W. Ljungbergh, A. Tonderski, J. Johnander, H. Caesar, K. Åström, M. Felsberg, and C. Petersson (2024) NeuroNCAP: photorealistic closed-loop safety testing for autonomous driving. In ECCV, Cited by: §A.2.

[^66]: I. Loshchilov and F. Hutter (2019) Decoupled weight decay regularization. In ICLR, Cited by: Table 6, Table 6.

[^67]: A. Maddukuri, Z. Jiang, L. Y. Chen, S. Nasiriany, Y. Xie, Y. Fang, W. Huang, Z. Wang, Z. Xu, N. Chernyadev, S. Reed, K. Goldberg, A. Mandlekar, L. Fan, and Y. Zhu (2025) Sim-and-real co-training: a simple recipe for vision-based robotic manipulation. In RSS, Cited by: §1, §2.4.

[^68]: A. Naumann, X. Gu, T. Dimlioglu, M. Bojarski, A. Degirmenci, A. Popov, D. Bisla, M. Pavone, U. Muller, and B. Ivanovic (2025) Data scaling laws for end-to-end autonomous driving. In CVPR, Cited by: §A.3, §1.

[^69]: NVIDIA, N. C. Johan Bjorck andFernando Castañeda, X. Da, R. Ding, L. ”. Fan, Y. Fang, D. Fox, F. Hu, S. Huang, J. Jang, Z. Jiang, J. Kautz, K. Kundalia, L. Lao, Z. Li, Z. Lin, K. Lin, G. Liu, E. Llontop, L. Magne, A. Mandlekar, A. Narayan, S. Nasiriany, S. Reed, Y. L. Tan, G. Wang, Z. Wang, J. Wang, Q. Wang, J. Xiang, Y. Xie, Y. Xu, Z. Xu, S. Ye, Z. Yu, A. Zhang, H. Zhang, Y. Zhao, R. Zheng, and Y. Zhu (2025) GR00T N1: an open foundation model for generalist humanoid robots. arxiv preprint arxiv:2503.14734. Cited by: §1, §2.4.

[^70]: OpenScene Contributors (2023) OpenScene: the largest up-to-date 3d occupancy prediction benchmark in autonomous driving. Note: [https://github.com/OpenDriveLab/OpenScene](https://github.com/OpenDriveLab/OpenScene) Cited by: Appendix F.

[^71]: A. Popov, A. Degirmenci, D. Wehr, S. Hegde, R. Oldja, A. Kamenev, B. Douillard, D. Nistér, U. Muller, R. Bhargava, et al. (2024) Mitigating covariate shift in imitation learning for autonomous vehicles using latent space generative world models. arXiv preprint arXiv:2409.16663. Cited by: §1.

[^72]: A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. (2021) Learning transferable visual models from natural language supervision. In ICML, Cited by: §A.3, §1.

[^73]: Y. Shi, Y. Dong, Y. Ding, Y. Wang, X. Zhu, S. Zhou, W. Liu, H. Tian, R. Wang, H. Wang, et al. (2025) Realunify: do unified models truly benefit from unification? a comprehensive benchmark. arXiv preprint arXiv:2509.24897. Cited by: §E.2.

[^74]: W. Sun, X. Lin, Y. Shi, C. Zhang, H. Wu, and S. Zheng (2025) Sparsedrive: end-to-end autonomous driving via sparse scene representation. In ICRA, Cited by: §A.1.

[^75]: T. Tan, Y. Zheng, R. Liang, Z. Wang, K. Zheng, J. Zheng, J. Li, X. Zhan, and J. Liu (2025) Flow matching-based autonomous driving planning with advanced interactive behavior modeling. In NeurIPS, Cited by: §E.1.

[^76]: M. Treiber, A. Hennecke, and D. Helbing (2000) Congested traffic states in empirical observations and microscopic simulations. Physical review E. Cited by: §A.2, §E.2, §1, §2.3.

[^77]: X. Wang, Z. Zhu, G. Huang, X. Chen, J. Zhu, and J. Lu (2024) Drivedreamer: towards real-world-drive world models for autonomous driving. In ECCV, Cited by: §A.2.

[^78]: X. Weng, B. Ivanovic, Y. Wang, Y. Wang, and M. Pavone (2024) Para-drive: parallelized architecture for real-time autonomous driving. In CVPR, Cited by: §A.1.

[^79]: P. Wu, X. Jia, L. Chen, J. Yan, H. Li, and Y. Qiao (2022) Trajectory-guided control prediction for end-to-end autonomous driving: a simple yet strong baseline. In NeurIPS, Cited by: §A.1.

[^80]: Y. Wu, H. Zhang, T. Lin, L. Huang, S. Luo, R. Wu, C. Qiu, W. Ke, and T. Zhang (2025) Generating multimodal driving scenes via next-scene prediction. In CVPR, Cited by: §E.2.

[^81]: C. Xie, B. Sun, T. Li, J. Wu, Z. Hao, X. Lang, and H. Li (2026) LatentVLA: efficient vision-language models for autonomous driving via latent action prediction. arXiv preprint arXiv:2601.05611. Cited by: §A.1.

[^82]: Z. Xie, Z. Liu, Z. Peng, W. Wu, and B. Zhou (2025) Vid2Sim: realistic and interactive simulation from video for urban navigation. In CVPR, Cited by: §A.2.

[^83]: Z. Xing, X. Zhang, Y. Hu, B. Jiang, T. He, Q. Zhang, X. Long, and W. Yin (2025) Goalflow: goal-driven flow matching for multimodal trajectories generation in end-to-end autonomous driving. In CVPR, Cited by: §A.1, §2.4.

[^84]: R. Xu, H. Lin, W. Jeon, H. Feng, Y. Zou, L. Sun, J. Gorman, K. Tolstaya, S. Tang, B. White, et al. (2025) WOD-e2e: waymo open dataset for end-to-end driving in challenging long-tail scenarios. arXiv preprint arXiv:2510.26125. Cited by: §1.

[^85]: Z. Xu, B. Li, H. Gao, M. Gao, Y. Chen, M. Liu, C. Yan, H. Zhao, S. Feng, and H. Zhao (2025) Challenger: affordable adversarial driving video generation. arXiv preprint arXiv:2505.15880. Cited by: §A.2.

[^86]: Y. Yan, H. Lin, C. Zhou, W. Wang, H. Sun, K. Zhan, X. Lang, X. Zhou, and S. Peng (2024) Street gaussians: modeling dynamic urban scenes with gaussian splatting. In ECCV, Cited by: §A.2, §2.2.

[^87]: J. Yang, J. Huang, Y. Chen, Y. Wang, B. Li, Y. You, A. Sharma, M. Igl, P. Karkus, D. Xu, et al. (2025) Storm: spatio-temporal reconstruction model for large-scale outdoor scenes. In ICLR, Cited by: §E.2.

[^88]: J. Yang, K. Chitta, S. Gao, L. Chen, Y. Shao, X. Jia, H. Li, A. Geiger, X. Yue, and L. Chen (2025) ReSim: reliable world simulation for autonomous driving. In NeurIPS, Cited by: §1, §2.4.

[^89]: Y. Yang, H. Tian, Y. Shi, W. Xie, Y. Zhang, Y. Dong, Y. Hu, L. Wang, R. He, C. Shan, et al. (2025) A survey of unified multimodal understanding and generation: advances and challenges. Authorea Preprints. Cited by: §E.2.

[^90]: Y. Ye, J. Cen, J. Chen, and Z. Lu (2025) Self-evolved imitation learning in simulated world. arXiv preprint arXiv:2509.19460. Cited by: §E.1.

[^91]: J. You, X. Jia, Z. Zhang, Y. Zhu, and J. Yan (2024) Bench2Drive-r: turning real world data into reactive closed-loop autonomous driving benchmark by generative model. arXiv preprint arXiv:2412.09647. Cited by: §A.2.

[^92]: X. Zhai, A. Kolesnikov, N. Houlsby, and L. Beyer (2022) Scaling vision transformers. In CVPR, Cited by: §A.3, §1.

[^93]: C. Zhang, S. Biswas, K. Wong, K. Fallah, L. Zhang, D. Chen, S. Casas, and R. Urtasun (2024) Learning to drive via asymmetric self-play. In ECCV, Cited by: §A.3.

[^94]: H. Zhang, C. Li, W. Wu, S. Mao, Y. Zhang, H. Tian, I. Vulić, Z. Zhang, L. Wang, T. Tan, et al. (2025) Scaling and beyond: advancing spatial reasoning in mllms requires new recipes. arXiv preprint arXiv:2504.15037. Cited by: §E.2.

[^95]: H. Zhang, W. Wu, C. Li, N. Shang, Y. Xia, Y. Huang, Y. Zhang, L. Dong, Z. Zhang, L. Wang, et al. (2025) Latent sketchpad: sketching visual thoughts to elicit multimodal reasoning in mllms. arXiv preprint arXiv:2510.24514. Cited by: §E.2.

[^96]: Y. Zhang, H. Zhang, H. Tian, C. Fu, S. Zhang, J. Wu, F. Li, K. Wang, Q. Wen, Z. Zhang, et al. (2025) MME-realworld: could your multimodal llm challenge high-resolution real-world scenarios that are difficult for humans?. In ICLR, Cited by: §A.3.

[^97]: Z. Zhang, A. Liniger, D. Dai, F. Yu, and L. Van Gool (2021) End-to-end urban driving by imitating a reinforcement learning coach. In ICCV, Cited by: §2.3.

[^98]: Y. Zheng, R. Liang, K. ZHENG, J. Zheng, L. Mao, J. Li, W. Gu, R. Ai, S. E. Li, X. Zhan, and J. Liu (2025) Diffusion-based planning for autonomous driving with flexible guidance. In ICLR, Cited by: §E.1.

[^99]: Y. Zheng, Z. Xia, Q. Zhang, T. Zhang, B. Lu, X. Huo, C. Han, Y. Li, M. Yu, B. Jin, et al. (2024) Preliminary investigation into data scaling laws for imitation learning-based end-to-end autonomous driving. arXiv preprint arXiv:2412.02689. Cited by: §A.3, §1.

[^100]: H. Zhou, L. Lin, J. Wang, Y. Lu, D. Bai, B. Liu, Y. Wang, A. Geiger, and Y. Liao (2024) HUGSIM: a real-time, photo-realistic and closed-loop simulator for autonomous driving. arXiv preprint arXiv:2412.01718. Cited by: §A.2.

[^101]: J. Zhou, L. Lyu, Z. Tian, C. Zhuo, and Y. Li (2025) SafeMVDrive: multi-view safety-critical driving video synthesis in the real world domain. arXiv preprint arXiv:2505.17727. Cited by: §A.2.

[^102]: Y. Zhou, M. Simon, Z. Peng, S. Mo, H. Zhu, M. Guo, and B. Zhou (2024) Simgen: simulator-conditioned driving scene generation. In NeurIPS, Cited by: §A.2.

[^103]: Y. Zhou, N. Ye, W. Ljungbergh, T. Li, J. Yang, Z. Yang, H. Zhu, C. Petersson, and H. Li (2025) Decoupled diffusion sparks adaptive scene generation. In ICCV, Cited by: §A.2, §E.2.
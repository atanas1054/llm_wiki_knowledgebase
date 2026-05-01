---
title: "FeaXDrive: Feasibility-aware Trajectory-Centric Diffusion Planning for End-to-End Autonomous Driving"
source: "https://arxiv.org/html/2604.12656v2"
author:
published:
created: 2026-05-01
description:
tags:
  - "clippings"
---
Baoyun Wang <sup>1</sup> [wangby@tongji.edu.cn](https://arxiv.org/html/2604.12656v2/mailto:wangby@tongji.edu.cn) Zhuoren Li <sup>2</sup> [1911055@tongji.edu.cn](https://arxiv.org/html/2604.12656v2/mailto:1911055@tongji.edu.cn) Ran Yu [2433113@tongji.edu.cn](https://arxiv.org/html/2604.12656v2/mailto:2433113@tongji.edu.cn) Yu Che [rain\_car@tongji.edu.cn](https://arxiv.org/html/2604.12656v2/mailto:rain_car@tongji.edu.cn) Xinrui Zhang [2210805@tongji.edu.cn](https://arxiv.org/html/2604.12656v2/mailto:2210805@tongji.edu.cn) Ming Liu [2110215@tongji.edu.cn](https://arxiv.org/html/2604.12656v2/mailto:2110215@tongji.edu.cn) Jia Hu [hujia@tongji.edu.cn](https://arxiv.org/html/2604.12656v2/mailto:hujia@tongji.edu.cn) Lv Chen [lyuchen@ntu.edu.sg](https://arxiv.org/html/2604.12656v2/mailto:lyuchen@ntu.edu.sg) Bo Leng [lengbo@tongji.edu.cn](https://arxiv.org/html/2604.12656v2/mailto:lengbo@tongji.edu.cn)

###### Abstract

End-to-end diffusion planning has shown strong potential for autonomous driving, but the physical feasibility of generated trajectories remains insufficiently addressed. In particular, generated trajectories may exhibit local geometric irregularities, violate trajectory-level kinematic constraints, or deviate from the drivable area, indicating that the commonly used noise-centric formulation in diffusion planning is not yet well aligned with the trajectory space where feasibility is more naturally characterized. To address this issue, we propose FeaXDrive, a feasibility-aware trajectory-centric diffusion planning method for end-to-end autonomous driving. The core idea is to treat the clean trajectory as the unified object for feasibility-aware modeling throughout the diffusion process. Built on this trajectory-centric formulation, FeaXDrive integrates adaptive curvature-regularized training to improve intrinsic geometric and kinematic feasibility, drivable-area guidance within reverse diffusion sampling to enhance consistency with the drivable area, and feasibility-aware GRPO post-training to further improve planning performance while balancing trajectory-space feasibility. Experiments on the NAVSIM benchmark show that FeaXDrive achieves strong closed-loop planning performance while substantially improving trajectory-space feasibility. These findings highlight the importance of explicitly modeling trajectory-space feasibility in end-to-end diffusion planning and provide a step toward more reliable and physically grounded autonomous driving planners.

###### keywords:

Autonomous driving, End-to-end, Diffusion model, Trajectory-centric diffusion planning, Trajectory feasibility

\[inst1\] organization=College of Automotive and Energy Engineering, Tongji University, city=Shanghai, postcode=201804, country=China \[inst2\] organization=College of Transportation Engineering, Tongji University, city=Shanghai, postcode=201804, country=China \[inst3\] organization=School of Mechanical and Aerospace Engineering, Nanyang Technological University, postcode=639798, country=Singapore

## 1 Introduction

Autonomous driving is expected to play an important role in future intelligent transportation systems [^32]. In recent years, End-to-end (E2E) learning paradigm has attracted increasing attention, as it aims to directly map scene observations to driving actions within a unified framework [^16]. Representative works such as UniAD, VAD, and related E2E planning methods have demonstrated the potential of unified perception, prediction, and planning for autonomous driving [^16] [^20]. Building on this paradigm, VLM-enhanced E2E approaches further introduce multimodal semantic understanding and reasoning capabilities, thereby improving generalization and enabling better interpretation of complex traffic scenes, long-tail events, and high-level navigation intentions [^41] [^46] [^61]. Representative examples include LMDrive, which explores language-guided closed-loop E2E driving [^41], and DriveVLM, which demonstrates the potential of VLMs for understanding and planning in complex and long-tail driving scenarios [^46]. Meanwhile, diffusion-based planning has emerged as an increasingly prominent direction in E2E autonomous driving, owing to its strong capability for modeling multimodal driving behaviors and rich trajectory distributions [^59] [^33].

However, the physical feasibility of generated trajectories remains insufficiently addressed in existing E2E planning methods. For example, generated trajectories may exhibit abrupt point-wise discontinuities or unnatural deflections, and may violate kinematic limits when viewed from the perspective of the overall trajectory. Moreover, the generated trajectory, together with the corresponding vehicle spatial occupancy, may deviate from the drivable area, thereby breaking consistency between the trajectory and the geometric constraints of the scene, violating basic road-geometry constraints and potentially compromising traffic-rule compliance and the safety of other road users [^10]. Our reproduced baselines further show that drivable-area non-compliance constitutes a major source of planning failure (Table 1). In essence, these problems are different aspects of the same underlying issue: although the model is capable of generating semantically plausible future trajectories, the resulting trajectories do not satisfy both the intrinsic geometric and kinematic requirements of the trajectory itself and the spatial constraints imposed by the road environment [^25] [^44] [^39]. We refer to this notion of executability in trajectory space as trajectory-space feasibility.

Table 1: Failure cause distribution in score-zero planning scenes of reproduced diffusion-based planners. Percentages are computed with respect to the score-zero scenes of each planner.

| Failure Cause | DiffusionDrive [^33] | ReCogDrive [^30] |
| --- | --- | --- |
| Drivable-area non-compliance | 69.25% | 56.59% |
| At-fault collision | 32.42% | 45.44% |
| Both | 1.67% | 2.03% |

A common formulation in diffusion-based planning is noise-centric parameterization, in which the model predicts the noise term or noise residuals during both training and inference [^15] [^43] [^24]. Although this formulation is effective for general generative tasks, its prediction objective lies in noise space, making it difficult to explicitly represent or effectively characterize the physical feasibility of trajectories. As a result, feasibility-related signals can only influence the generation process indirectly through intermediate variables, making them difficult to incorporate into training and sampling in a stable and precise manner. This separation between the prediction space and the feasibility space results in longer and less direct propagation paths for feasibility-related signals, weaker training supervision, and less intuitive correction during inference, while also reducing the physical interpretability of the overall method.

Based on this observation, we construct E2E diffusion planning in a trajectory-centric manner, such that the future clean trajectory serves as the unified core object in both training and inference. This reformulation is particularly suitable for autonomous driving planning, because unlike high-dimensional generation targets such as natural images, a planned trajectory is low-dimensional, highly structured, and directly tied to physically interpretable planning variables [^28]. Local geometric regularity, curvature-related constraints, kinematic feasibility, and the spatial relationship between the vehicle footprint and the drivable area are all naturally expressed in clean trajectory space rather than noise space [^25] [^44] [^39]. As a result, directly modeling and optimizing the clean trajectory provides a unified interface for feasibility-aware training and sampling guidance.

In this work, we propose FeaXDrive. The core idea is to treat the clean trajectory as the explicit carrier of feasibility-related information throughout the diffusion process. By making it the shared optimization object for feasibility-aware modeling throughout training, inference, and post-training, the proposed method provides a unified interface for feasibility enhancement in trajectory space. Specifically, during training, we impose adaptive differentiable curvature constraints directly on the predicted clean trajectory; during inference, we inject drivable-area guidance into the clean trajectory estimated at each reverse sampling step, allowing local road-geometry priors to directly influence trajectory generation; and during post-training, we further incorporate feasibility-aware GRPO fine-tuning to improve planning performance while balancing trajectory-space feasibility. An overview of the proposed method is shown in Fig. 1.

This design allows feasibility-related information to be imposed at the same representation level as the generated trajectory. As a result, FeaXDrive provides a unified basis for training-time feasibility regularization, inference-time guidance, and post-training feasibility-aware optimization, thereby improving trajectory-space feasibility.

![[frame.png|Refer to caption]]

Figure 1: Overview of the proposed FeaXDrive. Compared with noise-centric diffusion planning, FeaXDrive adopts a trajectory-centric formulation in which the predicted clean trajectory serves as the unified object for feasibility-aware modeling. On this basis, the method combines training-time curvature feasibility regularization and inference-time drivable-area guidance to enhance trajectory-space feasibility throughout the diffusion planning process.

Our main contributions are summarized as follows:

- We propose a trajectory-centric diffusion planning framework for feasibility-aware E2E autonomous driving. Different from conventional noise-centric diffusion planners, FeaXDrive treats the predicted clean trajectory as the unified object throughout the diffusion process, which provides a direct trajectory-space interface for feasibility modeling across training, inference, and post-training, where intrinsic trajectory feasibility and drivable-area compliance can be explicitly modeled and optimized.
- We introduce an adaptive curvature-based feasibility regularization strategy to improve the intrinsic feasibility of generated trajectories. Based on differentiable curvature estimation from the predicted clean trajectory, we construct a speed-adaptive curvature bound that combines geometric curvature limits and lateral-acceleration limits. The resulting feasibility-regularized loss is jointly optimized with the trajectory prediction loss, suppressing local trajectory irregularities and curvature spikes during training.
- We develop a drivable-area guidance mechanism to improve drivable-area compliance during inference. By constructing a local drivable-area signed distance field and evaluating the vehicle footprint rather than only the trajectory centerline, the proposed guidance objective injects road-geometry priors into each reverse diffusion step. This mechanism progressively steers the diffusion sampling process toward drivable-area-compliant trajectories.
- We further incorporate feasibility-aware Group Relative Policy Optimization (GRPO) fine-tuning for policy optimization. By augmenting the evaluation-score-based planning reward with explicit trajectory-space feasibility preferences, the planner is encouraged to generate trajectories that better balance closed-loop planning performance and feasibility, thereby extending feasibility modeling to downstream policy optimization.

## 2 Related Work

### 2.1 E2E Autonomous Driving Planning

Autonomous driving systems were initially developed under modular pipelines, where perception, prediction, planning, and control were treated as separate components [^1] [^26] [^11]. As learning-based methods advanced, research gradually moved toward more unified architectures, eventually leading to E2E frameworks that directly map scene observations to planned trajectories. Representative methods such as TransFuser [^38], UniAD [^16], VAD [^20], and Hydra-MDP [^31] have demonstrated the potential of integrating perception, prediction, and planning in a single E2E architecture, thereby reducing the complexity of hand-crafted modular pipelines.

Building on this paradigm, subsequent research introduced large language models and vision-language models into E2E driving systems. Early works such as DriveGPT4 [^52], GPT-Driver [^34], Agent-Driver [^35], and LMDrive [^41] explored LLMs/VLMs for driving explanation, planning-oriented reasoning, and closed-loop E2E driving, showing that language modeling can provide richer high-level semantic priors than purely vision-based approaches. Subsequent methods, including DriveVLM [^46], OmniDrive [^48], Senna [^19], and ORION [^12], further advanced this direction through hierarchical planning, integrated perception-reasoning-planning, language-guided trajectory planning, and vision-language-instructed action generation. More recently, E2E frameworks have moved toward tighter integration of high-level reasoning and action generation, reflecting a broader shift toward more unified generative modeling frameworks for autonomous driving [^18] [^61].

### 2.2 Diffusion-Based E2E Trajectory Planning

The standard E2E paradigm based on imitation learning (IL) remains limited in its ability to capture the multimodal distribution of expert behaviors. Diffusion models [^45] have therefore emerged as an important generative framework for trajectory generation and autonomous driving planning. Compared with deterministic regression-based methods, diffusion-based planning is better suited to modeling the inherently multimodal nature of autonomous driving and can generate full future trajectories or action sequences through iterative denoising. Foundational diffusion methods [^15] [^43] provide a general framework for conditional generation and stable sampling, while action-diffusion methods [^6] further demonstrate the potential of diffusion models for continuous action modeling and policy learning. In autonomous driving and related trajectory-generation tasks, existing diffusion-based methods can be roughly grouped into three lines: multimodal trajectory or behavior distribution modeling, as exemplified by Guided Conditional Diffusion for Controllable Traffic Simulation [^60] and GoalFlow [^51]; action or policy diffusion, represented by Diffusion Policy [^6] and DiffE2E [^58], which apply diffusion directly in action space or hybrid action representations; and efficient diffusion planning for autonomous driving, represented by DiffusionDrive [^33], which improves sampling efficiency through truncated diffusion and anchor priors. In addition, works such as M2Diffuser [^53] show that diffusion models can also be combined with explicit trajectory optimization under stronger structural constraints.

Overall, existing diffusion-based trajectory planning has shown strong potential, yet trajectory-space feasibility has received relatively limited attention. Although the broader diffusion literature has explored guidance, constrained sampling, and projection-based refinement [^7] [^59], unified feasibility modeling in trajectory space across both training and inference is still lacking in autonomous driving.

### 2.3 Feasibility-Enhanced E2E Trajectory Planning

Existing studies on trajectory feasibility in autonomous driving have explored several directions. Some works improve consistency with scene structure and road geometry through map-aware representations, road-constrained losses, or additional geometric priors, thereby enhancing compliance with the drivable area and lane structure [^40] [^9] [^13]. Other methods adopt a predict-then-constrain paradigm, in which candidate trajectories are first generated by a learning-based model and then refined through external optimizers, MPC, or replacement mechanisms [^2] [^47] [^49]. More recent studies further inject constraints, rewards, or guidance signals directly into generation and policy learning, for example through sampling guidance [^54] [^59], reward modeling [^17], aligned policy optimization [^62] [^57] [^56] [^21], or human feedback [^27].

Despite these advances, existing feasibility-enhanced trajectory planning methods still have several limitations. Many rely on post-generation correction, validation, or replacement rather than unified feasibility modeling of the generation process itself [^2] [^47] [^49], making feasibility enforcement external and weakly coupled with trajectory generation. In guided generation frameworks, feasibility-related objectives are often introduced only as auxiliary signals rather than being directly integrated into both training and inference [^54] [^59] [^17] [^56] [^30], which may lead to weaker supervision, less intuitive inference-time correction, and reduced physical interpretability.

## 3 Problem Formulation

### 3.1 End-to-End Autonomous Driving Planning

We formulate E2E autonomous driving as conditional trajectory generation under a diffusion framework. Given a scene condition $\mathbf{c}$, which summarizes the driving context including sensor observations, historical ego states, and navigation instructions, the E2E autonomous driving system aims to generate a future motion trajectory. Beyond standard diffusion-based trajectory generation, this work focuses on trajectory-space feasibility, which remains insufficiently addressed in existing diffusion-based E2E approaches. Formally, we denote the future ego trajectory as:

$$
\mathbf{x}_{0}=[\mathbf{s}_{1},\mathbf{s}_{2},\ldots,\mathbf{s}_{H}]\in\mathbb{R}^{H\times d},
$$

where $\mathbf{s}_{i}$ denotes the ego waypoint at the $i$ -th future time step, $H$ is the planning horizon, and $d$ is the waypoint dimensionality. Each ego waypoint is represented as:

$$
\mathbf{s}_{i}=(p^{x}_{i},p^{y}_{i},p^{\theta}_{i}),
$$

where $(p^{x}_{i},p^{y}_{i})$ denotes the position of the ego pose center, and $p^{\theta}_{i}$ denotes the heading angle. Accordingly, the E2E planner aims to learn the conditional distribution $\mathbf{x}_{0}\sim p(\mathbf{x}_{0}\mid\mathbf{c})$, thereby directly mapping scene understanding to trajectory generation within a unified framework.

### 3.2 Diffusion-Based Trajectory Planning

Under the above E2E autonomous driving formulation, we model the conditional distribution of the future ego trajectory given the scene condition $\mathbf{c}$ using a diffusion generative framework. Let $\mathbf{x}_{0}$ denote the clean trajectory. In standard diffusion modeling, the forward process progressively injects Gaussian noise into $\mathbf{x}_{0}$, yielding a noisy intermediate state $\mathbf{x}_{t}$ at diffusion step $t$:

$$
\mathbf{x}_{t}=\sqrt{\bar{\alpha}_{t}}\,\mathbf{x}_{0}+\sqrt{1-\bar{\alpha}_{t}}\,\epsilon,\qquad\epsilon\sim\mathcal{N}(0,I),
$$

where $\bar{\alpha}_{t}$ denotes the cumulative signal-preservation coefficient determined by the noise schedule. Correspondingly, the reverse diffusion process aims to progressively recover the future ego trajectory from the noisy state, given the current diffusion state $\mathbf{x}_{t}$, the diffusion step $t$, and the scene condition $\mathbf{c}$. A common formulation in diffusion-based trajectory planning is noise-centric parameterization, in which the network is trained to predict the noise term in the current diffusion state:

$$
\hat{\epsilon}=f_{\theta}(\mathbf{x}_{t},t,\mathbf{c}).
$$

Under this formulation, both the learning objective and the reverse sampling updates are primarily defined in noise space, thereby modeling the conditional trajectory distribution $p(\mathbf{x}_{0}\mid\mathbf{c})$. Although diffusion-based planning is effective in modeling multimodal future driving behaviors and generating complete trajectories through iterative denoising, the noise-centric parameterization is not naturally aligned with trajectory-space feasibility, since feasibility-related properties are difficult to explicitly represent or directly optimize in noise space.

To address this issue, in this paper, we further construct a diffusion-based trajectory planning framework in which the clean trajectory is explicitly predicted and serves as the unified object for feasibility-aware modeling throughout training, inference, and post-training. The subsequent Methods section presents the corresponding design in detail and develops training-time feasibility regularization, inference-time geometric guidance, and post-training feasibility-aware optimization within this trajectory-centric framework.

## 4 Methods

This section details the proposed FeaXDrive, a feasibility-aware framework built around a trajectory-centric diffusion planning pipeline. By using the clean trajectory as the explicit interface, FeaXDrive incorporates adaptive differentiable curvature-regularized training for improving intrinsic trajectory feasibility, drivable-area guidance for improving drivable-area consistency during inference, and feasibility-aware GRPO post-training policy optimization for further improving planning performance while balancing trajectory-space feasibility. The overall architecture of FeaXDrive is illustrated in Fig. 2.

![[architecture.png|Refer to caption]]

Figure 2: Overall architecture of FeaXDrive. Under a trajectory-centric formulation, the predicted clean trajectory serves as the shared object for feasibility-aware modeling throughout training, inference, and post-training. The method integrates adaptive differentiable curvature-regularized training, drivable-area guidance during reverse diffusion sampling, and feasibility-aware GRPO post-training.

### 4.1 Trajectory-Centric Diffusion Planning

As described in Sec. 3, the planner takes scene images and navigation-related textual instructions as VLM inputs to extract visual-semantic features, which, together with the historical trajectory and ego states, serve as the conditioning information for diffusion planning. Standard diffusion planning typically adopts a noise-centric parameterization, where the network predicts the corresponding noise term $\hat{\epsilon}$ given the noisy trajectory $\mathbf{x}_{t}$, the diffusion step $t$, and the scene condition $\mathbf{c}$. Based on the above discussion, FeaXDrive adopts a trajectory-centric parameterization. Specifically, given the current diffusion state $\mathbf{x}_{t}$, diffusion step $t$, and scene condition $\mathbf{c}$, the planner directly predicts the corresponding clean trajectory estimate:

$$
\hat{\mathbf{x}}_{0}^{(t)}=f_{\theta}(\mathbf{x}_{t},t,\mathbf{c}),
$$

where $\hat{\mathbf{x}}_{0}^{(t)}$ denotes the estimated future trajectory at the current diffusion step, and $f_{\theta}$ is the conditional denoising network. The subsequent feasibility modeling in our method is built on this predicted clean trajectory estimate. During training, predicted clean trajectory estimate $\hat{\mathbf{x}}_{0}^{(t)}$ is directly supervised by the ground-truth future trajectory $\mathbf{x}^{\text{gt}}_{0}$; during inference, the current clean trajectory estimate is also explicitly produced at each reverse sampling step. Concretely, the basic supervision term in training is defined as

$$
\mathcal{L}_{x_{0}}=\left\|\hat{\mathbf{x}}_{0}^{(t)}-\mathbf{x}_{0}^{\mathrm{gt}}\right\|_{2}^{2}.
$$

During inference, at the $t$ -th reverse diffusion step, the planner first predicts $\hat{\mathbf{x}}_{0}^{(t)}$ from the current noisy state $\mathbf{x}_{t}$, and then uses this clean trajectory estimate to construct the next-state update:

$$
\mathbf{x}_{t-1}=G\!\left(\mathbf{x}_{t},\hat{\mathbf{x}}_{0}^{(t)},t\right),
$$

where $G(\cdot)$ denotes the reverse update operator associated with the specific sampler.

### 4.2 Adaptive Curvature-Regularized Training

Under the unified trajectory-centric parameterization, we introduce an adaptive curvature-regularized training strategy to improve the intrinsic feasibility of generated trajectories. By imposing explicit regularization in the training stage, the model is encouraged to reduce its tendency to generate trajectories with geometric spikes, local irregularities, or violations of speed-adaptive feasibility bounds.

#### 4.2.1 Differentiable Curvature Estimation and Adaptive Curvature Bound

Curvature estimation from discrete waypoints is sensitive to higher-order geometric variations and can be easily affected by local fluctuations and waypoint-level noise. To improve the stability of curvature estimation, we apply a lightweight differentiable smoothing operation to the planar coordinates of the predicted clean trajectory. Let the discrete planar position sequence of the predicted trajectory be

$$
\hat{\mathbf{P}}_{i}=(\hat{p}^{x}_{i},\hat{p}^{y}_{i}),\quad i=1,\ldots,H.
$$

We apply a fixed lightweight 1D convolution kernel to the planar position sequence of the predicted trajectory, yielding a smoothed position trajectory for curvature estimation.

$$
\tilde{\mathbf{P}}_{i}=\mathcal{S}\!\left(\hat{\mathbf{P}}_{i}\right),
$$

where $\mathcal{S}(\cdot)$ denotes the differentiable smoothing operator. This operation preserves the global trajectory trend while effectively suppressing local discrete spikes, thereby improving the robustness of subsequent curvature estimation.

After obtaining the smoothed trajectory, we estimate curvature based on arc-length parameterization rather than directly using temporal finite-difference approximation. This is because the spatial spacing between waypoints is not always uniform across different scenarios. Compared with directly differentiating a discrete time sequence, arc-length parameterization is more consistent with the geometric definition of a trajectory and is therefore better suited to characterizing its geometric shape.

First, the arc-length increment between adjacent smoothed waypoints is computed as

$$
\Delta\ell_{i}=\max\!\left(\left\|\tilde{\mathbf{P}}_{i+1}-\tilde{\mathbf{P}}_{i}\right\|_{2},\epsilon_{\ell}\right),\quad i=1,\ldots,H-1.
$$

where $\epsilon_{\ell}>0$ is the minimum arc-length constant used to prevent local arc-length degeneration. This gives the cumulative arc-length parameter $\ell_{i}$. Under the arc-length parameter $\ell$, the trajectory is regarded as a 2D curve $(\tilde{x}(\ell),\tilde{y}(\ell))$, and its first and second derivatives with respect to arc length are estimated. The corresponding curvature is estimated as

$$
\kappa_{i}=\frac{\tilde{x}^{\prime}(\ell_{i})\tilde{y}^{\prime\prime}(\ell_{i})-\tilde{y}^{\prime}(\ell_{i})\tilde{x}^{\prime\prime}(\ell_{i})}{\left(\tilde{x}^{\prime}(\ell_{i})^{2}+\tilde{y}^{\prime}(\ell_{i})^{2}\right)^{3/2}+\epsilon_{\kappa}},
$$

where $\epsilon_{\kappa}$ is a numerical stabilizer term. This arc-length-based estimation makes the curvature measure better reflect the intrinsic geometric shape of the trajectory, rather than artifacts introduced by discrete-time parameterization.

A fixed geometric curvature threshold is insufficient to characterize curvature-related feasibility across different motion regimes. In low-speed scenarios, curvature anomalies are mainly associated with local geometric irregularities and discrete waypoint fluctuations. As speed increases, however, even moderate curvature may induce excessive lateral-acceleration demand. Therefore, we construct a speed-aware curvature bound that combines a fixed geometric upper bound with lateral-acceleration limits.

Let the speed at the $i$ -th trajectory point be $v_{i}$, the maximum allowable lateral acceleration be $a_{\max}^{\mathrm{lat}}$, and the fixed geometric curvature upper bound be $\kappa_{\max}^{\mathrm{geo}}$. According to the lateral acceleration relation

$$
\left|a_{\mathrm{lat}}\right|=v^{2}\left|\kappa\right|
$$

the curvature upper bound implied by the lateral dynamic constraint can be derived as

$$
\kappa_{\max,i}^{\mathrm{dyn}}=\frac{a_{\max}^{\mathrm{lat}}}{v_{i}^{2}+\epsilon_{v}},
$$

where $\epsilon_{v}$ is a numerical stability term. Furthermore, we define the final curvature bound as

$$
\kappa_{i}^{\mathrm{adp}}=\min\!\left(\kappa_{\max}^{\mathrm{geo}},\ \kappa_{\max,i}^{\mathrm{dyn}}\right)=\min\!\left(\kappa_{\max}^{\mathrm{geo}},\ \frac{a_{\max}^{\mathrm{lat}}}{v_{i}^{2}+\epsilon_{v}}\right).
$$

This definition is essentially a hybrid geometric–dynamic constraint, reflecting the automatic switching of the dominant constraint term across different speed regimes. When the vehicle speed is low, the effective threshold is often given by the fixed geometric curvature upper bound $\kappa_{\max}^{\mathrm{geo}}$; as speed increases, the dynamic term gradually becomes more restrictive, causing the effective curvature upper bound to tighten automatically with the motion state, thereby more effectively constraining kinematic curvature violations. As a result, the adopted curvature constraint not only preserves local geometric regularity, but also provides a feasibility constraint that is more consistent with vehicle kinematics.

#### 4.2.2 Trajectory Feasibility-Aware Loss

Based on the above curvature estimation and adaptive bound, we define a dynamics-aware training loss to penalize trajectory segments whose curvature exceeds the speed-adaptive curvature bound:

$$
\mathcal{L}_{\mathrm{cur}}=\frac{1}{H}\sum_{i=1}^{H}\max\!\left(|\kappa_{i}|-\kappa_{i}^{\mathrm{adp}},0\right)^{2}.
$$

Accordingly, the total training loss can be written as

$$
\mathcal{L}_{\mathrm{train}}=\mathcal{L}_{x_{0}}+\lambda_{\mathrm{cur}}\mathcal{L}_{\mathrm{cur}}.
$$

where $\lambda_{\mathrm{cur}}$ is the weight that balances trajectory supervision and kinematic regularization. The role of $\mathcal{L}_{\mathrm{cur}}$ is to impose a more physically meaningful bias on the trajectory distribution, thereby suppressing local geometric irregularities and improving trajectory-level kinematic feasibility.

### 4.3 Constraint-Aware Inference with Drivable-Area Guidance

To further improve trajectory-space feasibility, especially the spatial consistency between the generated trajectory and the drivable area, we introduce constraint-aware diffusion sampling with drivable-area guidance during inference. The core idea is that, in the stages of reverse diffusion, the sampler no longer relies only on the model’s generative distribution; instead, local drivable-area geometric priors are injected as guidance into the clean trajectory estimate at each step, enabling online scene-aware geometric correction.

#### 4.3.1 Guidance within Reverse Diffusion Sampling

Starting from an initial noisy trajectory $\mathbf{x}_{T}\sim\mathcal{N}(0,I)$, the reverse diffusion process iteratively performs guidance-aware updates for $t=T,T-1,\dots,1$. At each reverse sampling step $t$, we first predict a clean trajectory estimate $\hat{\mathbf{x}}_{0}^{(t)}$ from the current diffusion state $\mathbf{x}_{t}$. Instead of directly using $\hat{\mathbf{x}}_{0}^{(t)}$ to generate the next state, we apply a constraint-based correction based on the local road-geometry prior $\mathcal{M}$, yielding a guided trajectory estimate:

$$
\tilde{\mathbf{x}}_{0}^{(t)}=\mathcal{C}\!\left(\hat{\mathbf{x}}_{0}^{(t)};\mathcal{M}\right),
$$

where $\mathcal{M}$ denotes the local drivable-area geometric prior, and $\mathcal{C}(\cdot)$ denotes the constraint operator corresponding to the guidance. The sampler then continues the reverse update based on the corrected clean trajectory:

$$
\mathbf{x}_{t-1}=G\!\left(\mathbf{x}_{t},\tilde{\mathbf{x}}_{0}^{(t)},t\right),\qquad t=T,\dots,1.
$$

After the final reverse step, the resulting clean trajectory is taken as the planned trajectory.

The geometric guidance in our method acts on the predicted $\mathbf{x}_{0}$ at every reverse sampling step and is directly integrated into the reverse sampling chain. It is not an external post-processing step applied after trajectory generation, but an online geometric correction mechanism within the sampling loop. As a result, the guidance directly influences the subsequent evolution of the sampling chain, steering the trajectory toward better consistency with the drivable area during the progressive denoising process.

#### 4.3.2 Local Drivable-Area SDF Construction

To introduce drivable-area priors during inference, we define a local drivable-area geometric prior $\mathcal{M}$, which is instantiated in the current implementation as a drivable-area signed distance field (SDF) constructed from a local HD map.

Specifically, we first transform the local road-geometry information into the ego-centric coordinate system with respect to the current ego vehicle, and construct a spatial representation of the drivable region within a finite local window. We then rasterize this local drivable region and compute the corresponding signed distance field. Let the local drivable region be $\mathcal{D}\subset\mathbb{R}^{2}$, with boundary $\partial\mathcal{D}$. For any point $\mathbf{q}\in\mathbb{R}^{2}$ in the local ego-centric plane, we define the signed distance field as

$$
S(\mathbf{q})=\begin{cases}\operatorname{dist}(\mathbf{q},\partial\mathcal{D}),&\mathbf{q}\in\mathcal{D},\\
0,&\mathbf{q}\in\partial\mathcal{D},\\
-\operatorname{dist}(\mathbf{q},\partial\mathcal{D}),&\mathbf{q}\notin\mathcal{D}.\end{cases}
$$

where $\mathcal{D}$ denotes the drivable region and $\partial\mathcal{D}$ denotes its boundary. Accordingly, the sign of $S(\mathbf{q})$ characterizes the topological relationship between the point and the drivable region, while its magnitude $\lvert S(\mathbf{q})\rvert$ explicitly measures the distance from $\mathbf{q}$ to the boundary, i.e., the geometric safety margin or the degree of off-road violation. In particular, when $S(\mathbf{q})\geq m$, the point lies inside the drivable region and maintains a safety margin of at least $m$ from the boundary.

The SDF transforms the drivable-area constraint from a discrete inside/outside binary test into a continuous geometric distance signal. This not only enables us to determine whether a trajectory goes off-road, but also quantifies its safety margin and degree of boundary violation, thereby providing stable and differentiable geometric information for subsequent gradient-based guidance.

It should be noted that, in our problem formulation, $\mathcal{M}$ is regarded as a unified interface for local road-geometry priors, rather than being tied to any specific map representation. In our current experiments, $\mathcal{M}$ is instantiated from a local HD map, but it can also be replaced in the future by local geometric information provided by lightweight maps, online mapping, or implicit road priors.

#### 4.3.3 Footprint-Level Drivable-Area Guidance

Applying drivable-area constraints only to the trajectory center point can easily overlook the relationship between the actual occupied region of the vehicle and the road boundary. In particular, in turning, near-boundary driving, or narrow-road scenarios, the fact that the center point remains inside the road does not necessarily imply that the entire vehicle stays within the drivable region. Therefore, we adopt a footprint-level geometric constraint.

For the vehicle waypoint state at future time step $i$, we construct a rectangular vehicle footprint according to the vehicle length and width. Let the relative coordinates of the four footprint corners in the vehicle coordinate system be $\{\delta_{j}\}_{j=1}^{4}$. Their positions in the local ego-centric plane are then given by

$$
\mathbf{p}_{i,j}=\begin{bmatrix}p^{x}_{i}\\
p^{y}_{i}\end{bmatrix}+R(p^{\theta}_{i})\,\delta_{j},\qquad j=1,\ldots,4,
$$

where $R(p^{\theta}_{i})$ denotes the 2D rotation matrix determined by the heading angle $p^{\theta}_{i}$. We then perform bilinear sampling on the local SDF at these continuous coordinates to obtain the corresponding signed distance at each corner point:

$$
d_{i,j}=S(\mathbf{p}_{i,j}),\qquad j=1,\ldots,4.
$$

This footprint-based sampling strategy better reflects the true occupied geometry of the vehicle than a center-point constraint. On the one hand, it can explicitly capture cases where a vehicle corner leaves the drivable region while the center point still remains inside. On the other hand, since SDF sampling is continuous and differentiable, gradients can be propagated directly to the vehicle position and heading, enabling more refined geometric guidance.

After obtaining the footprint-level distances, we define a soft barrier guidance objective with a safety-margin term to measure the consistency between the current clean trajectory and the drivable region:

$$
\mathcal{L}_{\mathrm{drv}}\!\left(\hat{\mathbf{x}}_{0}^{(t)};\mathcal{M}\right)=\frac{1}{4H}\sum_{i=1}^{H}\sum_{j=1}^{4}\phi\!\left(m_{\mathrm{safe}}-d_{i,j}\right),
$$

where $m_{\mathrm{safe}}$ denotes the desired safety margin to the boundary, and $\phi(\cdot)$ is implemented as a softplus barrier function. When the minimum footprint distance is sufficiently large, the guidance objective approaches zero; when the footprint approaches the boundary or leaves the drivable region, it increases rapidly.

To reduce unnecessary interference with reasonable trajectories, we adopt a trigger mechanism based on whether the footprint goes out of bounds. Guidance is activated only when the footprint points of the current predicted trajectory leave the drivable region or come excessively close to the boundary. For triggered samples, at sampling step $t$, we perform one or multiple updates along the gradient direction of this objective with respect to $\hat{\mathbf{x}}_{0}^{(t)}$, yielding the guided clean trajectory estimate:

$$
\tilde{\mathbf{x}}_{0}^{(t)}=\hat{\mathbf{x}}_{0}^{(t)}-\eta_{t}\Phi\!\left(\nabla_{\hat{\mathbf{x}}_{0}^{(t)}}\mathcal{L}_{\mathrm{drv}}\right),
$$

where $\eta_{t}$ denotes the guidance step size at sampling step $t$, and $\Phi(\cdot)$ denotes the gradient normalization and scale modulation operator. In practice, we normalize the gradients on the positional dimensions to avoid excessively large variations in SDF gradient scales across different scenes. In addition, an independent scaling factor can be introduced for the heading dimension to balance geometric correction and trajectory smoothness.

In summary, drivable-area guidance performs online progressive correction on the clean trajectory estimate at each sampling step using the local drivable-area geometry of the current scene, gradually steering it away from non-drivable regions and back toward drivable-area-compliant trajectory generation.

Compared with the adaptive curvature-regularized training, drivable-area guidance provides scene-specific geometric correction for the current scene. The former mainly improves the intrinsic feasibility of trajectories, while the latter mainly enhances spatial consistency with the drivable region. By combining these two modules under a unified trajectory-centric representation, we develop a feasibility-aware diffusion planning method for autonomous driving trajectory planning.

### 4.4 Reinforcement Learning Fine-Tuning with Feasibility-Aware GRPO

Starting from the supervised diffusion planner, we further fine-tune the diffusion planning policy using reinforcement learning (RL) [^22] to improve closed-loop planning performance beyond imitation learning while preserving trajectory-space feasibility. Specifically, we instantiate this RL fine-tuning stage with the proposed Feasibility-Aware GRPO. Following the group-relative policy optimization principle [^42] [^14], Feasibility-Aware GRPO adapts relative policy optimization to the diffusion generation chain and explicitly incorporates trajectory-space feasibility into the reward design. The core idea is to augment the evaluation-score-based planning reward with feasibility preferences, so that policy optimization favors candidate trajectories that achieve both strong benchmark performance and high trajectory-space feasibility.

Specifically, given a scene condition $\mathbf{c}$, the current policy samples a group of candidate trajectories

$$
\left\{\mathbf{x}_{0}^{(g)}\right\}_{g=1}^{G},\qquad\mathbf{x}_{0}^{(g)}\sim\pi_{\theta}(\cdot\mid\mathbf{c}),
$$

and obtains the corresponding rewards through the planning evaluator

$$
r_{g}=\mathcal{R}\!\left(\mathbf{x}_{0}^{(g)},\mathbf{c}\right).
$$

Here, $\mathcal{R}(\cdot)$ jointly characterizes task quality and feasibility preference, and is defined as

$$
\mathcal{R}\left(\mathbf{x}_{0},\mathbf{c}\right)=\mathcal{R}_{\mathrm{task}}\left(\mathbf{x}_{0},\mathbf{c}\right)+\lambda_{\mathrm{fea}}\mathcal{R}_{\mathrm{fea}}\left(\mathbf{x}_{0},\mathbf{c}\right),
$$

where $\mathcal{R}_{\mathrm{task}}$ measures the task-level planning quality, $\mathcal{R}_{\mathrm{fea}}$ represents the trajectory feasibility preference, and $\lambda_{\mathrm{fea}}$ is the trade-off coefficient between the two terms. As a result, the reward optimized by GRPO is no longer determined solely by benchmark-oriented performance, but instead explicitly incorporates a preference modeling for trajectory-space feasibility. In this work, $\mathcal{R}_{\mathrm{fea}}$ is characterized by the speed-adaptive curvature feasibility defined in Section 4.2. Specifically, we directly adopt the curvature estimation method and the speed-adaptive constraint criterion introduced in Section 4.2, and incorporate them into the post-training reward design. If a generated trajectory satisfies the curvature-feasibility requirement defined in Section 4.2, it receives a higher feasibility reward; otherwise, its feasibility reward is reduced accordingly, causing it to be disadvantaged in the within-group comparison. In this way, the trajectory-space feasibility modeling introduced above further enters the post-training policy optimization process through feasibility-aware reward shaping.

After obtaining the rewards of $G$ candidate trajectories for the same scene, GRPO performs relative normalization of the rewards within the group to construct the relative advantage of each trajectory:

$$
A_{g}=\frac{r_{g}-\mu_{r}}{\sigma_{r}+\epsilon_{r}},
$$

where $\mu_{r}$ and $\sigma_{r}$ denote the mean and standard deviation of the rewards within the same group, respectively, and $\epsilon_{r}$ is a numerical stability term. Since the diffusion planner generates the final trajectory through a multi-step reverse denoising process, policy optimization does not act only on the terminal trajectory variable, but on the entire denoising chain that generates the trajectory. Accordingly, the GRPO policy optimization term can be written as

$$
\mathcal{L}_{\mathrm{RL}}=-\mathbb{E}\!\left[A_{g}\sum_{t=1}^{T}w_{t}\log\pi_{\theta}\!\left(\mathbf{x}_{t-1}^{(g)}\mid\mathbf{x}_{t}^{(g)},\mathbf{c}\right)\right],
$$

where $w_{t}$ denotes the denoising-step-related weight. This objective uses the trajectory-level relative advantage to weight the log-likelihood terms along the reverse denoising chain, encouraging the policy to increase the likelihood of candidate trajectories that achieve both high task rewards and favorable feasibility scores. In our implementation, we adopt a single-update GRPO objective for diffusion denoising chains. Each sampled group is used for one policy update, and we do not perform multi-epoch PPO-style reuse of the same samples. Therefore, we do not introduce a clipped likelihood-ratio objective with respect to a separate old policy. The resulting optimization reduces to an advantage-weighted log-likelihood objective over the reverse denoising chain.

In addition, to prevent RL post-training from deviating excessively from the policy prior learned during the IL stage, we introduce a behavior-cloning regularization term based on a fixed reference policy $\pi_{\mathrm{ref}}$:

$$
\mathcal{L}_{\mathrm{BC}}=-\mathbb{E}_{\mathbf{c},\,\mathbf{x}_{1:T}^{\mathrm{ref}}\sim\pi_{\mathrm{ref}}}\left[\sum_{t=1}^{T}\log\pi_{\theta}\!\left(\mathbf{x}_{t-1}^{\mathrm{ref}}\mid\mathbf{x}_{t}^{\mathrm{ref}},\mathbf{c}\right)\right].
$$

Here, $\pi_{\mathrm{ref}}$ is the fixed IL policy, and $\mathbf{x}_{T}^{\mathrm{ref}}\rightarrow\cdots\rightarrow\mathbf{x}_{0}^{\mathrm{ref}}$ denotes the reference denoising chain. The final post-training objective is

$$
\mathcal{L}=\mathcal{L}_{\mathrm{RL}}+\lambda_{\mathrm{BC}}\mathcal{L}_{\mathrm{BC}}.
$$

Overall, as the post-training stage of this work, Feasibility-Aware GRPO strengthens the model’s preference for high-quality and feasible trajectories from the perspective of policy optimization through feasibility-aware reward shaping. Therefore, the proposed trajectory-centric diffusion planner not only incorporates trajectory-space feasibility modeling during the IL training stage and the sampling stage, but also remains compatible with subsequent policy optimization, thereby further improving the overall balance between benchmark performance and trajectory-space feasibility.

## 5 Experiments

This section evaluates the proposed method in terms of standard planning performance, trajectory-space feasibility, module-wise contributions, and inference efficiency. We first present the experimental setup, then report the main results, followed by ablation and feasibility analysis, efficiency evaluation, and qualitative visualization.

### 5.1 Experimental Setup

#### 5.1.1 Dataset and Benchmark

We conduct experiments on the NAVSIM benchmark [^10], a planning-oriented autonomous driving dataset built on OpenScene [^36], which is a redistribution of nuPlan [^23]. NAVSIM provides multimodal driving observations together with closed-loop evaluation for end-to-end autonomous driving planning, and is split into navtrain (1,192 training scenes) and navtest (136 evaluation scenes). In each scene, the planner takes scene images, historical ego states, and navigation-related context as input, and predicts the future ego trajectory. Following the standard NAVSIM evaluation protocol, we report both benchmark planning metrics and the trajectory-space feasibility metrics considered in this work.

#### 5.1.2 Implementation Details

Our model follows a VLM-conditioned diffusion planning pipeline and consists of two main components: a vision-language encoder and a diffusion-based trajectory planner. Given a front-view scene image, navigation-related textual instructions, and historical ego states, the vision-language encoder first extracts semantic driving-context representations from the visual and textual inputs. These VLM hidden states, together with historical ego-state features and diffusion-timestep embeddings, form the condition context $\mathbf{c}$ for the diffusion planner. The diffusion planner is implemented as a DiT-based trajectory denoising network [^37], which takes the noisy trajectory state and $\mathbf{c}$ as input and directly predicts the clean future ego trajectory under a trajectory-centric diffusion parameterization. The trajectory denoising network adopts a DiT architecture, with $8$ attention heads and $16$ transformer layers. Based on this pipeline, FeaXDrive introduces adaptive curvature-regularized training during IL training, drivable-area guidance during reverse diffusion sampling, and feasibility-aware GRPO during post-training.

To define the curvature bound used in both training and evaluation, we align the geometric curvature limit with the minimum turning radius of the Chrysler Pacifica, the data-collection vehicle used in NAVSIM. According to the official specification, the minimum turning radius of the Chrysler Pacifica is approximately $19.8\,\mathrm{ft}$ ($\approx 6.0\,\mathrm{m}$) [^8], which, under a low-speed geometric approximation, corresponds to $\kappa_{\mathrm{geo}}\approx 1/{R_{\min}}={1}/{6.0}\approx 0.166\,\mathrm{m}^{-1}.$ Accordingly, we set the fixed geometric curvature bound to $\kappa_{\mathrm{geo}}=0.166\,\mathrm{m}^{-1}$, and set the maximum allowable lateral acceleration to $a_{\max}^{\mathrm{lat}}=6\,\mathrm{m/s}^{2}$ ($\approx 0.61g$) as an upper bound for near-limit maneuvering. Together, these values are used to construct the speed-aware curvature bound that combines a fixed geometric upper bound with lateral-acceleration limits.

The IL-based diffusion model is first trained for $100$ epochs on $4$ A800 GPUs, using bf16 mixed precision and distributed data parallel (DDP), with a per-GPU batch size of $32$ and a total batch size of $128$. The vision-language encoder is initialized from the same pretrained VLM checkpoint as ReCogDrive [^30], based on InternVL3-2B [^5], and its parameters are frozen during training.

We further conduct a feasibility-aware GRPO fine-tuning stage. Starting from the IL-trained model, we perform GRPO fine-tuning for $1$ epoch on $8$ A800 GPUs with a per-GPU batch size of $8$. Unlike score-only post-training, the GRPO reward incorporates curvature-violation terms together with benchmark score terms, thereby further improving planning performance while balancing trajectory-space feasibility.

During inference, we use DDIM sampling for all variants. Unless otherwise specified, all internal variants share the same inference settings.

#### 5.1.3 Evaluation Metrics

We evaluate our method using two groups of metrics:

- Standard planning metrics: For the NAVSIM benchmark, we follow the official Predictive Driver Model Score (PDMS), including no-at-fault collision (NC), drivable-area compliance (DAC), time-to-collision (TTC), comfort (Comf.), and ego progress (EP). Higher PDMS indicates better overall planning performance.
- Trajectory-space feasibility metrics: To evaluate the main focus of this work, we further consider trajectory-space feasibility. Specifically, we report the curvature violation rate under the adaptive curvature bound and the drivable-area violation rate. Let $N$ denote the total number of evaluated planning scenes. For scene $i$, let $\mathbb{I}^{(i)}_{\kappa}=1$ if the generated trajectory violates the adaptive curvature bound, and $\mathbb{I}^{(i)}_{\mathrm{drv}}=1$ if the generated trajectory violates the drivable-area constraint. The two rates are defined as
	$$
	\mathrm{Curvature\ Viol.\ Rate}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{I}^{(i)}_{\kappa},\qquad\mathrm{Drivable\ Area\ Viol.\ Rate}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{I}^{(i)}_{\mathrm{drv}}.
	$$
	The former reflects the intrinsic geometric and kinematic aspect of trajectory-space feasibility, while the latter reflects consistency with the drivable area. Together, these metrics evaluate whether the generated trajectories remain feasible in trajectory space while maintaining competitive benchmark performance.

### 5.2 Main Results

Table 2 reports the main closed-loop planning results on the NAVSIM benchmark. Overall, the proposed method achieves strong planning performance under both imitation learning (IL) and reinforcement learning fine-tuning (RLFT). Under the IL setting, FeaXDrive-IL achieves a PDMS of 88.7, outperforming all other IL baselines, including DiffusionDrive ($88.1$), WoTE ($88.3$), and ReCogDrive-IL ($86.8$). In particular, our method achieves the highest DAC ($97.5$) and the highest ego progress ($83.3$) among the IL methods, indicating that the proposed method remains highly competitive on the standard NAVSIM benchmark even before RLFT.

After RLFT, FeaXDrive further improves the PDMS to 90.0, demonstrating that the proposed planner remains compatible with downstream policy optimization. Although its final PDMS is slightly lower than that of ReCogDrive w/GRPO ($90.5$), our method achieves a substantially higher DAC ($98.3$ vs. $96.7$), indicating better compliance with the drivable area. Overall, these results show that the proposed method maintains competitive benchmark performance while exhibiting stronger drivable-area consistency.

Table 2: Main closed-loop planning results on the NAVSIM benchmark. RLFT denotes reinforcement learning fine-tuning; for FeaXDrive, it is instantiated as feasibility-aware GRPO.

<table><tbody><tr><td>Method type</td><td>Model</td><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Comf. <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td rowspan="11">IL</td><td>VADv2 <sup><a href="#fn:3">3</a></sup></td><td>97.2</td><td>76.0</td><td>100</td><td>91.6</td><td>89.1</td><td>80.9</td></tr><tr><td>Driving-GPT <sup><a href="#fn:4">4</a></sup></td><td>98.9</td><td>79.7</td><td>95.6</td><td>94.9</td><td>90.7</td><td>82.4</td></tr><tr><td>Hydra-MDP <sup><a href="#fn:31">31</a></sup></td><td>97.9</td><td>77.6</td><td>100</td><td>92.9</td><td>91.7</td><td>83.0</td></tr><tr><td>UniAD <sup><a href="#fn:16">16</a></sup></td><td>97.8</td><td>78.8</td><td>100</td><td>92.9</td><td>91.9</td><td>83.4</td></tr><tr><td>PARA-Drive <sup><a href="#fn:50">50</a></sup></td><td>97.9</td><td>79.3</td><td>99.8</td><td>93.0</td><td>92.4</td><td>84.0</td></tr><tr><td>TransFuser <sup><a href="#fn:38">38</a></sup></td><td>97.7</td><td>79.2</td><td>100</td><td>92.8</td><td>92.8</td><td>84.0</td></tr><tr><td>DRAMA <sup><a href="#fn:55">55</a></sup></td><td>98.0</td><td>80.1</td><td>100</td><td>94.8</td><td>93.1</td><td>85.5</td></tr><tr><td>ReCogDrive-IL <sup><a href="#fn:30">30</a></sup></td><td>98.3</td><td>81.1</td><td>100</td><td>94.3</td><td>95.1</td><td>86.8</td></tr><tr><td>DiffusionDrive <sup><a href="#fn:33">33</a></sup></td><td>98.2</td><td>82.2</td><td>100</td><td>94.7</td><td>96.2</td><td>88.1</td></tr><tr><td>WoTE <sup><a href="#fn:29">29</a></sup></td><td>98.5</td><td>81.9</td><td>99.9</td><td>94.9</td><td>97.3</td><td>88.3</td></tr><tr><td>FeaXDrive-IL (Ours)</td><td>98.1</td><td>83.3</td><td>100</td><td>93.6</td><td>97.5</td><td>88.7</td></tr><tr><td rowspan="3">IL+RLFT</td><td>TransFuser w/GRPO <sup><a href="#fn:38">38</a></sup></td><td>98.0</td><td>88.5</td><td>100</td><td>96.6</td><td>94.7</td><td>87.9</td></tr><tr><td>ReCogDrive w/GRPO <sup><a href="#fn:30">30</a></sup></td><td>98.1</td><td>85.9</td><td>100</td><td>95.0</td><td>96.7</td><td>90.5</td></tr><tr><td>FeaXDrive (Ours)</td><td>98.2</td><td>84.2</td><td>100</td><td>94.7</td><td>98.3</td><td>90.0</td></tr></tbody></table>

To further evaluate the main focus of this work, Table 3 reports the curvature violation rate under a unified evaluation pipeline. We compare our method with two reproduced diffusion-based baselines, namely DiffusionDrive [^33] and ReCogDrive [^30]. Since such feasibility metrics are generally not reported in prior work and are not always directly available from released checkpoints or evaluation code, we reproduce these representative baselines to enable a fair comparison under the same evaluation setup. Under IL training, FeaXDrive-IL achieves a curvature violation rate of only $0.88\%$, substantially lower than DiffusionDrive ($8.59\%$) and ReCogDrive-IL ($8.05\%$). After Feasibility-Aware GRPO fine-tuning, FeaXDrive maintains a low curvature violation rate of $2.40\%$, while ReCogDrive w/GRPO increases to $15.5\%$. These results suggest that score-oriented post-training may improve benchmark metrics at the cost of trajectory-space feasibility, whereas the proposed feasibility-aware design is able to preserve much stronger curvature feasibility while still achieving competitive benchmark performance.

Table 3: Comparison of curvature violation rates among reproduced diffusion-based planners.

| Method | DiffusionDrive [^33] | ReCogDrive-IL [^30] | ReCogDrive w/GRPO [^30] | FeaXDrive-IL (Ours) | FeaXDrive w/FA-GRPO (Ours) |
| --- | --- | --- | --- | --- | --- |
| Curvature Viol. Rate $\downarrow$ | 8.59% | 8.05% | 15.5% | 0.88% | 2.40% |

Overall, the results demonstrate that FeaXDrive not only delivers strong planning performance on NAVSIM, but also enhances trajectory-space feasibility. Across both IL training and post-training fine-tuning, the proposed method consistently improves planning performance while balancing trajectory-space feasibility, validating the effectiveness of the proposed design.

### 5.3 Ablation and Feasibility Analysis

We further analyze the contribution of each module in the proposed method from the perspectives of benchmark performance and trajectory-space feasibility. Table 4 reports the IL-stage ablation results, while Table 5 compares different post-training strategies. Figs. 3 and 4 further visualize the effect of adaptive curvature-regularized training and drivable-area guidance on representative feasibility indicators.

Table 4: IL-stage ablation study of the proposed method on benchmark performance and trajectory-space feasibility.

| Method | x0-pred | Curvature Regularization | Sampling Guidance | PDMS $\uparrow$ | EP $\uparrow$ | DAC $\uparrow$ | Drivable Area Viol. Rate $\downarrow$ | Curvature Viol. Rate $\downarrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | ✗ | ✗ | ✗ | 85.32 | 79.56 | 93.84 | 6.16% | 11.36% |
| Trajectory-centric | ✓ | ✗ | ✗ | 86.56 | 81.26 | 94.58 | 5.42% | 7.51% |
| \+ Training-time feasibility enhancement | ✓ | ✓ | ✗ | 86.57 | 81.32 | 94.94 | 5.06% | 0.13% |
| FeaXDrive-IL | ✓ | ✓ | ✓ | 88.75 | 83.34 | 97.46 | 2.54% | 0.88% |

![[x1 32.png|Refer to caption]]

Figure 3: Comparison of curvature violation counts under different IL-stage ablation settings.

![[x2 30.png|Refer to caption]]

Figure 4: Comparison of drivable-area violation counts under different IL-stage ablation settings.

Effect of the trajectory-centric diffusion planning. Comparing the noise-centric diffusion planning baseline with the trajectory-centric diffusion planning method in Table 4, replacing the noise-centric formulation with the trajectory-centric one consistently improves both planning performance and trajectory-space feasibility. Specifically, PDMS increases from $85.32$ to $86.56$, ego progress rises from $79.56$ to $81.26$, and DAC improves from $93.84$ to $94.58$. Meanwhile, the curvature violation rate decreases from $11.36\%$ to $7.51\%$, and the drivable-area violation rate drops from $6.16\%$ to $5.42\%$. These results indicate that trajectory-centric diffusion planning not only improves trajectory planning performance, but also provides a suitable and unified modeling foundation for subsequent feasibility-aware training and inference in trajectory space.

Effect of adaptive curvature-regularized training. Adding adaptive curvature-regularized training on top of the trajectory-centric formulation produces the most significant gain in curvature-related feasibility. As shown in Table 4, the curvature violation rate drops from $7.51\%$ to $0.13\%$, while PDMS, EP, DAC, and drivable-area violation rate change only marginally. This indicates that the proposed training strategy mainly improves the intrinsic geometric and kinematic feasibility of generated trajectories, without sacrificing benchmark performance.

This trend is further confirmed by Fig. 3, where the number of curvature-violation instances is substantially reduced in both speed ranges. We report the statistics in the 0-2 m/s and $\geq 2$ m/s ranges because curvature violations may exhibit different characteristics across motion regimes. In lower-speed cases, curvature anomalies are more likely to be influenced by local geometric irregularities and discretization effects, whereas with increasing speed, curvature violations tend to become more closely associated with trajectory-level kinematic feasibility. In the 0-2 m/s range, the violation count decreases from 826 in the noise-centric baseline to 677 in the trajectory-centric version, and further to only 8 after adaptive curvature-regularized training. In the $\geq 2$ m/s range, the count drops from 554 to 235, and then further to 8. These results further show that the proposed training improves the intrinsic geometric and kinematic feasibility of trajectories.

Effect of drivable-area guided sampling. Compared with +adaptive curvature-regularized training, PDMS increases from $86.57$ to $88.75$, DAC improves from $94.94$ to $97.46$, and the drivable-area violation rate decreases from $5.06\%$ to $2.54\%$. Meanwhile, the curvature violation rate increases moderately from $0.13\%$ to $0.88\%$. This result is consistent with the intended role of the guidance module: it mainly improves the drivable-area aspect of trajectory-space feasibility by steering the generated trajectory back toward the drivable area during sampling. The same trend is reflected in Fig. 4, where the number of drivable-area violations decreases from $748$ in the baseline to $658$ in the trajectory-centric model, and further to $317$ after introducing feasibility guidance.

Post-training with feasibility-aware GRPO. Table 5 compares standard GRPO and the proposed feasibility-aware GRPO. Starting from FeaXDrive-IL, standard GRPO improves PDMS from $88.75$ to $90.56$, but also increases the curvature violation rate from $0.88\%$ to $5.79\%$. In contrast, feasibility-aware GRPO achieves a PDMS of $90.00$ while keeping the curvature violation rate at a much lower level of $2.40\%$. At the same time, it attains slightly higher DAC ($98.31$ vs. $98.28$) and a slightly lower drivable-area violation rate ($1.69\%$ vs. $1.72\%$) than standard GRPO. These results indicate that standard benchmark-oriented GRPO tends to improve score at the expense of feasibility, whereas the proposed feasibility-aware GRPO achieves a better balance between benchmark performance and trajectory-space feasibility, with only a limited reduction in PDMS.

Table 5: Comparison of post-training fine-tuning strategies for FeaXDrive-IL. FA-GRPO denotes feasibility-aware GRPO.

| Method | PDMS $\uparrow$ | EP $\uparrow$ | DAC $\uparrow$ | Drivable Area Viol. Rate $\downarrow$ | Curvature Viol. Rate $\downarrow$ |
| --- | --- | --- | --- | --- | --- |
| FeaXDrive-IL | 88.75 | 83.34 | 97.46 | 2.54% | 0.88% |
| FeaXDrive w/GRPO | 90.56 | 85.13 | 98.28 | 1.72% | 5.79% |
| FeaXDrive w/FA-GRPO | 90.00 | 84.20 | 98.31 | 1.69% | 2.40% |

Overall analysis. Taken together, the ablation results reveal a clear division of roles among the proposed modules. The trajectory-centric reformulation provides a unified foundation for feasibility-aware training, inference-time geometric guidance, and post-training feasibility-aware optimization; adaptive curvature-regularized training mainly improves the intrinsic geometric and kinematic feasibility of trajectories; drivable-area guidance primarily enhances the drivable-area aspect of trajectory-space feasibility during sampling; and feasibility-aware GRPO further improves planning performance while balancing trajectory-space feasibility during post-training. Overall, the step-by-step ablation results support the design of FeaXDrive and show that the proposed method enhances trajectory-space feasibility while maintaining strong closed-loop planning performance.

### 5.4 Inference Efficiency

We further analyze the inference efficiency of FeaXDrive. Fig. 5 presents the latency breakdown of the proposed planner. Overall, the total online inference latency is $348.73$ ms. Among all components, the VLM backbone dominates the computational cost, with a latency of $245.33$ ms ($70.3\%$ of the total), followed by the planner module with $82.96$ ms ($23.8\%$). In contrast, the additional overhead introduced by the proposed feasibility-aware components remains small: local SDF construction costs $16.03$ ms ($4.6\%$), while drivable-area guidance adds only $4.41$ ms ($1.3\%$). These results indicate that the primary computational bottleneck still lies in the visual-language backbone, while the proposed feasibility-aware guidance introduces limited extra cost.

This breakdown also helps explain the practicality of the proposed method. Although FeaXDrive introduces trajectory-space feasibility modeling, its online geometric guidance remains lightweight compared with the backbone and planner. In particular, the overhead of feasibility guidance is $4.41$ ms, which is small compared with the overall inference budget. This suggests that the proposed feasibility-aware design improves trajectory-space feasibility with modest additional online cost.

![[feaxdrive_latency_breakdown.png|Refer to caption]]

Figure 5: Latency breakdown of FeaXDrive inference.

### 5.5 Qualitative Results

We further provide qualitative comparisons between the noise-centric baseline and FeaXDrive on representative planning cases. As shown in Fig. 6, the selected examples illustrate three typical issues closely related to trajectory-space feasibility, including trajectory-level kinematic infeasibility, local geometric irregularities, and drivable-area violation.

![[case1_baseline.png|Refer to caption]]

(a) Baseline

In the first group of examples, the baseline produces trajectories with sharp turning patterns, leading to curvature infeasibility from a trajectory-level kinematic perspective. By comparison, FeaXDrive generates trajectories with smoother turning profiles and more feasible curvature evolution, showing that the proposed trajectory-centric feasibility modeling effectively improves kinematic plausibility.

In the second group, the baseline trajectories exhibit obvious local geometric irregularities. In contrast, FeaXDrive produces smoother trajectories with more coherent local geometry. These improvements are consistent with the effect of the proposed adaptive curvature-regularized training, which suppresses local geometric spikes and improves the intrinsic geometric and kinematic feasibility of trajectories.

In the third group, the baseline trajectories deviate from the drivable area. By contrast, FeaXDrive keeps the predicted trajectory much closer to the drivable area and avoids obvious drivable-area violations. This result qualitatively verifies the effect of drivable-area guidance, which progressively corrects the clean trajectory during reverse diffusion sampling.

Overall, these qualitative examples are consistent with the quantitative results reported in the previous sections. Compared with the baseline, FeaXDrive generates trajectories with stronger trajectory-space feasibility, exhibiting better geometric regularity, improved kinematic feasibility, and better consistency with the drivable area.

## 6 Discussion

The results of this work suggest that different aspects of trajectory-space feasibility may interact in a non-trivial manner. Adaptive curvature-regularized training suppresses curvature violations, whereas drivable-area guidance further improves benchmark performance and drivable-area compliance but may moderately relax the curvature optimum achieved by training. Similarly, benchmark-oriented GRPO improves score more aggressively but sacrifices trajectory-space feasibility, while feasibility-aware GRPO preserves a better balance between planning performance and trajectory-space feasibility.

This work still has several limitations. First, the feasibility-aware GRPO stage mainly incorporates the intrinsic geometric and kinematic aspect of trajectory-space feasibility, and does not yet unify all aspects of trajectory-space feasibility within the reward. Second, drivable-area guidance currently relies on local map-derived geometric priors, although such priors could in principle also be provided by lightweight maps, online mapping, or other compact scene representations. Future work may explore more unified feasibility modeling across training, inference, and post-training, as well as lighter-weight scene priors and more efficient VLM reasoning for end-to-end autonomous driving planning.

## 7 Conclusion

This paper presented FeaXDrive, a feasibility-aware trajectory-centric diffusion planning method for end-to-end autonomous driving. Built on a trajectory-centric formulation that provides a unified foundation for feasibility-aware modeling, FeaXDrive integrates adaptive curvature-regularized training, drivable-area guidance during reverse diffusion sampling, and feasibility-aware GRPO post-training.

Experiments on NAVSIM showed that FeaXDrive achieves strong closed-loop planning performance while improving trajectory-space feasibility. The results demonstrated that the trajectory-centric formulation provides a more suitable and unified foundation for feasibility-aware modeling, while adaptive curvature-regularized training improves intrinsic geometric and kinematic feasibility, drivable-area guidance enhances consistency with the drivable area, and feasibility-aware GRPO further improves planning performance while balancing trajectory-space feasibility during post-training. Overall, this work highlights the importance of explicitly modeling trajectory-space feasibility in end-to-end diffusion planning and provides a step toward more reliable and physically grounded autonomous driving planners.

[^1]: Self-driving cars: a survey. Expert systems with applications 165, pp. 113816. Cited by: §2.1.

[^2]: Injecting knowledge in data-driven vehicle trajectory predictors. Transportation research part C: emerging technologies 128, pp. 103010. Cited by: §2.3, §2.3.

[^3]: Vadv2: end-to-end vectorized autonomous driving via probabilistic planning. arXiv preprint arXiv:2402.13243. Cited by: Table 2.

[^4]: Drivinggpt: unifying driving world modeling and planning with multi-modal autoregressive transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 26890–26900. Cited by: Table 2.

[^5]: Internvl: scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 24185–24198. Cited by: §5.1.2.

[^6]: Visuomotor policy learning via action diffusion. Google Patents. Note: US Patent App. 18/594,842 Cited by: §2.2.

[^7]: Projected generative diffusion models for constraint satisfaction. arXivorg. Cited by: §2.2.

[^8]: 2026 chrysler pacifica limited specifications. Note: [https://www.chrysler.com/pacifica/specs.limited.html](https://www.chrysler.com/pacifica/specs.limited.html) Official vehicle specification page. Accessed: 2026-04-08 Cited by: §5.1.2.

[^9]: Ellipse loss for scene-compliant motion prediction. In 2021 IEEE International Conference on Robotics and Automation (ICRA), pp. 8558–8564. Cited by: §2.3.

[^10]: Navsim: data-driven non-reactive autonomous vehicle simulation and benchmarking. Advances in Neural Information Processing Systems 37, pp. 28706–28719. Cited by: §1, §5.1.1.

[^11]: Obstacle detection for intelligent robots based on the fusion of 2d lidar and depth camera. International Journal of Hydromechatronics 7 (1), pp. 67–88. Cited by: §2.1.

[^12]: Orion: a holistic end-to-end autonomous driving framework by vision-language instructed action generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 24823–24834. Cited by: §2.1.

[^13]: Trajectory prediction in autonomous driving with a lane heading auxiliary loss. IEEE Robotics and Automation Letters 6 (3), pp. 4907–4914. Cited by: §2.3.

[^14]: DeepSeek-r1 incentivizes reasoning in llms through reinforcement learning. Nature 645 (8081), pp. 633–638. Cited by: §4.4.

[^15]: Denoising diffusion probabilistic models. Advances in neural information processing systems 33, pp. 6840–6851. Cited by: §1, §2.2.

[^16]: Planning-oriented autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 17853–17862. Cited by: §1, §2.1, Table 2.

[^17]: Gen-drive: enhancing diffusion generative driving policies with reward modeling and reinforcement learning fine-tuning. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pp. 3445–3451. Cited by: §2.3, §2.3.

[^18]: EMMA: end-to-end multimodal model for autonomous driving. Transactions on Machine Learning Research. Cited by: §2.1.

[^19]: Senna: bridging large vision-language models and end-to-end autonomous driving. arXiv preprint arXiv:2410.22313. Cited by: §2.1.

[^20]: Vad: vectorized scene representation for efficient autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 8340–8350. Cited by: §1, §2.1.

[^21]: Alphadrive: unleashing the power of vlms in autonomous driving via reinforcement learning and reasoning. arXiv preprint arXiv:2503.07608. Cited by: §2.3.

[^22]: Hybrid action-based reinforcement learning for multiobjective compatible autonomous driving. IEEE Transactions on Neural Networks and Learning Systems (), pp. 1–14. External Links: [Document](https://dx.doi.org/10.1109/TNNLS.2026.3674573) Cited by: §4.4.

[^23]: Towards learning-based planning: the nuplan benchmark for real-world autonomous driving. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pp. 629–636. Cited by: §5.1.1.

[^24]: Elucidating the design space of diffusion-based generative models. Advances in neural information processing systems 35, pp. 26565–26577. Cited by: §1.

[^25]: Real-time motion planning methods for autonomous on-road driving: state-of-the-art and future research directions. Transportation Research Part C: Emerging Technologies 60, pp. 416–442. Cited by: §1, §1.

[^26]: Seamless overtaking maneuvers for automated driving: integrated motion planning based on hybrid model predictive control. IEEE Transactions on Industrial Electronics. Cited by: §2.1.

[^27]: Finetuning generative trajectory model with reinforcement learning from human feedback. arXiv e-prints, pp. arXiv–2503. Cited by: §2.3.

[^28]: Back to basics: let denoising generative models denoise. arXiv preprint arXiv:2511.13720. Cited by: §1.

[^29]: End-to-end driving with online trajectory evaluation via bev world model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 27137–27146. Cited by: Table 2.

[^30]: Recogdrive: a reinforced cognitive framework for end-to-end autonomous driving. arXiv preprint arXiv:2506.08052. Cited by: Table 1, §2.3, §5.1.2, §5.2, Table 2, Table 2, Table 3, Table 3.

[^31]: Hydra-mdp: end-to-end multimodal planning with multi-target hydra-distillation. arXiv preprint arXiv:2406.06978. Cited by: §2.1, Table 2.

[^32]: Safety-enhanced deep reinforcement learning for autonomous driving: dare to make mistakes to learn better and faster. IEEE Transactions on Intelligent Transportation Systems (), pp. 1–13. External Links: [Document](https://dx.doi.org/10.1109/TITS.2026.3670584) Cited by: §1.

[^33]: Diffusiondrive: truncated diffusion model for end-to-end autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 12037–12047. Cited by: Table 1, §1, §2.2, §5.2, Table 2, Table 3.

[^34]: Gpt-driver: learning to drive with gpt. arXiv preprint arXiv:2310.01415. Cited by: §2.1.

[^35]: A language agent for autonomous driving. In First Conference on Language Modeling, Cited by: §2.1.

[^36]: OpenScene: autonomous grand challenge toolkits. Note: [https://github.com/OpenDriveLab/OpenScene](https://github.com/OpenDriveLab/OpenScene) Accessed: 2026-04-07 Cited by: §5.1.1.

[^37]: Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4195–4205. Cited by: §5.1.2.

[^38]: Multi-modal fusion transformer for end-to-end autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 7077–7087. Cited by: §2.1, Table 2, Table 2.

[^39]: Trajectory planning and tracking control in autonomous driving system: leveraging machine learning and advanced control algorithms. Engineering Science and Technology, an International Journal 64, pp. 101950. Cited by: §1, §1.

[^40]: Scene compliant trajectory forecast with agent-centric spatio-temporal grids. IEEE Robotics and Automation Letters 5 (2), pp. 2816–2823. Cited by: §2.3.

[^41]: Lmdrive: closed-loop end-to-end driving with large language models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 15120–15130. Cited by: §1, §2.1.

[^42]: Deepseekmath: pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300. Cited by: §4.4.

[^43]: Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502. Cited by: §1, §2.2.

[^44]: A review of the motion planning and control methods for automated vehicles. Sensors 23 (13), pp. 6140. Cited by: §1, §1.

[^45]: Diffusion-driven hybrid unknown input observer for vehicle dynamics estimation. IEEE Transactions on Industrial Electronics 73 (4), pp. 6097–6110. External Links: [Document](https://dx.doi.org/10.1109/TIE.2025.3626623) Cited by: §2.2.

[^46]: DriveVLM: the convergence of autonomous driving and large vision-language models. In Conference on Robot Learning, pp. 4698–4726. Cited by: §1, §2.1.

[^47]: Safetynet: safe planning for real-world self-driving vehicles using machine-learned policies. In 2022 International Conference on Robotics and Automation (ICRA), pp. 897–904. Cited by: §2.3, §2.3.

[^48]: Omnidrive: a holistic llm-agent framework for autonomous driving with 3d perception, reasoning and planning. arXiv preprint arXiv:2405.01533 1 (2), pp. 3. Cited by: §2.1.

[^49]: Trajectory optimization with dynamic drivable corridor-based collision avoidance. Applied Sciences 15 (13), pp. 7051. Cited by: §2.3, §2.3.

[^50]: Para-drive: parallelized architecture for real-time autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15449–15458. Cited by: Table 2.

[^51]: Goalflow: goal-driven flow matching for multimodal trajectories generation in end-to-end autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 1602–1611. Cited by: §2.2.

[^52]: Drivegpt4: interpretable end-to-end autonomous driving via large language model. IEEE Robotics and Automation Letters 9 (10), pp. 8186–8193. Cited by: §2.1.

[^53]: M 2 diffuser: diffusion-based trajectory optimization for mobile manipulation in 3d scenes. IEEE Transactions on Pattern Analysis and Machine Intelligence. Cited by: §2.2.

[^54]: Diffusion-es: gradient-free planning with diffusion for autonomous and instruction-guided driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 15342–15353. Cited by: §2.3, §2.3.

[^55]: Drama: an efficient end-to-end motion planner for autonomous driving with mamba. arXiv preprint arXiv:2408.03601. Cited by: Table 2.

[^56]: SafeVLA: towards safety alignment of vision-language-action model via constrained learning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, Cited by: §2.3, §2.3.

[^57]: SafeAuto: knowledge-enhanced safe autonomous driving with multimodal foundation models. In International Conference on Machine Learning, pp. 76497–76517. Cited by: §2.3.

[^58]: DiffE2E: rethinking end-to-end driving with a hybrid diffusion-regression-classification policy. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, Cited by: §2.2.

[^59]: Diffusion-based planning for autonomous driving with flexible guidance. In The Thirteenth International Conference on Learning Representations, Cited by: §1, §2.2, §2.3, §2.3.

[^60]: Guided conditional diffusion for controllable traffic simulation. In 2023 IEEE international conference on robotics and automation (ICRA), pp. 3560–3566. Cited by: §2.2.

[^61]: AutoVLA: a vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, Cited by: §1, §2.1.

[^62]: DiffusionDriveV2: reinforcement learning-constrained truncated diffusion modeling in end-to-end autonomous driving. arXiv preprint arXiv:2512.07745. Cited by: §2.3.
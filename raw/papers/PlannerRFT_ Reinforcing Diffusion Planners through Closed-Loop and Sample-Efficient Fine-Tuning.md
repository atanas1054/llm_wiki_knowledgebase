---
title: "PlannerRFT: Reinforcing Diffusion Planners through Closed-Loop and Sample-Efficient Fine-Tuning"
source: "https://arxiv.org/html/2601.12901v1"
author:
published:
created: 2026-06-23
description:
tags:
  - "clippings"
---
Hongchen Li <sup>1,2,3</sup>  Tianyu Li <sup>2,3</sup>  Jiazhi Yang <sup>3</sup>  Haochen Tian <sup>3</sup>    
Caojun Wang <sup>1,2,3</sup>  Lei Shi <sup>4</sup>  Mingyang Shang <sup>5</sup>  Zengrong Lin <sup>5</sup>    
Gaoqiang Wu <sup>5</sup>  Zhihui Hao <sup>5</sup>  Xianpeng Lang <sup>5</sup>  Jia Hu ${}^{1~\textrm{\Letter}}$  Hongyang Li ${}^{3~\textrm{\Letter}}$  
<sup>1</sup> Tongji University   <sup>2</sup> Shanghai Innovation Institute  
<sup>3</sup> OpenDriveLab at The University of Hong Kong   <sup>4</sup> Meituan   <sup>5</sup> Li Auto Inc.    
  
[https://opendrivelab.com/PlannerRFT](https://opendrivelab.com/PlannerRFT)

###### Abstract

Diffusion-based planners have emerged as a promising approach for human-like trajectory generation in autonomous driving. Recent works incorporate reinforcement fine-tuning to enhance the robustness of diffusion planners through reward-oriented optimization in a generation–evaluation loop. However, they struggle to generate multi-modal, scenario-adaptive trajectories, hindering the exploitation efficiency of informative rewards during fine-tuning. To resolve this, we propose PlannerRFT, a sample-efficient reinforcement fine-tuning framework for diffusion-based planners. PlannerRFT adopts a dual-branch optimization that simultaneously refines the trajectory distribution and adaptively guides the denoising process toward more promising exploration, without altering the original inference pipeline. To support parallel learning at scale, we develop nuMax, an optimized simulator that achieves 10 times faster rollout compared to native nuPlan. Extensive experiments shows that PlannerRFT yields state-of-the-art performance with distinct behaviors emerging during the learning process.

<sup>†</sup>

## 1 Introduction

Diffusion-based planners have recently emerged as a powerful probabilistic paradigm for generating human-like and socially compatible driving trajectories in dynamic environments [^5] [^56] [^32]. Such planners acquire driving skills from large-scale human demonstrations via imitation learning (IL). Despite the capability of modeling dexterous behaviors, these methods suffer from distributional shift and objective misalignment, limiting their robustness and reliability in real-world deployment [^39] [^10] [^41] [^33].

![[x1 35.png|Refer to caption]]

Figure 1: Comparison to Denoising Strategies across various diffusion planning paradigms. (a) Vanilla diffusion planners suffer from mode collapse, offering limited exploration. (b) Anchor-based methods are oriented towards scenario-agnostic actions, leading to noisy interactions. (c) Our policy-guided denoising enables both multi-modal and scenario-adaptive sampling, yielding stable and efficient exploration for optimization.

Reinforcement learning (RL) offers a potential alternative. Through simulator-assisted exploration and reward-oriented optimization, RL-based planners can scale with large-scale simulated data and simple rewards [^17] [^55] [^7]. Recent generation-evaluation reinforcement fine-tuning (RFT) paradigms [^23] [^16] [^52] [^25] demonstrate the balance between training efficiency and improvements in closed-loop planning performance. In this paradigm, a trajectory generator serves as an actor to produce diverse candidate trajectories, which are then evaluated in simulation and iteratively refined through group-wise reinforcement fine-tuning [^44]. The overall performance of this paradigm primarily depends on the generator’s exploration capability, that is, the distribution of candidate trajectories. This drives two key requirements: (i) Multi-modality, the ability to generate diverse maneuver hypotheses under the same situation; and (ii) Adaptivity, the capacity to self-adjust exploration distribution toward more promising behaviors, such as AlphaGo [^45] used Monte Carlo Tree Search (MCTS) [^6] for adaptive exploration.

However, vanilla diffusion-based planners suffer from modality collapse [^32], where trajectories generated from different noise inputs converge to nearly identical results throughout denoising process <sup>2</sup>, as illustrated in Fig. 1 (a). This collapse limits the exploration capability, leaving reinforcement fine-tuning without an effective optimization signal. To mitigate this issue, anchor-based diffusion planners [^32] [^51] initialize the denoising process from anchor-centered Gaussian distributions rather than pure Gaussian noise, enabling the generation of diverse and maneuver-consistent trajectories. Nevertheless, these fixed, scenario-agnostic anchors are suboptimal for reward-oriented optimization. As shown in Fig. 1(b), a part of anchors yield scene-compatible maneuvers, while many others produce context-conflicting motions, which introduce noisy gradients and hinder stable reinforcement optimization. Overall, exploration effectiveness requires not only diverse but also scene-consistent maneuvers, which, in turn, facilitate efficient reinforcement fine-tuning.

To this end, we propose PlannerRFT, a closed-loop and sample-efficient framework for diffusion-based Planner Reinforcement Fine-tuning. As shown in Fig. 1 (c), PlannerRFT performs policy-guided denoising to achieve multi-modality and scenario-adaptive trajectory sampling, providing group-wise trajectory optimization with more stable and efficient exploration. For scalable closed-loop training, we develop a GPU-accelerated simulator, nuMax, which supports high-throughput parallel rollouts.

For multi-modality, PlannerRFT introduces an energy-based classifier guidance [^37] that injects residual offsets into the denoising process, enabling the model to generate diverse maneuver trajectories. For adaptivity, a dedicated Exploration Policy learns an adaptive guidance scale to modulate exploration according to scenario context, achieving scenario-aware trajectory generation. The Exploration Policy is optimized through closed-loop interaction with the simulator using Proximal Policy Optimization (PPO) [^43], guiding the planner toward temporally consistent, safe, and efficient behaviors during reinforcement fine-tuning.

For trajectory optimization, we adopt Group Relative Policy Optimization (GRPO) [^44] to fine-tune the diffusion planner denoising process. To stabilize optimization in challenging scenarios, we introduce a survival reward formulation that accumulates non-terminal trajectory rewards, encouraging the planner to delay failure and improve long-horizon viability. To enhance the scalability and efficiency of online rollouts, we develop nuMax, a GPU-parallel simulator built upon Waymax [^12] and calibrated for the large-scale nuPlan benchmark [^24], achieving up to 10× faster simulation speed than the native nuPlan simulator.

Extensive evaluations on the nuPlan benchmark demonstrate that PlannerRFT achieves state-of-the-art performance. Compared with the IL-pretrained baseline, PlannerRFT demonstrates notable gains in handling failure scenarios such as collisions and off-road events, leading to improved driving safety. Furthermore, PlannerRFT exhibits distinct, human-like driving behaviors, with safer and more efficient maneuvers, thereby highlighting the effectiveness of our reinforcement fine-tuning framework. We summarize our contributions as follows:

- We present PlannerRFT, a closed-loop reinforcement fine-tuning framework for diffusion-based planners that enhances the RL sampling efficiency through policy-guided denoising.
- We design an exploration policy that adaptively modulates trajectory sampling across scenarios and cooperates with group-wise reinforcement optimization for stable fine-tuning. To support large-scale online training, we further develop nuMax, a GPU-parallel simulator calibrated for the nuPlan benchmark.
- Extensive experiments on nuPlan demonstrate that PlannerRFT achieves state-of-the-art performance, while notably enhancing safety and robustness in challenging driving scenarios.

## 2 Related Work

Diffusion Planners for Autonomous Driving. Recently, diffusion models have been widely applied to decision-making and planning tasks in autonomous driving, including motion planning [^56] [^47] [^50], traffic simulation [^57] [^20] [^22], and end-to-end driving policy learning [^29] [^19] [^11] [^36] [^32]. Representative works include Diffusion Planner [^56], which jointly models surrounding agents’ trajectories and ego-vehicle planning; Nexus [^57], which introduces flexible noise scheduling to balance reactivity and goal orientation for the traffic scenario simulation; DiffusionDrive [^32], which generates multimodal trajectories via a truncated diffusion process for end-to-end driving; and RecogDrive [^29], which incorporates vision-language tokens followed by a diffusion head for trajectory generation. Beyond modeling complex distributions, diffusion planners offer strong flexibility through guidance-based denoising [^56] [^47] [^50] that enables controllable trajectory generation. However, rule-based guidance strategies introduce competing gradient signals (e.g., between collision avoidance and ride comfort) and impose a fixed guidance strength [^47]. These limitations lead to substantial performance variability across diverse driving scenarios.

![[x2 34.png|Refer to caption]]

Figure 2: Overview of PlannerRFT. We enhance multi-modality during RL sampling through Guided Denoising, with guidance scales modulated by the Exploration Policy to generate scenario-adaptive trajectories ( Sec. 4.2 ). The planner gathers on-policy interaction data through Closed-Loop Rollout in simulation ( 4.3 ). A dual-branch optimization framework performs Trajectory Optimization and Exploration Optimization to steer the denoising process ( 4.4 ).

Reinforcement Fine-tuning for Driving Planners. Reinforcement learning leverages probabilistic modeling to enable sampling-based exploration and policy optimization. Recent studies on reinforcement fine-tuning for driving planners generally follow three paradigms. The first discretizes trajectories into a vocabulary of motion tokens [^26] [^30] [^31], optimizing the selection probability of each token under different driving scenarios [^35] [^58] [^13], similar to RFT in LLMs. Yet, such discretization inherently constrains planner expressiveness, as a larger token set better captures trajectory diversity but also increases computational complexity and optimization dimensionality. Another paradigm models each trajectory step as a continuous distribution via auto-regressive generation [^54] [^48] [^27], which inherently suffers from error accumulation and temporal instability across sequential decisions. Diffusion models exhibit an inherent advantage, as their denoising process operates in a probabilistic manner to generate actions, making them well-suited for reinforcement learning in continuous action spaces and temporally consistent decision processes. However, diffusion-based planners in autonomous driving tend to exhibit modality collapse, which restricts exploration during RFT and consequently hinders effective policy adaptation.

## 3 Preliminary

Task Definition. Motion planning aims to generate safe and feasible trajectories for the ego vehicle in dynamic driving environments [^34] [^53] [^18]. This work focuses on enhancing the closed-loop performance of IL-pretrained diffusion planners via reinforcement fine-tuning, yielding improved safety, comfort, and efficiency in motion planning.

Planner Architecture. We adopt a pretrained diffusion planner following the commonly used architecture, consisting of a shared scene encoder and a Diffusion Transformer (DiT) [^40] decoder. The scene encoder fuses scene inputs, including surrounding agents, map features, and static obstacles, into the environment representation $F_{\text{scene}}$. The navigation command is encoded as $F_{\text{navi}}$. Given the noisy trajectory samples $\mathbf{x}^{k}$ and diffusion timestep $k$, the DiT decoder iteratively denoises the latent samples, conditioned on both the scene and navigation embeddings, mathematically:

$$
\hat{\mathbf{x}}_{0}^{k}=\text{DiT}_{\theta}\left(\text{MLP}(\mathbf{x}^{k});F_{\text{scene}};F_{\text{navi}};t\right).
$$

## 4 Method

In this section, we begin with an overview of our PlannerRFT in Section 4.1. We then delve into policy-guided denoising in Section 4.2, followed by the closed-loop rollout process in Section 4.3 and the policy optimization in Section 4.4. Finally, we summarize best practices for PlannerRFT in Section 4.5.

### 4.1 Overview of PlannerRFT

As illustrated in Fig. 2, given an IL-pretrained diffusion planner, PlannerRFT aims to enhance its closed-loop planning performance by adopting the generation–evaluation paradigm with GRPO. During RFT, the IL-pretrained planner is duplicated and frozen as a global reference. We introduce policy-guided denoising to improve the multi-modality and adaptivity of the trajectory sampling. To achieve this, we plug in an Exploration Policy on the original model architecture and use closed-loop rollout and PPO to optimize the policy.

### 4.2 Policy-guided Denoising

Guided Denoising. Vanilla diffusion planners generate single-pass trajectories and tend to modality collapse, leading to limited trajectory diversity for RL sampling. To alleviate this limitation and promote exploration, we adopt the energy-based classifier guidance [^9] [^37] that injects residual offsets into the denoising process. This enables the planner to generate diverse trajectories in the vicinity of the reference trajectory. Specifically, we decompose the injected guidance into lateral and longitudinal components. At each timestep $\tau$, given the current planner’s predicted waypoints $\mathbf{x}$ and the reference waypoints $\mathbf{x}^{\text{ref}}$, the lateral guidance energy function $\Psi_{\text{lat.}}$ is formulated as:

$$
\Psi_{\text{lat.}}=\frac{1}{T}\sum_{\tau=1}^{T}\left(\mathbf{n}_{\tau}^{\perp}\left(\mathbf{x}_{\tau}-\mathbf{x}_{\tau}^{\text{ref}}\right)-\lambda_{\text{lat.}}\eta_{\text{lat.}}\right)^{2},\eta_{\text{lat.}}\in[-1,1],
$$

where $\mathbf{n}_{\tau}^{\perp}$ is the unit normal vector, $\lambda_{\text{lat.}}$ is the maximum lateral offset (meters), and $\eta_{\text{lat.}}$ is the lateral guidance scale. The longitudinal guidance modulates the deviation of the planned velocity $\mathbf{v}$ with the reference velocity $\mathbf{v}^{\text{ref}}$, as:

$$
\Psi_{\text{lon.}}=\frac{1}{T}\sum_{\tau=1}^{T}\left(\mathbf{n}_{\tau}^{\parallel}\!\big(\mathbf{v}_{\tau}-\lambda_{\text{lon.}}\eta_{\text{lon.}}\mathbf{v}_{\tau}^{\text{ref}}\big)\right)^{2},\eta_{\text{lon.}}\in[-1,1],
$$

where $\mathbf{n}_{\tau}^{\parallel}$ is the unit tangent vector, $\lambda_{\text{lon.}}$ is a constant maximum relative speed deviation (percentage), and $\eta_{\text{lon.}}$ is the longitudinal guidance scale. These two energy functions yield decoupled and orthogonal gradients, enabling multi-modal trajectory generation through different combinations of $(\eta_{\text{lat.}},\eta_{\text{lon.}})$. No explicit map- or vehicle-level collision constraints are imposed; this simplified guidance formulation allows infeasible samples to act as negative feedback for RL optimization.

Design of Exploration Policy. We introduce the Exploration Policy module, which learns to modulate the guidance scales $(\eta_{\text{lat.}},\eta_{\text{lon.}})$ conditioned on driving contexts $\mathbf{s}$ and reference waypoints. This learnable exploration enables the planner to generate context-aware maneuvers, thereby improving exploration effectiveness during RL sampling. Formulated as:

$$
\bm{\eta}\sim\pi_{\phi}(\cdot\mid\mathbf{s},\mathbf{x}^{\text{ref}}).
$$

Concretely, we use the reference trajectory as a frozen prior to provide PlannerRFT with a stable and well-trained imitation-learning distribution. The reference trajectory is encoded through an MLP-Mixer into a compact token and fused with the scene embedding via a cross-attention module, capturing the interaction between the reference motion and the surrounding environment. Based on this fused representation, the Guidance Head predicts the parameters of two Beta distributions governing the lateral and longitudinal guidance scales. In parallel, the Value Head $V_{\psi}$ estimates the state-value $V(s_{t})$ to assist policy optimization.

![[x3 31.png|Refer to caption]]

Figure 3: Illustration of nuMax. (a) Scenario cache: nuPlan scenarios are preprocessed and cached for fast loading during large-scale RL rollouts; (b) LQR tracker and scorer: vehicle kinematics and reward computation are calibrated to match nuPlan; and (c) Distributed RL training pipeline: enables communication between PyTorch DistributedDataParallel (DDP) workers and the JAX-based simulator.

Trajectory Sampling. During RFT, we repeatedly sample the guidance scales $(\eta_{\text{lat.}}^{(k)},\eta_{\text{lon.}}^{(k)})$ from the Beta distributions learned by the Exploration Policy. Each sampled pair specifies a distinct driving modality and modulates the guided denoising process toward a corresponding trajectory $\hat{\mathbf{x}}^{(k)}$. Repeating this process $K$ times yields a diverse set of trajectories $\mathcal{X}=\{\hat{\mathbf{x}}^{(k)},(\eta_{\text{lat.}}^{(k)},\eta_{\text{lon.}}^{(k)})\}_{k=1}^{K}$. Formally, the Exploration Policy dynamically modulates the classifier-guided denoising gradients as:

$$
\nabla_{\mathbf{x}}\log p(\mathbf{\eta}|\mathbf{x})\approx-\nabla_{\mathbf{x}}\big[\Psi_{\text{lat.}}(\mathbf{x};\eta_{\text{lat.}})+\Psi_{\text{lon.}}(\mathbf{x};\eta_{\text{lon.}})\big],
$$

thereby enabling the Fine-tuned DiT to produce adaptive and human-like trajectories across scenarios.

### 4.3 Closed-loop Rollout

The nuMax Simulator. Unlike IL methods trained on pre-collected offline datasets, RL is trained on simulated data that is collected during the training process. Therefore, enhancing simulation throughput is essential for accelerating model iteration and achieving scalable training, given limited computational resources. To this end, we develop nuMax, a GPU-parallel simulator that enables 10 times faster rollout speed compared with the native nuPlan simulator. Our implementation builds upon Waymax [^12] and V-Max [^2], with further implementation details provided in the supplementary material.

Rollout Planning. At each simulation step, the fine-tuned planner generates a set of $K$ candidate trajectories $\mathbf{\mathcal{X}}$ under different guidance scales. To provide diverse training experiences for reinforcement learning, one trajectory $\mathbf{x}^{{}^{\prime}}$ and its corresponding guidance scales $(\eta_{\text{lat.}}^{\prime},\eta_{\text{lon.}}^{\prime})$ are randomly selected from the candidate set. Only the first action of the selected trajectory is executed in the closed-loop simulator to update the environment state from $s_{t}$ to $s_{t+1}$ and obtain the immediate reward $r_{t}$. The current state $s_{t}$, the selected guidance scales $(\eta_{\text{lat.}}^{\prime},\eta_{\text{lon.}}^{\prime})$, and the received reward $r_{t+1}$ are stored in the replay buffer $\mathcal{B}$ for subsequent policy updates $(s_{t},\eta_{\text{lat.}}^{{}^{\prime}},\eta_{\text{lon.}}^{\prime},r_{t+1},V(s_{t}))$.

### 4.4 Policy Optimization

Exploration Policy Optimization. The Exploration Policy $\pi_{\phi}$ is optimized following the PPO framework. Specifically, the goal of this optimization is to provide temporally consistent, efficient, safe, and comfortable exploration directions during closed-loop planning. Future rewards are propagated backward through Generalized Advantage Estimation (GAE), allowing the policy to refine its current exploratory decisions based on the long-term trajectory performance observed in closed-loop rollouts. Through iterative rollouts and updates, the Exploration Policy learns Beta-distribution parameters that adaptively set $(\eta_{\text{lat}},\eta_{\text{lon}})$ to the driving context, improving exploration effectiveness.

Trajectory Optimization. The Fine-tuned DiT focuses on long-horizon planning conditioned on the current scenario. We evaluate the trajectory based on the Predictive Driver Model Score (PDMS) over a prediction horizon $T_{r}$ in an open-loop manner. However, direct use of the terminal reward (collision and off-road) leads to optimization stagnation in hard scenarios, as all candidate trajectories collapse to zero reward once a failure occurs, resulting in no optimization gradient within the group. To alleviate this issue, we introduce a survival reward formulation that accumulates trajectory-level rewards only over valid, non-terminal segments. Formally, given a per-step termination reward sequence ${R^{\text{term}}}_{\tau=1}^{T_{r}}$, the survival reward is defined as:

$$
R_{\text{surv}}=\frac{1}{T_{r}}\sum_{\tau=1}^{T_{r}}R_{\tau}^{\text{term}}\prod_{j=1}^{\tau}\mathbb{I}[R_{j}^{\text{term}}\neq 0].
$$

This formulation encourages the planner to optimize toward trajectories that delay the failure event, improving exploration in hard scenarios.

We fine-tune the planner’s trajectory distribution via the GRPO framework. Following DPPO [^42] and ReCogDrive [^29], the diffusion denoising process is formulated as a Markov Decision Process, where each denoising step represents a Gaussian transition. By updating the Gaussian parameters during RFT, the planner better aligns with reward-oriented objectives, improving closed-loop stability and planning performance.

### 4.5 Best Practices for PlannerRFT

We summarize the best practices for effectively fine-tuning diffusion-based planners with PlannerRFT as follows.

Fine-tune DDIM Denoising. We adopt a 5-step DDIM [^46] denoising scheme. Compared with ODE-based denoising, DDIM introduces stochasticity that enhances exploration, while requiring far fewer steps than DDPM [^14] to maintain high training efficiency.

Zero-initialization of Exploration Policy. The Exploration Policy is initialized to produce zero-mean lateral and longitudinal guidance scales. This initialization ensures unbiased exploration around the reference trajectory and mitigates performance drops at the early stage of fine-tuning.

Plug-and-play Fine-tuning. During RFT, the Reference DiT and Exploration Policy are integrated to guide the denoising process, facilitating exploration and policy refinement. At deployment, these modules are removed, enabling the planner to retain its original diffusion structure while delivering improved trajectory performance.

Hard-case Fine-tuning. Incorporating a moderate proportion of challenging scenarios significantly improves the planner’s robustness, while an excessively hard training set may degrade overall performance. Further analysis of fine-tuning data selection is provided in the Section 5.3.

![[x4 29.png|Refer to caption]]

Figure 4: Qualitative Comparison of Pretrained Planner and RFT Planner. In each frame shot, the simulation position and planning trajectory are marked as orange, the ground-truth position ground-truth trajectory recorded in the driving log are marked as gray blue, respectively.

## 5 Experiments

This section aims to explore the following research questions: 1) Can PlannerRFT improve the closed-loop planning performance of diffusion-based planners through reinforcement fine-tuning? 2) Does the Exploration Policy enhance sample efficiency through the policy guided denoising? 3) Will the fine-tuned planner exhibit distinct behavioral patterns from imitation learning? 4) What are the key factors that influence the effectiveness of RFT training?

### 5.1 Setup and Protocals

Benchmarks and Baselines. We evaluate PlannerRFT on the large-scale nuPlan benchmark [^24]. The Val14 benchmark [^8] is used to assess model performance under general driving scenarios, while the Test14-hard benchmark [^4] includes more complex and challenging situations, reflecting the model’s robustness in hardcore scenarios. All evaluations are performed within the nuPlan closed-loop simulator, which supports both non-reactive and reactive background traffic settings. In the non-reactive setting, surrounding vehicles follow pre-recorded trajectories; in contrast, the reactive setting employs an Intelligent Driver Model (IDM) [^49] that dynamically adjusts surrounding vehicles’ behaviors according to the ego vehicle’s actions, providing a more realistic simulation of real-world interactions. We compare PlannerRFT against a wide range of baseline methods, including rule-based planners (IDM [^49], PDM-Closed [^8]), learning-based planners (PlanTF [^4], GameFormer [^15], PLUTO [^3]), and recent generative planning approaches (Diffusion Planner [^56], Flow Planner [^47]). The final evaluation score is computed as the average across all scenarios, ranging from 0 to 100, where a higher score indicates better planning performance.

Table 1: Closed-loop Planning Results on nuPlan Dataset. The highest and the second-best results of each benchmark are denoted by bold and underline.

<table><tbody><tr><td rowspan="2">Type</td><td rowspan="2">Planner</td><td colspan="2">Val14</td><td colspan="2">Test14-hard</td></tr><tr><td>NR</td><td>R</td><td>NR</td><td>R</td></tr><tr><td>Expert</td><td>Log-replay</td><td>93.53</td><td>80.32</td><td>85.96</td><td>68.80</td></tr><tr><td rowspan="2">Rule</td><td>IDM</td><td>75.60</td><td>77.33</td><td>56.15</td><td>62.26</td></tr><tr><td>PDM-Closed</td><td>92.84</td><td>92.12</td><td>65.08</td><td>75.19</td></tr><tr><td rowspan="7">Learning</td><td>PDM-Open</td><td>53.53</td><td>54.24</td><td>33.51</td><td>35.83</td></tr><tr><td>GameFormer</td><td>13.32</td><td>8.69</td><td>7.08</td><td>6.69</td></tr><tr><td>PlanTF</td><td>84.27</td><td>76.95</td><td>69.70</td><td>61.61</td></tr><tr><td>PLUTO</td><td>88.89</td><td>78.11</td><td>70.03</td><td>59.74</td></tr><tr><td>Diffusion Planner</td><td>89.87</td><td>82.80</td><td>75.99</td><td>69.22</td></tr><tr><td>Flow Planner</td><td>90.43</td><td>83.31</td><td>76.47</td><td>70.42</td></tr><tr><td>PlannerRFT(Ours)</td><td>89.96</td><td>84.46</td><td>77.16</td><td>72.21</td></tr></tbody></table>

Pretrain. We adopt the Diffusion Planner [^56] as our IL-pretrained planner, which is trained on 1 million clips from the nuPlan dataset. We replace the ODE-based DPM-solver [^38] denoising with a 5-step DDIM sampler. Compared with the ODE sampler, the DDIM sampler achieves nearly the same performance while introducing stochasticity that enhances exploration, and its reduced number of denoising steps further improves the efficiency of RL training.

Fine-tune Dataset. For reinforcement fine-tuning, we collect 144,494 non-overlapping scenarios from nuPlan at 10 Hz sampling rate. Each scenario contains 20 frames of history, one current frame, and 150 frames of future trajectory, totaling 171 frames. We evaluate all scenes using the pretrained planner and construct three datasets according to performance scores: (1) Fail, including 10,417 collision or off-road cases; (2) Lt90, including all low-score (less than 90) cases, totaling 24,691 scenes; and (3) All, which includes all available scenes.

RFT Details. All experiments are conducted on 8 NVIDIA H100 GPUs. The fine-tuning process runs for 40M environment steps. Hyperparameters for PPO and GRPO optimization are provided in the supplementary material.

Table 2: Ablation on Exploration Policy. $\text{IL Pretrain}_{\texttt{DDIM}}$ denotes the pretrained Diffusion Planner with 5 steps of DDIM denoising. All planners use the same 5-step DDIM denoising setup. $\mathcal{D}$ denotes the modality of the sampled trajectory group, consistent with the definition in DiffusionDrive [^32], $\bar{r}$ and $s_{r}$ denote the mean and standard deviation of the corresponding rewards, respectively.

| Exploration Type | R-score $\uparrow$ | Collisions | TTC | Drivable | Comfort | Progress | Speed | NR-score $\uparrow$ | $\mathcal{D}~(\%)$ | $\bar{r}\uparrow$ | $s_{r}$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $\text{IL Pretrain}_{\texttt{DDIM}}$ | 68.18 | 86.58 | 79.05 | 94.48 | 86.03 | 76.99 | 97.20 | 76.01 | \- | \- | \- |
| w/o Guidance | 68.83(+0.65) | 86.03 | 79.41 | 94.48 | 87.87 | 77.12 | 97.35 | 76.34(+0.33) | 5.65 | 69.06 | 0.02 |
| w/ Uniform Dist. | 65.82(-2.36) | 84.37 | 75.74 | 93.01 | 80.88 | 76.19 | 97.59 | 75.19(-0.82) | 39.78 | 60.44 | 0.12 |
| w/ Fixed Beta Dist. | 70.65(+2.47) | 87.68 | 80.88 | 94.85 | 84.56 | 77.34 | 97.71 | 76.61(+0.60) | 27.73 | 71.50 | 0.07 |
| PlannerRFT(Ours) | 72.21(+4.03) | 88.97(+2.39) | 84.93(+5.34) | 95.59(+1.11) | 85.66 | 77.17 | 98.03 | 77.16(+1.15) | 25.34 | 73.88 | 0.06 |

### 5.2 Main Results

Comparison with SOTAs. Tab. 1 presents the planning results under both challenging (Test14-hard) and general (Val14) test settings. Compared with the pretrained Diffusion Planner, our PlannerRFT improves closed-loop planning performance across all four benchmarks. Notably, in reactive traffic settings, PlannerRFT achieves substantial gains, with improvements of $+1.66$ points on the Val14 benchmark and $+2.99$ points on the Test14-hard benchmark. This suggests that closed-loop rollouts expose the planner to a broader range of interaction patterns, mitigating distribution shift, while the iterative feedback during rollouts enables the model to continuously refine and improve its trajectories. Compared with other SOTA planners, PlannerRFT achieves the best overall performance in three out of four benchmarks. However, in non-reactive regular scenarios (Val14-NR), the performance improvement remains marginal. This may stem from the inherent distributional bias of non-reactive environments. Notably, PlannerRFT yields a $+2.99$ points improvement on the Test14-hard-NR set, which contains dynamic, interaction-heavy scenarios, highlighting its effectiveness.

Qualitative Results. Distinct behaviors compared with IL pretraining emerge during reinforcement fine-tuning. Through reward-oriented optimization, the planner adapts its driving policy toward safer and more efficient maneuvers. Fig. 4 illustrates an out-of-distribution (OOD) scenario for the IL-pretrained planner. As shown in Fig. 4 (a), the pretrained planner attempts a lane change but fails to handle interactive conflicts, causing the ego vehicle to get stuck between two lanes and collide at $t_{\text{sim}}=12\text{s}$. After 10M fine-tuning steps, as shown in Fig. 4 (b), the planner learns to avoid the collision through lane keeping, while achieving safety, but at the cost of efficiency. With 25M steps, as shown in Fig. 4 (c), the planner executes a decisive lane-change maneuver, achieving both safety and efficiency.

![[x5 26.png|Refer to caption]]

Figure 5: Visualization of Different Exploration Policies. (a) Without guidance: denoising from random noise. (b) Uniform exploration policy: ( η lat., lon. ) (\\eta\_{\\text{lat.}},\\eta\_{\\text{lon.}}) are sampled from a uniform distribution. (c) Fixed exploration policy: are sampled from the non-learnable Beta distribution initialized from the Exploration Policy’s zero parameters. (d) Our policy-guided denoising: exploration adapt to the driving context.

### 5.3 Ablation Study

Effectiveness of Exploration Policy. We evaluate four exploration policies in RL sampling: 1) denoising from random noise without any guidance, 2) sampling the guidance scale from a uniform distribution, $\mathbf{\eta}\sim\mathcal{U}(-\mathbf{\lambda},\mathbf{\lambda})$, 3) sampling the guidance scale from a fixed Beta distribution, and 4) our policy-guided denoising.

As for multi-modality, we use the diversity score $\mathcal{D}$ from the DiffusionDrive as the evaluation metric, which is based on the mean Intersection over Union (mIoU) between each sampled trajectory and the others of the trajectory group. A higher diversity score $\mathcal{D}$ means less trajectory overlap, in terms of more exploration variety. As shown in Tab. 2, compared with denoising from random noise, our guided denoising improves trajectory diversity during RL sampling. The visualization results in Fig. 5 demonstrate that our guidance enables the planner to generate continuous and smooth trajectories around the reference trajectory, enhancing both lateral and longitudinal diversity, thereby enhancing exploration.

For adaptivity, we compute the mean and standard deviation of rewards across all trajectories sampled in each GRPO group. As shown in Tab. 2, the uniform exploration policy yields the highest diversity score but also the worst performance. This is because its scenario-agnostic sampling introduces excessively large reward variance, causing training instability and repeated episodes of reward collapse, as illustrated in Fig. 6. In contrast, the fixed exploration policy stabilizes training by restricting the exploration range, but the overly limited search space also constrains the achievable performance ceiling. Our policy-guided denoising exploration adaptively adjusts the exploration direction based on the context, achieving both stable training and higher closed-loop performance.

Table 3: Ablations of fine-tuning data distribution.

<table><tbody><tr><td rowspan="2">Training Type</td><td rowspan="2">Dataset</td><td colspan="2">Val14</td><td colspan="2">Test14-hard</td></tr><tr><td>NR</td><td>R</td><td>NR</td><td>R</td></tr><tr><td>IL Pretrain</td><td>All</td><td>89.87</td><td>82.80</td><td>75.99</td><td>69.22</td></tr><tr><td>IL Fine-tune</td><td>Lt90</td><td>88.91</td><td>82.08</td><td>74.32</td><td>67.55</td></tr><tr><td rowspan="3">RL Fine-tune</td><td>Fail</td><td>82.97</td><td>77.48</td><td>69.26</td><td>63.75</td></tr><tr><td>All</td><td>89.93</td><td>84.88</td><td>75.50</td><td>70.43</td></tr><tr><td>Lt90</td><td>89.96</td><td>84.46</td><td>77.16</td><td>72.21</td></tr></tbody></table>

Table 4: Ablations of the GRPO reward type and reward horizon.

<table><tbody><tr><td rowspan="2">Reward Type</td><td rowspan="2">Horizon (s)</td><td colspan="2">Val14</td><td colspan="2">Test14-hard</td></tr><tr><td>NR</td><td>R</td><td>NR</td><td>R</td></tr><tr><td>Terminal</td><td>4</td><td>89.78</td><td>84.27</td><td>76.81</td><td>71.59</td></tr><tr><td rowspan="3">Survival</td><td>2</td><td>89.54</td><td>84.08</td><td>76.49</td><td>70.10</td></tr><tr><td>4</td><td>89.96</td><td>84.46</td><td>77.16</td><td>72.21</td></tr><tr><td>6</td><td>89.66</td><td>84.31</td><td>76.96</td><td>71.91</td></tr></tbody></table>

Table 5: Ablations of the maximum guidance offset $\lambda$ on the Test14-hard Reactive benchmark.

<table><tbody><tr><td></td><td colspan="3"><math><semantics><msub><mi>λ</mi> <mtext>lon.</mtext></msub><annotation>\lambda_{\text{lon.}}</annotation></semantics></math> (%)</td></tr><tr><td><math><semantics><msub><mi>λ</mi> <mrow><mtext>lat.</mtext><mo></mo> <mrow><mo>(</mo><mi>m</mi><mo>)</mo></mrow></mrow></msub> <annotation>\lambda_{\text{lat.}~(m)}</annotation></semantics></math></td><td>10</td><td>25</td><td>50</td></tr><tr><td>1.0</td><td>69.94</td><td>71.41</td><td>70.26</td></tr><tr><td>2.5</td><td>70.64</td><td>72.21</td><td>71.95</td></tr><tr><td>5.0</td><td>70.11</td><td>71.63</td><td>69.99</td></tr></tbody></table>

![[x6 22.png|Refer to caption]]

Figure 6: Closed-loop performance of safety metrics during training under different exploration policy. Score denotes the nuPlan aggregate score, NC refers to No at-fault Collisions, DAC represents Drivable Area Compliance, and TTC indicates Time-to-Collision. Our adaptive exploration policy achieves consistently higher performance and stability across all metrics compared to fixed, uniform, and unguided exploration baselines.

Impact of Fine-tuning Data Distribution. We find that the composition of training scenarios substantially alters the characteristics of the learning process. As shown in Table 5, training exclusively on collision cases (Fail) causes severe performance degradation across all benchmarks, indicating that overly hard scenarios can make the planner forget how to handle regular driving maneuvers. In contrast, training on all available scenarios (All) includes a large number of easy cases, leading to a weak optimization signal and limited gains on hard scenarios. The best results are obtained when fine-tuning on a balanced dataset (Lt90) that combines collision and low-score cases. This suggests that an appropriate proportion of hard cases is essential for effective RFT. For completeness, we also include an IL fine-tuning baseline trained on the same Lt90 dataset. The IL-finetuned model performs worse, confirming that PlannerRFT’s gains arise from effectively learning under the hard training distribution through exploration, rather than from additional training iterations.

Effect of Reward Type and Horizon. Table 5 compares different reward formulations and horizons of the GRPO reward. The terminal reward performs comparably to survival on Val14 but degrades on Test14-hard, where collisions or off-route events frequently reset the reward to zero. In contrast, the survival reward encourages trajectories that delay failure, enabling continuous improvement in closed-loop settings. For the reward horizon, a short 2 $s$ horizon underperforms due to limited temporal context, while 4 $s$ and 6 $s$ horizons yield similar results, suggesting that a moderate horizon length is sufficient for fine-tuning.

Effect of the Maximum Guidance Offset $\lambda$. We grid search the maximum lateral and longitudinal guidance offset $\lambda$, as shown in Tab. 5. A small $\lambda$ limits exploration and constrains policy optimization, while a large $\lambda$ drives the policy too far away from the human expert behavior distribution. Both challenge the optimization stability. Instead, a moderate $\lambda$ thus provides an appropriate trade-off between exploration and exploitation.

## 6 Conclusion and Outlook

In this paper, we present PlannerRFT, a closed-loop and sample-efficient reinforcement fine-tuning framework for diffusion-based planners. Experiments on nuPlan verify its improvements in closed-loop performance. Comparisons with an IL fine-tuning baseline show that these gains arise from effective exploration rather than additional training iterations. Analyses of different exploration policies further highlight PlannerRFT’s scenario-adaptive advantage in sample efficiency.

Limitations and Future Work. PlannerRFT is currently verified on planners with structured abstract inputs, instead of sensory observations like images. Its applicability to visuomotor planners remains underexplored [^1] [^21] [^28]. Nonetheless, its sample-efficient designs upon a pretrained policy laid the foundation of training end-to-end planners in a closed-loop manner with RL, which is left as our future work.

## Acknowledgments

This work is in part supported by the Project supported by the Young Scientists Fund of the National Natural Science Foundation of China (Grant No. 62206172), National Natural Science Foundation of China (Grant No. 52372317) and the Beijing Nova Program. We also appreciate the general research sponsorship from Li Auto.

## References

Supplementary Material

## Appendix A Discussions

Towards a better understanding of this work, we supplement intuitive questions that may raise.

Q1. What makes PlannerRFT effective? PlannerRFT’s effectiveness stems from three key factors:

Enhance Lateral movement. The expert trajectory distribution is dominated by straight-driving maneuvers, causing IL planners to underfit lateral skills such as lane changes and obstacle avoidance. PlannerRFT introduces structured lateral perturbations through guided denoising and reinforces them with reward-oriented optimization, enabling lane changes and obstacle-avoidance behaviors in complex scenes, as shown in Fig. A6 and Fig. 4.

Plan-Motion Alignment. IL planners mimic expert trajectories without considering the downstream controller, leading to execution failures in narrow or high-precision maneuvers. PlannerRFT evaluates executed closed-loop trajectories via the simulator’s vehicle dynamics and updates the planner with execution-level rewards such as collision and off-road penalties. This closed-loop correction bridges the gap between planning and execution and markedly improves maneuver precision and controller feasibility in challenging environments, as shown in Fig. A7.

Trial-and-error Rollouts. Real traffic is dynamic, and blindly reproducing expert behavior can be unsafe. PlannerRFT leverages trial-and-error closed-loop RL to adapt to dynamic traffic, markedly improving interaction capability, as shown in Fig. A8 and Fig. A9, which also explains its strong gains on reactive-traffic benchmarks. We further observe that RL helps mitigate the causal-confusion issues inherent in IL, as shown in Fig. A10.

Q2. Why adopt a dual-branch optimization (PPO and GRPO), and how is training stability maintained? The exploration policy adjusts guidance scales at every simulation step and directly affects long-horizon closed-loop behavior, making PPO suitable for online learning. In contrast, the DiT generator outputs high-dimensional multi-step trajectories that are better optimized offline with group-based updates, where GRPO provides efficient training. Training stability is further ensured by applying policy-guided denoising around a fixed reference trajectory rather than the evolving DiT outputs, and the well-trained reference trajectory distribution prevents collapse and yields steadily improving rewards, as shown in Fig. 6.

Algorithm 1 Guided Denoising for RL Sampling

Current observation $o_{t}$, scene encoder $E_{\text{scene}}$, route encoder $E_{\text{navi}}$, reference DiT $D_{\text{ref}}$, fine-tuned DiT $D_{\theta}$, exploration policy $\pi_{\phi}$, GRPO group size $G_{\text{grpo}}$

// Step 1: Scenario encoding.

 $F_{\text{scene}}\leftarrow E_{\text{scene}}(o_{t})$ $F_{\text{navi}}\leftarrow E_{\text{navi}}(o_{t})$

// Step 2: Get reference trajectory.

$x^{\text{ref}}_{S}\leftarrow z$, $z\sim\mathcal{N}(\mathbf{0},\mathbf{I})$

for $i=1$ to $S$ do $\triangleright$ $S=5$ DDIM steps

    $s\leftarrow s_{i}$ $\triangleright$ VP-SDE timestep

    $x^{\text{ref}}_{s-1}\leftarrow D_{\text{ref}}(x^{\text{ref}}_{s},s,F_{\text{scene}},F_{\text{navi}})$

end for

 $x^{\text{ref}}\leftarrow x^{\text{ref}}_{0}$

// Step 3: Get adaptive exploration direction.

 $(a_{\text{lat.}},b_{\text{lat.}},a_{\text{lon.}},b_{\text{lon}})\leftarrow\pi_{\phi}\!\left(x^{\text{ref}},F_{\text{scene}},F_{\text{navi}}\right)$

// Step 4: Sample multi-modal guidance scales.

for $k=1$ to $G_{\text{grpo}}$ do

    $\eta_{\text{lat}}^{(k)}\sim\mathrm{Beta}(a_{\text{lat}},b_{\text{lat}}),\qquad\eta_{\text{lon}}^{(k)}\sim\mathrm{Beta}(a_{\text{lon}},b_{\text{lon}})$

end for

// Step 5: Guided denoising.

for $k=1$ to $G_{\text{grpo}}$ do

    $x^{(k)}_{S}\leftarrow z$, $z\sim\mathcal{N}(\mathbf{0},\mathbf{I})$

   for $i=1$ to $S$ do

      $s\leftarrow s_{i}$

      $x^{(k)}_{s-1}\leftarrow D_{\theta}(x^{(k)}_{s},s,F_{\text{scene}},F_{\text{navi}},\eta_{\text{lat}}^{(k)},\eta_{\text{lon}}^{(k)},x^{\text{ref}})$ $\triangleright$ classifier-guided denoising following Eq. 5

   end for

    $x^{(k)}\leftarrow x^{(k)}_{0}$

end for

return $\{x^{(k)}\}_{k=1}^{G_{\text{grpo}}}$ $\triangleright$ Multi-modal trajectory samples

Q3. What are potential applications and future directions with the PlannerRFT framework and the nuMax simulator?

Model: Build upon a simple diffusion planner, PlannerRFT can enhance the multi-modality and adaptive sampling in RL. We freeze the encoder and fine-tune only the trajectory DiT, which suggests a potential ability to act as a unified decoder for different input modalities, such as sensor-based E2E planners or language-conditioned VLM/VLA planners.

Simulator: To support our training pipeline, we develop nuMax, a fast online RL simulator designed to facilitate academic research on the nuPlan benchmark. Additional implementation details, limitations, and future development plans are provided in Appendix C.

## Appendix B Implementation Details of PlannerRFT

Training Details. Fig. A2 show the training pipeline for the PlannerRFT framework. For the commonly used online RL, the training pipeline involves two steps: (1) RL sampling and (2) policy update.

For RL sampling, PlannerRFT adopt the policy-guide denoising to generate multi-modal and scenario-adaptive trajectocy group. Algorithm 1 outlines the details of the policy-guide denoising process. We adopt a 5 steps DDIM sampling during training and inference for computational efficiency and exploration stochasticity.

$$
x_{s-1}=\sqrt{\alpha_{s-1}}\,\hat{x}_{s}^{0}+\sqrt{1-\alpha_{s-1}-\sigma_{s}^{2}}\,\epsilon_{\theta}(x_{s},s)+\sigma_{s}z
$$
 
$$
\epsilon_{\theta}(x_{s},s)=\frac{x_{s}-\sqrt{\alpha_{s}\,\hat{x}_{s}^{0}}}{\sqrt{1-\alpha_{s}}}
$$
 
$$
\sigma_{s}=\eta\cdot\sqrt{\frac{1-\alpha_{s-1}}{1-\alpha_{s}}},
$$

where $s$ is the denoising timestep, $\hat{x}_{s}^{0}$ is the model-predicted clean trajectory at timestep $s$, $\sigma_{s}$ controls the stochasticity of DDIM sampling, and $z\sim\mathcal{N}(0,I)$ denotes standard Gaussian noise. Specifically, we set $\eta=1$ during RL training to encourage stochastic exploration, and $\eta=0$ during evaluation for deterministic sampling.

For policy update, PlannerRFT consists of two learnable modules: the exploration policy and the fine-tuned DiT, which have different optimization goals and are trained with different optimization losses.

We use the PPO loss to update the exploration policy, aiming to maximize the long-term cumulative reward in closed-loop planning.

$$
\displaystyle\mathcal{L}_{\text{PPO}}(\phi)=\mathbb{E}_{t}\Big[
$$
$$
\displaystyle\mathcal{L}_{\text{clip}}(\phi)-c_{v}\,(V_{\phi}(s_{t})-V_{t}^{\text{target}})^{2}
$$
 
$$
\displaystyle+c_{e}\,\mathcal{H}\!\left(\pi_{\phi}(\cdot|o_{t})\right)\Big]
$$
 
$$
\displaystyle\mathcal{L}_{\text{clip}}(\phi)
$$
 
$$
\displaystyle=\min\Big(r_{t}(\phi)A_{t},
$$
$$
\displaystyle\qquad\quad\mathrm{clip}\big(r_{t}(\phi),1-\epsilon,1+\epsilon\big)A_{t}\Big)
$$
 
$$
r_{t}(\phi)=\frac{\pi_{\phi}(\eta_{t}\mid o_{t})}{\pi_{\phi{\text{old}}}(\eta_{t}\mid o_{t})},
$$

where $\mathcal{L}_{\text{clip}}$ is the clipped policy objective, $r_{t}(\phi)$ is the importance sampling ratio, $A_{t}$ is the advantage estimate, $V_{\phi}(s_{t})$ is the value prediction, $\mathcal{H}(\pi_{\phi})$ denotes the entropy bonus, and $c_{v},c_{e}$ are the value and entropy coefficients. These hyperparameters are summarized in Tab. A1.

Table A1: Hyperparameters for PlannerRFT.

<table><tbody><tr><td colspan="2">Hyperparameter</td><td>Value</td></tr><tr><td rowspan="2">Guidance</td><td>Max. Lateral Offset <math><semantics><msub><mi>λ</mi> <mtext>lat.</mtext></msub><annotation>\lambda_{\text{lat.}}</annotation></semantics></math></td><td><math><semantics><mrow><mn>2.5</mn> <mo></mo><mrow><mo>(</mo><mi>m</mi><mo>)</mo></mrow></mrow> <annotation>2.5~(m)</annotation></semantics></math></td></tr><tr><td>Max. Longitudinal Offset <math><semantics><msub><mi>λ</mi> <mtext>lon.</mtext></msub><annotation>\lambda_{\text{lon.}}</annotation></semantics></math></td><td><math><semantics><mrow><mn>25</mn> <mrow><mo>(</mo><mo>%</mo><mo>)</mo></mrow></mrow> <annotation>25~(\%)</annotation></semantics></math></td></tr><tr><td rowspan="14">PPO</td><td>Samples</td><td>40M</td></tr><tr><td>Initial Learning Rate</td><td><math><semantics><mrow><mn>2.5</mn> <mo>×</mo> <msup><mn>10</mn> <mrow><mo>−</mo> <mn>4</mn></mrow></msup></mrow> <annotation>2.5\times 10^{-4}</annotation></semantics></math></td></tr><tr><td>Learning Rate Schedule</td><td>Cosine decay</td></tr><tr><td>Number of Envs.</td><td>128</td></tr><tr><td>Env. Steps per Iteration</td><td>32</td></tr><tr><td>Batch Size</td><td>4096</td></tr><tr><td>Mini-batch Size</td><td>4096</td></tr><tr><td>Steps per Epoch</td><td>1</td></tr><tr><td>Epochs</td><td>4</td></tr><tr><td>Value Coefficient <math><semantics><msub><mi>c</mi> <mi>v</mi></msub> <annotation>c_{v}</annotation></semantics></math></td><td>0.5</td></tr><tr><td>Entropy Coefficient <math><semantics><msub><mi>c</mi> <mi>e</mi></msub> <annotation>c_{e}</annotation></semantics></math></td><td>0.01</td></tr><tr><td>Discount Factor</td><td>0.99</td></tr><tr><td>GAE <math><semantics><mi>λ</mi> <annotation>\lambda</annotation></semantics></math></td><td>0.95</td></tr><tr><td>Clip Range <math><semantics><mi>ϵ</mi> <annotation>\epsilon</annotation></semantics></math></td><td>0.2</td></tr><tr><td></td><td>Max Gradient Norm</td><td>0.5</td></tr><tr><td rowspan="8">GRPO</td><td>Initial Learning rate</td><td><math><semantics><mrow><mn>2.5</mn> <mo>×</mo> <msup><mn>10</mn> <mrow><mo>−</mo> <mn>4</mn></mrow></msup></mrow> <annotation>2.5\times 10^{-4}</annotation></semantics></math></td></tr><tr><td>Learning Rate Schedule</td><td>Cosine decay</td></tr><tr><td>Group Size <math><semantics><msub><mi>G</mi> <mrow><mi>g</mi> <mo></mo><mi>r</mi> <mo></mo><mi>p</mi> <mo></mo><mi>o</mi></mrow></msub> <annotation>G_{grpo}</annotation></semantics></math></td><td>8</td></tr><tr><td>Mini-batch Size</td><td>4096</td></tr><tr><td>Steps per Epoch</td><td>6</td></tr><tr><td>Epochs</td><td>1</td></tr><tr><td>Denoising Discount Factor <math><semantics><mi>γ</mi> <annotation>\gamma</annotation></semantics></math></td><td>0.8</td></tr><tr><td>BC loss weight <math><semantics><msub><mi>c</mi> <mi>b</mi></msub> <annotation>c_{b}</annotation></semantics></math></td><td>0.4</td></tr></tbody></table>

Table A2: Comparison of different inference types. “ $\text{Diffusion Planner}_{\texttt{DPM}}$ ” is the official 10-step DPM-solver version of Diffusion Planner [^56]. “w/ guid.” denotes inference with guided denoising, where the guidance scale is set to the mean of the Beta distribution. “w/o guid.” denotes inference without guidance.

| Model | Steps | Latency ($ms$) | Val14-NR | Val14-R |
| --- | --- | --- | --- | --- |
| $\text{Diffusion Planner}_{\texttt{DPM}}$ | 10 | 86.43 | 89.87 | 82.80 |
| PlannerRFT w/ guid. | 10 | 75.48 | 89.83 | 83.93 |
| PlannerRFT w/o guid. | 5 | 34.27 | 89.96 | 84.46 |

We use the GRPO loss to update the fine-tuned DiT, aiming to maximize the reward within the prediction horizon at the current timestep. Following DPPO [^42], each conditional step in the diffusion chain is a Gaussian policy:

$$
\pi_{\theta}(x_{s-1}\mid x_{s})=\mathcal{N}\!\Big(x_{s-1};\;\mu_{\theta}(x_{s},s),\;\sigma_{s}^{2}I\Big)
$$
 
$$
\displaystyle\mu_{\theta}(x_{s},s)=
$$
 
$$
\displaystyle\sqrt{\alpha_{s-1}}\,\hat{x}^{0}_{s}
$$
 
$$
\displaystyle+\sqrt{1-\alpha_{s-1}-\sigma_{s}^{2}}\,\epsilon_{\theta}(x_{s},s),
$$

where $\mu_{\theta}(x_{s},s)$ is the deterministic update term in DDIM, and $\sigma_{s}^{2}I$ controls the sampling stochasticity. Therefore, the optimization objective is to adjust the denoising process such that the conditional policy $\pi_{\theta}(x_{s-1}\mid x_{s})$ is shifted toward trajectories with higher expected rewards:

$$
\displaystyle\mathcal{L}_{G}
$$
 
$$
\displaystyle=-\frac{1}{G_{grpo}}\sum_{k=1}^{G_{grpo}}\frac{1}{S}\sum_{s=1}^{S}\gamma^{s-1}\log\pi_{\theta}\!\left(x^{(k)}_{s-1}\mid x^{(k)}_{s}\right)\,\hat{A}_{k}
$$
 
$$
\displaystyle\quad-\;c_{b}\,\frac{1}{G_{grpo}}\sum_{k=1}^{G_{grpo}}\frac{1}{S}\sum_{t=1}^{S}\log\pi_{\theta}\!\left(\tilde{x}^{(k)}_{s-1}\mid\tilde{x}^{(k)}_{s}\right),
$$

where $\gamma$ is the denoising discount factor. Following ReCogDrive [^29], we incorporate a behavior cloning loss to prevent policy collapse during exploration, and $c_{b}$ denotes the weight of the behavior cloning term.

Inference Details. For inference, we adopt the same 5-step DDIM sampler as in the training phase, without guided denoising or reliance on the reference planner. Table A2 compares different inference settings in terms of denoising steps, latency, and closed-loop performance on the Val14 benchmark. PlannerRFT with guided denoising requires twice as many denoising steps because each step depends on the reference planner’s trajectory. This additional dependency also results in nearly double the inference latency compared to the unguided version. Performance-wise, the guided version is slightly worse than PlannerRFT without guidance. Exploration guidance provides a directional prior that enables sampling around the intended exploration direction, yielding both positive and negative trajectory examples that help refine the learned distribution. Under a limited training budget, however, the model may not fully capture an accurate distribution mean, particularly in fine-grained maneuvering scenarios, leading to slight performance degradation. These effects collectively highlight the sampling efficiency of PlannerRFT.

![[x7 18.png|Refer to caption]]

Figure A1: Visualization of a Cached Scenario. We cache scene elements within a 200 m radius of the ego vehicle, including lanes, dynamic agents, static obstacles, and navigation routes. Lane polygons are drawn in gray with blue centerlines, while navigation routes are highlighted in red. Surrounding vehicles are drawn as black rectangles, pedestrians and cyclists in purple, and static objects in red.

## Appendix C Implementation Details of nuMax

We develop nuMax, a JAX-based, GPU-parallel simulator built upon Waymax [^12], to support large-scale closed-loop training on nuPlan. nuMax achieves significantly higher simulation speed compared to the official nuPlan simulator by re-designing the data representation, scene update pipeline, and agent dynamics entirely around JAX’s functional. Below, we summarize the implementation details.

Scenario Cache. Efficient high-throughput data loading is crucial for large-scale training in closed-loop simulation. In the official nuPlan dataset, scenario recordings are stored in an SQLite <sup>3</sup> database and HD maps reside in a GeoPandas <sup>4</sup> dataframe, requiring the simulator to query both sources at every simulation step to retrieve lane geometry, dynamic agents, and scene context. This stepwise database access limits the simulation throughput. Consequently, nuMax pre-caches the training scenario based on ScenarioMax <sup>5</sup>, a high-performance toolkit for autonomous vehicle scenario-based testing and dataset conversion. Specifically, for temporal context, we extract a fixed window consisting of 20 past frames, the current frame, and 150 future frames at a sampling interval of 0.1 s. For spatial context, we crop all scene elements within a 200 m radius centered on the ego vehicle, including lanes, dynamic agents, static obstacles, and navigation routes. An example visualization of the scenario cache is shown in Fig. A1. All processed data are serialized into TFRecord <sup>6</sup> files, enabling fast sequential I/O, efficient GPU loading, and full compatibility with JAX’s parallelized execution model, thereby eliminating runtime database queries and supporting high-throughput simulation in nuMax.

Tracker and Scorer. Reliable vehicle motion tracking is essential for robust closed-loop simulation. Waymax adopts a vehicle controller built on Perfect Control, but we found it occasionally understeers in sharp-turn scenarios. In nuMax, we replace the perfect-control controller with the two-stage motion controller from the official nuPlan-devkit <sup>7</sup>, which consists of an LQR tracker and a kinematic bicycle model. Our implementation follows the controller used in PDM-Closed <sup>8</sup>, which extends the controller to support batched trajectory inputs, enabling nuMax to track an entire group of candidate trajectories simultaneously during rollouts. We reimplement the two-stage controller in JAX and integrate it into nuMax’s GPU-parallel simulation pipeline, where XLA <sup>9</sup> compilation further accelerates tracking and improves overall computational efficiency.

Comprehensive and principled reward evaluation is essential for effective reinforcement learning policy optimization. Building upon the metrics provided by Waymax, we further incorporate the official nuPlan scoring framework. In particular, our scorer includes both terminal penalties and soft penalties, enabling a more complete assessment of driving quality and safety during closed-loop rollouts.

For terminal penalties, once the ego violates any terminal condition, the simulation is immediately terminated and the reward is set to zero. The terminal penalties include:

- Collision (Col): if the ego vehicle collides with surrounding vehicles, pedestrians, cyclists, or static objects.
- Off road (DAC): if the ego vehicle drove off the drivable area.

For soft penalties, the ego aims to minimize these penalties while avoiding any terminal violations. The soft penalties include:

- Wrong direction (WD): if the ego vehicle drives against the designated lane direction.
- Time to collision (TTC): if the ego vehicle violates the time-to-collision (TTC) safety threshold.
- Comfort (C): if the ego vehicle exhibits excessive longitudinal/lateral acceleration, jerk, or steering rate.
- Ego Progress (EP): measured as the normalized ratio between the ego’s accumulated route progress and that of the expert.
- Speeding (Speed): if the ego vehicle exceeds the speed limit of the current lane or route segment.

Each component score lies within $[0,1]$, and the final reward is obtained by weighted aggregation of all metrics:

$$
\begin{split}r_{t}&=(\texttt{Col}\cdot\texttt{DAC}\cdot\texttt{WD})\\
&\times\left(\frac{w_{1}\texttt{TTC}+w_{2}\texttt{EP}+w_{3}\texttt{C}+w_{4}\texttt{Speed}}{\sum_{i}w_{i}}\right)\in[0,1],\end{split}
$$

where we adopt the official nuPlan weights: $w_{1}=w_{2}=5.0$, $w_{3}=2.0$, and $w_{4}=4.0$. For the open-loop scorer used for GRPO trajectory evaluation, we adopt the same metric terms, but replace the terminal-penalty computation with the survival formulation defined in Eq. 6.

![[x8 13.png|Refer to caption]]

Figure A2: Distributed RL Training Pipeline. Policy inference and learning run in PyTorch DDP, while environment simulation is executed in JAX on rank-0 to avoid XLA conflicts. Observations and replay samples are scattered to all ranks for guided denoising and gradient updates, and planned trajectories are gathered back to rank-0 for simulation, with synchronization at every step.

Distributed RL Training Pipeline. For the reinforcement learning pipeline, we refer to V-Max [^2], a JAX-based high-performance framework built on the Brax <sup>10</sup> engine, which integrates simulation pipelines, observation wrappers, and evaluation metrics. However, most diffusion-based planners widely adopted in the community are implemented in PyTorch, and re-implementing the entire model stack in JAX would be both costly and incompatible with existing pretrained models. To preserve compatibility and facilitate broader community adoption, we therefore retain the policy model in PyTorch. Based on these considerations, we design a hybrid distributed RL training pipeline that couples JAX-based simulation with PyTorch Distributed Data Parallel (DDP) for policy inference and learning.

Fig. A2 illustrates our hybrid distributed RL training pipeline. All JAX-based simulation is executed on rank-0 to avoid XLA device conflicts and duplicate backend initialization that would arise from launching independent JAX runtimes on each PyTorch DDP rank. We distribute the observations and replay samples to all DDP ranks, and aggregate the planned trajectories back to rank 0 for simulation. In addition, we ensure that all ranks are synchronized before each simulation step.

Limitations of nuMax. nuMax currently inherits two key limitations. First, due to XLA’s static-shape constraints, supporting other model input representations would require additional post-processing or re-caching, highlighting the need for a general scenario cache interface. Second, surrounding-vehicle simulation is log-replay rather than IDM, as the latter slows down training; improving the efficiency of the IDM traffic simulation remains an important direction for future optimization.

Table A3: Closed-loop Planning Results on nuPlan Val14, Test14-hard, and Test14 benchmarks. “ $\text{Diffusion Planner}_{\texttt{DPM}}$ ” employs the official 10-step DPM-solver. “ $\text{Diffusion Planner}_{\texttt{DDIM}}$ ” employs a 5-step DDIM sampler, identical to that used in PlannerRFT.

<table><tbody><tr><td rowspan="2">Type</td><td rowspan="2">Planner</td><td colspan="2">Val14</td><td colspan="2">Test14-hard</td><td colspan="2">Test14-random</td></tr><tr><td>NR</td><td>R</td><td>NR</td><td>R</td><td>NR</td><td>R</td></tr><tr><td>Expert</td><td>Log-replay</td><td>93.53</td><td>80.32</td><td>85.96</td><td>68.80</td><td>94.03</td><td>75.86</td></tr><tr><td rowspan="2">Rule</td><td>IDM</td><td>75.60</td><td>77.33</td><td>56.15</td><td>62.26</td><td>70.39</td><td>74.42</td></tr><tr><td>PDM-Closed</td><td>92.84</td><td>92.12</td><td>65.08</td><td>75.19</td><td>90.05</td><td>91.63</td></tr><tr><td rowspan="8">Learning</td><td>PDM-Open</td><td>53.53</td><td>54.24</td><td>33.51</td><td>35.83</td><td>52.81</td><td>57.23</td></tr><tr><td>GameFormer</td><td>13.32</td><td>8.69</td><td>7.08</td><td>6.69</td><td>11.36</td><td>9.31</td></tr><tr><td>PlanTF</td><td>84.27</td><td>76.95</td><td>69.70</td><td>61.61</td><td>85.62</td><td>79.58</td></tr><tr><td>PLUTO</td><td>88.89</td><td>78.11</td><td>70.03</td><td>59.74</td><td>89.90</td><td>78.62</td></tr><tr><td><math><semantics><msub><mtext>Diffusion Planner</mtext> <mtext>DPM</mtext></msub> <annotation>\text{Diffusion Planner}_{\texttt{DPM}}</annotation></semantics></math></td><td>89.87</td><td>82.80</td><td>75.99</td><td>69.22</td><td>89.19</td><td>82.93</td></tr><tr><td><math><semantics><msub><mtext>Diffusion Planner</mtext> <mtext>DDIM</mtext></msub> <annotation>\text{Diffusion Planner}_{\texttt{DDIM}}</annotation></semantics></math></td><td>89.81</td><td>82.94</td><td>76.01</td><td>68.18</td><td>89.14</td><td>82.63</td></tr><tr><td>Flow Planner</td><td>90.43</td><td>83.31</td><td>76.47</td><td>70.42</td><td>89.88</td><td>82.93</td></tr><tr><td>PlannerRFT(Ours)</td><td>89.96(+0.15)</td><td>84.46(+1.52)</td><td>77.16(+1.15)</td><td>72.21(+4.03)</td><td>90.76(+1.62)</td><td>85.80(+3.17)</td></tr></tbody></table>

## Appendix D Additional Ablation Studies

Table A4: Ablation on Guidance Type. Results are reported on the Test14-random reactive benchmark.

<table><tbody><tr><td></td><td colspan="2">Guidance Choices</td><td colspan="7">Closed-loop metrics <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>Training Type</td><td>lateral</td><td>longitudinal</td><td>Collisions</td><td>TTC</td><td>Drivable</td><td>Comfort</td><td>Progress</td><td>Speed</td><td>R-score</td></tr><tr><td><math><semantics><msub><mtext>IL Pretrain</mtext> <mtext>DDIM</mtext></msub> <annotation>\text{IL Pretrain}_{\texttt{DDIM}}</annotation></semantics></math></td><td>✗</td><td>✗</td><td>86.58</td><td>79.05</td><td>94.48</td><td>86.03</td><td>76.99</td><td>97.20</td><td>68.18</td></tr><tr><td rowspan="3">PlannerRFT</td><td>✗</td><td>✓</td><td>87.50</td><td>81.62</td><td>92.65</td><td>86.76</td><td>77.54</td><td>97.99</td><td>69.59</td></tr><tr><td>✓</td><td>✗</td><td>87.31</td><td>80.88</td><td>94.85</td><td>87.50</td><td>76.38</td><td>97.32</td><td>70.18</td></tr><tr><td>✓</td><td>✓</td><td>88.97</td><td>84.93</td><td>95.59</td><td>85.66</td><td>77.17</td><td>98.03</td><td>72.21</td></tr></tbody></table>

Additional planning results in Test14-Random. Table A3 reports the closed-loop performance on the Test14-random benchmark, which contains 261 randomly selected scenarios from the nuPlan Planning Challenge. As shown in Tab. A3, the 5-step DDIM sampler and the ODE-based DPM-solver yield nearly identical results, ensuring a fair comparison. PlannerRFT achieves the best performance in both non-reactive (NR) and reactive (R) settings in Test14-random, improving over the pretrained planner by +1.62 (NR) and +3.17 (R).

Ablation on Guidance Choices. We evaluate the effectiveness of lateral and longitudinal guidance by enabling them individually in the policy-guided denoising process. As shown in Tab. A4, lateral guidance improves performance in Drivable and Comfort, as it enhances sharp-turn performance and produces smoother lateral maneuvers. In contrast, longitudinal guidance performs better in terms of Collisions, TTC, Progress, and Speed, as these behaviors can be controlled through acceleration and deceleration. Combining both forms of guidance results in the best performance, highlighting the complementary effect of lateral and longitudinal exploration in optimizing closed-loop planning.

Ablation on Group Size. We evaluate the impact of group size on performance by testing three different values of $G_{\text{grpo}}$ in the Test14-hard benchmark. As shown in Tab. A5, using a group size of 4 results in suboptimal performance compared to larger group sizes. When the group size is increased to 8 or 12, performance improves, with scores stabilizing around 72.21 (R-score) and 77.16 (NR-score). We choose a group size of 8 to strike an optimal balance between performance and computational efficiency.

Table A5: Ablation on Group number. Results are reported on the Test14-random benchmark.

<table><tbody><tr><td rowspan="2"><math><semantics><msub><mi>G</mi> <mrow><mi>g</mi> <mo></mo><mi>r</mi> <mo></mo><mi>p</mi> <mo></mo><mi>o</mi></mrow></msub> <annotation>G_{grpo}</annotation></semantics></math></td><td colspan="6">Closed-loop metrics <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>R-score</td><td>Coll.</td><td>Driv.</td><td>C.</td><td>Prog.</td><td>NR-score</td></tr><tr><td>4</td><td>71.24</td><td>86.40</td><td>94.85</td><td>84.93</td><td>78.99</td><td>76.31</td></tr><tr><td>8</td><td>72.21</td><td>88.97</td><td>95.59</td><td>85.66</td><td>77.17</td><td>77.16</td></tr><tr><td>12</td><td>72.29</td><td>88.97</td><td>95.22</td><td>85.66</td><td>77.62</td><td>77.04</td></tr></tbody></table>

## Appendix E Additional Qualitative Results

Additional Visualization of Policy-Guided Denoising. We show qualitative comparisons of planned trajectories from Diffusion Planner, DiffusionDrive, and PlannerRFT with policy-guided denoising, as shown in Fig. A3. With policy-guided denoising, PlannerRFT generates a group of multi-modality and scenario-adaptive trajectories for sampling efficient RL training.

Qualitative Results on Safety-Critical Scenarios. We present closed-loop planning results of PlannerRFT on safety-critical scenarios, as shown in Fig. A4 and Fig. A5. These examples demonstrate the planner’s enhanced safety awareness, improved maneuver robustness, and better handling of dynamic interactions.

Qualitative Results on Obstacle Avoidance Scenarios. We present the closed-loop planning results for PlannerRFT on obstacle avoidance scenarios, as shown in Fig. A6 and Fig. A7. These examples demonstrate the planner’s ability to laterally avoid obstacles and to execute precise maneuvers in narrow spaces.

Qualitative Results in Reactive Traffic. We present the closed-loop planning results for PlannerRFT in reactive traffic, as shown in Fig. A8 and Fig. A9. These examples demonstrate the planner’s enhanced decision-making and planning capability in interactive scenarios.

Qualitative Results on Causal-Confusion Scenarios. We present closed-loop planning results on a causal-confusion scenario, as shown in Fig. A10, which illustrate the advantage of RL in mitigating the causal-confusion issues inherent in imitation learning.

## Appendix F License of Assets

Data for nuPlan [^24] are provided under the CC-BY-NC 4.0 license. Our IL-pretrained model follows the implementation of Diffusion Planner [^56]. As the original repository <sup>11</sup> does not provide an explicit license, the referenced code is used solely for academic research and reproducibility purposes, and all rights remain with the original authors. nuMax is a re-implementation of Waymax [^12] for non-commercial research, in accordance with the Waymax License Agreement for Non-Commercial Use <sup>12</sup>. Scenario caching in nuMax is developed upon ScenarioMax, which is released under the Apache-2.0 license. Our RL training pipeline references V-Max [^2] and Brax, distributed under the MIT License and Apache-2.0 License, respectively. The reinforcement learning algorithms further draw upon DPPO [^42] and ReCogDrive [^29], which are released under the MIT License and Apache-2.0 License. All source code and models developed in this work will be made publicly available under the Apache License 2.0.

![[x9 7.png|Refer to caption]]

Figure A3: Visualization of Diffusion Planner (a, w/o guided denoising), DiffusionDrive (b) and PlannerRFT (c, w/ guided denoising). For Diffusion Planner and PlannerRFT, we resample 4 trajectories, for DiffusionDrive we use 20 anchor noises. We visualize the planned trajectory over a 4 s e c o n d second horizon. Note that DiffusionDrive is evaluated on the NAVSIM navtest split with camera and LiDAR inputs; for visualization, we render all planners on the same scenario shared between NAVSIM and nuPlan. PlannerRFT demonstrates multi-modal and scenario-adaptive trajectory generation through policy-guided denoising.

![[x10 9.png|Refer to caption]]

Figure A4: Intersection Pedestrian Avoidance. The ego vehicle intends to make a right turn at an intersection while pedestrians are crossing. (a) The IL Pretrained planner collision with a pedestrian at t sim = 7.5 s t\_{\\text{sim}}=7.5~s. (b) PlannerRFT waits for all pedestrians to finish crossing and then proceeds with the right turn. In each frame shot, the simulation position and planning trajectory are marked as orange, the ground-truth position ground-truth trajectory recorded in the driving log are marked as gray and blue, respectively. Surrounding vehicles are marked as black rectangles with white arrows indicating heading. The pedestrians are marked as purple, and the static objects are marked as red. The lane polygons are marked gray and the navigation routes are marked as light blue with centerline arrows

![[x11 6.png|Refer to caption]]

Figure A5: Emergency Brake in Reactive Traffic. A safety-critical scenario in which surrounding vehicles governed by the IDM policy enter a deadlock at the initial timestep ( t sim = 0 s t\_{\\text{sim}}=0~s ), blocking all traffic, while the ego vehicle approaches at high speed as recorded in the log. (a) The IL-pretrained planner fails to brake in time and collides with the preceding vehicle. (b) PlannerRFT detects the stationary lead vehicle and applies braking at 1 t\_{\\text{sim}}=1~s, successfully avoiding the collision.

![[x12 5.png|Refer to caption]]

Figure A6: S-Curve Lane Change. The ego vehicle starts in the left lane of an S-curve, with a stationary vehicle ahead in the same lane. (a) The IL-pretrained planner keeps the lane and collides with the stationary vehicle at t sim = 7 s t\_{\\text{sim}}=7~s. (b) PlannerRFT performs a lane change to the right at 4 t\_{\\text{sim}}=4~s, bypassing the stationary vehicle.

![[x13 8.png|Refer to caption]]

Figure A7: Traffic-Cone Narrowing. The ego vehicle is driving on a curved road with traffic cones. (a) The IL Pretrained planner based ego vehicle fails to avoid the traffic cone in time, colliding with it at t sim=13 s t\_{\\text{sim=13}}s. (b) PlannerRFT enables the ego vehicle to finely adjust the trajectory, successfully steering the ego vehicle between the two cones.

![[x14 4.png|Refer to caption]]

Figure A8: Blocked Right-Turn in Reactive Traffic. The ego vehicle intends to turn right at the upcoming intersection. (a) The IL Pretrained planner causes the ego vehicle to forcibly change lanes, leading to a collision with a long vehicle on the right at t sim = 12 s t\_{\\text{sim}}=12~s (b) PlannerRFT enables the ego vehicle to consider the long vehicle proceeding straight, hence the ego vehicle decides to delay the lane change, which avoiding a collision.

![[x15 4.png|Refer to caption]]

Figure A9: Unprotected Right-Turn in Reactive Traffic. The ego vehicle attempts a right turn at an intersection while surrounding vehicles approach from the cross traffic. (a) The IL Pretrained planner causes the ego vehicle to hesitate when turning right, ultimately leading to a collision with an oncoming vehicle from the left at t sim = 10 s t\_{\\text{sim}}=10~s (b) PlannerRFT enables the ego vehicle smoothly and successfully completes the right turn before the arrival of the oncoming vehicle from the left.

![[x16 2.png|Refer to caption]]

Figure A10: A Causal-Confusion Scenario. The ego vehicle turns right at an intersection. (a) The IL pretrained planner directs the ego vehicle to turn right and then pull over to the side of the road. Off-road at t sim = 10 s t\_{\\text{sim}=10~s}. This behavior is likely due to causal confusion: a large number of scenarios in the training data where vehicles turn right and stop to pick up passengers. (b) PlannerRFT guides the ego vehicle to turn right and then proceed straight, a maneuver that aligns with common sense and avoids causal confusion.

[^1]: W. Cao, M. Hallgarten, T. Li, D. Dauner, X. Gu, C. Wang, Y. Miron, M. Aiello, H. Li, I. Gilitschenski, et al. (2025) Pseudo-simulation for autonomous driving. arXiv preprint arXiv:2506.04218. Cited by: §6.

[^2]: V. Charraut, T. Tournaire, W. Doulazmi, and T. Buhet (2025) V-Max: making rl practical for autonomous driving. arXiv preprint arXiv:2503.08388. Cited by: Appendix C, Appendix F, §4.3.

[^3]: J. Cheng, Y. Chen, and Q. Chen (2024) Pluto: pushing the limit of imitation learning-based planning for autonomous driving. arXiv preprint arXiv:2404.14327. Cited by: §5.1.

[^4]: J. Cheng, Y. Chen, X. Mei, B. Yang, B. Li, and M. Liu (2024) Rethinking imitation-based planners for autonomous driving. In ICRA, Cited by: §5.1.

[^5]: C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song (2023) Diffusion Policy: visuomotor policy learning via action diffusion. IJRR. Cited by: §1.

[^6]: R. Coulom (2006) Efficient selectivity and backup operators in monte-carlo tree search. In ECML, Cited by: §1.

[^7]: M. F. Cusumano-Towner, D. Hafner, A. Hertzberg, B. Huval, A. Petrenko, E. Vinitsky, E. Wijmans, T. W. Killian, S. Bowers, O. Sener, et al. (2025) Robust autonomy emerges from self-play. In ICML, Cited by: §1.

[^8]: D. Dauner, M. Hallgarten, A. Geiger, and K. Chitta (2023) Parting with misconceptions about learning-based vehicle motion planning. In CoRL, Cited by: §5.1.

[^9]: P. Dhariwal and A. Nichol (2021) Diffusion models beat gans on image synthesis. In NeurIPS, Cited by: §4.2.

[^10]: H. Gao, S. Chen, B. Jiang, B. Liao, Y. Shi, X. Guo, Y. Pu, H. Yin, X. Li, X. Zhang, Y. Zhang, W. Liu, Q. Zhang, and X. Wang (2025) RAD: training an end-to-end driving policy via large-scale 3dgs-based reinforcement learning. In NeurIPS, Cited by: §1.

[^11]: Y. Gao, Y. Wang, A. Jiang, H. Yuwen, W. Shuo, S. Hao, and W. Jijun (2025) DiffVLA++: bridging cognitive reasoning and end-to-end driving through metric-guided alignment. arXiv preprint arXiv:2510.17148. Cited by: §2.

[^12]: C. Gulino, J. Fu, W. Luo, G. Tucker, E. Bronstein, Y. Lu, J. Harb, X. Pan, Y. Wang, X. Chen, et al. (2023) Waymax: an accelerated, data-driven simulator for large-scale autonomous driving research. In NeurIPS, Cited by: Appendix C, Appendix F, §1, §4.3.

[^13]: S. Hamdan, C. Sima, Z. Yang, H. Li, and F. Guney (2025) ETA: efficiency through thinking ahead, a dual approach to self-driving with large models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, Cited by: §2.

[^14]: J. Ho, A. Jain, and P. Abbeel (2020) Denoising diffusion probabilistic models. In NeurIPS, Cited by: §4.5.

[^15]: Z. Huang, H. Liu, and C. Lv (2023) GameFormer: game-theoretic modeling and learning of transformer-based interactive prediction and planning for autonomous driving. In ICCV, Cited by: §5.1.

[^16]: Z. Huang, X. Weng, M. Igl, Y. Chen, Y. Cao, B. Ivanovic, M. Pavone, and C. Lv (2025) Gen-Drive: enhancing diffusion generative driving policies with reward modeling and reinforcement learning fine-tuning. In ICRA, Cited by: §1.

[^17]: B. Jaeger, D. Dauner, J. Beißwenger, S. Gerstenecker, K. Chitta, and A. Geiger (2025) CaRL: learning scalable planning policies with simple rewards. In CoRL, Cited by: §1.

[^18]: X. Jia, Y. Gao, L. Chen, J. Yan, P. L. Liu, and H. Li (2023) Driveadapter: breaking the coupling barrier of perception and planning in end-to-end autonomous driving. In ICCV, Cited by: §3.

[^19]: A. Jiang, Y. Gao, Z. Sun, Y. Wang, J. Wang, J. Chai, Q. Cao, Y. Heng, H. Jiang, Y. Dong, et al. (2025) DiffVLA: vision-language guided diffusion planning for autonomous driving. arXiv preprint arXiv:2505.19381. Cited by: §2.

[^20]: C. Jiang, A. Cornman, C. Park, B. Sapp, Y. Zhou, D. Anguelov, et al. (2023) MotionDiffuser: controllable multi-agent motion prediction using diffusion. In CVPR, Cited by: §2.

[^21]: J. Jiang, N. Song, J. Li, X. Zhu, and L. Zhang (2025) RealEngine: simulating autonomous driving in realistic context. arXiv preprint arXiv:2505.16902. Cited by: §6.

[^22]: M. Jiang, Y. Bai, A. Cornman, C. Davis, X. Huang, H. Jeon, S. Kulshrestha, J. Lambert, S. Li, X. Zhou, et al. (2024) SceneDiffuser: efficient and controllable driving simulation initialization and rollout. In NeurIPS, Cited by: §2.

[^23]: S. Jiao, K. Qian, H. Ye, Y. Zhong, Z. Luo, S. Jiang, Z. Huang, Y. Fang, J. Miao, Z. Fu, et al. (2025) EvaDrive: evolutionary adversarial policy optimization for end-to-end autonomous driving. arXiv preprint arXiv:2508.09158. Cited by: §1.

[^24]: N. Karnchanachari, D. Geromichalos, K. S. Tan, N. Li, C. Eriksen, S. Yaghoubi, N. Mehdipour, G. Bernasconi, W. K. Fong, Y. Guo, et al. (2024) Towards learning-based planning: the nuplan benchmark for real-world autonomous driving. In ICRA, Cited by: Appendix F, §1, §5.1.

[^25]: D. Li, J. Ren, Y. Wang, X. Wen, P. Li, L. Xu, K. Zhan, Z. Xia, P. Jia, X. Lang, et al. (2025) Finetuning generative trajectory model with reinforcement learning from human feedback. arXiv preprint arXiv:2503.10434. Cited by: §1.

[^26]: K. Li, Z. Li, S. Lan, Y. Xie, Z. Zhang, J. Liu, Z. Wu, Z. Yu, and J. M. Alvarez (2025) Hydra-MDP++: advancing end-to-end driving via expert-guided hydra-distillation. arXiv preprint arXiv:2503.12820. Cited by: §2.

[^27]: Q. Li, X. Jia, S. Wang, and J. Yan (2024) Think2Drive: efficient reinforcement learning by thinking with latent world model for autonomous driving (in carla-v2). In ECCV, Cited by: §2.

[^28]: T. Li, Y. Qiu, Z. Wu, C. Lindström, P. Su, M. Nießner, and H. Li (2025) MTGS: multi-traversal gaussian splatting. arXiv preprint arXiv:2503.12552. Cited by: §6.

[^29]: Y. Li, K. Xiong, X. Guo, F. Li, S. Yan, G. Xu, L. Zhou, L. Chen, H. Sun, B. Wang, et al. (2025) ReCogDrive: a reinforced cognitive framework for end-to-end autonomous driving. arXiv preprint arXiv:2506.08052. Cited by: Appendix B, Appendix F, §2, §4.4.

[^30]: Z. Li, K. Li, S. Wang, S. Lan, Z. Yu, Y. Ji, Z. Li, Z. Zhu, J. Kautz, Z. Wu, et al. (2024) Hydra-MDP: end-to-end multimodal planning with multi-target hydra-distillation. arXiv preprint arXiv:2406.06978. Cited by: §2.

[^31]: Z. Li, S. Wang, S. Lan, Z. Yu, Z. Wu, and J. M. Alvarez (2025) Hydra-NeXt: robust closed-loop driving with open-loop training. arXiv preprint arXiv:2503.12030. Cited by: §2.

[^32]: B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang, et al. (2025) DiffusionDrive: truncated diffusion model for end-to-end autonomous driving. In CVPR, Cited by: §1, §1, §2, Table 2, Table 2.

[^33]: H. Lin, Y. Zhang, W. Ding, J. Wu, and D. Zhao (2025) Model-based policy adaptation for closed-loop end-to-end autonomous driving. In NeurIPS, Cited by: §1.

[^34]: H. Liu, L. Chen, Y. Qiao, C. Lv, and H. Li (2024) Reasoning multi-agent behavioral topology for interactive autonomous driving. In NeurIPS, Cited by: §3.

[^35]: H. Liu, T. Li, H. Yang, L. Chen, C. Wang, K. Guo, H. Tian, H. Li, H. Li, and C. Lv (2025) Reinforced refinement with self-aware expansion for end-to-end autonomous driving. arXiv preprint arXiv:2506.09800. Cited by: §2.

[^36]: S. Liu, W. Chen, W. Li, Z. Wang, L. Yang, J. Huang, Y. Zhang, Z. Huang, Z. Cheng, and H. Yang (2025) BridgeDrive: diffusion bridge policy for closed-loop trajectory planning in autonomous driving. arXiv preprint arXiv:2509.23589. Cited by: §2.

[^37]: C. Lu, H. Chen, J. Chen, H. Su, C. Li, and J. Zhu (2023) Contrastive energy prediction for exact energy-guided diffusion sampling in offline reinforcement learning. In ICML, Cited by: §1, §4.2.

[^38]: C. Lu, Y. Zhou, F. Bao, J. Chen, C. Li, and J. Zhu (2022) Dpm-solver: a fast ode solver for diffusion probabilistic model sampling in around 10 steps. In NeurIPS, Cited by: §5.1.

[^39]: Y. Lu, J. Fu, G. Tucker, X. Pan, E. Bronstein, R. Roelofs, B. Sapp, B. White, A. Faust, S. Whiteson, et al. (2023) Imitation is not enough: robustifying imitation with reinforcement learning for challenging driving scenarios. In IROS, Cited by: §1.

[^40]: W. Peebles and S. Xie (2023) Scalable diffusion models with transformers. In ICCV, Cited by: §3.

[^41]: Z. Peng, W. Luo, Y. Lu, T. Shen, C. Gulino, A. Seff, and J. Fu (2024) Improving agent behaviors with rl fine-tuning for autonomous driving. In ECCV, Cited by: §1.

[^42]: A. Z. Ren, J. Lidard, L. L. Ankile, A. Simeonov, P. Agrawal, A. Majumdar, B. Burchfiel, H. Dai, and M. Simchowitz (2025) Diffusion policy policy optimization. In ICLR, Cited by: Appendix B, Appendix F, §4.4.

[^43]: J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov (2017) Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347. Cited by: §1.

[^44]: Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu, et al. (2024) DeepSeekMath: pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300. Cited by: §1, §1.

[^45]: D. Silver, A. Huang, C. J. Maddison, A. Guez, L. Sifre, G. Van Den Driessche, J. Schrittwieser, I. Antonoglou, V. Panneershelvam, M. Lanctot, et al. (2016) Mastering the game of go with deep neural networks and tree search. Nature. Cited by: §1.

[^46]: J. Song, C. Meng, and S. Ermon (2021) Denoising diffusion implicit models. In ICLR, Cited by: §4.5.

[^47]: T. Tan, Y. Zheng, R. Liang, Z. Wang, K. Zheng, J. Zheng, J. Li, X. Zhan, and J. Liu (2025) Flow matching-based autonomous driving planning with advanced interactive behavior modeling. arXiv preprint arXiv:2510.11083. Cited by: §2, §5.1.

[^48]: X. Tang, M. Kan, S. Shan, and X. Chen (2025) Plan-R1: safe and feasible trajectory planning as language modeling. arXiv preprint arXiv:2505.17659. Cited by: §2.

[^49]: M. Treiber, A. Hennecke, and D. Helbing (2000) Congested traffic states in empirical observations and microscopic simulations. Physical review E. Cited by: §5.1.

[^50]: L. Wang, Ö. Ş. Taş, M. Steiner, and C. Stiller (2025) FlowDrive: moderated flow matching with data balancing for trajectory planning. arXiv preprint arXiv:2509.21961. Cited by: §2.

[^51]: Z. Xing, X. Zhang, Y. Hu, B. Jiang, T. He, Q. Zhang, X. Long, and W. Yin (2025) GoalFlow: goal-driven flow matching for multimodal trajectories generation in end-to-end autonomous driving. In CVPR, Cited by: §1.

[^52]: B. Yang, H. Su, N. Gkanatsios, T. Ke, A. Jain, J. Schneider, and K. Fragkiadaki (2024) Diffusion-ES: gradient-free planning with diffusion for autonomous and instruction-guided driving. In CVPR, Cited by: §1.

[^53]: B. Zhang, J. Li, N. Song, and L. Zhang (2025) Perception in plan: coupled perception and planning for end-to-end autonomous driving. arXiv preprint arXiv:2508.11488. Cited by: §3.

[^54]: D. Zhang, J. Liang, K. Guo, S. Lu, Q. Wang, R. Xiong, Z. Miao, and Y. Wang (2025) CarPlanner: consistent auto-regressive trajectory planning for large-scale reinforcement learning in autonomous driving. In CVPR, Cited by: §2.

[^55]: Z. Zhang, A. Liniger, D. Dai, F. Yu, and L. Van Gool (2021) End-to-end urban driving by imitating a reinforcement learning coach. In ICCV, Cited by: §1.

[^56]: Y. Zheng, R. Liang, K. Zheng, J. Zheng, L. Mao, J. Li, W. Gu, R. Ai, S. E. Li, X. Zhan, and J. Liu (2025) Diffusion-based planning for autonomous driving with flexible guidance. In ICLR, Cited by: Table A2, Table A2, Appendix F, §1, §2, §5.1, §5.1.

[^57]: Y. Zhou, N. Ye, W. Ljungbergh, T. Li, J. Yang, Z. Yang, H. Zhu, C. Petersson, and H. Li (2025) Decoupled diffusion sparks adaptive scene generation. In ICCV, Cited by: §2.

[^58]: Z. Zhou, T. Cai, Y. Zhao, Z. Huang, B. Zhou, and J. Ma (2025) AutoVLA: a vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. arXiv preprint arXiv:2506.13757. Cited by: §2.
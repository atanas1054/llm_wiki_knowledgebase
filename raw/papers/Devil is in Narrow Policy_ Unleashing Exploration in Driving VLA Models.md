---
title: "Devil is in Narrow Policy: Unleashing Exploration in Driving VLA Models"
source: "https://arxiv.org/html/2603.06049"
author:
published:
created: 2026-04-05
description:
tags:
  - "clippings"
---
Canyu Chen <sup>1,3</sup>  Yuguang Yang <sup>2,3</sup> <sup>1</sup>  Zhewen Tan <sup>6</sup>  Yizhi Wang <sup>7</sup>  Ruiyi Zhan <sup>6</sup>  
Haiyan Liu <sup>4</sup>  Xuanyao Mao <sup>4</sup>  Jason Bao <sup>4</sup>  Xinyue Tang <sup>4</sup>  
Linlin Yang <sup>5</sup>  Bingchuan Sun <sup>4</sup> <sup>2</sup>  Yan Wang <sup>3</sup> <sup>2</sup>  Baochang Zhang <sup>8</sup>  
  
<sup>1</sup> National Superior College for Engineers, Beihang University  
<sup>2</sup> School of Electronic Information Engineering, Beihang University  
<sup>3</sup> Institute for AI Industry Research, Tsinghua University   <sup>4</sup> Lenovo Group Limited  
<sup>5</sup> State Key Laboratory of Media Convergence and Communication, Communication University of China  
School of { <sup>6</sup> Computer Science and Engineering, <sup>7</sup> Cyber Science and Technology,  
<sup>8</sup> Artificial Intelligence }, Beihang University Equal contribution:{chencanyu,guangbuaa}@buaa.edu.cnCorresponding author: lyang@cuc.edu.cn, sunbc1@lenovo.com, wangyan@air.tsinghua.edu.cnProject Lead

###### Abstract

We identify a fundamental Narrow Policy limitation undermining the performance of autonomous VLA models, where driving Imitation Learning (IL) tends to collapse exploration and limit the potential of subsequent Reinforcement Learning (RL) stages, which often saturate prematurely due to insufficient feedback diversity. Thereby, we propose Curious-VLA, a framework that alleviates the “exploit–explore” dilemma through a two-stage design. During IL, we introduce a Feasible Trajectory Expansion (FTE) strategy to generate multiple physically valid trajectories and a step-wise normalized trajectory representation to adapt this diverse data. In the RL stage, we present Adaptive Diversity-Aware Sampling (ADAS) that prioritizes high-diversity samples and introduce Spanning Driving Reward (SDR) with a focal-style weighting to amplify reward’s value span for improving sensitivity to driving quality. On the Navsim benchmark, Curious-VLA achieves state-of-the-art results (PDMS 90.3, EPDMS 85.4) and a Best-of-N PDMS of 94.8, demonstrating its effectiveness in unlocking the exploratory potential of VLA models. Code: [https://github.com/Mashiroln/curious\_vla.git](https://github.com/Mashiroln/curious_vla.git).

## 1 Introduction

![[x1 11.png|Refer to caption]]

(a) Quantitative comparison for Behavioral Diagnostics.

How to effectively leverage Vision-Language Models (VLMs) remains a central research question in autonomous driving. Recent studies have progressively extended VLMs from perception and understanding to decision-making, facilitating end-to-end Vision-Language-Action (VLA) systems for driving. Early works such as DriveGPT [^17] and LINGO [^32] [^34] focused on enhancing driving-scene comprehension with language instruction/behavior. Subsequent research has advanced toward reasoning and control, incorporating a richer environmental context [^53] [^13] [^27] [^25] [^31] within a single framework.

Current driving VLAs can be broadly categorized into two dominant paradigms: (i) VLA-Planner [^27] [^23] which relies on an additional trajectory planner module to predict future motion distributions, and (ii) VLA-Token [^54] [^53] [^33] which directly produces trajectory tokens from the LLM decoder. Despite architectural differences, both paradigms adhere to a similar two-stage training pipeline: an initial Imitation Learning (IL) stage via Supervised Fine Tuning (SFT) to acquire basic trajectory planning and reasoning capability, and a Reinforcement Learning (RL) stage with chain-of-thought (CoT) optimization to enhance reasoning [^54] [^33] [^27]. However, this two-stage pipeline (first IL, then RL) suffers from a fundamental Narrow Policy (NP) limitation, characterized by an inherent exploit–explore imbalance — the IL stage over-exploits ground-truth trajectories, leading to collapsed exploration and consequently restricting policy updates during RL fine-tuning.

This NP problem has been largely neglected in previous driving VLA works. We evaluate two representative baselines, QwenVL-2.5 (VLA-Token) and ReCogDrive (VLA-Planner), on the Navsim navtrain subset [^10]. For each model, we sample $k$ trajectories by $k$ times of inferences, and evaluate them using three Behavioral Diagnostics: (i) Diversity, measured by mean pairwise ADE/FDE, which quantifies trajectory spread; (ii) Quality, measured by min-ADE/FDE, indicating the best feasible trajectory; (iii) Performance, equal to mean PDMS of Navsimv1 [^10]. As shown in Fig. 1(a), both baselines exhibit evident exploration collapse, with extremely low trajectory diversity and limited trajectory quality. Fig. 1(b) further illustrates this effect—despite multiple feasible routes, the sampled trajectories converge to a single mode, even leading to unsafe behaviors. The narrow policy learned through SFT results in a low-entropy initialization for the subsequent RL stage. We theoretically analyze this phenomenon in Sec. 3.2. Since critic-free RL algorithms (*e.g*., GRPO [^40] [^14]) rely on diverse samples to estimate policy gradients, such narrow policy lead to early saturation and limited learning feedback [^48] [^8]. Consequently, the GRPO RL-training undermines VLA’s performances (see Sec 5.5 experiments).

To breakup the limitation, we propose Curious-VLA, a novel training framework that systematically unleashes exploration for VLM itself without any additional module. In the IL stage, we consider the ground-truth (GT) trajectory as just one of the potential human driving behaviors. Therefore, we introduce Feasible Trajectory Expansion (FTE) data synthesizing scheme by generating multiple physically valid driving paths, so called feasible trajectory, with VLA-Planner’s diffusion module [^27]. This data synthesizing scheme largely increases the training trajectory’s diversity. To accommodate these more diverse trajectories, we normalize each trajectory in a step-wise manner, enhancing the separability of diverse driving behaviors and consequently alleviating the NP problem. In the RL stage, to encourage exploration, we further introduce two complementary components: an Adaptive Diversity-Aware Sampling (ADAS) strategy and the Spanning Driving Reward (SDR). ADAS prioritizes samples that exhibit exploratory variance by dropping the training examples whose predicted trajectories remain highly similar across multiple inference passes. This encourages the policy to refine diverse driving behaviors and prevents premature convergence toward a single dominant pattern. Besides, to further promote effective exploration, we introduce the SDR that reformulates the original driving reward by amplifying its reward value span through a focal-loss style function, which improves the reward function’s sensitivity to driving quality. Our contributions are as follows:

- We identify the “Narrow Policy” problem, a fundamental bottleneck in the IL-RL pipeline that hinders autonomous driving VLAs. Furthermore, we introduce Behavioral Diagnostics to quantitatively verify this phenomenon.
- We propose Curious-VLA, a novel framework that systemically fosters the exploration of VLA model.
- On the Navsim benchmark, Curious-VLA achieves the State-of-The-Art (SoTA) performance of 90.3 PDMS. Furthermore, its Best-of-N PDMS of 94.8 effectively validates that our methods successfully unleash the exploration potential of VLA models.

## 2 Related Work

![[x3 9.png|Refer to caption]]

Figure 2: Overall Pipeline of Curious-VLA. As identified in Sec. 3.2, existing VLAs with the IL-RL pipeline suffer from “Narrow Policy” and tend to generate overlap and unsafe behaviors. In this case, we introduce Curious-VLA, which takes into account both the think process and planing waypoint, to improve the diversity, quality and performance (middle panel). Specifically, Curious-VLA alleviates the Narrow Policy in both the IL stage (left panel) and the RL stage (right panel). For IL, to improve the separability of trajectory patterns, we generate diverse trajectories, structure the driving reasoning process into a four-stage CoT, and normalize each prediction step (See Sec. 4.1 ). For RL, to address the advantage collapse problem and sustain exploration, we introduce adaptive diverse aware sampling and spanning driving reward (See Sec. 4.2 ).

VLA Models for Autonomous Driving. End-to-End autonomous driving systems [^16] [^20] [^5] [^28] [^30] [^19] are rapidly moving from modular pipelines that decouple driving tasks to perception, prediction, planning to a unified architecture. Early research primarily utilized VLMs for scene understanding and reasoning, such as captioning, question answering, or intention recognition in driving scenarios [^34] [^17]. Recent advancements have extended VLMs to direct action planning, diverging into two paradigms, VLA-Planner and VLA-Token. The first paradigm, VLA-Planner employs the VLM as a semantic reasoner that guides an external planner (*e.g*., CarLLaVA [^37], RegCogDrive [^27], ORION [^13], ImagiDrive [^23], DriveVLA-W0 [^25]). Another paradigm, VLA-Token treats trajectory planning as a sequence generation task, where the VLM directly predicts action or waypoint tokens via auto-regressive or diffusion language model. Representative approaches based on text waypoint include EMMA [^18], OpenEMMA [^45], AdaThinkDrive [^33], Impromptu-VLA [^4] and Poutine [^38]. In contrast, some methods represent trajectories as action tokens, including AutoVLA [^54] and SMART [^44]. Despite different paradigms, existing methods suffer from Narrow Policy, primarily stemming from the explore-exploit dilemma, especially the lack of diversity. Our Curious-VLA follows VLA-Token and proposes a systematic framework regarding data, sampling and reward during both IL and RL, thus unlocking the potential of VLA’s exploration.

Reinforcement Learning with Verifiable Reward. Recent works extend Reinforcement Learning from Human Feedback (RLHF) [^6] [^35] to Verifiable Reward (RLVR) [^40], optimizing policies by measurable outcomes instead of subjective preferences. DeepSeek-R1 [^14] and GRPO [^40] eliminate value networks through group-based normalization, improving stability over PPO [^39]. Follow-up methods such as DAPO [^48] and KL-CoV/CLIP-Cov [^8] refine advantage estimation or enhance exploration via diversity or curiosity [^11] [^9]. In autonomous driving, RLVR-style fine-tuning has been adopted in TrajHF [^22], EvaDrive [^21], AutoVLA [^54], and ReCogDrive [^27], forming a standard imitation–reinforcement pipeline for VLA models. However, the exploration of RL in VLA is still limited. It is crucial to improve RL’s reward mechanisms to enhance exploration. To address this, we form a Diversity-Aware Reinforcement Learning approach by improving both the training data sampling and reward function design.

## 3 Preliminary and Narrow Policy

We first formulate the VLA training pipeline in Sec. 3.1 and then provide an analysis of the neglected Narrow Policy in Sec. 3.2.

### 3.1 Preliminary: VLA Training Pipeline

Following prior works [^54] [^16], we formulate a driving VLA as a unified generative policy $\pi_{\theta}$ that maps multimodal observations $\mathcal{X}$ into an action sequence $\tau=\{w_{1},\dots,w_{T}\}$ with length $T$, where $w_{i}$ represents the ego’s spatial and speed action. Specifically, the multimodal input $\mathcal{X}$ includes multi-view camera images $\mathcal{C}$, textual instructions $\mathcal{I}$ (*e.g*., “turn left”), and ego-vehicle states $\mathcal{S}$ (*e.g*., speed, acceleration, past control actions). The policy output $\tau$ could be a sequence of waypoints for VLA-Planner or discretized tokens for VLA-Token. Here, we adopt the VLA-Token paradigm as default, consisting of two stages: Imitation Learning (IL) and Reinforcement Learning (RL).

Imitation Learning. In VLA-Token paradigm, the policy is initialized by SFT that maximize the likelihood of generated text output trajectory tokens $\mathbf{y}^{*}$ via a cross entropy loss:

$$
\mathcal{L}_{\text{SFT}}(\theta)=-\mathop{\mathbb{E}}_{(\mathcal{X},\mathbf{y}^{*})\sim\mathcal{D}}\left[\frac{1}{L}\sum_{t=1}^{L}\log\pi_{\theta}(y_{t}^{*}|\mathbf{y}^{*}_{<t},\mathcal{X})\right],
$$

which establishes VLA’s basic planning capabilities.

Reinforcement Learning. To foster genuine environmental understanding and active driving reasoning, the SFT policy $\pi_{\text{sft}}$ is further refined using RL. For RL, Group Relative Policy Optimization (GRPO) [^40] is commonly used and can eliminate the value network by normalizing the advantages within a sampled group. For each input $\mathcal{X}$, a group of $G$ outputs $\{\mathbf{y}_{i}\}_{i=1}^{G}$ is sampled from the old policy $\pi_{\theta_{\text{old}}}$. The training objective combines a clipped loss and a KL divergence constraint:

$$
\displaystyle\mathcal{J}_{\text{GRPO}}(\theta)=
$$
 
$$
\displaystyle~~~~\mathop{\mathbb{E}}_{\mathcal{X},\{\mathbf{y}_{i}\}\sim\pi_{\theta_{\text{old}}}}\bigg[\frac{1}{G}\sum_{i=1}^{G}\Big(\hat{\mathcal{J}}_{i}(\theta)-\beta\mathbb{D}_{\text{KL}}(\pi_{\theta}||\pi_{\text{sft}})\Big)\bigg],
$$

where $\hat{\mathcal{J}}_{i}(\theta)$ is the clipped advantage term:

$$
\hat{\mathcal{J}}_{i}(\theta)=\min\left(\rho_{i}(\theta)A_{i},\text{clip}\left(\rho_{i}(\theta),1-\epsilon,1+\epsilon\right)A_{i}\right),
$$

with likelihood ratio $\rho_{i}(\theta)=\frac{\pi_{\theta}(\mathbf{y}_{i}|\mathcal{X})}{\pi_{\theta_{\text{old}}}(\mathbf{y}_{i}|\mathcal{X})}$. The advantage $A_{i}$ is computed by standardizing the group rewards:

$$
A_{i}=\frac{R(\mathbf{y}_{i})-\mu_{R}}{\sigma_{R}+\xi}.
$$

Here, $\mu_{R},\sigma_{R}$ are the mean and standard deviation of rewards within the group.

### 3.2 Analysis of the Narrow Policy (NP)

We first analyze the emergence of NP problems in (1)-(3). Then, we provide practical analysis metrics in (4) towards the NP problem.

(1) Optimization Objective Mismatch. The cross-entropy loss in Eq. 1 treats all non-ground-truth tokens as equally incorrect [^24] —it lacks any notion of spatial or functional proximity between trajectory tokens, which should be physically continuous. Formally, let $z_{t}$ be the model’s output logit, $\hat{y}_{t}$ be the predicted token and $y_{t}^{*}$ be the ground-truth (GT) token at step $t$. $\hat{y}_{t}$ is generated using the causal context $\{\mathbf{y}^{*}_{<t},\mathcal{X}\}$. The gradient of the total loss $\mathcal{L}_{\text{SFT}}$ with respect to this single logit $z_{t}$ is:

$$
\frac{\partial\mathcal{L}_{\text{SFT}}}{\partial z_{t}}=\pi_{\theta}(y_{k}|\mathbf{y}^{*}_{<t},\mathcal{X})-\mathbb{I}(\hat{y}_{t}=y_{t}^{*}).
$$

While the penalty magnitude scales with the model’s confidence $\pi_{\theta}(y_{t}|\cdot)$, the optimization objective itself offers no smoother incentive for near-correct predictions over clearly erroneous ones. For example, compared with regression loss, when the ground truth is 31.5, it is unclear which CE loss would be higher for the discrete tokens 31.4 and 21.4. This discrete, per-token supervision encourages overconfidence in $\mathbf{y}^{*}$, collapsing the policy distribution around a single expert mode.

(2) Horizon Physical Scale Mismatch during SFT. The other bottleneck is rooted in the trajectory representation itself. Fig. 3 shows that future waypoints are predicted in an ego-centric coordinate frame, where distant horizons exhibit larger spatial variance. For example, the variance of the waypoints’ coordinates at $t=4s$ are orders of magnitude larger than that at $t=0.5s$. In result, far-horizon losses dominate $\mathcal{L}_{\text{SFT}}$, while near-horizon actions (which determine steering precision) contribute negligibly. This imbalance reduces VLA’s capability to learn behavioral diversity.

(3) Advantage Collapse in RL. When the policy $\pi_{\theta}$ collapses to a single trajectory mode, rewards become nearly identical across samples, *i.e*., $R(\mathbf{y}_{i})\approx\mu_{R}$. Consequently, $\sigma_{R}\to 0$ and thus $A_{i}\approx 0$:

$$
\lim_{\sigma_{R}\to 0}A_{i}=\frac{R(\mathbf{y}_{i})-\mu_{R}}{\sigma_{R}+\xi}\to 0,
$$

leading to vanishing gradients in Eq. LABEL:eq:grpo.

(4) Behavioral Diagnostics. To quantitatively diagnose the narrow policy phenomenon, we introduce three complementary metrics collectively referred to as Behavioral Diagnostics. Given an input scenario $\mathcal{X}$, we sample $k=8$ trajectories from the policy $\pi_{\theta}$, producing a trajectory set $\mathcal{T}=\{\tau_{1},\tau_{2},\dots,\tau_{k}\}$ over a 4-second (8-step) horizon. Let $\tau^{*}$ denote the ground-truth trajectory. The diagnostics are defined as follows:

- Diversity: Measures the spread of the policy’s exploration. We compute the mean pairwise Average Displacement Error (pADE) or Final Displacement Error (pFDE) between all sampled trajectories in $\mathcal{T}$. A lower value indicates limited behavioral diversity and thus reduced exploration capacity.
- Quality: Evaluates the best feasible outcome within the sampled set. It is measured by the minimum ADE/FDE with respect to $\tau^{*}$ across all trajectories in $\mathcal{T}$. This metric reflects whether diverse exploration still preserves optimal planning quality.
- Performance: Assesses overall driving competence using the mean PDMS [^10] score from the Navsimv1 benchmark, which integrates safety, comfort, and efficiency.

Together, these diagnostics reveal both the breadth and effectiveness of policy exploration. A well-balanced model should exhibit high Diversity@ $k$, low Quality@ $k$ (indicating at least one good sample), and high Performance@ $k$. In contrast, the collapse of Diversity@ $k$ alongside stagnant Quality@ $k$ directly signals the onset of the NP bottleneck. The experimental results are in Sec. 5.4.

## 4 Curious-VLA

The overall pipeline of Curious-VLA is shown in Fig. 2. Specifically, Curious-VLA consists of Feasible Trajectory Expansion in Imitation Learning and Diversity-Aware Reinforcement Learning.

### 4.1 Feasible Trajectory Expansion in Imitation Learning

To address the narrow policy problem rooted in IL, we design Feasible Trajectory Expansion (FTE) to balance the explore–exploit trade-off. FTE builds on standard SFT and comprises 1) Exploratory Data Expansion (DE), 2) Chain-of-Thought Data Synthesis (CoT), and 3) Step-wise Normalization (SN).

Exploratory Data Expansion. We first identify 12k challenging driving segments (multi-lane, intersection, occlusion) from the 103k NavTrain set [^10] using Qwen2.5-VL-72B filtering. Then, leveraging diffusion-based ReCogDrive [^27], we generate diverse trajectories by perturbing diffusion latents. All candidate trajectories are filtered using the PDMS scorer to ensure safety compliance. FTE expands data both within-intent (sampling around the same driving goal) and across-intent (altering route-level decisions), resulting in 142k safe and diverse samples.

Chain-of-Thought Data Synthesis. Following previous methodology [^42] [^38], we structure the driving reasoning process into a four-stage chain in single-turn dialogue: (i) critical object perception, (ii) driving explanation, (iii) meta-behavior description, and final (iv) trajectory prediction. We leverage Qwen2.5-VL-72B to automatically generate these structured reasoning sequences for our entire expanded dataset. The implementation details are available in the Supplement.

![[x4 8.png|Refer to caption]]

Figure 3: Visualization of horizon physical scale mismatch by waypoints distribution.

Step-wise Normalization. To handle the horizon-scale imbalance mentioned in Sec. 3.2 (2), we normalize each prediction step $t$ independently:

$$
\tilde{w}_{t}=\frac{w_{t}-\mu_{t}}{\sigma_{t}},\quad\hat{w}_{t}=\hat{\tilde{w}}_{t}\sigma_{t}+\mu_{t},
$$

where $(\mu_{t},\sigma_{t})$ are per-step statistics from the training set. During SFT, we use $\tilde{w}_{t}$ for training. For testing, the predicted $\hat{\tilde{w_{t}}}$ need to de-normalize to trajectory $\hat{w_{t}}$. Fig. 3 shows that this normalization equalizes gradient magnitudes across horizons, improving the separability of trajectory patterns and providing a balanced foundation for exploration.

### 4.2 Diversity-Aware Reinforcement Learning

To sustain exploration in RL, we introduce two complementary mechanisms: Adaptive Diversity-Aware Sampling (ADAS) and Spanning Driving Reward (SDR).

Input: Full dataset $\mathcal{D}_{total}$, initial policy $\pi_{0}$

Hparam: Offline rollout size $M$, group size $G$   Diversity threshold $\epsilon_{div}$, Confidence margin $\epsilon_{conf}$;

Output: Refined policy $\pi_{E}$

for *outer-loop $e=1,\dots,E$* do

    // Phase 1: Offline Filtration

    Initialize active set $\mathcal{D}_{active}\leftarrow\emptyset$;

    for *scenario $x\in\mathcal{D}_{total}$* do

       Generate $M$ offline rollouts using $\pi_{e-1}$;

       Estimate reward stats ($\mu_{R},\sigma_{R}$) and success rate $p\leftarrow\mu_{R}/R_{max}$;

       if *$p^{G}+(1-p)^{G}<\epsilon_{div}$ and $|\sigma_{R}-\sigma_{Bernoulli}|<\epsilon_{conf}$* then

          $\mathcal{D}_{active}\leftarrow\mathcal{D}_{active}\cup\{x\}$;

       end if

    end for

   // Phase 2: GRPO Training

    while *not end of epoch* do

       Sample batch of scenarios $\mathcal{B}\subset\mathcal{D}_{active}$;

       For each $x\in\mathcal{B}$, generate online group of size $G$ using $\pi_{e-1}$;

       Update $\pi_{e}$ via policy gradient (Eq. LABEL:eq:grpo);

    end while

end for

return *$\pi_{E}$*

Algorithm 1 Adaptive Diversity-Aware Sampling (ADAS)

Adaptive Diversity-Aware Sampling. ADAS dynamically selects scenarios that yield diverse rollouts under stochastic policies, maintaining sufficient reward variance for stable GRPO optimization as discussed in Sec. 3.2 (3).

We model the outcome variability of each scenario as a simplified Bernoulli process, where each rollout corresponds to a binary trial of success (high PDMS) or failure (low PDMS) with probability $p$. This approximation captures the most extreme case of reward distribution, providing a simple yet effective measure of diversity potential. At the beginning of each training outer-loop, we re-sample a new active training set from entire train data. For each training scenario $x$, we periodically perform $M$ offline rollouts ($M\gg G$) using the current policy to estimate its empirical reward distribution. The average normalized PDMS across these rollouts serves as the success probability estimate $\hat{p}$. A scenario $x$ is included in the active training set only if it satisfies two diversity-related conditions:

$$
\displaystyle\hat{p}^{G}+(1-\hat{p})^{G}<\epsilon_{\text{div}},
$$
$$
\displaystyle|\sigma_{R}-\sqrt{\hat{p}(1-\hat{p})}R_{\text{range}}|<\epsilon_{\text{conf}},
$$

where $\epsilon_{\text{div}}$, $\epsilon_{\text{conf}}$ are predefined thresholds. The first term bounds the probability that all $G$ online rollouts yield identical outcomes (either all success or all failure), ensuring sufficient variability across samples. The second term enforces consistency between the empirical standard deviation $\sigma_{R}$ and the theoretical Bernoulli variance $\sqrt{p(1-p)}R_{\text{range}}$ within a confidence margin $\epsilon_{\text{conf}}$, filtering out unstable or noisy scenarios.

Spanning Driving Reward. To further amplify the exploration signals, we redesign the reward based on the Navsim metrics PDMS and EPDMS [^10]. Each metric is computed as the product of safety constraints ($C$) and weighted objectives ($M$):

$$
\text{PDMS}=\prod_{c\in C}c\times\frac{\sum_{m\in M}w_{m}\cdot m}{\sum_{m\in M}w_{m}},
$$

where $C=\{\text{NC, DAC}\}$ (No Collisions, Drivable Area Compliance) and $M=\{\text{EP, TTC, C}\}$ (Ego Progress, Time to Collision, Comfort) with weights $w_{m}=\{5,5,2\}$. We reformulate this into a focal-style spanning objective:

$$
R_{\text{span}}=\prod_{c\in C}c\cdot\frac{\sum_{m\in M}w^{\prime}_{m}\cdot(1-(1-m)^{\gamma_{m}})}{\sum_{m\in M}w^{\prime}_{m}},
$$

where $\gamma_{m}$ is the hyperparameter. The EPDMS [^10] [^3] reuses the same calculation structure and extends $C$ with $\{\text{DDC, TLC}\}$ (Driving Direction, Traffic Light Compliance), extends $M$ with $\{\text{LK, EC}\}$ (Lane Keeping, two-frame Extended Comfort), with the extra weights $\{2,2\}$. This focal-style design magnifies differences between suboptimal and optimal behaviors, improving the reward’s sensitivity to the driving quality.

## 5 Experiment

![[x5 8.png|Refer to caption]]

Figure 4: Qualitative comparison with BEV and Camera. We can see that our Curious-VLA achieves more feasible trajectories.

Table 1: Comparison of different methods on Navsim V1 benchmark. The model with <sup>†</sup> indicates evaluation using Best-of-N sampling. The best and second best results are bolded and underlined, respectively. Continuous regresses the trajectory with an additional MLP/Transformer head. Discrete Action decodes action tokens that discretize each dimension of the trajectory into several bins. Text Waypoint decodes textual numbers that represent the trajectory.

<table><tbody><tr><th></th><td>Base Model</td><td>Sensors</td><td>Trajectory</td><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>C <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><th colspan="9">Human GT</th><td></td></tr><tr><th>Human <sup><a href="#fn:10">10</a></sup></th><td>-</td><td>-</td><td>-</td><td>100.0</td><td>100.0</td><td>87.5</td><td>100.0</td><td>99.9</td><td>94.8</td></tr><tr><th colspan="9">Classical E2E</th><td></td></tr><tr><th>Ego-MLP <sup><a href="#fn:29">29</a></sup></th><td>-</td><td>6x C</td><td>Continuous</td><td>93.1</td><td>78.3</td><td>63.2</td><td>84.0</td><td>100.0</td><td>66.4</td></tr><tr><th>UniAD <sup><a href="#fn:16">16</a></sup></th><td>-</td><td>3x C + L</td><td>Continuous</td><td>97.7</td><td>92.8</td><td>79.2</td><td>92.8</td><td>100.0</td><td>84.0</td></tr><tr><th>DiffusionDrive <sup><a href="#fn:30">30</a></sup></th><td>-</td><td>3x C + L</td><td>Continuous</td><td>98.2</td><td>96.2</td><td>82.2</td><td>94.7</td><td>100.0</td><td>88.1</td></tr><tr><th>WoTE <sup><a href="#fn:26">26</a></sup></th><td>-</td><td>3x C + L</td><td>Continuous</td><td>98.5</td><td>96.8</td><td>81.9</td><td>94.4</td><td>99.9</td><td>88.3</td></tr><tr><th colspan="9">VLA(VLA-Planner)</th><td></td></tr><tr><th>ImagiDrive-A <sup><a href="#fn:23">23</a></sup></th><td>InternVL2.5-4B</td><td>1x C</td><td>Continuous</td><td>98.1</td><td>96.2</td><td>80.1</td><td>94.4</td><td>100.0</td><td>86.9</td></tr><tr><th>ImagiDrive-S</th><td>InternVL2.5-4B</td><td>1x C</td><td>Continuous</td><td>98.6</td><td>96.2</td><td>80.5</td><td>94.5</td><td>100.0</td><td>87.4</td></tr><tr><th>ReCogDrive <sup><a href="#fn:27">27</a></sup></th><td>InternVL2-8B</td><td>3x C</td><td>Continuous</td><td>98.2</td><td>97.8</td><td>83.5</td><td>95.2</td><td>100.0</td><td>89.6</td></tr><tr><th>DriveVLA-W0 <sup><a href="#fn:25">25</a></sup></th><td>Emu-3-8B</td><td>1x C</td><td>Continuous</td><td>98.7</td><td>99.1</td><td>93.3</td><td>95.3</td><td>99.3</td><td>90.2</td></tr><tr><th>DriveVLA-W0 <sup>†</sup></th><td>Emu-3-8B</td><td>1x C</td><td>Continuous</td><td>99.3</td><td>97.4</td><td>88.3</td><td>97.0</td><td>99.9</td><td>93.0</td></tr><tr><th colspan="9">VLA(VLA-Token)</th><td></td></tr><tr><th>Qwen2.5-VL <sup><a href="#fn:27">27</a></sup></th><td>Qwen2.5-VL-7B</td><td>1x C</td><td>Text Waypoint</td><td>97.8</td><td>92.1</td><td>78.3</td><td>92.8</td><td>100.0</td><td>83.3</td></tr><tr><th>InternVL2 <sup><a href="#fn:27">27</a></sup></th><td>InternVL2-8B</td><td>1x C</td><td>Text Waypoint</td><td>97.0</td><td>92.4</td><td>78.9</td><td>91.8</td><td>100.0</td><td>83.3</td></tr><tr><th>AutoVLA <sup><a href="#fn:54">54</a></sup></th><td>Qwen2.5-VL-3B</td><td>3x C</td><td>Discrete Action</td><td>98.4</td><td>95.6</td><td>81.9</td><td>98.0</td><td>99.9</td><td>89.1</td></tr><tr><th>AutoVLA <sup>†</sup></th><td>Qwen2.5-VL-3B</td><td>3x C</td><td>Discrete Action</td><td>99.1</td><td>98.8</td><td>87.9</td><td>97.2</td><td>100.0</td><td>92.1</td></tr><tr><th>AdaThinkDrive <sup><a href="#fn:33">33</a></sup></th><td>InternVL3-8B</td><td>1x C</td><td>TextWaypoint</td><td>98.4</td><td>97.8</td><td>84.4</td><td>95.2</td><td>100.0</td><td>90.3</td></tr><tr><th>AdaThinkDrive <sup>†</sup></th><td>InternVL3-8B</td><td>1x C</td><td>Text Waypoint</td><td>99.1</td><td>98.8</td><td>87.9</td><td>95.2</td><td>100.0</td><td>93.0</td></tr><tr><th>Curious-VLA</th><td>Qwen2.5-VL-3B</td><td>1x C</td><td>Text Waypoint</td><td>98.4</td><td>96.9</td><td>88.5</td><td>97.9</td><td>98.1</td><td>90.3</td></tr><tr><th>Curious-VLA <sup>†</sup></th><td>Qwen2.5-VL-3B</td><td>1x C</td><td>Text Waypoint</td><td>99.5</td><td>99.0</td><td>91.8</td><td>99.3</td><td>98.4</td><td>94.8</td></tr></tbody></table>

Table 2: Comparison of different methods with the Extended PDMS metric of NAVISM V2.

| Method | NC $\uparrow$ | DAC $\uparrow$ | DDC $\uparrow$ | TLC $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | LK $\uparrow$ | C $\uparrow$ | EC $\uparrow$ | EPDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ego-MLP [^29] | 93.1 | 77.9 | 92.7 | 99.6 | 86.0 | 91.5 | 89.4 | 98.3 | 85.4 | 64.0 |
| VADv2 [^20] | 97.3 | 91.7 | 98.2 | 99.9 | 77.6 | 92.7 | 66.0 | 100.0 | 97.4 | 76.6 |
| TransFuser [^5] | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 |
| HydraMDP++ [^28] | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | 81.4 |
| DriveSuprim [^47] | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 | 83.1 |
| ARTEMIS [^12] | 98.3 | 95.1 | 98.6 | 99.8 | 81.5 | 97.4 | 96.5 | 98.3 | – | 83.1 |
| ReCogDrive [^27] | 98.3 | 95.2 | 99.5 | 99.8 | 87.1 | 97.5 | 96.6 | 98.3 | 86.5 | 83.6 |
| DiffusionDrive [^30] | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | 84.5 |
| Curious-VLA | 98.4 | 96.9 | 99.2 | 99.8 | 88.5 | 97.9 | 96.9 | 98.1 | 81.5 | 85.3 |

Table 3: Open-loop evaluation on the nuScenes Benchmark.

<table><thead><tr><th rowspan="2">Method</th><th colspan="2">ST-P3 metrics</th><th colspan="2">UniAD metrics</th></tr><tr><th>L2 (m) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th><th>Collision (%) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th><th>L2 (m) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th><th>Collision (%) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th></tr></thead><tbody><tr><th>ST-P3 <sup><a href="#fn:15">15</a></sup></th><td>2.11</td><td>0.71</td><td>–</td><td>–</td></tr><tr><th>VAD <sup><a href="#fn:20">20</a></sup></th><td>0.37</td><td>0.14</td><td>–</td><td>–</td></tr><tr><th>UniAD <sup><a href="#fn:16">16</a></sup></th><td>0.69</td><td>0.12</td><td>1.03</td><td>0.31</td></tr><tr><th>EMMA <sup><a href="#fn:18">18</a></sup></th><td>0.32</td><td>–</td><td>–</td><td>–</td></tr><tr><th>OpenEMMA <sup><a href="#fn:45">45</a></sup></th><td>2.81</td><td>–</td><td>–</td><td>–</td></tr><tr><th>OpenDriveVLA <sup><a href="#fn:53">53</a></sup></th><td>0.33</td><td>0.10</td><td>0.67</td><td>0.30</td></tr><tr><th>AutoVLA <sup><a href="#fn:54">54</a></sup></th><td>0.48</td><td>0.13</td><td>0.86</td><td>0.35</td></tr><tr><th>Impromptu VLA <sup><a href="#fn:4">4</a></sup></th><td>0.33</td><td>0.13</td><td>0.67</td><td>0.38</td></tr><tr><th>Curious-VLA</th><td>0.31</td><td>0.10</td><td>0.60</td><td>0.33</td></tr></tbody></table>

### 5.1 Datasets and Evaluation Metrics

Navsim (v1/v2). We conduct experiments on Navsim-v1 [^10] and v2 [^3], which are built on OpenScene [^46] [^7] [^43] (a redistribution of nuPlan [^2]) for large-scale, non-reactive simulation. Agent inputs include 8 surround-view cameras and 5 LiDAR sensors, optionally with up to 3 historical frames (1.5s @ 2Hz). We build our training set from the only CAMERA-FRONT view of the official navtrain split ($\sim$ 103k samples), following the procedure described in Sec. 4.1. For evaluation, we report the official PDMS (for v1) and EPDMS (for v2) scores, which are formally described in Sec. 4.2.

NuScenes. To verify real-world generalization, we conduct supplementary experiments on nuScenes. This dataset contains 1000 20s scenes ($\sim$ 1.4M camera images and $\sim$ 390k LiDAR scans). Following the evaluation protocols of UniAD [^16] and ST-P3 [^15], we report L2 distance error (L2) and Collision rate for direct comparison.

### 5.2 Implement Details

Curious-VLA is a pure VLM that directly auto-regresses text waypoint trajectories without any additional planner module, initialized from Qwen2.5-VL-3B [^1]. Our two-stage training pipeline consists of: 1) An Imitation Learning (by SFT) stage using LLaMA-Factory [^52], and 2) A Reinforcement Learning stage adapted from [^41] and [^51].All experiments are conducted on 8 NVIDIA H100 GPUs using DeepSpeed ZeRO-1 [^36]. For SFT, we train for 6 total epochs with a global batch size of 128. For RL, we train for 130 steps in total with 3 outer-loop for ADAS. The count of rollout is 8 and the actor global batch size is 256. More details can be found in the supplementary materials.

### 5.3 Main Results

Open-loop Results on Navsim v1 Benchmark. As shown in Tab. 1, Curious-VLA reaches a PDMS of 90.3, establishing a new SOTA record under the single-front-camera input on Navsim v1. It not only far exceeds traditional E2E methods using Camera+LiDAR inputs (e.g., WoTE [^26] at 88.3) but also demonstrates remarkable efficiency among VLM-based approaches. Notably, despite using a smaller (3B) and earlier (Qwen2.5-VL) base model, Curious-VLA’s performance is comparable to the VLA-Planner method (DriveVLA-W0 [^25] at 90.2), and the VLA-Token method (AdaThinkDrive [^33], 90.3). Compared to AutoVLA [^54] which uses the same base model, our method achieves a significant 1.2 PDMS improvement (90.3 vs. 89.1). Although the Comfort (C) metric is slightly reduced, the synchronous enhancement in both Ego Progress (EP) and Time to Collision (TTC) is strong evidence of our success in overcoming key performance obstacles.

More importantly, following the AutoVLA Best-of-N setting (N=6), Curious-VLA <sup>†</sup> achieves a PDMS score of 94.8. This result surpasses the AdaThinkDrive <sup>†</sup> (93.0) by 1.8 PDMS and matches the Human GT level. This exceptional performance stems from our model’s ability to predict trajectories with both high diversity and high quality, enabling it to make diverse yet correct decisions in complex scenarios. It effectively validates that our unleashing exploration mechanism has successfully alleviated the narrow policy problem. More visualization in 4.

Open-loop Results on Navsim v2 Benchmark. We further evaluated Curious-VLA on the more challenging Navsim v2, using the official navtest split. As shown in Tab. 2, our method achieves an EPDMS composite score of 85.3, establishing a new SOTA. This represents a +0.8 improvement over the DiffusionDrive (84.5). The score is supported by our model’s robust performance across several critical sub-metrics, particularly in Drivable Area Compliance (DAC), Ego Progress (EP), and Time to Collision (TTC).

Open-loop Results on Nuscenes Benchmark. We applied our training pipeline to a 28k Nuscenes training subset [^4], using an ADE-based reward [^54] for the RL stage. As shown in Tab. 3, Curious-VLA also achieves excellent results on L2 error and Collision rate in 3s, outperforming existing VLAs and E2E models.

### 5.4 Analytical Experiments of Exploration.

To analyze how our method progressively addresses the narrow policy problem, we use the Behavioral Diagnostics evaluated on the navtrain subset. In in Tab. 4. Both the Qwen2.5-VL baseline and ReCogDrive suffer from severely limited diversity. While adding expanded CoT data alone (+ FTE, w/o SN) degrades trajectory quality, introducing Step-wise Normalization (+ FTE) significantly boosts diversity while maintaining high performance (91.31 mean-PDMS). This shows that SN is the key to effectively learning from diverse trajectories. Finally, the RL stage (+ FTE + RL) further pushes both diversity and quality to the best levels (1.415 mean-pFDE and 0.547 minFDE).

Table 4: Exploration analysis (all metrics @k=8). We evaluate Quality (minADE/FDE $\downarrow$), Diversity (mean-pADE/FDE $\uparrow$), and Perf. (mean-PDMS $\uparrow$). In the Quality and Diversity columns, values correspond to ADE / FDE.

| Method | Stage | Quality | Diversity | Perf. |
| --- | --- | --- | --- | --- |
| ReCogDrive | IL+RL | 0.295 / 0.621 | 0.148 / 0.325 | 90.95 |
| Qwen2.5-VL | IL | 0.481 / 1.052 | 0.090 / 0.200 | 90.69 |
| \+ FTE (w/o SN) | IL | 0.513 / 1.129 | 0.170 / 0.381 | 90.65 |
| \+ FTE | IL | 0.480 / 1.078 | 0.346 / 0.803 | 91.31 |
| \+ FTE + RL | IL+RL | 0.269 / 0.547 | 0.641 / 1.415 | 91.55 |

### 5.5 Ablation Study

Ablation on Feasible Trajectory Expansion. We perform an IL-stage ablation to assess the contribution of our Feasible Trajectory Expansion in Tab. 5. Following standard RL practice that initializes from high-quality reasoning models (*e.g*., [^35] [^14] [^54]), we first build a strong baseline by adding CoT supervision. This yields a performance gain of $\mathbf{+1.7}$ PDMS and serves as the basis for subsequent ablations.

FTE comprises two other training designs: Exploratory Data Expansion (DE) and Step-wise Normalization (SN). The core challenge is to leverage DE effectively: adding DE alone (without SN) yields ${85.2}$ PDMS, which is worse than the CoT baseline ($85.6$), indicating that naive data expansion is difficult to scale. SN serves as the necessary catalyst—combining DE with SN produces the best SFT policy ($\mathbf{87.6}$ PDMS). Combined with Tab. 4, these results show that SN successfully converts diverse exploratory examples into actionable knowledge and provides an optimal initialization for downstream RL.

Ablation on Diversity-Aware RL. We ablate our RL-stage training designs in Tab. 6. The key challenge is filtering data to ensure GRPO receives non-zero advantages and meaningful gradients. We found that strategies inspired by difficulty-aware sampling (*e.g*., [^49]), such as our Human Difficulty implementation, consistently lead to training collapse (35.2 PDMS), similar to Random Sample and Full Trainset. This suggests that avoiding zero-advantage scenarios is critical. We then explored filtering based on reward diversity, inspired by the “medium-difficulty samples” [^50]. A simple heuristic Reject Unimodal Distribution Strategy successfully avoids collapse and achieves 88.8 PDMS. This confirms that filtering for reward diversity is a viable direction. Our proposed ADAS, which statistically validates this diversity via a Bernoulli test (see Sec. 4.2), performs even better (89.6 PDMS). Finally, combining the full ADAS (with 3 outer-loops) and Spanning Driving Reward (SDR) achieves the optimal 90.3 PDMS.

Table 5: Ablation study on Imitation Learning approaches. We evaluate the impact of three key Feasible Trajectory Expansion components: DE (Exploratory Data Expansion),CoT (Chain-of-Thought) and SN (Step-wise Normalization).

| DE | CoT | SN | NC $\uparrow$ | DAC $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | C $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $\times$ | $\times$ | $\times$ | 97.7 | 91.8 | 85.8 | 96.8 | 98.4 | 83.9 |
| $\times$ | ✓ | $\times$ | 98.2 | 93.2 | 85.8 | 97.3 | 98.4 | 85.6 |
| ✓ | ✓ | $\times$ | 98.0 | 93.0 | 85.9 | 97.2 | 98.4 | 85.2 |
| $\times$ | ✓ | ✓ | 98.2 | 94.3 | 86.7 | 97.3 | 98.4 | 86.9 |
| ✓ | ✓ | ✓ | 98.3 | 95.1 | 86.5 | 97.6 | 98.3 | 87.6 |

Table 6: Ablation study on RL approaches. We evaluate the effect of different Sampling strategies and the SDR (Spanning Driving Reward) on Navsim v1. The $i$ x denotes a total of $i$ outer-loop iterations of ADAS.

| Sampling Strategy | SDR | NC $\uparrow$ | DAC $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | C $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Human Difficulty | $\times$ | 73.9 | 43.7 | 94.9 | 70.3 | 97.1 | 35.2 |
| Reject Unimodal | $\times$ | 98.4 | 96.0 | 87.0 | 97.8 | 98.4 | 88.8 |
| ADAS(1x) | $\times$ | 98.2 | 96.3 | 88.6 | 97.6 | 98.2 | 89.6 |
| ADAS(3x) | $\times$ | 98.4 | 96.8 | 88.5 | 97.8 | 98.1 | 90.1 |
| ADAS(3x) | $\checkmark$ | 98.4 | 96.9 | 88.5 | 97.9 | 98.1 | 90.3 |

## 6 Conclusion

The “exploit-explore” dilemma represents a fundamental and persistent challenge in end-to-end autonomous driving systems. For VLA, although imitation learning (IL) provides a robust foundation by leveraging high-value ground-truth data that encapsulate the most probable driving behaviors, it still suffers from limited behavioral diversity due to data scarcity. Reinforcement Learning (RL) by itself, as the subsequent process of IL, is insufficient to solve the dilemma, particularly well-suited for critic-free RL algorithms (*e.g*. GRPO) in VLMs. Only through significantly enhanced diversity in both data and policy representations can we achieve more comprehensive and reliable planning capabilities in open-world environments.

In this paper, we first reveal and analyze the largely neglected Narrow Policy in Drive VLA due to the exploit-explore imbalance. Accordingly, we introduce Curious-VLA, a systematic framework regarding data, sampling, and reward during both IL and RL and therefore balance exploitation and exploration, paving the way for more capable and reliable autonomous driving systems.

## Acknowledgements

This research was supported by the National Natural Science Foundation of China (Grant No. 62550184), Xiongan AI Institute, Lenovo Research and Wuxi Research Institute of Applied Technologies, Tsinghua University under Grant 20242001120.

## References

Supplementary Material  

## Appendix A Training Implement Details

we provide more detailed configurations for both the Imitation Learning (SFT) and Reinforcement Learning (RL) stages.

### A.1 Training Stages on SFT

Since the base Qwen2.5-VL model does not inherently support the specific <think> token, we introduce external tokens (<thinking></thinking> to wrap the driving explanation part, <answer></answer> to wrap the trajectory prediction part) to the tokenizer. To ensure the model effectively learns this structured reasoning format without compromising its visual encoding capabilities, we adopt a two-stage SFT strategy:

- Thinking Alignment. In this stage, we freeze the vision encoder and the projector while optimizing only the LLM backbone. It primarily focuses on aligning the model with the structured CoT format and external tokens, ensuring the LLM captures the syntactic structure of the thinking process without disturbing the pretrained visual representations.
- End-to-End Fine-tuning. In the second stage, we unfreeze all parameters to end-to-end fine-tuning. This step enables joint optimization of the vision encoder and LLM, effectively bridging the vision-language space while adapting the pretrained knowledge to the domain shift of autonomous driving.

### A.2 Hyper-Parameters

We provide the detailed hyper-parameters for SFT in Table 7 and GRPO in Table 8.

Table 7: Training configurations for SFT.

| Setting | SFT |
| --- | --- |
| Base Model | Qwen2.5-VL-3B |
| Max Pixels | 262144 |
| Global Batch Size | 128 |
| Epochs | 3(align) / 3(fine-tune) |
| Trainset Samples | 103k(navtrain) + 39k(DE) |
| Learning Rate | 4e-5 / 5e-6 |
| Weight Decay | 0.05 |
| Warmup Ratio | 0.10 |
| bfloat16 | ✓ |
| Global Batch Size | 128 |
| GPUs | 8 |

Table 8: Training configurations for GRPO.

| Setting | GRPO |
| --- | --- |
| Max Pixels | 262144 |
| Rollout Batch Size | 256 |
| Actor Global Batch Size | 256 |
| N Rollout(Group Size) | 8 |
| Outer Loops | 3 |
| Total Steps | 130 |
| Active Samples | 6k/3k/1k (for 3 outer-loop) |
| bfloat16 | ✓ |
| GPUs | 8 |

### A.3 Details on Span Driving Reward

In Sec. 1, we redesign the reward function by adapting the EPDMS into an additive focal objective with strict safety constraints. Furthermore, we remove EC for training efficiency, and the sparse reward $R_{sparse}$ defined by Eq. 10 uses $C^{\prime}=\{\text{NC, DAC, DDC, TLC}\}$, and $M^{\prime}=\{\text{EP, TTC, C, LK}\}$, the weights are $w^{\prime}_{m}=\{5,5,2,2\}$ and focal exponents are $\gamma_{m}=\{0.5,0.5,1.0,1.0\}$.

## Appendix B Data Pipeline

### B.1 Details on Exploratory Data Expansion

Semantic-Aware Challenging Sceneario Filtering. We first identify 12k challenging driving segments (e.g., multi-lane roads, intersections, occlusions) from the 103k navtrain split. Although visual grounding models like Grounding-DINO can detect road elements, and the Navsim dataset itself provides rich SemanticMapLayers annotations, these sources cannot directly filter for challenging scenes that semantically possess “multiple feasible trajectories.” Therefore, we employ the following prompt with Qwen2.5-VL-72B to screen for these scenarios for data expansion. As shown in the prompt in Fig. 5, the VLM is instructed to focus on complex road elements and identify scenarios that allow for diverse maneuvers.

<svg id="A2.F5.pic1" class="ltx_picture ltx_centering" height="2330.43" overflow="visible" version="1.1" viewBox="0 0 600 2330.43" width="600"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="translate(0,2330.43) matrix(1 0 0 -1 0 0)" fill="#000000" stroke="#000000" stroke-width="0.4pt"><g style="--ltx-fill-color:#404040;" fill="#404040" fill-opacity="1.0"><path style="stroke:none" d="M 0 5.91 L 0 2324.52 C 0 2327.78 2.64 2330.43 5.91 2330.43 L 594.09 2330.43 C 597.36 2330.43 600 2327.78 600 2324.52 L 600 5.91 C 600 2.64 597.36 0 594.09 0 L 5.91 0 C 2.64 0 0 2.64 0 5.91 Z"></path></g><g style="--ltx-fill-color:#F2F2F2;" fill="#F2F2F2" fill-opacity="1.0"><path style="stroke:none" d="M 1.97 5.91 L 1.97 1324.73 L 598.03 1324.73 L 598.03 5.91 C 598.03 3.73 596.27 1.97 594.09 1.97 L 5.91 1.97 C 3.73 1.97 1.97 3.73 1.97 5.91 Z"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 2310.85)"><foreignObject style="--ltx-fg-color:#FFFFFF;--ltx-fo-width:27.94em;--ltx-fo-height:0.69em;--ltx-fo-depth:49.19em;" width="556.69" height="993.88" transform="matrix(1 0 0 -1 0 13.67)" overflow="visible" color="#FFFFFF">Prompt and Examples for Sceneario Filtering </foreignObject></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 126.97)"><foreignObject style="--ltx-fg-color:#000000;--ltx-fo-width:27.94em;--ltx-fo-height:59.52em;--ltx-fo-depth:5.68em;" width="556.69" height="1299.14" transform="matrix(1 0 0 -1 0 1185.96)" overflow="visible" color="#000000">Question: &lt;image&gt; Give you a scene during driving. Observe the provided image to identify challenging scenarios, such as intersections, multi-lane roads, and occlusions. Pay strict attention to the ego vehicle’s current lane, ground markings, and traffic signs. Provide your analysis strictly in the format below. Do not add any extra explanations. Always make conservative decisions. Unless the non-lane-keeping driving intention is very safe and necessary, only stay in the lane. Analysis Guidelines • An ”intent” is an immediately executable and traffic-compliant high-level maneuver (e.g., turn left, change lane). • Do not list intents that are clearly unreasonable or illegal (e.g., turning left from a straight-only lane, or changing lanes over a solid line). Output Format 1. SCENE_SUMMARY: [Provide a brief (200-word max) description of the ego vehicle’s situation.] 2. ALL_INTENTS: [List all plausible driving intents, separated by commas.] 3. NON_LANE_KEEPING_INTENT: [Based on the list in #2, is there any valid intent other than lane following? Answer ”Yes” or ”No”].<br>Answer Example 1: 1. SCENE_SUMMARY: The ego vehicle is on a two-lane road with a clear view ahead. The road is relatively empty, with no immediate traffic or pedestrians in sight. On the left side, there are trees, a sidewalk, and streetlights, while the right side has some vegetation and a N̈o Parkings̈ign. The road markings include a double yellow line in the center and a single yellow line on the right edge. The sky is partly cloudy, and a bridge is visible in the distance. The overall environment appears calm and safe for driving. There are no immediate obstacles or hazards on the road. 2. ALL_INTENTS: continue straight 3. NON_LANE_KEEPING_INTENT: No.<br>Answer Example 2: 1. SCENE_SUMMARY: The ego vehicle is driving on a busy urban street, likely in a commercial area with multiple lanes and traffic signals. The vehicle is positioned in the right lane, following a red pickup truck. To the left, there is an orange van with advertisements, and to the right, there are various commercial buildings, including a Starbucks and a Planet 13 dispensary. The traffic signal ahead is red, indicating that vehicles must stop. The road markings are clear, with solid and dashed lines indicating lane boundaries. There are no immediate obstacles or pedestrians in the immediate vicinity of the ego vehicle. The overall environment suggests a typical city street with moderate traffic. 2. ALL_INTENTS: stop at the red light, change lane left, change lane right. 3. NON_LANE_KEEPING_INTENT: Yes</foreignObject></g></g></svg>

Figure 5: The prompt template and representative answer examples used to filter semantic-aware challenging scenarios with multiple feasible driving intents.

Generative Trajectory Expansion. To construct a robust exploratory dataset, we implement a rigorous expansion pipeline using the DDIM planner from ReCogDrive. To induce behavioral variance beyond deterministic outcomes, we modify the standard DDIM sampling by scaling the standard deviation of the Gaussian noise injected during the reverse denoising steps. The expansion process includes:

1. Hybrid Sampling Strategy: We apply distinct sampling methods based on scene complexity. For the entire navtrain dataset, we perform intra-intent ($k=32$) inference using the original prompts to explore execution variations within the same intent. For the 12k filtered challenging scenes, we additionally execute inter-intent inference by systematically altering the intention prompts (switching among Go Straight, Turn Left, Turn Right, and Unknown) to uncover plausible intents.
2. Safety and Diversity Filtering: All generated candidates undergo a dual-criteria filter. For safety, a trajectory is retained only if its PDMS score exceeds $95.0$ and not less than the score of the human GT. For diversity, we employ a greedy selection based on geometric distance. We remove trajectories near the human GT and iteratively retain candidates, pruning those within a predefined distance margin.

This pipeline ultimately yields 39k non-ground-truth yet physically feasible exploratory samples, significantly broadening the policy’s behavioral coverage beyond the narrow human ground truth.

### B.2 Details on Chain-of-Thought

Following Poutine [^38], we adopt a structured reasoning approach to explicitly model the decision-making process. The model processes the input through input context and four thinking tasks:

Input Context. The input $\mathcal{X}$ comprises a single CAMERA-FRONT image, the ego-vehicle’s kinematic history (past 1.5s trajectory, current velocity and acceleration), and the high-level driving intent.

Thinking Tasks. The CoT follows this sequence:

1. Critical Object Perception: Identify the presence of critical objects [^38] that might influence the future path.
2. Driving Explanation: Generate a concise, natural-language rationale (approximate 100 words).
3. Meta-Behavior Description: Classify the intended behavior into discrete categories [^38], specifically selecting the appropriate Speed and Command.
4. Trajectory Prediction: Predict the optimal 4-second trajectory waypoints (normalized) conditioned on the previous steps.

.

## Appendix C Inference Efficiency

We evaluate real-time efficiency under serial inputs. As shown in Tab. 9, Curious-VLA achieves 1.57s per sample, which is 7.74s faster than AutoVLA’s text waypoint mode and competitive with its optimized Action+RFT mode (1.31s). The efficiency gain primarily benefits from our concise output template and single-view (1x C) input.

Table 9: Inference Latency Comparison. Text: Text waypoint; Action: Action token. AutoVLA uses a Fast-Slow Dual-System.

| Method | Setting | Latency (s) |
| --- | --- | --- |
| AutoVLA | Dual-Sys (Text) | 9.31 |
| AutoVLA | Dual-Sys (Action) | 3.95 |
| AutoVLA | Dual-Sys (Action + RFT) | 1.31 |
| Curious-VLA | Slow Think only (Text) | 1.57 |

## Appendix D RL Training Stability

We provide the RL training curves to demonstrate the stability of our training pipeline. As shown in Fig. 6, we report Val Reward and Test PDMS over the entire 130 training steps (ADAS 3x, 3 outer-loops as described in Alg. 1). Outer-loop transitions are determined by Val Reward trends. In contrast, the Random Sample baseline (without ADAS) shows RL collapse, confirming the necessity of diversity-aware sampling.

We further analyze training stability across multiple runs in Fig. 7. We perform $k=4$ independent training runs (ADAS 1x). The Critic (on trainset) and Val Reward curves demonstrate consistent improvement with low variance across all trials.

## Appendix E External Analytical Experiments

We further extend the analysis to DiffusionDrive [^30] in Tab. 10. Despite its diverse 20-candidate pool(@all), the final Top-1 confidence-selected trajectory(@1) collapses to a single mode with the lowest diversity (0.037 / 0.076 mean-pADE/FDE), confirming that the narrow policy problem persists even in diffusion-based planners. In contrast, Curious-VLA achieves a superior balance across Quality, Diversity, and Performance.

Table 10: Extended exploration analysis. DiffusionDrive is evaluated with its 20 denoised candidates(@all) and the confidence Top-1 selection(@1). All metrics @k=8.

| Method | Output | Quality | Diversity | Perf. |
| --- | --- | --- | --- | --- |
| DiffusionDrive | Diff.@all | 0.218 / 0.430 | 0.571 / 1.175 | 87.60 |
| DiffusionDrive | Diff.@1 | 0.350 / 0.720 | 0.037 / 0.076 | 88.10 |
| Curious-VLA | AR@1 | 0.269 / 0.547 | 0.641 / 1.415 | 91.55 |

![[x6 7.png|Refer to caption]]

Figure 6: RL Training Curves. Val Reward and Test PDMS over 130 steps (ADAS 3x). The Random Sample baseline shows RL collapse.

![[x7 7.png|Refer to caption]]

Figure 7: Stability Analysis. k = 4 k=4 training runs (ADAS 1x). Critic and Val Reward curves show consistent improvement with low variance.

## Appendix F More Visualization of Curious-VLA

As shown in Fig. 8, Curious-VLA successfully alleviates the narrow policy bottleneck, ensuring diverse feasible behaviors.

![[sv1.png|Refer to caption]]

Figure 8: More visualization between Curious-VLA(top) and Qwen2.5-VL(bottom).

[^1]: S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang, et al. (2025) Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923. Cited by: §5.2.

[^2]: H. Caesar, J. Kabzan, K. S. Tan, W. K. Fong, E. Wolff, A. Lang, L. Fletcher, O. Beijbom, and S. Omari (2021) NuPlan: a closed-loop ml-based planning benchmark for autonomous vehicles. In CVPR ADP3 workshop, Cited by: §5.1.

[^3]: W. Cao, M. Hallgarten, T. Li, D. Dauner, X. Gu, C. Wang, Y. Miron, M. Aiello, H. Li, I. Gilitschenski, et al. (2025) Pseudo-simulation for autonomous driving. arXiv preprint arXiv:2506.04218. Cited by: §4.2, §5.1.

[^4]: H. Chi, H. Gao, Z. Liu, J. Liu, C. Liu, J. Li, K. Yang, Y. Yu, Z. Wang, W. Li, et al. (2025) Impromptu vla: open weights and open data for driving vision-language-action models. arXiv preprint arXiv:2505.23757. Cited by: §2, §5.3, Table 3.

[^5]: K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger (2022) Transfuser: imitation with transformer-based sensor fusion for autonomous driving. IEEE transactions on pattern analysis and machine intelligence 45 (11), pp. 12878–12895. Cited by: §2, Table 2.

[^6]: P. F. Christiano, J. Leike, T. Brown, M. Martic, S. Legg, and D. Amodei (2017) Deep reinforcement learning from human preferences. Advances in neural information processing systems 30. Cited by: §2.

[^7]: O. Contributors (2023) OpenScene: the largest up-to-date 3d occupancy prediction benchmark in autonomous driving. Note: [https://github.com/OpenDriveLab/OpenScene](https://github.com/OpenDriveLab/OpenScene) Cited by: §5.1.

[^8]: G. Cui, Y. Zhang, J. Chen, L. Yuan, Z. Wang, Y. Zuo, H. Li, Y. Fan, H. Chen, W. Chen, et al. (2025) The entropy mechanism of reinforcement learning for reasoning language models. arXiv preprint arXiv:2505.22617. Cited by: §1, §2.

[^9]: R. Dai, L. Song, H. Liu, Z. Liang, D. Yu, H. Mi, Z. Tu, R. Liu, T. Zheng, H. Zhu, et al. (2025) Cde: curiosity-driven exploration for efficient reinforcement learning in large language models. arXiv preprint arXiv:2509.09675. Cited by: §2.

[^10]: D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, et al. (2024) Navsim: data-driven non-reactive autonomous vehicle simulation and benchmarking. Advances in Neural Information Processing Systems 37, pp. 28706–28719. Cited by: §1, 3rd item, §4.1, §4.2, §4.2, §5.1, Table 1.

[^11]: G. Dong, H. Mao, K. Ma, L. Bao, Y. Chen, Z. Wang, Z. Chen, J. Du, H. Wang, F. Zhang, et al. (2025) Agentic reinforced policy optimization. arXiv preprint arXiv:2507.19849. Cited by: §2.

[^12]: R. Feng, N. Xi, D. Chu, R. Wang, Z. Deng, A. Wang, L. Lu, J. Wang, and Y. Huang (2025) ARTEMIS: autoregressive end-to-end trajectory planning with mixture of experts for autonomous driving. External Links: 2504.19580, [Link](https://arxiv.org/abs/2504.19580) Cited by: Table 2.

[^13]: H. Fu, D. Zhang, Z. Zhao, J. Cui, D. Liang, C. Zhang, D. Zhang, H. Xie, B. Wang, and X. Bai (2025) Orion: a holistic end-to-end autonomous driving framework by vision-language instructed action generation. arXiv preprint arXiv:2503.19755. Cited by: §1, §2.

[^14]: D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang, X. Bi, et al. (2025) Deepseek-r1: incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948. Cited by: §1, §2, §5.5.

[^15]: S. Hu, L. Chen, P. Wu, H. Li, J. Yan, and D. Tao (2022) St-p3: end-to-end vision-based autonomous driving via spatial-temporal feature learning. In European Conference on Computer Vision, pp. 533–549. Cited by: §5.1, Table 3.

[^16]: Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, et al. (2023) Planning-oriented autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 17853–17862. Cited by: §2, §3.1, §5.1, Table 1, Table 3.

[^17]: X. Huang, E. M. Wolff, P. Vernaza, T. Phan-Minh, H. Chen, D. S. Hayden, M. Edmonds, B. Pierce, X. Chen, P. E. Jacob, et al. (2024) Drivegpt: scaling autoregressive behavior models for driving. arXiv preprint arXiv:2412.14415. Cited by: §1, §2.

[^18]: J. Hwang, R. Xu, H. Lin, W. Hung, J. Ji, K. Choi, D. Huang, T. He, P. Covington, B. Sapp, et al. (2024) Emma: end-to-end multimodal model for autonomous driving. arXiv preprint arXiv:2410.23262. Cited by: §2, Table 3.

[^19]: X. Jia, J. You, Z. Zhang, and J. Yan (2025) Drivetransformer: unified transformer for scalable end-to-end autonomous driving. arXiv preprint arXiv:2503.07656. Cited by: §2.

[^20]: B. Jiang, S. Chen, Q. Xu, B. Liao, J. Chen, H. Zhou, Q. Zhang, W. Liu, C. Huang, and X. Wang (2023) Vad: vectorized scene representation for efficient autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 8340–8350. Cited by: §2, Table 2, Table 3.

[^21]: S. Jiao, K. Qian, H. Ye, Y. Zhong, Z. Luo, S. Jiang, Z. Huang, Y. Fang, J. Miao, Z. Fu, et al. (2025) EvaDrive: evolutionary adversarial policy optimization for end-to-end autonomous driving. arXiv preprint arXiv:2508.09158. Cited by: §2.

[^22]: D. Li, J. Ren, Y. Wang, X. Wen, P. Li, L. Xu, K. Zhan, Z. Xia, P. Jia, X. Lang, et al. (2025) Finetuning generative trajectory model with reinforcement learning from human feedback. arXiv preprint arXiv:2503.10434. Cited by: §2.

[^23]: J. Li, B. Zhang, X. Jin, J. Deng, X. Zhu, and L. Zhang (2025) ImagiDrive: a unified imagination-and-planning framework for autonomous driving. arXiv preprint arXiv:2508.11428. Cited by: §1, §2, Table 1.

[^24]: X. Li, C. Lv, W. Wang, G. Li, L. Yang, and J. Yang (2022) Generalized focal loss: towards efficient representation learning for dense object detection. IEEE transactions on pattern analysis and machine intelligence 45 (3), pp. 3139–3153. Cited by: §3.2.

[^25]: Y. Li, S. Shang, W. Liu, B. Zhan, H. Wang, Y. Wang, Y. Chen, X. Wang, Y. An, C. Tang, et al. (2025) DriveVLA-w0: world models amplify data scaling law in autonomous driving. arXiv preprint arXiv:2510.12796. Cited by: §1, §2, §5.3, Table 1.

[^26]: Y. Li, Y. Wang, Y. Liu, J. He, L. Fan, and Z. Zhang (2025) End-to-end driving with online trajectory evaluation via bev world model. arXiv preprint arXiv:2504.01941. Cited by: §5.3, Table 1.

[^27]: Y. Li, K. Xiong, X. Guo, F. Li, S. Yan, G. Xu, L. Zhou, L. Chen, H. Sun, B. Wang, et al. (2025) ReCogDrive: a reinforced cognitive framework for end-to-end autonomous driving. arXiv preprint arXiv:2506.08052. Cited by: §1, §1, §1, §2, §2, §4.1, Table 1, Table 1, Table 1, Table 2.

[^28]: Z. Li, K. Li, S. Wang, S. Lan, Z. Yu, Y. Ji, Z. Li, Z. Zhu, J. Kautz, Z. Wu, et al. (2024) Hydra-mdp: end-to-end multimodal planning with multi-target hydra-distillation. arXiv preprint arXiv:2406.06978. Cited by: §2, Table 2.

[^29]: Z. Li, Z. Yu, S. Lan, J. Li, J. Kautz, T. Lu, and J. M. Alvarez (2024) Is ego status all you need for open-loop end-to-end autonomous driving?. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14864–14873. Cited by: Table 1, Table 2.

[^30]: B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang, et al. (2025) Diffusiondrive: truncated diffusion model for end-to-end autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 12037–12047. Cited by: Appendix E, §2, Table 1, Table 2.

[^31]: R. Liu, L. Kong, D. Li, and H. Zhao (2025) OccVLA: vision-language-action model with implicit 3d occupancy supervision. arXiv preprint arXiv:2509.05578. Cited by: §1.

[^32]: W. T. Ltd. (2023) LINGO-1: exploring natural language for autonomous driving. Note: Available at: [https://wayve.ai/thinking/lingo-natural-language-autonomous-driving/](https://wayve.ai/thinking/lingo-natural-language-autonomous-driving/) Cited by: §1.

[^33]: Y. Luo, F. Li, S. Xu, Z. Lai, L. Yang, Q. Chen, Z. Luo, Z. Xie, S. Jiang, J. Liu, et al. (2025) AdaThinkDrive: adaptive thinking via reinforcement learning for autonomous driving. arXiv preprint arXiv:2509.13769. Cited by: §1, §2, §5.3, Table 1.

[^34]: A. Marcu, L. Chen, J. Hünermann, A. Karnsund, B. Hanotte, P. Chidananda, S. Nair, V. Badrinarayanan, A. Kendall, J. Shotton, et al. (2024) LingoQA: visual question answering for autonomous driving. In European Conference on Computer Vision, pp. 252–269. Cited by: §1, §2.

[^35]: L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al. (2022) Training language models to follow instructions with human feedback. Advances in neural information processing systems 35, pp. 27730–27744. Cited by: §2, §5.5.

[^36]: S. Rajbhandari, J. Rasley, O. Ruwase, and Y. He (2020) Zero: memory optimizations toward training trillion parameter models. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, pp. 1–16. Cited by: §5.2.

[^37]: K. Renz, L. Chen, A. Marcu, J. Hünermann, B. Hanotte, A. Karnsund, J. Shotton, E. Arani, and O. Sinavski (2024) Carllava: vision language models for camera-only closed-loop driving. arXiv preprint arXiv:2406.10165. Cited by: §2.

[^38]: L. Rowe, R. de Schaetzen, R. Girgis, C. Pal, and L. Paull (2025) Poutine: vision-language-trajectory pre-training and reinforcement learning post-training enable robust end-to-end autonomous driving. arXiv preprint arXiv:2506.11234. Cited by: item 1, item 3, §B.2, §2, §4.1.

[^39]: J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov (2017) Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347. Cited by: §2.

[^40]: Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu, et al. (2024) Deepseekmath: pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300. Cited by: §1, §2, §3.1.

[^41]: G. Sheng, C. Zhang, Z. Ye, X. Wu, W. Zhang, R. Zhang, Y. Peng, H. Lin, and C. Wu (2024) HybridFlow: a flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256. Cited by: §5.2.

[^42]: C. Sima, K. Renz, K. Chitta, L. Chen, H. Zhang, C. Xie, J. Beißwenger, P. Luo, A. Geiger, and H. Li (2024) Drivelm: driving with graph visual question answering. In European conference on computer vision, pp. 256–274. Cited by: §4.1.

[^43]: C. Sima, W. Tong, T. Wang, L. Chen, S. Wu, H. Deng, Y. Gu, L. Lu, P. Luo, D. Lin, and H. Li (2023) Scene as occupancy. arXiv. External Links: 2306.02851 Cited by: §5.1.

[^44]: W. Wu, X. Feng, Z. Gao, and Y. Kan (2024) Smart: scalable multi-agent real-time motion generation via next-token prediction. Advances in Neural Information Processing Systems 37, pp. 114048–114071. Cited by: §2.

[^45]: S. Xing, C. Qian, Y. Wang, H. Hua, K. Tian, Y. Zhou, and Z. Tu (2025) Openemma: open-source multimodal model for end-to-end autonomous driving. In Proceedings of the Winter Conference on Applications of Computer Vision, pp. 1001–1009. Cited by: §2, Table 3.

[^46]: Z. Yang, L. Chen, Y. Sun, and H. Li (2023) Visual point cloud forecasting enables scalable autonomous driving. arXiv preprint arXiv:2312.17655. Cited by: §5.1.

[^47]: W. Yao, Z. Li, S. Lan, Z. Wang, X. Sun, J. M. Alvarez, and Z. Wu (2025) DriveSuprim: towards precise trajectory selection for end-to-end planning. External Links: 2506.06659, [Link](https://arxiv.org/abs/2506.06659) Cited by: Table 2.

[^48]: Q. Yu, Z. Zhang, R. Zhu, Y. Yuan, X. Zuo, Y. Yue, W. Dai, T. Fan, G. Liu, L. Liu, et al. (2025) Dapo: an open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476. Cited by: §1, §2.

[^49]: J. Zhang and C. Zuo (2025) GRPO-lead: a difficulty-aware reinforcement learning approach for concise mathematical reasoning in language models. External Links: 2504.09696, [Link](https://arxiv.org/abs/2504.09696) Cited by: §5.5.

[^50]: H. Zheng, Y. Zhou, B. R. Bartoldson, B. Kailkhura, F. Lai, J. Zhao, and B. Chen (2025) Act only when it pays: efficient reinforcement learning for llm reasoning via selective rollouts. External Links: 2506.02177, [Link](https://arxiv.org/abs/2506.02177) Cited by: §5.5.

[^51]: Y. Zheng, J. Lu, S. Wang, Z. Feng, D. Kuang, and Y. Xiong (2025) EasyR1: an efficient, scalable, multi-modality rl training framework. Note: [https://github.com/hiyouga/EasyR1](https://github.com/hiyouga/EasyR1) Cited by: §5.2.

[^52]: Y. Zheng, R. Zhang, J. Zhang, Y. Ye, Z. Luo, Z. Feng, and Y. Ma (2024) LlamaFactory: unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), Bangkok, Thailand. External Links: [Link](http://arxiv.org/abs/2403.13372) Cited by: §5.2.

[^53]: X. Zhou, X. Han, F. Yang, Y. Ma, and A. C. Knoll (2025) Opendrivevla: towards end-to-end autonomous driving with large vision language action model. arXiv preprint arXiv:2503.23463. Cited by: §1, §1, Table 3.

[^54]: Z. Zhou, T. Cai, S. Z. Zhao, Y. Zhang, Z. Huang, B. Zhou, and J. Ma (2025) AutoVLA: a vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. arXiv preprint arXiv:2506.13757. Cited by: §1, §2, §2, §3.1, §5.3, §5.3, §5.5, Table 1, Table 3.
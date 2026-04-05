---
title: "Senna-2: Aligning VLM and End-to-End Driving Policy for Consistent Decision Making and Planning"
source: "https://arxiv.org/html/2603.11219v1"
author:
published:
created: 2026-04-05
description:
tags:
  - "clippings"
---
Yuehao Song <sup>1</sup>  Shaoyu Chen <sup>2,†</sup>  Hao Gao <sup>1</sup>  Yifan Zhu <sup>2</sup>  Weixiang Yue <sup>2</sup>  Jialv Zou <sup>1</sup>    
Bo Jiang <sup>1</sup>  Zihao Lu <sup>2</sup>  Yu Wang <sup>2</sup>  Qian Zhang <sup>2</sup>  Xinggang Wang ${}^{1,\textrm{\Letter}}$  
<sup>1</sup>  Huazhong University of Science & Technology  
<sup>2</sup>  Horizon Robotics  
[https://ambitious-idiot.github.io/senna2-project](https://ambitious-idiot.github.io/senna2-project)  
[https://github.com/hustvl/Senna](https://github.com/hustvl/Senna)

###### Abstract

Vision-language models (VLMs) enhance the planning capability of end-to-end (E2E) driving policy by leveraging high-level semantic reasoning. However, existing approaches often overlook the dual-system consistency between VLM’s high-level decision and E2E’s low-level planning. As a result, the generated trajectories may misalign with the intended driving decisions, leading to weakened top-down guidance and decision-following ability of the system. To address this issue, we propose Senna-2, an advanced VLM-E2E driving policy that explicitly aligns the two systems for consistent decision-making and planning. Our method follows a consistency-oriented three-stage training paradigm. In the first stage, we conduct driving pre-training to achieve preliminary decision-making and planning, with a decision adapter transmitting VLM decisions to E2E policy in the form of implicit embeddings. In the second stage, we align the VLM and the E2E policy in an open-loop setting. In the third stage, we perform closed-loop alignment via bottom-up Hierarchical Reinforcement Learning in 3DGS environments to reinforce the safety and efficiency. Extensive experiments demonstrate that Senna-2 achieves superior dual-system consistency (19.3% F1 score improvement) and significantly enhances driving safety in both open-loop (5.7% FDE reduction) and closed-loop settings (30.6% AF-CR reduction).

<sup>2</sup>

## 1 Introduction

Reliable autonomous driving systems are expected to integrate high-level decision making as guidance for low-level trajectory planning. Ensuring such decision–planning consistency enables the generation of trajectories that faithfully reflect driving intentions. Recent end-to-end (E2E) driving frameworks [^13] [^20] [^18] [^46] [^27] [^42] [^5] have demonstrated strong capability in mapping sensory inputs to driving plans through unified perception–prediction–planning pipelines. However, they still fall short in high-level reasoning and decision-making, which are crucial for ensuring safety and efficiency in complex driving scenarios. Meanwhile, vision–language models (VLMs) [^4] [^40] [^1] [^2] [^36] exhibit powerful abilities in scenario understanding and causal reasoning. Recent studies [^19] [^48] have integrated the VLM into the E2E policy to effectively remedy this cognitive defect.

![[x1 3.png|Refer to caption]]

Figure 1: 19

Despite recent advances in VLM-E2E driving policies [^29] [^17] [^10], a significant consistency gap remains between high-level decision making and low-level trajectory planning. As illustrated in Fig. 1(a), existing methods typically use VLM decisions to guide the planning of the E2E policy. However, without explicit alignment between them, the resulting trajectories often deviate from driving intentions and lead to incorrect driving directions or mismatched speed changes. Such inconsistency weakens the top-down guidance ability of the VLM and interpretability, ultimately causing suboptimal or even unsafe driving behavior.

We identify the root cause as the absence of explicit alignment between high-level decision making and low-level planning. To address the issue, we develop a three-stage consistency-oriented training framework that aligns the VLM’s high-level decision making with the E2E planner’s low-level planning, ensuring consistent decision making and planning across the two systems:

1. Driving pre-training: The VLM and E2E policy are trained on large-scale driving data to achieve preliminary ability of decision making and planning, while the decision adapter is optimized to bridge the two systems.
2. Open-loop alignment: We identify inconsistencies between the VLM and the E2E policy by analyzing the kinematic discrepancies between the planned trajectories and the corresponding VLM decisions, and selectively refine these cases to enhance dual-system consistency.
3. Closed-loop alignment with Hierarchical Reinforcement Learning (HRL): We further optimize the system in closed-loop 3DGS environments [^11] via online HRL. To balance safety and efficiency, we design composite rewards that jointly consider both aspects. Based on these rewards, the policy is optimized hierarchically: we first refine the E2E planner through longitudinal scaling penalties, and then update the VLM accordingly.

Built upon this training pipeline, we introduce Senna-2, a unified VLM–E2E driving system that ensures consistent decision-making and planning. Senna-2 integrates the semantic reasoning power of the VLM with the fine-grained planning capability of the E2E policy, leading to safer and more coherent driving behaviors. This design offers two key advantages: (1) The VLM acts as an interpretable human–machine interface that both regulates driving behavior and explains high-level decisions. (2) The combination of the VLM’s strong generalization and the E2E policy’s adaptability enhances robustness against long-tail scenarios, achieving more consistent and controllable planning compared with existing methods [^19] (Fig. 1(b, c)).

We evaluate Senna-2 across multiple dimensions, including decision-planning consistency, open-loop planning performance, and closed-loop driving stability. Experimental results demonstrate that Senna-2 achieves superior consistency between decision making and planning (19.3% F1 score improvement compared to Senna [^19]), while maintaining competitive accuracy and robustness in both open-loop benchmarks (final displacement error reduction by 5.7%) and closed-loop benchmarks (at-fault collision rate reduction by 30.6%).

The contributions are summarized as follows:

- We present Senna-2, a unified VLM-E2E driving policy that achieves consistent decision making and planning.
- We propose a consistency-oriented three-stage training paradigm that progressively aligns high-level decisions and low-level planning via large-scale pre-training, open-loop alignment, and closed-loop alignment with HRL.
- Experiments on large-scale driving benchmarks validate that Senna-2 significantly improves dual-system consistency, planning accuracy, and closed-loop robustness.

## 2 Related Work

### 2.1 End-to-End Autonomous Driving

End-to-end autonomous driving policies evolve from unified perception–planning frameworks toward probabilistic, generative, and closed-loop consistent policies. UniAD [^13] and VAD [^20] pioneer unified architectures that integrate perception, prediction, and planning to realize trajectory reasoning directly from sensor inputs. SparseDrive [^35] further explores sparse representations to improve efficiency without sacrificing global context. VADv2 [^18] and the Hydra-MDP series [^26] [^24] extend the paradigm with multi-modal planning to enhance behavioral diversity and robustness. Generative methods [^46] [^42] [^22] [^27] [^52] model the trajectory distribution as a generation task and yield more expressive driving policies beyond deterministic regression. Meanwhile, RAD [^11] utilizes closed-loop optimization to narrow the gap between open-loop training and real-world deployment. Despite these advances, most end-to-end frameworks lack high-level reasoning, motivating the integration of vision–language models for more controllable and explainable decision-making.

![[x2 2.png|Refer to caption]]

Figure 2: Overall model architecture of Senna-2. Text and visual inputs are processed by the VLM to produce high-level driving decisions, which are converted by the Decision Adapter into VLM condition embeddings. The E2E planner then fuses the VLM condition with its own E2E features to generate a trajectory consistent with the high-level decisions.

### 2.2 VLM for Autonomous Driving

Recent progress in vision–language models (VLMs) has introduced a new paradigm for autonomous driving. Early work [^31] [^7] [^44] [^49] formulates driving as a language-centric problem and employs VLMs for scene understanding through question answering. Another line of research [^15] [^41] [^48] [^25] [^34] [^37] [^39] endows VLMs with trajectory generation capability by directly integrating planning into the VLM architecture. However, as the semantic and action spaces are inherently heterogeneous [^16], such tight coupling often leads to unstable optimization. To address this, Senna [^19] decouples high-level decision making from low-level planning and leverages VLM decisions to guide end-to-end policies. Subsequent works further enhance this paradigm by introducing explicit reasoning processes [^21] [^50] or hierarchical instructions [^29]. More recent approaches incorporate generative planners [^17] [^10] to synthesize diverse, context-aware trajectories guided by high-level decisions. Nevertheless, current VLM-based driving systems still lack explicit consistency constraints between the two systems, which leads to potential misalignment between decision making and planning.

### 2.3 Cross-System Consistency in Driving

While VLM-based frameworks introduce high-level decisions to driving systems, it remains challenging to maintain consistency between decision making and planning. SimLingo [^33] proposes an action dreaming task to evaluate the alignment between instructions and planning. VLM-AD [^43] and ALN-P3 [^30] propose feature alignment between the perception, prediction, and planning features of the VLM and the E2E policy. RDADriver [^14] proposes a reasoning-decision alignment constraint between the paired CoTs and planning results. However, these approaches mainly focus on data-level or representation-level alignment and lack explicit consistency constraints between decision making and planning.

## 3 Method

### 3.1 Dual-System Architecture

Our model architecture consists of three components: the VLM module, the decision adapter, and the end-to-end driving policy, as illustrated in Fig. 2. The VLM module generates the high-level decisions, the decision adapter translates them into conditioning features, and the end-to-end policy produces final driving planning guided by these features.

##### VLM.

We adopt Qwen2.5-VL-3B [^2] as the base model for high-level driving decision-making to achieve a good trade-off between performance and efficiency. The input consists of a single front-view frame and the text input, including the system prompt, a navigation command, and the ego speed. After tokenization, the visual and textual tokens are jointly fed into the language model to predict the driving decision. The driving decision is composed of speed control and direction control. Speed control includes acceleration, deceleration, keep speed, and stop. Direction control covers go straight, turn left, turn right, change lane left, and change lane right. These structured meta actions serve as interpretable decisions that guide the trajectory planning.

##### Decision Adapter.

We design a decision adapter to transform high-level decisions produced by the VLM into representations compatible with the end-to-end driving model. Specifically, the adapter outputs two complementary types of tokens: VLM tokens and decision tokens. We extract the final hidden states from the VLM and project them through an MLP to obtain the VLM tokens $T_{\mathrm{vlm}}$, which preserves the reasoning context from the VLM. Meanwhile, to enhance the model’s awareness of VLM decisions, we introduce learnable category embeddings for speed and direction control. According to the decoded meta-actions, the velocity embedding $T_{\mathrm{vel}}$ and the direction embedding $T_{\mathrm{dir}}$ are selected as decision tokens, respectively. The VLM tokens and the two decision tokens are then fused to form the VLM condition feature:

$$
F_{\mathrm{vlm}}=\mathrm{MLP}\big(\mathrm{Concat}(T_{\mathrm{vlm}},T_{\mathrm{vel}},T_{\mathrm{dir}})\big).
$$

This fusion design combines the expressiveness of VLM tokens in capturing global semantics with the structured interpretability of VLM decisions.

![[x3 2.png|Refer to caption]]

Figure 3: Consistency-oriented training recipe. We perform three training stages, including driving pre-training, open-loop alignment, and closed-loop alignment with Hierarchical Reinforced Learning (HRL).

##### End-to-End Driving Policy.

Our end-to-end driving policy (E2E policy) consists of an E2E backbone, multiple perception heads, and a diffusion-based planner. The E2E backbone extracts spatio-temporal features from multi-view image sequences. The perception heads then predict features of static map elements, dynamic agents, and navigation cues based on the spatio-temporal features and navigation commands. These features are concatenated to form the overall E2E condition. The planner employs a DiT architecture [^32] and incorporates the E2E condition through cross attention [^38]. To introduce high-level decision guidance, we inject the VLM condition into the planning network using AdaLN [^32]. This mechanism enables the VLM condition to globally modulate the planning process, guiding the generated trajectories to align with high-level decisions.

### 3.2 Training Recipe

We adopt a three-stage training paradigm: driving pre-training, open-loop alignment, and closed-loop alignment with HRL, as shown in Fig. 3.

#### 3.2.1 Driving Pre-Training

We first perform driving pre-training to equip the VLM with basic driving knowledge and align it with the E2E policy.

##### VLM Pre-Training.

We define a kinematic mapping function $f_{K}$ that converts ground-truth trajectories into meta-actions $d$ composed of speed and direction components, following the design in Senna [^19]. This meta-action text serves as the answer in a question–answering (QA) formulation. Meanwhile, the front-view image, navigation text, and ego state are used to construct the question. We train the VLM through QA-based supervision:

$$
\mathcal{L}_{\mathrm{VLM}}=-\sum\log P(d_{t}\mid d_{<t},Q),
$$

where $Q$ denotes the question input and $d_{t}$ represents the $t$ -th meta-action token.

##### E2E Pre-Training.

We model the trajectory generation as a diffusion-based planning problem [^27]. We set the prediction target as the residual trajectory [^47], *i.e.*, the difference between the expert trajectory and the reference trajectory extrapolated from the initial speed. The planner takes the perception tokens as a condition and predicts this residual trajectory through a diffusion denoising process. During training, we adopt the standard diffusion loss between the predicted and ground-truth noise:

$$
\mathcal{L}_{\mathrm{E2E}}=\mathbb{E}_{\mathbf{r}_{0},\epsilon,t}[\left\lVert\epsilon-\epsilon_{\theta}(\mathbf{r}_{t},t,c)\right\rVert^{2}],
$$

where $\mathbf{r}_{0}$ is the ground-truth residual trajectory, $\mathbf{r}_{t}$ is the noisy sample at time step $t$, $c$ represents the E2E condition tokens, $\epsilon$ is the ground-truth noise, and $\epsilon_{\theta}$ is the predicted noise produced by the E2E model parameterized by $\theta$.

##### Adapter Pre-Training.

After standalone training, we introduce the decision adapter. We freeze the VLM parameters and optimize both the decision adapter and the E2E policy solely using the end-to-end planning loss $\mathcal{L}_{\mathrm{E2E}}$. This ensures that the adapter can bridge the modules without affecting the decision-making capabilities of the VLM.

#### 3.2.2 Open-Loop Alignment

In the second stage, we adopt an open-loop alignment training scheme. The core idea is to use the consistency between the VLM decision and the E2E planning as an explicit optimization signal. When the two systems agree, the model reinforces the corresponding behaviors; otherwise, it updates the policy under external supervision to correct the gap.

Specifically, we employ a kinematic mapping function $f_{K}$ to project the predicted trajectory $\tau$ into its corresponding decision category, which is then compared with the VLM decision $d$ to assess their behavioral consistency. To formalize this, we define a simple yet effective binary consistency indicator:

$$
\mathcal{C}(\tau,d)=\begin{cases}1&f_{K}(\tau)=d,\\
0&otherwise.\end{cases}
$$

The two are considered consistent when they belong to the same category, and inconsistent otherwise.

For inconsistent samples, the mismatch between high-level decisions and low-level planning leads to unreliable behaviors. To maintain the rationality of planning while reducing the consistency gap, we apply explicit supervision using expert trajectories and corresponding decision labels. In contrast, for consistent samples, the model regards its prediction as an internally coherent behavior. Inspired by negative sample reinforcement [^51], we skip external supervision for these samples and treat the prediction itself as an implicit expert signal that provides self-reinforcing feedback.

Based on this mechanism, the loss function is defined as:

$$
\mathcal{L}_{\mathrm{stage2}}=(1-\mathcal{C}(\tau,d))(\mathcal{L}_{\mathrm{E2E}}+\gamma\mathcal{L}_{\mathrm{VLM}}),
$$

where $\gamma$ is a balancing weight. This adaptive training strategy allows the model to achieve higher consistency through dynamic supervision.

#### 3.2.3 Closed-Loop Alignment with HRL

While open-loop alignment ensures policy consistency in regular scenarios, its reliance on offline supervision limits performance in out-of-distribution situations [^8]. To address this, we introduce a closed-loop alignment via online Hierarchical Reinforcement Learning (HRL), which better aligns decisions and planning, thereby enhancing both safety and efficiency in real-world scenarios.

To enable closed-loop training, we first collect a large set of high-risk, dense-traffic clips from driving demonstrations. Each clip is converted into an independent digital driving environment using 3D Gaussian Splatting (3DGS) [^11]. During training, the VLM–E2E policy is deployed in a subset of environments to control the ego vehicle and generate rollouts. These rollouts are then used to compute rewards and optimize the policy. Optimization follows a bottom-up hierarchical scheme: the low-level planner is first optimized using safety and efficiency rewards, and the resulting improvements are propagated to update the high-level decision, ensuring consistent alignment between the two systems.

##### Low-Level Planner Reward Design.

For the low-level planner, we design two complementary rewards based on the positional and motion states of the ego and surrounding vehicles. The safety reward is computed from the time-to-collision (TTC), where trajectories are penalized in the longitudinal direction if a collision risk is detected ($\mathrm{TTC}<3s$) to discourage unsafe behaviors. The efficiency reward is applied when the ego speed is substantially below both the navigation speed limit and the reference demonstration speed. These rollouts are treated as low-efficiency samples and are encouraged using longitudinal extension to promote smoother and faster driving. Together, these rewards guide the low-level planner to balance caution with efficiency during closed-loop training. The above mechanism can be formulated as follows:

$$
\displaystyle\mathcal{L}_{\mathrm{safe}}
$$
 
$$
\displaystyle=\mathbb{E}_{\tau\sim\pi_{\theta}}\sum^{T-1}_{t=1}\mathbbm{1}_{\big({f_{\mathrm{ttc}}}(\tau)<\delta_{t}\big)}\left\lVert\tau_{t+1}-\mathrm{sg}(\tau_{t})\right\rVert_{2}^{2},
$$
$$
\displaystyle\mathcal{L}_{\mathrm{eff}}
$$
 
$$
\displaystyle=\mathbb{E}_{\tau\sim\pi_{\theta}}\sum^{T-1}_{t=1}\mathbbm{1}_{\big({f_{\mathrm{v}}}(\tau)<\delta_{v}\big)}\left\lVert\tau_{t}-\mathrm{sg}(\tau_{t+1})\right\rVert_{2}^{2},
$$
$$
\displaystyle\mathcal{L}_{\mathrm{low}}
$$
 
$$
\displaystyle=\mathcal{L}_{\mathrm{safe}}+\mathcal{L}_{\mathrm{eff}},
$$

where $\pi_{\theta}$ denotes the driving policy parameterized by $\theta$, $\tau$ denotes the planned trajectory of length $T$, $\tau_{t}$ denotes the $t$ -th trajectory point of $\tau$, $f_{\mathrm{ttc}}(\tau)$ and $f_{v}(\tau)$ represent the TTC and velocity indicators, $\delta_{t}$ and $\delta_{v}$ are their respective thresholds, $\mathrm{sg}(\cdot)$ is the stop-gradient operator, and $\mathbbm{1}_{(\cdot)}$ is the indicator function.

##### High-Level Decision Alignment.

For the high level, we align the VLM decision with the optimized low-level planning. We map the refined trajectory to its corresponding high-level decision via the kinematic mapping function $f_{K}$. Suppose the VLM decision is inconsistent with the trajectory-corresponding decision. In that case, we further penalize the probability distribution of the VLM decision to ensure dual-system correspondence:

$$
\mathcal{L}_{\mathrm{high}}=-\log P(f_{K}(\tau)\mid Q),
$$

where $d$ denotes the VLM decision and $Q$ denotes the question input. The final loss function can be formulated as:

$$
\mathcal{L}_{\mathrm{stage3}}=\mathcal{L}_{\mathrm{high}}+\beta\mathcal{L}_{\mathrm{low}},
$$

where $\beta$ is a balancing weight between high-level and low-level losses.

Table 1: [^19]

<table><tbody><tr><td rowspan="2">   Method</td><td colspan="3">   Path (F1) <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td colspan="4">   Speed (F1) <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td rowspan="2">   Avg.</td></tr><tr><td>   Straight</td><td>   Left</td><td>   Right</td><td>   Keep</td><td>   Acc.</td><td>   Dec.</td><td>   Stop</td></tr><tr><td>   Senna <sup><a href="#fn:19">19</a></sup></td><td>   0.763</td><td>   0.533</td><td>   0.574</td><td>   0.550</td><td>   0.612</td><td>   0.628</td><td>   0.802</td><td>   0.637</td></tr><tr><td>   gray!12 Ours</td><td>   0.809</td><td>   0.664</td><td>   0.710</td><td>   0.754</td><td>   0.769</td><td>   0.780</td><td>   0.838</td><td>   0.760</td></tr></tbody></table>

## 4 Experiments

### 4.1 Experimental Settings

##### Dataset.

We collect $\approx 10,000$ hours (360M frames) of expert human driving demonstrations in real-world environments. Each demonstration includes synchronized multi-view videos, odometry data with detailed annotations of maps, traffic agents, and navigation information. Based on the odometry data, we generate high-level decision labels for first- and second-stage training. In the third stage, we select 1,300 high-risk driving clips, ranging from 15 to 40 seconds in length, and reconstruct them into 3DGS environments. We split 1044 clips for training and 256 clips for closed-loop evaluation.

##### Metrics.

We thoroughly evaluate Senna-2’s performance across three aspects: decision-planning consistency, open-loop metrics, and closed-loop metrics. Decision-planning consistency and open-loop metrics are evaluated on a subset of 100-hour driving data. Closed-loop metrics are evaluated in 3DGS environments.

To evaluate decision-planning consistency, we compute the F1 score between the VLM decisions and those obtained from the E2E trajectories via kinematic mapping.

Table 2: Open-loop quantitative comparisons with existing method. Our method outperforms existing methods in both displacement error and collision rate metrics.

| Method | FDE (m) $\downarrow$ | ADE (m) $\downarrow$ | CR (%) $\downarrow$ | DCR (%) $\downarrow$ | SCR (%) $\downarrow$ |
| --- | --- | --- | --- | --- | --- |
| TransFuser [^5] | 0.844 | 0.297 | 0.981 | 0.827 | 0.154 |
| VAD [^20] | 0.722 | 0.262 | 0.621 | 0.554 | 0.067 |
| GenAD [^46] | 0.806 | 0.290 | 0.520 | 0.491 | 0.030 |
| ResAD [^47] | 0.634 | 0.234 | 0.378 | 0.367 | 0.011 |
| Senna [^19] | 0.633 | 0.236 | 0.294 | 0.286 | 0.008 |
| gray!12 Ours | 0.597 | 0.225 | 0.288 | 0.283 | 0.005 |

The open-loop evaluation metrics include: Final Displacement Error (FDE): The Euclidean distance between the predicted trajectory endpoint and the ground-truth endpoint; Average Displacement Error (ADE): The average distance error between the predicted and ground-truth trajectories over the entire time horizon; Collision Rate (CR): The proportion of predicted trajectories that collide with other traffic participants or static obstacles in the scene; Dynamic Collision Rate (DCR): The proportion of collisions involving only dynamic objects (*e.g.*, vehicles or pedestrians); and Static Collision Rate (SCR): The proportion of collisions involving only static objects (*e.g.*, curbs).

The closed-loop evaluation metrics include: At-fault Collision Rate (AF-CR): The proportion of driving clips where collisions occur due to the ego vehicle’s inappropriate decisions; Collision Rate (CR): The overall proportion of driving clips where collisions occur during the whole rollout; and Safety@1 / Safety@2: The proportion of safe driving clips where the minimum value of time-to-collision (TTC) with surrounding agents during the whole rollout exceeds 1 or 2 seconds, respectively.

Table 3: Closed-loop quantitative comparisons with existing methods on the 3DGS evaluation benchmark. Our method significantly improves safety and reduces the collision rate.

| Method | CR $\downarrow$ | AF-CR $\downarrow$ | Safety@1 $\uparrow$ | Safety@2 $\uparrow$ |
| --- | --- | --- | --- | --- |
| TransFuser [^5] | 0.435 | 0.269 | 0.531 | 0.454 |
| VAD [^20] | 0.502 | 0.280 | 0.458 | 0.362 |
| GenAD [^46] | 0.557 | 0.244 | 0.402 | 0.332 |
| ResAD [^47] | 0.509 | 0.288 | 0.469 | 0.399 |
| VADv2 [^18] | 0.422 | 0.199 | 0.514 | 0.458 |
| RAD [^11] | 0.281 | 0.113 | 0.613 | 0.543 |
| Senna [^19] | 0.310 | 0.111 | 0.638 | 0.539 |
| gray!20 Ours | 0.269 | 0.077 | 0.667 | 0.565 |

### 4.2 Main Results

#### 4.2.1 Comparisons with Existing Methods

We evaluate Senna-2 against state-of-the-art methods in both open-loop and closed-loop settings, with all models trained on the same dataset for a fair comparison. As shown in Tab. 2, Senna-2 reduces FDE by 5.7% over prior methods in the open-loop evaluation. Closed-loop results in Tab. 3 further confirm its effectiveness, yielding a 30.6% decrease in AF-CR. Notably, in closed-loop evaluation, Senna-2 outperforms RL-based baselines, *e.g.*, RAD [^11], despite both employing closed-loop reinforcement training. This indicates that the gains stem from our alignment strategy rather than closed-loop reinforcement alone. Overall, these results demonstrate that our method consistently enhances prediction accuracy and driving safety in real-world scenarios.

#### 4.2.2 Decision-Planning Consistency

We perform both quantitative and qualitative analyses to assess the dual-system consistency between VLM decisions and E2E planning.

![[x4 2.png|Refer to caption]]

Figure 4: Closed-loop speed control in an empty-road scenario. We visualize (a) driving trajectories, (b) speed curves, and (c) mileage curves under different VLM decisions. Our method exhibits strong decision-following ability. The low-level planning follows the high-level decision for speed control. Normal denotes using the VLM-predicted decision, while accelerate, keep and decelerate denotes using the fixed ones during the whole rollout.

##### Quantitative Results.

As shown in Tab. 1, we compare the decision-planning consistency between our method and Senna [^19]. Our approach achieves a 19.3% improvement in average F1 score, indicating that it significantly improves the alignment between the VLM and the E2E policy. To intuitively illustrate open-loop controllability, we visualize the relative speed distributions under fixed-speed control actions in Fig. 1. As shown in Fig. 1(b), Senna exhibits limited separability among the three control modes, indicating weak control differentiation. In contrast, Senna-2 (Fig. 1(c)) produces clearly separated distributions, demonstrating more consistent top-down guidance.

##### Qualitative Analysis.

We present a closed-loop case study that qualitatively examines the behaviors under different fixed decisions. We perform multiple rollouts with different decisions in the same 3DGS environment. The visualization of the front-view scenes with the corresponding velocity and mileage curves is shown in Fig. 4. Our method produces stable and clearly differentiated control profiles for each action, demonstrating effective and interpretable planning consistency in real-world driving scenarios.

### 4.3 Ablation Study

Table 4: Ablation study on the model architecture. Dual system: whether the VLM is integrated; VLM token / Dec. token: whether the VLM token/decision token is used in the decision adapter.

| Dual system | VLM token | Dec. token | Open-loop FDE (m) $\downarrow$ | Open-loop CR (%) $\downarrow$ | Closed-loop AF-CR $\downarrow$ | Avg. F1 $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| ✗ | ✗ | ✗ | 0.695 | 0.387 | 0.288 | \- |
| ✓ | ✗ | ✓ | 0.634 | 0.299 | 0.148 | 0.660 |
| ✓ | ✓ | ✗ | 0.624 | 0.312 | 0.151 | 0.688 |
| gray!12 ✓ | ✓ | ✓ | 0.567 | 0.281 | 0.144 | 0.701 |

Table 5: Ablation study on the training stages. Our three-stage training pipeline strikes a balance between open-loop and closed-loop performance, while also enhancing the consistency between high-level decisions and low-level planning.

| Stage 1 | Stage 2 | Stage 3 | Open-loop FDE (m) $\downarrow$ | Open-loop CR (%) $\downarrow$ | Closed-loop AF-CR $\downarrow$ | Avg. F1 $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| ✓ |  |  | 0.567 | 0.281 | 0.144 | 0.701 |
| ✓ | ✓ |  | 0.575 | 0.290 | 0.118 | 0.764 |
| gray!12 ✓ | ✓ | ✓ | 0.597 | 0.288 | 0.077 | 0.760 |

Table 6: Performance on the NAVSIM v2 navtest Benchmark.

| Method | NC $\uparrow$ | DAC $\uparrow$ | DDC $\uparrow$ | TL $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | LK $\uparrow$ | HC $\uparrow$ | EC $\uparrow$ | EPDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Transfuser [^5] | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 |
| Hydra-MDP++ [^24] | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | 81.4 |
| DriveSuprim [^45] | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 | 83.1 |
| ARTEMIS [^9] | 98.3 | 95.1 | 98.6 | 99.8 | 81.5 | 97.4 | 96.5 | 98.3 | \- | 83.1 |
| DiffusionDrive [^27] | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | 84.5 |
| ResAD [^47] | 97.8 | 97.2 | 99.5 | 99.8 | 88.2 | 96.9 | 97.0 | 98.4 | 88.2 | 85.5 |
| gray!12 Ours | 98.5 | 97.8 | 99.5 | 99.8 | 88.1 | 97.5 | 97.0 | 98.6 | 88.4 | 86.6 |

![[x5 2.png|Refer to caption]]

Figure 5: 19

##### Model Architecture.

As illustrated in Tab. 4, we present an ablation study on the model architecture to further validate the contributions of the VLM and the decision adapter. All experiments in this table are trained using only Stage 1 to ensure a fair and independent comparison.

To evaluate the effectiveness of the dual-system design, we compare the pure E2E baseline without VLM guidance (row 1) and the proposed dual-system model (row 4). Senna-2 consistently improves safety and planning accuracy, yielding a 27.4% reduction in collision rate and a 0.128-meter decrease in FDE. This confirms that high-level decisions provided by the VLM offer reliable guidance that enhances the stability and safety of the E2E planner.

We further ablate the decision adapter by removing the VLM token (row 2) and decision tokens (row 3). Excluding the VLM token leads to clear degradation (11.8% FDE increase and 6.4% CR increase), while removing the decision token also harms performance (10.1% FDE increase and 11.0% CR increase). These results show that VLM semantics and explicit awareness of decision categories are both essential for accurate and consistent planning.

##### Training Stage.

We further present an ablation study validating the effectiveness of our proposed training recipe in Tab. 5. Training with only Stage 1 yields the lowest open-loop FDE, as the model is optimized purely for trajectory regression without considering dual-system interactions. However, this also leads to lower consistency and weaker closed-loop performance. Introducing Stage 2 significantly strengthens dual-system consistency with a 9.0% F1 score improvement, demonstrating the significance of explicit alignment. Incorporating the full Stage 3 directly improves closed-loop behavior with a 34.7% AF-CR reduction, leading to better generalization in open-world driving scenarios. These results show that the three stages play complementary roles and the complete pipeline achieves a balance between open-loop accuracy, closed-loop robustness, and decision–planning consistency.

### 4.4 Performance on Open-Source NAVSIM v2 Benchmark

We further finetune our model on the NAVSIM training split [^6] and evaluate the planning performance of Senna-2 on the NAVSIM v2 benchmark [^3]. As demonstrated in Tab. 6, our approach yields an EPDMS of 86.6, which surpasses the previous best-performing model by 1.1 EPDMS. This result underscores the robustness of Senna-2 and its capacity to master complex driving scenes.

### 4.5 Qualitative Results

We further present closed-loop qualitative comparisons between Senna-2 and Senna [^19] in Fig. 5. In Fig. 5(a), Senna exhibits a clear planning misalignment: although the VLM issues a correct decelerate command, the planner fails to respond in time, ultimately resulting in a rear-end collision with the lead vehicle. In contrast, Senna-2 maintains consistent decision–planning alignment and achieves a safe stop. In Fig. 5(b), Senna suffers from decision errors in a cut-in scenario. It first outputs an incorrect accelerate command, causing the planner to miss the proper braking window and collide with the merging vehicle. By 7.5 s, it further issues an unreasonable turn-right action after the collision. Senna-2, however, consistently provides the correct deceleration decision to manage the merge safely, and produces a reasonable change-line-left command that completes the avoidance maneuver. These results indicate that Senna-2 ensures more reliable dual-system consistency and safer closed-loop behavior in challenging situations.

## 5 Conclusion and Limitation

In this work, we present Senna-2, a dual-system driving policy that aligns a Vision-Language Model (VLM) with an end-to-end (E2E) driving policy for consistent decision-making and planning. By introducing a three-stage training strategy comprising Driving Pre-Training, Open-Loop Alignment, and Closed-Loop Alignment with HRL, Senna-2 effectively bridges the VLM decision-making with the planning of E2E models. Extensive experiments on both open-loop and closed-loop benchmarks demonstrate that Senna-2 significantly improves driving safety and interpretability over state-of-the-art baselines. This work highlights the potential of integrating multimodal reasoning into end-to-end driving systems, paving the way toward more reliable, explainable, and human-aligned autonomous driving.

A limitation is that the VLM currently cannot achieve real-time inference (10 Hz) on on-board edge devices. The VLM and E2E policy operate asynchronously, with a memory bank caching VLM features for dual-system interaction. Achieving fully synchronous cooperation requires further hardware optimization.

## References

## Appendices

## A.1 Training Configurations

In this section, we summarize the training configurations and hyperparameters used across the three stages of our framework: Driving Pre-training, Open-Loop Alignment, and Closed-Loop Alignment with HRL.

### A.1.1 Driving Pre-training

Driving pre-training includes the pre-training of the vision-language model (VLM), the end-to-end model (E2E), and the driving adapter.

##### VLM Pre-Training.

We use Qwen2.5VL-3B [^2] as our base model. For each driving demonstration, we sample frames at 1 fps. For every sampled frame, we obtain the corresponding decision label by applying the kinematic mapping function to the expert trajectories. The detailed training configurations are listed in Tab. A1.

Table A1: Training configurations for VLM pre-training.

| Config | Value |
| --- | --- |
| learning rate | $5\times 10^{-5}$ |
| learning rate schedule | cosine decay |
| optimizer | AdamW [^23] [^28] |
| optimizer hyper-parameters | $\beta_{1},\beta_{2},\epsilon=0.9,0.999,1\times 10^{-8}$ |
| weight decay | $1\times 10^{-4}$ |
| batch size | 512 |
| training steps | 10k |
| training GPU | 128 NVIDIA L20 |

##### E2E Pre-Training.

We provide the detailed hyperparameters for E2E pre-training in Tab. A2.

Table A2: Training configurations for E2E pre-training.

| Config | Value |
| --- | --- |
| learning rate | $1\times 10^{-4}$ |
| learning rate schedule | cosine decay |
| optimizer | AdamW [^23] [^28] |
| optimizer hyperparameters | $\beta_{1},\beta_{2},\epsilon=0.9,0.999,1\times 10^{-8}$ |
| weight decay | $1\times 10^{-4}$ |
| batch size | 12288 |
| training steps | 30k |
| training GPU | 1024 NVIDIA H20 |

##### Adapter Pre-Training.

We train all parameters of the driving adapter from scratch and finetune all parameters of the E2E planner. The detailed hyperparameters for adapter pre-training are provided in Tab. A3.

Table A3: Training configurations for adapter pre-training.

| Config | Value |
| --- | --- |
| learning rate | $5\times 10^{-5}$ |
| learning rate schedule | cosine decay |
| optimizer | AdamW [^23] [^28] |
| optimizer hyper-parameters | $\beta_{1},\beta_{2},\epsilon=0.9,0.999,1\times 10^{-8}$ |
| weight decay | $1\times 10^{-4}$ |
| batch size | 512 |
| training steps | 10k |
| training GPU | 128 NVIDIA L20 |

### A.1.2 Open-Loop Alignment

In each iteration, we first perform a two-step DDIM rollout to generate the predicted trajectory and compare it with the VLM decision. Only the inconsistent samples are used for training. The loss weight $\gamma$ is set to 1. During optimization, we apply LoRA finetuning [^12] to the VLM while fully finetuning the remaining modules. The detailed hyperparameters are listed in Tab. A4.

Table A4: Training configurations for open-loop alignment.

| Config | Value |
| --- | --- |
| learning rate | $1\times 10^{-5}$ |
| learning rate schedule | cosine decay |
| optimizer | AdamW [^23] [^28] |
| optimizer hyper-parameters | $\beta_{1},\beta_{2},\epsilon=0.9,0.999,1\times 10^{-8}$ |
| weight decay | 0 |
| batch size | 384 |
| training steps | 5k |
| LoRA hyperparameters | $r,\alpha=8,16$ |
| training GPU | 128 NVIDIA L20 |

### A.1.3 Closed-Loop Alignment with HRL

We assign one 3DGS environment to each GPU and perform rollouts at 10 Hz in every loop. After each rollout, we compute per-frame TTC and speed. Frames with a TTC below 3 seconds are regarded as safety-critical, while frames whose speed and distance fall behind the expert’s are treated as inefficient. To avoid introducing noisy data, all frames after the first collision are discarded. The remaining valid rollout frames are subsampled by taking one frame every four for training. The loss weight $\beta$ is set to 1. For model updates, we follow the same strategy as in Stage 2, applying LoRA finetuning [^12] to the VLM and full-parameter finetuning to the remaining components. The detailed hyperparameters are listed in Tab. A5.

Table A5: Training configurations for closed-loop alignment.

| Config | Value |
| --- | --- |
| learning rate | $1\times 10^{-5}$ |
| learning rate schedule | cosine decay |
| optimizer | AdamW [^23] [^28] |
| optimizer hyper-parameters | $\beta_{1},\beta_{2},\epsilon=0.9,0.999,1\times 10^{-8}$ |
| weight decay | 0 |
| batch size | 128 |
| training steps | 500 |
| LoRA hyperparameters | $r,\alpha=8,16$ |
| training GPU | 64 NVIDIA L20 |

## A.2 Kinematic Mapping Function

In this section, we introduce the implementation details on the kinematic mapping function $f_{k}(\tau)$. This function is used to convert trajectories into high-level decision labels for two purposes: generating ground-truth decisions from expert data, and inferring decisions from E2E planning for consistency checking with VLM outputs. We describe the function in two parts: speed and direction control.

For speed control, given a trajectory, we first extract the first 1.5s subsequence $\tau=\left\{\tau_{t}\right\}^{14}_{t=0}$ with a timestep of $\Delta t=0.1$ s. The speed $v_{t}$ and acceleration $a_{t}$ are computed as

$$
v_{t}=\frac{\lVert\tau_{t}-\tau_{t-1}\rVert}{\Delta t},\quad a_{t}=\frac{v_{t}-v_{t-1}}{\Delta t},
$$

and smooth both with an average filter of window size $w=5$. Let $f_{\text{lcs}}$ return the length of the longest continuous True segment in a Boolean sequence, and $\text{RMS}(\cdot)$ return the mean square sum. Then the trajectory is labeled as accelerate if

$$
\begin{cases}a_{1}>0,\quad f_{\text{lcs}}(a_{t}>0.3)\geq 8,\\
\max(a_{t})>0.6,\quad\text{RMS}(a_{t})>0.4.\end{cases}
$$

The trajectory is labeled as decelerate if

$$
\begin{cases}a_{1}<0,\quad f_{\text{lcs}}(a_{t}<-0.3)\geq 8,\\
\min(a_{t})<-0.6,\quad\text{RMS}(a_{t})>0.4.\end{cases}
$$

If neither condition is met, we compute the average speed $\bar{v}$ and acceleration $\bar{a}$. The label is assigned as keep speed if

$$
\begin{cases}|\bar{a}|<0.3\cdot s_{a},\\
\max(|a_{t}|)<0.6\cdot s_{a},\end{cases}
$$

or stop if $\bar{v}<0.5$, where

$$
s_{a}=\begin{cases}2.5,&\bar{v}>25,\\
2,&20<\bar{v}\leq 25,\\
1.5,&10<\bar{v}\leq 20,\\
1.25,&5<\bar{v}\leq 10,\\
1,&otherwise.\\
\end{cases}
$$

If none of the above conditions hold, the speed label is considered unknown and ignored.

For direction control, we consider two cases. For the predicted trajectory, we first compute the average speed $\bar{v}$, the maximum and minimum lateral displacement $\Delta y_{\text{max}}$ and $\Delta y_{\text{min}}$, and the maximum yaw variation $\Delta\psi_{\text{max}}$ within the 1.5-second horizon, relative to the initial time. The direction decision label $d^{\text{kmf}}_{\text{dir}}$ is assigned as

$$
d^{\text{kmf}}_{\text{dir}}=\begin{cases}\text{{left}},&\text{if }\Delta\psi_{\text{max}}>s_{\psi}\text{ and }\Delta y_{\text{max}}>s_{y},\\
\text{{right}},&\text{if }\Delta\psi_{\text{max}}>s_{\psi}\text{ and }\Delta y_{\text{min}}<-s_{y},\\
\text{{staight}},&otherwise,\\
\end{cases}
$$

where $s_{\psi}=\frac{\pi}{36}$, and

$$
s_{y}=\begin{cases}3,&\bar{v}>15,\\
2.4,&10<\bar{v}\leq 15,\\
1.5,&5<\bar{v}\leq 10,\\
0.9,&3<\bar{v}\leq 5,\\
0.45,&otherwise.\\
\end{cases}
$$

For VLM pre-training, we employ frame-level expert annotations $d^{\text{gt}}_{\text{dir}}$ as ground truth. These annotations encompass multiple categories, including go straight, turn left, turn right, change lane left, and change lane right. During consistency checking, the fine-grained VLM decisions are projected onto the coarse decision classes used by the E2E policy as follows:

$$
d^{\text{vlm}}_{\text{dir}}=\begin{cases}\text{{left}},&\text{if }d^{\text{gt}}_{\text{dir}}\in\left\{\text{{turn left}, {change lane left}}\right\},\\
\text{{right}},&\text{if }d^{\text{gt}}_{\text{dir}}\in\left\{\text{{turn right}, {change lane right}}\right\},\\
\text{{staight}},&otherwise.\\
\end{cases}
$$
![[x6 1.png|Refer to caption]]

Figure A1: Learning curve across the whole training pipeline. (a) Driving pretraining. (b) Open-loop alignment. (c) Closed-loop alignment with HRL. The dashed line denotes the performance of the extended driving-pretraining baseline.

![[x7 1.png|Refer to caption]]

Figure A2: 19

## A.3 Training Dynamics

As illustrated in Fig. A1, the learning curves highlight the effectiveness of our three-stage pipeline. Driving pretraining enables the model to acquire core driving competencies from large-scale demonstrations efficiently. Open-loop alignment substantially enhances decision–planning consistency, delivering greater performance gains than simply prolonging pretraining. Finally, closed-loop alignment with HRL requires only limited additional training yet produces marked improvements in interactive driving performance.

## A.4 More Qualitative Results

We present additional visualizations in Fig. A2 and Fig. A3 that compare our method against the baseline [^19] across a wide range of scenarios, including cut-in, right-turn, lane-change, and hard-braking events. These qualitative results highlight the performance gains and improved consistency achieved by our approach under diverse driving conditions, particularly in high-risk interactions where the baseline often reacts late or produces inconsistent planning, leading to near-collision behaviors. Corresponding video visualizations are included in the supplementary material for clearer dynamic interpretation.

![[x8 2.png|Refer to caption]]

Figure A3: More qualitative results in diverse scenarios. (a) Our policy safely handles a hard-braking event, while the baseline causes a collision. (b) During a right turn, our method avoids the cyclist, whereas the baseline crashes. (c) Our policy completes the lane change, but the baseline drifts across lanes. (d) Our method aborts the lane change upon hazard detection, while the baseline is rear-ended.

[^1]: J. Bai, S. Bai, S. Yang, S. Wang, S. Tan, P. Wang, J. Lin, C. Zhou, and J. Zhou (2023) Qwen-vl: a frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966. Cited by: §1.

[^2]: S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang, et al. (2025) Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923. Cited by: §A.1.1, §1, §3.1.

[^3]: W. Cao, M. Hallgarten, T. Li, D. Dauner, X. Gu, C. Wang, Y. Miron, M. Aiello, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, A. Geiger, and K. Chitta (2025) Pseudo-simulation for autonomous driving. In 9th Annual Conference on Robot Learning, Cited by: §4.4.

[^4]: Z. Chen, J. Wu, W. Wang, W. Su, G. Chen, S. Xing, M. Zhong, Q. Zhang, X. Zhu, L. Lu, et al. (2024) Internvl: scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 24185–24198. Cited by: §1.

[^5]: K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger (2022) Transfuser: imitation with transformer-based sensor fusion for autonomous driving. IEEE Transactions on Pattern Analysis and Machine Intelligence 45 (11), pp. 12878–12895. Cited by: §1, Table 2, Table 3, Table 6.

[^6]: D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, et al. (2024) Navsim: data-driven non-reactive autonomous vehicle simulation and benchmarking. Advances in Neural Information Processing Systems 37, pp. 28706–28719. Cited by: §4.4.

[^7]: K. Ding, B. Chen, Y. Su, H. Gao, B. Jin, C. Sima, X. Li, W. Zhang, P. Barsch, H. Li, and H. Zhao (2024) Hint-AD: holistically aligned interpretability in end-to-end autonomous driving. In 8th Annual Conference on Robot Learning, Cited by: §2.2.

[^8]: H. Dong, W. Xiong, D. Goyal, Y. Zhang, W. Chow, R. Pan, S. Diao, J. Zhang, K. Shum, and T. Zhang (2023) RAFT: reward ranked finetuning for generative foundation model alignment. Trans. Mach. Learn. Res.. Cited by: §3.2.3.

[^9]: R. Feng, N. Xi, D. Chu, R. Wang, Z. Deng, A. Wang, L. Lu, J. Wang, and Y. Huang (2025) Artemis: autoregressive end-to-end trajectory planning with mixture of experts for autonomous driving. arXiv preprint arXiv:2504.19580. Cited by: Table 6.

[^10]: H. Fu, D. Zhang, Z. Zhao, J. Cui, D. Liang, C. Zhang, D. Zhang, H. Xie, B. Wang, and X. Bai (2025-10) ORION: a holistic end-to-end autonomous driving framework by vision-language instructed action generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 24823–24834. Cited by: §1, §2.2.

[^11]: H. Gao, S. Chen, B. Jiang, B. Liao, Y. Shi, X. Guo, Y. Pu, haoran yin, X. Li, xinbang zhang, ying zhang, W. Liu, Q. Zhang, and X. Wang (2025) RAD: training an end-to-end driving policy via large-scale 3DGS-based reinforcement learning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, Cited by: item 3, §2.1, §3.2.3, §4.2.1, Table 3.

[^12]: E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen, et al. (2022) Lora: low-rank adaptation of large language models.. ICLR 1 (2), pp. 3. Cited by: §A.1.2, §A.1.3.

[^13]: Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, L. Lu, X. Jia, Q. Liu, J. Dai, Y. Qiao, and H. Li (2023-06) Planning-oriented autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 17853–17862. Cited by: §1, §2.1.

[^14]: Z. Huang, T. Tang, S. Chen, S. Lin, Z. Jie, L. Ma, G. Wang, and X. Liang (2024) Making large language models better planners with reasoning-decision alignment. In European Conference on Computer Vision, pp. 73–90. Cited by: §2.3.

[^15]: J. Hwang, R. Xu, H. Lin, W. Hung, J. Ji, K. Choi, D. Huang, T. He, P. Covington, B. Sapp, Y. Zhou, J. Guo, D. Anguelov, and M. Tan (2025) EMMA: end-to-end multimodal model for autonomous driving. Transactions on Machine Learning Research. External Links: ISSN 2835-8856 Cited by: §2.2.

[^16]: P. Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, et al. (2025) $\pi_{0.5}$: A vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054. Cited by: §2.2.

[^17]: A. Jiang, Y. Gao, Z. Sun, Y. Wang, J. Wang, J. Chai, Q. Cao, Y. Heng, H. Jiang, Y. Dong, et al. (2025) Diffvla: vision-language guided diffusion planning for autonomous driving. arXiv preprint arXiv:2505.19381. Cited by: §1, §2.2.

[^18]: B. Jiang, S. Chen, H. Gao, B. Liao, Q. Zhang, W. Liu, and X. Wang (2026) VADv2: end-to-end autonomous driving via probabilistic planning. In The Fourteenth International Conference on Learning Representations, Cited by: §1, §2.1, Table 3.

[^19]: B. Jiang, S. Chen, B. Liao, X. Zhang, W. Yin, Q. Zhang, C. Huang, W. Liu, and X. Wang (2024) Senna: bridging large vision-language models and end-to-end autonomous driving. arXiv preprint arXiv:2410.22313. Cited by: Figure 1, Figure 1, §1, §1, §1, Figure A2, Figure A2, §2.2, §3.2.1, Table 1, Table 1, Table 1, Figure 5, Figure 5, §4.2.2, §4.5, Table 2, Table 3, §A.4.

[^20]: B. Jiang, S. Chen, Q. Xu, B. Liao, J. Chen, H. Zhou, Q. Zhang, W. Liu, C. Huang, and X. Wang (2023-10) VAD: vectorized scene representation for efficient autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 8340–8350. Cited by: §1, §2.1, Table 2, Table 3.

[^21]: B. Jiang, S. Chen, Q. Zhang, W. Liu, and X. Wang (2025) Alphadrive: unleashing the power of vlms in autonomous driving via reinforcement learning and reasoning. arXiv preprint arXiv:2503.07608. Cited by: §2.2.

[^22]: X. Jiang, Y. Ma, P. Li, L. Xu, X. Wen, K. Zhan, Z. Xia, P. Jia, X. Lang, and S. Sun (2025) TransDiffuser: end-to-end trajectory generation with decorrelated multi-modal representation for autonomous driving. arXiv preprint arXiv:2505.09315. Cited by: §2.1.

[^23]: D. P. Kingma and J. Ba (2014) Adam: a method for stochastic optimization. arXiv preprint arXiv:1412.6980. Cited by: Table A1, Table A2, Table A3, Table A4, Table A5.

[^24]: K. Li, Z. Li, S. Lan, Y. Xie, Z. Zhang, J. Liu, Z. Wu, Z. Yu, and J. M. Alvarez (2025) Hydra-mdp++: advancing end-to-end driving via expert-guided hydra-distillation. arXiv preprint arXiv:2503.12820. Cited by: §2.1, Table 6.

[^25]: Y. Li, K. Xiong, X. Guo, F. Li, S. Yan, G. Xu, L. Zhou, L. Chen, H. Sun, B. WANG, K. Ma, G. Chen, H. Ye, W. Liu, and X. Wang (2026) ReCogDrive: a reinforced cognitive framework for end-to-end autonomous driving. In The Fourteenth International Conference on Learning Representations, Cited by: §2.2.

[^26]: Z. Li, K. Li, S. Wang, S. Lan, Z. Yu, Y. Ji, Z. Li, Z. Zhu, J. Kautz, Z. Wu, et al. (2024) Hydra-mdp: end-to-end multimodal planning with multi-target hydra-distillation. arXiv preprint arXiv:2406.06978. Cited by: §2.1.

[^27]: B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang, and X. Wang (2025-06) DiffusionDrive: truncated diffusion model for end-to-end autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 12037–12047. Cited by: §1, §2.1, §3.2.1, Table 6.

[^28]: I. Loshchilov and F. Hutter (2017) Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101. Cited by: Table A1, Table A2, Table A3, Table A4, Table A5.

[^29]: Y. Lu, J. Tu, Y. Ma, and X. Zhu (2025-10) ReAL-ad: towards human-like reasoning in end-to-end autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 27783–27793. Cited by: §1, §2.2.

[^30]: Y. Ma, B. Yaman, X. Ye, M. Yurt, J. Luo, A. Mallik, Z. Wang, and L. Ren (2025) ALN-p3: unified language alignment for perception, prediction, and planning in autonomous driving. arXiv preprint arXiv:2505.15158. Cited by: §2.3.

[^31]: S. Park, M. Lee, J. Kang, H. Choi, Y. Park, J. Cho, A. Lee, and D. Kim (2024) Vlaad: vision and language assistant for autonomous driving. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 980–987. Cited by: §2.2.

[^32]: W. Peebles and S. Xie (2023) Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4195–4205. Cited by: §3.1.

[^33]: K. Renz, L. Chen, E. Arani, and O. Sinavski (2025) Simlingo: vision-only closed-loop autonomous driving with language-action alignment. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 11993–12003. Cited by: §2.3.

[^34]: H. Shao, Y. Hu, L. Wang, G. Song, S. L. Waslander, Y. Liu, and H. Li (2024) Lmdrive: closed-loop end-to-end driving with large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15120–15130. Cited by: §2.2.

[^35]: W. Sun, X. Lin, Y. Shi, C. Zhang, H. Wu, and S. Zheng (2025) Sparsedrive: end-to-end autonomous driving via sparse scene representation. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pp. 8795–8801. Cited by: §2.1.

[^36]: H. Tao, B. Liao, S. Chen, H. Yin, Q. Zhang, W. Liu, and X. Wang (2025) InfiniteVL: synergizing linear and sparse attention for highly-efficient, unlimited-input vision-language models. arXiv preprint arXiv:2512.08829. Cited by: §1.

[^37]: X. Tian, J. Gu, B. Li, Y. Liu, Y. Wang, Z. Zhao, K. Zhan, P. Jia, X. Lang, and H. Zhao (2025-06–09 Nov) DriveVLM: the convergence of autonomous driving and large vision-language models. In Proceedings of The 8th Conference on Robot Learning, Proceedings of Machine Learning Research, Vol. 270, pp. 4698–4726. Cited by: §2.2.

[^38]: A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin (2017) Attention is all you need. Advances in neural information processing systems 30. Cited by: §3.1.

[^39]: S. Wang, Z. Yu, X. Jiang, S. Lan, M. Shi, N. Chang, J. Kautz, Y. Li, and J. M. Alvarez (2025-06) OmniDrive: a holistic vision-language dataset for autonomous driving with counterfactual reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 22442–22452. Cited by: §2.2.

[^40]: W. Wang, Z. Gao, L. Gu, H. Pu, L. Cui, X. Wei, Z. Liu, L. Jing, S. Ye, J. Shao, et al. (2025) Internvl3.5: advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265. Cited by: §1.

[^41]: S. Xing, C. Qian, Y. Wang, H. Hua, K. Tian, Y. Zhou, and Z. Tu (2025) Openemma: open-source multimodal model for end-to-end autonomous driving. In Proceedings of the Winter Conference on Applications of Computer Vision, pp. 1001–1009. Cited by: §2.2.

[^42]: Z. Xing, X. Zhang, Y. Hu, B. Jiang, T. He, Q. Zhang, X. Long, and W. Yin (2025) Goalflow: goal-driven flow matching for multimodal trajectories generation in end-to-end autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 1602–1611. Cited by: §1, §2.1.

[^43]: Y. Xu, Y. Hu, Z. Zhang, G. P. Meyer, S. K. Mustikovela, S. Srinivasa, E. M. Wolff, and X. Huang (2025) VLM-AD: end-to-end autonomous driving through vision-language model supervision. In 9th Annual Conference on Robot Learning, Cited by: §2.3.

[^44]: Z. Xu, Y. Zhang, E. Xie, Z. Zhao, Y. Guo, K. K. Wong, Z. Li, and H. Zhao (2024) Drivegpt4: interpretable end-to-end autonomous driving via large language model. IEEE Robotics and Automation Letters. Cited by: §2.2.

[^45]: W. Yao, Z. Li, S. Lan, Z. Wang, X. Sun, J. M. Alvarez, and Z. Wu (2025) Drivesuprim: towards precise trajectory selection for end-to-end planning. arXiv preprint arXiv:2506.06659. Cited by: Table 6.

[^46]: W. Zheng, R. Song, X. Guo, C. Zhang, and L. Chen (2024) Genad: generative end-to-end autonomous driving. In European Conference on Computer Vision, pp. 87–104. Cited by: §1, §2.1, Table 2, Table 3.

[^47]: Z. Zheng, S. Chen, H. Yin, X. Zhang, J. Zou, X. Wang, Q. Zhang, and L. Zhang (2025) ResAD: normalized residual trajectory modeling for end-to-end autonomous driving. arXiv preprint arXiv:2510.08562. Cited by: §3.2.1, Table 2, Table 3, Table 6.

[^48]: X. Zhou, X. Han, F. Yang, Y. Ma, and A. C. Knoll (2025) Opendrivevla: towards end-to-end autonomous driving with large vision language action model. arXiv preprint arXiv:2503.23463. Cited by: §1, §2.2.

[^49]: X. Zhou, L. Shan, and X. Gui (2025) Dynrsl-vlm: enhancing autonomous driving perception with dynamic resolution vision-language models. arXiv preprint arXiv:2503.11265. Cited by: §2.2.

[^50]: Z. Zhou, T. Cai, S. Z. Zhao, Y. Zhang, Z. Huang, B. Zhou, and J. Ma (2025) AutoVLA: a vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, Cited by: §2.2.

[^51]: X. Zhu, M. Xia, Z. Wei, W. Chen, D. Chen, and Y. Meng (2025) The surprising effectiveness of negative reinforcement in LLM reasoning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, Cited by: §3.2.2.

[^52]: J. Zou, S. Chen, B. Liao, Z. Zheng, Y. Song, L. Zhang, Q. Zhang, W. Liu, and X. Wang (2025) DiffusionDriveV2: reinforcement learning-constrained truncated diffusion modeling in end-to-end autonomous driving. arXiv preprint arXiv:2512.07745. Cited by: §2.1.
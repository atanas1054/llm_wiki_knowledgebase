---
title: "Unleashing VLA Potentials in Autonomous Driving via Explicit Learning from Failures"
source: "https://arxiv.org/html/2603.01063v1"
author:
published:
created: 2026-04-28
description:
tags:
  - "clippings"
---
Yuechen Luo <sup>1*</sup>, Qimao Chen <sup>1*</sup>, Fang Li <sup>2*</sup>, Shaoqing Xu <sup>2,‡</sup>, Jiaxin Liu <sup>1</sup>, Ziying Song <sup>3</sup>,  
Zhi-xin Yang <sup>2,✉</sup>, Fuxi Wen <sup>1,✉</sup>  
<sup>1</sup> Tsinghua University, <sup>2</sup> University of Macau, <sup>3</sup> Beijing Jiaotong University  
luo-yc24@mails.tsinghua.edu.cn

###### Abstract

Vision-Language-Action (VLA) models for autonomous driving often hit a performance plateau during Reinforcement Learning (RL) optimization. This stagnation arises from exploration capabilities constrained by previous Supervised Fine-Tuning (SFT), leading to “persistent failures” in long-tail scenarios. In these critical situations, all explored actions yield a zero-value driving score. This information-sparse reward signals a failure, yet fails to identify its root cause—whether it is due to incorrect planning, flawed reasoning, or poor trajectory execution. To address this limitation, we propose VLA with Explicit Learning from Failures (ELF-VLA), a framework that augments RL with structured diagnostic feedback. Instead of relying on a vague scalar reward, our method produces detailed, interpretable reports that identify the specific failure mode. The VLA policy then leverages this explicit feedback to generate a Feedback-Guided Refinement. By injecting these corrected, high-reward samples back into the RL training batch, our approach provides a targeted gradient, which enables the policy to solve critical scenarios that unguided exploration cannot. Extensive experiments demonstrate that our method unlocks the latent capabilities of VLA models, achieving state-of-the-art (SOTA) performance on the public NAVSIM benchmark for overall PDMS, EPDMS and high-level planning accuracy.

<sup>†</sup>

## 1 Introduction

The development of autonomous driving systems is undergoing a paradigm shift from traditional modular architectures to end-to-end frameworks [^4] [^11]. Vision-Language-Action (VLA) models are at the forefront of this transition [^15]. These models map raw camera sensor inputs to coherent vehicle motion commands by applying supervised fine-tuning (SFT) and reinforcement learning (RL) to large Vision-Language Models (VLMs). This integrated design eliminates manually engineered interfaces and supports large-scale, data-driven policy learning. Notably, VLA models can generate intermediate reasoning trajectories via a “think” module, mimicking human problem-solving strategies, offering a promising direction toward achieving explainable and trustworthy autonomous driving [^41] [^25] [^19].

![[introv6.png|Refer to caption]]

Figure 1: The comparison between RL fine-tuning of general VLA and ELF-VLA. Top: VLA training with RL algorithm suffers from a performance plateau: in certain scenarios, the policy model’s rollouts consistently yield low-scoring answers, trapping the agent and preventing it from discovering a better policy. Bottom: ELF-VLA addresses this by using a teacher model to provide structured feedback, which is then used to re-rollout a refinement, forcing the policy to break through this performance plateau.

Despite this progress, RL fine-tuning continues to exhibit a performance plateau: we observe that after SFT, the model’s policy exploration capability is severely constrained by the SFT dataset’s limitations, where common scenarios are highly prevalent and the safety-critical scenarios that rigorously test the autonomous system’s capabilities are rare [^23] [^8]. Consequently, under safety-critical and challenging scenarios (such as complex unprotected left turns or emergency evasions), all exploratory rollouts consistently fail, yielding a zero driving score, as shown in the top row of Fig. 1. Existing VLA-RL approaches simplify performance evaluation during training to a single scalar reward (e.g., PDMS [^7]). When the model fails, this information-sparse reward is insufficient to pinpoint the root cause of the error, making it unclear whether the failure stems from the cumulative errors of high-level planning in the “think” module, flawed cognitive reasoning about critical targets, or dynamic deficiencies in the low-level trajectory.

To address these limitations and enable continuous learning, this paper proposes a novel VLA training framework for autonomous driving that bridges failure diagnosis and policy correction. As shown in the bottom row of Fig. 1, the main idea is to provide feedback with structured failure analysis to help VLA with its “Think-then-Act” architecture, rather than relying on simple scalar rewards. This approach features two core innovations:

- VLA Ability-aligned Feedback: We introduce a feedback mechanism using a teacher model, which is triggered when the VLA encounters persistent failures. This model generates a structured diagnostic report aligned with VLA’s ability that pinpoints specific errors within the VLA’s planning, reasoning, or execution levels.
- Feedback-Guided Refinement and Re-injection: The VLA policy model (student) leverages this diagnostic report to generate a corrected trajectory. This high-reward corrected sample is then re-injected into the GRPO training batch. This process provides a goal-directed gradient signal that was previously non-existent in the rollout batch.

Through extensive evaluations on the Navsim benchmark, our method demonstrates significant performance improvements over existing VLA baselines. Our approach achieves SOTA performance on both the overall driving metric (PDMS) and high-level planning accuracy. By bridging explainable feedback with policy correction, our work provides a practical path for VLA models to overcome performance plateaus in autonomous driving.

## 2 Related Work

VLA models for autonomous driving. The integration of visual and textual data for unified perception, planning, and decision-making has led to a surge of interest in Vision-Language Models (VLMs) for autonomous driving in recent years. Two primary paradigms currently dominate the field. The first is dedicated to scene understanding and high-level reasoning [^13] [^14] [^28] [^33] [^26]. An example of this approach, Senna [^13], processes sensory inputs to generate meta-actions for downstream planners, yet substantial improvements in actual driving performance have not been fully realized. The alternative paradigm concentrates on the direct prediction of driving trajectories from raw inputs [^12] [^35] [^30] [^40] [^24] [^34] [^10] [^18]. A notable development, aimed at enhancing model interpretability and accuracy, is the increasing use of intermediate reasoning (e.g., Chain-of-Thought, CoT) to reveal internal cognitive processes. Evidence from EMMA [^12], ReasonPlan [^24], and Sce2DriveX [^40] confirms that domain-specific reasoning markedly improves the precision of trajectory forecasting.

RL fine-tuning for VLA models. Currently, VLA models in autonomous driving are typically trained using a two-stage paradigm: an initial supervised fine-tuning (SFT) phase on a driving dataset, followed by a reinforcement learning (RL) training phase. In this framework, the efficacy of the RL stage is heavily dependent on the performance of the preceding SFT. Current approaches [^25] [^19] [^41] employ the Group Relative Policy Optimization (GRPO) [^31] algorithm for RL training, where rewards are measured by VLA’s driving score (e.g., PDMS). This leads to a significant training deficiency: the VLA model, following SFT phase, struggles to handle the rare, long-tail scenarios present in the training dataset. Consequently, when the model enters RL stage, its driving score in these specific scenarios remains extremely low, regardless of the number of rollouts. This causes the model’s overall learning to stagnate, resulting in a performance plateau. In LLM literature, approaches have successfully used non-numerical feedback, such as textual critiques, to provide detailed guidance [^39] [^3]. Others [^37] [^27] have used mix-policy methods to internalize knowledge from high-quality data, enhancing exploration and policy quality. Inspired by these approaches, our work introduces a feedback mechanism to address this limitation in the autonomous driving domain, which employs a teacher model to analyze and rectify the VLA model’s erroneous driving behaviors by structured feedback, correcting them into proper actions.

## 3 Methods

![[mainv4.png|Refer to caption]]

Figure 2: Overview of ELF-VLA. First, the model is pre-trained on an autonomous driving Q&A dataset to provide it with foundational driving knowledge. Subsequently, it undergoes SFT on a mixed dataset of “Base Inputs” and “Feedback Inputs”, enabling it to learn trajectory prediction and feedback-based refinement simultaneously. Finally, in the RL phase, a teacher model is used to generate feedback, thereby reducing the proportion of zero-reward rollouts.

In this section, we present our proposed method (Fig. 2), which contains two main components: (1) a two-stage Supervised fine-Tuning (SFT) process, and (2) a Reinforcement Learning (RL) framework enhanced with failure feedback.

### 3.1 VLA Inputs Formulation

In our method, the VLA model is as the generator and refiner simultaneously, designed to accept two distinct types of inputs: the original, feedback-free base inputs, and the feedback inputs, which incorporate corrective guidance.

Base inputs. The base input query $q^{base}$ includes a front-view image denoted as $I_{\text{cam}}$, high-level navigation commands $q_{\text{com}}$ (e.g., Move Forward, Turn Left, Turn Right), ego state information $q_{\text{ego}}$ (e.g., velocity and acceleration), and the historical trajectory of the last three frames $q_{\text{his}}=\{h_{t-3},h_{t-2},h_{t-1}\}$ at a frequency of 2Hz.

Feedback inputs. Based on the base inputs, the VLA model outputs an original response $o$ consisting of trajectory with CoT (details in Appendix 6.2), which is then classified based on a threshold $s$: responses with a PDMS exceeding $s$ are deemed “correct” ($o_{c}$), while those with a score below $s$ are deemed “wrong” ($o_{w}$).

$$
q_{fb}=\begin{cases}<q_{base},o_{c},f^{rule}>&\text{if}\ o_{c},\\
<q_{base},o_{w},f^{teacher}>&\text{if}\ o_{w}.\end{cases}
$$

In the case of a correct response, the corresponding feedback inputs are composed of three components: the original base inputs $q_{base}$, the correct response $o_{c}$ itself, and a rule-based positive feedback $f^{rule}$. For wrong responses, we employ an external intervention through a VLM teacher model. This teacher model takes the base inputs $q^{base}$, the erroneous trajectory $o_{w}$, and the ground-truth trajectory $o_{gt}$ to generate a structured feedback $f^{teacher}$. This feedback includes (1) Meta Action Analysis, (2) Think Process Analysis, (3) Safety Failure Analysis, (4) Efficiency Failure Analysis and (5) Actionable Correction (with lateral and longitudinal components). The final feedback inputs for the VLA are then constructed by combining the base inputs, the original wrong response $o_{i}$, and the generated structured feedback. Detailed examples of base inputs and feedback inputs are in the Appendix 6.2.

### 3.2 Two-Stage SFT for Cognition and Refinement

We employ a two-stage supervised fine-tuning procedure to develop a model that combines both driving knowledge and trajectory planning capabilities. The first stage is designed to infuse the model with general driving knowledge. The second stage is dedicated to equipping the model with the capacity for trajectory prediction, along with the ability to implement refinements based on received feedback.

As shown in Fig. 2, in the first stage, the model is pretrained on a large dataset of driving-related Q&A pairs to enhance its understanding of driving domain cognition. This dataset is assembled from a diverse collection of open-source driving QA datasets, including DriveLM [^32], LingoQA [^28], ImpromptuVLA [^5], and other open-source driving datasets [^29] [^9] [^34]. In addition, we constructed a multi-turn Q&A reasoning dataset for NAVSIM following the CoT paradigm. This phase addresses tasks like road boundary estimation (drivable area), critical object identification (object localization), ego action prediction, and related traffic semantics. Further details on dataset composition are provided in the Appendix 6.1.

Subsequently, the second stage introduces the trajectory prediction and refinement task. For each query $q^{base}$ and $q^{fb}$ (defined in Sec. 3.3), the model’s output is supervised by the ground-truth trajectory $o_{gt}$, aiming to maximize the conditional likelihood:

$$
\mathcal{L}_{\text{SFT}}=\mathbb{E}_{(q,o)\sim\mathcal{D}}\big[-\log\pi_{\theta}(o\mid q)\big].
$$

$\mathcal{D}$ denotes the dataset including $\{(q_{base},o_{gt}),(q_{fb},o_{gt})\}$ and $\pi_{\theta}$ denotes the VLA model. This mixed-dataset training approach equips the model with the dual capabilities of trajectory prediction and feedback-based trajectory refinement, thereby enabling the model to leverage failure feedback during the reinforcement learning phase.

### 3.3 RL with Failure Feedback

![[methodv3.png|Refer to caption]]

Figure 3: Overview of GRPO with feedback. The policy model generates initial responses. Based on the rewards, teacher model (Qwen3-VL-32B) provides feedback, guiding the policy to sample improved refinement responses. A high-quality refinement response is selected and combined with the initial response set for joint optimization. Policy Shaping is applied to the final probability.

Our failure-feedback mechanism is applied during the rollout phase of the GRPO algorithm, inspired by [^39]. Conventional VLA models for autonomous driving typically hit a performance bottleneck during RL training. This occurs because they cannot manage complex, long-tail scenarios; consequently, the trajectories sampled in these situations receive extremely low driving scores, resulting in a sparse reward problem. Our method addresses this by introducing a feedback mechanism that successfully boosts the model’s driving scores in these critical scenarios, enabling the agent to break through the performance plateau.

Efficient Difficult-Sample Curation. Before introducing the GRPO with a feedback mechanism, we first perform a cost-effective data curation to maximize training efficiency. Naive RL training often wastes resources on overly simple (already mastered) scenarios that provide weak learning signals. Our curation aims to filter out these samples and focus the agent on high-value, informative scenarios, which include both difficult samples (where the model consistently fails) and ambiguous samples (where the model is most uncertain). To achieve this, we utilize the SFT model to sample $N$ rollouts for each query, estimating its mean reward and reward variance. We then discard samples characterized by high mean reward and low variance, as these indicate consistent success. This strategy effectively concentrates training on the difficult (low mean, low variance) and ambiguous (high variance) scenarios. Through this method, we filter the initial 85k training entries down to a core dataset of 24k high-value scenarios.

Reward modeling. To incentivize the VLA model to learn effective driving behaviors and to ensure the stability of its output format, we designed a reward function with three components: the PDMS Reward, the Format Reward, and the Goal Reward. The PDMS Reward $r_{traj}$ is a comprehensive trajectory evaluation metric based on the Predictive Driver Model Score [^7]. It is represented as a continuous value ranging from 0 to 1. The specific formula used to calculate this score is provided in Appendix 7. The Format Reward $r_{fmt}$ is a binary (1 or 0) reward designed to enforce adherence to the required output format strictly. Finally, the Goal Reward $r_{goal}$ incentivizes endpoint accuracy by assigning a tiered reward based on the L1 distance to the GT endpoint. Detailed calculations for each reward are provided in the Appendix 7.1. The overall reward in the reinforcement learning process is computed by integrating four designed reward components, which are presented as follows:

$$
r=r_{traj}+r_{fmt}+r_{goal}.\vskip-5.0pt
$$

GRPO with feedback. Our method employs a feedback mechanism, illustrated in Fig. 3, to refine trajectories and increase rewards, thereby enabling the VLA model to surpass its performance plateau. More specifically, the process begins by sampling a batch of trajectory responses $\{o_{i}\}_{i=1}^{n}$ using the base inputs $q^{base}$. The rewards for this batch $\{r_{i}\}_{i=1}^{n}$ are then computed, which include $\{r_{traj,i}\}_{i=1}^{n}$, $\{r_{fmt,i}\}_{i=1}^{n}$, and $\{r_{goal,i}\}_{i=1}^{n}$. Based on the predefined threshold $s$, these responses are then classified into two groups: correct responses $\{o_{c}\}$ and wrong responses $\{o_{w}\}$. Finally, following Sec. 3.1, both the correct and wrong responses are processed accordingly and then assembled to create the final feedback inputs $\{q_{i}^{fb}\}_{i=1}^{n}$.

Subsequently, the VLA model generates a new batch of responses $\{o_{i}^{fb}\}_{i=1}^{n}$ from the feedback inputs ($\{q_{i}^{fb}\}_{i=1}^{n}$) and calculates their rewards $\{r_{i}^{fb}\}_{i=1}^{n}$. We randomly select $k$ responses from the subset of $\{o_{i}^{fb}\}_{i=1}^{n}$ whose trajectory rewards ($r_{traj}^{fb}$) exceed the original batch’s maximum trajectory reward $max(r_{traj})$. If fewer than $k$ such “better” responses exist, the remaining slots are filled by duplicating the original response that achieved this maximum reward. This results in a final batch of $n+k$ rollout samples $\{(q_{i}^{final},o_{i}^{final})\}_{i=1}^{n+k}$, from which the relative advantage $\{A_{i}\}_{i=1}^{n+k}$ is then computed. Details of the rollout update algorithm are in Alg. 1. The final optimization objective for GRPO is defined as follows:

$$
\mathcal{J}(\theta)=\mathbb{E}_{D_{final}\sim\pi_{\theta_{old}}}\left[\frac{1}{n}\sum_{i=1}^{n}\mathcal{J}_{i}+\frac{1}{k}\sum_{j=1}^{k}\mathcal{J}_{j}^{fb}-\beta\mathbb{D}_{KL}\right],
$$
 
$$
\mathcal{J}_{i}=\min\big(c_{i}(\theta)A_{i},\text{clip}\left(c_{i}(\theta),1-\epsilon,1+\epsilon\right)A_{i}\big),
$$
 
$$
\mathcal{J}_{j}^{fb}=f(c_{j}^{fb}(\theta))A_{j}^{fb},
$$

where $D_{final}=\{(q_{i}^{final},o_{i}^{final})\}_{i=1}^{n+k}$, $\pi_{ref}$ is the reference policy of the initial SFT model, and $\beta$ is a hyper-parameter. We apply the CLIP only to the original batch rollout samples, not to the responses that have been refined by feedback. To compute the advantages, we first merge both sets of rewards into a unified set $r_{union}$. The mean and standard deviation of this combined set are then used to normalize the rewards and calculate the relative advantages, $A_{i}$ and $A_{j}^{fb}$, as follows:

$$
r_{union}=\{{r_{j}}\}_{j=1}^{n}\cup\{{r_{j^{\prime}}^{fb}}\}_{j^{\prime}=1}^{k},
$$
 
$$
A_{i}=\frac{r_{i}-\text{mean}(r_{union})}{\text{std}(r_{union})},A_{i}^{fb}=\frac{r_{j}^{fb}-\text{mean}(r_{union})}{\text{std}(r_{union})},
$$

For the token-level probability ratios of the feedback-generated outputs, a challenge arises from a mismatch in conditioning. These samples are generated using the feedback query $q^{fb}$, but our optimization objective is conditioned on the base query $q^{base}$. This discrepancy can cause the refined responses $\{o^{fb}\}$ to have very low probabilities under the optimization policy, leading to high variance, potential gradient explosion, and training instability. Therefore, inspired by LUFFY [^37], we employ Policy Shaping $f(x)=\frac{x}{x+\gamma}$ ($0<\gamma<1$). This technique assigns higher weights to low-probability tokens within the $\{o^{fb}\}$. Such mechanism encourages the model to learn valuable knowledge from rare but correct trajectories that might otherwise be overlooked. The standard ratio $c_{i}(\theta)$ and the shaped ratio $f(c_{j}^{fb}(\theta))$ are defined as:

$$
c_{i}(\theta)=\frac{\pi_{\theta}(o_{i}|q^{base})}{\pi_{{old}}(o_{i}|q^{base})},f(c_{j}^{fb}(\theta))=\frac{\pi_{\theta}(o_{j}^{fd}|q^{base})}{\pi_{{\theta}}(o_{j}^{fd}|q^{base})+\gamma}.
$$

Algorithm 1 Rollout Update of GRPO with Feedback

$\{q_{i}^{base}\}_{i=1}^{n}$: Base Inputs

$s$: Score Threshold

$k$: Number of Refinements

$\{q_{i}^{final}\}_{i=1}^{n+k}$: Final Query

// Step1: Initial Rollout

 $\{o_{i}\}_{i=1}^{n}=\{\texttt{VLA-MODEL}(q_{i}^{base})\}_{i=1}^{n}$ $\{r_{traj,i}\}_{i=1}^{n}=\{r_{traj}(o_{i})\}_{i=1}^{n}$ $\{f_{i}^{teacher}\}_{i=1}^{n}=\{\texttt{Teacher-Model}(q_{base},o_{i},o_{gt})\}_{i=1}^{n}$ $\{q_{i}^{fb}\}_{i=1}^{n}=\begin{cases}<q_{i}^{base},o_{i},f^{rule}>,&\text{if}\ r_{traj}(o_{i})\geq s\\
<q_{i}^{base},o_{i},f_{i}^{teacher}>,&\text{if}\ r_{traj}(o_{i})<s\end{cases}$

// Step2: Rollout with Feedback

 $\{o_{i}^{fb}\}_{i=1}^{n}=\{\texttt{VLA-Model}(q_{i}^{fb})\}_{i=1}^{n}$ $\{r_{traj,i}^{fb}\}_{i=1}^{n}=\{r_{traj}(o_{i}^{fb})\}_{i=1}^{n}$

// Step3: Final Query and Output Construction

 $r_{\max}\leftarrow\max_{i}(\{r_{traj,i}\}_{i=1}^{n})$ $\{o_{j}^{fb}\}_{j=1}^{k}=\texttt{Select}(k,\{o_{m}^{fb}\mid r_{traj,m}^{fb}>r_{\max}\})$ $\{q_{i}^{final}\}_{i=1}^{n+1}=\{q_{i}^{base}\}_{i=1}^{n}+\{q_{j}^{base}\}_{j=1}^{k}$ $\{o_{i}^{final}\}_{i=1}^{n+1}=\{o_{i}\}_{i=1}^{n}+\{o_{j}^{fb}\}_{j=1}^{k}$

return $\{q_{i}^{final}\}_{i=1}^{n+k}$, $\{o_{i}^{final}\}_{i=1}^{n+k}$

## 4 Experiment

Table 1: Comparison with state-of-the-art methods on the NAVSIMv1 with PDMS.

| Method | Image | Lidar | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | CF $\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Constant Velocity |  |  | 68.0 | 57.8 | 50.0 | 100 | 19.4 | 20.6 |
| Ego Status MLP |  |  | 93.0 | 77.3 | 83.6 | 100 | 62.8 | 65.6 |
| UniAD [^11] | ✓ |  | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| TransFuser [^6] | ✓ | ✓ | 97.7 | 92.8 | 92.8 | 100 | 84.0 | 84.0 |
| DiffusionDrive [^22] | ✓ | ✓ | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| WoTE [^17] | ✓ | ✓ | 98.5 | 96.8 | 94.9 | 99.9 | 81.9 | 88.3 |
| Hydra-NeXt [^21] | ✓ |  | 98.1 | 97.7 | 94.6 | 100 | 81.8 | 88.6 |
| AutoVLA-3B [^41] | ✓ |  | 98.4 | 95.6 | 98.0 | 100 | 81.9 | 89.1 |
| DriveVLA-W0-3B [^36] | ✓ |  | 98.7 | 99.1 | 95.3 | 99.3 | 83.3 | 90.3 |
| GoalFlow [^36] | ✓ | ✓ | 98.4 | 98.3 | 94.6 | 100 | 85.0 | 90.3 |
| InternVL3-8B-SFT | ✓ |  | 98.5 | 95.5 | 95.3 | 100 | 81.2 | 87.4 |
| InternVL3-8B-RL | ✓ |  | 98.5 | 96.7 | 95.4 | 100 | 83.2 | 89.0 |
| gray!30 ELF-VLA-8B(Ours) | ✓ |  | 98.9 | 98.1 | 96.0 | 100 | 85.3 | 91.0 |

Table 2: Comparison with state-of-the-art methods on the NAVSIMv2 with EPDMS.

| Method | NC $\uparrow$ | DAC $\uparrow$ | DDC $\uparrow$ | TLC $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | LK $\uparrow$ | HC $\uparrow$ | EC $\uparrow$ | EPDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HydraMDP++ [^20] | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | 81.4 |
| DriveSuprem [^38] | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 | 83.1 |
| Recogdrive-8B [^18] | 98.3 | 95.2 | 99.5 | 99.8 | 87.1 | 97.5 | 96.6 | 98.3 | 86.5 | 83.6 |
| DiffusionDrive [^22] | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | 84.5 |
| DriveVLA-W0-3B [^16] | 98.5 | 99.1 | 98.0 | 99.7 | 86.4 | 98.1 | 93.2 | 97.9 | 58.9 | 86.1 |
| gray!30 ELF-VLA-8B(Ours) | 98.9 | 98.1 | 99.4 | 99.8 | 88.5 | 98.4 | 96.9 | 98.3 | 87.2 | 87.1 |

### 4.1 Implementation details

Dataset. We perform comprehensive experiments and evaluations on NAVSIM [^7], a planning-oriented autonomous driving dataset built on the OpenScene. In addition to reasoning data collected from NAVSIM, we also leverage several open-source datasets, as described in Sec. 3.2.

Metric. We evaluate our method’s performance on two distinct aspects of autonomous driving: high-level planning and trajectory prediction.

For high-level planning evaluation, we use High-Level Planning Accuracy, which strictly requires the entire meta-action, comprising both longitudinal speed and lateral path, to match the ground truth exactly. Here, the ground truth is generated by GT trajectory, detailed infos in Appendix 9.

For trajectory prediction evaluation on the NAVSIM benchmark, we utilize the Predictive Driver Model Score (PDMS) for NAVSIMv1 [^7] and the Extended Predictive Driver Model Score (EPDMS) for NAVSIMv2 [^2] as the closed-loop planning metrics.

Training Details. We use InternVL3-8B [^42] as the base model, trained in three stages. First, we pretrain on a large-scale driving knowledge dataset. Second, we fine-tune the model on a hybrid dataset, consisting of curated Navsim planning dataset (with CoT annotations) and the feedback dataset from Sec. 3. Third, we apply reinforcement learning using 32 NVIDIA H20 GPUs. We use Qwen3-VL-32B [^1] as the teacher model. Key RL parameters include 8 rollouts per batch, a threshold $s=0.8$, a policy shaping parameter $\gamma=0.1$, and $k=1$ refinement response. Additional details and hyperparameters are in the Appendix 8.1.

### 4.2 Performance Comparison

Navsim Benchmark. Tab. 1 presents the performance comparison between ELF-VLA and current leading methods on the NAVSIMv1 benchmark. Under the vision-only setting, ELF-VLA achieves a PDMS of 91.0, establishing a new state-of-the-art (SOTA). This result represents a significant improvement of 0.7 PDMS over the previous best vision-only method, DriveVLA. Furthermore, ELF-VLA outperforms the SFT-only (InternVL3-8B-SFT) and traditional RL (InternVL3-8B-RL) baselines by 3.6 and 2.0 PDMS, respectively. On the NAVSIMv2 benchmark (Tab. 2), ELF-VLA continues its strong performance by achieving a new SOTA of 87.1 EPDMS. This score surpasses the previous best from DriveVLA-W0 by 1.0 EPDMS. These findings demonstrate that our method ELF-VLA substantially enhances the model’s driving capabilities over conventional RL approaches, particularly in addressing challenging driving scenarios. Moreover, the outstanding performance across both benchmarks confirms that ELF-VLA is not merely overfitting to the PDMS metric; rather, it exhibits robust generalization by excelling on the distinct and more comprehensive EPDMS as well.

Quantitative Evaluation. We compare the performance of ELF-VLA (Tab. 3) against several carefully designed ablation models (detailed definitions in Appendix 7):

- SFT (Baseline): The base model trained solely with Supervised Finetuning.
- GRPO: The SFT model was further finetuned using the conventional GRPO algorithm.
- GT-GRPO: The SFT model finetuned on a response set augmented with Ground Truth (GT) trajectories, which are added directly.
- Rule-GRPO: The SFT model finetuned on a response set augmented with new responses, which are regenerated based on feedback from predefined rules.
- ELF-VLA: The SFT model finetuned on a response set augmented with new, refined responses, which are regenerated based on structured feedback from our teacher model.

Table 3: Performance comparison of ELF-VLA against conventional GRPO and other feedback strategies.

| Method | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | CF $\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| SFT | 98.5 | 95.5 | 95.3 | 100 | 81.3 | 87.4 |
| GRPO | 98.5 | 96.7 | 95.4 | 100 | 83.2 | 89.0 |
| GT-GRPO | 98.1 | 97.1 | 93.5 | 100 | 85.2 | 89.2 |
| Rule-GRPO | 98.3 | 97.3 | 94.8 | 100 | 84.5 | 89.6 |
| gray!30 ELF-VLA | 98.9 | 98.1 | 96.0 | 100 | 85.3 | 91.0 |

![[rollout_ratio.png|Refer to caption]]

Figure 4: Ratio of total-failure samples measured during the RL training phase for GRPO, GT-GRPO, Rule-GRPO, and ELF-VLA. A total failure indicates all rollouts for a sample failed on a specific metric (PDMS below s, NC of 0 and DAC of 0, respectively).

Notably, ELF-VLA achieves the best overall performance. Our method outperforms the conventional GRPO method by 2.0 PDMS. This demonstrates that by introducing structured feedback and regenerating superior, in-distribution trajectories, our approach helps the model resolve persistent failure issues. Furthermore, ELF-VLA surpasses GT-GRPO and Rule-GRPO by 1.8 and 1.4 PDMS, respectively. This highlights the distinct limitations of these two baselines. For GT-GRPO, the GT trajectories exhibit a significant distributional shift from the original VLA-generated responses. The low likelihood of these GT responses makes optimization difficult. For Rule-GRPO, the feedback from predefined rules has a limited impact on the model. This process is akin to simple self-refinement and lacks granular guidance, causing the model to fail to learn effective trajectory correction from such simplistic feedback. In contrast, ELF-VLA utilizes the teacher model’s extensive general knowledge to perform a deep, structured analysis of the original response. The VLA model receives this comprehensive feedback, which enables it to learn from failures and refine the trajectory. This process results in a superior, more easily optimizable refined trajectory.

Total-Failure Ratio Analysis. We analyze the failure rates during the RL training phase across these models, as shown in Fig. 4. Specifically, we measure the proportion of samples where all rolled-out trajectories fail simultaneously on key metrics: PDMS, DAC, and NC. As illustrated in the figure, while intermediate strategies like GT-GRPO and Rule-GRPO help reduce failure ratios, ELF-VLA demonstrates the most significant improvements across all metrics. ELF-VLA reduces the total-failure PDMS rate from 2.73% (for GRPO) to just 1.08%, with similarly strong reductions observed for NC and DAC. This result further validates that our method enables the model to learn from its mistakes, address the persistent failure problem, and ultimately enhance overall driving safety and robustness.

High-Level Planning Evaluation. As shown in Tab. 4, our results highlight the clear advantage of ELF-VLA in high-level planning. ELF-VLA achieves the best results in both longitudinal Speed Accuracy and lateral Path Accuracy, achieving the highest overall Planning Accuracy of 80.3%, 1.0% higher than conventional GRPO. Moreover, compared to open-source models, ELF-VLA outperforms the significantly larger Qwen2.5-VL-72B model by 51.6% in accuracy. This improvement stems from the teacher model providing refined meta actions, where the VLA model learns to internalize. This demonstrates that ELF-VLA can learn from failure cases to refine its high-level planning.

Table 4: Comparison of high-level planning on NAVSIM.

| Method | Speed Acc.$\uparrow$ | Path Acc.$\uparrow$ | Accuracy $\uparrow$ |
| --- | --- | --- | --- |
| Qwen2.5-VL-7B | 37.8 | 61.3 | 19.1 |
| InternVL3-8B | 40.9 | 58.7 | 20.1 |
| Qwen2.5-VL-32B | 46.6 | 55.3 | 27.6 |
| Qwen2.5-VL-72B | 49.4 | 62.6 | 28.7 |
| SFT | 84.2 | 90.7 | 79.2 |
| GRPO | 84.3 | 90.8 | 79.3 |
| GT-GRPO | 83.5 | 90.5 | 78.4 |
| Rule-GRPO | 84.5 | 91.2 | 79.5 |
| gray!30 ELF-VLA | 85.8 | 92.5 | 80.3 |

![[visual.jpg|Refer to caption]]

Figure 5: Visualization of trajectory refinement process by ELF-VLA on the NAVSIM dataset. Visualization of the initial Wrong Trajectories ( red ), the Ground Truth ( green ), and the final Refined Trajectory ( blue ). A teacher-generated Feedback guides the refinement of a Wrong Trajectory into a Refined Trajectory. Colored text in the Feedback details the specific refinements that have been applied.

### 4.3 Ablation Studies

On GRPO with Training Data. We investigate the impact of training data volume and composition for RL, as shown in Tab. 5. Using the full 85k dataset (89.1 PDMS) or a randomly sampled 24k subset (88.9 PDMS) both yield suboptimal results. In contrast, our curated 24k dataset (24k\*), guided by Sec. 3.3, achieves the best performance of 91.0 PDMS. This suggests that the full 85k dataset is dominated by simple scenarios that provide limited learning signals, which weakens the overall gradient signal and leads to inefficient policy updates focused on already-mastered scenarios. Our curation strategy effectively distills the most valuable data. Combined with our feedback mechanism, this data allows for targeted training on these complex scenarios. This approach ultimately improves model performance and enhances training efficiency.

Table 5: Ablation study on the number of training data in RL. $\dagger$: Randomly sampled. \*: Curated as Sec. 3.3.

| Num. | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | CF $\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| 85k | 98.5 | 96.8 | 95.3 | 100 | 83.4 | 89.1 |
| 24k $\dagger$ | 98.4 | 96.8 | 95.2 | 100 | 83.1 | 88.9 |
| gray!30 24k\* | 98.9 | 98.1 | 96.0 | 100 | 85.3 | 91.0 |

On GRPO with Feedback. Tab. 6 analyzes our feedback components. We first vary the number of refinement responses, $k$. Optimal performance (91.0 PDMS) is achieved at $k=1$. Increasing $k$ degrades performance, dropping to 89.0 at $k=4$. This suggests that while a single targeted refinement is effective, multiple feedback-based responses may distract the policy. We also evaluate Policy Shaping (PS). Removing PS (at $k=1$) causes a significant 1.7% drop in PDMS from 91.0 to 89.3. This confirms PS is critical for preventing training collapse and formatting errors, ensuring the model can properly learn from high-advantage, low-probability refinement trajectories.

Table 6: Ablation study on the number of refinement responses $k$ and the use of Policy Shaping (PS).

| $k$ | PS | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | CF $\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | ✓ | 98.5 | 96.7 | 95.4 | 100 | 83.3 | 89.0 |
| 2 | ✓ | 98.2 | 97.5 | 94.3 | 100 | 84.9 | 89.7 |
| 1 | ✗ | 98.5 | 97.0 | 94.9 | 100 | 83.9 | 89.3 |
| gray!30 1 | ✓ | 98.9 | 98.1 | 96.0 | 100 | 85.3 | 91.0 |

### 4.4 Visualization of Refinement Process

Fig. 5 illustrates a qualitative example where ELF-VLA corrects a fault trajectory in a complex left-turn scenario. The initial fault trajectory (red curve) led to a potential collision, rooted in a significant misestimation of a key obstacle (predicted: 15.57m ahead, 8.11m left). Our teacher model provides structured feedback, precisely identifying this “Think Process” error and estimating the more accurate location (11.43m ahead, 4.11m left). Concurrently, it offers actionable corrections, such as adjustments to the target lateral position and longitudinal speed. Based on this feedback, the model generates the Refined Trajectory (blue curve). The corresponding “Key Obstacle Analysis” in the refined plan reflects this corrected perception, enabling the agent to plot a safer trajectory that successfully avoids the obstacle. More results can be found in Appendix 8.2.

## 5 Conclusion

This paper proposes ELF-VLA, a framework for explicit learning from failures. Our approach augments the VLA policy with a powerful teacher model that produces structured diagnostic reports and identifies the underlying failure mode whenever a failure occurs. The policy then leverages this explicit, human-like feedback to synthesize a corrected, high-reward trajectory. By re-injecting these corrected samples into the RL training batch, ELF-VLA delivers targeted gradients that enable the policy to resolve challenging scenarios where unguided exploration rarely overcomes.

The primary limitation of this method lies in its dependence on an external teacher model, which inherently bounds the student model’s performance by the teacher’s analytical capabilities. Furthermore, all experiments were conducted on the Navsim benchmark, a non-reactive simulation environment. Future work will involve exploring the role of different teacher models as well as performing closed-loop evaluations on more diverse datasets.

## References

Supplementary Material

In this supplementary material, we present comprehensive implementation details regarding data construction (Section 6) and reward design (Section 7), as well as additional experimental results, including ablation studies on the training pipeline and visualizations of the trajectory refinement process (Section 8).

## 6 Data Construction Details

### 6.1 Details of Pre-training Data

We assembled a diverse collection of open-source driving QA datasets followed by ReCogDrive [^18], including DriveLM [^32], LingoQA [^28], ImpromptuVLA [^5], NuScenes-QA [^29], NuInstruct [^9], OminiDrive [^34].

### 6.2 Details of SFT Dataset

CoT Construction. Following the data construction paradigm outlined in [^25], we generate high-quality CoT supervision by systematically synthesizing future trajectory data with scene-level semantics. In terms of dynamic entities, we filter and identify agents that actively interact with the ego vehicle based on spatio-temporal relationships. These agents are classified into three distinct categories: CIPO-1 refers to the leading vehicle in the current lane which primarily imposes longitudinal constraints; CIPO-2 includes vehicles from adjacent lanes that are inferred to merge or cut in based on lane geometry and relative velocity; and motion interaction encompasses entities whose predicted trajectories spatially intersect with the ego vehicle, indicating a high risk of collision. Regarding static elements, we leverage the NAVSIM map to reconstruct lane topology and extract critical boundary features, such as road curvature and lane centerlines, to ensure the drivable area is strictly defined. Furthermore, Qwen3-VL-32B is employed to provide fine-grained descriptions of environmental attributes, translating visual cues. By integrating these dynamic interactions, static constraints, and semantic descriptions, we curate scenarios to construct step-wise reasoning annotations that logically bridge perception, prediction, and planning. Details of CoT are shown in Fig. 10 and Fig. 11.

Base Inputs and Feedback Inputs Construction. For the base inputs, we provide the vehicle’s current velocity, acceleration, and historical trajectory as prompts to help the model better predict its path, as shown in Fig. 6. For the feedback inputs, we use Qwen3VL-32B to generate structured feedback by prompting it with the Wrong Trajectory ($o_{w}$), Ground Truth ($o_{gt}$), detailed Navsim Metric Scores, and task requirements. These explicit inputs enable the teacher model to diagnose failure causes and generate detailed corrective guidance. Details are shown in Fig. 7.

![[base_prompt.jpg|Refer to caption]]

Figure 6: Prompt design of VLA Model (Base Inputs).

![[prompt1.jpg|Refer to caption]]

Figure 7: Prompt design of Teacher Model (Feedback Inputs).

SFT Data Construction. To equip the model with a preliminary capability for trajectory refinement, we construct a feedback dataset by randomly sampling 4k correct and 4k incorrect responses from generated trajectories during the inference stage. These responses are paired with corresponding feedback based on their evaluation against a PDMS threshold $s$. Specifically, trajectory of responses scoring above $s$ are designated as “correct” ($o_{c}$) and are accompanied by rule-based feedback ($f^{rule}$) to reinforce successful behaviors. Conversely, responses trajectory scoring below $s$ are labeled as “wrong” ($o_{w}$) and are associated with teacher-guided feedback ($f^{teacher}$), which provides corrective feedback via inference from Qwen3-VL-32B.

## 7 Method Details

### 7.1 Detailed Rewards Design in RL

PDMS Reward. We input the trajectories predicted by the model into the Navsim simulator for evaluation. The simulator evaluates the driving quality of each trajectory based on several key metrics, including No At-Fault Collisions, Drivable Area Compliance, Ego Progress, Time to Collision, and Driving Comfort. It then produces a composite score, known as the Predictive Driver Model Score (PDMS), which serves as the reward signal for this component. This evaluation metric, PDMS, is used for the trajectory reward $r_{\text{traj}}$, a continuous value ranging from 0 to 1.

Format Reward. The reward term $r_{fmt}$ is designed to enforce structural validity, offering a total of 1.0 point distributed evenly between two requirements. The first half (0.5 point) is awarded if the output correctly incorporates two distinct sections: $<$ think $>$ … $<$ /think $>$ and $<$ answer $>$ … $<$ /answer $>$. The remaining 0.5 point validates the syntax of the predicted trajectory points, ensuring the output is well-formed and machine-parsable.

Goal Reward. To promote precise alignment between the predicted endpoint and the ground truth, we employ a piecewise reward function, $r_{goal}$, calculated via the L1 distance. The specific formulation is defined as follows:

$$
r_{goal}=\begin{cases}1&\text{if }0<dis<2\\
0.8&\text{if }2\leq dis<4\\
0.6&\text{if }4\leq dis<6\\
0.4&\text{if }6\leq dis<10\\
0.2&\text{if }10\leq dis<15\\
0&\text{if }dis>15\\
\end{cases}
$$

## 8 Experiment Details

### 8.1 Implementation Details

Model Architecture. We use InternVL3-8B [^42], a vision-language foundation model that combines a 300M-parameter InternViT visual encoder with a 7B-parameter Qwen2.5 language model. It features a resolution-adaptive visual input mechanism that processes images by dynamically adjusting the scale and granularity of feature extraction based on the content. This design specifically applies fine-grained processing to complex regions and coarse feature extraction to simpler areas. This enables the model to maintain high visual fidelity while optimizing computational efficiency.

Training Parameters and Hardware Configuration. The training process comprises three stages: The first stage conducts supervised fine-tuning on a large-scale, high-quality driving knowledge dataset with diverse instruction-following examples and scene-aware annotations, using 2 epochs, a batch size of 1, a learning rate of $1\times 10^{-5}$, 4 gradient accumulation steps, 0.05 weight decay, and 0.05 weight ratio. The second stage further fine-tunes the model on a NAVSIM planning dataset (with CoT annotations) and the feedback dataset, employing 2 epochs, a batch size of 2, a learning rate of $4\times 10^{-5}$, 2 gradient accumulation steps, and retaining the same weight decay and ratio (0.05 each). The third stage applies reinforcement learning with GRPO on a curated Navsim planning dataset, using a learning rate of $2\times 10^{-6}$, a batch size of 3, 16 gradient accumulation steps, 0.05 weight decay, 0.05 weight ratio, 8 generations, a temperature of 1.2, and 2 iterations. This stage utilizes 32 NVIDIA H20 GPUs. Additionally, we configure the threshold $s=0.8$, the policy shaping parameter $\gamma=0.1$, and set the refinement response to $k=1$. All experiments are conducted on NVIDIA H20 GPUs, using PyTorch 2.5.0 with CUDA 12.3 under Ubuntu. The first two stages are trained with 16 GPUs for approximately 2 days and 8 hours, respectively, the third stage with 32 GPUs for 18 hours. During inference, to generate CoT and trajectory, our model achieves a latency of 0.1s (accelerated by vLLM). Detailed hyperparameters are shown in Tab 7.

![[feedback.jpg|Refer to caption]]

Figure 8: Comparison of feedback mechanisms. Unlike the binary signals in Rule-GRPO (a), our teacher-based feedback in ELF-VLA (b) provides structured diagnostics and concrete, actionable strategies to guide trajectory refinement.

Table 7: Training hyperparameters of ELF-VLA

<table><tbody><tr><td>Stage</td><td>Hyper-parameter</td><td>Value</td></tr><tr><td rowspan="6">Pretrain</td><td>Epochs</td><td>2</td></tr><tr><td>Batch size</td><td>1</td></tr><tr><td>Learning rate</td><td><math><semantics><mrow><mn>1</mn> <mo>×</mo> <msup><mn>10</mn> <mrow><mo>−</mo> <mn>5</mn></mrow></msup></mrow> <annotation>1\times 10^{-5}</annotation></semantics></math></td></tr><tr><td>Gradient accumulation steps</td><td>4</td></tr><tr><td>Weight decay</td><td>0.05</td></tr><tr><td>Weight ratio</td><td>0.05</td></tr><tr><td rowspan="6">2</td><td>Epochs</td><td>2</td></tr><tr><td>Batch size</td><td>2</td></tr><tr><td>Learning rate</td><td><math><semantics><mrow><mn>4</mn> <mo>×</mo> <msup><mn>10</mn> <mrow><mo>−</mo> <mn>5</mn></mrow></msup></mrow> <annotation>4\times 10^{-5}</annotation></semantics></math></td></tr><tr><td>Gradient accumulation steps</td><td>2</td></tr><tr><td>Weight decay</td><td>0.05</td></tr><tr><td>Weight ratio</td><td>0.05</td></tr><tr><td rowspan="8">3</td><td>Epochs</td><td>3</td></tr><tr><td>Batch size</td><td>2</td></tr><tr><td>Learning rate</td><td><math><semantics><mrow><mn>2</mn> <mo>×</mo> <msup><mn>10</mn> <mrow><mo>−</mo> <mn>6</mn></mrow></msup></mrow> <annotation>2\times 10^{-6}</annotation></semantics></math></td></tr><tr><td>Gradient accumulation steps</td><td>16</td></tr><tr><td>Weight decay</td><td>0.05</td></tr><tr><td>Weight ratio</td><td>0.05</td></tr><tr><td>Number of generations</td><td>8</td></tr><tr><td>Number of iterations</td><td>2</td></tr><tr><td></td><td>Temperature</td><td>1.2</td></tr><tr><td></td><td>Threshold(<math><semantics><mi>s</mi> <annotation>s</annotation></semantics></math>)</td><td>0.8</td></tr><tr><td></td><td>Number of refiniements</td><td>1</td></tr><tr><td></td><td>Policy shaping weight(<math><semantics><mi>γ</mi> <annotation>\gamma</annotation></semantics></math>)</td><td>0.1</td></tr></tbody></table>

Metric. To evaluate trajectory prediction within the NAVSIM benchmark, we adopt the Predictive Driver Model Score (PDMS) for NAVSIMv1 [^7] and the Extended Predictive Driver Model Score (EPDMS) for NAVSIMv2 [^2] as our primary closed-loop planning metrics.

For NAVSIMv1, PDMS integrates five sub-metrics: No At-Fault Collision (NC), Drivable Area Compliance (DAC), Time-to-Collision (TTC), Comfort (C), and Ego Progress (EP) to produce a comprehensive closed-loop planning score. Its calculation formula is defined as follows:

$$
PDMS=NC\times DAC\times\left(\frac{5\times EP+5\times TTC+2\times C}{12}\right),
$$

For NAVSIMv2, EPDMS metric includes several components categorized as penalties or weighted subscores. Its key metrics are No at-Fault Collision (NC), Drivable Area Compliance (DAC), Driving Direction Compliance (DDC), Traffic Light Compliance (TLC), Ego Progress (EP), Time to Collision (TTC), Lane Keeping (LK), History Comfort (HC), and Extended Comfort (EC). Its calculation formula is defined as follows:

$$
\begin{split}EPDMS={}&NC\times DAC\times DDC\times TLC\times\\
&\left(\frac{5EP+2LK+2HC+5TTC+2EC}{16}\right).\end{split}
$$

Comparison of Feedback Mechanisms. As outlined in Sec 4.2, the distinction between Rule-GRPO and ELF-VLA lies fundamentally in the mechanism and granularity of the feedback. Rule-GRPO relies on static heuristics that provide binary feedback which simply indicates correctness or failure, subsequently referencing the ground truth to guide the regeneration of the correct answer. In contrast, ELF-VLA generates online, instance-specific feedback. By leveraging the fine-grained metrics within PDMS, it analyzes the specific causes of error for each faulty trajectory, providing detailed reasoning and constructive guidance to steer the correction process. As illustrated in Fig. 8, rule-based feedback offers limited constraints, and ELF-VLA’s feedback delivers comprehensive diagnostics for precise trajectory optimization.

![[meta1.jpg|Refer to caption]]

Figure 9: Illustration of high-level actions and ground truth generation. The top panels list the discrete categories for longitudinal and lateral planning. The bottom panels demonstrate the labeling criteria: longitudinal states are determined by sliding-window acceleration analysis, while lateral behaviors are identified based on the relationship between the vehicle’s trajectory and map topology.

High-Level Planning Accuracy. To evaluate the model’s decision-making capabilities, we employ the high-level planning accuracy metric, which assesses the precision across two distinct dimensions: longitudinal speed and lateral path. The longitudinal dimension classifies vehicle behavior into four discrete states: accelerate, decelerate, maintain speed, and stop. The lateral dimension encompasses five directional maneuvers: keep lane, turn left, turn right, change to the left lane, and change to the right lane. As illustrated in Fig. 9, the ground truth labels are derived directly from the trajectories. Specifically, longitudinal states are determined by calculating acceleration via a sliding window. For lateral maneuvers, we construct a road topology graph to identify the specific action based on the alignment between the future path and the lane structure.

### 8.2 More Ablation Studies

On Training Pipeline. Tab. 8 presents the ablation results of the three-stage training pipeline for ELF-VLA. Using only NAVSIM trajectory data for SFT, the model achieves a PDMS of 85.3. Adding pretraining on a large-scale driving QA dataset boosts the score to 87.4 PDMS, representing a 2.1 increase. Incorporating Feedback-Grpo further improves performance to 91.0 PDMS, a gain of 3.6. These results demonstrate that both pretraining and the Feedback-Grpo strategy play a crucial role in enhancing the model’s understanding and reasoning capabilities.

Table 8: Ablation Study on ELF-VLA Components. We evaluate the effect of pre-training, supervised fine-tuning, and reinforcement learning on driving performance using NAVSIM benchmark.

| Model | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | CF $\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| SFT | 98.5 | 93.4 | 95.1 | 100 | 78.8 | 85.3 |
| Pre+SFT | 98.5 | 95.5 | 95.3 | 100 | 81.2 | 87.4 |
| gray!30 Pre+SFT+RL | 98.9 | 98.1 | 96.0 | 100 | 85.3 | 91.0 |

On Pre-training for Feedback-GRPO. We conducted an ablation study excluding pre-training datasets (see Tab. 9). Even without pre-training, ELF-VLA achieves 90.0 PDMS, surpassing the baseline (86.9 PDMS) by +3.1. This confirms that the performance gains are primarily driven by our GRPO design rather than just pretrain data.

Table 9: Performance of ELF-VLA without pre-training. “w/o” denotes “without”.

| Method | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | CF $\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| InternVL3-8B(w/o pretrain) | 97.9 | 93.9 | 94.4 | 100 | 82.8 | 86.9 |
| gray!30 ELF-VLA-8B(w/o pretrain) | 98.1 | 97.0 | 94.4 | 100 | 86.2 | 90.0 |

On Feedback Threshold $s$. We investigate the sensitivity of the threshold $s$, which serves as the criterion for triggering the teacher model’s feedback mechanism (as formulated in Sec. 3.1). Specifically, the teacher is invoked to generate refinement guidance only when the model’s response score falls below $s$. As shown in Tab. 10, the model achieves peak performance at $s=0.8$. Lower thresholds (e.g., 0 and 0.5) yield suboptimal results, primarily because they restrict the scope of correction to complete failures (score 0), neglecting marginally poor samples that still require improvement. By setting $s=0.8$, we effectively expand the refinement scope to capture and correct these suboptimal cases. Conversely, aggressively increasing the threshold to 0.9 leads to a performance degradation. This suggests that samples scoring in the range of $[0.8,0.9)$ are already sufficiently high-quality. Forcing refinement on these valid responses yields no positive gain and may instead introduce noise, causing the optimization process to diverge from the optimal policy. Consequently, our ELF-VLA employs $s=0.8$ to balance correction coverage and training stability.

Table 10: Ablation Study on the Feedback Threshold $s$.

| Threshold | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | CF $\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 98.4 | 97.1 | 94.3 | 100 | 84.7 | 89.4 |
| 0.5 | 98.6 | 97.4 | 95.3 | 100 | 84.9 | 90.1 |
| 0.9 | 98.4 | 97.3 | 94.9 | 100 | 84.8 | 89.8 |
| gray!30 0.8 | 98.9 | 98.1 | 96.0 | 100 | 85.3 | 91.0 |

### 8.3 Visualization of Refinement Process

Fig. 10 and Fig. 11 presents additional qualitative examples demonstrating the efficacy of ELF-VLA in complex scenarios. As illustrated, our structured feedback mechanism plays a critical role in the refinement loop. First, it validates the correctness of high-level planning decisions. Second, and crucially, it rectifies the intermediate CoT reasoning process. This step effectively mitigates the risk of error accumulation, ensuring that the CoT serves its intended purpose of enhancing trajectory prediction rather than introducing hallucinations or cascading faults. To achieve this, we leverage the spatial reasoning capabilities of the teacher model (Qwen3-VL-32B) to provide precise localization of key obstacles. Furthermore, aided by the granular scoring within PDMS, we conduct a detailed diagnosis of failure modes, specifically categorized into safety failures and efficiency failures. This comprehensive analysis culminates in the generation of concrete, actionable correction strategies to refine the final output.

![[visual1.jpg|Refer to caption]]

Figure 10: Visualization of trajectory refinement process by ELF-VLA on the NAVSIM dataset. Visualization of the initial Wrong Trajectories ( red ), the Ground Truth ( green ), and the final Refined Trajectory ( blue ).

![[visual2.jpg|Refer to caption]]

Figure 11: Visualization of trajectory refinement process by ELF-VLA on the NAVSIM dataset. Visualization of the initial Wrong Trajectories ( red ), the Ground Truth ( green ), and the final Refined Trajectory ( blue ).

[^1]: S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang, et al. (2025) Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923. Cited by: §4.1.

[^2]: W. Cao, M. Hallgarten, T. Li, D. Dauner, X. Gu, C. Wang, Y. Miron, M. Aiello, H. Li, I. Gilitschenski, et al. (2025) Pseudo-simulation for autonomous driving. arXiv preprint arXiv:2506.04218. Cited by: §4.1, §8.1.

[^3]: A. Chen, J. Scheurer, J. A. Campos, T. Korbak, J. S. Chan, S. R. Bowman, K. Cho, and E. Perez (2024) Learning from natural language feedback. Transactions on machine learning research. Cited by: §2.

[^4]: L. Chen, P. Wu, K. Chitta, B. Jaeger, A. Geiger, and H. Li (2024) End-to-end autonomous driving: challenges and frontiers. IEEE Transactions on Pattern Analysis and Machine Intelligence. Cited by: §1.

[^5]: H. Chi, H. Gao, Z. Liu, J. Liu, C. Liu, J. Li, K. Yang, Y. Yu, Z. Wang, W. Li, et al. (2025) Impromptu vla: open weights and open data for driving vision-language-action models. arXiv preprint arXiv:2505.23757. Cited by: §3.2, §6.1.

[^6]: K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger (2022) Transfuser: imitation with transformer-based sensor fusion for autonomous driving. IEEE transactions on pattern analysis and machine intelligence 45 (11), pp. 12878–12895. Cited by: Table 1.

[^7]: D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, et al. (2024) Navsim: data-driven non-reactive autonomous vehicle simulation and benchmarking. Advances in Neural Information Processing Systems 37, pp. 28706–28719. Cited by: §1, §3.3, §4.1, §4.1, §8.1.

[^8]: W. Ding, C. Xu, M. Arief, H. Lin, B. Li, and D. Zhao (2023) A survey on safety-critical driving scenario generation—a methodological perspective. IEEE Transactions on Intelligent Transportation Systems 24 (7), pp. 6971–6988. Cited by: §1.

[^9]: X. Ding, J. Han, H. Xu, X. Liang, W. Zhang, and X. Li (2024) Holistic autonomous driving understanding by bird’s-eye-view injected multi-modal large models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13668–13677. Cited by: §3.2, §6.1.

[^10]: H. Fu, D. Zhang, Z. Zhao, J. Cui, D. Liang, C. Zhang, D. Zhang, H. Xie, B. Wang, and X. Bai (2025) Orion: a holistic end-to-end autonomous driving framework by vision-language instructed action generation. arXiv preprint arXiv:2503.19755. Cited by: §2.

[^11]: Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, et al. (2023) Planning-oriented autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 17853–17862. Cited by: §1, Table 1.

[^12]: J. Hwang, R. Xu, H. Lin, W. Hung, J. Ji, K. Choi, D. Huang, T. He, P. Covington, B. Sapp, et al. (2024) Emma: end-to-end multimodal model for autonomous driving. arXiv preprint arXiv:2410.23262. Cited by: §2.

[^13]: B. Jiang, S. Chen, B. Liao, X. Zhang, W. Yin, Q. Zhang, C. Huang, W. Liu, and X. Wang (2024) Senna: bridging large vision-language models and end-to-end autonomous driving. arXiv preprint arXiv:2410.22313. Cited by: §2.

[^14]: B. Jiang, S. Chen, Q. Zhang, W. Liu, and X. Wang (2025) Alphadrive: unleashing the power of vlms in autonomous driving via reinforcement learning and reasoning. arXiv preprint arXiv:2503.07608. Cited by: §2.

[^15]: S. Jiang, Z. Huang, K. Qian, Z. Luo, T. Zhu, Y. Zhong, Y. Tang, M. Kong, Y. Wang, S. Jiao, et al. (2025) A survey on vision-language-action models for autonomous driving. arXiv preprint arXiv:2506.24044. Cited by: §1.

[^16]: Y. Li, S. Shang, W. Liu, B. Zhan, H. Wang, Y. Wang, Y. Chen, X. Wang, Y. An, C. Tang, et al. (2025) DriveVLA-w0: world models amplify data scaling law in autonomous driving. arXiv preprint arXiv:2510.12796. Cited by: Table 2.

[^17]: Y. Li, Y. Wang, Y. Liu, J. He, L. Fan, and Z. Zhang (2025) End-to-end driving with online trajectory evaluation via bev world model. arXiv preprint arXiv:2504.01941. Cited by: Table 1.

[^18]: Y. Li, K. Xiong, X. Guo, F. Li, S. Yan, G. Xu, L. Zhou, L. Chen, H. Sun, B. Wang, et al. (2025) ReCogDrive: a reinforced cognitive framework for end-to-end autonomous driving. arXiv preprint arXiv:2506.08052. Cited by: §2, Table 2, §6.1.

[^19]: Y. Li, M. Tian, D. Zhu, J. Zhu, Z. Lin, Z. Xiong, and X. Zhao (2025) Drive-r1: bridging reasoning and planning in vlms for autonomous driving with reinforcement learning. arXiv preprint arXiv:2506.18234. Cited by: §1, §2.

[^20]: Z. Li, K. Li, S. Wang, S. Lan, Z. Yu, Y. Ji, Z. Li, Z. Zhu, J. Kautz, Z. Wu, et al. (2024) Hydra-mdp: end-to-end multimodal planning with multi-target hydra-distillation. arXiv preprint arXiv:2406.06978. Cited by: Table 2.

[^21]: Z. Li, S. Wang, S. Lan, Z. Yu, Z. Wu, and J. M. Alvarez (2025) Hydra-next: robust closed-loop driving with open-loop training. arXiv preprint arXiv:2503.12030. Cited by: Table 1.

[^22]: B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang, et al. (2025) Diffusiondrive: truncated diffusion model for end-to-end autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 12037–12047. Cited by: Table 1, Table 2.

[^23]: H. X. Liu and S. Feng (2024) Curse of rarity for autonomous vehicles. nature communications 15 (1), pp. 4808. Cited by: §1.

[^24]: X. Liu, Z. Zhong, Y. Guo, Y. Liu, Z. Su, Q. Zhang, J. Wang, Y. Gao, Y. Zheng, Q. Lin, et al. (2025) ReasonPlan: unified scene prediction and decision reasoning for closed-loop autonomous driving. arXiv preprint arXiv:2505.20024. Cited by: §2.

[^25]: Y. Luo, F. Li, S. Xu, Z. Lai, L. Yang, Q. Chen, Z. Luo, Z. Xie, S. Jiang, J. Liu, et al. (2025) AdaThinkDrive: adaptive thinking via reinforcement learning for autonomous driving. arXiv preprint arXiv:2509.13769. Cited by: §1, §2, §6.2.

[^26]: Z. Luo, K. Qian, J. Wang, Y. Luo, J. Miao, Z. Fu, Y. Wang, S. Jiang, Z. Huang, Y. Hu, et al. (2025) MTRDrive: memory-tool synergistic reasoning for robust autonomous driving in corner cases. arXiv preprint arXiv:2509.20843. Cited by: §2.

[^27]: L. Ma, H. Liang, M. Qiang, L. Tang, X. Ma, Z. H. Wong, J. Niu, C. Shen, R. He, Y. Li, et al. (2025) Learning what reinforcement learning can’t: interleaved online fine-tuning for hardest questions. arXiv preprint arXiv:2506.07527. Cited by: §2.

[^28]: A. Marcu, L. Chen, J. Hünermann, A. Karnsund, B. Hanotte, P. Chidananda, S. Nair, V. Badrinarayanan, A. Kendall, J. Shotton, et al. (2024) LingoQA: visual question answering for autonomous driving. In European Conference on Computer Vision, pp. 252–269. Cited by: §2, §3.2, §6.1.

[^29]: T. Qian, J. Chen, L. Zhuo, Y. Jiao, and Y. Jiang (2024) Nuscenes-qa: a multi-modal visual question answering benchmark for autonomous driving scenario. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38, pp. 4542–4550. Cited by: §3.2, §6.1.

[^30]: Z. Qiao, H. Li, Z. Cao, and H. X. Liu (2025) Lightemma: lightweight end-to-end multimodal model for autonomous driving. arXiv preprint arXiv:2505.00284. Cited by: §2.

[^31]: Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu, et al. (2024) Deepseekmath: pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300. Cited by: §2.

[^32]: C. Sima, K. Renz, K. Chitta, L. Chen, H. Zhang, C. Xie, J. Beißwenger, P. Luo, A. Geiger, and H. Li (2024) Drivelm: driving with graph visual question answering. In European conference on computer vision, pp. 256–274. Cited by: §3.2, §6.1.

[^33]: X. Tian, J. Gu, B. Li, Y. Liu, Y. Wang, Z. Zhao, K. Zhan, P. Jia, X. Lang, and H. Zhao (2024) Drivevlm: the convergence of autonomous driving and large vision-language models. arXiv preprint arXiv:2402.12289. Cited by: §2.

[^34]: S. Wang, Z. Yu, X. Jiang, S. Lan, M. Shi, N. Chang, J. Kautz, Y. Li, and J. M. Alvarez (2025) Omnidrive: a holistic vision-language dataset for autonomous driving with counterfactual reasoning. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 22442–22452. Cited by: §2, §3.2, §6.1.

[^35]: S. Xing, C. Qian, Y. Wang, H. Hua, K. Tian, Y. Zhou, and Z. Tu (2025) Openemma: open-source multimodal model for end-to-end autonomous driving. In Proceedings of the Winter Conference on Applications of Computer Vision, pp. 1001–1009. Cited by: §2.

[^36]: Z. Xing, X. Zhang, Y. Hu, B. Jiang, T. He, Q. Zhang, X. Long, and W. Yin (2025) Goalflow: goal-driven flow matching for multimodal trajectories generation in end-to-end autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 1602–1611. Cited by: Table 1, Table 1.

[^37]: J. Yan, Y. Li, Z. Hu, Z. Wang, G. Cui, X. Qu, Y. Cheng, and Y. Zhang (2025) Learning to reason under off-policy guidance. arXiv preprint arXiv:2504.14945. Cited by: §2, §3.3.

[^38]: W. Yao, Z. Li, S. Lan, Z. Wang, X. Sun, J. M. Alvarez, and Z. Wu (2025) DriveSuprim: towards precise trajectory selection for end-to-end planning. arXiv preprint arXiv:2506.06659. Cited by: Table 2.

[^39]: X. Zhang, H. Sun, Y. Zhang, K. Feng, C. Lu, C. Yang, and H. Meng (2025) Critique-grpo: advancing llm reasoning with natural language and numerical feedback. arXiv preprint arXiv:2506.03106. Cited by: §2, §3.3.

[^40]: R. Zhao, Q. Yuan, J. Li, H. Hu, Y. Li, C. Zheng, and F. Gao (2025) Sce2drivex: a generalized mllm framework for scene-to-drive learning. arXiv preprint arXiv:2502.14917. Cited by: §2.

[^41]: Z. Zhou, T. Cai, S. Z. Zhao, Y. Zhang, Z. Huang, B. Zhou, and J. Ma (2025) AutoVLA: a vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. arXiv preprint arXiv:2506.13757. Cited by: §1, §2, Table 1.

[^42]: J. Zhu, W. Wang, Z. Chen, Z. Liu, S. Ye, L. Gu, H. Tian, Y. Duan, W. Su, J. Shao, et al. (2025) Internvl3: exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479. Cited by: §4.1, §8.1.
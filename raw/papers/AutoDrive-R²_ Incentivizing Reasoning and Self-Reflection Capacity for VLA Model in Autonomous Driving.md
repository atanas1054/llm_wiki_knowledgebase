---
title: "AutoDrive-R²: Incentivizing Reasoning and Self-Reflection Capacity for VLA Model in Autonomous Driving"
source: "https://arxiv.org/html/2509.01944v1"
author:
published:
created: 2026-04-05
description:
tags:
  - "clippings"
---
Zhenlong Yuan <sup>1</sup> <sup><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\dagger"><semantics><mo>†</mo> <annotation>\dagger</annotation></semantics></math></sup>, Jing Tang <sup>1</sup>, Jinguo Luo <sup>1</sup>, Rui Chen <sup>1</sup>, Chengxuan Qian <sup>1</sup>,  
Lei Sun <sup>1</sup> <sup><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\ddagger"><semantics><mo>‡</mo> <annotation>\ddagger</annotation></semantics></math></sup>, Xiangxiang Chu <sup>1</sup>, Yujun Cai <sup>2</sup>, Dapeng Zhang <sup>3</sup> <sup>§</sup>, Shuo Li <sup>4</sup>  
<sup>1</sup> AMAP, Alibaba Group  <sup>2</sup> University of Queensland  <sup>3</sup> Lanzhou University  <sup>4</sup> Case Western Reserve University  

###### Abstract

Vision–Language–Action (VLA) models in autonomous driving systems have recently demonstrated transformative potential by integrating multimodal perception with decision-making capabilities. However, the interpretability and coherence of the decision process and the plausibility of action sequences remain largely underexplored. To address these issues, we propose AutoDrive-R², a novel VLA framework that enhances both reasoning and self-reflection capabilities of autonomous driving systems through chain-of-thought (CoT) processing and reinforcement learning (RL). Specifically, we first propose an innovative CoT dataset named nuScenesR²-6K for supervised fine-tuning, which effectively builds cognitive bridges between input information and output trajectories through a four-step logical chain with self-reflection for validation. Moreover, to maximize both reasoning and self-reflection during the RL stage, we further employ the Group Relative Policy Optimization (GRPO) algorithm within a physics-grounded reward framework that incorporates spatial alignment, vehicle dynamic, and temporal smoothness criteria to ensure reliable and realistic trajectory planning. Extensive evaluation results across both nuScenes and Waymo datasets demonstrates the state-of-the-art performance and robust generalization capacity of our proposed method.

![[Uncaptioned image]](https://arxiv.org/html/2509.01944v1/x1.png)

Figure 1: AutoDrive-R² can effectively achieve planning trajectories across multiple benchmarks compared with other models. Given vehicle’s initial visual inputs and language information, AutoDrive-R² achieves comprehensive contextual reasoning for precise trajectory planning. Our model achieves state-of-the-art performance on both nuScenes and Waymo benchmarks.

<sup>1</sup> <sup>2</sup>

## 1 Introduction

Autonomous driving technologies have witnessed rapid advancements in recent years. These systems typically take sensor data as input and then output planning trajectories. Traditional pipelines [^15] [^5] usually adopt architectures with separate perception, mapping, prediction, and planning modules. Such design may suffer from two key limitations: error accumulation and lack joint optimization across components, leading to performance degradation. In contrast, modern methods [^11] [^13] [^7] unify these complex systems into a single end-to-end paradigm, which naturally offers three main benefits: system simplification, enhanced robustness, and alleviated error accumulation.

However, these end-to-end methods primarily focus on trajectory planning while lacking the contextual reasoning necessary for complex driving scenarios. To address this limitation, recent works integrate Vision-Language Models (VLMs) into autonomous driving systems, leveraging their pre-trained reasoning capabilities to enhance decision-making in challenging situations [^24] [^29] [^27]. Unlike traditional approaches that train perception-policy modules from scratch, VLM-based methods instead fine-tune pre-trained models by leveraging pre-training on millions of image-text pairs, enhancing vehicles to interpret dynamic traffic situations and develop sophisticated navigation strategies. Despite promising results, current systems still struggle with consistently producing accurate planning outputs.

Building upon VLMs, Vision-Language-Action models (VLA) further extend reasoning capabilities to final action prediction, enabling robots and autonomous vehicles to generate precise actions from visual inputs and textual instructions [^32]. This advancement has led to the adoption of similar action generation mechanisms in autonomous driving, with approaches like $\pi$ 0 [^3] inspiring the development of action tokenizers that predicts precise trajectories [^37].

However, current VLA approaches in autonomous driving typically face two critical limitations that hinder their practical deployment: First, existing trajectory generation framework often produce physically infeasible outputs. Existing approaches that directly generate textual commands or waypoints via VLMs frequently produce physically-infeasible outputs and exhibit model collapse. While intermediate representations like meta-actions or latent action tokens have been proposed to mitigate these issues, these designs violate the end-to-end optimization principle and significantly increase model complexity overhead. Second, current systems demonstrate inadequate reasoning capabilities for complex driving scenarios. Since most methods employ simplistic reasoning strategies, they fail to account for both complicated road condition and vehicle kinematic constraints, resulting in predicted trajectories significantly deviates from real-world requirements. These limitations underscore the critical need for a novel VLA framework that balances architectural simplicity, robust contextual understanding, and strict physical constraints.

To overcome these challenges, we propose AutoDrive-R², a novel VLA framework that enhances both reasoning quality and physical feasibility through a two-stage training approach. Our key insight is that effective autonomous driving requires structured reasoning processes that can be systematically validated and refined. Specifically, to address the inadequacy of contextual reasoning for complex driving scenarios, we first construct nuScenesR²-6K, a chain-of-thought (CoT) dataset for supervised fine-tuning (SFT). nuScenesR²-6K is the first dataset in autonomous driving that stimulates both reasoing and self-reflection capabilities for VLA models. Unlike prior autonomous driving datasets, nuScenesR²-6K provides not only ground-truth trajectories but also reasoning and self-reflection steps, ensuring both the correctness and causal plausibility of driving behavior.

Furthermore, to resolve the challenge of physically infeasible trajectory generation, we further develop a physics-grounded reward framework tailored to group relative policy optimization (GRPO) of autonomous driving tasks. By explicitly incorporates spatial alignment, vehicle dynamic and temporal smoothness constraints into consideration, our physics-grounded reward enables reinforcement learning to adapt to diverse driving scenarios and vehicle dynamics while maintaining physical feasibility and motion comfort. Comprehensive experiments on nuScenes and Waymo benchmarks demonstrate that AutoDrive-R² achieves state-of-the-art performance. Our key contributions include:

- We introduce AutoDrive-R², a novel VLA framework that enables semantic reasoning with self-reflection step and trajectory planning from visual information and language instructions.
- We propose nuScenesR <sup>2</sup> -6K, an innovative CoT dataset adopting a four-step logic chain with self-reflection to help build foundational perception capabilities after SFT.
- We propose a RL-based post-training method based on GRPO, which incorporates physics-grounded rewards as constraint to refine planning trajectory for diverse scenes.

## 2 Related Work

### 2.1 Autonomous Driving

In recent years, autonomous driving has evolved from traditional modular pipelines—comprising perception, online mapping, prediction, and planning—toward end-to-end learning-based approaches [^11] [^13] [^26]. UniAD [^11] was the first to integrate all sub-tasks into a cascaded model, achieving significant improvements over traditional modular approaches. Some methods [^13] [^33] [^7] extract bird’s-eye view features and predict planning trajectories via multiple stages of interaction modeling.

With the emergence of vision-language models (VLMs), researchers have increasingly integrated both large language models (LLMs) and VLMs into autonomous driving to enhance overall system performance. Several approaches [^31] [^24] incorporate pretrained LLMs to generate driving actions along with interpretable textual explanations. Furthermore, DriveVLM [^27] incorporates specialized reasoning modules to improve situational understanding, while DriveMM [^12] processes multi-view video and image inputs to enhance generalization in vehicle control. DriveMLM [^29] introduces a behavior planning module to generate optimal driving decisions with rationales.

Moreover, the recent success of vision-language-action (VLA) models in robotics offers a new perspective for autonomous driving. DriveMoE [^32] builds on the embodied AI framework $\pi$ 0 [^3] and introduces Action-MoE by training routing networks to activate expert modules for diverse driving behaviors. Furthermore, OpenDriveVLA [^36] proposes an agent-environment-ego interaction model for precise trajectory planning. AutoVLA [^37] directly predicts semantic reasoning and trajectory plans from visual inputs and language prompts.

### 2.2 General VLMs

In recent years, the success of large language models (LLMs) [^34] [^4] [^28] has motivated researchers to extend them into vision-language models (VLMs) [^21] [^38] [^8], which integrate textual and visual data for richer multimodal representation. CLIP [^21], a pioneering work, combines image and text features using an image encoder and a text encoder to predict correct pairings of image-text examples via a zero-shot learning strategy. Similarly, BLIP [^17] and BLIP-2 [^18] are trained using an image-text contrastive (ITC) loss to align vision and language representations, along with an image-text matching (ITM) loss to distinguish between positive and negative image-text pairs, thereby enhancing visual representation grounded in textual context. Inspired by these methods, many VLMs—such as LLaVA [^20] and Qwen2.5VL [^2] —further enhance the robustness and representation capabilities of pretrained vision encoders by integrating a large language model as the text encoder (e.g., LLaMA [^28]). OmniGen2 [^30] represents another notable VLM, employing two distinct decoding pathways for text and image modalities with unshared parameters and a decoupled image tokenizer. Notably, DeepSeek-V3 [^19] introduces a robust Mixture-of-Experts (MoE) language model that employs an auxiliary-loss-free strategy for load balancing, achieving both efficient and cost-effective inference.

### 2.3 Reinforcement Learning for Post-Training

Reinforcement learning (RL) has been widely adopted in large language models, and researchers have found that reinforcement learning from human feedback (RLHF) [^16] can significantly enhance their reasoning capabilities. Among these methods, Proximal Policy Optimization (PPO) [^23] was initially used in simulated robotic locomotion and Atari game environments, and later employed by OpenAI to fine-tune GPT [^1], resulting in substantial improvements in text generation tasks. Unlike conventional RLHF methods, direct Preference Optimization (DPO) introduces a new reward model parameterization that eliminates the need for sampling during fine-tuning [^22]. Reward Fine-Tuning (RFT) [^35] is another RL-based approach that demonstrates strong performance in mathematical reasoning tasks. Furthermore, Guided Reward Policy Optimization (GRPO) [^25] effectively improves the reasoning capabilities of LLMs without relying on external toolkits or voting mechanisms. DeepSeek-R1 [^10], for example, leverages GRPO to fine-tune its model and achieves superior performance compared to existing methods. Group policy gradient (GPG) is a minimalist RL method that enhances the reasoning ability of large language models without relying on supervised fine-tuning or complex tricks and shows strong performance across various tasks [^9]. Inspired by these approaches, recent work [^6] adopts similar fine-tuning strategies to enhance the reasoning capabilities of multimodal models.

![[x2 11.png|Refer to caption]]

Figure 2: Pipeline of AutoDrive-R². We adopt a two-stage training process. The first stage introduce an innovative CoT dataset named nuScenesR²-6K for SFT. The nuScenesR²-6K adopts a four-step logical chain with self-reflection to generate valuable chain-of-thought data. The second stage propose an novel physics-grounded reward framework within the GRPO algorithm for RL, which incorporates spatial alignment, vehicle dynamic, and temporal smoothness for reliable trajectory planning.

## 3 Methodology

### 3.1 Overview

In this section, we present an overview of AutoDrive-R². The target of trajectory planning task requires the model to forecast a vehicle’s future motion based on its historical sensor data and contextual information. Formally, given a sequence of historical vehicle states $H$ (including position, acceleration, velocity, steering angle, etc.) and its camera image $F$, the model $M$ outputs predicted bird’s-eye view (BEV) trajectory coordinates $T$ over the next 3 seconds at 0.5-second intervals, defined as $T=M(H,F)$.

As depicted in Fig. 2, our training process contains two stages. In the first stage, we construct a high-quality dataset nuScenesR <sup>2</sup> -6K for cold start to build cognitive bridges between input information and output trajectories through a four-step logical chain with self-reflection for validation. In the second stage, we employ a physics-based reinforcement learning framework that integrates spatial alignment, vehicle dynamic and temporal smoothness criteria to ensure physically feasible and safe trajectory generation.

### 3.2 Logistic CoT Dataset with Self-Reflection

The success of VLA models in autonomous driving critically depends on their ability to produce both interpretable reasoning and physically feasible actions. However, existing training approaches often fail to establish this dual requirement, leading to models that either lack explainable decision-making processes or generate unrealistic trajectories. To investigate this challenge, we initially explored direct reinforcement learning optimization for trajectory planning, following recent advances in reasoning-based RL [^10]. However, preliminary experiments revealed that models trained exclusively on RL exhibited significant degradation in trajectory planning compared to models pre-trained with SFT before RL. Therefore, we constructed a high-quality cold-start dataset named nuScenesR²-6K in advanced to cultivate the model’s foundational understanding of trajectory planning.

To this end, we manually annotate 6,000 image-trajectory pairs from the nuScenes training set, followed by leveraging the advanced Qwen2.5-VL-72B model to synthesize chain-of-thought reasoning sequences. Specifically, as illustrated in Fig. 2 (a), given the front-view image combined with the vehicle’s historical states as input and corresponding ground truth trajectories as output, we predefine a specific CoT prompt to guide the model in constructing reasoning sequences in the following format: *"<think>thinking process here</think><answer>($x_{1},y_{1}$), …,($x_{n},y_{n}$)</answer>"*.

Moreover, we observe that many existing approaches rely on universal prompts for problem-to-answer reasoning, lacking structured guidance for rational analysis. While this strategy proves effective for simple tasks, it frequently fails when confronted with complex mathematical or logical problems. To address this limitation, our CoT prompt design systematically decomposes trajectory planning into three interdependent reasoning stages:

1. Image-Driven Analysis: Establishing foundational scene understanding (e.g., obstacle and lane localization, traffic sign detection) to anchor subsequent reasoning.
2. Physics-based Calculation: Leveraging kinematic equations (e.g., angular momentum conservation) to translate abstract observations into quantifiable predictions.
3. Contextual Logic Synthesis: Integrating domain-specific knowledge (e.g., intersection traffic rules) to ensure predictions align with real-world driving regulations.

To further enhance robustness and the correctness of answer, we explicitly introduce a self-reflection phase as the fourth step, inspired by mathematical reasoning frameworks that validate conclusions through backward-checking. This allows the model to verify the coherence of its reasoning and correct potential contradictions. Therefore, our prompt implements a four-step logic chain:

|  | Visualization → Calculation → Logic → Reflection |  |
| --- | --- | --- |

which delivers both systematic and error-resilient reasoning. Further details are shown in supplementary materials.

Ultimately, the nuScenesR²-6K dataset is adpoted to for supervised fine-tuning of Qwen2-VL-7B model, thus yielding our stage-1 model. The pre-trained model can effectively achieve trajectory planning via a structured, step-by-step reasoning mechanism with self-reflection.

### 3.3 Group Relative Policy Optimization (GRPO)

We follow the GRPO algorithm [^14] to train the model. Unlike traditional approaches that rely on critic networks to estimate value functions, GRPO introduces a pairwise comparison mechanism among candidate response. This design not only simplifies the architecture but also reduces computational overhead during training. The methodology begins by generating $G$ distinct candidate responses $o=\{o_{1},\ldots,o_{G}\}$ for a given input question $q$ through policy sampling. For our specific task, we implement two rule-based verifiable reward functions to assess response quality:

#### 3.3.1 Accuracy Reward

To better adapt to trajectory planning task, we define a physics-grounded accuracy rewards $r_{acc}$ which integrates spatial, kinematic, and temporal constraints for evaluation. Details are specified in the following section.

#### 3.3.2 Format Reward

The format reward $r_{acc}$ enforces strict adherence to the required output format. The model must produce responses in the form: *"<think>thinking process here</think><answer>($x_{1},y_{1}$), …,($x_{n},y_{n}$)</answer>"*. A value of 1 is assigned if the format is correct, otherwise 0. In summary, the total reward for a response $o_{i}$ is defined by:

$$
r_{i}=r_{\text{acc}}^{i}+r_{\text{format}}^{i}.
$$

To quantify the relative quality of all response $\{r_{1},\ldots,r_{G}\}$, GRPO normalizes these scores by subtracting the group mean and dividing by the standard deviation. Consequently, the advantage for each response can be formulated by:

$$
A_{i}=\frac{r_{i}-\text{mean}(\{r_{i}\}_{i=1}^{G})}{\text{std}(\{r_{i}\}_{i=1}^{G})},
$$

where $A_{i}$ is the relative advantage of the $i$ -th answer. Then the optimization objective further incorporates a regularization term to ensure the updated policy $\pi_{\theta}$ remains close to the original reference policy $\pi_{\text{ref}}$. This is achieved by adding a KL-divergence term $D_{\text{KL}}(\cdot\,\|\,\cdot)$ to the loss function:

$$
\displaystyle J_{\mathrm{GRPO}}(\theta)=\mathbb{E}_{q\sim P(Q),\{o_{i}\}^{N}_{i=1}\sim\pi_{\theta_{\text{old}}}{(O|q)}}
$$
 
$$
\displaystyle\left[\sum_{i=1}^{G}\frac{\pi_{\theta}(o_{i}|q)}{\pi_{\theta_{\text{old}}}(o_{i}|q)}\cdot A_{i}-\beta D_{\text{KL}}(\pi_{\theta}\|\pi_{\text{ref}})\right],
$$

where $\beta$ acts as a hyperparameter to balance the trade-off between exploration and stability during optimization.

### 3.4 Physics-Grounded Accuracy Rewards

In autonomous driving tasks, traditional reward function designs often focus solely on trajectory position error, while neglecting the complex constraints in geometric, dynamical, and temporal dimensions. To address this issue, we propose a physics-grounded reward framework that integrates spatial alignment, vehicle dynamics, and temporal continuity to comprehensively guide the model in generating safe, feasible, and comfortable driving strategies. This multi-dimensional approach not only ensures geometric accuracy but also explicitly incorporates the physical limitations of real-world vehicles and the perceptual requirements for motion smoothness, creating a holistic optimization objective.

#### 3.4.1 Spatial Alignment: Balancing Maneuverability

The foundation of any trajectory reward function lies in its ability to align predicted paths with target routes. We define a spatial accuracy term $r_{\text{pos}}$ as the mean squared Euclidean distance between predicted and ground-truth coordinates:

$$
r_{\text{pos}}=\frac{1}{N}\sum_{i=1}^{N}\left((x^{i}-x_{\text{gt}}^{i})^{2}+(y^{i}-y_{\text{gt}}^{i})^{2}\right),
$$

where $N$ denotes the number of time steps, $x^{i},y^{i}$ represent predicted coordinates at the $i$ -th time step, while $x_{\text{gt}}^{i},y_{\text{gt}}^{i}$ correspond to the ground-truth values. This formulation prioritizes global path adherence by penalizing deviations across all time steps, ensuring the vehicle remains on the intended route. However, focusing only on minimizing position error may produce physical-implausible results. For instance, strictly following the shortest path might bring about abrupt steering or acceleration, which not only violate vehicle kinematics but also compromise passenger comfort. To balance geometric precision with practical feasibility, we introduce additional constraints derived from vehicle dynamics.

#### 3.4.2 Vehicle Dynamics: Bridging Perception & Control

Autonomous driving systems must respect the real-world physical limitations, which are governed by steering kinematics and longitudinal dynamics. Ignoring them may result in trajectories that are impossible to execute (e.g., requiring infinite torque for abrupt steering changes) or uncomfortable for passengers. To ensure kinematic feasibility, we penalize deviations in steering angles through the following term $r_{\text{ste}}$:

$$
r_{\text{ste}}=\frac{1}{N}\sum_{j=1}^{N}\left(\theta^{j}-\theta_{\text{gt}}^{i}\right)^{2},
$$

where $\theta^{j}$ and $\theta_{gt}^{j}$ respectively denotes the predicted and corresponding ground-truth steering angle at $j$ -th time step. Additionally, we address unphysical acceleration/braking patterns by introducing an additional velocity constraint term:

$$
r_{\text{vel}}=\frac{1}{N}\sum_{k=1}^{N}\left(v^{k}-v_{\text{gt}}^{k}\right)^{2},
$$

where $v^{k}$ and $v_{gt}^{j}$ respectively represents the predicted and corresponding ground-truth velocity at the $k$ -th time step. In summary, both $r_{\text{ste}}$ and $r_{\text{vel}}$ enforce compliance with vehicle-specific constraints, ensuring generated trajectories are both physically realizable and socially acceptable in mixed traffic scenarios. These constraints explicitly bridge the gap between perception-driven planning and actuator-level control, ensuring that predicted trajectories align with the physical boundaries while maintaining ride quality.

#### 3.4.3 Temporal Smoothness: Navigation Reliability

Temporal discontinuities in trajectory predictions fundamentally undermine the reliability of autonomous driving systems. When steering or acceleration commands exhibit sudden jumps between time steps, the predicted trajectories may lose coherence, which further compromises the system’s ability to maintain stable, predictable motion patterns required for safe navigation. To address this, we introduce a temporal smoothness term $r_{\text{tem}}$ that penalizes rapid variations in consecutive control signals:

$$
r_{\text{tem}}=\frac{1}{N}\sum_{j=1}^{N}\left(\theta^{j}-\theta^{j-1}\right)^{2}+\frac{1}{N}\sum_{k=1}^{N}\left(v^{k}-v^{k-1}\right)^{2}.
$$

Such design ensures temporal coherence of predicted trajectories. By explicitly constraining the rate of change in steering and velocity, the reward function filters out unstable oscillations that could destabilize the vehicle’s state estimation. This regularization effect strengthens the model’s ability to generalize across diverse driving scenarios while maintaining safety margins during execution.

#### 3.4.4 Integrated Reward Function

The final reward function synthesizes all dimensions with learnable weights:

$$
r_{\text{acc}}=\lambda_{\text{pos}}\cdot r_{\text{pos}}+\lambda_{\text{ste}}\cdot r_{\text{ste}}+\lambda_{\text{vel}}\cdot r_{\text{vel}}+\lambda_{\text{tem}}\cdot r_{\text{tem}}.
$$

Here, $\lambda_{\text{pos}},\lambda_{\text{ste}},\lambda_{\text{vel}},\lambda_{\text{tem}}$ are learnable coefficients that balance trade-offs between competing objectives. We set them all equal one in experiment. This holistic formulation ensures the model generates trajectories that are geometrically accurate, dynamically feasible, and temporally smooth, addressing the multifaceted challenges of autonomous driving.

## 4 Experiment

### 4.1 Experimental Settings

#### 4.1.1 Datasets

For training, we adopt nuScenesR <sup>2</sup> -6K dataset. This dataset contains 6k image-trajectory sample pairs, each includes a front-view image and a 3-second trajectory planning with 0.5-second intervals. The Qwen2.5-VL-7B model is fine-tuned on these samples for SFT to establish foundational perception capabilities before RL. For evaluation, our mtehod is tested on nuScenes and Waymo datasets, both offering comprehensive autonomous driving data. The nuScenes dataset contains 1,000 urban driving scenes with six synchronized camera views to support planning tasks. The Waymo dataset includes 4,021 driving segments, capturing eight camera views and ego-vehicle trajectories.

#### 4.1.2 Details

We implement experiments on both Qwen2.5-VL-3B and Qwen2.5-VL-7B models. In both stages, the learning rate is set to 5e-7 with an accumulated total batch size of 1. The GRPO is configured with a maximum completion length of 4,096 tokens and samples 6 responses per input.

#### 4.1.3 Evaluation Metrics

We adopt the L2 distance (in meters) between the predicted and ground truth trajectories at future time horizons of 1s, 2s, and 3s, along with the average L2 error. For all models, we utilize the official checkpoints and conduct evaluations under the same evaluation codes.

<table><tbody><tr><th rowspan="2">Method</th><td colspan="4">L2 Error (m) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td></tr><tr><th colspan="5"><em>Open-source Generalist Vision Language Models</em></th></tr><tr><th>Llama-3.2-11B-Vision</th><td>1.54</td><td>3.31</td><td>3.91</td><td>2.92</td></tr><tr><th>DeepSeek-VL2-16B</th><td>0.66</td><td>1.68</td><td>2.92</td><td>1.75</td></tr><tr><th>LLaMA-3.2-11B-Vision</th><td>0.52</td><td>1.42</td><td>2.68</td><td>1.54</td></tr><tr><th>Qwen-2.5-VL-3B</th><td>2.69</td><td>4.16</td><td>5.78</td><td>4.21</td></tr><tr><th>Qwen-2.5-VL-7B</th><td>0.46</td><td>1.33</td><td>2.55</td><td>1.45</td></tr><tr><th colspan="5"><em>Training-based Driving Specialists (Existing Methods)</em></th></tr><tr><th>UniAD</th><td>0.42</td><td>0.64</td><td>0.91</td><td>0.66</td></tr><tr><th>VAD</th><td>0.17</td><td>0.34</td><td>0.60</td><td>0.37</td></tr><tr><th>BEV-Planner</th><td>0.16</td><td>0.32</td><td>0.57</td><td>0.35</td></tr><tr><th>Ego-MLP</th><td>0.15</td><td>0.32</td><td>0.59</td><td>0.35</td></tr><tr><th colspan="5"><em>Ours and Key Competitors (Specialized Driving Models)</em></th></tr><tr><th>DriveVLM</th><td>0.18</td><td>0.34</td><td>0.68</td><td>0.40</td></tr><tr><th>OmniDrive</th><td>0.14</td><td>0.29</td><td>0.55</td><td>0.33</td></tr><tr><th>DriveVLM-Dual</th><td>0.15</td><td>0.29</td><td>0.48</td><td>0.31</td></tr><tr><th>EMMA</th><td>0.14</td><td>0.29</td><td>0.54</td><td>0.32</td></tr><tr><th>EMMA+</th><td>0.13</td><td>0.27</td><td>0.48</td><td>0.29</td></tr><tr><th>Imprompt-VLA</th><td>0.13</td><td>0.27</td><td>0.53</td><td>0.30</td></tr><tr><th>AutoDrive-R² 3B</th><td>0.35</td><td>0.49</td><td>0.62</td><td>0.49</td></tr><tr><th>AutoDrive-R² 7B</th><td>0.13</td><td>0.19</td><td>0.25</td><td>0.19</td></tr></tbody></table>

Table 1: Trajectory planning L2 errors on nuScenes dataset.

### 4.2 Evaluation Results

#### 4.2.1 Results on nuScenes Datasets

Tab. 1 compares the prediction errors among our method and existing approaches on the nuScenes dataset. Notably, our approach consistently achieves the best performance across all time intervals, surpassing current leading methods such as EMMA+, which are trained on substantially larger internal datasets with 103k scenarios. In contrast, our training data consists of only 6k curated CoT samples for stage 1 and another 6k for stage 2, approximately 11.65% the size of EMMA+’s dataset. Furthermore, our model demonstrates significant improvements over Qwen2-VL-7B, reducing L2 errors by 86.9%, despite having a considerably smaller parameter count.

#### 4.2.2 Zero-shot performance on Waymo Datasets

Moreover, Tab. 2 demonstrates the robust zero-shot capabilities of our model. Specifically, our method respectively reduces L2 errors by 33.3% and 90.7% compared to the latest EMMA+ method and Qwen2-VL-72B baseline models. Overall, our model consistently delivers precise trajectory predictions across multiple datasets, establishing its state-of-the-art performance and generalization capability.

<table><tbody><tr><th rowspan="2">Method</th><td colspan="4">L2 Error (m) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td></tr><tr><th colspan="5"><em>Generalist VLMs + Specialized Driving Models</em></th></tr><tr><th>Qwen-2.5-VL-3B</th><td>2.98</td><td>5.05</td><td>7.38</td><td>5.14</td></tr><tr><th>Qwen-2.5-VL-7B</th><td>1.66</td><td>1.82</td><td>2.92</td><td>2.13</td></tr><tr><th>DriveVLM</th><td>0.17</td><td>0.34</td><td>0.75</td><td>0.42</td></tr><tr><th>EMMA</th><td>0.12</td><td>0.28</td><td>0.61</td><td>0.34</td></tr><tr><th>EMMA+</th><td>0.11</td><td>0.25</td><td>0.53</td><td>0.30</td></tr><tr><th>AutoDrive-R² 3B</th><td>0.23</td><td>0.36</td><td>0.51</td><td>0.37</td></tr><tr><th>AutoDrive-R² 7B</th><td>0.11</td><td>0.19</td><td>0.29</td><td>0.20</td></tr></tbody></table>

Table 2: Trajectory planning L2 errors on Waymo dataset.

![[x3 12.png|Refer to caption]]

Figure 3: Qualitative comparison of trajectory planning performance across Qwen2.5-VL-7B, EMMA+, and our AutoDrive-R² on the nuScenes dataset. Note that blue lines denote predicted trajectories while green lines represent ground truth trajectories.

#### 4.2.3 Model Size

In Tab. 1 and 2, we compare 3B and 7B variants of Qwen2.5-VL within our two-stage training framework to analyze impact of different model size. While the 7B model achieves superior performance with an average L2 error of 0.19m, the 3B version demonstrates a notable improvement than its baseline. The disparity highlights that larger models inherently capture more complex patterns, but the two-stage framework (SFT + GRPO) effectively compensates for the 3B model’s limited capacity by enforcing strict trajectory constraints and contextual logic synthesis.

#### 4.2.4 Visualization

In Fig. 4, we present a comparative analysis of our method against other approaches in the nuScenes dataset. Notably, Qwen2.5-VL-7B fails to generate accurate predictions in specific scenarios (e.g., (b) and (d)), whereas EMMA+ exhibits significant trajectory deviation. In contrast, our method consistently achieves more reliable and physically feasible trajectory planning under varying illumination environments and complex motion patterns.

<table><tbody><tr><th rowspan="2">Method</th><td colspan="4">L2 Error (m) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td></tr><tr><th>Qwen2.5-VL-7B</th><td>0.46</td><td>1.33</td><td>2.55</td><td>1.45</td></tr><tr><th>Qwen2.5-VL-7B + <em>SFT</em></th><td>0.17</td><td>0.27</td><td>0.36</td><td>0.27</td></tr><tr><th>Qwen2.5-VL-7B + <em>RL</em></th><td>0.21</td><td>0.33</td><td>0.44</td><td>0.33</td></tr><tr><th><em>SFT:</em> w/o. Four.</th><td>0.19</td><td>0.25</td><td>0.32</td><td>0.25</td></tr><tr><th><em>SFT:</em> w/o. Self.</th><td>0.17</td><td>0.23</td><td>0.29</td><td>0.23</td></tr><tr><th><em>RL:</em> w/o. <math><semantics><msub><mi>r</mi> <mtext>pos</mtext></msub> <annotation>r_{\text{pos}}</annotation></semantics></math></th><td>0.32</td><td>0.53</td><td>0.72</td><td>0.53</td></tr><tr><th><em>RL:</em> w/o. <math><semantics><msub><mi>r</mi> <mtext>ste</mtext></msub> <annotation>r_{\text{ste}}</annotation></semantics></math></th><td>0.14</td><td>0.20</td><td>0.27</td><td>0.21</td></tr><tr><th><em>RL:</em> w/o. <math><semantics><msub><mi>r</mi> <mtext>vel</mtext></msub> <annotation>r_{\text{vel}}</annotation></semantics></math></th><td>0.15</td><td>0.21</td><td>0.29</td><td>0.22</td></tr><tr><th><em>RL:</em> w/o. <math><semantics><msub><mi>r</mi> <mtext>tem</mtext></msub> <annotation>r_{\text{tem}}</annotation></semantics></math></th><td>0.15</td><td>0.23</td><td>0.34</td><td>0.24</td></tr><tr><th>AutoDrive-R² 7B</th><td>0.13</td><td>0.19</td><td>0.25</td><td>0.19</td></tr></tbody></table>

Table 3: Ablation studies of trajectory planning L2 errors on nuScenes dataset to validate each proposed component.

<table><thead><tr><th rowspan="2">Method</th><th colspan="4">L2 Error (m) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th></tr><tr><th>1s</th><th>2s</th><th>3s</th><th>Avg.</th></tr></thead><tbody><tr><th>w/. num = 2</th><td>0.16</td><td>0.23</td><td>0.31</td><td>0.23</td></tr><tr><th>w/.num = 4</th><td>0.14</td><td>0.20</td><td>0.26</td><td>0.20</td></tr><tr><th>w/. num = 6</th><td>0.13</td><td>0.19</td><td>0.25</td><td>0.19</td></tr><tr><th>w/. num = 8</th><td>0.13</td><td>0.19</td><td>0.25</td><td>0.19</td></tr></tbody></table>

Table 4: Ablation studies of L2 errors on nuScenes dataset to validate different number of generation per input.

### 4.3 Ablation Studies

#### 4.3.1 Training Stages

Drawing inspiration from DeepSeek-R1-Zero, we first attempt to train the model solely through RL. As shown in Tab. 3, the model purely trained on RL (7B + *RL*) underperforms that of SFT (7B + *SFT*) by 22.2% in average L2 error. We attribute this gap to the model’s inability to establish structured reasoning chains, as RL struggles to explore the high-dimensional reasoning space required for multi-step calculations and contextual logic synthesis. This observation validates the necessity of our two-stage training.

#### 4.3.2 Supervised Fine-tuning (SFT)

In the first stage, the baseline Qwen2.5-VL-7B (7B) achieves an average L2 error of 1.45m, whereas the SFT model (7B + *SFT*) trained on the nuScenesR²-6K dataset reduces this to 0.27m, demonstrating an 81.4% improvement. This significant enhancement highlights the effectiveness of adopting SFT training in establishing foundational reasoning capabilities. Moreover, removing four-step reasoning structure (w/o. Four.) increases the error to 0.25m, indicating a 31.5% degradation compared to AutoDrive-R². Similarly, eliminating self-reflection (w/o. Self.) results in 0.23m error, representing a 21.1% decline relative to AutoDrive-R². This emphasize the interdependence of both four-step logical chain and self-reflection mechanism in constructing high-quality CoT dataset.

#### 4.3.3 Reinforcement Learning (RL)

In the second stage, we evaluate the contribution of individual reward components within the physics-grounded framework of AutoDrive-R². Specifically, spatial alignment is critical for maintaining global geometric path accuracy, as its removal (w/o. $r_{\text{pos}}$) increases the error to 0.53m, much higher than the full model. Moreover, steering angle regulation ensures kinematic feasibility by penalizing abrupt changes in steering adjustments, and its absence (w/o. $r_{\text{ste}}$) leads to a 10.5% degradation (0.21m). Additionally, velocity consistency constraints ensure adherence to target speed profiles by penalizing deviations in predicted velocity from ground-truth values, and their exclusion (w/o. $r_{\text{vel}}$) raises the error to 0.22m. Finally, temporal smoothness penalties suppress unstable control patterns by penalizing abrupt changes in steering and velocity across time steps, and their removal (w/o. $r_{\text{tem}}$) results in a 26.3% increase in error (0.24m). By integrating all four components into our physics-grounded reward, AutoDrive-R² achieves an optimal 0.19m L2 error, confirming the necessity of each element in achieving spatial, kinematic and temporal criteria for reliable trajectory planning.

#### 4.3.4 Additional Experiments and Analysis

We conducted an experiment to analyze the impact of different number of generation during the second-stage RL training (GRPO). As shown in Tab. 4, increasing the number of candidate responses from 2 to 6 consistently reduces the L2 error across all time steps. Specifically, when generating 6 responses per input (w/. num = 6), the model achieves the lowest average L2 error of 0.19 m, outperforming the 0.20 m and 0.23 m results for (w/. num = 4) and (w/. num = 2), respectively. However, the marginal gains diminish beyond (w/. num = 6), indicating a trade-off between computational cost and performance improvement. Therefore, we opt for num = 6 to balances accuracy and efficiency.

## 5 Conclusion

In this work, we propose AutoDrive-R², a novel VLA framework designed for reasoning-guided trajectory planning in autonomous driving. AutoDrive-R² effectively balances semantic understanding with real-world constraints through a two-stage training framework: (1) a SFT stage adopting the nuScenesR²-6K dataset, which employs a four-step CoT process to cultivate structured reasoning and self-reflection for validation, and (2) a RL stage leveraging GRPO training to refine trajectory planning under physics-grounded rewards. Experiments validate the effectiveness of AutoDrive-R², achieving SOTA performance on both nuScenes and Waymo datasets, demonstrating strong zero-shot generalization. Future efforts will focus on multi-agent coordination and real-time sensor fusion integration to further improve adaptability in complicated environments.

## References

  

Supplementary Material  

## Appendix A Summary

This supplementary material provides more specific details of our method. The first section presents the detailed input prompts used in the SFT stage to generate CoT data. The second section exhibits more visualization results of trajectory planning tasks. The third section introduces an "Aha Moment" analysis demonstrating the model’s self-correction capability during reasoning. The fourth section presents the mathematical derivation of physics-grounded reward functions. The final section presents more hyperparameter configurations for experiments.

### A.1 Detailed Prompts to Generate CoT Data

During the supervised fine-tuning (SFT) stage of AutoDrive-R², we designed a structured input prompt to generate high-quality chain-of-thought (CoT) data for the nuScenesR²-6K dataset. The prompt template is as follows:

[⬇](data:text/plain;base64,IyMjIFByb21wdDoKWW91IGFyZSBnaXZlbiBhbiBpbWFnZSwgYSBkcml2aW5nLXJlbGF0ZWQgcXVlc3Rpb24sIGFuZCBpdHMgYW5zd2VyLiBHZW5lcmF0ZSBhIGZvdXItc3RhZ2UgcmVhc29uaW5nIHByb2Nlc3Mgd2l0aCBleHBsaWNpdCBtYXRoZW1hdGljYWwgbW9kZWxpbmcgYW5kIHNlbGYtdmFsaWRhdGlvbi4gRW5nYWdlIGluIGFuIGludGVybmFsIGRpYWxvZ3VlIHVzaW5nIGV4cHJlc3Npb25zIHN1Y2ggYXMgJ2xldCBtZSB0aGluaycsICd3YWl0JywgJ0htbScsICdvaCwgSSBzZWUnLCAnbGV0J3MgYnJlYWsgaXQgZG93bicsIGV0Yywgb3Igb3RoZXIgbmF0dXJhbCBsYW5ndWFnZSB0aG91Z2h0IGV4cHJlc3Npb25zLiBJdCdzIGVuY291cmFnZWQgdG8gaW5jbHVkZSBzZWxmLXJlZmxlY3Rpb24gb3IgdmVyaWZpY2F0aW9uIGluIHRoZSByZWFzb25pbmcgcHJvY2Vzcy4KCiMjIyBJbnB1dCBGb3JtYXQ6Ci0gU3lzdGVtIEluc3RydWN0aW9uczoge29yaWdpbmFsX3Rhc2t9Ci0gUGFzdCBWZWhpY2xlIFN0YXR1czoge29yaWdpbmFsX2luZm9ybWF0aW9ufQotIFByZWRpY3Rpb24gVGFzazoge29yaWdpbmFsX3Byb2JsZW19Ci0gQW5zd2VyOiB7b3JpZ2luYWxfc29sdXRpb259CgojIyMgT3V0cHV0IEZvcm1hdDoKIyMjIDEuIFZpc3VhbCBBbmFseXNpcwoiSW1hZ2UgYW5hbHlzaXMgcmVzdWx0czoKLSBWZWhpY2xlJ3MgaW50ZW5kZWQgZGlyZWN0aW9uOiBMZWZ0IHR1cm4gKHN0ZWVyaW5nIHdoZWVsIGFuZ2xlOiBcdGhldGEgcmFkKQotIE9ic3RhY2xlcyBhaGVhZDoKICAqIENhciBkZXRlY3RlZCBhaGVhZCAobW92aW5nIHJpZ2h0L2xlZnQvc3RyYWlnaHQpCiAgKiBQZWRlc3RyaWFuIGNyb3NzaW5nIHJvYWQgKGxlZnQvcmlnaHQgc2lkZSkKLSBUcmFmZmljIHNpZ25hbDogc2lnbmFsX3N0YXR1cyBkZXRlY3RlZCAocmVkIC8gZ3JlZW4gLyB5ZWxsb3cpIgoKIyMjIDIuIE1vdGlvbiBNb2RlbGluZwoiVXNpbmcgaGlzdG9yaWNhbCBkYXRhIGluIFBhc3QgVmVoaWNsZSBTdGF0dXMgd2l0aCBOIHRpbWUgcG9pbnRzOgokdF8xOiBbeD14XzEsIHk9eV8xXSwgdj12XzFtL3MsIGE9KGFfe3hfMX0sIGFfe3lfMX0pbS9zXjIkCiR0XzI6IFt4PXhfMiwgeT15XzJdLCB2PXZfMm0vcywgYT0oYV97eF8yfSwgYV97eV8yfSltL3NeMiQKLi4uCiR0X246IFt4PXhfbiwgeT15X25dLCB2PXZfbm0vcywgYT0oYV97eF9ufSwgYV97eV9ufSltL3NeMiQKCkNhbGN1bGF0aW9uczoKLSBBdmVyYWdlIGFjY2VsZXJhdGlvbjoKICAkYV97eF97YXZnfX0gPSAoXHN1bSBhX3t4X2l9KS9OID0gYV97eF97YXZnfX1tL3NeMiAkCiAgJGFfe3lfe2F2Z319ID0gKFxzdW0gYV97eV9pfSkvTiA9IGFfe3lfe2F2Z319bS9zXjIgJAotIFZlbG9jaXR5IHByZWRpY3Rpb246CiAgJHZfeCA9IHZfbiArIGFfe3hfe2F2Z319IFx0aW1lcyBcZGVsdGFfdCA9IHZfe3QwfSArIGFfe3hfe2F2Z319IFx0aW1lcyBcZGVsdGFfdCAkCiAgJHZfeSA9IHZfbiArIGFfe3lfe2F2Z319IFx0aW1lcyBcZGVsdGFfdCA9IHZfe3QwfSArIGFfe3lfe2F2Z319IFx0aW1lcyBcZGVsdGFfdCAkCi0gUG9zaXRpb24gcHJlZGljdGlvbjoKICAkeCh0KzEpID0geF9uICsgdl94IFx0aW1lcyBcZGVsdGFfdCArIDAuNSBcdGltZXMgYV97eF97YXZnfX0gXHRpbWVzIFxkZWx0YV90XjIgJAogICR5KHQrMSkgPSB5X24gKyB2X3kgXHRpbWVzIFxkZWx0YV90ICsgMC41IFx0aW1lcyBhX3t5X3thdmd9fSBcdGltZXMgXGRlbHRhX3ReMiAkCi0gTGF0ZXJhbCBvZmZzZXQ6ICRcZGVsdGFfeSA9IHYgXHRpbWVzIHRhbihcdGhldGEpID0gdl97dDB9IFx0aW1lcyB0YW4oXHRoZXRhKSAkCgojIyMgMy4gTG9naWNhbCBEZWR1Y3Rpb25zCiJTYWZldHkgY2hlY2s6Ci0gSWYgZm9sbG93aW5nIHRoaXMgdHJhamVjdG9yeSwgd2lsbCB0aGUgdmVoaWNsZToKICAqIFJ1biBhIHJlZCBsaWdodD8gJFxyaWdodGFycm93JCB5ZXMvbm8KICAqIENvbGxpZGUgd2l0aCBjYXIgYWhlYWQ/ICRccmlnaHRhcnJvdyQgeWVzL25vCiAgKiBIaXQgcGVkZXN0cmlhbiBjcm9zc2luZz8gJFxyaWdodGFycm93JCB5ZXMvbm8KLSBDb25jbHVzaW9uOiByZWNvbW1lbmRlZF9hY3Rpb24gKGUuZy4sICdTdG9wIGltbWVkaWF0ZWx5JywgJ1JlZHVjZSBzcGVlZCB0byA1bS9zJykiCgojIyMgNC4gU2VsZi1SZWZsZWN0aW9uIFZhbGlkYXRpb24KIlZhbGlkYXRpb246Ci0gQXNzdW1wdGlvbiBjaGVjazoKICAqIFByZWRpY3RlZCBwb3NpdGlvbiAoeD14X3ByZWQsIHk9eV9wcmVkKSByZXF1aXJlcyBhdmVyYWdlIHNwZWVkIG9mIHYgbS9zCiAgKiBJcyB0aGlzIHNwZWVkIGFjaGlldmFibGUgd2l0aCB2ZWhpY2xlJ3MgYWNjZWxlcmF0aW9uIGhpc3Rvcnk/ICRccmlnaHRhcnJvdyQgeWVzL25vCi0gQWRqdXN0bWVudDoKICAqIElmIG5vdCBmZWFzaWJsZSAkXHJpZ2h0YXJyb3ckIE1vZGlmeSB0cmFqZWN0b3J5IGJ5IHJlZHVjaW5nIHNwZWVkIG9yIGluY3JlYXNpbmcgc3RvcHBpbmcgZGlzdGFuY2Ui)

\### Prompt:

You are given an image, a driving-related question, and its answer. Generate a four-stage reasoning process with explicit mathematical modeling and self-validation. Engage in an internal dialogue using expressions such as ’let me think’, ’wait’, ’Hmm’, ’oh, I see’, ’let’s break it down’, etc, or other natural language thought expressions. It’s encouraged to include self-reflection or verification in the reasoning process.

\### Input Format:

\- System Instructions: {original\_task}

\- Past Vehicle Status: {original\_information}

\- Prediction Task: {original\_problem}

\- Answer: {original\_solution}

\### Output Format:

\### 1. Visual Analysis

"Image analysis results:

\- Vehicle’s intended direction: Left turn (steering wheel angle: \\theta rad)

\- Obstacles ahead:

\* Car detected ahead (moving right/left/straight)

\* Pedestrian crossing road (left/right side)

\- Traffic signal: signal\_status detected (red / green / yellow)"

\### 2. Motion Modeling

"Using historical data in Past Vehicle Status with N time points:

$t\_1: \[x=x\_1, y=y\_1\], v=v\_1m/s, a=(a\_{x\_1}, a\_{y\_1})m/s^2$

$t\_2: \[x=x\_2, y=y\_2\], v=v\_2m/s, a=(a\_{x\_2}, a\_{y\_2})m/s^2$

...

$t\_n: \[x=x\_n, y=y\_n\], v=v\_nm/s, a=(a\_{x\_n}, a\_{y\_n})m/s^2$

Calculations:

\- Average acceleration:

$a\_{x\_{avg}} = (\\sum a\_{x\_i})/N = a\_{x\_{avg}}m/s^2 $

$a\_{y\_{avg}} = (\\sum a\_{y\_i})/N = a\_{y\_{avg}}m/s^2 $

\- Velocity prediction:

$v\_x = v\_n + a\_{x\_{avg}} \\times \\delta\_t = v\_{t0} + a\_{x\_{avg}} \\times \\delta\_t $

$v\_y = v\_n + a\_{y\_{avg}} \\times \\delta\_t = v\_{t0} + a\_{y\_{avg}} \\times \\delta\_t $

\- Position prediction:

$x(t+1) = x\_n + v\_x \\times \\delta\_t + 0.5 \\times a\_{x\_{avg}} \\times \\delta\_t^2 $

$y(t+1) = y\_n + v\_y \\times \\delta\_t + 0.5 \\times a\_{y\_{avg}} \\times \\delta\_t^2 $

\- Lateral offset: $\\delta\_y = v \\times tan(\\theta) = v\_{t0} \\times tan(\\theta) $

\### 3. Logical Deductions

"Safety check:

\- If following this trajectory, will the vehicle:

\* Run a red light? $\\rightarrow$ yes/no

\* Collide with car ahead? $\\rightarrow$ yes/no

\* Hit pedestrian crossing? $\\rightarrow$ yes/no

\- Conclusion: recommended\_action (e.g., ’Stop immediately’, ’Reduce speed to 5m/s’)"

\### 4. Self-Reflection Validation

"Validation:

\- Assumption check:

\* Predicted position (x=x\_pred, y=y\_pred) requires average speed of v m/s

\* Is this speed achievable with vehicle’s acceleration history? $\\rightarrow$ yes/no

\- Adjustment:

\* If not feasible $\\rightarrow$ Modify trajectory by reducing speed or increasing stopping distance"

This structured prompt ensures nuScenesR²-6K dataset contains diverse and causally plausible reasoning process, which are critical for cultivating the model’s foundational perception and planning capabilities before RL fine-tuning.

![[x4 11.png|Refer to caption]]

Figure 4: Qualitative comparison of trajectory planning performance across Qwen2.5-VL-7B, EMMA+, and our AutoDrive-R² on the nuScenes dataset. Note that blue lines denote predicted trajectories while green lines represent ground truth trajectories.

![[x5 10.png|Refer to caption]]

Figure 5: Visualization comparison bewtwee Qwen2.5-VL-7B our AutoDrive-R² on the nuScenes dataset.

![[x6 9.png|Refer to caption]]

Figure 6: Visualization comparison bewtwee Qwen2.5-VL-7B our AutoDrive-R² on the nuScenes dataset.

### A.2 Visualization Results

Fig. 4 provides additional visualization results in trajectory planning tasks between our AutoDrive-R² and other methods on the nuScenes dataset. Notably, our method consistently outperforms other approaches in predicting both reliable and physically-feasible trajectories, demonstrating the state-of-the-art performance of our proposed method.

To further illustrate the advantages of our structured reasoning process, we present two representative comparisons in Fig. 5 and Fig. 6. These visualizations explicitly contrast the four-stage CoT reasoning (AutoDrive-R²) with the single-step reasoning of Qwen2.5-VL-7B. As can be seen, the Qwen2.5-VL model predicts a trajectory that deviates from the lane marking due to its simplified reasoning approach. The model’s single-stage analysis fails to account for the vehicle’s kinematic constraints and results in an unrealistic leftward drift. In contrast, AutoDrive-R²’s four-stage process systematically validates its predictions. These examples demonstrate how our structured CoT framework enables systematic error detection and correction, resulting in trajectories that are both geometrically accurate and physically feasible. The integration of physics-grounded rewards in the GRPO stage further ensures these corrections align with real-world driving constraints.

![[x7 9.png|Refer to caption]]

Figure 7: "Aha Moment" of our AutoDrive-R² on the nuScenes dataset.

### A.3 "Aha Moment"

A compelling insight observed during the development of AutoDrive-R² is the emergence of a “reasoning self-correction moment”, where the model systematically identifies and resolves contradictions in its initial trajectory planning. Here’s a representative example from our experiments:

> Wait, wait. Wait. That’s a reasoning self-correction moment I can flag here.  
> Let’s reevaluate this step-by-step to identify if the predicted trajectory aligns with the traffic rules and vehicle dynamics …  
> Let me recalculate using the updated velocity: …

This reasoning self-correction moment demonstrates AutoDrive-R²’s ability to re-examine its own assumptions and refine predictions through structured self-validation. As shown in Fig. 7, similar behavior is observed during trajectory planning tasks: the model exhibits an emergent capacity to detect inconsistencies in its motion modeling and resolve them via iterative recalculations.

> let me think, the predicted trajectory seems reasonable given the vehicle’s past behavior and the current environment. The slight left turn is consistent with the steering angle, and the vehicle is maintaining a safe distance from the car ahead.  
> oh, i see, the lateral offset is minimal, which is expected given the small steering angle.  
> wait, the vehicle’s speed is increasing, which is consistent with the positive acceleration. hmm, everything checks out, so no adjustments are needed.

This autonomous reasoning process, characterized by four-stage self-reflection, is a hallmark of AutoDrive-R²’s training pipeline. The integration of physics-grounded rewards in the GRPO stage ensures that even minor corrections (e.g., adjusting lateral offsets or deceleration rates) are validated against real-world constraints.

### A.4 Vehicle Kinematics and Passenger Comfort in Physics-Grounded Rewards

The physical constraints of autonomous driving systems are deeply rooted in vehicle kinematics and passenger comfort principles. Vehicle kinematics governs the relationship between steering geometry, tire friction, and acceleration limits, ensuring that predicted trajectories adhere to the physical capabilities of the vehicle. For instance, abrupt steering adjustments can violate the minimum turning radius determined by the vehicle’s wheelbase $L$ and maximum steering angle $\delta_{\text{max}}$. The minimum turning radius $R_{\text{min}}$ is defined as

$$
R_{\text{min}}=\frac{L}{\sin(\delta_{\text{max}})},
$$

where $L$ is the distance between the front and rear axles (wheelbase), and $\delta_{\text{max}}$ is the maximum achievable steering angle of the front wheels. Any trajectory requiring a tighter turn than $R_{\text{min}}$ would be physically infeasible, leading to tire slippage or loss of control. Additionally, lateral acceleration $a_{c}$ during cornering must satisfy

$$
a_{c}=\frac{v^{2}}{R}\leq\mu g,
$$

where $v$ is the vehicle speed, $R$ is the turning radius, $\mu$ is the tire-road friction coefficient (typically $\mu\approx 0.8$), and $g$ is gravitational acceleration ($9.81\,\text{m/s}^{2}$). Exceeding this threshold results in unsafe side-slip, particularly on low-friction surfaces like wet or icy roads. Beyond kinematic feasibility, passenger comfort is critically tied to smooth motion dynamics. Sudden changes in acceleration, known as jerk $j(t)$, directly impact rider experience. Jerk is defined as

$$
j(t)=\frac{da(t)}{dt},
$$

where $a(t)$ is the instantaneous acceleration. Human tolerance for jerk is generally below $2.5\,\text{m/s}^{3}$, and abrupt steering or acceleration adjustments (e.g., $\theta_{j}-\theta_{j-1}$ or $v_{k}-v_{k-1}$) can induce discomfort jerky motions. Furthermore, rapid maneuvers amplify vibrations in the vehicle suspension system, modeled as a second-order differential equation:

$$
m\ddot{x}+c\dot{x}+kx=F(t),
$$

where $m$ is the sprung mass (mass supported by the suspension), $c$ is the damping coefficient, $k$ is the spring stiffness, $x$ is the vertical displacement of the suspension, and $F(t)$ is external forces (e.g., centrifugal force during sharp turns). Excessive $F(t)$ due to abrupt motions overwhelms the suspension, increasing perceived jolts and reducing ride quality.

These principles are directly addressed in the physics-grounded reward framework. By penalizing abrupt changes in steering angle and velocity through temporal smoothness terms $r_{\text{tem}}$, the method ensures that trajectories remain within the vehicle’s kinematic limits while minimizing jerk and vibration. This approach aligns with the experimental validation in the main text, where removing $r_{\text{tem}}$ led to a 26.3% increase in trajectory error, underscoring the necessity of balancing geometric accuracy with physical and physiological constraints.

### A.5 More Hyperparameter Configurations

Our method is implemented on a machine with an Intel(R) Xeon(R) Platinum 8480+ and eight 8 NVIDIA H20 GPUs with 90G memory. The training process ran for 750 epochs without freezing the vision transformer (ViT) backbone, and the number of generation is set to 6 in GRPO algorithm.

[^1]: Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, and et al. Gpt-4 technical report, 2024.

[^2]: Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. *arXiv preprint arXiv:2502.13923*, 2025.

[^3]: Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, James Tanner, Quan Vuong, Anna Walling, Haohuan Wang, and Ury Zhilinsky. $\pi_{0}$: A vision-language-action flow model for general robot control, 2024.

[^4]: Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners, 2020.

[^5]: Long Chen, Lukas Platinsky, Stefanie Speichert, Błażej Osiński, Oliver Scheel, Yawei Ye, Hugo Grimmett, Luca Del Pero, and Peter Ondruska. What data do we need for training an av motion planner? In *2021 IEEE International Conference on Robotics and Automation (ICRA)*, pages 1066–1072. IEEE, 2021.

[^6]: Rui Chen, Lei Sun, Jing Tang, Geng Li, and Xiangxiang Chu. Finger: Content aware fine-grained evaluation with reasoning for ai-generated videos. *arXiv preprint arXiv:2504.10358*, 2025.

[^7]: Shaoyu Chen, Bo Jiang, Hao Gao, Bencheng Liao, Qing Xu, Qian Zhang, Chang Huang, Wenyu Liu, and Xinggang Wang. Vadv2: End-to-end vectorized autonomous driving via probabilistic planning, 2024.

[^8]: Xiangxiang Chu, Limeng Qiao, Xinyang Lin, Shuang Xu, Yang Yang, Yiming Hu, Fei Wei, Xinyu Zhang, Bo Zhang, Xiaolin Wei, et al. Mobilevlm: A fast, strong and open vision language assistant for mobile devices. *arXiv preprint arXiv:2312.16886*, 2023.

[^9]: Xiangxiang Chu, Hailang Huang, Xiao Zhang, Fei Wei, and Yong Wang. Gpg: A simple and strong reinforcement learning baseline for model reasoning. *arXiv preprint arXiv:2504.02546*, 2025.

[^10]: Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, and et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025.

[^11]: Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai Wang, Lewei Lu, Xiaosong Jia, Qiang Liu, Jifeng Dai, Yu Qiao, and Hongyang Li. Planning-oriented autonomous driving. In *2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 17853–17862, 2023.

[^12]: Zhijian Huang, Chengjian Fen, Feng Yan, Baihui Xiao, Zequn Jie, Yujie Zhong, Xiaodan Liang, and Lin Ma. Drivemm: All-in-one large multimodal model for autonomous driving. *arXiv preprint arXiv:2412.07689*, 2024.

[^13]: Bo Jiang, Shaoyu Chen, Qing Xu, Bencheng Liao, Jiajie Chen, Helong Zhou, Qian Zhang, Wenyu Liu, Chang Huang, and Xinggang Wang. Vad: Vectorized scene representation for efficient autonomous driving. In *2023 IEEE/CVF International Conference on Computer Vision (ICCV)*, pages 8306–8316, 2023.

[^14]: Bo Jiang, Shaoyu Chen, Qian Zhang, Wenyu Liu, and Xinggang Wang. Alphadrive: Unleashing the power of vlms in autonomous driving via reinforcement learning and reasoning, 2025.

[^15]: Alex Kendall, Jeffrey Hawke, David Janz, Przemyslaw Mazur, Daniele Reda, John-Mark Allen, Vinh-Dieu Lam, Alex Bewley, and Amar Shah. Learning to drive in a day. In *2019 International Conference on Robotics and Automation (ICRA)*, pages 8248–8254. IEEE, 2019.

[^16]: Nathan Lambert. Reinforcement learning from human feedback, 2025.

[^17]: Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. BLIP: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In *Proceedings of the 39th International Conference on Machine Learning*, pages 12888–12900. PMLR, 2022.

[^18]: Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In *Proceedings of the 40th International Conference on Machine Learning*, pages 19730–19742. PMLR, 2023.

[^19]: Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, and et al. Deepseek-v3 technical report, 2025.

[^20]: Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In *NeurIPS*, 2023.

[^21]: Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021.

[^22]: Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. *Advances in Neural Information Processing Systems*, 36:53728–53741, 2023.

[^23]: John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms, 2017.

[^24]: Hao Shao, Yuxuan Hu, Letian Wang, Guanglu Song, Steven L. Waslander, Yu Liu, and Hongsheng Li. Lmdrive: Closed-loop end-to-end driving with large language models. In *2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 15120–15130, 2024a.

[^25]: Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. *arXiv preprint arXiv:2402.03300*, 2024b.

[^26]: Wenchao Sun, Xuewu Lin, Yining Shi, Chuang Zhang, Haoran Wu, and Sifa Zheng. Sparsedrive: End-to-end autonomous driving via sparse scene representation, 2024.

[^27]: Xiaoyu Tian, Junru Gu, Bailin Li, Yicheng Liu, Yang Wang, Zhiyong Zhao, Kun Zhan, Peng Jia, Xianpeng Lang, and Hang Zhao. Drivevlm: The convergence of autonomous driving and large vision-language models, 2024.

[^28]: Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models, 2023.

[^29]: Wenhai Wang, Jiangwei Xie, ChuanYang Hu, Haoming Zou, Jianan Fan, Wenwen Tong, Yang Wen, Silei Wu, Hanming Deng, Zhiqi Li, Hao Tian, Lewei Lu, Xizhou Zhu, Xiaogang Wang, Yu Qiao, and Jifeng Dai. Drivemlm: Aligning multi-modal large language models with behavioral planning states for autonomous driving, 2023.

[^30]: Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, Ze Liu, Ziyi Xia, Chaofan Li, Haoge Deng, Jiahao Wang, Kun Luo, Bo Zhang, Defu Lian, Xinlong Wang, Zhongyuan Wang, Tiejun Huang, and Zheng Liu. Omnigen2: Exploration to advanced multimodal generation, 2025.

[^31]: Zhenhua Xu, Yujia Zhang, Enze Xie, Zhen Zhao, Yong Guo, Kwan-Yee K Wong, Zhenguo Li, and Hengshuang Zhao. Drivegpt4: Interpretable end-to-end autonomous driving via large language model. *IEEE Robotics and Automation Letters*, 2024.

[^32]: Zhenjie Yang, Yilin Chai, Xiaosong Jia, Qifeng Li, Yuqian Shao, Xuekai Zhu, Haisheng Su, and Junchi Yan. Drivemoe: Mixture-of-experts for vision-language-action model in end-to-end autonomous driving, 2025.

[^33]: Tengju Ye, Wei Jing, Chunyong Hu, Shikun Huang, Lingping Gao, Fangzhen Li, Jingke Wang, Ke Guo, Wencong Xiao, Weibo Mao, Hang Zheng, Kun Li, Junbo Chen, and Kaicheng Yu. Fusionad: Multi-modality fusion for prediction and planning tasks of autonomous driving, 2023.

[^34]: Gokul Yenduri, Ramalingam M, Chemmalar Selvi G, Supriya Y, Gautam Srivastava, Praveen Kumar Reddy Maddikunta, Deepti Raj G, Rutvij H Jhaveri, Prabadevi B, Weizheng Wang, Athanasios V. Vasilakos, and Thippa Reddy Gadekallu. Generative pre-trained transformer: A comprehensive review on enabling technologies, potential applications, emerging challenges, and future directions, 2023.

[^35]: Zheng Yuan, Hongyi Yuan, Chengpeng Li, Guanting Dong, Keming Lu, Chuanqi Tan, Chang Zhou, and Jingren Zhou. Scaling relationship on learning mathematical reasoning with large language models, 2023.

[^36]: Xingcheng Zhou, Xuyuan Han, Feng Yang, Yunpu Ma, and Alois C. Knoll. Opendrivevla: Towards end-to-end autonomous driving with large vision language action model, 2025a.

[^37]: Zewei Zhou, Tianhui Cai, Seth Z. Zhao, Yun Zhang, Zhiyu Huang, Bolei Zhou, and Jiaqi Ma. Autovla: A vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning, 2025b.

[^38]: Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. *arXiv preprint arXiv:2304.10592*, 2023.
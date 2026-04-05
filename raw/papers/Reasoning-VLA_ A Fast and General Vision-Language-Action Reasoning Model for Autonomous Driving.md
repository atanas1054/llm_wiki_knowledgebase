---
title: "Reasoning-VLA: A Fast and General Vision-Language-Action Reasoning Model for Autonomous Driving"
source: "https://arxiv.org/html/2511.19912v1"
author:
published:
created: 2026-04-05
description:
tags:
  - "clippings"
---
Dapeng Zhang <sup>1,2</sup>, Zhenlong Yuan <sup>3</sup>, Zhangquan Chen <sup>4</sup>, Chih-Ting Liao <sup>5</sup>, Yinda Chen <sup>6</sup>  
Fei Shen <sup>2</sup> <sup>*</sup>, Qingguo Zhou <sup>1</sup> <sup>*</sup>, Tat-Seng Chua <sup>2</sup>  
<sup>1</sup> Lanzhou University, China; <sup>2</sup> National University of Singapore, Singapore  
<sup>3</sup> University of Science and Technology of China, China; <sup>4</sup> Tsinghua University, China  
<sup>5</sup> University of New South Wales, Australia; <sup>6</sup> University of Science and Technology of China, China  
<sup>*</sup> Corresponding authors  

###### Abstract

Vision-Language-Action (VLA) models have recently shown strong decision-making capabilities in autonomous driving. However, existing VLAs often struggle with achieving efficient inference and generalizing to novel autonomous vehicle configurations and driving scenarios. In this paper, we propose Reasoning-VLA, a general and fast action-generation VLA framework. The proposed model employs a set of learnable action queries, initialized via Gaussian sampling from ground-truth trajectories within the training corpus. These learnable queries interact with reasoning-enhanced vision–language features to generate continuous action trajectories in parallel. To promote robust generalization, we consolidate eight publicly available autonomous driving datasets into a standardized, Chain-of-Thought reasoning–based, and easy-to-use data format for model training. Leveraging both supervised learning and reinforcement learning fine-tuning, extensive empirical evaluations across multiple benchmarks demonstrate that Reasoning-VLA achieves state-of-the-art performance, superior generalization capability, and the excellent inference speed reported to date. Code: [https://github.com/xipi702/Reasoning-VLA](https://arxiv.org/html/2511.19912v1/github%E9%A6%96%E9%A1%B5)

![[x1 5.png|Refer to caption]]

Figure 1: Reasoning-VLA is an efficient Vision–Language–Action (VLA) framework for autonomous driving that employs parallel actions to interact with reasoning-enhanced vision–language models (VLMs), enabling one-step prediction of future trajectories. The model is trained on our unified and generalized autonomous driving dataset using a combination of supervised fine-tuning (SFT) and reinforcement learning (RL), guided by specifically designed rule-based reward functions.

## 1 Introduction

Autonomous driving (AD) is a highly complex system that requires precise environmental perception and reliable driving behavior generation. Traditional end-to-end AD methods initially advanced the field but face issues such as poor scalability, cumulative errors, and limited generalization across hardware and datasets. These limitations hinder their generalization ability to new driving scenarios. Recently, foundation models—especially large language and vision–language models like CLIP, Qwen2.5-VL, and DeepSeek-V3 \[clip, Qwen2.5VL, deepseekv3\] —have shown remarkable generalization through large-scale pretraining. Their capabilities offer a promising direction for building more flexible and robust AD systems.

Building on these advancements, contemporary frameworks in robotic manipulation and autonomous driving increasingly adopt vision–language generative paradigms (e.g., autoregressive or diffusion-based models \[pi0, openvla, autodriver2, drivemoe\]), collectively referred to as Vision–Language–Action (VLA) models. These systems generate fine-grained action trajectories from high-level visual–linguistic reasoning, thereby enhancing flexibility and practicality in motion planning and control. Leveraging large-scale pretrained foundation models, recent approaches such as DriveMOE \[drivemoe\] have achieved strong benchmark performance while simultaneously improving interpretability and robustness capabilities in autonomous driving tasks.

Despite these promising results, several challenges hinder the widespread deployment of VLAs in autonomous driving: 1) Most existing VLA architectures are based on autoregressive or diffusion models that require multiple inference steps to generate actions, limiting their suitability for real-time, high-frequency control. 2) Current VLA methods lack robust generalization to new vehicle platforms or unseen driving scenarios. We argue that developing a general-purpose foundation VLA requires diverse, large-scale datasets that encompass various environments and vehicle configurations. 3) Existing fine-tuning strategies are often inefficient in exploring the full potential of VLAs, constraining their generalization capability.

To address these challenges, we propose Reasoning-VLA, an efficient and generalist VLA framework that establishes a new state-of-the-art for autonomous driving. First, we design a novel interaction mechanism between action and vision–language modalities by introducing a set of learnable action queries initialized via Gaussian Sampling from ground-truth trajectories in the dataset. These learnable queries interact with reasoning-enhanced vision–language representations through cross-attention to extract action-related information efficiently and generate continuous trajectories in parallel. Second, to enable generalization, we construct a unified, Chain-of-Thought reasoning-based dataset that merges eight publicly available autonomous driving datasets into a coherent and easy-to-use format. This dataset covers diverse vehicle platforms and driving scenarios, enhancing the generalization ability of Reasoning-VLA. Finally, we adopt a two-stage training strategy that combines supervised fine-tuning (SFT) and reinforcement learning (RL) to fully exploit the model’s reasoning and planning potential. Extensive experiments demonstrate that Reasoning-VLA significantly improves generalization ability, planning performance, and inference speed compared with existing VLA approaches. To summarize, the main contributions are as follows:

- We propose Reasoning-VLA, an efficient and fast VLA framework that employs learnable action queries to interact with reasoning-enhanced vision–language representations, enabling one-step parallel action generation.
- We initialize learnable action queries via Gaussian Distribution Sampling from ground-truth trajectories, improving model efficiency.
- We construct a unified, Chain-of-Thought reasoning-based autonomous driving dataset that merges eight existing datasets, facilitating generalization across vehicle types and driving environments.
- We employ a combined SFT and RL fine-tuning strategy augmented with physical and dynamic rewards to enhance the general reasoning ability of Reasoning-VLA, achieving substantial improvements over prior methods.

## 2 Related Work

### 2.1 Classic Autonomous Driving

Classic autonomous driving (AD) methods have been developed over many years, evolving from modular systems to modern end-to-end learning frameworks \[lss, bevdet4d, mapexpert, streammapnet, pnpnet, plan1, ehss\]. Early AD systems were typically constructed by cascading these single-task modules into a sequential pipeline \[bevformer, detr3d, vectormapnet, mapexpert, prediction1\]. However, such designs suffer from error accumulation, where inaccuracies in upstream tasks propagate through subsequent modules, ultimately degrading overall system performance. To address this issue, recent research has shifted toward end-to-end learning-based approaches that integrate all sub-tasks into a unified framework. Modern open-source end-to-end AD methods increasingly rely on bird’s-eye view (BEV) feature representations and generate planning trajectories through cross-interactions among internal components \[uniad, vad, vad2, thinktwice, bench2drive\]. Meanwhile, other approaches exploit sparse feature extraction from the 3D environment to directly infer results from image features, thereby avoiding the computational cost of constructing explicit BEV features \[detr3d, sparsedrive\]. Collectively, these advances have simplified the traditional multi-stage AD pipeline, marking the beginning of a new era of data-driven autonomous driving.

### 2.2 Vision-Language-Action Models

With the rapid advancement of VLMs in recent years \[clip, minigpt-4, blip, blip2, Qwen2.5VL, deepseekv3, internvl3\], researchers have increasingly integrated VLMs into autonomous driving and robotic systems to enhance overall performance. For instance, DriveVLM and DriveMM \[drivevlm, drivemm\] incorporate VLM modules to improve situational understanding and enhance generalization in vehicle control. DriveMLM \[drivemlm\] introduces a behavior planning module that produces optimal driving decisions along with rationales.

Although these methods effectively model vision-language representations, they often neglect the role of action generation, limiting their practical applicability. To address this, recent works have explored integrating vision-language understanding with action prediction, directly fine-tuning large pre-trained VLMs to estimate robot actions \[rt2, pi0, purevla\]. These approaches, commonly referred to as Vision–Language–Action (VLA) models. Recent representative VLA methods demonstrate significant performance improvements. OpenVLA \[openvla\] employs a pre-trained VLM combined with a discretization bin tokenizer to predict actions. Similarly, $\pi_{0.5}$ \[pi0.5\] leverages co-training and hybrid multimodal examples—incorporating robot observations, language instructions, and low-level actions—within a single unified model, achieving SOTA performance. The success of VLAs in robotics provides a promising direction for autonomous driving. Some approaches extend novel VLM architectures to train billion-parameter policies with task-specific modifications, offering a direct pathway for AD systems to benefit from rapid VLM advancements \[autodriver2\]. However, most existing VLA methods rely on autoregressive or diffusion-based training and inference, which inherently limits their speed and efficiency \[drivemoe\].

### 2.3 Fine-tuning with Reinforcement Learning

With the development of extensive pre-training techniques and high-level general capabilities, Reinforcement Learning (RL) has achieved remarkable success in advancing the reasoning and decision-making abilities of LLMs. Reinforcement Learning from Human Feedback (RLHF) approaches, such as PPO \[ppo\], typically require training a reward model to optimize the policy network. However, this process can be complex and often unstable. Notably, models such as GPT-4 \[gpt4\] follow this RL-based fine-tuning paradigm. Building upon PPO, DPO \[dpo\] fine-tunes pre-trained models to follow instructions and align with human preferences, while eliminating the need for sampling during fine-tuning. Similarly, Qwen3 \[qwenimage\] employs DPO to improve performance in applications. Another variant, GRPO \[grpo\], uses sampling to estimate advantages, thereby effectively enhancing the reasoning capabilities of actors. For example, DeepSeek-R1 \[deepseekr1\] applies GRPO to advance LLM reasoning, emphasizing self-evolution rather than fine-tuning data. Inspired by these methods, recent works have adopted analogous RL-based fine-tuning strategies to improve the reasoning and decision-making capabilities of autonomous driving models \[autodriver2\].

## 3 Method

As illustrated in Fig. 1, the Reasoning-VLA framework comprises three main components: (1) a reasoning-enhanced vision–language model (VLM) backbone, (2) an action module that interacts with the VLM and enables parallel decoding of action trajectories, and (3) a multi-stage intermediate refinement module. In the following sections, we present a detailed description of our approach to developing a VLA framework for autonomous driving and highlight key insights.

### 3.1 Preliminaries: Vision-Language Models

In this work, we adopt Qwen2.5-VL \[Qwen2.5VL\] as our foundational model. Qwen2.5-VL effectively simulates human-like analytical thinking, supporting multi-step reasoning, deliberate planning, and problem-solving. Qwen2.5-VL incorporates several architectural innovations: a redesigned Vision Transformer (ViT) with 2D-RoPE and windowed attention for computational efficiency; an MLP-based vision–language merger that compresses visual features into tokens suitable for the LLM; and a large language model initialized with pre-trained Qwen2.5 weights. The model not only exhibits strong vision–language understanding but also maintains robust LLM reasoning capabilities. Furthermore, it generalizes effectively across domains without requiring task-specific fine-tuning, making it a suitable base model for applications such as autonomous driving and action execution in real-world scenarios.

### 3.2 The Structure of Reasoning-VLA

Most existing Vision–Language–Action (VLA) methods either rely on a specialized action tokenizer to convert actions into a format compatible with LLMs—followed by autoregressive generation or employ diffusion/flow matching modules to refine VLM hidden states or noise in order to produce continuous action values. In contrast, our Reasoning-VLA, built on the Qwen2.5-VL symbolic reasoning framework, fundamentally differs from these autoregression-based and diffusion-based approaches \[pi0, openvla, autodriver2\]. To bridge vision-language representations and action prediction, Reasoning-VLA comprises three primary components: A pre-trained VLM reasoning backbone; A VL-to-Action module that leverages a set of learnable action queries for parallel action decoding; A refinement module that enhances action prediction performance. As illustrated in Fig. 1, the learnable action queries are designed with the same feature dimensionality as the Qwen2.5-VL reasoning model. These queries undergo self-attention and cross-attention with the VLM simultaneously. By employing additional learnable queries, Reasoning-VLA can predict action chunks in a single step, rather than generating actions token by token, as required in autoregressive approaches. The features from these action queries, together with intermediate VLM representations, are subsequently processed by a series of refinement modules to produce the final action trajectories. The architectural design of Reasoning-VLA offers four key advantages:

1\. Leverages the reasoning capabilities of the VLM for more informed and context-aware action generation.

2\. Parallel action generation via action queries enables significantly higher inference speed compared to autoregressive or diffusion-based methods.

3\. Learnable action queries are initialized with Gaussian-distributed ground-truth actions, improving model performance.

4\. Refinement modules interact with intermediate hidden states to enhance feature representation and trajectory accuracy.

### 3.3 Learnable Action Queries and Initialization

#### 3.3.1 Learnable Action Queries

We initialize a set of learnable action queries $AQ\in T\times N\times D$. Here $T$ is the number of future time steps to be predicted, $N$ is the dimensions of action trajectory coordinate, $D$ is the feature dimensionality. As shown in Fig.2. Unlike VLMs, which embedded input tokens into embeddings, our action queries are initialized as learnable parameters. This design provides greater flexibility and expressive capacity, enabling parallel prediction of action trajectories, and offering an efficient alternative to sequential token generation.

#### 3.3.2 Learnable Action Query Initialization

To accelerate training convergence, we also initialize the action queries with a set of predefined parameters. These predefined parameters must satisfy two criteria: 1. The predefined parameters must match the shape of action queries; 2. The reasonable initial values that reflect typical action distributions. Given that the total number of action values is $T\times N$, in our method, we predict future $T$ steps for $N$ coordinates (e.g., $x$, $y$), total action query is $N\times T$. We have to generate $T\times N$ action queries with $D$ dimensions.

Specifically, we extract the action trajectory values of each frame firstly (each frame have $N\times T$ action values, the total action trajectory values are $D_{all}\times N\times T$, where $D_{all}$ is the total number of frames in our datasets), then we calculate the mean action values, such as, $x_{1},y_{1},x_{2},y_{2},x_{i},y_{i},...,x_{N},y_{N}$, each $x_{i},y_{i}$ represents the average coordinate for the corresponding position. To match the feature dimension of action queries, we extend the $N\times T$ action values to $N\times T\times D$, by sampling $D$ values from a Gaussian distribution with the previously calculated mean and variance. This procedure completes the initialization of the learnable action queries, providing a well-structured and informative starting point for training.

### 3.4 How Do Actions Interact with Vision-Language Reasoning?

![[x2 4.png|Refer to caption]]

Figure 2: The action module interacts with the vision-language model (VLM). The learnable action queries are initialized using a Gaussian distribution derived from the ground-truth (GT) action data. Through self-attention and cross-attention mechanisms with the reasoning VLM, the model transfers the generalized reasoning capability from the VL to A.

Unlike autoregression-based or diffusion-based vision-language-action (VLA) methods, our approach employs independent learnable action queries to predict action trajectories. Consequently, the interaction between the action module and the vision-language model (VLM) differs substantially. Since the action queries are not tied to the VLM’s token representations, they first perform self-attention and then interact with the VLM through cross-attention, as illustrated in Fig. 2. Through these attention mechanisms, the action queries can extract rich and semantically meaningful information from the VLM’s hidden states, which contain extensive reasoning content. This interaction strategy provides a significant advantage: the action queries can generate all expected actions in parallel during a single forward pass, enabling efficient action chunking. This contrasts with autoregression-based VLAs that require sequential token-by-token processing. Our approach reduces action generation from more than $N\times T$ sequential passes to a single pass, substantially improving both training and inference efficiency. Furthermore, we eliminate the discretization process used in autoregressive VLAs, which often degrades fine-grained action details. In addition, we replace the causal attention mask with bidirectional mask, allowing models to predict all actions simultaneously.

### 3.5 Action Refinement Module

To further enhance the representation quality and accuracy of the predicted action trajectories, we introduce an Action Refinement Module (ARM). Specifically, the ARM takes the selected hidden states of the action queries as input and refines them through a combination of multilayer perceptron (MLP) and attention mechanisms. Unlike next-token prediction methods (e.g., $\pi_{0}$), which employ discrete action representations, our approach adopts a regression-based strategy to generate continuous actions. This design preserves the efficiency benefits of parallel action prediction while improving the precision and smoothness of the resulting action trajectories.

### 3.6 SFT and RL

Drawing inspiration from recent advances in VLMs, we employ two complementary training strategies to enhance the generalization ability of our model: supervised fine-tuning (SFT) and reinforcement learning (RL) fine-tuning.

SFT. In this stage, we utilize our unified reasoning dataset to construct structured reasoning chains. Prior studies in VLMs have shown that base models tend to generate tangential or unstructured responses without supervised fine-tuning. Therefore, the SFT process is essential for establishing a solid foundation for subsequent RL training. Reasoning-VLA demonstrates excellent performance on the unified reasoning dataset after SFT.

RL. Although SFT effectively fits the training data, it often struggles to generalize to unseen or out-of-distribution scenarios. To address this limitation, we apply the GRPO \[grpo\] during RL fine-tuning. Unlike conventional policy-based methods, GRPO replaces the critic model—typically as large as the policy model—with an estimation of group scores. This design not only simplifies the overall architecture but also significantly reduces computational overhead during training. The rule-based reward functions used for RL optimization are introduced in the following subsection.

### 3.7 Reward Functions

Normally, AD methods use BEV 2-dimensions coordinates $x,y$ to optimize the loss function, while neglecting physical trajectory constraints and vehicle dynamics. In our RL fine-tuning stage, we design two types of verifiable reward functions to more accurately evaluate and enhance the quality of generated responses.

Physical Trajectory Reward Different from most regression-based reward functions that employ the mean squared Euclidean distance $\frac{1}{N}\sum_{i=1}^{N}(x^{i}-x_{\text{gt}}^{i})^{2}$, we adopt a weighted Euclidean distance to better align the predicted coordinates with the ground-truth trajectories. Specifically, our physical trajectory reward is defined as:

$$
r_{\text{traj}}=1-\frac{1}{N}\sum_{i=1}^{N}\gamma^{i}\left(\alpha(x^{i}-x_{\text{gt}}^{i})^{2}+\beta(y^{i}-y_{\text{gt}}^{i})^{2}\right)\vskip-5.69046pt
$$

where $N$ is the number of trajectory steps, $x^{i}$ and $y^{i}$ are the predicted coordinates at the $i$ -th time step, and $x_{\text{gt}}^{i}$ and $y_{\text{gt}}^{i}$ are their corresponding ground-truth values. Because the $x$ and $y$ coordinates in autonomous driving often differ in scale, the weighting factors $\alpha$ and $\beta$ are introduced to balance their respective contributions to the reward. The term $\gamma^{i}$ represents a discount factor that reduces the influence of future trajectory points. This reward function encourages the autonomous vehicle to follow the desired route by penalizing deviations across the entire trajectory.

![[8data.png|Refer to caption]]

Figure 3: Statistical distribution of the unified dataset.

Table 1: Open-loop performance on the nuScenes dataset. Our fully generalized methods, Reasoning-VLA-3B and Reasoning-VLA-7B, follow the complete SFT and RL training process described in the Methods section. The training dataset is our unified dataset, which is constructed from NAVSIM \[navsim\], nuScenes \[nuscenes\], Waymo \[waymo\], Argoverse-V2 \[argoverse\], KITTI \[kitti\], Mapillary \[mapillary\], ONCE \[once\], and IDD \[idd\]. The validation dataset comprises the corresponding nuScenes validation clips from the unified dataset. Reasoning-VLA-7B+ is fine-tuned with an additional RL process using the corresponding nuScenes training clips from the unified dataset. \*: Official checkpoints re-validated with corrected metrics, sourced from \[st-p3\]. Reasoning-VLA-7B represents our general model.

<table><tbody><tr><th rowspan="2">Methods</th><td colspan="4">L2 (m) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td colspan="4">Collision Rate (%) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td></tr><tr><th colspan="9">End2End Autonomous Driving Methods</th></tr><tr><th>ST-P3 <cite>[st-p3]</cite></th><td>1.33</td><td>2.11</td><td>2.90</td><td>2.11</td><td>0.23</td><td>0.62</td><td>1.27</td><td>0.71</td></tr><tr><th>UniAD <cite>[uniad]</cite> *</th><td>0.45</td><td>0.70</td><td>1.04</td><td>0.73</td><td>0.62</td><td>0.58</td><td>0.63</td><td>0.61</td></tr><tr><th>VAD <cite>[vad]</cite> *</th><td>0.41</td><td>0.70</td><td>1.05</td><td>0.72</td><td>0.07</td><td>0.17</td><td>0.41</td><td>0.22</td></tr><tr><th>PPAD <cite>[ppad]</cite></th><td>0.30</td><td>0.69</td><td>1.26</td><td>0.75</td><td>0.03</td><td>0.22</td><td>0.73</td><td>0.33</td></tr><tr><th>SparseDrive <cite>[sparsedrive]</cite></th><td>0.29</td><td>0.63</td><td>0.97</td><td>0.63</td><td>0.03</td><td>0.09</td><td>0.19</td><td>0.10</td></tr><tr><th colspan="9">VLM & VLA Autonomous Driving Methods</th></tr><tr><th>DriveVLM-Dual <cite>[drivevlm]</cite></th><td>0.15</td><td>0.29</td><td>0.48</td><td>0.31</td><td>0.05</td><td>0.08</td><td>0.17</td><td>0.10</td></tr><tr><th>OmniDrive <cite>[omnidrive]</cite></th><td>0.14</td><td>0.29</td><td>0.55</td><td>0.33</td><td>0.00</td><td>0.13</td><td>0.78</td><td>0.30</td></tr><tr><th>EMMA+ <cite>[emma]</cite></th><td>0.13</td><td>0.27</td><td>0.48</td><td>0.29</td><td>–</td><td>–</td><td>–</td><td>–</td></tr><tr><th>Impromptu-VLA <cite>[impromptuvla]</cite></th><td>0.13</td><td>0.27</td><td>0.53</td><td>0.30</td><td>–</td><td>–</td><td>–</td><td>–</td></tr><tr><th colspan="9">Our Reasoning-VLA Methods</th></tr><tr><th>Reasoning-VLA-3B</th><td>0.08</td><td>0.33</td><td>0.48</td><td>0.30</td><td>0.04</td><td>0.13</td><td>0.23</td><td>0.13</td></tr><tr><th>Reasoning-VLA-7B</th><td>0.05</td><td>0.20</td><td>0.44</td><td>0.23</td><td>0.01</td><td>0.07</td><td>0.15</td><td>0.08</td></tr><tr><th>Reasoning-VLA-7B+</th><td>0.05</td><td>0.19</td><td>0.41</td><td>0.22</td><td>0.02</td><td>0.06</td><td>0.13</td><td>0.07</td></tr></tbody></table>

Vehicle Dynamic Reward In autonomous driving, most existing studies rarely incorporate vehicle kinematic and dynamic constraints into motion trajectory prediction. However, these constraints exert a non-negligible influence on the vehicle’s behavior and overall driving safety. To address this limitation, we propose a Vehicle Dynamics Reward that explicitly accounts for steering and acceleration to constrain the limitations of real-world vehicle dynamics. This design establishes a dynamic constraint optimization objective that ensures physically feasible and stable motion trajectories. The generated action trajectories are governed by both steering kinematics and acceleration dynamics. Specifically, the maximum steering angle is limited to 40 degrees, and the maximum acceleration is constrained to 0.6 gravity. Moreover, abrupt changes in steering or acceleration may lead to vehicle instability or discomfort for passengers. To achieve comfortable and safe driving behavior, the steering constraint reward is defined as:

$$
r_{\text{steer}}=\frac{1}{N-1}\sum_{j=1}^{N-1}\left\{\begin{aligned} 1,|\left(y^{j}-y^{j-1}\right)/\left(x^{j}-x^{j-1}\right)|<0.84\\
0,|\left(y^{j}-y^{j-1}\right)/\left(x^{j}-x^{j-1}\right)|\geq 0.84\\
\end{aligned}\right.
$$

where $(x^{j},y^{j})$ and $(x^{j-1},y^{j-1)}$ respectively denote the predicted coordinates at $j-th$ and $(j-1)-th$ time step. In this reward function, we reward with 1 when the turning angle is less than 0.84.

We further introduce an Acceleration Reward to constrain non-physical vehicle dynamics. The acceleration reward is defined as:

$$
\begin{split}acc_{j}&=\frac{\sqrt{(x^{j+1}-x^{j})^{2}+(y^{j+1}-y^{j})^{2}}}{T^{2}}\\
&-\frac{\sqrt{(x^{j}-x^{j-1})^{2}+(y^{j}-y^{j-1})^{2}}}{T^{2}}\end{split}
$$
 
$$
r_{\text{acc}}=\frac{1}{N-2}\sum_{j=1}^{N-2}\left\{\begin{aligned} 1,\quad|acc_{j}|<6\\
0,\quad|acc_{j}|\geq 6\\
\end{aligned}\right.
$$

Here $N$ is the number of trajectory steps, $T$ is the time interval between consecutive actions, $j$ is the $j$ -th trajectory step.

As shown in equation above, the reward function effectively constrains steering and acceleration within physical limits, ensuring that the generated action trajectories are both physically realizable and socially acceptable in mixed traffic scenarios. This design further reinforces the autonomous driving system’s ability to maintain stable and reasonable motion patterns, which are essential for safe and comfortable driving. The final reward $r_{total}$ is defined as the weighted sum of $r_{traj}$, $r_{steer}$ and $r_{acc}$.

$$
r_{\text{total}}=\theta_{1}r_{\text{traj}}+\theta_{2}r_{\text{steer}}+\theta_{3}r_{\text{acc}}
$$

Here, $\theta_{1},\theta_{2},\theta_{3}$ are coefficients that balance the contributions of each sub-reward.

## 4 Unified Datasets

To capture diverse driving scenarios and further improve generalization, we specifically selected eight widely used autonomous driving datasets as the foundation for our unified dataset: NAVSIM \[navsim\], nuScenes \[nuscenes\], Waymo \[waymo\], Argoverse-V2 \[argoverse\], KITTI \[kitti\], Mapillary \[mapillary\], ONCE \[once\], and IDD \[idd\]. However, many of the original clips lack meaningful text-image associations and often have coarse annotations, limiting their suitability for vision-language-action (VLA) reasoning and creative generation.

From these sources, we carefully selected over 75,000 high-quality clips to form a reasoning-intensive dataset. Each clip was processed using a strong reasoning VLM to generate Chain-of-Thought descriptions, followed by comprehensive human verification and visualization to ensure correctness and annotation quality. The final dataset is provided in a consistent, standardized format, facilitating downstream training and evaluation. The statistical analysis of the resulting unified dataset is presented in Fig. 3. The processing pipeline for the unified dataset is illustrated in Fig. 5 of Appendix B. A representation of the dataset is also provided in the Appendix B.

Table 2: Closed-loop performance on the NeuroNCAP. We utilize the challenging closed-loop NeuroNCAP simulator to emulate a wide range of complex real-world driving scenarios. NeuroNCAP provides pretrained rendering model checkpoints, making it particularly well-suited for evaluating our method. For our experiments, we downloaded the NeuroAD weights, adapted the evaluation scripts accordingly, and conducted the closed-loop evaluation. Since NeuroNCAP offers a standardized benchmark and evaluation metrics commonly used by other methods, we adhered to its recommended configuration. The Reasoning-VLA modules and fine-tuning process are identical to those employed in the open-loop evaluation. \*: Sourced from \[nrsad, bridgeadb\].

<table><tbody><tr><th rowspan="2">Methods</th><td colspan="4">NeuroNCAP Score <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td colspan="4">Collision Rate (%) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><td>Stationary</td><td>Frontal</td><td>Side</td><td>Avg.</td><td>Stationary</td><td>Frontal</td><td>Side</td><td>Avg.</td></tr><tr><th colspan="9">End2End & VLA Autonomous Driving Methods</th></tr><tr><th>UniAD <cite>[uniad]</cite> *</th><td>0.84</td><td>0.10</td><td>1.26</td><td>0.73</td><td>87.8</td><td>98.4</td><td>79.6</td><td>88.6</td></tr><tr><th>VAD <cite>[vad]</cite> *</th><td>0.47</td><td>0.04</td><td>1.45</td><td>0.66</td><td>96.2</td><td>99.6</td><td>81.6</td><td>92.5</td></tr><tr><th>SparseDrive <cite>[sparsedrive]</cite> *</th><td>–</td><td>–</td><td>–</td><td>0.92</td><td>–</td><td>–</td><td>–</td><td>93.9</td></tr><tr><th>BridgeAD-B <cite>[bridgeadb]</cite> *</th><td>–</td><td>–</td><td>–</td><td>1.60</td><td>–</td><td>–</td><td>–</td><td>72.6</td></tr><tr><th>Impromptu-VLA <cite>[impromptuvla]</cite></th><td>1.77</td><td>2.31</td><td>2.10</td><td>2.15</td><td>70.0</td><td>59.0</td><td>65.0</td><td>65.5</td></tr><tr><th colspan="9">Our Reasoning-VLA Methods</th></tr><tr><th>Reasoning-VLA-3B</th><td>1.88</td><td>2.29</td><td>1.94</td><td>2.04</td><td>63.7</td><td>60.4</td><td>64.1</td><td>62.7</td></tr><tr><th>Reasoning-VLA-7B</th><td>1.93</td><td>2.57</td><td>2.24</td><td>2.25</td><td>59.8</td><td>56.0</td><td>62.2</td><td>59.4</td></tr><tr><th>Reasoning-VLA-7B+</th><td>2.06</td><td>2.33</td><td>2.17</td><td>2.19</td><td>57.9</td><td>57.4</td><td>64.0</td><td>59.8</td></tr></tbody></table>

## 5 Experiments

We conduct experiments to evaluate Reasoning-VLA as an efficient VLA method for autonomous driving, assess the effectiveness of our training process, and explore its potential as a unified base model for specific autonomous driving tasks. The experiments are designed to answer the following questions:

1\. How does Reasoning-VLA compare to prior autonomous driving VLA, when evaluated across multiple datasets and under various generalization scenarios?

2\. How does each design affect the performance of fine-tuned Reasoning-VLA on general autonomous driving tasks?

3\. Can the design of Reasoning-VLA influence inference efficiency (action generation throughput and latency) and make it more accessible?

### 5.1 Experiment Setups

In our experiments, we mainly evaluate Reasoning-VLA’s performance on unified AD datasets, which are constructed from eight autonomous driving datasets. To fairly compare with existing methods, we retain the original training and testing splits of each dataset. During training, we shuffle the unified datasets and fine-tune Reasoning-VLA sequentially using SFT followed by RL. The decay learning rate are start from 5e-4 and e-6 form SFT and RL separately, the accumulated size is 2. Training is performed for 4 epochs for SFT and 1 epoch for RL, using a total batch size of 8 distributed across 8 H200 GPUs. For open-loop evaluation, we use the same testing and validation clips as employed by prior methods. For closed-loop evaluation, the model is tested on the NeuroNCAP benchmark to enable a fair comparison with other approaches.

### 5.2 Main Comparison Results

#### 5.2.1 Open-loop Evaluation

Since our goal is to propose a generalized VLA model for autonomous driving, we train our model using the proposed unified dataset. To ensure a fair comparison with prior methods, we adopt the same validation splits and report results on open-loop benchmarks. The open-loop performance on the nuScenes dataset is summarized in Table 1. Three main models are presented in this table: Reasoning-VLA-3B: Based on Qwen2.5-VL-3B, trained using the complete SFT and RL process. Reasoning-VLA-7B: Based on Qwen2.5-VL-7B and fine-tuned using the SFT and RL process. Reasoning-VLA-7B+: Similar to Reasoning-VLA-7B, but additionally fine-tuned with RL on selected nuScenes training clips from the unified dataset. Our results show that the purely generalized model, Reasoning-VLA-7B, surpasses previous works across benchmarks, achieving substantial improvements of +23.3% in average L2 and +20.0% in average Collision Rate over the existing best methods. Reasoning-VLA-3B also achieves results comparable to state-of-the-art methods. When fine-tuned with GRPO on specific datasets (i.e., selected nuScenes training clips from the unified dataset), our generalized model demonstrates excellent task-specific performance. As shown in the last row of Table 1, the additional fine-tuning further improves performance across all time intervals: Reasoning-VLA-7B+ achieves increases of 4.3% and 12.5% over Reasoning-VLA-7B in average L2 and Collision Rate, respectively. These results indicate that our approach provides significant improvements in open-loop evaluation, highlighting the strong generalization capability of the Reasoning-VLA architecture. Consequently, it can serve as an effective base model for downstream autonomous driving tasks.

Table 3: Generalized performance on our unifed dataset. We trained two models using the unified dataset: Reasoning-VLA-7B + SFT: This model is fine-tuned using only supervised fine-tuning (SFT). Reasoning-VLA-7B + SFT + RL: This model undergoes the full training process, including both SFT and reinforcement learning (RL). The training dataset for both models is the unified training dataset. For evaluation, the dataset splits follow the recommendations provided by each original dataset.

<table><tbody><tr><th rowspan="3">Datasets</th><td colspan="4">Reasoning-VLA-7B + SFT</td><td colspan="4">Reasoning-VLA-7B + SFT + RL</td></tr><tr><td colspan="4">L2 (m) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td colspan="4">L2 (m) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td></tr><tr><th>NAVSIM <cite>[navsim]</cite></th><td>0.05</td><td>0.18</td><td>0.43</td><td>0.22</td><td>0.04</td><td>0.18</td><td>0.41</td><td>0.21</td></tr><tr><th>nuScenes <cite>[nuscenes]</cite></th><td>0.06</td><td>0.23</td><td>0.48</td><td>0.26</td><td>0.05</td><td>0.20</td><td>0.44</td><td>0.23</td></tr><tr><th>Waymo <cite>[waymo]</cite></th><td>0.04</td><td>0.15</td><td>0.44</td><td>0.21</td><td>0.03</td><td>0.14</td><td>0.48</td><td>0.22</td></tr><tr><th>Argoverse-V2 <cite>[argoverse]</cite></th><td>0.01</td><td>0.13</td><td>0.45</td><td>0.20</td><td>0.01</td><td>0.14</td><td>0.43</td><td>0.19</td></tr><tr><th>KITTI <cite>[kitti]</cite></th><td>0.02</td><td>0.15</td><td>0.48</td><td>0.22</td><td>0.01</td><td>0.15</td><td>0.43</td><td>0.20</td></tr><tr><th>Mapillary <cite>[mapillary]</cite></th><td>0.04</td><td>0.44</td><td>0.92</td><td>0.47</td><td>0.04</td><td>0.41</td><td>1.01</td><td>0.49</td></tr><tr><th>ONCE <cite>[once]</cite></th><td>0.07</td><td>0.49</td><td>0.87</td><td>0.48</td><td>0.06</td><td>0.43</td><td>0.90</td><td>0.46</td></tr><tr><th>IDD <cite>[idd]</cite></th><td>0.02</td><td>0.29</td><td>0.77</td><td>0.36</td><td>0.03</td><td>0.27</td><td>0.81</td><td>0.37</td></tr><tr><th>Unified</th><td>0.05</td><td>0.20</td><td>0.47</td><td>0.24</td><td>0.05</td><td>0.19</td><td>0.43</td><td>0.23</td></tr></tbody></table>

#### 5.2.2 Closed-loop Evaluation

We use NeuroNCAP \[neuroncap\] as the closed-loop real-world simulator because it provides renderings of novel, unseen scenarios. Most existing closed-loop real-world simulators are limited in their rendering of the reactions of surrounding objects, such as vehicles and pedestrians, whereas NeuroNCAP offers pretrained rendering model checkpoints, making it particularly well-suited for evaluating our method. As shown in Table 2, the three main models evaluated are the same as those in the open-loop experiments.

Our methods demonstrate significant advantages in closed-loop performance on NeuroNCAP. The generalized model, Reasoning-VLA-7B, substantially outperforms prior methods in terms of NeuroNCAP Score and Collision Rate, achieving an average NeuroNCAP Score of 2.25 and an average Collision Rate of 59.4. When additionally fine-tuned with RL on selected nuScenes training clips from the unified dataset, performance on stationary scenarios shows slight improvement; however, overall performance decreases. This is because the smaller nuScenes dataset adjusts the model to fit that specific data, thereby reducing its generalization ability in closed-loop evaluation. Notably, even Reasoning-VLA-3B surpasses competing methods, achieving more than a 4.3% improvement in average Collision Rate. These results demonstrate the strong generalization capability of our model, particularly in closed-loop environments that involve previously unseen scenarios.

Table 4: Ablation study of components contributions. R-VLA (Reasoning-VLA) is a 7B-parameter model. All experiments were conducted on our unified dataset and evaluated using a selected subset of the nuScenes dataset extracted from the unified dataset

<table><tbody><tr><th rowspan="2">Methods</th><td colspan="4">L2 (m) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td></tr><tr><th>Qwen2.5-VL-7B</th><td>0.46</td><td>1.33</td><td>2.55</td><td>1.45</td></tr><tr><th>R-VLA(w/o AQ)+SFT</th><td>0.09</td><td>0.31</td><td>0.55</td><td>0.32</td></tr><tr><th>R-VLA(w/o AQ)+SFT+RL</th><td>0.08</td><td>0.30</td><td>0.52</td><td>0.30</td></tr><tr><th>R-VLA(w/o AQ-Init)+SFT</th><td>0.06</td><td>0.27</td><td>0.55</td><td>0.29</td></tr><tr><th>R-VLA(w/o AQ-Init)+SFT+RL</th><td>0.08</td><td>0.23</td><td>0.50</td><td>0.27</td></tr><tr><th>R-VLA(w/o ARM)+SFT</th><td>0.06</td><td>0.28</td><td>0.53</td><td>0.29</td></tr><tr><th>R-VLA(w/o ARM)+SFT+RL</th><td>0.05</td><td>0.24</td><td>0.57</td><td>0.29</td></tr><tr><th>R-VLA+SFT</th><td>0.06</td><td>0.23</td><td>0.48</td><td>0.26</td></tr><tr><th>R-VLA+SFT+RL</th><td>0.05</td><td>0.20</td><td>0.44</td><td>0.23</td></tr></tbody></table>

#### 5.2.3 Generalized Performance

To evaluate the generalization capability of Reasoning-VLA, we tested Reasoning-VLA-7B on eight sub-datasets. As shown in Table 3, our results demonstrate that Reasoning-VLA exhibits strong generalization performance. The model was trained on the unified dataset and evaluated separately on each sub-dataset. We observed that the L2 performance for each sub-dataset closely matches that of the overall unified validation dataset. The variance of the L2 values is minimal, with variance values of 0.012 and 0.014 for average L2 in Reasoning-VLA-7B with SFT and Reasoning-VLA-7B with SFT plus RL, respectively. These results indicate that our method maintains robust generalization across different driving scenarios and vehicle configurations. Besides, the SFT+RL training strategy achieves an improvement compare to SFT alone, highlighting the effectiveness of RL.

### 5.3 Ablation Study

#### 5.3.1 Key Design Contributions

We conducted ablation studies to evaluate the effectiveness of key component designs, with results summarized in Table 4. Five experimental groups were constructed using different combinations of model components. As shown in Group-1 of Tab.4, the evaluation result of original Qwen2.5-VL-7B in poor performance. Differently, in Group-2, we replaced the learnable action queries with non-learnable queries and trained the model exclusively on the unified dataset. This modification yielded suboptimal results, achieving an average L2 of 0.32 and 0.30 with SFT and SFT+RL fine-tuning, respectively. In Group-3, only the learnable action query initialization was removed, resulting in a slight performance degradation compared to the full Reasoning-VLA model (Group-5). The results from Groups 2 and 3 suggest that learnable action queries significantly contribute to the model’s ability to generalize across diverse autonomous driving scenarios, thereby enhancing the overall performance of Reasoning-VLA. In Group-4, the action refinement module was removed, and actions were directly regressed from parallel action outputs using an MLP. This sequential strategy led to a modest performance drop relative to Group-5. Overall, these ablation studies demonstrate that each component of Reasoning-VLA contributes to its final performance, confirming the effectiveness of the proposed design.

More experiments (including ablation studies, generalization performance, efficiency performance and qualitative results) are illustrated in Appendix A.

## 6 Conclusions

This paper presents a general and efficient VLA framework based on a reasoning-enhanced vision-language model for autonomous driving. The proposed method introduces learnable action queries, initialized through Gaussian sampling from ground-truth trajectories, which interact with reasoning-augmented vision–language features to generate continuous action trajectories in parallel, thereby significantly improving inference efficiency. To enhance generalization, we unify eight existing autonomous driving datasets into a standardized, reasoning-based, and easy-to-use unified dataset. Following supervised fine-tuning (SFT) and reinforcement learning (RL) optimization, our method demonstrates outstanding performance and strong generalization capabilities in autonomous driving tasks.

Supplementary Material  

## 7 Appendix A

### 7.1 Zero-Shot Performance

Table 5: Zero shot performance on our unified dataset. The unified dataset is divided into two parts: the training set, which includes data from NAVSIM, Waymo, KITTI, and ONCE, and the evaluation sets, which consist of the remaining four sub-datasets along with the unified validation set.

<table><tbody><tr><th rowspan="3">Datasets</th><td colspan="4">Reasoning-VLA-7B + SFT</td><td colspan="4">Reasoning-VLA-7B + SFT + RL</td></tr><tr><td colspan="4">L2 (m) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td colspan="4">L2 (m) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td></tr><tr><th>nuScenes <cite>[nuscenes]</cite></th><td>0.07</td><td>0.24</td><td>0.52</td><td>0.28</td><td>0.08</td><td>0.26</td><td>0.50</td><td>0.28</td></tr><tr><th>Argoverse-V2 <cite>[argoverse]</cite></th><td>0.03</td><td>0.18</td><td>0.59</td><td>0.27</td><td>0.04</td><td>0.18</td><td>0.55</td><td>0.26</td></tr><tr><th>Mapillary <cite>[mapillary]</cite></th><td>0.08</td><td>0.59</td><td>1.09</td><td>0.59</td><td>0.07</td><td>0.57</td><td>1.01</td><td>0.55</td></tr><tr><th>IDD <cite>[idd]</cite></th><td>0.09</td><td>0.41</td><td>0.96</td><td>0.49</td><td>0.08</td><td>0.38</td><td>0.93</td><td>0.46</td></tr><tr><th>All</th><td>0.07</td><td>0.28</td><td>0.57</td><td>0.31</td><td>0.07</td><td>0.27</td><td>0.53</td><td>0.29</td></tr></tbody></table>

We also conducted a ”zero shot” experiment to further validate the generalization capability of our model. Specifically, the unified dataset was partitioned into distinct scenarios, where the NAVSIM, Waymo, KITTI, and ONCE sub-datasets were used for training, and the remaining four sub-datasets served as validation sets. As shown in Table 5, our method exhibits strong generalization performance on unseen datasets. This experiment confirms that the proposed Reasoning-VLA possesses robust adaptability to new driving scenarios and tasks, highlighting its potential as a general-purpose autonomous driving framework.

### 7.2 Performance on NAVSIM

Moreover, Tab.6 demonstrates the evaluation results on the NAVSIM evaluation. Compared to the Para-Drive method, our approach achieves respective improvements of 0.8, 5.1, 1.4, and 7.7 in DAC, TTC, EP, and PDMS metrics. Overall, the proposed model consistently delivers accurate and reliable predictions across the NAVSIM evaluations, establishing its state-of-the-art performance and strong generalization capability.

Table 6: Performance on the NAVSIM. \*: Sourced from \[navsim\].

| Methods | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comfort $\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| TransFuser \[transfuser\] \* | 97.7 | 92.8 | 92.8 | 100 | 79.2 | 84.0 |
| UniAD \[uniad\] \* | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| Para-Drive \[paradrive\] \* | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 |
| Reasoning-VLA-7B | 97.8 | 93.2 | 98.1 | 99.8 | 80.7 | 91.7 |

### 7.3 More Ablation Studies

#### 7.3.1 The Source of Performance: Generalization Ability or Data Contribution?

To further demonstrate that the SOTA performance of Reasoning-VLA arises from its generalization capabilities rather than from reliance on a specific dataset, we conducted two types of experiments, as shown in Tables 7 and 8. We evaluated two types of fine-tuned models:

Reasoning-VLA Fine-tuned on the nuScenes Dataset: The Reasoning-VLA (3B and 7B) models were fine-tuned exclusively on the selected nuScenes subset extracted from the unified dataset.

Reasoning-VLA Fine-tuned on the Unified Dataset: The Reasoning-VLA (3B and 7B) models were fine-tuned on the entire unified dataset.

Open-loop evaluations were performed on the corresponding nuScenes validation subset of the unified dataset. As shown in Table 7, the Reasoning-VLA models fine-tuned on the unified dataset outperform those fine-tuned solely on the nuScenes subset in terms of average L2 error and collision rate. Specifically, Reasoning-VLA-7B fine-tuned on the selected nuScenes subset achieves an average L2 error of 0.25 and a collision rate of 0.10, which are 8.7% and 25% lower, respectively, than the general Reasoning-VLA-7B fine-tuned on the unified dataset. Closed-loop evaluations, summarized in Table 8, further indicate that models fine-tuned on the unified dataset outperform those trained only on the nuScenes subset across all metrics. These results confirm that Reasoning-VLA possesses strong generalization capabilities in autonomous driving scenarios, comparable to those observed in VLMs.

Table 7: Generalization performance on the Open-loop Metrics.

<table><tbody><tr><th rowspan="2">Methods</th><td colspan="4">L2 (m) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td colspan="4">Collision Rate (%) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td></tr><tr><th colspan="9">Reasoning-VLA Finetuned with nuScenes Dataset</th></tr><tr><th>Reasoning-VLA-3B</th><td>0.10</td><td>0.38</td><td>0.51</td><td>0.33</td><td>0.05</td><td>0.13</td><td>0.27</td><td>0.15</td></tr><tr><th>Reasoning-VLA-7B</th><td>0.07</td><td>0.23</td><td>0.46</td><td>0.25</td><td>0.01</td><td>0.08</td><td>0.20</td><td>0.10</td></tr><tr><th colspan="9">Reasoning-VLA Finetuned with Our Unified Dataset</th></tr><tr><th>Reasoning-VLA-3B</th><td>0.08</td><td>0.33</td><td>0.48</td><td>0.30</td><td>0.04</td><td>0.13</td><td>0.23</td><td>0.13</td></tr><tr><th>Reasoning-VLA-7B</th><td>0.05</td><td>0.20</td><td>0.44</td><td>0.23</td><td>0.01</td><td>0.07</td><td>0.15</td><td>0.08</td></tr></tbody></table>

Table 8: Generalization performance on the Closed-loop Metrics.

<table><tbody><tr><th rowspan="2">Methods</th><td colspan="4">NeuroNCAP Score <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td colspan="4">Collision Rate (%) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><td>Stationary</td><td>Frontal</td><td>Side</td><td>Avg.</td><td>Stationary</td><td>Frontal</td><td>Side</td><td>Avg.</td></tr><tr><th colspan="9">Reasoning-VLA Finetuned with nuScenes Dataset</th></tr><tr><th>Reasoning-VLA-3B</th><td>1.67</td><td>2.16</td><td>1.83</td><td>1.89</td><td>69.0</td><td>63.3</td><td>66.6</td><td>66.3</td></tr><tr><th>Reasoning-VLA-7B</th><td>1.79</td><td>2.44</td><td>2.12</td><td>2.12</td><td>61.5</td><td>57.1</td><td>65.1</td><td>61.3</td></tr><tr><th colspan="9">Reasoning-VLA Finetuned with Our Unified Dataset</th></tr><tr><th>Reasoning-VLA-3B</th><td>1.88</td><td>2.29</td><td>1.94</td><td>2.04</td><td>63.7</td><td>60.4</td><td>64.1</td><td>62.7</td></tr><tr><th>Reasoning-VLA-7B</th><td>1.93</td><td>2.57</td><td>2.24</td><td>2.25</td><td>59.8</td><td>56.0</td><td>62.2</td><td>59.4</td></tr></tbody></table>

#### 7.3.2 Inference Efficiency

To evaluate the inference efficiency of Reasoning-VLA, we conducted experiments summarized in Table 9. Compared to existing autoregression-based VLMs, our method achieves superior performance using the same backbone. Reasoning-VLA can generate multiple future trajectories (e.g., 6 or 10 trajectories) in a single inference step, whereas autoregression-based VLA/VLMs must generate these trajectories sequentially. Even when employing the efficient bin-tokenizer proposed by OpenVLA \[openvla\] and $\pi_{0}$ \[pi0\], these methods require at least 12 to 20 steps to generate the desired trajectories, including both reasoning and trajectory tokens. Our experiments show that Reasoning-VLA achieves a generation speed of 0.089s per inference for 10 trajectories using vLLM, which is approximately 61 times faster than the autoregression-based Qwen2.5-VL-7B for the same number of trajectories. These results clearly demonstrate the superior inference efficiency of the Reasoning-VLA design.

Table 9: The Efficiency Comparisons. Steps: Theoretical number of VLM inference steps required to complete a single prediction process. Speed(s): Measured inference time to generate a complete prediction process. All experiments were conducted on an NVIDIA H200 GPU using vLLM. Traj: Number of predicted trajectories.

| Methods | Steps | Speed(s) |
| --- | --- | --- |
| Qwen2.5-VL-7B(6 Traj) | $\gg 12$ | 5.396 |
| Qwen2.5-VL-7B(10 Traj) | $\gg 20$ | 5.472 |
| Reasoning-VLA-7B(6 Traj) | 1 | 0.081 |
| Reasoning-VLA-7B(10 Traj) | 1 | 0.089 |

#### 7.3.3 Model Size

As is well known, the performance of LLMs and VLMs generally improves with an increase in model parameters. To analyze the impact of model size, we compare the 3B and 7B variants of our Reasoning-VLA. As shown in Tables 1 and 2, the Reasoning-VLA-7B model achieves superior performance, with an average L2 error of 0.23 and an average NeuroNCAP Score of 2.25, representing improvements of 30.4% and 9.3%, respectively, over the Reasoning-VLA-3B model. This performance gap indicates that larger models inherently possess stronger representational capacity, enabling them to capture more complex patterns and achieve better results.

### 7.4 Qualitative Results of Action Trajectories

We also provide qualitative results to further demonstrate the effectiveness of Reasoning-VLA. As illustrated in Fig.4, the visualization of predicted action trajectories across eight datasets highlights the strong generalization capability of our method. Notably, Reasoning-VLA produces consistent and accurate trajectory predictions even in previously unseen scenarios, confirming its robustness and adaptability.

![[x3 3.png|Refer to caption]]

Figure 4: Qualitative Results of Action Trajectories. Reasoning-VLA predictions on eight different datasets.Red lines denote GT trajectories while green lines represent predicted trajectories.

## 8 Appendix B

### 8.1 Unified Dataset

Existing individual autonomous driving datasets are often limited in scope, providing narrow coverage of the diverse scenarios encountered in real-world driving. To address this, we aggregate eight public datasets to construct a unified, reasoning-intensive dataset designed to support Chain-of-Thought generation with a strong reasoning VLM. This unified dataset is organized into a coherent, easy-to-use format to facilitate model training and enhance the generalization capability of Reasoning-VLA.

The processing pipeline for the unified dataset is illustrated in Fig. 5. First, all source datasets are converted into a standardized data format. The resulting image-text pairs are then input into a VLM, which generates detailed reasoning content according to a predefined labeling protocol. This reasoning output undergoes a rule-based verification process, followed by human review. During human verification, annotators assess the clips along with their associated labels to select the final set of high-quality data.

![[x4 3.png|Refer to caption]]

Figure 5: Pipeline for generating the unified reasoning dataset.

### 8.2 CoT Reasoning Unifided Dataset Format and Prompt

During the SFT stage of our method, we designed a structured input prompt to facilitate the generation of high-quality chain-of-thought (CoT) reasoning data. The prompt template is presented as follows:

[⬇](data:text/plain;base64,CiB7J3JvbGUnOiAnc3lzdGVtJywgJ2NvbnRlbnQnOiBbeyd0eXBlJzogJ3RleHQnLCAndGV4dCc6ICdZb3UgYXJlIGEgaGVscGZ1bCBhc3Npc3RhbnQnfV19LAoKIHsncm9sZSc6ICd1c2VyJywgJ2NvbnRlbnQnOiBbeyd0eXBlJzogJ2ltYWdlJywgJ2ltYWdlJzogJ251U2NlbmNlc180NDFfNDAwMC5DQU1fRlJPTlQucG5nJ30sIHsndHlwZSc6ICdpbWFnZScsICdpbWFnZSc6ICdudVNjZW5jZXNfNDQxXzQwMDAuQ0FNX0xFRlQucG5nJ30sIHsndHlwZSc6ICdpbWFnZScsICdpbWFnZSc6ICdudVNjZW5jZXNfNDQxXzQwMDAuQ0FNX1JJR0hULnBuZyd9LCB7J3R5cGUnOiAndGV4dCcsICd0ZXh0JzogIllvdSBhcmUgYW4gYXV0b25vbW91cyBkcml2aW5nIGFnZW50LiBZb3UgaGF2ZSBhY2Nlc3MgdG8gbXVsdGktdmlldyBjYW1lcmEgaW1hZ2VzIG9mIGEgdmVoaWNsZTogKDEpIGZyb250IHZpZXcgKHdoaWNoIHlvdSBzaG91bGQgZm9jdXMgb24gd2l0aCB0aGUgbW9zdCBhdHRlbnRpb24pIDxpbWFnZT4sICgyKSBmcm9udCByaWdodCB2aWV3IDxpbWFnZT4sIGFuZCAoMykgZnJvbnQgbGVmdCB2aWV3IDxpbWFnZT4uIFlvdXIgdGFzayBpcyB0byBkbyB5b3VyIGJlc3QgdG8gcHJlZGljdCBmdXR1cmUgd2F5cG9pbnRzIGZvciB0aGUgdmVoaWNsZSBvdmVyIHRoZSBuZXh0IDEwIHRpbWVzdGVwcywgZ2l2ZW4gdGhlIHZlaGljbGUncyBpbnRlbnQgaW5mZXJyZWQgZnJvbSB0aGUgaW1hZ2VzLiBQcm92aWRlZCBhcmUgdGhlIHByZXZpb3VzIGVnbyB2ZWhpY2xlIHN0YXR1cyByZWNvcmRlZCBvdmVyIHRoZSBsYXN0IDMuMCBzZWNvbmRzIChhdCAwLjUtc2Vjb25kIGludGVydmFscykuIFRoaXMgaW5jbHVkZXMgdGhlIHggYW5kIHkgY29vcmRpbmF0ZXMgb2YgdGhlIGVnbyB2ZWhpY2xlLiBQb3NpdGl2ZSB4IG1lYW5zIGZvcndhcmQgZGlyZWN0aW9uIHdoaWxlIHBvc2l0aXZlIHkgbWVhbnMgbGVmdHdhcmRzLiBUaGUgZGF0YSBpcyBwcmVzZW50ZWQgaW4gdGhlIGZvcm1hdCBbeCwgeV06KHQtMy4wcykgWy0yMS45NSwgLTAuMTFdLCBBY2NlbGVyYXRpb246IFggMC4yMiwgWSAwLjIxIG0vc14yLCBWZWxvY2l0eTogWCA2LjkzLCBZIDAuMCBtL3MsICh0LTIuNXMpIFstMTguNDIsIC0wLjA3XSwgQWNjZWxlcmF0aW9uOiBYIDAuMTksIFkgMC4yMiBtL3NeMiwgVmVsb2NpdHk6IFggNy4wMywgWSAwLjAgbS9zLCAodC0yLjBzKSBbLTE0Ljg4LCAtMC4wNV0sIEFjY2VsZXJhdGlvbjogWCAwLjI2LCBZIDAuMTUgbS9zXjIsIFZlbG9jaXR5OiBYIDcuMTYsIFkgMC4wIG0vcywgKHQtMS41cykgWy0xMS4yMiwgLTAuMDJdLCBBY2NlbGVyYXRpb246IFggMC4xNiwgWSAwLjE1IG0vc14yLCBWZWxvY2l0eTogWCA3LjI1LCBZIDAuMCBtL3MsICh0LTEuMHMpIFstNy4xNSwgMC4wMl0sIEFjY2VsZXJhdGlvbjogWCAtMC4yMSwgWSAwLjE2IG0vc14yLCBWZWxvY2l0eTogWCA3LjIzLCBZIDAuMCBtL3MsICh0LTAuNXMpIFstMy41MiwgMC4wMl0sIEFjY2VsZXJhdGlvbjogWCAtMC4zOSwgWSAwLjE5IG0vc14yLCBWZWxvY2l0eTogWCA3LjA5LCBZIDAuMCBtL3MsICh0LTAuMHMpIFswLjAsIDAuMF0sIEFjY2VsZXJhdGlvbjogWCAtMC41MSwgWSAtMC4wNiBtL3NeMiwgVmVsb2NpdHk6IFggNi44OCwgWSAwLjAgbS9zXG5cblBsZWFzZSB0aGluayBkZWVwbHkuIEVuZ2FnZSBpbiBhbiBpbnRlcm5hbCBkaWFsb2d1ZSBvdGhlciBuYXR1cmFsIGxhbmd1YWdlIHRob3VnaHQgZXhwcmVzc2lvbnMgSXQncyBhIHJlYXNvbmluZyBwcm9jZXNzLiBQcm92aWRlIHlvdXIgcmVhc29uaW5nIGJldHdlZW4gdGhlIDx0aGluaz4gPC90aGluaz4gdGFncywgYW5kIHRoZW4gZ2l2ZSB5b3VyIGFuc3dlciBiZXR3ZWVuIHRoZSA8YW5zd2VyPiA8L2Fuc3dlcj4gdGFncy4gUHJlZGljdGVkIGZ1dHVyZSBtb3ZlbWVudCBkZXRhaWxzIGZvciB0aGUgbmV4dCA1IHNlY29uZHMgKHNhbXBsZWQgYXQgMC41LXNlY29uZCBpbnRlcnZhbHMpLCBpbmNsdWRpbmcgQkVWIGxvY2F0aW9uIGluIHggYW5kIHkgZGlyZWN0aW9ucyAoaW4gbWV0ZXJzKS4gUG9zaXRpdmUgeCBtZWFucyBmb3J3YXJkIGRpcmVjdGlvbiB3aGlsZSBwb3NpdGl2ZSB5IG1lYW5zIGxlZnR3YXJkcy4gVGhlIG91dHB1dCBpcyBmb3JtYXR0ZWQgYXMgW3gsIHldLiJ9XX0sCgogeydyb2xlJzogJ2Fzc2lzdGFudCcsICdjb250ZW50JzogW3sndHlwZSc6ICd0ZXh0JywgJ3RleHQnOiAiPHRoaW5rPlxuTGV0IG1lIHRoaW5rLiBUbyByZXBocmFzZSB0aGUgcXVlc3Rpb24gaW4gYSB3YXkgdGhhdCByZXF1aXJlcyBDaGFpbi1vZi1UaG91Z2h0IHJlYXNvbmluZyB3aXRoIG51bWVyaWNhbCBvciBtYXRoZW1hdGljYWwgZXhwcmVzc2lvbnMsIHdlIHNob3VsZCBicmVhayBkb3duIHRoZSBwcmVkaWN0aW9uIG9mIGZ1dHVyZSB3YXlwb2ludHMgaW50byBzbWFsbGVyIHN0ZXBzLCBzdGFydGluZyBmcm9tIHVuZGVyc3RhbmRpbmcgdGhlIHByb3ZpZGVkIGRhdGEgYW5kIGFwcGx5aW5nIHJlbGV2YW50IHBoeXNpY3MgZXF1YXRpb25zLiBcblxuVGhlIG9yaWdpbmFsIHF1ZXN0aW9uIGFza3MgZm9yIHByZWRpY3RpbmcgdGhlIGZ1dHVyZSB3YXlwb2ludHMgZGlyZWN0bHkgZnJvbSB0aGUgZ2l2ZW4gdmVoaWNsZSBzdGF0dXMsIGJ1dCBsZXQncyBkZXJpdmUgdGhlIHdheXBvaW50cyB0aHJvdWdoIGludGVybWVkaWF0ZSBjYWxjdWxhdGlvbnMuIFxuIFxuT2gsIEkgc2VlLiBUaGUgcXVlc3Rpb24gbm93IG5lZWRzIHRvIGJlIGZyYW1lZCBpbiBzdWNoIGEgd2F5IHRoYXQgdGhlIHJlc3BvbmRlciB1bmRlcnN0YW5kcyB0aGV5IG5lZWQuXG48L3RoaW5rPlxuPGFuc3dlcj48fHBsYWNlX2hvbGRlcnw+PHxwbGFjZV9ob2xkZXJ8Pjx8cGxhY2VfaG9sZGVyfD48fHBsYWNlX2hvbGRlcnw+PHxwbGFjZV9ob2xkZXJ8Pjx8cGxhY2VfaG9sZGVyfD48fHBsYWNlX2hvbGRlcnw+PHxwbGFjZV9ob2xkZXJ8Pjx8cGxhY2VfaG9sZGVyfD48fHBsYWNlX2hvbGRlcnw+PHxwbGFjZV9ob2xkZXJ8Pjx8cGxhY2VfaG9sZGVyfD48fHBsYWNlX2hvbGRlcnw+PHxwbGFjZV9ob2xkZXJ8Pjx8cGxhY2VfaG9sZGVyfD48fHBsYWNlX2hvbGRlcnw+PHxwbGFjZV9ob2xkZXJ8Pjx8cGxhY2VfaG9sZGVyfD48fHBsYWNlX2hvbGRlcnw+PHxwbGFjZV9ob2xkZXJ8PjwvYW5zd2VyPiJ9XX0sCiAgeydhY3Rpb25zJzogYXJyYXkoW1swLiAgICAgICAgLCAwLiAgICAgICAgXSwKICAgICAgIFswLjQwMDQ2NTYxLCAwLjM5NzE2Mjg0XSwKICAgICAgIFswLjM5MjIxMzgxLCAwLjMzNTkzNzQxXSwKICAgICAgIFswLjM5MjQzNDk3LCAwLjMxMjg0MTQ5XSwKICAgICAgIFswLjM4ODc1NjY4LCAwLjI4OTQyODA1XSwKICAgICAgIFswLjM4NDY3MDQ4LCAwLjI3MzExNjk1XSwKICAgICAgIFswLjM4NDA5NDA3LCAwLjI2ODI5MjY3XSwKICAgICAgIFswLjM4ODI5MTUxLCAwLjI3NDM3MDc4XSwKICAgICAgIFswLjM5MzExMjcgLCAwLjI4Nzk3OTI0XSwKICAgICAgIFswLjM5OTY4MzYyLCAwLjI5OTkyOTI1XV0pfQo=)

{’role’: ’system’, ’content’: \[{’type’: ’text’, ’text’: ’You are a helpful assistant’}\]},

{’role’: ’user’, ’content’: \[{’type’: ’image’, ’image’: ’nuScences\_441\_4000.CAM\_FRONT.png’}, {’type’: ’image’, ’image’: ’nuScences\_441\_4000.CAM\_LEFT.png’}, {’type’: ’image’, ’image’: ’nuScences\_441\_4000.CAM\_RIGHT.png’}, {’type’: ’text’, ’text’: "You are an autonomous driving agent. You have access to multi-view camera images of a vehicle: (1) front view (which you should focus on with the most attention) <image>, (2) front right view <image>, and (3) front left view <image>. Your task is to do your best to predict future waypoints for the vehicle over the next 10 timesteps, given the vehicle’s intent inferred from the images. Provided are the previous ego vehicle status recorded over the last 3.0 seconds (at 0.5-second intervals). This includes the x and y coordinates of the ego vehicle. Positive x means forward direction while positive y means leftwards. The data is presented in the format \[x, y\]:(t-3.0s) \[-21.95, -0.11\], Acceleration: X 0.22, Y 0.21 m/s^2, Velocity: X 6.93, Y 0.0 m/s, (t-2.5s) \[-18.42, -0.07\], Acceleration: X 0.19, Y 0.22 m/s^2, Velocity: X 7.03, Y 0.0 m/s, (t-2.0s) \[-14.88, -0.05\], Acceleration: X 0.26, Y 0.15 m/s^2, Velocity: X 7.16, Y 0.0 m/s, (t-1.5s) \[-11.22, -0.02\], Acceleration: X 0.16, Y 0.15 m/s^2, Velocity: X 7.25, Y 0.0 m/s, (t-1.0s) \[-7.15, 0.02\], Acceleration: X -0.21, Y 0.16 m/s^2, Velocity: X 7.23, Y 0.0 m/s, (t-0.5s) \[-3.52, 0.02\], Acceleration: X -0.39, Y 0.19 m/s^2, Velocity: X 7.09, Y 0.0 m/s, (t-0.0s) \[0.0, 0.0\], Acceleration: X -0.51, Y -0.06 m/s^2, Velocity: X 6.88, Y 0.0 m/s\\n\\nPlease think deeply. Engage in an internal dialogue other natural language thought expressions It’s a reasoning process. Provide your reasoning between the <think> </think> tags, and then give your answer between the <answer> </answer> tags. Predicted future movement details for the next 5 seconds (sampled at 0.5-second intervals), including BEV location in x and y directions (in meters). Positive x means forward direction while positive y means leftwards. The output is formatted as \[x, y\]."}\]},

{’role’: ’assistant’, ’content’: \[{’type’: ’text’, ’text’: "<think>\\nLet me think. To rephrase the question in a way that requires Chain-of-Thought reasoning with numerical or mathematical expressions, we should break down the prediction of future waypoints into smaller steps, starting from understanding the provided data and applying relevant physics equations. \\n\\nThe original question asks for predicting the future waypoints directly from the given vehicle status, but let’s derive the waypoints through intermediate calculations. \\n \\nOh, I see. The question now needs to be framed in such a way that the responder understands they need.\\n</think>\\n<answer><|place\_holder|><|place\_holder|><|place\_holder|><|place\_holder|><|place\_holder|><|place\_holder|><|place\_holder|><|place\_holder|><|place\_holder|><|place\_holder|><|place\_holder|><|place\_holder|><|place\_holder|><|place\_holder|><|place\_holder|><|place\_holder|><|place\_holder|><|place\_holder|><|place\_holder|><|place\_holder|></answer>"}\]},

{’actions’: array(\[\[0., 0. \],

\[0.40046561, 0.39716284\],

\[0.39221381, 0.33593741\],

\[0.39243497, 0.31284149\],

\[0.38875668, 0.28942805\],

\[0.38467048, 0.27311695\],

\[0.38409407, 0.26829267\],

\[0.38829151, 0.27437078\],

\[0.3931127, 0.28797924\],

\[0.39968362, 0.29992925\]\])}

### 8.3 Training Details

The training details of SFT and RL are illustrated below.

#### 8.3.1 SFT

[⬇](data:text/plain;base64,CmJhdGNoX3NpemUgOApncmFkaWVudF9hY2N1bXVsYXRpb25fc3RlcHMgMgpsZWFybmluZ19yYXRlIDVlLTUKYmYxNgpncmFkaWVudF9jaGVja3BvaW50aW5nIHRydWUKYXR0bl9pbXBsZW1lbnRhdGlvbiBmbGFzaF9hdHRlbnRpb25fMgpudW1fdHJhaW5fZXBvY2hzIDQKbWF4X2dyYWRfbm9ybSA1Cgp6ZXJvMiBjb25maWc6CnsKICAgICJmcDE2IjogewogICAgICAgICJlbmFibGVkIjogImF1dG8iLAogICAgICAgICJsb3NzX3NjYWxlIjogMCwKICAgICAgICAibG9zc19zY2FsZV93aW5kb3ciOiAxMDAwLAogICAgICAgICJpbml0aWFsX3NjYWxlX3Bvd2VyIjogMTYsCiAgICAgICAgImh5c3RlcmVzaXMiOiAyLAogICAgICAgICJtaW5fbG9zc19zY2FsZSI6IDEKICAgIH0sCiAgICAiYmYxNiI6IHsKICAgICAgICAiZW5hYmxlZCI6ICJhdXRvIgogICAgfSwKICAgICJvcHRpbWl6ZXIiOiB7CiAgICAgICAgInR5cGUiOiAiQWRhbVciLAogICAgICAgICJwYXJhbXMiOiB7CiAgICAgICAgICAgICJsciI6ICJhdXRvIiwKICAgICAgICAgICAgImJldGFzIjogImF1dG8iLAogICAgICAgICAgICAiZXBzIjogImF1dG8iLAogICAgICAgICAgICAid2VpZ2h0X2RlY2F5IjogImF1dG8iCiAgICAgICAgfQogICAgfSwKICAgICJ6ZXJvX29wdGltaXphdGlvbiI6IHsKICAgICAgICAic3RhZ2UiOiAyLAogICAgICAgICJvZmZsb2FkX29wdGltaXplciI6IHsKICAgICAgICAgICAgImRldmljZSI6ICJub25lIiwKICAgICAgICAgICAgInBpbl9tZW1vcnkiOiB0cnVlCiAgICAgICAgfSwKICAgICAgICAiYWxsZ2F0aGVyX3BhcnRpdGlvbnMiOiB0cnVlLAogICAgICAgICJhbGxnYXRoZXJfYnVja2V0X3NpemUiOiAyZTgsCiAgICAgICAgIm92ZXJsYXBfY29tbSI6IGZhbHNlLAogICAgICAgICJyZWR1Y2Vfc2NhdHRlciI6IHRydWUsCiAgICAgICAgInJlZHVjZV9idWNrZXRfc2l6ZSI6IDJlOCwKICAgICAgICAiY29udGlndW91c19ncmFkaWVudHMiOiB0cnVlCiAgICB9LAogICAgImdyYWRpZW50X2FjY3VtdWxhdGlvbl9zdGVwcyI6ICJhdXRvIiwKICAgICJncmFkaWVudF9jbGlwcGluZyI6ICJhdXRvIiwKICAgICJzdGVwc19wZXJfcHJpbnQiOiAxMDAsCiAgICAidHJhaW5fYmF0Y2hfc2l6ZSI6ICJhdXRvIiwKICAgICJ0cmFpbl9taWNyb19iYXRjaF9zaXplX3Blcl9ncHUiOiAiYXV0byIsCiAgICAid2FsbF9jbG9ja19icmVha2Rvd24iOiBmYWxzZQp9)

batch\_size 8

gradient\_accumulation\_steps 2

learning\_rate 5e-5

bf16

gradient\_checkpointing true

attn\_implementation flash\_attention\_2

num\_train\_epochs 4

max\_grad\_norm 5

zero2 config:

{

"fp16": {

"enabled": "auto",

"loss\_scale": 0,

"loss\_scale\_window": 1000,

"initial\_scale\_power": 16,

"hysteresis": 2,

"min\_loss\_scale": 1

},

"bf16": {

"enabled": "auto"

},

"optimizer": {

"type": "AdamW",

"params": {

"lr": "auto",

"betas": "auto",

"eps": "auto",

"weight\_decay": "auto"

}

},

"zero\_optimization": {

"stage": 2,

"offload\_optimizer": {

"device": "none",

"pin\_memory": true

},

"allgather\_partitions": true,

"allgather\_bucket\_size": 2e8,

"overlap\_comm": false,

"reduce\_scatter": true,

"reduce\_bucket\_size": 2e8,

"contiguous\_gradients": true

},

"gradient\_accumulation\_steps": "auto",

"gradient\_clipping": "auto",

"steps\_per\_print": 100,

"train\_batch\_size": "auto",

"train\_micro\_batch\_size\_per\_gpu": "auto",

"wall\_clock\_breakdown": false

}

#### 8.3.2 RL

[⬇](data:text/plain;base64,Cm1heF9wcm9tcHRfbGVuZ3RoIDE2Mzg0Cm1heF9jb21wbGV0aW9uX2xlbmd0aCA3NjgKYmF0Y2hfc2l6ZSA4CmdyYWRpZW50X2FjY3VtdWxhdGlvbl9zdGVwcyAyCmxlYXJuaW5nX3JhdGUgMWUtNgpscl9zY2hlZHVsZXJfdHlwZSAiY29zaW5lIgp3ZWlnaHRfZGVjYXkgMC4wMQpiZjE2CmdyYWRpZW50X2NoZWNrcG9pbnRpbmcgdHJ1ZQp0ZW1wb3JhbCB0cnVlCmxlbl9jb250cm9sIHRydWUKYXR0bl9pbXBsZW1lbnRhdGlvbiBmbGFzaF9hdHRlbnRpb25fMgptYXhfcGl4ZWxzIDQwMTQwOApudW1fdHJhaW5fZXBvY2hzIDEKYmV0YSAwLjA0Cm1heF9ncmFkX25vcm0KbnVtX2dlbmVyYXRpb25zIDgKCnplcm8zIGNvbmZpZzoKCiAgICAiZnAxNiI6IHsKICAgICAgICAiZW5hYmxlZCI6ICJhdXRvIiwKICAgICAgICAibG9zc19zY2FsZSI6IDAsCiAgICAgICAgImxvc3Nfc2NhbGVfd2luZG93IjogMTAwMCwKICAgICAgICAiaW5pdGlhbF9zY2FsZV9wb3dlciI6IDE2LAogICAgICAgICJoeXN0ZXJlc2lzIjogMiwKICAgICAgICAibWluX2xvc3Nfc2NhbGUiOiAxCiAgICB9LAogICAgImJmMTYiOiB7CiAgICAgICAgImVuYWJsZWQiOiAiYXV0byIKICAgIH0sCgogICAgInplcm9fb3B0aW1pemF0aW9uIjogewogICAgICAgICJzdGFnZSI6IDMsCiAgICAgICAgIm9mZmxvYWRfb3B0aW1pemVyIjogewogICAgICAgICAgICAiZGV2aWNlIjogIm5vbmUiLAogICAgICAgICAgICAicGluX21lbW9yeSI6IHRydWUKICAgICAgICB9LAogICAgICAgICJvZmZsb2FkX3BhcmFtIjogewogICAgICAgICAgICAiZGV2aWNlIjogIm5vbmUiLAogICAgICAgICAgICAicGluX21lbW9yeSI6IHRydWUKICAgICAgICB9LAogICAgICAgICJvdmVybGFwX2NvbW0iOiB0cnVlLAogICAgICAgICJjb250aWd1b3VzX2dyYWRpZW50cyI6IHRydWUsCiAgICAgICAgInN1Yl9ncm91cF9zaXplIjogMWU5LAogICAgICAgICJyZWR1Y2VfYnVja2V0X3NpemUiOiAiYXV0byIsCiAgICAgICAgInN0YWdlM19wcmVmZXRjaF9idWNrZXRfc2l6ZSI6ICJhdXRvIiwKICAgICAgICAic3RhZ2UzX3BhcmFtX3BlcnNpc3RlbmNlX3RocmVzaG9sZCI6ICJhdXRvIiwKICAgICAgICAic3RhZ2UzX21heF9saXZlX3BhcmFtZXRlcnMiOiAxZTksCiAgICAgICAgInN0YWdlM19tYXhfcmV1c2VfZGlzdGFuY2UiOiAxZTksCiAgICAgICAgInN0YWdlM19nYXRoZXJfMTZiaXRfd2VpZ2h0cwogICAgICAgICAgX29uX21vZGVsX3NhdmUiOiB0cnVlCiAgICB9LAoKICAgICJncmFkaWVudF9hY2N1bXVsYXRpb25fc3RlcHMiOiAiYXV0byIsCiAgICAiZ3JhZGllbnRfY2xpcHBpbmciOiAiYXV0byIsCiAgICAic3RlcHNfcGVyX3ByaW50IjogMTAwLAogICAgInRyYWluX2JhdGNoX3NpemUiOiAiYXV0byIsCiAgICAidHJhaW5fbWljcm9fYmF0Y2hfc2l6ZV9wZXJfZ3B1IjogImF1dG8iLAogICAgIndhbGxfY2xvY2tfYnJlYWtkb3duIjogZmFsc2UKfQ==)

max\_prompt\_length 16384

max\_completion\_length 768

batch\_size 8

gradient\_accumulation\_steps 2

learning\_rate 1e-6

lr\_scheduler\_type "cosine"

weight\_decay 0.01

bf16

gradient\_checkpointing true

temporal true

len\_control true

attn\_implementation flash\_attention\_2

max\_pixels 401408

num\_train\_epochs 1

beta 0.04

max\_grad\_norm

num\_generations 8

zero3 config:

"fp16": {

"enabled": "auto",

"loss\_scale": 0,

"loss\_scale\_window": 1000,

"initial\_scale\_power": 16,

"hysteresis": 2,

"min\_loss\_scale": 1

},

"bf16": {

"enabled": "auto"

},

"zero\_optimization": {

"stage": 3,

"offload\_optimizer": {

"device": "none",

"pin\_memory": true

},

"offload\_param": {

"device": "none",

"pin\_memory": true

},

"overlap\_comm": true,

"contiguous\_gradients": true,

"sub\_group\_size": 1e9,

"reduce\_bucket\_size": "auto",

"stage3\_prefetch\_bucket\_size": "auto",

"stage3\_param\_persistence\_threshold": "auto",

"stage3\_max\_live\_parameters": 1e9,

"stage3\_max\_reuse\_distance": 1e9,

"stage3\_gather\_16bit\_weights

\_on\_model\_save": true

},

"gradient\_accumulation\_steps": "auto",

"gradient\_clipping": "auto",

"steps\_per\_print": 100,

"train\_batch\_size": "auto",

"train\_micro\_batch\_size\_per\_gpu": "auto",

"wall\_clock\_breakdown": false

}

### 8.4 Reward Function Implements

Our reward functions significantly influence the RL process. For trajectory reward function:

$$
r_{\text{traj}}=1-\frac{1}{N}\sum_{i=1}^{N}\gamma^{i}\left(\alpha(x^{i}-x_{\text{gt}}^{i})^{2}+\beta(y^{i}-y_{\text{gt}}^{i})^{2}\right)
$$

normally, we can select hyper-parameters (1 in this function) to achieve an accurate reward. Some times we need change the 1 to large numbers such as 2. We can also replaced with another easy way:

$$
r_{\text{traj}}=1-min(1.0,\frac{1}{N}\sum_{i=1}^{N}\gamma^{i}\left(\alpha(x^{i}-x_{\text{gt}}^{i})^{2}+\beta(y^{i}-y_{\text{gt}}^{i})^{2}\right))
$$
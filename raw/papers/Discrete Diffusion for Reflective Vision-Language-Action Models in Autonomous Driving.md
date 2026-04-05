---
title: "Discrete Diffusion for Reflective Vision-Language-Action Models in Autonomous Driving"
source: "https://arxiv.org/html/2509.20109v1"
author:
published:
created: 2026-04-05
description:
tags:
  - "clippings"
---
Pengxiang Li <sup>1*‡</sup>, Yinan Zheng <sup>2*‡</sup>, Yue Wang <sup>1*</sup>, Huimin Wang <sup>1†</sup>,  
  
Hang Zhao <sup>2</sup>, Jingjing Liu <sup>2</sup>, Xianyuan Zhan <sup>2</sup>, Kun Zhan <sup>1</sup>, Xianpeng Lang <sup>1</sup>  
  
<sup>1</sup> LiAuto   <sup>2</sup> Tsinghua University

###### Abstract

End-to-End (E2E) solutions have emerged as a mainstream approach for autonomous driving systems, with Vision-Language-Action (VLA) models representing a new paradigm that leverages pre-trained multimodal knowledge from Vision-Language Models (VLMs) to interpret and interact with complex real-world environments. However, these methods remain constrained by the limitations of imitation learning, which struggles to inherently encode physical rules during training. Existing approaches often rely on complex rule-based post-refinement, employ reinforcement learning that remains largely limited to simulation, or utilize diffusion guidance that requires computationally expensive gradient calculations. To address these challenges, we introduce ReflectDrive, a novel learning-based framework that integrates a reflection mechanism for safe trajectory generation via discrete diffusion. We first discretize the two-dimensional driving space to construct an action codebook, enabling the use of pre-trained Diffusion Language Models for planning tasks through fine-tuning. Central to our approach is a safety-aware reflection mechanism that performs iterative self-correction without gradient computation. Our method begins with goal-conditioned trajectory generation to model multi-modal driving behaviors. Based on this, we apply local search methods to identify unsafe tokens and determine feasible solutions, which then serve as safe anchors for inpainting-based regeneration. Evaluated on the NAVSIM benchmark, ReflectDrive demonstrates significant advantages in safety-critical trajectory generation, offering a scalable and reliable solution for autonomous driving systems.

<sup>*</sup> <sup>†</sup> <sup>‡</sup>

## 1 Introduction

Autonomous driving (AD) is guiding the transportation industry toward a safer and more efficient future [^30]. Within this trend, End-to-End (E2E) systems [^12] [^5] have emerged as the mainstream alternative to traditional modular designs [^3], which are prone to error accumulation between interdependent modules. They have also largely replaced rule-based methods [^10] [^32] that demand extensive human engineering effort. Meanwhile, Vision-Language-Action (VLA) models [^19] [^14] offer a new solution by incorporating pre-trained knowledge from Vision-Language Models (VLMs) [^13] [^2]. Equipped with enhanced generalization capabilities, VLA models can interpret visual scenes and understand human instructions to directly output planning trajectories, thereby improving adaptability in challenging situations.

However, eixsting learning-based methods does not resolve the core challenge in imitation learning-based driving systems. Specifically, behavior cloning fails to inherently encode inviolable physical rules, such as collision avoidance or adherence to drivable areas [^24]. As a result, a generated trajectory may be highly probable under the model’s distribution yet still violate critical safety constraints. Consequently, existing deployed solutions often rely on significant human priors, such as trajectory anchors [^21] or rule-based generated paths [^8]. These priors offer a reliable initial solution for the learning system, but they also necessitate substantial post-processing, particularly in complex scenarios. Concurrently, more advanced solutions are emerging. Some methods integrate reinforcement learning [^17] [^18] [^15] [^7] with human-designed reward functions to enhance causal reasoning. However, most existing studies remain confined to the simulation level. From a deployment perspective, these approaches typically require unsafe online rollouts and suffer from training instability, especially in large-scale models [^37]. Although guidance mechanisms in diffusion models provide a promising alternative by enabling controllable generation during inference [^38] [^16] [^39], they often experience slow sampling speeds due to gradient computations and are highly sensitive to parameter tuning, which can lead to numerical instability.

To address these challenges, we pioneer the use of discrete diffusion [^1] for planning to meet the demand for verifiable and controllable E2E driving systems. A key advantage of this approach is its operation in a discrete action space, which facilitates the seamless incorporation of critical safety constraints through search, masking, and sampling techniques during trajectory generation. This results in a hybrid framework in which learned behaviors can be rigorously guided by prior knowledge, shifting away from black-box planning toward trustworthy and interpretable decision-making. Inspired by these insights, we propose ReflectDrive, a novel learning-based framework that integrates a reflection mechanism for safe trajectory generation via discrete diffusion. Specifically, we first discretize the two-dimensional driving space to construct a action codebook, enabling the representation of vehicle trajectories through discrete codebook embeddings. This representation allows us to leverage a pre-trained Diffusion Language Models (DLMs) [^36] [^26] for planning tasks via fine-tuning. The approach facilitates parallel decoding and bidirectional feature fusion within a unified architecture that supports scalable training. Based on this fine-tuned model, our reflection mechanism begins with goal-conditioned generation, where the goal point guides the generation process to capture diverse multi-modal driving behaviors. Furthermore, the framework integrates safety metrics to evaluate the generated multi-modal trajectories. For unsafe waypoints, we perform a local search to identify a feasible solution, which then serves as a safe anchor token for trajectory inpainting. The entire process operates without gradient computation, enabling parallel generation and the injection of safety constraints during trajectory regeneration. Evaluations on the real-world autonomous driving benchmark NAVSIM [^9] demonstrate the feasibility of employing discrete diffusion for trajectory generation. Equipped with our reflection mechanism, ReflectDrive achieves near human-level closed-loop performance. Our contributions are summarized as follows:

- We pioneer the application of discrete diffusion for E2E autonomous driving trajectory generation and integrate it into a VLA model for scalable training.
- We introduce reflection mechanism, a novel inference-time guidance framework specifically designed for the denoising process in discrete diffusion, integrating external safety validation with efficient discrete token optimization.
- We evaluate our method on real-world driving benchmarks, proving that the framework can enforce hard safety constraints without compromising behavioral coherence.

## 2 Related Work

#### End-to-End Autonomous Driving.

E2E methods [^12] [^5] have emerged as a promising solution to largely replace rule-based approaches due to their superior scalability. Recently, VLA models [^14] [^28] [^40] have arisen as a new paradigm, incorporating world knowledge from pre-trained VLMs to enhance performance in long-tail scenarios. Additionally, VLA architectures can accept human instructions to support human-preferred driving behaviors [^19], while language serves as an interpretable intermediate representation for improved explainability [^31] [^33].

#### Beyond Imitation Learning.

Current mainstream pipelines still operate within imitation learning-based frameworks, which suffer from causal confusion and lack verifiable safety guarantees. Many studies have attempted to address this issue, which can be broadly categorized as follows: 1) The model uses trajectory anchors, which are derived from clustered trajectory data or rule-based proposals, as conditioning inputs and is designed to predict offsets for further trajectory refinement [^8]. Hydra-MDP [^21] utilizes trajectory anchors as candidates for post-selection, while DiffusionDrive [^22] employs anchors as starting points and uses a pseudo-diffusion process for refinement. Although these methods exhibit improved reliability, they rely heavily on rule-based design. 2) Reinforcement learning methods enhance model capabilities through exploration [^29] [^20] [^4] [^24]; for instance, GIGAFLOW [^7] significantly improves performance via self-play in simulation. However, online rollouts are infeasible for real-world vehicle deployment, and simulation training faces the sim-to-real gap. Although recent advances in world models [^11] offer a potential solution, they still struggle with out-of-distribution simulation. 3) Other methods, such as guidance mechanisms for diffusion models, enable the injection of reward signals during the denoising process [^16] [^39]. Diffusion Planner [^38] represents a pioneering effort in applying diffusion models to closed-loop planning tasks. Although it utilizes guidance to adjust behavior during inference, the method relies on additional gradient computations, resulting in high computational cost. In this paper, we propose a novel reflection mechanism based on discrete diffusion that naturally incorporates safety constraints through search, masking, and inpainting during trajectory generation.

## 3 Preliminaries

### 3.1 Autonomous Driving Planning

We formulate the autonomous driving planning task as learning a conditional distribution $p(\tau\mid c)$, where the goal is to generate a future trajectory $\tau$. Each waypoint is expressed in the ego-vehicle frame, conditioned on a scene context $c$ that includes multi-view images, instructions, and ego-vehicle state. The primary challenge in planning is that trajectories must adhere to traffic rules and safety constraints, which is difficult for imitation learning-based methods due to the absence of explicit signals to ensure strict compliance with these requirements.

### 3.2 Discrete Diffusion

Discrete diffusion models [^1] [^25] [^23] have emerged as a powerful non-autoregressive paradigm for generating structured sequences. This process is defined by a forward corruption process and a learned reverse denoising process.

#### Forward and Reverse Process.

The forward process degrades a clean sequence of discrete tokens $\mathbf{y}=(\mathbf{y}_{1},\dots,\mathbf{y}_{i},\dots,\mathbf{y}_{L})$ over a series of $S$ timesteps. At each step $s\in\{1,\dots,S\}$, a noisy version of the sequence, $\tilde{\mathbf{y}}^{(s)}$, is created by masking a subset of the tokens in $\mathbf{y}$. Specifically, a binary mask $\mathbf{m}^{(s)}=(m^{(s)}_{1},\dots,m^{(s)}_{i},\dots,m^{(s)}_{L})\in\{0,1\}^{L}$ is sampled, and each token $\mathbf{y}_{i}$ is replaced with a special \[MASK\] token if $m^{(s)}_{i}=1$. The number of masked tokens is determined by a noise schedule, such as a cosine schedule, which typically increases the masking ratio as $s$ approaches $S$. The core learning task is to train a model $p_{\theta}$ to reverse this corruption. This model learns to predict the original tokens at the masked positions, conditioned on the unmasked tokens, the timestep $s$, and any external context $c$. The model is trained by minimizing the negative log-likelihood objective:

$$
\mathcal{L}(\theta)=\mathbb{E}_{\mathbf{y},c,s,\mathbf{m}^{(s)}}\left[-\sum_{i:\,m^{(s)}_{i}=1}\log p_{\theta}\!\big(\mathbf{y}_{i}\,\big|\,\tilde{\mathbf{y}}^{(s)},c,s\big)\right].
$$

#### Model Inference.

To generate a new sequence, the process starts with a fully masked sequence, $\tilde{\mathbf{y}}^{(S)}$. The model then iteratively refines this sequence for $S$ steps. In each step, the model predicts a probability distribution for the tokens at the masked positions. A subset of these predictions is then sampled and fixed, while the rest are re-masked for the next refinement step. A central advantage of this framework, and one especially critical to our work, is its capacity for inpainting, defined as the ability to reconstruct masked segments of a sequence while maintaining consistency with the context from unmasked tokens. Additionally, the discrete token structure supports efficient search and constraint integration, making it possible to guide trajectories using safety constraints.

## 4 Method

In this section, we present ReflectDrive, a novel learning-based framework that integrates a reflection mechanism to facilitate safe trajectory generation via discrete diffusion, as illustrated in Figure 1. We first introduce a trajectory discretization method tailored for integration into a masked diffusion process. A pre-trained diffusion language model is then employed for trajectory generation. Finally, we propose a reflection mechanism specifically designed to ensure safety during the trajectory generation process. This mechanism leverages diffusion inpainting and capitalizes on the advantages of discrete token spaces for efficient constraint-based search.

### 4.1 Discrete Diffusion for Autonomous Driving Planning

![[x1 4.png|Refer to caption]]

Figure 1: ReflectDrive Framework Overview.

#### Trajectory Discretization.

To represent continuous waypoints in a discrete format, we quantize each 2D coordinate $(x,y)$ by mapping its $x$ and $y$ values independently to the closest tokens in their respective 1D codebooks. We define a uniform 1D codebook $\mathcal{A}=\{a_{1},a_{2},\dots\}$ by discretizing a spatial range $[-M,M]$ with resolution $\Delta_{g}$. A quantizer $\mathcal{Q}$ maps a real value to its nearest token, and its inverse recovers the coordinate. Each 2D waypoint is thus represented by a token pair $(\mathbf{y}_{j,x},\mathbf{y}_{j,y})$, and the full trajectory becomes a flattened sequence $\mathbf{y}=\mathcal{Q}(\tau)=(\mathbf{y}_{1,x},\mathbf{y}_{1,y},\dots,\mathbf{y}_{N,x},\mathbf{y}_{N,y})\in\mathcal{A}^{2N}$. At first glance, discretization may appear to cause some loss in trajectory precision. However, in practical deployment, the resolution can be adjusted to control accuracy, or different codebook partitioning strategies can be employed. Most importantly, discretization facilitates efficient search for feasible solutions in the Bird’s-Eye View (BEV) space. Experimental results in Section 5.2 and Figure 3 further demonstrate that, with discrete representations, our reflection mechanism significantly enhances the safety of the generated trajectories.

#### Discrete Diffusion Model.

Based on our discretized trajectory representation, we instantiate the trajectory planner using the discrete diffusion framework described in Section 3. In practice, we employ a VLA model as the planner, initialized from a pre-trained Diffusion Language Model [^36] [^26] that exhibits strong pre-training performance in understanding driving scenarios. The model can generate a tokenized trajectory $\mathbf{y}$ conditioned on a scene context $c$ (multi-view images, language instruction, ego state). The model is trained via the denoising objective in Eq. 1 using autonomous driving planning datasets for supervised fine-tuning. This provides the inherent capability for bidirectional inpainting, which serves as the foundation of our method. It enables the model to perform holistic parallel refinement and elegantly repair trajectories around externally guided safety edits during the reflective inference process.

### 4.2 Reflective Inference

With the discrete diffusion-based VLA model as our foundation, we introduce a reflective inference framework to bridge the gap between imitation learning and safety-critical deployment. This framework operates in two stages: goal-conditioned trajectory generation and safety-guided regeneration. The entire process is guided by a set of specialized scoring functions.

#### Scoring Function Definitions.

To systematically evaluate trajectories, our framework incorporates three distinct scoring functions. The detailed composition of these functions, which are designed based on established autonomous driving evaluation principles, is provided in Appendix C.

- Global Scorer ($S_{\text{global}}(\tau)$): This scorer evaluates the overall quality of a complete trajectory, considering both safety and coherence, and returns a value of zero if any critical rule is violated.
- Safety Scorer ($S_{\text{safe}}(\tau)$): This scorer acts as a safety oracle to identify specific points of failure.
- Local Scorer ($S_{\text{local}}(a_{x},a_{y})$): This scorer evaluates each candidate token pair $(a_{x},a_{y})$ using a comprehensive function that assesses its impact on the trajectory’s safety and coherence.

#### Goal-Conditioned Generation.

To ensure our planner can reason about high-level, global intents that go beyond simple local adjustments, the process begins with generating a diverse set of trajectory proposals. This procedure is essential for multi-modal driving behavior modeling and serves as a necessary step for subsequent regeneration. Since the local search in our safety-aware regeneration stage is intentionally constrained for efficiency, it cannot accommodate large-scale changes, such as taking a different turn at an intersection, which require broader exploration. We first use the model to produce a probability distribution for the terminal waypoint tokens, $p_{\theta}(\mathbf{y}_{N}\mid c,s)$, where $\mathbf{y}_{N}=(\mathbf{y}_{N,x},\mathbf{y}_{N,y})$. From this distribution, we sample a set of high-probability goal candidates. We then apply Non-Maximum Suppression (NMS) [^27] to obtain a spatially diverse set of $K$ candidate goals, $\mathcal{G}=\{G_{1},\dots,G_{K}\}$:

$$
\mathcal{G}=\text{NMS}\left(\operatorname{TopK}_{K^{\prime}}\big(p_{\theta}(\mathbf{y}_{N}\mid c,s)\big),\,d_{\text{NMS}},\,K\right)
$$

where $\operatorname{TopK}_{K^{\prime}}(\cdot)$ is an operator that selects the $K^{\prime}$ most probable goal candidates from the model’s output distribution. The $\text{NMS}(\cdot)$ function then filters this set using a distance threshold $d_{\text{NMS}}$ to produce the final, spatially diverse set $\mathcal{G}$ of size $K$. For practical deployment, a dedicated goal generation model could be used to improve the accuracy and quality of goal points. However, for simplicity, we employ the same model for both goal generation and trajectory planning. Then, for each goal $G_{k}\in\mathcal{G}$, we generate a full trajectory $\tau_{k}$ by sampling from the conditional distribution $p_{\theta}(\mathbf{y}_{1:2N-2}\mid G_{k},c,s)$ via inpainting. The resulting $K$ trajectories are evaluated using the Global Scorer $S_{\text{global}}(\cdot)$, which assesses each plan based on a combination of metrics including goal progress. The top-scoring trajectory $\tau^{*}$ is then selected for further refinement.

$$
\tau^{*}=\operatorname*{arg\,max}_{\tau_{k},k=1,\dots,K}S(\tau_{k}).
$$
![[x2 3.png|Refer to caption]]

Figure 2: Safety-Guided Regeneration Pipeline.

#### Safety-Guided Regeneration.

The selected trajectory $\tau^{*}$, while coherent, may still violate physical constraints. We address this with an iterative, gradient-free refinement loop that forms a dialogue between the generative model and an external safety oracle, as shown in Figure 2.

- Trajectory Evaluation. The process begins when the Safety Scorer $S_{\text{safe}}(\cdot)$ evaluates the de-quantized trajectory and identifies the specific waypoints that are unsafe. The oracle assigns a safety score to each original waypoint based on the worst violation (e.g., drivable area infraction) within a local time window. This allows it to precisely pinpoint unsafe waypoints.
- Safety Anchors Search. For the earliest waypoint that violates a safety threshold, we perform a highly efficient local search within a small Manhattan neighborhood $\mathcal{N}_{\delta}$ of the original tokens to identify an improved token pair, rather than resorting to complex continuous optimization. The corrected token pair that maximizes the local safety score is then designated as a safety anchor.
- Trajectory Inpainting. We then leverage the diffusion model’s powerful inpainting capability to regenerate the surrounding trajectory segments conditioned on safety anchors. This single-pass regeneration allows the model to naturally re-establish global coherence around the safety-driven edit. This cycle of identifying violations, performing discrete corrections, and re-inpainting continues until the plan is fully safe or a computational budget is met.

This refinement process operates as an iterative loop. In each iteration, The top-scoring trajectory $\tau^{*}$ is evaluated by the Safety Scorer at each waypoint $t$. The algorithm proceeds sequentially through the waypoints to find the first index $t^{*}$ for which the score $S_{\text{safe}}(\tau^{*})$ falls below a predefined safety threshold. If no such waypoint exists, the trajectory is deemed safe and the process terminates. If a violation is found at index $t^{*}$, the Local Scorer is then employed to find an improved token pair within a local neighborhood $\mathcal{N}_{\delta}$ by solving:

$$
(\mathbf{y}^{\prime}_{t^{*},x},\mathbf{y}^{\prime}_{t^{*},y})=\operatorname*{arg\,max}_{(a_{x},a_{y})\in\mathcal{N}_{\delta}(\mathbf{y}_{t^{*},x},\mathbf{y}_{t^{*},y})}S_{\text{local}}(a_{x},a_{y}).
$$

The original token at $t^{*}$ is replaced by this new, optimized pair, which serves as a fixed safety anchor for the subsequent inpainting step. The refinement cycle then continues with this updated trajectory. In practice, the reflective inference process is designed for real-time performance. The local search for corrective tokens is efficient, as it operates over a small, discrete neighborhood (e.g., a Manhattan distance $\delta\leq 10$) rather than requiring expensive gradient-based optimization. In practice, we find that most safety violations are resolved within 1–3 iterations of reflection, resulting in a manageable inference overhead.

## 5 Experiments

### 5.1 Benchmark and baselines

Evaluation Setups. In our implementation, the VLA model backbone is initialized from a publicly available pre-trained Vision-Language Model (LLaDA-V [^36]) and utilizes classifier-free guidance for trajectory generation. Input images are obtained from the front, front-left, and front-right cameras. The language instruction provides a high-level navigational command, such as “turn left” or “go straight,” along with textual descriptions of the ego vehicle’s status. We evaluate our model on the large-scale real-world autonomous driving benchmark NAVSIM [^9] for closed-loop performance assessment. Following the official protocol, performance is reported with the PDMS score (higher is better), aggregated from five metrics: *NC* (no-collision rate), *DAC* (drivable area compliance), *TTC* (time-to-collision safety), *Comfort* (bounded acceleration/jerk) and *EP* (ego progress). We run all the methods under the official closed‑loop simulator and report averages on the public test split. Our planner uses camera‑only inputs unless otherwise stated; we also include Camera+LiDAR baselines to provide a more comprehensive comparison.

#### Baselines.

We compare ReflectDrive to other autonomous driving systems. For example, vanilla E2E planners that purely use sensor information as input and output trajectories, such as UniAD [^12], Para-Drive [^34], Transfuser [^6]. As well as augmented E2E planners that incorporate clustering results as auxiliary information like Hydra-MDP [^21], DiffusionDrive [^22], and GoalFlow [^35], the PDMS scores will be higher than vanilla E2E planners due to additional information. We also include recent AutoVLA [^40] model that unifies reasoning and action generation within a single autoregressive generation model, the PMDS score is the highest among VLA planners. For our model family, the table lists: *ReflectDrive (w/o R.I.)* trained with discrete masked diffusion adding classifier-free guidance at inference without reflective inference; *ReflectDrive* adding goal-conditioned generation and safety-guided regeneration, where the safety-guided regeneration relies on the reward model where surrounding obstacles are moving at constant speeds; *ReflectDrive <sup>†</sup>* adding goal-conditioned generation and safety-guided regeneration, where the safety-guided regeneration relies on the reward model where surrounding obstacles are ground-truth agents.

![[goodcase.png|Refer to caption]]

Table 1: NAVSIM Closed-Loop Results. Methods are grouped by their core architectural paradigm. The † symbol denotes our method using a privileged ground-truth oracle for reflection, serving as an analytical upper bound. Best result per column is in bold (higher is better).

[^1]: Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. *Advances in neural information processing systems*, 34:17981–17993, 2021.

[^2]: Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. *arXiv preprint arXiv:2502.13923*, 2025.

[^3]: Mayank Bansal, Alex Krizhevsky, and Abhijit Ogale. Chauffeurnet: Learning to drive by imitating the best and synthesizing the worst. *arXiv preprint arXiv:1812.03079*, 2018.

[^4]: Zhong Cao, Kun Jiang, Weitao Zhou, Shaobing Xu, Huei Peng, and Diange Yang. Continuous improvement of self-driving cars using dynamic confidence-aware reinforcement learning. *Nature Machine Intelligence*, 5(2):145–158, 2023.

[^5]: Li Chen, Penghao Wu, Kashyap Chitta, Bernhard Jaeger, Andreas Geiger, and Hongyang Li. End-to-end autonomous driving: Challenges and frontiers. *arXiv preprint arXiv:2306.16927*, 2023.

[^6]: Kashyap Chitta, Aditya Prakash, Bernhard Jaeger, Zehao Yu, Katrin Renz, and Andreas Geiger. Transfuser: Imitation with transformer-based sensor fusion for autonomous driving. *Pattern Analysis and Machine Intelligence (PAMI)*, 2023.

[^7]: Marco Cusumano-Towner, David Hafner, Alex Hertzberg, Brody Huval, Aleksei Petrenko, Eugene Vinitsky, Erik Wijmans, Taylor Killian, Stuart Bowers, Ozan Sener, et al. Robust autonomy emerges from self-play. *arXiv preprint arXiv:2502.03349*, 2025.

[^8]: Daniel Dauner, Marcel Hallgarten, Andreas Geiger, and Kashyap Chitta. Parting with misconceptions about learning-based vehicle motion planning. In *Conference on Robot Learning*, pp. 1268–1281. PMLR, 2023.

[^9]: Daniel Dauner, Marcel Hallgarten, Tianyu Li, Xinshuo Weng, Zhiyu Huang, Zetong Yang, Hongyang Li, Igor Gilitschenski, Boris Ivanovic, Marco Pavone, Andreas Geiger, and Kashyap Chitta. Navsim: Data-driven non-reactive autonomous vehicle simulation and benchmarking. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2024.

[^10]: Haoyang Fan, Fan Zhu, Changchun Liu, Liangliang Zhang, Li Zhuang, Dong Li, Weicheng Zhu, Jiangtao Hu, Hongye Li, and Qi Kong. Baidu apollo em motion planner, 2018.

[^11]: Yanchen Guan, Haicheng Liao, Zhenning Li, Jia Hu, Runze Yuan, Guohui Zhang, and Chengzhong Xu. World models for autonomous driving: An initial survey. *IEEE Transactions on Intelligent Vehicles*, 2024.

[^12]: Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai Wang, et al. Planning-oriented autonomous driving. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 17853–17862, 2023.

[^13]: Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. *arXiv preprint arXiv:2410.21276*, 2024.

[^14]: Jyh-Jing Hwang, Runsheng Xu, Hubert Lin, Wei-Chih Hung, Jingwei Ji, Kristy Choi, Di Huang, Tong He, Paul Covington, Benjamin Sapp, et al. Emma: End-to-end multimodal model for autonomous driving. *arXiv preprint arXiv:2410.23262*, 2024.

[^15]: Bernhard Jaeger, Daniel Dauner, Jens Beißwenger, Simon Gerstenecker, Kashyap Chitta, and Andreas Geiger. Carl: Learning scalable planning policies with simple rewards. *arXiv preprint arXiv:2504.17838*, 2025.

[^16]: Chiyu Jiang, Andre Cornman, Cheolho Park, Benjamin Sapp, Yin Zhou, Dragomir Anguelov, et al. Motiondiffuser: Controllable multi-agent motion prediction using diffusion. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 9644–9653, 2023.

[^17]: Leslie Pack Kaelbling, Michael L Littman, and Andrew W Moore. Reinforcement learning: A survey. *Journal of artificial intelligence research*, 4:237–285, 1996.

[^18]: Alex Kendall, Jeffrey Hawke, David Janz, Przemyslaw Mazur, Daniele Reda, John-Mark Allen, Vinh-Dieu Lam, Alex Bewley, and Amar Shah. Learning to drive in a day. In *2019 international conference on robotics and automation (ICRA)*, pp. 8248–8254. IEEE, 2019.

[^19]: Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. *arXiv preprint arXiv:2406.09246*, 2024.

[^20]: B Ravi Kiran, Ibrahim Sobh, Victor Talpaert, Patrick Mannion, Ahmad A Al Sallab, Senthil Yogamani, and Patrick Pérez. Deep reinforcement learning for autonomous driving: A survey. *IEEE transactions on intelligent transportation systems*, 23(6):4909–4926, 2021.

[^21]: Zhenxin Li, Kailin Li, Shihao Wang, Shiyi Lan, Zhiding Yu, Yishen Ji, Zhiqi Li, Ziyue Zhu, Jan Kautz, Zuxuan Wu, et al. Hydra-mdp: End-to-end multimodal planning with multi-target hydra-distillation. *arXiv preprint arXiv:2406.06978*, 2024.

[^22]: Bencheng Liao, Shaoyu Chen, Haoran Yin, Bo Jiang, Cheng Wang, Sixu Yan, Xinbang Zhang, Xiangyu Li, Ying Zhang, Qian Zhang, and Xinggang Wang. Diffusiondrive: Truncated diffusion model for end-to-end autonomous driving. *arXiv preprint arXiv:2411.15139*, 2024.

[^23]: Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution. *arXiv preprint arXiv:2310.16834*, 2023.

[^24]: Yiren Lu, Justin Fu, George Tucker, Xinlei Pan, Eli Bronstein, Rebecca Roelofs, Benjamin Sapp, Brandyn White, Aleksandra Faust, Shimon Whiteson, et al. Imitation is not enough: Robustifying imitation with reinforcement learning for challenging driving scenarios. In *2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, pp. 7553–7560. IEEE, 2023.

[^25]: Chenlin Meng, Kristy Choi, Jiaming Song, and Stefano Ermon. Concrete score matching: Generalized score matching for discrete data. *Advances in Neural Information Processing Systems*, 35:34532–34545, 2022.

[^26]: Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. *arXiv preprint arXiv:2502.09992*, 2025.

[^27]: Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. *Advances in neural information processing systems*, 28, 2015.

[^28]: Katrin Renz, Long Chen, Elahe Arani, and Oleg Sinavski. Simlingo: Vision-only closed-loop autonomous driving with language-action alignment. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pp. 11993–12003, 2025.

[^29]: Shai Shalev-Shwartz, Shaked Shammah, and Amnon Shashua. Safe, multi-agent, reinforcement learning for autonomous driving, 2016.

[^30]: Ardi Tampuu, Tambet Matiisen, Maksym Semikin, Dmytro Fishman, and Naveed Muhammad. A survey of end-to-end driving: Architectures and training methods. *IEEE Transactions on Neural Networks and Learning Systems*, 33(4):1364–1384, 2020.

[^31]: Xiaoyu Tian, Junru Gu, Bailin Li, Yicheng Liu, Yang Wang, Zhiyong Zhao, Kun Zhan, Peng Jia, Xianpeng Lang, and Hang Zhao. Drivevlm: The convergence of autonomous driving and large vision-language models. *arXiv preprint arXiv:2402.12289*, 2024.

[^32]: Martin Treiber, Ansgar Hennecke, and Dirk Helbing. Congested traffic states in empirical observations and microscopic simulations. *Physical review E*, 62(2):1805, 2000.

[^33]: Shihao Wang, Zhiding Yu, Xiaohui Jiang, Shiyi Lan, Min Shi, Nadine Chang, Jan Kautz, Ying Li, and Jose M Alvarez. Omnidrive: A holistic vision-language dataset for autonomous driving with counterfactual reasoning. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pp. 22442–22452, 2025.

[^34]: Xinshuo Weng, Boris Ivanovic, Yan Wang, Yue Wang, and Marco Pavone. Para-drive: Parallelized architecture for real-time autonomous driving. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 15449–15458, 2024.

[^35]: Zebin Xing, Xingyu Zhang, Yang Hu, Bo Jiang, Tong He, Qian Zhang, Xiaoxiao Long, and Wei Yin. Goalflow: Goal-driven flow matching for multimodal trajectories generation in end-to-end autonomous driving. *arXiv preprint arXiv:2503.05689*, 2025.

[^36]: Zebin You, Shen Nie, Xiaolu Zhang, Jun Hu, Jun Zhou, Zhiwu Lu, Ji-Rong Wen, and Chongxuan Li. Llada-v: Large language diffusion models with visual instruction tuning. *arXiv preprint arXiv:2505.16933*, 2025.

[^37]: Yinan Zheng, Jianxiong Li, Dongjie Yu, Yujie Yang, Shengbo Eben Li, Xianyuan Zhan, and Jingjing Liu. Safe offline reinforcement learning with feasibility-guided diffusion model. In *The Twelfth International Conference on Learning Representations*, 2024. URL [https://openreview.net/forum?id=j5JvZCaDM0](https://openreview.net/forum?id=j5JvZCaDM0).

[^38]: Yinan Zheng, Ruiming Liang, Kexin ZHENG, Jinliang Zheng, Liyuan Mao, Jianxiong Li, Weihao Gu, Rui Ai, Shengbo Eben Li, Xianyuan Zhan, and Jingjing Liu. Diffusion-based planning for autonomous driving with flexible guidance. In *The Thirteenth International Conference on Learning Representations*, 2025. URL [https://openreview.net/forum?id=wM2sfVgMDH](https://openreview.net/forum?id=wM2sfVgMDH).

[^39]: Ziyuan Zhong, Davis Rempe, Danfei Xu, Yuxiao Chen, Sushant Veer, Tong Che, Baishakhi Ray, and Marco Pavone. Guided conditional diffusion for controllable traffic simulation. In *2023 IEEE International Conference on Robotics and Automation (ICRA)*, pp. 3560–3566. IEEE, 2023.

[^40]: Zewei Zhou, Tianhui Cai, Seth Z Zhao, Yun Zhang, Zhiyu Huang, Bolei Zhou, and Jiaqi Ma. Autovla: A vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. *arXiv preprint arXiv:2506.13757*, 2025.
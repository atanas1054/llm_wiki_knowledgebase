---
title: "DiffusionDriveV2: Reinforcement Learning-Constrained Truncated Diffusion Modeling in End-to-End Autonomous Driving"
source: "https://arxiv.org/html/2512.07745"
author:
published:
created: 2026-04-16
description:
tags:
  - "clippings"
---
Jialv Zou <sup>1,⋄</sup>  Shaoyu Chen <sup>3</sup>  Bencheng Liao <sup>2,1</sup>  Zhiyu Zheng <sup>4,⋄</sup>  Yuehao Song <sup>1,⋄</sup>  
Lefei Zhang <sup>4</sup>  Qian Zhang <sup>3</sup>  Wenyu Liu <sup>1</sup>  Xinggang Wang ${}^{1,\textrm{\Letter}}$  
<sup>1</sup> School of Electronic Information and Communications, Huazhong University of Science & Technology  
<sup>2</sup> Institute of Artificial Intelligence, Huazhong University of Science & Technology  
<sup>3</sup> Horizon Robotics     <sup>4</sup> School of Computer Science, Wuhan University

###### Abstract

Diffusion models for trajectory planning in end-to-end autonomous driving often suffer from mode collapse, tending to generate conservative and homogeneous behaviors. While DiffusionDrive employs predefined anchors representing different driving intentions to partition the action space and generate diverse trajectories, its reliance on imitation learning lacks sufficient constraints, resulting in a dilemma between diversity and consistent high quality. In this work, we propose DiffusionDriveV2, which leverages reinforcement learning to both constrain low-quality modes and explore for superior trajectories. This significantly enhances the overall output quality while preserving the inherent multimodality of its core Gaussian Mixture Model. First, we use scale-adaptive multiplicative noise, ideal for trajectory planning, to promote broad exploration. Second, we employ intra-anchor GRPO to manage advantage estimation among samples generated from a single anchor, and inter-anchor truncated GRPO to incorporate a global perspective across different anchors, preventing improper advantage comparisons between distinct intentions (e.g., turning vs. going straight), which can lead to further mode collapse. DiffusionDriveV2 achieves 91.2 PDMS on the NAVSIM v1 dataset and 85.5 EPDMS on the NAVSIM v2 dataset in closed-loop evaluation with an aligned ResNet-34 backbone, setting a new record. Further experiments validate that our approach resolves the dilemma between diversity and consistent high quality for truncated diffusion models, achieving the best trade-off. Code and model will be available at [https://github.com/hustvl/DiffusionDriveV2](https://github.com/hustvl/DiffusionDriveV2)

<sup>†</sup>

## 1 Introduction

In recent years, with the growing maturity of traditional tasks such as 3D object detection \[huang2021bevdet, li2024bevformer, yang2023bevformer\], multi-object tracking \[zhang2022bytetrack, zhang2021fairmot\], pre-training \[yang2024unipad, zou2025mim4d, zhang2025visionpad\], online mapping \[liao2022maptr, liao2025maptrv2\] and motion prediction \[chai2019multipath, varadarajan2022multipath++\], the development wave in autonomous driving systems has shifted towards end-to-end autonomous driving (E2E-AD), which directly learns a driving policy from raw sensor inputs.

Early approaches in this field have limitations in terms of modeling. Traditional end-to-end unimodal planners \[jiang2023vad, hu2023planning, chitta2022transfuser\] regress a single trajectory and fail to propose alternatives for complex driving scenarios with high uncertainty. Selection-based methods \[chen2024vadv2, li2024hydra, li2025hydra\] use a large, static vocabulary of candidate trajectories, but this discretization offers limited flexibility.

Recently, several approaches have employed diffusion models for trajectory generation \[chi2023diffusion, liao2025diffusiondrive, xing2025goalflow, li2025recogdrive, zheng2025resad\], which can dynamically produce a small set of candidate trajectories conditioned on the surrounding scene. However, directly applying vanilla diffusion models to multi-modal trajectory generation faces the challenge of mode collapse, converging to a single high-probability mode and thus failing to capture the diversity of potential futures, as shown in Fig. 1(a). To address this problem, DiffusionDrive \[liao2025diffusiondrive\] proposes constructing the prior distribution of the initial noise using a Gaussian Mixture Model (GMM) defined by multiple predefined trajectory anchors. This structured prior partitions the entire generation space into multiple subspaces, each corresponding to a specific driving intention (e.g., one mode for lane changing, another for driving straight), thereby effectively promoting the generation of diverse behavioral modes.

![[x1 20.png|Refer to caption]]

Figure 1: Comparison of various models. (a) Vanilla Diffusion models are prone to mode collapse, collapsing diverse possibilities into a single trajectory. (b) DiffusionDrive generates trajectories with excellent multimodality, yet constrained by imitation learning, it also produces numerous colliding ones (circled in red ) as most negative modes lack supervision during training, posing a major threat to the system’s overall quality. (c) DiffusionDriveV2 leverages reinforcement learning to apply constraints to multi-modal trajectories, guiding the model to generate both diverse and consistent high-quality trajectories.

However, DiffusionDrive faces a fundamental dilemma between the diversity and consistent high quality of its generated trajectories, stemming from its reliance on the imitation learning (IL) paradigm. While the GMM prior enforces diverse mode generation, its training objective, designed to maximize the likelihood of expert trajectories across the entire mixture model, is simplified in practice to optimizing only the parameters of the single positive mode (i.e., the one closest to the expert trajectory). Consequently, it neglects to impose any explicit constraints on trajectories sampled from the negative modes, which constitute the vast majority of the samples. This leads to the model generating high-quality trajectories alongside a multitude of unconstrained, low-quality, and often colliding ones, failing to guarantee consistently high quality, as shown in Fig. 1(b).

This hazardous mix forces reliance on a downstream selector, which is often less robust than the generator due to much fewer parameters. This over-reliance poses a significant risk, as this component is prone to failure when filtering many low-quality trajectories, particularly in out-of-distribution scenarios.

Reinforcement Learning (RL) provides a powerful solution to this dilemma. In contrast to IL, which is constrained to a single positive mode, RL operates on an exploration-constraint paradigm. On one hand, it raises the model’s lower bound by applying goal-alignment constraints to all modes, rewarding desired behaviors while simultaneously penalizing unsafe actions from negative modes. On the other hand, it raises the model’s upper bound by pushing the model to explore a broader action space, seeking policies that may exceed the expert’s in quality and efficiency.

Spurred by the success of DeepSeek-R1 \[guo2025deepseek\], several works \[li2025recogdrive, song2025breaking\] have introduced GRPO to E2E-AD. However, their application has been limited to vanilla diffusion models. Unlike these approaches, in anchored truncated diffusion models, each predefined trajectory anchor represents a distinct driving intention. Naively performing advantage estimation between trajectories corresponding to different driving intentions would exacerbate mode collapse. For instance, trajectories for turning left and going straight should coexist rather than be compared for superiority. This insight motivates us to propose Intra-Anchor GRPO to prevent mode collapse by performing group advantage estimation exclusively within each anchor, thereby blocking comparisons between different intents, and introduce Inter-Anchor Truncated GRPO to provide a global perspective and stabilize training.

With these innovations, we propose a novel framework DiffusionDriveV2, which leverages RL to address the dilemma between diversity and consistent high quality of DiffusionDrive that stem from its reliance on IL. We benchmark our method on the planning-oriented NAVSIM v1 \[dauner2024navsim\] and NAVSIM v2 \[cao2025pseudo\] datasets using closed-loop evaluations. DiffusionDriveV2 sets a new state-of-the-art on both benchmarks, achieving 91.2 PDMS on NAVSIM v1 and 85.5 EPDMS on NAVSIM v2 with ResNet-34 backbone, which represents a substantial improvement over previous methods. Furthermore, compared to other diffusion-based generative models, DiffusionDriveV2 achieves the best trade-off between trajectory diversity and consistent high quality.

Our contributions can be summarized as follows:

- We propose DiffusionDriveV2, a novel approach that introduces RL to address the dilemma between diversity and consistent high quality of DiffusionDrive, which is caused by the incomplete multi-modal supervision in IL. To the best of our knowledge, DiffusionDriveV2 is the first work to directly confront this dilemma and propose a solution.
- We introduce Intra-Anchor GRPO and Inter-Anchor Truncated GRPO to solve the issue of inability to perform group advantage estimation across different modes within the Gaussian Mixture Model framework when directly adapting vanilla GRPO to DiffusionDrive. DiffusionDriveV2 is the first work to successfully migrate GRPO to a truncated diffusion model.
- We leverage scale-adaptive multiplicative noise as the exploration noise instead of additive noise, which helps preserve the smoothness and coherence of the exploratory trajectories.
- Extensive evaluations on the NAVSIM v1 and NAVSIM v2 benchmarks demonstrate that DiffusionDriveV2 significantly improves overall output quality while preserving the ability of the underlying Gaussian Mixture Model to generate multi-modal trajectories, leading to the state-of-the-art performance.

## 2 Related Work

![[x2 18.png|Refer to caption]]

Figure 2: Overall architecture of DiffusionDriveV2. Trajectories of different colors represent distinct anchored intents. Solid lines indicate high-quality trajectories, while dashed lines indicate low-quality ones. The truncated diffusion decoder, limited by incomplete supervision in IL, produces low-quality trajectories ( overtake, right turn ) alongside high-quality ones ( go straight ). To address this, we first apply multiplicative Gaussian noise to push the model to explore the nearby action space. We then propose Anchored Truncated GRPO, which performs intra-group advantage estimation to optimize the model, steering it away from collisions and towards high-quality trajectories. The resulting refined trajectories for and become collision-free, while the trajectories become more optimal rather than overly conservative. Finally, a mode selector chooses the most goal-aligned trajectory from the refined trajectories.

#### End-to-End Autonomous Driving.

Traditional autonomous driving (AD) systems rely on a highly modular pipeline, which suffers from limitations such as error propagation and information loss between components. UniAD \[hu2023planning\] represents a pioneering work that addresses these issues by integrating multiple perception tasks into a single, fully differentiable framework, showcasing the potential of the end-to-end approach. VAD \[jiang2023vad\] further improves the system’s efficiency by employing a vectorized scene representation. Subsequently, a series of methods \[chen2024vadv2, li2024hydra, li2025hydra, yao2025drivesuprim\], represented by VADv2 \[chen2024vadv2\] and Hydra-MDP \[li2024hydra\], have shifted to a multi-modal planning framework by performing rule-based scoring and sampling over a fixed vocabulary of anchor trajectories. More recently, Diffusion Policy \[chi2023diffusion\] has emerged as a powerful approach for E2E-AD, effectively modeling intricate, multi-modal distributions in high-dimensional action spaces. Diffusion models learn the data distribution through an iterative denoising process and have achieved remarkable performance in image generation tasks \[ho2020denoising, song2020denoising, dhariwal2021diffusion, rombach2022high\]. DiffusionDrive \[liao2025diffusiondrive\] highlights the challenge of mode collapse for diffusion models in E2E-AD, introduces an anchor-based truncated denoising strategy to counteract it, and drastically improves efficiency by reducing the required denoising steps to just two. Overall, diffusion-based generative models show significant potential for E2E-AD, offering both high quality and efficient generation. However, these methods are all fundamentally constrained by the imitation learning paradigm, which means they typically either face the challenge of mode collapse or run the risk of generating numerous low-quality trajectories.

#### Reinforcement Learning for Autonomous Driving.

Reinforcement Learning drives an agent to explore by interacting with an environment and maximizing the cumulative return to learn the optimal policy. RL has proven its effectiveness in various domains, from training Large Language Models (LLMs) such as DeepSeek-R1 \[guo2025deepseek\] and OpenAI O1 \[openai2024o1\], to mastering complex games like AlphaGo \[silver2016mastering\] and AlphaGo Zero \[silver2017mastering\]. Recently, the application of RL in autonomous driving has been increasingly explored. A series of works \[chen2021learning, hu2024solving, lu2023imitation\] have investigated the use of RL in non-photorealistic simulators, such as CARLA \[dosovitskiy2017carla\]. RAD \[gao2025rad\] trains an E2E-AD agent in a realistic 3DGS \[kerbl20233d\] environment to bridge the sim-to-real gap. Inspired by the advancements of Deepseek-R1, the application of GRPO has extended to E2E-AD, with AlphaDrive \[jiang2025alphadrive\] pioneering its integration into planning reasoning. While subsequent research \[li2025recogdrive, li2025finetuning\] have investigated applying GRPO to diffusion-based generative trajectory models, these efforts have been confined to direct implementation on vanilla diffusion, thus suffering from mode collapse.

## 3 Preliminary

#### End-to-End Autonomous Driving.

The E2E-AD system learns an expert driving policy via imitation learning, mapping raw sensor data to future ego-vehicle trajectory predictions. The trajectory is represented by a sequence of future waypoints, denoted as $\tau=\left\{\left(x_{n},y_{n}\right)\right\}_{n=1}^{N_{f}},$ where $\left(x_{n},y_{n}\right)$ is the location of each waypoint at time $n$ and $N_{f}$ represents the planning horizon.

#### Truncated Diffusion Model.

Diffusion policy models \[chi2025diffusion, janner2022planning\] learn a reverse Markovian noise process to generate trajectories by iteratively refining a random Gaussian noise. However, experiments show that vanilla diffusion models often suffer from mode collapse, failing to generate diverse driving behaviors. This makes it difficult for them to handle complex driving scenarios and provide a rich set of alternative trajectories, such as car-following versus overtaking, or going straight versus turning left at an intersection.

To overcome the mode collapse problem of vanilla diffusion models, DiffusionDrive \[liao2025diffusiondrive\] proposes modeling the trajectory distribution as a Gaussian Mixture Model Distribution. It achieves this by representing a discrete set of driving intents as a set of $N_{anchor}$ anchor trajectories $\left\{\mathbf{a}^{k}\right\}_{k=1}^{N_{\text{anchor }}}$ clustered using K-Means from the expert driving behaviors. Each anchor corresponds to a specific region of the trajectory space, thereby representing a particular driving intent, such as overtaking, turning left, or keeping straight. The trajectory distribution for anchor $\mathbf{a}^{k}$ can be expressed as:

$$
p\left(\tau^{k}\mid\mathbf{a}^{k},z\right)=\mathcal{N}\left(\tau^{k}\mid\mathbf{a}^{k}+\mu^{k}(z),\Sigma^{k}(z)\right).
$$

Notably, unlike vanilla diffusion models that directly predict trajectories from random noise, DiffusionDrive is trained to predict the offset between a trajectory and its corresponding anchor $\mathbf{a}^{k}$, with $\mu^{k}(z)$ representing a scene-specific offset from the anchor state $\mathbf{a}^{k}$ conditioned on scene context $z$. The entire trajectory distribution can be represented as:

$$
p(\mathbf{\tau}\mid z)=\sum_{k=1}^{N_{anchor}}s\left(\mathbf{a}^{k}\mid z\right)p\left(\tau^{k}\mid\mathbf{a}^{k},z\right).
$$

This results in a Gaussian Mixture Model (GMM) distribution, where $s\left(\mathbf{a}^{k}\mid z\right)$ is the mixture weight denoting the probability of choosing the driving intent associated with anchor $\mathbf{a}^{k}$, given the scene context $z$.

DiffusionDrive utilizes a truncated diffusion process, which shortens the standard noise schedule to diffuse each anchor trajectory into a corresponding anchored Gaussian distribution:

$$
\tau_{t}^{k}=\sqrt{\bar{\alpha}_{t}}\mathbf{a}^{k}+\sqrt{1-\bar{\alpha}_{t}}\boldsymbol{\epsilon},\quad\boldsymbol{\epsilon}\sim\mathcal{N}(0,\mathbf{I}),
$$

where $t\in[1,T_{\text{trunc}}]$ and $T_{\text{trunc}}\ll T$ is the number of truncated diffusion steps. During training, DiffusionDrive takes noisy trajectories $\left\{\tau^{k}_{t}\right\}_{k=1}^{N_{\text{anchor}}}$ as input and predicts denoised trajectories $\left\{\hat{\tau}^{k}\right\}_{k=1}^{N_{\text{anchor}}}$ and the probability scores $\hat{s}^{k}$, where $s^{k}$ is a shorthand for $s\left(\mathbf{a}^{k}\mid z\right)$ in Eq. (2).

However, DiffusionDrive is still constrained by the limitations of IL. Although its anchor-based design mitigates mode collapse and provides diverse trajectory options, the training process is fundamentally limited by the fact that only a single GT trajectory is available per scene. Consequently, the model must still select one anchor as the positive mode for optimization during training. The anchor closest to the GT trajectory $\tau_{gt}$ is assigned as the positive sample ($y^{k}$ = 1) and the others as negative samples ($y^{k}$ = 0). The training objective is:

$$
\mathcal{L}=\sum_{k=1}^{N_{\text{anchor}}}[y^{k}\mathcal{L}_{\text{rec}}(\hat{\tau}^{k},\tau_{\text{gt}})+\mathcal{L}_{\text{BCE}}(\hat{s}^{k},y^{k})],
$$

Due to the constraints of IL, only a single mode receives supervision in each scene. As a result, while the model might generate diverse trajectories, it also produces numerous low-quality ones that could lead to collisions, posing a significant hazard to the system.

## 4 Method

### 4.1 Truncated Diffusion Generator

The overall architecture of our proposed method, DiffusionDriveV2, is illustrated in Fig. 2.

To generate multi-modal trajectories, we directly employ DiffusionDrive as our trajectory generator, leveraging its pre-trained weights for IL on GT trajectories. This provides a cold start, equipping our model with an initial capability for multi-modal trajectory generation. Conditioned on features extracted by the perception network, the truncated diffusion decoder takes the noisy trajectories $\left\{\tau^{k}_{t}\right\}_{k=1}^{N_{\text{anchor}}}$ sampled from the anchored Gaussian distribution as input and iteratively refines them over $N_{infer}$ steps to produce the final clean trajectories.

### 4.2 Reinforcement Learning for Diffusion Generator

Although DiffusionDrive demonstrates strong capabilities in generating multi-modal trajectories, it inherits a critical limitation from imitation learning, namely a lack of supervision on negative modes. This often results in the generation of low-quality trajectories, posing a significant threat to the system. To address this, we introduce trajectory-level reinforcement learning objectives to apply constraints across all modes and push the model to explore superior driving policies. Inspired by DPPO \[ren2024diffusion\], we treat the denoising process as a Markov Decision Process (MDP). Each conditional denoising step in the diffusion chain initiated from anchor $\mathbf{a}^{k}$ is a Gaussian policy:

$$
\displaystyle\pi_{\theta}\left(\tau_{t-1}^{k}\mid\tau_{t}^{k},z,\mathbf{a}^{k}\right)=
$$
 
$$
\displaystyle\mathcal{N}\left(\tau_{t-1}^{k};\mu_{\theta}\left(\tau_{t}^{k},t,z,\mathbf{a}^{k}\right),\right.
$$
$$
\displaystyle\qquad\left.\eta\left(1-\alpha_{t}\right)I\right),
$$

$\mu_{\theta}\left(\tau_{t},t,z,\mathbf{a}^{k}\right)$ is the model-predicted mean and $\alpha_{t}$ is determined by a predefined noise schedule.

This equation is a Gaussian likelihood, which can be evaluated analytically and is amenable to the policy gradient updates with REINFORCE \[williams1992simple\]:

$$
\nabla_{\theta}{\mathcal{J}}\left(\pi_{\theta}^{k}\right)=\mathbb{E}_{\pi_{\theta}^{k}}\left[\sum_{t=1}^{T_{trunc}}\nabla_{\theta}\log\pi_{\theta}^{k}\left(\tau_{t-1}^{k}\mid\tau_{t}^{k}\right)A_{t}^{k}\right],
$$

where $A_{t}^{k}$ denotes the advantage function.

![[x3 18.png|Refer to caption]]

Figure 3: Comparison with Different Noise Strategies for Exploration. The green solid line denotes the original trajectory, while the blue and red dashed lines represent the trajectories after applying exploration noise.

### 4.3 Scale-Adaptive Multiplicative Exploration Noise

DiffusionDrive applies the DDIM \[song2020denoising\] update rule to drastically reduce the number of denoising steps. This update rule is typically used as a deterministic sampler by setting $\eta=0$. To enable broader exploration and prevent the issue of calculating a likelihood over a Dirac distribution, we introduce exploration noise by setting $\eta=1$ during training (equivalent to applying DDPM \[ho2020denoising\]), while keeping $\eta=0$ during validation for deterministic inference.

However, due to the inherent scale inconsistency between the proximal and distal segments of a trajectory, simply applying additive Gaussian noise at each point disrupts the trajectory’s structural integrity and degrades the quality of exploration. As shown in Fig. 3(a), this process, where additive Gaussian noise $\epsilon_{add}=\left\{\left(\epsilon_{x,n},\epsilon_{y,n}\right)\right\}_{n=1}^{N_{f}}$ is applied to a normalized trajectory $\tau=\left\{\left(x_{n},y_{n}\right)\right\}_{n=1}^{N_{f}}$ typically produces a jagged exploratory path that resembles a broken line, thereby losing its original smoothness. To preserve trajectory coherence, we propose a method that adds only two multiplicative Gaussian noises, one longitudinal and one lateral. It can be expressed as $\tau^{\prime}=(1+\epsilon_{mul})\tau$, where $\epsilon_{mul}=\left(\epsilon_{long},\epsilon_{lat}\right).$ This scale-adaptive multiplicative noise ensures the resulting exploratory paths remain smooth, as illustrated in Fig. 3(b).

### 4.4 Intra-Anchor GRPO for Trajectory Generation

As a reinforcement learning method designed for settings with multiple agents or modes, Group Relative Policy Optimization (GRPO) \[shao2024deepseekmath\] updates each agent’s policy in relation to a shared, group-level baseline. This approach diverges from conventional PPO \[schulman2017proximal\] by defining the policy gradient through an advantage function that is normalized by group-conditioned expectations. By optimizing for non-differentiable objectives with trajectory-level rewards, GRPO enhances standard imitation learning, guiding the diffusion model to produce diverse, goal-oriented trajectories with the potential to surpass human-driver performance.

However, naively using trajectories sampled from different anchors as the “group” for the GRPO policy update would be counterproductive. This approach contradicts our core motivation for using anchors to partition the trajectory space into distinct regions that correspond to different driving intentions, and it would even lead to mode collapse. For instance, if samples from anchors representing ’turn right’ and ’go straight’ (as shown by the red and green trajectories in Fig. 2) were optimized relative to each other, the policy would likely collapse to the single, more common ’go straight’ mode. These anchors represent fundamentally different intents and should not be directly compared within the same optimization group.

Based on this insight, we propose Intra-Anchor GRPO. For each anchor, we first generate a group of G trajectory variations by diffusing the anchor with random Gaussian noises and exploration noise. We then perform the GRPO update within this group of G trajectories, rather than across groups from different anchors. This approach constrains the policy optimization to the state space of each specific behavioral intent, guiding the model to generate safer and more goal-oriented trajectories without compromising its multi-modal capabilities. The RL loss function can be represented as:

$$
\displaystyle L_{RL}=
$$
 
$$
\displaystyle-\frac{1}{N_{anchor}}\sum_{k=1}^{N_{anchor}}\frac{1}{G}\sum_{i=1}^{G}\frac{1}{T_{trunc}}\sum_{t=1}^{T_{trunc}}
$$
 
$$
\displaystyle\quad\gamma_{t-1}\log\pi_{\theta}\left(\tau_{t-1}^{k,i}\mid\tau_{t}^{k,i}\right)A^{k,i},
$$

where $\gamma_{t-1}$ is the discount coefficient mitigating instability in early denoising steps, and $A^{k,i}$ is the advantage function, which GRPO estimates by computing the group-relative advantage, thereby avoiding the need for a value model.

$$
A^{k,i}=\frac{r^{k,i}-\operatorname{mean}\left(\left\{r^{k,1},r^{k,2},\cdots,r^{k,G}\right\}\right)}{\operatorname{std}\left(\left\{r^{k,1},r^{k,2},\cdots,r^{k,G}\right\}\right)},
$$

denotes the group relative advantage with $r^{k,i}=R(\tau_{0}^{k,i})$. A single reward estimate $R(\tau_{0}^{k,i})$, calculated from the final clean trajectory $\tau_{0}^{k,i}$, is applied to all denoising steps in the diffusion chain, with the influence of each step being scaled by denoising discount $\gamma_{t-1}\in\left(0,1\right)$.

Furthermore, analogous to how the original GRPO provides regularization by adding a KL divergence between the policy model and the reference model, we incorporate an additional imitation learning loss to prevent the model from overfitting and potentially compromising its general driving capabilities. The combined loss is

$$
L=L_{RL}+\lambda L_{IL},
$$

with weight coefficient $\lambda\in\left(0,1\right)$.

### 4.5 Inter-Anchor Truncated GRPO for Trajectory Generation

While Intra-Anchor GRPO prevents mode collapse, completely isolating the modes introduces a new problem: the advantage estimates lose global comparability. For example, a suboptimal but safe trajectory in one mode could get a negative advantage, while a dangerous, colliding trajectory in another mode might get a positive advantage if it’s the ’best’ sample within its own group. This reliance on local, intra-group comparisons can provide a misleading learning signal to the model.

To address this issue, we introduce Inter-Anchor Truncated GRPO, which is guided by a simple yet powerful principle: reward relative improvements, but only penalize absolute failures. We implement this by modifying the advantage estimates $A^{k,i}$ from Intra-Anchor GRPO. Specifically, we truncate all negative advantages to zero and assign a strong penalty of -1 to any trajectory that results in a collision:

$$
A_{trunc}^{k,i}=\begin{cases}-1&\text{if collision,}\\
\max(0,A^{k,i})&\text{otherwise.}\end{cases}
$$

This provides the model with a clear and consistent learning signal. This truncated advantage $A_{trunc}^{k,i}$ subsequently replaces $A^{k,i}$ in the RL loss calculation (Eq. (7)).

### 4.6 Mode Selector

We append a final mode selector to our model, which is responsible for choosing the optimal, goal-aligned trajectory from multi-modal predictions representing distinct intents. A higher score corresponds to a stronger alignment with the overall objective. Specifically, the trajectory coordinates serve as queries, first interacting with BEV features via deformable spatial cross-attention, and then being refined by cross-attention layers with agent and map queries. Finally, the context-rich representation is passed to a Multi-Layer Perceptron (MLP) to predict the score. Inspired by DriveSuprim \[yao2025drivesuprim\], we employ a two-stage, coarse-to-fine scorer. The process begins with a coarse scorer that initially selects the top-k candidate trajectories, which are subsequently passed to a fine-grained scorer for a more detailed selection. The score learning uses the binary cross-entropy (BCE) loss.

For the continuous metric, we introduce an additional Margin-Rank loss:

$$
\mathcal{L}_{\text{rank }}=\frac{1}{N}\sum_{i,j}\max\left(0,-\operatorname{sign}\left(s_{i}-s_{j}\right)\cdot\left(\hat{s}_{i}-\hat{s}_{j}\right)+m\right),
$$

where $s$ are the ground truth, while $\hat{s}$ are predicted scores. The margin $m$ is a positive hyperparameter. This loss guides the model to compare the relative quality of trajectories, which avoids the difficulty of directly regressing their absolute continuous values. As a result, the model’s ability to discriminate between subtle differences is further enhanced.

## 5 Experiments

### 5.1 Benchmark

#### Dataset.

We evaluate DiffusionDriveV2 on the NAVSIM v1 \[dauner2024navsim\] and NAVSIM v2 \[cao2025pseudo\] datasets. NAVSIM offers a collection of real-world, planning-centric driving scenarios built upon OpenScene \[contributors2023openscene\], which is a compact version of the extensive nuPlan \[caesar2021nuplan\] dataset. It features data from a sensor suite combining eight cameras for a 360° field of view (FOV) with a merged point cloud from five LiDARs. The dataset is split into navtrain (1,192 training scenes) and navtest (136 evaluation scenes).

| Method | Img. Backbone | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf. $\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PARA-Drive \[weng2024drive\] | ResNet-34 | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 |
| VADv2 \[chen2024vadv2\] | ResNet-34 | 97.2 | 89.1 | 91.6 | 100 | 76.0 | 80.9 |
| UniAD \[hu2023planning\] | ResNet-34 | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| Transfuser \[chitta2022transfuser\] | ResNet-34 | 97.7 | 92.8 | 92.8 | 100 | 79.2 | 84.0 |
| DRAMA \[yuan2024drama\] | ResNet-34 | 98.0 | 93.1 | 94.8 | 100 | 80.1 | 85.5 |
| Hydra-MDP\* \[li2024hydra\] | ResNet-34 | 98.3 | 96.0 | 94.6 | 100 | 78.7 | 86.5 |
| Hydra-MDP++\* \[li2025hydra\] | ResNet-34 | 97.6 | 96.0 | 93.1 | 100 | 80.4 | 86.6 |
| GoalFlow\* \[xing2025goalflow\] | ResNet-34 | 98.3 | 93.8 | 94.3 | 100 | 79.8 | 85.7 |
| ARTEMIS \[feng2025artemis\] | ResNet-34 | 98.3 | 95.1 | 94.3 | 100 | 81.4 | 87.0 |
| DiffusionDrive \[liao2025diffusiondrive\] | ResNet-34 | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| WoTE \[li2025end\] | ResNet-34 | 98.5 | 96.8 | 94.9 | 99.9 | 81.9 | 88.3 |
| DIVER \[song2025breaking\] | ResNet-34 | 98.5 | 96.5 | 94.9 | 100 | 82.6 | 88.3 |
| DriveSuprim \[yao2025drivesuprim\] | ResNet-34 | 97.8 | 97.3 | 93.6 | 100 | 86.7 | 89.9 |
| Hydra-MDP \[li2024hydra\] | V2-99 | 98.4 | 97.8 | 93.9 | 100 | 86.5 | 90.3 |
| GoalFlow \[xing2025goalflow\] | V2-99 | 98.4 | 98.3 | 94.6 | 100 | 85.0 | 90.3 |
| DiffusionDriveV2 (Ours) | ResNet-34 | 98.3 | 97.9 | 94.8 | 99.9 | 87.5 | 91.2 |

Table 1: Comparison on NAVSIM v1 navtest split with closed-loop metrics. \*For fair comparison, we use the official scores of versions with the same ResNet-34 backbone.

### 5.2 Implementation Details

To ensure a fair comparison, our model employs the same ResNet-34 \[he2016deep\] backbone as Transfuser and DiffusionDrive, while also matching the diffusion decoder size used in DiffusionDrive. DiffusionDriveV2 takes three cropped and downscaled forward-facing camera images, concatenated as a 1024 $\times$ 256 image, and a rasterized BEV representation of the LiDAR point cloud as input. We use the pre-trained weights from DiffusionDrive as a cold start. Our model is then trained for 10 epochs on the navtrain split using reinforcement learning. We use the AdamW optimizer with a learning rate of $2\times 10^{-4}$ and a total batch size of 512, distributed across 8 NVIDIA L20 GPUs. The mode selector was trained for 20 epochs using the same configuration. In inference, similar to DiffusionDrive, our model can generate predictions using only 2 denoising steps. Detailed hyperparameter settings and dataset metric details are provided in the supplementary material.

### 5.3 Main Results

#### Result on NAVSIM v1.

As shown in Tab. 1, DiffusionDriveV2 achieves a state-of-the-art performance on NAVSIM v1 navtest split, with a PDMS of 91.2. Our model outperforms DiffusionDrive by 3.1 PDMS and significantly boosts the EP score by 5.3, which demonstrates that our model provides higher-quality and more efficient driving strategies. This improvement is attributed to our carefully designed reinforcement learning method. Compared to DIVER, which is also based on reinforcement learning, DiffusionDriveV2 also surpasses it by 2.9 PDMS. This result demonstrates the superior effectiveness of our Intra-Anchor GRPO and Inter-Anchor Truncated GRPO training framework. Moreover, DiffusionDriveV2 equipped with only a ResNet-34 backbone (21.8M params) still outperforms GoalFlow and Hydra-MDP, which are built upon the larger V2-99 \[lee2020centermask\] backbone (96.9M params).

#### Result on NAVSIM v2.

DiffusionDriveV2 maintains its strong performance on the more challenging NAVSIM v2 dataset, achieving a new state-of-the-art EPDMS as shown in Tab. 2. To ensure a fair comparison, all models evaluated in this benchmark utilize the same ResNet-34 backbone.

| Method | NC $\uparrow$ | DAC $\uparrow$ | DDC $\uparrow$ | TL $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | LK $\uparrow$ | HC $\uparrow$ | EC $\uparrow$ | EPDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ego Status MLP | 93.1 | 77.9 | 92.7 | 99.6 | 86.0 | 91.5 | 89.4 | 98.3 | 85.4 | 64.0 |
| Transfuser \[chitta2022transfuser\] | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 |
| Hydra-MDP++ \[li2025hydra\] | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | 81.4 |
| DriveSuprim \[yao2025drivesuprim\] | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 | 83.1 |
| ARTEMIS \[feng2025artemis\] | 98.3 | 95.1 | 98.6 | 99.8 | 81.5 | 97.4 | 96.5 | 98.3 | \- | 83.1 |
| DiffusionDriveV2 (Ours) | 97.7 | 96.6 | 99.2 | 99.8 | 88.9 | 97.2 | 96.0 | 97.8 | 91.0 | 85.5 |

Table 2: Comparison on NAVSIM v2 navtest split with extended closed-loop metrics.

#### Diversity and Quality.

Inspired by DIVER \[song2025breaking\], we introduce a Diversity Metric (denoted as $Div.$) to quantitatively evaluate a model’s ability to generate multi-modal trajectories. The metric defines the unnormalized pairwise diversity at waypoint $n$ as:

$$
Div_{\text{raw }}^{n}=\frac{2}{M(M-1)}\sum_{i=1}^{M-1}\sum_{j=i+1}^{M}\left(\left\|\mathbf{p}^{i}_{n}-\mathbf{p}^{j}_{n}\right\|_{2}\right).
$$

To ensure scale consistency for trajectories across different scenarios, they are normalized by the average trajectory scale:

$$
Div^{n}=\min\left(1,\frac{Div^{n}_{\mathrm{raw}}}{\epsilon+\frac{1}{M}\sum_{m=1}^{M}\left\|\mathbf{p}^{m}_{n}\right\|_{2}}\right).
$$

We report the average $Div^{n}$ across all waypoints as the final $Div.$ score. To assess the overall quality of the generated trajectories, we further report the Top-K PDMS. Since regression-based and selection-based E2E-AD methods can only generate deterministic trajectories, we exclusively compare our approach against other diffusion-based methods in the navtest dataset. Following DiffusionDrive, each model generates 20 trajectories for evaluation. The results are presented in Tab. 3.

| Method | $Div.$ | PDMS@1 | PDMS@5 | PDMS@10 |
| --- | --- | --- | --- | --- |
| $\text{Transfuser}_{\text{TD}}$ \[liao2025diffusiondrive\] | 0.1 | 85.7 | 85.7 | 85.7 |
| DiffusionDrive \[liao2025diffusiondrive\] | 42.3 | 93.5 | 84.3 | 75.3 |
| DiffusionDriveV2 (Ours) | 30.3 | 94.9 | 91.1 | 84.4 |

Table 3: Comparison of Diversity and Top-K PDMS for the raw generated trajectories on navtest. PDMS@K denotes the PDMS score evaluated on the Top-K ranked trajectories.

The results presented here are the models’ raw outputs, evaluated before their respective selection modules (i.e., DiffusionDrive’s classifier and DiffusionDriveV2’s selector). The goal of this analysis is to determine the extent to which these models produce low-quality trajectories and depend on their selection modules to ensure high quality. This over-reliance is a critical concern, as selector modules typically have fewer parameters, leading to weaker generalization capabilities. Consequently, they are prone to failure in out-of-distribution scenarios, creating a significant hazard for the system.

The results validate our theory. Vanilla diffusion methods achieve consistent generation quality but lack diversity, collapsing into a single “bored” trajectory. DiffusionDrive achieves very high generation diversity but fails to guarantee consistent high quality. In contrast, DiffusionDriveV2 utilizes a meticulously designed RL algorithm to achieve an exploration-constraint effect. It imposes constraints on all modes, raising the model’s lower bound (Top-10 PDMS), and pushes the model to explore better policies, raising its upper bound (Top-1 PDMS). DiffusionDriveV2 resolves the dilemma between diversity and consistent high quality for truncated diffusion models, achieving the best trade-off.

### 5.4 Ablation Studies

We conduct a series of ablation studies to verify the effectiveness of each design in DiffusionDriveV2. In this section, all ablation studies are conducted with the same set of hyperparameters for a fair comparison and are trained for fewer epochs for rapid validation.

#### The Type of Exploration Noise.

Tab. 4 shows the results of using different exploration noises. The findings confirm that scale-adaptive multiplicative noise is superior to additive noise, as it effectively addresses the scale inconsistency between the proximal and distal parts of a trajectory.

| Noise Type | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf. $\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| Add. | 98.1 | 97.2 | 94.4 | 99.9 | 85.3 | 89.7 |
| Multi. | 98.1 | 97.6 | 94.5 | 99.9 | 85.7 | 90.1 |

Table 4: Comparison of different type of exploration noise, where Add. and Multi. denote Additive Noise and Multiplicative Noise, respectively.

#### Impact of Intra-Anchor GRPO.

Tab. 5 shows the impact of Intra-Anchor GRPO. The results confirm our theoretical analysis, demonstrating that for a Gaussian Mixture Model like DiffusionDrive, which partitions the action space by driving intentions, performing advantage estimation within each anchor is critically important.

| Intra-Anchor | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf. $\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| ✗ | 97.9 | 97.3 | 93.8 | 99.5 | 84.9 | 89.2 |
| ✓ | 98.1 | 97.6 | 94.5 | 99.9 | 85.7 | 90.1 |

Table 5: Impact of Intra-Anchor GRPO.

#### Impact of Inter-Anchor Truncated GRPO.

Tab. 6 shows the impact of Inter-Anchor Truncated GRPO. Without this component, the model is unable to perform cross-mode comparisons, as the advantage estimates lose their global comparability and provide a misleading learning signal. In contrast, our method introduces global information for cross-mode evaluation, leading to superior performance.

| Inter-Trunc. | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf. $\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| ✗ | 97.7 | 97.3 | 93.6 | 99.9 | 85.7 | 89.5 |
| ✓ | 98.1 | 97.6 | 94.5 | 99.9 | 85.7 | 90.1 |

Table 6: Impact of Inter-Anchor Truncated GRPO. Inter-Trunc. stands for Inter-Anchor Truncated GRPO.

## 6 Conclusion

In this work, we have presented DiffusionDriveV2. By combining our proposed Intra-Anchor GRPO and Inter-Anchor Truncated GRPO with a scale-adaptive multiplicative exploration noise, our framework resolves the dilemma between diversity and consistent high quality for DiffusionDrive, which is caused by the incomplete multi-modal supervision from its imitation learning paradigm. Comprehensive experiments and qualitative comparisons validate that DiffusionDriveV2 achieves the best trade-off between consistent high planning quality and mode diversity and delivers state-of-the-art closed-loop performance.

Supplementary Material

## 7 Further Experiment Settings

#### Dataset and Metrics

NAVSIM v1 evaluates each trajectory by feeding it into a simulator, which then provides a score based on the trajectory’s interaction with the environment. The PDM score (PDMS) serves as the closed-loop planning metric and is calculated as:

$$
\mathrm{PDMS}=NC\times DAC\times\left(\frac{5\times EP+5\times TTC+2\times C}{12}\right),
$$

where the sub-metrics NC, DAC, TTC, C, EP represent the No At-Fault Collisions, Drivable Area Compliance, Time to Collision, Comfort, and Ego Progress.

NAVSIM v2 proposes an Extended PDM score (EPDMS), based on the previous formulation, which can be expressed as:

$$
\displaystyle\mathrm{EPDMS}={}\mathrm{NC}\times\mathrm{DAC}\times\mathrm{DDC}\times\mathrm{TL}\times
$$
 
$$
\displaystyle\frac{(5\times\mathrm{TTC}+2\times\mathrm{C}+5\times\mathrm{EP}+5\times\mathrm{LK}+5\times\mathrm{EC})}{22}
$$

The extended sub-metrics DDC, TL, LK, HC, and EC correspond to the Driving Direction Compliance, Traffic Lights Compliance, Lane Keeping, History Comfort and Extended Comfort, respectively.

#### Training Details

Our training process is primarily divided into two stages. In the first stage, we train the model using reinforcement learning. We use AdamW with a learning rate of $2\times 10^{-4}$, weight decay of $1\times 10^{-4}$, and a cosine learning rate schedule with a 10% linear warmup. We train for 10 epochs with a batch size of 512. Additionally, we set the minimum standard deviation of the multiplicative exploration noise to $0.04$ to push the model towards sufficient exploration and prevent entropy collapse. Furthermore, we set the standard deviation to be at least $0.1$ when evaluating the Gaussian likelihood $\log\pi_{\theta}\left(\tau_{t-1}^{k,i}\mid\tau_{t}^{k,i}\right)$, which improves training stability by avoiding large magnitude gradients. The denoising discount factor $\gamma$ is set to 0.8 to downweight the contribution of earlier noisy denoising steps in the policy gradient.

In the second stage, we train a mode selector to select the most goal-aligned trajectory from a set of multi-modal trajectories. The learning rate and batch size are the same as in the first stage. We train for 20 epochs with a batch size of 512. To enhance the robustness of our selector, we employ data augmentation during its training phase. Specifically, we augment the trajectories from our model by applying multiplicative Gaussian exploration noise, sampled from a standard deviation range of 0.1-0.2. Furthermore, we supplement the training data by randomly sampling 1% of trajectories from the fixed GTRS \[li2025generalized\] trajectory vocabulary, which further improves the selector’s robustness. The detailed hyperparameters are shown in Tab. 7.

Table 7: Hyperparameters for DiffusionDriveV2.

<table><thead><tr><th>Stage</th><th>Parameter</th><th>Value</th></tr></thead><tbody><tr><td rowspan="10">I</td><td>Number of epochs</td><td>10</td></tr><tr><td>Batch size</td><td>512</td></tr><tr><td>Learning rate</td><td><math><semantics><mrow><mn>2</mn> <mo></mo><msup><mi>e</mi> <mrow><mo>−</mo> <mn>4</mn></mrow></msup></mrow> <annotation>2e^{-4}</annotation></semantics></math></td></tr><tr><td>Weight decay</td><td><math><semantics><mrow><mn>1</mn> <mo></mo><msup><mi>e</mi> <mrow><mo>−</mo> <mn>4</mn></mrow></msup></mrow> <annotation>1e^{-4}</annotation></semantics></math></td></tr><tr><td>Warmup ratio</td><td>0.10</td></tr><tr><td>Learning rate schedule</td><td>Cosine</td></tr><tr><td>BC loss weight</td><td>0.1</td></tr><tr><td>Minimum denoising std</td><td>0.04</td></tr><tr><td>Minimum log-variance std</td><td>0.10</td></tr><tr><td>Discount factor</td><td>0.8</td></tr><tr><td rowspan="9">II</td><td>Number of epochs</td><td>20</td></tr><tr><td>Batch size</td><td>512</td></tr><tr><td>Learning rate</td><td><math><semantics><mrow><mn>2</mn> <mo></mo><msup><mi>e</mi> <mrow><mo>−</mo> <mn>4</mn></mrow></msup></mrow> <annotation>2e^{-4}</annotation></semantics></math></td></tr><tr><td>Weight decay</td><td><math><semantics><mrow><mn>1</mn> <mo></mo><msup><mi>e</mi> <mrow><mo>−</mo> <mn>4</mn></mrow></msup></mrow> <annotation>1e^{-4}</annotation></semantics></math></td></tr><tr><td>Warmup ratio</td><td>0.10</td></tr><tr><td>Learning rate schedule</td><td>Cosine</td></tr><tr><td>Number of aug trajs.</td><td>2</td></tr><tr><td>Aug noise std</td><td>(0.1, 0.2)</td></tr><tr><td>Percentage of GTRS vocab</td><td>1%</td></tr></tbody></table>

## 8 Further Ablation Studies

#### Effect of designs in mode selector.

Tab. 8 shows the effectiveness of our design choices in the mode selector. We observe that a more refined selector design can yield modest performance gains. Following DriveSuprim, we implement a two-stage, coarse-to-fine selector. A coarse selector first identifies the top-k candidate trajectories, which are then passed to a fine-grained selector for a more detailed comparison. This two-stage process leads to a 0.2 PDMS improvement over a single-stage approach. The auxiliary ranking loss further enhances the model’s capability to discern fine-grained differences in the continuous metrics, contributing an additional 0.2 PDMS gain.

| Model | Description | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf. $\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| $\mathcal{M}_{0}$ | Base Model | 97.9 | 97.4 | 94.1 | 99.9 | 85.4 | 89.7 |
| $\mathcal{M}_{1}$ | $\mathcal{M}_{0}+$ Coarse2Fine | 98.0 | 97.4 | 94.3 | 99.9 | 85.5 | 89.9 |
| $\mathcal{M}_{2}$ | $\mathcal{M}_{1}+$ Rank Loss | 98.1 | 97.6 | 94.5 | 99.9 | 85.7 | 90.1 |

Table 8: Ablation for design choices of mode selector.

#### Impact of mode selector.

DiffusionDriveV2 employs a more complex mode selector compared to DiffusionDrive’s classifier. To verify that our model’s superiority is not primarily attributed to this more complex selector, we performed an ablation study, with results presented in Tab. 9. Equipped with the same mode selector as DiffusionDriveV2, DiffusionDrive improved by 1 PDMS, but still lags significantly behind DiffusionDriveV2.

| Model | Selector | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf. $\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DiffusionDrive | ✗ | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| DiffusionDrive | ✓ | 97.2 | 97.0 | 92.2 | 100 | 86.8 | 89.1 |
| DiffusionDriveV2 | ✓ | 98.3 | 97.9 | 94.8 | 99.9 | 87.5 | 91.2 |

Table 9: The impact of mode selector.

## 9 Qualitative Comparison

In this section, we provide additional qualitative comparisons of Vanilla Diffusion, DiffusionDrive, and DiffusionDriveV2 in challenging scenarios from the navtest split of the planning-oriented NAVSIM dataset \[dauner2024navsim\]. For each method, we generate 20 trajectories. We use highly transparent colors to represent the candidate trajectories and highlight the Top-1 and Top-10 trajectories.

#### Going straight scenarios.

Fig. 4 illustrates the performance of different methods in relatively simple straight-driving scenarios. Vanilla Diffusion collapses to a single trajectory. While DiffusionDrive performs well in the straight-driving task, it still generates numerous colliding and off-road trajectories (circled in red). In contrast, the trajectories produced by DiffusionDriveV2 are more focused, with all candidate trajectories being of high quality.

#### Turning scenarios.

Fig. 6 illustrates the performance of different methods in turning scenarios. Similarly, Vanilla Diffusion collapses to a single trajectory. For DiffusionDrive, while the Top-1 trajectory performs well, the Top-10 trajectory exhibits delayed turning, resulting in a collision; moreover, it generates numerous colliding candidate trajectories. In contrast, DiffusionDriveV2 ensures high quality across all generated trajectories while preserving diversity. For instance, in Fig. 7(a), the Top-1 trajectory adopts a more conservative car-following behavior, whereas the Top-10 trajectory executes a more aggressive overtaking maneuver on the curve.

#### Multi-modal scenarios.

To evaluate the capability of each method in handling complex driving scenarios, we conducted visualizations in scenarios with multiple potential solutions, primarily at intersections, as shown in Fig. 8. Vanilla Diffusion still suffers from mode collapse, failing to provide alternative solutions. While DiffusionDrive offers diverse trajectory options (e.g., turning vs. going straight), it fails to guarantee consistent high quality. In contrast, DiffusionDriveV2 effectively resolves the dilemma between diversity and consistent high quality.

![[x4 17.png|Refer to caption]]

(a) Straight Scenario 1.

![[x6 15.png|Refer to caption]]

(a) Straight Scenario 3.

![[x7 12.png|Refer to caption]]

(a) Turning Scenario 1.

![[x8 9.png|Refer to caption]]

(a) Turning Scenario 2.

![[x10 4.png|Refer to caption]]

(a) Multi-Modal Scenario 1.
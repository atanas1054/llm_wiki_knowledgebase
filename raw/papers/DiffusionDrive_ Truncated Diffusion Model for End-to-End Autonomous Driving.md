---
title: "DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving"
source: "https://arxiv.org/html/2411.15139v1"
author:
published:
created: 2026-04-15
description:
tags:
  - "clippings"
---
Bencheng Liao <sup>1,2,⋄</sup>  Shaoyu Chen <sup>2,3</sup>  Haoran Yin <sup>3</sup>  Bo Jiang <sup>2,⋄</sup>  Cheng Wang <sup>1,2,⋄</sup>  Sixu Yan <sup>2</sup>    
Xinbang Zhang <sup>3</sup>  Xiangyu Li <sup>3</sup>  Ying Zhang <sup>3</sup>  Qian Zhang <sup>3</sup>  Xinggang Wang ${}^{2~\textrm{{\char 0\relax}}}$  
    <sup>1</sup> Institute of Artificial Intelligence, Huazhong University of Science & Technology  
    <sup>2</sup> School of EIC, Huazhong University of Science & Technology  
    <sup>3</sup> Horizon Robotics  
Code & Model & Demo: [hustvl/DiffusionDrive](https://github.com/hustvl/DiffusionDrive)

###### Abstract

Recently, the diffusion model has emerged as a powerful generative technique for robotic policy learning, capable of modeling multi-mode action distributions. Leveraging its capability for end-to-end autonomous driving is a promising direction. However, the numerous denoising steps in the robotic diffusion policy and the more dynamic, open-world nature of traffic scenes pose substantial challenges for generating diverse driving actions at a real-time speed. To address these challenges, we propose a novel truncated diffusion policy that incorporates prior multi-mode anchors and truncates the diffusion schedule, enabling the model to learn denoising from anchored Gaussian distribution to the multi-mode driving action distribution. Additionally, we design an efficient cascade diffusion decoder for enhanced interaction with conditional scene context. The proposed model, DiffusionDrive, demonstrates 10 $\times$ reduction in denoising steps compared to vanilla diffusion policy, delivering superior diversity and quality in just 2 steps. On the planning-oriented NAVSIM dataset, with aligned ResNet-34 backbone, DiffusionDrive achieves 88.1 PDMS without bells and whistles, setting a new record, while running at a real-time speed of 45 FPS on an NVIDIA 4090. Qualitative results on challenging scenarios further confirm that DiffusionDrive can robustly generate diverse plausible driving actions.

<sup>†</sup>

## 1 Introduction

End-to-end autonomous driving has gained significant attention in recent years due to advancements in perception models (detection [^21] [^35] [^14], tracking [^45] [^47] [^46], online mapping [^27] [^24] [^25], *etc*.), which directly learns the driving policy from the raw sensor inputs. This data-driven approach offers a scalable and robust alternative to traditional rule-based motion planning, which often struggles to generalize to complex real-world driving settings.

![[x1 19.png|Refer to caption]]

Figure 1: The comparison of different end-to-end paradigms. (a) Single mode regression 17 13 6. (b) Sampling from vocabulary 3 22. (c) Vanilla diffusion policy 5 16. (d) The proposed truncated diffusion policy.

To effectively learn from data, mainstream end-to-end planners (*e.g*., Transfuser [^6], UniAD [^13], VAD [^17]) typically regress a single-mode trajectory from an ego-query as shown in Fig. 1a. However, this paradigm does not account for the inherent uncertainty and multi-mode <sup>†</sup> nature of driving behaviors. Recently, VADv2 [^17] introduces a large fixed vocabulary of anchor trajectories (4096 anchors) to discretize the continuous action space and capture a broader range of driving behaviors, and then samples from these anchors based on predicted scores as shown in Fig. 1b. However, this large fixed-vocabulary paradigm is fundamentally constrained by the number and quality of anchor trajectories, often failing in out-of-vocabulary scenarios. Furthermore, managing a large number of anchors presents significant computational challenges for real-time applications. Rather than discretizing the action space, diffusion model [^5] has proven to be a powerful generative decision-making policy in the robotics domain, which can directly sample multi-mode physically plausible actions from a Gaussian distribution via an iterative denoising process.

This inspires us to replicate the success of the diffusion model in the robotics domain to end-to-end autonomous driving. We apply the vanilla robotic diffusion policy to the well-known single-mode-regression method, Transfuser [^6], by proposing a variant, Transfuser ${}_{\text{DP}}$, which replaces the deterministic MLP regression head with a conditional diffusion model [^28]. Though Transfuser ${}_{\text{DP}}$ improves planning performance, two major issues arise: 1) The numerous 20 denoising steps in the vanilla DDIM diffusion policy introduce heavy computational consumption during inference as shown in Tab. 2, hindering the real-time application for autonomous driving. 2) The trajectories sampled from different Gaussian noises severely overlap with each other, as illustrated in Fig. 2. This underscores the non-trivial challenge of taming the diffusion models for the dynamic and open-world traffic scenes.

Unlike the vanilla diffusion policy, which samples actions from a random Gaussian noise conditioned on scene context, human drivers adhere to established driving patterns that they dynamically adjust in response to real-time traffic conditions. This insight motivates us to embed these prior driving patterns into the diffusion policy by partitioning the Gaussian distribution into multiple sub-Gaussian distributions centered around prior anchors, referred to as anchored Gaussian distribution. It is implemented by truncating the diffusion schedule to introduce a small portion of Gaussian noise around the prior anchors as shown in Fig. 3. Thanks to the multi-mode distributional expressivity of the diffusion model, the proposed truncated diffusion policy effectively covers the potential action space without requiring a large set of fixed anchors, as VADv2 does. With more reasonable initial noise samples from the anchored Gaussian distribution, we can truncate the denoising process, reducing the required steps from 20 to just 2—a substantial speedup that satisfies the real-time requirements of autonomous driving.

![[x2 17.png|Refer to caption]]

(a) Top-1’s going straight and diverse top-10’s lane changing.

To enhance the interaction with conditional scene context, we propose an efficient transformer-based diffusion decoder that interacts not only with structured queries from the perception module but also with Bird’s Eye View (BEV) and perspective view (PV) features through a sparse deformable attention mechanism [^51]. Additionally, we introduce a cascade mechanism to iteratively refine the trajectory reconstruction within the diffusion decoder at each denoising step.

With these innovations, we present DiffusionDrive, a diffusion model for real-time end-to-end autonomous driving. We benchmark our method on the planning-oriented NAVSIM dataset [^9] using non-reactive simulation and closed-loop evaluations. Without bells and whistles, DiffusionDrive achieves 88.1 PDMS on NAVSIM navtest split with the aligned ResNet-34 backbone, significantly outperforming previous state-of-the-art methods. Even compared to the NAVSIM challenge-winning solution Hydra-MDP- $\mathcal{V}_{8192}$ -W-EP [^22], which follows VADv2 with 8192 anchor trajectories and further incorporates post-processing and additional supervision, DiffusionDrive still outperforms it by 1.6 PDMS through directly learning from human demonstrations and inferring without post-processing, while running at real-time speed of 45 FPS on an NVIDIA 4090. We further validate the superiority of DiffusionDrive on popular nuScenes dataset [^2] with open-loop evaluations, DiffusionDrive runs 1.8 $\times$ faster than VAD and outperforms it [^17] by 20.8% lower L2 error and 63.6% lower collision rate with the same ResNet-50 backbone, demonstrating state-of-the-art planning performance.

Our contributions can be summarized as follows:

- We firstly introduce the diffusion model to the field of end-to-end autonomous driving and propose a novel truncated diffusion policy to address the issues of mode collapse and heavy computational overhead found in direct adaptation of vanilla diffusion policy to the traffic scene.
- We design an efficient transformer-based diffusion decoder that interacts with the conditional information in a cascaded manner for better trajectory reconstruction.
- Without bells and whistles, DiffusionDrive significantly outperforms previous state-of-the-art methods, achieving a record-breaking 88.1 PDMS on the NAVSIM navtest split with the same backbone, while maintaining real-time performance at 45 FPS on an NVIDIA 4090.
- We qualitatively demonstrate that DiffusionDrive can generate more diverse and plausible trajectories, exhibiting high-quality multi-mode driving actions in various challenging scenarios.

## 2 Related Work

End-to-end autonomous driving. UniAD [^13], as a pioneering work, demonstrates the potential of end-to-end autonomous driving by integrating multiple perception tasks to enhance planning performance. VAD [^17] further explores the use of compact vectorized scene representations to improve efficiency. Subsequently, a series of works [^20] [^38] [^48] [^23] [^36] [^10] [^4] [^6] have adopted the single-trajectory planning paradigm to enhance planning performance further. More recently, VADv2 [^3] shifts the paradigm towards multi-mode planning by scoring and sampling from a large fixed vocabulary of anchor trajectories. Hydra-MDP [^22] improves the scoring mechanism of VADv2 by introducing extra supervision from a rule-based scorer. SparseDrive [^32] explores an alternative BEV-free solution. Unlike existing multi-mode planning approaches, we propose a novel paradigm that leverages powerful generative diffusion models for end-to-end autonomous driving.

Diffusion model for traffic simulation. Driving diffusion policy has been explored in the traffic simulation by leveraging only abstract perception groundtruth [^18] [^7] [^15] [^37]. MotionDiffuser [^18] and CTG [^50] are pioneering applications of diffusion models for multi-agent motion prediction, using a conditional diffusion model to sample target trajectories from Gaussian noise. CTG++ [^49] further incorporates a large language model (LLM) for language-driven guidance, improving usability and enabling realistic traffic simulations. Diffusion-ES [^41] replaces reward-gradient-guided denoising with evolutionary search. VBD [^15] introduces a scene-consistent scenario optimizer using a conditional diffusion model with game-theoretical guidance to generate abstract safety-critical driving scenarios. Moving beyond diffusion models limited to traffic simulation with perception groundtruth, our approach unlocks the potential of diffusion models for real-time, end-to-end autonomous driving through our proposed truncated diffusion policy and efficient diffusion decoder.

![[x4 16.png|Refer to caption]]

Figure 3: Illustration of truncated diffusion policy by comparing with vanilla diffusion policy. We truncate the diffusion process and only add a small portion of Gaussian noise to diffuse the anchor trajectories. Then, we train the diffusion model to reconstruct the ground-truth trajectory from the anchored Gaussian distribution with conditional scene context. During the inference, we also truncate the denoising process by starting from the better samples in the anchored Gaussian distribution than the pure Gaussian noise.

Diffusion model for robotic policy learning. Diffusion policy [^5] demonstrates the great potential in robotic policy learning, effectively capturing multi-mode action distributions and high-dimensional action spaces. Diffuser [^16] proposes an unconditional diffusion model for trajectory sampling, incorporating techniques such as classifier-free guidance and image inpainting to achieve guided sampling. Subsequently, numerous works have applied diffusion models to various robotic tasks, including stationary manipulation [^44] [^1], mobile manipulation [^40], autonomous navigation [^30] [^42], quadruped locomotion [^31], and dexterous manipulation [^39]. However, directly applying vanilla diffusion policy to end-to-end autonomous driving poses unique challenges, as it requires real-time efficiency and the generation of plausible multi-mode trajectories in dynamic and open-world traffic scenes. In this work, we propose a novel truncated diffusion policy to address these challenges, introducing concepts that have not yet been explored in the robotics field.

## 3 Method

![[x5 17.png|Refer to caption]]

Figure 4: Overall architecture of DiffusionDrive. (a) DiffusionDrive can integrate various existing perception modules and different sensor inputs. (b) The designed diffusion decoder takes the sampled noisy trajectories from anchored Gaussian distribution as input and progressively denoises them with enhanced interactions with the conditional scene context in a cascade manner to generate the final predictions.

### 3.1 Preliminary

Task formulation. End-to-end autonomous driving takes raw sensor data as input and predicts the future trajectory of the ego-vehicle. The trajectory is represented as a sequence of waypoints $\tau=\{(x_{t},y_{t})\}_{t=1}^{T_{f}}$, where $T_{f}$ denotes the planning horizon, and $(x_{t},y_{t})$ is the location of each waypoint at time $t$ in the current ego-vehicle coordinate system.

Conditional diffusion model. The conditional diffusion model poses a forward diffusion process as gradually adding noise to the data sample, which can be defined as:

$$
q\left(\tau^{i}\mid\tau^{0}\right)=\mathcal{N}\left(\tau^{i};\sqrt{\bar{\alpha}^{i}}\boldsymbol{\tau}^{0},\left(1-\bar{\alpha}^{i}\right)\mathbf{I}\right),
$$

where $\tau^{0}$ is the clean data sample, and $\tau^{i}$ is the data sample with noise at time $i$ (Note: we use superscript $i$ to denote diffusion timestep). The constant $\bar{\alpha^{i}}=\prod_{s=1}^{i}\alpha^{s}=\prod_{s=1}^{i}(1-\beta^{s})$ and $\beta^{s}$ is the noise schedule. We train the reverse process model $f_{\theta}(\tau^{i},z,i)$ to predict $\tau^{0}$ from $\tau^{i}$ with the guidance of conditional information $z$, where $\theta$ is the trainable model parameter. During inference, the trained diffusion model $f_{\theta}$ progressively refines from the random noise $\tau^{T}$ sampled in Gaussian distribution to the predicted clean data sample $\tau^{0}$ with the guidance of conditional information $z$, which is defined as:

$$
p_{\theta}\left(\boldsymbol{\tau}^{0}\mid z\right)=\int p\left(\boldsymbol{\tau}^{T}\right)\prod_{i=1}^{T}p_{\theta}\left(\boldsymbol{\tau}^{i-1}\mid\boldsymbol{\tau}^{i},z\right)\mathrm{d}\boldsymbol{\tau}^{1:T}.
$$

### 3.2 Investigation

Turn Transfuser [^6] into conditional diffusion model. We begin from the representative deterministic end-to-end planner Transfuser [^6] and turn it into a generative model Transfuser ${}_{\text{DP}}$ by simply replacing the regression MLP layers with the conditional diffusion model UNet following vanilla diffusion policy [^5]. During the evaluation, we sample a random noise and progressively refine it with 20 steps. Tab. 2 shows that Transfuser ${}_{\text{DP}}$ achieves better planning quality than deterministic Transfuser.

Mode collapse. To further investigate the multi-mode property of the vanilla diffusion policy in driving, we sampled 20 random noises from Gaussian distribution and denoised them using 20 steps. As shown in Fig. 2, the different random noises converge to similar trajectories after the denoising process. To quantitatively analyze the phenomenon of mode collapse, we define a mode diversity score $\mathcal{D}$ based on the mean Intersection over Union (mIoU) between each denoised trajectory and the union of all denoised trajectories:

$$
\mathcal{D}=1-\frac{1}{N}\sum_{i=1}^{N}\frac{\text{Area}(\tau_{i}\cap\bigcup_{j=1}^{N}\tau_{j})}{\text{Area}(\tau_{i}\cup\bigcup_{j=1}^{N}\tau_{j})},
$$

where $\tau_{i}$ represents the $i$ -th denoised trajectory, $N$ is the total number of sampled trajectories and $\bigcup_{j=1}^{N}\tau_{j}$ is the union of all denoised trajectories. A higher mIoU indicates less diversity of the denoised trajectories. The quantitative mode diversity results in Tab. 2 further validate the observations presented in Fig. 2.

Heavy denoising overhead. The DDIM [^29] diffusion policy requires 20 denoising steps to transform random noise into a feasible trajectory, which introduces significant computational overhead, reducing the FPS from 60 to 7, as shown in Tab. 2, and making it impractical for real-time online driving applications.

| Method | Input | Img. Backbone | Anchor | NC $\uparrow$ | DAC $\uparrow$ | TTC $\uparrow$ | Comf. $\uparrow$ | EP $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UniAD [^13] | Camera | ResNet-34 [^11] | 0 | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| PARA-Drive [^38] | Camera | ResNet-34 [^11] | 0 | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 |
| LTF [^6] | Camera | ResNet-34 [^11] | 0 | 97.4 | 92.8 | 92.4 | 100 | 79.0 | 83.8 |
| Transfuser [^6] | C & L | ResNet-34 [^11] | 0 | 97.7 | 92.8 | 92.8 | 100 | 79.2 | 84.0 |
| DRAMA [^43] | C & L | ResNet-34 [^11] | 0 | 98.0 | 93.1 | 94.8 | 100 | 80.1 | 85.5 |
| VADv2- $\mathcal{V}_{8192}$ [^3] | C & L | ResNet-34 [^11] | 8192 | 97.2 | 89.1 | 91.6 | 100 | 76.0 | 80.9 |
| Hydra-MDP- $\mathcal{V}_{8192}$ [^22] | C & L | ResNet-34 [^11] | 8192 | 97.9 | 91.7 | 92.9 | 100 | 77.6 | 83.0 |
| Hydra-MDP- $\mathcal{V}_{8192}$ -W-EP [^22] | C & L | ResNet-34 [^11] | 8192 | 98.3 | 96.0 | 94.6 | 100 | 78.7 | 86.5 |
| DiffusionDrive (Ours) | C & L | ResNet-34 [^11] | 20 | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |

Table 1: Comparison on planning-oriented NAVSIM navtest split with closed-loop metrics. “C & L” denotes the use of both camera and LiDAR as sensor inputs. “ $\mathcal{V}_{8192}$ ” denotes 8192 anchors. “Hydra-MDP- $\mathcal{V}_{8192}$ -W-EP” is a variant of Hydra-MDP [^22], which is further trained to fit the EP evaluation metric with additional supervision from the rule-based evaluator and uses weighted confidence post-processing. DiffusionDrive simply learns from human demonstrations and infers without post-processing. The best and the second best results are denoted by bold and underline.

<table><thead><tr><th rowspan="2">Method</th><th rowspan="2">NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th rowspan="2">DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th rowspan="2">TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th rowspan="2">Comf.<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th rowspan="2">EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th></th><th colspan="4">Plan Module Time</th><th rowspan="2"><math><semantics><mrow><mi>𝒟</mi> <mo>↑</mo></mrow> <annotation>\mathcal{D}\uparrow</annotation></semantics></math></th><th rowspan="2">Para.<math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th><th rowspan="2">FPS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th></tr><tr><th>PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>Arch.</th><th>Step Time <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th><th>Steps <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th><th>Total <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th></tr></thead><tbody><tr><td>Transfuser</td><td>97.7</td><td>92.8</td><td>92.8</td><td>100</td><td>79.2</td><td>84.0</td><td>MLP</td><td>0.2ms</td><td>1</td><td>0.2ms</td><td>0%</td><td>56M</td><td>60</td></tr><tr><td>Transfuser <math><semantics><msub><mtext>DP</mtext></msub> <annotation>{}_{\text{DP}}</annotation></semantics></math></td><td>97.5</td><td>93.7</td><td>92.7</td><td>100</td><td>79.4</td><td>84.6 <math><semantics><msub><mtext>+0.6</mtext></msub> <annotation>{}_{\textbf{\text{\color[rgb]{0,0.88,0}+0.6}}}</annotation></semantics></math></td><td>UNet</td><td>6.5ms</td><td>20</td><td>130.0ms</td><td>11%</td><td>101M</td><td>7</td></tr><tr><td>Transfuser <math><semantics><msub><mtext>TD</mtext></msub> <annotation>{}_{\text{TD}}</annotation></semantics></math></td><td>97.9</td><td>94.2</td><td>93.9</td><td>100</td><td>80.2</td><td>85.7 <math><semantics><msub><mtext>+1.7</mtext></msub> <annotation>{}_{\textbf{\text{\color[rgb]{0,0.88,0}+1.7}}}</annotation></semantics></math></td><td>UNet</td><td>6.9ms</td><td>2</td><td>13.8ms</td><td>70%</td><td>102M</td><td>27</td></tr><tr><td>DiffusionDrive</td><td>98.2</td><td>96.2</td><td>94.7</td><td>100</td><td>82.2</td><td>88.1 <math><semantics><msub><mtext>+4.1</mtext></msub> <annotation>{}_{\textbf{\text{\color[rgb]{0,0.88,0}+4.1}}}</annotation></semantics></math></td><td>Dec.</td><td>3.8ms</td><td>2</td><td>7.6ms</td><td>74%</td><td>60M</td><td>45</td></tr></tbody></table>

Table 2: Roadmap from Transfuser to DiffusionDrive on NAVSIM navtest split. “Transfuser ${}_{\text{DP}}$ ” denotes Transfuser with vanilla DDIM diffusion policy [^5]. “Transfuser ${}_{\text{TD}}$ ” denotes Transfuser with truncated diffusion policy. “Step Time” denotes the runtime of each denoising step. “FPS” and runtime are measured on an NVIDIA 4090 GPU. “ $\mathcal{D}$ ” denotes the mode diversity score defined in Eq. (3).

### 3.3 Truncated Diffusion

Human driving follows fixed patterns, unlike the random noise denoising in vanilla diffusion policy. Motivated by this, we propose a truncated diffusion policy that begins the denoising process from an anchored Gaussian distribution instead of a standard Gaussian distribution. To enable the model to learn to denoise from the anchored Gaussian distribution to the desired driving policy, we further truncate the diffusion schedule during training, adding only a small amount of Gaussian noise to the anchors.

Training. We first construct the diffusion process by adding Gaussian noise to anchors $\{\mathbf{a}_{k}\}_{k=1}^{N_{\text{anchor}}}$ clustered by K-Means on the training set, where $\mathbf{a}_{k}=\{(x_{t},y_{t})\}_{t=1}^{T_{f}}$. We truncate the diffusion noise schedule to diffuse the anchors to the anchored Gaussian distribution:

$$
\tau_{k}^{i}=\sqrt{\bar{\alpha}^{i}}\mathbf{a}_{k}+\sqrt{1-\bar{\alpha}^{i}}\boldsymbol{\epsilon},\quad\boldsymbol{\epsilon}\sim\mathcal{N}(0,\mathbf{I}),
$$

where $i\in[1,T_{\text{trunc}}]$ and $T_{\text{trunc}}\ll T$ is the truncated diffusion steps.

During training, the diffusion decoder $f_{\theta}$ takes as input $N_{\text{anchor}}$ noisy trajectories $\{\tau_{k}^{i}\}_{k=1}^{N_{\text{anchor}}}$ and predicts classification scores $\{\hat{s}_{k}\}_{k=1}^{N_{\text{anchor}}}$ and denoised trajectories $\{\hat{\tau}_{k}\}_{k=1}^{N_{\text{anchor}}}$:

$$
\{\hat{s}_{k},\hat{\tau}_{k}\}_{k=1}^{N_{\text{anchor}}}=f_{\theta}(\{\tau_{k}^{i}\}_{k=1}^{N_{\text{anchor}}},z),
$$

where $z$ represents the conditional information. We assign the noisy trajectory around the closest anchor to the ground truth trajectory $\tau_{\text{gt}}$ as positive sample ($y_{k}=1$) and others as negative samples ($y_{k}=0$). The training objective combines trajectory reconstruction and classification:

$$
\mathcal{L}=\sum_{k=1}^{N_{\text{anchor}}}[y_{k}\mathcal{L}_{\text{rec}}(\hat{\tau}_{k},\tau_{\text{gt}})+\lambda\text{BCE}(\hat{s}_{k},y_{k})],
$$

where $\lambda$ balances the simple L1 reconstruction loss $\mathcal{L}_{\text{rec}}$ and binary cross-entropy (BCE) classification loss.

Inference. We use a truncated denoising process that starts with noisy trajectories sampled from the anchored Gaussian distribution and progressively denoises them to final predictions. At each denoising timestep, the estimated trajectories from the previous step are passed to the diffusion decoder $f_{\theta}$, which predicts classification scores $\{\hat{s}_{k}\}_{k=1}^{N_{\text{infer}}}$ and coordinates $\{\hat{\tau}_{k}\}_{k=1}^{N_{\text{infer}}}$. After obtaining the current timestep’s predictions, we apply the DDIM [^29] update rule to sample trajectories for the next timestep.

Inference flexibility. A key advantage of our approach lies in its inference flexibility. While the model is trained with $N_{\text{anchor}}$ trajectories, the inference process can accommodate an arbitrary number of trajectory samples $N_{\text{infer}}$, where $N_{\text{infer}}$ can be dynamically adjusted based on computational resources or application requirements.

### 3.4 Architecture

The overall architecture of our proposed method, DiffusionDrive, is illustrated in Fig. 4. DiffusionDrive can integrate various existing perception modules used in previous end-to-end planners [^13] [^17] [^6] [^32] and take different sensor inputs. The designed diffusion decoder is tailored for the complex and challenging driving application, which has enhanced interactions with the conditional scene context.

Diffusion decoder. Given the set of sampled noisy trajectories $\{\hat{\tau}_{k}\}_{k=1}^{N_{\text{infer}}}$ from the anchored Gaussian distribution, we begin by applying deformable spatial cross-attention [^51] [^35] [^26] to interact with Bird’s Eye View (BEV) or Perspective View (PV) features based on the trajectory coordinates. Subsequently, cross-attention is performed between the trajectory features and the agent/map queries derived from the perception module, followed by a feed-forward network (FFN). To encode the diffusion timestep information, we utilize a Timestep Modulation layer, which is followed by a Multi-Layer Perceptron (MLP) that predicts the confidence score and the offset relative to the initial noisy trajectory coordinates. The output from this diffusion decoder layer serves as the input for the subsequent cascade diffusion decoder layer. DiffusionDrive further reuses the cascade diffusion decoder to iteratively denoise the trajectory during inference, with parameters shared across the different denoising timesteps. The final trajectory with the highest confidence score is selected as the output.

## 4 Experiment

<table><tbody><tr><td rowspan="2">ID</td><td>UNet</td><td>Ego Query</td><td>Spatial</td><td>Agent/Map</td><td>Cascade</td><td rowspan="2">Param.<math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td colspan="6">Planning Metric</td></tr><tr><td>Decoder</td><td>Interaction</td><td>Cross-attn</td><td>Cross-attn</td><td>Decoder</td><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Comf.<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>PDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>1</td><td>✓</td><td>✓</td><td>✗</td><td>✗</td><td>✗</td><td>102M</td><td>97.9</td><td>94.2</td><td>93.9</td><td>100</td><td>80.2</td><td>85.7</td></tr><tr><td>2</td><td>✗</td><td>✓</td><td>✗</td><td>✗</td><td>✗</td><td>57M</td><td>88.7</td><td>83.2</td><td>80.0</td><td>84.8</td><td>43.3</td><td>55.1</td></tr><tr><td>3</td><td>✗</td><td>✓</td><td>✓</td><td>✗</td><td>✗</td><td>58M</td><td>98.2</td><td>95.4</td><td>94.4</td><td>100</td><td>81.3</td><td>87.1</td></tr><tr><td>4</td><td>✗</td><td>✓</td><td>✗</td><td>✓</td><td>✗</td><td>58M</td><td>97.9</td><td>93.5</td><td>93.8</td><td>100</td><td>79.8</td><td>85.1</td></tr><tr><td>5</td><td>✗</td><td>✓</td><td>✓</td><td>✓</td><td>✗</td><td>59M</td><td>98.0</td><td>95.8</td><td>94.4</td><td>100</td><td>81.7</td><td>87.4</td></tr><tr><td>6</td><td>✗</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>60M</td><td>98.2</td><td>96.2</td><td>94.7</td><td>100</td><td>82.2</td><td>88.1</td></tr></tbody></table>

Table 3: Ablation for design choices. “Cascade Decoder” indicates that we stack 2 cascade diffusion decoder layers. ID-1 refers to Transfuser ${}_{\text{TD}}$ in Tab. 2, utilizing conditional UNet and interaction with the ego-query, which Transfuser uses to directly regress the single-mode trajectory.

| Steps | Param. | NC | DAC | TTC | Comf. | EP | PDMS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 60M | 98.3 | 96.0 | 94.7 | 100 | 82.1 | 87.9 |
| 2 | 60M | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| 3 | 60M | 98.2 | 96.3 | 94.7 | 100 | 92.2 | 88.1 |

Table 4: Denoising step number.

| Stages | Param. | NC | DAC | TTC | Comf. | EP | PDMS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 59M | 98.0 | 95.8 | 94.4 | 100 | 81.7 | 87.4 |
| 2 | 60M | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| 4 | 65M | 98.4 | 96.2 | 94.9 | 100 | 82.4 | 88.2 |

Table 5: Cascade stages.

| $N_{\text{infer}}$ | Param. | NC | DAC | TTC | Comf. | EP | PDMS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 60M | 97.9 | 93.5 | 93.1 | 100 | 80.0 | 84.9 |
| 20 | 60M | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| 40 | 60M | 98.5 | 96.2 | 94.8 | 100 | 82.5 | 88.2 |

Table 6: Number of sampled noises $N_{\text{infer}}$.

### 4.1 Dataset

NAVSIM. The NAVSIM dataset [^9] is a real-world planning-oriented dataset builds upon OpenScene [^8], a compact redistribution of nuPlan [^19], the largest publicly available annotated driving dataset. NAVSIM leverages eight cameras to achieve a full 360 <sup>∘</sup> FOV, along with a merged LiDAR point cloud derived from five sensors. Annotations are provided at a frequency of 2Hz and include both HD maps and object bounding boxes. The dataset is designed to emphasize challenging driving scenarios involving dynamic changes in driving intentions, while deliberately excluding trivial situations such as stationary scenes or constant-speed driving.

NAVSIM benchmarks planning performance using non-reactive simulations and closed-loop metrics for comprehensive evaluation. In this paper, we employ the proposed PDM score (PDMS) [^9], which is a weighted combination of several sub-scores: no at-fault collisions (NC), drivable area compliance (DAC), time-to-collision (TTC), comfort (Comf.), and ego progress (EP).

### 4.2 Implementation Detail

We adopt the same perception modules and ResNet-34 backbone [^11] as Transfuser for fair comparison. In the diffusion decoder layer, we employ spatial cross-attention to only interact with BEV features following Transfuser’s BEV-based setting. We only perform agent cross-attention, since the perception module of Transfuser does not include vectorized map construction. We stack 2 cascade diffusion decoder layers and apply truncated diffusion policy with 20 clustered anchors. The training diffusion schedule is truncated by 50/1000 to diffuse the anchors, while during inference, we use only 2 denoising steps and select the top-1 scoring predicted trajectory for evaluation. The training and inference recipe directly follows Transfuser: We use three cropped and downscaled forward-facing camera images, concatenated as a 1024 $\times$ 256 image, and a rasterized BEV LiDAR as input; DiffusionDrive is trained on navtrain split from scratch for 100 epochs with AdamW optimizer on 8 NVIDIA 4090 GPUs with total batch size of 512, setting the learning rate to $6\times 10^{-4}$. No test-time augmentation is applied and the final output for evaluation on navtest split is 8-waypoint trajectory over 4 seconds. Further details refer to Appendix.

### 4.3 Quantitative Comparison

Tab. 1 compares DiffusionDrive with state-of-the-art methods on NAVSIM navtest split. With the same ResNet-34 backbone, DiffusionDrive achieves 88.1 PDMS score, demonstrating significant superior performance over the previous learning-based methods. Compared to VADv2, DiffusionDrive surpasses it by 7.2 PDMS while reducing the number of anchors from 8192 to 20, representing a 400 $\times$ reduction. DiffusionDrive also outperforms Hydra-MDP, which follows VADv2’s sampling-from-vocabulary paradigm, with a 5.1 PDMS improvement. Even compared to the Hydra-MDP- $\mathcal{V}_{8192}$ -W-EP, which is a variant of Hydra-MDP [^22] by further training to fit the EP evaluation metric with additional supervision and using weighted confidence post-processing, DiffusionDrive still outperforms it by 3.5 EP and 1.6 overall PDMS, relying solely on a straightforward learning-from-human approach without any post-processing. Compared to the Transfuser baseline, where we only differ in the planning module, DiffusionDrive delivers a notable 4.1 PDMS improvement, outperforming it across all sub-scores.

### 4.4 Roadmap

<table><thead><tr><th rowspan="2">Method</th><th rowspan="2">Input</th><th rowspan="2">Img. Backbone</th><th colspan="4">L2 (m) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th><th colspan="4">Collision Rate (%) <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th><th></th></tr><tr><th>1s</th><th>2s</th><th>3s</th><th>Avg.</th><th>1s</th><th>2s</th><th>3s</th><th>Avg.</th><th>FPS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th></tr></thead><tbody><tr><td>ST-P3 <sup><a href="#fn:12">12</a></sup></td><td>Camera</td><td>EffNet-b4 <sup><a href="#fn:33">33</a></sup></td><td>1.33</td><td>2.11</td><td>2.90</td><td>2.11</td><td>0.23</td><td>0.62</td><td>1.27</td><td>0.71</td><td>1.6</td></tr><tr><td>UniAD <sup><a href="#fn:13">13</a></sup></td><td>Camera</td><td>ResNet-101 <sup><a href="#fn:11">11</a></sup></td><td>0.45</td><td>0.70</td><td>1.04</td><td>0.73</td><td>0.62</td><td>0.58</td><td>0.63</td><td>0.61</td><td>1.8</td></tr><tr><td>OccNet <sup><a href="#fn:34">34</a></sup></td><td>Camera</td><td>ResNet-50 <sup><a href="#fn:11">11</a></sup></td><td>1.29</td><td>2.13</td><td>2.99</td><td>2.14</td><td>0.21</td><td>0.59</td><td>1.37</td><td>0.72</td><td>2.6</td></tr><tr><td>VAD <sup><a href="#fn:17">17</a></sup></td><td>Camera</td><td>ResNet-50 <sup><a href="#fn:11">11</a></sup></td><td>0.41</td><td>0.70</td><td>1.05</td><td>0.72</td><td>0.07</td><td>0.17</td><td>0.41</td><td>0.22</td><td>4.5</td></tr><tr><td>SparseDrive <sup><a href="#fn:32">32</a></sup></td><td>Camera</td><td>ResNet-50 <sup><a href="#fn:11">11</a></sup></td><td>0.29</td><td>0.58</td><td>0.96</td><td>0.61</td><td>0.01</td><td>0.05</td><td>0.18</td><td>0.08</td><td>9.0</td></tr><tr><td>DiffusionDrive (Ours)</td><td>Camera</td><td>ResNet-50 <sup><a href="#fn:11">11</a></sup></td><td>0.27</td><td>0.54</td><td>0.90</td><td>0.57</td><td>0.03</td><td>0.05</td><td>0.16</td><td>0.08</td><td>8.2</td></tr></tbody></table>

Table 7: Comparison on nuScenes dataset with open-loop metrics. FPS is measured on a single NVIDIA 4090 GPU following the recipe of SparseDrive [^32]. Metric calculation follows ST-P3 [^12].

In Tab. 2, converting Transfuser into the generative Transfuser ${}_{\text{DP}}$ using vanilla diffusion policy improves the PDMS score by 0.6 and the mode diversity score $\mathcal{D}$ by 11%. However, it also significantly increases the overhead of the planning module, requiring 20 $\times$ more denoising steps and 32 $\times$ the step time, resulting in a total 650 $\times$ increase in runtime overhead. With the proposed truncated diffusion policy, Transfuser ${}_{\text{TD}}$ reduces the number of denoising steps from 20 to 2 while achieving an increase of 1.1 in PDMS and a 59% improvement in mode diversity. By further incorporating the proposed diffusion decoder, the final model, DiffusionDrive, reaches 88.1 PDMS and 74% mode diversity score $\mathcal{D}$. Compared to the Transfuser ${}_{\text{DP}}$, DiffusionDrive shows improvements in 3.5 PDMS and 64% mode diversity, and a 10 $\times$ reduction in denoising steps, resulting in a 6 $\times$ speedup in FPS. This enables real-time, high-quality, multi-mode planning.

### 4.5 Ablation Study

Effect of designs in diffusion decoder. Tab. 3 shows the effectiveness of our design choices in the diffusion decoder. ID-1 is the Transfuser ${}_{\text{TD}}$ in the Tab. 2. By comapring ID-6 and ID-1, we can see that the proposed diffusion decoder reduce the 39% parameters and significantly improves the planning quality by 2.4 PDMS. ID-2 shows severe performance degeneration due to the lack of rich and hierarchical interaction with the environment. By comparing ID-2 and ID-3, we can see that spatial cross-attention is vital for accurate planning. ID-5 shows that the proposed cascade mechanism is effective and can further improve the performance.

Denoising step number. Tab. 6 shows that due to the reasonable start point, DiffusionDrive can achieve a good planning quality with only 1 step. Further increasing the denoising steps can improve the planning quality, and make it enjoy the flexible inference given the complexity of the environment.

Cascade stages. Tab. 6 ablates the impact of cascade stage number. Increasing the stage number can improve the planning quality but saturate at the 4 stages and cost more parameters and inference time at each step.

Number of sampled noises $N_{\text{infer}}$. As stated in Sec. 3.3, DiffusionDrive can generate varied trajectories by simply sampling a variable number of noises from anchored Gaussian distribution. Tab. 6 shows that 10 sampled noises can already achieve a decent planning quality. By sampling more noises, DiffusionDrive can cover potential planning action space and lead to improved planning quality.

### 4.6 Qualitative Comparison

Since the PDMS planning metric calculates based on the top-1 scoring trajectory and our proposed $\mathcal{D}$ score evaluates mode diversity, these metrics alone cannot fully capture the quality of diverse trajectories. To further validate the quality of multi-mode trajectories, we visualize the planning results of Transfuser, Transfuser ${}_{\text{DP}}$ and DiffusionDrive on challenging scenarios of NAVSIM navtest split in Fig. 2. The results indicate that the multi-mode trajectories generated by DiffusionDrive are not only diverse but also of high quality. In Fig. 2(a), the top-1 scoring trajectory generated by DiffusionDrive closely resembles the ground-truth trajectory, while the highlighted top-10 scoring trajectory surprisingly tries to perform high-quality lane changing. In Fig. 2(b), the highlighted top-10 scoring trajectory also performs a lane change, and a neighboring low-scoring trajectory further interacts with surrounding agents to effectively avoid collisions.

### 4.7 Quantitative Comparison on nuScenes dataset

The nuScenes dataset is previously popular benchmark for end-to-end planning. Since the major scenarios of nuScenes are simple and trivial situations, we only perform comparison in Tab. 7. We implement DiffusionDrive on top of SparseDrive [^32] following its training and inference recipe using open-loop metrics proposed in ST-P3 [^12]. We stack 2 cascade diffusion decoder layers and apply the truncated diffusion policy with 18 clustered anchors.

As shown in Tab. 7, DiffusionDrive reduces the average L2 error of SparseDrive by 0.04m, achieving the lowest L2 error and average collision rate against previous state-of-the-art methods. While DiffusionDrive is also efficient and runs 1.8 $\times$ faster than VAD with 20.8% lower L2 error and 63.6% lower collision rate.

## 5 Conclusion

In this work, we propose a novel generative driving decision-making model, DiffusionDrive, for end-to-end autonomous driving by incorporating the proposed truncated diffusion policy and efficient cascade diffusion decoder. DiffusionDrive can denoise a variable number of samples from an anchored Gaussian distribution to generate diverse planning trajectories at real-time speeds. Comprehensive experiments and qualitative comparisons validate the superiority of DiffusionDrive in planning quality, running efficiency, and mode diversity.

[^1]: Anurag Ajay, Yilun Du, Abhi Gupta, Joshua B. Tenenbaum, Tommi S. Jaakkola, and Pulkit Agrawal. Is conditional generative modeling all you need for decision making? In *ICLR*, 2023.

[^2]: Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In *CVPR*, 2020.

[^3]: Shaoyu Chen, Bo Jiang, Hao Gao, Bencheng Liao, Qing Xu, Qian Zhang, Chang Huang, Wenyu Liu, and Xinggang Wang. Vadv2: End-to-end vectorized autonomous driving via probabilistic planning. *arXiv preprint arXiv:2402.13243*, 2024a.

[^4]: Zhili Chen, Maosheng Ye, Shuangjie Xu, Tongyi Cao, and Qifeng Chen. Ppad: Iterative interactions of prediction and planning for end-to-end autonomous driving. In *ECCV*, 2024b.

[^5]: Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. In *RSS*, 2023.

[^6]: Kashyap Chitta, Aditya Prakash, Bernhard Jaeger, Zehao Yu, Katrin Renz, and Andreas Geiger. Transfuser: Imitation with transformer-based sensor fusion for autonomous driving. *TPAMI*, 2022.

[^7]: Younwoo Choi, Ray Coden Mercurius, Soheil Mohamad Alizadeh Shabestary, and Amir Rasouli. Dice: Diverse diffusion model with scoring for trajectory prediction. In *IV*, 2024.

[^8]: OpenScene Contributors. Openscene: The largest up-to-date 3d occupancy prediction benchmark in autonomous driving. [https://github.com/OpenDriveLab/OpenScene](https://github.com/OpenDriveLab/OpenScene), 2023.

[^9]: Daniel Dauner, Marcel Hallgarten, Tianyu Li, Xinshuo Weng, Zhiyu Huang, Zetong Yang, Hongyang Li, Igor Gilitschenski, Boris Ivanovic, Marco Pavone, Andreas Geiger, and Kashyap Chitta. Navsim: Data-driven non-reactive autonomous vehicle simulation and benchmarking. In *NeurIPS*, 2024.

[^10]: Xunjiang Gu, Guanyu Song, Igor Gilitschenski, Marco Pavone, and Boris Ivanovic. Producing and leveraging online map uncertainty in trajectory prediction. In *CVPR*, 2024.

[^11]: Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *CVPR*, 2016.

[^12]: Shengchao Hu, Li Chen, Penghao Wu, Hongyang Li, Junchi Yan, and Dacheng Tao. St-p3: End-to-end vision-based autonomous driving via spatial-temporal feature learning. In *ECCV*, 2022.

[^13]: Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai Wang, et al. Planning-oriented autonomous driving. In *CVPR*, 2023.

[^14]: Junjie Huang, Guan Huang, Zheng Zhu, Yun Ye, and Dalong Du. Bevdet: High-performance multi-camera 3d object detection in bird-eye-view. *arXiv preprint arXiv:2112.11790*, 2021.

[^15]: Zhiyu Huang, Zixu Zhang, Ameya Vaidya, Yuxiao Chen, Chen Lv, and Jaime Fernández Fisac. Versatile scene-consistent traffic scenario generation as optimization with diffusion. *arXiv preprint arXiv:2404.02524*, 2024.

[^16]: Michael Janner, Yilun Du, Joshua B. Tenenbaum, and Sergey Levine. Planning with diffusion for flexible behavior synthesis. In *ICLR*, 2022.

[^17]: Bo Jiang, Shaoyu Chen, Qing Xu, Bencheng Liao, Jiajie Chen, Helong Zhou, Qian Zhang, Wenyu Liu, Chang Huang, and Xinggang Wang. Vad: Vectorized scene representation for efficient autonomous driving. In *ICCV*, 2023a.

[^18]: Chiyu Jiang, Andre Cornman, Cheolho Park, Benjamin Sapp, Yin Zhou, Dragomir Anguelov, et al. Motiondiffuser: Controllable multi-agent motion prediction using diffusion. In *CVPR*, 2023b.

[^19]: Napat Karnchanachari, Dimitris Geromichalos, Kok Seang Tan, Nanxiang Li, Christopher Eriksen, Shakiba Yaghoubi, Noushin Mehdipour, Gianmarco Bernasconi, Whye Kit Fong, Yiluan Guo, et al. Towards learning-based planning: The nuplan benchmark for real-world autonomous driving. In *ICRA*, 2024.

[^20]: Yingyan Li, Lue Fan, Jiawei He, Yuqi Wang, Yuntao Chen, Zhaoxiang Zhang, and Tieniu Tan. Enhancing end-to-end autonomous driving with latent world model. *arXiv preprint arXiv:2406.08481*, 2024a.

[^21]: Zhiqi Li, Wenhai Wang, Hongyang Li, Enze Xie, Chonghao Sima, Tong Lu, Yu Qiao, and Jifeng Dai. Bevformer: Learning bird’s-eye-view representation from multi-camera images via spatiotemporal transformers. In *ECCV*, 2022.

[^22]: Zhenxin Li, Kailin Li, Shihao Wang, Shiyi Lan, Zhiding Yu, Yishen Ji, Zhiqi Li, Ziyue Zhu, Jan Kautz, Zuxuan Wu, et al. Hydra-mdp: End-to-end multimodal planning with multi-target hydra-distillation. *arXiv preprint arXiv:2406.06978*, 2024b.

[^23]: Zhiqi Li, Zhiding Yu, Shiyi Lan, Jiahan Li, Jan Kautz, Tong Lu, and Jose M Alvarez. Is ego status all you need for open-loop end-to-end autonomous driving? In *CVPR*, 2024c.

[^24]: Bencheng Liao, Shaoyu Chen, Xinggang Wang, Tianheng Cheng, Qian Zhang, Wenyu Liu, and Chang Huang. MapTR: Structured modeling and learning for online vectorized HD map construction. In *ICLR*, 2023.

[^25]: Bencheng Liao, Shaoyu Chen, Yunchi Zhang, Bo Jiang, Qian Zhang, Wenyu Liu, Chang Huang, and Xinggang Wang. Maptrv2: An end-to-end framework for online vectorized hd map construction. *IJCV*, 2024.

[^26]: Xuewu Lin, Tianwei Lin, Zixiang Pei, Lichao Huang, and Zhizhong Su. Sparse4d: Multi-view 3d object detection with sparse spatial-temporal fusion. *arXiv preprint arXiv:2211.10581*, 2022.

[^27]: Yicheng Liu, Tianyuan Yuan, Yue Wang, Yilun Wang, and Hang Zhao. Vectormapnet: End-to-end vectorized hd map learning. In *ICML*, 2023.

[^28]: Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In *MICCAI*, 2015.

[^29]: Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In *ICLR*, 2021.

[^30]: Ajay Sridhar, Dhruv Shah, Catherine Glossop, and Sergey Levine. Nomad: Goal masked diffusion policies for navigation and exploration. In *ICRA*, 2024.

[^31]: Maria Stamatopoulou, Jianwei Liu, and Dimitrios Kanoulas. Dippest: Diffusion-based path planner for synthesizing trajectories applied on quadruped robots. *arXiv preprint arXiv:2405.19232*, 2024.

[^32]: Wenchao Sun, Xuewu Lin, Yining Shi, Chuang Zhang, Haoran Wu, and Sifa Zheng. Sparsedrive: End-to-end autonomous driving via sparse scene representation. *arXiv preprint arXiv:2405.19620*, 2024.

[^33]: Mingxing Tan and Quoc Le. Efficientnet: Rethinking model scaling for convolutional neural networks. In *ICML*, 2019.

[^34]: Wenwen Tong, Chonghao Sima, Tai Wang, Li Chen, Silei Wu, Hanming Deng, Yi Gu, Lewei Lu, Ping Luo, Dahua Lin, et al. Scene as occupancy. In *ICCV*, 2023.

[^35]: Yue Wang, Vitor Campagnolo Guizilini, Tianyuan Zhang, Yilun Wang, Hang Zhao, and Justin Solomon. Detr3d: 3d object detection from multi-view images via 3d-to-2d queries. In *CoRL*, 2022.

[^36]: Yuqi Wang, Jiawei He, Lue Fan, Hongxin Li, Yuntao Chen, and Zhaoxiang Zhang. Driving into the future: Multiview visual forecasting and planning with world model for autonomous driving. In *CVPR*, 2024a.

[^37]: Yixiao Wang, Chen Tang, Lingfeng Sun, Simone Rossi, Yichen Xie, Chensheng Peng, Thomas Hannagan, Stefano Sabatini, Nicola Poerio, Masayoshi Tomizuka, et al. Optimizing diffusion models for joint trajectory prediction and controllable generation. In *ECCV*, 2024b.

[^38]: Xinshuo Weng, Boris Ivanovic, Yan Wang, Yue Wang, and Marco Pavone. Para-drive: Parallelized architecture for real-time autonomous driving. In *CVPR*, 2024a.

[^39]: Zehang Weng, Haofei Lu, Danica Kragic, and Jens Lundell. Dexdiffuser: Generating dexterous grasps with diffusion models. *arXiv preprint arXiv:2402.02989*, 2024b.

[^40]: Sixu Yan, Zeyu Zhang, Muzhi Han, Zaijin Wang, Qi Xie, Zhitian Li, Zhehan Li, Hangxin Liu, Xinggang Wang, and Song-Chun Zhu. M2diffuser: Diffusion-based trajectory optimization for mobile manipulation in 3d scenes. *arXiv preprint arXiv:2410.11402*, 2024.

[^41]: Brian Yang, Huangyuan Su, Nikolaos Gkanatsios, Tsung-Wei Ke, Ayush Jain, Jeff Schneider, and Katerina Fragkiadaki. Diffusion-es: Gradient-free planning with diffusion for autonomous driving and zero-shot instruction following. In *CVPR*, 2024.

[^42]: Wenhao Yu, Jie Peng, Huanyu Yang, Junrui Zhang, Yifan Duan, Jianmin Ji, and Yanyong Zhang. Ldp: A local diffusion planner for efficient robot navigation and collision avoidance. *arXiv preprint arXiv:2407.01950*, 2024.

[^43]: Chengran Yuan, Zhanqi Zhang, Jiawei Sun, Shuo Sun, Zefan Huang, Christina Dao Wen Lee, Dongen Li, Yuhang Han, Anthony Wong, Keng Peng Tee, et al. Drama: An efficient end-to-end motion planner for autonomous driving with mamba. *arXiv preprint arXiv:2408.03601*, 2024.

[^44]: Yanjie Ze, Gu Zhang, Kangning Zhang, Chenyuan Hu, Muhan Wang, and Huazhe Xu. 3d diffusion policy. In *RSS*, 2024.

[^45]: Fangao Zeng, Bin Dong, Yuang Zhang, Tiancai Wang, Xiangyu Zhang, and Yichen Wei. Motr: End-to-end multiple-object tracking with transformer. In *ECCV*, 2022.

[^46]: Yifu Zhang, Chunyu Wang, Xinggang Wang, Wenjun Zeng, and Wenyu Liu. Fairmot: On the fairness of detection and re-identification in multiple object tracking. *IJCV*, 2021.

[^47]: Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan, Ping Luo, Wenyu Liu, and Xinggang Wang. Bytetrack: Multi-object tracking by associating every detection box. In *ECCV*, 2022.

[^48]: Wenzhao Zheng, Ruiqi Song, Xianda Guo, Chenming Zhang, and Long Chen. Genad: Generative end-to-end autonomous driving. In *ECCV*, 2024.

[^49]: Ziyuan Zhong, Davis Rempe, Yuxiao Chen, Boris Ivanovic, Yulong Cao, Danfei Xu, Marco Pavone, and Baishakhi Ray. Language-guided traffic simulation via scene-level diffusion. In *CoRL*, 2023a.

[^50]: Ziyuan Zhong, Davis Rempe, Danfei Xu, Yuxiao Chen, Sushant Veer, Tong Che, Baishakhi Ray, and Marco Pavone. Guided conditional diffusion for controllable traffic simulation. In *ICRA*, 2023b.

[^51]: Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable detr: Deformable transformers for end-to-end object detection. In *ICLR*, 2021.
---
title: "Fine-tuning is Not Enough: A Parallel Framework for Collaborative Imitation and Reinforcement Learning in End-to-end Autonomous Driving"
source: "https://arxiv.org/html/2603.13842v3"
author:
published:
created: 2026-06-23
description:
tags:
  - "clippings"
---
Zhexi Lian <sup>1,†</sup>, Haoran Wang <sup>1,†</sup>, Xuerun Yan <sup>1,2,†</sup>, Weimeng Lin <sup>1</sup>, Xianhong Zhang <sup>1</sup>,  
Yongyu Chen <sup>3</sup>, Jia Hu <sup>1,🖂</sup>  
<sup>1</sup> Tongji University, <sup>2</sup> Nanyang Technological University, <sup>3</sup> Chery Automobile  
† Equal contribution    🖂 Corresponding author  
The code repository: [https://github.com/zhexilian/PaIR-Drive](https://github.com/zhexilian/PaIR-Drive)

###### Abstract

End-to-end autonomous driving is typically built upon imitation learning (IL), yet its performance is constrained by the quality of human demonstrations. To overcome this limitation, recent methods incorporate reinforcement learning (RL) through sequential fine-tuning. However, such a paradigm remains suboptimal: sequential RL fine-tuning can introduce policy drift and often leads to a performance ceiling due to its dependence on the pretrained IL policy. To address these issues, we propose PaIR-Drive, a general Parallel framework for collaborative Imitation and Reinforcement learning in end-to-end autonomous driving. During training, PaIR-Drive separates IL and RL into two parallel branches with conflict-free training objectives, enabling fully collaborative optimization. This design eliminates the need to retrain RL when applying a new IL policy. During inference, RL leverages the IL policy to further optimize the final plan, allowing performance beyond prior knowledge of IL. Furthermore, we introduce a tree-structured trajectory neural sampler to group relative policy optimization (GRPO) in the RL branch, which enhances exploration capability. Extensive analysis on NAVSIMv1 and v2 benchmark demonstrates that PaIR-Drive achieves Competitive performance of 91.2 PDMS and 87.9 EPDMS, building upon Transfuser and DiffusionDrive IL baselines. PaIR-Drive consistently outperforms existing RL fine-tuning methods, and could even correct human experts’ suboptimal behaviors. Qualitative results further confirm that PaIR-Drive can effectively explore and generate high-quality trajectories.

## 1 Introduction

![[Intro_humanbad.png|Refer to caption]]

Figure 1: Examples of human’s bad behaviors in the real-world dataset NAVSIM. (a) Singapore: The human drives in the wrong direction on the opposite lane; (b) Las Vegas: The human violates traffic light and turns left.

![[Intro_existing_method_0314.png|Refer to caption]]

Figure 2: Comparisons of existing training schemes and ours for end-to-end autonomous driving. (a) One-shot IL → \\to RL: IL-based training with subsequent RL fine-tuning; (b) Iterative IL ↔ \\leftrightarrow RL: alternately conducting IL training and RL fine-tuning; (c) Ours parallel framework for collaborative IL and RL.

End-to-end autonomous driving has been developing at a fast pace in recent years. Current mainstream end-to-end autonomous driving methods are typically built upon imitation learning (IL), which aims to mimic human experts’ demonstrations directly from sensor inputs, as exemplified by UniAD [^20], VAD [^23], DiffusionDrive [^39], etc. Although effective in learning stability, the IL policy is constrained by the quality of human expert demonstrations: (1) IL policy may blindly mimic human’s bad behaviors [^19] [^43]. Fig. 1 shows some bad behaviors extracted from the large-scale real world dataset NAVSIM, which may misguide the IL policy; (2) IL policy also suffers from low-value driving scenarios. For instance, the dominance of straight-driving scenes (73.9%) in the nuScenes dataset may cause the IL policy to lack knowledege on dealing with other scenario types [^37]. Hence, IL-based autonomous driving faces significant challenges in applications.

A potential solution to improve the IL policy is fine-tuning through reinforcement learning (RL). By leveraging reward functions to guide the optimization direction, RL enables the policy to refine its behaviors based on trial-and-error feedback in a closed loop fashion [^18] [^5]. Existing methods have made great progress in sequentially fine-tuning IL policy via RL, which can be categorized as two types: (a) one-shot IL $\to$ RL: IL-based training with subsequent RL based fine-tuning [^57] [^34] [^35] [^47] [^19] [^26] [^16] [^1] [^38] [^29], and (b) iterative IL $\leftrightarrow$ RL: alternately conducting IL training and RL fine-tuning [^44] [^14] [^3] [^12] [^52] [^11] [^42] [^7]. Fig. 2 (a) shows the one-shot IL $\to$ RL, a scheme that has gained significant attention after the notable success of DeepSeek-R1 [^10]. Some designs include pure on-policy RL fine-tuning through PPO, GRPO, etc, to enhance driving reasoning capability [^57] [^35] [^47] or trajectory generation quality [^34] [^16], but these random sampling-based RL fine-tuning results in low sample efficiency when interacting with the environment. Other designs incorporate IL regularization into RL fine-tuning, such as adding an imitation loss to RL rewards [^21] [^2] or adding expert demonstrations to the replay buffer [^19]. However, this RL fine-tuning scheme continues to face challenges in policy drift (even resulting in lower performance than the original IL policy) and may lead to a performance ceiling due to its dependence on the pretrained IL policy.

Taking into account the above issues, the iterative IL $\leftrightarrow$ RL scheme has been proposed, in which IL and RL updates are repeatedly switched, as shown in Fig. 2 (b). This scheme helps to regularize the policy optimization, allowing IL to anchor the policy distribution toward expert behaviors, while RL gradually improves performance through exploration. Common designs include inserting IL updates after several RL updates [^44] [^14], condition-activated RL among IL iterations [^11] [^42], adversarial turn-based updates between IL and RL [^25]. However, IL and RL fine-tuning still conduct upon the same policy network. Given that IL and RL involve different optimization objectives, this fine-tuning scheme could be trapped in a local minimum due to destructive conflicts of inconsistent optimization directions. Hence, a critical scientific question we need to answer is:

*How can we design a unified reinforcement and imitation learning framework to harmonize training objectives, and ultimately surpass IL’s prior performance?*

To this end, as shown in Fig. 2 (c), we propose PaIR-Drive, a Parallel framework for collaborative Imitation and Reinforcement learning in end-to-end autonomous driving. The key insight of PaIR-Drive is breaking the upper performance limit of sequential fine-tuning through our parallel framework. In the parallel IL+RL scheme illustrated in Fig. 3, the IL branch follows a typical sensor encoding $\rightarrow$ perception fusion $\rightarrow$ trajectory decoding pipeline. IL’s trajectory output is supervised by the human trajectory. Simultaneously, the RL branch takes the BEV feature maps and human expert trajectory as queries to further explore better trajectories. To be specific, we design a tree-structured trajectory neural sampler to predict the trajectory point offsets of driving intentions unseen in human demonstrations. The sampler operates recurrently, where the trajectory tree expansion at each step is conditioned on previous steps. Finally, we use trajectories and their simulated rewards for GRPO to update the policy. As for the inference scheme in Fig. 4, we can just replace the human trajectory in the RL branch with the trajectory generated by the IL branch, while employing an additional trained reward world model (RWM) to evaluate and select the final plan. The main contributions are listed as follows:

![[overall_framework.png|Refer to caption]]

Figure 3: Training process illustration of the parallel scheme of PaIR-Drive. IL branch follows a typical end-to-end planning fashion and is supervised by the human trajectory. Simultaneously, the RL branch builds upon human trajectories and aims to further explore better trajectories. In the RL branch, a tree-structured trajectory neural sampler is designed to recurrently predict the trajectory point offsets of driving intentions unseen in human demonstrations. Finally, we use trajectories and their simulated rewards for GRPO to update the policy.

- We propose PaIR-Drive, a general parallel framework of IL and RL for end-to-end autonomous driving. By decoupling IL and RL into parallel optimization branches, PaIR-Drive leverages IL to learn human-level driving behavior and RL explores how to surpass human expert performance. Extensive experiments on NAVSIM v1 and v2 benchmarks demonstrate competitive performance of PaIR-Drive.
- PaIR-Drive could serve as a general performance enhancement toolkit. It can be seamlessly integrated into any IL-based autonomous driving method. Its RL branch is built upon human expert demonstrations rather than a specific IL policy, making the framework flexible, adaptive, and widely applicable across different systems.
- We introduce a tree-structured trajectory neural sampler to GRPO in the RL branch, which enhances exploration efficiency and improves trajectory quality. Qualitative results showcase the exploration and high-quality trajectory generation capabilities.

## 2 Related works

IL-based end-to-end autonomous driving. This type of method aims to map sensor inputs to trajectories directly with human expert supervision. UniAD [^20] plays a crucial role in this area as it firstly leverages the advantages of perception and prediction modules for planning. VAD [^23] [^6] introduces vectorized representations for end-to-end planning. Hydra-MDP [^36] employs knowledge distillation to derive supervision from both rule-based planners and human demonstrations. Diffusion policy [^39] [^55] [^22] [^15] [^50], world models [^56] [^32] [^17] [^31] [^28] [^27], flow matching [^51] [^48], vision-language models [^54] [^49] [^40] [^46], etc, are introduced step-by-step, further enhancing the IL driving policy. However, the performance of IL is constrained by the quality of demonstrations and suffers from low-value driving scenarios.

RL-based end-to-end autonomous driving. This type of method enables the policy to refine its behaviors based on reward guidance and trial-and-error feedback. Existing methods mainly leverage RL to fine-tune the IL policy, including one-shot IL $\to$ RL and iterative IL $\leftrightarrow$ RL [^30] [^41]. RAD [^16] pretrains a IL policy and fine-tunes it through 3DGS simulation-based RL. Carplanner [^53] combines an auto-regressive RL with an imitation loss to achieve SOTA performance on the challenging large-scale real-world dataset nuPlan. AlphaDrive [^24] follows a supervised fine-tuning and reinforced fine-tuning scheme and is the first to leverage the advantages of GRPO. PlanRL [^3] switches between IL and RL learning when facing different conditions. However, given that IL and RL involve different optimization objectives, the fine-tuning scheme could be trapped in a local minimum due to destructive conflicts of optimization directions. Moreover, current IL and RL fine-tuning continues to face challenges in policy drift and may lead to a performance ceiling due to its dependence on the pretrained IL policy.These issues motivate our PaIR-Drive to break the upper performance limit of sequential fine-tuning through our parallel scheme.

![[inference.png|Refer to caption]]

Figure 4: Inferring process illustration of the parallel scheme of PaIR-Drive. Compared with Fig. 3, we replace the human trajectory in the RL branch with the trajectory generated by the IL branch, while employing an additional trained reward world model to evaluate and select the final plan.

## 3 PaIR-Drive

In this section, we introduce four key components of the PaIR-Drive: IL and RL branches formulation (Sec. 3.1), the tree-structured trajectory neural sampler (Sec. 3.2), the training scheme (Sec. 3.3), and the inferring scheme (Sec. 3.4).

### 3.1 Problem formulation

IL branch formulation. The IL branch of the PaIR-Drive takes ego status, multi-view RGB images from cameras, and point clouds from the lidar as inputs. We use an image encoder and a lidar encoder to obtain perception features $\mathbf{F}_{img}=ImgEncoder(Img),\mathbf{F}_{pcd}=LidarEncoder(Pcd)$. Following a sequential perception fusion and trajectory decoder module, the IL branch generates trajectory output referred to following equations.

$$
\displaystyle\tau_{0:T}^{IL}=TajDecoder(\mathbf{F}_{BEV})
$$
 
$$
\displaystyle\mathbf{F}_{BEV}=PercepFusion(\mathbf{F}_{img},\mathbf{F}_{pcd})
$$
 
$$
\displaystyle\tau_{0:T}^{IL}=\{w_{0}^{IL},w_{1}^{IL},\dots,w_{T}^{IL}\}
$$

$\tau_{0:T}^{IL}\in\mathbb{R}^{(T+1)\times 3}$ denotes the planned trajectory in the future time horizon $T$, which includes trajectory point $w_{t}^{IL}$ at each time step $t$. $w_{t}^{IL}$ contains longitudinal position $x_{t}$, lateral position $y_{t}$, and heading $h_{t}$ in the ego coordinate. The $PercepFusion$ and $TajDecoder$ are borrowed from Transfuser [^8]. We would like to emphasize that this IL branch can be replaced by any IL-based autonomous driving policy to generate $\tau_{0:T}^{IL}$.

RL branch formulation. The RL branch of the PaIR-Drive aims to explore better trajectories. It also takes ego status, multi-view RGB images from cameras, and point clouds from the lidar as inputs and generates BEV feature maps $\mathbf{F}_{BEV}$. Then, we introduce a tree-structured trajectory neural sampler $TreeSampler_{i},i\in Intention$. It takes $\mathbf{F}_{BEV}$ and a human expert trajectory as input and predicts the trajectory point offsets relative to the expert trajectory in each intention space (left, right, accelerating, decelerating, etc) in a recurrent manner:

$$
\displaystyle\Delta w_{t,i}^{RL}=TreeSampler_{i}(w_{t,i}^{RL},\mathbf{F}_{BEV})
$$
 
$$
\displaystyle w_{t,i}^{RL}=\begin{cases}w_{0}^{Human},&t=0\text{ if }\ \text{training}\\
w_{0}^{IL},&t=0\text{ if }\ \text{inferring}\\
w_{t-1,i}^{RL}+\Delta w_{t-1,i}^{RL},&t>0\end{cases}
$$
 
$$
\displaystyle\Delta w_{t,i}^{RL}=\{\Delta x_{t},\Delta y_{t},\Delta h_{t}\}
$$
 
$$
\displaystyle\tau_{0:T,i}^{RL}=\{w_{0,i}^{RL},w_{1,i}^{RL},\dots,w_{T,i}^{RL}\}
$$

$\Delta w_{t,i}^{RL}$ denotes the trajectory point offsets under intention $i$ which include longitudinal position offset $\Delta x_{t}$, lateral position offset $\Delta y_{t}$, and heading offset $\Delta h_{t}$ in the ego coordinate. The RL branch’s trajectory output $\tau_{0:T,i}^{RL}$ with different intentions $i$ would expand around the reference trajectory (human expert trajectory in training and IL branch’s trajectory output in inferring) with a tree structure. Finally, we use the trajectories and their simulation-based rewards for GRPO to update the policy.

![[tree_sampler.png|Refer to caption]]

Figure 5: Illustration of the tree-structured trajectory neural sampler with the capability of generating trajectories under different driving intentions unseen in human demonstrations.

### 3.2 Tree-structured trajectory neural sampler

The core component of the RL branch is the tree-structured trajectory neural sampler. It aims to predict the trajectory point offsets relative to the reference trajectory (human expert trajectory in training and IL branch’s trajectory output in inferring) under different driving intentions. The prediction follows a recurrent manner so that trajectories associated with different intentions are progressively expanded, forming a trajectory tree that branches out along the temporal dimension.

Expansion with intentions. The trajectory offset prediction incorporates $N$ different driving intentions such as Left, Right, Acc.(accelerating), Dec.(decelerating), etc. Assuming that the trajectory tree has already branched out to $M$ trajectories at time step $t$, we need to predict trajectory points offsets with different intentions and expand the original trajectory tree to $M\times N$ at time step $t+1$. Given that the whole trajectory could branch out $T^{N}$ trajectories which are high-dimensional, we expand trajectories every two steps and candidates with higher exploitative value are selected for further rollouts (More details can be found in the supplementary materials). Hence, the neural sampler could ensure exploration efficiency.

Architecture. The architecture is illustrated in Fig. 5. The input includes $\mathbf{F}_{BEV}$ and the trajectory tree expanded during previous steps. We first generate trajectory tokens $\mathbf{Token}_{traj}\in\mathbb{R}^{(T+1)\times 128}$ through a trajectory encoder. Then we concatenate $\mathbf{Token}_{intetion}\in\mathbb{R}^{N\times 128}$ produced from intentions with trajectory tokens $\mathbf{Token}_{traj}$. We use a series of multi-head self-attention and cross-attention blocks to capture the latent interaction mechanism between BEV space, trajectory space, and intention space. We predict the trajectory offsets for different intentions using the offset prediction head $OffsetHead$, and predict the log-probability of each offset through the score head $ScoreHead$. In contrast to previous methods, which generate trajectories through randomly sampling [^34] [^57], the $OffsetHead$ regularizes the offsets prediction range of a specific intention so as to avoid sampling inefficiency (More details can be found in supplementary materials). Finally, we add the offsets prediction to the original trajectory tree and output the trajectory tree at the current step.

### 3.3 Training scheme

This section introduces how we conduct training on the parallel IL+RL framework.

IL branch training. The IL branch training follows typical supervision fashion as shown in Fig. 3 (I). We optimize the L1 error between the IL branch’s trajectory output $\tau_{0:T}^{IL}$ and the human expert trajectory $\tau_{0:T}^{Human}$:

$$
\mathcal{L}_{IL}=\mathrm{L1loss}(\tau_{0:T}^{IL},\tau_{0:T}^{Human})
$$

The IL branch training is not associated with the RL branch and can be conducted independently.

RL branch training. The RL branch aims to explore better trajectories. Based on the human expert trajectory, the tree-structured trajectory neural sampler branches out to a trajectory tree with $G$ different intentions $\tau_{0:T}^{RL}=\{\tau_{0:T,i}^{RL},i=1\dots G,i\in Intention\}$. These trajectories are simulated in the NAVSIM simulator online and evaluated by a predefined reward $r_{i}$, which includes driving safety, drivable area compliance, driving efficiency, driving comfort, etc. The current RL policy $\pi_{\theta}$ is then optimized through GRPO using the normalized group-relative advantage $A_{i}$.

$$
\displaystyle\mathcal{J}_{RL}=\frac{1}{G}\sum_{i=1}^{G}(\mathcal{J}_{i}-\beta\mathbb{D}_{KL}(\pi_{\theta}|\pi_{old}))
$$
 
$$
\displaystyle\mathcal{J}_{i}=\mathrm{min}(\frac{\pi_{\theta}(\tau_{0:T,i}^{RL})}{\pi_{old}(\tau_{0:T,i}^{RL})}A_{i},\mathrm{clip}(\frac{\pi_{\theta}(\tau_{0:T,i}^{RL})}{\pi_{old}(\tau_{0:T,i}^{RL})},1-\epsilon,1+\epsilon)A_{i})
$$
 
$$
\displaystyle A_{i}=\frac{r_{i}-\mathrm{mean}\left(\{r_{j}\}_{j=1}^{G}\right)}{\mathrm{std}\left(\{r_{j}\}_{j=1}^{G}\right)}
$$

where $\pi_{old}$ denotes the previous RL policy; $\epsilon$ and $\beta$ are hyperparameters of the clipping range and the KL divergence regularization weight. It can be seen that the RL branch training doesn’t depend on the IL branch, so that RL and IL can be conducted in our parallel scheme.

### 3.4 Inferring scheme

The inferring scheme is illustrated in Fig. 4. There are only two differences compared to training the parallel scheme. First, the human expert trajectory is directly replaced by the IL branch’s trajectory output. This design is simple but quite beneficial for IL, which has been validated by our experiments. It also means that the RL branch could serve as a general performance enhancement toolkit and be seamlessly integrated into any IL-based autonomous driving method without any retraining. Second, after obtaining the final trajectory tree, we use a reward world model (RWM) to score the trajectories and select the best trajectory as the final plan. The RWM is a lightweight, data-driven alternative to traditional simulators. It predicts the reward $r_{i}$ and the confidence $conf_{i}$ of the trajectory $\tau_{0:T,i}^{RL}$ based on current driving commands $c$, and BEV feature maps $\mathbf{F}_{BEV}$:

$$
r_{i},conf_{i}=RWM(\mathbf{F}_{BEV},c,\tau_{0:T,i}^{RL})
$$

This design can filter those RL branch’s exploratory trajectories that are worse than the IL branch’s trajectory output.

Table 1: The human suboptimal behavior improvement capability evaluation on NAVSIMv1 benchmark. The green score means the improvement compared with the human driver.

Data split Agent NC $\uparrow$ DAC $\uparrow$ EP $\uparrow$ TTC $\uparrow$ C $\uparrow$ PDMS $\uparrow$ Human bad v1 human 100.0 100.0 60.1 99.6 94.8 82.3 human w/ PaIR-Drive 100.0 100.0 62.5 99.9 97.5 83.9(+1.6) Navtest human 100.0 100.0 87.4 100.0 99.6 94.7 human w/ PaIR-Drive 100.0 100.0 89.6 100.0 99.5 95.5(+0.8)

Table 2: The human suboptimal behavior improvement capability evaluation on NAVSIMv2 benchmark. The green score means the improvement compared with the human driver.

<table><tbody><tr><td>Data split</td><td>Agent</td><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DDC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TLC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>LK <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>HC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EPDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td rowspan="2">Human bad v2</td><td>human</td><td>100.0</td><td>100.0</td><td>97.0</td><td>70.4</td><td>83.2</td><td>99.6</td><td>46.2</td><td>82.7</td><td>56.6</td><td>50.0</td></tr><tr><td>human w/ PaIR-Drive</td><td>100.0</td><td>100.0</td><td>98.3</td><td>77.5</td><td>84.2</td><td>99.4</td><td>66.5</td><td>82.4</td><td>49.3</td><td>60.8(+10.8)</td></tr><tr><td rowspan="2">Navtest</td><td>human</td><td>100.0</td><td>100.0</td><td>99.7</td><td>97.4</td><td>87.4</td><td>100.0</td><td>87.4</td><td>98.1</td><td>90.1</td><td>90.3</td></tr><tr><td>human w/ PaIR-Drive</td><td>100.0</td><td>100.0</td><td>99.9</td><td>98.0</td><td>89.6</td><td>100.0</td><td>91.7</td><td>98.1</td><td>86.4</td><td>91.9(+1.6)</td></tr></tbody></table>

## 4 Experiments

### 4.1 Experimental Setup

Benchmarks. We train and evaluate PaIR-Drive on the large-scale real-world simulation benchmarks NAVSIMv1 [^9] and NAVSIMv2 [^4], which are designed to evaluate autonomous driving performance in complex urban scenarios. The benchmarks include high-quality and challenging scenarios extracted from the Openscene dataset.

Metrics. The NAVSIMv1 and NAVSIMv2 benchmarks use PDMS and EPDMS, respectively, to evaluate driving behavior performance. PDMS is a combination of several sub-scores and multiplicative penalties, including No-Collision (NC), Drivable Area Compliance (DAC), Ego Vehicle Progress (EP), Time-to-Collision (TTC), and Comfort (C) [^9].

$$
\mathrm{PDMS}=\mathrm{NC}\times\mathrm{DAC}\times\frac{5\times\mathrm{EP}+5\times\mathrm{TTC}+2\times\mathrm{C}}{12}
$$

EPDMS extends the PDMS, introducing Driving Direction Compliance (DDC), Traffic Light Compliance (TLC), Lane Keeping (LK), History Comfort(HC), and Extended Comfort (EC) [^4]. $Humanfilter$ can filter those sub-scores that human scores zero, but won’t affect the final EPDMS.

$$
\begin{aligned} \mathrm{EPDMS}&=Humanfilter(\mathrm{NC}\times\mathrm{DAC}\times\mathrm{DDC}\times\mathrm{TLC}\\
&\quad\times\frac{5\times\mathrm{EP}+5\times\mathrm{TTC}+2\times\mathrm{LK}+2\times\mathrm{HC}+2\times\mathrm{EC}}{16})\end{aligned}
$$

Data splits. The official navtest split is used for evaluation. Moreover, to evaluate the human experts’ suboptimal behaviors improvement capability of the PaIR-Drive, we extract the human bad v1 and human bad v2 splits from the navtest split. The scenarios in human bad v1 are those in which human’s PDMS less than 85. The scenarios in human bad v2 are those in which human’s EPDMS less than 80.

Implementation Details. The IL branch and RL branch of the PaIR-Drive both utilize RGB images with a resolution of $1024\times 256$ and Lidar’s ponit cloud feature with a resolution of $256\times 256$. ResNet34 was selected as the perception backbone. The IL branch was trained for 50 epochs using 4 NVIDIA L40 GPUs, each with a batch size of 32. The training used the AdamW optimizer with a learning rate of 1e-4. The RL branch was also trained for 50 epochs using 4 NVIDIA L40 GPUs, each with a batch size of 16. The AdamW optimizer was also adopted, and the learning rate began with 2e-5 and decayed by a cosine schedule. The group size and clip range $\epsilon$ of GRPO were set to 15 and 0.2, respectively. We adopted a dynamic KL divergence regularization weight $\beta$ to stabilize the training process.

Table 3: Comparison with methods on NAVSIMv1 benchmark. The best and second-best scores are highlighted in bold and underlined, respectively. The green score means the improvement compared with the origin IL policy. †: using the best-of-N (N=6) strategy following [^57].

Method type Agent Source NC $\uparrow$ DAC $\uparrow$ EP $\uparrow$ TTC $\uparrow$ C $\uparrow$ PDMS $\uparrow$ IL AutoVLA w/o GRPO [^57] NIPS’25 96.9 92.4 75.8 88.1 99.9 80.5 VADv2 [^6] ICCV’23 97.2 89.1 76.0 91.6 100.0 80.9 Transfuser w/o RL [^8] TPAMI’23 97.7 92.8 79.2 92.8 100.0 84.0 ReCogDrive w/o RL [^34] arxiv’25 98.3 95.1 81.1 94.3 100.0 86.8 ARTEMIS [^13] arxiv’25 98.3 95.1 81.4 94.3 100.0 87.0 DiffusionDrive [^39] CVPR’25 98.2 96.2 82.2 94.7 100.0 88.1 WoTE [^33] ICCV’25 98.5 96.8 81.9 94.9 99.9 88.3 DriveDPO w/o RL [^45] NIPS’25 97.9 97.3 84.0 93.6 100.0 88.8 Sequential IL+RL Transfuser w/ GRPO [^8] TPAMI’23 98.0 94.7 88.5 96.6 100.0 87.9(+3.9) ReCogDrive w/ GRPO [^34] arxiv’25 98.2 97.8 83.5 95.2 99.8 89.6(+2.8) DriveDPO w/ DPO [^45] NIPS’25 98.5 98.1 84.3 94.8 99.9 90.0(+1.2) AutoVLA w/ GRPO <sup>†</sup> [^57] NIPS’25 99.1 97.1 87.6 97.1 100.0 92.1(+11.6) Parallel IL+RL Transfuser w/ PaIR-Drive ours 99.1 96.1 88.1 98.2 93.1 89.7(+5.7) Transfuser w/ PaIR-Drive <sup>†</sup> ours 99.5 99.2 88.0 99.2 98.1 93.3(+9.3) DiffusionDrive w/ PaIR-Drive ours 99.1 97.6 88.3 98.5 94.1 91.2(+3.1) DiffusionDrive w/ PaIR-Drive <sup>†</sup> ours 99.6 99.5 88.1 99.5 98.6 94.0(+5.9)

Table 4: Comparison with SOTA methods on NAVSIMv2 benchmark. The best and second-best scores are highlighted in bold and underlined, respectively. The green score means the improvement compared with the origin IL policy. †: using the best-of-N (N=6) strategy following [^57].

<table><tbody><tr><td>Method type</td><td>Agent</td><td>source</td><td>NC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DAC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DDC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TLC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EP <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TTC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>LK <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>HC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EPDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td rowspan="5">IL</td><td>VADv2 <sup><a href="#fn:6">6</a></sup></td><td>ICCV’23</td><td>97.3</td><td>91.7</td><td>98.2</td><td>99.9</td><td>77.6</td><td>92.7</td><td>98.2</td><td>100.0</td><td>97.4</td><td>76.6</td></tr><tr><td>Transfuser w/o RL <sup><a href="#fn:8">8</a></sup></td><td>TPAMI’23</td><td>97.2</td><td>91.8</td><td>99.2</td><td>99.8</td><td>87.6</td><td>95.7</td><td>95.7</td><td>98.4</td><td>87.7</td><td>79.7</td></tr><tr><td>ARTEMIS <sup><a href="#fn:13">13</a></sup></td><td>arxiv’25</td><td>98.3</td><td>95.1</td><td>98.6</td><td>99.8</td><td>81.5</td><td>97.4</td><td>96.5</td><td>100.0</td><td>98.3</td><td>83.1</td></tr><tr><td>WOTE <sup><a href="#fn:33">33</a></sup></td><td>ICCV’25</td><td>98.5</td><td>96.8</td><td>98.8</td><td>99.8</td><td>86.1</td><td>97.9</td><td>95.5</td><td>98.3</td><td>82.9</td><td>84.2</td></tr><tr><td>DiffusionDrive <sup><a href="#fn:39">39</a></sup></td><td>CVPR’25</td><td>98.0</td><td>96.0</td><td>99.5</td><td>99.8</td><td>87.7</td><td>97.1</td><td>97.2</td><td>98.3</td><td>87.6</td><td>84.3</td></tr><tr><td rowspan="2">Sequential IL+RL</td><td>RecogDrive w/ GRPO <sup><a href="#fn:34">34</a></sup></td><td>arxiv’25</td><td>98.3</td><td>95.2</td><td>99.5</td><td>99.8</td><td>87.1</td><td>97.5</td><td>96.6</td><td>98.3</td><td>86.5</td><td>83.6</td></tr><tr><td>Transfuser w/ GRPO <sup><a href="#fn:8">8</a></sup></td><td>TPAMI’23</td><td>98.0</td><td>94.7</td><td>99.3</td><td>99.8</td><td>88.5</td><td>96.6</td><td>96.4</td><td>98.3</td><td>89.3</td><td>83.8(+4.1)</td></tr><tr><td rowspan="4">Parallel IL+RL</td><td>Transfuser w/ PaIR-Drive</td><td>ours</td><td>99.1</td><td>96.1</td><td>99.4</td><td>100.0</td><td>88.1</td><td>98.2</td><td>96.2</td><td>94.3</td><td>74.2</td><td>86.6(+6.9)</td></tr><tr><td>Transfuser w/ PaIR-Drive <sup>†</sup></td><td>ours</td><td>99.5</td><td>99.0</td><td>99.6</td><td>100.0</td><td>87.8</td><td>99.2</td><td>97.6</td><td>97.2</td><td>72.0</td><td>88.5(+8.8)</td></tr><tr><td>DiffusionDrive w/ PaIR-Drive</td><td>ours</td><td>99.1</td><td>97.6</td><td>99.5</td><td>100.0</td><td>88.3</td><td>98.5</td><td>96.9</td><td>94.8</td><td>74.0</td><td>87.9(+3.6)</td></tr><tr><td>DiffusionDrive w/ PaIR-Drive <sup>†</sup></td><td>ours</td><td>99.6</td><td>99.5</td><td>99.7</td><td>100.0</td><td>88.1</td><td>99.5</td><td>98.3</td><td>97.7</td><td>76.4</td><td>89.6(+5.3)</td></tr></tbody></table>

### 4.2 Main Results

This section reports the main results of the PaIR-Drive for answering the following questions.

Can PaIR-Drive correct suboptimal human behaviors? The answer is affirmative. As shown in Tab. 1, PaIR-Drive improves upon suboptimal human demonstrations. Using human experts as the base IL policy, PaIR-Drive achieves consistent gains over human performance, with +1.6 and +0.8 PDMS improvements on the Human bad v1 and Navtest splits, respectively. Similarly, as presented in Tab. 2, PaIR-Drive substantially corrects human suboptimal behaviors in metrics such as DDC, TLC, and LK, achieving +10.8 and +1.6 EPDMS gains on the Human bad v2 and Navtest splits, which demonstrates the exploratory capability.

Does PaIR-Drive need retraining when applied to a new IL policy? The answer is negative. We directly apply PaIR-Drive to different IL policies, including DiffusionDrive and Transfuser, and compare it with competitive pure IL and sequential IL+RL methods. As shown in the PDMS results of Tab. 3, PaIR-Drive improves Transfuser by +5.7 PDMS. PaIR-Drive improves DiffusionDrive by +3.1 PDMS, and achieving a competitive performance of 91.2 PDMS. If we use the best-of-N (N=6) strategy, the performance can reach 94.0 PDMS. Consistent conclusions can be reached in the EPDMS results, as shown in Tab. 4. PaIR-Drive improves over IL policies, achieving +6.9 and +3.6 EPDMS gains on Transfuser and DiffusionDrive, respectively, and surpassing all other methods across most metrics. Interestingly, the performance enhancement of PaIR-Drive tends to diminish with stronger IL policies, as a well-performed IL policy may leave less room for reinforcement refinement and exploration.

![[RLtypes_0314.png|Refer to caption]]

Figure 6: Comparison between our parallel scheme and conventional sequential scheme. Left is the PDMS improvement results of our PaIR-Drive, and right is the results of conventional sequential-RL-based methods.

### 4.3 Ablation studies

This section reports the ablation studies on our PaIR-Drive and some interesting findings.

The parallel IL+RL framework matters. Based on the same Transfuser baseline, PaIR-Drive (Fig. 6 left) reaches 89.7 PDMS, outperforming the sequential counterpart (87.9) in Fig. 6 right. Moreover, with the integration of PaIR-Drive, Transfuser achieves superior performance to stronger IL baselines such as RecogDrive (86.8 PDMS) and DriveDPO (88.8 PDMS), which originally outperform Transfuser under pure IL training. Furthermore, although DiffusionDrive lags behind DriveDPO in the IL-only setting, it surpasses DriveDPO once enhanced by PaIR-Drive, achieving 91.2 PDMS.

Table 5: Ablation studies. The baseline IL policy is Transfuser. offset pred. - predict the trajectory offsets. Traj pred. - generate the whole trajectory directly. All results follow the best-of-N (N=6) strategy [^57].

(a) Ablation on tree-structured sampling.

| ID | Tree-structured | PDMS $\uparrow$ | EPDMS $\uparrow$ |
| --- | --- | --- | --- |
| 1 | ✗ offset pred. | 88.8 | 81.6 |
| 2 | ✗ traj pred. | 87.9 | 83.8 |
| 3 | ✓ offset pred. | 93.3 | 88.5 |

(b) Ablation on GRPO group number.

| ID | Group number | PDMS $\uparrow$ | EPDMS $\uparrow$ |
| --- | --- | --- | --- |
| 1 | 5 | 89.1 | 80.6 |
| 2 | 9 | 89.3 | 81.4 |
| 3 | 12 | 93.3 | 86.9 |
| 4 | 15 | 93.3 | 88.5 |

![[visualization_0314.png|Refer to caption]]

Figure 7: Visualization analysis. We compare PaIR-Drive with DiffusionDrive and Transfuser w/ GRPO. Our PaIR-Drive shows (a) better collision avoidance, (b) more compliant with drivable area.

Sampling trajectories with a tree-structured design proves beneficial. As shown in the last three rows of Tab. 5 (a), the tree-structured design (ID: 3) yields clear gains over non-structured variants (ID: 1 and 2). It improves PDMS from 88.8/87.9 to 93.3 and EPDMS from 81.6/83.8 to 88.5, highlighting its critical role in stabilizing and enhancing trajectory learning. Interestingly, we found that without the tree structure, trajectory prediction (ID: 2) outperforms offset prediction (ID: 1) in EPDMS but not in PDMS. This is primarily because generating a continuous trajectory ensures driving comfort, which is captured in EPDMS.

More GRPO group number of trajectories helps. Tab. 5 (b) shows that increasing the GRPO group number leads to consistent performance gains. As the group number grows from 5 to 15, PDMS improves from 89.1 to 93.3, and EPDMS rises from 80.6 to 88.5. This demonstrates that due to our tree-structure design, larger group size leads to richer trajectory diversity, enabling more effective policy optimization under the parallel framework.

RWM is not the decisive factor. As shown in Tab. 6, directly applying IL + RWM yields only limited gains (+2.7 EPDMS), whereas our PaIR-Drive + RWM achieves substantially better performance (+5.3 EPDMS). It demonstrates that the improvement does not come from RWM re-ranking alone.

Table 6: Ablation on the dependance of RWM. The pretrained IL policy is Diffusiondrive.

ID Agent PDMS $\uparrow$ EPDMS $\uparrow$ 1 Vanilla IL 88.1 84.3 2 IL + RWM 90.2 87.0 3 PaIR-Drive + RWM 94.0 89.6

### 4.4 Visualization analysis

This section reports three representative cases to validate the advantages of our PaIR-Drive.

Case (a): Better avoiding collision. As shown in Fig. 7 (a), the front vehicle brakes heavily. The human expert and the Transfuser w/ GRPO both brake in time to ensure safety. However, the DiffusionDrive fails to decelerate and collides with the front vehicle. Interestingly, our PaIR-Drive learns to change lane proactively, balancing collision avoidance with improved driving mobility. The result confirms the advantage of the multi-intentions trajectory expansion of our PaIR-Drive.

Case (b): More compliant with drivable area. As shown in Fig. 7 (b), the human driver successfully pass through the roundabout. DiffusionDrive and Transfuser w/GRPO show unstable trajectories, drifting slightly during the entry phase. In contrast, PaIR-Drive follows a clean and well-aligned arc into the correct lane, maintaining stability and consistency throughout the maneuver. This case confirms the advantage of our multi-intention trajectory expansion, enabling PaIR-Drive to better reason over complex geometric structures like roundabouts.

## 5 Conclusion

In this paper, we introduce PaIR-Drive, a general Parallel framework for collaborative Imitation and Reinforcement learning in end-to-end autonomous driving. By decoupling IL and RL into parallel optimization branches, PaIR-Drive leverages IL to learn human-level driving behavior and RL to explore how to surpass human expert performance. PaIR-Drive could serve as a general performance enhancement toolkit and be seamlessly integrated into any IL-based autonomous driving method without any retraining. A tree-structured trajectory neural sampler is introduced to GRPO in the RL branch, which further enhances exploration efficiency and improves trajectory quality. Our PaIR-Drive achieves competitive performance on NAVSIM v1 and v2 benchmarks. Visualization analysis showcases the exploration and high-quality trajectory generation capabilities of PaIR-Drive. Overall, our key insight is breaking the upper performance limit of sequential fine-tuning through our innovative parallel framework. We hope our work could inspire further research in different IL+RL schemes of end-to-end autonomous driving.

## 6 Acknowledgment

This paper is partially supported by National Natural Science Foundation of China (Grant No. 52372317 and 52302412), Yangtze River Delta Science and Technology Innovation Joint Force (No. YDZX20233100004027), Shanghai Automotive Industry Science and Technology Development Foundation (No. 2404), Xiaomi Young Talents Program, the Fundamental Research Funds for the Central Universities (22120230311), and Tongji Zhongte Chair Professor Foundation (No. 000000375-2018082).

[^1]: L. Ankile, A. Simeonov, I. Shenfeld, M. Torne, and P. Agrawal (2025) From imitation to refinement-residual rl for precise assembly. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pp. 01–08. Cited by: §1.

[^2]: P. J. Ball, L. Smith, I. Kostrikov, and S. Levine (2023) Efficient online reinforcement learning with offline data. In International Conference on Machine Learning, pp. 1577–1594. Cited by: §1.

[^3]: A. Bhaskar, Z. Mahammad, S. R. Jadhav, and P. Tokekar (2024) Planrl: a motion planning and imitation learning framework to bootstrap reinforcement learning. arXiv preprint arXiv:2408.04054. Cited by: §1, §2.

[^4]: W. Cao, M. Hallgarten, T. Li, D. Dauner, X. Gu, C. Wang, Y. Miron, M. Aiello, H. Li, I. Gilitschenski, et al. (2025) Pseudo-simulation for autonomous driving. arXiv preprint arXiv:2506.04218. Cited by: §4.1, §4.1.

[^5]: M. Chen, L. Sun, T. Li, H. Sun, Y. Zhou, C. Zhu, H. Wang, J. Z. Pan, W. Zhang, H. Chen, F. Yang, Z. Zhou, and W. Chen (2025) ReSearch: learning to reason with search for llms via reinforcement learning. arXiv preprint arXiv:2503.19470. External Links: 2503.19470, [Link](https://arxiv.org/abs/2503.19470) Cited by: §1.

[^6]: S. Chen, B. Jiang, H. Gao, B. Liao, Q. Xu, Q. Zhang, C. Huang, W. Liu, and X. Wang (2024) Vadv2: end-to-end vectorized autonomous driving via probabilistic planning. arXiv preprint arXiv:2402.13243. Cited by: §2, Table 3, Table 4.

[^7]: C. Cheng, X. Yan, N. Wagener, and B. Boots (2018) Fast policy learning through imitation and reinforcement. arXiv preprint arXiv:1805.10413. Cited by: §1.

[^8]: K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger (2023) TransFuser: imitation with transformer-based sensor fusion for autonomous driving. IEEE Transactions on Pattern Analysis and Machine Intelligence 45 (11), pp. 12878–12895. External Links: [Document](https://dx.doi.org/10.1109/TPAMI.2022.3200245) Cited by: §3.1, Table 3, Table 3, Table 4, Table 4.

[^9]: D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, et al. (2024) Navsim: data-driven non-reactive autonomous vehicle simulation and benchmarking. Advances in Neural Information Processing Systems 37, pp. 28706–28719. Cited by: §4.1, §4.1.

[^10]: DeepSeek-AI and D. Guo (2025) DeepSeek-r1: incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948. External Links: 2501.12948, [Link](https://arxiv.org/abs/2501.12948) Cited by: §1.

[^11]: Y. Deng, H. Bansal, F. Yin, N. Peng, W. Wang, and K. Chang (2025) Openvlthinker: complex vision-language reasoning via iterative sft-rl cycles. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, Cited by: §1, §1.

[^12]: P. Dong, A. M. Lessing, A. S. Chen, and C. Finn (2025) Reinforcement learning via implicit imitation guidance. arXiv preprint arXiv:2506.07505. Cited by: §1.

[^13]: R. Feng, N. Xi, D. Chu, R. Wang, Z. Deng, A. Wang, L. Lu, J. Wang, and Y. Huang (2025) Artemis: autoregressive end-to-end trajectory planning with mixture of experts for autonomous driving. arXiv preprint arXiv:2504.19580. Cited by: Table 3, Table 4.

[^14]: P. FINE (2025) IN–ril: interleaved reinforcement and imita-tion learning for policy fine-tuning. openreview.net. Cited by: §1, §1.

[^15]: H. Fu, D. Zhang, Z. Zhao, J. Cui, D. Liang, C. Zhang, D. Zhang, H. Xie, B. Wang, and X. Bai (2025) Orion: a holistic end-to-end autonomous driving framework by vision-language instructed action generation. arXiv preprint arXiv:2503.19755. Cited by: §2.

[^16]: H. Gao, S. Chen, B. Jiang, B. Liao, Y. Shi, X. Guo, Y. Pu, H. Yin, X. Li, X. Zhang, Y. Zhang, W. Liu, Q. Zhang, and X. Wang (2025) RAD: training an end-to-end driving policy via large-scale 3dgs-based reinforcement learning. arXiv preprint arXiv:2502.13144. External Links: 2502.13144 Cited by: §1, §2.

[^17]: Y. Guan, H. Liao, Z. Li, J. Hu, R. Yuan, G. Zhang, and C. Xu (2024) World models for autonomous driving: an initial survey. IEEE Transactions on Intelligent Vehicles. Cited by: §2.

[^18]: D. Guo, D. Yang, and H. Zhang (2025) DeepSeek-r1 incentivizes reasoning in llms through reinforcement learning.. Nature 645, 633–638.. Cited by: §1.

[^19]: H. Hu, S. Mirchandani, and D. Sadigh (2024) Imitation bootstrapped reinforcement learning. arXiv. Cited by: §1, §1.

[^20]: Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, L. Lu, X. Jia, Q. Liu, J. Dai, Y. Qiao, and H. Li (2023) Planning-oriented autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, Cited by: §1, §2.

[^21]: Z. Huang, J. Wu, and C. Lv (2022) Efficient deep reinforcement learning with imitative expert priors for autonomous driving. IEEE Transactions on Neural Networks and Learning Systems 34 (10), pp. 7391–7403. Cited by: §1.

[^22]: A. Jiang, Y. Gao, Z. Sun, Y. Wang, J. Wang, J. Chai, Q. Cao, Y. Heng, H. Jiang, Y. Dong, et al. (2025) Diffvla: vision-language guided diffusion planning for autonomous driving. arXiv preprint arXiv:2505.19381. Cited by: §2.

[^23]: B. Jiang, S. Chen, Q. Xu, B. Liao, J. Chen, H. Zhou, Q. Zhang, W. Liu, C. Huang, and X. Wang (2023) VAD: vectorized scene representation for efficient autonomous driving. ICCV. Cited by: §1, §2.

[^24]: B. Jiang, S. Chen, Q. Zhang, W. Liu, and X. Wang (2025) Alphadrive: unleashing the power of vlms in autonomous driving via reinforcement learning and reasoning. arXiv preprint arXiv:2503.07608. Cited by: §2.

[^25]: B. Lee, R. Hachiuma, Y. M. Ro, Y. F. Wang, and Y. Wu (2025) Unified reinforcement and imitation learning for vision-language models. arXiv e-prints, pp. arXiv–2510. Cited by: §1.

[^26]: F. Leiva and J. Ruiz-del-Solar (2024) Combining rl and il using a dynamic, performance-based modulation over learning signals and its application to local planning. arXiv preprint arXiv:2405.09760. (en). Cited by: §1.

[^27]: G. Li, Y. Cao, Q. Chen, X. Gao, Y. Yang, and J. Pu (2025) Papl-slam: principal axis-anchored monocular point-line slam. IEEE Robotics and Automation Letters. Cited by: §2.

[^28]: G. Li, K. Ren, L. Xu, Z. Zheng, C. Jiang, X. Gao, B. Dai, J. Pu, M. Yu, and J. Pang (2026) ARTDECO: toward high-fidelity on-the-fly reconstruction with hierarchical gaussian structure and feed-forward guidance. In The Fourteenth International Conference on Learning Representations, Cited by: §2.

[^29]: H. Li, Y. Zuo, J. Yu, Y. Zhang, Z. Yang, K. Zhang, X. Zhu, Y. Zhang, T. Chen, G. Cui, et al. (2025) Simplevla-rl: scaling vla training via reinforcement learning. arXiv preprint arXiv:2509.09674. Cited by: §1.

[^30]: H. Li, T. Li, J. Yang, H. Tian, C. Wang, L. Shi, M. Shang, Z. Lin, G. Wu, Z. Hao, et al. (2026) PlannerRFT: reinforcing diffusion planners through closed-loop and sample-efficient fine-tuning. arXiv preprint arXiv:2601.12901. Cited by: §2.

[^31]: J. Li, J. Wu, D. Hu, X. Huang, B. Sun, Z. Hao, X. Lang, X. Zhu, and L. Zhang (2026) SGDrive: scene-to-goal hierarchical world cognition for autonomous driving. arXiv preprint arXiv:2601.05640. Cited by: §2.

[^32]: Y. Li, L. Fan, J. He, Y. Wang, Y. Chen, Z. Zhang, and T. Tan (2024) Enhancing end-to-end autonomous driving with latent world model. arXiv preprint arXiv:2406.08481. Cited by: §2.

[^33]: Y. Li, Y. Wang, Y. Liu, J. He, L. Fan, and Z. Zhang (2025) End-to-end driving with online trajectory evaluation via bev world model. arXiv preprint arXiv:2504.01941. Cited by: Table 3, Table 4.

[^34]: Y. Li, K. Xiong, X. Guo, F. Li, S. Yan, G. Xu, L. Zhou, L. Chen, H. Sun, B. Wang, K. Ma, G. Chen, H. Ye, W. Liu, and X. Wang (2025) ReCogDrive: a reinforced cognitive framework for end-to-end autonomous driving. arXiv preprint arXiv:2506.08052. External Links: 2506.08052, [Link](https://arxiv.org/abs/2506.08052) Cited by: §1, §3.2, Table 3, Table 3, Table 4.

[^35]: Y. Li, M. Tian, D. Zhu, J. Zhu, Z. Lin, Z. Xiong, and X. Zhao (2025) Drive-r1: bridging reasoning and planning in vlms for autonomous driving with reinforcement learning. arXiv preprint arXiv:2506.18234. External Links: 2506.18234, [Link](https://arxiv.org/abs/2506.18234) Cited by: §1.

[^36]: Z. Li, K. Li, S. Wang, S. Lan, Z. Yu, Y. Ji, Z. Li, Z. Zhu, J. Kautz, Z. Wu, et al. (2024) Hydra-mdp: end-to-end multimodal planning with multi-target hydra-distillation. arXiv preprint arXiv:2406.06978. Cited by: §2.

[^37]: Z. Li, Z. Yu, S. Lan, J. Li, J. Kautz, T. Lu, and J. M. Alvarez (2024-06) Is ego status all you need for open-loop end-to-end autonomous driving?. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 14864–14873. Cited by: §1.

[^38]: X. Liang, T. Wang, L. Yang, and E. Xing (2018) Cirl: controllable imitative reinforcement learning for vision-based self-driving. In Proceedings of the European conference on computer vision (ECCV), pp. 584–599. Cited by: §1.

[^39]: B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang, and X. Wang (2025) DiffusionDrive: truncated diffusion model for end-to-end autonomous driving. CVPR. Cited by: §1, §2, Table 3, Table 4.

[^40]: Q. Lin, F. Yang, and C. Zhu (2026) Harnessing the power of foundation models for accurate material classification. arXiv preprint arXiv:2603.17390. Cited by: §2.

[^41]: H. Liu, T. Li, H. Yang, L. Chen, C. Wang, K. Guo, H. Tian, H. Li, H. Li, and C. Lv (2026) Reinforced refinement with self-aware expansion for end-to-end autonomous driving. IEEE Transactions on Pattern Analysis and Machine Intelligence. Cited by: §2.

[^42]: X. Liu, T. Yoneda, R. L. Stevens, M. R. Walter, and Y. Chen (2023) Blending imitation and reinforcement learning for robust policy improvement. arXiv preprint arXiv:2310.01737. Cited by: §1, §1.

[^43]: Y. Lu, J. Fu, G. Tucker, X. Pan, E. Bronstein, R. Roelofs, B. Sapp, B. White, A. Faust, S. Whiteson, D. Anguelov, and S. Levine (2023) Imitation is not enough: robustifying imitation with reinforcement learning for challenging driving scenarios. In 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), Cited by: §1.

[^44]: L. Ma, H. Liang, M. Qiang, L. Tang, X. Ma, Z. H. Wong, J. Niu, C. Shen, R. He, Y. Li, et al. (2025) Learning what reinforcement learning can’t: interleaved online fine-tuning for hardest questions. arXiv preprint arXiv:2506.07527. Cited by: §1, §1.

[^45]: S. Shang, Y. Chen, Y. Wang, Y. Li, and Z. Zhang (2025) DriveDPO: policy learning via safety dpo for end-to-end autonomous driving. arXiv preprint arXiv:2509.17940. Cited by: Table 3, Table 3.

[^46]: H. Song, D. Qu, Y. Yao, Q. Chen, Q. Lv, Y. Tang, M. Shi, G. Ren, M. Yao, B. Zhao, et al. (2025) Hume: introducing system-2 thinking in visual-language-action model. arXiv preprint arXiv:2505.21432. Cited by: §2.

[^47]: X. Tang, M. Kan, S. Shan, and X. Chen (2025) Plan-r1: safe and feasible trajectory planning as language modeling. arXiv preprint arXiv:2505.17659. External Links: 2505.17659, [Link](https://arxiv.org/abs/2505.17659) Cited by: §1.

[^48]: L. Wang, Ö. Ş. Taş, M. Steiner, and C. Stiller (2025) FlowDrive: moderated flow matching with data balancing for trajectory planning. arXiv preprint arXiv:2509.21961. Cited by: §2.

[^49]: W. Weng, T. Wu, L. Chen, S. Xie, Z. Wang, X. Xu, J. Song, and H. T. Shen (2026) Language-grounded decoupled action representation for robotic manipulation. External Links: 2603.12967, [Link](https://arxiv.org/abs/2603.12967) Cited by: §2.

[^50]: P. Wu, P. Zhang, Z. Wang, D. Wang, B. Zhao, and X. Li (2026) Closed-loop action chunks with dynamic corrections for training-free diffusion policy. arXiv preprint arXiv:2603.01953. Cited by: §2.

[^51]: Z. Xing, X. Zhang, Y. Hu, B. Jiang, T. He, Q. Zhang, X. Long, and W. Yin (2025) Goalflow: goal-driven flow matching for multimodal trajectories generation in end-to-end autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 1602–1611. Cited by: §2.

[^52]: G. Xudong, F. Dawei, K. Xu, Y. Zhai, C. Yao, W. Wang, B. Ding, and H. Wang (2024) Iterative regularized policy optimization with imperfect demonstrations. In Forty-first International Conference on Machine Learning, Cited by: §1.

[^53]: D. Zhang, J. Liang, K. Guo, S. Lu, Q. Wang, R. Xiong, Z. Miao, and Y. Wang (2025) Carplanner: consistent auto-regressive trajectory planning for large-scale reinforcement learning in autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 17239–17248. Cited by: §2.

[^54]: P. Zhang, Y. Su, P. Wu, D. An, L. Zhang, Z. Wang, D. Wang, Y. Ding, B. Zhao, and X. Li (2025) Cross from left to right brain: adaptive text dreamer for vision-and-language navigation. arXiv preprint arXiv:2505.20897. Cited by: §2.

[^55]: Y. Zheng, R. Liang, K. ZHENG, J. Zheng, L. Mao, J. Li, W. Gu, R. Ai, S. E. Li, X. Zhan, et al. (2025) Diffusion-based planning for autonomous driving with flexible guidance. In The Thirteenth International Conference on Learning Representations, Cited by: §2.

[^56]: Y. Zheng, P. Yang, Z. Xing, Q. Zhang, Y. Zheng, Y. Gao, P. Li, T. Zhang, Z. Xia, P. Jia, et al. (2025) World4Drive: end-to-end autonomous driving via intention-aware physical latent world model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 28632–28642. Cited by: §2.

[^57]: Z. Zhou, T. Cai, S. Z. Zhao, Y. Zhang, Z. Huang, B. Zhou, and J. Ma (2025) AutoVLA: a vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. arXiv preprint arXiv:2506.13757. External Links: 2506.13757, [Link](https://arxiv.org/abs/2506.13757) Cited by: §1, §3.2, Table 3, Table 3, Table 3, Table 3, Table 4, Table 4, Table 5, Table 5.
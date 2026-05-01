---
title: "HAD: Combining Hierarchical Diffusion with Metric-Decoupled RL for End-to-End Driving"
source: "https://arxiv.org/html/2604.03581v1"
author:
published:
created: 2026-05-01
description:
tags:
  - "clippings"
---
Wenhao Yao <sup>1</sup>    Xinglong Sun <sup>2</sup>    Zhenxin Li <sup>1</sup>    Shiyi Lan <sup>2</sup>    Zi Wang <sup>2</sup>   Jose M. Alvarez <sup>2</sup>    Zuxuan Wu <sup>1,*</sup> \[ \[ [whyao23@m.fudan.edu.cn](https://arxiv.org/html/2604.03581v1/mailto:whyao23@m.fudan.edu.cn)

###### Abstract

End-to-end planning has emerged as a dominant paradigm for autonomous driving, where recent models often adopt a scoring-selection framework to choose trajectories from a large set of candidates, with diffusion-based decoding showing strong promise. However, directly selecting from the entire candidate space remains difficult to optimize, and Gaussian perturbations used in diffusion often introduce unrealistic trajectories that complicate the denoising process. In addition, for training these models, reinforcement learning (RL) has shown promise, but existing end-to-end RL approaches typically rely on a single coupled reward without structured signals, limiting optimization effectiveness. To address these challenges, we propose HAD, an end-to-end planning framework with a Hierarchical Diffusion Policy that decomposes planning into a coarse-to-fine process. To improve trajectory generation, we introduce Structure-Preserved Trajectory Expansion, which produces realistic candidates while maintaining kinematic structure. For policy learning, we develop Metric-Decoupled Policy Optimization (MDPO) to enable structured RL optimization across multiple driving objectives. Extensive experiments show that HAD achieves new state-of-the-art performance on both NAVSIM and HUGSIM, outperforming prior arts by a huge margin: +2.3 EPDMS on NAVSIM and +4.9 Route Completion on HUGSIM.

<sup>†</sup>

## 1 Introduction

Planning is a pivotal component in the autonomous driving pipeline. Recent advances have led to a paradigm shift toward end-to-end planning \[hu2023planning, chen2024vadv2, li2024hydra, liao2024diffusiondrive, zou2025diffusiondrivev2\], where perception, prediction, and planning are integrated into a unified model that directly outputs the driving trajectory from raw sensor inputs. By eliminating intermediate hand-crafted modules, such approaches mitigate error accumulation and enable more holistic reasoning over the driving scene \[hu2023planning\].

To enhance end-to-end planner performance, several works \[chen2024vadv2, li2024hydra, yao2025drivesuprim, li2025ztrs, li2025generalized, liao2024diffusiondrive\] adopt a scoring-selection paradigm, where a policy ranks candidate trajectories from discretized action space represented by a fixed vocabulary of anchors (e.g. 8192) and selects the best one. However, performance is fundamentally limited by the coverage and quality of the anchor set, causing failures in out-of-vocabulary scenarios \[liao2024diffusiondrive\]. To increase behavioral diversity, recent works \[liao2024diffusiondrive, zou2025diffusiondrivev2, li2026plannerrft, gao2025diffvla++, jiang2025diffvla, li2025recogdrive, liu2025bridgedrive\] explore generation with anchored-diffusion, which samples additional trajectories with Gaussian noise perturbations around the trajectory anchors.

Despite these advances, several limitations remain. First, as shown in Fig. 1 (a), the candidate action space is extremely large, making it a difficult optimization problem for a policy model to directly learn to rank and select from the entire set in a single pass \[yao2025drivesuprim\]. Second, naive Gaussian noise sampling is not well suited for driving trajectories. Even in anchored diffusion setups \[liao2024diffusiondrive\], Gaussian perturbations often generate unrealistic motions, such as oscillatory or zigzag trajectories. These low-quality candidates introduce substantial noise into the optimization process and make the denoising stage more challenging.

![[x1 29.png|Refer to caption]]

Figure 1: Comparison with existing end-to-end planning methods. Prior approaches search the entire driving space and use a single coupled reward from online simulation. Our method narrows the search via Hierarchical Diffusion Policy and approximates metric-decoupled rewards through offline retrieval.

Moreover, to train these end-to-end policy models, imitation learning \[hu2023planning, chen2024vadv2, liao2024diffusiondrive, chitta2022transfuser\] has been the predominant paradigm. However, recent work has shown that imitation learning is fundamentally limited by the quality of human demonstrations \[li2024hydra, li2025ztrs\]. Therefore, some works \[zou2025diffusiondrivev2, li2025ztrs, li2026plannerrft\] try to incorporate reinforcement learning (RL), where the policy learns to predict trajectories that maximize a reward signal reflecting driving quality, independent of human demonstrations.

Nevertheless, two challenges remain for RL in end-to-end driving. First, existing methods typically rely on a single coupled reward, such as the aggregated PDM score \[dauner2024navsim\]. However, driving requires balancing multiple criteria—e.g., collision avoidance, lane keeping, etc—while a single scalar reward is often too coarse and can lead to reward hacking. Recent RL advances in the LLM domain, such as GDPO \[liu2026gdpo\], explore multiple rewards but still aggregate them into a single objective. Second, reward computation must be efficient in practice. Existing approaches \[zou2025diffusiondrivev2, li2026plannerrft\] compute rewards on-the-fly via simulation during training, introducing huge computational overhead and reducing training efficiency.

To address the aforementioned limitations, we propose HAD. To reduce the difficulty of directly decoding a trajectory from the entire action space, we introduce a Hierarchical Diffusion Policy that follows the coarse-to-fine reasoning process of human driving. The policy consists of two diffusion-based denoising stages, as shown in Fig. 1 (b). The first stage identifies a small set of plausible high-level driving intentions represented by trajectory anchors. The second stage then refines these candidates by performing low-level trajectory denoising, effectively “zooming in” on the plausible intentions to produce the final driving maneuver. To generate high-quality trajectory candidates around these intention anchors for low-level trajectory refinement, we further propose Structure-Preserved Trajectory Expansion, which maintains the intrinsic kinematic structure of driving trajectories. Instead of standard Gaussian noise sampling, which often disrupts trajectory structure, we project trajectory anchors from Cartesian space into polar coordinates and apply carefully designed radial scaling and angular offset perturbations. Finally, to better capture the multi-dimensional nature of driving objectives, we introduce Metric-Decoupled Policy Optimization (MDPO), an RL optimization framework that explicitly decouples training into multiple heads, each optimizing a specific driving metric. This design enables more structured and targeted optimization across different aspects of driving behavior. To further improve training efficiency, we also introduce a Trajectory Reward Retrieval mechanism that enables near-instant reward computation via pre-computation and caching of rewards on trajectory anchors combined with training-time nearest-neighbor matching, as shown in Fig. 1 (d).

We comprehensively evaluate HAD across multiple benchmarks. On the widely used NAVSIM \[dauner2024navsim\] benchmark, HAD achieves 90.2 PDMS on NAVSIM v1 and 88.6 EPDMS on NAVSIM v2, outperforming the previous state-of-the-art method \[jiao2025evadrive\] by 2.3 points. In addition, HAD demonstrates strong closed-loop planning capability on the HUGSIM \[zhou2024hugsim\] benchmark, achieving a 47.5 Route Completion rate and a 30.8 HD-Score, surpassing prior art \[li2025ztrs\] by 4.9 points in Route Completion and 1.9 points in HD-Score.

Our contribution can be summarized as follows:

- We propose Hierarchical Diffusion Policy that decomposes planning into two stages: high-level intention establishment and low-level trajectory refinement, which reduces the optimization difficulty of trajectory selection from large action space.
- We propose Structure-Preserved Trajectory Expansion to sample high-quality trajectory candidates, maintaining kinematic structure for denoising.
- We develop Metric-Decoupled Policy Optimization (MDPO) to enable fine-grained RL training balancing multiple driving objectives, with an Offline Trajectory Reward Retrieval Scheme to efficiently acquire reward via precomputed anchor rewards and nearest-neighbor retrieval during training.
- HAD achieves achieves state-of-the-art performance on NAVSIM and HUGSIM benchmarks.

## 2 Related Works

### 2.1 End-to-end Planning

End-to-end planning methods unify the autonomous driving pipeline from perception to planning into a single network, directly processing raw sensor inputs to output the final driving trajectory. Early approaches \[hu2023planning, jiang2023vad, shao2023reasonnet, weng2024para, sun2025sparsedrive\] apply imitation learning. UniAD \[hu2023planning\] was the first work to adopt the planning-oriented pipeline to center all other driving tasks around planning. Some methods \[jiang2023vad, sun2025sparsedrive, jia2025drivetransformer\] discover sparse scene representation to extract more valuable features for planning. However, the optimization objective of imitation learning is singular, making models risky of causal confusion problem. To resolve the limitation, selection-based methods \[philion2020lift, phan-minh2020covernet, chen2024vadv2, li2024hydra, wang2025enhancing, sima2025centaur, li2025hydranext, yao2025drivesuprim\] introduce a predefined trajectory vocabulary, and further use multiple metrics to score each trajectory, selecting the most suitable candidate as the model output. Generative approaches \[zheng2024genad, jiang2023motiondiffuser, zheng2025diffusionbased, xing2025goalflow, liao2024diffusiondrive\] utilize VAE \[kingma2014auto\] or Diffusion Policy \[chi2023diffusionpolicy\] to model dynamic trajectory distribution to reach multi-modal planning. Recently, to resolve some hard corner cases in driving, some other methods \[shao2024lmdrive, li2025recogdrive, hegde2025distilling\] utilize or distill the general VLM \[touvron2023llama, bai2025qwen3vl\] knowledge to build robust and instruction-followed planners. The above methods directly predict trajectories from the whole unstructured, large driving space. Different from these one-stage approaches, our hierarchical modeling pipeline follows coarse-to-fine human driving logic and further eases the causal fusion problem.

### 2.2 Reinforcement Learning for Planning

Several approaches \[toromanoff2020end, jia2023driveadapter, gao2025rad, li2025endtoend, yang2025raw2drive, jiao2025evadrive, zou2025diffusiondrivev2, li2026plannerrft\] leverage reinforcement learning (RL) in planning to better align trajectory outputs with pre-defined driving rules. MaRLn \[toromanoff2020end\] discovers the potential of RL on planning, but suffers from low training efficiency. RAD \[gao2025rad\] establishes a virtual interactive driving environment based on 3DGS to adopt RL training. Some methods like DriveAdapter \[jia2023driveadapter\] and Raw2Drive \[yang2025raw2drive\] focus on narrowing the gap between raw sensor data and privileged data required for RL training. Other works \[jiao2025evadrive, zou2025diffusiondrivev2, li2026plannerrft\] try to integrate RL training with the superior diffusion-based method. EvaDrive \[jiao2025evadrive\] introduces a multi-round adversarial RL paradigm to train trajectory generation and scoring models for given scenarios. DiffusionDriveV2 \[zou2025diffusiondrivev2\] proposes Intra-Anchor GRPO and Inter-Anchor GRPO to improve RL for trajectory anchors. These methods require closed-loop driving simulators or reconstructed environments to get a single ensembled trajectory reward, which is computationally expensive and causes policy fusion. Our Trajectory Reward Retrieval Scheme efficiently gets the reward during training, and the Metric-Decoupled Policy Optimization treats the reward as a multi-dimensional vector, improving the policy granularity.

### 2.3 Hierarchical Modeling

Hierarchical modeling is beneficial to progressively refine predictions from coarse to fine, enabling models to handle complex tasks more effectively while improving robustness and stability. It has been widely adopted in various domains of computer vision, like object detection \[ren2015fasterrcnn, cai2018cascade, zhu2021deformable, zhang2023dino\_detr\], segmentation \[shi2024part2object, li2022deep\], and image generation \[gafni2022makeascene, ramesh2022hierarchical, razavi2019generating, tian2024visual\]. A few methods \[yao2025drivesuprim, wang2025cognitive, yin2025diffrefiner\] employ hierarchical modeling in end-to-end planning. However, they either employ hierarchical modeling in a fixed trajectory set \[yao2025drivesuprim\] or ignore modeling detailed exploration in the local region \[wang2025cognitive, yin2025diffrefiner\], resulting in low-quality output trajectories. Our proposed Hierarchical Diffusion Policy explores the local region well through the Structure-preserved Trajectory Expansion Algorithm, leading to accurate output planning trajectories.

## 3 Methods

In this section, we present the proposed method HAD. We first introduce the Hierarchical Diffusion Policy with the Structure-Preserved Trajectory Expansion algorithm. Next, we present Metric-Decoupled Policy Optimization (MDPO), a reinforcement learning algorithm designed for end-to-end driving. Finally, we introduce an efficient reward retrieval scheme to improve training efficiency.

### 3.1 Hierarchical Diffusion Strategy

![[x2 27.png|Refer to caption]]

Figure 2: Overview of HAD. The Hierarchical Diffusion Policy decomposes planning into Driving Intention Establishment and Local Trajectory Refinement. MDPO provides decoupled, structured optimization signals for training.

HAD introduces a novel hierarchical diffusion strategy, which treats the trajectory denoising process in a coarse-to-fine manner, decoupling trajectory planning into high-level Driving Intention Establishment and low-level Local Trajectory Refinement.

The framework is illustrated in Fig. 2. First, the Driving Intention Decoder establishes high-level driving intentions by selecting a small set of plausible trajectory anchors. These anchors are then expanded using the Structure-Preserved Trajectory Expansion algorithm to generate diverse local maneuvers around each intention while preserving the intrinsic kinematic structure of trajectories. Second, the Trajectory Refinement Decoder performs fine-grained adjustments within this local region to produce the final trajectory.

Driving Intention Establishment   This stage pre-defines a sparse set of $M_{1}$ trajectory anchors $\{\tau_{j}\}_{j=1}^{M_{1}}$ to roughly cover different driving intentions. During training, these anchors apply a diffusion forward process:

$$
\tilde{\tau}_{j}^{(i)}=\sqrt{\bar{\alpha}^{(i)}}\tau_{j}+\sqrt{1-\bar{\alpha}^{(i)}}\boldsymbol{\epsilon},\quad\boldsymbol{\epsilon}\sim\mathcal{N}(0,\mathbf{I})
$$

where $\tau_{j}=\left\{(x_{j,t},y_{j,t})\right\}_{t=1}^{T}$ denotes the $j$ -th trajectory including $T$ coordinates, $\bar{\alpha}^{(i)}$ represents the cumulative noise schedule at diffusion timestep $i\in[1,T_{\mathrm{trunc}}]$, and $\boldsymbol{\epsilon}$ is the gaussian noise. HAD leverages a Transformer-based decoder \[vaswani2017attention, carion2020detr\] to implement trajectory denoising. The encoded perturbed trajectories interact with the environment feature to produce refined trajectories and their corresponding confidence scores:

$$
\displaystyle cond=\mathrm{Enc_{env}}(Img,Lidar,Status)
$$
 
$$
\displaystyle\left\{f_{j}\right\}_{j=1}^{M_{1}}=\mathrm{Enc_{traj}}\left(\left\{\tilde{\tau}_{j}\right\}_{j=1}^{M_{1}}\right)
$$
 
$$
\displaystyle\left\{(\hat{\tau}_{\mathrm{gbl},j},\hat{s}_{\mathrm{gbl},j}^{(c)})\right\}_{j=1}^{M_{1}}=\mathrm{Dec_{gbl}}\left(\left\{f_{j}\right\}_{j=1}^{M_{1}},cond\right)
$$

where $\mathrm{Enc_{env}}$ is a multi-modal environment encoder identical to the Transfuser \[chitta2022transfuser\] backbone, fusing features from images $Img$, LiDAR $Lidar$ and ego-vehicle status $Status$ to extract environmental feature $cond$, $f_{j}$ is the $j$ -th trajectory query encoded by trajectory MLP encoder $\mathrm{Enc_{traj}}$, and $\mathrm{Dec_{gbl}}$ is the Driving Intention Decoder. The decoder yields two critical outputs: the denoised trajectory $\hat{\tau}_{\mathrm{gbl},j}$ and the classification score $\hat{s}_{\mathrm{gbl},j}^{(c)}$. The score reveals whether the denoised trajectory is in the same driving sub-region as the human trajectory. We select candidates with top- $K$ classification score as the filtered trajectory set $\mathcal{T}_{\mathrm{gbl}}=\{\hat{\tau}_{\mathrm{gbl},j}\mid j\in\mathcal{I}_{\text{topK}}\}$, where $\mathcal{I}_{\text{topK}}=\operatorname{argsort}_{K}(\{\hat{s}_{\mathrm{gbl},j}^{(c)}\}_{j=1}^{M_{1}})$, to represent driving intention.

Structure-Preserved Trajectory Expansion   To ensure a comprehensive exploration of the local driving region around the plausible intentions, HAD incorporates a Structure-Preserved Trajectory Expansion algorithm. This approach enriches trajectory diversity while strictly maintaining the intrinsic kinematic trajectory structure, easing the risk of trajectory structure damage caused by naive Gaussian noise sampling \[liao2024diffusiondrive, zou2025diffusiondrivev2, li2026plannerrft, chi2023diffusionpolicy\]. Given a trajectory $\tau=\{(x_{t},y_{t})\}_{t=1}^{T}$ and an expansion number $n_{\mathrm{exp}}$, this algorithm outputs an expanded trajectory set $\mathcal{T}_{\mathrm{exp}}=\{\tau_{j}\}_{j=1}^{n_{\mathrm{exp}}\cdot n_{\mathrm{exp}}}$. Specifically, the algorithm first projects the Cartesian coordinates of $\tau$ into the polar domain, represented as $\tau=\{(\rho_{t},\theta_{t})\}_{t=1}^{T}$. Moreover, we define a set of radial scaling coefficients $\{\lambda_{u}\}_{u=1}^{n_{\mathrm{exp}}}$ and angular offset coefficients $\{\delta_{v}\}_{v=1}^{n_{\mathrm{exp}}}$. The algorithm traverses these coefficients to apply linear transformations in the polar space, resulting in the expanded trajectories:

$$
\displaystyle\rho_{t}^{(u,v)}=\lambda_{u}\cdot\rho_{t},\quad\theta_{t}^{(u,v)}=\theta_{t}+\delta_{v}
$$
 
$$
\displaystyle x_{t}^{(u,v)}=\rho_{t}^{(u,v)}\cos\theta_{t}^{(u,v)},\quad y_{t}^{(u,v)}=\rho_{t}^{(u,v)}\sin\theta_{t}^{(u,v)}
$$

The resulting trajectory is denoted as $\tau^{(u,v)}=\{(x_{t}^{(u,v)},y_{t}^{(u,v)})\}_{t=1}^{T}$. The more detailed algorithm process is listed in Appendix A.

We apply the trajectory expansion algorithm to the top- $K$ candidates $\mathcal{T}_{\mathrm{gbl}}$, yielding an expanded set $\mathcal{T}_{\mathrm{gbl\text{-}exp}}=\{\tau_{j}\}_{j=1}^{M_{2}}$, where $M_{2}=K\cdot n_{\mathrm{exp}}^{2}$ denotes the total number of trajectories in the second stage.

Local Trajectory Refinement   This stage performs fine-grained trajectory refinement over the expanded candidates to produce the final refined local driving maneuver. Given the expanded trajectory set $\mathcal{T}_{\mathrm{gbl\text{-}exp}}=\{\tau_{j}\}_{j=1}^{M_{2}}$, the model employs the Trajectory Refinement Decoder $\mathrm{Dec_{lcl}}$ to refine these candidates:

$$
\displaystyle\left\{g_{j}\right\}_{j=1}^{M_{2}}=\mathrm{Enc_{traj}}\left(\left\{\tau_{j}\right\}_{j=1}^{M_{2}}\right)
$$
 
$$
\displaystyle\left\{(\hat{\tau}_{\mathrm{lcl},j},\hat{s}_{\mathrm{lcl},j}^{(\mathrm{mc})})\right\}_{j=1}^{M_{2}}=\mathrm{Dec_{lcl}}\left(\left\{g_{j}\right\}_{j=1}^{M_{2}},cond\right)
$$

where $cond$ represents the environmental features extracted in the first denoising stage, $\tau_{j}$ denotes the expanded trajectory, $g_{j}$ is the encoded expanded trajectory query, and $\hat{\tau}_{\mathrm{lcl},j}$ is the refined trajectory.

Besides decoding trajectories, HAD introduces multiple MLP heads in $\mathrm{Dec_{lcl}}$ to predict different metrics for each decoded trajectory, denoted as $\hat{s}_{\mathrm{lcl},j}^{(\mathrm{mc})}$. Specifically, the metric set $\mathrm{mc}$ covers three categories: (i) $\hat{s}_{\mathrm{lcl},j}^{(\mathrm{dist})}$, the distance to the expert trajectory, (ii) $\hat{s}_{\mathrm{lcl},j}^{(m)}$, safety-critical metrics defined in the NAVSIM benchmark \[dauner2024navsim\], and (iii) $\hat{s}_{\mathrm{lcl},j}^{(m\text{-}\mathrm{rl})}$, the metric-decoupled reinforcement learning weights (detailed in Sec. 3.2). The final planning output is derived through a weighted average of the refined candidates:

$$
\displaystyle\hat{s}_{\mathrm{lcl},j}^{(\mathrm{pdms})}=\sum_{mp}{\lambda^{(mp)}\log{\hat{s}_{\mathrm{lcl},j}^{(mp)}}}+\lambda_{\mathrm{avg}}\log\left({\sum_{ma}{\lambda^{(ma)}s_{\mathrm{lcl},j}^{(ma)}}}\right)
$$
 
$$
\displaystyle\hat{s}_{\mathrm{lcl},j}=\gamma_{\mathrm{dist}}\hat{s}_{\mathrm{lcl},j}^{(\mathrm{dist})}+\gamma_{\mathrm{pdms}}\hat{s}_{\mathrm{lcl},j}^{(\mathrm{pdms})}+\gamma_{\mathrm{rl}}\hat{s}_{\mathrm{lcl},j}^{(\mathrm{rl})}
$$
 
$$
\displaystyle\hat{\tau}=\sum_{j}w_{j}\hat{\tau}_{\mathrm{lcl},j},\quad\text{where }w_{j}=\mathtt{Softmax}(\hat{s}_{\mathrm{lcl},j})
$$

where $\hat{s}_{\mathrm{lcl},j}^{(\mathrm{pdms})}$ is the linear combination of logarithm of different metric scores $\hat{s}_{\mathrm{lcl},j}^{(m)}$, $mp$ and $ma$ are penalty metrics and average metrics. $\hat{s}_{\mathrm{lcl},j}^{(\mathrm{rl})}$ ensembles the metric-decoupled reinforcement learning weights $\hat{s}_{\mathrm{lcl},j}^{(m\text{-}\mathrm{rl})}$, following similar score ensemble procedure as $\hat{s}_{\mathrm{lcl},j}^{(\mathrm{pdms})}$. $\gamma_{\mathrm{dist}}$, $\gamma_{\mathrm{pdms}}$, and $\gamma_{\mathrm{rl}}$ are coefficients, and $\hat{\tau}_{\mathrm{lcl},j}$ is the trajectory output from the Local Trajectory Refinement stage.

### 3.2 Metric-Decoupled Policy Optimization

To better align diffusion-decoded trajectories with safety-related driving rules, we employ reinforcement learning for policy optimization. Prior end-to-end RL approaches, such as DiffusionDriveV2 \[zou2025diffusiondrivev2\], typically rely on a single aggregated reward representing overall trajectory quality. However, such a coarse reward signal can hinder effective optimization and may lead to reward hacking \[skalse2022defining\]. To address this issue, we propose Metric-Decoupled Policy Optimization (MDPO), which enables more structured optimization across multiple driving metrics.

Specifically, the first stage of the Hierarchical Diffusion Policy generates $K$ candidate trajectories. Through a trajectory expansion algorithm, these candidates are transformed into $K$ sets of local trajectories $\{\mathcal{T}_{\text{gbl-exp},k}\}_{k=1}^{K}$, where each set contains $M_{\mathrm{sub}}=n_{\mathrm{exp}}^{2}$ trajectories. For the $k$ -th set $\mathcal{T}_{\text{gbl-exp},k}=\left\{\hat{\tau}_{\mathrm{lcl},j}\right\}_{j\in\mathcal{I}_{k}}$, MDPO first computes the selection probability $p_{j}^{(m)}$ for each trajectory on each safety metric \[dauner2024navsim\], then calculates the reward $J_{k}$ for the $k$ -th trajectory set:

$$
\displaystyle p_{j}^{(m)}=\frac{\exp\left(\hat{s}_{\mathrm{lcl},j}^{(m\text{-}\mathrm{rl})}\right)}{\sum\limits_{i\in\mathcal{I}_{k}}\exp\left(\hat{s}_{\mathrm{lcl},i}^{(m\text{-}\mathrm{rl})}\right)}
$$
 
$$
\displaystyle\bar{r}_{j}^{(m)}=\frac{r_{j}^{(m)}-\operatorname{mean}\left(\{r_{i}^{(m)}\mid i\in\mathcal{I}_{k}\}\right)}{\operatorname{std}\left(\{r_{i}^{(m)}\mid i\in\mathcal{I}_{k}\}\right)+\epsilon}
$$
 
$$
\displaystyle J_{k}=\sum\limits_{m}{\alpha^{(m)}\sum_{j\in\mathcal{I}_{k}}p_{j}^{(m)}\cdot\bar{r}_{j}^{(m)}}
$$

where $m$ denotes the safety metrics, and $\hat{s}_{\mathrm{lcl},j}^{(m\text{-}\mathrm{rl})}$ is the RL logit predicted by the model for the $j$ -th trajectory regarding the $m$ -th metric. $\bar{r}_{j}^{(m)}$ represents the normalized reward for metric $m$, $\mathcal{I}_{k}$ denotes trajectory indices in the $k$ -th set. Finally, $\alpha^{(m)}$ is the predefined coefficient for each metric, $J_{k}$ is the reward of the $k$ -th trajectory set. The total reward $J$ is defined as the mean reward across all trajectory sets: $J=\frac{1}{K}\sum_{k=1}^{K}J_{k}$.

### 3.3 Efficient Offline Trajectory Reward Retrieval

![[x3 27.png|Refer to caption]]

Figure 3: Comparison between simulation-based trajectory evaluation and our proposed Offline Reward Retrieval scheme.

Driving reward scores typically require running a simulator to compute safety metrics such as collision avoidance \[zou2025diffusiondrivev2\]. However, such a simulation is a CPU-intensive task and time-consuming, making it inefficient and difficult to scale when executed on-the-fly during training, as illustrated in Fig. 3 (a). Therefore, we propose an efficient Offline Trajectory Reward Retrieval Scheme based on driving area discretization, as shown in Fig. 3 (b). Inspired by recent trajectory-selection frameworks \[li2024hydra, li2025hydranext\], HAD incorporates a extensive trajectory vocabulary $\mathcal{V}=\{\tau_{\mathrm{ref},j}\}_{j=1}^{N}$, providing dense $N$ trajectories covering the whole feasible driving area. For each trajectory $\tau_{\mathrm{ref},j}$ in the vocabulary $\mathcal{V}$, we calculate its safety metric scores $s_{\mathrm{ref},j}^{(m)}$ in advance, and these scores are stored in a trajectory reward caching table. During online training, for an arbitrary output trajectory $\hat{\tau}$ predicted by the model, our scheme performs a nearest-neighbor query to find the closest reference trajectory $\tau_{\mathrm{ref},j^{*}}$ in $\mathcal{V}$:

$$
\tau_{\mathrm{ref},j^{*}}=\underset{\tau_{\mathrm{ref},j}\in\mathcal{V}}{\operatorname{arg\,min\,}}\mathrm{Dist}(\hat{\tau},\tau_{\mathrm{ref},j})
$$

The corresponding scores $s_{\mathrm{ref},j^{*}}^{(m)}$ are directly utilized as the approximation of the simulation score of $\hat{\tau}$.

This strategy converts expensive online simulation into an efficient nearest-neighbor trajectory retrieval from the offline cache. As a result, reinforcement learning rewards can be obtained much more efficiently, significantly improving training efficiency.

### 3.4 Loss Function

The loss function of HAD consists of three components: the perception loss $L_{\mathrm{percept}}$, the driving intent establishment loss $L_{\mathrm{global}}$, and the local trajectory refinement loss $L_{\mathrm{local}}$:

$$
L=\lambda_{\mathrm{percept}}L_{\mathrm{percept}}+\lambda_{\mathrm{global}}L_{\mathrm{global}}+\lambda_{\mathrm{local}}L_{\mathrm{local}}
$$

The perception loss $L_{\mathrm{percept}}$ follows the design of Transfuser \[chitta2022transfuser\], including the detection loss for 3D bounding box regression, the classification loss for object categories, and the semantic segmentation loss on bird’s-eye-view (BEV) representation. The driving intent establishment loss $L_{\mathrm{global}}$ is employed to supervise the output of the first-stage decoder $\mathrm{Dec_{gbl}}$, comprising the trajectory regression loss and the region classification loss, which determines whether the predicted trajectory and the ground-truth human trajectory belong to the same spatial region. The local trajectory refinement loss $L_{\mathrm{local}}$ aims to supervise the refined trajectories and their multi-dimensional quality metrics produced in the second stage, including the distance to the human expert trajectory, supervision for safety metrics, and the reinforcement learning (RL) objective. More details about the loss function are listed in Appendix B.

## 4 Experiments

### 4.1 Implementation Details

Datasets   The experiments are mainly conducted on two datasets: NAVSIM \[dauner2024navsim\] and HUGSIM \[zhou2024hugsim\]. NAVSIM is an open-loop planning benchmark. The output trajectory is judged by a simulator to determine whether this trajectory follows driving rules. The evaluation metric of NAVSIM is PDMS, which is aggregated by several driving sub-metrics:

$$
\mathrm{PDMS}=\left(\prod_{m\in S_{\mathrm{pen}}}{\mathrm{s}_{m}}\right)\times\left(\frac{\sum_{w\in S_{\mathrm{avg}}}{\mathrm{w}_{w}\times\mathrm{s}_{w}}}{\sum_{w\in S_{\mathrm{avg}}}{\mathrm{w}_{w}}}\right)
$$

where $S_{\mathrm{pen}}$ and $S_{\mathrm{avg}}$ denote penalty and weighted average metrics. NAVSIM develops two evaluation schemes: NAVSIM v1 and NAVSIM v2. In NAVSIM v1, the sub-metrics in $S_{\mathrm{pen}}$ include No Collisions (NC) and Drivable Area Compliance (DAC), while $S_{\mathrm{avg}}$ includes Ego Progress (EP), Time-to-Collision (TTC), and Comfort (C). NAVSIM v2 introduces 4 new sub-metrics: Driving Direction Compliance (DDC) and Traffic Light Compliance (TLC) belong to $S_{\mathrm{pen}}$, while Lane Keeping (LK) and Extended Comfort (EC) belong to $S_{\mathrm{avg}}$.

HUGSIM is a closed-loop planning benchmark covering over 400 simulation scenarios. These scenarios are categorized into four difficulty levels: Easy, Medium, Hard, and Extreme. HUGSIM provides a comprehensive metric, HD-Score, to evaluate the closed-loop performance of planning models across multiple dimensions, including No Collision (NC), Driveable Area Compliance (DAC), Time-to-Collision (TTC), Comfort (COM), and Route Completion ($R_{c}$). During the simulation, the HD-Score at each timestep $t$ is calculated as follows:

$$
\text{HD-Score}_{t}=\left(\prod\limits_{m\in\{NC,DAC\}}{score_{m}}\right)\cdot\left(\frac{\sum_{w\in\{TTC,COM\}}{weight_{w}\cdot score_{w}}}{\sum_{w\in\{TTC,COM\}}weight_{w}}\right)
$$

The final HD-Score is obtained by averaging the scores across all timesteps and multiplying the route completion rate $R_{c}$.

Model Details   The backbone of HAD is identical to Transfuser \[chitta2022transfuser\]. The environment encoder $\mathrm{Enc_{env}}$ adopts a dual-modal pipeline, both the image and LiDAR branches utilize a ResNet34 \[he2016deep\] backbone. The resulting environment feature $cond$ comprises encoded query vectors of other agents and an $8\times 8$ BEV feature map. The trajectory encoder $\mathrm{Enc_{traj}}$ is a 2-layer MLP, while both the Driving Intention Decoder $\mathrm{Dec_{gbl}}$ and Trajectory Refinement Decoder $\mathrm{Dec_{lcl}}$ consist of 1 Transformer layer. For the Hierarchical Diffusion Policy, we set the number of predefined candidate trajectories $M_{1}$ to 20. After the first-stage denoising, $K=2$ top-scoring trajectories are selected. In the trajectory expansion algorithm, the expansion factor $n_{\mathrm{exp}}$ is set to 5, the corresponding radial scaling coefficient set $\{\lambda_{u}\}$ is $\{0.92,0.96,1.0,1.04,1.08\}$, and the angular coefficient set $\{\delta_{v}\}$ is $\{-6^{\circ},-3^{\circ},0^{\circ},3^{\circ},6^{\circ}\}$. This process generates $M_{\mathrm{sub}}=25$ trajectories within each local region and $M_{2}=50$ trajectories in total. The trajectory vocabulary $\mathcal{V}$ for the Offline Reward Approximation Scheme consists of $N=8192$ trajectories. The coefficients $\gamma_{\mathrm{dist}}$, $\gamma_{\mathrm{pdms}}$, and $\gamma_{\mathrm{rl}}$ in Equ. 10 are 0.6, 0.05, and 0.01. We train two model variants: HAD and HAD-L. HAD requires both image and LiDAR inputs. HAD-L follows Latent Transfuser \[chitta2022transfuser\], replacing LiDAR inputs with learnable positional embeddings, thus only requiring image inputs. More model details and training details are shown in the Appendix C.

Table 1: Results on NAVSIM v1. “C+L” denotes that the model requires both RGB image and LiDAR as inputs, while “C” indicates that the model only receives image input.

| Method | Input | NC $\uparrow$ | DAC $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | C $\uparrow$ | PDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Human | — | 100 | 100 | 87.5 | 100 | 99.9 | 94.8 |
| Transfuser \[chitta2022transfuser\] | C+L | 97.7 | 92.8 | 79.2 | 92.8 | 100 | 84.0 |
| UniAD \[hu2023planning\] | C | 97.8 | 91.9 | 78.8 | 92.9 | 100 | 83.4 |
| VADv2 \[chen2024vadv2\] | C | 97.9 | 91.7 | 77.6 | 92.9 | 100 | 83.0 |
| LAW \[li2025enhancing\] | C | 96.4 | 95.4 | 81.7 | 88.7 | 99.9 | 84.6 |
| DRAMA \[yuan2024drama\] | C+L | 98.0 | 93.1 | 80.1 | 94.8 | 100 | 85.5 |
| GoalFlow \[xing2025goalflow\] | C+L | 98.3 | 93.8 | 79.8 | 94.3 | 100 | 85.7 |
| Hydra-MDP \[li2024hydra\] | C+L | 98.3 | 96.0 | 78.7 | 94.6 | 100 | 86.5 |
| DiffusionDrive \[liao2024diffusiondrive\] | C+L | 98.2 | 96.2 | 82.2 | 94.7 | 100 | 88.1 |
| WoTE \[li2025endtoend\] | C+L | 98.5 | 96.8 | 81.9 | 94.9 | 99.9 | 88.3 |
| DriveSuprim \[yao2025drivesuprim\] | C | 97.8 | 97.3 | 86.7 | 93.6 | 100 | 89.9 |
| HAD | C+L | 98.2 | 97.3 | 87.4 | 97.5 | 100 | 90.2 |
| HAD-L | C | 98.1 | 97.2 | 87.2 | 97.3 | 100 | 89.9 |

Table 2: Results on NAVSIM v2. “C+L” denotes that the model requires both RGB image and LiDAR as inputs, while “C” indicates that the model only receives image input.

| Method | Input | NC $\uparrow$ | DAC $\uparrow$ | DDC $\uparrow$ | TLC $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | LK $\uparrow$ | HC $\uparrow$ | EC $\uparrow$ | EPDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Human | — | 100 | 100 | 99.8 | 100 | 87.4 | 100 | 100 | 98.1 | 90.1 | 90.3 |
| Ego Status MLP | — | 93.1 | 77.9 | 92.7 | 99.6 | 86.0 | 91.5 | 89.4 | 98.3 | 85.4 | 64.0 |
| Transfuser \[chitta2022transfuser\] | C+L | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 |
| HydraMDP++ \[li2024hydramdp\_pp\] | C | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | 81.4 |
| DriveSuprim \[yao2025drivesuprim\] | C | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 | 83.1 |
| DiffusionDriveV2 \[zou2025diffusiondrivev2\] | C+L | 97.7 | 96.6 | 99.2 | 99.8 | 88.9 | 97.2 | 96.0 | 97.8 | 91.0 | 85.5 |
| DiffRefiner \[yin2025diffrefiner\] | C | 98.5 | 97.4 | 99.6 | 99.8 | 87.6 | 97.7 | 97.7 | 98.3 | 86.2 | 86.2 |
| EvaDrive \[jiao2025evadrive\] | C | 98.8 | 98.5 | 98.9 | 99.8 | 96.6 | 98.4 | 94.3 | 97.8 | 55.9 | 86.3 |
| HAD | C+L | 98.2 | 97.3 | 99.2 | 99.8 | 87.4 | 97.5 | 95.2 | 98.3 | 86.2 | 88.6 |
| HAD-L | C | 98.1 | 97.2 | 99.3 | 99.8 | 87.2 | 97.3 | 95.3 | 98.3 | 86.4 | 88.5 |

### 4.2 Quantitative Results

Result on NAVSIM   Tab. 1 and Tab. 2 present the performance of HAD on the NAVSIM benchmark. On NAVSIM v1, HAD achieves 90.2 PDMS, outperforming DiffusionDrive by 2.1 PDMS. On the more challenging NAVSIM v2 benchmark, HAD reaches 88.6 EPDMS, surpassing previous state-of-the-art methods such as DiffRefiner by 2.4 and EvaDrive by 2.3 PDMS, respectively. Furthermore, the camera-only variant HAD-L achieves nearly identical performance to HAD on NAVSIM v1 and v2, with gaps of only 0.3% and 0.1%, demonstrating robustness across different input modalities. The inference speed is reported in Appendix D.1.

Table 3: Results on the HUGSIM dataset. “RC” denotes the Route Completion rate, and “HDS” represents the HD-Score metric. The asterisk symbol “\*” indicates that the results are evaluated on both public and private datasets; otherwise, the results refer to the performance on the public dataset only.

<table><tbody><tr><td rowspan="2">Method</td><td colspan="2">Easy</td><td colspan="2">Medium</td><td colspan="2">Hard</td><td colspan="2">Extreme</td><td colspan="2">Overall</td></tr><tr><td>RC</td><td>HDS</td><td>RC</td><td>HDS</td><td>RC</td><td>HDS</td><td>RC</td><td>HDS</td><td>RC</td><td>HDS</td></tr><tr><td>UniAD* <cite>[hu2023planning]</cite></td><td>58.6</td><td>48.7</td><td>41.2</td><td>29.5</td><td>40.4</td><td>27.3</td><td>26.0</td><td>14.3</td><td>40.6</td><td>28.9</td></tr><tr><td>VAD* <cite>[jiang2023vad]</cite></td><td>38.7</td><td>24.3</td><td>27.0</td><td>9.9</td><td>25.5</td><td>10.4</td><td>23.0</td><td>8.2</td><td>27.9</td><td>12.3</td></tr><tr><td>Latent TransFuser* <cite>[chitta2022transfuser]</cite></td><td>68.4</td><td>52.8</td><td>40.7</td><td>24.6</td><td>36.9</td><td>19.8</td><td>25.5</td><td>8.1</td><td>41.4</td><td>24.8</td></tr><tr><td>Latent TransFuser <cite>[chitta2022transfuser]</cite></td><td>60.4</td><td>42.5</td><td>39.4</td><td>17.7</td><td>32.7</td><td>11.8</td><td>27.9</td><td>10.6</td><td>37.9</td><td>18.0</td></tr><tr><td>GTRS-Dense <cite>[li2025generalized]</cite></td><td>64.2</td><td>55.5</td><td>50.0</td><td>39.0</td><td>20.7</td><td>11.7</td><td>22.3</td><td>14.3</td><td>38.0</td><td>28.6</td></tr><tr><td>ZTRS <cite>[li2025ztrs]</cite></td><td>74.4</td><td>60.8</td><td>50.9</td><td>34.2</td><td>32.7</td><td>20.5</td><td>21.9</td><td>11.0</td><td>42.6</td><td>28.9</td></tr><tr><td>HAD-L</td><td>76.9</td><td>65.9</td><td>49.3</td><td>31.4</td><td>36.4</td><td>18.0</td><td>39.1</td><td>22.5</td><td>47.5</td><td>30.8</td></tr></tbody></table>

Result on HUGSIM   Tab. 3 presents the evaluation results of HAD-L on the high-fidelity closed-loop benchmark HUGSIM. HAD-L achieves 47.5 Route Completion and 30.8 HD-Score. Compared to ZTRS, HAD-L improves the RC and HD-Score by 4.9 and 1.9 points. The performance improvement becomes even more pronounced in challenging scenarios. In extreme scenarios characterized by frequent and aggressive agent attacks, HAD-L maintains an RC of 39.1 and an HD-Score of 22.5, significantly surpassing all baseline methods by a huge margin. These results highlight the model’s exceptional driving capabilities in highly interactive, challenging environments.

### 4.3 Ablation Studies

Table 4: Ablation on different trajectory evaluation metrics in hierarchical denoising stages.

<table><tbody><tr><td colspan="3">Global Denoising</td><td colspan="3">Local Denoising</td><td rowspan="2">EPDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>Dist.</td><td>Safety</td><td>RL</td><td>Dist.</td><td>Safety</td><td>RL</td></tr><tr><td>✓</td><td></td><td></td><td>✓</td><td></td><td></td><td>86.7</td></tr><tr><td>✓</td><td></td><td></td><td>✓</td><td>✓</td><td></td><td>87.3</td></tr><tr><td>✓</td><td></td><td></td><td>✓</td><td>✓</td><td>✓</td><td>88.6</td></tr><tr><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>87.2</td></tr></tbody></table>

Table 5: Ablation on the number of first-stage selected trajectories and expanded trajectories.

<table><tbody><tr><td colspan="2">Training</td><td colspan="2">Inference</td><td rowspan="2">EPDMS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>Top-K</td><td><math><semantics><msubsup><mi>n</mi> <mi>exp</mi> <mn>2</mn></msubsup> <annotation>n_{\mathrm{exp}}^{2}</annotation></semantics></math></td><td>Top-K</td><td><math><semantics><msubsup><mi>n</mi> <mi>exp</mi> <mn>2</mn></msubsup> <annotation>n_{\mathrm{exp}}^{2}</annotation></semantics></math></td></tr><tr><td>1</td><td><math><semantics><mrow><mn>5</mn> <mo>×</mo> <mn>5</mn></mrow> <annotation>5\times 5</annotation></semantics></math></td><td>1</td><td><math><semantics><mrow><mn>5</mn> <mo>×</mo> <mn>5</mn></mrow> <annotation>5\times 5</annotation></semantics></math></td><td>88.1</td></tr><tr><td>2</td><td><math><semantics><mrow><mn>5</mn> <mo>×</mo> <mn>5</mn></mrow> <annotation>5\times 5</annotation></semantics></math></td><td>2</td><td><math><semantics><mrow><mn>5</mn> <mo>×</mo> <mn>5</mn></mrow> <annotation>5\times 5</annotation></semantics></math></td><td>88.5</td></tr><tr><td>2</td><td><math><semantics><mrow><mn>5</mn> <mo>×</mo> <mn>5</mn></mrow> <annotation>5\times 5</annotation></semantics></math></td><td>2</td><td><math><semantics><mrow><mn>7</mn> <mo>×</mo> <mn>7</mn></mrow> <annotation>7\times 7</annotation></semantics></math></td><td>88.6</td></tr><tr><td>4</td><td><math><semantics><mrow><mn>5</mn> <mo>×</mo> <mn>5</mn></mrow> <annotation>5\times 5</annotation></semantics></math></td><td>4</td><td><math><semantics><mrow><mn>5</mn> <mo>×</mo> <mn>5</mn></mrow> <annotation>5\times 5</annotation></semantics></math></td><td>86.4</td></tr><tr><td>20</td><td><math><semantics><mrow><mn>5</mn> <mo>×</mo> <mn>5</mn></mrow> <annotation>5\times 5</annotation></semantics></math></td><td>20</td><td><math><semantics><mrow><mn>5</mn> <mo>×</mo> <mn>5</mn></mrow> <annotation>5\times 5</annotation></semantics></math></td><td>79.8</td></tr></tbody></table>

Trajectory Metric Scores in Hierarchical Modeling   In our hierarchical modeling, we use the distance towards the human expert trajectory to select top-K candidates in the Driving Intention Establishment stage, while utilizing multiple metrics in the Local Trajectory Refinement stage, as shown in Equ. 8. The result in Tab. 4 shows the influence of different metric scores within the hierarchical framework, including the distance, safety metric scores, and RL weights. When the model only utilizes the distance to the human trajectory as supervision in trajectory refinement, it only achieves an 86.7 EPDMS. Incorporating the safety metric as scoring supervision raises the performance to 87.3 EPDMS. Furthermore, adding reinforcement learning training in the Local Trajectory Refinement stage further improves the EPDMS to 88.6. This indicates that using comprehensive metrics and adopting RL training helps the model learns a more robust driving policy aligning with driving rules. Moreover, introducing safety metric scoring and RL supervision in the Driving Intention Establishment stage leads to a performance drop to 87.2. This indicates that the whole unstructured driving space makes it difficult for complex reward signals to provide valuable information.

Trajectory Expansion Algorithm   Our proposed Structure-preserved Trajectory Expansion Algorithm effectively explores the local driving region while preserving the trajectory kinematic structure. Tab. 5 and Tab. 6 further show the superiority. Tab. 5 shows the ablation on the number of first-stage filtered trajectories $K$ and the expansion scale $n_{\mathrm{exp}}\times n_{\mathrm{exp}}$. The results show that constraining $K$ to a small range (e.g., $K=2$) significantly enhances model performance, achieving the optimal 88.6 EPDMS. When $K$ is increased to include all 20 trajectories, the EPDMS drops to 79.8. This indicates the necessity to adopt hierarchical modeling in end-to-end planning. Furthermore, increasing the sampling density from $5\times 5$ to $7\times 7$ during inference yields a performance gain of 0.1 EPDMS. This suggests that a denser local area search during inference helps the model precisely locate better trajectories.

Tab. 6 compares our trajectory expansion algorithm with other approaches illustrated in Fig. 4. Directly adding random noise to trajectories damages their intrinsic structure, only leading to 84.9 EPDMS. Expanding the trajectory in the Cartesian coordinate system \[zou2025diffusiondrivev2\] (XY Expand) preserves the trajectory structure, but fails to provide a spatially balanced exploration. As shown in Fig. 4, when the trajectory y-coordinate is small, the exploration range along the y-axis becomes severely restricted. In comparison, our expansion in the polar coordinate system can make a sufficient coverage of the local area, leading to the best 88.6 EPDMS.

![[x4 25.png|Refer to caption]]

Figure 4: Illustration of different trajectory expansion algorithms. Directly adding random noise is harmful to trajectory kinematic structure, expanding trajectory in the Cartesian space (XY Expand) suffer from insufficient exploration on local region, and expansion in the polar space (Polar Expand) leads to comprehensive exploration.

Table 6: Comparison of different trajectory expansion algorithms.

| Expansion | NC $\uparrow$ | DAC $\uparrow$ | DDC $\uparrow$ | TLC $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | LK $\uparrow$ | HC $\uparrow$ | EC $\uparrow$ | EPDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random Noise | 98.1 | 93.3 | 99.3 | 99.8 | 87.6 | 96.9 | 95.6 | 98.3 | 85.4 | 84.9 |
| XY Expand | 98.2 | 93.8 | 99.4 | 99.8 | 87.6 | 97.1 | 95.4 | 98.3 | 87.5 | 85.9 |
| Polar Expand | 98.2 | 97.3 | 99.2 | 99.8 | 87.4 | 97.5 | 95.2 | 98.3 | 86.2 | 88.6 |

Table 7: Ablation on the effectiveness of MDPO. Results show that Metric-Decoupled Policy Optimization improves overall performance compared to using a coupled reward and optimization.

| Reward | Decoupled | NC $\uparrow$ | DAC $\uparrow$ | DDC $\uparrow$ | TLC $\uparrow$ | EP $\uparrow$ | TTC $\uparrow$ | LK $\uparrow$ | HC $\uparrow$ | EC $\uparrow$ | EPDMS $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dist. |  | 92.0 | 91.2 | 96.6 | 99.5 | 88.7 | 91.6 | 85.1 | 41.3 | 3.9 | 62.6 |
| PDMS |  | 97.9 | 96.7 | 99.3 | 99.7 | 88.1 | 97.1 | 94.7 | 98.2 | 84.5 | 87.8 |
| Dist. + PDMS |  | 97.9 | 96.2 | 99.1 | 99.8 | 88.0 | 96.9 | 94.3 | 98.3 | 80.8 | 86.9 |
| Decoupled Metrics | ✓ | 98.2 | 97.3 | 99.2 | 99.8 | 87.4 | 97.5 | 95.2 | 98.3 | 86.2 | 88.6 |

Superiority of MDPO   Tab. 7 investigates the RL setting, showing the superiority of MDPO. Firstly, we validate the reward choice when adopting a single RL weight head. The first three rows in Tab. 7 reveal that adopting the comprehensive PDMS score as the reward function can help the model achieve the optimal 87.8 EPDMS. After adopting our proposed MDPO to decouple different metric rewards, the performance is further improved to 88.6 EPDMS, which shows that reward decoupling leads to a better driving policy.

### 4.4 Training Efficiency Analysis

According to our testing, the training efficiency of the model is significantly enhanced by the proposed Offline RL Reward Retrieval Scheme. By utilizing this retrieval mechanism, the reward acquisition latency for a single trajectory is reduced from 0.2449s to 0.0042s. Consequently, the total training duration is decreased from 64.4 hours to 13.6 hours. This shows that, compared with online simulation-based training such as DiffusionDriveV2 \[zou2025diffusiondrivev2\], our scheme can achieve a 5 $\times$ speedup, demonstrating superior efficiency.

## 5 Conclusion

In this paper, we propose HAD to address the critical limitations of current end-to-end planners on the optimization burden in large action spaces and the challenges of online reinforcement learning. HAD introduces a Hierarchical Diffusion Policy to decompose planning into coarse-to-fine denoising stages, a Structure-preserved Trajectory Expansion algorithm to generate diverse trajectory candidates while keeping their kinematic structure, and a Metric-Decoupled Policy Optimization (MDPO) framework with an Offline Reward Retrieval Scheme to efficiently learn a fine-grained driving policy. Experimental results on the NAVSIM and HUGSIM benchmarks demonstrate the superior performance of HAD on both open-loop and closed-loop planning.

## Appendix

## Appendix A Detailed Process of Structure-Preserved Trajectory Expansion

In this section, we provide the detailed algorithm process of the Structure-Preserved Trajectory Expansion Algorithm in Sec. 3.1.

Given a trajectory $\tau=\{(x_{t},y_{t})\}_{t=1}^{T}$ and an expansion number $n_{\mathrm{exp}}$, the Structure-Preserved Trajectory Expansion Algorithm outputs an expanded trajectory set $\mathcal{T}_{\mathrm{exp}}=\{\tau_{j}\}_{j=1}^{n_{\mathrm{exp}}\cdot n_{\mathrm{exp}}}$. Specifically, the algorithm first projects the Cartesian coordinates of $\tau$ into the polar domain $\tau_{\mathrm{polar}}=\{(\rho_{t},\theta_{t})\}_{t=1}^{T}$, then it defines a set of radial scaling coefficients $\{\lambda_{u}\}_{u=1}^{n_{\mathrm{exp}}}$ and angular offset coefficients $\{\delta_{v}\}_{v=1}^{n_{\mathrm{exp}}}$. The algorithm traverses these coefficients to transform the original trajectory and get multiple expanded trajectories. The detailed algorithm process is shown in Alg. 1.

## Appendix B Loss Function Details

This section provides details of the loss function $L$ used for model training. The loss of HAD comprises three components: the perception loss $L_{\mathrm{percept}}$, the loss for the Driving Intention Establishment stage $L_{\mathrm{global}}$, and the loss for the Local Trajectory Refinement stage $L_{\mathrm{local}}$.

Perception Loss   The perception loss $L_{\mathrm{percept}}$ follows the Transfuser \[chitta2022transfuser\] design and incorporates several auxiliary perception tasks, including a detection loss $L_{\mathrm{box}}$ for 3D bounding box regression, a classification loss $L_{\mathrm{label}}$ for predicting the category labels of other agents, and a BEV semantic segmentation loss $L_{\mathrm{bev}}$:

$$
L_{\mathrm{percept}}=\lambda_{\mathrm{box}}L_{\mathrm{box}}+\lambda_{\mathrm{label}}L_{\mathrm{label}}+\lambda_{\mathrm{bev}}L_{\mathrm{bev}}
$$

specifically, $L_{\mathrm{box}}$ employs an $L_{1}$ loss to optimize the location of the bounding boxes of other agents, $L_{\mathrm{label}}$ employs cross-entropy loss to distinguish different categories such as vehicles and pedestrians, and $L_{\mathrm{bev}}$ improves the model’s understanding of drivable areas through pixel-level semantic mask prediction.

Driving Intention Establishment Loss   The loss for the Driving Intention Establishment stage $L_{\mathrm{global}}$ is used to supervise the output of the first-stage decoder $\mathrm{Dec_{gbl}}$, following an anchor-based diffusion planner \[liao2024diffusiondrive\]. For the $M_{1}$ initially generated trajectories, the loss function consists of a trajectory regression loss $L_{\mathrm{gbl,reg}}$ and a region classification loss $L_{\mathrm{gbl,cls}}$:

$$
\displaystyle L_{\mathrm{gbl,reg}}=\sum_{j=1}^{M_{1}}\mathbf{1}\left(j\in\mathcal{I}_{\mathrm{pos}}\right)\left\lVert\hat{\tau}_{\mathrm{gbl},j}-\tau_{\mathrm{gt}}\right\rVert_{2}^{2}
$$
 
$$
\displaystyle L_{\mathrm{gbl,cls}}=\mathrm{CrossEntropy}\left(\hat{\mathbf{s}}_{\mathrm{gbl}}^{(c)},\,\mathbf{y}\right)
$$
 
$$
\displaystyle L_{\mathrm{global}}=\lambda_{\mathrm{reg}}L_{\mathrm{gbl,reg}}+\lambda_{\mathrm{cls}}L_{\mathrm{gbl,cls}}
$$

where $\tau_{\mathrm{gt}}$ represents the human expert trajectory, $\hat{\tau}_{\mathrm{gbl},j}$ is the $j$ -th denoised candidate trajectory, and $\mathcal{I}_{\mathrm{pos}}$ denotes the index of the denoised trajectory that is located in the same driving sub-region as the expert trajectory. $\hat{\mathbf{s}}_{\mathrm{gbl}}^{(c)}=\left(\hat{s}_{\mathrm{gbl,1}}^{(c)},\hat{s}_{\mathrm{gbl,2}}^{(c)},\dots,\hat{s}_{\mathrm{gbl,M_{1}}}^{(c)}\right)$ represents the predicted classification scores for each trajectory, and $\mathbf{y}=\left(y_{1},y_{2},\dots,y_{M_{1}}\right)$ is the ground truth label for region classification, where each element $y_{j}$ indicates whether the $j$ -th candidate trajectory lies in the same driving sub-region as the expert trajectory.

Input: Trajectory $\tau=\{(x_{t},y_{t})\}_{t=1}^{T}$, expansion number $n_{\mathrm{exp}}$, radial scaling coefficient set $\{\lambda_{u}\}_{u=1}^{n_{\mathrm{exp}}}$, angular offset coefficient set $\{\delta_{v}\}_{v=1}^{n_{\mathrm{exp}}}$

Output: Expanded trajectory set $\mathcal{T}_{\mathrm{exp}}=\{\tau_{j}\}_{j=1}^{n_{\mathrm{exp}}\cdot n_{\mathrm{exp}}}=\{\tau^{(u,v)}\}_{u=1,v=1}^{n_{\mathrm{exp}},n_{\mathrm{exp}}}$

// Convert Cartesian coordinates to polar coordinates

 $\tau_{\mathrm{polar}}\leftarrow\emptyset$

for *$t\leftarrow 1$ to $T$* do

    $\rho_{t}\leftarrow\sqrt{x_{t}^{2}+y_{t}^{2}}$     $\theta_{t}\leftarrow\operatorname{atan2}(y_{t},x_{t})$     $\tau_{\mathrm{polar}}\leftarrow\tau_{\mathrm{polar}}\cup\{(\rho_{t},\theta_{t})\}$

end for

// Perform pattern expansion based on coefficient sets

 $\mathcal{T}_{\mathrm{exp}}\leftarrow\emptyset$

for *$u\leftarrow 1$ to $n_{\mathrm{exp}}$* do

    for *$v\leftarrow 1$ to $n_{\mathrm{exp}}$* do

       $\tau^{(u,v)}\leftarrow\emptyset$

       for *$t\leftarrow 1$ to $T$* do

          $\rho_{t}^{(u)}\leftarrow\rho_{t}\cdot\lambda_{u}$           $\theta_{t}^{(v)}\leftarrow\theta_{t}+\delta_{v}$           $x_{t}^{(u,v)}\leftarrow\rho_{t}^{(u)}\cdot\cos\theta_{t}^{(v)}$           $y_{t}^{(u,v)}\leftarrow\rho_{t}^{(u)}\cdot\sin\theta_{t}^{(v)}$           $\tau^{(u,v)}\leftarrow\tau^{(u,v)}\cup\{(x_{t}^{(u,v)},y_{t}^{(u,v)})\}$

       end for

       $\mathcal{T}_{\mathrm{exp}}\leftarrow\mathcal{T}_{\mathrm{exp}}\cup\{\tau^{(u,v)}\}$

    end for

end for

return $\mathcal{T}_{\mathrm{exp}}$

Algorithm 1 Structure-Preserved Trajectory Expansion

Local Trajectory Refinement Loss   The loss of the Local Trajectory Refinement stage $L_{\mathrm{local}}$ supervises the refined trajectories output in the second stage with multiple quality metrics and RL weights. In Equ. 8, $\mathrm{Dec_{lcl}}$ predicts several metrics, including distance, safety metric scores, and metric-decoupled reinforcement learning weights. For the distance score $\hat{s}_{\mathrm{lcl},j}^{(\mathrm{dist})}$, the model calculates the distance $d_{j}$ between the predicted trajectory $\hat{\tau}_{\mathrm{lcl},j}$ and the expert trajectory $\tau_{\mathrm{gt}}$, and convert the distance to a soft label $d_{\mathrm{norm},j}$ using Gaussian kernel and max-normalization. The binary cross-entropy (BCE) loss $L_{\mathrm{dist}}$ is computed between the soft label $d_{\mathrm{norm},j}$ and the predicted score $\hat{s}_{\mathrm{lcl},j}^{(\mathrm{dist})}$. Secondly, for the safety metric scores $\hat{s}_{\mathrm{lcl},j}^{(m)}$, the model uses the reference trajectory scores $s_{\mathrm{ref},j^{*}}^{(m)}$ retrieved via the Trajectory Reward Retrieval Scheme (detailed in Sec. 3.3) as the ground truth to calculate the BCE loss $L_{\mathrm{safe}}$. Finally, a reinforcement learning loss $L_{\mathrm{rl}}$ is introduced to maximize the expected reward $J$. The Local Trajectory Refinement loss is shown as follows:

$$
\displaystyle d_{\mathrm{norm},j}=\frac{\exp(-\beta\|\hat{\tau}_{\mathrm{lcl},j}-\tau_{\mathrm{gt}}\|_{2}^{2})}{\max\limits_{i\in\{1,\dots,M_{2}\}}\{\exp(-\beta\|\hat{\tau}_{\mathrm{lcl},i}-\tau_{\mathrm{gt}}\|_{2}^{2})\}}
$$
 
$$
\displaystyle L_{\mathrm{dist}}=\sum_{j=1}^{M_{2}}\mathrm{BCE}(\hat{s}_{\mathrm{lcl},j}^{(\mathrm{dist})},d_{\mathrm{norm},j})
$$
 
$$
\displaystyle L_{\mathrm{safe}}=\sum_{m}\sum_{j=1}^{M_{2}}\mathrm{BCE}(\hat{s}_{\mathrm{lcl},j}^{(m)},s_{\mathrm{ref},j^{*}}^{(m)})
$$
 
$$
\displaystyle L_{\mathrm{rl}}=-J=-\frac{1}{K}\sum_{k=1}^{K}\sum_{m}\alpha^{(m)}\sum_{j\in\mathcal{I}_{k}}p_{j}^{(m)}\cdot\bar{r}_{j}^{(m)}
$$
 
$$
\displaystyle L_{\mathrm{local}}=\lambda_{\mathrm{dist}}L_{\mathrm{dist}}+\lambda_{\mathrm{safe}}L_{\mathrm{safe}}+\lambda_{\mathrm{rl}}L_{\mathrm{rl}}
$$

where $\beta$ denotes the coefficient of the Gaussian function, $M_{2}$ is the number of expanded trajectories in Local Trajectory Refinement stage, $m$ represents safety evaluation metrics defined by NAVSIM \[dauner2024navsim, cao2025pseudo\], including NC, DAC, DDC, TLC, EP, TTC, LK, HC, $\alpha^{(m)}$ denotes the coefficient used to ensemble RL rewards on decoupled metrics, $\mathcal{I}_{k}$ denotes trajectory indices in the k-th expansion set, $p^{(m)}$ and $\bar{r}_{j}^{(m)}$ denotes the selection probability and normalized reward, identical to Equ. 14 in the main text.

The overall loss of HAD is the linear combination of three components mentioned above:

$$
L=\lambda_{\mathrm{percept}}L_{\mathrm{percept}}+\lambda_{\mathrm{global}}L_{\mathrm{global}}+\lambda_{\mathrm{local}}L_{\mathrm{local}}
$$

## Appendix C Model Details and Training Details

This section supplies some model details and training details not mentioned in Sec 4.1.

Supplementary Model Details   In Equ. 9 of the main text, HAD utilizes the following equation to ensemble different predicted metric scores:

$$
\hat{s}_{\mathrm{lcl},j}^{(\mathrm{pdms})}=\sum_{\mathrm{mp}}{\lambda^{(\mathrm{mp})}\log{\hat{s}_{\mathrm{lcl},j}^{(\mathrm{mp})}}}+\lambda_{\mathrm{avg}}\log\left({\sum_{\mathrm{ma}}{\lambda^{(\mathrm{ma})}s_{\mathrm{lcl},j}^{(\mathrm{ma})}}}\right)
$$

where $mp$ and $ma$ are penalty metrics (NC, DAC, DDC, TLC) and average metrics (TTC, EP, LK, HC). The coefficient $\lambda_{\mathrm{avg}}$ is set to 6.0. The value of other coefficients $\lambda^{(\mathrm{mp})}$ and $\lambda^{(\mathrm{ma})}$ is shown in Tab. 8.

Table 8: The inference coefficients on each metric of NAVSIM. “Mul” denotes the multiplied penalties, and “Avg” denotes the weighted averages.

<table><tbody><tr><td></td><td colspan="4">Mul</td><td colspan="4">Avg</td></tr><tr><td></td><td>NC</td><td>DAC</td><td>DDC</td><td>TLC</td><td>EP</td><td>TTC</td><td>LK</td><td>HC</td></tr><tr><td>Coefficient</td><td>0.5</td><td>0.5</td><td>0.3</td><td>0.1</td><td>5.0</td><td>5.0</td><td>2.0</td><td>1.0</td></tr></tbody></table>

Training Details   HAD is trained on 8 NVIDIA A100 GPUs for 85 epochs. We employ the AdamW optimizer with an initial learning rate of $6\times 10^{-4}$. The learning rate follows a Cosine Annealing schedule with Warmup steps (WarmupCosLR). During the first 3 epochs, the learning rate increases linearly from 0 to $6\times 10^{-4}$, then it decays following a cosine curve to $1\times 10^{-6}$ in the remaining epochs. The coefficients for the model loss functions are listed in Tab. 9.

Table 9: Coefficients in different loss functions.

| Equ. No. | Coefficient | Value |
| --- | --- | --- |
| 19 | $\lambda_{\mathrm{box}},\;\lambda_{\mathrm{label}},\;\lambda_{\mathrm{bev}}$ | 1.0,  10.0,  14.0 |
| 22 | $\lambda_{\mathrm{reg}},\;\lambda_{\mathrm{cls}}$ | 8.0,  10.0 |
| 27 | $\lambda_{\mathrm{dist}},\;\lambda_{\mathrm{safe}},\;\lambda_{\mathrm{rl}}$ | 10.0,  1.0,  1.0 |
| 28 | $\qquad\lambda_{\mathrm{percept}},\;\lambda_{\mathrm{global}},\;\lambda_{\mathrm{local}}\qquad$ | 1.0,  12.0,  12.0 |

## Appendix D Supplementary Experimental Results

### D.1 Parameters and Inference Speed

We compare the number of parameters and inference speed of different approaches. The result is shown in Tab. 10. HAD achieves an inference speed of about 30 FPS on a single NVIDIA A100 GPU with 63M parameters, making it capable of real-time inference.

Table 10: Comparison of parameter number and inference speed. “\*” indicates the EPDMS result is from our re-implementation.

| Model | Input | EPDMS $\uparrow$ | Param. $\downarrow$ | FPS $\uparrow$ |
| --- | --- | --- | --- | --- |
| Transfuser \[chitta2022transfuser\] | C+L | 76.7 | 56M | 56.4 |
| HydraMDP++ \[li2024hydramdp\_pp\] | C | 81.4 | 53M | 32.0 |
| DriveSuprim \[yao2025drivesuprim\] | C | 83.1 | 61M | 27.2 |
| DiffusionDrive \[liao2024diffusiondrive\] | C+L | 87.5\* | 61M | 42.1 |
| HAD | C+L | 88.6 | 63M | 30.4 |
| HAD-L | C | 88.5 | 63M | 29.4 |

### D.2 Performance on NavHard

Tab. 11 presents the performance of various methods on the NavHard benchmark \[cao2025pseudo\]. NavHard is a highly challenging open-loop evaluation benchmark for end-to-end autonomous driving built upon NAVSIM. It adopts a two-stage evaluation protocol: the first stage consists of standard NAVSIM open-loop testing in real-world scenarios, while the second stage utilizes 3D Gaussian Splatting to synthesize diverse difficult scenarios, such as abnormal driving behaviors, to test model performance.

In Tab. 11, HAD-L achieves 32.3 EPDMS on the NavHard benchmark, surpassing DiffusionDrive \[yao2025drivesuprim\] and Latent Transfuser \[chitta2022transfuser\], which also utilizes Transfuser dual-branch backbone. However, a significant gap remains when compared to methods such as DriveSuprim \[yao2025drivesuprim\] and GTRS-Dense \[li2025generalized\].

The reason for this performance gap lies in the high sensitivity of the backbone to sensor noise. HAD-L adopts the Transfuser-based dual-branch backbone that produces BEV feature maps by fusing image and LiDAR features. As illustrated in Fig. 5, the images synthesized by the Gaussian Splatting model in the second stage of NavHard are imperfect and contain noticeable visual noise, which makes it difficult for the Transfuser backbone to extract representative BEV features. This finding provides a valuable insight for future research. Future works may explore end-to-end planning architectures that do not explicitly rely on BEV feature maps to enhance robustness when facing extreme weather or high-noise sensor inputs.

Table 11: Result on NavHard. “\*” indicates that the model uses a dual-branch backbone similar to Transfuser.

<table><tbody><tr><td>Method</td><td>Backbone</td><td>Stage</td><td>NC</td><td>DAC</td><td>DDC</td><td>TLC</td><td>EP</td><td>TTC</td><td>LK</td><td>HC</td><td>EC</td><td>EPDMS</td></tr><tr><td>PDM-Closed <cite>[dauner2023parting]</cite></td><td>-</td><td>Stage 1 Stage 2</td><td>94.4 88.1</td><td>98.8 90.6</td><td>100 96.3</td><td>99.5 98.5</td><td>100 100</td><td>93.5 83.1</td><td>99.3 73.7</td><td>87.7 91.5</td><td>36.0 25.4</td><td>51.3</td></tr><tr><td>LTF <cite>[chitta2022transfuser]</cite></td><td>ResNet34*</td><td>Stage 1 Stage 2</td><td>96.2 77.7</td><td>79.5 70.2</td><td>99.1 84.2</td><td>99.5 98.0</td><td>84.1 85.1</td><td>95.1 75.6</td><td>94.2 45.4</td><td>97.5 95.7</td><td>79.1 75.9</td><td>23.1</td></tr><tr><td>GTRS-Dense <cite>[li2025generalized]</cite></td><td>V2-99</td><td>Stage 1 Stage 2</td><td>98.7 91.4</td><td>95.8 89.2</td><td>99.4 94.4</td><td>99.3 98.8</td><td>72.8 69.5</td><td>98.7 90.1</td><td>95.1 54.6</td><td>96.9 94.1</td><td>40.4 49.7</td><td>41.7</td></tr><tr><td rowspan="2">DriveSuprim <cite>[yao2025drivesuprim]</cite></td><td>ResNet34</td><td>Stage 1 Stage 2</td><td>97.2 88.6</td><td>96.2 86.0</td><td>99.3 89.4</td><td>99.6 98.4</td><td>71.4 74.7</td><td>96.7 86.5</td><td>94.7 55.3</td><td>97.3 97.7</td><td>47.1 59.4</td><td>39.5</td></tr><tr><td>V2-99</td><td>Stage 1 Stage 2</td><td>98.9 87.9</td><td>95.1 88.8</td><td>99.2 89.6</td><td>99.6 98.8</td><td>76.1 80.3</td><td>99.1 86.0</td><td>94.7 53.5</td><td>97.6 97.1</td><td>54.2 56.1</td><td>42.1</td></tr><tr><td>DiffusionDrive <cite>[liao2024diffusiondrive]</cite></td><td>ResNet34*</td><td>Stage 1 Stage 2</td><td>96.8 80.1</td><td>86.0 72.8</td><td>98.8 84.4</td><td>99.3 98.4</td><td>84.0 85.9</td><td>95.8 76.6</td><td>96.7 46.4</td><td>97.6 96.3</td><td>66.7 40.5</td><td>27.5</td></tr><tr><td>HAD-L</td><td>ResNet34*</td><td>Stage 1 Stage 2</td><td>96.2 80.2</td><td>90.4 75.2</td><td>98.6 83.7</td><td>99.3 98.4</td><td>83.6 86.6</td><td>95.8 77.0</td><td>96.0 47.2</td><td>97.8 95.9</td><td>78.2 70.5</td><td>32.3</td></tr></tbody></table>

![[x5 23.png|Refer to caption]]

Figure 5: Synthesized data from the second stage of the NavHard benchmark.

![[x6 20.png|Refer to caption]]

Figure 6: Visualization result on the NAVSIM dataset. In each example, the green trajectory represents the ground truth from the human expert, the red trajectories are generated by other methods, and the blue trajectory is produced by HAD. Zoom in for a better view.

### D.3 Visualization Results

We provide the visualization results of HAD in Fig. 6 and Fig. 7.

Results on NAVSIM   Fig. 6 illustrates a qualitative comparison between Hydra-MDP++ \[li2024hydramdp\_pp\], DriveSuprim \[yao2025drivesuprim\], and HAD across various challenging scenarios. We validate the superiority of our proposed method in complex scene interaction and long-distance driving robustness.

In the complex interaction scenarios shown in the first two rows of Fig. 6, the ego vehicle must generate precise trajectories to avoid potential collisions. The red trajectories generated by selection-based methods exhibit clear decision-making limitations, resulting in insufficient safety margins from other vehicles. In contrast, HAD demonstrates superior environmental understanding capability. The blue trajectories generated by our method maintain appropriate safety distances from other vehicles and closely align with the expert trajectories. This indicates that the Hierarchical Diffusion Policy can effectively identify the optimal trajectory within the local driving region.

![[x7 16.png|Refer to caption]]

Figure 7: Visualization result on the HUGSIM dataset.

In the long-distance straight-line scenario shown in the third row, HAD demonstrates excellent driving direction maintenance. The blue trajectory remains near the lane centerline even at long range, without accumulating decision-making bias over time. It validates that by applying the MDPO algorithm, the model learns a policy with greater long-term robustness than pure supervised learning, effectively preventing horizontal drift in long-distance planning.

Results on HUGSIM   Fig. 7 visualizes the output of HAD in the closed-loop HUGSIM benchmark, highlighting the model’s real-time avoidance capabilities during dynamic interactive maneuvers.

In the scenarios shown in the first two rows of Fig. 7, when the ego vehicle is driving in a single lane and encounters a slow-moving vehicle ahead, HAD accurately identifies the available space on the right and plans an avoidance trajectory that shifts toward that side. In the more complex intersection scenarios presented in the bottom two rows, HAD demonstrates strong decision-making capability when facing oncoming traffic. The model adaptively adjusts its speed and direction, generating a smooth bypassing trajectory.

The robust driving behavior observed in closed-loop planning further validates the effectiveness of the Metric-Decoupled Policy Optimization algorithm. The reinforcement learning strategy enables the model to make decisions that comply with multiple driving rules, thereby ensuring safe and smooth vehicle behavior in highly dynamic environments.
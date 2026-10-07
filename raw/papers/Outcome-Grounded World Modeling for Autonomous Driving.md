---
title: "Outcome-Grounded World Modeling for Autonomous Driving"
source: "https://arxiv.org/html/2609.36438v1"
author:
published:
created: 2026-10-06
description:
tags:
  - "clippings"
---
Jieyuan Pei <sup>1,2,3∗‡</sup>   Meiyi Lu <sup>4∗</sup>   Sining Ang <sup>5</sup>   Yubo Zhao <sup>6</sup>   Zhangyi Hu <sup>3</sup>   Mingwei Xu <sup>7</sup>   Haokai Ding <sup>8</sup>  
Wei Li <sup>9</sup>  Zihan You <sup>1,10</sup>  Jianwei Zheng <sup>9</sup>  Li Yu <sup>11</sup>  Yifeng Pan <sup>11</sup>  Ji Tao <sup>11</sup>  Rongjunchen Zhang <sup>2</sup>  Yan Wang <sup>1†</sup> <sup>1</sup> Institute for AI Industry Research (AIR), Tsinghua University   <sup>2</sup> HiThink Research  
<sup>3</sup> The Hong Kong University of Science and Technology (Guangzhou)   <sup>4</sup> Zhejiang University  
<sup>5</sup> University of Science and Technology of China   <sup>6</sup> SMBU   <sup>7</sup> University of Washington  
<sup>8</sup> Mohamed bin Zayed University of Artificial Intelligence   <sup>9</sup> Zhejiang University of Technology   <sup>10</sup> Southeast University  
<sup>11</sup> Changan Automobile     <sup>∗</sup> Equal contribution   <sup>†</sup> Corresponding author   <sup>‡</sup> Work done during an internship [Project Page](https://guobapei.github.io/World4Scorer/)       [Hugging Face](https://huggingface.co/pei2333/World4Scorer)       [Code](https://github.com/GuobaPei/World4Scorer)

###### Abstract

Autonomous driving requires choosing a safe and efficient plan as surrounding traffic evolves. Generate-and-select planners propose multiple trajectories and score them for execution, and they have outperformed representative direct-prediction baselines on NAVSIM. Their scorer must compare plans that were never executed. Driving logs record the future of only the executed trajectory, so matching the logged future can leave predictions for the alternatives unconstrained; a simulator, in contrast, can label the outcome of every candidate. We introduce World4Scorer, which builds the scorer as a trajectory-conditioned JEPA-style predictor: it predicts a state for each candidate and reads the candidate’s scores from that state. Simulator outcome labels supervise the states of all candidates, and the observed future of the executed trajectory anchors the predictor to real scene evolution. Because one predictor produces every candidate’s state, the anchor can constrain shared parameters used to score unexecuted plans, while the future itself is needed only during training. Generated candidates mostly score well, so a scene-matched bank adds low-scoring plans to the outcome supervision; framewise choices can conflict, so inertial re-ranking keeps consecutive selections consistent. World4Scorer achieves state-of-the-art NAVSIM-v2 performance and a strong adapted-system result on closed-loop Bench2Drive. With the LeWM world model and planning budget fixed, outcome-based scoring also improves manipulation planning on the OGBench-Cube benchmark.

![[teaser_paradigms_compact.png|Refer to caption]]

Figure 1: Gradient routing in generator–scorer planning. (a) Outcome labels train the scorer. (b) A world model adds future prediction, but only the executed trajectory has an observed future. (c) In World4Scorer, outcome labels supervise the state of every candidate, and the observed future anchors the predictor.

## 1 Introduction

Autonomous driving requires turning sensor observations into safe, efficient, and comfortable motion. Direct trajectory-prediction methods such as TransFuser [^5] and UniAD [^13] map scene features to a driving plan. Recent generate-and-select planners have achieved stronger performance than representative direct-prediction baselines on NAVSIM [^29] [^28] [^21]. They use a trajectory generator to propose alternatives and a learned scorer to choose one for execution (Figure 1a). To benefit from these alternatives, the scorer must identify which plan best balances safety, progress, and comfort.

Scoring a plan requires assessing its consequences as surrounding traffic evolves. World models provide predictive features for planning [^23] [^11] and candidate evaluation [^24] (Figure 1b). Their training faces a supervision gap: each driving log records the future of the executed trajectory, while alternative trajectories have no corresponding future observations. Matching the logged future can leave predictions for unobserved alternatives unconstrained, even though these predictions decide which plan is selected.

Candidate outcomes fill this gap. Simulation can label whether each alternative plan collides, leaves the drivable area, or makes progress, without an observed visual future for that plan. When a predictor produces the state from which each candidate is scored, these labels supervise the state of every candidate, including plans that were never executed. The observed future serves as an anchor: it supervises the state of the executed trajectory and ties the shared predictor to real scene evolution.

We introduce World4Scorer, an *outcome-grounded world modeling* framework in which a trajectory-conditioned predictor is the scorer (Figure 1c). Following the joint-embedding predictive architecture (JEPA) [^2] [^3], the predictor maps each candidate to a state; score heads read candidate outcomes from this state, and a visual readout predicts the future features of the executed trajectory. The core is this shared predictor, trained by both candidate outcomes and the observed future. As support, a perturbed candidate bank widens outcome coverage, and training-free inertial re-ranking checks continuity with the previous plan.

- Outcome-grounded world modeling. We build the candidate scorer as a trajectory-conditioned predictor. Simulator outcomes supervise the predicted state of every generated and bank candidate, including plans that were never executed, while the observed future of the executed plan anchors the shared predictor to real scene evolution.
- Complementary supervision. Outcome labels reach every candidate; the observed future reaches one trajectory yet constrains the shared predictor. In controlled tests, an observed-future target improves planning but a current-frame target does not, mismatched outcome labels degrade ranking, and reassigning candidate states lowers PDMS in every trial.
- Evidence across benchmarks and domains. World4Scorer achieves state-of-the-art performance on NAVSIM-v2 and leads the compared methods on NAVSIM-v1. Cross-setting tests extend these results: an adapted system is strong in closed-loop Bench2Drive, and outcome-based scoring improves manipulation with a frozen world model.

## 2 Related Work

#### End-to-end autonomous driving.

TransFuser [^5] and UniAD [^13] regress a trajectory from scene features. Candidate-based planners compare alternatives from vocabularies or generators. VADv2 [^19] predicts probabilities over a fixed vocabulary. DiffusionDrive [^29] refines anchors and selects by learned confidence. GTRS [^28] scores vocabulary and generated candidates, and DriveSuprim [^44] filters candidates from coarse to fine. CLOVER [^1] trains its generator on evaluator-filtered pseudo-experts and its scorer on evaluator sub-scores. DrivoR [^21] learns proposals and simulator-supervised scores with separate decoders. We follow its generator–scorer framework and make the scorer a predictor trained by simulator labels, anchored by observed futures.

#### Predictive representations for planning.

Future prediction supports both trajectory generation [^23] [^40] [^11] and candidate evaluation [^24] [^20] [^48]. Drive-JEPA [^38] uses video-JEPA pretraining to provide predictive features for planning. We focus on the supervision available during candidate comparison: observed futures cover executed trajectories, whereas simulator outcomes cover every candidate; World4Scorer trains one predictor with both.

![[Figure2_lewm.png|Refer to caption]]

Figure 2: Overview of World4Scorer. (a) The trajectory generator proposes 64 candidates, scored by World4Scorer before inertial re-ranking. (b) The visual readout and score heads train the shared predictor: future observations supervise visual prediction for the executed trajectory; outcome labels supervise scoring for all generated candidates and 16 bank trajectories.

## 3 World4Scorer

World4Scorer follows a generator–scorer framework: it proposes trajectories, predicts a state for each candidate, and scores these states before inertial re-ranking (Figure 2). We use “world model” in a decision-focused sense: latent states carry candidate outcomes, not full futures.

### 3.1 Overview

At inference (Figure 2a), DINOv2 [^33] with LoRA [^12] encodes four camera views into scene tokens $s_{t}$. A transformer generates $N=64$ trajectories, $\mathcal{C}_{t}=\{\tau_{t,i}\}_{i=1}^{N}$. The predictor maps each trajectory query to a state using the scene tokens. Score heads read each state together with the ego state. Inertial re-ranking then adds compatibility with the previous plan.

At training time (Figure 2b), winner-take-all imitation supervises the generator. Outcome labels supervise all 64 generated candidates and 16 trajectories sampled from a scene-matched bank. A separate query for the logged trajectory predicts its observed visual future. These three query sets use the same predictor, allowing visual prediction and outcome scoring to train shared parameters.

### 3.2 Shared Prediction and Scoring

An action encoder $\phi$ maps each candidate trajectory to a query token. The shared predictor $P$ (Latent World Model in Figure 2) processes all candidate queries together with the scene tokens:

$$
q_{t,i}=\phi(\operatorname{sg}[\tau_{t,i}]),\quad Q_{t}=[q_{t,1},\ldots,q_{t,N}],\quad Z_{t}=P(Q_{t},s_{t}).
$$

Its four transformer blocks exchange information among candidates through self-attention and incorporate scene information through cross-attention, followed by output normalization. Row $z_{t,i}$ of $Z_{t}$ is the predicted state for candidate $i$, and $\operatorname{sg}$ stops gradients.

The score heads $H$ read each candidate state after adding the encoded ego state $e_{t}$: $\boldsymbol{\ell}_{t,i}=H(z_{t,i}+e_{t})$. They predict collision avoidance (NC), drivable-area compliance (DAC), driving direction (DDC), time-to-collision (TTC), progress (EP), and comfort (C). Let $p_{t,i,m}=\sigma(\ell_{t,i,m})$, $\mathcal{G}=\{\mathrm{NC,DAC,DDC}\}$, and $\mathcal{A}=\{\mathrm{TTC,EP,C}\}$. The ranking score is

$$
\hat{J}_{t,i}=\sum_{m\in\mathcal{G}}\alpha_{m}\log p_{t,i,m}+\log\!\left(\sum_{m\in\mathcal{A}}\beta_{m}p_{t,i,m}\right).
$$

Here $\alpha_{m}$ and $\beta_{m}$ are per-benchmark weights (Appendix A.2). The highest-scoring candidate defines the framewise choice; inertial re-ranking further incorporates previous-plan compatibility.

Detaching candidate coordinates prevents scoring gradients from reaching the generator through its trajectory outputs. The scoring loss still trains $\phi$, $P$, $H$, and the scene features.

### 3.3 Outcome-Grounded Predictive Learning

Only the executed trajectory has an observed future; generated and bank candidates lack future observations but have outcome labels. Outcome labels supervise every candidate state, and the observed future anchors the predictor through the executed trajectory (Figure 2b).

Trajectory-conditioned JEPA. Following JEPA [^2] [^3], the predictor takes scene tokens and the logged trajectory query. Its target is the frozen DINOv2 embedding of the observed front view two seconds after the current frame, mean-pooled to 384 dimensions:

$$
z_{t}^{\rm exec}=P([\phi(\tau_{t}^{\rm exec})],\operatorname{sg}[s_{t}]),\qquad u_{t}^{\rm fut}=f_{\rm DINO}(o_{t+2\mathrm{s}}).
$$

The linear visual readout $g$ predicts this feature with a cosine loss:

$$
\mathcal{L}_{\rm fut}=\frac{1}{M}\sum_{n=1}^{M}\left[1-\operatorname{sim}_{\cos}\bigl(g(z_{n}^{\rm exec}),u_{n}^{\rm fut}\bigr)\right].
$$

Here $n$ indexes $M$ scenes. With scene tokens detached, this loss updates $\phi$, $P$, and $g$; at deployment, the predictor needs only current observations and candidate trajectories. Because every query passes through the same $P$, this single target also constrains the parameters that produce each candidate’s state. The self-attention query and key projections receive no gradient from this single-query loss and are trained by the outcome losses alone (Appendix A.2).

Learning from candidate outcomes. Outcome labels $y_{i}$ supervise $H(z_{i}+e)$:

$$
\mathcal{L}_{\rm out}=\mathcal{L}_{\rm score}+\lambda_{\rm bank}\mathcal{L}_{\rm bank}=\sum_{i}w_{i}\,\ell_{\rm out}\bigl(H(z_{i}+e),y_{i}\bigr),
$$

where $i$ covers generated and bank candidates, $\ell_{\rm out}$ uses each set’s loss components, and $w_{i}$ includes averaging and bank weighting (Section 3.4). Outcome supervision thus reaches every candidate state, including those without visual targets, and requires distinctions in safety, progress, and comfort. A scene supplies outcome labels for up to 80 candidates and a future target for one.

Why future supervision still matters. Models that fit the available outcome labels may still disagree on other candidates [^10], including which one to select [^32]. We bound this variation with a local generalized Gauss–Newton model around a fitted $\theta_{0}$, keeping each predictor call in its own context.

###### Proposition 1 (Future supervision and decision stability)

Let $\mu,\lambda_{\rm fut},\epsilon>0$, and let the executed-query readout be differentiable and nonzero at $\theta_{0}$. Let $G_{\rm out}$ (with a trust-region term $\mu I$) and $G_{\rm fut}$ be the generalized Gauss–Newton curvatures of the BCE outcome and cosine future losses at $\theta_{0}$, and let $\rho_{ij}(G)$ be the largest first-order change $|c_{ij}^{\top}\delta|$ over $\delta^{\top}G\delta\leq\epsilon^{2}$. Then

$$
\rho_{ij}^{\rm out+fut}=\rho_{ij}(G_{\rm out}+\lambda_{\rm fut}G_{\rm fut})\leq\rho_{ij}(G_{\rm out})=\rho_{ij}^{\rm out},
$$

strictly if and only if $J_{\rm fut}G_{\rm out}^{-1}c_{ij}\neq 0$, where $c_{ij}=\nabla_{\theta}(\hat{J}_{i}-\hat{J}_{j})$ and $J_{\rm fut}$ is the Jacobian of the normalized readout. If $\hat{J}_{i}-\hat{J}_{j}>\rho_{ij}^{\rm out+fut}$, the linearized ordering $i\succ j$ holds throughout.

Figure 3 shows how future constraints can remove local disagreement on candidate choice. A smaller variation range makes a candidate’s lead harder to reverse within the allowed neighborhood. The future loss does not constrain every unexecuted candidate uniformly: it narrows a comparison only along directions that shared parameters couple to the executed query’s readout. Appendix C.4 proves this local result; Section 4.3 evaluates the learning effects predicted by this mechanism.

Figure 3: Complementary supervision. Two score surfaces fit the outcome labels but favor different candidates. In this example, future constraints remove that disagreement while allowing different predictions elsewhere.

### 3.4 Outcome Supervision across Candidates

Imitation training yields mostly expert-like, high-scoring candidates (light bars in Figure 4), so adverse outcomes such as collisions and drivable-area violations form a small share of the training examples.

Following CLOVER [^1], we use a scene-matched bank of perturbed trajectories with precomputed simulator labels. The bank covers a broader score range, including unsafe plans (dark bars). We sample $16{}$ bank trajectories per step to complement the generated candidates.

Figure 4: Candidate bank. The bank adds low-scoring plans.

Both sets use separate calls to the same $\phi$, $P$, and $H$. For generated candidates, $\mathcal{L}_{\rm score}$ sums binary cross-entropy over six components, using soft targets for progress. For bank candidates, $\mathcal{L}_{\rm bank}$ averages binary cross-entropy over five non-progress components and adds $0.1$ times the progress L1 loss. Each component loss is averaged over scenes and candidates in its set.

Winner-take-all L1 imitation, $\mathcal{L}_{\rm WTA}$, supervises the candidate closest to the logged trajectory. Joint training combines all four losses:

$$
\mathcal{L}=\mathcal{L}_{\rm WTA}+\mathcal{L}_{\rm score}+\lambda_{\rm fut}\mathcal{L}_{\rm fut}+\lambda_{\rm bank}\mathcal{L}_{\rm bank},
$$

with $\lambda_{\rm fut}=1$ and $\lambda_{\rm bank}=0.5$. We down-weight the bank loss to limit its influence on shared features while retaining supervision from trajectories outside the generated candidate set.

### 3.5 Inertial Re-Ranking

We observe abrupt changes between plans selected in adjacent frames. Framewise scoring can favor a high-scoring trajectory that is incompatible with the previous choice; similar candidate scores can also make selection sensitive to small changes. We therefore check each candidate against the previous plan $\pi_{t-1}$ and use this compatibility to re-rank the current candidates.

For NAVSIM-v2, we instantiate the check with the benchmark’s two-frame extended-comfort (EC) criterion [^4]. A deterministic linear-quadratic-regulator (LQR) controller tracks the previous plan and each current candidate with a kinematic bicycle model, producing ego-state rollouts. After aligning the two rollouts on their common horizon, we compare their acceleration, jerk, yaw-rate, and yaw-acceleration profiles. The candidate is marked compatible when all four RMS differences fall below their respective benchmark thresholds, giving the binary flag

$$
c_{t,i}=c_{\rm EC}(\pi_{t-1},\tau_{t,i})\in\{0,1\}.
$$

Logged human plans pass this check on 90.1% of adjacent frame pairs (Appendix E.1).

Figure 5: Consecutive plans through a right turn. Frames are $0.5$  s apart. In (b), re-ranking selects a plan closer to the preceding choice and the human trajectory. Both views share map coordinates.

The check uses only the previous plan, current candidates, and controller rollouts, so it requires no future observations or labels at deployment. Within each log, we process frames chronologically and carry the selected plan forward. The first frame uses the learned score alone.

We add the compatibility result as a log-domain penalty to the learned utility:

$$
\Psi_{t,i}\coloneqq\log(0.01+0.99\,c_{t,i}),\qquad i_{t}^{\star}\in\arg\max_{1\leq i\leq N}\bigl[\hat{J}_{t,i}+\lambda_{\rm state}\Psi_{t,i}\bigr].
$$

With $\lambda_{\rm state}=1$, compatible candidates receive no penalty and retain their learned-score ordering; incompatible candidates receive $\Psi_{t,i}=-\log 100$. Re-ranking adds no learned parameters or training loss. We report both framewise and re-ranked results on NAVSIM-v2.

## 4 Experiments

We organize the experiments around four research questions:

- RQ1. *Does World4Scorer improve driving performance on NAVSIM and Bench2Drive?*
- RQ2. *What do future targets, candidate outcomes, and inertial re-ranking contribute?*
- RQ3. *Do candidate-specific predictive states guide the model’s trajectory selection?*
- RQ4. *Can outcome-based scoring improve manipulation with a frozen world model?*

![[b2d_main.png|Refer to caption]]

Table 1: Planning performance on NAVSIM-v1 and NAVSIM-v2 (%). Input C: cameras, L: LiDAR; superscripts 1 and 2 denote native NAVSIM-v1 and NAVSIM-v2. Best and second-best results are bold and underlined. We evaluate with the latest official NAVSIM devkits.

[^1]: Sining Ang, Yuguang Yang, Canyu Chen, and Yan Wang. CLOVER: Closed-loop value estimation and ranking for end-to-end autonomous driving planning. *arXiv preprint arXiv:2605.15120*, 2026.

[^2]: Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 15619–15629, 2023.

[^3]: Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, and Nicolas Ballas. Revisiting feature prediction for learning visual representations from video. *Transactions on Machine Learning Research*, 2024.

[^4]: Wei Cao, Marcel Hallgarten, Tianyu Li, Daniel Dauner, Xunjiang Gu, Caojun Wang, Yakov Miron, Marco Aiello, Hongyang Li, Igor Gilitschenski, Boris Ivanovic, Marco Pavone, Andreas Geiger, and Kashyap Chitta. Pseudo-simulation for autonomous driving. In *Conference on Robot Learning*, volume 305 of *Proceedings of Machine Learning Research*, pp. 4709–4722, 2025.

[^5]: Kashyap Chitta, Aditya Prakash, Bernhard Jaeger, Zehao Yu, Katrin Renz, and Andreas Geiger. TransFuser: Imitation with transformer-based sensor fusion for autonomous driving. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 45(11):12878–12895, 2023. doi: 10.1109/TPAMI.2022.3200245.

[^6]: Timothée Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need registers. In *International Conference on Learning Representations*, 2024.

[^7]: Daniel Dauner, Marcel Hallgarten, Tianyu Li, Xinshuo Weng, Zhiyu Huang, Zetong Yang, Hongyang Li, Igor Gilitschenski, Boris Ivanovic, Marco Pavone, Andreas Geiger, and Kashyap Chitta. NAVSIM: Data-driven non-reactive autonomous vehicle simulation and benchmarking. In *Advances in Neural Information Processing Systems, Datasets and Benchmarks Track*, volume 37, 2024.

[^8]: Lan Feng, Yang Gao, Eloi Zablocki, Quanyi Li, Wuyang Li, Sichao Liu, Matthieu Cord, and Alexandre Alahi. RAP: 3D rasterization augmented end-to-end planning. In *International Conference on Learning Representations*, 2026.

[^9]: Tanner Fiez, Lalit Jain, Kevin Jamieson, and Lillian Ratliff. Sequential experimental design for transductive linear bandits. In *Advances in Neural Information Processing Systems*, volume 32, 2019.

[^10]: Christopher Grimm, André Barreto, Satinder P. Singh, and David Silver. The value equivalence principle for model-based reinforcement learning. In *Advances in Neural Information Processing Systems*, volume 33, 2020.

[^11]: Yufeng Hong, Xiaotian Zhou, Yingyan Li, Xiangpo Zhou, Lin Liu, Yadan Luo, Shaoqing Xu, Lei Yang, and Ziying Song. Drivefuture: Future-aware latent world models for autonomous driving. *arXiv preprint arXiv:2605.09701*, 2026.

[^12]: Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In *International Conference on Learning Representations*, 2022.

[^13]: Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai Wang, Lewei Lu, Xiaosong Jia, Qiang Liu, Jifeng Dai, Yu Qiao, and Hongyang Li. Planning-oriented autonomous driving. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 17853–17862, 2023.

[^14]: Feiyang Jia, Lin Liu, Ziying Song, Caiyan Jia, Hangjun Ye, Xiaoshuai Hao, and Long Chen. DriveWorld-VLA: Unified latent-space world modeling with vision-language-action for autonomous driving. In *International Conference on Machine Learning*, 2026.

[^15]: Xiaosong Jia, Yulu Gao, Li Chen, Junchi Yan, Patrick Langechuan Liu, and Hongyang Li. DriveAdapter: Breaking the coupling barrier of perception and planning in end-to-end autonomous driving. In *IEEE/CVF International Conference on Computer Vision*, 2023a.

[^16]: Xiaosong Jia, Penghao Wu, Li Chen, Jiangwei Xie, Conghui He, Junchi Yan, and Hongyang Li. Think twice before driving: Towards scalable decoders for end-to-end autonomous driving. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2023b.

[^17]: Xiaosong Jia, Zhenjie Yang, Qifeng Li, Zhiyuan Zhang, and Junchi Yan. Bench2Drive: Towards multi-ability benchmarking of closed-loop end-to-end autonomous driving. In *Advances in Neural Information Processing Systems, Datasets and Benchmarks Track*, volume 37, pp. 819–844, 2024.

[^18]: Bo Jiang, Shaoyu Chen, Qing Xu, Bencheng Liao, Jiajie Chen, Helong Zhou, Qian Zhang, Wenyu Liu, Chang Huang, and Xinggang Wang. VAD: Vectorized scene representation for efficient autonomous driving. In *IEEE/CVF International Conference on Computer Vision*, pp. 8340–8350, 2023.

[^19]: Bo Jiang, Shaoyu Chen, Hao Gao, Bencheng Liao, Qian Zhang, Wenyu Liu, and Xinggang Wang. VADv2: End-to-end vectorized autonomous driving via probabilistic planning. In *International Conference on Learning Representations*, 2026. URL [https://arxiv.org/abs/2402.13243](https://arxiv.org/abs/2402.13243).

[^20]: Jungho Kim, Jiyong Oh, Seunghoon Yu, Hongjae Shin, Donghyuk Kwak, and Jun Won Choi. Safedrive: Fine-grained safety reasoning for end-to-end driving in a sparse world. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 24854–24864, 2026. arXiv:2602.18887.

[^21]: Ellington Kirby, Alexandre Boulch, Yihong Xu, Yuan Yin, Gilles Puy, Éloi Zablocki, Andrei Bursuc, Spyros Gidaris, Renaud Marlet, Florent Bartoccioni, Anh-Quan Cao, Nermin Samet, Tuan-Hung Vu, and Matthieu Cord. Driving on registers. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 32058–32069, 2026. arXiv:2601.05083.

[^22]: Ilya Kostrikov, Ashvin Nair, and Sergey Levine. Offline reinforcement learning with implicit q-learning. In *International Conference on Learning Representations*, 2022.

[^23]: Yingyan Li, Lue Fan, Jiawei He, Yuqi Wang, Yuntao Chen, Zhaoxiang Zhang, and Tieniu Tan. Enhancing end-to-end autonomous driving with latent world model. In *International Conference on Learning Representations*, 2025a.

[^24]: Yingyan Li, Yuqi Wang, Yang Liu, Jiawei He, Lue Fan, and Zhaoxiang Zhang. End-to-end driving with online trajectory evaluation via BEV world model. In *IEEE/CVF International Conference on Computer Vision*, pp. 27137–27146, 2025b.

[^25]: Yingyan Li, Shuyao Shang, Weisong Liu, Bing Zhan, Haochen Wang, Yuqi Wang, Yuntao Chen, Xiaoman Wang, Yasong An, Chufeng Tang, Lu Hou, Lue Fan, and Zhaoxiang Zhang. Drivevla-w0: World models amplify data scaling law in autonomous driving. In *International Conference on Learning Representations*, 2026a. arXiv:2510.12796.

[^26]: Yongkang Li, Kaixin Xiong, Xiangyu Guo, Fang Li, Sixu Yan, Gangwei Xu, Lijun Zhou, Long Chen, Haiyang Sun, Bing Wang, Kun Ma, Guang Chen, Hangjun Ye, Wenyu Liu, and Xinggang Wang. ReCogDrive: A reinforced cognitive framework for end-to-end autonomous driving. In *International Conference on Learning Representations*, 2026b. arXiv:2506.08052.

[^27]: Zhenxin Li, Wenhao Yao, Zi Wang, Xinglong Sun, Jingde Chen, Nadine Chang, Maying Shen, Jingyu Song, Zuxuan Wu, Shiyi Lan, and Jose M. Alvarez. ZTRS: Zero-imitation end-to-end autonomous driving with trajectory scoring. *arXiv preprint arXiv:2510.24108v1*, 2025c.

[^28]: Zhenxin Li, Wenhao Yao, Zi Wang, Xinglong Sun, Joshua Chen, Nadine Chang, Maying Shen, Zuxuan Wu, Shiyi Lan, and Jose M. Alvarez. Generalized trajectory scoring for end-to-end multimodal planning. *arXiv preprint arXiv:2506.06664*, 2025d.

[^29]: Bencheng Liao, Shaoyu Chen, Haoran Yin, Bo Jiang, Cheng Wang, Sixu Yan, Xinbang Zhang, Xiangyu Li, Ying Zhang, Qian Zhang, and Xinggang Wang. Diffusiondrive: Truncated diffusion model for end-to-end autonomous driving. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 12037–12047, 2025. doi: 10.1109/CVPR52734.2025.01124.

[^30]: Hongbo Lu, Liang Yao, Chenghao He, Haoyu Wang, Xiang Gu, Xianfei Li, Wenlong Liao, Tao He, and Pai Peng. The DAWN of world-action interactive models. *arXiv preprint arXiv:2605.11550*, 2026.

[^31]: Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, and Randall Balestriero. LeWorldModel: Stable end-to-end joint-embedding predictive architecture from pixels. *arXiv preprint arXiv:2603.19312*, 2026.

[^32]: Jayanta Mandi, Víctor Bucarey, Maxime Mulamba Ke Tchomba, and Tias Guns. Decision-focused learning: Through the lens of learning to rank. In *Proceedings of the 39th International Conference on Machine Learning*, volume 162 of *Proceedings of Machine Learning Research*, pp. 14935–14947, 2022.

[^33]: Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jégou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. *Transactions on Machine Learning Research*, 2024.

[^34]: Seohong Park, Kevin Frans, Benjamin Eysenbach, and Sergey Levine. OGBench: Benchmarking offline goal-conditioned RL. In *International Conference on Learning Representations*, 2025.

[^35]: Zihao Sheng, Xin Ye, Jingru Luo, Sikai Chen, and Liu Ren. ExploreVLA: Dense world modeling and exploration for end-to-end autonomous driving. In *European Conference on Computer Vision*, 2026.

[^36]: Vlad Sobal, Wancong Zhang, Kyunghyun Cho, Randall Balestriero, Tim G. J. Rudner, and Yann LeCun. Learning from reward-free offline data: A case for planning with latent dynamics models. In *Advances in Neural Information Processing Systems*, 2025.

[^37]: Ziying Song, Caiyan Jia, Lin Liu, Lei Yang, Shengkai Zhang, Feiyang Jia, Fengda Zhao, Peiliang Wu, Shaoqing Xu, Chen Lv, and Yadan Luo. GraphWorld: Long-horizon planning with world models for end-to-end autonomous driving. *arXiv preprint arXiv:2606.16274*, 2026.

[^38]: Linhan Wang, Zichong Yang, Chen Bai, Guoxiang Zhang, Xiaotong Liu, Xiaoyin Zheng, Xiao-Xiao Long, Chang-Tien Lu, and Cheng Lu. Drive-jepa: Video jepa meets multimodal trajectory distillation for end-to-end driving. *arXiv preprint arXiv:2601.22032*, 2026.

[^39]: Penghao Wu, Xiaosong Jia, Li Chen, Junchi Yan, Hongyang Li, and Yu Qiao. Trajectory-guided control prediction for end-to-end autonomous driving: A simple yet strong baseline. In *Advances in Neural Information Processing Systems*, volume 35, pp. 6119–6132, 2022.

[^40]: Tianze Xia, Yongkang Li, Lijun Zhou, Jingfeng Yao, Kaixin Xiong, Haiyang Sun, Bing Wang, Kun Ma, Guang Chen, Hangjun Ye, Wenyu Liu, and Xinggang Wang. Drivelaw: Unifying planning and video generation in a latent driving world. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 39701–39712, 2026. arXiv:2512.23421.

[^41]: Yifang Xu, Jiahao Cui, Zhihao Zhu, Hanlin Shang, Shan Luan, Mingwang Xu, Feipeng Cai, Neng Zhang, Yaoyi Li, Jia Cai, and Siyu Zhu. WAM-flow: Parallel coarse-to-fine motion planning via discrete flow matching for autonomous driving. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 24918–24928, 2026. arXiv:2512.06112.

[^42]: Chen Yang, Yuhao Wei, Ze Xu, Ziheng Zou, Shuang Liang, Delin Ouyang, Lingfeng Qi, Jie Li, and Guofa Li. Lwdrive: Layer-wise world-model-guided vision-language model planning for autonomous driving. *arXiv preprint arXiv:2606.29879*, 2026a.

[^43]: Pengxuan Yang, Ben Lu, Zhongpu Xia, Chao Han, Yinfeng Gao, Teng Zhang, Kun Zhan, Xianpeng Lang, Yupeng Zheng, and Qichao Zhang. WorldRFT: Latent world model planning with reinforcement fine-tuning for autonomous driving. *Proceedings of the AAAI Conference on Artificial Intelligence*, 40(14):11649–11657, 2026b. doi: 10.1609/aaai.v40i14.38149.

[^44]: Wenhao Yao, Zhenxin Li, Shiyi Lan, Zi Wang, Xinglong Sun, Jose M. Alvarez, and Zuxuan Wu. Drivesuprim: Towards precise trajectory selection for end-to-end planning. *Proceedings of the AAAI Conference on Artificial Intelligence*, 40(14):11910–11918, 2026a. doi: 10.1609/aaai.v40i14.38178.

[^45]: Ziyang Yao, Haochen Liu, Yuncheng Jiang, Zeyu Zhu, Zibin Guo, Jingru Wang, Tianle Liu, Jianwei Cui, Kuiyuan Yang, Hongwei Xie, Jingwei Zhao, Guang Chen, and Hangjun Ye. Discrete-WAM: Unified discrete vision-action token editing for world-policy learning. *arXiv preprint arXiv:2606.05645*, 2026b.

[^46]: Bo Zhao, Xinting Zhao, Naifan Li, Erkang Cheng, and Haibin Ling. UniTeD: Unified temporal diffusion for joint perception and planning in autonomous driving. In *European Conference on Computer Vision*, 2026.

[^47]: Boyang Zheng, Nanye Ma, Shengbang Tong, and Saining Xie. Diffusion transformers with representation autoencoders. In *International Conference on Learning Representations*, 2026.

[^48]: Yupeng Zheng, Pengxuan Yang, Zebin Xing, Qichao Zhang, Yuhang Zheng, Yinfeng Gao, Pengfei Li, Teng Zhang, Zhongpu Xia, Peng Jia, Xianpeng Lang, and Dongbin Zhao. World4drive: End-to-end autonomous driving via intention-aware physical latent world model. In *IEEE/CVF International Conference on Computer Vision*, pp. 28632–28642, 2025.

[^49]: Jialv Zou, Shaoyu Chen, Bencheng Liao, Zhiyu Zheng, Yuehao Song, Lefei Zhang, Qian Zhang, Wenyu Liu, and Xinggang Wang. DiffusionDriveV2: Reinforcement learning-constrained truncated diffusion modeling in end-to-end autonomous driving. *arXiv preprint arXiv:2512.07745*, 2025.
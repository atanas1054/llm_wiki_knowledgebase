---
title: "All Roads Lead to Rome: Incentivizing Divergent Thinking in Vision-Language Models"
source: "https://arxiv.org/html/2604.00479v1"
author:
published:
created: 2026-06-18
description:
tags:
  - "clippings"
---
Xinyu Tian <sup>1</sup>   Shu Zou <sup>1,2</sup>   Zhaoyuan Yang <sup>3</sup>   Mengqi He <sup>1</sup>   Peter Tu <sup>3</sup>   Jing Zhang <sup>1</sup>  
<sup>1</sup> Australian National University   <sup>2</sup> Shanghai AI Lab   <sup>3</sup> GE Research  
<sup>1</sup> firstname.lastname@anu.edu.au, <sup>3</sup> firstname.lastname@ge.com

###### Abstract

Recent studies have demonstrated that Reinforcement Learning (RL), notably Group Relative Policy Optimization (GRPO), can intrinsically elicit and enhance the reasoning capabilities of Vision-Language Models (VLMs). However, despite the promise, the underlying mechanisms that drive the effectiveness of RL models as well as their limitations remain underexplored. In this paper, we highlight a fundamental behavioral distinction between RL and base models, where the former engages in deeper yet narrow reasoning, while base models, despite less refined along individual path, exhibit broader and more diverse thinking patterns. Through further analysis of training dynamics, we show that GRPO is prone to diversity collapse, causing models to prematurely converge to a limited subset of reasoning strategies while discarding the majority of potential alternatives, leading to local optima and poor scalability. To address this, we propose Multi-Group Policy Optimization (MUPO), a simple yet effective approach designed to incentivize divergent thinking across multiple solutions, and demonstrate its effectiveness on established benchmarks. Project page: [https://xytian1008.github.io/MUPO/](https://xytian1008.github.io/MUPO/).

## 1 Introduction

Since the advent of Large Language Models (LLMs) \[touvron2023llama, Singh\_2025, seed2025seed1\], reasoning has emerged as a critical capability for addressing complex tasks. Early approaches typically rely on prompt engineering \[kojima2022large, zhou2022least\], which elicit chain-of-thought solutions, or on Supervised Fine-Tuning (SFT) \[muennighoff2025s1, chen2024expanding\], where models are trained on high-quality trajectories to emulate human-like thinking. More recently, the rise of Reinforcement Learning (RL), notably Group Relative Policy Optimization (GRPO) \[shao2024deepseekmath, schulman2017proximal\], facilitates self-reflection and verification, highlighting remarkable self-improving capabilities. Consequently, researchers have increasingly sought to integrate reasoning into Vision-Language Models (VLMs) \[bai2025qwen2, yang2025qwen3, guo2025seed1\]. By leveraging RL, VLMs acquire the ability to extract, analyze and reflect over visual information, achieving significant gains on both challenging logical visual questions \[yang2025r1, huang2025vision\] and traditional tasks \[liu2025visual, shen2025vlm\].

However, despite the promise, the underlying mechanisms driving the effectiveness of RL models, as well as their limitations, remain underexplored. Early doubts arise from LLMs, where \[yue2025does\] observe that although RL models demonstrate higher accuracy, their performance ceilings remain constrained by the capabilities of base models, lacking the ability to expand novel reasoning patterns. More recently, in the context of VLMs, \[li2025think, xia2025visionary\] notice that incorporating reasoning, in some cases, leads RL models to underperform base models. These findings lead us to ask: Do RL models truly outperform their base counterparts?

Motivated by this, we conduct a behavioral comparison between RL and base models on established benchmarks. Our results indicate that, when limited to a single attempt, RL models generally achieve higher accuracy than their base counterparts. However, when multiple samplings are permitted, base models are consistently capable of solving a broader number of problems. Notably, in failure cases where RL models are unable to handle despite multiple attempts, we find that base models often succeed by leveraging through diverse and alternative reasoning pathways that are not captured by RL models. For instance, as shown in Fig. 1, for geometry problems, RL models tend to rely exclusively on equation solving, which is prone to logical errors. In contrast, base models are capable of proposing simpler, verification-based strategies. Similarly, in object counting involving large quantities, RL models consistently adopt tedious sequential enumeration. Base models, however, are often able to employ efficient elimination strategies that reach the correct answer in significantly fewer steps.

![[x10 8.png|Refer to caption]]

Figure 1: The failure cases of RL models. We use Vision-R1 \[ huang2025vision \] as a representative RL model, with its corresponding base model being Qwen2.5-VL-7B bai2025qwen2. The examples are selected from MathVerse zhang2024mathverse and MathVista lu2023mathvista, respectively. For each question, we set the sampling temperature to 1.0 and generate multiple responses, each of which is displayed in a gray box. Main differences in the proposed reasoning strategies are annotated in blue and pink, while correct and incorrect answers are highlighted in green red, respectively.

These observations reveal a fundamental distinction between RL models and their vanilla base counterparts. During reasoning, RL models, despite demonstrating deeper deliberation, tend to be conservative, often adhering to a dominant strategy. In contrast, base models, although less refined along a single path, display divergent thinking, frequently exploring potential alternative solutions. Intuitively, the latter more closely aligns with human problem-solving, in which individuals, when given multiple attempts, often approach the same problem from varied perspectives, increasing their chances of success, especially on challenging tasks where common strategies may fail or be error-prone. Our further experiments reveal that models also exhibit similar patterns, where greater diversity in reasoning significantly enhances the probability of reaching correct answers.

The above findings indicate that models trained with RL, *i.e*., GRPO, appear to forgo the inherent divergent thinking exhibited by their base counterparts. To further investigate the cause of this shift, we examine the evolution of reasoning diversity throughout the training process. We observe that during the early training stage, diversity declines sharply to a negligible level, suggesting the model rapidly converges on a narrow set of strategies while discarding the vast majority of potential paths. As a result, the model concentrates on optimizing this limited subset for most of the training period. This collapse in diversity has two issues: 1) exploitation over exploration, where the model prioritizes a dominant strategy over exploring alternative modes, leading to local optima; 2) poor scalability, where convergent reasoning struggles to cover the broad spectrum of problems, thereby constraining test-time scaling capabilities.

Based on the above, a natural question is: can we preserve divergent thinking from base models during RL, enabling them to reason deeply about individual solutions while also maintaining a repertoire of diverse strategies? To address this, we propose Multi-Group Policy Optimization (MUPO), a drop-in replacement of GRPO to incentivize divergent reasoning. Inspired by the diversity collapse in GRPO, we partition model responses into multiple groups. Instead of computing advantages globally, MUPO performs localized advantage estimation within each group and introduces diversity reward to promote greater separation among groups. Intuitively, each group serves as a distinct realization of a reasoning strategy, and MUPO aims for the model to not only generate a range of diverse strategies but also to refine each effectively. Our result model, MUPO-Thinker-7B, demonstrates the ability to explore diverse reasoning paths in search of globally optimal solutions and achieves average gains of $2\sim 7\%$ over strong baselines on established benchmarks, setting a new state of the art.

In summary, our contributions are as follows:

- We highlight a fundamental difference in reasoning behavior between RL models and base models: the former engage in deeper yet narrowly focused reasoning, whereas the latter, despite being less sophisticated, exhibit broader and more diverse thought patterns.
- We find that GRPO is prone to diversity collapse, causing the model to search within a narrow set of strategies while disregarding the majority of potential alternatives, leading to local optima and limited scaling capabilities.
- We propose MUPO, a straightforward yet effective policy algorithm designed to incentivize divergent thinking across multiple solutions, and demonstrate its effectiveness through comprehensive experimental validation.

## 2 Related Work

![[x11 5.png|Refer to caption]]

Figure 2: The impact of reasoning diversity on model performance. In (A) and (B), we report acc@k for both RL and base models on established benchmarks, with color intensity decreasing as k = 1, 2 4 k=1,2,4. In (C) and (D), we plot relationship between reasoning diversity and the corresponding acc@4 scores. Each point is based on a set of responses, and a regression line is fitted to capture the overall trend.

Reasoning in VLMs. Enhancing reasoning capabilities has become a critical objective in the pursuit of general intelligence. In the context of VLMs, this progression evolves from prompt engineering which elicits chain-of-thought reasoning \[kojima2022large, zhou2022least, tian2024argue, tian2025black\], to SFT on high-quality, human-designed trajectories \[muennighoff2025s1, cai2024internlm2\]. More recently, the advent of RL, particularly GRPO \[shao2024deepseekmath, schulman2017proximal\], has shifted the paradigm from passive distillation toward active self-improvement, empowering models to autonomously discover and refine optimal strategies. Consequently, a growing body of research explores reasoning via diverse RL strategies \[peng2025skywork, tan2025reason, hu2025diversity\], dataset construction \[yang2025r1, meng2025mm, tian2025identifying, yao2023training\], and reward shaping \[wang2025perception, yang2025look, zou2025simlabel, yao2025simple\]. Despite these advances, the mechanisms of RL models and their limitations remain underexplored. In this work, we show that while RL models exhibit higher accuracy, they tend to forgo the divergent thinking of base models, thereby limiting the potential to solve a broader range of problems.

RL for Reasoning. As researchers notice that sparse yet easily verifiable rewards, *e.g*., accuracy and format, yield unexpectedly strong performance in RL, subsequent studies have emerged aiming to advance GRPO along multiple dimensions. For instance, \[yu2025dapo, zhang2025gvpo\] adopt a sampling-based perspective, prioritizing informative trajectories to enhance training efficiency. In parallel, \[tian2025more, jian2025look, zou2026unlocking\] focus on visual features, promoting perceptually-aware reasoning through fine-grained reward design. Recently, another line of research \[wang2025beyond, cheng2025reasoning, jiang2025risk\] explores the integration of uncertainty via entropy mechanisms to encourage exploration. However, these approaches fail to foster reasoning across divergent strategies. In contrast, we propose MUPO, which enables models to reason with both depth and breadth, ultimately guiding them toward the discovery of optimal solutions.

Test-Time Scaling. Given the substantial expense of training, allocating extra compute at test time is becoming a cost-effective strategy to enhance performance \[snell2024scaling, muennighoff2025s1\]. In the case of VLMs, this manifests as allowing for more thinking budgets. Such scaling law can be categorized into two types: sequential and parallel. The former promotes deeper reasoning through patterns such as step-by-step generation, self-reflection, and verification \[vl-rethinker, liu2025visual, he2025few\]. The latter emphasizes divergent exploration by sampling multiple candidates and aggregating via self-consistency or verifiers \[zheng2025parallel, wang2025visualprm\]. However, existing RL algorithms predominantly focus on sequential thinking, *i.e*., refining along a single reasoning path, while neglecting alternative branches, which limits their scaling potentials. Our proposed method, MUPO, bridges this gap by integrating parallel thinking into RL, yielding significant gains in both accuracy and scalability.

## 3 Exploring Divergent Thinking in VLMs

In this section, we begin with a comparison between existing RL models and their base counterparts, focusing on diversity as a key axis. We examine how variations in reasoning diversity influence model performance. Building on this, we further investigate the dynamics of diversity throughout the GRPO training process, revealing a critical issue where models tend to optimize within a narrow subset of strategies while discarding other potential alternatives.

![[x12 4.png|Refer to caption]]

Figure 3: The diversity collapse of GRPO. In (A), we plot the evolution of reasoning diversity across training steps. In (B), we present an illustration of policy distribution over training to highlight the contrasting dynamics of convergent and divergent thinking. The gray region denotes rewards associated with different reasoning trajectories, while the blue curve indicates corresponding sampling probabilities.

### 3.1 Divergent vs Convergent

To assess the impact of reasoning diversity to model performance, here we conduct a simple motivational experiment.

Experimental Setup. We consider models at two scales: 3B and 7B. For the 3B setting, we select VLAA-Thinker \[chen2025sft\] and LMM-R1 \[peng2025lmm\], while for the 7B setting, we consider Vision-R1 \[huang2025vision\] and R1-OneVision \[yang2025r1\], with Qwen2.5-VL-3B and Qwen2.5-VL-7B being the base models \[bai2025qwen2\]. All models are evaluated on a suite of reasoning-centric benchmarks, including MathVerse \[zhang2024mathverse\], LogicVista \[xiao2024logicvista\], WeMath \[qiao2024we\] and HallusionBench \[guan2024hallusionbench\]. The sampling temperature is set to $1.0$ to enable generation of multiple responses.

Metric. Beyond standard accuracy that only evaluates a single path, we introduce a more relaxed metric, acc@k, which is considered positive if at least one of $k$ sampled trajectories leads to the correct answer. We set $k=1$, $2$ and $4$ to assess the model’s ability to reach the correct answer when given multiple attempts. To quantify reasoning diversity, we employ Qwen3-Embedding-0.6B \[zhang2025qwen3\] to encode the reasoning segments of generated responses and compute cosine distances to measure differences. The diversity across multiple trajectories is then calculated as the pairwise average.

Results. Fig. 2 displays the impact of reasoning diversity on models, from which we may derive following key insights.

RL models dive depth, base models seek breadth. As shown in Fig. 2 (A) and (B), when $k=1$, RL models markedly outperform their base counterparts, reflecting sophisticated reasoning along a single trajectory. However, as $k$ increases, a notable shift occurs: base models succeed in solving substantially more problems, while the gains of RL models remain marginal. This observation suggests that base models exhibit a greater capacity to generate effective alternative solutions in challenging cases, indicating their higher potential. Our qualitative analysis in Fig. 1 further highlights this contrast, where RL models tend to adopt narrowly focused reasoning, yet base models display divergent reasoning, approaching problems from varied perspectives.

![[x13 7.png|Refer to caption]]

Figure 4: The t-SNE projection of reasoning embeddings. We analyze a successful case where RL models produce correct answers, and a failure case where they cannot despite multiple samplings.

Divergent thinking increases the odds of success. To further investigate the above distinctions, we perform multiple runs of base models and plot the relationship between reasoning diversity and corresponding acc@4 scores, as shown in Fig. 2 (C) and (D). We observe a strong positive correlation: as the reasoning diversity increases, acc@4 improves substantially. Intuitively, this indicates that tackling a problem through diverse paths rather than adhering to a single strategy facilitates the discovery of correct answers. This observation aligns with real-world problem-solving, where most tasks admit multiple viable solutions, and broader exploration often leads to more effective outcomes. Such findings also offer an explanation for the superior performance of base models over RL under parallel thinking.

To provide a deeper understanding, we visualize the t-SNE projection of reasoning embeddings for both RL and base models in Fig. 4. It reveals that RL reasoning embeddings are more densely clustered, while those of base models are more sparsely distributed. This structural difference explains higher pass rate of RL models in successful cases due to concentrated distribution. However, in failure cases, their narrow embedding region fails to cover any correct trajectory. In contrast, base models, benefiting from a wider and more diverse reasoning space, are more likely to retrieve correct solutions from alternative regions.

### 3.2 Diversity Collapse

![[x14 3.png|Refer to caption]]

Figure 5: The overview of MUPO. The upper part illustrates the high-level pipeline, where responses are partitioned into multiple groups, and the overall optimization objective is formulated as a composition of multiple GRPO objectives, each corresponding to a group. In the lower part, we present the advantage computation for a group, in which we introduce diversity reward to encourage inter-group separation.

The above findings indicate that RL, when applied to base models, diminishes inherent divergent thinking, steering them toward a single, specialized reasoning strategy. To gain deeper insights into this behavioral shift, we examine the evolution of diversity throughout the training process.

Experimental Setup. We adopt GRPO as the typical RL algorithm and consider two base models: Qwen2.5-VL-3B and Qwen2.5-VL-7B \[bai2025qwen2\]. The models are trained on ViRL39K \[vl-rethinker\], a high-quality and comprehensive dataset, for around half an epoch. For each step, we generate $10$ responses per example and record diversity on validation set.

GRPO narrows the mind before learning begins. As shown in Fig. 3 (A), we observe that reasoning diversity drops sharply during the early training stage, *i.e*., first 20 steps, where the model has been exposed to only limited training data. This collapse suggests a premature convergence to a narrow set of strategies, resulting in the exclusion of most potential solution paths and dedicating the majority of training to refining only a small subset. Such phenomenon has two key issues. 1) Exploitation over exploration: the model favors unimodal optimization, neglecting alternative modes and thereby becoming susceptible to local optima; 2) Limited scalability: as discussed in Section 3.1, this convergent reasoning fails to generalize across a broad range of questions, constraining the scaling capabilities.

In Fig. 3 (B), we provide a conceptual illustration of the training dynamics for convergent and divergent reasoning. Given that a problem may have multiple valid strategies, each corresponding to a distinct mode as shown in the figure, convergent training, *i.e*., GRPO, tends to select one mode early and progressively sharpens the distribution toward it. In contrast, divergent training, *i.e*., our expectation, encourages the model to explore and refine multiple modes, enabling the discovery of globally optimal solutions.

## 4 The Proposed Method

### 4.1 Preliminary

To introduce our approach, we first revisit the standard RL algorithm, notably Group Relative Policy Optimization (GRPO). In the multimodal setting, consider a dataset $\mathcal{D}$, where each example comprises a question $q$, a gold answer $y$, and corresponding visual input $I$. GRPO is designed to promote the generation of high-quality responses by contrasting multiple candidate outputs, rewarding superior ones while penalizing less effective alternatives. Given a policy model $\pi_{\theta}$, the objective is to maximize the following:

$$
\displaystyle\mathcal{J}_{\rm GRPO}(\theta)=\mathbb{E}_{(q,y,\mathcal{I})\sim\mathcal{D},\{o_{i}\}^{\left|G\right|}_{i=1}\sim{\pi}_{\theta_{\rm old}}(\cdot\mid q,\mathcal{I})}
$$
 
$$
\displaystyle\ \ \left[\frac{1}{\left|G\right|}\sum_{i=1}^{\left|G\right|}\min\Bigl(r_{i}(\theta)\hat{A}_{i},{\rm clip}\Bigl(r_{i}(\theta),1-\epsilon,1+\epsilon\Bigr)\hat{A}_{i}\Bigr)\right]
$$
 
$$
\displaystyle{\rm with}\ r_{i}(\theta)=\frac{\pi_{\theta}(o_{i}\mid q,\mathcal{I})}{\pi_{\theta_{\rm old}}(o_{i}\mid q,\mathcal{I})},
$$

where $G$ represents the set of sequences $\{o_{1},o_{2},...,o_{\left|G\right|}\}$ sampled from the old policy $\pi_{\rm old}$, $\epsilon$ controls the clipping bounds to maintain on-policy training. It is optional to impose a KL constraint toward the reference policy for stable optimization. The estimated token-level advantage $\hat{A}_{i}$ is derived by broadcasting the normalized sequence-level reward $R_{i}$ across all token positions, which is defined as follows:

$$
\displaystyle\hat{A}_{i}=\frac{R_{i}-{\rm mean}\left(\mathbf{R}\right)}{{\rm std}\left(\mathbf{R}\right)},\quad i=1,\cdots,\left|G\right|,
$$

where $\mathbf{R}=\{R_{1},R_{2},...,R_{\left|G\right|}\}$ indicates the reward of the sequence group. A common choice for reward design is the verifiable reward, which provides direct feedback based on metrics such as accuracy and format consistency.

### 4.2 Multi-Group Policy Optimization

Despite the promise of GRPO, our findings in Section 3.2 reveal that it is prone to diversity collapse, where models prematurely converge to a subset of reasoning paths while abandoning alternative strategies, leading to local optima. To address this, we propose Multi-Group Policy Optimization (MUPO), a simple yet effective drop-in replacement of GRPO to incentivize divergent thinking. Inspired by the intra-group diversity collapse, MUPO partitions responses into multiple groups. Unlike GRPO which computes advantages globally, MUPO performs localized advantage estimation within each group, and introduces a diversity reward to promote inter-group separation. Intuitively, each group serves as a distinct realization of a reasoning strategy, and MUPO aims to maintain multiple modes while refining each of them, thereby achieving both breadth and depth. The overview of our approach has been presented in Fig. 5.

Multi-Group Objective. Given $N$ sampled responses at each step, we partition them into $K$ groups $\{G_{k}\}_{k=1}^{K}$ based on the embedding space as described in Section 3.1. Specifically, we apply constrained clustering \[bradley2000constrained\] to group responses with similar trajectories, while enforcing minimum group size $G_{\rm min}$ for reliable advantage estimation. This enables each group to capture a distinct reasoning mode. The objective of MUPO is to maximize the following:

$$
\displaystyle\mathcal{J}_{\rm MUPO}(\theta)=\mathbb{E}_{(q,y,\mathcal{I})\sim\mathcal{D},\{o_{i}\}^{N}_{i=1}\sim{\pi}_{\theta_{\rm old}}(\cdot\mid q,\mathcal{I})}
$$
 
$$
\displaystyle\hskip-2.84526pt\Biggl[\sum_{k=1}^{K}\frac{w_{k}}{\left|G_{k}\right|}\underbrace{\sum_{i=1}^{\left|G_{k}\right|}\min\Bigl(r_{i}(\theta)\hat{A}_{i}^{k},{\rm clip}\Bigl(r_{i}(\theta),1-\epsilon,1+\epsilon\Bigr)\hat{A}_{i}^{k}\Bigr)}_{\text{$\mathcal{J}_{\rm GRPO}$}}\Biggr]
$$
 
$$
\displaystyle{\rm with}\ r_{i}(\theta)=\frac{\pi_{\theta}(o_{i}^{k}\mid q,\mathcal{I})}{\pi_{\theta_{\rm old}}(o_{i}^{k}\mid q,\mathcal{I})},\ \ w_{k}=\left(\frac{N}{K\left|G_{k}\right|}\right)^{\beta}.
$$

Essentially, MUPO can be regarded as a composition of multiple GRPO objectives, enabling the search for optima from diverse modes. $w_{k}$ is a load-balance scaler controlled by sensitivity exponent $\beta$, which modulates the contribution of each group to the overall objective, preventing larger groups from dominating the optimization process. The advantage $\hat{A}_{i}^{k}$ is estimated locally within each group to ensure the refinement of each mode proceeds independently:

$$
\displaystyle\hat{A}_{i}^{k}=\frac{R_{i}^{k}-{\rm mean}\left(\mathbf{R}\right)}{{\rm std}\left(\mathbf{R}\right)},\quad i=1,\cdots,\left|G_{k}\right|.
$$

Diversity Reward. To encourage models to explore various reasoning strategies, we introduce a diversity reward that increases the separation between groups. Specifically, for a given trajectory $o_{i}^{k}$ from $G_{k}$, we compute its diversity reward as the average distance between its reasoning embedding and those of responses from all other groups:

$$
\displaystyle R_{\rm div}=\frac{1}{N-\left|G_{k}\right|}\sum_{\begin{subarray}{c}m=1\\
m\neq k\end{subarray}}^{K}\sum_{j=1}^{\left|G_{m}\right|}d(o_{i}^{k},o_{j}^{m}),
$$

where $o_{j}^{m}$ indicates the trajectory from $G_{m}$, and $d(\cdot,\cdot)$ denotes the cosine distance between the reasoning embeddings of two responses. This ensures that responses within a group that are more distant from other groups receive a higher advantage. The final reward for $o_{i}^{k}$ is computed as:

$$
\displaystyle R_{i}^{k}=R_{\rm acc}+R_{\rm fmt}+\lambda\cdot\mathbf{1}\left[R_{\rm acc}=1\right]\cdot R_{\rm div}.
$$

$R_{\rm acc}$ and $R_{\rm fmt}$ denote accuracy and format reward, respectively. We impose an accuracy condition on the diversity reward to prevent reward hacking, where models pursue diverse outputs at the expense of correctness. $\lambda$ is the weight of the diversity reward and is annealed over the current training step $t_{\rm cur}$ according to the cosine schedule:

$$
\displaystyle\lambda=\lambda_{\rm min}+\frac{\lambda_{\rm max}-\lambda_{\rm min}}{2}\left(1+\cos\left(\pi\cdot\frac{t_{\rm cur}}{t_{\rm max}}\right)\right),
$$

where $t_{\rm max}$ is the total number of training steps, $\lambda_{\rm max}$ and $\lambda_{\rm min}$ specify the desired initial and final values of $\lambda$ with a smooth and monotonic decay. This design encourages broad exploration of diverse reasoning strategies in the early training, while gradually shifting the focus toward identifying globally optimal solutions in later stages.

## 5 Experiment and Results

Implementation Details. Similar to settings in Section 3.2, we train all models on ViRL39K \[vl-rethinker\] for 2 epochs with a learning rate of $1e^{-6}$. A random subset of the dataset is used in ablation study for efficiency. We select Qwen2.5-VL \[bai2025qwen2\] at 3B and 7B parameter scales as base models. For group reward computation, we generate $N=15$ responses per example with a sampling temperature of $1.0$. For MUPO, unless otherwise specified, we partition responses into $K=3$ groups with a minimum size of $G_{\rm min}=3$. We set the load-balance exponent $\beta=1$, and the initial and final values of diversity reward weight $\lambda_{\rm max}=0.4$ and $\lambda_{\rm min}=0.1$.

Benchmarks. We denote models trained with our method as MUPO-Thinker and evaluate on nine reasoning benchmarks spanning various task types: mathematical benchmarks including MathVerse \[zhang2024mathverse\], MathVista \[lu2023mathvista\], MathVision \[wang2024measuring\], LogicVista \[xiao2024logicvista\], WeMath \[qiao2024we\], and Geometry3K \[lu2021inter\], as well as general-purpose ones encompassing MMStar \[chen2024we\], HallusionBench \[guan2024hallusionbench\] and MMVet \[yu2023mm\].

Baseline Methods. To verify the effectiveness of our approach, we compare MUPO-Thinker against existing strong baselines, including InternVL2.5 \[chen2024expanding\], R1-OneVision \[yang2025r1\], VLAA-Thinker \[chen2025sft\], Vision-R1 \[huang2025vision\], VLM-R1 \[shen2025vlm\] and LMM-R1 \[peng2025lmm\], spanning both 3B and 7B models. We also report scores of proprietary models such as GPT-5-Thinking \[Singh\_2025\] and Gemini-2.5-Pro \[comanici2025gemini\] for reference. The baseline results are primarily cited from the corresponding papers, secondarily from the OpenCompass leaderboard, and reproduced when neither source is available.

Metrics. To evaluate both the depth and breadth of reasoning, we report acc@1 and acc@4 as defined in Section 3.1. For the former, we employ greedy decoding, which yields the most confident responses. For the latter, we generate 4 candidates per example with a sampling temperature of $1.0$, thereby assessing the model’s test-time scaling capabilities. Please refer to Appendix A for more configuration details.

Models MathVerse \[zhang2024mathverse\] MathVista \[lu2023mathvista\] MathVision \[wang2024measuring\] LogicVista \[xiao2024logicvista\] WeMath \[qiao2024we\] Geometry3k \[lu2021inter\] Average Acc@1 Acc@4 Acc@1 Acc@4 Acc@1 Acc@4 Acc@1 Acc@4 Acc@1 Acc@4 Acc@1 Acc@4 Acc@1 Acc@4 Closed-Source Models GPT-5-Thinking\* \[wang2025perception\] 81.2 85.5 81.9 86.1 72.0 79.2 70.0 81.5 71.1 78.4 79.9 84.3 76.1 82.5 Gemini-2.5-Pro\* \[comanici2025gemini\] 76.9 79.3 80.9 85.2 69.1 72.5 73.8 76.4 78.0 82.7 77.2 80.1 76.0 79.4 Open-Source Models Qwen2.5-VL-7B \[bai2025qwen2\] 40.7 55.2 62.3 78.5 23.2 41.6 42.6 59.3 33.1 50.1 38.5 54.4 40.1 56.5 InternVL2.5-8B\* \[chen2024expanding\] 34.5 42.3 68.2 72.8 25.6 29.4 38.3 44.7 38.6 43.5 44.8 48.0 41.7 46.8 R1-OneVision-7B \[yang2025r1\] 46.4 49.9 64.1 69.4 29.9 34.1 45.6 52.5 44.6 47.9 46.1 50.4 46.1 50.7 VLAA-Thinker-7B \[chen2025sft\] 48.2 51.6 68.0 70.8 26.4 30.3 48.5 54.5 41.5 46.8 50.6 55.2 47.2 51.5 Vision-R1-7B \[huang2025vision\] 52.4 56.1 73.5 75.6 28.2 32.9 49.7 53.8 41.6 44.3 49.0 54.1 49.1 52.8 Our Models MUPO-Thinker-3B 41.3 50.8 64.3 72.5 27.8 35.4 42.8 50.3 36.5 46.8 45.1 52.9 43.0 51.5 MUPO-Thinker-7B 53.9 61.7 77.9 82.4 31.3 39.7 50.6 61.5 44.1 48.6 52.1 59.1 51.6 58.8

Table 1: The evaluation of our method on mathematical benchmarks. We report scores of both open-source and proprietary VLMs. The best and second best results are bolded and underlined, respectively. \* indicates the results sourced from the OpenCompass leaderboard.

### 5.1 Main Results

Models MMStar \[chen2024we\] HallBench \[guan2024hallusionbench\] MMVet \[yu2023mm\] Average Acc@1 Acc@4 Acc@1 Acc@4 Acc@1 Acc@4 Acc@1 Acc@4 QwenVL \[bai2025qwen2\] 59.2 74.8 50.0 65.6 64.8 74.1 58.0 71.5 InternVL \[chen2024expanding\] 63.2 70.6 49.0 58.3 62.8 69.0 58.3 66.0 R1-OV \[yang2025r1\] 64.7 67.5 52.5 57.6 65.2 67.3 60.8 64.1 VLAA \[chen2025sft\] 66.1 69.1 54.7 56.9 70.0 72.7 63.6 66.2 V-R1 \[huang2025vision\] 66.3 68.9 55.4 59.0 68.2 70.5 63.3 66.1 MUPO 68.7 75.4 57.5 63.6 70.6 78.2 65.6 72.4

Table 2: The results of MUPO on general-purpose benchmarks. All models considered in this evaluation are of the 7B scale.

MUPO establishes a new state of the art. As shown in Table 1 and 2, MUPO-Thinker-7B achieves an average acc@1 improvement of $2.5\%$ ($49.1\%\rightarrow 51.6\%$) on mathematical benchmarks and $2.3\%$ ($63.3\%\rightarrow 65.6\%$) on general-purpose benchmarks over previous best results. Similarly, in Table 3, MUPO-Thinker-3B surpasses existing strong baselines of the same scale with notable gains of $2.0\%$ ($41.5\%\rightarrow 43.5\%$) and $2.4\%$ ($55.4\%\rightarrow 57.8\%$) on two types of benchmarks, respectively. This suggests that incorporating divergent thinking enables models to discover globally superior solutions and demonstrate more sophisticated reasoning along individual path, highlighting the importance of reasoning diversity for effective RL training.

![[x15 3.png|Refer to caption]]

Figure 6: The t-SNE projection of our reasoning embeddings. The selected examples here correspond to the ones presented in Fig. 4.

MUPO exhibits stronger test-time scaling capabilities. As shown in Table 1 and 2, MUPO-Thinker-7B significantly outperforms existing strong RL models in acc@4 by $6.0\%$ ($52.8\%\rightarrow 58.8\%$) on mathematical benchmarks and $6.2\%$ ($66.2\%\rightarrow 72.4\%$) on general-purpose benchmarks, while surpassing the base model of the same scale. In Table 3, MUPO-Thinker-3B also shows a substantial performance gap over recent baseline models, achieving an average gain of $5.9\%$ ($50.1\%\rightarrow 56.0\%$). Moreover, this scalability improvement enables MUPO-Thinker-3B to attain performance comparable to several strong 7B baselines. These results demonstrate that our approach successfully integrates the complementary strengths of RL and base models, significantly raising the upper bound of their capabilities.

### 5.2 Further Discussion

Models Params Mathematics General Average Acc@1 Acc@4 Acc@1 Acc@4 Acc@1 Acc@4 QwenVL \[bai2025qwen2\] 3B 33.4 47.0 51.3 62.3 40.0 52.1 InternVL \[chen2024expanding\] 2B 26.5 34.4 53.1 58.9 35.3 42.6 VLM-R1 \[shen2025vlm\] 4B 37.8 42.3 53.9 57.7 43.2 47.4 VLAA \[chen2025sft\] 3B 39.5 43.3 55.1 59.3 44.7 48.6 LMM-R1 \[peng2025lmm\] 4B 41.5 45.5 55.4 59.2 46.1 50.1 MUPO 3B 43.5 51.5 57.8 65.2 48.3 56.0

Table 3: The comparison of MUPO with baselines at 3B scale. We report average scores of mathematical and general benchmarks.

![[x16 1.png|Refer to caption]]

Figure 7: The qualitative analysis. We consider Vision-R1 \[ huang2025vision \] and MUPO-Thinker as typical GRPO and MUPO models and evaluate on an example from MMStar chen2024we with greedy decoding. We reorganize responses with numbering and omit special tags for better visualization.

![[x17 1.png|Refer to caption]]

Figure 8: The learning curves of accuracy and diversity reward. We use exponential moving average (EMA) for smoothing.

MUPO approaches problems with diverse strategies. To verify whether MUPO indeed induces divergent thinking, we visualize the distribution of reasoning embeddings produced by our model, as shown in Fig. 6. In contrast to GRPO, which tends to sample from a narrow region, MUPO exhibits a broad, multimodal structure, with each mode corresponding to a distinct solution strategy. This proves particularly advantageous in the failure case on the right, where GRPO fails to handle the problem, while MUPO successfully learns the correct reasoning from alternative modes, providing a clear explanation for its superior performance.

MUPO is capable of discovering better solutions. To further understand the benefits of divergent training in MUPO, we conduct a qualitative analysis in Fig. 7. In this spatial reasoning example, where the task is to estimate the height of a building, GRPO adopts a rigid layer-by-layer estimation, which is prone to cumulative errors and ultimately fails. In contrast, our model leverages surrounding reference objects to precisely locate the building height range. This illustrates that MUPO enables models to learn a broader set of reasoning strategies and to generate smarter solutions when faced with various problem types.

| K | MathVerse | MathVista | MathVision | MMStar | HallBench | Average |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 46.9 | 69.1 | 24.1 | 64.8 | 54.7 | 51.9 |
| 2 | 49.4 | 72.3 | 28.0 | 65.2 | 56.7 | 54.3 |
| 3 | 51.2 | 74.1 | 29.3 | 65.8 | 56.5 | 55.4 |
| 4 | 50.9 | 74.6 | 29.4 | 65.1 | 56.3 | 55.3 |
| 5 | 50.6 | 74.8 | 29.1 | 64.5 | 55.9 | 55.0 |

Table 4: The benchmark accuracy varying number of groups $K$.

MUPO learns from exploration to exploitation. In Fig. 8, we plot the learning curves of accuracy and diversity reward to further analyze the training dynamics of MUPO. As accuracy steadily improves, the diversity reward exhibits a distinct rise-fall-plateau trend. The initial rise indicates increasing distances between response groups, suggesting the model is actively exploring diverse modes. The slight decline in the middle is attributed to the annealing weight of the diversity reward, since the model begins to exploit the most promising strategy it has discovered. The final plateau reflects stabilization in training, implying the model has situated around effective solutions. We also present pairwise diversity of MUPO on validation set in Fig. 3 (A), which shows a much more gradual decline in contrast to sharp collapse observed in GRPO, indicating MUPO achieves a balanced trade-off between exploration and exploitation.

![[x18 1.png|Refer to caption]]

Figure 9: The ablation study of initial and final values of diversity reward weight λ max \\lambda\_{\\rm max} and min \\lambda\_{\\rm min} on average benchmark accuracy.

The number of groups $K$. In Table 4, we investigate the impact of number of groups $K$ on accuracy. Notably, when $K=1$, the training process degrades to GRPO. As $K$ increases, the accuracy rises rapidly and peaks at $K=3$. Moreover, for mathematical benchmarks, larger $K$ yields better performance, whereas smaller $K$ is more suitable for general problems. This reflects an intrinsic task nature: the former desires more flexible and diverse strategies, while the latter favors more uniform and structured reasoning.

The diversity reward weight $\lambda_{\rm max}$ and $\lambda_{\rm\min}$. MUPO adopts an annealing schedule for the diversity reward weight, progressively degrading from the initial value $\lambda_{\rm max}=0.4$ to the final value $\lambda_{\rm min}=0.1$ over the course of training. As shown in Fig. 9, we perform an ablation study by fixing one of the parameters as its default while varying the other. The results show the optimal values align with our default settings. Increasing either parameter causes the diversity reward to dominate at the expense of accuracy, while reducing them weakens exploration, amplifying the risk of convergence to local optima. More results such as ablation of $\beta$ and limitation analysis are provided in Appendix.

## 6 Conclusion

In this paper, we identify a fundamental distinction between the behavioral patterns of RL and base models, where the former tends to engage in deeper yet narrow reasoning trajectories, while base models, despite less refined along individual path, exihibit broader and more diverse reasoning strategies. Through further analysis of training dynamics, we find that GRPO is prone to diversity collapse, causing models to converge to a limited set of strategies while neglecting the majority of potential alternatives, leading to local optima and poor scalability. To address this, we propose MUPO, a simple yet effective policy algorithm designed to incentivizes divergent thinking across multiple solutions and demonstrate its effectivenss on established benchmarks.

#### Acknowledgement

This research was, in part, funded by the U.S. Government – DARPA TIAMAT HR00112490421. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the U.S. Government.
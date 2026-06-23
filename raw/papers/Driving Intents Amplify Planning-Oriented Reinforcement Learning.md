---
title: "Driving Intents Amplify Planning-Oriented Reinforcement Learning"
source: "https://arxiv.org/html/2605.12625v2"
author:
published:
created: 2026-06-23
description:
tags:
  - "clippings"
---
Hengtong Lu ${}^{1,2*\text{\Letter}}$  Victor Shea-Jay Huang <sup>1,3∗</sup>  Chengmin Yang <sup>1</sup>  Pengfei Jing <sup>1,2</sup> Jifeng Dai <sup>2</sup>  Yan Xie <sup>1</sup>  Benjin Zhu ${}^{1,2*\text{\Letter}\bigstar}$ <sup>1</sup> Li Auto    <sup>2</sup> Tsinghua University <sup>3</sup> CUHK MMLab     
luhengtong@lixiang.com,   zhubenjin@lixiang.com  
Project page:  [https://mind-omni.github.io/](https://mind-omni.github.io/)

###### Abstract

Continuous-action policies trained on a single demonstrated trajectory per scene suffer from *mode collapse*: samples cluster around the demonstrated maneuver and the policy cannot represent semantically distinct alternatives. Under preference-based evaluation, this caps best-of- $N$ performance – even oracle selection cannot recover what the sampling distribution does not contain. We introduce DIAL, a two-stage Driving-Intent-Amplified reinforcement Learning framework for preference-aligned continuous-action driving policies. In the first stage, DIAL conditions the flow-matching action head on a discrete intent label with classifier-free guidance (CFG), which expands the sampling distribution along distinct maneuver modes and breaks single-demonstration mode collapse. In the second stage, DIAL carries this expanded distribution into preference RL through *multi-intent Group Relative Policy Optimization (GRPO)*, which spans all intent classes within every preference group and prevents fine-tuning from re-collapsing around the currently preferred mode. Instantiated for end-to-end driving with eight rule-derived intents and evaluated on WOD-E2E: competitive Vision-to-Action (VA) and Vision-Language-Action (VLA) Supervised Finetuning (SFT) baselines plateau below the human-driven demonstration at best-of- $128$, with the strongest prior (RAP) capping at ater Feedback Score (RFS) $8.5$ even with best-of- $64$; intent-CFG sampling lifts this ceiling to RFS $9.14$ at best-of- $128$, surpassing both the prior best (RAP, $8.5$) and the human-driven demonstration ($8.13$) for the first time; and multi-intent GRPO improves held-out RFS from $7.681$ to $8.211$, while every single-intent baseline peaks lower and degrades by training end. These results suggest that the bottleneck of preference RL on continuous-action policies trained from demonstrations is not only how to update the policy, but how to expand and preserve the sampling distribution being optimized.

## 1 Introduction

Demonstrations record what happened, not what should have happened. A logged trajectory is one physically feasible future realized in a particular scene. Before the decision was made, several futures may still have been admissible: the vehicle may brake earlier or later, keep its lane or prepare a merge, proceed assertively or yield conservatively. The demonstration certifies one executed behavior, not that this behavior is uniquely safe or most preferred by human evaluators. That distinction is easy to miss when continuous-action policies are trained as geometric imitation, but it becomes central once policies are evaluated by preference alignment rather than by distance to a single recorded log.

![[x1 36.png|Refer to caption]]

Figure 1: Driving intents amplify planning-oriented RL by exposing within-scene preference contrast. (a) Under SFT + ordinary sampling, K rollouts collapse into one maneuver basin and their RFS scores are nearly identical ( Δ ≈ 0 \\Delta\\textsc{RFS}\\approx 0 ), so the group-relative advantage is uninformative. (b) Under intent-conditioned CFG sampling, = 8 K{=}8 rollouts (one per driving intent) spread across distinct basins and their scores spread widely ( ≫ \\Delta\\textsc{RFS}\\gg 0 ), exposing the preference contrast that planning-oriented RL amplifies.

Across robotic manipulation, navigation, and end-to-end driving [^58] [^17] [^56] [^2] [^7], continuous-action policies are typically trained by imitation or supervised fine-tuning on a single demonstrated trajectory per scene. With expressive flow-matching or diffusion action heads, this produces smooth local control near the demonstration but does little to expose alternative maneuvers: sampled trajectories cluster around the logged path, and the policy collapses to a single behavioral mode. Multimodal trajectory representations have long addressed this for prediction [^4] [^27] [^31] [^36], but for the ego policy of an action planner, single-demonstration supervision remains the default.

Mode collapse is hard to see when policies are scored by distance to the demonstration, because the metric rewards exactly the behavior collapse concentrates on. Preference-based evaluation makes it visible. The WOD-E2E RFS benchmark [^45] scores predicted trajectories against multiple human-annotated alternatives, so the logged GT is one scored candidate rather than an oracle target, and best-of- $K$ RFS over a policy’s samples quantifies the rater-preferred quality reachable under oracle selection. Across four competitive SFT VA and VLA baselines (Figure 4), best-of- $K$ RFS saturates *below* the logged G round Truth (GT) (RFS $8.13$) even at $K{=}128$: a mode-collapsed policy does not reach the rater-preferred regions, regardless of sample budget.

Therefore, We propose DIAL, Driving-Intent-Amplified reinforcement Learning, a two-stage training framework that first expands the sampling distribution of a continuous-action driving policy and then preserves this expansion during preference RL. The first stage uses intent-conditioned CFG to decode the same scene into semantically distinct maneuver modes – turning versus yielding, changing lanes versus keeping the lane, accelerating versus braking – so the flow-matching action head produces alternatives beyond coordinate-level perturbations of one logged path. With eight rule-derived intents (cruise, lane change L/R, turn L/R, U-turn, accelerate, decelerate), intent-conditioned best-of- $K$ already matches the human-driven demonstration at $K{\approx}8$, and pooling proposals across all eight intents reaches RFS $9.14$ at $K{=}128$. The strongest prior end-to-end driving planner, RAP [^9], caps at RFS $8.5$ even with best-of- $64$ over a learned rater; intent-conditioned sampling surpasses for the first time both this prior ceiling and the human-driven demonstration (RFS $8.13$).

The second stage of DIAL is needed because an expanded best-of- $N$ ceiling does not directly produce a deployable policy: oracle selection is unavailable at inference, and the policy still commits to one trajectory per frame. Reinforcement fine-tuning is the natural tool for capturing the ceiling, but standard GRPO does not preserve the expansion. Rollouts drawn from a single intent – whether the GT, predicted, or top-rated – re-collapse the rollout group around one mode, leaving group-relative advantages without preference contrast [^32] [^35]. The bottleneck is the sampling distribution, not the policy update, as shown in Figure 1.

To preserve the diversity created by the first stage, DIAL uses *multi-intent GRPO*. In the first stage, we condition the flow-matching action head on the eight driving intents and train with classifier-free guidance dropout. In the second stage, for each scene we sample a small number of trajectories under each intent, pool the eight intent-conditioned rollout sets into a single group, and compute group-relative advantages over the pooled set. At deployment, a learned intent classifier selects the intent and the same intent-conditioned generator decodes the final trajectory, so the proposal mechanism used during RL remains active at inference.

On a deterministic 338/100 split of the RFS-labeled WOD-E2E validation pool, DIAL improves held-out RFS from $7.681$ to $8.211$ over its SFT initialization, while every single-intent variant peaks lower and declines by its final checkpoint. Preference alignment for continuous-action policies trained from demonstrations is not the problem of imitating logged trajectories more accurately, nor of running a stronger optimizer; it is the problem of escaping mode collapse and preserving the escape through preference RL.

Contributions.

1. We identify single-demonstration SFT mode collapse as a key bottleneck for preference RL on continuous-action policies. On WOD-E2E, the best-of- $N$ RFS ceiling of competitive SFT baselines saturates below the logged human-driven demonstration even at $K{=}128$, showing that preference optimization is limited by what the policy can sample.
2. We show that intent-conditioned classifier-free guidance breaks this sampling ceiling by expanding the flow-matching action head along semantically distinct maneuver modes. With eight rule-derived driving intents, intent-CFG sampling reaches best-of- $128$ RFS of $9.14$, surpassing for the first time both the prior best planner RAP (RFS $8.5$ with best-of- $64$) and the human-driven demonstration (RFS $8.13$).
3. We introduce DIAL, Driving-Intent-Amplified Learning, a two-stage training framework that combines intent-CFG proposal expansion with multi-intent GRPO diversity preservation. By spanning all intent classes within every preference group, DIAL preserves the expanded sampling distribution during RL and improves held-out RFS from $7.681$ to $8.211$, while every single-intent variant peaks lower and declines by the end of training.

## 2 Related Work

DIAL connects four lines of work. End-to-end autonomous driving has moved from sensor-fusion imitation and scalable decoding toward planning-oriented, vectorized, sparse, generative, and diffusion-based planners [^12] [^16] [^6] [^38] [^54] [^21] [^59] [^11] [^8] [^43] [^34] [^15]. These systems motivate planning as the central output, while WOD-E2E turns evaluation toward rater-preference scores rather than logged-trajectory imitation alone [^45].

Vision-language-action models couple perception, language-conditioned representations, and continuous control for robotics and driving [^58] [^17] [^2] [^56]. In autonomous driving, related VLM, LLM, and VLA systems study interpretable driving, graph VQA, closed-loop language-conditioned driving, multimodal driving language models, knowledge-driven driving, counterfactual reasoning datasets, reasoning-to-planning RL, latent-action prediction, scene-adaptive experts, style-aware actions, world models, and VLA reasoning [^47] [^37] [^33] [^24] [^25] [^42] [^40] [^55] [^20] [^52] [^29] [^44] [^23] [^50] [^10] [^51] [^39] [^18] [^49]. Our work is not primarily a larger driving VLA; it isolates when preference optimization has useful maneuver-level proposals to rank.

Multimodal driving prediction has long represented future behavior through anchors, trajectory sets, latent modes, or intention queries [^4] [^27] [^31] [^36]. We use intent for a different role, namely to enlarge the maneuver-level proposal support available to a continuous ego policy.

The optimization builds on PPO/RLHF-style preference learning and group-relative policy updates [^32] [^57] [^28] [^35], as well as recent diffusion and flow policy optimization [^7] [^22] [^30] [^1] [^53] [^26] [^3] [^48] [^41] [^13]. Our focus is the interaction between intent-diverse proposals and RFS-guided reinforcement learning for planning-oriented continuous-action VLA driving. A fuller discussion is provided in Appendix A.

## 3 Method

DIAL is a two-stage training pipeline built on top of MindVLA-U1 [^14]. Given a driving scene $x$, the policy produces a future trajectory $\tau\in\mathbb{R}^{T\times d}$, and under RFS supervision the logged trajectory is one scored candidate rather than an oracle target. Stage 1 conditions the diffusion action head on a discrete driving intent with classifier-free guidance, expanding the sampling distribution along intent-level axes. Stage 2 runs GRPO over an intent-balanced rollout group that spans the full intent set per scene, preserving intent diversity through preference optimization. At inference, a lightweight intent classifier predicts the intent from visual context, and the same intent-conditioned generator decodes the final trajectory.

![[x2 35.png|Refer to caption]]

Figure 2: Overview of DIAL. (a) Stage 1 — CFG Imitation Training. The diffusion action head is conditioned on a discrete intent c i c\_{i}; CFG dropout ( p drop p\_{\\mathrm{drop}} ) teaches the model both conditional and unconditional action distributions. (b) Stage 2 — Multi-Intent GRPO. Per scene, K = 16 K{=}16 trajectories are sampled as S 2 S{=}2 rollouts × \\times | 𝒞 8 |\\mathcal{C}|{=}8 intents, scored by the RFS rater, and used to update the policy via GRPO against the SFT reference π 0 \\pi\_{0}

### 3.1 Intent-Conditioned CFG

We condition the flow-matching action head on a discrete driving intent $c$. The intent set is deliberately small: cruise, lane change left/right, turn left/right, U-turn, accelerate, and decelerate. During imitation training, these labels are inferred from trajectory geometry using displacement, heading change, lateral shift, and speed change. An intent embedding is added to the noisy-action/time suffix before the language model predicts the flow velocity. Classifier-free dropout also replaces some intent labels with an unconditional placeholder, so the same generator learns both conditional and unconditional action distributions. At inference, a lightweight prefix classifier predicts the intent from the visual-state context and CFG combines conditional and unconditional velocity predictions.

This stage is not meant to encode a complete driving ontology. Its purpose is to prevent single-trajectory SFT from collapsing distinct intent basins into one continuation. For the same scene, changing $c$ should change the proposal at the level raters can judge, not merely add coordinate noise.

### 3.2 Multi-Intent GRPO

For each scene, preference optimization is applied to an intent-balanced proposal group that spans the entire intent set, not to multiple noise seeds drawn from a single intent. We sample $S$ trajectories per intent, giving

$$
K=|\mathcal{C}|S
$$

proposals. The current implementation uses $|\mathcal{C}|=8$ and $S=2$, so each scene contributes $K=16$ trajectories. Each trajectory is generated by the stochastic SDE sampler of the flow-matching head, and its cumulative path log-probability is stored for policy-ratio replay.

The reward is the rater feedback score:

$$
R_{i}=\textsc{RFS}(\tau_{i};\mathcal{P}_{x}),
$$

where $\mathcal{P}_{x}$ denotes the rater trajectories and labels for scene $x$. Auxiliary geometric costs can be included, but the main implementation keeps them at zero and uses the preference reward described below. Advantages are normalized within the proposals of the same scene:

$$
A_{i}=\frac{R_{i}-\frac{1}{K}\sum_{j=1}^{K}R_{j}}{\operatorname{std}_{j=1}^{K}(R_{j})+\epsilon}.
$$

This is the step that makes intent diversity useful for RL: if all proposals express the same intent, the group has little preference contrast; if intents expose different intents, RFS supplies a meaningful within-scene ranking. Because the group always spans all eight intents rather than $K$ noise variants of one, every intent class receives a preference signal in the same update.

### 3.3 Reward-Hacking-Aware RFS Reward

The reported evaluation metric remains the standard WOD-E2E RFS score. For training, however, using the evaluator unchanged creates two avoidable reward-hacking paths. Standard RFS scores only the $3$ s and $5$ s anchors, so intermediate waypoints can drift while the two anchors stay inside the acceptable boxes. It also applies a hard maximum over up to three rater trajectories, so a model can overfit to the easiest high-label rater geometry on the RL training split.

We therefore use a training-side RFS variant with the same rater preference inputs but different aggregation. First, the rater maximum is replaced by label-softmax aggregation. For rater label $y_{p}$, geometric decay $d_{p,a}$ at anchor $a$, and temperature $\tau$, the per-anchor score is

$$
\widetilde{R}_{a}=\sum_{p}\frac{\exp(\tau y_{p})}{\sum_{q}\exp(\tau y_{q})}y_{p}d_{p,a}.
$$

The weights depend only on fixed rater labels, not on model-controlled geometry, so the policy cannot manipulate the aggregation by moving toward one rater. We tune $\tau$ on the split-aware validation protocol and report the sensitivity in Section 4.6.2. Second, the training anchors are densified from $\{3,5\}$ seconds to $\{1,2,3,4,5\}$ seconds. The additional anchors penalize discontinuous or implausible intermediate motion while preserving the canonical $3$ s and $5$ s scoring points used by evaluation.

### 3.4 Policy Update

Each sampled trajectory is replayed under the same intent condition used to generate it. Let $\log p_{\theta_{\mathrm{old}}}(\tau_{i}\mid x,c_{i})$ be the stored SDE path log-probability and $\log p_{\theta}(\tau_{i}\mid x,c_{i})$ the replayed current log-probability. The clipped GRPO objective is

$$
\displaystyle\rho_{i}=\exp\!\left(\log p_{\theta}(\tau_{i}\mid x,c_{i})-\log p_{\theta_{\mathrm{old}}}(\tau_{i}\mid x,c_{i})\right),
$$
$$
\displaystyle\mathcal{L}_{\mathrm{GRPO}}=-\frac{1}{K}\sum_{i=1}^{K}\min\!\left(\rho_{i}A_{i},\operatorname{clip}(\rho_{i},1-\epsilon_{\ell},1+\epsilon_{h})A_{i}\right).
$$

We use symmetric clipping with $\epsilon_{\ell}=\epsilon_{h}=0.2$. The update is regularized against the starting SFT policy $\theta_{0}$ using a reference path penalty,

$$
\mathcal{R}_{\mathrm{ref}}=\exp(\Delta_{i})-\Delta_{i}-1,\quad\Delta_{i}=\log p_{\theta_{0}}(\tau_{i}\mid x,c_{i})-\log p_{\theta}(\tau_{i}\mid x,c_{i}).
$$

The final loss is $\mathcal{L}_{\mathrm{GRPO}}+\beta\mathcal{R}_{\mathrm{ref}}$ with $\beta=0.002$. $\mathcal{R}_{\mathrm{ref}}$ is the standard $k_{3}$ KL estimator computed on the cumulative SDE path log-probability and acts as a vanilla reference-policy penalty.

## 4 Experiments

### 4.1 Experimental Setup

We evaluate DIAL on MindVLA-U1 under the WOD-E2E RFS protocol. The first-stage SFT model is trained on the Waymo training split. Preference optimization uses the RFS-labeled validation pool. We split the 438 validation sequences by a deterministic hash of sequence\_id: 338 sequences are used for RL training and 100 sequences are held out for evaluation, with split\_seed=43. All RL comparisons below use this split.

The main run starts from the ckpt8k intent-conditioned SFT checkpoint and applies multi-intent CFG GRPO with $C=8$ intents and $S=2$ samples per intent, giving $K=CS=16$ proposals per scene. Training uses batch size $4$, constant learning rate $5\times 10^{-7}$, CPS SDE sampling with noise level $0.5$, PPO clipping $0.2$, and reference-path coefficient $\beta=0.002$ on the cumulative-log-probability $k_{3}$ KL estimator. The training-side reward uses the reward-hacking-aware RFS variant described in Section 3.3; evaluation reports standard Waymo RFS. Held-out RFS is the main selection metric; full-split RFS and trust-region rate (TR) describe the selected checkpoint.

### 4.2 Main Results

Baseline training protocol. For Table 1, we retrain and adapt the official baseline implementations under a common Waymo-only task-training protocol. The supervised fine-tuning stage is constructed exclusively from the Waymo training split through our Waymo dataloader, which provides camera observations, ego-motion context, navigation intent, and future trajectory supervision, and converts them into the dataset format required by each model. The same Waymo SFT source is used for ReCogDrive [^19], AutoVLA [^56], Curious-VLA [^5], and WAM-Flow [^46]; no non-Waymo driving dataset is used for task-specific supervised fine-tuning. When RL is applied, each policy is initialized from its Waymo-SFT checkpoint and optimized on the Waymo RFS subset with GRPO-style rollouts and the MindVLA/Waymo RFS reward. Thus, official baseline codebases and released weights serve only as architectural implementations or initializations, while all task-specific SFT data and RL reward data in our experiments come from Waymo.

Claim. Under this controlled Waymo-only task-training protocol, DIAL produces the strongest held-out improvement among the RL-trained systems in Table 1. Each peak is compared with the step-0 evaluation from the same RL stage, so the reported gain isolates preference optimization rather than SFT training progress. Starting from MindVLA-U1 at $7.696$ held-out RFS, DIAL reaches $8.211$ ($+0.515$), with full-split RFS rising from $7.369$ to $8.631$. The baselines improve less on the same held-out split: WAM-Flow gains $+0.087$, Curious-VLA $+0.146$, AutoVLA $+0.043$, and ReCogDrive $+0.315$. DIAL also has the highest peak held-out score and the largest TR increase ($54.7\%$ to $68.0\%$), while ReCogDrive is the strongest baseline peak at $7.714$. These comparisons are consistent with the paper’s central mechanism: preference optimization is most effective when intent-conditioned sampling exposes maneuver-level alternatives that RFS can rank, rather than only local coordinate variants of a logged path.

Table 1: WOD-E2E RFS on the 338/100 held-out split (split\_seed=43) after Waymo-only SFT and preference optimization. All models use the same task-specific Waymo SFT source. Within each block, the peak row is the RL checkpoint with the highest held-out RFS and the init row is the step-0 evaluation from the same RL stage; $\Delta_{\mathrm{RL}}$ is the held-RFS change from init; TR and Full RFS are at the peak checkpoint.

<table><tbody><tr><th>Model</th><th>Action Representation</th><td>Stage</td><td>Held RFS</td><td><math><semantics><msub><mi>Δ</mi> <mi>RL</mi></msub> <annotation>\Delta_{\mathrm{RL}}</annotation></semantics></math></td><td>TR</td><td>Full RFS</td></tr><tr><th rowspan="2">WAM-Flow</th><th rowspan="2">discrete flow</th><td>SFT init</td><td>5.547</td><td>–</td><td>24.0%</td><td>5.757</td></tr><tr><td>RL peak</td><td>5.634</td><td>+0.087</td><td>19.0%</td><td>6.111</td></tr><tr><th rowspan="2">Curious-VLA</th><th rowspan="2">action token</th><td>SFT init</td><td>5.808</td><td>–</td><td>30.0%</td><td>5.827</td></tr><tr><td>RL peak</td><td>5.954</td><td>+0.146</td><td>31.0%</td><td>7.157</td></tr><tr><th rowspan="2">AutoVLA</th><th rowspan="2">action token</th><td>SFT init</td><td>6.744</td><td>–</td><td>46.0%</td><td>6.809</td></tr><tr><td>RL peak</td><td>6.787</td><td>+0.043</td><td>47.0%</td><td>6.780</td></tr><tr><th rowspan="2">ReCogDrive</th><th rowspan="2">DiT diffusion</th><td>SFT init</td><td>7.399</td><td>–</td><td>58.0%</td><td>7.244</td></tr><tr><td>RL peak</td><td>7.714</td><td>+0.315</td><td>65.4%</td><td>7.543</td></tr><tr><th rowspan="2">DIAL</th><th rowspan="2">continuous flow</th><td>SFT init</td><td>7.696</td><td>–</td><td>54.7%</td><td>7.369</td></tr><tr><td>RL peak</td><td>8.211</td><td>+0.515</td><td>68.0%</td><td>8.631</td></tr></tbody></table>

### 4.3 Intent-CFG Proposal Ceiling

Claim. Intent-conditioned CFG expands the proposal support beyond ordinary SFT sampling, raising the best-of- $N$ RFS ceiling above the logged human demonstration before any RL update.

On the same RFS-labeled scenes, we compare three proposal supports: the logged trajectory, best-of- $N$ from the ordinary SFT policy (four competitive VLA baselines: WAM-Flow, Curious-VLA, AutoVLA, ReCogDrive), and best-of- $N$ from the intent-conditioned SFT policy under four per-sample intent strategies. Figure 4 plots the expected best-of- $K$ RFS as $K$ grows from $1$ to $128$. The logged trajectory scores $8.13$ RFS (red dashed reference). All four baseline VLAs saturate *below* the GT line at $K=128$, confirming that ordinary stochastic sampling from the SFT policy cannot recover trajectories the human rater would prefer over the log. The intent-conditioned policy with any single-intent strategy (gt, classifier-predicted, top-rater, or random) crosses the GT line already at $K\approx 8$, because intent conditioning guides proposals into distinct semantic basins rather than perturbing a single maneuver mode. Pooling proposals across all eight intent classes with equal per-intent budget (right-most curve, $K=8\times n_{\text{per-intent}}$) extends the ceiling to $9.14$ at $K=128$, an improvement of $+1.07$ over the best baseline.

![[x3 32.png|Refer to caption]]

Figure 3: Pre-RL proposal ceiling. Best-of- K RFS vs. budget. Gray dashed: four SFT baselines all saturate below GT ( 8.13, red dashed) at = 128 K{=}128. Blue: intent-conditioned SFT under four strategies ( gt, top-rater predicted random ), all cross GT at ≈ 8 K{\\approx}8. Navy: 8-intent equal-budget pooling reaches 9.14 at.

### 4.4 Multi-Intent vs Single-Intent

Claim. Spanning the full intent set within each preference group is necessary; any single-intent recipe – regardless of how the conditioning intent is selected – either collapses or plateaus before DIAL’s peak.

We hold all other settings fixed at the main DIAL recipe and vary only how each rollout’s conditioning intent is chosen, while keeping the per-scene budget at $K=16$. Multi-intent GRPO (the main run) draws $K=16$ rollouts per scene by spanning all $8$ intent classes with $S=2$ per intent. The four single-intent baselines use $C=1$ and $S=16$ noise seeds, so the total budget remains $K=16$. The single-intent variant is then defined by how the one intent is picked: gt from the geometric intent of the logged trajectory; predicted from the deployment intent classifier head; top-rater from the highest-rated rater trajectory’s geometry (a label-leakage upper bound); and random from a uniform draw at every batch.

Table 2 compares held-out RFS peaks across group compositions at the same total budget $K=16$. DIAL reaches a peak of $8.211$ ($+0.515$ vs SFT init). None of the four single-intent baselines reaches $8.00$, and all ultimately decline. Random, the strongest single-intent baseline, peaks at $7.992$ but falls $0.22$ short of DIAL. Predicted reaches $7.864$ with a delayed collapse; its training-split RFS continues rising to $8.156$ while held-out RFS falls, the standard signature of reward hacking. Gt and top-rater collapse more sharply. At DIAL’s peak checkpoint, all single-intent variants lie $0.67$ – $1.98$ RFS points below. Spanning the full intent set is therefore necessary both for reaching a competitive peak and for maintaining preference contrast throughout training. Training dynamics for all variants are visualized in Figure 5 (Appendix B).

Table 2: Multi-intent group vs single-intent baselines, all sharing the main DIAL recipe and per-scene budget $K=16$. TR and Full RFS are reported at each variant’s held-peak checkpoint.

| Group composition | $C$ | $S$ | $K$ | Held peak | TR | Full RFS |
| --- | --- | --- | --- | --- | --- | --- |
| gt (single, geometric) | 1 | 16 | 16 | 7.783 | 65.0% | 7.733 |
| predicted (single, classifier) | 1 | 16 | 16 | 7.864 | 57.0% | 8.331 |
| top-rater (single, leakage) | 1 | 16 | 16 | 7.728 | 61.0% | 7.545 |
| random (single, uniform) | 1 | 16 | 16 | 7.992 | 64.0% | 8.035 |
| DIAL (multi-intent, main) | 8 | 2 | 16 | 8.211 | 68.0% | 8.631 |

### 4.5 Diversity Preservation Analysis

Claim. Multi-intent CFG GRPO prevents proposal diversity from collapsing during preference optimization, preserving the expanded sampling distribution established in Stage 1.

We support the claim with diversity metrics computed by inference on already-trained checkpoints. No additional RL training is required.

Inter-intent proposal distance (D1) and RFS spread (D2). For each scene we decode under all $8$ intents (with CFG disabled to obtain pure intent-conditional samples), compute D1 as the mean pairwise ADE across the $\binom{8}{2}=28$ trajectory pairs, and D2 as the RFS standard deviation across the $8$ trajectories. All RL variants are evaluated at the same checkpoint (iter 4800) for a fair comparison. Table 3 shows that DIAL retains the highest D2 ($0.75$), meaning different intents still produce quality-differentiated trajectories after preference optimization. Single-intent variants show two failure modes: random collapses spatially (D1 $=2.40$ m, lowest), producing near-identical trajectories regardless of the intent conditioning, while top-rater shows spatial scatter (D1 $=7.08$ m, highest) but poor quality differentiation (low D2). The diversity dividend *gap* $=$ D3@16 $-$ D3@1 measures how much additional RFS is recoverable by selecting among $16$ intent-conditioned proposals. DIAL preserves a gap of $+2.04$ (close to the SFT initialization’s $+2.23$), while all single-intent variants lose between $0.19$ and $0.57$ gap versus DIAL. Note that DIAL’s D3@16 ($6.540$) is slightly below the SFT initialization’s ($6.617$): RL concentrates probability toward better-scoring deployment modes at the cost of a small reduction in the best-of- $16$ diversity ceiling – the expected exploitation–diversity trade-off, which does not affect greedy deployment (held-out RFS improves from $7.696$ to $8.211$). Crucially, the gap ordering matches the held-out RFS ranking exactly: the more diversity is preserved during RL, the higher the final RFS.

Table 3: Diversity metrics at iter 4800 for a fair cross-variant comparison. D1: mean pairwise ADE across 8 intent-conditional trajectories per scene. D2: per-scene RFS std across the same 8 trajectories. D3@1/D3@16: best-of-1 and best-of-16 held-out RFS. Gap $=$ D3@16 $-$ D3@1: diversity dividend. The SFT init row uses the ckpt8k checkpoint before any RL.

| Sampling | D1 (m) | D2 | D3@1 | D3@16 | Gap |
| --- | --- | --- | --- | --- | --- |
| SFT init (no RL) | 6.43 | 0.52 | 4.387 | 6.617 | +2.23 |
| DIAL (multi-intent) | 4.17 | 0.75 | 4.500 | 6.540 | +2.04 |
| Single: random | 2.40 | 0.43 | 4.381 | 6.231 | +1.85 |
| Single: predicted | 4.82 | 0.64 | 4.372 | 6.172 | +1.80 |
| Single: top-rater | 7.08 | 0.65 | 4.240 | 6.020 | +1.78 |
| Single: GT intent | 6.02 | 0.59 | 4.229 | 5.889 | +1.66 |

### 4.6 Ablation Studies

#### 4.6.1 Samples per Intent

Claim. Holding the intent set at $C=8$ fixed, the number of samples per intent $S$ controls a trade-off between within-intent advantage variance and reward-hacking onset.

Figure 4 sweeps $S\in\{1,2,3,4\}$ at fixed $C=8$, giving total group sizes $K\in\{8,16,24,32\}$. $S=1$ ($K=8$) supplies one rollout per intent and gives the cheapest group; the held-out peak is $7.962$ and the run remains stable throughout training. $S=2$ ($K=16$, the main configuration) reaches the highest peak $8.211$. $S=3$ ($K=24$) reaches a peak of $8.094$ and $S=4$ ($K=32$) peaks at $8.033$, confirming that higher $S$ shifts the peak earlier and slightly reduces peak height. Across all runs, $S=2$ remains the peak-height sweet spot; $S=1$ remains the deployment-stability operating point.

#### 4.6.2 Reward Shaping

Standard Waymo RFS uses a hard maximum over rater trajectories and scores only the $3$ s and $5$ s anchors, creating avoidable reward-hacking paths during RL. We replace the maximum with a label-softmax aggregation (temperature $\tau$) and densify the scored anchors to $\{1,2,3,4,5\}$ s; Table 4 ablates these choices. The peak-height ordering is non-monotone in $\tau$: $\tau=0.3$ gives the highest held-out peak ($8.211$), while row D ($\tau=1.0$ + dense anchors) achieves the strongest full-split score ($8.728$) with substantially better end-of-training stability. We report $\tau=0.3$ as the main configuration for peak performance; row D is the deployment-friendly alternative.

Table 4: Training-side reward ablation on the 338/100 held-out split (§4.1). All variants share the multi-intent main recipe ($C=8$, $S=2$, $K=16$). TR and Full peak are reported at the held-peak checkpoint.

| Reward variant | Aggr | Anchor | $\tau$ | Held peak | TR | Full peak |
| --- | --- | --- | --- | --- | --- | --- |
| A. Vanilla | max | sparse | – | 7.990 | 68.0% | 8.328 |
| B. Dense anchors only | max | dense | – | 7.965 | 67.0% | 8.430 |
| C. Softmax only | softmax | sparse | 1.0 | 8.147 | 72.0% | 8.485 |
| D. Softmax + dense (deployment-friendly) | softmax | dense | 1.0 | 8.130 | 62.0% | 8.728 |
| Softmax + dense ($\tau{=}0.5$) | softmax | dense | 0.5 | 8.097 | 62.0% | 8.634 |
| DIAL (main) | softmax | dense | 0.3 | 8.211 | 68.0% | 8.631 |
| Mean ($\tau{\to}0$) + dense | mean | dense | – | 7.834 | 60.0% | 8.098 |

## 5 Conclusion

We identified single-demonstration SFT mode collapse as the primary bottleneck for preference RL on continuous-action driving policies: when sampled trajectories cluster around one maneuver mode, the group-relative reward signal carries no maneuver-level contrast, and RL cannot improve beyond what the sampling distribution already contains. DIAL addresses this with a two-stage training pipeline. Stage 1 conditions the diffusion action head on discrete driving intents with classifier-free guidance, expanding the sampling distribution into semantically distinct maneuver basins before any RL update. Stage 2 runs multi-intent GRPO over an intent-balanced proposal group per scene, preserving the expanded distribution through preference optimization; any single-intent rollout recipe re-collapses the group and degrades by training end. Together, the two stages lift the best-of- $N$ RFS ceiling to $9.14$ at $K{=}128$ – surpassing both the strongest prior planner and the human-driven demonstration – and improve held-out single-trajectory RFS from $7.696$ to $8.211$, with no single-intent variant reaching $8.00$.

##### Limitations.

Several aspects of the current implementation bound the scope of the conclusions. First, the eight driving intents are derived from trajectory geometry by hand-coded rules; they capture common maneuver categories but may not cover uncommon long-tail behaviors, and the labeling heuristic can misclassify ambiguous trajectories. Second, RL training uses the $438$ -sequence RFS-labeled validation pool, which is small relative to the full Waymo training split; the reward signal may not fully represent the distribution of challenging or rare scenarios.

## References

## Appendix A Extended Related Work

##### Vision-language-action policies.

Vision-language-action models adapt large multimodal representations to action generation. RT-2 studies how web-scale vision-language pretraining can transfer to robot control [^58]; OpenVLA provides an open-source VLA model for robotic manipulation [^17]; and $\pi_{0}$ uses a flow-based action model for general robot control [^2]. In autonomous driving, WOD-E2E defines a vision-based end-to-end benchmark with challenging long-tail scenarios and rater-feedback evaluation [^45], while AutoVLA studies a VLA driving model with adaptive reasoning and reinforcement fine-tuning [^56]. Our work is not primarily a larger VLA architecture. It isolates a proposal-support bottleneck: a continuous-action VLA can only benefit from preference RL if it first proposes maneuver alternatives that the preference signal can rank.

##### Intent and multimodal trajectory proposals.

Driving behavior is intrinsically multimodal. MultiPath predicts a distribution over anchor trajectory hypotheses [^4], CoverNet casts behavior prediction as classification over a representative trajectory set [^27], Trajectron++ uses a graph-structured generative model for dynamically feasible multi-agent forecasting [^31], and Motion Transformer combines global intention localization with local trajectory refinement [^36]. These methods show that future motion is better represented as a structured set of alternatives than as a single regression target. DIAL transfers that lesson to ego-policy optimization: the intent label is not the final prediction target, but a control variable that forces the proposal generator to expose distinct maneuvers before RFS-guided RL.

##### Generative continuous-action policies.

Diffusion and flow models are attractive action heads because they can represent complex continuous trajectory distributions. Diffusion Policy formulates visuomotor control as conditional action diffusion [^7], and flow matching trains continuous normalizing flows by regressing velocity fields along probability paths [^22]. Recent policy-optimization methods adapt these generators to reward optimization, including DPPO for diffusion policies [^30], FlowQ for offline RL with energy-guided flow policies [^1], ReinFlow for online fine-tuning of flow matching policies [^53], and flow matching policy gradients [^26]. Our setting is complementary: we retain a flow-style continuous generator, but structure its sampling group by driving intent so that the reward sees maneuver-level variation rather than only coordinate-level noise.

##### Preference optimization and group-relative updates.

Preference optimization commonly regularizes a learned policy against a reference policy while increasing the likelihood of preferred outputs. PPO supplies the clipped policy-gradient backbone [^32], RLHF-style fine-tuning applies learned human preference rewards to language models [^57], DPO removes explicit reward-model RL by optimizing pairwise preferences directly [^28], and GRPO uses group-relative advantages to reduce the need for a learned value function [^35]. WOD-E2E’s RFS provides a planning-facing preference signal for trajectories rather than text [^45]. DIAL therefore keeps the group-relative update, but defines the group as intent-diverse continuous proposals from the same scene and regularizes the update on the replayed continuous generation path.

## Appendix B Training Dynamics

Figure 5 plots held-out RFS throughout RL training for DIAL and the four single-intent baselines from Section 4.4, all sharing the same per-scene budget $K=16$.

Three patterns are visible. First, DIAL (multi-intent) rises to its peak held-out RFS of $8.211$ and subsequently declines only modestly, maintaining a substantially higher level than all single-intent variants throughout. The relative stability reflects that intent-diverse rollouts supply meaningful within-group preference contrast at every iteration: the group always spans semantically distinct maneuvers, so the reward signal never saturates.

Second, all four single-intent variants peak lower and exhibit sharper post-peak declines. Random reaches the highest single-intent peak ($7.992$) but falls well below DIAL by training end. Gt and top-rater collapse earliest, consistent with the observation that a fixed intent quickly exhausts preference contrast once the policy learns to score well on that intent.

Third, predicted displays the canonical reward-hacking signature: training-split RFS continues rising after the held-out peak while held-out RFS falls, confirming that optimizing a single intent aligned with the classifier’s prediction overfits to the RL training split rather than learning transferable preferences. This divergence between training-split and held-out RFS is absent in DIAL, where spanning all intents provides a structural barrier against distribution-specific overfitting.

![[x5 27.png|Refer to caption]]

Figure 5: Held-out RFS throughout RL training. DIAL (multi-intent, C = 8 C=8, S 2 S=2 ) peaks highest and declines least among all variants. Single-intent baselines ( 1 C=1 16 S=16 ) all peak lower and collapse more sharply. Predicted (dashed) shows the reward-hacking signature: training-split continues rising while held-out falls after the peak. All variants share the same per-scene budget K K=16 and are evaluated on the 338/100 held-out split.

[^1]: FlowQ: energy-guided flow policies for offline reinforcement learning. arXiv preprint arXiv:2505.14139. Cited by: Appendix A, §2.

[^2]: $\pi_{0}$: a vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164. Cited by: Appendix A, §1, §2.

[^3]: Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301. Cited by: §2.

[^4]: Multipath: multiple probabilistic anchor trajectory hypotheses for behavior prediction. arXiv preprint arXiv:1910.05449. Cited by: Appendix A, §1, §2.

[^5]: Devil is in narrow policy: unleashing exploration in driving vla models. arXiv preprint arXiv:2603.06049. Cited by: §4.2.

[^6]: Vadv2: end-to-end vectorized autonomous driving via probabilistic planning. arXiv preprint arXiv:2402.13243. Cited by: §2.

[^7]: Diffusion policy: visuomotor policy learning via action diffusion. The International Journal of Robotics Research 44 (10-11), pp. 1684–1704. Cited by: Appendix A, §1, §2.

[^8]: Transfuser: imitation with transformer-based sensor fusion for autonomous driving. IEEE transactions on pattern analysis and machine intelligence 45 (11), pp. 12878–12895. Cited by: §2.

[^9]: Rap: 3d rasterization augmented end-to-end planning. arXiv preprint arXiv:2510.04333. Cited by: §1.

[^10]: StyleVLA: driving style-aware vision language action model for autonomous driving. arXiv preprint arXiv:2603.09482. Cited by: §2.

[^11]: St-p3: end-to-end vision-based autonomous driving via spatial-temporal feature learning. In European Conference on Computer Vision, pp. 533–549. Cited by: §2.

[^12]: Planning-oriented autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 17853–17862. Cited by: §2.

[^13]: Tide: temporal-aware sparse autoencoders for interpretable diffusion transformers in image generation. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40, pp. 435–443. Cited by: §2.

[^14]: MindVLA-u1: vla beats va with unified streaming architecture for autonomous driving. arXiv preprint arXiv:2605.12624. Cited by: §3.

[^15]: Think twice before driving: towards scalable decoders for end-to-end autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21983–21994. Cited by: §2.

[^16]: Vad: vectorized scene representation for efficient autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 8340–8350. Cited by: §2.

[^17]: Openvla: an open-source vision-language-action model. arXiv preprint arXiv:2406.09246. Cited by: Appendix A, §1, §2.

[^18]: DriveVLA-w0: world models amplify data scaling law in autonomous driving. arXiv preprint arXiv:2510.12796. Cited by: §2.

[^19]: Recogdrive: a reinforced cognitive framework for end-to-end autonomous driving. arXiv preprint arXiv:2506.08052. Cited by: §4.2.

[^20]: Drive-r1: bridging reasoning and planning in vlms for autonomous driving with reinforcement learning. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40, pp. 6708–6716. Cited by: §2.

[^21]: Diffusiondrive: truncated diffusion model for end-to-end autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 12037–12047. Cited by: §2.

[^22]: Flow matching for generative modeling. arXiv preprint arXiv:2210.02747. Cited by: Appendix A, §2.

[^23]: Last-vla: thinking in latent spatio-temporal space for vision-language-action in autonomous driving. arXiv preprint arXiv:2603.01928. Cited by: §2.

[^24]: Dolphins: multimodal language model for driving. In European Conference on Computer Vision, pp. 403–420. Cited by: §2.

[^25]: Gpt-driver: learning to drive with gpt. arXiv preprint arXiv:2310.01415. Cited by: §2.

[^26]: Flow matching policy gradients. arXiv preprint arXiv:2507.21053. Cited by: Appendix A, §2.

[^27]: Covernet: multimodal behavior prediction using trajectory sets. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 14074–14083. Cited by: Appendix A, §1, §2.

[^28]: Direct preference optimization: your language model is secretly a reward model. Advances in neural information processing systems 36, pp. 53728–53741. Cited by: Appendix A, §2.

[^29]: Nord: a data-efficient vision-language-action model that drives without reasoning. arXiv preprint arXiv:2602.21172. Cited by: §2.

[^30]: Diffusion policy policy optimization. arXiv preprint arXiv:2409.00588. Cited by: Appendix A, §2.

[^31]: Trajectron++: dynamically-feasible trajectory forecasting with heterogeneous data. In European conference on computer vision, pp. 683–700. Cited by: Appendix A, §1, §2.

[^32]: Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347. Cited by: Appendix A, §1, §2.

[^33]: Lmdrive: closed-loop end-to-end driving with large language models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 15120–15130. Cited by: §2.

[^34]: Safety-enhanced autonomous driving using interpretable sensor fusion transformer. In Conference on Robot Learning, pp. 726–737. Cited by: §2.

[^35]: Deepseekmath: pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300. Cited by: Appendix A, §1, §2.

[^36]: Motion transformer with global intention localization and local movement refinement. Advances in Neural Information Processing Systems 35, pp. 6531–6543. Cited by: Appendix A, §1, §2.

[^37]: Drivelm: driving with graph visual question answering. In European conference on computer vision, pp. 256–274. Cited by: §2.

[^38]: Sparsedrive: end-to-end autonomous driving via sparse scene representation. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pp. 8795–8801. Cited by: §2.

[^39]: Learning vision-language-action world models for autonomous driving. arXiv preprint arXiv:2604.09059. Cited by: §2.

[^40]: Omnidrive: a holistic vision-language dataset for autonomous driving with counterfactual reasoning. In Proceedings of the computer vision and pattern recognition conference, pp. 22442–22452. Cited by: §2.

[^41]: Diffusion policies as an expressive policy class for offline reinforcement learning. arXiv preprint arXiv:2208.06193. Cited by: §2.

[^42]: Dilu: a knowledge-driven approach to autonomous driving with large language models. arXiv preprint arXiv:2309.16292. Cited by: §2.

[^43]: Trajectory-guided control prediction for end-to-end autonomous driving: a simple yet strong baseline. Advances in Neural Information Processing Systems 35, pp. 6119–6132. Cited by: §2.

[^44]: LatentVLA: efficient vision-language models for autonomous driving via latent action prediction. arXiv preprint arXiv:2601.05611. Cited by: §2.

[^45]: Wod-e2e: waymo open dataset for end-to-end driving in challenging long-tail scenarios. arXiv preprint arXiv:2510.26125. Cited by: Appendix A, Appendix A, §1, §2.

[^46]: WAM-flow: parallel coarse-to-fine motion planning via discrete flow matching for autonomous driving. arXiv preprint arXiv:2512.06112. Cited by: §4.2.

[^47]: Drivegpt4: interpretable end-to-end autonomous driving via large language model. IEEE Robotics and Automation Letters 9 (10), pp. 8186–8193. Cited by: §2.

[^48]: Using human feedback to fine-tune diffusion models without any reward model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8941–8951. Cited by: §2.

[^49]: Vla-r1: enhancing reasoning in vision-language-action models. arXiv preprint arXiv:2510.01623. Cited by: §2.

[^50]: SAMoE-vla: a scene adaptive mixture-of-experts vision-language-action model for autonomous driving. arXiv preprint arXiv:2603.08113. Cited by: §2.

[^51]: Reasoning-vla: a fast and general vision-language-action reasoning model for autonomous driving. arXiv preprint arXiv:2511.19912. Cited by: §2.

[^52]: OpenREAD: reinforced open-ended reasoning for end-to-end autonomous driving with llm-as-critic. arXiv preprint arXiv:2512.01830. Cited by: §2.

[^53]: ReinFlow: fine-tuning flow matching policy with online reinforcement learning. arXiv preprint arXiv:2505.22094. Cited by: Appendix A, §2.

[^54]: Genad: generative end-to-end autonomous driving. In European Conference on Computer Vision, pp. 87–104. Cited by: §2.

[^55]: Opendrivevla: towards end-to-end autonomous driving with large vision language action model. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40, pp. 13782–13790. Cited by: §2.

[^56]: Autovla: a vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. arXiv preprint arXiv:2506.13757. Cited by: Appendix A, §1, §2, §4.2.

[^57]: Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593. Cited by: Appendix A, §2.

[^58]: Rt-2: vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, pp. 2165–2183. Cited by: Appendix A, §1, §2.

[^59]: DiffusionDriveV2: reinforcement learning-constrained truncated diffusion modeling in end-to-end autonomous driving. arXiv preprint arXiv:2512.07745. Cited by: §2.
---
title: GSPO vs. GRPO
type: concept
sources: [raw/papers/WAM-Diff_ A Masked Diffusion VLA Framework with MoE and Online Reinforcement Learning for Autonomous Driving.md, raw/papers/WAM-Flow_ Parallel Coarse-to-Fine Motion Planning via Discrete Flow Matching for Autonomous Driving.md, raw/papers/ReCogDrive_ A Reinforced Cognitive Framework for End-to-End Autonomous Driving.md, raw/papers/DriveFine_ Refining-Augmented Masked Diffusion VLA for Precise and Robust Driving.md, raw/papers/Devil is in Narrow Policy_ Unleashing Exploration in Driving VLA Models.md, raw/papers/Unleashing VLA Potentials in Autonomous Driving via Explicit Learning from Failures.md, raw/papers/SpanVLA_ Efficient Action Bridging and Learning from Negative-Recovery Samples for Vision-Language-Action Model.md, raw/papers/Understanding R1-Zero-Like Training_ A Critical Perspective.md, raw/papers/All Roads Lead to Rome_ Incentivizing Divergent Thinking in Vision-Language Models.md, raw/papers/Plan-R1_ Safe and Feasible Trajectory Planning as Language Modeling.md, raw/papers/PlannerRFT_ Reinforcing Diffusion Planners through Closed-Loop and Sample-Efficient Fine-Tuning.md, raw/papers/Fine-tuning is Not Enough_ A Parallel Framework for Collaborative Imitation and Reinforcement Learning in End-to-end Autonomous Driving.md, raw/papers/Driving Intents Amplify Planning-Oriented Reinforcement Learning.md, raw/papers/DisCO_ Reinforcing Large Reasoning Models with Discriminative Constrained Optimization.md, raw/papers/DAPO_ An Open-Source LLM Reinforcement Learning System at Scale.md]
related: [concepts/rl-for-ad.md, concepts/mixture-of-experts.md, concepts/diffusion-planner.md, concepts/r1-zero-like-training.md, concepts/divergent-thinking-in-vlms.md, concepts/nuplan-benchmark.md, concepts/parallel-il-rl.md, concepts/intent-conditioned-planning.md, concepts/discriminative-policy-optimization.md, sources/wam-diff.md, sources/wam-flow.md, sources/recogdrive.md, sources/drivefine.md, sources/curious-vla.md, sources/elf-vla.md, sources/spanvla.md, sources/understanding-r1-zero-like-training.md, sources/all-roads-lead-to-rome.md, sources/plan-r1.md, sources/plannerrft.md, sources/pair-drive.md, sources/dial.md, sources/disco.md, sources/dapo.md]
created: 2026-05-01
updated: 2026-06-23
confidence: high
---

## What It Is

GRPO is the dominant RL fine-tuning recipe in the wiki: sample a group of trajectories, score them with a driving reward, normalize advantages within the group, and update the policy. GSPO is WAM-Diff's sequence-level variant designed for masked diffusion with sparse MoE routing.

## Difference

| Aspect | GRPO | GSPO |
| --- | --- | --- |
| Unit of likelihood ratio | Usually token/action step or generated trajectory depending on implementation | Whole sequence, length-normalized |
| Main use in wiki | ReCogDrive, WAM-Flow, Curious-VLA, ELF-VLA, SpanVLA, DynVLA, Plan-R1 | WAM-Diff |
| Failure addressed | Exploration collapse, simulator reward alignment, CoT efficiency, safety-critical reward normalization | MoE routing instability under token-level RL |
| Best fit | Standard VLA, diffusion, DFM, action expert policies | Masked diffusion policies with sparse expert routing |

## Dr. GRPO Correction

[[sources/understanding-r1-zero-like-training.md]] shows that standard GRPO can be biased by two normalizers: response-length normalization, which can under-penalize long incorrect outputs, and per-question reward-std normalization, which can downweight high-variance medium-difficulty samples.

Dr. GRPO removes both and uses the group-centered reward difference directly, $\hat{A}=R_i-\bar{R}$. This is not a replacement for GSPO's MoE-specific sequence stabilization; it is a correction to the basic GRPO advantage/loss normalization. NoRD later uses this correction in AD, where it turns a weak reasoning-free SFT model from +0.67% PDMS under GRPO to +11.68% under Dr. GRPO.

## VD-GRPO: Safety-Critical Variance Correction

[[sources/plan-r1.md]] identifies a related but domain-specific normalization failure. In safety-gated trajectory planning, unsafe rollout groups often have larger reward variance because collision or drivable-area violations zero out rewards while safe groups vary mainly through soft terms. Standard GRPO's division by group standard deviation therefore applies an implicit `1 / sigma_R` group weight and suppresses rare unsafe groups.

VD-GRPO keeps group centering but replaces the standard-deviation denominator with a fixed global scale `c`: $\hat{A}\propto(R_i-\bar{R})/c$. This is close in spirit to Dr. GRPO's removal of std normalization, but Plan-R1 motivates it through safety-priority preservation and the RL/KL scale balance in trajectory planning.

## PlannerRFT: Group Construction and Survival Reward

[[sources/plannerrft.md]] leaves GRPO's relative update intact but changes the distribution from which each group is built. A PPO-trained Exploration Policy samples scene-conditioned lateral and longitudinal denoising guidance, avoiding both low-diversity Gaussian groups and unstable uniform perturbations.

It also changes the reward signal in groups where every trajectory eventually fails. The survival reward accumulates valid non-terminal segments, so an action that delays failure has higher relative reward than an immediate collision. This addresses zero-variance terminal groups through reward shaping, whereas Dr. GRPO and VD-GRPO address standard-deviation normalization in non-degenerate groups.

The distinctions are:

| Method | Primary bottleneck | Intervention |
| --- | --- | --- |
| Dr. GRPO | Difficulty/length bias | Remove response-length and group-std normalizers |
| VD-GRPO | Unsafe high-variance groups are downweighted | Replace group std with fixed global scale |
| PlannerRFT | Poor or all-failed trajectory groups | Learn adaptive sampling; reward survival duration |

## PaIR-Drive: Tree-Structured Group Construction

[[sources/pair-drive.md]] uses standard mean/std-normalized GRPO, but constructs each group by recurrently branching intention-conditioned residual trajectories around a human reference. Its intervention is therefore upstream of advantage normalization: improve the semantic coverage of the group before computing relative rewards.

With TransFuser and Best-of-6, tree-structured residuals score 93.3 PDMS / 88.5 EPDMS, compared with 88.8/81.6 for unstructured residuals. Increasing the GRPO group from 5 to 15 raises EPDMS from 80.6 to 88.5. Unlike PlannerRFT's learned continuous guidance distribution, PaIR-Drive expands explicit intention branches and retains the RL proposal module at inference.

## DIAL: Intent-Balanced GRPO Groups

[[sources/dial.md]] fixes total group size at 16 and changes semantic composition. Standard alternatives draw 16 noise samples from one GT, predicted, top-rater, or random intent. DIAL draws two samples from each of eight intents and normalizes advantages across the pooled group.

The best single-intent variant reaches held-out RFS 7.992; multi-intent DIAL reaches 8.211. This demonstrates a fourth GRPO design axis beyond loss normalization, reward shaping, and group size: **which behavioral modes are guaranteed to appear in each group**. DIAL retains ordinary mean/std advantage normalization; its gain comes from group support and preference contrast.

## DAPO: Systems-Scale GRPO Recipe

[[sources/dapo.md]] retains group-normalized advantages but changes four surrounding mechanisms:

- asymmetric clipping (`0.2` lower, `0.28` upper) to preserve low-probability exploration;
- dynamic resampling until every retained group has both correct and incorrect responses;
- token-level rather than sample-level loss aggregation;
- filtering/soft punishment for overlong truncated responses.

The cumulative recipe raises Qwen2.5-32B AIME24 avg@32 from 30 under naive GRPO to 50. This is not an isolated optimizer ablation: the gain includes rollout filtering, reward shaping, data transformation, loss reduction, and distributed-system behavior.

DAPO's dynamic sampling solves zero-gradient groups by discarding them. PlannerRFT, ELF-VLA, and DIAL instead try to recover learning signal through graded survival rewards, corrected samples, or diverse group construction. That distinction is critical for safety tasks where all-failed groups may be the most valuable cases.

## DisCO: Replacing GRPO with Discrimination

[[sources/disco.md]] provides a stricter analysis of binary-reward GRPO. Standard normalized advantages yield a per-question weight $\sqrt{p(1-p)}$; Dr. GRPO changes it to $p(1-p)$ rather than eliminating difficulty bias. Both suppress easy and hard mixed groups.

DisCO abandons the group-relative policy-gradient objective. It directly separates positive and negative rollout scores, removes ratio clipping, uses DRO to emphasize high-scoring hard negatives, and enforces old-to-new KL through a squared-hinge trust-region penalty. This is an objective replacement, not another GRPO variant.

The limitation is equally important: DisCO requires both positive and negative samples. An all-failed driving group remains uninformative unless combined with survival rewards, graded ranking, recovery generation, or another mechanism.

## MUPO: Multi-Group GRPO for Reasoning Diversity

[[sources/all-roads-lead-to-rome.md]] identifies another GRPO failure mode: **diversity collapse** across reasoning strategies. GRPO-trained VLMs often improve greedy `acc@1` but sample a narrow set of similar reasoning paths, limiting `acc@4` and parallel test-time scaling.

MUPO partitions sampled responses into multiple embedding-space groups, computes advantages locally within each group, and adds an accuracy-gated diversity reward between groups. It is closest to GRPO in objective form, but it changes the grouping structure so multiple reasoning modes can be optimized in parallel.

## Takeaways

- GRPO is a family of recipes, not a single comparable algorithm; reward design and sampling policy dominate outcomes.
- GSPO should be read as an architecture-specific stabilization method for MoE masked diffusion, not a universal replacement for GRPO.
- When comparing RL gains, separate reward source, group size, sampling diversity, KL/reference model, and whether the update is token-level or sequence-level.
- Treat response-length growth during GRPO as ambiguous evidence: it may reflect better reasoning, optimizer length bias, or both.
- Treat multi-sample improvements as a diversity question: MUPO shows that extra samples help only if they cover distinct reasoning modes.
- Treat safety-gated AD rewards as a priority-structure problem: Plan-R1 shows that group std normalization can invert the intended emphasis on rare unsafe cases.

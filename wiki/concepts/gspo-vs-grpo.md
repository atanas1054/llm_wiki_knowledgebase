---
title: GSPO vs. GRPO
type: concept
sources: [raw/papers/WAM-Diff_ A Masked Diffusion VLA Framework with MoE and Online Reinforcement Learning for Autonomous Driving.md, raw/papers/WAM-Flow_ Parallel Coarse-to-Fine Motion Planning via Discrete Flow Matching for Autonomous Driving.md, raw/papers/ReCogDrive_ A Reinforced Cognitive Framework for End-to-End Autonomous Driving.md, raw/papers/DriveFine_ Refining-Augmented Masked Diffusion VLA for Precise and Robust Driving.md, raw/papers/Devil is in Narrow Policy_ Unleashing Exploration in Driving VLA Models.md, raw/papers/Unleashing VLA Potentials in Autonomous Driving via Explicit Learning from Failures.md, raw/papers/SpanVLA_ Efficient Action Bridging and Learning from Negative-Recovery Samples for Vision-Language-Action Model.md, raw/papers/Understanding R1-Zero-Like Training_ A Critical Perspective.md, raw/papers/All Roads Lead to Rome_ Incentivizing Divergent Thinking in Vision-Language Models.md, raw/papers/Plan-R1_ Safe and Feasible Trajectory Planning as Language Modeling.md]
related: [concepts/rl-for-ad.md, concepts/mixture-of-experts.md, concepts/diffusion-planner.md, concepts/r1-zero-like-training.md, concepts/divergent-thinking-in-vlms.md, sources/wam-diff.md, sources/wam-flow.md, sources/recogdrive.md, sources/drivefine.md, sources/curious-vla.md, sources/elf-vla.md, sources/spanvla.md, sources/understanding-r1-zero-like-training.md, sources/all-roads-lead-to-rome.md, sources/plan-r1.md]
created: 2026-05-01
updated: 2026-06-18
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

---
title: R1-Zero-Like Training
type: concept
sources: [raw/papers/Understanding R1-Zero-Like Training_ A Critical Perspective.md, raw/papers/NoRD_ A Data-Efficient Vision-Language-Action Model that Drives without Reasoning.md, raw/papers/AutoDrive-R²_ Incentivizing Reasoning and Self-Reflection Capacity for VLA Model in Autonomous Driving.md, raw/papers/Alpamayo-R1_ Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail.md, raw/papers/All Roads Lead to Rome_ Incentivizing Divergent Thinking in Vision-Language Models.md, raw/papers/Plan-R1_ Safe and Feasible Trajectory Planning as Language Modeling.md]
related: [sources/understanding-r1-zero-like-training.md, sources/nord.md, sources/autodrive-r2.md, sources/alpamayo-r1.md, sources/all-roads-lead-to-rome.md, sources/plan-r1.md, concepts/gspo-vs-grpo.md, concepts/rl-for-ad.md, concepts/chain-of-thought-for-ad.md, concepts/foundation-backbones-for-ad.md, concepts/divergent-thinking-in-vlms.md, concepts/action-tokenization.md]
created: 2026-06-18
updated: 2026-06-18
confidence: high
---

## What It Is

R1-Zero-like training means applying outcome-reward RL directly to a base model without first doing supervised instruction or chain-of-thought fine-tuning. In the math-reasoning setting, the reward is usually a verifier over the final answer. In AD papers, the analogy appears when VLAs use GRPO-style reinforcement fine-tuning to optimize trajectory, safety, or reasoning rewards.

The central lesson from [[sources/understanding-r1-zero-like-training.md]] is that R1-Zero-like gains are not explained by RL alone. Base-model priors, prompt templates, question coverage, and optimizer normalization can all manufacture or suppress apparent reasoning improvements.

## Key Mechanics

| Factor | Why it matters |
| --- | --- |
| Base model | Qwen2.5-Math already answers math questions well with no template, so it is not a blank slate. |
| Template | A mismatched template can destroy existing capability before RL reconstructs it. |
| Exploration | The base policy must sample at least some rewarding trajectories; pass@k is a useful pre-RL diagnostic. |
| Reward | Binary verifiers make math RL clean; AD rewards are usually multi-metric, safety-gated, and simulator-dependent. |
| Optimizer | Standard GRPO can introduce length and difficulty bias; Dr. GRPO removes those normalizers. |

## Dr. GRPO

Standard GRPO uses a group-centered, std-normalized advantage and usually normalizes loss by response length:

$$
\hat{A}^{GRPO} = \frac{R_i-\bar{R}}{std(R)}
$$

Dr. GRPO removes both the response-length normalization and the per-question reward-std normalization:

$$
\hat{A}^{DrGRPO} = R_i-\bar{R}
$$

This matters because:
- length normalization can under-penalize long incorrect outputs;
- reward-std normalization can downweight high-variance medium-difficulty samples;
- response length can grow for optimizer reasons, not because better reasoning emerged.

## AD Relevance

NoRD ([[sources/nord.md]]) imports Dr. GRPO into driving RL. Its weak SFT policy has many high-variance intermediate scenes; standard GRPO attenuates those scenes, while Dr. GRPO gives them usable gradient signal. This is the closest direct AD use of the R1-Zero-like training analysis.

AutoDrive-R2 and Alpamayo-R1 use DeepSeek-R1-style reasoning/RL language, but their driving setting is not identical to math R1-Zero:
- rewards are physical, safety, consistency, or LRM-as-critic signals rather than only final-answer verification;
- supervised CoT or chain-of-causality data is still important;
- simulator or trajectory metrics introduce reward-design risk absent from simple math verification.

Plan-R1 ([[sources/plan-r1.md]]) is a cleaner "planning as language modeling" analogy: it pretrains a motion-token predictor on expert data, then RL-aligns the ego planner with rule-based safety and feasibility rewards. The important R1 lesson is again not "GRPO just works"; Plan-R1 argues that standard GRPO's per-group variance normalization actively suppresses rare unsafe groups, and replaces it with VD-GRPO.

## Interpretation Rules

- Treat "RL from base" claims cautiously when the base model is Qwen-derived or heavily domain-pretrained.
- Separate **capability recovery** from **capability creation**: a bad template can make RL look more powerful than it is.
- Do not treat "Aha moment" or self-reflection as proof of RL emergence; these behaviors can exist before RL and do not guarantee higher inference accuracy.
- When using GRPO in AD, inspect reward variance groups. Low variance and high variance can both starve learning, but for different reasons.
- Inspect whether RL preserved strategy diversity. [[sources/all-roads-lead-to-rome.md]] shows GRPO can improve `acc@1` while hurting multi-sample breadth, so extra rollouts may fail to recover alternative solutions.
- For safety-gated planning rewards, inspect whether normalization preserves the intended priority order. [[sources/plan-r1.md]] shows a case where collision-critical groups should be amplified but standard GRPO downweights them.

## Open Questions

- Does Dr. GRPO remain preferable when AD rewards are dense, continuous, and safety-gated rather than binary?
- Can pass@k-style exploration diagnostics predict which VLA policies will benefit from RFT before expensive simulator rollouts?
- Are Qwen-VL driving backbones similarly affected by hidden pretraining priors, analogous to Qwen2.5-Math's no-template behavior?
- How much of adaptive reasoning in driving VLAs is genuine planning improvement versus template- or reward-induced verbosity?

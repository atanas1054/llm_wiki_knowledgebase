---
title: Discriminative Policy Optimization
type: concept
sources: [raw/papers/DisCO_ Reinforcing Large Reasoning Models with Discriminative Constrained Optimization.md, raw/papers/DAPO_ An Open-Source LLM Reinforcement Learning System at Scale.md, raw/papers/Understanding R1-Zero-Like Training_ A Critical Perspective.md, raw/papers/Plan-R1_ Safe and Feasible Trajectory Planning as Language Modeling.md]
related: [sources/disco.md, sources/dapo.md, sources/understanding-r1-zero-like-training.md, sources/plan-r1.md, concepts/gspo-vs-grpo.md, concepts/r1-zero-like-training.md, concepts/rl-for-ad.md]
created: 2026-06-23
updated: 2026-06-23
confidence: medium
---

# Discriminative Policy Optimization

Discriminative policy optimization treats rewarded and unrewarded generations as positive and negative examples. Instead of multiplying policy ratios by group-relative advantages, it directly learns a scoring function that ranks positive outputs above negatives while constraining policy drift.

[[sources/disco.md]] develops this idea for binary-verifiable mathematical reasoning. Its relevance to autonomous driving is conceptual because driving rewards are usually continuous and multi-objective.

## GRPO as Weighted Discrimination

For a question whose old-policy success probability is $p$, binary-reward GRPO can be rewritten as a positive-minus-negative score objective weighted by:

$$
\omega_{GRPO}(p)=\sqrt{p(1-p)}.
$$

This weight is small for both hard and easy inputs. Dr. GRPO removes reward-standard-deviation normalization but retains:

$$
\omega_{Dr.GRPO}(p)=p(1-p).
$$

Thus optimizer-induced difficulty bias and group degeneracy are distinct:

- **Difficulty weighting:** mixed groups exist, but easy/hard questions receive small objective weight.
- **Degenerate groups:** all candidates have one label, so no positive/negative contrast exists at all.

DisCO removes the former, not the latter.

## Score Choices

| Score | Definition | Interpretation |
| --- | --- | --- |
| Log-L | Mean token log-likelihood | REINFORCE-like score without clipping |
| L-ratio | Mean token ratio $\pi_\theta/\pi_{old}$ | TRPO-like surrogate without clipping |

Using one unclipped score for both classes avoids the positive/negative asymmetry and flat regions introduced by PPO-style clipping.

## Hard-Negative Weighting

Uniform pairwise AUC can look good when one positive outranks most negatives but remains below one dangerous high-scoring negative. DisCO's DRO objective applies log-sum-exp over negatives, concentrating optimization on the hardest negative according to temperature $\tau$.

For driving, this maps naturally to candidate sets where most invalid trajectories are obviously bad but one subtly unsafe or rule-violating proposal receives high policy score. The method would need continuous or ordinal preference labels rather than the paper's binary correctness split.

## Trust-Region Constraint

DisCO enforces old-to-new KL through:

$$
\beta[D_{KL}(\pi_{old}\Vert\pi_\theta)-\delta]^2_+.
$$

This differs from a constant KL regularizer:

- inside the trust region, it contributes no gradient;
- outside, its effective coefficient grows with violation size;
- the old policy is refreshed each outer iteration.

The design aims to preserve exploration without permanently pulling the policy toward a fixed reference.

## Relationship to Driving GRPO Fixes

| Method | Failure addressed | Mechanism |
| --- | --- | --- |
| Dr. GRPO / NoRD | Reward-std difficulty attenuation | Remove std normalization |
| VD-GRPO / Plan-R1 | Unsafe groups downweighted by variance | Fixed global advantage scale |
| PlannerRFT | Poor/all-failed diffusion groups | Adaptive exploration + survival reward |
| DIAL | Same-mode groups lack preference contrast | Intent-balanced sampling |
| DisCO | Binary GRPO difficulty weighting, clipping instability, negative imbalance | Direct discrimination + hard negatives + KL constraint |

DisCO changes the optimization objective most fundamentally. The other methods largely preserve policy-gradient/group-relative structure and repair normalization, reward, or sampling.

## DAPO Contrast

[[sources/dapo.md]] is the strongest systems-oriented contrast to DisCO. DAPO retains normalized group advantages and clipping but raises the upper clip, filters degenerate groups, reweights tokens by changing loss reduction, and reshapes truncation rewards. DisCO argues that DAPO still carries $\sqrt{p(1-p)}$ question weighting and that widened clipping can cause excessive entropy growth.

The published comparisons answer different questions. DAPO demonstrates a complete 32B training system with dynamic sampling and 20k-token rollouts; DisCO isolates objectives on 1.5B/7B/8B models with 8k rollouts and explicitly omits DAPO dynamic sampling because of its roughly 3× sampling cost. Treating either as a universally stronger algorithm ignores those protocol differences.

## Conditions for Driving Transfer

A driving adaptation must decide:

- whether labels are binary feasibility, ordered preference, or continuous metric vectors;
- how to retain information from all-failed and all-success groups;
- whether positives/negatives are defined per scene or globally;
- how collision, drivable area, progress, comfort, and rule compliance define hard negatives;
- whether sequence score is a trajectory likelihood, denoising-path likelihood, token likelihood, or selector score;
- how the KL threshold interacts with safety-critical policy changes;
- whether hard-negative emphasis overfits simulator artifacts.

## Practical Interpretation

DisCO is not evidence that driving GRPO should be replaced immediately. It is evidence that removing standard-deviation normalization does not necessarily remove all difficulty-dependent weighting, and that a positive/negative ranking formulation may be more principled when rewards are genuinely binary and mixed groups are available.

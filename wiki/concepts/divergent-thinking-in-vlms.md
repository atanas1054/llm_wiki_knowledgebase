---
title: Divergent Thinking in VLMs
type: concept
sources: [raw/papers/All Roads Lead to Rome_ Incentivizing Divergent Thinking in Vision-Language Models.md, raw/papers/Devil is in Narrow Policy_ Unleashing Exploration in Driving VLA Models.md, raw/papers/Understanding R1-Zero-Like Training_ A Critical Perspective.md]
related: [sources/all-roads-lead-to-rome.md, sources/curious-vla.md, sources/understanding-r1-zero-like-training.md, concepts/gspo-vs-grpo.md, concepts/best-of-n.md, concepts/r1-zero-like-training.md, concepts/chain-of-thought-for-ad.md]
created: 2026-06-18
updated: 2026-06-18
confidence: high
---

## What It Is

Divergent thinking is the ability of a VLM to approach the same problem through multiple distinct reasoning strategies. [[sources/all-roads-lead-to-rome.md]] frames this as the missing half of RL reasoning: GRPO improves depth along a single trajectory, but can collapse breadth across alternatives.

## Why It Matters

Parallel test-time scaling only helps if the samples are meaningfully different. If a model generates four copies of the same flawed strategy, `acc@4` barely improves over `acc@1`. If it samples distinct solution modes, one attempt may discover a route that the dominant strategy misses.

## MUPO

Multi-Group Policy Optimization (MUPO) is a GRPO-style algorithm that:

1. samples `N` responses;
2. embeds the reasoning segments;
3. clusters responses into `K` strategy groups;
4. computes local advantages within each group;
5. adds an accuracy-gated diversity reward across groups.

This makes each group a separate search mode: the model can refine multiple strategies instead of collapsing onto one early winner.

## Relationship to Existing Wiki Concepts

| Concept | Connection |
| --- | --- |
| [[concepts/gspo-vs-grpo.md]] | MUPO is another GRPO variant, but it targets strategy diversity rather than MoE sequence stability or normalization bias. |
| [[concepts/best-of-n.md]] | BoN gains require diverse candidates; MUPO explains why correlated samples saturate. |
| [[concepts/r1-zero-like-training.md]] | Adds a third caution for RL-from-base: even if RL improves accuracy, it may erase useful base-model breadth. |
| [[sources/curious-vla.md]] | Curious-VLA diagnoses narrow driving-policy diversity; MUPO diagnoses narrow VLM reasoning-strategy diversity. |

## AD Interpretation

This paper is not about driving, but the analogy is strong:

- AD trajectory sampling can collapse to a narrow action mode, just as VLM reasoning can collapse to a narrow reasoning mode.
- Best-of-N and candidate scoring only help when candidates cover different plausible maneuvers.
- Reward design should distinguish superficial variation from useful diversity: in driving, that means distinct safe trajectories or reasoning modes, not jitter around the same path.

## Open Questions

- Can MUPO-style multi-group advantages improve NAVSIM/Bench2Drive RFT when groups are clustered by trajectory geometry rather than reasoning embeddings?
- Should AD VLAs use separate diversity rewards for text reasoning and action trajectories?
- Can learned candidate scorers, such as CLEAR-style or HybridDriveVLA-style rankers, benefit from MUPO-trained candidate diversity?
- How should diversity be measured when the output includes both CoT text and continuous trajectory points?

---
title: Reasoning Faithfulness in Driving VLAs
type: concept
sources: ["raw/papers/[-0.5mm] GRAVA GRAVA_ Grounded Reasoning-to-Action Representation and Learning for Autonomous Driving.md", raw/papers/Qwen-Drive-1.0_ An Initial Step towards a Vision-Language Foundation Model for Autonomous Driving.md, raw/papers/Alpamayo-R1_ Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail.md, raw/papers/Senna-2_ Aligning VLM and End-to-End Driving Policy for Consistent Decision Making and Planning.md, raw/papers/SpanVLA_ Efficient Action Bridging and Learning from Negative-Recovery Samples for Vision-Language-Action Model.md, raw/papers/Unifying Language-Action Understanding and Generation for Autonomous Driving.md, raw/papers/ORION_ A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation.md, raw/papers/AutoVLA_ A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning.md, raw/papers/UniUGP_ Unifying Understanding, Generation, and Planing For End-to-end Autonomous Driving.md, raw/papers/AutoDrive-R²_ Incentivizing Reasoning and Self-Reflection Capacity for VLA Model in Autonomous Driving.md, raw/papers/AdaThinkDrive_ Adaptive Thinking via Reinforcement Learning for Autonomous Driving.md, raw/papers/NoRD_ A Data-Efficient Vision-Language-Action Model that Drives without Reasoning.md, raw/papers/HERMES_ A Holistic End-to-End Risk-Aware Multimodal Embodied System with Vision–Language Models for Long-Tail Autonomous Driving.md, raw/papers/OneVL_ One-Step Latent Reasoning and Planning with Vision-Language Explanation.md]
related: [sources/grava.md, sources/qwen-drive-1.0.md, sources/alpamayo-r1.md, sources/senna2.md, sources/spanvla.md, sources/linkvla.md, sources/orion.md, sources/autovla.md, sources/uniugp.md, sources/autodrive-r2.md, sources/adathinkdrive.md, sources/nord.md, sources/hermes.md, sources/onevl.md, concepts/chain-of-thought-for-ad.md, concepts/dual-system-vla.md, concepts/rl-for-ad.md, concepts/teacher-pseudo-labels.md, concepts/evaluation-variance.md, research-directions.md]
created: 2026-09-27
updated: 2026-09-30
confidence: medium
---

# Reasoning Faithfulness in Driving VLAs

## What It Is

A reasoning trace is **faithful** if the action actually depends on its content, and **explanatory** if that content is what caused the action. [[concepts/chain-of-thought-for-ad.md]] asks *whether* CoT helps planning. This page asks *what the trace is doing when it does*. The candidate readings are:

1. **Rationale.** Scene evidence in the trace determines the action.
2. **Verbalized action.** The trace states the maneuver, and the action head executes the statement.
3. **Extra conditioning capacity.** More tokens and computation help regardless of content.
4. **Ignored.** The head does not read the trace at all.

---

## The Evidence

| Paper | Test or mechanism | Result | Which reading it supports |
|---|---|---|---|
| [[sources/grava.md]] | **Intervention**: swap low-reward reasoning for high-reward reasoning in the same scene; fixed decoder | Normalized PDMS 0.438 → 0.823; win rate 3% → 55% | Rules out *ignored*; consistent with *verbalized action* |
| [[sources/grava.md]] | Ablation ladder (internal benchmark, all RL-trained) | Action-only 74.8 → coarse 77.9 → grounded objects 85.5 → full 90.1 CDS | Content matters, not just length. Partly *rationale* |
| [[sources/grava.md]] | Numerical-grounding density (distances/speeds in the trace) | 0.804 → 0.899 across bins | Correlational; no length control |
| [[sources/qwen-drive-1.0.md]] | Authors' own limitation | "The generated trajectory does not always adhere to the textual rationale"; the reasoning effect is +0.4 PDMS | Names *extra capacity* as the alternative |
| [[sources/alpamayo-r1.md]] | Binary CoC–action consistency reward in RL (trajectory → meta-action must match the parsed decision) | Enforced by training | Enforces *verbalized action* |
| [[sources/spanvla.md]] | Reasoning–action inconsistency penalty in $r_{CoT}$ | Enforced | Same |
| [[sources/senna2.md]] | Explicit VLM-decision ↔ E2E-trajectory alignment | +19.3% consistency F1 | Dual-system consistency, not trace content |
| [[sources/linkvla.md]] | Action captioning: $(V,A)\to L$ as an auxiliary task | +0.16 DS | Bidirectional language–action coupling |
| [[sources/orion.md]] | Latent alignment between reasoning and the action VAE | Framework-level gain | Alignment by construction |
| [[sources/nord.md]] | No reasoning at all | 85.6 PDMS; the optimizer, not CoT, was the bottleneck | CoT is not necessary on NAVSIM |
| [[sources/adathinkdrive.md]] | Think vs. non-think by scene difficulty | Think helps only on hard scenes | Content matters where scenes demand it |

---

## The Hindsight-Annotation Problem {#hindsight}

**Most traces in the wiki are written with access to the future they are meant to predict:**

| Paper | How the trace's decision is produced |
|---|---|
| [[sources/grava.md]] | `trajectory.intent` from the GT trajectory feeds `ego_decision`; QA targets state "the future trajectory shows…" |
| [[sources/qwen-drive-1.0.md]] | A rule-based classifier derives maneuver components **from the recorded future** as a "motion prior" for the trace writer |
| [[sources/alpamayo-r1.md]] | Human annotators see the **full 0–8 s window** plus the decision label |
| [[sources/autovla.md]], [[sources/uniugp.md]], [[sources/autodrive-r2.md]] | Frontier-VLM or GT-grounded traces with a GT trajectory hint |

**Why this matters.** If the final decision in the trace is a relabelled future, training teaches the model to *state the maneuver first and then execute it*. An intervention that swaps traces (GRAVA) will then show a large effect even if the grounded evidence plays no causal role. Faithfulness to the *statement* is demonstrated; faithfulness to the *evidence* is not.

This is also what the consistency rewards above optimize. Alpamayo-R1 and SpanVLA reward agreement between the parsed decision and the trajectory's meta-action, which makes the trace a reliable **verbalized action** by construction. That is useful for interpretability and control, but it is a different property from explanation.

---

## Experiments That Would Separate the Readings

1. **Shuffled trace** (proposed on [[concepts/chain-of-thought-for-ad.md#trace-doubt]]). Condition on a trace from a different scene. If performance holds, the gain is *capacity*; if it collapses, the content is scene-specific.
2. **Evidence-only intervention.** Keep the stated maneuver fixed but edit the grounded evidence (move a box, change a distance). If the action changes, the head reads the evidence and not only the conclusion. No paper has done this.
3. **Decision-free targets.** Build the ego decision only from the object-level decisions (GRAVA's graph without the `trajectory.intent` edge) and rerun the reasoning ladder. If the gain survives, the evidence is doing work.
4. **Length-matched null traces.** Filler tokens of the same length separate capacity from content. This also answers GRAVA's density analysis.

---

## Open Questions

*Wiki-wide open questions are collected in [[research-directions.md]].*

- Does any method's trace remain predictive of its action on **navhard** or other shifted inputs, where hindsight-derived maneuvers are least reliable?
- Is a *verbalized action* sufficient for the practical uses people want from CoT (auditability, human override, rule checking)? If so, faithfulness to evidence may matter less than claimed.
- Rule-grounded traces derived from an executable planner (Neuro-Symbolic Drive, not yet ingested) are faithful to the planner by construction. Do they transfer to learned planners?

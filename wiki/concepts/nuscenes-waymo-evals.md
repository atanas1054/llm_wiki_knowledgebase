---
title: nuScenes and Waymo Evaluations
type: concept
sources: [raw/papers/AutoVLA_ A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning.md, raw/papers/HERMES_ A Holistic End-to-End Risk-Aware Multimodal Embodied System with Vision–Language Models for Long-Tail Autonomous Driving.md, raw/papers/UniUGP_ Unifying Understanding, Generation, and Planing For End-to-end Autonomous Driving.md, raw/papers/Reasoning-VLA_ A Fast and General Vision-Language-Action Reasoning Model for Autonomous Driving.md, raw/papers/DriveVA_ Video Action Models are Zero-Shot Drivers.md, raw/papers/ExploreVLA_ Dense World Modeling and Exploration for End-to-End Autonomous Driving.md, raw/papers/OneDrive_ Unified Multi-Paradigm Driving with Vision-Language-Action Models.md, raw/papers/From Forecasting to Planning_ Policy World Model for Collaborative State-Action Prediction.md]
related: [concepts/navsim-benchmark.md, concepts/bench2drive.md, concepts/world-model-for-ad.md, sources/autovla.md, sources/hermes.md, sources/uniugp.md, sources/reasoning-vla.md, sources/driveva.md, sources/explorevla.md, sources/onedrive.md, sources/policy-world-model.md]
created: 2026-05-01
updated: 2026-05-01
confidence: high
---

## What They Measure

nuScenes and Waymo-style evaluations in this wiki are mostly open-loop: L2 displacement, collision proxy metrics, planning error, or WaymoE2E risk/route scores. They are useful for trajectory imitation and transfer, but they do not replace closed-loop NAVSIM or interactive Bench2Drive evaluation.

## Common Metrics

| Metric family | Typical use | Caveat |
| --- | --- | --- |
| L2/ADE/FDE | nuScenes trajectory accuracy | Rewards matching logged behavior, not necessarily safe closed-loop behavior. |
| Collision rate | nuScenes/Waymo proxy safety | Often computed against logged agents without reactive simulation. |
| RFS | WaymoE2E long-tail risk | Dataset/task-specific; not comparable to PDMS or DS. |
| FID/FVD | World-model visual quality | Video realism does not guarantee planning quality. |

## Takeaways

- Treat nuScenes/Waymo as complementary evidence for generalization, not as direct leaderboard substitutes for NAVSIM or Bench2Drive.
- Zero-shot transfer claims should report absolute values, not only percent improvement over one baseline.
- World-model papers need both generation metrics and downstream planning metrics; strong FVD alone is insufficient.

## OneDrive nuScenes Result

**OneDrive** ([[sources/onedrive.md]]) reports one of the strongest nuScenes open-loop planning entries in the wiki:

| Method | L2 Avg | Collision Avg | Notes |
| --- | --- | --- | --- |
| SOLVE-VLM | 0.28 | 0.20 | AR/text VLM path |
| ColaVLA | 0.30 | 0.23 | Non-AR baseline |
| **OneDrive** | **0.28** | **0.18** | Single causal decoder; detection/lane/planning query sequence |

This is meaningful evidence for the architecture, but it remains open-loop: the result should not be treated as equivalent to NAVSIM PDMS or Bench2Drive driving score.

## Policy World Model nuScenes Result

**Policy World Model** ([[sources/policy-world-model.md]]) reports a safety-skewed nuScenes result: it does not dominate L2, but it has the lowest collision rate in its comparison table.

| Method | Ego status | L2 Avg | Collision Avg | Notes |
| --- | --- | ---: | ---: | --- |
| PWM | No | 0.78 | 0.07 | Better collision than Drive-OccWorld 0.11 and LAW 0.19; worse L2 than those methods. |
| PWM | Yes | 0.41 | 0.04 | Best collision in the paper's ego-status table; L2 trails Omni-Q 0.33, BEV-Planner 0.35, and VAD-Base 0.37. |

The result is useful evidence for future-frame forecasting as a safety prior. It should still be interpreted as open-loop nuScenes evidence, not as proof of closed-loop behavior under interactive agents.

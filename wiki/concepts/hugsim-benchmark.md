---
title: HUGSIM Benchmark
type: concept
sources: [raw/papers/HAD_ Combining Hierarchical Diffusion with Metric-Decoupled RL for End-to-End Driving.md, raw/papers/Latent-WAM_ Latent World Action Modeling for End-to-End Autonomous Driving.md]
related: [sources/had.md, sources/latent-wam.md, concepts/navsim-benchmark.md, concepts/bench2drive.md]
created: 2026-05-01
updated: 2026-05-01
confidence: medium
---

## What It Is

HUGSIM is a closed-loop autonomous driving benchmark used by HAD to test interactive planning beyond NAVSIM's non-reactive simulator. The benchmark includes more than 400 simulation scenarios split by difficulty: Easy, Medium, Hard, and Extreme.

## Metrics

HUGSIM reports route completion (RC) and HD-Score (HDS). HD-Score combines safety and driving-quality terms, including no-collision and drivable-area compliance with weighted time-to-collision and comfort components, then scales by route completion.

These numbers are not directly comparable to NAVSIM PDMS or EPDMS. HUGSIM is a separate closed-loop benchmark with different scenario construction, agent interaction, and scoring.

## HAD Result

| Method | Easy RC | Easy HDS | Medium RC | Medium HDS | Hard RC | Hard HDS | Extreme RC | Extreme HDS | Overall RC | Overall HDS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HAD-L | 65.9 | 51.2 | 52.1 | 34.9 | 50.8 | 30.4 | 39.1 | 22.5 | 47.5 | 30.8 |
| Latent-WAM | 84.2 | 72.5 | 42.5 | 24.0 | 30.6 | 12.2 | 35.5 | 18.1 | 45.9 | 28.9 |

HAD-L's HUGSIM result is useful because it evaluates the same planner family outside NAVSIM and includes an extreme split where the model drops to 39.1 RC / 22.5 HDS. The paper reports public-split results for most baselines; starred baselines use public+private scenarios, so comparison scope should be checked before treating the table as a clean leaderboard.

Latent-WAM ([[sources/latent-wam.md]]) reports zero-shot HUGSIM using the NAVSIM-v2-trained model. It has stronger Easy RC/HDS than HAD-L but lower overall HDS, mainly because Medium and Hard HDS are weaker. Treat the two rows as evidence of different generalization profiles rather than a clean universal ranking.

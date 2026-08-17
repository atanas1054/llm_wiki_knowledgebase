---
title: PhysicalAI-Autonomous-Vehicles Benchmark
type: concept
sources: [raw/papers/DriveWAM_ Video Generative Priors Enable Scalable World-Action Modeling for Autonomous Driving.md, raw/papers/Alpamayo-R1_ Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail.md]
related: [sources/drivewam.md, sources/alpamayo-r1.md, concepts/nuscenes-waymo-evals.md, concepts/navsim-benchmark.md, concepts/world-model-for-ad.md]
created: 2026-08-17
updated: 2026-08-17
confidence: medium
---

## What It Is

A large-scale real-world driving benchmark released by NVIDIA alongside Alpamayo-R1 ([[sources/alpamayo-r1.md]]). Per [[sources/drivewam.md]] (the first wiki source to evaluate on it):

- **~1,700 hours** of driving logs
- **306,152 clips** of 20 seconds each
- Official splits: **153,625 train / 90,928 val / 61,599 test**
- Front-view camera stream + ego-motion labels (multi-sensor logs exist; papers so far use front-view only)

Note: the wiki's [[sources/alpamayo-r1.md]] page (ingested 2026-04) recorded "all evaluations on internal NVIDIA datasets" as a limitation. DriveWAM's usage shows the dataset side has since been publicly released as a benchmark — that limitation is now partially superseded, though Alpamayo's AlpaSim closed-loop evaluation remains internal.

## Metrics

Open-loop trajectory imitation: **ADE** (Average Displacement Error) and **FDE** (Final Displacement Error) over 3-second and 4-second future horizons. All caveats from [[concepts/nuscenes-waymo-evals.md]] apply — displacement error against a logged human trajectory is not closed-loop driving quality, and there is no reactive simulation.

## Reported Results (DriveWAM's curated 1,000-clip test subset)

| Method | Params | ADE@3s ↓ | FDE@3s ↓ | ADE@4s ↓ | FDE@4s ↓ | Training data |
|---|---|---:|---:|---:|---:|---|
| VaVAM (released ckpt, ≤3s) | 1.3B | 2.31 | 4.32 | – | – | ~1,700 h OpenDV |
| Alpamayo-1.5 | 10B | 0.80 | 2.31 | 1.44 | 4.18 | ~80,000 h (incl. PhysicalAI-AV train) |
| **DriveWAM** | 5B + 8B | **0.47** | **1.35** | **0.83** | **2.47** | 100k curated clips (~556 h) |

DriveWAM roughly halves Alpamayo-1.5's ADE/FDE at both horizons while training on ~2 orders of magnitude less data — though on in-distribution curated clips, with a frozen 8B VLM in the loop.

## Test-Subset Caveat

There is currently **no standard public test protocol** in the wiki's sources: DriveWAM curates its own 1,000-clip test subset (VLM tagging with Qwen3-VL-8B, rule-weighted interest scores; rare-event + high-interest + 200 common-scene clips). Comparisons on this subset are the curating paper's own construction; VaVAM is further handicapped (checkpoint supports only 3s), and Alpamayo-1.5 is evaluated under a single-trajectory front-camera protocol chosen by DriveWAM. Treat cross-method numbers as indicative, not leaderboard-grade, until an official test protocol is adopted by multiple papers.

## Role in the Wiki

This is the wiki's first large-scale *real-world data-scaling* benchmark: DriveWAM's 4k → 20k → 100k clip study (ADE@4s 1.01 → 0.94 → 0.83 with guidance) is run here, complementing NAVSIM (closed-loop non-reactive, small) and Bench2Drive (CARLA closed-loop, synthetic).

## Open Questions

- Will other groups adopt the official 61,599-clip test split (or a shared subset) so results become comparable across papers?
- Does performance on curated rare-event clips predict closed-loop behavior? No paper has yet paired PhysicalAI-AV with a reactive evaluation.

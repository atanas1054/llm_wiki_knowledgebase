---
title: Navhard and OOD Evaluation
type: concept
sources: [raw/papers/DriveFine_ Refining-Augmented Masked Diffusion VLA for Precise and Robust Driving.md, raw/papers/SpanVLA_ Efficient Action Bridging and Learning from Negative-Recovery Samples for Vision-Language-Action Model.md, raw/papers/HAD_ Combining Hierarchical Diffusion with Metric-Decoupled RL for End-to-End Driving.md]
related: [concepts/navsim-benchmark.md, concepts/rl-for-ad.md, sources/drivefine.md, sources/spanvla.md, sources/had.md]
created: 2026-05-01
updated: 2026-05-01
confidence: medium
---

## What It Is

Navhard is the harder NAVSIM-v2-style evaluation regime surfaced in recent papers to test out-of-distribution and difficult planning scenes beyond standard navtest.

## Why It Matters

Standard NAVSIM-v1 PDMS is increasingly saturated by strong single-sample and Best-of-N methods. Navhard exposes a wider gap: SpanVLA reports 40.1 EPDMS on navhard despite 86.4 EPDMS on NAVSIM-v2 navtest, showing that high navtest scores do not imply robust OOD driving.

## Current Wiki Evidence

| Method | Navhard result | Caveat |
| --- | --- | --- |
| DriveFine | Reports navhard evaluation in its robustness section | Uses its own scorer/regime details; compare carefully. |
| SpanVLA | 40.1 EPDMS for both reported stages | Submetrics differ while headline EPDMS is identical, so stage interpretation is uncertain. |
| HAD-L | 32.3 EPDMS | Better than DiffusionDrive/LTF under the paper's Transfuser-style comparison, but below DriveSuprim and GTRS-Dense. |

## HAD-L NavHard Result

HAD-L reports 32.3 EPDMS on NavHard. The result is useful because the same model is much stronger on standard NAVSIM-v2 navtest (88.5 EPDMS), so the gap quantifies the OOD brittleness of a high-performing real-time non-VLM planner.

The paper attributes part of the remaining gap to BEV feature sensitivity under 3DGS image synthesis noise. Treat this as a model-specific caveat rather than a general conclusion about all diffusion planners.

## Lint Rule

When a paper claims NAVSIM progress, check whether it reports navhard or another OOD split. If not, mark the claim as standard-split only.

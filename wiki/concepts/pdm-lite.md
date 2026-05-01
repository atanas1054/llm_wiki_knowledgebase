---
title: PDM-Lite
type: concept
sources: [raw/papers/UniDriveVLA_ Unifying Understanding, Perception, and Action Planning for Autonomous Driving.md, raw/papers/Unifying Language-Action Understanding and Generation for Autonomous Driving.md, raw/papers/AutoMoT_ A Unified Vision-Language-Action Model with Asynchronous Mixture-of-Transformers for End-to-End Autonomous Driving.md]
related: [concepts/bench2drive.md, concepts/selection-based-planning.md, sources/unidrivevla.md, sources/linkvla.md, sources/automot.md]
created: 2026-05-01
updated: 2026-05-01
confidence: medium
---

## What It Is

PDM-Lite is a privileged planner/fallback used in some Bench2Drive comparisons. It can use information that a normal end-to-end policy would not necessarily have at deployment, such as privileged route or map/waypoint structure.

## Why It Matters

Bench2Drive tables often mix methods with and without PDM-Lite-style assistance. This can inflate driving score and success rate relative to pure learned policies. UniDriveVLA explicitly reports a "best without PDM-Lite" framing; LinkVLA and other Bench2Drive leaders should be compared only after checking whether privileged assistance is used.

## Lint Rule

Any Bench2Drive SOTA claim should state whether PDM-Lite or a comparable privileged fallback is used. If the paper does not disclose this clearly, mark the comparison as uncertain rather than treating it as a clean end-to-end leaderboard result.

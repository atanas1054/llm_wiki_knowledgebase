---
title: Navhard and OOD Evaluation
type: concept
sources: [raw/papers/DriveFine_ Refining-Augmented Masked Diffusion VLA for Precise and Robust Driving.md, raw/papers/SpanVLA_ Efficient Action Bridging and Learning from Negative-Recovery Samples for Vision-Language-Action Model.md, raw/papers/HAD_ Combining Hierarchical Diffusion with Metric-Decoupled RL for End-to-End Driving.md, raw/papers/GeoWAM_ Visual Geometry World Action Models for Autonomous Driving.md]
related: [concepts/navsim-benchmark.md, concepts/hugsim-benchmark.md, concepts/rl-for-ad.md, concepts/world-model-for-ad.md, sources/drivefine.md, sources/spanvla.md, sources/had.md, sources/geowam.md, sources/drivelaw.md, sources/drivevla-w0.md]
created: 2026-05-01
updated: 2026-09-02
confidence: medium
---

## What It Is

Navhard is NAVSIM-v2's hard split, evaluated under a **two-stage pseudo-closed-loop protocol** built on 3D Gaussian Splatting reconstructions. The planner predicts an ego trajectory; the benchmark renders a new observation from the resulting ego pose and feeds it back for the next planning step. Planning errors therefore change what the planner subsequently sees, so the benchmark measures whether a model recovers from its own accumulated deviations — something standard navtest cannot test.

Stage 1 evaluates the original scenes; Stage 2 evaluates synthetic reactive scenes. The official aggregation multiplies stage scores per group and branch before averaging, so a single combined EPDMS is roughly the *product* of stage performance — which is why combined navhard numbers sit near 30 while per-stage numbers sit near 80.

## Why It Matters

NAVSIM-v1 PDMS is saturated (Best-of-6 already matches human ground truth — see [[concepts/best-of-n.md]]) and navtest EPDMS is compressed into a few points at the top. Navhard is not close to saturated. The best combined score in the wiki is **36.6**, and the gap between methods is large enough to rank them.

It is also the wiki's cheapest reactive evaluation. [[concepts/bench2drive.md]] needs CARLA and [[concepts/hugsim-benchmark.md]] needs its own splatting pipeline; navhard runs inside the NAVSIM stack a paper is already using.

## ⚠ Two Reporting Conventions

Papers report navhard in two incompatible ways, and the wiki has been mixing them:

- **Combined EPDMS** — one number spanning both stages, roughly the product of stage performance. [[sources/geowam.md]] and its baselines use this. Values land in the 11–37 range.
- **Per-stage EPDMS** — separate Stage 1 and Stage 2 numbers. [[sources/drivefine.md]] and [[sources/spanvla.md]] use this. Values land in the 40–75 range.

A per-stage pair can be *roughly* converted by multiplying (DriveFine's 74.4 / 41.0 gives about 30.5), but this is an estimate only — the official aggregation multiplies within each group and branch before averaging, which is not the same as multiplying the averages. Do not treat converted values as leaderboard entries.

[[sources/had.md]]'s 32.3 is ambiguous: it is reported as a single number with no Stage 2 companion, which is consistent with either convention. It sits plausibly in the combined range next to DVGT-2's 31.7, but the wiki cannot confirm this.

## Combined-EPDMS Leaderboard

From [[sources/geowam.md]] Table 3. Methods marked † are trained with reinforcement learning or direct PDMS-score supervision; the paper greys them out to separate supervision regimes.

| Method | Combined EPDMS ↑ | Notes |
|---|---:|---|
| **GeoWAM** | **36.6** | Geometry world model, deterministic $\ell_1$ regression head, no RL |
| EponaV2 † | 36.1 | *not ingested* |
| NavFormer † | 34.1 | *not ingested* |
| LTFv6 / LEAD † | 31.9 | *not ingested* |
| DVGT-2 | 31.7 | *not ingested* — GeoWAM's own initialization |
| [[sources/drivelaw.md]] | 30.6 | Video-DiT mid-denoising latents as planning state |
| LTF | 25.1 | Transfuser-family baseline |
| [[sources/drivevla-w0.md]] | 24.4 | AR + diffusion world models, training-time only |
| Ego MLP | 14.1 | Ego-status-only baseline |
| Constant velocity | 11.4 | Floor |

**GeoWAM leads while using strictly weaker supervision than the three methods below it.** EponaV2, NavFormer, and LTFv6 all use RL or direct PDMS-score supervision; GeoWAM uses $\ell_1$ trajectory regression. Its margin over EponaV2 is only +0.5, so the ranking is not robust — but the supervision asymmetry runs against it, which makes the result more interesting than the gap size suggests.

**The +4.9 over DVGT-2 is the load-bearing number.** On navtest GeoWAM beats its own DVGT-2 initialization by only +0.6; on navhard the same architectural addition — future-geometry forecasting — is worth eight times more. That is precisely what a world-model thesis predicts: anticipation should matter most where errors compound. It is the strongest evidence in the wiki that world modeling buys robustness rather than open-loop accuracy, and neither GeoWAM nor any other paper remarks on it.

## Stage 2 Is Where Everything Collapses

The per-stage submetrics in GeoWAM's table expose a failure signature the aggregate scores hide. Every method — including the constant-velocity baseline — loses roughly half its **lane keeping** between stages:

| Method | LK Stage 1 | LK Stage 2 | NC Stage 1 | NC Stage 2 |
|---|---:|---:|---:|---:|
| Constant velocity | 78.6 | 47.9 | 88.8 | 83.2 |
| Ego MLP | 83.5 | 40.8 | 93.2 | 77.2 |
| LTF | 94.2 | 45.4 | 96.2 | 77.7 |
| DriveVLA-W0 | 96.4 | 46.8 | 96.8 | 76.8 |
| DriveLaW | 96.2 | 45.8 | 97.3 | 82.5 |
| DVGT-2 | 95.5 | 48.0 | 97.2 | 77.8 |
| EponaV2 † | 97.3 | 50.1 | 97.3 | 83.6 |
| GeoWAM | 96.0 | 49.9 | 97.7 | 80.4 |

Lane keeping falls from ~96 to ~48 for every learned planner, and no-at-fault collision from ~97 to ~80. **The spread between the best and worst learned method on Stage 2 LK is under 5 points, while the Stage 1 spread is over 12** — under the reactive protocol, methods that look clearly separated collapse toward a common failure mode. Extended comfort behaves similarly, falling from the 60–79 band to 45–67.

This is the same picture [[concepts/hugsim-benchmark.md]] shows on its Extreme tier, where every method lands between 0.06 and 0.14 HD-Score. Two independent reactive benchmarks agree: **current planners degrade to near-indistinguishable once their own errors drive the observations**, and open-loop rankings do not predict which degrade least.

## Per-Stage Reports

| Method | Stage 1 EPDMS | Stage 2 EPDMS | Caveat |
| --- | ---: | ---: | --- |
| [[sources/drivefine.md]] | 74.4 | 41.0 | Leads Stage 1 by +5.5 over ReCogDrive; approx. 30.5 if converted to combined |
| ReCogDrive | 68.9 | 37.8 | Reported within DriveFine's table |
| DiffusionDrive | 66.7 | 40.5 | Reported within DriveFine's table |
| [[sources/spanvla.md]] | 40.1 | 40.1 | Identical headline for both stages while submetrics differ — stage interpretation uncertain |
| [[sources/had.md]] | 32.3 | – | Convention ambiguous; see the warning above |

SpanVLA's 40.1 EPDMS on navhard against 86.4 on navtest remains the wiki's cleanest single-method statement of the gap: **high navtest scores do not imply robust OOD driving.** HAD-L makes the same point from 88.5 navtest down to 32.3, and attributes part of it to BEV feature sensitivity under 3DGS synthesis noise — a model-specific diagnosis, not a general one.

## Open Questions

- **Does the combined/per-stage split hide a real ranking?** DriveFine's 74.4/41.0 converts to roughly 30.5, which would place it below DVGT-2 and DriveLaW — but the conversion is an approximation of an aggregation the wiki has not verified. Someone reporting both conventions for one checkpoint would settle it in one run.
- **Why does world modeling help eight times more on navhard than navtest?** GeoWAM's +0.6 / +4.9 split over DVGT-2 is the only measurement of this in the wiki, from a single paper with no ablations. If it replicates, it reframes what world-model pretraining is *for* — robustness under compounding error rather than open-loop accuracy — and implies navtest is the wrong benchmark for evaluating it.
- **Is Stage 2 lane-keeping collapse a planner failure or a rendering artifact?** Every method including constant velocity loses about half its LK, which is suspicious. If 3DGS renderings degrade as the ego pose leaves the recorded trajectory, part of the drop measures the benchmark rather than the planner. No paper has separated these.
- **Does RL help here?** Three of the four methods above 31 use RL or PDMS-score supervision, but GeoWAM tops them without it. With margins of 0.5–2.5 points and single runs, the wiki cannot say whether RL buys OOD robustness.

## Lint Rule

When a paper claims NAVSIM progress, check whether it reports navhard or another OOD split. If not, mark the claim as standard-split only. If it does, **check which reporting convention it uses** before placing the number.

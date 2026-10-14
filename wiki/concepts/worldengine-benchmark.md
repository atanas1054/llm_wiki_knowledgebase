---
title: WorldEngine Closed-Loop Benchmark
type: concept
sources: ["raw/papers/ManeuverFlow_ Maneuver Hypothesis Generation with Future Evidence for Trajectory Evaluation and Refinement in End-to-End Autonomous Driving.md", "raw/papers/SimVTA_ Simulation-Grounded Video-Trajectory Alignment for Driving World Action Models.md"]
related: [sources/maneuverflow.md, sources/simvta.md, sources/simscale.md, concepts/data-driven-simulators.md, concepts/navsim-benchmark.md, concepts/navhard-ood-evaluation.md, concepts/hugsim-benchmark.md, concepts/alpasim-benchmark.md, concepts/evaluation-variance.md, research-directions.md]
created: 2026-10-14
updated: 2026-10-14
confidence: low
---

# WorldEngine Closed-Loop Benchmark

## What It Is

**World Engine** (arXiv 2606.19836, OpenDriveLab and collaborators; not ingested) is a 3D Gaussian Splatting simulator over reconstructed NAVSIM scenes, described by its authors as infrastructure for "post-training" driving policies. It appears in the wiki in two roles within two days of ingests:

| Role | Paper | What World Engine does |
|---|---|---|
| **Training renderer** | [[sources/simvta.md]] | Renders front-view futures along the policy's own executed trajectories on navtrain scenes, as targets for a world-action model's video loss during RL |
| **Closed-loop benchmark** | [[sources/maneuverflow.md]] | Runs a planner closed-loop on "all 288 publicly released high-risk scenarios reconstructed from NAVSIM navtest logs", re-rendering the cameras as the ego moves |

This page is about the benchmark. Everything known about it here comes from one results table in [[sources/maneuverflow.md]] and the renderer measurements in [[sources/simvta.md]]; the benchmark paper itself is not ingested, so the confidence is **low**.

## Protocol (as described by ManeuverFlow)

- **Scenarios**: 288 high-risk scenarios reconstructed from **navtest** logs, the NAVSIM evaluation split. How "high-risk" was selected is not described.
- **Traffic**: **Log** (background agents replay the recording) or **IDM** (agents follow the Intelligent Driver Model and react to the ego).
- **Ego**: replans at every step from the newly rendered observations.
- **Sensors**: cameras only. There is no rendered LiDAR, so camera+LiDAR planners cannot run; ManeuverFlow evaluates only its camera-only LTF variant.
- **Metrics**: **SR**, the fraction of episodes without a collision or drivable-area violation; **EP**, closed-loop ego progress; **PDMS\***, a composite of closed-loop driving quality. PDMS\* is not NAVSIM's PDMS, and its formula is not given in the ingested paper.

## Results

From [[sources/maneuverflow.md]] Table 3. ManeuverFlow's values are means ± sd over three trained checkpoints; the baselines' source is not stated and they carry no spread.

| Method | Traffic | SR ↑ | EP ↑ | PDMS\* ↑ | NAVSIM v1 PDMS (same paper) |
|---|---|---:|---:|---:|---:|
| VADv2 | Log | 77.08 | 48.00 | 63.81 | 89.3 (version uncertain) |
| VADv2 | IDM | 73.66 | 46.71 | 60.98 | |
| DiffusionDrive | Log | 86.24 | 45.09 | 69.83 | 88.1 |
| DiffusionDrive | IDM | 85.85 | 43.88 | 68.22 | |
| MeanFuser | Log | 87.25 | 55.90 | 73.81 | 89.0 |
| MeanFuser | IDM | 87.15 | 55.60 | 73.28 | |
| **ManeuverFlow (LTF)** | Log | **87.48 ± 0.92** | **60.68 ± 0.91** | **75.19 ± 0.85** | 89.98 |
| **ManeuverFlow (LTF)** | IDM | **87.92 ± 0.87** | **60.89 ± 0.81** | **74.58 ± 0.64** | |

### What the table shows

- **Reactive traffic barely changes the scores.** IDM against Log: ManeuverFlow −0.61 PDMS\*, MeanFuser −0.53, DiffusionDrive −1.61, VADv2 −2.83; ManeuverFlow's success rate goes *up* under IDM. Either the 288 scenarios are rarely decided by how other agents respond, or IDM agents behave close to the log in them. The wiki cannot tell which.
- **The leaders separate on progress, not safety.** ManeuverFlow, MeanFuser and DiffusionDrive are within 1.7 points of each other on SR; EP spans 43.9–60.9.
- **Closed-loop seed variance is about ten times NAVSIM's.** ManeuverFlow's three checkpoints have sd 0.64–0.92 on PDMS\*, against 0.05 on navtest PDMS. Its margin over MeanFuser (+1.4 Log, +1.3 IDM) is 1.5–2 of its own sds; MeanFuser's spread is unknown. See [[concepts/evaluation-variance.md]].
- **One NAVSIM-order inversion, if the VADv2 rows are the same model.** In ManeuverFlow's tables VADv2 is third of the four on NAVSIM v1 (89.3, above MeanFuser's 89.0) and last here by 10 points. The 89.3 row is itself unlike every other VADv2 row in the wiki, so the two may be different checkpoints.

## How It Relates to the Wiki's Other Rendered Benchmarks

| | WorldEngine | [[concepts/navhard-ood-evaluation.md]] (Stage 2) | [[concepts/hugsim-benchmark.md]] | [[concepts/alpasim-benchmark.md]] |
|---|---|---|---|---|
| Source logs | NAVSIM navtest | NAVSIM (navhard subset) | nuScenes, KITTI-360, Waymo, PandaSet | PAI-AV (NVIDIA) |
| Closed loop | Yes, replanning every step | Two stages, not a rollout | Yes | Yes |
| Reactive agents | Optional (IDM) | Synthetic reactive scenes | Yes | Yes |
| Scenarios | 288 | 244 or 450 Stage-1 scenes (disputed) | 345 or 436 | 916 |
| Domain relative to NAVSIM training | **Same logs as navtest** | Same dataset family | Zero-shot | Different dataset |
| Ingested users | 1 | Many | 6 | 1 |

- **It is in-domain.** Its scenes come from navtest, the split NAVSIM methods are tuned on, so it tests closed-loop competence on familiar scenes rather than transfer, which HUGSIM and AlpaSim test.
- **Renderer quality is measured only on logged paths.** [[sources/simvta.md]] reports World Engine renders at FID 7.6 / FVD 21.8 against real frames along logged trajectories, and concedes that renders degrade away from logged poses (its Figure 4 shows smearing near a bus and a road shoulder). In a closed-loop benchmark the ego leaves the log by design.
- **A train–test renderer confound is now possible.** A policy post-trained on World Engine renders, as in [[sources/simvta.md]], and then scored on WorldEngine would be trained and tested in the same renderer, the situation [[concepts/data-driven-simulators.md#shortcut]] flags for SimScale and navhard. No paper has done both yet.
- **Shared authorship.** World Engine, SimScale, SimVTA and the NAVSIM/pseudo-simulation benchmark papers share OpenDriveLab authors ([[sources/simvta.md]], [[sources/simscale.md]]).

## Open Questions

*Wiki-wide open questions are collected in [[research-directions.md]].*

- **What is PDMS\*, and how are the 288 scenarios selected?** Both are in the World Engine paper, which the wiki has not ingested.
- **Does WorldEngine rank methods the way HUGSIM and navhard do?** No method in the wiki has been run on WorldEngine and another closed-loop benchmark.
- **Why does reactive traffic matter so little?** A per-scenario comparison of Log and IDM runs would show whether IDM agents ever change the outcome in these scenes.
- **Is a WorldEngine gain transfer or renderer familiarity** for a policy trained on World Engine renders? A model trained without renders, evaluated against one trained with them, on both WorldEngine and HUGSIM, would separate them.

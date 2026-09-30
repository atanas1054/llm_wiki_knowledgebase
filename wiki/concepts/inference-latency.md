---
title: Inference Latency and Cost
type: concept
sources: ["raw/papers/MM-Future_ Multi-Mode Joint World–Action Modeling for Autonomous Driving.md", "raw/papers/DriveReferee_ Geometric Safety Verdicts Need Not Be Learned for Driving World-Action Models.md", "raw/papers/WALT_ Learning World-Model-Aligned Latent Trajectories for Autonomous Driving.md", "raw/papers/MomWorld_ Momentum-Aware Latent World Model for Long-Horizon Autonomous Driving.md", "raw/papers/PhysWAM_ Physically Consistent World Action Model for Autonomous Driving.md", "raw/papers/AD-E2E-JEPA_ A Joint-Embedding Predictive Architecture For End-to-End Autonomous Driving.md", raw/papers/DiffusionDrive_ Truncated Diffusion Model for End-to-End Autonomous Driving.md, raw/papers/HAD_ Combining Hierarchical Diffusion with Metric-Decoupled RL for End-to-End Driving.md, raw/papers/Unifying Language-Action Understanding and Generation for Autonomous Driving.md, raw/papers/PlannerRFT_ Reinforcing Diffusion Planners through Closed-Loop and Sample-Efficient Fine-Tuning.md, raw/papers/DriveVLA-W0_ World Models Amplify Data Scaling Law in Autonomous Driving.md, raw/papers/Latent-WAM_ Latent World Action Modeling for End-to-End Autonomous Driving.md, raw/papers/Drive-HWM_ Hierarchical World Models for Dynamic-Latent Guided Autonomous Driving.md, raw/papers/Alpamayo-R1_ Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail.md, "raw/papers/Metis_ A Generalizable and Efficient World-Action Model for Autonomous Driving and Urban Navigation.md", raw/papers/OneDrive_ Unified Multi-Paradigm Driving with Vision-Language-Action Models.md, raw/papers/Adaptive-WAM_ Quality-Guided Early-Exit Planningfrom Intermediate Video-Diffusion Features.md, raw/papers/DriveFine_ Refining-Augmented Masked Diffusion VLA for Precise and Robust Driving.md, raw/papers/OneVL_ One-Step Latent Reasoning and Planning with Vision-Language Explanation.md, "raw/papers/SUV_ Future Scene Understanding as Video Generation for End-to-End Driving.md", raw/papers/SimWAM_ A Simple World Action Model for End-to-End Autonomous Driving.md, raw/papers/FeaXDrive_ Feasibility-aware Trajectory-Centric Diffusion Planning for End-to-End Autonomous Driving.md, raw/papers/BrainWAM_ Action-Space Coordination of Semantic Priors and Predictive Dynamics for Autonomous Driving.md, raw/papers/SpanVLA_ Efficient Action Bridging and Learning from Negative-Recovery Samples for Vision-Language-Action Model.md, raw/papers/AdaThinkDrive_ Adaptive Thinking via Reinforcement Learning for Autonomous Driving.md, raw/papers/Percept-WAM_ Perception-Enhanced World-Awareness-Action Model for Robust End-to-End Autonomous Driving.md, raw/papers/DriveWAM_ Video Generative Priors Enable Scalable World-Action Modeling for Autonomous Driving.md, raw/papers/See Tomorrow, Act Today_ Foresight-Driven Autonomous Driving.md, raw/papers/AutoVLA_ A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning.md, raw/papers/Epona_ Autoregressive Diffusion World Model for Autonomous Driving.md, "raw/papers/[-0.5mm] GRAVA GRAVA_ Grounded Reasoning-to-Action Representation and Learning for Autonomous Driving.md", raw/papers/Qwen-Drive-1.0_ An Initial Step towards a Vision-Language Foundation Model for Autonomous Driving.md, raw/papers/LWDrive_ Layer-Wise World-Model-Guided Vision-Language ModelPlanning for Autonomous Driving.md, raw/papers/DriveFuture_ Future-Aware Latent World Models for Autonomous Driving.md, raw/papers/WA-JEPA_ Rethinking the Video JEPA Paradigm forWorld-Action Modeling in Autonomous Driving.md]
related: [sources/mm-future.md, sources/drivereferee.md, sources/walt.md, sources/momworld.md, sources/physwam.md, concepts/hugsim-benchmark.md, sources/ad-e2e-jepa.md, sources/da-wam.md, sources/dreameraD.md, concepts/selection-based-planning.md, sources/diffusiondrive.md, sources/had.md, sources/linkvla.md, sources/plannerrft.md, sources/drivevla-w0.md, sources/latent-wam.md, sources/drive-hwm.md, sources/alpamayo-r1.md, sources/metis.md, sources/onedrive.md, sources/adaptive-wam.md, sources/drivefine.md, sources/onevl.md, sources/suv.md, sources/simwam.md, sources/feaxdrive.md, sources/brainwam.md, sources/spanvla.md, sources/adathinkdrive.md, sources/percept-wam.md, sources/drivewam.md, sources/foresight.md, sources/autovla.md, sources/epona.md, sources/grava.md, sources/qwen-drive-1.0.md, sources/lwdrive.md, sources/drivefuture.md, sources/wa-jepa.md, concepts/world-model-for-ad.md, concepts/chain-of-thought-for-ad.md, concepts/diffusion-planner.md, concepts/action-tokenization.md, concepts/wam-attention-masks.md, concepts/evaluation-variance.md, research-directions.md]
created: 2026-09-27
updated: 2026-09-30
confidence: medium
---

# Inference Latency and Cost

## What It Tracks

What a planner costs per decision, what that cost is made of, and whether the reported numbers can be compared. Latency is mentioned on 49 source pages, but **no two papers in the wiki report it under the same hardware, batch size and pipeline boundary.** This page collects the numbers and the structural lessons, and it keeps a list of papers that report none.

---

## The Recorded Numbers

The hardware column is what the source page records. "–" means the paper does not say. Scores are NAVSIM-v1 PDMS unless marked.

| Method | Latency | Hardware | Score at that setting | What dominates |
|---|---:|---|---|---|
| [[sources/plannerrft.md]] | 34.3 ms | – | nuPlan planner | 5-step DDIM diffusion planner |
| [[sources/diffusiondrive.md]] | ~22 ms (45 FPS) | – | 88.1 | 2-step truncated diffusion, 60M decoder |
| [[sources/had.md]] | ~33 ms (30.4 FPS) | A100 | 90.2 | Hierarchical diffusion, non-VLM |
| [[sources/linkvla.md]] | 48 ms (**CoT excluded**) | – | 91.01 DS (Bench2Drive) | Coarse-to-fine two-pass decode; autoregressive (AR) decoding was 361 ms |
| [[sources/drivevla-w0.md]] | 74–240 ms (own); **690 ms as measured by SUV** | – / RTX 4090 | 88.4–90.2 | Action-expert variant |
| [[sources/alpamayo-r1.md]] | 99 ms (40 reasoning tokens) | – | internal | Flow-matching (FM) decode 8.75 ms vs AR 222 ms |
| [[sources/latent-wam.md]] | 107 ms | – | v2 | Compact latent world model |
| [[sources/drive-hwm.md]] | 84.8 ms per step; ≈678 ms if one step is one waypoint | H200 | 93.8 / 93.3 | Emu3-8B fast model (81.6 ms) |
| [[sources/momworld.md]] | 138.9 ms (7.2 FPS); 130.5 ms without its refinement flow | – | nuScenes 6 s avg L2 1.17 (**the nuScenes model, not the NAVSIM one**) | MomAD-class sparse planner plus a 40-step rollout of two 256-d vectors; the 4-step residual flow adds 8.4 ms (≈2.1 ms per step) |
| [[sources/metis.md]] | **147 ms** (2 steps); 480 ms (10) | RTX 4090 | 89.2 / 89.5 EPDMS | Flow sampler steps; no video at inference |
| [[sources/onedrive.md]] | 156 ms | H20 | 86.8 | Unified causal decoder |
| [[sources/adaptive-wam.md]] | **170 ms** avg (284 ms worst case) | A100 | 89.9 EPDMS | Early exit from one Wan forward pass |
| [[sources/drivefine.md]] | 207 ms (4 steps) | – | 90.47 | Masked-diffusion steps |
| [[sources/mm-future.md]] | **233 ms** (64 paired hypotheses); 132 ms (32); 79 ms (16) | H800, bf16 | 93.4 (navtrain) / 94.0 (trainval); 93.3 at 32 | Two Euler steps over 64 × (256 future tokens + 8 action tokens) in a 16-layer, width-1024 transformer. Action-only at 32 hypotheses: 65 ms |
| [[sources/onevl.md]] | 0.24 s (MLP head) / 4.46 s (AR) | – | 86.83 / 88.84 | AR waypoint decoding |
| [[sources/suv.md]] | **288 ms** (2 steps); 1,356 ms (10) | RTX 4090 | 91.0 EPDMS | Joint denoising of 4 video streams |
| [[sources/feaxdrive.md]] | 349 ms | – | 90.0 | VLM backbone 245 ms |
| [[sources/brainwam.md]] | 475–644 ms | H20 | 89.5 | VLM + 1–3 video steps |
| [[sources/simwam.md]] | 518 ms (297 ms at 5 steps) | – | 91.5 (90.1 at 5 steps) | ≈45 ms per step + ≈70 ms fixed |
| [[sources/spanvla.md]] | 0.67 s | – | 90.3 | Adaptive CoT + FM expert |
| [[sources/adathinkdrive.md]] | 0.68 / 0.74 / 0.86 s | – | 88.3 / 90.3 / 88.9 | Non-think / adaptive / always-think |
| [[sources/percept-wam.md]] | 707 ms | – | 90.2 | Down from 1,174 ms after optimization |
| [[sources/drivewam.md]] | 871–1,262 ms per 4 s chunk | – | 90.1 | Chunked video generation |
| [[sources/foresight.md]] | 900 ms (**870 ms world model**) | H100 | 89.3 | Frozen generator run to a finished future |
| [[sources/autovla.md]] | ~1 Hz | – | 89.1 | AR action tokens + CoT |
| [[sources/physwam.md]] | **9.4 GPU-s** (30 steps); 4.4 (8 steps); 3.5 (4 steps). Medoid of 8: ×8 | RTX PRO 6000 | 90.3 EPDMS | Joint denoising of three-camera video + metric depth + motion in a 15.2B model. **Slowest planner recorded here** (next: OneVL's AR variant at 4.46 s) |
| [[sources/ad-e2e-jepa.md]] (zero-shot search) | **0.8 s** (256 candidates); 18.2 s (8,192) | A100 | 67.3 / 72.9 EPDMS, **oracle goal, not a planner score** | 8-step latent rollout per candidate, about 2.2–3.1 ms each. The same search on dense DINOv3 tokens takes 91.8 s (DINO-WM) / 101.0 s (JEPA-WM) |

The last row is a search over a world model, not a policy forward pass, and its score uses the ground-truth future frame. It is listed for its cost structure ([lesson 6](#per-candidate)). [[sources/ad-e2e-jepa.md]] reports no latency for its imitation-learning model.

**Simplicity claimed, latency not reported**: [[sources/redrive.md]] argues for an encoder-plus-planner pipeline with no inference-time future prediction (407M parameters, five flow steps, one camera) and gives no timing. It is probably among the cheaper 90+ entries; the paper does not let that be checked.

**Latency for a different model than the headline**: [[sources/momworld.md]] times its camera-only nuScenes model. Its NAVSIM model scores 16,384 candidates on 2048×512 camera + LiDAR input and has no timing.

**FLOPs in place of latency**: [[sources/walt.md]] reports 297 → 207 GFLOPs per denoising step for its trajectory head (8 waypoint tokens → 2 latent tokens). The frozen world model is excluded and neither a step count nor a wall-clock time is given.

**A selector priced, a policy not**: [[sources/drivereferee.md]] reports a median of 30 ms per candidate (CPU) for its analytic safety check and 1.34× policy samples on average, since a second sample is drawn in 34.3% of scenes. It gives no time for the policy sample itself, a 16B joint video–action generation. [[sources/physwam.md]], on the same backbone with three cameras and a depth stream, measured 9.4 GPU-seconds per plan.

**Report no latency at all**: [[sources/grava.md]], whose long box-token traces make it probably the most expensive AR trace here; [[sources/qwen-drive-1.0.md]]; [[sources/lwdrive.md]]; [[sources/drivefuture.md]]; [[sources/wa-jepa.md]], with 12-step joint scene denoising.

---

## Six Structural Lessons

### 1. For flow and diffusion planners, cost is the step count, and 2 steps usually suffice {#steps}

Three same-backbone Wan2.2-5B world-action models (WAMs) sweep the solver steps:

| Steps | Metis v2 / navhard / ms | SUV v2 / navhard / ms | SimWAM v1 / ms |
|---:|---|---|---|
| 1 | 87.2 / 30.4 / 110 | 89.8 / 33.0 / 177 | **68.9** (collapse) / – |
| 2 | **89.2** / 31.2 / **147** | **91.0** / 36.1 / **288** | – |
| 5 | 89.4 / 31.4 / 280 | – | 90.1 / 297 |
| 10 | 89.5 / 32.2 / 480 | 91.0 / 36.9 / 1,356 | 91.5 / 518 |

- **For Metis and SUV, two steps capture nearly all of the navtest score**, and the remaining steps buy 0.3–0.8 on navhard. **SimWAM is the exception.** Its action flow collapses at one step (68.9) and gives up 1.4 PDMS at five, so the step budget is architecture-specific, not universal. Latency is linear in steps: about 45 ms per step on SimWAM, and 37–40 ms on Metis. So the reported default of 10 steps is usually a choice of *headline*, not of deployment point.
- [[sources/diffusiondrive.md]] made the same argument for classic diffusion (20 DDIM steps at 130 ms → 2 truncated steps at 7.6 ms).

### 2. Generating video at inference is the most expensive thing a planner can do

| Paper | With future generation | Without | Ratio |
|---|---:|---:|---:|
| [[sources/foresight.md]] | 900 ms | – | 97% of the budget is the world model |
| [[sources/adaptive-wam.md]] | 13.22 s (full video) | 170 ms (intermediate feature) | ~78× |
| [[sources/metis.md]] | 1.38 s (10 steps) | 0.48 s (10 steps) | **2.9×** at matched steps |
| [[sources/epona.md]] | ~870 ms | 50 ms (20 Hz, planning-only) | ~17× |

[[sources/physwam.md]] sets the upper end. It generates three views of RGB and metric depth for every plan, with bidirectional attention between the future and the action, so there is no cheaper path to skip to:

| Steps | GPU-s per plan | Δ PDMS vs. 30 steps |
|---:|---:|---:|
| 4 | 3.5 | −0.6 (one sample) |
| 8 | 4.4 | +0.7 (three samples) |
| 15 | 6.0 | −0.4 (one sample) |
| 30 | 9.4 | 0 |

- **About 2.6 s is fixed** and each step adds about 0.23 s, so even four steps is 3.5 s.
- **The step count does not move the score outside sampler noise** (sd 0.2–0.3), which agrees with lesson 1. The 30-step headline is a protocol choice.
- **Consensus selection multiplies the bill.** The medoid of eight costs about 75 GPU-seconds per scene at 30 steps for +0.3 PDMS on navtest and +1.7 on navhard.
- **Its closed-loop HUGSIM result is in simulation time.** The simulator waits for the planner, so a 0.5 s replanning interval is served by a 9.4 s computation.
- The paper lists this as a limitation and notes that "an action-only inference path was not tested".

Watch for **step count and video generation being bundled into one headline speedup.** Metis's "8×" is 2.9× from skipping video times 3.3× from fewer steps. Whether video *at inference* is worth its cost is the [test-time imagination](world-model-for-ad.md#test-time-imagination) question. On navtest it is not. On navhard, one-way access buys +4.1 EPDMS ([[sources/suv.md]]) for about 2× the cost of the no-access variant, which SUV does not report.

### 3. Autoregressive decoding is the other large cost

- [[sources/linkvla.md]]: AR 361 ms → coarse-to-fine 48 ms.
- [[sources/alpamayo-r1.md]]: AR 222 ms vs flow matching 8.75 ms for the trajectory.
- [[sources/onevl.md]]: AR waypoints take 4.46 s vs 0.24 s for an MLP head.

Text chain-of-thought (CoT) adds cost roughly in proportion to its length. AdaThinkDrive's always-think costs +0.18 s over non-think, and its adaptive mode recovers most of that. **Purely AR VLAs that emit reasoning before actions ([[sources/grava.md]], [[sources/autovla.md]]) pay both costs, and GRAVA reports neither.** Per-token costs vary by 20× with deployment engineering: SpanVLA reports 33 ms/token against Alpamayo's 1.75 ms/token optimized.

### 4. The same model measures differently in different papers

[[sources/drivevla-w0.md]] reports 74–240 ms depending on its action expert. [[sources/suv.md]] times it at **690 ms on an RTX 4090** and Metis times Epona at 0.32 s, both under "the same protocol" as their own models. Pipeline boundaries (is the VLM prefill included? CoT? perception?), batch size and GPU generation all move the number. [[sources/linkvla.md]]'s 48 ms excludes CoT entirely.

**Rule**: compare latencies only within one paper's table, and only when the paper states that both rows use the same hardware and the same pipeline boundary.

### 5. Averages hide worst cases

- [[sources/adaptive-wam.md]]'s routed exit averages 170 ms but costs **284 ms at worst**, above its 190 ms fixed baseline.
- [[sources/drive-hwm.md]]'s peak step is 107.2 ms against an 84.8 ms mean.

A real-time budget has to be sized to the peak, and almost no paper reports it.

### 6. Per-candidate rollouts are linear in candidates and steep in tokens {#per-candidate}

[[sources/ad-e2e-jepa.md]] is the only paper here that reports the cost of rolling a world model out once per candidate trajectory at several candidate counts and two state sizes ([[sources/mm-future.md]], below, reports three candidate counts at one state size):

| State per frame | Candidates | Rollout | Time per scene (A100) |
|---|---:|---|---:|
| 512 tokens × 1024 (dense DINOv3; DINO-WM / JEPA-WM) | 256 | 8 steps, 4 s | 91.8 / 101.0 s |
| **32 tokens × 256** (projected) | 256 | 8 steps, 4 s | **0.8 s** |
| 32 tokens × 256 | 1,024 | 8 steps, 4 s | 2.5 s |
| 32 tokens × 256 | 8,192 | 8 steps, 4 s | 18.2 s |

- **Token count is the lever.** 16× fewer tokens and 4× narrower embeddings give 115–126× at a fixed candidate count. Attention cost is quadratic in tokens, so most of this is the 16×.
- **Candidates are linear**: about 2.2 ms each at scale, plus a fixed cost near 0.2 s.
- **The quoted speedup and the quoted best score are different rows.** "100×" is at 256 candidates; the 72.9 headline is at 8,192, where the advantage over the dense baseline at 256 is 5×. This is the same bundling as lesson 2.
- **0.8 s is still not a control-rate number.** It is slower than all but the video-generating policies in the table above and produces one 4 s plan. What it buys is evaluation: all 12,146 navtest scenes in 2.7 hours against roughly two weeks for the dense baselines.
- **For scale**: [[sources/dreameraD.md]] reports 0.03 s per frame for its one-step latent world model and also scores 256 vocabulary candidates. [[sources/da-wam.md]] predicts 32 per-candidate futures at a 0.5 s horizon and reports no latency.
- **A joint world–action model at the same price per candidate.** [[sources/mm-future.md]] generates each candidate's trajectory and its 4 s future together, as 256 learned tokens, in two Euler steps: 79 ms at 16 hypotheses, 132 ms at 32 and 233 ms at 64 on an H800, or about **3.2 ms per extra hypothesis**. The action-only version costs about 0.9 ms per hypothesis. Its future-aware scorer adds nothing measurable. This is the first per-candidate-future planner here with a latency inside one replanning interval.

---

## Open Questions

*Wiki-wide open questions are collected in [[research-directions.md]].*

- **What does GRAVA-style grounded reasoning cost?** It emits box tokens, metric states and per-object chains before a primitive action (4,096-token limit, 8B), and no latency is reported. It is the method class where the answer matters most.
- **Is the navhard gain from future access worth the extra latency?** SUV reports the with-access cost only. Its no-access variant (90.7 / 32.8) needs no video denoising and is presumably 2–3× cheaper.
- **Could a common harness be built?** One GPU, batch 1, pipeline from image to trajectory, reporting mean and p99. Only the Wan2.2-5B family is close to comparable today (Metis and SUV both use an RTX 4090 and report per-step sweeps).

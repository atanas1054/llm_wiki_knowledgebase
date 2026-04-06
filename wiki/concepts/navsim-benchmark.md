---
title: NAVSIM Benchmark
type: concept
sources: [raw/papers/ReCogDrive_ A Reinforced Cognitive Framework for End-to-End Autonomous Driving.md, raw/papers/WAM-Flow_ Parallel Coarse-to-Fine Motion Planning via Discrete Flow Matching for Autonomous Driving.md, raw/papers/Senna-2_ Aligning VLM and End-to-End Driving Policy for Consistent Decision Making and Planning.md, raw/papers/Discrete Diffusion for Reflective Vision-Language-Action Models in Autonomous Driving.md, raw/papers/Reasoning-VLA_ A Fast and General Vision-Language-Action Reasoning Model for Autonomous Driving.md, raw/papers/Percept-WAM_ Perception-Enhanced World-Awareness-Action Model for Robust End-to-End Autonomous Driving.md, raw/papers/DriveFine_ Refining-Augmented Masked Diffusion VLA for Precise and Robust Driving.md, raw/papers/Devil is in Narrow Policy_ Unleashing Exploration in Driving VLA Models.md, raw/papers/AutoVLA_ A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning.md, raw/papers/AdaThinkDrive_ Adaptive Thinking via Reinforcement Learning for Autonomous Driving.md]
related: [sources/recogdrive.md, sources/wam-flow.md, sources/senna2.md, sources/reflectdrive.md, sources/reasoning-vla.md, sources/percept-wam.md, sources/drivefine.md, sources/curious-vla.md, sources/autovla.md, sources/adathinkdrive.md, concepts/rl-for-ad.md, concepts/discrete-flow-matching.md, concepts/dual-system-vla.md, concepts/inference-time-safety.md, concepts/perception-for-planning.md]
created: 2026-04-05
updated: 2026-04-05
confidence: high
---

## What It Is

NAVSIM is a planning-oriented autonomous driving benchmark built on OpenScene (a redistribution of nuPlan). It provides a **non-reactive simulation** environment for closed-loop planning evaluation.

## Dataset

- Built on **OpenScene** / **nuPlan**
- **Sensors**: 8 cameras (1920×1080) + fused LiDAR point cloud from 5 sensors across current + 3 prior frames
- **Splits**: navtrain (1,192 training scenes), navtest (136 evaluation scenes)
- 85,109 trajectory-based QA pairs in navtrain

## Primary Metric: PDMS

**Predictive Driver Model Score**:
$$\text{PDMS} = NC \times DAC \times \frac{5 \times EP + 5 \times TTC + 2 \times C}{12}$$

Sub-metrics:
| Metric | Meaning |
|--------|---------|
| NC | No At-Fault Collision |
| DAC | Drivable Area Compliance |
| EP | Ego Progress |
| TTC | Time-to-Collision |
| C (Comf.) | Comfort |

NC and DAC are multiplicative gates — a collision zeroes the score.

## Extended Metrics (Hydra-MDP++)

- **TL**: Traffic Light Compliance
- **LK**: Lane Keeping Ability
- **DDC**: Driving Direction Compliance
- **EC**: Extended Comfort
- **EPDMS**: Extended PDMS incorporating the above

## Non-Reactive Simulator

NAVSIM's simulator does not model other agents reacting to the ego vehicle. This simplifies evaluation (and RL training) but means scores may not reflect performance in fully interactive scenarios.

## SOTA Progression

| Method | Modality | PDMS | Notes |
|--------|----------|------|-------|
| PARA-Drive | Camera | 84.0 | |
| ARTEMIS | Camera+LiDAR | 87.0 | |
| DiffusionDrive | Camera+LiDAR | 88.1 | |
| WoTE | Camera+LiDAR | 88.3 | |
| AutoVLA | 3 Cam | 89.1 | AR + GRPO |
| ReCogDrive | 3 Cam | 89.6 | AR + Diffusion + GRPO |
| ReflectDrive | 3 Cam | >89.1 (claimed) | Masked diffusion + reflective inference; exact number not in markdown |
| WAM-Flow (5-step) | 1 Cam | 90.3 | DFM + GRPO |
| Percept-WAM\* | Cam+LiDAR | 90.2 | World-PV/BEV tokens + SFT; no RL; comparison vs. DiffusionDrive only |
| **Reasoning-VLA-7B** | **3 Cam** | **91.7 (claimed)** | **Learnable queries + GT-based GRPO; comparison only vs. old baselines** |
| DriveFine | 1 Cam | 90.7 | Masked diffusion + block-MoE refinement + GRPO; head-to-head with ReCogDrive/AdaThinkDrive |
| **DriveFine\*** | **1 Cam** | **91.8** | **Score-based RFT (extra trained scorer)** |
| Curious-VLA | 1 Cam | 90.3 | FTE + ADAS + SDR; Qwen2.5-VL-3B; text waypoint |
| **Curious-VLA† (N=6)** | **1 Cam** | **94.8** | **Best-of-6; matches human GT (94.8)** |
| AutoVLA (post-RFT) | 3 Cam | 89.11 | Physical action codebook + GRPO with CoT length penalty |
| AutoVLA (Best-of-N) | 3 Cam | 92.12 | Oracle-selected from 6 candidates |
| AdaThinkDrive | 1 Cam | 90.3 | Adaptive Think Reward GRPO; vision-only; InternVL3-8B |
| **AdaThinkDrive (BoN-4)** | **1 Cam** | **93.0** | **Best-of-4; vision-only** |

**Caveat on Reasoning-VLA (91.7)**: Table 6 in the paper compares only against TransFuser/UniAD/Para-Drive (~84 PDMS). No head-to-head with WAM-Flow (90.3) or Percept-WAM (90.2). Unverified. See [[sources/reasoning-vla.md]].

**Caveat on Percept-WAM (90.2)**: uses LiDAR for BEV token initialization; compared against DiffusionDrive (88.1), Hydra-MDP (86.5), DRAMA (85.5) — no head-to-head with WAM-Flow (90.3) or ReCogDrive (89.6). See [[sources/percept-wam.md]].

**Caveat on DriveFine\* (91.8)**: uses score-based RFT with an additional trained scorer beyond the NAVSIM simulator. DriveFine base without the extra scorer achieves 90.7 PDMS. See [[sources/drivefine.md]].

**Caveat on Curious-VLA (90.3)**: also compares head-to-head with AdaThinkDrive (90.3) and DriveVLA-W0 (90.2) from the same table — these comparisons are internally consistent. The paper's NAVSIM-v1 table does not include DriveFine (90.7) — so DriveFine remains the single-sample SOTA. See [[sources/curious-vla.md]].

**Caveat on AdaThinkDrive (90.3)**: Table I does not include WAM-Flow (90.3, 1 cam) or Curious-VLA (90.3, 1 cam) — three papers independently claim 90.3 PDMS on NAVSIM-v1 with no direct head-to-head. Paper claims "+1.7 vs. best vision-only baseline" referencing Hydra-NeXt (88.6, a non-VLM method); VLM baselines at 90.3 exist in the wiki but are not compared. AdaThinkDrive BoN-4 (93.0) exceeds AutoVLA BoN-6 (92.12) and is second only to Curious-VLA BoN-6 (94.8). No NAVSIM-v2 / EPDMS reported. See [[sources/adathinkdrive.md]].

DriveFine (90.7, 1xC) remains the broadly-verified **single-sample single-camera NAVSIM-v1 SOTA**. Curious-VLA (90.3) is competitive but slightly below DriveFine at single sample; Curious-VLA† (94.8 BoN-6) is the **best-of-N result**, matching human GT.

**Caveat on AutoVLA (89.11)**: Table 1 compares against Hydra-MDP (91.26), Centaur (92.10), TrajHF (93.95) — older non-VLM baselines — and does not include DriveFine, WAM-Flow, or Curious-VLA. AutoVLA post-RFT (89.11) is below the current single-sample SOTA (DriveFine 90.7). The 89.11 figure cross-validates with the "AutoVLA 89.1" entry in Curious-VLA's comparison table. See [[sources/autovla.md]].

## NAVSIM-v2 SOTA (EPDMS)

| Method         | EPDMS    | Notable weakness                                               |
| -------------- | -------- | -------------------------------------------------------------- |
| Artemis        | 83.1     | —                                                              |
| RecogDrive     | 83.6     | —                                                              |
| WAM-Flow       | 84.7     | EC = 73.9 (low vs. RecogDrive's 87.7)                          |
| DiffusionDrive | 84.5     | —                                                              |
| ResAD          | 85.5     | —                                                              |
| **Senna-2**    | **86.6** | Requires separate NAVSIM fine-tune                             |
| Curious-VLA    | 85.3     | FTE + ADAS + SDR; EC = 81.5 (comfort regression vs. baselines) |

Senna-2 ([[sources/senna2.md]]) held the NAVSIM-v2 SOTA at 86.6 EPDMS but was surpassed by DriveFine. However, DriveFine's 89.7 EPDMS uses a **bug-fixed NAVSIM** scorer that systematically changes scores — not directly comparable to prior methods evaluated on the bugged version. DriveFine also achieves 87.1 EPDMS on the bugged version, outperforming Senna-2 (86.6) under comparable conditions.

**Curious-VLA NAVSIM-v2 note**: The paper's v2 table compares against Ego-MLP, VADv2, TransFuser, HydraMDP++, DriveSuprim, ARTEMIS, ReCogDrive, and DiffusionDrive — but not Senna-2 (86.6) or DriveFine (87.1 on old scorer). Curious-VLA's 85.3 EPDMS is not directly comparable to those higher results. Notable weakness: EC = 81.5 vs. DiffusionDrive's 87.7 — exploration creates more aggressive maneuvers. See [[sources/curious-vla.md]].

## Navhard Benchmark

An out-of-distribution evaluation built on top of NAVSIM. Uses **3DGS (Gaussian splatting)** to synthesize novel scenarios beyond the training distribution. Two-stage evaluation:

| Method | Stage 1 EPDMS | Stage 2 EPDMS |
|--------|-------------|-------------|
| ReCogDrive | 68.9 | 37.8 |
| DiffusionDrive | 66.7 | 40.5 |
| **DriveFine** | **74.4** | **41.0** |

DriveFine leads by +5.5 on Stage 1. Stage 2 (harder OOD) remains a challenge for all methods. See [[sources/drivefine.md]].

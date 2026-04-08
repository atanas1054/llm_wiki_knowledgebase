---
title: NAVSIM Benchmark
type: concept
sources: [raw/papers/ReCogDrive_ A Reinforced Cognitive Framework for End-to-End Autonomous Driving.md, raw/papers/WAM-Flow_ Parallel Coarse-to-Fine Motion Planning via Discrete Flow Matching for Autonomous Driving.md, raw/papers/Senna-2_ Aligning VLM and End-to-End Driving Policy for Consistent Decision Making and Planning.md, raw/papers/Discrete Diffusion for Reflective Vision-Language-Action Models in Autonomous Driving.md, raw/papers/Reasoning-VLA_ A Fast and General Vision-Language-Action Reasoning Model for Autonomous Driving.md, raw/papers/Percept-WAM_ Perception-Enhanced World-Awareness-Action Model for Robust End-to-End Autonomous Driving.md, raw/papers/DriveFine_ Refining-Augmented Masked Diffusion VLA for Precise and Robust Driving.md, raw/papers/Devil is in Narrow Policy_ Unleashing Exploration in Driving VLA Models.md, raw/papers/AutoVLA_ A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning.md, raw/papers/AdaThinkDrive_ Adaptive Thinking via Reinforcement Learning for Autonomous Driving.md, raw/papers/FutureSightDrive_ Thinking Visually with Spatio-Temporal CoT for Autonomous Driving.md, raw/papers/DriveDreamer-Policy_ A Geometry-Grounded World–Action Model for Unified Generation and Planning.md, raw/papers/DriveVLA-W0_ World Models Amplify Data Scaling Law in Autonomous Driving.md, raw/papers/FLARE_ Learning Future-Aware Latent Representations from Vision-Language Models for Autonomous Driving.md]
related: [sources/recogdrive.md, sources/wam-flow.md, sources/senna2.md, sources/reflectdrive.md, sources/reasoning-vla.md, sources/percept-wam.md, sources/drivefine.md, sources/curious-vla.md, sources/autovla.md, sources/adathinkdrive.md, sources/futuresightdrive.md, sources/drivedreamer-policy.md, sources/drivevla-w0.md, sources/flare.md, concepts/rl-for-ad.md, concepts/discrete-flow-matching.md, concepts/dual-system-vla.md, concepts/inference-time-safety.md, concepts/perception-for-planning.md]
created: 2026-04-05
updated: 2026-04-07
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

| Method                    | Modality     | PDMS               | Notes                                                                                                     |
| ------------------------- | ------------ | ------------------ | --------------------------------------------------------------------------------------------------------- |
| PARA-Drive                | Camera       | 84.0               |                                                                                                           |
| FSDrive                   | Camera       | 85.1               | VQ-VAE AR world model + visual ST-CoT; Qwen2-VL-2B; compared vs. pre-2025 baselines only                  |
| DriveDreamer-Policy       | 3 Cam        | 89.2               | Depth+video+action WAM; Qwen3-VL-2B; compared vs. world-model methods only (excludes DriveFine, WAM-Flow) |
| DriveVLA-W0★              | 1 Cam        | 90.2               | AR world model + query-based expert + trajectory anchors (not single-sample); Emu3-8B                     |
| **DriveVLA-W0† (BoN-6)**  | **1 Cam**    | **93.0**           | **AR world model + AR expert + best-of-6; 1 cam**                                                         |
| FLARE-4B (SFT)            | 1 Cam        | 86.9               | DINOv2 feature prediction + DiT + LoRA; no external pretraining; best VLM-based SFT                       |
| FLARE-4B (RFT)            | 1 Cam        | 91.4               | +GRPO with BC regularization; single-sample; best single-sample VLM-based RFT                             |
| ARTEMIS                   | Camera+LiDAR | 87.0               |                                                                                                           |
| DiffusionDrive            | Camera+LiDAR | 88.1               |                                                                                                           |
| WoTE                      | Camera+LiDAR | 88.3               |                                                                                                           |
| AutoVLA                   | 3 Cam        | 89.1               | AR + GRPO                                                                                                 |
| ReCogDrive                | 3 Cam        | 89.6               | AR + Diffusion + GRPO                                                                                     |
| ReflectDrive              | 3 Cam        | >89.1 (claimed)    | Masked diffusion + reflective inference; exact number not in markdown                                     |
| WAM-Flow (5-step)         | 1 Cam        | 90.3               | DFM + GRPO                                                                                                |
| Percept-WAM\*             | Cam+LiDAR    | 90.2               | World-PV/BEV tokens + SFT; no RL; comparison vs. DiffusionDrive only                                      |
| **Reasoning-VLA-7B**      | **3 Cam**    | **91.7 (claimed)** | **Learnable queries + GT-based GRPO; comparison only vs. old baselines**                                  |
| DriveFine                 | 1 Cam        | 90.7               | Masked diffusion + block-MoE refinement + GRPO; head-to-head with ReCogDrive/AdaThinkDrive                |
| **DriveFine\***           | **1 Cam**    | **91.8**           | **Score-based RFT (extra trained scorer)**                                                                |
| Curious-VLA               | 1 Cam        | 90.3               | FTE + ADAS + SDR; Qwen2.5-VL-3B; text waypoint                                                            |
| **Curious-VLA† (N=6)**    | **1 Cam**    | **94.8**           | **Best-of-6; matches human GT (94.8)**                                                                    |
| AutoVLA (post-RFT)        | 3 Cam        | 89.11              | Physical action codebook + GRPO with CoT length penalty                                                   |
| AutoVLA (Best-of-N)       | 3 Cam        | 92.12              | Oracle-selected from 6 candidates                                                                         |
| AdaThinkDrive             | 1 Cam        | 90.3               | Adaptive Think Reward GRPO; vision-only; InternVL3-8B                                                     |
| **AdaThinkDrive (BoN-4)** | **1 Cam**    | **93.0**           | **Best-of-4; vision-only**                                                                                |

**Caveat on Reasoning-VLA (91.7)**: Table 6 in the paper compares only against TransFuser/UniAD/Para-Drive (~84 PDMS). No head-to-head with WAM-Flow (90.3) or Percept-WAM (90.2). Unverified. See [[sources/reasoning-vla.md]].

**Caveat on Percept-WAM (90.2)**: uses LiDAR for BEV token initialization; compared against DiffusionDrive (88.1), Hydra-MDP (86.5), DRAMA (85.5) — no head-to-head with WAM-Flow (90.3) or ReCogDrive (89.6). See [[sources/percept-wam.md]].

**Caveat on DriveFine\* (91.8)**: uses score-based RFT with an additional trained scorer beyond the NAVSIM simulator. DriveFine base without the extra scorer achieves 90.7 PDMS. See [[sources/drivefine.md]].

**Caveat on Curious-VLA (90.3)**: also compares head-to-head with AdaThinkDrive (90.3) and DriveVLA-W0 (90.2) from the same table — these comparisons are internally consistent. The paper's NAVSIM-v1 table does not include DriveFine (90.7) — so DriveFine remains the single-sample SOTA. See [[sources/curious-vla.md]].

**Caveat on AdaThinkDrive (90.3)**: Table I does not include WAM-Flow (90.3, 1 cam) or Curious-VLA (90.3, 1 cam) — three papers independently claim 90.3 PDMS on NAVSIM-v1 with no direct head-to-head. Paper claims "+1.7 vs. best vision-only baseline" referencing Hydra-NeXt (88.6, a non-VLM method); VLM baselines at 90.3 exist in the wiki but are not compared. AdaThinkDrive BoN-4 (93.0) exceeds AutoVLA BoN-6 (92.12) and is second only to Curious-VLA BoN-6 (94.8). No NAVSIM-v2 / EPDMS reported. See [[sources/adathinkdrive.md]].

**Caveat on FSDrive (85.1)**: Table 2 compares only against PARA-Drive (84.0), LAW (84.6), and pre-2025 baselines. No head-to-head with ReCogDrive (89.6), WAM-Flow (90.3), or DriveFine (90.7). 85.1 PDMS is below the wiki median for camera-only VLMs. See [[sources/futuresightdrive.md]].

**Caveat on DriveVLA-W0 (90.2★)**: uses query-based expert with trajectory anchors (multi-candidate Hydra-MDP style selection) — not single-sample. The underlying single-sample query-based expert achieves 88.4 PDMS. BoN-6 (93.0) is second only to Curious-VLA BoN-6 (94.8) in the wiki. EC = 58.9 on NAVSIM-v2 (lowest in wiki). World model bypassed at inference. See [[sources/drivevla-w0.md]].

**Caveat on DriveDreamer-Policy (89.2)**: Table 1 compares within "world-model-based" category vs. LAW, DrivingGPT, WoTE, Epona, FSDrive, PWM, plus VLA methods AutoVLA (89.1), DriveVLA-W0 (88.4), and ReCogDrive *IL-only* (86.5). The RL-trained ReCogDrive (89.6) is excluded, as are DriveFine (90.7), WAM-Flow (90.3), AdaThinkDrive (90.3), Curious-VLA (90.3). 89.2 PDMS is not SOTA in the full NAVSIM-v1 field. See [[sources/drivedreamer-policy.md]].

**Caveat on FLARE-4B RFT (91.4)**: compares against ReCogDrive-2B/8B RFT and DriveVLA-W0 single-sample (90.2 anchor-based), beating all under single-sample VLM-based evaluation. AutoVLA BoN-6 (92.1) and DriveVLA-W0 BoN-6 (93.0) use a different inference strategy and are not directly comparable. No head-to-head with DriveFine (90.7) or WAM-Flow (90.3) in the paper's table. See [[sources/flare.md]].

FLARE-4B RFT (91.4) is now the **highest single-sample VLM-based NAVSIM-v1 result** in the wiki among the papers that include fair single-sample comparisons. DriveFine (90.7, 1xC) remains the most broadly-verified single-sample result with explicit head-to-head against contemporary methods. Curious-VLA† (94.8 BoN-6) is the **best-of-N result**, matching human GT.

**Caveat on AutoVLA (89.11)**: Table 1 compares against Hydra-MDP (91.26), Centaur (92.10), TrajHF (93.95) — older non-VLM baselines — and does not include DriveFine, WAM-Flow, or Curious-VLA. AutoVLA post-RFT (89.11) is below the current single-sample SOTA (DriveFine 90.7). The 89.11 figure cross-validates with the "AutoVLA 89.1" entry in Curious-VLA's comparison table. See [[sources/autovla.md]].

## NAVSIM-v2 SOTA (EPDMS)

| Method                   | EPDMS    | EC   | Notable weakness / caveat                                      |
| ------------------------ | -------- | ---- | -------------------------------------------------------------- |
| Artemis                  | 83.1     | 89.1 | —                                                              |
| RecogDrive               | 83.6     | 87.7 | —                                                              |
| DiffusionDrive           | 84.5     | 87.7 | —                                                              |
| WAM-Flow                 | 84.7     | 73.9 | Low EC (comfort); high safety metrics                          |
| ResAD                    | 85.5     | —    | —                                                              |
| Curious-VLA              | 85.3     | 81.5 | EC regression from exploration; thin v2 comparison set        |
| Senna-2                  | 86.6     | —    | Requires separate NAVSIM fine-tune                             |
| DriveVLA-W0              | 86.1     | 58.9 | Lowest EC in wiki; world model training-time only              |
| **DriveDreamer-Policy**  | **88.7** | 79.4 | Thin comparison set (only DriveVLA-W0 as VLA baseline); low EC |
| FLARE-4B                 | 86.3     | 87.5 | Omits Senna-2 (86.6) and DDP (88.7) from comparison; best EC among VLA wiki entries |

DriveDreamer-Policy ([[sources/drivedreamer-policy.md]]) surpasses Senna-2 (86.6) by +2.1 EPDMS, remaining the wiki NAVSIM-v2 SOTA at 88.7. FLARE (86.3) claims SOTA but excludes Senna-2 and DDP from its comparison table — the 86.3 result falls below both. **Caveat**: Table 2 only compares against DriveVLA-W0 (86.1) as a VLA baseline; Senna-2, DriveFine, and Curious-VLA are not included — the 88.7 EPDMS result is likely genuine but lacks direct head-to-head verification. EC = 79.4 — geometry/video world learning improves safety-oriented metrics but not extended comfort.

DriveFine's 89.7 EPDMS uses a **bug-fixed NAVSIM** scorer — not directly comparable to prior methods evaluated on the bugged version. DriveFine also achieves 87.1 EPDMS on the bugged version, outperforming Senna-2 (86.6) and DriveDreamer-Policy (88.7 on standard scorer) under comparable conditions — the relationship between DriveFine and DDP on a common scorer is unclear.

**Curious-VLA NAVSIM-v2 note**: compares against Ego-MLP, VADv2, TransFuser, HydraMDP++, DriveSuprim, ARTEMIS, ReCogDrive, and DiffusionDrive — not Senna-2 (86.6) or DriveDreamer-Policy (88.7). EC = 81.5 vs. DiffusionDrive's 87.7 — exploration creates more aggressive maneuvers. See [[sources/curious-vla.md]].

## Navhard Benchmark

An out-of-distribution evaluation built on top of NAVSIM. Uses **3DGS (Gaussian splatting)** to synthesize novel scenarios beyond the training distribution. Two-stage evaluation:

| Method | Stage 1 EPDMS | Stage 2 EPDMS |
|--------|-------------|-------------|
| ReCogDrive | 68.9 | 37.8 |
| DiffusionDrive | 66.7 | 40.5 |
| **DriveFine** | **74.4** | **41.0** |

DriveFine leads by +5.5 on Stage 1. Stage 2 (harder OOD) remains a challenge for all methods. See [[sources/drivefine.md]].

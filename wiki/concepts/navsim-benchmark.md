---
title: NAVSIM Benchmark
type: concept
sources: [raw/papers/ReCogDrive_ A Reinforced Cognitive Framework for End-to-End Autonomous Driving.md, raw/papers/WAM-Flow_ Parallel Coarse-to-Fine Motion Planning via Discrete Flow Matching for Autonomous Driving.md, raw/papers/Senna-2_ Aligning VLM and End-to-End Driving Policy for Consistent Decision Making and Planning.md, raw/papers/Discrete Diffusion for Reflective Vision-Language-Action Models in Autonomous Driving.md, raw/papers/Reasoning-VLA_ A Fast and General Vision-Language-Action Reasoning Model for Autonomous Driving.md, raw/papers/Percept-WAM_ Perception-Enhanced World-Awareness-Action Model for Robust End-to-End Autonomous Driving.md, raw/papers/DriveFine_ Refining-Augmented Masked Diffusion VLA for Precise and Robust Driving.md, raw/papers/Devil is in Narrow Policy_ Unleashing Exploration in Driving VLA Models.md, raw/papers/AutoVLA_ A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning.md, raw/papers/AdaThinkDrive_ Adaptive Thinking via Reinforcement Learning for Autonomous Driving.md, raw/papers/FutureSightDrive_ Thinking Visually with Spatio-Temporal CoT for Autonomous Driving.md, raw/papers/DriveDreamer-Policy_ A Geometry-Grounded World–Action Model for Unified Generation and Planning.md, raw/papers/DriveVLA-W0_ World Models Amplify Data Scaling Law in Autonomous Driving.md, raw/papers/FLARE_ Learning Future-Aware Latent Representations from Vision-Language Models for Autonomous Driving.md, raw/papers/DreamerAD_ Efficient Reinforcement Learning via Latent World Model for Autonomous Driving.md, raw/papers/Vega_ Learning to Drive with Natural Language Instructions.md, raw/papers/NoRD_ A Data-Efficient Vision-Language-Action Model that Drives without Reasoning.md, raw/papers/DiffusionDrive_ Truncated Diffusion Model for End-to-End Autonomous Driving.md, raw/papers/DiffusionDriveV2_ Reinforcement Learning-Constrained Truncated Diffusion Modeling in End-to-End Autonomous Driving.md, raw/papers/Epona_ Autoregressive Diffusion World Model for Autonomous Driving.md, raw/papers/From Representational Complementarity to Dual Systems_ Synergizing VLM and Vision-Only Backbones for End-to-End Driving.md, raw/papers/WAM-Diff_ A Masked Diffusion VLA Framework with MoE and Online Reinforcement Learning for Autonomous Driving.md, raw/papers/DriveSuprim_ Towards Precise Trajectory Selection for End-to-End Planning.md, raw/papers/DriveVA_ Video Action Models are Zero-Shot Drivers.md, raw/papers/ExploreVLA_ Dense World Modeling and Exploration for End-to-End Autonomous Driving.md]
related: [sources/recogdrive.md, sources/wam-flow.md, sources/senna2.md, sources/reflectdrive.md, sources/reasoning-vla.md, sources/percept-wam.md, sources/drivefine.md, sources/curious-vla.md, sources/autovla.md, sources/adathinkdrive.md, sources/futuresightdrive.md, sources/drivedreamer-policy.md, sources/drivevla-w0.md, sources/flare.md, sources/dreameraD.md, sources/vega.md, sources/nord.md, sources/diffusiondrive.md, sources/diffusiondrive-v2.md, sources/epona.md, sources/hybriddriveVLA.md, sources/wam-diff.md, sources/drivesuprim.md, sources/driveva.md, sources/explorevla.md, concepts/rl-for-ad.md, concepts/discrete-flow-matching.md, concepts/dual-system-vla.md, concepts/inference-time-safety.md, concepts/perception-for-planning.md, concepts/best-of-n.md, concepts/selection-based-planning.md, concepts/world-model-for-ad.md]
created: 2026-04-05
updated: 2026-04-23
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

| Method                    | Modality                      | PDMS               | Notes                                                                                                                                                                |
| ------------------------- | ----------------------------- | ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PARA-Drive                | Camera                        | 84.0               |                                                                                                                                                                      |
| FSDrive                   | Camera                        | 85.1               | VQ-VAE AR world model + visual ST-CoT; Qwen2-VL-2B; compared vs. pre-2025 baselines only                                                                             |
| DriveDreamer-Policy       | 3 Cam                         | 89.2               | Depth+video+action WAM; Qwen3-VL-2B; compared vs. world-model methods only (excludes DriveFine, WAM-Flow)                                                            |
| DriveVLA-W0★              | 1 Cam                         | 90.2               | AR world model + query-based expert + trajectory anchors (not single-sample); Emu3-8B                                                                                |
| **DriveVLA-W0† (BoN-6)**  | **1 Cam**                     | **93.0**           | **AR world model + AR expert + best-of-6; 1 cam**                                                                                                                    |
| FLARE-4B (SFT)            | 1 Cam                         | 86.9               | DINOv2 feature prediction + DiT + LoRA; no external pretraining; best VLM-based SFT                                                                                  |
| FLARE-4B (RFT)            | 1 Cam                         | 91.4               | +GRPO with BC regularization; single-sample; best single-sample VLM-based RFT                                                                                        |
| DreamerAD                 | C-Only                        | 88.7               | Latent world model RL (SF-WM + AD-RM + vocab sampling); Epona backbone; best world-model-class RL                                                                    |
| Vega                      | 1 Cam                         | 87.9               | Instruction-conditioned AR+diffusion; InstructScene 100K; no RL; 1 camera                                                                                            |
| **Vega† (BoN-6)**         | **1 Cam**                     | **89.8**           | **Best-of-6; instruction-conditioned**                                                                                                                               |
| Epona                     | Camera                        | 86.2               | AR+Diffusion WM; 2.5B; front cam, no aux supervision; comparison vs. pre-2025 baselines only; backbone for DreamerAD                                                 |
| ARTEMIS                   | Camera+LiDAR                  | 87.0               |                                                                                                                                                                      |
| DiffusionDrive            | Camera+LiDAR                  | 88.1               | Truncated diffusion (20 anchors, 2 steps); ResNet-34; 45 FPS; canonical non-VLM diffusion baseline                                                                   |
| **DiffusionDriveV2**      | **Camera+LiDAR**              | **91.2**           | **DiffusionDrive + Intra/Inter-Anchor GRPO + multiplicative noise; ResNet-34; highest non-VLM result; beats V2-99 methods**                                          |
| WoTE                      | Camera+LiDAR                  | 88.3               |                                                                                                                                                                      |
| NoRD                      | 3 Cam                         | 85.6               | No reasoning, no LiDAR; 80K samples; Dr. GRPO; data-efficiency frontier                                                                                              |
| **NoRD-BoN* (6)**         | **3 Cam**                     | **92.4**           | **Best-of-6 oracle; no reasoning; surpasses AutoVLA-BoN (92.1)**                                                                                                     |
| AutoVLA                   | 3 Cam                         | 89.1               | AR + GRPO                                                                                                                                                            |
| ReCogDrive                | 3 Cam                         | 89.6†              | AR + Diffusion + GRPO; †DreamerAD's table cites 90.8 (NeurIPS camera-ready); earlier arXiv = 89.6                                                                    |
| ReflectDrive              | 3 Cam                         | >89.1 (claimed)    | Masked diffusion + reflective inference; exact number not in markdown                                                                                                |
| WAM-Flow (5-step)         | 1 Cam                         | 90.3               | DFM + GRPO                                                                                                                                                           |
| Percept-WAM\*             | Cam+LiDAR                     | 90.2               | World-PV/BEV tokens + SFT; no RL; comparison vs. DiffusionDrive only                                                                                                 |
| **Reasoning-VLA-7B**      | **3 Cam**                     | **91.7 (claimed)** | **Learnable queries + GT-based GRPO; comparison only vs. old baselines**                                                                                             |
| DriveFine                 | 1 Cam                         | 90.7               | Masked diffusion + block-MoE refinement + GRPO; head-to-head with ReCogDrive/AdaThinkDrive                                                                           |
| **DriveFine\***           | **1 Cam**                     | **91.8**           | **Score-based RFT (extra trained scorer)**                                                                                                                           |
| Curious-VLA               | 1 Cam                         | 90.3               | FTE + ADAS + SDR; Qwen2.5-VL-3B; text waypoint                                                                                                                       |
| **Curious-VLA† (N=6)**    | **1 Cam**                     | **94.8**           | **Best-of-6; matches human GT (94.8)**                                                                                                                               |
| AutoVLA (post-RFT)        | 3 Cam                         | 89.11              | Physical action codebook + GRPO with CoT length penalty                                                                                                              |
| AutoVLA (Best-of-N)       | 3 Cam                         | 92.12              | Oracle-selected from 6 candidates                                                                                                                                    |
| AdaThinkDrive             | 1 Cam                         | 90.3               | Adaptive Think Reward GRPO; vision-only; InternVL3-8B                                                                                                                |
| **AdaThinkDrive (BoN-4)** | **1 Cam**                     | **93.0**           | **Best-of-4; vision-only**                                                                                                                                           |
| **HybridDriveVLA**        | **Cam (VLM+ViT dual-branch)** | **92.1**           | **VLM+ViT branches + interpolation + trajectory scorer; directly compares vs. DiffusionDriveV2 (91.2) and iPad (91.7); runs both branches simultaneously**           |
| DualDriveVLA              | Cam (VLM+ViT fast–slow)       | 91.0               | HybridDriveVLA with 15% VLM invocation; 3.2× throughput vs. VLM-only; ablation result only                                                                           |
| WAM-Diff                  | 1 Cam                         | 91.0               | Masked diffusion + LoRA MoE (64 experts) + GSPO (G=3); LLaDA-V 8.4B; reverse-causal decoding; compares vs. DiffusionDrive/ReCogDrive/DriveVLA-W0                     |
| DriveVA                   | Cam (5B video backbone)       | 90.9               | Wan2.2-TI2V-5B backbone; joint DiT video+action; zero-shot nuScenes −78.9% L2 / −83.3% CR vs. PWM; Table 1 sub-scores truncated in source; no RL; U. Twente |
| **DriveSuprim**           | **Cam (ViT-L, selection)**    | **93.5**           | **Non-VLM selection-based; 8192-trajectory vocab + coarse-to-fine (→256) + rotation aug + EMA self-distill; highest non-BoN result in wiki; no LiDAR; Fudan+NVIDIA** |
| ExploreVLA                | 1 Cam                         | 90.4               | Show-o (Phi-1.5 + MAGVIT-v2); dense RGB+depth world model SFT; safety-gated entropy exploration reward; GRPO LoRA; 1 cam; NAVSIM v1 Table 1 omits WAM-Diff/FLARE/DiffusionDriveV2/HybridDriveVLA |
| **ExploreVLA† (BoN-6)**   | **1 Cam**                     | **93.7**           | **Best-of-6; 2nd highest BoN in wiki after Curious-VLA (94.8); BoN vs. DriveSuprim single-sample not a fair comparison** |

**Caveat on Reasoning-VLA (91.7)**: Table 6 in the paper compares only against TransFuser/UniAD/Para-Drive (~84 PDMS). No head-to-head with WAM-Flow (90.3) or Percept-WAM (90.2). Unverified. See [[sources/reasoning-vla.md]].

**Caveat on Percept-WAM (90.2)**: uses LiDAR for BEV token initialization; compared against DiffusionDrive (88.1), Hydra-MDP (86.5), DRAMA (85.5) — no head-to-head with WAM-Flow (90.3) or ReCogDrive (89.6). See [[sources/percept-wam.md]].

**Caveat on DriveFine\* (91.8)**: uses score-based RFT with an additional trained scorer beyond the NAVSIM simulator. DriveFine base without the extra scorer achieves 90.7 PDMS. See [[sources/drivefine.md]].

**Caveat on Curious-VLA (90.3)**: also compares head-to-head with AdaThinkDrive (90.3) and DriveVLA-W0 (90.2) from the same table — these comparisons are internally consistent. The paper's NAVSIM-v1 table does not include DriveFine (90.7) — so DriveFine remains the single-sample SOTA. See [[sources/curious-vla.md]].

**Caveat on AdaThinkDrive (90.3)**: Table I does not include WAM-Flow (90.3, 1 cam) or Curious-VLA (90.3, 1 cam) — three papers independently claim 90.3 PDMS on NAVSIM-v1 with no direct head-to-head. Paper claims "+1.7 vs. best vision-only baseline" referencing Hydra-NeXt (88.6, a non-VLM method); VLM baselines at 90.3 exist in the wiki but are not compared. AdaThinkDrive BoN-4 (93.0) exceeds AutoVLA BoN-6 (92.12) and is second only to Curious-VLA BoN-6 (94.8). No NAVSIM-v2 / EPDMS reported. See [[sources/adathinkdrive.md]].

**Caveat on FSDrive (85.1)**: Table 2 compares only against PARA-Drive (84.0), LAW (84.6), and pre-2025 baselines. No head-to-head with ReCogDrive (89.6), WAM-Flow (90.3), or DriveFine (90.7). 85.1 PDMS is below the wiki median for camera-only VLMs. See [[sources/futuresightdrive.md]].

**Caveat on DriveVLA-W0 (90.2★)**: uses query-based expert with trajectory anchors (multi-candidate Hydra-MDP style selection) — not single-sample. The underlying single-sample query-based expert achieves 88.4 PDMS. BoN-6 (93.0) is second only to Curious-VLA BoN-6 (94.8) in the wiki. EC = 58.9 on NAVSIM-v2 (lowest in wiki). World model bypassed at inference. See [[sources/drivevla-w0.md]].

**Caveat on DriveDreamer-Policy (89.2)**: Table 1 compares within "world-model-based" category vs. LAW, DrivingGPT, WoTE, Epona, FSDrive, PWM, plus VLA methods AutoVLA (89.1), DriveVLA-W0 (88.4), and ReCogDrive *IL-only* (86.5). The RL-trained ReCogDrive (89.6) is excluded, as are DriveFine (90.7), WAM-Flow (90.3), AdaThinkDrive (90.3), Curious-VLA (90.3). 89.2 PDMS is not SOTA in the full NAVSIM-v1 field. See [[sources/drivedreamer-policy.md]].

**Caveat on FLARE-4B RFT (91.4)**: compares against ReCogDrive-2B/8B RFT and DriveVLA-W0 single-sample (90.2 anchor-based), beating all under single-sample VLM-based evaluation. AutoVLA BoN-6 (92.1) and DriveVLA-W0 BoN-6 (93.0) use a different inference strategy and are not directly comparable. No head-to-head with DriveFine (90.7) or WAM-Flow (90.3) in the paper's table. See [[sources/flare.md]].

**DriveSuprim (93.5)** ([[sources/drivesuprim.md]]) is the **highest non-BoN result in the wiki** — non-VLM, selection-based, ViT-L backbone. FLARE-4B RFT (91.4) remains the **highest single-sample VLM-based result** among papers with fair comparisons. WAM-Diff (91.0) slots second among VLMs but omits FLARE from its comparison table. DriveFine (90.7, 1xC) is the most broadly-verified single-sample VLM result with direct head-to-head against contemporary methods. Approximate ordering (single-pass, non-BoN): **DriveSuprim 93.5** (selection, ViT-L) > HybridDriveVLA 92.1 (ensemble) > FLARE 91.4 (VLM) > WAM-Diff 91.0 ≈ DualDriveVLA 91.0 > DriveFine 90.7 > Curious-VLA/WAM-Flow/AdaThinkDrive 90.3. **BoN-6 ranking**: Curious-VLA† (94.8 BoN-6) = human GT; **ExploreVLA† (93.7 BoN-6)** is 2nd; AdaThinkDrive BoN-4 / DriveVLA-W0 BoN-6 (both 93.0) are 3rd.

**Caveat on Vega (86.9 / 89.4 BoN-6)**: Table 1 (NAVSIM-v2) compares against DriveVLA-W0 (86.1) as the only VLA baseline; DriveDreamer-Policy (88.7), DreamerAD (87.7), Senna-2 (86.6), and FLARE (86.3) are not included. The BoN-6 result (89.4) likely exceeds DDP (88.7) in absolute terms but cannot be confirmed without head-to-head comparison. Table 2 (NAVSIM-v1) compares against AutoVLA (89.1) and ReCogDrive (89.6) — Vega single (87.9) is below both. EC = 76.3 (single) / 84.5 (BoN) — no extended comfort regression from instruction following, but not best-in-class. No RL stage. See [[sources/vega.md]].

**Caveat on DreamerAD (88.7)**: Table 2 compares against WorldRFT (87.8), Epona (86.2), World4Drive (85.1), and VLA methods AutoVLA (89.1) and RecogDrive (90.8). AutoVLA (89.1) and RecogDrive (90.8) use stronger vision encoders; DreamerAD uses encoder pretrained exclusively on unsupervised video — making direct numerical comparison across encoder classes misleading. The 88.7 PDMS is the strongest result for Epona-class world-model methods. DreamerAD is not compared against FLARE (91.4), DriveFine (90.7), or WAM-Flow (90.3). See [[sources/dreameraD.md]].

**Caveat on Epona (86.2 PDMS)**: NAVSIM table compares only against pre-2025 baselines (UniAD 83.4, PARA-Drive 84.6, LAW 84.6, TransFuser 84.0, DRAMA 85.5, VADv2 80.9). No head-to-head with DiffusionDrive (88.1), ReCogDrive (89.6), WAM-Flow (90.3), DriveFine (90.7), FLARE (91.4), or DiffusionDriveV2 (91.2) — all of which exceed it. Camera-only, front camera, no auxiliary supervision. Epona's primary contribution is world model quality (FVD 82.8 NuScenes SOTA) and long-horizon generation (120s), not NAVSIM planning. DreamerAD adds latent RL on top of Epona to reach 88.7 PDMS. No NAVSIM-v2 / EPDMS reported. See [[sources/epona.md]].

**Caveat on WAM-Diff (91.0 PDMS / 89.7 EPDMS)**: NAVSIM-v1 table compares against UniAD (83.4), DiffusionDrive (88.1), ReCogDrive (90.8), DriveVLA-W0 (90.2) — a reasonable but incomplete set; no head-to-head with FLARE (91.4), DiffusionDriveV2 (91.2), or HybridDriveVLA (92.1). NAVSIM-v2 table compares against TransFuser, HydraMDP++, ARTEMIS, DiffusionDrive, DriveVLA-W0 — excludes DriveDreamer-Policy (88.7), DreamerAD (87.7), and Senna-2 (86.6). WAM-Diff's 89.7 EPDMS, if genuine, would be a new single-sample SOTA on NAVSIM-v2; uses "v2.2 codebase version" which may or may not coincide with DriveFine's bug-fixed scorer (DriveFine's 89.7 EPDMS is explicitly bug-fixed; WAM-Diff's is on the standard v2.2 release). EC = 78.5, below DiffusionDrive (87.7) — masked diffusion doesn't specifically target extended comfort. 1 camera (front-only), no temporal history; both acknowledged as limitations. See [[sources/wam-diff.md]].

**Caveat on HybridDriveVLA (92.1 PDMS / 85.5 EPDMS)**: Table 4 compares against DiffusionDriveV2 (91.2), iPad (91.7), WAM-diff/WAM-Flow (91.0), ReCogDrive (90.8) — the comparison set is relatively up-to-date and includes the main recent SOTA contenders. HybridDriveVLA runs **two complete models simultaneously** (VLM + ViT branches) — it is an ensemble method, not a single-system result. It should not be directly compared with single-model numbers. NAVSIM-v2 DAC = 92.2 (notably lower than DiffusionDriveV2's 96.6) — interpolated trajectories occasionally violate drivable area constraints. EPDMS 85.5 ties DiffusionDriveV2 and remains well below DriveDreamer-Policy (88.7), DreamerAD (87.7), and the v2 leaders. See [[sources/hybriddriveVLA.md]].

**Caveat on ExploreVLA (90.4 / 93.7 BoN-6 PDMS; 88.8 EPDMS)**: NAVSIM v1 Table 1 omits WAM-Diff (91.0), DriveFine (90.7–91.8), HybridDriveVLA (92.1), FLARE (91.4), and DiffusionDriveV2 (91.2) — ExploreVLA single-sample (90.4) is below all these. The claim that BoN-6 (93.7) surpasses DriveSuprim (93.5) conflates BoN-6 vs. single-sample inference regimes. NAVSIM v2 Table 2 omits WAM-Diff (89.7), DriveDreamer-Policy (88.7), and DreamerAD (87.7); ExploreVLA's "SOTA" claim on v2 is overstated — WAM-Diff (89.7) surpasses it. ExploreVLA single-sample (90.4 v1, 88.8 v2) is competitive but not the frontier. 1 camera (front-view only). Stage 2 GRPO not applied to nuScenes. EC = 86.8 is notably strong for a 1-cam GRPO method. See [[sources/explorevla.md]].

**Caveat on DriveVA (90.9 PDMS)**: Table 1 (NAVSIM comparison) header is present in the source file but data rows are truncated — per-metric sub-scores (NC, DAC, EP, TTC, C) and the specific comparison methods are unavailable. The 90.9 PDMS claim is from the abstract/text. The paper groups methods as "World Model Methods" vs. "Traditional End-to-End" and claims SOTA within the world model category — comparison may exclude non-world-model leaders (DriveSuprim 93.5, HybridDriveVLA 92.1, FLARE 91.4, DiffusionDriveV2 91.2). Zero-shot nuScenes/Bench2Drive gains are relative to PWM only — no comparison to VLA methods on those benchmarks. No NAVSIM-v2 / EPDMS reported. 5B Wan2.2-TI2V backbone, no latency numbers, no RL stage. Confidence: medium (table truncation prevents full verification). See [[sources/driveva.md]].

**Caveat on DriveSuprim (93.5 PDMS / 87.1 EPDMS)**: NAVSIM-v1 comparison table (Table 2) compares within selection-based methods — Hydra-MDP and DiffusionDrive at all backbones — no head-to-head with VLM methods (FLARE 91.4, WAM-Diff 91.0, DriveFine 90.7) or HybridDriveVLA (92.1). The 93.5 figure is credible as an improvement over Hydra-MDP (89.9 ViT-L) within the same paradigm. NAVSIM-v2 table (Table 3) compares against TransFuser, HydraMDP++, and ARTEMIS — excludes DriveDreamer-Policy (88.7), DreamerAD (87.7), WAM-Diff (89.7), and Senna-2 (86.6). 87.1 EPDMS slots below DreamerAD (87.7) and well below WAM-Diff (89.7). EC = 78.6 (middle range). DriveSuprim uses Camera-only (3-cam ViT-L), no LiDAR, no VLM — comparison with multi-modal methods should account for modality differences. See [[sources/drivesuprim.md]].

**Caveat on DiffusionDriveV2 (91.2 PDMS / 85.5 EPDMS)**: NAVSIM v1 table compares against DriveSuprim (89.9), Hydra-MDP V2-99 (90.3), GoalFlow V2-99 (90.3), DIVER (88.3), WoTE (88.3) — fair comparison; no head-to-head with FLARE (91.4) or WAM-Flow (90.3) at the VLM-era frontier. NAVSIM v2 table only includes Transfuser, Hydra-MDP++, DriveSuprim, and ARTEMIS — excludes DriveDreamer-Policy (88.7), DreamerAD (87.7), and Senna-2 (86.6). DiffusionDriveV2 (85.5 EPDMS) is below all three omitted methods. EC = 91.0 is the highest in the wiki, suggesting the RL training strongly improves comfort and extended comfort specifically. Uses ResNet-34 + Camera+LiDAR, no VLM; 2 denoising steps at inference. See [[sources/diffusiondrive-v2.md]].

**Caveat on DiffusionDrive (88.1)**: Table 1 compares against Transfuser-era baselines (UniAD 83.4, Transfuser 84.0, DRAMA 85.5, VADv2 80.9, Hydra-MDP-W-EP 86.5) — no VLA-era methods included. Uses Camera+LiDAR (ResNet-34); not comparable to camera-only VLMs. 88.1 PDMS was SOTA at publication (November 2024) but has been superseded by ReCogDrive (89.6), WAM-Flow (90.3), DriveFine (90.7), and FLARE (91.4). DiffusionDrive is now the canonical non-VLM diffusion baseline. See [[sources/diffusiondrive.md]].

**Caveat on AutoVLA (89.11)**: Table 1 compares against Hydra-MDP (91.26), Centaur (92.10), TrajHF (93.95) — older non-VLM baselines — and does not include DriveFine, WAM-Flow, or Curious-VLA. AutoVLA post-RFT (89.11) is below the current single-sample SOTA (DriveFine 90.7). The 89.11 figure cross-validates with the "AutoVLA 89.1" entry in Curious-VLA's comparison table. See [[sources/autovla.md]].

**Caveat on NoRD (85.6 / 92.4 BoN-6)**: Table 3 compares against BEV-based methods (UniAD, Transfuser, Hydra-MDP, DiffusionDrive) and VLA-based methods (AutoVLA, RecogDrive). NoRD single (85.6) is below all VLA baselines in its own table. The key claim is data efficiency — fewer than 80K samples with no reasoning annotations, 3 cameras only, no LiDAR. NoRD-BoN (92.4) is oracle best-of-6 and surpasses AutoVLA-BoN (92.1) — but BoN inference is not a realistic deployment mode. No head-to-head with DriveFine (90.7), WAM-Flow (90.3), or FLARE (91.4). See [[sources/nord.md]].

## NAVSIM-v2 SOTA (EPDMS)

| Method                  | EPDMS    | EC   | Notable weakness / caveat                                                                                                  |
| ----------------------- | -------- | ---- | -------------------------------------------------------------------------------------------------------------------------- |
| Artemis                 | 83.1     | 89.1 | —                                                                                                                          |
| RecogDrive              | 83.6     | 87.7 | —                                                                                                                          |
| DiffusionDrive          | 84.5     | 87.7 | —                                                                                                                          |
| WAM-Flow                | 84.7     | 73.9 | Low EC (comfort); high safety metrics                                                                                      |
| ResAD                   | 85.5     | —    | —                                                                                                                          |
| Curious-VLA             | 85.3     | 81.5 | EC regression from exploration; thin v2 comparison set                                                                     |
| Senna-2                 | 86.6     | —    | Requires separate NAVSIM fine-tune                                                                                         |
| DriveVLA-W0             | 86.1     | 58.9 | Lowest EC in wiki; world model training-time only                                                                          |
| **DriveDreamer-Policy** | **88.7** | 79.4 | Thin comparison set (only DriveVLA-W0 as VLA baseline); low EC                                                             |
| FLARE-4B                | 86.3     | 87.5 | Omits Senna-2 (86.6) and DDP (88.7) from comparison; best EC among VLA wiki entries                                        |
| DriveSuprim             | 87.1     | 78.6 | Selection-based ViT-L; coarse-to-fine + rotation aug + EMA self-distill; comparison only vs. HydraMDP++ — excludes DDP, DreamerAD, WAM-Diff |
| DreamerAD               | 87.7     | 72.4 | Latent world model RL; beats WorldRFT (86.7) and Senna-2 (86.6); compares directly vs. WorldRFT in table                   |
| Vega (single)           | 86.9     | 76.3 | Instruction-conditioned; no RL; 1 camera; comparison vs. DriveVLA-W0 only                                                  |
| **Vega BoN-6**          | **89.4** | 84.5 | Best-of-6; likely surpasses DDP (88.7) but no direct head-to-head                                                          |
| DiffusionDriveV2        | 85.5     | 91.0 | DiffusionDrive + RL; EC = 91.0 (very strong); comparison table excludes DDP, DreamerAD, Senna-2                            |
| HybridDriveVLA          | 85.5     | 87.0 | VLM+ViT dual-branch ensemble; ties DDV2; DAC = 92.2 (lowest in table); comparison excludes DDP, DreamerAD, Senna-2         |
| **WAM-Diff**            | **89.7** | 78.5 | Masked diffusion + MoE + GSPO; 1 cam; comparison excludes DDP (88.7), DreamerAD (87.7), Senna-2 (86.6); uses v2.2 codebase |
| ExploreVLA              | 88.8     | 86.8 | Show-o backbone; safety-gated entropy GRPO; 1 cam; Table 2 omits WAM-Diff (89.7), DDP (88.7), DreamerAD (87.7); EC = 86.8 (strong) |

DreamerAD ([[sources/dreameraD.md]]) achieves 87.7 EPDMS — surpasses Senna-2 (86.6), WorldRFT (86.7), and FLARE (86.3). Unlike DDP, DreamerAD compares directly against WorldRFT and Epona in its own table, strengthening its position.

**WAM-Diff (89.7)** ([[sources/wam-diff.md]]) is the current wiki NAVSIM-v2 SOTA at single-sample — but **no direct head-to-head** with DriveDreamer-Policy (88.7), DreamerAD (87.7), or Senna-2 (86.6); those methods are absent from WAM-Diff's comparison table. The v2.2 codebase version may or may not be directly comparable to DriveFine's bug-fixed scorer. If the 89.7 number is consistent with the standard scorer, WAM-Diff sets a new single-sample NAVSIM-v2 SOTA. EC = 78.5 (below DDP 79.4 and much below FLARE 87.5).

**Vega BoN-6 (89.4)** ([[sources/vega.md]]) holds the best-of-N wiki NAVSIM-v2 result — no direct head-to-head with DriveDreamer-Policy (88.7), DreamerAD (87.7), or Senna-2 (86.6). Vega uses BoN-6 (best of 6 samples), a different inference regime from single-sample results.

DriveDreamer-Policy ([[sources/drivedreamer-policy.md]]) remains the highest **single-sample** NAVSIM-v2 result at 88.7 among methods that have been directly compared. FLARE (86.3) claims SOTA but excludes Senna-2 and DDP from its comparison table — the 86.3 result falls below both. **Caveat**: Table 2 only compares against DriveVLA-W0 (86.1) as a VLA baseline; Senna-2, DriveFine, and Curious-VLA are not included — the 88.7 EPDMS result is likely genuine but lacks direct head-to-head verification. EC = 79.4 — geometry/video world learning improves safety-oriented metrics but not extended comfort.

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

---
title: Wiki Index
type: comparison
sources: []
related: [concepts/discrete-flow-matching.md, concepts/diffusion-planner.md, concepts/rl-for-ad.md, concepts/vlm-domain-adaptation.md, concepts/navsim-benchmark.md, concepts/world-model-for-ad.md, concepts/dual-system-vla.md, concepts/inference-time-safety.md, concepts/perception-for-planning.md, concepts/best-of-n.md, concepts/bench2drive.md, concepts/chain-of-thought-for-ad.md, concepts/mixture-of-experts.md, concepts/selection-based-planning.md, concepts/action-tokenization.md, concepts/gspo-vs-grpo.md, concepts/pdm-lite.md, concepts/nuscenes-waymo-evals.md, concepts/foundation-backbones-for-ad.md, concepts/navhard-ood-evaluation.md, concepts/hugsim-benchmark.md]
created: 2026-04-05
updated: 2026-05-01
confidence: high
---

# Wiki Index

Master catalog of all wiki pages. Updated on every ingest.

---

## Sources

| Page                                      | Description                                                                                                                                                                          |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [ReCogDrive](sources/recogdrive.md)       | VLM + diffusion planner + RL for end-to-end AD; PDMS 89.6 on NAVSIM-v1                                                                                                               |
| [WAM-Flow](sources/wam-flow.md)           | Discrete flow matching VLA for AD; PDMS 90.3 on NAVSIM-v1; 1 camera only                                                                                                             |
| [UniUGP](sources/uniugp.md)               | Unified VLA + world model; CoT + video generation + FM trajectory; SOTA nuScenes FID/FVD and DriveLM                                                                                 |
| [Senna-2](sources/senna2.md)              | Dual-system VLM + E2E alignment; 3DGS HRL; +19.3% consistency F1; EPDMS 86.6 on NAVSIM-v2                                                                                            |
| [ReflectDrive](sources/reflectdrive.md)   | Masked discrete diffusion + gradient-free reflective inference; goal-conditioned NMS + safety anchor inpainting; claims >AutoVLA on NAVSIM-v1                                        |
| [Reasoning-VLA](sources/reasoning-vla.md) | Learnable action queries (1-step parallel); unified 8-dataset corpus; GT-based GRPO; 91.7 PDMS claimed (comparison scope limited); 61× faster than AR                                |
| [Percept-WAM](sources/percept-wam.md)     | World-PV/BEV tokens unify 2D/3D perception + planning in one VLM; IoU-aware confidence; four-query trajectory decoder; 90.2 PDMS; 707ms latency                                      |
| [ORION](sources/orion.md)                 | QT-Former + Vicuna LLM + VAE generative planner; reasoning-action latent alignment; 77.74 DS / 54.62% SR on Bench2Drive (+14.28 DS vs. SOTA at time)                                 |
| [LinkVLA](sources/linkvla.md)             | Shared language-action codebook; bidirectional alignment (action captioning); C2F 2-pass decoder; 91.01 DS / 74.55% SR Bench2Drive SOTA; 48ms latency                                |
| [HERMES](sources/hermes.md)               | Offline VLM annotation → BGE-M3 embeddings → risk-aware Tri-Modal student; WOD-E2E long-tail SOTA 6.81 RFS; no closed-loop eval                                                      |
| [DriveFine](sources/drivefine.md)         | Block-MoE masked diffusion (LLaDA-8B) + hybrid offline/online RL; 90.7/91.8 PDMS NAVSIM-v1; 89.7 EPDMS (bug-fixed) NAVSIM-v2                                                         |
| [Curious-VLA](sources/curious-vla.md)     | Narrow Policy diagnosis (IL→RL diversity collapse); FTE + ADAS + SDR; 90.3 PDMS / 85.3 EPDMS; BoN-6 94.8 matching human GT; Qwen2.5-VL-3B 1xC                                        |
| [AutoVLA](sources/autovla.md)             | K-Disk physical action codebook (K=2048); dual-mode SFT + GRPO with CoT length penalty; adaptive fast/slow reasoning; 89.11 PDMS / 92.12 BoN; 3xC                                    |
| [AutoMoT](sources/automot.md)             | Frozen Qwen3-VL-4B UE + 1.6B AE from scratch; layer-wise KV cache async (7.6× speedup); 87.34 DS Bench2Drive; catastrophic forgetting evidence for VLM fine-tuning                   |
| [AutoDrive-R²](sources/autodrive-r2.md)   | 4-step CoT with self-reflection (backward-check) + physics GRPO (pos/steering/vel/temporal); 0.19m nuScenes / 0.20m Waymo zero-shot; 6K training samples                             |
| [Alpamayo-R1](sources/alpamayo-r1.md)     | Cosmos-Reason backbone + CoC dataset (700K, hybrid labeling) + FM action expert (unicycle dynamics); 3-reward GRPO (LRM-as-critic + consistency + safety); 99ms; internal evals only |
| [AdaThinkDrive](sources/adathinkdrive.md) | Adaptive Think Reward GRPO (mode-comparison per scene); dual-mode SFT (Think+NonThink same query); 90.3 PDMS / 93.0 BoN-4; InternVL3-8B; 1 camera; 14% faster than always-Think |
| [FutureSightDrive](sources/futuresightdrive.md) | Visual spatio-temporal CoT (VQ-VAE AR future frame with lane dividers + 3D boxes); dual-role VLA (world model + inverse dynamics); 85.1 PDMS NAVSIM; 0.96m L2 nuScenes; FID 10.1; Qwen2-VL-2B |
| [DriveDreamer-Policy](sources/drivedreamer-policy.md) | Geometry-grounded WAM; causal depth→video→action FM generators; Qwen3-VL-2B; 89.2 PDMS NAVSIM-v1; 88.7 EPDMS NAVSIM-v2; FVD 53.59 |
| [DriveVLA-W0](sources/drivevla-w0.md) | Supervision deficit framing; AR + diffusion world models (training-time only); MoE action expert; 90.2★ PDMS (anchors) / 93.0 BoN-6; scaling reversal FM→AR at 70M frames |
| [UniDriveVLA](sources/unidrivevla.md) | HUST + Xiaomi EV; MoT 3-expert (und/per/act) + masked joint attention; sparse 5-task perception; 3-stage progressive training; 78.37 DS Bench2Drive (best w/o PDM-Lite); 0.51m L2 nuScenes no-ego |
| [FLARE](sources/flare.md) | OpenDriveLab + Li Auto; annotation-free DINOv2 future feature prediction + DiT + BC-GRPO; 86.9 PDMS SFT / 91.4 PDMS RFT (strong VLM RFT; later DynVLA reports 91.7 with comparison caveats); 1 camera |
| [Epona](sources/epona.md) | AR+Diffusion WM (MST+TrajDiT+VisDiT, 2.5B); chain-of-forward training; FVD 82.8 NuScenes SOTA; 120s generation; 86.2 PDMS NAVSIM-v1 (pre-VLA baselines only); backbone for DreamerAD |
| [DreamerAD](sources/dreameraD.md) | Latent world model RL; SF-WM (80× speedup) + AD-RM (latent rewards) + Gaussian vocab sampling; 87.7 EPDMS NAVSIM-v2 / 88.7 PDMS NAVSIM-v1; Epona backbone |
| [Vega](sources/vega.md) | Instruction-conditioned AR+Diffusion (Bagel-7B/MoT); InstructScene 100K; future image as dense supervision; 86.9 EPDMS / 89.4 BoN-6 NAVSIM-v2; open-ended NL instruction following |
| [NoRD](sources/nord.md) | Reasoning-free VLA; k-disc tokens (2048); Dr. GRPO over GRPO (+11.68% vs +0.67%); 85.6 PDMS / 92.4 BoN-6 NAVSIM; 3rd RFS WaymoE2E with 6–17× less data; difficulty bias identification |
| [DiffusionDrive](sources/diffusiondrive.md) | Truncated diffusion (20 anchors, 2 steps); cascade decoder (60M, 45 FPS); 88.1 PDMS NAVSIM; 74% mode diversity; canonical non-VLM diffusion baseline; ResNet-34 + C+L |
| [DiffusionDriveV2](sources/diffusiondrive-v2.md) | DiffusionDrive + Intra/Inter-Anchor GRPO + multiplicative exploration noise; 91.2 PDMS NAVSIM-v1 / 85.5 EPDMS NAVSIM-v2; strong non-VLM diffusion baseline; ResNet-34 + C+L |
| [HybridDriveVLA / DualDriveVLA](sources/hybriddriveVLA.md) | 3-RQ complementarity analysis (CKA/CCA/SAE); VLM+ViT dual-branch + style-axis interpolation + trajectory scorer; 92.10 PDMS NAVSIM-v1; fast–slow DualDriveVLA 91.0 PDMS @ 3.2× throughput |
| [WAM-Diff](sources/wam-diff.md) | Masked diffusion VLA (LLaDA-V 8.4B) + LoRA MoE (64 experts) + GSPO (sequence-level RL); reverse-causal decoding; 91.0 PDMS NAVSIM-v1; 89.7 EPDMS NAVSIM-v2 |
| [DriveSuprim](sources/drivesuprim.md) | Non-VLM selection-based (8192 vocab); coarse-to-fine (→256) + rotation aug + EMA self-distill; **93.5 PDMS NAVSIM-v1** (highest non-BoN in wiki); 87.1 EPDMS NAVSIM-v2 |
| [DriveVA](sources/driveva.md) | Wan2.2-TI2V-5B video backbone; single DiT over joint [video latents ‖ action tokens]; +19.5 PDMS from video supervision; 90.9 PDMS NAVSIM-v1; zero-shot −78.9% L2 nuScenes; table truncated |
| [ExploreVLA](sources/explorevla.md) | Show-o (Phi-1.5 + MAGVIT-v2); dense RGB+depth world model SFT; safety-gated entropy exploration reward (GRPO); 90.4 PDMS / 93.7 BoN-6 NAVSIM-v1; 88.8 EPDMS NAVSIM-v2; 1 cam |
| [ELF-VLA](sources/elf-vla.md) | InternVL3-8B with explicit learning from failures; Qwen3-VL-32B teacher diagnostics + feedback-guided refinement injected into GRPO; 91.0 PDMS NAVSIM-v1 / 87.1 EPDMS NAVSIM-v2 |
| [DynVLA](sources/dynvla.md) | Dynamics CoT: compact ego/environment dynamics tokens before action tokens; VQ tokenizer + SFT + GRPO RFT; 91.7 PDMS NAVSIM-v1; 88.34 DS Bench2Drive |
| [SpanVLA](sources/spanvla.md) | Sparse-KV action bridge + continuous flow-matching action expert initialized from history; GRPO over positive/negative/recovery samples; 90.3 PDMS NAVSIM-v1; 86.4 EPDMS NAVSIM-v2 |
| [OneDrive](sources/onedrive.md) | Single causal VLM decoder for AR text + parallel detection/lane/planning queries; InternVL3 attention-transfer finding; 0.28 L2 / 0.18 collision nuScenes; 86.8 PDMS NAVSIM-v1; 156ms NAVSIM latency |
| [OneVL](sources/onevl.md) | One-step latent CoT with language and visual auxiliary decoders; future-frame world-model supervision; 88.84 PDMS NAVSIM-v1 at answer-only latency; 0.24s MLP deployment variant |
| [HAD](sources/had.md) | Hierarchical diffusion + polar trajectory expansion + MDPO/offline reward retrieval; 90.2 PDMS NAVSIM-v1; 88.6 EPDMS NAVSIM-v2; 47.5 RC / 30.8 HDS HUGSIM |
| [Latent-WAM](sources/latent-wam.md) | Compact latent world-action model with DINOv2 scene queries, WorldMirror geometric distillation, and causal DLWM; 89.3 EPDMS NAVSIM-v2; 45.9 RC / 28.9 HDS HUGSIM |
| [Drive-JEPA](sources/drive-jepa.md) | V-JEPA video pretraining + multimodal trajectory distillation + momentum-aware selection; 93.3 PDMS NAVSIM-v1; 87.8 EPDMS NAVSIM-v2; 64.52 DS Bench2Drive |
| [FeaXDrive](sources/feaxdrive.md) | Feasibility-aware trajectory-centric diffusion planning with adaptive curvature regularization, drivable-area guidance, and FA-GRPO; 90.0 PDMS NAVSIM-v1 |
| [Policy World Model](sources/policy-world-model.md) | Action-free video world model + future-frame rationales for planning; 28-token frame tokenizer; 88.1 PDMS NAVSIM-v1; 0.41 L2 / 0.04 collision nuScenes w/ ego |

---

## Concepts

| Page | Description |
|------|-------------|
| [Discrete Flow Matching](concepts/discrete-flow-matching.md) | DFM over token spaces via CTMC; parallel bidirectional generation; geometry-aware Gibbs paths; metric-aligned numerical tokenizer |
| [Diffusion-Based Trajectory Planner](concepts/diffusion-planner.md) | DDPM/DiT applied to continuous trajectory generation; MoT coupling; DFM and FM comparisons |
| [Reinforcement Learning for Autonomous Driving](concepts/rl-for-ad.md) | RL approaches in AD; GRPO applied to diffusion and DFM policies; sim-assisted RL |
| [VLM Domain Adaptation for Autonomous Driving](concepts/vlm-domain-adaptation.md) | Adapting general VLMs to driving via data curation, SFT, CoT integration; multi-stage training |
| [NAVSIM Benchmark](concepts/navsim-benchmark.md) | Planning benchmark, PDMS/EPDMS metrics, non-reactive simulator; current SOTA |
| [World Models for Autonomous Driving](concepts/world-model-for-ad.md) | Video, latent, feature, and dynamics-token world models for planning; coupling and evaluation caveats |
| [Dual-System VLA](concepts/dual-system-vla.md) | VLM for decisions + E2E for trajectory; decision adapter; kinematic mapping; consistency alignment |
| [Inference-Time Safety](concepts/inference-time-safety.md) | Gradient-free safety correction at inference; discrete token search + inpainting-as-repair; taxonomy vs. guidance/RL/anchors |
| [Perception-Enhanced Planning](concepts/perception-for-planning.md) | World-PV/BEV tokens; grid-conditioned parallel AR detection; IoU-aware confidence calibration; sparse MoT perception (UniDriveVLA); cosine similarity collapse evidence |
| [Best-of-N Sampling](concepts/best-of-n.md) | Oracle trajectory selection from N samples; NAVSIM-v1 saturated at BoN-6 (94.8 = human GT); implications for benchmark interpretation; DreamerAD as deployable BoN variant |
| [Bench2Drive Benchmark](concepts/bench2drive.md) | CARLA V2 closed-loop; interactive agents; DS + SR metrics; SOTA LinkVLA 91.01 DS; PDM-Lite caveat; contrast with NAVSIM |
| [Chain-of-Thought for AD](concepts/chain-of-thought-for-ad.md) | Text/visual/self-reflection CoT types; annotation methods (frontier VLM, GT-grounded, LRM-as-critic); adaptive CoT (AdaThinkDrive); NoRD challenges necessity; efficiency tradeoffs |
| [Mixture of Experts for AD](concepts/mixture-of-experts.md) | 4 MoE patterns: sparse LoRA (WAM-Diff), block-level task routing (DriveFine), MoT frozen+trained (AutoMoT), 3-stream MoT (UniDriveVLA); RL routing instability → GSPO; catastrophic forgetting evidence |
| [Selection-Based Planning](concepts/selection-based-planning.md) | Fixed-vocabulary trajectory scoring; coarse-to-fine filtering; oracle ceiling 98.7 PDMS (top-256); hard-negative / directional bias / hard-label failure modes; DriveSuprim, DreamerAD, HybridDriveVLA |
| [Action Tokenization and Codebooks](concepts/action-tokenization.md) | Discrete action vocabularies, learned codebooks, continuous action experts, and tokenizer tradeoffs across AD VLAs |
| [GSPO vs. GRPO](concepts/gspo-vs-grpo.md) | Sequence-level RL for MoE/masked-diffusion policies vs. token/group-level GRPO recipes |
| [PDM-Lite](concepts/pdm-lite.md) | Privileged oracle planner/fallback caveat for Bench2Drive comparisons |
| [nuScenes and Waymo Evaluations](concepts/nuscenes-waymo-evals.md) | Open-loop L2/collision/RFS metrics and why they do not substitute for NAVSIM or Bench2Drive |
| [Foundation Backbones for AD](concepts/foundation-backbones-for-ad.md) | Qwen, InternVL, Cosmos, Wan, Show-o, and other backbone choices used in driving VLAs |
| [Navhard and OOD Evaluation](concepts/navhard-ood-evaluation.md) | NAVSIM-v2 navhard, distribution-shift scoring, and OOD caveats |
| [HUGSIM Benchmark](concepts/hugsim-benchmark.md) | Closed-loop planning benchmark with route completion and HD-Score; HAD reports 47.5 RC / 30.8 HDS |

---

## Entities

*(none yet)*

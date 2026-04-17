# Activity Log

Append-only log of all wiki operations.

---

## 2026-04-17 — Ingest: Epona

**Source**: `raw/papers/Epona_ Autoregressive Diffusion World Model for Autonomous Driving.md`  
**arXiv**: 2506.24113  
**Org**: Horizon Robotics, Tsinghua, PKU, NJU, HKUST, NTU, Tencent  
**Venue**: ICCV 2025

**Pages created**:
- `wiki/sources/epona.md` — full source summary with 7 figures (Figure 1 URL-only, not locally available), all tables, complete architecture description (MST + TrajDiT + VisDiT), chain-of-forward training formulation, and full ablation data

**Concept pages updated**:
- `wiki/concepts/world-model-for-ad.md` — expanded Pattern 2 stub (Autoregressive WM + Diffusion Planner) into full section covering MST architecture, chain-of-forward, shared latent ablation, inference modes, and DreamerAD relationship; linked Epona to FID/planning tables; bumped frontmatter
- `wiki/concepts/navsim-benchmark.md` — added Epona (86.2 PDMS) to SOTA v1 table; added caveat paragraph noting pre-2025-baseline-only comparison; bumped frontmatter

**Source pages updated**:
- `wiki/sources/dreameraD.md` — added `sources/epona.md` to related frontmatter

**Index updated**: added Epona row to Sources table (inserted above DreamerAD).

**Key facts**:
- FVD 82.8 NuScenes (SOTA at time; −7.4% vs Vista 89.4); generation horizon 120s / 600 frames (vs Vista 15s)
- 86.2 PDMS NAVSIM-v1 camera-only; comparison table excludes all VLA-era methods (DriveFine 90.7, WAM-Flow 90.3, etc.)
- Joint video+trajectory training critical: disabling VisDiT → PDMS 86.2 → 78.1 (−8.1)
- Chain-of-forward training: 1-step velocity estimate prevents autoregressive drift in long-horizon generation
- Real-time planning (20 Hz) only with VisDiT disabled; full generation is ~2.3s/frame
- Epona is DreamerAD's base model; DreamerAD adds latent RL → 88.7 PDMS (+2.5)
- Figure 1 in source file is a URL reference to arxiv — not saved locally as an asset

---

## 2026-04-16 — Ingest: DiffusionDriveV2

**Source**: `raw/papers/DiffusionDriveV2_ Reinforcement Learning-Constrained Truncated Diffusion Modeling in End-to-End Autonomous Driving.md`  
**arXiv**: 2512.07745  
**Org**: HUST (EIC + AI Institute), Horizon Robotics, Wuhan University  
**Venue**: December 2024

**Pages created**:
- `wiki/sources/diffusiondrive-v2.md` — full source summary with all 8 figures, all tables, and complete method description including Intra/Inter-Anchor GRPO formulations and multiplicative noise derivation

**Concept pages updated**:
- `wiki/concepts/rl-for-ad.md` — added "DiffusionDriveV2: Anchored Truncated GRPO" section; updated GRPO comparison table with DiffusionDriveV2 row; bumped frontmatter `updated` and `sources`/`related`
- `wiki/concepts/diffusion-planner.md` — added DiffusionDriveV2 subsection under DiffusionDrive section; updated comparison table to include V2 row; bumped frontmatter
- `wiki/concepts/navsim-benchmark.md` — added DiffusionDriveV2 to NAVSIM v1 SOTA table (91.2) and v2 SOTA table (85.5 EPDMS); added caveat paragraph; bumped frontmatter

**Index updated**: added DiffusionDriveV2 row to Sources table.

**Key facts**:
- 91.2 PDMS NAVSIM-v1 (highest non-VLM result in wiki; +3.1 over DiffusionDrive); 85.5 EPDMS NAVSIM-v2
- EC = 91.0 on NAVSIM-v2 (highest extended comfort in wiki)
- NAVSIM-v2 caveat: 85.5 EPDMS below DriveDreamer-Policy (88.7), DreamerAD (87.7), Senna-2 (86.6); those methods excluded from V2's comparison table
- Intra-Anchor GRPO prevents mode collapse from cross-intent advantage comparison (+0.9 PDMS ablation)
- Inter-Anchor Truncated GRPO provides global collision penalty floor (+0.6 PDMS ablation)
- Multiplicative exploration noise preserves trajectory smoothness (+0.4 PDMS ablation)

---

## 2026-04-15 — Lint + New Concept Pages

**Lint fixes applied**:
- `wiki/concepts/world-model-for-ad.md` — Pattern 8 (FLARE) was displaced after the SOTA tables (after Patterns 9 and 10); moved to its correct position between Pattern 7 (DriveVLA-W0) and Pattern 9 (DreamerAD). `updated` bumped to 2026-04-15.
- `wiki/concepts/navsim-benchmark.md` — Added disambiguation dagger to ReCogDrive SOTA row: 89.6 (arXiv) vs. 90.8 (NeurIPS camera-ready as cited by DreamerAD).
- `wiki/concepts/inference-time-safety.md` — Added DriveFine Block-MoE to taxonomy table with training-time vs. inference-time contrast; added `sources/drivefine.md` and `sources/diffusiondrive.md` to `related` and `sources` frontmatter; `updated` bumped to 2026-04-15.

**New concept pages created**:
- `wiki/concepts/best-of-n.md` — Oracle BoN sampling; NAVSIM-v1 saturation at BoN-6 (94.8 = human GT); DreamerAD vocabulary sampling as deployable variant; implications for benchmark interpretation
- `wiki/concepts/bench2drive.md` — CARLA V2 closed-loop benchmark; DS + SR metrics; full SOTA table (LinkVLA 91.01 DS current SOTA); PDM-Lite caveat; contrast with NAVSIM
- `wiki/concepts/chain-of-thought-for-ad.md` — Text/visual/self-reflection CoT taxonomy; 3 annotation strategies (frontier VLM, GT-grounded, LRM-as-critic); adaptive CoT (AdaThinkDrive); NoRD as reasoning-free counterpoint; efficiency tradeoff table

**Index updated**: added 3 new concept rows.

---

## 2026-04-15 — Ingest: DiffusionDrive

**Source**: `raw/papers/DiffusionDrive_ Truncated Diffusion Model for End-to-End Autonomous Driving.md`  
**arXiv**: 2411.15139v1  
**Org**: HUST (Institute of AI + School of EIC); Horizon Robotics  
**Venue**: pre-VLA era (November 2024)

**Pages created**:
- `wiki/sources/diffusiondrive.md`

**Pages updated**:
- `wiki/concepts/diffusion-planner.md` — added full DiffusionDrive section: two failure modes of vanilla diffusion (mode collapse 11% diversity, 7 FPS), truncated diffusion policy (20 K-Means anchors, T_trunc=50/1000, 2 DDIM steps), cascade diffusion decoder (deformable spatial+agent/map cross-attention, 2 layers shared params, 60M/-39% params), progression table (Transfuser→DD), DiffusionDrive vs. VLA-era comparison; added DiffusionDrive as first row of design space table; updated DFM comparison to include source link; updated sources/related frontmatter
- `wiki/concepts/navsim-benchmark.md` — filled in DiffusionDrive table row note (truncated diffusion, 20 anchors, 2 steps, ResNet-34, 45 FPS); added DiffusionDrive caveat (comparison scope limited to Transfuser-era; 88.1 PDMS SOTA at publication, superseded by VLA methods); added DiffusionDrive to sources/related frontmatter
- `wiki/index.md` — added DiffusionDrive row

**Assets embedded**:
- `x1 19.png` — paradigm comparison (single-mode / vocab / vanilla diffusion / truncated diffusion)
- `x2 17.png` — mode diversity visualization (vanilla diffusion mode collapse)
- `x4 16.png` — truncated vs. vanilla diffusion schedule illustration
- `x5 17.png` — overall DiffusionDrive architecture

**Key findings**:
- Vanilla diffusion policy applied to driving: 11% mode diversity (near-complete collapse), 7 FPS — both unacceptable
- Truncated diffusion: start from anchored Gaussian (20 K-Means clusters), truncate forward schedule to 50/1000 steps, denoise in 2 steps → 74% diversity, 27 FPS
- Cascade decoder (deformable BEV/PV + agent/map cross-attention, 2 layers shared, 60M) beats UNet (101M) by +2.4 PDMS at −39% params → 45 FPS
- Spatial cross-attention is critical: removing it collapses PDMS from 87.1 to 55.1 (−32 PDMS)
- Inference flexibility: N_infer decoupled from N_anchor — dynamically scale trajectory hypotheses
- 88.1 PDMS was SOTA at publication; now the canonical non-VLM diffusion baseline, superseded by ReCogDrive (89.6), WAM-Flow (90.3), DriveFine (90.7), FLARE (91.4)
- nuScenes: 0.57m avg L2 / 0.08 collision (beats VAD by −20.8% L2, −63.6% collision, 1.8× faster)

---

## 2026-04-15 — Ingest: NoRD

**Source**: `raw/papers/NoRD_ A Data-Efficient Vision-Language-Action Model that Drives without Reasoning.md`  
**arXiv**: 2602.21172v1  
**Org**: Applied Intuition; Texas A&M University; UC Berkeley

**Pages created**:
- `wiki/sources/nord.md`

**Pages updated**:
- `wiki/concepts/rl-for-ad.md` — added NoRD section: difficulty bias identification (polarized reward distribution), GRPO attenuation mechanism (std normalization kills high-variance gradients), Dr. GRPO formulation (remove std, DAPO asymmetric clipping, no KL), reward design (format+length+PDMS), sub-metric comparison table, position in GRPO landscape; added NoRD row to GRPO reward comparison table
- `wiki/concepts/vlm-domain-adaptation.md` — added NoRD section: reasoning-free hypothesis, adaptation design (no CoT at any stage), data efficiency finding, contrast with FLARE and AutoVLA; added NoRD row to strategy comparison table
- `wiki/concepts/navsim-benchmark.md` — added NoRD (85.6 PDMS, no reasoning, no LiDAR, 3C, 80K samples) and NoRD-BoN (92.4, BoN-6, surpasses AutoVLA-BoN); added caveat on comparison scope and data efficiency framing
- `wiki/index.md` — added NoRD row

**Assets embedded** (all in raw/assets/):
- `x1 18.png` — training pipeline comparison (existing vs. NoRD)
- `difficulty_plot.png` — reward distribution for NoRD-base
- `grpo_steps.png` — GRPO training step analysis
- `comparison_figure.png` — GRPO vs. Dr. GRPO qualitative (sharp turn + lane change)
- `nord.png` — model architecture
- `x2 16.png` — NAVSIM Pareto frontier
- `navsim_examples.png` — qualitative NAVSIM results
- `waymo_results.png` — qualitative WaymoE2E results
- `nord_efficient.png` — token and runtime efficiency
- `contour_plots.png` — training improvement per variance group
- `x4 15.png` — training and validation curves
- `prompt_example.png` — inference example

**Key findings**:
- Standard GRPO fails on weak SFT policies because high intra-group variance attenuates GRPO advantages: +0.67% gain only
- Dr. GRPO (remove std normalization + DAPO asymmetric clipping, no KL) achieves +11.68% from same weak base
- Reasoning annotations are not the bottleneck: NoRD matches AutoVLA-BoN (92.4 vs. 92.1) with no CoT and 60% less data
- WaymoE2E: best ADE@3 (1.2504) with 6–17× less training data than SOTA; 3rd RFS (7.709) without reasoning or ensembling
- First identification of difficulty bias failure mode in autonomous driving domain
- Connection: Curious-VLA identified advantage collapse (policy too narrow → $\sigma_R \to 0$); NoRD identifies advantage attenuation (policy too weak → $\sigma_R$ too large); both starve GRPO from opposite distributional extremes

---

## 2026-04-08 — Ingest: Vega

**Source**: `raw/papers/Vega_ Learning to Drive with Natural Language Instructions.md`  
**arXiv**: 2603.25741v1  
**Org**: Tsinghua University + GigaAI

**Pages created**:
- `wiki/sources/vega.md`

**Pages updated**:
- `wiki/concepts/world-model-for-ad.md` — added Pattern 10: Instruction-Conditioned World Model (Vega); updated World Model vs. VLA table to include NL instruction following row; updated sources/related frontmatter
- `wiki/concepts/vlm-domain-adaptation.md` — added Vega section on instructional driving paradigm; added InstructScene annotation pipeline; updated final strategy comparison table (now 15 rows); updated sources/related frontmatter
- `wiki/concepts/navsim-benchmark.md` — added Vega to NAVSIM-v1 (87.9 / 89.8 BoN-6) and NAVSIM-v2 (86.9 / 89.4 BoN-6) SOTA tables; updated SOTA note (Vega BoN-6 likely new NAVSIM-v2 wiki SOTA but no direct head-to-head); added Vega caveat note; updated frontmatter
- `wiki/index.md` — added Vega row

**Key concepts**:
- Instructional driving: open-ended NL instruction → different trajectory in same scene (vs. imitation driving with fixed expert target)
- InstructScene: 100K automated instruction annotations via Qwen2.5-VL-72B two-stage pipeline (scene understanding → instruction formulation) + rule-based ego-motion labels
- Dense supervision bridge: future image prediction resolves instruction-to-action gap; action-only SFT fails catastrophically (51.8 PDMS); world modeling enables it (77.9→86.9 EPDMS)
- Integrated AR+Diffusion transformer (Bagel-7B, MoT): all transformer params duplicated per understanding/generation module; no information bottleneck (vs. external diffuser)
- Duplicate latent trick: noisy copy for denoising + clean copy for conditioning → joint multi-task training in single forward pass
- Lightweight action expert (hidden=256): separate from understanding/generation modules; diffusion as action planner fails catastrophically (19.7 PDMS)
- CFG: text/ViT/action tokens dropped during training → instruction guidance strength at inference
- NAVSIM-v2: 86.9 EPDMS (single, no RL) / 89.4 BoN-6; NAVSIM-v1: 87.9 PDMS / 89.8 BoN-6; 1 camera only
- EC = 76.3 (single) / 84.5 (BoN) — improved by instruction-consistent planning but not best-in-class

---

## 2026-04-08 — Ingest: DreamerAD

**Source**: `raw/papers/DreamerAD_ Efficient Reinforcement Learning via Latent World Model for Autonomous Driving.md`  
**arXiv**: 2603.24587v1  
**Org**: Chongqing Chang'an Technology Co., Ltd.

**Pages created**:
- `wiki/sources/dreameraD.md`

**Pages updated**:
- `wiki/concepts/rl-for-ad.md` — added DreamerAD section (latent world model RL; SF-WM + AD-RM + Gaussian vocab sampling); added DreamerAD row to GRPO comparison table; clarified DreamerAD's unique position as only method using latent features (not simulator) as RL reward source; updated frontmatter sources/related
- `wiki/concepts/world-model-for-ad.md` — added Pattern 9: Latent World Model as RL Reward Source; resolved open question "can world model provide RL rewards?" with DreamerAD evidence; updated frontmatter sources/related
- `wiki/concepts/navsim-benchmark.md` — added DreamerAD to NAVSIM-v1 SOTA table (88.7 PDMS); added DreamerAD to NAVSIM-v2 SOTA table (87.7 EPDMS, EC=72.4); added DreamerAD caveat note; added contextual note that DreamerAD becomes second in wiki behind DDP; updated frontmatter
- `wiki/index.md` — added DreamerAD row

**Key concepts**:
- First latent-space RL framework for AD: rewards from learned AD-RM on denoised Video DiT features, not PDM simulator (at RL time)
- Shortcut Forcing (SF-WM): recursive multi-resolution teacher-student distillation; 100→1 step, 80× speedup, 0.03s/frame, no EPDMS degradation
- PCA finding: denoised latent features show structured spatial/semantic coherence → sufficient for reward learning without decoding
- AD-RM data efficiency: 20% training data ≈ 100% reward model performance; high-quality latent representations simplify reward learning
- Safety-first log-sigmoid reward: collisions force log(σ(r)) → −∞, dominating total reward without manual safety weights
- Gaussian vocabulary sampling: Mahalanobis ranking over 8192→256 filtered trajectories; avoids WorldRFT dynamic discontinuity and Flow-GRPO SDE mismatch
- NAVSIM-v2 87.7 EPDMS: +2.6 over Epona; safety metrics NC +0.9, DAC +1.5, TTC +1.1; EP −0.8 (safety-efficiency tradeoff)
- NAVSIM-v1 88.7 PDMS: best within world-model encoder class; below VLA SOTA (FLARE 91.4, RecogDrive 90.8) using stronger encoders

---

## 2026-04-07 — Ingest: FLARE

**Source**: `raw/papers/FLARE_ Learning Future-Aware Latent Representations from Vision-Language Models for Autonomous Driving.md`  
**arXiv**: 2601.05611v2  
**Org**: OpenDriveLab + Li Auto

**Pages created**:
- `wiki/sources/flare.md`

**Pages updated**:
- `wiki/concepts/world-model-for-ad.md` — added Pattern 8: DINOv2 semantic feature prediction as self-supervised auxiliary loss; action-conditional FFP; prediction target ablation; contrast with DriveVLA-W0 (VAE latents) and FSDrive (visual CoT); updated open questions
- `wiki/concepts/rl-for-ad.md` — added FLARE's BC-regularized GRPO section; BC vs. KL comparison; updated GRPO reward comparison table (now 10 methods)
- `wiki/concepts/vlm-domain-adaptation.md` — added annotation-free adaptation section; positioning table (FLARE vs. DriveVLA-W0 as only annotation-free methods); updated strategy table (now 13 rows)
- `wiki/concepts/navsim-benchmark.md` — added FLARE SFT (86.9) and RFT (91.4) rows; updated EPDMS table (86.3, EC=87.5); updated SOTA statement (FLARE 91.4 best single-sample VLM-based, caveat: no head-to-head with DriveFine/WAM-Flow)
- `wiki/index.md` — added FLARE row

**Key concepts**:
- Annotation-free paradigm: no VQA/CoT needed; DINOv2 patch features as dense self-supervision
- Future Feature Predictor (FFP): spatial queries modulated by action vector z → cross-attention on visual latents → predict DINOv2 patches of next frame
- Action-conditional prediction: FFP conditioned on z simulates how planned action changes the scene
- Prediction target hierarchy: spatial DINO (86.9) > global DINO (85.9) > pixels (84.7) > none (83.4)
- BC regularization instead of KL divergence in GRPO Stage 2 (motivated by DriveFine reward hacking finding)
- NAVSIM-v1: 86.9 SFT (best VLM SFT, no external data), 91.4 RFT (best single-sample VLM)
- NAVSIM-v2: 86.3 EPDMS (comparison scope excludes Senna-2 86.6 and DDP 88.7); EC=87.5 (healthy)
- Two-stage MAP fusion: visual MAP → N_v latents; ego-state-conditioned action MAP → single decision vector z

---

## 2026-04-07 — Ingest: UniDriveVLA

**Source**: `raw/papers/UniDriveVLA_ Unifying Understanding, Perception, and Action Planning for Autonomous Driving.md`  
**arXiv**: 2604.02190v1  
**Org**: HUST + Xiaomi EV + University of Macau

**Pages created**:
- `wiki/sources/unidrivevla.md`

**Pages updated**:
- `wiki/concepts/dual-system-vla.md` — added MoT as third structural paradigm; UniDriveVLA + AutoMoT design comparison; Masked Joint Attention pattern; MoT ablation table; updated master comparison table (now 5 methods)
- `wiki/concepts/perception-for-planning.md` — added sparse query-based perception section; cosine similarity collapse evidence; perception–reasoning conflict diagnosis; updated comparison table (now 6 approaches including UniDriveVLA)
- `wiki/concepts/vlm-domain-adaptation.md` — added UniDriveVLA section: interference diagnosis, MoT fix, 3-stage progressive training, general VQA degradation data; updated strategy comparison table (now 12 approaches)
- `wiki/index.md` — added UniDriveVLA row; updated perception-for-planning description

**Key concepts**:
- Perception–reasoning conflict: cosine similarity → 1 in shared-weight decoder = feature collapse
- MoT: decoupled und/per/act experts; und causally masked from per/act; per reads und; act reads both
- Sparse perception: K-Means instance banks; 5-task unified decoder (det/map/ego/motion/occ); two-pass enrichment via masked joint attention
- 3-stage training: full VLM SFT → LoRA + 0.5× LR joint → VLM frozen specialization
- MoT ablation: +14.4pp General VQA, +4.1pp DriveBench, −0.108m L2 vs. shared-weight
- General VQA after adaptation still −19.7pp vs. base Qwen3-VL (MoT reduces but doesn't eliminate forgetting)
- Bench2Drive: 78.37 DS best w/o PDM-Lite; 11.78 comfort (lowest in table)
- nuScenes: 0.51m avg L2 no-ego (Large, best shown); with-ego 0.42m (FSDrive at 0.28m is better)

---

## 2026-04-07 — Ingest: DriveVLA-W0

**Source**: `raw/papers/DriveVLA-W0_ World Models Amplify Data Scaling Law in Autonomous Driving.md`  
**arXiv**: 2510.12796v1  
**Org**: CASIA + Yinwang Intelligent Technology

**Pages created**:
- `wiki/sources/drivevla-w0.md`

**Pages updated**:
- `wiki/concepts/world-model-for-ad.md` — added Pattern 7: training-time-only world modeling for data scaling; supervision deficit framing; VQ AR vs. diffusion WM design; generalization and scaling findings
- `wiki/concepts/navsim-benchmark.md` — added DriveVLA-W0 (90.2★ anchor-based, 93.0 BoN-6); v2 table updated (86.1 EPDMS, EC=58.9); added caveat on anchor-based 90.2
- `wiki/concepts/diffusion-planner.md` — added action decoder scaling reversal section (FM vs. AR vs. query-based at 103k vs. 70M frames)
- `wiki/index.md` — added DriveVLA-W0 row

**Key concepts**:
- Supervision deficit: sparse action signal wastes VLA capacity; future image prediction as dense self-supervision
- AR world model (VQ/Emu3): predicts current frame tokens; diffusion WM (ViT/Qwen2.5-VL): predicts future frame $I_{t+1}$
- World modeling unlocks cross-dataset generalization; action-only VLAs overfit and degrade (VLA-VQ: −9.5%)
- 70M-frame scaling: WM adds +28.8% ADE (VQ), +15.9% collision (ViT) vs. action-only at scale
- Action decoder reversal: FM > AR at 103k frames; AR > FM at 70M frames
- 90.2 PDMS uses trajectory anchors (not single-sample); single-sample query-based = 88.4 PDMS
- EC = 58.9 on NAVSIM-v2 (lowest in wiki)

---

## 2026-04-07 — Ingest: DriveDreamer-Policy

**Source**: `raw/papers/DriveDreamer-Policy_ A Geometry-Grounded World–Action Model for Unified Generation and Planning.md`  
**arXiv**: 2604.01765v1  
**Org**: GigaAI + University of Toronto + CUHK MMLab

**Pages created**:
- `wiki/sources/drivedreamer-policy.md`

**Pages updated**:
- `wiki/concepts/world-model-for-ad.md` — added Pattern 6: geometry-grounded causal WAM (depth→video→action); added NAVSIM FVD table; updated Open Questions
- `wiki/concepts/navsim-benchmark.md` — added DDP (89.2 PDMS, 88.7 EPDMS new SOTA); updated NAVSIM-v2 SOTA table with DDP; added caveat on comparison scope and ReCogDrive IL-only baseline issue
- `wiki/index.md` — added DriveDreamer-Policy row

**Key concepts**:
- Causal depth→video→action ordering: single LLM forward pass, no iterative cross-branch refinement
- Depth as geometric scaffold: reduces FVD −18.6% for video; +0.5 PDMS for planning alone
- Depth+video combined: +1.2 PDMS over action-only baseline (88.0→89.2)
- NAVSIM-v2 88.7 EPDMS (EC=79.4): new apparent SOTA, surpasses Senna-2 (86.6) by +2.1
- No RL; single-stage multi-task training; pseudo-label depth from DA3
- Comparison gaps: excludes DriveFine (90.7), WAM-Flow (90.3), uses IL-only ReCogDrive (86.5)

---

## 2026-04-07 — Ingest: FutureSightDrive

**Source**: `raw/papers/FutureSightDrive_ Thinking Visually with Spatio-Temporal CoT for Autonomous Driving.md`  
**arXiv**: 2505.17685v3  
**Org**: Xi'an Jiaotong University + Amap (Alibaba Group)

**Pages created**:
- `wiki/sources/futuresightdrive.md`

**Pages updated**:
- `wiki/concepts/world-model-for-ad.md` — added Pattern 5: Visual CoT as Planning Intermediate (FSDrive); updated nuScenes FID and planning SOTA tables; updated Open Questions
- `wiki/concepts/vlm-domain-adaptation.md` — added FSDrive section (vocabulary expansion, visual CoT modality, modality gap ablation); updated CoT design space table (8 rows); updated strategy comparison table (11 rows)
- `wiki/concepts/navsim-benchmark.md` — added FSDrive (85.1 PDMS) to SOTA table; added caveat on comparison scope
- `wiki/index.md` — added FutureSightDrive row

**Key concepts**:
- Visual spatio-temporal CoT: generated unified future frame (red lane dividers + 3D boxes) as planning intermediate
- Dual-role VLA: world model (generates visual CoT) + inverse dynamics model (plans from current obs + visual CoT)
- Vocabulary expansion: VQ-VAE tokens appended to text vocabulary, no architectural change, ~0.3% of prior methods' data
- Progressive generation: lane dividers → 3D boxes → full frame enforces physical laws before appearance
- CoT ablation: visual ST-CoT reduces collision 31% vs. no CoT; text CoT only 8.6%
- 85.1 PDMS NAVSIM (pre-2025 comparisons only); 0.96m L2 nuScenes (no ego status, 2B); FID 10.1

---

## 2026-04-06 — Ingest: AdaThinkDrive

**Source**: `raw/papers/AdaThinkDrive_ Adaptive Thinking via Reinforcement Learning for Autonomous Driving.md`
**arXiv**: 2509.13769v1
**Orgs**: Xiaomi EV, Tsinghua University

**Pages created**:
- `wiki/sources/adathinkdrive.md` — full source summary with all figures and Tables I–VI (NAVSIM comparison, Think/NonThink SFT/RL comparison, inference time, per-level analysis, training ablation, reward ablation)

**Pages updated**:
- `wiki/concepts/rl-for-ad.md` — added AdaThinkDrive section: empirical CoT-hurts-simple finding, 4-component GRPO reward, Adaptive Think Reward Algorithm 1 (dynamic scene relabeling via rollout comparison, T=0.9 threshold), ablation results, AdaThinkDrive vs. AutoVLA comparison table; updated overall reward comparison table to 9-method version; removed stale statement that AutoVLA "is the only" efficiency approach; updated sources/related frontmatter
- `wiki/concepts/vlm-domain-adaptation.md` — added AdaThinkDrive section: scene complexity categorization (3 levels, CIPO-1/2/Motion Interaction), dual-mode SFT (same-query Think+NonThink vs. AutoVLA's separate fast/slow), comparison table; updated strategy comparison table to 9 rows; updated sources/related frontmatter
- `wiki/concepts/navsim-benchmark.md` — added AdaThinkDrive (90.3) and BoN-4 (93.0) to SOTA table; added AdaThinkDrive caveat (no WAM-Flow/Curious-VLA head-to-head, Hydra-NeXt reference baseline is non-VLM); updated sources/related frontmatter

**Index updated**: yes

**Key findings**:
- Empirical proof that CoT hurts in simple scenarios (first paper in wiki to establish this rigorously with 3-level complexity analysis)
- Adaptive Think Reward achieves +2.0 vs. Never-Think RL and +1.4 vs. Always-Think RL — adaptive beats both fixed modes on all levels
- 84% Non-Think in simple scenes, 96% Think in challenging scenes — clean behavioral confirmation
- Three papers (WAM-Flow, Curious-VLA, AdaThinkDrive) independently claim 90.3 PDMS on NAVSIM-v1 with no head-to-head comparison; DriveFine (90.7) remains the single-sample SOTA

---

## 2026-04-05 — Ingest: Alpamayo-R1

**Source**: `raw/papers/Alpamayo-R1_ Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail.md`
**arXiv**: 2511.00088v1
**Org**: NVIDIA

**Pages created**:
- `wiki/sources/alpamayo-r1.md` — full source summary with all figures and Tables 6–13 (open-loop CoC ablation nominal+challenging, closed-loop AlpaSim, RL ablation, LingoQA backbone comparison, FM vs. AR decoding, vision encoding comparison, inference latency breakdown)

**Pages updated**:
- `wiki/concepts/rl-for-ad.md` — added Alpamayo-R1 section: LRM-as-critic (generation-verification gap rationale), binary CoC-action consistency reward, trajectory quality reward (L2+collision+jerk), critical Table 9 finding (reasoning-only RL hurts ADE+consistency), Boltzmann RL data curation, GRPO formulation, full 8-method reward comparison table; updated sources/related frontmatter
- `wiki/concepts/vlm-domain-adaptation.md` — added Alpamayo-R1 section: Cosmos-Reason Physical AI backbone (LingoQA comparison, complementarity with AutoMoT finding), CoC dataset (3 desiderata, hybrid 2-stage human + GPT-5 labeling, +132.8% causal score), CoT comparison table with causal locality and decision grounding flags, updated 8-row strategy comparison table; updated sources/related frontmatter
- `wiki/concepts/diffusion-planner.md` — added unicycle dynamics control representation, FM action expert formulation (OT path, Euler integration), dual representation rationale (discrete training for GRPO + FM for inference), FM vs. AR decoding Table 11 (97% vs. 44% comfort), UniUGP FM comparison; added AR1 row to design space table; updated sources/related frontmatter

**Index updated**: yes

**Key findings**:
- Reasoning-only RL hurts action quality (ADE 2.12→2.19m) — consistency reward is essential for grounding reasoning to executable behavior
- Cosmos-Reason Physical AI pre-training (+6.4% LingoQA vs. Qwen2.5-VL-7B) without catastrophic forgetting — supports domain-aligned pre-training over general VLM fine-tuning
- Flow matching dominates AR trajectory decoding on comfort (97% vs. 44%) and closed-loop safety (1.27 vs. 0.59 AlpaSim at-fault)
- No NAVSIM/nuScenes/Bench2Drive results — internal NVIDIA dataset only; direct comparison with wiki peers is not possible

---

## 2026-04-05 — Ingest: AutoDrive-R²

**Source**: `raw/papers/AutoDrive-R²_ Incentivizing Reasoning and Self-Reflection Capacity for VLA Model in Autonomous Driving.md`
**arXiv**: https://arxiv.org/html/2509.01944v1
**Assets read**: x2 11.png (pipeline overview), x3 12.png (qualitative comparison)

**Pages created**:
- `wiki/sources/autodrive-r2.md` — full source summary with all figures and 4 tables (nuScenes L2, Waymo zero-shot L2, ablation study, group size ablation)

**Pages updated**:
- `wiki/concepts/rl-for-ad.md` — added AutoDrive-R² physics-grounded reward section (4-component MSE formulas + ablation), SFT cold-start necessity empirical confirmation, GT-based GRPO comparison table; updated sources/related frontmatter
- `wiki/concepts/vlm-domain-adaptation.md` — added AutoDrive-R² self-reflection CoT section (nuScenesR²-6K dataset, 4-step chain, self-reflection backward-check, "aha moment", CoT comparison table); updated sources/related frontmatter

**Index updated**: yes

**Note**: AutoDrive-R² 7B achieves 0.19m avg L2 on nuScenes with only 6K training samples — substantially better than EMMA+ (0.29m, ~103K). The self-reflection step (4th CoT stage: backward-checking physical feasibility before emitting answer) is unique across all wiki papers. The RL-only ablation (0.33m vs. SFT-only 0.27m vs. full 0.19m) independently confirms the Curious-VLA finding that SFT cold-start quality is prerequisite for effective RL. The physics reward ablation shows spatial alignment is indispensable (removal → 0.53m near-collapse); temporal smoothness is the second most critical component.

---

## 2026-04-05 — Ingest: AutoMoT

**Source**: `raw/papers/AutoMoT_ A Unified Vision-Language-Action Model with Asynchronous Mixture-of-Transformers for End-to-End Autonomous Driving.md`
**arXiv**: https://arxiv.org/html/2603.14851v1
**Venue**: ICML
**Assets read**: x1 13.png (four-paradigm comparison), x2 10.png (architecture overview), x3 11.png (attention pattern visualization)

**Pages created**:
- `wiki/sources/automot.md` — full source summary with all figures and 8 tables (Bench2Drive, nuScenes open-loop, reasoning benchmarks, VLM boundary ablation, sync vs. async planning, sync vs. async decision, component ablation, Senna decision benchmark)

**Pages updated**:
- `wiki/concepts/dual-system-vla.md` — added AutoMoT layer-wise KV cache async pattern; updated comparison table to include AutoMoT vs. Senna/Senna-2/ReCogDrive; updated sources/related frontmatter
- `wiki/concepts/vlm-domain-adaptation.md` — added AutoMoT frozen VLM empirical evidence section (catastrophic forgetting table); added AutoMoT to strategy comparison table; updated sources/related frontmatter

**Index updated**: yes

**Note**: AutoMoT's primary contribution is the **catastrophic forgetting finding** — AD fine-tuning of VLMs gives marginal scene understanding gain (+0.2 LingoQA) while destroying general reasoning (TallyQA −35%, InfoVQA −44%). This is the only paper in the wiki with systematic evidence against VLM fine-tuning. Bench2Drive 87.34 DS is best among VLM-augmented methods in its comparison table, but LinkVLA (91.01) is absent and likely supersedes it. No NAVSIM results — cannot compare with recent PDMS leaders.

---

## 2026-04-05 — Ingest: AutoVLA

**Source**: `raw/papers/AutoVLA_ A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning.md`
**arXiv**: https://arxiv.org/html/2506.13757v1
**Assets read**: x1 12.png (overview), x2 9.png (4-paradigm comparison), x3 10.png (training pipeline), x4 9.png (data scaling), x5 9.png (RFT results), x6 8.png (Waymo E2E), x7 8.png (action codebook), x8 5.png (reasoning annotation pipeline), x9 1.png (Waymo reasoning examples), x10 1.png (nuPlan reasoning examples), x11 2.png (system prompt)

**Pages created**:
- `wiki/sources/autovla.md` — full source summary with all 11 figures and 3 tables (NAVSIM, Bench2Drive, action tokenization ablation)

**Pages updated**:
- `wiki/concepts/rl-for-ad.md` — added AutoVLA adaptive reasoning via CoT length penalty section; reward table comparing all GRPO reward designs in wiki; updated sources/related frontmatter
- `wiki/concepts/navsim-benchmark.md` — added AutoVLA 89.11 PDMS (post-RFT) and 92.12 BoN to SOTA table; added comparison-scope caveat; updated sources/related frontmatter
- `wiki/concepts/diffusion-planner.md` — added AR over physical codebook (AutoVLA) as 10th paradigm in design space table; updated sources/related frontmatter
- `wiki/concepts/vlm-domain-adaptation.md` — added AutoVLA dual-mode SFT section (data scaling finding, GT-hint annotation, adaptive reasoning via RFT); updated sources/related frontmatter

**Index updated**: yes

**Note**: AutoVLA post-RFT 89.11 is below current SOTA (DriveFine 90.7, Curious-VLA 90.3). The primary contribution is not SOTA performance but the CoT length penalty mechanism — the only approach in the wiki that explicitly optimizes for reasoning *efficiency* rather than just quality. The physical action codebook ablation (59.24 → 80.54 PDMS for text waypoint vs. physical tokens) is a strong argument against text waypoint representations. The data scaling finding (CoT < action-only at < 50k samples) is a useful calibration for choosing when CoT training is worth the cost.

---

## 2026-04-05 — Ingest: Curious-VLA

**Source**: `raw/papers/Devil is in Narrow Policy_ Unleashing Exploration in Driving VLA Models.md`
**arXiv**: https://arxiv.org/html/2603.06049
**Assets read**: x1 11.png (behavioral diagnostics quantitative), x3 9.png (overall pipeline), x4 8.png (horizon scale mismatch visualization), x5 8.png (qualitative BEV comparison)

**Pages created**:
- `wiki/sources/curious-vla.md` — full source summary with all 4 figures and 6 tables (NAVSIM v1, NAVSIM v2, nuScenes, behavioral diagnostics, FTE ablation, RL ablation)

**Pages updated**:
- `wiki/concepts/rl-for-ad.md` — added Narrow Policy analysis (3 root causes, advantage collapse formula), Behavioral Diagnostics framework, FTE (DE+CoT+SN), ADAS (Bernoulli filter), SDR (focal-loss reward); updated sources/related frontmatter
- `wiki/concepts/navsim-benchmark.md` — added Curious-VLA 90.3 PDMS (v1) and 94.8 BoN-6 to SOTA table; added 85.3 EPDMS (v2) with comparison-scope caveat; updated sources/related frontmatter

**Index updated**: yes

**Note**: Curious-VLA (90.3, 1xC, 3B) ties AdaThinkDrive (8B) and is slightly below DriveFine (90.7, 1xC). DriveFine remains single-sample SOTA. The BoN-6 result of 94.8 is the most significant finding: it matches human GT (94.8) and validates that FTE+DARL successfully unlocks exploration potential. Critical negative findings: (1) difficulty-based RL sampling causes training collapse (35.2 PDMS) — not hard scenarios, but diverse-outcome scenarios are needed for GRPO; (2) DE alone without SN hurts performance (85.2 < 85.6 baseline) — diversity expansion must be paired with step-wise normalization to be effective.

---

## 2026-04-05 — Ingest: DriveFine

**Source**: `raw/papers/DriveFine_ Refining-Augmented Masked Diffusion VLA for Precise and Robust Driving.md`
**arXiv**: https://arxiv.org/html/2602.14577v1
**Assets read**: x1 10.png (decoding comparison), x2 8.png (RFT reward hacking finding), x3 8.png (irreversible decoding failures), x4 7.png (architecture overview), x5 7.png (hybrid RL pipeline), x6 6.png (before/after refinement), x7 6.png (PDMS-latency trade-off)

**Pages created**:
- `wiki/sources/drivefine.md` — full source summary with all 7 figures and 7 tables (NAVSIM v1 PDMS, v2 EPDMS, Navhard EPDMS, component ablation, PDMS/EPDMS robustness, refinement block count, group size sensitivity)

**Pages updated**:
- `wiki/concepts/navsim-benchmark.md` — added DriveFine 90.7/91.8 PDMS (v1) and 89.7 EPDMS (v2, bug-fixed); added Navhard benchmark section; updated SOTA summary; updated sources/related frontmatter
- `wiki/concepts/rl-for-ad.md` — added DriveFine reward-hacking finding (diffusion planners lose EPDMS under PDMS GRPO); added hybrid offline+online RL for refinement expert; updated sources/related frontmatter
- `wiki/concepts/diffusion-planner.md` — added block-MoE refinement section; DriveFine vs. ReflectDrive comparison table; updated design space table (9th paradigm); updated sources/related frontmatter

**Index updated**: yes

**Note**: DriveFine (90.7, 1xC) is now the broadly-verified single-camera NAVSIM-v1 SOTA, surpassing WAM-Flow (90.3). DriveFine* (91.8) requires an additional trained scorer. NAVSIM-v2 89.7 EPDMS uses a bug-fixed scorer not comparable to prior results. The reward-hacking finding (diffusion planners degrade EPDMS under PDMS GRPO while token-based VLAs do not) is a practically important negative result for the field.

---

## 2026-04-05 — Ingest: HERMES

**Source**: `raw/papers/HERMES_ A Holistic End-to-End Risk-Aware Multimodal Embodied System with Vision–Language Models for Long-Tail Autonomous Driving.md`
**arXiv**: https://arxiv.org/html/2602.00993v1
**Assets read**: x1 9.png (paradigm comparison), x2 7.png (architecture overview), x3 7.png (Intent Modulator), x4 6.png (Risk Planning Cross-Attention), x5 6.png (nighttime rain qualitative), x6 5.png (low-visibility residential qualitative), x7 5.png (construction zone qualitative), x8 4.png (urban intersection qualitative)

**Pages created**:
- `wiki/sources/hermes.md` — full source summary with all 8 figures and 4 tables (overall performance, category-wise RFS, ablation, prompt design)

**Pages updated**:
- `wiki/concepts/vlm-domain-adaptation.md` — added HERMES offline VLM annotation / teacher-student distillation section; comparison table of VLM adaptation strategies; long-tail as an adaptation axis

**Index updated**: yes

**Note**: HERMES is the only paper in the wiki targeting WOD-E2E (Waymo real-world open-loop). Not comparable to NAVSIM or Bench2Drive papers. The offline-annotator-only pattern (VLM never runs at inference) is new to the wiki. Caveat: baseline fairness is questionable — LightEMMA is zero-shot, HERMES trains end-to-end on the full training split. Historical motion state is the most critical component (−1.30 RFS without it), outweighing semantic embeddings (−0.60 RFS).

---

## 2026-04-05 — Ingest: LinkVLA

**Source**: `raw/papers/Unifying Language-Action Understanding and Generation for Autonomous Driving.md`
**arXiv**: https://arxiv.org/html/2603.01441v1
**Assets read**: x1 8.png (latency-performance overview), x3 6.png (architecture), x4 5.png (bidirectional objective), x5 5.png (qualitative instruction following), x6 4.png (uniform vs. log grid), x7 4.png (additional qualitative)

**Pages created**:
- `wiki/sources/linkvla.md` — full source summary with all 7 figures and 9 tables (Bench2Drive, latency, instruction following, DriveLM VQA/commentary, closed-loop ablation, soft-label, navigation modality, codebook size, σ ablation)

**Pages updated**:
- `wiki/concepts/diffusion-planner.md` — added shared codebook C2F paradigm (8th in design space table); full section on unified token space, bidirectional objective, and C2F decoder; updated sources/related frontmatter

**Index updated**: yes

**Note**: LinkVLA sets new Bench2Drive SOTA (91.01 DS, 74.55% SR), surpassing ORION (77.74) and SimLingo (85.07). Evaluated only on Bench2Drive/CARLA — no NAVSIM comparison possible. CoT latency is excluded from the reported 48ms. The bidirectional action-captioning objective is the most principled alignment contribution: no new data required, works by enriching shared token embeddings through the inverse task.

---

## 2026-04-05 — Ingest: ORION

**Source**: `raw/papers/ORION_ A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation.md`
**arXiv**: https://arxiv.org/html/2503.19755v1
**Assets read**: x1 7.png (4-paradigm comparison), x2 6.png (full ORION pipeline), x3 5.png (QT-Former architecture), x4 4.png (qualitative results), x5 4.png (paradigm ablation bar chart), x6 3.png (Chat-B2D annotation pipeline)

**Pages created**:
- `wiki/sources/orion.md` — full source summary with all 6 figures and 7 tables (Bench2Drive closed-loop, Multi-Ability, VAE vs. diffusion, QT-Former ablation, history query ablation, VQA+planning joint training, nuScenes open-loop)

**Pages updated**:
- `wiki/concepts/diffusion-planner.md` — added ORION VAE-based reasoning-action alignment section; updated design space table to include VAE+GRU as 6th paradigm; updated sources/related frontmatter

**Index updated**: yes

**Note**: ORION is evaluated on Bench2Drive (CARLA V2) only — not NAVSIM. nuScenes open-loop 0.34 avg L2 is competitive but below Senna (0.22). VAE clearly outperforms diffusion as the generative planner (77.74 vs. 71.97 DS). Weakness: Merging (25%) and Give Way (30%) Multi-Ability scores — lane-changing decisions remain hard for VLM causal reasoning.

---

## 2026-04-05 — Ingest: Senna-2

**Source**: `raw/papers/Senna-2_ Aligning VLM and End-to-End Driving Policy for Consistent Decision Making and Planning.md`
**arXiv**: https://arxiv.org/abs/2603.11219
**Assets read**: x1 3.png (consistency gap motivation), x2 2.png (architecture), x3 2.png (three-stage training recipe), x4 2.png (speed control qualitative), x5 2.png (collision scenario qualitative), x6 1.png (training curves), x7 1.png (additional qualitative)

**Pages created**:
- `wiki/sources/senna2.md` — full source summary with all figures and tables
- `wiki/concepts/dual-system-vla.md` — dual-system VLM + E2E architecture pattern; decision adapter; kinematic mapping; consistency alignment methods; HRL contrast

**Pages updated**:
- `wiki/concepts/rl-for-ad.md` — added Senna-2 HRL section: 3DGS environments, bottom-up hierarchical RL, longitudinal penalties, contrast with NAVSIM GRPO
- `wiki/concepts/vlm-domain-adaptation.md` — added Senna-2's consistency-oriented adaptation, kinematic mapping, selective open-loop alignment
- `wiki/concepts/navsim-benchmark.md` — updated NAVSIM-v2 SOTA table; Senna-2 now leads at 86.6 EPDMS

**Index updated**: yes

---

## 2026-04-05 — Ingest: UniUGP

**Source**: `raw/papers/UniUGP_ Unifying Understanding, Generation, and Planing For End-to-end Autonomous Driving.md`
**arXiv**: https://arxiv.org/abs/2512.09864
**Assets read**: x1 2.png (architecture), x2 1.png (data pipeline), x3 1.png (world model ablation), x4 1.png (trajectory-controllable generation), x5 1.png (long-tail QA), x8 1.png (CoT reasoning examples)

**Pages created**:
- `wiki/sources/uniugp.md` — full source summary
- `wiki/concepts/world-model-for-ad.md` — world model integration with VLA planning; FID/FVD metrics; coupling patterns

**Pages updated**:
- `wiki/concepts/diffusion-planner.md` — added UniUGP's continuous FM planning + MoT architecture; world model co-training signal
- `wiki/concepts/vlm-domain-adaptation.md` — added UniUGP's staged training, CoT integration, instruction following via data, long-tail data approach

**Index updated**: yes

---

## 2026-04-05 — Ingest: WAM-Flow

**Source**: `raw/papers/WAM-Flow_ Parallel Coarse-to-Fine Motion Planning via Discrete Flow Matching for Autonomous Driving.md`
**arXiv**: https://arxiv.org/abs/2512.06112

**Pages created**:
- `wiki/sources/wam-flow.md` — full source summary
- `wiki/concepts/discrete-flow-matching.md` — DFM theory, CTMC dynamics, geometry-aware Gibbs paths, metric-aligned numerical tokenizer

**Pages updated**:
- `wiki/concepts/diffusion-planner.md` — added DFM vs. diffusion comparison table
- `wiki/concepts/rl-for-ad.md` — added WAM-Flow GRPO section; contrasts DFM-GRPO with diffusion-chain MDP approach
- `wiki/concepts/navsim-benchmark.md` — updated SOTA tables (v1 + v2); WAM-Flow now leads both

**Index updated**: yes

---

## 2026-04-05 — Ingest: Percept-WAM

**Source**: `raw/papers/Percept-WAM_ Perception-Enhanced World-Awareness-Action Model for Robust End-to-End Autonomous Driving.md`
**arXiv**: https://arxiv.org/html/2511.19221v1
**Assets read**: x1 6.png (motivation), x2 5.png (architecture), x3 4.png (IoU confidence dataset), x5 3.png (grid query tokens), x6 2.png (trajectory decoder), x7 2.png (confidence calibration scatter), x10.png (PV perception qualitative), x11 1.png (BEV perception qualitative), x12.png (trajectory planning qualitative)

**Pages created**:
- `wiki/sources/percept-wam.md` — full source summary with all 9 figures and 7 tables (PV perception, BEV perception, trajectory planning, IoU confidence ablation, BEV ablation, decoding efficiency, dataset tasks)
- `wiki/concepts/perception-for-planning.md` — new concept: World-PV/BEV tokens; grid-conditioned parallel AR; IoU-aware confidence calibration; four-query modality-aligned decoder; comparison of perception integration approaches

**Pages updated**:
- `wiki/concepts/diffusion-planner.md` — added Percept-WAM's four-query MLP decoder section; attention-masked modality alignment; reuse of perception tokens
- `wiki/concepts/navsim-benchmark.md` — added Percept-WAM\* 90.2 PDMS with caveat (LiDAR-assisted, limited comparison scope)

**Index updated**: yes

---

## 2026-04-05 — Ingest: Reasoning-VLA

**Source**: `raw/papers/Reasoning-VLA_ A Fast and General Vision-Language-Action Reasoning Model for Autonomous Driving.md`
**arXiv**: https://arxiv.org/html/2511.19912v1
**Assets read**: x1 5.png (framework + training pipeline), x2 4.png (action module cross-attention), 8data.png (dataset distribution), x3 3.png (qualitative across 8 datasets), x4 3.png (dataset construction pipeline)

**Pages created**:
- `wiki/sources/reasoning-vla.md` — full source summary with all 5 figures and 9 tables (nuScenes, NeuroNCAP, NAVSIM, generalized, zero-shot, ablation, efficiency, unified/nuScenes comparison, closed-loop comparison)

**Pages updated**:
- `wiki/concepts/rl-for-ad.md` — added Reasoning-VLA GT-based GRPO section; physics reward table; contrast with NAVSIM GRPO (simulator vs. trajectory-only)
- `wiki/concepts/vlm-domain-adaptation.md` — added Reasoning-VLA unified 8-dataset corpus section; CoT pipeline; zero-shot generalization findings
- `wiki/concepts/diffusion-planner.md` — added learnable action queries paradigm section; design space comparison table (5 paradigms)
- `wiki/concepts/navsim-benchmark.md` — added Reasoning-VLA 91.7 PDMS claim with caveat (comparison scope limited to old baselines only)

**Index updated**: yes

**Note**: Reasoning-VLA claims 91.7 PDMS on NAVSIM but compares only against TransFuser/UniAD/Para-Drive (~84 PDMS). No head-to-head vs. WAM-Flow (90.3) or ReCogDrive (89.6). WAM-Flow remains the last verified SOTA.

---

## 2026-04-05 — Ingest: ReflectDrive

**Source**: `raw/papers/Discrete Diffusion for Reflective Vision-Language-Action Models in Autonomous Driving.md`
**arXiv**: https://arxiv.org/html/2509.20109v1
**Assets read**: x1 4.png (framework overview), x2 3.png (safety-guided regeneration pipeline), goodcase.png (DAC + TTC violation correction), easy_case.png (1-step easy cases), medium_case.png (1–3 step medium cases), hard_case.png (1–5 step hard cases)

**Pages created**:
- `wiki/sources/reflectdrive.md` — full source summary with all figures; architecture, two-phase reflective inference, three model variants, limitations
- `wiki/concepts/inference-time-safety.md` — gradient-free inference-time safety; taxonomy vs. diffusion guidance/RL/anchors; scoring functions; inpainting-as-repair mechanism; limitations

**Pages updated**:
- `wiki/concepts/diffusion-planner.md` — added ReflectDrive masked discrete diffusion section; 3-way comparison table (continuous diffusion / DFM / masked diffusion)
- `wiki/concepts/discrete-flow-matching.md` — added DFM vs. masked discrete diffusion distinction table; clarified WAM-Flow (CTMC) vs. ReflectDrive (BERT-style) are distinct paradigms despite both being "discrete diffusion"
- `wiki/concepts/navsim-benchmark.md` — added ReflectDrive to v1 SOTA table (>89.1 claimed, exact number missing from markdown)

**Index updated**: yes

**Note**: Table 1 (NAVSIM closed-loop results) was not rendered in the paper's markdown conversion. Exact NC/DAC/TTC/Comf/EP sub-metrics unavailable.

---

## 2026-04-05 — Ingest: ReCogDrive

**Source**: `raw/papers/ReCogDrive_ A Reinforced Cognitive Framework for End-to-End Autonomous Driving.md`
**arXiv**: https://arxiv.org/html/2506.08052v1

**Pages created**:
- `wiki/sources/recogdrive.md` — full source summary
- `wiki/concepts/diffusion-planner.md` — diffusion-based trajectory planning
- `wiki/concepts/rl-for-ad.md` — RL for autonomous driving (incl. GRPO/diffusion RL)
- `wiki/concepts/vlm-domain-adaptation.md` — VLM adaptation for driving domain
- `wiki/concepts/navsim-benchmark.md` — NAVSIM benchmark and PDMS metric

**Index updated**: yes

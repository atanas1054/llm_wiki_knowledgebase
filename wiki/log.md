# Activity Log

Append-only log of all wiki operations.

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

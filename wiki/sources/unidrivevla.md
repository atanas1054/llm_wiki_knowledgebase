---
title: "UniDriveVLA: Unifying Understanding, Perception, and Action Planning for Autonomous Driving"
type: source-summary
sources: [raw/papers/UniDriveVLA_ Unifying Understanding, Perception, and Action Planning for Autonomous Driving.md]
related: [concepts/dual-system-vla.md, concepts/perception-for-planning.md, concepts/vlm-domain-adaptation.md, concepts/diffusion-planner.md, sources/orion.md, sources/automot.md, sources/percept-wam.md]
created: 2026-04-07
updated: 2026-04-07
confidence: high
---

**arXiv**: 2604.02190v1 (2026-04-03)  
**Org**: HUST + Xiaomi EV + University of Macau  
**Benchmark highlight**: 78.37 DS Bench2Drive (best w/o PDM-Lite); 0.51m avg L2 nuScenes no-ego (Large)

---

## Problem: Perception–Reasoning Conflict in Shared-Weight VLAs

Existing VLA approaches face a fundamental dilemma: 2D VLMs preserve native semantic reasoning but have limited spatial perception; adding 3D spatial representations to shared-weight decoders degrades reasoning. The paper measures this conflict via cosine similarity between LLM tokens and perception tokens — in a shared-weight decoder the similarity progressively increases toward 1, indicating feature collapse into nearly identical representations.

The root cause: **jointly optimizing spatial perception and semantic reasoning within shared model parameters introduces representation interference**.

![Fig 1: VLA paradigm comparison](../../raw/assets/x2%2015.png)

*Figure 1: (a) Vanilla 2D VLA — strong reasoning, limited spatial perception. (b) 3D-enhanced VLA — better perception but degraded reasoning. (c) UniDriveVLA — MoT decouples understanding, perception, and action.*

---

## Architecture: Mixture-of-Transformers (MoT) with Masked Joint Attention

**Backbone**: Qwen3-VL (SigLIP-2 vision encoder + MLP merger + Qwen3 LM)  
- Base: 2B parameters; Large: 8B parameters  
- Input: 6-view cameras at 960×544

### Three Expert Streams

| Expert | Token type | Role |
|--------|-----------|------|
| Understanding (und) | Language/text tokens | Semantic reasoning, scene understanding |
| Perception (per) | Sparse perception query tokens | 3D detection, HD map, ego, motion, occupancy |
| Action (act) | Continuous trajectory tokens | Flow-matching trajectory generation |

Each expert has its own projection, FFN, and normalization layers. The joint objective:
$$\mathcal{L}_\text{total} = \lambda_1 \mathcal{L}_\text{ar} + \lambda_2 \mathcal{L}_\text{per} + \lambda_3 \mathcal{L}_\text{act}$$

### Masked Joint Attention

All tokens are concatenated in order [und; per; act] and attend globally with mask matrix **M**:

$$\mathbf{Z} = \text{Softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right)\mathbf{V}$$

Visibility pattern:
- **und** → causal self-attention only (does not see per or act tokens — preserves VLM pretraining)
- **per** → attends to und (acquires semantic context) + self
- **act** → attends to both und and per (aggregates semantic + spatial for planning)

![Fig 2: Cosine similarity analysis and performance comparison](../../raw/assets/x3%2016.png)

*Figure 2: (a) Cosine similarity between LLM tokens and perception tokens across layers. Shared-weight: similarity → 1 (collapse). MoT: low similarity maintained. (b) MoT outperforms shared-weight on all three axes.*

![Fig 3: Architecture overview](../../raw/assets/x5%2014.png)

*Figure 3: UniDriveVLA architecture. MoT with three experts; masked joint attention routes information asymmetrically across experts; sparse perception queries aggregate from 2D VLM features.*

![Fig 4: Masked Joint Attention illustration](../../raw/assets/x6%2012.png)

*Figure 4: Masked Joint Attention. und is causally masked from per/act; per reads und; act reads both.*

---

## Sparse Perception Paradigm

Unlike BEV-encoder approaches (OmniDrive, OpenDriveVLA), UniDriveVLA constructs sparse spatial perception directly from multi-scale 2D visual features:

- **Task-specific sparse queries** initialized from K-Means instance banks (dataset-level clustering)
- Queries updated through: temporal interaction → intra-task reasoning → inter-task communication → deformable feature aggregation → task-wise refinement
- **Five tasks in one unified sparse decoder**: 3D detection, HD map prediction, ego-status estimation, motion forecasting, occupancy (auxiliary latent branch)

First-pass perception outputs are projected into VLM hidden space and interact with understanding and action branches via masked joint attention, enabling semantic enrichment of sparse perception features. The enriched features are then projected back and refined by a second decoder pass — a two-pass design.

---

## Three-Stage Progressive Training

| Stage | What trains | Key details |
|-------|-------------|-------------|
| **Stage 1** — VLM anchor | Full VLM fine-tune | 3 epochs; lr=4e-5; 3:7 driving-to-general ratio (FineVision); preserves semantic foundations |
| **Stage 2** — Joint optimization | VLM (LoRA) + Perception + Action | 30 epochs; AdamW; base lr=2e-4; VLM LR multiplier 0.5× (effective 1e-4); EMA; jointly trains AR + perception + FM trajectory |
| **Stage 3** — Expert specialization | Perception + Action experts only | 15 epochs; VLM frozen; lr=1e-4; adds motion forecasting objective; EMA |

LoRA in Stage 2 limits VLM parameter drift while enabling adaptation. VLM freeze in Stage 3 fully protects semantic reasoning during final specialization.

---

## Results

### Table 1: Bench2Drive Closed-Loop Planning

| Method | Avg L2↓ | DS↑ | SR(%)↑ | Efficiency↑ | Comfort↑ |
|--------|---------|-----|-------|------------|---------|
| AutoVLA† (PDM-Lite) | — | 78.84 | 57.73 | 146.93 | 39.33 |
| SimLingo† (PDM-Lite) | — | 85.94 | 66.82 | 244.18 | 25.49 |
| R2SE† (PDM-Lite) | — | 86.28 | 69.54 | 243.89 | 23.26 |
| DriveMOE | 0.38 | 74.22 | 48.64 | 175.96 | 15.31 |
| Orion | 0.68 | 77.74 | 54.62 | 151.48 | 17.38 |
| **UniDriveVLA** | **0.72** | **78.37** | **51.82** | **198.86** | **11.78** |

Bold = best without PDM-Lite. UniDriveVLA has the best Driving Score and Efficiency in the non-PDM-Lite category, but the lowest Comfort (11.78 vs. Orion 17.38) and does not exceed Orion on Success Rate (51.82 vs. 54.62).

### Table 2: Bench2Drive Multi-Ability Scores (%)

| Method | Merging | Overtaking | Emerg. Brake | Give Way | Traffic Sign | Mean |
|--------|---------|-----------|-------------|---------|-------------|------|
| Orion | 25.00 | 71.11 | 78.33 | 33.00 | 69.15 | 54.72 |
| **UniDriveVLA** | **38.75** | **80.00** | 50.00 | 30.00 | 58.95 | 51.53 |

UniDriveVLA is best on Merging and Overtaking but trails Orion substantially on Emergency Brake (50.00 vs. 78.33) and mean score (51.53 vs. 54.72).

### Table 3: nuScenes Planning (L2 avg / Collision avg)

**With ego status (*):**

| Method | ST-P3 Avg L2↓ | ST-P3 Avg CR↓ | UniAD Avg L2↓ | UniAD Avg CR↓ | LLM |
|--------|-------------|-------------|--------------|--------------|-----|
| FSDrive* | 0.28 | 0.10 | 0.45 | 0.16 | Qwen2-VL-3B |
| AutoVLA* | 0.48 | 0.13 | 0.86 | 0.35 | Qwen2.5-VL-3B |
| OpenDriveVLA* | 0.33 | 0.10 | 0.67 | 0.30 | Qwen2.5-VL-3B |
| UniDriveVLA-Base* | 0.43 | 0.10 | 0.77 | 0.23 | Qwen3-VL-2B |
| UniDriveVLA-Large* | 0.42 | 0.10 | 0.74 | 0.20 | Qwen3-VL-8B |

**Without ego status:**

| Method | ST-P3 Avg L2↓ | ST-P3 Avg CR↓ | UniAD Avg L2↓ | UniAD Avg CR↓ | LLM |
|--------|-------------|-------------|--------------|--------------|-----|
| VAD | 0.72 | 0.21 | — | — | — |
| UniAD | 0.73 | 0.61 | 1.03 | 0.77 | — |
| SparseDrive‡ | 0.55 | 0.08 | 0.99 | 0.21 | — |
| FSDrive | 0.53 | 0.17 | 0.96 | 0.40 | Qwen2-VL-3B |
| UniDriveVLA-Base | 0.54 | 0.17 | 0.96 | 0.41 | Qwen3-VL-2B |
| **UniDriveVLA-Large** | **0.51** | **0.11** | **0.90** | **0.27** | Qwen3-VL-8B |

UniDriveVLA-Large achieves the best avg L2 in the no-ego setting under both protocols. With ego, FSDrive (0.28 ST-P3) outperforms UniDriveVLA-Large (0.42).

### Table 4: nuScenes Perception (validation set)

| Method | Det mAP↑ | NDS↑ | Map AP_ped | Map AP_div | Map AP_bound | Map mAP↑ | Motion minADE↓ | Motion minFDE↓ |
|--------|---------|-----|-----------|----------|------------|---------|--------------|--------------|
| UniAD | 0.380 | 0.359 | — | — | — | — | 0.710 | 1.020 |
| VAD | 0.276 | 0.397 | 0.406 | 0.515 | 0.506 | 0.476 | — | — |
| SparseDrive | 0.418 | 0.525 | 0.499 | 0.570 | 0.584 | 0.551 | 0.600 | 0.960 |
| EgoFSD | 0.410 | 0.528 | 0.549 | 0.557 | 0.573 | 0.560 | — | — |
| HiP-AD | 0.424 | 0.535 | — | — | — | 0.571 | 0.610 | — |
| UniDriveVLA-Base | 0.397 | 0.434 | 0.462 | 0.556 | 0.543 | 0.520 | 1.396 | 2.289 |
| UniDriveVLA-Large | 0.407 | 0.460 | 0.491 | 0.556 | 0.557 | 0.535 | 1.264 | 2.121 |

UniDriveVLA is above VAD but behind SparseDrive/HiP-AD on detection and mapping. Motion forecasting (minADE 1.264) significantly trails SparseDrive (0.600).

### Table 5: Planning Ablation (nuScenes, no-ego, UniAD avg)

| Baseline | Ego | Det | Map | Occ | Motion | L2↓ | CR(%)↓ |
|----------|-----|-----|-----|-----|--------|-----|--------|
| ✓ | | | | | | 0.75 | 0.27 |
| ✓ | ✓ | | | | | 0.61 | 0.21 |
| ✓ | ✓ | ✓ | | | | 0.58 | 0.10 |
| ✓ | ✓ | ✓ | ✓ | | | 0.58 | 0.14 |
| ✓ | ✓ | ✓ | ✓ | ✓ | | **0.53** | 0.14 |
| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 0.54 | 0.17 |

Key findings: ego-state is the largest single gain (0.75→0.61). Detection is critical for collision reduction (0.21→0.10). Occupancy provides the best L2 (→0.53). Map adds no gain over detection alone. Motion slightly *hurts* both L2 and CR in the current setting.

### Table 6: DriveBench Understanding

| Method | Percep.↑ | Predict.↑ | Plan.↑ | Behav.↑ | Avg↑ |
|--------|---------|---------|-------|--------|-----|
| LLaVA-1.5 | 23.22 | 22.02 | 29.15 | 13.60 | 22.00 |
| GPT-4o | 35.37 | 51.30 | 75.75 | 45.40 | 51.96 |
| ReCogDrive† (pretrain only) | 64.95 | 49.34 | 70.20 | 42.36 | 56.71 |
| UniDriveVLA | 36.78 | 43.13 | 66.98 | 60.97 | 51.97 |

UniDriveVLA's strongest category is Behavior (60.97, above GPT-4o). Planning (66.98) is near GPT-4o (75.75). Perception (36.78) trails ReCogDrive pretrain (64.95) significantly — note ReCogDrive is pretrain-only without action training, providing a data-focused upper bound.

### Table 7: MoT Ablation — Shared-Weight vs. MoT

| Architecture | General VQA(%)↑ | DriveBench(%)↑ | Det NDS↑ | Map mAP↑ | L2(m)↓ | CR(%)↓ |
|---|---|---|---|---|---|---|
| Shared-Weight Decoder | 31.1 | 50.8 | 0.437 | 0.516 | 0.641 | 0.175 |
| **Mixture-of-Transformers** | **45.5** | **54.9** | **0.439** | **0.516** | **0.533** | **0.140** |
| **Δ** | **+14.4pp** | **+4.1pp** | **+0.002** | **+0.000** | **−0.108m** | **−0.035** |

MoT dramatically improves general VQA (+14.4pp) and meaningfully improves planning (−0.108m L2). Perception (NDS, map mAP) is essentially unchanged — the main benefit is resolving the reasoning vs. planning tradeoff, not improving raw perception accuracy.

### Table 8: General Multimodal Benchmarks

| Model | Params | MMStar↑ | MMMU↑ | RealWorldQA↑ | AI2D↑ | MME↑ | VLMsAreBlind↑ | ChartQA↑ |
|-------|--------|---------|-------|------------|-------|------|-------------|---------|
| Qwen2.5-VL | 7B | 63.7 | 51.1 | 69.0 | 82.5 | 2317 | 40.3 | 82.5 |
| Qwen3-VL | 8B | 63.0 | 52.8 | 69.0 | 83.2 | 2364 | 61.9 | 83.8 |
| InternVL3 | 8B | 69.1 | 54.8 | 67.8 | 83.9 | 2393 | 42.0 | 86.6 |
| GPT-5 Nano | — | 41.3 | 57.6 | 60.7 | 65.7 | — | 40.2 | 48.6 |
| **UniDriveVLA** | **8B** | **43.3** | **47.3** | **49.9** | **76.3** | **1876** | **26.6** | **76.3** |

UniDriveVLA's base Qwen3-VL has 63.0 MMStar and 52.8 MMMU. After driving adaptation UniDriveVLA has 43.3 MMStar (−19.7pp) and 47.3 MMMU (−5.5pp). Despite MoT reducing interference, driving adaptation still meaningfully degrades general multimodal reasoning.

---

## Limitations

1. **PDM-Lite caveat**: UniDriveVLA's 78.37 DS is best *without* PDM-Lite, but PDM-Lite methods (SimLingo 85.94, R2SE 86.28) substantially outperform it. The non-PDM-Lite comparison excludes LinkVLA (91.01 DS) which uses the standard Bench2Drive setup — the paper's comparison scope is narrow.

2. **Lowest Comfort in table** (11.78 vs. Orion 17.38, DriveMOE 15.31). High efficiency (198.86) paired with low comfort suggests aggressive/jerky trajectories not penalized by the DS metric. Root cause not analyzed.

3. **Success Rate below Orion** (51.82% vs. 54.62%): despite higher DS, task completion rate is worse. The efficiency-comfort tradeoff may cause mission failures in edge cases.

4. **Motion forecasting behind specialists**: minADE 1.264 vs. SparseDrive 0.600 — a 2.1× gap. The unified decoder does not match task-specific models on motion.

5. **Map ablation adds no gain over detection** (Table 5: both 0.58 L2). Unclear whether map queries add value for planning vs. just consuming parameters.

6. **General VQA degradation despite MoT** (Table 8): MMStar 43.3 vs. base 63.0 (−19.7pp). MoT reduces interference (Table 7: +14.4pp vs. shared-weight), but driving adaptation still significantly impairs general reasoning. The residual gap suggests MoT is necessary but not sufficient.

7. **No NAVSIM evaluation**: the paper only evaluates on Bench2Drive (closed-loop) and nuScenes (open-loop). No comparison against DriveFine, WAM-Flow, DriveVLA-W0, or any PDMS-based method.

8. **No RL fine-tuning**: all training is SFT-based. The paper does not apply GRPO or any RL stage — a gap given that most NAVSIM leaders use GRPO.

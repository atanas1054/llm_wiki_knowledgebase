---
title: "DriveDreamer-Policy: A Geometry-Grounded World–Action Model for Unified Generation and Planning"
type: source-summary
sources: [raw/papers/DriveDreamer-Policy_ A Geometry-Grounded World–Action Model for Unified Generation and Planning.md]
related: [concepts/world-model-for-ad.md, concepts/navsim-benchmark.md, concepts/diffusion-planner.md]
created: 2026-04-07
updated: 2026-04-07
confidence: high
---

**Paper**: DriveDreamer-Policy: A Geometry-Grounded World–Action Model for Unified Generation and Planning  
**Orgs**: GigaAI, University of Toronto, CUHK MMLab  
**arXiv**: 2604.01765v1

---

## Summary

DriveDreamer-Policy (DDP) extends the World-Action Model (WAM) paradigm by adding explicit **depth generation** as a 3D geometric scaffold before video imagination and trajectory planning. The core argument: existing WAMs model 2D appearance without geometric grounding, limiting their utility for occlusion reasoning, free-space estimation, and distance-to-collision cues. DDP introduces a causal **depth → video → action** attention ordering in a single forward pass: depth queries attend to scene + LLM context; video queries additionally attend to depth; action queries attend to depth and video. This structured flow provides geometry-aware world representations to all downstream generators without extra synchronization or iterative refinement.

---

## Architecture

![[x1 15.png|DriveDreamer-Policy vs. prior work taxonomy]]

Figure 1: Vision-based planners ignore future world; world models lack planners; world-action models combine both but operate on 2D appearance. DDP extends WAMs with explicit 3D depth generation.

![[x2 13.png|DriveDreamer-Policy pipeline overview]]

Figure 2: LLM processes multi-view images + language + action context + learnable query groups → world/action embeddings → three modular generators (depth, video, action).

**LLM backbone**: Qwen3-VL-2B

**Input**: multi-view RGB images + natural language instruction + current action (embedded via 2-layer MLP encoder)

**Learnable query groups**: 64 depth-query tokens, 64 video-query tokens, 8 action-query tokens — fixed-size bottleneck interface

**Causal attention ordering** (within single LLM forward pass):
$$\text{depth queries} \rightarrow \text{video queries} \rightarrow \text{action queries}$$

Video queries can attend to depth context; action queries can attend to both. No iterative cross-branch refinement.

**Action representation**: $(x, y, \cos\theta, \sin\theta)$ — continuous, avoids angular wrap-around, encourages smooth turn dynamics

### Three Generators (all flow matching)

**Depth Generator** (pixel-space diffusion transformer):
- Initialized from PPD (PixelPerfect Depth, NeurIPS'25)
- Conditioned on LLM world-depth embeddings via cross-attention
- Input: noisy depth + current RGB; predicts denoising update
- Ground-truth depth from **Depth Anything 3** (pseudo-labels, not real sensor depth)
- Generates in pixel space (low dimensionality, sharp boundaries)

**Video Generator** (Wan-2.1-T2V-1.3B, adapted for image-to-video):
- Conditioned on LLM world-video embeddings + CLIP visual condition (appearance identity)
- 9 future frames, 144×256 resolution
- Video queries causally receive depth context from LLM → geometry-aware video

**Action Generator** (standalone diffusion transformer):
- Conditioned on LLM action embeddings
- Action embeddings implicitly aggregate depth + video context via causal LLM attention
- Can run independently (planning-only mode) or with full generation

**Flow matching objective** (shared across generators):
$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{x_0, x_1, t}\left[\|v_\theta(x_t, t \mid c) - (x_1 - x_0)\|_2^2\right]$$
$$x_t = (1-t)x_0 + t x_1, \quad t \sim \mathcal{U}(0,1)$$

**Joint training loss**:
$$\mathcal{L} = 0.1\,\mathcal{L}_d + \mathcal{L}_v + \mathcal{L}_a$$

Depth loss down-weighted (0.1) — geometry is auxiliary to planning and video.

**Training**: single stage, 100K steps, batch=32, 8×NVIDIA H20 GPUs, AdamW lr=1e-5; navtrain only (no extra datasets or extra pre-training beyond initialized backbones)

---

## Results

### NAVSIM-v1 Planning (Table 1)

| Method | Sensors | NC ↑ | DAC ↑ | TTC ↑ | C ↑ | EP ↑ | PDMS ↑ |
|---|---|---|---|---|---|---|---|
| AutoVLA | 3C | 98.4 | 95.6 | 98.0 | 99.9 | 81.9 | 89.1 |
| **DriveDreamer-Policy** | **3C** | **98.4** | **97.1** | **95.1** | **100.0** | **83.5** | **89.2** |
| DriveVLA-W0 | 1C | 98.7 | 96.2 | 95.5 | 100.0 | 82.2 | 88.4 |
| WoTE | 3C+L | 98.5 | 96.8 | 94.4 | 99.9 | 81.9 | 88.3 |
| PWM | 1C | 98.6 | 95.9 | 95.4 | 100.0 | 81.8 | 88.1 |
| DiffusionDrive | 3C+L | 98.2 | 96.2 | 94.7 | 100.0 | 82.2 | 88.1 |
| Epona | 3C | 97.9 | 95.1 | 93.8 | 99.9 | 80.4 | 86.2 |
| ReCogDrive* (IL only) | 3C | 98.1 | 94.7 | 94.2 | 100.0 | 80.9 | 86.5 |
| FSDrive | 3C | 98.2 | 93.8 | 93.3 | 99.9 | 80.1 | 85.1 |

**Caveat**: Table 1 excludes DriveFine (90.7), WAM-Flow (90.3), AdaThinkDrive (90.3), Curious-VLA (90.3) — all above 89.2. The paper categorizes against "world-model-based methods" only and claims SOTA within that category. The ReCogDrive entry (86.5 ★) is the **IL-only** version; the RL-trained version achieves 89.6 PDMS and is not included.

### NAVSIM-v2 Planning (Table 2, EPDMS)

| Method | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | EPDMS |
|---|---|---|---|---|---|---|---|---|---|---|
| **DriveDreamer-Policy** | **98.4** | **97.1** | **99.5** | **99.9** | **87.9** | **97.7** | **97.6** | **98.3** | 79.4 | **88.7** |
| DriveVLA-W0 | 98.5 | 99.1 | 98.0 | 99.7 | 86.4 | 98.1 | 93.2 | 97.9 | 58.9 | 86.1 |
| DiffusionDrive | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | 84.5 |

88.7 EPDMS surpasses Senna-2 (86.6 EPDMS, prior wiki NAVSIM-v2 SOTA) by +2.1. **Caveat**: Table 2 only compares DriveVLA-W0 as VLA baseline; Senna-2, DriveFine, Curious-VLA not included. EC = 79.4 is notably low — geometry + video learning improves safety margins but not extended comfort.

### World Generation (Table 3)

**(a) Video generation (front-view, NAVSIM)**

| Method | LPIPS ↓ | PSNR ↑ | FVD ↓ |
|---|---|---|---|
| PWM | 0.23 | 21.57 | 85.95 |
| **DriveDreamer-Policy** | **0.20** | 21.05 | **53.59** |

−38% FVD improvement vs. PWM (the only other reported method; both single front-view for fair comparison).

**(b) Depth generation (NAVSIM)**

| Method | AbsRel ↓ | δ₁ ↑ | δ₂ ↑ | δ₃ ↑ |
|---|---|---|---|---|
| PPD (zero-shot) | 18.5 | 80.4 | 94.0 | 97.2 |
| PPD (fine-tuned) | 9.3 | 91.4 | 98.3 | 99.5 |
| **DriveDreamer-Policy** | **8.1** | **92.8** | **98.6** | **99.5** |

LLM conditioning improves over fine-tuned PPD — global semantic context resolves locally ambiguous regions.

---

## Ablations

### World Learning for Planning (Table 4)

| Depth | Video | NC | DAC | TTC | C | EP | PDMS |
|---|---|---|---|---|---|---|---|
| ✗ | ✗ | 98.0 | 96.3 | 94.4 | 100.0 | 82.5 | 88.0 |
| ✓ | ✗ | 98.1 | 96.7 | 94.9 | 100.0 | 82.8 | 88.5 |
| ✗ | ✓ | 98.1 | 97.0 | 95.0 | 100.0 | 83.1 | 88.9 |
| ✓ | ✓ | 98.4 | 97.1 | 95.1 | 100.0 | 83.5 | **89.2** |

**Key finding**: Depth (+0.5) and video (+0.9) provide complementary gains; combined (+1.2) exceeds sum (+1.4 expected, +1.2 observed). Each modality provides different safety cues: depth = free space / distance-to-collision; video = dynamic agent evolution.

**Baseline (no world learning: 88.0 PDMS)** is already strong — the LLM alone is competitive. World learning adds +1.2 on top of a strong VLA baseline.

### Depth for Video Generation (Table 5)

| Strategy | LPIPS ↓ | PSNR ↑ | FVD ↓ |
|---|---|---|---|
| Video only | 0.22 | 19.89 | 65.82 |
| **Depth + Video** | **0.20** | **21.05** | **53.59** |

Depth joint learning reduces FVD by −18.6%. Confirms depth provides an effective 3D scaffold for coherent future prediction.

### Query Budget (Table 6)

| Depth | Video | Action | AbsRel ↓ | FVD ↓ | PDMS ↑ |
|---|---|---|---|---|---|
| 32 | 32 | 4 | 9.7 | 57.97 | 88.9 |
| **64** | **64** | **8** | **8.1** | **53.59** | **89.2** |

More query tokens = higher capacity for all modalities. PDMS: +0.3; FVD: −4.4; AbsRel: −1.6.

---

## Qualitative Analysis

![[x3 14.png|Visualization of depth, video, and action outputs]]

Figure 3: Generated depth (truncated to 80m), future video frames, and predicted trajectories overlaid on BEV. Top: slows down appropriately (more conservative than human). Bottom: aligns with human trajectory.

![[x5 12.png|World learning ablation visualization]]

Figure 4: Columns = Action-Only, Depth-Action, Video-Action, Depth-Video-Action. Rows = three scenario types. World learning consistently produces safer, more human-aligned trajectories. Depth resolves occlusion/free-space; video adds temporal dynamics; combined eliminates residual errors.

---

## Limitations

1. **NAVSIM-v1 comparison gap**: excludes DriveFine (90.7), WAM-Flow (90.3), AdaThinkDrive (90.3), Curious-VLA (90.3) — 89.2 PDMS is not SOTA in the full NAVSIM field; claims "outperforms all considered methods" only within their selected world-model comparison set
2. **ReCogDrive baseline is IL-only (86.5)**: the RL version achieves 89.6; including it would have changed the VLA comparison picture
3. **NAVSIM-v2 thin comparison**: only DriveVLA-W0 (86.1) as VLA baseline; Senna-2 (86.6), Curious-VLA (85.3), DriveFine not included — 88.7 EPDMS is likely genuine new SOTA but unconfirmed head-to-head
4. **EC = 79.4 on v2**: low extended comfort; depth+video improves safety-oriented metrics but trajectories remain somewhat aggressive
5. **Pseudo-label depth**: Depth Anything 3 provides training targets — no real sensor depth (LiDAR) used; depth evaluation also against pseudo-labels, not ground truth
6. **No RL**: single-stage multi-task learning only; most wiki peers at comparable PDMS use GRPO
7. **Low generation resolution**: 144×256 for both depth and video
8. **Video: front-view only** for FVD evaluation (to match PWM); surround-view generation quality unreported
9. **No nuScenes, Bench2Drive, or Waymo evaluation**

---

## Key Cross-References

- **Geometry-grounded WAM pattern**: [[concepts/world-model-for-ad.md]] — Pattern 6: causal depth→video→action conditioning
- **NAVSIM-v2 new SOTA**: [[concepts/navsim-benchmark.md]] — 88.7 EPDMS surpasses Senna-2 (86.6)
- **FM action generator**: [[concepts/diffusion-planner.md]] — modular flow-matching action expert with LLM cross-attention conditioning

---
title: "Percept-WAM: Perception-Enhanced World-Awareness-Action Model for Robust End-to-End Autonomous Driving"
type: source-summary
sources: [raw/papers/Percept-WAM_ Perception-Enhanced World-Awareness-Action Model for Robust End-to-End Autonomous Driving.md]
related: [concepts/perception-for-planning.md, concepts/diffusion-planner.md, concepts/navsim-benchmark.md, concepts/vlm-domain-adaptation.md]
created: 2026-04-05
updated: 2026-04-05
confidence: high
---

## Citation

Jianhua Han, Meng Tian, Jiangtong Zhu, Fan He, Huixin Zhang, Sitong Guo, Dechang Zhu, Hao Tang, Pei Xu, Yuze Guo, Minzhe Niu, Haojie Zhu, Qichao Dong, Xuechao Yan, Siyuan Dong, Lu Hou, Qingqiu Huang, Xiaosong Jia, Hang Xu.  
Yinwang Intelligent Technology Co. Ltd. + Fudan University.  
arXiv: https://arxiv.org/html/2511.19221v1

---

## Problem Statement

Two failure modes in current VLA planners:

1. **QA-style spatial reasoning** (EMMA, DriveVLM): spatial understanding encoded as question-answer pairs ("what distance is the car ahead?"). Yields indirect, non-persistent localization signals; produces duplicate detections with poorly calibrated confidence in crowded scenes.

2. **Encoder–decoder pipelines without LLM** (DiffusionDrive, Diffusion Planner): sacrifice VLM's reasoning capacity for spatial precision. Brittle in long-tail scenarios that require language-level scene understanding.

Percept-WAM proposes a third path: embed **persistent, metrically grounded world states** as World-PV and World-BEV tokens directly inside the VLM, preserving general reasoning while adding geometric competence.

---

## Architecture

![](<../../raw/assets/x1 6.png>)
*Figure 1: Motivation. QA-style methods lack persistent states. Diffusion encoders lack reasoning. Percept-WAM integrates both: World-PV/BEV tokens carry spatial coordinates + confidence and feed downstream trajectory decoding.*

![](<../../raw/assets/x2 5.png>)
*Figure 2: Full architecture. Streaming camera inputs → Image Encoder → World-PV tokens. Optional LiDAR → PointPillars encoder → World-BEV token initialization. Percept-WAM (InternVL2-8B) processes all tokens: prefill stage produces World-PV/BEV; autoregressive decoding handles language; Action Head runs parallel decoding for trajectory. Streaming KV cache enables temporal efficiency.*

**Backbone**: InternVL2-8B (pretrained, frozen backbone preserved for general reasoning).

### Token Taxonomy

| Token family | Spatial scope | Initialized from | Outputs |
|---|---|---|---|
| **World-PV** | Perspective-view image plane | ViT image features, patchified to H×W grid | 2D det, mono-3D det, seg |
| **World-BEV** | Bird's-eye-view, ego-centered | Random (camera-only) or PointPillars features (LiDAR) | BEV 3D det, BEV map seg |
| **World-Action** | N trajectory points | Random, Q∈ℝ^{N×C} per modality | Trajectory prediction |
| **Text tokens** | — | Standard LLM tokenizer | Language reasoning, QA |

---

## World-PV: Perspective-View Perception

Image features from InternVL's ViT are patchified into an H×W grid. Each cell becomes a **localized query** for single-object perception at that spatial location.

### Grid-Conditioned Detection

![](<../../raw/assets/x5 3.png>)
*Figure 4: Grid query mechanism. World-PV tokens are bilinearly interpolated to produce a grid of localized queries. Each grid token independently predicts one bounding box: `<grid> cls <box> x,y,w,h </box> <conf> s </conf>`. Grid tokens are mutually masked across locations → parallel AR decoding.*

**2D detection output format**: `cls, <box> x,y,w,h </box>, <conf> s </conf>`  
**3D detection output format**: `cls, <box> x,y,z,w,h,ℓ,θ,vx,vy </box>, <conf> s </conf>`  

Continuous values discretized into integer bins; supervised with cross-entropy (following Pix2Seq). Category as free text → natural open-vocabulary detection.

**High-resolution input**: InternVL-style dynamic tiling — high-res images split into non-overlapping tiles, each encoded by shared ViT, fused via global positional alignment. Preserves long-range detail without quadratic memory growth.

### IoU-Aware Confidence

![](<../../raw/assets/x3 4.png>)
*Figure 3: Confidence-tuning dataset generation. Random-perturbation of GT boxes produces near-uniform IoU distribution (poor calibration). Model-prediction dataset has realistic skewed distribution → better aligned with actual inference behavior.*

Standard VLM confidence = softmax(class logits) → overconfident on false positives (lower-right region in Fig 6: high predicted confidence, low IoU with GT).

**Percept-WAM's fix**: add an IoU-prediction token per box, trained in two modes:
- **GT samples**: IoU fixed to 1; model learns class + box without confidence supervision (avoids collapse)
- **Confidence-tuning samples**: model predicts IoU from perturbed labels; only confidence loss applied

![](<../../raw/assets/x7 2.png>)
*Figure 6: Predicted confidence (x-axis) vs. actual IoU with GT (y-axis). After IoU-based confidence training (real model-pred distribution), points cluster near the diagonal — the model's confidence scores are well-calibrated.*

**Final inference score** = class_conf × predicted_IoU

Ablation (nuImages 2D detection):

| Confidence method | AP | AP₅₀ | AP₇₅ |
|---|---|---|---|
| Baseline (class score only) | 48.1 | 70.9 | 51.4 |
| + IoU conf (random-perturb) | 46.9 | 70.0 | 50.7 |
| + IoU conf (uniform model-pred) | 46.2 | 69.1 | 49.3 |
| **+ IoU conf (real model-pred)** | **49.6** | **70.4** | **53.7** |

Key insight: distribution alignment is critical. Random-perturb and uniform sampling both hurt. The realistic model-prediction distribution (skewed toward low IoU) reflects actual test-time conditions and gives +1.5 AP, +2.3 AP₇₅.

### Segmentation

Cast as feature retrieval: model predicts K=16 `<MASK>` tokens; masks retrieved by dot-product similarity with World-PV tokens. All categories produced in one forward pass. Loss: cross-entropy + sigmoid focal + Dice.

---

## World-BEV: Bird's-Eye-View Perception

A set of learnable query tokens forming an H×W BEV grid centered on ego vehicle. World-BEV tokens attend to World-PV tokens via cross-attention → implicit PV-to-BEV lifting in a purely data-driven manner.

**Camera-only**: BEV tokens randomly initialized, trained from scratch.  
**With LiDAR**: PointPillars → PixelUnshuffle → MLP → BEV token initialization. Injects metrically grounded 3D priors.

Grid resolution: 40×40 for detection, 10×10 for segmentation.

BEV 3D detection format: `cls, <box> x,y,z,w,h,l,θ,vx,vy </box>` — continuous values discretized into [0, 1024) integer bins; predicted in parallel AR from grid query tokens.

BEV map segmentation: 16 `<MASK>` tokens per category; independent binary segmentation per class (handles overlapping categories like crosswalk ⊂ drivable area).

---

## World-Action: Trajectory Decoding

![](<../../raw/assets/x6 2.png>)
*Figure 5: Four parallel MLP decoders with modality-specific attention masks. Q_ego attends to ego-state tokens only; Q_pv to World-PV only; Q_bev to World-BEV only; Q_full accesses all features → final trajectory output.*

Four query sets $\mathbf{Q} \in \mathbb{R}^{N \times C}$ (randomly initialized):

| Query | Attends to | Purpose |
|---|---|---|
| $\mathbf{Q}_\text{ego}$ | Ego-state tokens | Kinematic grounding |
| $\mathbf{Q}_\text{pv}$ | World-PV tokens | Semantic/appearance context |
| $\mathbf{Q}_\text{bev}$ | World-BEV tokens | 3D spatial context |
| $\mathbf{Q}_\text{full}$ | All tokens | Fused final trajectory |

All four decoded in parallel during training via Smooth-L1 loss. Only $\mathbf{Q}_\text{full}$ used at inference. The separate decoders prevent over-reliance on any single modality.

**Trajectory scoring (Percept-WAM\*)**: inspired by Hydra-MDP, scores and selects optimal trajectory from a static candidate vocabulary. Adds inference overhead but significantly improves Comfort score (92.8 → 99.5).

### Streaming Inference

KV cache reuse: trajectory at time T attends only to frames T and T−1. Dual-recomputation KV cache mechanism mitigates training-inference distribution drift. Reduces latency 40% (1174 → 707ms) with near-zero accuracy loss.

| Decoding method | L2 Avg ↓ | Latency ↓ |
|---|---|---|
| AR | 0.397 | 2700ms |
| AR + Streaming | 0.406 | 2209ms |
| AR (cluster, AutoVLA-style) | 0.392 | 1470ms |
| Query-based | 0.382 | 1174ms |
| **Query-based + Streaming** | **0.384** | **707ms** |

---

## Training

Two-stage curriculum:
1. **Stage 1**: PV/BEV perception consolidation — spatial grounding established
2. **Stage 2**: E2E VLA fine-tuning — trajectory decoder aligned with world tokens

Optimizer: AdamW, LR 0.0002, weight decay 0.01, cosine decay + 1000-step warmup, mixed precision + gradient checkpointing.

**Task coverage** (single model, Table 2):

| Domain | Tasks | Datasets |
|---|---|---|
| PV | 2D det, mono 3D det, instance seg, semantic seg, grounding | nuImages, nuScenes, COCO, Waymo |
| BEV | 3D det, BEV seg | nuScenes |
| Trajectory | Waypoint prediction | nuScenes, NAVSIM |
| Language | Driving QA | DriveLM |

---

## Results

### E2E Trajectory Planning (Table 3)

| Method | L2 Avg ↓ | NC ↑ | DAC ↑ | TTC ↑ | Comf ↑ | EP ↑ | PDMS ↑ |
|---|---|---|---|---|---|---|---|
| UniAD | 0.46 | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| VAD | 0.37 | 97.2 | 89.1 | 91.6 | 100 | 76.0 | 80.9 |
| DiffusionDrive | 0.57 | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| DriveVLM | 0.40 | — | — | — | — | — | — |
| BEV-Planner | 0.35 | — | — | — | — | — | — |
| DRAMA | — | 98.0 | 93.1 | 94.8 | 100 | 80.1 | 85.5 |
| Hydra-MDP | — | 98.3 | 96.0 | 94.6 | 100 | 78.7 | 86.5 |
| Percept-WAM | 0.38 | 98.7 | 97.8 | 93.2 | 92.8 | 84.4 | 88.6 |
| **Percept-WAM\*** | **0.36** | **98.8** | **98.6** | **94.4** | **99.5** | **84.8** | **90.2** |

Percept-WAM\* achieves 90.2 PDMS (+2.1 over DiffusionDrive). **Comfort drops to 92.8 on base model** — trajectory scoring in \* variant is required to recover comfort (99.5). DAC (98.6) is the highest of any method listed.

### PV Perception (Table 1)

| Task | Dataset | Percept-WAM (AD+Gen) | Best specialist |
|---|---|---|---|
| 2D Detection | nuImages | 49.6 mAP | Mask R-CNN: 47.8 |
| 2D Detection | COCO | **51.7 mAP** | LMM-Det: 47.5 |
| Mono 3D Det | nuScenes | 33.0 mAP | FCOS3D: 32.1 |
| Instance Seg | nuImages | 41.7 mAP | Mask R-CNN: 38.6 |
| Instance Seg | COCO | 45.9 mAP | VisionLLM: 25.2 |
| Semantic Seg | nuImages | 62.8 mIoU | — |
| Semantic Seg | ADE20K | 54.3 mIoU | Mask2Former: 52.4 |
| Semantic Seg | COCOstuff | 50.3 mIoU | DeepLab V2: 33.2 |

![](<../../raw/assets/x10.png>)
*Figure 7: PV perception qualitative. (a) 2D detection — crowded scene. (b) Mono 3D detection. (c) Instance segmentation. (d) Semantic segmentation. (e–f) Open-vocabulary: detects "bird" and "shopping cart" not seen in AD training data.*

### BEV Perception (Table 4)

| Method | mAP | NDS | Drivable | Ped | Lane | Vehicle |
|---|---|---|---|---|---|---|
| BEVFormer | 0.360 | 0.438 | — | — | — | — |
| DETR3D | 0.375 | 0.448 | 80.7 | — | 21.3 | 43.2 |
| PointPillars | 0.523 | 0.613 | — | — | — | — |
| BEVFusion | 0.685 | 0.714 | 85.5 | 60.5 | 67.7 | — |
| **Percept-WAM** | **0.589** | **0.645** | **87.0** | **70.9** | 62.7 | 60.2 |

Surpasses PointPillars on detection. Surpasses BEVFusion on drivable area (87.0 vs. 85.5) and pedestrian crossing (70.9 vs. 60.5). Falls short on lane divider (62.7 vs. 67.7).

![](<../../raw/assets/x11 1.png>)
*Figure 8: BEV perception qualitative. Left: 3D bounding boxes projected onto 6 surrounding cameras. Right: BEV map segmentation (drivable area, pedestrian crossing, lane divider, vehicle).*

### BEV Detection Ablation (Table 6)

| Configuration | mAP | NDS |
|---|---|---|
| Baseline (camera only) | 25.0 | 25.7 |
| + LiDAR encoder init | 33.2 | 32.2 |
| + Data augmentation | 41.3 | 39.2 |
| + Grid resolution (20×20 → 40×40) | 50.4 | 46.6 |
| + MLP parallel decoding (16× speedup) | 50.4 | 43.7 |

LiDAR encoder initialization is the single largest contributor (+8.2 mAP). Switching to MLP parallel decoding maintains mAP while providing 16× inference speedup (small NDS drop of 2.9).

### Trajectory Planning Qualitative

![](<../../raw/assets/x12.png>)
*Figure 9: Construction zone navigation. Left: 6-camera surround view showing construction equipment and oncoming vehicles. Right: BEV trajectory — model successfully yields to oncoming vehicle while making right turn through construction zone.*

---

## Limitations

1. **Incomplete NAVSIM comparison**: Table 3 compares only against DiffusionDrive (88.1), Hydra-MDP (86.5), DRAMA (85.5). No head-to-head vs. WAM-Flow (90.3), ReCogDrive (89.6), or Reasoning-VLA (91.7 claimed). The 90.2 score is competitive with but likely below WAM-Flow.

2. **Comfort collapse on base model**: Percept-WAM without trajectory scoring achieves Comf=92.8 vs. all other methods at 99-100. Requires the \* variant (trajectory candidate scoring) to recover, adding inference complexity.

3. **LiDAR dependence for BEV**: Camera-only BEV detection baseline is only 25.0 mAP; final 58.9 mAP requires LiDAR initialization. The camera-only BEV capability is significantly weaker.

4. **No RL training**: No GRPO or RL fine-tuning applied. All other SOTA methods in this wiki use RL. The conclusion explicitly lists RL as future work.

5. **Perception gains over modest baselines**: Outperforming PointPillars (LiDAR-only detector from 2019) is a low bar for a 2025 model. BEVFusion (camera+LiDAR) still significantly outperforms on BEV detection (0.685 vs. 0.589).

6. **Percept-WAM\* not fully E2E**: Hydra-MDP-style trajectory scoring is a rule-based post-processing step over a static vocabulary, not learned end-to-end.

---

## Relationship to Other Papers

| Paper | Perception integration | Action head | RL | PDMS |
|---|---|---|---|---|
| ReCogDrive | None (VLM features only) | Continuous DiT | GRPO | 89.6 |
| WAM-Flow | None | DFM tokens | GRPO | 90.3 |
| ReflectDrive | None | Masked diffusion | None | >89.1 |
| Reasoning-VLA | None | Learnable queries (1-step) | GRPO (GT) | 91.7 (claimed) |
| **Percept-WAM** | **World-PV + World-BEV tokens** | **Four-query MLP** | **None** | **90.2** |

Percept-WAM is the only paper in this wiki that explicitly integrates 2D/3D perception tasks as first-class training objectives alongside trajectory prediction, conceptually extending UniAD's planning-oriented multi-task learning into the VLM paradigm.

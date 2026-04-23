---
title: "DriveSuprim: Towards Precise Trajectory Selection for End-to-End Planning"
type: source-summary
sources: [raw/papers/DriveSuprim_ Towards Precise Trajectory Selection for End-to-End Planning.md]
related: [concepts/navsim-benchmark.md, concepts/selection-based-planning.md, concepts/best-of-n.md, concepts/rl-for-ad.md, concepts/diffusion-planner.md]
created: 2026-04-23
updated: 2026-04-23
confidence: high
---

## One-Line Summary

Coarse-to-fine trajectory selection + rotation augmentation + EMA self-distillation on top of Hydra-MDP; **93.5% PDMS** NAVSIM-v1 and **87.1% EPDMS** NAVSIM-v2 (both SOTA for selection-based methods, no extra data).

**arXiv**: 2506.06659v1  
**Org**: Fudan University + NVIDIA  
**Venue**: 2025 (preprint)

---

## Context and Motivation

Selection-based planning (e.g., Hydra-MDP) scores a fixed vocabulary of N candidate trajectories and selects the highest-scoring one. The oracle study in this paper (Table 1) reveals the method's theoretical ceiling:

| Top-K oracle | NC | DAC | EP | TTC | C | PDMS |
|---|---|---|---|---|---|---|
| Top-1 (current single-best) | 99.0 | 98.7 | 86.5 | 96.2 | 100 | 91.9 |
| Top-4 | 99.4 | 99.6 | 89.6 | 98.0 | 100 | 94.5 |
| Top-16 | 99.7 | 99.8 | 92.0 | 99.1 | 100 | 96.1 |
| Top-256 | 100 | 100 | 97.1 | 99.9 | 100 | 98.7 |
| Human GT | 100 | 100 | 87.5 | 100 | 99.9 | 94.8 |

With oracle selection from just the top-4 candidates, performance (94.5) nearly matches human GT (94.8). With 256, it reaches 98.7. The ceiling is very high — the bottleneck is **selector quality**, not candidate quality.

Three specific failure modes prevent current methods from reaching this ceiling:

1. **Hard negatives**: The vocabulary contains thousands of obviously bad trajectories ("easy negatives") that dominate training. The model never develops fine-grained discrimination for subtly unsafe trajectories ("hard negatives") that look plausible.
2. **Directional bias**: Only 8% of NAVSIM ground-truth trajectories involve turns >30°. Training distribution is dominated by straight-ahead driving, causing weak turning performance.
3. **Hard binary labels**: BCE against {0,1} safety scores creates unstable training near the boundary; models become hypersensitive to minor trajectory variations.

---

## Method: DriveSuprim

![[x1 23.png|Overall pipeline]]
*Figure 1: The overall pipeline. Green = GT, red = obviously unsafe, orange = hard negative (subtly wrong), blue = DriveSuprim output.*

![[x2 21.png|Model architecture]]
*Figure 2: Model architecture with coarse-to-fine Trajectory + Refinement Decoders, rotation augmentation, and self-distillation framework.*

### 3.1 Coarse-to-Fine Trajectory Selection

**Problem**: single-stage scoring over 8192 trajectories forces the model to simultaneously discriminate easy and hard negatives. Easy negatives dominate, starving hard-negative signal.

**Solution**: a two-stage funnel.

**Stage 1 — Coarse filtering** (identical to Hydra-MDP):
- Trajectory encoder (2-layer MLP) + Transformer Decoder cross-attending image features
- Scores all 8192 trajectories on each metric (L2 + NC + DAC + EP + TTC + C/EC)
- Selects top-256 ("filtered trajectories") by combined score

**Stage 2 — Fine-grained scoring** (Refinement Decoder):
- Separate 3-layer Transformer Decoder applied only to the 256 filtered trajectories
- Outputs layer-wise refined scores $s_{j,l}^{(m)}$
- Final selection uses last-layer output

The key insight from ablation (Table 5): simply adding 3 more layers to the Trajectory Decoder gives **+0 EPDMS**. Adding layer-wise scoring gives **+0 EPDMS**. Only the trajectory filtering (narrowing from 8192 to 256) gives **+0.8 EPDMS**. The gain is purely from the model seeing only hard negatives in Stage 2.

$$L_{\text{ori}} = L_{\text{coarse}} + L_{\text{refine}}$$

where $L_{\text{coarse}} = L_{\text{imi}} + \sum_{m,i} \text{BCE}(s_i^{(m)}, y_i^{(m)})$ and $L_{\text{refine}}$ applies the same loss only over filtered trajectories $T_{\text{filter}}$.

### 3.2 Rotation-Based Data Augmentation

**Problem**: 8% turning trajectories → models fail on sharp turns despite high overall PDMS.

**Solution**: simulate ego-vehicle rotation via camera image shifting.

![[x3 21.png|Rotation augmentation]]
*Figure 3: Rotation augmentation via pseudo-panoramic view and horizontal crop window.*

- Sample $\theta \sim U[-\pi/6, +\pi/6]$ (±30°)
- Concatenate front-left ($l_0$), front ($f$), front-right ($r_0$) cameras into pseudo-panoramic view
- Crop the standard FOV window by shifting horizontally based on $\theta$
- Rotate GT trajectory by $-\theta$ (each waypoint rotated around initial position)
- Compute augmentation loss $L_{\text{aug}}$ (same formulation as $L_{\text{ori}}$)

**Effect** (Figure 4): the original dataset has trajectories concentrated in the forward direction. Post-augmentation, all directions appear at similar frequency.

![[trajectories_ori_vs_rotated.png|Trajectory distribution comparison]]
*Figure 4: High-score trajectory distribution — original (forward-heavy) vs. augmented (uniform). Color bar = normalized frequency.*

### 3.3 EMA Self-Distillation with Soft Labels

**Problem**: hard {0,1} BCE targets cause unstable training near safety thresholds — a trajectory just below the collision threshold is treated identically to a catastrophically unsafe one.

**Solution**: EMA teacher generates soft labels bounded by ground truth.

- **Teacher**: EMA copy of student; momentum linearly increases 0.992 → 0.996 (first half), then fixed at 0.998
- **Soft label**: $\hat{y}_i^{(m)} = y_i^{(m)} + \text{clip}(s_{i,\text{teacher}}^{(m)} - y_i^{(m)},\ -0.15,\ +0.15)$
- Threshold $\delta_m = 0.15$ (best in ablation; Table 7)
- Student trains on both original ($L_{\text{ori}}$) and augmented ($L_{\text{aug}}$) data plus soft-label loss ($L_{\text{soft}}$)
- Teacher used at **inference** (not student)

For the imitation loss, the teacher's predicted trajectory can shift the GT target by at most 1 meter ($L_{\text{imi-soft}}$).

$$L = L_{\text{ori}} + L_{\text{aug}} + L_{\text{soft}}$$

---

## Architecture Details

| Component | Design |
|---|---|
| Image encoder | ResNet34 / VoVNet (V2-99) / ViT-Large (pre-trained by Depth Anything) |
| Input resolution | 2048×512 (ResNet/VoVNet), 1024×256 (ViT-L) |
| Camera setup | 3-camera (l₀, f, r₀) — best in ablation (Table 11) |
| Trajectory encoder | 2-layer MLP |
| Trajectory Decoder (Stage 1) | 3-layer Transformer Decoder, dim=256 |
| Refinement Decoder (Stage 2) | 3-layer Transformer Decoder, dim=256 |
| Vocabulary size | 8192 trajectories |
| Filtered candidates (top-K) | 256 (best in Table 6) |
| Prediction heads | L2 distance: 2-layer MLP; metric scores: linear layer |
| Training | 8× NVIDIA A100, Adam, lr=7.5×10⁻⁵, batch=8/GPU |
| ViT-L training | 6 epochs; ResNet34: 10 epochs (EMA delayed 3 epochs) |

**5-camera vs 3-camera**: The 5-camera setting (adding l₁ and r₁) gives 86.9 EPDMS, slightly below 3-camera (87.1). More cameras do not help — wider context is not necessary; the 3-camera FOV already covers the relevant range for the rotation augmentation.

---

## Main Results

### NAVSIM v1 (PDMS)

| Method | Backbone | NC | DAC | EP | TTC | C | PDMS |
|---|---|---|---|---|---|---|---|
| Human Agent | — | 100 | 100 | 87.5 | 100 | 99.9 | 94.8 |
| DiffusionDrive | ResNet34 | 98.2 | 96.2 | 82.2 | 94.7 | 100 | 88.1 |
| Hydra-MDP | ResNet34 | 98.3 | 96.0 | 78.7 | 94.6 | 100 | 86.5 |
| **DriveSuprim** | **ResNet34** | 97.8 | 97.3 | 86.7 | 93.6 | 100 | **89.9 (+1.8)** |
| Hydra-MDP | V2-99 | 98.4 | 97.8 | 86.5 | 93.9 | 100 | 90.3 |
| **DriveSuprim** | **V2-99** | 98.0 | 98.2 | 90.0 | 94.2 | 100 | **92.1 (+1.8)** |
| Hydra-MDP | ViT-L | 98.4 | 97.7 | 85.0 | 94.5 | 100 | 89.9 |
| **DriveSuprim** | **ViT-L** | 98.6 | 98.6 | 91.3 | 95.5 | 100 | **93.5 (+3.6)** |

EP (ego progress) is the biggest beneficiary — the turning augmentation directly enables the model to make progress through turning scenarios rather than stalling.

### NAVSIM v2 (EPDMS)

| Method | Backbone | NC | DAC | DDC | TL | EP | TTC | LK | HC | EC | EPDMS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Human Agent | — | 100 | 100 | 99.8 | 100 | 87.4 | 100 | 100 | 98.1 | 90.1 | 90.3 |
| HydraMDP++ | ViT-L | 98.5 | 98.5 | 99.5 | 99.7 | 87.4 | 97.9 | 95.8 | 98.2 | 75.7 | 85.6 |
| **DriveSuprim** | **ViT-L** | 98.4 | 98.6 | 99.6 | 99.8 | 90.5 | 97.8 | 97.0 | 98.3 | 78.6 | **87.1 (+1.5)** |

### Turning scenario performance (EPDMS, ViT-L)

| Scenario | HydraMDP++ | DriveSuprim | Delta |
|---|---|---|---|
| NAVTESTₗₑ꜀ₜ (>30° left) | 68.7 | 71.6 | +2.9 |
| NAVTESTꜰₒᵣwₐᵣd (near-straight) | 87.2 | 88.1 | +0.9 |
| NAVTESTᵣᵢ𝓰ₕₜ (>30° right) | 77.7 | 79.7 | +2.0 |

Improvement is 2–3× larger for turning scenarios than straight driving — confirming the rotation augmentation hypothesis.

---

## Ablation Studies

### Module ablation (ViT-L, NAVSIM v2, Table 4)

| Multi-stage | Aug Data | Self-distill | EPDMS |
|---|---|---|---|
| ✓ | ✓ | ✓ | **87.1** |
| ✗ | ✓ | ✓ | 85.6 (−1.5) |
| ✓ | ✗ | ✓ | 86.4 (−0.7) |
| ✓ | ✓ | ✗ | 85.6 (−1.5) |

Multi-stage refinement and self-distillation each contribute +1.5; augmentation contributes +0.7.

### Coarse-to-fine evolution (Table 5)

| Configuration | EPDMS |
|---|---|
| Single-stage (Hydra-MDP baseline) | 85.6 |
| + 6-layer decoder (more params) | 85.3 (−0.3) |
| + Layer-wise scoring | 85.6 (+0.3) |
| + Trajectory filtering (256) | **86.4 (+0.8)** |

Parameter count and layer-wise scoring add nothing. Only trajectory filtering helps — the model must see a concentrated hard-negative set.

### Refinement hyperparameters (Table 6)

| Stage Layers | Top-K | EPDMS |
|---|---|---|
| 1 | 256 | 86.6 |
| **3** | **256** | **87.1** |
| 6 | 256 | 86.6 |
| 3 | 64 | 86.5 |
| 3 | 512 | 86.9 |
| 3 | 1024 | 86.4 |

3 layers + 256 filtered trajectories is optimal. Too many (1024) or too few (64) filtered candidates both hurt.

### Soft-label threshold (Table 7)

| δₘ | EPDMS |
|---|---|
| 0.00 (hard labels) | 86.8 |
| **0.15** | **87.1** |
| 0.30 | 86.8 |
| 0.70 | 86.5 |
| 1.00 | 86.6 |

A small but non-zero threshold (0.15) is optimal — pure hard labels or fully relaxed soft labels both underperform.

### FOV settings (Table 11)

| Cameras | EPDMS |
|---|---|
| 1 (front only) | 85.4 |
| **3 (l₀, f, r₀)** | **87.1** |
| 5 (l₁, l₀, f, r₀, r₁) | 86.9 |

3 cameras outperform both 1 and 5, suggesting optimal coverage for the augmentation pipeline.

---

## Visualization

![[x4 20.png|Visualization results]]
*Figure 5: Comparison with Hydra-MDP++ (first row) vs. DriveSuprim (second row). Green = GT, red = Hydra-MDP++, blue = DriveSuprim.*

Four scenarios demonstrated:
1. **Overtaking before crossroad**: Hydra-MDP++ incorrectly turns left (collision risk); DriveSuprim executes correct overtaking
2. **Hard negative discrimination**: both models produce similar-looking trajectories; Hydra-MDP++ deviates at endpoint; DriveSuprim matches GT
3–4. **Sharp turns**: DriveSuprim generates smooth accurate trajectories; Hydra-MDP++ struggles

![[x5 20.png|Five-camera views]]
*Figure 6: Five cameras: l₁, l₀, f, r₀, r₁.*

---

## Inference Score Combination

At inference, per-metric scores are log-combined into a final ranking score:

$$s_i = \sum_m \lambda^{(m)} \log s_i^{(m)} + \lambda_{\text{avg}} \log\!\left(\sum_n \lambda^{(n)} s_i^{(n)}\right)$$

where $m$ indexes multiplier/imitation metrics and $n$ indexes weighted-average metrics. $\lambda_{\text{avg}} = 8.0$ (v1) or $6.0$ (v2).

| Metric | Type | v1 coefficient | v2 coefficient |
|---|---|---|---|
| Imitation (L2) | Imi | 0.05 | 0.02 |
| NC | Mul | 0.5 | 0.5 |
| DAC | Mul | 0.5 | 0.5 |
| DDC | Mul | — | 0.3 |
| TL | Mul | — | 0.1 |
| EP | Avg | 5.0 | 5.0 |
| TTC | Avg | 5.0 | 5.0 |
| C / HC | Avg | 2.0 | 1.0 |
| LK | Avg | — | 2.0 |

---

## Limitations

1. **Two-stage is the ceiling**: extending to 3+ stages (multi-stage refinement) provides no further improvement.
2. **Inference speed**: two-pass scoring is slower than single-stage methods; the paper acknowledges this but provides no latency numbers.
3. **NAVSIM-only evaluation**: no Bench2Drive or real closed-loop testing.
4. **Non-reactive simulation**: standard NAVSIM caveat — other agents do not respond to ego.
5. **EC not directly predicted**: extended comfort is a byproduct, not an explicit prediction head; this may cap v2 performance compared to methods that explicitly optimize it.
6. **No VLM reasoning**: purely rule-score-based selection — no semantic scene understanding.

---

## Position in the Wiki

DriveSuprim (93.5 PDMS ViT-L) is the **highest single-sample, non-VLM, non-ensemble result on NAVSIM-v1** in the wiki, surpassing DiffusionDriveV2 (91.2, Camera+LiDAR ResNet34) and HybridDriveVLA (92.1, dual-model ensemble). It falls below Curious-VLA BoN-6 (94.8) and AdaThinkDrive BoN-4 (93.0) which use oracle selection.

On NAVSIM-v2, 87.1 EPDMS slots between DreamerAD (87.7) and HydraMDP++ (85.6). EC = 78.6 — in the middle range.

The oracle study (Table 1) directly shows that the selection-based ceiling is 94.5 PDMS with only 4 oracle-chosen candidates from 8192. This is a key data point for the [[concepts/best-of-n.md]] discussion of fixed-vocabulary oracle selection vs. stochastic BoN sampling.

See [[concepts/selection-based-planning.md]] for the broader paradigm context.

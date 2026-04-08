---
title: "FutureSightDrive: Thinking Visually with Spatio-Temporal CoT for Autonomous Driving"
type: source-summary
sources: [raw/papers/FutureSightDrive_ Thinking Visually with Spatio-Temporal CoT for Autonomous Driving.md]
related: [concepts/world-model-for-ad.md, concepts/vlm-domain-adaptation.md, concepts/navsim-benchmark.md]
created: 2026-04-07
updated: 2026-04-07
confidence: high
---

**Paper**: FutureSightDrive: Thinking Visually with Spatio-Temporal CoT for Autonomous Driving  
**Orgs**: Xi'an Jiaotong University, Amap (Alibaba Group), DAMO Academy  
**arXiv**: 2505.17685v3

---

## Summary

FSDrive argues that textual CoT compresses rich visual information into lossy symbols, introducing a "modality gap" between perception and planning. The fix: replace text CoT with a **visual spatio-temporal CoT** — a generated unified future image frame with red lane dividers and 3D detection boxes overlaid. The same VLA plays two roles: (1) world model generating the visual CoT, then (2) inverse dynamics model conditioning trajectory planning on it. Vocabulary expansion via MoVQGAN VQ-VAE tokens appended to the MLLM text vocabulary activates image generation with ~0.3% of the data used by prior methods, without architectural change.

---

## Architecture

![[compare.png|Comparison of different CoT strategies]]

Figure 1: Textual CoT compresses spatial information into abstract symbols; image-text CoT introduces cross-modal inconsistency; spatio-temporal CoT captures temporal and spatial relationships in a single unified image.

![[method_final.png|FSDrive overview]]

Figure 2: Overview. MLLM is trained for next-token prediction over both text and VQ-VAE image tokens. During inference: generate unified future frame (visual CoT) → plan trajectory conditioned on current obs + visual CoT.

**Backbone**: Qwen2-VL-2B (also tested with LLaVA-7B)  
**Image tokenizer**: MoVQGAN VQ-VAE (vocabulary extended at 128×192 resolution)

### Vocabulary Expansion

The sole architectural modification: VQ-VAE codebook tokens are appended to the MLLM's text vocabulary. No encoder/decoder changes. This allows the model to predict image tokens in raster order using the same next-token prediction objective as text.

### Trajectory Planning Formula

$$P(W_{t}\mid I_{t}, Q_{CoT}, opt(T_{com}, T_{ego})) = \prod_{i=1}^{n} P_\theta(w_i \mid w_{<i}, I_t, Q_{CoT}, opt(T_{com}, T_{ego}))$$

Where $Q_{CoT}$ is the generated visual spatio-temporal CoT (unified future frame with overlaid priors).

### Progressive Generation (Pre-training)

Lane dividers first → 3D bounding boxes → full future frame:

$$P(Q_f \mid Q_l, Q_d) = \prod_{t=1}^{h \cdot w} P_\theta(q_i \mid q_{<i}, Q_l, Q_d)$$

This enforces physical laws (static road structure → dynamic agent layout → appearance) before rendering fine-grained details.

---

## 2-Stage Training

### Stage 1: Unified Pre-training

Jointly trains:
- **VQA** on OmniDrive-nuScenes (preserves semantic understanding)
- **Future frame generation** on unlabeled nuScenes video (world model capacity)
- **Progressive generation** on annotated nuScenes: lane dividers → 3D detection → full frame

All tasks use the same next-token prediction objective. No specialized heads or objectives.

### Stage 2: Supervised Fine-tuning

- **Scene understanding**: DriveLM GVQA dataset
- **Trajectory planning**: nuScenes (VAD/UniAD protocol) with visual CoT as intermediate
- 12 epochs, 8× NVIDIA RTX A6000, lr=1e-4, batch=16

---

## Results

### nuScenes Trajectory Planning (Table 1)

#### ST-P3 Metrics (no ego status, Qwen2-VL-2B)

| Method | Avg L2 ↓ | Avg Collision ↓ | LLM |
|---|---|---|---|
| Doe-1 | 0.70 | 0.21 | Lumina-mGPT-7B |
| OmniDrive | 0.84 | 0.94 | LLaVA-7B |
| **FSDrive** | **0.53** | **0.17** | Qwen2-VL-2B |

#### UniAD Metrics (no ego status, Qwen2-VL-2B)

| Method | L2 1s | L2 2s | L2 3s | Avg L2 ↓ | Col 1s | Col 2s | Col 3s | Avg Col ↓ |
|---|---|---|---|---|---|---|---|---|
| ELM | 0.34 | 1.23 | 2.57 | 1.38 | 0.12 | 0.50 | 2.36 | 0.99 |
| OccWorld | 0.52 | 1.27 | 2.41 | 1.40 | 0.12 | 0.40 | 2.08 | 0.87 |
| Doe-1 | 0.50 | 1.18 | 2.11 | 1.26 | 0.04 | 0.37 | 1.19 | 0.53 |
| **FSDrive (2B)** | **0.40** | **0.89** | **1.60** | **0.96** | **0.07** | **0.12** | **1.02** | **0.40** |
| FSDrive* (2B, ego) | 0.18 | 0.39 | 0.77 | 0.45 | 0.00 | 0.06 | 0.42 | 0.16 |
| FSDrive (7B, LLaVA) | 0.36 | 1.01 | 1.90 | 1.09 | 0.08 | 0.34 | 1.11 | 0.51 |
| FSDrive* (7B, ego) | 0.22 | 0.51 | 0.94 | 0.56 | 0.02 | 0.07 | 0.53 | 0.21 |

**Key comparisons**:
- Beats Doe-1 (also VQ-VAE visual generation) on both L2 and collision at smaller model size (2B vs. 7B)
- Without ego status, 0.96m avg L2 on UniAD metrics; FSDrive notes ego status inflates scores (BEV-Planner finding)

### NAVSIM (Table 2)

| Method | NC ↑ | DAC ↑ | TTC ↑ | Comf. ↑ | EP ↑ | PDMS ↑ |
|---|---|---|---|---|---|---|
| PARA-Drive | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 |
| LAW | 96.4 | 95.4 | 88.7 | 99.9 | 81.7 | 84.6 |
| **FSDrive** | **98.2** | **93.8** | **93.3** | **99.9** | **80.1** | **85.1** |

**Caveat**: Table 2 compares only against PARA-Drive (84.0), LAW (84.6), and other pre-2025 baselines. Does not include ReCogDrive (89.6), WAM-Flow (90.3), DriveFine (90.7), or any wiki paper. 85.1 PDMS is below the wiki median for camera-only methods.

### Future Frame Generation (Table 3, FID ↓)

| Method | Type | Resolution | FID ↓ |
|---|---|---|---|
| DriveDreamer | Diffusion | 128×192 | 52.6 |
| Drive-WM | Diffusion | 192×384 | 15.8 |
| GenAD | Diffusion | 256×448 | 15.4 |
| Doe-1 | Autoregressive | 384×672 | 15.9 |
| GEM | Diffusion | 576×1024 | 10.5 |
| **FSDrive** | **Autoregressive** | **128×192** | **10.1** |

FSDrive achieves competitive FID (10.1) at lower resolution (128×192) vs. diffusion models at larger resolutions. Better than Doe-1 (15.9, higher res, 7B) from a 2B model.

### DriveLM Scene Understanding (Table 4)

| Method | Accuracy | ChatGPT | BLEU_1 | ROUGE_L | CIDEr | Match | Final Score ↑ |
|---|---|---|---|---|---|---|---|
| Cube-LLM | 0.39 | 0.89 | 0.16 | 0.20 | 0.31 | 0.39 | 0.50 |
| OmniDrive | 0.70 | 0.65 | 0.52 | 0.73 | 0.13 | 0.37 | 0.56 |
| **FSDrive** | **0.72** | **0.63** | **0.76** | **0.74** | **0.17** | **0.39** | **0.57** |

Margin over OmniDrive (+0.01 Final Score) is thin.

---

## Ablations

### Pre-training Components (Table 5, UniAD no ego status)

| VQA | Future Frames | 3D Detection | Lane Divider | Avg L2 ↓ | Avg Collision ↓ |
|---|---|---|---|---|---|
| ✗ | ✗ | ✗ | ✗ | 1.22 | 0.67 |
| ✓ | ✗ | ✗ | ✗ | 1.19 | 0.65 |
| ✗ | ✓ | ✗ | ✗ | 1.02 | 0.60 |
| ✗ | ✗ | ✓ | ✗ | 1.17 | 0.61 |
| ✗ | ✗ | ✗ | ✓ | 1.06 | 0.65 |
| ✓ | ✓ | ✓ | ✓ | 0.98 | 0.58 |

**Key finding**: future frame generation alone provides the biggest single gain (−16.4% L2, −15.8% collision vs. baseline). VQA alone has almost no effect (+0 L2). All tasks combined is best.

### CoT Type Ablation (Table 6, UniAD no ego status)

| CoT Type | Avg L2 ↓ | Avg Collision ↓ | Collision improvement |
|---|---|---|---|
| None | 0.98 | 0.58 | — |
| Text CoT | 0.97 | 0.53 | −8.6% |
| Image-text CoT | 0.98 | 0.50 | −13.8% |
| **Spatio-temporal CoT** | **0.96** | **0.40** | **−31.0%** |

L2 improvement from visual CoT is marginal (~2%); collision rate improvement is substantial (+31%). Visual CoT primarily helps the model anticipate and avoid future collisions, not improve path accuracy.

### Generation Quality Ablation (Table 7)

| Pre-training Data | Progressive Method | FID ↓ |
|---|---|---|
| None | ✗ | 29.4 |
| ~100k | ✗ | 16.2 |
| ~200k | ✗ | 12.7 |
| ~200k | ✓ | **10.1** |

Progressive generation (lane dividers → boxes → full frame) adds −2.6 FID on top of 200k data. Both data scale and progressive method contribute.

---

## Qualitative Analysis

![[vis.png|Qualitative CoT examples]]

Figure 3: Red trajectory = prediction, green = GT. Without visual CoT, erroneous navigation instructions cause significant trajectory deviations. FSDrive mitigates instruction errors through its inverse dynamics modeling (plans from visual future observation, not just text command).

---

## Limitations

1. **Front-view generation only**: generates future frame for front camera; no surround-view visual CoT despite using 6-camera input for planning
2. **NAVSIM comparison gap**: Table 2 excludes all wiki-era methods (ReCogDrive 89.6, WAM-Flow 90.3, DriveFine 90.7); 85.1 PDMS is below wiki median for camera-only
3. **2B backbone throughout**: all primary results use Qwen2-VL-2B; advantage over 7B+ may not hold at scale
4. **VQ-VAE quantization artifacts**: 128×192 generation resolution is very low; planning quality may degrade on distribution-shifted or detail-dependent scenes
5. **No RL fine-tuning**: SFT-only; no GRPO or reward-based optimization unlike most wiki peers
6. **Progressive generation adds latency**: sequential lane → box → full frame generation must complete before trajectory decoding; inference time not reported
7. **DriveLM margin is thin**: +0.01 over OmniDrive (Final Score 0.57 vs. 0.56)
8. **L2 improvement from visual CoT is marginal** (0.98→0.96 avg); the main gain is collision reduction (31%); the model's trajectory accuracy is not substantially improved by visual reasoning
9. **No NAVSIM-v2 / EPDMS**; no Bench2Drive; no Waymo evaluation

---

## Key Cross-References

- **Visual world model role**: [[concepts/world-model-for-ad.md]] — FSDrive adds Pattern 5: VQ-VAE autoregressive visual CoT as planning intermediate
- **Vocabulary expansion strategy**: [[concepts/vlm-domain-adaptation.md]] — minimal-change approach to add visual generation without architectural modification
- **NAVSIM standing**: [[concepts/navsim-benchmark.md]] — 85.1 PDMS (pre-2025 comparisons only); well below current wiki SOTA

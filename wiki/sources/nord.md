---
title: "NoRD: A Data-Efficient Vision-Language-Action Model that Drives without Reasoning"
type: source-summary
sources: [raw/papers/NoRD_ A Data-Efficient Vision-Language-Action Model that Drives without Reasoning.md]
related: [concepts/rl-for-ad.md, concepts/vlm-domain-adaptation.md, concepts/navsim-benchmark.md, sources/autovla.md, sources/curious-vla.md, sources/recogdrive.md, sources/flare.md]
created: 2026-04-15
updated: 2026-04-15
confidence: high
---

## Overview

**NoRD** (No Reasoning for Driving) is a reasoning-free, data-efficient Vision-Language-Action model for autonomous driving from Applied Intuition and Texas A&M / UC Berkeley. It challenges the prevailing assumption that competitive VLA performance requires large-scale datasets with Chain-of-Thought reasoning annotations.

The core contribution is two-fold:
1. Identifying that GRPO fails on weak (data-efficient) SFT policies due to **difficulty bias** — not due to the absence of reasoning
2. Applying **Dr. GRPO** as a drop-in fix that enables RL post-training from a small, reasoning-free initialization

**arXiv**: 2602.21172v1  
**Org**: Applied Intuition; Texas A&M University; UC Berkeley

---

## Figure 1: Training Pipeline Comparison

![[raw/assets/x1 18.png]]

**(a) Existing approaches**: large-scale reasoning data generation → extensive SFT with CoT → RL fine-tuning  
**(b) NoRD**: small-scale dataset → direct SFT (no reasoning) → Dr. GRPO RL fine-tuning tailored for weak SFT

---

## Problem Statement

The dominant VLA training paradigm:
1. SFT on large-scale datasets with CoT reasoning annotations (e.g., 212K+ samples)
2. GRPO RL post-training on closed-loop metrics (PDM score, RFS)

Three non-scalable costs:
- **Data cost**: vast quantities of specialized driving scenarios
- **Annotation cost**: high-quality reasoning traces for each sample
- **Inference cost**: reasoning tokens add latency impractical for real-time deployment

**NoRD's hypothesis**: competitive performance is achievable with <60% data and no reasoning annotations, *if* the RL optimizer is fixed.

---

## Architecture

![[raw/assets/nord.png]]

**Figure 5**: NoRD directly predicts action tokens without reasoning traces.

**Base model**: Qwen-2.5VL-3B-Instruct

**Inputs**:
- RGB images: front, front-left, front-right cameras (3 frames)
- Past ego-trajectory
- Current speed and acceleration

**Output**: future ego-trajectory at 10 Hz via discrete trajectory tokens

**k-disc tokenization** (vocabulary size = 2048):
- All training trajectories interpolated to 10 Hz, segmented into 0.5s intervals
- Segments clustered into 2048 clusters by contour distance
- Cluster centers form a trajectory codebook; any trajectory encoded as ≤8 (NAVSIM) or ≤10 (WaymoE2E) tokens
- Trajectory tokens appended to Qwen vocabulary; initialized from multivariate normal parameterized by existing token embeddings

**No reasoning tokens at train or inference time** → 3× fewer tokens than reasoning-based VLAs.

---

## Two-Stage Training

### Stage 1: Supervised Fine-Tuning with Limited Data (NoRD-base)

- Trained on **80,000 NAVSIM training samples** (vs. 212K+ for AutoVLA)
- No reasoning annotations
- Standard next-token prediction on trajectory tokens
- Intentionally weak — majority of learning offloaded to RL stage

### Stage 2: RL Post-Training via Dr. GRPO

See §Difficulty Bias and Dr. GRPO below.

**Implementation**:
- NAVSIM: 30 A100 GPUs, 160 RL steps, lr = 5×10⁻⁶, group size G=8, temperature 1.0
- WaymoE2E: 32 GPUs, 150 steps, lr = 1×10⁻⁶, 12K SFT samples + 450 RLFT samples
- Framework: verl + FSDP + vLLM for rollouts

---

## Difficulty Bias in GRPO

### Empirical Observation

![[raw/assets/difficulty_plot.png]]

**Figure 2**: Group-mean PDM score distribution for NoRD-base. Band = mean intra-group std.

Two reward regimes for NoRD-base rollouts (G=8):

| Region | Group-mean PDM | Intra-group std | Samples |
|--------|---------------|-----------------|---------|
| High-mean | ≥0.8 | Low | Small fraction — simple scenarios (straight driving) |
| Low-mean | ≤0.15 | Low | Small fraction — fully OOD |
| Intermediate-mean | [0.2, 0.65] | **High** | **Majority** — complex maneuvers (sharp turns, lane changes) |

The weak SFT model fails more often than succeeds on complex maneuvers → intermediate mean, high variance.

### Why GRPO Fails

GRPO advantage formula:
$$\hat{A}_{i,t}^{\text{GRPO}} = \frac{r(o_i \mid x) - \frac{1}{G}\sum_{j=1}^G r(o_j \mid x)}{\text{std}_{j=1,\ldots,G}(r(o_j \mid x))}$$

When std is large (high-variance intermediate group), the denominator attenuates the advantage → **GRPO provides near-zero gradient signal** for exactly the scenarios the model needs most to improve.

Result: GRPO learns only from the small fraction of low-variance samples (easy or fully-solved). The high-variance majority — the complex driving scenarios — provide no effective learning signal.

**Table 1: GRPO vs. Dr. GRPO on NoRD-base (NAVSIM test)**

| Model | PDMS |
|-------|------|
| NoRD-base | 76.66 |
| NoRD-base + GRPO | 77.18 (+0.67%) |
| **NoRD-base + Dr. GRPO** | **85.62 (+11.68%)** |

---

## Dr. GRPO: Removing Difficulty Bias

![[raw/assets/comparison_figure.png]]

**Figure 4**: Qualitative comparison. With Dr. GRPO, NoRD learns sharp turns and lane changes. GRPO fails (red collision).

Dr. GRPO removes the std normalization term:
$$\hat{A}_{i,t}^{\text{DrGRPO}} = r(o_i \mid x) - \frac{1}{G}\sum_{j=1}^G r(o_j \mid x)$$

Full Dr. GRPO objective:
$$L_{\text{DrGRPO}} = \sum_{t=1}^{|o_i|} \min\left(\frac{\pi_\theta}{\pi_{\theta_\text{old}}} \hat{A}_{i,t}^{\text{DrGRPO}},\ \text{clip}\left(\frac{\pi_\theta}{\pi_{\theta_\text{old}}}, 1-\epsilon_l, 1+\epsilon_h\right) \hat{A}_{i,t}^{\text{DrGRPO}}\right)$$

**Additional design choices**:
- DAPO-style **asymmetric clipping** ($\epsilon_l \neq \epsilon_h$): prevents entropy collapse during RL
- **No KL-divergence regularization** (following original Dr. GRPO)

**Effect**: all scenarios contribute gradient signal proportional to absolute reward advantage, regardless of intra-group variance. Complex maneuvers (high-variance) now drive learning.

---

## Reward Functions

**Format reward** $r_f \in \{0, 0.25\}$: valid space-separated `TRAJ_XXXX` tokens in correct range [0, 2047]

**Length reward** $r_l \in \{0, 0.25\}$: correct token count (8 for NAVSIM, 10 for WaymoE2E)

**Dataset-specific reward** $r_d$:

*NAVSIM PDM Score*:
$$\text{PDMS} = \text{NC} \times \text{DAC} \times \frac{5 \cdot \text{TTC} + 2 \cdot C + 5 \cdot \text{EP}}{12}$$

*WaymoE2E Normalized RFS* (range [0,1] from scored [4,10] human trajectory similarity):
$$\text{Normalized RFS} = \frac{\max(\max_r(s_r), 4) - 4}{6}$$

Total reward: $r = r_f + r_l + r_d$ (all normalized to [0,1])

---

## Supplementary: GRPO vs. Dr. GRPO Breakdown

![[raw/assets/contour_plots.png]]

**Figure 10**: Training improvement patterns across variance groups.  
- Low-variance (a): GRPO shows improvements for initial scores in [0.8, 1.0]  
- Medium/high-variance (b,c): **Dr. GRPO dominates** with denser improvements above y=x

![[raw/assets/x4 15.png]]

**Figure 11**: Training and validation curves. Dr. GRPO (red) consistently and substantially outperforms GRPO (blue).

**Table 4: Detailed component comparison (NAVSIM test)**

| Method | PDMS | Collision | DAC | Direction | Progress | TTC | Comfort |
|--------|------|-----------|-----|-----------|----------|-----|---------|
| NoRD-base | 76.66 | 96.45 | 86.37 | 94.62 | 71.58 | 90.37 | 99.97 |
| +GRPO | 77.18 | 91.89 | 90.12 | 91.84 | 80.06 | 80.13 | 99.96 |
| **+Dr. GRPO** | **85.62** | **97.56** | **94.92** | **95.94** | **79.30** | **93.53** | **100** |

Dr. GRPO improves all sub-metrics except Ego Progress (−0.76 vs. GRPO's +8.48) — the safety-first log-sigmoid-style behavior (NC×DAC gate) penalizes risky progress attempts.

---

## Results

### NAVSIM (navtest subset)

![[raw/assets/x2 16.png]]

**Table 3**: NAVSIM test set comparison.

| Method | w/o R | w/o L | C | PDMS | Collision | DAC | Progress | TTC | Comfort |
|--------|--------|--------|---|------|-----------|-----|----------|-----|---------|
| AutoVLA | ✗ | ✓ | 12 | 89.1 | 98.4 | 95.6 | 81.9 | 98.0 | 99.9 |
| AutoVLA-BoN* | ✗ | ✓ | 12 | 92.1 | 99.1 | 97.1 | 87.6 | 97.1 | 100 |
| RecogDrive | ✗ | ✓ | 12 | 89.6 | 98.2 | 97.9 | 83.5 | 95.2 | 99.8 |
| **NoRD** | **✓** | **✓** | **3** | **85.6** | **97.6** | **94.9** | **79.3** | **93.5** | **100** |
| **NoRD-BoN*** | **✓** | **✓** | **3** | **92.4** | **99.2** | **98.3** | **86.4** | **97.8** | **99.9** |

*BoN = best-of-N oracle: best PDM score out of 6 outputs with different random seeds.

**NoRD** is the only NAVSIM entry: no reasoning traces, 3 cameras only, no LiDAR, 80K training samples.  
**NoRD-BoN (92.4)** surpasses AutoVLA-BoN (92.1) with BoN-6.

![[raw/assets/navsim_examples.png]]

**Figure 7**: NoRD safely executes sharp turns, respects traffic lights, avoids collisions. Trajectory shown in red.

### WaymoE2E

![[raw/assets/waymo_results.png]]

**Figure 8**: NoRD handles challenging OOD scenarios: unsafe pedestrian crossing, construction sites.

**Table 2**: WaymoE2E test results.

| Model | w/o Reason | w/o Ensemble | RFS | ADE@3 |
|-------|------------|--------------|-----|-------|
| Poutine | ✗ | ✓ | **7.986** | 1.2055 |
| HMVLM | ✗ | ✓ | 7.736 | 1.3269 |
| DiffusionLTF | ✓ | ✗ | 7.717 | 1.3561 |
| UniPlan | ✓ | ✗ | 7.692 | 1.3083 |
| AutoVLA | ✗ | ✓ | 7.556 | 1.3507 |
| **NoRD** | **✓** | **✓** | **7.709** | **1.2504** |

NoRD ranks 3rd on RFS (7.709) and **best ADE@3 (1.2504)** — with 6–17× fewer training samples than top competitors. Trained on only 12K SFT + 450 RLFT samples.

**Detailed WaymoE2E scores** (Table 6):

| Scenario | Score |
|----------|-------|
| Construction | 8.073 |
| Intersection | 7.925 |
| Pedestrian | 7.778 |
| Multi-Lane Maneuver | 7.826 |
| Single-Lane Maneuver | 8.309 |
| Spotlight (rare) | 6.531 |
| Average | 7.709 |
| ADE@3s | 1.250 |
| ADE@5s | 2.893 |

---

## Efficiency

![[raw/assets/nord_efficient.png]]

**Figure 9**: NoRD is the most token-efficient (a) and runtime-efficient (b) VLA.

Data efficiency summary:

| Method | Training data | NAVSIM PDMS |
|--------|--------------|-------------|
| AutoVLA | 212K+ (reasoning) | 89.1 |
| RecogDrive | ~130K (reasoning) | 89.6 |
| **NoRD** | **80K (no reasoning)** | **85.6** |

**Token count**: 3× fewer than reasoning-based VLAs (no CoT tokens at train or inference).

**Ablation: k-disc vocabulary size** (Table 5):

| Vocabulary | PDMS |
|-----------|------|
| 512 | 83.07 |
| **2048** | **85.62** |

Larger vocabulary better represents sharp turns and complex maneuvers.

---

## Limitations

1. **Dr. GRPO still imperfect**: difficulty bias is mitigated but not fully resolved; acknowledged by the authors
2. **Camera-only**: no LiDAR or HD map — performance may degrade in occlusion-heavy or complex urban scenes
3. **BoN comparison**: NoRD-BoN (92.4) requires 6 inference passes; oracle selection not applicable in real deployment
4. **Two benchmarks only**: NAVSIM and WaymoE2E — no evaluation on Bench2Drive or nuScenes; generalization to other settings unverified
5. **Small WaymoE2E training set**: 12K samples; long-tail robustness on highly diverse scenarios unclear
6. **Backbone dependency**: approach assumes Qwen-2.5VL has sufficient prior visual knowledge; may not generalize to weaker or smaller base VLMs
7. **Ego Progress regression**: Dr. GRPO trades some ego progress (−0.76 from GRPO) for safety gains — systematic safety-efficiency tradeoff observed but not fully studied

---

## Key Takeaways

1. **Reasoning annotations are not necessary** for competitive VLA performance — the bottleneck was the optimizer, not the data format
2. **GRPO difficulty bias** (std normalization attenuation) explains why weak SFT policies fail to improve under standard GRPO — this is the first identification of this failure mode in the AD domain
3. **Dr. GRPO** is a lightweight, drop-in fix: remove std from advantage → 11.68% improvement (vs. 0.67% for GRPO)
4. **Data efficiency frontier**: NoRD achieves competitive NAVSIM PDMS with 60% less data and the best WaymoE2E ADE@3 with 6–17× less data than SOTA
5. **Token efficiency**: 3× fewer tokens enables faster inference — practically important for real-time deployment

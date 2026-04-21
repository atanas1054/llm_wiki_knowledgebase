---
title: "DriveFine: Refining-Augmented Masked Diffusion VLA for Precise and Robust Driving"
type: source-summary
sources: [raw/papers/DriveFine_ Refining-Augmented Masked Diffusion VLA for Precise and Robust Driving.md]
related: [concepts/diffusion-planner.md, concepts/rl-for-ad.md, concepts/navsim-benchmark.md, concepts/discrete-flow-matching.md, sources/recogdrive.md, sources/reflectdrive.md]
created: 2026-04-05
updated: 2026-04-05
confidence: high
---

**arXiv**: https://arxiv.org/html/2602.14577v1  
**Affiliation**: Xiaomi EV + AIR  
**Benchmark**: NAVSIM v1, v2, Navhard  
**Base model**: LLaDA-8B (masked diffusion LLM)

---

## Core Thesis

Two dominant VLA planning paradigms have complementary weaknesses:

| | Diffusion-based (ReCogDrive, DiffusionDrive) | Token-based (AutoVLA, AdaThinkDrive) |
|--|--|--|
| **Strength** | Iterative multi-step refinement; can correct trajectory errors across denoising steps | Unified cross-modal token space; stronger GRPO generalization (no reward hacking) |
| **Weakness** | Weak VLM coupling → reward hacking under PDMS GRPO; loses EPDMS; slow training | Irreversible committed tokens; causal error accumulation; no self-correction |

DriveFine uses a **masked diffusion LLM** (parallel, bidirectional decoding) as the base and injects a **plug-and-play block-MoE refinement expert** to gain iterative correction without breaking the token-based training paradigm.

---

## Figure 1 — Decoding Mechanism Comparison

![[raw/assets/DriveFine_ Refining-Augmented Masked Diffusion VLA for Precise and Robust Driving/x1 10.png]]

Three paradigms:
- **(a) Diffusion**: parallel denoising across many steps
- **(b) AR token-by-token**: sequential, irreversible
- **(c) DriveFine**: generate all tokens in parallel first → single refinement step

---

## Figure 2 — PDMS-Oriented RFT Comparison

![[raw/assets/DriveFine_ Refining-Augmented Masked Diffusion VLA for Precise and Robust Driving/x2 8.png]]

**Key empirical finding**: when PDMS-oriented reinforcement fine-tuning is applied:
- **Token-based** (InternVL): PDMS ↑, EPDMS ↑ — both improve simultaneously
- **Diffusion-based** (ReCogDrive, DiffusionDrive): PDMS ↑, **EPDMS ↓** — reward hacking

This is attributed to the weak coupling between separate diffusion planners and the VLM backbone: GRPO optimizes the planner's PDMS at the expense of the planner losing generalizable patterns encoded in the pretrained LLM weights.

---

## Figure 3 — Irreversible Decoding Failures

![[raw/assets/DriveFine_ Refining-Augmented Masked Diffusion VLA for Precise and Robust Driving/x3 8.png]]

Masked diffusion decoding without refinement exacerbates irreversibility: tokens decoded early (high confidence but lacking global context) become outliers that cannot be revised, causing trajectory-level failures (collision, off-road).

---

## Architecture

### Figure 4 — DriveFine Overview

![[raw/assets/DriveFine_ Refining-Augmented Masked Diffusion VLA for Precise and Robust Driving/x4 7.png]]

### Action Tokenization

- **Spatial**: $[-100\text{m}, +100\text{m}]$ → **4,000 bins @ 0.05m** resolution (longitudinal + lateral, shared codebook)
- **Heading**: $[-90°, +90°]$ → **1,800 bins @ 0.1°/bin**
- All bins appended to LLM vocabulary; decoded as standard language tokens

### Vision Tower

SigLIP-384: single front-view image → 8 patches of 384×384 → continuous visual tokens.

### Generation Expert (Masked Diffusion LLM)

**Training** (masked cross-entropy on masked tokens):
$$\mathcal{L}_\theta = -\mathbb{E}_{t, p_0, r_0, r_t}\left[\frac{1}{t}\sum_{i=1}^L \mathbb{I}[r_t^i = [\text{M}]] \log p_\theta(r_0^i \mid p_0, r_t)\right]$$

**Inference**: start from fully masked trajectory → $s=12$ iterative unmasking steps using confidence-prioritized cosine schedule.

### Block-MoE Refinement Expert

LLaDA-8B has 32 transformer blocks. Split:
- Blocks 0–27 (28 blocks): **shared blocks** — run once, output common contextual representation
- Blocks 28–31 (4 blocks): replicated into two expert sets:
  - **Generation expert blocks**: original last 4 blocks; input = masked tokens $[\text{M}]$
  - **Refinement expert blocks**: identical copy; input = fully unmasked (committed) tokens

| Property | Value |
|----------|-------|
| Expert activation | Explicit: prompted by special task token at inference |
| Gradient isolation | Refinement gradients confined to refinement blocks only; shared + generation blocks frozen from refinement path |
| Extra parameters | n=4 blocks ≈ +1B on 8B base (9B total) |
| Extra inference time | 1 additional forward pass through 4 blocks |

**Design rationale**: generation and refinement share all lower-level contextual representations. Only the final task-specific blocks differ. This allows:
1. Pretrained foundational knowledge of LLaDA preserved completely
2. Plug-and-play insertion post-hoc (refinement expert can be added without retraining generation expert)
3. Parallel training of both experts with synchronous rollouts

**Inference pipeline**: 12 generation steps → 1 refinement step (total).

---

## Hybrid Reinforcement Learning

### Figure 5 — Hybrid RL Pipeline

![[raw/assets/DriveFine_ Refining-Augmented Masked Diffusion VLA for Precise and Robust Driving/x5 7.png]]

### GRPO for Generation Expert

Standard GRPO with G=10 group size; s-step progressive sampling with τ-step aggregation (balances alignment/efficiency):

$$\hat{A}_i = r_i - \text{mean}(\{r_i\}_{i=1}^G)$$

Reward computed in NAVSIM non-reactive simulator. 1 epoch, batch=16, lr=1×10⁻⁶.

### Hybrid Offline+Online RL for Refinement Expert

Generation expert produces G=10 candidate trajectories with rewards $\{r_i\}$. These serve as **offline anchors** for the refinement expert:

**Offline advantage matrix** (pairwise reward differences, zero mean by construction):
$$\hat{A}^{of}_{ij} = r_i - r_j, \quad \forall i,j \in G$$

Benefits: dense reward signal (vs. GRPO's group-average baseline); squared differences are always non-zero; no extra sampling.

**Online advantage** (K=6 self-refinements per anchor trajectory):
$$\hat{A}^{on}_{ik} = \hat{r}_{ik} - r_i, \quad \forall i \in G, k \in K$$

Encourages the refinement expert to actively explore beyond the offline upper bound.

**Hybrid loss**: weighted sum of offline and online policy gradient objectives.

Both experts trained **synchronously**: generator samples online → feed directly to refiner as offline data + trigger K online refinement rollouts per sample.

---

## Results

### Table 1 — NAVSIM v1 (PDMS)

| Method | Sensors | PDMS |
|--------|---------|------|
| DiffusionDrive | 3xC+L | 88.1 |
| AutoVLA | 3xC | 89.1 |
| ReCogDrive | 1xC | 89.6 |
| DriveVLA-W0 | 1xC | 90.2 |
| AdaThinkDrive | 1xC | 90.3 |
| **DriveFine** | **1xC** | **90.7** |
| **DriveFine\*** (score-based RFT) | **1xC** | **91.8** |
| DriveFine† (best-of-6) | 1xC | 94.2 |

DriveFine base (90.7): most broadly-verified single-sample single-camera result; superseded in absolute terms by WAM-Diff (91.0) and FLARE-4B RFT (91.4) — neither of which includes DriveFine in their comparison tables.  
DriveFine\*: uses an **additional trained scorer** for reward computation — comparisons with GRPO-only methods should note this.  
DriveFine†: best-of-6 multi-sample oracle, not practical for deployment.

### Table 2 — NAVSIM v2 (EPDMS)

| Method | EPDMS |
|--------|-------|
| DiffusionDrive | 84.5 |
| Senna-2 | 86.6 |
| DriveVLA-W0 | 86.5 |
| DriveFine (bugged NAVSIM) | 87.1 |
| **DriveFine† (bug-fixed NAVSIM)** | **89.7** |

⚠️ **Caveat**: DriveFine's 89.7 uses a **bug-fixed NAVSIM** scoring version (bug caused systematic underestimation). Prior methods (Senna-2 86.6, WAM-Flow 84.7) used the bugged version. Cannot directly compare without re-running baselines on the fixed scorer.

### Table 3 — Navhard (EPDMS) — OOD Benchmark

Uses **3DGS (Gaussian splatting)** to generate scenarios beyond training distribution. Two-stage evaluation.

| Method | Stage 1 EPDMS | Stage 2 EPDMS |
|--------|-------------|-------------|
| ReCogDrive | 68.9 | 37.8 |
| DiffusionDrive | 66.7 | 40.5 |
| **DriveFine** | **74.4** | **41.0** |

+5.5 EPDMS over ReCogDrive on Stage 1. Strong OOD generalization, attributed to unified token space being more robust than separately-coupled diffusion planners.

---

## Qualitative Results

### Figure 6 — Before/After Refinement

![[raw/assets/DriveFine_ Refining-Augmented Masked Diffusion VLA for Precise and Robust Driving/x6 6.png]]

Red = trajectory before refinement (generation expert output); Green = after refinement. Refinement corrects individual outlier tokens that would otherwise cause collisions or off-road driving, and smooths trajectory-level fluctuations from non-causal decoding.

### Figure 7 — PDMS-Latency Trade-off

![[raw/assets/DriveFine_ Refining-Augmented Masked Diffusion VLA for Precise and Robust Driving/x7 6.png]]

At s=4 steps: **90.47 PDMS, 207ms latency** — matching ReCogDrive-8B at much lower step count. Step count is a continuous latency-quality dial.

---

## Ablation Studies

### Table 4 — Core Component Ablation

| ID | SFT | GRPO | Offline-RFT | Online-RFT | PDMS |
|----|-----|------|------------|-----------|------|
| 1 | ✓ | ✗ | ✗ | ✗ | 86.7 |
| 2 | ✓ | ✓ | ✗ | ✗ | 89.6 (+2.9) |
| 3 | ✓ | ✓ | ✓ | ✗ | 90.3 (+0.7) |
| **4** | **✓** | **✓** | **✓** | **✓** | **90.8 (+0.5)** |

GRPO is the largest single contributor (+2.9). Refinement adds +1.2 total.

### Table 5 — PDMS/EPDMS Robustness of Token-Based VLA

| Config | PDMS | EPDMS |
|--------|------|-------|
| DriveFine-SFT | 86.7 | 86.8 |
| +RFT(PDMS) | 90.5 | **89.1** |

EPDMS improves in sync with PDMS (+2.3 EPDMS while +3.8 PDMS). Contrast with diffusion-based planners which lose EPDMS under same PDMS-oriented RFT.

### Table 6 — Refinement Block Count

| n blocks | 0 | 1 | 2 | 4 | 6 | 8 |
|---------|---|---|---|---|---|---|
| Params (B) | 8 | 8.25 | 8.5 | 9 | 9.5 | 10 |
| PDMS | 89.6 | 90.0 | 90.4 | **90.8** | 90.8 | 90.7 |

Sweet spot at n=4 (+1B params). Even n=1 (+250M params, +0.4 PDMS) is cost-effective.

### Table 7 — GRPO Group Size Sensitivity

| G | 0 | 2 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|
| PDMS | 86.7 | 88.4 | 89.1 | 89.4 | **89.6** | 89.6 |

Monotonic improvement; saturates at G=8. Even G=2 gives +1.7 PDMS over SFT.

---

## Limitations

1. **DriveFine\* uses extra scorer** — 91.8 PDMS requires a separately trained scoring module beyond the NAVSIM simulator. Not a fair comparison with pure GRPO methods.
2. **NAVSIM v2 caveat** — 89.7 EPDMS uses bug-fixed NAVSIM; prior SOTA (Senna-2 86.6) used bugged version. Direct comparison not valid without re-running baselines.
3. **Single front camera** — all results use 1xC, limiting peripheral perception vs. 3-camera/LiDAR systems.
4. **SFT data from ReCogDrive** — no original data pipeline; relies on ReCogDrive's QA pairs and textualized trajectories verbatim.
5. **One-shot refinement** — only a single refinement pass at inference; errors not caught in that pass persist.
6. **Navhard Stage 2 still low** (41.0) — severe OOD (3DGS-generated) remains hard; refinement helps less when generation quality degrades under distribution shift.

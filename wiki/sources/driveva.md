---
title: "DriveVA: Video Action Models are Zero-Shot Drivers"
type: source-summary
sources: [raw/papers/DriveVA_ Video Action Models are Zero-Shot Drivers.md]
related: [concepts/navsim-benchmark.md, concepts/world-model-for-ad.md, concepts/diffusion-planner.md, concepts/vlm-domain-adaptation.md]
created: 2026-04-23
updated: 2026-04-23
confidence: medium
---

## One-Line Summary

Unified video-action world model fine-tuned from Wan2.2-TI2V-5B; single DiT jointly denoises future video latents + action tokens; **90.9 PDMS** NAVSIM-v1; strong zero-shot transfer to nuScenes (−78.9% L2) and Bench2Drive (−52.5% L2) vs. prior world-model SOTA.

**arXiv**: 2604.04198v1  
**Org**: University of Twente + (multiple affiliations)  
**Confidence**: medium — Table 1 (NAVSIM comparison sub-scores) is truncated in the source file; per-metric NC/DAC/EP/TTC/C breakdown unavailable.

---

## Context and Motivation

Two gaps motivate DriveVA:

**Gap 1 — Limited generalization of VLMs**: VLMs pretrained on image-text pairs learn semantic knowledge ("what is what") but lack spatiotemporal priors ("how the world moves"). This limits zero-shot transfer across datasets and sensor domains. Large-scale video generation models, trained on web-scale video, encode realistic motion patterns and physically plausible scene dynamics — richer priors for robust driving.

**Gap 2 — Loose video-trajectory coupling**: existing world-model-based planners treat video prediction and action generation as separate or loosely coupled stages:
- *Cascaded*: video generated first, trajectory conditioned on it (or vice versa) — UniUGP, DriveDreamer-Policy
- *Auxiliary*: video prediction as training signal only, bypassed at inference — DriveVLA-W0, FLARE, Vega
- *Parallel branches*: separate decoders on shared latent — Epona (TrajDiT ‖ VisDiT)

Mismatches between imagined futures and executed actions accumulate over time. DriveVA's answer: place future video latents and action tokens in the **same noisy generative target** and denoise them jointly with a single DiT.

---

## Architecture: DriveVA

![[x1 24.png|DriveVA overview]]
*Figure 1: Unified video–trajectory rollout. History frames → rollout future video clip (top) + ego trajectory aligned with visual evolution (middle). Bottom: zero-shot comparisons on nuScenes and Bench2Drive.*

![[x2 22.png|Overall pipeline]]
*Figure 2: Overall pipeline. History obs + ego state + language instruction → text encoder + video VAE → unified DiT predicts future video latents and action tokens jointly → video continuation module for long-horizon consistency.*

### Backbone: Wan2.2-TI2V-5B

DriveVA builds on **Wan2.2-TI2V-5B** (a 5B-parameter text-to-image-to-video generation model) — the same backbone family used in DriveDreamer-Policy (Wan-2.1-T2V-1.3B) and UniUGP (Wan2.1), but at larger scale and with the text-to-image-to-video variant for conditioning on a reference frame.

Components inherited:
- **3D-causal VAE**: encodes video frames into latent tokens with temporal downsampling; causality ensures single frame → single latent (efficient history conditioning)
- **Frozen text encoder**: language instruction → fixed-length token sequence, injected via cross-attention (not concatenated — keeps spatiotemporal token count compact)

### Problem Formulation

At timestep $l$, given:
- Language instruction $\mathcal{T}$ (high-level command)
- History observation buffer $\mathcal{O}_l = \{F_{l-m+1}, \ldots, F_l\}$ (m frames)
- Ego state $q_l$ (velocity $v_x, v_y$)

Jointly predict:
1. **Action chunk** $\mathcal{A}_{l+1:l+K}$: K future $(x, y, \text{yaw})$ 3-vectors
2. **Future video clip** $\mathcal{F}_{l+1:l+N}$: N frames encoded as latent sequence

### Unified DiT Decoder

Input is split into **condition block** (noise-free) and **generative target block** (jointly denoised):

$$\mathbf{X}_\text{cond}^{(l)} = [\mathbf{S}_l,\ \mathbf{V}'_{l-m+1}, \ldots, \mathbf{V}'_l] \qquad \text{(ego state + history latents)}$$

$$\mathbf{Y}_0^{(l)} = [\mathbf{V}'_{l+1}, \ldots, \mathbf{V}'_{l+n_\text{pred}},\ \mathbf{A}_{l+1:l+K}] \qquad \text{(future video latents + action tokens)}$$

A single DiT $f_\theta$ predicts the conditional velocity field:

$$\hat{\mathbf{v}}_\theta^{(l,s)} = f_\theta\!\left([\mathbf{X}_\text{cond}^{(l)}, \mathbf{Y}^{(l,s)}],\ s\ \mid \mathbf{T}\right)$$

where $\mathbf{Y}^{(l,s)} = (1-s)\boldsymbol{\epsilon} + s\mathbf{Y}_0^{(l)}$ is the noisy interpolation at flow time $s$. The flow-matching loss:

$$\mathcal{L}_\text{FM} = \mathbb{E}_{l,s,\mathbf{Y}_0,\boldsymbol{\epsilon}}\left[\left\|\hat{\mathbf{v}}_\theta^{(l,s)} - (\mathbf{Y}_0^{(l)} - \boldsymbol{\epsilon})\right\|_2^2\right]$$

**Key coupling**: video latents and action tokens are in the same noisy $\mathbf{Y}^{(l,s)}$ at the same flow time — both denoised by the same network in a single pass. This is structurally deeper coupling than separate branches (Epona's TrajDiT ‖ VisDiT) or sequential stages.

### Video Continuation Module

History conditioning: the 3D-causal VAE encodes the full observation buffer $\mathcal{O}_l$ into history latents $\mathcal{V}^\text{his}_l = \{V_{l-m+1}, \ldots, V_l\}$. These condition the future prediction directly (not just the first frame), providing long-range visual priors.

Rolling inference: execute $\mathcal{A}_{l+1:l+K}$ → receive new observations → slide history window → repeat. Each step is a short video-continuation problem. This converts unbounded long-horizon prediction into a progressive sequence of bounded short predictions.

**Inference efficiency**: as few as **2 sampling steps** (flow-matching integration steps) reach near-optimal closed-loop performance — enabling efficient recurrent decision-making despite the generative architecture.

### Tokenization Details

| Token type | Encoding | Sequence length |
|---|---|---|
| History video frame $V_t$ | Raster-flatten → linear projection | $h \cdot w$ per frame |
| Ego state $q_l$ | MLP → $L_S$ tokens | $L_S$ |
| Future video latent $V_{l+j}$ | Same as history (generative target) | $h \cdot w \cdot n_\text{pred}$ |
| Action $a_{l+i}$ ∈ ℝ³ | MLP → 1 token | K total |
| Language instruction $\mathcal{T}$ | Frozen text encoder → cross-attention | $L_T$ (decoupled) |

---

## Key Empirical Finding: Video Supervision is the Main Driver

The most important quantitative result in the paper (Table 5.5, from text):

| Training objective | PDMS |
|---|---|
| Action-only (no video supervision) | 71.4 |
| **+ Joint video supervision (DriveVA)** | **90.9** |
| **Gain** | **+19.5** |

A +19.5 PDMS gain from adding video supervision is the **largest single-component gain** reported in any wiki paper. This directly validates the world-model hypothesis: planning benefits primarily from video-level dense supervision that forces action-video consistency, not from architecture choices or data augmentation.

The authors argue this is because:
- Video supervision provides dense temporal grounding of scene dynamics at every timestep
- The gain requires actions to be **consistent** with the imagined future — loose coupling (auxiliary signal) does not achieve the same result
- This distinguishes joint generation (DriveVA) from auxiliary video prediction (DriveVLA-W0, FLARE, Vega)

---

## Main Results

### NAVSIM v1 (Table 1 — partially truncated)

**90.9 PDMS** claimed. The comparison table header is present in the source file but the data rows are truncated — per-metric sub-scores (NC, DAC, EP, TTC, C) are unavailable from this source.

The paper groups methods as "Traditional End-to-End" vs. "World Model Methods" and claims SOTA within the world model category. At 90.9 PDMS, DriveVA slots in the wiki between WAM-Diff (91.0) and DriveFine (90.7) — not overall SOTA (DriveSuprim 93.5), but competitive among VLM and world-model approaches.

### Zero-Shot Generalization (nuScenes, zero-shot from NAVSIM training)

| Metric | PWM (prior SOTA WM) | DriveVA | Improvement |
|---|---|---|---|
| Avg L2 error ↓ | — | — | **−78.9%** |
| Collision rate ↓ | — | — | **−83.3%** |

Absolute numbers for PWM and DriveVA are not available in the source file (table truncated). The comparison is against PWM specifically — not against VLA methods or full nuScenes SOTA.

### Zero-Shot to Bench2Drive (real→simulation, NAVSIM-trained model on CARLA)

| Metric | PWM | DriveVA | Improvement |
|---|---|---|---|
| Avg L2 error ↓ | — | — | **−52.5%** |
| Collision rate ↓ | — | — | **−52.4%** |

Same caveat: absolute numbers missing; comparison is against PWM only.

![[x3 22.png|NAVSIM comparison table]]
*Figure/Table 3: NAVSIM comparison table — data rows truncated in source file.*

---

## Contrast with Related Wiki Methods

| Aspect | Epona (P2) | DDP (P6) | DriveVLA-W0 (P7) | FLARE (P8) | **DriveVA (P11)** |
|---|---|---|---|---|---|
| Backbone | From-scratch AR+DiT | Wan-2.1-1.3B | Emu3/Qwen2.5-VL | DINOv2+DiT | **Wan2.2-5B** |
| Video-action coupling | Parallel branches (shared F) | Causal stages (depth→video→action) | Training only | Auxiliary loss only | **Joint target, one DiT** |
| Video at inference | Optional VisDiT | Optional depth/video | ✗ | ✗ | ✓ (required) |
| Zero-shot eval | ✗ | ✗ | ✗ | ✗ | ✓ nuScenes + B2D |
| NAVSIM-v1 PDMS | 86.2 | 89.2 | 90.2★ | 91.4 | **90.9** |
| No. params | 2.5B | ~2B (Qwen3-VL) | 7–8B | 4B | **5B** |
| RL stage | ✗ | ✗ | ✗ | ✓ GRPO | ✗ |

★ DriveVLA-W0's 90.2 uses trajectory anchors (multi-candidate selection); single-pass = 88.4.

DriveVA's strongest differentiator is **zero-shot generalization**, which no other wiki world-model paper demonstrates quantitatively at this scale.

---

## Limitations

1. **Table 1 truncated**: NAVSIM sub-scores (NC, DAC, EP, TTC, C) unavailable from source. Cannot verify which specific methods were compared against, or whether the comparison includes DriveSuprim (93.5), HybridDriveVLA (92.1), FLARE (91.4), or DiffusionDriveV2 (91.2).
2. **Zero-shot comparison vs. PWM only**: the zero-shot nuScenes and Bench2Drive improvements are reported relative to PWM (a specific world-model planner), not against full SOTA on those benchmarks. The percentage improvements are impressive but not verifiable without absolute numbers.
3. **No NAVSIM-v2 / EPDMS**: only v1 PDMS reported. Cannot compare with v2 leaders (WAM-Diff 89.7, DreamerAD 87.7, DriveDreamer-Policy 88.7).
4. **5B backbone**: Wan2.2-TI2V-5B is substantially larger than most wiki methods. No latency or throughput numbers reported.
5. **No RL**: purely SFT + flow matching fine-tuning. Most competitive wiki methods (FLARE 91.4, DriveFine 90.7, WAM-Diff 91.0) include RL stages.
6. **Video at inference is required**: unlike Epona (optional VisDiT) or DriveVLA-W0 (no video at inference), DriveVA requires joint video generation at every step — may be computationally prohibitive for real-time deployment.
7. **Reality gap for Bench2Drive**: testing a real-driving-trained model on CARLA simulation is challenged by known appearance, dynamics, and agent behavior gaps.

---

## Position in the Wiki

DriveVA (90.9 PDMS) slots near DriveFine (90.7) and below WAM-Diff (91.0) and FLARE (91.4) on NAVSIM-v1, but its primary contribution is **zero-shot generalization** — a dimension not measured by NAVSIM PDMS. The +19.5 PDMS gain from joint video supervision is the strongest quantitative argument for tightly-coupled video-action generation in the wiki.

The critical architectural distinction from other world-model methods in the wiki is the **single shared generative process**: video latents and action tokens are noised and denoised *together* by one DiT, rather than by separate modules conditioned on shared intermediate representations.

See [[concepts/world-model-for-ad.md]] (Pattern 11) for paradigm context and comparison with all other world-model approaches.

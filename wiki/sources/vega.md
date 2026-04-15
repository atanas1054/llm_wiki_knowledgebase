---
title: "Vega: Learning to Drive with Natural Language Instructions"
type: source-summary
sources: [raw/papers/Vega_ Learning to Drive with Natural Language Instructions.md]
related: [concepts/world-model-for-ad.md, concepts/vlm-domain-adaptation.md, concepts/navsim-benchmark.md, concepts/diffusion-planner.md]
created: 2026-04-08
updated: 2026-04-08
confidence: high
---

## Citation

Zuo, S., Li, Y., Zheng, W., Zhu, Z., Zhou, J., Lu, J. — *Vega: Learning to Drive with Natural Language Instructions*. arXiv:2603.25741v1 (2026). Tsinghua University + GigaAI. Code: https://github.com/zuosc19/Vega

## One-line Summary

Vega is the first unified vision-language-**world**-action model for instruction-based AD: it jointly generates instruction-compliant future images and trajectories from natural language commands using an integrated AR+diffusion transformer, with world modeling as a dense supervision bridge that enables instruction-following — achieving 86.9 EPDMS / 89.4 BoN-6 on NAVSIM-v2.

## Problem Addressed

**Current VLAs follow averaged expert policies or a closed set of simple commands** ("turn left", "go straight"), failing on open-ended NL instructions like "overtake the front car to catch the next green light." A naive fix — fine-tuning a VLA on instruction-annotated data — achieves only ~60 PDMS (Qwen2.5-VL + planning head baseline). Root cause: **sparse action supervision** cannot bridge the information gap between high-dimensional visual+instruction inputs and low-dimensional trajectory outputs.

Vega addresses this via **world modeling as dense supervision**: future image generation compels the model to learn the causal chain instruction → action → visual outcome.

## InstructScene Dataset

100K instruction-annotated driving scenes built on NAVSIM (85K train, 12K test). Two-stage automated annotation pipeline:

**Stage 1 — Scene Understanding (Qwen2.5-VL-72B)**:
- Input: 4 current + 10 future frames at 2Hz, 1920×1080 front-view
- Output: scene description (traffic participants, static objects) + driving behavior narration (actions + interactions in next 10 frames)

**Stage 2 — Instruction Formulation (Qwen2.5-VL-72B)**:
- Input: visual frames + Stage 1 descriptions
- Output: concise high-level driving instruction $L_t$ aligning with the actions taken

**Rule-based supplement**: speed/acceleration/turn-rate thresholds → NL ego-motion labels. VLMs are inaccurate at perceiving ego motion numerically; rule-based labels provide precise motion cues. Combined with VLM-generated instructions as auxiliary prompts to produce accurate + diverse instructions.

## Architecture

![Figure 2: Vega overview](../../../raw/assets/x3%2017.png)

*Figure 3: Integrated AR+Diffusion transformer. Understanding (AR) and generation (diffusion) transformers share joint attention while using separate MoT parameter sets.*

### Encoding Multi-modal Inputs

| Modality | Encoder | Output |
|----------|---------|--------|
| Text instructions $L_t$ | Qwen2.5 tokenizer | Token sequence |
| Images $I_t$ (visual understanding) | VAE encoder | Latent $F_t^V$ |
| Images $I_t$ (visual richness) | SigLIP2 ViT | ViT features appended to $F_t^V$ |
| Actions $A_t$ | Linear head (after relative conversion) | $\Delta x, \Delta y, \Delta\theta$ per step |

Actions are converted from absolute 2D trajectory to **relative movements** between consecutive steps, normalizing across timesteps for a shared distribution.

### Interleaved Sequence and Causal Mask

Unified sequence $S = [I_0, L_0, A_0, \ldots, I_n, L_n, A_n]$ with a **blocked lower-triangular** causal attention mask:
- Each block (image / action / instruction) attends only to prior blocks
- Within text: strictly lower-triangular + consecutive RoPE indices
- Within image/action: full attention + shared RoPE index + sinusoidal position

**Duplicate latent trick** (key for joint training): each latent that acts as both prediction target and conditioning input is duplicated:
- $F_t^{noisy}$: receives diffusion noise → used for denoising loss
- $F_t^{clean}$: unperturbed → used as condition for subsequent tokens

$F_t^{noisy}$ is masked from later tokens. This resolves the training/inference mismatch (at inference, action is denoised first, then image is denoised conditioned on clean action).

### Integrated Transformer (MoT — Mixture of Transformers)

Unlike MoE (which only separates FFN), **all** transformer parameters (attention + FFN) are duplicated per modality/capability:

| Module | Function | Hidden size | Init |
|--------|----------|-------------|------|
| Understanding transformer | Visual + text understanding | 3584, 28 layers | Bagel-7B |
| Generation transformer | Image generation | 3584, 28 layers | Bagel-7B |
| Action expert | Trajectory planning | **256** | Random |

Action expert uses 256-dim hidden to reduce computation — confirmed by ablation to outperform using VLM or diffusion modules directly (Table 4).

During forward pass: interleaved sequence split into segments → respective modules → reassembled for global causal attention → extracted per-modality for output heads.

### Training and Inference

**Training** — joint objective:
$$\mathcal{L}_{pretrain} = \lambda_A \mathcal{L}_A + \lambda_V \mathcal{L}_V, \quad \lambda_A = \lambda_V = 1.0$$

- $\mathcal{L}_A$: MSE of denoised relative action (flow matching, random noise timestep $m$)
- $\mathcal{L}_V$: MSE of denoised VAE latents of future image at step $t+K$ (conditioned on clean action)

**Classifier-Free Guidance (CFG)**: randomly drop text / ViT / clean VAE / clean action tokens during training → allows inference-time guidance strength tuning.

**Inference**: sequential denoising — action first (DDIM), then future image conditioned on denoised action. CFG applied with both image and text guidance enabled.

**Training setup**: 200K steps, 8×H20 GPUs, 4 historical images, 8 future action steps, lr=2e-5 with 2500 warmup, per-device batch=1, EMA decay=0.9999.

## Why World Modeling Bridges the Instruction Gap

Critical ablation (Table 3, NAVSIM-v1 PDMS / NAVSIM-v2 EPDMS):

| Setting | PDMS | EPDMS |
|---------|------|-------|
| Action only (no image prediction) | 51.8 | 48.9 |
| Random frame prediction | 77.3 | 75.2 |
| **Next frame prediction (default)** | **77.9** | **76.0** |

**Action-only catastrophically fails** (51.8 PDMS) — the sparse trajectory signal alone cannot ground instruction-following in complex scenes. Future frame prediction provides a dense pixel-level supervision that forces the model to learn instruction → action → visual outcome causality.

**Frame choice has minimal impact** (77.3 vs. 77.9): what matters is the dense supervision structure, not the specific future frame. This is consistent with DriveVLA-W0's finding that world modeling benefit is primarily representational.

**Baseline VLA comparison**: a Qwen2.5-VL + planning head trained on InstructScene achieves only ~60 PDMS despite identical instruction data — confirming world modeling (not data) is the key driver.

## Key Empirical Results

### NAVSIM-v2 (EPDMS) — Table 1

| Method | NC↑ | DAC↑ | DDC↑ | TLC↑ | EP↑ | TTC↑ | LK↑ | HC↑ | EC↑ | **EPDMS↑** |
|--------|-----|------|------|------|-----|------|-----|-----|-----|-----------|
| DiffusionDrive | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | **87.7** | 84.5 |
| DriveVLA-W0 | 98.5 | 99.1 | 98.0 | 99.7 | 86.4 | 98.1 | 93.2 | 97.9 | 58.9 | 86.1 |
| **Vega** | **98.9** | **95.3** | **99.4** | **99.9** | **87.0** | **98.4** | **96.5** | **98.3** | 76.3 | **86.9** |
| **Vega BoN-6** | **99.2** | **96.6** | **99.5** | **99.9** | **87.5** | **98.7** | **97.4** | **98.4** | **84.5** | **89.4** |

Vega BoN-6 achieves best NC, DDC, TLC, TTC, LK, HC — metrics aligned with instruction compliance. EC = 76.3 (single) / 84.5 (BoN) — below DiffusionDrive (87.7).

**Caveat**: comparison table only includes DriveVLA-W0 (86.1) as the VLA baseline; DriveDreamer-Policy (88.7), Senna-2 (86.6), FLARE (86.3), DreamerAD (87.7) not compared directly.

### NAVSIM-v1 (PDMS) — Table 2

| Method | Sensors | NC↑ | DAC↑ | TTC↑ | Comf↑ | EP↑ | **PDMS↑** |
|--------|---------|-----|------|------|-------|-----|---------|
| AutoVLA | 3x Cam | 98.4 | 95.6 | 98.0 | 99.9 | 81.9 | 89.1 |
| ReCogDrive | 3x Cam | 98.2 | 97.8 | 95.2 | 99.8 | 83.5 | 89.6 |
| AutoVLA BoN-6 | 3x Cam | 99.1 | 97.1 | 97.1 | 100.0 | 87.6 | 92.1 |
| DriveVLA-W0 BoN-6 | 1x Cam | 99.3 | 97.4 | 97.0 | 99.9 | 88.3 | 93.0 |
| **Vega** | **1x Cam** | **98.9** | **95.3** | **96.1** | **100.0** | **81.6** | **87.9** |
| **Vega BoN-6** | **1x Cam** | **99.2** | **96.6** | **96.9** | **100.0** | **83.4** | **89.8** |

Single-sample 87.9 PDMS is solid for 1-camera SFT-only. BoN-6 89.8 competitive but below DriveVLA-W0 BoN-6 (93.0) and AutoVLA BoN-6 (92.1).

### Ablation — Action Expert (Table 4)

| Setting | PDMS | EPDMS |
|---------|------|-------|
| Diffusion module as action planner | 19.7 | 19.6 |
| VLM module as action planner | 77.6 | 75.7 |
| **Action expert (hidden=256)** | **77.9** | **76.0** |

Diffusion module catastrophically fails as an action planner — the diffusion process is not suited for generating compact trajectory representations. The lightweight action expert slightly outperforms the full VLM module while using ~14× smaller hidden size.

### Instruction-Following Qualitative Results

![Figure 5: Instruction-based planning examples](../../../raw/assets/x5%2016.png)

*Figure 5: Same scenario, different instructions → different trajectories. Front-camera + BEV overlay. Vega successfully modulates speed up/down/maintain to follow instructions.*

![Figure 6: Future image generation conditioned on instructions](../../../raw/assets/x6%2014.png)

*Figure 6: Instruction-conditioned generation at an intersection. Two different instruction sets produce both different action sequences and different future image predictions — both action and image are mutually consistent.*

Key qualitative capability: at intersection scenarios where multiple valid behaviors exist, Vega produces both different trajectories AND different corresponding future images for different instructions — demonstrating genuine instruction-causal world modeling.

## Contributions

1. **InstructScene**: first large-scale instruction-annotated driving dataset (100K scenes, NAVSIM-based, automated two-stage VLM annotation)

2. **Instructional driving paradigm**: explicit formalization of instruction-conditioned AD — separate from imitation driving and command-following

3. **Integrated AR+Diffusion transformer**: unified architecture for NL understanding, world modeling, and action planning with MoT and duplicate latent trick

4. **Dense supervision via world modeling**: empirically demonstrates that future image prediction resolves the instruction-to-action bridging problem — Action-only SFT fails (51.8), world modeling enables it (77.9→86.9 EPDMS with CFG+BoN)

5. **Multi-trajectory generation**: same scene → diverse valid trajectories conditioned on different NL instructions — first in the wiki

## Limitations

1. **Single front-view camera only**: no side/rear context; may miss agents entering from sides during lane changes or merges

2. **No RL/GRPO fine-tuning**: single-sample 87.9 PDMS is below FLARE (91.4), DriveFine (90.7), WAM-Flow (90.3). Authors note RL can be added orthogonally — instruction-following capability is an independent dimension

3. **BoN-6 comparison gap**: Vega BoN-6 (89.4 EPDMS) likely surpasses DriveDreamer-Policy (88.7) but no direct comparison in the table; only DriveVLA-W0 (86.1) appears as the VLA baseline

4. **EC = 76.3 (single)**: extended comfort below DiffusionDrive (87.7) and FLARE (87.5) — consistent with diffusion-based planners producing slightly more aggressive trajectories; BoN-6 recovers to 84.5

5. **Instruction noise**: automated annotation can produce imprecise or ambiguous instructions; rule-based supplements have limited diversity (closed-set motion categories)

6. **World model supervision is structural, not content-specific**: random frame ≈ next frame in performance — the benefit comes from dense prediction task structure, not from predicting the "right" future moment

7. **Heavy action ablation**: using the diffusion module for actions catastrophically fails (19.7 PDMS) — the architecture requires the dedicated lightweight action expert; not plug-and-play

## Position in the Wiki

| Aspect | Vega | Prior best |
|--------|------|-----------|
| Instruction following | ✓ (open-ended NL) | ✗ (most limited to commands) |
| World model + instruction | ✓ | ✗ |
| NAVSIM-v2 BoN-6 | **89.4** | 88.7 (DDP) |
| NAVSIM-v2 single | 86.9 | 88.7 (DDP) |
| NAVSIM-v1 single | 87.9 | 91.4 (FLARE) |
| Cameras | 1 (front) | 1–3 |
| RL stage | ✗ | ✓ (most SOTA methods) |

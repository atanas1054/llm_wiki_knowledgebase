---
title: "WAM-Flow: Parallel Coarse-to-Fine Motion Planning via Discrete Flow Matching for Autonomous Driving"
type: source-summary
sources: [raw/papers/WAM-Flow_ Parallel Coarse-to-Fine Motion Planning via Discrete Flow Matching for Autonomous Driving.md]
related: [concepts/discrete-flow-matching.md, concepts/rl-for-ad.md, concepts/diffusion-planner.md, concepts/navsim-benchmark.md, sources/recogdrive.md]
created: 2026-04-05
updated: 2026-04-05
confidence: high
---

## Citation

Yifang Xu, Jiahao Cui, Feipeng Cai, et al. (Fudan University + Yinwang Intelligent Technology)
arXiv: https://arxiv.org/abs/2512.06112
Code: https://github.com/fudan-generative-vision/WAM-Flow

## One-Line Summary

A VLA model that casts ego-trajectory planning as **discrete flow matching over a structured token space**, enabling fully parallel coarse-to-fine refinement with a tunable compute–accuracy tradeoff — 90.3 PDMS SOTA on NAVSIM-v1 using only 1 front camera.

## Problem Statement

Existing VLA planners split into two camps:
- **Autoregressive** (EMMA, DrivingGPT, AutoVLA): sequential token-by-token decoding, slow, exposure-bias accumulation
- **Continuous diffusion** (DiffusionDrive, Artemis, DiffVLA): parallel sampling but continuous Gaussian processes, no native coarse-to-fine control

WAM-Flow explores a third paradigm: **Discrete Flow Matching (DFM)** over tokenized trajectories, which offers parallel generation, bidirectional refinement, and a clean step-count knob for compute–accuracy tradeoff.

## Architecture

![Figure 1: WAM-Flow architecture — front-view image + navigation command + ego state → parallel DFM denoising → 8 waypoints over 4 seconds. The model is first trained via SFT, then aligned via simulator-guided GRPO with safety, ego-progress, and comfort rewards.](<../../raw/assets/Figure_2.png>)

### Backbone
Janus-1.5B (DeepSeek's autoregressive VLM), converted to a non-causal flow model through a multi-stage training curriculum.

### Inputs
- 1× front-view camera (384×384, encoded by SigLIP into 576 visual tokens)
- Natural-language navigation command
- Ego-vehicle state: position, heading, velocity, acceleration

### Output
8 waypoints spanning 4 seconds (x, y, heading × 8 = 24 scalar tokens).

### Metric-Aligned Numerical Tokenizer
Standard text token embeddings are poor for continuous scalars. WAM-Flow introduces:
- Codebook of **20,001 tokens** covering [-100, 100] at 0.01 resolution
- L2-normalized linear projection embeddings
- **Triplet-margin ranking loss**: for any triplet (i, j, k) where |v_i − v_j| < |v_i − v_k|, enforce d_ij < d_ik via:
  $$\mathcal{L}_\text{num} = \mathbb{E}[\max(0, d_{ij} - d_{ik} + \alpha)]$$

This ensures latent distances reflect scalar proximity, enabling stable coarse-to-fine refinement.

### Discrete Flow Matching Objective
Defines a Gibbs conditional probability path:
$$p_t(x|x_1) = \text{softmax}(-\beta_t \cdot d(x, x_1))$$

where $d$ is a weighted coordinate-wise distance mixing tokenizer-induced distances (for scalars), circular metrics (for angles), and semantic distances (for text). A CTMC with rate:
$$u_t(x, z | x_1) = p_t(x|x_1) \dot{\beta}_t [d(z, x_1) - d(x, x_1)]_+$$

steers tokens toward the target. Training minimizes cross-entropy loss against the true posterior.

### Simulator-Guided GRPO
Composite reward decomposes PDMS into:
- **Safety gates** (multiplicative): No-Collision (NC), Drivable Area Compliance (DAC)
- **Performance objectives** (weighted average): Ego Progress (EP), Time-to-Collision (TTC), Comfort (C)
  - Optimal weights: EP:TTC:C = 5:5:2
- Optimal group size: G = 3

GRPO is applied to DFM with parallel generation preserved during RL — a novel combination.

## Four-Stage Training Curriculum

![Figure 2: Full training curriculum — Stage 1 initializes numerical embeddings; Stage 2 enhances driving scene perception via large-scale VQA; Stage 3 SFTs on trajectory data; Stage 4 applies simulator-guided GRPO.](<../../raw/assets/Figure_3.png>)

| Stage | What | Data | Epochs | Params |
|-------|------|------|--------|--------|
| 1 | Numerical embedding warmup | nuPlan 668K | 4 | 0.4B |
| 2 | VQA domain pretraining | 6.5M VQA (3.4M general + 3.1M RecogDrive driving) | 3 | 1.5B |
| 3 | Supervised fine-tuning | nuPlan 668K | 2 | 1.5B |
| 4 | GRPO RL alignment | NAVSIM 103K | 0.5 | 1.5B |

Hardware: 4 × 8 Ascend 910B NPUs.

## Results

### NAVSIM-v1 Full Comparison (Table 1)

| Method | Paradigm | Backbone | Input | NC ↑ | DAC ↑ | TTC ↑ | Comf. ↑ | EP ↑ | PDMS ↑ |
|--------|----------|----------|-------|------|-------|-------|---------|------|--------|
| VADv2 | — | — | 6× Cam | 97.2 | 89.1 | 91.6 | 100 | 76.0 | 80.9 |
| Transfuser | — | — | 3× Cam + LiDAR | 97.7 | 92.8 | 92.8 | 100 | 79.2 | 84.0 |
| Hydra-MDP++ | — | — | 3× Cam + LiDAR | 98.3 | 96.0 | 94.6 | 100 | 78.7 | 86.5 |
| Artemis | Diff. | — | 6× Cam | 98.3 | 95.1 | 94.3 | 99.8 | 81.4 | 87.0 |
| DiffusionDrive | Diff. | — | 3× Cam + LiDAR | 98.2 | 96.0 | 94.8 | 100 | 82.2 | 88.1 |
| DrivingGPT | AR | LLaMA2-7B | 1× Cam | 98.1 | 90.7 | 94.9 | 95.6 | 79.7 | 82.4 |
| FSDrive | AR | Qwen2-VL-2B | 6× Cam | 98.2 | 93.8 | 93.3 | 99.9 | 80.1 | 85.1 |
| Epona | AR+Diff. | DiT-2.5B | 1× Cam | 97.9 | 95.1 | 93.8 | 99.9 | 80.4 | 86.2 |
| AutoVLA | AR | Qwen2.5-3B | 3× Cam | 98.4 | 95.6 | 98.0 | 99.9 | 81.9 | 89.1 |
| ReCogDrive | AR+Diff. | InternVL3-8B | 3× Cam | 98.2 | 97.5 | 95.2 | 99.9 | 83.5 | 89.6 |
| **WAM-Flow** | **DFM** | **Janus-1.5B** | **1× Cam** | **99.2** | **98.3** | **97.0** | **99.7** | **82.3** | **90.3** |

### NAVSIM-v2 (Table 4)

| Method | NC ↑ | DAC ↑ | DDC ↑ | TLC ↑ | EP ↑ | TTC ↑ | LK ↑ | HC ↑ | EC ↑ | EPDMS ↑ |
|--------|------|-------|-------|-------|------|-------|------|------|------|---------|
| Ego Status | 93.1 | 77.9 | 92.7 | 99.6 | 86.0 | 91.5 | 89.4 | 98.3 | 85.4 | 64.0 |
| VADv2 | 97.3 | 91.7 | 98.2 | 99.7 | 77.6 | 92.7 | 66.0 | 98.3 | 83.3 | 76.6 |
| TransFuser | 97.7 | 92.8 | 98.3 | 99.7 | 79.2 | 92.8 | 67.6 | 98.3 | 87.2 | 77.8 |
| HydraMDP++ | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | 81.4 |
| Artemis | 98.3 | 95.1 | 98.6 | 99.8 | 81.5 | 97.4 | 96.5 | 98.3 | — | 83.1 |
| RecogDrive | 98.3 | 94.2 | 98.8 | 99.8 | 86.5 | 97.3 | 96.8 | 98.3 | 87.7 | 83.6 |
| **WAM-Flow** | **98.5** | **94.5** | **99.5** | **99.8** | **86.9** | **96.8** | **97.4** | **97.6** | **73.9** | **84.7** |

Note: WAM-Flow leads EPDMS overall but has a significant EC (Extended Comfort) gap vs. RecogDrive (73.9 vs 87.7).

### Inference Speed vs PDMS — Coarse-to-Fine (Table 6)

| Method | Paradigm | Steps | PDMS ↑ | Infer Time ↓ |
|--------|----------|-------|--------|-------------|
| FSDrive | AR | — | 85.1 | 10.58s |
| Epona | AR+Diff. | — | 86.2 | 1.24s |
| ReCogDrive | AR+Diff. | — | 89.6 | 0.42s |
| Janus-1.5B (AR baseline) | AR | — | — | 0.27s |
| WAM-Flow | DFM | 1 | 89.1 | 0.09s |
| WAM-Flow | DFM | 2 | 89.7 | 0.19s |
| WAM-Flow | DFM | 3 | 90.0 | 0.29s |
| **WAM-Flow** | **DFM** | **5** | **90.3** | **0.48s** |
| WAM-Flow | DFM | 10 | 90.2 | 0.94s |

3× faster than Janus AR baseline at 1 step (0.09s vs 0.27s).

### nuScenes Collision Rate (Table 7, selected)

| Method | ST-P3 Avg ↓ | UniAD Avg ↓ |
|--------|------------|------------|
| UniAD | 0.12 | 0.31 |
| AutoVLA | 0.20 | 0.31 |
| GPT-Driver | 0.17 | 0.44 |
| **WAM-Flow** | **0.12** | **0.23** |

Lowest average collision rate among all VLA methods under UniAD metrics. Perfect 0.00% at 1-second horizon.

### Ablation — Component Contributions (Table 5)

| Num. Tokenizer | Metric-aligned | Pretraining | SG GRPO | NC ↑ | DAC ↑ | TTC ↑ | Comf. ↑ | EP ↑ | PDMS ↑ |
|:-:|:-:|:-:|:-:|------|-------|-------|---------|------|--------|
| ✗ | ✗ | ✗ | ✗ | 95.8 | 87.5 | 88.6 | 99.5 | 71.7 | 76.2 |
| ✓ | ✗ | ✗ | ✗ | 97.0 | 91.3 | 91.0 | 98.9 | 76.4 | 81.1 |
| ✓ | ✓ | ✗ | ✗ | 97.4 | 92.6 | 95.3 | 99.3 | 77.5 | 83.4 |
| ✓ | ✓ | ✓ | ✗ | 98.5 | 95.1 | 94.4 | 99.5 | 81.8 | 86.7 |
| ✓ | ✓ | ✗ | ✓ | 98.4 | 96.1 | 95.3 | 99.5 | 79.3 | 86.9 |
| ✓ | ✓ | ✓ | ✓ | **99.2** | **98.3** | **97.0** | **99.7** | **82.3** | **90.3** |

### Ablation — GRPO Group Size (Table 2)

| Group Size | NC ↑ | DAC ↑ | TTC ↑ | Comf. ↑ | EP ↑ | PDMS ↑ |
|-----------|------|-------|-------|---------|------|--------|
| w/o GRPO | 98.5 | 95.1 | 94.4 | 99.5 | 81.8 | 86.7 |
| 2 | 99.4 | 97.3 | 96.8 | 99.7 | 80.7 | 89.2 |
| **3** | **99.2** | **98.3** | **97.0** | **99.7** | **82.3** | **90.3** |
| 4 | 99.3 | 97.6 | 96.5 | 99.8 | 82.0 | 89.6 |

### Ablation — Reward Weights EP:TTC:Comfort (Table 3)

| Weights | NC ↑ | DAC ↑ | TTC ↑ | Comf. ↑ | EP ↑ | PDMS ↑ |
|---------|------|-------|-------|---------|------|--------|
| 5:20:2 | 99.5 | 98.3 | 97.9 | 99.6 | 80.1 | 89.7 |
| 5:5:8 | 99.4 | 98.1 | 96.9 | 99.7 | 82.1 | 90.1 |
| 20:5:2 | 99.4 | 98.1 | 96.4 | 99.3 | 82.7 | 90.0 |
| **5:5:2** | **99.2** | **98.3** | **97.0** | **99.7** | **82.3** | **90.3** |

### Ablation — Pretraining Scale

![Figure 5: PDMS vs. number of pretraining epochs on 6.5M VQA data. Performance peaks at 3 epochs (+3.3 PDMS over 0 epochs).](<../../raw/assets/ablation_pretrain_epoch.png>)

![Figure 6: PDMS vs. pretraining dataset scale (0.65M vs 6.5M). Numerical tokenizer gains +1.9 at 0.65M and +1.4 more at 6.5M. Text tokenizer gains +5.2 and +2.6 respectively — confirming data scaling law.](<../../raw/assets/ablation_pretrain_data.png>)

## Qualitative Results

![Figure 3: Qualitative comparison on NAVSIM — WAM-Flow produces stable, human-like trajectories in closed-loop simulation.](<../../raw/assets/navsim_comp.png>)

![Figure 4: WAM-Flow on NAVSIM across different scene types — straight roads, intersections, lane changes.](<../../raw/assets/navsim_vis.png>)

## Notable Findings

- **Single camera beats multi-camera + LiDAR** systems on PDMS — DFM paradigm compensates for reduced sensor input.
- **DFM is more efficient than AR**: parallel decoding removes the sequential bottleneck; the CTMC formulation allows O(1)-per-step generation regardless of trajectory length.
- **GRPO transfers to DFM**: the parallel generation mechanism of DFM is preserved during RL, unlike diffusion where the denoising chain MDP requires special treatment.
- **Data scaling law confirmed**: pretraining with 6.5M VQA beats 0.65M; performance peaks at 3 epochs.

## Limitations

1. **Single front camera** — lateral and rear scene context is absent; failure cases in merging/lane-change scenarios not analyzed.
2. **Extended Comfort (EC) gap** — EC = 73.9 on NAVSIM-v2 vs. RecogDrive's 87.7. Root cause not discussed.
3. **No reasoning chain** — WAM-Flow is a black box; dual-system methods produce interpretable NL explanations.
4. **Non-reactive simulator** — NAVSIM doesn't model agent reactions; interactive scenarios are untested.
5. **10-step regression** — slight PDMS drop at 10 steps (90.2 < 90.3) suggests CTMC sampling degrades at high step counts.
6. **Dependence on RecogDrive VQA data** — Stage 2 uses 3.1M pairs from RecogDrive; not a fully independent system.
7. **Computationally expensive training** — four-stage curriculum on 32 NPUs; practicality for resource-constrained research is unclear.

## Connections to Other Wiki Pages

- [[concepts/discrete-flow-matching.md]] — the theoretical foundation (CTMC, probability paths, DFM objectives)
- [[concepts/diffusion-planner.md]] — DFM is a discrete analogue to continuous diffusion; key differences: parallel vs. iterative denoising, discrete vs. continuous state space
- [[concepts/rl-for-ad.md]] — GRPO applied to DFM (parallel generation preserved unlike diffusion-as-MDP)
- [[concepts/navsim-benchmark.md]] — WAM-Flow sets new SOTA on both NAVSIM-v1 and v2
- [[sources/recogdrive.md]] — competitor; provides 3.1M VQA data used in WAM-Flow Stage 2 pretraining

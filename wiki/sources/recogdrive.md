---
title: "ReCogDrive: A Reinforced Cognitive Framework for End-to-End Autonomous Driving"
type: source-summary
sources: [raw/papers/ReCogDrive_ A Reinforced Cognitive Framework for End-to-End Autonomous Driving.md]
related: [concepts/diffusion-planner.md, concepts/rl-for-ad.md, concepts/vlm-domain-adaptation.md, concepts/navsim-benchmark.md, sources/wam-flow.md, sources/uniugp.md]
created: 2026-04-05
updated: 2026-04-05
confidence: high
---

## Citation

Yongkang Li, Kaixin Xiong, et al. (Huazhong University of Science and Technology + Xiaomi EV)
arXiv: https://arxiv.org/abs/2506.08052
Project: https://xiaomi-research.github.io/recogdrive

## One-Line Summary

ReCogDrive integrates a domain-adapted InternVL3-8B VLM with a DDPM diffusion trajectory planner, fine-tuned via simulator-assisted GRPO-style RL, achieving 89.6 PDMS on NAVSIM using camera only.

## Problem Statement

Applying VLMs to end-to-end autonomous driving introduces three failure modes:
1. **Domain gap** — VLMs pre-trained on internet image-text data lack driving-specific knowledge
2. **Action space mismatch** — language tokens cannot precisely represent continuous trajectories (floating-point discretization, occasional output collapse/hallucination in AR decoding)
3. **Imitation learning pathology** — behavior cloning learns the average of multi-modal expert demonstrations, which is unsafe at ambiguous decision points

## System Overview

![Figure 1: ReCogDrive overview — trained on 3.1M driving QA + diverse instruction data. Capable of tasks from low-level scene perception, region recognition, and motion prediction to high-level driving planning and decision making. Outputs: planning command, scene description, reasoning trace.](<../../raw/assets/x1.png>)

An 8B-parameter system with two main modules:
- **VLM backbone**: InternVL3-8B (InternViT-300M visual encoder + Qwen2.5-7B LLM)
- **Diffusion planner**: DDPM scheduler + DiT blocks (self-attention over waypoints + cross-attention to VLM hidden states)

The VLM is frozen in stages 2 and 3; only the diffusion planner is trained after stage 1.

## Architecture

![Figure 2: Model architecture and training pipeline. Left: driving pre-training stage (VLM fine-tuned on diverse QA tasks — detect, describe, trajectory, traffic light). Right: imitation + RL stage (VLM frozen, diffusion planner trained). Bottom: DiT block detail showing self-attention conditioned on ego status via AdaLayerNorm, then cross-attention to VLM tokens.](<../../raw/assets/x2.png>)

### VLM (InternVL3-8B)
- Input: front-view image → 448×448 patches + thumbnail → InternViT → pixel shuffle + MLP projector → 256 visual tokens per patch
- Combined with text tokens (navigation command, ego state) → LLM
- Output: textual trajectory $T_{traj}$, reasoning trace $T_{reason}$, scene description $T_{desc}$

### Diffusion Planner (DiT)
- Input: noisy trajectory $\mathbf{x}_t \in \mathbb{R}^{N \times 3}$ (N waypoints × [x, y, heading])
- Embedded by action encoder $E_{act}$; history trajectory embedded by $E_{his}$
- Concatenated with average-pooled VLM hidden states $\bar{F}_h$ as DiT input $z_t$:
  $$z_t = \text{concat}(E_{act}(\mathbf{x}_t),\; E_{his}(\mathbf{x}_{hist}),\; \bar{F}_h)$$
- **DiT block**: self-attention over waypoints (AdaLayerNorm conditioned on ego status + timestep) → cross-attention to full VLM hidden sequence $F_h$
- Output: denoised continuous trajectory $\mathbf{x}_0 \in \mathbb{R}^{N \times 3}$
- Training loss: $\mathcal{L}_{dif} = \mathbb{E}_{z_t, c} || \epsilon - \epsilon_\pi(z_t, c) ||^2$ (no classifier-free guidance — found to destabilize trajectory generation)

## Three-Stage Training Paradigm

### Stage 1: Driving-Specific VLM Pre-training
- Fine-tune InternVL3-8B via SFT for 3 epochs on **3.1M high-quality driving QA pairs**
- Data sources: 12 open-source datasets (DriveLM, LingoQA, DRAMA, NuScenes-QA, Talk2Car, SUTD, DriveGPT4, MAPLM, NuInstruct, CODA-LM, OmniDrive, Senna) + 775K auto-annotated NAVSIM QA
- Also includes 665K LLaVA instruct-tuning data to preserve general instruction-following
- Data quality pipeline (see below)

### Stage 2: Diffusion Planner via Imitation Learning
- VLM frozen; train diffusion planner for **200 epochs** via DDPM behavior cloning
- Loss: MSE between predicted and ground-truth noise

### Stage 3: Simulator-Assisted Reinforcement Learning
- VLM frozen; fine-tune diffusion planner for **10 epochs** using NAVSIM non-reactive simulator
- Treats the diffusion denoising chain $(x_T, x_{T-1}, \ldots, x_0)$ as an internal MDP
- Samples G trajectories per scene; evaluates each in NAVSIM to obtain PDMS reward $r_i$
- Group-normalized advantage: $\hat{A}_i = (r_i - \text{mean}(r_{1..G})) / \sqrt{\text{var}(r_{1..G})}$
- Combined loss:
  $$L = L_{RL} - \lambda L_{BC} = -\frac{1}{G}\sum_i \frac{1}{T}\sum_t \gamma^{t-1} \log\pi_\theta(x_{t-1}^{(i)} | x_t^{(i)}) \hat{A}_i \;-\; \lambda \frac{1}{G}\sum_i \frac{1}{T}\sum_t \log\pi_\theta(\tilde{x}_{t-1}^{(i)} | \tilde{x}_t^{(i)})$$
- Optimal: $\gamma = 0.6$, $\lambda = 0.01$, $\sigma_{min} = 0.02$

![Figure 3: Training paradigm comparison. (a) Imitation learning: diffusion planner trained offline to mimic GT trajectories, learns averaged/suboptimal paths at intersections with multiple valid GT trajectories. (b) Reinforcement learning (GRPO): G trajectories sampled and evaluated in NAVSIM simulator for collisions, drivable area, comfort; group computation yields advantages $A_1 \ldots A_G$ for policy loss.](<../../raw/assets/x3.png>)

## Data Quality Pipeline

![Figure 5 (appendix): Dataset construction pipeline — Step 1: collect open-source driving QA + NAVSIM samples. Step 2: normalize, augment, filter. Step 3: automatic annotation pipeline for perception/prediction/planning tasks. Pie chart shows category distribution of final 3.1M dataset.](<../../raw/assets/x5.png>)

1. **Normalization**: unify all bounding-box formats to InternVL3 standard — `<car><FRONT_VIEW><box>[x1,y1,x2,y2]</box>` with coordinates scaled to [0, 1000]
2. **Augmentation**: LLM-paraphrased question templates + Qwen2.5-VL answer rewriting for linguistic diversity
3. **Filtering**: Qwen2.5-VL quality scoring; remove pairs scoring < 60 → retains **2.3M of 3.2M** pairs
4. **Auto-annotation**: Qwen2.5-VL-driven pipeline on NAVSIM yields **775K QA pairs** covering scene description, key object ID, behavior narration, road marking, traffic light, VRU detection, motion prediction, planning command prediction, counterfactual reasoning

## Results

### NAVSIM-v1 Full Comparison (Table 1)

| Method | Cam | LiDAR | NC ↑ | DAC ↑ | TTC ↑ | Comf. ↑ | EP ↑ | PDMS ↑ |
|--------|:---:|:-----:|------|-------|-------|---------|------|--------|
| Constant Velocity | | | 68.0 | 57.8 | 50.0 | 100 | 19.4 | 20.6 |
| Ego Status MLP | | | 93.0 | 77.3 | 83.6 | 100 | 62.8 | 65.6 |
| VADv2 | ✓ | | 97.2 | 89.1 | 91.6 | 100 | 76.0 | 80.9 |
| DrivingGPT | ✓ | | 98.9 | 90.7 | 94.9 | 95.6 | 79.7 | 82.4 |
| Hydra-MDP | ✓ | ✓ | 97.9 | 91.7 | 92.9 | 100 | 77.6 | 83.0 |
| UniAD | ✓ | | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| LTF | ✓ | | 97.4 | 92.8 | 92.4 | 100 | 79.0 | 83.8 |
| BevDrive | ✓ | ✓ | 97.7 | 92.5 | 92.9 | 100 | 78.7 | 83.8 |
| TransFuser | ✓ | ✓ | 97.7 | 92.8 | 92.8 | 100 | 79.2 | 84.0 |
| PARA-Drive | ✓ | | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 |
| DRAMA | ✓ | ✓ | 98.0 | 93.1 | 94.8 | 100 | 80.1 | 85.5 |
| Hydra-MDP++ | ✓ | ✓ | 98.3 | 96.0 | 94.6 | 100 | 78.7 | 86.5 |
| ARTEMIS | ✓ | ✓ | 98.3 | 95.1 | 94.3 | 100 | 81.4 | 87.0 |
| DiffusionDrive | ✓ | ✓ | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| WoTE | ✓ | ✓ | 98.5 | 96.8 | 94.9 | 99.9 | 81.9 | 88.3 |
| QwenVL2.5† | ✓ | | 97.8 | 92.1 | 92.8 | 100 | 78.3 | 83.3 |
| InternVL3† | ✓ | | 97.0 | 92.4 | 91.8 | 100 | 78.9 | 83.3 |
| **ReCogDrive** | **✓** | | **98.2** | **97.8** | **95.2** | **99.8** | **83.5** | **89.6** |

† = fine-tuned on NAVSIM trajectory data only (no driving QA).

### Extended Metrics NAVSIM (Table 7, appendix)

| Method | NC ↑ | DAC ↑ | EP ↑ | TTC ↑ | C ↑ | TL ↑ | DDC ↑ | LK ↑ | EC ↑ | EPDMS ↑ |
|--------|------|-------|------|-------|-----|------|-------|------|------|---------|
| TransFuser | 97.7 | 92.8 | 79.2 | 92.8 | 100 | 99.9 | 98.3 | 67.6 | 95.3 | 77.8 |
| VADv2 | 97.3 | 91.7 | 77.6 | 92.7 | 100 | 99.9 | 98.2 | 66.0 | 97.4 | 76.6 |
| Hydra-MDP | 97.5 | 96.3 | 80.1 | 93.0 | 100 | 99.9 | 98.3 | 65.5 | 97.4 | 79.8 |
| Hydra-MDP++ | 97.9 | 96.5 | 79.2 | 93.4 | 100 | 100.0 | 98.9 | 67.2 | 97.7 | 80.6 |
| ARTEMIS | 98.3 | 95.1 | 81.5 | 97.4 | 100 | 99.8 | 98.6 | 96.5 | 98.3 | 83.1 |
| **ReCogDrive** | **98.3** | **95.2** | **87.1** | **97.5** | **98.3** | **99.8** | **99.5** | **96.6** | **86.5** | **83.6** |

ReCogDrive leads on DDC, LK, EP, and TTC. Note EC = 86.5 (vs ARTEMIS 98.3) is a weakness.

### Ablation — Component Contributions (Table 2)

| Config | NC ↑ | DAC ↑ | TTC ↑ | Comf. ↑ | EP ↑ | PDMS ↑ |
|--------|------|-------|-------|---------|------|--------|
| InternVL3 baseline (trajectory SFT only) | 97.0 | 92.4 | 91.8 | 100 | 78.9 | 83.3 |
| + Driving pre-training | 98.2 | 94.5 | 94.5 | 100 | 80.4 | 86.2 (+2.9) |
| + Diffusion planner | 98.3 | 95.1 | 94.3 | 100 | 81.1 | 86.8 (+0.6) |
| + **RL fine-tuning** | **98.2** | **97.8** | **95.2** | **99.8** | **83.5** | **89.6 (+2.8)** |

RL is the single largest improvement (+2.8). Diffusion planner alone adds only +0.6 — its value is enabling RL via the denoising chain MDP.

### Ablation — QA Data Scale and Quality (Table 3)

| Training Data | NC ↑ | DAC ↑ | TTC ↑ | Comf. ↑ | EP ↑ | PDMS ↑ |
|--------------|------|-------|-------|---------|------|--------|
| 85K (LQ) | 97.0 | 92.4 | 91.8 | 100 | 78.9 | 83.3 |
| 1.5M (LQ) | 97.5 | 93.6 | 93.2 | 100 | 79.4 | 84.6 |
| 3.2M (LQ) | 97.9 | 93.8 | 94.1 | 100 | 79.7 | 85.3 |
| **3.1M (HQ, filtered)** | **98.2** | **94.5** | **94.5** | **100** | **80.4** | **86.2** |

Quality filtering (+0.9 PDMS) outperforms a further 100K data increase — **quality > quantity**.

### Ablation — Discount Factor γ (Table 4)

| γ | NC ↑ | DAC ↑ | TTC ↑ | Comf. ↑ | EP ↑ | PDMS ↑ |
|---|------|-------|-------|---------|------|--------|
| 0.5 | 97.8 | 97.4 | 94.6 | 99.7 | 82.7 | 88.8 |
| **0.6** | **98.2** | **97.8** | **95.2** | **99.8** | **83.5** | **89.6** |
| 0.8 | 98.1 | 97.4 | 95.2 | 100 | 82.7 | 89.1 |
| 1.0 | 97.8 | 97.3 | 94.5 | 99.9 | 82.3 | 88.5 |

γ=1.0 assigns equal weight to all denoising steps including early noisy ones, destabilizing learning. γ=0.6 focuses updates on late, reliable denoising steps.

### Ablation — BC Loss Weight λ (Table 5)

| λ | NC ↑ | DAC ↑ | TTC ↑ | Comf. ↑ | EP ↑ | PDMS ↑ |
|---|------|-------|-------|---------|------|--------|
| 0.001 | 97.0 | 97.0 | 93.2 | 98.3 | 78.9 | 85.9 |
| **0.01** | **98.2** | **97.8** | **95.2** | **99.8** | **83.5** | **89.6** |
| 0.1 | 98.1 | 97.2 | 94.0 | 99.7 | 83.4 | 88.6 |

λ=0.001 → training collapse (too little BC stabilization). λ=0.1 → over-constrained exploration.

### Ablation — Min Sampling σ_min (Table 6)

| σ_min | NC ↑ | DAC ↑ | TTC ↑ | Comf. ↑ | EP ↑ | PDMS ↑ |
|-------|------|-------|-------|---------|------|--------|
| 0.01 | 97.1 | 97.2 | 93.1 | 100 | 83.0 | 88.0 |
| **0.02** | **98.2** | **97.8** | **95.2** | **99.8** | **83.5** | **89.6** |
| 0.04 | 97.8 | 97.6 | 94.8 | 99.3 | 80.3 | 87.8 |

Clipping noise variance to a nonzero minimum encourages trajectory diversity; too large destabilizes training.

## Qualitative Results

![Figure 4: ReCogDrive qualitative results on NAVSIM — (a) left turn, (b) go straight, (c) intersection. Shows front-view image, predicted trajectory (red line), scene description, attention highlights on key objects (taxi, traffic lights, pedestrians), and high-level planning commands.](<../../raw/assets/x4.png>)

![Appendix figure: RL impact — bird's-eye view comparison of trajectories w/o RL vs. w/ RL across 4 scenarios (straight road, T-junction, complex intersection, curve). Without RL: trajectories often leave drivable area or collide. With RL: safe, smooth paths that stay within lane boundaries.](<../../raw/assets/x1 1.png>)

## Notable Design Choices

- **No LiDAR**: achieves SOTA with camera-only input, beating all LiDAR+camera competitors at time of publication
- **VLM frozen in stages 2–3**: preserves VLM generalization; diffusion planner acts as adapter between language and action space
- **Non-reactive sim for RL**: avoids the difficulty of building a full interactive simulator; NAVSIM non-reactive sim is sufficient for collision/comfort rewards
- **RL addresses IL averaging pathology**: at an intersection with two valid GT trajectories (left turn + right turn), IL learns an averaged trajectory going straight through the curb; RL explores both and commits to the safe one

## Limitations

1. **Non-reactive simulator**: NAVSIM does not model other agents reacting to the ego vehicle; RL rewards may not transfer to fully interactive scenarios
2. **Single front camera**: like WAM-Flow, no lateral/rear scene context
3. **EC (Extended Comfort) gap**: EPDMS table shows EC = 86.5 vs ARTEMIS 98.3 — the RL reward does not directly optimize extended comfort metrics
4. **Limitations section deferred to supplementary**: main paper does not explicitly analyze failure cases

## Connections to Other Wiki Pages

- [[concepts/diffusion-planner.md]] — DiT architecture, DDPM scheduler, diffusion chain as MDP
- [[concepts/rl-for-ad.md]] — GRPO-style RL applied to diffusion planner; group-normalized advantage; discount γ for denoising chain stability
- [[concepts/vlm-domain-adaptation.md]] — 3.1M QA construction, quality pipeline, data scaling law
- [[concepts/navsim-benchmark.md]] — PDMS metric definition; ReCogDrive held NAVSIM-v1 SOTA until WAM-Flow
- [[sources/wam-flow.md]] — WAM-Flow (DFM, 90.3 PDMS) supersedes ReCogDrive; uses ReCogDrive's VQA data in stage 2 pretraining
- [[sources/uniugp.md]] — UniUGP uses MoT (tighter VLM-planner coupling) vs. ReCogDrive's cross-attention; adds video generation

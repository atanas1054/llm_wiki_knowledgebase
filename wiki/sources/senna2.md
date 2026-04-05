---
title: "Senna-2: Aligning VLM and End-to-End Driving Policy for Consistent Decision Making and Planning"
type: source-summary
sources: [raw/papers/Senna-2_ Aligning VLM and End-to-End Driving Policy for Consistent Decision Making and Planning.md]
related: [concepts/dual-system-vla.md, concepts/rl-for-ad.md, concepts/vlm-domain-adaptation.md, concepts/navsim-benchmark.md, sources/recogdrive.md]
created: 2026-04-05
updated: 2026-04-05
confidence: high
---

## Citation

Yuehao Song, Shaoyu Chen, Hao Gao, et al. (Huazhong University of Science & Technology + Horizon Robotics)
arXiv: https://arxiv.org/abs/2603.11219
Project: https://ambitious-idiot.github.io/senna2-project
Code: https://github.com/hustvl/Senna

## One-Line Summary

A dual-system VLM + E2E driving policy that explicitly aligns high-level VLM decisions with low-level trajectory planning via a three-stage paradigm (pre-training → open-loop alignment → 3DGS closed-loop HRL), achieving +19.3% decision-planning F1 and −30.6% at-fault collision rate vs. Senna.

## Problem Statement

Existing VLM-E2E systems (e.g., Senna, ReCogDrive, Alpamayo-R1) use VLM decisions to *guide* an E2E planner, but without explicit alignment the planner can generate trajectories that contradict the VLM's intent. For example: the VLM outputs "Decelerate, Go Straight" but the trajectory speeds up or turns. This **consistency gap** weakens interpretability and top-down control, and can lead to unsafe behavior.

![Figure 1: (a) existing methods — VLM decisions guide E2E policy but without alignment, trajectories deviate from intended decisions. (b) Senna — limited separability between speed control modes (distributions overlap). (c) Senna-2 — clearly separated speed control distributions demonstrating strong decision-following.](<../../raw/assets/x1 3.png>)

## Architecture: Dual-System with Decision Adapter

![Figure 2: Architecture overview — text and visual inputs processed in parallel by VLM (top) and E2E backbone (bottom). Decision Adapter converts VLM decisions into VLM condition embeddings (VLM tokens + decision tokens → fusion → VLM condition). E2E planner fuses VLM condition via AdaLN and E2E features via cross-attention to generate trajectory.](<../../raw/assets/x2 2.png>)

### VLM (Qwen2.5-VL-3B)
- Input: single front-view frame + system prompt + navigation command + ego speed
- Output: structured **meta-action** — one of 20 combinations:
  - Speed control: Acceleration, Deceleration, Keep Speed, Stop (4 options)
  - Direction control: Go Straight, Turn Left, Turn Right, Change Lane Left, Change Lane Right (5 options)
- These discrete meta-actions are interpretable and serve as the human-machine interface

### Decision Adapter
Converts VLM meta-action decisions into representations compatible with the E2E planner. Two complementary token types:
- **VLM tokens** $T_{vlm}$: MLP projection of VLM final hidden states — preserves global semantic reasoning context
- **Decision tokens** $T_{vel}$, $T_{dir}$: learnable category embeddings indexed by decoded speed/direction action — explicit categorical awareness

Fused via MLP:
$$F_{vlm} = \text{MLP}(\text{Concat}(T_{vlm}, T_{vel}, T_{dir}))$$

Ablation confirms both token types are necessary (removing VLM token: +11.8% FDE; removing decision token: +10.1% FDE).

### E2E Driving Policy
- **E2E backbone**: multi-view image sequences → spatio-temporal features
- **Perception heads**: predict map elements, dynamic agents, navigation cues → E2E condition
- **DiT planner**: diffusion-based
  - VLM condition $F_{vlm}$ injected via **AdaLN** (global modulation — affects all waypoints uniformly)
  - E2E condition injected via **cross-attention** (fine-grained spatial grounding)
  - Predicts **residual trajectory** $\mathbf{r}_0$ = expert trajectory − kinematic extrapolation from initial speed
  - Training loss: $\mathcal{L}_{E2E} = \mathbb{E}[||\epsilon - \epsilon_\theta(\mathbf{r}_t, t, c)||^2]$

**AdaLN for VLM injection vs. cross-attention for E2E**: AdaLN globally modulates the planning process (VLM sets the "tone" for the whole trajectory), while cross-attention gives fine-grained spatial grounding from perception features.

### Kinematic Mapping $f_K$
A deterministic function converting a continuous trajectory into its corresponding meta-action category. Used at all three training stages:
- Stage 1: generates decision labels from GT trajectories for VLM pre-training
- Stage 2: checks consistency between E2E output trajectory and VLM decision
- Stage 3: propagates optimized trajectory back to align VLM

### Asynchronous Operation
VLM cannot run at 10 Hz on edge devices. A **memory bank** caches VLM features, which the E2E planner uses at full inference frequency. VLM and E2E operate asynchronously (acknowledged as a limitation).

## Three-Stage Training Paradigm

![Figure 3: Training recipe. Stage 1: VLM pre-training on meta-action QA + E2E pre-training on trajectories + adapter pre-training (VLM frozen). Stage 2: open-loop alignment via kinematic mapping consistency check — apply full supervision for inconsistent samples, skip (no-grad) for consistent ones. Stage 3: closed-loop alignment with HRL in 3DGS environments — bottom-up hierarchical: optimize E2E planner first, then align VLM to match.](<../../raw/assets/x3 2.png>)

### Stage 1 — Driving Pre-training

Three sub-stages:

**VLM pre-training**: QA formulation — front-view frame + navigation + ego speed → predict meta-action text:
$$\mathcal{L}_{VLM} = -\sum \log P(d_t | d_{<t}, Q)$$
Labels generated by applying $f_K$ to ground-truth trajectories.

**E2E pre-training**: standard diffusion loss on residual trajectory.

**Adapter pre-training**: VLM frozen; adapter + E2E planner fine-tuned on $\mathcal{L}_{E2E}$.

Training scale: ~10,000 hours (360M frames) of real-world driving demonstrations.

| Component | GPU | Steps |
|-----------|-----|-------|
| VLM pre-training | 128 × NVIDIA L20 | 10K |
| E2E pre-training | **1,024 × NVIDIA H20** | 30K |

### Stage 2 — Open-Loop Alignment

**Core idea**: use decision-planning consistency as an explicit training signal — reinforce what's already consistent, correct only what's inconsistent.

**Binary consistency indicator**:
$$\mathcal{C}(\tau, d) = \begin{cases} 1 & f_K(\tau) = d \\ 0 & \text{otherwise} \end{cases}$$

**Selective loss** — zero loss for consistent samples (self-reinforcing), full supervision for inconsistent:
$$\mathcal{L}_{stage2} = (1 - \mathcal{C}(\tau, d))(\mathcal{L}_{E2E} + \gamma \mathcal{L}_{VLM})$$

This is elegant: consistent samples are treated as implicit expert signals without needing external supervision. Only the gap cases are corrected.

Effect from ablation: Stage 2 alone improves F1 from 0.701 → 0.764 (+9.0%), AF-CR from 0.144 → 0.118 (−18.1%).

### Stage 3 — Closed-Loop Alignment with HRL

**3DGS environments**: 1,300 high-risk, dense-traffic driving clips reconstructed as photorealistic 3D Gaussian Splatting environments. 1,044 for training, 256 for closed-loop evaluation.

Unlike NAVSIM (non-reactive, pre-rendered): agents in 3DGS react to ego actions, visual appearance is photorealistic (VLM sees real-looking images, not replays).

**Bottom-up hierarchical optimization**:

**(i) Low-level planner (E2E)** — two complementary rewards:
$$\mathcal{L}_{safe} = \mathbb{E}_{\tau \sim \pi_\theta} \sum_{t=1}^{T-1} \mathbf{1}_{[f_{ttc}(\tau) < \delta_t]} ||\tau_{t+1} - \text{sg}(\tau_t)||_2^2$$
$$\mathcal{L}_{eff} = \mathbb{E}_{\tau \sim \pi_\theta} \sum_{t=1}^{T-1} \mathbf{1}_{[f_v(\tau) < \delta_v]} ||\tau_t - \text{sg}(\tau_{t+1})||_2^2$$
- Safety: longitudinal penalty when TTC < 3s (pushes trajectory point forward to avoid collision)
- Efficiency: longitudinal extension when ego speed too low (pushes trajectory point forward to accelerate)
- Note: `sg(·)` = stop gradient; only one endpoint is updated to create directed pressure

**(ii) High-level VLM** — aligned to match the optimized trajectory:
$$\mathcal{L}_{high} = -\log P(f_K(\tau) | Q)$$
If VLM decision ≠ trajectory's kinematic category, penalize the VLM to reduce that decision's probability.

**Final Stage 3 loss**: $\mathcal{L}_{stage3} = \mathcal{L}_{high} + \beta \mathcal{L}_{low}$

Effect from ablation: Stage 3 reduces AF-CR from 0.118 → 0.077 (−34.7% additional reduction from Stage 2).

## Results

### Decision-Planning Consistency (Table 1)

F1 score between VLM decisions and E2E trajectory decisions (via kinematic mapping):

| Method | Straight | Left | Right | Keep | Acc. | Dec. | Stop | **Avg.** |
|--------|---------|------|-------|------|------|------|------|---------|
| Senna | 0.763 | 0.533 | 0.574 | 0.550 | 0.612 | 0.628 | 0.802 | 0.637 |
| **Senna-2** | **0.809** | **0.664** | **0.710** | **0.754** | **0.769** | **0.780** | **0.838** | **0.760** |

+19.3% average F1. Largest gains in direction (left: +24.6%, right: +23.7%) and speed keeping (+37.1%).

### Open-Loop Planning (Table 2)

| Method | FDE (m) ↓ | ADE (m) ↓ | CR (%) ↓ | DCR (%) ↓ | SCR (%) ↓ |
|--------|----------|----------|---------|---------|---------|
| TransFuser | 0.844 | 0.297 | 0.981 | 0.827 | 0.154 |
| VAD | 0.722 | 0.262 | 0.621 | 0.554 | 0.067 |
| GenAD | 0.806 | 0.290 | 0.520 | 0.491 | 0.030 |
| ResAD | 0.634 | 0.234 | 0.378 | 0.367 | 0.011 |
| Senna | 0.633 | 0.236 | 0.294 | 0.286 | 0.008 |
| **Senna-2** | **0.597** | **0.225** | **0.288** | **0.283** | **0.005** |

Best on all metrics. −5.7% FDE vs. Senna.

### Closed-Loop Planning — 3DGS Benchmark (Table 3)

| Method | CR ↓ | AF-CR ↓ | Safety@1 ↑ | Safety@2 ↑ |
|--------|-----|--------|----------|----------|
| TransFuser | 0.435 | 0.269 | 0.531 | 0.454 |
| VAD | 0.502 | 0.280 | 0.458 | 0.362 |
| GenAD | 0.557 | 0.244 | 0.402 | 0.332 |
| VADv2 | 0.422 | 0.199 | 0.514 | 0.458 |
| RAD | 0.281 | 0.113 | 0.613 | 0.543 |
| Senna | 0.310 | 0.111 | 0.638 | 0.539 |
| **Senna-2** | **0.269** | **0.077** | **0.667** | **0.565** |

−30.6% AF-CR vs. Senna. Beats RAD (0.113) despite both using RL — gain attributed to alignment strategy, not RL alone.

### NAVSIM-v2 (Table 6, after fine-tuning on NAVSIM)

| Method | NC ↑ | DAC ↑ | DDC ↑ | TL ↑ | EP ↑ | TTC ↑ | LK ↑ | HC ↑ | EC ↑ | EPDMS ↑ |
|--------|------|-------|-------|------|------|-------|------|------|------|---------|
| TransFuser | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 |
| Hydra-MDP++ | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | 81.4 |
| DriveSuprim | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 | 83.1 |
| ARTEMIS | 98.3 | 95.1 | 98.6 | 99.8 | 81.5 | 97.4 | 96.5 | 98.3 | — | 83.1 |
| DiffusionDrive | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | 84.5 |
| ResAD | 97.8 | 97.2 | 99.5 | 99.8 | 88.2 | 96.9 | 97.0 | 98.4 | 88.2 | 85.5 |
| **Senna-2** | **98.5** | **97.8** | **99.5** | **99.8** | **88.1** | **97.5** | **97.0** | **98.6** | **88.4** | **86.6** |

NAVSIM-v2 SOTA as of April 2026. Note: this result uses a separate NAVSIM fine-tune variant.

### Ablation — Architecture (Table 4, Stage 1 only)

| Dual system | VLM token | Dec. token | FDE ↓ | CR ↓ | AF-CR ↓ | F1 ↑ |
|:-----------:|:---------:|:----------:|-------|------|--------|------|
| ✗ | ✗ | ✗ | 0.695 | 0.387 | 0.288 | — |
| ✓ | ✗ | ✓ | 0.634 | 0.299 | 0.148 | 0.660 |
| ✓ | ✓ | ✗ | 0.624 | 0.312 | 0.151 | 0.688 |
| ✓ | ✓ | ✓ | **0.567** | **0.281** | **0.144** | **0.701** |

### Ablation — Training Stages (Table 5)

| Stage 1 | Stage 2 | Stage 3 | FDE ↓ | CR ↓ | AF-CR ↓ | F1 ↑ |
|:-------:|:-------:|:-------:|-------|------|--------|------|
| ✓ | | | 0.567 | 0.281 | 0.144 | 0.701 |
| ✓ | ✓ | | 0.575 | 0.290 | 0.118 | 0.764 |
| ✓ | ✓ | ✓ | **0.597** | **0.288** | **0.077** | **0.760** |

Note: Stage 3 slightly increases open-loop FDE (0.575 → 0.597) — closed-loop RL trades away some open-loop trajectory precision. Stage 2 alone improves F1 most (+9.0%); Stage 3 most improves closed-loop safety (AF-CR −34.7% from Stage 2 baseline).

## Qualitative Results

![Figure 4: Closed-loop speed control in empty-road scenario. Senna-2 produces clearly differentiated control profiles for Accelerate/Keep/Decelerate/Normal — (a) front views, (b) speed over time, (c) mileage over time. All four modes have non-overlapping trajectories, demonstrating strong decision-following.](<../../raw/assets/x4 2.png>)

![Figure 5: Qualitative comparison Senna vs. Senna-2. (a) Rear-end collision scenario: Senna's planner fails to slow despite VLM issuing "Decelerate" — collides. Senna-2 maintains consistent decision-planning and stops safely. (b) Cut-in scenario: Senna issues incorrect "Accelerate" command, misses braking window, collides; then issues unreasonable "Turn Right" post-collision. Senna-2 correctly decelerates then changes lane left to safely avoid.](<../../raw/assets/x5 2.png>)

![Figure 6: Training curves — Open-loop FDE and Closed-loop AF-CR decrease together with Consistency F1 as training progresses, confirming the three objectives improve jointly.](<../../raw/assets/x6 1.png>)

![Figure 7 (appendix): Additional closed-loop qualitative comparisons across 3 further scenarios — Senna-2 consistently produces correct deceleration decisions where Senna accelerates or issues inconsistent commands.](<../../raw/assets/x7 1.png>)

## Limitations

1. **VLM not real-time on edge** — cannot run at 10 Hz on on-board devices; async memory bank is a workaround. Fully synchronous deployment requires hardware optimization.
2. **Private proprietary dataset** — 10K hours of internal data + 1,300 proprietary 3DGS clips. NAVSIM result requires a separate fine-tune; no fully public training recipe.
3. **Kinematic mapping is approximate** — $f_K$ maps continuous trajectories to discrete categories; ambiguous boundary trajectories may flip labels, injecting noise into the consistency check.
4. **Single front-view for VLM** — lateral and rear context absent from decision-making.
5. **Open-loop FDE increases with Stage 3** (0.575 → 0.597) — safety-efficiency tradeoff from RL.
6. **3DGS trained only on high-risk clips** — RL sees a non-representative distribution; may overfit to collision avoidance.
7. **Bottom-up HRL** — E2E optimized first, then VLM aligned to match; E2E optimization errors propagate to VLM.

## Connections to Other Wiki Pages

- [[concepts/dual-system-vla.md]] — Senna-2 is the primary example of the dual-system VLM + E2E architecture pattern
- [[concepts/rl-for-ad.md]] — HRL with 3DGS; bottom-up hierarchical RL; contrasts with NAVSIM-based GRPO in ReCogDrive/WAM-Flow
- [[concepts/vlm-domain-adaptation.md]] — Senna-2's decision adapter, selective open-loop alignment, kinematic mapping
- [[concepts/navsim-benchmark.md]] — Senna-2 is current NAVSIM-v2 SOTA (86.6 EPDMS, after fine-tune)
- [[sources/recogdrive.md]] — ReCogDrive uses a single-system approach (VLM + diffusion planner via cross-attention) vs. Senna-2's explicit dual-system consistency alignment

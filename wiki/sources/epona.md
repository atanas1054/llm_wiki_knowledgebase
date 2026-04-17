---
title: "Epona: Autoregressive Diffusion World Model for Autonomous Driving"
type: source-summary
sources: [raw/papers/Epona_ Autoregressive Diffusion World Model for Autonomous Driving.md]
related: [sources/dreameraD.md, concepts/world-model-for-ad.md, concepts/navsim-benchmark.md, concepts/diffusion-planner.md, concepts/rl-for-ad.md]
created: 2026-04-17
updated: 2026-04-17
confidence: high
---

## Overview

**Epona** is a 2.5B-parameter **autoregressive diffusion world model** for autonomous driving from Horizon Robotics, Tsinghua, PKU, NJU, HKUST, NTU, and Tencent (arXiv June 2025; ICCV 2025). It bridges the gap between two existing world model paradigms — video diffusion models (high quality, fixed length, no causal structure) and GPT-style AR models (flexible length, degraded quality from tokenization) — by combining causal AR temporal modeling with continuous-space diffusion generation.

**arXiv**: 2506.24113  
**Code**: [Kevin-thu/Epona](https://github.com/Kevin-thu/Epona)  
**Successor**: DreamerAD ([[sources/dreameraD.md]]) builds on Epona as its backbone, adding latent RL to push NAVSIM PDMS from 86.2 → 88.7.

---

## Problem: Why Both Prior Paradigms Fail

![[raw/assets/x3 19.png]]

*Figure 3: Three world model paradigms. (Top) GPT-style AR: tokenizes continuous images into discrete tokens, then predicts token-by-token — degrades visual quality. (Middle) Video diffusion: jointly generates all future frames at once — no causal structure, fixed length only. (Bottom, Ours) Epona: autoregressively predicts fine-grained future frames in continuous space.*

**Video diffusion-based** (e.g., Vista): models the joint distribution over past + fixed-length future frames simultaneously:
$$p\!\left(\{\mathbf{O}_{T+i}\}_{i=1}^n,\{\mathbf{O}_t, \mathbf{a}_t\}_{t=1}^T\right)$$
No causal structure → cannot generate variable-length sequences → cannot model per-timestep planning choices.

**GPT-based** (e.g., GAIA-1, DrivingGPT): discretizes images into token sequences, predicts token-by-token:
$$\prod_{i=1}^L p(\mathbf{t}_i \mid \mathbf{t}_{<i}, \{\mathbf{O}_t, \mathbf{a}_t\}_{t=1}^T)$$
Quantization degrades visual fidelity and trajectory precision. Causal transformers can only predict the next single action, not a long-horizon trajectory.

**Epona's approach**: decouple temporal dynamics modeling (AR in latent space) from fine-grained generation (diffusion DiTs):
- Plan trajectory: $\pi(\{\mathbf{a}_{T\to T+i}\}_{i=1}^n \mid \{\mathbf{O}_t, \mathbf{a}_{t-1\to t}\}_{t=1}^T)$
- Generate next frame: $p(\mathbf{O}_{T+1} \mid \{\mathbf{O}_t, \mathbf{a}_{t-1\to t}\}_{t=1}^T, \mathbf{a}_{T\to T+1})$

---

## Architecture

![[raw/assets/x2 19.png]]

*Figure 2: Epona overview. MST processes T history frames → compact latent F. TrajDiT predicts future trajectory. VisDiT generates next frame. Chain-of-forward feeds self-predicted frames back as history for long-horizon autoregressive generation.*

Three modules totaling **2.5B parameters**:

| Module | Params | Role |
|--------|--------|------|
| **MST** (Multimodal Spatiotemporal Transformer) | 1.3B, 12 layers | Causal temporal + spatial attention over T history frames → compact latent F |
| **VisDiT** (Next-frame Prediction DiT) | 1.2B, 12 layers | Rectified flow over next frame; conditioned on F + action |
| **TrajDiT** (Trajectory Planning DiT) | 50M, 2 layers | Rectified flow over 3-second trajectory; conditioned on F |

### MST: Multimodal Spatiotemporal Transformer

Interleaves causal temporal attention (across frames) and spatial attention (within frames):

```
E ← rearrange(E, (b t) l c → (b l) t c)
E ← CausalTemporalLayer(E, CausalMask)
E ← rearrange(E, (b l) t c → (b t) l c)
E ← MultimodalSpatialLayer(E)
```

Where $\mathbf{E} \in \mathbb{R}^{B \times T \times (L+3) \times D}$ concatenates flattened visual latent patches ($L = H \times W$) and action tokens (dim 3: $\Delta\theta$, $\Delta x$, $\Delta y$) along the spatial dimension. Causal mask ensures only past frames inform each timestep.

Output: last-frame embedding $\mathbf{F} \in \mathbb{R}^{B \times (L+3) \times D}$ — the compact representation of all history, conditioning both DiTs.

### TrajDiT and VisDiT: Dual-Single-Stream DiT

![[raw/assets/x9 4.png]]

*Figure 9: DiT architecture. Dual-stream phase: condition (F) and noise processed separately with attention-only coupling. Single-stream phase: both concatenated for full fusion. Action control (VisDiT only) provides scale/shift via adaptive modulation.*

Both use a FLUX/HunyuanVideo-inspired **Dual-Single-Stream** design:
- **Dual-stream phase**: historical latent F and noisy trajectory/frame processed independently, linked only via cross-attention
- **Single-stream phase**: concatenated for unified processing

**Training loss** (rectified flow for both):
$$\mathcal{L}_{traj} = \mathbb{E}\!\left[\|v_{traj}(\bar{\mathbf{a}}_{(t)}, t) - (\bar{\mathbf{a}} - \epsilon)\|^2\right]$$
$$\mathcal{L}_{vis} = \mathbb{E}\!\left[\|v_{vis}(\mathbf{Z}_{T+1(t)}, t) - (\mathbf{Z}_{T+1} - \epsilon)\|^2\right]$$
$$\mathcal{L} = \mathcal{L}_{traj} + \mathcal{L}_{vis}$$

**Modular inference**:
- **Full mode**: MST + TrajDiT + VisDiT → trajectory + next frame
- **Planning-only mode**: MST + TrajDiT only → trajectory at 20 Hz (0.02s + 0.03s at 10 steps)
- VisDiT at 100 steps adds ~2s/frame — disabled for real-time planning

### DCAE: 32× Deep Compression Autoencoder

Standard VAEs compress by 8×. DCAE (Deep Compression AE) compresses by 32×, reducing latent tokens by 16× — enabling conditioning on longer historical contexts at the same memory budget. Training resolution: 512×1024.

**Limitation**: DCAE is an image encoder with no temporal modeling → inter-frame flickering when decoding video frame by frame.

---

## Temporal-aware DCAE Decoder

To fix flickering: adds **spatiotemporal self-attention layers** before the DCAE decoder while keeping the encoder frozen. Enables multi-frame interactions during decoding without changing the encoding pipeline.

**Ablation on NuPlan test set**:

| Method | FVD₁₀ ↓ | FVD₂₅ ↓ | FVD₄₀ ↓ |
|--------|---------|---------|---------|
| w/o Temporal Module | 52.95 | 76.46 | 100.11 |
| **Ours** | **50.77** | **61.46** | **74.88** |

The benefit compounds at longer horizons — temporal consistency becomes increasingly important as generation length increases.

---

## Chain-of-Forward Training

![[raw/assets/x4 18.png]]

*Figure 4: Chain-of-forward training. Periodically, the model's own velocity-estimated predictions are used as history for subsequent forward passes, simulating the inference distribution during training.*

**Problem**: teacher-forcing training uses GT history; autoregressive inference uses self-predicted frames. Distribution mismatch → error accumulation → rapid quality degradation past 10–20 seconds.

**Fix**: every 10 training steps, perform 3 forward passes using self-predicted frames as context. Instead of full denoising (expensive), estimate the clean frame in one step via the predicted velocity:
$$\hat{x}_{(0)} = x_{(t)} + t \cdot v_\Theta(x_{(t)}, t)$$

This exposes the model to its own prediction noise during training at near-zero extra cost. The estimated $\hat{x}_{(0)}$ is fed as the next conditioning frame.

![[raw/assets/x7 13.png]]

*Figure 7: Without chain-of-forward (left): visual quality degrades after 10–20 seconds. With chain-of-forward (right): high-quality generation sustained for minute-long videos.*

**FID comparison over time (NuPlan test)**: the gap between with/without chain-of-forward training grows monotonically with generation length — confirming that autoregressive drift is the dominant failure mode for long-horizon generation, and that chain-of-forward directly addresses it.

---

## Inference Speed (Single NVIDIA 4090)

| Module | 10 steps | 100 steps |
|--------|----------|-----------|
| MST | ~0.02s | ~0.02s |
| TrajDiT | ~0.03s | ~0.3s |
| VisDiT | ~0.3s | ~2s |

**Planning-only** (MST + TrajDiT, 10 steps): ~0.05s → **20 Hz real-time**.  
**Full generation** (MST + TrajDiT + VisDiT, 100 steps): ~2.3s per frame.

---

## Results

### Video Generation — NuScenes (Table 1)

| Method | FID ↓ | FVD ↓ | Max Duration |
|--------|-------|-------|-------------|
| DriveGAN | 73.4 | 502.3 | N/A |
| DriveDreamer | 52.6 | 452.0 | 4s / 48 frames |
| Vista | 6.9 | 89.4 | 15s / 150 frames |
| DrivingWorld | 7.4 | 90.9 | 40s / 400 frames |
| **Epona (Ours)** | **7.5** | **82.8** | **120s / 600 frames** |

FVD 82.8 beats Vista (89.4) by −7.4% and DrivingWorld (90.9). FID 7.5 is marginally worse than Vista (6.9) — attributed to the aggressive 32× DCAE compression vs. standard VAEs. Generation length extends from Vista's 15s to 120s (8×).

![[raw/assets/x5 18.png]]

*Figure 5: Qualitative comparison with Vista. Epona maintains visual consistency and detail over long horizons. Vista's rollout-based extension shows visible quality degradation.*

![[raw/assets/x6 16.png]]

*Figure 6: Trajectory-controlled video generation. Given different predefined trajectories, Epona generates corresponding future scenes — enabling simulation of diverse and extreme driving scenarios.*

![[raw/assets/x10 5.png]]

*Figure 10: 140-second ultra-long video generation. High visual quality maintained throughout, with buildings and vehicles preserved in detail.*

### Planning — NuScenes (Table 3)

All methods use multi-view camera except where noted. Epona uses **front camera only** with **no auxiliary supervision** (no map labels, no 3D boxes, no motion annotations).

| Method | Input | Aux. Supervision | Avg L2 ↓ | Avg Collision ↓ |
|--------|-------|-----------------|----------|----------------|
| UniAD | Multi-cam | Map + Box + Motion + Tracklets + Occ | 1.03m | 0.31% |
| VAD-Base | Multi-cam | Map + Box + Motion | 1.22m | 0.53% |
| GenAD | Multi-cam | Map + Box + Motion | 0.91m | 0.43% |
| Doe-1 | Front cam | QA | 1.26m | 0.53% |
| **Epona (Ours)** | **Front cam** | **None** | **1.25m** | **0.36%** |

Notable: 1s collision rate = **0.01%** (lowest in the table) — Epona learns basic traffic rules (stop at red light, etc.) purely from next-frame prediction without any explicit safety annotation.

### Planning — NAVSIM v1 (Table 4)

| Method | Input | NC ↑ | DAC ↑ | TTC ↑ | Comf. ↑ | EP ↑ | PDMS ↑ |
|--------|-------|------|-------|-------|---------|------|--------|
| UniAD | Camera | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| PARA-Drive | Camera | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.6 |
| LAW | Camera | 96.4 | 95.4 | 88.7 | 99.9 | 81.7 | 84.6 |
| DRAMA | Cam+LiDAR | 98.0 | 93.1 | 94.8 | 100 | 80.1 | 85.5 |
| **Epona (Ours)** | **Camera** | **97.9** | **95.1** | **93.8** | **99.9** | **80.4** | **86.2** |

Camera-only, no perception labels. Conditioned on 2-second history to predict 4-second future.

**Caveat**: comparison table only includes pre-2025 baselines (UniAD 83.4, DRAMA 85.5, etc.). No head-to-head with DiffusionDrive (88.1), ReCogDrive (89.6), WAM-Flow (90.3), DriveFine (90.7), FLARE (91.4), or DiffusionDriveV2 (91.2). Epona's 86.2 is below the full wiki VLA field.

---

## Ablation Studies

### Joint Training (Table 5)

| Config | PDMS |
|--------|------|
| Ours w/o Joint Training (trajectory only, no VisDiT) | 78.1 |
| **Ours (joint)** | **86.2** |

**−8.1 PDMS without video supervision.** The shared latent F is trained to be predictive of future visual states — removing VisDiT means F is only optimized for trajectory prediction, losing the rich scene dynamics captured by visual generation. This is the strongest evidence in the paper that the world model signal is essential for planning quality.

### Context Length (Table 7)

| Frames conditioned | FVD₁₀ ↓ | FVD₂₅ ↓ | FVD₄₀ ↓ |
|-------------------|---------|---------|---------|
| 2 | 59.85 | 81.58 | 103.70 |
| 5 | 55.46 | 71.28 | 86.76 |
| **10** | **50.77** | **61.46** | **74.88** |

More history consistently improves quality. 10 frames = upper limit given memory constraints.

---

## Training Details

- **Parameters**: 2.5B total (MST 1.3B + VisDiT 1.2B + TrajDiT 50M)
- **Dataset**: NuPlan (variable length driving videos) + 700 NuScenes scenes; trained **from scratch**
- **Resolution**: 512×1024 (all images resized)
- **Encoder**: DCAE (32× compression, continuous latents)
- **Objective**: Rectified flow for both video and trajectory
- **Optimizer**: AdamW, lr = 1×10⁻⁴, weight decay = 5×10⁻²
- **Training**: 48 NVIDIA A100 GPUs, ~2 weeks, 600k iterations, batch 96
- **Chain-of-Forward**: every 10 steps, 3 forward passes with velocity-estimated self-predictions
- **DiT sampling**: 100 steps (standard); 10 steps for fast inference

---

## Relationship to DrivingGPT (Table 8)

Direct comparison on NAVSIM against the most comparable concurrent world model:

| Method | PDMS |
|--------|------|
| DrivingGPT | 82.4 |
| **Epona** | **86.2** |

DrivingGPT uses discrete tokens (AR) and predicts single-step actions. Epona predicts full 3-second multi-step trajectories in continuous space via TrajDiT.

---

## Key Takeaways

1. **Decoupled spatiotemporal factorization enables both long-horizon generation and real-time planning**: the MST produces a compact causal latent F that is reused by both DiTs. Switching between generation mode and planning-only mode requires only enabling/disabling VisDiT — no architectural changes.

2. **Joint video + trajectory supervision is essential**: removing VisDiT while keeping TrajDiT drops PDMS by 8.1 (86.2 → 78.1). The visual generation task forces the shared latent F to encode richer scene dynamics than trajectory prediction alone can provide.

3. **Chain-of-forward training solves autoregressive drift for minute-long generation**: by simulating inference-time prediction errors during training (with a cheap 1-step velocity estimate), Epona generates 120s / 600 frames without quality degradation. This is a broadly applicable technique for any autoregressive generative model.

4. **FVD SOTA at 82.8 on NuScenes** at the time of publication — and generation horizon 8× longer than prior best (Vista 15s → Epona 120s).

5. **Self-supervised traffic rule learning**: lowest 1s collision rate on NuScenes (0.01%) with zero safety annotation — Epona learns red-light stopping purely from next-frame visual prediction.

6. **DreamerAD is Epona + latent RL**: the same group showed that adding SF-WM distillation + AD-RM reward model over Epona's latents pushes NAVSIM PDMS from 86.2 → 88.7 (+2.5) and EPDMS to 87.7, making Epona the strongest pure-SFT world model baseline in the wiki. See [[sources/dreameraD.md]].

---

## Limitations

1. **NAVSIM planning below VLA SOTA**: 86.2 PDMS (camera-only) is below DiffusionDrive (88.1, Camera+LiDAR), ReCogDrive (89.6), WAM-Flow (90.3), DriveFine (90.7), FLARE (91.4), and DiffusionDriveV2 (91.2). The comparison table excludes all of these.

2. **Real-time planning requires disabling video generation**: the 20 Hz rate is only achievable when VisDiT is off. Full world model inference (MST + TrajDiT + VisDiT, 100 steps) takes ~2.3s per frame.

3. **FID slightly worse than Vista** (7.5 vs 6.9): 32× DCAE compression introduces visual artifacts. The paper acknowledges this and plans to improve DCAE in future work.

4. **Massive training cost**: 48 A100 GPUs × ~2 weeks. Not reproducible without significant compute resources.

5. **Front camera only for planning**: multi-camera methods have structural advantages for lateral awareness; NuScenes L2 comparison is against multi-view systems.

6. **No NAVSIM-v2 / EPDMS reported**.

7. **Superseded for planning by DreamerAD**: for pure planning purposes, DreamerAD is the preferred successor (+2.5 PDMS, +1.5 EPDMS). Epona's primary value is as a **world generation model** and as evidence that joint video+trajectory training benefits planning.

---
title: "AutoMoT: Asynchronous Mixture-of-Transformers for End-to-End Autonomous Driving"
type: source-summary
sources: [raw/papers/AutoMoT_ A Unified Vision-Language-Action Model with Asynchronous Mixture-of-Transformers for End-to-End Autonomous Driving.md]
related: [concepts/dual-system-vla.md, concepts/vlm-domain-adaptation.md, concepts/diffusion-planner.md, sources/senna2.md, sources/recogdrive.md, sources/autovla.md, sources/linkvla.md, sources/orion.md]
created: 2026-04-05
updated: 2026-04-05
confidence: high
---

**arXiv**: https://arxiv.org/html/2603.14851v1  
**Venue**: ICML  
**Affiliation**: Nanyang Technological University  
**Project**: https://automot-website.github.io/

## One-Line Summary

AutoMoT freezes a Qwen3-VL-4B Understanding Expert entirely and trains a separate 1.6B Action Expert from scratch, connecting them via layer-wise shared KV cache and causal joint attention — enabling 7.6× faster asynchronous inference with <1.24% L2 degradation, 87.34 DS on Bench2Drive, and empirical evidence that fine-tuning VLMs on AD data causes catastrophic forgetting on general reasoning benchmarks.

## Core Problem Statement

Three limitations of existing VLA integration strategies:
1. **Distributional misalignment** — intermediate representations (meta-actions, latent codes) between VLM reasoning space and action space cause semantic gaps
2. **VLM capability degradation** — fine-tuning pre-trained VLMs on AD-specific tasks causes catastrophic forgetting of general reasoning
3. **Inference latency** — synchronous coupling of reasoning and action at the same frequency is impractical at real-time AD control rates

## Architecture: Mixture-of-Transformers (MoT)

![[x2 10.png|AutoMoT four-paradigm comparison]]
![[x3 11.png|AutoMoT architecture: UE + AE + joint attention + diffusion refiner]]

*Figure 2: Four paradigms of VLM-AD integration. AutoMoT (Fig. 1d) uses MoT with joint attention in a shared latent space, enabling asynchronous fast-slow inference.*

### Understanding Expert (UE)
- **Backbone**: Qwen3-VL-4B dense model
- **Input**: multi-view multi-frame RGB images $I^{RGB}$, textual prompts $\ell$ (system prompts + instructions)
- **Output**: semantic reasoning tokens (CoT for complex/long-tail scenarios)
- **Training**: **fully frozen throughout** — never fine-tuned on any AD dataset

### Action Expert (AE)
- **Parameters**: ~1.6B, **trained from scratch**
- **Input**: current RGB $I^{RGB}_t$, LiDAR BEV features $I^{BEV}_t$, action queries $Q(t)$
- **Output**:
  - Meta-actions $\hat{Z}_t = \{\hat{z}_{t+h}\}_{h=1}^{3}$ — 3-second horizon at 1s intervals (up to 20 combinations: 6 lateral × longitudinal)
  - Temporal waypoints $\hat{Y}_t$ — 6 points at 0.5s intervals (3s horizon)
  - Spatial route points $\bar{Y}_t$ — 20 nodes parameterizing reference path
- **Training losses**: $\mathcal{L}_{DM}$ (cross-entropy on decision tokens) + $\mathcal{L}_{traj}^{temp}$ + $\mathcal{L}_{traj}^{spatial}$ (L1 on waypoints/routes)

### Joint Attention Sharing

The UE produces layer-wise KV representations at update time $\tau(t)$, stored in a persistent KV cache:
$$\mathcal{C}^{\tau(t)}=\{K^{l}_{scene}(\tau(t)),V^{l}_{scene}(\tau(t))\}_{l=1}^{L}$$

The AE at time $t$ concatenates its own KV with the cached UE KV:
$$\tilde{K}^{l}(t)=[K^{l}_{scene}(\tau(t))\;\|\;K^{l}_{act}(t)], \quad \tilde{V}^{l}(t)=[V^{l}_{scene}(\tau(t))\;\|\;V^{l}_{act}(t)]$$

Final joint attention:
$$\mathrm{Attn}^{l}(t)=\mathrm{softmax}\!\left(\frac{Q^{l}_{act}(t)\,\tilde{K}^{l}(t)^{\top}}{\sqrt{d}}\right)\tilde{V}^{l}(t)$$

**Attention pattern rules**:
- Within-task (intra-modal): **bidirectional attention**
- Cross-task (UE→DM, UE+DM→planning): **causal attention** (planning conditioned on understanding and decision; not vice versa)

This design allows AE to draw on UE's frozen scene reasoning without the UE being updated.

## Asynchronous Inference

UE runs at low frequency; AE reuses stale cached KV at every high-frequency action step. $\tau(t) \leq t$ — scene context from the most recent UE update.

**Ablation: async (KV cache) vs. synchronized (Table 5):**

| Setting | L2_avg ↓ | Latency ↓ |
|---------|-----------|-----------|
| AutoMoT-S (synchronized, no cache) | 0.322 | 0.38s |
| **AutoMoT (async + KV cache)** | **0.326** | **0.05s** |

+1.24% L2 regression for **86.8% latency reduction (7.6× speedup)**. Decision accuracy equally negligible drop (Table 6: joint accuracy 53.49% → 53.10%).

**Training for asynchrony**: NuSync/PDM-Meta include *temporally asynchronous samples* — historical frames are consecutive but the RGB+BEV pair is selected 0.5–1s ahead. This trains the AE to tolerate temporal misalignment between UE scene context and current action step.

## Key Empirical Finding: Frozen VLM Preserves General Reasoning

The most important contribution. Fine-tuning VLM backbone on AD data yields **catastrophic forgetting** on general reasoning benchmarks with minimal planning benefit (Table 4):

| Benchmark | Task | AutoMoT (frozen) | AutoMoT (AD fine-tuned) | Δ |
|-----------|------|-----------------|------------------------|---|
| LingoQA | AD scene understanding | 67.00 | 67.20 | +0.20 |
| OmniDrive | Counterfactual planning | 18.20 | 67.80 | +49.60 |
| ScienceQA | General knowledge | 88.60 | 87.80 | −0.80 |
| FigureQA | General knowledge | 97.60 | 91.20 | −6.40 |
| **TallyQA** | **General reasoning** | **81.40** | **52.40** | **−35.3%** |
| **InfographicVQA** | **General reasoning** | **89.30** | **50.20** | **−43.8%** |
| **VizWiz** | **General reasoning** | **75.60** | **50.20** | **−33.6%** |

**Interpretation**: AD scene understanding fine-tuning only meaningfully helps action-level tasks (OmniDrive +49.6) while catastrophically degrading high-level general reasoning (TallyQA −35%, InfoVQA −44%). Basic recognition (ScienceQA, FigureQA) is largely preserved; compositional multi-step reasoning collapses.

**Functional boundary principle**: pre-trained VLMs are best suited for high-level scene understanding via semantic prompting alone; fine-tuning should be restricted to action-level components. See [[concepts/vlm-domain-adaptation.md]].

## Datasets

### NuSync (new, released)
- 80.1K samples based on nuScenes
- Input: 4 consecutive historical RGB frames + RGB+BEV pair
- Synchronous and asynchronous variants (0.5s and 1.0s temporal offset)
- Labels: meta-actions over 3s horizon at 1s intervals (lateral: turn left/slight left/straight/slight right/turn right; longitudinal: accelerate/slow/keep/stop) — up to 20 combinations

### PDM-Meta (new, CARLA)
- Based on PDM-Lite; longitudinal decisions only (lateral boundaries ambiguous in sim)

*These are claimed to be the first open-source asynchronous multi-frame meta-action datasets.*

### Training datasets for trajectory planning
- nuScenes: 1,000 urban scenes, 6-camera
- CARLA-Garage: 500K+ frames from CARLA simulator

## Results

### Bench2Drive Closed-loop (Table 1)

| Method | Expert | Modality | VLM | DS ↑ | SR (%) ↑ |
|--------|--------|----------|-----|------|----------|
| DiffusionDrive | PDM-Lite | C+L | — | 77.68 | 57.72 |
| TransFuser++ | PDM-Lite | C+L | — | 84.21 | 67.27 |
| ORION | Think2Drive | C | ✓ | 77.74 | 54.62 |
| AutoVLA | PDM-Lite | C | ✓ | 78.84 | 57.73 |
| SimLingo | PDM-Lite | C | ✓ | 85.07 | 67.27 |
| **AutoMoT** | **PDM-Lite** | **C+L** | **✓** | **87.34** | **70.00** |

AutoMoT leads all VLM-augmented methods and beats SimLingo (+2.27 DS) trained without action-dreamer augmentation. **Caveat**: LinkVLA (91.01 DS / 74.55% SR) is not in the table and likely supersedes AutoMoT on Bench2Drive.

### nuScenes Open-loop (Table 2)

| Method | UE Fine-tuned | L2_avg ↓ | Collision avg ↓ |
|--------|--------------|----------|-----------------|
| UniAD | — | 0.69 | 0.12 |
| DriveVLM-Dual | ✓ | 0.31 | 0.10 |
| SpaceDrive | ✓ | 0.32 | 0.23 |
| OpenREAD | ✓ | 0.36 | 0.11 |
| AutoVLA | ✓ | 0.40 | 0.20 |
| OpenEMMA | ✗ | 2.81 | — |
| **AutoMoT** | **✗** | **0.32** | **0.07** |

AutoMoT achieves the **lowest collision rate (0.07)** despite not fine-tuning the UE — competitive L2 with fine-tuned methods. OpenEMMA (also no fine-tuning) is far worse at 2.81 L2 — showing policy learning, not just VLM prompting, is critical.

### Reasoning Benchmarks (Table 3)

| Method | LingoQA | OmniDrive | CODA-LM | TallyQA | InfoVQA |
|--------|---------|-----------|---------|---------|---------|
| ReCogDrive (UE fine-tuned) | 67.20 | 0.82 | 5.90 | 69.60 | 75.80 |
| Robotron-Drive (UE fine-tuned) | 59.20 | 0.82 | 6.20 | 63.40 | 42.60 |
| OpenEMMA (UE not fine-tuned) | 48.00 | 0.43 | 4.80 | 80.00 | 71.40 |
| **AutoMoT (UE frozen)** | **67.00** | **0.89** | **6.07** | **81.40** | **89.30** |

AutoMoT with frozen UE matches ReCogDrive (fine-tuned) on LingoQA (67.00 vs. 67.20), exceeds on OmniDrive (0.89 vs. 0.82), and vastly outperforms on general-domain benchmarks (TallyQA: 81.4 vs. 69.6; InfoVQA: 89.3 vs. 75.8).

### Component Ablation (Table 8, nuScenes)

| Method | L2@1s | L2@2s | L2@3s | L2_avg |
|--------|-------|-------|-------|--------|
| AutoMoT (full) | 0.14 | 0.29 | 0.54 | 0.32 |
| AutoMoT-R (random UE, no pretrained VLM) | 0.16 | 0.33 | 0.60 | 0.36 |
| AutoMoT-P (no decision-making objective) | 0.14 | 0.30 | 0.58 | 0.34 |

- Pre-trained UE contributes +0.04 L2_avg improvement over random-init UE; degradation worse at longer horizons (long-horizon planning relies heavily on high-quality scene understanding)
- Decision-making objective contributes +0.02 L2_avg improvement — meta-action supervision improves trajectory quality

### Decision-Making (Table 6/7)

**NuSync asynchronous setting** (Table 6):
| Setting | Joint Acc Avg |
|---------|--------------|
| AutoMoT-S (synchronized) | 53.49% |
| AutoMoT (async + KV cache) | 53.10% |

**Senna benchmark** (Table 7):
| Model | Accuracy |
|-------|----------|
| Senna | 88.47% |
| **AutoMoT** | **90.92%** |

## Implementation Details

| Setting | Value |
|---------|-------|
| Understanding Expert | Qwen3-VL-4B (frozen) |
| Action Expert | ~1.6B, trained from scratch |
| UE training | None |
| AE lr | 1e-4 → 2e-5 |
| AE hardware | 8x NVIDIA A100 |
| Trajectory points | 6 temporal + 20 spatial route |
| Decision horizon | 3s @ 1s intervals |
| Modality | Camera + LiDAR |
| Optional head | Diffusion-based planner refiner (Appendix A.3) |

## Limitations

1. **Bench2Drive comparison incomplete** — LinkVLA (91.01 DS) not in Table 1; AutoMoT likely below current SOTA
2. **No NAVSIM evaluation** — cannot compare with DriveFine (90.7), WAM-Flow (90.3), or Curious-VLA (90.3) on PDMS
3. **Requires LiDAR** — Camera + LiDAR modality; not comparable to 1xC camera-only systems
4. **Frozen UE off-domain risk** — UE never adapts to driving distribution; driving-specific symbols/lane markings/traffic signs may not match pre-training coverage
5. **Scene staleness** — async operation tolerates 0.5–1s stale scene context; in fast-changing scenarios (emergency braking, cut-in) this could be unsafe
6. **No RL/GRPO** — purely SFT-based; no reward-aligned fine-tuning (unlike DriveFine, WAM-Flow, Curious-VLA)
7. **Diffusion refiner ambiguity** — Appendix A.3 discusses planning head variants; unclear which is used for Table 1 main results
8. **OmniDrive counterfactual gap** — frozen UE gets 18.20 vs. fine-tuned 67.80 on OmniDrive counterfactual planning — action-level reasoning still requires fine-tuning

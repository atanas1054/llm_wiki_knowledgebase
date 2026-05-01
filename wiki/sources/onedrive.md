---
title: "OneDrive: Unified Multi-Paradigm Driving with Vision-Language-Action Models"
type: source-summary
sources: [raw/papers/OneDrive_ Unified Multi-Paradigm Driving with Vision-Language-Action Models.md]
related: [concepts/dual-system-vla.md, concepts/vlm-domain-adaptation.md, concepts/perception-for-planning.md, concepts/diffusion-planner.md, concepts/navsim-benchmark.md, concepts/nuscenes-waymo-evals.md, concepts/foundation-backbones-for-ad.md, concepts/action-tokenization.md, sources/recogdrive.md, sources/percept-wam.md, sources/unidrivevla.md, sources/automot.md]
created: 2026-05-01
updated: 2026-05-01
confidence: high
---

## Citation

**Paper**: OneDrive: Unified Multi-Paradigm Driving with Vision-Language-Action Models  
**arXiv**: https://arxiv.org/html/2604.17915v1  
**Authors**: Yiwei Zhang, Xuesong Chen, Jin Gao, Hanshi Wang, Fudong Ge, Weiming Hu, Shaoshuai Shi, Zhipeng Zhang  
**Code**: https://github.com/Z1zyw/OneDrive  

## Core Idea

OneDrive argues that the useful transferable part of a pretrained VLM decoder is **causal attention**, not the text-specialized feed-forward network. It keeps a single pretrained VLM decoder as the common attention backbone, inserts visual tokens, structured perception queries, planning queries, and text tokens into one sequence, and handles heterogeneous outputs in one transformer:

- Autoregressive text generation remains native.
- 3D detection and lane queries are decoded in parallel.
- Planning queries produce continuous trajectory points in parallel.
- Shallow mixed layers add query-only self-attention and task-specific FFNs for structured outputs.
- Deeper layers stay closer to the pretrained LLM decoder for text generation.

This is a direct alternative to the dominant fragmented designs: separate VLM + E2E modules, Q-Former cascades, or isolated perception/planning decoders.

## Figures

![[intro-arch-cmp-oneDrive.drawio.png|Figure 1: Dual-system, Q-Former/cascaded, and OneDrive single-decoder architectures.]]

![[ar-paral-cmp.drawio.png|Figure 2: Autoregressive vs. parallel decoding paradigms in driving models.]]

![[unidrivevla-main.drawio.png|Figure 3: OneDrive architecture with image tokens, detection queries, lane queries, planning queries, and text tokens in one causal decoder.]]

## Architecture

The unified sequence is:

$$
\mathbf{Z}=[\mathbf{X}_{img},\mathbf{Q}_{det},\mathbf{Q}_{lane},\mathbf{Q}_{plan},\mathbf{X}_{text}]
$$

Structured queries are placed after image tokens so they can condition on visual context through the original causal self-attention. Planning tokens are placed after perception queries, so trajectory prediction can implicitly condition on detection/lane features through the same attention path.

For image and structured query tokens, OneDrive augments RoPE with 3D positional embeddings:

$$
\mathbf{Q}=\text{RoPE}(\mathbf{X}W_q)+e^{3D},\quad
\mathbf{K}=\text{RoPE}(\mathbf{X}W_k)+e^{3D}
$$

Text tokens keep the original RoPE formulation. Query-token residuals are removed in causal attention to stabilize adaptation.

### Mixed Decoder Layers

OneDrive adds two structured-output adaptations only in shallow layers:

- **Query-only self-attention** among perception queries:

$$
\mathbf{Q}_{perception}=\text{SelfAttn}_q([\mathbf{Q}_{det},\mathbf{Q}_{lane}])
$$

- **Task-specific FFNs** for detection, lane, and planning queries:

$$
\mathbf{Q}'=\text{FFN}_t(\tilde{\mathbf{Q}}),\quad t\in\{\text{det},\text{lane},\text{plan}\}
$$

Text tokens continue using the pretrained language FFNs.

## Training Recipe

Three stages:

| Stage | Trainable emphasis | Objective |
| --- | --- | --- |
| Perception-language pretraining | Frozen ViT; causal attention fine-tuned; LoRA on LLM decoder; perception self-attn/FFNs/heads initialized and trained | $\mathcal{L}_{pretrain}=\lambda_{perc}\mathcal{L}_{perc}+\mathcal{L}_{text}$ |
| Planning adaptation | Add planning queries, planning FFN, planning MLP; LoRA continues; perception modules fixed | $\mathcal{L}_{adaptation}=\lambda_{plan}\mathcal{L}_{plan}+\mathcal{L}_{text}$ |
| Joint fine-tuning | All modules including ViT | $\mathcal{L}_{joint}=\lambda_{perc}\mathcal{L}_{perc}+\lambda_{plan}\mathcal{L}_{plan}+\mathcal{L}_{text}$ |

nuScenes uses InternVL3-1B, 20 epochs per stage, LR $1\times10^{-4}$, batch size 64 on 64 NVIDIA H20 GPUs. NAVSIM uses InternVL3-2B initialized from ReCogDrive and planning queries only, LR $1\times10^{-4}$, batch size 128.

## Tables

### Table 1: Pretrained Weight Transfer Diagnostic

Diagnostic study on transferring pretrained LLM weights to a parallel decoder. Check mark means pretrained initialization; cross means random initialization.

| VLM | Attention | FFN | NDS |
| --- | --- | --- | --- |
| InternVL3-1B | Yes | Yes | 31.95 |
| InternVL3-1B | Yes | No | **32.05** |
| InternVL3-1B | No | Yes | 29.90 |
| InternVL3-1B | No | No | 31.48 |
| Qwen2.5-VL-3B | Yes | Yes | 27.14 |
| Qwen2.5-VL-3B | Yes | No | **31.37** |
| Qwen2.5-VL-3B | No | Yes | 27.95 |
| Qwen2.5-VL-3B | No | No | 30.15 |

**Takeaway**: attention transfers; text-specialized FFNs do not. For Qwen2.5-VL-3B, reusing attention but randomizing FFN is +1.22 NDS over fully random, while reusing both collapses to 27.14.

### Table 2: nuScenes Open-Loop Planning

| Method | Type | Reference | L2 1s | L2 2s | L2 3s | L2 Avg | Col 1s | Col 2s | Col 3s | Col Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DriveVLM | Text | CoRL 2024 | 0.18 | 0.34 | 0.68 | 0.40 | - | - | - | - |
| DriveVLM-Dual | Text | CoRL 2024 | 0.15 | 0.29 | 0.48 | 0.31 | - | - | - | - |
| OmniDrive | Text | CVPR 2025 | 0.14 | 0.29 | 0.55 | 0.33 | 0.00 | 0.13 | 0.78 | 0.30 |
| EMMA | Text | TMLR | 0.14 | 0.29 | 0.54 | 0.32 | - | - | - | - |
| EMMA+ | Text | TMLR | 0.13 | 0.27 | 0.48 | 0.29 | - | - | - | - |
| ImpromptuVLA | Text | NeurIPS 2025 | 0.13 | 0.27 | 0.53 | 0.30 | - | - | - | - |
| SOLVE-VLM | Text | CVPR 2025 | 0.13 | 0.25 | 0.47 | 0.28 | 0.00 | 0.16 | 0.43 | 0.20 |
| VGGDrive | Text | CVPR 2026 | 0.14 | 0.28 | 0.51 | 0.31 | 0.02 | 0.10 | 0.55 | 0.22 |
| UniAD | Action, no ego | CVPR 2023 | 0.59 | 1.01 | 1.48 | 1.03 | 0.16 | 0.51 | 1.64 | 0.77 |
| VAD-Base | Action, no ego | ICCV 2023 | 0.69 | 1.22 | 1.83 | 1.25 | 0.06 | 0.68 | 2.52 | 1.09 |
| BEV-Planner | Action, no ego | CVPR 2024 | 0.30 | 0.52 | 0.83 | 0.55 | 0.10 | 0.37 | 1.30 | 0.59 |
| UniAD | Action | CVPR 2023 | 0.20 | 0.42 | 0.75 | 0.46 | 0.02 | 0.25 | 0.84 | 0.37 |
| VAD-Base | Action | ICCV 2023 | 0.17 | 0.34 | 0.60 | 0.37 | 0.04 | 0.27 | 0.67 | 0.33 |
| AD-MLP | Action | ArXiv 2023 | 0.15 | 0.32 | 0.59 | 0.35 | 0.00 | 0.27 | 0.85 | 0.37 |
| BEV-Planner | Action | CVPR 2024 | 0.16 | 0.32 | 0.57 | 0.35 | 0.00 | 0.29 | 0.73 | 0.34 |
| SOLVE-E2E | Action | CVPR 2025 | 0.14 | 0.28 | 0.50 | 0.31 | 0.04 | 0.17 | 0.68 | 0.30 |
| ColaVLA | Action | CVPR 2026 | 0.14 | 0.27 | 0.50 | 0.30 | 0.04 | 0.17 | 0.47 | 0.23 |
| **OneDrive** | **Action** | - | **0.13** | **0.25** | **0.46** | **0.28** | **0.00** | **0.12** | **0.43** | **0.18** |

OneDrive matches the best text-based L2 average and beats ColaVLA on both average L2 (0.28 vs. 0.30) and average collision rate (0.18 vs. 0.23).

### Table 3: NAVSIM navtest Closed-Loop SFT

| Method | Reference | NC | DAC | TTC | Comfort | EP | PDMS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ego Status MLP | - | 93.0 | 77.3 | 83.6 | 100 | 62.8 | 65.6 |
| VADv2 | ArXiv 2024 | 97.2 | 89.1 | 91.6 | 100 | 76.0 | 65.6 |
| DrivingGPT | ICCV 2025 | 98.9 | 90.7 | 94.9 | 95.6 | 79.7 | 82.4 |
| UniAD | CVPR 2023 | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| TransFuser | PAMI 2022 | 97.7 | 92.8 | 92.8 | 100 | 79.2 | 84.0 |
| PARA-Drive | CVPR 2024 | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 |
| DRAMA | ArXiv 2024 | 98.0 | 93.1 | 94.8 | 100 | 80.1 | 85.5 |
| Hydra-MDP | ArXiv 2024 | 98.3 | 96.0 | 94.6 | 100 | 78.7 | 86.5 |
| DiffusionDrive | CVPR 2025 | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| Qwen2.5-VL-8B | - | 97.8 | 92.1 | 92.8 | 100 | 78.3 | 83.3 |
| InternVL3-8B | - | 97.0 | 92.1 | 91.8 | 100 | 78.9 | 83.3 |
| AutoVLA (SFT) | NeurIPS 2025 | 96.9 | 94.4 | 88.1 | 99.9 | 75.8 | 80.5 |
| ReCogDrive (SFT) | ICLR 2026 | 98.1 | 94.7 | 94.2 | 100 | 80.9 | 86.5 |
| Query Decoder Baseline | - | - | - | - | - | - | 85.0 |
| **OneDrive** | - | **98.4** | **95.2** | **94.9** | **100** | **81.1** | **86.8** |

OneDrive is not a NAVSIM frontier method in the wiki, but it improves over its query-decoder baseline (85.0 -> 86.8) and ReCogDrive SFT (86.5) while staying in a supervised-only regime.

### Table 4: Text Evaluation vs. OmniDrive

| Method | L2 1s | L2 2s | L2 3s | L2 Avg |
| --- | --- | --- | --- | --- |
| OmniDrive-7B | 0.14 | 0.29 | 0.55 | 0.33 |
| **OneDrive-Text-1B** | 0.15 | 0.29 | **0.51** | **0.32** |

The unified perception/planning decoder does not destroy text-conditioned trajectory generation.

### Table 5: Text Supervision Impact

| Task | Text loss | NDS/mAP | L2 1s | L2 2s | L2 3s | L2 Avg | Col 1s | Col 2s | Col 3s | Col Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3D Det | No | 32.31 / 22.47 | - | - | - | - | - | - | - | - |
| 3D Det | Yes | **33.94 / 24.39** | - | - | - | - | - | - | - | - |
| E2E | No | - | 0.13 | 0.27 | 0.51 | 0.31 | 0.04 | 0.19 | 0.98 | 0.40 |
| E2E | Yes | - | 0.13 | 0.27 | 0.51 | 0.31 | **0.02** | 0.20 | **0.85** | **0.36** |

Text supervision acts as a mild regularizer for the shared causal attention.

### Table 6: Planning Loss Weight

| $\lambda_{plan}$ | NDS/mAP | L2 1s | L2 2s | L2 3s | L2 Avg | Col 1s | Col 2s | Col 3s | Col Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.25 | 44.82 / 35.10 | 0.127 | 0.258 | 0.473 | 0.287 | 0.000 | 0.078 | 0.449 | **0.176** |
| 0.5 | 45.45 / 35.58 | 0.126 | 0.255 | 0.467 | 0.283 | 0.000 | 0.039 | 0.488 | **0.176** |
| 1.0 | **45.75 / 35.95** | **0.126** | **0.252** | **0.461** | **0.280** | 0.000 | 0.117 | **0.430** | 0.182 |
| 2.0 | 44.72 / 34.01 | 0.127 | 0.258 | 0.472 | 0.286 | 0.000 | 0.098 | 0.762 | 0.287 |

$\lambda_{plan}=1.0$ is the best balance for NDS/mAP and L2; too much planning weight hurts perception and collision stability.

### Table 7: Inference Latency on H20

| Setting | Method | AR? | Latency |
| --- | --- | --- | --- |
| NAVSIM | ReCogDrive | No | 263 ms |
| NAVSIM | **OneDrive** | No | **156 ms** |
| nuScenes | OmniDrive | Yes | 3727 ms |
| nuScenes | SOLVE-VLM | Yes | 3719 ms |
| nuScenes | ColaVLA | No | 727 ms |
| nuScenes | **OneDrive** | No | **513 ms** |

The headline latency claim is NAVSIM 263 -> 156 ms, roughly 40% lower than ReCogDrive. On nuScenes, OneDrive is 29% faster than ColaVLA while processing all camera views through the LLM path.

### Table 8: Multi-Stage Training

| Stage | NDS/mAP | L2 1s | L2 2s | L2 3s | L2 Avg | Col 1s | Col 2s | Col 3s | Col Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Only Planning Adaptation | - | 0.14 | 0.29 | 0.54 | 0.32 | 0.02 | 0.23 | 1.01 | 0.42 |
| Perception Pretrain | 33.18 / 22.66 | - | - | - | - | - | - | - | - |
| Planning Adaptation | 33.18 / 22.66 | 0.13 | 0.26 | 0.49 | 0.29 | 0.02 | 0.12 | 0.66 | 0.29 |
| Joint Training | **45.75 / 35.95** | **0.13** | **0.25** | **0.46** | **0.28** | **0.00** | **0.12** | **0.43** | **0.18** |

Perception pretraining helps planning before joint training; joint training provides the largest perception and collision-rate gains.

### Table 9: Token Sequence Ablation

| Token sequence | Training | L2 1s | L2 2s | L2 3s | L2 Avg | Col 1s | Col 2s | Col 3s | Col Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Lane -> Det -> Planning | Adaptation | 0.14 | 0.28 | 0.52 | 0.31 | 0.02 | 0.18 | 0.82 | 0.34 |
| Lane -> Det -> Planning | Joint | 0.13 | 0.27 | 0.50 | 0.30 | 0.02 | 0.21 | 0.64 | 0.29 |
| Det -> Lane -> Planning | Adaptation | 0.13 | 0.26 | 0.49 | 0.29 | 0.00 | 0.12 | 0.66 | 0.27 |
| Det -> Lane -> Planning | Joint | **0.13** | **0.25** | **0.46** | **0.28** | **0.00** | **0.12** | **0.43** | **0.18** |

Detection before lane before planning is consistently better than lane before detection.

### Table 10: 3D Detection with InternVL3-ViT

| Method | NDS | mAP |
| --- | --- | --- |
| StreamPETR (frozen) | 31.48 | 20.26 |
| **OneDrive (frozen)** | **33.94** | **24.39** |
| StreamPETR | 45.31 | 36.11 |
| **OneDrive** | **47.73** | **40.03** |

OneDrive beats StreamPETR under the same InternVL3-ViT backbone, but the paper notes that InternVL3-ViT remains weaker than standard detection-oriented ViT-Large backbones because VLM token resolution is lower and pretraining is language-centric.

## Relationships

- **vs. ReCogDrive**: ReCogDrive uses a VLM plus a query/diffusion-style planner. OneDrive folds planning queries into the VLM decoder and reports lower NAVSIM latency.
- **vs. Percept-WAM**: both inject structured perception into a VLM-like system, but Percept-WAM uses World-PV/BEV tokens and a four-query trajectory decoder; OneDrive uses one causal decoder with task-specific query FFNs.
- **vs. UniDriveVLA / AutoMoT**: MoT separates expert streams to avoid interference. OneDrive keeps one attention backbone and isolates heterogeneity via shallow query self-attention plus task FFNs.
- **vs. Reasoning-VLA**: both use query-based parallel action prediction; OneDrive also unifies perception and text generation in the same causal decoder.

## Limitations

1. **Not a NAVSIM frontier result**: 86.8 PDMS is below many wiki methods and is SFT-only. The contribution is architectural unification, not leaderboard dominance.
2. **No RL/RFT stage**: the paper does not test whether the unified decoder remains stable under GRPO or other online RL.
3. **Detection backbone gap**: InternVL3-ViT underperforms detection-specialized ViT-Large backbones; the paper attributes this to reduced token resolution and language-centric pretraining.
4. **Scaling not established**: the future-work section explicitly says larger architectures and datasets remain untested.
5. **NAVSIM setup is simplified**: NAVSIM uses planning queries only, without the full detection/lane query stack used on nuScenes.
6. **No NAVSIM-v2/navhard/Bench2Drive**: robustness and interactive simulation performance are unknown.
7. **Large training budget**: nuScenes training uses 64 H20 GPUs, which limits reproducibility despite code/model availability.


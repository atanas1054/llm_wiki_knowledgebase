---
title: "UniUGP: Unifying Understanding, Generation, and Planning for End-to-end Autonomous Driving"
type: source-summary
sources: [raw/papers/UniUGP_ Unifying Understanding, Generation, and Planing For End-to-end Autonomous Driving.md]
related: [concepts/world-model-for-ad.md, concepts/vlm-domain-adaptation.md, concepts/diffusion-planner.md, sources/recogdrive.md, sources/wam-flow.md]
created: 2026-04-05
updated: 2026-04-05
confidence: high
---

## Citation

Hao Lu, Ziyang Liu, Guangfeng Jiang, Yuanfei Luo, et al. (ByteDance Seed + HKUST-GZ)
arXiv: https://arxiv.org/abs/2512.09864
Project: https://seed-uniugp.github.io/

## One-Line Summary

A unified VLA + world model framework that jointly produces **CoT reasoning, future video, and trajectory** via a three-expert hybrid architecture (MoT + cascaded DiT) — the first system to simultaneously have all four: reasoning, NL interaction, continuous video generation, continuous FM trajectory.

## Problem Statement

Two existing paradigms have complementary strengths:
- **VLA models** (ReCogDrive, AutoVLA, WAM-Flow): strong world knowledge and reasoning from pre-trained VLMs; can't leverage unlabeled video for visual causal learning
- **World models** (Epona, OccWorld): learn visual causal dynamics from next-frame prediction; lack LLM reasoning capabilities

Additionally, long-tail scenarios (rare accidents, small obstacles, ambiguous intersections) are underrepresented in existing benchmarks. UniUGP addresses all three gaps simultaneously.

## What Makes UniUGP Unique (Table 1)

Among all unified models in the comparison, UniUGP is the **only system** that checks all boxes:

| Category | Method | Model | Reasoning | NL Interaction | Generation | Cont. Gen | Plan Method | Cont. Plan |
|----------|--------|-------|:---------:|:--------------:|:----------:|:---------:|:-----------:|:----------:|
| World Model | Epona | — | ✗ | ✗ | Video | ✓ | FM | ✓ |
| VLM | OmniDrive | LLaMA2-7B | ✗ | ✗ | — | — | Text | ✗ |
| VLA | ReCogDrive | Qwen2.5-VL-3B | ✓ | ✗ | — | — | Diffusion | ✓ |
| VLA | AutoVLA | InternVL3-8B | ✓ | ✗ | — | — | Codebook | ✗ |
| VLA | Alpamayo-R1 | Cosmos-Reason | ✓ | ✗ | — | — | FM | ✓ |
| Unified | FSDrive | Qwen2-VL-2B | ✗ | ✗ | Video | ✗ | Text | ✗ |
| **Unified** | **UniUGP** | **Qwen2.5-VL-3B** | **✓** | **✓** | **Video** | **✓** | **FM** | **✓** |

## Architecture: Hybrid Expert System

![Figure 1: UniUGP architecture — three hybrid experts. The Understanding Expert (Qwen2.5-VL-3B) does next-token prediction for causal reasoning. The Planning Expert forms a MoT with the Understanding Expert, predicting future actions via flow matching velocity. The Generation Expert (Wan2.1) is cascaded to produce future video conditioned on understanding states and action embeddings.](<../../raw/assets/x1 2.png>)

### Overview

Three experts with two coupling patterns:

```
[Images + Language] → Understanding Expert (Qwen2.5-VL-3B)
                              ↕ shared attention (MoT)
[Ego state + noised actions] → Planning Expert (flow matching)
                              ↓
Understanding hidden states + Action embeddings
                              ↓
                     Generation Expert (Wan2.1 DiT) → Future video
```

### Understanding Expert
- Backbone: **Qwen2.5-VL-3B**
- Input: multi-frame images (via ViT), text instructions (via tokenizer) → cross-modal tokens $\mathbf{x}^{und}$
- Output: next-token prediction logits for CoT reasoning and scene understanding
- Loss: $\mathcal{L}_{und} = \mathbb{E}_{x_i^{und}}[-\log P(x_i^{und} | \mathbf{x}_{<i}^{und})]$

### Planning Expert (MoT coupled with Understanding)
- Uses **continuous flow matching**: noised action $\mathbf{a}_\tau = \tau\mathbf{a} + (1-\tau)\epsilon$, $\tau \sim [0,1]$
- History ego states + noised actions projected to planning tokens $\mathbf{x}^{plan}$
- **MoT (Mixture-of-Transformers)**: understanding and planning tokens attend jointly via shared MHSA, then split into modality-specific FFNs:
  $$\mathbf{h}_o^{und}, \mathbf{h}_o^{plan} = \text{MHSA}([\text{QKV}_{und}(\mathbf{x}^{und}), \text{QKV}_{plan}(\mathbf{x}^{plan})])$$
  $$\mathbf{h}_{ffn}^{und} = \text{FFN}_{und}(\mathbf{h}_o^{und}), \quad \mathbf{h}_{ffn}^{plan} = \text{FFN}_{plan}(\mathbf{h}_o^{plan})$$
- Output: denoising vector field $\mathbf{u}_\tau^{plan}$; unprojected to action space
- Loss: $\mathcal{L}_{plan} = \mathbb{E}[||\mathbf{u}_\tau^{plan} - (\epsilon - \mathbf{a})||_2]$

**Key property of MoT vs. cross-attention coupling**: understanding and planning tokens co-attend at every layer, not just via a one-directional conditioning. This is tighter integration than ReCogDrive (which uses cross-attention from planner to VLM hidden states).

### Generation Expert (cascaded)
- Based on **Wan2.1** (DiT architecture, pre-trained weights inherited)
- Input: VAE-encoded history frames $\mathbf{v}^{hist}$ + noised future frames $\mathbf{v}_\tau^{fut}$
- Condition: understanding hidden states $\mathbf{h}^{und}$ + projected action embeddings $\mathbf{A}$
  $$\mathbf{u}_\tau^{gen} = \mathcal{W}([\mathbf{v}^{hist}, \mathbf{v}_\tau^{fut}], [\mathbf{h}^{und}, \mathbf{A}], \tau)$$
- During inference: $\mathbf{A} = \text{Proj.}(\hat{\mathbf{a}})$ from planning expert output
- During training: 50% ground truth actions, 50% single-step denoised predictions (scheduled sampling)
- **Can be disabled at inference** (e.g., mobile deployment) without affecting planning/understanding performance
- Loss: $\mathcal{L}_{gen} = \mathbb{E}[||\mathbf{u}_\tau^{gen} - (\epsilon - \mathbf{v}^{fut})||_2]$

**Overall loss**: $\mathcal{L}_{total} = 0.3\mathcal{L}_{und} + 0.5\mathcal{L}_{plan} + 0.2\mathcal{L}_{gen}$

**Why generation helps planning**: Adding the generation expert improves planning L2 from 1.72→1.45. The mechanism: the world model forces the VLA to attend to distant objects and visual causal dynamics that matter for future video consistency — which also matter for trajectory accuracy.

## Long-tail Dataset Construction

![Figure 2: Dataset construction pipeline — Step 1: data collection from long-tail sources (accident prediction, small obstacles, accident relationships, reasoning & instruction following). Step 2: data processing into four task categories (Perception/Understanding, Future Trajectory-Based CoT Reasoning, Planning, Instruction Following).](<../../raw/assets/x2 1.png>)

### Data Sources
- **DADA-2000, Lost and Found, StreetHazards, SOM, AADV** — rare obstacles, out-of-distribution objects, pre-accident sequences
- **Waymo-E2E** — 4,021 segments with rare/high-risk events at <0.003% frequency; 5-second trajectory prediction

### Task Categories

1. **Perception/Understanding**: True/False and multiple-choice QA on:
   - Small long-tail object recognition
   - Accident subject relationships (who/what is involved)
   - Accident prediction (pre-collision sequences)

2. **CoT Reasoning** — 4-component traces:
   - Scene description (time, weather, traffic condition)
   - Object detection (traffic signals, dynamic agents, rare objects)
   - Intention inference (how objects affect ego vehicle's future driving)
   - Action decision (CoT reasoning grounded in predicted future trajectories)

3. **Planning**: Future 5-second trajectory on Waymo-E2E long-tail segments

4. **Instruction following**: Trajectory matched to NL navigation commands (Turn Left / Go Straight / Turn Right) — the only unified AD system with this capability

![Figure 5: Example long-tail perception QA — True/False questions about accident scenarios from anomaly datasets.](<../../raw/assets/x5 1.png>)

![Figure 8 (appendix): Example CoT reasoning outputs — the model generates grounded natural language explanations such as "The all-way stop sign legally requires the ego vehicle to come to a complete stop. Cross traffic and a pedestrian nearby require stopping for safety before proceeding."](<../../raw/assets/x8 1.png>)

CoT construction: a frontier VLM is prompted with the known future trajectory + scene context to generate the reasoning trace; manually calibrated afterwards.

## Four-Stage Training Curriculum

| Stage | Experts trained | Data | Steps | Notes |
|-------|----------------|------|-------|-------|
| 1 | Understanding only | 10 datasets + ImpromptuVLA (80K clips) | 1M | Basic scenario understanding |
| 2 | Generation + Planning | nuScenes, nuPlan, Waymo, Lyft, Cosmos | 4M | Visual dynamics + motion planning |
| 3 | Understanding only | Custom CoT dataset | 1M | Causal reasoning capability |
| 4 | All three | Mix stages 1:2:3 at 0.1:0.4:0.5 | 4M | Multi-capability fusion |

Full hyperparameters (Table 2):

| Hyperparameter | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|----------------|---------|---------|---------|---------|
| Trained components | Understanding | Gen + Plan | Understanding | All three |
| Learning rate | 1e-4 | 1e-4 | 1e-4 | 1e-4 |
| Understanding resolution | 224×224 | — | 224×224 | 224×224 |
| Generation resolution | — | 512×512 | — | 512×512 |
| Batch size | 64 | 64 | 64 | 64 |
| Training steps | 1M | 4M | 1M | 4M |
| GPU resources | 8×8×80GB | 8×8×80GB | 8×8×80GB | 8×8×80GB |

Total hardware: **512 GPUs (8 nodes × 8 × 80GB)**.

## Results

### Custom Long-tail Benchmark (Table 3)

| Model | Small obj. ↑ | Relationship ↑ | Acc. Pred. ↑ | CoT GPT ↑ | CoT BLEU ↑ | Plan L2 (3s) ↓ | Follow L2 (3s) ↓ |
|-------|:-----------:|:--------------:|:------------:|:---------:|:----------:|:--------------:|:----------------:|
| GPT-4o | 64.2% | 63.5% | 72.8% | 0.55 | 0.125 | 2.63 | 2.58 |
| Qwen-2.5-VL-72B | 75.8% | 74.9% | 81.5% | 0.72 | 0.188 | 1.94 | 1.89 |
| UniUGP w/o CoT | 86.5% | 85.7% | 93.2% | 0.83 | 0.218 | 1.58 | 1.53 |
| UniUGP w/o Gen | 83.7% | 82.9% | 90.6% | 0.80 | 0.203 | 1.72 | 1.67 |
| **UniUGP (full)** | **89.3%** | **88.6%** | **95.8%** | **0.88** | **0.240** | **1.45** | **1.40** |

![Figure 3: Ablation on world model presence — with generation expert, the VLA focuses more on distant, causally relevant objects (e.g., a distant vehicle about to cut in). Without it, attention is more near-field.](<../../raw/assets/x3 1.png>)

### nuScenes Planning (Table 4)

| Method | Input | Aux. Supervision | L2 1s ↓ | L2 2s ↓ | L2 3s ↓ | Avg L2 ↓ | Col. 1s ↓ | Col. 2s ↓ | Col. 3s ↓ | Avg Col. ↓ |
|--------|-------|-----------------|---------|---------|---------|---------|---------|---------|---------|-----------|
| ST-P3 | Camera | Map+Box+Depth | 1.33 | 2.11 | 2.90 | 2.11 | 0.23 | 0.62 | 1.27 | 0.71 |
| UniAD | Camera | Map+Box+Motion+Occ | 0.48 | 0.96 | 1.65 | 1.03 | 0.05 | 0.17 | 0.71 | 0.31 |
| OccWorld | Camera | 3D-Occ | 0.52 | 1.27 | 2.41 | 1.40 | 0.12 | 0.40 | 2.08 | 0.87 |
| GenAD | Camera | Map+Box+Motion | 0.36 | 0.83 | 1.55 | 0.91 | 0.06 | 0.23 | 1.00 | 0.43 |
| Doe-1 | Camera* | QA | 0.50 | 1.18 | 2.11 | 1.26 | 0.04 | 0.37 | 1.19 | 0.53 |
| Epona | Camera* | None | 0.61 | 1.17 | 1.98 | 1.25 | 0.01 | 0.22 | 0.85 | 0.36 |
| **UniUGP** | **Camera*** | **QA** | **0.58** | **1.14** | **1.95** | **1.23** | **0.01** | **0.19** | **0.81** | **0.33** |

Camera* = front camera only. UniUGP beats Epona and Doe-1 under comparable input, but well below multi-camera SOTA.

### Future Frame Generation — nuScenes (Table 5)

| Method | Type | Resolution | FID ↓ | FVD ↓ |
|--------|------|-----------|-------|-------|
| DriveDreamer | Diff | 128×192 | 52.6 | 452.0 |
| Drive-WM | Diff | 192×384 | 15.8 | 122.7 |
| GenAD | Diff | 256×448 | 15.4 | 184.0 |
| GEM | Diff | 576×1024 | 10.5 | — |
| Doe-1 | AR | 384×672 | 15.9 | — |
| FSDrive | AR | 128×192 | 10.1 | — |
| Epona | Diff | 576×1024 | 7.5 | 82.8 |
| **UniUGP** | **AR+Diff** | **512×512** | **7.4** | **75.9** |

SOTA on both FID and FVD, despite lower resolution than Epona.

![Figure 4: Trajectory-controllable generation — modifying the trajectory fed into the generation model changes the generated future video, demonstrating the generation expert is physically grounded in the planned path.](<../../raw/assets/x4 1.png>)

### DriveLM GVQA (Table 6)

| Method | Acc. ↑ | GPT ↑ | BLEU-1 ↑ | ROUGE-L ↑ | CIDEr ↑ | Match ↑ | Final ↑ |
|--------|:------:|:-----:|:--------:|:---------:|:-------:|:-------:|:-------:|
| DriveLM baseline | 0.00 | 0.65 | 0.05 | 0.08 | 0.10 | 0.28 | 0.32 |
| Cube-LLM | 0.39 | 0.89 | 0.16 | 0.20 | 0.31 | 0.36 | 0.50 |
| TrackingMeetsLMM | 0.60 | 0.58 | 0.72 | 0.72 | 0.04 | 0.36 | 0.52 |
| SimpleLLM4AD | 0.66 | 0.57 | 0.76 | 0.73 | 0.15 | 0.35 | 0.53 |
| OmniDrive | 0.70 | 0.65 | 0.52 | 0.73 | 0.13 | 0.37 | 0.56 |
| FSDrive | 0.72 | 0.63 | 0.76 | 0.74 | 0.17 | 0.39 | 0.57 |
| **UniUGP** | **0.74** | **0.64** | **0.78** | **0.76** | **0.19** | **0.41** | **0.59** |

## Key Ablation Findings

- **Generation expert** improves planning L2 (1.72→1.45) and all understanding/reasoning metrics — visual causal learning transfers to trajectory quality
- **CoT module** independently improves planning (L2 1.58→1.45 when added on top of gen)
- Both together achieve the best performance — understanding, generation, and planning are mutually reinforcing

## Limitations

1. **No NAVSIM evaluation** — no PDMS numbers; can't compare against WAM-Flow (90.3) or ReCogDrive (89.6) on the community-standard closed-loop benchmark.
2. **nuScenes planning not SOTA** — L2=1.23m vs GenAD's 0.91m; the unified model trades planning precision for breadth.
3. **Custom benchmark self-evaluation** — the primary showcase benchmark is self-constructed; no blind external validation.
4. **Extreme compute** — 512 GPUs for up to 4M steps; not replicable at academic scale.
5. **Wan2.1 deployment cost** — full model requires a large video generation model; inference time not reported.
6. **50% teacher forcing for generation** — using ground-truth action embeddings 50% of training time creates train/test gap.
7. **CoT BLEU is low (0.24)** — lexical divergence from reference even when semantically sound; GPT scoring is subjective.

## Connections to Other Wiki Pages

- [[concepts/world-model-for-ad.md]] — UniUGP is the primary example of world model + VLA integration in this wiki
- [[concepts/vlm-domain-adaptation.md]] — MoT architecture, multi-stage training, CoT integration approach
- [[concepts/diffusion-planner.md]] — flow matching for continuous trajectory planning; MoT as tighter coupling than cross-attention
- [[sources/recogdrive.md]] — ReCogDrive uses cross-attention VLM→DiT coupling; UniUGP uses MoT (tighter) + adds video generation
- [[sources/wam-flow.md]] — discrete flow matching (WAM-Flow) vs. continuous flow matching (UniUGP planning expert)

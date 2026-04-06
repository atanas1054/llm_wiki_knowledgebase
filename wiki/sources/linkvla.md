---
title: "LinkVLA: Unifying Language-Action Understanding and Generation for Autonomous Driving"
type: source-summary
sources: [raw/papers/Unifying Language-Action Understanding and Generation for Autonomous Driving.md]
related: [concepts/diffusion-planner.md, concepts/vlm-domain-adaptation.md, sources/orion.md]
created: 2026-04-05
updated: 2026-04-05
confidence: high
---

**arXiv**: https://arxiv.org/html/2603.01441v1  
**Affiliation**: Zhejiang University + Li Auto  
**Benchmark**: Bench2Drive (CARLA V2) only; no NAVSIM or nuScenes evaluation

---

## Core Thesis

Current VLA methods suffer from two problems:
1. **Language-action misalignment** — the model may reason "change lane left" correctly but output a lane-keep trajectory
2. **AR generation inefficiency** — sequential waypoint decoding is too slow for deployment

LinkVLA solves both simultaneously with three interlocking innovations: a shared discrete codebook for language and action tokens, a bidirectional language-action training objective, and a coarse-to-fine (C2F) parallel decoder.

---

## Figure 1 — Motivation and Overview

![[raw/assets/Unifying Language-Action Understanding and Generation for Autonomous Driving/x1 8.png]]

Shows latency vs. driving performance trade-off. LinkVLA C2F achieves the best of both worlds: 48ms latency at 91.01 DS, compared to SimLingo (34ms, 85.07 DS) and ORION (65ms, 77.74 DS).

---

## Figure 2 — Architecture Overview

![[raw/assets/Unifying Language-Action Understanding and Generation for Autonomous Driving/x3 6.png]]

Three core components illustrated: (1) unified tokenization — log BEV grid merged with text vocabulary; (2) bidirectional objective — action generation and action understanding swapping L/A roles; (3) C2F decoder — endpoint + linear coarse path + parallel refinement.

---

## Architecture

**Backbone**: InternVL2-1B (InternViT-300M-448px + Qwen2-0.5B-Instruct)  
**LoRA**: rank=32, α=64  
**Output per frame**: 20 geometric path tokens + 10 temporal waypoint tokens

---

## Innovation 1: Unified Tokenization Framework

### Shared Codebook

Trajectory waypoints are quantized into a discrete BEV grid and merged with the text vocabulary into a single codebook of size $K = K_\text{text} + K_\text{action}$. Both language and action tokens are processed within the same VLM, eliminating the architectural modality gap.

### Log Coordinate Transform

Naive uniform BEV grids allocate equal resolution everywhere — wasteful for far-field points, insufficient near-field. LinkVLA applies a signed logarithmic transform:

$$z' = \text{sign}(z) \cdot \log(1 + k \cdot |z|), \quad k = 5$$

This concentrates grid cells near the ego vehicle where precision matters most and compresses the far-field into fewer tokens.

**BEV space**: $x \in [0, 50]$m, $y \in [-30, 30]$m → transformed → uniform 0.1-step grid → **56 × 101 = 5,656 action tokens**

### Figure S1 — Uniform vs. Log Grid

![[raw/assets/Unifying Language-Action Understanding and Generation for Autonomous Driving/x6 4.png]]

Left: uniform grid — dense far-field tokens wasted. Right: log grid — dense near-field, sparse far-field, aligned with waypoint distribution.

### Spatial Soft-Labeling

Instead of a one-hot training target, the GT token gets a Gaussian soft target distribution over all grid neighbors:

$$q(a) = \frac{1}{Z} \exp\left(-\frac{\|\text{pos}(a) - \text{pos}(a_{gt})\|_2^2}{2\sigma^2}\right)$$

Parameters: radius $R = 10$ cells, $\sigma = 1.2$. The generation loss becomes a soft cross-entropy:

$$\mathcal{L}_\text{generation} = -\sum_{a \in \mathcal{C}_\text{action}} q(a) \log p(a)$$

This embeds spatial topology into the loss — the model learns that adjacent tokens are semantically similar — making it robust to minor GT annotation errors.

---

## Innovation 2: Bidirectional Language-Action Objective

### Figure 3 — Bidirectional Training

![[raw/assets/Unifying Language-Action Understanding and Generation for Autonomous Driving/x4 5.png]]

Inspired by the image-captioning / text-to-image duality in vision-language modeling. For VLAs, both directions exist:

| Direction | Task | Input → Output | Loss |
|-----------|------|---------------|------|
| **Generation** | Action generation | $(V, L) \to A$ | $\mathcal{L}_\text{generation}$ |
| **Understanding** | Action captioning | $(V, A) \to L$ | $\mathcal{L}_\text{understanding} = -\sum_j \log p(l_j \mid V, A, l_{<j})$ |

Total loss: $\mathcal{L}_\text{total} = \mathcal{L}_\text{generation} + \lambda \mathcal{L}_\text{understanding}$

**Implementation**: same decoder handles both tasks by randomly swapping the roles of $L$ and $A$ as the prediction target. During a training step, a $(V, L, A)$ tuple is randomly reordered to either $[V, A, L]$ (supervise $L$) or $[V, L, A]$ (supervise $A$). No additional data curation required.

**Why it works**: forcing the model to recover the language instruction from a given trajectory enriches the semantic grounding of action token embeddings — they become intrinsically linked to linguistic descriptions rather than being purely spatial coordinates.

---

## Innovation 3: Coarse-to-Fine (C2F) Action Generation

AR generation of $T$ waypoints requires $T$ sequential forward passes. C2F collapses this to **two passes**.

### Training With Endpoint Prior

The ground-truth sequence is reordered as $\{w_T, w_1, w_2, \ldots, w_{T-1}\}$ during training. A special goal token is placed at position 0, teaching the model to associate it with endpoint prediction.

For refinement training: a coarse trajectory is simulated via linear interpolation from GT endpoint, quantized to tokens, and the model is trained to map these coarse tokens to fine-grained GT waypoints.

### Inference — Two Passes

**Pass 1 — Endpoint prediction:**
$$\hat{w}_T = \text{VLM}(\text{goal\_token} \mid V, L, \text{CoT})$$

**Coarse trajectory construction** (linear interpolation):
$$w_i^\text{coarse} = w_0 + \frac{i}{T}(w_T - w_0), \quad i \in \{1, \ldots, T\}$$

**Pass 2 — Parallel refinement:**
Tokenized $\mathcal{W}_\text{coarse}$ fed as input → VLM predicts all $T$ refined waypoints **in parallel** via cross-attention on visual-language context.

**Result**: AR 361ms → C2F 48ms (**86% reduction**).

---

## Inference Pipeline

At inference, LinkVLA uses a Chain-of-Thought (CoT) approach:
1. Model generates a **textual rationale** (driving commentary)
2. Conditioned on commentary + vision, C2F generates the final trajectory

Note: CoT latency is **excluded** from the reported 48ms figure (variable, query-dependent).

---

## Results

### Table 1 — Bench2Drive SOTA

| Method | DS ↑ | SR (%) ↑ | Efficiency ↑ | Comfort ↑ | Multi-Ability Mean ↑ |
|--------|-------|----------|------------|---------|---------------------|
| DriveTransformer | 63.46 | 35.01 | 100.64 | 20.78 | 38.60 |
| ORION | 77.74 | 54.62 | 151.48 | 17.38 | 54.72 |
| AutoVLA | 78.84 | 57.73 | 146.93 | 39.33 | — |
| SimLingo | 85.07 | 67.27 | 259.23 | 33.67 | 67.28 |
| **LinkVLA (Ours)** | **91.01** | **74.55** | **255.84** | **34.62** | **73.40** |

+5.94 DS and +7.28% SR over prior SOTA (SimLingo). High efficiency (255.84 ≈ SimLingo's 259.23) and substantially better comfort than ORION (34.62 vs 17.38).

### Multi-Ability Breakdown

| Scenario | SimLingo | ORION | **LinkVLA** |
|----------|---------|-------|------------|
| Merging | 53.75 | 25.00 | **60.00** |
| Overtake | 68.89 | 71.11 | **80.00** |
| Brake | 81.67 | 78.33 | **93.33** |
| Give-Way | 50.00 | 30.00 | **50.00** |
| Traffic-Sign | 82.11 | 69.15 | **83.68** |
| **Mean** | 67.28 | 54.72 | **73.40** |

### Table 2 — Latency vs. Performance

| Method | Type | Latency ↓ | DS ↑ |
|--------|------|----------|------|
| ORION | VAE | 65ms | 77.74 |
| SimLingo | MLP | 34ms | 85.07 |
| LinkVLA AR | AR | 361ms | 90.66 |
| **LinkVLA C2F** | **C2F** | **48ms** | **91.01** |

C2F: 86% faster than AR, +13.27 DS over ORION, +5.94 DS over SimLingo at only 14ms more latency.

---

## Qualitative Results

### Figure 4 — Diverse Instruction Following

![[raw/assets/Unifying Language-Action Understanding and Generation for Autonomous Driving/x5 5.png]]

Shows trajectory generation adhering to various language instructions (speed up, slow down, lane change) in complex environments with obstacles and dynamic agents.

### Figure S2 — Additional Closed-Loop Scenarios

![[raw/assets/Unifying Language-Action Understanding and Generation for Autonomous Driving/x7 4.png]]

Intersection navigation and obstacle avoidance scenarios during CARLA closed-loop evaluation.

---

## Ablation Studies

### Table 3 — Instruction Following (Action Dreaming)

| Config | Faster | Slower | Target Speed | Lane Change | Object | Stop | **Mean** |
|--------|--------|--------|-------------|------------|--------|------|---------|
| Baseline | 81.42 | 61.83 | 66.27 | 75.53 | 74.69 | 60.93 | 70.11 |
| + Token | 88.44 | 65.24 | 63.37 | 88.49 | 84.34 | 99.88 | 81.63 |
| + C2F | 93.16 | 55.86 | 69.24 | 95.45 | 85.38 | 92.14 | 81.87 |
| **+ Align.** | **96.48** | **65.57** | **74.73** | **97.42** | **91.41** | **97.34** | **87.16** |

Tokenization is the dominant contributor (+11.5 pp). Alignment adds balanced gains across all categories.

### Table 4 — Language Quality (DriveLM-hard VQA + Commentary)

| Config | VQA SPICE | VQA BLEU | VQA R-L | Comm. SPICE | Comm. BLEU | Comm. R-L |
|--------|-----------|----------|---------|------------|-----------|----------|
| Baseline | 66.7 | 68.9 | 71.5 | 49.2 | 60.3 | 64.3 |
| + Token | 69.7 | 70.5 | 73.1 | 53.3 | 63.7 | 68.0 |
| + C2F | 71.3 | 69.9 | 73.4 | 53.6 | 61.6 | 67.3 |
| **+ Align.** | **73.0** | **74.7** | **77.0** | **57.4** | **65.7** | **70.8** |

Alignment improves language metrics — confirming that the action understanding objective enriches shared token representations.

### Table 5 — Closed-Loop Ablation

| Config | DS | SR (%) |
|--------|-----|--------|
| Baseline | 85.07 | 67.27 |
| + Token | 89.57 | 73.18 |
| + C2F | 89.85 | 72.27 |
| **+ Align.** | **91.01** | **74.55** |

### Table 6 — Soft-Label Effect

| | DS | SR (%) |
|--|-----|--------|
| Without soft label | 90.85 | 72.73 |
| **With soft label** | **91.01** | **74.55** |

+0.16 DS, +1.82% SR — modest but consistent.

### Table 7 — Navigation Modality

| | DS | SR (%) |
|--|-----|--------|
| GPS target point | 91.01 | 74.55 |
| Navigation command | 91.25 | 73.18 |

Both modalities work equivalently — same model handles either input form.

### Table S1 — Action Codebook Size

| k | Tokens | DS | SR (%) |
|---|--------|-----|--------|
| 5.0 | 5,656 | **91.01** | **74.55** |
| 10.0 | 7,245 | 89.85 | 70.45 |

More tokens (finer resolution) hurts — a larger discrete vocabulary confuses the language model without meaningful precision gains.

### Table S2 — Soft-Label Spread σ

| σ | DS | SR (%) |
|---|-----|--------|
| 1.2 | **91.01** | **74.55** |
| 3.0 | 89.73 | 69.55 |

Tighter Gaussian (σ=1.2) outperforms broad smoothing — overly soft targets lose the spatial precision benefit.

---

## Limitations

1. **CoT latency excluded** — 48ms omits text rationale generation (variable overhead). Real deployment latency is higher; not directly comparable to non-CoT systems
2. **Bench2Drive / CARLA only** — no NAVSIM, no nuScenes, no real-world evaluation; cross-benchmark comparison with WAM-Flow or ReCogDrive is impossible
3. **1B parameter backbone** — smallest InternVL2 variant; complex scene reasoning may be capacity-limited relative to 7B+ models used in other papers
4. **C2F coarse path bias** — linear interpolation assumes straight motion; refinement must overcome this prior in sharp turns or emergency stops, potentially restricting trajectory diversity
5. **Tokenization resolution ceiling** — finer grids (k=10) degrade performance; quantization error is irreducible for precise near-ego control
6. **SimLingo latency** — SimLingo still wins on raw latency (34ms vs 48ms); for strict real-time constraints, MLP-head approaches retain the edge

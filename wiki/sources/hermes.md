---
title: "HERMES: A Holistic End-to-End Risk-Aware Multimodal Embodied System with VLMs for Long-Tail Autonomous Driving"
type: source-summary
sources: [raw/papers/HERMES_ A Holistic End-to-End Risk-Aware Multimodal Embodied System with Vision–Language Models for Long-Tail Autonomous Driving.md]
related: [concepts/vlm-domain-adaptation.md, concepts/inference-time-safety.md]
created: 2026-04-05
updated: 2026-04-05
confidence: high
---

**arXiv**: https://arxiv.org/html/2602.00993v1  
**Affiliation**: University of Wisconsin–Madison (Civil & Environmental Engineering)  
**Benchmark**: WOD-E2E (Waymo Open Dataset for E2E driving) — **not NAVSIM or Bench2Drive**  
**Focus**: Long-tail safety-critical scenarios in real-world mixed-traffic environments

---

## Core Thesis

Existing VLM-powered E2E methods focus on nominal driving and lack dedicated mechanisms for rare but high-impact events (pedestrian incursions, sudden cut-ins, sensor occlusions). HERMES addresses this with a **teacher-student distillation** pattern: a large cloud VLM (Qwen3-VL-Flash) pre-annotates training data with structured safety context **offline**, which is then encoded and injected into a lightweight student planner through a dedicated risk-aware fusion pipeline.

---

## Figure 1 — Paradigm Comparison

![[raw/assets/HERMES_ A Holistic End-to-End Risk-Aware Multimodal Embodied System with Vision–Language Models for Long-Tail Autonomous Driving/x1 9.png]]

| Paradigm | Approach | Gap |
|----------|----------|-----|
| (a) Traditional E2E | Unified differentiable pipeline | Lacks semantic reasoning for long-tail |
| (b) VLM/VLA methods | Foundation model for reasoning + planning | Overlooks explicit safety/risk modeling |
| **(c) HERMES** | **VLM-generated long-tail context → risk-aware E2E student** | Bridges semantic safety reasoning and actionable planning |

---

## Figure 2 — HERMES Architecture Overview

![[raw/assets/HERMES_ A Holistic End-to-End Risk-Aware Multimodal Embodied System with Vision–Language Models for Long-Tail Autonomous Driving/x2 7.png]]

Two components: (1) **Long-Tail Instruction Embedding** (offline teacher) and (2) **Tri-Modal Driving Module** (online student).

---

## Component 1: Long-Tail Instruction Embedding (Teacher)

### Prompt Design

A carefully crafted prompt template elicits exactly five structured outputs from Qwen3-VL-Flash:

**Table I — Structured VLM Prompt:**

| Category | Field | Description |
|----------|-------|-------------|
| Input | System role | "Autonomous driving planner specializing in safety-critical long-tail analysis" |
| Input | Camera images | 4 grouped inputs: front trio (FL/F/FR), rear trio (RL/R/RR), side-left, side-right |
| Input | Motion states | Historical positions [x,y], velocities [v_x,v_y], accelerations [a_x,a_y] + high-level intent |
| **Long-Tail Scene Context** | Scene description | Per-viewpoint hazard-centric descriptions: occlusions, abnormal behaviors, rare objects, adverse conditions, compound edge cases |
| **Long-Tail Planning Context** | Risk level | Low / Medium / High based on scene complexity, VRU presence, collision likelihood |
| **Long-Tail Planning Context** | Intention | Go straight / Turn left / Turn right / Stop |
| **Long-Tail Planning Context** | High-level plan | Concise directive (yield, brake, proceed cautiously…) |
| **Long-Tail Planning Context** | Plan rationale | Grounded in observable elements only — no unobserved inference |

Output is strictly constrained to five fields in a fixed format for consistent downstream integration.

### Text Encoding

Both outputs are encoded by **BGE-M3** (a strong sentence embedding model) into:
- `scene_emb` $\in \mathbb{R}^{B \times D_\text{text}}$ — Long-Tail Scene Context embedding
- `risk_plan_emb` $\in \mathbb{R}^{B \times D_\text{text}}$ — Long-Tail Planning Context embedding

These are fixed offline and stored with the training data. The student model treats them as read-only inputs.

---

## Component 2: Tri-Modal Driving Module (Student)

A sequential fusion pipeline processing three modalities: vision, historical state, and semantic text embeddings.

### Vision Encoder

ViT backbone processes 8-camera surround-view images (224×224). Patch tokens (excluding CLS token) are projected to $D_\text{model}$ and augmented with **learnable camera-specific and viewpoint-specific embeddings** to encode multi-view spatial priors.

Output: $v_\text{tokens} \in \mathbb{R}^{B \times N \times D_\text{model}}$, $v_\text{global} \in \mathbb{R}^{B \times D_\text{model}}$ (mean-pooled).

### State Encoder

16-frame historical vehicle states (position, velocity, acceleration) projected to latent space → multi-layer transformer encoder → mean-pooled over time dimension.

Output: $s_\text{global} \in \mathbb{R}^{B \times D_\text{model}}$

### Scene Fusion

Text-guided aggregation of visual tokens conditioned on scene semantics:

$$c_\text{scene} = \text{Attn}(\text{scene\_vec}, v_\text{tokens})$$
$$\tilde{s} = \text{MLP}([v_\text{global} \, \| \, c_\text{scene}])$$
$$\text{scene\_context} = v_\text{global} + \tilde{s}$$

The cross-attention uses `scene_vec` (projected from `scene_emb`) as the query and $v_\text{tokens}$ as keys and values. The residual connection to $v_\text{global}$ preserves low-level visual information when semantic guidance is uninformative.

### Planning Context Fusion

$$\text{plan\_context} = \text{MLP}([\text{scene\_context} \, \| \, s_\text{global}])$$

Merges scene-aware visual representation with historical motion dynamics.

### Figure 3 — Intent Modulator

![[raw/assets/HERMES_ A Holistic End-to-End Risk-Aware Multimodal Embodied System with Vision–Language Models for Long-Tail Autonomous Driving/x3 7.png]]

**FiLM-style** (feature-wise linear modulation) conditioning on discrete intent $\in \{0,1,2,3\}$ (unknown/straight/left/right):

$$\text{plan\_context\_intent} = \text{plan\_context} \odot \sigma(\text{scale}) + \text{shift}$$

Intent embedding → 2-layer MLP → scale and shift parameters. Sigmoid ensures scale stays in (0,1), providing smooth multiplicative modulation.

### Figure 4 — Risk Planning Cross-Attention

![[raw/assets/HERMES_ A Holistic End-to-End Risk-Aware Multimodal Embodied System with Vision–Language Models for Long-Tail Autonomous Driving/x4 6.png]]

The key safety mechanism. Rather than directly substituting semantic guidance into the planning representation, it uses a **scaled residual** to limit its influence:

$$\text{plan\_context\_ctrl} = \text{LN}(\text{plan\_context\_intent} + \alpha \cdot c)$$

where $c$ is the cross-attention output (query = `plan_context_intent`, KV = `risk_vec` from `risk_plan_emb`), and $\alpha = 0.3$ is a fixed control scalar.

**Design rationale**: α-scaling prevents the model from over-relying on potentially noisy or overly conservative VLM planning instructions in common conditions, while still allowing semantic injection to modulate trajectory in safety-critical ones.

### Temporal Decoder

Learnable temporal queries attend to `plan_context_ctrl` via a query-based transformer decoder → MLP → relative displacement sequence → accumulated → trajectory $\in \mathbb{R}^{B \times 20}$ (5-second, 20-step horizon).

---

## Dataset: WOD-E2E

| Split | Segments |
|-------|---------|
| Train | 2,037 |
| Val | 479 |
| **Test** | **1,505 (access restricted — all results on val)** |

**10 long-tail scenario categories**: Interaction, Construction, Cyclists, Pedestrian, Single-Lane Maneuvers, Multi-Lane Maneuvers, Special Vehicles, Cut-ins, Others, Foreign Object Debris (FOD).

Real-world Waymo data, ~12 hours, diverse urban and suburban environments.

---

## Metrics

| Metric | Description |
|--------|-------------|
| **RFS** (Rater Feedback Score) | Primary metric — evaluates against multiple human-rated reference trajectories; more safety-aligned than single-GT L2 |
| ADE@3s, ADE@5s | Average Displacement Error at 3s/5s |
| FDE@3s, FDE@5s | Final Displacement Error at 3s/5s |

---

## Implementation

- Single NVIDIA RTX 4090, batch=16, 3 epochs
- Adam, lr=1e-4, cosine annealing to 5e-6, warmup over first epoch
- Gradient clipping: $\ell_2$ norm ≤ 5.0
- α = 0.3 (fixed)
- Images: 224×224, ImageNet normalization
- Teacher VLM: Qwen3-VL-Flash (offline)
- Text encoder: BGE-M3

---

## Results

### Table II — Overall Performance

| Method | RFS ↑ | ADE@3s ↓ | ADE@5s ↓ | FDE@3s ↓ | FDE@5s ↓ |
|--------|--------|---------|---------|---------|---------|
| UniAD | 5.78 | 6.50 | 10.81 | 12.14 | 21.41 |
| VAD | 4.45 | 3.19 | 5.81 | 5.85 | 12.30 |
| LightEMMA | 6.11 | 3.07 | 5.41 | 6.07 | 11.32 |
| **HERMES** | **6.81** | **0.82** | **1.82** | **1.73** | **4.81** |

⚠️ **Caveat**: LightEMMA is **zero-shot** (no fine-tuning on WOD-E2E). UniAD and VAD use "limited adaptation". HERMES trains end-to-end on the full training split. The dramatic ADE/FDE improvement partly reflects training advantage, not architecture alone.

### Table III — Category-Wise RFS

| Method | Global | Interaction | Construction | Cyclists | Pedestrian | Single-Lane | Multi-Lane | Spec. Veh. | Cut-ins | Others | FOD |
|--------|--------|------------|-------------|---------|-----------|------------|-----------|-----------|---------|--------|-----|
| UniAD | 5.78 | 6.08 | 6.07 | 5.50 | 6.45 | 5.78 | 5.82 | 6.04 | 4.64 | 5.91 | 5.23 |
| VAD | 4.45 | 4.42 | 4.11 | 4.40 | 4.66 | 4.68 | 5.46 | 4.30 | 4.50 | 4.33 | 4.30 |
| LightEMMA | 6.11 | 6.32 | 6.55 | 5.83 | 6.12 | 5.24 | 6.38 | 6.18 | 6.00 | 6.44 | 6.16 |
| **HERMES** | **6.81** | **6.96** | **7.26** | **6.78** | **7.11** | **7.28** | **6.71** | **6.72** | **6.54** | 6.41 | 6.37 |

HERMES leads 8/10 categories. Weakest: Others (6.41 vs 6.44 LightEMMA), FOD (6.37 vs 6.16 LightEMMA). Biggest wins: Construction (+0.71), Pedestrian (+0.99), Single-Lane (+2.04 vs LightEMMA 5.24).

### Table IV — Ablation

| Variant | RFS | ADE@3s | ADE@5s | FDE@3s | FDE@5s |
|---------|-----|--------|--------|--------|--------|
| No Instruction | 6.21 | 0.92 | 1.94 | 1.98 | 4.82 |
| No Intent | 6.56 | 0.91 | 1.94 | 2.00 | 4.82 |
| **No State** | **5.51** | **2.48** | **4.22** | **4.69** | **8.59** |
| **Full HERMES** | **6.81** | **0.82** | **1.82** | **1.73** | **4.81** |

**Largest impact**: No State (−1.30 RFS, 3× ADE degradation). Historical motion is the most critical modality.  
**Second**: No Instruction (−0.60 RFS). Semantic embeddings improve safety-aligned behavior beyond what pure vision+motion provides.  
**Smallest**: No Intent (−0.25 RFS). FiLM conditioning helps but is not load-bearing.

---

## Qualitative Results

### Figure 5 — Nighttime Driving in Heavy Rain

![[raw/assets/HERMES_ A Holistic End-to-End Risk-Aware Multimodal Embodied System with Vision–Language Models for Long-Tail Autonomous Driving/x5 6.png]]

Long-Tail Scene Context identifies low visibility and wet road reflections. HERMES produces a conservative, smooth turning trajectory with controlled speed — appropriate for conditions where visual cues are unreliable.

### Figure 6 — Extremely Low-Visibility Residential Street

![[raw/assets/HERMES_ A Holistic End-to-End Risk-Aware Multimodal Embodied System with Vision–Language Models for Long-Tail Autonomous Driving/x6 5.png]]

Despite no immediate obstacle, the risk-aware planning context triggers cautious, centered trajectory to account for potential pedestrians and reduced friction.

### Figure 7 — Construction Zone with Lane Channelization

![[raw/assets/HERMES_ A Holistic End-to-End Risk-Aware Multimodal Embodied System with Vision–Language Models for Long-Tail Autonomous Driving/x7 5.png]]

Temporary barricades and non-standard lane markings — rare in training data. Long-Tail Planning Context anticipates lane change needed; HERMES generates a smooth lateral maneuver at controlled speed.

### Figure 8 — Complex Urban Intersection

![[raw/assets/HERMES_ A Holistic End-to-End Risk-Aware Multimodal Embodied System with Vision–Language Models for Long-Tail Autonomous Driving/x8 4.png]]

Wide intersection with crosswalks and latent pedestrian risk. HERMES maintains a stable straight trajectory while encoding readiness to yield — appropriate moderate-risk behavior.

---

## Limitations

1. **Offline-only VLM reasoning** — annotations are generated at training time; novel test-time scenarios outside the annotation distribution receive no fresh risk reasoning. The risk awareness is baked into training-data embeddings.
2. **No closed-loop evaluation** — open-loop on WOD-E2E val only. Paper explicitly defers closed-loop to future work.
3. **Baseline fairness caveat** — zero-shot LightEMMA vs. fully trained HERMES; UniAD/VAD receive "limited adaptation". ADE/FDE gap partly reflects training regime differences.
4. **Fixed α=0.3 scalar** — risk cross-attention control is hand-tuned, not adaptive. The model explicitly annotates risk level (low/medium/high) but does not use it to modulate α dynamically.
5. **FOD and "Others" weakness** — VLM cannot provide useful priors for completely novel objects with no linguistic analog (Foreign Object Debris). Categories requiring pure geometric reasoning rather than semantic labeling benefit less.
6. **Single GPU, 3 epochs** — small-scale training; generalization to harder or more diverse long-tail distributions uncertain.

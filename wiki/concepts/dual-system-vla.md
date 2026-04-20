---
title: Dual-System VLA for Autonomous Driving
type: concept
sources: [raw/papers/Senna-2_ Aligning VLM and End-to-End Driving Policy for Consistent Decision Making and Planning.md, raw/papers/AutoMoT_ A Unified Vision-Language-Action Model with Asynchronous Mixture-of-Transformers for End-to-End Autonomous Driving.md, raw/papers/UniDriveVLA_ Unifying Understanding, Perception, and Action Planning for Autonomous Driving.md, raw/papers/From Representational Complementarity to Dual Systems_ Synergizing VLM and Vision-Only Backbones for End-to-End Driving.md]
related: [sources/senna2.md, sources/recogdrive.md, sources/automot.md, sources/unidrivevla.md, sources/hybriddriveVLA.md, concepts/vlm-domain-adaptation.md, concepts/diffusion-planner.md, concepts/rl-for-ad.md, concepts/perception-for-planning.md, concepts/best-of-n.md]
created: 2026-04-05
updated: 2026-04-19
confidence: high
---

## What Is a Dual-System VLA?

A dual-system vision-language-action (VLA) model separates autonomous driving into two explicit subsystems:

1. **High-level system (VLM)**: reasons about the scene and produces a discrete, interpretable driving decision (e.g., "Decelerate, Turn Left")
2. **Low-level system (E2E policy)**: generates a continuous trajectory guided by the VLM's decision

This contrasts with **single-system** approaches (WAM-Flow, UniUGP planning expert) where a single model predicts the trajectory directly, and with **unified** approaches (ReCogDrive) where the VLM's hidden states are used as a conditioning signal without explicit decision alignment.

A third structural paradigm has emerged: **Mixture-of-Transformers (MoT)** (AutoMoT, UniDriveVLA), where a single model hosts multiple expert streams with decoupled parameters but shared global attention. This is neither purely dual-system (no separate VLM + E2E modules) nor shared-weight single-system (parameters are decoupled per task). See [MoT Paradigm](#mot-paradigm-automot-unidrivevla) below.

## The Consistency Gap Problem

Without explicit alignment between the two systems, the VLM decision and the E2E trajectory can contradict each other:
- VLM outputs "Decelerate, Go Straight" → E2E planner accelerates or turns
- VLM outputs "Turn Left" → trajectory goes straight

This gap **weakens top-down guidance**: the VLM is part of the architecture but not actually controlling behavior. It also undermines interpretability — the stated decision doesn't match the action.

**Root cause**: the VLM and E2E planner optimize different loss functions on different representations. Without a mechanism that explicitly penalizes mismatches, nothing forces them to agree.

## Architecture Pattern

```
Sensor input ──┬── VLM ──────────────┐
               │   (discrete decision) │
               │         ↓            │
               │   Decision Adapter   │
               │   (decision tokens + │
               │    VLM hidden states)│
               │         ↓            ▼
               └── E2E backbone ──→ Planner → Trajectory
                   (perception tokens)
```

### VLM Role
- Produces structured **meta-actions**: discrete combinations of speed control × direction control
- Serves as interpretable human-machine interface
- In Senna-2: Qwen2.5-VL-3B, 20 possible meta-actions (4 speed × 5 direction)

### Decision Adapter
Bridges the VLM and E2E policy by converting discrete decisions into continuous conditioning features. In Senna-2, two complementary token types:
- **VLM tokens**: projected from VLM hidden states — semantic context
- **Decision tokens**: learnable category embeddings indexed by decoded meta-action — explicit categorical signal

### E2E Policy Injection
Two mechanisms in Senna-2:
- **AdaLN** (Adaptive Layer Norm): VLM condition globally modulates the planner — sets the "tone" for the whole trajectory
- **Cross-attention**: E2E perception features provide fine-grained spatial grounding


## Consistency Alignment Methods

### Kinematic Mapping $f_K$
A deterministic function converting a continuous planned trajectory into its corresponding meta-action category. Key tool for measuring and enforcing consistency. Used for:
- Generating VLM training labels from GT trajectories (Stage 1)
- Checking whether E2E output matches VLM decision (Stage 2)
- Propagating optimized trajectory decisions back to update VLM (Stage 3)

**Limitation**: maps continuous trajectories to discrete categories — boundary cases may flip, introducing label noise.

### Open-Loop Alignment (Senna-2 Stage 2)
Selective training based on consistency:
$$\mathcal{C}(\tau, d) = \begin{cases} 1 & f_K(\tau) = d \\ 0 & \text{otherwise} \end{cases}$$
$$\mathcal{L}_{stage2} = (1 - \mathcal{C}(\tau, d))(\mathcal{L}_{E2E} + \gamma \mathcal{L}_{VLM})$$

- **Consistent samples**: zero loss — treated as self-reinforcing implicit expert signals
- **Inconsistent samples**: full supervised correction

This is elegant: the alignment objective is binary and requires no additional labels beyond what's already used in pre-training.

### Closed-Loop Alignment (Senna-2 Stage 3: HRL)
Bottom-up hierarchical optimization in 3DGS photorealistic environments:
1. **Low-level** (E2E planner): safety reward (longitudinal penalty if TTC < 3s) + efficiency reward (extension if too slow)
2. **High-level** (VLM): align to match the optimized trajectory: $\mathcal{L}_{high} = -\log P(f_K(\tau) | Q)$

Contrasts with NAVSIM-based GRPO (WAM-Flow, ReCogDrive):
- 3DGS agents **react to ego** — the environment is interactive
- VLM sees **photorealistic video** — can run visual reasoning on real-looking frames
- **Hierarchical**: low-level optimized first, high-level updated to match (bottom-up)
- NAVSIM GRPO: all-at-once policy gradient on the full planner using simulator rewards

## Tradeoffs vs. Single-System Approaches

| Aspect | Dual-System | Single-System |
|--------|-------------|---------------|
| Interpretability | High — decision is explicit and inspectable | Low — black box |
| Top-down controllability | High — user can override VLM decision | None |
| Consistency enforcement | Possible (with explicit alignment) | N/A (no separate decision) |
| Optimization complexity | High — two subsystems must be jointly aligned | Lower |
| Runtime cost | VLM + E2E (may need async) | E2E only (or unified model) |
| Failure modes | Decision-planning gap; VLM latency | Exposure bias (AR); precision (diffusion) |

## Asynchronous Operation
In practice, the VLM cannot run at the E2E policy's frequency (10 Hz) on edge hardware. Solution: a **memory bank** caches VLM features; the E2E planner uses cached features at full speed. VLM refreshes less frequently. This introduces a **staleness tradeoff** — VLM decision may be from an older frame, but the trajectory remains current.

**AutoMoT** ([[sources/automot.md]]) implements the most principled version of this pattern using a **layer-wise shared KV cache**. Rather than caching final-layer VLM outputs, AutoMoT caches the UE's per-layer key-value pairs $\mathcal{C}^{\tau(t)} = \{K^l_{scene}(\tau(t)), V^l_{scene}(\tau(t))\}_{l=1}^L$ and concatenates them into the AE's attention computation at every layer:
$$\tilde{K}^l(t)=[K^l_{scene}(\tau(t))\;\|\;K^l_{act}(t)], \quad \mathrm{Attn}^l(t)=\mathrm{softmax}\!\left(\frac{Q^l_{act}(t)\,\tilde{K}^l(t)^\top}{\sqrt{d}}\right)\tilde{V}^l(t)$$

This enables AE to run at high frequency (0.05s latency) while UE updates at low frequency — **86.8% latency reduction (7.6× speedup)** vs. synchronous execution with only +1.24% L2 degradation. AutoMoT also trains on *asynchronous samples* (UE context 0.5–1s ahead of AE step), teaching the AE to tolerate temporal misalignment explicitly.

## State of the Art (as of April 2026)

### Closed-Loop 3DGS Benchmark (Senna-2 paper)

| Method | AF-CR ↓ | CR ↓ | Safety@1 ↑ |
|--------|--------|-----|-----------|
| RAD (RL-based E2E) | 0.113 | 0.281 | 0.613 |
| Senna | 0.111 | 0.310 | 0.638 |
| **Senna-2** | **0.077** | **0.269** | **0.667** |

Senna-2 beats both RL-only (RAD) and the original dual-system (Senna) — the alignment strategy contributes beyond RL alone.

## MoT Paradigm: AutoMoT + UniDriveVLA {#mot-paradigm-automot-unidrivevla}

Mixture-of-Transformers provides a third structural path: instead of a separate VLM + E2E pipeline, a single model hosts multiple expert transformer streams that attend to each other via controlled masking. Parameters are decoupled across streams but computation is unified in a single forward pass.

### Why MoT?

The core motivation differs between papers:
- **AutoMoT**: efficiency — frozen UE runs at low frequency, lightweight AE runs at high frequency using cached KV pairs; avoids catastrophic forgetting
- **UniDriveVLA**: accuracy — joint optimization of spatial perception and semantic reasoning in shared weights causes representation interference (cosine similarity → 1); expert decoupling eliminates this conflict

### UniDriveVLA's MoT Design

Three experts: Understanding (und), Perception (per), Action (act). Each has its own QKV projections, FFN, and normalization. All tokens attend globally via Masked Joint Attention with asymmetric visibility:

| Expert | Sees | Purpose |
|--------|------|---------|
| und | Causal self only | Preserves VLM pretraining; semantic reasoning |
| per | und + self | Acquires semantic context for spatial queries |
| act | und + per + self | Aggregates both for trajectory generation |

**Key insight**: und is fully protected from per and act — the VLM's causal language modeling is never disrupted by spatial perception gradients.

**Evidence of benefit** (Table 7 of UniDriveVLA):

| Architecture | General VQA↑ | DriveBench↑ | L2(m)↓ | CR(%)↓ |
|---|---|---|---|---|
| Shared-Weight | 31.1% | 50.8% | 0.641 | 0.175 |
| **MoT** | **45.5%** | **54.9%** | **0.533** | **0.140** |
| **Δ** | **+14.4pp** | **+4.1pp** | **−0.108m** | **−0.035** |

General VQA improvement (+14.4pp) is primarily driven by preventing feature collapse in the understanding expert — the expert streams maintain low cosine similarity across layers rather than collapsing to shared representations.

### Comparison: MoT Designs

| Feature | AutoMoT | UniDriveVLA |
|---------|---------|------------|
| # experts | 2 (UE + AE) | 3 (und + per + act) |
| VLM training | Frozen | LoRA Stage 2; frozen Stage 3 |
| Motivation | Efficiency (async) | Accuracy (anti-interference) |
| Async execution | ✓ layer-wise KV cache (7.6×) | ✗ |
| Perception stream | ✗ | ✓ (sparse queries) |
| Evaluation | Bench2Drive | Bench2Drive + nuScenes |
| Result | 87.34 DS | 78.37 DS |

AutoMoT achieves higher DS partly because it uses KV-cache async execution and a purpose-built 1.6B action expert; UniDriveVLA adds a perception stream but is newer and uses Bench2Drive Think2Drive demonstrations.

## Comparison of Dual-System and MoT Designs

| Feature              | Senna (v1)            | Senna-2                                | ReCogDrive                       | AutoMoT                                      | UniDriveVLA                          | HybridDriveVLA / DualDriveVLA           |
| -------------------- | --------------------- | -------------------------------------- | -------------------------------- | -------------------------------------------- | ------------------------------------ | --------------------------------------- |
| VLM backbone         | —                     | Qwen2.5-VL-3B                          | InternVL3-8B                     | Qwen3-VL-4B                                  | Qwen3-VL-2B/8B                       | InternVL-2B (VLM) + ViT-large           |
| VLM training         | Fine-tuned            | Fine-tuned                             | Fine-tuned                       | **Frozen**                                   | LoRA→Frozen                          | Fine-tuned (RecogDrive recipe)          |
| Decision type        | Meta-actions          | Meta-actions (4×5)                     | Text + reasoning                 | Meta-actions (20 combos)                     | Continuous FM trajectory             | Full trajectory (each branch)           |
| Planner              | E2E (diffusion)       | DiT (residual diffusion)               | DiT (cross-attention)            | AE from scratch + optional diffusion refiner | FM action expert (act stream)        | DiT (separate for each branch)          |
| VLM→planner bridge   | Decision conditioning | Decision Adapter (tokens + AdaLN)      | Cross-attention to hidden states | Layer-wise shared KV cache                   | Masked Joint Attention               | Trajectory scorer (no weight sharing)   |
| Explicit consistency | ✗                     | ✓ (kinematic mapping + selective loss) | ✗                                | ✗                                            | ✗                                    | ✗ (complementarity instead)             |
| RL alignment         | ✗                     | ✓ (HRL with 3DGS)                      | ✓ (GRPO with NAVSIM)             | ✗                                            | ✗                                    | ✗ (scorer only)                         |
| Async execution      | Heuristic cache       | Heuristic cache                        | No                               | ✓ Layer-wise KV cache (7.6× speedup)         | ✗                                    | ✓ DualDriveVLA (15% VLM; 3.2× speedup) |
| Perception stream    | ✗                     | ✗                                      | ✗                                | ✗                                            | ✓ (sparse 5-task queries)            | ✗                                       |
| System type          | Dual                  | Dual                                   | Single (tight)                   | MoT (dual)                                   | MoT (unified 3-expert)               | Parallel complementary branches         |

## Representational Complementarity: HybridDriveVLA / DualDriveVLA {#complementarity-hybriddriveVLA}

This paper takes a fundamentally different framing from Senna-2's alignment paradigm. Instead of forcing VLM decisions and E2E trajectories to *agree*, it treats the VLM and ViT branches as **complementary candidate generators** and asks: can we exploit the diversity between them?

### The Key Empirical Finding

Plugging InternVL-2B (VLM) and ViT-large into the same RecogDrive diffusion planner produces **behaviorally complementary trajectories**:

- VLM tends to be more aggressive (faster, more willing to merge/accelerate)
- ViT tends to be more conservative (more likely to brake and yield)
- The ground-truth expert trajectory **often lies between** the two styles
- Each side decisively outperforms the other on ~2–3% of test scenarios (|ΔPDMS| > 20%)

Oracle best-of-2 (pick the better per scenario): **93.58 PDMS**, up from 90.80 (VLM single). This exploitable gap is the paper's central finding.

### Why This Is Different from Traditional Dual-System

Traditional dual-system (Senna-2): VLM provides discrete meta-action → E2E executes it. Goal: make the trajectory **consistent with** the VLM decision.

Complementarity framing (HybridDriveVLA): VLM and ViT each produce **complete independent trajectories**. Goal: **select between** them (and interpolations) using trajectory-level signals.

| Dimension | Senna-2 (alignment) | HybridDriveVLA (complementarity) |
|---|---|---|
| VLM output | Discrete meta-action | Full trajectory |
| ViT/E2E output | Full trajectory | Full trajectory |
| Goal | Consistency (VLM → trajectory) | Selection (best trajectory wins) |
| Key tool | Kinematic mapping + selective loss | Trajectory scorer |
| Interaction | Top-down guidance | Parallel candidate generation |
| Inference cost | VLM + E2E + adapter | 2× (both branches) or ~1.15× (DualDriveVLA) |

### HybridDriveVLA

**Step 1 — Candidate construction**: interpolate between VLM and ViT endpoints along the style axis:
$$\tau_\alpha = \alpha \cdot \tau_\text{ViT} + (1-\alpha) \cdot \tau_\text{VLM}, \quad \alpha \in \{0.1, \ldots, 0.9\}$$

11-candidate set: both endpoints + 9 interpolations. The expert often lies in the interior of this segment.

**Step 2 — Scorer selection**: DrivoR-style trajectory scorer predicts PDMS sub-score components from decoded waypoints + scene tokens. Scorer is explicitly separated from the generator (re-embeds finalized trajectories rather than reading generator latents).

**Result**: 92.10 PDMS on NAVSIM-v1 (new SOTA in paper's comparison table, which includes DiffusionDriveV2 91.2 and iPad 91.7).

### DualDriveVLA: Fast–Slow Deployment

Run ViT by default; if scorer confidence $\hat{s}(\tau_\text{ViT}) < \gamma$, invoke VLM + full 11-candidate selection.

- At γ that routes 15% of scenarios to the VLM: **91.00 PDMS** at **3.2× throughput** vs. VLM-only
- All performance gain from HybridDriveVLA is preserved with 85% ViT-only fast-path acceptance

### Representation Analysis Findings (RQ1)

Backbone-level VLM–ViT CKA: **~0.22** (low).  
DiT-level (after policy training) CKA: **~0.54** (substantially higher).

The planner compresses heterogeneous visual signals into a more shared decision space. Despite this, the residual mismatch is sufficient to produce complementary behaviors.

**Key negative result**: trying to predict per-scenario winners using representation features alone (SAE shared/unique energies, CCA statistics, Random Forest, attention gate) yields at most 90.96 PDMS — barely above the 90.80 VLM baseline and far below the 93.58 oracle. Representation statistics are poor predictors of trajectory superiority; trajectory-level scoring is necessary.

## Open Questions

- Can the kinematic mapping $f_K$ be learned (soft/continuous) rather than rule-based, to avoid category-boundary noise?
- Does dual-system consistency generalize to more fine-grained decision spaces (beyond 20 meta-actions)?
- Can the VLM be distilled into the E2E policy after alignment training, eliminating the runtime VLM latency?
- How does dual-system alignment interact with GRPO-style reward shaping (used in WAM-Flow/ReCogDrive)?
- Does MoT's anti-interference benefit hold at larger scales (>8B) where shared-weight models also benefit from more parameters?
- Can UniDriveVLA's perception + action MoT be combined with Senna-2's consistency alignment for further gains?

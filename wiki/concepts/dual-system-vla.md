---
title: Dual-System VLA for Autonomous Driving
type: concept
sources: [raw/papers/Senna-2_ Aligning VLM and End-to-End Driving Policy for Consistent Decision Making and Planning.md]
related: [sources/senna2.md, sources/recogdrive.md, concepts/vlm-domain-adaptation.md, concepts/diffusion-planner.md, concepts/rl-for-ad.md]
created: 2026-04-05
updated: 2026-04-05
confidence: high
---

## What Is a Dual-System VLA?

A dual-system vision-language-action (VLA) model separates autonomous driving into two explicit subsystems:

1. **High-level system (VLM)**: reasons about the scene and produces a discrete, interpretable driving decision (e.g., "Decelerate, Turn Left")
2. **Low-level system (E2E policy)**: generates a continuous trajectory guided by the VLM's decision

This contrasts with **single-system** approaches (WAM-Flow, UniUGP planning expert) where a single model predicts the trajectory directly, and with **unified** approaches (ReCogDrive) where the VLM's hidden states are used as a conditioning signal without explicit decision alignment.

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

## Comparison of Dual-System Designs

| Feature | Senna (v1) | Senna-2 | ReCogDrive |
|---------|-----------|---------|-----------|
| VLM backbone | — | Qwen2.5-VL-3B | InternVL3-8B |
| Decision type | Meta-actions | Meta-actions (4×5) | Text trajectory + reasoning |
| Planner | E2E (diffusion) | DiT (residual diffusion) | DiT (cross-attention to VLM) |
| VLM→planner bridge | Decision conditioning | Decision Adapter (tokens + AdaLN) | Cross-attention to hidden states |
| Explicit consistency | ✗ | ✓ (kinematic mapping + selective loss) | ✗ |
| RL alignment | ✗ | ✓ (HRL with 3DGS) | ✓ (GRPO with NAVSIM) |
| System type | Dual | Dual | Single (tight coupling) |

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

## State of the Art (as of April 2026)

### Closed-Loop 3DGS Benchmark (Senna-2 paper)

| Method | AF-CR ↓ | CR ↓ | Safety@1 ↑ |
|--------|--------|-----|-----------|
| RAD (RL-based E2E) | 0.113 | 0.281 | 0.613 |
| Senna | 0.111 | 0.310 | 0.638 |
| **Senna-2** | **0.077** | **0.269** | **0.667** |

Senna-2 beats both RL-only (RAD) and the original dual-system (Senna) — the alignment strategy contributes beyond RL alone.

## Open Questions

- Can the kinematic mapping $f_K$ be learned (soft/continuous) rather than rule-based, to avoid category-boundary noise?
- Does dual-system consistency generalize to more fine-grained decision spaces (beyond 20 meta-actions)?
- Can the VLM be distilled into the E2E policy after alignment training, eliminating the runtime VLM latency?
- How does dual-system alignment interact with GRPO-style reward shaping (used in WAM-Flow/ReCogDrive)?

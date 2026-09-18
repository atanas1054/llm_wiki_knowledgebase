---
title: Dual-System VLA for Autonomous Driving
type: concept
sources: [raw/papers/Drive-HWM_ Hierarchical World Models for Dynamic-Latent Guided Autonomous Driving.md, raw/papers/LWDrive_ Layer-Wise World-Model-Guided Vision-Language ModelPlanning for Autonomous Driving.md, raw/papers/BrainWAM_ Action-Space Coordination of Semantic Priors and Predictive Dynamics for Autonomous Driving.md, raw/papers/Senna-2_ Aligning VLM and End-to-End Driving Policy for Consistent Decision Making and Planning.md, raw/papers/AutoMoT_ A Unified Vision-Language-Action Model with Asynchronous Mixture-of-Transformers for End-to-End Autonomous Driving.md, raw/papers/UniDriveVLA_ Unifying Understanding, Perception, and Action Planning for Autonomous Driving.md, raw/papers/From Representational Complementarity to Dual Systems_ Synergizing VLM and Vision-Only Backbones for End-to-End Driving.md, raw/papers/OneDrive_ Unified Multi-Paradigm Driving with Vision-Language-Action Models.md, raw/papers/DriveWAM_ Video Generative Priors Enable Scalable World-Action Modeling for Autonomous Driving.md]
related: [sources/drive-hwm.md, sources/lwdrive.md, sources/brainwam.md, sources/senna2.md, sources/recogdrive.md, sources/automot.md, sources/unidrivevla.md, sources/hybriddriveVLA.md, sources/onedrive.md, sources/drivewam.md, concepts/vlm-domain-adaptation.md, concepts/diffusion-planner.md, concepts/rl-for-ad.md, concepts/perception-for-planning.md, concepts/best-of-n.md, concepts/world-model-for-ad.md]
created: 2026-04-05
updated: 2026-09-14
confidence: high
---

## What Is a Dual-System VLA?

A dual-system vision-language-action (VLA) model separates autonomous driving into two explicit subsystems:

1. **High-level system (VLM)**: reasons about the scene and produces a discrete, interpretable driving decision (e.g., "Decelerate, Turn Left")
2. **Low-level system (E2E policy)**: generates a continuous trajectory guided by the VLM's decision

This contrasts with **single-system** approaches (WAM-Flow, UniUGP planning expert) where a single model predicts the trajectory directly, and with **unified** approaches (ReCogDrive) where the VLM's hidden states are used as a conditioning signal without explicit decision alignment.

A third structural paradigm has emerged: **Mixture-of-Transformers (MoT)** (AutoMoT, UniDriveVLA), where a single model hosts multiple expert streams with decoupled parameters but shared global attention. This is neither purely dual-system (no separate VLM + E2E modules) nor shared-weight single-system (parameters are decoupled per task). See [MoT Paradigm](#mot-paradigm-automot-unidrivevla) below.

A fourth, more recent alternative is **single-decoder unification** (OneDrive): rather than separating systems or experts, image tokens, perception queries, planning queries, and text tokens all pass through one causal VLM decoder. It preserves the pretrained attention backbone and isolates structured-output heterogeneity with shallow query self-attention plus task-specific FFNs.

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

## Closed-Loop 3DGS Results (single-source, Senna-2 paper)

The table below comes entirely from Senna-2's own reconstructed-3DGS evaluation and has no independent replication in the wiki — no other ingested paper reports AF-CR / CR / Safety@1 on a comparable 3DGS environment. It is evidence for the alignment mechanism, not a benchmark standing.

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

## OneDrive: Single-Decoder Unification

**OneDrive** ([[sources/onedrive.md]]) takes the opposite route from dual-system designs: it removes the separate structured decoder. The same causal VLM decoder receives image tokens, detection queries, lane queries, planning queries, and text tokens. Structured queries condition on images through pretrained causal attention, while query-only self-attention and task-specific FFNs supply the parallel-prediction behavior that language FFNs cannot provide.

| Feature | OneDrive |
| --- | --- |
| VLM backbone | InternVL3-1B for nuScenes; InternVL3-2B initialized from ReCogDrive for NAVSIM |
| System split | None: one causal decoder |
| Structured outputs | Detection queries, lane queries, planning queries |
| Text generation | Preserved in the same decoder |
| Key diagnostic | Pretrained attention transfers; pretrained FFNs often hurt |
| NAVSIM result | 86.8 PDMS SFT, below frontier but above query-decoder baseline 85.0 |
| Latency | 156 ms on NAVSIM vs. ReCogDrive 263 ms |

This makes OneDrive a useful counterpoint to AutoMoT and UniDriveVLA. MoT prevents interference by decoupling expert parameters; OneDrive keeps one shared attention backbone and moves heterogeneity into shallow query interaction and task FFNs.

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

## Inverted Dual System: DriveWAM's Advisory VLM {#inverted-dual-system-drivewam}

Every dual-system design above puts the VLM **on top**: it decides, and the low-level policy executes. DriveWAM ([[sources/drivewam.md]]) inverts the hierarchy. A pretrained video diffusion transformer is the policy and produces both the future video and the ego action; a **frozen** Qwen3-VL-8B sits beside it and contributes two sentences of guidance per 4-second chunk.

| Dimension | Senna-2 (alignment) | AutoMoT (async MoT) | DriveWAM (advisory) |
|---|---|---|---|
| Who plans | E2E planner, guided by VLM meta-action | Action expert, conditioned on cached UE context | Video DiT, from its own generated future |
| VLM output | Discrete meta-action (4×5) | Hidden KV pairs per layer | Free-text, two sentences, <50 words |
| Bridge | Decision Adapter (tokens + AdaLN) | Layer-wise shared KV cache | Cross-attention with block-diagonal text mask |
| VLM training | Fine-tuned | Frozen | Frozen, and outside the gradient path entirely |
| Update rate | Per frame (cached) | Low frequency, async | Once per 4s chunk (125 ms, amortized) |
| Consistency enforcement | Explicit (kinematic mapping + selective loss) | None | None — guidance is a soft condition, never verified against the action |

Three properties distinguish it:

**Text as the interface.** The bridge is natural language, not embeddings or categories. The VLM could be replaced without retraining the policy, and the guidance is human-readable at every decision step — stronger interpretability than hidden-state conditioning, weaker controllability than Senna-2's meta-actions since nothing forces the trajectory to obey the text.

**Temporal locality is enforced structurally.** Because guidance is regenerated per chunk, a block-diagonal text mask restricts chunk $k{+}1$ to attend only to $g_k$. Without it, full-clip parallel training would let early chunks read guidance generated at later decision steps — a causality leak specific to chunk-wise guidance that clip-level conditioning never faces.

**No consistency mechanism, and the paper does not claim one.** This is the clearest structural gap versus Senna-2: DriveWAM never checks whether the generated action matches the guidance text. The guidance ablation shows it helps (ADE@4s 0.92 → 0.83 at 100k clips), but "helps on average" is weaker than Senna-2's per-sample consistency objective, and there is no reported measurement of how often the trajectory contradicts its own stated intent.

## BrainWAM: Coordination in Action Space, Not Representation Space {#brainwam}

[[sources/brainwam.md]] is a dual-system design where **neither system is the fast one and neither is the slow one**. The two branches are a VLA (Qwen3-VL-4B, semantic priors) and a WAM (Wan2.2-TI2V-5B, predictive dynamics), run in parallel and combined at the level of **8 action tokens each**.

**What it contributes that Senna-2, AutoMoT, and HybridDriveVLA do not**: a measured failure mode for the obvious alternative. Fusing the two systems' *raw tokens* in one attention pool (the paper's Tri-MoT baseline) scores 87.8 PDMS, **below the WAM branch alone at 88.1**, with strictly more information. See [Where MoT Breaks](mixture-of-experts.md#modality-competition).

**Branch complementarity is measured, not assumed** (NAVSIM-v1 PDMS):

| Variant | PDMS |
|---|---:|
| VLA-only | 86.1 |
| WAM-only | 88.1 |
| Tri-MoT (raw-token fusion) | 87.8 |
| **BrainWAM** (action-space coordination) | **89.5** |

Note that on NAVSIM the predictive branch beats the semantic branch by 2.0 on its own, so the coordination mechanism's honest attribution is **+1.4 over WAM-only**, not +3.4 over VLA-only.

The qualitative split matches [[sources/hybriddriveVLA.md]]'s findings: VLA wins on navigation-instruction following and traffic-light / brake-light semantics; WAM wins on interactive negotiation and trajectory feasibility on curves. BrainWAM reports this as figure panels rather than the set-level failure-overlap statistics HybridDriveVLA computes — the weaker form of the same claim.

### Freeze-Then-Coordinate, With a Reason

The wiki's clearest quantitative argument for freezing pretrained branches in a two-backbone planner:

| Stage-3 update strategy | PDMS |
|---|---:|
| Full-model fine-tuning | 88.8 |
| **CAB + CIF + action decoder only** | **89.5** |

The supporting datum is the useful part: **the VLA branch reaches 86.1 PDMS after 54K steps while the WAM branch needs 81K steps to reach 88.1.** Unfrozen, the two pathways update at different rates, so the coordination modules chase representations that are still moving. That is a mechanism rather than a heuristic, and it independently supports [[sources/automot.md]]'s frozen-UE choice (motivated by catastrophic forgetting) and [[sources/foresight.md]]'s two-phase schedule (motivated by capacity imbalance). Three papers, three different reasons, one conclusion.

### Comparison Against the Other Two-Backbone Designs

| Method | System A | System B | Interface | Score |
|---|---|---|---|---|
| [[sources/automot.md]] | Frozen Qwen3-VL-4B | 1.6B action expert | Layer-wise KV cache; asymmetric; async | 87.34 DS Bench2Drive |
| [[sources/hybriddriveVLA.md]] | VLM branch | ViT branch | Style-axis interpolation + trajectory scorer | 92.1 PDMS |
| [[sources/drivewam.md]] | Wan2.2-5B policy | Frozen Qwen3-VL-8B advisor | **Natural language**; one-way; no gradient | 90.1 PDMS |
| **BrainWAM** | Qwen3-VL-4B VLA | Wan2.2-5B WAM | **8 action tokens**; bidirectional gated cross-attn | 89.5 PDMS |

BrainWAM's interface is the narrowest that still carries gradients in both directions; DriveWAM's is narrower still (two sentences of text) but one-way and gradient-free. **The trend across all four is toward deliberately constricted interfaces**, and BrainWAM is the first to give an explicit reason why widening them hurts.

## LWDrive: One Interface, Consumed Six Times {#lwdrive}

[[sources/lwdrive.md]] names the pathology this page has been circling and gives it a taxonomy. Its Figure 1 splits prior work into **(a) direct VLM-to-trajectory decoding** and **(b) single-stage fusion**, where "VLM semantics are usually injected only once and remain weakly coupled with subsequent trajectory correction," and proposes (c): the VLM output is an *intent anchor*, and refinement consumes VLM representations repeatedly at increasing depth.

Structurally this is a dual-system design where **System A is Qwen2.5-VL-3B and System B is a six-stage proposal refiner over BEV features**, and the interface is not narrowed — it is *repeated*. Stage $r$ reads hidden states from Qwen layer $6r$ through Bridge Attention, which pools proposal self-memory, action-query and ego-state memory, and VLM foresight memory into one attention. The proposal pool is initialised from the **pooled action-query latent** rather than the decoded trajectory, so the coarse plan is never a bottleneck the way a text or waypoint interface is.

**How this bears on the constricted-interface trend above.** BrainWAM's negative result is specific: mixing a *clean semantic stream* with a *denoising* one in a shared attention pool causes modality competition, and the fix is to compress each branch to 8 action tokens. LWDrive puts a clean semantic stream (VLM hidden states) next to a clean geometric one (BEV features) — no denoising stream anywhere at inference — and a wide repeated interface works fine. **The two results are compatible and jointly sharpen the rule**: what needs constricting is the coupling to a *generative* branch, not VLM coupling as such.

**What the ablation supports and what it does not.** Removing Bridge Attention costs 2.0 PDMS and removing BEV grounding costs 2.7, so the repeated interface is doing real work. But **reading six depths instead of six times from the final layer is worth only +0.2** — so the value is in *repetition and BEV grounding*, not in depth diversity. The "inject once" complaint against category (b) is answered by the evidence; the "inject at many depths" prescription is not.

| Method | System A | System B | Interface | Score |
|---|---|---|---|---|
| [[sources/senna2.md]] | VLM decision maker | E2E policy | 20 discrete meta-actions | 86.6 EPDMS |
| [[sources/drivewam.md]] | Frozen Qwen3-VL-8B advisor | Wan2.2-5B policy | Natural language, one-way | 90.1 PDMS |
| [[sources/brainwam.md]] | Qwen3-VL-4B VLA | Wan2.2-5B WAM | 8 action tokens, bidirectional | 89.5 PDMS |
| **[[sources/lwdrive.md]]** | **Qwen2.5-VL-3B (frozen in stage 2)** | **6-stage BEV proposal refiner** | **Full hidden states, read six times at six depths** | **92.0 PDMS** |
| **[[sources/drive-hwm.md]]** | **V-JEPA flow-latent predictor (no decision)** | **Emu3-8B + AR action expert** | **One Dynamic-Aware Latent, FiLM scale-and-shift** | **93.8 / 93.3 PDMS** |

## Drive-HWM: Fast–Slow by Temporal Role {#temporal-role}

Every design above splits *what kind of work* each system does. [[sources/drive-hwm.md]] splits **how often each system runs**, and it is the first entry here whose slow system emits no decision of any kind.

The paper draws the boundary itself, and the distinction is real:

> "Several driving VLAs further adopt fast–slow designs to balance decision quality and computational cost: routine scenarios use direct action generation, whereas challenging situations invoke more expensive semantic or chain-of-thought reasoning. These methods separate reasoning modes and allocate computation according to scenario complexity. In contrast, our hierarchy separates explicit future representation prediction from action generation according to their temporal roles."

Three axes now exist on this page, and they are independent:

| Axis | What triggers the slow system | Examples |
|---|---|---|
| **Scene difficulty** | a routing decision per scene | [[sources/autovla.md]], [[sources/adathinkdrive.md]], [[sources/deepsight.md]], [[sources/clear.md]] |
| **Module cost** | always, but cached or downsampled | DualDriveVLA ([[sources/hybriddriveVLA.md]], 15% of frames), [[sources/drivewam.md]] (frozen advisor per chunk), [[sources/automot.md]] (layer-wise KV cache) |
| **Temporal role** | **a fixed period — every $N$ steps, unconditionally** | **Drive-HWM** ($N=8$) |

**What is genuinely new is the slow system's output type.** Senna-2 emits meta-actions, DriveWAM emits natural-language guidance, LWDrive emits an intent anchor, BrainWAM emits 8 action tokens — all of them *decisions*, at some level of abstraction. Drive-HWM's slow branch emits a **prediction about the world** and explicitly nothing else: §III-B states "the slow branch therefore does not output actions or an explicit trajectory." The fast model is not executing or refining a plan; it is reading a forecast. That removes the consistency problem this page is largely organized around — there is no decision to be inconsistent with — and replaces it with a different question, which is whether the forecast is *used*. Drive-HWM's answer is the next-frame auxiliary loss, conditioned on the Dynamic-Aware Latent specifically so that "the fast model [makes] effective use of the dynamic context instead of ignoring it during action training."

**On the interface, it lands opposite BrainWAM and reaches the same place.** [[sources/brainwam.md]] found raw-token sharing in one attention pool actively harmful (Tri-MoT 87.8 below its own WAM-only 88.1) and fixed it with a narrow 8-token gated cross-attention bottleneck. Drive-HWM never puts the slow output in the token stream at all — FiLM modulates hidden-state channels, leaving "the token organization of the pretrained backbone" untouched — and its own sweep ranks the options:

| Conditioning | PDMS |
|---|---:|
| Concatenation | 92.5 |
| Cross-attention | 93.0 |
| Gated cross-attention | 93.1 |
| AdaLN | 93.3 |
| **FiLM** | **93.8** |

**Both papers put concatenation last, and both conclude the backbone's sequence layout should not be disturbed.** That is now two independent architectures agreeing, by different mechanisms, that a world-model branch should reach a pretrained multimodal policy through a *narrow, non-token* channel. FiLM is the cheaper of the two.

**The caveat is large and belongs here rather than on the world-model page.** The temporal-role split is a claim about *deployment*, and NAVSIM cannot test it: the benchmark is single-shot, so with $N=8$ at its 8-pose convention the slow model fires exactly once per scenario and the fast model never receives a second observation. See [[concepts/navsim-benchmark.md]]. **What Table IV measures is a conditioning stream, not a schedule** — and it is worth +0.3 or +0.8 PDMS depending on which of the paper's [two result sets](../sources/drive-hwm.md#two-result-sets) is correct, against 1.3 for the conditioning mechanism alone. The rate hierarchy is the least-supported part of a paper whose other measurements are clean.

## Open Questions

- Can the kinematic mapping $f_K$ be learned (soft/continuous) rather than rule-based, to avoid category-boundary noise?
- Does DriveWAM's advisory arrangement need a consistency mechanism? Senna-2's evidence suggests unaligned VLM guidance under-delivers; DriveWAM reports aggregate gains but no VLM-action agreement rate.
- Does dual-system consistency generalize to more fine-grained decision spaces (beyond 20 meta-actions)?
- Can the VLM be distilled into the E2E policy after alignment training, eliminating the runtime VLM latency?
- How does dual-system alignment interact with GRPO-style reward shaping (used in WAM-Flow/ReCogDrive)?
- Does MoT's anti-interference benefit hold at larger scales (>8B) where shared-weight models also benefit from more parameters?
- Can UniDriveVLA's perception + action MoT be combined with Senna-2's consistency alignment for further gains?
- **Does a fixed-period slow branch beat a difficulty-routed one?** Drive-HWM runs its slow model every $N=8$ steps unconditionally; CLEAR, AdaThinkDrive and AutoVLA all spend slow compute only where a router says it is needed. Nobody has compared the two policies at matched average cost, and the comparison is well-posed: Drive-HWM's slow branch is 25.6 ms, so a router that fired it on a third of scenes would free budget for a larger one. See [Fast–Slow by Temporal Role](#temporal-role).
- **Is a decision-free slow system enough?** Drive-HWM is the first design here whose slow branch emits a forecast rather than a plan, which eliminates the VLM-action consistency problem this page is built around. Whether that is a simplification or a loss is untested — no paper compares a forecast-only slow branch against a meta-action or intent-anchor one under a fixed fast policy.

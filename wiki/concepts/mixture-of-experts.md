---
title: Mixture of Experts for Autonomous Driving VLAs
type: concept
sources: [raw/papers/BrainWAM_ Action-Space Coordination of Semantic Priors and Predictive Dynamics for Autonomous Driving.md, raw/papers/WAM-Diff_ A Masked Diffusion VLA Framework with MoE and Online Reinforcement Learning for Autonomous Driving.md, raw/papers/DriveFine_ Refining-Augmented Masked Diffusion VLA for Precise and Robust Driving.md, raw/papers/DriveVLA-W0_ World Models Amplify Data Scaling Law in Autonomous Driving.md, raw/papers/AutoMoT_ A Unified Vision-Language-Action Model with Asynchronous Mixture-of-Transformers for End-to-End Autonomous Driving.md, raw/papers/UniDriveVLA_ Unifying Understanding, Perception, and Action Planning for Autonomous Driving.md]
related: [sources/brainwam.md, sources/wam-diff.md, sources/drivefine.md, sources/drivevla-w0.md, sources/automot.md, sources/unidrivevla.md, concepts/rl-for-ad.md, concepts/diffusion-planner.md, concepts/vlm-domain-adaptation.md, concepts/perception-for-planning.md]
created: 2026-04-21
updated: 2026-09-04
confidence: high
---

## What It Is

Mixture of Experts (MoE) is an architectural pattern that replaces a monolithic component (layer, module, or full model) with multiple specialized sub-networks ("experts"), activating only a subset per input. In the VLA-for-AD literature it appears in four distinct forms, each solving a different problem.

## Taxonomy

### 1. Sparse Token-Level MoE (FFN replacement)

The standard LLM-era MoE: each transformer FFN is replaced by N expert FFNs; a router dispatches each token to K of N experts. Only activated experts compute; the rest are dormant. Capacity is measured per expert (fraction of tokens it sees).

**Wiki example**: WAM-Diff ([[sources/wam-diff.md]])
- LoRA experts inside LLaDA-V backbone FFNs
- 64 experts, rank 32, expert-choice routing, capacity = 0.1 per expert
- Frozen base FFN $W_0$ acts as a **shared expert** (always active)
- For input $z$: $\text{MoE}(z) = W_0 z + \sum_{i=1}^{N} g_i(z) \cdot B_i A_i z$ where $g_i(z) = \text{Softmax}(W_g z)$
- Adds +0.5B params but only ~0.05B activated per token (8.4B base → effectively 8.45B at inference)
- Ablation: 16 experts → 85.0 PDMS; 64 experts → 86.6 PDMS; no MoE → 84.7 PDMS (+1.9 gain)
- BEV expert activation heatmaps confirm different experts activate for different driving scenarios (lane change, intersection, parking)

### 2. Block-Level MoE (module specialization)

Entire transformer blocks are replicated and assigned different roles; routing is explicit (task token, not learned softmax).

**Wiki example**: DriveFine ([[sources/drivefine.md]])
- Last 4 transformer blocks (blocks 28–31 of LLaDA-8B) duplicated into two sets:
  - **Generation expert** — receives masked tokens `[M]`; performs standard masked diffusion
  - **Refinement expert** — receives fully committed tokens; performs iterative error correction
- Routing is not learned: inference-time task token tells the model which block set to activate
- Plug-and-play: refinement expert added without retraining the generation backbone
- Both experts trained **synchronously** with different objectives (GRPO for generation; hybrid offline+online RL for refinement)
- The separation enables refinement expert to specialize on error correction trajectories that the generation expert passes through, including explicit on-policy exploration

### 3. Mixture-of-Transformers (MoT) — Model-Level Specialization

Two or more complete (or near-complete) transformer models operate in parallel and share information via cross-attention or KV cache injection. Each model specializes on a different modality or cognitive role.

**Wiki example: AutoMoT** ([[sources/automot.md]])
- **Understanding Expert (UE)**: Qwen3-VL-4B, **fully frozen**, runs on multi-view multi-frame RGB + text prompts → semantic reasoning / CoT
- **Action Expert (AE)**: ~1.6B trained from scratch, runs on current RGB + LiDAR BEV + action queries → waypoints, meta-actions, route
- **KV-cache coupling**: UE produces layer-wise KV at update time $\tau(t)$; AE's attention at time $t$ concatenates its own KV with cached UE KV
  - Within-task: bidirectional attention; cross-task: causal (AE conditions on UE, not vice versa)
- **Asynchronous inference**: UE runs at low frequency; AE reuses stale UE KV at every high-frequency action step → 7.6× latency reduction (<1.24% L2 regression)
- **Critical finding**: AD fine-tuning the UE causes **catastrophic forgetting** (−14.8% TallyQA, −13.5% InfoVQA); freezing UE preserves general reasoning while matching or exceeding fine-tuned baselines on AD-specific benchmarks

**Wiki example: UniDriveVLA** ([[sources/unidrivevla.md]])
- **Three expert streams**: understanding (reasoning), perception (sparse detection), action (trajectory)
- Each expert has own projection, FFN, normalization
- **Masked joint attention**: information flows asymmetrically — understanding attends to everything, action can attend to understanding, perception interleaved via sparse queries
- Enables a single unified training objective combining 5 perception tasks + trajectory prediction

**Wiki example: BrainWAM** ([[sources/brainwam.md]]) — *the negative result that bounds this pattern*
- Uses **Dual-MoT twice**: once inside the WAM branch (Wan2.2-5B video expert + action expert) and once inside the VLA branch (Qwen3-VL-4B + action expert). Both are shared self-attention with modality-specific FFNs, and both work.
- Its **Tri-MoT** baseline extends this to three streams in one pool — VLM tokens, video-generator tokens, action tokens — and **scores 87.8 PDMS, below the WAM-only branch at 88.1**, with strictly more information and comparable parameters.
- Diagnosis: **modality competition**. Action tokens attend more to VLM tokens than VGM tokens across most layers, because VLM tokens are clean large-scale-pretrained abstractions while VGM tokens are mid-denoising and low-signal. The easier modality wins the shared attention budget and suppresses the predictive one.
- Fix: **do not share an attention pool at all**. Each branch compresses to 8 action tokens at 1024 dim; the branches exchange information only through those, via gated cross-attention (CAB, 16.8M) and a 2-layer Transformer fusion (CIF, 49.3M). 89.5 PDMS.

### Where MoT Breaks: The Modality-Competition Boundary {#modality-competition}

BrainWAM's Tri-MoT is the first MoT design in this wiki that **loses to its own single-branch ablation**. Comparing it against the MoT designs that succeed gives a reasonably clean boundary condition.

| Design | Streams in the shared pool | Information flow | Result |
|---|---|---|---|
| [[sources/simwam.md]] mask ablation | video + action (**no VLM**) | bidirectional | 90.2 vs. 90.3 isolated — no effect |
| [[sources/driveva.md]] | video + action (**no VLM**) | joint denoising | 90.9 — works |
| [[sources/drivewam.md]] | video + action; VLM advises **in text, outside attention** | one-way, gradient-free | 90.1 — works |
| [[sources/automot.md]] | frozen UE + trained AE | **asymmetric** (AE reads UE's KV, not the reverse) | works; enables 7.6× async speedup |
| [[sources/unidrivevla.md]] | understanding + perception + action | **masked** (understanding attends to all; action attends to understanding) | works |
| **BrainWAM Tri-MoT** | **VLM + VGM + action** | **symmetric, unmasked** | **87.8 < 88.1 WAM-only** |

**Stream count is not the culprit** — UniDriveVLA runs three streams successfully. What separates the failures from the successes is the combination of (a) a *clean pretrained semantic stream* competing with (b) an *iteratively-denoised stream*, under (c) *symmetric, unmasked* attention. Every successful design in this table breaks at least one of those conditions, usually (c).

**The untested middle ground**: nobody has tried Tri-MoT with UniDriveVLA-style masking, which would preserve raw-token access while removing the symmetric competition. BrainWAM jumps straight to an 8-token bottleneck — and its own ablation shows cross-stream communication saturates after two CAB blocks (28 blocks score the same 89.3 as 2), so whatever the branches need to exchange is low-dimensional. The bottleneck may be doing less of the work than the separation is.

### 4. Lightweight Action Expert alongside VLA Backbone

A dedicated small model handles trajectory output while the large VLA backbone handles scene understanding. Unlike MoT, these are not symmetric experts — one is dominant, the other is an addendum.

**Wiki example: DriveVLA-W0** ([[sources/drivevla-w0.md]])
- 500M action expert alongside a large VLA backbone; joint attention over concatenated QKV
- Three expert variants tested: query-based (L1 regression), autoregressive (FAST tokens), flow matching (ODE)
- Previous action features $A_{t-1}$ always prefilled as temporal prior
- Latency reduction to 63.1% of standalone VLA baseline (74ms vs. 118ms)
- The "MoE" naming here is informal — there is no routing; it is a dedicated side network

## Comparison Across Wiki Papers

| Paper | MoE Type | N Experts | Routing | Params Added | PDMS Gain | Primary Purpose |
|-------|----------|-----------|---------|--------------|-----------|-----------------|
| WAM-Diff | Sparse LoRA (FFN) | 64 | Expert-choice softmax | +0.5B (0.05B active) | +1.9 PDMS | Capacity scaling |
| DriveFine | Block-level task MoE | 2 (Gen / Ref) | Explicit task token | ~2× last-4-blocks | Qualitative only | Error correction specialization |
| AutoMoT | MoT (frozen UE + trained AE) | 2 models | Implicit (role-based) | 1.6B AE | 87.34 DS Bench2Drive | Forgetting prevention + async latency |
| UniDriveVLA | MoT (3 streams) | 3 streams | Masked joint attention | Per-stream FFN/proj | 78.37 DS Bench2Drive | Unified multi-task learning |
| DriveVLA-W0 | Side expert | 3 variants | None (dedicated) | 500M | 63% latency | Modular action decoding |
| BrainWAM | Dual-MoT x2 + action-space bridge | 2 branches (2 streams each) | None; 8-token CAB/CIF interface | 16.8M CAB + 49.3M CIF | 89.5 PDMS (+1.4 over WAM-only) | Avoiding modality competition |

## MoE and RL: The Routing Stability Problem

Sparse token-level MoE creates a specific challenge for token-level RL (GRPO): updating the policy on token $t$ changes the routing decision for token $t$, causing different experts to activate on the next forward pass. This means the gradient signal for token $t$'s policy update is computed under routing conditions that will not reproduce — a form of **credit assignment instability** specific to MoE.

**WAM-Diff's response**: GSPO (Group Sequence Policy Optimization) applies the policy update at the **sequence level** — a single advantage estimate for the entire trajectory, with the PPO clip applied in sequence-likelihood space. This sidesteps per-token routing changes by treating the trajectory as an atomic unit.

See [[concepts/rl-for-ad.md]] for the GSPO formulation.

## MoE and Catastrophic Forgetting

AutoMoT ([[sources/automot.md]]) provides the strongest evidence in the wiki that fine-tuning a VLM backbone on AD data is **actively harmful** to general reasoning:

| Benchmark            | AutoMoT (frozen UE) | AutoMoT (AD fine-tuned UE) | Δ      |
| -------------------- | ------------------- | -------------------------- | ------ |
| TallyQA (counting)   | 81.40               | 66.90                      | −14.8% |
| InfoVQA (doc QA)     | 89.30               | 75.80                      | −13.5% |
| LingoQA (driving QA) | 67.00               | 67.20                      | ≈0     |
| OmniDrive (driving)  | 0.89                | 0.82                       | +8.5%  |

Frozen UE matches fine-tuned on driving-specific benchmarks while preserving general reasoning. The MoT pattern (frozen UE + trained AE from scratch) is the proposed solution.

## Why Multiple Distinct MoE Patterns?

The four patterns address different bottlenecks:

| Bottleneck | Solution | Paper |
|------------|----------|-------|
| Model capacity limited by single FFN | Sparse token-level MoE | WAM-Diff |
| Error correction needs different objective than generation | Block-level task routing | DriveFine |
| VLM forgetting general reasoning when fine-tuned | Freeze UE, train AE from scratch (MoT) | AutoMoT |
| Unifying multi-task learning across modalities | MoT with masked joint attention | UniDriveVLA |
| Latency from large VLA backbone at inference | Lightweight side expert with async update | DriveVLA-W0 |

## Open Questions

- **Expert count scaling**: WAM-Diff shows 64 > 16 > no-MoE (+1.9 total PDMS), but the curve is not shown beyond 64. At what point does expert count saturate for trajectory planning?
- **Block-level vs. token-level**: DriveFine's explicit task routing vs. WAM-Diff's learned routing — is explicit routing more sample-efficient, or does learned routing generalize better to novel scenarios?
- **MoT forgetting generalization**: AutoMoT shows forgetting on one VLM (Qwen3-VL-4B). Does the same pattern hold for larger VLMs? Does any fine-tuning approach preserve both general reasoning and AD planning?
- **Unified MoE+RL**: DriveFine applies different RL objectives per expert (GRPO for gen, hybrid RL for refinement). Could a single sequence-level objective like GSPO work across both block types?

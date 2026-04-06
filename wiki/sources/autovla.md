---
title: "AutoVLA: VLA with Adaptive Reasoning and Reinforcement Fine-Tuning"
type: source-summary
sources: [raw/papers/AutoVLA_ A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning.md]
related: [concepts/rl-for-ad.md, concepts/navsim-benchmark.md, concepts/diffusion-planner.md, concepts/vlm-domain-adaptation.md, sources/orion.md, sources/linkvla.md, sources/curious-vla.md]
created: 2026-04-05
updated: 2026-04-05
confidence: high
---

**arXiv**: https://arxiv.org/html/2506.13757v1  
**Affiliation**: UCLA  
**Project**: https://autovla.github.io/

## One-Line Summary

AutoVLA unifies reasoning and trajectory generation in a single AR VLM (Qwen2.5-VL-3B) using a K-Disk-clustered physical action codebook (K=2048), and introduces a CoT length penalty in GRPO RFT to train adaptive fast/slow reasoning — achieving +10.6% PDMS and −66.8% runtime over SFT.

## Core Problem Statement

Two limitations of existing VLA driving models:
1. **Physically-infeasible or complex action generation** — text waypoints struggle with numerical precision; intermediate meta-actions or latent decoders break end-to-end optimization or add complexity
2. **Inflexible reasoning** — all scenarios use fixed CoT strategy; DriveVLM solves this with separate VLM+E2E modules (expensive); no method adaptively switches within a single model

## Architecture

![[x1 12.png|AutoVLA overview: VLM + physical action tokens + adaptive CoT]]

*Figure 1: AutoVLA takes multi-view images (3xC: front, front-left, front-right), 4-frame temporal history, ego states, and language instructions. Outputs reasoning text + physical action token sequence.*

![[x3 10.png|Training pipeline: SFT with CoT distillation + RFT with GRPO]]

*Figure 3: SFT uses Qwen2.5-VL-72B-distilled CoT reasoning data alongside trajectory-only data. RFT applies GRPO with PDMS reward minus CoT length penalty.*

**Base model**: Qwen2.5-VL-3B  
**Inputs**: 3x cameras (front, front-left, front-right), 4 frames each (2 Hz = current + 3 prior), ego speed/acceleration/history actions, navigation instruction  
**Output format**: [reasoning tokens] + [action_i, action_j, …, action_k] (10 tokens = 5s horizon)

## Physical Action Tokenization

![[x7 8.png|Action codebook visualization: WOMD real-world (a) and CARLA simulation (b)]]

*Figure S1: 2048 action tokens each encoding (Δx, Δy, Δθ) for 0.5s. Grey = all tokens, blue = 300 random samples. Real-world codebook from WOMD; separate simulation codebook from CARLA-Garage.*

**K-Disk clustering** from Waymo Open Motion Dataset (WOMD):
1. Sample 0.5-second motion segments from all vehicle trajectories
2. Iteratively select diverse segments: no two within δ=0.05m average contour distance
3. Extract (Δx, Δy, Δθ) for each selected segment
4. Result: **K=2048 action tokens** covering the majority of vehicle movement patterns

Each action token encodes a **physically feasible** short-term vehicle maneuver. Tokens added to VLM vocab as special tokens `<action_0>`, …, `<action_2047>`. During inference, 10 autoregressively generated tokens are decoded by sequentially composing displacements from the current ego pose.

**Separate codebook for CARLA** (same procedure, different source data) due to vehicle dynamics differences.

**Ablation vs. text waypoints** (one-shot nuPlan):

| Method | PDMS ↑ | Avg. L2 (m) ↓ | Avg. Col. (%) ↓ | Runtime (s) ↓ |
|--------|--------|----------------|-----------------|---------------|
| Text Waypoint | 59.24 | 1.29 | 0.98 | 9.31 |
| **Physical Action** | **80.54** | **0.86** | **0.35** | **3.95** |

**Interpretation**: LLMs struggle with precise numerical reasoning (text waypoints require decoding floats); physical tokens remove this bottleneck. Physical tokens also run faster (no numerical decoding overhead).

## Training Stage 1: Supervised Fine-tuning (SFT)

**Dual-mode training** — both fast and slow thinking in same model:
- **Fast thinking**: fixed short template "reasoning not needed" + action tokens
- **Slow thinking**: 4-stage CoT template + reasoning sequence + action tokens

**Loss function**:
$$\mathcal{L}_{i}^{\text{SFT}}=w_{i}\cdot\left(\mathcal{L}_{\text{LM},i}+\lambda_{a}\mathcal{L}_{\text{action},i}\right), \quad w_{i}=\begin{cases}\lambda_{\text{cot}}&\text{if CoT present}\\ 1&\text{otherwise}\end{cases}$$

- $\lambda_a = 1$ (action loss weight)
- $\lambda_{cot} = 40$ (upweight CoT samples to compensate for smaller count)
- $\mathcal{L}_{LM}$: standard causal LM loss over all output tokens
- $\mathcal{L}_{action}$: auxiliary loss over action token positions only (positions $L+1$ to $L+T$)

**Training**: 5 epochs, 8x NVIDIA L40S, lr = 1e-5, FSDP, BFloat16, grad checkpointing, effective batch = 32.

## Reasoning Data Collection

![[x8 5.png|Automated reasoning annotation pipeline (Waymo E2E example)]]

*Figure S2: Qwen2.5-VL-72B annotates 4-stage CoT with GT meta-action as hint, reducing nonsensical outputs.*

**Pipeline**: Qwen2.5-VL-72B annotates 4 components per scenario:
1. Scene description and analysis
2. Critical object identification
3. Surrounding agents' intention prediction
4. Decision-making + meta-action

**GT meta-action as hint** included in the prompt → guides causal explanation directly linked to scene context → reduces revision need.

**Quality check**: 88.8% accuracy on 3,000 randomly sampled annotations (human binary scoring). Erroneous samples corrected or discarded.

**Corpus size**:
| Source | Count |
|--------|-------|
| nuPlan CoT annotations | 45.6k |
| Waymo E2E CoT annotations | 7.2k |
| DriveLM (nuScenes + CARLA, reformatted) | varied |

**Data scaling finding** (Fig. 4):  
- CoT underperforms action-only at < 50k training samples (reasoning is hard to learn from limited data)
- CoT surpasses action-only at ≥ 50k — reasoning provides scalability advantage at larger data regimes
- nuScenes (mostly simple scenarios): action-only outperforms CoT throughout — domain matters

![[x4 9.png|Data scaling effect (log-scale x-axis): both datasets show consistent improvement]]

*Figure 4: Increasing training data consistently improves PDMS (nuPlan) and L2/collision (nuScenes). CoT advantage only emerges at large scale.*

## Training Stage 2: Reinforcement Fine-tuning (RFT)

**Key insight**: reward = driving quality minus CoT length penalty — trains model to use fast thinking in simple scenarios, slow thinking only when needed.

**Reward function**:
$$r = r_{\text{Driving}} - \lambda_r \cdot r_{\text{CoT}}$$

- $r_{\text{Driving}}$ = PDMS (NAVSIM/nuPlan) or ADE (Waymo E2E)
- $r_{\text{CoT}}$ = CoT length penalty (penalizes unnecessarily long reasoning chains)

**GRPO objective** (standard group-relative advantage):
$$A_i = \frac{r_i - \text{mean}(\{r_j\})}{\text{std}(\{r_j\})}$$

KL divergence term regularizes toward SFT reference policy: $\mathbb{D}_{KL}(\pi_\theta \| \pi_{ref}) = \frac{\pi_{ref}}{\pi_\theta} - \log\frac{\pi_{ref}}{\pi_\theta} - 1$

**RFT configuration**: LoRA adapter, lr = 3e-5, β = 0.04, 6,000 steps, single policy update per step (simplified objective, no clipping). G (group size) ablation: larger G → better performance via broader exploration.

**Adaptive reasoning result** (Fig. 5):
- Simple straight-road scenarios → model outputs `<fast>` with minimal text
- Complex scenarios (intersections, construction zones) → model outputs full 4-stage CoT
- +10.6% PDMS improvement over SFT post-RFT
- −66.8% average runtime reduction over 500 test scenarios

![[x5 9.png|RFT results: (a) PDMS + runtime before/after, (b) training reward curves, (c) qualitative comparison]]

*Figure 5: (a) Both PDMS improves and runtime drops post-RFT. (b) Larger group size G → better convergence. (c) SFT model produces suboptimal plans (error accumulation); RFT model produces better trajectories.*

## Results

### NAVSIM-v1 / nuPlan (Table 1)

| Method                 | PDMS ↑    |
| ---------------------- | --------- |
| Ego Status MLP         | 66.40     |
| TransFuser             | 83.88     |
| DRAMA                  | 86.87     |
| Hydra-MDP              | 91.26     |
| Centaur                | 92.10     |
| TrajHF                 | 93.95     |
| AutoVLA (one-shot SFT) | 80.54     |
| **AutoVLA (post-RFT)** | **89.11** |
| AutoVLA (Best-of-N)    | 92.12     |

**Note**: Table 1 compares against older non-VLM methods (Hydra-MDP, TrajHF) and doesn't include DriveFine (90.7), WAM-Flow (90.3), or Curious-VLA (90.3). AutoVLA post-RFT 89.11 matches the 89.1 entry in Curious-VLA's Table 1 (cross-verified).

### Bench2Drive CARLA Closed-loop (Table 2)

| Method | Driving Score ↑ | Success Rate (%) ↑ | Efficiency ↑ | Comfort ↑ |
|--------|-----------------|---------------------|-------------|-----------|
| UniAD-Base | 45.81 | 16.36 | 129.21 | 43.58 |
| VAD | 42.35 | 15.00 | 157.94 | 46.01 |
| DriveAdapter | 64.22 | 33.08 | 70.22 | 16.01 |
| ORION | 77.74 | 54.62 | 151.48 | 17.38 |
| **AutoVLA** | **78.84** | **57.73** | **146.93** | **39.33** |

Marginal improvement over ORION (+1.1 DS, +3.1% SR). Both are superseded by LinkVLA (91.01 DS / 74.55% SR). Notably AutoVLA achieves higher Comfort (39.33 vs. ORION's 17.38) — physical action codebook produces smoother trajectories.

### Waymo E2E

No numeric table in paper — results reported qualitatively via bar charts (Fig. 6). Pretraining on nuPlan + nuScenes improves Waymo RFS; adding CoT further helps; RFT (ADE reward) achieves best RFS. Construction zone qualitative shows accurate occlusion reasoning + detour planning.

![[x6 8.png|Waymo E2E results under different training settings + construction zone reasoning example]]

*Figure 6: Progressive improvement: pretraining → +CoT → +RFT (ADE). Construction zone shows model reasoning about occlusion and planning detour.*

## Implementation Details

| Setting | Value |
|---------|-------|
| Base model | Qwen2.5-VL-3B |
| Cameras | Front, front-left, front-right (3x) |
| Temporal frames | 4 per camera @ 2 Hz |
| Action horizon | 5s (10 tokens × 0.5s) |
| Action codebook size | 2048 (K-Disk, WOMD) |
| SFT hardware | 8x NVIDIA L40S |
| SFT lr | 1e-5 |
| SFT epochs | 5 |
| SFT effective batch | 32 |
| RFT adapter | LoRA |
| RFT lr | 3e-5 |
| RFT KL weight β | 0.04 |
| RFT steps | 6,000 |

## Reasoning Examples

![[x9 1.png|Waymo E2E reasoning annotation examples (stop sign + construction zone)]]
*Figure S3: Pipeline correctly handles "stopped at stop sign can now proceed" vs. indefinite stopping; correctly interprets construction road control.*

![[x10 1.png|nuPlan reasoning annotation examples (stop signs vs. traffic lights in different lanes)]]
*Figure S4: Pipeline distinguishes functional relevance of stop signs and traffic lights across lanes (doesn't stop at every sign).*

![[x11 2.png|System prompt and user message format]]
*Figure S5: Structured system prompt defines role + CoT format; user message includes multi-view images, ego state, and driving instruction.*

## Limitations

1. **NAVSIM comparison gap** — Table 1 lacks head-to-head vs. DriveFine (90.7), WAM-Flow (90.3), Curious-VLA (90.3); AutoVLA 89.11 is likely below current SOTA
2. **Bench2Drive superseded** — 78.84 DS is modest vs. LinkVLA (91.01); ORION comparison is the most relevant in the table
3. **No NAVSIM-v2 (EPDMS) results** — cannot compare on extended metrics
4. **~1 Hz inference** — near-real-time but not real-time; heavy GPU dependency
5. **Separate codebooks per domain** — real-world and simulation codebooks diverge, complicating domain-agnostic deployment
6. **Waymo E2E metrics unavailable** — no numeric comparison table; only relative bar charts
7. **CoT length penalty is heuristic** — λ_r balance between driving quality and reasoning length requires tuning per dataset

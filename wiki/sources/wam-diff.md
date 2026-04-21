---
title: "WAM-Diff: A Masked Diffusion VLA Framework with MoE and Online Reinforcement Learning for Autonomous Driving"
type: source-summary
sources: [raw/papers/WAM-Diff_ A Masked Diffusion VLA Framework with MoE and Online Reinforcement Learning for Autonomous Driving.md]
related: [concepts/diffusion-planner.md, concepts/rl-for-ad.md, concepts/navsim-benchmark.md, concepts/discrete-flow-matching.md, sources/drivefine.md, sources/reflectdrive.md, sources/wam-flow.md, sources/recogdrive.md]
created: 2026-04-21
updated: 2026-04-21
confidence: high
---

**arXiv**: 2512.11872v1  
**Org**: Fudan University + Yinwang Intelligent Technology Co., Ltd  
**Code**: https://github.com/fudan-generative-vision/WAM-Diff

## TL;DR

WAM-Diff is the first VLA for autonomous driving to combine discrete **masked diffusion** with a sparse **LoRA MoE** architecture and online **GSPO** reinforcement learning. It builds on LLaDA-V (8.4B params) and achieves 91.0 PDMS on NAVSIM-v1 and 89.7 EPDMS on NAVSIM-v2. The key design insight is that masked diffusion's inherent flexibility in decoding order — supporting causal, reverse-causal, and random schedules — is particularly well-suited to scenario-aware trajectory generation.

## Architecture

![[main_arch.png]]
*Figure 2: WAM-Diff overview. Multimodal inputs encoded by SigLIP-2 + MoE backbone; masked diffusion decoder iteratively generates future trajectory tokens from fully masked state.*

Four core components:

**Image encoder**: SigLIP-2. Input 1920×1080 image → 15 non-overlapping 384×384 patches + 1 full-image resize = 16 patches → 2,185 visual tokens → MLP → 4,096-dim.

**Text encoder**: Base vocabulary of 126,349 tokens extended by 20,001 numerical waypoint tokens = 146,350 total.

**Backbone**: LLaDA-V with sparse LoRA MoE integrated into FFNs. Total 8.4B params; MoE adds +0.5B; ~0.05B activated at inference.

**Decoder**: Masked diffusion over hybrid token sequences (numerical waypoints interleaved with semantic tokens).

## Hybrid Discrete Action Tokenization

Continuous waypoint coordinates are uniformly quantized over $[-100, 100]$ at resolution $0.01$ → 20,001 distinct numerical tokens. Each 2D waypoint is an ordered pair $\langle x, y \rangle$ of scalar tokens; max absolute error per coordinate = 0.005. These 20,001 tokens are merged into the LLM vocabulary alongside standard text tokens, enabling seamless interleaving of metric waypoints and semantic language tokens (e.g., `lane-keep`, `turn-left`, `yield`).

**Advantage over text waypoints**: AutoVLA showed text waypoint representations yield 59.24 PDMS vs. 80.54 for physical codebook tokens — WAM-Diff's quantized numerical tokens operate in a similar hybrid regime, combining language grounding with numerical precision.

## Masked Diffusion for Trajectory Generation

**Forward corruption** (training): each token independently masked with probability $r \sim \mathcal{U}(0,1)$:
$$q_r(x_r|x_0) = \prod_{i=1}^{L}\Big[(1-r)\mathbf{1}\{x_r^i=x_0^i\}+r\mathbf{1}\{x_r^i=M\}\Big]$$

**Training loss** (masked cross-entropy, normalized by expected masked count):
$$\mathcal{L}(\theta)=-\mathbb{E}_{x_0,r,x_r}\Bigg[\frac{1}{r}\sum_{i=1}^{L}\mathbf{1}\{x_r^i=M\}\log p_\theta\big(x_0^i|x_r,c_t\big)\Bigg]$$

**Inference**: start from fully masked sequence $x_{r_0} = M^L$; iteratively infill all masked tokens → remask lowest-confidence subset → repeat for T=32 steps with decreasing mask rate. Fixed output length: 32 tokens. Fast-dLLM decoding for acceleration.

**Two advantages for VLA**:
1. Parallel global prediction at each step (not left-to-right token-by-token)
2. Bidirectional context — each prediction conditioned on all currently visible tokens regardless of position

## Flexible Decoding Schedules

![[scheduler.png]]
*Figure 3: Top row — remasking policies controlling token update order (random, causal, reverse-causal). Bottom row — mask-rate schedules regulating decoding efficiency.*

![[teaser.png]]
*Figure 1: WAM-Diff supports flexible decoding orders. (a) causal, (b) reverse-causal, (c) random — each suited to different driving scenarios. (d) MoE+GSPO yields superior performance in challenging situations.*

The remasking policy controls **which** masked tokens get resolved first:

| Schedule | Update Order | Best Suited For | PDMS |
|---|---|---|---|
| **Random** | Confidence-based (re-mask lowest-confidence predictions) | Balanced, general scenarios | 90.0 |
| **Causal** | Near-term waypoints first (temporal order) | Turning maneuvers requiring kinematic coherence | 88.9 |
| **Reverse-Causal** | Far-future waypoints first | Car-following, oncoming traffic, long-range intent | **91.0** |

Reverse-causal is the default. Intuition: in car-following or oncoming scenarios, the long-horizon intent (keep following distance, yield) should be resolved first, with near-term actions refined subsequently. Ablation gains vs. baseline (no decoding schedule): +2.0 PDMS.

## Scaling with LoRA MoE

For input $z$, the LoRA MoE layer with $N$ experts:
$$o = W_0 z + \sum_{i=1}^{N} g_i(z) E_i(z)$$
where $E_i(z) = B_i A_i z$ (rank-$r$ low-rank experts; $A_i \in \mathbb{R}^{r \times d_{in}}$, $B_i \in \mathbb{R}^{d_{out} \times r}$) and $g_i(z) = \text{Softmax}(W_g z)$.

**Configuration**: 64 experts, rank 32, input/output dim 4096, dropout 0.05. Expert-choice routing with capacity 0.1. The original frozen FFN acts as a shared expert ($W_0$).

**Multi-task training**: LoRA MoE jointly trained on 668K nuPlan trajectory samples + 800K driving-oriented VQA samples. This lifts the model from pure motion imitation to semantic scene understanding and traffic-rule reasoning.

**Ablation: MoE configurations**

| Config | PDMS |
|---|---|
| w/o MoE | 84.7 |
| 16 experts | 85.0 |
| **64 experts** | **86.6** |
| Rank 8 | 85.5 |
| **Rank 32** | **86.6** |

MoE gain over no-MoE: +1.9 PDMS. Figure 6 shows BEV expert activation heatmaps confirming that different experts specialize across driving scenarios.

![[x2 20.png]]
*Figure 6: BEV visualizations of motion planning trajectories and expert activation heatmaps for different driving scenarios — confirming expert specialization.*

## GSPO: Group Sequence Policy Optimization

![[gspo.png]]
*Figure 4: GSPO integrates multi-factor safety rewards (no collisions, drivable-area compliance, TTC, comfort, ego progress) into masked diffusion trajectory optimization.*

**Motivation**: standard token-level GRPO is unstable for MoE architectures. Each token update changes expert routing decisions for that token, causing incoherent gradient signals across the sequence. GSPO resolves this by optimizing at the **sequence level** — each complete trajectory receives a single advantage, and the policy update is measured in sequence-likelihood space.

### GSPO Formulation

1. Sample $G=3$ candidate sequences $\{x_i\}_{i=1}^G \sim \pi_{\theta_\text{old}}(\cdot|c_t)$
2. Evaluate each in NAVSIM → rewards $R_i = r(c_t, x_i)$
3. Group-normalized advantage: $\hat{A}_i = \frac{R_i - \text{mean}(\{R_j\})}{\text{std}(\{R_j\})}$
4. Length-normalized sequence likelihood ratio (estimated via one-step unmasking):
$$s_i(\theta) = \exp\!\left(\frac{1}{|x_i|}\sum_{k=1}^{|x_i|}\log\frac{\pi_\theta(x_{i,k}|c_t)}{\pi_{\theta_\text{old}}(x_{i,k}|c_t)}\right)$$
5. Clipped PPO-style objective at sequence level:
$$J_\text{GSPO}(\theta) = \mathbb{E}_x\!\left[\frac{1}{G}\sum_{i=1}^G \min\!\Big(s_i(\theta)\hat{A}_i,\,\text{clip}\big(s_i(\theta),1-\epsilon,1+\epsilon\big)\hat{A}_i\Big)\right]$$

**Key property**: the clipping constraint is in *sequence-likelihood* space, not token space — appropriate for MoE since the constraint applies to the whole trajectory rather than individual routing decisions. This provides stable credit to the MoE policy without disrupting expert specialization.

### GSPO vs. GRPO

![[gspo2grpo.png]]
*Figure 10: Training reward curves. GSPO maintains steadily higher reward than GRPO throughout training, demonstrating more stable optimization.*

| Aspect | GRPO (token-level) | GSPO (sequence-level) |
|---|---|---|
| Advantage unit | Per token | Per trajectory |
| MoE routing stability | Unstable (each token update re-routes) | Stable (trajectory as atomic unit) |
| Credit assignment | Token-wise | Sequence-level |
| Training curves | Noisy, lower ceiling | Steady improvement |
| Group size (optimal) | 2–4 typical | **3** (WAM-Diff) |

**Ablation: GSPO group size**

| Config | PDMS |
|---|---|
| w/o GSPO | 86.6 |
| GSPO G=2 | 88.9 |
| **GSPO G=3** | **91.0** |

GSPO adds +5.3 PDMS (86.6 → 91.0) — the single largest gain in the WAM-Diff ablation stack.

**Reward design ablation** (Table 6): using only individual sub-rewards (TTC only, Comfort only, EP only) each improves the targeted metric but degrades overall PDMS. Using the full composite PDMS reward achieves the best balance:

| Reward | PDMS |
|---|---|
| TTC only | 90.7 |
| Comfort only | 90.6 |
| EP only | 90.3 |
| **All (PDMS)** | **91.0** |

## Classifier-Free Guidance

CFG applied during masked diffusion inference: instruction/context tokens dropped with some probability during training; guidance scale applied at inference. Consistent improvement with scale:

| CFG Scale | PDMS |
|---|---|
| w/o CFG | 82.3 |
| 1 | 83.2 |
| 3 | 83.6 |
| 5 | 84.4 |
| **7** | **84.7** |

Ablation contribution: +2.4 PDMS.

## Four-Stage Training

| Stage | What | Data | Epochs | Notes |
|---|---|---|---|---|
| I: MoE Warm-up | Backbone frozen; LoRA experts only | nuPlan 668K trajectories | 0.2 | Initialize planning capability; prevent mode collapse |
| II: Multi-task SFT | All params unfrozen; joint trajectory+VQA | nuPlan 668K + VQA 800K | 1 | Couple motion prediction with semantic scene understanding |
| III: NAVSIM Adaptation | Full fine-tuning | NAVSIM 103K | 3 | Bridge distribution gap nuPlan → NAVSIM |
| IV: GSPO RL | Online RL | NAVSIM 103K | 2 (×2 epochs) | First epoch → reference model for second epoch |

Hardware: $4 \times 8$ Ascend 910B NPUs. All stages: AdamW, lr $1\times10^{-5}$, cosine schedule, warmup 0.02, weight decay 0, batch size 1, bf16.

## Results

### NAVSIM-v1

| Method | NC↑ | DAC↑ | TTC↑ | Comf.↑ | EP↑ | **PDMS↑** |
|---|---|---|---|---|---|---|
| UniAD | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| DiffusionDrive | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| ReCogDrive | 97.9 | 97.3 | 94.9 | 100 | 87.3 | 90.8 |
| DriveVLA-W0 | 98.7 | 99.1 | 95.3 | 99.3 | 83.3 | 90.2 |
| **WAM-Diff (Ours)** | **99.1** | **98.3** | **96.5** | **99.9** | **84.4** | **91.0** |

WAM-Diff achieves 91.0 PDMS — outperforming DiffusionDrive (+2.9), ReCogDrive (+0.2), DriveVLA-W0 (+0.8). NC (99.1) is the highest in the comparison table, indicating strong collision avoidance.

### NAVSIM-v2 (Extended Metrics)

| Method | NC↑ | DAC↑ | DDC↑ | TLC↑ | EP↑ | TTC↑ | LK↑ | HC↑ | EC↑ | **EPDMS↑** |
|---|---|---|---|---|---|---|---|---|---|---|
| TransFuser | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 |
| HydraMDP++ | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | 81.4 |
| DiffusionDrive | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | 84.5 |
| DriveVLA-W0 | 98.5 | 99.1 | 98.0 | 99.7 | 86.4 | 98.1 | 93.2 | 97.9 | 58.9 | 86.1 |
| **WAM-Diff (Ours)** | **99.0** | **98.4** | **99.3** | **99.9** | **87.0** | **98.6** | **96.2** | **98.1** | **78.5** | **89.7** |

89.7 EPDMS — +5.2 over DiffusionDrive, +3.6 over DriveVLA-W0. EC = 78.5 (below DiffusionDrive's 87.7 and DriveFine's reported values — the masked diffusion approach doesn't specifically target extended comfort). Note: this uses "NAVSIM-v2 v2.2 codebase version" — see caveat below.

### nuScenes

| Method | ST-P3 Avg. Coll. ↓ | UniAD Avg. Coll. ↓ |
|---|---|---|
| UniAD | 0.12 | 0.31 |
| GPT-Driver | 0.17 | 0.44 |
| AutoVLA | 0.20 | 0.31 |
| **WAM-Diff (Ours)** | **0.11** | **0.28** |

Best average collision rate among all VLA methods under both evaluation protocols. Matches UniAD (non-VLA) under ST-P3 metrics.

## Full Ablation Stack (Table 5)

| Config | DS | CFG | MoE | GSPO | PDMS |
|---|---|---|---|---|---|
| Baseline | ✗ | ✗ | ✗ | ✗ | 80.3 |
| + Discrete tokenization | ✓ | ✗ | ✗ | ✗ | 82.3 (+2.0) |
| + CFG (scale 7) | ✓ | ✓ | ✗ | ✗ | 84.7 (+2.4) |
| + LoRA MoE + VQA | ✓ | ✓ | ✓ | ✗ | 86.6 (+1.9) |
| **+ GSPO (G=3)** | ✓ | ✓ | ✓ | ✓ | **91.0 (+4.4)** |

Note: Table 5 shows +4.4 for GSPO step; Table 3 ablation shows 86.6 → 91.0 = +4.4. The summary in the text says "+5.3 PDMS" which appears to reference the GSPO gain from w/o GSPO = 86.6.

## Qualitative Results

![[x1 22.png]]
*Figure 5: Qualitative comparison with open-source methods (DiffusionDrive, TransFuser) on NAVSIM benchmark.*

![[x3 20.png]]
*Figure 8: GSPO ablation — trajectory improvements in complex driving scenarios after online RL.*

![[x4 19.png]]
*Figure 9: Decoding order analysis — causal schedule effective for turns; reverse-causal for car-following and oncoming scenarios.*

![[x5 19.png]]
*Figure 11: Additional qualitative comparisons with DiffusionDrive and TransFuser.*

![[x6 17.png]]
*Figure 12: Additional qualitative comparisons.*

![[x7 14.png]]
*Figure 13: Qualitative examples with causal decoding schedule.*

![[x8 10.png]]
*Figure 14: Qualitative examples with reverse-causal decoding schedule.*

## Limitations and Failure Cases

![[fc2.png]]
*Figure 7: Representative failure cases.*

1. **Single front-view camera only**: no surround perception → obstacles outside the front FOV cause failures. Root cause: computational constraints. Future work: 3D vision encoder with textual feature alignment.

2. **No temporal history**: single-frame input only. The model cannot infer other agents' motion patterns or intent from their historical trajectories → suboptimal planning decisions in dynamic interactive scenarios. Future work: efficient architectures for temporal context integration.

These two limitations create the failure modes shown in Figure 7: (a) side-impact collisions from vehicles entering from off-FOV angles; (b) incorrect intent inference for cutting-in vehicles with no observable motion history.

## Relationship to Wiki Papers

| Aspect | WAM-Diff | DriveFine | ReflectDrive | WAM-Flow |
|---|---|---|---|---|
| Base paradigm | Masked diffusion | Masked diffusion | Masked diffusion | Discrete flow matching |
| Backbone | LLaDA-V (8.4B) | LLaDA-8B | LLaDA-V | Custom |
| MoE | LoRA MoE (64 experts) | Block-MoE (refinement expert) | None | None |
| RL | GSPO (sequence-level) | Hybrid offline+online GRPO | None (inference-time safety) | GRPO |
| Decoding schedule | Causal/RCausal/Random | Fixed | Iterative reflection | DFM transport |
| NAVSIM-v1 PDMS | 91.0 | 90.7 / 91.8★ | >89.1 (claimed) | 90.3 |
| NAVSIM-v2 EPDMS | 89.7 | 89.7 (bug-fixed scorer) | — | 84.7 |
| 1-cam | Yes | Yes | Yes | Yes |

**Key WAM-Diff differentiator vs. DriveFine**: DriveFine uses token-level GRPO + a separate refinement expert to catch and correct errors; WAM-Diff uses sequence-level GSPO which avoids the token-level instability problem but has no explicit error correction mechanism. DriveFine's block-MoE is an architectural add-on to a standard masked diffusion backbone; WAM-Diff's LoRA MoE is a capacity scaling mechanism inside the backbone FFNs.

**Key WAM-Diff differentiator vs. WAM-Flow**: WAM-Flow uses CTMC discrete flow matching (transport from noise → data), while WAM-Diff uses masked diffusion (absorbing-state infilling). Both achieve 90.3–91.0 PDMS from the same Fudan-adjacent research lineage, suggesting the exact discrete generation mechanism (DFM vs. masked diffusion) matters less than the MoE + RL combination. WAM-Flow uses WAM (Waypoint-Anchored Masking) while WAM-Diff uses the standard masked NLL objective.

---
title: "DriveVLA-W0: World Models Amplify Data Scaling Law in Autonomous Driving"
type: source-summary
sources: [raw/papers/DriveVLA-W0_ World Models Amplify Data Scaling Law in Autonomous Driving.md]
related: [concepts/world-model-for-ad.md, concepts/navsim-benchmark.md, concepts/diffusion-planner.md]
created: 2026-04-07
updated: 2026-04-07
confidence: high
---

**Paper**: DriveVLA-W0: World Models Amplify Data Scaling Law in Autonomous Driving  
**Orgs**: CASIA (Institute of Automation, Chinese Academy of Sciences), Yinwang Intelligent Technology  
**arXiv**: 2510.12796v1

---

## Summary

DriveVLA-W0 identifies the **"supervision deficit"** as the fundamental bottleneck limiting VLA scalability: fine-tuning on expert actions alone compresses high-dimensional visual inputs to a handful of low-dimensional waypoints, leaving most of the VLA's representational capacity idle. The proposed fix — future image prediction as dense self-supervised signal — forces the model to learn environment dynamics, unlocking data scaling laws that action-only supervision cannot achieve. Two world model variants are implemented, one per VLA paradigm (discrete tokens vs. continuous features), and a lightweight MoE Action Expert decouples inference latency from the large backbone. A scaling study at 70M frames (proprietary) provides the strongest empirical case for world modeling as a training-time self-supervision strategy.

---

## Architecture

![[x1 16.png|Supervision deficit and world modeling as catalyst]]

Figure 1: (a) Standard VLAs supervised only on actions; DriveVLA-W0 additionally supervised on future visual scenes. (b) World modeling enables sustained performance gains with data scaling, unlike action-only baselines that saturate.

![[x2 14.png|AR and Diffusion World Model architectures]]

Figure 2: (a) AR World Model: discrete visual token prediction on Emu3-8B. (b) Diffusion World Model: latent diffusion conditioned on VLA features on Qwen2.5-VL-7B.

### VLA Baseline

Both variants share an input structure: interleaved multimodal history $S_t = [L_{t-H}, V_{t-H}, A_{t-H-1}, \ldots, L_t, V_t, A_{t-1}]$ processed autoregressively by the backbone. Actions are tokenized with the FAST tokenizer into discrete tokens; action prediction uses standard cross-entropy.

### World Model Variant A — AR World Model (VQ backbone: Emu3-8B)

Predicts the **current** frame's discrete visual tokens autoregressively, conditioned on preceding context:

$$\mathcal{L}_{\text{WM-AR}} = -\sum_{i=1}^{N} \log P(v_i \mid S_{<V_t}, v_{<i})$$

Joint training: $\mathcal{L} = \mathcal{L}_{action} + \alpha \mathcal{L}_{WM-AR}$

At inference, visual token prediction is bypassed for latency. Images can be generated via MoVQGAN decoder for visualization.

### World Model Variant B — Diffusion World Model (ViT backbone: Qwen2.5-VL-7B)

Predicts the **next** frame $I_{t+1}$ (future, not current — avoids pure reconstruction) via latent diffusion conditioned on current VLA output features:

$$\mathcal{L}_{\text{WM-Diff}} = \mathbb{E}_{z_{t+1}, \epsilon, k}\left[\|\epsilon - \hat{\epsilon}(z_{t+1,k}, k, F_t^V, F_t^A)\|^2\right]$$

Joint training: $\mathcal{L} = \mathcal{L}_{action} + \beta \mathcal{L}_{WM-Diff}$

Key design choice: conditioning on *action* features $F_t^A$ forces the model to predict the visual consequence of a specific action, not a generic future.

### MoE Action Expert (500M parameters)

![[x3 15.png|MoE architecture and three action decoding schemes]]

Figure 3: (a) Joint Attention: VLA Expert and Action Expert share QKV across token sequence dimension. (b-d) Three expert variants: query-based, autoregressive, flow matching.

A lightweight 500M action expert operates alongside the large VLA backbone. **Joint Attention** mechanism:

$$Q = [Q_{\text{VLA}}; Q_{\text{AE}}], \quad K = [K_{\text{VLA}}; K_{\text{AE}}], \quad V = [V_{\text{VLA}}; V_{\text{AE}}]$$

Attention computed jointly, output split and routed back to each expert. Previous action features $A_{t-1}$ are always prefilled as temporal prior.

**Three expert variants**:

| Expert | Mechanism | Output |
|---|---|---|
| Query-based | Learnable queries → joint attention → MLP regression | Continuous waypoints, L1 loss |
| Autoregressive | FAST token prediction → cross-entropy | Discrete tokens → FAST detokenizer |
| Flow Matching | Vector field $v_\phi$: straight-line path from noise to action | Continuous, ODE integration |

### Two-Stage Training

1. **Stage 1**: Pretrain VLA backbone on NuPlan with 6VA interleaved sequence; joint world model + action loss
2. **Stage 2**: Integrate Action Expert; VLA backbone processes 2VA input; supervised only by action expert loss

NAVSIM: 8k NuPlan pretraining + 4k NAVSIM fine-tune; 8×L20 GPUs, batch=48, lr=2e-4, 256×144 images.  
In-house: 50k pretraining + 30k fine-tune; 64 GPUs, batch=256.

---

## Results

### NAVSIM-v1 (Table 1)

| Method | Sensors | NC ↑ | DAC ↑ | TTC ↑ | C ↑ | EP ↑ | PDMS ↑ |
|---|---|---|---|---|---|---|---|
| ReCogDrive | 3C | 98.2 | 97.8 | 95.2 | 99.8 | 83.5 | 89.6 |
| DriveVLA-W0★ | 1C | 98.7 | 99.1 | 95.3 | 99.3 | 83.3 | **90.2** |
| AutoVLA† (BoN-6) | 3C | 99.1 | 97.1 | 97.1 | 100.0 | 87.6 | 92.1 |
| **DriveVLA-W0† (BoN-6)** | **1C** | **99.3** | **97.4** | **97.0** | **99.9** | **88.3** | **93.0** |

**★ Caveat on 90.2**: uses "query-based action expert with multiple trajectory anchors following Hydra-MDP" — not single-sample. Trajectory anchors = generating multiple anchor candidates and selecting; analogous to a soft BoN. Presented in Table 1 alongside pure single-sample results (ReCogDrive 89.6, AutoVLA 89.1) without prominent footnote. The underlying single-sample performance of the query-based expert is 88.4 PDMS (Table 4).

**BoN-6 (93.0)**: uses AR action expert + best-of-6 selection. Matches AdaThinkDrive BoN-4 (93.0) and exceeds AutoVLA BoN-6 (92.1 — cross-paper comparison, may differ in selection method).

### NAVSIM-v2 (Table 2)

| Method | EPDMS | EC |
|---|---|---|
| DiffusionDrive | 84.5 | 87.7 |
| **DriveVLA-W0** | **86.1** | **58.9** |

EC = 58.9 is by far the lowest in the wiki — world model training produces good safety/progress but very poor extended comfort compliance.

### In-House 70M-Frame Dataset (Table 3)

| Model | 70k ADE | 700k ADE | 70M ADE | 70M Col. |
|---|---|---|---|---|
| VLA (VQ) baseline | 2.852m | 1.542m | 1.483m | 4.88% |
| VLA-W0 (VQ) + WM | 2.748m | 1.599m | **1.056m** | **3.92%** |
| VLA (ViT) baseline | 3.152m | 1.420m | 1.105m | 3.59% |
| VLA-W0 (ViT) + WM | 2.527m | 1.344m | **1.064m** | **3.02%** |

At 70M frames: VQ world model +28.8% ADE, +19.7% collision vs. action-only baseline. ViT +3.7% ADE, +15.9% collision.

At small scale (70k), world model VQ slightly *hurts* ADE (+3.6% worse) — the dense WM signal requires more data to overcome its optimization overhead.

---

## Key Findings

### Finding 1: World Modeling Unlocks Cross-Dataset Generalization (Table 7)

| Model | Train-from-scratch PDMS | NuPlan→NAVSIM PDMS | Transfer Δ |
|---|---|---|---|
| TransFuser-50M | 79.2 | 83.6 | +5.5% |
| TransFuser-7B | 77.9 | 71.6 | **−8.1%** |
| VLA-VQ | 68.7 | 62.2 | **−9.5%** |
| VLA-W0-VQ (Ours) | 80.7 | 85.6 | **+6.1%** |
| VLA-ViT | 70.3 | 70.6 | +0.4% |
| VLA-W0-ViT (Ours) | 83.9 | 85.3 | **+1.7%** |

Larger action-only VLAs overfit NuPlan's action distribution and **degrade** when fine-tuned on NAVSIM. World model VLAs learn transferable visual representations → consistently benefit from pretraining.

![[x4 13.png|Generalization with data scaling]]

Figure 4: Red arrows = action-only models hurt by pretraining; green arrows = world model models benefiting. Visual domain similar (both road scenes) but action distributions differ significantly.

### Finding 2: Action Decoder Scaling Reversal (Table 4)

| Expert | NAVSIM PDMS | 70M ADE | 70M Col. |
|---|---|---|---|
| Query-based | **88.4** | 1.124m | 4.53% |
| Flow Matching | 87.2 | 1.036m | 3.98% |
| AR | 85.3 | **1.007m** | **2.95%** |

At NAVSIM scale (103k frames): query-based > FM > AR  
At 70M frames: AR > FM > query-based

**Explanation**: At small scale, continuous decoders (query-based, FM) win because trajectory distribution is simpler and their precision advantage matters. At massive scale, AR's teacher-forced training scales better; FM is too sample-inefficient to converge on the complex 70M trajectory distribution; query-based hits a representational bottleneck.

### Finding 3: FID ↔ PDMS Positive Correlation (Table 8)

| Sequence | FID ↓ | PDMS ↑ |
|---|---|---|
| 2VA | 9.847 | 84.1 |
| 6VA | 4.610 | 85.6 |
| MoVQGAN upper bound | 3.007 | — |

Better generation fidelity = better planning performance. (Note: only 2 data points — weak evidence.)

---

## Ablations

### Sequence Design (Table 5)

| Pretrain | Finetune | PDMS |
|---|---|---|
| None | 2VA | 80.7 |
| 6V (vision only) | 2VA | 84.1 |
| **6VA (vision+action)** | **2VA** | **85.6** |

Vision-action interleaving (+1.5 vs. vision-only): grounding visual predictions in ego actions forces learning of causal dynamics, not just visual appearance.

### Sequence Length (Table 6)

| Pretrain | Finetune | PDMS |
|---|---|---|
| VA | VA | 83.3 |
| 2VA | 2VA | 84.2 |
| **6VA** | **2VA** | **85.6** |

Longer temporal context → better dynamics capture. +1.3 PDMS from 1-step to 6-step history.

### Temporal Horizon (Table 9)

| Input type | Temporal interval | PDMS |
|---|---|---|
| VA (current only) | — | 82.9 |
| VAVA (2 frames) | 4 seconds | 84.3 |
| **VAVA (2 frames)** | **1 second** | **85.6** |

1 second interval optimal. Too long (4s) = excessive scene variation; too short (current only) = no temporal dynamics.

### Latency (Figure 5)

![[x5 13.png|Latency analysis across action expert types]]

| Expert | Latency (NAVSIM) | Latency (70M) | Scales with |
|---|---|---|---|
| Query-based | 74ms | 74ms | Constant |
| Flow Matching (10 steps) | 145ms | 145ms | Constant |
| AR | 95ms (5.6 tok avg) | 170ms (17.8 tok avg) | Token length |
| Full VLA baseline | 118ms | 240ms | Token length |

MoE reduces latency to 63.1% of baseline (74ms vs. 118ms for query-based).

---

## Limitations

1. **90.2 PDMS uses trajectory anchors** — effectively a multi-candidate selection scheme, not single-sample; the pure single-sample query-based performance is 88.4 PDMS (Table 4), below DriveFine (90.7) and WAM-Flow (90.3)
2. **EC = 58.9 on NAVSIM-v2** — lowest in the wiki; world model training does not improve comfort compliance
3. **In-house 70M dataset is proprietary** — the paper's most compelling scaling results are not reproducible by the community
4. **Design asymmetry between WM variants**: AR WM predicts current frame (reconstruction-adjacent); Diffusion WM predicts future frame — different hypotheses tested under the same "world modeling" label
5. **World model bypassed at inference** — training-time benefit only; no inference-time visual reasoning (unlike FSDrive)
6. **FID ↔ PDMS correlation uses only 2 data points** — claimed as "compelling evidence" but statistically weak
7. **Single front-view camera only** throughout all experiments
8. **No nuScenes, Bench2Drive, or Waymo evaluation**
9. **Scaling reversal claim requires in-house data** to observe — on NAVSIM alone, FM and query-based both outperform AR

---

## Key Cross-References

- **Supervision deficit + world model for VLA scaling**: [[concepts/world-model-for-ad.md]] — Pattern 7: training-time-only world modeling for data scaling
- **NAVSIM standings**: [[concepts/navsim-benchmark.md]] — 90.2★ (anchors), 93.0 BoN-6 (1 cam); 86.1 EPDMS v2 (EC=58.9)
- **Action decoder scaling reversal (FM vs. AR)**: [[concepts/diffusion-planner.md]] — FM loses to AR at massive scale

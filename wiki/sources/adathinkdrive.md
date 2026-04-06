---
title: "AdaThinkDrive: Adaptive Thinking via Reinforcement Learning for Autonomous Driving"
type: source-summary
sources: [raw/papers/AdaThinkDrive_ Adaptive Thinking via Reinforcement Learning for Autonomous Driving.md]
related: [concepts/rl-for-ad.md, concepts/vlm-domain-adaptation.md, concepts/navsim-benchmark.md]
created: 2026-04-06
updated: 2026-04-06
confidence: high
---

**Paper**: AdaThinkDrive: Adaptive Thinking via Reinforcement Learning for Autonomous Driving  
**Orgs**: Xiaomi EV, Tsinghua University, University of Macau, NTU, Peking University  
**arXiv**: 2509.13769v1

---

## Summary

AdaThinkDrive argues that CoT reasoning hurts performance in simple scenarios — a claim verified empirically on InternVL3-8B/2B across 3 complexity levels (Non-Think wins at Level 1, Think wins at Levels 2–3). The system trains a dual-mode model (Think/Non-Think) via SFT, then uses an **Adaptive Think Reward** in GRPO that dynamically relabels scene complexity based on the evolving policy's own rollout comparisons, teaching the model to selectively reason.

---

## Empirical Motivation

![[fig1a_new1.png|CoT performance vs. scene complexity]]

Figure 1a: For both InternVL3-8B and 2B, Non-Think outperforms Think at Level 1 (simple); Think consistently outperforms at Levels 2 and 3. The optimal reasoning strategy is scene-complexity-dependent.

This motivates the full system: instead of always applying CoT (over-computation in simple cases) or never applying it (under-performance in complex cases), learn *when* to reason.

---

## Architecture

**Backbone**: InternVL3-8B  
**Input**: Single front-view camera + high-level navigation command + ego state (velocity, acceleration) + 3-frame historical trajectory  
**Output mode**: joint distribution $\mathcal{P}(m, o \mid q) = \mathcal{P}(m \mid q) \cdot \mathcal{P}(o \mid q, m)$ where $m \in \{\text{Think}, \text{Non-Think}\}$

**Objective**: learn policy $\pi: q \mapsto m$ maximizing expected utility:
$$m(q) = \arg\max_{m \in \mathcal{M}} \mathbb{E}_{o \sim \mathcal{P}(o \mid q, m)}[\mathcal{U}(q, o)]$$

---

## Data Preparation

### Pre-training Data

Open-source driving QA datasets: DriveLM, LingoQA, ImpromptuVLA, NuScenes-QA, NuInstruct, OmniDrive. Also constructs NAVSIM multi-turn CoT QA covering: road boundary estimation, critical object identification, ego action prediction, and scene understanding subtasks.

### Hybrid SFT Data

For each query $q$, two responses are generated:
- `{q, o^Think}`: `<think>reasoning</think><answer>trajectory</answer>` — full reasoning trace
- `{q, o^NonThink}`: `<answer>trajectory</answer>` — direct output, no reasoning

$$\mathcal{D}^{SFT} = \{\{q, o^{Think}\}, \{q, o^{Non\text{-}Think}\}\}_{q \in \mathcal{Q}}$$

**CoT content in Think mode** (rule-based + Qwen2.5-VL-72B annotated):
- Road boundary estimation from NAVSIM map (lane topology + critical boundary features along ego's future path)
- Dynamic agent classification into three types:
  - **CIPO-1**: Closest In-Path Object (in ego lane)
  - **CIPO-2**: Likely to merge (determined by lane geometry + relative position)
  - **Motion Interaction**: Predicted trajectory intersects ego future path
- Traffic light states and weather (auto-annotated by Qwen2.5-VL-72B)

![[16f834c7829a576f.jpg|CIPO agent classification]]

Figure 3: CIPO-1 (in-lane), CIPO-2 (merge candidate), Motion Interaction (trajectory-crossing) agent categories.

### Scene Complexity Categorization

Three levels based on two binary conditions (boundary proximity + critical objects present):

| Level | Boundary Near? | Critical Objects? | Label |
|---|---|---|---|
| 1 | ✗ | ✗ | Simple ($\mathcal{D}^-$) |
| 2 | One of two | — | Moderate |
| 3 | ✓ | ✓ | Challenging ($\mathcal{D}^+$) |

Levels 2 and 3 together form $\mathcal{D}^+$ (challenging); Level 1 is $\mathcal{D}^-$ (simple). These serve as cold-start labels for Stage 3 RL.

---

## 3-Stage Training

### Stage 1: Pre-training (driving QA)
SFT on large-scale driving QA to build world knowledge and driving commonsense. 2 epochs, lr=1e-5, batch=1.

### Stage 2: Dual-mode SFT (trajectory)
Joint CE loss over both Think and Non-Think outputs for the same query:
$$\mathcal{L}_{SFT} = \mathbb{E}_{(q,o) \sim \mathcal{D}^{SFT}} [-\log \pi_\theta(o \mid q)]$$

No bias toward either style — the model learns to produce both. 2 epochs, lr=4e-5, batch=2.

### Stage 3: GRPO with Adaptive Think Reward
64 NVIDIA H20 GPUs, 2 epochs, lr=2e-6, batch=4.

**Total reward**:
$$\mathcal{R}(q, a) = \mathcal{R}_{traj} + \mathcal{R}_{fmt} + \mathcal{R}_{endpoint} + \mathcal{R}_{adaptive}$$

---

## Reward Design

### PDMS Reward ($\mathcal{R}_{traj}$)
NAVSIM PDMS for the predicted trajectory. Discrete 0–1.

### Format Reward ($\mathcal{R}_{fmt}$)
Enforces correct use of `<think>...</think>` and `<answer>...</answer>` tags plus standardized trajectory representation. Discrete violation penalty.

### Endpoint Reward ($\mathcal{R}_{endpoint}$)
Piecewise L1 distance to GT trajectory endpoint:

| L1 Distance | Reward |
|---|---|
| < 2m | 1.0 |
| < 4m | 0.8 |
| < 6m | 0.6 |
| < 10m | 0.4 |
| < 15m | 0.2 |
| ≥ 15m | 0.0 |

Penalizes large errors while rewarding small deviations near the endpoint.

### Adaptive Think Reward ($\mathcal{R}_{adaptive}$) — Core Novelty

![[fig5.jpg|Adaptive Think Reward diagram]]

Figure 4: Adaptive Think Reward adjusts reasoning behavior by identifying misclassified scenes via rollout comparison.

For each scene with initial complexity label $D$, compute:
- $S_{Think}$: average PDMS of Think rollouts
- $S_{NoThink}$: average PDMS of Non-Think rollouts
- $C_{Think}$, $C_{NoThink}$: count of each mode in the rollout group
- Threshold $T = 0.9$

**Algorithm 1:**

**Case D=0 (Simple):**
- If $S_{Think} > S_{NoThink}$ AND $S_{Think} > T$ AND $C_{Think} > C_{NoThink}$:
  → Scene relabeled as Challenging → $\text{Reward}_{Think} = 1$, $\text{Reward}_{NoThink} = 0$
- Else:
  → Maintained as Simple → $\text{Reward}_{Think} = 0$, $\text{Reward}_{NoThink} = 1$

**Case D=1 (Challenging):**
- If $S_{NoThink} > S_{Think}$ AND $S_{NoThink} > T$ AND $C_{NoThink} > C_{Think}$:
  → Scene relabeled as Simple → $\text{Reward}_{Think} = 0$, $\text{Reward}_{NoThink} = 1$
- Else:
  → Maintained as Challenging → $\text{Reward}_{Think} = 1$, $\text{Reward}_{NoThink} = 0$

**Key property**: The initial scene labels from Stage 2 are anchors only — the model corrects them dynamically based on its own rollout performance. If the policy has improved enough that Non-Think suffices on a previously-challenging scene (or vice versa), the reward relabels accordingly. This prevents reward stagnation as the policy evolves.

### GRPO Objective

$$\mathcal{J}(\theta) = \mathbb{E}_{q, \{o_i\} \sim \pi_{\theta_{old}}} \left[ \frac{1}{G} \sum_{i=1}^G \mathcal{J}_i - \beta \mathbb{D}_{KL}(\pi_\theta \| \pi_{ref}) \right]$$
$$\mathcal{J}_i = \min(c_i A_i, \text{clip}(c_i, 1-\epsilon, 1+\epsilon) A_i), \quad c_i = \frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{old}}(o_i \mid q)}$$

Advantage $A_i$ is the normalized reward difference within the group.

---

## Results

### NAVSIM-v1 Benchmark (Table I)

| Method | Camera | LiDAR | NC↑ | DAC↑ | TTC↑ | Comf↑ | EP↑ | PDMS↑ |
|---|---|---|---|---|---|---|---|---|
| Hydra-NeXt | ✓ | ✗ | 98.1 | 97.7 | 94.6 | 100 | 81.8 | 88.6 |
| DiffusionDrive | ✓ | ✓ | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| GoalFlow | ✓ | ✓ | 98.4 | 98.3 | 94.6 | 100 | 85.0 | 90.3 |
| **AdaThinkDrive** | **✓** | **✗** | **98.4** | **97.8** | **95.2** | **100** | **84.4** | **90.3** |
| AdaThinkDrive BoN-4 | ✓ | ✗ | 99.1 | 98.8 | 97.2 | 100 | 87.9 | **93.0** |

Vision-only SOTA at 90.3 PDMS, matching GoalFlow (camera+LiDAR). **Caveat**: Table I does not include WAM-Flow (90.3, 1 cam) or Curious-VLA (90.3, 1 cam) — all three independently claim 90.3 PDMS; direct head-to-head absent.

### Adaptive vs. Fixed Reasoning (Table II)

| Model | Mode | PDMS |
|---|---|---|
| Non-Think SFT | w/o CoT | 83.3 |
| Think SFT | w/ CoT | 86.2 |
| Non-Think RL | w/o CoT | 88.3 |
| Think RL | w/ CoT | 88.9 |
| **AdaThinkDrive** | **Adaptive** | **90.3** |

+2.0 vs. Non-Think RL; +1.4 vs. Think RL.

### Per-Level Analysis (Table IV)

| Comparison | Level | Baseline PDMS | AdaThinkDrive | Δ |
|---|---|---|---|---|
| vs. Non-Think RL | Level 1 (simple) | 88.5 | 90.7 | +2.2 |
| vs. Think RL | Level 3 (challenging) | 87.8 | 89.8 | +2.0 |

Adaptive beats the optimal fixed mode at every complexity level.

### Inference Time (Table III)

| Model | Infer Time↓ | PDMS↑ |
|---|---|---|
| Non-Think RL | 0.68s | 88.3 |
| **AdaThinkDrive** | **0.74s** | **90.3** |
| Think RL | 0.86s | 88.9 |

+9% time vs. Non-Think RL; −14% vs. Think RL; +2.0 PDMS vs. Non-Think RL.

### Behavioral Analysis (Figure 5)

![[fig7.jpg|Think vs. Non-Think ratio by scene level]]

- Level 1 (simple): **84% Non-Think**, 16% Think
- Level 3 (challenging): 4% Non-Think, **96% Think**

Confirms the model has learned the intended adaptive policy.

---

## Ablations

### Training Pipeline (Table V)

| Config | PDMS |
|---|---|
| SFT only | 86.2 |
| Pre-training + SFT | 87.5 (+1.3) |
| **Pre-training + SFT + RL** | **90.3 (+2.8)** |

Pre-training contributes +1.3; RL contributes +2.8 (largest single gain).

### Reward Components (Table VI)

| PDMS+Format | +Endpoint | +Adaptive Think | PDMS |
|---|---|---|---|
| ✓ | | | 88.1 |
| ✓ | ✓ | | 89.1 |
| ✓ | ✓ | ✓ | **90.3** |

Adaptive Think Reward adds +1.2 on top of PDMS+Format+Endpoint. Without it, the model reaches only 89.1.

---

## Qualitative Analysis

![[fig6.jpg|Qualitative trajectory comparisons]]

Figure 6: Simple scenario (top) — Think RL misclassifies a distant object as critical, reasons unnecessarily, strays from drivable area. AdaThinkDrive skips reasoning, outputs smooth trajectory. Challenging scenario (bottom) — Non-Think RL fails to assess distance to lead vehicle. AdaThinkDrive identifies the critical object and generates a safe trajectory.

---

## Limitations

1. **Single front camera only** — all wiki peers use 6–8 cameras; no surround-view, no depth cues from additional cameras
2. **NAVSIM-v1 only** — no NAVSIM-v2/EPDMS, no Bench2Drive, nuScenes, or Waymo evaluation
3. **Comparison gap**: Table I excludes WAM-Flow (90.3, 1 cam) and Curious-VLA (90.3, 1 cam); claims "+1.7 vs. best vision-only" but the best vision-only reference (Hydra-NeXt 88.6) is a non-VLM method — VLM baselines at the same score exist in the wiki
4. **No safety reward**: PDMS serves as proxy; no explicit collision or jerk penalty (unlike AR1)
5. **Static initial scene categorization**: boundary proximity + critical objects only — may mis-categorize semantically complex scenes that lack these geometric signals
6. **Endpoint reward is piecewise (6 bins)**: coarser signal than continuous MSE (AutoDrive-R²)
7. **No NAVSIM-v2 / multi-camera evaluation**: extended metrics and surround-view capability not tested

---

## Key Cross-References

- **Adaptive Think Reward vs. AutoVLA CoT length penalty**: [[concepts/rl-for-ad.md]] — both train adaptive reasoning via GRPO but via different mechanisms (mode comparison vs. length penalty)
- **Dual-mode SFT**: [[concepts/vlm-domain-adaptation.md]] — same-query Think/Non-Think training vs. AutoVLA's fast/slow mixture
- **NAVSIM standing**: [[concepts/navsim-benchmark.md]] — 90.3 PDMS (vision-only), 93.0 BoN-4

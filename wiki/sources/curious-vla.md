---
title: "Curious-VLA: Devil is in Narrow Policy"
type: source-summary
sources: [raw/papers/Devil is in Narrow Policy_ Unleashing Exploration in Driving VLA Models.md]
related: [concepts/rl-for-ad.md, concepts/navsim-benchmark.md, concepts/diffusion-planner.md, concepts/vlm-domain-adaptation.md, sources/recogdrive.md]
created: 2026-04-05
updated: 2026-04-05
confidence: high
---

**arXiv**: https://arxiv.org/html/2603.06049  
**Affiliation**: Beihang University, Tsinghua University AIR, Lenovo Group  
**Code**: https://github.com/Mashiroln/curious_vla

## One-Line Summary

Curious-VLA identifies "Narrow Policy" (IL over-exploitation → GRPO advantage collapse) as a root cause of premature RL saturation in driving VLAs, and addresses it with diversity-expanding IL (FTE) and diversity-preserving RL (ADAS + SDR), achieving 90.3 PDMS with a 3B model and BoN-6 PDMS of 94.8 matching human GT.

## Motivation: The Narrow Policy Problem

Existing driving VLAs follow a two-stage pipeline: SFT (IL) → GRPO (RL). The authors show this pipeline has a **fundamental flaw**: IL collapses policy diversity, which starves RL of useful feedback.

**Three root causes:**

### (1) Optimization Objective Mismatch
Cross-entropy loss treats all non-GT tokens as equally wrong — no spatial proximity between trajectory tokens. Gradient w.r.t. logit $z_t$:
$$\frac{\partial\mathcal{L}_{\text{SFT}}}{\partial z_{t}}=\pi_{\theta}(y_{k}|\mathbf{y}^{*}_{<t},\mathcal{X})-\mathbb{I}(\hat{y}_{t}=y_{t}^{*})$$
There is no smoother incentive for near-correct predictions (e.g., token 31.4 vs 21.4 when GT = 31.5). This encourages overconfidence in $\mathbf{y}^*$, collapsing the policy around a single expert mode.

### (2) Horizon Physical Scale Mismatch
Future waypoints in ego-centric coordinates have much larger variance at distant horizons ($t=4s$) than near horizons ($t=0.5s$). Far-horizon losses dominate $\mathcal{L}_{\text{SFT}}$, while near-horizon steering actions — which determine behavioral diversity — contribute negligibly.

![[x4 8.png|Horizon scale mismatch: waypoint variance grows with prediction horizon]]

*Figure 3: Waypoints at $t=4s$ have orders-of-magnitude larger variance than $t=0.5s$. Without normalization, far-horizon losses dominate and near-horizon diversity signals are suppressed.*

### (3) Advantage Collapse in RL
When IL produces a narrow (low-entropy) policy, GRPO samples are nearly identical → rewards nearly identical → $\sigma_R \to 0$ → advantages vanish:
$$\lim_{\sigma_{R}\to 0}A_{i}=\frac{R(\mathbf{y}_{i})-\mu_{R}}{\sigma_{R}+\xi}\to 0$$
This produces vanishing gradients and premature RL saturation.

## Behavioral Diagnostics Framework

To quantitatively verify NP, the authors introduce three complementary metrics (all evaluated at $k=8$ samples per scenario):

| Metric | Measures | Formula | Desired |
|--------|---------|---------|---------|
| **Diversity** | Policy spread | Mean pairwise ADE/FDE across all pairs in $\mathcal{T}$ | ↑ high |
| **Quality** | Best feasible sample | $\min_{\tau_i \in \mathcal{T}} \text{ADE/FDE}(\tau_i, \tau^*)$ | ↓ low |
| **Performance** | Overall driving | Mean PDMS@$k$ in NAVSIM-v1 | ↑ high |

![[x1 11.png|Behavioral Diagnostics quantitative comparison]]

*Figure 1(a): Both QwenVL-2.5 (VLA-Token) and ReCogDrive (VLA-Planner) exhibit severely collapsed diversity with limited quality after IL. Well-balanced model: high Diversity, low Quality (min FDE), high Performance.*

**Baseline comparison (Tab. 4):**

| Method | Stage | Quality (ADE/FDE ↓) | Diversity (pADE/pFDE ↑) | Mean-PDMS ↑ |
|--------|-------|---------------------|------------------------|------------|
| ReCogDrive | IL+RL | 0.295 / 0.621 | 0.148 / 0.325 | 90.95 |
| Qwen2.5-VL (baseline) | IL | 0.481 / 1.052 | 0.090 / 0.200 | 90.69 |
| + FTE (w/o SN) | IL | 0.513 / 1.129 | 0.170 / 0.381 | 90.65 |
| + FTE | IL | 0.480 / 1.078 | 0.346 / 0.803 | 91.31 |
| **+ FTE + RL** | **IL+RL** | **0.269 / 0.547** | **0.641 / 1.415** | **91.55** |

Key observation: FTE without SN improves diversity but hurts quality (diverse but bad trajectories). SN is the critical catalyst enabling diverse trajectories to be learned effectively.

## Overall Pipeline

![[x3 9.png|Curious-VLA overall pipeline]]

*Figure 2: IL stage (left) — FTE generates diverse trajectories, CoT adds reasoning, SN normalizes per-step. RL stage (right) — ADAS filters for diversity, SDR amplifies reward span.*

## Stage 1: Feasible Trajectory Expansion (FTE)

FTE addresses the NP problem during IL through three sub-components:

### 1a. Exploratory Data Expansion (DE)
- Identify **12k challenging segments** (multi-lane, intersections, occlusion) from the 103k NavTrain set using Qwen2.5-VL-72B filtering
- Use **ReCogDrive's diffusion module** to generate diverse trajectories by perturbing diffusion latents
- Expand both **within-intent** (sampling around same goal) and **across-intent** (altering route-level decisions)
- Filter all candidates with NAVSIM PDMS scorer for safety
- Result: **142k safe and diverse samples** (from 103k original)

### 1b. Chain-of-Thought Data Synthesis (CoT)
Structure driving reasoning into a **4-stage chain** in single-turn dialogue:
1. Critical object perception
2. Driving explanation
3. Meta-behavior description
4. Trajectory prediction

Qwen2.5-VL-72B generates these structured reasoning sequences for the full expanded dataset.  
**SFT implementation**: two-stage — (i) freeze vision encoder + projector, align LLM to CoT format; (ii) unfreeze all, end-to-end fine-tuning.

### 1c. Step-wise Normalization (SN)
Normalize each prediction step $t$ independently to equalize gradient magnitudes across horizons:
$$\tilde{w}_{t}=\frac{w_{t}-\mu_{t}}{\sigma_{t}}, \quad \hat{w}_{t}=\hat{\tilde{w}}_{t}\sigma_{t}+\mu_{t}$$
where $(\mu_t, \sigma_t)$ are per-step statistics from the training set. During inference, de-normalize to recover physical trajectory.

**Effect**: equalizes gradient magnitudes across time horizons, improves near-horizon behavioral separability, enables effective learning from diverse trajectories.

## Stage 2: Diversity-Aware RL

### 2a. Adaptive Diversity-Aware Sampling (ADAS)

ADAS maintains sufficient reward variance for stable GRPO by filtering out scenarios where all rollouts yield identical outcomes.

**Bernoulli approximation**: model each scenario's outcome as a binary trial (success = high PDMS, failure = low PDMS) with estimated success probability $\hat{p} = \mu_R / R_{max}$.

**Two inclusion conditions** (both must be satisfied):
$$\hat{p}^{G}+(1-\hat{p})^{G}<\epsilon_{\text{div}}$$
$$|\sigma_{R}-\sqrt{\hat{p}(1-\hat{p})}R_{\text{range}}|<\epsilon_{\text{conf}}$$

- Condition 1: bounds probability that all $G$ online rollouts are identical (either all-success or all-failure)
- Condition 2: enforces consistency between empirical $\sigma_R$ and theoretical Bernoulli variance (filters noisy/unstable scenarios)

**Algorithm**: 3 outer-loops; at each loop, $M \gg G$ offline rollouts to estimate $\hat{p}$, build active set $\mathcal{D}_{active}$, then standard GRPO on active set.

**Critical negative finding**: difficulty-based sampling (human difficulty heuristic) causes **training collapse** (35.2 PDMS) — zero-advantage scenarios must be avoided, not hard ones selected.

### 2b. Spanning Driving Reward (SDR)

Reformulate PDMS/EPDMS reward with a **focal-loss style transformation** to amplify differences between suboptimal and optimal behaviors:

**Standard PDMS structure:**
$$\text{PDMS}=\prod_{c\in C}c\times\frac{\sum_{m\in M}w_{m}\cdot m}{\sum_{m\in M}w_{m}}$$
where $C=\{\text{NC, DAC}\}$, $M=\{\text{EP, TTC, C}\}$, $w_m=\{5,5,2\}$.

**SDR reformulation:**
$$R_{\text{span}}=\prod_{c\in C}c\cdot\frac{\sum_{m\in M}w^{\prime}_{m}\cdot(1-(1-m)^{\gamma_{m}})}{\sum_{m\in M}w^{\prime}_{m}}$$

The focal term $(1-(1-m)^{\gamma_m})$ compresses high-scoring metrics (reducing ceiling effect) and amplifies differences in low-scoring metrics (sharpening gradients where improvement is needed most). EPDMS extends $C$ with $\{\text{DDC, TLC}\}$ and $M$ with $\{\text{LK, EC}\}$.

## Results

### NAVSIM-v1 (Table 1)

| Method | Base Model | Sensors | Traj Type | NC | DAC | EP | TTC | C | PDMS |
|--------|-----------|---------|-----------|-----|-----|-----|-----|---|------|
| Human GT | — | — | — | 100.0 | 100.0 | 87.5 | 100.0 | 99.9 | 94.8 |
| WoTE | — | 3xC+L | Continuous | 98.5 | 96.8 | 81.9 | 94.4 | 99.9 | 88.3 |
| ReCogDrive | InternVL2-8B | 3xC | Continuous | 98.2 | 97.8 | 83.5 | 95.2 | 100.0 | 89.6 |
| DriveVLA-W0 | Emu-3-8B | 1xC | Continuous | 98.7 | 99.1 | 93.3 | 95.3 | 99.3 | 90.2 |
| AutoVLA | Qwen2.5-VL-3B | 3xC | Discrete Action | 98.4 | 95.6 | 81.9 | 98.0 | 99.9 | 89.1 |
| AutoVLA† (N=6) | Qwen2.5-VL-3B | 3xC | Discrete Action | 99.1 | 98.8 | 87.9 | 97.2 | 100.0 | 92.1 |
| AdaThinkDrive | InternVL3-8B | 1xC | Text Waypoint | 98.4 | 97.8 | 84.4 | 95.2 | 100.0 | 90.3 |
| AdaThinkDrive† (N=6) | InternVL3-8B | 1xC | Text Waypoint | 99.1 | 98.8 | 87.9 | 95.2 | 100.0 | 93.0 |
| **Curious-VLA** | **Qwen2.5-VL-3B** | **1xC** | **Text Waypoint** | **98.4** | **96.9** | **88.5** | **97.9** | **98.1** | **90.3** |
| **Curious-VLA† (N=6)** | **Qwen2.5-VL-3B** | **1xC** | **Text Waypoint** | **99.5** | **99.0** | **91.8** | **99.3** | **98.4** | **94.8** |

**Notable**: 90.3 PDMS with a 3B model matches AdaThinkDrive (8B). BoN-6 94.8 matches human GT and surpasses AdaThinkDrive† (93.0) by +1.8 PDMS.

### NAVSIM-v2 (Table 2)

| Method | NC | DAC | DDC | TLC | EP | TTC | LK | C | EC | EPDMS |
|--------|----|-----|-----|-----|-----|-----|-----|----|----|-------|
| DiffusionDrive | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | 84.5 |
| ReCogDrive | 98.3 | 95.2 | 99.5 | 99.8 | 87.1 | 97.5 | 96.6 | 98.3 | 86.5 | 83.6 |
| **Curious-VLA** | **98.4** | **96.9** | **99.2** | **99.8** | **88.5** | **97.9** | **96.9** | **98.1** | **81.5** | **85.3** |

**Note**: EC (Extended Comfort) 81.5 is lower than ReCogDrive (86.5) — exploration creates more aggressive maneuvers that reduce comfort.

### nuScenes Open-Loop (Table 3)

| Method | ST-P3 L2 (↓) | ST-P3 Collision (↓) | UniAD L2 (↓) | UniAD Collision (↓) |
|--------|-------------|---------------------|-------------|---------------------|
| EMMA | 0.32 | — | — | — |
| OpenDriveVLA | 0.33 | 0.10 | 0.67 | 0.30 |
| AutoVLA | 0.48 | 0.13 | 0.86 | 0.35 |
| Impromptu VLA | 0.33 | 0.13 | 0.67 | 0.38 |
| **Curious-VLA** | **0.31** | **0.10** | **0.60** | **0.33** |

### FTE Ablation (Table 5)

| DE | CoT | SN | NC | DAC | EP | TTC | C | PDMS |
|----|-----|-----|-----|-----|-----|-----|---|------|
| ✗ | ✗ | ✗ | 97.7 | 91.8 | 85.8 | 96.8 | 98.4 | 83.9 |
| ✗ | ✓ | ✗ | 98.2 | 93.2 | 85.8 | 97.3 | 98.4 | 85.6 |
| ✓ | ✓ | ✗ | 98.0 | 93.0 | 85.9 | 97.2 | 98.4 | **85.2** (worse than CoT alone) |
| ✗ | ✓ | ✓ | 98.2 | 94.3 | 86.7 | 97.3 | 98.4 | 86.9 |
| ✓ | ✓ | ✓ | 98.3 | 95.1 | 86.5 | 97.6 | 98.3 | **87.6** |

**Key takeaway**: DE alone hurts (85.2 < 85.6 baseline); SN is the critical catalyst. Only DE+CoT+SN together achieve the best SFT policy (87.6 PDMS).

### RL Ablation (Table 6)

| Sampling Strategy | SDR | NC | DAC | EP | TTC | C | PDMS |
|------------------|-----|-----|-----|-----|-----|---|------|
| Human Difficulty | ✗ | 73.9 | 43.7 | 94.9 | 70.3 | 97.1 | **35.2 (collapse)** |
| Reject Unimodal | ✗ | 98.4 | 96.0 | 87.0 | 97.8 | 98.4 | 88.8 |
| ADAS (1x) | ✗ | 98.2 | 96.3 | 88.6 | 97.6 | 98.2 | 89.6 |
| ADAS (3x) | ✗ | 98.4 | 96.8 | 88.5 | 97.8 | 98.1 | 90.1 |
| **ADAS (3x)** | **✓** | **98.4** | **96.9** | **88.5** | **97.9** | **98.1** | **90.3** |

![[x5 8.png|Qualitative BEV comparison showing Curious-VLA diverse yet safe trajectories]]

*Figure 4: Qualitative BEV+camera comparison. Curious-VLA produces diverse yet safe trajectories vs. collapsed single-mode outputs from baselines.*

## Implementation Details

| Setting | Value |
|---------|-------|
| Base model | Qwen2.5-VL-3B |
| Hardware | 8x NVIDIA H100 |
| SFT epochs | 3 (align) + 3 (fine-tune) |
| SFT batch size | 128 |
| RL steps | 130 total, 3 outer-loops |
| RL rollouts (G) | 8 |
| RL batch size | 256 |
| SFT data | 142k (FTE-expanded from 103k NavTrain) |
| Sensors | 1x front camera only |
| Trajectory type | Text waypoint (8-step, 4s horizon) |

## Limitations

1. **Comfort regression**: NAVSIM-v2 EC = 81.5 vs. ReCogDrive's 86.5 — exploration trades some comfort for trajectory diversity
2. **Comparison scope**: NAVSIM-v2 table does not include Senna-2 (86.6 EPDMS) or DriveFine (87.1 EPDMS on old scorer) — cannot determine if 85.3 represents true SOTA
3. **ADAS overhead**: $M \gg G$ offline rollouts per scenario at each outer-loop adds substantial compute vs. vanilla GRPO
4. **Single camera**: 1xC constraint limits perception of side/rear agents; DriveVLA-W0 achieves 90.2 with 1xC but uses a different (non-text-waypoint) paradigm
5. **No closed-loop evaluation**: open-loop NAVSIM may not capture all benefits of diversity (reaction to other agents not modeled)
6. **nuScenes RL reward differs**: uses ADE-based reward rather than PDMS — reward design is task-specific, not universal

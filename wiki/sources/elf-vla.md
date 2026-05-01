---
title: "ELF-VLA: Explicit Learning from Failures for Autonomous Driving"
type: source-summary
sources: [raw/papers/Unleashing VLA Potentials in Autonomous Driving via Explicit Learning from Failures.md]
related: [concepts/rl-for-ad.md, concepts/chain-of-thought-for-ad.md, concepts/vlm-domain-adaptation.md, concepts/navsim-benchmark.md, sources/curious-vla.md, sources/adathinkdrive.md, sources/nord.md]
created: 2026-04-28
updated: 2026-04-28
confidence: high
---

**arXiv**: https://arxiv.org/html/2603.01063v1  
**Authors**: Yuechen Luo, Qimao Chen, Fang Li, Shaoqing Xu, Jiaxin Liu, Ziying Song, Zhi-xin Yang, Fuxi Wen  
**Affiliation**: Tsinghua University, University of Macau, Beijing Jiaotong University

## One-Line Summary

ELF-VLA attacks the "persistent failure" regime of VLA RL, where all GRPO rollouts receive near-zero driving score, by having a Qwen3-VL-32B teacher produce structured failure diagnostics and using the student to generate a corrected feedback-guided refinement that is re-injected into the GRPO batch, reaching 91.0 PDMS on NAVSIM-v1 and 87.1 EPDMS on NAVSIM-v2 with InternVL3-8B.

## Core Problem: Persistent Failures

The paper starts from a specific GRPO failure mode: after SFT, rare long-tail scenes can produce rollout groups where every sampled trajectory fails. A scalar reward such as PDMS says that the group failed, but not whether the root cause was:

- high-level planning error in the meta-action,
- flawed "think" reasoning about critical objects,
- low-level trajectory execution error,
- safety failure such as collision or drivable-area violation,
- efficiency failure such as poor progress.

When every rollout is bad, group-relative advantages provide little actionable direction. ELF-VLA turns those zero-value cases into targeted correction examples.

![[introv6.png|Figure 1: General RL plateau vs. ELF-VLA failure-feedback refinement loop.]]

*Figure 1: Standard RL can remain trapped when all rollouts fail; ELF-VLA uses teacher feedback to force a corrected re-rollout.*

## Method Overview

![[mainv4.png|Figure 2: ELF-VLA training pipeline.]]

*Figure 2: Three stages: driving QA pretraining, mixed base/feedback SFT, and RL with online teacher feedback.*

### Inputs: Base vs. Feedback

The model is trained as both a generator and a refiner.

**Base input** includes:

- front-view image,
- high-level navigation command,
- ego velocity and acceleration,
- historical trajectory from the last three frames at 2 Hz.

**Feedback input** appends the original model response and either positive rule feedback or teacher diagnostics:

$$
q_{fb}=\begin{cases}<q_{base},o_{c},f^{rule}>&\text{if }o_c,\\
<q_{base},o_{w},f^{teacher}>&\text{if }o_w.
\end{cases}
$$

Wrong responses are those below a PDMS threshold $s$. The teacher receives the base input, the wrong trajectory, and the GT trajectory, then returns structured feedback with:

- Meta Action Analysis,
- Think Process Analysis,
- Safety Failure Analysis,
- Efficiency Failure Analysis,
- Actionable Correction split into lateral and longitudinal guidance.

### Two-Stage SFT

Stage 1 pretrains InternVL3-8B on a broad driving QA corpus assembled from DriveLM, LingoQA, ImpromptuVLA, NuScenes-QA, NuInstruct, and OmniDrive, plus NAVSIM CoT-style reasoning tasks.

Stage 2 trains on a mixed dataset:

$$
\mathcal{D}=\{(q_{base},o_{gt}),(q_{fb},o_{gt})\}
$$

with likelihood loss:

$$
\mathcal{L}_{\text{SFT}}=\mathbb{E}_{(q,o)\sim\mathcal{D}}\big[-\log\pi_{\theta}(o\mid q)\big].
$$

This gives the model a learned interface for using feedback before RL starts.

### GRPO with Failure Feedback

![[methodv3.png|Figure 3: GRPO with teacher feedback and policy shaping.]]

*Figure 3: Initial rollouts are scored, teacher feedback is generated for failures, the model samples refinements, and the best refinement is injected into the final optimization batch.*

The rollout procedure is:

1. Sample $n$ base responses from $q^{base}$.
2. Compute trajectory, format, and goal rewards.
3. For responses below threshold $s$, call Qwen3-VL-32B to generate structured feedback.
4. Re-rollout from feedback inputs $q^{fb}$.
5. Select $k$ refined responses whose trajectory reward exceeds the best original rollout.
6. Optimize over the merged original-plus-refined batch of size $n+k$.

The final objective separates original samples and feedback-refined samples:

$$
\mathcal{J}(\theta)=\mathbb{E}_{D_{final}\sim\pi_{\theta_{old}}}\left[\frac{1}{n}\sum_{i=1}^{n}\mathcal{J}_{i}+\frac{1}{k}\sum_{j=1}^{k}\mathcal{J}_{j}^{fb}-\beta\mathbb{D}_{KL}\right]
$$

Original rollouts use clipped PPO-style ratios:

$$
\mathcal{J}_{i}=\min\big(c_i(\theta)A_i,\text{clip}(c_i(\theta),1-\epsilon,1+\epsilon)A_i\big).
$$

Feedback samples are not clipped; they use policy shaping:

$$
\mathcal{J}_{j}^{fb}=f(c_j^{fb}(\theta))A_j^{fb}, \quad f(x)=\frac{x}{x+\gamma}.
$$

The merged reward set is used for advantage normalization:

$$
r_{union}=\{r_j\}_{j=1}^{n}\cup\{r_{j'}^{fb}\}_{j'=1}^{k}
$$

$$
A_i=\frac{r_i-\text{mean}(r_{union})}{\text{std}(r_{union})}, \quad
A_j^{fb}=\frac{r_j^{fb}-\text{mean}(r_{union})}{\text{std}(r_{union})}.
$$

**Why policy shaping matters:** feedback refinements are generated under $q^{fb}$ but optimized under $q^{base}$. Their base-conditioned likelihood can be low, so naive ratios are high-variance. The shaped ratio $x/(x+\gamma)$ gives stable credit to rare but correct refinements.

## Reward Design

The total RL reward is:

$$
r=r_{traj}+r_{fmt}+r_{goal}.
$$

| Component | Definition | Purpose |
|-----------|------------|---------|
| $r_{traj}$ | NAVSIM PDMS in [0, 1] | Closed-loop trajectory quality |
| $r_{fmt}$ | 0.5 for valid `<think>...</think>` and `<answer>...</answer>`, 0.5 for parsable trajectory syntax | Prevent malformed outputs |
| $r_{goal}$ | Piecewise endpoint L1 reward | Encourage endpoint accuracy |

Goal reward:

$$
r_{goal}=\begin{cases}
1&0<dis<2\\
0.8&2\leq dis<4\\
0.6&4\leq dis<6\\
0.4&6\leq dis<10\\
0.2&10\leq dis<15\\
0&dis>15
\end{cases}
$$

## Data Curation

Before RL, the SFT model samples $N$ rollouts per query to estimate mean reward and reward variance. Samples with high mean and low variance are discarded as already-mastered. The final RL set keeps:

- difficult samples: low mean, low variance,
- ambiguous samples: high variance,
- 24k high-value scenarios from the original 85k training entries.

This differs from Curious-VLA's ADAS: ELF-VLA intentionally keeps persistent failures because teacher feedback can turn them into corrected training examples, whereas standard GRPO would not learn from them.

## Results

### NAVSIM-v1 (Table 1)

| Method | Image | Lidar | NC | DAC | TTC | CF | EP | PDMS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Constant Velocity |  |  | 68.0 | 57.8 | 50.0 | 100 | 19.4 | 20.6 |
| Ego Status MLP |  |  | 93.0 | 77.3 | 83.6 | 100 | 62.8 | 65.6 |
| UniAD | yes |  | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| TransFuser | yes | yes | 97.7 | 92.8 | 92.8 | 100 | 84.0 | 84.0 |
| DiffusionDrive | yes | yes | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| WoTE | yes | yes | 98.5 | 96.8 | 94.9 | 99.9 | 81.9 | 88.3 |
| Hydra-NeXt | yes |  | 98.1 | 97.7 | 94.6 | 100 | 81.8 | 88.6 |
| AutoVLA-3B | yes |  | 98.4 | 95.6 | 98.0 | 100 | 81.9 | 89.1 |
| DriveVLA-W0-3B | yes |  | 98.7 | 99.1 | 95.3 | 99.3 | 83.3 | 90.3 |
| GoalFlow | yes | yes | 98.4 | 98.3 | 94.6 | 100 | 85.0 | 90.3 |
| InternVL3-8B-SFT | yes |  | 98.5 | 95.5 | 95.3 | 100 | 81.2 | 87.4 |
| InternVL3-8B-RL | yes |  | 98.5 | 96.7 | 95.4 | 100 | 83.2 | 89.0 |
| **ELF-VLA-8B** | **yes** |  | **98.9** | **98.1** | **96.0** | **100** | **85.3** | **91.0** |

ELF-VLA improves over its own standard-GRPO baseline by +2.0 PDMS and over SFT by +3.6 PDMS.

### NAVSIM-v2 (Table 2)

| Method | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | EPDMS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HydraMDP++ | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | 81.4 |
| DriveSuprim | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 | 83.1 |
| RecogDrive-8B | 98.3 | 95.2 | 99.5 | 99.8 | 87.1 | 97.5 | 96.6 | 98.3 | 86.5 | 83.6 |
| DiffusionDrive | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | 84.5 |
| DriveVLA-W0-3B | 98.5 | 99.1 | 98.0 | 99.7 | 86.4 | 98.1 | 93.2 | 97.9 | 58.9 | 86.1 |
| **ELF-VLA-8B** | **98.9** | **98.1** | **99.4** | **99.8** | **88.5** | **98.4** | **96.9** | **98.3** | **87.2** | **87.1** |

ELF-VLA's EC of 87.2 is strong relative to many exploration-heavy methods, but the paper's v2 table excludes WAM-Diff (89.7), ExploreVLA (88.8), DriveDreamer-Policy (88.7), DreamerAD (87.7), and Vega BoN-6 (89.4).

### Feedback Strategy Ablation (Table 3)

| Method | NC | DAC | TTC | CF | EP | PDMS |
| --- | --- | --- | --- | --- | --- | --- |
| SFT | 98.5 | 95.5 | 95.3 | 100 | 81.3 | 87.4 |
| GRPO | 98.5 | 96.7 | 95.4 | 100 | 83.2 | 89.0 |
| GT-GRPO | 98.1 | 97.1 | 93.5 | 100 | 85.2 | 89.2 |
| Rule-GRPO | 98.3 | 97.3 | 94.8 | 100 | 84.5 | 89.6 |
| **ELF-VLA** | **98.9** | **98.1** | **96.0** | **100** | **85.3** | **91.0** |

GT injection helps less than teacher feedback because GT trajectories are low-likelihood under the VLA policy; rule feedback helps less because it lacks instance-specific diagnosis.

![[rollout_ratio.png|Figure 4: Total-failure ratios during RL.]]

*Figure 4: ELF-VLA reduces total-failure PDMS rate from 2.73% under GRPO to 1.08%, with similar reductions for NC and DAC.*

### High-Level Planning (Table 4)

| Method | Speed Acc. | Path Acc. | Accuracy |
| --- | --- | --- | --- |
| Qwen2.5-VL-7B | 37.8 | 61.3 | 19.1 |
| InternVL3-8B | 40.9 | 58.7 | 20.1 |
| Qwen2.5-VL-32B | 46.6 | 55.3 | 27.6 |
| Qwen2.5-VL-72B | 49.4 | 62.6 | 28.7 |
| SFT | 84.2 | 90.7 | 79.2 |
| GRPO | 84.3 | 90.8 | 79.3 |
| GT-GRPO | 83.5 | 90.5 | 78.4 |
| Rule-GRPO | 84.5 | 91.2 | 79.5 |
| **ELF-VLA** | **85.8** | **92.5** | **80.3** |

The high-level planning result supports the paper's claim that teacher feedback corrects reasoning/meta-action errors, not only waypoint geometry.

![[visual.jpg|Figure 5: Teacher feedback correcting a wrong trajectory.]]

*Figure 5: Example left-turn failure where structured feedback corrects obstacle localization and produces a safer refined trajectory.*

### RL Data Curation (Table 5)

| Num. | NC | DAC | TTC | CF | EP | PDMS |
| --- | --- | --- | --- | --- | --- | --- |
| 85k | 98.5 | 96.8 | 95.3 | 100 | 83.4 | 89.1 |
| 24k random | 98.4 | 96.8 | 95.2 | 100 | 83.1 | 88.9 |
| **24k curated** | **98.9** | **98.1** | **96.0** | **100** | **85.3** | **91.0** |

The full dataset underperforms because simple already-mastered cases dilute the gradient signal.

### Refinement and Policy Shaping (Table 6)

| k | PS | NC | DAC | TTC | CF | EP | PDMS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | yes | 98.5 | 96.7 | 95.4 | 100 | 83.3 | 89.0 |
| 2 | yes | 98.2 | 97.5 | 94.3 | 100 | 84.9 | 89.7 |
| 1 | no | 98.5 | 97.0 | 94.9 | 100 | 83.9 | 89.3 |
| **1** | **yes** | **98.9** | **98.1** | **96.0** | **100** | **85.3** | **91.0** |

One refinement is best; multiple refinements may inject distracting or noisy alternatives. Policy shaping is critical (+1.7 PDMS at k=1).

## Appendix Details

### Prompting and Feedback Figures

![[base_prompt.jpg|Figure 6: Base input prompt for the VLA model.]]

![[prompt1.jpg|Figure 7: Teacher-model feedback prompt.]]

![[feedback.jpg|Figure 8: Rule feedback vs. teacher structured diagnostics.]]

![[meta1.jpg|Figure 9: High-level action labels and GT generation.]]

![[visual1.jpg|Figure 10: Additional trajectory refinement examples.]]

![[visual2.jpg|Figure 11: Additional trajectory refinement examples.]]

### Training Hyperparameters (Table 7)

| Stage | Setting | Value |
| --- | --- | --- |
| Pretrain | Epochs | 2 |
| Pretrain | Batch size | 1 |
| Pretrain | Learning rate | 1e-5 |
| Pretrain | Gradient accumulation | 4 |
| Pretrain | Weight decay | 0.05 |
| Stage 2 SFT | Epochs | 2 |
| Stage 2 SFT | Batch size | 2 |
| Stage 2 SFT | Learning rate | 4e-5 |
| Stage 2 SFT | Gradient accumulation | 2 |
| Stage 3 RL | Epochs | 3 |
| Stage 3 RL | Batch size | 2 |
| Stage 3 RL | Learning rate | 2e-6 |
| Stage 3 RL | Gradient accumulation | 16 |
| Stage 3 RL | Number of generations | 8 |
| Stage 3 RL | Number of iterations | 2 |
| Stage 3 RL | Temperature | 1.2 |
| Stage 3 RL | Threshold s | 0.8 |
| Stage 3 RL | Number of refinements | 1 |
| Stage 3 RL | Policy shaping gamma | 0.1 |

Implementation note: the body text says RL uses 32 NVIDIA H20 GPUs, threshold $s=0.8$, $\gamma=0.1$, and $k=1$. It also reports about 0.1s inference latency with vLLM for CoT+trajectory generation.

### Training Pipeline Ablation (Table 8)

| Model | NC | DAC | TTC | CF | EP | PDMS |
| --- | --- | --- | --- | --- | --- | --- |
| SFT | 98.5 | 93.4 | 95.1 | 100 | 78.8 | 85.3 |
| Pre+SFT | 98.5 | 95.5 | 95.3 | 100 | 81.2 | 87.4 |
| **Pre+SFT+RL** | **98.9** | **98.1** | **96.0** | **100** | **85.3** | **91.0** |

### No-Pretrain Ablation (Table 9)

| Method | NC | DAC | TTC | CF | EP | PDMS |
| --- | --- | --- | --- | --- | --- | --- |
| InternVL3-8B without pretrain | 97.9 | 93.9 | 94.4 | 100 | 82.8 | 86.9 |
| **ELF-VLA-8B without pretrain** | **98.1** | **97.0** | **94.4** | **100** | **86.2** | **90.0** |

This supports the claim that the RL feedback design, not only extra pretraining data, drives most of the gain.

### Feedback Threshold (Table 10)

| Threshold s | NC | DAC | TTC | CF | EP | PDMS |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 98.4 | 97.1 | 94.3 | 100 | 84.7 | 89.4 |
| 0.5 | 98.6 | 97.4 | 95.3 | 100 | 84.9 | 90.1 |
| 0.9 | 98.4 | 97.3 | 94.9 | 100 | 84.8 | 89.8 |
| **0.8** | **98.9** | **98.1** | **96.0** | **100** | **85.3** | **91.0** |

$s=0.8$ is the best tradeoff: low thresholds miss suboptimal but fixable outputs, while $s=0.9$ over-refines already adequate trajectories and adds noise.

## Key Contributions

1. **Failure diagnosis as RL signal**: converts all-failed rollout groups from sparse scalar rewards into actionable teacher critiques.
2. **Feedback-guided refinement**: trains the student to consume diagnostics and produce corrected trajectories, then injects those corrected samples into the GRPO batch.
3. **Policy shaping for low-probability refinements**: stabilizes learning from rare corrected outputs whose base-conditioned likelihood would otherwise be too small.
4. **High-value RL curation**: filters 85k training cases to 24k difficult/ambiguous scenarios, avoiding gradient dilution from easy cases.
5. **High-level planning measurement**: evaluates speed/path meta-action accuracy, showing gains in decision quality as well as trajectory score.

## Limitations

1. **Teacher dependence**: performance is bounded by Qwen3-VL-32B's ability to diagnose errors; incorrect teacher feedback could teach wrong repairs.
2. **Training-time compute**: online teacher calls for failed rollouts add substantial RL cost; the paper reports 32 H20 GPUs for stage 3.
3. **GT trajectory in teacher prompt**: teacher feedback uses ground-truth trajectory during training, so the mechanism is not directly available at inference without another source of corrective signal.
4. **NAVSIM-only experiments**: no nuScenes, WaymoE2E, Bench2Drive, or fully interactive closed-loop validation.
5. **Non-reactive simulator**: NAVSIM cannot evaluate how other agents would respond to the refined ego action.
6. **Comparison scope**: v1 table omits DriveSuprim (93.5), HybridDriveVLA (92.1), FLARE (91.4), DiffusionDriveV2 (91.2), WAM-Diff (91.0), and ExploreVLA (90.4); v2 table omits WAM-Diff (89.7), ExploreVLA (88.8), DriveDreamer-Policy (88.7), DreamerAD (87.7), and Vega BoN-6 (89.4).

## Relationship to Nearby Methods

| Method | Failure/exploration issue | Intervention | Key distinction |
|--------|---------------------------|--------------|-----------------|
| Curious-VLA | Narrow policy causes advantage collapse | FTE + ADAS + SDR | Avoids zero-variance cases by restoring diversity |
| NoRD | Weak SFT causes high-variance difficulty bias | Dr. GRPO without std normalization | Fixes optimizer normalization, no CoT |
| ExploreVLA | Safe novelty under-explored | World-model entropy bonus gated by PDMS | Rewards safe OOD exploration |
| **ELF-VLA** | Persistent failures where all rollouts are bad | Teacher diagnosis + corrected re-rollout injection | Converts failures into supervised-like high-advantage refinements |

ELF-VLA is complementary to diversity methods: Curious-VLA asks the policy to explore better; ELF-VLA asks a teacher to explain why exploration failed and converts that explanation into a corrected sample.

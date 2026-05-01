---
title: "SpanVLA: Efficient Action Bridging and Learning from Negative-Recovery Samples for Vision-Language-Action Model"
type: source-summary
sources: [raw/papers/SpanVLA_ Efficient Action Bridging and Learning from Negative-Recovery Samples for Vision-Language-Action Model.md]
related: [concepts/diffusion-planner.md, concepts/rl-for-ad.md, concepts/chain-of-thought-for-ad.md, concepts/vlm-domain-adaptation.md, concepts/navsim-benchmark.md, sources/autovla.md, sources/alpamayo-r1.md, sources/recogdrive.md, sources/dynvla.md]
created: 2026-04-28
updated: 2026-04-28
confidence: high
---

**arXiv**: https://arxiv.org/html/2604.19710v1
**Project**: https://spanvla.github.io/
**Authors**: Zewei Zhou, Ruining Yang, Xuewei (Tony) Qi, Yiluan Guo, Sherry X. Chen, Tao Feng, Kateryna Pistunova, Yishan Shen, Lili Su, Jiaqi Ma
**Affiliations**: UCLA, Motional, Northeastern University

## One-Line Summary

SpanVLA combines an autoregressive VLM reasoning backbone with a sparse-KV-cache flow-matching action expert initialized from historical trajectory, then applies GRPO over positive, negative, and recovery samples to reach **90.3 PDMS on NAVSIM-v1**, **86.4 EPDMS on NAVSIM-v2 navtest**, and **40.1 EPDMS on NAVSIM-v2 navhard**.

## Source Integrity Note

The local clipped markdown ends immediately after the Table 1 caption. Main text after that point, Tables 2-5, supplementary sections, and supplementary figure captions were recovered from the arXiv HTML source linked above. Local extracted assets included Figures 1-4; missing Figures 5-7 and S1-S6 were downloaded from arXiv into `raw/assets/spanvla-*`.

## Core Motivation

SpanVLA targets two VLA bottlenecks:

- **Autoregressive action latency**: action-token decoding grows linearly with trajectory length and is too slow for high-frequency control.
- **Positive-only imitation fragility**: standard SFT/RFT mostly learns from expert positives, ignoring real negative behaviors and recovery corrections collected during testing.

The paper's answer is to keep autoregressive reasoning in the VLM, but delegate continuous trajectory generation to a flow-matching action expert conditioned on sparse VLM KV-cache and historical trajectory initialization.

![[x1 27.png|Figure 1: SpanVLA framework.]]

*Figure 1: SpanVLA uses a VLM backbone for adaptive reasoning, a bridge over sparse KV-cache, a flow-matching action expert, SFT, and GRPO RFT over positive, negative, and recovery samples.*

## Architecture

### VLM Backbone

The backbone is Qwen2.5-VL-3B. Inputs include multi-frame, multi-view images and text prompts containing high-level instruction, ego acceleration, velocity, and history trajectory.

Default vision setting:

- 3 camera views at inference/training default: front, front-left, front-right.
- 4 history frames per camera sampled at 2 Hz.
- The system can extend to more camera streams; mReasoning itself has 8 surrounding cameras.

The VLM generates structured reasoning and, during training, discrete action tokens from the AutoVLA physical action codebook:

$$
[\mathcal{T}_{Reason}, A_{token}] = VLM(\mathcal{V}^t, \mathcal{T}^t)
$$

At inference, once the special action-generation token appears, the VLM stops reasoning and hands off to the action expert.

### Efficient Action Bridging

![[x2 25.png|Figure 2: Efficient action bridging.]]

*Figure 2: Sparse VLM layers provide KV-cache to a lightweight bridge. The flow-matching action expert starts from historical trajectory initialization instead of Gaussian noise.*

The action bridge processes sparse VLM KV-cache layers from the sequence `[vision, text, reasoning]`. The action expert predicts a vector field conditioned on:

- sparse VLM KV-cache,
- historical trajectory embedding,
- flow-matching time embedding.

Unlike ReCogDrive/Alpamayo-style action generation from Gaussian noise, SpanVLA learns:

$$
\mathbf{a}_{his} \rightarrow \mathbf{a}_{future}
$$

with optimal-transport conditional flow matching:

$$
\mathbf{a}_{\tau}=\tau\mathbf{a}+(1-\tau)\mathbf{a}_{his}
$$

$$
\mathbf{a}_{t+\Delta\tau}=\mathbf{a}_{t}+\Delta\tau f_{\theta}(\mathbf{a}_{t},\tau,\mathbf{c}_{vlm})
$$

Training injects Gaussian noise into historical trajectory embeddings for robustness. The VLM backbone and bridging module are stop-gradiented when training the action bridge, preserving the VLM representation.

## mReasoning Dataset

![[x3 25.png|Figure 3: mReasoning distribution and negative-recovery examples.]]

*Figure 3: mReasoning emphasizes complex scenarios and includes real negative-recovery pairs.*

mReasoning is an in-house real-world reasoning dataset:

- **30K positive reasoning samples** from expert driving logs.
- **3K negative samples**: suboptimal real-world ego trajectories.
- **3K recovery samples**: expert corrective actions for the same scenario families.
- Regions: Las Vegas, Boston, Pittsburgh, Singapore.
- Scenario types: ego lane changes, lane bias, VRUs, construction zones, stop signs, cut-ins, lead-vehicle braking.
- History horizon: 1.5 s; future horizon: 5 s at 2 Hz.
- Sensor setting in mReasoning: 8 surrounding cameras plus object annotations and HD maps.

The CoT annotation pipeline uses Gemini-3-Pro. It identifies critical elements, filters irrelevant factors, selects longitudinal/lateral actions, then compresses the result into a concise reasoning trace. A human check over 250 samples reports **80.2%** overall accuracy across critical components, driving decision, and reasoning trace.

## SFT

SpanVLA uses a two-stage SFT recipe:

1. Train the VLM backbone to output reasoning plus discrete action tokens.
2. Freeze the VLM backbone and fine-tune the action expert / bridge without action-token generation.

The VLM SFT combines language-token and action-token losses:

$$
\mathcal{L}_{LM}=-\frac{1}{N}\sum_i \log p_{\theta}(o_i|o_{<i},\mathcal{V}^t,\mathcal{T}^t)
$$

$$
\mathcal{L}_{Action}=-\frac{1}{T}\sum_{i=L+1}^{L+T}\log p_{\theta}(o_i|o_{<i},\mathcal{V}^t,\mathcal{T}^t)
$$

The action bridge uses conditional flow-matching loss:

$$
\mathcal{L}_{FM}=\mathbb{E}\|f_{\theta}(\mathbf{a}_{\tau},\tau,\mathbf{c}_{vlm})-\mathbf{v}_{\tau}(\mathbf{a}_{\tau})\|^2
$$

where $\mathbf{v}_{\tau}=\mathbf{a}-\mathbf{a}_{his}$.

## RFT with Negative-Recovery Samples

SpanVLA applies GRPO to the VLM backbone with LoRA adapters. Implementation details:

- RFT learning rate: $3\times10^{-5}$.
- Group sample size: 64.
- Single policy update per step, so their simplified implementation does not require clipping or maintaining an old policy.
- KL regularization keeps the policy close to the SFT reference.

The reward combines driving quality, negative avoidance, recovery imitation, and CoT length/alignment:

$$
r=r_{Driving}-w_N r_{Negative}+w_R r_{Recovery}-\lambda_C r_{CoT}
$$

where:

- $r_{Driving}$ is PDMS for NAVSIM-v1 and EPDMS for NAVSIM-v2.
- $r_{Negative}$ penalizes trajectories close to negative real-world behavior.
- $r_{Recovery}$ rewards trajectories close to expert recovery behavior.
- $r_{CoT}$ penalizes long reasoning and action-reasoning inconsistency.
- Negative/recovery shaping is bounded by an L2/ADE proximity threshold to avoid unbounded repulsion or over-imitation.

RFT schedule:

- 2K positive warm-up samples.
- 4K mixed samples.
- Best recipe: 3K positive + 0.5K negative + 0.5K recovery in the mixed stage.

## Main Results

### Table 1: NAVSIM-v1 navtest

| Method | Cam | LiDAR | PDMS | NC | DAC | EP | TTC | Comfort |
|---|---|---|---:|---:|---:|---:|---:|---:|
| TransFuser | Yes | Yes | 84.0 | 97.8 | 92.6 | 78.9 | 92.9 | 100.0 |
| DRAMA | Yes | Yes | 86.9 | 98.2 | 95.2 | 81.3 | 94.2 | 100.0 |
| Hydra-MDP | Yes | Yes | 86.5 | 98.3 | 96.0 | 78.7 | 94.6 | 100.0 |
| DiffusionDrive | Yes | Yes | 88.1 | 98.2 | 96.2 | 82.2 | 94.7 | 100.0 |
| WoTE | Yes | Yes | 88.3 | 98.5 | 96.8 | 81.9 | 94.4 | 99.9 |
| ReCogDrive | Yes | No | 89.6 | 98.2 | 97.8 | 83.5 | 95.2 | 99.8 |
| DriveVLA-W0 | Yes | No | 90.2 | 98.7 | 99.1 | 83.3 | 95.3 | 99.3 |
| AutoVLA | Yes | No | 89.1 | 98.4 | 95.6 | 81.9 | 98.0 | 99.9 |
| SpanVLA (One-shot) | Yes | No | 82.1 | 97.5 | 90.8 | 76.9 | 93.7 | 99.5 |
| **SpanVLA (Post-RFT)** | **Yes** | **No** | **90.3** | **99.1** | **97.1** | **86.3** | **95.2** | **100.0** |

**Wiki positioning:** 90.3 PDMS is competitive with WAM-Flow/AdaThinkDrive/Curious-VLA but below DriveSuprim 93.5, HybridDriveVLA 92.1, DynVLA/Reasoning-VLA 91.7, FLARE 91.4, DiffusionDriveV2 91.2, and WAM-Diff/ELF-VLA/DualDriveVLA 91.0.

### Table 2: NAVSIM-v2 navtest

| Method | EPDMS | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TransFuser | 76.7 | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 |
| DiffusionDrive | 84.5 | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 |
| Hydra-MDP++ | 81.4 | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 |
| DriveSuprim | 83.1 | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 |
| ARTEMIS | 83.1 | 98.3 | 95.1 | 98.6 | 99.8 | 81.5 | 97.4 | 96.5 | 98.3 | - |
| DriveVLA-W0 | 86.1 | 98.5 | 99.1 | 98.0 | 99.7 | 86.4 | 98.1 | 93.2 | 97.9 | 58.9 |
| SpanVLA (One-shot) | 79.4 | 96.4 | 89.4 | 97.8 | 99.8 | 87.4 | 95.8 | 94.5 | 98.2 | 81.6 |
| **SpanVLA (Post-RFT)** | **86.4** | **98.8** | **96.7** | **99.2** | **99.8** | **86.3** | **97.7** | **95.0** | **98.3** | **85.1** |

**Wiki positioning:** 86.4 EPDMS is below WAM-Diff 89.7, Vega BoN-6 89.4, ExploreVLA 88.8, DriveDreamer-Policy 88.7, DreamerAD 87.7, DriveSuprim/ELF-VLA 87.1, and Vega single 86.9 in the wiki.

### Table 3: NAVSIM-v2 navhard

| Method | Stage | EPDMS | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LTF | 1 | 23.1 | 96.2 | 79.6 | 99.1 | 99.6 | 84.1 | 95.1 | 94.2 | 97.6 | 79.1 |
| LTF | 2 | 23.1 | 77.8 | 70.2 | 84.3 | 98.1 | 85.1 | 75.7 | 45.4 | 95.8 | 76.0 |
| RAP | 1 | 36.9 | 97.1 | 94.4 | 98.8 | 99.8 | 83.9 | 96.9 | 94.7 | 96.4 | 66.2 |
| RAP | 2 | 36.9 | 83.2 | 83.9 | 87.4 | 98.0 | 86.9 | 80.4 | 52.3 | 95.2 | 52.4 |
| **SpanVLA** | **1** | **40.1** | **98.4** | **94.3** | **97.8** | **99.9** | **85.7** | **97.2** | **94.2** | **97.6** | **72.1** |
| **SpanVLA** | **2** | **40.1** | **86.9** | **84.3** | **87.1** | **98.2** | **85.5** | **82.7** | **62.3** | **96.8** | **67.4** |

The table reports the same headline EPDMS for both stages while submetrics differ, so the paper likely follows navhard's two-stage reporting convention where final score aggregates the two stages. Treat direct comparison with NAVSIM-v2 navtest cautiously.

## Efficiency and Action Bridging

### Table 4: Action Policy Runtime on NAVSIM-v1

| Method | Action count | PDMS | VLM encoding + prefill | Reasoning generation | Trajectory generation | Total |
|---|---:|---:|---:|---:|---:|---:|
| AutoVLA | 10 | 89.1 | 0.09s | 0.76s (23 tokens) | 0.40s (12 tokens) | 1.25s |
| AutoVLA | 50 | - | 0.09s | 0.76s (23 tokens) | 1.72s (52 tokens) | 2.57s |
| SpanVLA (L1 Head) | 10 | 85.1 | 0.09s | 0.50s (15 tokens) | 0.02s | 0.61s |
| SpanVLA (L1 Head) | 50 | - | 0.09s | 0.50s (15 tokens) | 0.02s | 0.61s |
| **SpanVLA (FM)** | **10** | **90.3** | **0.09s** | **0.50s (15 tokens)** | **0.08s (5 steps)** | **0.67s (-46%)** |
| **SpanVLA (FM)** | **50** | **-** | **0.09s** | **0.50s (15 tokens)** | **0.08s (5 steps)** | **0.67s (-74%)** |

Flow matching is slightly slower than the L1 head but far stronger in PDMS. Crucially, trajectory-generation time does not grow with waypoint count the way AR action decoding does.

### Table 5: Bridging Layer and Historical Initialization Ablation

| Metric | Full Caching | Last Layer | Interval 4 | Interval 2 without HI | Interval 2 |
|---|---:|---:|---:|---:|---:|
| PDMS | 88.1 | 79.3 | 82.2 | 86.4 | **90.3** |
| Trajectory generation | 0.18s | 0.01s | 0.05s | 0.08s | **0.08s** |

Historical initialization contributes +3.9 PDMS over interval-2 without HI. Sparse interval-2 features recover stronger performance than last-layer-only or interval-4 while remaining faster than full caching.

## RFT and Qualitative Figures

![[x4 23.png|Figure 4: RFT results and positive-sample qualitative comparison.]]

*Figure 4: RFT with positive + negative + recovery samples reaches 90.3 PDMS, compared with SFT-only 82.1, positive-only RFT 86.9, and positive+negative RFT 89.0. Qualitatively, RFT can remove unnecessary slow-thinking in a simple roundabout case.*

![[spanvla-x5.png|Figure 5: RFT data-recipe comparison.]]

*Figure 5: Best 6K RFT recipe is 2K positive warm-up + 3K positive + 0.5K negative + 0.5K recovery. Warm-up improves over direct mixed training.*

![[spanvla-x6.png|Figure 6: Negative-sample qualitative comparison.]]

*Figure 6: Negative samples help avoid overly conservative stopping or hesitation; after RFT, SpanVLA merges earlier in construction narrowing and changes lanes more decisively.*

![[spanvla-x7.png|Figure 7: Recovery-sample qualitative comparison.]]

*Figure 7: Recovery samples teach corrective behavior, such as completing turns under construction constraints and borrowing an adjacent lane temporarily to avoid obstacles.*

## Supplementary Figures

![[spanvla-xs1.png|Figure S1: Complex mReasoning positive scenarios.]]

*Figure S1: Cut-in and intersection examples from mReasoning.*

![[spanvla-xs2.png|Figure S2: Reasoning data examples.]]

*Figure S2: Positive reasoning samples, including an oncoming vehicle encroaching into ego's lane and a pedestrian gesturing the ego vehicle to proceed.*

![[spanvla-xs3-bridge-mode.png|Figure S3: Bridge mode comparison.]]

*Figure S3: Action-space bridge mode performs better than latent-space and sequential alternatives.*

![[spanvla-xs4.png|Figure S4: Negative/recovery reward ablation.]]

*Figure S4: Ablates the negative/recovery term, its weight, and the L2/ADE activation threshold.*

![[spanvla-xs5.png|Figure S5: Additional negative-sample qualitative results.]]

*Figure S5: More cases where negative samples help avoid conservative stopping and proceed through turns or straight driving.*

![[spanvla-xs6.png|Figure S6: Additional recovery-sample qualitative results.]]

*Figure S6: More recovery cases, including construction-zone lane constraints and yellow-to-red light stopping.*

## Relationships

| Related method | Relationship |
|---|---|
| AutoVLA | SpanVLA reuses adaptive reasoning and the 2048 action-token codebook for VLM-side training, but replaces AR action generation at inference with a flow-matching expert. |
| Alpamayo-R1 | Both use a flow-matching action expert conditioned on VLM features. SpanVLA uses sparse KV-cache and historical initialization; AR1 uses a unicycle-dynamics action representation and reports much lower optimized token latency. |
| ReCogDrive | Both bridge VLM features to continuous planners. SpanVLA replaces diffusion from noise with FM from history and uses negative-recovery RFT. |
| ELF-VLA | Both target failure/long-tail robustness during RL. ELF-VLA injects teacher-corrected feedback rollouts; SpanVLA uses real negative trajectories and expert recoveries with shaped rewards. |
| DynVLA | Both keep explicit reasoning before action. DynVLA reasons through compact dynamics tokens; SpanVLA reasons in text but decodes action through a continuous FM expert. |

## Limitations

- **Runtime still not deployment-ready**: current runtime is 1.5 Hz. The paper notes no hardware-level or deployment acceleration has been applied; Alpamayo reports 1.75 ms/token under optimized deployment, whereas SpanVLA is 33 ms/token.
- **mReasoning is proprietary/in-house**: the 30K reasoning samples and 3K+3K negative-recovery pairs are not established public benchmarks.
- **Comparison scope is incomplete**: NAVSIM-v1 table omits DriveSuprim, HybridDriveVLA, DynVLA, FLARE, DiffusionDriveV2, WAM-Diff, ELF-VLA, ExploreVLA, and DriveFine.
- **v2 SOTA claim is weak in wiki context**: SpanVLA's 86.4 EPDMS is below several existing wiki entries, although it beats the limited VLA comparison set in its own table.
- **Reward shaping remains hand-designed**: negative/recovery L2 thresholds and weights are sensitive; the paper identifies better reward functions for negative-recovery data as future work.
- **Action-reasoning alignment is rule-based**: keyword and geometric heuristics may miss nuanced maneuvers or incorrectly penalize valid reasoning in rare scenarios.


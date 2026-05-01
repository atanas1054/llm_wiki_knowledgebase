---
title: "DynVLA: Learning World Dynamics for Action Reasoning in Autonomous Driving"
type: source-summary
sources: [raw/papers/DynVLA_ Learning World Dynamics for Action Reasoning in Autonomous Driving.md]
related: [concepts/world-model-for-ad.md, concepts/chain-of-thought-for-ad.md, concepts/rl-for-ad.md, concepts/navsim-benchmark.md, concepts/bench2drive.md, sources/drivevla-w0.md, sources/futuresightdrive.md, sources/explorevla.md]
created: 2026-04-28
updated: 2026-04-28
confidence: high
---

**arXiv**: https://arxiv.org/html/2603.11041v1
**Venue**: ICML / Machine Learning
**Authors**: Shuyao Shang, Bing Zhan, Yunfei Yan, Yuqi Wang, Yingyan Li, Yasong An, Xiaoman Wang, Jierui Liu, Lu Hou, Lue Fan, Zhaoxiang Zhang, Tieniu Tan

## One-Line Summary

DynVLA introduces **Dynamics CoT**: instead of generating text reasoning or future pixels, the VLA first generates compact discrete tokens representing ego-centric and environment-centric world dynamics, then generates action tokens, achieving **91.7 PDMS on NAVSIM-v1** and **88.34 DS / 72.73 SR on Bench2Drive** with much lower reasoning latency than textual or visual CoT.

## Motivation: CoT Without Text or Pixel Redundancy

The paper frames existing AD CoT into three modes:

- **Textual CoT**: interpretable but weak at fine-grained spatiotemporal relationships and high-latency due to long language traces.
- **Visual CoT**: richer spatially but inefficient because future image generation must model irrelevant background and texture.
- **Dynamics CoT**: compact transition tokens that encode the future evolution relevant to planning.

![[x1 26.png|Figure 1: Textual CoT vs. Visual CoT vs. Dynamics CoT.]]

*Figure 1: Dynamics CoT compresses future dynamics into a small number of tokens instead of long text traces or dense visual outputs.*

## Method Overview

![[x2 24.png|Figure 2: DynVLA architecture and Dynamics Tokenizer.]]

*Figure 2: Adjacent observations are encoded into ego-centric and environment-centric VQ dynamics tokens; tokens are decoded to future image/BEV and then used as CoT before action generation.*

![[x3 24.png|Figure 3: DynVLA training pipeline.]]

*Figure 3: Train Dynamics Tokenizer, SFT on dynamics-token-before-action sequences, then RFT with trajectory-level reward and KL regularization.*

DynVLA has three stages:

1. **Dynamics Tokenizer** learns discrete dynamics tokens from adjacent frames.
2. **Dynamics CoT SFT** trains the VLA to output dynamics tokens before action tokens.
3. **Dynamics CoT RFT** applies GRPO using PDMS plus format reward.

## Dynamics Tokenizer

### Decoupled Dynamics

The tokenizer factorizes dynamics into:

- **ego-centric dynamics**: ego vehicle motion and viewpoint transformation,
- **environment-centric dynamics**: changes from other traffic participants and scene evolution.

Input images $O_t$ and $O_{t+1}$ are patchified, encoded with a Transformer dynamics encoder, and queried by two learnable query sets:

$$
(e^{ego}_t,e^{env}_t)=E_{dyn}(\mathbf{x}_t,\mathbf{x}_{t+1};Q_{ego},Q_{env})
$$

Each branch uses a separate VQ codebook:

$$
\mathcal{D}_t=[\mathcal{D}^{ego}_t,\mathcal{D}^{env}_t].
$$

### Action-Based Regularization

Pure reconstruction is under-constrained: ego forward motion can be confused with a leading vehicle moving backward. DynVLA adds an action decoder from ego dynamics:

$$
\mathcal{L}_{act-reg}=\left\lVert\hat{\mathbf{a}}_{t\to t+1}-\mathbf{a}_{t\to t+1}\right\rVert_2^2
$$

This forces ego dynamics to encode ego motion instead of letting the branches collapse into ambiguous reconstruction codes.

### Cross-View Consistency

The same dynamics tokens must reconstruct both:

$$
\widehat{O}_{t+1}=D^{img}_{dyn}(\mathbf{x}_t,z_t), \quad
\widehat{BEV}_{t+1}=D^{bev}_{dyn}(\mathbf{b}_t,z_t).
$$

Image reconstruction uses MSE + LPIPS. BEV reconstruction uses cross-entropy. The tokenizer objective is:

$$
\mathcal{L}=\mathcal{L}^{img}_{recon}+\lambda_{bev}\mathcal{L}^{bev}_{recon}+\lambda_{vq}\mathcal{L}_{VQ}+\lambda_{act-reg}\mathcal{L}_{act-reg}.
$$

## Dynamics CoT Sequence

For a $K$-step horizon, future dynamics are extracted as:

$$
\mathcal{D}_{t+k}=E_{dyn}(O_{t+k},O_{t+k+1}), \quad 0\leq k\leq K-1.
$$

The output sequence is:

$$
\mathbf{y}=[\langle BOD\rangle,\mathcal{D}_{t:t+K-1},\langle EOD\rangle,\langle BOA\rangle,\mathcal{A}_{t:t+N-1},\langle EOA\rangle].
$$

Actions are encoded with the FAST tokenizer. SFT optimizes:

$$
\mathcal{L}_{SFT}=\mathcal{L}_{dyn}+\lambda_{act}\mathcal{L}_{act}.
$$

This forces a causal generation order: reason about dynamics first, then act.

## RFT / GRPO

Reward:

$$
r=r_{traj}+\lambda_{fmt}r_{fmt}
$$

where $r_{traj}$ is NAVSIM PDMS and $r_{fmt}$ enforces the required Dynamics CoT token organization.

GRPO uses group-normalized advantages:

$$
\hat{A}_i=\frac{r_i-\mathrm{mean}(\{r_j\}_{j=1}^G)}{\mathrm{std}(\{r_j\}_{j=1}^G)}
$$

with clipped ratios and KL regularization against the SFT reference model.

## Implementation Details

| Component | Setting |
|-----------|---------|
| Base model | EMU3 |
| Sensor input | current front-view image + front-view image from 1s earlier |
| Dynamics tokens | 8 total: 4 ego + 4 environment |
| Dynamics horizon | K=2, 2 seconds |
| Dynamics sequence length | 16 tokens |
| Dynamics codebooks | 64 ego + 64 environment = 128 dynamics token types |
| Action tokenizer | FAST, 2048 action token types |
| VQ embedding dim | 32 |
| Dynamics encoder | Transformer, 12 layers, hidden dim 1024 |
| Image/BEV decoders | Transformer, 8 layers each |
| Tokenizer training | 200k steps, 8x L20, batch 32, lr 1e-4 |
| SFT | 4k steps, 8x L20, batch 6, lr 1e-4 |
| RFT | 6k steps, 6x H800, batch 6, grad accumulation 6, lr 2e-6 |
| KL coefficient | 1e-3 |

## Results

### NAVSIM-v1 (Table 1)

| Method | NC | DAC | TTC | C | EP | PDMS |
|---|---|---|---|---|---|---|
| Human | 100.0 | 100.0 | 100.0 | 99.9 | 87.5 | 94.8 |
| VADv2 | 97.2 | 89.1 | 91.6 | 100 | 76.0 | 80.9 |
| UniAD | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| TransFuser | 97.7 | 92.8 | 92.8 | 100 | 79.2 | 84.0 |
| PARA-Drive | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 |
| LAW | 96.4 | 95.4 | 88.7 | 99.9 | 81.7 | 84.6 |
| Epona | 97.9 | 95.1 | 93.8 | 99.9 | 80.4 | 86.2 |
| Hydra-MDP | 98.3 | 96.0 | 94.6 | 100 | 78.7 | 86.5 |
| DiffusionDrive | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| WoTE | 98.5 | 96.8 | 94.9 | 99.9 | 81.9 | 88.3 |
| DriveDPO | 98.5 | 98.1 | 94.8 | 99.9 | 84.3 | 90.0 |
| ReCogDrive | 98.2 | 97.8 | 95.2 | 99.8 | 83.5 | 89.6 |
| DriveVLA-W0 | 98.7 | 99.1 | 95.3 | 99.3 | 83.3 | 90.2 |
| AutoVLA | 98.4 | 95.6 | 98.0 | 99.9 | 81.9 | 89.1 |
| AdaThinkDrive | 98.4 | 97.8 | 95.2 | 100 | 84.4 | 90.3 |
| AutoDrive-R2 | 98.3 | 94.4 | 95.6 | 100 | 81.6 | 90.3 |
| FSDrive | 98.2 | 93.8 | 93.3 | 99.9 | 80.1 | 85.1 |
| PWM | 98.6 | 95.9 | 95.4 | 100 | 81.8 | 88.1 |
| **DynVLA** | **98.6** | **98.7** | **95.5** | **100** | **86.8** | **91.7** |

DynVLA's strongest sub-metric gain is ego progress (EP=86.8), consistent with its claim that dynamics reasoning improves foresight and intent-aware planning.

**Comparison caveat:** the NAVSIM table omits DriveSuprim (93.5), HybridDriveVLA (92.1), FLARE (91.4), DiffusionDriveV2 (91.2), WAM-Diff (91.0), ELF-VLA (91.0), and ExploreVLA (90.4). DynVLA is a strong VLA result but not the overall wiki SOTA.

### Bench2Drive (Table 2)

| Method | DS | SR | Mean Multi-Ability |
|---|---:|---:|---:|
| Think2Drive (privileged) | 91.85 | 85.41 | 86.26 |
| PDM-Lite (privileged) | 97.02 | 92.27 | 92.82 |
| AD-MLP | 18.05 | 0.00 | 0.87 |
| TCP | 40.70 | 15.00 | 14.63 |
| VAD | 42.35 | 15.00 | 18.07 |
| UniAD | 45.81 | 16.36 | 15.55 |
| ThinkTwice | 62.44 | 31.23 | 37.17 |
| DriveAdapter | 64.22 | 33.08 | 42.08 |
| Drivetransformer | 63.46 | 35.01 | 38.60 |
| Raw2Drive | 71.36 | 50.24 | 53.34 |
| ORION | 77.74 | 54.62 | 54.72 |
| MindDrive | 78.04 | 55.09 | 56.94 |
| AutoVLA | 78.84 | 57.73 | - |
| TF++ | 84.21 | 67.27 | 64.39 |
| SimLingo | 85.07 | 67.27 | - |
| **DynVLA** | **88.34** | **72.73** | **72.23** |

DynVLA is below LinkVLA (91.01 DS / 74.55 SR in the wiki) but above AutoMoT (87.34 DS) and the non-privileged baselines listed in its own table.

### In-House Dataset (Table 3)

| Model | ADE (m) | Collision Rate (permyriad) |
|---|---:|---:|
| TransFuser | 1.746 | 5.63 |
| DriveVLA-W0 (VQ) | 1.599 | 5.20 |
| DriveVLA-W0 (ViT) | 1.344 | 5.13 |
| **DynVLA** | **1.215** | **4.04** |

This is a 700k-frame in-house dataset. The result supports scaling, but the dataset is not public and the comparison set is limited.

## Analysis and Ablations

### CoT Design and Latency (Table 4)

| CoT Content       | Latency (s) |       NC |      DAC |      TTC |       C |       EP |     PDMS |
| ----------------- | ----------: | -------: | -------: | -------: | ------: | -------: | -------: |
| None              |        0.20 |     98.3 |     93.8 |     94.6 |    99.9 |     79.5 |     85.6 |
| Scene Description |        3.04 |     98.4 |     93.4 |     94.4 |    99.9 |     79.3 |     85.3 |
| Meta Action       |        0.43 |     98.3 |     94.3 |     94.6 |     100 |     79.8 |     86.0 |
| Future Image      |        2.29 |     98.7 |     94.4 |     95.0 |    99.9 |     80.0 |     86.3 |
| Optical Flow      |        2.29 |     98.6 |     94.4 |     95.3 |     100 |     80.0 |     86.4 |
| **Dynamics**      |    **0.37** | **98.6** | **95.3** | **95.5** | **100** | **80.6** | **87.2** |

Dynamics CoT gives the best SFT-stage PDMS while adding only 0.17s over no-CoT. Scene-description CoT is both slow and worse than no-CoT.

### Training Stages (Table 5)

| Base | Dyn CoT | SFT | RFT | NC | DAC | TTC | C | EP | PDMS |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| EMU3 | no | yes | no | 98.3 | 93.8 | 94.6 | 99.9 | 79.5 | 85.6 |
| EMU3 | yes | yes | no | 98.6 | 95.3 | 95.5 | 100 | 80.6 | 87.2 |
| EMU3 | no | yes | yes | 98.6 | 96.7 | 95.6 | 100 | 82.3 | 88.7 |
| **EMU3** | **yes** | **yes** | **yes** | **98.6** | **98.7** | **95.5** | **100** | **86.8** | **91.7** |
| Qwen2.5-VL | no | yes | no | 98.3 | 93.5 | 94.8 | 100 | 79.1 | 85.3 |
| Qwen2.5-VL | yes | yes | no | 98.8 | 94.4 | 95.8 | 100 | 79.9 | 86.6 |
| Qwen2.5-VL | no | yes | yes | 98.7 | 96.1 | 96.1 | 100 | 81.6 | 88.4 |
| **Qwen2.5-VL** | **yes** | **yes** | **yes** | **98.8** | **97.9** | **95.9** | **99.9** | **85.8** | **91.0** |

Dynamics CoT improves both SFT and RFT. On EMU3, Dyn CoT SFT adds +1.6 PDMS and the full Dyn CoT + RFT stack adds +6.1 over SFT without CoT.

### Tokenizer Design (Table 6)

| Model | Decouple | Image | BEV | NC | DAC | TTC | C | EP | PDMS |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| w/o CoT | - | - | - | 98.3 | 93.8 | 94.6 | 99.9 | 79.5 | 85.6 |
| Dyn CoT | no | yes | yes | 98.5 | 94.0 | 94.9 | 100 | 79.5 | 85.8 |
| Dyn CoT | yes | no | yes | 98.4 | 94.5 | 94.8 | 100 | 79.8 | 86.2 |
| Dyn CoT | yes | yes | no | 98.6 | 94.8 | 95.0 | 100 | 80.6 | 86.7 |
| **Dyn CoT** | **yes** | **yes** | **yes** | **98.6** | **95.3** | **95.5** | **100** | **80.6** | **87.2** |

Without decoupling, the tokenizer almost collapses to the no-CoT baseline. Both image and BEV reconstruction matter; BEV provides cross-view semantic regularization.

![[x5 22.png|Figure 5: Codebook activation/collapse without dynamics decoupling.]]

*Figure 5: Separate ego/environment branches plus action regularization prevent codebook collapse.*

### Prediction Horizon (Table 7)

| Horizon | Latency (s) | NC | DAC | TTC | C | EP | PDMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| w/o CoT | 0.20 | 98.3 | 93.8 | 94.6 | 99.9 | 79.5 | 85.6 |
| K=1 | 0.27 | 98.6 | 94.6 | 95.0 | 100 | 80.1 | 86.5 |
| **K=2** | **0.37** | **98.6** | **95.3** | **95.5** | **100** | **80.6** | **87.2** |
| K=3 | 0.49 | 98.6 | 94.7 | 95.2 | 100 | 80.3 | 86.7 |
| K=4 | 0.61 | 98.5 | 94.7 | 95.3 | 99.9 | 80.2 | 86.6 |

The 2s horizon is best: K=1 lacks lookahead, while K>2 adds uncertain future dynamics and latency.

### Token Count and Allocation (Table 8)

| Dyn Num | N_ego | N_env | NC | DAC | TTC | C | EP | PDMS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **8** | **4** | **4** | **98.6** | **95.3** | **95.5** | **100** | **80.6** | **87.2** |
| 8 | 2 | 6 | 98.7 | 94.9 | 95.3 | 100 | 80.5 | 86.9 |
| 8 | 6 | 2 | 98.6 | 94.5 | 95.0 | 100 | 80.2 | 86.4 |
| 4 | 2 | 2 | 98.5 | 94.6 | 95.1 | 100 | 80.1 | 86.4 |
| 16 | 8 | 8 | 98.5 | 94.7 | 94.9 | 100 | 80.3 | 86.5 |

Balanced 4 ego + 4 environment tokens is the sweet spot. More tokens introduce redundant dynamics representations rather than improving planning.

## Qualitative Findings

![[x4 22.png|Figure 4: Transferability of learned dynamics tokens.]]

*Figure 4: Dynamics tokens transferred across scenes decode into plausible future image/BEV states, suggesting the token codes represent reusable dynamics rather than memorized pixels.*

![[x6 19.png|Figure 6: Planning behavior with and without Dynamics CoT.]]

*Figure 6: Dynamics CoT improves stopping, foresight, and road-geometry-aware behavior.*

![[x7 15.png|Figure 7: Decoded future image and BEV from the Dynamics Tokenizer.]]

![[x8 11.png|Figure 8: Additional dynamics transfer visualizations.]]

![[x9 5.png|Figure 9: Additional qualitative planning comparisons.]]

## Limitations and Failure Cases

The authors explicitly note that inaccurate dynamics traces can propagate into wrong actions, analogous to flawed CoT in language reasoning.

![[x10 6.png|Figure 10: Failure cases.]]

*Figure 10: Failures include incorrect inference of surrounding vehicle intent, drivable-area misidentification during large turns, and ambiguous dynamics under degraded visual observations such as heavy rain.*

Key limitations:

1. **Dynamics hallucination risk**: wrong future dynamics can mislead the action generator.
2. **Long-horizon uncertainty**: K>2 degrades performance, suggesting dynamics tokens become unreliable farther into the future.
3. **Front-view only**: the main setting uses current and prior front-view images; side/rear dynamics may be under-observed.
4. **No NAVSIM-v2 / EPDMS result**: cannot assess extended comfort, lane keeping, DDC, or TLC.
5. **Comparison omissions**: NAVSIM table excludes several stronger contemporary methods; Bench2Drive table excludes LinkVLA.
6. **In-house dataset not public**: scaling result is useful but not independently comparable.

## Relationship to Wiki Methods

| Method | Intermediate reasoning | Inference role | Main tradeoff |
|---|---|---|---|
| AutoVLA / AdaThinkDrive | Text CoT | Generated before action | Interpretable but slower and spatially coarse |
| FSDrive | Future visual frame | Mandatory visual CoT | Spatially rich but pixel-heavy |
| ExploreVLA | RGB+depth prediction entropy | Reward source during RL | Novelty reward, not explicit action reasoning |
| DriveVLA-W0 | Future/current image prediction | Training-time only | Dense supervision but no inference-time reasoning |
| **DynVLA** | **Discrete dynamics tokens** | **Generated before action as CoT** | **Compact and fast, but vulnerable to dynamics hallucination** |

DynVLA is best viewed as a middle ground between text CoT and visual CoT: it keeps an explicit "think-then-act" generation order, but the thought is a small latent dynamics trace rather than a sentence or image.

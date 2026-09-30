---
title: "WALT: Learning World-Model-Aligned Latent Trajectories for Autonomous Driving"
type: source-summary
sources: ["raw/papers/WALT_ Learning World-Model-Aligned Latent Trajectories for Autonomous Driving.md"]
related: [concepts/action-tokenization.md, concepts/diffusion-planner.md, concepts/world-model-for-ad.md, concepts/navsim-benchmark.md, concepts/foundation-backbones-for-ad.md, concepts/evaluation-variance.md, concepts/inference-latency.md, sources/reworld.md, sources/clear.md, sources/auto-jepa.md, sources/epona.md, sources/geowam.md, sources/geoworldad.md, sources/suv.md, sources/qwen-drive-1.0.md, sources/lwdrive.md, sources/drivevla-w0.md, sources/unified-driving-tokens.md, sources/coworld-vla.md, sources/drivelaw.md, sources/ad-e2e-jepa.md, sources/redrive.md]
created: 2026-09-30
updated: 2026-09-30
confidence: medium
---

# WALT

**Paper**: WALT: Learning World-Model-Aligned Latent Trajectories for Autonomous Driving
**Authors**: Mingkai Jia, Jiaxin Guo, Zhijian Shu, Jiawei Xu, Mingxiao Li, Jintao Cheng, Ping Tan, Wei Yin
**Orgs**: HKUST, Horizon Robotics, CUHK, Nanjing University of Posts and Telecommunications, Nankai University
**arXiv**: 2609.30436v1
**Code**: not stated
**Source**: `raw/papers/WALT_ Learning World-Model-Aligned Latent Trajectories for Autonomous Driving.md`

---

## What It Is

A paper about the **action side** of a world-model planner. The world model (EponaV2) is frozen and untouched. What changes is the space the trajectory head generates in.

1. **Stage 1, a trajectory tokenizer.** An autoencoder maps the 8 future waypoints to 2 latent tokens. Each token has a *semantic* part, pulled toward the frozen world model's features of the same scene by a CLIP-style contrastive loss, and a *reconstruction* part that keeps the geometry.
2. **Stage 2, latent planning.** The tokenizer is frozen. The world model's flow-matching trajectory head is trained to generate the 2 latent tokens and a decoder turns them back into waypoints.

Four of the eight authors (J. Xu, Z. Shu, M. Jia, M. Li) appear in EponaV2's author list, so the baseline is the group's own model.

**Headline, against raw-waypoint generation on the same frozen backbone: NAVSIM v1 89.4 → 89.8 PDMS, NAVSIM v2 87.3 → 87.9 EPDMS, trajectory-head FLOPs per denoising step −30.5%.** No RL, no scorer.

---

## Key Takeaways

- **The whole effect is +0.41 PDMS and +0.6 EPDMS, from single runs.** That is below the threshold at which this wiki reads an ablation delta as detected ([[concepts/evaluation-variance.md]]).
- **Four action representations are indistinguishable.** Raw waypoints 89.42, a plain trajectory autoencoder 89.48, JEPA-style trajectory pretraining 89.46, REPA-style alignment to a trajectory encoder 89.49. Trajectory-only representation learning does nothing for the planner. This is the cleanest result in the paper.
- **On v2 the gain is extended comfort.** EC rises 68.3 → 73.4. That one sub-score accounts for about 0.6 of the 0.76 closed-form gain; the other eight net +0.16.
- **On v1 the gain is NC and TTC, and DAC falls.** NC +0.49, TTC +1.08, DAC −0.33. Those safety gains do not reappear in the v2 sub-scores of the same comparison (NC +0.1, TTC +0.1).
- **"Compact" means fewer tokens, not fewer numbers.** 8 waypoints × 3 values = 24 scalars become 2 tokens × 32 channels = 64 scalars.
- **The contrastive loss aligns pooled vectors.** Averaging all token-pair cosines equals the dot product of two mean vectors, so nothing in the loss ties a trajectory token to an image location. The "correspondence maps" are not a supervised quantity.
- **WALT's latents separate driving behaviours less than the alternatives and plan better.** Mean within-class minus between-class similarity: raw waypoints 0.53, JEPA-Traj 0.39, WALT 0.33.
- **RL on the same base is worth more.** EponaV2's headline rows in other ingested tables are 90.4 PDMS / 88.9 EPDMS (the v1 row is marked as RL-trained). WALT recovers about 40% of that gap without RL. The headline rows are not in WALT's tables.

---

## Method

![[letraj_pipe.png|WALT pipeline. Stage 1: a trajectory encoder feeds a semantic branch and a geometry branch; semantic tokens are aligned with frozen world-state tokens by the WALT loss, the two token sets are concatenated into compressed latents, and a trajectory decoder reconstructs the waypoints. Stage 2: frozen visual tokenizer and driving world model produce world-state tokens for a frozen visual predictor and a trainable trajectory planner, whose compressed latents are decoded by the frozen trajectory decoder]]

*Figure 2: Stage 1 trains the dual-branch tokenizer against a frozen world model. Stage 2 trains only the trajectory planner head to generate the latents.*

### Dual-branch trajectory tokenizer

- Input: $P=8$ ego-centric waypoints $(x,y,\text{yaw})$ at 0.5 s spacing.
- Output: $K=2$ tokens (ratio $q=P/K=4$), each 32-d: **24 semantic channels + 8 reconstruction channels**.
- A shared encoder feeds two refiners, giving $z_{sem}$ and $z_{rec}$. The latent is the channel-wise concatenation $z_A=[z_{sem};z_{rec}]$ and the decoder returns $\hat A=\mathcal D(z_A)$.

$$\mathcal L_{\text{tokenizer}}=\lambda_{rec}\,\mathbb E_A\big[\lVert\hat A-A\rVert_1\big]+\lambda_{align}\,\mathcal L_{\text{WALT}},\qquad \lambda_{align}=0.1$$

- The whole semantic part is masked with probability $p_{sem}=0.5$ during training, so the reconstruction part alone must be able to recover the trajectory.

### World-model alignment {#world-model-alignment}

The frozen world model (visual encoder and backbone; prediction heads unused) gives hidden states $H_w\in\mathbb R^{N_b\times M\times d_w}$ for each scene frame. A learned projector maps the semantic tokens to that width, $U=g(z_{sem})$. The score between trajectory sample $i$ and scene sample $j$ is the mean of all token-pair cosines:

$$S_{ij}=\frac1{KM}\sum_{k=1}^{K}\sum_{\ell=1}^{M}\frac{U_{ik}^{\mathsf T}H_{w,j\ell}}{\lVert U_{ik}\rVert_2\lVert H_{w,j\ell}\rVert_2},\qquad \mathcal L_{\text{WALT}}=\tfrac12\operatorname{CE}(\alpha S,y)+\tfrac12\operatorname{CE}(\alpha S^{\mathsf T},y)$$

- Positives share sample and frame index. Every other sample *or frame* in the global (cross-GPU) batch is a negative.
- $\alpha=\exp(s)$ is a learned, bounded logit scale.
- The input to the tokenizer is the **ground-truth future trajectory**. Scene features are never concatenated into the latent.

**An identity worth writing down.** Because the double sum factorizes,

$$S_{ij}=\bar u_i^{\mathsf T}\bar h_j,\qquad \bar u_i=\frac1K\sum_k\frac{U_{ik}}{\lVert U_{ik}\rVert},\quad \bar h_j=\frac1M\sum_\ell\frac{H_{w,j\ell}}{\lVert H_{w,j\ell}\rVert}$$

the objective sees only the mean of the unit-normalized trajectory tokens and the mean of the unit-normalized scene tokens. It is a **pooled** alignment written in token-pair form.

### Latent trajectory generation

Rectified flow on the concatenated latent, conditioned on the frozen world state:

$$z_t=(1-t)\epsilon+t\,z_A,\qquad \mathcal L_{\text{flow}}=\mathbb E_{A,\epsilon,t}\Big[\big\lVert\hat v_A(z_t,t;H_w)-(z_A-\epsilon)\big\rVert_2^2\Big]$$

Inference integrates from noise with Euler steps and decodes once: $\hat A=\mathcal D(\hat z_1)$. The number of steps is not stated.

### The two trajectory-only controls

| Variant | Generation target | Extra supervision |
|---|---|---|
| Raw waypoints (= EponaV2's head) | 8 waypoint tokens | – |
| + trajectory AE | 2 latent tokens | Reconstruction only |
| **JEPA-Traj** | 2 latent tokens | Predict the semantic tokens of a later overlapping sub-trajectory from an earlier one, conditioned on the time offset; SIGReg against collapse (weight $5\times10^{-4}$, other settings from LeWorldModel) |
| **REPA-Traj** | 8 waypoint tokens | Cosine alignment (weight 0.1) of the planner's trajectory stream, before the joint single-stream blocks, to the frozen JEPA-Traj semantic tokens |
| **WALT** | 2 latent tokens | Contrastive alignment of the semantic tokens to the frozen world model |

### Implementation

EponaV2 as the frozen world model, with its default model and planner-training settings. Tokenizer: 100 epochs, learning rate $1\times10^{-4}$, **32 H20 GPUs**. Stage 2 fine-tunes only the trajectory head.

---

## Figures

![Figure 1: Overview of WALT. Left, the baseline generates raw waypoints from frozen world-state tokens. Right, WALT generates compact trajectory latents aligned with the frozen world-model features, with a world–trajectory correspondence map](https://arxiv.org/html/2609.30436v1/teaser.png)

*Figure 1: Baseline (raw waypoints from frozen world-state tokens, with a PCA overlay of the world model's features) against WALT (latents aligned with those features, with a cosine-similarity map). This image is linked from arXiv; the clipping did not save it to `raw/assets/`.*

Figure 2 is in the Method section.

![[letraj_qual.png|Four cases (braking, straight, turn, curve). Rows: front-view image, world–trajectory correspondence heat map overlaid on the image, predicted trajectory in orange against ground truth in black]]

*Figure 3: Qualitative trajectories and correspondence maps. The maps are the cosine similarity between projected semantic tokens of the planner-generated latent and the frozen front-view visual tokens, averaged over trajectory tokens. Warm regions are broad: the lead vehicle and lane in (a), but also building facades and sky.*

![[letraj_wera.png|Trajectory representation analysis for raw waypoints, JEPA-Traj semantic latents and WALT semantic latents. Top: t-SNE embeddings coloured by six behaviour classes. Middle: histograms of within-class and between-class cosine similarity. Bottom: six-by-six class-pair mean-similarity matrices]]

*Figure 4: Representation analysis on identical NAVSIM v1 test samples. Six behaviour classes: left lane change, left turn, right lane change, right turn, start, straight.*

### Figure 4's class-pair mean cosine similarities, transcribed

| Class pair | Raw waypoints | JEPA-Traj | WALT |
|---|---:|---:|---:|
| *Within class* | | | |
| Left lane change | 0.93 | 0.75 | 0.87 |
| Left turn | 0.96 | 0.72 | 0.78 |
| Right lane change | 0.94 | 0.89 | 0.93 |
| Right turn | 0.95 | 0.78 | 0.81 |
| Start | 0.83 | 0.86 | 0.92 |
| Straight | 0.95 | 0.72 | 0.88 |
| *Between classes* | | | |
| Left lane – left turn | 0.61 | 0.42 | 0.53 |
| Left lane – right lane | 0.63 | 0.56 | 0.71 |
| Left lane – right turn | −0.10 | 0.32 | 0.42 |
| Left lane – start | 0.79 | 0.32 | 0.56 |
| Left lane – straight | 0.86 | 0.66 | 0.83 |
| Left turn – right lane | −0.11 | 0.28 | 0.36 |
| Left turn – right turn | −0.79 | 0.14 | 0.19 |
| Left turn – start | 0.30 | 0.31 | 0.46 |
| Left turn – straight | 0.28 | 0.39 | 0.47 |
| Right lane – right turn | 0.60 | 0.60 | 0.67 |
| Right lane – start | 0.75 | 0.24 | 0.46 |
| Right lane – straight | 0.86 | 0.69 | 0.80 |
| Right turn – start | 0.19 | 0.25 | 0.37 |
| Right turn – straight | 0.27 | 0.39 | 0.52 |
| Start – straight | 0.83 | 0.37 | 0.64 |
| **Mean within** | 0.93 | 0.79 | 0.87 |
| **Mean between** | 0.40 | 0.40 | 0.53 |
| **Within − between** | 0.53 | 0.39 | 0.33 |

The three summary rows are unweighted means of the printed values, computed for this page.

---

## Tables

### Table I: NAVSIM v1 (reported results; "different model and training configurations")

| Method | Venue | NC | DAC | EP | TTC | C | PDMS |
|---|---|---:|---:|---:|---:|---:|---:|
| Human | – | 100 | 100 | 87.5 | 100 | 99.9 | 94.8 |
| LAW | ICLR'25 | 96.4 | 95.4 | 81.7 | 88.7 | 99.9 | 84.6 |
| DrivingGPT | ICCV'25 | 98.9 | 90.7 | 79.7 | 94.9 | 95.6 | 82.4 |
| World4Drive | ICCV'25 | 97.4 | 94.3 | 79.9 | 92.8 | 100 | 85.1 |
| Epona | ICCV'25 | 97.9 | 95.1 | 80.4 | 93.8 | 99.9 | 86.2 |
| PWM | NeurIPS'25 | 98.6 | 95.9 | 81.8 | 95.4 | 100 | 88.1 |
| TISA (w/o MOPT) | ICRA'26 | 98.0 | 95.5 | 81.1 | 93.8 | – | 86.8 |
| AdaThinkDrive (w/o RL) | ICRA'26 | 98.9 | 95.3 | 80.6 | 96.0 | 100 | 87.5 |
| Mimir | RA-L'26 | 98.2 | 97.5 | 83.6 | 94.6 | 100 | 89.3 |
| PRIX | RA-L'26 | 98.1 | 96.3 | 82.3 | 94.1 | 100 | 87.8 |
| ARTEMIS | RA-L'26 | 98.3 | 95.1 | 81.4 | 94.3 | 100 | 87.0 |
| DriveVLA-W0 | ICLR'26 | 98.4 | 95.3 | 80.9 | 95.2 | 100 | 87.2 |
| DriveLaW | CVPR'26 | 99.0 | 97.1 | 81.3 | 96.7 | 100 | 89.1 |
| EponaV2 (w/o RL) | arXiv'26 | 98.6 | 97.3 | 83.6 | 95.3 | 99.9 | 89.4 |
| **WALT** | – | **99.1** | 97.0 | **83.6** | 96.3 | **100** | **89.8** |

### Table II: NAVSIM v2 navtest

\* original evaluator, before the human-penalty aggregation update. Unstarred rows are the corrected evaluator.

| Method | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | EPDMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Human | 100 | 100 | 99.8 | 100 | 87.4 | 100 | 100 | 98.1 | 90.1 | 94.5 |
| DriveVLA-W0 | 98.4 | 95.2 | 99.4 | 99.9 | 86.6 | 97.9 | 97.8 | 98.3 | 82.7 | 86.9 |
| ARTEMIS \* | 98.3 | 95.1 | 98.6 | 99.8 | 81.5 | 97.4 | 96.5 | 98.3 | – | 83.1 |
| PRIX \* ⚠ | 98.0 | 85.6 | 99.5 | 99.8 | 87.4 | 97.2 | 97.1 | 98.3 | 87.6 | 84.2 |
| DriveWorld-VLA | 98.6 | 99.1 | 99.6 | 99.8 | 87.4 | 97.9 | 97.0 | 97.8 | 78.6 | 86.8 |
| EponaV2 (w/o RL) | 98.4 | 96.7 | 99.6 | 99.9 | 87.6 | 98.1 | 98.0 | 98.1 | 68.3 | 87.3 |
| **WALT** | 98.5 | 96.8 | 99.6 | 99.9 | 87.4 | **98.2** | 97.9 | 98.3 | 73.4 | **87.9** |

⚠ PRIX's DAC is printed 85.6. [[sources/lwdrive.md]] carries the same row with DAC 95.6, and only 95.6 is arithmetically consistent with 84.2; see [Reading the Results](#table-notes).

### Table III: Action-side representation, NAVSIM v1, same frozen backbone

| Method | NC | DAC | EP | TTC | C | PDMS |
|---|---:|---:|---:|---:|---:|---:|
| Raw waypoints baseline | 98.62 | 97.32 | 83.60 | 95.26 | 99.93 | 89.42 |
| + trajectory AE (w/o $\mathcal L_{\text{WALT}}$) | 98.62 | 97.38 | 83.63 | 95.32 | 99.95 | 89.48 |
| JEPA-Traj | 98.55 | 97.37 | 82.85 | 95.91 | 100.00 | 89.46 |
| REPA-Traj | 98.48 | **97.46** | **83.79** | 95.11 | 99.98 | 89.49 |
| **WALT** | **99.11** | 96.99 | 83.63 | **96.34** | 100.00 | **89.83** |

### Table IV: Trajectory-generation cost for one denoising step

Latent-generation cost includes decoding. The frozen world model is excluded.

| Representation | Compression ratio | GFLOPs ↓ |
|---|---|---:|
| Raw waypoint generation | – | 297.47 |
| WALT latent generation | 4× | 206.88 |

---

## Reading the Results

### 1. Five variants, one of them 0.35 above the other four

| Step | Δ PDMS |
|---|---:|
| Raw waypoints → latent generation with a plain autoencoder | +0.06 |
| Autoencoder → JEPA-Traj (trajectory-only self-supervision on the latent) | −0.02 |
| Raw waypoints → REPA-Traj (align the planner's trajectory stream to a trajectory encoder) | +0.07 |
| Autoencoder → WALT (contrastive alignment to the frozen world model) | **+0.35** |

- **The nulls are the solid part.** Generating in a learned trajectory latent instead of waypoints, giving that latent a JEPA objective, and REPA-aligning the planner to it all land within 0.07 of the baseline. The action representation, taken alone, is not a lever on this backbone.
- **The positive result is one run at +0.35.** Sampler-seed noise for flow planners on navtest has been measured at 0.013–0.05 ([[concepts/evaluation-variance.md]]), so +0.35 is probably outside sampler noise. Training-seed noise has never been measured and the paper reports no seeds.
- **It is a trade inside the sub-scores.** NC +0.49 and TTC +1.08 against DAC −0.33, with EP flat. The closed form of the mean sub-scores moves 87.51 → 88.10.

### 2. v2 tells a different story about the same gain {#ec}

| | NC | DAC | EP | TTC | LK | HC | EC | EPDMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EponaV2 (w/o RL) | 98.4 | 96.7 | 87.6 | 98.1 | 98.0 | 98.1 | 68.3 | 87.3 |
| WALT | 98.5 | 96.8 | 87.4 | 98.2 | 97.9 | 98.3 | 73.4 | 87.9 |
| Δ | +0.1 | +0.1 | −0.2 | +0.1 | −0.1 | +0.2 | **+5.1** | +0.6 |

- The closed-form EPDMS of the mean sub-scores moves 86.23 → 87.00. Giving WALT the baseline's EC leaves 86.39. **About four fifths of the v2 gain is extended comfort.**
- EC compares the plans made at consecutive frames. A plausible reading is that a 2-token latent decoded through a smooth decoder gives less frame-to-frame jitter than eight independently denoised waypoint tokens. If so, the gain comes from *generating in a latent*, which Table III prices at +0.06 on v1.
- **The paper cannot separate the two.** Table III is v1 only. There is no v2 row for the plain autoencoder, so the EC gain cannot be assigned to the alignment loss.
- The v1 safety gains (NC +0.49, TTC +1.08) shrink to +0.1 each under v2's definitions, and DAC changes sign (−0.33 on v1, +0.1 on v2).

### 3. What the alignment can and cannot put into the latent

- **The latent is a function of the trajectory alone.** The tokenizer sees eight waypoints. Contrastive alignment can reorganize trajectory space so that trajectories from similar scenes lie close together. It cannot add scene information that the waypoints do not imply.
- **The planner's target becomes partly predictable from its own condition.** In Stage 2 the head is conditioned on $H_w$ and must generate $z_{sem}$, which was trained to agree with (pooled) $H_w$. 48 of the 64 target scalars are of this kind. This is a reasonable mechanism for an easier generation problem. The paper does not test it (no training-loss curves, no step-count sweep).
- **The loss is pooled** ([identity above](#world-model-alignment)), so the correspondence maps in Figures 1 and 3 show how similar each visual token is to one pooled trajectory vector. The paper says the maps "reflect representation-level correspondence rather than precise localization". They are not evidence that the latent attends to the lead vehicle.
- **Negatives include other frames of the same clip**, which are near-duplicates of the positive scene. The effect of that choice is not examined.

### 4. Behaviour separability is not what helps

From the transcribed Figure 4 matrices:

- WALT's between-class similarity is higher than JEPA-Traj's for **all 15 class pairs** (the paper says "several").
- In WALT's space, left lane change is as similar to straight driving (0.83) as to itself (0.87).
- Ranked by within-minus-between separation: raw waypoints 0.53, JEPA-Traj 0.39, WALT 0.33. Ranked by PDMS: WALT, then the other two tied.

The paper draws the right conclusion ("stronger separation under trajectory-only learning does not necessarily yield a larger PDMS gain") and adds that the visualizations do not establish which contextual information causes the improvement.

### 5. "Compact" and the cost claim

- **Scalars go up.** 24 → 64. Even the reconstruction part alone (2 × 8 = 16) is two thirds of the input size. The compression is in sequence length, which is what the transformer head pays for.
- **Reconstruction error is never reported.** There is no L1 or ADE for the autoencoder, with or without the semantic part.
- **The FLOPs are for the head only, per step.** 297.47 → 206.88 GFLOPs. A 75% cut in trajectory tokens gives a 30.5% cut in cost, so most of the head's cost is processing the world-state tokens it conditions on.
- **No latency, no step count, no parameter count, and the world model is excluded.** 200–300 GFLOPs per denoising step is large for a planner head; the total is $N$ times that.
- **Training cost is not small.** The tokenizer for an 8-waypoint sequence is trained for 100 epochs on 32 H20 GPUs, because every batch needs the frozen world model's features.

### 6. Table notes {#table-notes}

**The tables are scoped to "no RL" and that scope hides the group's own stronger row.**

| | PDMS | EPDMS | EC | Source |
|---|---:|---:|---:|---|
| EponaV2, no RL (raw waypoints) | 89.4 | 87.3 | 68.3 | This paper |
| **WALT**, no RL | 89.8 | 87.9 | 73.4 | This paper |
| EponaV2, headline | 90.4 | 88.9 | 77.4 | [[sources/geowam.md]], [[sources/geoworldad.md]], [[sources/suv.md]], [[sources/qwen-drive-1.0.md]] |

- The 90.4 row is marked as RL-trained in Qwen-Drive's table, and EponaV2's navhard row is marked RL-supervised in GeoWAM's. The 88.9 row has different sub-scores from the no-RL row here, so it is a different checkpoint, presumably the same RL model.
- On that reading RL is worth +1.0 PDMS / +1.6 EPDMS on this base and WALT is worth +0.4 / +0.6. Whether the two stack is not tested.
- EC goes 68.3 (no RL) → 73.4 (WALT) → 77.4 (headline).

**Where the numbers sit.** Both of WALT's rows pass the wiki's arithmetic checks (v1 +1.7 above the closed form; v2 residual −0.9, corrected-like, as the caption states). Among no-RL, no-scorer entries in this wiki, 89.8 PDMS is below ReDrive 91.0, WA-JEPA 91.8, SUV 90.8, ReWorld 90.4 and CoWorld-VLA 90.0. In the corrected v2 cohort, 87.9 is below every 89+ entry and just above DreamerAD 87.7. "Best overall PDMS" and "strongest EPDMS among all the methods" hold inside tables of 14 and 6 learned methods.

**Row-level.**

| Row | Note |
|---|---|
| v2 PRIX \* | **DAC 85.6 is a misprint for 95.6.** With 85.6 the closed form is 77.5, which is 6.7 *below* the reported 84.2; no row in the wiki has a residual of that sign and size. With 95.6 it is 86.6 (+2.4, the usual pre-fix gap), and 95.6 is what [[sources/lwdrive.md]]'s copy of the row has |
| v2 DriveVLA-W0 86.9 | Not the 86.1 / EC 58.9 row that most ingested tables carry. This row (EC 82.7) appears elsewhere only in [[sources/geowam.md]]'s table. It is unstarred and its residual (−0.9) agrees |
| v2 DriveWorld-VLA 86.8 | Unstarred, but its residual is +2.6, the pre-fix signature. Already a standing exception on [[concepts/navsim-benchmark.md]] |
| v2 ARTEMIS \* | Printed with HC 98.3 and EC blank |
| v1 DriveVLA-W0 87.2 | Matches the PDMS of the flow-matching-head variant in [[sources/drivevla-w0.md]]'s ablation, not its 88.4 query-based result or its 90.2 anchor-based headline. Unlabelled |
| v1 TISA, Mimir | New to the wiki (86.8 without its policy-refinement stage; 89.3) |

The thirteen learned v1 rows with complete sub-scores sit 1.7–4.0 above their closed form, so none pairs sub-scores with another model's aggregate.

---

## Relationships

- **[[sources/reworld.md]]**: the nearest result. ReWorld pulls a flow planner's action tokens toward their own cross-attention readout of frozen video features and gains **+0.4 PDMS** (89.1 → 89.5). WALT pulls a trajectory tokenizer's latent toward frozen world-model features and gains +0.35 to +0.41. Two groups, two backbones, two places to apply the alignment, the same size of effect.
- **[[sources/clear.md]]**: the earlier latent-trajectory generator in this wiki. CLEAR's trajectory VAE is pretrained with a maneuver-classification head; WALT's autoencoder is pretrained with a world-model contrastive loss. WALT's claim to be the first to learn a generative trajectory latent "by transferring information from a frozen driving world model" is specific to that teacher.
- **[[sources/auto-jepa.md]]**: also embeds future ego trajectories in a learned latent and aligns a scene-side prediction to it. There the latent is a retrieval key over recorded trajectories; here it is a generation target with a decoder. WALT cites it accurately.
- **[[sources/ad-e2e-jepa.md]]**: the other ingested driving paper that imports SIGReg from LeWorldModel, there for a visual projector, here for JEPA-Traj. In both papers the SIGReg component has no ablation of its own.
- **[[sources/unified-driving-tokens.md]]** and **[[sources/coworld-vla.md]]**: the visual-side counterparts. Changing the *visual* tokenizer's objectives moves a fixed readout by several PDMS (85.5 → 91.8); changing the *action* representation here moves a fixed backbone by tenths. CoWorld-VLA's 2×2 found representation and planner to be substitutes when the other is strong, which is the regime WALT is in.
- **[[sources/drivelaw.md]]**: "planner size is not the bottleneck". WALT adds that, on a strong frozen backbone, the planner's output space is not much of one either.
- **[[sources/epona.md]]**: the predecessor of WALT's frozen world model. EponaV2 itself is not ingested.
- **Not ingested**: EponaV2 (the backbone and baseline), WorldDrive (joint vision–motion representation, the coupled alternative WALT argues against), LeWorldModel and REPA (the two recipes adapted for the controls), Mimir, TISA.

---

## Limitations

**Evidence**

1. **Small effects, single runs, no seeds.** +0.41 PDMS, +0.6 EPDMS.
2. **The ablation is v1-only.** The v2 gain is mostly EC and cannot be attributed to alignment rather than to latent generation.
3. **No ablation of the design itself.** The 24/8 channel split, $p_{sem}=0.5$, $\lambda_{align}=0.1$, $K=2$, the contrastive form (against cosine or regression alignment), and which world-model layer supplies $H_w$ are all fixed.
4. **No tokenizer metrics.** Reconstruction error, and how much the decoder relies on the semantic part, are not reported.
5. **One backbone.** "Without modifying the world model" is shown for EponaV2 only, by EponaV2's authors.

**Claims**

6. **"Compact"** holds for token count and not for dimensionality (24 → 64 scalars).
7. **The correspondence maps** visualize a quantity the loss does not supervise per token.
8. **The efficiency claim** is per-step head FLOPs. No wall-clock time, no end-to-end figure, no step count.
9. **The comparison tables** exclude RL methods, including EponaV2 with RL, and omit every no-RL method above 89.8 / 87.9.

**Scope**

10. **navtest only.** No navhard, although EponaV2 has a navhard result (36.1 with RL), and no closed-loop benchmark.
11. **DAC falls on v1** (97.32 → 96.99) and is the lowest of the five variants. The paper describes the remaining metrics as "competitive".
12. **No code or checkpoint is mentioned.**

**Source conversion**

13. Three of four figures are in `raw/assets/`. Figure 1 (`teaser.png`) is a remote arXiv link in the clipping. All four tables are present. The front matter's author field is empty; authors are taken from the body.

---

## Key Cross-References

- [[concepts/action-tokenization.md]] — a continuous latent trajectory tokenizer, and a four-way null on trajectory-only representation learning.
- [[concepts/diffusion-planner.md]] — flow matching in a learned trajectory latent; the second alignment result of about +0.4 PDMS after ReWorld.
- [[concepts/world-model-for-ad.md]] — Pattern 42: the frozen world model as a teacher for the action space.
- [[concepts/navsim-benchmark.md]] — 89.8 / 87.9; EponaV2 without RL; a DAC misprint caught by the residual check.
- [[concepts/foundation-backbones-for-ad.md]] — a frozen driving world model as an alignment target.
- [[concepts/evaluation-variance.md]] — two evaluators attributing one gain to different sub-scores.
- [[concepts/inference-latency.md]] — head-only FLOPs, no latency.

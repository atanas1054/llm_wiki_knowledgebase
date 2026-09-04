---
title: "GeoWAM: Visual Geometry World Action Models for Autonomous Driving"
type: source-summary
sources: [raw/papers/GeoWAM_ Visual Geometry World Action Models for Autonomous Driving.md]
related: [sources/geoworldad.md, concepts/world-model-for-ad.md, concepts/navsim-benchmark.md, concepts/navhard-ood-evaluation.md, concepts/perception-for-planning.md, concepts/foundation-backbones-for-ad.md, concepts/diffusion-planner.md, sources/wa-jepa.md, sources/auto-jepa.md, sources/drive-jepa.md, sources/latent-wam.md, sources/sgdrive.md, sources/drivelaw.md, sources/simwam.md, sources/policy-world-model.md, sources/epona.md, sources/drivevla-w0.md, sources/deepsight.md]
created: 2026-09-02
updated: 2026-09-04
confidence: high
---

# GeoWAM

GeoWAM's argument is a state-space argument: **pixels encode geometry and motion only indirectly**, entangled with appearance, texture, and illumination, so a video world model can produce visually plausible futures by capturing photometric regularities without ever recovering the 3D transformations that generated them. Geometry, by contrast, is *native* to driving — point clouds explicitly encode spatial structure, changes in geometry directly reveal motion, and scene geometry and ego trajectories live in the **same 3D coordinate frame**.

So GeoWAM replaces future-image prediction with **future point-map prediction**. A DVGT-2 geometry encoder builds a multi-level memory from historical multiview frames; a future-geometry decoder forecasts dense metric point maps for the next 8 steps; and a geometry-conditioned action head infers future ego motion from that forecast geometry through a stop-gradient, in what the paper calls an inverse-dynamics-like formulation.

It reports 90.2 EPDMS on NAVSIM v2 navtest and **36.6 EPDMS on navhard**, the latter ahead of every listed baseline including methods trained with RL or direct PDMS-score supervision.

**Source**: `raw/papers/GeoWAM_ Visual Geometry World Action Models for Autonomous Driving.md`
**arXiv**: https://arxiv.org/html/2608.23486v2 (published 2026-08-25)
**Project page**: https://yiren-lu.com/project_pages/geowam/
**Authors**: Yiren Lu, Xin Ye (corresponding), Jiaming Liu, Philip Jacobson, Jin Yao, Yi-chung Chen, Liam Merino, Dhruva Dixith Kurra, Min Cai, Tom Lampo, Yu Yin (corresponding), Danhua Guo, Burhan Yaman (project lead) — **Uber AV Labs** + Case Western Reserve University

> **Read the [protocol warning](#the-navsim-v2-number-cannot-be-placed) before comparing 90.2 EPDMS to anything else in this wiki** — but note it has since been **narrowed to two rows**. GeoWAM's NAVSIM-v2 table gives Transfuser an EPDMS of 84.0 from submetrics *digit-for-digit identical* to the ones four other ingested papers score at 76.7. However, [[sources/geoworldad.md]] independently reproduces GeoWAM's DVGT-2 (89.6) and EponaV2 (88.9) rows exactly while reporting the standard Transfuser and DiffusionDrive values, so the Transfuser/DiffusionDrive rows are best read as anomalies rather than as evidence of a separate protocol. **90.2 and GeoWorldAD's 90.4 are comparable to each other.** See [The Sibling Paper](#the-sibling-paper-geoworldad).

## Key Takeaways

- **A genuinely new world-model state space.** Every world model in the wiki predicts pixels, video latents, semantic features, occupancy voxels, symbolic state, or an action latent. GeoWAM predicts **dense metric point maps** — one 3D point per image pixel in the ego frame, per future step, per camera. It is the first entry whose prediction target lives in the same coordinate system as the output trajectory.
- **No occupancy or 3D box annotation required.** Unlike OccWorld/Drive-OccWorld, which need voxelized ground truth, GeoWAM's point-map targets are derived from off-the-shelf geometry foundation models. Training needs **only RGB**. This is the sharpest contrast with [[sources/sgdrive.md]], which buys 3D structure by paying for occupancy and box labels.
- **The action head is deterministic single-trajectory regression** — no anchors, no mode classification, no diffusion, no sampling. That it reaches the top of two tables is a real counterpoint to the field's drift toward generative planners.
- **navhard 36.6 EPDMS is the more impressive result**, beating EponaV2† 36.1, NavFormer† 34.1, and LTFv6† 31.9 — all three of which use RL or direct PDMS-score supervision that GeoWAM does not.
- **Geometry forecasting beats video-then-reconstruct at long horizons.** Mean Abs Rel 0.257 vs. 0.274 for Epona+DVGT, and mean δ<1.25 of 0.754 vs. 0.655. But at the 1 s horizon Epona+DVGT is *better* on δ<1.25 (0.732 vs. 0.708); GeoWAM's advantage only opens up from 2 s onward.
- **The honest attribution of the planning gain is +0.6 EPDMS.** DVGT-2 — GeoWAM's own initialization, and already a geometry model — scores 89.6 to GeoWAM's 90.2. The paper's claim that geometry world modeling "yields substantially stronger driving policies than image-based alternatives" rests on comparisons against *other papers' methods*, never against its own architecture trained with a pixel objective.
- **There are no ablations.** Not one. For a paper whose entire thesis is that one pretraining objective beats another, the controlled experiment is absent.

## Method

### Why Geometry (the paper's argument)

Video world models are optimized to model the distribution of future *observations*. That objective "does not require them to explicitly recover the underlying physical dynamics that generate those observations" — a model can satisfy it with visual spatiotemporal regularities while the 3D transformations remain implicit and hard to recover. Geometry inverts this: forecasting future point maps gives direct supervision on spatial structure and scene dynamics, in the coordinate frame where actions are executed.

This is a *representation* claim, and it sits alongside [[sources/drivelaw.md]]'s controlled finding that video-generator latents beat BEV features and VLM hidden states under a fixed planner. Both papers argue about which representation should condition a planner; they reach opposite conclusions about pixels, and neither tests the other's alternative.

### Stage 1: Visual Geometry World Model

**Multiview geometry encoding.** Given a $K$-frame multiview history $\mathbf{I}_{t-K+1:t}$ with $V$ cameras, a **DVGT-2** encoder $\mathcal{E}_\theta$ produces multi-level tokens at $L$ selected feature levels. At each step $\tau$ and level $\ell$ it emits two token types — geometry tokens $\mathbf{X}_\tau^\ell\in\mathbb{R}^{V\times P\times D}$ for spatial scene structure and ego tokens $\mathbf{E}_\tau^\ell\in\mathbb{R}^{V\times N_e\times D}$ for ego-motion context — concatenated as $\mathbf{Z}_\tau^\ell=[\mathbf{X}_\tau^\ell;\mathbf{E}_\tau^\ell]$:

$$
\mathcal{Z}_{t}=\left\{\mathbf{Z}_{\tau}^{\ell}\right\}_{\tau,\ell}=\mathcal{E}_{\theta}(\mathbf{I}_{t-K+1:t})
$$

The two-stream token design matters: the ego stream is what Stage 2 later hijacks for planning.

**Future geometry decoding.** A learned query seed $\mathbf{q}^\mathrm{geom}$ is replicated across future step, view, and spatial location, with additive learned temporal and view embeddings plus a 2D sinusoidal positional embedding:

$$
\mathbf{Q}_{t+k}^{\mathrm{geom},v,p}=\mathbf{q}^{\mathrm{geom}}+\mathbf{e}_{K+k}^{\mathrm{time}}+\mathbf{e}_{v}^{\mathrm{view}}+\mathbf{e}_{p}^{\mathrm{2D}}
$$

Each decoder layer does two things: **causal temporal self-attention** across the $F$ future steps at each spatial location, then **cross-attention to the historical memory** $\mathcal{Z}_t$. A per-level output projection $\mathbf{W}_\ell$ maps the shared latent to each feature level:

$$
\hat{\mathcal{U}}_{t+1:t+F}=\mathcal{D}_{\phi}(\mathcal{Q}_{t+1:t+F}^{\mathrm{geom}},\mathcal{Z}_{t}),\qquad
\hat{\mathbf{X}}_{t+k}^{\ell}=\hat{\mathbf{U}}_{t+k}\mathbf{W}_{\ell}^{\mathsf{T}}
$$

A shared **Point DPT** head $\mathcal{G}_\psi$ decodes the multi-level features into a dense point map and per-pixel confidence:

$$
\left(\hat{\mathbf{P}}_{t+k},\hat{\mathbf{C}}_{t+k}\right)=\mathcal{G}_{\psi}\left(\left\{\hat{\mathbf{X}}_{t+k}^{\ell}\right\}_{\ell=1}^{L}\right)
$$

with $\hat{\mathbf{P}}_{t+k}\in\mathbb{R}^{V\times H\times W\times 3}$ storing one 3D point per pixel **in the ego coordinate system at time $t+k$**. Future image appearance is never reconstructed.

**Supervision — a hybrid of JEPA-style alignment and dense regression.** Future images are pushed through the *same* encoder in a detached target branch to give patch-feature targets $\bar{\mathbf{X}}^\ell_{t+k}$, aligned by cosine distance:

$$
\mathcal{L}_{\mathrm{feat}}=\frac{1}{FL}\sum_{k=1}^{F}\sum_{\ell=1}^{L}\left(1-\cos\left(\hat{\mathbf{X}}_{t+k}^{\ell},\operatorname{sg}\!\left(\bar{\mathbf{X}}_{t+k}^{\ell}\right)\right)\right)
$$

Future images go only to the target branch — never to the forecasting branch, and never at inference. On top of this sits an explicit dense objective combining Euclidean point regression, confidence-aware regression, and multi-scale surface-normal consistency:

$$
\mathcal{L}_{\mathrm{point}}^{\mathrm{future}}=\mathcal{L}_{\mathrm{reg}}+\mathcal{L}_{\mathrm{conf}}+\mathcal{L}_{\mathrm{normal}},\qquad
\mathcal{L}_{\mathrm{pre}}=\mathcal{L}_{\mathrm{feat}}+\mathcal{L}_{\mathrm{point}}^{\mathrm{future}}+\mathcal{L}_{\mathrm{point}}^{\mathrm{current}}
$$

$\mathcal{L}^\mathrm{current}_\mathrm{point}$ applies the same objective to the encoded current frame, "anchoring the geometry encoder while it learns to forecast."

**$\mathcal{L}_\mathrm{feat}$ is a JEPA objective in all but name** — stop-gradient target branch, cosine alignment in latent space, no pixel reconstruction. The differences from [[sources/wa-jepa.md]] are that GeoWAM uses a plain shared encoder rather than an EMA target, and that it pairs the latent objective with dense point-map regression that plausibly does the anti-collapse work an EMA teacher would otherwise do.

### Stage 2: The Action Branch

**Future ego-token decoding.** $N_e$ learned ego-query seeds, one per ego-token slot, with temporal and view embeddings:

$$
\mathbf{Q}_{t+k}^{\mathrm{ego},v,n}=\mathbf{q}_{n}^{\mathrm{ego}}+\mathbf{e}_{K+k}^{\mathrm{time}}+\mathbf{e}_{v}^{\mathrm{view}}
$$

Each ego-decoder layer applies causal temporal self-attention across future steps (independently per view and slot), then cross-attends to **both** the historical memory and the predicted future geometry:

$$
\hat{\mathbf{E}}_{t+1:t+F}=\mathcal{D}_{\eta}^{\mathrm{ego}}\left(\mathbf{Q}^{\mathrm{ego}},\mathcal{Z}_{t},\operatorname{sg}\!\left(\hat{\mathcal{U}}_{t+1:t+F}\right)\right)
$$

**The stop-gradient is the design's load-bearing choice.** Trajectory loss cannot propagate through the predicted future geometry, so planning supervision never reshapes the geometry it conditions on. The paper's stated rationale is twofold: preserve the forecasting capability acquired in pretraining, and enforce the inverse-dynamics reading — ego motion is *inferred from* scene evolution, not co-adapted with it.

This is the same asymmetric-coupling idea as [[sources/wa-jepa.md]]'s stop-gradient, but pointing the **opposite way**. WA-JEPA blocks the *scene* loss from updating the *action* stream so that action supervision shapes the world representation. GeoWAM blocks the *action* loss from updating the *geometry* stream so that world modeling stays pristine. Two papers, same mechanism, opposite priorities, neither ablated.

**Trajectory decoding.** The action head takes deepest-level historical ego tokens $\mathbf{E}^L_{t-K+1:t}$ and predicted future ego tokens, concatenates them temporally, refines with a causal temporal transformer, then has a learned trajectory query cross-attend to the refined sequence. A regression head emits a **single** trajectory:

$$
\hat{\mathbf{A}}_{t}=\mathcal{H}_{\omega}\left(\mathbf{E}_{t-K+1:t},\hat{\mathbf{E}}_{t+1:t+F}\right),\qquad \hat{\mathbf{a}}_{t+k}=(\hat{x}_{t+k},\hat{y}_{t+k},\hat{\theta}_{t+k})
$$

Heading $\hat\theta$ is predicted alongside position — as in WA-JEPA, and unlike most $(x,y)$-only NAVSIM planners. The paper is explicit: "without trajectory anchors, mode classification, or iterative sampling."

$$
\mathcal{L}_{\mathrm{plan}}=\mathcal{L}_{\mathrm{pre}}+\lambda_{\mathrm{traj}}\mathcal{L}_{\mathrm{traj}}+\lambda_{\mathrm{pose}}\mathcal{L}_{\mathrm{pose}}
$$

with $\ell_1$ trajectory regression, an auxiliary $\ell_1$ loss on relative poses between historical frames, and $\lambda_\mathrm{traj}=\lambda_\mathrm{pose}=5$. Both geometry objectives are retained during finetuning.

### Implementation

| Setting | Value |
|---|---|
| Geometry encoder | DVGT-2 (encoder + point head initialized from it) |
| Future geometry decoder | 6 transformer layers, hidden 1024, 16 heads |
| Pretraining input | 3 historical frames, 2–8 camera views dynamically sampled |
| Forecast horizon | $F=8$ future frames at 2 Hz (4 s) |
| Pretraining data | OpenScene, nuScenes, Bench2Drive, Waymo, KITTI, Argoverse 2, DDAD |
| Pretraining schedule | 161 epochs, AdamW, weight decay 0.05, bfloat16 |
| Learning rates | future decoder $10^{-4}$; pretrained components $2\times10^{-5}$; 5% linear warmup + cosine decay |
| Planning finetune | 40 epochs on NAVSIM navtrain, **8 camera views**, 3 historical frames |
| Planning LRs | future decoder + new action head $10^{-4}$; other pretrained params $2\times10^{-5}$ |
| Loss weights | $\lambda_\mathrm{traj}=\lambda_\mathrm{pose}=5$ |
| Hardware | **not reported anywhere** |

## Figures

![[GeoWAM_teaser.png]]

**Figure 1.** The core argument in one picture. Given the same current observation, a video world model predicts how pixel values evolve, leaving the underlying 3D transformations implicit and hard to recover; a geometry world model predicts future 3D structure, whose evolution exposes those transformations directly in a representation aligned with motion planning.

![[GeoWAM_pipeline.png]]

**Figure 2.** Architecture. Historical multiview frames → geometry encoder → multi-level memory of geometry and ego/pose tokens. The future geometry decoder applies temporal self-attention and cross-attends to that memory, producing future geometry tokens decoded by Point DPT into dense point maps. In the action branch the predicted geometry tokens condition the future ego/pose decoder **through a stop-gradient connection**, and the trajectory head maps the resulting ego/pose tokens to the future trajectory.

![[GeoWAM_viz.png]]

**Figure 3.** Qualitative future-geometry prediction for left-turn, straight, and right-turn cases, aggregating all future steps into one scene with boxes marking predicted ego poses. The paper highlights two things worth checking against the images: in the left-turn case another vehicle follows the ego through the turn in the predicted geometry, which would indicate the model captures surrounding-agent dynamics and not just ego motion; and in the straight case the predicted trajectory steers around a roadside vehicle. Trees, poles, and road markings survive the horizon. These are three hand-picked scenes with no failure cases shown.

## Tables

### Table 1: Future Geometry Prediction (nuScenes validation)

Predicted point maps are converted to ray depth — distance from the predicted 3D point to the ego origin — and scored with absolute relative error and threshold accuracy. Video baselines are run as *generate frames, then reconstruct with DVGT*, giving all methods a common geometric output.

| Method | Abs Rel 1s ↓ | 2s | 3s | 4s | **mean** | δ<1.25 1s ↑ | 2s | 3s | 4s | **mean** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Epona + DVGT | 0.229 | 0.263 | 0.292 | 0.310 | 0.274 | **0.732** | 0.677 | 0.620 | 0.589 | 0.655 |
| Cosmos 3 + DVGT | 0.300 | 0.376 | 0.405 | 0.422 | 0.376 | 0.588 | 0.513 | 0.464 | 0.447 | 0.503 |
| VGGT-World | 0.272 | 0.329 | 0.342 | 0.357 | 0.325 | 0.612 | 0.553 | 0.513 | 0.497 | 0.544 |
| **GeoWAM** | **0.228** | **0.245** | **0.256** | **0.297** | **0.257** | 0.708 | **0.769** | **0.746** | **0.703** | **0.754** |

**The horizon structure is the interesting part.** At 1 s, GeoWAM and Epona+DVGT are effectively tied on Abs Rel (0.228 vs. 0.229) and Epona is *better* on threshold accuracy (0.732 vs. 0.708). GeoWAM's advantage appears from 2 s onward and widens: at 4 s the δ<1.25 gap is 0.703 vs. 0.589. Note also that GeoWAM's own δ<1.25 is *non-monotone* — it rises from 0.708 at 1 s to 0.769 at 2 s before declining — which the paper does not comment on and which is odd for a forecasting metric.

The comparison also beats VGGT-World, the one baseline that forecasts geometry directly rather than through a video model, by a wide margin (0.257 vs. 0.325 mean Abs Rel).

### Table 2: NAVSIM v2 navtest

| Method | NC ↑ | DAC ↑ | DDC ↑ | TLC ↑ | EP ↑ | TTC ↑ | LK ↑ | HC ↑ | EC ↑ | EPDMS ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Transfuser | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 84.0 |
| Hydra-MDP++ | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | 81.4 |
| DriveSuprim | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 | 83.1 |
| ARTEMIS | 98.3 | 95.1 | 98.6 | 99.8 | 81.5 | 97.4 | 96.5 | 98.3 | – | 83.1 |
| DiffusionDrive | 98.2 | 96.2 | 99.5 | 99.8 | 87.4 | 97.3 | 96.9 | 98.4 | 87.7 | 88.2 |
| WoTE | 98.5 | 96.8 | 98.8 | 99.8 | 86.1 | 97.9 | 95.5 | 98.3 | 82.9 | 87.7 |
| DriveVLA-W0 | 98.4 | 95.2 | 99.4 | 99.9 | 86.6 | 97.9 | 97.8 | 98.3 | 82.7 | 86.9 |
| PWM | 98.8 | 95.9 | 99.4 | 99.9 | 86.4 | 98.4 | 97.6 | 98.3 | 85.3 | 88.2 |
| DriveLaW | 98.7 | 96.9 | 99.6 | 99.8 | 87.5 | 98.3 | 97.6 | 98.4 | 77.4 | 88.6 |
| DVGT-2 | 98.7 | 97.9 | **99.7** | **99.9** | 87.9 | 98.0 | **98.2** | 98.2 | 77.0 | 89.6 |
| EponaV2 | 98.5 | 97.4 | 99.5 | **99.9** | 87.9 | 98.1 | 97.7 | 98.2 | 77.4 | 88.9 |
| **GeoWAM** | 98.7 | 97.7 | **99.7** | **99.9** | 87.0 | 98.1 | 97.9 | 98.3 | 86.8 | **90.2** |

GeoWAM's EC of 86.8 is the standout submetric — second only to DiffusionDrive's 87.7 and far above the 77.0–77.4 band that DVGT-2, EponaV2, and DriveLaW occupy. A deterministic single-trajectory regressor has no frame-to-frame jitter to suppress, which is exactly the weakness sampled planners show on this metric ([[sources/wa-jepa.md]] 88.1 is comparable; [[sources/auto-jepa.md]] 75.2 and DriveVLA-W0 58.9 are not).

### The NAVSIM v2 Number Cannot Be Placed {#the-navsim-v2-number-cannot-be-placed}

**This table is not commensurable with the wiki's NAVSIM-v2 column, and the proof is inside the table itself.**

| Method | Submetrics (NC/DAC/DDC/TLC/EP/TTC/LK/HC/EC) | EPDMS in GeoWAM | EPDMS elsewhere |
|---|---|---:|---:|
| Transfuser | 96.9 / 89.9 / 97.8 / 99.7 / 87.1 / 95.4 / 92.7 / 98.3 / 87.2 — **identical in both** | **84.0** | **76.7** ([[sources/wa-jepa.md]], [[sources/drive-jepa.md]], wiki table) |
| DiffusionDrive | differ by ≤0.3 on DAC/DDC/EP/LK/HC | **88.2** | **84.5** ([[sources/wa-jepa.md]], labelled *corrected*) |

Transfuser is the airtight case: **nine identical submetrics, two different aggregate scores, 7.3 points apart.** No aggregation rule can produce both. DiffusionDrive is nearly as strong — submetric differences of at most 0.3 cannot yield a 3.7-point EPDMS gap.

Worse, GeoWAM's table is not internally consistent either. Its DriveSuprim (83.1) and Hydra-MDP++ (81.4) rows are digit-for-digit identical to Drive-JEPA's ResNet-34 rows *including* the EPDMS, while Transfuser and DiffusionDrive have been recomputed. So the table mixes numbers from at least two conventions, and there is no way to tell which convention GeoWAM's own 90.2 belongs to.

This goes beyond the two-protocol picture established when [[sources/wa-jepa.md]] was ingested. WA-JEPA classified DiffusionDrive's 84.5 and Transfuser's 76.7 as *corrected* and *pre-fix* respectively; GeoWAM, which states it uses "the official human-penalty protocol," produces different values for both. Either there are more than two aggregation variants in circulation, or at least one paper's classification is wrong. The wiki cannot resolve it from published tables. See [[concepts/navsim-benchmark.md]].

### Table 3: navhard Two-Stage Pseudo-Closed-Loop

navhard approximates closed loop with 3DGS-reconstructed scenes: the planner predicts a trajectory, the benchmark renders a new observation from the resulting ego pose and feeds it back, so planning errors affect subsequent observations. Methods marked † use reinforcement learning or direct PDMS-score supervision.

| Method | Stage | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | EPDMS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CV | S1 | 88.8 | 42.8 | 70.6 | 99.3 | 77.5 | 87.3 | 78.6 | 97.1 | 60.4 | 11.4 |
| | S2 | 83.2 | 59.1 | 76.5 | 98.0 | 71.3 | 81.1 | 47.9 | 97.1 | 61.9 | |
| Ego MLP | S1 | 93.2 | 55.7 | 86.6 | 99.3 | 81.2 | 92.2 | 83.5 | 97.5 | 77.7 | 14.1 |
| | S2 | 77.2 | 51.9 | 74.4 | 98.2 | 77.1 | 75.0 | 40.8 | 97.8 | 79.8 | |
| LTF | S1 | 96.2 | 79.5 | 99.1 | 99.5 | 84.1 | 95.1 | 94.2 | 97.5 | 79.1 | 25.1 |
| | S2 | 77.7 | 70.2 | 84.2 | 98.0 | 85.1 | 75.6 | 45.4 | 95.7 | 75.9 | |
| DriveVLA-W0 | S1 | 96.8 | 83.3 | 99.0 | 99.6 | 84.6 | 95.3 | 96.4 | 97.6 | 78.2 | 24.4 |
| | S2 | 76.8 | 64.3 | 79.9 | 98.3 | 89.2 | 75.0 | 46.8 | 95.8 | 53.1 | |
| DriveLaW | S1 | 97.3 | 89.1 | 99.2 | 99.6 | 84.3 | 97.1 | 96.2 | 97.8 | 67.6 | 30.6 |
| | S2 | 82.5 | 67.6 | 83.5 | 98.1 | 84.8 | 78.5 | 45.8 | 96.4 | 57.3 | |
| DVGT-2 | S1 | 97.2 | 91.3 | 98.4 | 99.8 | 84.8 | 95.5 | 95.5 | 97.5 | 71.4 | 31.7 |
| | S2 | 77.8 | 73.8 | 81.3 | 98.3 | 91.5 | 73.2 | 48.0 | 83.9 | 45.1 | |
| LTFv6 † | S1 | 96.5 | 86.6 | 99.2 | 99.5 | 84.4 | 95.1 | 94.4 | 97.7 | 76.4 | 31.9 |
| | S2 | 79.8 | 75.5 | 86.2 | 97.8 | 89.5 | 76.0 | 50.0 | 95.2 | 66.7 | |
| NavFormer † | S1 | 96.2 | 92.4 | 95.7 | 99.6 | 83.8 | 96.0 | 94.7 | 96.4 | 60.9 | 34.1 |
| | S2 | 85.7 | 81.0 | 83.5 | 97.6 | 90.1 | 82.4 | 48.2 | 94.9 | 48.4 | |
| EponaV2 † | S1 | 97.3 | 90.7 | 99.4 | 100.0 | 83.3 | 97.3 | 97.3 | 97.6 | 60.9 | 36.1 |
| | S2 | 83.6 | 78.0 | 88.0 | 98.9 | 86.0 | 80.3 | 50.1 | 96.1 | 52.0 | |
| **GeoWAM** | S1 | **97.7** | 91.5 | 99.1 | 99.8 | 83.8 | 95.8 | 96.0 | 97.8 | **79.0** | **36.6** |
| | S2 | 80.4 | 76.3 | 87.3 | 98.7 | 88.9 | 76.2 | 49.9 | 94.0 | 56.0 | |

Three things this table shows better than anything else in the wiki:

- **Stage 2 is where everything breaks.** Every method loses 15–20 points of NC and roughly half its LK between stages — LK falls from 96.0 to 49.9 for GeoWAM, from 97.3 to 50.1 for EponaV2, and even the constant-velocity baseline drops 78.6 → 47.9. Once the planner's own errors drive the rendered observations, lane keeping collapses across the board. This is a far more legible failure signature than the aggregate scores the wiki has been tracking.
- **GeoWAM beats three methods that use supervision it does not have.** EponaV2, NavFormer, and LTFv6 all use RL or direct PDMS-score supervision (the paper greys them out for this reason); GeoWAM uses $\ell_1$ trajectory regression and still leads. Its margin over EponaV2 is only +0.5, but the supervision asymmetry runs the other way.
- **+4.9 over DVGT-2 here versus +0.6 on navtest.** Whatever future-geometry forecasting adds over a static geometry model, it shows up eight times more strongly under the reactive protocol. That is the paper's most interesting unremarked result — and it is exactly what a world-model thesis would predict, since anticipation should matter more when errors compound.

## Relationships

- **DVGT-2** (un-ingested) is the single most important comparison and appears in both planning tables. It is GeoWAM's encoder *and* point-head initialization, and it is the strongest navtest baseline at 89.6. GeoWAM is best understood as **DVGT-2 plus future forecasting**: +0.6 EPDMS on navtest, +4.9 on navhard. Everything else in the paper's comparison is against methods with different backbones and data.
- **[[sources/latent-wam.md]]** — the closest existing use of geometry. Latent-WAM distills WorldMirror/VGGT geometric features into compact latents at training time and discards the teacher at inference; GeoWAM makes metric geometry the actual prediction target and keeps the geometry branch live. Latent-WAM's ablation is instructive here: geometric distillation moved it 88.3 → 89.3 EPDMS, and LoRA was insufficient for that distillation target. Both papers agree geometry is valuable for planning; they disagree on whether it should be a target or a teacher.
- **[[sources/sgdrive.md]]** — the other 3D-structured world model, and the cleanest cost contrast. SGDrive forecasts occupancy voxels, 3D boxes, and a goal pose, which requires occupancy and box annotation; GeoWAM's point maps come from geometry foundation models and need only RGB. SGDrive's targets are human-interpretable by construction; GeoWAM's dense point maps are visualizable but not semantically labelled. SGDrive also found that structured perception of the *present* (+2.5 PDMS) mattered more than forecasting the future (+0.8) — GeoWAM's navtest/navhard split (+0.6 / +4.9 over DVGT-2, where DVGT-2 supplies the present) points the same way on navtest and the opposite way on navhard.
- **[[sources/wa-jepa.md]]** — three overlaps worth naming. Both use an asymmetric stop-gradient between world and action streams, in **opposite directions**. Both predict heading alongside position. And GeoWAM's $\mathcal{L}_\mathrm{feat}$ is a JEPA-style cosine alignment to a stop-gradiented target branch — precisely the *deterministic* latent objective WA-JEPA measures as harmful (90.7 vs. 91.1 EPDMS, from temporal-mean collapse). GeoWAM pairs it with dense point-map regression, which may anchor the representation where WA-JEPA's pure regression collapsed, but nothing in either paper tests this. **This is the most testable cross-paper question the two raise.**
- **[[sources/auto-jepa.md]]** — the opposite answer to the same question. Both reject dense future-*image* reconstruction. Auto-JEPA concludes the minimal sufficient target is the ego trajectory latent and predicts nothing about the scene; GeoWAM concludes the target should be dense 3D scene structure. They bracket the design space: the least and the most that a planning world model can predict. Auto-JEPA gets agent selectivity for free from an ego-motion target; GeoWAM gets metric structure but must forecast every pixel's 3D position.
- **[[sources/drivelaw.md]] / [[sources/simwam.md]]** — the video-prior camp GeoWAM is arguing against. GeoWAM's Table 1 supports its case at long horizons (video-then-reconstruct degrades faster), but it never runs DriveLaW's controlled experiment: fix the planner, swap only the conditioning representation. DriveLaW found video-generator latents beat BEV by +5.0 and VLM hidden states by +2.6 under exactly that control. Geometry is not in DriveLaW's comparison and pixels are not in GeoWAM's.
- **[[sources/policy-world-model.md]] / [[sources/epona.md]]** — PWM appears at 88.2 in GeoWAM's navtest table; Epona is the strongest geometry-prediction baseline when paired with DVGT. Epona's 0.732 δ<1.25 at 1 s beating GeoWAM's 0.708 is a real, if narrow, win for the video route at short horizons.
- **Deterministic regression as a live option** — GeoWAM tops two tables with a single regressed trajectory and no anchors, modes, or sampling, at a moment when nearly every strong wiki entry is a diffusion, flow-matching, or selection planner. Its EC of 86.8 against the 77-band suggests part of the reason. See [[concepts/diffusion-planner.md]].
- **Six un-ingested methods** appear here at or near the top: **DVGT-2** (89.6 navtest / 31.7 navhard) is now the most consequential gap in the wiki, followed by **EponaV2** (88.9 / 36.1), **NavFormer** (34.1), **LTFv6/LEAD** (31.9), **VGGT-World**, and **DVGT**. Uber AV Labs' own **VLGA** and **UniDrive-WM** are cited but not evaluated against.

## Limitations

**The comparison cannot be placed**

- **GeoWAM's NAVSIM-v2 table gives Transfuser 84.0 from submetrics identical to the 76.7 reported by three other sources**, and DiffusionDrive 88.2 against WA-JEPA's 84.5 from near-identical submetrics. Its own DriveSuprim and Hydra-MDP++ rows match Drive-JEPA's byte for byte, EPDMS included. The table therefore mixes conventions, and 90.2 cannot be ranked against the wiki's v2 column in either direction. This is not a criticism unique to GeoWAM — it is the clearest evidence yet that NAVSIM-v2 EPDMS is not currently a comparable number across papers.
- No NAVSIM-v1 PDMS is reported, so there is no second axis to sanity-check the v2 placement against.

**Attribution**

- **The gain over DVGT-2 is +0.6 EPDMS on navtest.** DVGT-2 is GeoWAM's own initialization and already a geometry model, so this is the cleanest available measurement of what *future forecasting* adds — and it is small. The abstract's claim that visual geometry world modeling "yields substantially stronger driving policies than image-based alternatives" is supported only by cross-paper comparisons where backbone, data, and training all differ.
- **There is not a single ablation in the paper.** No geometry-pretraining vs. no-pretraining row, no $\mathcal{L}_\mathrm{feat}$ vs. $\mathcal{L}_\mathrm{point}$ decomposition, no stop-gradient ablation, no forecast-horizon sweep, no camera-count study. Most conspicuously, **the paper never trains its own architecture with a pixel-prediction objective** — the one experiment that would isolate geometry-vs-pixels rather than GeoWAM-vs-other-papers. The thesis is a representation claim tested only through published baselines.
- The stop-gradient is described twice as central to the inverse-dynamics formulation and never measured. WA-JEPA has the identical omission with the identical mechanism pointed the other way.

**Method-level**

- **The action head is deterministic and unimodal.** One regressed trajectory, no anchors, no sampling. It buys the strong EC (86.8) and costs everything multimodality provides — no Best-of-N, no proposal diversity, no way to expose alternative maneuvers at an ambiguous intersection. Given the mode-collapse literature the wiki tracks ([[concepts/intent-conditioned-planning.md]], Curious-VLA), the absence of any discussion of this is a gap rather than an oversight.
- **Point-map targets are pseudo-labels from geometry foundation models**, and the encoder is initialized from DVGT-2. The supervision ceiling is whatever those models get right, and if the target generator is in the same family as the encoder, part of what is being learned is that model's biases. The paper presents "requires only RGB" as a pure advantage without discussing the dependency.
- $\mathcal{L}_\mathrm{feat}$ uses a **plain shared encoder with stop-gradient**, not an EMA target as in V-JEPA/WA-JEPA. Shared-encoder stop-gradient setups are the classic collapse risk; the dense point objectives presumably prevent it, but the paper neither raises the issue nor shows a representation-health metric of the kind WA-JEPA reports.
- The "inverse-dynamics-like" framing is loose. True inverse dynamics recovers actions from an observed state transition; here a learned decoder produces ego tokens conditioned on predicted geometry. It is conditioning with a stop-gradient, not inversion.

**Evaluation**

- **Geometry prediction is evaluated on nuScenes validation while nuScenes is in the pretraining mix**, and the paper never states whether the validation split is excluded. The same concern applies more sharply to planning: **OpenScene is a pretraining dataset and NAVSIM navtest derives from OpenScene.** Neither exclusion is stated. This is the same undisclosed overlap flagged for [[sources/wa-jepa.md]]'s nuPlan pretraining, and it now looks systemic rather than incidental.
- The geometry comparison has three baselines, two of which are two-stage pipelines (video model → DVGT) whose errors compound by construction. That is arguably the paper's point, but it makes the margin partly an artifact of pipeline depth rather than of representation choice.
- GeoWAM's own δ<1.25 is non-monotone in horizon (0.708 at 1 s, 0.769 at 2 s), which is unexplained and slightly undermines reading the metric as a clean forecasting curve.
- Bench2Drive is in the *pretraining* mix but no Bench2Drive closed-loop result is reported. No HUGSIM either. For a paper claiming closed-loop strength, navhard is the only reactive evidence.

**Reproducibility**

- **No compute is reported at all** — no GPU count, no hours, nothing — for 161 epochs of pretraining across seven datasets plus 40 epochs of finetuning with 8 camera views.
- **No latency, FPS, or parameter count.** A DVGT-2-scale encoder over 8 views and 3 frames, a 6-layer 1024-d decoder, and a dense Point DPT head decoding $V\times H\times W$ points per future step is not obviously deployable, and the wiki's comparable entries all report inference cost.
- No seed variance, single run. Less worrying than usual given the deterministic head, but the +0.6 over DVGT-2 sits well inside the range where it would matter.

## The Sibling Paper: GeoWorldAD

[[sources/geoworldad.md]] (NTU + Xiaomi EV + Zhejiang, arXiv 2607.17521) is an independent geometry world-action model from a different continent, built on the same DVGT-2 lineage, reporting **90.4 EPDMS navtest against GeoWAM's 90.2**. Neither paper cites the other.

| | **GeoWAM** | GeoWorldAD |
|---|---|---|
| Backbone | DVGT-2 | StreamVGGT → EgoStreamVGGT |
| Future target | **dense metric point maps**, 8 steps | **latent tokens** supervised by future depth, 4 chunks / 2 s |
| Present grounding | multi-level memory | multi-scale tokens (layers 4/11/17/23), iteratively consumed |
| Action head | **deterministic single-trajectory regression** | 64 proposals + simulator-distilled scorer |
| navtest EPDMS | 90.2 | **90.4** |
| navhard | **36.6** | not reported |
| Ablations | **none** | three |

### It supplies two of the three missing experiments

This page's headline criticism has been that GeoWAM contains **no ablations at all**. GeoWorldAD runs two of the three that were most wanted:

- **The coordinate-frame argument, measured.** GeoWAM asserts that geometry's advantage is living in the same frame as the trajectory. GeoWorldAD tests it: an anchor-frame StreamVGGT with 4D reconstruction supervision beats a from-scratch planner by only **0.6 PDMS** (84.2 → 84.8) *while lowering NC, DAC, and TTC*; re-expressing the same model's point maps in per-timestep ego frames — pure re-parameterization — is worth **+2.5** (→ 87.3). The argument holds, and it turns out to be conditional on actually doing the alignment rather than automatic from choosing a geometric target.
- **With vs. without a future.** GeoWorldAD's present-geometry-only planner scores 89.3 PDMS / 87.6 EPDMS; adding latent future geometry gives 91.0 / 90.4, with the gain concentrated in ego progress (+3.3 / +2.8). That is the ablation this page notes GeoWAM lacks — though it is confounded by 64K unmatched extra training steps.

**Neither paper runs the third and most important one.** *"Geometry beats pixels"* still has no controlled test: no version of either architecture is trained with a pixel or video future target under an otherwise fixed planner. Two independent papers, same thesis, same missing experiment.

### It also narrows the protocol warning

The [warning at the top of this page](#the-navsim-v2-number-cannot-be-placed) should now be read as covering **two rows, not the whole table**. GeoWorldAD's v2 table reports **DVGT-2 at 89.6 and EponaV2 at 88.9 — identical to GeoWAM's** — while giving **Transfuser 76.7 and DiffusionDrive 84.5**, the values GeoWAM records as 84.0 and 88.2. A second independent paper reproducing GeoWAM's headline anchors alongside the standard baseline values makes "two anomalous rows" a far more economical explanation than "a third aggregation protocol."

Practical consequence: **GeoWAM's 90.2 and GeoWorldAD's 90.4 are comparable to each other**, both measured against DVGT-2 at 89.6 — so GeoWAM's honest +0.6 attribution and GeoWorldAD's +0.8 sit on the same scale. See [GeoWorldAD Narrows the GeoWAM Anomaly](../concepts/navsim-benchmark.md#geowam-narrowed).

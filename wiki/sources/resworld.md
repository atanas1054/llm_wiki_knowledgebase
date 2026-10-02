---
title: "ResWorld: Temporal Residual World Model for End-to-End Autonomous Driving"
type: source-summary
sources: ["raw/papers/ResWorld_ Temporal Residual World Model for End-to-End Autonomous Driving.md"]
related: [concepts/world-model-for-ad.md, concepts/nuscenes-waymo-evals.md, concepts/navsim-benchmark.md, concepts/evaluation-variance.md, concepts/perception-for-planning.md, concepts/inference-latency.md, sources/drivefuture.md, sources/momworld.md, sources/wa-jepa.md, sources/redrive.md, sources/drive-hwm.md, sources/simwam.md, sources/drivevla-w0.md, sources/auto-jepa.md, sources/diffusiondrive.md, sources/foresight.md, research-directions.md]
created: 2026-10-02
updated: 2026-10-02
confidence: medium
---

# ResWorld

**Paper**: ResWorld: Temporal Residual World Model for End-to-End Autonomous Driving
**Authors**: Jinqing Zhang, Zehua Fu, Zelin Xu, Wenying Dai, Qingjie Liu, Yunhong Wang
**Orgs**: State Key Laboratory of Virtual Reality Technology and Systems, Beihang University; Hangzhou Innovation Institute, Beihang University; Beijing Jingwei Hirain Technologies; Zhongguancun Laboratory
**arXiv**: 2602.10884v1
**Code**: https://github.com/mengtan00/ResWorld
**Source**: `raw/papers/ResWorld_ Temporal Residual World Model for End-to-End Autonomous Driving.md`

---

## What It Is

A BEV planner with a small "world model" whose only input is **how the scene changed over the last few frames**.

Past BEV feature maps are warped into the current ego frame. Subtracting pooled scene queries of adjacent frames leaves the parts of the scene that moved; the paper calls these **temporal residuals** and treats them as a detection-free stand-in for dynamic objects. A few attention layers turn the residuals into a "future" BEV map, which is the current BEV map plus a learned dynamic-object term. A second planning stage then reads that map at the points where a first-pass trajectory says the car will be.

The argument has two parts:
1. **Static structure does not need predicting.** If the future map is expressed in the current ego frame, roads and buildings stay where they are, so the current BEV map already describes them.
2. **The future map should feed the trajectory directly**, not only serve as an auxiliary task.

There is **no future-prediction loss**. The only losses are L1 on the first-pass and final trajectories. Adding a loss toward the real future map makes the planner worse (Table 4).

| Benchmark | Setting | Result |
|---|---|---|
| nuScenes (UniAD-style metrics) | No ego status / with ego status | 0.65 / 0.59 m average L2; 0.23% / 0.17% average collision |
| nuScenes (VAD-style metrics) | No ego status / with ego status | 0.35 / 0.30 m; 0.07% / 0.06% |
| NAVSIM-v1 navtest | No perception supervision | 87.3 PDMS |
| NAVSIM-v1 navtest | Detection and map supervision, **no temporal residuals** (agent queries instead) | 88.3 PDMS |
| NAVSIM-v1 navtest | Detection and map, with history frames and temporal residuals | **89.0 PDMS** |

---

## Key Takeaways

- **Its "world model" has no world-model loss.** The future BEV map is trained only by the trajectory loss, through the refinement stage. Supervising it toward the real next-frame map *hurts*: average collision 0.17% → 0.21% and L2 0.59 → 0.61 m. On a conventional world-model branch the same supervision does nothing measurable (0.21% → 0.23%). This joins [[sources/drivefuture.md]] (no future-prediction loss) and [[sources/wa-jepa.md]] (regression on future latents worse than none) on the side where a future latent does better without a future target.
- **Most of the gain survives deleting the world model at inference.** The first-pass ("prior") trajectory uses exactly the baseline's architecture. Trained jointly with the world model and refinement stage, it scores 0.61 m / 0.18%, against 0.65 / 0.28 for the baseline and 0.59 / 0.17 for the full model. The refinement stage the paper is built around adds 0.02 m and 0.01 points at inference. The paper itself suggests deploying the prior trajectory.
- **Residual input beats full-scene input, by a little.** With the same refinement stage and no future supervision, residual input gives 0.59 m / 0.17% and the paper's "normal world model" 0.61 m / 0.21%. The residual is a first difference of scene queries, the same quantity [[sources/momworld.md]] uses to initialize its momentum state.
- **The NAVSIM headline does not use the named mechanism.** The 88.3 PDMS model replaces temporal residuals with detection agent queries, "since previous methods did not utilize historical frames". Only the 89.0 model uses residuals. All three NAVSIM numbers are below the wiki's frontier, and the table compares only with Transfuser-era baselines.
- **One NAVSIM row looks misprinted.** The 88.3 model's TTC is 98.9, four points above every other row including its own history variant (95.6). The closed-form check agrees: its PDMS sits +1.0 above the closed form of its sub-scores, against +2.0 to +4.1 for every other row in the table.
- **The nuScenes differences are small in absolute terms.** The ablation deltas are 0.02–0.06 m of L2 and 0.02–0.11 points of collision rate, from single runs. With roughly 6,000 validation samples, 0.04 points is a few samples.
- **"Collapse prevention" is shown only by a heatmap**, and the figure's caption describes its rows differently from the figure labels and the text. The future map is never compared with a real future.

---

## Method

![[resworld.png|ResWorld overview. Multi-view images at times t, t−1 to t−k become BEV features. The fused BEV feature gives fused scene queries and a prior trajectory. Scene queries from each timestamp are subtracted pairwise to give temporal residuals, which the Temporal Residual World Model turns into a future BEV feature. Future-Guided Trajectory Refinement applies deformable attention on the future BEV feature, using the prior trajectory as reference points, and outputs the final trajectory]]

*Figure 2: Overall framework. Prior trajectory prediction (top right), temporal residuals (bottom), the Temporal Residual World Model (centre) and Future-Guided Trajectory Refinement (bottom right).*

### 1. BEV features and the prior trajectory

- **Encoder.** GeoBEV, the first author's own BEV detector, chosen because residuals need BEV maps that align well across frames. ResNet-50, 256×704 images on nuScenes, $k=2$ past frames.
- **Temporal fusion.** Past maps $\mathbf B_{t-1},\dots,\mathbf B_{t-k}$ are warped into the current ego frame (as in BEVDet4D) and fused:

$$\mathbf B_{fuse}=\mathrm{Conv}(\mathrm{Concat}(\mathbf B_t,\mathbf B_{t-1},\dots,\mathbf B_{t-k}))$$

- **Sparse scene queries** (TokenLearner): a spatial attention map weights the BEV map and average pooling gives $N_s$ queries, which go through self-attention.

$$\mathbf S_{fuse}=\mathrm{AvgPool}(\mathrm{SA}(\mathbf B_{fuse})\odot\mathbf B_{fuse})$$

- **Prior trajectory.** $N_t$ waypoint queries, one per future timestamp, cross-attend to $\mathbf S_{fuse}$; an MLP decodes $\mathbf T_{prior}\in\mathbb R^{N_t\times2}$. This is SSR's perception-free planning module.

### 2. Temporal residuals

The same spatial attention map, computed from $\mathbf B_{fuse}$, pools every past frame:

$$\mathbf S_i=\mathrm{AvgPool}(\mathrm{SA}(\mathbf B_{fuse})\odot\mathbf B_i)$$

Residuals are differences of adjacent frames, $\mathbf R_i=\mathbf S_i-\mathbf S_{i-1}$, giving $k$ residuals. Because all maps share the current ego frame, a static surface gives a near-zero residual and a moving one does not.

### 3. Temporal Residual World Model (TR-World)

![[tr_world.png|TR-World. Each temporal residual passes through its own self-attention block; the outputs are summed, then a TokenFuser combines them with the fused BEV feature to give the future BEV feature]]

*Figure 3: Structure of TR-World.*

$$\hat{\mathbf R}=\sum_{i=t-k+1}^{t}\mathrm{SelfAttention}(\mathbf R_i),\qquad \mathbf B_{future}=\mathrm{MLP}(\mathbf B_{fuse})\otimes\hat{\mathbf R}+\mathbf B_{fuse}$$

The TokenFuser term is the inverse of TokenLearner: an MLP maps $\mathbf B_{fuse}$ to $N_s$ spatial weight maps, which spread the $N_s$ residual tokens back over the BEV grid. The future map stays in the **current** ego frame, so static content is carried over unchanged by the $+\mathbf B_{fuse}$ term. The future horizon of $\mathbf B_{future}$ is not defined; the paper says it should hold information "for a future time period".

### 4. Future-Guided Trajectory Refinement (FGTR)

$$\mathbf W=\mathrm{DeformAttention}(\mathbf W,\mathbf B_{future},\mathbf T_{prior}),\qquad \mathbf T_{final}=\mathrm{MLP}(\mathbf W)$$

Each waypoint query samples $\mathbf B_{future}$ around its own prior waypoint. The paper's reading: the query can "check whether the ego vehicle will collide with other objects or drive out of the drivable area", and the waypoints' positions and timestamps act as "sparse spatial-temporal supervision" on the future map.

### 5. Loss

$$\mathcal L=\mathrm{L1}(\mathbf T_{prior},\mathbf T_{GT})+\mathrm{L1}(\mathbf T_{final},\mathbf T_{GT})$$

Nothing else. "We do not utilize real future data to generate the label for supervising $\mathbf B_{future}$."

### NAVSIM variant

A TransFuser-like model: two ResNet-34 backbones on concatenated camera images and a LiDAR BEV map, trained 100 epochs (batch 512, learning rate $6\times10^{-4}$). In the main variant, **detection agent queries replace the temporal residuals** as the world model's input. A history-frame variant (⋆) uses residuals. A perception-free variant is also reported; what its world model reads is not stated.

---

## Figures

![[world_model_small.png|Panel (a) of Figure 1: a conventional world-model framework. Frame T multi-view images give a current scene representation, which with a trajectory enters a world model that outputs a predicted scene, aligned with the scene representation encoded from frame T+1]]

*Figure 1(a): the "normal" world-model framework, in which the predicted scene is aligned with the encoded next frame. Panel (b), ResWorld's framework, was not saved with the clipping.*

Figures 2 and 3 are in the Method section.

![[collapse.png|Two rows of five BEV feature heatmaps from five driving scenes. The top row, labelled w/o FGTR, shows one bright blob near the centre with little else, similar across scenes. The bottom row, labelled w/ FGTR, shows several elongated structures that differ between scenes]]

*Figure 4: future BEV features across five scenes. The figure labels the rows "w/o FGTR" (top) and "w/ FGTR" (bottom), as does the text. The caption instead describes the top row as "supervised using real future data". No quantitative collapse measure is reported.*

![[vis2.png|One nuScenes scene: six camera views and a BEV plot with the ground-truth trajectory in green, ResWorld's in red and SSR's in blue. SSR's trajectory curves right into a red car at the kerb, circled in both the front camera and the BEV plot; ResWorld's follows the ground truth]]

*Figure 5: qualitative comparison with SSR. Boxes and lane lines are drawn from annotations; the dashed circle marks the collision SSR's trajectory would cause. The circled car appears to be parked, which is the kind of object the paper's own limitations say temporal residuals cannot see.*

---

## Tables

### Table 1: nuScenes open-loop planning

Upper block: UniAD-style metrics. Lower block (‡): VAD-style metrics, temporal averages of the UniAD ones. ∗ evaluated by the authors with official models and code. ◊ ego status used in the planner, following BEV-Planner++.

| Method | Auxiliary task | L2 1s | L2 2s | L2 3s | **L2 Avg** | CR 1s | CR 2s | CR 3s | **CR Avg** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ST-P3 | Det & Map | 1.72 | 3.26 | 4.86 | 3.28 | 0.44 | 1.08 | 3.01 | 1.51 |
| UniAD | Det & Track & Map & Motion & Occ | 0.48 | 0.96 | 1.65 | 1.03 | 0.05 | 0.17 | 0.71 | 0.31 |
| OccNet | Det & Map & Occ | 1.29 | 2.13 | 2.99 | 2.14 | 0.21 | 0.59 | 1.37 | 0.72 |
| PARA-Drive | Det & Track & Map & Motion & Occ | 0.40 | 0.77 | 1.31 | 0.83 | 0.07 | 0.25 | 0.60 | 0.30 |
| GenAD | Det & Map & Motion | 0.36 | 0.83 | 1.55 | 0.91 | 0.06 | 0.23 | 1.00 | 0.43 |
| SSR ∗ | None | 0.25 | 0.64 | 1.33 | 0.74 | 0.08 | 0.12 | 0.72 | 0.31 |
| **ResWorld** | None | 0.22 | 0.56 | 1.17 | **0.65** | 0.02 | 0.04 | 0.64 | **0.23** |
| **ResWorld ◊** | None | 0.19 | 0.50 | 1.08 | **0.59** | 0.02 | 0.06 | 0.43 | **0.17** |
| ST-P3 ‡ | Det & Map | 1.33 | 2.11 | 2.90 | 2.11 | 0.23 | 0.62 | 1.27 | 0.71 |
| UniAD ‡ | Det & Track & Map & Motion & Occ | 0.44 | 0.67 | 0.96 | 0.69 | 0.04 | 0.08 | 0.23 | 0.12 |
| VAD ‡ | Det & Map & Motion | 0.41 | 0.70 | 1.05 | 0.72 | 0.07 | 0.17 | 0.41 | 0.22 |
| BEV-Planner++ ◊‡ | None | 0.16 | 0.32 | 0.57 | 0.35 | 0.00 | 0.29 | 0.73 | 0.34 |
| PARA-Drive ‡ | Det & Track & Map & Motion & Occ | 0.25 | 0.46 | 0.74 | 0.48 | 0.14 | 0.23 | 0.39 | 0.25 |
| LAW ‡ | None | 0.26 | 0.57 | 1.01 | 0.61 | 0.14 | 0.21 | 0.54 | 0.30 |
| LAW ‡ | Det & Map & Motion | 0.24 | 0.46 | 0.76 | 0.49 | 0.08 | 0.10 | 0.39 | 0.19 |
| GenAD ‡ | Det & Map & Motion | 0.28 | 0.49 | 0.78 | 0.52 | 0.08 | 0.14 | 0.34 | 0.19 |
| SparseDrive ‡ | Det & Track & Map & Motion | 0.29 | 0.58 | 0.96 | 0.61 | 0.01 | 0.05 | 0.18 | 0.08 |
| Drive-OccWorld ‡ | Occ | 0.25 | 0.44 | 0.72 | 0.47 | 0.03 | 0.08 | 0.22 | 0.11 |
| SSR ∗‡ | None | 0.19 | 0.36 | 0.62 | 0.39 | 0.10 | 0.10 | 0.24 | 0.15 |
| MomAD ‡ | Det & Track & Map & Motion | 0.31 | 0.57 | 0.91 | 0.60 | 0.01 | 0.05 | 0.22 | 0.09 |
| DiffusionDrive ‡ | Det & Track & Map & Motion | 0.27 | 0.54 | 0.90 | 0.57 | 0.03 | 0.05 | 0.16 | 0.08 |
| **ResWorld ‡** | None | 0.17 | 0.32 | 0.55 | **0.35** | 0.01 | 0.02 | 0.16 | **0.07** |
| **ResWorld ◊‡** | None | 0.14 | 0.27 | 0.49 | **0.30** | 0.01 | 0.03 | 0.14 | **0.06** |

All of ResWorld's averages reproduce from its per-horizon values. The Drive-OccWorld row cites reference 19 (SSR) instead of 39.

### Table 2: NAVSIM-v1 navtest

⋆ uses a history frame to compute temporal residuals. The last column is computed here: reported PDMS minus $\mathrm{NC}\cdot\mathrm{DAC}\cdot(5\,\mathrm{TTC}+2\,\mathrm C+5\,\mathrm{EP})/12$.

| Method | Auxiliary task | NC | DAC | TTC | Comf. | EP | PDMS | Reported − closed form |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LAW | None | 96.4 | 95.4 | 88.7 | 99.9 | 81.7 | 84.6 | +4.0 |
| World4Drive | None | 97.4 | 94.3 | 92.8 | 100 | 79.9 | 85.1 | +3.7 |
| **ResWorld** | None | 98.1 | 95.6 | 94.3 | 100 | 81.8 | **87.3** | +2.9 |
| UniAD | Det & Map | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 | +4.1 |
| PARA-Drive | Det & Map | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 | +4.0 |
| TransFuser | Det & Map | 97.7 | 92.8 | 92.8 | 100 | 79.2 | 84.0 | +3.9 |
| DRAMA | Det & Map | 98.0 | 93.1 | 94.8 | 100 | 80.1 | 85.5 | +3.8 |
| VADv2 | Det & Map | 97.2 | 89.1 | 91.6 | 100 | 76.0 | 80.9 | +6.0 |
| Hydra-MDP-W-EP | Det & Map | 98.3 | 96.0 | 94.6 | 100 | 78.7 | 86.5 | +2.6 |
| DiffusionDrive | Det & Map | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 | +2.7 |
| **ResWorld** | Det & Map | 98.2 | 96.4 | **98.9** ⚠ | 100 | 82.5 | **88.3** | **+1.0** ⚠ |
| **ResWorld ⋆** | Det & Map | 98.9 | 96.5 | 95.6 | 100 | 83.1 | **89.0** | +2.0 |

⚠ With TTC 94.9 the residual would be +2.6, in line with DiffusionDrive's +2.7 at a similar score. The printed 98.9 is probably a misprint.

### Table 3: Component ablation (nuScenes, UniAD-style metrics)

| Ego status | TR-World | FGTR | L2 1s | L2 2s | L2 3s | **L2 Avg** | CR 1s | CR 2s | CR 3s | **CR Avg** |
|:-:|:-:|:-:|---:|---:|---:|---:|---:|---:|---:|---:|
| ✗ | | | 0.25 | 0.62 | 1.27 | 0.71 | 0.02 | 0.25 | 0.64 | 0.31 |
| ✗ | ✓ | ✓ | 0.22 | 0.56 | 1.17 | 0.65 | 0.02 | 0.04 | 0.64 | 0.23 |
| ✓ | | | 0.21 | 0.55 | 1.18 | 0.65 | 0.02 | 0.12 | 0.70 | 0.28 |
| ✓ | ✓ | | 0.19 | 0.51 | 1.12 | 0.61 | 0.02 | 0.10 | 0.64 | 0.25 |
| ✓ | | ✓ | 0.20 | 0.52 | 1.12 | 0.61 | 0.02 | 0.10 | 0.55 | 0.22 |
| ✓ | ✓ | ✓ | 0.19 | 0.50 | 1.08 | 0.59 | 0.02 | 0.06 | 0.43 | 0.17 |

"TR-World only" optimizes the trajectory implicitly "in the manner of SSR"; "FGTR only" refines the prior trajectory on the **current** BEV map. The paper's percentages check out: without ego status −8.4% L2 and −25.8% collision; with ego status −9.2% and −39.3%.

### Table 4: World-model input and future supervision (nuScenes, with ego status, FGTR on)

| World model | Future supervision | L2 1s | L2 2s | L2 3s | **L2 Avg** | CR 1s | CR 2s | CR 3s | **CR Avg** |
|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|
| Normal world model | ✓ | 0.20 | 0.52 | 1.11 | 0.61 | 0.02 | 0.12 | 0.57 | 0.23 |
| Normal world model | ✗ | 0.20 | 0.53 | 1.11 | 0.61 | 0.02 | 0.12 | 0.49 | 0.21 |
| TR-World | ✓ | 0.19 | 0.51 | 1.12 | 0.61 | 0.02 | 0.08 | 0.53 | 0.21 |
| **TR-World** | ✗ | 0.19 | 0.50 | 1.08 | **0.59** | 0.02 | 0.06 | 0.43 | **0.17** |

"Future supervision" uses "real future data" at $t+1$. The loss form and the normal world model's input are not specified.

### Table 5: Prior against final trajectory (nuScenes, with ego status)

| Trajectory | L2 1s | L2 2s | L2 3s | **L2 Avg** | CR 1s | CR 2s | CR 3s | **CR Avg** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.21 | 0.55 | 1.18 | 0.65 | 0.02 | 0.12 | 0.70 | 0.28 |
| Prior (baseline architecture, trained with TR-World + FGTR) | 0.20 | 0.53 | 1.12 | 0.61 | 0.02 | 0.10 | **0.41** | 0.18 |
| Final | 0.19 | 0.50 | 1.08 | **0.59** | 0.02 | 0.06 | 0.43 | **0.17** |

---

## Reading the Results

### 1. A world model trained only by the planner {#no-future-loss}

| Paper | What the future latent is trained by | What adding a future target does |
|---|---|---|
| [[sources/wa-jepa.md]] | Flow matching on future latents | Regression instead of nothing: 91.1 → 90.7 EPDMS |
| [[sources/drivefuture.md]] | Planning loss only, plus training-time attention grounding to the real future (annealed away) | No future-prediction loss is tried |
| [[sources/redrive.md]] | L1 regression on future JEPA latents, action-conditioned, isolated from the planner | **Helps**: +0.7 PDMS |
| **ResWorld** | **Planning loss only** | **Hurts**: 0.17% → 0.21% collision, 0.59 → 0.61 m (TR-World); no effect on the normal branch |

- The paper's explanation: a target at $t+1$ ties the map to one instant, while the planning loss lets it hold "a future time period". That is an argument about the target's horizon, not its form; the loss form is not stated.
- **"Collapse" is asserted from one figure.** No feature-diversity statistic is given, in contrast to WA-JEPA's directional-similarity and change-magnitude measures ([[concepts/world-model-for-ad.md#objective-form]]).
- **Whether $\mathbf B_{future}$ predicts anything is never tested.** It is not compared with the real future map, and no probe reads object positions from it. "More accurate future BEV features" is inferred from planning metrics.

### 2. Most of it is training-time shaping {#training-time}

| | L2 Avg | CR Avg | Modules at inference |
|---|---:|---:|---|
| Baseline | 0.65 | 0.28 | Encoder + prior head |
| **Prior trajectory of the full model** | **0.61** | **0.18** | **Encoder + prior head** |
| Final trajectory of the full model | 0.59 | 0.17 | + TR-World + FGTR |

- Of the full improvement (0.06 m, 0.11 points), the prior head already has 0.04 m and 0.10 points. The world model and refinement stage at inference add 0.02 m and 0.01 points, and the prior is better than the final trajectory at 3 s (0.41% against 0.43%).
- The paper draws the deployment conclusion itself: train with larger TR-World and FGTR modules, then "take prior trajectories as output for higher efficiency during inference".
- This matches the wiki's training-time-only pattern ([[sources/simwam.md]], [[sources/drivevla-w0.md]]). Here the world model is trained by no world-model objective, so the shaping signal is the planning loss routed through a second stage that reads future-named features.
- **A control is missing**: a two-stage refinement trained the same way without the residual branch. Table 3's "FGTR only" row (refinement on the current map) gets 0.61 / 0.22 at the final output; its prior trajectory is not reported.

### 3. Residuals as a rate variable {#residuals}

- A residual is a first difference of pooled features after ego-motion compensation. [[sources/momworld.md]] initializes its momentum from the same kind of difference and supervises momentum against first differences of future features. [[sources/drive-hwm.md]] finds optical flow the best long-horizon target. Three papers now put an explicit motion quantity into a driving world model.
- **ResWorld's is the only matched comparison of motion-only input against full-scene input**: 0.59 / 0.17 against 0.61 / 0.21 (Table 4, both without future supervision). Small, on nuScenes, single runs.
- **What residuals cannot see.** The paper's own limitation: stationary pedestrians and parked cars give no residual and are handled only by the prior branch. Figure 5's showcase is a collision with what appears to be a parked car.
- **Alignment matters.** Static structure cancels only if past maps are warped accurately, which is why the BEV encoder was changed to GeoBEV. On NAVSIM, where history was not used for the main variant, the residuals were replaced.

### 4. NAVSIM {#navsim}

- **88.3 PDMS is not a temporal-residual result.** Agent queries from the detection head replace residuals. 89.0, with history, is the mechanism's NAVSIM number.
- **TTC 98.9 on the 88.3 row is probably a misprint** (see the residual column in Table 2).
- **Comparison scope.** The table stops at DiffusionDrive 88.1. On the wiki's v1 ladder, 89.0 is in the DriveLaW (89.1) / DriveDreamer-Policy (89.2) band, and below every scorer-cohort and RL entry above 90. See [[concepts/navsim-benchmark.md#resworld]].
- **The perception-free 87.3 is above LAW (84.6) and World4Drive (85.1)**, the two latent world models in the table. What its world model reads is not stated.
- **Camera + LiDAR**, TransFuser-style, two ResNet-34 backbones. The paper calls NAVSIM "closed-loop"; it is non-reactive and single-shot.
- No NAVSIM-v2, navhard or closed-loop benchmark.

### 5. nuScenes {#nuscenes}

- **Two metric conventions, both reported**, which is to the paper's credit. Under VAD-style averaging, 0.35 m without ego status matches BEV-Planner++ with ego status (0.35); with ego status ResWorld reaches 0.30.
- **On L2, ego status is worth as much as the whole method.** In Table 3, adding ego status to the baseline gives 0.71 → 0.65 m; adding TR-World and FGTR without ego status also gives 0.71 → 0.65 m. On collision the method is worth more (0.31 → 0.23, against 0.31 → 0.28 for ego status). The paper reads the combined result as showing the framework "prevents overfitting due to over-reliance on ego status". The table does not test that.
- **The collision deltas are tiny.** Every ablation difference in collision rate is 0.02–0.11 percentage points, averaged over three horizons. On a validation split of roughly 6,000 samples, 0.04 points is two or three samples. There are no seeds or intervals. See [[concepts/nuscenes-waymo-evals.md#resworld]].
- **Most of the collision gain is at 2 s.** Without ego status the 3 s collision rate is 0.64% for both baseline and full model; the average moves because 2 s goes from 0.25% to 0.04%.

---

## Relationships

- **[[sources/drivefuture.md]]**: also a future latent with no future-prediction loss. DriveFuture grounds it to the real future during training and anneals the grounding away; ResWorld never shows it the real future, and finds that doing so hurts.
- **[[sources/momworld.md]]**: momentum initialized from the difference of current and previous scene queries, a close relative of the temporal residual. MomWorld supervises its rollout against future features; ResWorld supervises nothing.
- **[[sources/wa-jepa.md]]** / **[[sources/redrive.md]]**: the standing disagreement on whether regression toward future latents helps. ResWorld adds a third vote on the "hurts" side, with the loss form unstated.
- **[[sources/drive-hwm.md]]**: motion (optical flow) is the best long-horizon target there; motion (feature residuals) is the better input here.
- **[[sources/simwam.md]]**, **[[sources/drivevla-w0.md]]**: world modeling as a training-time-only influence. ResWorld's Table 5 is a within-paper measurement of the same effect without any world-model loss.
- **[[sources/auto-jepa.md]]**: argues agent-relevance emerges from an ego-motion target. ResWorld builds agent emphasis in by construction (only moving content enters the world model) and loses parked agents as a result.
- **[[sources/foresight.md]]**: another world-model paper with a VAD-style nuScenes table. Its LAW row (0.61 / 0.30) matches ResWorld's; its World4Drive row (0.50 / 0.16) is absent from ResWorld's nuScenes table, although World4Drive appears in ResWorld's NAVSIM table.
- **[[sources/diffusiondrive.md]]**: the strongest baseline in both of ResWorld's tables.

---

## Limitations

**Evidence**

1. **No measurement of the world model as a predictor.** The future map is never compared with the real future or probed.
2. **Collapse prevention is shown by one figure**, whose caption contradicts its own labels.
3. **The NAVSIM headline (88.3) does not use temporal residuals**, and its TTC is probably misprinted.
4. **Small effects without error bars.** nuScenes deltas of 0.02–0.06 m and 0.02–0.11 collision points, single runs.
5. **Missing controls**: refinement on the current map with its prior trajectory reported; the normal world model's input; the loss form of "future supervision".
6. **Comparison scope.** NAVSIM stops at DiffusionDrive 88.1; nothing from the 90+ range. No navhard, NAVSIM-v2 or closed-loop benchmark.

**Method**

7. **Parked cars and standing pedestrians are invisible to residuals** (stated by the paper).
8. **Residuals depend on ego-motion alignment and BEV geometric quality**; misalignment would turn static structure into apparent motion. Not analysed.
9. **The future map has no defined horizon** and no ego motion; it is the current map plus a dynamic term in the current frame.

**Not reported**: latency, parameter count, the perception-free NAVSIM variant's world-model input, seeds.

**Source conversion**: five figure files in `raw/assets/`; Figure 1 has only panel (a). All five tables reproduced. The clipping's front-matter author field is empty; authors and affiliations are taken from the body.

---

## Key Cross-References

- [[concepts/world-model-for-ad.md]] — Pattern 45: a residual-only world model trained by the planning loss alone ([#residual-world-model](../concepts/world-model-for-ad.md#residual-world-model)).
- [[concepts/navsim-benchmark.md]] — 89.0 / 88.3 / 87.3 PDMS; a closed-form check that flags one TTC.
- [[concepts/nuscenes-waymo-evals.md]] — two metric conventions, ego status, and collision deltas of a few samples.
- [[concepts/evaluation-variance.md]] — ablation effects below the sample granularity.
- [[concepts/perception-for-planning.md]] — 87.3 without perception supervision against 88.3 / 89.0 with it.

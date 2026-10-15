---
title: "LAW: Enhancing End-to-End Autonomous Driving with Latent World Model"
type: source-summary
sources: ["raw/papers/Enhancing End-to-End Autonomous Driving with Latent World Model.md"]
related: [sources/worldrft.md, sources/epona.md, sources/drivefuture.md, sources/dynvla.md, sources/latent-wam.md, sources/dreameraD.md, sources/foresight.md, sources/resworld.md, sources/deepsight.md, sources/drive-jepa.md, sources/redrive.md, sources/momworld.md, sources/da-wam.md, sources/wa-jepa.md, sources/foredrive.md, sources/world4scorer.md, sources/drive-hwm.md, sources/raw2drive.md, sources/think2drive.md, sources/policy-world-model.md, sources/metis.md, sources/drivewam.md, concepts/world-model-for-ad.md, concepts/navsim-benchmark.md, concepts/nuscenes-waymo-evals.md, concepts/perception-for-planning.md, concepts/evaluation-variance.md, concepts/teacher-pseudo-labels.md, concepts/inference-latency.md, research-directions.md]
created: 2026-10-15
updated: 2026-10-15
confidence: medium
---

# LAW: Enhancing End-to-End Autonomous Driving with Latent World Model

**Authors**: Yingyan Li, Lue Fan, Jiawei He, Yuqi Wang, Yuntao Chen, Zhaoxiang Zhang, Tieniu Tan
**Orgs**: Institute of Automation, Chinese Academy of Sciences (NLPR; State Key Laboratory of Multimodal AI Systems); School of Future Technology, UCAS
**arXiv**: 2406.08481v2. Version 1 (June 2024) is a shorter paper. ICLR 2025 per [[sources/worldrft.md]]'s reference list; the clipping names no venue.
**Code**: github.com/BraveGroup/LAW
**Source**: `raw/papers/Enhancing End-to-End Autonomous Driving with Latent World Model.md`. The clipping has all six figures and nine tables. Tables 1 and 4–9 are HTML and are rebuilt below. ⚠ `raw/assets/vis_supp_2.png` is **truncated**: rows 1,726–2,457 of 2,457 do not decode, so the second scene's VAD panel is missing.

**What the two versions contain.** v1 reports only the perception-free model on nuScenes (0.61 m / 0.30%, with **19.5 FPS**) and CARLA Town05 Long (70.1 DS); it has **no NAVSIM result**. v2, the version here, adds the perception-based model, NAVSIM and the appendix and drops the latency columns. (v1 was checked against its arXiv HTML during this ingest.) [[sources/momworld.md]]'s FPS column carries LAW at 19.5 on an RTX 3090.

**Lineage.** This is the root of the wiki's training-time latent world-model line.
- [[sources/worldrft.md]] keeps its world decoder and loss and calls them "LAW's". World4Drive, not ingested, does the same.
- [[sources/drivefuture.md]] names LAW as the work it succeeds, and LAW's first author is one of its three equal-contribution first authors.
- [[sources/dynvla.md]] shares five of its authors (Yingyan Li, Yuqi Wang, Lue Fan, Zhaoxiang Zhang, Tieniu Tan).
- LAW is a NAVSIM-v1 baseline on 26 ingested source pages, 24 of them at 84.6.

**Confidence: medium.** Every nuScenes average reproduces from its per-horizon values, and the NAVSIM rows pass the closed-form check. The ablations are well designed: an action-free control, a horizon sweep and an architecture sweep. Against that:
- The nuScenes ablation differences are a few samples on single runs.
- The CARLA world-model effect (+2.2 DS) is inside its own reported run spread.
- The NAVSIM gain (+7.1) is measured on a weak base (77.5).
- Whether the future latents are stop-gradient targets is not stated, and how the camera inputs are set up on NAVSIM is not stated either.

---

## What It Is

**An auxiliary loss, not a planner component.** An MLP fuses the current scene latents with the planner's own predicted waypoints. A small transformer then predicts the next frame's latents, and an MSE loss compares them with the encoder's latents for the real next frame. The world model is used only for training; nothing reads its prediction at inference.

- **Action-aware latents.** The $M$ predicted waypoints are flattened to a $2M$ vector, concatenated onto every latent vector, and mapped back to the latent width by an MLP.
- **Latent world model.** Transformer blocks (self-attention + FFN) across the latent vectors; on BEV latents, deformable self-attention. Output: $\hat V_{t+1}$.
- **Loss.** $\mathcal L_\text{latent}=\frac1L\sum_i\|\hat v^i_{t+1}-v^i_{t+1}\|_2$, written as an L2 norm and called MSE. The paper says the task "jointly optimizes the current scene feature learning and ego trajectory prediction", and Figure 1 draws the supervision flowing back through the world model into both the latents and the predicted trajectory. **So the trajectory is not detached.** Whether $V_{t+1}$ is detached, or comes from an EMA encoder, is not stated.
- **Perception-free framework.** Image backbone plus PETR-style 3D position embeddings. Then one learnable query per camera cross-attends to that camera's features, so the latents are **one vector per view** as written (Figure 2 draws six). $M$ waypoint queries cross-attend to them and an MLP decodes waypoints (L1). Loss: $\mathcal L_\text{latent}+\mathcal L_\text{waypoint}$.
- **Perception-based framework.** A VAD-style model: a BEVFormer encoder, then agent and map queries for motion prediction and map construction, then waypoint queries over agent and map features. Loss adds VAD's perception terms. Training has two stages: 48 epochs of perception, then 12 with the waypoint and latent losses.
- **Configurations.**

  | Benchmark | Framework | Backbone | Input | Training |
  |---|---|---|---|---|
  | nuScenes | Perception-free | Swin-T | Six cameras at 800×320 | 12 epochs, 8 × A6000, batch 8, lr 5e-5 |
  | nuScenes | Perception-based | VAD's | Six cameras | 48 + 12 epochs |
  | NAVSIM | Perception-free | ResNet-34 | Images at 640×320; camera count not stated | 20 epochs, batch 32, lr 1e-4 |
  | CARLA (Town05 Long, 0.9.10.1) | Perception-free | ResNet-34 with TCP's head | 900×256 | 60 epochs, batch 128 |

  The CARLA training data is 189K frames from the privileged Roach expert, following TCP. nuScenes uses no ego-status history.
- **Default horizon**: the next keyframe, 0.5 s ahead, although 1.5 s scores better in Table 6.

![[fig1 2.png|Figure 1: (a) annotation-based perception tasks used to supervise scene features (occupancy prediction, motion prediction, depth estimation, detection, tracking, map segmentation), drawn as an optional input. (b) the self-supervised task: at time T, the encoder produces visual latents and the decoder a predicted trajectory; both go into a latent world model whose predicted latents are supervised by the encoder's latents of frame T+1. Dotted green arrows run back from the supervision through the world model to the latents and the trajectory.]]

**Figure 1** is the paper's argument: replace perception labels with a temporal target. The dotted arrows show the latent loss training the trajectory as well as the encoder.

![[Pipeline 3.png|Figure 2: overall framework. Top: six images at T go through an encoder to visual latents and a decoder to waypoints; latents and waypoints form action-aware latents, the latent world model predicts latents for T+1, supervised by the encoder's latents from six images at T+1. Bottom left: the perception-free framework, a perspective-view encoder producing six visual latents and a waypoint decoder with waypoint queries. Bottom right: the perception-based framework, a BEV encoder whose flattened BEV feature feeds motion-prediction and map-construction heads (agent and map queries, supervised), then a waypoint decoder.]]

**Figure 2**: in the perception-free branch the six visual latents are what the world model predicts.

| Benchmark | LAW | Without latent prediction | Notes |
|---|---:|---:|---|
| nuScenes, perception-free (L2 m / collision %) | 0.61 / 0.30 | 0.71 / 0.41 | Temporal-average (VAD) collision; no ego status |
| nuScenes, perception-based | **0.49 / 0.19** | 0.54 / 0.25 | VAD-style perception labels |
| NAVSIM-v1 PDMS (navtest) | **84.6** | 77.5 | Perception-free, ResNet-34 |
| CARLA Town05 Long DS | **70.1 ± 2.6** | 67.9 ± 2.1 | TCP head; Roach data |
| Latency | 19.5 FPS (v1 only) | – | RTX 3090 per [[sources/momworld.md]]'s table |

---

## Key Takeaways

- **Conditioning on the action is most of the effect.** Table 4 has the control most world-model papers skip: the same world model with the trajectory input removed.

  | Framework | No world model | Action-free prediction | Action-conditioned prediction |
  |---|---|---|---|
  | Perception-free | 0.71 / 0.41 | 0.68 / 0.37 | **0.61 / 0.30** |
  | Perception-based | 0.54 / 0.25 | 0.52 / 0.21 | **0.49 / 0.19** |

  - Predicting the future without the action gives about a third of the L2 gain.
  - **What the action input adds may be partly a second imitation loss.** The trajectory input is the planner's own prediction, it is not detached, and the target is the logged next frame, which was produced by the logged ego motion. So minimizing the latent loss also pulls the predicted trajectory toward the logged one, through the world model. The paper's wording ("jointly optimizes ... ego trajectory prediction") is consistent with that. No run detaches the trajectory to separate a representation effect from a trajectory-supervision effect. [[sources/redrive.md]]'s Stage 3 uses a frozen predictor as an explicit critic of the planner; LAW's version is implicit and trained jointly.
  - On a split of about 6,000 samples, each collision step in this table is one to four samples ([[concepts/nuscenes-waymo-evals.md#law]]).
- **On NAVSIM the latent loss is worth +7.1 PDMS on a weak base.** Without it the perception-free model scores 77.5 (NC 94.4, DAC 89.4, TTC 84.8, EP 75.1). With it the model reaches 84.6, above the camera-only, perception-supervised LTF (83.8 in [[sources/worldrft.md]]'s table) and TransFuser (84.0, camera + LiDAR). That is among the largest world-model-loss effects in the wiki (Epona loses 8.1 without joint video training, [[sources/epona.md]]), against +0.1 to +0.95 for auxiliary future losses on strong bases ([[sources/foredrive.md]], [[sources/redrive.md]], [[sources/world4scorer.md]]). It fits a regularizer on an under-supervised model: one latent per camera, a single frame, no perception labels. The gain is mostly DAC (+6.0) and EP (+6.6). TTC (88.7) stays the lowest of the learned planners in its table.
- **Regression on future latents helps, and gets worse with horizon.** Table 6:
  - 0.5 s ahead: 0.61 / 0.30. 1.5 s: **0.58 / 0.25**. 3 s: 0.63 / 0.27. **10 s: 0.72 / 0.43**, slightly worse than no world model (0.71 / 0.41).
  - Few papers here sweep the horizon of a regression latent target. This sweep fits the entropy reading of [[sources/wa-jepa.md]] (regression fails as the target gets more uncertain), on a target of six vectors.
  - The 10 s target exists only for keyframes at least 10 s before the end of a 20 s nuScenes scene. How the missing targets are handled, and so how much less data the 10 s run sees, is not stated.
  - **The headline uses 0.5 s**, not the better 1.5 s.
- **The world model needs to mix tokens.** Table 7: linear projection 0.70 / 0.42 (no better than no world model), two-layer MLP 0.64 / 0.33, transformer 0.61 / 0.30.
- **History adds more than the world model does.** Appendix Tables 8–9, on the **front camera only**:
  - Predicting 1.5 s and then 3 s autoregressively improves on 1.5 s alone (0.73 / 0.32 → 0.69 / 0.29).
  - Adding one history frame (−1.5 s) as input then takes it from 0.68 / 0.33 to **0.55 / 0.17**. That beats the six-camera, single-frame headline, but changes the input, not only the world model.
- **CARLA: the latent loss is inside the run spread.** 67.9 ± 2.1 → 70.1 ± 2.6 DS, with route completion down (98.6 → 97.8) and the infraction score up (0.68 → 0.72). Most of LAW's margin over TCP (57.2) belongs to its re-implementation: without the latent loss it already scores 67.9, above DriveAdapter (65.9) and ThinkTwice (65.0), the two "surpassed" methods the text names. It is level with Interfuser (68.3).
- **Perception labels still help on nuScenes.** Perception-based 0.49 / 0.19 against perception-free 0.61 / 0.30, and the latent loss helps both (−0.05 m / −0.06 points and −0.10 m / −0.11 points). On NAVSIM only the perception-free model is run ([[concepts/perception-for-planning.md#law]]).
- **How later tables carry it.**
  - **NAVSIM 84.6** (96.4 / 95.4 / 88.7 / 99.9 / 81.7) is printed identically on 24 ingested source pages. Camera labels vary: "1×C" in [[sources/metis.md]], [[sources/auto-jepa.md]], [[sources/hyworldvla.md]] and [[sources/drive-hwm.md]]; "SV" in [[sources/drivewam.md]]; "C" elsewhere. The paper states only a 640×320 input.
  - **A second LAW row, 83.8 PDMS labelled "C & L"** (NC 97.4, DAC 93.3, EP 78.8, TTC 91.9), appears in [[sources/drive-jepa.md]] and [[sources/redrive.md]]. It is in neither version of LAW's paper: v1 has no NAVSIM, and v2 has 84.6 with cameras only. It passes the closed-form check (−4.02), so it is a plausible row from somewhere, but not LAW's published one ([[concepts/navsim-benchmark.md#law]]).
  - **nuScenes 0.61 / 0.30 and 0.49 / 0.19** match [[sources/foresight.md]], [[sources/resworld.md]], [[sources/policy-world-model.md]] and [[sources/worldrft.md]]. [[sources/deepsight.md]] puts the 0.61 / 0.30 row in a table captioned "with ego status"; LAW states it uses none.
- **The DA-WAM taxonomy misfiles it.** [[concepts/world-model-for-ad.md]] reproduces [[sources/da-wam.md]]'s table, which lists LAW under "loosely coupled latent fusion", where the future reaches the scorer. In LAW nothing reads the predicted latent at inference; it is a training target only, as [[sources/physwam.md]]'s table has it.

---

## Results

### Table 1: nuScenes open-loop planning (no ego status)

Collision averages use VAD's computation; ‡ = BEV-Planner's own collision computation.

| Method | L2 1s | 2s | 3s | Avg. | Col. 1s | 2s | 3s | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NMP | – | – | 2.31 | – | – | – | 1.92 | – |
| SA-NMP | – | – | 2.05 | – | – | – | 1.59 | – |
| FF | 0.55 | 1.20 | 2.54 | 1.43 | 0.06 | 0.17 | 1.07 | 0.43 |
| EO | 0.67 | 1.36 | 2.78 | 1.60 | 0.04 | 0.09 | 0.88 | 0.33 |
| ST-P3 | 1.33 | 2.11 | 2.90 | 2.11 | 0.23 | 0.62 | 1.27 | 0.71 |
| UniAD | 0.48 | 0.96 | 1.65 | 1.03 | 0.05 | 0.17 | 0.71 | 0.31 |
| VAD | 0.41 | 0.70 | 1.05 | 0.72 | 0.07 | 0.17 | 0.41 | 0.22 |
| BEV-Planner | 0.30 | 0.52 | 0.83 | 0.55 | 0.10‡ | 0.37‡ | 1.30‡ | 0.59‡ |
| LAW (perception-free) | 0.26 | 0.57 | 1.01 | 0.61 | 0.14 | 0.21 | 0.54 | 0.30 |
| **LAW (perception-based)** | **0.24** | **0.46** | **0.76** | **0.49** | 0.08 | **0.10** | **0.39** | **0.19** |

- All averages reproduce from their horizons.
- **The table crosses conventions.** UniAD's 1.03 / 0.31 is its per-horizon value; in the temporal-average convention LAW's own rows use, UniAD is 0.69 / 0.12 ([[concepts/nuscenes-waymo-evals.md]]), which is below both LAW collision averages. "State-of-the-art results in both L2 displacement and collision rates" holds against the rows as printed. [[sources/worldrft.md]]'s nuScenes table, from a later paper in the same line, has the same layout.
- The perception-free model is 0.06 m behind BEV-Planner on L2, and its collision rate cannot be compared with BEV-Planner's because they are computed differently.

### Table 2: NAVSIM-v1 navtest (perception-free)

Residual = closed form NC·DAC·(5 TTC + 2 C + 5 EP)/12 − reported PDMS ([[concepts/navsim-benchmark.md#residual-sign]]).

| Method | NC | DAC | TTC | Comf. | EP | PDMS | Residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| Human | 100 | 100 | 100 | 99.9 | 87.5 | 94.8 | −0.03 |
| Constant Velocity | 69.9 | 58.8 | 49.3 | 100 | 49.3 | 21.6 | +2.14 |
| Ego Status MLP | 93.0 | 77.3 | 83.6 | 100 | 62.8 | 65.6 | −9.77 |
| TransFuser | 97.7 | 92.8 | 92.8 | 100 | 79.2 | 84.0 | −3.91 |
| UniAD | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 | −4.12 |
| PARA-Drive | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 | −4.01 |
| **LAW** | 96.4 | **95.4** | 88.7 | 99.9 | **81.7** | **84.6** | −3.99 |

- The planner rows sit in the usual v1 band; Ego-MLP's −9.77 matches [[sources/clover.md]] and [[sources/drivor.md]]. Constant velocity's positive residual is NAVSIM's own baseline row.
- "State-of-the-art results in overall PDMS" is against NAVSIM's own baselines. LAW is the table's lowest learned planner on NC and TTC.

### Table 3: CARLA Town05 Long (mean ± sd over runs)

| Method | Supervision | RC | IS | DS |
|---|---|---:|---:|---:|
| CILRS | Expert | 10.3 ± 0.0 | 0.75 ± 0.05 | 7.8 ± 0.3 |
| LBC | Expert | 31.9 ± 2.2 | 0.66 ± 0.02 | 12.3 ± 2.0 |
| Transfuser | Expert, Dep., Seg., Map., Det. | 47.5 ± 5.3 | 0.77 ± 0.04 | 31.0 ± 3.6 |
| Roach | Expert | 96.4 ± 2.1 | 0.43 ± 0.03 | 41.6 ± 1.8 |
| LAV | Expert, Seg., Map., Det. | 69.8 ± 2.3 | 0.73 ± 0.02 | 46.5 ± 2.3 |
| TCP | Expert | 80.4 ± 1.5 | 0.73 ± 0.02 | 57.2 ± 1.5 |
| MILE | Expert, Map., Det. | 97.4 ± 0.8 | 0.63 ± 0.03 | 61.1 ± 3.2 |
| ThinkTwice | Expert, Dep., Seg., Det. | 95.5 ± 2.0 | 0.69 ± 0.05 | 65.0 ± 1.7 |
| DriveAdapter | Expert, Map., Det. | 94.4 | 0.72 | 65.9 |
| Interfuser | Expert, Map., Det. | 95.0 ± 2.9 | – | 68.3 ± 1.9 |
| **LAW** | Expert, latent prediction | **97.8 ± 0.9** | 0.72 ± 0.03 | **70.1 ± 2.6** |

- The number of runs is not stated. LAW and Interfuser are 1.8 apart, inside either one's spread.
- Town05 Long is the CARLA 0.9.10 benchmark, not the CARLA v2 Bench2Drive on [[concepts/bench2drive.md]]. [[sources/raw2drive.md]]'s table also lists LAW as a CARLA v1 method.

### Table 4: what the world model reads (nuScenes)

| Framework | Visual latent | Predicted trajectory | L2 1s | 2s | 3s | Avg. | Col. 1s | 2s | 3s | Avg. |
|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Perception-free | | | 0.32 | 0.67 | 1.14 | 0.71 | 0.20 | 0.30 | 0.73 | 0.41 |
| | ✓ | | 0.30 | 0.64 | 1.12 | 0.68 | 0.18 | 0.27 | 0.66 | 0.37 |
| | ✓ | ✓ | **0.26** | **0.57** | **1.01** | **0.61** | **0.14** | **0.21** | **0.54** | **0.30** |
| Perception-based | | | 0.30 | 0.52 | 0.80 | 0.54 | 0.09 | 0.17 | 0.48 | 0.25 |
| | ✓ | | 0.27 | 0.49 | 0.80 | 0.52 | 0.08 | 0.12 | 0.42 | 0.21 |
| | ✓ | ✓ | **0.24** | **0.46** | **0.76** | **0.49** | **0.08** | **0.10** | **0.39** | **0.19** |

"No input" means no world model at all. The action-free perception-free average (0.687) is printed truncated as 0.68.

### Table 5: latent prediction on NAVSIM and CARLA

| Latent prediction | NC | DAC | TTC | Comf. | EP | PDMS | RC | IS | DS |
|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| × | 94.4 | 89.4 | 84.8 | 100.0 | 75.1 | 77.5 | 98.6 ± 0.8 | 0.68 ± 0.02 | 67.9 ± 2.1 |
| ✓ | **96.4** | **95.4** | **88.7** | 99.9 | **81.7** | **84.6** | 97.8 ± 0.9 | **0.72 ± 0.03** | **70.1 ± 2.6** |

The NAVSIM base's residual is −7.21, outside the usual planner band (−1.7 to −6.0), as weak planners' residuals tend to be (Ego-MLP −9.77). Its sub-scores are internally consistent.

### Table 6: prediction horizon (nuScenes, perception-free)

| Horizon | L2 1s | 2s | 3s | Avg. | Col. 1s | 2s | 3s | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 s (headline) | 0.26 | 0.57 | 1.01 | 0.61 | 0.14 | 0.21 | 0.54 | 0.30 |
| **1.5 s** | 0.26 | **0.54** | **0.93** | **0.58** | 0.14 | **0.17** | **0.45** | **0.25** |
| 3.0 s | 0.28 | 0.59 | 1.01 | 0.63 | **0.13** | 0.20 | 0.48 | 0.27 |
| 10.0 s | 0.33 | 0.67 | 1.16 | 0.72 | 0.24 | 0.28 | 0.76 | 0.43 |

### Table 7: world-model architecture (nuScenes, perception-free)

| Architecture | L2 Avg. | Col. Avg. |
|---|---:|---:|
| Linear projection | 0.70 | 0.42 |
| Two-layer MLP | 0.64 | 0.33 |
| **Transformer blocks** | **0.61** | **0.30** |

Per-horizon: linear 0.31 / 0.65 / 1.14 and 0.26 / 0.34 / 0.66; MLP 0.27 / 0.58 / 1.07 and 0.17 / 0.23 / 0.59.

### Table 8: autoregressive multi-step prediction (front camera only)

| Predicted future | L2 1s | 2s | 3s | Avg. | Col. 1s | 2s | 3s | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.5 s | 0.34 | 0.69 | 1.17 | 0.73 | 0.12 | 0.22 | 0.63 | 0.32 |
| 1.5 s → 3 s | **0.31** | **0.65** | **1.12** | **0.69** | **0.11** | **0.19** | **0.57** | **0.29** |

### Table 9: history frames as world-model input (front camera only)

| Predicted future | Input frames | L2 1s | 2s | 3s | Avg. | Col. 1s | 2s | 3s | Avg. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.5 s → 3 s | 0 s | 0.30 | 0.64 | 1.09 | 0.68 | 0.14 | 0.23 | 0.62 | 0.33 |
| 1.5 s → 3 s | −1.5 s, 0 s | **0.26** | **0.51** | **0.87** | **0.55** | **0.08** | **0.09** | **0.33** | **0.17** |

The first row is Table 8's model fine-tuned 6 more epochs on single frames. Its collision rate gets worse (0.29 → 0.33), so the extra fine-tuning alone does not help. The history input in the second row reaches the planner too, not only the world model, so the +0.13 m is not a world-model effect.

---

## Qualitative Figures

All four compare perception-based LAW with VAD on nuScenes. Each panel has six cameras and a BEV of predicted map, agents and the ego plan. No counts are given.

![[vis_main_comparison.png|Figure 3: one scene, LAW (top) and VAD (bottom), command "Go Straight". A dark hatchback is directly ahead in the front camera. LAW's BEV shows boxes for the vehicles ahead (red circle) with short forecasts; VAD's BEV has no box there, and its ego plan (yellow circle in the front camera) runs forward into the hatchback.]]

**Figure 3**: VAD misses the car ahead and plans into it; LAW detects it.

![[vis_supp_1.png|Figure 4: two scenes, LAW above VAD in each. First, a wet road: LAW's map has straight, continuous lane dividers and road boundaries; VAD's boundaries are bent and broken. Second, a multi-lane road: LAW's map includes the left-hand lanes with vehicles on them, which VAD's map lacks.]]

**Figure 4**: map construction.

![[vis_supp_2.png|Figure 5 (file truncated in raw/assets): first scene, command "Turn Right": LAW's BEV shows a motorcycle behind the ego (circled, also visible in the back camera) with a forecast, and VAD's BEV shows nothing there. Second scene: a pedestrian on the left pavement, circled in the front-left camera, appears as an agent in LAW's BEV; the VAD panel for this scene is cut off in the stored file.]]

**Figure 5**: agent motion prediction. The stored image is truncated; the second scene's VAD panel is lost.

![[vis_supp_3.png|Figure 6: two scenes, LAW above VAD. First, dense traffic behind a box truck: LAW's BEV has a cluster of vehicle boxes with forecasts; VAD's has fewer, with a misplaced map element ahead. Second, a street lined with parked cars on the left: LAW's BEV shows about ten parked-car boxes, VAD's one.]]

**Figure 6**: occluded and crowded scenes. In every figure LAW looks more complete. They show the perception-based model, where the latent loss is worth 0.05 m and 0.06 collision points; the perception-free model, where it matters most, is not visualized.

---

## Table and Text Audit

| Check | Result |
|---|---|
| nuScenes averages, Tables 1 and 4–9 | All reproduce; three are truncated rather than rounded (0.687 → 0.68) |
| NAVSIM closed form | Planner rows −3.91 to −4.12; LAW −3.99; base without latent loss −7.21 |
| "State-of-the-art ... in both L2 and collision" (nuScenes, perception-based) | Against rows as printed; UniAD is printed per-horizon (1.03 / 0.31), while its temporal-average 0.69 / 0.12 beats LAW's collision |
| "Our perception-free framework demonstrates competitive performance" (nuScenes) | L2 behind BEV-Planner; collision behind VAD (0.30 against 0.22) |
| "State-of-the-art results in overall PDMS" (NAVSIM) | Against NAVSIM's own baselines; lowest NC and TTC of the learned planners |
| "Outperforms all existing methods ... surpasses ThinkTwice and DriveAdapter" (CARLA) | Its own base without the latent loss already does (67.9); the latent loss adds 2.2, inside the spread |
| "The model achieves the best performance at the 1.5-second horizon" | True; the headline uses 0.5 s |
| "Mean Squared Error" | Written as an unsquared L2 norm |
| Stop-gradient on targets | Not stated |
| Cameras and ego status on NAVSIM | Not stated |
| Number of CARLA runs | Not stated |
| Figure 5 file | Truncated in `raw/assets/` (bottom 30% missing) |
| Second LAW NAVSIM row (83.8, "C & L") in [[sources/drive-jepa.md]], [[sources/redrive.md]] | In neither version of this paper |

---

## Limitations

1. **The world-model effect and a trajectory-supervision effect are not separated.** The action input is the undetached predicted trajectory and the target is the logged next frame, so the latent loss also trains the planner. There is no detached-trajectory arm.
2. **The NAVSIM gain is measured on a weak base** (77.5, perception-free, single frame). The size of the effect on a strong planner is not tested here; later papers find +0.1 to +0.95.
3. **The closed-loop gain is within noise** (+2.2 DS against sd 2.1–2.6, run count unstated) on a CARLA v1 benchmark.
4. **Few-sample nuScenes ablations**, single runs.
5. **Target construction is unstated**: whether $V_{t+1}$ is stop-gradient or EMA, and how the 10 s target handles scene ends.
6. **NAVSIM inputs are unstated** (cameras, ego status), and later tables disagree about them.
7. **The best measured configurations are not the headline**: 1.5 s horizon, autoregressive multi-step and history input are all better and appear only in ablations, partly on a front-camera model.

---

## Relation to Other Wiki Pages

- [[sources/worldrft.md]]: keeps LAW's world decoder and loss and never ablates them. LAW's own ablation is the one that prices the loss.
- [[sources/drivefuture.md]]: the same first author's successor, which drops the future-prediction loss altogether and uses the future latent as a condition instead.
- [[sources/latent-wam.md]], [[sources/dreameraD.md]]: the CASIA group around WorldRFT and World4Drive, building on LAW's line with geometry teachers and RL in latent rollouts.
- [[sources/wa-jepa.md]], [[sources/redrive.md]], [[sources/foredrive.md]], [[sources/world4scorer.md]]: later measurements of regression on future latents (harmful at multi-view scene level in WA-JEPA; helpful when action-conditioned in ReDrive; small auxiliary effects in ForeDrive and World4Scorer).
- [[sources/drive-hwm.md]]: the other horizon-dependent result (optical flow far, RGB near), with a different target and objective.
- [[sources/resworld.md]]: a world model trained by the planning loss alone, the opposite limiting case to LAW's MSE target.
- [[sources/da-wam.md]]: per-candidate futures; its taxonomy lists LAW under fusion at inference, which LAW does not do.
- Not ingested: **World4Drive** (LAW + semantic pseudo-labels + intention; 85.1), **VAD** (the perception-based base), **TCP** (the CARLA head and data protocol), **BEV-Planner**, **SSR**.

---

## Open Questions

- **Is the action-conditioned gain representation or trajectory supervision?** Rerun Table 4's last row with the predicted trajectory detached before it enters the world model, and once with the logged trajectory as input.
- **Does the latent loss still pay on a strong base?** The +7.1 on NAVSIM is from 77.5. The same loss on a 88+ planner, or with LAW's better 1.5 s horizon, would place it beside the later +0.1 to +0.95 results.
- **Does the horizon optimum move with the target's entropy?** LAW's sweep uses six vectors; the same sweep on dense BEV latents (its perception-based setting) would say whether the optimum is a property of the horizon or of the target's size.
- **Are the targets stop-gradient?** An MSE target produced by the same encoder can drift toward easy-to-predict features. The paper does not say how it avoids that, and no feature statistic is reported.

---
title: "DVGT-2: Vision-Geometry-Action Model for Autonomous Driving at Scale"
type: source-summary
sources: ["raw/papers/DVGT-2_ Vision-Geometry-Action Model for Autonomous Driving at Scale.md"]
related: [sources/geowam.md, sources/geoworldad.md, sources/physwam.md, sources/unified-driving-tokens.md, sources/diffusiondrive.md, sources/drivor.md, sources/drive-hwm.md, sources/drivevla-w0.md, sources/drivesuprim.md, sources/world4scorer.md, sources/latent-wam.md, concepts/navsim-benchmark.md, concepts/navhard-ood-evaluation.md, concepts/nuscenes-waymo-evals.md, concepts/perception-for-planning.md, concepts/foundation-backbones-for-ad.md, concepts/teacher-pseudo-labels.md, concepts/inference-latency.md, concepts/diffusion-planner.md, concepts/world-model-for-ad.md, research-directions.md]
created: 2026-10-14
updated: 2026-10-14
confidence: medium
---

# DVGT-2: Vision-Geometry-Action Model for Autonomous Driving at Scale

**Paper**: DVGT-2: Vision-Geometry-Action Model for Autonomous Driving at Scale
**Authors**: Sicheng Zuo, Zixun Xie, Wenzhao Zheng, Shaoqing Xu, Fang Li, Hanbing Li, Long Chen, Zhi-Xin Yang, Jiwen Lu
**Orgs**: Tsinghua University, Xiaomi EV, University of Macau, Peking University
**arXiv**: 2604.00813v3 (v1 April 2026)
**Code**: github.com/wzzheng/DVGT; project page wzzheng.net/DVGT-2
**Source**: `raw/papers/DVGT-2_ Vision-Geometry-Action Model for Autonomous Driving at Scale.md`. The clipping has all text and all eleven tables. Figure 7 (memory and latency plots) is missing: the arXiv HTML serves it only as SVG, so it was cropped from page 15 of the arXiv PDF (`raw/assets/dvgt2_fig7.png`). Figure A.1 was a remote link and was downloaded from the arXiv HTML (`raw/assets/dvgt2_teaser_supp.png`).

**Confidence: medium.** The planning numbers are well corroborated: three later papers ([[sources/geowam.md]], [[sources/geoworldad.md]], [[sources/physwam.md]]) print DVGT-2's rows with identical sub-scores, and code is released. But the paper's central claim, that dense geometry is the right intermediate for planning, has **no planning ablation at all**. Its v2 "outperforms all SOTA" compares its own corrected-like rows with pre-fix baselines. The geometry comparisons set a model trained on these datasets against zero-shot ones, without naming the evaluation ground truth. Several numbers disagree between tables and figures.

---

## What It Is

**One 1.8B-parameter transformer that, at each frame, reconstructs metric 3D point maps for every camera, estimates the ego pose relative to the previous frame, and plans the future trajectory, while caching a fixed window of past frames instead of reprocessing them.** The paper calls the idea Vision-Geometry-Action (VGA): dense 3D geometry as the intermediate between pixels and action, in place of sparse perception (boxes, maps) or language.

1. **Encoder.** DINOv3 ViT-L over each camera image. One pose token and eight trajectory tokens are appended to each view's visual tokens.
2. **Geometry transformer.** 24 blocks (1024-d, 16 heads), each with intra-view attention, cross-view attention over the current frame's views, and **temporal causal attention** from the current frame to a cache of the past $W$ frames' per-layer features. Relative temporal positions (MRoPE-I) keep cached features valid as the window slides, so nothing is recomputed. The cache is first-in, first-out with $W=4$.
3. **Heads.** A DPT head decodes point maps in the **current ego frame**. The pose and trajectory tokens are pooled across views and decoded by two **DiffusionDrive-style anchor-based truncated-diffusion heads** with 20 anchors each (pose: two denoising steps). Ego status (velocity, acceleration, driving command) is added to the trajectory token.
4. **Training.** Geometry pre-training without streaming (160K iterations), then planning plus streaming (80K), on a mixture of nuScenes, OpenScene, Waymo, KITTI and DDAD at 2 Hz, with depth labels from **MoGe-2** after threshold filtering. A further 40K iterations on NAVSIM (8 views, 4 frames) gives **DVGT-2-NAVSIM**. About ten days on 64 H20 GPUs in total.

**Headline results.**

| Setting | DVGT-2 (general) | DVGT-2-NAVSIM |
|---|---:|---:|
| NAVSIM-v1 navtest PDMS | 88.6 | **90.3** |
| NAVSIM-v2 navtest EPDMS | 88.9 | **89.6** |
| nuScenes open-loop L2 / collision (ST-P3 averaging) | 0.78 m / 0.19% | – |
| nuScenes open-loop L2 / collision (VAD averaging) | 0.41 m / 0.22% | – |
| Online latency per frame (8 views, $W=4$) | about 0.26–0.27 s, about 70 GB | – |

---

## Key Takeaways

- **The rows the wiki has carried as "DVGT-2" are DVGT-2-NAVSIM.** GeoWAM's and GeoWorldAD's DVGT-2 rows (90.3 PDMS, 89.6 EPDMS) match DVGT-2-NAVSIM's sub-scores digit for digit. The general model is 88.6 / 88.9. [[sources/physwam.md]] had already printed both; this paper confirms them.
- **DVGT-2's trajectory head samples; it is not deterministic.** It is DiffusionDrive's truncated diffusion with 20 clustered anchors. The wiki had described GeoWAM's comparison with DVGT-2 as holding the head fixed ("DVGT-2, which also uses a deterministic head"), which is wrong. GeoWAM replaces this head with a deterministic regressor *and* adds future forecasting, so its +0.6 navtest / +4.9 navhard over DVGT-2 prices the two changes together.
- **The VGA thesis is asserted, not ablated.** No run removes the geometry loss, skips geometry pre-training, or swaps the geometry target for boxes or language under the same model. The paradigm comparison in Figure 2 and Tables 5–7 is across papers with different backbones, data, sensors and training. The nearest controlled evidence in the wiki is [[sources/geoworldad.md]]'s: a geometry-pretrained encoder 89.3 PDMS against 84.2 from scratch, and an ego-frame geometry convention worth +2.5 over an anchor-frame one. DVGT-2 makes the same ego-frame choice (local point maps plus relative pose), but for efficiency.
- **The v2 headline crosses evaluators.** Its Table 6 is the [shared baseline block](../concepts/navsim-benchmark.md#dvgt2), all seven rows digit-identical to [[sources/drive-hwm.md]]'s. By the [residual sign](../concepts/navsim-benchmark.md#residual-sign) its own rows are corrected-like (−0.96, −0.80) and TransFuser, Hydra-MDP++, DriveSuprim and DiffusionDrive pre-fix-like (+1.3 to +2.5). The "88.9 outperforms all SOTA methods" margin over DiffusionDrive is +4.4 as printed. On one scale it is +0.7: DiffusionDrive's corrected value is 88.2 in [[sources/world4scorer.md]] and GeoWAM. DVGT-2-NAVSIM's margin is +1.4.
- **"Without fine-tuning" does not mean out of domain.** The general model is trained with trajectory supervision on OpenScene, which supplies NAVSIM's training logs and 77% of its batches, and on nuScenes (6%). It is applied to both benchmarks without a benchmark-specific stage, but both are in its training mix. On the two higher-speed datasets that are only 5–6% of batches each, its 1-second L2 is 2.0–2.1 m (Table A.2).
- **Streaming makes it constant-cost, not cheap.** About 0.26 s and about 70 GB per frame for eight views (hardware not stated), against 110 ms and 0.5 GB for [[sources/drivor.md]] on an A100 at 93.1 PDMS. At the first frame, where no cache exists, DVGT-2 is already about twice as fast as DVGT in Figure 7(b) (about 0.26 s against 0.55 s). So part of the speed-up over its predecessor comes from something other than caching; the lighter two-step pose head is the difference the paper names.
- **Geometry: ray depth improves, global geometry does not.** It has the best Abs Rel and δ<1.25 on all five datasets. Global point accuracy is worse than its batch-processing predecessor DVGT on nuScenes, DDAD and KITTI, and pose AUC is lower on OpenScene, nuScenes and DDAD. The paper attributes this to accumulating relative poses and a short window, and says so. The abstract's "superior geometry reconstruction performance" holds for local depth only.

---

## Method

### Three paradigms (§3.1)

![[e2e_vla_VGA.png|Figure 2: Comparison of different paradigms for end-to-end autonomous driving. Conventional end-to-end models rely on sparse perception representations for scene understanding. VLA models predict language descriptions to interpret driving scenarios. Our VGA model reconstructs dense 3D geometry to facilitate safe planning.]]

The paper frames three ways from images to a trajectory:

- **Conventional end-to-end:** $\mathbf{Z}=\mathcal{F}_\text{perc}(\mathbf{I}_{t-T:t})$, $\mathbf{V}=\mathcal{F}_\text{pred}(\mathbf{Z})$, $\mathbf{A}_t=\mathcal{F}_\text{plan}(\mathbf{Z},\mathbf{V})$, with sparse boxes and maps. Occupancy is dense but voxel-quantized.
- **VLA:** $\mathbf{A}_t,\mathbf{L}_t=\mathcal{M}_\text{VLA}(\mathbf{I}_{t-T:t})$, where language is "inherently ambiguous and coarse-grained".
- **VGA:** $\mathbf{A}_t,\mathbf{P}_{t-T:t},\mathbf{E}_{t-T:t}=\mathcal{M}_\text{VGA}(\mathbf{I}_{t-T:t})$, with pixel-aligned point maps $\mathbf{P}$ and ego poses $\mathbf{E}$.

The argument for VGA is that point maps are continuous (no quantization), cover foreground and background, and are temporally consistent when poses are modelled. It is an argument; no experiment isolates it.

### Streaming geometry (§3.2)

![[streaming.png|Figure 3: Comparison of different paradigms for geometry reconstruction. Batch-processing models like DVGT compute pair-wise relations across all frames, incurring an overall O(T²) complexity. Full-history streaming models like StreamVGGT extract temporal cues from the entire history, leading to an O(T) per-frame complexity. In contrast, our sliding-window streaming strategy attends to a fixed-size cache of length W, achieving a constant O(W) per-frame complexity.]]

| Paradigm | Example | Per new frame | Coordinate frame |
|---|---|---|---|
| Batch | VGGT, π³, DVGT | Recompute all $T$ frames, $O(T^2)$ overall | First frame |
| Full-history streaming | StreamVGGT, CUT3R | Attend to all cached frames, $O(T)$ | First frame, so the whole history must be kept |
| **Sliding window** | **DVGT-2** | **Attend to the last $W$ cached frames, $O(W)$** | **Current ego frame; pose relative to the previous frame** |

$$\mathbf{P}_t,\mathbf{E}_t,\mathbf{C}_{t-W+1:t}=\mathcal{G}_\text{window}([\mathbf{I}_t,\mathbf{C}_{t-W:t-1}])$$

Dropping the first-frame anchor is what lets the cache be bounded. The cost is that any global quantity (a global point cloud, a global pose) has to be built by chaining relative poses, which drifts.

### Architecture (§3.3, Appendix B)

![[framework 3.png|Figure 4: Overall architecture of DVGT-2. Our model consists of an image encoder, a geometry transformer with temporal causal attention, and a set of prediction heads to jointly output geometry reconstruction and trajectory planning.]]

![[infer.png|Figure 5: Efficient inference of DVGT-2. Given the current frame multi-view input and the cache of past W frames, our model performs efficient geometry reconstruction and trajectory planning in an online manner, avoiding recomputing historical frames.]]

$$\mathbf{A}_t,\mathbf{P}_t,\mathbf{E}_t,\mathbf{C}_{t-W+1:t}=\mathcal{M}_\text{DVGT-2}(\mathbf{I}_t,\mathbf{C}_{t-W:t-1})$$

| Component | Setting |
|---|---|
| Image encoder | DINOv3 ViT-L (following DVGT) |
| Extra tokens | 1 pose token and 8 trajectory tokens per view and frame |
| Geometry transformer | $L=24$ blocks; intra-view, cross-view and temporal causal attention per block; 1024-d, 16 heads; QK-Norm, LayerScale 0.01 |
| Temporal position | MRoPE-I (relative), so cached features need no update |
| Cache | Per-layer features of the past $W$ frames, FIFO; $W=4$ for all reported results |
| Point head | DPT on blocks 4, 11, 17, 23; point maps in the current ego frame |
| Pose and trajectory heads | Tokens pooled across views; two anchor-based truncated-diffusion heads after DiffusionDrive (4 self-attention layers across frames, 2 cross-attention layers), 20 anchors each from clustering; pose head two-step |
| Ego status | Velocity, acceleration and driving command, MLP-embedded into the trajectory token |
| Outputs | $\mathbf{P}_t\in\mathbb{R}^{V\times H\times W\times3}$; $\mathbf{E}_t\in\mathbb{R}^7$ (translation, quaternion) relative to the previous frame; $\mathbf{A}_t\in\mathbb{R}^{N\times3}$ ($x$, $y$, yaw) |
| Size | About 1.8B parameters |

### Training (§4.2, Appendices B–C)

| Stage | What | Iterations | Data |
|---|---|---:|---|
| 1 | Geometry pre-training, streaming off | 160K | Five-dataset mix; 2–8 views, 2–24 frames |
| 2 | VGA training: trajectory supervision added, streaming on | 80K | Same mix |
| NAVSIM fine-tune | DVGT-2-NAVSIM | 40K | OpenScene only, 8 views, 4 frames, aspect 1.6, batch size 1 |

AdamW, cosine schedule, peak learning rate 1e-4, 8K warmup, gradient clipping 1.0, bf16, gradient checkpointing. Long edge resized to 512 px, short edge randomly centre-cropped to 144–320 px, colour jitter, blur and grayscale per frame. Dataset ratio per batch nuScenes : OpenScene : Waymo : KITTI : DDAD = **6 : 77 : 6 : 5 : 6**. Each iteration draws an aspect ratio in [1.6, 3.3] and a view count in [2, 8], then fills a 48-image-per-GPU budget with the longest sequence it allows. Total: about ten days on 64 H20 GPUs.

**Geometry labels** are dense depth from **MoGe-2**, a monocular metric depth model, filtered by thresholds, following DVGT. The paper calls this "annotation-efficient geometric supervision": no human labels, but a teacher's.

### Table A.4: Datasets (2 Hz)

| Dataset | Train scenes | Test scenes | Min frames | Max frames | Avg frames | Avg aspect ratio | Views |
|---|---:|---:|---:|---:|---:|---:|---:|
| nuScenes | 700 | 150 | 32 | 41 | 40 | 1.77 | 6 |
| OpenScene | 19376 | 2026 | 1 | 41 | 38 | 1.77 | 8 |
| Waymo | 798 | 202 | 34 | 40 | 40 | 1.77 | 5 |
| KITTI | 138 | 13 | 2 | 1033 | 62 | 3.31 | 2 |
| DDAD | 150 | 50 | 10 | 20 | 17 | 1.59 | 6 |

§4.1 gives OpenScene 19,736 training scenes; this table gives 19,376.

---

## Results: Geometry

**Protocol (§4.4).** Online, frame-by-frame prediction on 16-frame multi-view sequences, metrics averaged over the 16 frames. Batch models are run incrementally, one more frame per step. DVGT-2 uses $W=4$ and its local point maps and relative poses are chained into the first frame's coordinates before scoring. Acc and Comp are global point-cloud distances; Abs Rel and δ<1.25 score ray depth (distance from the point to the current ego centre); AUC scores relative pose error. **\*** marks models aligned to sparse LiDAR after the fact to recover metric scale. The evaluation ground truth is not stated; the training labels are MoGe-2 depth.

### Table 1: OpenScene

| Method | Paradigm | Acc ↓ | Comp ↓ | Abs Rel ↓ | δ<1.25 ↑ | AUC ↑ | Time |
|---|---|---:|---:|---:|---:|---:|---:|
| VGGT\* | Full-Seq. | 1.705 | 1.711 | 0.280 | 0.669 | 74.4 | ~5.31 s |
| MapAnything | Full-Seq. | 3.269 | 4.214 | 0.476 | 0.253 | 69.5 | ~2.28 s |
| DVGT | Full-Seq. | **0.412** | 0.491 | 0.048 | 0.971 | **76.6** | ~1.88 s |
| CUT3R\* | Streaming | 1.858 | 2.245 | 0.275 | 0.596 | 36.9 | ~0.35 s |
| StreamVGGT\* | Streaming | 2.209 | 2.060 | 0.303 | 0.620 | 74.1 | ~1.94 s |
| Driv3R\* | Streaming | 0.884 | 1.693 | 0.188 | 0.740 | – | ~0.56 s |
| **DVGT-2** | Streaming | 0.440 | **0.450** | **0.040** | **0.977** | 70.3 | **~0.27 s** |

### Table 2: nuScenes

| Method | Paradigm | Acc ↓ | Comp ↓ | Abs Rel ↓ | δ<1.25 ↑ | AUC ↑ |
|---|---|---:|---:|---:|---:|---:|
| VGGT\* | Full-Seq. | 1.944 | 2.071 | 0.348 | 0.573 | 83.9 |
| MapAnything | Full-Seq. | 4.447 | 4.860 | 0.562 | 0.271 | 84.9 |
| DVGT | Full-Seq. | **0.469** | **0.508** | 0.067 | 0.955 | **86.4** |
| CUT3R\* | Streaming | 2.055 | 2.611 | 0.330 | 0.547 | 44.5 |
| StreamVGGT\* | Streaming | 2.414 | 2.284 | 0.375 | 0.563 | 86.0 |
| Driv3R\* | Streaming | 0.742 | 1.345 | 0.189 | 0.721 | – |
| **DVGT-2** | Streaming | 0.775 | 0.792 | **0.055** | **0.965** | 84.5 |

### Table 3: Waymo

| Method | Paradigm | Acc ↓ | Comp ↓ | Abs Rel ↓ | δ<1.25 ↑ | AUC ↑ |
|---|---|---:|---:|---:|---:|---:|
| VGGT\* | Full-Seq. | 3.202 | 3.390 | 0.333 | 0.567 | 84.3 |
| MapAnything | Full-Seq. | 9.903 | 8.365 | 0.492 | 0.217 | 82.3 |
| DVGT | Full-Seq. | 1.752 | 2.263 | 0.102 | 0.922 | **86.0** |
| CUT3R\* | Streaming | 3.369 | 4.192 | 0.290 | 0.563 | 50.9 |
| StreamVGGT\* | Streaming | 3.207 | 2.960 | 0.302 | 0.604 | 85.6 |
| Driv3R\* | Streaming | **0.800** | **1.311** | 0.168 | 0.770 | – |
| **DVGT-2** | Streaming | 1.238 | 1.367 | **0.073** | **0.949** | 85.9 |

### Table 4: DDAD

| Method | Paradigm | Acc ↓ | Comp ↓ | Abs Rel ↓ | δ<1.25 ↑ | AUC ↑ |
|---|---|---:|---:|---:|---:|---:|
| VGGT\* | Full-Seq. | 2.322 | 2.879 | 0.798 | 0.395 | 86.3 |
| MapAnything | Full-Seq. | 7.442 | 7.928 | 1.836 | 0.207 | 87.4 |
| DVGT | Full-Seq. | **0.751** | **1.017** | 0.145 | 0.848 | **95.3** |
| CUT3R\* | Streaming | 2.827 | 4.724 | 0.875 | 0.317 | 48.6 |
| StreamVGGT\* | Streaming | 2.655 | 2.731 | 0.810 | 0.419 | 91.9 |
| Driv3R\* | Streaming | 0.950 | 1.259 | 0.185 | 0.740 | – |
| **DVGT-2** | Streaming | 1.770 | 1.837 | **0.093** | **0.919** | 92.5 |

### Table A.3: KITTI

| Method | Paradigm | Acc ↓ | Comp ↓ | Abs Rel ↓ | δ<1.25 ↑ | AUC@30 ↑ |
|---|---|---:|---:|---:|---:|---:|
| VGGT\* | Full-Seq. | 1.154 | 1.294 | 0.158 | 0.801 | **96.91** |
| MapAnything | Full-Seq. | 1.807 | **1.006** | 0.184 | 0.727 | 90.51 |
| DVGT | Full-Seq. | **0.846** | 1.468 | 0.136 | 0.849 | 87.63 |
| CUT3R\* | Streaming | 0.973 | 2.054 | 0.217 | 0.660 | 51.82 |
| StreamVGGT\* | Streaming | 3.393 | 2.181 | 0.365 | 0.467 | 95.57 |
| Driv3R\* | Streaming | 0.864 | 1.083 | 0.164 | 0.784 | – |
| **DVGT-2** | Streaming | 1.615 | 3.238 | **0.087** | **0.942** | 80.36 |

**Reading the five tables together.**

| Dataset | Ray depth (Abs Rel, δ) vs best other | Global Acc vs DVGT | Pose AUC vs DVGT |
|---|---|---|---|
| OpenScene | Best | 0.440 vs 0.412 (worse) | 70.3 vs 76.6 |
| nuScenes | Best | 0.775 vs 0.469 (worse) | 84.5 vs 86.4 |
| Waymo | Best | 1.238 vs 1.752 (better; Driv3R 0.800 best) | 85.9 vs 86.0 |
| DDAD | Best | 1.770 vs 0.751 (worse) | 92.5 vs 95.3 |
| KITTI | Best | 1.615 vs 0.846 (worse; Comp worst of seven) | 80.36 vs 87.63 |

- **Local geometry is the clear win**, on every dataset: Abs Rel 0.040–0.093 against DVGT's 0.048–0.145.
- **Global geometry and pose are mostly worse than DVGT**, the predecessor from the same group, in four of five datasets. The paper explains this (chained relative poses drift, a 4-frame window has no global context, a lighter two-step pose head) and says so in §4.4.
- **The comparison is not like for like.** DVGT and DVGT-2 are trained on these five datasets' training splits with MoGe-2 labels; VGGT, MapAnything, CUT3R and StreamVGGT are general models run zero-shot, most of them needing LiDAR alignment for scale. The margins over them measure in-domain training as much as architecture.

### Table A.1: window size (dataset not stated)

| Window size | Acc ↓ | Abs Rel ↓ |
|---:|---:|---:|
| 2 | 0.613 | 0.042 |
| 4 | 0.480 | 0.042 |
| 6 | **0.474** | 0.042 |
| 8 | 0.501 | 0.042 |

Global accuracy improves to $W=6$ and degrades at 8, which the paper attributes to accumulated pose error. Ray depth does not move. **Window size is never varied for planning.** The values match none of Tables 1–4 or A.3 (at $W=4$, OpenScene is 0.440 / 0.040), so the evaluation set is unknown. The text calls this table "Tab. A.2".

### Figure 6: qualitative

![[vis 3.png|Figure 6: Qualitative visualizations. These results demonstrate that DVGT-2 can predict high-fidelity dense scene geometry and perform robust trajectory planning.]]

![[teaser 4.png|Figure 1: DVGT-2 is a streaming visual geometry transformer specifically designed for autonomous driving. It inputs multi-view images and jointly predicts the 3D pointmaps, the ego poses, and future trajectory planning in an online manner.]]

![[dvgt2_teaser_supp.png|Figure A.1: Video demonstration of DVGT-2's geometry reconstruction and trajectory planning based on online multi-view inputs on the validation set (one frame; downloaded from the arXiv HTML).]]

---

## Results: Planning

### Table 5: NAVSIM-v1 navtest

† = RL post-training. C = camera, L = LiDAR. "Future States" = world-model methods. The last column is this wiki's closed form of the printed sub-scores minus the reported PDMS (per-scene averaging makes it negative for every v1 row; see [[concepts/navsim-benchmark.md]]).

| Method | Input | Aux. Sup. | NC ↑ | DAC ↑ | TTC ↑ | Comf. ↑ | EP ↑ | PDMS ↑ | Closed form − reported |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| PARA-Drive | C | Map & Mot. & Occ | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 | −4.0 |
| VADv2 | C | Map & Mot. & Traffic | 97.2 | 89.1 | 91.6 | 100 | 76.0 | 80.9 | −6.0 |
| UniAD | C | Map & Box & Mot. & Occ | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 | −4.1 |
| Transfuser | C & L | Map & Box | 97.7 | 92.8 | 92.8 | 100 | 79.2 | 84.0 | −3.9 |
| Hydra-MDP | C & L | Map & Box | 98.3 | 96.0 | 94.6 | 100 | 78.7 | 86.5 | −2.6 |
| GoalFlow | C & L | Map & Box | 98.3 | 93.8 | 94.3 | 100 | 79.8 | 85.7 | −3.5 |
| ARTEMIS | C & L | Map & Box | 98.3 | 95.1 | 94.3 | 100 | 81.4 | 87.0 | −3.0 |
| DiffusionDrive | C & L | Map & Box | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 | −2.7 |
| WoTE | C & L | Map & Box | 98.5 | 96.8 | 94.9 | 99.9 | 81.9 | 88.3 | −2.2 |
| DriveSuprim | C & L | Map & Box | 97.8 | 97.3 | 93.6 | 100 | 86.7 | 89.9 | −2.6 |
| AutoVLA | C | Language | 96.9 | 92.4 | 88.1 | 99.9 | 75.8 | 80.5 | −4.5 |
| AdaThinkDrive | C | Language | 98.5 | 94.4 | 94.9 | 100 | 79.9 | 86.2 | −3.0 |
| ReCogDrive | C | Language | 98.3 | 95.1 | 94.3 | 100 | 81.1 | 86.8 | −2.9 |
| DriveVLA-W0 | C | Future States | 98.7 | 99.1 | 95.3 | 99.3 | 83.3 | 90.2 | −1.2 |
| AutoVLA† | C | Language & RL | 98.4 | 95.6 | 98.0 | 99.9 | 85.9 | 89.1 | −1.4 |
| ReCogDrive† | C | Language & RL | 98.2 | 97.8 | 95.2 | 99.8 | 83.5 | 89.6 | −2.1 |
| **DVGT-2** | C | Dense Geometry | 97.8 | 97.2 | 93.9 | 100 | 83.4 | 88.6 | −2.5 |
| **DVGT-2-NAVSIM** | C | Dense Geometry | **98.7** | 97.9 | **95.8** | 100 | 84.3 | **90.3** | −1.7 |

- **"New SOTA" holds in this table only, by 0.1 over DriveVLA-W0** (90.2, an anchor-based configuration; see [[sources/drivevla-w0.md]]). DriveSuprim appears at 89.9, not its 93.5 ViT-L headline. The wiki's v1 ladder has many single-pass entries above 90.3, including [[sources/drivor.md]] 93.1 (camera-only, navtrain).
- **The general model is 88.6 with no NAVSIM-specific stage**, level with AutoVLA† and below the two RL-trained VLAs. The 40K-iteration NAVSIM fine-tune adds +1.7, mostly TTC (+1.9) and NC (+0.9).
- VADv2's 80.9 here and DrivoR's 86.2 cite the **same 2024 arXiv paper**, so the [three circulating VADv2 values](drivor.md) cannot be told apart by citation.

### Table 6: NAVSIM-v2 navtest

The last two columns are this wiki's additions: the closed form of the printed sub-scores minus the reported EPDMS (the [residual sign](../concepts/navsim-benchmark.md#residual-sign)), and the reading it gives.

| Method | NC ↑ | DAC ↑ | DDC ↑ | TL ↑ | EP ↑ | TTC ↑ | LK ↑ | HC ↑ | EC ↑ | EPDMS ↑ | Residual | Reading |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Ego Status MLP | 93.1 | 77.9 | 92.7 | 99.6 | 86.0 | 91.5 | 89.4 | 98.3 | 85.4 | 64.0 | −4.0 | (ego-status row, set aside) |
| Transfuser | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 | +1.3 | pre-fix-like |
| Hydra-MDP++ | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | 81.4 | +2.2 | pre-fix-like |
| DriveSuprim | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 | 83.1 | +2.3 | pre-fix-like |
| ARTEMIS | 98.3 | 95.1 | 98.6 | 99.8 | 81.5 | 97.4 | 96.5 | 98.3 | – | 83.1 | – | EC blank; the closed form would need EC 80.6 |
| DiffusionDrive | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | 84.5 | +2.5 | pre-fix-like (corrected value 88.2 elsewhere) |
| DriveVLA-W0 | 98.5 | 99.1 | 98.0 | 99.7 | 86.4 | 98.1 | 93.2 | 97.9 | 58.9 | 86.1 | −1.3 | the block's known odd row |
| **DVGT-2** | 97.8 | 97.2 | 99.6 | **99.9** | **88.4** | 97.3 | 98.1 | 98.2 | 83.2 | 88.9 | −1.0 | corrected-like |
| **DVGT-2-NAVSIM** | **98.7** | **97.9** | **99.7** | **99.9** | 87.9 | **98.0** | **98.2** | 98.2 | 77.0 | **89.6** | −0.8 | corrected-like |

- **Every baseline row is the [shared baseline block](../concepts/navsim-benchmark.md#dvgt2)**, digit-identical to [[sources/drive-hwm.md]]'s Table II, including ARTEMIS's blank EC. DVGT-2 (v1 April 2026) is the earliest-dated ingested paper to carry the complete block. The clipping is v3, so whether v1 already had this table is not checked.
- **Same-scale margins.** By the residual sign the paper's own rows and the four strongest learned baselines sit on different evaluators. On one scale: DiffusionDrive's corrected 88.2 gives DVGT-2 +0.7 and DVGT-2-NAVSIM +1.4. By closed form, which needs no evaluator: DVGT-2 87.9 and DVGT-2-NAVSIM 88.8 against DiffusionDrive 87.0, DriveSuprim 85.4, DriveVLA-W0 84.8 and Hydra-MDP++ 83.6. The ordering survives; the margins are about a quarter of the printed ones.
- **The NAVSIM fine-tune trades comfort for safety.** EPDMS +0.7, from NC +0.9, DAC +0.7 and TTC +0.7, while **EC falls 83.2 → 77.0** and EP 88.4 → 87.9, with the same diffusion head. [[sources/geowam.md]], initialized from this model with a deterministic head, scores EC 86.8.
- NC and DAC are identical in Tables 5 and 6 for each model, as expected for one checkpoint scored twice.

### Table 7: nuScenes open-loop

† = VAD-style temporal averaging of the per-horizon metrics; unmarked rows use the ST-P3 convention. The table has no ego-status column; DVGT-2 feeds velocity, acceleration and command to its planner, and nuScenes is 6% of its training batches. All averages reproduce from the per-horizon values.

| Method | Input | Aux. Sup. | L2 1s | L2 2s | L2 3s | L2 Avg | Col. 1s | Col. 2s | Col. 3s | Col. Avg |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IL | LiDAR | None | 0.44 | 1.15 | 2.47 | 1.35 | 0.08 | 0.27 | 1.95 | 0.77 |
| NMP | LiDAR | Box & Motion | 0.53 | 1.25 | 2.67 | 1.48 | 0.04 | 0.12 | 0.87 | 0.34 |
| FF | LiDAR | Freespace | 0.55 | 1.20 | 2.54 | 1.43 | 0.06 | 0.17 | 1.07 | 0.43 |
| EO | LiDAR | Freespace | 0.67 | 1.36 | 2.78 | 1.60 | 0.04 | 0.09 | 0.88 | 0.33 |
| ST-P3 | Camera | Map & Box & Depth | 1.33 | 2.11 | 2.90 | 2.11 | 0.23 | 0.62 | 1.27 | 0.71 |
| UniAD | Camera | Map & Box & Mot. & Occ | 0.48 | 0.96 | 1.65 | 1.03 | 0.05 | 0.17 | 0.71 | 0.31 |
| VAD-Tiny | Camera | Map & Box & Mot. | 0.60 | 1.23 | 2.06 | 1.30 | 0.31 | 0.53 | 1.33 | 0.72 |
| VAD-Base | Camera | Map & Box & Mot. | 0.54 | 1.15 | 1.98 | 1.22 | 0.04 | 0.39 | 1.17 | 0.53 |
| GenAD | Camera | Map & Box & Mot. | 0.36 | 0.83 | 1.55 | 0.91 | 0.06 | 0.23 | 1.00 | 0.43 |
| OccWorld | Camera | Occ | 0.52 | 1.27 | 2.41 | 1.40 | 0.12 | 0.40 | 2.08 | 0.87 |
| OccNet | Camera | Occ & Map & Box | 1.29 | 2.13 | 2.99 | 2.14 | 0.21 | 0.59 | 1.37 | 0.72 |
| GaussianAD | Camera | Map & Box & Mot. & Occ | 0.40 | **0.64** | **0.88** | **0.64** | 0.09 | 0.38 | 0.81 | 0.42 |
| OmniDrive | Camera | Map & Box & Language | 0.40 | 0.80 | 1.32 | 0.84 | 0.04 | 0.46 | 2.32 | 0.94 |
| Doe-1 | Camera | Language | 0.50 | 1.18 | 2.11 | 1.26 | 0.04 | 0.37 | 1.19 | 0.53 |
| UniUGP | Camera | Language | 0.58 | 1.14 | 1.95 | 1.23 | 0.01 | 0.19 | 0.81 | 0.33 |
| **DVGT-2** | Camera | Dense Geometry | **0.25** | 0.67 | 1.43 | 0.78 | **0.00** | **0.07** | **0.50** | **0.19** |
| VAD-Tiny† | Camera | Map & Box & Mot. | 0.46 | 0.76 | 1.12 | 0.78 | 0.21 | 0.35 | 0.58 | 0.38 |
| VAD-Base† | Camera | Map & Box & Mot. | 0.41 | 0.70 | 1.05 | 0.72 | 0.07 | 0.17 | 0.41 | 0.22 |
| OccWorld-D† | Camera | Occ | 0.39 | 0.73 | 1.18 | 0.77 | 0.11 | 0.19 | 0.67 | 0.32 |
| GenAD† | Camera | Map & Box & Mot. | 0.28 | 0.49 | 0.78 | 0.52 | 0.08 | **0.14** | **0.34** | **0.19** |
| GaussianAD† | Camera | Map & Box & Occ | 0.34 | 0.47 | **0.60** | 0.47 | 0.49 | 0.49 | 0.51 | 0.50 |
| **DVGT-2†** | Camera | Dense Geometry | **0.20** | **0.37** | 0.66 | **0.41** | **0.04** | **0.14** | 0.47 | 0.22 |

- **"Significantly lower collision rate" holds in the ST-P3 convention only** (0.19 against UniUGP 0.33 and UniAD 0.31). In the VAD convention it is 0.22, level with VAD-Base† and above GenAD† 0.19.
- **L2 is best in the VAD convention and behind GaussianAD in the ST-P3 one.** DVGT-2 is best at 1 s and degrades fastest with horizon. It is the only row known to use ego status, which [[concepts/nuscenes-waymo-evals.md]] shows can be worth as much as a method on L2.

### Table A.2: 1-second L2 per dataset (validation)

| Dataset | L2 (m) ↓ |
|---|---:|
| NuScenes | 0.56 |
| OpenScene | **0.20** |
| Waymo | 0.78 |
| KITTI | 2.12 |
| DDAD | 2.00 |

- **OpenScene, 77% of batches, is best; KITTI and DDAD fail.** The paper blames higher driving speeds and anchors clustered on OpenScene-dominated data.
- **nuScenes 0.56 m here against 0.25 m (or 0.20 m†) at 1 s in Table 7.** Both are the general model on nuScenes validation; the difference is unexplained.

### Figure 7: online efficiency (8 views)

![[dvgt2_fig7.png|Figure 7: Efficiency comparison of online inference. We report the per-frame latency and memory cost of different methods on multi-frame, 8-view sequences. (a) Memory comparison; (b) latency comparison. Cropped from page 15 of the arXiv PDF; the arXiv HTML serves this figure only as SVG.]]

| Method | Memory, from the plot | Latency per frame, from the plot | Fails |
|---|---|---|---|
| VGGT | 79 GB at 1 frame, rising | 0.6 s → 4.7 s at 8 frames | OOM at about 8 frames |
| DVGT | 67 GB at 1 frame, rising | 0.55 s → 2.4 s at 9 frames | OOM at about 10 frames |
| StreamVGGT | 75 GB at 1 frame, linear | 0.6 s → 5.7 s at 31 frames | OOM at about 31 frames |
| **DVGT-2** | **about 70 GB, constant from frame 3** | **about 0.26 s, constant** | "stable at 1200+ frames" |

- **Constant, not small.** About 70 GB per frame of eight views, against 0.5 GB for [[sources/drivor.md]]. The GPU is not named (training used H20s, which have 96 GB). The latency is for geometry and planning together, and the compared models only reconstruct; a planning-only path that skips the DPT head is not timed.
- **The speed-up is not all streaming.** At the first frame no cache exists, yet DVGT-2 is about twice as fast as DVGT. The paper's own explanation for its weaker pose (a two-step pose head against DVGT's four-step residual refinement) is one such difference; resolution is not stated for either.
- **Table 1 and Figure 7 cannot both describe one setup.** Table 1 reports VGGT and DVGT average latencies over 16-frame, 8-view OpenScene sequences (5.31 s and 1.88 s), while Figure 7 has both running out of memory before 10 frames of 8-view input.

---

## Table and Figure Audit {#table-audit}

| Item | Problem |
|---|---|
| Table 6 | The paper's own rows are corrected-like and four baselines pre-fix-like; "outperforms all SOTA methods" crosses evaluators. Same-scale margin over DiffusionDrive +0.7 / +1.4 |
| Table 6 | The complete shared baseline block, including ARTEMIS with a blank EC (HC 98.3). [[sources/physwam.md]] prints the same row with HC blank and EC 98.3, so the one 98.3 value has now been placed in each column |
| Table 7 | Collision claim true in one averaging convention only; ego status used by DVGT-2 and not marked |
| Table A.2 vs Table 7 | nuScenes 1-second L2 0.56 m against 0.25 m / 0.20 m |
| Table A.1 | Dataset not stated; values match no other table; cited in the text as "Tab. A.2" |
| Table 1 vs Figure 7 | VGGT and DVGT latencies averaged over 16 frames in Table 1; both OOM before 10 frames in Figure 7 |
| Figure 7 | DVGT-2 already about 2× faster than DVGT at one frame, where the cache cannot help |
| §4.1 vs Table A.4 | OpenScene training scenes 19,736 against 19,376 |
| Abstract | "Superior geometry reconstruction performance on various datasets": true for ray depth, not for global points or pose against DVGT |
| §4.1 | Geometry evaluation ground truth not stated (training labels are MoGe-2 depth) |
| VADv2 | 80.9 cited to the 2024 arXiv paper; [[sources/drivor.md]] cites the same paper at 86.2 |
| §4.5 | NAVSIM is called "closed-loop" throughout; v1 is non-reactive and v2 pseudo-simulation |

---

## Limitations

1. **No planning ablation.** Geometry loss on or off, geometry pre-training on or off, window size, number of frames, head type: none is varied for planning. Everything said about why VGA plans well rests on cross-paper tables.
2. **Cross-evaluator v2 comparison**, as above. The paper does not name its devkit version.
3. **In-domain "generalization."** Both planning benchmarks' training data are in the mix, at 77% and 6% of batches. The two datasets that are only 5–6% each are where planning fails (L2 above 2 m at 1 s).
4. **Geometry comparisons favour the in-domain models** (DVGT, DVGT-2 trained on these datasets with MoGe-2 labels, against zero-shot baselines), and the evaluation ground truth is not stated. If it is MoGe-2-derived, DVGT-2 is scored against its own teacher; see [[concepts/teacher-pseudo-labels.md]].
5. **Heavy.** 1.8B parameters, about 70 GB and about 0.26 s per 8-view frame, ten days on 64 H20s. No planning-only timing; no hardware named for inference.
6. **Single runs**, with no variance on any number. The diffusion head samples, so the NAVSIM scores carry sampling noise that is not reported.
7. **Navhard is not reported.** The only navhard number for DVGT-2 in the wiki (31.7) comes from GeoWAM's table, presumably for the NAVSIM model.

---

## Relation to Other Wiki Pages

- **[[sources/geowam.md]]** initializes its encoder and point head from DVGT-2 and reports it as its strongest baseline. Three corrections follow from this paper. (1) GeoWAM's "DVGT-2" rows are DVGT-2-NAVSIM. (2) DVGT-2's trajectory head is truncated diffusion with 20 anchors, so GeoWAM's +0.6 navtest / +4.9 navhard swaps the head as well as adding forecasting, and the EC gap (86.8 against 77.0) is a same-lineage contrast between a deterministic and a sampled head, confounded with forecasting. (3) The same diffusion head gives EC 83.2 before NAVSIM fine-tuning, so 6 of the 10 EC points can move with training alone.
- **[[sources/geoworldad.md]]** builds on StreamVGGT, not DVGT-2, but positions itself against "a planner built on a single-layer geometry feature (DVGT-2)" and carries DVGT-2-NAVSIM's rows. Its ablations supply what DVGT-2 lacks: a geometry-pretrained encoder against scratch (89.3 against 84.2 PDMS) and an ego-frame against an anchor-frame geometry model (+2.5). DVGT-2's local point maps and relative poses are that ego-frame convention, chosen here for efficiency.
- **[[sources/physwam.md]]** prints both DVGT-2 variants in v1 and v2, the only earlier ingest to separate them.
- **[[sources/unified-driving-tokens.md]]** takes depth pseudo-labels from DVGT, this model's predecessor.
- **[[sources/diffusiondrive.md]]** supplies the pose and trajectory heads.
- **[[sources/drivor.md]]** is the contrast in cost: a 41M camera-only planner at 110 ms and 0.5 GB scores 93.1 PDMS on navtrain, against DVGT-2-NAVSIM's 90.3 at 1.8B, about 260 ms and about 70 GB with geometry decoded.
- **[[sources/drive-hwm.md]]** and the other users of the shared baseline block: DVGT-2's Table 6 is the block entire, at the earliest arXiv date among them.
- **[[concepts/perception-for-planning.md]]**: dense geometry pseudo-labels as the auxiliary target; see [the VGA section](../concepts/perception-for-planning.md#vga).
- **[[concepts/foundation-backbones-for-ad.md]]**: the geometry backbone family; DVGT-2 is DINOv3 ViT-L plus a 24-block geometry transformer, fully trained.

---

## Open Questions

- **What does the geometry objective buy for planning?** The same model with the point-map loss removed, or without stage-1 geometry pre-training, would answer the paper's own thesis.
- **Does the window matter for planning?** Window size is ablated only for global point accuracy; NAVSIM fine-tuning fixes 4 frames.
- **Does one checkpoint plan across rigs, or only near its dominant dataset?** One model runs on 6- and 8-camera rigs, but planning error is 0.20 m on the 77% dataset and 2.0–2.1 m on the two 5–6% ones. See the [research-directions question](../research-directions.md).
- **What does navhard say about the general model?** Only the NAVSIM model's navhard number exists, from GeoWAM.
- **What is the geometry evaluation ground truth?** If it is MoGe-2 depth, the ray-depth lead over zero-shot models is partly agreement with the teacher.

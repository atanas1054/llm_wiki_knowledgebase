---
title: "GeoWorldAD: Geometry World Action Model for Autonomous Driving"
type: source-summary
sources: [raw/papers/GeoWorldAD_ Geometry World Action Model for Autonomous Driving.md]
related: [concepts/world-model-for-ad.md, concepts/navsim-benchmark.md, concepts/foundation-backbones-for-ad.md, concepts/perception-for-planning.md, concepts/selection-based-planning.md, sources/geowam.md, sources/adaptive-wam.md, sources/drivelaw.md, sources/da-wam.md, sources/simwam.md, sources/foresight.md, sources/brainwam.md, sources/latent-wam.md, sources/epona.md, sources/drivevla-w0.md, sources/wa-jepa.md, sources/drive-jepa.md, sources/drivesuprim.md, sources/diffusiondrive.md, sources/sgdrive.md, sources/deepsight.md]
created: 2026-09-04
updated: 2026-09-04
confidence: high
---

**Paper**: GeoWorldAD: Geometry World Action Model for Autonomous Driving
**Authors**: Songyan Zhang, Jinyuan Tian, Hanbing Li, Daqi Liu, Hao Chen, Wenhui Huang, Fang Li, Guang Chen, Hangjun Ye, Long Chen, Kuiyuan Yang, Chen Lv
**Orgs**: Nanyang Technological University + Xiaomi EV + Zhejiang University
**arXiv**: 2607.17521v2

---

## Summary

This is the wiki's **second** geometry world-action model, and it arrives from a completely different group than [[sources/geowam.md]] (NTU + Xiaomi EV vs. Uber AV Labs). The two share a thesis, share a common ancestor in **DVGT-2**, report near-identical NAVSIM-v2 scores (90.4 vs. 90.2), and **do not cite each other**.

GeoWorldAD's contribution over GeoWAM is that it runs the ablations GeoWAM did not, and two of them are the most useful geometry results in the wiki:

1. **Coordinate frame is worth more than the foundation model.** An off-the-shelf StreamVGGT with 4D reconstruction supervision beats a from-scratch planner by **0.6 PDMS** (84.2 → 84.8) and actually *lowers* NC, DAC, and TTC. Re-expressing the same model's point maps in the **ego-camera frame of each timestep** — a pure re-parameterization, no new capacity — takes it to **87.3**. Geometry pretraining in the wrong frame is nearly worthless.

2. **Readout depth again, from a different angle.** Feeding the planner all 24 decoder layers in one interaction stage scores 87.6; iterating four times on the *final* layer scores 88.2; **four selected layers {4, 11, 17, 23} with iterative refinement after each scores 89.3.** This is the natural follow-up to [[sources/adaptive-wam.md]]'s finding that readout depth is worth ~4.8 PDMS, from a different backbone family, and it says the answer is *several* depths consumed progressively rather than one.

The headline: **91.0 PDMS on NAVSIM v1 and 90.4 EPDMS on v2**, camera-only, no map/box/occupancy supervision. Its latent-future-geometry module is worth **+1.7 PDMS / +2.8 EPDMS** over the present-geometry-only planner — the **first sizeable positive result for a shared imagined future** in this wiki. That ablation is not compute-matched, which is the main thing standing between it and a genuinely important finding.

---

## Positioning

![[teaserv3_4hist.png|Comparison of modular pipelines, single-layer geometry planners, video world models, and GeoWorldAD's geometry world action model]]

**Figure 1**: Given a consecutive video input, GeoWorldAD provides progressively optimized trajectory planning based on present and future geometry guidance. (a) conventional modular stacks; (b) a planner built on a single-layer geometry feature (DVGT-2); (c) video-generation world models supplying pixel-space futures; (d) GeoWorldAD.

The paper's two complaints about prior work are specific and both get tested later:

- **Against single-layer geometry planners (DVGT-2)**: "a single geometry layer may struggle to capture the diverse spatial cues required for planning: fine-grained geometry features are helpful for depicting obstacle boundaries and drivable areas, while higher-level geometry features can encode broader scene structure and agent layout." → Table 6.
- **Against pixel-space world models (Epona, DriveLaW, DriveVLA-W0)**: "RGB representations are redundant and provide limited geometric guidance." → asserted, **not tested** (see Limitations).

It also distinguishes itself from **EponaV2**, which predicts future semantic and depth maps but builds its planning representation on Qwen3-VL features: GeoWorldAD's claim is that the *present-scene* grounding must also come from a geometry foundation model, not just the future supervision.

---

## Method

![[frameworkv8_4hist.png|GeoWorldAD: EgoStreamVGGT geometry model, Q-Former geometry world model producing latent future tokens, and a geometry-oriented action model with iterative refinement]]

**Figure 2**: A video geometry model, a geometry world model, and a geometry-oriented action model handle 4D scene reconstruction, future depth estimation, and trajectory planning respectively. The reconstruction and future-depth decoders are omitted for clarity — and are **not required at planning inference**.

### EgoStreamVGGT: the coordinate-frame fix

The backbone is **StreamVGGT**, a streaming 4D geometry foundation model: a DINOv2 encoder plus a 24-block transformer decoder with frame- and global-attention, producing point maps, depth, and camera parameters via DPT heads.

Multi-scale geometry tokens are taken from four layers:

$$\mathcal{G}_{t}=\left(G_{t}^{4},G_{t}^{11},G_{t}^{17},G_{t}^{23}\right)$$

**The critical modification is a coordinate-system change, not an architectural one.** StreamVGGT reconstructs everything in the anchor frame of the first video frame, while trajectories live in the *moving* ego frame — so misalignment grows over time. EgoStreamVGGT expresses each point map in the ego-camera coordinate system of **its own timestep**, and represents camera poses as relative transforms between *adjacent* frames rather than to the anchor.

Reconstruction losses are unchanged in form ($L_{\mathrm{recon}}=L_{\mathrm{camera}}+L_{\mathrm{depth}}+L_{\mathrm{pmap}}$, Huber on camera, confidence-weighted L1 + gradient-matching on depth and point maps); only the target coordinate system changes.

### Geometry world model: latent future tokens

Learnable future tokens $Q_{\mathrm{fut}}\in\mathbb{R}^{K\times M\times C}$ with **K = 4 chunks spanning 2 seconds** and **M = 64 tokens per chunk**, plus a learnable temporal embedding per chunk. Ego status (velocity, steering state, high-level command) is MLP-projected into $E_{\mathrm{ego}}$ and concatenated with the geometry tokens.

A Q-Former-style module runs four geometry-guided aggregation stages, one per selected layer. Each stage cross-attends the future tokens to that layer's present geometry, then applies **causal** temporal self-attention across future chunks:

$$Q_{\mathrm{fut}}=\mathrm{CausalSelfAttn}\left(\mathrm{CrossAttn}\left(Q_{\mathrm{fut}},\left[G_{t}^{\ell};E_{\mathrm{ego}}\right]\right)\right)$$

Future geometry tokens are then produced by conditioning present geometry on the future latents,

$$\hat{G}_{t+k}^{\ell}=\mathrm{CrossAttn}\left(G_{t}^{\ell},Q_{\mathrm{fut}}^{k}\right)$$

and decoded to **future depth maps** through the *same* DPT head used for the present, supervised by ground-truth future depth with a confidence-weighted L1 + gradient loss. **The future-depth loss does not update the DPT head** — a deliberate stop-gradient so the shared decoder is not distorted by the harder future task.

### Geometry world action model

Trajectory queries $Q_{\mathrm{traj}}\in\mathbb{R}^{R\times T_{p}\times d}$ with **R = 64 proposals, $T_p$ = 8 waypoints, d = 1024**. Refinement proceeds in **five stages**: four present-geometry stages (one per selected layer, each cross-attending to $\mathcal{G}_t^\ell$ and $E_{\mathrm{ego}}$, each decoding proposals through a shared MLP), then **one future-geometry stage** attending to $Q_{\mathrm{fut}}$.

Every stage is supervised with a min-over-proposals objective, exponentially down-weighted for earlier stages:

$$L_{\mathrm{traj}}^{(j)}=\min_{r}\left\|P_{r}^{(j)}-\hat{P}_{\mathrm{gt}}\right\|_{1},\qquad L_{\mathrm{traj}}=\sum_{j=1}^{5}\lambda_{j}L_{\mathrm{traj}}^{(j)}$$

A proposal-scoring head is trained with BCE against the **NAVSIM simulator's own PDMS composition**:

$$S_{\mathrm{gt}}=\mathrm{NC}\times\mathrm{DAC}\times\frac{5\,\mathrm{EP}+5\,\mathrm{TTC}+2\,\mathrm{Comf}}{12}$$

This is privileged simulator distillation of the Hydra-MDP class — see [[concepts/selection-based-planning.md]].

Total objective: $L = L_{\mathrm{traj}} + L_{\mathrm{score}} + L_{\mathrm{recon}} + L_{\mathrm{wm}}$.

### Three-stage training

| Stage | What trains | Data | Steps |
|---|---|---|---|
| 1 | EgoStreamVGGT (from StreamVGGT init) | OpenScene + nuScenes + ParallelDomain + RealDriveSim (10:10:1:1) | 23K |
| 2a | Geometry world model (future depth + retained 4D recon) | OpenScene | 47K |
| 2b | Planner on present geometry only → **GeoAD** | NAVSIM navtrain | 32K |
| 3 | Full GeoWorldAD; future-geometry block **zero-initialized** so it starts as identity | NAVSIM navtrain | **+64K** |

32× NVIDIA H20, global batch 64, AdamW, lr 1e-4 (stages 1–2) and 1e-5 (stage 3), cosine schedule.

---

## Results

### Table 1 — NAVSIM v1 navtest (PDMS)

| Method | Input | Aux. Sup. | NC ↑ | DAC ↑ | TTC ↑ | Comf. ↑ | EP ↑ | PDMS ↑ |
|---|:-:|---|---:|---:|---:|---:|---:|---:|
| VADv2 | C | Map & Mot. & Traffic | 97.2 | 89.1 | 91.6 | 100 | 76.0 | 80.9 |
| UniAD | C | Map & Box & Mot. & Occ | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| PARA-Drive | C | Map & Mot. & Occ | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 |
| Transfuser | C & L | Map & Box | 97.7 | 92.8 | 92.8 | 100 | 79.2 | 84.0 |
| GoalFlow | C & L | Map & Box | 98.3 | 93.8 | 94.3 | 100 | 79.8 | 85.7 |
| DiffusionDrive | C & L | Map & Box | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| WoTE | C & L | Map & Box | 98.5 | 96.8 | 94.9 | 99.9 | 81.9 | 88.3 |
| DriveSuprim | C & L | Map & Box | 97.8 | 97.3 | 93.6 | 100 | 86.7 | 89.9 |
| **iPad** | C | Map & Box | 98.6 | **98.3** | 94.9 | 100 | **88.0** | **91.7** |
| Epona | C | Future States | 97.9 | 95.1 | 93.8 | 99.9 | 80.4 | 86.2 |
| WorldDrive | C | Future States | 98.4 | 96.8 | 95.2 | 100 | 83.3 | 89.0 |
| DriveLaW | C | Future States | **99.0** | 97.1 | **96.7** | 100 | 81.3 | 89.1 |
| DriveVLA-W0 | C | Future States | 98.7 | **99.1** | 95.3 | 99.3 | 83.3 | 90.2 |
| EponaV2 | C | Future States | 98.6 | 97.9 | 95.7 | 100 | 84.8 | 90.4 |
| LFG | C | Dense Geometry | 98.2 | 93.7 | 94.4 | 100 | 79.1 | 85.2 |
| DVGT-2 | C | Dense Geometry | 98.7 | 97.9 | 95.8 | 100 | 84.3 | 90.3 |
| **GeoWorldAD (ours)** | C | Dense & Future Geo. | **99.0** | 97.8 | 95.8 | 99.9 | 85.9 | **91.0** |

**The claim is correctly scoped.** The paper says "best performance among **perception-free** methods on NAVSIM v1" — iPad's 91.7 is in the same table and beats it, but iPad uses map and box supervision. Within the no-structured-supervision group the ordering holds: +0.7 over DVGT-2 (its own lineage) and +0.6 over EponaV2.

**Baseline hygiene is good on v1.** Transfuser 84.0, UniAD 83.4, PARA-Drive 84.0, VADv2 80.9, DiffusionDrive 88.1, WoTE 88.3, Epona 86.2, DriveLaW 89.1 all match this wiki's canonical values. Notably it lists **DriveVLA-W0 at 90.2** — the anchor-based headline — rather than the 87.2 reimplementation row that [[sources/drivelaw.md]], [[sources/brainwam.md]], and [[sources/adaptive-wam.md]] all propagate. DriveSuprim appears at 89.9, the widely-circulated non-ViT-L figure rather than 93.5.

**Five methods new to the wiki**: iPad 91.7, EponaV2 90.4 / 88.9, DVGT-2 90.3 / 89.6, WorldDrive 89.0, LFG 85.2.

### Table 2 — NAVSIM v2 navtest (EPDMS)

| Method | NC ↑ | DAC ↑ | DDC ↑ | TL ↑ | EP ↑ | TTC ↑ | LK ↑ | HC ↑ | EC ↑ | EPDMS ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Transfuser | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | **98.3** | **87.2** | 76.7 |
| DriveSuprim | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | **98.3** | 77.0 | 83.1 |
| DiffusionDrive | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | **98.3** | 87.7 | 84.5 |
| DriveVLA-W0 | 98.5 | **99.1** | 98.0 | 99.7 | 86.4 | 98.1 | 93.2 | 97.9 | 58.9 | 86.1 |
| EponaV2 | 98.5 | 97.4 | 99.5 | **99.9** | 87.9 | 98.1 | 97.7 | 98.2 | 77.4 | 88.9 |
| DVGT-2 | 98.7 | 97.9 | **99.7** | **99.9** | 87.9 | 98.0 | **98.2** | 98.2 | 77.0 | 89.6 |
| **GeoWorldAD (ours)** | **99.0** | 97.8 | 99.6 | 99.7 | **89.1** | **98.6** | 97.6 | 98.0 | 82.2 | **90.4** |

**EP 89.1 and TTC 98.6 lead the table**, and EC 82.2 is respectable for a world-model method (DriveVLA-W0 58.9, DVGT-2 and EponaV2 both ~77).

**This table resolves part of the GeoWAM protocol puzzle** — see [The GeoWAM Comparison](#geowam) below.

---

## Ablations

### Table 3 — Latent future geometry {#future-ablation}

| | v1 NC | v1 TTC | v1 EP | **v1 PDMS** | v2 NC | v2 TTC | v2 EP | **v2 EPDMS** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GeoAD (present geometry only) | 98.9 | 95.7 | 82.6 | **89.3** | 98.9 | 98.3 | 86.3 | **87.6** |
| GeoWorldAD (+ latent future) | 99.0 | 95.8 | 85.9 | **91.0** | 99.0 | 98.6 | 89.1 | **90.4** |
| **Δ** | +0.1 | +0.1 | **+3.3** | **+1.7** | +0.1 | +0.3 | **+2.8** | **+2.8** |

**This is the largest positive result for inference-time future prediction in the wiki**, and the profile is exactly what the paper predicts: safety metrics essentially unchanged, **ego progress up 3.3 / 2.8**. The stated mechanism is that a planner without future guidance behaves conservatively under uncertainty; anticipating how free space evolves lets it commit.

It is also, structurally, the configuration this wiki has repeatedly measured as worthless: **one shared future for all 64 proposals**, not one per candidate. [[sources/da-wam.md]]'s matched ablation puts that at **−0.50 PDMS**; [[sources/foresight.md]] measured +0.3; [[sources/simwam.md]]'s isolated mask measured ~0. GeoWorldAD measures +1.7 for the same shape of mechanism with a **geometric** target.

**The confound that has to be stated.** GeoAD is the Stage-2 checkpoint at **32K planner steps**; GeoWorldAD is that checkpoint plus **64K more steps** in Stage 3. The comparison is not compute-matched, and the extra training is 3× the original planner budget. The zero-initialized future block means the two models are *identical at the start of Stage 3*, so the delta is genuinely attributable to Stage 3 — but Stage 3 changes two things at once. A GeoAD trained for a further 64K steps is the missing row.

### Table 4 — Geometry representation and supervision {#coordinate-frame}

All variants use present geometry only, no future tokens.

| Pretrained model | Aux. sup. | NC ↑ | DAC ↑ | TTC ↑ | Comf. ↑ | EP ↑ | PDMS ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| Scratch | – | 98.1 | 94.6 | 93.9 | 99.1 | 76.0 | 84.2 |
| StreamVGGT | 4D recon | 97.9 | 93.4 | 92.8 | 99.8 | 80.2 | 84.8 |
| EgoStreamVGGT | – | 98.4 | 95.1 | 95.0 | 99.9 | 81.7 | **87.3** |
| EgoStreamVGGT | 4D recon | **98.9** | **97.2** | **95.7** | 99.9 | **82.6** | **89.3** |

**Row 2 is the striking one.** A pretrained streaming 4D geometry foundation model, with its reconstruction objective retained, beats a from-scratch planner by **0.6 PDMS** — and *degrades* NC (98.1 → 97.9), DAC (94.6 → 93.4), and TTC (93.9 → 92.8). It buys ego progress and nothing else.

Changing only the coordinate parameterization recovers **+2.5** (84.8 → 87.3) with gains on every metric, and adding back joint 4D reconstruction supervision another **+2.0** (→ 89.3).

**The transferable claim**: a geometry foundation model is not a drop-in prior. Its output frame has to match the frame the action lives in, and a mismatched frame is worth almost nothing. This is the first *measurement* of the coordinate-frame argument that [[sources/geowam.md]] makes rhetorically ("scene geometry and ego trajectories live in the same 3D coordinate frame") and never tests.

It also parallels [[sources/adaptive-wam.md]]'s adaptation ladder (frozen Wan 84.20 → joint LoRA 90.62) from a different direction: both find that using a foundation model off the shelf costs several PDMS, and that the fix is cheap.

### Table 6 — Geometry aggregation strategy {#aggregation}

![[supp_planning_geo.png|Three geometry aggregation strategies: all-layers single-stage, final-layer iterative, and multi-scale iterative]]

**Figure 3**: (a) trajectory tokens interact once with geometry tokens from all 24 layers — DVGT-2's strategy; (b) trajectory tokens sit on the final layer and refine iteratively through a shared decoder; (c) GeoWorldAD — multi-scale tokens with iterative optimization after each aggregation.

| Geo. layers | Iterations | NC ↑ | DAC ↑ | TTC ↑ | Comf. ↑ | EP ↑ | PDMS ↑ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 | 1 | 98.5 | 95.7 | 95.1 | 99.7 | 81.5 | 87.6 |
| 1 (final) | 4 | 98.6 | 95.5 | 95.2 | 99.8 | **82.9** | 88.2 |
| **4** | **4** | **98.9** | **97.2** | **95.7** | **99.9** | 82.6 | **89.3** |

The decomposition is clean and the two axes buy different things:

- **Iterative refinement buys progress**: EP 81.5 → 82.9 going from one interaction stage to four, with collision metrics flat.
- **Multi-scale buys safety**: DAC 95.5 → 97.2 and NC 98.6 → 98.9 going from one layer to four, with EP essentially unchanged.

Access to all 24 layers in a *single* stage is the worst of the three (87.6) despite having the most information — the paper attributes this to "limited optimization depth," i.e. one cross-attention cannot absorb both low-level boundary detail and high-level layout.

**Read against [[sources/adaptive-wam.md]]**: that paper showed *which single layer* you read is worth up to 4.8 PDMS and that a mid-network layer beats the final one. GeoWorldAD's answer, on a different backbone, is that you should not pick one — but that simply concatenating everything does not work either. Both papers agree the field's default (read the last layer) is wrong.

### Tables 5/7 and 8 — Geometry quality

| Method | OpenScene AbsRel ↓ | OpenScene δ<1.25 ↑ | nuScenes AbsRel ↓ | nuScenes δ<1.25 ↑ | KITTI AbsRel ↓ | KITTI δ<1.25 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| StreamVGGT | 0.236 | 65.6 | 0.265 | 58.2 | 0.173 | 72.2 |
| **EgoStreamVGGT** | **0.141** | **86.5** | **0.117** | **88.5** | **0.077** | **95.5** |

| Method | nuScenes ATE ↓ | RPE trans ↓ | RPE rot ↓ | OpenScene ATE ↓ | RPE trans ↓ | RPE rot ↓ |
|---|---:|---:|---:|---:|---:|---:|
| StreamVGGT | 14.79 | 1.77 | **0.47** | 8.66 | 1.00 | 1.53 |
| **EgoStreamVGGT** | **5.78** | **0.63** | 1.31 | **4.07** | **0.39** | **0.92** |

Large improvements throughout, **except nuScenes rotational RPE, which regresses 0.47 → 1.31 — nearly 3× worse.** The prose says EgoStreamVGGT "significantly reduces trajectory-level and translational pose errors," which is precisely true and quietly excludes the one column that moved the wrong way. Since relative rotation between adjacent frames is exactly what the ego-aligned reparameterization changes, this is the column most worth an explanation, and it gets none.

These tables are **not a clean ego-alignment ablation**: EgoStreamVGGT is both re-parameterized *and* fine-tuned on four driving datasets, while StreamVGGT is off the shelf. Table 4 is the cleaner instrument for the planning claim.

---

## The GeoWAM Comparison {#geowam}

Two independent "geometry world action model" papers, same year, same DVGT-2 ancestor, no mutual citation:

| | [[sources/geowam.md]] | **GeoWorldAD** |
|---|---|---|
| Org | Uber AV Labs + Case Western | NTU + Xiaomi EV + Zhejiang |
| Backbone | DVGT-2 | StreamVGGT → **EgoStreamVGGT** |
| Future target | **Dense metric point maps**, 8 steps | **Latent tokens** supervised by future depth, 4 chunks / 2 s |
| Present grounding | multi-level memory | **multi-scale tokens, 4 layers, iterative** |
| Action head | deterministic single-trajectory regression | 64 proposals + simulator-distilled scorer |
| NAVSIM-v2 navtest | 90.2 EPDMS | **90.4 EPDMS** |
| navhard | **36.6** | not reported |
| Ablations | **none** | three (future, coordinate frame, aggregation) |

**GeoWorldAD supplies two of the three experiments GeoWAM was criticized for lacking** — a with/without-future ablation and a coordinate-frame ablation. **It does not supply the third**, which is the one both papers' central thesis actually needs: *neither paper trains its own architecture with a pixel or video future target.* "Geometry beats pixels" remains an inter-paper comparison in both cases.

### The protocol finding

GeoWorldAD's v2 table **shares GeoWAM's two headline anchors exactly** — DVGT-2 at 89.6 and EponaV2 at 88.9 — while reporting **Transfuser at 76.7 and DiffusionDrive at 84.5**, the values GeoWAM gives as 84.0 and 88.2.

This narrows the wiki's [GeoWAM protocol warning](../concepts/navsim-benchmark.md#three-protocols) considerably. The economical explanation is no longer "GeoWAM is on a third protocol" but "**GeoWAM's Transfuser and DiffusionDrive rows are anomalous while its DVGT-2 and EponaV2 rows are sound**" — because a second, independent paper reproduces the latter pair digit-for-digit and the standard pair alongside them. Tally on Transfuser is now **four papers to one** (WA-JEPA, Drive-JEPA, DA-WAM, GeoWorldAD vs. GeoWAM).

Practically: **GeoWAM's 90.2 and GeoWorldAD's 90.4 are comparable to each other**, both anchored on DVGT-2 89.6. This does not rescue the wiki's global corrected/pre-fix partition — GeoWorldAD's own table still pairs a pre-fix Transfuser (76.7) with a corrected DiffusionDrive (84.5) — but it removes GeoWAM from "cannot be placed at all."

---

## Visualizations

![[recon_vis_0.png|4D reconstruction comparison, StreamVGGT vs EgoStreamVGGT]]
![[recon_vis_1.png|4D reconstruction comparison, second scene]]

**Figures 4–5**: StreamVGGT vs. EgoStreamVGGT 4D reconstruction.

![[recon_vis_2.png|4D reconstruction comparison, third scene]]
![[recon_vis_3.png|4D reconstruction comparison, fourth scene]]

**Figures 6–7**: Further reconstruction comparisons.

![[wm1.png|Future depth prediction visualization]]
![[wm2.png|Future depth prediction, second example]]

**Figures 8–9**: Future depth prediction from the geometry world model — the only qualitative evidence that the latent future tokens encode anything future-like.

---

## Limitations

1. **The +1.7 future-geometry ablation is not compute-matched.** GeoAD has 32K planner steps; GeoWorldAD has 96K. The missing row — GeoAD trained a further 64K steps — is the difference between "geometry futures are the first shared future that works" and "more training helps." Given that this wiki has three prior measurements of shared-future conditioning at roughly zero, that row matters a great deal.

2. **"RGB representations are redundant and provide limited geometric guidance" is never tested.** Like GeoWAM, GeoWorldAD argues geometry beats pixels by comparing against *other papers' pixel-based methods*, never against its own architecture with a pixel future target. Two geometry papers, same missing experiment.

3. **91.0 PDMS is below the wiki frontier.** CLEAR/DA-WAM 93.7, DriveSuprim 93.5, Drive-JEPA 93.3, HybridDriveVLA 92.1, WA-JEPA 91.8, DynVLA 91.7. Its own table contains iPad at 91.7, which beats it — the paper handles this correctly by scoping the claim to perception-free methods, but "state-of-the-art" in the abstract is doing more work than the table supports.

4. **The proposal scorer is trained on NAVSIM-simulator PDMS labels.** Privileged distillation of the Hydra-MDP class, same caveat as DriveSuprim, Drive-JEPA, Auto-JEPA, DA-WAM, and Adaptive-WAM's auxiliary model. The 64-proposal + scorer design also means the headline is not a single-trajectory result.

5. **Heavy and undocumented compute.** 32× H20 across three stages (23K + 47K + 32K + 64K steps) on four datasets including two synthetic ones (ParallelDomain, RealDriveSim). **No latency, no FPS, no parameter count anywhere** — for a method that runs a 24-block geometry decoder plus a Q-Former plus five refinement stages, and whose closest relative ([[sources/adaptive-wam.md]]) makes efficiency its entire contribution.

6. **nuScenes rotational pose error regresses ~3×** (0.47 → 1.31) and is excluded from the prose by careful wording. It is the metric most directly affected by the paper's central modification.

7. **Tables 5 and 7 are the same table printed twice**, and the depth/pose comparison conflates ego-alignment with driving-domain fine-tuning.

8. **No navhard, no HUGSIM, no Bench2Drive, no nuScenes planning.** navhard is the conspicuous gap: GeoWAM's navhard result (36.6) is its strongest, and GeoWAM's own navtest/navhard split (+0.6 vs +4.9) is the wiki's only evidence that geometry world modeling pays off more under reactive protocols. GeoWorldAD tests only the protocol where that effect was smallest.

9. **Single runs, no seed variance**, against ablation deltas of +0.1 on several sub-metrics.

10. **The paper's own stated limitation**: the planner operates on fixed-length clips despite the backbone supporting streaming; KV caching for streaming trajectory inference is future work.

11. **Does not cite [[sources/geowam.md]]** — a contemporaneous paper with the same name-space, the same thesis, the same ancestor, and a score 0.2 lower.

12. **The v2 table mixes evaluator conventions** by this wiki's partition (Transfuser 76.7 pre-fix, DiffusionDrive 84.5 corrected) — the fifth ingested table shown to do so.

---

## Key Cross-References

- **The sibling paper**: [[sources/geowam.md]] — same thesis, same ancestor, near-identical score, no mutual citation. GeoWorldAD supplies the ablations GeoWAM lacks and partially rehabilitates GeoWAM's protocol standing; neither runs the geometry-vs-pixels experiment their shared thesis requires.
- **Coordinate frame as a first-class variable**: [[concepts/foundation-backbones-for-ad.md]] — Table 4's 84.8 → 87.3 from a pure re-parameterization is the wiki's first measurement that a geometry foundation model's *output frame* matters more than the model itself.
- **Readout depth, second data point**: [[sources/adaptive-wam.md]] found a mid-network single layer beats the final layer by 4.80 on Wan2.2; GeoWorldAD finds four selected layers with iterative refinement beat both the final layer alone (+1.1) and all 24 at once (+1.7) on StreamVGGT. Two backbones, same conclusion that the default is wrong.
- **The shared-future question, reopened**: [[concepts/world-model-for-ad.md]] — GeoWorldAD is the first sizeable *positive* result for a shared imagined future (+1.7 / +2.8), against DA-WAM's −0.50, ForeSight's +0.3, and SimWAM's ~0. The distinguishing variable is a geometric rather than photometric target; the confound is 64K unmatched training steps.
- **Progress vs. safety**: the +3.3 EP with flat NC/TTC is the cleanest instance in the wiki of a future-prediction module buying *progress* rather than safety, which is the trade-off [[sources/drivelaw.md]] (NC 99.0 / EP 81.3) and [[sources/da-wam.md]]'s shared-future row (EP collapse 91.36 → 88.68) both sit on the wrong side of.
- **New methods for the gap list**: DVGT-2 (arXiv 2604.00813) is now cited as a baseline by two ingested papers and is the direct ancestor of both geometry WAMs; EponaV2 (2605.14696), iPad (2505.15111), WorldDrive (2603.14948), LFG (CVPR'26), and StreamVGGT (2507.11539) also appear.

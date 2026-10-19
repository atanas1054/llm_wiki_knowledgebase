---
title: "SparseDriveV2: Scoring is All You Need for End-to-End Autonomous Driving"
type: source-summary
sources: ["raw/papers/SparseDriveV2_ Scoring is All You Need for End-to-End Autonomous Driving.md"]
related: [sources/toad.md, sources/hydra-mdp-pp.md, sources/drivesuprim.md, sources/diffusiondrive-v2.md, sources/diffusiondrive.md, sources/clover.md, sources/da-wam.md, sources/wa-jepa.md, sources/drivefuture.md, sources/coworld-vla.md, sources/momworld.md, sources/drivor.md, sources/auto-jepa.md, sources/ad-e2e-jepa.md, sources/idol.md, sources/maneuverflow.md, sources/world4scorer.md, sources/drivereferee.md, sources/linkvla.md, sources/think2drive.md, sources/deepsight.md, sources/latent-wam.md, sources/wam-diff.md, sources/drivefine.md, concepts/selection-based-planning.md, concepts/navsim-benchmark.md, concepts/bench2drive.md, concepts/best-of-n.md, concepts/inference-latency.md, concepts/teacher-pseudo-labels.md, concepts/diffusion-planner.md, research-directions.md]
created: 2026-10-14
updated: 2026-10-19
confidence: medium
---

# SparseDriveV2: Scoring is All You Need for End-to-End Autonomous Driving

**Authors**: Wenchao Sun, Xuewu Lin, Keyu Chen, Zixiang Pei, Xiang Li, Yining Shi, Sifa Zheng
**Orgs**: Tsinghua University; Horizon; Horizon Continental Technology (Beijing)
**arXiv**: 2603.29163v1 (March 2026). [[sources/momworld.md]]'s table lists it as ECCV'26; the clipping names no venue.
**Code**: github.com/swc-17/SparseDriveV2 ("code and model are released").
**Source**: `raw/papers/SparseDriveV2_ Scoring is All You Need for End-to-End Autonomous Driving.md`. The clipping has all five figures and all six tables; Tables 5 and 6 are HTML and are rebuilt below.

**Lineage.** The first author and three co-authors (Sun, Lin, Shi, Zheng) wrote SparseDrive (ICRA 2025), whose sparse, BEV-free scene encoder this paper keeps. Before this ingest the wiki knew SparseDriveV2 only as a row in other papers' tables: the NAVSIM-v2 tables of five ingested papers (CLOVER, DA-WAM, WA-JEPA, ReDrive, SUV), DA-WAM's v1 table, and on Bench2Drive [[sources/momworld.md]]'s.

**Confidence: medium.** Every printed row passes the wiki's arithmetic checks. Its own NAVSIM-v2 row is self-stated in both evaluator columns, and its Bench2Drive rows match the copy MomWorld prints six months later. Against that:
- No latency or memory is reported for a method whose second contribution is efficient scoring.
- Every ablation is a single run on the test split.
- The scaling study (Table 1) states neither the evaluator nor the backbone.
- The NAVSIM-v1 table leaves out the scoring-based rows that beat it, while the v2 table prints the same methods' weaker rows.
- There is no navhard result, and the Bench2Drive training data is not stated.

---

## What It Is

**A fixed trajectory vocabulary, made dense by factorizing it, and scored in two passes.**

- **The question it asks.** Do fixed vocabularies lose to generated proposals because generation is needed, or because a vocabulary of a few thousand entries is too coarse? Table 1 retrains Hydra-MDP with 1,024 to 16,384 anchors: EPDMS rises with every doubling and stops only when training memory runs out.
- **Factorized vocabulary.** Each trajectory is split into a **geometric path**, resampled every 1 m out to 50 m with a validity mask where the trajectory is shorter, and a **velocity profile**, the average speed over each 0.5 s step for 4 s. Paths and profiles are clustered separately by K-means over training trajectories. Any path can be composed with any profile: integrate the speeds to cumulative distance, then interpolate along the path. 1,024 paths × 256 profiles = **262,144 composable trajectories**, "32× denser" than the usual 8,192.
- **Scene encoding** follows SparseDrive: multi-view image features with no BEV grid, from a ResNet-34 over three cameras ($l_0$, $f$, $r_0$) at 256×512, plus an ego-status encoder.
- **Coarse factorized scoring.** Paths and profiles are scored separately:
  - Each path embedding (MLP) plus the ego-status features attends to the scene, by deformable aggregation along the path geometry (or multi-head cross-attention in the ablation).
  - Each velocity embedding attends to the scene by cross-attention.
  - MLP heads score both. Two decoder layers filter progressively: **128 paths × 64 profiles** after the first (8,192 combinations, Hydra-MDP's usual vocabulary size), **20 × 20 = 400** after the second, and 20 × 10 = 200 for NAVSIM-v2.
- **Fine trajectory scoring.** A composed trajectory's feature is the sum of its path and velocity features, **re-conditioned** on the scene by deformable aggregation so that path and speed are no longer treated as independent (a sharp turn at high speed). Sub-score heads then predict the metric terms.
- **Training.**
  - Path, velocity and trajectory scores get soft cross-entropy against $\mathrm{Softmax}(-\lambda d)$, where $d$ is the distance to the logged trajectory's path (masked mean squared), profile (L1) and trajectory (squared L2).
  - Following Hydra-MDP, BCE heads learn a rule-based teacher's sub-scores, but **only for the 400 (200) composed trajectories that survive the filter**. Those depend on the model's own coarse scores, so the teacher labels must be computed during training: the v2 configuration cuts the second-layer profiles to 10 "to accelerate metric ground-truth computation".
  - Loss: $\mathcal L_\text{path}+\mathcal L_\text{vel}+\mathcal L_\text{traj}+\alpha\mathcal L_\text{metric}$; the $\lambda$s and $\alpha$ are not given.
- **Inference.** The predicted sub-scores are combined with the benchmark's own weights, and the argmax is driven.
- **NAVSIM training**: navtrain, 8 × L20 (48 GB), batch 128, 10 epochs, learning rate 1e-4, no weight decay.
- **Bench2Drive configuration** (Appendix A), a different model:
  - ResNet-50 over six cameras at 256×704; paths to 15 m, profiles over 3 s.
  - SparseDrive's detection, online mapping and motion prediction as auxiliary tasks, and path and velocity embeddings that also attend to the agent and map queries.
  - Two-stage training: 100 epochs of perception, then 10 joint epochs, on 16 × L20.
  - **No metric supervision, so pure imitation.**
  - Control: lateral, a preview point on the selected path at $d=0.5\,v_\text{ego}+2.5$ m; longitudinal, the profile's first speed.

![[overview_.png|Figure 1: SparseDriveV2 overview. Left: (a) a trajectory as waypoints at a fixed time step; (b) factorization into a geometric path (fixed spatial step) and a velocity profile (speeds per time step); (c) reconstruction by integrating speed to distance and interpolating along the path. Top right: (d) path and velocity vocabularies built by extraction and clustering of demonstration trajectories, composed into the trajectory vocabulary. Bottom right: (e) scene encoder Φ and ego-status encoder Ψ; (f) coarse factorized scoring, with path and velocity embedders, scene interaction modules and scorers selecting top-K of each; (g) fine-grained scoring of the composed trajectories after re-conditioning on the scene, giving the final decision.]]

**Figure 1** is the whole method in one panel. Note that what reaches stage (g) is a top-$K$ set chosen per scene, not the vocabulary.

| Benchmark | SparseDriveV2 | Notes |
|---|---:|---|
| NAVSIM-v1 PDMS (navtest) | **92.0** | ResNet-34, three cameras, navtrain; NC 98.5, DAC 98.4, TTC 95.0, EP 88.6 |
| NAVSIM-v2 EPDMS, corrected evaluator | **90.1** | Self-stated; EC 78.4 |
| NAVSIM-v2 EPDMS*, original evaluator | 86.7 | Same row; the fix is worth +3.4 |
| Bench2Drive DS / SR | **89.15 / 70.00** | ResNet-50, six cameras, pure imitation; multi-ability mean 67.67 |
| navhard, HUGSIM, nuScenes | – | Not reported |
| Latency / memory | – | Not reported |

---

## Key Takeaways

- **Denser is better within each design, with no saturation shown.**
  - **Retrained Hydra-MDP:** 1,024 → 16,384 anchors gains +2.33 EPDMS, between +0.45 and +0.78 per doubling. Training memory grows by about 1.9 MB per anchor (9.5 → 38.9 GB), and 32,768 runs out of memory on a 48 GB card.
  - **Factorized grid:** 512 × 128 → 1,024 × 256 gains +1.4 (88.7 → 90.1). At equal size (131,072) more paths beat more profiles, 89.5 against 89.2.
  - These are single runs within one design each, and Table 1 states neither the evaluator nor the backbone.
- **"Static" describes the grid, not the candidates that get scored.** The metric-trained scorer sees **400 composed trajectories (200 on v2) chosen per scene** by a coarse filter, and the filter is trained by imitation only: soft labels from distance to the logged path and speed profile. That is [[sources/drivesuprim.md]]'s coarse-to-fine design (8,192 → 256) on a factorized grid. So the comparison the title makes (static scoring against dynamic generation) is really scene-conditioned *selection* from a fixed grid against scene-conditioned *generation*. No oracle is reported, so whether the denser grid helps by coverage or by ranking is unmeasured. [[sources/drivesuprim.md]]'s oracle study says coverage is not the bottleneck: 98.7 PDMS is within 256 of 8,192 candidates ([[concepts/selection-based-planning.md#theoretical-ceiling-oracle-study]]).
- **92.0 PDMS is the highest single-pass NAVSIM-v1 entry in the wiki whose only image encoder is a ResNet-34**, above DiffusionDriveV2 (91.2, camera + LiDAR, RL) and iPad (91.7). It is also above the generated-proposal rows in its own table. Its table leaves out the scoring-based rows above it: DriveSuprim V2-99 92.1 and ViT-L 93.5, and [[sources/drivor.md]] 93.1 (January 2026). Table 3 nonetheless prints DriveSuprim's and Hydra-MDP++'s **V2-99** rows on v2, where SparseDriveV2 beats them.
- **Its own v2 row is self-stated in both columns, 86.7 / 90.1 (+3.4), and its margin depends on the evaluator.**
  - Over DiffusionDriveV2 it leads by **+1.2 pre-fix and +2.6 corrected** (the paper quotes the second).
  - When posted, 90.1 was above every corrected-protocol entry in the wiki dated earlier: WAM-Diff and DriveFine 89.7, Latent-WAM 89.3.
  - About a dozen corrected-cohort entries now sit above it. All were posted after it except Discrete-WAM (90.4, not ingested and undated here) and DrivoR, whose 90.66 is a September measurement of a January model.
  - On a ResNet-34 it is level with [[sources/maneuverflow.md]] (90.14, camera + LiDAR) and is camera-only.
- **Table 3 is the earliest print in the wiki of four rows that later tables carry**:
  - DiffusionDriveV2's **corrected 87.5** (DiffusionDriveV2's own paper reports only 85.5).
  - The **V2-99** rows DriveSuprim **86.0** and Hydra-MDP++ **85.1**.
  - ARTEMIS's duplicated **EC 98.3** (DiffusionDriveV2's table prints "–").

  [[sources/clover.md]]'s seven baseline rows and [[sources/da-wam.md]]'s nine are SparseDriveV2's rows digit for digit, and DA-WAM labels the two V2-99 rows "ViT/L". **The wiki's earlier reading, that the SparseDriveV2 and DiffusionDriveV2 deltas were each confirmed by two independent groups, does not survive.** Every later print postdates this table, so the agreement is consistent with copying one source ([[concepts/navsim-benchmark.md#sparsedrivev2]]).
- **Extended comfort is 78.4**, in the band of the other fixed-vocabulary scorers in its table (Hydra-MDP++ 70.9 / 77.4, DriveSuprim 77.0 / 77.9). The single-trajectory and diffusion rows sit at 87.2–91.0 (Transfuser, DiffusionDriveV2). Per-frame selection from a grid shows the same comfort cost the wiki has recorded for generated pools ([[concepts/selection-based-planning.md#clover]]).
- **On Bench2Drive, 89.15 DS / 70.00 SR from a scorer trained by imitation alone.** That is the highest non-VLM entry on [[concepts/bench2drive.md]], second to [[sources/linkvla.md]] (91.01, posted earlier the same month and absent from its table), and 2.7 DS below the privileged expert behind the base set ([[sources/think2drive.md]], 91.85).
  - Its multi-ability mean is 67.67, and Merging 66.25 is the best on the wiki's page.
  - In its own table it is below HiP-AD on overtaking, emergency braking and traffic signs, so "consistently outperform" holds for DS, SR and the mean only. Comfort (18.32) is among the lowest.
  - The training data and expert protocol are not stated.
- **It settles one Bench2Drive attribution.** SparseDrive's own first author prints 44.54 DS / 16.71 SR as **SparseDrive's**, as [[sources/momworld.md]] (MomAD's first author) does. Two earlier tables in the wiki give that row to MomAD ([[concepts/bench2drive.md#momworld]]).

---

## Results

### Table 1: Hydra-MDP vocabulary scaling (EPDMS on NAVSIM-v2; training memory on one 48 GB L20)

| #Anchors | EPDMS | Δ per doubling | Memory (MB) |
|---:|---:|---:|---:|
| 1,024 | 85.02 | | 9,531 |
| 2,048 | 85.80 | +0.78 | 11,451 |
| 4,096 | 86.33 | +0.53 | 15,513 |
| 8,192 | 86.78 | +0.45 | 23,261 |
| 16,384 | 87.35 | +0.57 | 38,877 |
| 32,768 | – | | OOM |

- Memory rises by 1.88–1.98 MB per added anchor at every step, so the cost is linear in vocabulary size.
- **Evaluator, backbone and cameras are not stated.** At 8,192 anchors (the standard Hydra-MDP size) it reads 86.78. Table 3 has Hydra-MDP++ at 81.4 (ResNet-34) and 85.1 (V2-99) pre-fix, so this is either a stronger configuration or the corrected evaluator. Either way the comparison is within the table only.
- The text calls this "without exhibiting saturation". The per-doubling gains do not shrink monotonically (+0.45, then +0.57), but four steps of single runs cannot separate a slowing curve from noise.

### Table 2: NAVSIM-v1 navtest

Residual = closed form NC·DAC·(5 TTC + 2 C + 5 EP)/12 − reported PDMS ([[concepts/navsim-benchmark.md#residual-sign]]); v1 rows normally sit 1–6 above their closed form, so the residual is negative.

| Method | Img. backbone | NC | DAC | TTC | Comf. | EP | PDMS | Residual |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| VADv2 | ResNet-34 | 97.2 | 89.1 | 91.6 | 100 | 76.0 | 80.9 | −5.99 |
| UniAD | ResNet-34 | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 | −4.12 |
| Transfuser | ResNet-34 | 97.7 | 92.8 | 92.8 | 100 | 79.2 | 84.0 | −3.91 |
| PARA-Drive | ResNet-34 | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 | −4.01 |
| DRAMA | ResNet-34 | 98.0 | 93.1 | 94.8 | 100 | 80.1 | 85.5 | −3.80 |
| GoalFlow | ResNet-34 | 98.3 | 93.8 | 94.3 | 100 | 79.8 | 85.7 | −3.45 |
| Hydra-MDP | ResNet-34 | 98.3 | 96.0 | 94.6 | 100 | 78.7 | 86.5 | −2.63 |
| Hydra-MDP++ | ResNet-34 | 97.6 | 96.0 | 93.1 | 100 | 80.4 | 86.6 | −3.25 |
| ARTEMIS | ResNet-34 | 98.3 | 95.1 | 94.3 | 100 | 81.4 | 87.0 | −2.98 |
| FUMP | ResNet-34 | 98.1 | 96.2 | 94.2 | 100 | 82.0 | 87.8 | −2.79 |
| DiffusionDrive | ResNet-34 | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 | −2.72 |
| WoTE | ResNet-34 | 98.5 | 96.8 | 94.9 | 99.9 | 81.9 | 88.3 | −2.18 |
| DIVER | ResNet-34 | 98.5 | 96.5 | 94.9 | 100 | 82.6 | 88.3 | −2.16 |
| DriveSuprim | ResNet-34 | 97.8 | 97.3 | 93.6 | 100 | 86.7 | 89.9 | −2.55 |
| DiffusionDriveV2 | ResNet-34 | 98.3 | 97.9 | 94.8 | 99.9 | 87.5 | 91.2 | −2.08 |
| iPad | ResNet-34 | 98.6 | 98.3 | 94.9 | 100 | 88.0 | 91.7 | −1.68 |
| Hydra-MDP | V2-99 | 98.4 | 97.8 | 93.9 | 100 | 86.5 | 90.3 | −1.92 |
| GoalFlow | V2-99 | 98.4 | 98.3 | 94.6 | 100 | 85.0 | 90.3 | −1.79 |
| **SparseDriveV2** | **ResNet-34** | 98.5 | **98.4** | **95.0** | 99.9 | **88.6** | **92.0** | −1.72 |

- **All 19 rows pass** (−1.68 to −5.99). The Hydra-MDP++, DiffusionDriveV2 and DriveSuprim rows match their source pages ([[sources/hydra-mdp-pp.md]], [[sources/diffusiondrive-v2.md]], [[sources/drivesuprim.md]]); the Hydra-MDP V2-99 row is the one DriveSuprim prints.
- SparseDriveV2 has the table's best DAC, TTC and EP; iPad's NC (98.6) is higher.
- **Left out**: DriveSuprim V2-99 **92.1** and ViT-L 93.5, Hydra-MDP++ V2-99 91.0, [[sources/drivor.md]] 93.1 and [[sources/drive-jepa.md]] 93.3 (both January 2026). "Outperforming both prior scoring-based planners and dynamic trajectory generation methods" holds inside the table; DriveSuprim's V2-99 configuration is 0.1 above it. The paper's specific claim, that a ResNet-34 beats GoalFlow and Hydra-MDP on V2-99, is true as worded.

### Table 3: NAVSIM-v2 navtest

EPDMS* = original evaluator ("human-behavior filtering was not applied"); EPDMS = the updated official implementation. Residual = closed form NC·DAC·DDC·TL·(5 EP + 5 TTC + 2 LK + 2 HC + 2 EC)/16 − reported.

| Method | Img. backbone | NC | DAC | DDC | TL | EP | TTC | LK | HC | EC | EPDMS* | EPDMS | Resid. vs EPDMS* | Resid. vs EPDMS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ego Status MLP | ResNet-34 | 93.1 | 77.9 | 92.7 | 99.6 | 86.0 | 91.5 | 89.4 | 98.3 | 85.4 | 64.0 | – | −4.00 | |
| Transfuser | ResNet-34 | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 | – | +1.28 | |
| Hydra-MDP++ | ResNet-34 | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | 81.4 | – | +2.16 | |
| DriveSuprim | ResNet-34 | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 | 83.1 | – | +2.28 | |
| ARTEMIS ⚠ | ResNet-34 | 98.3 | 95.1 | 98.6 | 99.8 | 81.5 | 97.4 | 96.5 | 98.3 | 98.3 | 83.1 | – | +2.03 | |
| DiffusionDriveV2 | ResNet-34 | 97.7 | 96.6 | 99.2 | 99.8 | 88.9 | 97.2 | 96.0 | 97.8 | 91.0 | 85.5 | 87.5 | +2.10 | +0.10 |
| Hydra-MDP++ | V2-99 | 98.4 | 98.0 | 99.4 | 99.8 | 87.5 | 97.7 | 95.3 | 98.3 | 77.4 | 85.1 | – | +2.67 | |
| DriveSuprim | V2-99 | 97.8 | 97.9 | 99.5 | 99.9 | 90.6 | 97.1 | 96.6 | 98.3 | 77.9 | 86.0 | – | +2.28 | |
| **SparseDriveV2** | **ResNet-34** | 98.1 | **98.1** | **99.6** | 99.8 | **91.1** | 97.3 | **96.9** | 98.2 | 78.4 | **86.7** | **90.1** | +2.32 | **−1.08** |

- **The labels and the residual signs agree on every labelled row.** Pre-fix-only rows sit at +1.28 to +2.67 (Ego-MLP −4.00, as in every copy). SparseDriveV2's corrected value is at −1.08. DiffusionDriveV2's corrected 87.5 is at +0.10, its standing exception: the printed sub-scores are its original-evaluator run's.
- **The fix is +3.4 on SparseDriveV2** and +2.0 on DiffusionDriveV2, so the margin between them grows from +1.2 to +2.6. The correction does not preserve gaps ([[concepts/navsim-benchmark.md#evaluator-drift-this-table-mixes-two-protocols]]).
- **Provenance.** The first five rows and DiffusionDriveV2's pre-fix row are [[sources/diffusiondrive-v2.md]]'s own v2 table (December 2025), which prints ARTEMIS's EC as "–". Four values are new here, and this is their earliest appearance in the wiki:
  - ⚠ ARTEMIS EC 98.3, a duplicate of its own HC.
  - DiffusionDriveV2's corrected 87.5.
  - The V2-99 rows Hydra-MDP++ 85.1 and DriveSuprim 86.0. DriveSuprim's own paper reports v2 only for ViT-L (87.1, EP 90.5, EC 78.6), so the V2-99 label is the consistent one, and [[sources/wa-jepa.md]] uses it too.

  SparseDriveV2 and DiffusionDriveV2 both list Horizon affiliations, so the corrected DiffusionDriveV2 value may come from shared access to its checkpoint; the paper does not say how it was obtained.
- **"Outperforms ... Hydra-MDP++ and DriveSuprim that adopt larger backbones"** holds against the V2-99 rows. DriveSuprim's ViT-L 87.1 (pre-fix) is above 86.7 and is not in the table.
- SparseDriveV2 has the table's best DAC (98.1), DDC, EP and LK. **EP 91.1** is 3.7 above the human agent's 87.4. EC 78.4 is the cost.

### Table 4: Bench2Drive closed loop (\* = expert feature distillation)

| Method | Driving Score | Success Rate (%) | Efficiency | Comfort |
|---|---:|---:|---:|---:|
| UniAD | 45.81 | 16.36 | 129.21 | 43.58 |
| VAD | 42.35 | 15.00 | 157.94 | 46.01 |
| SparseDrive | 44.54 | 16.71 | 170.21 | 48.63 |
| GenAD | 44.81 | 15.90 | – | – |
| DiFSD | 52.02 | 21.00 | 178.30 | – |
| DriveTransformer | 63.46 | 35.01 | 100.64 | 20.78 |
| Hydra-NeXt | 73.86 | 50.00 | 197.76 | 20.68 |
| SimLingo | 86.02 | 67.27 | 259.23 | 33.67 |
| HiP-AD | 86.77 | 69.09 | 203.12 | 19.36 |
| TCP-traj\* | 59.90 | 30.00 | 76.54 | 18.08 |
| ThinkTwice\* | 62.44 | 31.23 | 69.33 | 16.22 |
| DriveAdapter\* | 64.22 | 33.08 | 70.22 | 16.01 |
| **SparseDriveV2** | **89.15** | **70.00** | 199.84 | 18.32 |

- **SparseDrive 44.54 / 16.71 is printed by SparseDrive's own first author.** [[sources/orion.md]] and [[sources/deepsight.md]] assign that row to MomAD; [[sources/momworld.md]] assigns it to SparseDrive. Two first authors now agree on SparseDrive.
- SimLingo is 86.02 here, as in MomWorld (the wiki's Bench2Drive page also records 85.07 and 85.94).
- The SparseDriveV2, HiP-AD, Hydra-NeXt and SimLingo rows are the ones MomWorld prints six months later.
- **Left out**: [[sources/linkvla.md]] 91.01, [[sources/dynvla.md]] 88.34 and [[sources/automot.md]] 87.34, all posted earlier the same month (March 2026); ORION 77.74 and AutoVLA 78.84, both below it. The privileged expert Think2Drive (91.85) is not shown.

### Table 5: Bench2Drive multi-ability (%)

| Method | Merging | Overtaking | Emergency Brake | Give Way | Traffic Sign | Mean |
|---|---:|---:|---:|---:|---:|---:|
| AD-MLP | 0.00 | 0.00 | 0.00 | 0.00 | 4.35 | 0.87 |
| UniAD | 14.10 | 17.78 | 21.67 | 10.00 | 14.21 | 15.55 |
| VAD | 8.11 | 24.44 | 18.64 | 20.00 | 19.15 | 18.07 |
| DriveTransformer | 17.57 | 35.00 | 48.36 | 40.00 | 52.10 | 38.60 |
| Hydra-NeXt | 40.00 | 64.44 | 61.67 | 50.00 | 50.00 | 53.22 |
| HiP-AD | 50.00 | **84.44** | **83.33** | 40.00 | **72.10** | 65.98 |
| TCP-traj\* | 12.50 | 22.73 | 52.72 | 40.00 | 46.63 | 34.92 |
| ThinkTwice\* | 13.72 | 22.93 | 52.99 | 50.00 | 47.78 | 37.48 |
| DriveAdapter\* | 14.55 | 22.61 | 54.04 | 50.00 | 50.45 | 38.33 |
| **SparseDriveV2** | **66.25** | 75.55 | 75.00 | **50.00** | 71.57 | **67.67** |

- All ten means match the five printed abilities to 0.01 (DriveTransformer's 38.606 is printed 38.60).
- SparseDriveV2 leads on Merging and the mean. **HiP-AD is higher on three of the five abilities**, so "consistently outperform prior approaches" does not hold per ability.
- On the wiki's page: Merging 66.25 is above [[sources/linkvla.md]] (60.00) and [[sources/deepsight.md]] (60.00). The mean is below theirs (73.40, 70.20).

### Table 6: ablations (NAVSIM-v2, corrected EPDMS; the split is not named, but the full model's 90.1 is the navtest headline)

| $N_p$ | $N_v$ | Grid size | EPDMS |
|---:|---:|---:|---:|
| 512 | 128 | 65,536 | 88.7 |
| 512 | 256 | 131,072 | 89.2 |
| 1,024 | 128 | 131,072 | 89.5 |
| 1,024 | 256 | 262,144 | **90.1** |

| Path–scene interaction | Re-conditioning | EPDMS |
|---|---|---:|
| MHA | ✗ | 87.7 |
| MHA | ✓ | 89.9 |
| DFA | ✗ | 89.9 |
| DFA | ✓ | **90.1** |

- **Re-conditioning is worth +2.2 with cross-attention and +0.2 with deformable aggregation.** Either one alone gets to 89.9. The text's "improves results under both attention types" is true, but on the deployed design the step is 0.2, from a single run.
- Doubling paths is worth more than doubling profiles at equal grid size (+0.3).
- No ablation covers the filter's top-$K$, the metric-supervision weight $\alpha$, the camera count or the training set. None is repeated.

---

## Qualitative Figures (Appendix B)

Green = logged human trajectory, red = GTRS-Dense ("the advanced scoring-based baseline"), blue = SparseDriveV2. **GTRS-Dense appears in no table of the paper.** It is NVIDIA's dense-vocabulary scorer with 16,384 entries, which [[sources/momworld.md]] and [[sources/simscale.md]] use, and so a prior vocabulary twice the 8,192 the "32× denser" claim is measured against.

![[smooth_turn.png|Figure 2: three turning scenes, front-camera views with trajectories overlaid and BEV maps. SparseDriveV2 (blue) follows the human's turn (green) closely; GTRS-Dense (red) cuts the corner or turns on a different arc.]]

**Figure 2**: sharp turns; SparseDriveV2's turn tracks the human's, GTRS-Dense's departs from it.

![[efficiency.png|Figure 3: three urban scenes in which SparseDriveV2 (blue) and the human (green) proceed; no red trajectory is visible because GTRS-Dense stays stationary.]]

**Figure 3**: GTRS-Dense stays stationary (its trajectory has no length), while SparseDriveV2 proceeds with the human.

![[high_level.png|Figure 4: three scenes where the human turns (a right turn into a driveway, two turns at intersections); SparseDriveV2 (blue) turns with the human while GTRS-Dense (red) goes straight.]]

**Figure 4**: intent. The human turns, GTRS-Dense goes straight, SparseDriveV2 turns. The paper credits the geometric path representation.

![[navigation.png|Figure 5: failure cases. In the first and third scenes the human turns (left, then right) while SparseDriveV2 (blue) and GTRS-Dense (red) both go straight; in the second SparseDriveV2 drifts toward an adjacent lane while the human stays.]]

**Figure 5** (failure cases): wrong navigation decisions, "possibly due to insufficient navigation information". Both planners go straight where the human turns, so the failure is shared with the baseline. NAVSIM gives the planner only a discrete driving command.

Each figure shows three picked scenes, with no counts. The Bench2Drive videos are supplementary and not in the clipping.

---

## Table and Text Audit

| Check | Result |
|---|---|
| v1 closed form, 19 rows | All pass (−1.68 to −5.99); SparseDriveV2 −1.72 |
| v2 residual signs against stated columns | Agree on every labelled row (pre-fix +1.28 to +2.67; corrected −1.08; DiffusionDriveV2 +0.10, its known exception) |
| Table 5 means | All ten consistent to 0.01 |
| Table 6 vs headline | Full model 90.1 = Table 3's corrected value |
| Bench2Drive rows vs [[sources/momworld.md]] | SparseDriveV2, HiP-AD, Hydra-NeXt, SimLingo and SparseDrive identical, abilities included |
| "navtrain (1192 training scenes) and navtest (136 evaluation scenes)" | These are **log** counts ([[sources/hydra-mdp-pp.md]]); navtest has 12,146 scenarios |
| "32× denser than prior methods (1,024 × 256 vs. 8,192)" | A count of composable combinations. Only 400 (200 on v2) are scored at trajectory level. GTRS-Dense, shown in Figures 2–5, scores 16,384 |
| "Outperforming both prior scoring-based planners and dynamic trajectory generation methods" (v1) | Inside its table; DriveSuprim V2-99 92.1 and ViT-L 93.5 are left out while their v2 V2-99 row is printed |
| "Outperforms ... DriveSuprim that adopt larger backbones" (v2) | Against V2-99 only; DriveSuprim ViT-L 87.1 pre-fix is higher |
| "Consistently outperform prior approaches" (Bench2Drive) | DS, SR and mean ability yes; three of five abilities, efficiency (HiP-AD 203.12, SimLingo 259.23) and comfort no |
| Table 1 | Evaluator, backbone and cameras unstated |
| Training details | $\lambda_p,\lambda_v,\lambda_\tau,\alpha$ unstated; Bench2Drive training data unstated |

---

## Limitations

1. **No inference cost for an efficiency claim.** Neither latency nor memory is reported for SparseDriveV2 on either benchmark. The only cost figure is training memory for the Hydra-MDP baseline.
2. **The thesis is not tested against a matched alternative.** "Static vocabularies can match dynamic generation" rests on cross-paper rows (DiffusionDriveV2 with LiDAR and RL; iPad). No generated-proposal variant with the same encoder and scorer is trained.
3. **No oracle and no coverage measurement.** Whether the 400 survivors contain better trajectories than a smaller grid's survivors, or are only ranked better, is not shown. The imitation-trained filter may discard trajectories the metric scorer would prefer, and this is not measured either.
4. **Model selection and ablations on the test split**, all single runs. The deformable-aggregation re-conditioning step is 0.2.
5. **Evaluator in the training loop.** The rule-based teacher labels the model's own filtered candidates during training, so the NAVSIM model is in the [scorer cohort](../concepts/selection-based-planning.md#top-of-leaderboard). There is no navhard or other out-of-distribution test.
6. **Comparison scope.** The v1 table leaves out the stronger scoring rows whose weaker v2 rows it prints. The method shown in every qualitative figure is in no table.
7. **Two different models under one name.** The NAVSIM model (ResNet-34, three cameras, metric distillation, no auxiliary perception) and the Bench2Drive model (ResNet-50, six cameras, auxiliary detection, mapping and motion, pure imitation) share only the vocabulary and scoring design. The pair is not a transfer result.
8. **Bench2Drive training data is unstated**, so its 2.7-DS gap to Think2Drive cannot be placed under either expert protocol.
9. **Extended comfort is 78.4** with no continuity term, and the paper does not discuss it.

---

## Relation to Other Wiki Pages

- [[sources/toad.md]] *(2026-10-19)*: uses this paper's NAVSIM scorer as a test-time search reward and as a re-ranker of other planners' proposals. As a reward it leaves iPad at 34.6 navhard EPDMS (base 34.7) and drops Hydra-MDP to 29.6 (from 40.9). As a re-ranker of Hydra-MDP's proposals it gives 9.5 (Stage-1 DAC 52.9). TOAD reads this as a vocabulary-trained scorer being inaccurate off its grid; an interface mismatch with the factorized, re-conditioned candidates it was trained on is not ruled out.
- [[sources/hydra-mdp-pp.md]], [[sources/drivesuprim.md]]: the Hydra lineage it scales. DriveSuprim's 8,192 → 256 filter is the template for the two-layer filter; DriveSuprim's oracle study is the counter-evidence on coverage.
- [[sources/diffusiondrive-v2.md]]: the dynamic-generation rival in both tables. Its corrected 87.5 first appears in this paper's Table 3.
- [[sources/clover.md]], [[sources/da-wam.md]]: their v2 baseline rows are this paper's Table 3 rows. [[sources/wa-jepa.md]], [[sources/drivefuture.md]] and [[sources/coworld-vla.md]] later print the DiffusionDriveV2 pair (WA-JEPA also this paper's pair); [[sources/redrive.md]] and [[sources/suv.md]] print the corrected values only.
- [[sources/momworld.md]]: prints its Bench2Drive rows, abilities included, and the same SparseDrive attribution.
- [[sources/auto-jepa.md]]: retrieval (300 of 110,335 recorded trajectories) is the other scene-dependent way to narrow a fixed set. [[sources/ad-e2e-jepa.md]] scales a fixed vocabulary 256 → 8,192 under an endpoint criterion (+5.6 EPDMS). [[sources/idol.md]] finds 128 / 256 / 512 anchors flat (89.8 / 90.0 / 89.6 PDMS).
- [[sources/drivor.md]], [[sources/clover.md]], [[sources/world4scorer.md]], [[sources/maneuverflow.md]]: the generated-pool side, with oracle measurements this paper lacks.
- [[sources/drivereferee.md]]: tests an "argmax (SparseDriveV2 style)" decision rule against gated rules on two candidates (−0.04 against +0.23).
- [[sources/linkvla.md]], [[sources/think2drive.md]], [[sources/deepsight.md]]: the Bench2Drive neighbors and the expert.
- Not ingested: **GTRS / GTRS-Dense** (the qualitative baseline), **iPad** (91.7), **HiP-AD** (86.77 DS), **Hydra-NeXt**, **SparseDrive**, **GoalFlow**, **ARTEMIS**.

---

## Open Questions

- **Coverage or ranking?** An oracle over the 400 survivors, and over the full 262,144 grid, would say whether density buys better candidates or easier ranking, the split [[sources/clover.md]] measured for a generated pool.
- **Does the imitation-trained filter throw away what the metric scorer would pick?** The metric heads never see the 261,744 trajectories the filter drops.
- **Same encoder, same scorer, generated proposals instead of a grid**: the matched test the title implies.
- **What does it cost?** Latency and memory at 400 and at 200 candidates, against DiffusionDriveV2 and DriveSuprim on one GPU.
- **Does metric supervision help in CARLA**, where the Bench2Drive model has none, and on navhard, where the NAVSIM model is not run?
- **Would a continuity term recover extended comfort** as it did for the generated pools of Drive-JEPA, World4Scorer and CLOVER?

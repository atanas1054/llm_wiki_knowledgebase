---
title: Navhard and OOD Evaluation
type: concept
sources: ["raw/papers/Learning to Drive from a World Model.md", "raw/papers/MomWorld_ Momentum-Aware Latent World Model for Long-Horizon Autonomous Driving.md", "raw/papers/ReDrive_ Shaping Representations with World Modeling for End-to-End Driving.md", "raw/papers/PhysWAM_ Physically Consistent World Action Model for Autonomous Driving.md", "raw/papers/Metis_ A Generalizable and Efficient World-Action Model for Autonomous Driving and Urban Navigation.md", "raw/papers/SUV_ Future Scene Understanding as Video Generation for End-to-End Driving.md", raw/papers/DriveFuture_ Future-Aware Latent World Models for Autonomous Driving.md, raw/papers/DriveFine_ Refining-Augmented Masked Diffusion VLA for Precise and Robust Driving.md, raw/papers/SpanVLA_ Efficient Action Bridging and Learning from Negative-Recovery Samples for Vision-Language-Action Model.md, raw/papers/HAD_ Combining Hierarchical Diffusion with Metric-Decoupled RL for End-to-End Driving.md, raw/papers/GeoWAM_ Visual Geometry World Action Models for Autonomous Driving.md]
related: [sources/learning-to-drive-from-a-world-model.md, concepts/data-driven-simulators.md, sources/momworld.md, sources/redrive.md, sources/physwam.md, sources/metis.md, sources/suv.md, sources/drivefuture.md, concepts/selection-based-planning.md, concepts/navsim-benchmark.md, concepts/hugsim-benchmark.md, concepts/rl-for-ad.md, concepts/world-model-for-ad.md, sources/drivefine.md, sources/spanvla.md, sources/had.md, sources/geowam.md, sources/drivelaw.md, sources/drivevla-w0.md, research-directions.md]
created: 2026-05-01
updated: 2026-10-02
confidence: medium
---

## What It Is

Navhard is NAVSIM-v2's hard split, evaluated under a **two-stage pseudo-closed-loop protocol** built on 3D Gaussian Splatting reconstructions. The planner predicts an ego trajectory; the benchmark renders a new observation from the resulting ego pose and feeds it back for the next planning step. Planning errors therefore change what the planner subsequently sees, so the benchmark measures whether a model recovers from its own accumulated deviations — something standard navtest cannot test.

Stage 1 evaluates the original scenes; Stage 2 evaluates synthetic reactive scenes. The official aggregation multiplies stage scores per group and branch before averaging, so a single combined EPDMS is roughly the *product* of stage performance — which is why combined navhard numbers sit near 30 while per-stage numbers sit near 80.

## Why It Matters

NAVSIM-v1 PDMS is saturated (Best-of-6 already matches human ground truth — see [[concepts/best-of-n.md]]) and navtest EPDMS is compressed into a few points at the top. Navhard is not close to saturated. The best combined score in the wiki is **55.5** ([[sources/drivefuture.md]]), the floor is 11.4, and the gap between methods is large enough to rank them.

**The ceiling moved a long way with one ingest.** Before [[sources/drivefuture.md]], this page's leaderboard came entirely from [[sources/geowam.md]] and topped out at 36.6. DriveFuture's Table 1 contains fourteen entries, five of them above 42, and **not one method in common with GeoWAM's table.** The reason is visible at a glance: every entry above 42 scores its own candidates. See [The Scorer-Equipped Cohort](#scorer-cohort).

It is also the wiki's cheapest reactive evaluation. [[concepts/bench2drive.md]] needs CARLA and [[concepts/hugsim-benchmark.md]] needs its own splatting pipeline; navhard runs inside the NAVSIM stack a paper is already using.

## ⚠ Two Reporting Conventions

Papers report navhard in two incompatible ways, and the wiki has been mixing them:

- **Combined EPDMS** — one number spanning both stages, roughly the product of stage performance. [[sources/geowam.md]] and its baselines use this. Values land in the 11–37 range.
- **Per-stage EPDMS** — separate Stage 1 and Stage 2 numbers. [[sources/drivefine.md]] and [[sources/spanvla.md]] use this. Values land in the 40–75 range.

A per-stage pair can be *roughly* converted by multiplying (DriveFine's 74.4 / 41.0 gives about 30.5), but this is an estimate only — the official aggregation multiplies within each group and branch before averaging, which is not the same as multiplying the averages. Do not treat converted values as leaderboard entries.

[[sources/had.md]]'s 32.3 is ambiguous: it is reported as a single number with no Stage 2 companion, which is consistent with either convention. It sits plausibly in the combined range next to DVGT-2's 31.7, but the wiki cannot confirm this.

**SpanVLA's 40.1 is resolved: it is a combined score.** [[sources/spanvla.md]] printed 40.1 against both stage rows, which this page flagged as uninterpretable. [[sources/drivefuture.md]]'s Table 1 reproduces SpanVLA's nine Stage-1 and nine Stage-2 submetrics exactly and prints **a single 40.1 spanning both rows**, in a table whose other entries are verifiably combined. SpanVLA therefore belongs on the combined leaderboard, not in the per-stage table, and a repeated headline across stage rows should be read as a formatting convention rather than a per-stage result. **One paper reprinting another's submetrics resolved an ambiguity neither paper's own text could.**

## The Scorer-Equipped Cohort {#scorer-cohort}

From [[sources/drivefuture.md]] Table 1. All values are the **combined** two-stage EPDMS. The **Scores?** column marks methods that rank their own candidates against a learned or rule-based proxy for the benchmark metric.

| Method | Backbone | Combined EPDMS | Scores? | Family |
|---|---|---:|:-:|---|
| **DriveFuture + GTRS-Dense** | V2-99 | **55.5** | Yes | Latent world model + diffusion |
| DrivoR *(not ingested)* | ViT-S | 54.6 | Yes | E2E |
| SimScale *(not ingested)* | V2-99 | 53.2 | Yes | E2E |
| GTRS-E *(not ingested)* | V2-99+EVA-ViT-L+ViT-L | 49.4 | Yes | Selection-based |
| ZTRS *(not ingested)* | V2-99 | 48.1 | Yes | Selection-based |
| DiffVLA *(not ingested)* | V2-99 + ViT-L/14 | 45.0 | Yes | VLA |
| [[sources/momworld.md]] | V2-99, camera + LiDAR | 42.8 | Yes | GTRS-Dense scorer reading a shared latent future memory, plus a residual flow on the selected plan. **+1.1 over its base** |
| [[sources/drivesuprim.md]] | V2-99 | 42.1 | Yes | Selection-based |
| GTRS-Dense *(not ingested)* | V2-99 | 41.7 | Yes | Selection-based. Row from [[sources/had.md]]'s table; MomWorld's base and DriveFuture's scorer |
| [[sources/spanvla.md]] | Qwen2.5-VL-3B | 40.1 | No | VLA |
| [[sources/physwam.md]], medoid of 8 samples | Cosmos 3 Nano (15.2B) | 39.8 | **No labels, but a selector**: the sample closest to the other seven | Video + metric-depth WAM |
| [[sources/physwam.md]], one sample | Cosmos 3 Nano (15.2B) | 38.1 | No | Video + metric-depth WAM |
| [[sources/suv.md]] | Wan2.2-5B + 1B action expert | 36.9 | No | Video WAM (four future streams) |
| [[sources/metis.md]] | Wan2.2-5B + 1B action expert | 32.2 | No | Video WAM (video dropped at inference) |
| **DriveFuture, no scorer** | V2-99 | **34.6** | **No** | Latent world model + diffusion |
| [[sources/redrive.md]] | V-JEPA 2 ViT-L + 103M action DiT | 34.4 | No | JEPA world model in training only; nothing predicted at inference |
| World4Drive *(not ingested)* | ResNet-34 | 34.9 | No | Latent world model |
| MindDrive *(not ingested)* | ResNet-34 | 30.9 | No | World model + VLM |
| Senna-E2E | ResNet-50 | 27.2 | No | E2E |
| GuideFlow *(not ingested)* | ResNet-34 | 27.1 | No | Flow matching |
| [[sources/diffusiondrive.md]] | ResNet-34 | 24.2 | No | Diffusion |
| TransFuser | ResNet-34 | 23.1 | No | E2E |

**The split is almost perfectly clean.** Every method above 42 scores its candidates; every method below 35 does not. SpanVLA at 40.1 is the only unscored entry in the gap, and DriveFuture's unscored 34.6 sits with the world models it is competing against rather than with the leaderboard it tops. *(2026-09-30: [[sources/physwam.md]] adds two more entries to the gap, 38.1 and 39.8, and SUV's 36.9 was already there. The statement that survives is "every method above 42 scores its candidates with a learned scorer".)*

### The scorer is worth +20.9, measured inside one paper {#scorer-price}

[[sources/drivefuture.md]] is the first paper here to report the same checkpoint with and without a scorer on navhard. Its ablations are all run "without GTRS-Dense scorer"; the best ablation row is **34.6** and its submetrics match the unscored stage-wise row in its Table 7 digit-for-digit. So:

| Configuration | Combined EPDMS |
|---|---:|
| No future frames in training | 30.9 |
| + future-frame grounding (the paper's mechanism) | 34.6 |
| + **GTRS-Dense scorer over 100 proposals** | **55.5** |

**The scorer is worth 5.6x what the world model is worth**, and **85% of the distance** from the weakest configuration (30.9) to the headline (55.5). That reframes this whole page: navhard's leaderboard is currently a ranking of proposal *selectors*, and the proposal *generator* moves it by a few points. See [[concepts/selection-based-planning.md]].

Two caveats before generalizing. DriveFuture's scored and unscored rows differ in nothing but selection, which is what makes the number clean — but it is one method, one run, and the scorer (GTRS-Dense) is trained against a proxy for the benchmark's own metric, so part of the +20.9 is benchmark-specific. And the trade it buys is visible: Stage-1 EC falls 76.9 -> 66.2 and Stage-2 EC 75.9 -> 45.6. Safer, more rule-compliant proposals are less comfortable; EPDMS's multiplicative penalties make that a good trade under this metric and not necessarily under any other.

### Bridging the two tables {#bridge}

The two navhard leaderboards on this page share **no method name**, so merging them requires an assumption. There is exactly one row that bridges them, and it is instructive:

| Paper | Name given | Combined EPDMS | Stage-1 submetrics (NC/DAC/DDC/TLC/EP/TTC/LK/HC/EC) |
|---|---|---:|---|
| [[sources/spanvla.md]] | **LTF** | 23.1 | 96.2 / 79.6 / 99.1 / 99.6 / 84.1 / 95.1 / 94.2 / 97.6 / 79.1 |
| [[sources/drivefuture.md]] | **TransFuser** | 23.1 | 96.2 / 79.5 / 99.1 / 99.5 / 84.1 / 95.1 / 94.2 / 97.5 / 79.1 |
| [[sources/geowam.md]] | **LTF** | **25.1** | NC 96.2, LK 94.2 (Stage 2: NC 77.7, LK 45.4) — the four values this wiki recorded, **all identical** |

Two papers agree on 23.1 for a row whose submetrics are identical to within rounding, and **disagree on what to call it** — LTF and TransFuser are distinct NAVSIM baselines (camera-only vs. camera+LiDAR), so at least one label is wrong. GeoWAM aggregates the same four submetrics this wiki has on file to **25.1**.

*(2026-09-30: superseded in part. [[sources/physwam.md]] traces the 25.1 to the benchmark paper itself, so it is not GeoWAM's aggregation. See [Two Baseline Lineages](#two-lineages).)*

**This is the same offset GeoWAM shows on navtest**, where it scores TransFuser at 84.0 from submetrics four other papers aggregate to 76.7 (see [[concepts/navsim-benchmark.md]]). On navhard the offset is smaller — about +2 on a weak baseline — but it points the same way. The working assumption this page adopts is that **the two tables are on approximately the same scale, +/- 2 points**, which is enough to rank across them but not enough to separate neighbours.

Under that assumption GeoWAM's 36.6 is **eighth**, between SpanVLA (40.1) and World4Drive (34.9) — not the reactive/OOD leader this page previously called it. The claim that survives unchanged is the narrower one GeoWAM actually earns: it leads *its own table*, against RL-supervised methods, without RL.

### Does the combined score really multiply the stages?

Recomputing the EPDMS formula from DriveFuture's published submetrics and multiplying the two stage scores reproduces the combined column to within **0.5 to 3.0 points** across rows (TransFuser 23.6 vs. 23.1; SpanVLA 41.4 vs. 40.1; DriveFuture 52.5 vs. 55.5), with the sign of the residual varying. That is consistent with this page's product reading and with the official protocol's per-group, per-branch aggregation happening *before* averaging. **It is close enough to sanity-check a published row and not close enough to convert a per-stage pair into a leaderboard entry** — which is what the warning above already says.

## Combined-EPDMS Leaderboard

From [[sources/geowam.md]] Table 3. Methods marked † are trained with reinforcement learning or direct PDMS-score supervision; the paper greys them out to separate supervision regimes.

| Method | Combined EPDMS ↑ | Notes |
|---|---:|---|
| **GeoWAM** | **36.6** | Geometry world model, deterministic $\ell_1$ regression head, no RL |
| EponaV2 † | 36.1 | *not ingested* |
| NavFormer † | 34.1 | *not ingested* |
| LTFv6 / LEAD † | 31.9 | *not ingested* |
| DVGT-2 | 31.7 | *not ingested* — GeoWAM's own initialization |
| [[sources/drivelaw.md]] | 30.6 | Video-DiT mid-denoising latents as planning state |
| LTF | 25.1 | Transfuser-family baseline |
| [[sources/drivevla-w0.md]] | 24.4 | AR + diffusion world models, training-time only |
| Ego MLP | 14.1 | Ego-status-only baseline |
| Constant velocity | 11.4 | Floor |

> **Scope note (added after [[sources/drivefuture.md]]).** This table contains no method that scores its own candidates, and [the cohort above](#scorer-cohort) shows five such methods between 42 and 55. Under the +/-2-point bridge established [here](#bridge), GeoWAM's 36.6 ranks eighth overall. Read the rankings below as internal to this table.

**GeoWAM leads its own table while using strictly weaker supervision than the three methods below it.** EponaV2, NavFormer, and LTFv6 all use RL or direct PDMS-score supervision; GeoWAM uses $\ell_1$ trajectory regression. Its margin over EponaV2 is only +0.5, so the ranking is not robust — but the supervision asymmetry runs against it, which makes the result more interesting than the gap size suggests.

**The +4.9 over DVGT-2 is the load-bearing number.** On navtest GeoWAM beats its own DVGT-2 initialization by only +0.6; on navhard the same architectural addition — future-geometry forecasting — is worth eight times more. That is precisely what a world-model thesis predicts: anticipation should matter most where errors compound. It is the strongest evidence in the wiki that world modeling buys robustness rather than open-loop accuracy, and neither GeoWAM nor any other paper remarks on it.

**SUV adds the first controlled inference-path effect on navhard** ([[sources/suv.md]]). The paper reports 36.9 combined (Stage 1 82.3 / Stage 2 43.9, per-stage scores), second among unscored methods behind SpanVLA 40.1. Its Stage-1 DAC of 94.2 is the best unscored value here, and its Stage-2 LK of 47.2 sits in the usual 45–50 band. More important is its ablation: **letting the action expert read the generated future adds +4.1 to +4.5 on navhard but only +0.3 to +0.9 on navtest**. This is the first evidence that navhard can detect a mechanism navtest cannot. See [[concepts/world-model-for-ad.md#navhard-access]].

**Metis supplies two more navhard-only effects, and a split-size discrepancy** ([[sources/metis.md]]).
- *Mask*: at 320×384, letting future video read the action beats full isolation by +2.2 navhard, against +0.5 on navtest.
- *Video prior*: Wan2.1-1.3B matches Wan2.2-5B on navtest but loses 2.4 on navhard.
- Together with SUV's +4.1 access effect, **three same-backbone mechanisms are now invisible on navtest and visible on navhard**. That is the strongest case yet that navhard measures something navtest does not.
- *Split size*: Metis describes navhard as **244 Stage-1 / 4,164 Stage-2** scenarios. SUV describes it as **450 / 5,462**, yet copies Metis's rows. Its DiffusionDrive (27.5) and LTF (24.4) baselines also differ from other papers' copies (24.2; 25.1). Treat cross-paper navhard comparisons as provisional until the split is pinned down.

**PhysWAM: the best unscored result outside RL, and three pieces of bookkeeping** ([[sources/physwam.md]]).
- *Result*: 38.1 with one sample, 39.8 with the medoid of eight. Stage 1 is mid-table (77.0); Stage 2 is where it leads its table, with the best NC (83.9), DAC (80.2) and TTC (81.6). Stage-2 lane keeping is 48.9, in the usual band.
- *Mechanism effect*: its depth–motion loss is worth +1.9 on navhard **and** +1.9 on navtest. This is the first mechanism on this page that is not larger on navhard, so the navtest-small / navhard-large pattern is a property of inference-path mechanisms (access, masks, prior scale), not of every world-model addition.
- *A label-free selector*: choosing the sample nearest the other seven adds **+1.7** on navhard against +0.1 EPDMS on navtest. It needs no labels and no simulator. It is an order of magnitude below a learned scorer (+20.9) and it multiplies inference cost by eight.
- *Comfort*: EC is 90.5 on navtest, 67.6 on Stage 1 and 54.4 on Stage 2 for the same model.
- *Split size*: **450 Stage-1 / 5,462 Stage-2**. Two papers (SUV, PhysWAM) now say 450 / 5,462 against Metis's 244 / 4,164.
- *Stage scores with the combined score*: 77.0 and 48.8 give 38.1; the product of the means is 37.6. The paper states the reason directly: the combined score "multiplies the two stages per scene before averaging, so it is not a function of the two stage means."

**ReDrive: the best result with no future at inference, and the most Stage-1-heavy profile** ([[sources/redrive.md]]).
- *Result*: 34.4 combined, from Stage 1 **82.3** and Stage 2 42.3. The Stage-1 score ties SUV's for the best unscored value; Stage-1 DAC is 93.6.
- *Against the access hypothesis*: three models now land within 0.3 of each other on navtest and spread on navhard in the order the hypothesis predicts.

| Model | Future at inference | navtest EPDMS | navhard S1 / S2 | navhard |
|---|---|---:|---:|---:|
| [[sources/suv.md]], with access | Generated and read | 91.0 | 82.3 / 43.9 | 36.9 |
| [[sources/redrive.md]] | None | 90.8 | 82.3 / 42.3 | 34.4 |
| [[sources/suv.md]], no access | None | 90.7 | – | 32.8 |
| [[sources/metis.md]] | None | 89.5 | 75.8 / 41.7 | 32.2 |

  ReDrive narrows the no-access gap to 2.5 on a different backbone (a JEPA encoder against a video generator). Stage 1 is where it matches SUV and Stage 2 is where it falls behind, which is the stage that re-renders from a displaced pose. This is a cross-paper pattern with single runs, not a controlled result.
- *Comfort*: EC drops from 73.3 to 54.2 between stages.
- *Baselines*: every comparison row is digit-identical to Metis's Table 1 (including its GTRS-sourced LTF at 24.4), so the table tops out at 32.2 and omits SUV, GeoWAM, EponaV2, SpanVLA and the scorer cohort. The split size is not stated.
- *No ablation is run on navhard.* Its future-prediction loss is worth +0.7 PDMS on navtest; whether it is worth more here, as the other training-time mechanisms on this page are, is unmeasured.

**MomWorld: a future memory inside the scorer, worth +1.1, with safety traded for progress** ([[sources/momworld.md]]).
- *Result*: 42.8 combined. It is built on GTRS-Dense (published 41.7) with a camera + LiDAR V2-99 backbone and a 16,384-trajectory vocabulary, so it belongs to the scorer cohort and ranks seventh in it.
- *What the +1.1 is made of*. Against GTRS-Dense's published row:

| Stage | Method | NC | DAC | EP | TTC | LK | HC | EC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | GTRS-Dense | 98.7 | 95.8 | 72.8 | 98.7 | 95.1 | 96.9 | 40.4 |
| 1 | MomWorld | 96.9 | 93.6 | 80.4 | 96.9 | 96.4 | 97.6 | 60.0 |
| 2 | GTRS-Dense | 91.4 | 89.2 | 69.5 | 90.1 | 54.6 | 94.1 | 49.7 |
| 2 | MomWorld | 86.4 | 88.2 | 81.7 | 84.4 | 55.3 | 97.0 | 54.7 |

  Progress and comfort rise (Stage-2 EP +12.2, Stage-1 EC +19.6). The collision-related sub-scores fall in both stages (Stage-2 NC −5.0, TTC −5.7). The product of closed-form stage scores is 43.8 for GTRS-Dense and 43.2 for MomWorld, so the reported +1.1 is inside this page's [±2 conversion error](#does-the-combined-score-really-multiply-the-stages) and is not visible in the mean sub-scores.
- *Its table has the top removed.* The baseline rows are digit-identical to [[sources/drivefuture.md]]'s lineage-B table. All six entries above 42.8 are absent (DriveFuture 55.5, DrivoR 54.6, SimScale 53.2, GTRS-E 49.4, ZTRS 48.1, DiffVLA 45.0), and the paper states it "achieves the best EPDMS of 42.8". DriveFuture's corresponding author is MomWorld's first author.
- *Ablations are not used here.* MomWorld's ablation tables report navhard for 27 configurations, but they show exact cross-column regularities that the wiki cannot explain ([[sources/momworld.md#regularities]]). The no-component row is GTRS-Dense's published row.
- *Protocol*: the paper says navhard is scored "with corrected human-reference filtering". Split size is not stated.

### Two Baseline Lineages, and Where LTF 25.1 Comes From {#two-lineages}

PhysWAM's Table 7 carries a provenance note that settles a question the [bridge](#bridge) left open. Its LTF row is "as reported in EponaV2, which reprints the LTF score of the benchmark paper" (Cao et al., *Pseudo-Simulation for Autonomous Driving*). So **25.1 is the benchmark paper's own LTF value**, passed through EponaV2 to GeoWAM, SUV and PhysWAM. GeoWAM did not compute it.

That reverses the direction of the bridge's suspicion. The same Stage-1 submetrics carry two combined scores:

| Row (identical Stage-1 submetrics) | Combined | Carried by |
|---|---:|---|
| LTF | **25.1** | Benchmark paper → EponaV2 → GeoWAM, SUV, PhysWAM |
| LTF / "TransFuser" | **23.1** | SpanVLA, DriveFuture |

**Two lineages, 2.0 apart on the one row they share.** Lineage A traces to the benchmark paper; lineage B is the DriveFuture / SpanVLA table. Whether the gap is an evaluator version or an aggregation difference cannot be told from published tables. The bridge's "±2 points" should be read as a **possible systematic offset of about 2 points, with lineage A higher**, resting on a single row.

DiffusionDrive also appears with two values, 27.5 (Metis, copied from GTRS; PhysWAM, unattributed) and 24.2 (DriveFuture). Those rows have different submetrics (Stage-1 LK 96.7 against 90.8), so they are two evaluations of the method and not one row aggregated twice. They do not add evidence for the offset.

**Consequences, all conditional on the offset being real.**
- [[sources/physwam.md]] compares its 38.1 against DriveFuture's unscored 34.6, which is a lineage-B number placed in a lineage-A table.
- SpanVLA's 40.1 is also lineage B. Its lead over PhysWAM (38.1) and SUV (36.9) would then be understated by this page, not overstated.
- A third LTF row exists: Metis copies an LTF from GTRS with different submetrics and 24.4. That is a different checkpoint, not a third aggregation.
- 4D-WAM (35.9, not ingested) enters the wiki through PhysWAM's table. Its Stage-1 EP of 98.6 and Stage-2 EP of 97.6 are outliers by ten points and are worth checking at the source.

## Stage 2 Is Where Everything Collapses

The per-stage submetrics in GeoWAM's table expose a failure signature the aggregate scores hide. Every method — including the constant-velocity baseline — loses roughly half its **lane keeping** between stages:

| Method | LK Stage 1 | LK Stage 2 | NC Stage 1 | NC Stage 2 |
|---|---:|---:|---:|---:|
| Constant velocity | 78.6 | 47.9 | 88.8 | 83.2 |
| Ego MLP | 83.5 | 40.8 | 93.2 | 77.2 |
| LTF | 94.2 | 45.4 | 96.2 | 77.7 |
| DriveVLA-W0 | 96.4 | 46.8 | 96.8 | 76.8 |
| DriveLaW | 96.2 | 45.8 | 97.3 | 82.5 |
| DVGT-2 | 95.5 | 48.0 | 97.2 | 77.8 |
| EponaV2 † | 97.3 | 50.1 | 97.3 | 83.6 |
| GeoWAM | 96.0 | 49.9 | 97.7 | 80.4 |

Lane keeping falls from ~96 to ~48 for every learned planner, and no-at-fault collision from ~97 to ~80. **The spread between the best and worst learned method on Stage 2 LK is under 5 points, while the Stage 1 spread is over 12** — under the reactive protocol, methods that look clearly separated collapse toward a common failure mode. Extended comfort behaves similarly, falling from the 60–79 band to 45–67.

This is the same picture [[concepts/hugsim-benchmark.md]] shows on its Extreme tier, where every method lands between 0.06 and 0.14 HD-Score. Two independent reactive benchmarks agree: **current planners degrade to near-indistinguishable once their own errors drive the observations**, and open-loop rankings do not predict which degrade least.

### Correction: the collapse is not universal, and the exception is scoring {#lk-correction}

The paragraph above generalized from GeoWAM's ten baselines, none of which scores its own candidates. [[sources/drivefuture.md]]'s table contains six methods that do, and they break the band:

| Method | Stage-1 LK | Stage-2 LK | Scores candidates? |
|---|---:|---:|:-:|
| TransFuser | 94.2 | 45.4 | No |
| [[sources/diffusiondrive.md]] | 90.8 | 49.2 | No |
| MindDrive | 94.4 | 49.2 | No |
| World4Drive | 87.7 | 52.3 | No |
| **DriveFuture, no scorer** | 94.9 | **47.6** | **No** |
| GTRS-E | 96.0 | 53.9 | Yes |
| [[sources/drivesuprim.md]] | 94.7 | 53.5 | Yes |
| GTRS-Dense (from [[sources/had.md]]) | 95.1 | 54.6 | Yes |
| [[sources/momworld.md]] | 96.4 | 55.3 | Yes |
| DrivoR | 94.9 | 56.1 | Yes |
| **DriveFuture + scorer** | 98.7 | **58.3** | **Yes** |
| SimScale | 95.8 | 60.1 | Yes |
| ZTRS | 96.2 | 60.4 | Yes |
| [[sources/spanvla.md]] | 94.2 | **62.3** | No |

**The 45-50 band was a property of the sample, not of the benchmark.** Stage-2 lane keeping ranges from 45.4 to 62.3 here — a 17-point spread against the "under 5 points" this page previously recorded — and the ordering tracks candidate scoring almost perfectly. DriveFuture supplies the controlled version: **the same checkpoint moves 47.6 to 58.3 by adding a scorer and changing nothing else.**

Two things survive the correction, and one does not:

- **Survives**: every method still loses roughly 35-45 points of lane keeping between stages. Stage 2 is genuinely much harder, and nothing here approaches its own Stage-1 number.
- **Survives**: the open question below about whether part of the drop is a 3DGS rendering artifact. A scorer that evaluates candidates against map geometry would partly compensate for a degraded *observation*, which is consistent with both explanations rather than deciding between them.
- **Does not survive**: "methods that look clearly separated collapse toward a common failure mode." They separate by 17 points on the metric that was supposed to show the collapse, and the separating variable is selection.

## Per-Stage Reports

| Method | Stage 1 EPDMS | Stage 2 EPDMS | Caveat |
| --- | ---: | ---: | --- |
| [[sources/drivefine.md]] | 74.4 | 41.0 | Leads Stage 1 by +5.5 over ReCogDrive; approx. 30.5 if converted to combined |
| ReCogDrive | 68.9 | 37.8 | Reported within DriveFine's table |
| DiffusionDrive | 66.7 | 40.5 | Reported within DriveFine's table |
| ~~[[sources/spanvla.md]]~~ | – | – | **Resolved: 40.1 is a combined score.** Moved to [The Scorer-Equipped Cohort](#scorer-cohort) |
| [[sources/had.md]] | 32.3 | – | Convention ambiguous; see the warning above |

SpanVLA's 40.1 combined EPDMS on navhard against 86.4 on navtest remains the wiki's cleanest single-method statement of the gap: **high navtest scores do not imply robust OOD driving.** [[sources/drivefuture.md]] makes the same point from the other end — 89.9 navtest, 55.5 navhard — and its unscored 34.6 shows most of that gap is closed by selection rather than by the policy. HAD-L makes the same point from 88.5 navtest down to 32.3, and attributes part of it to BEV feature sensitivity under 3DGS synthesis noise — a model-specific diagnosis, not a general one.

## Open Questions

*Wiki-wide open questions are collected in [[research-directions.md]].*

- **Does the combined/per-stage split hide a real ranking?** DriveFine's 74.4/41.0 converts to roughly 30.5, which would place it below DVGT-2 and DriveLaW — but the conversion is an approximation. The [product check](#does-the-combined-score-really-multiply-the-stages) now bounds the error at roughly 0.5-3 points, so the conversion is usable as a sanity check and still not as a leaderboard entry. Someone reporting both conventions for one checkpoint would settle it in one run.
- **How much of any navhard result is the generator and how much is the selector?** [[sources/drivefuture.md]] is the only paper that answers this for its own model: +20.9 of its 55.5 is a GTRS-Dense scorer, against +3.7 for the world model it is named after. Every other entry above 42 also scores, and none reports an unscored ablation. **Until a second paper reports both, navhard rankings should be read as rankings of selection pipelines.**
- **Does future-conditioning survive a strong scorer?** DriveFuture's ablations are all run without one, and its scored Stage-1 safety metrics are near-saturated (NC 99.8, DAC 99.8, DDC 100.0). A +3.7 proposal-quality gain has very little room left to express itself there. The experiment is one run of an existing configuration. *(2026-09-30: [[sources/momworld.md]] is the first entry that puts a predicted future inside a strong scorer. It is +1.1 over GTRS-Dense's published score, with NC and TTC lower and EP higher. The base is a published number and not a matched rerun, and the paper's ablations cannot be used, so the question stays open. The sign of the sub-score change is worth noting: future conditioning bought progress, not safety.)*
- **Why does world modeling help eight times more on navhard than navtest?** GeoWAM's +0.6 / +4.9 split over DVGT-2 is the only measurement of this in the wiki, from a single paper with no ablations. *(Lint 2026-09-30: no longer the only one. SUV (+0.3 / +4.1 for future access) and Metis (+0.5 / +2.2 for the mask, ≈0 / +2.4 for prior scale) show the same asymmetry on one backbone, and [[sources/physwam.md]]'s geometric loss is the first exception at +1.9 / +1.9.)* If it replicates, it reframes what world-model pretraining is *for* — robustness under compounding error rather than open-loop accuracy — and implies navtest is the wrong benchmark for evaluating it.
- **Is Stage 2 lane-keeping collapse a planner failure or a rendering artifact?** Every method including constant velocity loses 35-45 points of LK, which is suspicious. If 3DGS renderings degrade as the ego pose leaves the recorded trajectory, part of the drop measures the benchmark rather than the planner. No paper has separated these. The [scoring correction](#lk-correction) narrows the question rather than answering it: a candidate scorer recovers 10 points of Stage-2 LK on a fixed checkpoint, which is compatible with either explanation. *(2026-10-02: the mechanism exists in a related renderer. [[sources/learning-to-drive-from-a-world-model.md]] reports that reprojected views degrade with the ego's offset from the log (its simulator is kept under 4 m), that the artifacts correlate with the offset strongly enough for a policy to read them, and that night lighting fails first. That is reprojection, not 3DGS, and a training-time observation, so it shows the explanation is plausible, not that it applies here. See [[concepts/data-driven-simulators.md#failure-modes]].)*
- **Does RL help here?** Three of the four methods above 31 use RL or PDMS-score supervision, but GeoWAM tops them without it. *(Lint 2026-09-30: this counts [[sources/geowam.md]]'s table only, which contains no candidate-scoring method; see the scope note on that table.)* With margins of 0.5–2.5 points and single runs, the wiki cannot say whether RL buys OOD robustness.

## Lint Rule

When a paper claims NAVSIM progress, check whether it reports navhard or another OOD split. If not, mark the claim as standard-split only. If it does, **check which reporting convention it uses** before placing the number — and **check whether the number includes a trajectory scorer**, which on this split is worth several times what any published architectural mechanism is worth. Then **check the table's top row against the [cohort table](#scorer-cohort)**: two ingested navhard tables omit the whole cohort above them ([[sources/redrive.md]]'s stops at 32.2 and [[sources/momworld.md]]'s at 42.1).

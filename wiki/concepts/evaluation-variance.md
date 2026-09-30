---
title: Evaluation Variance and Single-Run Reporting
type: concept
sources: ["raw/papers/DriveReferee_ Geometric Safety Verdicts Need Not Be Learned for Driving World-Action Models.md", "raw/papers/WALT_ Learning World-Model-Aligned Latent Trajectories for Autonomous Driving.md", "raw/papers/MomWorld_ Momentum-Aware Latent World Model for Long-Horizon Autonomous Driving.md", "raw/papers/PhysWAM_ Physically Consistent World Action Model for Autonomous Driving.md", "raw/papers/AD-E2E-JEPA_ A Joint-Embedding Predictive Architecture For End-to-End Autonomous Driving.md", raw/papers/WA-JEPA_ Rethinking the Video JEPA Paradigm forWorld-Action Modeling in Autonomous Driving.md, raw/papers/CoWorld-VLA_ Thinking in a Multi-Expert World Model for Autonomous Driving.md, "raw/papers/SUV_ Future Scene Understanding as Video Generation for End-to-End Driving.md", raw/papers/Adaptive-WAM_ Quality-Guided Early-Exit Planningfrom Intermediate Video-Diffusion Features.md, raw/papers/How Can Driving World Models Do Counterfactual Prediction_.md, "raw/papers/[-0.5mm] GRAVA GRAVA_ Grounded Reasoning-to-Action Representation and Learning for Autonomous Driving.md", "raw/papers/Metis_ A Generalizable and Efficient World-Action Model for Autonomous Driving and Urban Navigation.md", raw/papers/SimWAM_ A Simple World Action Model for End-to-End Autonomous Driving.md, raw/papers/Drive-HWM_ Hierarchical World Models for Dynamic-Latent Guided Autonomous Driving.md, raw/papers/DA-WAM_ Decision-Aligned Future Latents for Driving World Models.md, raw/papers/BrainWAM_ Action-Space Coordination of Semantic Priors and Predictive Dynamics for Autonomous Driving.md, raw/papers/Auto-JEPA_ A Latent World Model of Continuous Intent for End-to-End Autonomous Driving.md, raw/papers/See Tomorrow, Act Today_ Foresight-Driven Autonomous Driving.md, raw/papers/DriveFuture_ Future-Aware Latent World Models for Autonomous Driving.md, raw/papers/Driving Intents Amplify Planning-Oriented Reinforcement Learning.md]
related: [sources/drivereferee.md, sources/walt.md, sources/momworld.md, sources/physwam.md, sources/ad-e2e-jepa.md, sources/wa-jepa.md, sources/coworld-vla.md, sources/suv.md, sources/adaptive-wam.md, sources/driving-wm-counterfactuals.md, sources/grava.md, sources/metis.md, sources/simwam.md, sources/drive-hwm.md, sources/da-wam.md, sources/brainwam.md, sources/auto-jepa.md, sources/foresight.md, sources/drivefuture.md, sources/dial.md, concepts/navsim-benchmark.md, concepts/navhard-ood-evaluation.md, concepts/wam-attention-masks.md, concepts/inference-latency.md]
created: 2026-09-27
updated: 2026-09-30
confidence: medium
---

# Evaluation Variance and Single-Run Reporting

## What It Tracks

How much of a reported difference could be noise, and which kind of noise. Twenty-two source pages flag "single run, no seed variance". This page separates the **four sources of variance** that are routinely conflated and records every measurement the wiki has.

---

## Four Different Things Called "Variance"

| Source | What varies | Measured in the wiki? | Size |
|---|---|---|---|
| **1. Evaluator determinism** | Nothing. The same checkpoint is re-scored | [[sources/suv.md]]: 6 repeats, identical to 2 decimals | 0 |
| **2. Sampler seed** | Diffusion/flow noise with fixed weights | [[sources/wa-jepa.md]], [[sources/coworld-vla.md]], [[sources/physwam.md]] | **0.013–0.30** EPDMS std, depending on the architecture |
| **3. Scene sampling** | Which scenes are in the test split | Computable from [[sources/suv.md]]; **observed directly in [[sources/ad-e2e-jepa.md]]** (100 scenes vs. all 12,146) | SE of the mean ≈ **0.16** on navtest; several points on a 100-scene subset |
| **4. Training seed** | Initialization, data order, the whole run | **Never, on any driving benchmark** | Unknown |

**The number that licenses ablation claims is #4, and nobody has measured it.** [[sources/coworld-vla.md]] is the first paper to say so: variance over independently trained models "was not conducted because of the high training cost". [[sources/physwam.md]] is the second: "each training configuration was run once. We report sample-to-sample variation (Table 1) but not run-to-run variation".

---

## The Measurements

### Sampler seed (fixed weights)

| Paper | Sampler | Seeds | Metric | Mean | Std |
|---|---|---:|---|---:|---:|
| [[sources/wa-jepa.md]] | 12-step flow | 10 | EPDMS | 91.70 | **0.053** (range 0.18) |
| [[sources/coworld-vla.md]] | 10-step rectified flow | 6 | PDMS / EPDMS | 89.98 / 89.99 | **0.013 / 0.014** |
| [[sources/physwam.md]] | 30-step UniPC over video + depth + motion | not stated beside the sd (4 samples at 30 steps in its step table) | "PDMS" / EPDMS | 91.4 / 90.3 | **0.24 / 0.30** |

**The range is now a factor of twenty.** PhysWAM's sample-to-sample sd is 5 to 20 times the two earlier figures. The likely reason is what is being sampled: WA-JEPA and CoWorld-VLA draw a trajectory (and compact latents) under strong conditioning, while PhysWAM draws an entire multi-view future from noise with no anchors, and its eight samples are diverse enough that an oracle over them gains +3.9 PDMS. **Sampler noise is a property of the architecture and has to be measured per model.** For PhysWAM it is the same size as its label-free selection gain (+0.3 PDMS) and as its smaller ablation deltas (0.3–0.4).

Its step sweep shows what this does to a single-sample table: 4 steps −0.6 (one sample), 8 steps +0.7 (three samples), 15 steps −0.4 (one sample), all against the 30-step mean. The sweep is not monotone, and two of its three points are single draws within about two standard deviations of zero.

### Scene-level dispersion

[[sources/suv.md]] reports per-scene standard deviations of **17.84 PDMS / 18.21 EPDMS over 12,146 navtest scenes**. The standard error of a mean is therefore about 0.16.

- **For a paired comparison** (two models on the same scenes), the relevant SE is smaller and depends on how correlated the two models' per-scene scores are. *(2026-09-30: [[sources/drivereferee.md]] now reports it; see [below](#paired-ci).)*
- **navhard is much noisier.** There are only 244 or 450 Stage-1 scenes, depending on the paper; see [[concepts/navhard-ood-evaluation.md]]. If navhard's per-scene SD is similar, the SE is roughly **0.9–1.2 points**. That is the scale of several recent "navhard-only" mechanism effects (+2.2 Metis mask, +2.4 Metis video prior), though smaller than SUV's +4.1.

### Paired confidence intervals, measured {#paired-ci}

[[sources/drivereferee.md]] pairs every comparison at the scene level on full navtest and reports bootstrap 95% intervals over 10,000 resamples. These are the first such numbers in the wiki.

| What is compared | Reported Δ EPDMS | 95% half-width | Implied SE |
|---|---:|---:|---:|
| Two checkpoints with independent samples (preference-distilled against base) | +0.92 | 0.21 | 0.11 |
| Two training variants of one model | +0.29 | 0.16 | 0.08 |
| Two selectors on the same candidates | −0.01 | 0.15 | 0.07 |
| Selection against none, same first sample (weak base) | +0.30 | 0.12 | 0.06 |
| Selection against none, same first sample (strong base) | −0.04 | 0.08 | 0.04 |

- **Pairing roughly halves the noise.** An unpaired difference of two navtest means has an SE near 0.23 (0.16 × √2). Two checkpoints paired on scenes: 0.11.
- **A working threshold follows.** On full navtest, a difference between two single-sample checkpoints smaller than about **0.2 EPDMS** is not distinguishable from scene sampling alone, before any training-seed variance is added.
- **The interval shrinks as two systems agree on more scenes.** Selection variants that change a few hundred plans have half the SE of two separately trained models.
- These intervals do not include training seeds. Each model here is one run.

### A subset that reorders the full set {#subset-reordering}

[[sources/ad-e2e-jepa.md]] is the first paper here to report the **same checkpoints on a 100-scene subsample and on all 12,146 navtest scenes**. It does so because its dense-token baselines need 92–101 s per scene and cannot be run on the full set. That makes scene-sampling variance visible instead of computed:

| Variant (zero-shot, oracle goal) | EPDMS, 100 scenes | EPDMS, 12,146 scenes | Rank: subset → full |
|---|---:|---:|---|
| navtrain | **76.6** | 63.5 | 1 → 3 |
| navtrain + rollout | 70.4 | 64.9 | 4 → 2 |
| trainval | 72.1 | 63.2 | 3 → 4 |
| trainval + rollout | 72.3 | **67.3** | 2 → 1 |

- **The subset's best variant is the full set's third.** The spread among the four on the subset (6.2) is larger than on the full set (4.1), and it is ordered differently.
- **Submetrics on the subset are counts.** The 6.2-point drop from adding the rollout loss on navtrain is DAC 87 → 77, which is ten scenes.
- **Expected size.** With navtest's per-scene SD of about 18 for a strong planner, 100 scenes give an SE near 1.8. These zero-shot planners score in the 60s–70s with more zero-valued scenes, so a per-scene SD of 30–40 and an SE of 3–4 points is more plausible. Differences of 2–6 EPDMS between rows of that table are inside it. Top-1 hit rates there carry a binomial SE near 5 points.
- **The subset and the full set also differ systematically.** Extended comfort is dropped on the subset (no adjacent scenes), which accounts for roughly half of the 12-point EPDMS† gap for the navtrain variant.
- The paper says so in one sentence ("may partly reflect variance from evaluating only 100 scenes") and still draws its headline figure from the subset.

**Rule**: a comparison that exists only on a subsample of about 100 scenes can support "comparable" or a difference of tens of points (LeWM 48.3 against 68–77). It cannot order methods that are within several points of each other.

### A generation metric with a measured floor {#fvd-floor}

Scene sampling affects generation metrics far more than planning scores, and until [[sources/physwam.md]] no paper here had measured it. It scores **recorded NAVSIM clips against other recorded clips**:

| Comparison | Clips per side | FVD | FID |
|---|---:|---:|---:|
| Recorded vs. recorded (floor) | 600 | **91.5 ± 3.2** | 11.0 ± 0.3 |
| PhysWAM vs. recorded (disjoint scenes) | 600 | 111.3 ± 6.8 | 15.4 ± 0.6 |
| PhysWAM vs. recorded (same scenes) | 1,200 | 42.2 | 6.8 |

- Real video scores 91.5 against real video at this sample size. **An FVD is only meaningful as an excess over a floor at a stated clip count.**
- The same generator reads 111.3 or 42.2 depending on the protocol.
- Three of the four NAVSIM FVD values it compares against (PWM 86.0, DriveDreamer-Policy 53.6, CoWorld-VLA 32.7) come with no clip count.

See the NAVSIM generation table on [[concepts/world-model-for-ad.md]].

### Selection effects that inflate single numbers

- [[sources/adaptive-wam.md]] reports the **validation-best checkpoint per seed, aggregated over ten seeds**. That is a selection procedure, not a variance estimate, and it is optimistic relative to a single run.
- [[sources/suv.md]] selects checkpoints on the NAVSIM validation PDM score.
- [[sources/dial.md]] compares methods at their best held-out checkpoint, even though several later collapse.
- Best-of-N oracle rows ([[concepts/best-of-n.md]]) are the extreme case.

### Too little variance: ablation columns that are functions of each other {#column-regularity}

Every other entry on this page is about noise that papers under-report. [[sources/momworld.md]] raises the opposite check: whether a table has as much scatter as independent runs should.

| What was checked | What independent runs would show | What MomWorld's tables show |
|---|---|---|
| Sign of each step in a cumulative component study | Some metrics flat or reversed at some steps | All 45 increments (9 metrics × 5 steps) improve, each column by a near-constant step |
| Two different metrics across ablation rows (L2@6 and TPC@6) | Correlated, with scatter | Difference 0.84–0.88 m over 27 rows (sd 0.008); across published methods it spans 0.66–0.96 |
| The same metric at two horizons (L2@3, L2@6) | Correlated, with scatter | L2@6 = 2.31 + 3.0·(L2@3 − 0.86) to the printed digit for 7 of 8 ablated variants, and within 0.01 for the eighth |
| Two benchmarks with different base models (NAVSIM PDMS, nuScenes L2) | Weakly related at best | PDMS ≈ 90.2 − 20·(L2@3 − 0.86) within 0.2 for 7 of 8 ablated variants |

The wiki draws no conclusion about cause. It treats the affected effect sizes as unverified and does not cite them on concept pages.

**The check is cheap and general.** For any ablation table with four or more rows and two or more metrics: difference adjacent rows, and regress one column on another. Sampler noise alone is 0.01–0.3 on NAVSIM and training-seed noise is larger, so residuals that vanish at the printed precision need an explanation. The known benign cause is a column that is *computed* from another (an average, or a closed-form aggregate), which is why the check is only informative between separately measured quantities.

### Outside NAVSIM

[[sources/driving-wm-counterfactuals.md]] reports means over five seeds with a **±maximum deviation** (not a standard deviation). This is the only multi-seed generative evaluation in the wiki.

---

## What This Means for Reading the Wiki

1. **Sampler noise is usually not the binding constraint, but it is model-specific.** *(2026-09-30: [[sources/physwam.md]] measures 0.24–0.30 for a model that samples a full multi-view future. Check the paper's own figure before applying the 0.01–0.05 range below.)* At 0.01–0.05 it is 2–3 orders of magnitude below the **+2.0 to +3.8** EPDMS shift from the [evaluator correction](navsim-benchmark.md#evaluator-drift-this-table-mixes-two-protocols). Protocol drift matters far more than seeds for cross-paper comparison.
2. **Ablation deltas under ~0.5 should be read as "no detectable effect".** Examples: SimWAM's mask ablation (0.2 spread), SUV's per-stream access on navtest (≤0.2), Qwen-Drive's reasoning effect (+0.4), GRAVA's +0.18 margin for the best purely autoregressive model, [[sources/walt.md]]'s +0.41 PDMS headline (its v1 and v2 sub-scores attribute the gain to different terms: NC and TTC on v1, extended comfort on v2). This is not because sampler noise is that large, but because training-seed variance is unmeasured and plausibly several times larger.
3. **navhard effects need to be several points to be interpretable.** The recent pattern of "invisible on navtest, large on navhard" ([[concepts/wam-attention-masks.md]]) is the most important open finding that currently rests on single runs over a few hundred scenes.
4. **Headline margins of 0.1–0.7 between top methods are within unmeasured training noise.** Examples: SUV vs Metis top-6, GRAVA vs Curious-VLA, SUV vs Poutine on WOD-E2E (+0.03 RFS).
5. **Subsampled evaluations need their own error bar.** A 100-scene NAVSIM subset has an SE of roughly 2–4 points and reordered one paper's own variants ([above](#subset-reordering)).
6. **An ablation table can also be too clean.** Monotone gains on every metric at every step, or columns that are exact functions of each other, are a reason to withhold the effect sizes ([above](#column-regularity)).

---

## Open Questions

- **What is training-seed variance for a NAVSIM planner?** Three seeds of one cheap model (a DiffusionDrive-class planner trains in hours) would bound it and settle rule 2 above.
- **What is paired per-scene variance between two published checkpoints?** It needs only released checkpoints and the evaluator, no training. *(2026-09-30: measured within one paper by [[sources/drivereferee.md]]: a 95% half-width of about 0.21 EPDMS for two checkpoints of one model family; see [above](#paired-ci). Across two different published methods it is still unmeasured.)*
- **What is navhard's per-scene SD?** Any paper that reports per-scene navhard scores would let every navhard effect in the wiki be given an error bar.

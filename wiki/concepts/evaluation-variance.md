---
title: Evaluation Variance and Single-Run Reporting
type: concept
sources: ["raw/papers/IDOL_ Inverse-Dynamics-Guided Future Prediction for End-to-End Autonomous Driving.md", "raw/papers/HyWorldVLA_ A Vision-Language-Action Model with Hybrid World Modeling for Autonomous Driving.md", "raw/papers/Outcome-Grounded World Modeling for Autonomous Driving.md", "raw/papers/RAD_ Training an End-to-End Driving Policy via Large-Scale 3DGS-based Reinforcement Learning.md", "raw/papers/ForeDrive_ Foresight-Guided End-to-End Autonomous Driving with a Planning-Relevant Latent World Model.md", "raw/papers/Raw2Drive_ Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving(in CARLA v2).md", "raw/papers/Think2Drive_ Efficient Reinforcement Learning by Thinking with Latent World Model for Autonomous Driving (in CARLA-v2).md", "raw/papers/SimScale_ Learning to Drive via Real-World Simulation at Scale.md", "raw/papers/ResWorld_ Temporal Residual World Model for End-to-End Autonomous Driving.md", "raw/papers/Learning to Drive from a World Model.md", "raw/papers/DriveReferee_ Geometric Safety Verdicts Need Not Be Learned for Driving World-Action Models.md", "raw/papers/WALT_ Learning World-Model-Aligned Latent Trajectories for Autonomous Driving.md", "raw/papers/MomWorld_ Momentum-Aware Latent World Model for Long-Horizon Autonomous Driving.md", "raw/papers/PhysWAM_ Physically Consistent World Action Model for Autonomous Driving.md", "raw/papers/AD-E2E-JEPA_ A Joint-Embedding Predictive Architecture For End-to-End Autonomous Driving.md", raw/papers/WA-JEPA_ Rethinking the Video JEPA Paradigm forWorld-Action Modeling in Autonomous Driving.md, raw/papers/CoWorld-VLA_ Thinking in a Multi-Expert World Model for Autonomous Driving.md, "raw/papers/SUV_ Future Scene Understanding as Video Generation for End-to-End Driving.md", raw/papers/Adaptive-WAM_ Quality-Guided Early-Exit Planningfrom Intermediate Video-Diffusion Features.md, raw/papers/How Can Driving World Models Do Counterfactual Prediction_.md, "raw/papers/[-0.5mm] GRAVA GRAVA_ Grounded Reasoning-to-Action Representation and Learning for Autonomous Driving.md", "raw/papers/Metis_ A Generalizable and Efficient World-Action Model for Autonomous Driving and Urban Navigation.md", raw/papers/SimWAM_ A Simple World Action Model for End-to-End Autonomous Driving.md, raw/papers/Drive-HWM_ Hierarchical World Models for Dynamic-Latent Guided Autonomous Driving.md, raw/papers/DA-WAM_ Decision-Aligned Future Latents for Driving World Models.md, raw/papers/BrainWAM_ Action-Space Coordination of Semantic Priors and Predictive Dynamics for Autonomous Driving.md, raw/papers/Auto-JEPA_ A Latent World Model of Continuous Intent for End-to-End Autonomous Driving.md, raw/papers/See Tomorrow, Act Today_ Foresight-Driven Autonomous Driving.md, raw/papers/DriveFuture_ Future-Aware Latent World Models for Autonomous Driving.md, raw/papers/Driving Intents Amplify Planning-Oriented Reinforcement Learning.md]
related: [sources/idol.md, sources/hyworldvla.md, sources/world4scorer.md, sources/rad.md, sources/foredrive.md, sources/raw2drive.md, sources/think2drive.md, sources/simscale.md, sources/resworld.md, sources/learning-to-drive-from-a-world-model.md, sources/drivereferee.md, sources/walt.md, sources/momworld.md, sources/physwam.md, sources/ad-e2e-jepa.md, sources/wa-jepa.md, sources/coworld-vla.md, sources/suv.md, sources/adaptive-wam.md, sources/driving-wm-counterfactuals.md, sources/grava.md, sources/metis.md, sources/simwam.md, sources/drive-hwm.md, sources/da-wam.md, sources/brainwam.md, sources/auto-jepa.md, sources/foresight.md, sources/drivefuture.md, sources/dial.md, concepts/navsim-benchmark.md, concepts/navhard-ood-evaluation.md, concepts/wam-attention-masks.md, concepts/inference-latency.md, research-directions.md]
created: 2026-09-27
updated: 2026-10-12
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
| **4. Training seed** | Initialization, data order, the whole run | **Once**: [[sources/foredrive.md]], 3 seeds of one recipe (2026-10-06) | **0.1** PDMS sd (89.7–89.9) for a 118M single-camera DiffusionDrive-style planner |

**The number that licenses ablation claims is #4, and until ForeDrive nobody had measured it.** [[sources/coworld-vla.md]] is the first paper to say so: variance over independently trained models "was not conducted because of the high training cost". [[sources/physwam.md]] is the second: "each training configuration was run once. We report sample-to-sample variation (Table 1) but not run-to-run variation".

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

### Training seed, measured once {#training-seed}

*(2026-10-06)* [[sources/foredrive.md]] retrains its default recipe (DINOv3 ViT-B/16 front camera, JEPA-style latent predictor, DiffusionDrive-style planner, 118.4M parameters, 100 epochs) from three seeds: **89.7–89.9 PDMS on navtest, sd 0.1**. Its tables report the primary run (89.9), the top of that range.

- **It is the first number of its kind here**, and it is small: about twice the largest sampler-seed sd for a trajectory-sampling model (WA-JEPA 0.053) and a third of PhysWAM's sampler sd.
- **Three seeds give a rough sd.** With n = 3 the true sd could plausibly be two or three times larger.
- **It covers one recipe.** The paper's ablations are single runs; its Base was not re-seeded. A model that fine-tunes a large pretrained network for a few epochs, or samples whole futures, may spread more.
- **What it changes.** Rule 2 below assumed training-seed variance "plausibly several times larger" than sampler noise. For this recipe it is about 0.1, so ablation deltas of 0.3–0.5 between separately trained runs are a few sd and not obviously noise. Whether that transfers to other architectures is the open part.

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

**Clustering by log** *(2026-10-06)*. [[sources/world4scorer.md]] bootstraps over the **136 navtest logs** rather than the 12,146 scenes, because consecutive scenes of one log are correlated. Its intervals on fixed-pool interventions: matched-state regret 5.14 [4.66, 5.62]; a scorer shuffle +14.72 [+13.72, +15.75]; inertial re-ranking with the correct previous plan against a two-step-old one +1.01 [0.86, 1.16]. Scene-level bootstraps such as DriveReferee's will be somewhat narrower than log-clustered ones on the same comparison. The same paper's training ablations are single-seed and use the last epoch while its headline uses a validation-selected epoch (93.96 vs 94.0 for the same configuration), so its +0.95 for the future target has no interval.

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

**A 10-clip ablation set in closed loop** *(2026-10-06)*. [[sources/raw2drive.md]] runs every ablation on Dev10, ten Bench2Drive clips "carefully selected ... challenging and representative, with low variance", three repeats each. Success rates are reported in tenths of a route (7.5/10), and the full model scores 83.5 DS there against 71.36 on all 220 routes, so the subset is easier than the benchmark. Its ablations behave as cliffs (83.5 with every component, 0–65.6 without any one), which a 10-clip set can show but cannot rank.

**Rule**: a comparison that exists only on a subsample of about 100 scenes can support "comparable" or a difference of tens of points (LeWM 48.3 against 68–77). It cannot order methods that are within several points of each other.

### A private hard subset on its own scale {#noisy-subset}

[[sources/hyworldvla.md]] scores six models on 655 rain/fog scenes from the OpenScene test split. The subset is not released and its selection rule is not described. Three properties make its numbers within-paper only:

- **Spread between near-identical models is large.** DriveVLA-W0 (90.2 on navtest) and the paper's own pixel-only rerun of the same recipe (89.91) score 61.18 and 69.95 on the subset: 0.29 apart on navtest, 8.77 apart there.
- **It is not uniformly harder than navtest.** The λ3 = 1.0 configuration scores 92.94 on it and 89.75 on navtest (EP 90.8 against 83.4). The subset's scenario mix differs from navtest's, so a drop from navtest to the subset mixes degradation with composition.
- **Scene-sampling error is at least about 0.7 PDMS** (navtest's per-scene sd of 17.84 over √655), and probably more, since most models fail drivable area in about a quarter of its scenes.

Its main effect (+13.7 for a fine-tuning loss) is far outside that error; its loss-weight increments of 2.4–3.5 are a few standard errors each.

### Run-to-run scatter along a scaling axis on navhard {#scaling-scatter}

*(2026-10-06)* [[sources/simscale.md]] trains a separate model at each of five data sizes, for ten planner–label combinations, and plots navhard EPDMS for each. Each step adds 7–33% more training data, so the curves should be smooth if run-to-run noise were small. Read from the enlarged figure:

| Curve | Five consecutive points (navhard EPDMS) | Largest reversal |
|---|---|---:|
| DiffusionDrive, recovery labels | 29.0, 30.5, 27.8, 28.6, 30.0 | −2.7 |
| LTF, planner labels | 29.7, 30.1, 30.2, 29.3, 30.2 | −0.9 |
| GTRS-Dense ResNet-34, recovery labels | 41.7, 41.3, 42.5, 41.7, 40.8 | −0.9 |
| GTRS-Dense V2-99, planner labels | 43.4, 44.9, 44.6, 46.5, 45.6 | −0.9 |

- **Independently trained neighbours reverse by up to 2.7 EPDMS.** This is not a training-seed measurement (the data changes too), but it is the wiki's first look at how much separately trained navhard models scatter around a smooth trend. It is consistent with the scene-sampling SE of about 1 point estimated above for 244 Stage-1 scenes.
- **The same paper argues from differences of that size**: reward-only against expert labels (+0.3 / +0.8), reactive against non-reactive (+0.7 / +1.1), a ResNet-34 ensemble (+0.8).
- **Its tables also carry a selection effect.** Tables 1 and 2 report, for each planner, the best of two or three label variants, chosen on the test benchmark.

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

**Two more floors, from the tokenizer** *(2026-10-02)*. [[sources/learning-to-drive-from-a-world-model.md]] measures what the image VAE alone costs before any generation:
- **LPIPS 0.148** for VAE-reconstructed test frames against the originals. No generated frame can score below it.
- **A pose estimator's error** on real video, before and after VAE compression: forward-speed MAE 0.464 → 0.594 m/s (+28%), yaw-rate MAE 0.00211 → 0.00254 rad/s (+20%), lane-line position 0.159 → 0.160 m (+1%). Pose errors measured on generated video should be read against the compressed column.

Together with PhysWAM's 0.29° yaw floor, these are the wiki's three measured floors for checks on generated video.

### Selection effects that inflate single numbers

- [[sources/adaptive-wam.md]] reports the **validation-best checkpoint per seed, aggregated over ten seeds**. That is a selection procedure, not a variance estimate, and it is optimistic relative to a single run.
- [[sources/suv.md]] selects checkpoints on the NAVSIM validation PDM score.
- [[sources/dial.md]] compares methods at their best held-out checkpoint, even though several later collapse.
- Best-of-N oracle rows ([[concepts/best-of-n.md]]) are the extreme case.
- [[sources/simscale.md]] reports the best of two or three training-label variants per planner on the test split, and its tables are 0.5–1.9 EPDMS above every point its own scaling figure plots for the same GTRS-Dense configurations ([[sources/simscale.md#figure-table]]).

### Too little variance: ablation columns that are functions of each other {#column-regularity}

Every other entry on this page is about noise that papers under-report. [[sources/momworld.md]] raises the opposite check: whether a table has as much scatter as independent runs should.

| What was checked | What independent runs would show | What MomWorld's tables show |
|---|---|---|
| Sign of each step in a cumulative component study | Some metrics flat or reversed at some steps | All 45 increments (9 metrics × 5 steps) improve, each column by a near-constant step |
| Two different metrics across ablation rows (L2@6 and TPC@6) | Correlated, with scatter | Difference 0.84–0.88 m over 27 rows (sd 0.008); across published methods it spans 0.66–0.96 |
| The same metric at two horizons (L2@3, L2@6) | Correlated, with scatter | L2@6 = 2.31 + 3.0·(L2@3 − 0.86) to the printed digit for 7 of 8 ablated variants, and within 0.01 for the eighth |
| Two benchmarks with different base models (NAVSIM PDMS, nuScenes L2) | Weakly related at best | PDMS ≈ 90.2 − 20·(L2@3 − 0.86) within 0.2 for 7 of 8 ablated variants |

The wiki draws no conclusion about cause. It treats the affected effect sizes as unverified and does not cite them on concept pages.

A single instance of the same kind *(2026-10-12)*: in [[sources/idol.md]], two different configurations (Concat-MLP fusion, Table 13; a 128-anchor vocabulary, Table 14) print identical values in all six columns (98.7 / 97.6 / 95.5 / 100 / 83.8 / 89.8). One coincidence at one-decimal precision is possible; it is recorded, not read as a regularity.

**The check is cheap and general.** For any ablation table with four or more rows and two or more metrics: difference adjacent rows, and regress one column on another. Sampler noise alone is 0.01–0.3 on NAVSIM and training-seed noise is larger, so residuals that vanish at the printed precision need an explanation. The known benign cause is a column that is *computed* from another (an average, or a closed-form aggregate), which is why the check is only informative between separately measured quantities.

### Outside NAVSIM

[[sources/driving-wm-counterfactuals.md]] reports means over five seeds with a **±maximum deviation** (not a standard deviation). This is the only multi-seed generative evaluation in the wiki.

**Closed-loop CARLA spread over three runs** *(2026-10-06)*. [[sources/think2drive.md]] reports mean ± sd over three runs on 390 single-scenario CARLA routes (whether training or evaluation runs is not stated):

| Policy | DS | Weighted DS | Route completion |
|---|---:|---:|---:|
| Think2Drive (model-based RL) | 83.8 ± 1 | 89.0 ± 0.2 | 99.6 ± 0.1 |
| PPO reimplementation | 57.5 ± 9 | 54.8 ± 0.5 | 96.4 ± 1.1 |
| TCP camera student | 36.4 ± 12.2 | 29.6 ± 0.2 | 85.9 ± 8.3 |

- **Closed-loop driving scores can move by ten points between runs** for weaker policies, against 1 point for the strong one. One run of a weak CARLA policy says little.
- **Two metrics computed from the same runs disagree about the spread by a factor of 18–60.** The paper's Weighted DS has sds of 0.2–0.5 beside DS sds of 9–12. Its definition (route completion times penalties raised to infraction counts over scenario density) does not obviously damp variance that much, and the paper does not explain it. Treat its spread as unverified.

**Private 3DGS benchmarks: clip counts, and two that disagree** *(2026-10-06)*. [[sources/rad.md]] reports collision and deviation as fractions of 337 reconstructed clips. Near a collision ratio of 0.15 the binomial SE is about 0.019 (six clips), so its headline (77 → 30 clips) is robust and several of its ablation differences (49–56 clips) are not. The same methods on [[sources/senna2.md]]'s 256-clip 3DGS benchmark, built by the same institutions:

| Method | CR, RAD's benchmark | CR, Senna-2's benchmark |
|---|---:|---:|
| TransFuser | 0.320 | 0.435 |
| VADv2 | 0.270 | 0.422 |
| RAD | 0.089 | 0.281 |

- **The ordering holds; the effect size does not.** RAD's advantage over VADv2 is 3.0× on one benchmark and 1.5× on the other, and its own collision ratio triples. The benchmarks differ in clips, agent behaviour (replay vs. reactive, by each paper's description) and perhaps checkpoint.
- **Rule**: a ratio on a private closed-loop benchmark supports "better on that benchmark", with a binomial interval from the clip count; its size does not transfer.

**Field metrics from unmatched cohorts** *(2026-10-02)*. [[sources/learning-to-drive-from-a-world-model.md]] compares two deployed openpilot steering policies by the share of driving that was engaged:

| Policy | Trips | Engaged time | Engaged distance |
|---|---:|---:|---:|
| Trained in a reprojective simulator | 47,047 | 27.63% | 48.10% |
| Trained in a world-model simulator | 40,026 | 29.92% | 52.49% |

- The paper gives "approximately two months of driving from a cohort of 500 users" and nothing on how users, periods, vehicles or routes were assigned to the two policies. Engagement in a Level 2 system depends on all of them, and on driver trust.
- No interval is given. The trip counts are large, but trips are clustered within about 500 users, so the effective sample is closer to the user count. The larger uncertainty is the comparison design.
- The same paper's closed-loop tests are pass counts over 24 and 20 scenarios, where the two policies differ by one scenario.
- **Rule**: a real-world usage metric without a matched design supports "deployable", not "better".

**nuScenes ablations below sample granularity** *(2026-10-02)*. [[sources/resworld.md]]'s ablation tables separate variants by 0.02–0.11 percentage points of average collision rate. On a validation split of roughly 6,000 samples, 0.04 points is two or three samples; the paper's headline ablation conclusion (future supervision hurts: 0.17% → 0.21%) rests on a difference of that size, from single runs. The same arithmetic applies to MomWorld's Turning-nuScenes margins ([[concepts/nuscenes-waymo-evals.md#the-robustness-subsets]]).

---

## What This Means for Reading the Wiki

1. **Sampler noise is usually not the binding constraint, but it is model-specific.** *(2026-09-30: [[sources/physwam.md]] measures 0.24–0.30 for a model that samples a full multi-view future. Check the paper's own figure before applying the 0.01–0.05 range below.)* At 0.01–0.05 it is 2–3 orders of magnitude below the **+2.0 to +3.8** EPDMS shift from the [evaluator correction](navsim-benchmark.md#evaluator-drift-this-table-mixes-two-protocols). Protocol drift matters far more than seeds for cross-paper comparison.
2. **Ablation deltas under ~0.5 should be read as "no detectable effect".** Examples: SimWAM's mask ablation (0.2 spread), SUV's per-stream access on navtest (≤0.2), Qwen-Drive's reasoning effect (+0.4), GRAVA's +0.18 margin for the best purely autoregressive model, [[sources/walt.md]]'s +0.41 PDMS headline (its v1 and v2 sub-scores attribute the gain to different terms: NC and TTC on v1, extended comfort on v2). This is not because sampler noise is that large, but because training-seed variance was unmeasured and plausibly several times larger. *(2026-10-06: measured once, at sd 0.1 PDMS over three seeds for [[sources/foredrive.md]]'s recipe; see [above](#training-seed). The rule stays as a default for papers that report no seeds.)*
3. **navhard effects need to be several points to be interpretable.** The recent pattern of "invisible on navtest, large on navhard" ([[concepts/wam-attention-masks.md]]) is the most important open finding that currently rests on single runs over a few hundred scenes.
4. **Headline margins of 0.1–0.7 between top methods are within unmeasured training noise.** Examples: SUV vs Metis top-6, GRAVA vs Curious-VLA, SUV vs Poutine on WOD-E2E (+0.03 RFS).
5. **Subsampled evaluations need their own error bar.** A 100-scene NAVSIM subset has an SE of roughly 2–4 points and reordered one paper's own variants ([above](#subset-reordering)).
6. **An ablation table can also be too clean.** Monotone gains on every metric at every step, or columns that are exact functions of each other, are a reason to withhold the effect sizes ([above](#column-regularity)).

---

## Open Questions

*Wiki-wide open questions are collected in [[research-directions.md]].*

- **What is training-seed variance for a NAVSIM planner?** Three seeds of one cheap model (a DiffusionDrive-class planner trains in hours) would bound it and settle rule 2 above. *(2026-10-06: done once, by [[sources/foredrive.md]], for exactly such a planner: sd 0.1 PDMS over three seeds. Still unmeasured for large pretrained models, for navhard, and for v2 EPDMS.)*
- **What is paired per-scene variance between two published checkpoints?** It needs only released checkpoints and the evaluator, no training. *(2026-09-30: measured within one paper by [[sources/drivereferee.md]]: a 95% half-width of about 0.21 EPDMS for two checkpoints of one model family; see [above](#paired-ci). Across two different published methods it is still unmeasured.)*
- **What is navhard's per-scene SD?** Any paper that reports per-scene navhard scores would let every navhard effect in the wiki be given an error bar. *(2026-10-06: still unreported. [[sources/simscale.md]]'s scaling curves give an indirect bound: separately trained models one data step apart reverse by up to 2.7 EPDMS; see [above](#scaling-scatter).)*

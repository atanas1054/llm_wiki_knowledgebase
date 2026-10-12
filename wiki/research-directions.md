---
title: Research Directions
type: directions
sources: []
related: [sources/idol.md, sources/hyworldvla.md, sources/world4scorer.md, sources/rad.md, sources/foredrive.md, sources/raw2drive.md, sources/think2drive.md, sources/simscale.md, sources/resworld.md, concepts/data-driven-simulators.md, sources/learning-to-drive-from-a-world-model.md, concepts/evaluation-variance.md, concepts/navsim-benchmark.md, concepts/alpasim-benchmark.md, concepts/hugsim-benchmark.md, concepts/bench2drive.md, concepts/navhard-ood-evaluation.md, concepts/physicalai-av-benchmark.md, concepts/world-model-for-ad.md, concepts/wam-attention-masks.md, concepts/counterfactual-prediction.md, concepts/perception-for-planning.md, concepts/teacher-pseudo-labels.md, concepts/visual-tokenization.md, concepts/chain-of-thought-for-ad.md, concepts/reasoning-faithfulness.md, concepts/general-capability-retention.md, concepts/dual-system-vla.md, concepts/adaptive-routing.md, concepts/selection-based-planning.md, concepts/inference-time-safety.md, concepts/divergent-thinking-in-vlms.md, concepts/r1-zero-like-training.md, concepts/mixture-of-experts.md, concepts/inference-latency.md, concepts/best-of-n.md, concepts/foundation-backbones-for-ad.md]
created: 2026-09-30
updated: 2026-10-12
confidence: medium
---

# Research Directions

The wiki-wide list of open research questions, grouped by theme. Each entry states a question, what the ingested papers say about it so far, and where the detail lives. The "Open Questions" section of each concept page remains the full local record; this page collects the questions that matter across pages.

Every claim traces to a wiki page. The file is checked and updated on every ingest and every lint. Answered questions are moved to [Answered](#answered), not deleted.

Entry format:

```
### The question, as a heading
- **Status**: open | partially answered
- **Known so far**: the evidence to date, with numbers and [[page]] citations
- **See**: [[pages]] that discuss it
- **Updated**: YYYY-MM-DD
```

---

## 1. Evaluation and benchmark validity

### How large are the error bars on NAVSIM and navhard comparisons?
- **Status**: partially answered
- **Known so far**: Until 2026-10-06 training-seed variance had never been measured. The three earlier figures are sampler-seed noise: 0.053 over 10 seeds ([[sources/wa-jepa.md]]), 0.013 over 6 ([[sources/coworld-vla.md]]) and 0.24–0.30 ([[sources/physwam.md]]). Paired scene-sampling noise is measured once, within one model family: a 95% half-width of about 0.21 EPDMS on navtest ([[sources/drivereferee.md]]). Across two different published methods it is unmeasured, and no paper reports per-scene navhard scores, so no navhard effect in the wiki has an error bar. [[sources/simscale.md]] gives the first look at run-to-run scatter on navhard: separately trained models one data step apart (7–33% more data) reverse by up to 2.7 EPDMS along curves that should be smooth. That is not a training-seed measurement, since the data changes too, but it suggests navhard differences under about 2 points need more than one run. Outside NAVSIM, closed-loop CARLA is noisier still: [[sources/think2drive.md]] reports DS standard deviations of 1, 9 and 12.2 over three runs for three policies on 390 single-scenario routes. **Training-seed variance has now been measured once**: [[sources/foredrive.md]] retrains a 118M single-camera planner from three seeds, 89.7–89.9 PDMS on navtest, sd 0.1. One recipe, three seeds, navtest only. Private closed-loop benchmarks add their own: [[sources/rad.md]]'s 337-clip collision ratio has a binomial SE near 0.02, and the same methods on a second 3DGS benchmark from the same institutions ([[sources/senna2.md]]) keep their order but halve RAD's advantage. [[sources/world4scorer.md]] reports the first log-clustered intervals, bootstrapping over the 136 navtest logs rather than over scenes (a fixed-pool scorer's regret: 5.14 [4.66, 5.62]); its training ablations are single runs. [[sources/hyworldvla.md]] illustrates both halves of the problem. Its named navtest components are worth 0.24–0.42 PDMS each, single runs, near ForeDrive's training-seed sd of 0.1. Its private 655-scene rain/fog subset has a scene-sampling SE of at least about 0.7 PDMS, and two pixel-only models of the same recipe, 0.29 apart on navtest, are 8.77 apart there.
- **See**: [[concepts/evaluation-variance.md]], [[sources/simscale.md]], [[sources/think2drive.md]], [[sources/rad.md]], [[sources/world4scorer.md]], [[sources/hyworldvla.md]]
- **Updated**: 2026-10-12

### Can NAVSIM-v2 EPDMS be compared across papers at all?
- **Status**: open
- **Known so far**: There are three EPDMS lineages (the Hydra-MDP++ formula, the pre-fix evaluator and the corrected one), and at least eleven ingested v2 tables mix conventions. Two papers score Transfuser at 76.7 and 84.0 from identical sub-metrics. The residual-sign heuristic (+1 to +4 for pre-fix rows, slightly negative for corrected ones) identifies the protocol of most rows but has exceptions (four until 2026-10-06, three since; see below). The current rule is to trust a column label only for the paper's own row. navhard has its own provenance problems. [[sources/simscale.md]]'s camera-only LTF (24.4) and DiffusionDrive (27.5) baselines are digit-identical to rows Metis labels "copied from GTRS", while its own scaling figure starts LTF at about 27.5. The "SimScale 53.2" row in DriveFuture's navhard table does not appear in the SimScale paper, whose best is 50.9. SimScale's navtest table is pre-fix throughout by residual sign (+1.4 to +2.9) and names no evaluator. [[sources/foredrive.md]]'s v2 table is another unlabelled mix (five pre-fix-like rows, nine corrected-like), and its "+2.2 over Drive-JEPA" crosses evaluators; it also introduces a corrected-like ReCogDrive-IL row at 86.6 beside the shared block's pre-fix 83.6. [[sources/world4scorer.md]]'s v2 table is the most uniformly corrected-like yet (18 of 21 rows with negative residuals). It prints DiffusionDrive at **88.2** on sub-scores within 0.3 of the 84.5 row's, as GeoWAM did without being cited, with a corrected-like residual (−0.9). The simpler reading is now that 84.5 is the pre-fix value, which removes one of the heuristic's four exceptions and reverses the wiki's earlier verdict that GeoWAM's row was the anomaly. The same table prints TransFuser at 83.9 on its own corrected-like sub-scores; new rows for Drive-JEPA (93.7 / 90.8), DrivoR (90.7) and SafeDrive (87.5, residual +4.0, the largest of any planner row); and ReCogDrive and WAM-Flow with EC 27.6 and 33.8 against 87.7 and 73.9 elsewhere. See also the new question on extended comfort below. The re-reading has knock-on effects that the 2026-10-06 lint found unpropagated: [[sources/adaptive-wam.md]]'s and [[sources/geoworldad.md]]'s tables were counted among the mixing tables because their DiffusionDrive 84.5 was taken as corrected. Under the new reading Adaptive-WAM's three named baselines are all pre-fix (its human agent is the pre-fix 90.3), GeoWorldAD's mix rests only on its DVGT-2 and EponaV2 rows, both "corrected" rows in [[sources/lwdrive.md]]'s table carry pre-fix-like residuals, and [[sources/redrive.md]]'s table, captioned corrected, carries one pre-fix row. So the "at least eleven" count of mixing tables needs recounting: up to three tables may drop out and ReDrive's may join. In the shared baseline block, TransFuser, ARTEMIS and DiffusionDrive would all be pre-fix, DriveVLA-W0 86.1 (residual −1.3) stays the odd one, and Hydra-MDP++ 81.4 and DriveSuprim 83.1 match neither cohort. Which source first printed 84.5, and under which evaluator, would settle it. A new failure mode, shifted rows: [[sources/hyworldvla.md]]'s v2 table prints DreamerAD's published row under Latent-WAM's name (87.7, EC 72.4) and DriveWorld-VLA's 86.8 row under DreamerAD's, so Latent-WAM's own 89.3 is missing and the claimed margin over it is +2.0 instead of +0.4. The same table is the tenth recorded user of the shared baseline block in one unlabelled column: the block's TransFuser, HydraMDP++, DriveSuprim, DiffusionDrive and ReCogDrive rows are pre-fix-like (+1.25 to +2.48), while the paper's own row (−0.91) and ExploreVLA (−0.68) are corrected-like. [[sources/idol.md]] extends the residual-sign fingerprint to navhard. In its stage-1-only table every baseline's reported score sits 1.8–7.6 below the closed form of its own sub-scores and IDOL's 0.4 above; by closed form Hydra-MDP (V2-99) leads IDOL 80.4 to 75.9, reversing the claimed +3.1. Its two-stage table shows the same split more weakly (product of stage closed forms: baselines 0.2–2.2 above the reported score, IDOL 2.6 below). Its navtest v2 table is another mix: IDOL (−0.60) and MeanFuser (−0.90) corrected-like, the pre-fix human agent (90.3) and five of the shared block's six rows pre-fix-like, plus an unidentified "VADv2 85.8" row.
- **See**: [[concepts/navsim-benchmark.md]], [[sources/hydra-mdp-pp.md]], [[sources/simscale.md]], [[sources/foredrive.md]], [[sources/world4scorer.md]], [[sources/adaptive-wam.md]], [[sources/geoworldad.md]], [[sources/hyworldvla.md]], [[sources/idol.md]], [[concepts/navhard-ood-evaluation.md]]
- **Updated**: 2026-10-12

### How much of NAVSIM-v2 EPDMS measures frame-to-frame consistency rather than planning?
- **Status**: open
- **Known so far**: On frozen candidate pools scored with simulator sub-scores, v1 and v2 choose different candidates in 43% of scenes, extended comfort accounts for 98% of the utility gap, and choosing for v2 costs 1.3 PDMS on v1 ([[sources/world4scorer.md]]). Selecting with the benchmark's own EC criterion against the previous frame's plan adds about +1.6 EPDMS to two different planners at fixed candidates (World4Scorer 91.4 → 93.0, DrivoR 90.66 → 92.27) and lifts EC above the human agent's. [[sources/drive-jepa.md]]'s momentum-aware selector is the same kind of fix (EC 47.9 → 84.8), and per-frame selection without one scores EC 35–44 ([[sources/ad-e2e-jepa.md]]). Both fixes need an agent that sees navtest frames in log order and remembers its last plan, which standard NAVSIM evaluation does not assume. World4Scorer's changed selections move closer to the human plan (1.87 → 1.51 m), but no paper tests whether the EC gain shows up as smoother closed-loop driving.
- **See**: [[concepts/navsim-benchmark.md]], [[concepts/selection-based-planning.md]], [[sources/world4scorer.md]], [[sources/drive-jepa.md]]
- **Updated**: 2026-10-06

### Is NAVSIM-v1 saturated?
- **Status**: open
- **Known so far**: Best-of-6 reaches 94.8 PDMS, the human ground-truth score (Curious-VLA). Single-pass scores now reach 94.0, trained on navtrain ([[sources/world4scorer.md]]) and with the val split added ([[sources/mm-future.md]]), above CLEAR and DA-WAM at 93.7, and almost every entry above 92.0 trains a scorer or an RL reward against the benchmark's own metric. If oracle selection already matches the logged human, gains above about 93 may measure selection quality and not driving quality. World4Scorer's 94.0 comes from a scorer trained on simulator outcomes. On DrivoR's released 64-candidate pools the oracle is 99.33 PDMS, DrivoR's own scorer 93.69 and World4Scorer's 94.19: about 5 points of selection headroom remain over a generated pool.
- **See**: [[concepts/best-of-n.md]], [[concepts/navsim-benchmark.md]], [[concepts/selection-based-planning.md]], [[sources/world4scorer.md]]
- **Updated**: 2026-10-06

### Do NAVSIM rankings survive a reactive closed loop?
- **Status**: partially answered
- **Known so far**: The only closed-loop reproduction of NAVSIM leaders inverts the order: in AlpaSim, SimWAM (91.5 PDMS) is last on at-fault score (0.30) and Alpamayo-R1, which reports no NAVSIM number, is first (0.58) ([[sources/qwen-drive-1.0.md]]). That is one group's reproduction of two methods not designed for the simulator. Of the entries above 92 PDMS, two have a closed-loop number: [[sources/mm-future.md]] (93.4), 32.3 HD-Score on HUGSIM, below WA-JEPA's 44.6 (91.8 PDMS), and World4Scorer on Bench2Drive (below). None has been run in AlpaSim, and no paper has run AlpaSim and HUGSIM side by side, so it is unknown whether the two closed-loop benchmarks agree with each other. Outside NAVSIM, [[sources/learning-to-drive-from-a-world-model.md]] shows the same inversion inside one paper: the policy with the best held-out trajectory error (0.361, behaviour cloning) passes 5/24 closed-loop lane-centre tests, while two on-policy policies with worse error (0.369, 0.394) pass 24/24. The second: [[sources/world4scorer.md]] (94.0) scores 73.27 DS / 46.36% SR on Bench2Drive as an adapted system (CARLA ground-truth ego state, a route point 20 m ahead, route re-ranking), below twelve entries on that ladder. The adaptation is unablated and its baselines are not matched, so it is weak evidence either way. A small within-paper check on Bench2Drive: [[sources/idol.md]] lists three ResNet-34 BEV world-model planners on both. IDOL leads both (90.0 PDMS, 63.22 DS), but its two baselines swap: SeerDrive is above WoTE on NAVSIM (88.9 vs 88.3) and below it on Bench2Drive (58.32 vs 61.71). The CARLA adaptation and training data are not described, and 63.22 DS is below every non-baseline entry on the wiki's Bench2Drive ladder.
- **See**: [[concepts/alpasim-benchmark.md]], [[concepts/hugsim-benchmark.md]], [[concepts/bench2drive.md]], [[concepts/data-driven-simulators.md]], [[sources/world4scorer.md]], [[sources/idol.md]]
- **Updated**: 2026-10-12

### How much of a navhard score is the generator and how much is the selector?
- **Status**: partially answered
- **Known so far**: [[sources/drivefuture.md]] is the only paper to report one checkpoint with and without a scorer: 30.9 → 34.6 (its world model) → 55.5 (GTRS-Dense over 100 proposals). Every navhard entry above 42 scores its own candidates and none of the others reports an unscored ablation. [[sources/momworld.md]] puts a predicted future inside a strong scorer and is +1.1 over GTRS-Dense's published score, but that base is not a matched rerun and the page is held at low confidence. Whether future conditioning survives a strong scorer is still open. On navtest there are two partial measurements: [[sources/physwam.md]] reports first sample, a medoid-of-8 selector and the oracle on one checkpoint (90.3 → 90.4 EPDMS for the selector), and [[sources/mm-future.md]] prices a learned scorer at +8.2 PDMS between a one-trajectory and a 32-trajectory model. A third ingredient is now priced: training data. [[sources/simscale.md]] adds 3DGS-rendered off-log scenes with corrective labels and gains +5.1 to +5.8 for two unscored planners and +6.1 to +8.6 for a scorer (GTRS-Dense), so data and selection add rather than substitute. The 53.2 "SimScale" entry in the scorer cohort is therefore a data-plus-scorer pipeline. A navtest scorer swap at fixed candidates now exists: on DrivoR's released pools, [[sources/world4scorer.md]]'s scorer scores 94.19 PDMS against DrivoR's own 93.69 (pool oracle 99.33; no ranking signal 71.10). It reports no navhard, although DrivoR is second on that leaderboard. [[sources/idol.md]] scores its 256 refined anchors with WoTE's reward model (trained on five cached PDM metrics) and reaches 38.0 with a camera-only ResNet-34 variant, the lowest scored entry in the cohort; it reports no unscored or unrefined navhard run, and its margins over its baselines cross evaluator scales.
- **See**: [[concepts/navhard-ood-evaluation.md]], [[concepts/selection-based-planning.md]], [[sources/simscale.md]], [[sources/world4scorer.md]], [[sources/idol.md]]
- **Updated**: 2026-10-12

### Is the navhard Stage-2 lane-keeping collapse a planner failure or a rendering artifact?
- **Status**: partially answered
- **Known so far**: Every method, including constant velocity, loses 35–45 points of lane keeping in Stage 2. If 3DGS renderings degrade as the ego pose leaves the recorded trajectory, part of the drop measures the benchmark. A candidate scorer recovers 10 points of Stage-2 LK on a fixed checkpoint, which fits either explanation. No paper has separated them. The rendering explanation is plausible: in a reprojective simulator, [[sources/learning-to-drive-from-a-world-model.md]] reports artifacts that grow with the ego's offset from the log (range kept under 4 m) and correlate with it strongly enough for a policy to exploit. That is reprojection, not 3DGS. **Partly answered by a planner that sees no rendering.** PDM-Closed, run on ground-truth state, loses 25.6 points of Stage-2 lane keeping (99.3 → 73.7), with NC, DAC and TTC also down 6–10 points ([[sources/simscale.md]]). At least that much of the drop is in the Stage-2 scenarios. Learned planners trained on real data lose 37–50 points in the same table, so 10–25 points remain unexplained. Co-training on 3DGS renders of off-log states recovers 10–14 of them for unscored planners (LTF 47.9 → 58.3, DiffusionDrive 46.4 → 60.8), which still fits both explanations.
- **See**: [[concepts/navhard-ood-evaluation.md]], [[concepts/data-driven-simulators.md]], [[sources/simscale.md]]
- **Updated**: 2026-10-06

### Does a navhard gain from 3DGS training data measure recovery or renderer familiarity?
- **Status**: open
- **Known so far**: [[sources/simscale.md]] co-trains planners on 3DGS-rendered off-log states and gains most where the test is also 3DGS-rendered at displaced poses (navhard Stage 2), less in Stage 1 and least on real-image navtest. Rendering quality matters by up to 1.6 EPDMS (PSNR ≥ 27 vs < 27). The training engine and the benchmark may share a codebase. The separating experiment, rendering the unperturbed log through the same engine with the human label, has not been run; nor has a test on a non-3DGS closed-loop benchmark.
- **See**: [[sources/simscale.md]], [[concepts/navhard-ood-evaluation.md]], [[concepts/data-driven-simulators.md]]
- **Updated**: 2026-10-06

### Does the expert that generated Bench2Drive's data bound its leaderboard?
- **Status**: open
- **Known so far**: Bench2Drive's official training data comes from [[sources/think2drive.md]], a privileged BEV planner trained by model-based RL that scores 91.85 DS / 85.41 SR on the benchmark; the rule-based PDM-Lite, the other expert protocol, scores 97.02 / 92.27 (both from [[sources/dynvla.md]]'s table). The best Think2Drive-protocol student in the wiki is 5.6 DS below its teacher ([[sources/deepsight.md]], 86.23) and LinkVLA, protocol undisclosed, is 0.84 below it. Papers disagree about which protocol some methods used. No paper reports a base-set-only student above the expert, or retrains one method under both protocols. Tighter coupling to the expert does not help: [[sources/raw2drive.md]] trains a camera policy with Think2Drive's own world model, reward heads and policy and lands 20.5 DS below it (71.36), further than imitation students of the expert's data. Its tables also give the teacher's ability profile: give-way 90.00, where every student in the wiki scores 0–50, so that weakness is not inherited from the expert.
- **See**: [[concepts/bench2drive.md]], [[concepts/pdm-lite.md]], [[sources/think2drive.md]], [[sources/raw2drive.md]]
- **Updated**: 2026-10-06

### Do private 3DGS closed-loop benchmarks agree with each other?
- **Status**: open
- **Known so far**: Two 3DGS benchmarks built by the same institutions (HUST and Horizon Robotics) score the same four imitation methods in the same order but RAD's collision ratio at 0.089 ([[sources/rad.md]], 337 clips, log-replayed agents) and 0.281 ([[sources/senna2.md]], 256 clips, agents described as reactive), so RAD's advantage over VADv2 is 3.0× on one and 1.5× on the other. Both are private. The public reconstructions in the wiki (navhard's Stage 2, HUGSIM, AlpaSim) have not been run against these. Which differences (clip selection, agent behaviour, checkpoint) drive the gap is unknown.
- **See**: [[sources/rad.md]], [[concepts/evaluation-variance.md]], [[concepts/data-driven-simulators.md]]
- **Updated**: 2026-10-06

### Can HUGSIM results be put on one table, and does anything move its Extreme tier?
- **Status**: open
- **Known so far**: HUGSIM numbers fall into two incompatible eras (345 and 436 scenarios) and the controller commit is unstated for [[sources/physwam.md]], [[sources/mm-future.md]] and Latent-WAM. Every ingested method scores 0.06–0.14 on the Extreme tier. NAVSIM-only models were thought to collapse after the Easy tier (PhysWAM 86.9 → 30.1, Latent-WAM 72.5 → 24.0, against WA-JEPA 79.8 → 55.6), but MM-Future, also NAVSIM-only, has the opposite shape (Easy 53.8, Medium 40.0). Pretraining data, camera count and objective are confounded.
- **See**: [[concepts/hugsim-benchmark.md]]
- **Updated**: 2026-09-30

### Will PhysicalAI-AV converge on one protocol, and where does data scaling stop helping?
- **Status**: open
- **Known so far**: Two papers use two protocols and report opposite orderings; DriveWAM's ordering against Alpamayo-1.5 inverts between subsets by a factor of about 3. The leakage-free subset costs 0.04–0.06 m uniformly. Both scaling studies on the benchmark stop while still improving, so the knee on real logs has not been found.
- **See**: [[concepts/physicalai-av-benchmark.md]]
- **Updated**: 2026-09-30

---

## 2. World models: what the predicted future is for

### Does world modeling buy open-loop accuracy or closed-loop robustness?
- **Status**: partially answered
- **Known so far**: Four effects on navtest / navhard point the same way: [[sources/geowam.md]] future geometry +0.6 / +4.9, [[sources/suv.md]] future access +0.3 / +4.1, [[sources/metis.md]] asymmetric mask +0.5 / +2.2 and video-prior scale ≈0 / +2.4. One mechanism breaks the pattern: [[sources/physwam.md]]'s geometric loss is worth +1.9 on both. Most world-model papers optimize and report navtest only, which may be the wrong benchmark for the question. By analogy only: when a world model is used as the training environment instead ([[sources/learning-to-drive-from-a-world-model.md]]), held-out trajectory error gets worse (0.361 → 0.394) and closed-loop lane-centre tests go from 5/24 to 24/24. A caution on reading the asymmetry: [[sources/simscale.md]] shows the same navtest-small / navhard-large shape with no world model at all. Co-training on 3DGS-rendered off-log states is worth +0.8 to +2.9 on navtest and +5.1 to +8.6 on navhard for three planner families. So the asymmetry may belong to anything that exposes a planner to off-log states (or to 3DGS images), not to world modeling in particular. The same shape appears off navhard: [[sources/hyworldvla.md]]'s fine-tuning latent loss is worth +0.42 PDMS on navtest and +13.7 on a private 655-scene rain/fog subset of the OpenScene test split. That subset is weakly calibrated (see the new question on weather robustness below). [[sources/idol.md]] is +1.7 PDMS over WoTE on navtest, +10.1 on two-stage navhard and +1.51 DS on Bench2Drive. The navhard gap is probably about half that on a common evaluator scale, and IDOL's mechanism ablations are navtest-only, so it does not separate the two readings.
- **See**: [[concepts/world-model-for-ad.md]], [[concepts/navhard-ood-evaluation.md]], [[sources/simscale.md]], [[sources/hyworldvla.md]], [[sources/idol.md]]
- **Updated**: 2026-10-12

### Does generating the future at test time ever pay off?
- **Status**: partially answered
- **Known so far**: On navtest, little: SimWAM finds no benefit over isolation, and [[sources/foresight.md]] pays 870 ms for +0.3 PDMS. The largest navtest effect is a geometric future: [[sources/geoworldad.md]]'s latent future-geometry module, shared by 64 proposals, is worth +1.7 PDMS / +2.8 EPDMS, not compute-matched. On navhard, one-way access is worth +4.1 EPDMS ([[sources/suv.md]]) while the bidirectional version is worst ([[sources/metis.md]]). Whether it matters for long-horizon rollout or reactive closed-loop interaction, which NAVSIM cannot measure, is open. WA-JEPA's ablations never separate its training objective from its inference-time scene denoising. A small navtest effect for a *latent* future: [[sources/foredrive.md]] consumes predicted DINOv3 latents and gains +0.4 to +0.6 PDMS over the same training without consumption, for 26 ms; zeroing them at test costs 1.4 and feeding a wrong future costs 2.4. No navhard. With [[sources/da-wam.md]]'s shared JEPA-latent future at −0.5, shared feature-space futures now disagree in sign on navtest; only photometric shared futures are consistently near zero. A third shared feature-space data point, and a cheap one: [[sources/hyworldvla.md]]'s VLM predicts one video-VAE latent through a query token and an MLP, and letting the action expert read it is worth +0.30 PDMS on navtest (single run, no generation step). Shared feature-space futures now stand at +0.4 to +0.6 (ForeDrive), +0.3 (HyWorldVLA) and −0.5 (DA-WAM). Per-candidate latent futures read back into the candidate: [[sources/idol.md]] rolls out a latent BEV future for each of 256 anchors and refines each anchor's query from its own rollout. Feedback is worth +1.3 PDMS (future states) to +2.7 (a learned encoder of consecutive states plus a second rollout–refine pass) over the same model without feedback, at about 57 ms on an RTX 3090. Not compute-matched against the no-feedback base; navtest only; single runs.
- **See**: [[concepts/world-model-for-ad.md]], [[concepts/wam-attention-masks.md]], [[sources/foredrive.md]], [[sources/geoworldad.md]], [[sources/hyworldvla.md]], [[sources/idol.md]]
- **Updated**: 2026-10-12

### Does world-model supervision buy robustness to sensor degradation (rain, fog, lighting)?
- **Status**: open
- **Known so far**: One NAVSIM test exists. [[sources/hyworldvla.md]] scores 655 rain/fog scenes from the OpenScene test split: 86.87 PDMS for its hybrid model against 73.18 for the same model without the latent loss in fine-tuning, 69.95 for its pixel-only arm, and 61.18 / 67.49 / 60.65 for DriveVLA-W0, DriveLaW and WoTE. The subset score rises with the latent-loss weight (84.52 → 92.94) while navtest falls (90.47 → 89.75). The subset is private, its selection rule is not described, and the baselines' run protocol is not given; at the largest weight the model scores higher on the "noisy" subset than any model in the paper scores on navtest, and two pixel-only models 0.29 apart on navtest are 8.77 apart on it ([[concepts/evaluation-variance.md]]). Its target is a pixel-reconstructing VAE code, so whether the effect needs a latent target at all, or would follow from any regularizer or from weather augmentation, is untested. Its illumination figures use AI-edited images. On nuScenes-C, [[sources/momworld.md]] reports weather-corruption collision rates with margins of 0.001–0.04 points ([[concepts/nuscenes-waymo-evals.md]]). No closed-loop or navhard test of weather robustness exists.
- **See**: [[sources/hyworldvla.md]], [[concepts/world-model-for-ad.md]], [[concepts/evaluation-variance.md]], [[sources/momworld.md]], [[concepts/nuscenes-waymo-evals.md]]
- **Updated**: 2026-10-12

### What does a transition encoder over candidate-conditioned futures encode?
- **Status**: open
- **Known so far**: [[sources/idol.md]] feeds pairs of consecutive predicted BEV states, rolled out under each candidate anchor, through a learned MLP it calls an inverse dynamics model. At equal parameters it beats a reader of the states alone by +1.1 PDMS and of fixed latent differences by +0.9 (single runs, navtest). Because each rollout is generated under its own anchor, the action behind the transition is already known, and the module's only direct target is the expert's correction to that anchor (speed, acceleration and heading-rate differences). The wiki's earlier inverse-dynamics planners read an action from a future generated without it ([[sources/drivewam.md]], [[sources/futuresightdrive.md]]). No probe of the module's features and no ablation of its correction loss is reported, so "inverse dynamics" and "learned residual correction over imagined pairs" are not separated.
- **See**: [[sources/idol.md]], [[concepts/world-model-for-ad.md]], [[sources/drivewam.md]], [[sources/futuresightdrive.md]]
- **Updated**: 2026-10-12

### Who should attend to whom among observation, future and action tokens?
- **Status**: open
- **Known so far**: SimWAM, Metis and SUV share Wan2.2-5B and differ mainly in the mask. On navtest every non-bidirectional mask lands within 0.5 EPDMS; on navhard, future-reads-action is +2.2 and action-reads-future is +4.1, yet the bidirectional mask is worst. No single model has run the full set of cells. WA-JEPA blocks the gradient from action to future while Metis relies on it. Outside that backbone, [[sources/mm-future.md]] finds bidirectional best on navtest, but only with modality-specific branches (+0.4; −0.1 in a single DiT). Every number is a single run. A second stop-gradient result: [[sources/foredrive.md]] lets the action read a predicted future but blocks planning gradients into the predictor, +0.4 PDMS over letting them through at matched final forecast quality.
- **See**: [[concepts/wam-attention-masks.md]], [[sources/foredrive.md]]
- **Updated**: 2026-10-06

### Are per-candidate futures really futures, and do they help at a realistic horizon?
- **Status**: open
- **Known so far**: Two per-candidate-future scorers agree in sign and size: +0.15 PDMS ([[sources/da-wam.md]], 0.5 s futures) and +0.4 for letting the scorer read the future ([[sources/mm-future.md]], 4 s futures). In both, only the expert-matched future is supervised (31 of 32 and 63 of 64 per scene are not). MM-Future shows action sensitivity (futures of extreme trajectories are 17% further apart in RMS), not correctness. [[sources/ad-e2e-jepa.md]] offers a hit-rate instrument (53.8% top-1 among 256) and shows 4 s rollouts cost about 3 ms per candidate, but neither DA-WAM nor MM-Future has been measured with it. A third design avoids unsupervised futures by not predicting appearance for alternatives: [[sources/world4scorer.md]] supervises every candidate's latent state with simulator outcomes and only the logged trajectory's with a +2 s feature. Its tests do not establish that the states are futures: shuffling them breaks selection, but just as much for DrivoR's non-predictive scorer, and they cluster by outcome only slightly more than raw trajectories do (80.2% vs 76.8%, on training scenes). A third use of them: [[sources/idol.md]] refines each candidate from its own rollout before scoring, rather than scoring the candidate against its future, and the refinement is worth more on navtest (+2.7) than either scorer design (+0.15, +0.4). Whether the non-expert rollouts are supervised is not stated, the rollout length is not given, and the per-candidate states are never checked against any future.
- **See**: [[concepts/world-model-for-ad.md]], [[concepts/selection-based-planning.md]], [[sources/world4scorer.md]], [[sources/idol.md]]
- **Updated**: 2026-10-12

### When outcome labels supervise every candidate, does an observed future still add anything?
- **Status**: open
- **Known so far**: [[sources/world4scorer.md]] supervises all 80 candidate states per scene with simulator outcomes and the logged trajectory's state with a pooled front-view DINOv2 feature 2 s ahead. The future target is worth +0.95 PDMS (93.01 → 93.96), and the same objective with a current-frame target −0.31, one seed each, last epoch. Its Proposition 1 bounds local score variation for any added readout, so it predicts a narrowing for the current-frame target as well and cannot explain the difference. Auxiliary future losses elsewhere are worth +0.1 ([[sources/foredrive.md]]) to +0.7 ([[sources/redrive.md]]). There is no seed repeat, no navhard result and no full-scale ablation of the outcome bank.
- **See**: [[concepts/world-model-for-ad.md]], [[concepts/selection-based-planning.md]], [[sources/world4scorer.md]]
- **Updated**: 2026-10-06

### Do the two world-model supervision rules compose?
- **Status**: open
- **Known so far**: Objective form follows target entropy: regression on multi-view scene latents is worse than no future prediction (90.7 vs 91.1 EPDMS, [[sources/wa-jepa.md]]). Target content follows horizon: optical flow is best far out and RGB one step ahead ([[sources/drive-hwm.md]]). Nobody has crossed them. Two results complicate the first rule: L1 regression on future latents helps in [[sources/redrive.md]] (+0.7), and [[sources/geowam.md]] pairs cosine alignment with dense point-map regression without collapsing. It is also unknown which other deterministic predictors (DeepSight, FLARE, Latent-WAM, AD-E2E-JEPA) are losing performance to the same effect. A third data point on the harmful side: [[sources/resworld.md]]'s residual-input world model, trained only by the planning loss, gets worse when its predicted map is supervised toward the real next-frame map (nuScenes collision 0.17% → 0.21%, L2 0.59 → 0.61 m). The loss form is unstated and the difference is a few samples. A regression case on the helpful side: [[sources/foredrive.md]] regresses (L1) single-front-camera DINOv3 latents, and consuming the regressed futures helps (+0.4 to +0.6), consistent with the entropy reading (one camera, a lower-entropy target than WA-JEPA's four). Its predictions are over-smoothed and the planner ignores their order. Another low-entropy regression target on the helpful side: [[sources/world4scorer.md]]'s cosine loss on one pooled 384-d front-view feature 2 s ahead (effective rank 49) is worth +0.95 PDMS in one run. A two-sided case: [[sources/hyworldvla.md]] regresses (MSE) a single-camera video-VAE latent, a pixel-reconstruction code. On navtest it helps at weight 0.1 (+0.42 over none in fine-tuning) and hurts at 0.2 and 1.0 (90.01, 89.75 against 90.59); on its rain/fog subset it helps monotonically up to 1.0 (73.18 → 92.94).
- **See**: [[concepts/world-model-for-ad.md]], [[sources/resworld.md]], [[sources/foredrive.md]], [[sources/world4scorer.md]], [[sources/hyworldvla.md]]
- **Updated**: 2026-10-12

### Geometry, pixels or flow: which future representation helps a planner most?
- **Status**: open
- **Known so far**: [[sources/drivelaw.md]] holds the planner fixed and spans 84.1 → 89.1 PDMS across representation families, but geometry is not in its comparison. GeoWAM argues geometry beats pixels using other papers' numbers. [[sources/geoworldad.md]] prices the coordinate frame alone at +2.5 PDMS, while [[sources/drive-hwm.md]]'s linear probe puts an image-space flow latent first for future ego motion (83.7) and an ego-frame BEV latent last (71.2). No paper compares the three under one planner. Pixels against a latent under one planner: [[sources/hyworldvla.md]]'s Emu3 VLA scores 89.91 PDMS with VQ pixel tokens as the only future target, 87.50 with a video-VAE latent only, and 90.59 with both (pixels in pre-training, the latent kept through fine-tuning). That is the opposite order to DriveLaW's (video latents first), but its latent is an MSE-regressed VAE code, not a generator's internal state, and no run without a future target is reported. Geometry is still absent.
- **See**: [[concepts/world-model-for-ad.md]], [[sources/hyworldvla.md]]
- **Updated**: 2026-10-12

### How much of a "world-model-shaped" representation is world modeling?
- **Status**: open
- **Known so far**: [[sources/redrive.md]]'s encoder keeps its gain under a frozen probe (86.7 → 90.2), but it was fine-tuned by both the planning loss and the future loss, and the paper's own split is +3.5 for unfreezing against +0.7 for future prediction. The same question applies to every training-time-only world model that reports a with/without-future-loss delta only end to end. Related: [[sources/physwam.md]] does not isolate whether its +1.9 EPDMS needs the depth–motion cross term or only a pose loss. [[sources/resworld.md]] is the limiting case: its "world model" has no future loss at all, adding one hurts, and its first-pass trajectory, which has the baseline's architecture, keeps most of the gain (nuScenes 0.65 → 0.61 m, 0.28 → 0.18% collision, against 0.59 / 0.17 for the full model). The shaping is done by the planning loss routed through a future-named branch. [[sources/foredrive.md]] separates forecast quality from usefulness: forecasting as an auxiliary loss is worth +0.1; across four training paradigms, the most accurate forecasts (latent cosine 0.843) plan worst (87.9) and the least accurate (0.759) best (89.9); a frozen probe reads the expert trajectory better from its predicted latents (ADE 1.55) than from the real future latents (2.10); shuffling horizons at test changes nothing, and real future latents add only +0.2. Its "future" works as planning-shaped context. [[sources/world4scorer.md]] runs the control this question needs, once: the same auxiliary objective and pathway with a current-frame target scores 92.70 PDMS, with a +2 s target 93.96, with none 93.01. Single seed, last epoch, and the two targets have mean cosine 0.94.
- **See**: [[concepts/world-model-for-ad.md]], [[sources/resworld.md]], [[sources/foredrive.md]], [[sources/world4scorer.md]]
- **Updated**: 2026-10-06

### Does the generated video follow the generated action?
- **Status**: open
- **Known so far**: [[sources/physwam.md]] is the only joint model that checks: a 0.80° median yaw disagreement over 4 s (floor 0.29°) and 67% agreement where the plan departs from the log. DriveVA, DriveWAM, SUV and WA-JEPA make the same joint-generation claim without reporting it. For an action-*conditioned* model the answer is "partly": [[sources/learning-to-drive-from-a-world-model.md]] runs a pose estimator on generated video and finds a commanded ±0.5 m lateral deviation rendered "not to its full extent", against a floor measured on real and VAE-compressed video (VAE compression alone adds 28% to forward-speed error). The size of the shortfall is in a figure the clipping lacks.
- **See**: [[concepts/world-model-for-ad.md]], [[concepts/data-driven-simulators.md]]
- **Updated**: 2026-10-02

### Can a world model replace the simulator as the source of reward or cost?
- **Status**: partially answered
- **Known so far**: [[sources/dreameraD.md]] shows a latent reward model can replace simulator calls during RL rollout (87.7 EPDMS) but still needs the simulator to annotate its vocabulary. [[sources/ad-e2e-jepa.md]] has cheap 4 s rollouts over 256–8,192 candidates but selects with the ground-truth future frame; that oracle-goal search reaches only 67.3–72.9 EPDMS against 85.4 for the same paper's imitation model. Nobody has attached a learned reward or goal predictor to a compressed rollout model and reported a real planning score. For supervision other than reward, the answer is yes: [[sources/learning-to-drive-from-a-world-model.md]] replaces the simulator entirely, as frame generator and as source of action labels (a future-anchored plan head), and the resulting steering policy is deployed in openpilot. Lateral control only. The earliest instance in the wiki is in CARLA: [[sources/think2drive.md]] trains a privileged BEV planner entirely on a DreamerV3 world model's predicted rewards and terminations (83.8 DS on 390 single-scenario routes, against 57.5 for model-free PPO with the same engineering). As in DreamerAD, the reward head is learned from the simulator's reward, and unlike DreamerAD the simulator keeps supplying the current policy's experience. Where the reward is learned matters: [[sources/raw2drive.md]] finds a reward head on camera latents destabilizes training (10-clip Dev10 DS 83.5 → 46.6, adding a continuation head 34.5), because near-identical frames carry opposite rewards, and takes the reward from a world model of ground-truth BEV instead. DreamerAD's dense multi-horizon reward on video latents is the opposite result under different targets.
- **See**: [[concepts/world-model-for-ad.md]], [[concepts/data-driven-simulators.md]], [[sources/think2drive.md]], [[sources/raw2drive.md]]
- **Updated**: 2026-10-06

### Can a driving world model predict counterfactuals?
- **Status**: partially answered
- **Known so far**: For retrospective counterfactuals over recorded episodes, no: action-conditioned generation recovers 0.38 (Vista) and 0.31 (DrivingWorld), below the 0.5 no-preference point ([[sources/driving-wm-counterfactuals.md]]). Open parts: whether a model trained to condition on the factual continuation (abduction) would beat the paper's depth-plus-splatting transport, whether the failure is causal or a domain shift to CARLA renders, whether anything carries over to decision time, and whether any counterfactual metric predicts planning quality. A model conditioned on the history and the recorded continuation now exists ([[sources/learning-to-drive-from-a-world-model.md]]'s future-anchored world model), but it uses the continuation as an endpoint the rollout must reach and has not been evaluated on counterfactuals.
- **See**: [[concepts/counterfactual-prediction.md]], [[concepts/world-model-for-ad.md]]
- **Updated**: 2026-10-02

### Where and how should a planner read a video prior?
- **Status**: open
- **Known so far**: [[sources/adaptive-wam.md]] shows readout depth matters about 40 times more than noise level (4.80 vs 0.15 PDMS) and that a mid-network exit beats the final block, on one backbone family with one head. Three papers agree a single forward pass suffices and ForeSight disagrees (100 steps). DriveLaW's collapse at ten denoising iterations (89.1 → 23.2) is undiagnosed. No other paper reports which layer it reads.
- **See**: [[concepts/world-model-for-ad.md]]
- **Updated**: 2026-09-30

### Does the scale of the video prior matter?
- **Status**: partially answered
- **Known so far**: On navtest, barely: SimWAM holds the planner fixed and gets 90.2 from Wan2.1-1.3B against 90.3 from Wan2.2-5B, while a weak prior (LTX-Video, 88.7) costs and a driving-pretrained one (Cosmos-Predict2.5, 90.4) helps most. On navhard, [[sources/metis.md]] finds the same 1.3B-versus-5B swap worth 2.4 EPDMS. Scale may matter exactly where navtest cannot see it. Zero-shot transfer, DriveVA's distinguishing claim, is untested.
- **See**: [[concepts/world-model-for-ad.md]], [[concepts/foundation-backbones-for-ad.md]]
- **Updated**: 2026-09-30

### What happens when a planner that consumes a predicted future is given a wrong one?
- **Status**: open
- **Known so far**: [[sources/foredrive.md]] measures it on a fixed checkpoint: removing the predicted future costs 1.4 PDMS (mostly progress and drivable area), while a structured but wrong future (another scene's prediction, or the current latent copied forward) costs 2.4, with TTC down 2 points. Two other papers found the mirror problem on the training side: a planner conditioned on a clean or oracle future it cannot produce at inference degrades ([[sources/drivelaw.md]] 89.1 → 23.2; [[sources/drivefuture.md]] 28.9 when the oracle is held too long). No paper measures how often a deployed world model's prediction is wrong in the way that hurts, or tests a planner that can detect an unreliable future.
- **See**: [[sources/foredrive.md]], [[concepts/world-model-for-ad.md]]
- **Updated**: 2026-10-06

### Does generation quality matter for planning?
- **Status**: open
- **Known so far**: [[sources/reworld.md]] improves nuScenes FVD 81.3 → 61.9 and NAVSIM PDMS 89.1 → 90.4 in one paper through disjoint mechanisms, and never measures whether the video-side objective helps planning. FVD itself is hard to rank on: [[sources/physwam.md]] measures a recorded-versus-recorded FVD floor of 91.5 at 600 clips. The question has a training-environment version: [[sources/learning-to-drive-from-a-world-model.md]] reports LPIPS improving with world-model size and data (in a figure the clipping lacks; VAE floor 0.148) but trains policies in one world model only, so simulator quality is never linked to policy quality. Within one 3DGS renderer, a little: [[sources/simscale.md]] trains on matched scenes from reconstructions with novel-view PSNR ≥ 27 and < 27, and the high-fidelity data wins all eight cells by +0.1 to +1.6 navhard EPDMS. For latent forecasts the relation inverts within one paper: [[sources/foredrive.md]]'s better-aligned forecasts come with lower PDMS (87.9 at cosine 0.843 vs 89.9 at 0.759).
- **See**: [[concepts/world-model-for-ad.md]], [[concepts/navsim-benchmark.md]], [[concepts/data-driven-simulators.md]], [[sources/simscale.md]], [[sources/foredrive.md]]
- **Updated**: 2026-10-06

### Should a world model be given only what moves?
- **Status**: open
- **Known so far**: Three papers put an explicit motion quantity into a driving world model. As input, [[sources/resworld.md]] feeds only frame-to-frame differences of ego-aligned scene queries and beats a full-scene input by 0.02 m L2 and 0.04 collision points on nuScenes (single runs, a few samples apart); static structure is copied from the current map instead of predicted. As a target, [[sources/drive-hwm.md]] finds optical flow best at long horizons and RGB best one step ahead. As a state, [[sources/momworld.md]] initializes momentum from the same kind of feature difference but reports no prediction error with or without it. The known cost: stationary agents (parked cars, waiting pedestrians) give no residual and reach the planner only through the static branch, as ResWorld states. No paper has tested a motion-only design on a closed-loop or navhard split.
- **See**: [[concepts/world-model-for-ad.md]]
- **Updated**: 2026-10-02

### Which data-driven simulator trains the best policy: reprojection, reconstruction or a learned world model?
- **Status**: open
- **Known so far**: One paper compares two families under one supervision source. [[sources/learning-to-drive-from-a-world-model.md]] trains the same steering policy in a reprojective simulator and in a 500M world model: 24/24 against 24/24 on lane-centre tests, 20/20 against 19/20 on lane changes, 0.369 against 0.394 held-out error. The world-model policy is engaged more in the field (29.92% against 27.63% of time), but the two cohorts are not matched. The test is lateral only, while reprojection's stated weakness is longitudinal range (under 4 m). Reconstruction-based training ([[sources/senna2.md]], 3DGS) and latent world-model RL ([[sources/dreameraD.md]]) are each evaluated on different benchmarks with different supervision. [[sources/simscale.md]] adds a reconstruction-based training result on NAVSIM (3DGS with IDM agents; +5 to +9 navhard EPDMS) and compares supervision sources inside it, but no second family. It also differs from openpilot on a second axis: its off-log states come from a fixed perturbation sampler, not from the policy. Reconstruction now has an online-RL result: [[sources/rad.md]] trains PPO in 3,968 3DGS clips and cuts collisions 0.229 → 0.089 against its own imitation-only post-training, on a private benchmark in the same renderer, with log-replayed agents.
- **See**: [[concepts/data-driven-simulators.md]], [[concepts/world-model-for-ad.md]], [[sources/simscale.md]], [[sources/rad.md]]
- **Updated**: 2026-10-06

### Does a clean VLM stream suppress an iteratively refined stream in shared attention?
- **Status**: open
- **Known so far**: [[sources/brainwam.md]] measures it for VLM plus video denoising (Tri-MoT 87.8 below WAM-only 88.1); SimWAM's two-modality ablation shows no such effect without a VLM. Untested for occupancy diffusion, flow-matched latents or JEPA predictors, and untested whether one-way masking (UniDriveVLA, AutoMoT) avoids it.
- **See**: [[concepts/world-model-for-ad.md]], [[concepts/mixture-of-experts.md]]
- **Updated**: 2026-09-30

---

## 3. Perception and representation

### Is explicit perception supervision necessary for planning?
- **Status**: open
- **Known so far**: Both routes work. [[sources/auto-jepa.md]] reaches 91.3 PDMS with a frozen encoder and no perception labels; [[sources/wcog-vla.md]] reaches 92.9 with 3D boxes and per-agent futures. Removing WCog-VLA's 3D perception costs 3.3 PDMS (89.3 → 86.0), yet its 3D pretraining is worth only +1.1 while the planner still emits text, which suggests explicit structure needs a continuous action head to be used. [[sources/foresight.md]] loses 1.1 PDMS (89.3 → 88.2) when its whole perception branch is deleted. No paper compares the supervised and unsupervised routes at matched capacity and data. [[sources/resworld.md]] runs both routes in one NAVSIM codebase: 87.3 PDMS without perception labels, 88.3 with detection and map supervision. The supervised variant also feeds its world model detection agent queries instead of temporal residuals, so the +1.0 is not a clean price. In a model-based RL setting it was: [[sources/raw2drive.md]]'s camera world model scores 17.4 DS on its 10-clip set without a BEV-mask decoder head and 83.5 with it, and aligning its encoder to ground-truth BEV features cell by cell is required (0.0 without any alignment).
- **See**: [[concepts/perception-for-planning.md]], [[sources/raw2drive.md]]
- **Updated**: 2026-10-06

### Does agent selectivity predict driving quality?
- **Status**: open
- **Known so far**: [[sources/auto-jepa.md]] measures 2.97× occlusion selectivity in latent space (mean 1−cos 0.080 vs 0.027) and shows behavioural consequences on three hand-picked scenes. The occlusion protocol has been run on one model. No paper correlates an occlusion-sensitivity statistic with PDMS, collision rate or interaction-scenario performance, and Auto-JEPA has no ablation against plain waypoint regression through the same trajectory encoder.
- **See**: [[concepts/perception-for-planning.md]], [[concepts/world-model-for-ad.md]]
- **Updated**: 2026-09-30

### Do teacher pseudo-labels cap what the student can learn?
- **Status**: partially answered
- **Known so far**: Swapping a teacher's depth for LiDAR depth under one planner has not been run. [[sources/physwam.md]] does half of it: generating depth alone adds nothing to planning (87.2 vs 87.5 PDMS without it), while a raw-LiDAR point loss that also involves the generated motion improves depth by 8–15% AbsRel and planning by +1.9 EPDMS. Whether teacher quality (for example Depth Anything V2 vs V3, SAM 2 vs SAM 3) propagates to planning is untested, and agent-generated reasoning corpora rarely carry a measured noise rate (Qwen-Drive reports 55.9% for its own). For a planning teacher: [[sources/simscale.md]]'s PDM-Closed teacher scores 51.3 on navhard and its students reach 48.0 (single) and 50.9 (three-model ensemble), so the cap is not reached. Benchmark sub-score labels, which no trajectory teacher limits, do slightly better than the teacher's trajectories (+0.3 / +0.8). The teacher's style is inherited: its low extended comfort (EC 36.0 / 25.4) shows up in the students' Stage-2 EC (LTF 70.6 → 42.9, DiffusionDrive 72.8 → 31.9). For a privileged driving teacher: Bench2Drive's base set was generated by [[sources/think2drive.md]], which scores 91.85 DS there; the best Think2Drive-protocol student in the wiki scores 86.23 ([[sources/deepsight.md]]). On the teacher's own benchmark a camera student (TCP, 200K frames) keeps 36.4 of 83.8 DS. See the new question on whether the expert bounds Bench2Drive. A student coupled to the same teacher at every level of training (latent targets, dynamics, rewards, policy initialization and distillation) does worse than imitation students: [[sources/raw2drive.md]] scores 71.36 against the teacher's 91.85.
- **See**: [[concepts/teacher-pseudo-labels.md]], [[concepts/world-model-for-ad.md]], [[sources/simscale.md]], [[sources/think2drive.md]], [[concepts/bench2drive.md]], [[sources/raw2drive.md]]
- **Updated**: 2026-10-06

### Is it video pretraining or the JEPA objective that transfers to planning?
- **Status**: open
- **Known so far**: V-JEPA 2 beats image-pretrained encoders in every sweep that includes it ([[sources/redrive.md]]: +5.5 PDMS over DINOv2, and fine-tuning DINOv2 under a predictive loss does not close the gap, 84.4 vs 89.9). But V-JEPA 2 is the only video-pretrained entry in those tables, and no paper includes a video-pretrained non-JEPA control. Video-generation priors (SimWAM, DriveLaW) are also effective without a JEPA objective. [[sources/ad-e2e-jepa.md]] inherits the opposite default (DINOv3 over V-JEPA 2) from a robotics ablation the wiki can cite only second-hand. An image-pretrained encoder under a JEPA-style predictive loss reaches 90.4 PDMS in [[sources/foredrive.md]] (DINOv3 ViT-L, one camera), not far below [[sources/redrive.md]]'s V-JEPA 2 at 91.0. Different planners, so it is not the missing control, but the image-encoder penalty is smaller there than in ReDrive's own DINOv2 run (84.4).
- **See**: [[concepts/foundation-backbones-for-ad.md]], [[sources/foredrive.md]]
- **Updated**: 2026-10-06

### When should a foundation backbone be frozen, LoRA-adapted or fully fine-tuned?
- **Status**: open
- **Known so far**: Off-the-shelf use is priced three times: a BEV head on frozen Qwen-initialized features trails a dedicated detector by 6.34 mAP and unfreezing recovers +10.46; a frozen video DiT costs 6.42 PDMS; LoRA collapses under DINOv2 geometric distillation. [[sources/redrive.md]] prices unfreezing its encoder at +3.5 PDMS. The page's rule of thumb (LoRA when the pretrained objective already matches the downstream one, full fine-tuning otherwise) is fitted to two papers, neither of which tests the other's setting.
- **See**: [[concepts/foundation-backbones-for-ad.md]]
- **Updated**: 2026-09-30

### Does discrete visual tokenization survive at the frontier?
- **Status**: open
- **Known so far**: The best token-based planner scores 91.8 PDMS against 94.0 for continuous-latent methods (93.7 when this entry was written); whether the gap is quantization loss or tokenizer immaturity is untested. [[sources/unified-driving-tokens.md]] argues one tokenizer can serve planning and world modeling and then uses different tokenizers for each; its geometry-enhanced variant is worse on every reconstruction metric. Its tokenizer needs a DINOv3-B forward pass and reports no latency.
- **See**: [[concepts/visual-tokenization.md]]
- **Updated**: 2026-10-06

---

## 4. Reasoning and the VLM

### Does grounded reasoning help on a public benchmark, and is the trace faithful?
- **Status**: open
- **Known so far**: [[sources/grava.md]] gives the most graded ablation: action-only 74.8 → coarse reasoning 77.9 → box-grounded objects 85.5 → full grounded reasoning 90.1, all RL-trained, on an internal long-tail benchmark. Without RL, full reasoning scores 73.1. Editing the trace edits the action (0.44 → 0.82 normalized PDMS), but the trace's final maneuver is copied from the expert trajectory at annotation time. Unknown: whether the ladder holds on a public benchmark, whether any trace stays predictive of the action on navhard, and whether a verbalized action is enough for auditing.
- **See**: [[concepts/chain-of-thought-for-ad.md]], [[concepts/reasoning-faithfulness.md]]
- **Updated**: 2026-09-30

### What does driving adaptation cost the VLM, and is retained general capability worth anything for driving?
- **Status**: open
- **Known so far**: Under one 15-benchmark protocol, Qwen-Drive scores 66.41 against its base model's 67.40 while Alpamayo-1.5 scores 14.65, MiMo-Embodied 26.53 and UniDriveVLA 48.73 ([[sources/qwen-drive-1.0.md]]). The distinguishing variable looks like the data mixture (26% general-purpose); nothing between 0% and 26% has been run. Qwen-Drive freezes the VLM during RL, so retention after GRPO through the VLM is unmeasured for every method that does it. The only evidence on usefulness points the other way: the whole knowledge-adaptation stage is worth +0.08 RFS for planning.
- **See**: [[concepts/general-capability-retention.md]]
- **Updated**: 2026-09-30

### When should slow reasoning run, and does it need to emit a decision?
- **Status**: open
- **Known so far**: [[sources/drive-hwm.md]] runs its slow branch every 8 steps unconditionally (25.6 ms) and has it emit a forecast, not a plan; CLEAR, AdaThinkDrive and AutoVLA route slow compute by difficulty. The two policies have never been compared at matched average cost, and no paper compares a forecast-only slow branch against a meta-action one under a fixed fast policy. Drive-HWM's rate hierarchy is worth +0.3 to +0.8 PDMS on NAVSIM, where the slow model runs once per scenario, so the design's purpose is untested in a closed loop. Adaptive CoT's robustness when its complexity classifier fails on OOD scenes is also unmeasured.
- **See**: [[concepts/dual-system-vla.md]], [[concepts/adaptive-routing.md]], [[concepts/chain-of-thought-for-ad.md]]
- **Updated**: 2026-09-30

### Advisory VLM or VLM in the action path?
- **Status**: open
- **Known so far**: [[sources/drivewam.md]]'s frozen Qwen3-VL-8B only emits text guidance and helps at every data scale, but the paper reports no VLM–action agreement rate, and Senna-2's evidence is that unaligned guidance under-delivers. No paper compares the advisory arrangement against fine-tuning the VLM into the action path at matched backbone and data, or distils the VLM into the fast policy after alignment.
- **See**: [[concepts/dual-system-vla.md]], [[concepts/world-model-for-ad.md]]
- **Updated**: 2026-09-30

---

## 5. RL, selection and diversity

### Do learned scorers overfit the benchmark's own metric?
- **Status**: open
- **Known so far**: Almost every non-BoN NAVSIM-v1 entry above 92.0 PDMS trains a scorer or an RL reward against the benchmark metric. [[sources/mm-future.md]] shows what that looks like on v2: ego progress 92.2 (human agent 87.4) with driving direction, lane keeping and history comfort the lowest in its table, because its scorer learned the v1 components. Whether PDMS-supervised routing and scoring stay reliable in interactive, reactive settings is untested. [[sources/simscale.md]] extends the pattern to training data: its simulated states, its expert trajectories and its reward labels are all filtered or computed with EPDMS, and it is evaluated only on NAVSIM. [[sources/world4scorer.md]] goes one step further on v2: it runs the metric's own extended-comfort check at inference against the previous plan (EC 77.0 → 92.8, above the human agent's 90.1; +1.6 EPDMS, and +1.61 when applied to DrivoR). Its scorer predicts only v1 components, and how its separate v2 combination weights were chosen is not stated. It reaches the world-model VLAs too: DriveVLA-W0's 90.2★ and [[sources/hyworldvla.md]]'s 90.59 both come from a Hydra-MDP-style vocabulary head whose targets combine the expert trajectory with the evaluation metrics. On the same backbones, flow-matching heads without metric targets score 87.2 and 88.76, gaps (+3.0, +1.83) larger than either paper's world-model effect.
- **See**: [[concepts/selection-based-planning.md]], [[concepts/adaptive-routing.md]], [[sources/simscale.md]], [[sources/world4scorer.md]], [[sources/hyworldvla.md]]
- **Updated**: 2026-10-12

### Does RL buy out-of-distribution robustness?
- **Status**: open
- **Known so far**: Within [[sources/geowam.md]]'s navhard table, which contains no candidate-scoring method, three of the four methods above 31 use RL or PDMS-score supervision, but GeoWAM (36.6) tops them without it. Margins are 0.5–2.5 points on single runs, so the wiki cannot say. Whether GRPO-trained NAVSIM methods (FLARE, DriveFine) are competitive on Bench2Drive's interactive scenarios is also unreported. Data without RL buys a good deal of it: [[sources/simscale.md]]'s off-log simulation data, with no RL, adds +5.1 to +8.6 navhard EPDMS across three planner families, more than any margin in GeoWAM's table. Whether RL on the same simulated states would add to that is untested. One closed-loop data point outside NAVSIM: RL in 3DGS cuts collisions on held-out reconstructed clips from 0.229 to 0.089 against the same architecture's imitation-only post-training ([[sources/rad.md]]), but in the training renderer, against non-reactive agents, on a private benchmark whose second instance halves the effect.
- **See**: [[concepts/navhard-ood-evaluation.md]], [[concepts/bench2drive.md]], [[sources/simscale.md]], [[sources/rad.md]]
- **Updated**: 2026-10-06

### Does reinforcement learning from camera input beat imitation at matched data and backbone?
- **Status**: open
- **Known so far**: The only camera RL result on Bench2Drive, [[sources/raw2drive.md]], scores 71.36 DS: +7.1 over the best imitation baseline in its own table (DriveAdapter 64.22) and about 20 below the best imitation methods in the wiki (LinkVLA 91.01, DynVLA 88.34). Its RL stage starts from a privileged policy and is worth +25 DS on a 10-clip subset over not fine-tuning. No paper trains an imitation student and an RL student with the same encoder, data and privileged supervision. [[sources/think2drive.md]]'s own camera student, trained by imitation on 200K frames, kept 36.4 of the teacher's 83.8 DS on a different benchmark.
- **See**: [[sources/raw2drive.md]], [[concepts/rl-for-ad.md]], [[concepts/bench2drive.md]]
- **Updated**: 2026-10-06

### Does an inference-time safety check add anything to a policy already trained on the rule?
- **Status**: partially answered
- **Known so far**: The nearest measurement says no. [[sources/drivereferee.md]] applies one geometric rule at inference: +0.30 EPDMS on the untrained base, and +0.06 PDMS / −0.04 EPDMS on a policy already distilled on it. Whether reflective inference stacks with a GRPO-trained base in other architectures (masked diffusion, discrete flow matching) is still open.
- **See**: [[concepts/inference-time-safety.md]]
- **Updated**: 2026-09-30

### How should diversity be rewarded and measured when the output is text plus a trajectory?
- **Status**: open
- **Known so far**: Best-of-6 on NAVSIM-v1 matches the human trajectory's score (Curious-VLA, 94.8 PDMS). Untested: MUPO-style multi-group advantages with groups clustered by trajectory geometry, separate diversity rewards for reasoning and action, and whether learned scorers benefit from diversity-trained candidates.
- **See**: [[concepts/divergent-thinking-in-vlms.md]], [[concepts/best-of-n.md]]
- **Updated**: 2026-09-30

### Is a reward needed for closed-loop training, or are future-anchored action labels enough?
- **Status**: open
- **Known so far**: [[sources/learning-to-drive-from-a-world-model.md]] trains a deployed policy on its own rollouts with no reward: a plan head conditioned on the recorded future labels every state the policy reaches. Lane keeping and lane changes emerge (24/24, 19–20/20 against 5/24, 8/20 off-policy). The policy can at best match its teacher, and the teacher is never evaluated on its own. Reward-based closed-loop training exists in other simulators ([[sources/senna2.md]]: penalties in 3DGS; [[sources/dreameraD.md]]: a learned latent reward), but no paper compares labels against reward in one simulator. [[sources/simscale.md]] compares expert-trajectory labels with per-trajectory benchmark sub-scores (a reward signal, used by regression, not policy gradient) for one scorer in one 3DGS simulator: sub-scores win by +0.3 / +0.8 navhard EPDMS, within single-run noise, and only when expert imitation is kept on the real data (dropping it costs −0.7 / −3.1). The opposite end of the comparison is now ingested: [[sources/think2drive.md]] uses a hand-shaped reward and no labels in a reactive simulator (CARLA v2), and its policy then became the label source for Bench2Drive. Neither paper trains the other's way. In one 3DGS simulator, [[sources/rad.md]] compares off-policy imitation, RL and both (collision ratio 0.229 / 0.143 / 0.089 over 337 clips), and finds PPO without hand-designed directional auxiliary losses no better than imitation (0.249). Its labels are the logged expert's, not on-policy ones, so the openpilot comparison is still missing.
- **See**: [[concepts/rl-for-ad.md]], [[concepts/data-driven-simulators.md]], [[concepts/teacher-pseudo-labels.md]], [[sources/simscale.md]], [[sources/think2drive.md]], [[sources/rad.md]]
- **Updated**: 2026-10-06

### Do policies trained in rendered simulators learn the renderer's artifacts?
- **Status**: open
- **Known so far**: [[sources/learning-to-drive-from-a-world-model.md]] states that reprojection artifacts correlate with the ego's offset from the log, which is what the corrective label encodes, and that the policy exploits them ("cheating or shortcut learning"). Its remedy is a roughly 700-bit Gaussian-noise bottleneck on the feature extractor. Neither the exploitation nor the remedy is measured, and whether the world-model simulator has the same problem is not discussed. No other training paper in the wiki (3DGS-based or world-model-based) reports a test for it. An evaluation-side version now exists. [[sources/simscale.md]] trains on 3DGS renders of off-log poses and gains most on navhard's 3DGS-rendered Stage 2 (+5.6 to +7.6 per-stage EPDM, against +0.8 to +5.3 in Stage 1 and +0.8 to +2.9 on real-image navtest). Its training engine (MTGS) and the navhard benchmark paper share two author names with it; whether Stage 2 uses the same renderer is not stated. No probe and no zero-perturbation rendering control is reported. [[sources/rad.md]] compares a policy trained on 3DGS renders against imitation baselines that never saw one, evaluated in the same renderer, and ends every training episode at a collision or deviation because the renders degrade afterwards. Neither confound is tested.
- **See**: [[concepts/data-driven-simulators.md]], [[concepts/perception-for-planning.md]], [[sources/simscale.md]], [[concepts/navhard-ood-evaluation.md]], [[sources/rad.md]]
- **Updated**: 2026-10-06

### Which RL recipe suits dense, safety-gated driving rewards?
- **Status**: open
- **Known so far**: Whether Dr. GRPO remains preferable when rewards are dense, continuous and safety-gated rather than binary is untested. It is also unknown at what data scale reasoning-free Dr. GRPO training (NoRD) catches up with CoT-supervised training on 212K+ samples (AutoVLA), and whether a single sequence-level objective can serve both expert types in a mixture-of-experts planner. One controlled data point outside NAVSIM: in CARLA, a model-based actor-critic trained in latent imagination beats PPO with identical engineering, 83.8 vs 57.5 DS on single-scenario routes and 98.6% vs 1% route completion on long routes ([[sources/think2drive.md]]). Its account of PPO's failure is that a mid-training policy reset, needed to escape conflicting scenario optima, is cheap only when a world model supplies dense rewards. The ablation behind it has no numbers in the clipping. From cameras, the same recipe works only when bootstrapped from a privileged system: [[sources/raw2drive.md]]'s RL fine-tuning adds +25 Dev10 DS over transplanting the privileged policy (58.4 → 83.5), and the system scores 0 without its latent alignment. In 3DGS with camera input, [[sources/rad.md]]'s PPO needs three additions to help: per-axis advantages and clipping over a factorized lateral / longitudinal action, rule-directed auxiliary losses weighted by the advantage (without them 0.249 CR against 0.229 for imitation), and imitation steps interleaved 4:1 (without them the policy drifts, 0.345 m from the expert).
- **See**: [[concepts/r1-zero-like-training.md]], [[concepts/chain-of-thought-for-ad.md]], [[concepts/mixture-of-experts.md]], [[sources/think2drive.md]], [[sources/raw2drive.md]], [[sources/rad.md]]
- **Updated**: 2026-10-06

### Does simulated off-log data keep paying as it grows, and which planners can use it?
- **Status**: open
- **Known so far**: [[sources/simscale.md]] adds up to 237K simulated scenes to about 100K real ones (2.3:1). Under PDM-Closed labels a regression planner (LTF) flattens near 1:1 while DiffusionDrive and a GTRS-Dense scorer keep rising; under retrieval labels that steer back to the log, every planner peaks early. The gain from simulation does not shrink as real data grows from 10K to 100K (+5.6 to +7.5 EPDMS for ResNet-34, read from the figure). The fits use five or six single-run points per curve, adjacent points reverse by up to 2.7 EPDMS, and no extrapolation is tested. Whether an on-policy sampler in the same renderer would scale differently is unknown.
- **See**: [[sources/simscale.md]], [[concepts/diffusion-planner.md]], [[concepts/data-driven-simulators.md]]
- **Updated**: 2026-10-06

### Should a pseudo-expert be conservative or exploratory?
- **Status**: open
- **Known so far**: In one simulator, [[sources/simscale.md]]'s conservative teacher (retrieving the human trajectory that returns to the log) helps at small data sizes and then stops helping; the exploratory one (PDM-Closed on ground-truth state) keeps paying but passes on its poor comfort. Averaging scorers trained on both, plus a reward-only one, adds +0.8 / +2.9 navhard EPDMS. openpilot's teacher ([[sources/learning-to-drive-from-a-world-model.md]]) is a third kind, a learned plan head that sees the future, and has never been compared with either.
- **See**: [[sources/simscale.md]], [[concepts/teacher-pseudo-labels.md]], [[concepts/data-driven-simulators.md]]
- **Updated**: 2026-10-06

---

## 6. Efficiency

### What does each method actually cost at inference?
- **Status**: open
- **Known so far**: Recorded latencies span 22 ms to 9.4 s on different hardware, and only the Wan2.2-5B family is close to comparable (Metis and SUV both use an RTX 4090). Missing figures matter most where the cost is likely highest: GRAVA's grounded reasoning (8B, 4,096-token limit) reports none, SUV reports only its with-access cost, and [[sources/physwam.md]] pays 9.4 GPU-seconds per plan with no action-only path tested. Per-candidate futures are cheap once the state is a few hundred tokens (about 3.2 ms per hypothesis in [[sources/mm-future.md]]). [[sources/raw2drive.md]] separates its costs: 600 ms for the BEVFormer camera encoder, 2 ms each for world model and policy (hardware unstated). [[sources/foredrive.md]]: 58.2 ms on an H20 for 90.0 EPDMS, of which latent foresight is 26 ms; the fastest entry at or above 90 EPDMS. [[sources/world4scorer.md]]: 106.5 ms fp32 on an A100 (23.0 ms fp16) for 94.0 PDMS with 35.5M parameters, plus 125 ms of CPU re-ranking in its v2 configuration; the scorer itself takes 1.4 ms. [[sources/hyworldvla.md]] (Emu3-8B VLA plus action expert) reports no latency; its base DriveVLA-W0 reports 74 ms for the same head. [[sources/idol.md]]: about 57 ms (17.65 FPS) on an RTX 3090 for 256 per-anchor latent BEV rollouts on 64 tokens, two passes, 69.4M parameters.
- **See**: [[concepts/inference-latency.md]], [[concepts/wam-attention-masks.md]], [[sources/raw2drive.md]], [[sources/foredrive.md]], [[sources/world4scorer.md]], [[sources/hyworldvla.md]], [[sources/idol.md]]
- **Updated**: 2026-10-12

---

## Answered

Questions closed by an ingested paper, each with the paper and its result.

- **Is the comfort deficit inherent to sampled planners?** No. [[sources/physwam.md]] samples video, depth and motion from noise and scores 96.3 closed-loop comfort and 90.5 navtest EC. What it does differently from WA-JEPA, which scores far lower on closed-loop comfort, is not isolated. See [[concepts/hugsim-benchmark.md]]. (2026-09-30)
- **Does performance on curated rare-event clips predict closed-loop behaviour?** No. The AlpaSim reproduction in [[sources/qwen-drive-1.0.md]] inverts the ordering. The remaining open part is tracked under "Do NAVSIM rankings survive a reactive closed loop?". See [[concepts/physicalai-av-benchmark.md]]. (2026-09-30)

---
title: "DriveReferee: Geometric Safety Verdicts Need Not Be Learned for Driving World-Action Models"
type: source-summary
sources: ["raw/papers/DriveReferee_ Geometric Safety Verdicts Need Not Be Learned for Driving World-Action Models.md"]
related: [concepts/inference-time-safety.md, concepts/selection-based-planning.md, concepts/best-of-n.md, concepts/rl-for-ad.md, concepts/discriminative-policy-optimization.md, concepts/evaluation-variance.md, concepts/navsim-benchmark.md, concepts/world-model-for-ad.md, concepts/perception-for-planning.md, concepts/inference-latency.md, concepts/foundation-backbones-for-ad.md, sources/physwam.md, sources/drivefuture.md, sources/feaxdrive.md, sources/reflectdrive.md, sources/hydra-mdp-pp.md, sources/plan-r1.md, sources/adaptive-wam.md, sources/reworld.md, sources/metis.md, sources/suv.md, sources/wa-jepa.md, sources/da-wam.md]
created: 2026-09-30
updated: 2026-09-30
confidence: high
---

# DriveReferee

**Paper**: DriveReferee: Geometric Safety Verdicts Need Not Be Learned for Driving World-Action Models
**Authors**: Fengcheng Yu, Dhruv Parikh, Junjie Ye, Maulik Bhatt, Thang Vu, Igor Vasiljevic, Vitor Guizilini, Yue Wang
**Orgs**: University of Southern California, Woven by Toyota, Toyota Research Institute
**arXiv**: 2609.22762v1
**Code**: not stated
**Source**: `raw/papers/DriveReferee_ Geometric Safety Verdicts Need Not Be Learned for Driving World-Action Models.md`

---

## What It Is

A paper about **how to check a generated trajectory** in a video world-action model (WAM), from the group behind [[sources/physwam.md]] (all eight authors are PhysWAM authors) and on the same Cosmos 3 backbone.

Its argument: the benchmark's collision and drivable-area rules are known geometry. What is missing at deployment is the map those rules run on. So learn the map, and **compute** the verdict.

Three parts:

1. **A WAM policy.** Cosmos3-Nano (16B; frozen reasoner tower, 8B trainable generation tower), one front camera, five frames in, eight future video frames and eight action steps out by flow matching.
2. **An analytic referee** $R(\tau,M)$ with no learned parameters. It re-simulates a trajectory with the evaluator's LQR tracker and bicycle model, then checks five footprint points against a drivable-area raster and a time-indexed vehicle-occupancy raster.
3. **A BEV readout** (21.23M parameters, Lift-Splat style, on frozen WAM features) that predicts those two rasters from the camera.

The referee is used twice:

| Placement | Map | What it does |
|---|---|---|
| Training | Ground truth $M_{gt}$ | Picks a winner and a loser among the policy's own samples; a pairwise preference loss distills them into the policy |
| Deployment | Predicted $\hat M$ | Checks the default plan; on an alarm, draws one more sample and may replace the plan |

**Headline, full navtest: 92.02 PDMS / 91.56 EPDMS** with one camera and no external data. The imitation-only base is 91.08 / 90.69.

---

## Key Takeaways

- **Nearly all of the gain is the training-time distillation.** It is worth **+0.92 EPDMS** (95% CI +0.71 to +1.13) and uses **462 preference pairs and 850 training steps**. The distilled policy alone scores 91.96 PDMS / 91.61 EPDMS in a single pass.
- **Test-time selection adds nothing once the policy is distilled.** +0.06 PDMS (CI −0.01 to +0.14) and −0.04 EPDMS. With ground-truth maps it would add +0.08. The "full" headline system costs 1.34× the policy samples for no measurable benefit over the distilled policy.
- **On the undistilled base, selection is worth +0.30 EPDMS**, which is a net 44 scenes out of 12,146 (53 hard-gate failures fixed, 9 introduced).
- **"Need not be learned" is a non-inferiority result.** A learned verifier given the same predicted map differs by −0.01 EPDMS with a CI of ±0.15. The two best learned verifiers on image features score +0.23 and +0.26 against the referee's +0.30; none of these differences is significant. All of it is measured at $K=2$ candidates.
- **The analytic referee is itself an approximation.** On ground-truth maps it recovers 63% of the official evaluator's drivable-area violations and 79% of its collisions (precision 0.995 and 0.931). On predicted maps its clearance margin is off by 0.50 m on average, more than the 0.4 m alarm threshold.
- **Over half of the distillation gain does not need the preference term.** Fine-tuning on the referee-selected winners alone gives +0.56 of the +0.92.
- **It is the first ingested paper to report paired, per-scene bootstrap confidence intervals on navtest.** They put the 95% half-width of a difference between two checkpoints at about 0.2 EPDMS, and of a selection variant on fixed candidates at about 0.1.
- **The supervision is privileged.** The preference labels come from the benchmark's own collision and drivable-area logic on annotated maps, and the readout is trained on those maps.

---

## Method

![[fig2_sep03_1527.png|DriveReferee overview. Front-camera frames, ego state and a navigation command enter a policy with a frozen reasoner and a trained generator, which outputs future video and candidate plans. A readout g_phi predicts a BEV map from frozen visual features and ego poses; ground-truth maps from privileged annotations are used in training only. The analytic referee (kinematic rollout, footprint checks, metric margins) sends verdicts to a training block that forms preference pairs for a preference loss, and to a deployment block that raises an alarm and, if so, ranks K−1 more samples]]

*Figure 2: Overview. The referee has zero learned parameters and two placements: on $M_{gt}$ for preference distillation (green) and on $\hat M$ for alarm-gated selection (blue).*

### Setup

- Observation $o=(I_{-H:0},c,s_0)$: five front-camera frames, a discrete navigation command, current speed and heading.
- Plan $\tau=\{(x_j,y_j,\psi_j)\}_{j=1}^{T_f}$ in the current ego frame, $T_f=8$ steps at 2 Hz.
- Scene state $M=(M_{\mathrm{drv}},M_{\mathrm{veh}})$: a static drivable-area raster and $1+T_f$ vehicle-occupancy rasters, 0.4 m cells.

### The analytic referee

1. **Kinematic rollout.** The waypoints are interpolated and tracked with the evaluator's LQR controller and kinematic bicycle model from speed $v_0$, giving simulated poses $p_1,\dots,p_{N_s}$.
2. **Geometric checks.** At each pose, five query points $Q(p_i)$ (four corners and the centre) are tested against $M_{\mathrm{drv}}$ and the nearest time slice of $M_{\mathrm{veh}}$.
3. **Metric margins.** Distance transforms $D_{\mathrm{drv}}$ and $D^{(i)}_{\mathrm{veh}}$ give clearance to the nearest unsafe cell (zero once a boundary is crossed).

$$n_{\mathrm{dac}}=\sum_{i=1}^{N_s}\mathbf 1\big[\exists q\in Q(p_i):M_{\mathrm{drv}}(q)=0\big],\qquad n_{\mathrm{nc}}=\sum_{i=1}^{N_s}\mathbf 1\big[\exists q\in Q(p_i):M^{(i)}_{\mathrm{veh}}(q)=1\big]$$

$$m_{\mathrm{dac}}=\min_{i,\,q\in Q(p_i)}D_{\mathrm{drv}}(q),\qquad m_{\mathrm{nc}}=\min_{i,\,q\in Q(p_i)}D^{(i)}_{\mathrm{veh}}(q)$$

Aggregates: $n(\tau)=n_{\mathrm{dac}}+n_{\mathrm{nc}}$ and $m(\tau)=\min(m_{\mathrm{dac}},m_{\mathrm{nc}})$. The referee does not score progress, comfort, time-to-collision, lane direction or traffic lights.

### Training placement: preference distillation on ground-truth maps

For each training scene the policy draws $K_t=5$ samples. $\mathcal P$ is the set that passes both checks on $M_{gt}$, $\mathcal V$ the set that violates at least one. Scenes with both non-empty are kept.

$$\tau^{w}=\arg\max_{\tau\in\mathcal P}m(\tau),\qquad \tau^{l}=\arg\min_{\tau\in\mathcal V}d(\tau,\tau^{\ast})$$

The winner is the passing sample with the largest clearance, chosen without the expert. The loser is the violating sample closest to the expert $\tau^\ast$, a hard negative.

$$\mathcal L=w_v\,\mathcal L^{\mathrm{vid}}_{\mathrm{FM}}+w_a\,\mathcal L^{\mathrm{act}}_{\mathrm{FM}}(\tau^{w})-\log\sigma\Big(\beta\big[\ell_\theta(\tau^{l})-\ell_\theta(\tau^{w})\big]\Big)$$

- $\ell_\theta(\tau)$ is the per-sample action flow-matching loss, evaluated for both members at the same timestep. The loss difference stands in for a log-likelihood ratio, as in diffusion preference optimization.
- The first two terms keep the video loss on the recorded future and the action loss on the winner. There is no frozen reference policy.
- Settings: 462 pairs, 850 steps, learning rate $2\times10^{-5}$, $\beta=10$, $w_v=w_a=10$.

### Deployment placement: gated selection on predicted maps

The readout $g_\phi$ predicts per-pixel depth distributions, lifts frozen multi-layer features into BEV, aligns history with the known ego poses, adds visibility channels, and decodes $\hat M_{\mathrm{drv}}$ and $\hat M_{\mathrm{veh}}$. Only $\phi$ is trained, on rasterized privileged annotations.

One sample $\bar\tau$ is the default plan. An alarm is raised when

$$n(\bar\tau)>0\qquad\text{or}\qquad m(\bar\tau)\le\Delta,\qquad \Delta=0.4\ \text{m}$$

On an alarm the policy draws $K-1$ more samples ($K=2$). Candidates are ranked by fewer violating poses, then larger clearance, then distance to the default. A candidate replaces the default only if it removes at least $\delta_n=1$ violating pose or, at equal count, gains at least $\Delta$ of clearance. Margins are compared on the same predicted map, so a shared local map bias partly cancels.

---

## Figures

![[fig1_a_sep01.png|Panel (a) of Figure 1: a ground-truth map M_gt, marked as missing at deployment, and a candidate trajectory enter a known geometric rule, which outputs a safety verdict]]

*Figure 1, panel (a): privileged evaluation. The rule is known; $M_{gt}$ is unavailable at deployment. Only this panel was saved with the clipping.*

Figure 2 is in the Method section.

![[fig_forest_v3.png|Forest plot of paired delta EPDMS against the base for nine verdict sources with 95 percent confidence intervals and the number of replaced plans: three map-blind controls, five learned verifiers and the analytic referee]]

*Figure 3: Matched-budget verdict-source comparison on navtest ($n=12{,}146$). Same pre-sampled $K=2$ candidates, all sources calibrated to about 1.35× sampling budget.*

Values printed in Figure 3:

| Verdict source | Group | Δ EPDMS | CI excludes 0? | Replaced plans |
|---|---|---:|:-:|---:|
| Random | Map-blind control | +0.02 | No | 2,093 |
| Smoother | Map-blind control | −0.01 | No | 1,966 |
| Conservative | Map-blind control | −0.02 | No | 1,514 |
| Score gating (Hydra-MDP) | Learned verifier | +0.23 | Yes | 1,246 |
| Argmax selection (SparseDriveV2) | Learned verifier | −0.04 | No | 2,161 |
| Confidence gating (DriveVer) | Learned verifier | +0.26 | Yes | 1,540 |
| Per-metric distillation (Hydra-MDP) | Learned verifier | +0.06 | No | 2,174 |
| Pairwise ranking (Bradley–Terry) | Learned verifier | 0.00 | No | 2,017 |
| **Analytic referee** | Analytic | **+0.30** | Yes | 1,339 |

![[fig_scale.png|Delta EPDMS of the same-map learned verdict against the fraction of verdict-training data (5, 10, 25, 50, 100 percent), with confidence intervals, against a dashed line at +0.35 for the analytic referee with zero labels]]

*Figure 4: Verdict-supervision scaling in the same-map setting, at a fixed 2.0× sampling budget. 100% is 66,385 training candidates. Points are the mean of two log-level draws.*

| Verdict-training data | 5% | 10% | 25% | 50% | 100% | Analytic referee (0 labels) |
|---|---:|---:|---:|---:|---:|---:|
| Δ EPDMS | +0.00 | +0.07 | +0.17 | +0.33 | +0.27 | +0.35 |

![[fig4_qual_v3.png|Deployment-time verification on predicted maps. Left: one intervention with a context view, a zoom on the flagged state where the default plan crosses the predicted drivable boundary by 0.40 m while the selected plan keeps 1.26 m, a signed-margin curve over time, and ground-truth against generated future frames with the boundary projected. Right: four more rescued scenes in bird's-eye view]]

*Figure 5: (a) One intervention on the base policy with $K=2$. The default plan crosses the predicted boundary by 0.40 m at 3.8 s; the selected plan keeps 1.26 m. (b) Four further replacements found by automatic screening, where the default leaves the ground-truth drivable area and the selected plan stays inside both boundaries.*

---

## Tables

### Table I: World-model-based planners on navtest (12,146 scenes)

| Method | Input | Video | NC | DAC | TTC | Comf. | EP | PDMS | EPDMS |
|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|
| DrivingGPT | 1×C | ✓ | 98.9 | 90.7 | 94.9 | 95.6 | 79.7 | 82.4 | — |
| WoTE | 3×C+L | ✗ | 98.5 | 96.8 | 94.9 | 99.9 | 81.9 | 88.3 | — |
| DriveVLA-W0 | 1×C | ✓ † | 98.7 | 99.1 | 95.3 | 99.3 | 83.3 | 90.2 | 86.1 |
| PWM | 1×C | ✓ | 98.6 | 95.9 | 95.4 | 100.0 | 81.8 | 88.1 | — |
| DriveLaW ‡ | 1×C | ✓ † | 99.0 | 97.1 | 96.7 | 100.0 | 81.3 | 89.1 | — |
| CoPhy § | C+L | ✗ | 99.0 | 98.2 | 96.8 | 100.0 | 85.3 | 91.4 | 86.1 |
| DriveDreamer-Policy | 3×C | ✓ | 98.4 | 97.1 | 95.1 | 100.0 | 83.5 | 89.2 | 88.7 |
| Metis | 1×C | ✓ | 98.3 | 97.1 | 94.7 | 100.0 | 83.4 | 89.1 | 89.5 |
| DriveVA ‡ | 1×C | ✓ | 99.2 | 97.5 | 98.7 | 100.0 | 83.5 | 90.9 | — |
| UNIVERSE | 1×C | ✓ † | 99.1 | 97.6 | 98.5 | 100.0 | 83.6 | 91.0 | — |
| Ours (base, 4B backbone) | 1×C | ✓ | 99.22 | 97.37 | 96.86 | 100.00 | 84.09 | 90.47 | 90.03 |
| Ours (full, 4B backbone) | 1×C | ✓ | 99.21 | 98.25 | 96.86 | 100.00 | 85.36 | 91.50 | 91.08 |
| Ours (base, 16B backbone) | 1×C | ✓ | 99.31 | 97.79 | 97.41 | 100.00 | 84.45 | 91.08 | 90.69 |
| **Ours (full, 16B backbone)** | 1×C | ✓ | **99.49** | **98.64** | 97.91 | 100.00 | 84.99 | **92.02** | **91.56** |

† trained with video generation, none generated at test time. ‡ DriveVA uses extra CARLA data; DriveLaW uses nuPlan and nuScenes. § CoPhy uses external VQA data and multi-candidate inference. Metis is its single-pass result. "Full" is referee distillation plus $K=2$ gated selection. The 4B and 16B models are Cosmos3-Edge and Cosmos3-Nano with 2B and 8B trainable generation towers.

### Table II: Training design ablations (paired Δ EPDMS over the imitation-only base)

| Variant | Δ EPDMS |
|---|---:|
| *Preference objective* | |
| $\beta=0$ | +0.56 |
| Full preference objective | **+0.92** |
| *Winner construction* | |
| Nearest to expert | +0.63 |
| Largest safety margin (ours) | **+0.92** |

### Table III: Video generation quality

The two "ours" rows share navtest scenes, protocol and seeds. Prior-work values are for context only.

| Method | FVD ↓ | LPIPS ↓ | PSNR ↑ | Frames | Resolution |
|---|---:|---:|---:|---|---|
| Base (imitation only) | 23.90 | 0.4166 | 18.75 | 9 @ 2 Hz | 832×480 |
| + referee distillation | 24.25 | 0.4126 | 18.92 | 9 @ 2 Hz | 832×480 |
| DriveDreamer-Policy | 53.59 | — | — | 9 | 144×256 |
| PWM | 85.95 | — | — | 10 | 128×224 |
| ForgeDrive | 69.2 | — | — | 8 @ 2 Hz | 512×1024 |
| DrivingGPT | 142.6 | — | — | 12 | 288×512 |

### Results given only in the text

| Quantity | Value |
|---|---|
| **Referee against the official evaluator, ground-truth maps** | 1,060 navtest scenes where the base scores zero, 5 samples each (5,300 candidates) |
| Gate-decision agreement | 91.0% (always-safe predictor: 73.5%) |
| Drivable-area gate | precision 0.995, recall 0.631 |
| Collision gate | precision 0.931, recall 0.792 |
| **Predicted against ground-truth map, default plan, all scenes** | $m_{\mathrm{dac}}$ correlation 0.895, mean absolute difference 0.502 m |
| **Placements (paired, 95% CI)** | |
| Distillation over base | +0.92 EPDMS [+0.71, +1.13] |
| Gated selection on the undistilled base | +0.30 EPDMS [+0.18, +0.42]; fixes 53 hard-gate failures, introduces 9 |
| Gated selection on the distilled policy | +0.06 PDMS [−0.01, +0.14]; −0.04 EPDMS [−0.12, +0.04] |
| The same with the ground-truth map | +0.08 EPDMS [+0.03, +0.13]; +0.10 PDMS |
| Winner by margin against winner nearest the expert | +0.29 EPDMS [+0.14, +0.45] |
| **Same-map learned verifier** (4.15M parameters, 66,385 labelled candidates) | |
| Against the analytic referee | −0.01 EPDMS [−0.15, +0.14] |
| Same plan selected | 83.4% of scenes |
| Policy samples per scene | 1.57 (analytic: about 1.35); 1.4× as many replacements |
| Same-map Bradley–Terry variant, against the base | −0.10 EPDMS [−0.22, +0.02] |
| **System** | |
| Distilled policy, single pass | 91.96 PDMS / 91.61 EPDMS |
| 4B pipeline gain | +1.03 PDMS (16B: +0.94) |
| Referee runtime | median 30 ms per candidate, CPU |
| Alarm rate / mean policy samples | 34.3% of scenes / 1.34× |

---

## Reading the Results

### 1. Where the 0.9 points come from {#decomposition}

| System | Policy samples | PDMS | EPDMS |
|---|---:|---:|---:|
| Base (imitation only) | 1 | 91.08 | 90.69 |
| Base + gated selection | 1.35 | – | 90.99 (+0.30) |
| Distilled | 1 | 91.96 | **91.61** |
| Distilled + gated selection ("full") | 1.34 | **92.02** | 91.56 |
| Distilled + gated selection on ground-truth maps | 1.34 | 92.06 | 91.69 |

(The last row and the 90.99 are the base or distilled score plus the paper's paired delta.)

- **The single-pass distilled policy is the result.** The headline row adds a second 16B video-and-action sample in a third of scenes and lands within noise of it, slightly higher on PDMS and slightly lower on EPDMS.
- **The paper says this itself**: "most of the benefit of deployment-time selection is already learned during preference distillation."
- **Even a perfect map leaves +0.08.** After distillation, a second sample rarely differs from the first in a way these two rules can see.

### 2. What "need not be learned" was tested on

- **The headroom is tiny at $K=2$.** The whole selection effect is +0.30 ± 0.12. The referee replaces 1,339 plans, and 62 of those change a hard-gate outcome (53 fixed, 9 broken).
- **Every comparison between verdict sources is inside that noise.** +0.23, +0.26 and +0.30 have overlapping intervals. The same-map learned verifier is within ±0.15 of the referee. The supported statement is *no detectable difference*, not *the analytic rule is better*.
- **The learned verifiers are small re-implementations.** The same-map one has 4.15M parameters and 66,385 labelled candidates. Learned scorers elsewhere in the wiki are trained on thousands of vocabulary trajectories per scene over the whole training set ([[sources/hydra-mdp-pp.md]]). The scaling curve is still rising at 50% and its last two points (+0.33, +0.27) differ by less than their error bars.
- **The comparison is on EPDMS, which is to the paper's credit.** The referee checks only NC and DAC, while the learned verifiers predict the full metric set and could have gained on progress, TTC or comfort. At $K=2$ they did not.
- **Two things the figure does establish.**
  - *Gating matters more than the verdict source.* Always taking the higher-scoring candidate (argmax) is −0.04 with 2,161 replacements, while gated versions of learned scores are +0.23 and +0.26.
  - *A second sample is not better on average.* Random replacement is +0.02.

### 3. The referee is a lossy copy of the evaluator {#fidelity}

| Audit | Result | Consequence |
|---|---|---|
| Ground-truth maps, drivable area | Precision 0.995, **recall 0.631** | It misses more than a third of the official violations on this subset. A "passing" winner can be an official violator |
| Ground-truth maps, collision | Precision 0.931, recall 0.792 | One in five official collisions is missed |
| Predicted maps, drivable-area margin | Correlation 0.895, **MAE 0.502 m** | The mean error exceeds the 0.4 m alarm and replacement threshold |
| Predicted maps, collision margin | **Not reported** | The vehicle-occupancy readout has to forecast other vehicles for 4 s from one front camera; its accuracy is unmeasured |

- High precision is what hard-negative mining needs, and the paper says so. Low recall matters for the other half of each pair.
- The audit set is unusual. It is described as the 1,060 navtest scenes "where the base policy scores zero". The base policy's own sub-scores (DAC 97.79, NC 99.31) imply roughly 270 drivable-area and at most about 85 collision failures out of 12,146. How 1,060 was reached is not explained. Within those scenes, only about half of the re-sampled candidates fail a gate (always-safe agreement 73.5% over two gates).
- The difference between rule and evaluator is attributed to the 0.4 m grid and five query points against the evaluator's exact polygons.

### 4. What the distillation is

- **It is small.** 462 pairs and 850 steps move an 8B-parameter tower by +0.92 EPDMS. The gain repeats on the 4B model (+1.03 PDMS) with the same readout, referee and settings.
- **It is mostly drivable-area compliance.** Full system against base, 16B: DAC +0.85, NC +0.18, TTC +0.50, EP +0.54. 4B: DAC +0.88, NC −0.01, TTC unchanged, EP +1.27. The collision half of the referee leaves little trace on the smaller model.
- **More than half of it is self-training on filtered samples.** With $\beta=0$ the pairwise term is off and the model is only fine-tuned on the 462 referee-selected winners: +0.56. The preference term adds +0.36.
- **The winner rule matters.** Largest clearance beats nearest-to-expert by +0.29 (CI +0.14 to +0.45). The preferred trajectory is chosen with no reference to the human driver.
- **A control is missing.** No row continues plain imitation on the same scenes for the same 850 steps, so the share of +0.56 that is "more fine-tuning on hard scenes" is unknown.
- **The supervision is the benchmark's metric.** The referee re-implements NAVSIM's collision and drivable-area gates with the evaluator's own tracker and dynamics, on annotated maps. This is the same kind of privileged signal as a PDMS reward in RL or a distilled scorer ([[concepts/selection-based-planning.md]]), applied as a preference.

### 5. Paired confidence intervals, for the first time here {#paired-ci}

| Comparison | 95% half-width | Implied SE |
|---|---:|---:|
| Two checkpoints, independent samples (distilled against base) | 0.21 | 0.11 |
| Two training variants (winner rule) | 0.16 | 0.08 |
| Selection against no selection, same first sample | 0.12 | 0.06 |
| Two verdict sources on the same candidates | 0.15 | 0.07 |
| Selection on the distilled policy | 0.08 | 0.04 |

- An unpaired difference of two navtest means has an SE of about 0.23 (from the per-scene SD of about 18 in [[sources/suv.md]]). Pairing on scenes roughly halves it for two different checkpoints.
- These intervals cover scene sampling. They do not cover training-seed variance, which no paper has measured.
- See [[concepts/evaluation-variance.md#paired-ci]].

### 6. Against PhysWAM, the same group's previous paper {#vs-physwam}

| | [[sources/physwam.md]] | DriveReferee base | DriveReferee distilled |
|---|---|---|---|
| Backbone | Cosmos 3 Nano | Cosmos3-Nano | Cosmos3-Nano |
| Cameras | 3 | 1 | 1 |
| Generated | Video + metric depth + ego motion | Video + actions | Video + actions |
| EPDMS | 90.3 (88.4 without its depth–motion loss) | 90.69 | 91.61 |
| PDMS | "91.4" (not a v1 measurement) | 91.08 | 91.96 |
| Selection result | Medoid of 8: +0.1; oracle of 8: +3.8 | Analytic referee, $K=2$: +0.30 | +0.0 |

- **A one-camera, imitation-only fine-tune of the same backbone is 0.4 EPDMS above PhysWAM's full system.** DriveReferee does not cite PhysWAM or include it in Table I, and the two training recipes differ in ways neither paper isolates.
- **The v1 column is a v1 column this time.** All four DriveReferee rows sit 0.9–1.5 above the closed form of their own sub-scores and have v1-scale EP (84–85). PhysWAM's "v1" rows sat at 0.0 with v2 sub-scores ([[concepts/navsim-benchmark.md#physwam]]).
- **FVD loses its floor.** PhysWAM reported 111.3 at 600 clips, 42.2 at 1,200, and a recorded-against-recorded floor of 91.5 at 600. Table III gives 23.90 with no clip count. The internal comparison (23.90 against 24.25, same scenes and seeds) is sound; the cross-paper column is not, as the caption says.
- **Oracle headroom is not reported here.** PhysWAM found +3.8 EPDMS among 8 samples. How much of that a two-rule referee could reach at larger $K$ is the natural follow-up, and the paper tests only $K=2$.

### 7. Table notes

- **The EPDMS evaluator is not stated and no v2 sub-scores are printed**, so the residual check cannot be run on 91.56. The column mixes conventions in its baselines (DriveVLA-W0 86.1 is the shared-block row; Metis 89.5 is corrected-like).
- **Scope of the claim.** The paper says it outperforms "the listed generative world-action models". Not listed: [[sources/wa-jepa.md]] (91.8 / 91.7), SimWAM (91.5), [[sources/suv.md]] (90.8 / 91.0), [[sources/da-wam.md]] (93.7, a learned scorer on per-candidate futures) and PhysWAM. On PDMS, 92.02 is above all of these except DA-WAM. On EPDMS, WA-JEPA's 91.7 is higher if the evaluators match.
- **DriveVLA-W0's 90.2** is its anchor-based multi-candidate result, listed without a mark.
- **"No external training data"** is accurate about datasets. The method does use map and box annotations (for the readout) and the evaluator's gate logic on them (for the preference pairs).
- **TTC is 96.86 for both 4B rows** while DAC and EP move by about a point.
- New to the wiki: CoPhy (91.4 PDMS, a semantic-BEV physical-safety objective with RL), UNIVERSE (91.0), DriveVer, ForgeDrive, PerceptDrive.

### 8. Cost

- A 30 ms CPU referee is cheap. The policy is not: each sample is a 16B joint video–action generation. The same group measured 9.4 GPU-seconds per plan for PhysWAM's three-camera, depth-generating model on this backbone; no figure is given here.
- On the base policy, alarms fire in about a third of scenes and change a hard-gate outcome in 0.5% of them (62 of 12,146). The full system's alarm rate is 34.3%.

---

## Relationships

- **[[sources/physwam.md]]**: same group, same backbone, and the paper whose label-free medoid selector this one replaces with a geometric one. See [the comparison above](#vs-physwam).
- **[[sources/drivefuture.md]]**: the wiki's price for a learned scorer is +20.9 EPDMS on navhard over 100 proposals. DriveReferee's analytic selector is +0.30 on navtest over 2. The two numbers are not in conflict; they are at opposite ends of candidate count, split difficulty and base strength.
- **[[sources/hydra-mdp-pp.md]]**: the origin of the learned-verdict recipe DriveReferee argues against (one distilled head per evaluator sub-score). Hydra-MDP++ found the selection *rule* its largest lever; here the gating rule matters more than who supplies the score.
- **[[sources/feaxdrive.md]]**: the nearest mechanism. It also samples a drivable-area distance field at the vehicle's corners, but uses the gradient to correct a diffusion sample, and takes the local map as given. DriveReferee predicts the map and selects instead of steering.
- **[[sources/reflectdrive.md]]**: the other inference-time safety method that needs no gradients. Its scorers are also rule-based checks (NC, DAC, TTC), with a ground-truth-oracle variant marking the ceiling. See [[concepts/inference-time-safety.md]].
- **[[sources/plan-r1.md]]**: puts the same two gates (collision, drivable area) into training as multiplicative reward terms under GRPO. DriveReferee puts them into a pairwise preference on a few hundred scenes.
- **[[sources/reworld.md]]**: also trains a flow planner against a mined hard negative close to the expert (its repulsion term, +0.7 PDMS). DriveReferee's loser is the expert-nearest violator among the policy's own samples.
- **[[sources/adaptive-wam.md]]**: reported that more than 95% of navtest candidate groups are tied, jointly perfect or jointly zero. That is consistent with 62 consequential replacements out of 1,339 here.
- **[[sources/metis.md]] / [[sources/suv.md]]**: WAMs that report oracle best-of-6 as headroom. DriveReferee is the first WAM paper here to test a deployable selector against matched alternatives.

---

## Limitations

**The central claim**

1. **Tested at $K=2$ only**, where selection is worth +0.30 ± 0.12 and no two verdict sources can be separated.
2. **The learned baselines are small and re-implemented**, trained on at most 66,385 labels. Their scaling curve has not clearly saturated.
3. **Two rules only.** Collision and drivable area. The paper lists lane direction and traffic lights as future work; TTC, progress and comfort are also outside the referee.
4. **The analytic rule has 63% / 79% recall** against the evaluator it copies, on ground-truth maps.

**The system**

5. **Test-time selection is dead weight in the final system** (+0.06 PDMS, −0.04 EPDMS after distillation) at 1.34× the sampling cost.
6. **Predicted-map error (0.50 m) exceeds the decision threshold (0.4 m)**, and collision-map accuracy is not reported.
7. **Privileged supervision** for both the preference pairs and the readout.
8. **No latency for the policy**, which is a 16B video generator.

**The evidence**

9. **No control for extra fine-tuning** in the distillation ablation.
10. **The fidelity-audit subset (1,060 scenes) is not reconcilable with the base policy's sub-scores** as described.
11. **navtest only, non-reactive.** No navhard, no HUGSIM, although the same group ran both for PhysWAM. A safety mechanism is evaluated only where other agents replay a log.
12. **EPDMS evaluator unstated**; no v2 sub-scores.
13. **Confidence intervals cover scene sampling**, not training seeds. Single training runs.
14. **The paper's own stated limitations**: two rules; map errors cause wrong verdicts; tied to NAVSIM's safety logic on a raster.

**Source conversion**

15. Five figure files are in `raw/assets/`; Figure 1 has only panel (a). All three tables are present, and the values printed inside Figures 3 and 4 are transcribed above. No code link. The front matter's author field is empty.

---

## Key Cross-References

- [[concepts/inference-time-safety.md]] — an analytic rule on a predicted map, and the finding that it stops paying once the rule is distilled into the policy.
- [[concepts/selection-based-planning.md]] — a computed verdict against learned ones at matched budget; gating against argmax.
- [[concepts/best-of-n.md]] — a deployable two-sample selector next to the oracle numbers of the same backbone.
- [[concepts/rl-for-ad.md]] — preference distillation from 462 pairs as an alternative to reward-based RL.
- [[concepts/evaluation-variance.md]] — the first paired per-scene confidence intervals on navtest.
- [[concepts/navsim-benchmark.md]] — 92.02 / 91.56 (91.96 / 91.61 single pass).
- [[concepts/world-model-for-ad.md]] — a video WAM at 92 PDMS whose video quality is unchanged by a 0.9-point planning gain.
- [[concepts/perception-for-planning.md]] — a BEV readout on frozen WAM features used by a rule, not by the planner.
- [[concepts/foundation-backbones-for-ad.md]] — Cosmos3-Edge (4B) against Cosmos3-Nano (16B): +0.6.

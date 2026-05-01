---
title: "FeaXDrive: Feasibility-aware Trajectory-Centric Diffusion Planning for End-to-End Autonomous Driving"
type: source-summary
sources: [raw/papers/FeaXDrive_ Feasibility-aware Trajectory-Centric Diffusion Planning for End-to-End Autonomous Driving.md]
related: [concepts/diffusion-planner.md, concepts/rl-for-ad.md, concepts/navsim-benchmark.md, concepts/inference-time-safety.md, sources/diffusiondrive.md, sources/diffusiondrive-v2.md, sources/recogdrive.md, sources/had.md]
created: 2026-05-01
updated: 2026-05-01
confidence: high
---

# FeaXDrive

FeaXDrive reframes diffusion planning around the predicted clean trajectory `x0` rather than the noise residual. The goal is to make feasibility constraints act directly in trajectory space: curvature/kinematics during training, drivable-area compliance during reverse sampling, and feasibility-aware GRPO during post-training.

**Source**: `raw/papers/FeaXDrive_ Feasibility-aware Trajectory-Centric Diffusion Planning for End-to-End Autonomous Driving.md`  
**arXiv**: https://arxiv.org/html/2604.12656v2  
**Authors**: Baoyun Wang, Zhuoren Li, Ran Yu, Yu Che, Xinrui Zhang, Ming Liu, Jia Hu, Lv Chen, Bo Leng  
**Affiliations**: Tongji University; Nanyang Technological University

## Key Takeaways

- The paper argues that noise-centric diffusion is poorly aligned with planning feasibility because curvature, kinematics, and drivable-area constraints are naturally defined over the clean trajectory, not over noise.
- FeaXDrive directly predicts the clean trajectory estimate at each diffusion step and uses it as the common object for supervised loss, curvature regularization, drivable-area guidance, and GRPO reward shaping.
- The method improves feasibility substantially: FeaXDrive-IL reduces curvature violation rate to 0.88% vs. DiffusionDrive 8.59% and ReCogDrive-IL 8.05% under the paper's reproduced evaluation.
- Feasibility-aware GRPO reaches 90.00 PDMS while keeping curvature violations at 2.40%; standard GRPO reaches a higher 90.56 PDMS but raises curvature violations to 5.79%.
- Final NAVSIM-v1 performance is competitive but not wiki-frontier: 90.0 PDMS is below DriveSuprim, HybridDriveVLA, DynVLA, FLARE, DiffusionDriveV2, WAM-Diff, DriveFine, DriveVA, and Drive-JEPA in the broader wiki.

## Method

### Trajectory-Centric Diffusion

Standard diffusion planners often predict noise:

```text
epsilon_hat = f_theta(x_t, t, c)
```

FeaXDrive predicts the clean trajectory estimate directly:

```text
x0_hat^(t) = f_theta(x_t, t, c)
```

The planner then performs reverse diffusion updates using the clean trajectory estimate. This makes `x0_hat` the explicit interface for downstream feasibility modeling.

### Adaptive Curvature-Regularized Training

The method estimates curvature from the predicted clean trajectory after a differentiable smoothing operation. Curvature is computed with arc-length parameterization to avoid artifacts from non-uniform waypoint spacing.

The allowable curvature bound is speed-adaptive:

```text
kappa_adp_i = min(kappa_geo_max, a_lat_max / (v_i^2 + eps_v))
```

Implementation details use a Chrysler Pacifica minimum turning radius of about 6.0 m, giving `kappa_geo = 0.166 m^-1`, and a lateral acceleration limit of `6 m/s^2`. The curvature loss penalizes only violations above the adaptive bound.

### Drivable-Area Guidance During Sampling

During inference, FeaXDrive builds a local drivable-area signed distance field (SDF) from the local map. Instead of checking only trajectory center points, it samples the SDF at the four corners of the vehicle footprint for each future waypoint. A soft barrier objective is activated when the footprint approaches or leaves the drivable area.

The guidance modifies the clean trajectory estimate inside each reverse diffusion step, not as a post-processing operation after sampling. This makes it an inference-time diffusion-guidance mechanism over `x0`.

### Feasibility-Aware GRPO

The GRPO reward augments benchmark task reward with a trajectory feasibility preference:

```text
R(x0, c) = R_task(x0, c) + lambda_fea R_fea(x0, c)
```

In this paper, `R_fea` is based on the adaptive curvature feasibility test. The result is a more conservative post-training objective: it trades 0.56 PDMS against standard GRPO for much lower curvature violation.

## Figures

![[frame.png]]

**Figure 1.** FeaXDrive overview. The method replaces noise-centric diffusion planning with trajectory-centric diffusion planning and adds training-time curvature regularization plus inference-time drivable-area guidance.

![[architecture.png]]

**Figure 2.** Full architecture: frozen VLM scene-context encoding, trajectory-centric DiT planner, adaptive curvature-regularized training, drivable-area guided sampling, and feasibility-aware GRPO post-training.

![[x1 32.png]]

**Figure 3.** Curvature violation counts by speed range. Noise-centric baseline has 826 low-speed and 554 higher-speed violations; trajectory-centric reduces them to 677 and 235; feasibility training reduces both to 8.

![[x2 30.png]]

**Figure 4.** Drivable-area violation counts. Baseline has 748 violations, trajectory-centric has 658, and adding feasibility guidance reduces this to 317.

![[feaxdrive_latency_breakdown.png]]

**Figure 5.** Inference latency breakdown: VLM backbone 245.33 ms, planner 82.96 ms, SDF build 16.03 ms, guidance 4.41 ms, total 348.73 ms.

![[case1_baseline.png]]

**Figure 6.** Qualitative example showing baseline vs. FeaXDrive trajectory behavior in BEV and front-camera view. The paper uses these examples to illustrate kinematic infeasibility, local geometric irregularity, and drivable-area violation in the baseline, with FeaXDrive producing smoother and more compliant trajectories.

## Tables

### Table 1: Failure Causes in Score-Zero Planning Scenes

| Failure cause | DiffusionDrive | ReCogDrive |
| --- | ---: | ---: |
| Drivable-area non-compliance | 69.25% | 56.59% |
| At-fault collision | 32.42% | 45.44% |
| Both | 1.67% | 2.03% |

### Table 2: NAVSIM Main Results

| Method type | Model | NC | EP | Comfort | TTC | DAC | PDMS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IL | VADv2 | 97.2 | 76.0 | 100 | 91.6 | 89.1 | 80.9 |
| IL | Driving-GPT | 98.9 | 79.7 | 95.6 | 94.9 | 90.7 | 82.4 |
| IL | Hydra-MDP | 97.9 | 77.6 | 100 | 92.9 | 91.7 | 83.0 |
| IL | UniAD | 97.8 | 78.8 | 100 | 92.9 | 91.9 | 83.4 |
| IL | PARA-Drive | 97.9 | 79.3 | 99.8 | 93.0 | 92.4 | 84.0 |
| IL | TransFuser | 97.7 | 79.2 | 100 | 92.8 | 92.8 | 84.0 |
| IL | DRAMA | 98.0 | 80.1 | 100 | 94.8 | 93.1 | 85.5 |
| IL | ReCogDrive-IL | 98.3 | 81.1 | 100 | 94.3 | 95.1 | 86.8 |
| IL | DiffusionDrive | 98.2 | 82.2 | 100 | 94.7 | 96.2 | 88.1 |
| IL | WoTE | 98.5 | 81.9 | 99.9 | 94.9 | 97.3 | 88.3 |
| IL | FeaXDrive-IL | 98.1 | 83.3 | 100 | 93.6 | 97.5 | 88.7 |
| IL+RLFT | TransFuser w/GRPO | 98.0 | 88.5 | 100 | 96.6 | 94.7 | 87.9 |
| IL+RLFT | ReCogDrive w/GRPO | 98.1 | 85.9 | 100 | 95.0 | 96.7 | 90.5 |
| IL+RLFT | FeaXDrive | 98.2 | 84.2 | 100 | 94.7 | 98.3 | 90.0 |

### Table 3: Curvature Violation Rates

| Metric | DiffusionDrive | ReCogDrive-IL | ReCogDrive w/GRPO | FeaXDrive-IL | FeaXDrive w/FA-GRPO |
| --- | ---: | ---: | ---: | ---: | ---: |
| Curvature violation rate | 8.59% | 8.05% | 15.5% | 0.88% | 2.40% |

### Table 4: IL-Stage Ablation

| Method | x0 prediction | Curvature regularization | Sampling guidance | PDMS | EP | DAC | Drivable-area violation rate | Curvature violation rate |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | no | no | no | 85.32 | 79.56 | 93.84 | 6.16% | 11.36% |
| Trajectory-centric | yes | no | no | 86.56 | 81.26 | 94.58 | 5.42% | 7.51% |
| + training-time feasibility enhancement | yes | yes | no | 86.57 | 81.32 | 94.94 | 5.06% | 0.13% |
| FeaXDrive-IL | yes | yes | yes | 88.75 | 83.34 | 97.46 | 2.54% | 0.88% |

### Table 5: Post-Training Fine-Tuning Strategies

| Method | PDMS | EP | DAC | Drivable-area violation rate | Curvature violation rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| FeaXDrive-IL | 88.75 | 83.34 | 97.46 | 2.54% | 0.88% |
| FeaXDrive w/GRPO | 90.56 | 85.13 | 98.28 | 1.72% | 5.79% |
| FeaXDrive w/FA-GRPO | 90.00 | 84.20 | 98.31 | 1.69% | 2.40% |

## Relationships

- **DiffusionDrive**: FeaXDrive starts from the diffusion-planning lineage but argues that noise prediction makes feasibility indirect. Its clean-trajectory parameterization gives curvature and drivable-area constraints a direct target.
- **ReCogDrive**: FeaXDrive uses the same general VLM-conditioned diffusion family and InternVL3-2B initialization, but emphasizes physical trajectory feasibility. ReCogDrive w/GRPO scores higher on PDMS in Table 2 (90.5 vs. 90.0), while FeaXDrive has better DAC and much lower curvature violation.
- **DiffusionDriveV2 / HAD**: all use diffusion-family planners plus RL-style optimization, but FeaXDrive's contribution is not anchor hierarchy or metric-decoupled reward; it is trajectory-space feasibility modeling across training, inference, and post-training.
- **Inference-time safety methods**: FeaXDrive's drivable-area guidance is gradient-based and integrated into reverse sampling. It is not a post-hoc discrete repair loop like ReflectDrive and not a learned refinement expert like DriveFine.

## Limitations

- The final PDMS is competitive but not leaderboard-leading in the broader wiki; the contribution is feasibility improvement rather than absolute NAVSIM-v1 SOTA.
- Feasibility-aware GRPO only incorporates curvature feasibility into the reward. Drivable-area guidance is used at inference, but the GRPO reward does not yet unify all feasibility aspects.
- Drivable-area guidance relies on local map-derived geometric priors and SDF construction. The paper notes that lighter maps or online mapping could replace this later, but current experiments still assume usable local road geometry.
- Standard GRPO highlights a trade-off: it reaches higher PDMS than FA-GRPO but substantially worsens curvature violation, so benchmark score and trajectory feasibility can conflict.
- The method is VLM-latency dominated: total median inference is 348.73 ms, with the VLM backbone alone taking 245.33 ms.
- Evaluation is NAVSIM-only; there is no NAVSIM-v2/EPDMS, navhard, Bench2Drive, HUGSIM, nuScenes, or Waymo result.

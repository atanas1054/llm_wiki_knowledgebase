---
title: nuScenes and Waymo Evaluations
type: concept
sources: ["raw/papers/ResWorld_ Temporal Residual World Model for End-to-End Autonomous Driving.md", "raw/papers/MomWorld_ Momentum-Aware Latent World Model for Long-Horizon Autonomous Driving.md", "raw/papers/SUV_ Future Scene Understanding as Video Generation for End-to-End Driving.md", raw/papers/LWDrive_ Layer-Wise World-Model-Guided Vision-Language ModelPlanning for Autonomous Driving.md, raw/papers/Adaptive-WAM_ Quality-Guided Early-Exit Planningfrom Intermediate Video-Diffusion Features.md, raw/papers/See Tomorrow, Act Today_ Foresight-Driven Autonomous Driving.md, raw/papers/AutoVLA_ A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning.md, raw/papers/HERMES_ A Holistic End-to-End Risk-Aware Multimodal Embodied System with Vision–Language Models for Long-Tail Autonomous Driving.md, raw/papers/UniUGP_ Unifying Understanding, Generation, and Planing For End-to-end Autonomous Driving.md, raw/papers/Reasoning-VLA_ A Fast and General Vision-Language-Action Reasoning Model for Autonomous Driving.md, raw/papers/DriveVA_ Video Action Models are Zero-Shot Drivers.md, raw/papers/ExploreVLA_ Dense World Modeling and Exploration for End-to-End Autonomous Driving.md, raw/papers/OneDrive_ Unified Multi-Paradigm Driving with Vision-Language-Action Models.md, raw/papers/From Forecasting to Planning_ Policy World Model for Collaborative State-Action Prediction.md, raw/papers/Driving Intents Amplify Planning-Oriented Reinforcement Learning.md, raw/papers/SimWAM_ A Simple World Action Model for End-to-End Autonomous Driving.md]
related: [sources/resworld.md, sources/momworld.md, sources/suv.md, sources/qwen-drive-1.0.md, sources/lwdrive.md, sources/adaptive-wam.md, sources/foresight.md, concepts/navsim-benchmark.md, concepts/bench2drive.md, concepts/world-model-for-ad.md, concepts/intent-conditioned-planning.md, concepts/best-of-n.md, concepts/physicalai-av-benchmark.md, sources/autovla.md, sources/hermes.md, sources/uniugp.md, sources/reasoning-vla.md, sources/driveva.md, sources/explorevla.md, sources/onedrive.md, sources/policy-world-model.md, sources/dial.md, sources/drivewam.md, sources/simwam.md]
created: 2026-05-01
updated: 2026-10-02
confidence: high
---

## What They Measure

nuScenes and Waymo-style evaluations in this wiki are mostly open-loop: L2 displacement, collision proxy metrics, planning error, or WaymoE2E risk/route scores. They are useful for trajectory imitation and transfer, but they do not replace closed-loop NAVSIM or interactive Bench2Drive evaluation.

## Common Metrics

| Metric family | Typical use | Caveat |
| --- | --- | --- |
| L2/ADE/FDE | nuScenes trajectory accuracy | Rewards matching logged behavior, not necessarily safe closed-loop behavior. |
| Collision rate | nuScenes/Waymo proxy safety | Often computed against logged agents without reactive simulation. |
| RFS | WaymoE2E long-tail risk | Dataset/task-specific; not comparable to PDMS or DS. |
| FID/FVD | World-model visual quality | Video realism does not guarantee planning quality. |

## WOD-E2E Rater Feedback Score

[[sources/dial.md]] uses WOD-E2E RFS, which scores a predicted trajectory against up to three human-rated alternative trajectories rather than treating the logged path as the unique target. This makes RFS useful for detecting proposal support: a policy may generate a path preferred over the logged demonstration.

DIAL reports the logged trajectory at RFS 8.13 and an intent-pooled Best-of-128 ceiling of 9.14. That does not mean the model deploys at 9.14; oracle selection is required. Its intent-classified deployment-oriented held-out peak is 8.211.

Protocol caveats:

- RL uses 338 of the 438 labeled validation sequences.
- The remaining 100 sequences are used to select checkpoints and reward hyperparameters, so “held-out” means validation, not untouched test.
- “Full RFS” includes RL-training sequences.
- Standard RFS scores 3 s and 5 s anchors with a hard maximum over raters; DIAL uses denser anchors and label-softmax aggregation only during training.
- RFS is open-loop preference alignment and does not directly measure reactive collision avoidance or closed-loop stability.
- The paper tabulates `TR` but the available source extraction does not define it.

## WOD-E2E RFS: the wiki's entries in one place {#rfs-table}

[[sources/qwen-drive-1.0.md]] supplies the first table here that places several ingested methods side by side on the **test** split.

| Method | Split | RFS | Note |
|---|---|---:|---|
| Human driver (logged trajectory) | val | 8.13 | the reference [[sources/dial.md]] also reports |
| Qwen-Drive-1.0-RL | val | **8.45** | RL trained on this split's rater annotations - in-sample |
| Qwen-Drive-1.0-RL, no shared-ADE anchor | val | **8.68** | ablation only; the highest number in that paper |
| MindVLA-U1 (RL) | val | 8.20 | not ingested |
| [[sources/dial.md]] | held-out val | 8.211 | intent-balanced GRPO; oracle Best-of-128 ceiling 9.14 |
| Qwen-Drive-1.0-SFT | val | 7.95 | with or without reasoning |
| [[sources/suv.md]] | test | 7.94 | Wan2.2-5B video WAM, no RL; ADE 1.24 / 2.90; its table omits Qwen-Drive and MindVLA-U1 |
| **Qwen-Drive-1.0-RL** | **test** | **7.91** | **+0.13 from RL; +0.02 from reasoning** |
| MindVLA-U1 (RL) | test | 7.87 | not ingested |
| [[sources/nord.md]] | test | 7.71 | reasoning-free, 6-17x less data |
| [[sources/autovla.md]] | test | 7.56 | |
| [[sources/hermes.md]] | not stated | 6.81 | distilled risk-aware student |

Two protocol notes this page now tracks:

- **In-sample RFS is not a driving claim.** Qwen-Drive states it plainly: because the val rater annotations supply its reward, 8.45 above the human 8.13 "indicates effective optimization of preference alignment on the training scenarios rather than generalization beyond human driving." The un-anchored ablation reaching 8.68 makes the point harder. **Only test-split RFS should be read as a result**, and the spread there across all ingested methods is 7.56-7.91.
- **Two papers give different counts for the annotated validation set.** DIAL uses 338 of **438** labelled validation sequences for RL and holds out 100; Qwen-Drive trains RL on **479** rater-annotated scenarios. Whether these are different dataset versions, different filters, or different definitions of "annotated" is unresolved, and it affects what "held out" means in both papers.

## Takeaways

- Treat nuScenes/Waymo as complementary evidence for generalization, not as direct leaderboard substitutes for NAVSIM or Bench2Drive.
- Zero-shot transfer claims should report absolute values, not only percent improvement over one baseline. (DriveVA's paper did not; SimWAM's Table 6 supplied them later — 0.84 L2 / 0.06 collision.)
- World-model papers need both generation metrics and downstream planning metrics; strong FVD alone is insufficient.
- The same caveats extend to the newer, much larger [[concepts/physicalai-av-benchmark.md]], which also reports ADE/FDE only. Scale improves coverage of rare events but does not convert an open-loop displacement metric into evidence about closed-loop behavior — and where a paper curates its own test subset (as [[sources/drivewam.md]] does), the comparison is not yet leaderboard-grade.

## OneDrive nuScenes Result

**OneDrive** ([[sources/onedrive.md]]) reports one of the strongest nuScenes open-loop planning entries in the wiki:

| Method | L2 Avg | Collision Avg | Notes |
| --- | --- | --- | --- |
| SOLVE-VLM | 0.28 | 0.20 | AR/text VLM path |
| ColaVLA | 0.30 | 0.23 | Non-AR baseline |
| **OneDrive** | **0.28** | **0.18** | Single causal decoder; detection/lane/planning query sequence |

This is meaningful evidence for the architecture, but it remains open-loop: the result should not be treated as equivalent to NAVSIM PDMS or Bench2Drive driving score.

## Zero-Shot NAVSIM → nuScenes: The WAM Cluster

[[sources/simwam.md]]'s Table 6 is the wiki's first side-by-side of NAVSIM-trained world-action models evaluated on nuScenes **without fine-tuning or auxiliary supervision**, and it resolves a gap this page previously flagged: DriveVA's paper reported only percentage improvements over PWM, never absolutes.

| Method | Finetuned | L2 Avg ↓ | Collision Avg ↓ |
| --- | --- | ---: | ---: |
| UniAD (reference, finetuned) | ✓ | 1.03 | 0.31 |
| GenAD (reference, finetuned) | ✓ | 0.91 | 0.43 |
| Epona (reference, finetuned) | ✓ | 1.25 | 0.36 |
| DriveVA | ✗ | **0.84** | 0.06 |
| DriveWAM | ✗ | 0.96 | 0.06 |
| SimWAM | ✗ | 0.96 | **0.04** |

Two things stand out. All three zero-shot WAMs beat every finetuned baseline on collision rate by roughly an order of magnitude, which is the strongest evidence in the wiki that video-prior training transfers as a *safety* prior rather than a trajectory-matching one. And the L2/collision split is stark: SimWAM ties DriveWAM on L2 (0.96) while halving collisions (0.04 versus 0.06), and DriveVA leads on L2 (0.84) without leading on collisions. This is the clearest illustration on this page of why L2 and collision rate should not be collapsed into one ranking — L2 rewards agreement with the logged nuScenes expert, which a NAVSIM-trained policy has no reason to reproduce.

**Caveat**: the DriveWAM row is not corroborated by [[sources/drivewam.md]], whose ingested v1 clipping contains no nuScenes evaluation at all. Either SimWAM reproduced it or cited a later revision.

## Policy World Model nuScenes Result

**Policy World Model** ([[sources/policy-world-model.md]]) reports a safety-skewed nuScenes result: it does not dominate L2, but it has the lowest collision rate in its comparison table.

| Method | Ego status | L2 Avg | Collision Avg | Notes |
| --- | --- | ---: | ---: | --- |
| PWM | No | 0.78 | 0.07 | Better collision than Drive-OccWorld 0.11 and LAW 0.19; worse L2 than those methods. |
| PWM | Yes | 0.41 | 0.04 | Best collision in the paper's ego-status table; L2 trails Omni-Q 0.33, BEV-Planner 0.35, and VAD-Base 0.37. |

The result is useful evidence for future-frame forecasting as a safety prior. It should still be interpreted as open-loop nuScenes evidence, not as proof of closed-loop behavior under interactive agents.

## ForeSight: Hedging the Benchmark While Reporting On It

[[sources/foresight.md]] is a useful specimen of a pattern this page exists to name. Its nuScenes paragraph opens by conceding that "the scenarios and evaluation protocols in nuScenes are relatively simple and the metrics are not entirely comprehensive," citing the ego-status critique, then reports its own result on those metrics as "competitive performance."

| Method | Type | L2 Avg ↓ | Collision Avg ↓ |
| --- | --- | ---: | ---: |
| BEV-Planner | Planning | **0.46** | 0.49 |
| PARA-Drive | Planning | 0.48 | 0.25 |
| World4Drive | Planning + WM | 0.50 | 0.16 |
| GenAD | Planning | 0.52 | 0.19 |
| BridgeAD | Planning | 0.59 | 0.09 |
| MomAD | Planning | 0.60 | 0.09 |
| SparseDrive | Planning | 0.61 | **0.08** |
| LAW | Planning + WM | 0.61 | 0.30 |
| **ForeSight** | **Planning + WM** | **0.62** | **0.18** |
| UniAD | Planning | 0.69 | 0.12 |
| VAD-Base | Planning | 0.72 | 0.22 |

ForeSight wins no column. It is eighth of eleven on average L2 and sixth on average collision, and it is beaten on **both** by World4Drive — the other world-model entry in its own table. This is the same method that leads its NAVSIM category at 89.3 PDMS, which is the most direct illustration on this page of the divergence between the two benchmarks: **a design that helps under NAVSIM's closed-loop PDM scoring can be neutral-to-negative under nuScenes L2**, because L2 rewards reproducing the logged expert and a 2.5B generated-future prior has no particular reason to do that.

The hedge is fair on the merits — the ego-status critique is real and this page endorses it. But a paper that believes the metric is uninformative should say its result is uninformative, not that it is competitive.

**Secondary finding (Table 8)**: swapping the world model from Epona to Vista, planner fixed, degrades 6 of 8 columns (L2 0.62 → 0.64, collision 0.18 → 0.27). Presented as evidence of architecture-agnosticism, which it is, in the weak sense that the framework tolerates the swap rather than benefiting from it.

## The Zero-Shot WAM Cluster, Extended

[[sources/adaptive-wam.md]] adds a fourth NAVSIM-trained WAM evaluated on nuScenes without fine-tuning, and it is the first to report the full horizon breakdown alongside DriveVA rather than percentage deltas.

| Method | FT | L2 1s | L2 2s | L2 3s | **L2 Avg** | Col 1s | Col 2s | Col 3s | **Col Avg** |
| --- | :-: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| UniAD | yes | 0.48 | 0.96 | 1.65 | 1.03 | 0.05 | 0.17 | 0.71 | 0.31 |
| GenAD | yes | 0.36 | 0.83 | 1.55 | 0.91 | 0.06 | 0.23 | 1.00 | 0.43 |
| Epona | yes | 0.61 | 1.17 | 1.98 | 1.25 | 0.01 | 0.22 | 0.85 | 0.36 |
| DriveVLA-W0 | no | 0.43 | 1.26 | 2.60 | 1.43 | 0.22 | 0.66 | 1.42 | 0.77 |
| PWM | no | 2.06 | 3.91 | 6.00 | 3.99 | 0.12 | 0.15 | 0.86 | 0.36 |
| DriveVA | no | **0.33** | 0.76 | **1.43** | **0.84** | **0.00** | **0.07** | **0.12** | **0.06** |
| **Adaptive-WAM** | no | 0.35 | **0.71** | 1.58 | 0.88 | **0.00** | 0.09 | 0.15 | 0.08 |

Two observations. **The horizon profile differs from the average**: Adaptive-WAM leads DriveVA at 2s and trails it at 3s, so the 0.84-vs-0.88 average conceals a crossover rather than uniform dominance — another reason to distrust horizon-averaged L2 as a single ranking. And **the collision-rate story from the SimWAM cluster holds and strengthens**: both fully-generative and early-exit WAMs land at 0.06-0.08% average collision against 0.31-0.43% for fine-tuned nuScenes-native planners, roughly an order of magnitude, without ever seeing the target domain.

The efficiency contrast is worth carrying here too. DriveVA achieves its 0.84 by executing the full Wan backbone and generating future images; Adaptive-WAM reaches 0.88 with a single conditional forward to an intermediate block at **170 ms**. On this benchmark the extra 12+ seconds of video synthesis buys 0.04 m and 0.02 percentage points.

## LWDrive: The Protocol Is Not Stated {#lwdrive}

[[sources/lwdrive.md]] reports **0.37 m average $L_2$ and 0.16% collision** on the ST-P3 protocol, the best $L_2$ in its own table by 0.03 m over DriveVLM and RDA-Driver — and RDA-Driver's collision rate (0.10%) is better.

Two cautions, one generic and one specific to this paper.

**The generic one is the whole point of this page.** LWDrive consumes ego status (velocity, steering state, navigation command), and its own table carries Ego-MLP at 0.78 / 0.38 and BEV-Planner at 0.55 / 0.59 — the two rows that exist in the literature precisely to show that ego-status shortcuts dominate open-loop $L_2$. A 0.03 m margin on this metric carries no information about planning quality, and the paper's inference that it "can transfer beyond NAVSIM-style scoring" is not supported by a 3 cm difference.

**The specific one is that the protocol is unrecoverable.** The paper says only that it evaluates on "three planning benchmarks: NAVSIM, NAVSIM-v2, and nuScenes." It never states whether the nuScenes number comes from a **separately trained nuScenes model** or from **zero-shot transfer of the NAVSIM model**, and its implementation section describes only the two NAVSIM training stages. Given the cluster above — where the zero-shot column is the interesting one and fine-tuned nuScenes-native planners have collision rates an order of magnitude worse — that distinction changes how the row should be read entirely, and there is no way to settle it from the text. **The row is therefore recorded here but not placed in the zero-shot table.**

## MomWorld: a Six-Second Protocol, a Consistency Metric, and Three Robustness Subsets {#six-second}

[[sources/momworld.md]] is the first ingested paper to use the **six-second nuScenes protocol** that its predecessor MomAD introduced, and it brings four evaluation devices this page had not recorded.

| Device | What it is | Size |
|---|---|---|
| Six-second planning | A separately trained 12-waypoint model, scored at every second from 1 to 6 s | Full validation split |
| **TPC** (Trajectory Prediction Consistency) | Mean distance between matching waypoints of two consecutive planning outputs; reported at 4, 5 and 6 s | Full validation split |
| Turning-nuScenes | Samples whose ground-truth ego moves more than 25 m between 0.5 and 3.0 s | 680 samples, 17 scenes |
| Adv-nuSc | Adversarial agent trajectories generated by Challenger | 156 scenes, 6,115 samples |
| nuScenes-C | 27 corruption types at 5 severities; snow, rain and fog are used | Validation split |

### The six-second table

| Method | L2 1s | 3s | 6s | Avg 1–6 s | Col 1s | 3s | 6s | Avg 1–6 s | TPC avg (4–6 s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UniAD | 0.47 | 1.35 | 3.07 | 1.70 | 0.25 | 0.61 | 2.51 | 1.06 | 1.90 |
| SparseDrive | 0.43 | 1.23 | 2.95 | 1.59 | 0.19 | 0.56 | 2.33 | 0.97 | 1.66 |
| World4Drive | 0.42 | 1.21 | 2.79 | 1.53 | 0.16 | 0.47 | 2.14 | 0.87 | – |
| Epona | 0.39 | 1.17 | 2.75 | 1.50 | 0.14 | 0.45 | 2.23 | 0.87 | – |
| GuideFlow | 0.42 | 1.21 | 2.63 | 1.48 | 0.12 | 0.42 | 2.15 | 0.86 | – |
| LAW | 0.40 | 1.16 | 2.61 | 1.46 | 0.19 | 0.57 | 2.31 | 0.96 | – |
| MomAD | 0.41 | 1.13 | 2.45 | 1.42 | 0.17 | 0.54 | 2.13 | 0.90 | 1.42 |
| DIVER | 0.38 | 1.10 | 2.49 | 1.37 | 0.13 | 0.44 | 2.11 | 0.87 | – |
| **MomWorld** | **0.27** | **0.86** | **2.31** | **1.17** | **0.02** | 0.42 | **1.97** | **0.79** | **1.19** |

### Three things to know before citing a row from it

**1. A six-second row is not comparable with a three-second row, even at the same horizon.** The six-second models are much worse at 1–3 s than the same methods' three-second models.

| Method | 3-s model, avg L2 / collision (1–3 s) | 6-s model, avg over the same 1–3 s |
|---|---|---|
| MomAD | 0.60 / 0.09 (from [[sources/foresight.md]]'s table) | 0.80 / 0.34 |
| SparseDrive | 0.61 / 0.08 | 0.84 / 0.35 |
| UniAD | 0.69 / 0.12 | 0.91 / 0.41 |

Collision rates at 1–3 s are about four times higher in the six-second table. Two explanations are possible and the paper separates neither.
- *Training for twelve waypoints costs near-term accuracy.*
- *The metric convention differs.* The three-second table uses the ST-P3 convention, which averages each horizon over all earlier waypoints. UniAD's per-horizon values elsewhere on this page are 0.48 / 0.96 / 1.65, close to the six-second table's 0.47 / 0.91 at 1–2 s and not at 3 s (1.35).

MomWorld's three-second model has no main-table result in the ingested clipping.

**2. The "long-horizon" gains are largest at the shortest horizons.** Against MomAD, MomWorld's L2 falls 34% at 1 s, 39% at 2 s and 6% at 6 s. Its collision rate at 4 s equals MomAD's (0.83). A 6-s average hides this the way a 3-s average hides Adaptive-WAM's crossover with DriveVA in the zero-shot table above.

**3. TPC is not an independent metric in this paper's ablations.** Across 27 ablation rows, L2@6 minus TPC@6 stays between 0.84 and 0.88 m (standard deviation 0.008). Across methods in the table above the same difference ranges from 0.66 to 0.96. See [[sources/momworld.md#regularities]]. TPC is a sensible metric; this paper's ablation tables are not evidence about it.

### The robustness subsets

MomWorld is best or tied in all 22 of its cells across Turning-nuScenes, Adv-nuSc and nuScenes-C, by 0.001 to 0.04 percentage points.

- Turning-nuScenes has 680 samples. One sample is 0.15 percentage points, so a 0.01-point difference in collision rate (0.67 → 0.66 at 3 s) is smaller than one sample.
- The paper states the comparison values are "literature references rather than matched reruns".
- Adv-nuSc average collision: MomWorld 0.733, GraphWorld 0.742, DIVER 0.752, SparseDrive 1.026, UniAD 3.950, VAD 7.050. The spread among the three newest rows (all from one group) is 0.019 points; the spread to the older baselines is where the subset discriminates.

**Use for this page.** The subsets are worth knowing as instruments: turning-only, adversarial-agent and corrupted-sensor slices are exactly what the open-loop average hides. The MomWorld margins on them are too small to read.

## ResWorld: Both Averaging Conventions, Ego Status Reported Both Ways, and Ablations of a Few Samples {#resworld}

[[sources/resworld.md]] is a perception-free BEV planner (GeoBEV + ResNet-50, 256×704, three frames) with a residual-input world model. Its nuScenes table is better organised than most on this page:

| Convention | ResWorld, no ego status | ResWorld, ego status in the planner | Strongest comparison row |
|---|---|---|---|
| UniAD-style (per horizon) | 0.65 m / 0.23% | 0.59 m / 0.17% | SSR (re-evaluated by the authors) 0.74 / 0.31 |
| VAD-style (temporal average) | 0.35 m / 0.07% | **0.30 m / 0.06%** | BEV-Planner++ (ego status) 0.35 / 0.34; DiffusionDrive 0.57 / 0.08 |

- **Both conventions and both ego-status settings are printed**, and the averages reproduce from the per-horizon values. That is what the ego-status critique endorsed [above](#foresight-hedging-the-benchmark-while-reporting-on-it) asks for.
- **Ego status is as large as the method on L2.** In its ablation, ego status alone takes the baseline from 0.71 to 0.65 m, and the method alone does the same. On collision the method is larger (0.31 → 0.23 against 0.31 → 0.28).
- **The ablations are a few samples apart.** Collision differences between ablation rows are 0.02–0.11 percentage points, averaged over three horizons. On the validation split (roughly 6,000 samples), 0.04 points is two or three samples, the same granularity problem as the Turning-nuScenes margins [above](#the-robustness-subsets). No seeds.
- **The gain sits at one horizon.** Without ego status, 3 s collision is 0.64% for both baseline and full model; the average falls because 2 s goes from 0.25% to 0.04%.
- **Its LAW row matches [[sources/foresight.md]]'s** (0.61 m / 0.30%, VAD-style), a small cross-paper consistency check.

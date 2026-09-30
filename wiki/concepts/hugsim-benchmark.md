---
title: HUGSIM Benchmark
type: concept
sources: ["raw/papers/MM-Future_ Multi-Mode Joint World–Action Modeling for Autonomous Driving.md", raw/papers/HAD_ Combining Hierarchical Diffusion with Metric-Decoupled RL for End-to-End Driving.md, raw/papers/Latent-WAM_ Latent World Action Modeling for End-to-End Autonomous Driving.md, raw/papers/WA-JEPA_ Rethinking the Video JEPA Paradigm forWorld-Action Modeling in Autonomous Driving.md, "raw/papers/PhysWAM_ Physically Consistent World Action Model for Autonomous Driving.md"]
related: [sources/mm-future.md, sources/had.md, sources/latent-wam.md, sources/wa-jepa.md, sources/physwam.md, concepts/navsim-benchmark.md, concepts/bench2drive.md, concepts/world-model-for-ad.md, concepts/inference-latency.md, research-directions.md]
created: 2026-05-01
updated: 2026-09-30
confidence: medium
---

## What It Is

HUGSIM is a closed-loop autonomous driving benchmark built on Gaussian-splatting reconstructions of real driving logs, used to test interactive planning beyond NAVSIM's non-reactive simulator. Scenarios are drawn from four source datasets — nuScenes, KITTI-360, Waymo, and PandaSet — and split by difficulty into Easy, Medium, Hard, and Extreme.

Its distinguishing property among the wiki's closed-loop benchmarks is that **it is naturally a zero-shot test**. Because the source datasets are separate from NAVSIM and Bench2Drive, a NAVSIM-trained planner evaluated on HUGSIM is being tested for cross-domain generalization, not just closed-loop competence. [[sources/latent-wam.md]] and [[sources/wa-jepa.md]] both use it this way.

## Metrics

HUGSIM reports route completion (RC) and HD-Score (HDS). HD-Score combines safety and driving-quality terms — no-collision and drivable-area compliance with weighted time-to-collision and comfort — then scales by route completion. Some papers also report a HUGSIM-internal PDMS, which is *not* NAVSIM's PDMS.

These numbers are not comparable to NAVSIM PDMS or EPDMS. HUGSIM has different scenario construction, reactive agents, and scoring.

Note the scale convention differs by paper: [[sources/had.md]] and [[sources/latent-wam.md]] report on a 0-100 scale, [[sources/wa-jepa.md]] on 0-1. The tables below preserve each paper's convention.

## ⚠ The Benchmark Changed: Two Incompatible Eras

**Results on this page split into two groups that cannot be compared.** HUGSIM grew from **345 scenarios to 436**, and [PR #57](https://github.com/hyzhou404/HUGSIM/pull/57) applied a **trajectory-to-heading coordinate-order correction** to the controller. [[sources/wa-jepa.md]] pins commit [`ead17f2`](https://github.com/hyzhou404/HUGSIM/commit/ead17f2ad97f71fd21fa6f66237a7c05364ed98e) and rescores every baseline under it; the HAD and Latent-WAM results predate that snapshot.

| Era | Scenarios | Heading fix | Entries |
|---|---|---|---|
| Earlier release | 345 | No | HAD-L (and DrivoR's *published* numbers). *Latent-WAM was filed here; its averages say otherwise, see [below](#mm-future)* |
| 436 scenarios, commit not stated | 436 | Unknown | The benchmark paper's UniAD / LTF / VAD, BeyondDrive, [[sources/physwam.md]], [[sources/mm-future.md]], and by weighting arithmetic [[sources/latent-wam.md]] |
| Current snapshot `ead17f2` | 436 | Yes (PR #57) | WA-JEPA, plus its rescored LTF / DrivoR / UniAD / VAD |

A coordinate-order fix in the controller changes how every planned trajectory is executed, so this is not a scenario-count adjustment that could be normalized away. **Do not read HAD-L's 30.8 HDS against WA-JEPA's 44.62 as a 14-point improvement.** This is the same class of problem as the NAVSIM-v2 evaluator drift documented in [[concepts/navsim-benchmark.md]], and it is now visible on both of the wiki's main benchmarks.

## Current Snapshot (436 scenarios, commit `ead17f2`)

All baselines rescored by WA-JEPA's authors under one code snapshot, sharing scenarios, ground-truth commands, controller, aggregation, and metric implementation, with each method keeping its native sensor configuration (LTF uses three front cameras, the rest four). Values on $[0,1]$.

| | WA-JEPA | LTF | DrivoR | UniAD | VAD |
|---|---:|---:|---:|---:|---:|
| NC | **0.6856** | 0.4428 | 0.5217 | 0.6555 | 0.4117 |
| DAC | **0.9635** | 0.9275 | 0.9559 | 0.9320 | 0.9028 |
| TTC | **0.6120** | 0.3751 | 0.4620 | 0.5156 | 0.2798 |
| Comf. | 0.6620 | 0.9478 | 0.9390 | 0.6633 | **0.9534** |
| PDMS *(HUGSIM-internal)* | **0.5717** | 0.3653 | 0.4475 | 0.4940 | 0.2831 |
| RC | **0.5689** | 0.3804 | 0.4721 | 0.4383 | 0.3006 |
| **HD-Score** | **0.4462** | 0.2310 | 0.3252 | 0.3124 | 0.1393 |
| Easy HDS ($n{=}80$) | **0.7977** | 0.6608 | 0.7799 | 0.6395 | 0.4197 |
| Medium HDS ($n{=}157$) | **0.5563** | 0.1547 | 0.2911 | 0.3718 | 0.0849 |
| Hard HDS ($n{=}96$) | **0.3060** | 0.1204 | 0.2000 | 0.2099 | 0.0770 |
| Extreme HDS ($n{=}103$) | 0.1362 | 0.1167 | **0.1407** | 0.0632 | 0.0626 |

Three observations the aggregate hides:

- **The Extreme tier goes to DrivoR**, 0.1407 vs. 0.1362, on 103 of 436 scenarios — roughly a quarter of the benchmark. Every method is near the floor there (0.06-0.14), which is the real story: **the hardest quarter of HUGSIM is essentially unsolved**, mirroring navhard Stage 2's stall near 40 EPDMS.
- **Comfort is where world-model planners lose.** WA-JEPA scores 0.6620 against ~0.95 for LTF, DrivoR, and VAD. UniAD is similarly poor at 0.6633. The pattern matches NAVSIM-v2's EC column, where world-model and flow/diffusion planners routinely trail rule-based and anchor-based ones — a continuous sampler has no mechanism enforcing kinematic consistency across closed-loop timesteps.
- **The gains concentrate in the middle.** WA-JEPA is +0.265 over DrivoR on Medium and +0.106 on Hard, but only +0.018 on Easy and −0.005 on Extreme. Closed-loop improvements are showing up where scenarios are hard enough to separate methods and not so hard that everything fails.

### Aggregation Robustness

WA-JEPA reports the same comparison under three aggregation rules — a check almost no ingested paper runs:

| Aggregation | WA-JEPA | LTF | DrivoR | UniAD | VAD |
|---|---:|---:|---:|---:|---:|
| Primary (difficulty-weighted by count 80/157/96/103) | **0.4462** | 0.2310 | 0.3252 | 0.3124 | 0.1393 |
| Dataset-uniform | **0.4483** | 0.2300 | 0.3246 | 0.3085 | 0.1304 |
| Scenario-uniform | **0.4464** | 0.2243 | 0.3194 | 0.3082 | 0.1266 |

Rankings are stable to within 0.002 across all three. HUGSIM's aggregation choice is therefore *not* a source of the disagreement between papers — the scenario set and controller version are.

### Per-Dataset Transfer

None of these datasets appears in WA-JEPA's training (nuPlan for Stage 1, NAVSIM navtrain for Stage 2), so every column is zero-shot.

| Dataset | $n$ | WA-JEPA | LTF | DrivoR | UniAD | VAD |
|---|---:|---:|---:|---:|---:|---:|
| nuScenes | 88 | **0.4725** | 0.3334 | 0.3830 | 0.3405 | 0.2069 |
| KITTI-360 | 113 | **0.2963** | 0.0969 | 0.2175 | 0.0550 | 0.0272 |
| Waymo | 108 | **0.5542** | 0.2478 | 0.4025 | 0.4372 | 0.1376 |
| PandaSet | 127 | **0.4702** | 0.2419 | 0.2955 | 0.4012 | 0.1500 |

KITTI-360 is uniformly the hardest domain and Waymo the easiest, for every method — so domain difficulty is a property of the reconstruction quality and scenario mix, not of any one planner. Winning all four separately is better evidence of domain-general transfer than the aggregate, and it is the strongest closed-loop generalization result in the wiki.

## PhysWAM: a Second Method on 436 Scenarios, Under an Unpinned Protocol {#physwam}

[[sources/physwam.md]] runs "all 436 HUGSIM episodes" zero-shot from NAVSIM-only training, with the same per-difficulty counts as WA-JEPA (80 / 157 / 96 / 103). It names no commit and quotes its baselines "as reported by the benchmark paper" and by BeyondDrive, where WA-JEPA rescored everything. Values ×100.

| | PhysWAM | PhysWAM w/o CPP | BeyondDrive *(as reported)* | UniAD *(as reported)* | LTF *(as reported)* | VAD *(as reported)* |
|---|---:|---:|---:|---:|---:|---:|
| Easy RC / HDS | **93.5 / 86.9** | 92.6 / 85.4 | 76.8 / 65.6 | 58.6 / 48.7 | 68.4 / 52.8 | 38.7 / 24.3 |
| Medium RC / HDS | **47.4** / 30.1 | 44.9 / 26.8 | 43.0 / **31.4** | 41.2 / 29.5 | 40.7 / 24.6 | 27.0 / 9.9 |
| Hard RC / HDS | 38.6 / 25.2 | 37.8 / 23.9 | 35.5 / 26.3 | **40.4 / 27.3** | 36.9 / 19.8 | 25.5 / 10.4 |
| Extreme RC / HDS | 26.1 / 13.4 | 25.4 / 12.0 | **29.6 / 16.2** | 26.0 / 14.3 | 25.5 / 8.1 | 23.0 / 8.2 |
| **Overall RC / HDS** | **48.9 / 35.5** | 47.5 / 33.4 | 46.2 / 34.8 † | 40.6 / 28.9 | 41.4 / 24.8 | 27.9 / 12.3 |
| NC / DAC / TTC / Comf. (overall) | 54.4 / 96.6 / 46.4 / 96.3 | 52.8 / 96.5 / 44.5 / 96.2 | – | – | – | – |

† BeyondDrive's overall is an unweighted mean of the four levels. The others are weighted by episode count. On one convention PhysWAM leads by more than the table shows: 38.9 vs 34.8 unweighted, or 35.5 vs 33.0 weighted. The paper states both.

**How it sits against the pinned snapshot above.**

| HD-Score | WA-JEPA (pinned, rescored) | PhysWAM (unpinned) |
|---|---:|---:|
| Easy | 79.8 | **86.9** |
| Medium | **55.6** | 30.1 |
| Hard | **30.6** | 25.2 |
| Extreme | 13.6 | 13.4 |
| Overall | **44.6** | 35.5 |
| Comfort | 66.2 | **96.3** |

- **PhysWAM does not cite WA-JEPA**, so the comparison is this page's. The scenario set and weighting match. The controller version is unknown for PhysWAM.
- **The quoted baselines differ from the rescored ones by 1.6–2.3 points**, in both directions: UniAD 28.9 quoted against 31.2 rescored, LTF 24.8 against 23.1, VAD 12.3 against 13.9. That is the size of the protocol uncertainty between the two tables. The 9-point overall gap and the 25-point Medium gap are well outside it.
- **The profiles differ more than the totals.** PhysWAM is the best Easy-tier result on this page and falls to BeyondDrive / UniAD levels from Medium on. WA-JEPA keeps 55.6 on Medium. [[sources/latent-wam.md]], also trained on NAVSIM only, had the same Easy-heavy shape in the earlier release (72.5 / 24.0 / 12.2). WA-JEPA is the one with multi-view nuPlan video pretraining. That is a pattern across three papers, not a controlled result.
- **Comfort 96.3 is a counterexample to "comfort is where world-model planners lose".** PhysWAM is a sampled flow-matching planner that generates the whole future, and its closed-loop comfort matches LTF and VAD. The deficit in WA-JEPA and UniAD is therefore not inherent to sampling.
- **Drivable-area compliance is above 93.8 at every difficulty** (96.6 overall, level with WA-JEPA's 96.4). The score is lost on NC and TTC, which is where a drivable-area hinge loss in training would not help.
- **The mechanism effect is measured in closed loop**: its depth–motion loss is worth +2.1 HD-Score and +1.4 RC, and keeping that loss active late in training is worth +1.2 HD-Score while changing nothing open-loop.
- **Per dataset** (RC / HDS, plain episode means): nuScenes 56.7 / 42.9, Waymo 55.8 / 42.8, KITTI-360 41.1 / 31.0, PandaSet 41.3 / 24.3. KITTI-360 Hard and Extreme are 6.1 and 0.7 HD-Score.
- **Simulation time, not wall-clock.** "The simulator advances only after the planner returns." One plan costs 9.4 GPU-seconds against a 0.5 s replanning interval, so the result says nothing about real-time closed-loop driving. See [[concepts/inference-latency.md]].

## MM-Future: a Third Unpinned Result on 436 Scenarios, and a Correction for Latent-WAM {#mm-future}

[[sources/mm-future.md]] evaluates its NAVSIM-trained (trainval) model zero-shot "over 436 scenarios", with four cameras and no stated commit. Values ×100.

| HD-Score | Easy | Medium | Hard | Extreme | Overall | RC overall |
|---|---:|---:|---:|---:|---:|---:|
| WA-JEPA (pinned, rescored) | 79.8 | **55.6** | **30.6** | 13.6 | **44.6** | 56.9 |
| PhysWAM | **86.9** | 30.1 | 25.2 | 13.4 | 35.5 | 48.9 |
| BeyondDrive (as reported) | 65.6 | 31.4 | 26.3 | 16.2 | 34.8 † | 46.2 |
| DrivoR (rescored by WA-JEPA) | 78.0 | 29.1 | 20.0 | 14.1 | 32.5 | 47.2 |
| **MM-Future** | 53.8 | 40.0 | 27.3 | 8.6 | 32.3 | 44.5 |
| Latent-WAM (as reported) | 72.5 | 24.0 | 12.2 | **18.1** | 28.9 | 45.9 |
| UniAD (as reported) | 48.7 | 29.5 | 27.3 | 14.3 | 28.9 | 40.6 |

† unweighted mean; 33.0 weighted.

- **Its claim to the highest HD-Score holds against the four baselines in its table** (VAD, LTF, UniAD, Latent-WAM). WA-JEPA, PhysWAM and BeyondDrive are higher on the same scenario count and are not cited. It is level with DrivoR, whose register-token encoder it builds on.
- **The profile is unlike the other NAVSIM-trained models.** Easy is the lowest of the recent methods, Medium is second only to WA-JEPA, and Extreme is the lowest in the table. Route completion is below Latent-WAM's.
- **No sub-metrics, no per-dataset split, no commit.** Whether the low Easy score is collisions, comfort or route completion cannot be told.
- Its scorer is trained on NAVSIM's simulator sub-scores and favours progress on NAVSIM v2 (EP 92.2). How that interacts with a reactive benchmark is not examined.

**What the table shows about Latent-WAM.** This page filed Latent-WAM under the 345-scenario release. MM-Future prints Latent-WAM's numbers in a table captioned 436 scenarios, and the arithmetic supports the caption:

| Method | Reported overall RC / HDS | 80/157/96/103-weighted mean of its four tiers |
|---|---|---|
| Latent-WAM | 45.9 / 28.9 | 45.88 / 28.91 |
| MM-Future | 44.5 / 32.3 | 44.52 / 32.32 |
| UniAD, LTF, VAD (benchmark paper) | 40.6 / 28.9, 41.4 / 24.8, 27.9 / 12.3 | 40.63 / 28.95, 41.36 / 24.82, 27.87 / 12.25 |
| HAD-L | 47.5 / 30.8 | 51.27 / 33.97 |

Latent-WAM's averages match the 436-scenario weights to the second decimal on both metrics. HAD-L's do not. **So Latent-WAM was scored on 436 scenarios and belongs with the unpinned 436 group; only HAD-L (and DrivoR's published numbers) remain on the 345 release.** What is still unknown for that group is the controller version (before or after PR #57).

MM-Future prints UniAD's overall HD-Score as 28.6; the weighted mean of the tiers it prints is 28.9.

## Earlier Release (345 scenarios, pre-PR #57)

| Method | Easy RC | Easy HDS | Medium RC | Medium HDS | Hard RC | Hard HDS | Extreme RC | Extreme HDS | Overall RC | Overall HDS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HAD-L | 65.9 | 51.2 | 52.1 | 34.9 | 50.8 | 30.4 | 39.1 | 22.5 | 47.5 | 30.8 |
| Latent-WAM | 84.2 | 72.5 | 42.5 | 24.0 | 30.6 | 12.2 | 35.5 | 18.1 | 45.9 | 28.9 |

HAD-L's result is useful because it evaluates the same planner family outside NAVSIM and includes an extreme split where the model drops to 39.1 RC / 22.5 HDS. The paper reports public-split results for most baselines; starred baselines use public+private scenarios, so comparison scope should be checked before treating the table as a clean leaderboard.

[[sources/latent-wam.md]] reports zero-shot HUGSIM using its NAVSIM-v2-trained model. It has stronger Easy RC/HDS than HAD-L but lower overall HDS, mainly because Medium and Hard are weaker. Treat the two rows as different generalization profiles rather than a clean ranking.

**Neither row is comparable to the section above.** Both would need rescoring under `ead17f2` to enter the current table, and neither paper is mentioned by WA-JEPA.

*(2026-09-30: for Latent-WAM this should read "not known to be comparable". Its overall scores equal the 80 / 157 / 96 / 103-weighted means of its tiers, so it was scored on 436 scenarios; only the controller version is unknown. See [above](#mm-future).)*

## Open Questions

*Wiki-wide open questions are collected in [[research-directions.md]].*

- **Where do HAD-L and Latent-WAM actually sit now?** Both are the wiki's other closed-loop-capable planners and neither has been rescored under the current snapshot. Until someone runs them, the 44.62 vs. 30.8 gap is uninterpretable.
- **[Answered: no] Is the comfort deficit inherent to sampled planners?** *([[sources/physwam.md]] samples video, depth and motion from noise and scores 96.3 closed-loop comfort and 90.5 navtest EC. What it does differently from WA-JEPA is unisolated; candidates are its SE(3) relative-pose action rows and a metric pose loss.)* WA-JEPA (0.662) and UniAD (0.663) are far below LTF, DrivoR, and VAD (~0.95) on closed-loop comfort, and the same ordering appears in NAVSIM-v2 EC. Drive-JEPA's momentum-aware selector fixed the open-loop version of this ([[sources/drive-jepa.md]], EC 47.9 → 84.8) by comparing each proposal against the previously selected trajectory. No closed-loop planner in the wiki has tried the analogous fix.
- **Does anything move the Extreme tier?** Every ingested method scores 0.06-0.14 there (BeyondDrive, not ingested, is reported at 0.16 in PhysWAM's table). This is the closest closed-loop analogue to navhard Stage 2, and like it, no ingested method has made progress.
- **Should HUGSIM become the wiki's primary closed-loop benchmark?** It has properties Bench2Drive lacks — real-log reconstructions rather than CARLA assets, natural zero-shot structure, and per-difficulty reporting. What it lacks is adoption: only five ingested papers report it (HAD, Latent-WAM, WA-JEPA, PhysWAM and MM-Future), against far more for Bench2Drive.
- **Which protocol did PhysWAM and BeyondDrive run?** Both report on 436 episodes without naming a commit. Rescoring either under `ead17f2`, or WA-JEPA's authors confirming the benchmark paper's baseline numbers, would put all three on one table.
- **Why do NAVSIM-only models collapse after the Easy tier?** PhysWAM (86.9 → 30.1) and Latent-WAM (72.5 → 24.0) both do; WA-JEPA, with nuPlan multi-view video pretraining, does not (79.8 → 55.6). Pretraining data, camera count (3 vs 4) and the JEPA objective are all confounded.
  *(2026-09-30: [[sources/mm-future.md]], also NAVSIM-only, has the opposite shape: a low Easy score (53.8) and the second-best Medium (40.0). So the collapse is not a property of NAVSIM-only training as such.)*

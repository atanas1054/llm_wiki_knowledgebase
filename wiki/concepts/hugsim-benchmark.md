---
title: HUGSIM Benchmark
type: concept
sources: [raw/papers/HAD_ Combining Hierarchical Diffusion with Metric-Decoupled RL for End-to-End Driving.md, raw/papers/Latent-WAM_ Latent World Action Modeling for End-to-End Autonomous Driving.md, raw/papers/WA-JEPA_ Rethinking the Video JEPA Paradigm forWorld-Action Modeling in Autonomous Driving.md]
related: [sources/had.md, sources/latent-wam.md, sources/wa-jepa.md, concepts/navsim-benchmark.md, concepts/bench2drive.md, concepts/world-model-for-ad.md]
created: 2026-05-01
updated: 2026-09-02
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
| Earlier release | 345 | No | HAD-L, Latent-WAM (and DrivoR's *published* numbers) |
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

## Earlier Release (345 scenarios, pre-PR #57)

| Method | Easy RC | Easy HDS | Medium RC | Medium HDS | Hard RC | Hard HDS | Extreme RC | Extreme HDS | Overall RC | Overall HDS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HAD-L | 65.9 | 51.2 | 52.1 | 34.9 | 50.8 | 30.4 | 39.1 | 22.5 | 47.5 | 30.8 |
| Latent-WAM | 84.2 | 72.5 | 42.5 | 24.0 | 30.6 | 12.2 | 35.5 | 18.1 | 45.9 | 28.9 |

HAD-L's result is useful because it evaluates the same planner family outside NAVSIM and includes an extreme split where the model drops to 39.1 RC / 22.5 HDS. The paper reports public-split results for most baselines; starred baselines use public+private scenarios, so comparison scope should be checked before treating the table as a clean leaderboard.

[[sources/latent-wam.md]] reports zero-shot HUGSIM using its NAVSIM-v2-trained model. It has stronger Easy RC/HDS than HAD-L but lower overall HDS, mainly because Medium and Hard are weaker. Treat the two rows as different generalization profiles rather than a clean ranking.

**Neither row is comparable to the section above.** Both would need rescoring under `ead17f2` to enter the current table, and neither paper is mentioned by WA-JEPA.

## Open Questions

- **Where do HAD-L and Latent-WAM actually sit now?** Both are the wiki's other closed-loop-capable planners and neither has been rescored under the current snapshot. Until someone runs them, the 44.62 vs. 30.8 gap is uninterpretable.
- **Is the comfort deficit inherent to sampled planners?** WA-JEPA (0.662) and UniAD (0.663) are far below LTF, DrivoR, and VAD (~0.95) on closed-loop comfort, and the same ordering appears in NAVSIM-v2 EC. Drive-JEPA's momentum-aware selector fixed the open-loop version of this ([[sources/drive-jepa.md]], EC 47.9 → 84.8) by comparing each proposal against the previously selected trajectory. No closed-loop planner in the wiki has tried the analogous fix.
- **Does anything move the Extreme tier?** Every method scores 0.06-0.14 there. This is the closest closed-loop analogue to navhard Stage 2, and like it, no ingested method has made progress.
- **Should HUGSIM become the wiki's primary closed-loop benchmark?** It has properties Bench2Drive lacks — real-log reconstructions rather than CARLA assets, natural zero-shot structure, and per-difficulty reporting. What it lacks is adoption: only three ingested papers report it, against far more for Bench2Drive.
